"""Manages Google OAuth token lifecycle.

Two operating modes:

1. **Local mode** (default). Tokens are stored in the ``app_state``
   SQLite table via :class:`~captain_claw.session.SessionManager`. This
   captain-claw instance performs the OAuth dance itself and holds the
   refresh token.

2. **Flight Deck client mode**. When
   ``config.google_oauth.flight_deck_url`` is set, this instance does
   **not** run its own OAuth flow. Instead, every call that needs an
   access token hits ``{flight_deck_url}/fd/google/access_token`` and
   ``/fd/google/credentials`` to retrieve a freshly-refreshed token
   managed by Flight Deck. This lets many captain-claw agents running
   on different ports / hosts share a single Google connection without
   juggling refresh tokens or redirect URIs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.google_oauth import (
    STATE_KEY_TOKENS,
    STATE_KEY_USER,
    GoogleOAuthTokens,
    fetch_user_info,
    refresh_access_token,
    revoke_token,
)
from captain_claw.logging import get_logger
from captain_claw.session import SessionManager

log = get_logger(__name__)


# ── Module-level connection cache ─────────────────────────────────────
#
# Sync-readable flag used by tool-registry filters (which run in a sync
# context and can't await). Updated by every GoogleOAuthManager call
# that inspects or mutates token state. Treat as a hint: callers that
# need a definitive answer should still `await mgr.is_connected()`.
_GOOGLE_CONNECTED: bool = False
_GOOGLE_CONNECTED_AT: float = 0.0
_GOOGLE_CACHE_MAX_AGE: float = 120.0  # seconds


def _mark_google_connected(connected: bool) -> None:
    global _GOOGLE_CONNECTED, _GOOGLE_CONNECTED_AT
    _GOOGLE_CONNECTED = bool(connected)
    _GOOGLE_CONNECTED_AT = time.time()


def is_google_connected_cached(max_age: float | None = None) -> bool:
    """Synchronous best-effort check for Google OAuth connection state.

    Returns *True* only when an async call has recently confirmed tokens
    are present. Stale caches are reported as *False* so callers err on
    the side of hiding Google-dependent features until freshly checked.
    """
    age_limit = _GOOGLE_CACHE_MAX_AGE if max_age is None else max_age
    if _GOOGLE_CONNECTED_AT <= 0.0:
        return False
    if (time.time() - _GOOGLE_CONNECTED_AT) > age_limit:
        return False
    return _GOOGLE_CONNECTED


class GoogleOAuthManager:
    """Manages Google OAuth token storage and refresh."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._sm = session_manager
        self._cached_tokens: GoogleOAuthTokens | None = None
        # Cache the Flight-Deck-provided credentials JSON briefly so
        # hot paths (per-request LLM calls) don't re-hit Flight Deck
        # on every invocation.
        self._fd_creds_cache: dict[str, Any] | None = None
        self._fd_creds_cached_at: float = 0.0

    # ── flight-deck client helpers ─────────────────────────

    @staticmethod
    def _flight_deck_base() -> str:
        """Return the Flight Deck base URL, or ``""`` when disabled.

        Resolution order:
        1. Explicit ``config.google_oauth.flight_deck_url``.
        2. ``FD_URL`` env var — injected automatically by Flight Deck
           when it spawns captain-claw agents, so Google tools "just
           work" the moment the user connects in the FD UI.
        """
        url = (get_config().google_oauth.flight_deck_url or "").strip().rstrip("/")
        if url:
            return url
        env_url = (os.environ.get("FD_URL", "") or "").strip().rstrip("/")
        return env_url

    @staticmethod
    def _flight_deck_headers() -> dict[str, str]:
        secret = (get_config().google_oauth.flight_deck_secret or "").strip()
        if not secret:
            secret = (os.environ.get("FD_AGENT_SHARED_SECRET", "") or "").strip()
        if secret:
            return {"X-Agent-Secret": secret}
        return {}

    def _is_flight_deck_client(self) -> bool:
        return bool(self._flight_deck_base())

    async def _fd_get_access_token(self) -> GoogleOAuthTokens | None:
        base = self._flight_deck_base()
        if not base:
            return None
        url = f"{base}/fd/google/access_token"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=self._flight_deck_headers())
                if resp.status_code == 404:
                    return None  # Not configured / not connected upstream.
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.warning("Flight Deck access_token fetch failed: %s", exc)
            return None
        return GoogleOAuthTokens(
            access_token=data.get("access_token", ""),
            refresh_token="",  # Flight Deck keeps the refresh token.
            token_type=data.get("token_type", "Bearer"),
            expires_at=float(data.get("expires_at", 0.0)) or (time.time() + 3300),
            scope=data.get("scope", ""),
        )

    async def _fd_get_credentials(self) -> dict[str, Any] | None:
        # Short 60s memo so burst-calls don't hammer the FD endpoint.
        if self._fd_creds_cache and (time.time() - self._fd_creds_cached_at) < 60:
            return self._fd_creds_cache

        base = self._flight_deck_base()
        if not base:
            return None
        url = f"{base}/fd/google/credentials"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=self._flight_deck_headers())
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.warning("Flight Deck credentials fetch failed: %s", exc)
            return None

        self._fd_creds_cache = data
        self._fd_creds_cached_at = time.time()
        return data

    # ── token access ────────────────────────────────────────

    async def get_tokens(self) -> GoogleOAuthTokens | None:
        """Load tokens, refreshing if expired.

        In Flight Deck client mode this always calls out to Flight Deck
        (which handles its own refresh). In local mode it reads from
        ``app_state`` and refreshes in-process.
        """
        if self._is_flight_deck_client():
            if self._cached_tokens and not self._cached_tokens.is_expired():
                _mark_google_connected(True)
                return self._cached_tokens
            tokens = await self._fd_get_access_token()
            if tokens:
                self._cached_tokens = tokens
                _mark_google_connected(True)
            else:
                _mark_google_connected(False)
            return tokens

        if self._cached_tokens and not self._cached_tokens.is_expired():
            _mark_google_connected(True)
            return self._cached_tokens

        raw = await self._sm.get_app_state(STATE_KEY_TOKENS)
        if not raw:
            _mark_google_connected(False)
            return None

        try:
            tokens = GoogleOAuthTokens.from_dict(json.loads(raw))
        except Exception as exc:
            log.warning("Failed to deserialize stored OAuth tokens: %s", exc)
            _mark_google_connected(False)
            return None

        if tokens.is_expired():
            tokens = await self._try_refresh(tokens)
            if tokens is None:
                _mark_google_connected(False)
                return None

        self._cached_tokens = tokens
        _mark_google_connected(bool(tokens and tokens.refresh_token))
        return tokens

    async def store_tokens(self, tokens: GoogleOAuthTokens) -> None:
        """Persist tokens to ``app_state`` (local mode only)."""
        if self._is_flight_deck_client():
            log.debug("store_tokens ignored — running in Flight Deck client mode.")
            return
        self._cached_tokens = tokens
        await self._sm.set_app_state(
            STATE_KEY_TOKENS,
            json.dumps(tokens.to_dict(), ensure_ascii=True),
        )
        _mark_google_connected(bool(tokens and tokens.refresh_token))

    async def store_user_info(self, user: dict[str, Any]) -> None:
        """Persist Google user profile to ``app_state`` (local mode only)."""
        if self._is_flight_deck_client():
            return
        await self._sm.set_app_state(
            STATE_KEY_USER,
            json.dumps(user, ensure_ascii=True),
        )

    # ── vertex credentials ─────────────────────────────────

    async def get_vertex_credentials(self) -> dict[str, Any] | None:
        """Return an ``authorized_user`` credentials dict for LiteLLM.

        Returns *None* when OAuth is not connected or tokens cannot be
        refreshed.
        """
        if self._is_flight_deck_client():
            data = await self._fd_get_credentials()
            if not data:
                return None
            return data.get("credentials")

        tokens = await self.get_tokens()
        if not tokens:
            return None

        cfg = get_config()
        oauth = cfg.google_oauth
        if not oauth.client_id or not oauth.client_secret:
            return None

        return tokens.to_vertex_credentials_json(
            client_id=oauth.client_id,
            client_secret=oauth.client_secret,
        )

    async def get_vertex_project_location(self) -> tuple[str | None, str | None]:
        """Return ``(project_id, location)`` — from Flight Deck when client,
        otherwise from the local config."""
        if self._is_flight_deck_client():
            data = await self._fd_get_credentials()
            if data:
                return data.get("project_id") or None, data.get("location") or None
            return None, None
        cfg = get_config().google_oauth
        return (cfg.project_id or None, cfg.location or None)

    # ── user info ──────────────────────────────────────────

    async def get_user_info(self) -> dict[str, Any] | None:
        """Return the cached Google user profile, or *None*.

        In Flight Deck client mode user info lives on the Flight Deck
        side and isn't needed for tool calls — returns *None*.
        """
        if self._is_flight_deck_client():
            return None
        raw = await self._sm.get_app_state(STATE_KEY_USER)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ── status ─────────────────────────────────────────────

    async def is_connected(self) -> bool:
        """Return *True* when a valid access token can be obtained."""
        if self._is_flight_deck_client():
            tokens = await self.get_tokens()
            connected = bool(tokens and tokens.access_token)
        else:
            tokens = await self.get_tokens()
            connected = tokens is not None and bool(tokens.refresh_token)
        _mark_google_connected(connected)
        return connected

    # ── disconnect ─────────────────────────────────────────

    async def disconnect(self) -> None:
        """Revoke tokens and clear all stored OAuth state.

        In Flight Deck client mode this is a no-op — disconnect must be
        done from the Flight Deck UI so every other agent sharing the
        connection is kept in sync.
        """
        if self._is_flight_deck_client():
            self._cached_tokens = None
            self._fd_creds_cache = None
            _mark_google_connected(False)
            log.info("Disconnect ignored — manage the connection via Flight Deck.")
            return

        tokens = await self.get_tokens()
        if tokens:
            if tokens.refresh_token:
                await revoke_token(tokens.refresh_token)
            elif tokens.access_token:
                await revoke_token(tokens.access_token)

        self._cached_tokens = None
        await self._sm.delete_app_state(STATE_KEY_TOKENS)
        await self._sm.delete_app_state(STATE_KEY_USER)
        _mark_google_connected(False)
        log.info("Google OAuth disconnected — tokens cleared.")

    # ── internal ───────────────────────────────────────────

    async def _try_refresh(
        self,
        tokens: GoogleOAuthTokens,
    ) -> GoogleOAuthTokens | None:
        """Attempt to refresh an expired access token (local mode only).

        On success the fresh tokens are persisted and returned.
        On failure *None* is returned; stored tokens are left in place
        in case the failure was transient.
        """
        if not tokens.refresh_token:
            log.warning("No refresh token available — cannot refresh.")
            return None

        cfg = get_config()
        oauth = cfg.google_oauth
        if not oauth.client_id or not oauth.client_secret:
            log.warning("Google OAuth client_id/secret missing — cannot refresh.")
            return None

        try:
            fresh = await refresh_access_token(
                refresh_token=tokens.refresh_token,
                client_id=oauth.client_id,
                client_secret=oauth.client_secret,
            )
            await self.store_tokens(fresh)

            try:
                user = await fetch_user_info(fresh.access_token)
                await self.store_user_info(user)
            except Exception:
                pass

            log.info("Google OAuth access token refreshed successfully.")
            return fresh

        except Exception as exc:
            log.warning("Google OAuth token refresh failed: %s", exc)
            return None
