"""Manages Google OAuth token lifecycle with persistent storage.

Tokens are stored in the ``app_state`` SQLite table via
:class:`~captain_claw.session.SessionManager`.  The manager handles
loading, refreshing, and revoking tokens, and produces the
``authorized_user`` credentials dict consumed by LiteLLM's Vertex AI
provider.
"""

from __future__ import annotations

import json
from typing import Any

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


class GoogleOAuthManager:
    """Manages Google OAuth token storage and refresh."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._sm = session_manager
        self._cached_tokens: GoogleOAuthTokens | None = None

    # ── token access ────────────────────────────────────────

    async def get_tokens(self) -> GoogleOAuthTokens | None:
        """Load tokens from ``app_state``, refreshing if expired.

        Returns *None* when no tokens are stored or refresh fails.
        """
        if self._cached_tokens and not self._cached_tokens.is_expired():
            return self._cached_tokens

        raw = await self._sm.get_app_state(STATE_KEY_TOKENS)
        if not raw:
            return None

        try:
            tokens = GoogleOAuthTokens.from_dict(json.loads(raw))
        except Exception as exc:
            log.warning("Failed to deserialize stored OAuth tokens: %s", exc)
            return None

        if tokens.is_expired():
            tokens = await self._try_refresh(tokens)
            if tokens is None:
                return None

        self._cached_tokens = tokens
        return tokens

    async def store_tokens(self, tokens: GoogleOAuthTokens) -> None:
        """Persist tokens to ``app_state``."""
        self._cached_tokens = tokens
        await self._sm.set_app_state(
            STATE_KEY_TOKENS,
            json.dumps(tokens.to_dict(), ensure_ascii=True),
        )

    async def store_user_info(self, user: dict[str, Any]) -> None:
        """Persist Google user profile to ``app_state``."""
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

    # ── user info ──────────────────────────────────────────

    async def get_user_info(self) -> dict[str, Any] | None:
        """Return the cached Google user profile, or *None*."""
        raw = await self._sm.get_app_state(STATE_KEY_USER)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ── status ─────────────────────────────────────────────

    async def is_connected(self) -> bool:
        """Return *True* when valid (or refreshable) tokens exist."""
        tokens = await self.get_tokens()
        return tokens is not None and bool(tokens.refresh_token)

    # ── disconnect ─────────────────────────────────────────

    async def disconnect(self) -> None:
        """Revoke tokens and clear all stored OAuth state."""
        tokens = await self.get_tokens()
        if tokens:
            # Best-effort revocation — try refresh token first, then access.
            if tokens.refresh_token:
                await revoke_token(tokens.refresh_token)
            elif tokens.access_token:
                await revoke_token(tokens.access_token)

        self._cached_tokens = None
        await self._sm.delete_app_state(STATE_KEY_TOKENS)
        await self._sm.delete_app_state(STATE_KEY_USER)
        log.info("Google OAuth disconnected — tokens cleared.")

    # ── internal ───────────────────────────────────────────

    async def _try_refresh(
        self,
        tokens: GoogleOAuthTokens,
    ) -> GoogleOAuthTokens | None:
        """Attempt to refresh an expired access token.

        On success the fresh tokens are persisted and returned.
        On failure *None* is returned and stored tokens are cleared.
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

            # Also refresh user-info while we have a valid token.
            try:
                user = await fetch_user_info(fresh.access_token)
                await self.store_user_info(user)
            except Exception:
                pass  # Non-critical — cached user info is fine.

            log.info("Google OAuth access token refreshed successfully.")
            return fresh

        except Exception as exc:
            log.warning("Google OAuth token refresh failed: %s", exc)
            # Don't clear tokens — the refresh token might still work later
            # (e.g. transient network error).
            return None
