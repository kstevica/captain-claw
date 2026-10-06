"""Manages Google OAuth token lifecycle.

Two operating modes:

1. **Local mode** (default). Tokens are stored in the ``app_state``
   SQLite table via :class:`~captain_claw.session.SessionManager`. This
   captain-claw instance performs the OAuth dance itself and holds the
   refresh token.

2. **Flight Deck client mode**. When
   ``config.google_oauth.flight_deck_url`` (or the ``FD_URL`` Flight Deck
   injects at spawn) is set, this instance does **not** run its own OAuth
   flow. Instead, every call that needs an access token hits
   ``{flight_deck_url}/fd/google/access_token`` and
   ``/fd/google/credentials`` to retrieve a freshly-refreshed token
   managed by Flight Deck — the connection of this agent's OWNER, whom
   Flight Deck resolves from the ``X-Agent-Auth`` token it minted at spawn.
   Agents the deck didn't spawn are refused (403), and the refusal's reason
   is raised as :class:`FlightDeckRefused` so the Google tools report it
   instead of telling a connected user to "connect Google".
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.drive_client import DriveNotConnected
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
#
# Per principal (A2): "" is the owner's entry, "spk:<speaker_id>" a shared-
# agent member's (their OWN Google for this agent, written only by
# speaker_status()). An unverified member or a thread that lost the speaker
# context has no key: never read, never written (→ not connected).
_GOOGLE_CONNECTED: dict[str, tuple[bool, float]] = {}
_GOOGLE_CACHE_MAX_AGE: float = 120.0  # seconds


def _cache_key() -> str | None:
    """The cache key for whoever this code runs for (see above)."""
    from captain_claw import speaker as _speaker

    p = _speaker.current()
    if p is None:
        return None if _speaker.identity_lost() else ""
    if not p.speaker_id:
        return None
    return f"spk:{p.speaker_id}"


def _mark_google_connected(connected: bool) -> None:
    key = _cache_key()
    if key is None:
        return
    _GOOGLE_CONNECTED[key] = (bool(connected), time.time())


def _cache_entry(max_age: float) -> tuple[bool, float] | None:
    key = _cache_key()
    if key is None:
        return None
    entry = _GOOGLE_CONNECTED.get(key)
    if entry is None:
        return None
    _connected, at = entry
    if at <= 0.0 or (time.time() - at) > max_age:
        return None
    return entry


def is_google_connected_cached(max_age: float | None = None) -> bool:
    """Synchronous best-effort check for Google OAuth connection state.

    Returns *True* only when an async call has recently confirmed tokens
    are present. Stale caches are reported as *False* so callers err on
    the side of hiding Google-dependent features until freshly checked.
    Per principal: a member never reads the owner's flag.
    """
    entry = _cache_entry(_GOOGLE_CACHE_MAX_AGE if max_age is None else max_age)
    return bool(entry and entry[0])


def google_cache_fresh(max_age: float) -> bool:
    """Whether this principal's cached status is younger than *max_age*."""
    return _cache_entry(max_age) is not None


class FlightDeckRefused(DriveNotConnected):
    """Flight Deck answered this agent's Google token request with 401/403.

    Carries FD's own reason (``detail``) — e.g. an agent FD didn't spawn, or
    one it can't attribute to a user — which is NOT "Google isn't connected",
    so the tools must not send the user off to reconnect. A ``RuntimeError``
    (the google_* tools report ``str(exc)`` of those) and a
    :class:`~captain_claw.drive_client.DriveNotConnected` (the Drive/VFS paths
    handle it like any other unusable connection).
    """

    def __init__(self, status: int, detail: str, *, auth_disabled: bool = False) -> None:
        self.status = status
        self.detail = detail
        # FD said "no Google via FD here: auth is disabled" — a single-tenant
        # deck. Only the retired gws tool acted on it (it kept its own
        # credentials there); the google_* tools treat it as not connected.
        self.auth_disabled = auth_disabled
        super().__init__(
            f"Flight Deck refused this agent's Google request (HTTP {status}): "
            f"{detail or 'no reason given'}"
        )


def _fd_refusal(resp: httpx.Response) -> FlightDeckRefused | None:
    """A :class:`FlightDeckRefused` for a 401/403 from Flight Deck, else None."""
    if resp.status_code not in (401, 403):
        return None
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = str(body.get("detail") or "")
    except Exception:
        pass
    auth_disabled = resp.headers.get("X-FD-Google-Unavailable", "").strip().lower() == "auth-disabled"
    return FlightDeckRefused(resp.status_code, detail[:500], auth_disabled=auth_disabled)


class GoogleOAuthManager:
    """Manages Google OAuth token storage and refresh."""

    def __init__(self, session_manager: SessionManager) -> None:
        self._sm = session_manager
        self._cached_tokens: GoogleOAuthTokens | None = None
        # Whose tokens `_cached_tokens` are (FD mode): reused only for the
        # same principal (see _cache_key).
        self._cached_tokens_key: str | None = None
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
        if not url:
            url = (os.environ.get("FD_URL", "") or "").strip().rstrip("/")
        # Defensive: strip a stray trailing ``/fd`` (callers add ``/fd/...``
        # themselves; a base ending in ``/fd`` produces a double prefix).
        if url.endswith("/fd"):
            url = url[:-3].rstrip("/")
        return url

    @staticmethod
    def _member_call() -> bool:
        """A shared-agent member's call (or one from a thread that lost the
        speaker context while member work is live) — never the owner's."""
        from captain_claw import speaker as _speaker

        return _speaker.member_bound() or _speaker.identity_lost()

    @staticmethod
    def _flight_deck_headers(*, as_speaker: bool = True) -> dict[str, str]:
        """This agent's credentials for FD's Google endpoints — sent only to
        :meth:`_flight_deck_base`, which comes from config / the env FD pins at
        spawn, never from a session or a websocket message.

        ``X-Agent-Secret``: ``google_oauth.flight_deck_secret``, else the deck
        secret — ``FD_AGENT_SHARED_SECRET``, then the per-deck ``agent_secret``
        file (as ``tools.flight_deck._fd_agent_headers`` sends). Under
        FD_LOCKDOWN FD refuses (401) an agent call without it, even from
        loopback.

        *as_speaker* (default): for a shared-agent member's call, add the
        turn's ``X-FD-Speaker-Grant`` — raises
        :class:`~captain_claw.speaker.SpeakerGrantMissing` when the member
        has none (never sent as the owner). Every request with these headers
        also sends ``params=speaker.grant_params()``.
        """
        headers: dict[str, str] = {}
        secret = (get_config().google_oauth.flight_deck_secret or "").strip()
        if not secret:
            secret = (os.environ.get("FD_AGENT_SHARED_SECRET", "") or "").strip()
        if not secret:
            try:
                from captain_claw.flight_deck.agent_secret import get_or_create_agent_secret

                secret = get_or_create_agent_secret()
            except Exception:  # noqa: BLE001 — the header is an upgrade, never a blocker
                secret = ""
        if secret:
            headers["X-Agent-Secret"] = secret
        # The per-agent web_auth token lets Flight Deck resolve WHICH user's
        # Google connection to return — the shared secret can't. Without one
        # FD refuses (403): it never falls back to some other user's account.
        # (httpx sends no Origin / Sec-Fetch-* headers, which FD's agent
        # endpoints refuse as browser traffic.)
        token = str(getattr(getattr(get_config(), "web", None), "auth_token", "") or "").strip()
        if token:
            headers["X-Agent-Auth"] = token
        if as_speaker:
            from captain_claw import speaker as _speaker

            headers.update(_speaker.grant_headers())
        return headers

    def _is_flight_deck_client(self) -> bool:
        return bool(self._flight_deck_base())

    async def _fd_get_access_token(self) -> GoogleOAuthTokens | None:
        """The access token from Flight Deck — the owner's, or (with the
        turn's grant) a shared-agent member's own; None when FD is
        unreachable or has none (404 — not configured / not connected).
        Raises :class:`FlightDeckRefused` on a 401/403, keeping FD's reason,
        and — with NO request — for a member without a usable grant."""
        from captain_claw import speaker as _speaker

        base = self._flight_deck_base()
        if not base:
            return None
        url = f"{base}/fd/google/access_token"
        try:
            headers = self._flight_deck_headers()
            params = _speaker.grant_params()
        except _speaker.SpeakerGrantMissing:
            raise FlightDeckRefused(403, _speaker.NO_GRANT_MESSAGE) from None
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers, params=params)
        except Exception as exc:
            log.warning("Flight Deck access_token fetch failed: %s", exc)
            return None
        if resp.status_code == 404:
            return None  # Not configured / not connected upstream.
        refusal = _fd_refusal(resp)
        if refusal:
            log.warning("Flight Deck refused access_token: %s", refusal)
            raise refusal
        try:
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
            # Vertex LLM credentials are the owner's (the owner pays for
            # member turns too) — never the grant header or marker.
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=self._flight_deck_headers(as_speaker=False))
                if resp.status_code == 404:
                    return None
                refusal = _fd_refusal(resp)
                if refusal:  # the LLM path wants None; keep FD's reason in the log
                    log.warning("Flight Deck refused credentials: %s", refusal)
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
        (which handles its own refresh) and raises :class:`FlightDeckRefused`
        when Flight Deck refuses this agent — the google_* tools surface that
        reason. In local mode it reads from ``app_state`` and refreshes
        in-process.

        A shared-agent member's call never reads this agent's own tokens
        (they are the owner's): outside Flight Deck it is refused. Under
        Flight Deck it carries the turn's grant (see _fd_get_access_token);
        the member's status cache is written only by :meth:`speaker_status`.
        """
        member = self._member_call()
        if member and not self._is_flight_deck_client():
            from captain_claw.speaker import NO_GRANT_MESSAGE

            raise FlightDeckRefused(403, NO_GRANT_MESSAGE)

        if self._is_flight_deck_client():
            key = _cache_key()
            if (self._cached_tokens and not self._cached_tokens.is_expired()
                    and key is not None and self._cached_tokens_key == key):
                if not member:
                    _mark_google_connected(True)
                return self._cached_tokens
            try:
                tokens = await self._fd_get_access_token()
            except FlightDeckRefused:
                self._cached_tokens = None
                self._cached_tokens_key = None
                if not member:
                    _mark_google_connected(False)
                raise
            if tokens and key is not None:
                self._cached_tokens = tokens
                self._cached_tokens_key = key
            else:
                self._cached_tokens = None
                self._cached_tokens_key = None
            if not member:
                _mark_google_connected(bool(tokens))
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

    async def speaker_status(self) -> bool:
        """A shared-agent member's Google status for THIS agent, from Flight
        Deck's grant-aware ``/fd/google/agent_status`` (never a token fetch).

        Not an FD client, no usable grant, a non-JSON-object 200, a 403, a
        404 from an older Flight Deck or a network error → not connected and
        not enabled. Marks the member's cache entry, records the status for
        :func:`captain_claw.speaker.member_google_enabled` and returns
        ``connected``.
        """
        from captain_claw import speaker as _speaker

        connected = enabled = False
        base = self._flight_deck_base()
        if base:
            try:
                headers = self._flight_deck_headers(as_speaker=True)
                params = _speaker.grant_params()
            except _speaker.SpeakerGrantMissing:
                headers = None
                params = {}
            if headers is not None:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        resp = await client.get(
                            f"{base}/fd/google/agent_status", headers=headers, params=params,
                        )
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except Exception:
                            data = None
                        if isinstance(data, dict):
                            connected = bool(data.get("connected"))
                            enabled = bool(data.get("enabled"))
                except Exception as exc:
                    log.warning("Flight Deck agent_status fetch failed: %s", exc)
        _mark_google_connected(connected)
        _speaker.note_member_google(connected, enabled)
        return connected

    async def is_connected(self) -> bool:
        """Return *True* when a valid access token can be obtained."""
        if self._member_call():
            # A member's status comes from Flight Deck (their own Google, only
            # with their opt-in for this agent) — never a token fetch.
            return await self.speaker_status()
        if self._is_flight_deck_client():
            try:
                tokens = await self.get_tokens()
            except FlightDeckRefused:
                tokens = None
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
