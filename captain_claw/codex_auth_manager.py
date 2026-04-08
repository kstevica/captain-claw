"""Shared access to OpenAI "Sign in with ChatGPT" tokens (Codex CLI auth).

Captain Claw reuses the OAuth tokens that the ``codex`` CLI caches in
``~/.codex/auth.json``. Those tokens are intended for the *Responses*
endpoint at ``https://chatgpt.com/backend-api/codex/responses`` — the
same endpoint the Codex CLI hits — and include an ``access_token``
(JWT, ~24h lifetime) plus an ``account_id`` that must be sent as the
``chatgpt-account-id`` header.

This module is the *single source of truth* for resolving those
tokens inside a running captain-claw process. Two modes are supported:

* **Flight Deck mode** — when ``FD_URL`` is set in the environment the
  manager pulls tokens from Flight Deck's ``/fd/codex/access_token``
  endpoint. Flight Deck itself re-reads ``~/.codex/auth.json`` on
  demand, so every captain-claw sub-agent spawned by Flight Deck shares
  one centrally-managed connection — no matter which host the agent
  runs on, as long as it can reach FD.

* **Local mode** — otherwise, the manager reads ``~/.codex/auth.json``
  directly. This is how standalone captain-claw installs (no Flight
  Deck) pick up whatever Codex CLI has cached.

The ``ChatGPTResponsesProvider`` uses
:meth:`CodexAuthManager.get_headers` before every request. On a 401
response, it also calls :meth:`CodexAuthManager.invalidate_cache` and
retries once — letting us recover transparently when the Codex CLI
refreshes the token in the background.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

_CODEX_AUTH_PATH = Path("~/.codex/auth.json").expanduser()

# Consider the token stale this many seconds before its real exp so
# that a request that takes a few seconds to complete can't end up
# with a mid-flight expiry.
_EXP_SAFETY_MARGIN = 60.0


@dataclass
class CodexTokens:
    access_token: str
    account_id: str
    expires_at: float  # unix seconds; 0 if unknown
    email: str = ""
    plan: str = ""

    def is_stale(self, now: float | None = None) -> bool:
        if self.expires_at <= 0:
            return False  # unknown — treat as non-stale, rely on 401 retry
        t = now if now is not None else time.time()
        return t >= (self.expires_at - _EXP_SAFETY_MARGIN)

    def to_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.access_token}",
        }
        if self.account_id:
            headers["chatgpt-account-id"] = self.account_id
        return headers


def _decode_jwt_unverified(token: str) -> dict | None:
    """Decode a JWT's payload without verifying the signature.

    Captain Claw does not need to verify these tokens cryptographically
    — we only use ``exp`` + ``email`` + plan metadata for UX. OpenAI's
    backend is still the one that actually validates them when we call
    the Responses API.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # JWT base64url, no padding
        pad = "=" * (-len(payload_b64) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64 + pad)
        return json.loads(decoded.decode("utf-8"))
    except Exception:
        return None


def _parse_auth_json(data: dict) -> CodexTokens | None:
    """Pull the bits we care about out of the ``~/.codex/auth.json`` dict."""
    tokens = (data or {}).get("tokens") or {}
    access_token = str(tokens.get("access_token") or "").strip()
    account_id = str(tokens.get("account_id") or "").strip()
    if not access_token:
        return None

    expires_at = 0.0
    email = ""
    plan = ""

    # Try to pull ``exp`` / ``email`` / ``plan`` out of the JWT payload
    # for a richer status view. Failures are non-fatal.
    claims = _decode_jwt_unverified(access_token) or {}
    try:
        exp_val = claims.get("exp")
        if isinstance(exp_val, (int, float)):
            expires_at = float(exp_val)
    except Exception:
        pass

    # The id_token usually carries email + plan details in a nested
    # ``https://api.openai.com/auth`` claim. Fall back to the
    # access_token payload when present.
    id_token = str(tokens.get("id_token") or "").strip()
    id_claims = _decode_jwt_unverified(id_token) if id_token else None
    source_claims = id_claims or claims
    try:
        email = str(source_claims.get("email") or "")
        auth_info = source_claims.get("https://api.openai.com/auth") or {}
        if isinstance(auth_info, dict):
            plan = str(auth_info.get("chatgpt_plan_type") or "")
    except Exception:
        pass

    return CodexTokens(
        access_token=access_token,
        account_id=account_id,
        expires_at=expires_at,
        email=email,
        plan=plan,
    )


def load_tokens_from_disk() -> CodexTokens | None:
    """Read and parse ``~/.codex/auth.json`` synchronously. ``None`` when
    the file is missing or malformed."""
    if not _CODEX_AUTH_PATH.exists():
        return None
    try:
        raw = _CODEX_AUTH_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        log.warning("Failed to read %s: %s", _CODEX_AUTH_PATH, exc)
        return None
    return _parse_auth_json(data)


def _flight_deck_base() -> str:
    """Return the Flight Deck base URL, or ``""`` when not running under FD."""
    return (os.environ.get("FD_URL") or "").rstrip("/")


def _flight_deck_headers() -> dict[str, str]:
    secret = (os.environ.get("FD_AGENT_SHARED_SECRET") or "").strip()
    if secret:
        return {"X-Agent-Secret": secret}
    return {}


class CodexAuthManager:
    """Resolve / cache / refresh Codex OAuth headers for one provider.

    A single instance is created per :class:`ChatGPTResponsesProvider`
    (which keeps a long-lived ``httpx.AsyncClient``). The manager
    maintains an in-memory cache so most requests are O(1); on
    staleness or 401, it re-fetches from FD (or disk) and updates the
    cache.
    """

    def __init__(self) -> None:
        self._cached: CodexTokens | None = None
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    async def _fetch_from_fd(self) -> CodexTokens | None:
        base = _flight_deck_base()
        if not base:
            return None
        url = f"{base}/fd/codex/access_token"
        try:
            client = await self._http()
            resp = await client.get(url, headers=_flight_deck_headers())
            if resp.status_code != 200:
                log.debug("FD /fd/codex/access_token returned %s", resp.status_code)
                return None
            data = resp.json()
            access_token = str(data.get("access_token") or "").strip()
            if not access_token:
                return None
            return CodexTokens(
                access_token=access_token,
                account_id=str(data.get("account_id") or "").strip(),
                expires_at=float(data.get("expires_at") or 0.0),
                email=str(data.get("email") or ""),
                plan=str(data.get("plan") or ""),
            )
        except Exception as exc:
            log.debug("Failed to fetch Codex tokens from FD: %s", exc)
            return None

    async def _load_fresh(self) -> CodexTokens | None:
        """Return a freshly-loaded tokens object, preferring FD when set."""
        if _flight_deck_base():
            fd_tokens = await self._fetch_from_fd()
            if fd_tokens is not None:
                return fd_tokens
            # Fall through to disk if FD is unreachable / not configured
            # on the FD host. This keeps local dev smooth.
        return load_tokens_from_disk()

    async def get_tokens(self, *, force_refresh: bool = False) -> CodexTokens | None:
        """Return a currently-valid :class:`CodexTokens`, refreshing if stale."""
        if not force_refresh and self._cached and not self._cached.is_stale():
            return self._cached
        fresh = await self._load_fresh()
        if fresh is not None:
            self._cached = fresh
        return self._cached

    async def get_headers(self, *, force_refresh: bool = False) -> dict[str, str]:
        tokens = await self.get_tokens(force_refresh=force_refresh)
        if tokens is None:
            return {}
        return tokens.to_headers()

    def invalidate_cache(self) -> None:
        self._cached = None
