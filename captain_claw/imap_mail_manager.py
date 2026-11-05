"""Agent-side access to the owner's IMAP/SMTP inboxes (Flight Deck only).

The deck keeps every inbox's settings and password and does the IMAP / SMTP
work itself (``flight_deck.imap_routes``); this agent only asks
``{FD}/fd/imap/agent/<action>`` with the same credentials it uses for Google
(``X-Agent-Secret`` + its ``X-Agent-Auth`` token). The deck answers for the
agent's OWNER, and only with the inboxes the owner lets this agent use.

There is no standalone mode (an agent outside Flight Deck has no imap_mail —
the registry hides it), and a shared-agent member's turn gets nothing: members
can't use the owner's inboxes. That is refused here, before any request.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.logging import get_logger

log = get_logger(__name__)

WHERE = "Flight Deck → Connections → Email (IMAP)"
NOT_CONNECTED = (
    f"No email inbox is connected for this agent. The user connects one (any IMAP/SMTP mailbox) in {WHERE}."
)
NO_FLIGHT_DECK = (
    "imap_mail needs Flight Deck: this agent isn't running under a Flight Deck, which holds the "
    f"inboxes ({WHERE})."
)
MEMBER_REFUSED = (
    "Email inboxes aren't available during a shared-agent member's turn — only the agent's owner "
    "can use their inboxes through it."
)

# The owner's inboxes as last seen, readable synchronously by the tool registry
# (which can't await). Refreshed once per turn; stale reads as "not connected".
_STATE: tuple[bool, float, list[dict[str, Any]]] = (False, 0.0, [])
_CACHE_MAX_AGE = 120.0


def is_imap_connected_cached(max_age: float | None = None) -> bool:
    connected, at, _accounts = _STATE
    age = time.time() - at
    return connected and at > 0 and age <= (_CACHE_MAX_AGE if max_age is None else max_age)


def cached_accounts() -> list[dict[str, Any]]:
    return list(_STATE[2]) if is_imap_connected_cached() else []


def _mark(connected: bool, accounts: list[dict[str, Any]] | None = None) -> None:
    global _STATE
    _STATE = (bool(connected), time.time(), list(accounts or []))


class ImapRefused(RuntimeError):
    pass


class ImapMailClient:
    """Calls Flight Deck's /fd/imap agent routes."""

    @staticmethod
    def base() -> str:
        return GoogleOAuthManager._flight_deck_base()

    @staticmethod
    def member_call() -> bool:
        return GoogleOAuthManager._member_call()

    async def status(self) -> dict[str, Any] | None:
        """``{"connected", "accounts"}`` — None when Flight Deck couldn't say."""
        if self.member_call():
            return {"connected": False, "accounts": []}
        base = self.base()
        if not base:
            _mark(False)
            return {"connected": False, "accounts": []}
        try:
            headers = GoogleOAuthManager._flight_deck_headers(as_speaker=False)
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{base}/fd/imap/agent_status", headers=headers)
        except Exception as exc:
            log.info("Flight Deck IMAP agent_status failed: %s", exc)
            return None
        if resp.status_code in (401, 403, 404, 503):
            _mark(False)
            return {"connected": False, "accounts": []}
        try:
            data = resp.json()
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        accounts = [a for a in data.get("accounts") or [] if isinstance(a, dict)]
        _mark(bool(data.get("connected")) and bool(accounts), accounts)
        return {"connected": bool(accounts), "accounts": accounts}

    async def is_connected(self) -> bool:
        """Refresh the cached flag (once per agent turn)."""
        if self.member_call():
            return False
        st = await self.status()
        if st is None:
            return is_imap_connected_cached()
        return bool(st["connected"])

    async def call(self, action: str, payload: dict[str, Any], *, timeout: float = 120.0
                   ) -> tuple[int, dict[str, Any]]:
        """``(HTTP status, JSON body)``. Raises :class:`ImapRefused` for a
        member's turn or no Flight Deck; httpx errors go up as they are (the
        caller decides what an unanswered send means)."""
        if self.member_call():
            raise ImapRefused(MEMBER_REFUSED)
        base = self.base()
        if not base:
            raise ImapRefused(NO_FLIGHT_DECK)
        headers = GoogleOAuthManager._flight_deck_headers(as_speaker=False)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base}/fd/imap/agent/{action}", json=payload, headers=headers)
        try:
            data = resp.json()
        except Exception:
            data = {"detail": resp.text[:500]}
        if not isinstance(data, dict):
            data = {"detail": str(data)[:500]}
        if resp.status_code == 404 and "No email inbox" in str(data.get("detail") or ""):
            _mark(False)
        return resp.status_code, data
