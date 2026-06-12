"""Agent-side async result delivery.

When the agent finishes work that nobody is actively waiting for — a cron job,
a deferred "check X in 15 min", a scheduled task — the result has to find its
way back to wherever the request came from. This routes it.

Strategy:
  1. Under Flight Deck (``FD_URL`` set), hand the result + the session's durable
     origin to FD's ``POST /fd/deliver``. FD owns the channel↔address bindings
     and the delivery credentials (WhatsApp Cloud API, Telegram, channel bus),
     so it is the right place to actually send. This is the durable path: the
     origin is read from ``session.metadata`` (persisted), not an in-memory map.
  2. Standalone (no FD), the caller's local fallback handles it (e.g. the agent's
     own Telegram bridge) — see web_server's cron output callback.

Returns True only when the result was actually handed off for delivery.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from captain_claw.logging import get_logger
from captain_claw.origin import get_session_origin

log = get_logger(__name__)


def _fd_url() -> str:
    return (os.environ.get("FD_URL", "") or os.environ.get("FD_INTERNAL_URL", "")).strip().rstrip("/")


async def deliver_to_origin(agent: Any, session_id: str, text: str) -> bool:
    """Route an async result for ``session_id`` back to its origin via FD.

    Returns True if FD accepted it for delivery, False otherwise (so the caller
    can fall back to a local channel)."""
    text = (text or "").strip()
    if not text:
        return False

    # Resolve the session (it may not be the agent's *current* session — a cron
    # job fires in a session that was switched into and out of).
    session = None
    try:
        cur = getattr(agent, "session", None)
        if cur is not None and str(getattr(cur, "id", "")) == str(session_id):
            session = cur
        elif getattr(agent, "session_manager", None) is not None:
            session = await agent.session_manager.load_session(str(session_id))
    except Exception:
        session = None
    if session is None:
        return False

    origin = get_session_origin(session)
    if not origin:
        return False  # nothing to route to — let the caller's fallback try

    fd_url = _fd_url()
    if not fd_url:
        return False  # standalone: caller handles local delivery

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{fd_url}/fd/deliver",
                json={"session_id": str(session_id), "origin": origin, "text": text},
            )
        if resp.status_code == 200 and resp.json().get("ok"):
            return True
        log.debug("deliver_to_origin: FD returned %s — %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        log.debug("deliver_to_origin: FD handoff failed: %s", exc)
    return False
