"""Agent side of Crew: whether the `crew` tool is offered, and its calls to Flight
Deck. Crew lives in Flight Deck only (the approval, the workers, their worktrees);
an agent without a Flight Deck never sees the tool. Owner-only: a shared agent's
member never reaches it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

MEMBER_REFUSED = "Crew isn't available during a shared-agent member's turn: only the agent's owner can use it."
NO_FLIGHT_DECK = "Crew needs Flight Deck: this agent wasn't started by one."
_TIMEOUT = 180.0  # a pull request pushes first; a clone happens in the background
# What a task can hold. Flight Deck enforces them (crew_attach); they're here so a file that can't
# go isn't sent all the way first.
MAX_FILE_BYTES = 50_000_000
MAX_TASK_BYTES = 200_000_000
MAX_TASK_FILES = 30

_USABLE: tuple[bool, float] = (False, 0.0)
_CACHE_MAX_AGE = 120.0
_REFRESH_MIN_GAP = 20.0
_LAST_CHECK_AT = 0.0


def _fd_base() -> str:
    from captain_claw.fd_client import flight_deck_base

    return flight_deck_base()


def _fd_headers() -> dict[str, str]:
    from captain_claw.fd_client import flight_deck_headers

    return flight_deck_headers()


def under_flight_deck() -> bool:
    return bool(_fd_base())


def member_call() -> bool:
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    return GoogleOAuthManager._member_call()


def _mark(usable: bool) -> None:
    global _USABLE
    _USABLE = (bool(usable), time.time())


def crew_usable() -> bool:
    """Offer the tool? (sync, for the tool registry; fails closed)"""
    usable, at = _USABLE
    return usable and at > 0 and time.time() - at <= _CACHE_MAX_AGE


async def refresh(force: bool = False) -> bool:
    """Ask Flight Deck whether the owner is approved (while so, at most every ~20 s)."""
    global _LAST_CHECK_AT
    if not under_flight_deck() or member_call():
        _mark(False)
        return False
    usable, _at = _USABLE
    if not force and usable and time.time() - _LAST_CHECK_AT < _REFRESH_MIN_GAP:
        _mark(True)
        return True
    _LAST_CHECK_AT = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_fd_base()}/fd/crew/agent/status", headers=_fd_headers())
    except Exception as exc:  # noqa: BLE001
        log.debug("Flight Deck Crew status check failed: %s", exc)
        return crew_usable()
    if resp.status_code != 200:
        _mark(False)
        return False
    usable = bool(resp.json().get("usable"))
    _mark(usable)
    return usable


def _answer(resp: httpx.Response) -> tuple[int, Any]:
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"detail": resp.text[:500]}


async def call(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    """``(status, json)`` from ``/fd/crew/agent/<path>``."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.request(method, f"{_fd_base()}/fd/crew/agent/{path}", headers=_fd_headers(), json=body)
    return _answer(resp)


async def upload(path: Path, name: str, media_type: str) -> tuple[int, Any]:
    """A file sent ahead of the task or message it goes with: ``(status, json)``, its id in the json."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        with path.open("rb") as file:
            resp = await client.post(f"{_fd_base()}/fd/crew/agent/uploads", headers=_fd_headers(),
                                     files={"file": (name, file, media_type)})
    return _answer(resp)
