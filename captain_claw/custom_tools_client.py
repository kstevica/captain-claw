"""Agent side of custom tools: whether the `custom_tools` tool is offered, the
owner's tools for the prompt, and the calls to Flight Deck.

Custom tools live in Flight Deck only (flight_deck/external_tools.py): it keeps
them, runs them and decides whose they are. An agent without a Flight Deck never
sees the tool. A shared agent's member can list, describe and run tools during
their own turn: every such call carries the turn's speaker grant, and Flight
Deck runs the tool as the member (their tools, secrets and files), never as the
owner.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

NO_FLIGHT_DECK = "Custom tools need Flight Deck: this agent wasn't started by one."
NO_GRANT = "Custom tools aren't available here: this turn has no member grant."
_TIMEOUT = 360.0  # a call can run 300 s, plus copying its files

_STATE: dict[str, Any] = {"usable": False, "at": 0.0, "tools": [], "can_build": False}
_CACHE_MAX_AGE = 300.0
_REFRESH_MIN_GAP = 20.0


def _fd_base() -> str:
    from captain_claw.fd_client import flight_deck_base

    return flight_deck_base()


def _fd_headers() -> dict[str, str]:
    from captain_claw.fd_client import flight_deck_headers

    return flight_deck_headers()


def under_flight_deck() -> bool:
    return bool(_fd_base())


def member_call() -> bool:
    from captain_claw import speaker

    return speaker.member_bound() or speaker.identity_lost()


def usable() -> bool:
    """Offer the tool? (sync, for the tool registry; fails closed)"""
    return bool(_STATE["usable"]) and time.time() - float(_STATE["at"]) <= _CACHE_MAX_AGE


def owner_tools() -> list[dict[str, str]]:
    """The owner's usable tools as last seen (ref, title, description), for the prompt."""
    return list(_STATE["tools"]) if usable() else []


def can_build() -> bool:
    return usable() and bool(_STATE["can_build"])


def _set(usable_now: bool, tools: list | None = None, build: bool = False) -> None:
    _STATE.update(usable=bool(usable_now), at=time.time(), tools=list(tools or []), can_build=bool(build))


def reset_for_tests() -> None:
    _STATE.update(usable=False, at=0.0, tools=[], can_build=False)


async def refresh(force: bool = False) -> bool:
    """Ask Flight Deck whether custom tools are on, and which the owner has. On the
    owner's turns only: a member's turn keeps what the owner's last one learned."""
    if not under_flight_deck():
        _set(False)
        return False
    if member_call():
        return usable()
    if not force and usable() and time.time() - float(_STATE["at"]) < _REFRESH_MIN_GAP:
        return True
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{_fd_base()}/fd/external-tools/agent/status", headers=_fd_headers())
    except Exception as exc:  # noqa: BLE001
        log.debug("Flight Deck custom tools status check failed: %s", exc)
        return usable()
    if resp.status_code != 200:
        _set(False)
        return False
    data = resp.json() if resp.content else {}
    if not data.get("enabled"):
        _set(False)
        return False
    tools = [{"ref": str(t.get("ref") or ""), "title": str(t.get("title") or ""),
              "description": str(t.get("description") or "")}
             for t in data.get("tools") or [] if isinstance(t, dict) and t.get("ref")]
    _set(True, tools, bool(data.get("can_build")))
    return True


def _answer(resp: httpx.Response) -> tuple[int, Any]:
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, {"detail": resp.text[:500]}


async def call(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    """``(status, json)`` from ``/fd/external-tools/agent/<path>``. A member's turn
    sends its grant, so Flight Deck acts as the member; without one it refuses
    here, never as the owner."""
    from captain_claw import speaker

    headers = _fd_headers()
    try:
        grant = speaker.grant_headers()
        params = speaker.grant_params()
    except speaker.SpeakerGrantMissing:
        return 403, {"detail": NO_GRANT}
    headers.update(grant)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.request(method, f"{_fd_base()}/fd/external-tools/agent/{path}",
                                    headers=headers, params=params, json=body)
    return _answer(resp)
