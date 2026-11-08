"""An agent's way to its owner's Course (flight_deck/course_routes.py).

Flight Deck keeps the goals (the deck's, the owner's and this agent's own) and
writes the live ones into the agent's prompt (``course_context.md``). This
client is the ``course`` tool's way to more than the prompt holds — progress,
last moved, review dates — and to record evidence and progress, through
``/fd/course/agent/*``: owner only, so a shared agent's member never reaches it.
There is no standalone mode: without Flight Deck there is no Course.
"""

from __future__ import annotations

from typing import Any

import httpx

MEMBER_REFUSED = (
    "Course isn't available during a shared-agent member's turn — it belongs to the "
    "agent's owner."
)
_OLD_DECK = "this Flight Deck has no Course for agents yet — it needs an update and a restart"


class CourseError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _fd_base() -> str:
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    return GoogleOAuthManager._flight_deck_base()


def _member_call() -> bool:
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    return GoogleOAuthManager._member_call()


def _fd_headers() -> dict[str, str]:
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    # Owner-only: never the member grant — /fd/course/agent/* is not grant-aware.
    return GoogleOAuthManager._flight_deck_headers(as_speaker=False)


def usable() -> bool:
    """Whether the course tool can be offered: under Flight Deck, and not a
    member's turn (sync, no network)."""
    try:
        return bool(_fd_base()) and not _member_call()
    except Exception:
        return False


def _detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return ""
    detail = data.get("detail") if isinstance(data, dict) else ""
    return detail if isinstance(detail, str) else ""


async def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    if _member_call():
        raise CourseError(0, MEMBER_REFUSED)
    if not _fd_base():
        raise CourseError(0, "Course lives in Flight Deck, and this agent isn't running under one")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{_fd_base()}/fd/course/agent/{path}", json=body,
                                     headers=_fd_headers())
    except Exception as exc:
        raise CourseError(0, f"Could not reach Flight Deck for Course: {type(exc).__name__}") from None
    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            raise CourseError(200, "Flight Deck returned a non-JSON answer") from None
        return data if isinstance(data, dict) else {}
    detail = _detail(resp)
    if resp.status_code in (404, 405) and detail in ("", "Not Found", "Method Not Allowed"):
        detail = _OLD_DECK
    raise CourseError(resp.status_code, detail or f"Flight Deck refused (HTTP {resp.status_code})")


async def view() -> dict[str, Any]:
    return await _post("view", {})


async def link(goal_id: str, kind: str, ref_id: str, note: str = "") -> dict[str, Any]:
    return await _post("link", {"goal_id": goal_id, "kind": kind, "ref_id": ref_id, "note": note})


async def report(goal_id: str, progress: float, note: str = "",
                 looks_achieved: bool = False) -> dict[str, Any]:
    return await _post("report", {"goal_id": goal_id, "progress": progress, "note": note,
                                  "looks_achieved": looks_achieved})


async def propose(*, horizon: str, title: str, success: str = "", scope: str = "agent",
                  detail: str = "", parent_id: str = "", rationale: str = "") -> dict[str, Any]:
    """Suggest a goal — for this agent (``scope="agent"``) or its owner
    (``scope="user"``); it waits for the owner."""
    return await _post("propose", {"scope": scope, "horizon": horizon, "title": title,
                                   "success": success, "detail": detail, "parent_id": parent_id,
                                   "rationale": rationale})


async def propose_change(goal_id: str, changes: dict[str, Any], rationale: str = "") -> dict[str, Any]:
    """Suggest a change to one of the owner's goals or this agent's own."""
    return await _post("propose_change", {"goal_id": goal_id, "changes": changes,
                                          "rationale": rationale})
