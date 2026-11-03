"""An agent's way to its owner's Line.

Flight Deck keeps the Line (``flight_deck/line.py``): one timeline per user of
everything the deck and the agents said on their own — scheduled tasks' and
Autonomous Work's results, finished runs, proposals, payments to look at,
fleet changes. The agent only reads it, through ``/fd/line/agent/*`` — owner
only, so a shared agent's member never reaches it.

:func:`news` is the one-line count at the start of the owner's own turns
(:func:`news_line`); the ``line`` tool reads the items. There is no
standalone mode: without Flight Deck there is no Line.
"""

from __future__ import annotations

from typing import Any

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

MEMBER_REFUSED = (
    "The Line isn't available during a shared-agent member's turn — it belongs to the "
    "agent's owner."
)
_OLD_DECK = "this Flight Deck has no Line for agents yet — it needs an update and a restart"

# What a kind is called in the context line (singular, plural).
KIND_WORDS: dict[str, tuple[str, str]] = {
    "proposal": ("proposal", "proposals"),
    "nudge": ("nudge", "nudges"),
    "result": ("result", "results"),
    "run": ("finished run", "finished runs"),
    "ask": ("question", "questions"),
    "payment": ("payment", "payments"),
    "mail": ("mail notice", "mail notices"),
    "agent": ("agent notice", "agent notices"),
    "fleet": ("fleet change", "fleet changes"),
    "share": ("share", "shares"),
    "deck": ("deck notice", "deck notices"),
}


class LineError(Exception):
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

    # Owner-only: never the member grant — /fd/line/agent/* is not grant-aware.
    return GoogleOAuthManager._flight_deck_headers(as_speaker=False)


def usable() -> bool:
    """Whether the line tool can be offered: under Flight Deck, and not a
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
        raise LineError(0, MEMBER_REFUSED)
    if not _fd_base():
        raise LineError(0, "The Line lives in Flight Deck, and this agent isn't running under one")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{_fd_base()}/fd/line/agent/{path}", json=body, headers=_fd_headers())
    except Exception as exc:
        raise LineError(0, f"Could not reach Flight Deck for the Line: {type(exc).__name__}") from None
    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            raise LineError(200, "Flight Deck returned a non-JSON answer") from None
        return data if isinstance(data, dict) else {}
    detail = _detail(resp)
    if resp.status_code in (404, 405) and detail in ("", "Not Found", "Method Not Allowed"):
        detail = _OLD_DECK
    raise LineError(resp.status_code, detail or f"Flight Deck refused (HTTP {resp.status_code})")


async def unread(limit: int = 20) -> dict[str, Any]:
    """The items this agent hasn't read yet, oldest first (moves its cursor)."""
    return await _post("unread", {"limit": limit})


async def items(*, kind: str = "", needs_you: bool | None = None, query: str = "", since: str = "",
                limit: int = 20) -> dict[str, Any]:
    body: dict[str, Any] = {"kind": kind, "query": query, "since": since, "limit": limit}
    if needs_you is not None:
        body["needs_you"] = needs_you
    return await _post("items", body)


async def item(item_id: str) -> dict[str, Any]:
    return await _post("item", {"id": item_id})


# ── what's new, once per owner turn ─────────────────────────────────


async def news() -> dict[str, Any] | None:
    """What came on the owner's Line since this agent was last told — once
    per owner turn (the telling moves the agent's cursor in Flight Deck).
    None when it can't be asked."""
    if not _fd_base() or _member_call():
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{_fd_base()}/fd/line/agent/news", json={}, headers=_fd_headers())
    except Exception as exc:
        log.debug("Flight Deck Line news failed: %s", exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def news_line(data: dict[str, Any] | None, *, detail: str = "normal") -> str:
    """One context line from Flight Deck's news ("" when nothing is new):
    counts by kind only, never an item's text."""
    if not isinstance(data, dict) or not data.get("connected"):
        return ""
    new = int(data.get("new") or 0)
    if new <= 0 or detail == "nano":
        return ""
    kinds = data.get("kinds") if isinstance(data.get("kinds"), dict) else {}
    parts = []
    for kind, n in sorted(kinds.items(), key=lambda kv: -int(kv[1] or 0)):
        n = int(n or 0)
        if n > 0:
            one, many = KIND_WORDS.get(str(kind), ("notice", "notices"))
            parts.append(f"{n} {one if n == 1 else many}")
    waiting = int(data.get("needs_you") or 0)
    when = "in the last day" if data.get("first_look") else "since your last turn"
    if detail == "micro":
        return (f"Line ({when}): {new} new" + (f", {waiting} need the user" if waiting else "")
                + " — the line tool reads them.")
    need = f" {waiting} {'needs' if waiting == 1 else 'need'} the user." if waiting else ""
    return (
        f"Your user's Line {when}: {', '.join(parts) or f'{new} new'}.{need} Read them with the "
        "line tool (action='news') if they bear on what the user is doing or need them; otherwise "
        "don't bring them up."
    )
