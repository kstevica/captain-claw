"""The whole email behind a task, for the Taskboard's bigger view.

The board keeps the first 500 characters of a message, with its line breaks gone:
enough to search and triage by, not to read. This reads the task's thread from
Gmail with the user's own token when they open the task bigger, and keeps none of
it: nothing goes on the board and no model reads it, so what is kept and for how
long stays as it was (30 days, gone when the inbox is switched off, when the key
that names the thread goes with it).

  GET /fd/taskboard/tasks/{id}/email
    → {messages: [{id, from_name, from_address, to, cc, date, subject, text,
                   attachments}], more, reason}
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from captain_claw import gmail_inbound
from captain_claw.flight_deck import taskboard_store
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/taskboard", tags=["taskboard"])

MAX_THREADS = 3    # a task holding more (merged) reads its latest ones
MAX_MESSAGES = 20  # the latest of them

_NOT_CONNECTED = "Connect Google to read the whole email here."
_NOT_READ = "Gmail didn't give this email: it may have been deleted, or be in another mailbox."


def _message(msg: dict) -> dict[str, Any]:
    """One Gmail message as the view shows it: without what it quotes (each of the
    thread's messages is there itself), unless that is all it says."""
    inb = gmail_inbound.parse_message(msg)
    return {
        "id": inb.message_id, "from_name": inb.from_name, "from_address": inb.from_addrs[0] if inb.from_addrs else "",
        "to": list(inb.to_addrs), "cc": list(inb.cc_addrs), "date": inb.internal_ms / 1000.0, "subject": inb.subject,
        "text": gmail_inbound.strip_quoted(inb.text) or inb.text, "attachments": list(inb.attachment_names),
    }


async def whole_email(uid: str, task_id: str) -> dict[str, Any]:
    """The emails of the task's Gmail threads, oldest first, read now. *reason* says
    why there are none when the task has a thread; a task with no thread has
    nothing to say. LookupError when the task isn't this user's."""
    import httpx

    from captain_claw.flight_deck import gmail_autoreply as ar
    from captain_claw.flight_deck.google_oauth_routes import get_valid_google_access_token

    board = taskboard_store.get_store()
    if await asyncio.to_thread(board.get_task, uid, task_id) is None:
        raise LookupError(task_id)
    links = await asyncio.to_thread(board.links, uid, [task_id])
    prefix = "gmail:thread:"
    threads = list(dict.fromkeys(
        k for k in (str(link["key"]).removeprefix(prefix) for link in links.get(task_id, [])
                    if str(link["key"]).startswith(prefix)) if ar._ID_RE.match(k)))
    if not threads:
        return {"messages": [], "more": 0, "reason": ""}
    try:
        token = await get_valid_google_access_token(uid)
    except Exception as exc:  # noqa: BLE001 — the task's own text is still there to read
        log.warning("Taskboard email: no token", user=uid, error=str(exc)[:200])
        token = None
    if not token:
        return {"messages": [], "more": 0, "reason": _NOT_CONNECTED}
    found: list[dict] = []
    async with httpx.AsyncClient(timeout=ar.GMAIL_TIMEOUT_S) as client:
        for thread_id in threads[-MAX_THREADS:]:
            thread = await ar._thread(client, token, thread_id)
            found += [m for m in (thread or {}).get("messages") or []
                      if isinstance(m, dict) and "DRAFT" not in (m.get("labelIds") or [])]
    if not found:
        return {"messages": [], "more": 0, "reason": _NOT_READ}
    messages = sorted((_message(m) for m in found), key=lambda m: m["date"])
    return {"messages": messages[-MAX_MESSAGES:], "more": max(0, len(messages) - MAX_MESSAGES), "reason": ""}


@router.get("/tasks/{task_id}/email")
async def taskboard_email(task_id: str, user: dict = Depends(get_current_user)) -> dict:
    """The whole email behind one of the caller's tasks, read from their Gmail now."""
    from captain_claw.flight_deck.taskboard import _uid

    try:
        return await whole_email(_uid(user), task_id)
    except LookupError:
        raise HTTPException(404, "task not found") from None
