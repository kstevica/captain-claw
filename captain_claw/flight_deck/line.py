"""The Line: one timeline per user for everything the deck says on its own.

Autonomous Work's nudges and proposals, finished runs, fleet changes, payments
to look at, shares, mail the deck sent — each is one item on the user's Line
(the ``line_items`` table), whatever wrote it. The Line page shows them all.

The bell is the Line's **Needs you** list: the items that wait for the user
(something to approve, answer, review or fix) or that failed. ``db.add_notification``
is the bell's old way in and stays it: it writes a Line item, its ``type`` (the
bell's own word: ``payment``, ``run_error``, ``autonomy_proposal``…) mapped to
a ``kind`` and whether it needs the user by ``classify``.

An item has two states. ``read_at``: seen, in the bell or on the Line. And,
when it needs the user, ``resolved_at``: answered. A proposal is answered on
the autonomy ledger (approved, rejected, expired) — ``resolve_answered`` marks
its item once the ledger says it no longer waits; any other item that needs
the user counts as answered once it is read (until its own answer is wired
in). The bell lists what still waits on top, however old.

Old items go after ``RETENTION_DAYS``; one that still waits never does.

What the deck has an agent do on its own reaches the Line from the deck,
which hears the result: a scheduled task's run (fd_scheduler), an Autonomous
Work action or a Taskboard hand-over ending on the ledger (``action_done``),
a plan's step (plans). The agent's own chat keeps its copy as before. A run
that answers somewhere else (WhatsApp, Telegram…) leaves a copy here too.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

_log = logging.getLogger(__name__)

RETENTION_DAYS = 30
# How often a write may also clear out old items (seconds).
_PRUNE_EVERY_S = 3600.0
_last_prune = 0.0

TITLE_MAX = 200
BODY_MAX = 4000

# What the Line holds, and what the page calls each.
KIND_LABELS: dict[str, str] = {
    "proposal": "Proposals",
    "nudge": "Nudges",
    "result": "Results",
    "run": "Runs",
    "ask": "Questions",
    "payment": "Payments",
    "mail": "Mail",
    "agent": "Agents",
    "fleet": "Fleet",
    "share": "Sharing",
    "deck": "Deck",
}

# A bell type → (kind, needs the user). A type not listed needs the user (it
# was in the bell before the Line; keep showing it there).
_TYPES: dict[str, tuple[str, bool]] = {
    "autonomy_proposal": ("proposal", True),
    "autonomy_nudge": ("nudge", False),
    "run": ("run", False),
    "run_error": ("run", True),
    "bat_done": ("run", False),
    "bat_error": ("run", True),
    "bat_ask": ("ask", True),
    # A Crew task waits for a decision, an approval, or is stuck or failed.
    "crew_task": ("run", True),
    "fleet": ("fleet", False),
    "deck_update": ("deck", True),
    # What an agent did on its own, heard by the deck: done, or failed.
    "scheduled_task": ("result", False),
    "scheduled_task_error": ("result", True),
    "autonomy_done": ("result", False),
    "autonomy_failed": ("result", True),
    "plan_step": ("result", False),
    "plan_step_error": ("result", True),
    "telegram": ("ask", True),
    "payment": ("payment", True),
    "share": ("share", False),
    "invite_accepted": ("share", False),
    "email_sent": ("mail", False),
    "email_autoreply": ("mail", False),
    "email_autoreply_held": ("mail", True),
    "email_autoreply_issue": ("mail", True),
    "email_autoreply_blocked": ("mail", True),
    "agent_setup": ("agent", True),
    "agent_setup_error": ("agent", True),
    "agent_forge": ("agent", True),
    "agent_forge_error": ("agent", True),
    "profile_research": ("agent", True),
    "profile_research_error": ("agent", True),
    "whatsapp_web": ("deck", True),
    "info": ("deck", False),
    "success": ("deck", False),
    "warning": ("deck", True),
    "error": ("deck", True),
}
# What a bell item is about can say more than its type: these wait for the
# user whatever the type (a member's suggested lesson is an "info").
_NEEDS_YOU_REFS = frozenset({"agent_lessons", "purchase", "telegram_link"})
# Generic types that are about an agent.
_AGENT_REFS = frozenset({"agent_lessons", "agent_exam", "agent_setup", "agent_forge", "profile"})
# Items answered somewhere else, not by being read.
ANSWERED_REFS = frozenset({"autonomy_action"})
# Whose words an item's text is: the deck's own, an agent's, or someone
# else's (an email, a WhatsApp chat) — what a reader may trust of it.
_TRUST: dict[str, str] = {
    "autonomy_proposal": "agent",
    "autonomy_nudge": "agent",
    "email_autoreply": "untrusted",
    "email_autoreply_held": "untrusted",
    "email_autoreply_issue": "untrusted",
    "email_autoreply_blocked": "untrusted",
    "whatsapp_web": "untrusted",
}


def classify(bell_type: str, ref_type: str = "") -> tuple[str, bool, str]:
    """A bell item's (kind, needs the user, trust)."""
    kind, needs_you = _TYPES.get(bell_type, ("deck", True))
    if ref_type in _AGENT_REFS and kind == "deck":
        kind = "agent"
    return kind, needs_you or ref_type in _NEEDS_YOU_REFS, _TRUST.get(bell_type, "deck")


def _norm(user_id: str | None) -> str:
    from captain_claw.flight_deck.autonomy import _norm_user

    return _norm_user(user_id)


def retention_cutoff(now: datetime | None = None) -> str:
    return ((now or datetime.now(timezone.utc)) - timedelta(days=RETENTION_DAYS)).isoformat()


async def post(
    user_id: str, kind: str, title: str, body: str = "", *, bell_type: str = "",
    source: str = "", needs_you: bool = False, ref_type: str = "", ref_id: str = "",
    trust: str = "deck",
) -> str:
    """One item on ``user_id``'s Line; its id, "" when it couldn't be
    written. Never raises. (The bell's writers use ``db.add_notification``,
    which lands here too.)"""
    title = str(title or "").strip()
    if not title:
        return ""
    try:
        from captain_claw.flight_deck.auth import get_db

        db = get_db()
        item_id = await db.add_line_item(
            _norm(user_id), kind if kind in KIND_LABELS else "deck", bell_type or kind,
            title[:TITLE_MAX], str(body or "").strip()[:BODY_MAX], source=source,
            needs_you=needs_you, ref_type=ref_type, ref_id=ref_id, trust=trust,
        )
        await maybe_prune(db)
        return item_id
    except Exception as exc:  # noqa: BLE001
        _log.debug("line item for %s not written: %s", user_id, exc)
        return ""


# post_once's items being written right now: a second call for one of them
# (two "done"s close together) must not pass the check before the first writes.
_writing: set[tuple[str, str, str]] = set()


async def post_once(user_id: str, kind: str, title: str, body: str = "", *,
                    ref_type: str, ref_id: str, **kw: Any) -> str:
    """``post``, unless this user's Line already has an item about
    ``ref_type`` / ``ref_id``. Never raises."""
    key = (_norm(user_id), ref_type, ref_id)
    if key in _writing:
        return ""
    _writing.add(key)
    try:
        from captain_claw.flight_deck.auth import get_db

        if ref_id and await get_db().count_notifications_since(
                key[0], "", ref_type=ref_type, ref_id=ref_id):
            return ""
        return await post(user_id, kind, title, body, ref_type=ref_type, ref_id=ref_id, **kw)
    except Exception as exc:  # noqa: BLE001
        _log.debug("line: could not check %s/%s: %s", ref_type, ref_id, exc)
        return ""
    finally:
        _writing.discard(key)


def soon(fn: Callable[..., Awaitable[Any]], *args: Any) -> None:
    """Run ``fn(*args)`` on the deck's loop, from sync code that may be on a
    worker thread (the Taskboard writes the ledger off the loop). No loop (a
    script, a test without one): nothing."""
    try:
        asyncio.get_running_loop().create_task(fn(*args))
        return
    except RuntimeError:
        pass
    try:
        from captain_claw.flight_deck.taskboard_passive import _main_loop

        loop = _main_loop()
        if loop is not None:
            asyncio.run_coroutine_threadsafe(fn(*args), loop)
    except Exception as exc:  # noqa: BLE001
        _log.debug("line: nothing to run %s on: %s", getattr(fn, "__name__", fn), exc)


def agent_name(slug: str) -> str:
    """An agent's name on this deck, else its slug."""
    if not slug:
        return ""
    try:
        from captain_claw.flight_deck.server import _load_process_registry

        return str((_load_process_registry().get(slug) or {}).get("name") or slug)
    except Exception:  # noqa: BLE001
        return slug


async def action_done(row: dict[str, Any]) -> str:
    """An Autonomous Work action, or a Taskboard hand-over, ended on the
    ledger: its outcome on the Line, once. A nudge is already there (it went
    to the bell as one). Never raises."""
    kind = str(row.get("kind") or "")
    if kind == "nudge":
        return ""
    failed = str(row.get("outcome") or "") == "fail"
    title = " ".join(str(row.get("title") or kind or "An action").split())
    return await post_once(
        str(row.get("user_id") or ""), "result", f"{'Failed' if failed else 'Done'}: {title}",
        str(row.get("outcome_note") or ""), bell_type="autonomy_failed" if failed else "autonomy_done",
        source=agent_name(str(row.get("target") or "")), needs_you=failed,
        ref_type="autonomy_result", ref_id=str(row.get("id") or ""), trust="agent",
    )


async def maybe_prune(db: Any) -> int:
    """Clear out items past the retention, at most once an hour. Never raises."""
    global _last_prune
    now = time.monotonic()
    if _last_prune and now - _last_prune < _PRUNE_EVERY_S:
        return 0
    _last_prune = now
    try:
        return await db.prune_line_items(retention_cutoff())
    except Exception as exc:  # noqa: BLE001
        _log.debug("line prune failed: %s", exc)
        return 0


async def resolve_answered(user_id: str) -> int:
    """Mark answered the items of proposals that no longer wait on the
    ledger. How many; never raises."""
    try:
        from captain_claw.flight_deck.auth import get_db
        from captain_claw.flight_deck.autonomy import get_store

        db = get_db()
        uid = _norm(user_id)
        open_refs = await db.unresolved_line_refs(uid, "autonomy_action")
        if not open_refs:
            return 0
        waiting = {str(a.get("id") or "") for a in
                   get_store().list_actions(uid, status="awaiting_approval", limit=500)}
        done = [ref for ref in open_refs if ref not in waiting]
        return await db.resolve_line_items(uid, "autonomy_action", done) if done else 0
    except Exception as exc:  # noqa: BLE001
        _log.debug("line: answered proposals of %s not resolved: %s", user_id, exc)
        return 0


def view(row: dict[str, Any]) -> dict[str, Any]:
    """A Line item as the page reads it."""
    return {
        "id": row["id"],
        "kind": row.get("kind") or "deck",
        "type": row.get("type") or "",
        "title": row.get("title") or "",
        "body": row.get("body") or "",
        "source": row.get("source") or "",
        "needs_you": bool(row.get("needs_you")),
        "resolved": bool(row.get("resolved_at")),
        "read": bool(row.get("read_at")),
        "ref_type": row.get("ref_type") or "",
        "ref_id": row.get("ref_id") or "",
        "trust": row.get("trust") or "deck",
        "created_at": row.get("created_at") or "",
    }
