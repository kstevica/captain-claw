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
"""

from __future__ import annotations

import logging
import time
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
