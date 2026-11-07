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

Channels (``fan_out``): besides Flight Deck, each item can reach the user's
WhatsApp and Telegram — each switched on by its old switch
(``nudge_to_whatsapp`` / ``notify_telegram``), at a level: "everything", or
"needs_you" (what waits for them or failed, and nudges) — and one agent's
chat, as a card it keeps as context (never a turn). Fleet changes never leave
the Line page. owner_notify rings the phones for nudges and proposals itself;
their items only go on to the agent. A copy of something that already answered
on a channel (a run, a scheduled task's own delivery) goes to no phone. In the
user's quiet hours phones get nothing but nudges. The deck turns fan-out on at
startup (``enable_fan_out``): a script or a test never sends.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
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
    "talk": "Asked",
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
    # A custom tool or an agent asks the person, in a Flight Deck dialog (person_asks).
    "person_ask": ("ask", True),
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
ANSWERED_REFS = frozenset({"autonomy_action", "person_ask"})
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
    trust: str = "deck", to_phones: bool = True, thread_id: str = "",
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
            needs_you=needs_you, ref_type=ref_type, ref_id=ref_id, trust=trust, to_phones=to_phones,
            thread_id=thread_id,
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


def user_agents(user_id: str) -> list[dict[str, str]]:
    """The agents a user may pick for their Line or hand an item to: their own
    process agents (every one, with sign-in off), by name."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    from captain_claw.flight_deck.server import _load_process_registry

    me = _norm(user_id)
    out = []
    for slug, entry in _load_process_registry().items():
        if _fd_auth_enabled() and str(entry.get("owner") or "") != me:
            continue
        out.append({"slug": slug, "name": str(entry.get("name") or slug)})
    return sorted(out, key=lambda a: a["name"].lower())


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
    # With the Taskboard task it's about, it joins that task's thread.
    try:
        from captain_claw.flight_deck.owner_notify import task_of

        task = await asyncio.to_thread(task_of, dict(row))
    except Exception:  # noqa: BLE001
        task = ""
    title = " ".join(str(row.get("title") or kind or "An action").split())
    return await post_once(
        str(row.get("user_id") or ""), "result", f"{'Failed' if failed else 'Done'}: {title}",
        str(row.get("outcome_note") or ""), bell_type="autonomy_failed" if failed else "autonomy_done",
        source=agent_name(str(row.get("target") or "")), needs_you=failed,
        ref_type="autonomy_result", ref_id=str(row.get("id") or ""), trust="agent",
        thread_id=f"task:{task}" if task else "",
    )


# ── Channels ─────────────────────────────────────────────────────────

LEVELS = ("everything", "needs_you", "off")
_fan_out_on = False
# Set while owner_notify writes an item whose phones it rings itself.
phones_done: contextvars.ContextVar[bool] = contextvars.ContextVar("line_phones_done", default=False)
_PHONE_BODY_MAX = 1500


def enable_fan_out(on: bool = True) -> None:
    """The deck's startup: items go out to the user's channels from now on."""
    global _fan_out_on
    _fan_out_on = on


def levels(cfg: dict[str, Any]) -> dict[str, str]:
    """The user's channels from their autonomy config: WhatsApp / Telegram
    level ("off" while their switch is off), the agent and its level."""
    def level(switch: bool, value: Any) -> str:
        return "off" if not switch else (str(value) if value in ("everything", "needs_you") else "needs_you")

    agent = str(cfg.get("line_agent") or "").strip()
    agent_level = str(cfg.get("line_agent_level") or "everything")
    return {
        "whatsapp": level(bool(cfg.get("nudge_to_whatsapp", True)), cfg.get("line_whatsapp")),
        "telegram": level(bool(cfg.get("notify_telegram", False)), cfg.get("line_telegram")),
        "agent": agent,
        "agent_level": agent_level if agent and agent_level in ("everything", "needs_you") else "off",
    }


def covers(level: str, kind: str, needs_you: bool) -> bool:
    """Whether a channel at ``level`` gets an item of ``kind``."""
    if kind == "fleet" or level == "off":
        return False
    return level == "everything" or needs_you or kind == "nudge"


def phones_take_results(user_id: str) -> bool:
    """Whether the user's phones get everything (results too) — then an agent's
    mirror of a result isn't relayed by its chat bridges as well."""
    try:
        from captain_claw.flight_deck.autonomy import resolve_config

        lv = levels(resolve_config(user_id))
        return "everything" in (lv["whatsapp"], lv["telegram"])
    except Exception:  # noqa: BLE001
        return False


def phone_text(item: dict[str, Any]) -> str:
    body = str(item.get("body") or "").strip()
    if len(body) > _PHONE_BODY_MAX:
        body = body[:_PHONE_BODY_MAX] + "…"
    return f"{item.get('title') or ''}\n{body}".strip()


def after_insert(user_id: str, item: dict[str, Any], *, to_phones: bool = True) -> None:
    """A new item (db.add_line_item): on to the user's channels, once fan-out
    is on. Reads :data:`phones_done` here, in the writer's context."""
    if _fan_out_on:
        rung = phones_done.get()
        soon(fan_out, user_id, dict(item), to_phones and not rung, rung)


async def fan_out(user_id: str, item: dict[str, Any], to_phones: bool = True,
                  rung: bool = False) -> dict[str, Any]:
    """Send one item to the channels that take it. What each did, None where
    it didn't go; never raises. ``rung``: owner_notify rang its phones."""
    out: dict[str, Any] = {"whatsapp": None, "telegram": None, "agent": None}
    try:
        from captain_claw.flight_deck.autonomy import get_store, resolve_config
        from captain_claw.flight_deck.owner_notify import _quiet

        cfg = resolve_config(user_id)
        lv = levels(cfg)
        kind, needs = str(item.get("kind") or ""), bool(item.get("needs_you"))
        hushed = kind != "nudge" and _quiet(cfg)
        store = get_store()
        if to_phones and not hushed:
            text = phone_text(item)
            if covers(lv["whatsapp"], kind, needs):
                from captain_claw.flight_deck.fd_dispatch import _nudge_waids
                from captain_claw.flight_deck.whatsapp_bridge import push_to_waid

                waids, _skipped = _nudge_waids(cfg)
                sent = 0
                for waid in waids:
                    if await push_to_waid(waid, text):
                        sent += 1
                out["whatsapp"] = [sent, len(waids)]
                if waids:
                    store.log(user_id, "line → whatsapp", f"{item.get('title')} — {sent}/{len(waids)}")
            if covers(lv["telegram"], kind, needs):
                from captain_claw.flight_deck.fd_scheduler import _deliver_telegram_owner

                res = await _deliver_telegram_owner(user_id, text)
                out["telegram"] = bool(res.get("ok"))
                store.log(user_id, "line → telegram" + ("" if res.get("ok") else " skipped"),
                          str(item.get("title") or ""))
        slug = lv["agent"]
        if slug and covers(lv["agent_level"], kind, needs) and item.get("source") != agent_name(slug):
            # A phone has it already: the agent's chat bridges don't relay it too.
            phones = (rung or to_phones and not hushed) and (covers(lv["whatsapp"], kind, needs)
                                                              or covers(lv["telegram"], kind, needs))
            out["agent"] = await push_to_agent(slug, item, fd_delivers=phones)
    except Exception as exc:  # noqa: BLE001
        _log.debug("line fan-out for %s failed: %s", user_id, exc)
    return out


async def push_to_agent(slug: str, item: dict[str, Any], *, fd_delivers: bool = False) -> bool:
    """One item into agent ``slug``'s chat (a ``line_item`` frame; the agent
    keeps it as a card and as context, it runs no turn). An agent that isn't
    running — asleep, stopped — isn't woken: False."""
    import websockets

    from captain_claw.flight_deck.fd_scheduler import resolve_agent_by_slug

    host, port, auth = resolve_agent_by_slug(slug)
    if not port:
        return False
    url = f"ws://{host}:{port}/ws" + (f"?token={auth}" if auth else "")
    frame = {"type": "line_item", "title": str(item.get("title") or ""),
             "body": str(item.get("body") or ""), "kind": str(item.get("kind") or ""),
             **({"fd_delivers": True} if fd_delivers else {})}
    try:
        async with websockets.connect(url, open_timeout=5, close_timeout=3) as ws:
            welcome = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            if welcome.get("type") != "welcome":
                return False
            while True:  # the session's replay comes first
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "replay_done":
                    break
            await ws.send(json.dumps(frame))
        return True
    except Exception as exc:  # noqa: BLE001
        _log.debug("line item to agent %s failed: %s", slug, exc)
        return False


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
        # What it belongs with: an item, its hand-overs and their answers, or
        # everything about one Taskboard task.
        "thread_id": row.get("thread_id") or row["id"],
    }
