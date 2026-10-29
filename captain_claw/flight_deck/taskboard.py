"""Taskboard — every task the caller's agents received or sent, on one board.

  GET  /fd/taskboard/status              the sources (tracked or not), each view's count, the sweep
  GET  /fd/taskboard/tasks               one view, searched, filtered, faceted and paged
  GET  /fd/taskboard/tasks/{id}          one task: its timeline, links, parties, decisions, proposals
  POST /fd/taskboard/tasks/{id}/action   a correction: done · reopen · dismiss · not_task · snooze ·
                                         unsnooze · rename · owes_next · ack_failed · undo
  POST /fd/taskboard/sync                read the sources now, for the caller's board
  GET  /fd/taskboard/summary             {needs_me} — the tile badge's poll
  GET/PUT /fd/taskboard/settings

A task belongs to exactly one board (``user_id``): the caller sees only their
own, and someone else's task is a 404 — there is no owner or admin view of
anyone else's board. The kiosk lock is the UI's; this API is the boundary.

What feeds the board is in ``taskboard_sources`` (a sweep every 120 s while
Flight Deck runs); the store, lanes and search in ``taskboard_store``. A
proposal waiting on a task is approved or rejected through the existing
``/fd/autonomy/actions/{id}/approve|reject``, a held Gmail reply through
``/fd/google/gmail-autoreply/activity/{id}/send|dismiss``; ``POST /sync``
after either moves the row to its new lane at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from captain_claw.flight_deck import taskboard_sources, taskboard_store
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.taskboard_store import (
    CHANNEL_LABELS,
    DIRECTION_LABELS,
    EVIDENCE_LABELS,
    FACETS,
    KIND_LABELS,
    RESULT_LABELS,
    STATUS_LABELS,
    VIEWS,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/taskboard", tags=["taskboard"])

ACTIONS = ("done", "reopen", "dismiss", "not_task", "snooze", "unsnooze", "rename", "owes_next",
           "ack_failed", "undo")
PASSIVE_SOURCES = ("telegram_inbox", "whatsapp_web", "gmail")

# What the Sources sheet lists: what v0 reads, and what is still to come.
SOURCES: list[dict[str, Any]] = [
    {"key": "fd_scheduler", "label": "Scheduled tasks", "tracked": True, "on": True,
     "note": "Every scheduled task, each of its runs and how its result was delivered."},
    {"key": "autonomy", "label": "Autonomous Work proposals", "tracked": True, "on": True,
     "note": "Proposals waiting for your approval, and what became of them."},
    {"key": "follow_ups", "label": "Follow-ups", "tracked": True, "on": True,
     "note": "Open loops Autonomous Work keeps an eye on, and when each is due."},
    {"key": "gmail_autoreplies", "label": "Email auto-replies", "tracked": True, "on": True,
     "note": "Emails a topic auto-reply held for you or answered."},
    {"key": "gmail_sends", "label": "Email your agents sent", "tracked": True, "on": True,
     "note": "Every email sent through Captain's Gmail connection."},
    {"key": "agent_links", "label": "Linked systems", "tracked": True, "on": True,
     "note": "Messages from systems linked to your agents, and what your agents sent them."},
    {"key": "agent_chats", "label": "Agent chats on every channel", "tracked": False, "on": False,
     "note": "Coming: web chat, WhatsApp, Telegram, devices and the rest, with what the agents did."},
    {"key": "telegram_inbox", "label": "Your Telegram inbox", "tracked": False, "on": False,
     "note": "Coming, and only if you switch it on."},
    {"key": "whatsapp_web", "label": "Your WhatsApp chats", "tracked": False, "on": False,
     "note": "Coming, and only if you switch it on."},
    {"key": "gmail_inbox", "label": "Your Gmail inbox", "tracked": False, "on": False,
     "note": "Coming, and only if you switch it on."},
    {"key": "history", "label": "The last 30 days", "tracked": False, "on": False,
     "note": "Coming: your agents' earlier chats, filled in once."},
    {"key": "not_covered", "label": "Not tracked", "tracked": False, "on": False,
     "note": "Outlook inbound mail, Slack, Discord and Twitter, Docker agents' history, calendar invites, "
             "payment approvals and delegations between agents."},
]

_HIDDEN = ("user_id", "agent_owner_id", "search_text", "fingerprint", "extracted_fp", "decided_fp")
_FACET_LABELS = {"status": STATUS_LABELS, "direction": DIRECTION_LABELS, "kind": KIND_LABELS,
                 "evidence": EVIDENCE_LABELS, "result": RESULT_LABELS}
_SWEEP_STOP_S = 5.0
_SYNC_GAP_S = 2.0

_loop_task: asyncio.Task | None = None
_loop_stop: asyncio.Event | None = None
# Each user's own sweep: the one reading now, and the next one waiting.
_RUNNING_SYNCS: dict[str, asyncio.Future] = {}
_QUEUED_SYNCS: dict[str, asyncio.Future] = {}


def configure(*, data_dir: Path) -> None:
    taskboard_store.configure(data_dir=data_dir)


def store() -> taskboard_store.TaskboardStore:
    return taskboard_store.get_store()


def start() -> None:
    """Start the sweep (Flight Deck's lifespan). ``FD_TASKBOARD_DISABLED=true``
    keeps it off; the board then fills only when someone opens it."""
    global _loop_task, _loop_stop
    if os.environ.get("FD_TASKBOARD_DISABLED", "").lower() in ("true", "1", "yes") or _loop_task is not None:
        return
    _loop_stop = asyncio.Event()
    _loop_task = asyncio.create_task(taskboard_sources.sweep_loop(_loop_stop))


async def stop() -> None:
    global _loop_task, _loop_stop
    if _loop_stop is not None:
        _loop_stop.set()
    if _loop_task is not None:
        try:
            await asyncio.wait_for(_loop_task, timeout=_SWEEP_STOP_S)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass
    _loop_task = _loop_stop = None


def forget_agent(agent_owner_id: str | None, slug: str) -> int:
    """An agent removed for good: its tasks go from every board."""
    return store().delete_agent(agent_owner_id, slug)


def forget_user(user_id: str) -> int:
    """A user deleted: their board goes."""
    return store().delete_user(user_id)


# ── Rows → what the UI shows ──────────────────────────────────────────


def _uid(user: dict) -> str:
    return str(user.get("id") or "") or "local"


def _actor(user: dict) -> str:
    return str(user.get("display_name") or user.get("email") or "You")


def _csv(raw: str) -> list[str]:
    return [v.strip() for v in (raw or "").split(",") if v.strip()][:50]


def _loads(raw: Any, default: Any) -> Any:
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw else default
    except ValueError:
        return default
    return value if isinstance(value, type(default)) else default


def _waiting_proposals(uid: str) -> dict[str, dict]:
    """``{task_id: {id, status, title}}`` — the newest Autonomous Work action
    awaiting approval on each of the user's tasks (read live, so the row moves
    lanes the moment it is approved)."""
    try:
        from captain_claw.flight_deck.autonomy import get_store as autonomy_store
        actions = autonomy_store().list_actions(uid, status="awaiting_approval", limit=500)
    except Exception as exc:
        log.debug("Taskboard: proposals unavailable", error=str(exc))
        return {}
    by_key = {f"autonomy:{a['id']}": a for a in actions}
    out: dict[str, dict] = {}
    held = store().tasks_for_keys(uid, list(by_key))
    for a in actions:  # newest first
        task_id = held.get(f"autonomy:{a['id']}")
        if task_id:
            out.setdefault(task_id, {"id": a["id"], "status": a["status"], "title": a.get("title") or ""})
    return out


def _link_ids(links: list[dict], proposal: dict | None) -> dict[str, str]:
    """The ids the UI links back with (Open in Scheduled, Continue in chat…),
    from the task's keys — the newest of each kind."""
    out = {"scheduled_job_id": "", "session_id": "", "session_agent": "", "autonomy_id": "",
           "followup_id": "", "gmail_thread_id": "", "autoreply_id": "", "link_id": ""}
    for link in links:
        key = str(link.get("key") or "")
        kind, _, rest = key.partition(":")
        if key.startswith("sched:job:"):
            out["scheduled_job_id"] = out["scheduled_job_id"] or key.removeprefix("sched:job:")
        elif kind == "session":
            slug, _, sid = rest.partition(":")
            out["session_agent"], out["session_id"] = slug, sid
        elif kind == "autonomy":
            out["autonomy_id"] = rest
        elif kind == "fu":
            out["followup_id"] = out["followup_id"] or rest
        elif key.startswith("gmail:thread:"):
            out["gmail_thread_id"] = out["gmail_thread_id"] or key.removeprefix("gmail:thread:")
        elif kind == "autoreply":
            out["autoreply_id"] = rest
        elif kind == "link":
            out["link_id"] = out["link_id"] or rest.partition(":")[0]
    if proposal:
        out["autonomy_id"] = proposal["id"]
    return out


def _item(row: dict, parties: list[dict], links: list[dict], proposal: dict | None, lane: str) -> dict:
    item = {k: v for k, v in row.items() if k not in _HIDDEN}
    item["channels"] = _loads(row.get("channels"), [])
    item["user_locks"] = _loads(row.get("user_locks"), {})
    item["flags"] = _loads(row.get("flags"), {})
    item["next_steps"] = _loads(row.get("next_steps"), [])
    item["private"] = bool(row.get("private"))
    item["parties"] = parties[:5]
    item["lane"] = lane
    item["proposal"] = proposal
    item["links"] = _link_ids(links, proposal)
    return item


def _items(uid: str, rows: list[dict], proposals: dict[str, dict]) -> list[dict]:
    ids = [r["id"] for r in rows]
    parties = store().parties_for(uid, ids)
    links = store().links(uid, ids)
    return [_item(r, parties.get(r["id"], []), links.get(r["id"], []), proposals.get(r["id"]),
                  str(r.get("lane") or "other")) for r in rows]


def _task_item(uid: str, row: dict, proposals: dict[str, dict] | None = None) -> dict:
    proposals = _waiting_proposals(uid) if proposals is None else proposals
    lane = store().lane_of(uid, row["id"], proposal_tasks=list(proposals))
    return _items(uid, [{**row, "lane": lane}], proposals)[0]


def _facet_label(dim: str, value: str, label: str) -> str:
    if dim == "agent":
        return label or value or "No agent"
    if dim == "channel":
        return CHANNEL_LABELS.get(value.partition(":")[0], value) if value else "No channel"
    if dim == "party":
        return label or value
    return _FACET_LABELS.get(dim, {}).get(value, value)


def _facets(raw: dict[str, list[tuple[str, str, int]]]) -> dict[str, list[dict]]:
    return {dim: [{"id": v or "", "label": _facet_label(dim, v or "", label or ""), "count": n}
                  for v, label, n in raw.get(dim, [])] for dim in FACETS}


def _counts(uid: str, proposals: dict[str, dict] | None = None) -> dict[str, int]:
    proposals = _waiting_proposals(uid) if proposals is None else proposals
    return store().counts(uid, proposal_tasks=list(proposals))


def _owned(uid: str, task_id: str) -> dict:
    row = store().get_task(uid, task_id)
    if row is None:
        raise HTTPException(404, "task not found")
    return row


# ── Routes ────────────────────────────────────────────────────────────


@router.get("/status")
async def taskboard_status(user: dict = Depends(get_current_user)) -> dict:
    """What the board tracks (and what not yet), each view's count and the sweep."""
    uid = _uid(user)
    counts = await asyncio.to_thread(_counts, uid)
    return {"sources": SOURCES, "counts": counts, "sweep": taskboard_sources.sweep_state(uid)}


@router.get("/tasks")
async def taskboard_tasks(
    view: str = "needs_me", q: str = "", status: str = "", channel: str = "", agent: str = "",
    direction: str = "", kind: str = "", party: str = "", evidence: str = "", result: str = "",
    date_from: float | None = Query(None, alias="from"), date_to: float | None = Query(None, alias="to"),
    sort: str = "recent", limit: int = 50, offset: int = 0,
    user: dict = Depends(get_current_user),
) -> dict:
    """The caller's tasks in one view, searched (accent-folded, every word)
    and filtered (comma-separated values per filter), with each facet's
    counts under the other filters and each view's count under all of them."""
    uid = _uid(user)
    if view not in VIEWS:
        raise HTTPException(400, f"view is one of {', '.join(VIEWS)}")
    filters = {"status": _csv(status), "channel": _csv(channel), "agent": _csv(agent),
               "direction": _csv(direction), "kind": _csv(kind), "party": _csv(party),
               "evidence": _csv(evidence), "result": _csv(result)}

    def page() -> dict:
        proposals = _waiting_proposals(uid)
        res = store().query(uid, view=view, q=q[:200], filters=filters, date_from=date_from, date_to=date_to,
                            sort=sort, limit=limit, offset=offset, proposal_tasks=list(proposals))
        return {"items": _items(uid, res["rows"], proposals), "total": res["total"],
                "facets": _facets(res["facets"]), "counts": res["counts"], "view": view}

    return await asyncio.to_thread(page)


@router.get("/tasks/{task_id}")
async def taskboard_task(task_id: str, user: dict = Depends(get_current_user)) -> dict:
    """One of the caller's tasks, whole: its timeline across channels (the
    latest 500 events, and how many there are), the keys that link it back,
    every party, the decisions about it and the Autonomous Work actions on it
    (their status now)."""
    uid = _uid(user)

    def detail() -> dict:
        row = _owned(uid, task_id)
        links = store().links(uid, [task_id]).get(task_id, [])
        return {
            "task": _task_item(uid, row),
            "events": [{k: v for k, v in e.items() if k != "user_id"} for e in store().events(uid, task_id)],
            "events_total": store().event_count(uid, task_id),
            "links": [{"key": link["key"], "strength": link["strength"], "created_at": link["created_at"]}
                      for link in links],
            "parties": store().parties_for(uid, [task_id]).get(task_id, []),
            "decisions": [{k: v for k, v in d.items() if k != "user_id"} for d in store().decisions(uid, task_id)],
            "proposals": _proposals(uid, links),
        }

    return await asyncio.to_thread(detail)


def _proposals(uid: str, links: list[dict]) -> list[dict]:
    ids = [str(link["key"]).removeprefix("autonomy:") for link in links if str(link["key"]).startswith("autonomy:")]
    if not ids:
        return []
    try:
        from captain_claw.flight_deck.autonomy import get_store as autonomy_store
        actions = [autonomy_store().get_action(i) for i in ids]
    except Exception as exc:
        log.debug("Taskboard: proposals unavailable", error=str(exc))
        return []
    keep = ("id", "status", "title", "kind", "rationale", "risk", "outcome", "outcome_note", "created_at",
            "dispatched_at", "completed_at")
    return [{k: a.get(k) for k in keep} for a in actions if a and a.get("user_id") == uid]


class ActionBody(BaseModel):
    action: str
    value: Any = None


@router.post("/tasks/{task_id}/action")
async def taskboard_action(task_id: str, body: ActionBody, user: dict = Depends(get_current_user)) -> dict:
    """Correct one of the caller's tasks. The field it touches is locked — no
    source writes it again — until the correction is undone (``undo``)."""
    uid = _uid(user)
    if body.action not in ACTIONS:
        raise HTTPException(400, f"action is one of {', '.join(ACTIONS)}")

    def apply() -> dict:
        try:
            row = store().correct(uid, task_id, body.action, body.value, actor=_actor(user))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        if row is None:
            raise HTTPException(404, "task not found")
        return {"task": _task_item(uid, row)}

    return await asyncio.to_thread(apply)


async def _user_sync(uid: str) -> dict:
    """The caller's own sweep — one at a time and one per 2 s for each user.
    A call while one waits to start joins it; a call while one reads gets the
    next (which still sees what changed just now). A waiting call holds no
    worker thread."""
    queued = _QUEUED_SYNCS.get(uid)
    if queued is None:
        queued = _QUEUED_SYNCS[uid] = asyncio.ensure_future(_next_sync(uid, _RUNNING_SYNCS.get(uid)))
    return await asyncio.shield(queued)


async def _next_sync(uid: str, before: asyncio.Future | None) -> dict:
    if before is not None:
        await asyncio.wait([before])
    last = taskboard_sources.user_synced_at(uid)
    if last is not None and last + _SYNC_GAP_S > time.time():
        await asyncio.sleep(last + _SYNC_GAP_S - time.time())
    me = asyncio.current_task()
    _QUEUED_SYNCS.pop(uid, None)
    _RUNNING_SYNCS[uid] = me
    try:
        return await asyncio.to_thread(taskboard_sources.run_sweep, store(), uid)
    finally:
        if _RUNNING_SYNCS.get(uid) is me:
            _RUNNING_SYNCS.pop(uid, None)


@router.post("/sync")
async def taskboard_sync(user: dict = Depends(get_current_user)) -> dict:
    """Read every source into the caller's board now (the dialog opens, or a
    proposal was just approved), then the counts."""
    uid = _uid(user)
    res = await _user_sync(uid)
    counts = await asyncio.to_thread(_counts, uid)
    return {"ok": not res["problems"], "new_events": sum(res["new_events"].values()), "counts": counts,
            "sweep": taskboard_sources.sweep_state(uid)}


@router.get("/summary")
async def taskboard_summary(user: dict = Depends(get_current_user)) -> dict:
    """The tile badge: how many tasks need the caller."""
    counts = await asyncio.to_thread(_counts, _uid(user))
    return {"needs_me": counts["needs_me"]}


def _settings(uid: str) -> dict:
    s = store().get_settings(uid)
    passive = s.get("passive") or {}
    return {
        "auto_close": bool(s["auto_close"]),
        "jev_per_day": int(s["jev_per_day"]),
        "extract_per_day": int(s["extract_per_day"]),
        "passive": {k: {"on": bool((passive.get(k) or {}).get("on")),
                        "consented_at": (passive.get(k) or {}).get("consented_at")} for k in PASSIVE_SOURCES},
        "backfill_done_at": s.get("backfill_done_at"),
        "updated_at": s.get("updated_at"),
    }


@router.get("/settings")
async def taskboard_settings(user: dict = Depends(get_current_user)) -> dict:
    return await asyncio.to_thread(_settings, _uid(user))


class SettingsBody(BaseModel):
    auto_close: bool | None = None
    jev_per_day: int | None = Field(None, ge=0, le=1000)
    extract_per_day: int | None = Field(None, ge=0, le=1000)


@router.put("/settings")
async def taskboard_save_settings(body: SettingsBody, user: dict = Depends(get_current_user)) -> dict:
    """Whether Jev may close a task by itself (on by default) and the daily
    caps on Jev questions and extractions (0 = none). The opt-in inboxes are
    switched on with their consent, not here."""
    uid = _uid(user)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if "auto_close" in fields:
        fields["auto_close"] = int(fields["auto_close"])

    def save() -> dict:
        store().save_settings(uid, **fields)
        return _settings(uid)

    return await asyncio.to_thread(save)

