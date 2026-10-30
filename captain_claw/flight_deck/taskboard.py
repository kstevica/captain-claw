"""Taskboard — every task the caller's agents received or sent, on one board.

  GET  /fd/taskboard/status              the sources (tracked or not), each view's count, the sweep
  GET  /fd/taskboard/tasks               one view, searched, filtered, faceted and paged
  GET  /fd/taskboard/tasks/{id}          one task: its timeline, links, parties, decisions, proposals
  POST /fd/taskboard/tasks/{id}/action   a correction: done · reopen · dismiss · not_task · snooze ·
                                         unsnooze · rename · owes_next · ack_failed · undo ·
                                         confirm_task · not_done
  GET  /fd/taskboard/tasks/{id}/why      the latest decisions about a task, and why it is where it is
  POST /fd/taskboard/sync                read the sources now, for the caller's board
  GET  /fd/taskboard/summary             {needs_me} — the tile badge's poll
  GET/PUT /fd/taskboard/settings         auto-close, the daily caps (and today's counts), Jev,
                                         the opt-in inboxes (taskboard_passive)
  GET  /fd/taskboard/export              the caller's tasks under the same filters, as CSV or JSONL

v2 routes live with what they do: ``/tasks/{id}/propose`` (taskboard_nextsteps),
``/tasks/{id}/merge|unmerge|split|not_same`` (taskboard_threads) and
``/backfill`` (taskboard_backfill); each module has its own router.

A task belongs to exactly one board (``user_id``): the caller sees only their
own, and someone else's task is a 404 — there is no owner or admin view of
anyone else's board. The kiosk lock is the UI's; this API is the boundary.

What feeds the board is in ``taskboard_sources`` (a sweep every 120 s while
Flight Deck runs) and ``taskboard_ingest`` (the agents' turns); what each task
is, who owes it and whether it is done in ``taskboard_triage`` (extraction on
the agent's own model, then Jev — rules without it); the store, lanes and
search in ``taskboard_store``. A
proposal waiting on a task is approved or rejected through the existing
``/fd/autonomy/actions/{id}/approve|reject``, a held Gmail reply through
``/fd/google/gmail-autoreply/activity/{id}/send|dismiss``; ``POST /sync``
after either moves the row to its new lane at once.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel

from captain_claw.flight_deck import taskboard_sources, taskboard_store, taskboard_triage
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
from captain_claw.outbound_guard import _fold

log = get_logger(__name__)

router = APIRouter(prefix="/fd/taskboard", tags=["taskboard"])

ACTIONS = ("done", "reopen", "dismiss", "not_task", "snooze", "unsnooze", "rename", "owes_next",
           "ack_failed", "undo", "confirm_task", "not_done")
PASSIVE_SOURCES = ("telegram_inbox", "whatsapp_web", "gmail")

# What the Sources sheet lists: what the board reads, what it reads only once
# the user switches it on, and what it doesn't.
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
    {"key": "agent_chats", "label": "Agent chats on every channel", "tracked": True, "on": True,
     "note": "Every turn your agents take for you — web chat, WhatsApp, Telegram, devices, linked systems "
             "and the rest — with the tool calls in it and what they sent."},
    # v2 passive: off until the user switches one on (taskboard_passive.sources says which is).
    {"key": "telegram_inbox", "label": "Your Telegram inbox", "tracked": False, "on": False,
     "note": "Off — read only once you switch it on."},
    {"key": "whatsapp_web", "label": "Your WhatsApp chats", "tracked": False, "on": False,
     "note": "Off — read only once you switch it on."},
    {"key": "gmail_inbox", "label": "Your Gmail inbox", "tracked": False, "on": False,
     "note": "Off — read only once you switch it on."},
    # v2 history: taskboard_backfill.sources_view says how the fill stands (and adds the to-do lists).
    {"key": "history", "label": "The last 30 days", "tracked": False, "on": False,
     "note": "Your agents' chats of the last 30 days, filled in once."},
    {"key": "not_covered", "label": "Not tracked", "tracked": False, "on": False,
     "note": "Outlook inbound mail; Slack, Discord and Twitter; Docker agents' history and to-do lists; chats "
             "compacted away before the history was filled in; calendar invites; a voice note asking several "
             "things (one task); payment approvals; delegations between agents; a Messenger chat as your own. "
             "Next steps are suggested for requests and sent messages only, and a reply in a chat goes only "
             "where Captain can write: WhatsApp numbers on the deck's allowlist, Telegram chats of the deck's "
             "own bot, devices and Messenger chats while connected."},
]

_HIDDEN = ("user_id", "agent_owner_id", "search_text", "fingerprint", "extracted_fp", "decided_fp")
_FACET_LABELS = {"status": STATUS_LABELS, "direction": DIRECTION_LABELS, "kind": KIND_LABELS,
                 "evidence": EVIDENCE_LABELS, "result": RESULT_LABELS}
_SWEEP_STOP_S = 5.0
_SYNC_GAP_S = 2.0
_WHY_MAX = 20
JEV_PER_DAY_MAX = 2000
EXTRACT_PER_DAY_MAX = 1000

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
    """Start the sweep and the triage worker (Flight Deck's lifespan).
    ``FD_TASKBOARD_DISABLED=true`` keeps both off; the board then fills only
    when someone opens it, and nothing is triaged."""
    global _loop_task, _loop_stop
    if os.environ.get("FD_TASKBOARD_DISABLED", "").lower() in ("true", "1", "yes") or _loop_task is not None:
        return
    _loop_stop = asyncio.Event()
    _loop_task = asyncio.create_task(taskboard_sources.sweep_loop(_loop_stop))
    taskboard_triage.start()


async def stop() -> None:
    global _loop_task, _loop_stop
    await taskboard_triage.stop()
    # v2 history: a backfill still reading stops (it starts again on the next open).
    from captain_claw.flight_deck import taskboard_backfill

    await taskboard_backfill.stop()
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
    """``{task_id: ProposalView}`` — the newest Autonomous Work action
    awaiting approval on each of the user's tasks (read live, so the row moves
    lanes the moment it is approved), a board next step with its wording and
    recipients; next steps whose task closed expire first
    (taskboard_nextsteps.waiting_proposals)."""
    from captain_claw.flight_deck import taskboard_nextsteps

    return taskboard_nextsteps.waiting_proposals(uid)


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


# The kinds a model names a party by (taskboard_triage): such a party, never
# verified, is only a name read out of the text.
_READ_KINDS = ("email", "phone", "telegram", "name")


def _shown_parties(uid: str, parties: list[dict]) -> list[dict]:
    """The parties as the UI shows them: ``me`` on the board user's own (a
    deck user's id is never the name to show), and each person once — a
    party read out of the text adds nothing when someone of its folded name
    is there already (a recorded party, the board user going by their own
    name, or one read before it; one with an address wins over a bare name).
    A recorded party is always kept."""
    def name(p: dict) -> str:
        return " ".join(_fold(str(p.get("display") or "")).split())

    def read(p: dict) -> bool:
        return not p.get("verified") and p.get("kind") in _READ_KINDS

    recorded = {name(p) for p in parties if not read(p)}
    best: dict[str, int] = {}
    for i, p in enumerate(parties):
        n = name(p)
        if n and read(p) and n not in recorded:
            j = best.get(n)
            if j is None or (parties[j]["kind"] == "name" and p["kind"] != "name"):
                best[n] = i
    return [{**p, "me": p.get("kind") == "fd_user" and p.get("address") == uid}
            for i, p in enumerate(parties)
            if not (read(p) and name(p) and (name(p) in recorded or best.get(name(p)) != i))]


def _item(uid: str, row: dict, parties: list[dict], links: list[dict], proposal: dict | None, lane: str) -> dict:
    item = {k: v for k, v in row.items() if k not in _HIDDEN}
    item["channels"] = _loads(row.get("channels"), [])
    item["user_locks"] = _loads(row.get("user_locks"), {})
    item["flags"] = _loads(row.get("flags"), {})
    item["next_steps"] = _loads(row.get("next_steps"), [])
    item["private"] = bool(row.get("private"))
    item["parties"] = _shown_parties(uid, parties)[:5]
    item["lane"] = lane
    item["proposal"] = proposal
    item["links"] = _link_ids(links, proposal)
    return item


def _items(uid: str, rows: list[dict], proposals: dict[str, dict]) -> list[dict]:
    ids = [r["id"] for r in rows]
    parties = store().parties_for(uid, ids)
    links = store().links(uid, ids)
    return [_item(uid, r, parties.get(r["id"], []), links.get(r["id"], []), proposals.get(r["id"]),
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


def _facets(uid: str, raw: dict[str, list[tuple[str, str, int]]]) -> dict[str, list[dict]]:
    """Each filter's values with their counts; a People value says whether
    it is the board user (``me``, as their parties do), so the board names
    them "You" whatever the tasks on the page are."""
    def value(dim: str, v: str, label: str, n: int) -> dict:
        out = {"id": v, "label": _facet_label(dim, v, label), "count": n}
        return {**out, "me": v == f"fd_user:{uid}"} if dim == "party" else out

    return {dim: [value(dim, v or "", label or "", n) for v, label, n in raw.get(dim, [])] for dim in FACETS}


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
    out = {"sources": SOURCES, "counts": counts, "sweep": taskboard_sources.sweep_state(uid)}
    # v2 history (taskboard_backfill): the 30-day backfill — started by the
    # first look while the board's background work runs, never twice — and
    # the Sources sheet's lines for it and for the agents' to-do lists.
    from captain_claw.flight_deck import taskboard_backfill

    out["backfill"] = await taskboard_backfill.status(uid, start=_loop_task is not None)
    out["sources"] = taskboard_backfill.sources_view(out["sources"], out["backfill"])
    # v2 passive (taskboard_passive): the user's own inboxes, tracked while switched on.
    from captain_claw.flight_deck import taskboard_passive

    out["sources"] = await taskboard_passive.sources_view(uid, out["sources"])
    return out


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
    filters = _filters(status=status, channel=channel, agent=agent, direction=direction, kind=kind, party=party,
                       evidence=evidence, result=result)

    def page() -> dict:
        proposals = _waiting_proposals(uid)
        res = store().query(uid, view=view, q=q[:200], filters=filters, date_from=date_from, date_to=date_to,
                            sort=sort, limit=limit, offset=offset, proposal_tasks=list(proposals))
        return {"items": _items(uid, res["rows"], proposals), "total": res["total"],
                "facets": _facets(uid, res["facets"]), "counts": res["counts"], "view": view}

    return await asyncio.to_thread(page)


def _filters(**raw: str) -> dict[str, list[str]]:
    """The multi-value filters of a list (or an export) query, each comma-separated."""
    return {name: _csv(raw.get(name, "")) for name in FACETS}


# ── Export (v2): the caller's tasks as a file ─────────────────────────


EXPORT_FORMATS = ("csv", "jsonl")
_EXPORT_MAX = 10000
_EXPORT_COLUMNS = (
    "id", "title", "status", "lane", "owes_next", "kind", "direction", "channel", "channels", "agent", "parties",
    "ask", "result_text", "result", "evidence", "urgency", "due_at", "created_at", "last_event_at", "closed_at",
    "closed_by", "closed_by_party", "why", "historical", "private", "retention_until")
_EXPORT_TIMES = ("due_at", "created_at", "updated_at", "last_event_at", "closed_at", "retention_until")
# A cell a spreadsheet would read as a formula (someone's message may start so).
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _export_party(p: dict) -> dict:
    """A party as the file names it: never a deck user's id (the board user
    is "You", another deck user their name)."""
    deck_user = p.get("kind") == "fd_user"
    name = "You" if p.get("me") else str(p.get("name") or p.get("display") or "")
    if deck_user and not p.get("me") and name == str(p.get("address") or ""):
        name = "Someone else"
    return {"role": p.get("role") or "", "name": name, "kind": "" if deck_user else str(p.get("kind") or ""),
            "address": "" if deck_user else str(p.get("address") or ""), "verified": bool(p.get("verified"))}


def _export_rows(uid: str, rows: list[dict], proposals: dict[str, dict]) -> list[dict]:
    """One record per task — what the board shows of it, every party."""
    ids = [r["id"] for r in rows]
    parties = store().parties_for(uid, ids)
    out = []
    for item in _items(uid, rows, proposals):
        flags = item["flags"]
        out.append({
            **{k: item.get(k) for k in ("id", "title", "status", "lane", "owes_next", "kind", "direction", "channel",
                                        "channels", "agent_slug", "ask", "result_text", "result", "evidence",
                                        "urgency", "closed_by", "closed_by_party", "private", *_EXPORT_TIMES)},
            "agent": item.get("agent_name") or item.get("agent_slug") or "",
            "parties": [_export_party(p) for p in _shown_parties(uid, parties.get(item["id"], []))],
            "why": str(flags.get("why") or ""), "historical": bool(flags.get("historical")),
            "flags": {k: v for k, v in flags.items() if k != "why"},
            "proposal": {k: (item["proposal"] or {}).get(k) for k in ("id", "status", "title")}
            if item.get("proposal") else None,
        })
    return out


def _iso_time(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if value else ""
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_FORMULA_START) else text


def _csv_cells(rec: dict) -> list[str]:
    def party(p: dict) -> str:
        who = p["name"] or p["address"]
        if p["address"] and p["address"] != who:
            who = f"{who} <{p['address']}>"
        return f"{who} ({p['role']})" if p["role"] else who

    shown = {**rec, "channels": ", ".join(rec.get("channels") or []),
             "parties": "; ".join(party(p) for p in rec["parties"]),
             "historical": "yes" if rec["historical"] else "", "private": "yes" if rec.get("private") else "",
             **{k: _iso_time(rec.get(k)) for k in _EXPORT_TIMES}}
    return [_cell(shown.get(col)) for col in _EXPORT_COLUMNS]


@router.get("/export")
async def taskboard_export(
    format: str = "csv", view: str = "all", q: str = "", status: str = "", channel: str = "", agent: str = "",
    direction: str = "", kind: str = "", party: str = "", evidence: str = "", result: str = "",
    date_from: float | None = Query(None, alias="from"), date_to: float | None = Query(None, alias="to"),
    sort: str = "recent", user: dict = Depends(get_current_user),
) -> Response:
    """The caller's own tasks — the view, search and filters of ``/tasks``
    — as a file to save (``Content-Disposition: attachment``): CSV, one
    task a row, or JSONL, one task a line. Only ever the caller's board
    (someone else's tasks are never in it, whoever asks), without the
    search text or fingerprints, and never a deck user's id; at most
    10,000 tasks, in the list's order (``sort``)."""
    uid = _uid(user)
    if format not in EXPORT_FORMATS:
        raise HTTPException(400, f"format is one of {', '.join(EXPORT_FORMATS)}")
    if view not in VIEWS:
        raise HTTPException(400, f"view is one of {', '.join(VIEWS)}")
    filters = _filters(status=status, channel=channel, agent=agent, direction=direction, kind=kind, party=party,
                       evidence=evidence, result=result)

    def read() -> list[dict]:
        proposals = _waiting_proposals(uid)
        out: list[dict] = []
        while len(out) < _EXPORT_MAX:
            res = store().query(uid, view=view, q=q[:200], filters=filters, date_from=date_from, date_to=date_to,
                                sort=sort, limit=200, offset=len(out), proposal_tasks=list(proposals),
                                with_facets=False)
            out += _export_rows(uid, res["rows"], proposals)
            if not res["rows"] or len(out) >= res["total"]:
                break
        return out[:_EXPORT_MAX]

    records = await asyncio.to_thread(read)
    if format == "jsonl":
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        media = "application/x-ndjson"
    else:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(_EXPORT_COLUMNS)
        writer.writerows(_csv_cells(r) for r in records)
        body, media = "\ufeff" + buf.getvalue(), "text/csv"  # the BOM: a spreadsheet reads it as UTF-8
    name = f"taskboard-{view}-{time.strftime('%Y%m%d')}.{format}"
    return Response(body, media_type=f"{media}; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})


@router.get("/tasks/{task_id}")
async def taskboard_task(task_id: str, user: dict = Depends(get_current_user)) -> dict:
    """One of the caller's tasks, whole: its timeline across channels (the
    latest 500 events, and how many there are), the keys that link it back,
    every party, the decisions about it and the Autonomous Work actions on it
    (their status now) — and (v2) the tasks merged into it. A task merged
    into another reads as itself (``merged_into`` says where it went)."""
    from captain_claw.flight_deck import taskboard_threads

    uid = _uid(user)

    def detail() -> dict:
        row = _owned(uid, task_id)
        links = store().links(uid, [task_id]).get(task_id, [])
        return {
            **taskboard_threads.detail_extras(uid, task_id),
            "task": _task_item(uid, row),
            "events": [{k: v for k, v in e.items() if k != "user_id"} for e in store().events(uid, task_id)],
            "events_total": store().event_count(uid, task_id),
            "links": [{"key": link["key"], "strength": link["strength"], "created_at": link["created_at"]}
                      for link in links],
            "parties": _shown_parties(uid, store().parties_for(uid, [task_id]).get(task_id, [])),
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


@router.get("/tasks/{task_id}/why")
async def taskboard_why(task_id: str, user: dict = Depends(get_current_user)) -> dict:
    """"Why is this here?": the latest decisions about one of the caller's
    tasks (latest first, at most 20) — who judged (Jev, the rules, the
    agent's model, or the user's own corrections), the question set, the
    probabilities, the move and why in plain words — and why it stands where
    it does now."""
    uid = _uid(user)

    def read() -> dict:
        row = _owned(uid, task_id)
        decisions = store().decisions(uid, task_id, limit=_WHY_MAX)
        items = [{"at": d["at"], "judge": d["judge"], "qset": d["qset"], "applied": d["applied"],
                  "why": d["why"], "answers": d["answers"]} for d in decisions]
        for e in store().events(uid, task_id):
            if e["kind"] == "correction" and e["source"] == "user":
                undone = bool((e.get("refs") or {}).get("undone"))
                items.append({"at": e["at"], "judge": "user", "qset": "", "applied": e["summary"], "answers": {},
                              "why": "You undid this later." if undone
                              else "You did this yourself, so no machine pass changes it."})
        items.sort(key=lambda i: -float(i["at"] or 0))
        return {"items": items[:_WHY_MAX], "summary": taskboard_triage.why_summary(row, decisions)}

    return await asyncio.to_thread(read)


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
    """The tile badge: how many tasks need the caller — leaving out those Jev
    admitted at low confidence (they wait in Needs me, unannounced)."""
    uid = _uid(user)

    def badge() -> int:
        return store().badge_count(uid, proposal_tasks=list(_waiting_proposals(uid)))

    return {"needs_me": await asyncio.to_thread(badge)}


def _settings(uid: str) -> dict:
    s = store().get_settings(uid)
    passive = s.get("passive") or {}
    return {
        "auto_close": bool(s["auto_close"]),
        "jev_per_day": int(s["jev_per_day"]),
        "extract_per_day": int(s["extract_per_day"]),
        "counters": store().daily_counts(uid),
        "passive": {k: {"on": bool((passive.get(k) or {}).get("on")),
                        "consented_at": (passive.get(k) or {}).get("consented_at")} for k in PASSIVE_SOURCES},
        "backfill_done_at": s.get("backfill_done_at"),
        "updated_at": s.get("updated_at"),
    }


async def _settings_view(uid: str) -> dict:
    """The settings, today's counts and whether Jev is connected for the
    caller (their own key, or the team's)."""
    out = await asyncio.to_thread(_settings, uid)
    out["jev"] = await taskboard_triage.jev_connection(uid)
    # v2 passive (taskboard_passive): each inbox's switch, and whether it is there to read.
    from captain_claw.flight_deck import taskboard_passive

    out["passive"] = await taskboard_passive.settings_view(uid)
    return out


@router.get("/settings")
async def taskboard_settings(user: dict = Depends(get_current_user)) -> dict:
    return await _settings_view(_uid(user))


class SettingsBody(BaseModel):
    auto_close: bool | None = None
    jev_per_day: int | None = None
    extract_per_day: int | None = None
    passive: dict[str, Any] | None = None  # v2 passive: {inbox: {on, consent}}


@router.put("/settings")
async def taskboard_save_settings(body: SettingsBody, user: dict = Depends(get_current_user)) -> dict:
    """Whether Jev may close a task by itself (on by default; off, a task it
    is sure of is marked "probably done" for the user to confirm) and the
    daily caps: tasks Jev is asked about (0–2000) and extractions (0–1000);
    0 = none, rules alone then. An opt-in inbox is switched on only with its
    consent (``passive: {inbox: {on: true, consent: true}}``) and off
    deleting what it put on the board — then the answer is ``{settings,
    deleted}`` (the settings' fields also at the top, as otherwise)."""
    uid = _uid(user)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    passive = fields.pop("passive", None)
    for name, top in (("jev_per_day", JEV_PER_DAY_MAX), ("extract_per_day", EXTRACT_PER_DAY_MAX)):
        if name in fields and not 0 <= fields[name] <= top:
            raise HTTPException(400, f"{name} is a number from 0 to {top}")
    if "auto_close" in fields:
        fields["auto_close"] = int(fields["auto_close"])
    deleted = None
    if passive is not None:  # v2 passive (taskboard_passive): checked whole before anything moves
        from captain_claw.flight_deck import taskboard_passive

        deleted = await taskboard_passive.apply(uid, passive)
    await asyncio.to_thread(lambda: store().save_settings(uid, **fields))
    out = await _settings_view(uid)
    return out if deleted is None else {**out, "settings": out, "deleted": deleted}

