"""Taskboard sources — the records Flight Deck already keeps, read as tasks (v0).

Six adapters, one per FD ledger, each turning its rows into tasks and events
on the board of the user the row belongs to:

  fd_scheduler       a standing task per scheduled job; each run and each of
                     its deliveries an event (running → agents working, ok →
                     done, error → failed, the next run opens it again; a job
                     switched off waits on nobody, a deleted one is dismissed)
  follow_ups         a task per tracked open loop (due → needs me, not yet due
                     → waiting on them), linked to its Gmail thread
  autonomy           a task per Autonomous Work action (awaiting approval →
                     needs me), or an event on the follow-up it is about
  gmail_autoreplies  a task per inbound thread an auto-reply held or sent
  gmail_sends        an event on the thread's task — or a "sent" task of its own
                     (an auto-reply's send is its auto-reply's; an agent's
                     waits ten minutes for the tool call that names its thread)
  agent_links        a task per message a linked system got an answer to, or
                     the agent sent it on its own

What an auto-reply or a link message said is never copied onto the board:
their ledgers clear it after a while, and the board keeps what it has. A deleted user's
rows (some ledgers keep them) go on nobody's board.

v1 adds two, over what the agents report (``taskboard_ingest``):

  agent_turns        automation turns waiting for the task above that their
                     run, action or message makes
  tool_audit         every tool call made in a turn, on that turn's task

The agents' turns themselves come in as they happen
(``POST /fd/taskboard/agent/turns``) and are kept 30 days (each full sweep
forgets older ones).

Every adapter's writes go through one place (``_Pass.ingest``): a task gets
one requester row per person (none beside the user on their own chat), and
someone writing in on a request that was done queues it for triage again
(``taskboard_ingest.inbound_after_done``).

Rows change after they are first read (a proposal is approved, a follow-up
done, a run finishes), so no adapter trusts a plain "newer than" cursor: each
reads what is new since its cursor (``source_cursors``) and also every row not
final yet. Reading a row twice adds nothing — every event is keyed by the
row and what it stands for (``run:<id>:start``, ``<id>:<state>:<n>``; a
state counts when it differs from the last one the board has from that row),
and only a new event moves the task. The sweep reads the stores' files
directly (it never wakes or calls an agent) every 120 s while Flight Deck
runs, and for one user when their board asks (``POST /fd/taskboard/sync``).
An adapter whose store or table isn't on this deck is skipped; one that
fails is logged and the rest go on.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from email.utils import getaddresses
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import taskboard_store
from captain_claw.flight_deck.taskboard_store import TaskboardStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

SWEEP_EVERY_S = 120.0
_FIRST_SWEEP_S = 20.0
_BATCH = 2000          # rows one adapter reads in one pass
_OVERLAP_S = 5.0       # a time cursor reaches back this far (rows read twice add nothing)
TURNS_KEPT_S = 30 * 86400

# Labels of the sources, as the Sources sheet names them.
SOURCE_LABELS = {
    "fd_scheduler": "Scheduled tasks",
    "follow_ups": "Follow-ups",
    "autonomy": "Autonomous Work proposals",
    "gmail_autoreplies": "Email auto-replies",
    "gmail_sends": "Email your agents sent",
    "agent_links": "Linked systems",
}
# And of what the agents report (v1): one row on the sheet, agent_chats.
AGENT_LABELS = {
    "agent_turns": "Agent chats on every channel",
    "tool_audit": "Your agents' tool calls",
    "agent_todos": "Your agents' to-do lists",  # v2 (taskboard_backfill)
}
# v2 passive: the user's own inboxes, each read only while they switch it on (taskboard_passive).
PASSIVE_LABELS = {
    "telegram_inbox": "Your Telegram inbox",
    "whatsapp_web": "Your WhatsApp chats",
    "gmail_inbox": "Your Gmail inbox",
}

_STATE: dict[str, Any] = {"running": False, "last_at": None, "message": ""}
_USER_SYNCED: dict[str, tuple[float, str]] = {}
_SWEEP_LOCK = threading.Lock()


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _epoch(iso: Any) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _json(raw: Any, default: Any) -> Any:
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw else default
    except ValueError:
        return default
    return value if isinstance(value, type(default)) else default


def _head(text: Any, n: int = 300) -> str:
    return " ".join(str(text or "").split())[:n]


def _addresses(raw: Any, role: str) -> list[dict]:
    """``"Ana <ana@x.co>, b@y.org"`` → parties (header addresses: verified)."""
    out = []
    for name, addr in getaddresses([str(raw or "")]):
        addr = addr.strip().lower()
        if "@" in addr:
            out.append({"role": role, "kind": "email", "address": addr, "display": name.strip() or addr,
                        "verified": True})
    return out


def _store_path(module: Any) -> Path:
    """Where *module*'s store keeps its file: its open store's, else the path
    it would open — without opening it (the scheduler's store marks runs left
    running as interrupted when it opens)."""
    store = getattr(module, "_STORE", None)
    return Path(store.db_path) if store is not None else Path(module._db_path())


def _rows(path: Path, sql: str, args: tuple | list = ()) -> list[dict]:
    """A SELECT on another store's file — none when the file isn't there (that
    store never ran on this deck). Read only; the file is never created."""
    if not path.is_file():
        return []
    conn = sqlite3.connect(str(path), timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, tuple(args)).fetchall()]
    finally:
        conn.close()


def _agent_index() -> dict[str, dict]:
    """``{slug: {owner, name, ref}}`` from FD's process registry (read from
    disk; the agents themselves aren't touched)."""
    try:
        from captain_claw.flight_deck.agent_sharing import process_ref
        from captain_claw.flight_deck.server import _load_process_registry
        registry = _load_process_registry()
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for slug, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        try:
            ref = process_ref(slug, entry)
        except Exception:
            ref = ""
        out[slug] = {"owner": str(entry.get("owner") or ""), "name": str(entry.get("name") or slug), "ref": ref}
    return out


def _user_ids() -> set[str] | None:
    """Everyone with an account on this deck — None when that can't be read
    (no users table: nobody was ever deleted from it)."""
    path = _fd_db_path()
    if not path.is_file():
        return None
    try:
        return {str(r["id"]) for r in _rows(path, "SELECT id FROM users")}
    except sqlite3.OperationalError:
        return None


class _Pass:
    """One sweep: the store, whose rows (everyone's, or one user's), the agents."""

    def __init__(self, store: TaskboardStore, user_id: str | None, *, agents: dict[str, dict] | None = None) -> None:
        self.store = store
        self.user_id = user_id
        self.auth = _auth_enabled()
        self.agents = _agent_index() if agents is None else agents
        self.users = _user_ids() if self.auth else None
        self.new_events: dict[str, int] = {}

    def board_user(self, raw: Any) -> str | None:
        """The board a row belongs to. Without sign-in there is one, the local
        user's; with it, a row with no real owner (a system job, a legacy
        ``local`` row) is on nobody's — and neither is a deleted user's (some
        ledgers keep their rows; their board went with them)."""
        if not self.auth:
            return "local"
        uid = str(raw or "").strip()
        if not uid or uid == "local":
            return None
        return uid if self.users is None or uid in self.users else None

    def mine(self, column: str) -> tuple[str, list]:
        """The SQL that keeps a one-user pass to that user's rows."""
        if self.user_id is None or not self.auth:
            return "", []
        return f" AND {column} = ?", [self.user_id]

    def cursor(self, source: str) -> str:
        return self.store.get_cursor(source)

    def save_cursor(self, source: str, value: str) -> None:
        # A one-user pass read only that user's rows: the cursor stays for
        # the next full sweep (which reads them again, adding nothing).
        if self.user_id is None:
            self.store.set_cursor(source, value)

    def agent(self, slug: str, name: str = "") -> dict:
        info = self.agents.get(slug or "")
        if info:
            return {"agent_slug": slug, "agent_name": info["name"], "agent_ref": info["ref"],
                    "agent_owner_id": info["owner"]}
        return {"agent_slug": slug or "", "agent_name": name or slug or "", "agent_ref": "", "agent_owner_id": ""}

    def ingest(self, source: str, uid: str, **kw: Any) -> tuple[str | None, bool]:
        if kw.get("parties"):
            kw["parties"] = _parties_to_add(self.store, uid, kw.get("keys") or [], kw["parties"])
        task_id, new = self.store.ingest(uid, **kw)
        if new:
            self.new_events[source] = self.new_events.get(source, 0) + 1
            ev = kw.get("event") or {}
            if task_id and ev.get("kind") == "received" and ev.get("actor_kind") == "party":
                # Someone writing on a request that was done (a reply on its thread).
                from captain_claw.flight_deck import taskboard_ingest

                if taskboard_ingest.inbound_after_done(self.store, uid, task_id, kw.get("keys") or [],
                                                       ref=str(ev.get("source_ref") or "")):
                    taskboard_ingest._kick(uid)
        return task_id, new


def _parties_to_add(store: TaskboardStore, uid: str, keys: list[str], parties: list[dict]) -> list[dict]:
    """What of *parties* a source adds to the task *keys* already find: a
    requester only when the task has none of that name yet (one person, not
    two rows) — and on the board user's own chat (``flags.own_chat``) none
    at all, the user is its only requester: someone it names is mentioned,
    unless they are on the task already. All of them for a new task."""
    if not any(p.get("role") == "requester" for p in parties):
        return list(parties)
    held = next((t for t in (store.task_for_key(uid, k) for k in keys) if t), None)
    if held is None:
        return list(parties)
    from captain_claw.outbound_guard import _fold

    task = store.get_task(uid, held) or {}
    own = bool(_json(task.get("flags"), {}).get("own_chat"))
    have = store.parties_for(uid, [held]).get(held, [])
    named = {_fold(p["display"] or p["address"]) for p in have if p["role"] == "requester"}
    on_task = {(p["kind"], p["address"]) for p in have}
    out = []
    for p in parties:
        if p.get("role") != "requester":
            out.append(p)
        elif own:
            if (p.get("kind") or "email", str(p.get("address") or "").strip()) not in on_task:
                out.append({**p, "role": "mentioned"})
        elif _fold(p.get("display") or p.get("address") or "") not in named:
            out.append(p)
    return out


# ── Scheduled tasks (fd_scheduler: scheduler_jobs + scheduler_runs) ───


_DELIVERY_CHANNEL = {"chat": "web", "email": "email"}


def _job_task(p: _Pass, job: dict) -> dict:
    from captain_claw.flight_deck.fd_scheduler import _task_title

    prompt = str(job.get("prompt") or "")
    flow = str(job.get("flow_id") or "")
    return {
        "title": _task_title(job), "ask": prompt or (f"Runs the flow {flow}" if flow else ""),
        **p.agent(str(job.get("agent_slug") or "")),
        "kind": "scheduled", "direction": "internal", "channel": "scheduled", "owes_next": "agent",
        "source": "fd_scheduler", "source_ref": str(job["id"]), "triage_state": "skipped",
    }


def _job_recipients(job: dict | None) -> list[dict]:
    if not job:
        return []
    from captain_claw.flight_deck.fd_scheduler import norm_delivery

    try:
        delivery = norm_delivery(job.get("delivery")) if job.get("delivery_kind") == "agent" else {}
    except ValueError:
        return []
    return _addresses(delivery.get("email_to"), "recipient") if delivery.get("email") else []


# A legacy WhatsApp delivery the number's mute (or allowlist) held back: the
# run itself went through.
_MUTED = "skipped:muted-or-not-allowed"


def _skipped(status: str) -> bool:
    """A run that never began (quiet hours, the daily allowance spent)."""
    s = (status or "").strip()
    return s.startswith("skipped") and s != _MUTED


def _run_end(status: str, deliveries: list[dict], result: str, agent_name: str) -> tuple[dict | None, str, str]:
    """What a finished run makes of its task: ``(patch, outcome, summary)`` —
    no patch for a run that never began (the task stays as it was)."""
    s = (status or "").strip()
    delivered = any(d.get("ok") for d in deliveries if isinstance(d, dict))
    done = {"status": "done", "owes_next": "nobody", "closed_by": "source", "closed_by_party": agent_name,
            "evidence": "verified" if delivered else "claimed", "result_text": result}
    if s.startswith("ok"):
        return {**done, "result": "success"}, "ok", _head(result) or "Run finished"
    if s.startswith("partial"):
        return {**done, "result": "partial"}, "ok", f"Run finished, {s.removeprefix('partial:').strip()}"
    if s == _MUTED:
        return {**done, "result": "partial"}, "ok", "Run finished, not delivered: the number is muted or not allowed"
    if s.startswith("skipped"):
        why = {"skipped:limit": "the daily allowance is spent", "skipped:quiet": "quiet hours"}.get(s, s)
        return None, "blocked", f"Run skipped: {why}"
    return ({"status": "failed", "result": "failed", "closed_by": "source", "result_text": result or s},
            "error", f"Run failed: {s.removeprefix('error:')}")


def _scheduler(p: _Pass) -> None:
    from captain_claw.flight_deck import fd_scheduler

    path = _store_path(fd_scheduler)
    mine, mine_args = p.mine("owner_id")
    jobs = {j["id"]: j for j in _rows(path, f"SELECT * FROM scheduler_jobs WHERE 1 = 1{mine} LIMIT ?",
                                      (*mine_args, _BATCH))}
    for job in jobs.values():
        uid = p.board_user(job.get("owner_id"))
        if uid is None:
            continue
        task = _job_task(p, job)
        slug, session = task["agent_slug"], str(job.get("agent_session_id") or "")
        # A job edited at the source shows its new name (and agent, when FD knows it).
        kept = ("title", "ask", "agent_slug", "agent_name", "agent_ref") if slug in p.agents else ("title", "ask")
        p.ingest("fd_scheduler", uid, keys=[f"sched:job:{job['id']}"], create=task,
                 touch={k: task[k] for k in kept},
                 extra_keys=[f"session:{slug}:{session}"] if slug and session else [],
                 parties=_job_recipients(job),
                 event={"source": "fd_scheduler", "source_ref": f"job:{job['id']}",
                        "at": _epoch(job.get("created_at")), "kind": "note", "channel": "scheduled",
                        "actor_kind": "me" if job.get("prompt_author") != "agent" else "agent",
                        "summary": f"Scheduled: {job.get('schedule') or ''}"})

    # Runs from the oldest one still going (it will change) or the next new one.
    start = int(p.cursor("fd_scheduler") or 0)
    runs = _rows(path, f"SELECT * FROM scheduler_runs WHERE id >= ?{mine} ORDER BY id LIMIT ?",
                 (start, *mine_args, _BATCH))
    for run in runs:
        uid = p.board_user(run.get("owner_id"))
        if uid is None:
            continue
        job = jobs.get(run["job_id"])
        create = _job_task(p, job) if job else {
            "title": str(run.get("job_name") or "") or "Scheduled task", **p.agent(str(run.get("agent_slug") or "")),
            "kind": "scheduled", "direction": "internal", "channel": "scheduled", "owes_next": "agent",
            "source": "fd_scheduler", "source_ref": str(run["job_id"]), "triage_state": "skipped"}
        keys = [f"sched:job:{run['job_id']}"]
        slug, session = create["agent_slug"], str(run.get("session_id") or "")
        extra = [f"sched:run:{run['id']}"] + ([f"session:{slug}:{session}"] if slug and session else [])
        refs = {"run_id": run["id"], "job_id": run["job_id"]}
        # A run that never began leaves the task as it was. One that did
        # opens it again — over the user's Done or Reopen too, which were
        # about the run before (their Dismiss or Not a task is about the job).
        idle = _skipped(str(run.get("status") or ""))
        cur = p.store.own_row_for_key(uid, keys[0])
        unlock = ("status",) if cur and cur["status"] not in ("dismissed", "not_task") and not idle else ()
        _, first_seen = p.ingest(
            "fd_scheduler", uid, keys=keys, create=create, extra_keys=extra, unlock=unlock,
            event={"source": "fd_scheduler", "source_ref": f"run:{run['id']}:start",
                   "at": run.get("started_at"), "kind": "status", "channel": "scheduled",
                   "actor_kind": "system", "actor": "Scheduler", "refs": refs,
                   "summary": f"Run started ({run.get('trigger') or 'scheduled'})"},
            patch=None if idle else {"status": "open", "owes_next": "agent", "evidence": "none"})
        if run.get("status") == "running":
            continue
        deliveries = [d for d in _json(run.get("deliveries"), []) if isinstance(d, dict)]
        at = run.get("finished_at") or run.get("started_at")
        for i, d in enumerate(deliveries):
            via = str(d.get("via") or "")
            ok = bool(d.get("ok"))
            note = _head(d.get("note"), 200)
            p.ingest("fd_scheduler", uid, keys=keys, create=create,
                     parties=_addresses(d.get("to"), "recipient") if via == "email" else [],
                     event={"source": "fd_scheduler", "source_ref": f"run:{run['id']}:delivery:{i}", "at": at,
                            "kind": "delivery", "channel": _DELIVERY_CHANNEL.get(via, via),
                            "actor_kind": "system", "actor": "Scheduler", "outcome": "ok" if ok else "error",
                            "evidence": "recorded" if ok else "", "refs": {**refs, "via": via},
                            "summary": f"Delivered by {via}" + (f": {note}" if note else "") if ok
                            else f"Not delivered by {via}" + (f": {note}" if note else "")})
        patch, outcome, summary = _run_end(str(run.get("status") or ""), deliveries,
                                           str(run.get("result") or ""), create["agent_name"])
        if patch is None and not first_seen:
            # Seen going, then skipped after all: no agent is on it now.
            patch = {"owes_next": "nobody"}
        p.ingest("fd_scheduler", uid, keys=keys, create=create, patch=patch,
                 event={"source": "fd_scheduler", "source_ref": f"run:{run['id']}:end", "at": at,
                        "kind": "status", "channel": "scheduled", "actor_kind": "agent",
                        "actor": create["agent_name"], "outcome": outcome, "summary": summary,
                        "refs": {**refs, "status": run.get("status")}})
    if runs:
        going = [r["id"] for r in runs if r.get("status") == "running"]
        p.save_cursor("fd_scheduler", str(min(going) if going else runs[-1]["id"] + 1))
    if path.is_file():  # no file, no jobs read: that isn't every job deleted
        _jobs_off_or_gone(p, jobs, runs, complete=len(jobs) < _BATCH)


def _jobs_off_or_gone(p: _Pass, jobs: dict, runs: list[dict], *, complete: bool) -> None:
    """A job switched off or deleted before a run moved its task: off, the
    task waits on nobody (until the job is on again); deleted, it is
    dismissed. Only open tasks with no run going are touched, each move an
    event of its own (``job:<id>:<off|on|gone>:<n>``)."""
    going = {str(r["job_id"]) for r in runs if r.get("status") == "running"}
    moves = {"off": ({"owes_next": "nobody"}, "Switched off"),
             "on": ({"owes_next": "agent"}, "Switched on again"),
             "gone": ({"status": "dismissed", "owes_next": "nobody", "closed_by": "source"},
                      "The scheduled task was deleted")}
    for t in p.store.keyed_tasks("sched:job:", user_id=p.user_id if p.auth else None, status="open"):
        job_id = t["key"].removeprefix("sched:job:")
        job = jobs.get(job_id)
        if job_id in going or (job is None and not complete):
            continue  # a run will move it; or the job wasn't read this pass
        state = "gone" if job is None else ("on" if job.get("enabled") else "off")
        ref = p.store.next_state_ref("fd_scheduler", f"job:{job_id}", state)
        if not ref or (state == "on" and ref.endswith(":0")):
            continue  # nothing changed — or on, as it always was
        patch, summary = moves[state]
        p.ingest("fd_scheduler", t["user_id"], keys=[t["key"]], patch=patch,
                 event={"source": "fd_scheduler", "source_ref": ref, "at": time.time(), "kind": "status",
                        "channel": "scheduled", "actor_kind": "system", "actor": "Scheduler",
                        "summary": summary, "refs": {"job_id": job_id, "state": state}})


# ── Follow-ups (events: follow_ups, + external_events for the thread) ─


def _follow_ups(p: _Pass) -> None:
    from captain_claw.flight_deck import events

    path = _store_path(events)
    mine, mine_args = p.mine("user_id")
    since = p.cursor("follow_ups")
    started = time.time()
    # Open ones every pass: one falls due without its row changing.
    rows = _rows(path, f"SELECT * FROM follow_ups WHERE (updated_at >= ? OR status = 'open'){mine} "
                       "ORDER BY updated_at LIMIT ?", (since, *mine_args, _BATCH))
    origin_ids = sorted({r["origin_event_id"] for r in rows if r.get("origin_event_id")})
    origins: dict[str, dict] = {}
    for i in range(0, len(origin_ids), 500):
        chunk = origin_ids[i:i + 500]
        for ev in _rows(path, f"SELECT id, source, metadata FROM external_events WHERE id IN "
                              f"({','.join('?' * len(chunk))})", chunk):
            origins[ev["id"]] = {"source": ev["source"], **_json(ev.get("metadata"), {})}
    now = time.time()
    for fu in rows:
        uid = p.board_user(fu.get("user_id"))
        if uid is None:
            continue
        origin = origins.get(fu.get("origin_event_id") or "", {})
        gmail = origin.get("source") == "gmail" or fu.get("source") == "gmail"
        sender = _addresses(origin.get("from"), "requester")
        thread = str(origin.get("thread_id") or "").strip()
        due = _epoch(fu.get("follow_up_at"))
        status = str(fu.get("status") or "open")
        create = {
            "title": _head(fu.get("summary"), 160) or "Follow-up", "ask": _head(fu.get("detail"), 600),
            "kind": "follow_up", "direction": "received" if gmail else "internal",
            "channel": "gmail" if gmail else "autonomy", "owes_next": "them", "due_at": due,
            "source": "follow_ups", "source_ref": fu["id"], "triage_state": "skipped",
        }
        keys = [f"fu:{fu['id']}"]
        extra = [f"gmail:thread:{thread}"] if thread else []
        p.ingest("follow_ups", uid, keys=keys, create=create, extra_keys=extra, parties=sender,
                 touch={"title": create["title"], "ask": create["ask"]},
                 event={"source": "follow_ups", "source_ref": f"{fu['id']}:made", "at": _epoch(fu.get("created_at")),
                        "kind": "received" if gmail else "note", "channel": create["channel"],
                        "actor_kind": "party" if sender else "system",
                        "actor": sender[0]["display"] if sender else "Autonomous work",
                        "summary": f"Tracked: {_head(fu.get('summary'), 200)}"})
        changed = _epoch(fu.get("updated_at"))
        if status == "open":
            # Due or not yet is a state of its own (and so is each new due time).
            is_due = due is not None and due <= now
            state = f"open:{'due' if is_due else 'later'}:{fu.get('follow_up_at') or ''}"
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(due)) if due else ""
            event_at = due if is_due else changed
            patch = {"status": "open", "owes_next": "me" if is_due else "them", "due_at": due,
                     "flags": {"stale": False}}
            summary = "Due now" if is_due else (f"Due {when}" if when else "Waiting")
        elif status == "stale":
            state, event_at, summary = "stale", changed, "Gone stale"
            patch = {"status": "open", "flags": {"stale": True}}
        elif status == "done":
            state, event_at, summary = "done", changed, "Marked done"
            patch = {"status": "done", "owes_next": "nobody", "result": "success", "closed_by": "source"}
        else:
            state, event_at, summary = status, changed, "Dismissed"
            patch = {"status": "dismissed", "owes_next": "nobody", "closed_by": "source"}
        ref = p.store.next_state_ref("follow_ups", fu["id"], state)
        if ref:
            p.ingest("follow_ups", uid, keys=keys, create=create, patch=patch,
                     event={"source": "follow_ups", "source_ref": ref, "at": event_at, "kind": "status",
                            "channel": create["channel"], "summary": summary,
                            "refs": {"follow_up_id": fu["id"], "state": state}})
    if rows:
        p.save_cursor("follow_ups", max(str(r.get("updated_at") or "") for r in rows) or since)
    elif not since:
        p.save_cursor("follow_ups", _iso(started - _OVERLAP_S))


# ── Autonomous Work (autonomy: autonomous_actions) ────────────────────


_UNSETTLED = ("awaiting_approval", "queued", "dispatched", "undone")


def _action_move(row: dict, agent_name: str) -> tuple[str, dict, float | None, str]:
    """``(event kind, patch, when, summary)`` for an action's status now."""
    status = str(row.get("status") or "")
    note = _head(row.get("outcome_note"), 300)
    if status == "awaiting_approval":
        # Waiting since it was made — or since now, held back after a dispatch.
        return "proposal", {"status": "open", "owes_next": "me"}, \
            None if row.get("dispatched_at") else _epoch(row.get("created_at")), "Waiting for your approval"
    if status == "queued":
        return "status", {"status": "open", "owes_next": "agent"}, None, \
            "Approved, waiting for an agent" + (f": {note}" if note else "")
    if status == "dispatched":
        return "approval", {"status": "open", "owes_next": "agent"}, _epoch(row.get("dispatched_at")), \
            f"Handed to {agent_name}" if agent_name else "Handed to an agent"
    if status == "done":
        if row.get("outcome") == "fail":
            return "status", {"status": "failed", "result": "failed", "closed_by": "source", "result_text": note}, \
                _epoch(row.get("completed_at")), "Failed" + (f": {note}" if note else "")
        return "status", {"status": "done", "owes_next": "nobody", "evidence": "claimed", "closed_by": "source",
                          "closed_by_party": agent_name,
                          "result": "success" if row.get("outcome") == "success" else "unknown",
                          "result_text": note}, _epoch(row.get("completed_at")), note or "Done"
    if status == "rejected":
        return "rejection", {"status": "dismissed", "owes_next": "nobody", "closed_by": "user"}, \
            _epoch(row.get("completed_at")), "Rejected"
    if status == "expired":
        return "status", {"status": "dismissed", "owes_next": "nobody", "closed_by": "source"}, \
            _epoch(row.get("completed_at")), "Expired: no approval in time"
    return "status", {"status": "dismissed", "owes_next": "nobody", "closed_by": "user"}, None, "Undone"


def _autonomy(p: _Pass) -> None:
    from captain_claw.flight_deck import autonomy

    path = _store_path(autonomy)
    mine, mine_args = p.mine("user_id")
    cur = _json(p.cursor("autonomy"), {})
    after, since = int(cur.get("rowid") or 0), str(cur.get("at") or "")
    started = _iso(time.time() - _OVERLAP_S)
    # New rows by rowid; rows not settled yet every pass; rows that settled
    # since the last pass by when (no updated_at to go by).
    rows = _rows(path, "SELECT rowid AS row_no, * FROM autonomous_actions WHERE status != 'candidate' "
                       f"AND (rowid > ? OR status IN ({','.join('?' * len(_UNSETTLED))}) OR completed_at >= ?)"
                       f"{mine} ORDER BY rowid LIMIT ?", (after, *_UNSETTLED, since, *mine_args, _BATCH))
    for row in rows:
        uid = p.board_user(row.get("user_id"))
        if uid is None:
            continue
        payload = _json(row.get("payload"), {})
        if payload.get("handover"):
            continue  # the board's own, and it says so itself (taskboard_handover)
        key = f"autonomy:{row['id']}"
        slug = next((s for s in (str(row.get("ref_id") or ""), str(row.get("target") or "")) if s in p.agents), "")
        agent = p.agent(slug)
        # An action about a tracked follow-up is that follow-up's next step:
        # an event on its task, which keeps its own status.
        host = p.store.task_for_key(uid, key)
        fu_id = str(payload.get("follow_up_id") or "")
        if host is None and not fu_id and row.get("kind") == "track":
            continue  # a new open loop tracked: the follow-up it made is the task
        if host is None and fu_id:
            host = p.store.task_for_key(uid, f"fu:{fu_id}")
            if host:
                p.store.add_link(uid, key, host)
        board = row.get("source") == autonomy.TASKBOARD_SOURCE
        if host is None and board:
            # A Taskboard next step is on its own task (ref_id) — this pass may
            # read it before propose() linked it there — and never a task of its own.
            host = str(row.get("ref_id") or "")
            if not host or p.store.get_task(uid, host) is None:
                continue
            p.store.add_link(uid, key, host)
        # Merged into another task, a proposal is still a proposal: its own row hears from its action.
        own = p.store.own_row_for_key(uid, key) or (p.store.get_task(uid, host) if host else None)
        attached = host is not None and (own or {}).get("kind") != "proposal"
        create = {"title": _head(row.get("title"), 160) or "Proposal", "ask": _head(row.get("rationale"), 600),
                  **agent, "kind": "proposal", "direction": "internal", "channel": "autonomy",
                  "owes_next": "me", "source": "autonomy", "source_ref": row["id"], "triage_state": "skipped"}
        p.ingest("autonomy", uid, keys=[key], create=create,
                 touch=None if attached or not slug else agent,
                 event={"source": "autonomy", "source_ref": f"{row['id']}:made", "at": _epoch(row.get("created_at")),
                        "kind": "proposal", "channel": "autonomy", "actor_kind": "agent" if not board else "system",
                        "actor": "Taskboard" if board else "Autonomous work",
                        "summary": f"Proposed: {_head(row.get('title'), 200)}",
                        "refs": {"autonomy_id": row["id"], "kind": row.get("kind")}})
        status = str(row.get("status") or "")
        # The judge may settle a done action's outcome after it is done.
        state = f"{status}:{row['outcome']}" if status == "done" and row.get("outcome") else status
        ref = p.store.next_state_ref("autonomy", row["id"], state)
        if not ref:
            continue
        kind, patch, at, summary = _action_move(row, agent["agent_name"] if slug else "")
        p.ingest("autonomy", uid, keys=[key], create=create, patch=None if attached else patch,
                 event={"source": "autonomy", "source_ref": ref, "at": at or time.time(), "kind": kind,
                        "channel": "autonomy", "actor_kind": "me" if status in ("rejected", "undone") else "system",
                        "summary": summary,
                        "outcome": {"fail": "error", "success": "ok"}.get(str(row.get("outcome") or ""), ""),
                        "refs": {"autonomy_id": row["id"], "state": state}})
    p.save_cursor("autonomy", json.dumps({"rowid": max([after, *(int(r["row_no"]) for r in rows)]),
                                          "at": started}))


# ── Flight Deck's own DB (flight-deck.db): auto-replies, sends, links ─


def _fd_db_path() -> Path:
    return taskboard_store.data_dir() / "flight-deck.db"


_AUTOREPLY_STATES = ("held", "approved", "sending", "sent", "unknown", "failed", "dismissed")


def _autoreply_move(row: dict) -> tuple[str, dict, str, str]:
    """``(event kind, patch, summary, evidence)`` for an auto-reply's state."""
    state = str(row.get("state") or "")
    topic = str(row.get("topic_name") or "")
    reason = _head(row.get("detail") or row.get("reason"), 200)
    if state == "held":
        return "proposal", {"status": "open", "owes_next": "me"}, \
            "Reply written and held for your approval" + (f" ({topic})" if topic else ""), ""
    if state in ("approved", "sending"):
        # approved_by is set only when the owner approved a held reply.
        return "approval", {"status": "open", "owes_next": "agent"}, \
            "Reply approved" + (" by you" if row.get("approved_by") else "") + ", sending", ""
    if state == "sent":
        # Not the reply itself: the ledger clears that after 14 days, and the
        # board would keep it for good.
        text = f"Auto-replied ({topic})" if topic else "Auto-replied"
        return "sent", {"status": "done", "owes_next": "nobody", "evidence": "verified", "result": "success",
                        "closed_by": "source", "closed_by_party": str(row.get("agent_name") or ""),
                        "result_text": text}, "Reply sent", "recorded"
    if state == "unknown":
        return "sent", {"status": "done", "owes_next": "nobody", "evidence": "claimed", "result": "unknown",
                        "closed_by": "source"}, "The reply may have gone out (Gmail didn't confirm)", "said"
    if state == "failed":
        return "status", {"status": "failed", "result": "failed", "closed_by": "source", "result_text": reason}, \
            "Auto-reply failed" + (f": {reason}" if reason else ""), ""
    return "status", {"status": "dismissed", "owes_next": "nobody", "closed_by": "source"}, "Held reply dismissed", ""


def _autoreplies(p: _Pass) -> None:
    mine, mine_args = p.mine("owner_id")
    since = p.cursor("gmail_autoreplies")
    # A failed row with no topic never got past matching (the fetch or the
    # classifier failed): an email nobody held or answered, not a task.
    rows = _rows(_fd_db_path(),
                 f"SELECT * FROM gmail_autoreplies WHERE updated_at >= ? AND state IN "
                 f"({','.join('?' * len(_AUTOREPLY_STATES))}) AND (state != 'failed' OR COALESCE(topic_id, '') != '')"
                 f"{mine} ORDER BY updated_at LIMIT ?",
                 (since, *_AUTOREPLY_STATES, *mine_args, _BATCH))
    for row in rows:
        uid = p.board_user(row.get("owner_id"))
        if uid is None:
            continue
        state = str(row.get("state") or "")
        if state == "dismissed" and not p.store.task_for_key(uid, f"autoreply:{row['id']}"):
            continue  # dismissed before it was ever on the board
        agent_ref = str(row.get("agent_ref") or "")
        slug = agent_ref.split(":")[1] if agent_ref.count(":") == 2 else ""
        thread = str(row.get("thread_id") or "").strip()
        key = f"gmail:thread:{thread}" if thread else f"gmail:msg:{row.get('inbound_message_id') or row['id']}"
        sender = _addresses(row.get("from_addr"), "requester")
        subject = _head(row.get("subject"), 160)
        topic = str(row.get("topic_name") or "")
        create = {"title": subject or f"Email from {row.get('from_addr') or 'someone'}",
                  "ask": subject + (f" — topic: {topic}" if topic else ""),
                  **p.agent(slug, str(row.get("agent_name") or "")), "kind": "held_reply", "direction": "received",
                  "channel": "gmail", "owes_next": "agent", "source": "gmail_autoreplies", "source_ref": row["id"],
                  "triage_state": "skipped"}
        refs = {"autoreply_id": row["id"], "thread_id": thread}
        inbound = str(row.get("inbound_message_id") or "")
        extra = [f"autoreply:{row['id']}"] + ([f"gmail:msg:{inbound}"] if inbound else [])
        # v2 threads: an email whose thread is new to the board, but which an
        # open task already holds by its own id, goes on that task.
        from captain_claw.flight_deck import taskboard_threads

        taskboard_threads.attach_or_create(
            p.store, uid, ingest=lambda u, **kw: p.ingest("gmail_autoreplies", u, **kw), agents=p.agents,
            attach=[f"gmail:msg:{inbound}"] if thread and inbound else [],
            keys=[key], create=create, parties=sender, extra_keys=extra,
            event={"source": "gmail_autoreplies", "source_ref": f"{row['id']}:received",
                   "at": _epoch(row.get("received_at")) or _epoch(row.get("created_at")), "kind": "received",
                   "channel": "gmail", "actor_kind": "party",
                   "actor": sender[0]["display"] if sender else str(row.get("from_addr") or ""),
                   "summary": f"Email: {subject}" if subject else "Email received", "refs": refs})
        # held → approved → (the send failed) → held again: each move counts.
        ref = p.store.next_state_ref("gmail_autoreplies", row["id"], state)
        if not ref:
            continue
        kind, patch, summary, evidence = _autoreply_move(row)
        p.ingest("gmail_autoreplies", uid, keys=[key], create=create, patch=patch,
                 event={"source": "gmail_autoreplies", "source_ref": ref,
                        "at": _epoch(row.get("updated_at")), "kind": kind, "channel": "gmail",
                        "actor_kind": "agent" if kind == "sent" else "system",
                        "actor": str(row.get("agent_name") or ""), "summary": summary, "evidence": evidence,
                        "outcome": {"sent": "ok", "failed": "error"}.get(state, ""),
                        "refs": {**refs, "state": state, "message_id": str(row.get("sent_message_id") or "")}})
    if rows:
        p.save_cursor("gmail_autoreplies", max(str(r.get("updated_at") or "") for r in rows) or since)


# gmail_send_routes.AUTOREPLY_ORIGINS: sends the auto-reply engine made.
_AUTOREPLY_ORIGINS = ("autoreply", "autoreply_approved")
# An agent's email not on the board yet waits this long for the tool-call
# record of the turn that sent it, which puts its thread on that turn's task.
_SEND_WAIT_S = 600.0


def _gmail_sends(p: _Pass) -> None:
    mine, mine_args = p.mine("owner_id")
    since = p.cursor("gmail_sends")
    rows = _rows(_fd_db_path(), f"SELECT * FROM gmail_sends WHERE created_at >= ?{mine} ORDER BY created_at LIMIT ?",
                 (since, *mine_args, _BATCH))
    now = time.time()
    put_off = ""  # the oldest row left for a later pass: the cursor stays on it
    for row in rows:
        uid = p.board_user(row.get("owner_id"))
        if uid is None or row.get("origin") in _AUTOREPLY_ORIGINS:
            continue  # an auto-reply's send: its own row says it was sent
        thread = str(row.get("thread_id") or "").strip()
        message = str(row.get("gmail_message_id") or "").strip()
        key = f"gmail:thread:{thread}" if thread else f"gmail:msg:{message or row['id']}"
        sent_at = _epoch(row.get("created_at"))
        if (thread and str(row.get("origin") or "agent") == "agent" and sent_at is not None
                and now - sent_at < _SEND_WAIT_S and not p.store.task_for_key(uid, key)):
            # Sent by an agent a moment ago: the chat that asked for it is on
            # the board once its tool call is read (with this thread) — not a
            # "sent" task of its own beside it.
            put_off = put_off or str(row.get("created_at") or "")
            continue
        to = str(row.get("to_addrs") or "")
        parties = _addresses(to, "recipient") + _addresses(row.get("cc_addrs"), "cc")
        sent = row.get("status") == "sent"
        label = str(row.get("agent") or "")
        agent = p.agent(label) if label in p.agents else p.agent("", label)
        subject = _head(row.get("subject"), 160)
        names = ", ".join(x["display"] for x in parties if x["role"] == "recipient") or to
        event = {"source": "gmail_sends", "source_ref": str(row["id"]), "at": _epoch(row.get("created_at")),
                 "kind": "sent", "channel": "gmail", "actor_kind": "agent", "actor": agent["agent_name"],
                 "summary": (f"Email sent to {names}" if sent else f"Email to {names} may have gone out")
                 + (f": {subject}" if subject else ""),
                 "outcome": "ok" if sent else "", "evidence": "recorded" if sent else "said",
                 "refs": {"message_id": message, "thread_id": thread, "origin": row.get("origin")}}
        extra = [f"gmail:msg:{message}"] if message and thread else []
        if p.store.task_for_key(uid, key):
            # On a thread the board already has: what was sent, as evidence.
            p.ingest("gmail_sends", uid, keys=[key], extra_keys=extra, parties=parties, event=event)
            continue
        # v2 threads: a reply to an email an open task holds (In-Reply-To) is
        # sent on that task, not a "sent" task of its own.
        from captain_claw.flight_deck import taskboard_threads

        answers = str(row.get("in_reply_to_message_id") or "").strip()
        taskboard_threads.attach_or_create(
            p.store, uid, ingest=lambda u, **kw: p.ingest("gmail_sends", u, **kw), agents=p.agents,
            attach=[f"gmail:msg:{answers}"] if answers else [],
            keys=[key], extra_keys=extra, parties=parties, event=event,
            create={"title": subject or f"Email to {names}", "ask": "", **agent, "kind": "sent",
                    "direction": "sent", "channel": "gmail", "owes_next": "nobody", "source": "gmail_sends",
                    "source_ref": str(row["id"]), "triage_state": "skipped"},
            patch={"status": "done", "owes_next": "nobody", "closed_by": "evidence",
                   "closed_by_party": agent["agent_name"], "evidence": "verified" if sent else "claimed",
                   "result": "success" if sent else "unknown", "result_text": f"Sent to {names}"})
    if rows:
        p.save_cursor("gmail_sends", put_off or max(str(r.get("created_at") or "") for r in rows) or since)


_LINK_WAITING_KEPT = 1000


def _links(p: _Pass) -> None:
    mine, mine_args = p.mine("l.user_id")
    cur = _json(p.cursor("agent_links"), {})
    after = int(cur.get("seq") or 0)
    waiting = [int(s) for s in cur.get("waiting") or []][-_LINK_WAITING_KEPT:]
    # New events by seq, and the ones that hadn't reached the system yet.
    rows = _rows(_fd_db_path(),
                 "SELECT e.*, l.user_id AS link_user, l.name AS link_name, l.agent_slug AS link_agent "
                 "FROM agent_link_events e JOIN agent_links l ON l.id = e.link_id "
                 f"WHERE (e.seq > ? OR e.seq IN ({','.join('?' * len(waiting)) or 'NULL'})){mine} "
                 "ORDER BY e.seq LIMIT ?", (after, *waiting, *mine_args, _BATCH))
    for row in rows:
        uid = p.board_user(row.get("link_user"))
        if uid is None:
            continue
        kind = str(row.get("kind") or "")
        name = str(row.get("link_name") or "") or "a linked system"
        pushed = kind == "push"
        # What was said, never the message itself: the link keeps its events
        # a week, and the board would keep a copy for good.
        text = {"push": f"Sent a message to {name}", "error": "The agent failed to answer"}.get(
            kind, f"Answered {name}")
        msg = f"push-{row['seq']}" if pushed or not row.get("re") else str(row["re"])
        key = f"link:{row['link_id']}:{msg}"
        agent = p.agent(str(row.get("link_agent") or ""))
        party = {"role": "recipient" if pushed else "requester", "kind": "link", "address": str(row["link_id"]),
                 "display": name, "verified": True}
        create = {"title": f"Message to {name}" if pushed else f"Message from {name}", "ask": "",
                  "result_text": text, **agent, "kind": "link", "direction": "sent" if pushed else "received",
                  "channel": "link", "owes_next": "agent", "source": "agent_links", "source_ref": str(row["seq"]),
                  "triage_state": "skipped"}
        refs = {"link_id": row["link_id"], "seq": row["seq"], "re": row.get("re") or ""}
        if kind == "error":
            patch = {"status": "failed", "result": "failed", "closed_by": "source", "result_text": text}
        else:
            patch = {"status": "open", "owes_next": "agent", "result_text": text}
        p.ingest("agent_links", uid, keys=[key], create=create, parties=[party], patch=patch,
                 event={"source": "agent_links", "source_ref": f"{row['seq']}:made", "at": _epoch(row.get("created_at")),
                        "kind": {"push": "sent", "error": "status"}.get(kind, "reply"), "channel": "link",
                        "actor_kind": "agent", "actor": agent["agent_name"],
                        "outcome": "error" if kind == "error" else "ok", "summary": text, "refs": refs})
        via = str(row.get("delivered") or "")
        if via and kind != "error":
            p.ingest("agent_links", uid, keys=[key], create=create,
                     patch={"status": "done", "owes_next": "nobody", "evidence": "verified", "result": "success",
                            "closed_by": "evidence", "closed_by_party": agent["agent_name"]},
                     event={"source": "agent_links", "source_ref": f"{row['seq']}:delivered",
                            "at": _epoch(row.get("created_at")), "kind": "delivery", "channel": "link",
                            "actor_kind": "system", "summary": f"Reached {name} ({via})", "outcome": "ok",
                            "evidence": "recorded", "refs": {**refs, "via": via}})
    if rows or waiting:
        # An error is final; a reply or push is read again until it is
        # delivered (or gone: events are kept a week).
        still = [int(r["seq"]) for r in rows if not r.get("delivered") and r.get("kind") != "error"]
        p.save_cursor("agent_links", json.dumps({"seq": max([after, *(int(r["seq"]) for r in rows)]),
                                                 "waiting": still[-_LINK_WAITING_KEPT:]}))


# ── The sweep ─────────────────────────────────────────────────────────


def _waiting_turns(p: _Pass) -> None:
    from captain_claw.flight_deck import taskboard_ingest

    taskboard_ingest.waiting_turns(p)


def _tool_calls(p: _Pass) -> None:
    from captain_claw.flight_deck import taskboard_ingest

    taskboard_ingest.tool_calls(p)


def _agent_todos(p: _Pass) -> None:
    from captain_claw.flight_deck import taskboard_backfill

    taskboard_backfill.agent_todos(p)


def _passive(key: str) -> Any:
    """v2 passive: the adapter of one of the user's own inboxes (taskboard_passive)."""
    def adapter(p: _Pass) -> None:
        from captain_claw.flight_deck import taskboard_passive

        taskboard_passive.sweep(p, key)

    return adapter


# In this order: an action finds the follow-up it is about, an automation turn
# the task its run or message made, a tool call its turn's task — and a send
# the thread an auto-reply or a turn's own send put on the board (the agent's
# email lands on the chat task that asked for it, not on a task of its own).
ADAPTERS = (
    ("fd_scheduler", _scheduler),
    ("follow_ups", _follow_ups),
    ("autonomy", _autonomy),
    ("gmail_autoreplies", _autoreplies),
    ("agent_links", _links),
    ("agent_todos", _agent_todos),
    ("agent_turns", _waiting_turns),
    ("tool_audit", _tool_calls),
    ("gmail_sends", _gmail_sends),
    # v2 passive, last: an email or a message an agent sent is on its own records by then.
    ("telegram_inbox", _passive("telegram_inbox")),
    ("whatsapp_web", _passive("whatsapp_web")),
    ("gmail_inbox", _passive("gmail")),
)


def prune_turns(store: TaskboardStore, now: float | None = None) -> int:
    """Forget the agents' turns older than 30 days (the tasks they made stay)."""
    n = store.prune_turns((time.time() if now is None else now) - TURNS_KEPT_S)
    if n:
        log.info("Taskboard turns pruned", turns=n)
    return n


def run_sweep(store: TaskboardStore | None = None, user_id: str | None = None) -> dict:
    """Read every source into the board — everyone's rows, or only *user_id*'s
    (their board asked). One sweep at a time. ``{new_events, problems}``."""
    store = store or taskboard_store.get_store()
    with _SWEEP_LOCK:
        p = _Pass(store, user_id)
        if user_id is None:
            _STATE["running"] = True
        problems: list[str] = []
        try:
            for name, adapter in ADAPTERS:
                try:
                    adapter(p)
                except sqlite3.OperationalError as exc:
                    if "no such table" in str(exc):
                        log.debug("Taskboard source not on this deck", source=name, error=str(exc))
                        continue
                    problems.append(name)
                    log.warning("Taskboard source failed", source=name, error=str(exc))
                except Exception as exc:
                    problems.append(name)
                    log.warning("Taskboard source failed", source=name, error=str(exc))
            if user_id is None:
                try:
                    prune_turns(store)
                except Exception as exc:
                    log.warning("Taskboard turns prune failed", error=str(exc))
        finally:
            now = time.time()
            labels = {**SOURCE_LABELS, **AGENT_LABELS, **PASSIVE_LABELS}  # v2 passive: its adapters' names
            message = ("Couldn't read " + ", ".join(labels[n] for n in problems)) if problems else ""
            if user_id is None:
                _STATE.update(running=False, last_at=now, message=message)
            else:
                _USER_SYNCED[user_id] = (now, message)
    return {"new_events": dict(p.new_events), "problems": problems, "message": message}


def user_synced_at(user_id: str) -> float | None:
    """When *user_id*'s own sync last finished."""
    mine = _USER_SYNCED.get(user_id)
    return mine[0] if mine else None


def sweep_state(user_id: str | None = None) -> dict:
    """``{running, last_at, message}`` — last_at the latest time this user's
    board was brought up to date (a full sweep or their own sync), message
    what that read couldn't."""
    last, message = _STATE["last_at"], str(_STATE["message"] or "")
    mine = _USER_SYNCED.get(user_id or "")
    if mine and (last is None or mine[0] > last):
        last, message = mine
    return {"running": bool(_STATE["running"]), "last_at": last, "message": message}


async def sweep_loop(stop: asyncio.Event) -> None:
    """A sweep 20 s after start, then every 120 s, until *stop*."""
    delay = _FIRST_SWEEP_S
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
            return
        except TimeoutError:
            pass
        delay = SWEEP_EVERY_S
        try:
            await asyncio.to_thread(run_sweep)
        except Exception as exc:
            log.warning("Taskboard sweep failed", error=str(exc))
