"""Taskboard store — ``taskboard.db``: every task a user's agents received or sent.

One row per task in ``tasks``, found again through ``task_links`` (a key per
record that names it: ``sched:job:<id>``, ``autonomy:<id>``, ``fu:<id>``,
``gmail:thread:<id>``…) and told through ``task_events`` — one row per thing
that happened, ``UNIQUE(source, source_ref)`` so feeding the same record twice
adds nothing. The whole schema of docs/taskboard-plan.md is created here,
including what later versions only fill (decisions, rules, turns, parties).

Strictly per user: every read and every write names the board user
(``user_id``) and touches only that user's rows; there is no owner or admin
view. A user's correction (Done, Rename, Snooze…) writes a ``correction``
event and locks the field it touched (``user_locks``), and no machine pass
writes a locked field again — only Undo lifts the lock (and a scheduled
task's next run the status's: the user's word was about the run before).

Search is accent-folded on both sides (``outbound_guard._fold``): the folded
haystack is kept in ``search_text`` and matched with LIKE, every term ANDed,
so "duric" finds "Đurić". Facets are counted under the other filters, as in
the File library. Lanes (Needs me, Waiting on others, Agents working, Done)
are computed at read time, never stored.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from captain_claw.outbound_guard import _fold

# Field vocabularies (docs/taskboard-plan.md, "Data model").
KINDS = ("request", "sent", "scheduled", "proposal", "follow_up", "held_reply", "link", "todo", "message")
DIRECTIONS = ("received", "sent", "internal")
STATUSES = ("open", "done", "failed", "dismissed", "not_task", "maybe")
OWES = ("me", "agent", "them", "nobody", "unclear")
EVIDENCE = ("verified", "claimed", "none")
RESULTS = ("success", "partial", "failed", "unknown", "")
CLOSED = ("done", "failed", "dismissed", "not_task")
VIEWS = ("needs_me", "waiting", "working", "done", "all")

STATUS_LABELS = {"open": "Open", "done": "Done", "failed": "Failed", "dismissed": "Dismissed",
                 "not_task": "Not a task", "maybe": "Maybe a task"}
KIND_LABELS = {"request": "Request", "sent": "Sent", "scheduled": "Scheduled task", "proposal": "Proposal",
               "follow_up": "Follow-up", "held_reply": "Held reply", "link": "Linked system",
               "todo": "To-do", "message": "Message"}
DIRECTION_LABELS = {"received": "Received", "sent": "Sent", "internal": "Internal"}
EVIDENCE_LABELS = {"verified": "Recorded", "claimed": "Agent says", "none": "No evidence"}
RESULT_LABELS = {"success": "Success", "partial": "Partly done", "failed": "Failed", "unknown": "Unknown",
                 "": "No result yet"}
OWES_LABELS = {"me": "Me", "agent": "The agent", "them": "Them", "nobody": "Nobody", "unclear": "Unclear"}
CHANNEL_LABELS = {
    "web": "Web chat", "whatsapp": "WhatsApp", "waweb": "WhatsApp Web", "telegram": "Telegram",
    "tg_inbox": "Telegram inbox", "gmail": "Gmail", "email": "Email", "device": "Device", "link": "Linked systems",
    "scheduled": "Scheduled tasks", "autonomy": "Autonomous work", "member": "Shared-agent members",
    "api": "API", "mcp": "MCP", "glasses": "Smart glasses", "messenger": "Messenger",
}

# Writing one of these is writing the status: a lock on "status" covers them.
_LOCK_GROUP = {"closed_at": "status", "closed_by": "status", "closed_by_party": "status", "result": "status"}
_CLIPS = {"title": 160, "ask": 600, "result_text": 600, "agent_name": 120, "closed_by_party": 120}
_JSON_FIELDS = {"channels", "user_locks", "flags", "next_steps"}
_PARTY_ORDER = {"requester": 0, "recipient": 1, "cc": 2, "doer": 3, "mentioned": 4}
_ROLES = tuple(_PARTY_ORDER)
_PAGE_MAX = 200
_SNOOZE_MAX_S = 366 * 86400

# Every table and its columns. A column missing from an older taskboard.db is
# added on open (PRAGMA table_info), so a later version only lists its own.
_TABLES: dict[str, list[tuple[str, str]]] = {
    "tasks": [
        ("id", "TEXT PRIMARY KEY"),
        ("user_id", "TEXT NOT NULL"),
        ("agent_owner_id", "TEXT NOT NULL DEFAULT ''"),
        ("agent_slug", "TEXT NOT NULL DEFAULT ''"),
        ("agent_ref", "TEXT NOT NULL DEFAULT ''"),
        ("agent_name", "TEXT NOT NULL DEFAULT ''"),
        ("title", "TEXT NOT NULL DEFAULT ''"),
        ("ask", "TEXT NOT NULL DEFAULT ''"),
        ("result_text", "TEXT NOT NULL DEFAULT ''"),
        ("kind", "TEXT NOT NULL DEFAULT 'request'"),
        ("direction", "TEXT NOT NULL DEFAULT 'received'"),
        ("channel", "TEXT NOT NULL DEFAULT ''"),
        ("channels", "TEXT NOT NULL DEFAULT '[]'"),
        ("status", "TEXT NOT NULL DEFAULT 'open'"),
        ("owes_next", "TEXT NOT NULL DEFAULT 'unclear'"),
        ("evidence", "TEXT NOT NULL DEFAULT 'none'"),
        ("result", "TEXT NOT NULL DEFAULT ''"),
        ("urgency", "INTEGER NOT NULL DEFAULT 0"),
        ("due_at", "REAL"),
        ("snooze_until", "REAL"),
        ("created_at", "REAL NOT NULL DEFAULT 0"),
        ("updated_at", "REAL NOT NULL DEFAULT 0"),
        ("last_event_at", "REAL"),
        ("closed_at", "REAL"),
        ("closed_by", "TEXT NOT NULL DEFAULT ''"),
        ("closed_by_party", "TEXT NOT NULL DEFAULT ''"),
        ("fingerprint", "TEXT NOT NULL DEFAULT ''"),
        ("extracted_fp", "TEXT NOT NULL DEFAULT ''"),
        ("decided_fp", "TEXT NOT NULL DEFAULT ''"),
        ("triage_state", "TEXT NOT NULL DEFAULT 'pending'"),
        ("triage_error", "TEXT NOT NULL DEFAULT ''"),
        ("triage_after", "REAL"),
        ("user_locks", "TEXT NOT NULL DEFAULT '{}'"),
        ("flags", "TEXT NOT NULL DEFAULT '{}'"),
        ("private", "INTEGER NOT NULL DEFAULT 0"),
        ("merged_into", "TEXT NOT NULL DEFAULT ''"),
        ("source", "TEXT NOT NULL DEFAULT ''"),
        ("source_ref", "TEXT NOT NULL DEFAULT ''"),
        ("next_steps", "TEXT NOT NULL DEFAULT '[]'"),
        ("retention_until", "REAL"),
        ("search_text", "TEXT NOT NULL DEFAULT ''"),
    ],
    "task_links": [
        ("user_id", "TEXT NOT NULL"),
        ("key", "TEXT NOT NULL"),
        ("task_id", "TEXT NOT NULL"),
        ("strength", "TEXT NOT NULL DEFAULT 'strong'"),
        ("created_at", "REAL NOT NULL DEFAULT 0"),
    ],
    "task_events": [
        ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("task_id", "TEXT NOT NULL"),
        ("user_id", "TEXT NOT NULL"),
        ("at", "REAL NOT NULL DEFAULT 0"),
        ("kind", "TEXT NOT NULL DEFAULT 'note'"),
        ("channel", "TEXT NOT NULL DEFAULT ''"),
        ("actor_kind", "TEXT NOT NULL DEFAULT 'system'"),
        ("actor", "TEXT NOT NULL DEFAULT ''"),
        ("summary", "TEXT NOT NULL DEFAULT ''"),
        ("source", "TEXT NOT NULL"),
        ("source_ref", "TEXT NOT NULL"),
        ("outcome", "TEXT NOT NULL DEFAULT ''"),
        ("refs", "TEXT NOT NULL DEFAULT '{}'"),
        ("evidence", "TEXT NOT NULL DEFAULT ''"),
        ("private", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "task_decisions": [
        ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("task_id", "TEXT NOT NULL"),
        ("user_id", "TEXT NOT NULL"),
        ("at", "REAL NOT NULL DEFAULT 0"),
        ("judge", "TEXT NOT NULL DEFAULT ''"),
        ("qset", "TEXT NOT NULL DEFAULT ''"),
        ("state_hash", "TEXT NOT NULL DEFAULT ''"),
        ("answers", "TEXT NOT NULL DEFAULT '{}'"),
        ("applied", "TEXT NOT NULL DEFAULT ''"),
        ("why", "TEXT NOT NULL DEFAULT ''"),
    ],
    "parties": [
        ("id", "TEXT PRIMARY KEY"),
        ("user_id", "TEXT NOT NULL"),
        ("display_name", "TEXT NOT NULL DEFAULT ''"),
        ("folded_name", "TEXT NOT NULL DEFAULT ''"),
        ("kind", "TEXT NOT NULL DEFAULT 'person'"),
    ],
    "party_handles": [
        ("user_id", "TEXT NOT NULL"),
        ("kind", "TEXT NOT NULL"),
        ("value", "TEXT NOT NULL"),
        ("party_id", "TEXT NOT NULL"),
    ],
    "task_parties": [
        ("task_id", "TEXT NOT NULL"),
        ("user_id", "TEXT NOT NULL"),
        ("role", "TEXT NOT NULL"),
        ("kind", "TEXT NOT NULL"),
        ("address", "TEXT NOT NULL"),
        ("display", "TEXT NOT NULL DEFAULT ''"),
        ("party_id", "TEXT NOT NULL DEFAULT ''"),
        ("verified", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "task_rules": [
        ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
        ("user_id", "TEXT NOT NULL"),
        ("match", "TEXT NOT NULL DEFAULT '{}'"),
        ("effect", "TEXT NOT NULL DEFAULT 'not_task'"),
        ("created_from", "TEXT NOT NULL DEFAULT ''"),
        ("hits", "INTEGER NOT NULL DEFAULT 0"),
        ("created_at", "REAL NOT NULL DEFAULT 0"),
    ],
    "turns": [
        ("turn_id", "TEXT PRIMARY KEY"),
        ("agent_owner_id", "TEXT NOT NULL DEFAULT ''"),
        ("user_id", "TEXT NOT NULL"),
        ("agent_slug", "TEXT NOT NULL DEFAULT ''"),
        ("session_id", "TEXT NOT NULL DEFAULT ''"),
        ("origin", "TEXT NOT NULL DEFAULT ''"),
        ("origin_detail", "TEXT NOT NULL DEFAULT ''"),
        ("channel", "TEXT NOT NULL DEFAULT ''"),
        ("source", "TEXT NOT NULL DEFAULT ''"),
        ("source_label", "TEXT NOT NULL DEFAULT ''"),
        ("speaker_label", "TEXT NOT NULL DEFAULT ''"),
        ("ext_refs", "TEXT NOT NULL DEFAULT '{}'"),
        ("text_head", "TEXT NOT NULL DEFAULT ''"),
        ("reply_head", "TEXT NOT NULL DEFAULT ''"),
        ("deliverables", "TEXT NOT NULL DEFAULT '[]'"),
        ("outcome", "TEXT NOT NULL DEFAULT ''"),
        ("started_at", "REAL"),
        ("ended_at", "REAL"),
        ("private", "INTEGER NOT NULL DEFAULT 0"),
        ("task_id", "TEXT NOT NULL DEFAULT ''"),
        # v1: what the gate read, what it made of the turn, and the record
        # numbers (``boot:seq``) each phase came under.
        ("session_name", "TEXT NOT NULL DEFAULT ''"),
        ("speaker_id", "TEXT NOT NULL DEFAULT ''"),
        ("member", "INTEGER NOT NULL DEFAULT 0"),
        ("opener", "INTEGER NOT NULL DEFAULT 0"),
        ("human", "INTEGER NOT NULL DEFAULT 0"),
        ("literal", "TEXT NOT NULL DEFAULT ''"),
        ("automation", "TEXT NOT NULL DEFAULT '{}'"),
        ("tool_calls", "INTEGER NOT NULL DEFAULT 0"),
        ("failed_tool_calls", "INTEGER NOT NULL DEFAULT 0"),
        ("gate", "TEXT NOT NULL DEFAULT ''"),
        ("task_key", "TEXT NOT NULL DEFAULT ''"),
        ("open_ref", "TEXT NOT NULL DEFAULT ''"),
        ("close_ref", "TEXT NOT NULL DEFAULT ''"),
        ("received_at", "REAL NOT NULL DEFAULT 0"),
        # v1.1: the board user's own conversation with the agent (their chat,
        # their devices, their own WhatsApp / Telegram) — the ingest's verdict.
        ("own_chat", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "source_cursors": [
        ("source", "TEXT PRIMARY KEY"),
        ("cursor", "TEXT NOT NULL DEFAULT ''"),
        ("updated_at", "REAL NOT NULL DEFAULT 0"),
    ],
    "taskboard_settings": [
        ("user_id", "TEXT PRIMARY KEY"),
        ("passive", "TEXT NOT NULL DEFAULT '{}'"),
        ("auto_close", "INTEGER NOT NULL DEFAULT 1"),
        ("jev_per_day", "INTEGER NOT NULL DEFAULT 200"),
        ("extract_per_day", "INTEGER NOT NULL DEFAULT 100"),
        ("counters", "TEXT NOT NULL DEFAULT '{}'"),
        ("backfill_done_at", "REAL"),
        ("updated_at", "REAL NOT NULL DEFAULT 0"),
    ],
}
# Table constraints, after the columns.
_TABLE_TAIL = {
    "task_links": "PRIMARY KEY (user_id, key)",
    "task_events": "UNIQUE (source, source_ref)",
    "party_handles": "PRIMARY KEY (user_id, kind, value)",
    "task_parties": "PRIMARY KEY (task_id, role, kind, address)",
}
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_tasks_lane ON tasks(user_id, status, last_event_at)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_owes ON tasks(user_id, owes_next)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_agent ON tasks(user_id, agent_slug)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_agent_owner ON tasks(agent_owner_id, agent_slug)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_merged ON tasks(merged_into)",
    "CREATE INDEX IF NOT EXISTS idx_links_task ON task_links(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_task ON task_events(task_id, at)",
    "CREATE INDEX IF NOT EXISTS idx_decisions_task ON task_decisions(task_id, at)",
    "CREATE INDEX IF NOT EXISTS idx_parties_user ON parties(user_id, folded_name)",
    "CREATE INDEX IF NOT EXISTS idx_task_parties_user ON task_parties(user_id, kind, address)",
    "CREATE INDEX IF NOT EXISTS idx_rules_user ON task_rules(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_turns_task ON turns(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_turns_received ON turns(received_at)",
    "CREATE INDEX IF NOT EXISTS idx_turns_gate ON turns(gate, received_at)",
    "CREATE INDEX IF NOT EXISTS idx_turns_open_ref ON turns(agent_slug, open_ref)",
    "CREATE INDEX IF NOT EXISTS idx_turns_close_ref ON turns(agent_slug, close_ref)",
)
# ── v2 passive (taskboard_passive): what an opt-in inbox gave a task ──
# The links and parties an inbox's messages put on a task (``what`` link:
# the key; party: ``[role, kind, address]``), so switching the inbox off — or
# its text reaching the end of its 30 days — takes exactly them away. Its
# events are known by their own ``source``, the tasks it made by theirs.
_TABLES["passive_marks"] = [
    ("user_id", "TEXT NOT NULL"),
    ("source", "TEXT NOT NULL"),
    ("task_id", "TEXT NOT NULL"),
    ("what", "TEXT NOT NULL"),
    ("ref", "TEXT NOT NULL"),
]
_TABLE_TAIL["passive_marks"] = "PRIMARY KEY (user_id, source, task_id, what, ref)"
_INDEXES = (*_INDEXES, "CREATE INDEX IF NOT EXISTS idx_passive_marks_task ON passive_marks(task_id)")
# Flags that repeat what a task's text said (a model read them from it).
_PASSIVE_TEXT_FLAGS = ("expected_deliverable", "blocker", "why", "answered", "answered_on")
# The fields a correction's ``before`` may hold an inbox's text in.
_PASSIVE_TEXT_FIELDS = ("title", "ask", "result_text", "next_steps")
# The opt-in inboxes' event and task sources (taskboard_passive.SOURCE).
PASSIVE_SOURCES = ("telegram_inbox", "whatsapp_web", "gmail_inbox")
_TASK_COLUMNS = {name for name, _ in _TABLES["tasks"]}
_WRITABLE = _TASK_COLUMNS - {"id", "user_id", "search_text", "fingerprint", "user_locks"}
# A turn record writes these (the gate's verdict and the record numbers are
# the store's own); JSON and yes/no columns are read back as such.
_TURN_FIELDS = {name for name, _ in _TABLES["turns"]} - {
    "turn_id", "gate", "task_id", "task_key", "open_ref", "close_ref", "received_at"}
_TURN_JSON = {"origin_detail": {}, "ext_refs": {}, "deliverables": [], "automation": {}}
_TURN_FLAGS = ("member", "opener", "human", "private", "own_chat")
TURN_DROPPED = "skipped:dropped"

# What a facet counts, by the column it reads.
_FACET_COLUMNS = {"status": "status", "channel": "channel", "agent": "agent_slug", "direction": "direction",
                  "kind": "kind", "evidence": "evidence", "result": "result"}
FACETS = (*_FACET_COLUMNS, "party")
# A party row's People value (v2): the person it is, else its address.
_PARTY_FACET = "CASE WHEN p.party_id != '' THEN 'party:' || p.party_id ELSE p.kind || ':' || p.address END"
_SORTS = {
    "recent": "COALESCE(t.last_event_at, t.created_at) DESC, t.id",
    "oldest": "t.created_at ASC, t.id",
    "due": "t.due_at IS NULL, t.due_at ASC, COALESCE(t.last_event_at, t.created_at) DESC",
    "urgency": "t.urgency DESC, COALESCE(t.last_event_at, t.created_at) DESC",
}


def _loads(raw: Any, default: Any) -> Any:
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw else default
    except ValueError:
        return default
    return value if isinstance(value, type(default)) else default


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _marks(values: Any) -> str:
    return ",".join("?" * len(values))


def fold_terms(q: str) -> list[str]:
    """A search box's words, folded as ``search_text`` is (at most 8)."""
    return [t for t in _fold(q or "").split() if t][:8]


def _turn_dict(row: sqlite3.Row) -> dict:
    turn = dict(row)
    for k, default in _TURN_JSON.items():
        turn[k] = _loads(turn.get(k), type(default)())
    for k in _TURN_FLAGS:
        turn[k] = bool(turn.get(k))
    return turn


class TaskboardStore:
    """``taskboard.db``: tasks, their links, events and parties, per board user."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            for table, columns in _TABLES.items():
                body = ", ".join(f"{name} {ddl}" for name, ddl in columns)
                tail = f", {_TABLE_TAIL[table]}" if table in _TABLE_TAIL else ""
                self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({body}{tail})")
                have = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
                for name, ddl in columns:
                    if name not in have:
                        # A board from an older version: the column it lacks.
                        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
            for ddl in _INDEXES:
                self._conn.execute(ddl)
            self._conn.commit()
            self._open_people()  # v2: the people behind the parties (threads section)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── Links ─────────────────────────────────────────────────────────

    def task_for_key(self, user_id: str, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT task_id FROM task_links WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
        return str(row[0]) if row else None

    def add_link(self, user_id: str, key: str, task_id: str, strength: str = "strong") -> bool:
        """Name *task_id* by *key* — unless another task already holds it (the
        first one keeps it). True when this call added it."""
        with self._lock:
            added = self._link(user_id, key, task_id, strength)
            self._conn.commit()
        return added

    def _link(self, user_id: str, key: str, task_id: str, strength: str) -> bool:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO task_links (user_id, key, task_id, strength, created_at) VALUES (?,?,?,?,?)",
            (user_id, key, task_id, strength, time.time()))
        return (cur.rowcount or 0) == 1

    def links(self, user_id: str, task_ids: list[str]) -> dict[str, list[dict]]:
        """Each task's keys, oldest first."""
        out: dict[str, list[dict]] = {t: [] for t in task_ids}
        if not task_ids:
            return out
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM task_links WHERE user_id = ? AND task_id IN ({_marks(task_ids)}) "
                "ORDER BY created_at, key", (user_id, *task_ids)).fetchall()
        for r in rows:
            out.setdefault(r["task_id"], []).append(dict(r))
        return out

    def tasks_for_keys(self, user_id: str, keys: list[str]) -> dict[str, str]:
        """``{key: task_id}`` for the keys a task holds."""
        out: dict[str, str] = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                for r in self._conn.execute(
                        f"SELECT key, task_id FROM task_links WHERE user_id = ? AND key IN ({_marks(chunk)})",
                        (user_id, *chunk)):
                    out[r["key"]] = r["task_id"]
        return out

    # ── Ingest (machine writes) ───────────────────────────────────────

    def ingest(
        self, user_id: str, *, keys: list[str], event: dict, create: dict | None = None,
        patch: dict | None = None, touch: dict | None = None, extra_keys: tuple[str, ...] | list[str] = (),
        parties: tuple[dict, ...] | list[dict] = (), unlock: tuple[str, ...] = (),
    ) -> tuple[str | None, bool]:
        """One record from a source, in one step: the task the first known of
        *keys* names (else a new one from *create*, named by all *keys*; with no
        *create*, nothing), *extra_keys* added when free, *touch* (descriptive
        fields kept current) and *parties* written, then *event* added. Only a
        new event applies *patch* — the status move it stands for — so a record
        read again changes nothing. Locked fields are never written: what of
        *patch* a lock kept out is kept on the event (``refs.held``), so Undo
        can give the source its say back. A new event first lifts the user's
        locks on *unlock* (a scheduled task's next run: the user's Done covered
        the run before). ``(task_id, event_was_new)``; task_id None when
        nothing matched."""
        at = float(event.get("at") or time.time())
        with self._lock:
            task_id = next((t for t in (self._key_task(user_id, k) for k in keys) if t), None)
            if task_id is None:
                if create is None:
                    return None, False
                task_id = self._insert_task(user_id, create, at)
                for k in keys:
                    self._link(user_id, k, task_id, "strong")
            for k in extra_keys:
                self._link(user_id, k, task_id, "weak" if k.startswith("autoreply:") else "strong")
            changed = False
            if touch:
                changed |= self._machine_write(task_id, touch, at)
            if parties:
                changed |= self._add_parties(user_id, task_id, parties) > 0
            locks = _loads((self._row(task_id) or {}).get("user_locks"), {})
            kept = {k: v for k, v in locks.items() if k not in unlock}
            held = _held(patch or {}, kept)
            if held:
                event = {**event, "refs": {**(event.get("refs") or {}), "held": held}}
            new = self._add_event(user_id, task_id, event)
            if new and kept != locks:
                self._write(task_id, {"user_locks": kept})
            if new and patch:
                self._machine_write(task_id, patch, at)
            if new or changed:
                self._refresh(task_id)
            self._conn.commit()
        return task_id, new

    def _key_task(self, user_id: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT task_id FROM task_links WHERE user_id = ? AND key = ?", (user_id, key)).fetchone()
        return str(row[0]) if row else None

    def _insert_task(self, user_id: str, fields: dict, at: float) -> str:
        task_id = "tb_" + secrets.token_hex(8)
        row = {k: v for k, v in fields.items() if k in _WRITABLE}
        for k, n in _CLIPS.items():
            if k in row:
                row[k] = str(row[k] or "")[:n]
        for k in _JSON_FIELDS & set(row):
            row[k] = json.dumps(row[k])
        if "channels" not in row and row.get("channel"):
            row["channels"] = json.dumps([row["channel"]])
        now = time.time()
        row.setdefault("created_at", at)
        row.update(id=task_id, user_id=user_id, updated_at=now)
        self._conn.execute(
            f"INSERT INTO tasks ({', '.join(row)}) VALUES ({_marks(row)})", tuple(row.values()))
        return task_id

    def _row(self, task_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def _machine_write(self, task_id: str, fields: dict, at: float) -> bool:
        """Write *fields* the way a source may: never a field the user locked
        (closing fields count as the status), ``claimed`` never over
        ``verified``, flags merged key by key. A status that closes the task
        stamps when; one that opens it clears who closed it. A status move
        (opening it, or a source closing it again) answers what came in after
        it was done: ``new_since_done`` goes, unless the same write sets it."""
        row = self._row(task_id)
        if row is None:
            return False
        locks = _loads(row["user_locks"], {})
        sets: dict[str, Any] = {}
        for field, value in fields.items():
            if field not in _WRITABLE or _LOCK_GROUP.get(field, field) in locks:
                continue
            if field == "flags":
                flags = _loads(row["flags"], {})
                merged = {**flags, **{k: v for k, v in (value or {}).items() if k not in locks}}
                if merged != flags:
                    sets["flags"] = json.dumps(merged)
                continue
            if field == "evidence" and value == "claimed" and row["evidence"] == "verified":
                continue
            if field in _CLIPS:
                value = str(value or "")[:_CLIPS[field]]
            if field in _JSON_FIELDS:
                value = json.dumps(value)
            sets[field] = value
        status = sets.get("status")
        if status in CLOSED:
            sets.setdefault("closed_at", at)
        elif status is not None and status != row["status"]:
            sets.update(closed_at=None, closed_by="", closed_by_party="")
            sets.setdefault("result", "")
        flags = _loads(sets.get("flags", row["flags"]), {})
        if (status is not None and flags.get("new_since_done") and "new_since_done" not in locks
                and "new_since_done" not in (fields.get("flags") or {})):
            sets["flags"] = json.dumps({**flags, "new_since_done": False})
        sets = {k: v for k, v in sets.items() if row.get(k) != v}
        if not sets:
            return False
        sets["updated_at"] = time.time()
        self._conn.execute(
            f"UPDATE tasks SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?", (*sets.values(), task_id))
        return True

    def _add_event(self, user_id: str, task_id: str, ev: dict) -> bool:
        at = float(ev.get("at") or time.time())
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO task_events (task_id, user_id, at, kind, channel, actor_kind, actor, summary, "
            "source, source_ref, outcome, refs, evidence, private) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, user_id, at, str(ev.get("kind") or "note"), str(ev.get("channel") or ""),
             str(ev.get("actor_kind") or "system"), str(ev.get("actor") or "")[:120],
             " ".join(str(ev.get("summary") or "").split())[:500], str(ev["source"]), str(ev["source_ref"]),
             str(ev.get("outcome") or ""), json.dumps(ev.get("refs") or {}), str(ev.get("evidence") or ""),
             int(bool(ev.get("private")))))
        if (cur.rowcount or 0) != 1:
            return False
        row = self._row(task_id) or {}
        n = self._conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)).fetchone()[0]
        channels = _loads(row.get("channels"), [])
        if ev.get("channel") and ev["channel"] not in channels:
            channels.append(str(ev["channel"]))
        self._conn.execute(
            "UPDATE tasks SET last_event_at = MAX(COALESCE(last_event_at, 0), ?), fingerprint = ?, channels = ? "
            "WHERE id = ?", (at, f"{cur.lastrowid}:{n}", json.dumps(channels), task_id))
        if ev.get("evidence") == "recorded":
            self._machine_write(task_id, {"evidence": "verified"}, at)
        return True

    def _add_parties(self, user_id: str, task_id: str, parties: list[dict] | tuple[dict, ...]) -> int:
        added = 0
        for p in parties:
            address = str(p.get("address") or "").strip()
            role = str(p.get("role") or "mentioned")
            if not address or role not in _ROLES:
                continue
            row = {"role": role, "kind": str(p.get("kind") or "email"), "address": address[:320],
                   "display": str(p.get("display") or "")[:160], "verified": int(bool(p.get("verified")))}
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO task_parties (task_id, user_id, role, kind, address, display, verified) "
                "VALUES (?,?,?,?,?,?,?)", (task_id, user_id, *row.values()))
            if (cur.rowcount or 0) == 1:
                added += 1
                self._link_party(user_id, task_id, row)  # v2: the person behind it (threads section)
        return added

    def _refresh(self, task_id: str) -> None:
        """Fold the task's searchable text again: title, ask, result, agent,
        channels, kind and parties (v2: and the names of the people they are)."""
        row = self._row(task_id)
        if row is None:
            return
        parts = [row["title"], row["ask"], row["result_text"], row["agent_name"], row["agent_slug"],
                 KIND_LABELS.get(row["kind"], row["kind"])]
        for ch in _loads(row["channels"], []) or [row["channel"]]:
            parts.append(CHANNEL_LABELS.get(str(ch).partition(":")[0], str(ch)))
        for p in self._conn.execute(
                "SELECT tp.display, tp.address, COALESCE(pp.display_name, '') AS name FROM task_parties tp "
                "LEFT JOIN parties pp ON pp.id = tp.party_id AND pp.user_id = tp.user_id WHERE tp.task_id = ?",
                (task_id,)):
            parts.extend((p["display"], p["address"], p["name"]))
        text = _fold(" ".join(str(x or "") for x in parts))
        self._conn.execute("UPDATE tasks SET search_text = ? WHERE id = ?", (" ".join(text.split()), task_id))

    def update_fields(self, user_id: str, task_id: str, fields: dict) -> bool:
        """A machine write outside an event (lock-aware, as in :meth:`ingest`)."""
        with self._lock:
            if self._owned(user_id, task_id) is None:
                return False
            changed = self._machine_write(task_id, fields, time.time())
            if changed:
                self._refresh(task_id)
            self._conn.commit()
        return changed

    def next_state_ref(self, source: str, row_key: str, state: str) -> str | None:
        """The ``source_ref`` for a source row now in *state*:
        ``<row_key>:<state>:<n>`` — or None when the latest state the board
        has from that row (``refs.state`` of its events) is *state* already, so
        a row read again, or changed in nothing that matters, adds nothing,
        while one that comes back to an earlier state (a proposal held for
        approval again) is a new event."""
        lo = f"{row_key}:"
        with self._lock:
            rows = self._conn.execute(
                "SELECT refs FROM task_events WHERE source = ? AND source_ref >= ? AND source_ref < ? "
                "ORDER BY id DESC", (source, lo, lo + "￿")).fetchall()
        seen = [r for r in (_loads(r[0], {}) for r in rows) if "state" in r]
        if seen and seen[0]["state"] == state:
            return None
        return f"{row_key}:{state}:{len(seen)}"

    # ── Reads ─────────────────────────────────────────────────────────

    def _owned(self, user_id: str, task_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ? AND user_id = ?", (task_id, user_id)).fetchone()
        return dict(row) if row else None

    def get_task(self, user_id: str, task_id: str) -> dict | None:
        with self._lock:
            return self._owned(user_id, task_id)

    def events(self, user_id: str, task_id: str, limit: int = 500) -> list[dict]:
        """The task's latest *limit* events (a scheduled task gathers a few
        every run), oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM task_events WHERE user_id = ? AND task_id = ? ORDER BY at DESC, id DESC LIMIT ?",
                (user_id, task_id, limit)).fetchall()
        out = []
        for r in reversed(rows):
            ev = dict(r)
            ev["refs"] = _loads(ev["refs"], {})
            ev["private"] = bool(ev["private"])
            out.append(ev)
        return out

    def event_count(self, user_id: str, task_id: str) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE user_id = ? AND task_id = ?", (user_id, task_id)).fetchone()[0])

    def keyed_tasks(self, prefix: str, *, user_id: str | None = None, status: str | None = None) -> list[dict]:
        """``{user_id, key, task_id}`` for each key starting with *prefix* — on
        every board, or *user_id*'s — of tasks in *status* when given."""
        sql = ("SELECT l.user_id, l.key, l.task_id FROM task_links l JOIN tasks t ON t.id = l.task_id "
               "WHERE l.key >= ? AND l.key < ?")
        args: list[Any] = [prefix, prefix + "￿"]
        if user_id is not None:
            sql += " AND l.user_id = ?"
            args.append(user_id)
        if status is not None:
            sql += " AND t.status = ?"
            args.append(status)
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args)]

    def parties_for(self, user_id: str, task_ids: list[str]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {t: [] for t in task_ids}
        if not task_ids:
            return out
        with self._lock:
            # v2: ``name`` — the person the party is, when the board knows them.
            rows = self._conn.execute(
                "SELECT tp.*, COALESCE(pp.display_name, '') AS name FROM task_parties tp LEFT JOIN parties pp "
                f"ON pp.id = tp.party_id AND pp.user_id = tp.user_id WHERE tp.user_id = ? AND tp.task_id IN "
                f"({_marks(task_ids)})", (user_id, *task_ids)).fetchall()
        for r in sorted(rows, key=lambda r: (_PARTY_ORDER.get(r["role"], 9), r["display"] or r["address"])):
            out.setdefault(r["task_id"], []).append({
                "role": r["role"], "kind": r["kind"], "address": r["address"],
                "display": r["display"], "party_id": r["party_id"], "name": r["name"],
                "verified": bool(r["verified"])})
        return out

    def decisions(self, user_id: str, task_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM task_decisions WHERE user_id = ? AND task_id = ? ORDER BY at DESC, id DESC LIMIT ?",
                (user_id, task_id, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["answers"] = _loads(d["answers"], {})
            out.append(d)
        return out

    # ── Lanes, search, facets ─────────────────────────────────────────

    @staticmethod
    def _needs_me(now: float, proposal_tasks: list[str]) -> tuple[str, list]:
        """Needs me: open, not snoozed, and the user's move (they owe it, a
        proposal waits, it's probably done, it's overdue) — or failed and not
        acknowledged since it failed. Never NULL (it is negated for the other
        lanes), so a flag that isn't there reads as 0."""
        waiting = f" OR t.id IN ({_marks(proposal_tasks)})" if proposal_tasks else ""
        sql = ("((t.status = 'open' AND (t.snooze_until IS NULL OR t.snooze_until <= ?) AND ("
               "t.owes_next = 'me' OR COALESCE(json_extract(t.flags, '$.probably_done'), 0) = 1 "
               f"OR (t.due_at IS NOT NULL AND t.due_at < ?){waiting})) "
               "OR (t.status = 'failed' AND COALESCE(json_extract(t.flags, '$.failed_ack'), 0) "
               "< COALESCE(t.closed_at, 0)))")
        return sql, [now, now, *proposal_tasks]

    def _lane_sql(self, now: float, proposal_tasks: list[str]) -> tuple[str, list]:
        needs, args = self._needs_me(now, proposal_tasks)
        sql = (f"CASE WHEN {needs} THEN 'needs_me' "
               "WHEN t.status = 'open' AND t.owes_next = 'them' THEN 'waiting' "
               "WHEN t.status = 'open' AND t.owes_next = 'agent' THEN 'working' "
               "WHEN t.status = 'done' THEN 'done' "
               "WHEN t.status = 'open' AND t.snooze_until > ? THEN 'snoozed' "
               "ELSE 'other' END")
        return sql, [*args, now]

    def _view_sql(self, view: str, now: float, proposal_tasks: list[str]) -> tuple[str, list]:
        needs, args = self._needs_me(now, proposal_tasks)
        if view == "needs_me":
            return needs, args
        if view == "waiting":
            return f"(t.status = 'open' AND t.owes_next = 'them' AND NOT {needs})", args
        if view == "working":
            return f"(t.status = 'open' AND t.owes_next = 'agent' AND NOT {needs})", args
        if view == "done":
            return "t.status = 'done'", []
        return "", []

    def _where(
        self, user_id: str, *, view: str, terms: list[str], filters: dict[str, list[str]],
        date_from: float | None, date_to: float | None, now: float, proposal_tasks: list[str],
        skip: str = "",
    ) -> tuple[str, list]:
        """The WHERE for one listing — every filter but *skip* (the facet being
        counted, or "view" for the per-view counts)."""
        where, args = ["t.user_id = ?", "t.merged_into = ''"], [user_id]
        if skip != "view":
            cond, a = self._view_sql(view, now, proposal_tasks)
            if cond:
                where.append(cond)
                args += a
        for term in terms:
            where.append("t.search_text LIKE ? ESCAPE '\\'")
            args.append(_like(term))
        statuses = filters.get("status") or []
        if skip != "status":
            if statuses:
                where.append(f"t.status IN ({_marks(statuses)})")
                args += statuses
            elif view == "all" or skip == "view":
                # All is everything but what the user said is not a task —
                # unless they ask for those by the Status facet.
                where.append("t.status != 'not_task'")
        for dim, col in _FACET_COLUMNS.items():
            values = filters.get(dim) or []
            if dim == "status" or dim == skip or not values:
                continue
            where.append(f"t.{col} IN ({_marks(values)})")
            args += values
        parties = filters.get("party") or []
        if parties and skip != "party":
            # v2: a person by their facet id (``party:<id>``), an address by ``<kind>:<address>``.
            where.append(
                "EXISTS (SELECT 1 FROM task_parties p WHERE p.task_id = t.id AND p.user_id = t.user_id "
                f"AND ((p.kind || ':' || p.address) IN ({_marks(parties)}) "
                f"OR (p.party_id != '' AND 'party:' || p.party_id IN ({_marks(parties)}))))")
            args += parties + parties
        if date_from is not None:
            where.append("COALESCE(t.last_event_at, t.created_at) >= ?")
            args.append(float(date_from))
        if date_to is not None:
            where.append("COALESCE(t.last_event_at, t.created_at) <= ?")
            args.append(float(date_to))
        return " AND ".join(where), args

    def query(
        self, user_id: str, *, view: str = "needs_me", q: str = "", filters: dict[str, list[str]] | None = None,
        date_from: float | None = None, date_to: float | None = None, sort: str = "recent",
        limit: int = 50, offset: int = 0, now: float | None = None, proposal_tasks: list[str] | None = None,
        with_facets: bool = True,
    ) -> dict:
        """One page of the user's tasks in *view*, searched and filtered, with
        each facet's counts under the other filters and each view's count
        under all of them (v2: not with *with_facets* False — the export pages
        through rows only). Rows carry their computed ``lane``."""
        now = time.time() if now is None else now
        view = view if view in VIEWS else "needs_me"
        filters = {k: [v for v in vals if v is not None] for k, vals in (filters or {}).items() if vals}
        proposal_tasks = list(proposal_tasks or [])
        terms = fold_terms(q)
        common = dict(view=view, terms=terms, filters=filters, date_from=date_from, date_to=date_to,
                      now=now, proposal_tasks=proposal_tasks)
        limit = max(1, min(int(limit or 50), _PAGE_MAX))
        offset = max(0, int(offset or 0))
        lane, lane_args = self._lane_sql(now, proposal_tasks)
        with self._lock:
            where, args = self._where(user_id, **common)
            total = self._conn.execute(f"SELECT COUNT(*) FROM tasks t WHERE {where}", args).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT t.*, {lane} AS lane FROM tasks t WHERE {where} "
                f"ORDER BY {_SORTS.get(sort, _SORTS['recent'])} LIMIT ? OFFSET ?",
                (*lane_args, *args, limit, offset)).fetchall()
            if not with_facets:
                return {"rows": [dict(r) for r in rows], "total": int(total), "facets": {}, "counts": {}}
            facets: dict[str, list[tuple[str, str, int]]] = {}
            for dim, col in _FACET_COLUMNS.items():
                w, a = self._where(user_id, **common, skip=dim)
                label = ", MAX(t.agent_name)" if dim == "agent" else ", ''"
                facets[dim] = [(r[0], r[1], int(r[2])) for r in self._conn.execute(
                    f"SELECT t.{col}{label}, COUNT(*) FROM tasks t WHERE {w} GROUP BY t.{col} "
                    "ORDER BY COUNT(*) DESC", a)]
            w, a = self._where(user_id, **common, skip="party")
            # v2: a person the board knows is one value, under their name
            # (``party:<id>``); an address nobody is known by stays itself.
            facets["party"] = [(r[0], r[1], int(r[2])) for r in self._conn.execute(
                f"SELECT {_PARTY_FACET} AS pid, COALESCE(NULLIF(MAX(pp.display_name), ''), "
                "NULLIF(MAX(p.display), ''), MAX(p.address)), COUNT(DISTINCT t.id) FROM tasks t "
                "JOIN task_parties p ON p.task_id = t.id AND p.user_id = t.user_id "
                f"LEFT JOIN parties pp ON pp.id = p.party_id AND pp.user_id = p.user_id WHERE {w} "
                "GROUP BY pid ORDER BY COUNT(DISTINCT t.id) DESC LIMIT 50", a)]
            counts = self._counts(user_id, **{**common, "view": "all"}, skip="view")
        return {"rows": [dict(r) for r in rows], "total": int(total), "facets": facets, "counts": counts}

    def _counts(self, user_id: str, **common: Any) -> dict[str, int]:
        now, proposal_tasks = common["now"], common["proposal_tasks"]
        where, args = self._where(user_id, **common)
        parts, part_args = [], []
        for view in ("needs_me", "waiting", "working", "done"):
            cond, a = self._view_sql(view, now, proposal_tasks)
            parts.append(f"COALESCE(SUM(CASE WHEN {cond} THEN 1 ELSE 0 END), 0)")
            part_args += a
        row = self._conn.execute(
            f"SELECT {', '.join(parts)}, COUNT(*) FROM tasks t WHERE {where}", (*part_args, *args)).fetchone()
        return {"needs_me": int(row[0]), "waiting": int(row[1]), "working": int(row[2]),
                "done": int(row[3]), "all": int(row[4])}

    def counts(self, user_id: str, *, now: float | None = None, proposal_tasks: list[str] | None = None) -> dict:
        """The user's count in each view, no filters (the tabs, the tile badge)."""
        with self._lock:
            return self._counts(user_id, view="all", terms=[], filters={}, date_from=None, date_to=None,
                                now=time.time() if now is None else now,
                                proposal_tasks=list(proposal_tasks or []), skip="view")

    def lane_of(self, user_id: str, task_id: str, *, now: float | None = None,
                proposal_tasks: list[str] | None = None) -> str:
        lane, args = self._lane_sql(time.time() if now is None else now, list(proposal_tasks or []))
        with self._lock:
            row = self._conn.execute(
                f"SELECT {lane} FROM tasks t WHERE t.id = ? AND t.user_id = ?", (*args, task_id, user_id)).fetchone()
        return str(row[0]) if row else "other"

    # ── Corrections ───────────────────────────────────────────────────

    def correct(self, user_id: str, task_id: str, action: str, value: Any = None, actor: str = "") -> dict | None:
        """Apply the user's correction *action* now: it writes the field(s),
        locks them against every later machine pass and leaves a
        ``correction`` event saying what changed (and what it was, for Undo).
        ``undo`` reverses the latest correction not yet undone — or, when Jev
        closed the task after it (decision #2: an auto-close comes with
        Undo), opens the task again. None when the task isn't this user's;
        ValueError for an action that can't apply."""
        now = time.time()
        with self._lock:
            row = self._owned(user_id, task_id)
            if row is None:
                return None
            last = self._last_correction(user_id, row) if action == "undo" else None
            jev_closed = (row["closed_by"] == "jev" and row["status"] in CLOSED and row["closed_at"] is not None
                          and (last is None or float(row["closed_at"]) > float(last[0]["at"])))
            if action == "undo" and not jev_closed:
                self._undo(user_id, row, now, actor, last)
            else:
                if jev_closed:
                    # A correction of its own (Undo again closes it as Jev had it).
                    fields, locks, _ = _correction(row, "reopen", None, now, actor)
                    action, summary = "undo_close", "Undid Jev's close"
                else:
                    fields, locks, summary = _correction(row, action, value, now, actor)
                old_locks = _loads(row["user_locks"], {})
                before = {k: row[k] for k in fields}
                if "flags" in fields:
                    before["flags"] = _loads(row["flags"], {})
                new_locks = {**old_locks, **{k: now for k in locks}}
                sets = {**fields, "user_locks": new_locks, "updated_at": now}
                self._write(task_id, sets)
                self._add_event(user_id, task_id, {
                    "source": "user", "source_ref": f"corr:{secrets.token_hex(8)}", "at": now,
                    "kind": "correction", "actor_kind": "me", "actor": actor, "summary": summary,
                    "evidence": "user",
                    "refs": {"action": action, "before": before, "locked_at": now,
                             "locks": {k: old_locks.get(k) for k in locks}, "undone": False}})
            self._refresh(task_id)
            self._conn.commit()
            return self._owned(user_id, task_id)

    def _write(self, task_id: str, sets: dict) -> None:
        sets = {k: (json.dumps(v) if k in _JSON_FIELDS else v) for k, v in sets.items()}
        self._conn.execute(
            f"UPDATE tasks SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?", (*sets.values(), task_id))

    def _last_correction(self, user_id: str, row: dict) -> tuple[sqlite3.Row, dict, set[str]] | None:
        """The latest correction that still holds a field: ``(event, refs,
        the fields it holds)``. A correction whose locks a source has since
        lifted (a scheduled task's next run) is settled and doesn't count."""
        locks = _loads(row["user_locks"], {})
        for ev in self._conn.execute(
                "SELECT id, at, refs, summary FROM task_events WHERE user_id = ? AND task_id = ? "
                "AND kind = 'correction' AND source = 'user' ORDER BY at DESC, id DESC", (user_id, row["id"])):
            refs = _loads(ev["refs"], {})
            if refs.get("action") == "undo" or refs.get("undone"):
                continue
            mine = {k for k in refs.get("locks") or {} if locks.get(k) == refs.get("locked_at", ev["at"])}
            if mine:
                return ev, refs, mine
        return None

    def _undo(self, user_id: str, row: dict, now: float, actor: str,
              last: tuple[sqlite3.Row, dict, set[str]] | None) -> None:
        """Take back *last* (:meth:`_last_correction`): what it wrote goes
        back to what it was, its locks to what they were, and what the
        sources said meanwhile about a field no lock holds now (``held`` on
        their events) is applied — so the task stands where its source has it
        today, not where it stood before the correction. With no correction
        left that holds a field, there is nothing to undo."""
        locks = _loads(row["user_locks"], {})
        if last is None:
            raise ValueError("there is no correction to undo")
        ev, refs, mine = last
        freed = set()
        for field in mine:
            was = refs["locks"][field]
            if was is None:
                locks.pop(field, None)
                freed.add(field)
            else:
                locks[field] = was
        sets: dict[str, Any] = {}
        for field, value in (refs.get("before") or {}).items():
            if field == "flags":
                # Only the flags this correction locked go back.
                flags = _loads(row["flags"], {})
                for key in mine - _TASK_COLUMNS:
                    if key in (value or {}):
                        flags[key] = value[key]
                    else:
                        flags.pop(key, None)
                sets["flags"] = flags
            elif field in _TASK_COLUMNS and _LOCK_GROUP.get(field, field) in mine:
                sets[field] = value
        self._write(row["id"], {**sets, "user_locks": locks, "updated_at": now})
        for later in self._conn.execute(
                "SELECT at, refs FROM task_events WHERE task_id = ? AND id > ? AND source != 'user' ORDER BY id",
                (row["id"], ev["id"])).fetchall():
            held = _loads(later["refs"], {}).get("held") or {}
            patch = {k: v for k, v in held.items() if k != "flags" and _LOCK_GROUP.get(k, k) in freed}
            flags = {k: v for k, v in (held.get("flags") or {}).items() if k in freed}
            if flags:
                patch["flags"] = flags
            if patch:
                self._machine_write(row["id"], patch, float(later["at"]))
        self._conn.execute("UPDATE task_events SET refs = ? WHERE id = ?",
                           (json.dumps({**refs, "undone": True}), ev["id"]))
        self._add_event(user_id, row["id"], {
            "source": "user", "source_ref": f"corr:{secrets.token_hex(8)}", "at": now, "kind": "correction",
            "actor_kind": "me", "actor": actor, "summary": f"Undid: {ev['summary']}", "evidence": "user",
            "refs": {"action": "undo", "undo_of": ev["id"]}})

    # ── Cursors and settings ──────────────────────────────────────────

    def get_cursor(self, source: str) -> str:
        with self._lock:
            row = self._conn.execute("SELECT cursor FROM source_cursors WHERE source = ?", (source,)).fetchone()
        return str(row[0]) if row else ""

    def set_cursor(self, source: str, cursor: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO source_cursors (source, cursor, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(source) DO UPDATE SET cursor = excluded.cursor, updated_at = excluded.updated_at",
                (source, cursor, time.time()))
            self._conn.commit()

    def get_settings(self, user_id: str) -> dict:
        with self._lock:
            row = self._conn.execute("SELECT * FROM taskboard_settings WHERE user_id = ?", (user_id,)).fetchone()
        d = dict(row) if row else {"user_id": user_id, "passive": "{}", "auto_close": 1, "jev_per_day": 200,
                                   "extract_per_day": 100, "counters": "{}", "backfill_done_at": None,
                                   "updated_at": None}
        d["passive"] = _loads(d["passive"], {})
        d["counters"] = _loads(d["counters"], {})
        d["auto_close"] = bool(d["auto_close"])
        return d

    def save_settings(self, user_id: str, **fields: Any) -> dict:
        allowed = {"passive", "auto_close", "jev_per_day", "extract_per_day", "counters", "backfill_done_at"}
        cols = {k: (json.dumps(v) if k in ("passive", "counters") else v) for k, v in fields.items() if k in allowed}
        cols["updated_at"] = time.time()
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO taskboard_settings (user_id) VALUES (?)", (user_id,))
            self._conn.execute(
                f"UPDATE taskboard_settings SET {', '.join(f'{k} = ?' for k in cols)} WHERE user_id = ?",
                (*cols.values(), user_id))
            self._conn.commit()
        return self.get_settings(user_id)

    # ── v1 triage (extraction, Jev decisions, daily caps) ─────────────

    def triage_users(self, now: float | None = None) -> list[str]:
        """The boards with a task waiting to be triaged now."""
        now = time.time() if now is None else now
        with self._lock:
            return [str(r[0]) for r in self._conn.execute(
                "SELECT DISTINCT user_id FROM tasks WHERE triage_state = 'pending' AND triage_after IS NOT NULL "
                "AND triage_after <= ? AND merged_into = ''", (now,))]

    def triage_queue(self, user_id: str, *, now: float | None = None, limit: int = 20) -> list[dict]:
        """The user's tasks waiting to be triaged now, the longest waiting first."""
        now = time.time() if now is None else now
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM tasks WHERE user_id = ? AND triage_state = 'pending' AND triage_after IS NOT NULL "
                "AND triage_after <= ? AND merged_into = '' ORDER BY triage_after, created_at, id LIMIT ?",
                (user_id, now, max(1, int(limit))))]

    def triage_view(self, user_id: str, task_id: str, events: int = 12) -> dict | None:
        """What triage reads about one task, at one moment: its row, its
        latest *events* (oldest first), its parties and the turns it came
        from (the register's rows, by ``turns.task_id`` or a ``turn:`` key,
        oldest first). None when it isn't this user's."""
        with self._lock:
            row = self._owned(user_id, task_id)
            if row is None:
                return None
            keys = [link["key"] for link in self.links(user_id, [task_id]).get(task_id, [])]
            turn_ids = [k.removeprefix("turn:") for k in keys if k.startswith("turn:")]
            by_key = f" OR turn_id IN ({_marks(turn_ids)})" if turn_ids else ""
            turns = [_turn_dict(r) for r in self._conn.execute(
                f"SELECT * FROM turns WHERE user_id = ? AND (task_id = ?{by_key}) "
                "ORDER BY COALESCE(started_at, 0), turn_id", (user_id, task_id, *turn_ids))]
            return {"task": row, "events": self.events(user_id, task_id, limit=events),
                    "parties": self.parties_for(user_id, [task_id]).get(task_id, []), "turns": turns}

    def triage_write(
        self, user_id: str, task_id: str, fingerprint: str, *, fields: dict | None = None,
        parties: tuple[dict, ...] | list[dict] = (), decision: dict | None = None, extracted: bool = False,
        decided: bool = False, state: str | None = None, error: str = "", after: float | None = None,
    ) -> bool:
        """Write what triage made of a task — only while its fingerprint is
        still *fingerprint* (a new event since the read leaves the task to the
        next pass, which reads it anew): *fields* as a machine pass writes
        them (never a field the user locked), *parties* unverified (a model
        named them: never a recipient) and only those the task doesn't name
        already, *decision* logged, the fingerprints
        it settles stamped, and its triage *state* (None leaves it). False
        when nothing was written."""
        with self._lock:
            row = self._owned(user_id, task_id)
            if row is None or row["fingerprint"] != fingerprint:
                return False
            now = time.time()
            changed = self._machine_write(task_id, fields, now) if fields else False
            if parties:
                fresh = self._new_parties(user_id, task_id, parties)
                changed |= self._add_parties(user_id, task_id, [{**p, "verified": False} for p in fresh]) > 0
            if decision:
                self._conn.execute(
                    "INSERT INTO task_decisions (task_id, user_id, at, judge, qset, state_hash, answers, applied, why) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (task_id, user_id, now, str(decision.get("judge") or ""), str(decision.get("qset") or ""),
                     str(decision.get("state_hash") or ""), json.dumps(decision.get("answers") or {}),
                     str(decision.get("applied") or "")[:300], str(decision.get("why") or "")[:400]))
            sets: dict[str, Any] = {}
            if extracted:
                sets["extracted_fp"] = fingerprint
            if decided:
                sets["decided_fp"] = fingerprint
            if state is not None:
                sets.update(triage_state=state, triage_error=error[:300],
                            triage_after=after if state == "pending" else None)
            if sets:
                self._write(task_id, sets)
            if changed:
                self._refresh(task_id)
            self._conn.commit()
        return True

    def _new_parties(self, user_id: str, task_id: str, parties: tuple[dict, ...] | list[dict]) -> list[dict]:
        """Of the *parties* a model named, those that add someone: not one
        whose folded name is a party's the task has already (its requester,
        the board user under their own name) or one named before it."""
        seen = {" ".join(_fold(str(r[0] or "")).split()) for r in self._conn.execute(
            "SELECT display FROM task_parties WHERE user_id = ? AND task_id = ?", (user_id, task_id))}
        out = []
        for p in parties:
            name = " ".join(_fold(str(p.get("display") or "")).split())
            if name and name in seen:
                continue
            seen.add(name)
            out.append(p)
        return out

    def daily_counts(self, user_id: str, day: str | None = None) -> dict:
        """The user's Jev questions and extractions on this UTC *day*:
        ``{day, jev, extract}`` — zeros once a new day starts."""
        day = day or utc_day()
        counters = self.get_settings(user_id)["counters"]
        out = {"day": day, "jev": 0, "extract": 0}
        if counters.get("day") == day:
            for k in ("jev", "extract"):
                try:
                    out[k] = max(0, int(counters.get(k) or 0))
                except (TypeError, ValueError):
                    pass
        return out

    def take_daily(self, user_id: str, counter: str, n: int = 1, *, day: str | None = None) -> int:
        """Take up to *n* of today's *counter* (``jev`` or ``extract``) under
        the user's cap (``jev_per_day`` / ``extract_per_day``; 0 = none):
        how many it got. A negative *n* gives back what was not used."""
        cap_field = {"jev": "jev_per_day", "extract": "extract_per_day"}[counter]
        day = day or utc_day()
        with self._lock:
            cap = int(self.get_settings(user_id)[cap_field] or 0)
            counts = self.daily_counts(user_id, day)
            got = max(0, min(n, cap - counts[counter])) if n >= 0 else -min(-n, counts[counter])
            if got:
                counts[counter] += got
                self.save_settings(user_id, counters=counts)
        return got

    def badge_count(self, user_id: str, *, now: float | None = None,
                    proposal_tasks: list[str] | None = None) -> int:
        """The tile badge: Needs me, less the tasks admitted at low
        confidence (Jev 0.65–0.8) — those wait in the list, unannounced —
        and (v2) those filled in from history, which never notify."""
        needs, args = self._needs_me(time.time() if now is None else now, list(proposal_tasks or []))
        with self._lock:
            return int(self._conn.execute(
                f"SELECT COUNT(*) FROM tasks t WHERE t.user_id = ? AND t.merged_into = '' AND {needs} "
                "AND COALESCE(json_extract(t.flags, '$.low_confidence'), 0) = 0 "
                "AND COALESCE(json_extract(t.flags, '$.historical'), 0) = 0", (user_id, *args)).fetchone()[0])

    # ── v1 turns (the agents' turn register, taskboard_ingest) ────────

    def save_turn(self, turn_id: str, phase: str, ref: str, fields: dict) -> str:
        """Keep one record of an agent's turn register, once: its *phase*
        (``open`` says where the turn came from and whose board it is,
        ``close`` how it ended — either may come first) under the record
        number it came with (*ref*, ``boot:seq``). ``new``; ``repeat`` when
        this phase of the turn, or that number of the agent's, is here
        already; ``refused`` when the turn is another agent's, or would move
        to another board."""
        col = "open_ref" if phase == "open" else "close_ref"
        slug, owner = str(fields.get("agent_slug") or ""), str(fields.get("agent_owner_id") or "")
        sets = {k: (json.dumps(v) if k in _TURN_JSON else int(bool(v)) if k in _TURN_FLAGS else v)
                for k, v in fields.items() if k in _TURN_FIELDS}
        with self._lock:
            row = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
            if row is not None and (row["agent_slug"], row["agent_owner_id"]) != (slug, owner):
                return "refused"
            if row is not None and (row[col] or row["gate"] == TURN_DROPPED):
                return "repeat"
            if self._conn.execute("SELECT 1 FROM turns WHERE agent_slug = ? AND (open_ref = ? OR close_ref = ?)",
                                  (slug, ref, ref)).fetchone():
                return "repeat"
            if row is not None and row["user_id"] and sets.get("user_id") not in (None, "", row["user_id"]):
                return "refused"
            sets[col] = ref
            if row is None:
                sets.update(turn_id=turn_id, received_at=time.time())
                sets.setdefault("user_id", "")
                self._conn.execute(f"INSERT INTO turns ({', '.join(sets)}) VALUES ({_marks(sets)})",
                                   tuple(sets.values()))
            else:
                if not sets.get("user_id"):
                    sets.pop("user_id", None)  # a close knows no board; the open said it
                self._conn.execute(f"UPDATE turns SET {', '.join(f'{k} = ?' for k in sets)} WHERE turn_id = ?",
                                   (*sets.values(), turn_id))
            self._conn.commit()
        return "new"

    def drop_turn(self, turn_id: str, agent_slug: str, agent_owner_id: str, ref: str) -> None:
        """A turn that is nobody's to see (a member's words naming no member):
        what any record of it said goes, and none said later is kept — the
        turn stays only as a mark that it was dropped."""
        with self._lock:
            row = self._conn.execute("SELECT agent_slug, agent_owner_id, open_ref FROM turns WHERE turn_id = ?",
                                     (turn_id,)).fetchone()
            if row is not None and ((row["agent_slug"], row["agent_owner_id"]) != (agent_slug, agent_owner_id)
                                    or row["open_ref"]):
                return  # another agent's turn, or one whose open was taken already
            self._conn.execute("DELETE FROM turns WHERE turn_id = ?", (turn_id,))
            self._conn.execute(
                "INSERT INTO turns (turn_id, user_id, agent_slug, agent_owner_id, gate, open_ref, received_at) "
                "VALUES (?, '', ?, ?, ?, ?, ?)", (turn_id, agent_slug, agent_owner_id, TURN_DROPPED, ref, time.time()))
            self._conn.commit()

    def get_turn(self, turn_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
        return _turn_dict(row) if row else None

    def set_turn(self, turn_id: str, **fields: Any) -> None:
        """What the gate made of a turn: its ``gate`` verdict, and the task it
        is on (``task_id``) with the key that finds it (``task_key``)."""
        sets = {k: v for k, v in fields.items() if k in ("gate", "task_id", "task_key")}
        if not sets:
            return
        with self._lock:
            self._conn.execute(f"UPDATE turns SET {', '.join(f'{k} = ?' for k in sets)} WHERE turn_id = ?",
                               (*sets.values(), turn_id))
            self._conn.commit()

    def waiting_turns(self, *, user_id: str | None = None, limit: int = 500) -> list[dict]:
        """Automation turns whose task wasn't on the board yet (the run that
        made it not read yet), the oldest first."""
        mine, args = (" AND user_id = ?", [user_id]) if user_id is not None else ("", [])
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM turns WHERE gate = 'waiting'{mine} ORDER BY received_at LIMIT ?",
                (*args, max(1, int(limit)))).fetchall()
        return [_turn_dict(r) for r in rows]

    def prune_turns(self, before: float) -> int:
        """Forget the turns first seen before *before*. The tasks they made
        keep their ``turn:`` key, so a record sent again adds nothing."""
        with self._lock:
            n = self._conn.execute("DELETE FROM turns WHERE received_at < ?", (float(before),)).rowcount
            self._conn.commit()
        return int(n or 0)

    def rules(self, user_id: str) -> list[dict]:
        """The user's task rules (``match`` read), the oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM task_rules WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()
        return [{**dict(r), "match": _loads(r["match"], {})} for r in rows]

    def add_rule(self, user_id: str, match: dict, effect: str, created_from: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO task_rules (user_id, match, effect, created_from, created_at) VALUES (?,?,?,?,?)",
                (user_id, json.dumps(match), effect, created_from, time.time()))
            self._conn.commit()
        return int(cur.lastrowid or 0)

    def rule_hit(self, user_id: str, rule_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE task_rules SET hits = hits + 1 WHERE id = ? AND user_id = ?",
                               (rule_id, user_id))
            self._conn.commit()

    # ── v2 next steps (taskboard_nextsteps) ───────────────────────────

    def task_states(self, user_id: str, task_ids: list[str]) -> dict[str, dict]:
        """``{task_id: {status, merged_into, flags}}`` for those of
        *task_ids* on *user_id*'s board — whether a next step waiting on one
        still has a task to wait for."""
        out: dict[str, dict] = {}
        with self._lock:
            for i in range(0, len(task_ids), 500):
                chunk = task_ids[i:i + 500]
                for r in self._conn.execute(
                        f"SELECT id, status, merged_into, flags FROM tasks WHERE user_id = ? AND id IN ({_marks(chunk)})",
                        (user_id, *chunk)):
                    out[r["id"]] = {"status": r["status"], "merged_into": r["merged_into"],
                                    "flags": _loads(r["flags"], {})}
        return out

    def proposal_made(self, user_id: str, task_id: str, action_id: str, title: str) -> bool:
        """A next step proposed on *task_id*: the task names its ledger action
        (``autonomy:<id>``, where the autonomy sweep finds it) and the timeline
        says so — the event the sweep writes for it, so it is there once.
        False when the task isn't this user's."""
        with self._lock:
            if self._owned(user_id, task_id) is None:
                return False
            self._link(user_id, f"autonomy:{action_id}", task_id, "strong")
            self._add_event(user_id, task_id, {
                "source": "autonomy", "source_ref": f"{action_id}:made", "at": time.time(), "kind": "proposal",
                "channel": "autonomy", "actor_kind": "system", "actor": "Taskboard",
                "summary": f"Proposed: {' '.join(str(title or '').split())[:200]}",
                "refs": {"autonomy_id": action_id, "kind": "tool_action"}})
            self._conn.commit()
        return True

    def proposal_rejected(self, user_id: str, action_id: str) -> str | None:
        """The user rejected the next step *action_id*: a ``rejection`` event
        on its task — the state event the autonomy sweep writes for it
        (:meth:`next_state_ref`), so it is there once — and no move: the task
        stays open, its next move as it was. The task's id, or None."""
        key = f"autonomy:{action_id}"
        with self._lock:
            task_id = self._key_task(user_id, key)
            ref = self.next_state_ref("autonomy", action_id, "rejected") if task_id else None
            if task_id and ref:
                self.ingest(user_id, keys=[key], event={
                    "source": "autonomy", "source_ref": ref, "at": time.time(), "kind": "rejection",
                    "channel": "autonomy", "actor_kind": "me", "summary": "Rejected",
                    "refs": {"autonomy_id": action_id, "state": "rejected"}})
        return task_id

    # ── v2 history (taskboard_backfill) ───────────────────────────────

    def holding(self) -> threading.RLock:
        """The board's lock, for a run of writes no other pass may read
        between (``with store.holding(): …``): the backfill admits a turn and
        marks it historical before the triage worker can take it up."""
        return self._lock

    def register_since(self, agent_slug: str) -> float | None:
        """When the agent's turn register first reported a turn the board
        keeps (its open, or the mark of one dropped) — from then on the
        register has its turns, and the backfill reads only what came
        before. None when it never did."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(COALESCE(started_at, received_at)) FROM turns WHERE agent_slug = ?",
                (agent_slug,)).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def history_state(self, user_id: str) -> dict:
        """How the user's 30-day backfill stands as saved: ``{state, done_at,
        progress, message}`` — ``done`` once ``backfill_done_at`` is stamped
        (it never runs again), else ``idle``."""
        done_at = self.get_settings(user_id).get("backfill_done_at")
        saved = _loads(self.get_cursor(f"history:{user_id}"), {}) if done_at is not None else {}
        progress = saved.get("progress") if isinstance(saved.get("progress"), dict) else {}
        return {"state": "done" if done_at is not None else "idle", "done_at": done_at,
                "progress": {k: int(progress.get(k) or 0) for k in ("agents_done", "agents_total", "turns", "tasks")},
                "message": str(saved.get("message") or "")}

    def finish_history(self, user_id: str, *, progress: dict, message: str) -> dict:
        """The user's backfill ran to its end: what it did is kept (counts and
        a sentence, never a task's text) and ``backfill_done_at`` stamped."""
        self.set_cursor(f"history:{user_id}", json.dumps({"progress": progress, "message": message[:500]}))
        self.save_settings(user_id, backfill_done_at=time.time())
        return self.history_state(user_id)

    # ── v2 passive (the opt-in inboxes, taskboard_passive) ────────────

    def passive_users(self, key: str) -> list[str]:
        """The boards that switched the opt-in inbox *key* on
        (``taskboard_settings.passive[key].on``)."""
        with self._lock:
            rows = self._conn.execute("SELECT user_id, passive FROM taskboard_settings").fetchall()
        return [str(r[0]) for r in rows if (_loads(r[1], {}).get(key) or {}).get("on") is True]

    def passive_seen(self, source: str, refs: list[str]) -> set[str]:
        """Which of *refs* (the ``source_ref`` of *source*'s events) the
        board has already."""
        out: set[str] = set()
        with self._lock:
            for i in range(0, len(refs), 500):
                chunk = refs[i:i + 500]
                out.update(str(r[0]) for r in self._conn.execute(
                    f"SELECT source_ref FROM task_events WHERE source = ? AND source_ref IN ({_marks(chunk)})",
                    (source, *chunk)))
        return out

    def _passive_rows(self, user_id: str, task_id: str | None) -> tuple[set[str], set[tuple[str, str, str]]]:
        if not task_id:
            return set(), set()
        links = {str(r[0]) for r in self._conn.execute(
            "SELECT key FROM task_links WHERE user_id = ? AND task_id = ?", (user_id, task_id))}
        parties = {(str(r[0]), str(r[1]), str(r[2])) for r in self._conn.execute(
            "SELECT role, kind, address FROM task_parties WHERE user_id = ? AND task_id = ?", (user_id, task_id))}
        return links, parties

    def passive_write(
        self, user_id: str, source: str, keys: list[str], write: Any, *, release: bool = False,
        retention_until: float | None = None,
    ) -> tuple[str | None, bool]:
        """One message of the inbox *source* onto the board: ``write()`` puts
        it there (an :meth:`ingest`, as a sweep adapter does), under the
        board's lock, so the links and parties it gave the task are known
        exactly and marked as the inbox's. With *release*, a task the inbox
        made gives up the first of *keys* (its conversation) first — the
        conversation's next message is a task of its own. A task the inbox
        made keeps its text as long as its newest message's
        (*retention_until*). ``write()``'s ``(task_id, event_was_new)``."""
        with self._lock:
            held = next((t for t in (self._key_task(user_id, k) for k in keys) if t), None)
            if release and held and (self._row(held) or {}).get("source") == source:
                self._conn.execute("DELETE FROM task_links WHERE user_id = ? AND key = ? AND task_id = ?",
                                   (user_id, keys[0], held))
                self._conn.execute("DELETE FROM passive_marks WHERE user_id = ? AND task_id = ? AND what = 'link' "
                                   "AND ref = ?", (user_id, held, keys[0]))
                held = next((t for t in (self._key_task(user_id, k) for k in keys) if t), None)
            links, parties = self._passive_rows(user_id, held)
            task_id, new = write()
            if task_id:
                if task_id != held:
                    links, parties = set(), set()
                now_links, now_parties = self._passive_rows(user_id, task_id)
                marks = [("link", k) for k in sorted(now_links - links)]
                marks += [("party", json.dumps(list(p))) for p in sorted(now_parties - parties)]
                self._conn.executemany(
                    "INSERT OR IGNORE INTO passive_marks (user_id, source, task_id, what, ref) VALUES (?,?,?,?,?)",
                    [(user_id, source, task_id, what, ref) for what, ref in marks])
                if retention_until is not None:
                    self._conn.execute(
                        "UPDATE tasks SET retention_until = MAX(COALESCE(retention_until, 0), ?) "
                        "WHERE id = ? AND source = ?", (float(retention_until), task_id, source))
                self._conn.commit()
        return task_id, new

    def passive_forget(self, user_id: str, source: str) -> int:
        """Everything the inbox *source* put on *user_id*'s board taken off
        it (they switched it off): each of its events, the tasks left with
        nothing else on them, and on the rest what it gave them
        (:meth:`_passive_take`). How many tasks went."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, task_id FROM task_events WHERE user_id = ? AND source = ?", (user_id, source)).fetchall()
            gone = self._passive_take(user_id, source, rows)
            # A task it made with nothing on it at all.
            empty = [str(r[0]) for r in self._conn.execute(
                "SELECT id FROM tasks t WHERE user_id = ? AND source = ? AND merged_into = '' "
                "AND NOT EXISTS (SELECT 1 FROM task_events e WHERE e.task_id = t.id)", (user_id, source))]
            self._passive_drop(user_id, empty)
            self._conn.execute("DELETE FROM passive_marks WHERE user_id = ? AND source = ?", (user_id, source))
            self._conn.commit()
        return gone + len(empty)

    def passive_purge(self, source: str, *, now: float, kept_s: float, user_id: str | None = None) -> int:
        """The inbox *source*'s events whose text is past its retention
        (``refs.retention_until``, else *kept_s* after it was written) taken
        off every board — or *user_id*'s — as :meth:`passive_forget` takes
        them. How many tasks went."""
        mine, args = (" AND user_id = ?", [user_id]) if user_id is not None else ("", [])
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, task_id, user_id FROM task_events WHERE source = ? "
                f"AND COALESCE(json_extract(refs, '$.retention_until'), at + ?) < ?{mine}",
                (source, float(kept_s), float(now), *args)).fetchall()
            gone = 0
            for uid in sorted({str(r["user_id"]) for r in rows}):
                gone += self._passive_take(uid, source, [r for r in rows if r["user_id"] == uid])
            # Marks of tasks gone some other way (a user deleted, an agent removed).
            self._conn.execute("DELETE FROM passive_marks WHERE source = ? AND task_id NOT IN (SELECT id FROM tasks)",
                               (source,))
            self._conn.commit()
        return gone

    def _passive_take(self, user_id: str, source: str, rows: list[sqlite3.Row]) -> int:
        """Take the events *rows* (``id``, ``task_id``) of the inbox *source*
        off *user_id*'s board and settle each task they were on: one left
        with no event goes; the rest keep what else happened on them
        (:meth:`_passive_settle`) — what the user did too (a rename, a
        split: told without the inbox's text, :meth:`_passive_scrub`). A
        task it made that was merged into another, whose messages are
        there, is retold from what is left of them, or goes with them. How
        many tasks went. Under the lock, not committed."""
        ids = [int(r["id"]) for r in rows]
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            self._conn.execute(f"DELETE FROM task_events WHERE user_id = ? AND id IN ({_marks(chunk)})",
                               (user_id, *chunk))
        tasks = [r for r in (self._owned(user_id, t) for t in sorted({str(r["task_id"]) for r in rows}))
                 if r is not None]
        merged, taken = [], set(ids)
        for row in self._conn.execute("SELECT * FROM tasks WHERE user_id = ? AND source = ? AND merged_into != ''",
                                      (user_id, source)).fetchall():
            moved = self._merged_events(user_id, row["id"])
            if taken & set(moved):  # its messages, where they went, are among those going now
                merged.append((dict(row), moved))
        # The notes that quote its text first: a task is retold from what is left.
        self._passive_scrub(user_id, [r["id"] for r in tasks if r["source"] == source] + [r["id"] for r, _ in merged])
        gone = []
        for row in tasks:
            left = self._conn.execute("SELECT id, at, channel, source, summary, refs FROM task_events "
                                      "WHERE task_id = ? ORDER BY at, id", (row["id"],)).fetchall()
            if left:
                self._passive_settle(user_id, source, row, left)
            else:
                gone.append(row["id"])
        for row, moved in merged:
            # Its own events are notes: what it is now is what is left of the messages its merge moved.
            left = sorted((e for chunk in (moved[i:i + 500] for i in range(0, len(moved), 500))
                           for e in self._conn.execute(
                               "SELECT id, at, channel, source, summary, refs FROM task_events WHERE source != 'user' "
                               f"AND id IN ({_marks(chunk)})", chunk).fetchall()),
                          key=lambda e: (float(e["at"]), int(e["id"])))
            if left:
                self._passive_settle(user_id, source, row, left)
            else:
                gone.append(row["id"])
        self._passive_drop(user_id, gone)
        return len(gone)

    def _merged_events(self, user_id: str, task_id: str) -> list[int]:
        """The events *task_id*'s merge moved where it went (its own then,
        and what was merged into it before) — [] when it isn't merged."""
        rec = self._merge_record(user_id, task_id)
        return [int(i) for i in (rec[1].get("events") or []) if isinstance(i, int)] if rec else []

    def _passive_drop(self, user_id: str, task_ids: list[str]) -> None:
        """Tasks an inbox's text is all that was left of, dropped — the notes
        on other tasks that name them told without their titles first."""
        if not task_ids:
            return
        self._passive_scrub(user_id, task_ids)
        unnamed = [r for i in range(0, len(task_ids), 500) for r in self._conn.execute(
            f"SELECT party_id, display FROM task_parties WHERE task_id IN ({_marks(task_ids[i:i + 500])})",
            task_ids[i:i + 500]).fetchall()]
        self._drop_tasks(task_ids)
        for i in range(0, len(task_ids), 500):
            chunk = task_ids[i:i + 500]
            self._conn.execute(f"DELETE FROM passive_marks WHERE task_id IN ({_marks(chunk)})", chunk)
        self._passive_unname(user_id, unnamed)

    def _passive_scrub(self, user_id: str, task_ids: list[str]) -> None:
        """What the user's own notes keep of an inbox's text that is going:
        a merge, unmerge, split or "not the same" note on one of *task_ids*,
        or naming one of them from another task, is told without the titles
        it quotes; and a correction on one keeps none of the text it was
        made over (``before``: an Undo never brings it back)."""
        ids = sorted(set(task_ids))
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            names = " OR ".join(f"json_extract(refs, '$.{k}') IN ({_marks(chunk)})" for k in _NOTE_REFS)
            for ev in self._conn.execute(
                    "SELECT id, task_id, summary, refs FROM task_events WHERE user_id = ? AND source = 'user' "
                    f"AND kind = 'correction' AND (task_id IN ({_marks(chunk)}) OR {names})",
                    (user_id, *chunk, *(chunk * len(_NOTE_REFS)))).fetchall():
                refs = _loads(ev["refs"], {})
                summary = _THREAD_PLAIN.get(str(refs.get("action") or ""), ev["summary"])
                before = refs.get("before")
                if ev["task_id"] in chunk and isinstance(before, dict):
                    before = {k: v for k, v in before.items() if k not in _PASSIVE_TEXT_FIELDS}
                    if isinstance(before.get("flags"), dict):
                        before["flags"] = {k: v for k, v in before["flags"].items() if k not in _PASSIVE_TEXT_FLAGS}
                    refs["before"] = before
                self._conn.execute("UPDATE task_events SET summary = ?, refs = ? WHERE id = ?",
                                   (summary, json.dumps(refs), ev["id"]))

    def _passive_settle(self, user_id: str, source: str, row: dict, left: list[sqlite3.Row]) -> None:
        """A task some of the inbox *source*'s events were taken off, with
        others still on it. Once nothing of the inbox is left there, the
        links and parties it gave the task go too. A task the inbox made
        says only what is left: its oldest message still kept — else nothing
        but a title from the first thing that happened on it otherwise, and
        none of what a model read out of the inbox's text (its unverified
        parties, the flags and steps it wrote). A title the user gave it
        stays."""
        task_id = row["id"]
        mine = [e for e in left if e["source"] == source]
        sets: dict[str, Any] = {}
        unnamed: list[sqlite3.Row] = []
        if not mine:
            for m in self._conn.execute("SELECT what, ref FROM passive_marks WHERE user_id = ? AND source = ? "
                                        "AND task_id = ?", (user_id, source, task_id)).fetchall():
                if m["what"] == "link":
                    self._conn.execute("DELETE FROM task_links WHERE user_id = ? AND key = ? AND task_id = ?",
                                       (user_id, m["ref"], task_id))
                elif m["what"] == "party":
                    role, kind, address = ([*_loads(m["ref"], []), "", "", ""])[:3]
                    where = "FROM task_parties WHERE task_id = ? AND role = ? AND kind = ? AND address = ?"
                    unnamed += self._conn.execute(f"SELECT party_id, display {where}",
                                                  (task_id, role, kind, address)).fetchall()
                    self._conn.execute(f"DELETE {where}", (task_id, role, kind, address))
            self._conn.execute("DELETE FROM passive_marks WHERE user_id = ? AND source = ? AND task_id = ?",
                               (user_id, source, task_id))
        if row["source"] == source:
            text = " ".join(str((mine or left)[0]["summary"] or "").split())
            if "title" not in _loads(row["user_locks"], {}):
                sets["title"] = text[:160] or "Message"
            kept = [float(_loads(e["refs"], {}).get("retention_until") or 0) for e in mine]
            sets.update(ask=text[:600] if mine else "", result_text="", next_steps=[],
                        retention_until=(max(kept) or None) if mine else None,
                        flags={k: v for k, v in _loads(row["flags"], {}).items() if k not in _PASSIVE_TEXT_FLAGS})
            if not mine:
                unnamed += self._conn.execute("SELECT party_id, display FROM task_parties WHERE task_id = ? "
                                              "AND verified = 0", (task_id,)).fetchall()
                self._conn.execute("DELETE FROM task_parties WHERE task_id = ? AND verified = 0", (task_id,))
        channels = list(dict.fromkeys(c for c in (row["channel"], *(e["channel"] for e in left)) if c))
        sets.update(fingerprint=f"{max(int(e['id']) for e in left)}:{len(left)}",
                    last_event_at=max(float(e["at"]) for e in left), channels=channels, updated_at=time.time())
        self._write(task_id, sets)
        self._refresh(task_id)
        self._passive_unname(user_id, unnamed)

    def _passive_unname(self, user_id: str, rows: list[sqlite3.Row]) -> None:
        """The people an inbox's party rows *rows* (gone now) named: one still
        on the board keeps no name only such a row gave them (an inbox's
        header, a chat's contact name) — they are called what a row left on
        the board says, else by their address."""
        said: dict[str, set[str]] = {}
        for r in rows:
            if r["party_id"] and _person_name(r["display"]):
                said.setdefault(str(r["party_id"]), set()).add(_person_name(r["display"]))
        for party_id, names in said.items():
            cur = self._conn.execute("SELECT display_name FROM parties WHERE id = ? AND user_id = ?",
                                     (party_id, user_id)).fetchone()
            if cur is None or cur[0] not in names:
                continue
            left = self._conn.execute("SELECT kind, address, display, verified FROM task_parties WHERE user_id = ? "
                                      "AND party_id = ? ORDER BY verified DESC", (user_id, party_id)).fetchall()
            if not left or any(_person_name(r["display"]) == cur[0] for r in left):
                continue  # nobody (forgotten) — or a record left says it too
            name = next((n for n in (_person_name(r["display"]) for r in left if r["verified"]) if n), "") or \
                _address_name(left[0]["kind"], left[0]["address"])
            self._conn.execute("UPDATE parties SET display_name = ?, folded_name = ? WHERE id = ? AND user_id = ?",
                               (name, _folded(name), party_id, user_id))
            for (task_id,) in self._conn.execute("SELECT DISTINCT task_id FROM task_parties WHERE user_id = ? "
                                                 "AND party_id = ?", (user_id, party_id)).fetchall():
                self._refresh(task_id)

    # ── v2 threads and people (taskboard_threads) ─────────────────────

    def resolve_id(self, user_id: str, task_id: str) -> str:
        """The task *task_id* reads as: itself — or, merged into another, the
        one it is part of now (on along the chain when that one was merged
        too). "" when it isn't this user's."""
        with self._lock:
            return self._root(user_id, task_id)

    def _root(self, user_id: str, task_id: str) -> str:
        found = ""
        for _ in range(_MERGE_DEPTH):
            row = self._conn.execute(
                "SELECT merged_into FROM tasks WHERE id = ? AND user_id = ?", (task_id, user_id)).fetchone()
            if row is None:
                break  # merged into a task gone since (its agent removed): this one stands for it
            found = task_id
            if not row[0]:
                break
            task_id = str(row[0])
        return found

    def merged_from(self, user_id: str, task_id: str) -> list[dict]:
        """``{id, title}`` of each task merged straight into *task_id*."""
        with self._lock:
            return [{"id": r[0], "title": r[1]} for r in self._conn.execute(
                "SELECT id, title FROM tasks WHERE user_id = ? AND merged_into = ? ORDER BY updated_at, id",
                (user_id, task_id))]

    def merge(self, user_id: str, task_id: str, into: str, actor: str = "") -> dict | None:
        """The user's word that *task_id* is part of *into* (of the task
        *into* is part of now, when that was merged too): its events, links
        and turns go over and its parties are copied, so whatever its sources
        say next lands on the target; it is hidden from every list, count and
        badge; each side gets a ``correction`` event, the target's recording
        what moved — :meth:`unmerge` puts back exactly that. The target
        afterwards, back in the triage queue (it has more to go by). None
        when either isn't this user's; ValueError when it can't be
        (:class:`ThreadConflict`: the task is merged already)."""
        now = time.time()
        with self._lock:
            src = self._owned(user_id, task_id)
            root = self._root(user_id, str(into or ""))
            dst = self._owned(user_id, root) if root else None
            if src is None or dst is None:
                return None
            if src["merged_into"]:
                raise ThreadConflict("This task is part of another one already.")
            if root == task_id:
                raise ValueError("A task can't be merged into itself.")
            moved = self._move_task(user_id, task_id, root)
            src_flags = {k: v for k, v in _loads(src["flags"], {}).items() if k != "same_as"}
            dst_flags = _loads(dst["flags"], {})
            if _same_id(dst_flags) == task_id:
                dst_flags.pop("same_as", None)  # the suggestion it answers
            self._write(task_id, {"merged_into": root, "triage_state": "skipped", "triage_after": None,
                                  "flags": src_flags, "updated_at": now})
            self._write(root, {"flags": dst_flags, "updated_at": now,
                               **_requeued(dst, now, self._has_text(moved["events"]))})
            self._thread_note(user_id, root, now, actor, f"Merged “{src['title']}” into this task", {
                "action": "merge", "merged": task_id, "triage_state": src["triage_state"], **moved,
                "undone": False})
            self._thread_note(user_id, task_id, now, actor, f"Merged into “{dst['title']}”",
                              {"action": "merged_into", "into": root})
            self._retold(root, task_id)
            self._conn.commit()
            return self._owned(user_id, root)

    def unmerge(self, user_id: str, task_id: str, actor: str = "") -> tuple[dict, dict | None] | None:
        """Take *task_id* back out of the task it was merged into: what the
        merge moved goes back (the events, links and turns still there — what
        came in on them since stays where it landed), the parties it brought
        leave, and it is on the board again, triaged anew, as is the task it
        was in. Merges come apart the other way round from how they were
        made: while the target is merged on into a third task, that comes
        first (:class:`ThreadConflict`). ``(the task, the one it was in)``;
        None when it isn't this user's."""
        now = time.time()
        with self._lock:
            src = self._owned(user_id, task_id)
            if src is None:
                return None
            if not src["merged_into"]:
                raise ThreadConflict("This task isn't part of another one.")
            dst = self._owned(user_id, src["merged_into"])
            if dst is not None and dst["merged_into"]:
                raise ThreadConflict(f"“{dst['title']}” was merged into another task since — take it out first.")
            rec = self._merge_record(user_id, task_id)
            refs = rec[1] if rec else {}
            if dst is not None and rec is not None:
                self._move_back(user_id, refs, dst["id"], task_id)
                self._conn.execute("UPDATE task_events SET refs = ? WHERE id = ?",
                                   (json.dumps({**refs, "undone": True}), rec[0]))
            was = str(refs.get("triage_state") or "skipped")
            state = "pending" if src["kind"] in _TRIAGED_KINDS and was != "skipped" else was
            self._write(task_id, {"merged_into": "", "updated_at": now, "triage_state": state,
                                  "triage_after": now if state == "pending" else None})
            self._thread_note(user_id, task_id, now, actor,
                              f"Taken back out of “{dst['title']}”" if dst else "Taken back out of a task since removed",
                              {"action": "unmerge", "from": src["merged_into"]})
            if dst is not None:
                self._write(dst["id"], {"updated_at": now, **_requeued(dst, now, True)})
                self._thread_note(user_id, dst["id"], now, actor, f"Took “{src['title']}” back out",
                                  {"action": "unmerged", "task": task_id})
                self._retold(dst["id"])
            self._retold(task_id)
            self._conn.commit()
            return self._owned(user_id, task_id), (self._owned(user_id, dst["id"]) if dst else None)

    def split(self, user_id: str, task_id: str, event_ids: list[int], actor: str = "") -> tuple[dict, dict] | None:
        """The user's word that some of a task's events are another task:
        they go to a new one (named by the first message among them), with
        the links only they name — a turn's, an email's; never one an event
        left behind names too, nor one no event names (a scheduled job's) —
        the turns behind those, and the people who wrote them (else the
        task's requesters). Each side gets a ``correction`` event; merging the
        new task back puts it together again. ``(the task, the new one)``;
        None when it isn't this user's; ValueError for events it can't split
        off (every event, a correction, one not on it)."""
        now = time.time()
        with self._lock:
            row = self._owned(user_id, task_id)
            if row is None:
                return None
            if row["merged_into"]:
                raise ThreadConflict("This task is part of another one — split that one.")
            evs = {int(r["id"]): dict(r) for r in self._conn.execute(
                "SELECT * FROM task_events WHERE user_id = ? AND task_id = ? ORDER BY id", (user_id, task_id))}
            ids = sorted({int(i) for i in event_ids})
            if not ids:
                raise ValueError("Pick the events to split off.")
            missing = next((i for i in ids if i not in evs), None)
            if missing is not None:
                raise ValueError(f"Event {missing} isn't on this task.")
            if any(evs[i]["kind"] == "correction" for i in ids):
                raise ValueError("A correction stays on the task it corrected.")
            left = [e for i, e in evs.items() if i not in ids and e["kind"] != "correction"]
            if not left:
                raise ValueError("Leave at least one event on the task.")
            moved = [evs[i] for i in ids]
            first = next((e for e in moved if e["kind"] == "received"), moved[0])
            said = " ".join(str(first["summary"] or "").split())
            title = said[:160] or f"Part of “{row['title']}”"[:160]
            request = row["kind"] == "request"
            flags = {k: v for k, v in _loads(row["flags"], {}).items() if k in ("own_chat", "historical")}
            # Named by an opt-in inbox's message, the new task is the inbox's:
            # kept as long as that text is, and gone with it (passive_forget).
            inbox = first["source"] if first["source"] in PASSIVE_SOURCES else ""
            kept = [float(_loads(e["refs"], {}).get("retention_until") or 0) for e in moved if e["source"] == inbox]
            new_id = self._insert_task(user_id, {
                **{k: row[k] for k in ("agent_owner_id", "agent_slug", "agent_ref", "agent_name", "kind", "direction",
                                       "private")},
                "title": title, "ask": said[:600], "channel": first["channel"] or row["channel"], "status": "open",
                "owes_next": "agent" if request else "unclear", "flags": flags,
                "triage_state": "pending" if request else "skipped", "triage_after": now if request else None,
                "source": inbox or "split", "source_ref": task_id,
                **({"retention_until": max(kept, default=0.0) or row["retention_until"]} if inbox else {})},
                float(first["at"] or now))
            named = {k for e in left for k in _event_keys(e)}
            theirs = {k for e in moved for k in _event_keys(e)} - named
            keys = [str(r[0]) for r in self._conn.execute(
                "SELECT key FROM task_links WHERE user_id = ? AND task_id = ?", (user_id, task_id)) if r[0] in theirs]
            turns = [k.removeprefix("turn:") for k in keys if k.startswith("turn:")]
            for table, col, values in (("task_events", "id", ids), ("task_links", "key", keys),
                                       ("turns", "turn_id", turns)):
                self._move_rows(table, col, values, user_id, task_id, new_id)
            parties = self._conn.execute(
                "SELECT * FROM task_parties WHERE user_id = ? AND task_id = ?", (user_id, task_id)).fetchall()
            wrote = {_folded(e["actor"]) for e in moved if e["actor_kind"] == "party"} - {""}
            mine = any(e["actor_kind"] == "me" for e in moved)
            pick = [p for p in parties if _folded(p["display"]) in wrote
                    or (mine and p["kind"] == "fd_user" and p["role"] == "requester")]
            copied = self._copy_parties(user_id, new_id, pick or [p for p in parties if p["role"] == "requester"])
            self._carry_marks(user_id, task_id, new_id, keys, copied)
            self._write(task_id, {"updated_at": now, **_requeued(row, now, True)})
            self._thread_note(user_id, task_id, now, actor, f"Split “{title}” off as a task of its own",
                              {"action": "split", "new_task": new_id, "events": ids, "links": keys})
            self._thread_note(user_id, new_id, now, actor, f"Split off “{row['title']}”",
                              {"action": "split_from", "task": task_id})
            self._retold(task_id, new_id)
            self._conn.commit()
            return self._owned(user_id, task_id), self._owned(user_id, new_id)

    def not_same(self, user_id: str, task_id: str, actor: str = "") -> dict | None:
        """The user's word that the task Jev took this one for
        (``flags.same_as``) is another: the suggestion goes, and its flag is
        locked so no pass suggests one again — Undo brings it back. None when
        the task isn't this user's; ValueError when nothing is suggested."""
        now = time.time()
        with self._lock:
            row = self._owned(user_id, task_id)
            if row is None:
                return None
            flags = _loads(row["flags"], {})
            same = flags.get("same_as")
            if not isinstance(same, dict) or not same.get("task_id"):
                raise ValueError("Nothing says this is the same as another task.")
            locks = _loads(row["user_locks"], {})
            self._write(task_id, {"flags": {k: v for k, v in flags.items() if k != "same_as"},
                                  "user_locks": {**locks, "same_as": now}, "updated_at": now})
            self._add_event(user_id, task_id, {
                "source": "user", "source_ref": f"corr:{secrets.token_hex(8)}", "at": now, "kind": "correction",
                "actor_kind": "me", "actor": actor, "evidence": "user",
                "summary": f"Not the same as “{str(same.get('title') or 'that task')[:160]}”",
                "refs": {"action": "not_same", "before": {"flags": flags}, "locked_at": now,
                         "locks": {"same_as": locks.get("same_as")}, "undone": False}})
            self._refresh(task_id)
            self._conn.commit()
            return self._owned(user_id, task_id)

    def _move_task(self, user_id: str, src: str, dst: str) -> dict:
        """Everything on *src* over to *dst* — its events, links and turns —
        and its parties copied (*src* keeps its own rows, for an unmerge):
        what moved, as a merge record keeps it."""
        moved = {
            "events": [int(r[0]) for r in self._conn.execute(
                "SELECT id FROM task_events WHERE user_id = ? AND task_id = ? ORDER BY id", (user_id, src))],
            "links": [str(r[0]) for r in self._conn.execute(
                "SELECT key FROM task_links WHERE user_id = ? AND task_id = ? ORDER BY created_at, key", (user_id, src))],
            "turns": [str(r[0]) for r in self._conn.execute(
                "SELECT turn_id FROM turns WHERE user_id = ? AND task_id = ?", (user_id, src))],
            "parties": self._copy_parties(user_id, dst, self._conn.execute(
                "SELECT * FROM task_parties WHERE user_id = ? AND task_id = ?", (user_id, src)).fetchall()),
        }
        for table in ("task_events", "task_links", "turns"):
            self._conn.execute(f"UPDATE {table} SET task_id = ? WHERE user_id = ? AND task_id = ?", (dst, user_id, src))
        self._carry_marks(user_id, src, dst, moved["links"], moved["parties"])
        return moved

    def _move_back(self, user_id: str, refs: dict, dst: str, src: str) -> None:
        """What a merge record says moved from *src*, back from *dst* (only
        what is still there); the party rows it added to *dst* go."""
        for table, col, field in (("task_events", "id", "events"), ("task_links", "key", "links"),
                                  ("turns", "turn_id", "turns")):
            self._move_rows(table, col, list(refs.get(field) or []), user_id, dst, src)
        for p in refs.get("parties") or []:
            if isinstance(p, list) and len(p) == 3:
                self._conn.execute(
                    "DELETE FROM task_parties WHERE task_id = ? AND user_id = ? AND role = ? AND kind = ? "
                    "AND address = ?", (dst, user_id, *p))
                self._conn.execute("DELETE FROM passive_marks WHERE user_id = ? AND task_id = ? AND what = 'party' "
                                   "AND ref = ?", (user_id, dst, json.dumps(list(p))))
        self._carry_marks(user_id, dst, src, list(refs.get("links") or []), [])

    def _carry_marks(self, user_id: str, src: str, dst: str, links: list[str], parties: list[list[str]]) -> None:
        """What an opt-in inbox gave *src* (v2 passive, ``passive_marks``)
        goes along to *dst*: the marks of the *links* that moved, and of the
        *parties* copied — so switching the inbox off still finds them."""
        for key in links:
            self._conn.execute("UPDATE OR IGNORE passive_marks SET task_id = ? WHERE user_id = ? AND task_id = ? "
                               "AND what = 'link' AND ref = ?", (dst, user_id, src, key))
        for p in parties:
            self._conn.execute(
                "INSERT OR IGNORE INTO passive_marks (user_id, source, task_id, what, ref) SELECT user_id, source, ?, "
                "what, ref FROM passive_marks WHERE user_id = ? AND task_id = ? AND what = 'party' AND ref = ?",
                (dst, user_id, src, json.dumps(list(p))))

    def _move_rows(self, table: str, col: str, values: list, user_id: str, src: str, dst: str) -> None:
        for i in range(0, len(values), 500):
            chunk = values[i:i + 500]
            self._conn.execute(
                f"UPDATE {table} SET task_id = ? WHERE user_id = ? AND task_id = ? AND {col} IN ({_marks(chunk)})",
                (dst, user_id, src, *chunk))

    def _copy_parties(self, user_id: str, dst: str, rows: list) -> list[list[str]]:
        """*rows* (task_parties rows) onto *dst*, as people the board knows:
        ``[role, kind, address]`` of each that *dst* didn't have."""
        added = []
        for p in rows:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO task_parties (task_id, user_id, role, kind, address, display, party_id, "
                "verified) VALUES (?,?,?,?,?,?,?,?)",
                (dst, user_id, p["role"], p["kind"], p["address"], p["display"], p["party_id"], p["verified"]))
            if (cur.rowcount or 0) == 1:
                added.append([p["role"], p["kind"], p["address"]])
        return added

    def _merge_record(self, user_id: str, task_id: str) -> tuple[int, dict] | None:
        """The latest merge of *task_id* not undone yet — wherever its record
        is now — as ``(event id, refs)``."""
        row = self._conn.execute(
            "SELECT id, refs FROM task_events WHERE user_id = ? AND kind = 'correction' AND source = 'user' "
            "AND json_extract(refs, '$.action') = 'merge' AND json_extract(refs, '$.merged') = ? "
            "AND COALESCE(json_extract(refs, '$.undone'), 0) = 0 ORDER BY id DESC LIMIT 1",
            (user_id, task_id)).fetchone()
        return (int(row[0]), _loads(row[1], {})) if row else None

    def _has_text(self, event_ids: list[int]) -> bool:
        """Whether any of *event_ids* is something someone wrote (a message, a reply)."""
        return any(self._conn.execute(
            f"SELECT 1 FROM task_events WHERE id IN ({_marks(chunk)}) AND kind IN ('received', 'reply') LIMIT 1",
            chunk).fetchone() for chunk in (event_ids[i:i + 500] for i in range(0, len(event_ids), 500)))

    def _thread_note(self, user_id: str, task_id: str, at: float, actor: str, summary: str, refs: dict) -> None:
        """The user's merge / unmerge / split on the task's timeline (a
        correction no Undo takes back: its own reverse does)."""
        self._add_event(user_id, task_id, {
            "source": "user", "source_ref": f"thread:{secrets.token_hex(8)}", "at": at, "kind": "correction",
            "actor_kind": "me", "actor": actor, "summary": summary, "evidence": "user", "refs": refs})

    def _retold(self, *task_ids: str) -> None:
        """A task's own account of its events once some moved in or out — its
        fingerprint, when the latest happened, the channels they touched (its
        first channel first) — and its searchable text again."""
        for task_id in task_ids:
            last_id, n, last_at = self._conn.execute(
                "SELECT MAX(id), COUNT(*), MAX(at) FROM task_events WHERE task_id = ?", (task_id,)).fetchone()
            row = self._row(task_id) or {}
            channels = [row["channel"]] if row.get("channel") else []
            for (ch,) in self._conn.execute(
                    "SELECT channel FROM task_events WHERE task_id = ? AND channel != '' GROUP BY channel "
                    "ORDER BY MIN(id)", (task_id,)):
                if ch not in channels:
                    channels.append(ch)
            self._conn.execute("UPDATE tasks SET fingerprint = ?, last_event_at = ?, channels = ? WHERE id = ?",
                               (f"{last_id}:{n}" if n else "", last_at, json.dumps(channels), task_id))
            self._refresh(task_id)

    # Turns and tasks a new turn may be part of (taskboard_threads.turn_host).

    def previous_turn(self, user_id: str, agent_slug: str, session_id: str, before: float,
                      turn_id: str) -> dict | None:
        """The turn before *turn_id* in one session of one agent on this
        board — the one it follows — or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM turns WHERE user_id = ? AND agent_slug = ? AND session_id = ? AND turn_id != ? "
                "AND started_at IS NOT NULL AND started_at <= ? ORDER BY started_at DESC, received_at DESC LIMIT 1",
                (user_id, agent_slug, session_id, turn_id, float(before))).fetchone()
        return _turn_dict(row) if row else None

    def open_request_from(self, user_id: str, *, kind: str, address: str, agent_slug: str, channel: str,
                          since: float, before: float | None = None) -> str:
        """The latest open request on this board that one recorded person
        (*kind*, *address*) made of one agent on *channel* someone else's
        chat (never the board user's own), with something on it since
        *since* — and, with *before*, made by then (a request asked after
        the message it would continue is no conversation it was in): the
        conversation they are still in. "" when there is none (a request a
        mute rule keeps quiet never is one)."""
        made, args = ("AND t.created_at <= ? ", [float(before)]) if before is not None else ("", [])
        with self._lock:
            row = self._conn.execute(
                "SELECT t.id FROM tasks t JOIN task_parties p ON p.task_id = t.id AND p.user_id = t.user_id "
                "WHERE t.user_id = ? AND t.agent_slug = ? AND t.channel = ? AND t.kind = 'request' "
                "AND t.status IN ('open', 'maybe') AND t.merged_into = '' AND t.triage_state != 'skipped' "
                "AND COALESCE(json_extract(t.flags, '$.own_chat'), 0) = 0 "
                f"AND COALESCE(t.last_event_at, t.created_at) >= ? {made}"
                "AND p.role = 'requester' AND p.kind = ? AND p.address = ? AND p.verified = 1 "
                "ORDER BY COALESCE(t.last_event_at, t.created_at) DESC, t.id LIMIT 1",
                (user_id, agent_slug, channel, float(since), *args, kind, address)).fetchone()
        return str(row[0]) if row else ""

    def same_task_candidates(self, user_id: str, task: dict, *, since: float, limit: int = 30) -> list[dict]:
        """Older tasks on this board that share a recorded person with *task*
        — someone the board knows, or the same recorded address; never the
        board user (they are on every one of their own chats) — with
        something on them since *since*: what "the same task?" is asked
        about. Never a merged, dismissed or not-a-task one, nor a standing
        task (scheduled), a proposal or a linked system's message — nor one
        an opt-in inbox made (its title is the inbox's text, which a
        suggestion on another task would outlive)."""
        with self._lock:
            mine = self._conn.execute(
                "SELECT party_id, kind, address FROM task_parties WHERE user_id = ? AND task_id = ? AND verified = 1 "
                "AND kind != 'name' AND NOT (kind = 'fd_user' AND address = ?)",
                (user_id, task["id"], user_id)).fetchall()
            ids = sorted({r["party_id"] for r in mine if r["party_id"]})
            raw = sorted({f"{r['kind']}:{r['address']}" for r in mine if not r["party_id"]})
            if not ids and not raw:
                return []
            same = []
            if ids:
                same.append(f"p.party_id IN ({_marks(ids)})")
            if raw:
                same.append(f"(p.party_id = '' AND p.verified = 1 AND (p.kind || ':' || p.address) IN ({_marks(raw)}))")
            created = float(task["created_at"] or 0)
            return [dict(r) for r in self._conn.execute(
                "SELECT DISTINCT t.* FROM tasks t JOIN task_parties p ON p.task_id = t.id AND p.user_id = t.user_id "
                "WHERE t.user_id = ? AND t.id != ? AND t.merged_into = '' AND t.status NOT IN ('not_task', 'dismissed') "
                f"AND t.kind NOT IN ('scheduled', 'proposal', 'link') AND t.source NOT IN ({_marks(PASSIVE_SOURCES)}) "
                "AND (t.created_at < ? OR (t.created_at = ? AND t.id < ?)) "
                f"AND COALESCE(t.last_event_at, t.created_at) >= ? AND ({' OR '.join(same)}) "
                "ORDER BY COALESCE(t.last_event_at, t.created_at) DESC, t.id LIMIT ?",
                (user_id, task["id"], *PASSIVE_SOURCES, created, created, task["id"], float(since), *ids, *raw,
                 max(1, int(limit))))]

    # People: parties and their handles, only ever on deterministic evidence.

    def _open_people(self) -> None:
        """On open: the indexes people are found by — and, once per board
        file, the people behind the parties recorded before v2 knew them."""
        with self._lock:
            for ddl in _PEOPLE_INDEXES:
                self._conn.execute(ddl)
            if self._conn.execute("SELECT 1 FROM source_cursors WHERE source = ?", (_PEOPLE_MARK,)).fetchone() is None:
                for r in self._conn.execute(
                        "SELECT * FROM task_parties WHERE party_id = '' AND verified = 1 ORDER BY rowid").fetchall():
                    self._link_party(r["user_id"], r["task_id"], dict(r))
                self._conn.execute("INSERT INTO source_cursors (source, cursor, updated_at) VALUES (?, 'done', ?)",
                                   (_PEOPLE_MARK, time.time()))
            self._conn.commit()

    def _link_party(self, user_id: str, task_id: str, p: dict) -> str:
        """The person behind one party row: the one a handle of it belongs to
        already — an exact handle match, never a name — or, for a recorded
        party (a header, the register, a bridge: ``verified``), a new person
        holding its handles. The handles one recorded row names are one
        person (a WhatsApp Web jid and the number in it). The board user's
        own handle is nobody's: they are "You". The person's id, or ""."""
        kind, address = str(p["kind"]), str(p["address"])
        if kind == "fd_user" and address == user_id:
            return ""
        handles = _handles(kind, address)
        if not handles:
            return ""
        recorded = bool(p.get("verified")) and kind in _HANDLE_KINDS
        held: list[str] = []
        for h in handles:
            r = self._conn.execute("SELECT party_id FROM party_handles WHERE user_id = ? AND kind = ? AND value = ?",
                                   (user_id, *h)).fetchone()
            if r and r[0] not in held:
                held.append(str(r[0]))
        if held:
            party_id = held[0]
            for other in held[1:] if recorded else ():
                self._join_parties(user_id, party_id, other)
        elif recorded:
            party_id = "pt_" + secrets.token_hex(8)
            name = _person_name(p.get("display")) or _address_name(kind, address)
            self._conn.execute("INSERT INTO parties (id, user_id, display_name, folded_name, kind) VALUES (?,?,?,?,?)",
                               (party_id, user_id, name, _folded(name), "org" if kind == "link" else "person"))
        else:
            return ""
        if recorded:
            for h in handles:
                self._conn.execute("INSERT OR IGNORE INTO party_handles (user_id, kind, value, party_id) "
                                   "VALUES (?,?,?,?)", (user_id, *h, party_id))
            self._name_party(user_id, party_id, p.get("display"))
        self._conn.execute(
            "UPDATE task_parties SET party_id = ? WHERE task_id = ? AND user_id = ? AND role = ? AND kind = ? "
            "AND address = ?", (party_id, task_id, user_id, p["role"], kind, address))
        return party_id

    def _name_party(self, user_id: str, party_id: str, name: Any, *, sure: bool = False) -> bool:
        """Call the person *name* — when it names someone (not an address or
        a stand-in) and they have no name yet, or *sure* (a deck user's own
        name, an agent's contact): a header or a bridge never renames them.
        Their tasks' searchable text follows. True when it changed."""
        new = _person_name(name)
        if not new:
            return False
        row = self._conn.execute("SELECT display_name FROM parties WHERE id = ? AND user_id = ?",
                                 (party_id, user_id)).fetchone()
        if row is None or row[0] == new or (not sure and _person_name(row[0])):
            return False
        self._conn.execute("UPDATE parties SET display_name = ?, folded_name = ? WHERE id = ? AND user_id = ?",
                           (new, _folded(new), party_id, user_id))
        for (task_id,) in self._conn.execute("SELECT DISTINCT task_id FROM task_parties WHERE user_id = ? "
                                             "AND party_id = ?", (user_id, party_id)).fetchall():
            self._refresh(task_id)
        return True

    def _join_parties(self, user_id: str, keep: str, other: str) -> None:
        """*other* is *keep* — a record holds a handle of each: one person,
        under *keep*, with every handle and party row of both."""
        if keep == other:
            return
        row = self._conn.execute("SELECT display_name FROM parties WHERE id = ? AND user_id = ?",
                                 (other, user_id)).fetchone()
        for table in ("party_handles", "task_parties"):
            self._conn.execute(f"UPDATE {table} SET party_id = ? WHERE user_id = ? AND party_id = ?",
                               (keep, user_id, other))
        self._conn.execute("DELETE FROM parties WHERE id = ? AND user_id = ?", (other, user_id))
        if row is not None:
            self._name_party(user_id, keep, row[0])  # a name it had, when this one has none

    def learn_party(self, user_id: str, party_id: str, *, name: str = "",
                    handles: tuple | list = ()) -> bool:
        """What a record the deck keeps says of one person for certain — a
        deck user's name and email, an agent's contact holding an email and a
        phone: *name* is theirs (over what a header or a bridge called them)
        and so are *handles*; one another person holds makes the two one, as
        the record does. True when anything changed."""
        changed = False
        with self._lock:
            if self._conn.execute("SELECT 1 FROM parties WHERE id = ? AND user_id = ?",
                                  (party_id, user_id)).fetchone() is None:
                return False
            for kind, value in handles:
                for h in _handles(str(kind), str(value or "")):
                    r = self._conn.execute("SELECT party_id FROM party_handles WHERE user_id = ? AND kind = ? "
                                           "AND value = ?", (user_id, *h)).fetchone()
                    if r is None:
                        self._conn.execute("INSERT INTO party_handles (user_id, kind, value, party_id) "
                                           "VALUES (?,?,?,?)", (user_id, *h, party_id))
                        changed = True
                    elif r[0] != party_id:
                        self._join_parties(user_id, party_id, str(r[0]))
                        changed = True
            changed |= self._name_party(user_id, party_id, name, sure=True)
            if changed:
                self._conn.commit()
        return changed

    def task_people(self, user_id: str, task_id: str) -> list[dict]:
        """The people on one task the board knows: ``{party_id, name,
        handles: [(kind, value)]}`` each."""
        with self._lock:
            out = []
            for (party_id,) in self._conn.execute(
                    "SELECT DISTINCT party_id FROM task_parties WHERE user_id = ? AND task_id = ? AND party_id != ''",
                    (user_id, task_id)).fetchall():
                row = self._conn.execute("SELECT display_name FROM parties WHERE id = ? AND user_id = ?",
                                         (party_id, user_id)).fetchone()
                if row is not None:
                    out.append({"party_id": party_id, "name": row[0], "handles": [
                        (h[0], h[1]) for h in self._conn.execute(
                            "SELECT kind, value FROM party_handles WHERE user_id = ? AND party_id = ? "
                            "ORDER BY kind, value", (user_id, party_id))]})
        return out

    def _with_merged(self, task_ids: list[str]) -> list[str]:
        """*task_ids* and every task merged into one of them (and into those):
        dropping a task drops what was merged into it — its events went
        there, and a hidden task with nothing left would never show again."""
        out, todo = list(dict.fromkeys(task_ids)), list(dict.fromkeys(task_ids))
        while todo:
            chunk, todo = todo[:500], todo[500:]
            for (child,) in self._conn.execute(
                    f"SELECT id FROM tasks WHERE merged_into IN ({_marks(chunk)})", chunk).fetchall():
                if child not in out:
                    out.append(child)
                    todo.append(child)
        return out

    def _forget_unused_people(self) -> None:
        """People no task names any more (its tasks dropped, an inbox switched
        off) are forgotten — names and handles — on every board."""
        self._conn.execute(
            "DELETE FROM parties WHERE NOT EXISTS (SELECT 1 FROM task_parties tp "
            "WHERE tp.user_id = parties.user_id AND tp.party_id = parties.id)")
        self._conn.execute(
            "DELETE FROM party_handles WHERE NOT EXISTS (SELECT 1 FROM parties p "
            "WHERE p.user_id = party_handles.user_id AND p.id = party_handles.party_id)")

    # ── Cleanup ───────────────────────────────────────────────────────

    def _drop_tasks(self, task_ids: list[str]) -> None:
        task_ids = self._with_merged(task_ids)  # v2: what was merged into them lived on them
        for i in range(0, len(task_ids), 500):
            chunk = task_ids[i:i + 500]
            marks = _marks(chunk)
            for table in ("task_links", "task_events", "task_decisions", "task_parties"):
                self._conn.execute(f"DELETE FROM {table} WHERE task_id IN ({marks})", chunk)
            self._conn.execute(f"UPDATE turns SET task_id = '' WHERE task_id IN ({marks})", chunk)
            self._conn.execute(f"DELETE FROM tasks WHERE id IN ({marks})", chunk)
            # v2: a "Same as …?" that named one of them names nothing now.
            self._conn.execute("UPDATE tasks SET flags = json_remove(flags, '$.same_as') "
                               f"WHERE json_extract(flags, '$.same_as.task_id') IN ({marks})", chunk)
        self._forget_unused_people()  # v2: nobody is kept who is on no task

    def delete_agent(self, agent_owner_id: str | None, slug: str) -> int:
        """Forget the tasks of an agent removed for good — its owner's on
        every board, members' included, and those recorded without an owner
        on the owner's own board (someone else's agent of the same name may
        have made those) — or everyone's with *agent_owner_id* None. Returns
        how many went."""
        if not slug:
            return 0
        with self._lock:
            if agent_owner_id is None:
                rows = self._conn.execute("SELECT id FROM tasks WHERE agent_slug = ?", (slug,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id FROM tasks WHERE agent_slug = ? "
                    "AND (agent_owner_id = ? OR (agent_owner_id = '' AND user_id = ?))",
                    (slug, agent_owner_id, agent_owner_id)).fetchall()
            ids = [r[0] for r in rows]
            self._drop_tasks(ids)
            self._conn.execute(
                "DELETE FROM turns WHERE agent_slug = ? AND (? IS NULL OR agent_owner_id = ? "
                "OR (agent_owner_id = '' AND user_id = ?))", (slug, agent_owner_id, agent_owner_id, agent_owner_id))
            self._conn.commit()
        return len(ids)

    def delete_user(self, user_id: str) -> int:
        """Forget everything on *user_id*'s board (the user was deleted)."""
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ?", (user_id,)).fetchone()[0]
            for table in ("task_links", "task_events", "task_decisions", "task_parties", "parties",
                          "party_handles", "task_rules", "turns", "taskboard_settings", "passive_marks", "tasks"):
                self._conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            # v2: their history's counts and their inboxes' read positions.
            self._conn.execute("DELETE FROM source_cursors WHERE source = ? OR (source LIKE 'passive:%' "
                               "AND substr(source, -length(?) - 1) = ':' || ?)",
                               (f"history:{user_id}", user_id, user_id))
            self._conn.commit()
        return int(n)


def _held(patch: dict, locks: dict) -> dict:
    """What of a source's *patch* the user's *locks* keep out."""
    held = {k: v for k, v in patch.items() if k != "flags" and _LOCK_GROUP.get(k, k) in locks}
    flags = {k: v for k, v in (patch.get("flags") or {}).items() if k in locks}
    if flags:
        held["flags"] = flags
    return held


def _correction(row: dict, action: str, value: Any, now: float, actor: str) -> tuple[dict, list[str], str]:
    """``(fields, locked fields, summary)`` for one correction; ValueError when
    it can't apply to *row*."""
    closing = {"closed_at": now, "closed_by": "user", "closed_by_party": actor[:120]}
    if action == "done":
        return {"status": "done", **closing}, ["status"], "Marked done"
    if action == "dismiss":
        return {"status": "dismissed", **closing}, ["status"], "Dismissed"
    if action == "not_task":
        return {"status": "not_task", **closing}, ["status"], "Marked not a task"
    if action == "reopen":
        return _reopened(row), ["status"], "Reopened"
    if action == "snooze":
        try:
            until = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("snooze needs a time (epoch seconds)") from None
        if until <= now:
            raise ValueError("snooze until a time still to come")
        if not math.isfinite(until) or until > now + _SNOOZE_MAX_S:
            raise ValueError("snooze for a year at most")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(until))
        return {"snooze_until": until}, ["snooze_until"], f"Snoozed until {when}"
    if action == "unsnooze":
        return {"snooze_until": None}, ["snooze_until"], "Snooze ended"
    if action == "rename":
        title = " ".join(str(value or "").split())[:160]
        if not title:
            raise ValueError("rename needs a title")
        return {"title": title}, ["title"], f"Renamed to “{title}”"
    if action == "owes_next":
        if value not in OWES:
            raise ValueError(f"owes_next is one of {', '.join(OWES)}")
        return {"owes_next": value}, ["owes_next"], f"Next move: {OWES_LABELS[value].lower()}"
    if action == "ack_failed":
        if row["status"] != "failed":
            raise ValueError("only a failed task is acknowledged")
        flags = {**_loads(row["flags"], {}), "failed_ack": now}
        return {"flags": flags}, ["failed_ack"], "Failure seen"
    if action == "confirm_task":
        # A task the board wasn't sure of: a maybe one (it goes to the agent),
        # or one admitted at low confidence (it counts in the badge now).
        flags = _loads(row["flags"], {})
        if row["status"] != "maybe" and not (row["status"] == "open" and flags.get("low_confidence")):
            raise ValueError("only a task the board wasn't sure of is confirmed")
        fields: dict[str, Any] = {"status": "open", "flags": {**flags, "low_confidence": False}}
        if row["status"] == "maybe":
            # v2: someone's message in the user's own inbox is theirs to answer.
            owes = "me" if row["kind"] == "message" else "agent"
            fields.update(owes_next=owes, closed_at=None, closed_by="", closed_by_party="", result="")
        return fields, ["status", "low_confidence"], "Confirmed as a task"
    if action == "not_done":
        flags = _loads(row["flags"], {})
        if not flags.get("probably_done") and row["status"] != "done":
            raise ValueError("only a task that looks done is marked not done")
        opened = _reopened(row) if row["status"] != "open" else {"status": "open"}
        fields = {**opened, "flags": {**(opened.get("flags") or flags), "probably_done": False}}
        return fields, ["status", "probably_done"], "Not done yet"
    raise ValueError(f"unknown action: {action}")


def _reopened(row: dict) -> dict:
    """The fields that open *row* again: who closed it and how cleared, and —
    when its close left nobody owing a move (a source's or a machine pass's
    close does) — the next move the user's: they just said it isn't
    finished, so it comes back in Needs me rather than in no lane at all.
    What came in after it was done (``new_since_done``) is what they open it
    for: seen, so a later close doesn't say it again."""
    fields: dict[str, Any] = {"status": "open", "closed_at": None, "closed_by": "", "closed_by_party": "",
                              "result": ""}
    if row["status"] != "open" and row["owes_next"] == "nobody":
        fields["owes_next"] = "me"
    flags = _loads(row["flags"], {})
    if flags.get("new_since_done"):
        fields["flags"] = {**flags, "new_since_done": False}
    return fields


# ── v2 threads and people (helpers of the section in TaskboardStore) ──


class ThreadConflict(ValueError):  # noqa: N818 — reads as the condition
    """A merge, unmerge or split the tasks' state doesn't allow now (a 409)."""


_MERGE_DEPTH = 32         # merges into merged tasks followed this far (a guard against a cycle)
_TRIAGED_KINDS = ("request", "sent", "message")  # the kinds triage reads (taskboard_triage._TRIAGED)
# A thread note's refs that name another task, and each note told without
# the titles it quotes (an inbox's text taken off the board, _passive_scrub).
_NOTE_REFS = ("merged", "into", "from", "task", "new_task")
_THREAD_PLAIN = {"merge": "Merged another task into this one", "merged_into": "Merged into another task",
                 "unmerge": "Taken back out of another task", "unmerged": "Took a merged task back out",
                 "split": "Split some of it off as a task of its own", "split_from": "Split off another task",
                 "not_same": "Not the same as the task Jev suggested"}
_PEOPLE_MARK = "v2:people"  # source_cursors: the parties recorded before v2 were looked at
_PEOPLE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_task_parties_party ON task_parties(user_id, party_id)",
    "CREATE INDEX IF NOT EXISTS idx_handles_party ON party_handles(user_id, party_id)",
)
# What a recorded party is someone by (a header's address, a bridge's number,
# jid or id, a deck user, a linked system); a model's ``phone`` only finds one.
_HANDLE_KINDS = ("email", "waid", "wa_jid", "tg_id", "fd_user", "link")
_PHONE_JID_RE = re.compile(r"^(\d{6,20})@(?:s\.whatsapp\.net|c\.us)$")
_ADDRESSY_RE = re.compile(r"@|^[+\d\s().-]+$")
_STAND_INS = ("you", "member", "someone", "someone else")


def _handles(kind: str, address: str) -> list[tuple[str, str]]:
    """The handles a party's address is: ``(kind, value)``, normalised — an
    email lower-cased, a number its digits, a WhatsApp Web jid of a phone
    also that number. None for a name, a group (a negative Telegram id) or
    a model's Telegram handle."""
    a = str(address or "").strip()
    if kind == "email":
        a = a.lower()
        return [("email", a)] if "@" in a else []
    if kind in ("waid", "phone"):
        digits = re.sub(r"\D", "", a)
        if kind == "phone":
            digits = digits[2:] if digits.startswith("00") else digits
            return [("waid", digits)] if len(digits) >= 7 else []
        return [("waid", digits)] if digits else []
    if kind == "wa_jid":
        a = a.lower()
        m = _PHONE_JID_RE.match(a)
        return ([("wa_jid", a)] + ([("waid", m.group(1))] if m else [])) if a else []
    if kind == "tg_id":
        return [("tg_id", a)] if re.fullmatch(r"\d{1,20}", a) else []
    if kind in ("fd_user", "link"):
        return [(kind, a)] if a else []
    return []


def _person_name(name: Any) -> str:
    """*name* when it names someone — not an address, a number, or a stand-in
    ("You", "Member") — else ""."""
    text = " ".join(str(name or "").split())[:160]
    if not text or _ADDRESSY_RE.search(text) or _folded(text) in _STAND_INS:
        return ""
    return text


def _address_name(kind: str, address: str) -> str:
    """What to call someone known only by an address (never a deck user's id)."""
    if kind == "waid":
        return f"+{address}"
    m = _PHONE_JID_RE.match(address.lower()) if kind == "wa_jid" else None
    if m:
        return f"+{m.group(1)}"
    return "Someone else" if kind == "fd_user" else address[:160]


def _folded(text: Any) -> str:
    return " ".join(_fold(str(text or "")).split())


def _same_id(flags: dict) -> str:
    same = flags.get("same_as")
    return str(same.get("task_id") or "") if isinstance(same, dict) else ""


# An event's refs, and the link each names (a split moves a link with the events that name it).
_EVENT_KEY_REFS = (("turn_id", "turn:"), ("autonomy_id", "autonomy:"), ("follow_up_id", "fu:"),
                   ("autoreply_id", "autoreply:"), ("thread_id", "gmail:thread:"), ("gmail_thread_id", "gmail:thread:"),
                   ("message_id", "gmail:msg:"), ("gmail_message_id", "gmail:msg:"))


def _event_keys(ev: dict) -> set[str]:
    """The links an event names by its refs: its turn's, its email's (thread
    and message), its auto-reply's, follow-up's and action's, a linked
    system's message."""
    refs = ev["refs"] if isinstance(ev.get("refs"), dict) else _loads(ev.get("refs"), {})
    keys = {f"{prefix}{refs[field]}" for field, prefix in _EVENT_KEY_REFS
            if isinstance(refs.get(field), str | int) and not isinstance(refs.get(field), bool) and str(refs[field])}
    if refs.get("link_id") and refs.get("re"):
        keys.add(f"link:{refs['link_id']}:{refs['re']}")
    return keys


def _requeued(row: dict, now: float, text: bool) -> dict:
    """What puts a task whose events changed under it (a merge, an unmerge,
    a split) back in the triage queue: a person's request or message that
    triage reads (never one a mute rule keeps quiet, nor one sorted off the
    board) — read again from its messages when *text* (messages came or
    went)."""
    if (row["kind"] not in _TRIAGED_KINDS or row["triage_state"] == "skipped"
            or row["status"] in ("not_task", "dismissed")):
        return {}
    return {"triage_state": "pending", "triage_after": now, **({"extracted_fp": ""} if text else {})}


def utc_day(now: float | None = None) -> str:
    """The UTC day, ``YYYY-MM-DD`` — what the daily caps count under (as the
    plan's own allowances do)."""
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() if now is None else now))


_STORE: TaskboardStore | None = None
_STORE_LOCK = threading.Lock()
_DATA_DIR: Path | None = None


def configure(*, data_dir: Path) -> None:
    global _DATA_DIR
    _DATA_DIR = Path(data_dir)


def data_dir() -> Path:
    if _DATA_DIR is not None:
        return _DATA_DIR
    from captain_claw.flight_deck.server import DATA_DIR
    return DATA_DIR


def get_store() -> TaskboardStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = TaskboardStore(data_dir() / "taskboard.db")
        return _STORE


def set_store(value: TaskboardStore | None) -> None:
    """Tests: point the board at another store."""
    global _STORE
    with _STORE_LOCK:
        _STORE = value
