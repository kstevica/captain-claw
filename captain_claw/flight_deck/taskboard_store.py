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
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO task_parties (task_id, user_id, role, kind, address, display, verified) "
                "VALUES (?,?,?,?,?,?,?)",
                (task_id, user_id, role, str(p.get("kind") or "email"), address[:320],
                 str(p.get("display") or "")[:160], int(bool(p.get("verified")))))
            added += int((cur.rowcount or 0) == 1)
        return added

    def _refresh(self, task_id: str) -> None:
        """Fold the task's searchable text again: title, ask, result, agent,
        channels, kind and parties."""
        row = self._row(task_id)
        if row is None:
            return
        parts = [row["title"], row["ask"], row["result_text"], row["agent_name"], row["agent_slug"],
                 KIND_LABELS.get(row["kind"], row["kind"])]
        for ch in _loads(row["channels"], []) or [row["channel"]]:
            parts.append(CHANNEL_LABELS.get(str(ch).partition(":")[0], str(ch)))
        for p in self._conn.execute(
                "SELECT display, address FROM task_parties WHERE task_id = ?", (task_id,)):
            parts.extend((p["display"], p["address"]))
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
            rows = self._conn.execute(
                f"SELECT * FROM task_parties WHERE user_id = ? AND task_id IN ({_marks(task_ids)})",
                (user_id, *task_ids)).fetchall()
        for r in sorted(rows, key=lambda r: (_PARTY_ORDER.get(r["role"], 9), r["display"] or r["address"])):
            out.setdefault(r["task_id"], []).append({
                "role": r["role"], "kind": r["kind"], "address": r["address"],
                "display": r["display"], "party_id": r["party_id"], "verified": bool(r["verified"])})
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
            where.append(
                "EXISTS (SELECT 1 FROM task_parties p WHERE p.task_id = t.id AND p.user_id = t.user_id "
                f"AND (p.kind || ':' || p.address) IN ({_marks(parties)}))")
            args += parties
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
    ) -> dict:
        """One page of the user's tasks in *view*, searched and filtered, with
        each facet's counts under the other filters and each view's count
        under all of them. Rows carry their computed ``lane``."""
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
            facets: dict[str, list[tuple[str, str, int]]] = {}
            for dim, col in _FACET_COLUMNS.items():
                w, a = self._where(user_id, **common, skip=dim)
                label = ", MAX(t.agent_name)" if dim == "agent" else ", ''"
                facets[dim] = [(r[0], r[1], int(r[2])) for r in self._conn.execute(
                    f"SELECT t.{col}{label}, COUNT(*) FROM tasks t WHERE {w} GROUP BY t.{col} "
                    "ORDER BY COUNT(*) DESC", a)]
            w, a = self._where(user_id, **common, skip="party")
            facets["party"] = [(f"{r[0]}:{r[1]}", r[2] or r[1], int(r[3])) for r in self._conn.execute(
                "SELECT p.kind, p.address, MAX(p.display), COUNT(DISTINCT t.id) FROM tasks t "
                f"JOIN task_parties p ON p.task_id = t.id AND p.user_id = t.user_id WHERE {w} "
                "GROUP BY p.kind, p.address ORDER BY COUNT(DISTINCT t.id) DESC LIMIT 50", a)]
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
        confidence (Jev 0.65–0.8) — those wait in the list, unannounced."""
        needs, args = self._needs_me(time.time() if now is None else now, list(proposal_tasks or []))
        with self._lock:
            return int(self._conn.execute(
                f"SELECT COUNT(*) FROM tasks t WHERE t.user_id = ? AND t.merged_into = '' AND {needs} "
                "AND COALESCE(json_extract(t.flags, '$.low_confidence'), 0) = 0", (user_id, *args)).fetchone()[0])

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

    # ── Cleanup ───────────────────────────────────────────────────────

    def _drop_tasks(self, task_ids: list[str]) -> None:
        for i in range(0, len(task_ids), 500):
            chunk = task_ids[i:i + 500]
            marks = _marks(chunk)
            for table in ("task_links", "task_events", "task_decisions", "task_parties"):
                self._conn.execute(f"DELETE FROM {table} WHERE task_id IN ({marks})", chunk)
            self._conn.execute(f"UPDATE turns SET task_id = '' WHERE task_id IN ({marks})", chunk)
            self._conn.execute(f"DELETE FROM tasks WHERE id IN ({marks})", chunk)

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
                          "party_handles", "task_rules", "turns", "taskboard_settings", "tasks"):
                self._conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
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
            fields.update(owes_next="agent", closed_at=None, closed_by="", closed_by_party="", result="")
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
