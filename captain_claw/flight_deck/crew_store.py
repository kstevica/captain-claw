"""Crew — where its work is kept. ``crew.db`` holds one row per task; each task's
folder under ``<data>/crew/<user>/`` holds what its CLI printed, turn by turn,
and the events read from that. Everything needed to pick a task up again is on
disk, so a Flight Deck restart loses nothing.

Every read and write names the user: someone else's task doesn't exist.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

ACTIVE = ("preparing", "queued", "running")
# Can't take another turn: its worktree is gone.
FINISHED = ("delivered", "closed")

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("id", "TEXT PRIMARY KEY"), ("user_id", "TEXT NOT NULL"), ("title", "TEXT NOT NULL"),
    ("kind", "TEXT NOT NULL"),  # ship · scout · clone (a repository got ready, with no worker) · setup (its environment)
    ("harness", "TEXT NOT NULL"), ("model", "TEXT DEFAULT ''"), ("effort", "TEXT DEFAULT ''"),
    ("model_used", "TEXT DEFAULT ''"),  # the one the CLI says it ran on; `model` is only what was asked for
    ("brief", "TEXT NOT NULL"),
    # preparing · queued · running · waiting · ready · failed · stopped · pr_open · delivered · closed
    ("status", "TEXT NOT NULL"), ("note", "TEXT DEFAULT ''"),
    ("repo", "TEXT NOT NULL"), ("repo_label", "TEXT DEFAULT ''"),
    ("repo_kind", "TEXT DEFAULT ''"),  # project · url · path
    ("repo_url", "TEXT DEFAULT ''"),  # the address, when it came from one: a restart mid-clone starts from it
    ("base_branch", "TEXT DEFAULT ''"), ("base_sha", "TEXT DEFAULT ''"),
    ("branch", "TEXT DEFAULT ''"), ("worktree", "TEXT DEFAULT ''"), ("git_common_dir", "TEXT DEFAULT ''"),
    ("session", "TEXT DEFAULT ''"), ("turn", "INTEGER DEFAULT 0"),
    ("pid", "INTEGER DEFAULT 0"), ("pid_start", "TEXT DEFAULT ''"),
    ("offset", "INTEGER DEFAULT 0"), ("turn_state", "TEXT DEFAULT '{}'"),
    ("events", "INTEGER DEFAULT 0"), ("pending", "TEXT DEFAULT '[]'"),
    ("final", "TEXT DEFAULT ''"), ("evidence", "TEXT DEFAULT '{}'"),
    ("tokens_in", "INTEGER DEFAULT 0"), ("tokens_out", "INTEGER DEFAULT 0"), ("cost_usd", "REAL DEFAULT 0"),
    ("deliver", "TEXT DEFAULT 'branch'"),  # branch · pr
    ("delivery", "TEXT DEFAULT '{}'"),
    ("results", "TEXT DEFAULT '{}'"),  # its files, copied to the owner's VFS (crew_runner._export_results)
    ("env", "TEXT DEFAULT '{}'"),  # what it was given to run and test in (crew_env.give_to_task)
    ("agent_slug", "TEXT DEFAULT ''"), ("source_host", "TEXT DEFAULT ''"), ("origin", "TEXT DEFAULT '{}'"),
    ("last_event_at", "REAL DEFAULT 0"), ("turn_started_at", "REAL DEFAULT 0"), ("stalled_at", "REAL DEFAULT 0"),
    ("created_at", "REAL NOT NULL"), ("updated_at", "REAL NOT NULL"), ("closed_at", "REAL DEFAULT 0"),
    ("archived_at", "REAL DEFAULT 0"),  # tidied off the page; its row and its results stay
)
_NAMES = {name for name, _ in _COLUMNS}
_JSON = {"pending": [], "evidence": {}, "delivery": {}, "results": {}, "origin": {}, "env": {}}

_ROOT: Path | None = None
_STORE: "CrewStore | None" = None
_STORE_LOCK = threading.Lock()


def root() -> Path:
    """``<data>/crew`` — this deck's own (each deck has its own FD_DATA_DIR)."""
    if _ROOT is not None:
        return _ROOT
    from captain_claw.flight_deck.server import DATA_DIR

    return Path(DATA_DIR) / "crew"


def set_root(path: Path | None) -> None:
    """For tests: another folder, and a store opened on it."""
    global _ROOT, _STORE
    _ROOT = path
    _STORE = None


def safe_uid(user_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", user_id or "local")[:80] or "local"


def user_dir(user_id: str) -> Path:
    path = root() / safe_uid(user_id)
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def task_dir(user_id: str, task_id: str) -> Path:
    path = user_dir(user_id) / "tasks" / task_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def turn_dir(user_id: str, task_id: str, turn: int) -> Path:
    path = task_dir(user_id, task_id) / "turns" / str(turn)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for name, default in _JSON.items():
        try:
            out[name] = json.loads(out.get(name) or "null") or default
        except (TypeError, ValueError):
            out[name] = default
    return out


class CrewStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        cols = ", ".join(f"{name} {spec}" for name, spec in _COLUMNS)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS tasks ({cols})")
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(tasks)")}
        for name, spec in _COLUMNS:
            if name not in have:
                self._conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {spec}")
        # What each CLI last said about itself: per user, since each has their own account.
        self._conn.execute("CREATE TABLE IF NOT EXISTS seen (user_id TEXT NOT NULL, harness TEXT NOT NULL, "
                           "model TEXT DEFAULT '', limits TEXT DEFAULT '{}', at REAL DEFAULT 0, "
                           "PRIMARY KEY (user_id, harness))")
        self._conn.execute("CREATE INDEX IF NOT EXISTS tasks_user ON tasks (user_id, created_at)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS tasks_status ON tasks (status)")
        self._conn.commit()

    def create(self, user_id: str, **fields: Any) -> dict[str, Any]:
        now = time.time()
        row = {k: self._encode(k, v) for k, v in fields.items() if k in _NAMES}
        row.update(id="crw_" + secrets.token_hex(6), user_id=user_id, created_at=now, updated_at=now)
        row.setdefault("status", "queued")
        marks = ", ".join("?" for _ in row)
        with self._lock:
            self._conn.execute(f"INSERT INTO tasks ({', '.join(row)}) VALUES ({marks})", tuple(row.values()))
            self._conn.commit()
        return self.get(user_id, row["id"]) or {}

    @staticmethod
    def _encode(name: str, value: Any) -> Any:
        return json.dumps(value, ensure_ascii=False) if name in _JSON else value

    def update(self, task_id: str, **fields: Any) -> None:
        row = {k: self._encode(k, v) for k, v in fields.items() if k in _NAMES}
        if not row:
            return
        row["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in row)
        with self._lock:
            self._conn.execute(f"UPDATE tasks SET {sets} WHERE id = ?", (*row.values(), task_id))
            self._conn.commit()

    def get(self, user_id: str, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tasks WHERE id = ? AND user_id = ?",
                                     (task_id, user_id)).fetchone()
        return _decode(row) if row else None

    def list_for(self, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE user_id = ? AND archived_at = 0 ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)).fetchall()
        return [_decode(r) for r in rows]

    def with_status(self, *statuses: str) -> list[dict[str, Any]]:
        marks = ", ".join("?" for _ in statuses)
        with self._lock:
            rows = self._conn.execute(f"SELECT * FROM tasks WHERE status IN ({marks}) ORDER BY created_at",
                                      statuses).fetchall()
        return [_decode(r) for r in rows]

    def running_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, COUNT(*) AS n FROM tasks WHERE status = 'running' GROUP BY user_id").fetchall()
        return {r["user_id"]: r["n"] for r in rows}

    def recent_repos(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT repo, repo_label, repo_kind, repo_url, MAX(created_at) AS at FROM tasks WHERE user_id = ?"
                " AND repo_kind IN ('url', 'path') GROUP BY repo ORDER BY at DESC LIMIT ?",
                (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def note_seen(self, user_id: str, harness: str, *, model: str = "", limits: dict | None = None) -> None:
        with self._lock:
            row = self._conn.execute("SELECT model, limits FROM seen WHERE user_id = ? AND harness = ?",
                                     (user_id, harness)).fetchone()
            merged = json.loads(row["limits"] or "{}") if row else {}
            merged.update(limits or {})
            self._conn.execute(
                "INSERT OR REPLACE INTO seen (user_id, harness, model, limits, at) VALUES (?, ?, ?, ?, ?)",
                (user_id, harness, model or (row["model"] if row else ""), json.dumps(merged), time.time()))
            self._conn.commit()

    def seen_for(self, user_id: str) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT harness, model, limits, at FROM seen WHERE user_id = ?", (user_id,)).fetchall()
        return {r["harness"]: {"model": r["model"], "limits": json.loads(r["limits"] or "{}"), "at": r["at"]} for r in rows}

    def forget_seen(self, user_id: str, harness: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM seen WHERE user_id = ? AND harness = ?", (user_id, harness))
            self._conn.commit()

    def usage_since(self, user_id: str, since: float) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT harness, COUNT(*) AS tasks, SUM(tokens_in + tokens_out) AS tokens, SUM(cost_usd) AS cost FROM tasks"
                " WHERE user_id = ? AND updated_at >= ? GROUP BY harness", (user_id, since)).fetchall()
        return {r["harness"]: {"tasks": r["tasks"], "tokens": r["tokens"] or 0, "cost_usd": round(r["cost"] or 0.0, 4)}
                for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def get_store() -> CrewStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = CrewStore(root() / "crew.db")
        return _STORE


def append_events(user_id: str, task_id: str, events: list[dict]) -> None:
    if not events:
        return
    path = task_dir(user_id, task_id) / "events.jsonl"
    with path.open("a", encoding="utf-8") as out:
        for event in events:
            out.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_events(user_id: str, task_id: str, after: int = 0, limit: int = 500) -> list[dict]:
    """Events numbered from 1; *after* is the last one the caller has."""
    path = task_dir(user_id, task_id) / "events.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as lines:
        for number, line in enumerate(lines, 1):
            if number <= after:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            event["n"] = number
            out.append(event)
            if len(out) >= limit:
                break
    return out
