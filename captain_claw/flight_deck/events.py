"""Event-ingestion spine (see docs/jarvis-actions-events-plan.md, #2).

A dedicated, normalized store for real-world signals — calendar changes, new mail,
webhooks — so the autonomous loop reacts to the *user's world*, not just to its own
reflections. Source adapters (pollers / webhook receivers) normalize signals into
``external_events``; the Arbiter reads ``new`` events as candidates and marks them
``surfaced``. Self-owned DB (matches the per-subsystem convention), WAL, status-flag
idempotency (``new → surfaced → acted | ignored``), dedup via ``dedup_key``.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

EVENT_STATUSES = ("new", "surfaced", "acted", "ignored")


def _norm_user(user_id: str | None) -> str:
    return (user_id or "").strip() or "local"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "events.db"
    return Path("~/.captain-claw/events.db").expanduser()


class EventsStore:
    """SQLite store for external events. Sync sqlite3 + lock, mirroring the other
    Flight Deck subsystem stores."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS external_events (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    source      TEXT NOT NULL,            -- gmail | calendar | webhook | manual | …
                    event_type  TEXT NOT NULL DEFAULT '', -- new_email | event_changed | …
                    summary     TEXT NOT NULL DEFAULT '', -- human-readable headline
                    body        TEXT NOT NULL DEFAULT '', -- full content
                    metadata    TEXT NOT NULL DEFAULT '{}',
                    dedup_key   TEXT NOT NULL DEFAULT '', -- source id; prevents re-ingest
                    status      TEXT NOT NULL DEFAULT 'new',
                    ingested_at TEXT NOT NULL,
                    processed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_user
                    ON external_events(user_id, status, ingested_at DESC);
                -- Dedup only applies when a dedup_key is provided.
                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup
                    ON external_events(user_id, source, dedup_key)
                    WHERE dedup_key != '';

                -- Per-(user, source) poll state: when we last polled + an opaque
                -- cursor (calendar syncToken, gmail historyId, …).
                CREATE TABLE IF NOT EXISTS poll_state (
                    user_id      TEXT NOT NULL,
                    source       TEXT NOT NULL,
                    last_poll_at REAL NOT NULL DEFAULT 0,
                    cursor       TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (user_id, source)
                );
                """
            )
            self._c().commit()

    # ── poll state (for source adapters) ───────────────────────────────

    def get_poll_state(self, user_id: str, source: str) -> dict[str, Any]:
        uid = _norm_user(user_id)
        with self._lock:
            r = self._c().execute(
                "SELECT last_poll_at, cursor FROM poll_state WHERE user_id = ? AND source = ?",
                (uid, source),
            ).fetchone()
        return {"last_poll_at": float(r["last_poll_at"]) if r else 0.0,
                "cursor": (r["cursor"] if r else "") or ""}

    def set_poll_state(self, user_id: str, source: str, *, last_poll_at: float, cursor: str | None = None) -> None:
        uid = _norm_user(user_id)
        with self._lock:
            conn = self._c()
            existing = self.get_poll_state(uid, source)
            cur = existing["cursor"] if cursor is None else cursor
            conn.execute(
                "INSERT INTO poll_state (user_id, source, last_poll_at, cursor)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(user_id, source) DO UPDATE SET"
                " last_poll_at = excluded.last_poll_at, cursor = excluded.cursor",
                (uid, source, last_poll_at, cur),
            )
            conn.commit()

    def add_event(
        self,
        user_id: str,
        *,
        source: str,
        event_type: str = "",
        summary: str = "",
        body: str = "",
        metadata: dict[str, Any] | None = None,
        dedup_key: str = "",
    ) -> dict[str, Any] | None:
        """Insert one event. With a ``dedup_key`` already seen (same user+source),
        it's a no-op returning None; otherwise returns the new row."""
        uid = _norm_user(user_id)
        eid = "evt_" + secrets.token_hex(8)
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "INSERT OR IGNORE INTO external_events"
                " (id, user_id, source, event_type, summary, body, metadata, dedup_key,"
                "  status, ingested_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)",
                (eid, uid, source, event_type, summary[:500], body[:8000],
                 json.dumps(metadata or {}), dedup_key, now),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None  # deduped
        return self.get_event(eid)

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._c().execute(
                "SELECT * FROM external_events WHERE id = ?", (event_id,)
            ).fetchone()
        return self._row(r) if r else None

    def list_new(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Oldest-first unprocessed events — the arbiter's intake queue."""
        uid = _norm_user(user_id)
        limit = max(1, min(100, limit))
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM external_events WHERE user_id = ? AND status = 'new'"
                " ORDER BY ingested_at ASC LIMIT ?",
                (uid, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def list_events(self, user_id: str, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        uid = _norm_user(user_id)
        limit = max(1, min(500, limit))
        with self._lock:
            if status:
                rows = self._c().execute(
                    "SELECT * FROM external_events WHERE user_id = ? AND status = ?"
                    " ORDER BY ingested_at DESC LIMIT ?",
                    (uid, status, limit),
                ).fetchall()
            else:
                rows = self._c().execute(
                    "SELECT * FROM external_events WHERE user_id = ?"
                    " ORDER BY ingested_at DESC LIMIT ?",
                    (uid, limit),
                ).fetchall()
        return [self._row(r) for r in rows]

    def mark(self, event_ids: list[str], status: str) -> None:
        if not event_ids or status not in EVENT_STATUSES:
            return
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            conn.executemany(
                "UPDATE external_events SET status = ?, processed_at = ? WHERE id = ?",
                [(status, now, eid) for eid in event_ids],
            )
            conn.commit()

    def cleanup(self, older_than_days: int = 30) -> int:
        """Drop processed (surfaced/acted/ignored) events older than N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, older_than_days))).isoformat()
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "DELETE FROM external_events WHERE status != 'new' AND ingested_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (ValueError, TypeError):
            d["metadata"] = {}
        return d


_STORE: EventsStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> EventsStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = EventsStore()
        return _STORE
