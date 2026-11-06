"""The external-memory outbox: mirrored items waiting to be written.

A dedicated SQLite DB (``external_memory.db``) in Flight Deck's data dir,
deck-isolated via ``FD_DATA_DIR`` like ``tool_audit.db``. An agent's mirror
call only queues a row here and returns at once; the drainer
(:func:`service.drain_loop`) writes rows to the owner's provider one at a time,
so a slow or rate-limited service never holds up an agent, and nothing queued
is lost to a restart.

One row per ``(owner, kind, source_id)``: the same insight sent twice is
queued once. States: ``pending`` (``next_at`` says when to try), ``sent``,
``failed`` (won't be retried until the user asks). Sent rows are kept a while
for the "recently synced" list, then pruned.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

SENT_RETENTION_S = 30 * 86400
MAX_ATTEMPTS = 12

_DDL = """
CREATE TABLE IF NOT EXISTS outbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id    TEXT NOT NULL,
    provider    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    agent       TEXT NOT NULL DEFAULT '',
    title       TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    next_at     REAL NOT NULL DEFAULT 0,
    last_error  TEXT NOT NULL DEFAULT '',
    remote_id   TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    sent_at     REAL,
    UNIQUE (owner_id, kind, source_id)
);
CREATE INDEX IF NOT EXISTS idx_outbox_due ON outbox(state, next_at);
CREATE INDEX IF NOT EXISTS idx_outbox_owner ON outbox(owner_id, state, id);
"""


class Outbox:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(str(self.path))
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_DDL)
            await db.commit()
            self._db = db
        return self._db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def enqueue(
        self, owner_id: str, provider: str, kind: str, source_id: str, payload: dict[str, Any],
        *, agent: str = "", title: str = "",
    ) -> bool:
        """Queue one item; False when this (owner, kind, source_id) is queued
        or sent already."""
        async with self._lock:
            db = await self._conn()
            cur = await db.execute(
                """INSERT OR IGNORE INTO outbox
                   (owner_id, provider, kind, source_id, agent, title, payload, created_at, next_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (owner_id, provider, kind, source_id, agent[:200], title[:300],
                 json.dumps(payload, ensure_ascii=False), time.time()),
            )
            await db.commit()
            return cur.rowcount > 0

    async def due(self, limit: int = 20, now: float | None = None) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT * FROM outbox WHERE state = 'pending' AND next_at <= ? ORDER BY id LIMIT ?",
            (time.time() if now is None else now, limit),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def mark_sent(self, row_id: int, remote_id: str = "") -> None:
        async with self._lock:
            db = await self._conn()
            await db.execute(
                "UPDATE outbox SET state='sent', sent_at=?, remote_id=?, last_error='', attempts=attempts+1 "
                "WHERE id=?",
                (time.time(), remote_id[:200], row_id),
            )
            await db.commit()

    async def mark_retry(self, row_id: int, error: str, delay_s: float) -> str:
        """Try again after *delay_s*; past MAX_ATTEMPTS the row fails.
        Returns the row's new state."""
        async with self._lock:
            db = await self._conn()
            cur = await db.execute("SELECT attempts FROM outbox WHERE id=?", (row_id,))
            row = await cur.fetchone()
            attempts = (int(row["attempts"]) if row else 0) + 1
            state = "failed" if attempts >= MAX_ATTEMPTS else "pending"
            await db.execute(
                "UPDATE outbox SET state=?, attempts=?, next_at=?, last_error=? WHERE id=?",
                (state, attempts, time.time() + max(0.0, delay_s), error[:500], row_id),
            )
            await db.commit()
            return state

    async def mark_failed(self, row_id: int, error: str) -> None:
        async with self._lock:
            db = await self._conn()
            await db.execute(
                "UPDATE outbox SET state='failed', attempts=attempts+1, last_error=? WHERE id=?",
                (error[:500], row_id),
            )
            await db.commit()

    async def defer_owner(self, owner_id: str, delay_s: float, error: str) -> int:
        """Hold every pending row of *owner_id* (key refused, quota full):
        no attempt is spent — they go once the cause is fixed."""
        async with self._lock:
            db = await self._conn()
            cur = await db.execute(
                "UPDATE outbox SET next_at=?, last_error=? WHERE owner_id=? AND state='pending'",
                (time.time() + delay_s, error[:500], owner_id),
            )
            await db.commit()
            return cur.rowcount

    async def retry_owner(self, owner_id: str) -> int:
        """Failed and deferred rows of *owner_id* go again now."""
        async with self._lock:
            db = await self._conn()
            cur = await db.execute(
                "UPDATE outbox SET state='pending', next_at=0, attempts=CASE WHEN state='failed' THEN 0 "
                "ELSE attempts END WHERE owner_id=? AND state IN ('failed', 'pending')",
                (owner_id,),
            )
            await db.commit()
            return cur.rowcount

    async def drop_owner(self, owner_id: str, *, pending_only: bool = True) -> int:
        """Forget *owner_id*'s queue (they disconnected): what hasn't gone
        never will — and all of it with ``pending_only=False``."""
        async with self._lock:
            db = await self._conn()
            if pending_only:
                cur = await db.execute(
                    "DELETE FROM outbox WHERE owner_id=? AND state IN ('pending', 'failed')", (owner_id,),
                )
            else:
                cur = await db.execute("DELETE FROM outbox WHERE owner_id=?", (owner_id,))
            await db.commit()
            return cur.rowcount

    async def counts(self, owner_id: str) -> dict[str, Any]:
        db = await self._conn()
        cur = await db.execute(
            "SELECT state, COUNT(*) AS n FROM outbox WHERE owner_id=? GROUP BY state", (owner_id,),
        )
        out: dict[str, Any] = {"pending": 0, "sent": 0, "failed": 0}
        for r in await cur.fetchall():
            out[str(r["state"])] = int(r["n"])
        cur = await db.execute(
            "SELECT MAX(sent_at) AS t FROM outbox WHERE owner_id=? AND state='sent'", (owner_id,),
        )
        row = await cur.fetchone()
        out["last_sent_at"] = float(row["t"]) if row and row["t"] else None
        cur = await db.execute(
            "SELECT last_error FROM outbox WHERE owner_id=? AND state IN ('pending','failed') "
            "AND last_error != '' ORDER BY id DESC LIMIT 1",
            (owner_id,),
        )
        row = await cur.fetchone()
        out["last_error"] = str(row["last_error"]) if row else ""
        return out

    async def recent(self, owner_id: str, limit: int = 30, state: str = "") -> list[dict[str, Any]]:
        db = await self._conn()
        if state:
            cur = await db.execute(
                "SELECT id, kind, source_id, agent, title, state, attempts, last_error, remote_id, created_at, "
                "sent_at FROM outbox WHERE owner_id=? AND state=? ORDER BY id DESC LIMIT ?",
                (owner_id, state, limit),
            )
        else:
            cur = await db.execute(
                "SELECT id, kind, source_id, agent, title, state, attempts, last_error, remote_id, created_at, "
                "sent_at FROM outbox WHERE owner_id=? ORDER BY id DESC LIMIT ?",
                (owner_id, limit),
            )
        return [dict(r) for r in await cur.fetchall()]

    async def prune(self, now: float | None = None) -> int:
        async with self._lock:
            db = await self._conn()
            cutoff = (time.time() if now is None else now) - SENT_RETENTION_S
            cur = await db.execute("DELETE FROM outbox WHERE state='sent' AND sent_at < ?", (cutoff,))
            await db.commit()
            return cur.rowcount
