"""The external-memory outbox: mirrored items waiting to be written.

A dedicated SQLite DB (``external_memory.db``) in Flight Deck's data dir,
deck-isolated via ``FD_DATA_DIR`` like ``tool_audit.db``. An agent's mirror
call only queues a row here and returns at once; the drainer
(:func:`service.drain_loop`) writes rows to the owner's provider one at a time,
so a slow or rate-limited service never holds up an agent, and nothing queued
is lost to a restart.

One row per ``(owner, target, kind, source_id)``: ``target`` is the memory
it goes to (a person can have several, each taking the kinds they chose), and
the same insight sent twice is queued once per memory. Rows from before there
were several memories have an empty target and the provider's name; the
drainer sends them to that person's memory of that provider. States: ``pending`` (``next_at`` says when to try), ``sent``,
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
    target      TEXT NOT NULL DEFAULT '',
    UNIQUE (owner_id, target, kind, source_id)
);
CREATE INDEX IF NOT EXISTS idx_outbox_due ON outbox(state, next_at);
CREATE INDEX IF NOT EXISTS idx_outbox_owner ON outbox(owner_id, state, id);
"""

# A table from before ``target``: its UNIQUE (owner, kind, source_id) would
# keep an item from being queued for a second memory, and SQLite can't change
# a table's constraints, so it is rebuilt once with every row kept.
_MIGRATE_TARGET = """
ALTER TABLE outbox RENAME TO outbox_v1;
""" + _DDL.replace("CREATE INDEX IF NOT EXISTS idx_outbox_due", "DROP INDEX IF EXISTS idx_outbox_due;\nCREATE INDEX IF NOT EXISTS idx_outbox_due").replace(
    "CREATE INDEX IF NOT EXISTS idx_outbox_owner", "DROP INDEX IF EXISTS idx_outbox_owner;\nCREATE INDEX IF NOT EXISTS idx_outbox_owner") + """
INSERT INTO outbox (id, owner_id, provider, kind, source_id, agent, title, payload, state, attempts, next_at,
                    last_error, remote_id, created_at, sent_at, target)
    SELECT id, owner_id, provider, kind, source_id, agent, title, payload, state, attempts, next_at,
           last_error, remote_id, created_at, sent_at, '' FROM outbox_v1;
DROP TABLE outbox_v1;
"""


def _scope(owner_id: str, target: str | None) -> tuple[str, tuple[Any, ...]]:
    """The WHERE clause for one owner's rows, or one memory's of theirs."""
    if target is None:
        return "owner_id=?", (owner_id,)
    return "owner_id=? AND target=?", (owner_id, target)


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
            cur = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='outbox'")
            if await cur.fetchone():
                cols = {r[1] for r in await (await db.execute("PRAGMA table_info(outbox)")).fetchall()}
                if "target" not in cols:
                    await db.executescript("BEGIN;\n" + _MIGRATE_TARGET + "\nCOMMIT;")
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
        *, agent: str = "", title: str = "", target: str = "",
    ) -> bool:
        """Queue one item for the memory *target*; False when this (owner,
        target, kind, source_id) is queued or sent already."""
        async with self._lock:
            db = await self._conn()
            cur = await db.execute(
                """INSERT OR IGNORE INTO outbox
                   (owner_id, provider, kind, source_id, agent, title, payload, created_at, next_at, target)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (owner_id, provider, kind, source_id, agent[:200], title[:300],
                 json.dumps(payload, ensure_ascii=False), time.time(), target),
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

    async def defer_owner(self, owner_id: str, delay_s: float, error: str, *, target: str | None = None) -> int:
        """Hold every pending row of *owner_id* (key refused, quota full) —
        only those for *target* when given: no attempt is spent, they go once
        the cause is fixed."""
        async with self._lock:
            db = await self._conn()
            where, args = _scope(owner_id, target)
            cur = await db.execute(
                f"UPDATE outbox SET next_at=?, last_error=? WHERE {where} AND state='pending'",
                (time.time() + delay_s, error[:500], *args),
            )
            await db.commit()
            return cur.rowcount

    async def retry_owner(self, owner_id: str, *, target: str | None = None) -> int:
        """Failed and deferred rows of *owner_id* (for *target*, when given)
        go again now."""
        async with self._lock:
            db = await self._conn()
            where, args = _scope(owner_id, target)
            cur = await db.execute(
                "UPDATE outbox SET state='pending', next_at=0, attempts=CASE WHEN state='failed' THEN 0 "
                f"ELSE attempts END WHERE {where} AND state IN ('failed', 'pending')",
                args,
            )
            await db.commit()
            return cur.rowcount

    async def drop_owner(self, owner_id: str, *, pending_only: bool = True, target: str | None = None) -> int:
        """Forget *owner_id*'s queue — for *target* only, when given (that
        memory was removed): what hasn't gone never will — and all of it with
        ``pending_only=False``."""
        async with self._lock:
            db = await self._conn()
            where, args = _scope(owner_id, target)
            if pending_only:
                cur = await db.execute(
                    f"DELETE FROM outbox WHERE {where} AND state IN ('pending', 'failed')", args,
                )
            else:
                cur = await db.execute(f"DELETE FROM outbox WHERE {where}", args)
            await db.commit()
            return cur.rowcount

    async def counts(self, owner_id: str, *, target: str | None = None) -> dict[str, Any]:
        db = await self._conn()
        where, args = _scope(owner_id, target)
        cur = await db.execute(
            f"SELECT state, COUNT(*) AS n FROM outbox WHERE {where} GROUP BY state", args,
        )
        out: dict[str, Any] = {"pending": 0, "sent": 0, "failed": 0}
        for r in await cur.fetchall():
            out[str(r["state"])] = int(r["n"])
        cur = await db.execute(
            f"SELECT MAX(sent_at) AS t FROM outbox WHERE {where} AND state='sent'", args,
        )
        row = await cur.fetchone()
        out["last_sent_at"] = float(row["t"]) if row and row["t"] else None
        cur = await db.execute(
            f"SELECT last_error FROM outbox WHERE {where} AND state IN ('pending','failed') "
            "AND last_error != '' ORDER BY id DESC LIMIT 1",
            args,
        )
        row = await cur.fetchone()
        out["last_error"] = str(row["last_error"]) if row else ""
        return out

    async def recent(self, owner_id: str, limit: int = 30, state: str = "",
                     *, target: str | None = None) -> list[dict[str, Any]]:
        db = await self._conn()
        where, args = _scope(owner_id, target)
        if state:
            where += " AND state=?"
            args = (*args, state)
        cur = await db.execute(
            "SELECT id, kind, source_id, agent, title, state, attempts, last_error, remote_id, created_at, "
            f"sent_at, target FROM outbox WHERE {where} ORDER BY id DESC LIMIT ?",
            (*args, limit),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def prune(self, now: float | None = None) -> int:
        async with self._lock:
            db = await self._conn()
            cutoff = (time.time() if now is None else now) - SENT_RETENTION_S
            cur = await db.execute("DELETE FROM outbox WHERE state='sent' AND sent_at < ?", (cutoff,))
            await db.commit()
            return cur.rowcount
