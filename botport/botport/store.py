"""SQLite persistence for BotPort concerns and stats."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from botport.models import Concern, ConcernExchange

DEFAULT_DB_PATH = Path("~/.botport/botport.db").expanduser()


class BotPortStore:
    """Async SQLite store for concerns history and stats."""

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path or DEFAULT_DB_PATH).expanduser()
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is not None:
            return self._db
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._create_tables()
        return self._db

    async def _create_tables(self) -> None:
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS concerns (

                id           TEXT PRIMARY KEY,
                from_instance TEXT NOT NULL,
                from_instance_name TEXT DEFAULT '',
                from_session TEXT DEFAULT '',
                assigned_instance TEXT,
                assigned_instance_name TEXT DEFAULT '',
                assigned_session TEXT,
                task         TEXT DEFAULT '',
                context      TEXT DEFAULT '{}',
                expertise_tags TEXT DEFAULT '[]',
                status       TEXT DEFAULT 'pending',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                timeout_at   TEXT DEFAULT '',
                metadata     TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS concern_messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                concern_id   TEXT NOT NULL,
                direction    TEXT NOT NULL,
                content      TEXT DEFAULT '',
                timestamp    TEXT NOT NULL,
                from_instance TEXT DEFAULT '',
                metadata     TEXT DEFAULT '{}',
                FOREIGN KEY (concern_id) REFERENCES concerns(id)
            );

            CREATE INDEX IF NOT EXISTS idx_concerns_status ON concerns(status);
            CREATE INDEX IF NOT EXISTS idx_concerns_from ON concerns(from_instance);
            CREATE INDEX IF NOT EXISTS idx_concerns_assigned ON concerns(assigned_instance);
            CREATE INDEX IF NOT EXISTS idx_concern_messages_cid ON concern_messages(concern_id);
        """)
        # Migrate: add name columns if upgrading from older schema.
        for col in ("from_instance_name", "assigned_instance_name"):
            try:
                await self._db.execute(
                    f"ALTER TABLE concerns ADD COLUMN {col} TEXT DEFAULT ''"
                )
            except Exception:
                pass  # Column already exists.
        await self._db.commit()

        # Create swarm tables.
        from botport.swarm.store import SwarmStore
        self._swarm_store = SwarmStore(self._db)
        await self._swarm_store.create_tables()

    @property
    def swarm(self) -> "SwarmStore":
        """Access the swarm store (available after DB init)."""
        return self._swarm_store

    async def save_concern(self, concern: Concern) -> None:
        """Upsert a concern and its messages."""
        db = await self._ensure_db()
        await db.execute(
            """INSERT INTO concerns
               (id, from_instance, from_instance_name, from_session,
                assigned_instance, assigned_instance_name, assigned_session,
                task, context, expertise_tags, status, created_at, updated_at,
                timeout_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 from_instance_name = excluded.from_instance_name,
                 assigned_instance = excluded.assigned_instance,
                 assigned_instance_name = excluded.assigned_instance_name,
                 assigned_session = excluded.assigned_session,
                 status = excluded.status,
                 updated_at = excluded.updated_at,
                 timeout_at = excluded.timeout_at,
                 metadata = excluded.metadata
            """,
            (
                concern.id,
                concern.from_instance,
                concern.from_instance_name,
                concern.from_session,
                concern.assigned_instance,
                concern.assigned_instance_name,
                concern.assigned_session,
                concern.task,
                json.dumps(concern.context),
                json.dumps(concern.expertise_tags),
                concern.status,
                concern.created_at,
                concern.updated_at,
                concern.timeout_at,
                json.dumps(concern.metadata),
            ),
        )
        await db.commit()

    async def save_exchange(self, concern_id: str, exchange: ConcernExchange) -> None:
        """Append a single exchange message to a concern."""
        db = await self._ensure_db()
        await db.execute(
            """INSERT INTO concern_messages
               (concern_id, direction, content, timestamp, from_instance, metadata)
               VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                concern_id,
                exchange.direction,
                exchange.content,
                exchange.timestamp,
                exchange.from_instance,
                json.dumps(exchange.metadata),
            ),
        )
        await db.commit()

    async def load_concern(self, concern_id: str) -> Concern | None:
        """Load a concern with its exchange history."""
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM concerns WHERE id = ?", (concern_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None

        messages: list[ConcernExchange] = []
        async with db.execute(
            "SELECT * FROM concern_messages WHERE concern_id = ? ORDER BY id",
            (concern_id,),
        ) as cursor:
            async for msg_row in cursor:
                messages.append(ConcernExchange(
                    direction=msg_row["direction"],
                    content=msg_row["content"],
                    timestamp=msg_row["timestamp"],
                    from_instance=msg_row["from_instance"],
                    metadata=json.loads(msg_row["metadata"] or "{}"),
                ))

        return Concern(
            id=row["id"],
            from_instance=row["from_instance"],
            from_instance_name=row["from_instance_name"] or "",
            from_session=row["from_session"] or "",
            assigned_instance=row["assigned_instance"],
            assigned_instance_name=row["assigned_instance_name"] or "",
            assigned_session=row["assigned_session"],
            task=row["task"] or "",
            context=json.loads(row["context"] or "{}"),
            expertise_tags=json.loads(row["expertise_tags"] or "[]"),
            status=row["status"] or "pending",
            messages=messages,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            timeout_at=row["timeout_at"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    async def list_concerns(
        self,
        status: str | None = None,
        limit: int = 50,
        include_terminal: bool = True,
    ) -> list[Concern]:
        """List concerns, optionally filtered by status."""
        db = await self._ensure_db()
        if status:
            query = "SELECT * FROM concerns WHERE status = ? ORDER BY updated_at DESC LIMIT ?"
            params: tuple[Any, ...] = (status, limit)
        elif not include_terminal:
            query = (
                "SELECT * FROM concerns "
                "WHERE status NOT IN ('closed', 'failed', 'timeout') "
                "ORDER BY updated_at DESC LIMIT ?"
            )
            params = (limit,)
        else:
            query = "SELECT * FROM concerns ORDER BY updated_at DESC LIMIT ?"
            params = (limit,)

        concerns: list[Concern] = []
        async with db.execute(query, params) as cursor:
            async for row in cursor:
                concerns.append(Concern(
                    id=row["id"],
                    from_instance=row["from_instance"],
                    from_instance_name=row["from_instance_name"] or "",
                    from_session=row["from_session"] or "",
                    assigned_instance=row["assigned_instance"],
                    assigned_instance_name=row["assigned_instance_name"] or "",
                    assigned_session=row["assigned_session"],
                    task=row["task"] or "",
                    context=json.loads(row["context"] or "{}"),
                    expertise_tags=json.loads(row["expertise_tags"] or "[]"),
                    status=row["status"] or "pending",
                    messages=[],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    timeout_at=row["timeout_at"] or "",
                    metadata=json.loads(row["metadata"] or "{}"),
                ))
        return concerns

    async def get_stats(self) -> dict[str, Any]:
        """Aggregate stats for the dashboard."""
        db = await self._ensure_db()

        stats: dict[str, Any] = {}

        # Total counts by status.
        async with db.execute(
            "SELECT status, COUNT(*) as cnt FROM concerns GROUP BY status"
        ) as cursor:
            by_status = {}
            async for row in cursor:
                by_status[row["status"]] = row["cnt"]
            stats["by_status"] = by_status
            stats["total"] = sum(by_status.values())

        # Last 24 hours.
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM concerns WHERE created_at >= ?", (cutoff,)
        ) as cursor:
            row = await cursor.fetchone()
            stats["last_24h"] = row["cnt"] if row else 0

        # Success rate (closed / (closed + failed + timeout)).
        closed = by_status.get("closed", 0)
        failed = by_status.get("failed", 0) + by_status.get("timeout", 0)
        total_resolved = closed + failed
        stats["success_rate"] = (closed / total_resolved * 100) if total_resolved > 0 else 100.0

        return stats

    async def cleanup_old(self, retention_days: int = 30) -> int:
        """Delete concerns older than retention_days. Returns count deleted."""
        db = await self._ensure_db()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

        async with db.execute(
            "SELECT id FROM concerns WHERE updated_at < ? AND status IN ('closed', 'failed', 'timeout')",
            (cutoff,),
        ) as cursor:
            ids = [row["id"] async for row in cursor]

        if not ids:
            return 0

        placeholders = ",".join("?" for _ in ids)
        await db.execute(
            f"DELETE FROM concern_messages WHERE concern_id IN ({placeholders})", ids
        )
        await db.execute(
            f"DELETE FROM concerns WHERE id IN ({placeholders})", ids
        )
        await db.commit()
        return len(ids)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
