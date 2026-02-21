"""Session management with SQLite storage."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)


def _utcnow_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


@dataclass
class Message:
    """A message in the session."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_arguments: dict[str, Any] | None = None
    token_count: int | None = None
    timestamp: str = field(default_factory=_utcnow_iso)


@dataclass
class Session:
    """A conversation session."""

    id: str
    name: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(
        self,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_arguments: dict[str, Any] | None = None,
        token_count: int | None = None,
    ) -> None:
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_calls": tool_calls,
            "tool_arguments": tool_arguments,
            "token_count": token_count,
            "timestamp": _utcnow_iso(),
        })
        self.updated_at = _utcnow_iso()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            messages=data.get("messages", []),
            created_at=data.get("created_at", _utcnow_iso()),
            updated_at=data.get("updated_at", _utcnow_iso()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CronJob:
    """A persisted Captain Claw pseudo-cron job."""

    id: str
    kind: str  # prompt | script | tool
    payload: dict[str, Any]
    schedule: dict[str, Any]
    session_id: str
    enabled: bool = True
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    last_run_at: str | None = None
    next_run_at: str = field(default_factory=_utcnow_iso)
    last_status: str = "pending"
    last_error: str | None = None
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    monitor_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "CronJob":
        """Create cron job from sqlite row."""
        chat_history_raw = "[]"
        monitor_history_raw = "[]"
        if len(row) > 12:
            chat_history_raw = row[12] or "[]"
        if len(row) > 13:
            monitor_history_raw = row[13] or "[]"
        return cls(
            id=str(row[0]),
            kind=str(row[1]),
            payload=json.loads(row[2]),
            schedule=json.loads(row[3]),
            session_id=str(row[4]),
            enabled=bool(int(row[5])),
            created_at=str(row[6]),
            updated_at=str(row[7]),
            last_run_at=str(row[8]) if row[8] else None,
            next_run_at=str(row[9]),
            last_status=str(row[10] or "pending"),
            last_error=str(row[11]) if row[11] else None,
            chat_history=json.loads(chat_history_raw),
            monitor_history=json.loads(monitor_history_raw),
        )


class SessionManager:
    """Manages conversation sessions with SQLite storage."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize session manager.
        
        Args:
            db_path: Optional database path override
        """
        if db_path is None:
            config = get_config()
            self.db_path = Path(config.session.path).expanduser()
        else:
            self.db_path = Path(db_path).expanduser()
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> None:
        """Ensure database is initialized."""
        if self._db is None:
            self._db = await aiosqlite.connect(str(self.db_path))
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    messages TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_name_updated_at ON sessions(name, updated_at DESC)"
            )
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS app_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS cron_jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_at TEXT,
                    next_run_at TEXT NOT NULL,
                    last_status TEXT NOT NULL DEFAULT 'pending',
                    last_error TEXT,
                    chat_history TEXT NOT NULL DEFAULT '[]',
                    monitor_history TEXT NOT NULL DEFAULT '[]'
                )
            """)
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_cron_jobs_enabled_next_run ON cron_jobs(enabled, next_run_at)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_cron_jobs_session_id ON cron_jobs(session_id)"
            )
            await self._ensure_cron_jobs_migrations()
            await self._db.commit()

    async def _ensure_cron_jobs_migrations(self) -> None:
        """Add missing cron_jobs columns for backward compatibility."""
        assert self._db is not None
        async with self._db.execute("PRAGMA table_info(cron_jobs)") as cursor:
            columns = await cursor.fetchall()
        existing = {str(row[1]) for row in columns}
        if "chat_history" not in existing:
            await self._db.execute(
                "ALTER TABLE cron_jobs ADD COLUMN chat_history TEXT NOT NULL DEFAULT '[]'"
            )
        if "monitor_history" not in existing:
            await self._db.execute(
                "ALTER TABLE cron_jobs ADD COLUMN monitor_history TEXT NOT NULL DEFAULT '[]'"
            )

    async def get_or_create_session(self, name: str = "default") -> Session:
        """Get or create a session.
        
        Args:
            name: Session name
        
        Returns:
            Session instance
        """
        await self._ensure_db()
        
        session = await self.load_session_by_name(name)
        if session:
            return session
        return await self.create_session(name=name)

    async def get_app_state(self, key: str) -> str | None:
        """Fetch one app state value by key."""
        await self._ensure_db()
        async with self._db.execute(
            "SELECT value FROM app_state WHERE key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return str(row[0])

    async def set_app_state(self, key: str, value: str) -> None:
        """Upsert one app state value."""
        await self._ensure_db()
        await self._db.execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, _utcnow_iso()),
        )
        await self._db.commit()

    async def delete_app_state(self, key: str) -> None:
        """Delete one app state key."""
        await self._ensure_db()
        await self._db.execute("DELETE FROM app_state WHERE key = ?", (key,))
        await self._db.commit()

    async def get_last_active_session_id(self) -> str | None:
        """Return tracked last active session id, if any."""
        return await self.get_app_state("last_active_session_id")

    async def set_last_active_session(self, session_id: str) -> bool:
        """Persist last active session id when session exists."""
        sid = (session_id or "").strip()
        if not sid:
            return False
        session = await self.load_session(sid)
        if not session:
            return False
        await self.set_app_state("last_active_session_id", sid)
        return True

    async def load_last_active_session(self) -> Session | None:
        """Load tracked last active session and clear stale pointer if missing."""
        sid = await self.get_last_active_session_id()
        if not sid:
            return None
        session = await self.load_session(sid)
        if session is None:
            await self.delete_app_state("last_active_session_id")
            return None
        return session

    async def create_session(self, name: str = "default", metadata: dict[str, Any] | None = None) -> Session:
        """Create and persist a new session."""
        await self._ensure_db()

        session = Session(
            id=str(uuid.uuid4()),
            name=name,
            metadata=metadata or {},
        )
        await self.save_session(session)
        log.info("Created new session", session_id=session.id, name=name)
        return session

    async def load_session(self, session_id: str) -> Session | None:
        """Get a session by ID.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session or None if not found
        """
        await self._ensure_db()
        
        async with self._db.execute(
            "SELECT id, name, messages, created_at, updated_at, metadata FROM sessions WHERE id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        
        if not row:
            return None
        
        return Session.from_dict({
            "id": row[0],
            "name": row[1],
            "messages": json.loads(row[2]),
            "created_at": row[3],
            "updated_at": row[4],
            "metadata": json.loads(row[5]),
        })

    async def get_session(self, session_id: str) -> Session | None:
        """Backward-compatible alias for load_session."""
        return await self.load_session(session_id)

    async def load_session_by_name(self, name: str) -> Session | None:
        """Load the most recently updated session by name."""
        await self._ensure_db()

        async with self._db.execute(
            """
            SELECT id, name, messages, created_at, updated_at, metadata
            FROM sessions
            WHERE name = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (name,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return Session.from_dict({
            "id": row[0],
            "name": row[1],
            "messages": json.loads(row[2]),
            "created_at": row[3],
            "updated_at": row[4],
            "metadata": json.loads(row[5]),
        })

    async def select_session(self, selector: str) -> Session | None:
        """Select a session by ID, name, or recent-list index.

        Resolution order:
        1. Exact ID
        2. Latest session with matching name
        3. Numeric index from recent session list (`#<n>` or `<n>`, 1-based)
        """
        key = selector.strip()
        if not key:
            return None

        by_id = await self.load_session(key)
        if by_id:
            return by_id

        by_name = await self.load_session_by_name(key)
        if by_name:
            return by_name

        index_text = key[1:] if key.startswith("#") else key
        if not index_text.isdigit():
            return None

        index = int(index_text)
        if index <= 0:
            return None

        sessions = await self.list_sessions(limit=max(20, index))
        if index > len(sessions):
            return None
        return sessions[index - 1]

    async def save_session(self, session: Session) -> None:
        """Save a session.
        
        Args:
            session: Session to save
        """
        await self._ensure_db()

        session.updated_at = _utcnow_iso()
        
        await self._db.execute("""
            INSERT OR REPLACE INTO sessions (id, name, messages, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.name,
            json.dumps(session.messages),
            session.created_at,
            session.updated_at,
            json.dumps(session.metadata),
        ))
        await self._db.commit()

    async def list_sessions(self, limit: int = 10) -> list[Session]:
        """List recent sessions.
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of sessions
        """
        await self._ensure_db()
        
        async with self._db.execute("""
            SELECT id, name, messages, created_at, updated_at, metadata
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
        
        return [
            Session.from_dict({
                "id": row[0],
                "name": row[1],
                "messages": json.loads(row[2]),
                "created_at": row[3],
                "updated_at": row[4],
                "metadata": json.loads(row[5]),
            })
            for row in rows
        ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            True if deleted, False if not found
        """
        await self._ensure_db()
        
        cursor = await self._db.execute(
            "DELETE FROM sessions WHERE id = ?",
            (session_id,),
        )
        await self._db.commit()
        if cursor.rowcount > 0:
            last_sid = await self.get_last_active_session_id()
            if last_sid == session_id:
                await self.delete_app_state("last_active_session_id")
        return cursor.rowcount > 0

    async def create_cron_job(
        self,
        kind: str,
        payload: dict[str, Any],
        schedule: dict[str, Any],
        session_id: str,
        next_run_at: str,
        enabled: bool = True,
    ) -> CronJob:
        """Create and persist a pseudo-cron job."""
        await self._ensure_db()

        now_iso = _utcnow_iso()
        job = CronJob(
            id=str(uuid.uuid4()),
            kind=kind,
            payload=payload,
            schedule=schedule,
            session_id=session_id,
            enabled=bool(enabled),
            created_at=now_iso,
            updated_at=now_iso,
            next_run_at=next_run_at,
            last_status="scheduled",
        )

        await self._db.execute(
            """
            INSERT INTO cron_jobs (
                id, kind, payload, schedule, session_id, enabled,
                created_at, updated_at, last_run_at, next_run_at, last_status, last_error,
                chat_history, monitor_history
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.kind,
                json.dumps(job.payload),
                json.dumps(job.schedule),
                job.session_id,
                1 if job.enabled else 0,
                job.created_at,
                job.updated_at,
                job.last_run_at,
                job.next_run_at,
                job.last_status,
                job.last_error,
                json.dumps(job.chat_history),
                json.dumps(job.monitor_history),
            ),
        )
        await self._db.commit()
        return job

    async def load_cron_job(self, job_id: str) -> CronJob | None:
        """Load one cron job by id."""
        await self._ensure_db()
        async with self._db.execute(
            """
            SELECT
                id, kind, payload, schedule, session_id, enabled,
                created_at, updated_at, last_run_at, next_run_at, last_status, last_error,
                chat_history, monitor_history
            FROM cron_jobs
            WHERE id = ?
            """,
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return CronJob.from_row(row)

    async def select_cron_job(self, selector: str, active_only: bool = False) -> CronJob | None:
        """Select cron job by id, #index, or numeric index."""
        key = selector.strip()
        if not key:
            return None

        by_id = await self.load_cron_job(key)
        if by_id:
            if active_only and not by_id.enabled:
                return None
            return by_id

        index_text = key[1:] if key.startswith("#") else key
        if not index_text.isdigit():
            return None

        index = int(index_text)
        if index <= 0:
            return None

        jobs = await self.list_cron_jobs(limit=max(200, index), active_only=active_only)
        if index > len(jobs):
            return None
        return jobs[index - 1]

    async def list_cron_jobs(self, limit: int = 200, active_only: bool = False) -> list[CronJob]:
        """List pseudo-cron jobs ordered by next run."""
        await self._ensure_db()
        if active_only:
            query = """
                SELECT
                    id, kind, payload, schedule, session_id, enabled,
                    created_at, updated_at, last_run_at, next_run_at, last_status, last_error,
                    chat_history, monitor_history
                FROM cron_jobs
                WHERE enabled = 1
                ORDER BY next_run_at ASC
                LIMIT ?
            """
            params = (limit,)
        else:
            query = """
                SELECT
                    id, kind, payload, schedule, session_id, enabled,
                    created_at, updated_at, last_run_at, next_run_at, last_status, last_error,
                    chat_history, monitor_history
                FROM cron_jobs
                ORDER BY enabled DESC, next_run_at ASC
                LIMIT ?
            """
            params = (limit,)

        async with self._db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [CronJob.from_row(row) for row in rows]

    async def get_due_cron_jobs(self, now_iso: str, limit: int = 10) -> list[CronJob]:
        """Return enabled jobs due to run at or before now."""
        await self._ensure_db()
        async with self._db.execute(
            """
            SELECT
                id, kind, payload, schedule, session_id, enabled,
                created_at, updated_at, last_run_at, next_run_at, last_status, last_error,
                chat_history, monitor_history
            FROM cron_jobs
            WHERE enabled = 1 AND next_run_at <= ?
            ORDER BY next_run_at ASC
            LIMIT ?
            """,
            (now_iso, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [CronJob.from_row(row) for row in rows]

    async def update_cron_job(
        self,
        job_id: str,
        *,
        enabled: bool | None = None,
        payload: dict[str, Any] | None = None,
        next_run_at: str | None = None,
        last_run_at: str | None = None,
        last_status: str | None = None,
        last_error: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        monitor_history: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Update mutable cron job fields."""
        await self._ensure_db()

        assignments: list[str] = ["updated_at = ?"]
        params: list[Any] = [_utcnow_iso()]

        if payload is not None:
            assignments.append("payload = ?")
            params.append(json.dumps(payload))
        if enabled is not None:
            assignments.append("enabled = ?")
            params.append(1 if enabled else 0)
        if next_run_at is not None:
            assignments.append("next_run_at = ?")
            params.append(next_run_at)
        if last_run_at is not None:
            assignments.append("last_run_at = ?")
            params.append(last_run_at)
        if last_status is not None:
            assignments.append("last_status = ?")
            params.append(last_status)
        if last_error is not None:
            assignments.append("last_error = ?")
            params.append(last_error)
        if chat_history is not None:
            assignments.append("chat_history = ?")
            params.append(json.dumps(chat_history))
        if monitor_history is not None:
            assignments.append("monitor_history = ?")
            params.append(json.dumps(monitor_history))

        params.append(job_id)
        cursor = await self._db.execute(
            f"UPDATE cron_jobs SET {', '.join(assignments)} WHERE id = ?",
            tuple(params),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def append_cron_job_history(
        self,
        job_id: str,
        *,
        chat_event: dict[str, Any] | None = None,
        monitor_event: dict[str, Any] | None = None,
        max_entries: int = 400,
    ) -> bool:
        """Append per-job chat/monitor events for easier inspection."""
        await self._ensure_db()
        job = await self.load_cron_job(job_id)
        if not job:
            return False

        chat_history = list(job.chat_history)
        monitor_history = list(job.monitor_history)
        if chat_event is not None:
            chat_history.append(chat_event)
            if len(chat_history) > max_entries:
                chat_history = chat_history[-max_entries:]
        if monitor_event is not None:
            monitor_history.append(monitor_event)
            if len(monitor_history) > max_entries:
                monitor_history = monitor_history[-max_entries:]

        return await self.update_cron_job(
            job_id,
            chat_history=chat_history,
            monitor_history=monitor_history,
        )

    async def delete_cron_job(self, job_id: str) -> bool:
        """Delete cron job by id."""
        await self._ensure_db()
        cursor = await self._db.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None


# Global session manager
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager."""
    global _manager
    _manager = manager
