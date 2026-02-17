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
        tool_arguments: dict[str, Any] | None = None,
        token_count: int | None = None,
    ) -> None:
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
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
            await self._db.commit()

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
