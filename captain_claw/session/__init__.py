"""Session management with SQLite storage."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from captain_claw.config import get_config, DEFAULT_DB_PATH
from captain_claw.logging import get_logger

log = get_logger(__name__)


@dataclass
class Message:
    """A message in the session."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Session:
    """A conversation session."""

    id: str
    name: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(
        self,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_arguments: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.updated_at = datetime.utcnow().isoformat()

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
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )


class SessionManager:
    """Manages conversation sessions with SQLite storage."""

    def __init__(self, db_path: Path | str | None = None):
        """Initialize session manager.
        
        Args:
            db_path: Optional database path override
        """
        config = get_config()
        self.db_path = Path(config.session.path).expanduser()
        
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
            await self._db.commit()

    async def get_or_create_session(self, name: str = "default") -> Session:
        """Get or create a session.
        
        Args:
            name: Session name
        
        Returns:
            Session instance
        """
        await self._ensure_db()
        
        # Try to get existing session
        async with self._db.execute(
            "SELECT id, name, messages, created_at, updated_at, metadata FROM sessions WHERE name = ? ORDER BY updated_at DESC LIMIT 1",
            (name,),
        ) as cursor:
            row = await cursor.fetchone()
        
        if row:
            return Session.from_dict({
                "id": row[0],
                "name": row[1],
                "messages": json.loads(row[2]),
                "created_at": row[3],
                "updated_at": row[4],
                "metadata": json.loads(row[5]),
            })
        
        # Create new session
        session = Session(
            id=str(uuid.uuid4()),
            name=name,
        )
        
        await self.save_session(session)
        log.info("Created new session", session_id=session.id, name=name)
        
        return session

    async def get_session(self, session_id: str) -> Session | None:
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

    async def save_session(self, session: Session) -> None:
        """Save a session.
        
        Args:
            session: Session to save
        """
        await self._ensure_db()
        
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
