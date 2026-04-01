"""Flight Deck SQLite database — users, settings, chat persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex


class FlightDeckDB:
    """Async SQLite store for Flight Deck multi-tenant data."""

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _create_tables(self) -> None:
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                email        TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                role         TEXT NOT NULL DEFAULT 'user',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                metadata     TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS user_sessions (
                id                 TEXT PRIMARY KEY,
                user_id            TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                refresh_token_hash TEXT NOT NULL,
                expires_at         TEXT NOT NULL,
                created_at         TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_sessions_user
                ON user_sessions(user_id);

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                agent_id   TEXT NOT NULL DEFAULT '',
                agent_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                ON chat_sessions(user_id);

            CREATE TABLE IF NOT EXISTS chat_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL DEFAULT '',
                metadata   TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id);

            CREATE TABLE IF NOT EXISTS system_settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                event_type TEXT NOT NULL,
                detail     TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_logs_user
                ON usage_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_usage_logs_type
                ON usage_logs(event_type);
            CREATE INDEX IF NOT EXISTS idx_usage_logs_created
                ON usage_logs(created_at);
        """)
        await self._db.commit()

    # ── Users ────────────────────────────────────────────────────────

    async def create_user(
        self, email: str, password_hash: str, display_name: str = "",
        role: str = "user",
    ) -> dict:
        now = _utcnow()
        uid = _uuid()
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email.lower().strip(), password_hash, display_name, role, now, now),
        )
        await self._db.commit()
        return {"id": uid, "email": email.lower().strip(), "display_name": display_name,
                "role": role, "created_at": now}

    async def get_user_by_email(self, email: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_user_by_id(self, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, email, display_name, role, created_at, updated_at, metadata"
            " FROM users WHERE id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_user(self, user_id: str, **fields) -> bool:
        assert self._db is not None
        allowed = {"email", "password_hash", "display_name", "role", "metadata"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [user_id]
        await self._db.execute(f"UPDATE users SET {set_clause} WHERE id = ?", vals)
        await self._db.commit()
        return True

    async def count_users(self) -> int:
        assert self._db is not None
        async with self._db.execute("SELECT COUNT(*) FROM users") as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    # ── Refresh sessions ─────────────────────────────────────────────

    async def create_refresh_session(
        self, user_id: str, refresh_token_hash: str, expires_at: str,
    ) -> str:
        sid = _uuid()
        now = _utcnow()
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO user_sessions (id, user_id, refresh_token_hash, expires_at, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (sid, user_id, refresh_token_hash, expires_at, now),
        )
        await self._db.commit()
        return sid

    async def get_refresh_session(self, session_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM user_sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def delete_refresh_session(self, session_id: str) -> None:
        assert self._db is not None
        await self._db.execute("DELETE FROM user_sessions WHERE id = ?", (session_id,))
        await self._db.commit()

    async def delete_user_refresh_sessions(self, user_id: str) -> None:
        assert self._db is not None
        await self._db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        await self._db.commit()

    async def cleanup_expired_sessions(self) -> None:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute("DELETE FROM user_sessions WHERE expires_at < ?", (now,))
        await self._db.commit()

    # ── User settings ────────────────────────────────────────────────

    async def get_all_settings(self, user_id: str) -> dict[str, str]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT key, value FROM user_settings WHERE user_id = ?", (user_id,)
        ) as cur:
            rows = await cur.fetchall()
            return {r["key"]: r["value"] for r in rows}

    async def set_settings(self, user_id: str, settings: dict[str, str]) -> None:
        assert self._db is not None
        now = _utcnow()
        for key, value in settings.items():
            await self._db.execute(
                "INSERT INTO user_settings (user_id, key, value, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value,"
                " updated_at = excluded.updated_at",
                (user_id, key, value, now),
            )
        await self._db.commit()

    # ── System settings (no FK, for global config) ────────────────

    async def get_system_setting(self, key: str) -> str | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT value FROM system_settings WHERE key = ?", (key,)
        ) as cur:
            row = await cur.fetchone()
            return row["value"] if row else None

    async def set_system_setting(self, key: str, value: str) -> None:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO system_settings (key, value, updated_at)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, now),
        )
        await self._db.commit()

    async def get_all_system_settings(self) -> dict[str, str]:
        assert self._db is not None
        async with self._db.execute("SELECT key, value FROM system_settings") as cur:
            rows = await cur.fetchall()
            return {r["key"]: r["value"] for r in rows}

    async def delete_setting(self, user_id: str, key: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM user_settings WHERE user_id = ? AND key = ?", (user_id, key)
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Chat sessions ────────────────────────────────────────────────

    async def list_chat_sessions(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_chat_session(self, session_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def upsert_chat_session(
        self, session_id: str, user_id: str, agent_id: str = "",
        agent_name: str = "",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO chat_sessions (id, user_id, agent_id, agent_name, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET agent_name = excluded.agent_name,"
            " updated_at = excluded.updated_at",
            (session_id, user_id, agent_id, agent_name, now, now),
        )
        await self._db.commit()
        return {"id": session_id, "user_id": user_id, "agent_id": agent_id,
                "agent_name": agent_name, "updated_at": now}

    async def delete_chat_session(self, session_id: str, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def get_chat_messages(
        self, session_id: str, user_id: str,
        limit: int = 100, before_id: int | None = None,
    ) -> list[dict]:
        assert self._db is not None
        # Verify ownership
        sess = await self.get_chat_session(session_id, user_id)
        if not sess:
            return []
        query = "SELECT * FROM chat_messages WHERE session_id = ?"
        params: list = [session_id]
        if before_id is not None:
            query += " AND id < ?"
            params.append(before_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        rows.reverse()  # Return chronological order
        return rows

    async def add_chat_messages(
        self, session_id: str, user_id: str, messages: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_chat_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids = []
        for msg in messages:
            cur = await self._db.execute(
                "INSERT INTO chat_messages (session_id, role, content, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, msg.get("role", ""), msg.get("content", ""),
                 msg.get("metadata", "{}"), now),
            )
            ids.append(cur.lastrowid)
        # Touch session
        await self._db.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id)
        )
        await self._db.commit()
        return ids

    # ── Usage logs ───────────────────────────────────────────────────

    async def log_usage(
        self, user_id: str, event_type: str, detail: str = "{}",
    ) -> int:
        assert self._db is not None
        now = _utcnow()
        cur = await self._db.execute(
            "INSERT INTO usage_logs (user_id, event_type, detail, created_at)"
            " VALUES (?, ?, ?, ?)",
            (user_id, event_type, detail, now),
        )
        await self._db.commit()
        return cur.lastrowid or 0

    async def get_usage_logs(
        self, user_id: str | None = None, event_type: str | None = None,
        since: str | None = None, limit: int = 200,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM usage_logs WHERE 1=1"
        params: list = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_usage_summary(
        self, user_id: str | None = None, since: str | None = None,
    ) -> dict[str, int]:
        """Return event counts grouped by event_type."""
        assert self._db is not None
        query = "SELECT event_type, COUNT(*) as cnt FROM usage_logs WHERE 1=1"
        params: list = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        query += " GROUP BY event_type"
        async with self._db.execute(query, params) as cur:
            rows = await cur.fetchall()
            return {r["event_type"]: r["cnt"] for r in rows}

    # ── Admin helpers ────────────────────────────────────────────────

    async def list_users(self, limit: int = 100, offset: int = 0) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT id, email, display_name, role, created_at, updated_at, metadata"
            " FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def delete_user(self, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await self._db.commit()
        return cur.rowcount > 0
