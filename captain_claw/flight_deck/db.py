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

            CREATE TABLE IF NOT EXISTS council_sessions (
                id              TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title           TEXT NOT NULL DEFAULT '',
                topic           TEXT NOT NULL DEFAULT '',
                session_type    TEXT NOT NULL DEFAULT 'brainstorm',
                verbosity       TEXT NOT NULL DEFAULT 'message',
                max_rounds      INTEGER NOT NULL DEFAULT 5,
                current_round   INTEGER NOT NULL DEFAULT 0,
                status          TEXT NOT NULL DEFAULT 'setup',
                moderator_mode  TEXT NOT NULL DEFAULT 'round-robin',
                moderator_agent TEXT NOT NULL DEFAULT '',
                agents          TEXT NOT NULL DEFAULT '[]',
                pinned_ids      TEXT NOT NULL DEFAULT '[]',
                config          TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_sessions_user
                ON council_sessions(user_id);

            CREATE TABLE IF NOT EXISTS council_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES council_sessions(id) ON DELETE CASCADE,
                round           INTEGER NOT NULL DEFAULT 1,
                agent_id        TEXT NOT NULL DEFAULT '',
                agent_name      TEXT NOT NULL DEFAULT '',
                role            TEXT NOT NULL,
                action          TEXT NOT NULL DEFAULT '',
                suitability     REAL NOT NULL DEFAULT 0.0,
                target_agent_id TEXT NOT NULL DEFAULT '',
                content         TEXT NOT NULL DEFAULT '',
                pinned          INTEGER NOT NULL DEFAULT 0,
                metadata        TEXT NOT NULL DEFAULT '{}',
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_messages_session
                ON council_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_council_messages_round
                ON council_messages(session_id, round);

            CREATE TABLE IF NOT EXISTS council_votes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES council_sessions(id) ON DELETE CASCADE,
                round           INTEGER NOT NULL,
                agent_id        TEXT NOT NULL,
                agent_name      TEXT NOT NULL DEFAULT '',
                vote            TEXT NOT NULL,
                reason          TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_votes_session
                ON council_votes(session_id);

            CREATE TABLE IF NOT EXISTS council_artifacts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES council_sessions(id) ON DELETE CASCADE,
                kind            TEXT NOT NULL,
                agent_id        TEXT NOT NULL DEFAULT '',
                agent_name      TEXT NOT NULL DEFAULT '',
                content         TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_council_artifacts_session
                ON council_artifacts(session_id);

            CREATE TABLE IF NOT EXISTS prompts (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title      TEXT NOT NULL DEFAULT '',
                content    TEXT NOT NULL DEFAULT '',
                files      TEXT NOT NULL DEFAULT '[]',
                tags       TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prompts_user
                ON prompts(user_id);
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

    # ── Council sessions ─────────────────────────────────────────────

    async def create_council_session(
        self, user_id: str, title: str, topic: str,
        session_type: str = "brainstorm", verbosity: str = "message",
        max_rounds: int = 5, moderator_mode: str = "round-robin",
        moderator_agent: str = "", agents: str = "[]", config: str = "{}",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        sid = _uuid()
        await self._db.execute(
            "INSERT INTO council_sessions"
            " (id, user_id, title, topic, session_type, verbosity, max_rounds,"
            "  current_round, status, moderator_mode, moderator_agent, agents,"
            "  pinned_ids, config, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'setup', ?, ?, ?, '[]', ?, ?, ?)",
            (sid, user_id, title, topic, session_type, verbosity, max_rounds,
             moderator_mode, moderator_agent, agents, config, now, now),
        )
        await self._db.commit()
        return {"id": sid, "user_id": user_id, "title": title, "topic": topic,
                "session_type": session_type, "verbosity": verbosity,
                "max_rounds": max_rounds, "current_round": 0, "status": "setup",
                "moderator_mode": moderator_mode, "moderator_agent": moderator_agent,
                "agents": agents, "pinned_ids": "[]", "config": config,
                "created_at": now, "updated_at": now}

    async def list_council_sessions(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM council_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_council_session(self, session_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM council_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_council_session(
        self, session_id: str, user_id: str, **fields,
    ) -> bool:
        assert self._db is not None
        allowed = {"title", "topic", "status", "current_round", "moderator_mode",
                   "moderator_agent", "agents", "pinned_ids", "config"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [session_id, user_id]
        cur = await self._db.execute(
            f"UPDATE council_sessions SET {set_clause} WHERE id = ? AND user_id = ?", vals,
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_council_session(self, session_id: str, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM council_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Council messages ─────────────────────────────────────────────

    async def get_council_messages(
        self, session_id: str, user_id: str,
        round_num: int | None = None, limit: int = 500,
    ) -> list[dict]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        query = "SELECT * FROM council_messages WHERE session_id = ?"
        params: list = [session_id]
        if round_num is not None:
            query += " AND round = ?"
            params.append(round_num)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def add_council_messages(
        self, session_id: str, user_id: str, messages: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids = []
        for msg in messages:
            cur = await self._db.execute(
                "INSERT INTO council_messages"
                " (session_id, round, agent_id, agent_name, role, action,"
                "  suitability, target_agent_id, content, pinned, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, msg.get("round", 1), msg.get("agent_id", ""),
                 msg.get("agent_name", ""), msg.get("role", "agent"),
                 msg.get("action", ""), msg.get("suitability", 0.0),
                 msg.get("target_agent_id", ""), msg.get("content", ""),
                 msg.get("pinned", 0), msg.get("metadata", "{}"), now),
            )
            ids.append(cur.lastrowid)
        await self._db.execute(
            "UPDATE council_sessions SET updated_at = ? WHERE id = ?", (now, session_id),
        )
        await self._db.commit()
        return ids

    async def toggle_council_pin(
        self, session_id: str, user_id: str, message_id: int,
    ) -> bool:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return False
        await self._db.execute(
            "UPDATE council_messages SET pinned = CASE WHEN pinned = 0 THEN 1 ELSE 0 END"
            " WHERE id = ? AND session_id = ?",
            (message_id, session_id),
        )
        await self._db.commit()
        return True

    # ── Council votes ────────────────────────────────────────────────

    async def add_council_votes(
        self, session_id: str, user_id: str, votes: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids = []
        for v in votes:
            cur = await self._db.execute(
                "INSERT INTO council_votes"
                " (session_id, round, agent_id, agent_name, vote, reason, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, v.get("round", 1), v.get("agent_id", ""),
                 v.get("agent_name", ""), v.get("vote", "abstain"),
                 v.get("reason", ""), now),
            )
            ids.append(cur.lastrowid)
        await self._db.commit()
        return ids

    async def get_council_votes(
        self, session_id: str, user_id: str, round_num: int | None = None,
    ) -> list[dict]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        query = "SELECT * FROM council_votes WHERE session_id = ?"
        params: list = [session_id]
        if round_num is not None:
            query += " AND round = ?"
            params.append(round_num)
        query += " ORDER BY id ASC"
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── Council artifacts ───────────────────────────────────────────

    async def get_council_artifacts(
        self, session_id: str, user_id: str, kind: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return []
        query = "SELECT * FROM council_artifacts WHERE session_id = ?"
        params: list = [session_id]
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY id ASC"
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def upsert_council_artifact(
        self, session_id: str, user_id: str,
        kind: str, agent_id: str, agent_name: str, content: str,
    ) -> int:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return 0
        now = _utcnow()
        # Delete existing artifact with same key, then insert
        await self._db.execute(
            "DELETE FROM council_artifacts WHERE session_id = ? AND kind = ? AND agent_id = ?",
            (session_id, kind, agent_id),
        )
        async with self._db.execute(
            "INSERT INTO council_artifacts"
            " (session_id, kind, agent_id, agent_name, content, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, kind, agent_id, agent_name, content, now),
        ) as cur:
            art_id = cur.lastrowid or 0
        await self._db.commit()
        return art_id

    async def delete_council_artifacts(
        self, session_id: str, user_id: str, kind: str | None = None,
    ) -> bool:
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return False
        if kind:
            await self._db.execute(
                "DELETE FROM council_artifacts WHERE session_id = ? AND kind = ?",
                (session_id, kind),
            )
        else:
            await self._db.execute(
                "DELETE FROM council_artifacts WHERE session_id = ?", (session_id,),
            )
        await self._db.commit()
        return True

    # ── Prompts ─────────────────────────────────────────────────────

    async def list_prompts(self, user_id: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM prompts WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def get_prompt(self, prompt_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM prompts WHERE id = ? AND user_id = ?",
            (prompt_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def create_prompt(
        self, user_id: str, title: str, content: str,
        files: str = "[]", tags: str = "[]",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        pid = _uuid()
        await self._db.execute(
            "INSERT INTO prompts (id, user_id, title, content, files, tags, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, user_id, title, content, files, tags, now, now),
        )
        await self._db.commit()
        return {"id": pid, "user_id": user_id, "title": title, "content": content,
                "files": files, "tags": tags, "created_at": now, "updated_at": now}

    async def update_prompt(
        self, prompt_id: str, user_id: str, **fields: str,
    ) -> dict | None:
        assert self._db is not None
        existing = await self.get_prompt(prompt_id, user_id)
        if not existing:
            return None
        allowed = {"title", "content", "files", "tags"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return existing
        updates["updated_at"] = _utcnow()
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [prompt_id, user_id]
        await self._db.execute(
            f"UPDATE prompts SET {sets} WHERE id = ? AND user_id = ?", vals,
        )
        await self._db.commit()
        return await self.get_prompt(prompt_id, user_id)

    async def delete_prompt(self, prompt_id: str, user_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM prompts WHERE id = ? AND user_id = ?",
            (prompt_id, user_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0
