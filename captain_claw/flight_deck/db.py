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

            -- ── Basna: router → selective spawn → weighted merge → learning ──
            CREATE TABLE IF NOT EXISTS basna_sessions (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title        TEXT NOT NULL DEFAULT '',
                intent       TEXT NOT NULL DEFAULT '',
                domain       TEXT NOT NULL DEFAULT '',
                difficulty   TEXT NOT NULL DEFAULT '',
                merge_kind   TEXT NOT NULL DEFAULT 'converge',
                status       TEXT NOT NULL DEFAULT 'routing',
                route        TEXT NOT NULL DEFAULT '{}',
                truth        TEXT NOT NULL DEFAULT '',
                confidence   REAL NOT NULL DEFAULT 0.0,
                config       TEXT NOT NULL DEFAULT '{}',
                progress     TEXT NOT NULL DEFAULT '[]',  -- JSON: execution progress log
                files        TEXT NOT NULL DEFAULT '[]',  -- JSON: attached files [{name,mime,size}]
                analysis     TEXT NOT NULL DEFAULT '{}',  -- JSON: cross-agent analysis (agreement/diffs/blind spots)
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_basna_sessions_user
                ON basna_sessions(user_id);

            CREATE TABLE IF NOT EXISTS basna_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT NOT NULL REFERENCES basna_sessions(id) ON DELETE CASCADE,
                archetype_id   TEXT NOT NULL DEFAULT '',
                role           TEXT NOT NULL DEFAULT '',
                provider       TEXT NOT NULL DEFAULT '',
                model          TEXT NOT NULL DEFAULT '',
                tier           TEXT NOT NULL DEFAULT '',
                weight_at_run  REAL NOT NULL DEFAULT 0.0,
                output         TEXT NOT NULL DEFAULT '',
                actions        TEXT NOT NULL DEFAULT '[]',  -- JSON: per-agent tool actions
                success        INTEGER,            -- NULL until scored; 1 = success, 0 = fail
                latency_ms     INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_basna_runs_session
                ON basna_runs(session_id);

            -- Learned, per-user reliability of each archetype within a domain.
            CREATE TABLE IF NOT EXISTS archetype_reliability (
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                archetype_id TEXT NOT NULL,
                domain       TEXT NOT NULL DEFAULT '',
                successes    INTEGER NOT NULL DEFAULT 0,
                fails        INTEGER NOT NULL DEFAULT 0,
                runs         INTEGER NOT NULL DEFAULT 0,
                weight       REAL NOT NULL DEFAULT 0.7,
                updated_at   TEXT NOT NULL,
                PRIMARY KEY (user_id, archetype_id, domain)
            );

            -- Vatra blackboard: cross-agent "asks" a specialist posts when it needs
            -- something outside its slice. The coordinator routes each to a helper
            -- and writes the answer back; the reporter folds answered asks in.
            CREATE TABLE IF NOT EXISTS basna_asks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL REFERENCES basna_sessions(id) ON DELETE CASCADE,
                from_owner   TEXT NOT NULL DEFAULT '',   -- asker archetype id
                from_subtask TEXT NOT NULL DEFAULT '',
                text         TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'open',  -- open|claimed|answered|dropped
                answer       TEXT NOT NULL DEFAULT '',
                answered_by  TEXT NOT NULL DEFAULT '',
                depth        INTEGER NOT NULL DEFAULT 0,
                note         TEXT NOT NULL DEFAULT '',      -- e.g. drop reason
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_basna_asks_session
                ON basna_asks(session_id);

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

            -- Per-user (per-tenant) agent archetypes. The base set lives in
            -- instructions/archetypes.json; rows here are added on top and, when
            -- archetype_id matches a base one, shadow it for that user. `data`
            -- holds the full archetype JSON (role, family, keywords,
            -- cognitive_mode, tier, tools, description, fleet_instructions,
            -- lead, reliability_seed).
            CREATE TABLE IF NOT EXISTS user_archetypes (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                archetype_id TEXT NOT NULL,
                data         TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                UNIQUE(user_id, archetype_id)
            );
            CREATE INDEX IF NOT EXISTS idx_user_archetypes_user
                ON user_archetypes(user_id);
        """)
        # Lightweight migrations: add columns introduced after a table first shipped.
        for table, col, ddl in [
            ("basna_runs", "actions", "TEXT NOT NULL DEFAULT '[]'"),
            ("basna_sessions", "progress", "TEXT NOT NULL DEFAULT '[]'"),
            ("basna_sessions", "files", "TEXT NOT NULL DEFAULT '[]'"),
            ("basna_sessions", "analysis", "TEXT NOT NULL DEFAULT '{}'"),
            ("basna_sessions", "title", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            except Exception:
                pass  # column already exists
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

    async def update_council_message(
        self, session_id: str, user_id: str, message_id: int, fields: dict,
    ) -> bool:
        """Patch a single message (used to checkpoint/finalize a streaming turn)."""
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return False
        allowed = ("content", "action", "suitability", "target_agent_id", "metadata")
        sets, params = [], []
        for k in allowed:
            if k in fields:
                sets.append(f"{k} = ?")
                params.append(fields[k])
        if not sets:
            return False
        params.extend([message_id, session_id])
        await self._db.execute(
            f"UPDATE council_messages SET {', '.join(sets)}"
            " WHERE id = ? AND session_id = ?",
            params,
        )
        await self._db.execute(
            "UPDATE council_sessions SET updated_at = ? WHERE id = ?",
            (_utcnow(), session_id),
        )
        await self._db.commit()
        return True

    async def delete_council_messages(
        self, session_id: str, user_id: str, round_num: int,
    ) -> int:
        """Delete all messages for a given round (used to restart a round).

        Returns the number of rows removed, or -1 if the session isn't owned by
        the user / doesn't exist.
        """
        assert self._db is not None
        sess = await self.get_council_session(session_id, user_id)
        if not sess:
            return -1
        cur = await self._db.execute(
            "DELETE FROM council_messages WHERE session_id = ? AND round = ?",
            (session_id, round_num),
        )
        await self._db.execute(
            "UPDATE council_sessions SET updated_at = ? WHERE id = ?",
            (_utcnow(), session_id),
        )
        await self._db.commit()
        return cur.rowcount

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

    # ── Basna sessions ───────────────────────────────────────────────

    async def create_basna_session(
        self, user_id: str, intent: str, config: str = "{}", title: str = "",
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        sid = _uuid()
        await self._db.execute(
            "INSERT INTO basna_sessions"
            " (id, user_id, title, intent, domain, difficulty, merge_kind, status,"
            "  route, truth, confidence, config, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, '', '', 'converge', 'routing', '{}', '', 0.0, ?, ?, ?)",
            (sid, user_id, title, intent, config, now, now),
        )
        await self._db.commit()
        return {"id": sid, "user_id": user_id, "title": title, "intent": intent, "domain": "",
                "difficulty": "", "merge_kind": "converge", "status": "routing",
                "route": "{}", "truth": "", "confidence": 0.0, "config": config,
                "created_at": now, "updated_at": now}

    async def list_basna_sessions(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM basna_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_basna_session(self, session_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM basna_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_basna_session(
        self, session_id: str, user_id: str, **fields,
    ) -> bool:
        assert self._db is not None
        allowed = {"title", "intent", "domain", "difficulty", "merge_kind", "status",
                   "route", "truth", "confidence", "config", "progress", "files", "analysis"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [session_id, user_id]
        cur = await self._db.execute(
            f"UPDATE basna_sessions SET {set_clause} WHERE id = ? AND user_id = ?", vals,
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def delete_basna_session(self, session_id: str, user_id: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "DELETE FROM basna_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Basna runs ───────────────────────────────────────────────────

    async def add_basna_runs(
        self, session_id: str, user_id: str, runs: list[dict],
    ) -> list[int]:
        assert self._db is not None
        sess = await self.get_basna_session(session_id, user_id)
        if not sess:
            return []
        now = _utcnow()
        ids: list[int] = []
        for r in runs:
            cur = await self._db.execute(
                "INSERT INTO basna_runs"
                " (session_id, archetype_id, role, provider, model, tier,"
                "  weight_at_run, output, actions, success, latency_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, r.get("archetype_id", ""), r.get("role", ""),
                 r.get("provider", ""), r.get("model", ""), r.get("tier", ""),
                 float(r.get("weight_at_run", 0.0)), r.get("output", ""),
                 r.get("actions", "[]"),
                 r.get("success"), int(r.get("latency_ms", 0)), now),
            )
            ids.append(cur.lastrowid or 0)
        await self._db.execute(
            "UPDATE basna_sessions SET updated_at = ? WHERE id = ?", (now, session_id),
        )
        await self._db.commit()
        return ids

    async def list_basna_runs(self, session_id: str, user_id: str) -> list[dict]:
        assert self._db is not None
        sess = await self.get_basna_session(session_id, user_id)
        if not sess:
            return []
        async with self._db.execute(
            "SELECT * FROM basna_runs WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def score_basna_run(
        self, run_id: int, user_id: str, success: bool,
    ) -> bool:
        """Mark a run success/fail, ownership-checked via the parent session."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_runs SET success = ?"
            " WHERE id = ? AND session_id IN"
            " (SELECT id FROM basna_sessions WHERE user_id = ?)",
            (1 if success else 0, run_id, user_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Vatra blackboard (cross-agent asks) ──────────────────────────

    async def create_vatra_ask(
        self, session_id: str, from_owner: str, from_subtask: str, text: str,
        depth: int = 0,
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        cur = await self._db.execute(
            "INSERT INTO basna_asks"
            " (session_id, from_owner, from_subtask, text, status, depth, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'open', ?, ?, ?)",
            (session_id, from_owner, from_subtask, text, depth, now, now),
        )
        await self._db.commit()
        return {"id": cur.lastrowid, "session_id": session_id, "from_owner": from_owner,
                "from_subtask": from_subtask, "text": text, "status": "open",
                "answer": "", "answered_by": "", "depth": depth, "created_at": now}

    async def list_vatra_asks(
        self, session_id: str, status: str | None = None, from_owner: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM basna_asks WHERE session_id = ?"
        params: list = [session_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        if from_owner:
            query += " AND from_owner = ?"
            params.append(from_owner)
        query += " ORDER BY id ASC"
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def count_vatra_asks(self, session_id: str) -> int:
        """Total asks ever created for this session — the budget counter."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT COUNT(*) AS n FROM basna_asks WHERE session_id = ?", (session_id,),
        ) as cur:
            row = await cur.fetchone()
            return int(row["n"]) if row else 0

    async def claim_vatra_ask(self, ask_id: int) -> bool:
        """Atomically move an ask open → claimed. Returns False if already taken."""
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_asks SET status = 'claimed', updated_at = ?"
            " WHERE id = ? AND status = 'open'",
            (_utcnow(), ask_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def answer_vatra_ask(self, ask_id: int, answer: str, answered_by: str) -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_asks SET status = 'answered', answer = ?, answered_by = ?,"
            " updated_at = ? WHERE id = ?",
            (answer, answered_by, _utcnow(), ask_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def drop_vatra_ask(self, ask_id: int, note: str = "") -> bool:
        assert self._db is not None
        cur = await self._db.execute(
            "UPDATE basna_asks SET status = 'dropped', note = ?, updated_at = ?"
            " WHERE id = ? AND status IN ('open', 'claimed')",
            (note, _utcnow(), ask_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Archetype reliability (learned routing weights) ──────────────

    @staticmethod
    def _reliability_weight(
        successes: int, fails: int, seed: float = 0.7, alpha: float = 4.0,
    ) -> float:
        """Bayesian-shrunk success rate toward `seed`, penalizing fails 2×.

        No data → returns `seed`; as runs accrue it approaches the empirical
        rate. Fails count double so an unreliable archetype decays quickly.
        """
        numerator = successes + alpha * seed
        denominator = successes + 2.0 * fails + alpha
        w = numerator / denominator if denominator > 0 else seed
        return max(0.05, min(0.99, w))

    async def get_archetype_reliability(
        self, user_id: str, domain: str | None = None,
    ) -> list[dict]:
        assert self._db is not None
        query = "SELECT * FROM archetype_reliability WHERE user_id = ?"
        params: list = [user_id]
        if domain is not None:
            query += " AND domain = ?"
            params.append(domain)
        async with self._db.execute(query, params) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_archetype_weight(
        self, user_id: str, archetype_id: str, domain: str, seed: float = 0.7,
    ) -> float:
        """Current learned weight for an archetype in a domain; `seed` if unseen."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT weight FROM archetype_reliability"
            " WHERE user_id = ? AND archetype_id = ? AND domain = ?",
            (user_id, archetype_id, domain),
        ) as cur:
            row = await cur.fetchone()
            return float(row["weight"]) if row else seed

    async def record_archetype_outcome(
        self, user_id: str, archetype_id: str, domain: str,
        success: bool, seed: float = 0.7,
    ) -> dict:
        """Upsert one outcome and recompute the learned weight."""
        assert self._db is not None
        now = _utcnow()
        async with self._db.execute(
            "SELECT successes, fails FROM archetype_reliability"
            " WHERE user_id = ? AND archetype_id = ? AND domain = ?",
            (user_id, archetype_id, domain),
        ) as cur:
            row = await cur.fetchone()
        successes = (row["successes"] if row else 0) + (1 if success else 0)
        fails = (row["fails"] if row else 0) + (0 if success else 1)
        runs = successes + fails
        weight = self._reliability_weight(successes, fails, seed)
        await self._db.execute(
            "INSERT INTO archetype_reliability"
            " (user_id, archetype_id, domain, successes, fails, runs, weight, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, archetype_id, domain) DO UPDATE SET"
            " successes = excluded.successes, fails = excluded.fails,"
            " runs = excluded.runs, weight = excluded.weight,"
            " updated_at = excluded.updated_at",
            (user_id, archetype_id, domain, successes, fails, runs, weight, now),
        )
        await self._db.commit()
        return {"user_id": user_id, "archetype_id": archetype_id, "domain": domain,
                "successes": successes, "fails": fails, "runs": runs,
                "weight": weight, "updated_at": now}

    async def get_basna_run(self, run_id: int, user_id: str) -> dict | None:
        """One run, ownership-checked via its parent session."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT r.* FROM basna_runs r JOIN basna_sessions s ON r.session_id = s.id"
            " WHERE r.id = ? AND s.user_id = ?",
            (run_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def adjust_archetype_reliability(
        self, user_id: str, archetype_id: str, domain: str,
        d_success: int, d_fail: int, seed: float = 0.7,
    ) -> dict:
        """Apply signed deltas to the success/fail counters and recompute weight.

        Used to *revise* a learned outcome (e.g. a human thumbs flipping an
        auto-scored run) without double-counting: move one from one bucket to the
        other rather than appending a fresh outcome. Counters never go negative.
        """
        assert self._db is not None
        now = _utcnow()
        async with self._db.execute(
            "SELECT successes, fails FROM archetype_reliability"
            " WHERE user_id = ? AND archetype_id = ? AND domain = ?",
            (user_id, archetype_id, domain),
        ) as cur:
            row = await cur.fetchone()
        successes = max(0, (row["successes"] if row else 0) + d_success)
        fails = max(0, (row["fails"] if row else 0) + d_fail)
        runs = successes + fails
        weight = self._reliability_weight(successes, fails, seed)
        await self._db.execute(
            "INSERT INTO archetype_reliability"
            " (user_id, archetype_id, domain, successes, fails, runs, weight, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, archetype_id, domain) DO UPDATE SET"
            " successes = excluded.successes, fails = excluded.fails,"
            " runs = excluded.runs, weight = excluded.weight,"
            " updated_at = excluded.updated_at",
            (user_id, archetype_id, domain, successes, fails, runs, weight, now),
        )
        await self._db.commit()
        return {"user_id": user_id, "archetype_id": archetype_id, "domain": domain,
                "successes": successes, "fails": fails, "runs": runs,
                "weight": weight, "updated_at": now}

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

    # ── User archetypes ──────────────────────────────────────────────

    async def list_user_archetypes(self, user_id: str) -> list[dict]:
        assert self._db is not None
        rows = await self._db.execute_fetchall(
            "SELECT * FROM user_archetypes WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    async def get_user_archetype(self, user_id: str, archetype_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM user_archetypes WHERE user_id = ? AND archetype_id = ?",
            (user_id, archetype_id),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def create_user_archetype(
        self, user_id: str, archetype_id: str, data: str,
    ) -> dict:
        assert self._db is not None
        now = _utcnow()
        aid = _uuid()
        await self._db.execute(
            "INSERT INTO user_archetypes (id, user_id, archetype_id, data, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (aid, user_id, archetype_id, data, now, now),
        )
        await self._db.commit()
        return {"id": aid, "user_id": user_id, "archetype_id": archetype_id,
                "data": data, "created_at": now, "updated_at": now}

    async def update_user_archetype(
        self, user_id: str, archetype_id: str, data: str,
    ) -> dict | None:
        assert self._db is not None
        existing = await self.get_user_archetype(user_id, archetype_id)
        if not existing:
            return None
        now = _utcnow()
        await self._db.execute(
            "UPDATE user_archetypes SET data = ?, updated_at = ?"
            " WHERE user_id = ? AND archetype_id = ?",
            (data, now, user_id, archetype_id),
        )
        await self._db.commit()
        return await self.get_user_archetype(user_id, archetype_id)

    async def delete_user_archetype(self, user_id: str, archetype_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "DELETE FROM user_archetypes WHERE user_id = ? AND archetype_id = ?",
            (user_id, archetype_id),
        ) as cur:
            await self._db.commit()
            return (cur.rowcount or 0) > 0
