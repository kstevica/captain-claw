"""Lupa's own product database — streams and their commission rounds.

Product-domain data only. Captain IDs (user_id, basna session ids, VFS project
names) are stored as opaque references; Captain knows nothing about this file.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS streams (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    pack TEXT NOT NULL DEFAULT '',
    vfs_project TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_streams_user ON streams(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS stream_sessions (
    stream_id TEXT NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    round_no INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'initial',
    created_at TEXT NOT NULL,
    PRIMARY KEY (stream_id, session_id)
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class LupaDB:
    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        # Additive migrations — each guarded so re-running is a no-op.
        try:
            await self._db.execute(
                "ALTER TABLE streams ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'")
        except aiosqlite.OperationalError:
            pass  # column exists
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ── streams ──────────────────────────────────────────────────────

    async def create_stream(self, user_id: str, title: str, pack: str) -> dict:
        assert self._db is not None
        now = _utcnow()
        sid = uuid.uuid4().hex
        await self._db.execute(
            "INSERT INTO streams (id, user_id, title, pack, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)", (sid, user_id, title, pack, now, now))
        await self._db.commit()
        return {"id": sid, "user_id": user_id, "title": title, "pack": pack,
                "vfs_project": "", "created_at": now, "updated_at": now}

    async def list_streams(self, user_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT s.*, COUNT(ss.session_id) AS rounds"
            " FROM streams s LEFT JOIN stream_sessions ss ON ss.stream_id = s.id"
            " WHERE s.user_id = ? GROUP BY s.id ORDER BY s.updated_at DESC",
            (user_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_stream(self, stream_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM streams WHERE id = ? AND user_id = ?",
            (stream_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def set_stream_vfs_project(self, stream_id: str, vfs_project: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE streams SET vfs_project = ?, updated_at = ? WHERE id = ?",
            (vfs_project, _utcnow(), stream_id))
        await self._db.commit()

    async def set_stream_settings(self, stream_id: str, settings: dict) -> None:
        import json
        assert self._db is not None
        await self._db.execute(
            "UPDATE streams SET settings = ?, updated_at = ? WHERE id = ?",
            (json.dumps(settings), _utcnow(), stream_id))
        await self._db.commit()

    # ── rounds ───────────────────────────────────────────────────────

    async def add_round(self, stream_id: str, session_id: str, kind: str) -> dict:
        assert self._db is not None
        async with self._db.execute(
            "SELECT COALESCE(MAX(round_no), 0) + 1 FROM stream_sessions WHERE stream_id = ?",
            (stream_id,),
        ) as cur:
            (round_no,) = await cur.fetchone()
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO stream_sessions (stream_id, session_id, round_no, kind, created_at)"
            " VALUES (?, ?, ?, ?, ?)", (stream_id, session_id, round_no, kind, now))
        await self._db.execute(
            "UPDATE streams SET updated_at = ? WHERE id = ?", (now, stream_id))
        await self._db.commit()
        return {"stream_id": stream_id, "session_id": session_id,
                "round_no": round_no, "kind": kind, "created_at": now}

    async def list_rounds(self, stream_id: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM stream_sessions WHERE stream_id = ? ORDER BY round_no",
            (stream_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def stream_for_session(self, session_id: str, user_id: str) -> dict | None:
        """The caller's stream containing this commission — the ownership check
        for every /api/commissions/* route."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT s.* FROM streams s JOIN stream_sessions ss ON ss.stream_id = s.id"
            " WHERE ss.session_id = ? AND s.user_id = ?", (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None
