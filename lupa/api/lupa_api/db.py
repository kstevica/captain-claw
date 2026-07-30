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

CREATE TABLE IF NOT EXISTS briefs (
    stream_id TEXT PRIMARY KEY REFERENCES streams(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    instruction TEXT NOT NULL,
    cadence_hours REAL NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run_at TEXT,
    last_session_id TEXT,
    next_run_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_briefs_due ON briefs(enabled, next_run_at);

CREATE TABLE IF NOT EXISTS second_opinions (
    session_id TEXT PRIMARY KEY,
    basna_session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS packs (
    slug TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    version INTEGER NOT NULL DEFAULT 0,
    manifest TEXT NOT NULL DEFAULT '{}',
    generation TEXT NOT NULL DEFAULT '{}',
    eval_state TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS creators (
    user_id TEXT PRIMARY KEY
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

    async def list_streams(self, user_id: str, pack: str | None = None) -> list[dict]:
        assert self._db is not None
        q = ("SELECT s.*, COUNT(ss.session_id) AS rounds"
             " FROM streams s LEFT JOIN stream_sessions ss ON ss.stream_id = s.id"
             " WHERE s.user_id = ?")
        args: list = [user_id]
        if pack is not None:
            q += " AND s.pack = ?"
            args.append(pack)
        q += " GROUP BY s.id ORDER BY s.updated_at DESC"
        async with self._db.execute(q, args) as cur:
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

    # ── standing briefs ──────────────────────────────────────────────

    async def upsert_brief(self, stream_id: str, user_id: str, instruction: str,
                           cadence_hours: float, enabled: bool) -> dict:
        """One standing brief per stream. (Re)creating one schedules the first
        run for NOW — the brief reports immediately, then every cadence."""
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO briefs (stream_id, user_id, instruction, cadence_hours,"
            " enabled, next_run_at, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(stream_id) DO UPDATE SET instruction = excluded.instruction,"
            " cadence_hours = excluded.cadence_hours, enabled = excluded.enabled,"
            " updated_at = excluded.updated_at",
            (stream_id, user_id, instruction, cadence_hours, int(enabled), now, now, now))
        await self._db.commit()
        return await self.get_brief(stream_id, user_id)  # type: ignore[return-value]

    async def get_brief(self, stream_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM briefs WHERE stream_id = ? AND user_id = ?",
            (stream_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def delete_brief(self, stream_id: str, user_id: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM briefs WHERE stream_id = ? AND user_id = ?",
            (stream_id, user_id))
        await self._db.commit()

    async def list_due_briefs(self, now: str) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM briefs WHERE enabled = 1 AND next_run_at <= ?", (now,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def mark_brief_ran(self, stream_id: str, session_id: str,
                             next_run_at: str) -> None:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "UPDATE briefs SET last_run_at = ?, last_session_id = ?,"
            " next_run_at = ?, updated_at = ? WHERE stream_id = ?",
            (now, session_id, next_run_at, now, stream_id))
        await self._db.commit()

    async def list_brief_rounds(self, user_id: str, limit: int = 20) -> list[dict]:
        """The inbox: recent scheduler-produced rounds across the user's streams."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT ss.*, s.title AS stream_title FROM stream_sessions ss"
            " JOIN streams s ON s.id = ss.stream_id"
            " WHERE s.user_id = ? AND ss.kind = 'brief'"
            " ORDER BY ss.created_at DESC LIMIT ?", (user_id, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── pack registry (Kalup: verticals are data, runtime-managed) ───

    async def upsert_seed_pack(self, slug: str, manifest: dict) -> None:
        """Import a repo pack as a published SYSTEM pack (owner ''). Only
        inserts — a runtime-edited row is never clobbered by a seed."""
        import json
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT OR IGNORE INTO packs (slug, owner_id, status, version,"
            " manifest, created_at, updated_at)"
            " VALUES (?, '', 'published', 1, ?, ?, ?)",
            (slug, json.dumps(manifest), now, now))
        await self._db.commit()

    async def create_pack(self, slug: str, owner_id: str, manifest: dict) -> dict:
        import json
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO packs (slug, owner_id, status, version, manifest,"
            " created_at, updated_at) VALUES (?, ?, 'draft', 0, ?, ?, ?)",
            (slug, owner_id, json.dumps(manifest), now, now))
        await self._db.commit()
        return (await self.get_pack(slug))  # type: ignore[return-value]

    async def get_pack(self, slug: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM packs WHERE slug = ?", (slug,),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def list_packs(self) -> list[dict]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM packs ORDER BY status DESC, updated_at DESC",
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def update_pack(self, slug: str, *, manifest: dict | None = None,
                          status: str | None = None, bump_version: bool = False,
                          generation: dict | None = None,
                          eval_state: dict | None = None) -> None:
        import json
        assert self._db is not None
        sets, args = ["updated_at = ?"], [_utcnow()]
        if manifest is not None:
            sets.append("manifest = ?"); args.append(json.dumps(manifest))
        if status is not None:
            sets.append("status = ?"); args.append(status)
        if bump_version:
            sets.append("version = version + 1")
        if generation is not None:
            sets.append("generation = ?"); args.append(json.dumps(generation))
        if eval_state is not None:
            sets.append("eval_state = ?"); args.append(json.dumps(eval_state))
        args.append(slug)
        await self._db.execute(f"UPDATE packs SET {', '.join(sets)} WHERE slug = ?", args)
        await self._db.commit()

    async def is_creator(self, user_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "SELECT 1 FROM creators WHERE user_id = ?", (user_id,),
        ) as cur:
            return await cur.fetchone() is not None

    async def add_creator(self, user_id: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT OR IGNORE INTO creators (user_id) VALUES (?)", (user_id,))
        await self._db.commit()

    # ── second opinions ──────────────────────────────────────────────

    async def create_second_opinion(self, session_id: str, basna_session_id: str,
                                    user_id: str) -> dict:
        assert self._db is not None
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO second_opinions (session_id, basna_session_id, user_id,"
            " status, created_at, updated_at) VALUES (?, ?, ?, 'running', ?, ?)",
            (session_id, basna_session_id, user_id, now, now))
        await self._db.commit()
        return {"session_id": session_id, "basna_session_id": basna_session_id,
                "user_id": user_id, "status": "running", "created_at": now}

    async def get_second_opinion(self, session_id: str, user_id: str) -> dict | None:
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM second_opinions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def set_second_opinion_status(self, session_id: str, status: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE second_opinions SET status = ?, updated_at = ? WHERE session_id = ?",
            (status, _utcnow(), session_id))
        await self._db.commit()

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
