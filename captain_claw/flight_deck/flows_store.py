"""Flows — persistent store for the Flight Deck Flow (process) engine.

A *Flow* is a declarative automation: a trigger plus an ordered list of steps
(deterministic tool calls + scoped agent-judgment nodes + branch/emit). Flows run
inside Flight Deck and dispatch their steps to the existing agent pool.

Storage is a dedicated SQLite DB (``flows.db``) in Flight Deck's data dir — it is
FD-global (one engine for the whole deck), unlike per-agent ``intentions.db``.

Three tables:
  flows           — the spec (trigger/steps/guardrails/output as JSON columns)
  flow_runs       — one row per execution (status + timing + error)
  flow_run_steps  — per-step results (for the live, step-by-step run log)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

_JSON_COLS = ("trigger", "steps", "guardrails", "output")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iso_after(seconds: int) -> str:
    from datetime import timedelta
    return (datetime.now(UTC) + timedelta(seconds=max(0, int(seconds)))).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class FlowStore:
    """Persistent store for Flows and their run history."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is not None:
            return self._db
        db = await aiosqlite.connect(str(self.db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS flows (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                description     TEXT,
                enabled         INTEGER NOT NULL DEFAULT 1,
                priority        INTEGER NOT NULL DEFAULT 50,
                trigger_json    TEXT,
                steps_json      TEXT,
                guardrails_json TEXT,
                output_json     TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_flows_enabled ON flows(enabled)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS flow_runs (
                id                   TEXT PRIMARY KEY,
                flow_id              TEXT NOT NULL,
                flow_name            TEXT,
                status               TEXT NOT NULL DEFAULT 'running',
                trigger_payload_json TEXT,
                started_at           TEXT NOT NULL,
                ended_at             TEXT,
                error                TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_flow ON flow_runs(flow_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON flow_runs(started_at)")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS flow_run_steps (
                run_id      TEXT NOT NULL,
                step_id     TEXT NOT NULL,
                seq         INTEGER NOT NULL,
                type        TEXT,
                status      TEXT NOT NULL,
                agent       TEXT,
                input_text  TEXT,
                output_text TEXT,
                ms          INTEGER,
                started_at  TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_steps_run ON flow_run_steps(run_id)")
        # Recursive flows: depth + frame (flow name) per step, for nested run logs.
        # ADD COLUMN on an existing DB is a no-op-if-present (wrapped in try).
        for col, ddl in (("depth", "INTEGER DEFAULT 0"), ("frame", "TEXT")):
            try:
                await db.execute(f"ALTER TABLE flow_run_steps ADD COLUMN {col} {ddl}")
            except Exception:
                pass  # column already exists
        # Synthesis (Phase 4): the scratch space — agent-authored throwaway flows
        # live alongside permanent ones, tagged by space/origin with dedup +
        # use-tracking + TTL columns. Migrate existing DBs in place.
        for col, ddl in (
            ("space", "TEXT NOT NULL DEFAULT 'user'"),   # 'user' | 'scratch'
            ("origin", "TEXT NOT NULL DEFAULT 'user'"),  # 'user' | 'agent'
            ("dsl_hash", "TEXT"),                        # canonical signature (dedup)
            ("author", "TEXT"),                          # which agent authored it
            ("use_count", "INTEGER NOT NULL DEFAULT 0"),
            ("last_used_at", "TEXT"),
            ("expires_at", "TEXT"),                      # TTL for GC (Phase 5)
        ):
            try:
                await db.execute(f"ALTER TABLE flows ADD COLUMN {col} {ddl}")
            except Exception:
                pass
        await db.execute("CREATE INDEX IF NOT EXISTS idx_flows_space ON flows(space)")
        await db.commit()
        self._db = db
        return db

    # ── flows CRUD ─────────────────────────────────────────────────────

    async def create_flow(self, spec: dict[str, Any]) -> str:
        db = await self._ensure_db()
        fid = str(spec.get("id") or "").strip() or _new_id("flow")
        now = _now()
        await db.execute(
            """INSERT INTO flows (id, name, description, enabled, priority,
                 trigger_json, steps_json, guardrails_json, output_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fid,
                str(spec.get("name") or "Untitled flow"),
                str(spec.get("description") or ""),
                1 if spec.get("enabled", True) else 0,
                int(spec.get("priority", 50)),
                json.dumps(spec.get("trigger") or {}),
                json.dumps(spec.get("steps") or []),
                json.dumps(spec.get("guardrails") or {}),
                json.dumps(spec.get("output") or {}),
                now,
                now,
            ),
        )
        await db.commit()
        return fid

    async def update_flow(self, fid: str, spec: dict[str, Any]) -> bool:
        db = await self._ensure_db()
        cur = await db.execute(
            """UPDATE flows SET name=?, description=?, enabled=?, priority=?,
                 trigger_json=?, steps_json=?, guardrails_json=?, output_json=?, updated_at=?
               WHERE id=?""",
            (
                str(spec.get("name") or "Untitled flow"),
                str(spec.get("description") or ""),
                1 if spec.get("enabled", True) else 0,
                int(spec.get("priority", 50)),
                json.dumps(spec.get("trigger") or {}),
                json.dumps(spec.get("steps") or []),
                json.dumps(spec.get("guardrails") or {}),
                json.dumps(spec.get("output") or {}),
                _now(),
                fid,
            ),
        )
        await db.commit()
        return cur.rowcount > 0

    async def set_enabled(self, fid: str, enabled: bool) -> bool:
        db = await self._ensure_db()
        cur = await db.execute(
            "UPDATE flows SET enabled=?, updated_at=? WHERE id=?",
            (1 if enabled else 0, _now(), fid),
        )
        await db.commit()
        return cur.rowcount > 0

    async def delete_flow(self, fid: str) -> bool:
        db = await self._ensure_db()
        cur = await db.execute("DELETE FROM flows WHERE id=?", (fid,))
        await db.commit()
        return cur.rowcount > 0

    async def get_flow(self, fid: str) -> dict[str, Any] | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM flows WHERE id=?", (fid,)) as cur:
            row = await cur.fetchone()
        return _row_to_flow(row) if row else None

    async def get_flow_by_name(self, name: str) -> dict[str, Any] | None:
        """Case-insensitive name lookup for `gosub`/`spawn` resolution. The
        **permanent** space wins over **scratch** (no silent shadowing); highest
        priority wins within a space."""
        db = await self._ensure_db()
        # Permanent first (space='user'), then scratch — ordered so the first row
        # is the winner.
        async with db.execute(
            """SELECT * FROM flows WHERE lower(name)=lower(?)
               ORDER BY (space='user') DESC, priority DESC LIMIT 1""",
            (name.strip(),),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_flow(row) if row else None

    async def list_flows(self) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        # The user-facing list shows the PERMANENT space only; scratch flows are
        # listed separately (list_scratch_flows).
        async with db.execute(
            "SELECT * FROM flows WHERE space='user' ORDER BY priority DESC, name"
        ) as cur:
            rows = await cur.fetchall()
        flows = [_row_to_flow(r) for r in rows]
        # Attach the most recent run summary for the list view.
        for f in flows:
            async with db.execute(
                "SELECT id, status, started_at, ended_at FROM flow_runs WHERE flow_id=? ORDER BY started_at DESC LIMIT 1",
                (f["id"],),
            ) as cur:
                lr = await cur.fetchone()
            f["last_run"] = dict(lr) if lr else None
        return flows

    async def enabled_flows(self) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        # Triggers only fire permanent flows; scratch flows are call-only.
        async with db.execute(
            "SELECT * FROM flows WHERE enabled=1 AND space='user' ORDER BY priority DESC, name"
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_flow(r) for r in rows]

    # ── scratch space (synthesized flows) ──────────────────────────────

    async def list_scratch_flows(self) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM flows WHERE space='scratch' ORDER BY last_used_at DESC, created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_flow(r) for r in rows]

    async def find_scratch_by_hash(self, dsl_hash: str) -> dict[str, Any] | None:
        if not dsl_hash:
            return None
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM flows WHERE space='scratch' AND dsl_hash=? LIMIT 1", (dsl_hash,)
        ) as cur:
            row = await cur.fetchone()
        return _row_to_flow(row) if row else None

    async def create_scratch_flow(
        self, spec: dict[str, Any], *, author: str = "", dsl_hash: str = "",
        ttl_seconds: int = 7 * 86400,
    ) -> str:
        """Store an agent-synthesized flow in the scratch space (origin=agent,
        call-only). Returns the new flow id."""
        db = await self._ensure_db()
        fid = _new_id("scratch")
        now = _now()
        expires = _iso_after(ttl_seconds)
        await db.execute(
            """INSERT INTO flows (id, name, description, enabled, priority,
                 trigger_json, steps_json, guardrails_json, output_json, created_at, updated_at,
                 space, origin, dsl_hash, author, use_count, last_used_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scratch', 'agent', ?, ?, 0, ?, ?)""",
            (
                fid,
                str(spec.get("name") or "Synthesized flow"),
                str(spec.get("description") or ""),
                1,
                int(spec.get("priority", 50)),
                json.dumps(spec.get("trigger") or {}),
                json.dumps(spec.get("steps") or []),
                json.dumps(spec.get("guardrails") or {}),
                json.dumps(spec.get("output") or {}),
                now, now, dsl_hash, author, now, expires,
            ),
        )
        await db.commit()
        return fid

    async def bump_use(self, flow_id: str, *, ttl_seconds: int = 7 * 86400) -> None:
        """Record a use of a scratch flow (count++, refresh last_used_at + TTL)."""
        db = await self._ensure_db()
        await db.execute(
            "UPDATE flows SET use_count=use_count+1, last_used_at=?, expires_at=? WHERE id=?",
            (_now(), _iso_after(ttl_seconds), flow_id),
        )
        await db.commit()

    async def promote_flow(self, flow_id: str, *, name: str | None = None) -> bool:
        """Move a scratch flow into the permanent space (Phase 5 wires the review
        UI; this is the underlying transition)."""
        db = await self._ensure_db()
        sets = ["space='user'", "origin='user'", "expires_at=NULL", "updated_at=?"]
        params: list[Any] = [_now()]
        if name:
            sets.insert(0, "name=?")
            params.insert(0, name)
        params.append(flow_id)
        cur = await db.execute(
            f"UPDATE flows SET {', '.join(sets)} WHERE id=? AND space='scratch'", params
        )
        await db.commit()
        return cur.rowcount > 0

    # ── runs ───────────────────────────────────────────────────────────

    async def start_run(self, flow_id: str, flow_name: str, payload: dict[str, Any] | None) -> str:
        db = await self._ensure_db()
        rid = _new_id("run")
        await db.execute(
            """INSERT INTO flow_runs (id, flow_id, flow_name, status, trigger_payload_json, started_at)
               VALUES (?, ?, ?, 'running', ?, ?)""",
            (rid, flow_id, flow_name, json.dumps(payload or {}), _now()),
        )
        await db.commit()
        return rid

    async def add_step_result(
        self, run_id: str, step_id: str, seq: int, *, type: str, status: str,
        agent: str = "", input_text: str = "", output_text: str = "", ms: int = 0,
        depth: int = 0, frame: str = "",
    ) -> None:
        db = await self._ensure_db()
        await db.execute(
            """INSERT INTO flow_run_steps (run_id, step_id, seq, type, status, agent, input_text, output_text, ms, started_at, depth, frame)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, step_id, seq, type, status, agent, input_text[:4000], output_text[:8000], ms, _now(), int(depth), frame),
        )
        await db.commit()

    async def set_run_status(self, run_id: str, status: str) -> None:
        """Update a live run's status (e.g. 'paused'/'running') without ending it."""
        db = await self._ensure_db()
        await db.execute(
            "UPDATE flow_runs SET status=? WHERE id=?", (status, run_id)
        )
        await db.commit()

    async def finish_run(self, run_id: str, status: str, error: str = "") -> None:
        db = await self._ensure_db()
        await db.execute(
            "UPDATE flow_runs SET status=?, ended_at=?, error=? WHERE id=?",
            (status, _now(), error or None, run_id),
        )
        await db.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM flow_runs WHERE id=?", (run_id,)) as cur:
            run = await cur.fetchone()
        if not run:
            return None
        async with db.execute(
            "SELECT * FROM flow_run_steps WHERE run_id=? ORDER BY seq", (run_id,)
        ) as cur:
            steps = await cur.fetchall()
        return {"run": dict(run), "steps": [dict(s) for s in steps]}

    async def list_runs(self, flow_id: str, limit: int = 50) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT id, status, started_at, ended_at, error FROM flow_runs WHERE flow_id=? ORDER BY started_at DESC LIMIT ?",
            (flow_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]


def _row_to_flow(row: Any) -> dict[str, Any]:
    d = dict(row)
    out: dict[str, Any] = {
        "id": d["id"],
        "name": d["name"],
        "description": d.get("description") or "",
        "enabled": bool(d.get("enabled", 1)),
        "priority": int(d.get("priority", 50)),
        "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"),
        # Synthesis / scratch metadata (defaults keep permanent flows unchanged).
        "space": d.get("space") or "user",
        "origin": d.get("origin") or "user",
        "author": d.get("author") or "",
        "use_count": int(d.get("use_count") or 0),
        "last_used_at": d.get("last_used_at"),
        "expires_at": d.get("expires_at"),
    }
    for col in _JSON_COLS:
        raw = d.get(f"{col}_json")
        try:
            out[col] = json.loads(raw) if raw else ({} if col != "steps" else [])
        except (ValueError, TypeError):
            out[col] = {} if col != "steps" else []
    return out
