"""Dubina — persistent store for Frontier Horizon runs.

A *run* drives one task through the horizon engine (escalation ladder + verifier).
Storage is a dedicated SQLite DB (``dubina.db``) in Flight Deck's data dir, FD-global.

Run history is **split per track** (design decision): coder and reasoning produce
different artifacts (diffs + test results vs. answer + claims) and different metrics
(tests-passing vs. agreement/critic-survival), so a shared table would be all-nullable.

Tables:
  dubina_coder_runs   — one row per coder run (status/timing/result/cost)
  dubina_reason_runs  — one row per reasoning run (same shape)
  dubina_run_steps    — per-attempt event log for the live run view (both tracks)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

TRACKS = ("coder", "reason")
_RUN_TABLE = {"coder": "dubina_coder_runs", "reason": "dubina_reason_runs"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return f"dub_{uuid.uuid4().hex[:12]}"


def _run_table(track: str) -> str:
    try:
        return _RUN_TABLE[track]
    except KeyError:
        raise ValueError(f"unknown track {track!r} (expected one of {TRACKS})") from None


_RUN_COLUMNS = """
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL DEFAULT '',
    task           TEXT NOT NULL DEFAULT '',
    base_tier      TEXT NOT NULL DEFAULT '',
    max_tier       TEXT NOT NULL DEFAULT '',
    compute_budget REAL NOT NULL DEFAULT 0.0,
    status         TEXT NOT NULL DEFAULT 'running',  -- running|passed|failed|budget|error
    passed         INTEGER,                          -- NULL until finished
    stopped_reason TEXT NOT NULL DEFAULT '',
    cost_spent     REAL NOT NULL DEFAULT 0.0,
    config         TEXT NOT NULL DEFAULT '{}',       -- JSON: samples/fix/modes/test_command...
    result         TEXT NOT NULL DEFAULT '{}',       -- JSON: final code/answer summary
    error          TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
"""


class DubinaStore:
    """Persistent store for Frontier Horizon runs and their step logs."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        await self._ensure_db()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is not None:
            return self._db
        db = await aiosqlite.connect(str(self.db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.executescript(f"""
            CREATE TABLE IF NOT EXISTS dubina_coder_runs ({_RUN_COLUMNS});
            CREATE TABLE IF NOT EXISTS dubina_reason_runs ({_RUN_COLUMNS});
            CREATE INDEX IF NOT EXISTS idx_dubina_coder_user
                ON dubina_coder_runs(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_dubina_reason_user
                ON dubina_reason_runs(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS dubina_run_steps (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                track       TEXT NOT NULL DEFAULT '',
                seq         INTEGER NOT NULL DEFAULT 0,  -- event order within a run
                step_id     TEXT NOT NULL DEFAULT '',
                tier        TEXT NOT NULL DEFAULT '',
                rung        INTEGER NOT NULL DEFAULT 0,
                kind        TEXT NOT NULL DEFAULT '',     -- single|vote|fix
                samples     INTEGER NOT NULL DEFAULT 0,
                passed      INTEGER NOT NULL DEFAULT 0,
                confidence  REAL NOT NULL DEFAULT 0.0,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_dubina_steps_run
                ON dubina_run_steps(run_id, seq);
        """)
        await db.commit()
        self._db = db
        return db

    # ── runs ─────────────────────────────────────────────────────────

    async def create_run(
        self,
        track: str,
        user_id: str,
        task: str,
        base_tier: str,
        max_tier: str,
        compute_budget: float,
        config: dict[str, Any] | None = None,
    ) -> str:
        table = _run_table(track)
        db = await self._ensure_db()
        run_id = _new_id()
        now = _now()
        await db.execute(
            f"""INSERT INTO {table}
                (id, user_id, task, base_tier, max_tier, compute_budget,
                 status, config, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
            (run_id, user_id, task, base_tier, max_tier, float(compute_budget),
             json.dumps(config or {}), now, now),
        )
        await db.commit()
        return run_id

    async def append_step(self, run_id: str, track: str, seq: int, event: dict[str, Any]) -> None:
        """Persist one engine event (a ladder attempt) for the live run view."""
        db = await self._ensure_db()
        await db.execute(
            """INSERT INTO dubina_run_steps
               (run_id, track, seq, step_id, tier, rung, kind, samples,
                passed, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, track, seq, str(event.get("step", "")), str(event.get("tier", "")),
             int(event.get("rung", 0)), str(event.get("kind", "")),
             int(event.get("samples", 0)), 1 if event.get("passed") else 0,
             float(event.get("confidence", 0.0)), _now()),
        )
        await db.commit()

    async def finish_run(
        self,
        track: str,
        run_id: str,
        *,
        status: str,
        passed: bool,
        stopped_reason: str,
        cost_spent: float,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        table = _run_table(track)
        db = await self._ensure_db()
        await db.execute(
            f"""UPDATE {table}
                SET status = ?, passed = ?, stopped_reason = ?, cost_spent = ?,
                    result = ?, error = ?, updated_at = ?
                WHERE id = ?""",
            (status, 1 if passed else 0, stopped_reason, float(cost_spent),
             json.dumps(result or {}), error, _now(), run_id),
        )
        await db.commit()

    async def get_run(self, track: str, run_id: str) -> dict | None:
        table = _run_table(track)
        db = await self._ensure_db()
        async with db.execute(f"SELECT * FROM {table} WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        run = _row_to_run(row)
        async with db.execute(
            "SELECT * FROM dubina_run_steps WHERE run_id = ? ORDER BY seq", (run_id,)
        ) as cur:
            run["steps"] = [dict(r) for r in await cur.fetchall()]
        return run

    async def list_runs(self, track: str, user_id: str, limit: int = 50) -> list[dict]:
        table = _run_table(track)
        db = await self._ensure_db()
        async with db.execute(
            f"""SELECT * FROM {table} WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?""",
            (user_id, int(limit)),
        ) as cur:
            return [_row_to_run(r) for r in await cur.fetchall()]


def _row_to_run(row: aiosqlite.Row) -> dict:
    run = dict(row)
    for col in ("config", "result"):
        try:
            run[col] = json.loads(run.get(col) or "{}")
        except (json.JSONDecodeError, TypeError):
            run[col] = {}
    if run.get("passed") is not None:
        run["passed"] = bool(run["passed"])
    return run
