"""Bat — durable store for the stubborn-finisher runs.

A *Bat run* drives one task through a long, restart-surviving loop until an
independent judge says it is done (Phase 3). This module is the durability
layer: a dedicated SQLite DB (``bat.db``) in Flight Deck's data dir, deck-isolated
via ``FD_DATA_DIR``.

Why its own store (not ``basna_sessions``): a Bat run may run for hours or days
and must survive an FD restart. That needs three things the Basna/Vatra spine
does not have — per-run **lease** columns (so exactly one FD drives a run, and a
crashed driver is re-adopted), per-step **checkpoints** written before and after
every attempt (so a restart resumes instead of redoing), and an append-only
**event** log (the in-memory ``_PROGRESS`` dict wipes under load and is lost on
restart).

Tables:
  bat_runs    — one row per run: status, origin, caps, cumulative spend, lease
  bat_steps   — per-step checkpoint, UPSERT on (run_id, step_key)
  bat_events  — append-only progress log for the live view

Mirrors ``dubina_store`` (aiosqlite + WAL, one shared connection). This module
imports nothing heavy from ``captain_claw`` so it stays cheap to import.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import aiosqlite

# Run lifecycle. 'retrying'/'waiting' are adoptable (a crashed driver resumes
# them); 'done'/'error'/'cancelled' are terminal.
RUNNING_STATES = ("planning", "running", "retrying", "waiting")
# awaiting_* runs stay adoptable so the supervisor can resume them once their
# ask is answered (the tick skips one whose ask is still open, so it won't spin).
ADOPTABLE_STATES = RUNNING_STATES + ("awaiting_plan", "awaiting_human")
TERMINAL_STATES = ("done", "error", "cancelled")

LEASE_STALE_SECONDS = 300  # a driver that hasn't heart-beaten in 5 min is dead


def _now() -> float:
    return time.time()


def _host() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "localhost"


def _loads(raw: Any, default: Any) -> Any:
    try:
        return json.loads(raw) if raw else default
    except (json.JSONDecodeError, TypeError):
        return default


_RUN_DDL = """
CREATE TABLE IF NOT EXISTS bat_runs (
    id                  TEXT PRIMARY KEY,
    owner_id            TEXT NOT NULL DEFAULT '',
    title               TEXT NOT NULL DEFAULT '',
    task                TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'planning',
    origin              TEXT NOT NULL DEFAULT '{}',
    source_host         TEXT NOT NULL DEFAULT 'localhost',
    source_port         INTEGER NOT NULL DEFAULT 0,
    config              TEXT NOT NULL DEFAULT '{}',
    cumulative_usd      REAL NOT NULL DEFAULT 0.0,
    cumulative_tokens   INTEGER NOT NULL DEFAULT 0,
    llm_usd_cap         REAL NOT NULL DEFAULT 0.0,   -- 0 = unbounded
    truth               TEXT NOT NULL DEFAULT '',
    analysis            TEXT NOT NULL DEFAULT '{}',
    stopped_reason      TEXT NOT NULL DEFAULT '',
    lease_host          TEXT NOT NULL DEFAULT '',
    lease_pid           INTEGER NOT NULL DEFAULT 0,
    lease_heartbeat_at  REAL NOT NULL DEFAULT 0.0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bat_runs_owner ON bat_runs(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bat_runs_status ON bat_runs(status);

CREATE TABLE IF NOT EXISTS bat_steps (
    run_id         TEXT NOT NULL,
    step_key       TEXT NOT NULL,
    seq            INTEGER NOT NULL DEFAULT 0,
    title          TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|skipped
    attempt        INTEGER NOT NULL DEFAULT 0,
    output         TEXT NOT NULL DEFAULT '',
    error          TEXT NOT NULL DEFAULT '',
    produced_file  TEXT NOT NULL DEFAULT '',
    updated_at     REAL NOT NULL,
    PRIMARY KEY (run_id, step_key)
);
CREATE INDEX IF NOT EXISTS idx_bat_steps_run ON bat_steps(run_id, seq);

CREATE TABLE IF NOT EXISTS bat_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    i         INTEGER NOT NULL DEFAULT 0,
    ts        REAL NOT NULL,
    stage     TEXT NOT NULL DEFAULT '',
    message   TEXT NOT NULL DEFAULT '',
    data      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_bat_events_run ON bat_events(run_id, i);

CREATE TABLE IF NOT EXISTS bat_asks (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    owner_id      TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL DEFAULT 'input',   -- plan_approval | input | secret
    question      TEXT NOT NULL DEFAULT '',
    options       TEXT NOT NULL DEFAULT '[]',
    step_key      TEXT NOT NULL DEFAULT '',
    secret        INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'open',     -- open | answered | cancelled | expired
    answer        TEXT NOT NULL DEFAULT '',         -- redacted when secret; the decision for plan_approval
    answered_via  TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    expires_at    REAL NOT NULL DEFAULT 0.0,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bat_asks_run ON bat_asks(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_bat_asks_owner_open ON bat_asks(owner_id, status);
"""


class BatStore:
    """Durable store for Bat runs, their step checkpoints and their event log."""

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
        await db.executescript(_RUN_DDL)
        await db.commit()
        self._db = db
        return db

    # ── runs ─────────────────────────────────────────────────────────

    async def create_run(
        self,
        *,
        run_id: str,
        owner_id: str,
        title: str,
        task: str,
        config: dict[str, Any] | None = None,
        origin: dict[str, Any] | None = None,
        source_host: str = "localhost",
        source_port: int = 0,
        llm_usd_cap: float = 0.0,
        status: str = "planning",
    ) -> str:
        db = await self._ensure_db()
        now = _now()
        await db.execute(
            """INSERT INTO bat_runs
               (id, owner_id, title, task, status, origin, source_host,
                source_port, config, llm_usd_cap, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, owner_id, title, task, status, json.dumps(origin or {}),
             source_host, int(source_port), json.dumps(config or {}),
             float(llm_usd_cap or 0.0), now, now),
        )
        await db.commit()
        return run_id

    async def get_run(self, run_id: str) -> dict | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM bat_runs WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        return _row_to_run(row) if row else None

    async def list_runs(self, owner_id: str, limit: int = 50) -> list[dict]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM bat_runs WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?",
            (owner_id, int(limit)),
        ) as cur:
            return [_row_to_run(r) for r in await cur.fetchall()]

    async def set_status(
        self,
        run_id: str,
        status: str,
        *,
        truth: str | None = None,
        analysis: dict[str, Any] | None = None,
        stopped_reason: str | None = None,
    ) -> None:
        db = await self._ensure_db()
        sets = ["status = ?", "updated_at = ?"]
        vals: list[Any] = [status, _now()]
        if truth is not None:
            sets.append("truth = ?")
            vals.append(truth)
        if analysis is not None:
            sets.append("analysis = ?")
            vals.append(json.dumps(analysis))
        if stopped_reason is not None:
            sets.append("stopped_reason = ?")
            vals.append(stopped_reason)
        vals.append(run_id)
        await db.execute(f"UPDATE bat_runs SET {', '.join(sets)} WHERE id = ?", vals)
        await db.commit()

    async def set_config(self, run_id: str, config: dict[str, Any]) -> None:
        db = await self._ensure_db()
        await db.execute("UPDATE bat_runs SET config = ?, updated_at = ? WHERE id = ?",
                         (json.dumps(config or {}), _now(), run_id))
        await db.commit()

    async def bump_cost(self, run_id: str, usd: float, tokens: int = 0) -> float:
        """Atomically add to the persisted cumulative spend. Returns the new USD
        total so a cap can be enforced across restarts (in-memory counters reset)."""
        db = await self._ensure_db()
        await db.execute(
            """UPDATE bat_runs
               SET cumulative_usd = cumulative_usd + ?,
                   cumulative_tokens = cumulative_tokens + ?,
                   updated_at = ?
               WHERE id = ?""",
            (float(usd or 0.0), int(tokens or 0), _now(), run_id),
        )
        await db.commit()
        async with db.execute("SELECT cumulative_usd FROM bat_runs WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        return float(row["cumulative_usd"]) if row else 0.0

    # ── lease (exactly one driver per run; crashed driver re-adopted) ──

    async def claim_lease(
        self,
        run_id: str,
        *,
        pid: int | None = None,
        host: str = "",
        now: float | None = None,
        stale_after: int = LEASE_STALE_SECONDS,
    ) -> bool:
        """Take the right to drive this run. A live lease (heart-beaten within
        ``stale_after`` and, on this host, a pid that still exists) is refused;
        a stale or dead lease is taken over. Returns True if we now hold it."""
        db = await self._ensure_db()
        now = now if now is not None else _now()
        pid = int(pid if pid is not None else os.getpid())
        host = host or _host()
        async with db.execute(
            "SELECT lease_host, lease_pid, lease_heartbeat_at FROM bat_runs WHERE id = ?",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        held_by_other = not (int(row["lease_pid"]) == pid and row["lease_host"] == host)
        if row["lease_pid"] and held_by_other:
            fresh = (now - float(row["lease_heartbeat_at"] or 0.0)) < stale_after
            alive = True
            if fresh and row["lease_host"] == host:
                try:
                    os.kill(int(row["lease_pid"]), 0)
                except (OSError, ProcessLookupError):
                    alive = False
            if fresh and alive:
                return False
        await db.execute(
            "UPDATE bat_runs SET lease_host = ?, lease_pid = ?, lease_heartbeat_at = ?,"
            " updated_at = ? WHERE id = ?",
            (host, pid, now, now, run_id),
        )
        await db.commit()
        return True

    async def heartbeat_lease(
        self, run_id: str, *, pid: int | None = None, host: str = "", now: float | None = None,
    ) -> bool:
        """Keep the lease warm. False when it has moved on — the driver stands down."""
        db = await self._ensure_db()
        now = now if now is not None else _now()
        pid = int(pid if pid is not None else os.getpid())
        host = host or _host()
        await db.execute(
            "UPDATE bat_runs SET lease_heartbeat_at = ? WHERE id = ? AND lease_pid = ? AND lease_host = ?",
            (now, run_id, pid, host),
        )
        await db.commit()
        async with db.execute(
            "SELECT lease_pid, lease_host FROM bat_runs WHERE id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
        return bool(row and int(row["lease_pid"]) == pid and row["lease_host"] == host)

    async def release_lease(self, run_id: str, *, pid: int | None = None, host: str = "") -> None:
        db = await self._ensure_db()
        pid = int(pid if pid is not None else os.getpid())
        host = host or _host()
        await db.execute(
            "UPDATE bat_runs SET lease_host = '', lease_pid = 0, lease_heartbeat_at = 0"
            " WHERE id = ? AND lease_pid = ? AND lease_host = ?",
            (run_id, pid, host),
        )
        await db.commit()

    async def adoptable_runs(
        self, *, now: float | None = None, stale_after: int = LEASE_STALE_SECONDS,
    ) -> list[dict]:
        """Non-terminal runs whose lease is empty or stale — candidates to drive
        or (after a restart) re-adopt."""
        db = await self._ensure_db()
        now = now if now is not None else _now()
        placeholders = ",".join("?" for _ in ADOPTABLE_STATES)
        async with db.execute(
            f"SELECT * FROM bat_runs WHERE status IN ({placeholders}) ORDER BY created_at",
            ADOPTABLE_STATES,
        ) as cur:
            rows = [_row_to_run(r) for r in await cur.fetchall()]
        out = []
        for r in rows:
            if not r["lease_pid"] or (now - float(r["lease_heartbeat_at"] or 0.0)) >= stale_after:
                out.append(r)
        return out

    # ── steps (checkpoints) ───────────────────────────────────────────

    async def seed_steps(self, run_id: str, steps: list[dict[str, Any]]) -> None:
        """Insert the planned steps if they are not present yet (idempotent — a
        re-adopt after restart keeps existing checkpoints)."""
        db = await self._ensure_db()
        now = _now()
        for seq, s in enumerate(steps):
            await db.execute(
                """INSERT OR IGNORE INTO bat_steps
                   (run_id, step_key, seq, title, status, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?)""",
                (run_id, str(s["step_key"]), int(s.get("seq", seq)),
                 str(s.get("title", "")), now),
            )
        await db.commit()

    async def upsert_step(
        self,
        run_id: str,
        step_key: str,
        *,
        status: str | None = None,
        title: str | None = None,
        output: str | None = None,
        error: str | None = None,
        produced_file: str | None = None,
        attempt: int | None = None,
        seq: int | None = None,
    ) -> None:
        db = await self._ensure_db()
        now = _now()
        await db.execute(
            """INSERT INTO bat_steps (run_id, step_key, seq, title, status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, step_key) DO NOTHING""",
            (run_id, step_key, int(seq or 0), title or "", status or "pending", now),
        )
        sets = ["updated_at = ?"]
        vals: list[Any] = [now]
        for col, v in (("status", status), ("title", title), ("output", output),
                       ("error", error), ("produced_file", produced_file)):
            if v is not None:
                sets.append(f"{col} = ?")
                vals.append(v)
        if attempt is not None:
            sets.append("attempt = ?")
            vals.append(int(attempt))
        if seq is not None:
            sets.append("seq = ?")
            vals.append(int(seq))
        vals.extend([run_id, step_key])
        await db.execute(
            f"UPDATE bat_steps SET {', '.join(sets)} WHERE run_id = ? AND step_key = ?", vals,
        )
        await db.commit()

    async def demote_running_steps(self, run_id: str) -> int:
        """On adopt after a crash, a step left 'running' never completed — demote
        it to 'pending' so the driver re-runs it (avoids the plans.py wedge where
        a 'running' step is never picked up again)."""
        db = await self._ensure_db()
        cur = await db.execute(
            "UPDATE bat_steps SET status = 'pending', updated_at = ?"
            " WHERE run_id = ? AND status = 'running'",
            (_now(), run_id),
        )
        await db.commit()
        return cur.rowcount

    async def list_steps(self, run_id: str) -> list[dict]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM bat_steps WHERE run_id = ? ORDER BY seq, step_key", (run_id,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── events (append-only progress) ─────────────────────────────────

    async def append_event(
        self, run_id: str, stage: str, message: str = "", **data: Any,
    ) -> int:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT COALESCE(MAX(i), -1) + 1 AS nxt FROM bat_events WHERE run_id = ?",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
        i = int(row["nxt"]) if row else 0
        await db.execute(
            "INSERT INTO bat_events (run_id, i, ts, stage, message, data) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, i, _now(), stage, message, json.dumps(data or {})),
        )
        await db.commit()
        return i

    async def list_events(self, run_id: str, since: int = 0, limit: int = 2000) -> list[dict]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM bat_events WHERE run_id = ? AND i >= ? ORDER BY i LIMIT ?",
            (run_id, int(since), int(limit)),
        ) as cur:
            out = []
            for r in await cur.fetchall():
                ev = dict(r)
                ev["data"] = _loads(ev.get("data"), {})
                out.append(ev)
            return out


    # ── asks (human-in-the-loop) ──────────────────────────────────────

    async def create_ask(
        self, *, ask_id: str, run_id: str, owner_id: str, kind: str, question: str,
        options: list | None = None, step_key: str = "", secret: bool = False,
        expires_at: float = 0.0,
    ) -> str:
        db = await self._ensure_db()
        now = _now()
        await db.execute(
            """INSERT INTO bat_asks
               (id, run_id, owner_id, kind, question, options, step_key, secret,
                status, created_at, expires_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
            (ask_id, run_id, owner_id, kind, question, json.dumps(options or []),
             step_key, 1 if secret else 0, now, float(expires_at or 0.0), now),
        )
        await db.commit()
        return ask_id

    async def get_ask(self, ask_id: str) -> dict | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM bat_asks WHERE id = ?", (ask_id,)) as cur:
            row = await cur.fetchone()
        return _row_to_ask(row) if row else None

    async def latest_open_ask(self, run_id: str) -> dict | None:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM bat_asks WHERE run_id = ? AND status = 'open' ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_ask(row) if row else None

    async def latest_ask(self, run_id: str) -> dict | None:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM bat_asks WHERE run_id = ? ORDER BY created_at DESC LIMIT 1", (run_id,),
        ) as cur:
            row = await cur.fetchone()
        return _row_to_ask(row) if row else None

    async def open_asks_for_owner(self, owner_id: str) -> list[dict]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM bat_asks WHERE owner_id = ? AND status = 'open' ORDER BY created_at",
            (owner_id,),
        ) as cur:
            return [_row_to_ask(r) for r in await cur.fetchall()]

    async def answer_ask(self, ask_id: str, answer: str, *, via: str = "") -> bool:
        """Compare-and-set open → answered. Returns False if it was already
        resolved (idempotent against double answers from two channels)."""
        db = await self._ensure_db()
        cur = await db.execute(
            "UPDATE bat_asks SET status = 'answered', answer = ?, answered_via = ?, updated_at = ?"
            " WHERE id = ? AND status = 'open'",
            (answer, via, _now(), ask_id),
        )
        await db.commit()
        return cur.rowcount > 0

    async def cancel_ask(self, ask_id: str) -> None:
        db = await self._ensure_db()
        await db.execute(
            "UPDATE bat_asks SET status = 'cancelled', updated_at = ? WHERE id = ? AND status = 'open'",
            (_now(), ask_id),
        )
        await db.commit()

    async def expire_asks(self, now: float | None = None) -> list[str]:
        """Mark open, past-expiry asks as expired. Returns their run_ids so the
        caller can fail those runs (a timed-out human-ask is a serious stop)."""
        db = await self._ensure_db()
        now = now if now is not None else _now()
        async with db.execute(
            "SELECT id, run_id FROM bat_asks WHERE status = 'open' AND expires_at > 0 AND expires_at <= ?",
            (now,),
        ) as cur:
            rows = await cur.fetchall()
        for r in rows:
            await db.execute(
                "UPDATE bat_asks SET status = 'expired', updated_at = ? WHERE id = ?", (now, r["id"]))
        await db.commit()
        return [r["run_id"] for r in rows]


def _row_to_ask(row: aiosqlite.Row) -> dict:
    ask = dict(row)
    ask["options"] = _loads(ask.get("options"), [])
    ask["secret"] = bool(ask.get("secret"))
    return ask


def _row_to_run(row: aiosqlite.Row) -> dict:
    run = dict(row)
    run["origin"] = _loads(run.get("origin"), {})
    run["config"] = _loads(run.get("config"), {})
    run["analysis"] = _loads(run.get("analysis"), {})
    return run
