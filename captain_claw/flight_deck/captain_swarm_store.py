"""Captain Swarm — the durable store for runs, experiments and their events.

docs/captain-swarm-plan.md §3. A run keeps improving one number; each
experiment is one attempt at it, run in one lane and measured by Captain. A
run may take a night, so everything it decides is written down as it goes and
a restart picks up where it was (the same lease pattern as ``bat_store``:
exactly one driver per run, a dead driver's run re-adopted).

Tables, in ``captain_swarm.db`` in Flight Deck's data dir (deck-isolated via
``FD_DATA_DIR``):

  cs_runs         one row per run: objective, state, baseline, champion, spend, lease
  cs_experiments  one row per attempt: hypothesis, lane, child, scores, verdict, lesson
  cs_events       append-only log for the page
  cs_versions     the texts a text-subject run tries (a set of rules): every version,
                  each with the one it was made from
  cs_intents      what a rehearsal would have done in the world, and didn't: an
                  outward call recorded instead of made, or a commit point

An experiment's ``base_sha`` / ``head_sha`` name what it started from and what it
made: a commit for code, a version id for a text subject.

Imports nothing heavy, so it stays cheap to import.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import aiosqlite

# 'awaiting_go': an agent started it and the owner hasn't said Go yet.
ACTIVE_STATES = ("baseline", "running")
ADOPTABLE_STATES = ACTIVE_STATES
WAITING_STATES = ("awaiting_go",)
TERMINAL_STATES = ("done", "stalled", "stopped", "failed", "cancelled", "declined")
# Experiment lifecycle. 'kept'/'discarded'/'disqualified'/'failed' are final.
EXP_FINAL = ("kept", "discarded", "disqualified", "failed")
LEASE_STALE_SECONDS = 300

_DDL = """
CREATE TABLE IF NOT EXISTS cs_runs (
    id                  TEXT PRIMARY KEY,
    owner_id            TEXT NOT NULL,
    title               TEXT NOT NULL DEFAULT '',
    objective           TEXT NOT NULL DEFAULT '{}',
    status              TEXT NOT NULL,
    stopped_reason      TEXT NOT NULL DEFAULT '',
    started_by          TEXT NOT NULL DEFAULT '',
    baseline            TEXT NOT NULL DEFAULT '{}',
    champion            TEXT NOT NULL DEFAULT '{}',
    state               TEXT NOT NULL DEFAULT '{}',
    round               INTEGER NOT NULL DEFAULT 0,
    experiments         INTEGER NOT NULL DEFAULT 0,
    spent_usd           REAL NOT NULL DEFAULT 0.0,
    spent_tokens        INTEGER NOT NULL DEFAULT 0,
    started_at          REAL NOT NULL DEFAULT 0.0,
    finished_at         REAL NOT NULL DEFAULT 0.0,
    lease_host          TEXT NOT NULL DEFAULT '',
    lease_pid           INTEGER NOT NULL DEFAULT 0,
    lease_heartbeat_at  REAL NOT NULL DEFAULT 0.0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cs_runs_owner ON cs_runs(owner_id, created_at);

CREATE TABLE IF NOT EXISTS cs_experiments (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    round           INTEGER NOT NULL,
    n               INTEGER NOT NULL,
    hypothesis      TEXT NOT NULL DEFAULT '',
    strategy_tag    TEXT NOT NULL DEFAULT '',
    why             TEXT NOT NULL DEFAULT '',
    lane            TEXT NOT NULL DEFAULT '',
    child_id        TEXT NOT NULL DEFAULT '',
    base_sha        TEXT NOT NULL DEFAULT '',
    branch          TEXT NOT NULL DEFAULT '',
    head_sha        TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL,
    verdict_reason  TEXT NOT NULL DEFAULT '',
    dev             TEXT NOT NULL DEFAULT '{}',
    holdout         TEXT NOT NULL DEFAULT '{}',
    guardrails      TEXT NOT NULL DEFAULT '{}',
    diff_stat       TEXT NOT NULL DEFAULT '',
    changed_files   TEXT NOT NULL DEFAULT '[]',
    lesson          TEXT NOT NULL DEFAULT '',
    cost_usd        REAL NOT NULL DEFAULT 0.0,
    tokens          INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cs_exp_run ON cs_experiments(run_id, round, n);

CREATE TABLE IF NOT EXISTS cs_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    ts          REAL NOT NULL,
    stage       TEXT NOT NULL,
    message     TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS cs_events_run ON cs_events(run_id, id);

CREATE TABLE IF NOT EXISTS cs_versions (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    parent      TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cs_versions_run ON cs_versions(run_id, created_at);

CREATE TABLE IF NOT EXISTS cs_intents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    exp_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    tool        TEXT NOT NULL,
    what        TEXT NOT NULL DEFAULT '',
    args        TEXT NOT NULL DEFAULT '{}',
    agent       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cs_intents_exp ON cs_intents(run_id, exp_id, id);
"""

# Columns added after a table first shipped: added in place on an older database.
_EXP_COLUMNS = (("change", "TEXT NOT NULL DEFAULT '{}'"),)

_RUN_JSON = ("objective", "baseline", "champion", "state")
_EXP_JSON = ("dev", "holdout", "guardrails", "changed_files", "change")


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
    except (ValueError, TypeError):
        return default


def _run(row: aiosqlite.Row) -> dict[str, Any]:
    out = dict(row)
    for k in _RUN_JSON:
        out[k] = _loads(out.get(k), {})
    return out


def _exp(row: aiosqlite.Row) -> dict[str, Any]:
    out = dict(row)
    for k in _EXP_JSON:
        out[k] = _loads(out.get(k), [] if k == "changed_files" else {})
    return out


class CaptainSwarmStore:
    """Durable store for Captain Swarm runs, their experiments and their events."""

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
        await db.executescript(_DDL)
        async with db.execute("PRAGMA table_info(cs_experiments)") as cur:
            have = {r["name"] for r in await cur.fetchall()}
        for col, ddl in _EXP_COLUMNS:
            if col not in have:
                await db.execute(f"ALTER TABLE cs_experiments ADD COLUMN {col} {ddl}")
        await db.commit()
        self._db = db
        return db

    # ── runs ─────────────────────────────────────────────────────────

    async def create_run(self, *, run_id: str, owner_id: str, title: str, objective: dict[str, Any],
                         status: str, started_by: str = "") -> dict[str, Any]:
        db = await self._ensure_db()
        now = _now()
        await db.execute(
            "INSERT INTO cs_runs (id, owner_id, title, objective, status, started_by, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, owner_id, title[:200], json.dumps(objective), status, started_by, now, now))
        await db.commit()
        return await self.get_run(run_id)  # type: ignore[return-value]

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM cs_runs WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        return _run(row) if row else None

    async def list_runs(self, owner_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM cs_runs WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?",
            (owner_id, int(limit))) as cur:
            return [_run(r) for r in await cur.fetchall()]

    async def update_run(self, run_id: str, **fields: Any) -> None:
        """Set *fields* on the run; the JSON columns take dicts."""
        if not fields:
            return
        db = await self._ensure_db()
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k} = ?")
            vals.append(json.dumps(v) if k in _RUN_JSON else v)
        cols.append("updated_at = ?")
        vals.append(_now())
        await db.execute(f"UPDATE cs_runs SET {', '.join(cols)} WHERE id = ?", (*vals, run_id))
        await db.commit()

    async def set_status(self, run_id: str, status: str, reason: str = "") -> None:
        fields: dict[str, Any] = {"status": status}
        if reason or status in TERMINAL_STATES:
            fields["stopped_reason"] = reason
        if status in TERMINAL_STATES:
            fields["finished_at"] = _now()
        await self.update_run(run_id, **fields)

    async def add_spend(self, run_id: str, usd: float, tokens: int) -> None:
        if not usd and not tokens:
            return
        db = await self._ensure_db()
        await db.execute(
            "UPDATE cs_runs SET spent_usd = spent_usd + ?, spent_tokens = spent_tokens + ?, updated_at = ?"
            " WHERE id = ?", (float(usd), int(tokens), _now(), run_id))
        await db.commit()

    # ── experiments ──────────────────────────────────────────────────

    async def add_experiment(self, *, exp_id: str, run_id: str, round_no: int, n: int, hypothesis: str,
                             strategy_tag: str = "", why: str = "", lane: str = "",
                             base_sha: str = "", change: dict[str, Any] | None = None) -> dict[str, Any]:
        db = await self._ensure_db()
        now = _now()
        await db.execute(
            "INSERT INTO cs_experiments (id, run_id, round, n, hypothesis, strategy_tag, why, lane, base_sha,"
            " change, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)",
            (exp_id, run_id, round_no, n, hypothesis[:2000], strategy_tag[:60], why[:1000], lane, base_sha,
             json.dumps(change or {}), now, now))
        await db.execute("UPDATE cs_runs SET experiments = experiments + 1, updated_at = ? WHERE id = ?",
                         (now, run_id))
        await db.commit()
        return await self.get_experiment(exp_id)  # type: ignore[return-value]

    async def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM cs_experiments WHERE id = ?", (exp_id,)) as cur:
            row = await cur.fetchone()
        return _exp(row) if row else None

    async def list_experiments(self, run_id: str) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM cs_experiments WHERE run_id = ? ORDER BY round, n", (run_id,)) as cur:
            return [_exp(r) for r in await cur.fetchall()]

    async def update_experiment(self, exp_id: str, **fields: Any) -> None:
        if not fields:
            return
        db = await self._ensure_db()
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k} = ?")
            vals.append(json.dumps(v) if k in _EXP_JSON else v)
        cols.append("updated_at = ?")
        vals.append(_now())
        await db.execute(f"UPDATE cs_experiments SET {', '.join(cols)} WHERE id = ?", (*vals, exp_id))
        await db.commit()

    # ── versions (text subjects) ─────────────────────────────────────

    async def add_version(self, *, version_id: str, run_id: str, content: dict[str, Any], parent: str = "") -> None:
        db = await self._ensure_db()
        await db.execute("INSERT INTO cs_versions (id, run_id, parent, content, created_at) VALUES (?, ?, ?, ?, ?)",
                         (version_id, run_id, parent, json.dumps(content), _now()))
        await db.commit()

    async def get_version(self, version_id: str) -> dict[str, Any] | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM cs_versions WHERE id = ?", (version_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return {**dict(row), "content": _loads(row["content"], {})}

    # ── intents (rehearsals) ─────────────────────────────────────────

    async def add_intent(self, *, run_id: str, exp_id: str, kind: str, tool: str, what: str,
                         args: dict[str, Any], agent: str = "") -> int:
        db = await self._ensure_db()
        cur = await db.execute(
            "INSERT INTO cs_intents (run_id, exp_id, kind, tool, what, args, agent, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, exp_id, kind[:20], tool[:80], what[:500], json.dumps(args)[:20000], agent[:120], _now()))
        await db.commit()
        return int(cur.lastrowid or 0)

    async def list_intents(self, run_id: str, exp_id: str = "", *, limit: int = 500) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        q, args = "SELECT * FROM cs_intents WHERE run_id = ?", [run_id]
        if exp_id:
            q += " AND exp_id = ?"
            args.append(exp_id)
        async with db.execute(q + " ORDER BY id LIMIT ?", (*args, int(limit))) as cur:
            return [{**dict(r), "args": _loads(r["args"], {})} for r in await cur.fetchall()]

    async def count_intents(self, run_id: str, exp_id: str) -> int:
        db = await self._ensure_db()
        async with db.execute("SELECT COUNT(*) FROM cs_intents WHERE run_id = ? AND exp_id = ?",
                              (run_id, exp_id)) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ── events ───────────────────────────────────────────────────────

    async def append_event(self, run_id: str, stage: str, message: str, detail: str = "") -> None:
        db = await self._ensure_db()
        await db.execute("INSERT INTO cs_events (run_id, ts, stage, message, detail) VALUES (?, ?, ?, ?, ?)",
                         (run_id, _now(), stage, message[:1000], detail[:4000]))
        await db.commit()

    async def list_events(self, run_id: str, *, since: int = 0, limit: int = 300) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM cs_events WHERE run_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
            (run_id, int(since), int(limit))) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        return list(reversed(rows))

    # ── lease (exactly one driver per run) ───────────────────────────

    async def claim_lease(self, run_id: str, *, pid: int | None = None, host: str = "",
                          now: float | None = None, stale_after: int = LEASE_STALE_SECONDS) -> bool:
        """Take the right to drive *run_id*. A live lease (fresh, and on this
        host a pid that still exists) is refused; a stale or dead one is taken."""
        db = await self._ensure_db()
        now = now if now is not None else _now()
        pid = int(pid if pid is not None else os.getpid())
        host = host or _host()
        async with db.execute("SELECT lease_host, lease_pid, lease_heartbeat_at FROM cs_runs WHERE id = ?",
                              (run_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        if row["lease_pid"] and not (int(row["lease_pid"]) == pid and row["lease_host"] == host):
            fresh = (now - float(row["lease_heartbeat_at"] or 0.0)) < stale_after
            alive = True
            if fresh and row["lease_host"] == host:
                try:
                    os.kill(int(row["lease_pid"]), 0)
                except (OSError, ProcessLookupError):
                    alive = False
            if fresh and alive:
                return False
        await db.execute("UPDATE cs_runs SET lease_host = ?, lease_pid = ?, lease_heartbeat_at = ? WHERE id = ?",
                         (host, pid, now, run_id))
        await db.commit()
        return True

    async def heartbeat_lease(self, run_id: str, *, pid: int | None = None, host: str = "") -> bool:
        """Keep the lease warm; False when another driver has it — stand down."""
        db = await self._ensure_db()
        pid = int(pid if pid is not None else os.getpid())
        host = host or _host()
        await db.execute("UPDATE cs_runs SET lease_heartbeat_at = ? WHERE id = ? AND lease_pid = ? AND lease_host = ?",
                         (_now(), run_id, pid, host))
        await db.commit()
        async with db.execute("SELECT lease_pid, lease_host FROM cs_runs WHERE id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        return bool(row and int(row["lease_pid"]) == pid and row["lease_host"] == host)

    async def release_lease(self, run_id: str, *, pid: int | None = None, host: str = "") -> None:
        db = await self._ensure_db()
        pid = int(pid if pid is not None else os.getpid())
        host = host or _host()
        await db.execute("UPDATE cs_runs SET lease_host = '', lease_pid = 0, lease_heartbeat_at = 0"
                         " WHERE id = ? AND lease_pid = ? AND lease_host = ?", (run_id, pid, host))
        await db.commit()

    async def adoptable_runs(self, *, now: float | None = None,
                             stale_after: int = LEASE_STALE_SECONDS) -> list[dict[str, Any]]:
        """Active runs nobody is driving: the lease is empty or stale."""
        db = await self._ensure_db()
        now = now if now is not None else _now()
        marks = ",".join("?" for _ in ADOPTABLE_STATES)
        async with db.execute(f"SELECT * FROM cs_runs WHERE status IN ({marks}) ORDER BY created_at",
                              ADOPTABLE_STATES) as cur:
            rows = [_run(r) for r in await cur.fetchall()]
        return [r for r in rows
                if not r["lease_pid"] or (now - float(r["lease_heartbeat_at"] or 0.0)) >= stale_after]
