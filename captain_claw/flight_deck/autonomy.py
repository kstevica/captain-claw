"""Autonomous Work — the closed-loop substrate (see docs/autonomous-work-plan.md).

This module owns two things, both Flight-Deck-level and per-user:

  * ``AutonomyStore`` — a small SQLite store (sync sqlite3 + lock, mirroring
    ``ConsciousnessStore``) holding the **action ledger** (the goal backlog the
    Arbiter writes to and the page reads from) and the **reliability** table
    (how well each action *kind* has worked, learned over time).
  * ``resolve_config`` — the effective ``AutonomousWorkConfig`` for one user:
    global defaults overlaid with that user's overrides from ``user_settings``,
    with ``autonomy_level`` clamped to the shipped ``max_autonomy_level`` ceiling.

Phase 1 ships the store, the config resolution, and the API surface. Nothing
acts yet: the Arbiter (Topic 1), dispatch (Topic 2), learning (Topic 3) and the
reflection feed (Topic 4) land in later phases and write through here.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from captain_claw.config import get_config

# autonomy_level ladder, weakest → strongest. Used for clamping to the ceiling.
AUTONOMY_LEVELS = ("off", "propose", "act_low_risk", "act")

# Lifecycle of a ledger row.
ACTION_STATUSES = (
    "candidate", "queued", "awaiting_approval", "dispatched",
    "done", "rejected", "expired",
)


def _norm_user(user_id: str | None) -> str:
    """Empty (auth disabled / internal) → 'local', matching consciousness."""
    return (user_id or "").strip() or "local"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "autonomy.db"
    cfg_path = (get_config().autonomous_work.db_path or "~/.captain-claw/autonomy.db")
    return Path(cfg_path).expanduser()


# ── Config resolution (global defaults + per-user overrides) ───────────

def _clamp_level(level: str, ceiling: str) -> str:
    """Clamp an autonomy level to the shipped ceiling (never exceed it)."""
    try:
        want = AUTONOMY_LEVELS.index(level)
    except ValueError:
        want = AUTONOMY_LEVELS.index("propose")
    try:
        cap = AUTONOMY_LEVELS.index(ceiling)
    except ValueError:
        cap = AUTONOMY_LEVELS.index("propose")
    return AUTONOMY_LEVELS[min(want, cap)]


def global_defaults() -> dict[str, Any]:
    """The global ``AutonomousWorkConfig`` as a plain dict."""
    return get_config().autonomous_work.model_dump()


def merge_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Overlay a per-user override dict on the global defaults and clamp level.

    Unknown keys in ``overrides`` are ignored so the stored blob can't inject
    junk; only keys present in the config model are honoured.
    """
    base = global_defaults()
    if overrides:
        for k, v in overrides.items():
            if k in base and k != "max_autonomy_level":
                base[k] = v
    base["autonomy_level"] = _clamp_level(
        str(base.get("autonomy_level") or "propose"),
        str(base.get("max_autonomy_level") or "propose"),
    )
    return base


def resolve_config(user_id: str) -> dict[str, Any]:
    """Effective config for ``user_id``: global defaults + their stored overrides.

    Per-user overrides live in the autonomy store's own ``autonomy_config`` table
    (keyed by the normalized user id, ``local`` in standalone). Self-contained —
    no FK to the users table — so it works identically in single-user and
    multi-tenant deployments.
    """
    return merge_config(get_store().get_overrides(user_id))


def save_config(user_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Persist a per-user override blob and return the new effective config."""
    # Keep only recognised, non-ceiling keys — the ceiling is server-owned.
    allowed = {k for k in global_defaults() if k != "max_autonomy_level"}
    clean = {k: v for k, v in (overrides or {}).items() if k in allowed}
    get_store().set_overrides(user_id, clean)
    return resolve_config(user_id)


# ── Store ──────────────────────────────────────────────────────────────

class AutonomyStore:
    """SQLite-backed action ledger + reliability. Sync sqlite3 + a lock,
    mirroring ConsciousnessStore."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                -- The goal backlog / ledger. One row per autonomous action the
                -- Arbiter considered or took, with full rationale for the audit
                -- trail surfaced on the Autonomous Work page.
                CREATE TABLE IF NOT EXISTS autonomous_actions (
                    id            TEXT PRIMARY KEY,
                    user_id       TEXT NOT NULL,
                    source        TEXT NOT NULL DEFAULT 'heartbeat',  -- reflection|intuition|intention|heartbeat|manual
                    kind          TEXT NOT NULL DEFAULT 'nudge',      -- nudge|run_prompt|basna|materialize_schedule
                    title         TEXT NOT NULL DEFAULT '',
                    rationale     TEXT NOT NULL DEFAULT '',
                    risk          TEXT NOT NULL DEFAULT 'normal',     -- low|normal|high
                    domain        TEXT NOT NULL DEFAULT 'general',
                    score         REAL NOT NULL DEFAULT 0.0,
                    status        TEXT NOT NULL DEFAULT 'candidate',
                    target        TEXT NOT NULL DEFAULT '',           -- agent slug / channel
                    ref_id        TEXT NOT NULL DEFAULT '',           -- linked intention/cron/basna id
                    payload       TEXT NOT NULL DEFAULT '{}',         -- json: kind-specific args
                    outcome       TEXT,                               -- success|fail|NULL
                    outcome_note  TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL,
                    dispatched_at TEXT,
                    completed_at  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_actions_user
                    ON autonomous_actions(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_actions_status
                    ON autonomous_actions(user_id, status);

                -- Learned reliability per (user, action kind, domain). Mirrors
                -- archetype_reliability; weight is Bayesian-shrunk toward seed.
                CREATE TABLE IF NOT EXISTS autonomy_reliability (
                    user_id    TEXT NOT NULL,
                    kind       TEXT NOT NULL,
                    domain     TEXT NOT NULL DEFAULT 'general',
                    successes  INTEGER NOT NULL DEFAULT 0,
                    fails      INTEGER NOT NULL DEFAULT 0,
                    runs       INTEGER NOT NULL DEFAULT 0,
                    weight     REAL NOT NULL DEFAULT 0.6,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, kind, domain)
                );

                -- Per-user config override blob (JSON). Self-owned, no FK, so it
                -- works for the 'local' standalone bucket and real tenants alike.
                CREATE TABLE IF NOT EXISTS autonomy_config (
                    user_id    TEXT PRIMARY KEY,
                    overrides  TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._c().commit()

    # ── per-user config overrides ─────────────────────────────────────

    def get_overrides(self, user_id: str) -> dict[str, Any]:
        uid = _norm_user(user_id)
        with self._lock:
            r = self._c().execute(
                "SELECT overrides FROM autonomy_config WHERE user_id = ?", (uid,)
            ).fetchone()
        if not r:
            return {}
        try:
            parsed = json.loads(r["overrides"] or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}

    def set_overrides(self, user_id: str, overrides: dict[str, Any]) -> None:
        uid = _norm_user(user_id)
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            conn.execute(
                "INSERT INTO autonomy_config (user_id, overrides, updated_at)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET"
                " overrides = excluded.overrides, updated_at = excluded.updated_at",
                (uid, json.dumps(overrides or {}), now),
            )
            conn.commit()

    # ── action ledger ─────────────────────────────────────────────────

    def add_action(
        self,
        user_id: str,
        *,
        kind: str,
        title: str,
        rationale: str = "",
        source: str = "heartbeat",
        risk: str = "normal",
        domain: str = "general",
        score: float = 0.0,
        status: str = "candidate",
        target: str = "",
        ref_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        uid = _norm_user(user_id)
        aid = "act_" + secrets.token_hex(8)
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            conn.execute(
                """
                INSERT INTO autonomous_actions
                    (id, user_id, source, kind, title, rationale, risk, domain,
                     score, status, target, ref_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (aid, uid, source, kind, title, rationale, risk, domain,
                 float(score), status, target, ref_id,
                 json.dumps(payload or {}), now),
            )
            conn.commit()
        return self.get_action(aid) or {}

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._c().execute(
                "SELECT * FROM autonomous_actions WHERE id = ?", (action_id,)
            ).fetchone()
        return self._row_to_action(r) if r else None

    def list_actions(
        self, user_id: str, *, status: str | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        uid = _norm_user(user_id)
        limit = max(1, min(500, limit))
        with self._lock:
            if status:
                rows = self._c().execute(
                    "SELECT * FROM autonomous_actions WHERE user_id = ? AND status = ?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (uid, status, limit),
                ).fetchall()
            else:
                rows = self._c().execute(
                    "SELECT * FROM autonomous_actions WHERE user_id = ?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (uid, limit),
                ).fetchall()
        return [self._row_to_action(r) for r in rows]

    def update_status(
        self,
        action_id: str,
        status: str,
        *,
        ref_id: str | None = None,
        outcome: str | None = None,
        outcome_note: str | None = None,
    ) -> dict[str, Any] | None:
        now = _utcnow_iso()
        sets = ["status = ?"]
        vals: list[Any] = [status]
        if ref_id is not None:
            sets.append("ref_id = ?")
            vals.append(ref_id)
        if outcome is not None:
            sets.append("outcome = ?")
            vals.append(outcome)
        if outcome_note is not None:
            sets.append("outcome_note = ?")
            vals.append(outcome_note)
        if status == "dispatched":
            sets.append("dispatched_at = ?")
            vals.append(now)
        if status in ("done", "rejected", "expired"):
            sets.append("completed_at = ?")
            vals.append(now)
        vals.append(action_id)
        with self._lock:
            conn = self._c()
            conn.execute(
                f"UPDATE autonomous_actions SET {', '.join(sets)} WHERE id = ?", vals
            )
            conn.commit()
        return self.get_action(action_id)

    @staticmethod
    def _row_to_action(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        try:
            d["payload"] = json.loads(d.get("payload") or "{}")
        except (ValueError, TypeError):
            d["payload"] = {}
        return d

    # ── reliability (learning) ─────────────────────────────────────────

    @staticmethod
    def _reliability_weight(
        successes: int, fails: int, seed: float = 0.6, alpha: float = 4.0,
    ) -> float:
        """Bayesian-shrunk success rate toward ``seed``, penalising fails 2×.

        Identical shape to the Basna archetype-reliability weight so the two
        learning loops behave the same.
        """
        numerator = successes + alpha * seed
        denominator = successes + 2.0 * fails + alpha
        w = numerator / denominator if denominator > 0 else seed
        return max(0.05, min(0.99, w))

    def record_outcome(
        self, user_id: str, kind: str, domain: str, success: bool, *, seed: float = 0.6,
    ) -> dict[str, Any]:
        uid = _norm_user(user_id)
        dom = (domain or "general").strip() or "general"
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            row = conn.execute(
                "SELECT successes, fails FROM autonomy_reliability"
                " WHERE user_id = ? AND kind = ? AND domain = ?",
                (uid, kind, dom),
            ).fetchone()
            successes = (row["successes"] if row else 0) + (1 if success else 0)
            fails = (row["fails"] if row else 0) + (0 if success else 1)
            weight = self._reliability_weight(successes, fails, seed=seed)
            conn.execute(
                """
                INSERT INTO autonomy_reliability
                    (user_id, kind, domain, successes, fails, runs, weight, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, kind, domain) DO UPDATE SET
                    successes = excluded.successes,
                    fails = excluded.fails,
                    runs = excluded.runs,
                    weight = excluded.weight,
                    updated_at = excluded.updated_at
                """,
                (uid, kind, dom, successes, fails, successes + fails, weight, now),
            )
            conn.commit()
        return {
            "user_id": uid, "kind": kind, "domain": dom,
            "successes": successes, "fails": fails, "runs": successes + fails,
            "weight": weight, "updated_at": now,
        }

    def list_reliability(self, user_id: str) -> list[dict[str, Any]]:
        uid = _norm_user(user_id)
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM autonomy_reliability WHERE user_id = ?"
                " ORDER BY weight DESC",
                (uid,),
            ).fetchall()
        return [dict(r) for r in rows]


_STORE: AutonomyStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> AutonomyStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = AutonomyStore()
        return _STORE
