"""Intentions — a control-plane primitive for *future actions under consideration*.

An **intention** sits between *noticing* (insights) and *doing* (cron/scheduler):
it carries a motivation (``why``), a trigger (when), and an approval lifecycle
(announce vs. ask) before it ever becomes a committed, executing task.

Two origins:
  * ``user``  — notes-to-self, surfaced back contextually (no approval).
  * ``agent`` — proactive proposals the agent announces (low-risk) or asks
    permission for (anything that sends/changes data).

Storage is a dedicated SQLite DB (``intentions.db``) next to ``sessions.db`` so
it travels with the agent's data dir and is isolated per agent (Flight Deck
sets a per-agent HOME). Decisions (the channel-agnostic "approve? / undo?"
queue) live in a sibling table so any channel can surface and resolve them.

Phase 1 = storage + CRUD + context injection. Channel wiring (delivery router,
WhatsApp/Flight-Deck/glasses resolvers, materialize-to-scheduler) is Phase 2.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)

# ── Vocabulary ────────────────────────────────────────────────────────

ORIGINS = frozenset({"user", "agent"})
RISKS = frozenset({"low", "normal", "high"})
APPROVAL_MODES = frozenset({"silent", "announce", "ask"})
STATUSES = frozenset({
    "proposed", "announced", "awaiting_approval", "active",
    "snoozed", "done", "declined", "expired", "cancelled",
})
# Statuses considered "open" — surfaced to the agent / eligible for action.
OPEN_STATUSES = frozenset({"proposed", "announced", "awaiting_approval", "active", "snoozed"})

CATEGORIES = frozenset({
    "reminder", "follow_up", "check_in", "automation", "suggestion", "other",
})

TRIGGER_TYPES = frozenset({"time", "event", "context", "manual"})
ACTION_TYPES = frozenset({"nudge", "run_prompt", "deliver", "materialize_schedule"})

DECISION_KINDS = frozenset({"approval", "announce_undo"})
DECISION_RESOLUTIONS = frozenset({"approved", "declined", "snoozed", "undone", "timeout"})

# JSON-serialised columns (parsed back to objects on read).
_JSON_FIELDS = ("trigger_spec", "action_spec", "audience", "provenance")
_DECISION_JSON_FIELDS = ("options", "target_hint")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _default_db_path() -> Path:
    """Place intentions.db alongside the session DB (respects per-agent HOME)."""
    try:
        base = Path(get_config().session.path).expanduser().parent
    except Exception:
        base = Path("~/.captain-claw").expanduser()
    return base / "intentions.db"


def _derive_approval_mode(origin: str, risk: str) -> str:
    """User notes are silent; agent low-risk announces; otherwise ask."""
    if origin == "user":
        return "silent"
    return "announce" if risk == "low" else "ask"


# ── Manager ───────────────────────────────────────────────────────────


class IntentionsManager:
    """Persistent store for intentions and their pending decisions."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path).expanduser() if db_path else _default_db_path()
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
            CREATE TABLE IF NOT EXISTS intentions (
                id                  TEXT PRIMARY KEY,
                origin              TEXT NOT NULL,
                title               TEXT NOT NULL,
                body                TEXT,
                why                 TEXT,
                category            TEXT,
                risk                TEXT NOT NULL DEFAULT 'normal',
                approval_mode       TEXT NOT NULL,
                status              TEXT NOT NULL,
                trigger_type        TEXT NOT NULL DEFAULT 'manual',
                trigger_spec        TEXT,
                action_type         TEXT NOT NULL DEFAULT 'nudge',
                action_spec         TEXT,
                repeat              TEXT,
                materialized_job_id TEXT,
                audience            TEXT,
                provenance          TEXT,
                source_session      TEXT,
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL,
                surfaced_at         TEXT,
                decided_at          TEXT,
                next_surface_at     TEXT,
                undo_until          TEXT,
                expires_at          TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_int_status ON intentions(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_int_origin ON intentions(origin)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_int_next ON intentions(next_surface_at)")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS intention_decisions (
                id            TEXT PRIMARY KEY,
                intention_id  TEXT NOT NULL,
                kind          TEXT NOT NULL,
                prompt_text   TEXT NOT NULL,
                options       TEXT,
                status        TEXT NOT NULL DEFAULT 'pending',
                resolution    TEXT,
                resolved_via  TEXT,
                target_hint   TEXT,
                created_at    TEXT NOT NULL,
                expires_at    TEXT,
                resolved_at   TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_dec_status ON intention_decisions(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_dec_intention ON intention_decisions(intention_id)")
        await db.commit()
        self._db = db
        return db

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            finally:
                self._db = None

    # ── intentions CRUD ──────────────────────────────────────────────

    async def create(
        self,
        *,
        origin: str,
        title: str,
        body: str = "",
        why: str = "",
        category: str = "other",
        risk: str = "normal",
        approval_mode: str | None = None,
        status: str | None = None,
        trigger_type: str = "manual",
        trigger_spec: dict | None = None,
        action_type: str = "nudge",
        action_spec: dict | None = None,
        repeat: str | None = None,
        audience: dict | None = None,
        provenance: dict | None = None,
        source_session: str = "",
        next_surface_at: str | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        origin = origin if origin in ORIGINS else "agent"
        risk = risk if risk in RISKS else "normal"
        if approval_mode not in APPROVAL_MODES:
            approval_mode = _derive_approval_mode(origin, risk)
        if status not in STATUSES:
            # Silent user notes go straight to active; agent proposals wait.
            status = "active" if approval_mode == "silent" else "proposed"
        now = _now_iso()
        row = {
            "id": _new_id(),
            "origin": origin,
            "title": title.strip(),
            "body": (body or "").strip(),
            "why": (why or "").strip(),
            "category": category if category in CATEGORIES else "other",
            "risk": risk,
            "approval_mode": approval_mode,
            "status": status,
            "trigger_type": trigger_type if trigger_type in TRIGGER_TYPES else "manual",
            "trigger_spec": json.dumps(trigger_spec) if trigger_spec else None,
            "action_type": action_type if action_type in ACTION_TYPES else "nudge",
            "action_spec": json.dumps(action_spec) if action_spec else None,
            "repeat": repeat,
            "materialized_job_id": None,
            "audience": json.dumps(audience) if audience else None,
            "provenance": json.dumps(provenance) if provenance else None,
            "source_session": source_session or "",
            "created_at": now,
            "updated_at": now,
            "surfaced_at": None,
            "decided_at": None,
            "next_surface_at": next_surface_at,
            "undo_until": None,
            "expires_at": expires_at,
        }
        db = await self._ensure_db()
        cols = ", ".join(row.keys())
        marks = ", ".join("?" for _ in row)
        await db.execute(f"INSERT INTO intentions ({cols}) VALUES ({marks})", tuple(row.values()))
        await db.commit()
        return _row_to_dict(row)

    async def get(self, intention_id: str) -> dict[str, Any] | None:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM intentions WHERE id = ?", (intention_id,)) as cur:
            r = await cur.fetchone()
        return _row_to_dict(dict(r)) if r else None

    async def list(
        self,
        *,
        origin: str | None = None,
        status: str | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        clauses, params = [], []
        if origin:
            clauses.append("origin = ?")
            params.append(origin)
        if status:
            clauses.append("status = ?")
            params.append(status)
        elif statuses:
            clauses.append(f"status IN ({', '.join('?' for _ in statuses)})")
            params.extend(statuses)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        async with db.execute(
            f"SELECT * FROM intentions {where} ORDER BY created_at DESC LIMIT ?", params
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_dict(dict(r)) for r in rows]

    async def update(self, intention_id: str, **fields: Any) -> bool:
        if not fields:
            return False
        allowed = {
            "title", "body", "why", "category", "risk", "approval_mode", "status",
            "trigger_type", "trigger_spec", "action_type", "action_spec", "repeat",
            "materialized_job_id", "audience", "provenance", "surfaced_at",
            "decided_at", "next_surface_at", "undo_until", "expires_at",
        }
        sets, params = [], []
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in _JSON_FIELDS and v is not None and not isinstance(v, str):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            params.append(v)
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(_now_iso())
        params.append(intention_id)
        db = await self._ensure_db()
        cur = await db.execute(
            f"UPDATE intentions SET {', '.join(sets)} WHERE id = ?", params
        )
        await db.commit()
        return cur.rowcount > 0

    async def set_status(self, intention_id: str, status: str, **extra: Any) -> bool:
        if status not in STATUSES:
            return False
        return await self.update(intention_id, status=status, **extra)

    async def get_for_context(self, limit: int = 10) -> list[dict[str, Any]]:
        """Open intentions to surface in the agent's prompt (newest first)."""
        return await self.list(statuses=list(OPEN_STATUSES), limit=limit)

    # ── decisions (channel-agnostic approve/undo queue) ──────────────

    async def create_decision(
        self,
        *,
        intention_id: str,
        kind: str,
        prompt_text: str,
        options: list[str] | None = None,
        target_hint: dict | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        kind = kind if kind in DECISION_KINDS else "approval"
        row = {
            "id": _new_id(),
            "intention_id": intention_id,
            "kind": kind,
            "prompt_text": prompt_text.strip(),
            "options": json.dumps(options or ["yes", "no", "later"]),
            "status": "pending",
            "resolution": None,
            "resolved_via": None,
            "target_hint": json.dumps(target_hint) if target_hint else None,
            "created_at": _now_iso(),
            "expires_at": expires_at,
            "resolved_at": None,
        }
        db = await self._ensure_db()
        cols = ", ".join(row.keys())
        marks = ", ".join("?" for _ in row)
        await db.execute(
            f"INSERT INTO intention_decisions ({cols}) VALUES ({marks})", tuple(row.values())
        )
        await db.commit()
        return _decision_to_dict(row)

    async def list_pending_decisions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM intention_decisions WHERE status = 'pending' "
            "ORDER BY created_at ASC LIMIT ?",
            (int(limit),),
        ) as cur:
            rows = await cur.fetchall()
        return [_decision_to_dict(dict(r)) for r in rows]

    async def resolve_decision(
        self, decision_id: str, resolution: str, via: str = ""
    ) -> dict[str, Any] | None:
        """Mark a pending decision resolved. Returns the decision row, or None."""
        if resolution not in DECISION_RESOLUTIONS:
            return None
        db = await self._ensure_db()
        async with db.execute(
            "SELECT * FROM intention_decisions WHERE id = ? AND status = 'pending'",
            (decision_id,),
        ) as cur:
            r = await cur.fetchone()
        if not r:
            return None
        await db.execute(
            "UPDATE intention_decisions SET status='resolved', resolution=?, "
            "resolved_via=?, resolved_at=? WHERE id = ?",
            (resolution, via, _now_iso(), decision_id),
        )
        await db.commit()
        out = dict(r)
        out.update({"status": "resolved", "resolution": resolution, "resolved_via": via})
        return _decision_to_dict(out)


# ── row parsing ───────────────────────────────────────────────────────


def _parse_json_fields(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    out = dict(row)
    for f in fields:
        v = out.get(f)
        if isinstance(v, str) and v:
            try:
                out[f] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
    return out


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return _parse_json_fields(row, _JSON_FIELDS)


def _decision_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    return _parse_json_fields(row, _DECISION_JSON_FIELDS)


# ── singleton ─────────────────────────────────────────────────────────

_manager: IntentionsManager | None = None


def get_intentions_manager() -> IntentionsManager:
    global _manager
    if _manager is None:
        _manager = IntentionsManager()
    return _manager
