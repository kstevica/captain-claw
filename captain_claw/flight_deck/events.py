"""Event-ingestion spine (see docs/jarvis-actions-events-plan.md, #2).

A dedicated, normalized store for real-world signals — calendar changes, new mail,
webhooks — so the autonomous loop reacts to the *user's world*, not just to its own
reflections. Source adapters (pollers / webhook receivers) normalize signals into
``external_events``; the Arbiter reads ``new`` events as candidates and marks them
``surfaced``. Self-owned DB (matches the per-subsystem convention), WAL, status-flag
idempotency (``new → surfaced → acted | ignored``), dedup via ``dedup_key``.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ``claimed``: taken by one of First Mate's factories (first_mate_events), waiting
# for its batch; the Arbiter reads only ``new``, so it never sees one.
EVENT_STATUSES = ("new", "surfaced", "acted", "ignored", "claimed")

# A row ``claim_handled`` wrote for a signal its source hasn't ingested yet:
# only the key (no type, no summary), so the source's later ``add_event`` is
# deduped. It has nothing to show — listings leave it out — until that
# ``add_event`` fills in the signal itself.
_PLACEHOLDER_SQL = "(event_type = '' AND summary = '')"


def _norm_user(user_id: str | None) -> str:
    return (user_id or "").strip() or "local"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    base = os.environ.get("FD_DATA_DIR", "").strip()
    if base:
        return Path(base).expanduser().resolve() / "events.db"
    return Path("~/.captain-claw/events.db").expanduser()


class EventsStore:
    """SQLite store for external events. Sync sqlite3 + lock, mirroring the other
    Flight Deck subsystem stores."""

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
            conn.execute("PRAGMA journal_mode=WAL")
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        with self._lock:
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS external_events (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    source      TEXT NOT NULL,            -- gmail | calendar | webhook | manual | …
                    event_type  TEXT NOT NULL DEFAULT '', -- new_email | event_changed | …
                    summary     TEXT NOT NULL DEFAULT '', -- human-readable headline
                    body        TEXT NOT NULL DEFAULT '', -- full content
                    metadata    TEXT NOT NULL DEFAULT '{}',
                    dedup_key   TEXT NOT NULL DEFAULT '', -- source id; prevents re-ingest
                    status      TEXT NOT NULL DEFAULT 'new',
                    surface_count INTEGER NOT NULL DEFAULT 0, -- arbiter passes that saw it
                    ingested_at TEXT NOT NULL,
                    processed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_user
                    ON external_events(user_id, status, ingested_at DESC);
                -- Dedup only applies when a dedup_key is provided.
                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup
                    ON external_events(user_id, source, dedup_key)
                    WHERE dedup_key != '';

                -- Per-(user, source) poll state: when we last polled + an opaque
                -- cursor (calendar syncToken, gmail historyId, …).
                CREATE TABLE IF NOT EXISTS poll_state (
                    user_id      TEXT NOT NULL,
                    source       TEXT NOT NULL,
                    last_poll_at REAL NOT NULL DEFAULT 0,
                    cursor       TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (user_id, source)
                );

                -- Open loops the arbiter chose to TRACK rather than act on now:
                -- soft reminders / requests / waiting-on-you items. They resurface
                -- to the arbiter when due and escalate with age (see arbiter.py).
                CREATE TABLE IF NOT EXISTS follow_ups (
                    id            TEXT PRIMARY KEY,
                    user_id       TEXT NOT NULL,
                    source        TEXT NOT NULL DEFAULT '',   -- gmail | calendar | manual | …
                    summary       TEXT NOT NULL DEFAULT '',   -- the ask / headline
                    detail        TEXT NOT NULL DEFAULT '',   -- who from + what they want
                    origin_event_id TEXT NOT NULL DEFAULT '', -- external_events.id (dedup)
                    status        TEXT NOT NULL DEFAULT 'open',-- open | done | dismissed | stale
                    created_at    TEXT NOT NULL,
                    follow_up_at  TEXT NOT NULL,              -- next time to resurface
                    surfaced_count INTEGER NOT NULL DEFAULT 0,-- times re-fed to the arbiter
                    nudged_count  INTEGER NOT NULL DEFAULT 0, -- nudges actually sent
                    last_surfaced_at TEXT,
                    updated_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_followups_due
                    ON follow_ups(user_id, status, follow_up_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_followups_origin
                    ON follow_ups(user_id, origin_event_id)
                    WHERE origin_event_id != '';
                """
            )
            # Migrate pre-existing DBs that predate surface_count.
            cols = {r["name"] for r in self._c().execute(
                "PRAGMA table_info(external_events)").fetchall()}
            if "surface_count" not in cols:
                self._c().execute(
                    "ALTER TABLE external_events ADD COLUMN surface_count INTEGER NOT NULL DEFAULT 0")
            self._c().commit()

    # ── poll state (for source adapters) ───────────────────────────────

    def get_poll_state(self, user_id: str, source: str) -> dict[str, Any]:
        uid = _norm_user(user_id)
        with self._lock:
            r = self._c().execute(
                "SELECT last_poll_at, cursor FROM poll_state WHERE user_id = ? AND source = ?",
                (uid, source),
            ).fetchone()
        return {"last_poll_at": float(r["last_poll_at"]) if r else 0.0,
                "cursor": (r["cursor"] if r else "") or ""}

    def set_poll_state(self, user_id: str, source: str, *, last_poll_at: float, cursor: str | None = None) -> None:
        uid = _norm_user(user_id)
        with self._lock:
            conn = self._c()
            existing = self.get_poll_state(uid, source)
            cur = existing["cursor"] if cursor is None else cursor
            conn.execute(
                "INSERT INTO poll_state (user_id, source, last_poll_at, cursor)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(user_id, source) DO UPDATE SET"
                " last_poll_at = excluded.last_poll_at, cursor = excluded.cursor",
                (uid, source, last_poll_at, cur),
            )
            conn.commit()

    def add_event(
        self,
        user_id: str,
        *,
        source: str,
        event_type: str = "",
        summary: str = "",
        body: str = "",
        metadata: dict[str, Any] | None = None,
        dedup_key: str = "",
    ) -> dict[str, Any] | None:
        """Insert one event. With a ``dedup_key`` already seen (same user+source),
        it's a no-op returning None — except that a placeholder
        :meth:`claim_handled` left for it gets this event's content (its status
        kept, so it stays handled); otherwise returns the new row."""
        uid = _norm_user(user_id)
        eid = "evt_" + secrets.token_hex(8)
        now = _utcnow_iso()
        content = (event_type, summary[:500], body[:8000], json.dumps(metadata or {}))
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "INSERT OR IGNORE INTO external_events"
                " (id, user_id, source, event_type, summary, body, metadata, dedup_key,"
                "  status, ingested_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'new', ?)",
                (eid, uid, source, *content, dedup_key, now),
            )
            if cur.rowcount == 0 and dedup_key and (event_type or summary):
                conn.execute(
                    "UPDATE external_events SET event_type = ?, summary = ?, body = ?,"
                    " metadata = ? WHERE user_id = ? AND source = ? AND dedup_key = ?"
                    f" AND {_PLACEHOLDER_SQL}",
                    (*content, uid, source, dedup_key),
                )
            conn.commit()
            if cur.rowcount == 0:
                return None  # deduped
        return self.get_event(eid)

    def claim_handled(self, user_id: str, source: str, dedup_key: str) -> bool:
        """Mark the signal ``dedup_key`` as already handled elsewhere (e.g. a
        Gmail topic auto-reply), so the arbiter never picks it up. An event
        already ingested keeps its content: a ``new`` one flips to ``acted``, a
        surfaced/acted/ignored one is left alone. Only when there is none yet is
        an ``acted`` placeholder written — so a later ``add_event`` of it is
        deduped (returns None) and fills it in; until then listings leave it
        out (it has nothing to show). True when a row was flipped or written."""
        if not dedup_key:
            return False  # nothing to dedup against
        uid = _norm_user(user_id)
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "UPDATE external_events SET status = 'acted', processed_at = ?"
                " WHERE user_id = ? AND source = ? AND dedup_key = ? AND status = 'new'",
                (now, uid, source, dedup_key),
            )
            if cur.rowcount == 0:
                # No new event to flip: a placeholder, unless a row is there
                # already (the unique key then ignores it).
                cur = conn.execute(
                    "INSERT OR IGNORE INTO external_events"
                    " (id, user_id, source, dedup_key, status, ingested_at, processed_at)"
                    " VALUES (?, ?, ?, ?, 'acted', ?, ?)",
                    ("evt_" + secrets.token_hex(8), uid, source, dedup_key, now, now),
                )
            conn.commit()
            return cur.rowcount > 0

    def release_claim(self, user_id: str, source: str, dedup_key: str) -> bool:
        """Undo :meth:`claim_handled` for a signal its claimant ended up NOT
        handling (an auto-reply that was skipped or failed), so the arbiter
        sees it after all: a claimed event goes back to ``new``, and a bare
        placeholder is dropped so a later ``add_event`` ingests it fresh.
        Only ``acted`` rows are touched — the spine writes that status for
        claims alone — so an event the arbiter surfaced or ignored is left
        as it is. True when a row changed."""
        if not dedup_key:
            return False
        uid = _norm_user(user_id)
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "DELETE FROM external_events WHERE user_id = ? AND source = ?"
                f" AND dedup_key = ? AND status = 'acted' AND {_PLACEHOLDER_SQL}",
                (uid, source, dedup_key),
            )
            changed = cur.rowcount
            if not changed:
                cur = conn.execute(
                    "UPDATE external_events SET status = 'new', processed_at = NULL"
                    " WHERE user_id = ? AND source = ? AND dedup_key = ? AND status = 'acted'",
                    (uid, source, dedup_key),
                )
                changed = cur.rowcount
            conn.commit()
            return changed > 0

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._c().execute(
                "SELECT * FROM external_events WHERE id = ?", (event_id,)
            ).fetchone()
        return self._row(r) if r else None

    def list_new(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Oldest-first unprocessed events — the arbiter's intake queue."""
        uid = _norm_user(user_id)
        limit = max(1, min(100, limit))
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM external_events WHERE user_id = ? AND status = 'new'"
                " ORDER BY ingested_at ASC LIMIT ?",
                (uid, limit),
            ).fetchall()
        return [self._row(r) for r in rows]

    def list_events(self, user_id: str, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Newest-first events (GET /fd/events, the Autonomous Work page) —
        without the placeholders :meth:`claim_handled` leaves for a signal not
        ingested yet: they have nothing to show."""
        uid = _norm_user(user_id)
        limit = max(1, min(500, limit))
        with self._lock:
            if status:
                rows = self._c().execute(
                    "SELECT * FROM external_events WHERE user_id = ? AND status = ?"
                    f" AND NOT {_PLACEHOLDER_SQL}"
                    " ORDER BY ingested_at DESC LIMIT ?",
                    (uid, status, limit),
                ).fetchall()
            else:
                rows = self._c().execute(
                    "SELECT * FROM external_events WHERE user_id = ?"
                    f" AND NOT {_PLACEHOLDER_SQL}"
                    " ORDER BY ingested_at DESC LIMIT ?",
                    (uid, limit),
                ).fetchall()
        return [self._row(r) for r in rows]

    def mark(self, event_ids: list[str], status: str) -> None:
        """Set the events' status. An ``acted`` event (claimed by
        :meth:`claim_handled`, handled elsewhere) is never moved back to
        another status: a pass that read it as ``new`` just before the claim
        can't undo it."""
        if not event_ids or status not in EVENT_STATUSES:
            return
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            conn.executemany(
                "UPDATE external_events SET status = ?, processed_at = ?"
                " WHERE id = ? AND (status != 'acted' OR ? = 'acted')",
                [(status, now, eid, status) for eid in event_ids],
            )
            conn.commit()

    # ── First Mate's claims (first_mate_events) ───────────────────────

    def claim_for(self, event_id: str, meta: dict[str, Any]) -> bool:
        """Take a ``new`` event for a First Mate factory: status ``claimed``,
        ``meta`` merged into its metadata, ``processed_at`` = when. False when
        it isn't ``new`` any more (the Arbiter or another claim got it first)."""
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            row = conn.execute("SELECT metadata FROM external_events WHERE id = ? AND status = 'new'",
                               (event_id,)).fetchone()
            if row is None:
                return False
            try:
                merged = {**json.loads(row["metadata"] or "{}"), **meta}
            except (ValueError, TypeError):
                merged = dict(meta)
            cur = conn.execute(
                "UPDATE external_events SET status = 'claimed', metadata = ?, processed_at = ?"
                " WHERE id = ? AND status = 'new'", (json.dumps(merged), now, event_id))
            conn.commit()
            return cur.rowcount > 0

    def list_claimed(self, user_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """A user's claimed events, oldest claim first."""
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM external_events WHERE user_id = ? AND status = 'claimed'"
                " ORDER BY processed_at ASC, ingested_at ASC LIMIT ?",
                (_norm_user(user_id), max(1, min(1000, limit)))).fetchall()
        return [self._row(r) for r in rows]

    def settle_claimed(self, event_ids: list[str], meta: dict[str, Any]) -> int:
        """Claimed events a run took: status ``acted``, ``meta`` (the run)
        merged into their metadata."""
        return self._move_claimed(event_ids, "acted", lambda m: {**m, **meta})

    def unclaim(self, event_ids: list[str]) -> int:
        """Claimed events back to ``new`` for the Arbiter, without First Mate's keys."""
        return self._move_claimed(event_ids, "new",
                                  lambda m: {k: v for k, v in m.items() if not k.startswith("fm_")})

    def _move_claimed(self, event_ids: list[str], status: str, change) -> int:
        if not event_ids:
            return 0
        now = None if status == "new" else _utcnow_iso()
        moved = 0
        with self._lock:
            conn = self._c()
            for eid in event_ids:
                row = conn.execute("SELECT metadata FROM external_events WHERE id = ? AND status = 'claimed'",
                                   (eid,)).fetchone()
                if row is None:
                    continue
                try:
                    meta = json.loads(row["metadata"] or "{}")
                except (ValueError, TypeError):
                    meta = {}
                moved += conn.execute(
                    "UPDATE external_events SET status = ?, metadata = ?, processed_at = ?"
                    " WHERE id = ? AND status = 'claimed'",
                    (status, json.dumps(change(meta)), now, eid)).rowcount
            conn.commit()
        return moved

    def defer(self, event_ids: list[str], max_attempts: int = 4) -> list[str]:
        """A pass saw these events but produced no action. Bump their attempt
        counter and keep them ``new`` so the next pass (pulse OR manual) sees them
        again — unless they've now been seen ``max_attempts`` times, in which case
        flip them to ``ignored`` so they stop competing. Returns the ids given up
        on (newly ignored)."""
        if not event_ids:
            return []
        max_attempts = max(1, max_attempts)
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            conn.executemany(
                "UPDATE external_events SET surface_count = surface_count + 1"
                " WHERE id = ? AND status = 'new'",
                [(eid,) for eid in event_ids],
            )
            cur = conn.execute(
                "SELECT id FROM external_events WHERE status = 'new'"
                " AND surface_count >= ? AND id IN (%s)"
                % ",".join("?" * len(event_ids)),
                (max_attempts, *event_ids),
            )
            spent = [r["id"] for r in cur.fetchall()]
            if spent:
                conn.executemany(
                    "UPDATE external_events SET status = 'ignored', processed_at = ?"
                    " WHERE id = ?",
                    [(now, eid) for eid in spent],
                )
            conn.commit()
        return spent

    # ── follow-ups (tracked open loops) ─────────────────────────────────

    def add_follow_up(
        self,
        user_id: str,
        *,
        summary: str,
        detail: str = "",
        source: str = "",
        origin_event_id: str = "",
        follow_up_at: str,
    ) -> dict[str, Any] | None:
        """Track a soft reminder/request as an open loop due at ``follow_up_at``.
        Deduped by ``origin_event_id`` (same email is never tracked twice) — a
        repeat returns None."""
        uid = _norm_user(user_id)
        fid = "fu_" + secrets.token_hex(8)
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "INSERT OR IGNORE INTO follow_ups"
                " (id, user_id, source, summary, detail, origin_event_id, status,"
                "  created_at, follow_up_at, surfaced_count, nudged_count, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, 0, 0, ?)",
                (fid, uid, source, summary[:500], detail[:2000], origin_event_id,
                 now, follow_up_at, now),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None  # already tracked
        return self.get_follow_up(fid)

    def get_follow_up(self, follow_up_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._c().execute(
                "SELECT * FROM follow_ups WHERE id = ?", (follow_up_id,)
            ).fetchone()
        return dict(r) if r else None

    def list_due_follow_ups(self, user_id: str, now_iso: str, limit: int = 5) -> list[dict[str, Any]]:
        """Open follow-ups whose time has come, soonest-due first."""
        uid = _norm_user(user_id)
        limit = max(1, min(50, limit))
        with self._lock:
            rows = self._c().execute(
                "SELECT * FROM follow_ups WHERE user_id = ? AND status = 'open'"
                " AND follow_up_at <= ? ORDER BY follow_up_at ASC LIMIT ?",
                (uid, now_iso, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_follow_ups(self, user_id: str, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        uid = _norm_user(user_id)
        limit = max(1, min(500, limit))
        with self._lock:
            if status:
                rows = self._c().execute(
                    "SELECT * FROM follow_ups WHERE user_id = ? AND status = ?"
                    " ORDER BY follow_up_at ASC LIMIT ?",
                    (uid, status, limit),
                ).fetchall()
            else:
                rows = self._c().execute(
                    "SELECT * FROM follow_ups WHERE user_id = ?"
                    " ORDER BY (status='open') DESC, follow_up_at ASC LIMIT ?",
                    (uid, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def mark_follow_up(self, follow_up_id: str, status: str) -> bool:
        if status not in ("open", "done", "dismissed", "stale"):
            return False
        now = _utcnow_iso()
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "UPDATE follow_ups SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, follow_up_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def touch_follow_up(
        self,
        follow_up_id: str,
        *,
        follow_up_at: str | None = None,
        surfaced: bool = False,
        nudged: bool = False,
    ) -> dict[str, Any] | None:
        """Reschedule a follow-up and/or bump its surfaced/nudged counters."""
        now = _utcnow_iso()
        sets = ["updated_at = ?"]
        params: list[Any] = [now]
        if follow_up_at is not None:
            sets.append("follow_up_at = ?")
            params.append(follow_up_at)
        if surfaced:
            sets.append("surfaced_count = surfaced_count + 1")
            sets.append("last_surfaced_at = ?")
            params.append(now)
        if nudged:
            sets.append("nudged_count = nudged_count + 1")
        params.append(follow_up_id)
        with self._lock:
            conn = self._c()
            conn.execute(
                f"UPDATE follow_ups SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()
        return self.get_follow_up(follow_up_id)

    def cleanup(self, older_than_days: int = 30) -> int:
        """Drop processed (surfaced/acted/ignored) events older than N days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, older_than_days))).isoformat()
        with self._lock:
            conn = self._c()
            cur = conn.execute(
                "DELETE FROM external_events WHERE status != 'new' AND ingested_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount

    @staticmethod
    def _row(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (ValueError, TypeError):
            d["metadata"] = {}
        return d


_STORE: EventsStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> EventsStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = EventsStore()
        return _STORE
