"""Cognitive Metrics — tracking system for musical cognition features.

Records events from tension tracking, maturation pipeline, tempo detection,
and processing depth to validate whether these features improve reasoning
quality.  Provides snapshot comparisons for before/after analysis.
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

# Valid event types for each feature.
VALID_EVENT_TYPES = frozenset({
    # Tension tracking
    "tension_created", "tension_resolved",
    # Maturation pipeline
    "maturation_started", "maturation_completed",
    # Tempo detection
    "tempo_detected", "mode_shift",
    # General nervous system
    "dream_cycle", "intuition_surfaced",
})

VALID_FEATURES = frozenset({
    "tension", "maturation", "tempo", "depth", "dream",
})


class CognitiveMetricsManager:
    """Manages the persistent cognitive metrics SQLite database."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            self.db_path = Path(cfg.cognitive_metrics.db_path).expanduser()
        else:
            self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def _ensure_db(self) -> None:
        if self._db is not None:
            return
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA temp_store=MEMORY")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_events (
                id          TEXT PRIMARY KEY,
                event_type  TEXT NOT NULL,
                feature     TEXT NOT NULL,
                session_id  TEXT,
                payload     TEXT,
                created_at  TEXT NOT NULL
            )
        """)
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_ce_type ON cognitive_events(event_type)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_ce_feature ON cognitive_events(feature)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_ce_session ON cognitive_events(session_id)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_ce_created ON cognitive_events(created_at DESC)")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_snapshots (
                id              TEXT PRIMARY KEY,
                session_id      TEXT,
                snapshot_type   TEXT NOT NULL,
                metrics         TEXT NOT NULL,
                created_at      TEXT NOT NULL
            )
        """)
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_cs_type ON cognitive_snapshots(snapshot_type)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_cs_created ON cognitive_snapshots(created_at DESC)")

        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def clear_all(self) -> int:
        """Delete ALL cognitive events and snapshots.  Returns total rows removed."""
        await self._ensure_db()
        assert self._db is not None

        total = 0
        for table in ("cognitive_events", "cognitive_snapshots"):
            cursor = await self._db.execute(f"DELETE FROM {table}")
            total += cursor.rowcount or 0
        await self._db.commit()
        return total

    # ── event recording ──────────────────────────────────────────────

    async def record_event(
        self,
        event_type: str,
        feature: str,
        *,
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Record a cognitive event.  Returns the event ID."""
        cfg = get_config()
        if not cfg.cognitive_metrics.enabled:
            return ""

        await self._ensure_db()
        assert self._db is not None

        event_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC).isoformat(timespec="seconds")

        await self._db.execute(
            """INSERT INTO cognitive_events (id, event_type, feature, session_id, payload, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_id, event_type, feature, session_id,
             json.dumps(payload) if payload else None, now),
        )
        await self._db.commit()

        log.debug("Cognitive event recorded", event_type=event_type, feature=feature, id=event_id)
        return event_id

    # ── querying ─────────────────────────────────────────────────────

    async def query_events(
        self,
        *,
        feature: str | None = None,
        event_type: str | None = None,
        session_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query cognitive events with optional filters."""
        await self._ensure_db()
        assert self._db is not None

        conditions: list[str] = []
        params: list[Any] = []

        if feature:
            conditions.append("feature = ?")
            params.append(feature)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        if until:
            conditions.append("created_at <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = await self._db.execute_fetchall(
            f"SELECT * FROM cognitive_events {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [self._event_to_dict(r) for r in rows]

    async def count_events(
        self,
        *,
        feature: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
    ) -> int:
        """Count events matching filters."""
        await self._ensure_db()
        assert self._db is not None

        conditions: list[str] = []
        params: list[Any] = []

        if feature:
            conditions.append("feature = ?")
            params.append(feature)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._db.execute_fetchall(
            f"SELECT COUNT(*) FROM cognitive_events {where}", params,
        )
        return rows[0][0] if rows else 0

    # ── snapshots ────────────────────────────────────────────────────

    async def take_snapshot(
        self,
        session_id: str | None = None,
        snapshot_type: str = "periodic",
    ) -> str:
        """Aggregate current state into a snapshot.  Returns snapshot ID."""
        await self._ensure_db()
        assert self._db is not None

        now = datetime.now(UTC).isoformat(timespec="seconds")
        snapshot_id = uuid.uuid4().hex[:12]

        # Aggregate metrics across features.
        metrics: dict[str, Any] = {}
        for feature in VALID_FEATURES:
            total = await self.count_events(feature=feature)
            metrics[feature] = {"total_events": total}

        # Tension-specific aggregates.
        tension_created = await self.count_events(event_type="tension_created")
        tension_resolved = await self.count_events(event_type="tension_resolved")
        metrics["tension"]["created"] = tension_created
        metrics["tension"]["resolved"] = tension_resolved
        metrics["tension"]["resolution_rate"] = (
            round(tension_resolved / tension_created, 3) if tension_created > 0 else 0.0
        )

        # Maturation throughput.
        mat_started = await self.count_events(event_type="maturation_started")
        mat_completed = await self.count_events(event_type="maturation_completed")
        metrics["maturation"]["started"] = mat_started
        metrics["maturation"]["completed"] = mat_completed
        metrics["maturation"]["completion_rate"] = (
            round(mat_completed / mat_started, 3) if mat_started > 0 else 0.0
        )

        # Tempo distribution.
        tempo_events = await self.query_events(event_type="tempo_detected", limit=100)
        if tempo_events:
            modes = [e.get("payload", {}).get("mode", "moderato") for e in tempo_events if e.get("payload")]
            mode_dist = {}
            for m in modes:
                mode_dist[m] = mode_dist.get(m, 0) + 1
            metrics["tempo"]["mode_distribution"] = mode_dist

        metrics["snapshot_at"] = now

        await self._db.execute(
            """INSERT INTO cognitive_snapshots (id, session_id, snapshot_type, metrics, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (snapshot_id, session_id, snapshot_type, json.dumps(metrics), now),
        )
        await self._db.commit()

        log.info("Cognitive snapshot taken", id=snapshot_id, type=snapshot_type)
        return snapshot_id

    async def compare_snapshots(
        self,
        before_id: str,
        after_id: str,
    ) -> dict[str, Any]:
        """Compute delta between two snapshots."""
        await self._ensure_db()
        assert self._db is not None

        before_row = await self._db.execute_fetchall(
            "SELECT metrics, created_at FROM cognitive_snapshots WHERE id = ?", (before_id,),
        )
        after_row = await self._db.execute_fetchall(
            "SELECT metrics, created_at FROM cognitive_snapshots WHERE id = ?", (after_id,),
        )

        if not before_row or not after_row:
            return {"error": "Snapshot not found"}

        before_metrics = json.loads(before_row[0][0])
        after_metrics = json.loads(after_row[0][0])

        delta: dict[str, Any] = {
            "before": {"id": before_id, "at": before_row[0][1]},
            "after": {"id": after_id, "at": after_row[0][1]},
            "changes": {},
        }

        for feature in VALID_FEATURES:
            b = before_metrics.get(feature, {})
            a = after_metrics.get(feature, {})
            feature_delta: dict[str, Any] = {}
            all_keys = set(list(b.keys()) + list(a.keys()))
            for key in all_keys:
                bv = b.get(key, 0)
                av = a.get(key, 0)
                if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                    feature_delta[key] = {"before": bv, "after": av, "delta": round(av - bv, 3)}
            if feature_delta:
                delta["changes"][feature] = feature_delta

        return delta

    async def list_snapshots(
        self,
        *,
        snapshot_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List snapshots."""
        await self._ensure_db()
        assert self._db is not None

        if snapshot_type:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM cognitive_snapshots WHERE snapshot_type = ? ORDER BY created_at DESC LIMIT ?",
                (snapshot_type, limit),
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM cognitive_snapshots ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [self._snapshot_to_dict(r) for r in rows]

    # ── summary ──────────────────────────────────────────────────────

    async def summary(self) -> dict[str, Any]:
        """Overall stats for the REST API."""
        await self._ensure_db()
        assert self._db is not None

        total_events = await self.count_events()
        rows = await self._db.execute_fetchall(
            "SELECT COUNT(*) FROM cognitive_snapshots",
        )
        total_snapshots = rows[0][0] if rows else 0

        feature_counts: dict[str, int] = {}
        for feature in VALID_FEATURES:
            feature_counts[feature] = await self.count_events(feature=feature)

        type_counts: dict[str, int] = {}
        rows = await self._db.execute_fetchall(
            "SELECT event_type, COUNT(*) FROM cognitive_events GROUP BY event_type",
        )
        for r in rows:
            type_counts[r[0]] = r[1]

        return {
            "total_events": total_events,
            "total_snapshots": total_snapshots,
            "by_feature": feature_counts,
            "by_event_type": type_counts,
        }

    # ── maintenance ──────────────────────────────────────────────────

    async def prune(self) -> int:
        """Prune old events beyond max_events cap."""
        await self._ensure_db()
        assert self._db is not None

        cfg = get_config()
        max_events = cfg.cognitive_metrics.max_events
        total = await self.count_events()

        if total <= max_events:
            return 0

        excess = total - max_events
        cursor = await self._db.execute(
            """DELETE FROM cognitive_events WHERE id IN (
                SELECT id FROM cognitive_events ORDER BY created_at ASC LIMIT ?
            )""",
            (excess,),
        )
        await self._db.commit()
        deleted = cursor.rowcount or 0
        if deleted:
            log.info("Cognitive metrics pruned", deleted=deleted)
        return deleted

    # ── internal ─────────────────────────────────────────────────────

    @staticmethod
    def _event_to_dict(row: Any) -> dict[str, Any]:
        cols = ["id", "event_type", "feature", "session_id", "payload", "created_at"]
        d: dict[str, Any] = {}
        for i, col in enumerate(cols):
            if i < len(row):
                val = row[i]
                if col == "payload" and isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                d[col] = val
        return d

    @staticmethod
    def _snapshot_to_dict(row: Any) -> dict[str, Any]:
        cols = ["id", "session_id", "snapshot_type", "metrics", "created_at"]
        d: dict[str, Any] = {}
        for i, col in enumerate(cols):
            if i < len(row):
                val = row[i]
                if col == "metrics" and isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                d[col] = val
        return d


# ── Singleton ────────────────────────────────────────────────────────

_manager: CognitiveMetricsManager | None = None


def get_cognitive_metrics_manager() -> CognitiveMetricsManager:
    """Return the global cognitive metrics manager."""
    global _manager
    if _manager is None:
        _manager = CognitiveMetricsManager()
    return _manager
