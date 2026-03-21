"""Nervous System — autonomous pattern-recognition and memory synthesis.

A background cognitive process that "dreams" — finding connections, patterns,
and hypotheses across all memory layers (working, semantic, deep, insights,
reflections) and surfacing them as intuitions to guide agent behaviour.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from captain_claw.config import get_config
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.agent import Agent

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────

VALID_THREAD_TYPES = frozenset({
    "connection", "pattern", "hypothesis", "association",
})

# BM25 rank threshold for dedup (FTS5 returns negative ranks; closer to 0 = better).
_DEDUP_RANK_THRESHOLD = -8.0

# Dreaming trigger defaults (overridden by config).
_DEFAULT_INTERVAL_MESSAGES = 12
_DEFAULT_COOLDOWN_SECONDS = 300


# ── NervousSystemManager ────────────────────────────────────────────

class NervousSystemManager:
    """Manages the persistent intuitions SQLite database."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            self.db_path = Path(cfg.nervous_system.db_path).expanduser()
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
            CREATE TABLE IF NOT EXISTS intuitions (
                id              TEXT PRIMARY KEY,
                content         TEXT NOT NULL,
                thread_type     TEXT NOT NULL,
                source_layers   TEXT NOT NULL,
                source_ids      TEXT,
                source_session  TEXT,
                confidence      REAL NOT NULL DEFAULT 0.5,
                importance      INTEGER NOT NULL DEFAULT 5,
                access_count    INTEGER NOT NULL DEFAULT 0,
                last_accessed   TEXT,
                validated       INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                decayed_at      TEXT,
                tags            TEXT
            )
        """)
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_intuitions_thread_type ON intuitions(thread_type)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_intuitions_confidence ON intuitions(confidence DESC)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_intuitions_importance ON intuitions(importance DESC)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_intuitions_created ON intuitions(created_at DESC)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_intuitions_access ON intuitions(access_count DESC)")

        # FTS5 for search + dedup.
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS intuitions_fts USING fts5(
                content, thread_type, tags,
                content=intuitions, content_rowid=rowid
            )
        """)

        # Sync triggers.
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS intuitions_ai AFTER INSERT ON intuitions BEGIN
                INSERT INTO intuitions_fts(rowid, content, thread_type, tags)
                VALUES (new.rowid, new.content, new.thread_type, new.tags);
            END
        """)
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS intuitions_ad AFTER DELETE ON intuitions BEGIN
                INSERT INTO intuitions_fts(intuitions_fts, rowid, content, thread_type, tags)
                VALUES ('delete', old.rowid, old.content, old.thread_type, old.tags);
            END
        """)
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS intuitions_au AFTER UPDATE ON intuitions BEGIN
                INSERT INTO intuitions_fts(intuitions_fts, rowid, content, thread_type, tags)
                VALUES ('delete', old.rowid, old.content, old.thread_type, old.tags);
                INSERT INTO intuitions_fts(rowid, content, thread_type, tags)
                VALUES (new.rowid, new.content, new.thread_type, new.tags);
            END
        """)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── CRUD ─────────────────────────────────────────────────────────

    async def add(
        self,
        content: str,
        thread_type: str,
        *,
        source_layers: list[str] | None = None,
        source_ids: list[str] | None = None,
        source_session: str | None = None,
        confidence: float = 0.5,
        importance: int = 5,
        tags: str | None = None,
    ) -> str | None:
        """Insert a new intuition.  Returns the ID, or None if deduped."""
        await self._ensure_db()
        assert self._db is not None

        thread_type = thread_type.lower().strip()
        if thread_type not in VALID_THREAD_TYPES:
            thread_type = "association"

        confidence = max(0.0, min(1.0, confidence))
        importance = max(1, min(10, importance))

        # Dedup: FTS similarity check.
        similar = await self.find_similar(content, limit=1)
        if similar:
            log.debug("Intuition deduped via FTS similarity", score=similar[0].get("rank"))
            return None

        now = datetime.now(UTC).isoformat(timespec="seconds")
        intuition_id = uuid.uuid4().hex[:12]

        await self._db.execute(
            """INSERT INTO intuitions
               (id, content, thread_type, source_layers, source_ids,
                source_session, confidence, importance, access_count,
                last_accessed, validated, created_at, updated_at, decayed_at, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, ?, ?, NULL, ?)""",
            (intuition_id, content, thread_type,
             json.dumps(source_layers or []),
             json.dumps(source_ids or []),
             source_session, confidence, importance, now, now, tags),
        )
        await self._db.commit()
        log.debug("Intuition stored", id=intuition_id, thread_type=thread_type)
        return intuition_id

    async def search(
        self,
        query: str,
        *,
        thread_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search intuitions — FTS5 prefix match first, falls back to LIKE."""
        await self._ensure_db()
        assert self._db is not None

        type_filter = ""
        type_params: list[Any] = []
        if thread_type:
            type_filter = "AND i.thread_type = ?"
            type_params = [thread_type.lower().strip()]

        tokens = query.strip().split()
        fts_query = " ".join(f'"{t}"*' for t in tokens if t)

        if fts_query:
            sql = f"""
                SELECT i.*, rank
                FROM intuitions_fts fts
                JOIN intuitions i ON i.rowid = fts.rowid
                WHERE intuitions_fts MATCH ?
                {type_filter}
                ORDER BY rank
                LIMIT ?
            """
            params: list[Any] = [fts_query, *type_params, limit]
            try:
                rows = await self._db.execute_fetchall(sql, params)
                if rows:
                    return [self._row_to_dict(r) for r in rows]
            except Exception:
                pass  # fall through to LIKE

        like_pat = f"%{query.strip()}%"
        sql_like = f"""
            SELECT * FROM intuitions
            WHERE (content LIKE ? OR tags LIKE ?)
            {type_filter}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params_like: list[Any] = [like_pat, like_pat, *type_params, limit]
        rows = await self._db.execute_fetchall(sql_like, params_like)
        return [self._row_to_dict(r) for r in rows]

    async def list_recent(
        self,
        limit: int = 20,
        *,
        thread_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """List most recent intuitions."""
        await self._ensure_db()
        assert self._db is not None

        if thread_type:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM intuitions WHERE thread_type = ? AND confidence >= ? ORDER BY created_at DESC LIMIT ?",
                (thread_type.lower().strip(), min_confidence, limit),
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM intuitions WHERE confidence >= ? ORDER BY created_at DESC LIMIT ?",
                (min_confidence, limit),
            )
        return [self._row_to_dict(r) for r in rows]

    async def get(self, intuition_id: str) -> dict[str, Any] | None:
        """Get a single intuition by ID."""
        await self._ensure_db()
        assert self._db is not None

        row = await self._db.execute_fetchall(
            "SELECT * FROM intuitions WHERE id = ?", (intuition_id,),
        )
        return self._row_to_dict(row[0]) if row else None

    async def update(self, intuition_id: str, **fields: Any) -> bool:
        """Update an existing intuition."""
        await self._ensure_db()
        assert self._db is not None

        allowed = {"content", "thread_type", "confidence", "importance", "tags", "validated"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return False

        updates["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [intuition_id]

        cursor = await self._db.execute(
            f"UPDATE intuitions SET {set_clause} WHERE id = ?", values,
        )
        await self._db.commit()
        return (cursor.rowcount or 0) > 0

    async def delete(self, intuition_id: str) -> bool:
        """Delete an intuition by ID."""
        await self._ensure_db()
        assert self._db is not None

        cursor = await self._db.execute(
            "DELETE FROM intuitions WHERE id = ?", (intuition_id,),
        )
        await self._db.commit()
        return (cursor.rowcount or 0) > 0

    # ── dedup helpers ────────────────────────────────────────────────

    async def find_similar(
        self,
        text: str,
        *,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """FTS5 similarity search for dedup."""
        await self._ensure_db()
        assert self._db is not None

        safe = " ".join(w for w in text.split() if len(w) > 2)[:200]
        if not safe:
            return []

        try:
            rows = await self._db.execute_fetchall(
                """SELECT i.*, rank FROM intuitions_fts fts
                   JOIN intuitions i ON i.rowid = fts.rowid
                   WHERE intuitions_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (safe, limit),
            )
        except Exception:
            return []

        return [
            self._row_to_dict(r) for r in rows
            if (r[-1] or 0) > _DEDUP_RANK_THRESHOLD
        ]

    # ── context / lifecycle ──────────────────────────────────────────

    async def get_for_context(
        self,
        query: str | None = None,
        *,
        limit: int = 4,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve intuitions for context injection.

        Ranks by (confidence * importance) with session and recency bonuses.
        """
        if query:
            return await self.search(query, limit=limit)

        await self._ensure_db()
        assert self._db is not None

        cfg = get_config()
        min_conf = cfg.nervous_system.min_confidence_for_context

        rows = await self._db.execute_fetchall(
            "SELECT * FROM intuitions WHERE confidence >= ? ORDER BY importance DESC, confidence DESC, created_at DESC LIMIT ?",
            (min_conf, limit * 3),  # Fetch extra for scoring.
        )

        items = [self._row_to_dict(r) for r in rows]

        # Apply session and recency bonuses for ranking.
        now = datetime.now(UTC)
        for item in items:
            score = (item.get("confidence", 0.5) * item.get("importance", 5))
            # Session bonus.
            if session_id and item.get("source_session") == session_id:
                score += 2.0
            # Recency bonus (accessed in last 24h).
            la = item.get("last_accessed")
            if la:
                try:
                    last_acc = datetime.fromisoformat(la)
                    if (now - last_acc).total_seconds() < 86400:
                        score += 1.0
                except (ValueError, TypeError):
                    pass
            item["_score"] = score

        items.sort(key=lambda x: x.get("_score", 0), reverse=True)

        # Clean up internal score and return top N.
        result = items[:limit]
        for item in result:
            item.pop("_score", None)
        return result

    async def record_access(self, intuition_id: str) -> None:
        """Increment access_count and update last_accessed."""
        await self._ensure_db()
        assert self._db is not None

        now = datetime.now(UTC).isoformat(timespec="seconds")
        await self._db.execute(
            "UPDATE intuitions SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, intuition_id),
        )
        await self._db.commit()

    async def validate(self, intuition_id: str) -> None:
        """Mark an intuition as validated — boosts confidence, protects from decay."""
        await self._ensure_db()
        assert self._db is not None

        now = datetime.now(UTC).isoformat(timespec="seconds")
        await self._db.execute(
            """UPDATE intuitions SET
               validated = 1,
               confidence = MIN(1.0, confidence + 0.2),
               importance = MIN(10, importance + 1),
               updated_at = ?
               WHERE id = ?""",
            (now, intuition_id),
        )
        await self._db.commit()

    # ── decay / consolidation ────────────────────────────────────────

    async def run_decay(self) -> int:
        """Decay unused intuitions and delete those below threshold."""
        await self._ensure_db()
        assert self._db is not None

        cfg = get_config()
        now = datetime.now(UTC)
        decay_days = cfg.nervous_system.decay_after_days
        decay_rate = cfg.nervous_system.decay_rate_per_day
        delete_thresh = cfg.nervous_system.delete_threshold

        # 1. Delete very low confidence unvalidated intuitions.
        cursor = await self._db.execute(
            "DELETE FROM intuitions WHERE confidence < ? AND validated = 0",
            (delete_thresh,),
        )
        deleted = cursor.rowcount or 0

        # 2. Decay unvalidated intuitions not accessed recently.
        cutoff = (now - timedelta(days=decay_days)).isoformat(timespec="seconds")
        rows = await self._db.execute_fetchall(
            """SELECT id, confidence, last_accessed, created_at FROM intuitions
               WHERE validated = 0 AND confidence >= ?
               AND (last_accessed IS NULL OR last_accessed < ?)
               AND created_at < ?""",
            (delete_thresh, cutoff, cutoff),
        )

        decayed = 0
        for row in rows:
            intuition_id, conf, last_acc, created = row[0], row[1], row[2], row[3]
            ref_date = last_acc or created
            try:
                ref_dt = datetime.fromisoformat(ref_date)
                days_inactive = (now - ref_dt).days - decay_days
                if days_inactive > 0:
                    new_conf = max(0.0, conf - (decay_rate * days_inactive))
                    await self._db.execute(
                        "UPDATE intuitions SET confidence = ?, decayed_at = ? WHERE id = ?",
                        (new_conf, now.isoformat(timespec="seconds"), intuition_id),
                    )
                    decayed += 1
            except (ValueError, TypeError):
                continue

        # 3. Enforce max_intuitions cap.
        max_count = cfg.nervous_system.max_intuitions
        total = await self.count()
        if total > max_count:
            excess = total - max_count
            await self._db.execute(
                """DELETE FROM intuitions WHERE id IN (
                    SELECT id FROM intuitions
                    WHERE validated = 0
                    ORDER BY confidence ASC, access_count ASC
                    LIMIT ?
                )""",
                (excess,),
            )

        await self._db.commit()

        if deleted or decayed:
            log.info("Nervous system decay pass", deleted=deleted, decayed=decayed)
        return deleted + decayed

    async def count(self) -> int:
        """Total intuition count."""
        await self._ensure_db()
        assert self._db is not None

        rows = await self._db.execute_fetchall("SELECT COUNT(*) FROM intuitions")
        return rows[0][0] if rows else 0

    async def stats(self) -> dict[str, Any]:
        """Aggregate stats for the REST API."""
        await self._ensure_db()
        assert self._db is not None

        total = await self.count()

        rows = await self._db.execute_fetchall(
            "SELECT AVG(confidence), AVG(importance) FROM intuitions",
        )
        avg_conf = round(rows[0][0] or 0, 3) if rows else 0
        avg_imp = round(rows[0][1] or 0, 1) if rows else 0

        type_rows = await self._db.execute_fetchall(
            "SELECT thread_type, COUNT(*) FROM intuitions GROUP BY thread_type",
        )
        type_dist = {r[0]: r[1] for r in type_rows}

        validated_rows = await self._db.execute_fetchall(
            "SELECT COUNT(*) FROM intuitions WHERE validated = 1",
        )
        validated = validated_rows[0][0] if validated_rows else 0

        return {
            "total": total,
            "avg_confidence": avg_conf,
            "avg_importance": avg_imp,
            "type_distribution": type_dist,
            "validated": validated,
        }

    # ── internal ─────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Convert an aiosqlite Row to a plain dict."""
        cols = [
            "id", "content", "thread_type", "source_layers", "source_ids",
            "source_session", "confidence", "importance", "access_count",
            "last_accessed", "validated", "created_at", "updated_at",
            "decayed_at", "tags",
        ]
        d: dict[str, Any] = {}
        for i, col in enumerate(cols):
            if i < len(row):
                val = row[i]
                # Parse JSON lists.
                if col in ("source_layers", "source_ids") and isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                d[col] = val
        # rank column may be appended by FTS joins.
        if len(row) > len(cols):
            d["rank"] = row[len(cols)]
        return d


# ── Singleton ────────────────────────────────────────────────────────

_manager: NervousSystemManager | None = None
_session_managers: dict[str, NervousSystemManager] = {}


def get_nervous_system_manager() -> NervousSystemManager:
    """Return the global nervous system manager."""
    global _manager
    if _manager is None:
        _manager = NervousSystemManager()
    return _manager


def get_session_nervous_system_manager(session_id: str) -> NervousSystemManager:
    """Return a per-session manager (for public computer mode)."""
    if session_id not in _session_managers:
        cfg = get_config()
        base = Path(cfg.nervous_system.db_path).expanduser().parent
        session_db = base / "intuitions_sessions" / f"intuitions_{session_id}.db"
        _session_managers[session_id] = NervousSystemManager(db_path=session_db)
    return _session_managers[session_id]


# ── Dreaming ─────────────────────────────────────────────────────────

# Track dreaming state per agent (set as attributes on the agent object).
_ATTR_LAST_MSG_IDX = "_nervous_last_dream_msg_idx"
_ATTR_LAST_TIME = "_nervous_last_dream_time"
_ATTR_RUNNING = "_nervous_dream_running"


async def dream(agent: Agent) -> list[dict[str, Any]]:
    """Core dreaming function — synthesize connections across all memory layers."""
    from captain_claw.llm import LLMResponse, Message
    from captain_claw.reflections import load_latest_reflection
    from captain_claw.session import get_session_manager

    cfg = get_config()

    # Use session-specific manager in public mode.
    if cfg.web.public_run and agent.session:
        mgr = get_session_nervous_system_manager(str(agent.session.id))
    else:
        mgr = get_nervous_system_manager()

    _emit = getattr(agent, "_emit_thinking", None)
    if callable(_emit):
        _emit("🧠 Nervous system dreaming started", tool="nervous_system", phase="tool")
    log.info("Nervous system dreaming started")

    # 1. Sample working memory — last 10 messages.
    recent_text = ""
    if agent.session and agent.session.messages:
        msgs = agent.session.messages[-10:]
        lines: list[str] = []
        for m in msgs:
            role = m.get("role", "unknown")
            content = str(m.get("content", ""))[:400]
            lines.append(f"[{role}] {content}")
        recent_text = "\n".join(lines)

    # 2. Sample insights — top 10 by importance.
    insights_text = ""
    try:
        from captain_claw.insights import get_insights_manager
        ins_mgr = get_insights_manager()
        insights = await ins_mgr.get_for_context(limit=10)
        if insights:
            lines = [f"- [{i['category']}] (imp:{i.get('importance', 5)}) {i['content']}" for i in insights]
            insights_text = "\n".join(lines)
    except Exception:
        pass

    # 3. Sample latest reflection.
    reflection_text = ""
    try:
        refl = load_latest_reflection()
        if refl and refl.summary:
            reflection_text = refl.summary.strip()[:600]
    except Exception:
        pass

    # 4. Sample semantic memory — keyword queries from recent messages.
    semantic_text = ""
    try:
        memory = getattr(agent, "memory", None)
        if memory and getattr(memory, "semantic", None):
            # Extract a few key phrases from recent messages for queries.
            query_text = ""
            if agent.session and agent.session.messages:
                last_msgs = agent.session.messages[-5:]
                query_text = " ".join(
                    str(m.get("content", ""))[:100]
                    for m in last_msgs if m.get("role") == "user"
                )[:300]

            if query_text:
                note, _ = memory.semantic.build_context_note(
                    query=query_text, max_items=3, max_snippet_chars=300, layer="l2",
                )
                semantic_text = note
    except Exception:
        pass

    # 5. Sample deep memory.
    deep_text = ""
    try:
        deep = getattr(agent, "_deep_memory", None)
        if deep and query_text:
            note, _ = deep.build_context_note(
                query=query_text, max_items=2, max_snippet_chars=200, layer="l2",
            )
            deep_text = note
    except Exception:
        pass

    # 6. Load existing intuitions for dedup context.
    existing = await mgr.list_recent(limit=5)
    existing_text = ""
    if existing:
        lines = [f"- [{i['thread_type']}] {i['content']}" for i in existing]
        existing_text = "\n".join(lines)

    # 7. Build LLM messages.
    system_prompt = agent.instructions.load("dreaming_system_prompt.md")
    user_prompt = agent.instructions.render(
        "dreaming_user_prompt.md",
        recent_messages=recent_text or "(No recent messages.)",
        insights_text=insights_text or "(No stored insights.)",
        reflection_text=reflection_text or "(No reflection available.)",
        semantic_text=semantic_text or "(No semantic memory matches.)",
        deep_text=deep_text or "(No deep memory matches.)",
        existing_intuitions=existing_text or "(No existing intuitions.)",
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]

    # 8. Call LLM.
    max_tokens = min(800, int(cfg.model.max_tokens))
    t0 = time.monotonic()

    response: LLMResponse = await agent._complete_with_guards(
        messages=messages,
        tools=None,
        interaction_label="nervous_system_dream",
        max_tokens=max_tokens,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = (response.content or "").strip()
    usage = response.usage or {}

    # 9. Parse JSON response.
    new_intuitions: list[dict[str, Any]] = []
    try:
        text = raw
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        parsed = json.loads(text)
        if isinstance(parsed, list):
            new_intuitions = parsed
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Failed to parse dreaming response", error=str(exc), raw=raw[:200])

    # 10. Store each intuition (dedup happens inside add()).
    stored: list[dict[str, Any]] = []
    session_id = agent.session.id if agent.session else None

    for item in new_intuitions[:3]:  # Cap at 3 per dream.
        if not isinstance(item, dict) or not item.get("content"):
            continue
        intuition_id = await mgr.add(
            content=str(item["content"]).strip(),
            thread_type=str(item.get("thread_type", "association")).strip(),
            source_layers=item.get("source_layers") or [],
            source_ids=item.get("source_ids") or [],
            source_session=session_id,
            confidence=float(item.get("confidence", 0.5)),
            importance=int(item.get("importance", 5)),
            tags=item.get("tags") or None,
        )
        if intuition_id:
            stored.append(item)

    # 11. Log usage.
    try:
        sm = get_session_manager()
        await sm.record_llm_usage(
            session_id=session_id,
            interaction="nervous_system_dream",
            provider=str(getattr(agent.provider, "provider", "") or getattr(agent.provider, "provider_name", "") or ""),
            model=str(getattr(agent.provider, "model", "") or ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            task_name="nervous_system_dream",
            byok=bool(getattr(agent, "_byok_active", False)),
        )
    except Exception as exc:
        log.warning("Failed to record dreaming LLM usage", error=str(exc))

    # 12. Run decay opportunistically.
    try:
        await mgr.run_decay()
    except Exception:
        pass

    log.info(
        "Nervous system dreaming completed",
        extracted=len(new_intuitions),
        stored=len(stored),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )

    if callable(_emit):
        if stored:
            previews = "; ".join(s.get("content", "")[:60] for s in stored[:3])
            _emit(f"🧠 {len(stored)} intuition(s) formed: {previews}", tool="nervous_system", phase="tool")
        else:
            _emit("🧠 Dreaming done — no new intuitions", tool="nervous_system", phase="tool")

    # Update tracking state.
    if agent.session:
        setattr(agent, _ATTR_LAST_MSG_IDX, len(agent.session.messages))
    setattr(agent, _ATTR_LAST_TIME, time.time())
    setattr(agent, _ATTR_RUNNING, False)

    return stored


# ── Trigger ──────────────────────────────────────────────────────────

async def maybe_dream(
    agent: Agent,
) -> list[dict[str, Any]] | None:
    """Conditionally trigger a dream cycle.  Non-blocking-safe."""
    try:
        cfg = get_config()
        if not cfg.nervous_system.enabled or not cfg.nervous_system.auto_dream:
            return None

        # Public mode guard.
        is_public = getattr(agent, "_is_public", False)
        if is_public and not cfg.nervous_system.allow_public:
            return None

        # Guard: only one dream at a time.
        if getattr(agent, _ATTR_RUNNING, False):
            return None
        setattr(agent, _ATTR_RUNNING, True)

        # Cooldown check.
        last_time = getattr(agent, _ATTR_LAST_TIME, 0.0)
        cooldown = cfg.nervous_system.dream_cooldown_seconds or _DEFAULT_COOLDOWN_SECONDS
        if time.time() - last_time < cooldown:
            setattr(agent, _ATTR_RUNNING, False)
            return None

        # Message count check.
        if not agent.session:
            setattr(agent, _ATTR_RUNNING, False)
            return None
        last_idx = getattr(agent, _ATTR_LAST_MSG_IDX, 0)
        interval = cfg.nervous_system.dream_interval_messages or _DEFAULT_INTERVAL_MESSAGES
        if len(agent.session.messages) - last_idx < interval:
            setattr(agent, _ATTR_RUNNING, False)
            return None

        return await dream(agent)

    except Exception as exc:
        log.warning("Nervous system dreaming failed (non-fatal)", error=str(exc))
        setattr(agent, _ATTR_RUNNING, False)
        return None
