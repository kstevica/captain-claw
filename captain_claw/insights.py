"""Persistent insights memory — auto-extracted facts from conversations.

Stores distilled, structured insights (not raw content) in a dedicated
SQLite database.  Insights are extracted automatically via background LLM
calls and can be searched/managed via the ``insights`` tool.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from captain_claw.config import get_config
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.agent import Agent

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────

VALID_CATEGORIES = frozenset({
    "contact", "decision", "preference", "fact",
    "deadline", "project", "workflow",
})

# BM25 rank threshold for dedup (FTS5 returns negative ranks; closer to 0 = better).
_DEDUP_RANK_THRESHOLD = -8.0

# Extraction trigger defaults (overridden by config).
_DEFAULT_INTERVAL_MESSAGES = 8
_DEFAULT_COOLDOWN_SECONDS = 60


# ── InsightsManager ──────────────────────────────────────────────────

class InsightsManager:
    """Manages the persistent insights SQLite database."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            self.db_path = Path(cfg.insights.db_path).expanduser()
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
            CREATE TABLE IF NOT EXISTS insights (
                id              TEXT PRIMARY KEY,
                content         TEXT NOT NULL,
                category        TEXT NOT NULL,
                entity_key      TEXT,
                importance      INTEGER NOT NULL DEFAULT 5,
                source_tool     TEXT,
                source_session  TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                expires_at      TEXT,
                tags            TEXT
            )
        """)
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_insights_entity_key ON insights(entity_key)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_insights_importance ON insights(importance DESC)")
        await self._db.execute("CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at DESC)")

        # FTS5 for search + dedup.
        await self._db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
                content, category, tags,
                content=insights, content_rowid=rowid
            )
        """)

        # Sync triggers.
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS insights_ai AFTER INSERT ON insights BEGIN
                INSERT INTO insights_fts(rowid, content, category, tags)
                VALUES (new.rowid, new.content, new.category, new.tags);
            END
        """)
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS insights_ad AFTER DELETE ON insights BEGIN
                INSERT INTO insights_fts(insights_fts, rowid, content, category, tags)
                VALUES ('delete', old.rowid, old.content, old.category, old.tags);
            END
        """)
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS insights_au AFTER UPDATE ON insights BEGIN
                INSERT INTO insights_fts(insights_fts, rowid, content, category, tags)
                VALUES ('delete', old.rowid, old.content, old.category, old.tags);
                INSERT INTO insights_fts(rowid, content, category, tags)
                VALUES (new.rowid, new.content, new.category, new.tags);
            END
        """)
        # Migration: Process of Thoughts — add source_message_id and supersedes_id.
        for col in ("source_message_id", "supersedes_id"):
            try:
                await self._db.execute(f"ALTER TABLE insights ADD COLUMN {col} TEXT DEFAULT NULL")
            except Exception:
                pass  # column already exists

        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── CRUD ─────────────────────────────────────────────────────────

    async def add(
        self,
        content: str,
        category: str,
        *,
        entity_key: str | None = None,
        importance: int = 5,
        source_tool: str | None = None,
        source_session: str | None = None,
        tags: str | None = None,
        expires_at: str | None = None,
        source_message_id: str | None = None,
        supersedes_id: str | None = None,
    ) -> str | None:
        """Insert a new insight.  Returns the insight ID, or None if deduped."""
        await self._ensure_db()
        assert self._db is not None

        category = category.lower().strip()
        if category not in VALID_CATEGORIES:
            category = "fact"

        # Dedup: entity_key exact match → supersede existing (preserves lineage).
        if entity_key:
            existing = await self.find_by_entity_key(entity_key)
            if existing:
                supersedes_id = existing["id"]
                await self.delete(existing["id"])
                log.debug("Insight superseded via entity_key", entity_key=entity_key,
                          old_id=supersedes_id)

        # Dedup: FTS similarity check.
        if not supersedes_id:
            similar = await self.find_similar(content, limit=1)
            if similar:
                log.debug("Insight deduped via FTS similarity", score=similar[0].get("rank"))
                return None

        now = datetime.now(UTC).isoformat(timespec="seconds")
        insight_id = uuid.uuid4().hex[:12]

        await self._db.execute(
            """INSERT INTO insights
               (id, content, category, entity_key, importance, source_tool,
                source_session, created_at, updated_at, expires_at, tags,
                source_message_id, supersedes_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (insight_id, content, category, entity_key, importance,
             source_tool, source_session, now, now, expires_at, tags,
             source_message_id, supersedes_id),
        )
        await self._db.commit()
        log.debug("Insight stored", id=insight_id, category=category,
                  supersedes=supersedes_id)
        return insight_id

    async def search(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search insights — tries FTS5 prefix match first, falls back to LIKE."""
        await self._ensure_db()
        assert self._db is not None

        cat_filter = ""
        cat_params: list[Any] = []
        if category:
            cat_filter = "AND i.category = ?"
            cat_params = [category.lower().strip()]

        # Build FTS query: add * to each token for prefix matching.
        tokens = query.strip().split()
        fts_query = " ".join(f'"{t}"*' for t in tokens if t)

        if fts_query:
            sql = f"""
                SELECT i.*, rank
                FROM insights_fts fts
                JOIN insights i ON i.rowid = fts.rowid
                WHERE insights_fts MATCH ?
                {cat_filter}
                ORDER BY rank
                LIMIT ?
            """
            params: list[Any] = [fts_query, *cat_params, limit]
            try:
                rows = await self._db.execute_fetchall(sql, params)
                if rows:
                    return [self._row_to_dict(r) for r in rows]
            except Exception:
                pass  # fall through to LIKE

        # Fallback: LIKE substring search.
        like_pat = f"%{query.strip()}%"
        sql_like = f"""
            SELECT * FROM insights
            WHERE (content LIKE ? OR tags LIKE ?)
            {cat_filter}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params_like: list[Any] = [like_pat, like_pat, *cat_params, limit]
        rows = await self._db.execute_fetchall(sql_like, params_like)
        return [self._row_to_dict(r) for r in rows]

    async def list_recent(
        self,
        limit: int = 20,
        *,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """List most recent insights."""
        await self._ensure_db()
        assert self._db is not None

        if category:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM insights WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category.lower().strip(), limit),
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT * FROM insights ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_dict(r) for r in rows]

    async def get(self, insight_id: str) -> dict[str, Any] | None:
        """Get a single insight by ID."""
        await self._ensure_db()
        assert self._db is not None

        row = await self._db.execute_fetchall(
            "SELECT * FROM insights WHERE id = ?", (insight_id,),
        )
        return self._row_to_dict(row[0]) if row else None

    async def update(self, insight_id: str, **fields: Any) -> bool:
        """Update an existing insight."""
        await self._ensure_db()
        assert self._db is not None

        allowed = {"content", "category", "importance", "tags", "entity_key", "expires_at"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return False

        updates["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [insight_id]

        cursor = await self._db.execute(
            f"UPDATE insights SET {set_clause} WHERE id = ?", values,
        )
        await self._db.commit()
        return (cursor.rowcount or 0) > 0

    async def delete(self, insight_id: str) -> bool:
        """Delete an insight by ID."""
        await self._ensure_db()
        assert self._db is not None

        cursor = await self._db.execute(
            "DELETE FROM insights WHERE id = ?", (insight_id,),
        )
        await self._db.commit()
        return (cursor.rowcount or 0) > 0

    async def clear_all(self) -> int:
        """Delete ALL insights.  Returns the number of rows removed."""
        await self._ensure_db()
        assert self._db is not None

        cursor = await self._db.execute("DELETE FROM insights")
        await self._db.commit()
        return cursor.rowcount or 0

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

        # Sanitize FTS query — remove special chars that break FTS5.
        safe = " ".join(w for w in text.split() if len(w) > 2)[:200]
        if not safe:
            return []

        try:
            rows = await self._db.execute_fetchall(
                """SELECT i.*, rank FROM insights_fts fts
                   JOIN insights i ON i.rowid = fts.rowid
                   WHERE insights_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (safe, limit),
            )
        except Exception:
            return []

        # rank is negative; closer to 0 = better match.
        return [
            self._row_to_dict(r) for r in rows
            if (r[-1] or 0) > _DEDUP_RANK_THRESHOLD  # stronger match than threshold
        ]

    async def find_by_entity_key(self, entity_key: str) -> dict[str, Any] | None:
        """Exact match on entity_key."""
        await self._ensure_db()
        assert self._db is not None

        rows = await self._db.execute_fetchall(
            "SELECT * FROM insights WHERE entity_key = ? LIMIT 1",
            (entity_key,),
        )
        return self._row_to_dict(rows[0]) if rows else None

    # ── context / maintenance ────────────────────────────────────────

    async def get_for_context(
        self,
        query: str | None = None,
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Retrieve insights for context injection.

        If *query* is given, does FTS search; otherwise returns top by
        importance + recency.
        """
        if query:
            return await self.search(query, limit=limit)

        await self._ensure_db()
        assert self._db is not None

        rows = await self._db.execute_fetchall(
            "SELECT * FROM insights ORDER BY importance DESC, created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_dict(r) for r in rows]

    async def prune_expired(self) -> int:
        """Delete insights whose expires_at is in the past."""
        await self._ensure_db()
        assert self._db is not None

        now = datetime.now(UTC).isoformat(timespec="seconds")
        cursor = await self._db.execute(
            "DELETE FROM insights WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        )
        await self._db.commit()
        deleted = cursor.rowcount or 0
        if deleted:
            log.info("Pruned expired insights", count=deleted)
        return deleted

    async def count(self) -> int:
        """Total insight count."""
        await self._ensure_db()
        assert self._db is not None

        rows = await self._db.execute_fetchall("SELECT COUNT(*) FROM insights")
        return rows[0][0] if rows else 0

    # ── internal ─────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Convert an aiosqlite Row to a plain dict."""
        cols = [
            "id", "content", "category", "entity_key", "importance",
            "source_tool", "source_session", "created_at", "updated_at",
            "expires_at", "tags", "source_message_id", "supersedes_id",
        ]
        d: dict[str, Any] = {}
        for i, col in enumerate(cols):
            if i < len(row):
                d[col] = row[i]
        # rank column may be appended by FTS joins.
        if len(row) > len(cols):
            d["rank"] = row[len(cols)]
        return d


# ── Singleton ────────────────────────────────────────────────────────

_manager: InsightsManager | None = None
_session_managers: dict[str, InsightsManager] = {}


def get_insights_manager() -> InsightsManager:
    """Return the global insights manager."""
    global _manager
    if _manager is None:
        _manager = InsightsManager()
    return _manager


def get_session_insights_manager(session_id: str) -> InsightsManager:
    """Return a per-session insights manager (for public computer mode)."""
    if session_id not in _session_managers:
        cfg = get_config()
        base = Path(cfg.insights.db_path).expanduser().parent
        session_db = base / "insights_sessions" / f"insights_{session_id}.db"
        _session_managers[session_id] = InsightsManager(db_path=session_db)
    return _session_managers[session_id]


# ── Extraction ───────────────────────────────────────────────────────

# Track extraction state per agent (set as attributes on the agent object).
_ATTR_LAST_MSG_IDX = "_insights_last_extraction_msg_idx"
_ATTR_LAST_TIME = "_insights_last_extraction_time"
_ATTR_RUNNING = "_insights_extraction_running"


async def extract_insights(
    agent: Agent,
    *,
    trigger: str = "periodic",
    tool_context: str | None = None,
) -> list[dict[str, Any]]:
    """Extract new insights from recent conversation context via LLM."""
    from captain_claw.llm import LLMResponse, Message
    from captain_claw.session import get_session_manager

    cfg = get_config()

    # Use session-specific manager in public mode.
    if cfg.web.public_run and agent.session:
        mgr = get_session_insights_manager(str(agent.session.id))
    else:
        mgr = get_insights_manager()

    # Emit to monitor/activity.
    _emit = getattr(agent, "_emit_thinking", None)
    trigger_label = trigger or "periodic"
    if callable(_emit):
        _emit(f"💡 Insight extraction started (trigger: {trigger_label})", tool="insights", phase="tool")
    log.info("Insight extraction started", trigger=trigger_label)

    # 1. Gather recent session messages.
    recent_text = ""
    if agent.session and agent.session.messages:
        msgs = agent.session.messages[-15:]
        lines: list[str] = []
        for m in msgs:
            role = m.get("role", "unknown")
            content = str(m.get("content", ""))[:500]
            lines.append(f"[{role}] {content}")
        recent_text = "\n".join(lines)

    # 2. Tool context (e.g., email body).
    tool_section = ""
    if tool_context:
        tool_section = f"\nTool output to analyze:\n{tool_context[:3000]}\n"

    # 3. Load recent insights for dedup context.
    existing = await mgr.list_recent(limit=20)
    existing_text = ""
    if existing:
        lines = [f"- [{i['category']}] {i['content']}" for i in existing]
        existing_text = "\n".join(lines)

    # 4. Build LLM messages.
    system_prompt = agent.instructions.load("insight_extraction_system_prompt.md")
    user_prompt = agent.instructions.render(
        "insight_extraction_user_prompt.md",
        recent_messages=recent_text or "(No recent messages.)",
        tool_output_section=tool_section,
        existing_insights=existing_text or "(No existing insights.)",
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]

    # 5. Call LLM.
    max_tokens = min(1000, int(cfg.model.max_tokens))
    t0 = time.monotonic()

    response: LLMResponse = await agent._complete_with_guards(
        messages=messages,
        tools=None,
        interaction_label="insight_extraction",
        max_tokens=max_tokens,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = (response.content or "").strip()
    usage = response.usage or {}

    # 6. Parse JSON response.
    new_insights: list[dict[str, Any]] = []
    try:
        # Handle markdown code fences.
        text = raw
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        parsed = json.loads(text)
        if isinstance(parsed, list):
            new_insights = parsed
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Failed to parse insight extraction response", error=str(exc), raw=raw[:200])

    # 7. Store each insight (dedup happens inside add()).
    stored: list[dict[str, Any]] = []
    session_id = agent.session.id if agent.session else None
    source = trigger if trigger != "periodic" else None

    # Process of Thoughts: find the most recent user message_id for provenance.
    source_message_id: str | None = None
    if agent.session and agent.session.messages:
        for m in reversed(agent.session.messages):
            if m.get("role") == "user" and m.get("message_id"):
                source_message_id = m["message_id"]
                break

    for item in new_insights[:5]:  # Cap at 5.
        if not isinstance(item, dict) or not item.get("content"):
            continue
        insight_id = await mgr.add(
            content=str(item["content"]).strip(),
            category=str(item.get("category", "fact")).strip(),
            entity_key=item.get("entity_key") or None,
            importance=int(item.get("importance", 5)),
            source_tool=source,
            source_session=session_id,
            tags=item.get("tags") or None,
            expires_at=item.get("expires_at") or None,
            source_message_id=source_message_id,
        )
        if insight_id:
            stored.append(item)

    # 8. Log usage.
    try:
        sm = get_session_manager()
        await sm.record_llm_usage(
            session_id=session_id,
            interaction="insight_extraction",
            provider=str(getattr(agent.provider, "provider", "") or getattr(agent.provider, "provider_name", "") or ""),
            model=str(getattr(agent.provider, "model", "") or ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            max_tokens=max_tokens,
            latency_ms=latency_ms,
            task_name="insight_extraction",
            byok=bool(getattr(agent, "_byok_active", False)),
        )
    except Exception as exc:
        log.warning("Failed to record insight extraction LLM usage", error=str(exc))

    # 9. Prune expired insights opportunistically.
    try:
        await mgr.prune_expired()
    except Exception:
        pass

    log.info(
        "Insight extraction completed",
        trigger=trigger_label,
        extracted=len(new_insights),
        stored=len(stored),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )

    # Emit completion to monitor/activity.
    if callable(_emit):
        if stored:
            previews = "; ".join(s.get("content", "")[:60] for s in stored[:3])
            _emit(f"💡 {len(stored)} insight(s) stored: {previews}", tool="insights", phase="tool")
        else:
            _emit("💡 Insight extraction done — nothing new to store", tool="insights", phase="tool")

    # Sister session hook — check if any insights involve approaching deadlines.
    if stored:
        try:
            from captain_claw.sister_session import on_insights_stored
            await on_insights_stored(agent, stored)
        except Exception as exc:
            log.debug("Sister session hook failed (non-fatal)", error=str(exc))

        # Brain Graph live update — broadcast new insight nodes.
        try:
            from captain_claw.web.rest_brain_graph import broadcast_graph_nodes
            broadcast_graph_nodes(agent, stored, node_type="insight")
        except Exception:
            pass  # non-fatal

    # Update tracking state.
    if agent.session:
        setattr(agent, _ATTR_LAST_MSG_IDX, len(agent.session.messages))
    setattr(agent, _ATTR_LAST_TIME, time.time())
    setattr(agent, _ATTR_RUNNING, False)

    return stored


# ── Trigger ──────────────────────────────────────────────────────────

async def maybe_extract_insights(
    agent: Agent,
    *,
    trigger: str = "periodic",
    tool_context: str | None = None,
) -> list[dict[str, Any]] | None:
    """Conditionally trigger insight extraction.  Non-blocking-safe."""
    try:
        cfg = get_config()
        if not cfg.insights.enabled or not cfg.insights.auto_extract:
            return None

        # Guard: only one extraction at a time.
        if getattr(agent, _ATTR_RUNNING, False):
            return None
        setattr(agent, _ATTR_RUNNING, True)

        # Cooldown check.
        last_time = getattr(agent, _ATTR_LAST_TIME, 0.0)
        cooldown = cfg.insights.extraction_cooldown_seconds or _DEFAULT_COOLDOWN_SECONDS
        if time.time() - last_time < cooldown:
            setattr(agent, _ATTR_RUNNING, False)
            return None

        if trigger == "periodic":
            # Message count check.
            if not agent.session:
                setattr(agent, _ATTR_RUNNING, False)
                return None
            last_idx = getattr(agent, _ATTR_LAST_MSG_IDX, 0)
            interval = cfg.insights.extraction_interval_messages or _DEFAULT_INTERVAL_MESSAGES
            if len(agent.session.messages) - last_idx < interval:
                setattr(agent, _ATTR_RUNNING, False)
                return None

        return await extract_insights(agent, trigger=trigger, tool_context=tool_context)

    except Exception as exc:
        log.warning("Insight extraction failed (non-fatal)", error=str(exc))
        setattr(agent, _ATTR_RUNNING, False)
        return None
