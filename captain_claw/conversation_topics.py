"""Conversation topic memory — automatic tagging/clustering over comms traffic.

A periodic pass (mirroring the dreaming / insight-extraction passes) takes the
recent comms-channel messages — user inputs, agent replies, and the turn's
narration — and groups them into persistent, cross-session **topics**. Each
topic carries a rolling summary, keywords, and the message excerpts that fed it.

The agent reaches this via the always-on ``topics`` tool (list / get / search),
so when a thread resurfaces ("back to the Munich trip") it pulls the whole
cluster instantly instead of re-deriving it from a long transcript.

Self-contained SQLite store (``conversation_topics.db``), sync sqlite3 + a lock,
matching the other Captain Claw memory stores.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Per-agent attrs for the periodic-pass guard (mirrors nervous_system).
_ATTR_RUNNING = "_topics_classify_running"
_ATTR_LAST_TIME = "_topics_last_classify_time"
_ATTR_LAST_MSG_IDX = "_topics_last_msg_idx"
_ATTR_NARRATION = "_topics_narration_buffer"

_MAX_NARRATION_BUFFER = 40   # cap buffered narration blurbs between passes
# Stored message text is kept (near-)whole so the panel shows full messages, not
# a 600-char stub. Only a short slice is fed to the classifier (token control).
_MAX_EXCERPT_CHARS = 16000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    from captain_claw.config import get_config
    raw = get_config().conversation_topics.db_path or "~/.captain-claw/conversation_topics.db"
    return Path(raw).expanduser()


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (label or "").lower()).strip("-")
    return s[:60] or "topic"


class ConversationTopicsManager:
    """SQLite store for conversation topics + their message excerpts."""

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
            cols = {r[1] for r in self._c().execute("PRAGMA table_info(topic_messages)").fetchall()} \
                if self._c().execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topic_messages'").fetchone() else set()
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    id          TEXT PRIMARY KEY,       -- slug
                    label       TEXT NOT NULL,
                    summary     TEXT NOT NULL DEFAULT '',
                    keywords    TEXT NOT NULL DEFAULT '',  -- comma-separated
                    msg_count   INTEGER NOT NULL DEFAULT 0,
                    first_seen  TEXT NOT NULL,
                    last_seen   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS topic_messages (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id  TEXT NOT NULL,
                    role      TEXT NOT NULL DEFAULT '',   -- user | agent | narration
                    channel   TEXT NOT NULL DEFAULT '',
                    excerpt   TEXT NOT NULL DEFAULT '',
                    msg_id    TEXT NOT NULL DEFAULT '',   -- session message_id (dedup/backfill)
                    ts        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tm_topic ON topic_messages(topic_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_tm_msgid ON topic_messages(msg_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts
                    USING fts5(id UNINDEXED, label, summary, keywords);
                """
            )
            if cols and "msg_id" not in cols:  # migrate a pre-existing table
                self._c().execute("ALTER TABLE topic_messages ADD COLUMN msg_id TEXT NOT NULL DEFAULT ''")
            self._c().commit()

    # ── reads (used by the tool) ────────────────────────────────────────

    def list_topics(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._c().execute(
                "SELECT id, label, summary, keywords, msg_count, first_seen, last_seen"
                " FROM topics ORDER BY last_seen DESC LIMIT ?",
                (max(1, min(200, limit)),),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_topic(self, topic_id: str, *, max_excerpts: int = 40) -> dict[str, Any] | None:
        with self._lock:
            r = self._c().execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
            if not r:
                # tolerate a label being passed instead of a slug
                r = self._c().execute("SELECT * FROM topics WHERE id = ?", (_slug(topic_id),)).fetchone()
            if not r:
                return None
            msgs = self._c().execute(
                "SELECT role, channel, excerpt, ts FROM topic_messages WHERE topic_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (r["id"], max(1, min(200, max_excerpts))),
            ).fetchall()
        d = dict(r)
        d["messages"] = [dict(m) for m in reversed(msgs)]  # oldest→newest
        return d

    def search_topics(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return self.list_topics(limit=limit)
        with self._lock:
            try:
                rows = self._c().execute(
                    "SELECT t.id, t.label, t.summary, t.keywords, t.msg_count, t.last_seen"
                    " FROM topics_fts f JOIN topics t ON t.id = f.id"
                    " WHERE topics_fts MATCH ? ORDER BY rank LIMIT ?",
                    (_fts_query(q), max(1, min(50, limit))),
                ).fetchall()
            except sqlite3.OperationalError:
                like = f"%{q}%"
                rows = self._c().execute(
                    "SELECT id, label, summary, keywords, msg_count, last_seen FROM topics"
                    " WHERE label LIKE ? OR summary LIKE ? OR keywords LIKE ?"
                    " ORDER BY last_seen DESC LIMIT ?",
                    (like, like, like, max(1, min(50, limit))),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── writes (used by the classifier) ─────────────────────────────────

    def upsert_topic(
        self, label: str, *, summary: str = "", keywords: list[str] | None = None,
    ) -> str:
        """Create or update a topic by slug. Merges summary/keywords; bumps last_seen."""
        tid = _slug(label)
        now = _utcnow()
        kw = ",".join(list(dict.fromkeys(keywords or []))[:12]) if keywords else ""
        with self._lock:
            conn = self._c()
            existing = conn.execute("SELECT id, keywords FROM topics WHERE id = ?", (tid,)).fetchone()
            if existing:
                merged_kw = existing["keywords"]
                if kw:
                    have = [k for k in (existing["keywords"] or "").split(",") if k]
                    merged_kw = ",".join(list(dict.fromkeys(have + kw.split(",")))[:12])
                conn.execute(
                    "UPDATE topics SET label = ?, summary = COALESCE(NULLIF(?, ''), summary),"
                    " keywords = ?, last_seen = ? WHERE id = ?",
                    (label[:120], summary[:1000], merged_kw, now, tid),
                )
            else:
                conn.execute(
                    "INSERT INTO topics (id, label, summary, keywords, msg_count, first_seen, last_seen)"
                    " VALUES (?, ?, ?, ?, 0, ?, ?)",
                    (tid, label[:120], summary[:1000], kw, now, now),
                )
            # Mirror into FTS (delete+insert is simplest for a small table).
            conn.execute("DELETE FROM topics_fts WHERE id = ?", (tid,))
            row = conn.execute("SELECT id, label, summary, keywords FROM topics WHERE id = ?", (tid,)).fetchone()
            conn.execute(
                "INSERT INTO topics_fts (id, label, summary, keywords) VALUES (?, ?, ?, ?)",
                (row["id"], row["label"], row["summary"], row["keywords"]),
            )
            conn.commit()
        return tid

    def add_messages(self, topic_id: str, messages: list[dict[str, Any]], *, cap: int = 40) -> None:
        """Append message excerpts to a topic, bump its count, prune to ``cap``."""
        if not messages:
            return
        now = _utcnow()
        with self._lock:
            conn = self._c()
            conn.executemany(
                "INSERT INTO topic_messages (topic_id, role, channel, excerpt, msg_id, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(topic_id, str(m.get("role") or ""), str(m.get("channel") or ""),
                  str(m.get("excerpt") or "")[:_MAX_EXCERPT_CHARS], str(m.get("msg_id") or ""),
                  str(m.get("ts") or now)) for m in messages],
            )
            conn.execute(
                "UPDATE topics SET msg_count = msg_count + ?, last_seen = ? WHERE id = ?",
                (len(messages), now, topic_id),
            )
            # Prune oldest excerpts beyond the cap.
            conn.execute(
                "DELETE FROM topic_messages WHERE topic_id = ? AND id NOT IN ("
                " SELECT id FROM topic_messages WHERE topic_id = ? ORDER BY id DESC LIMIT ?)",
                (topic_id, topic_id, max(1, cap)),
            )
            conn.commit()

    def classified_msg_ids(self) -> set[str]:
        """Session message_ids already assigned to a topic (so backfill skips them)."""
        with self._lock:
            rows = self._c().execute(
                "SELECT DISTINCT msg_id FROM topic_messages WHERE msg_id != ''"
            ).fetchall()
        return {r[0] for r in rows}

    def prune_topics(self, max_topics: int) -> int:
        """Drop the least-recently-seen topics beyond ``max_topics``."""
        with self._lock:
            conn = self._c()
            stale = conn.execute(
                "SELECT id FROM topics ORDER BY last_seen DESC LIMIT -1 OFFSET ?",
                (max(1, max_topics),),
            ).fetchall()
            ids = [r["id"] for r in stale]
            if ids:
                qmarks = ",".join("?" * len(ids))
                conn.execute(f"DELETE FROM topic_messages WHERE topic_id IN ({qmarks})", ids)
                conn.execute(f"DELETE FROM topics WHERE id IN ({qmarks})", ids)
                conn.execute(f"DELETE FROM topics_fts WHERE id IN ({qmarks})", ids)
                conn.commit()
            return len(ids)


def _fts_query(q: str) -> str:
    # OR the bare terms so partial matches work; quote to neutralise FTS syntax.
    terms = [t for t in re.split(r"\s+", q.strip()) if t]
    return " OR ".join(f'"{t}"' for t in terms) or '""'


_MANAGER: ConversationTopicsManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_topics_manager() -> ConversationTopicsManager:
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = ConversationTopicsManager()
        return _MANAGER


# ── narration buffer (fed by agent._emit_narration) ─────────────────────

def record_narration(agent: Any, text: str) -> None:
    """Buffer a narration blurb on the agent for the next classification pass."""
    t = (text or "").strip()
    if not t:
        return
    buf = getattr(agent, _ATTR_NARRATION, None)
    if buf is None:
        buf = []
        setattr(agent, _ATTR_NARRATION, buf)
    buf.append(t[:_MAX_EXCERPT_CHARS])
    if len(buf) > _MAX_NARRATION_BUFFER:
        del buf[: len(buf) - _MAX_NARRATION_BUFFER]


# ── periodic classification pass ────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a conversation topic organiser. You group messages from an "
    "assistant's comms channel (the user, the assistant's replies, and its "
    "progress narration) into a small set of durable TOPICS the assistant can "
    "recall later — e.g. 'Munich trip', 'Vesna VC deal', 'weekly portfolio brief'.\n\n"
    "You are given the EXISTING topics (reuse them whenever a message belongs) and "
    "a batch of NEW messages (numbered). Assign every substantive message to one "
    "topic; create a new topic only when nothing existing fits. Skip pure "
    "pleasantries/acks with no subject.\n\n"
    "Reply with ONLY a JSON array of objects, one per topic that got messages this "
    "batch:\n"
    '{"label": short human topic name (reuse an existing label verbatim when it '
    'fits), "summary": one or two sentences capturing the topic so far (refine the '
    'existing summary if given), "keywords": [up to 6 lowercase tags], '
    '"messages": [the integer indices from the NEW batch that belong here]}\n\n'
    "Keep topics broad enough to be reusable (aim for a handful, not one per "
    "message). Output ONLY the JSON array — start with '[' and end with ']'."
)


def _collect_new_messages(agent: Any, last_idx: int, cap: int) -> tuple[list[dict[str, Any]], int]:
    """Comms messages (user + assistant) since ``last_idx`` + buffered narration.
    Returns (items, new_last_idx)."""
    items: list[dict[str, Any]] = []
    msgs = agent.session.messages if agent.session else []
    new_idx = len(msgs)
    for m in msgs[last_idx:]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        items.append({
            "role": "user" if role == "user" else "agent",
            "channel": str((m.get("metadata") or {}).get("channel") or "") if isinstance(m.get("metadata"), dict) else "",
            "excerpt": content[:_MAX_EXCERPT_CHARS],
            "msg_id": str(m.get("message_id") or ""),
            "ts": str(m.get("timestamp") or _utcnow()),
        })
    # Narration buffered this window (cleared after).
    from captain_claw.config import get_config
    if get_config().conversation_topics.include_narration:
        buf = getattr(agent, _ATTR_NARRATION, None) or []
        for t in buf:
            items.append({"role": "narration", "channel": "", "excerpt": t[:_MAX_EXCERPT_CHARS], "ts": _utcnow()})
        setattr(agent, _ATTR_NARRATION, [])
    if len(items) > cap:
        items = items[-cap:]
    return items, new_idx


async def maybe_classify_topics(agent: Any) -> int | None:
    """Run a topic-classification pass if due (every N comms messages). Mirrors
    maybe_dream's guards. Never raises. Returns the number of topics touched."""
    try:
        from captain_claw.config import get_config
        cfg = get_config()
        tc = cfg.conversation_topics
        if not tc.enabled:
            return None
        if cfg.web.public_run and not tc.allow_public:
            return None
        if getattr(agent, _ATTR_RUNNING, False):
            return None
        if not agent.session or not agent.session.messages:
            return None
        last_time = getattr(agent, _ATTR_LAST_TIME, 0.0)
        if time.time() - last_time < (tc.cooldown_seconds or 120):
            return None
        last_idx = getattr(agent, _ATTR_LAST_MSG_IDX, 0)
        if len(agent.session.messages) - last_idx < (tc.interval_messages or 15):
            return None
        setattr(agent, _ATTR_RUNNING, True)
        try:
            return await classify_topics(agent)
        finally:
            setattr(agent, _ATTR_RUNNING, False)
            setattr(agent, _ATTR_LAST_TIME, time.time())
    except Exception as exc:
        log.warning("Topic classification failed (non-fatal)", exc_info=False)
        log.debug("topic classify error: %s", exc)
        setattr(agent, _ATTR_RUNNING, False)
        return None


async def classify_topics(agent: Any) -> int | None:
    from captain_claw.config import get_config

    tc = get_config().conversation_topics
    last_idx = getattr(agent, _ATTR_LAST_MSG_IDX, 0)
    items, new_idx = _collect_new_messages(agent, last_idx, tc.max_messages_per_pass)
    if not items:
        setattr(agent, _ATTR_LAST_MSG_IDX, new_idx)
        return 0
    touched = await _classify_and_store(agent, items)
    setattr(agent, _ATTR_LAST_MSG_IDX, new_idx)
    log.info("conversation topics: %d topic(s) touched from %d message(s)", touched, len(items))
    return touched


async def _classify_and_store(agent: Any, items: list[dict[str, Any]]) -> int:
    """One LLM classification call over ``items`` → upsert topics + store excerpts.
    Shared by the live pass and the backfill. Returns topics touched."""
    from captain_claw.config import get_config
    from captain_claw.llm import Message

    tc = get_config().conversation_topics
    mgr = get_topics_manager()
    # Show the classifier ALL current topics (most-recent first) so it reuses an
    # existing one instead of minting a near-duplicate. The prompt instructs it to
    # copy a matching label verbatim; upsert_topic then dedups by slug.
    existing = mgr.list_topics(limit=200)
    existing_block = "\n".join(f"- {t['label']}: {t['summary'][:160]}" for t in existing) or "(none yet)"
    batch_block = "\n".join(f"[{i}] ({it['role']}) {it['excerpt'][:300]}" for i, it in enumerate(items))
    user_prompt = f"EXISTING topics:\n{existing_block}\n\nNEW messages:\n{batch_block}"

    response = await agent._complete_with_guards(
        messages=[
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ],
        tools=None,
        interaction_label="conversation_topics",
        max_tokens=min(1200, int(get_config().model.max_tokens)),
    )
    groups = _parse_groups((response.content or "").strip())
    touched = 0
    for g in groups:
        label = str(g.get("label") or "").strip()
        if not label:
            continue
        idxs = [i for i in g.get("messages", []) if isinstance(i, int) and 0 <= i < len(items)]
        if not idxs:
            continue
        tid = mgr.upsert_topic(
            label, summary=str(g.get("summary") or ""),
            keywords=[str(k) for k in (g.get("keywords") or []) if str(k).strip()],
        )
        mgr.add_messages(tid, [items[i] for i in idxs], cap=tc.excerpts_per_topic)
        touched += 1
    mgr.prune_topics(tc.max_topics)
    return touched


async def backfill_topics(agent: Any, hours: int = 0) -> dict[str, Any]:
    """Classify past comms messages that don't yet belong to a topic. ``hours``
    limits the window (0 = all history). Skips already-classified messages and
    runs in batches of ``max_messages_per_pass``. Returns a summary dict."""
    from captain_claw.config import get_config
    tc = get_config().conversation_topics
    if not agent.session or not agent.session.messages:
        return {"ok": True, "classified": 0, "topics_touched": 0, "remaining": 0}

    cutoff = None
    if hours and hours > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    mgr = get_topics_manager()
    done_ids = mgr.classified_msg_ids()

    pending: list[dict[str, Any]] = []
    for m in agent.session.messages:
        if m.get("role") not in ("user", "assistant"):
            continue
        content = str(m.get("content") or "").strip()
        mid = str(m.get("message_id") or "")
        if not content or (mid and mid in done_ids):
            continue
        ts_raw = str(m.get("timestamp") or "")
        if cutoff is not None:
            try:
                if datetime.fromisoformat(ts_raw) < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        pending.append({
            "role": "user" if m.get("role") == "user" else "agent",
            "channel": "", "excerpt": content[:_MAX_EXCERPT_CHARS], "msg_id": mid, "ts": ts_raw or _utcnow(),
        })

    if not pending:
        return {"ok": True, "classified": 0, "topics_touched": 0, "remaining": 0}

    batch = max(5, int(tc.max_messages_per_pass))
    classified = touched = 0
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        try:
            touched += await _classify_and_store(agent, chunk)
            classified += len(chunk)
        except Exception as exc:
            log.warning("topic backfill chunk failed: %s", exc)
            break
    log.info("topic backfill: classified %d message(s) into %d topic touch(es)", classified, touched)
    return {"ok": True, "classified": classified, "topics_touched": touched,
            "remaining": max(0, len(pending) - classified)}


def _parse_groups(text: str) -> list[dict[str, Any]]:
    txt = (text or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
        txt = re.sub(r"\n?```$", "", txt).strip()
    data: Any = None
    try:
        data = json.loads(txt)
    except (ValueError, TypeError):
        m = re.search(r"\[.*\]", txt, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except (ValueError, TypeError):
                data = None
    if isinstance(data, dict):
        data = [data]
    return [g for g in data if isinstance(g, dict)] if isinstance(data, list) else []
