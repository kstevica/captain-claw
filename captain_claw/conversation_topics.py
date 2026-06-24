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
            def _cols(tbl: str) -> set[str]:
                if not self._c().execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)).fetchone():
                    return set()
                return {r[1] for r in self._c().execute(f"PRAGMA table_info({tbl})").fetchall()}
            tm_cols = _cols("topic_messages")
            t_cols = _cols("topics")
            self._c().executescript(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    id          TEXT PRIMARY KEY,       -- slug
                    label       TEXT NOT NULL,
                    summary     TEXT NOT NULL DEFAULT '',
                    keywords    TEXT NOT NULL DEFAULT '',  -- comma-separated
                    msg_count   INTEGER NOT NULL DEFAULT 0,
                    starred     INTEGER NOT NULL DEFAULT 0, -- pinned to the top
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
                -- Backfill progress: every message the backfill has ATTEMPTED, so it
                -- moves on even when the classifier puts a message in no topic
                -- (otherwise those loop forever).
                CREATE TABLE IF NOT EXISTS backfill_seen (msg_id TEXT PRIMARY KEY);

                -- User-defined groups (private, work, …) — many-to-many with topics.
                CREATE TABLE IF NOT EXISTS topic_groups (
                    id          TEXT PRIMARY KEY,   -- slug
                    name        TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS topic_group_members (
                    group_id  TEXT NOT NULL,
                    topic_id  TEXT NOT NULL,
                    PRIMARY KEY (group_id, topic_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tgm_topic ON topic_group_members(topic_id);
                """
            )
            if tm_cols and "msg_id" not in tm_cols:  # migrate pre-existing tables
                self._c().execute("ALTER TABLE topic_messages ADD COLUMN msg_id TEXT NOT NULL DEFAULT ''")
            if t_cols and "starred" not in t_cols:
                self._c().execute("ALTER TABLE topics ADD COLUMN starred INTEGER NOT NULL DEFAULT 0")
            self._c().commit()

    # ── reads (used by the tool) ────────────────────────────────────────

    def list_topics(self, limit: int = 40, order: str = "recent", group: str = "",
                    tags: list[str] | None = None) -> list[dict[str, Any]]:
        # Starred topics always float to the top; within each group, by recency
        # (newest message) or alphabetically. ``group`` and ``tags`` AND together
        # — a topic must be in the group AND carry EVERY active tag (substring
        # match against its keywords column).
        secondary = "label COLLATE NOCASE ASC" if order == "alpha" else "last_seen DESC"
        cols = "t.id, t.label, t.summary, t.keywords, t.msg_count, t.starred, t.first_seen, t.last_seen"
        tag_terms = [t.strip().lower() for t in (tags or []) if str(t).strip()]
        tag_where = " AND ".join(["LOWER(t.keywords) LIKE ?"] * len(tag_terms))
        tag_params = [f"%{t}%" for t in tag_terms]
        with self._lock:
            if group:
                sql = (
                    f"SELECT {cols} FROM topics t"
                    " JOIN topic_group_members m ON m.topic_id = t.id"
                    " WHERE m.group_id = ?"
                    + (f" AND {tag_where}" if tag_where else "")
                    + f" ORDER BY t.starred DESC, {secondary} LIMIT ?"
                )
                rows = self._c().execute(
                    sql, (_slug(group), *tag_params, max(1, min(300, limit))),
                ).fetchall()
            else:
                sql = (
                    f"SELECT {cols} FROM topics t"
                    + (f" WHERE {tag_where}" if tag_where else "")
                    + f" ORDER BY t.starred DESC, {secondary} LIMIT ?"
                )
                rows = self._c().execute(
                    sql, (*tag_params, max(1, min(300, limit))),
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
        d["groups"] = self.groups_for_topic(d["id"])
        return d

    def search_topics(self, query: str, limit: int = 10, order: str = "recent",
                      group: str = "", tags: list[str] | None = None) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return self.list_topics(limit=limit, order=order, group=group, tags=tags)
        # Substring (LIKE) match over label/summary/keywords — reliable partial
        # matching ("Bise" → "Biserka…"). Combined with group + tag filters
        # (AND): a topic must match the text AND be in the group AND carry
        # every active tag.
        like = f"%{q}%"
        secondary = "label COLLATE NOCASE ASC" if order == "alpha" else "last_seen DESC"
        tag_terms = [t.strip().lower() for t in (tags or []) if str(t).strip()]
        tag_where = " AND ".join(["LOWER(t.keywords) LIKE ?"] * len(tag_terms))
        tag_params = [f"%{t}%" for t in tag_terms]
        cols = "t.id, t.label, t.summary, t.keywords, t.msg_count, t.starred, t.last_seen"
        text_where = "(t.label LIKE ? OR t.summary LIKE ? OR t.keywords LIKE ?)"
        with self._lock:
            if group:
                sql = (
                    f"SELECT {cols} FROM topics t"
                    " JOIN topic_group_members m ON m.topic_id = t.id"
                    f" WHERE m.group_id = ? AND {text_where}"
                    + (f" AND {tag_where}" if tag_where else "")
                    + f" ORDER BY t.starred DESC, {secondary} LIMIT ?"
                )
                rows = self._c().execute(
                    sql, (_slug(group), like, like, like, *tag_params, max(1, min(300, limit))),
                ).fetchall()
            else:
                sql = (
                    f"SELECT {cols} FROM topics t WHERE {text_where}"
                    + (f" AND {tag_where}" if tag_where else "")
                    + f" ORDER BY t.starred DESC, {secondary} LIMIT ?"
                )
                rows = self._c().execute(
                    sql, (like, like, like, *tag_params, max(1, min(300, limit))),
                ).fetchall()
        return [dict(r) for r in rows]

    def set_star(self, topic_id: str, starred: bool) -> bool:
        with self._lock:
            conn = self._c()
            cur = conn.execute("UPDATE topics SET starred = ? WHERE id = ?",
                               (1 if starred else 0, topic_id))
            if cur.rowcount == 0:
                cur = conn.execute("UPDATE topics SET starred = ? WHERE id = ?",
                                   (1 if starred else 0, _slug(topic_id)))
            conn.commit()
        return cur.rowcount > 0

    # ── groups (many-to-many) ───────────────────────────────────────────

    def list_groups(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._c().execute(
                "SELECT g.id, g.name, COUNT(m.topic_id) AS count"
                " FROM topic_groups g LEFT JOIN topic_group_members m ON m.group_id = g.id"
                " GROUP BY g.id, g.name ORDER BY g.name COLLATE NOCASE ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def create_group(self, name: str) -> dict[str, Any] | None:
        name = (name or "").strip()
        if not name:
            return None
        gid = _slug(name)
        with self._lock:
            conn = self._c()
            conn.execute(
                "INSERT OR IGNORE INTO topic_groups (id, name, created_at) VALUES (?, ?, ?)",
                (gid, name[:60], _utcnow()),
            )
            conn.commit()
        return {"id": gid, "name": name[:60]}

    def delete_group(self, group_id: str) -> bool:
        gid = _slug(group_id)
        with self._lock:
            conn = self._c()
            cur = conn.execute("DELETE FROM topic_groups WHERE id = ?", (gid,))
            conn.execute("DELETE FROM topic_group_members WHERE group_id = ?", (gid,))
            conn.commit()
        return cur.rowcount > 0

    def groups_for_topic(self, topic_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._c().execute(
                "SELECT g.id, g.name FROM topic_group_members m"
                " JOIN topic_groups g ON g.id = m.group_id WHERE m.topic_id = ?"
                " ORDER BY g.name COLLATE NOCASE ASC",
                (topic_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_topic_groups(self, topic_id: str, group_ids: list[str]) -> list[dict[str, Any]]:
        """Replace a topic's group memberships with ``group_ids`` (slugs).
        ``topic_id`` is the stored slug (as the UI passes it)."""
        gids = [_slug(g) for g in group_ids if str(g).strip()]
        with self._lock:
            conn = self._c()
            conn.execute("DELETE FROM topic_group_members WHERE topic_id = ?", (topic_id,))
            # Only attach to groups that exist.
            existing = {r[0] for r in conn.execute("SELECT id FROM topic_groups").fetchall()}
            conn.executemany(
                "INSERT OR IGNORE INTO topic_group_members (group_id, topic_id) VALUES (?, ?)",
                [(g, topic_id) for g in gids if g in existing],
            )
            conn.commit()
        return self.groups_for_topic(topic_id)

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

    def reset_all(self, preserve_ids: list[str] | None = None) -> dict[str, int]:
        """Wipe all topics, their messages, and the backfill-progress markers — a
        clean slate so a fresh backfill reconsiders every message. When
        ``preserve_ids`` is given, those topics (with their messages and the
        seen-markers for their messages) are kept intact; everything else is
        wiped. The seen-markers for preserved topics stay so the next Generate
        pass doesn't try to reclassify already-categorised content."""
        preserve = {str(i).strip() for i in (preserve_ids or []) if str(i).strip()}
        with self._lock:
            conn = self._c()
            if not preserve:
                n = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
                conn.executescript(
                    "DELETE FROM topics; DELETE FROM topic_messages;"
                    " DELETE FROM topics_fts; DELETE FROM backfill_seen;"
                )
                conn.commit()
                return {"cleared_topics": int(n), "preserved": 0}
            placeholders = ",".join("?" * len(preserve))
            preserve_t = tuple(preserve)
            n = conn.execute(
                f"SELECT COUNT(*) FROM topics WHERE id NOT IN ({placeholders})",
                preserve_t,
            ).fetchone()[0]
            keep_seen = {r[0] for r in conn.execute(
                f"SELECT DISTINCT msg_id FROM topic_messages"
                f" WHERE topic_id IN ({placeholders}) AND msg_id != ''",
                preserve_t,
            ).fetchall()}
            conn.execute(f"DELETE FROM topic_messages WHERE topic_id NOT IN ({placeholders})", preserve_t)
            conn.execute(f"DELETE FROM topics WHERE id NOT IN ({placeholders})", preserve_t)
            conn.execute(f"DELETE FROM topics_fts WHERE id NOT IN ({placeholders})", preserve_t)
            conn.execute("DELETE FROM backfill_seen")
            if keep_seen:
                conn.executemany(
                    "INSERT OR IGNORE INTO backfill_seen (msg_id) VALUES (?)",
                    [(m,) for m in keep_seen],
                )
            conn.commit()
            return {"cleared_topics": int(n), "preserved": len(preserve)}

    def seen_msg_ids(self) -> set[str]:
        """Message ids the backfill has already attempted (stored or skipped)."""
        with self._lock:
            rows = self._c().execute("SELECT msg_id FROM backfill_seen").fetchall()
        return {r[0] for r in rows}

    def mark_seen(self, msg_ids: list[str]) -> None:
        ids = [m for m in msg_ids if m]
        if not ids:
            return
        with self._lock:
            conn = self._c()
            conn.executemany("INSERT OR IGNORE INTO backfill_seen (msg_id) VALUES (?)", [(m,) for m in ids])
            conn.commit()

    def refresh_excerpts(self, topic_id: str, session_map: dict[str, str]) -> int:
        """Re-sync a topic's stored excerpts to the full session text, by msg_id.
        Fixes topics whose excerpts were captured under an older, smaller cap."""
        updated = 0
        with self._lock:
            conn = self._c()
            # accept either the stored id or a label (→ slug)
            rows = conn.execute(
                "SELECT id, msg_id FROM topic_messages WHERE topic_id = ?", (topic_id,)
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    "SELECT id, msg_id FROM topic_messages WHERE topic_id = ?", (_slug(topic_id),)
                ).fetchall()
            for r in rows:
                mid = r["msg_id"]
                if mid and mid in session_map:
                    conn.execute(
                        "UPDATE topic_messages SET excerpt = ? WHERE id = ?",
                        (session_map[mid][:_MAX_EXCERPT_CHARS], r["id"]),
                    )
                    updated += 1
            conn.commit()
        return updated

    def combine_topics(self, target_id: str, source_ids: list[str]) -> dict[str, Any] | None:
        """Merge ``source_ids`` into ``target_id``: re-point their messages (dedup
        by msg_id), merge keywords + summaries, delete the sources. Returns the
        merged topic (with messages)."""
        target_id = _slug(target_id)
        now = _utcnow()
        with self._lock:
            conn = self._c()
            tgt = conn.execute("SELECT * FROM topics WHERE id = ?", (target_id,)).fetchone()
            if not tgt:
                return None
            seen = {r["msg_id"] for r in conn.execute(
                "SELECT msg_id FROM topic_messages WHERE topic_id = ? AND msg_id != ''", (target_id,)
            ).fetchall()}
            kw = [k for k in (tgt["keywords"] or "").split(",") if k]
            summary = tgt["summary"] or ""
            for raw_sid in source_ids:
                sid = _slug(raw_sid)
                if sid == target_id:
                    continue
                src = conn.execute("SELECT * FROM topics WHERE id = ?", (sid,)).fetchone()
                if not src:
                    continue
                for sm in conn.execute(
                    "SELECT id, msg_id FROM topic_messages WHERE topic_id = ?", (sid,)
                ).fetchall():
                    if sm["msg_id"] and sm["msg_id"] in seen:
                        conn.execute("DELETE FROM topic_messages WHERE id = ?", (sm["id"],))  # dup
                    else:
                        conn.execute("UPDATE topic_messages SET topic_id = ? WHERE id = ?", (target_id, sm["id"]))
                        if sm["msg_id"]:
                            seen.add(sm["msg_id"])
                for k in (src["keywords"] or "").split(","):
                    if k and k not in kw:
                        kw.append(k)
                if src["summary"] and src["summary"] not in summary:
                    summary = (summary + " " + src["summary"]).strip()
                conn.execute("DELETE FROM topics WHERE id = ?", (sid,))
                conn.execute("DELETE FROM topics_fts WHERE id = ?", (sid,))
            cnt = conn.execute(
                "SELECT COUNT(*) FROM topic_messages WHERE topic_id = ?", (target_id,)
            ).fetchone()[0]
            merged_kw = ",".join(kw[:12])
            summary = summary[:1000]
            conn.execute(
                "UPDATE topics SET msg_count = ?, keywords = ?, summary = ?, last_seen = ? WHERE id = ?",
                (cnt, merged_kw, summary, now, target_id),
            )
            conn.execute("DELETE FROM topics_fts WHERE id = ?", (target_id,))
            conn.execute(
                "INSERT INTO topics_fts (id, label, summary, keywords) VALUES (?, ?, ?, ?)",
                (target_id, tgt["label"], summary, merged_kw),
            )
            conn.commit()
        return self.get_topic(target_id, max_excerpts=200)

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
    "message). Keep any internal thinking MINIMAL — do not deliberate message by "
    "message; decide quickly and spend your output on the JSON, not on reasoning. "
    "Output ONLY the JSON array as your final answer — start with '[' and end with ']'."
)


def _collect_new_messages(agent: Any, last_idx: int, cap: int) -> tuple[list[dict[str, Any]], int]:
    """Comms messages (user + assistant) since ``last_idx`` + buffered narration.
    Returns (items, new_last_idx). Skips messages already classified into a topic
    (mirrors the backfill behaviour) so an agent restart — which resets last_idx
    to 0 in memory — doesn't re-classify the whole session and duplicate topics."""
    items: list[dict[str, Any]] = []
    msgs = agent.session.messages if agent.session else []
    new_idx = len(msgs)
    try:
        _mgr = get_topics_manager()
        done_ids = _mgr.classified_msg_ids() | _mgr.seen_msg_ids()
    except Exception:
        done_ids = set()
    for m in msgs[last_idx:]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        mid = str(m.get("message_id") or "")
        if mid and mid in done_ids:
            continue  # already classified into a topic — don't duplicate
        items.append({
            "role": "user" if role == "user" else "agent",
            "channel": str((m.get("metadata") or {}).get("channel") or "") if isinstance(m.get("metadata"), dict) else "",
            "excerpt": content[:_MAX_EXCERPT_CHARS],
            "msg_id": mid,
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
    existing = mgr.list_topics(limit=300)
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
        max_tokens=min(int(tc.classify_max_tokens), int(get_config().model.max_tokens)),
    )
    raw = (response.content or "").strip()
    groups = _parse_groups(raw)
    if not groups:
        log.warning("topic classify: no groups parsed from reply (len=%d): %s", len(raw), raw[:300])
    touched = 0
    for g in groups:
        label = str(g.get("label") or "").strip()
        if not label:
            continue
        # Tolerate string indices ("0"), floats, and out-of-range values — small
        # models often emit indices as strings, which silently dropped every group.
        idxs: list[int] = []
        for i in g.get("messages", []):
            try:
                n = int(i)
            except (ValueError, TypeError):
                continue
            if 0 <= n < len(items):
                idxs.append(n)
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


def refresh_topic(agent: Any, topic_id: str) -> dict[str, Any]:
    """Re-pull the FULL text for a topic's messages from the live session (by
    msg_id) and update the stored excerpts — fixes topics captured under an older
    truncation cap. Messages no longer in the session keep their stored text."""
    mgr = get_topics_manager()
    session_map: dict[str, str] = {}
    if agent.session and agent.session.messages:
        for m in agent.session.messages:
            mid = str(m.get("message_id") or "")
            content = str(m.get("content") or "")
            if mid and content:
                session_map[mid] = content
    updated = mgr.refresh_excerpts(topic_id, session_map)
    return {"ok": True, "updated": updated}


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
    # Skip messages already in a topic OR already attempted (so a message the
    # classifier puts in no topic still counts as processed and isn't reprocessed).
    done_ids = mgr.classified_msg_ids() | mgr.seen_msg_ids()

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

    # Process ONE batch per call and report how many are left. One LLM call per
    # request keeps each round well under the FD→agent proxy timeout; the UI
    # auto-continues until remaining hits 0. (Looping all batches server-side
    # blew past the 15s proxy timeout on "All history".)
    batch = max(5, int(tc.max_messages_per_pass))
    chunk = pending[:batch]
    try:
        touched = await _classify_and_store(agent, chunk)
    except Exception as exc:
        log.warning("topic backfill classify failed: %s", exc)
        return {"ok": False, "error": f"classification failed: {exc}"[:300],
                "classified": 0, "topics_touched": 0, "remaining": len(pending)}
    # Mark the whole chunk attempted so it's never reprocessed (guarantees progress).
    mgr.mark_seen([str(it.get("msg_id") or "") for it in chunk])
    remaining = max(0, len(pending) - len(chunk))
    log.info("topic backfill: classified %d message(s) into %d topic touch(es), %d remaining",
             len(chunk), touched, remaining)
    return {"ok": True, "classified": len(chunk), "topics_touched": touched, "remaining": remaining}


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
