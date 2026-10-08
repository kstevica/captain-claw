"""Conversation topic memory — automatic tagging/clustering over comms traffic.

A periodic pass (mirroring the dreaming / insight-extraction passes) takes the
recent conversation — what people typed, and the final reply of each turn they
opened (never fleet notices, cron prompts, correctives or tool steps) — and
groups it into persistent, cross-session **topics**. Each topic carries a
rolling summary, keywords, and the message excerpts that fed it.

The agent reaches this via the always-on ``topics`` tool (list / get / search),
so when a thread resurfaces ("back to the Munich trip") it pulls the whole
cluster instantly instead of re-deriving it from a long transcript.

Self-contained SQLite store (``conversation_topics.db``), sync sqlite3 + a lock,
matching the other Captain Claw memory stores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Per-agent attrs for the periodic-pass guard (mirrors nervous_system).
_ATTR_RUNNING = "_topics_classify_running"
_ATTR_LAST_TIME = "_topics_last_classify_time"
_ATTR_NARRATION = "_topics_narration_buffer"

_MAX_NARRATION_BUFFER = 40   # cap buffered narration blurbs between passes
_CHUNKS_PER_PASS = 3         # classifier batches per live pass (the rest waits)
# Existing topics the classifier sees: best matches for the batch + most recent.
_CLASSIFIER_RELEVANT = 15
_CLASSIFIER_RECENT = 10
_CLASSIFIER_TERMS = 150      # every content word of a batch; bm25 weighs them
_CLASSIFIER_SUMMARY_CHARS = 600
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
        # Topic id -> (text fingerprint, embedding) for the cosine leg.
        self._vectors: dict[str, tuple[str, list[float]]] = {}
        self._ensure_db()
        self._seen_from_classified_once()
        self._hide_synthetic_once()

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
                    last_seen   TEXT NOT NULL,
                    hidden      INTEGER NOT NULL DEFAULT 0  -- kept out of recall; still listed
                );
                CREATE TABLE IF NOT EXISTS topic_messages (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id  TEXT NOT NULL,
                    role      TEXT NOT NULL DEFAULT '',   -- user | agent | narration
                    channel   TEXT NOT NULL DEFAULT '',
                    excerpt   TEXT NOT NULL DEFAULT '',
                    msg_id    TEXT NOT NULL DEFAULT '',   -- session message_id (dedup/backfill)
                    ts        TEXT NOT NULL,
                    speaker   TEXT NOT NULL DEFAULT '',   -- shared-agent member id; '' = owner
                    session_id TEXT NOT NULL DEFAULT ''
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
                -- One-off maintenance markers (e.g. the synthetic-topic sweep ran).
                CREATE TABLE IF NOT EXISTS topics_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '');
                """
            )
            if tm_cols and "msg_id" not in tm_cols:  # migrate pre-existing tables
                self._c().execute("ALTER TABLE topic_messages ADD COLUMN msg_id TEXT NOT NULL DEFAULT ''")
            if tm_cols and "speaker" not in tm_cols:
                self._c().execute("ALTER TABLE topic_messages ADD COLUMN speaker TEXT NOT NULL DEFAULT ''")
            # After the ALTER: a pre-A2 table has no speaker column before it.
            self._c().execute(
                "CREATE INDEX IF NOT EXISTS idx_tm_speaker ON topic_messages(topic_id, speaker, id DESC)"
            )
            if t_cols and "starred" not in t_cols:
                self._c().execute("ALTER TABLE topics ADD COLUMN starred INTEGER NOT NULL DEFAULT 0")
            if t_cols and "hidden" not in t_cols:
                self._c().execute("ALTER TABLE topics ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
            if tm_cols and "session_id" not in tm_cols:
                self._c().execute("ALTER TABLE topic_messages ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
            self._c().commit()

    # ── reads (used by the tool) ────────────────────────────────────────

    def list_topics(self, limit: int = 40, order: str = "recent", group: str = "",
                    tags: list[str] | None = None, *, include_hidden: bool = False) -> list[dict[str, Any]]:
        # Starred topics always float to the top; within each group, by recency
        # (newest message) or alphabetically. ``group`` and ``tags`` AND together
        # — a topic must be in the group AND carry EVERY active tag (substring
        # match against its keywords column). Hidden topics are left out unless
        # asked for (the panel lists everything; recall never sees them).
        secondary = "label COLLATE NOCASE ASC" if order == "alpha" else "last_seen DESC"
        where, params = self._filters(group, tags, include_hidden)
        with self._lock:
            rows = self._c().execute(
                f"SELECT {_LIST_COLS} FROM topics t {where}"
                f" ORDER BY t.starred DESC, {secondary} LIMIT ?",
                (*params, max(1, min(300, limit))),
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _filters(group: str, tags: list[str] | None,
                 include_hidden: bool) -> tuple[str, list[Any]]:
        """JOIN/WHERE clause (on alias ``t``) for the group, tag and hidden filters."""
        joins, conds, params = "", [], []
        if group:
            joins = " JOIN topic_group_members m ON m.topic_id = t.id"
            conds.append("m.group_id = ?")
            params.append(_slug(group))
        for tag in (tags or []):
            if str(tag).strip():
                conds.append("LOWER(t.keywords) LIKE ?")
                params.append(f"%{str(tag).strip().lower()}%")
        if not include_hidden:
            conds.append("t.hidden = 0")
        return joins + (" WHERE " + " AND ".join(conds) if conds else ""), params

    def get_topic(self, topic_id: str, *, max_excerpts: int = 40,
                  speaker: str | None = None) -> dict[str, Any] | None:
        """One topic with its recent excerpts. *speaker* (a str) narrows the
        excerpts to that speaker's (``''`` = the owner's); ``msg_count`` stays
        the topic total."""
        with self._lock:
            r = self._c().execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
            if not r:
                # tolerate a label being passed instead of a slug
                r = self._c().execute("SELECT * FROM topics WHERE id = ?", (_slug(topic_id),)).fetchone()
            if not r:
                return None
            if speaker is None:
                msgs = self._c().execute(
                    "SELECT id, role, channel, excerpt, msg_id, ts, speaker, session_id FROM topic_messages"
                    " WHERE topic_id = ? ORDER BY id DESC LIMIT ?",
                    (r["id"], max(1, min(200, max_excerpts))),
                ).fetchall()
            else:
                msgs = self._c().execute(
                    "SELECT id, role, channel, excerpt, msg_id, ts, speaker, session_id FROM topic_messages"
                    " WHERE topic_id = ? AND speaker = ? ORDER BY id DESC LIMIT ?",
                    (r["id"], str(speaker), max(1, min(200, max_excerpts))),
                ).fetchall()
        d = dict(r)
        d["messages"] = [dict(m) for m in reversed(msgs)]  # oldest→newest
        d["groups"] = self.groups_for_topic(d["id"])
        return d

    def labels_for_speaker(self, speaker_id: str, limit: int = 8, since: str = "") -> list[str]:
        """Labels of the topics *speaker_id* (a member) has excerpts in, at or
        after *since*, most recently touched first (PR D, owner-only)."""
        if not speaker_id:
            return []
        try:
            with self._lock:
                rows = self._c().execute(
                    "SELECT t.label FROM topic_messages m JOIN topics t ON t.id = m.topic_id"
                    " WHERE m.speaker = ? AND m.ts >= ? GROUP BY t.id"
                    " ORDER BY MAX(m.id) DESC LIMIT ?",
                    (str(speaker_id), str(since or ""), max(1, int(limit))),
                ).fetchall()
            return [str(r[0]) for r in rows]
        except Exception:
            return []

    def search_topics(self, query: str, limit: int = 10, order: str = "recent",
                      group: str = "", tags: list[str] | None = None, *,
                      include_hidden: bool = False, embedder: Any = None) -> list[dict[str, Any]]:
        """Topics for a typed search: word matches first, by bm25 (every word
        also matches as a prefix, so "Bise" finds "Biserka"), then plain
        substring matches of the whole query, then topics that only match in
        meaning (``related``). ``order="alpha"`` sorts by label, starred first."""
        q = (query or "").strip()
        if not q:
            return self.list_topics(limit=limit, order=order, group=group, tags=tags,
                                    include_hidden=include_hidden)
        n = max(1, min(300, limit))
        _terms, fts, vec = self.rank_legs(q, group=group, tags=tags, include_hidden=include_hidden,
                                          embedder=embedder, prefix_all=True)
        rows = fts[:n]
        have = {r["id"] for r in rows}
        if len(rows) < n:
            like = f"%{q}%"
            where, params = self._filters(group, tags, include_hidden)
            text_where = "(t.label LIKE ? OR t.summary LIKE ? OR t.keywords LIKE ?)"
            where = f"{where} AND {text_where}" if where else f" WHERE {text_where}"
            with self._lock:
                extra = self._c().execute(
                    f"SELECT {_LIST_COLS} FROM topics t {where}"
                    " ORDER BY t.starred DESC, t.last_seen DESC LIMIT ?",
                    (*params, like, like, like, n),
                ).fetchall()
            for r in extra:
                if r["id"] not in have and len(rows) < n:
                    rows.append(dict(r))
                    have.add(r["id"])
        for r in vec:
            if r["id"] not in have and len(rows) < n:
                rows.append({**r, "related": True})
                have.add(r["id"])
        if order == "alpha":
            rows.sort(key=lambda r: (not r.get("starred"), str(r.get("label") or "").lower()))
        return rows

    def rank_topics(self, query: str, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Topics ranked by relevance to *query*: the two legs of
        ``rank_legs`` fused by reciprocal rank (k=60). Each row carries
        ``score``, ``bm25`` (lower is better, None when the words missed),
        ``cosine`` (None without an embedder or below its floor) and
        ``matched_terms``."""
        _terms, fts, vec = self.rank_legs(query, **kwargs)
        fused: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(fts):
            entry = fused.setdefault(row["id"], {**row, "score": 0.0, "cosine": None})
            entry["score"] += 1.0 / (_RRF_K + rank + 1)
        for rank, row in enumerate(vec):
            entry = fused.setdefault(row["id"], {**row, "score": 0.0, "bm25": None})
            entry["cosine"] = row["cosine"]
            entry["score"] += 1.0 / (_RRF_K + rank + 1)
        return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[: max(1, limit)]

    def rank_legs(self, query: str, *, group: str = "", tags: list[str] | None = None,
                  include_hidden: bool = False, embedder: Any = None, max_terms: int = 8,
                  prefix_all: bool = False) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
        """(terms, word matches, meaning matches) for *query*.

        Word matches: FTS5 bm25 over ``topics_fts`` (label x3, keywords x2,
        summary x1) on the query's content words, best first, each with
        ``bm25``. Meaning matches, when *embedder* (``texts -> normalized
        vectors``) is given: cosine against cached topic vectors, best first,
        each with ``cosine``; only those above an absolute floor and within
        reach of the best one. Rows of both carry ``matched_terms``."""
        terms = query_terms(query, max_terms=max_terms)
        where, params = self._filters(group, tags, include_hidden)
        fts_rows: list[dict[str, Any]] = []
        if terms:
            match = " OR ".join(_fts_term(t, prefix_all) for t in terms)
            cond = "topics_fts MATCH ?"
            sql_where = f"{where} AND {cond}" if where else f" WHERE {cond}"
            try:
                with self._lock:
                    fts_rows = [dict(r) for r in self._c().execute(
                        f"SELECT {_LIST_COLS}, bm25(topics_fts, 0.0, 3.0, 1.0, 2.0) AS bm25"
                        f" FROM topics_fts JOIN topics t ON t.id = topics_fts.id {sql_where}"
                        " ORDER BY bm25 LIMIT ?",
                        (*params, match, _CANDIDATES),
                    ).fetchall()]
            except sqlite3.Error as exc:
                log.debug("topic FTS query failed: %s", exc)
        vec_rows: list[dict[str, Any]] = []
        if embedder is not None and str(query or "").strip():
            vec_rows = self._cosine_candidates(query, where, params, embedder)
        for row in fts_rows + vec_rows:
            hay = _fold(" ".join(str(row.get(k) or "") for k in ("label", "summary", "keywords")))
            row["matched_terms"] = [t for t in terms if _stem(_fold(t)) in hay]
        return terms, fts_rows, vec_rows

    def _cosine_candidates(self, query: str, where: str, params: list[Any],
                           embedder: Any) -> list[dict[str, Any]]:
        """Topics by cosine to *query*; topic vectors are cached per text."""
        with self._lock:
            rows = [dict(r) for r in self._c().execute(
                f"SELECT {_LIST_COLS} FROM topics t {where}", tuple(params),
            ).fetchall()]
        if not rows:
            return []
        texts = {r["id"]: _topic_text(r) for r in rows}
        missing = [tid for tid, text in texts.items()
                   if self._vectors.get(tid, ("", []))[0] != _fingerprint(text)]
        try:
            vectors = embedder([str(query)] + [texts[tid] for tid in missing])
        except Exception as exc:
            log.debug("topic embedding failed: %s", exc)
            return []
        if not vectors or len(vectors) != len(missing) + 1:
            return []
        qvec = vectors[0]
        for tid, vec in zip(missing, vectors[1:]):
            self._vectors[tid] = (_fingerprint(texts[tid]), vec)
        scored = []
        for row in rows:
            vec = self._vectors.get(row["id"], ("", []))[1]
            if len(vec) != len(qvec):
                self._vectors.pop(row["id"], None)   # another provider's vector
                continue
            scored.append({**row, "cosine": round(sum(a * b for a, b in zip(qvec, vec)), 4)})
        scored.sort(key=lambda r: r["cosine"], reverse=True)
        if not scored:
            return []
        floor = max(_COSINE_FLOOR, _COSINE_REACH * scored[0]["cosine"])
        return [r for r in scored if r["cosine"] >= floor][:_CANDIDATES]

    def set_star(self, topic_id: str, starred: bool) -> bool:
        # Starring a topic also brings it back into recall (the panel's way
        # to undo a hide).
        sql = "UPDATE topics SET starred = ?" + (", hidden = 0" if starred else "") + " WHERE id = ?"
        with self._lock:
            conn = self._c()
            cur = conn.execute(sql, (1 if starred else 0, topic_id))
            if cur.rowcount == 0:
                cur = conn.execute(sql, (1 if starred else 0, _slug(topic_id)))
            conn.commit()
        return cur.rowcount > 0

    def recent_topics(self, limit: int = 10, *, include_hidden: bool = False) -> list[dict[str, Any]]:
        """The most recently touched topics, starred or not."""
        where = "" if include_hidden else " WHERE t.hidden = 0"
        with self._lock:
            rows = self._c().execute(
                f"SELECT {_LIST_COLS} FROM topics t{where} ORDER BY t.last_seen DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_hidden(self, topic_id: str, hidden: bool) -> bool:
        """Keep a topic out of recall (``topics`` tool, classifier context),
        or bring it back. The panel still lists it."""
        with self._lock:
            conn = self._c()
            cur = conn.execute("UPDATE topics SET hidden = ? WHERE id = ?",
                               (1 if hidden else 0, topic_id))
            if cur.rowcount == 0:
                cur = conn.execute("UPDATE topics SET hidden = ? WHERE id = ?",
                                   (1 if hidden else 0, _slug(topic_id)))
            conn.commit()
        return cur.rowcount > 0

    def hide_synthetic_topics(self, threshold: float = 0.6) -> list[str]:
        """Hide topics built mostly from machine text (fleet notices, cron
        prompts, correctives) filed as the user's messages before ingest was
        filtered. Starred topics are never hidden. Returns the hidden ids."""
        from captain_claw import msg_origin

        with self._lock:
            rows = self._c().execute(
                "SELECT m.topic_id, m.excerpt FROM topic_messages m JOIN topics t ON t.id = m.topic_id"
                " WHERE m.role = 'user' AND t.starred = 0 AND t.hidden = 0"
            ).fetchall()
        totals: dict[str, list[int]] = {}
        for row in rows:
            seen = totals.setdefault(row["topic_id"], [0, 0])
            seen[0] += 1
            if msg_origin.detect_literal(row["excerpt"]) is not None:
                seen[1] += 1
        ids = [tid for tid, (n, synthetic) in totals.items() if n >= 2 and synthetic / n >= threshold]
        if ids:
            with self._lock:
                conn = self._c()
                conn.executemany("UPDATE topics SET hidden = ? WHERE id = ?",
                                 [(_HIDDEN_BY_SWEEP, i) for i in ids])
                conn.commit()
        return ids

    def _seen_from_classified_once(self) -> None:
        """Before the live pass marked what it attempted, only the backfill
        did; mark everything already classified as attempted, once, so the
        seen watermark starts where the old pass left off."""
        try:
            with self._lock:
                conn = self._c()
                if conn.execute("SELECT 1 FROM topics_meta WHERE key = 'seen_from_classified_v1'").fetchone():
                    return
                conn.execute(
                    "INSERT OR IGNORE INTO backfill_seen (msg_id)"
                    " SELECT DISTINCT msg_id FROM topic_messages WHERE msg_id != ''"
                )
                conn.execute("INSERT OR REPLACE INTO topics_meta (key, value) VALUES ('seen_from_classified_v1', ?)",
                             (_utcnow(),))
                conn.commit()
        except Exception as exc:
            log.debug("seen-marker migration failed: %s", exc)

    def _hide_synthetic_once(self) -> None:
        """Run the synthetic-topic sweep once per store, so a topic the user
        brings back stays back."""
        try:
            with self._lock:
                done = self._c().execute(
                    "SELECT 1 FROM topics_meta WHERE key = 'synthetic_sweep_v1'").fetchone()
            if done:
                return
            hidden = self.hide_synthetic_topics()
            with self._lock:
                self._c().execute(
                    "INSERT OR REPLACE INTO topics_meta (key, value) VALUES ('synthetic_sweep_v1', ?)",
                    (json.dumps({"at": _utcnow(), "hidden": hidden}),),
                )
                self._c().commit()
            if hidden:
                log.info("conversation topics: hid %d machine-text topic(s)", len(hidden))
        except Exception as exc:
            log.debug("synthetic topic sweep failed: %s", exc)

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
                # New conversation filed under a topic the machine-text sweep
                # hid brings it back; a topic the user hid stays hidden.
                conn.execute(
                    "UPDATE topics SET label = ?, summary = COALESCE(NULLIF(?, ''), summary),"
                    " keywords = ?, last_seen = ?,"
                    " hidden = CASE WHEN hidden = ? THEN 0 ELSE hidden END WHERE id = ?",
                    (label[:120], summary[:1000], merged_kw, now, _HIDDEN_BY_SWEEP, tid),
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
        """Append message excerpts to a topic, bump its count, prune to ``cap``.

        Pruning is per ``(topic, speaker)`` (``''`` = the owner), so one
        shared-agent member's messages never evict the owner's or another
        member's excerpts. ``msg_count`` still counts every message.
        """
        if not messages:
            return
        now = _utcnow()
        rows = [(topic_id, str(m.get("role") or ""), str(m.get("channel") or ""),
                 str(m.get("excerpt") or "")[:_MAX_EXCERPT_CHARS], str(m.get("msg_id") or ""),
                 str(m.get("ts") or now), str(m.get("session_id") or ""),
                 str(m.get("speaker") or "")) for m in messages]
        with self._lock:
            conn = self._c()
            conn.executemany(
                "INSERT INTO topic_messages (topic_id, role, channel, excerpt, msg_id, ts, session_id, speaker)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.execute(
                "UPDATE topics SET msg_count = msg_count + ?, last_seen = ? WHERE id = ?",
                (len(messages), now, topic_id),
            )
            # Prune oldest excerpts beyond the cap, per speaker.
            for spk in sorted({row[-1] for row in rows}):
                conn.execute(
                    "DELETE FROM topic_messages WHERE topic_id = ? AND speaker = ? AND id NOT IN ("
                    " SELECT id FROM topic_messages WHERE topic_id = ? AND speaker = ?"
                    " ORDER BY id DESC LIMIT ?)",
                    (topic_id, spk, topic_id, spk, max(1, cap)),
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

    def unclassify_topic(self, topic_id: str) -> dict[str, Any]:
        """Drop a topic and free its messages so the next classifier pass can
        redistribute them (to other existing topics, or a new one). Removes the
        msg_ids from backfill_seen too — otherwise they'd stay 'seen' and the
        backfill loop would silently skip them. Useful when unrelated content
        got lumped into one topic and the user wants to redo it."""
        tid = str(topic_id or "")
        with self._lock:
            conn = self._c()
            # Accept stored id OR a label slug.
            if not conn.execute("SELECT 1 FROM topics WHERE id = ?", (tid,)).fetchone():
                tid = _slug(topic_id)
            if not conn.execute("SELECT 1 FROM topics WHERE id = ?", (tid,)).fetchone():
                return {"ok": False, "error": "topic not found", "freed": 0, "freed_ids": 0}
            msg_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT msg_id FROM topic_messages WHERE topic_id = ? AND msg_id != ''",
                (tid,),
            ).fetchall()]
            n_msgs = conn.execute(
                "SELECT COUNT(*) FROM topic_messages WHERE topic_id = ?", (tid,),
            ).fetchone()[0]
            if msg_ids:
                placeholders = ",".join("?" * len(msg_ids))
                conn.execute(
                    f"DELETE FROM backfill_seen WHERE msg_id IN ({placeholders})",
                    tuple(msg_ids),
                )
            conn.execute("DELETE FROM topic_messages WHERE topic_id = ?", (tid,))
            conn.execute("DELETE FROM topics WHERE id = ?", (tid,))
            conn.execute("DELETE FROM topics_fts WHERE id = ?", (tid,))
            conn.execute("DELETE FROM topic_group_members WHERE topic_id = ?", (tid,))
            conn.commit()
        return {"ok": True, "freed": int(n_msgs), "freed_ids": len(msg_ids)}

    def move_message(self, message_row_id: int | str, target_topic_id: str) -> dict[str, Any]:
        """Move a single ``topic_messages`` row to a different topic and keep both
        topics' msg_count + last_seen in sync. Accepts the target as a stored id
        or a label slug. Idempotent: moving to the same topic is a no-op."""
        try:
            rid = int(message_row_id)
        except (ValueError, TypeError):
            return {"ok": False, "error": "invalid message id"}
        tgt = str(target_topic_id or "")
        with self._lock:
            conn = self._c()
            row = conn.execute(
                "SELECT topic_id FROM topic_messages WHERE id = ?", (rid,),
            ).fetchone()
            if not row:
                return {"ok": False, "error": "message not found"}
            src_id = row["topic_id"]
            if not conn.execute("SELECT 1 FROM topics WHERE id = ?", (tgt,)).fetchone():
                tgt = _slug(target_topic_id)
            if not conn.execute("SELECT 1 FROM topics WHERE id = ?", (tgt,)).fetchone():
                return {"ok": False, "error": "target topic not found"}
            if src_id == tgt:
                return {"ok": True, "moved": False, "src": src_id, "target": tgt}
            conn.execute("UPDATE topic_messages SET topic_id = ? WHERE id = ?", (tgt, rid))
            now = _utcnow()
            for tid in (src_id, tgt):
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM topic_messages WHERE topic_id = ?", (tid,),
                ).fetchone()[0]
                conn.execute("UPDATE topics SET msg_count = ? WHERE id = ?", (cnt, tid))
            conn.execute("UPDATE topics SET last_seen = ? WHERE id = ?", (now, tgt))
            conn.commit()
        return {"ok": True, "moved": True, "src": src_id, "target": tgt}

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
        """Drop the least-recently-seen topics beyond ``max_topics`` (starred
        ones are dropped last)."""
        with self._lock:
            conn = self._c()
            stale = conn.execute(
                "SELECT id FROM topics ORDER BY starred DESC, last_seen DESC LIMIT -1 OFFSET ?",
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


_HIDDEN_BY_SWEEP = 2   # topics.hidden: 1 = the user hid it, 2 = the machine-text sweep did

_LIST_COLS = ("t.id, t.label, t.summary, t.keywords, t.msg_count, t.starred, t.hidden,"
              " t.first_seen, t.last_seen")
_CANDIDATES = 50     # per ranking leg, before fusion
_RRF_K = 60
# A meaning match counts only above this cosine and within this share of the
# best one (local model2vec: related topics score 0.4-0.8, unrelated <= 0.2).
_COSINE_FLOOR = 0.3
_COSINE_REACH = 0.75

# Function words dropped from search terms (English + Croatian).
_STOPWORDS = frozenset("""
the and for are but not you all any can had her was one our out has have him his how its may new now
old see two who did get got let put say she too use with that this from they will would there their
what about which when were your been into than then them these those some such only also just like
more most other over very much many each here where while why after before again because could
should does doing being both same own off once under until above below between through during
please thanks thank okay yes hello want need make know think tell give show find look help
back still really maybe something anything everything going able
sam smo ste jesam nije nisu bio bila bilo biti ima imam imamo nema samo ali ako ili kao koji koja
koje kojeg kojem kojim kad kada gdje kako zato zbog što sto tko neki neka neko nešto nesto ovo ovaj
ova taj ta to tamo ovdje jer još jos već vec sve svi sva svoj svoja moj moja tvoj tvoja naš nas
vaš vas njih njega njoj nego pa te ni niti treba trebam možeš mozes može moze molim hvala daj
""".split())


def _fold(text: str) -> str:
    """Lower-case, combining accents removed — what FTS5 unicode61 folds
    (č→c, š→s; not đ, which has no decomposition)."""
    decomposed = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def query_terms(text: str, max_terms: int = 8) -> list[str]:
    """Content words of *text* for topic search: 3+ letters, no stopwords,
    no bare short numbers; the longest *max_terms*, in their original order."""
    seen: dict[str, int] = {}
    for pos, word in enumerate(re.findall(r"\w+", str(text or "").lower())):
        word = word.strip("_")
        if len(word) < 3 or _fold(word) in _STOPWORDS or word in _STOPWORDS:
            continue
        if word.isdigit() and len(word) < 4:
            continue
        seen.setdefault(word, pos)
    keep = sorted(seen, key=lambda w: (-len(w), seen[w]))[: max(1, max_terms)]
    return sorted(keep, key=lambda w: seen[w])


def _stem(term: str) -> str:
    """The prefix a term matches on: long words drop their last two letters
    (inflection: putovanje/putovanja, invoices/invoice)."""
    if len(term) < 6:
        return term
    return term[: max(5, len(term) - 2)]


def _fts_term(term: str, prefix_all: bool) -> str:
    """A long word matches by its stem as a prefix; with *prefix_all* (typed
    search) every word does."""
    stem = _stem(term).replace('"', "")
    return f'"{stem}"*' if (prefix_all or len(term) >= 6) else f'"{stem}"'


def _topic_text(row: dict[str, Any]) -> str:
    return f"{row.get('label') or ''}. {row.get('summary') or ''} {row.get('keywords') or ''}".strip()


def _fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def topic_embedder(agent: Any) -> Any:
    """``texts -> vectors`` from the agent's semantic-memory embedding chain,
    or None when only the lexical hash fallback is available (it would just
    repeat the FTS leg)."""
    chain = getattr(getattr(getattr(agent, "memory", None), "semantic", None), "embedding_chain", None)
    if chain is None or not getattr(chain, "enabled", False):
        return None
    if str(getattr(chain, "active_provider_key", "")).startswith("local_hash"):
        return None

    def _embed(texts: list[str]) -> list[list[float]]:
        key, vectors = chain.embed_batch(list(texts))
        if str(key).startswith("local_hash"):
            raise RuntimeError("only the lexical fallback answered")
        return vectors

    return _embed


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
    # PR D: a turn that read members' private data narrates nothing into topics.
    from captain_claw import member_privacy
    if member_privacy.private_turn(agent):
        return
    from captain_claw.config import get_config
    if not get_config().conversation_topics.include_narration:
        return
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


def _msg_key(m: dict[str, Any]) -> str:
    """The id a session message is tracked by (a content hash for rows
    stored before message ids existed)."""
    mid = str(m.get("message_id") or "")
    if mid:
        return mid
    raw = f"{m.get('role')}|{m.get('timestamp')}|{str(m.get('content') or '')[:500]}"
    return "h:" + hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:20]


def _conversation_items(agent: Any, msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What a topic is made of: messages a person typed, and the final reply
    of each turn a person opened. Machine text filed as the user's (fleet
    notices, cron prompts, correctives), mid-turn narration, tool steps,
    replies to automated turns and members' private data stay out."""
    from captain_claw import member_privacy, msg_origin
    from captain_claw.speaker import principal_for

    # A shared-agent member instance stamps its member's id on every excerpt
    # (the owner's are ''), so the topics tool shows a member only theirs.
    _p = principal_for(agent)
    speaker_id = _p.speaker_id if _p is not None else ""
    session_id = str(getattr(getattr(agent, "session", None), "id", "") or "")
    items: list[dict[str, Any]] = []
    human_turn, turn_channel = False, ""
    n = len(msgs)
    for i, m in enumerate(msgs):
        role = str(m.get("role") or "")
        if role == "user":
            origin = msg_origin.origin_of(m)
            # A delegated result continues the turn that asked for it: its
            # relayed answer counts as that person's reply (the result text
            # itself never does).
            if origin in msg_origin.USER_OPENER_ORIGINS and origin != "delegated_result":
                human_turn = origin == "human"
                turn_channel = str(m.get("channel") or "")
            if origin != "human" or member_privacy.is_private(m):
                continue
            text, kind = msg_origin.model_view_text(m).strip(), "user"
        elif role == "assistant":
            if not human_turn or m.get("tool_calls") or msg_origin.origin_of(m) != "model":
                continue
            if member_privacy.is_private(m) or not _is_final_reply(msgs, i, n):
                continue
            text, kind = str(m.get("content") or "").strip(), "agent"
        else:
            continue
        if not text:
            continue
        items.append({
            "role": kind,
            "channel": turn_channel,
            "excerpt": text[:_MAX_EXCERPT_CHARS],
            "msg_id": _msg_key(m),
            "ts": str(m.get("timestamp") or _utcnow()),
            "speaker": speaker_id,
            "session_id": session_id,
        })
    return items


def _is_text_reply(m: dict[str, Any]) -> bool:
    from captain_claw import msg_origin

    return (
        m.get("role") == "assistant"
        and not m.get("tool_calls")
        and msg_origin.origin_of(m) == "model"
        and bool(str(m.get("content") or "").strip())
    )


def _is_final_reply(msgs: list[dict[str, Any]], i: int, n: int) -> bool:
    """The assistant message at *i* is its turn's last reply: no later text
    reply from the model comes before the next turn opener. Tool rows
    (delivery notes, TTS, cron notes), correctives and fleet notices in
    between don't count."""
    from captain_claw import msg_origin

    for j in range(i + 1, n):
        m = msgs[j]
        if m.get("role") == "user" and msg_origin.is_turn_opener(m):
            return True
        if _is_text_reply(m):
            return False
    return True


def _pending_items(agent: Any, cap: int = 0, start: int = 0) -> list[dict[str, Any]]:
    """Conversation items not yet in a topic, oldest first, at most *cap*
    (0 = all).

    Progress lives in the store, not in memory: everything after the newest
    item already classified or attempted is pending, so a restart or a
    compaction neither re-reads the session nor stalls the pass. A session
    with no classified item yet starts from its newest *cap* items; older
    history is the backfill's job."""
    msgs = list(agent.session.messages) if agent.session else []
    try:
        _mgr = get_topics_manager()
        seen_ids = _mgr.seen_msg_ids()
        done_ids = _mgr.classified_msg_ids() | seen_ids
    except Exception:
        seen_ids, done_ids = set(), set()
    convo = _conversation_items(agent, msgs[max(0, start):])
    # The watermark is what a pass attempted (seen); a turn filed by hand
    # (Topic chat's append) is skipped but doesn't move it past the rest.
    watermark = max((k for k, it in enumerate(convo) if it["msg_id"] in seen_ids), default=-1)
    pending = [it for it in convo[watermark + 1:] if it["msg_id"] not in done_ids]
    if cap and len(pending) > cap:
        pending = pending[:cap] if watermark >= 0 else pending[-cap:]
    return pending


def _collect_new_messages(agent: Any, last_idx: int = 0, cap: int = 0) -> tuple[list[dict[str, Any]], int]:
    """The pending conversation items (see ``_pending_items``) plus the
    narration buffered since the last pass when ``include_narration`` is on
    (the buffer is drained). Returns (items, len(messages))."""
    from captain_claw.config import get_config
    from captain_claw.speaker import principal_for

    items = _pending_items(agent, cap, last_idx)
    if get_config().conversation_topics.include_narration:
        _p = principal_for(agent)
        speaker_id = _p.speaker_id if _p is not None else ""
        for t in getattr(agent, _ATTR_NARRATION, None) or []:
            if cap and len(items) >= cap:
                break
            items.append({"role": "narration", "channel": "", "excerpt": t[:_MAX_EXCERPT_CHARS],
                          "ts": _utcnow(), "speaker": speaker_id})
        setattr(agent, _ATTR_NARRATION, [])
    return items, len(agent.session.messages) if agent.session else 0


async def maybe_classify_topics(agent: Any) -> int | None:
    """Run a topic-classification pass when enough new conversation has
    built up (``interval_messages`` items) and the cooldown has passed.
    Mirrors maybe_dream's guards. Never raises. Returns topics touched."""
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
        if len(_pending_items(agent)) < max(1, int(tc.interval_messages or 6)):
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
    """Classify the pending conversation now: up to ``_CHUNKS_PER_PASS``
    batches of ``max_messages_per_pass``, oldest first, each marked attempted
    once stored so the next pass moves on."""
    from captain_claw.config import get_config

    tc = get_config().conversation_topics
    per = max(5, int(tc.max_messages_per_pass))
    items, _ = _collect_new_messages(agent, 0, per * _CHUNKS_PER_PASS)
    if not items:
        return 0
    mgr = get_topics_manager()
    touched = 0
    for start in range(0, len(items), per):
        chunk = items[start:start + per]
        touched += await _classify_and_store(agent, chunk)
        mgr.mark_seen([str(it.get("msg_id") or "") for it in chunk])
    log.info("conversation topics: %d topic(s) touched from %d message(s)", touched, len(items))
    return touched


async def _classify_and_store(agent: Any, items: list[dict[str, Any]]) -> int:
    """One LLM classification call over ``items`` → upsert topics + store excerpts.
    Shared by the live pass and the backfill. Returns topics touched."""
    from captain_claw.config import get_config
    from captain_claw.llm import Message

    tc = get_config().conversation_topics
    mgr = get_topics_manager()
    # The classifier sees the existing topics most relevant to this batch plus
    # the most recent ones, with whole summaries, so it reuses a topic instead
    # of minting a near-duplicate (the prompt asks it to copy a matching label
    # verbatim; upsert_topic then dedups by slug).
    batch_block = "\n".join(f"[{i}] ({it['role']}) {it['excerpt'][:300]}" for i, it in enumerate(items))
    existing = await _classifier_topics(agent, mgr, " ".join(it["excerpt"][:300] for it in items))
    existing_block = "\n".join(
        f"- {t['label']}: {str(t.get('summary') or '')[:_CLASSIFIER_SUMMARY_CHARS]}" for t in existing
    ) or "(none yet)"
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


async def _classifier_topics(agent: Any, mgr: ConversationTopicsManager,
                             batch_text: str) -> list[dict[str, Any]]:
    """Existing topics shown to the classifier: the ``_CLASSIFIER_RELEVANT``
    best matches for the batch, then the ``_CLASSIFIER_RECENT`` most recent
    not already listed. Hidden topics are left out."""
    import asyncio

    embedder = topic_embedder(agent)
    try:
        relevant = await asyncio.to_thread(
            mgr.rank_topics, batch_text, _CLASSIFIER_RELEVANT,
            embedder=embedder, max_terms=_CLASSIFIER_TERMS,
        )
    except Exception as exc:
        log.debug("topic ranking for the classifier failed: %s", exc)
        relevant = []
    have = {t["id"] for t in relevant}
    recent = [t for t in mgr.recent_topics(_CLASSIFIER_RECENT + len(have)) if t["id"] not in have]
    return relevant + recent[:_CLASSIFIER_RECENT]


def refresh_topic(agent: Any, topic_id: str) -> dict[str, Any]:
    """Re-pull the FULL text for a topic's messages from the live session (by
    msg_id) and update the stored excerpts — fixes topics captured under an older
    truncation cap. Messages no longer in the session keep their stored text.
    The text follows the ingest rules (no surface rules block on user rows)."""
    mgr = get_topics_manager()
    session_map: dict[str, str] = {}
    if agent.session and agent.session.messages:
        from captain_claw import member_privacy, msg_origin

        for m in agent.session.messages:
            if member_privacy.is_private(m):
                continue                 # PR D: members' private data
            mid = str(m.get("message_id") or "")
            if m.get("role") == "user":
                content = msg_origin.model_view_text(m).strip()
            else:
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
    for item in _conversation_items(agent, list(agent.session.messages)):
        if item["msg_id"] in done_ids:
            continue
        if cutoff is not None:
            try:
                if datetime.fromisoformat(item["ts"]) < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        pending.append(item)

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


async def backfill_topics_from_history(agent: Any, limit: int = 200) -> dict[str, Any]:
    """Classify messages from the frozen ``session_history`` transcript archive
    into topics — so topics surface conversations that were compacted out of the
    live session and no longer live in ``agent.session.messages``.

    Mirrors :func:`backfill_topics`: one LLM batch per call, ``remaining`` drives
    the UI to auto-continue until it hits 0. Each archived line gets a stable
    synthetic ``msg_id`` (``{history_id}:{line}``) so re-runs dedup against
    already-classified/seen ids and never duplicate a topic."""
    from captain_claw.config import get_config

    tc = get_config().conversation_topics
    memory = getattr(agent, "memory", None)
    if memory is None or getattr(memory, "semantic", None) is None:
        return {"ok": True, "classified": 0, "topics_touched": 0, "remaining": 0,
                "error": "history memory not available"}
    snaps = memory.list_history(limit=max(1, int(limit)))
    if not snaps:
        return {"ok": True, "classified": 0, "topics_touched": 0, "remaining": 0}

    mgr = get_topics_manager()
    done_ids = mgr.classified_msg_ids() | mgr.seen_msg_ids()
    batch = max(5, int(tc.max_messages_per_pass))

    pending: list[dict[str, Any]] = []
    for snap in snaps:
        hid = str(snap.get("history_id") or "")
        if not hid:
            continue
        full = memory.get_history(hid)
        if not full:
            continue
        created = str(full.get("created_at") or _utcnow())
        for n, raw_line in enumerate(str(full.get("text") or "").splitlines()):
            m = re.match(r"^\[(\w+)\]\s*(.*)$", raw_line.strip())
            if not m:
                continue
            role_raw, content = m.group(1).lower(), m.group(2).strip()
            if role_raw not in ("user", "assistant") or not content:
                continue
            mid = f"{hid}:{n}"
            if mid in done_ids:
                continue
            pending.append({
                "role": "user" if role_raw == "user" else "agent",
                "channel": "history",
                "excerpt": content[:_MAX_EXCERPT_CHARS],
                "msg_id": mid,
                "ts": created,
            })
            if len(pending) >= batch:
                break
        if len(pending) >= batch:
            break

    if not pending:
        return {"ok": True, "classified": 0, "topics_touched": 0, "remaining": 0}
    chunk = pending[:batch]
    try:
        touched = await _classify_and_store(agent, chunk)
    except Exception as exc:
        log.warning("topic history-backfill classify failed: %s", exc)
        return {"ok": False, "error": f"classification failed: {exc}"[:300],
                "classified": 0, "topics_touched": 0, "remaining": len(pending)}
    # Mark the whole chunk attempted so unclassified lines aren't reprocessed.
    mgr.mark_seen([str(it.get("msg_id") or "") for it in chunk])
    # A full batch means more archive likely remains — keep the UI auto-continuing
    # (the next pass re-scans and skips now-seen ids); a short batch means done.
    remaining = 1 if len(chunk) >= batch else 0
    log.info("topic history-backfill: classified %d message(s) into %d topic touch(es)",
             len(chunk), touched)
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
