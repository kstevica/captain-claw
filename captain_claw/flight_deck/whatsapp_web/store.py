"""One linked account's messages: ``<user dir>/store.sqlite3``, written only by
the service.

What other people wrote is sealed with :mod:`fd_crypto` (the Telegram inbox's
key), so the file alone doesn't give it away: message text and captions, push
names, contact and chat names, media details (they hold the media key) and
reaction emoji. JIDs stay plain so chats and senders can be indexed — they are
phone numbers, but whatsmeow's ``session.sqlite3`` beside this file holds every
one of them in plain text anyway.

Ingest rules (docs/whatsapp-web-plan.md, from the PoC):

* a message is upserted on ``(chat, id)`` — a second event with the same id
  fills in, never adds a row;
* only :func:`parse.classify`'s ``message`` rows are stored; ``skip`` never writes;
* an edit replaces the text and sets ``edited_at``; a revoke blanks the text
  and media and sets ``deleted``;
* reactions live in their own table, keyed by the message they react to;
* a chat filed under a LID (``…@lid``) is folded into its phone-number chat as
  soon as the mapping is known, and stays folded for later messages.

Every method is synchronous and serialised by one lock; the engine calls them
in a worker thread.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import fd_crypto

# A search reads (and opens) at most this many of the newest messages — the
# whole chat when one is named.
SEARCH_SCAN = 20000
CHAT_SCAN = 100000
# An agent's first "what's new" covers this much.
FIRST_LOOK_SECONDS = 86400
RETENTION_CHOICES = (7, 30, 90, 365)
# Sending (Phase 3): off until the user turns it on, only to whitelisted chats.
DEFAULT_SIGNATURE = "— sent by my assistant"
DEFAULT_DAILY_LIMIT = 30
MAX_DAILY_LIMIT = 200
# Auto-replies: trigger words, each a word or a short phrase.
MAX_TRIGGERS = 10
MAX_TRIGGER_CHARS = 40
# Auto-reply decisions kept for the card (the newest; older ones are dropped).
AUTOREPLY_LOG_KEEP = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    jid TEXT PRIMARY KEY,
    is_group INTEGER NOT NULL DEFAULT 0,
    name TEXT NOT NULL DEFAULT '',
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    mute_until INTEGER NOT NULL DEFAULT 0,
    unread INTEGER NOT NULL DEFAULT 0,
    last_ts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    rowid INTEGER PRIMARY KEY,
    chat_jid TEXT NOT NULL,
    id TEXT NOT NULL,
    sender_jid TEXT NOT NULL DEFAULT '',
    from_me INTEGER NOT NULL DEFAULT 0,
    ts INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'text',
    body TEXT NOT NULL DEFAULT '',
    media TEXT,
    reply_to TEXT NOT NULL DEFAULT '',
    edited_at INTEGER,
    deleted INTEGER NOT NULL DEFAULT 0,
    live INTEGER NOT NULL DEFAULT 0,
    UNIQUE (chat_jid, id)
);
CREATE INDEX IF NOT EXISTS messages_chat_ts ON messages (chat_jid, ts);
CREATE INDEX IF NOT EXISTS messages_ts ON messages (ts);
CREATE INDEX IF NOT EXISTS messages_live ON messages (live, rowid);
CREATE TABLE IF NOT EXISTS reactions (
    chat_jid TEXT NOT NULL,
    target_id TEXT NOT NULL,
    sender_jid TEXT NOT NULL,
    emoji TEXT NOT NULL,
    ts INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (chat_jid, target_id, sender_jid)
);
CREATE TABLE IF NOT EXISTS contacts (
    jid TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    push_name TEXT NOT NULL DEFAULT '',
    business_name TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS lid_map (lid TEXT PRIMARY KEY, pn TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cursors (agent TEXT NOT NULL, key TEXT NOT NULL, value INTEGER NOT NULL,
                                    PRIMARY KEY (agent, key));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sends (
    rowid INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    chat_jid TEXT NOT NULL,
    agent TEXT NOT NULL DEFAULT '',
    text_hash TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    wa_id TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS sends_ts ON sends (ts);
-- Each auto-reply decision Flight Deck made (whatsapp_web_autoreply), the
-- details sealed: the messages, what the model thought and said.
CREATE TABLE IF NOT EXISTS autoreply_log (
    rowid INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    chat_jid TEXT NOT NULL,
    outcome TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS sends_chat ON sends (chat_jid, ts);
"""


class ChatNotFound(LookupError):
    def __init__(self, message: str, candidates: list[dict] | None = None) -> None:
        super().__init__(message)
        self.candidates = candidates or []


def _seal(value: Any) -> str:
    return fd_crypto.seal(value) if value not in (None, "", {}) else ""


def _unseal(token: str | None) -> Any:
    return fd_crypto.unseal(token) if token else None


def _text(token: str | None) -> str:
    value = _unseal(token)
    return value if isinstance(value, str) else ""


def number_of(jid: str) -> str:
    user, _, server = (jid or "").partition("@")
    return "+" + user if server == "s.whatsapp.net" and user.isdigit() else ""


def pick_chat(all_chats: list[dict[str, Any]], ref: str) -> dict[str, Any]:
    """A chat by JID, phone number, exact name, or a unique part of its name."""
    ref = str(ref or "").strip()
    if not ref:
        raise ChatNotFound("name a chat")
    digits = "".join(ch for ch in ref if ch.isdigit())
    for c in all_chats:
        if c["jid"] == ref or (digits and len(digits) >= 7 and c["number"] == "+" + digits.lstrip("0")):
            return c
    low = ref.casefold().lstrip("@")
    exact = [c for c in all_chats if c["title"].casefold() == low]
    if len(exact) == 1:
        return exact[0]
    part = exact or [c for c in all_chats if low in c["title"].casefold()]
    if len(part) == 1:
        return part[0]
    if part:
        raise ChatNotFound(f"several chats match {ref!r} — pick one by its id", part[:10])
    raise ChatNotFound(f"no chat matches {ref!r}")


# Reading several stores as one: the linked account's and what's kept from earlier
# links (kept_history). Lists come in that order, and the earlier list wins.


def merge_chats(lists: list[list[dict[str, Any]]], *, live: bool = True) -> list[dict[str, Any]]:
    """One list, latest activity first: a chat in several stores is one chat, its
    counts added up (with *live*, the first list is the linked account's and
    ``kept_only`` marks chats it doesn't have: nothing can be sent there)."""
    merged: dict[str, dict[str, Any]] = {}
    for index, chats in enumerate(lists):
        for c in chats:
            have = merged.get(c["jid"])
            if have is None:
                merged[c["jid"]] = {**c, "kept_only": not (live and index == 0)}
                continue
            have["messages"] += c["messages"]
            have["last_ts"] = max(have["last_ts"] or 0, c["last_ts"] or 0) or None
            if have["title"] in (have["jid"], have["number"]):
                have["title"] = c["title"]  # the first store only had the number
    return sorted(merged.values(), key=lambda c: c["last_ts"] or 0, reverse=True)


def merge_messages(lists: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    """Newest first; a message in several stores (a group both numbers were in,
    a number linked again) once."""
    if len(lists) == 1:
        return lists[0][:limit]
    seen: set[tuple[str, str]] = set()
    out = []
    for items in lists:
        for m in items:
            key = (m["chat_jid"], m["id"])
            if key not in seen:
                seen.add(key)
                out.append(m)
    out.sort(key=lambda m: m["ts"], reverse=True)
    return out[:limit]


def merge_contacts(lists: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for people in lists:
        for p in people:
            have = merged.get(p["jid"])
            if have is None:
                merged[p["jid"]] = dict(p)
                continue
            have["messages"] += p["messages"]
            have["groups"] = sorted({*have["groups"], *p["groups"]})[:10]
            for key in ("name", "push_name", "business_name", "number", "lid"):
                have[key] = have[key] or p[key]
    out = sorted(merged.values(), key=lambda c: (-c["messages"], c["name"] or c["push_name"] or c["jid"]))
    return out[: max(1, min(int(limit or 20), 100))]


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._lids: dict[str, str] = {r["lid"]: r["pn"] for r in self._db.execute("SELECT lid, pn FROM lid_map")}
        # Unsealed names, rebuilt after any write to contacts; push names already
        # stored (history repeats the sender's on every message).
        self._names_cache: dict[str, dict[str, str]] | None = None
        self._push_seen: dict[str, str] = {}

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── helpers ─────────────────────────────────────────────────────

    def _canon(self, jid: str) -> str:
        return self._lids.get(jid, jid) if jid.endswith("@lid") else jid

    def _tx(self):
        return _Tx(self._db)

    # ── ingest ──────────────────────────────────────────────────────

    def ingest(self, rows: list[dict[str, Any]], *, live: bool) -> dict[str, int]:
        """Apply parsed rows (:mod:`.parse`); counts of what each op did."""
        counts = {"message": 0, "edit": 0, "revoke": 0, "react": 0, "skip": 0}
        with self._lock, self._tx() as db:
            for r in rows:
                op = r.get("op") or "skip"
                if op == "skip" or not r.get("chat"):
                    counts["skip"] += 1
                    continue
                chat = self._canon(str(r["chat"]))
                sender = self._canon(str(r.get("sender") or ""))
                if op == "message":
                    if not r.get("id"):
                        continue
                    body = _seal({"t": r.get("text") or "", "p": r.get("push_name") or ""})
                    db.execute(
                        "INSERT INTO messages (chat_jid, id, sender_jid, from_me, ts, kind, body, media,"
                        " reply_to, live) VALUES (?,?,?,?,?,?,?,?,?,?)"
                        " ON CONFLICT (chat_jid, id) DO UPDATE SET"
                        " sender_jid = excluded.sender_jid, from_me = excluded.from_me,"
                        " ts = CASE WHEN excluded.ts > 0 THEN excluded.ts ELSE messages.ts END,"
                        " kind = excluded.kind,"
                        " body = CASE WHEN messages.deleted THEN messages.body ELSE excluded.body END,"
                        " media = CASE WHEN messages.deleted THEN NULL ELSE COALESCE(excluded.media, messages.media) END,"
                        " reply_to = excluded.reply_to, live = MAX(messages.live, excluded.live)",
                        (chat, str(r["id"]), sender, int(bool(r.get("from_me"))), int(r.get("ts") or 0),
                         str(r.get("kind") or "text"), body,
                         _seal(r["media"]) if r.get("media") else None,
                         str(r.get("reply_to") or ""), int(live)),
                    )
                    self._touch_chat(db, chat, bool(r.get("is_group")), int(r.get("ts") or 0))
                    if r.get("push_name") and not r.get("from_me") and sender:
                        self._push_name(db, sender, str(r["push_name"]))
                elif op == "edit":
                    old = db.execute("SELECT body FROM messages WHERE chat_jid=? AND id=?",
                                     (chat, str(r.get("target") or ""))).fetchone()
                    if old is None:
                        counts["skip"] += 1
                        continue
                    prev = _unseal(old["body"])
                    prev = prev if isinstance(prev, dict) else {}
                    db.execute("UPDATE messages SET body=?, edited_at=? WHERE chat_jid=? AND id=? AND deleted=0",
                               (_seal({"t": r.get("text") or "", "p": prev.get("p", "")}),
                                int(r.get("ts") or time.time()), chat, str(r.get("target"))))
                elif op == "revoke":
                    db.execute("UPDATE messages SET body='', media=NULL, deleted=1 WHERE chat_jid=? AND id=?",
                               (chat, str(r.get("target") or "")))
                elif op == "react":
                    target = str(r.get("target") or "")
                    who = "me" if r.get("from_me") else sender
                    if not target or not who:
                        continue
                    if r.get("emoji"):
                        db.execute(
                            "INSERT OR REPLACE INTO reactions (chat_jid, target_id, sender_jid, emoji, ts)"
                            " VALUES (?,?,?,?,?)",
                            (chat, target, who, _seal(str(r["emoji"])), int(r.get("react_ts") or r.get("ts") or 0)))
                    else:
                        db.execute("DELETE FROM reactions WHERE chat_jid=? AND target_id=? AND sender_jid=?",
                                   (chat, target, who))
                counts[op] = counts.get(op, 0) + 1
        return counts

    def _touch_chat(self, db: sqlite3.Connection, jid: str, is_group: bool, ts: int) -> None:
        db.execute(
            "INSERT INTO chats (jid, is_group, last_ts) VALUES (?,?,?)"
            " ON CONFLICT (jid) DO UPDATE SET last_ts = MAX(chats.last_ts, excluded.last_ts),"
            " is_group = MAX(chats.is_group, excluded.is_group)",
            (jid, int(is_group or jid.endswith("@g.us")), ts))

    def _push_name(self, db: sqlite3.Connection, jid: str, name: str) -> None:
        if self._push_seen.get(jid) == name:
            return
        self._push_seen[jid] = name
        self._names_cache = None
        db.execute(
            "INSERT INTO contacts (jid, push_name) VALUES (?,?)"
            " ON CONFLICT (jid) DO UPDATE SET push_name = excluded.push_name",
            (jid, _seal(name)))

    def put_chats(self, chats: list[dict[str, Any]]) -> None:
        """History's chat details (name, archived, pinned …)."""
        with self._lock, self._tx() as db:
            for c in chats:
                jid = self._canon(str(c.get("jid") or ""))
                if not jid:
                    continue
                self._touch_chat(db, jid, bool(c.get("is_group")), int(c.get("last_ts") or 0))
                db.execute(
                    "UPDATE chats SET archived=?, pinned=?, mute_until=?, unread=?,"
                    " name = CASE WHEN ? != '' THEN ? ELSE name END WHERE jid=?",
                    (int(bool(c.get("archived"))), int(bool(c.get("pinned"))), int(c.get("mute_until") or 0),
                     int(c.get("unread") or 0), c.get("name") or "", _seal(c.get("name") or ""), jid))

    def put_names(self, pushnames: list[dict[str, str]]) -> None:
        with self._lock, self._tx() as db:
            for p in pushnames:
                if p.get("jid") and p.get("name"):
                    self._push_name(db, self._canon(p["jid"]), p["name"])

    def put_contacts(self, contacts: list[dict[str, str]]) -> None:
        """The phone's address book (whatsmeow's contact store)."""
        with self._lock, self._tx() as db:
            self._names_cache = None
            for c in contacts:
                jid = self._canon(str(c.get("jid") or ""))
                name = str(c.get("full_name") or c.get("first_name") or "")
                if not jid:
                    continue
                db.execute(
                    "INSERT INTO contacts (jid, name, push_name, business_name) VALUES (?,?,?,?)"
                    " ON CONFLICT (jid) DO UPDATE SET"
                    " name = CASE WHEN excluded.name != '' THEN excluded.name ELSE contacts.name END,"
                    " push_name = CASE WHEN excluded.push_name != '' THEN excluded.push_name ELSE contacts.push_name END,"
                    " business_name = CASE WHEN excluded.business_name != '' THEN excluded.business_name"
                    " ELSE contacts.business_name END",
                    (jid, _seal(name), _seal(c.get("push_name") or ""), _seal(c.get("business_name") or "")))

    def put_groups(self, groups: list[dict[str, str]]) -> None:
        with self._lock, self._tx() as db:
            for g in groups:
                jid = str(g.get("jid") or "")
                if jid and g.get("name"):
                    self._touch_chat(db, jid, True, 0)
                    db.execute("UPDATE chats SET name=? WHERE jid=?", (_seal(g["name"]), jid))

    def put_lid_map(self, pairs: list[dict[str, str]]) -> int:
        """Remember LID ↔ phone pairs and fold every LID chat already kept into
        its phone-number chat. Returns how many chats were folded."""
        folded = 0
        with self._lock, self._tx() as db:
            for p in pairs:
                lid, pn = str(p.get("lid") or ""), str(p.get("pn") or "")
                if not (lid.endswith("@lid") and pn) or self._lids.get(lid) == pn:
                    continue
                self._lids[lid] = pn
                db.execute("INSERT OR REPLACE INTO lid_map (lid, pn) VALUES (?,?)", (lid, pn))
                if db.execute("SELECT 1 FROM chats WHERE jid=?", (lid,)).fetchone():
                    folded += 1
                    old = db.execute("SELECT * FROM chats WHERE jid=?", (lid,)).fetchone()
                    self._touch_chat(db, pn, bool(old["is_group"]), int(old["last_ts"]))
                    if old["name"]:
                        db.execute("UPDATE chats SET name = CASE WHEN name = '' THEN ? ELSE name END WHERE jid=?",
                                   (old["name"], pn))
                    db.execute("DELETE FROM chats WHERE jid=?", (lid,))
                db.execute("UPDATE OR IGNORE messages SET chat_jid=? WHERE chat_jid=?", (pn, lid))
                db.execute("DELETE FROM messages WHERE chat_jid=?", (lid,))  # duplicates of kept ones
                db.execute("UPDATE OR IGNORE messages SET sender_jid=? WHERE sender_jid=?", (pn, lid))
                db.execute("UPDATE OR IGNORE reactions SET chat_jid=? WHERE chat_jid=?", (pn, lid))
                db.execute("UPDATE OR IGNORE reactions SET sender_jid=? WHERE sender_jid=?", (pn, lid))
                db.execute("DELETE FROM reactions WHERE chat_jid=? OR sender_jid=?", (lid, lid))
                if db.execute("SELECT 1 FROM contacts WHERE jid=?", (lid,)).fetchone():
                    self._names_cache = None
                    self._push_seen.pop(lid, None)
                    db.execute("INSERT OR IGNORE INTO contacts (jid) VALUES (?)", (pn,))
                    db.execute(
                        "UPDATE contacts SET push_name = CASE WHEN push_name = '' THEN"
                        " (SELECT push_name FROM contacts WHERE jid=?) ELSE push_name END WHERE jid=?", (lid, pn))
                    db.execute("DELETE FROM contacts WHERE jid=?", (lid,))
        return folded

    # ── meta / retention ────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))

    def retention_days(self) -> int | None:
        raw = self.get_meta("retention_days")
        try:
            days = int(raw) if raw else None
        except ValueError:
            days = None
        return days if days in RETENTION_CHOICES else None

    def purge(self, *, now: float | None = None) -> int:
        """Drop messages older than the retention choice (keep everything: none)."""
        days = self.retention_days()
        if days is None:
            return 0
        cutoff = int((now or time.time()) - days * 86400)
        with self._lock, self._tx() as db:
            n = db.execute("DELETE FROM messages WHERE ts > 0 AND ts < ?", (cutoff,)).rowcount
            db.execute("DELETE FROM reactions WHERE ts > 0 AND ts < ?", (cutoff,))
        return n

    # ── reading ─────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) n, MIN(ts) lo, MAX(ts) hi FROM messages WHERE ts > 0").fetchone()
            chats = self._db.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
        return {"messages": int(row["n"] or 0), "chats": int(chats or 0),
                "oldest": int(row["lo"] or 0) or None, "newest": int(row["hi"] or 0) or None}

    def _names(self) -> dict[str, dict[str, str]]:
        if self._names_cache is None:
            out = {}
            for r in self._db.execute("SELECT jid, name, push_name, business_name FROM contacts"):
                out[r["jid"]] = {"name": _text(r["name"]), "push": _text(r["push_name"]),
                                 "business": _text(r["business_name"])}
            self._names_cache = out
        return self._names_cache

    @staticmethod
    def _person(jid: str, names: dict[str, dict[str, str]], push: str = "") -> str:
        n = names.get(jid) or {}
        return n.get("name") or push or n.get("push") or n.get("business") or number_of(jid) or jid

    def chats(self) -> list[dict[str, Any]]:
        """Every chat, latest activity first."""
        with self._lock:
            names = self._names()
            counts = {r["chat_jid"]: (int(r["n"]), int(r["hi"] or 0)) for r in self._db.execute(
                "SELECT chat_jid, COUNT(*) n, MAX(ts) hi FROM messages GROUP BY chat_jid")}
            rows = self._db.execute("SELECT * FROM chats").fetchall()
        out = []
        for r in rows:
            jid = r["jid"]
            n, hi = counts.get(jid, (0, 0))
            title = _text(r["name"]) or (self._person(jid, names) if not r["is_group"] else jid)
            out.append({
                "jid": jid, "title": title, "kind": "group" if r["is_group"] else "direct",
                "number": number_of(jid), "messages": n, "last_ts": max(hi, int(r["last_ts"] or 0)) or None,
                "archived": bool(r["archived"]), "pinned": bool(r["pinned"]), "unread": int(r["unread"] or 0),
            })
        out.sort(key=lambda c: (c["last_ts"] or 0), reverse=True)
        return out

    def find_chat(self, ref: str) -> dict[str, Any]:
        """A chat by JID, phone number, exact name, or a unique part of its name."""
        return pick_chat(self.chats(), ref)

    def messages(self, *, chat_jid: str | None = None, since: int | None = None, until: int | None = None,
                 query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        """Messages, newest first; ``query`` keeps those whose text, sender or
        chat name contains every word."""
        limit = max(1, min(int(limit or 50), 500))
        terms = [t for t in str(query or "").casefold().split() if t]
        where, args = ["1=1"], []
        if chat_jid:
            where.append("chat_jid = ?")
            args.append(chat_jid)
        if since is not None:
            where.append("ts >= ?")
            args.append(int(since))
        if until is not None:
            where.append("ts < ?")
            args.append(int(until))
        scan = (CHAT_SCAN if chat_jid else SEARCH_SCAN) if terms else limit
        with self._lock:
            names = self._names()
            titles = {r["jid"]: (_text(r["name"]), bool(r["is_group"]))
                      for r in self._db.execute("SELECT jid, name, is_group FROM chats")}
            rows = self._db.execute(
                f"SELECT * FROM messages WHERE {' AND '.join(where)} ORDER BY ts DESC, rowid DESC LIMIT ?",
                (*args, scan)).fetchall()
            reactions: dict[tuple[str, str], list[str]] = {}
            agent_ids = {r["wa_id"] for r in self._db.execute(
                "SELECT wa_id FROM sends WHERE status='sent' AND wa_id != ''")} if rows else set()
            if rows:
                keys = {(r["chat_jid"], r["id"]) for r in rows}
                for rr in self._db.execute(
                        "SELECT chat_jid, target_id, sender_jid, emoji FROM reactions WHERE chat_jid IN (%s)"
                        % ",".join("?" * len({k[0] for k in keys})), tuple({k[0] for k in keys})):
                    if (rr["chat_jid"], rr["target_id"]) in keys:
                        who = "you" if rr["sender_jid"] == "me" else self._person(rr["sender_jid"], names)
                        reactions.setdefault((rr["chat_jid"], rr["target_id"]), []).append(
                            f"{_text(rr['emoji'])} {who}")
        out: list[dict[str, Any]] = []
        for r in rows:
            body = _unseal(r["body"]) if r["body"] else None
            body = body if isinstance(body, dict) else {}
            media = _unseal(r["media"]) if r["media"] else None
            media = media if isinstance(media, dict) else {}
            title, is_group = titles.get(r["chat_jid"], ("", r["chat_jid"].endswith("@g.us")))
            chat_title = title or (r["chat_jid"] if is_group else self._person(r["chat_jid"], names))
            item = {
                "rowid": int(r["rowid"]),
                "chat_jid": r["chat_jid"],
                "chat": chat_title,
                "kind_of_chat": "group" if is_group else "direct",
                "id": r["id"],
                "ts": int(r["ts"] or 0),
                "from_me": bool(r["from_me"]),
                "sender": "you" if r["from_me"] else self._person(r["sender_jid"], names, str(body.get("p") or "")),
                "kind": r["kind"],
                "text": "" if r["deleted"] else str(body.get("t") or ""),
                "media": None if r["deleted"] or not media else {
                    k: media.get(k) for k in ("mimetype", "size", "seconds", "file_name")},
                "reply_to": r["reply_to"] or "",
                "edited": bool(r["edited_at"]),
                "deleted": bool(r["deleted"]),
                "reactions": reactions.get((r["chat_jid"], r["id"]), []),
                "unreadable": bool(r["body"]) and not body,
                # Sent from the user's account by one of their agents (auto-reply, send).
                "by_agent": bool(r["from_me"]) and r["id"] in agent_ids,
            }
            if terms:
                hay = " ".join((item["text"], item["sender"], item["chat"],
                                (item["media"] or {}).get("file_name") or "")).casefold()
                if not all(t in hay for t in terms):
                    continue
            out.append(item)
            if len(out) >= limit:
                break
        return out

    def media(self, chat_jid: str, message_id: str) -> dict[str, Any] | None:
        """One message's attachment details, unsealed (with what a download
        needs), or None when there's no such message."""
        with self._lock:
            r = self._db.execute(
                "SELECT id, kind, ts, from_me, sender_jid, deleted, media FROM messages WHERE chat_jid=? AND id=?",
                (chat_jid, str(message_id or "").strip().lstrip("#"))).fetchone()
        if r is None:
            return None
        media = _unseal(r["media"]) if r["media"] else None
        return {"id": r["id"], "kind": r["kind"], "ts": int(r["ts"] or 0), "from_me": bool(r["from_me"]),
                "sender_jid": r["sender_jid"], "deleted": bool(r["deleted"]),
                "media": media if isinstance(media, dict) else None}

    def contacts(self, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
        """People (address book, push names, everyone who wrote) matching *query*."""
        q = str(query or "").casefold().strip()
        digits = "".join(ch for ch in q if ch.isdigit())
        with self._lock:
            names = self._names()
            senders = {r["sender_jid"]: int(r["n"]) for r in self._db.execute(
                "SELECT sender_jid, COUNT(*) n FROM messages WHERE from_me=0 AND sender_jid != ''"
                " GROUP BY sender_jid")}
            groups: dict[str, set[str]] = {}
            for r in self._db.execute(
                    "SELECT DISTINCT sender_jid, chat_jid FROM messages WHERE chat_jid LIKE '%@g.us'"):
                groups.setdefault(r["sender_jid"], set()).add(r["chat_jid"])
            titles = {r["jid"]: _text(r["name"]) for r in self._db.execute(
                "SELECT jid, name FROM chats WHERE is_group=1")}
            lids = {pn: lid for lid, pn in self._lids.items()}
        out = []
        for jid in set(names) | set(senders):
            n = names.get(jid) or {}
            label = self._person(jid, names)
            hay = " ".join((label, n.get("push", ""), n.get("business", ""), jid)).casefold()
            if q and q not in hay and not (digits and len(digits) >= 4 and digits in jid):
                continue
            out.append({
                "jid": jid, "name": n.get("name", ""), "push_name": n.get("push", ""),
                "business_name": n.get("business", ""), "number": number_of(jid), "lid": lids.get(jid, ""),
                "messages": senders.get(jid, 0),
                "groups": sorted(t for t in (titles.get(g, "") for g in groups.get(jid, ())) if t)[:10],
            })
        out.sort(key=lambda c: (-c["messages"], c["name"] or c["push_name"] or c["jid"]))
        return out[: max(1, min(int(limit or 20), 100))]

    # ── sending: the policy and the log ─────────────────────────────

    def send_policy(self) -> dict[str, Any]:
        """The user's sending rules: off, nobody whitelisted, 30 a day and the
        signature on until they say otherwise."""
        with self._lock:
            meta = {r["key"]: r["value"] for r in self._db.execute(
                "SELECT key, value FROM meta WHERE key LIKE 'send_%' OR key LIKE 'signature%'")}
        try:
            allowed = json.loads(meta.get("send_allowed") or "[]")
        except ValueError:
            allowed = []
        try:
            limit = int(meta.get("send_daily_limit") or DEFAULT_DAILY_LIMIT)
        except ValueError:
            limit = DEFAULT_DAILY_LIMIT
        return {
            "enabled": meta.get("send_enabled") == "1",
            "allowed": [str(j) for j in allowed if isinstance(j, str)] if isinstance(allowed, list) else [],
            "daily_limit": max(1, min(limit, MAX_DAILY_LIMIT)),
            "signature": meta.get("signature", DEFAULT_SIGNATURE),
            "signature_on": meta.get("signature_on", "1") == "1",
        }

    def set_send_policy(self, policy: dict[str, Any]) -> None:
        values = {
            "send_enabled": "1" if policy.get("enabled") else "0",
            "send_allowed": json.dumps(sorted(set(policy.get("allowed") or []))),
            "send_daily_limit": str(int(policy.get("daily_limit") or DEFAULT_DAILY_LIMIT)),
            "signature": str(policy.get("signature") if policy.get("signature") is not None else DEFAULT_SIGNATURE),
            "signature_on": "1" if policy.get("signature_on", True) else "0",
        }
        with self._lock, self._tx() as db:
            for key, value in values.items():
                db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))

    @staticmethod
    def text_hash(chat_jid: str, text: str) -> str:
        """What the duplicate check compares: the chat and the text, folded."""
        folded = " ".join(str(text or "").casefold().split())
        return hashlib.sha256(f"{chat_jid}\n{folded}".encode()).hexdigest()

    def start_send(self, chat_jid: str, agent: str, text: str, *, now: float) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO sends (ts, chat_jid, agent, text_hash, body, status) VALUES (?,?,?,?,?, 'sending')",
                (int(now), chat_jid, agent, self.text_hash(chat_jid, text), _seal({"t": text})))
            return int(cur.lastrowid)

    def finish_send(self, rowid: int, status: str, *, wa_id: str = "", error: str = "") -> None:
        with self._lock:
            self._db.execute("UPDATE sends SET status=?, wa_id=?, error=? WHERE rowid=?",
                             (status, wa_id, error[:300], rowid))

    def send_counts(self, chat_jid: str, *, now: float) -> dict[str, int]:
        """Sends that count against the limits (sent, or maybe sent): the last
        24 hours overall, the last minute in this chat."""
        with self._lock:
            day = self._db.execute(
                "SELECT COUNT(*) FROM sends WHERE ts >= ? AND status IN ('sent', 'unknown', 'sending')",
                (int(now - 86400),)).fetchone()[0]
            minute = self._db.execute(
                "SELECT COUNT(*) FROM sends WHERE chat_jid=? AND ts >= ? AND status IN ('sent', 'unknown', 'sending')",
                (chat_jid, int(now - 60))).fetchone()[0]
        return {"day": int(day), "minute": int(minute)}

    def recent_same(self, chat_jid: str, text: str, *, since: float) -> str | None:
        """The status of the same text sent to this chat since *since*, if any."""
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM sends WHERE chat_jid=? AND text_hash=? AND ts >= ?"
                " AND status IN ('sent', 'unknown', 'sending') ORDER BY rowid DESC LIMIT 1",
                (chat_jid, self.text_hash(chat_jid, text), int(since))).fetchone()
        return row["status"] if row else None

    def sends(self, limit: int = 20) -> list[dict[str, Any]]:
        """The latest sends, newest first (for the card)."""
        with self._lock:
            rows = self._db.execute("SELECT * FROM sends ORDER BY rowid DESC LIMIT ?",
                                    (max(1, min(int(limit or 20), 100)),)).fetchall()
            titles = {r["jid"]: _text(r["name"]) for r in self._db.execute("SELECT jid, name FROM chats")}
            names = self._names()
        out = []
        for r in rows:
            body = _unseal(r["body"]) if r["body"] else None
            body = body if isinstance(body, dict) else {}
            jid = r["chat_jid"]
            out.append({
                "ts": int(r["ts"]), "chat_jid": jid,
                "chat": titles.get(jid) or self._person(jid, names), "agent": r["agent"],
                "status": r["status"], "error": r["error"], "text": str(body.get("t") or ""),
                "wa_id": r["wa_id"],
            })
        return out

    def add_autoreply_log(self, chat_jid: str, outcome: str, details: dict[str, Any], *, now: float) -> None:
        with self._lock, self._tx() as db:
            db.execute("INSERT INTO autoreply_log (ts, chat_jid, outcome, body) VALUES (?,?,?,?)",
                       (int(now), chat_jid, outcome[:80], _seal(details)))
            db.execute("DELETE FROM autoreply_log WHERE rowid <= (SELECT MAX(rowid) FROM autoreply_log) - ?",
                       (AUTOREPLY_LOG_KEEP,))

    def autoreply_log(self, *, limit: int = 50, chat_jid: str = "", ignored: bool = True) -> list[dict[str, Any]]:
        """The latest auto-reply decisions, newest first; without *ignored*,
        not the messages that didn't call the agent."""
        where, args = ["1=1"], []
        if chat_jid:
            where.append("chat_jid=?")
            args.append(chat_jid)
        if not ignored:
            where.append("outcome != 'skip:not addressed'")
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM autoreply_log WHERE {' AND '.join(where)} ORDER BY rowid DESC LIMIT ?",
                (*args, max(1, min(int(limit or 50), 200)))).fetchall()
            titles = {r["jid"]: _text(r["name"]) for r in self._db.execute("SELECT jid, name FROM chats")}
            names = self._names()
        out = []
        for r in rows:
            details = _unseal(r["body"]) if r["body"] else None
            jid = r["chat_jid"]
            out.append({"id": int(r["rowid"]), "ts": int(r["ts"]), "chat_jid": jid,
                        "chat": titles.get(jid) or self._person(jid, names), "outcome": r["outcome"],
                        **(details if isinstance(details, dict) else {})})
        return out

    def autoreply_config(self) -> dict[str, Any]:
        """Which whitelisted chats the agent answers by itself, which agent, and
        the extra words that count as mentioning it (its name always does)."""
        with self._lock:
            meta = {r["key"]: r["value"] for r in self._db.execute(
                "SELECT key, value FROM meta WHERE key LIKE 'autoreply_%'")}

        def as_list(raw: str | None) -> list[str]:
            try:
                value = json.loads(raw or "[]")
            except ValueError:
                return []
            return [str(v) for v in value if isinstance(v, str)] if isinstance(value, list) else []

        return {"chats": as_list(meta.get("autoreply_chats")), "agent": meta.get("autoreply_agent", ""),
                "triggers": as_list(meta.get("autoreply_triggers"))}

    def set_autoreply_config(self, config: dict[str, Any]) -> None:
        values = {
            "autoreply_chats": json.dumps(sorted(set(config.get("chats") or []))),
            "autoreply_agent": str(config.get("agent") or ""),
            "autoreply_triggers": json.dumps(list(config.get("triggers") or [])),
        }
        with self._lock, self._tx() as db:
            for key, value in values.items():
                db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))

    def agent_activity(self, chat_jid: str) -> dict[str, Any]:
        """In one chat: the ids of the messages agents sent, when the last one
        went, and when the user last wrote there themselves (not through an agent)."""
        with self._lock:
            sent = self._db.execute(
                "SELECT wa_id, ts FROM sends WHERE chat_jid=? AND status='sent' AND wa_id != ''",
                (chat_jid,)).fetchall()
            ids = {r["wa_id"] for r in sent}
            last_agent = max((int(r["ts"]) for r in sent), default=0)
            own = self._db.execute(
                "SELECT id, ts FROM messages WHERE chat_jid=? AND from_me=1 ORDER BY ts DESC LIMIT 50",
                (chat_jid,)).fetchall()
        last_owner = next((int(r["ts"]) for r in own if r["id"] not in ids), 0)
        return {"agent_ids": ids, "last_agent_ts": last_agent, "last_owner_ts": last_owner}

    def canon(self, jid: str) -> str:
        """A chat's JID as it's kept (a known LID → its phone-number JID)."""
        return self._canon(jid)

    def chat_title(self, jid: str) -> tuple[str, bool]:
        """(the chat's name as the user sees it, is it a group)."""
        with self._lock:
            row = self._db.execute("SELECT name, is_group FROM chats WHERE jid=?", (jid,)).fetchone()
            is_group = bool(row["is_group"]) if row else jid.endswith("@g.us")
            title = _text(row["name"]) if row else ""
            return (title or (jid if is_group else self._person(jid, self._names()))), is_group

    def sender_name(self, jid: str, push: str = "") -> str:
        with self._lock:
            return self._person(jid, self._names(), push)

    def chat_has_history(self, jid: str) -> bool:
        """An existing conversation: a group we're in, or a chat with messages."""
        with self._lock:
            row = self._db.execute("SELECT is_group FROM chats WHERE jid=?", (jid,)).fetchone()
            if row is None:
                return False
            if row["is_group"]:
                return True
            return self._db.execute("SELECT 1 FROM messages WHERE chat_jid=? LIMIT 1", (jid,)).fetchone() is not None

    # ── what's new, per agent ───────────────────────────────────────

    def news(self, agent: str, *, now: float | None = None) -> dict[str, Any]:
        """What others wrote since this agent last looked, per chat — counts and
        names, never text — and the look itself (the cursor moves). Only
        messages that arrived live count: history from the phone is old news.
        The first look covers the last day."""
        now = now or time.time()
        with self._lock:
            cursors = {r["key"]: int(r["value"]) for r in self._db.execute(
                "SELECT key, value FROM cursors WHERE agent=?", (agent,))}
            first = "" not in cursors
            top = self._db.execute("SELECT COALESCE(MAX(rowid), 0) FROM messages WHERE live=1").fetchone()[0]
            q = ("SELECT rowid, chat_jid FROM messages WHERE live=1 AND from_me=0 AND deleted=0 AND rowid > ?")
            args: list[Any] = [cursors.get("", 0)]
            if first:
                q += " AND ts >= ?"
                args.append(int(now - FIRST_LOOK_SECONDS))
            counts: dict[str, int] = {}
            for r in self._db.execute(q, args):
                if int(r["rowid"]) > cursors.get(r["chat_jid"], 0):
                    counts[r["chat_jid"]] = counts.get(r["chat_jid"], 0) + 1
            self._db.execute("INSERT OR REPLACE INTO cursors (agent, key, value) VALUES (?, '', ?)", (agent, int(top)))
        chats = [{"jid": c["jid"], "title": c["title"], "kind": c["kind"], "new": counts[c["jid"]]}
                 for c in self.chats() if c["jid"] in counts] if counts else []
        return {"chats": chats, "total": sum(counts.values()), "first_look": first}

    def mark_read(self, agent: str, items: list[dict[str, Any]]) -> None:
        """The agent read these: each chat's cursor moves to the newest of them."""
        newest: dict[str, int] = {}
        for item in items:
            chat, rid = str(item.get("chat_jid") or ""), int(item.get("rowid") or 0)
            if chat and rid > newest.get(chat, 0):
                newest[chat] = rid
        with self._lock:
            for chat, rid in newest.items():
                self._db.execute(
                    "INSERT INTO cursors (agent, key, value) VALUES (?,?,?)"
                    " ON CONFLICT (agent, key) DO UPDATE SET value = MAX(cursors.value, excluded.value)",
                    (agent, chat, rid))


class _Tx:
    """BEGIN … COMMIT (ROLLBACK on error) on an autocommit connection."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        self.db.execute("BEGIN")
        return self.db

    def __exit__(self, exc_type, *_: Any) -> None:
        self.db.execute("ROLLBACK" if exc_type else "COMMIT")
