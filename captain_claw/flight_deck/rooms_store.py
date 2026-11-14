"""Data rooms: a virtual data room anonymized into one vault, so Claude can
work on it without the names (``docs/data-room-plan.md``).

What is kept where, on this deck only:

* ``rooms.db`` — the rooms and their files. Anything that can carry a real
  name is sealed with the deck's data key (``fd_crypto``): a room's own name,
  its source, its seed terms and roles, every original file path. The
  codename (what Claude sees) and the anonymized paths are plain.
* ``rooms/<id>/raw/`` — the original files of an uploaded ZIP, each encrypted
  with the room vault's own key. A linked folder is read where it is.
* ``rooms/<id>/clean/`` — the anonymized copies; ``rooms/<id>/text/`` — the
  text Claude reads (also indexed for search in ``room_text``).

A room belongs to one person. Nothing here is shared, and admin act-as is
refused at the routes.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import aiosqlite

from captain_claw.flight_deck import fd_crypto

ROOM_STATES = ("setup", "running", "review", "released")
FILE_STATES = ("new", "done", "blocked", "failed", "unread")
MAX_ROOMS_PER_OWNER = 100
MAX_CODENAME = 80
ORIGINS = ("zip", "folder")

_DDL = """
CREATE TABLE IF NOT EXISTS rooms (
    id           TEXT PRIMARY KEY,
    owner_id     TEXT NOT NULL,
    codename     TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT '',
    vault_id     TEXT NOT NULL,
    vault_name   TEXT NOT NULL,
    state        TEXT NOT NULL DEFAULT 'setup',
    source       TEXT NOT NULL DEFAULT '',
    settings     TEXT NOT NULL DEFAULT '{}',
    seed         TEXT NOT NULL DEFAULT '',
    roles        TEXT NOT NULL DEFAULT '',
    job          TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    released_at  REAL,
    allowed      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_rooms_owner ON rooms(owner_id, updated_at);
CREATE TABLE IF NOT EXISTS room_files (
    room_id     TEXT NOT NULL,
    id          TEXT NOT NULL,
    origin      TEXT NOT NULL,
    path        TEXT NOT NULL,
    path_key    TEXT NOT NULL,
    clean_path  TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT '',
    size        INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT NOT NULL DEFAULT '',
    state       TEXT NOT NULL DEFAULT 'new',
    warnings    TEXT NOT NULL DEFAULT '[]',
    error       TEXT NOT NULL DEFAULT '',
    entities    TEXT NOT NULL DEFAULT '{}',
    text_chars  INTEGER NOT NULL DEFAULT 0,
    released    INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL,
    info        TEXT NOT NULL DEFAULT '{}',
    leftovers   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (room_id, id),
    UNIQUE (room_id, path_key)
);
CREATE TABLE IF NOT EXISTS meta (
    k  TEXT PRIMARY KEY,
    v  TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS room_text USING fts5(
    room_id UNINDEXED, file_id UNINDEXED, page UNINDEXED, body,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS mcp_grants (
    owner_id    TEXT NOT NULL,
    client_id   TEXT NOT NULL,
    rooms       TEXT NOT NULL DEFAULT '[]',
    all_rooms   INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (owner_id, client_id)
);
CREATE TABLE IF NOT EXISTS room_reads (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id      TEXT NOT NULL,
    owner_id     TEXT NOT NULL,
    client_id    TEXT NOT NULL DEFAULT '',
    at           REAL NOT NULL,
    tool         TEXT NOT NULL,
    file_id      TEXT NOT NULL DEFAULT '',
    page         INTEGER NOT NULL DEFAULT 0,
    chars        INTEGER NOT NULL DEFAULT 0,
    query_chars  INTEGER NOT NULL DEFAULT 0,
    alarm        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reads_room ON room_reads(room_id, id);
CREATE TABLE IF NOT EXISTS room_alarms (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id    TEXT NOT NULL,
    owner_id   TEXT NOT NULL,
    client_id  TEXT NOT NULL DEFAULT '',
    at         REAL NOT NULL,
    tool       TEXT NOT NULL,
    tokens     TEXT NOT NULL DEFAULT '[]',
    seen       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alarms_room ON room_alarms(room_id, id);
CREATE TABLE IF NOT EXISTS deliveries (
    room_id    TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    owner_id   TEXT NOT NULL,
    client_id  TEXT NOT NULL DEFAULT '',
    at         REAL NOT NULL,
    chars      INTEGER NOT NULL DEFAULT 0,
    payload    TEXT NOT NULL,
    report     TEXT NOT NULL DEFAULT '{}',
    seen       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (room_id, seq)
);
CREATE TABLE IF NOT EXISTS delivery_parts (
    room_id    TEXT NOT NULL,
    client_id  TEXT NOT NULL,
    key        TEXT NOT NULL,
    part       INTEGER NOT NULL,
    total      INTEGER NOT NULL,
    at         REAL NOT NULL,
    payload    TEXT NOT NULL,
    PRIMARY KEY (room_id, client_id, key, part)
);
CREATE TABLE IF NOT EXISTS room_tasks (
    room_id       TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    owner_id      TEXT NOT NULL,
    created_at    REAL NOT NULL,
    about         INTEGER,
    prompt        TEXT NOT NULL,
    prompt_tokens TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'waiting',
    client_id     TEXT NOT NULL DEFAULT '',
    taken_at      REAL,
    done_at       REAL,
    delivery      INTEGER,
    PRIMARY KEY (room_id, seq)
);
"""
TASK_STATES = ("waiting", "taken", "done", "cancelled")
PART_TTL = 24 * 3600.0  # parts of a delivery that never completed are dropped after this
READS_KEPT = 2000  # per room; older reads are dropped


class RoomError(ValueError):
    """A request about a room that can't be done (shown to the person)."""


def _seal_json(value: Any) -> str:
    return fd_crypto.seal(json.dumps(value, ensure_ascii=False)) if value not in (None, "", [], {}) else ""


def _open_json(sealed: str, default: Any) -> Any:
    if not sealed:
        return default
    raw = fd_crypto.unseal(sealed)
    if raw is None:
        raise fd_crypto.CryptoUnavailable("this room can't be opened with this deck's data key")
    try:
        return json.loads(raw)
    except ValueError:
        return default


def _open_text(sealed: str) -> str:
    if not sealed:
        return ""
    raw = fd_crypto.unseal(sealed)
    if raw is None:
        raise fd_crypto.CryptoUnavailable("this room can't be opened with this deck's data key")
    return raw


def clean_codename(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value or "")).strip()[:MAX_CODENAME]
    if not name:
        raise RoomError("a codename is required (what Claude sees, e.g. Project Falcon)")
    return name


class RoomStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.base = self.path.parent / "rooms"
        self._db: aiosqlite.Connection | None = None
        self._conn_lock = asyncio.Lock()

    async def _conn(self) -> aiosqlite.Connection:
        async with self._conn_lock:
            if self._db is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                db = await aiosqlite.connect(str(self.path))
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")
                await db.executescript(_DDL)
                # Columns added after a room store was first made.
                cur = await db.execute("PRAGMA table_info(room_files)")
                file_cols = {r[1] for r in await cur.fetchall()}
                if "info" not in file_cols:
                    await db.execute("ALTER TABLE room_files ADD COLUMN info TEXT NOT NULL DEFAULT '{}'")
                if "leftovers" not in file_cols:
                    await db.execute("ALTER TABLE room_files ADD COLUMN leftovers TEXT NOT NULL DEFAULT ''")
                cur = await db.execute("PRAGMA table_info(rooms)")
                if "allowed" not in {r[1] for r in await cur.fetchall()}:
                    await db.execute("ALTER TABLE rooms ADD COLUMN allowed TEXT NOT NULL DEFAULT ''")
                await db.commit()
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
                self._db = db
            return self._db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ── folders on disk ─────────────────────────────────────────────

    def room_dir(self, room_id: str) -> Path:
        if not re.fullmatch(r"r_[0-9a-f]{16}", room_id):
            raise RoomError("not a room id")
        return self.base / room_id

    def raw_path(self, room_id: str, file_id: str) -> Path:
        return self.room_dir(room_id) / "raw" / f"{file_id}.bin"

    def clean_file(self, room_id: str, file_id: str, ext: str) -> Path:
        ext = re.sub(r"[^a-z0-9]", "", ext.lower().lstrip("."))[:8]
        return self.room_dir(room_id) / "clean" / (f"{file_id}.{ext}" if ext else file_id)

    def text_file(self, room_id: str, file_id: str) -> Path:
        return self.room_dir(room_id) / "text" / f"{file_id}.md"

    # ── keys ────────────────────────────────────────────────────────

    async def _pepper(self) -> bytes:
        db = await self._conn()
        cur = await db.execute("SELECT v FROM meta WHERE k='path_pepper'")
        row = await cur.fetchone()
        if row is not None:
            value = fd_crypto.unseal(row[0])
            if not value:
                raise fd_crypto.CryptoUnavailable("the rooms' path key can't be opened with this deck's data key")
            return bytes.fromhex(value)
        await db.execute("INSERT OR IGNORE INTO meta (k, v) VALUES ('path_pepper', ?)",
                         (fd_crypto.seal(secrets.token_hex(32)),))
        await db.commit()
        return await self._pepper()

    async def path_key(self, origin: str, path: str) -> str:
        return hmac.new(await self._pepper(), f"{origin}\n{path}".encode(), hashlib.sha256).hexdigest()

    # ── rooms ───────────────────────────────────────────────────────

    def _view(self, row: Any) -> dict[str, Any]:
        r = dict(row)
        r["name"] = _open_text(r["name"])
        r["source"] = _open_json(r["source"], {})
        r["seed"] = _open_json(r["seed"], {})
        r["roles"] = _open_json(r["roles"], {})
        r["allowed"] = _open_json(r.get("allowed") or "", [])
        r["settings"] = json.loads(r["settings"] or "{}")
        r["job"] = json.loads(r["job"] or "{}")
        return r

    async def create(self, owner: str, codename: str, *, name: str = "", vault_id: str,
                     vault_name: str, room_id: str | None = None) -> dict[str, Any]:
        codename = clean_codename(codename)
        db = await self._conn()
        cur = await db.execute("SELECT COUNT(*) FROM rooms WHERE owner_id=?", (owner,))
        if (await cur.fetchone())[0] >= MAX_ROOMS_PER_OWNER:
            raise RoomError(f"you have {MAX_ROOMS_PER_OWNER} data rooms — delete some first")
        rid = room_id or new_room_id()
        now = time.time()
        await db.execute(
            """INSERT INTO rooms (id, owner_id, codename, name, vault_id, vault_name, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (rid, owner, codename, fd_crypto.seal(name.strip()[:200]) if name.strip() else "", vault_id,
             vault_name, now, now))
        await db.commit()
        for sub in ("raw", "clean", "text"):
            (self.room_dir(rid) / sub).mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.room_dir(rid), 0o700)
        except OSError:
            pass
        room = await self.get(owner, rid)
        assert room is not None
        return room

    async def get(self, owner: str, room_id: str) -> dict[str, Any] | None:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM rooms WHERE id=? AND owner_id=?", (room_id, owner))
        row = await cur.fetchone()
        return self._view(row) if row is not None else None

    async def get_any(self, room_id: str) -> dict[str, Any] | None:
        """A room by id, whoever owns it (the job runner, at deck start)."""
        db = await self._conn()
        cur = await db.execute("SELECT * FROM rooms WHERE id=?", (room_id,))
        row = await cur.fetchone()
        return self._view(row) if row is not None else None

    async def list_rooms(self, owner: str) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM rooms WHERE owner_id=? ORDER BY updated_at DESC", (owner,))
        return [self._view(r) for r in await cur.fetchall()]

    async def in_state(self, state: str) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM rooms WHERE state=?", (state,))
        return [self._view(r) for r in await cur.fetchall()]

    async def update(self, room_id: str, **fields: Any) -> None:
        sealed = {"name": lambda v: fd_crypto.seal(v.strip()[:200]) if v and v.strip() else "",
                  "source": _seal_json, "seed": _seal_json, "roles": _seal_json, "allowed": _seal_json}
        plain_json = {"settings", "job"}
        cols, values = [], []
        for key, value in fields.items():
            if key in sealed:
                value = sealed[key](value)
            elif key in plain_json:
                value = json.dumps(value, ensure_ascii=False)
            elif key == "codename":
                value = clean_codename(value)
            elif key not in ("state", "released_at"):
                raise RoomError(f"unknown room field: {key}")
            if key == "state" and value not in ROOM_STATES:
                raise RoomError(f"unknown room state: {value}")
            cols.append(f"{key}=?")
            values.append(value)
        if not cols:
            return
        cols.append("updated_at=?")
        values.append(time.time())
        db = await self._conn()
        await db.execute(f"UPDATE rooms SET {', '.join(cols)} WHERE id=?", (*values, room_id))
        await db.commit()

    async def delete(self, owner: str, room_id: str) -> bool:
        db = await self._conn()
        cur = await db.execute("DELETE FROM rooms WHERE id=? AND owner_id=?", (room_id, owner))
        if not cur.rowcount:
            await db.commit()
            return False
        for table in ("room_files", "room_text", "room_reads", "room_alarms", "deliveries", "delivery_parts",
                      "room_tasks"):
            await db.execute(f"DELETE FROM {table} WHERE room_id=?", (room_id,))
        await db.commit()
        await asyncio.to_thread(shutil.rmtree, self.room_dir(room_id), True)
        return True

    # ── files ───────────────────────────────────────────────────────

    def _file_view(self, row: Any) -> dict[str, Any]:
        f = dict(row)
        f["path"] = _open_text(f["path"])
        f.pop("path_key", None)
        f["warnings"] = json.loads(f["warnings"] or "[]")
        f["entities"] = json.loads(f["entities"] or "{}")
        f["info"] = json.loads(f.get("info") or "{}")
        f["leftovers"] = _open_json(f.get("leftovers") or "", {})
        f["released"] = bool(f["released"])
        return f

    async def add_file(self, room_id: str, *, origin: str, path: str, kind: str, size: int, sha256: str,
                       state: str = "new", error: str = "") -> tuple[str, bool]:
        """Add (or, same path again, replace) a file. Returns (file id, whether it
        replaced one): a replaced file goes back to ``new`` and out of release."""
        if origin not in ORIGINS:
            raise RoomError(f"unknown origin: {origin}")
        if state not in FILE_STATES:
            raise RoomError(f"unknown file state: {state}")
        key = await self.path_key(origin, path)
        db = await self._conn()
        cur = await db.execute("SELECT id, sha256 FROM room_files WHERE room_id=? AND path_key=?", (room_id, key))
        row = await cur.fetchone()
        now = time.time()
        if row is not None:
            if row["sha256"] == sha256:
                return row["id"], False
            await db.execute(
                """UPDATE room_files SET kind=?, size=?, sha256=?, state=?, error=?, warnings='[]',
                   entities='{}', info='{}', leftovers='', text_chars=0, released=0, clean_path='', updated_at=?
                   WHERE room_id=? AND id=?""",
                (kind, size, sha256, state, error, now, room_id, row["id"]))
            await db.execute("DELETE FROM room_text WHERE room_id=? AND file_id=?", (room_id, row["id"]))
            await db.commit()
            return row["id"], True
        fid = "f_" + secrets.token_hex(8)
        await db.execute(
            """INSERT INTO room_files (room_id, id, origin, path, path_key, kind, size, sha256, state, error,
                                       updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (room_id, fid, origin, fd_crypto.seal(path), key, kind, size, sha256, state, error, now))
        await db.commit()
        return fid, False

    async def files(self, room_id: str, *, released_only: bool = False) -> list[dict[str, Any]]:
        db = await self._conn()
        sql = "SELECT * FROM room_files WHERE room_id=?" + (" AND released=1" if released_only else "")
        cur = await db.execute(sql, (room_id,))
        out = [self._file_view(r) for r in await cur.fetchall()]
        return sorted(out, key=lambda f: f["path"].casefold())

    async def file(self, room_id: str, file_id: str) -> dict[str, Any] | None:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM room_files WHERE room_id=? AND id=?", (room_id, file_id))
        row = await cur.fetchone()
        return self._file_view(row) if row is not None else None

    async def set_file(self, room_id: str, file_id: str, **fields: Any) -> None:
        cols, values = [], []
        for key, value in fields.items():
            if key in ("warnings", "entities", "info"):
                value = json.dumps(value, ensure_ascii=False)
            elif key == "leftovers":
                value = _seal_json(value)  # real spellings: sealed like the paths
            elif key == "released":
                value = 1 if value else 0
            elif key == "state" and value not in FILE_STATES:
                raise RoomError(f"unknown file state: {value}")
            elif key not in ("state", "clean_path", "error", "text_chars", "kind"):
                raise RoomError(f"unknown file field: {key}")
            cols.append(f"{key}=?")
            values.append(value)
        if not cols:
            return
        cols.append("updated_at=?")
        values.append(time.time())
        db = await self._conn()
        await db.execute(f"UPDATE room_files SET {', '.join(cols)} WHERE room_id=? AND id=?",
                         (*values, room_id, file_id))
        await db.commit()

    async def release(self, room_id: str) -> int:
        """Every file that came through clean is released; returns how many."""
        db = await self._conn()
        cur = await db.execute("UPDATE room_files SET released=1 WHERE room_id=? AND state='done' AND released=0",
                               (room_id,))
        await db.commit()
        return cur.rowcount

    async def remove_file(self, room_id: str, file_id: str) -> None:
        db = await self._conn()
        await db.execute("DELETE FROM room_files WHERE room_id=? AND id=?", (room_id, file_id))
        await db.execute("DELETE FROM room_text WHERE room_id=? AND file_id=?", (room_id, file_id))
        await db.commit()

    # ── originals (ZIP) ─────────────────────────────────────────────

    def write_raw(self, room_id: str, file_id: str, blob: bytes, cipher: Any) -> None:
        path = self.raw_path(room_id, file_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cipher.encrypt(blob))

    def read_raw(self, room_id: str, file_id: str, cipher: Any) -> bytes:
        return cipher.decrypt(self.raw_path(room_id, file_id).read_bytes())

    # ── search (anonymized text only) ───────────────────────────────

    async def index_text(self, room_id: str, file_id: str, pages: list[tuple[str, str]]) -> None:
        db = await self._conn()
        await db.execute("DELETE FROM room_text WHERE room_id=? AND file_id=?", (room_id, file_id))
        await db.executemany("INSERT INTO room_text (room_id, file_id, page, body) VALUES (?, ?, ?, ?)",
                             [(room_id, file_id, page, body) for page, body in pages if body.strip()])
        await db.commit()

    async def search(self, room_id: str, query: str, *, limit: int = 20,
                     released_only: bool = True) -> list[dict[str, Any]]:
        words = re.findall(r"[\w⟦⟧]+", str(query or ""), re.UNICODE)[:12]
        if not words:
            return []
        match = " ".join('"' + w.replace('"', "") + '"' for w in words)
        db = await self._conn()
        sql = ("SELECT t.file_id, t.page, snippet(room_text, 3, '[', ']', ' … ', 24) AS snip, f.clean_path "
               "FROM room_text t JOIN room_files f ON f.room_id = t.room_id AND f.id = t.file_id "
               "WHERE room_text MATCH ? AND t.room_id = ?" + (" AND f.released = 1" if released_only else "")
               + " ORDER BY rank LIMIT ?")
        try:
            cur = await db.execute(sql, (match, room_id, max(1, min(int(limit), 100))))
        except aiosqlite.OperationalError:
            return []
        return [{"file_id": r["file_id"], "path": r["clean_path"], "page": r["page"], "snippet": r["snip"]}
                for r in await cur.fetchall()]


    # ── Claude's access (the room connector, P2) ────────────────────

    async def set_grant(self, owner: str, client_id: str, rooms: list[str], all_rooms: bool) -> None:
        """Which rooms one connected app (a Claude connector) may read."""
        db = await self._conn()
        now = time.time()
        await db.execute(
            """INSERT INTO mcp_grants (owner_id, client_id, rooms, all_rooms, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(owner_id, client_id)
               DO UPDATE SET rooms=excluded.rooms, all_rooms=excluded.all_rooms, updated_at=excluded.updated_at""",
            (owner, client_id, json.dumps(sorted(set(rooms))), 1 if all_rooms else 0, now, now))
        await db.commit()

    async def grant(self, owner: str, client_id: str) -> dict[str, Any] | None:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM mcp_grants WHERE owner_id=? AND client_id=?", (owner, client_id))
        row = await cur.fetchone()
        if row is None:
            return None
        g = dict(row)
        g["rooms"] = json.loads(g["rooms"] or "[]")
        g["all_rooms"] = bool(g["all_rooms"])
        return g

    async def grants(self, owner: str) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM mcp_grants WHERE owner_id=? ORDER BY created_at DESC", (owner,))
        out = []
        for row in await cur.fetchall():
            g = dict(row)
            g["rooms"] = json.loads(g["rooms"] or "[]")
            g["all_rooms"] = bool(g["all_rooms"])
            out.append(g)
        return out

    async def delete_grant(self, owner: str, client_id: str) -> bool:
        db = await self._conn()
        cur = await db.execute("DELETE FROM mcp_grants WHERE owner_id=? AND client_id=?", (owner, client_id))
        await db.commit()
        return bool(cur.rowcount)

    async def log_read(self, room_id: str, owner: str, client_id: str, tool: str, *, file_id: str = "",
                       page: int = 0, chars: int = 0, query_chars: int = 0, alarm: bool = False) -> None:
        """What Claude read: tool, file, page and sizes — never the text, never a query."""
        db = await self._conn()
        await db.execute(
            """INSERT INTO room_reads (room_id, owner_id, client_id, at, tool, file_id, page, chars, query_chars,
                                       alarm) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (room_id, owner, client_id, time.time(), tool, file_id, page, chars, query_chars, 1 if alarm else 0))
        await db.execute(
            """DELETE FROM room_reads WHERE room_id=? AND id <= (
                   SELECT id FROM room_reads WHERE room_id=? ORDER BY id DESC LIMIT 1 OFFSET ?)""",
            (room_id, room_id, READS_KEPT))
        await db.commit()

    async def reads(self, room_id: str, limit: int = 200) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM room_reads WHERE room_id=? ORDER BY id DESC LIMIT ?",
                               (room_id, max(1, min(int(limit), 1000))))
        return [dict(r) for r in await cur.fetchall()]

    async def add_alarm(self, room_id: str, owner: str, client_id: str, tool: str, tokens: list[str]) -> None:
        db = await self._conn()
        await db.execute(
            "INSERT INTO room_alarms (room_id, owner_id, client_id, at, tool, tokens) VALUES (?, ?, ?, ?, ?, ?)",
            (room_id, owner, client_id, time.time(), tool, json.dumps(tokens[:50])))
        await db.commit()

    async def alarms(self, room_id: str, *, unseen_only: bool = False) -> list[dict[str, Any]]:
        db = await self._conn()
        sql = "SELECT * FROM room_alarms WHERE room_id=?" + (" AND seen=0" if unseen_only else "") + \
            " ORDER BY id DESC LIMIT 100"
        cur = await db.execute(sql, (room_id,))
        out = []
        for row in await cur.fetchall():
            a = dict(row)
            a["tokens"] = json.loads(a["tokens"] or "[]")
            a["seen"] = bool(a["seen"])
            out.append(a)
        return out

    async def mark_alarms_seen(self, room_id: str) -> None:
        db = await self._conn()
        await db.execute("UPDATE room_alarms SET seen=1 WHERE room_id=?", (room_id,))
        await db.commit()

    # ── the Inbox: what Claude sent back (P3) ───────────────────────
    # Payloads are encrypted by the caller with the room vault's key: the
    # restored text is the most private thing a room holds.

    async def add_part(self, room_id: str, client_id: str, key: str, part: int, total: int,
                       payload: str) -> list[int]:
        """Keep one part of a delivery; returns the part numbers held so far."""
        db = await self._conn()
        now = time.time()
        await db.execute("DELETE FROM delivery_parts WHERE at < ?", (now - PART_TTL,))
        await db.execute(
            """INSERT INTO delivery_parts (room_id, client_id, key, part, total, at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(room_id, client_id, key, part)
               DO UPDATE SET payload=excluded.payload, at=excluded.at, total=excluded.total""",
            (room_id, client_id, key, part, total, now, payload))
        await db.commit()
        cur = await db.execute("SELECT part FROM delivery_parts WHERE room_id=? AND client_id=? AND key=? AND total=?"
                               " ORDER BY part", (room_id, client_id, key, total))
        return [r[0] for r in await cur.fetchall()]

    async def take_parts(self, room_id: str, client_id: str, key: str, total: int) -> list[str]:
        """The parts' payloads in order, removed from the waiting list."""
        db = await self._conn()
        cur = await db.execute("SELECT payload FROM delivery_parts WHERE room_id=? AND client_id=? AND key=? AND total=?"
                               " ORDER BY part", (room_id, client_id, key, total))
        out = [r[0] for r in await cur.fetchall()]
        await db.execute("DELETE FROM delivery_parts WHERE room_id=? AND client_id=? AND key=?",
                         (room_id, client_id, key))
        await db.commit()
        return out

    async def add_delivery(self, room_id: str, owner: str, client_id: str, payload: str, chars: int,
                           report: dict[str, Any]) -> int:
        db = await self._conn()
        cur = await db.execute("SELECT COALESCE(MAX(seq), 0) FROM deliveries WHERE room_id=?", (room_id,))
        seq = int((await cur.fetchone())[0]) + 1
        await db.execute(
            """INSERT INTO deliveries (room_id, seq, owner_id, client_id, at, chars, payload, report)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (room_id, seq, owner, client_id, time.time(), chars, payload, json.dumps(report, ensure_ascii=False)))
        await db.commit()
        return seq

    def _delivery_view(self, row: Any) -> dict[str, Any]:
        d = dict(row)
        d["report"] = json.loads(d["report"] or "{}")
        d["seen"] = bool(d["seen"])
        return d

    async def deliveries(self, room_id: str) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM deliveries WHERE room_id=? ORDER BY seq DESC", (room_id,))
        return [self._delivery_view(r) for r in await cur.fetchall()]

    async def delivery(self, room_id: str, seq: int) -> dict[str, Any] | None:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM deliveries WHERE room_id=? AND seq=?", (room_id, seq))
        row = await cur.fetchone()
        return self._delivery_view(row) if row is not None else None

    async def update_delivery(self, room_id: str, seq: int, *, report: dict[str, Any] | None = None,
                              payload: str | None = None) -> bool:
        db = await self._conn()
        sets, args = [], []
        if report is not None:
            sets.append("report=?")
            args.append(json.dumps(report, ensure_ascii=False))
        if payload is not None:
            sets.append("payload=?")
            args.append(payload)
        if not sets:
            return False
        cur = await db.execute(f"UPDATE deliveries SET {', '.join(sets)} WHERE room_id=? AND seq=?",
                               (*args, room_id, seq))
        await db.commit()
        return bool(cur.rowcount)

    async def grammar_waiting(self) -> list[tuple[str, int]]:
        """Deliveries whose case endings (P7) were still to be fixed."""
        db = await self._conn()
        cur = await db.execute(
            "SELECT room_id, seq FROM deliveries WHERE json_extract(report, '$.grammar.state') = 'waiting' "
            "ORDER BY at")
        return [(r[0], int(r[1])) for r in await cur.fetchall()]

    # ── tasks the person leaves for Claude (asked from the Inbox) ───────
    #
    # ``prompt`` is the person's own words and ``prompt_tokens`` what Claude
    # reads, both encrypted with the room vault's key by the caller.

    async def add_task(self, room_id: str, owner: str, prompt: str, prompt_tokens: str, about: int | None) -> int:
        db = await self._conn()
        cur = await db.execute("SELECT COALESCE(MAX(seq), 0) FROM room_tasks WHERE room_id=?", (room_id,))
        seq = int((await cur.fetchone())[0]) + 1
        await db.execute(
            """INSERT INTO room_tasks (room_id, seq, owner_id, created_at, about, prompt, prompt_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (room_id, seq, owner, time.time(), about, prompt, prompt_tokens))
        await db.commit()
        return seq

    async def tasks(self, room_id: str) -> list[dict[str, Any]]:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM room_tasks WHERE room_id=? ORDER BY seq DESC", (room_id,))
        return [dict(r) for r in await cur.fetchall()]

    async def task(self, room_id: str, seq: int) -> dict[str, Any] | None:
        db = await self._conn()
        cur = await db.execute("SELECT * FROM room_tasks WHERE room_id=? AND seq=?", (room_id, seq))
        row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def open_tasks(self, room_id: str) -> list[dict[str, Any]]:
        """Waiting or taken (not answered yet), oldest first."""
        db = await self._conn()
        cur = await db.execute("SELECT * FROM room_tasks WHERE room_id=? AND state IN ('waiting', 'taken') "
                               "ORDER BY seq", (room_id,))
        return [dict(r) for r in await cur.fetchall()]

    async def set_task(self, room_id: str, seq: int, **fields: Any) -> bool:
        allowed = {"state", "client_id", "taken_at", "done_at", "delivery"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return False
        db = await self._conn()
        cur = await db.execute(f"UPDATE room_tasks SET {', '.join(f'{k}=?' for k in sets)} WHERE room_id=? AND seq=?",
                               (*sets.values(), room_id, seq))
        await db.commit()
        return bool(cur.rowcount)

    async def mark_delivery_seen(self, room_id: str, seq: int) -> None:
        db = await self._conn()
        await db.execute("UPDATE deliveries SET seen=1 WHERE room_id=? AND seq=?", (room_id, seq))
        await db.commit()

    async def delete_delivery(self, room_id: str, seq: int) -> bool:
        db = await self._conn()
        cur = await db.execute("DELETE FROM deliveries WHERE room_id=? AND seq=?", (room_id, seq))
        await db.commit()
        return bool(cur.rowcount)


def new_room_id() -> str:
    return "r_" + secrets.token_hex(8)


# ── the store this deck uses ────────────────────────────────────────

_data_dir: Path | None = None
_store: RoomStore | None = None


def configure(*, data_dir: Path) -> None:
    global _data_dir
    _data_dir = Path(data_dir)


def store() -> RoomStore:
    global _store
    if _store is None:
        base = _data_dir
        if base is None:
            env = (os.environ.get("FD_DATA_DIR") or "").strip()
            if env:
                base = Path(env).expanduser().resolve()
            else:
                from captain_claw.flight_deck.server import DATA_DIR

                base = DATA_DIR
        _store = RoomStore(base / "rooms.db")
    return _store


def set_store(value: RoomStore | None) -> None:
    """Tests: point the deck at another store."""
    global _store
    _store = value


async def close() -> None:
    if _store is not None:
        await _store.close()
