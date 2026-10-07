"""User-facing relational datastore backed by a dedicated SQLite database.

Provides structured table management, CRUD operations, import/export,
and read-only raw SQL queries.  Completely separate from the session
and memory databases.

PR C (shared agents): the store is a commons. Every table and row records
its creator — ``''`` for the agent's owner (and everything from before PR C),
else the member's Flight Deck user id — in ``_ds_tables.created_by*`` and in
the hidden physical columns ``SYSTEM_COLUMNS`` of every ``ds_*`` table,
stamped in the same statement that writes. A member (``current_actor()``,
from the bound speaker principal) reads everything, adds rows to any table
and changes only rows and tables they created; every member refusal raises
:class:`MemberDeniedError` before anything is written. Mutations run under
one write lock per manager, from their first check to their commit.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import os
import re
import sqlite3
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)


class ProtectedError(Exception):
    """Raised when an operation violates a protection rule."""


class MemberDeniedError(ProtectedError):
    """A shared-agent member's datastore call refused (message = a DS_* text)."""


# ── PR C: creators, member limits, texts (contract part 0b §4, 0c §1) ─

SYSTEM_COLUMNS = ("_created_by", "_created_by_name")
MEMBER_MAX_TABLES = 10
MEMBER_TABLE_HEADROOM = 10
MEMBER_MAX_ROWS = 10_000
MEMBER_NAME_MAX = 120
MEMBER_ROW_HEADROOM_DIVISOR = 10   # member rows stop at max_rows_per_table - max_rows_per_table // 10
MEMBER_SQL_DEADLINE_S = 2.0        # member raw SELECT, read-only connection, then interrupted
MEMBER_IMPORT_MAX_BYTES = 25 * 1024 * 1024
MEMBER_IMPORT_MAX_UNZIPPED_BYTES = 100 * 1024 * 1024
# What a member's rows may STORE (an import's source caps say nothing about
# that: one xlsx shared string can fill every cell): one value at most
# MEMBER_MAX_VALUE_BYTES, a member's table / import at most
# MEMBER_MAX_COLUMNS columns, everything a member added at most
# MEMBER_MAX_STORED_BYTES across the store — counted under the write lock.
# A name a member gives a table or column (an xlsx header is a shared string
# too, stored in the schema) at most MEMBER_MAX_NAME_CHARS.
MEMBER_MAX_VALUE_BYTES = 4 * 1024 * 1024
MEMBER_MAX_COLUMNS = 100
MEMBER_MAX_NAME_CHARS = 128
MEMBER_MAX_STORED_BYTES = 50 * 1024 * 1024

DS_NOT_YOUR_ROWS = ('Some of the rows this would change were added by someone else. You can '
                    'change or delete only rows you added — narrow it with {"_mine": true}.')
DS_NOT_YOUR_TABLE = ("Only the person who created this table, or the agent's owner, can change "
                     "its structure, rename it or drop it.")
DS_FOREIGN_ROWS = ("Other people have added rows to this table, so only the agent's owner can "
                   "do that now.")
DS_OWNER_ONLY = "Only the agent's owner can protect or unprotect data."
DS_EXPRESSION_MEMBER = "In a shared chat, update_column takes a value, not an expression."
DS_PROJECT_MEMBER = ("In a shared chat only this agent's own datastore is available — leave out "
                     "`project`.")
DS_MEMBER_TABLE_LIMIT = "You can create at most 10 tables on this agent."
DS_MEMBER_NO_ROOM = "This agent's datastore has no room for more tables from members."
DS_MEMBER_ROW_LIMIT = "You can add at most 10,000 rows on this agent."
DS_IDENTITY_LOST = "The datastore can't tell who is asking right now — try again."
DS_MEMBER_UNAVAILABLE = "The datastore isn't available in shared chats on this agent."
DS_TABLE_NEARLY_FULL = ("This table is nearly full — only the agent's owner can add more rows "
                        "to it.")
DS_SQL_LIMITS = ("In a shared chat, sql runs plain SELECTs that finish within 2 seconds (no WITH "
                 "RECURSIVE) — narrow it, or use query.")
DS_IMPORT_TOO_LARGE = ("That file is too large to import in a shared chat (25 MB, or 100 MB "
                       "unzipped, at most).")
DS_MEMBER_VALUE_TOO_LARGE = ("In a shared chat one value can be at most 4 MB — shorten it or "
                             "split it up.")
DS_MEMBER_TOO_MANY_COLUMNS = "In a shared chat a table can have at most 100 columns."
DS_MEMBER_NAME_TOO_LONG = ("In a shared chat a table or column name can be at most 128 "
                           "characters.")
DS_MEMBER_STORAGE_LIMIT = "You can store at most 50 MB of data on this agent."
DS_MEMBER_TABLE_NAME = ("In a shared chat a table can't be named like a column, an SQL keyword, "
                        "an SQL function or one of SQLite's own tables — pick another name.")

# A member's raw SELECT (beyond the contract's deadline and row cap): one
# value at most _MEMBER_SQL_VALUE_MAX bytes (what a member may store in one
# value), the fetched result at most _MEMBER_SQL_RESULT_MAX — so a member
# can't exhaust the agent's memory.
_MEMBER_SQL_VALUE_MAX = MEMBER_MAX_VALUE_BYTES
_MEMBER_SQL_RESULT_MAX = 32 * 1024 * 1024


def _value_size(v: Any) -> int:
    if isinstance(v, (str, bytes, bytearray, memoryview)):
        return len(v)
    return 16


def _stored_size(v: Any) -> int:
    """Bytes *v* takes in a row — what ``octet_length`` reads back."""
    if v is None:
        return 0
    if isinstance(v, str):
        return len(v) if v.isascii() else len(v.encode("utf-8", "surrogatepass"))
    if isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    return len(str(v))


# Bytes a stored value takes, in SQL (octet_length reads only the record
# header — never a large value's overflow pages).
_OCTETS = ("octet_length({})" if sqlite3.sqlite_version_info >= (3, 43, 0)
           else "length(CAST({} AS BLOB))")

# Names a member's table can't take (raw_select maps table names to their
# ds_ tables textually): SQLite's keywords, rowid aliases and core functions
# (the connection's own function list is added at check time).
_SQL_RESERVED = frozenset("""
abort action add after all alter always analyze and as asc attach autoincrement before begin
between by cascade case cast check collate column commit conflict constraint create cross current
current_date current_time current_timestamp database default deferrable deferred delete desc
detach distinct do drop each else end escape except exclude exclusive exists explain fail filter
first following for foreign from full generated glob group groups having if ignore immediate in
index indexed initially inner insert instead intersect into is isnull join key last left like
limit match materialized natural no not nothing notnull null nulls of offset on or order others
outer over partition plan pragma preceding primary query raise range recursive references regexp
reindex release rename replace restrict returning right rollback row rows savepoint select set
table temp temporary then ties to transaction trigger unbounded union unique update using vacuum
values view virtual when where window with without true false rowid oid main
abs avg changes char coalesce concat concat_ws count date datetime format group_concat hex
ifnull iif instr json julianday last_insert_rowid length likelihood likely lower ltrim max min
nullif octet_length printf quote random randomblob round rtrim sign soundex sqlite_version
strftime string_agg substr substring sum time timediff total total_changes trim typeof unhex
unicode unixepoch unlikely upper zeroblob acos asin atan ceil ceiling cos degrees exp floor ln
log log10 log2 mod pi pow power radians sin sqrt tan trunc row_number rank dense_rank
percent_rank cume_dist ntile lag lead first_value last_value nth_value
json_each json_tree jsonb_each jsonb_tree generate_series dbstat
""".split())
# ... nor start like SQLite's own tables / table-valued pragmas or like an
# internal name (an owner's SQL may name ds_<table> directly).
_SQL_RESERVED_PREFIXES = ("sqlite_", "pragma_", "ds_")


# Text cells an export prefixes with "'" so a spreadsheet never runs them (J13).
_FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")


@dataclass(frozen=True)
class DatastoreActor:
    """Who a datastore call acts for."""

    kind: str       # "owner" | "member"
    user_id: str    # "" for the owner
    name: str       # name snapshot ("" for the owner)


OWNER_ACTOR = DatastoreActor("owner", "", "")


def current_actor() -> DatastoreActor:
    """The actor of the running call, from the bound speaker principal.

    No principal → the owner, unless the speaker context was lost while
    member work is live (a bare worker thread): then nobody can be told
    apart and every write is refused. An unverified or non-process member is
    refused too — never treated as the owner.
    """
    from captain_claw import speaker

    p = speaker.current()
    if p is None:
        if speaker.identity_lost():
            raise MemberDeniedError(DS_IDENTITY_LOST)
        return OWNER_ACTOR
    if not p.speaker_id:
        raise MemberDeniedError(DS_IDENTITY_LOST)
    if speaker.runtime_of(p) != "process":
        raise MemberDeniedError(DS_MEMBER_UNAVAILABLE)
    name = " ".join(str(p.display_name or "").split())[:MEMBER_NAME_MAX] or "Member"
    return DatastoreActor("member", p.speaker_id, name)


def creator_dict(created_by: Any, created_by_name: Any) -> dict[str, str]:
    """The wire ``Creator`` (contract part 0b §2.2) of a stamp."""
    cb = str(created_by or "")
    if cb:
        return {"kind": "member", "user_id": cb, "name": str(created_by_name or "")}
    return {"kind": "owner", "user_id": "", "name": ""}


def neutralize_rows(
    rows: list[list[Any]], creators: list[dict[str, str]] | None = None, mode: str = "all",
) -> list[list[Any]]:
    """Rows for an export file (J13): in every row (``"all"``) or every
    member-created row (``"members"``), a text cell starting with ``= + - @``,
    tab or CR gets a leading ``'``. ``"none"`` → the rows unchanged."""
    if mode not in ("all", "members"):
        return rows
    out: list[list[Any]] = []
    for i, row in enumerate(rows):
        if mode == "members":
            c = creators[i] if creators is not None and i < len(creators) else None
            if not (isinstance(c, dict) and c.get("kind") == "member"):
                out.append(row)
                continue
        out.append([
            "'" + v if isinstance(v, str) and v[:1] in _FORMULA_LEADS else v
            for v in row
        ])
    return out


# The managers whose write lock the running task holds (re-entrant writes).
_HELD: ContextVar[frozenset[int]] = ContextVar("ds_write_held", default=frozenset())

# ── Column type mapping ──────────────────────────────────────────────
# user-facing type → SQLite affinity
TYPE_MAP: dict[str, str] = {
    "text": "TEXT",
    "integer": "INTEGER",
    "real": "REAL",
    "boolean": "INTEGER",    # stored as 0/1
    "date": "TEXT",          # ISO date string
    "datetime": "TEXT",      # ISO datetime string
    "json": "TEXT",          # JSON-encoded string
}

VALID_TYPES = set(TYPE_MAP.keys())

# All user tables are prefixed to avoid clashing with meta tables.
TABLE_PREFIX = "ds_"

# Operators allowed in structured WHERE clauses.
_ALLOWED_OPS = {"=", "!=", "<", ">", "<=", ">=", "LIKE", "NOT LIKE", "IN", "NOT IN", "IS NULL", "IS NOT NULL"}

# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ColumnDef:
    name: str
    col_type: str  # one of VALID_TYPES
    position: int = 0


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    row_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""          # PR C: "" = the agent's owner, else a member's FD id
    created_by_name: str = ""     # name snapshot of that member


# ── DatastoreManager ─────────────────────────────────────────────────

class DatastoreManager:
    """Manages the user-facing relational datastore."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            self.db_path = Path(cfg.datastore.path).expanduser()
        else:
            self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None
        # PR C (J19): the connection is published only after the schema
        # migration; one write lock covers every mutation from its first
        # check to its commit; member raw SELECTs use their own read-only
        # connection.
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._sys_ok: set[str] = set()   # internal names with BOTH system columns
        self._ro_db: aiosqlite.Connection | None = None
        self._ro_lock = asyncio.Lock()
        self._ro_deadline = 0.0

    # ── lifecycle ────────────────────────────────────────────────────

    async def _ensure_db(self) -> None:
        if self._db is not None:
            return
        async with self._init_lock:
            if self._db is not None:
                return
            db = await aiosqlite.connect(str(self.db_path))
            try:
                sys_ok = await self._open_schema(db)
            except BaseException:
                await db.close()
                raise
            self._sys_ok = sys_ok
            self._db = db

    async def _open_schema(self, db: aiosqlite.Connection) -> set[str]:
        """PRAGMAs, meta tables and the eager PR C migration on a connection
        nobody else sees yet. Returns the tables that have both system columns."""
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        # Tolerate concurrent writers: a folder-bound datastore shared by a
        # Basna/Vatra run's agents can have several processes writing at once.
        # WAL allows many readers + one writer; the busy timeout makes a blocked
        # writer wait for the lock instead of failing with "database is locked".
        await db.execute("PRAGMA busy_timeout=5000")

        await db.execute("""
            CREATE TABLE IF NOT EXISTS _ds_tables (
                name             TEXT PRIMARY KEY,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL,
                created_by       TEXT NOT NULL DEFAULT '',
                created_by_name  TEXT NOT NULL DEFAULT ''
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS _ds_columns (
                table_name  TEXT NOT NULL,
                col_name    TEXT NOT NULL,
                col_type    TEXT NOT NULL DEFAULT 'text',
                position    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (table_name, col_name),
                FOREIGN KEY (table_name) REFERENCES _ds_tables(name) ON DELETE CASCADE
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS _ds_protections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name  TEXT NOT NULL,
                level       TEXT NOT NULL CHECK(level IN ('table','column','row','cell')),
                row_id      INTEGER,
                col_name    TEXT,
                reason      TEXT,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (table_name) REFERENCES _ds_tables(name) ON DELETE CASCADE,
                UNIQUE(table_name, level, row_id, col_name)
            )
        """)
        # PR C migration: a store from before PR C reads as owner-created.
        meta_cols = await self._column_names(db, "_ds_tables")
        for col in ("created_by", "created_by_name"):
            if col not in meta_cols:
                await db.execute(
                    f"ALTER TABLE _ds_tables ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
        async with db.execute("SELECT name FROM _ds_tables") as cur:
            names = [r[0] for r in await cur.fetchall()]
        async with db.execute("SELECT name FROM sqlite_master WHERE type = 'table'") as cur:
            physical = {r[0] for r in await cur.fetchall()}
        sys_ok: set[str] = set()
        for name in names:
            internal = self._internal_name(name)
            if internal not in physical:
                continue
            try:
                if await self._add_system_columns(db, internal):
                    sys_ok.add(internal)
            except Exception as exc:
                log.warning("Datastore creator migration skipped a table", table=name,
                            error=type(exc).__name__)
        await db.commit()
        return sys_ok

    @staticmethod
    async def _column_names(db: aiosqlite.Connection, table: str) -> set[str]:
        async with db.execute(f'PRAGMA table_info("{table}")') as cur:
            return {r[1] for r in await cur.fetchall()}

    @classmethod
    async def _add_system_columns(cls, db: aiosqlite.Connection, internal: str) -> bool:
        """Add the missing system columns and the creator index to *internal*;
        True when both columns are there afterwards."""
        have = await cls._column_names(db, internal)
        for col in SYSTEM_COLUMNS:
            if col not in have:
                await db.execute(
                    f'ALTER TABLE "{internal}" ADD COLUMN "{col}" TEXT NOT NULL DEFAULT \'\'')
        await db.execute(
            f'CREATE INDEX IF NOT EXISTS "ix_{internal}_created_by" ON "{internal}"("_created_by")')
        have = await cls._column_names(db, internal)
        return all(col in have for col in SYSTEM_COLUMNS)

    async def _sys_cols_present(self, internal: str) -> bool:
        """Whether *internal* has both system columns. Reads never ALTER, but a
        table another process created (or migrated) after this store was
        opened already has them on disk: trust the table, not just the cache."""
        if internal in self._sys_ok:
            return True
        assert self._db is not None
        try:
            have = await self._column_names(self._db, internal)
        except Exception:
            return False
        if all(col in have for col in SYSTEM_COLUMNS):
            self._sys_ok.add(internal)
            return True
        return False

    async def _ensure_system_columns(self, internal: str, actor: DatastoreActor) -> bool:
        """Lazy fallback, only inside :meth:`_writing`: a table another process
        created after this store was opened. False (owner only) when the
        columns can't be added — a member write is refused instead."""
        if internal in self._sys_ok:
            return True
        assert self._db is not None
        ok = False
        try:
            ok = await self._add_system_columns(self._db, internal)
            await self._db.commit()
        except Exception as exc:
            log.warning("Datastore creator migration failed", table=internal,
                        error=type(exc).__name__)
        if ok:
            self._sys_ok.add(internal)
            return True
        if actor.kind == "member":
            raise MemberDeniedError(DS_MEMBER_UNAVAILABLE)
        return False

    @contextlib.asynccontextmanager
    async def _writing(self):
        """Hold this store's write lock (re-entrant within one task)."""
        if id(self) in _HELD.get():
            yield
            return
        async with self._write_lock:
            tok = _HELD.set(_HELD.get() | {id(self)})
            try:
                yield
            except BaseException:
                # A write that failed half-way must not ride along with the
                # next one's commit (stamped with someone else's call).
                if self._db is not None:
                    with contextlib.suppress(Exception):
                        await self._db.rollback()
                raise
            finally:
                _HELD.reset(tok)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
        if self._ro_db is not None:
            await self._ro_db.close()
            self._ro_db = None

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize a table/column name to ``[a-z0-9_]``."""
        cleaned = re.sub(r"[^a-z0-9_]", "_", name.strip().lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned or cleaned[0].isdigit():
            cleaned = "t_" + cleaned
        return cleaned

    @staticmethod
    def _dedup_headers(headers: list[str]) -> list[str]:
        """Ensure all header names are unique by appending _2, _3, ... for dupes."""
        seen: dict[str, int] = {}
        result: list[str] = []
        for h in headers:
            if h not in seen:
                seen[h] = 1
                result.append(h)
            else:
                seen[h] += 1
                result.append(f"{h}_{seen[h]}")
        return result

    @staticmethod
    def _internal_name(user_name: str) -> str:
        return TABLE_PREFIX + DatastoreManager._safe_name(user_name)

    async def _resolve_table(self, name: str) -> tuple[str, str]:
        """Return ``(safe_name, internal_name)`` and verify the table exists."""
        await self._ensure_db()
        safe = self._safe_name(name)
        assert self._db is not None
        async with self._db.execute(
            "SELECT 1 FROM _ds_tables WHERE name = ?", (safe,)
        ) as cur:
            if not await cur.fetchone():
                raise ValueError(f"Table not found: {name}")
        return safe, self._internal_name(safe)

    async def _table_columns(self, safe_name: str) -> list[ColumnDef]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT col_name, col_type, position FROM _ds_columns "
            "WHERE table_name = ? ORDER BY position",
            (safe_name,),
        ) as cur:
            rows = await cur.fetchall()
        return [ColumnDef(name=r[0], col_type=r[1], position=r[2]) for r in rows]

    async def _row_count(self, internal: str) -> int:
        assert self._db is not None
        try:
            async with self._db.execute(f'SELECT COUNT(*) FROM "{internal}"') as cur:
                row = await cur.fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def _now(self) -> str:  # noqa: PLR6301
        return datetime.now(UTC).isoformat()

    # ── PR C: creators and member rules ──────────────────────────────

    async def _table_creator(self, safe: str) -> str:
        assert self._db is not None
        async with self._db.execute(
            "SELECT created_by FROM _ds_tables WHERE name = ?", (safe,)
        ) as cur:
            row = await cur.fetchone()
        return str(row[0] or "") if row else ""

    async def _require_table_owner(self, safe: str, actor: DatastoreActor) -> None:
        """A member changes the structure of tables they created only."""
        if actor.kind != "member":
            return
        if await self._table_creator(safe) != actor.user_id:
            raise MemberDeniedError(DS_NOT_YOUR_TABLE)

    def _scope_where(
        self, where_clause: str, params: Any, actor: DatastoreActor, *, foreign: bool = False,
    ) -> tuple[str, list[Any]]:
        """*where_clause* narrowed to the actor's own rows (or, ``foreign``, to
        everyone else's). The ONE place a member UPDATE / DELETE / count gets
        its creator test — keyed on the BUILT clause (``{"_all": true}``
        builds ``""``), never on the caller's filter."""
        if actor.kind != "member" and not foreign:
            return where_clause, list(params)
        test = 'NOT ("_created_by" = ?)' if foreign else '"_created_by" = ?'
        if where_clause:
            if not where_clause.startswith("WHERE "):
                raise ValueError("internal: a WHERE clause must start with 'WHERE '")
            return "WHERE (" + where_clause[6:] + ") AND " + test, [*params, actor.user_id]
        return "WHERE " + test, [actor.user_id]

    async def _foreign_count(
        self, internal: str, actor: DatastoreActor, where_clause: str = "", params: Any = (),
    ) -> int:
        """Rows matching *where_clause* that someone other than *actor* added."""
        if actor.kind != "member":
            return 0
        assert self._db is not None
        clause, args = self._scope_where(where_clause, params, actor, foreign=True)
        async with self._db.execute(f'SELECT COUNT(*) FROM "{internal}" {clause}', args) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def _require_no_foreign_rows(self, internal: str, actor: DatastoreActor) -> None:
        if actor.kind != "member":
            return
        if await self._foreign_count(internal, actor) > 0:
            raise MemberDeniedError(DS_FOREIGN_ROWS)

    async def _member_row_total(self, actor: DatastoreActor) -> int:
        """Rows *actor* added across the whole store (uses the creator index)."""
        if actor.kind != "member":
            return 0
        assert self._db is not None
        async with self._db.execute("SELECT name FROM _ds_tables") as cur:
            names = [r[0] for r in await cur.fetchall()]
        total = 0
        for name in names:
            internal = self._internal_name(name)
            if not await self._sys_cols_present(internal):
                continue      # no system columns → no member rows there
            async with self._db.execute(
                f'SELECT COUNT(*) FROM "{internal}" WHERE "_created_by" = ?', (actor.user_id,)
            ) as cur:
                row = await cur.fetchone()
            total += int(row[0]) if row else 0
        return total

    async def _stored_bytes(
        self, internal: str, cols: list[str], where_clause: str, params: Any,
    ) -> tuple[int, int]:
        """``(rows, bytes)`` *cols* take in the rows matching *where_clause*."""
        assert self._db is not None
        sums = "".join(", SUM(" + _OCTETS.format(f'"{c}"') + ")" for c in cols)
        async with self._db.execute(
            f'SELECT COUNT(*){sums} FROM "{internal}" {where_clause}', list(params)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return 0, 0
        return int(row[0] or 0), sum(int(v or 0) for v in row[1:])

    async def _member_bytes_left(self, actor: DatastoreActor) -> int | None:
        """Bytes *actor* may still store (None for the owner): their rows'
        values across the whole store, read through the creator index."""
        if actor.kind != "member":
            return None
        assert self._db is not None
        async with self._db.execute("SELECT name FROM _ds_tables") as cur:
            names = [r[0] for r in await cur.fetchall()]
        used = 0
        for name in names:
            internal = self._internal_name(name)
            if not await self._sys_cols_present(internal):
                continue      # no system columns → no member rows there
            cols = [c.name for c in await self._table_columns(name)]
            if cols:
                used += (await self._stored_bytes(
                    internal, cols, 'WHERE "_created_by" = ?', [actor.user_id]))[1]
        return MEMBER_MAX_STORED_BYTES - used

    @staticmethod
    def _member_row_bytes(values: Any) -> int:
        """Bytes a member's row (or SET) stores; refused when one value is
        over ``MEMBER_MAX_VALUE_BYTES``."""
        total = 0
        for v in values:
            size = _stored_size(v)
            if size > MEMBER_MAX_VALUE_BYTES:
                raise MemberDeniedError(DS_MEMBER_VALUE_TOO_LARGE)
            total += size
        return total

    async def _default_bytes(self, internal: str) -> dict[str, int]:
        """Column → bytes its DEFAULT stores in a row that leaves it out."""
        assert self._db is not None
        async with self._db.execute(f'PRAGMA table_info("{internal}")') as cur:
            info = await cur.fetchall()
        return {r[1]: _stored_size(str(r[4])) for r in info
                if r[4] is not None and r[1] not in SYSTEM_COLUMNS}

    async def _member_update_budget(
        self, actor: DatastoreActor, internal: str, new_values: dict[str, Any],
        where_clause: str, params: Any,
    ) -> None:
        """A member's UPDATE (scoped to their rows) must fit their stored bytes."""
        if actor.kind != "member":
            return
        per_row = self._member_row_bytes(new_values.values())
        clause, args = self._scope_where(where_clause, params, actor)
        n, old = await self._stored_bytes(internal, list(new_values), clause, args)
        grow = n * per_row - old
        if grow > 0 and grow > (await self._member_bytes_left(actor) or 0):
            raise MemberDeniedError(DS_MEMBER_STORAGE_LIMIT)

    async def _check_member_table_name(self, safe: str, actor: DatastoreActor) -> None:
        """A member's table is never named like a column, keyword, function,
        SQLite or internal table (raw_select must not rewrite those in someone
        else's SQL)."""
        if actor.kind != "member":
            return
        assert self._db is not None
        self._check_member_names(actor, safe)
        reserved = safe in _SQL_RESERVED or safe.startswith(_SQL_RESERVED_PREFIXES)
        if not reserved:
            async with self._db.execute(
                "SELECT 1 FROM _ds_columns WHERE col_name = ? LIMIT 1", (safe,)
            ) as cur:
                reserved = await cur.fetchone() is not None
        # The connection's functions and table-valued modules (json_each & co.).
        for pragma in ("pragma_function_list", "pragma_module_list"):
            if reserved:
                break
            with contextlib.suppress(Exception):
                async with self._db.execute(
                    f"SELECT 1 FROM {pragma} WHERE lower(name) = ? LIMIT 1", (safe,)
                ) as cur:
                    reserved = await cur.fetchone() is not None
        if reserved:
            raise MemberDeniedError(DS_MEMBER_TABLE_NAME)

    @staticmethod
    def _check_member_names(actor: DatastoreActor, *names: str) -> None:
        """A member's table / column names are at most MEMBER_MAX_NAME_CHARS."""
        if actor.kind == "member" and any(len(n) > MEMBER_MAX_NAME_CHARS for n in names):
            raise MemberDeniedError(DS_MEMBER_NAME_TOO_LONG)

    async def has_member_rows(self) -> bool:
        """Whether any table holds a row a member added (index range scan)."""
        await self._ensure_db()
        assert self._db is not None
        async with self._db.execute("SELECT name FROM _ds_tables") as cur:
            names = [r[0] for r in await cur.fetchall()]
        for internal in sorted(self._internal_name(n) for n in names):
            try:
                if not await self._sys_cols_present(internal):
                    continue
                async with self._db.execute(
                    f'SELECT 1 FROM "{internal}" WHERE "_created_by" > \'\' LIMIT 1'
                ) as cur:
                    if await cur.fetchone():
                        return True
            except Exception:
                continue
        return False

    # ── PR D: what members created, for the owner's agent (read-only) ──

    async def creator_summary(self, user_ids: Any) -> dict[str, dict]:
        """For every id: ``{"tables": [names they created, sorted], "rows":
        {table: n > 0}}`` — the rows counted through the creator index, only in
        tables that carry the PR C system columns. No write lock, no actor
        checks: read-only, for the owner's ``shared_agent_usage``."""
        ids = [str(u) for u in (user_ids or ()) if str(u or "")]
        out: dict[str, dict] = {u: {"tables": [], "rows": {}} for u in ids}
        if not ids:
            return out
        await self._ensure_db()
        assert self._db is not None
        marks = ",".join("?" * len(ids))
        async with self._db.execute(
            f"SELECT name, created_by FROM _ds_tables WHERE created_by IN ({marks}) ORDER BY name",
            ids,
        ) as cur:
            for name, created_by in await cur.fetchall():
                out[str(created_by)]["tables"].append(str(name))
        async with self._db.execute("SELECT name FROM _ds_tables ORDER BY name") as cur:
            names = [str(r[0]) for r in await cur.fetchall()]
        for name in names:
            internal = self._internal_name(name)
            try:
                if not await self._sys_cols_present(internal):
                    continue
                async with self._db.execute(
                    f'SELECT "_created_by", COUNT(*) FROM "{internal}" '
                    f'WHERE "_created_by" IN ({marks}) GROUP BY "_created_by"',
                    ids,
                ) as cur:
                    for uid, n in await cur.fetchall():
                        if int(n or 0) > 0 and str(uid) in out:
                            out[str(uid)]["rows"][name] = int(n)
            except Exception:
                continue
        return out

    async def rows_created_by(self, table: str, user_id: str, limit: int) -> tuple[list[dict], int]:
        """``(rows, total)``: the newest *limit* rows *user_id* added to
        *table* (system columns left out) and how many they added in all.
        Raises ``ValueError`` for an unknown table (read-only, no actor checks)."""
        safe, internal = await self._resolve_table(table)
        assert self._db is not None
        if not user_id or not await self._sys_cols_present(internal):
            return [], 0
        async with self._db.execute(
            f'SELECT COUNT(*) FROM "{internal}" WHERE "_created_by" = ?', (user_id,)
        ) as cur:
            row = await cur.fetchone()
        total = int(row[0]) if row else 0
        async with self._db.execute(
            f'SELECT * FROM "{internal}" WHERE "_created_by" = ? ORDER BY rowid DESC LIMIT ?',
            (user_id, max(1, int(limit))),
        ) as cur:
            cols = [d[0] for d in cur.description or ()]
            rows = await cur.fetchall()
        out = [{c: v for c, v in zip(cols, r) if c not in SYSTEM_COLUMNS} for r in rows]
        return out, total

    # ── table management ─────────────────────────────────────────────

    async def list_tables(self) -> list[TableInfo]:
        await self._ensure_db()
        assert self._db is not None
        async with self._db.execute(
            "SELECT name, created_at, updated_at, created_by, created_by_name "
            "FROM _ds_tables ORDER BY name"
        ) as cur:
            meta_rows = await cur.fetchall()
        tables: list[TableInfo] = []
        for name, created_at, updated_at, created_by, created_by_name in meta_rows:
            internal = self._internal_name(name)
            row_count = await self._row_count(internal)
            columns = await self._table_columns(name)
            tables.append(TableInfo(
                name=name, columns=columns, row_count=row_count,
                created_at=created_at, updated_at=updated_at,
                created_by=created_by or "", created_by_name=created_by_name or "",
            ))
        return tables

    async def describe_table(self, name: str) -> TableInfo:
        safe, internal = await self._resolve_table(name)
        columns = await self._table_columns(safe)
        row_count = await self._row_count(internal)
        assert self._db is not None
        async with self._db.execute(
            "SELECT created_at, updated_at, created_by, created_by_name "
            "FROM _ds_tables WHERE name = ?", (safe,)
        ) as cur:
            meta = await cur.fetchone()
        return TableInfo(
            name=safe, columns=columns, row_count=row_count,
            created_at=meta[0] if meta else "",
            updated_at=meta[1] if meta else "",
            created_by=(meta[2] or "") if meta else "",
            created_by_name=(meta[3] or "") if meta else "",
        )

    async def create_table(
        self, name: str, columns: list[dict[str, str]],
        unique: list[str] | None = None,
    ) -> TableInfo:
        actor = current_actor()
        async with self._writing():
            return await self._create_table(actor, name, columns, unique)

    async def _create_table(
        self, actor: DatastoreActor, name: str, columns: list[dict[str, str]],
        unique: list[str] | None,
    ) -> TableInfo:
        await self._ensure_db()
        assert self._db is not None
        cfg = get_config()

        existing = await self.list_tables()
        if actor.kind == "member":
            async with self._db.execute(
                "SELECT COUNT(*) FROM _ds_tables WHERE created_by = ?", (actor.user_id,)
            ) as cur:
                own = (await cur.fetchone())[0]
            if own >= MEMBER_MAX_TABLES:
                raise MemberDeniedError(DS_MEMBER_TABLE_LIMIT)
            if len(existing) >= cfg.datastore.max_tables - MEMBER_TABLE_HEADROOM:
                raise MemberDeniedError(DS_MEMBER_NO_ROOM)
        if len(existing) >= cfg.datastore.max_tables:
            raise ValueError(f"Table limit ({cfg.datastore.max_tables}) reached")

        safe = self._safe_name(name)
        internal = self._internal_name(safe)

        # Check for duplicate
        async with self._db.execute(
            "SELECT 1 FROM _ds_tables WHERE name = ?", (safe,)
        ) as cur:
            if await cur.fetchone():
                raise ValueError(f"Table already exists: {safe}")
        await self._check_member_table_name(safe, actor)

        if not columns:
            raise ValueError("At least one column required")
        if actor.kind == "member" and len(columns) > MEMBER_MAX_COLUMNS:
            raise MemberDeniedError(DS_MEMBER_TOO_MANY_COLUMNS)
        self._check_member_names(actor, *(
            self._safe_name(str(c.get("name", ""))) for c in columns if isinstance(c, dict)))

        # The creator columns sit right after _id (hidden: not in _ds_columns).
        col_defs: list[str] = [
            "_id INTEGER PRIMARY KEY AUTOINCREMENT",
            *(f'"{c}" TEXT NOT NULL DEFAULT \'\'' for c in SYSTEM_COLUMNS),
        ]
        col_objects: list[ColumnDef] = []
        seen: set[str] = set()

        for i, col in enumerate(columns):
            if not isinstance(col, dict):
                raise ValueError(
                    f"Column {i} must be an object with 'name' and 'type' keys "
                    f"(e.g. {{\"name\": \"title\", \"type\": \"text\"}}), got {type(col).__name__}: {col!r}"
                )
            col_name = self._safe_name(col.get("name", ""))
            col_type = (col.get("type", "text") or "text").lower().strip()
            if not col_name or col_name.startswith("_"):
                raise ValueError(f"Invalid column name: {col.get('name')}")
            if col_type not in VALID_TYPES:
                raise ValueError(f"Invalid type '{col_type}'. Valid: {sorted(VALID_TYPES)}")
            if col_name in seen:
                raise ValueError(f"Duplicate column: {col_name}")
            seen.add(col_name)
            col_defs.append(f'"{col_name}" {TYPE_MAP[col_type]}')
            col_objects.append(ColumnDef(name=col_name, col_type=col_type, position=i))

        # Optional UNIQUE key — the conflict target for upsert (idempotent writes,
        # so a resumed run updates a row instead of duplicating it).
        uniq_cols: list[str] = []
        if unique:
            for u in unique:
                us = self._safe_name(u)
                if us not in seen:
                    raise ValueError(f"unique column '{u}' is not one of the table's columns")
                if us not in uniq_cols:
                    uniq_cols.append(us)
        if uniq_cols:
            col_defs.append("UNIQUE (" + ", ".join(f'"{u}"' for u in uniq_cols) + ")")

        now = self._now()
        await self._db.execute(f'CREATE TABLE "{internal}" ({", ".join(col_defs)})')
        await self._db.execute(
            f'CREATE INDEX IF NOT EXISTS "ix_{internal}_created_by" ON "{internal}"("_created_by")')
        await self._db.execute(
            "INSERT INTO _ds_tables (name, created_at, updated_at, created_by, created_by_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (safe, now, now, actor.user_id, actor.name),
        )
        for c in col_objects:
            await self._db.execute(
                "INSERT INTO _ds_columns (table_name, col_name, col_type, position) "
                "VALUES (?, ?, ?, ?)",
                (safe, c.name, c.col_type, c.position),
            )
        await self._db.commit()
        self._sys_ok.add(internal)
        return TableInfo(name=safe, columns=col_objects, row_count=0, created_at=now, updated_at=now,
                         created_by=actor.user_id, created_by_name=actor.name)

    async def drop_table(self, name: str) -> bool:
        actor = current_actor()
        async with self._writing():
            safe, internal = await self._resolve_table(name)
            assert self._db is not None
            await self._check_table_protected(safe)
            if actor.kind == "member":
                await self._ensure_system_columns(internal, actor)
                await self._require_table_owner(safe, actor)
                await self._require_no_foreign_rows(internal, actor)
            await self._db.execute(f'DROP TABLE IF EXISTS "{internal}"')
            await self._db.execute("DELETE FROM _ds_columns WHERE table_name = ?", (safe,))
            await self._db.execute("DELETE FROM _ds_tables WHERE name = ?", (safe,))
            await self._db.commit()
            self._sys_ok.discard(internal)
            return True

    async def rename_table(self, old_name: str, new_name: str) -> TableInfo:
        """Rename a user table (both meta-data and the physical SQLite table)."""
        actor = current_actor()
        async with self._writing():
            await self._rename_table(actor, old_name, new_name)
        return await self.describe_table(self._safe_name(new_name))

    async def _rename_table(self, actor: DatastoreActor, old_name: str, new_name: str) -> None:
        old_safe, old_internal = await self._resolve_table(old_name)
        assert self._db is not None
        await self._check_table_protected(old_safe)
        if actor.kind == "member":
            await self._ensure_system_columns(old_internal, actor)
            await self._require_table_owner(old_safe, actor)
            await self._require_no_foreign_rows(old_internal, actor)

        new_safe = self._safe_name(new_name)
        if not new_safe:
            raise ValueError("Invalid new table name")
        if new_safe == old_safe:
            raise ValueError("New name is the same as the current name")

        # Check new name doesn't already exist
        async with self._db.execute(
            "SELECT 1 FROM _ds_tables WHERE name = ?", (new_safe,)
        ) as cur:
            if await cur.fetchone():
                raise ValueError(f"Table already exists: {new_safe}")
        await self._check_member_table_name(new_safe, actor)

        new_internal = self._internal_name(new_safe)
        now = self._now()

        # Temporarily disable foreign keys so we can update the parent
        # and children without constraint violations (no ON UPDATE CASCADE).
        await self._db.execute("PRAGMA foreign_keys=OFF")
        try:
            # Rename the physical SQLite table
            await self._db.execute(
                f'ALTER TABLE "{old_internal}" RENAME TO "{new_internal}"'
            )
            # SQLite keeps an index's name across a rename: move the creator
            # index to the new name, so a new table under the old name can
            # create its own.
            if old_internal in self._sys_ok:
                await self._db.execute(f'DROP INDEX IF EXISTS "ix_{old_internal}_created_by"')
                await self._db.execute(
                    f'CREATE INDEX IF NOT EXISTS "ix_{new_internal}_created_by" '
                    f'ON "{new_internal}"("_created_by")')
            # Update meta-tables (parent + children together)
            await self._db.execute(
                "UPDATE _ds_tables SET name = ?, updated_at = ? WHERE name = ?",
                (new_safe, now, old_safe),
            )
            await self._db.execute(
                "UPDATE _ds_columns SET table_name = ? WHERE table_name = ?",
                (new_safe, old_safe),
            )
            await self._db.execute(
                "UPDATE _ds_protections SET table_name = ? WHERE table_name = ?",
                (new_safe, old_safe),
            )
            await self._db.commit()
        finally:
            await self._db.execute("PRAGMA foreign_keys=ON")
        if old_internal in self._sys_ok:
            self._sys_ok.discard(old_internal)
            self._sys_ok.add(new_internal)

    # ── schema changes ───────────────────────────────────────────────

    async def add_column(
        self, table_name: str, col_name: str, col_type: str = "text",
        default: Any = None,
    ) -> bool:
        actor = current_actor()
        async with self._writing():
            return await self._add_column(actor, table_name, col_name, col_type, default)

    async def _add_column(
        self, actor: DatastoreActor, table_name: str, col_name: str, col_type: str,
        default: Any,
    ) -> bool:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)
        await self._require_table_owner(safe, actor)
        col_name = self._safe_name(col_name)
        col_type = col_type.lower().strip()
        if col_type not in VALID_TYPES:
            raise ValueError(f"Invalid type: {col_type}")
        if col_name.startswith("_"):
            raise ValueError(f"Column name cannot start with underscore: {col_name}")

        existing = await self._table_columns(safe)
        if any(c.name == col_name for c in existing):
            raise ValueError(f"Column already exists: {col_name}")
        if actor.kind == "member":
            if len(existing) >= MEMBER_MAX_COLUMNS:
                raise MemberDeniedError(DS_MEMBER_TOO_MANY_COLUMNS)
            self._check_member_names(actor, col_name)
            # A default is stored in every row that leaves the column out —
            # the rows already there too (read back as theirs, written out by
            # their next update): never in someone else's rows, and the
            # member's own must fit their stored bytes.
            size = self._member_row_bytes([default])
            if default is not None:
                await self._ensure_system_columns(internal, actor)
                await self._require_no_foreign_rows(internal, actor)
                if size * await self._row_count(internal) > (
                        await self._member_bytes_left(actor) or 0):
                    raise MemberDeniedError(DS_MEMBER_STORAGE_LIMIT)

        sqlite_type = TYPE_MAP[col_type]
        default_clause = ""
        if default is not None:
            default_clause = f" DEFAULT {self._quote_literal(default)}"
        await self._db.execute(
            f'ALTER TABLE "{internal}" ADD COLUMN "{col_name}" {sqlite_type}{default_clause}'
        )
        position = max((c.position for c in existing), default=-1) + 1
        await self._db.execute(
            "INSERT INTO _ds_columns (table_name, col_name, col_type, position) "
            "VALUES (?, ?, ?, ?)",
            (safe, col_name, col_type, position),
        )
        await self._db.execute(
            "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
            (self._now(), safe),
        )
        await self._db.commit()
        return True

    async def _require_own_clean_table(
        self, safe: str, internal: str, actor: DatastoreActor,
    ) -> None:
        """J4: a member renames / drops / retypes only a table they created,
        and only while nobody else has added rows to it."""
        if actor.kind != "member":
            return
        await self._ensure_system_columns(internal, actor)
        await self._require_table_owner(safe, actor)
        await self._require_no_foreign_rows(internal, actor)

    async def rename_column(self, table_name: str, old_name: str, new_name: str) -> bool:
        actor = current_actor()
        async with self._writing():
            return await self._rename_column(actor, table_name, old_name, new_name)

    async def _rename_column(
        self, actor: DatastoreActor, table_name: str, old_name: str, new_name: str,
    ) -> bool:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)
        await self._require_own_clean_table(safe, internal, actor)
        old_safe = self._safe_name(old_name)
        new_safe = self._safe_name(new_name)
        self._check_member_names(actor, new_safe)
        await self._check_column_protected(safe, old_safe)
        if new_safe.startswith("_"):
            raise ValueError(f"Column name cannot start with underscore: {new_safe}")

        existing = await self._table_columns(safe)
        if not any(c.name == old_safe for c in existing):
            raise ValueError(f"Column not found: {old_safe}")
        if any(c.name == new_safe for c in existing):
            raise ValueError(f"Column already exists: {new_safe}")

        await self._db.execute(
            f'ALTER TABLE "{internal}" RENAME COLUMN "{old_safe}" TO "{new_safe}"'
        )
        await self._db.execute(
            "UPDATE _ds_columns SET col_name = ? WHERE table_name = ? AND col_name = ?",
            (new_safe, safe, old_safe),
        )
        await self._db.execute(
            "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
            (self._now(), safe),
        )
        await self._db.commit()
        return True

    async def drop_column(self, table_name: str, col_name: str) -> bool:
        actor = current_actor()
        async with self._writing():
            return await self._drop_column(actor, table_name, col_name)

    async def _drop_column(self, actor: DatastoreActor, table_name: str, col_name: str) -> bool:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)
        await self._require_own_clean_table(safe, internal, actor)
        col_safe = self._safe_name(col_name)
        await self._check_column_protected(safe, col_safe)

        existing = await self._table_columns(safe)
        if not any(c.name == col_safe for c in existing):
            raise ValueError(f"Column not found: {col_safe}")
        if len(existing) <= 1:
            raise ValueError("Cannot drop the last column")

        await self._db.execute(f'ALTER TABLE "{internal}" DROP COLUMN "{col_safe}"')
        await self._db.execute(
            "DELETE FROM _ds_columns WHERE table_name = ? AND col_name = ?",
            (safe, col_safe),
        )
        await self._db.execute(
            "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
            (self._now(), safe),
        )
        await self._db.commit()
        return True

    async def change_column_type(
        self, table_name: str, col_name: str, new_type: str,
    ) -> bool:
        """Change a column's type via table rebuild with CAST."""
        actor = current_actor()
        async with self._writing():
            return await self._change_column_type(actor, table_name, col_name, new_type)

    async def _change_column_type(
        self, actor: DatastoreActor, table_name: str, col_name: str, new_type: str,
    ) -> bool:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)
        await self._require_own_clean_table(safe, internal, actor)
        col_safe = self._safe_name(col_name)
        await self._check_column_protected(safe, col_safe)
        new_type = new_type.lower().strip()
        if new_type not in VALID_TYPES:
            raise ValueError(f"Invalid type: {new_type}")

        existing = await self._table_columns(safe)
        target_col = next((c for c in existing if c.name == col_safe), None)
        if not target_col:
            raise ValueError(f"Column not found: {col_safe}")

        # Build new table schema. J17: the rebuild keeps the creator columns
        # (and their index) and the UNIQUE key upsert depends on.
        uniq = await self._unique_columns(internal)
        has_sys = await self._ensure_system_columns(internal, actor)
        tmp_internal = internal + "__tmp"
        col_defs = [
            "_id INTEGER PRIMARY KEY AUTOINCREMENT",
            *(f'"{c}" TEXT NOT NULL DEFAULT \'\'' for c in SYSTEM_COLUMNS),
        ]
        select_parts = ["_id"]
        select_parts += [f'"{c}"' if has_sys else "''" for c in SYSTEM_COLUMNS]
        for c in existing:
            if c.name == col_safe:
                sqlite_type = TYPE_MAP[new_type]
                col_defs.append(f'"{c.name}" {sqlite_type}')
                select_parts.append(f'CAST("{c.name}" AS {sqlite_type}) AS "{c.name}"')
            else:
                sqlite_type = TYPE_MAP.get(c.col_type, "TEXT")
                col_defs.append(f'"{c.name}" {sqlite_type}')
                select_parts.append(f'"{c.name}"')
        if uniq:
            col_defs.append("UNIQUE (" + ", ".join(f'"{u}"' for u in uniq) + ")")

        # One explicit transaction, so a failed rebuild rolls the DDL back
        # too (the legacy sqlite3 mode opens none for CREATE/DROP) and never
        # leaves "<internal>__tmp" behind to break every later call.
        # (A user table's name never holds "__", so the tmp name is ours.)
        await self._db.execute(f'DROP TABLE IF EXISTS "{tmp_internal}"')
        await self._db.commit()
        try:
            await self._db.execute("BEGIN")
            await self._db.execute(f'CREATE TABLE "{tmp_internal}" ({", ".join(col_defs)})')
            await self._db.execute(
                f'INSERT INTO "{tmp_internal}" SELECT {", ".join(select_parts)} FROM "{internal}"'
            )
            await self._db.execute(f'DROP TABLE "{internal}"')
            await self._db.execute(f'ALTER TABLE "{tmp_internal}" RENAME TO "{internal}"')
            await self._db.execute(
                f'CREATE INDEX IF NOT EXISTS "ix_{internal}_created_by" ON "{internal}"("_created_by")')
            await self._db.execute(
                "UPDATE _ds_columns SET col_type = ? WHERE table_name = ? AND col_name = ?",
                (new_type, safe, col_safe),
            )
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        except BaseException as exc:
            with contextlib.suppress(Exception):
                await self._db.rollback()
                await self._db.execute(f'DROP TABLE IF EXISTS "{tmp_internal}"')
                await self._db.commit()
            if isinstance(exc, sqlite3.IntegrityError) and "UNIQUE" in str(exc):
                raise ValueError(
                    f"converting {col_safe} would make values of the unique key collide; "
                    "the column was not changed") from None
            raise
        self._sys_ok.add(internal)
        return True

    # ── where clause builder ─────────────────────────────────────────

    def _build_where(
        self,
        where: dict[str, Any],
        valid_columns: set[str],
    ) -> tuple[str, list[Any]]:
        """Build a WHERE clause + params from a structured filter dict."""
        if not isinstance(where, dict):
            raise ValueError(
                f"'where' must be a JSON object like {{\"col\": \"value\"}} or "
                f"{{\"col\": {{\"op\": \">\", \"value\": 10}}}}, got {type(where).__name__}: {where!r}"
            )
        clauses: list[str] = []
        params: list[Any] = []

        for key, val in where.items():
            if key == "_all":
                continue
            if key == "_mine":
                # PR C: rows the caller added (the owner's are stamped '').
                if val is not True:
                    raise ValueError('"_mine" takes true')
                clauses.append('"_created_by" = ?')
                params.append(current_actor().user_id)
                continue
            # _id is the auto-generated primary key — allow it directly;
            # so are the creator columns (filter-only, never written by input).
            if key == "_id" or key in SYSTEM_COLUMNS:
                col = key
            else:
                col = self._safe_name(key)
                if col not in valid_columns:
                    raise ValueError(f"Unknown column in where: {key}")

            # A list of conditions on ONE column, ANDed together — this is how
            # a range is expressed: {"id": [{"op": ">=", "value": 370},
            # {"op": "<=", "value": 379}]}. A JSON object cannot carry two
            # "id" keys (the second silently wins), so without this form a
            # model asking for 370..379 gets everything up to 379 instead.
            if isinstance(val, list) and val and all(isinstance(v, dict) for v in val):
                for cond in val:
                    sub_clause, sub_params = self._build_where({key: cond}, valid_columns)
                    clauses.append(sub_clause[len("WHERE "):] if sub_clause.startswith("WHERE ") else sub_clause)
                    params.extend(sub_params)
                continue
            if isinstance(val, list):
                # Plain list of values → IN, the obvious reading.
                placeholders = ", ".join("?" for _ in val)
                clauses.append(f'"{col}" IN ({placeholders})')
                params.extend(val)
                continue

            if isinstance(val, dict):
                # Paired operators — {"op": [">=", "<="], "value": [241, 250]}.
                # Another way a model reaches for a range, and one that used to
                # dead-end at "Unsupported operator: ['>=', '<=']" while it kept
                # trying new phrasings of the same question.
                _raw_op = val.get("op", "=")
                if isinstance(_raw_op, list):
                    _vals = val.get("value")
                    _vals = _vals if isinstance(_vals, list) else [_vals] * len(_raw_op)
                    if len(_raw_op) != len(_vals):
                        raise ValueError(
                            f"'op' has {len(_raw_op)} operators but 'value' has "
                            f"{len(_vals)} values — they must pair up, e.g. "
                            '{"op": [">=", "<="], "value": [1, 10]}'
                        )
                    for _o, _v in zip(_raw_op, _vals):
                        sub_clause, sub_params = self._build_where(
                            {key: {"op": _o, "value": _v}}, valid_columns)
                        clauses.append(sub_clause[len("WHERE "):]
                                       if sub_clause.startswith("WHERE ") else sub_clause)
                        params.extend(sub_params)
                    continue

                op = str(_raw_op).upper().strip()
                # BETWEEN is what a model writes when it thinks in SQL. Accept
                # it as the two comparisons it stands for.
                if op in ("BETWEEN", "NOT BETWEEN"):
                    bounds = val.get("value")
                    if not isinstance(bounds, list) or len(bounds) != 2:
                        raise ValueError(
                            f"{op} needs exactly two values, e.g. "
                            '{"op": "BETWEEN", "value": [1, 10]}'
                        )
                    lo, hi = bounds
                    negate = op.startswith("NOT")
                    clauses.append(
                        f'"{col}" {"NOT " if negate else ""}BETWEEN ? AND ?')
                    params.extend([lo, hi])
                    continue
                if op not in _ALLOWED_OPS:
                    raise ValueError(
                        f"Unsupported operator: {op}. Allowed: "
                        f"{', '.join(sorted(_ALLOWED_OPS))}, BETWEEN. "
                        'For a range use one key with paired operators — '
                        '{"id": {"op": [">=", "<="], "value": [241, 250]}} — or '
                        '{"id": {"op": "BETWEEN", "value": [241, 250]}}.'
                    )
                if op in ("IS NULL", "IS NOT NULL"):
                    clauses.append(f'"{col}" {op}')
                elif op in ("IN", "NOT IN"):
                    values = val.get("value", [])
                    if not isinstance(values, list) or not values:
                        raise ValueError(f"IN operator requires a non-empty list")
                    placeholders = ", ".join("?" for _ in values)
                    clauses.append(f'"{col}" {op} ({placeholders})')
                    params.extend(values)
                else:
                    clauses.append(f'"{col}" {op} ?')
                    params.append(val.get("value"))
            else:
                clauses.append(f'"{col}" = ?')
                params.append(val)

        if not clauses:
            return "", params
        return "WHERE " + " AND ".join(clauses), params

    @staticmethod
    def _quote_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    # ── protection checks ────────────────────────────────────────────

    async def _check_table_protected(self, safe_name: str) -> None:
        """Raise ProtectedError if the table has table-level protection."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT reason FROM _ds_protections "
            "WHERE table_name = ? AND level = 'table'",
            (safe_name,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            reason = row[0] or "table is protected"
            raise ProtectedError(
                f"Table '{safe_name}' is protected: {reason}"
            )

    async def _check_column_protected(
        self, safe_name: str, col_name: str,
    ) -> None:
        """Raise ProtectedError if the column has column-level protection."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT reason FROM _ds_protections "
            "WHERE table_name = ? AND level = 'column' AND col_name = ?",
            (safe_name, col_name),
        ) as cur:
            row = await cur.fetchone()
        if row:
            reason = row[0] or "column is protected"
            raise ProtectedError(
                f"Column '{col_name}' in table '{safe_name}' is protected: {reason}"
            )

    async def _get_protected_row_ids(self, safe_name: str) -> set[int]:
        """Return set of row IDs with row-level protection."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT row_id FROM _ds_protections "
            "WHERE table_name = ? AND level = 'row' AND row_id IS NOT NULL",
            (safe_name,),
        ) as cur:
            rows = await cur.fetchall()
        return {r[0] for r in rows}

    async def _get_protected_cells(
        self, safe_name: str,
    ) -> dict[int, set[str]]:
        """Return {row_id: {col_name, ...}} for cell-level protections."""
        assert self._db is not None
        async with self._db.execute(
            "SELECT row_id, col_name FROM _ds_protections "
            "WHERE table_name = ? AND level = 'cell' "
            "AND row_id IS NOT NULL AND col_name IS NOT NULL",
            (safe_name,),
        ) as cur:
            rows = await cur.fetchall()
        result: dict[int, set[str]] = {}
        for row_id, col_name in rows:
            result.setdefault(row_id, set()).add(col_name)
        return result

    async def _resolve_affected_ids(
        self, internal: str, where: dict[str, Any],
        col_names: set[str],
    ) -> list[int]:
        """Get the _id values of rows matching a WHERE clause."""
        assert self._db is not None
        where_clause, where_params = self._build_where(where, col_names)
        sql = f'SELECT _id FROM "{internal}" {where_clause}'
        async with self._db.execute(sql, where_params) as cur:
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def _check_row_protection(
        self, safe_name: str, affected_ids: list[int],
    ) -> None:
        """Raise ProtectedError if any affected row is row-protected."""
        protected = await self._get_protected_row_ids(safe_name)
        blocked = protected & set(affected_ids)
        if blocked:
            ids_str = ", ".join(str(i) for i in sorted(blocked)[:5])
            raise ProtectedError(
                f"Row(s) {ids_str} in table '{safe_name}' are protected"
            )

    async def _check_cell_protection(
        self, safe_name: str, affected_ids: list[int],
        update_cols: set[str],
    ) -> None:
        """Raise ProtectedError if any affected cell is cell-protected."""
        cells = await self._get_protected_cells(safe_name)
        for rid in affected_ids:
            if rid in cells:
                overlap = cells[rid] & update_cols
                if overlap:
                    cols_str = ", ".join(sorted(overlap))
                    raise ProtectedError(
                        f"Cell(s) {cols_str} in row {rid} of table "
                        f"'{safe_name}' are protected"
                    )

    # ── protection CRUD ──────────────────────────────────────────────

    async def protect(
        self, table_name: str, level: str,
        row_id: int | None = None,
        col_name: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Add a protection rule. Returns the created protection dict."""
        actor = current_actor()
        async with self._writing():
            if actor.kind == "member":
                raise MemberDeniedError(DS_OWNER_ONLY)
            return await self._protect(table_name, level, row_id, col_name, reason)

    async def _protect(
        self, table_name: str, level: str, row_id: int | None,
        col_name: str | None, reason: str | None,
    ) -> dict[str, Any]:
        valid_levels = {"table", "column", "row", "cell"}
        if level not in valid_levels:
            raise ValueError(f"Invalid protection level: {level}. Valid: {sorted(valid_levels)}")

        safe, _ = await self._resolve_table(table_name)
        assert self._db is not None

        # Validate parameter combinations
        if level == "table":
            row_id = None
            col_name = None
        elif level == "column":
            row_id = None
            if not col_name:
                raise ValueError("col_name is required for column-level protection")
            col_name = self._safe_name(col_name)
            # Verify column exists
            columns = await self._table_columns(safe)
            if not any(c.name == col_name for c in columns):
                raise ValueError(f"Column not found: {col_name}")
        elif level == "row":
            if row_id is None:
                raise ValueError("row_id is required for row-level protection")
            col_name = None
        elif level == "cell":
            if row_id is None:
                raise ValueError("row_id is required for cell-level protection")
            if not col_name:
                raise ValueError("col_name is required for cell-level protection")
            col_name = self._safe_name(col_name)
            columns = await self._table_columns(safe)
            if not any(c.name == col_name for c in columns):
                raise ValueError(f"Column not found: {col_name}")

        now = self._now()
        try:
            await self._db.execute(
                "INSERT INTO _ds_protections "
                "(table_name, level, row_id, col_name, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (safe, level, row_id, col_name, reason, now),
            )
            await self._db.commit()
        except Exception as e:
            if "UNIQUE constraint" in str(e):
                raise ValueError("This protection already exists") from e
            raise

        return {
            "table_name": safe,
            "level": level,
            "row_id": row_id,
            "col_name": col_name,
            "reason": reason,
            "created_at": now,
        }

    async def unprotect(
        self, table_name: str, level: str,
        row_id: int | None = None,
        col_name: str | None = None,
    ) -> bool:
        """Remove a protection rule. Returns True if removed, False if not found."""
        actor = current_actor()
        async with self._writing():
            if actor.kind == "member":
                raise MemberDeniedError(DS_OWNER_ONLY)
            return await self._unprotect(table_name, level, row_id, col_name)

    async def _unprotect(
        self, table_name: str, level: str, row_id: int | None, col_name: str | None,
    ) -> bool:
        safe, _ = await self._resolve_table(table_name)
        assert self._db is not None

        if col_name:
            col_name = self._safe_name(col_name)

        # Normalize NULLs for matching
        if level == "table":
            row_id = None
            col_name = None
        elif level == "column":
            row_id = None
        elif level == "row":
            col_name = None

        cursor = await self._db.execute(
            "DELETE FROM _ds_protections "
            "WHERE table_name = ? AND level = ? "
            "AND row_id IS ? AND col_name IS ?",
            (safe, level, row_id, col_name),
        )
        removed = cursor.rowcount > 0
        if removed:
            await self._db.commit()
        return removed

    async def list_protections(
        self, table_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List protection rules, optionally filtered by table."""
        await self._ensure_db()
        assert self._db is not None

        if table_name:
            safe = self._safe_name(table_name)
            sql = (
                "SELECT id, table_name, level, row_id, col_name, reason, created_at "
                "FROM _ds_protections WHERE table_name = ? "
                "ORDER BY table_name, level, row_id, col_name"
            )
            params: tuple[Any, ...] = (safe,)
        else:
            sql = (
                "SELECT id, table_name, level, row_id, col_name, reason, created_at "
                "FROM _ds_protections "
                "ORDER BY table_name, level, row_id, col_name"
            )
            params = ()

        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()

        return [
            {
                "id": r[0],
                "table_name": r[1],
                "level": r[2],
                "row_id": r[3],
                "col_name": r[4],
                "reason": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]

    # ── data operations ──────────────────────────────────────────────

    async def insert_rows(
        self, table_name: str, rows: list[dict[str, Any]],
    ) -> int:
        actor = current_actor()
        async with self._writing():
            return await self._insert_rows(actor, table_name, rows)

    async def _member_row_caps(
        self, actor: DatastoreActor, internal: str, current_count: int, adding: int,
    ) -> None:
        """J5: a member's store-wide row budget and the owner's 10% of every table."""
        if actor.kind != "member":
            return
        if await self._member_row_total(actor) + adding > MEMBER_MAX_ROWS:
            raise MemberDeniedError(DS_MEMBER_ROW_LIMIT)
        cap = get_config().datastore.max_rows_per_table
        if current_count + adding > cap - cap // MEMBER_ROW_HEADROOM_DIVISOR:
            raise MemberDeniedError(DS_TABLE_NEARLY_FULL)

    async def _insert_rows(
        self, actor: DatastoreActor, table_name: str, rows: list[dict[str, Any]],
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)
        cfg = get_config()

        if not rows:
            return 0

        stamp = await self._ensure_system_columns(internal, actor)
        current_count = await self._row_count(internal)
        if current_count + len(rows) > cfg.datastore.max_rows_per_table:
            raise ValueError(
                f"Would exceed row limit ({cfg.datastore.max_rows_per_table}). "
                f"Current: {current_count}, inserting: {len(rows)}"
            )
        await self._member_row_caps(actor, internal, current_count, len(rows))
        # A member's stored bytes, counted row by row before each INSERT
        # (a refusal rolls back the whole call).
        left = await self._member_bytes_left(actor)
        defaults = await self._default_bytes(internal) if left is not None else {}

        columns = await self._table_columns(safe)
        col_names = {c.name for c in columns}

        inserted = 0
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Row {i} must be a JSON object mapping column names to values "
                    f"(e.g. {{\"name\": \"Alice\", \"age\": 30}}), got {type(row).__name__}: {row!r}"
                )
            # Filter to known columns only
            filtered = {self._safe_name(k): v for k, v in row.items() if self._safe_name(k) in col_names}
            if not filtered:
                continue
            col_list = list(filtered.keys())
            values = list(filtered.values())
            if left is not None:
                left -= self._member_row_bytes(values) + sum(
                    b for c, b in defaults.items() if c not in filtered)
                if left < 0:
                    raise MemberDeniedError(DS_MEMBER_STORAGE_LIMIT)
            if stamp:
                # The creator travels in the SAME statement as the row.
                col_list += list(SYSTEM_COLUMNS)
                values += [actor.user_id, actor.name]
            placeholders = ", ".join("?" for _ in col_list)
            col_clause = ", ".join(f'"{c}"' for c in col_list)
            await self._db.execute(
                f'INSERT INTO "{internal}" ({col_clause}) VALUES ({placeholders})',
                values,
            )
            inserted += 1

        if inserted:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return inserted

    async def _unique_columns(self, internal: str) -> list[str]:
        """Columns of the table's UNIQUE index — the upsert conflict target, if any."""
        assert self._db is not None
        async with self._db.execute(f'PRAGMA index_list("{internal}")') as cur:
            idx_rows = await cur.fetchall()
        for idx in idx_rows:
            # index_list row: (seq, name, unique, origin, partial)
            name, is_unique = idx[1], idx[2]
            origin = idx[3] if len(idx) > 3 else ""
            if not is_unique or origin == "pk":
                continue
            async with self._db.execute(f'PRAGMA index_info("{name}")') as cur2:
                info = await cur2.fetchall()
            cols = [r[2] for r in info if r[2] and not str(r[2]).startswith("_")]
            if cols:
                return cols
        return []

    async def upsert_rows(
        self, table_name: str, rows: list[dict[str, Any]],
        key_columns: list[str] | None = None,
    ) -> int:
        """Insert rows, but UPDATE an existing row on the table's UNIQUE key instead
        of duplicating it (idempotent — a resumed run re-running the same items just
        refreshes them). Needs a unique key: create the table with unique=[...], or
        pass key_columns. Returns the number of rows written."""
        actor = current_actor()
        async with self._writing():
            return await self._upsert_rows(actor, table_name, rows, key_columns)

    async def _upsert_rows(
        self, actor: DatastoreActor, table_name: str, rows: list[dict[str, Any]],
        key_columns: list[str] | None,
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)
        cfg = get_config()
        if not rows:
            return 0

        columns = await self._table_columns(safe)
        col_names = {c.name for c in columns}

        keys = [self._safe_name(k) for k in (key_columns or [])]
        keys = [k for k in keys if k in col_names]
        if not keys:
            keys = await self._unique_columns(internal)
        if not keys:
            raise ValueError(
                "upsert needs a unique key — create the table with a unique key "
                "(create_table with unique=[\"<col>\"]) or pass key_columns.")

        stamp = await self._ensure_system_columns(internal, actor)
        # Worst-case (all inserts) row-limit guard.
        current_count = await self._row_count(internal)
        if current_count + len(rows) > cfg.datastore.max_rows_per_table:
            raise ValueError(
                f"Would exceed row limit ({cfg.datastore.max_rows_per_table}). "
                f"Current: {current_count}, upserting: {len(rows)}")
        await self._member_row_caps(actor, internal, current_count, len(rows))

        # Validate every row first; a member's call is refused as a whole when
        # any row would overwrite a row someone else added (J3).
        prepared: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Row {i} must be a JSON object mapping column names to values, "
                    f"got {type(row).__name__}: {row!r}")
            filtered = {self._safe_name(k): v for k, v in row.items() if self._safe_name(k) in col_names}
            if not filtered:
                continue
            for k in keys:
                if k not in filtered:
                    raise ValueError(f"upsert row {i} is missing key column '{k}'")
            prepared.append(filtered)
        if actor.kind == "member":
            match = " AND ".join(f'"{k}" = ?' for k in keys)
            for filtered in prepared:
                async with self._db.execute(
                    f'SELECT "_created_by" FROM "{internal}" WHERE {match}',
                    [filtered[k] for k in keys],
                ) as cur:
                    found = await cur.fetchall()
                if any(str(r[0] or "") != actor.user_id for r in found):
                    raise MemberDeniedError(DS_NOT_YOUR_ROWS)

        left = await self._member_bytes_left(actor)
        defaults = await self._default_bytes(internal) if left is not None else {}
        written = 0
        for filtered in prepared:
            col_list = list(filtered.keys())
            values = list(filtered.values())
            conflict = ", ".join(f'"{k}"' for k in keys)
            update_cols = [c for c in col_list if c not in keys]
            if left is not None:
                # An insert stores the row; an update of the member's own
                # row stores the difference in its updated columns.
                grow = self._member_row_bytes(values)
                match = " AND ".join(f'"{k}" = ?' for k in keys)
                n, old = await self._stored_bytes(
                    internal, update_cols, f"WHERE {match}", [filtered[k] for k in keys])
                if n:
                    grow = self._member_row_bytes(filtered[c] for c in update_cols) - old
                else:
                    grow += sum(b for c, b in defaults.items() if c not in filtered)
                left -= grow
                if left < 0:
                    raise MemberDeniedError(DS_MEMBER_STORAGE_LIMIT)
            if stamp:
                col_list += list(SYSTEM_COLUMNS)
                values += [actor.user_id, actor.name]
            placeholders = ", ".join("?" for _ in col_list)
            col_clause = ", ".join(f'"{c}"' for c in col_list)
            if update_cols:
                # User columns only: a conflicting row keeps its creator.
                set_clause = ", ".join(f'"{c}" = excluded."{c}"' for c in update_cols)
                do = f"DO UPDATE SET {set_clause}"
                if actor.kind == "member":
                    do += ' WHERE "_created_by" = ?'
                    values.append(actor.user_id)
            else:
                do = "DO NOTHING"
            await self._db.execute(
                f'INSERT INTO "{internal}" ({col_clause}) VALUES ({placeholders}) '
                f'ON CONFLICT ({conflict}) {do}',
                values)
            written += 1

        if written:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe))
            await self._db.commit()
        return written

    async def update_rows(
        self, table_name: str,
        set_values: dict[str, Any],
        where: dict[str, Any] | None = None,
    ) -> int:
        actor = current_actor()
        async with self._writing():
            return await self._update_rows(actor, table_name, set_values, where)

    async def _update_rows(
        self, actor: DatastoreActor, table_name: str,
        set_values: dict[str, Any], where: dict[str, Any] | None,
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)

        if not isinstance(set_values, dict):
            raise ValueError(
                f"'set_values' must be a JSON object mapping column names to values "
                f"(e.g. {{\"status\": \"done\"}}), got {type(set_values).__name__}: {set_values!r}"
            )
        if not set_values:
            raise ValueError("set_values cannot be empty")

        columns = await self._table_columns(safe)
        col_names = {c.name for c in columns}

        # Determine which columns are being updated (safe names)
        update_col_set: set[str] = set()
        set_clauses: list[str] = []
        set_params: list[Any] = []
        for k, v in set_values.items():
            col = self._safe_name(k)
            if col not in col_names:
                raise ValueError(f"Unknown column: {k}")
            set_clauses.append(f'"{col}" = ?')
            set_params.append(v)
            update_col_set.add(col)

        where_clause = ""
        where_params: list[Any] = []
        if where:
            where_clause, where_params = self._build_where(where, col_names)

        if actor.kind == "member":
            # J3: the whole call is refused when it reaches a row someone
            # else added (no silent narrowing).
            await self._ensure_system_columns(internal, actor)
            if await self._foreign_count(internal, actor, where_clause, where_params) > 0:
                raise MemberDeniedError(DS_NOT_YOUR_ROWS)
            await self._member_update_budget(
                actor, internal, {self._safe_name(k): v for k, v in set_values.items()},
                where_clause, where_params)

        # Check row-level and cell-level protections
        if where:
            affected_ids = await self._resolve_affected_ids(
                internal, where, col_names,
            )
            if affected_ids:
                await self._check_row_protection(safe, affected_ids)
                await self._check_cell_protection(safe, affected_ids, update_col_set)

        where_clause, where_params = self._scope_where(where_clause, where_params, actor)
        sql = f'UPDATE "{internal}" SET {", ".join(set_clauses)} {where_clause}'
        cursor = await self._db.execute(sql, set_params + where_params)
        affected = cursor.rowcount
        if affected:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return affected

    async def update_column(
        self, table_name: str, col_name: str,
        value: Any = None, expression: str | None = None,
    ) -> int:
        actor = current_actor()
        async with self._writing():
            return await self._update_column(actor, table_name, col_name, value, expression)

    async def _update_column(
        self, actor: DatastoreActor, table_name: str, col_name: str,
        value: Any, expression: str | None,
    ) -> int:
        if actor.kind == "member" and expression:
            raise MemberDeniedError(DS_EXPRESSION_MEMBER)
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)
        if actor.kind == "member":
            await self._ensure_system_columns(internal, actor)
            await self._require_table_owner(safe, actor)
            await self._require_no_foreign_rows(internal, actor)
        col_safe = self._safe_name(col_name)
        await self._check_column_protected(safe, col_safe)

        columns = await self._table_columns(safe)
        if not any(c.name == col_safe for c in columns):
            raise ValueError(f"Column not found: {col_safe}")

        # Block if any rows are row-protected (update_column affects all rows)
        protected_rows = await self._get_protected_row_ids(safe)
        if protected_rows:
            ids_str = ", ".join(str(i) for i in sorted(protected_rows)[:5])
            raise ProtectedError(
                f"Cannot update entire column '{col_safe}': row(s) {ids_str} "
                f"in table '{safe}' are row-protected"
            )

        await self._member_update_budget(actor, internal, {col_safe: value}, "", [])
        scope, scope_params = self._scope_where("", [], actor)
        if expression:
            # Raw expression -- only allow simple math/string ops
            sql = f'UPDATE "{internal}" SET "{col_safe}" = {expression} {scope}'
            cursor = await self._db.execute(sql, scope_params)
        else:
            sql = f'UPDATE "{internal}" SET "{col_safe}" = ? {scope}'
            cursor = await self._db.execute(sql, [value, *scope_params])

        affected = cursor.rowcount
        if affected:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return affected

    async def delete_rows(
        self, table_name: str,
        where: dict[str, Any] | None = None,
    ) -> int:
        actor = current_actor()
        async with self._writing():
            return await self._delete_rows(actor, table_name, where)

    async def _delete_rows(
        self, actor: DatastoreActor, table_name: str, where: dict[str, Any] | None,
    ) -> int:
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        await self._check_table_protected(safe)

        columns = await self._table_columns(safe)
        col_names = {c.name for c in columns}

        if not where:
            raise ValueError("WHERE clause required. Pass {\"_all\": true} to delete all rows.")
        if not isinstance(where, dict):
            raise ValueError(
                f"'where' must be a JSON object like {{\"col\": \"value\"}} or "
                f"{{\"_all\": true}}, got {type(where).__name__}: {where!r}"
            )
        if where.get("_all") is True and where.get("_mine") is True:
            # {"_all": true, "_mine": true} means {"_mine": true}.
            where = {k: v for k, v in where.items() if k != "_all"}
        if actor.kind == "member":
            await self._ensure_system_columns(internal, actor)

        if where.get("_all") is True:
            # Check if any rows in the table are row-protected
            protected_rows = await self._get_protected_row_ids(safe)
            if protected_rows:
                ids_str = ", ".join(str(i) for i in sorted(protected_rows)[:5])
                raise ProtectedError(
                    f"Cannot delete all rows: row(s) {ids_str} "
                    f"in table '{safe}' are row-protected"
                )
            if await self._foreign_count(internal, actor) > 0:
                raise MemberDeniedError(DS_NOT_YOUR_ROWS)
            scope, scope_params = self._scope_where("", [], actor)
            cursor = await self._db.execute(f'DELETE FROM "{internal}" {scope}', scope_params)
        else:
            where_clause, where_params = self._build_where(where, col_names)
            if not where_clause:
                raise ValueError("Empty WHERE clause. Pass {\"_all\": true} to delete all rows.")
            if await self._foreign_count(internal, actor, where_clause, where_params) > 0:
                raise MemberDeniedError(DS_NOT_YOUR_ROWS)
            # Check row-level protections for targeted rows
            affected_ids = await self._resolve_affected_ids(
                internal, where, col_names,
            )
            if affected_ids:
                await self._check_row_protection(safe, affected_ids)
            where_clause, where_params = self._scope_where(where_clause, where_params, actor)
            cursor = await self._db.execute(f'DELETE FROM "{internal}" {where_clause}', where_params)

        affected = cursor.rowcount
        if affected:
            await self._db.execute(
                "UPDATE _ds_tables SET updated_at = ? WHERE name = ?",
                (self._now(), safe),
            )
            await self._db.commit()
        return affected

    # ── query operations ─────────────────────────────────────────────

    async def query(
        self, table_name: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        bypass_max: bool = False,
        *,
        include_creator: bool = False,
    ) -> dict[str, Any]:
        """Rows of one table. ``include_creator`` (PR C) adds ``"creators"``
        — one wire ``Creator`` per row, aligned with ``rows`` — without
        changing ``columns``/``rows`` (the creator columns appear there only
        when the caller names them in ``columns``)."""
        safe, internal = await self._resolve_table(table_name)
        assert self._db is not None
        cfg = get_config()

        table_cols = await self._table_columns(safe)
        col_names = {c.name for c in table_cols}
        passthrough = ("_id", *SYSTEM_COLUMNS)

        # Select clause
        if columns:
            select_cols = []
            for c in columns:
                c_safe = c if c in passthrough else self._safe_name(c)
                if c_safe not in col_names and c_safe not in passthrough:
                    raise ValueError(f"Unknown column: {c}")
                select_cols.append(f'"{c_safe}"')
            select_clause = ", ".join(select_cols)
            result_col_names = [(c if c in passthrough else self._safe_name(c)) for c in columns]
        else:
            select_clause = '"_id", ' + ", ".join(f'"{c.name}"' for c in table_cols)
            result_col_names = ["_id"] + [c.name for c in table_cols]
        # Reads never ALTER: a table without creator columns reads as the owner's.
        creator_cols = include_creator and await self._sys_cols_present(internal)
        if creator_cols:
            select_clause += ", " + ", ".join(f'"{c}"' for c in SYSTEM_COLUMNS)

        # Where
        where_clause = ""
        where_params: list[Any] = []
        if where:
            where_clause, where_params = self._build_where(where, col_names)

        # Order by
        order_clause = ""
        if order_by:
            parts = []
            for ob in order_by:
                ob = str(ob).strip()
                if ob.startswith("-"):
                    raw_col = ob[1:]
                    direction = "DESC"
                else:
                    raw_col = ob
                    direction = "ASC"
                # _id is the auto-generated primary key — pass through directly
                # (so are the creator columns).
                col = raw_col if raw_col in passthrough else self._safe_name(raw_col)
                parts.append(f'"{col}" {direction}')
            order_clause = "ORDER BY " + ", ".join(parts)

        # Limit
        if bypass_max:
            effective_limit = limit if limit is not None else cfg.datastore.max_query_rows
        else:
            effective_limit = min(limit or cfg.datastore.max_query_rows, cfg.datastore.max_query_rows)

        sql = (
            f'SELECT {select_clause} FROM "{internal}" '
            f'{where_clause} {order_clause} '
            f'LIMIT {effective_limit} OFFSET {offset}'
        )

        async with self._db.execute(sql, where_params) as cur:
            rows = await cur.fetchall()

        # Total count (for pagination info)
        count_sql = f'SELECT COUNT(*) FROM "{internal}" {where_clause}'
        async with self._db.execute(count_sql, where_params) as cur:
            total_row = await cur.fetchone()
            total = total_row[0] if total_row else 0

        out_rows = [list(r) for r in rows]
        result: dict[str, Any] = {
            "columns": result_col_names,
            "rows": out_rows,
            "total": total,
            "offset": offset,
            "limit": effective_limit,
        }
        if include_creator:
            if creator_cols:
                result["creators"] = [creator_dict(r[-2], r[-1]) for r in out_rows]
                result["rows"] = [r[:-2] for r in out_rows]
            else:
                result["creators"] = [creator_dict("", "") for _ in out_rows]
        return result

    async def raw_select(self, sql: str) -> dict[str, Any]:
        """Execute a read-only SQL query. Only SELECT is allowed.

        PR C: the creator columns are dropped from the result unless the SQL
        names them (J17). A member's statement (J19) runs on a separate
        read-only connection, at most ``MEMBER_SQL_DEADLINE_S`` and
        ``max_query_rows`` rows, without ``WITH RECURSIVE``.
        """
        await self._ensure_db()
        assert self._db is not None
        cfg = get_config()
        from captain_claw import speaker

        member = speaker.current() is not None

        stripped = sql.strip()
        # Validate it's a SELECT
        if not re.match(r"^SELECT\b", stripped, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed")

        # Block obvious mutation attempts
        danger = re.search(
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH)\b",
            stripped, re.IGNORECASE,
        )
        if danger:
            raise ValueError(f"Mutation keyword '{danger.group()}' not allowed in raw SELECT")
        if member and re.search(r"\bRECURSIVE\b", sql, re.IGNORECASE):
            raise ValueError(DS_SQL_LIMITS)
        if member and re.search(r"\bpragma_\w+", sql, re.IGNORECASE):
            # pragma_database_list & co. name host paths — plain SELECTs only.
            raise ValueError(DS_SQL_LIMITS)

        # Replace user table names with internal names.
        # Users may reference tables as-is; we need to add the ds_ prefix.
        async with self._db.execute("SELECT name, created_by FROM _ds_tables") as cur:
            known_tables = [(r[0], r[1] or "") for r in await cur.fetchall()]

        processed = stripped
        # Comments blanked, for spotting a CTE named like a member's table.
        uncommented = re.sub(r"/\*.*?\*/|--[^\n]*", " ", stripped, flags=re.DOTALL)
        for tbl, created_by in sorted(known_tables, key=lambda t: len(t[0]), reverse=True):
            internal = self._internal_name(tbl)
            name = re.escape(tbl)
            cte = (rf'\b{name}["`\]]?\s*(\([^()]*\))?\s*AS\s*(NOT\s+)?'
                   rf'(MATERIALIZED\s*)?\(')
            if not created_by:
                # Replace table name when it appears as a word boundary
                processed = re.sub(rf'\b{name}\b', f'"{internal}"', processed)
            elif not any(re.search(cte, text, re.IGNORECASE)
                         for text in (processed, uncommented)):
                # A member's table only where it is one — right after FROM or
                # JOIN, outside a '...' literal — so it never rewrites a
                # column, function, keyword, string (or a CTE of that name)
                # in someone else's SQL.
                parts = re.split(r"('(?:[^']|'')*')", processed)
                parts[::2] = [re.sub(
                    rf'\b(FROM|JOIN)\b(\s*)(["`]?){name}\3(?!\w)',
                    lambda m, i=internal: f'{m.group(1)}{m.group(2) or " "}"{i}"',
                    code, flags=re.IGNORECASE,
                ) for code in parts[::2]]
                processed = "".join(parts)

        # Enforce LIMIT
        max_rows = cfg.datastore.max_query_rows
        if not re.search(r"\bLIMIT\b", processed, re.IGNORECASE):
            processed = processed.rstrip(";") + f" LIMIT {max_rows}"

        if member:
            col_names, rows = await self._member_select(processed, max_rows)
        else:
            async with self._db.execute(processed) as cur:
                if cur.description:
                    col_names = [d[0] for d in cur.description]
                else:
                    col_names = []
                rows = await cur.fetchall()

        out_rows = [list(r) for r in rows]
        if not re.search(r"_created_by", sql, re.IGNORECASE):
            hidden = [i for i, c in enumerate(col_names) if c in SYSTEM_COLUMNS]
            if hidden:
                col_names = [c for i, c in enumerate(col_names) if i not in hidden]
                out_rows = [[v for i, v in enumerate(r) if i not in hidden] for r in out_rows]

        return {
            "columns": col_names,
            "rows": out_rows,
            "total": len(out_rows),
        }

    async def _member_select(self, processed: str, max_rows: int) -> tuple[list[str], list[Any]]:
        """Run a member's SELECT on the read-only connection, interrupted at
        the deadline; ``ValueError(DS_SQL_LIMITS)`` when it runs over."""
        async with self._ro_lock:
            if self._ro_db is None:
                uri = "file:" + urllib.parse.quote(str(self.db_path)) + "?mode=ro"
                ro = await aiosqlite.connect(uri, uri=True)
                await ro.execute("PRAGMA query_only=ON")
                with contextlib.suppress(Exception):
                    # One value (zeroblob(1e9), printf padding, …) can't
                    # allocate more than this on the member connection.
                    await ro._execute(
                        ro._conn.setlimit, sqlite3.SQLITE_LIMIT_LENGTH, _MEMBER_SQL_VALUE_MAX)
                self._ro_db = ro
            ro = self._ro_db
            self._ro_deadline = time.monotonic() + MEMBER_SQL_DEADLINE_S
            await ro.set_progress_handler(
                lambda: 1 if time.monotonic() > self._ro_deadline else 0, 10_000)
            try:
                async with ro.execute(processed) as cur:
                    col_names = [d[0] for d in cur.description] if cur.description else []
                    rows: list[Any] = []
                    size = 0
                    while len(rows) < max_rows:
                        batch = await cur.fetchmany(min(50, max_rows - len(rows)))
                        if not batch:
                            break
                        rows.extend(batch)
                        size += sum(_value_size(v) for r in batch for v in r)
                        if size > _MEMBER_SQL_RESULT_MAX or time.monotonic() > self._ro_deadline:
                            raise ValueError(DS_SQL_LIMITS)
            except (sqlite3.OperationalError, sqlite3.DataError) as exc:
                text = str(exc).lower()
                if "interrupted" in text or "too big" in text:
                    raise ValueError(DS_SQL_LIMITS) from None
                raise
            finally:
                await ro.set_progress_handler(None, 0)
        return col_names, rows

    # ── import helpers (upload flow) ─────────────────────────────────

    async def parse_headers(self, file_path: Path, file_type: str) -> list[str]:
        """Extract normalised column headers from a CSV or XLSX file.

        Returns safe-named header strings (fast — reads only the first row).
        """
        if file_type == "csv":
            with open(file_path, encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                raw = next(reader, None)
            if not raw:
                raise ValueError("CSV file has no headers")
            return [self._safe_name(h) for h in raw if h.strip()]
        if file_type == "xlsx":
            headers, _ = self._parse_xlsx(file_path)
            if not headers:
                raise ValueError("XLSX file has no data")
            return [self._safe_name(h) for h in headers if h.strip()]
        raise ValueError(f"Unsupported file type: {file_type}")

    async def find_matching_table(
        self, file_headers: list[str], file_stem: str,
    ) -> dict[str, Any] | None:
        """Find an existing table that matches the file by name or column overlap.

        Uses Jaccard similarity on column names plus a bonus for exact name
        match.  Returns ``None`` if no table exceeds the threshold.
        """
        tables = await self.list_tables()
        if not tables:
            return None

        safe_stem = self._safe_name(file_stem)
        file_set = set(file_headers)
        best: dict[str, Any] | None = None
        best_score = 0.0

        for t in tables:
            table_cols = {c.name for c in t.columns}
            intersection = file_set & table_cols
            union = file_set | table_cols
            jaccard = len(intersection) / len(union) if union else 0.0

            is_name_match = t.name == safe_stem
            score = jaccard + (0.3 if is_name_match else 0.0)

            if score > best_score:
                best_score = score
                best = {
                    "name": t.name,
                    "match_type": "exact_name" if is_name_match else "column_overlap",
                    "score": round(min(jaccard, 1.0), 3),
                    "matched_columns": sorted(intersection),
                    "unmatched_file_cols": sorted(file_set - table_cols),
                    "unmatched_table_cols": sorted(table_cols - file_set),
                    "table_columns": [c.name for c in t.columns],
                    "row_count": t.row_count,
                }

        if best:
            if best["match_type"] == "exact_name" and best["score"] > 0:
                return best
            if best["score"] >= 0.7:
                return best
        return None

    async def import_to_existing_table(
        self,
        file_path: Path,
        file_type: str,
        table_name: str,
        column_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Import file data into an existing table.

        *column_mapping* maps ``{file_safe_col: table_col}``.  If ``None``
        an identity mapping on matching safe names is used.
        """
        if file_type == "csv":
            text = file_path.read_text(encoding="utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            if not reader.fieldnames:
                raise ValueError("CSV file has no headers")
            raw_headers = list(reader.fieldnames)
            all_rows_raw: list[dict[str, Any]] = list(reader)
        elif file_type == "xlsx":
            headers_raw, data_rows = self._parse_xlsx(file_path)
            raw_headers = headers_raw
            all_rows_raw = [
                {headers_raw[i]: v for i, v in enumerate(row) if i < len(headers_raw)}
                for row in data_rows
            ]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Build orig→safe lookup
        safe_map: dict[str, str] = {}  # safe_name → original header
        for h in raw_headers:
            safe_map[self._safe_name(h)] = h

        # Resolve column mapping (identity by default)
        if column_mapping is None:
            # Map every safe file col that exists in the target table
            info = await self.describe_table(table_name)
            if info is None:
                raise ValueError(f"Table '{table_name}' not found")
            table_col_set = {c.name for c in info.columns}
            column_mapping = {sc: sc for sc in safe_map if sc in table_col_set}

        warnings: list[str] = []
        skipped = [sc for sc in safe_map if sc not in column_mapping]
        if skipped:
            warnings.append(f"Skipped file columns not in table: {', '.join(skipped)}")

        rows_to_insert: list[dict[str, Any]] = []
        for row in all_rows_raw:
            cleaned: dict[str, Any] = {}
            for safe_col, orig_col in safe_map.items():
                if safe_col in column_mapping:
                    target = column_mapping[safe_col]
                    val = row.get(orig_col)
                    cleaned[target] = self._coerce_value(val)
            if cleaned:
                rows_to_insert.append(cleaned)

        inserted = await self.insert_rows(table_name, rows_to_insert)
        return {
            "table": table_name,
            "rows_imported": inserted,
            "columns": sorted(column_mapping.values()),
            "warnings": warnings,
        }

    # ── import / export ──────────────────────────────────────────────

    async def _member_import_budget(self, actor: DatastoreActor, file_path: Path) -> int | None:
        """J19: a member's import source is size-capped, and its rows must fit
        their remaining row budget — both checked before anything is created.
        The budget (rows) for a member, None for the owner."""
        if actor.kind != "member":
            return None
        if file_path.stat().st_size > MEMBER_IMPORT_MAX_BYTES:
            raise MemberDeniedError(DS_IMPORT_TOO_LARGE)
        await self._ensure_db()
        return MEMBER_MAX_ROWS - await self._member_row_total(actor)

    async def _member_import_bytes(
        self, actor: DatastoreActor, headers: list[str], rows: Any,
    ) -> None:
        """A member's import must fit the column cap and their stored bytes —
        measured on the parsed values (an xlsx shared string counts in every
        cell it fills), before anything is created."""
        if actor.kind != "member":
            return
        if len(headers) > MEMBER_MAX_COLUMNS:
            raise MemberDeniedError(DS_MEMBER_TOO_MANY_COLUMNS)
        left = await self._member_bytes_left(actor) or 0
        for values in rows:
            left -= self._member_row_bytes(values)
            if left < 0:
                raise MemberDeniedError(DS_MEMBER_STORAGE_LIMIT)

    async def _import_rows(
        self, actor: DatastoreActor, safe: str, col_defs: list[dict[str, str]] | None,
        rows: list[dict[str, Any]], exists_hint: str,
    ) -> int:
        """Create the table (``col_defs``, unless appending) and insert *rows*
        under one write lock; a member's refused insert drops the table it
        just created, so nothing of the import is left."""
        async with self._writing():
            created = False
            try:
                if col_defs is not None:
                    try:
                        await self.create_table(safe, col_defs)
                    except ValueError as e:
                        if "already exists" in str(e):
                            raise ValueError(exists_hint) from e
                        raise
                    created = True
                return await self.insert_rows(safe, rows)
            except BaseException:
                if created and actor.kind == "member" and self._db is not None:
                    with contextlib.suppress(Exception):
                        await self._db.rollback()
                        await self.drop_table(safe)
                raise

    async def import_csv(
        self, file_path: Path, table_name: str | None = None,
        append: bool = False,
    ) -> dict[str, Any]:
        actor = current_actor()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        budget = await self._member_import_budget(actor, file_path)

        text = file_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV has no headers")

        headers = self._dedup_headers([self._safe_name(h) for h in reader.fieldnames])
        if budget is None:
            all_rows = list(reader)
        else:
            all_rows = []
            for row in reader:
                all_rows.append(row)
                if len(all_rows) > budget:
                    raise MemberDeniedError(DS_MEMBER_ROW_LIMIT)
        await self._member_import_bytes(
            actor, headers, ([v for k, v in r.items() if k is not None] for r in all_rows))

        if not table_name:
            table_name = file_path.stem

        safe = self._safe_name(table_name)

        # Infer types from data
        col_defs = None if append else self._infer_column_types(headers, all_rows)

        # Build row dicts with safe (deduped) column names
        # Map original fieldnames (in order) to deduped headers
        orig_to_dedup = dict(zip(reader.fieldnames, headers))
        rows_to_insert = []
        for row in all_rows:
            cleaned = {}
            for orig_key, val in row.items():
                safe_key = orig_to_dedup.get(orig_key, self._safe_name(orig_key))
                cleaned[safe_key] = self._coerce_value(val)
            rows_to_insert.append(cleaned)

        inserted = await self._import_rows(
            actor, safe, col_defs, rows_to_insert,
            f"Table '{safe}' already exists. Use append=true to add data, "
            f"or drop the table first.")
        return {"table": safe, "rows_imported": inserted, "columns": headers}

    async def import_xlsx(
        self, file_path: Path, table_name: str | None = None,
        sheet_name: str | None = None, append: bool = False,
    ) -> dict[str, Any]:
        actor = current_actor()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        budget = await self._member_import_budget(actor, file_path)
        if budget is not None:
            try:
                with zipfile.ZipFile(file_path) as zf:
                    unzipped = sum(i.file_size for i in zf.infolist())
            except zipfile.BadZipFile:
                raise ValueError("That isn't a readable .xlsx file") from None
            if unzipped > MEMBER_IMPORT_MAX_UNZIPPED_BYTES:
                raise MemberDeniedError(DS_IMPORT_TOO_LARGE)

        headers, all_rows = self._parse_xlsx(file_path, sheet_name)
        if not headers:
            raise ValueError("XLSX sheet has no data")
        if budget is not None and len(all_rows) > budget:
            raise MemberDeniedError(DS_MEMBER_ROW_LIMIT)
        await self._member_import_bytes(
            actor, headers, (row[:len(headers)] for row in all_rows))

        safe_headers = self._dedup_headers([self._safe_name(h) for h in headers])
        if not table_name:
            table_name = file_path.stem
        safe = self._safe_name(table_name)

        col_defs = None
        if not append:
            col_defs = self._infer_column_types(
                safe_headers,
                [{safe_headers[i]: v for i, v in enumerate(row) if i < len(safe_headers)} for row in all_rows[:100]],
            )

        rows_to_insert = []
        for row in all_rows:
            cleaned = {}
            for i, val in enumerate(row):
                if i < len(safe_headers):
                    cleaned[safe_headers[i]] = self._coerce_value(val)
            rows_to_insert.append(cleaned)

        inserted = await self._import_rows(
            actor, safe, col_defs, rows_to_insert,
            f"Table '{safe}' already exists. Use append=true to add data.")
        return {"table": safe, "rows_imported": inserted, "columns": safe_headers}

    async def _export_rows(
        self, table_name: str, columns: list[str] | None, where: dict[str, Any] | None,
        neutralize: str,
    ) -> tuple[list[str], list[list[Any]]]:
        """Columns and rows of a table export; ``neutralize`` (``"all"`` /
        ``"members"``) defuses formula-leading text cells (J13)."""
        cfg = get_config()
        mode = neutralize if neutralize in ("all", "members") else "none"
        result = await self.query(
            table_name, columns=columns, where=where,
            limit=cfg.datastore.max_export_rows,
            bypass_max=True,
            include_creator=mode != "none",
        )
        rows = neutralize_rows(result["rows"], result.get("creators"), mode)
        return result["columns"], rows

    async def export_csv(
        self, table_name: str, output_path: Path,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        *,
        neutralize: str = "none",
    ) -> Path:
        cols, rows = await self._export_rows(table_name, columns, where, neutralize)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        return output_path

    async def export_xlsx(
        self, table_name: str, output_path: Path,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        *,
        neutralize: str = "none",
    ) -> Path:
        cols, rows = await self._export_rows(table_name, columns, where, neutralize)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_xlsx(output_path, cols, rows)
        return output_path

    async def export_json(
        self, table_name: str, output_path: Path,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        *,
        neutralize: str = "none",   # accepted for symmetry; JSON has no formulas
    ) -> Path:
        cfg = get_config()
        result = await self.query(
            table_name, columns=columns, where=where,
            limit=cfg.datastore.max_export_rows,
            bypass_max=True,
        )
        cols = result["columns"]
        rows = [dict(zip(cols, row)) for row in result["rows"]]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
        return output_path

    async def export_sql_csv(self, sql: str, output_path: Path) -> Path:
        """Export a raw SELECT query result to CSV."""
        result = await self.raw_select(sql)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(result["columns"])
            writer.writerows(result["rows"])
        return output_path

    async def export_sql_xlsx(self, sql: str, output_path: Path) -> Path:
        """Export a raw SELECT query result to XLSX."""
        result = await self.raw_select(sql)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_xlsx(output_path, result["columns"], result["rows"])
        return output_path

    async def export_sql_json(self, sql: str, output_path: Path) -> Path:
        """Export a raw SELECT query result to JSON."""
        result = await self.raw_select(sql)
        cols = result["columns"]
        rows = [dict(zip(cols, row)) for row in result["rows"]]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
        return output_path

    # ── summary for context injection ────────────────────────────────

    async def get_tables_summary(self) -> list[TableInfo]:
        return await self.list_tables()

    # ── internal: CSV/XLSX helpers ───────────────────────────────────

    @staticmethod
    def _coerce_value(val: Any) -> Any:
        """Coerce a string value to the most appropriate Python type."""
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return None
        if isinstance(val, str):
            v = val.strip()
            if v.lower() in ("true", "yes"):
                return 1
            if v.lower() in ("false", "no"):
                return 0
            try:
                return int(v)
            except ValueError:
                pass
            try:
                return float(v)
            except ValueError:
                pass
        return val

    @staticmethod
    def _infer_column_types(
        headers: list[str], sample_rows: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Infer column types from sample data."""
        col_defs: list[dict[str, str]] = []
        for header in headers:
            values = [r.get(header) for r in sample_rows[:100] if r.get(header) is not None and str(r.get(header)).strip() != ""]
            col_type = "text"
            if values:
                all_int = all(_looks_int(v) for v in values)
                all_float = all(_looks_float(v) for v in values)
                all_bool = all(_looks_bool(v) for v in values)
                if all_bool:
                    col_type = "boolean"
                elif all_int:
                    col_type = "integer"
                elif all_float:
                    col_type = "real"
            col_defs.append({"name": header, "type": col_type})
        return col_defs

    @staticmethod
    def _parse_xlsx(
        file_path: Path, sheet_name: str | None = None,
    ) -> tuple[list[str], list[list[Any]]]:
        """Parse XLSX into headers + rows using ZIP/XML (no external deps)."""
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

        with zipfile.ZipFile(file_path) as zf:
            # Parse shared strings
            shared: list[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                ss_tree = ET.parse(zf.open("xl/sharedStrings.xml"))
                for si in ss_tree.findall(f"{{{ns}}}si"):
                    parts = []
                    for t_elem in si.iter(f"{{{ns}}}t"):
                        if t_elem.text:
                            parts.append(t_elem.text)
                    shared.append("".join(parts))

            # Find sheet
            wb_tree = ET.parse(zf.open("xl/workbook.xml"))
            sheets_el = wb_tree.findall(f"{{{ns}}}sheets/{{{ns}}}sheet")
            if not sheets_el:
                return [], []

            target_sheet = None
            if sheet_name:
                for s in sheets_el:
                    if s.get("name", "").lower() == sheet_name.lower():
                        target_sheet = s
                        break
                if not target_sheet:
                    raise ValueError(f"Sheet not found: {sheet_name}")
            else:
                target_sheet = sheets_el[0]

            # Resolve sheet path via relationships
            rel_tree = ET.parse(zf.open("xl/_rels/workbook.xml.rels"))
            rid = target_sheet.get(f"{{{ns_rel}}}id", "")
            sheet_path = None
            for rel in rel_tree.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
                if rel.get("Id") == rid:
                    sheet_path = "xl/" + rel.get("Target", "")
                    break

            if not sheet_path or sheet_path not in zf.namelist():
                # Fallback: try first worksheet
                candidates = [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet")]
                if not candidates:
                    return [], []
                sheet_path = sorted(candidates)[0]

            sheet_tree = ET.parse(zf.open(sheet_path))
            rows_data: list[list[Any]] = []

            for row_el in sheet_tree.findall(f"{{{ns}}}sheetData/{{{ns}}}row"):
                row_vals: dict[int, Any] = {}
                for cell in row_el.findall(f"{{{ns}}}c"):
                    ref = cell.get("r", "")
                    col_idx = _col_ref_to_index(ref)
                    cell_type = cell.get("t", "")
                    val_el = cell.find(f"{{{ns}}}v")
                    val_text = val_el.text if val_el is not None and val_el.text else ""

                    if cell_type == "s" and val_text:
                        idx = int(val_text)
                        value = shared[idx] if idx < len(shared) else ""
                    elif cell_type == "b":
                        value = val_text
                    else:
                        value = val_text
                    row_vals[col_idx] = value

                if row_vals:
                    max_col = max(row_vals.keys())
                    row_list = [row_vals.get(i, "") for i in range(max_col + 1)]
                    rows_data.append(row_list)

        if not rows_data:
            return [], []

        headers = [str(v) for v in rows_data[0]]
        data_rows = rows_data[1:]
        return headers, data_rows

    @staticmethod
    def _write_xlsx(
        output_path: Path, columns: list[str], rows: list[list[Any]],
    ) -> None:
        """Build a minimal XLSX file from columns + rows using ZIP/XML."""
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        ns_r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        ns_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
        ns_pkg = "http://schemas.openxmlformats.org/package/2006/relationships"

        # Collect all unique strings for shared strings table
        all_strings: list[str] = []
        string_index: dict[str, int] = {}
        for col in columns:
            s = str(col)
            if s not in string_index:
                string_index[s] = len(all_strings)
                all_strings.append(s)
        for row in rows:
            for val in row:
                if isinstance(val, str):
                    if val not in string_index:
                        string_index[val] = len(all_strings)
                        all_strings.append(val)

        # Build shared strings XML
        ss_root = ET.Element("sst", xmlns=ns, count=str(len(all_strings)), uniqueCount=str(len(all_strings)))
        for s in all_strings:
            si = ET.SubElement(ss_root, "si")
            t = ET.SubElement(si, "t")
            t.text = s

        # Build worksheet XML
        ws_root = ET.Element("worksheet", xmlns=ns)
        sd = ET.SubElement(ws_root, "sheetData")

        # Header row
        header_row = ET.SubElement(sd, "row", r="1")
        for ci, col_name in enumerate(columns):
            ref = _index_to_col_ref(ci) + "1"
            c = ET.SubElement(header_row, "c", r=ref, t="s")
            v = ET.SubElement(c, "v")
            v.text = str(string_index[str(col_name)])

        # Data rows
        for ri, row in enumerate(rows, start=2):
            row_el = ET.SubElement(sd, "row", r=str(ri))
            for ci, val in enumerate(row):
                ref = _index_to_col_ref(ci) + str(ri)
                if val is None:
                    continue
                if isinstance(val, str):
                    c = ET.SubElement(row_el, "c", r=ref, t="s")
                    v = ET.SubElement(c, "v")
                    v.text = str(string_index[val])
                elif isinstance(val, (int, float)):
                    c = ET.SubElement(row_el, "c", r=ref)
                    v = ET.SubElement(c, "v")
                    v.text = str(val)
                else:
                    s = str(val)
                    if s not in string_index:
                        string_index[s] = len(all_strings)
                        all_strings.append(s)
                    c = ET.SubElement(row_el, "c", r=ref, t="s")
                    v = ET.SubElement(c, "v")
                    v.text = str(string_index[s])

        # Build workbook XML
        wb_root = ET.Element("workbook", xmlns=ns)
        wb_root.set(f"xmlns:r", ns_r)
        sheets = ET.SubElement(wb_root, "sheets")
        sheet = ET.SubElement(sheets, "sheet", name="Sheet1", sheetId="1")
        sheet.set("r:id", "rId1")

        # Build content types
        ct_root = ET.Element("Types", xmlns=ns_ct)
        ET.SubElement(ct_root, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
        ET.SubElement(ct_root, "Default", Extension="xml", ContentType="application/xml")
        ET.SubElement(ct_root, "Override", PartName="/xl/workbook.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
        ET.SubElement(ct_root, "Override", PartName="/xl/worksheets/sheet1.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
        ET.SubElement(ct_root, "Override", PartName="/xl/sharedStrings.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml")

        # Build rels
        root_rels = ET.Element("Relationships", xmlns=ns_pkg)
        ET.SubElement(root_rels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", Target="xl/workbook.xml")

        wb_rels = ET.Element("Relationships", xmlns=ns_pkg)
        ET.SubElement(wb_rels, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet", Target="worksheets/sheet1.xml")
        ET.SubElement(wb_rels, "Relationship", Id="rId2", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings", Target="sharedStrings.xml")

        # Write ZIP
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", _et_to_string(ct_root))
            zf.writestr("_rels/.rels", _et_to_string(root_rels))
            zf.writestr("xl/workbook.xml", _et_to_string(wb_root))
            zf.writestr("xl/_rels/workbook.xml.rels", _et_to_string(wb_rels))
            zf.writestr("xl/worksheets/sheet1.xml", _et_to_string(ws_root))
            zf.writestr("xl/sharedStrings.xml", _et_to_string(ss_root))


# ── module-level helpers ─────────────────────────────────────────────

def _col_ref_to_index(ref: str) -> int:
    """Convert Excel column reference (e.g. 'AB3') to 0-based index."""
    col = "".join(c for c in ref if c.isalpha()).upper()
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _index_to_col_ref(idx: int) -> str:
    """Convert 0-based index to Excel column reference (e.g. 0 -> 'A')."""
    result = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        result = chr(rem + ord("A")) + result
    return result


def _et_to_string(element: ET.Element) -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + ET.tostring(
        element, encoding="unicode",
    )


def _looks_int(val: Any) -> bool:
    try:
        s = str(val).strip()
        int(s)
        return "." not in s
    except (ValueError, TypeError):
        return False


def _looks_float(val: Any) -> bool:
    try:
        float(str(val).strip())
        return True
    except (ValueError, TypeError):
        return False


def _looks_bool(val: Any) -> bool:
    return str(val).strip().lower() in ("true", "false", "yes", "no", "1", "0")


# ── singleton / session-scoped managers ──────────────────────────────

_manager: DatastoreManager | None = None
_session_managers: dict[str, DatastoreManager] = {}
_vfs_managers: dict[str, DatastoreManager] = {}
_path_managers: dict[str, DatastoreManager] = {}


def get_datastore_manager_at(db_path: Path | str) -> DatastoreManager:
    """Return a datastore manager for an EXPLICIT db file path, cached by path.

    Used by the Flight Deck server to read a folder-bound store under a specific
    user's VFS root (it can't rely on the ambient env that workers use)."""
    key = str(Path(db_path).expanduser().resolve())
    if key in _path_managers:
        return _path_managers[key]
    mgr = DatastoreManager(db_path=Path(db_path))
    _path_managers[key] = mgr
    return mgr


def get_datastore_manager() -> DatastoreManager:
    """Return the global (shared) datastore manager."""
    global _manager
    if _manager is None:
        _manager = DatastoreManager()
    return _manager


def get_vfs_datastore_manager(project: str, *, create: bool = True) -> DatastoreManager | None:
    """Return a datastore manager whose DB lives INSIDE a shared VFS project
    folder (``vfs:<project>/.datastore/store.db``).

    Every agent bound to that project (e.g. all workers in a Basna/Vatra run
    with the shared-datastore option on) resolves to the SAME relational store,
    so they can collaborate through tables instead of each keeping a private DB.
    Cached per project. Falls back to the global store if ``project`` is empty.

    ``create=False`` is for READ-ONLY access to ANOTHER run's datastore (a
    reference/prior-knowledge folder): it returns ``None`` when that folder has
    no datastore, instead of creating an empty one.

    PR C (J16): never for a shared-agent member — the cache is keyed by
    folder name only, so it can't tell whose VFS a folder belongs to.
    """
    from captain_claw import speaker

    if speaker.current() is not None:
        raise PermissionError(DS_PROJECT_MEMBER)
    key = (project or "").strip()
    if not key:
        return get_datastore_manager()
    if key in _vfs_managers:
        return _vfs_managers[key]
    from captain_claw.vfs import project_root
    db_path = project_root(key, create=create) / ".datastore" / "store.db"
    if not create and not db_path.is_file():
        return None
    mgr = DatastoreManager(db_path=db_path)
    _vfs_managers[key] = mgr
    return mgr


async def close_vfs_datastore_managers() -> None:
    """Close all VFS-folder-scoped datastore managers."""
    for mgr in _vfs_managers.values():
        await mgr.close()
    _vfs_managers.clear()


def resolve_datastore_manager(session_id: str | None = None) -> DatastoreManager:
    """The datastore an agent's tools read/write: the shared VFS-folder store when
    a run binds ``CLAW_DATASTORE_VFS``, else the public-computer per-session store,
    else the global store. Used by BOTH the datastore tool and the completion-gate
    verifier so they never disagree about which database a save landed in (a
    mismatch made the verifier report false 'save didn't persist' failures).

    A shared-agent member always gets the global store (PR C, J16) — the one
    their ``datastore`` calls use."""
    from captain_claw import speaker

    if speaker.current() is not None:
        return get_datastore_manager()
    vfs_project = os.environ.get("CLAW_DATASTORE_VFS", "").strip()
    if vfs_project:
        return get_vfs_datastore_manager(vfs_project)
    if get_config().web.public_run == "computer" and session_id:
        return get_session_datastore_manager(str(session_id))
    return get_datastore_manager()


def get_session_datastore_manager(session_id: str) -> DatastoreManager:
    """Return a per-session datastore manager (separate DB file).

    Used in public computer mode so that each session's tables are
    fully isolated from every other session.
    """
    key = session_id.strip()
    if key in _session_managers:
        return _session_managers[key]
    cfg = get_config()
    base = Path(cfg.datastore.path).expanduser().parent  # e.g. ~/.captain-claw/
    db_path = base / "datastore_sessions" / f"datastore_{key}.db"
    mgr = DatastoreManager(db_path=db_path)
    _session_managers[key] = mgr
    return mgr


async def close_session_datastore_managers() -> None:
    """Close all session-scoped datastore managers."""
    for mgr in _session_managers.values():
        await mgr.close()
    _session_managers.clear()
