"""An agent's SQLite stores in and out of a bundle.

Out: :func:`snapshot` takes a consistent copy with SQLite's backup API. Agents
run their stores in WAL mode, so copying ``insights.db`` as a file would miss
whatever still sits in ``insights.db-wal``; the backup API reads through the
WAL like any other connection, and works while the agent is running. The copy
is then trimmed to what the export carries (:func:`keep_only`) and vacuumed, so
rows it dropped don't linger in free pages.

In, for a new agent: the snapshot becomes the store after :func:`check` (an
integrity check, and nothing but tables, indexes, full-text tables and their
triggers in it). Into an existing agent's store, rows are copied by column
name (see the merge module), never the file.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

_FTS_SHADOWS = ("_data", "_idx", "_content", "_docsize", "_config")


class StoreError(ValueError):
    """A bundled store that can't be used."""


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def snapshot(src: Path, dst: Path) -> None:
    """A consistent copy of the live store *src* at *dst* (a single file, no WAL)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with closing(sqlite3.connect(str(src), timeout=30)) as source, \
            closing(sqlite3.connect(str(dst))) as target:
        source.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")
        target.commit()


def tables(db: Path) -> list[str]:
    """The store's own tables (not SQLite's, not a full-text table's shadow tables)."""
    with closing(_open_ro(db)) as c:
        return _tables(c)


def _tables(c: sqlite3.Connection) -> list[str]:
    rows = c.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    virtual = {n for n, sql in rows if str(sql or "").upper().startswith("CREATE VIRTUAL TABLE")}
    shadows = {v + s for v in virtual for s in _FTS_SHADOWS}
    return [n for n, _ in rows if not n.startswith("sqlite_") and n not in virtual and n not in shadows]


def _open_ro(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        c.execute("PRAGMA trusted_schema=OFF")
    except sqlite3.Error:
        pass
    return c


def counts(db: Path) -> dict[str, int]:
    """Rows per table."""
    with closing(_open_ro(db)) as c:
        return {t: int(c.execute(f"SELECT count(*) FROM {_q(t)}").fetchone()[0]) for t in _tables(c)}


def keep_only(db: Path, keep: set[str], *, blank: dict[str, tuple[str, ...]] | None = None) -> None:
    """Empty every table of the snapshot *db* but *keep* (their full-text
    indexes follow through the triggers), blank the *blank* columns, then
    vacuum so nothing dropped stays in the file."""
    with closing(sqlite3.connect(str(db))) as c:
        for t in _tables(c):
            if t not in keep:
                c.execute(f"DELETE FROM {_q(t)}")
        for t, cols in (blank or {}).items():
            if t not in keep or not _has_table(c, t):
                continue
            have = {r[1] for r in c.execute(f"PRAGMA table_info({_q(t)})")}
            for col in cols:
                if col in have:
                    c.execute(f"UPDATE {_q(t)} SET {_q(col)} = NULL WHERE {_q(col)} IS NOT NULL")
        c.commit()
        c.execute("VACUUM")


def _has_table(c: sqlite3.Connection, name: str) -> bool:
    return c.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def check(db: Path) -> None:
    """Refuse a bundled store that is not a sound SQLite database of tables:
    it fails SQLite's integrity check, or holds a view or a virtual table other
    than a full-text one (an agent's stores have none of those)."""
    try:
        with closing(_open_ro(db)) as c:
            ok = c.execute("PRAGMA integrity_check").fetchone()
            if not ok or ok[0] != "ok":
                raise StoreError(f"{db.name} is damaged ({ok[0] if ok else 'no answer'})")
            for kind, name, sql in c.execute("SELECT type, name, sql FROM sqlite_master"):
                text = " ".join(str(sql or "").split()).upper()
                if kind == "view":
                    raise StoreError(f"{db.name} holds a view ({name})")
                if kind == "table" and text.startswith("CREATE VIRTUAL TABLE") and \
                        " USING FTS5" not in text and " USING FTS5VOCAB" not in text:
                    raise StoreError(f"{db.name} holds a virtual table that isn't a search index ({name})")
    except sqlite3.DatabaseError as exc:
        raise StoreError(f"{db.name} is not a database the agent can use ({exc})") from exc
