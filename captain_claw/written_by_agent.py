"""Files this agent wrote, remembered across restarts.

The ``write`` tool won't replace a file holding unrelated content the agent
didn't write unless asked (overwrite=true, and then it keeps a copy beside it
— safety_guard setting ``keep_foreign_files``). This module answers "did the
agent write this file?".

A file counts as the agent's own while it still holds exactly what the agent
last wrote to it: a SHA-256 per resolved path, in a small SQLite file next to
the session DB. A file someone else changed since (the person, another
program) is theirs again, so the guard protects their edits too. The record
used to be a set in memory, so after a restart (idle sleep, an update) the
agent's own earlier files counted as someone else's: an unrelated rewrite of
yesterday's report was refused once and left a ``.previous`` copy behind.

Every function never raises. When the record can't be read, a file is not
the agent's own: the safe side, where the guard asks before replacing it.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

DB_NAME = "written_by_agent.db"
# Where the record lives instead of next to the session DB (tests/conftest.py
# points it at a temporary folder, so a test never writes into a real home).
DB_PATH_OVERRIDE: Path | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS written (
  path TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  at TEXT NOT NULL);
"""

_LOCK = threading.Lock()
_CONNS: dict[str, sqlite3.Connection] = {}


def db_path() -> Path:
    """Next to the session DB, like the other per-agent records."""
    if DB_PATH_OVERRIDE is not None:
        return Path(DB_PATH_OVERRIDE)
    from captain_claw.config import get_config

    return Path(get_config().session.path).expanduser().parent / DB_NAME


def _key(path: Any) -> str:
    return str(Path(path).expanduser().resolve())


def _digest(path: Any) -> str | None:
    try:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except OSError:
        return None


def _connect_locked() -> sqlite3.Connection:
    """The cached connection for the current :func:`db_path` (caller holds _LOCK)."""
    path = db_path()
    conn = _CONNS.get(str(path))
    if conn is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        _CONNS[str(path)] = conn
    return conn


def note(path: Any) -> None:
    """Remember the file, as it is on disk now, as the agent's own."""
    digest = _digest(path)
    if digest is None:
        return
    try:
        with _LOCK:
            conn = _connect_locked()
            conn.execute(
                "INSERT INTO written (path, sha256, at) VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(path) DO UPDATE SET sha256 = excluded.sha256, at = excluded.at",
                (_key(path), digest),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 — never raises
        log.debug("Could not record a file the agent wrote", error=str(exc))


def is_own(path: Any) -> bool:
    """Whether the file still holds exactly what this agent last wrote to it."""
    try:
        with _LOCK:
            row = _connect_locked().execute(
                "SELECT sha256 FROM written WHERE path = ?", (_key(path),)).fetchone()
    except Exception as exc:  # noqa: BLE001 — never raises
        log.debug("Could not read the record of files the agent wrote", error=str(exc))
        return False
    return bool(row) and row[0] == _digest(path)


def forget(path: Any) -> None:
    """Drop the file from the record: it is no longer the agent's own."""
    try:
        with _LOCK:
            conn = _connect_locked()
            conn.execute("DELETE FROM written WHERE path = ?", (_key(path),))
            conn.commit()
    except Exception as exc:  # noqa: BLE001 — never raises
        log.debug("Could not update the record of files the agent wrote", error=str(exc))
