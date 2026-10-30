"""The tool-call register's store: every tool call Flight Deck's agents report.

A dedicated SQLite DB (``tool_audit.db``) in Flight Deck's data dir, deck-
isolated via ``FD_DATA_DIR`` like ``payments.db``: it grows with every call an
agent makes, so it keeps its writes and its pruning away from the main DB.

Rows are append-only. Nothing edits who called what, when, with what — the
only change a row ever gets is its outcome, once, from ``running`` to how the
call ended (with the ids an ok send or draft named, ``refs``). Each row is
chained to the one before it with an HMAC over those
fields (key: ``tool_audit.key`` beside the DB, never in it), and its outcome
carries an HMAC of its own, so :meth:`ToolAuditStore.verify` finds a row that
was changed, slipped in or taken out by anyone without the key. Agents number
their records (``boot`` + ``seq``), so a record an agent made that never
arrived shows up as a gap.

Retention deletes the oldest rows; the chain then starts from an anchor (the
last deleted row's hash), and every prune is written to ``audit_events``.

``owner_id`` / ``speaker_id`` have no foreign key: rows outlive a deleted user
until retention removes them, and an auth-off deck's owner isn't a ``users`` row.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

import aiosqlite

DEFAULT_RETENTION_DAYS = 90
MIN_RETENTION_DAYS = 7
MAX_RETENTION_DAYS = 3650
GAP_SETTLE_S = 120          # rows this fresh may still have earlier ones in flight
GENESIS = "0" * 64

_DDL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id      TEXT NOT NULL DEFAULT '',
    member        INTEGER NOT NULL DEFAULT 0,
    speaker_id    TEXT NOT NULL DEFAULT '',
    speaker_label TEXT NOT NULL DEFAULT '',
    agent_slug    TEXT NOT NULL,
    agent_name    TEXT NOT NULL DEFAULT '',
    agent_ref     TEXT NOT NULL DEFAULT '',
    boot          TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    started_at    REAL NOT NULL,
    received_at   REAL NOT NULL,
    tool          TEXT NOT NULL,
    risk          TEXT NOT NULL DEFAULT 'other',
    params        TEXT NOT NULL DEFAULT '{}',
    session_id    TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT '',
    source_label  TEXT NOT NULL DEFAULT '',
    outcome       TEXT NOT NULL DEFAULT 'running',
    reason        TEXT NOT NULL DEFAULT '',
    duration_ms   INTEGER,
    ended_at      REAL,
    prev_hash     TEXT NOT NULL,
    hash          TEXT NOT NULL,
    end_mac       TEXT NOT NULL DEFAULT '',
    private       INTEGER NOT NULL DEFAULT 0,
    turn_id       TEXT NOT NULL DEFAULT '',
    refs          TEXT NOT NULL DEFAULT '{}',
    UNIQUE (agent_slug, boot, seq)
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_owner ON tool_calls(owner_id, id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_speaker ON tool_calls(speaker_id, id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent ON tool_calls(agent_slug, id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_started ON tool_calls(started_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_received ON tool_calls(received_at);
CREATE TABLE IF NOT EXISTS audit_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    at     REAL NOT NULL,
    kind   TEXT NOT NULL,
    actor  TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '{}'
);
"""

# The fields the chain covers, in order — everything but the outcome.
_CHAIN_FIELDS = ("owner_id", "member", "speaker_id", "speaker_label", "agent_slug", "agent_name",
                 "agent_ref", "boot", "seq", "started_at", "received_at", "tool", "risk", "params",
                 "session_id", "source", "source_label")
# Fields added later: chained only when set, so rows recorded before they
# existed still verify (a flag set or cleared afterwards still shows).
# ``private``: made while the agent's turn held members' private data — the
# agent never reads it back (tool_audit_digest). ``turn_id``: the agent's turn
# the call was made in (turn_register.py), what the Taskboard ties it to.
_LATE_CHAIN_FIELDS = ("private", "turn_id")
_END_FIELDS = ("outcome", "reason", "duration_ms", "ended_at")
# Settled with the outcome, under its MAC only when set: ``refs``, the ids an
# ok send, draft or calendar call's result named (tool_audit.result_refs).
_LATE_END_FIELDS = ("refs",)
_LATE_COLUMNS = {"private": "INTEGER NOT NULL DEFAULT 0", "turn_id": "TEXT NOT NULL DEFAULT ''",
                 "refs": "TEXT NOT NULL DEFAULT '{}'"}
_COLUMNS = (("id",) + _CHAIN_FIELDS + _LATE_CHAIN_FIELDS + _END_FIELDS + _LATE_END_FIELDS
            + ("prev_hash", "hash", "end_mac"))


_INT_FIELDS = frozenset({"member", "seq", "duration_ms", "private"})
_FLOAT_FIELDS = frozenset({"started_at", "received_at", "ended_at"})
_NULLABLE = frozenset({"duration_ms", "ended_at"})
_EMPTY = {"refs": "{}"}       # a late field's value when unset


def _norm(field: str, value: Any) -> Any:
    """A value as SQLite gives it back, so a row hashes the same when read."""
    if value is None and field in _NULLABLE:
        return None
    if field in _INT_FIELDS:
        return int(value or 0)
    if field in _FLOAT_FIELDS:
        return float(value or 0.0)
    if not value and field in _EMPTY:
        return _EMPTY[field]
    return "" if value is None else str(value)


def _set(field: str, value: Any) -> bool:
    """Is a late field set (so its MAC covers it)?"""
    return bool(value) and value != _EMPTY.get(field)


def _canon(values: list[Any]) -> bytes:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _load_key(path: Path) -> bytes:
    """The register's HMAC key, made on first use (0600)."""
    try:
        raw = path.read_text().strip()
        if len(raw) >= 64:
            return bytes.fromhex(raw)
    except (OSError, ValueError):
        pass
    key = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key.hex())
    return key


class ToolAuditStore:
    def __init__(self, db_path: Path | str, key_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path = Path(key_path) if key_path else self.db_path.with_suffix(".key")
        self._key: bytes | None = None
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()   # one writer: the chain is a sequence

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is not None:
            return self._db
        db = await aiosqlite.connect(str(self.db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.executescript(_DDL)
        async with db.execute("PRAGMA table_info(tool_calls)") as cur:
            have = {str(r[1]) for r in await cur.fetchall()}
        for col, decl in _LATE_COLUMNS.items():
            if col not in have:
                await db.execute(f"ALTER TABLE tool_calls ADD COLUMN {col} {decl}")
        # After the columns: a register from before turn_id has no column to index yet.
        await db.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_turn ON tool_calls(turn_id)")
        await db.commit()
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass
        self._key = _load_key(self.key_path)
        self._db = db
        return db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ── MACs ─────────────────────────────────────────────────────────

    def _row_mac(self, prev_hash: str, row: dict[str, Any]) -> str:
        assert self._key is not None
        values = [row.get(f) for f in _CHAIN_FIELDS]
        values += [[f, row[f]] for f in _LATE_CHAIN_FIELDS if _set(f, row.get(f))]
        msg = prev_hash.encode("ascii") + b"\n" + _canon(values)
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def _end_mac(self, row_hash: str, row: dict[str, Any]) -> str:
        assert self._key is not None
        values = [row.get(f) for f in _END_FIELDS]
        values += [[f, row[f]] for f in _LATE_END_FIELDS if _set(f, row.get(f))]
        msg = row_hash.encode("ascii") + b"\n" + _canon(values)
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    # ── meta ─────────────────────────────────────────────────────────

    async def _meta(self, key: str, default: str = "") -> str:
        db = await self._ensure_db()
        async with db.execute("SELECT value FROM audit_meta WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
        return str(row[0]) if row else default

    async def _set_meta(self, db: aiosqlite.Connection, key: str, value: str) -> None:
        await db.execute("INSERT INTO audit_meta (key, value) VALUES (?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    async def _last_hash(self, db: aiosqlite.Connection) -> str:
        async with db.execute("SELECT hash FROM tool_calls ORDER BY id DESC LIMIT 1") as cur:
            row = await cur.fetchone()
        if row:
            return str(row[0])
        return await self._meta("anchor_hash", GENESIS)

    async def add_event(self, kind: str, actor: str = "", detail: dict | None = None) -> None:
        db = await self._ensure_db()
        await db.execute("INSERT INTO audit_events (at, kind, actor, detail) VALUES (?, ?, ?, ?)",
                         (time.time(), kind, actor, json.dumps(detail or {})))
        await db.commit()

    async def events(self, limit: int = 50) -> list[dict]:
        db = await self._ensure_db()
        async with db.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,)) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d.get("detail") or "{}")
            except ValueError:
                d["detail"] = {}
            out.append(d)
        return out

    # ── settings ─────────────────────────────────────────────────────

    async def retention_days(self) -> int:
        raw = await self._meta("retention_days", "")
        try:
            return max(MIN_RETENTION_DAYS, min(MAX_RETENTION_DAYS, int(raw))) if raw else DEFAULT_RETENTION_DAYS
        except ValueError:
            return DEFAULT_RETENTION_DAYS

    async def set_retention_days(self, days: int, actor: str = "") -> int:
        days = max(MIN_RETENTION_DAYS, min(MAX_RETENTION_DAYS, int(days)))
        before = await self.retention_days()
        db = await self._ensure_db()
        await self._set_meta(db, "retention_days", str(days))
        await db.commit()
        if days != before:
            await self.add_event("retention_changed", actor, {"from": before, "to": days})
        return days

    # ── writes ───────────────────────────────────────────────────────

    async def record(self, records: list[dict[str, Any]]) -> int:
        """Store an agent's batch. Each record is already identified and
        cleaned by the route (owner, agent, risk, masked params). A ``start``
        or ``full`` record is a new row (a repeat is ignored); an ``end``
        record settles a ``running`` row once. Returns rows written."""
        written = 0
        async with self._lock:
            db = await self._ensure_db()
            prev = await self._last_hash(db)
            now = time.time()
            for rec in records:
                key = (rec["agent_slug"], rec["boot"], int(rec["seq"]))
                async with db.execute(
                    "SELECT id, hash, outcome FROM tool_calls WHERE agent_slug=? AND boot=? AND seq=?",
                    key,
                ) as cur:
                    existing = await cur.fetchone()
                ended = rec.get("phase") in ("full", "end")
                end_fields = _END_FIELDS + _LATE_END_FIELDS
                if existing is not None:
                    if ended and existing["outcome"] == "running":
                        end = {f: _norm(f, rec.get(f)) for f in end_fields}
                        mac = self._end_mac(str(existing["hash"]), end)
                        await db.execute(
                            "UPDATE tool_calls SET outcome=?, reason=?, duration_ms=?, ended_at=?, refs=?, "
                            "end_mac=? WHERE id=? AND outcome='running'",
                            (end["outcome"], end["reason"], end["duration_ms"], end["ended_at"], end["refs"],
                             mac, existing["id"]),
                        )
                        written += 1
                    continue
                row = {f: _norm(f, rec.get(f)) for f in _CHAIN_FIELDS + _LATE_CHAIN_FIELDS}
                row["received_at"] = now
                row_hash = self._row_mac(prev, row)
                if ended:
                    end = {f: _norm(f, rec.get(f)) for f in end_fields}
                    end_mac = self._end_mac(row_hash, end)
                else:
                    end = {"outcome": "running", "reason": "", "duration_ms": None, "ended_at": None,
                           "refs": _EMPTY["refs"]}
                    end_mac = ""
                cols = _CHAIN_FIELDS + _LATE_CHAIN_FIELDS + end_fields + ("prev_hash", "hash", "end_mac")
                values = ([row[f] for f in _CHAIN_FIELDS + _LATE_CHAIN_FIELDS] + [end[f] for f in end_fields]
                          + [prev, row_hash, end_mac])
                await db.execute(
                    f"INSERT INTO tool_calls ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    values,
                )
                prev = row_hash
                written += 1
            await db.commit()
        return written

    async def prune(self, now: float | None = None, actor: str = "retention") -> int:
        """Delete rows older than the retention, oldest first, moving the
        chain's anchor to the last one deleted."""
        days = await self.retention_days()
        cutoff = (now or time.time()) - days * 86400
        async with self._lock:
            db = await self._ensure_db()
            async with db.execute("SELECT max(id) FROM tool_calls WHERE received_at < ?", (cutoff,)) as cur:
                (last_id,) = await cur.fetchone()
            if not last_id:
                return 0
            async with db.execute("SELECT hash FROM tool_calls WHERE id=?", (last_id,)) as cur:
                (anchor_hash,) = await cur.fetchone()
            async with db.execute("SELECT count(*) FROM tool_calls WHERE id <= ?", (last_id,)) as cur:
                (n,) = await cur.fetchone()
            await db.execute("DELETE FROM tool_calls WHERE id <= ?", (last_id,))
            await self._set_meta(db, "anchor_id", str(last_id))
            await self._set_meta(db, "anchor_hash", str(anchor_hash))
            await db.execute("INSERT INTO audit_events (at, kind, actor, detail) VALUES (?, ?, ?, ?)",
                             (time.time(), "pruned", actor,
                              json.dumps({"rows": n, "through_id": last_id, "anchor_hash": anchor_hash,
                                          "retention_days": days})))
            await db.commit()
        return int(n)

    # ── reads ────────────────────────────────────────────────────────

    @staticmethod
    def _where(f: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if f.get("viewer"):
            clauses.append("(owner_id = ? OR speaker_id = ?)")
            args += [f["viewer"], f["viewer"]]
        if f.get("user"):
            clauses.append("(owner_id = ? OR speaker_id = ?)")
            args += [f["user"], f["user"]]
        for col in ("owner_id", "agent_slug", "agent_ref", "tool", "outcome", "risk", "source", "session_id",
                    "turn_id"):
            if f.get(col):
                clauses.append(f"{col} = ?")
                args.append(f[col])
        if f.get("not_source"):
            clauses.append("(source IS NULL OR source != ?)")
            args.append(f["not_source"])
        for col in ("member", "private"):
            if f.get(col) is not None:
                clauses.append(f"{col} = ?")
                args.append(1 if f[col] else 0)
        if f.get("since"):
            clauses.append("started_at >= ?")
            args.append(float(f["since"]))
        if f.get("until"):
            clauses.append("started_at < ?")
            args.append(float(f["until"]))
        if f.get("before_id"):
            clauses.append("id < ?")
            args.append(int(f["before_id"]))
        if f.get("q"):
            like = "%" + str(f["q"]).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            # An agent searching its own calls doesn't search their reasons: it
            # isn't shown every reason (tool_audit_digest.agent_reason).
            cols = ("params", "tool") if f.get("q_skip_reason") else ("params", "tool", "reason", "agent_name")
            clauses.append("(" + " OR ".join(f"{c} LIKE ? ESCAPE '\\'" for c in cols) + ")")
            args += [like] * len(cols)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    async def list(self, filters: dict[str, Any], limit: int = 100) -> list[dict]:
        db = await self._ensure_db()
        where, args = self._where(filters)
        async with db.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM tool_calls{where} ORDER BY id DESC LIMIT ?",
            (*args, int(limit)),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def iterate(self, filters: dict[str, Any], batch: int = 1000):
        """Every matching row, newest first, in batches (exports)."""
        before = None
        while True:
            rows = await self.list({**filters, "before_id": before}, limit=batch)
            if not rows:
                return
            for r in rows:
                yield r
            before = rows[-1]["id"]

    async def facets(self, filters: dict[str, Any]) -> dict[str, list]:
        """The agents, tools and sources the reader's rows mention (pickers)."""
        db = await self._ensure_db()
        where, args = self._where(filters)
        out: dict[str, list] = {}
        async with db.execute(
            f"SELECT agent_slug, max(agent_name) AS name, count(*) AS n FROM tool_calls{where} "
            "GROUP BY agent_slug ORDER BY n DESC LIMIT 200", args,
        ) as cur:
            out["agents"] = [{"slug": r["agent_slug"], "name": r["name"] or r["agent_slug"], "count": r["n"]}
                             for r in await cur.fetchall()]
        async with db.execute(
            f"SELECT tool, count(*) AS n FROM tool_calls{where} GROUP BY tool ORDER BY n DESC LIMIT 300", args,
        ) as cur:
            out["tools"] = [{"tool": r["tool"], "count": r["n"]} for r in await cur.fetchall()]
        async with db.execute(
            f"SELECT source, max(source_label) AS label, count(*) AS n FROM tool_calls{where} "
            "GROUP BY source ORDER BY n DESC LIMIT 50", args,
        ) as cur:
            out["sources"] = [{"source": r["source"], "label": r["label"] or r["source"], "count": r["n"]}
                              for r in await cur.fetchall() if r["source"]]
        return out

    async def summary(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Counts by outcome and by risk for the matching rows."""
        db = await self._ensure_db()
        where, args = self._where(filters)
        out: dict[str, Any] = {"total": 0, "outcomes": {}, "risks": {}}
        async with db.execute(f"SELECT outcome, risk, count(*) AS n FROM tool_calls{where} "
                              "GROUP BY outcome, risk", args) as cur:
            for r in await cur.fetchall():
                out["total"] += r["n"]
                out["outcomes"][r["outcome"]] = out["outcomes"].get(r["outcome"], 0) + r["n"]
                out["risks"][r["risk"]] = out["risks"].get(r["risk"], 0) + r["n"]
        return out

    async def verify(self, now: float | None = None) -> dict[str, Any]:
        """Walk the chain from its anchor: every row's MAC, its link to the row
        before, and each settled outcome's MAC; then the agents' numbering."""
        db = await self._ensure_db()
        anchor_id = int(await self._meta("anchor_id", "0") or 0)
        prev = await self._meta("anchor_hash", GENESIS)
        checked = 0
        problems: list[dict[str, Any]] = []
        last_id = anchor_id
        async with db.execute(f"SELECT {', '.join(_COLUMNS)} FROM tool_calls ORDER BY id") as cur:
            async for r in cur:
                row = dict(r)
                checked += 1
                if row["prev_hash"] != prev:
                    problems.append({"id": row["id"], "problem": "link",
                                     "detail": "The row before this one was removed or changed"})
                if self._row_mac(row["prev_hash"], row) != row["hash"]:
                    problems.append({"id": row["id"], "problem": "row",
                                     "detail": "This row was changed after it was recorded"})
                if row["outcome"] != "running" and self._end_mac(row["hash"], row) != row["end_mac"]:
                    problems.append({"id": row["id"], "problem": "outcome",
                                     "detail": "This row's outcome was changed after it was recorded"})
                prev = row["hash"]
                last_id = row["id"]
                if len(problems) >= 200:
                    break
        gaps = await self.gaps(now)
        return {"ok": not problems and not gaps, "checked": checked, "anchor_id": anchor_id,
                "last_id": last_id, "problems": problems, "gaps": gaps, "verified_at": time.time()}

    async def gaps(self, now: float | None = None) -> list[dict[str, Any]]:
        """Records an agent numbered that never arrived, per agent and boot
        (only among rows old enough that nothing earlier is still in flight)."""
        db = await self._ensure_db()
        settled = (now or time.time()) - GAP_SETTLE_S
        async with db.execute(
            "SELECT agent_slug, max(agent_name) AS name, owner_id, boot, min(seq) AS lo, max(seq) AS hi, "
            "count(*) AS n, min(started_at) AS first_at, max(started_at) AS last_at FROM tool_calls "
            "WHERE received_at < ? GROUP BY agent_slug, boot HAVING max(seq) - min(seq) + 1 > count(*) "
            "ORDER BY last_at DESC LIMIT 100",
            (settled,),
        ) as cur:
            rows = await cur.fetchall()
        return [{"agent_slug": r["agent_slug"], "agent_name": r["name"] or r["agent_slug"],
                 "owner_id": r["owner_id"], "boot": r["boot"], "first_seq": r["lo"], "last_seq": r["hi"],
                 "missing": (r["hi"] - r["lo"] + 1) - r["n"], "first_at": r["first_at"], "last_at": r["last_at"]}
                for r in rows]
