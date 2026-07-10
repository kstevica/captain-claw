"""Shared facts ledger — a run's single source of truth for load-bearing values.

Everything crossing between ensemble/collaborative workers today is prose:
board posts, ask answers, plan-time conventions. When worker B builds on worker
A's numbers it re-states them from prose, and nothing guarantees a value is
identical everywhere it appears — the root cause of the self-contradicting-
deliverable failure class (see docs/vatra-quality-tightening-findings.md, F1).

This is the machine-readable counterpart: one small SQLite per shared VFS
folder (``<project>/.facts.db``) holding canonical ``key → value`` records with
status, provenance, confidence and lineage. Workers read and write it through
the ``facts`` tool; the reporter gets a dump of it as the values the deliverable
must match; the consistency check (``research_consistency.verify``) cross-checks
the final text against it.

Conflict rule — the load-bearing design decision: a second writer offering a
DIFFERENT value for an existing key does NOT overwrite. The original stays
canonical, the offer is recorded in a conflicts table, and the writer gets the
conflict back so it can reconcile (check the source, post an ask). ``force=True``
replaces the value but still records the previous one — visible history, never a
silent flip. Same-value writes (within a relative tolerance for numbers) merge
metadata instead: provenance/status/confidence refresh, so a later verified
write upgrades an earlier assumed one.

Pure module: sqlite3 + stdlib only, no model, no network. Callers (the tool and
the route wiring) resolve WHICH folder; everything here takes an explicit path.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from captain_claw.logging import get_logger

log = get_logger(__name__)

DB_NAME = ".facts.db"

STATUSES = ("verified", "derived", "estimated", "assumed", "to_be_completed")
_DEFAULT_STATUS = "assumed"

#: Ledger is for load-bearing values, not trivia — a hard cap keeps a
#: keen worker from turning it into a scratchpad.
ROW_CAP = 200

#: Same-value tolerance (relative only — no absolute floor, so 0.35 vs 0.36
#: conflicts while 300000 vs 300000.4 merges).
_REL_TOL = 0.005


def db_path(project: Path | str) -> Path:
    return Path(project) / DB_NAME


def _db(project: Path | str) -> sqlite3.Connection:
    Path(project).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path(project))
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS facts(
        key TEXT PRIMARY KEY, value TEXT, value_num REAL, unit TEXT,
        status TEXT, provenance TEXT, confidence REAL, computed_from TEXT,
        updated_by TEXT, updated_at REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS conflicts(
        id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT,
        offered_value TEXT, offered_by TEXT, existing_value TEXT,
        forced INTEGER DEFAULT 0, created_at REAL)""")
    return con


def norm_key(key: str) -> str:
    """Canonical key form: trimmed, lowercased, spaces/punctuation → underscores."""
    k = re.sub(r"[^\w]+", "_", (key or "").strip().casefold())
    return k.strip("_")[:120]


def _as_num(value) -> float | None:
    try:
        return float(str(value).replace(",", "").replace("_", "").strip())
    except (TypeError, ValueError):
        return None


def _same_value(a: str, a_num: float | None, b: str, b_num: float | None) -> bool:
    if a_num is not None and b_num is not None:
        return abs(a_num - b_num) <= _REL_TOL * max(abs(a_num), abs(b_num))
    return a.strip().casefold() == b.strip().casefold()


def _row_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d.pop("value_num", None)
    return d


def upsert(project: Path | str, key: str, value, *,
           unit: str = "", status: str = "", provenance: str = "",
           confidence: float | None = None, computed_from: str = "",
           updated_by: str = "", force: bool = False) -> dict:
    """Write one fact. Returns a dict describing what happened:

    * ``{"ok": True, "action": "created"|"merged"|"forced", "fact": {...}}``
    * ``{"ok": False, "reason": "conflict", "existing": {...}}`` — different
      value offered without ``force``; NOT saved, conflict recorded.
    * ``{"ok": False, "reason": "full"}`` — row cap reached for a NEW key.
    """
    k = norm_key(key)
    if not k:
        return {"ok": False, "reason": "empty_key"}
    v = str(value if value is not None else "").strip()
    v_num = _as_num(v)
    st = (status or "").strip().casefold()
    if st not in STATUSES:
        st = "" if not st else _DEFAULT_STATUS  # unknown → weakest, absent → keep/default
    who = (updated_by or "").strip()[:80]
    now = time.time()

    con = _db(project)
    try:
        cur = con.execute("SELECT * FROM facts WHERE key = ?", (k,))
        existing = cur.fetchone()
        if existing is None:
            n = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            if n >= ROW_CAP:
                return {"ok": False, "reason": "full",
                        "message": f"ledger is at its {ROW_CAP}-row cap — "
                                   "keep it to load-bearing values"}
            con.execute(
                "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?)",
                (k, v, v_num, unit.strip()[:20], st or _DEFAULT_STATUS,
                 provenance.strip()[:400], confidence, computed_from.strip()[:200],
                 who, now))
            con.commit()
            return {"ok": True, "action": "created",
                    "fact": get(project, k, _con=con)}

        if _same_value(existing["value"], existing["value_num"], v, v_num):
            # Same value → merge metadata (a verified write upgrades an assumed one).
            con.execute(
                """UPDATE facts SET unit = COALESCE(NULLIF(?, ''), unit),
                   status = COALESCE(NULLIF(?, ''), status),
                   provenance = COALESCE(NULLIF(?, ''), provenance),
                   confidence = COALESCE(?, confidence),
                   computed_from = COALESCE(NULLIF(?, ''), computed_from),
                   updated_by = COALESCE(NULLIF(?, ''), updated_by),
                   updated_at = ? WHERE key = ?""",
                (unit.strip()[:20], st, provenance.strip()[:400], confidence,
                 computed_from.strip()[:200], who, now, k))
            con.commit()
            return {"ok": True, "action": "merged",
                    "fact": get(project, k, _con=con)}

        # Different value: record the offer either way; replace only on force.
        con.execute(
            "INSERT INTO conflicts(key, offered_value, offered_by, existing_value,"
            " forced, created_at) VALUES (?,?,?,?,?,?)",
            (k, v, who, existing["value"], 1 if force else 0, now))
        if not force:
            con.commit()
            return {"ok": False, "reason": "conflict",
                    "existing": _row_dict(existing),
                    "message": (f"'{k}' is already {existing['value']}"
                                f"{(' ' + existing['unit']) if existing['unit'] else ''} "
                                f"(status {existing['status']}, by "
                                f"{existing['updated_by'] or 'unknown'}). NOT saved — "
                                "reconcile with the team (check the source / post an "
                                "ask); use force=true only after agreeing the old "
                                "value is wrong.")}
        con.execute(
            """UPDATE facts SET value = ?, value_num = ?,
               unit = COALESCE(NULLIF(?, ''), unit),
               status = COALESCE(NULLIF(?, ''), status),
               provenance = COALESCE(NULLIF(?, ''), provenance),
               confidence = COALESCE(?, confidence),
               computed_from = COALESCE(NULLIF(?, ''), computed_from),
               updated_by = COALESCE(NULLIF(?, ''), updated_by),
               updated_at = ? WHERE key = ?""",
            (v, v_num, unit.strip()[:20], st, provenance.strip()[:400], confidence,
             computed_from.strip()[:200], who, now, k))
        con.commit()
        return {"ok": True, "action": "forced",
                "previous": existing["value"], "fact": get(project, k, _con=con)}
    finally:
        con.close()


def get(project: Path | str, key: str, _con: sqlite3.Connection | None = None) -> dict | None:
    con = _con or _db(project)
    try:
        r = con.execute("SELECT * FROM facts WHERE key = ?", (norm_key(key),)).fetchone()
        return _row_dict(r) if r else None
    finally:
        if _con is None:
            con.close()


def list_rows(project: Path | str) -> list[dict]:
    if not db_path(project).is_file():
        return []
    con = _db(project)
    try:
        return [_row_dict(r) for r in
                con.execute("SELECT * FROM facts ORDER BY key").fetchall()]
    finally:
        con.close()


def conflicts(project: Path | str) -> list[dict]:
    if not db_path(project).is_file():
        return []
    con = _db(project)
    try:
        return [dict(r) for r in
                con.execute("SELECT * FROM conflicts ORDER BY id").fetchall()]
    finally:
        con.close()


def export_rows(project: Path | str) -> list[dict]:
    """Numeric facts as ``[{key, value, unit}]`` for
    ``research_consistency.verify(ledger_rows=...)`` — text-vs-ledger checking."""
    if not db_path(project).is_file():
        return []
    con = _db(project)
    try:
        return [{"key": r["key"], "value": r["value_num"], "unit": r["unit"]}
                for r in con.execute(
                    "SELECT key, value_num, unit FROM facts "
                    "WHERE value_num IS NOT NULL ORDER BY key").fetchall()]
    finally:
        con.close()


def dump_markdown(project: Path | str, max_rows: int = 80) -> str:
    """Compact table of the ledger for the reporter/synthesizer prompt.
    "" when the ledger is empty (caller can then skip the whole section)."""
    rows = list_rows(project)
    if not rows:
        return ""

    def _cell(s) -> str:
        return str(s if s is not None else "").replace("|", "\\|").replace("\n", " ").strip() or "—"

    out = ["| key | value | unit | status | provenance | by |",
           "|-----|-------|------|--------|------------|----|"]
    for r in rows[:max_rows]:
        out.append(f"| {_cell(r['key'])} | {_cell(r['value'])} | {_cell(r['unit'])} "
                   f"| {_cell(r['status'])} | {_cell(r['provenance'])[:80]} "
                   f"| {_cell(r['updated_by'])} |")
    if len(rows) > max_rows:
        out.append(f"| … | ({len(rows) - max_rows} more rows — `facts` action=list) | | | | |")
    open_conf = [c for c in conflicts(project) if not c["forced"]]
    if open_conf:
        out.append("")
        out.append("**Unresolved value conflicts** (the ledger value stayed canonical; "
                   "surface these in Unresolved & assumptions):")
        for c in open_conf[-10:]:
            out.append(f"- `{c['key']}`: kept {c['existing_value']}, "
                       f"{c['offered_by'] or 'a teammate'} offered {c['offered_value']}")
    return "\n".join(out)
