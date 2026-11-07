"""A bundle into an agent that already exists: reset or append.

* **reset** — for every category the bundle carries, the agent's own is
  replaced by the bundle's: a store's tables of that category are emptied and
  filled with the bundle's rows, a folder emptied and filled, a file
  overwritten. What the bundle doesn't carry stays as it was.
* **append** — the bundle's items are added to what the agent has; what it
  already has wins. Rows go in through each store's own identity (its text
  primary key) and the dedup rules below — insights by ``entity_key`` and by
  their text, intentions by title, playbooks, scripts and APIs by name — and a
  row with only an integer id is added unless an identical one is there.
  Files that exist stay; the onboarding record's lists are joined. Its
  reflections come only to an agent with none: only the newest one is
  active, so a newer one from the file would replace the agent's own.

The bundled store is only read (read-only, ``trusted_schema=OFF``): rows are
copied by column name into the agent's own database, whose own triggers keep
its search indexes right. Never the file.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.agent_bundle import archive, catalog, stores
from captain_claw.logging import get_logger

log = get_logger(__name__)

MODES = ("reset", "append")

# Never merged: a store's own bookkeeping, which belongs to the agent it is in.
_SKIP_TABLES = frozenset({"insights_pending", "generator_state", "app_state", "backfill_seen"})


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


# table → columns whose (normalised) value names an item already there, in append
DEDUP: dict[str, tuple[str, ...]] = {
    "insights": ("entity_key", "content"),
    "intentions": ("title",),
    "intuitions": ("content",),
    "playbooks": ("name",),
    "scripts": ("name",),
    "apis": ("name",),
    "browser_workflows": ("name",),
    "contacts": ("email", "name"),
    "topics": ("label",),
}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _columns(c: sqlite3.Connection, table: str) -> list[tuple[str, str, int]]:
    """(name, type, pk position) per column."""
    return [(r[1], str(r[2] or "").upper(), int(r[5] or 0)) for r in c.execute(f"PRAGMA table_info({_q(table)})")]


def _integer_id(cols: list[tuple[str, str, int]]) -> str | None:
    """The column that is the table's rowid (a lone INTEGER PRIMARY KEY), or None."""
    pks = [c for c in cols if c[2]]
    if len(pks) == 1 and pks[0][1] == "INTEGER":
        return pks[0][0]
    return None


def _standalone_fts(c: sqlite3.Connection) -> list[str]:
    """Full-text tables that keep their own text (no ``content=``): copied like tables."""
    out = []
    for name, sql in c.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql LIKE 'CREATE VIRTUAL TABLE%'"):
        text = " ".join(str(sql or "").split()).lower()
        if " using fts5" in text and "content=" not in text.replace(" ", ""):
            out.append(name)
    return out


def merge_store(src: Path, dst: Path, tables: list[str], mode: str) -> dict[str, dict[str, int]]:
    """Copy *tables* of the bundled store *src* into the agent's store *dst*;
    per table, how many rows were added and skipped."""
    if mode not in MODES:
        raise ValueError(mode)
    stores.check(src)
    result: dict[str, dict[str, int]] = {}
    with closing(sqlite3.connect(f"file:{src}?mode=ro&immutable=1", uri=True)) as s, \
            closing(sqlite3.connect(str(dst), timeout=30)) as d:
        s.execute("PRAGMA trusted_schema=OFF")
        src_tables = {n: sql for n, sql in s.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")}
        wanted = [t for t in tables if t in src_tables and t not in _SKIP_TABLES]
        # a table's own search table (topics → topics_fts) goes with it
        fts = [t for t in _standalone_fts(s) if t.removesuffix("_fts") in wanted]
        for table in [*wanted, *fts]:
            if not d.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                sql = str(src_tables.get(table) or "")
                if not sql.upper().startswith("CREATE TABLE"):
                    continue                # a store's search table the agent makes itself
                d.execute(sql)              # a datastore table this agent hasn't got
            result[table] = _copy_table(s, d, table, mode)
        d.commit()
    return result


def _copy_table(s: sqlite3.Connection, d: sqlite3.Connection, table: str, mode: str) -> dict[str, int]:
    src_cols, dst_cols = _columns(s, table), _columns(d, table)
    have = {c[0] for c in dst_cols}
    cols = [c[0] for c in src_cols if c[0] in have]
    rowid = _integer_id(dst_cols)
    if mode == "reset":
        d.execute(f"DELETE FROM {_q(table)}")
    elif rowid:
        cols = [c for c in cols if c != rowid]          # its own ids; the bundle's could be another row's
    if not cols:
        return {"added": 0, "skipped": 0}
    keys = [k for k in DEDUP.get(table, ()) if k in cols] if mode == "append" else []
    seen: dict[str, set[str]] = {k: {_norm(v) for (v,) in d.execute(f"SELECT {_q(k)} FROM {_q(table)}") if _norm(v)}
                                 for k in keys}
    whole: set[tuple] | None = None
    if mode == "append" and rowid and not keys:
        whole = {tuple(r) for r in d.execute(f"SELECT {', '.join(_q(c) for c in cols)} FROM {_q(table)}")}
    fts_ids: set[str] | None = None
    if mode == "append" and table.endswith("_fts") and "id" in cols:
        fts_ids = {str(v) for (v,) in d.execute(f"SELECT id FROM {_q(table)}")}
    insert = (f"INSERT {'OR IGNORE ' if mode == 'append' else ''}INTO {_q(table)} "
              f"({', '.join(_q(c) for c in cols)}) VALUES ({', '.join('?' * len(cols))})")
    added = skipped = 0
    for row in s.execute(f"SELECT {', '.join(_q(c) for c in cols)} FROM {_q(table)}"):
        if keys:
            values = {k: _norm(row[cols.index(k)]) for k in keys}
            if any(v and v in seen[k] for k, v in values.items()):
                skipped += 1
                continue
        if whole is not None:
            if tuple(row) in whole:
                skipped += 1
                continue
            whole.add(tuple(row))
        if fts_ids is not None:
            rid = str(row[cols.index("id")])
            if rid in fts_ids:
                skipped += 1
                continue
            fts_ids.add(rid)
        n = d.execute(insert, tuple(row)).rowcount
        if n:
            added += 1
            for k in keys:
                if values[k]:
                    seen[k].add(values[k])
        else:
            skipped += 1
    return {"added": added, "skipped": skipped}


# ── the onboarding record ───────────────────────────────────────────────────

def _identity(item: Any) -> str:
    if isinstance(item, dict):
        for k in ("id", "text", "doc", "file", "name", "question"):
            if item.get(k):
                return f"{k}:{_norm(item[k])}"
        return json.dumps(item, sort_keys=True, default=str)
    return _norm(item)


def merge_record(mine: dict[str, Any], theirs: dict[str, Any]) -> dict[str, Any]:
    """The agent's onboarding record with the bundle's lists joined in (what it
    has wins); its other values stay its own."""
    out = dict(mine)
    for key, value in theirs.items():
        if isinstance(value, list):
            have = list(mine.get(key) or []) if isinstance(mine.get(key), list) else []
            ids = {_identity(x) for x in have}
            out[key] = have + [x for x in value if _identity(x) not in ids]
        elif key not in out:
            out[key] = value
    return out


# ── files and folders ───────────────────────────────────────────────────────

def merge_files(bundle: archive.Bundle, agent_dir: Path, categories: set[str], mode: str,
                inside) -> dict[str, int]:
    """The bundle's files of *categories* into *agent_dir*: reset empties a
    folder part first; append keeps what exists. ``inside(root, path)`` guards
    every write. The onboarding record is joined, not overwritten, in append."""
    written = kept = 0
    parts_seen: set[str] = set()
    reflections = agent_dir / catalog.BY_KEY["reflections"].path
    has_reflections = reflections.is_dir() and any(reflections.glob("*.md"))
    names = bundle.names("agent/")
    for name in names:
        part = catalog.part_for_path(name[len("agent/"):])
        if part and part.category in categories:
            parts_seen.add(part.key)
    if mode == "reset":
        for key in parts_seen:
            part = catalog.BY_KEY[key]
            target = agent_dir / part.path
            if part.kind == "dir" and target.is_dir() and not target.is_symlink() and inside(agent_dir, target):
                for child in target.iterdir():
                    if child.name in part.skip:
                        continue                 # another part's (knowledge/ inside the workspace)
                    shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    for name in names:
        rel = name[len("agent/"):]
        part = catalog.part_for_path(rel)
        if part is None or part.key not in parts_seen:
            continue
        dest = agent_dir / rel
        if not inside(agent_dir, dest.parent):
            continue
        if part.key == "setup_record" and mode == "append" and dest.is_file():
            try:
                mine = json.loads(dest.read_text(encoding="utf-8"))
                theirs = bundle.read_json(name)
            except (OSError, ValueError):
                continue
            if isinstance(mine, dict) and isinstance(theirs, dict):
                dest.write_text(json.dumps(merge_record(mine, theirs), ensure_ascii=False), encoding="utf-8")
                written += 1
            continue
        if part.key == "lessons":
            continue                             # written from the record below
        if part.key == "reflections" and mode == "append" and has_reflections:
            kept += 1
            continue
        if mode == "append" and dest.exists():
            kept += 1
            continue
        bundle.extract(name, dest)
        written += 1
    if "setup_record" in parts_seen or "lessons" in parts_seen:
        _write_lessons(bundle, agent_dir)
    return {"written": written, "kept": kept}


def _write_lessons(bundle: archive.Bundle, agent_dir: Path) -> None:
    """lessons.json from the (merged) record — or the bundle's, for an agent with no record."""
    from captain_claw.flight_deck import agent_lessons
    record_path = agent_dir / catalog.BY_KEY["setup_record"].path
    try:
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else None
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict):
        agent_lessons.write(agent_dir, record)
        return
    name = f"agent/{catalog.BY_KEY['lessons'].path}"
    if bundle.has(name):
        bundle.extract(name, agent_dir / catalog.BY_KEY["lessons"].path)
