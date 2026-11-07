"""Packages and versions: an agent passed on, then updated.

A consultant exports an agent as a named **package**; each export of that
package is the next **version**. On the exporting deck the agent's folder
keeps the counter (``.bundle-packages.json``).

The agent that imports a package keeps what each package put in
(``.bundle-imports.json`` in its folder): the ids of the rows added (with their
``updated_at``), the files written (with their sha256), the onboarding items
added and the scheduled tasks made. An **update** to a newer version takes
the earlier version's items out — not one the client changed since (an
insight edited, a file rewritten, a task switched on): that is theirs now —
and then appends the new version. Everything the client added stays.

Neither file is in the catalogue, so neither ever travels in a bundle.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.agent_bundle import archive, catalog, stores

PACKAGES_FILE = ".bundle-packages.json"
IMPORTS_FILE = ".bundle-imports.json"
_ID = re.compile(r"[0-9a-f]{16}")


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


# ── the exporting side ──────────────────────────────────────────────────────

def _key(name: str) -> str:
    return " ".join(str(name or "").split()).casefold()


def next_version(agent_dir: Path, name: str) -> dict[str, Any]:
    """The package *name* of this agent at its next version (the first is 1)."""
    name = " ".join(str(name or "").split())[:80]
    if not name:
        raise archive.BundleError("Give the package a name.")
    path = agent_dir / PACKAGES_FILE
    data = _read(path)
    entry = data.get(_key(name)) if isinstance(data.get(_key(name)), dict) else {}
    pid = str(entry.get("id") or "") if _ID.fullmatch(str(entry.get("id") or "")) else secrets.token_hex(8)
    name = str(entry.get("name") or name)            # as it was first named
    version = int(entry.get("version") or 0) + 1
    data[_key(name)] = {"id": pid, "name": name, "version": version, "made": time.time()}
    _write(path, data)
    return {"id": pid, "name": name, "version": version}


def packages(agent_dir: Path) -> list[dict[str, Any]]:
    """The packages this agent was exported as, with their last version."""
    return [{"name": str(v.get("name") or ""), "version": int(v.get("version") or 0)}
            for v in _read(agent_dir / PACKAGES_FILE).values() if isinstance(v, dict)]


def bundle_package(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The package a bundle is a version of, when it is one."""
    pkg = manifest.get("package")
    if not isinstance(pkg, dict) or not _ID.fullmatch(str(pkg.get("id") or "")):
        return None
    try:
        version = int(pkg.get("version") or 0)
    except (TypeError, ValueError):
        return None
    return {"id": str(pkg["id"]), "name": str(pkg.get("name") or "")[:80], "version": max(1, version)}


# ── the importing side ──────────────────────────────────────────────────────

def imported(agent_dir: Path) -> dict[str, Any]:
    """package id → what its last imported version put in."""
    return {k: v for k, v in _read(agent_dir / IMPORTS_FILE).items() if _ID.fullmatch(k) and isinstance(v, dict)}


def record(agent_dir: Path, package: dict[str, Any], items: dict[str, Any], *, made_by: str = "") -> None:
    data = _read(agent_dir / IMPORTS_FILE)
    data[package["id"]] = {"name": package["name"], "version": package["version"], "made_by": made_by,
                           "imported": time.time(), "items": items}
    _write(agent_dir / IMPORTS_FILE, data)


def add_schedules(agent_dir: Path, package_id: str, job_ids: list[str]) -> None:
    data = _read(agent_dir / IMPORTS_FILE)
    entry = data.get(package_id)
    if isinstance(entry, dict):
        items = entry.setdefault("items", {})
        items["schedules"] = [*items.get("schedules", []), *job_ids]
        _write(agent_dir / IMPORTS_FILE, data)


def updates_for(agents: list[tuple[str, str, Path]], package: dict[str, Any]) -> list[dict[str, Any]]:
    """Which of the importer's agents (slug, name, folder) hold an earlier
    version of *package* — an upload of a newer one offers to update them."""
    out = []
    for slug, name, agent_dir in agents:
        entry = imported(agent_dir).get(package["id"])
        if entry and int(entry.get("version") or 0) < package["version"]:
            out.append({"slug": slug, "name": name, "from_version": int(entry.get("version") or 0)})
    return out


def _sha(path: Path) -> str:
    return archive.sha256_file(path) if path.is_file() else ""


def items_of_new_agent(bundle: archive.Bundle, agent_dir: Path) -> dict[str, Any]:
    """Everything a bundle put into a new agent (or one it reset): every row id
    of its stores, every file, every onboarding item."""
    from captain_claw.flight_deck.agent_bundle.merge import record_identities
    rows: dict[str, dict[str, str]] = {}
    for name in bundle.names("stores/"):
        key = name[len("stores/"):].removesuffix(".db")
        part = catalog.BY_KEY.get(key)
        db = agent_dir / part.path if part else None
        if not db or not db.is_file():
            continue
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as c:
            for table in stores._tables(c):
                cols = {r[1] for r in c.execute(f'PRAGMA table_info("{table}")')}
                if "id" not in cols:
                    continue
                stamp = "updated_at" if "updated_at" in cols else "''"
                got = {str(i): str(u or "") for i, u in c.execute(f'SELECT id, {stamp} FROM "{table}"')}
                if got:
                    rows[f"{key}.{table}"] = got
    files = {}
    for name in bundle.names("agent/"):
        rel = name[len("agent/"):]
        part = catalog.part_for_path(rel)
        if part and part.key not in ("setup_record", "lessons") and (agent_dir / rel).is_file():
            files[rel] = _sha(agent_dir / rel)
    rec = _read(agent_dir / catalog.BY_KEY["setup_record"].path)
    return {"rows": rows, "files": files, "record": record_identities(rec) if rec else {}, "schedules": []}


def items_of_merge(done: dict[str, Any], agent_dir: Path) -> dict[str, Any]:
    """What an append put into an existing agent (apply_to_existing's result)."""
    rows = {}
    for key, tables in (done.get("stores") or {}).items():
        for table, got in tables.items():
            if got.get("ids"):
                rows[f"{key}.{table}"] = dict(got["ids"])
    files = {rel: _sha(agent_dir / rel) for rel in (done.get("files") or {}).get("paths") or []}
    return {"rows": rows, "files": files, "record": (done.get("files") or {}).get("record") or {}, "schedules": []}


def take_out(agent_dir: Path, items: dict[str, Any], *, owner_id: str = "") -> dict[str, int]:
    """Take an earlier version's items out of the agent — those nobody changed
    since. How many went, and how many stayed because they were changed."""
    from captain_claw.flight_deck.agent_bundle.merge import _identity
    gone = kept = 0
    for ref, ids in (items.get("rows") or {}).items():
        key, _, table = str(ref).partition(".")
        part = catalog.BY_KEY.get(key)
        db = agent_dir / part.path if part and part.kind == "sqlite" else None
        if not db or not db.is_file() or not isinstance(ids, dict):
            continue
        with closing(sqlite3.connect(db, timeout=30)) as c:
            cols = {r[1] for r in c.execute(f'PRAGMA table_info("{table}")')}
            if "id" not in cols:
                continue
            for rid, stamp in ids.items():
                if "updated_at" in cols and stamp:
                    now = c.execute(f'SELECT updated_at FROM "{table}" WHERE id = ?', (rid,)).fetchone()
                    if now and str(now[0] or "") != stamp:
                        kept += 1
                        continue
                gone += c.execute(f'DELETE FROM "{table}" WHERE id = ?', (rid,)).rowcount
            c.commit()
    for rel, digest in (items.get("files") or {}).items():
        path = agent_dir / rel
        part = catalog.part_for_path(str(rel))
        if part is None or not path.is_file() or path.is_symlink():
            continue
        if digest and _sha(path) != digest:
            kept += 1
            continue
        path.unlink()
        gone += 1
        root, parent = agent_dir / part.path, path.parent     # folders it leaves empty go, up to its part's
        while parent != root and root in parent.parents and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    record_items = items.get("record") or {}
    rec_path = agent_dir / catalog.BY_KEY["setup_record"].path
    rec = _read(rec_path)
    if rec and record_items:
        for group, ids in record_items.items():
            if isinstance(rec.get(group), list):
                drop = set(ids)
                before = len(rec[group])
                rec[group] = [x for x in rec[group] if _identity(x) not in drop]
                gone += before - len(rec[group])
        _write(rec_path, rec)
        from captain_claw.flight_deck import agent_lessons
        agent_lessons.write(agent_dir, rec)
    if items.get("schedules"):
        from captain_claw.flight_deck import fd_scheduler
        store = fd_scheduler.get_store()
        for jid in items["schedules"]:
            job = store.get(str(jid))
            if not job or (owner_id and str(job.get("owner_id") or "") not in ("", owner_id)):
                continue
            if job.get("enabled"):
                kept += 1                      # switched on since: the client's now
                continue
            store.delete(str(jid))
            gone += 1
    return {"gone": gone, "kept": kept}


def base_layer_on_new(agent_dir: Path) -> None:
    """A new agent made of someone else's package: its onboarding is the base
    (the new owner adds on top and can't change it) — as an agent made from a
    passed-on archetype (agent_package)."""
    from captain_claw.flight_deck import agent_lessons
    from captain_claw.flight_deck.agent_package import _as_base
    path = agent_dir / catalog.BY_KEY["setup_record"].path
    rec = _read(path)
    if rec:
        rec = _as_base(rec)
        _write(path, rec)
        agent_lessons.write(agent_dir, rec)


def note_instructions(agent_dir: Path, package_id: str, text: str) -> None:
    """The instructions this package set (their sha256): an update replaces
    them only while they are still the ones it set."""
    data = _read(agent_dir / IMPORTS_FILE)
    entry = data.get(package_id)
    if isinstance(entry, dict):
        entry.setdefault("items", {})["instructions"] = text_sha(text)
        _write(agent_dir / IMPORTS_FILE, data)


def text_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(" ".join(str(text or "").split()).encode()).hexdigest()
