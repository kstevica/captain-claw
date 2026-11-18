"""First Mate — extra data a person gives a factory, or one of its runs.

A factory's data (files, pasted notes, agents' files, Flight Deck files, a stretch
of a conversation) is kept with the factory, beside ``first_mate.db``, and every
run it starts from then on gets a copy in its folder's ``extra/`` folder. Data given
to one run goes straight into that run's ``extra/`` folder. Bat tells the planner
and each worker what's there (``bat_routes.extra_section``), and the reviewers
don't count it as results.

All of it arrives as staged uploads (:mod:`crew_attach`, via ``/fd/context``), so
every way in is checked once, there: whose agent, whose folder, no key files.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.flight_deck import crew_attach
from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck.first_mate_store import FirstMateError

MAX_FILES = 30
MAX_BYTES = 200_000_000
RUN_MAX_FILES = 60
RUN_MAX_BYTES = 400_000_000
_META = ".meta.json"
_ORIGINS = ".origins.json"  # in a run's extra/: where each file came from, for the page
_ID = re.compile(r"^[0-9a-f]{16}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def root() -> Path:
    return fms.get_store().db_path.parent / "first_mate_data"


def _folder(factory_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(factory_id or ""))[:80]
    if not safe:
        raise FirstMateError("no such factory")
    return root() / safe


def _mb(n: int) -> str:
    return f"{n // 1_000_000} MB"


# ── a factory's own ────────────────────────────────────────────────────

def items(factory_id: str) -> list[dict[str, Any]]:
    """The factory's data, oldest first."""
    folder = _folder(factory_id)
    out = []
    if folder.is_dir():
        for slot in folder.iterdir():
            try:
                meta = json.loads((slot / _META).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (slot / str(meta.get("name", ""))).is_file():
                out.append(meta)
    return sorted(out, key=lambda m: (m.get("added_at", ""), m.get("name", "")))


def _check_room(have: list[tuple[str, int]], adding: list[dict[str, Any]], max_files: int, max_bytes: int,
                what: str) -> None:
    if len(have) + len(adding) > max_files:
        raise FirstMateError(f"{what} can have {max_files} files of extra data")
    if sum(size for _, size in have) + sum(int(i.get("size") or 0) for i in adding) > max_bytes:
        raise FirstMateError(f"{what}'s extra data can come to {_mb(max_bytes)}")


def add(uid: str, factory_id: str, ids: list[str]) -> list[dict[str, Any]]:
    """Staged uploads become the factory's. Returns what was added."""
    staged = crew_attach.staged(uid, ids)
    if not staged:
        raise FirstMateError("choose something to add")
    have = items(factory_id)
    _check_room([(m["name"], int(m.get("size") or 0)) for m in have], staged, MAX_FILES, MAX_BYTES, "A factory")
    folder = _folder(factory_id)
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for item in staged:
        slot_id = secrets.token_hex(8)
        slot = folder / slot_id
        slot.mkdir()
        shutil.move(str(crew_attach.staged_file(uid, item)), str(slot / item["name"]))
        crew_attach.drop_upload(uid, item["id"])
        meta = {"id": slot_id, "name": item["name"], "size": int(item.get("size") or 0),
                "type": item.get("type") or crew_attach.type_of(item["name"]),
                "origin": item.get("origin") or "Uploaded", "added_at": _now()}
        (slot / _META).write_text(json.dumps(meta), encoding="utf-8")
        made.append(meta)
    return made


def file_of(factory_id: str, item_id: str) -> tuple[Path, dict[str, Any]]:
    """One item's file and what it is."""
    if not _ID.match(str(item_id or "")):
        raise FirstMateError("no such data")
    slot = _folder(factory_id) / item_id
    try:
        meta = json.loads((slot / _META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise FirstMateError("no such data") from None
    path = slot / str(meta.get("name", ""))
    if not path.is_file():
        raise FirstMateError("no such data")
    return path, meta


def drop(factory_id: str, item_id: str) -> bool:
    file_of(factory_id, item_id)
    shutil.rmtree(_folder(factory_id) / item_id, ignore_errors=True)
    return True


def drop_all(factory_id: str) -> None:
    """The factory is gone: so is its data."""
    shutil.rmtree(_folder(factory_id), ignore_errors=True)


# ── a run's ────────────────────────────────────────────────────────────

def _extra(run_folder: Path) -> Path:
    from captain_claw.flight_deck.bat_routes import EXTRA_DIR
    return run_folder / EXTRA_DIR


def _origins(extra: Path) -> dict[str, str]:
    try:
        got = json.loads((extra / _ORIGINS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return got if isinstance(got, dict) else {}


def place(run_folder: Path, sources: list[tuple[str, Path, str]]) -> list[str]:
    """Copies ``(name, file, origin)`` into a run's ``extra/`` folder, each under a name
    of its own there. Returns the names they got."""
    if not sources:
        return []
    extra = _extra(run_folder)
    extra.mkdir(parents=True, exist_ok=True)
    origins = _origins(extra)
    taken = {p.name for p in extra.iterdir()}
    named = crew_attach.named([{"name": crew_attach.clean_name(name)} for name, _, _ in sources], taken)
    out = []
    for item, (_, source, origin) in zip(named, sources):
        shutil.copyfile(source, extra / item["as"])
        origins[item["as"]] = origin
        out.append(item["as"])
    (extra / _ORIGINS).write_text(json.dumps(origins), encoding="utf-8")
    return out


def factory_sources(factory_id: str) -> list[tuple[str, Path, str]]:
    """What every run of the factory starts with."""
    out = []
    for meta in items(factory_id):
        try:
            path, _ = file_of(factory_id, meta["id"])
        except FirstMateError:
            continue
        out.append((meta["name"], path, f"{meta.get('origin') or 'Uploaded'} · from the factory"))
    return out


def staged_sources(uid: str, staged: list[dict[str, Any]]) -> list[tuple[str, Path, str]]:
    return [(i["name"], crew_attach.staged_file(uid, i), i.get("origin") or "Uploaded") for i in staged]


def _run_folder(run: dict[str, Any]) -> Path:
    from captain_claw.flight_deck.bat_routes import _run_folder as folder_of
    if not run.get("child_id"):
        raise FirstMateError("this run hasn't started yet")
    return folder_of({"id": run["child_id"], "owner_id": run["owner_id"]})


def run_items(run: dict[str, Any]) -> dict[str, Any]:
    """A run's extra data: each file's path inside ``extra/``, size, change time and
    where it came from, plus the run's folder for the page's file links."""
    from captain_claw.flight_deck.bat_routes import EXTRA_DIR, _project_of
    if not run.get("child_id"):
        return {"files": [], "project": "", "folder": EXTRA_DIR}
    extra = _extra(_run_folder(run))
    origins = _origins(extra)
    files = []
    if extra.is_dir():
        for here, dirs, names in os.walk(extra):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in names:
                p = Path(here) / name
                if name.startswith(".") or p.is_symlink():
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                rel = p.relative_to(extra).as_posix()
                files.append({"path": rel, "name": name, "size": st.st_size, "mtime": st.st_mtime,
                              "origin": origins.get(rel, "Added by a worker" if "/" in rel else "")})
    files.sort(key=lambda f: f["path"].lower())
    return {"files": files, "project": _project_of({"id": run["child_id"]}), "folder": EXTRA_DIR}


def add_to_run(uid: str, run: dict[str, Any], ids: list[str]) -> list[str]:
    """Staged uploads go into one run's ``extra/`` folder: its next steps see them."""
    staged = crew_attach.staged(uid, ids)
    if not staged:
        raise FirstMateError("choose something to add")
    have = [(f["path"], f["size"]) for f in run_items(run)["files"]]
    _check_room(have, staged, RUN_MAX_FILES, RUN_MAX_BYTES, "A run")
    names = place(_run_folder(run), staged_sources(uid, staged))
    for item in staged:
        crew_attach.drop_upload(uid, item["id"])
    return names


def drop_from_run(run: dict[str, Any], path: str) -> bool:
    from captain_claw.vfs import safe_join
    extra = _extra(_run_folder(run))
    target = safe_join(extra, path) if extra.is_dir() else None
    if target is None or target == extra or not target.is_file() or target.name.startswith("."):
        raise FirstMateError("no such data in this run")
    target.unlink()
    origins = _origins(extra)
    if origins.pop(Path(path).as_posix(), None) is not None:
        (extra / _ORIGINS).write_text(json.dumps(origins), encoding="utf-8")
    return True
