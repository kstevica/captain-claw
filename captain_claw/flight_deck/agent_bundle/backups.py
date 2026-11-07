"""Scheduled backups of a person's agents.

Each person picks, for their own agents: daily or weekly, the hour (the deck's
local time), how many to keep per agent, which agents (none named: all), and
optionally a folder in their files (VFS) where a copy also goes — a linked
folder there can be a synced disk. The settings live in their user settings
(``fd:agent-backups``; an auth-off deck keeps one set in system settings).

A backup is a full bundle without keys (export.build, preset "full"), kept in
``agent-bundles/backups/<agent>/``; past the number to keep, the oldest go.
Taking one doesn't stop the agent: its stores are snapshotted through
SQLite's backup API, consistent while it runs.

``backup_loop`` runs in Flight Deck and looks every few minutes for whose
backups are due.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.agent_bundle import archive, catalog, export
from captain_claw.logging import get_logger

log = get_logger(__name__)

SETTING = "fd:agent-backups"
LOCAL_SETTING = "fd:agent-backups:local"     # an auth-off deck: one set for everyone
TICK_S = 300
DEFAULTS: dict[str, Any] = {"enabled": False, "frequency": "daily", "hour": 3, "weekday": 6, "keep": 7,
                            "agents": [], "folder": "", "last_run": 0.0, "last_error": ""}
_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,80}")


def clean(raw: Any) -> dict[str, Any]:
    """Settings as kept: known keys only, each in its range."""
    s = {**DEFAULTS, **(raw if isinstance(raw, dict) else {})}
    folder = str(s.get("folder") or "").strip().replace("\\", "/").strip("/")
    if folder.lower().startswith("vfs:"):
        folder = folder[4:].lstrip("/")
    return {
        "enabled": bool(s["enabled"]),
        "frequency": "weekly" if s["frequency"] == "weekly" else "daily",
        "hour": min(23, max(0, int(s.get("hour") or 0))),
        "weekday": min(6, max(0, int(s.get("weekday") or 0))),
        "keep": min(60, max(1, int(s["keep"] if s.get("keep") is not None else 7))),
        "agents": [str(a) for a in s.get("agents") or [] if _SLUG.fullmatch(str(a))][:200],
        "folder": folder[:300] if ".." not in folder.split("/") else "",
        "last_run": float(s.get("last_run") or 0.0),
        "last_error": str(s.get("last_error") or "")[:500],
    }


async def load(owner_id: str) -> dict[str, Any]:
    from captain_claw.flight_deck.auth import get_db
    db = get_db()
    raw = await (db.get_setting(owner_id, SETTING) if owner_id else db.get_system_setting(LOCAL_SETTING))
    try:
        return clean(json.loads(raw) if raw else {})
    except (ValueError, TypeError):
        return clean({})


async def save(owner_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    from captain_claw.flight_deck.auth import get_db
    db = get_db()
    value = clean(settings)
    if owner_id:
        await db.set_settings(owner_id, {SETTING: json.dumps(value)})
    else:
        await db.set_system_setting(LOCAL_SETTING, json.dumps(value))
    return value


def due(settings: dict[str, Any], now: float | None = None) -> bool:
    """Is a scheduled backup due now? Once a day (or week), at or after its hour."""
    if not settings.get("enabled"):
        return False
    now = time.time() if now is None else now
    local = time.localtime(now)
    if local.tm_hour < settings["hour"]:
        return False
    if settings["frequency"] == "weekly":
        if local.tm_wday != settings["weekday"]:
            return False
        return now - settings["last_run"] > 6 * 86400
    return now - settings["last_run"] > 20 * 3600


def folder_of(slug: str) -> Path:
    return export.bundles_dir() / "backups" / slug


def records(owner_id: str, slug: str | None = None) -> list[dict[str, Any]]:
    """The person's backups (newest first), of one agent or all of theirs."""
    root = export.bundles_dir() / "backups"
    out = []
    for meta in root.glob(f"{slug or '*'}/*.json"):
        try:
            rec = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and str(rec.get("owner") or "") == owner_id and \
                meta.with_suffix(archive.EXTENSION).is_file():
            out.append({k: rec.get(k) for k in ("id", "slug", "file_name", "size", "made")})
    return sorted(out, key=lambda r: float(r.get("made") or 0), reverse=True)


def find(owner_id: str, bundle_id: str) -> tuple[Path, dict[str, Any]] | None:
    if not re.fullmatch(r"[0-9a-f]{24}", bundle_id or ""):
        return None
    for path in (export.bundles_dir() / "backups").glob(f"*/{bundle_id}{archive.EXTENSION}"):
        try:
            rec = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(rec.get("owner") or "") == owner_id:
            return path, rec
    return None


def prune(owner_id: str, slug: str, keep: int) -> int:
    """Past *keep* backups of the agent, the oldest go; how many went."""
    gone = 0
    for rec in records(owner_id, slug)[keep:]:
        path = folder_of(slug) / f"{rec['id']}{archive.EXTENSION}"
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        gone += 1
    return gone


def copy_to_folder(owner_id: str, folder: str, source: Path, file_name: str) -> str:
    """A copy of the backup into the person's files (VFS) — "" or why it couldn't."""
    from captain_claw import vfs
    if not folder:
        return ""
    proj = folder.split("/", 1)[0]
    root = vfs.user_root_of(owner_id or "", create=True)
    link = vfs.read_links_at(root).get(proj)
    if (isinstance(link, dict) and str(link.get("mode", "rw")).lower() == "ro") or vfs._drive_mount_root(root, proj):
        return f"“{proj}” is a read-only folder — the copy wasn't made."
    target = vfs.resolve_under(owner_id or "", "", f"{folder}/{file_name}")
    if target is None:
        return "That folder isn't one of your files."
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return ""


async def run(owner_id: str, settings: dict[str, Any] | None = None, *, slugs: list[str] | None = None) -> dict[str, Any]:
    """Back up the person's agents now (those *settings* name, or all); what was made."""
    from captain_claw.flight_deck import server as srv
    settings = settings or await load(owner_id)
    registry = srv._load_process_registry()
    mine = {s: e for s, e in registry.items() if not owner_id or str(e.get("owner") or "") == owner_id}
    wanted = slugs if slugs is not None else (settings["agents"] or list(mine))
    made: list[dict[str, Any]] = []
    problems: list[str] = []
    for slug in wanted:
        entry = mine.get(slug)
        if entry is None or not (srv.DATA_DIR / slug).is_dir():
            continue
        try:
            rec = await export.build(slug, owner_id, entry, categories=set(catalog.CATEGORIES), preset="full",
                                     made_by="Scheduled backup", keep_in=f"backups/{slug}")
        except Exception as exc:                        # one agent's trouble never stops the others
            log.warning("agent backup failed", agent=slug, error=str(exc)[:200])
            problems.append(f"{entry.get('name') or slug}: {str(exc)[:160]}")
            continue
        prune(owner_id, slug, settings["keep"])
        note = await asyncio.to_thread(copy_to_folder, owner_id, settings["folder"],
                                       folder_of(slug) / f"{rec['id']}{archive.EXTENSION}", rec["file_name"])
        if note:
            problems.append(note)
        made.append({k: rec[k] for k in ("id", "slug", "file_name", "size", "made")})
    return {"made": made, "problems": problems}


async def _tick() -> None:
    from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db
    owners: list[str] = []
    if _fd_auth_enabled():
        db = get_db()
        for user in await db.list_users(limit=10000):
            raw = await db.get_setting(str(user["id"]), SETTING)
            try:
                if raw and clean(json.loads(raw))["enabled"]:
                    owners.append(str(user["id"]))
            except (ValueError, TypeError):
                continue
    else:
        owners.append("")
    for owner_id in owners:
        settings = await load(owner_id)
        if not due(settings):
            continue
        result = await run(owner_id, settings)
        settings["last_run"] = time.time()
        settings["last_error"] = "; ".join(result["problems"])[:500]
        await save(owner_id, settings)
        log.info("scheduled agent backups", owner=owner_id or "-", made=len(result["made"]),
                 problems=len(result["problems"]))


async def backup_loop(stop: asyncio.Event) -> None:
    """Flight Deck's background loop: every few minutes, whose backups are due."""
    while not stop.is_set():
        try:
            await _tick()
        except Exception:
            log.warning("agent backups: a round failed", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=TICK_S)
        except TimeoutError:
            pass
