"""What's kept from WhatsApp links that ended: a number WhatsApp banned, a number
the user switched away from, a device removed on the phone.

An ended link's messages move, whole, to
``<data dir>/.kept/<user_id>/<entry>/`` (``store.sqlite3``, sealed as ever, and
``kept.json``). That folder is outside the user's link folder on purpose:
everything that replaces or removes a link deletes the link folder, and none of
it can take the history along. Only the user deletes an entry; the retention
they chose goes on counting too (:meth:`.engine.Engine.maintain`).

Their agents go on reading it (the engine's reads ``with_kept``), but nothing is
ever sent because of it: the send gate looks only at the linked account's own
store. An old number's chats are first contacts for a new number, and first
contacts are what gets a number banned.

Plain files and sqlite3, so Flight Deck can do this itself while the service is
off (:meth:`..whatsapp_web_supervisor.Supervisor.forget_local`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from captain_claw.flight_deck.whatsapp_web.link_state import now_iso, number_of

log = logging.getLogger("whatsapp_web.kept_history")

KEPT_DIR = ".kept"
STORE_FILES = ("store.sqlite3", "store.sqlite3-wal", "store.sqlite3-shm", "store.sqlite3-journal")
_ENTRY = re.compile(r"^\d{8}T\d{6}-[0-9a-f]{8}$")


def _user_dir(root: Path, user_id: str) -> Path:
    return Path(root) / KEPT_DIR / user_id


def _no_stats() -> dict[str, Any]:
    return {"messages": 0, "chats": 0, "oldest": None, "newest": None}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _prepare(path: Path) -> tuple[dict[str, Any] | None, int | None]:
    """A closed store's counts and retention choice ((None, None) when it can't
    be read). Its auto-reply log goes: nothing shows it once the link is gone."""
    try:
        con = sqlite3.connect(str(path))
        try:
            row = con.execute(
                "SELECT COUNT(*), MIN(CASE WHEN ts > 0 THEN ts END), MAX(ts) FROM messages").fetchone()
            chats = con.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            days = con.execute("SELECT value FROM meta WHERE key='retention_days'").fetchone()
            try:
                con.execute("DELETE FROM autoreply_log")
                con.commit()
            except sqlite3.OperationalError:
                pass  # a store from before the log existed
        finally:
            con.close()
    except sqlite3.Error as exc:
        log.warning("whatsapp_web: reading %s to keep it: %s", path, exc)
        return None, None
    stats = {"messages": int(row[0] or 0), "chats": int(chats or 0),
             "oldest": int(row[1] or 0) or None, "newest": int(row[2] or 0) or None}
    return stats, int(days[0]) if days and str(days[0]).isdigit() else None


def archive(root: Path, user_id: str, link_dir: Path, *, jid: str = "", linked_at: str = "",
            reason: str = "", now: float | None = None) -> str | None:
    """Move the link's messages (its store must be closed) into its user's kept
    folder; the entry's id, or None when there was nothing worth keeping (no
    store, or no message in it, which is then deleted). An OSError means they
    are still where they were: the caller must not delete the link folder."""
    source = Path(link_dir) / "store.sqlite3"
    if not source.exists():
        return None
    stats, days = _prepare(source)
    if stats is not None and not stats["messages"]:
        for name in STORE_FILES:
            (Path(link_dir) / name).unlink(missing_ok=True)
        return None
    moment = datetime.fromtimestamp(now, UTC) if now is not None else datetime.now(UTC)
    entry = f"{moment.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}"
    target = _user_dir(root, user_id) / entry
    # One at a time: mkdir's mode doesn't reach the parents it creates.
    for folder in (Path(root) / KEPT_DIR, _user_dir(root, user_id), target):
        folder.mkdir(mode=0o700, exist_ok=True)
    _write_json(target / "kept.json", {
        "number": number_of(jid), "jid": jid, "linked_at": linked_at, "reason": reason,
        "kept_at": now_iso(now), "retention_days": days, "stats": stats or _no_stats(),
    })
    # The main file last: an entry counts once it's there (see entries).
    for name in reversed(STORE_FILES):
        if (Path(link_dir) / name).exists():
            os.chmod(Path(link_dir) / name, 0o600)
            (Path(link_dir) / name).replace(target / name)
    return entry


def entries(root: Path, user_id: str) -> list[dict[str, Any]]:
    """What's kept for *user_id*, newest first. Reads no store, so it needs
    neither the service nor the data key."""
    try:
        folders = [d for d in _user_dir(root, user_id).iterdir() if _ENTRY.match(d.name)]
    except OSError:
        return []
    out = []
    for folder in sorted(folders, key=lambda d: d.name, reverse=True):
        if not (folder / "store.sqlite3").exists():
            continue
        meta = _read_json(folder / "kept.json")
        out.append({
            "id": folder.name, "number": str(meta.get("number") or ""), "jid": str(meta.get("jid") or ""),
            "linked_at": meta.get("linked_at") or None, "kept_at": meta.get("kept_at") or None,
            "reason": str(meta.get("reason") or ""), "retention_days": meta.get("retention_days"),
            "stats": meta.get("stats") if isinstance(meta.get("stats"), dict) else _no_stats(),
        })
    return out


def store_path(root: Path, user_id: str, entry_id: str) -> Path:
    return _user_dir(root, user_id) / entry_id / "store.sqlite3"


def users(root: Path) -> list[str]:
    """Everyone with something kept."""
    try:
        return sorted(d.name for d in (Path(root) / KEPT_DIR).iterdir() if d.is_dir())
    except OSError:
        return []


def set_stats(root: Path, user_id: str, entry_id: str, stats: dict[str, Any]) -> None:
    """The counts after a retention purge took some of it."""
    path = _user_dir(root, user_id) / entry_id / "kept.json"
    _write_json(path, {**_read_json(path), "stats": stats})


def discard(root: Path, user_id: str, entry_id: str = "") -> int:
    """Delete one entry (all of the user's without *entry_id*); how many went."""
    gone = 0
    for entry in entries(root, user_id):
        if not entry_id or entry["id"] == entry_id:
            shutil.rmtree(_user_dir(root, user_id) / entry["id"], ignore_errors=True)
            gone += 1
    try:
        _user_dir(root, user_id).rmdir()  # only when that was the last one
    except OSError:
        pass
    return gone
