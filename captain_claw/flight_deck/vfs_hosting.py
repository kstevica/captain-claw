"""VFS Hosting — publish a VFS folder as a static site or run a built app.

Two kinds of published entry, both reachable at a public, globally-unique name:

* ``static`` — the folder's files are served directly at ``/vfs/<name>/…``.
* ``app``    — a build's start command runs as a managed subprocess bound to a
  Flight-Deck-assigned ``PORT`` on 127.0.0.1, reverse-proxied at
  ``/vfs-apps/<name>/…`` (HTTP + WebSocket).

Serving is public (no login); *publishing* is owner-gated at the route layer.
This module is pure state + process control — the routes live in
``hosting_routes.py``. Server-side helpers (``DATA_DIR``, port picker, killer)
are imported lazily to avoid an import cycle with ``server.py``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path

from captain_claw.vfs import resolve_under

# name: lowercase alnum + dashes, 1–63 chars, must start alnum. Keeps the URL
# path segment clean and collision-checkable.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")

# In-memory Popen handles for running apps (name -> Popen). Survives within one
# server process; the on-disk registry is the source of truth across restarts.
_procs: dict[str, subprocess.Popen] = {}

# In-memory recent visits + running totals for static sites (per name). Reset on
# server restart — cheap, no per-request disk writes on a public path.
_VISITS: dict[str, deque] = {}
_VISIT_COUNTS: dict[str, int] = {}


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


def _data_dir() -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    return DATA_DIR


def _registry_file() -> Path:
    return _data_dir() / ".vfs-hosting.json"


def load_registry() -> dict:
    f = _registry_file()
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_registry(reg: dict) -> None:
    f = _registry_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f.suffix + ".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    os.replace(str(tmp), str(f))


def _proj_path(project: str, subdir: str) -> str:
    """Build the vfs-relative path string ``project[/subdir]`` for resolve_under.

    We always lead with the project so resolve_under treats the first segment as
    the project and the rest as the in-project path (handles multi-segment
    subdirs correctly, unlike passing a bare subdir)."""
    # Always keep the trailing slash so resolve_under parses the first segment
    # as the project (a bare "proj" would be read as a rel path under the
    # default project). "proj/" → project root; "proj/dist" → project/dist.
    sub = (subdir or "").strip("/")
    return f"{project}/{sub}"


def entry_dir(entry: dict) -> Path | None:
    """Resolve a published entry's on-disk base directory (owner-scoped, sandboxed)."""
    base = resolve_under(entry.get("owner", ""), entry.get("project", ""),
                         _proj_path(entry.get("project", ""), entry.get("subdir", "")))
    if base is None or not base.is_dir():
        return None
    return base


def resolve_static_file(entry: dict, subpath: str) -> Path | None:
    """Resolve a request subpath under a static entry's base dir, sandboxed."""
    base = entry_dir(entry)
    if base is None:
        return None
    parts = [p for p in (subpath or "").replace("\\", "/").split("/") if p not in ("", ".", "..")]
    target = base
    for part in parts:
        target = target / part
    target = target.resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        return None
    return target


# ── App process control ──────────────────────────────────────────────────

def app_is_alive(name: str) -> bool:
    p = _procs.get(name)
    if p and p.poll() is None:
        return True
    ent = load_registry().get(name) or {}
    pid = ent.get("pid")
    if pid:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ProcessLookupError, ValueError):
            pass
    return False


def app_port(name: str) -> int | None:
    ent = load_registry().get(name)
    if not ent or ent.get("kind") != "app":
        return None
    port = ent.get("port")
    return int(port) if port else None


def start_app(name: str) -> tuple[bool, str]:
    reg = load_registry()
    ent = reg.get(name)
    if not ent or ent.get("kind") != "app":
        return False, "not an app"
    if app_is_alive(name):
        return True, "already running"
    cmd = (ent.get("start_cmd") or "").strip()
    if not cmd:
        return False, "no start command configured"
    folder = entry_dir(ent)
    if folder is None:
        return False, "app folder not found"

    from captain_claw.flight_deck.server import _find_available_port
    port = _find_available_port(int(os.environ.get("VFS_APPS_PORT_BASE", "26100")))

    env = dict(os.environ)
    # The app must bind the port we assign, on localhost. We set several common
    # conventions so most frameworks pick it up without extra config.
    env["PORT"] = str(port)
    env["HOST"] = "127.0.0.1"
    env["HOSTNAME"] = "127.0.0.1"
    env["FD_VFS_APP"] = name
    # The public path this app is reverse-proxied under. Apps should prefix
    # absolute asset/API URLs with this so requests come back through the proxy
    # instead of hitting the Flight Deck root.
    env["FD_BASE_PATH"] = f"/vfs-apps/{name}/"

    log_path = _data_dir() / "vfs-apps-logs"
    log_path.mkdir(parents=True, exist_ok=True)
    try:
        logf = open(log_path / f"{name}.log", "a")
        # shell=True so a natural start command (e.g. `npm run start`,
        # `python app.py`) works; the command is the owner's own, on their box.
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(folder), env=env,
            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to launch: {exc}"

    _procs[name] = proc
    ent["port"] = port
    ent["pid"] = proc.pid
    ent["started_at"] = int(time.time())
    ent.pop("stopped", None)
    reg[name] = ent
    save_registry(reg)
    return True, f"started (pid {proc.pid}, port {port})"


def stop_app(name: str) -> tuple[bool, str]:
    from captain_claw.flight_deck.server import _kill_pid
    reg = load_registry()
    ent = reg.get(name)
    p = _procs.get(name)
    pid = p.pid if p else (ent or {}).get("pid")
    if pid:
        _kill_pid(int(pid))
    _procs.pop(name, None)
    if ent:
        ent["pid"] = None
        ent["port"] = None
        ent["stopped"] = True
        reg[name] = ent
        save_registry(reg)
    return True, "stopped"


def read_app_log(name: str, tail: int = 300) -> str:
    """Return the last ``tail`` lines of an app's captured stdout/stderr log."""
    f = _data_dir() / "vfs-apps-logs" / f"{name}.log"
    if not f.is_file():
        return ""
    try:
        return "\n".join(f.read_text(errors="replace").splitlines()[-tail:])
    except OSError:
        return ""


def record_visit(name: str, ip: str, path: str, ua: str) -> None:
    dq = _VISITS.get(name)
    if dq is None:
        dq = deque(maxlen=100)
        _VISITS[name] = dq
    dq.appendleft({"ip": ip, "path": path, "ua": ua, "at": int(time.time())})
    _VISIT_COUNTS[name] = _VISIT_COUNTS.get(name, 0) + 1


def get_visits(name: str) -> dict:
    return {"count": _VISIT_COUNTS.get(name, 0), "visits": list(_VISITS.get(name, []))}


def reconcile(name: str, entry: dict) -> dict:
    """Return the entry with a live ``running`` flag, clearing stale pid/port."""
    out = dict(entry)
    if entry.get("kind") == "app":
        alive = app_is_alive(name)
        out["running"] = alive
        if not alive and (entry.get("pid") or entry.get("port")):
            # Process died out from under us — clear the stale registration.
            reg = load_registry()
            if name in reg:
                reg[name]["pid"] = None
                reg[name]["port"] = None
                save_registry(reg)
            out["pid"] = None
            out["port"] = None
    return out
