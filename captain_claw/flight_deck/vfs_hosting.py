"""VFS Hosting — publish a VFS folder as a static site or run a built app.

Two kinds of published entry, both reachable at a public, globally-unique name:

* ``static`` — the folder's files are served directly at ``/vfs/<name>/…``.
* ``app``    — a build's start command runs as a managed subprocess bound to a
  Flight-Deck-assigned ``PORT`` on 127.0.0.1, reverse-proxied at
  ``/vfs-apps/<name>/…`` (HTTP + WebSocket).

Serving is public (no login); *publishing* is owner-gated at the route layer.
This module is pure state + process control — the routes live in
``hosting_routes.py``, and who may start an app, with what, is decided there
(``hosting_routes.launch``). Server-side helpers (``DATA_DIR``, killer) are
imported lazily to avoid an import cycle with ``server.py``.

An app's port comes from one range, ``VFS_APPS_PORT_BASE`` (26100) and the
``VFS_APPS_PORT_SPAN`` (100) ports after it.
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
    # "agent" would be /fd/hosting/agent/…, the routes agents call.
    return bool(_NAME_RE.match(name or "")) and name != "agent"


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


def _in_deck_internals(owner: str, base: Path) -> bool:
    """Is ``base`` FD's data dir, an ancestor of it, or inside it outside the
    owner's own VFS root? A published folder can only land there through a
    linked project whose registry entry ``POST /fd/vfs/links`` would have
    refused (a ``.vfs-links.json`` edited on disk) — and serving is public. The
    same read-time rule as ``vfs_routes.safe_link_target``."""
    from captain_claw.flight_deck.vfs_routes import path_within
    from captain_claw.vfs import user_root_of

    try:
        data = _data_dir().resolve()
        own = user_root_of(owner).resolve()
        b = base.resolve()
        if path_within(b, own):
            return False
        return path_within(b, data) or path_within(data, b)
    except (OSError, RuntimeError):
        return True


def entry_dir(entry: dict) -> Path | None:
    """Resolve a published entry's on-disk base directory (owner-scoped, sandboxed)."""
    base = resolve_under(entry.get("owner", ""), entry.get("project", ""),
                         _proj_path(entry.get("project", ""), entry.get("subdir", "")))
    if base is None or not base.is_dir():
        return None
    if _in_deck_internals(str(entry.get("owner", "") or ""), base):
        return None
    return base


# A folder published whole may be a repository: its history and secrets aren't the site.
_NEVER_SERVED = frozenset({".git", ".env", ".code", ".captain-claw"})


def resolve_static_file(entry: dict, subpath: str) -> Path | None:
    """Resolve a request subpath under a static entry's base dir, sandboxed."""
    base = entry_dir(entry)
    if base is None:
        return None
    parts = [p for p in (subpath or "").replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if any(p in _NEVER_SERVED for p in parts):
        return None
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


def port_range() -> range:
    def number(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except ValueError:
            return default

    base = number("VFS_APPS_PORT_BASE", 26100)
    return range(base, base + max(1, number("VFS_APPS_PORT_SPAN", 100)))


def _free_port(reg: dict, name: str) -> int | None:
    """The first port of the range that no other app holds and nothing listens on."""
    import socket

    held = {int(e["port"]) for n, e in reg.items() if n != name and e.get("kind") == "app" and e.get("port")}
    for port in port_range():
        if port in held:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    return None


def running_apps(*, owner: str = "", reach: str = "") -> list[str]:
    """Names of the apps running now, of one *owner* and/or started with one *reach*."""
    return [n for n, e in load_registry().items()
            if e.get("kind") == "app" and (not owner or e.get("owner") == owner)
            and (not reach or e.get("reach", "deck") == reach) and app_is_alive(n)]


def _tree_mb(root: int, by_pid: dict[int, dict], children: dict[int, list[int]]) -> float:
    total, seen, todo = 0.0, set(), [root]
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += float((by_pid.get(pid) or {}).get("rss_mb") or 0.0)
        todo.extend(children.get(pid, ()))
    return total


def police(memory_mb: int, hours: int) -> list[tuple[str, str]]:
    """Stop each running non-admin's app (``reach`` "crew") that is past *memory_mb*,
    counting everything it started, or has run *hours*; 0 is no limit. Each stop is
    said in its output. Returns ``(name, why)``. A process that leaves its parent
    behind (a daemon) isn't counted: it's out of the tree ``ps`` shows."""
    names = running_apps(reach="crew")
    if not names or not (memory_mb or hours):
        return []
    by_pid: dict[int, dict] = {}
    children: dict[int, list[int]] = {}
    if memory_mb:
        from captain_claw.flight_deck.system_routes import _snapshot_processes

        by_pid, children = _snapshot_processes()
    reg = load_registry()
    now = time.time()
    stopped = []
    for name in names:
        ent = reg.get(name) or {}
        why = ""
        started = float(ent.get("started_at") or 0)
        if hours and started and now - started > hours * 3600:
            why = f"it ran for {hours} hour{'s' if hours != 1 else ''}, as long as an app may run here at a time"
        elif memory_mb and ent.get("pid") and by_pid:
            used = _tree_mb(int(ent["pid"]), by_pid, children)
            if used > memory_mb:
                why = f"it used {used:.0f} MB of memory, more than the {memory_mb} MB an app may use here"
        if why:
            stop_app(name)
            note_in_log(name, f"Stopped: {why}.")
            stopped.append((name, why))
    return stopped


# A non-admin's app starts below the deck in the CPU's queue and writes no core
# dumps. A shell in front does it, so nothing runs in Python between fork and exec
# (preexec_fn isn't safe in a threaded server). The command is $0, never quoted in.
_HELD_BACK = ('ulimit -c 0 2>/dev/null; command -v nice >/dev/null 2>&1 && exec nice -n 10 /bin/sh -c "$0"; '
              'exec /bin/sh -c "$0"')


def start_app(name: str, *, env: dict[str, str] | None = None, reach: str = "deck") -> tuple[bool, str]:
    """Start *name*. *env* is the whole environment it gets (the owner's, chosen by
    ``hosting_routes.launch``); with none, Flight Deck's own minus its secrets, as
    an admin's app always had. ``reach="crew"`` also holds it back (``_HELD_BACK``)."""
    reg = load_registry()
    ent = reg.get(name)
    if not ent:
        return False, "not found"
    if ent.get("kind") != "app":
        # A static site has no process: starting it only puts it back online.
        ent.pop("stopped", None)
        save_registry(reg)
        return True, "back online"
    if app_is_alive(name):
        return True, "already running"
    cmd = (ent.get("start_cmd") or "").strip()
    if not cmd:
        return False, "no start command configured"
    folder = entry_dir(ent)
    if folder is None:
        return False, "app folder not found"

    port = _free_port(reg, name)
    if port is None:
        span = port_range()
        return False, f"no free port for it ({span.start}-{span.stop - 1} are all taken)"

    if env is None:
        from captain_claw.flight_deck.server import _FD_ONLY_ENV_VARS

        # FD's environment minus its own secrets (the list agents are spawned
        # without): the start command is a tenant's code, and FD_JWT_SECRET alone
        # would let it mint a session for any user of this deck, admins included.
        env = {k: v for k, v in os.environ.items() if k not in _FD_ONLY_ENV_VARS}
    else:
        env = dict(env)
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
    logf = None
    try:
        logf = open(log_path / f"{name}.log", "a")
        # Through a shell so a natural start command (e.g. `npm run start`,
        # `python app.py`) works; the command is the owner's own.
        if reach == "crew":
            proc = subprocess.Popen(  # noqa: S603 — the command is $0 of a fixed script
                ["/bin/sh", "-c", _HELD_BACK, cmd], cwd=str(folder), env=env,
                stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
            )
        else:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=str(folder), env=env,
                stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
            )
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to launch: {exc}"
    finally:
        if logf is not None:
            logf.close()  # the app has its own copy

    _procs[name] = proc
    ent["port"] = port
    ent["pid"] = proc.pid
    ent["started_at"] = int(time.time())
    ent["reach"] = reach
    ent.pop("stopped", None)
    reg[name] = ent
    save_registry(reg)
    return True, f"started (pid {proc.pid}, port {port})"


def note_in_log(name: str, text: str) -> None:
    """A line of Flight Deck's own in an app's output, where its owner looks."""
    folder = _data_dir() / "vfs-apps-logs"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{name}.log", "a") as logf:
            logf.write(f"\n[Flight Deck] {text}\n")
    except OSError:
        pass


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
    """Return the entry with a live ``running`` flag, clearing stale pid/port. A
    static site is running until it's stopped."""
    out = dict(entry)
    out["running"] = not entry.get("stopped")
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
