"""Flight Deck updates itself: stop the agents, pull the new version, restart.

A deck whose code is a git checkout follows the branch it's on (its upstream, so a
fork follows the fork). Only an admin starts one: a click in Console → Updates, an
Approve on what their agent asked (``crew action=update_deck``), or their switch to
let the deck install by itself. It only ever fast-forwards, to the commit that admin
saw, and refuses before anything has been touched when it can't.

The order, and what each step is there for:

1. **Rehearse.** The new commit is unpacked to a temp folder and started on a spare
   port, with an empty data folder and none of the deck's secrets, and its agents'
   code is imported. A version that can't start ends here, with nothing changed.
2. **Count down.** Everyone signed in has been told an update is coming since step 1
   began (a toast, :func:`notice`). Now they get a toast they can't close with a
   countdown, 2 minutes by default, and nothing is stopped until it ends.
3. **Stop the agents**, because they read their code from the checkout. Optionally
   after waiting for the ones mid-turn. Crew's workers keep going through this.
4. **Pull**, with ``deck_update_guard.py`` already running outside the deck, then
   ``pip install`` if the venv needs it (:func:`_install_args`).
5. **Restart in place**: uvicorn shuts down, then the process execs itself with what
   it was started with. Same pid, same terminal, same tmux pane. ``.env`` is read as on
   any start, so what the shell exported wins over an edit to the file. The agents come
   back the way they do on any deck start.
6. **Verdict.** The new version writes ``booted``; the guard checks it answers. If it
   doesn't, the guard puts the old commit back and starts that (see its docstring).
   Whoever started it hears in the bell, and every admin gets an email, from the first
   admin's connected Gmail or Outlook that can send it.

``<data>/update/state.json`` is the one record, and it outlives the restart.

``FD_UPDATE_REPO`` names the checkout when it can't be worked out, ``FD_UPDATE_POLL_S``
is how often the switch's check runs (default 15 minutes), ``FD_UPDATE_BOOT_TIMEOUT_S``
how long the new version gets to come up (default 5 minutes), ``FD_UPDATE_COUNTDOWN_S``
how long the countdown is (default 2 minutes).
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

# Set from server.py, which runs as __main__: an import of `server` here is a second
# copy, with its own uvicorn (none) and its own table of processes.
stop_agents: Callable[[], None] | None = None
start_agents: Callable[[], None] | None = None
_exit_server: Callable[[], None] | None = None
_startup_env: dict[str, str] = {}

_GIT_FLAGS = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "protocol.ext.allow=never")
_ACTIVE = ("rehearsing", "countdown", "waiting", "stopping", "pulling", "restarting", "booted", "rolling_back")
_MODES = ("off", "notify", "install")
_SETTING = "fd:deck-update"
_WAIT_FOR_TURNS_S = 300.0
_REHEARSAL_TIMEOUT_S = 240.0
# Pages look every 15 seconds, and a test start can take 6: long enough that each sees "soon".
_SOON_S = 20.0
_SHUTDOWN_TIMEOUT_S = 60.0
_MAX_COMMITS = 30
STARTUP_FAILURE = 3  # uvicorn's exit code for a start that failed

_task: asyncio.Task | None = None
_relaunch_asked = threading.Event()
_checked_at = 0.0


class RefusedError(Exception):
    """Why an update can't go ahead, in words for the admin."""


def _boot_timeout() -> float:
    return float(os.environ.get("FD_UPDATE_BOOT_TIMEOUT_S", "300") or 300)


def _countdown_s() -> float:
    return float(os.environ.get("FD_UPDATE_COUNTDOWN_S", "120") or 0)


def _poll_s() -> float:
    return float(os.environ.get("FD_UPDATE_POLL_S", "900") or 900)


def _srv():
    # Lazy: server.py imports this module.
    from captain_claw.flight_deck import server

    return server


# ── The record ──────────────────────────────────────────────────────


def _dir() -> Path:
    return Path(_srv().DATA_DIR) / "update"


def _state_path() -> Path:
    return _dir() / "state.json"


def read_state() -> dict[str, Any]:
    try:
        state = json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {"phase": "idle"}
    return state if isinstance(state, dict) else {"phase": "idle"}


def _write_state(**changes: Any) -> dict[str, Any]:
    state = {**read_state(), **changes}
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # It holds the environment the guard restarts the deck with.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(state, out, indent=1)
    os.replace(tmp, path)
    return state


def _step(text: str, **changes: Any) -> None:
    state = read_state()
    _write_state(steps=[*(state.get("steps") or []), {"at": time.time(), "text": text}], **changes)
    log.info("Deck update: %s", text)


def _view(state: dict[str, Any]) -> dict[str, Any]:
    """What the page gets. The environment, the command line and the paths stay here."""
    keep = ("phase", "from", "to", "by", "started_at", "finished_at", "steps", "error", "refused", "detached_log", "mail")
    view = {k: state[k] for k in keep if k in state}
    if state.get("phase") == "countdown":
        view["seconds_left"] = _seconds_left(state)
    return view


def _seconds_left(state: dict[str, Any]) -> int:
    return max(0, round(float(state.get("restart_at") or 0) - time.time()))


def notice() -> dict[str, Any]:
    """What everyone signed in is told, whoever they are: no versions, no names. *level* is ``soon``
    (it's being tried, a toast they can close) or ``restart`` (the countdown and after, a toast they
    can't). Seconds rather than a time, because a browser's clock isn't the deck's. *last* is how the
    last one ended, for a page that was following it. *id* tells one update from the next."""
    state = read_state()
    phase, update_id = state.get("phase"), state.get("started_at")
    if phase == "rehearsing":
        return {"level": "soon", "id": update_id}
    if phase in _ACTIVE:
        return {"level": "restart", "id": update_id, "seconds_left": _seconds_left(state) if phase == "countdown" else 0}
    return {"level": "none", "last": {"id": update_id, "phase": phase} if update_id else None}


# ── The checkout ────────────────────────────────────────────────────


def find_repo() -> tuple[Path | None, bool]:
    """The checkout this deck runs from, and whether its code *is* that checkout (an editable
    install). Otherwise the venv holds a copy, and that has to be installed again too."""
    if getattr(sys, "_MEIPASS", None):
        return None, False
    import captain_claw

    package = Path(captain_claw.__file__).resolve().parent.parent
    for candidate in (os.environ.get("FD_UPDATE_REPO"), package, Path.cwd()):
        if not candidate:
            continue
        where = Path(candidate).resolve()
        if (where / ".git").exists() and (where / "captain_claw" / "flight_deck" / "server.py").is_file():
            return where, where == package
    return None, False


def _git_env() -> dict[str, str]:
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


async def _git(repo: Path, *args: str, timeout: float = 120.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(repo), *_GIT_FLAGS, *args, env=_git_env(), stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"git {args[0]} took too long"
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace").strip()


def _clean_url(url: str) -> str:
    """A remote's address, without a password or token someone put in it."""
    if "://" not in url:
        return url
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


async def _commits(repo: Path, span: str) -> list[dict[str, str]]:
    rc, out = await _git(repo, "log", f"--max-count={_MAX_COMMITS}", "--format=%H%x1f%s%x1f%an%x1f%cI", span)
    if rc:
        return []
    rows = (line.split("\x1f") for line in out.splitlines())
    return [{"sha": r[0], "title": r[1], "author": r[2], "date": r[3]} for r in rows if len(r) == 4]


def _requirements(pyproject: str) -> set[str]:
    try:
        return set(tomllib.loads(pyproject).get("project", {}).get("dependencies", []))
    except tomllib.TOMLDecodeError:
        return set()


async def _new_requirements(repo: Path, head: str, target: str) -> list[str]:
    _, old = await _git(repo, "show", f"{head}:pyproject.toml")
    rc, new = await _git(repo, "show", f"{target}:pyproject.toml")
    return [] if rc else sorted(_requirements(new) - _requirements(old))


async def _blockers(repo: Path, head: str, upstream: str, target: str) -> list[str]:
    """What stops this update, found without touching anything."""
    out: list[str] = []
    if _exit_server is None:
        out.append("This deck wasn't started with the flight-deck command, so it can't restart itself.")
    rc, _ = await _git(repo, "merge-base", "--is-ancestor", head, target)
    if rc:
        out.append(f"This checkout has commits that {upstream} doesn't, so it can't simply move forward. "
                   "Put them on the branch, or remove them.")
        return out
    rc, _ = await _git(repo, "rev-parse", "-q", "--verify", "MERGE_HEAD")
    if rc == 0:
        out.append("A merge is under way in this checkout. Finish or abort it first.")
    rc, said = await _git(repo, "read-tree", "--dry-run", "-m", "-u", head, target)
    if rc:
        out.append(f"Files here would be overwritten: {said[-400:]}")
    return out


async def _notes(repo: Path, head: str, target: str) -> list[str]:
    """What the admin should know about these commits, none of it a reason to stop."""
    _, changed = await _git(repo, "diff", "--name-only", head, target)
    files = changed.splitlines()
    notes: list[str] = []
    if any(f.startswith("flight-deck/src/") for f in files) and not any(
            f.startswith("captain_claw/flight_deck/static/") for f in files):
        notes.append("The page's source changed but its built bundle didn't, so those changes won't show until "
                     "someone builds and commits it.")
    if any("whatsapp_web" in f for f in files):
        notes.append("WhatsApp Web's service outlives a restart, so it stays on the old version until it's restarted.")
    if await _new_requirements(repo, head, target):
        notes.append("New packages are needed. They're installed before the test start.")
    _, local = await _git(repo, "status", "--porcelain", "--untracked-files=no")
    if local:
        notes.append(f"{len(local.splitlines())} file(s) here have changes that aren't committed. They stay as they are.")
    return notes


# ── Who's working ───────────────────────────────────────────────────


def _running_agents() -> list[tuple[str, dict]]:
    srv = _srv()
    return [(slug, entry) for slug, entry in srv._load_process_registry().items()
            if entry.get("pid") and srv._process_is_alive(slug)]


async def busy_agents() -> list[str]:
    """The names of agents in a turn right now, or with a scheduled run in flight."""
    from captain_claw.flight_deck import agent_idle

    running = _running_agents()
    activity = await asyncio.gather(*(agent_idle._agent_activity(entry) for _, entry in running))
    return [entry.get("name") or slug for (slug, entry), seen in zip(running, activity, strict=True)
            if (seen or {}).get("busy") or agent_idle._scheduler_run_in_flight(slug)]


def _crew_running() -> int:
    try:
        from captain_claw.flight_deck import crew_store

        return len(crew_store.get_store().with_status("running"))
    except Exception:  # noqa: BLE001 — a count for a sentence on a page
        return 0


# ── What the page shows ─────────────────────────────────────────────


async def get_mode(db: Any) -> str:
    raw = await db.get_system_setting(_SETTING)
    try:
        mode = json.loads(raw).get("mode") if raw else "off"
    except (ValueError, AttributeError):
        mode = "off"
    return mode if mode in _MODES else "off"


async def set_mode(db: Any, mode: str) -> str:
    if mode not in _MODES:
        raise RefusedError(f"Updates can be {', '.join(_MODES)}.")
    await db.set_system_setting(_SETTING, json.dumps({"mode": mode}))
    return mode


async def status(*, fetch: bool = False) -> dict[str, Any]:
    """Where this deck is and what could come next. *fetch* asks the remote; without it this
    is what the last fetch brought."""
    global _checked_at

    job = _view(read_state())
    if in_progress():
        # The page is only following it, every couple of seconds.
        return {"available": True, "active": True, "job": job}
    repo, editable = find_repo()
    if repo is None:
        return {"available": False, "why": "This deck wasn't installed from a git checkout, so there's nothing to pull. "
                                           "Update it the way it was installed.", "job": job}
    rc, branch = await _git(repo, "symbolic-ref", "--short", "-q", "HEAD")
    if rc:
        return {"available": False, "why": f"The checkout at {repo} isn't on a branch, so there's nothing to follow.",
                "job": job}
    rc, upstream = await _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if rc:
        return {"available": False, "why": f"The branch {branch} at {repo} has no upstream, so there's nothing to follow. "
                                           f"Set one: git branch --set-upstream-to=origin/{branch}", "job": job}
    _, remote = await _git(repo, "config", "--get", f"branch.{branch}.remote")
    _, url = await _git(repo, "remote", "get-url", remote)
    out: dict[str, Any] = {"available": True, "active": False, "repo": str(repo), "editable": editable, "branch": branch,
                           "upstream": upstream, "remote_url": _clean_url(url) if remote != "." else "this repository", "job": job}
    if fetch and remote != ".":
        rc, said = await _git(repo, "fetch", "--quiet", remote, timeout=180)
        if rc:
            out["fetch_error"] = (f"Couldn't read {out['remote_url']}: {said[-300:]} The deck uses this computer's git "
                                  "login, so a private repository needs one that works without a prompt.")
        else:
            _checked_at = time.time()
    _, head = await _git(repo, "rev-parse", "HEAD")
    _, target = await _git(repo, "rev-parse", upstream)
    _, title = await _git(repo, "log", "-1", "--format=%s", head)
    _, behind = await _git(repo, "rev-list", "--count", f"{head}..{target}")
    out.update(head=head, head_title=title, target=target, behind=int(behind or 0), checked_at=_checked_at,
               commits=[], blockers=[], notes=[])
    if out["behind"]:
        out["commits"] = await _commits(repo, f"{head}..{target}")
        out["blockers"] = await _blockers(repo, head, upstream, target)
        out["notes"] = await _notes(repo, head, target)
        if read_state().get("refused") == target:
            out["notes"].insert(0, "The last try at this version didn't go through.")
    out["agents_running"] = len(_running_agents())
    out["agents_busy"] = await busy_agents()
    out["crew_running"] = _crew_running()
    return out


# ── The rehearsal ───────────────────────────────────────────────────


def bundle_problems(code: Path) -> list[str]:
    """The page's files that the page asks for and the commit doesn't have: a blank page, and with
    it no way to reach this page again. Follows from the HTML through every chunk."""
    static = code / "captain_claw" / "flight_deck" / "static"
    assets = static / "assets"
    queue = [name for page in static.glob("*.html")
             for name in re.findall(r"assets/([\w.\-]+\.(?:js|css))", page.read_text(errors="ignore"))]
    seen: set[str] = set()
    missing: list[str] = []
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        file = assets / name
        if not file.is_file():
            missing.append(name)
        elif name.endswith(".js"):
            queue += re.findall(r"(?:assets/|\./)([\w.\-]+-[\w\-]{8}\.(?:js|css))", file.read_text(errors="ignore"))
    return sorted(missing)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rehearsal_env(top: Path, code: Path) -> dict[str, str]:
    """Nothing of the deck's: no keys, no bots, no callbacks, nothing that could answer for it."""
    keep = ("PATH", "LANG", "LC_ALL", "TMPDIR", "USER", "LOGNAME", "SSL_CERT_FILE", "SSL_CERT_DIR")
    env = {k: os.environ[k] for k in keep if os.environ.get(k)}
    env.update(HOME=str(top / "home"), PYTHONPATH=str(code), FD_DATA_DIR=str(top / "data"),
               FD_JWT_SECRET=secrets.token_hex(32), PYTHONDONTWRITEBYTECODE="1")
    return env


def _end_group(proc: subprocess.Popen, wait_s: float = 5.0) -> None:
    import signal

    for sig, wait in ((signal.SIGTERM, wait_s), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


async def _boots(env: dict[str, str], top: Path) -> None:
    port = _free_port()
    log_path = top / "boot.log"
    with log_path.open("wb") as out:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "captain_claw.flight_deck.server", "--host", "127.0.0.1", "--port", str(port)],
            cwd=top / "run", env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )

    def what_it_said() -> str:
        return log_path.read_text(errors="replace")[-800:].strip()

    try:
        deadline = time.monotonic() + _REHEARSAL_TIMEOUT_S
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RefusedError(f"The new version doesn't start: {what_it_said()}")
                try:
                    health, page = await client.get(f"http://127.0.0.1:{port}/fd/health"), await client.get(f"http://127.0.0.1:{port}/")
                except httpx.HTTPError:
                    await asyncio.sleep(1.0)
                    continue
                if health.status_code == 200 and page.status_code == 200:
                    return
                raise RefusedError(f"The new version started but doesn't answer properly: /fd/health {health.status_code}, "
                              f"the page {page.status_code}.")
        raise RefusedError(f"The new version didn't answer within {int(_REHEARSAL_TIMEOUT_S)} seconds: {what_it_said()}")
    finally:
        await asyncio.to_thread(_end_group, proc)


async def rehearse(repo: Path, target: str) -> None:
    """Refuses (:class:`RefusedError`) a commit that can't start. Touches neither checkout nor data."""
    top = Path(tempfile.mkdtemp(prefix="deck-rehearsal-"))
    try:
        code = top / "code"
        for name in ("code", "home", "data", "run"):
            (top / name).mkdir()
        unpack = await asyncio.create_subprocess_shell(
            'git -C "$REPO" archive --format=tar "$SHA" captain_claw | tar -x -C "$CODE"',
            env={**_git_env(), "REPO": str(repo), "SHA": target, "CODE": str(code)},
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        said, _ = await unpack.communicate()
        if unpack.returncode:
            raise RefusedError(f"Couldn't unpack the new version: {said.decode(errors='replace')[-300:]}")
        missing = await asyncio.to_thread(bundle_problems, code)
        if missing:
            raise RefusedError(f"The new version's page is missing {len(missing)} of its files ({', '.join(missing[:3])}), so it "
                          "would open blank. Build the bundle and commit all of it.")
        env = _rehearsal_env(top, code)
        agents = await asyncio.to_thread(
            subprocess.run, [sys.executable, "-c", "import captain_claw.web_server"], cwd=top / "run", env=env,  # noqa: S603
            capture_output=True, text=True, timeout=_REHEARSAL_TIMEOUT_S, stdin=subprocess.DEVNULL, check=False,
        )
        if agents.returncode:
            raise RefusedError(f"The new version's agents wouldn't start: {agents.stderr[-800:].strip()}")
        await _boots(env, top)
    except subprocess.TimeoutExpired as exc:
        raise RefusedError("Importing the new version's agent code took too long.") from exc
    finally:
        shutil.rmtree(top, ignore_errors=True)


async def _pip(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pip", "install", *args, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 900)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "pip took too long"
    return proc.returncode or 0, (out or b"").decode("utf-8", "replace").strip()


# ── The update ──────────────────────────────────────────────────────


def in_progress() -> bool:
    return read_state().get("phase") in _ACTIVE


async def start(*, by: str, target: str, wait: bool = False, user_id: str = "", agent: dict | None = None) -> dict[str, Any]:
    """Begin an update to exactly *target*, the commit the admin saw; a newer one isn't covered."""
    global _task

    if in_progress():
        raise RefusedError("An update is already under way.")
    repo, editable = find_repo()
    if repo is None:
        raise RefusedError("This deck wasn't installed from a git checkout, so there's nothing to pull.")
    rc, head = await _git(repo, "rev-parse", "HEAD")
    rc2, upstream = await _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if rc or rc2:
        raise RefusedError("This checkout has no upstream to follow.")
    rc, full = await _git(repo, "rev-parse", "-q", "--verify", f"{target}^{{commit}}")
    if rc:
        raise RefusedError("That commit isn't here. Check for updates again.")
    if full == head:
        raise RefusedError("This deck is already on that version.")
    rc, _ = await _git(repo, "merge-base", "--is-ancestor", full, upstream)
    if rc:
        raise RefusedError(f"That commit isn't on {upstream}.")
    blockers = await _blockers(repo, head, upstream, full)
    if blockers:
        raise RefusedError(" ".join(blockers))
    # A fresh record: the last update's verdict goes with it, not what the switch has to remember.
    refused = read_state().get("refused")
    _state_path().unlink(missing_ok=True)
    _write_state(phase="rehearsing", to=full, **{"from": head}, by=by, user_id=user_id, agent=agent or {},
                 started_at=time.time(), steps=[], reported=False, refused=refused, told=full)
    _task = asyncio.create_task(_run(repo, editable, head, full, wait))
    return _view(read_state())


async def _run(repo: Path, editable: bool, head: str, target: str, wait: bool) -> None:
    stopped = False
    soon_until = time.monotonic() + min(_SOON_S, _countdown_s())
    try:
        needs = await _new_requirements(repo, head, target)
        if needs:
            _step(f"Installing what the new version needs: {', '.join(needs)}")
            rc, said = await _pip(*needs)
            if rc:
                raise RefusedError(f"Couldn't install {', '.join(needs)}: {said[-400:]}")
        _step("Starting the new version on a spare port")
        await rehearse(repo, target)
        _step("The new version starts")

        await asyncio.sleep(max(0.0, soon_until - time.monotonic()))
        await _count_down()
        if wait:
            await _wait_for_turns()
        _step(f"Stopping {len(_running_agents())} agent(s)", phase="stopping")
        stopped = True
        await asyncio.to_thread(stop_agents)

        _write_state(phase="pulling", **_restart_facts(repo, editable))
        _spawn_guard()
        _step(f"Moving the checkout to {target[:9]}")
        rc, said = await _git(repo, "merge", "--ff-only", "--quiet", target)
        if rc:
            raise RefusedError(f"git wouldn't move forward: {said[-400:]}")
        install = await _install_args(repo, editable, head, target)
        if install:
            _step("Installing the new version into the venv")
            rc, said = await _pip(*install)
            if rc:
                await _git(repo, "reset", "--keep", head)
                # From the start of a line: pip's last lines say what went wrong.
                raise RefusedError(f"Couldn't install the new version: {said[-600:].partition(chr(10))[2] or said}")
        _step("Restarting", phase="restarting")
        _request_relaunch()
    except RefusedError as exc:
        await _fail(str(exc), target, stopped)
    except Exception as exc:  # noqa: BLE001 — whatever it was, the deck stays up and says so
        log.warning("Deck update failed", error=str(exc), exc_info=True)
        await _fail(f"The update stopped: {exc}", target, stopped)


async def _install_args(repo: Path, editable: bool, head: str, target: str) -> tuple[str, ...]:
    """What ``pip install`` has to do after the pull. A copy in the venv needs the new code. An editable
    install already runs it, so only what pyproject.toml declares can be behind, like new commands.
    Packages went in before the rehearsal, hence ``--no-deps``."""
    if not editable:
        return "--no-deps", str(repo)
    _, changed = await _git(repo, "diff", "--name-only", head, target, "--", "pyproject.toml")
    return ("--no-deps", "-e", str(repo)) if changed else ()


async def _fail(why: str, target: str, stopped: bool) -> None:
    # The switch won't try this commit again; a person still can. The guard reads "failed" and leaves.
    _step(why, phase="failed", finished_at=time.time(), error=why, refused=target)
    if stopped and start_agents is not None:
        # They were only stopped for the update.
        await asyncio.to_thread(start_agents)
    await _report()


async def _count_down() -> None:
    seconds = _countdown_s()
    _step(f"Telling everyone it restarts in {int(seconds)} seconds", phase="countdown", restart_at=time.time() + seconds)
    await asyncio.sleep(seconds)


async def _wait_for_turns() -> None:
    deadline = time.monotonic() + _WAIT_FOR_TURNS_S
    busy = await busy_agents()
    if busy:
        _step(f"Waiting for {', '.join(busy)} to finish (up to {int(_WAIT_FOR_TURNS_S / 60)} minutes)", phase="waiting")
    while busy and time.monotonic() < deadline:
        await asyncio.sleep(5.0)
        busy = await busy_agents()
    if busy:
        _step(f"Still working, and stopped anyway: {', '.join(busy)}")


def _restart_facts(repo: Path, editable: bool) -> dict[str, Any]:
    host = os.environ.get("FD_BIND_HOST", "")
    host = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host  # noqa: S104
    return {
        "pid": os.getpid(), "repo": str(repo), "editable": editable, "cwd": os.getcwd(), "python": sys.executable,
        "argv": _relaunch_argv(), "env": _startup_env or dict(os.environ), "boot_timeout": _boot_timeout(),
        "url": f"http://{host}:{os.environ.get('FD_PORT', '25080')}", "registry": str(Path(_srv().DATA_DIR) / ".processes.json"),
    }


def _spawn_guard() -> None:
    """Started before the pull: what runs is this version's guard, already read from disk."""
    guard = Path(__file__).with_name("deck_update_guard.py")
    out = (_dir() / "guard.log").open("ab")
    try:
        subprocess.Popen([sys.executable, "-I", str(guard), str(_state_path())], cwd=_dir(), stdin=subprocess.DEVNULL,  # noqa: S603
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        out.close()


# ── The restart ─────────────────────────────────────────────────────


def make_server(config: Any) -> Any:
    """uvicorn's server, acting before asyncio.run winds down: that waits for worker threads,
    so one busy thread would hold the process up with no port open, and the guard with it."""
    import uvicorn

    class DeckServer(uvicorn.Server):
        async def startup(self, sockets=None):
            try:
                await super().startup(sockets)
                if self.started or not self.should_exit:
                    return
                code = STARTUP_FAILURE
            except SystemExit as exc:  # how newer versions of uvicorn say it
                code = exc.code if isinstance(exc.code, int) else STARTUP_FAILURE
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)

        async def shutdown(self, sockets=None):
            await super().shutdown(sockets)
            relaunch_if_asked()

    return DeckServer(config)


def can_restart(exit_server: Callable[[], None], env: dict[str, str]) -> None:
    """``main()``: how to end uvicorn, and what this process was started with."""
    global _exit_server, _startup_env

    _exit_server = exit_server
    _startup_env = dict(env)


def _relaunch_argv() -> list[str]:
    return [sys.executable, *sys.orig_argv[1:]]


def _request_relaunch() -> None:
    _relaunch_asked.set()
    # A shutdown that hangs (a stream, a stuck task) doesn't get to keep the deck down.
    threading.Thread(target=_relaunch_after_timeout, daemon=True).start()
    if _exit_server is not None:
        _exit_server()


def _relaunch_after_timeout() -> None:
    time.sleep(_SHUTDOWN_TIMEOUT_S)
    log.warning("Deck update: the shutdown is taking too long, restarting anyway")
    relaunch()


def relaunch_if_asked() -> None:
    """When uvicorn has shut down. Nothing happens unless an update asked for it."""
    if _relaunch_asked.is_set():
        relaunch()


def relaunch() -> None:
    env = _startup_env or dict(os.environ)
    sys.stdout.flush()
    sys.stderr.flush()
    # Same pid, same terminal: Ctrl-C and a supervisor's view of the process still work.
    os.execve(sys.executable, _relaunch_argv(), env)


# ── Coming up ───────────────────────────────────────────────────────


def _children_of_this_pid() -> set[int]:
    me = os.getpid()
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=6,  # noqa: S607
                             stdin=subprocess.DEVNULL, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    rows = (line.split() for line in out.splitlines())
    return {int(r[0]) for r in rows if len(r) == 2 and r[1] == str(me)}


def _reap(children: set[int]) -> None:
    """A Crew worker outlives a restart, and an exec keeps the pid, so it's still this process's child,
    one nothing here knows about. When it ended it would stay a zombie, and a zombie looks alive."""
    while children:
        for pid in list(children):
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                children.discard(pid)
                continue
            if done:
                children.discard(pid)
        time.sleep(2.0)


def on_boot() -> None:
    """Flight Deck's lifespan, first thing (nothing of this run has been started yet, so any
    child this pid has now comes from before the exec), and it says it's up at ``yield``."""
    state = read_state()
    phase = state.get("phase")
    if phase == "restarting":
        children = _children_of_this_pid()
        if children:
            threading.Thread(target=_reap, args=(children,), daemon=True).start()
    elif phase in ("rehearsing", "countdown", "waiting", "stopping", "pulling"):
        _write_state(phase="failed", finished_at=time.time(),
                     error="The deck was restarted while the update was under way, so it stopped there.")


def say_booted() -> None:
    if read_state().get("phase") == "restarting":
        _write_state(phase="booted")


# ── Telling people ──────────────────────────────────────────────────


def _result_text(state: dict[str, Any]) -> tuple[bool, str, str]:
    to, was = str(state.get("to") or "")[:9], str(state.get("from") or "")[:9]
    phase = state.get("phase")
    if phase == "done":
        return True, "Deck updated", f"This deck went from {was} to {to}, and its agents are coming back."
    if phase == "rolled_back":
        return False, "Deck update undone", (
            f"{state.get('error') or ''} The deck went back to {was} and is running again. "
            f"It now logs to {state.get('detached_log') or 'a file'}, not to the terminal it was started in.").strip()
    return False, "Deck update failed", str(state.get("error") or "The update stopped.")


async def _admins() -> list[str]:
    from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db

    if not _fd_auth_enabled():
        return ["local"]
    return [u["id"] for u in await get_db().list_users(limit=500) if u.get("role") == "admin"]


def _when(ts: float) -> str:
    """The date and time on this computer's clock, saying which clock."""
    return datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M %Z").strip()


def compose_mail(name: str, state: dict[str, Any], commits: list[dict[str, str]]) -> tuple[str, str, str]:
    """Subject, text and HTML of the email about one update."""
    was, to = str(state.get("from") or "")[:9], str(state.get("to") or "")[:9]
    phase = state.get("phase")
    when = _when(float(state.get("finished_at") or time.time()))
    took = int(float(state.get("finished_at") or 0) - float(state.get("started_at") or 0))
    if phase == "done":
        subject, lead = f"{name} updated to {to}: {when}", f"{name} went from {was} to {to}, and its agents are back or on their way."
    elif phase == "rolled_back":
        subject = f"{name} update undone, back on {was}: {when}"
        lead = (f"The update to {to} didn't come up, so {name} went back to {was} and is running again. It now logs to "
                f"{state.get('detached_log') or 'a file'}, not to the terminal it was started in, until it's restarted by hand.")
    else:
        subject, lead = f"{name} update failed: {when}", f"The update to {to} didn't go through."
    facts = [f"Started by: {state.get('by') or 'an admin'}", f"Finished: {when}, after {took} seconds"]
    why = str(state.get("error") or "").strip()
    titles = [f"{c['sha'][:9]} {c['title']}" for c in commits]
    text = "\n\n".join(filter(None, [lead, "\n".join(facts), f"What went wrong:\n{why}" if why else "",
                                      "Commits:\n" + "\n".join(titles) if titles else ""]))
    parts = [f"<p>{html.escape(lead)}</p>", "<p>" + "<br>".join(html.escape(f) for f in facts) + "</p>"]
    if why:
        parts.append(f"<p>What went wrong:</p><pre>{html.escape(why)}</pre>")
    if titles:
        parts.append("<p>Commits:</p><ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in titles) + "</ul>")
    return subject, text, "".join(parts)


async def _email_admins(state: dict[str, Any]) -> None:
    """To every admin, from whoever started it if they can send, else the first admin who can: a deck
    has no mailbox of its own. What became of it goes on the record, for the page to say."""
    from captain_claw.flight_deck.auth import _fd_auth_enabled, get_db
    from captain_claw.flight_deck.invite_routes import _branding, mail_options, send_from_admin

    if not _fd_auth_enabled():
        _write_state(mail={"sent": 0, "of": 0, "from": "", "error": "This deck has no accounts, so there's nobody to email."})
        return
    db = get_db()
    admins = [u for u in await db.list_users(limit=500) if u.get("role") == "admin" and u.get("email")]
    admins.sort(key=lambda u: u["id"] != state.get("user_id"))
    sender, via, sender_address = None, "", ""
    for admin in admins:
        options = await mail_options(db, admin["id"])
        via = next((v for v in ("google", "microsoft") if options[v]["available"]), "")
        if via:
            sender, sender_address = admin, options[via]["from"]
            break
    if sender is None:
        _write_state(mail={"sent": 0, "of": len(admins), "from": "", "error": (
            "No admin has a Gmail or Outlook connected that can send, so no email went out.")})
        return
    repo, _ = find_repo()
    commits = await _commits(repo, f"{state['from']}..{state['to']}") if repo and state.get("from") and state.get("to") else []
    name, _ = await _branding(db)
    subject, text, body = compose_mail(name, state, commits)
    sent, errors = 0, []
    for admin in admins:
        out = await send_from_admin(db, sender, via, admin["email"], subject, text, body)
        if out["sent"]:
            sent += 1
        else:
            errors.append(f"{admin['email']}: {out['error'][:200]}")
    if errors:
        log.warning("Deck update: an admin's email failed", errors=errors)
    _write_state(mail={"sent": sent, "of": len(admins), "from": sender_address, "error": "; ".join(errors)[:600]})


async def _bell(user_ids: list[str], title: str, body: str) -> None:
    from captain_claw.flight_deck import owner_notify

    for uid in user_ids:
        await owner_notify._to_bell(uid, "deck_update", title, body)


async def _report() -> None:
    """Once per update, to whoever started it, or every admin when the deck did it itself; and to
    the agent that asked, when one did."""
    state = read_state()
    if state.get("reported") or state.get("phase") not in ("done", "failed", "rolled_back"):
        return
    # What the guard needed to restart the deck has no reason to stay on disk.
    _write_state(reported=True, env={}, argv=[])
    ok, title, body = _result_text(state)
    try:
        await _email_admins(state)
    except Exception as exc:  # noqa: BLE001 — it says so on the page, and the rest still goes out
        log.warning("Deck update: couldn't email the admins", error=str(exc))
        _write_state(mail={"sent": 0, "of": 0, "from": "", "error": f"The email failed: {exc}"[:300]})
    try:
        await _bell([state["user_id"]] if state.get("user_id") else await _admins(), title, body)
        agent = state.get("agent") or {}
        if agent.get("slug"):
            await _tell_agent(agent, ok, title, body)
    except Exception as exc:  # noqa: BLE001 — the update is over either way
        log.warning("Deck update: couldn't tell anyone", error=str(exc))


async def _tell_agent(agent: dict, ok: bool, title: str, body: str) -> None:
    from captain_claw.flight_deck import agent_idle, agent_notify

    slug = str(agent["slug"])
    await agent_idle.wake(slug)
    # Agents come back one by one after a start.
    entry: dict = {}
    deadline = time.monotonic() + agent_idle.WAKE_TIMEOUT_S
    while time.monotonic() < deadline:
        entry = _srv()._load_process_registry().get(slug) or {}
        if entry.get("web_port") and await agent_idle._answers(int(entry["web_port"]), str(entry.get("web_auth") or "")):
            break
        await asyncio.sleep(1.0)
    await agent_notify.notify_source_agent(
        source_host="localhost", source_port=int(entry.get("web_port") or 0), origin=agent.get("origin") or {},
        kind="deck update", title=title, run_ref="crew action=update_deck", ok=ok, summary=body,
        verb="is done" if ok else "didn't go through",
    )


# ── The switch ──────────────────────────────────────────────────────


async def tick(db: Any) -> None:
    """One look: *notify* tells the admins about a new version once, *install* installs it when nobody's working."""
    mode = await get_mode(db)
    if mode == "off" or in_progress():
        return
    seen = await status(fetch=True)
    if not seen.get("available") or not seen.get("behind") or seen.get("fetch_error"):
        return
    target, state = seen["target"], read_state()
    if mode == "install" and not seen["blockers"] and target != state.get("refused") and not seen["agents_busy"]:
        # A turn can start during the countdown, and nobody chose to cut it short.
        await start(by="this deck, by itself", target=target, wait=True)
        return
    if state.get("told") != target:
        _write_state(told=target)
        n = seen["behind"]
        await _bell(await _admins(), "New version for this deck",
                    f"{n} new commit{'s' if n != 1 else ''} on {seen['upstream']}. Console → Updates installs it.")


async def loop(db: Any, stop: asyncio.Event) -> None:
    """Flight Deck's lifespan: the verdict on the update that just ended, then the switch."""
    # The guard needs a moment to write it, and a rollback takes a while.
    deadline = time.monotonic() + _boot_timeout() + 60
    while read_state().get("phase") in _ACTIVE and time.monotonic() < deadline:
        try:
            await asyncio.wait_for(stop.wait(), 1.0)
            return
        except TimeoutError:
            pass
    if read_state().get("phase") in _ACTIVE:
        # A power cut, or a guard that was killed: nobody is coming to say how it went.
        _write_state(phase="failed", finished_at=time.time(), error="The update's watcher went away before it gave a verdict.")
    await _report()
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), _poll_s())
            return
        except TimeoutError:
            pass
        try:
            await tick(db)
        except Exception as exc:  # noqa: BLE001 — the loop outlives any one bad look
            log.warning("Deck update check failed", error=str(exc))
