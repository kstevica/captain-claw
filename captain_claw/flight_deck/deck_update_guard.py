"""Stays outside Flight Deck while it restarts on a new version, and takes it back if it doesn't come up.

Run as a script, standard library only, and started *before* the pull: this is the
old version's copy, already in memory, so a bad new version can't break it. The
deck tells it what's going on through the update's state file (``deck_update.py``)
and it writes the verdict there, for the deck that comes up to report.

The new version counts as up once it has written ``booted`` and answers on its port.
If its process ends first, or that takes longer than ``boot_timeout``: the agents it
may have started are stopped, the checkout goes back to the commit it was on, and
the deck is started again. That one is detached and logs to ``<update>/deck.log``,
because the terminal it ran in went with the process that ended.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# The deck wrote one of these, so there's nothing left to watch.
_OVER = ("idle", "done", "failed", "rolled_back")
_PULL_TIMEOUT_S = 900.0


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, **changes) -> dict:
    state = {**_read(path), **changes}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)
    return state


def _say(line: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    # Ended, and whoever started it hasn't noticed yet: a zombie still "exists".
    try:
        seen = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5,  # noqa: S603, S607
                              stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return True
    return not seen.stdout.strip().startswith("Z")


def _answers(port_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{port_url}/fd/health", timeout=3) as resp:  # noqa: S310
            return resp.status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except OSError:
        return False


def _end(pid: int, wait_s: float = 10.0) -> None:
    """SIGTERM, then SIGKILL."""
    if not _alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _end_agents(registry: Path) -> None:
    """Whatever the failed version started would run on its code under the old deck, which keeps
    an agent it finds alive. The registry stays as it is: a dead pid in it means "start this again"."""
    try:
        entries = json.loads(registry.read_text())
    except (OSError, ValueError):
        return
    for entry in entries.values():
        if isinstance(entry, dict) and entry.get("pid"):
            _end(int(entry["pid"]), wait_s=5.0)


def _git(repo: str, *args: str) -> tuple[int, str]:
    done = subprocess.run(["git", "-C", repo, "-c", "core.hooksPath=/dev/null", *args], capture_output=True,  # noqa: S603, S607
                          text=True, timeout=300, stdin=subprocess.DEVNULL, check=False)
    return done.returncode, (done.stdout + done.stderr).strip()


def _go_back(path: Path, state: dict, why: str) -> None:
    _say(f"Going back to {state['from'][:9]}: {why}")
    state = _write(path, phase="rolling_back", error=why)
    _end(int(state["pid"]))
    _end_agents(Path(state["registry"]))

    rc, out = _git(state["repo"], "reset", "--keep", state["from"])
    if rc:
        _say(f"git reset failed: {out}")
        _write(path, phase="failed", finished_at=time.time(), error=(
            f"{why} Going back failed too, so the deck is down: {out[-300:]} "
            f"The checkout is {state['repo']} and it was on {state['from']}."))
        return
    if not state.get("editable"):
        # Installed into the venv as a copy, so the checkout going back isn't enough.
        subprocess.run([state["python"], "-m", "pip", "install", "--no-deps", state["repo"]], check=False,  # noqa: S603
                       stdin=subprocess.DEVNULL)

    log_path = path.parent / "deck.log"
    with log_path.open("ab") as log:
        proc = subprocess.Popen(state["argv"], cwd=state["cwd"], env=state["env"], stdin=subprocess.DEVNULL,  # noqa: S603
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    _say(f"Started the previous version as {proc.pid}, logging to {log_path}")
    deadline = time.monotonic() + float(state["boot_timeout"])
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        if _answers(state["url"]):
            _write(path, phase="rolled_back", pid=proc.pid, finished_at=time.time(), refused=state["to"],
                   detached_log=str(log_path))
            _say("Back on the previous version")
            return
        time.sleep(1.0)
    _write(path, phase="failed", finished_at=time.time(), refused=state["to"], error=(
        f"{why} The previous version didn't come back either, so the deck is down. See {log_path}."))
    _say("The previous version didn't come back")


def watch(path: Path) -> int:
    pulling_until = time.monotonic() + _PULL_TIMEOUT_S
    boot_until = 0.0
    while True:
        state = _read(path)
        phase = state.get("phase")
        if phase in _OVER:
            return 0
        if phase == "booted" and _answers(state["url"]):
            _write(path, phase="done", finished_at=time.time())
            _say(f"The new version is up: {state['to'][:9]}")
            return 0
        now = time.monotonic()
        if phase in ("restarting", "booted") and not boot_until:
            boot_until = now + float(state["boot_timeout"])
        if not _alive(int(state["pid"])):
            _go_back(path, state, "The new version's process ended before it was up.")
            return 1
        if boot_until and now > boot_until:
            _go_back(path, state, f"The new version wasn't up after {int(state['boot_timeout'])} seconds.")
            return 1
        if not boot_until and now > pulling_until:
            _go_back(path, state, "The update stopped before it reached the restart.")
            return 1
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(watch(Path(sys.argv[1])))
