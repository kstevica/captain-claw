"""Crew — a person's shell in their own project, from the browser.

Someone without SSH to this computer still has to be able to set a project up by
hand, and to look at what a worker left. It's the same reach their approved workers
already have (a non-admin's still needs its own tick on the Users page), with the same
home, and none of the logins saved for those workers.

A shell outlives its socket: a dropped connection in the middle of an install comes
back to it, with what it printed meanwhile. One with nobody attached goes after a while.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from captain_claw.logging import get_logger

log = get_logger(__name__)

_MAX_PER_USER = 4
_IDLE_S = 1800.0
_WATCH_S = 30.0
_POLL_BUSY_S, _POLL_QUIET_S = 0.02, 0.08
_WRITE_CHUNK = 4096
_MAX_INPUT = 1_000_000
_REAP_TRIES = 25
_SHELLS = ("/bin/bash", "/usr/bin/bash")


@dataclass
class _Shell:
    session: Any  # a terminal.daemon.PtySession
    uid: str
    target: str
    clients: int = 0
    last_seen: float = field(default_factory=time.time)
    watcher: asyncio.Task | None = None


_OPEN: dict[tuple[str, str], _Shell] = {}


def _rc_text(path_value: str, banner: str) -> str:
    """Their own shell setup first, since it may rewrite PATH, then what the project needs."""
    return "\n".join((
        '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"',
        f"export PATH={shlex.quote(path_value)}",
        '[ -f "$CREW_ACTIVATE" ] && . "$CREW_ACTIVATE"',
        r"PS1='\[\e[35m\]crew\[\e[0m\] \W \$ '",
        f"printf '%s\\n\\n' {shlex.quote(banner)}",
        "",
    ))


def _command(rc_file: Path, env: dict[str, str]) -> list[str]:
    for shell in _SHELLS:
        if Path(shell).exists():
            return [shell, "--rcfile", str(rc_file), "-i"]
    return [env.get("SHELL") or "/bin/sh", "-i"]


def _open_shell(uid: str, target: str, cwd: str, env: dict[str, str], rc_file: Path, banner: str, cols: int, rows: int) -> _Shell:
    from captain_claw.terminal.daemon import PtySession

    env = {**env, "BASH_SILENCE_DEPRECATION_WARNING": "1"}  # macOS's bash would otherwise ask them to change shell
    rc_file.write_text(_rc_text(env.get("PATH", ""), banner), encoding="utf-8")
    os.chmod(rc_file, 0o600)
    session = PtySession(_command(rc_file, env), cwd, env, cols, rows, mirror=False)
    session.spawn(env)
    session.attach(asyncio.get_running_loop())
    shell = _Shell(session, uid, target)
    shell.watcher = asyncio.create_task(_watch(shell))
    _OPEN[(uid, target)] = shell
    log.info("Crew: a shell was opened", user=uid, target=target, cwd=cwd)
    return shell


def _forget(shell: _Shell) -> None:
    if _OPEN.get((shell.uid, shell.target)) is shell:
        _OPEN.pop((shell.uid, shell.target), None)


async def _collect(pid: int | None) -> None:
    """No zombie behind it, and nothing it started in its group."""
    if not pid:
        return
    for attempt in range(_REAP_TRIES):
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if done:
            return
        if attempt == _REAP_TRIES // 2:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        await asyncio.sleep(0.2)


def _close(shell: _Shell) -> None:
    _forget(shell)
    session = shell.session
    pid = session.pid
    if pid and session.alive:
        try:
            os.killpg(pid, signal.SIGHUP)
        except (ProcessLookupError, PermissionError):
            pass
    session.kill()
    if shell.watcher is not None and shell.watcher is not asyncio.current_task():
        shell.watcher.cancel()
    try:
        job = asyncio.create_task(_collect(pid))
    except RuntimeError:
        return
    _KEEP.add(job)
    job.add_done_callback(_KEEP.discard)


_KEEP: set[asyncio.Task] = set()


async def _watch(shell: _Shell) -> None:
    """Goes when it has ended with nobody there to see it, or sat a long time unattended."""
    while True:
        await asyncio.sleep(_WATCH_S)
        if shell.clients:
            shell.last_seen = time.time()
            continue
        if not shell.session.alive or time.time() - shell.last_seen > _IDLE_S:
            _close(shell)
            return


def close_for(uid: str) -> None:
    """Their approval for this went: so do their shells."""
    for shell in [s for s in _OPEN.values() if s.uid == uid]:
        _close(shell)


def close_all() -> None:
    for shell in list(_OPEN.values()):
        _close(shell)


def open_count(uid: str) -> int:
    return sum(1 for s in _OPEN.values() if s.uid == uid)


def existing(uid: str, target: str) -> _Shell | None:
    shell = _OPEN.get((uid, target))
    if shell is None:
        return None
    if not shell.session.alive and not shell.clients:
        _close(shell)
        return None
    return shell


def at_limit(uid: str) -> bool:
    return open_count(uid) >= _MAX_PER_USER


def start(uid: str, target: str, cwd: str, env: dict[str, str], rc_file: Path, banner: str, cols: int, rows: int) -> _Shell:
    return _open_shell(uid, target, cwd, env, rc_file, banner, max(20, min(cols, 400)), max(5, min(rows, 200)))


async def _write(session: Any, data: bytes) -> None:
    """A paste can be bigger than the terminal's buffer takes in one go."""
    while data:
        try:
            session.write(data[:_WRITE_CHUNK])
        except BlockingIOError:
            await asyncio.sleep(0.01)
            continue
        except (OSError, RuntimeError):
            return
        data = data[_WRITE_CHUNK:]


async def _from_browser(ws: WebSocket, shell: _Shell) -> None:
    while True:
        message = await ws.receive()
        if message.get("type") == "websocket.disconnect":
            return
        try:
            event = json.loads(message.get("text") or "{}")
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "input":
            await _write(shell.session, str(event.get("data") or "").encode("utf-8", "surrogateescape")[:_MAX_INPUT])
        elif kind == "resize":
            try:
                shell.session.resize(max(20, min(int(event["cols"]), 400)), max(5, min(int(event["rows"]), 200)))
            except (KeyError, TypeError, ValueError):
                continue
        elif kind == "close":
            _close(shell)
            return


async def serve(ws: WebSocket, shell: _Shell) -> None:
    """One browser on one shell, until either goes. What it printed comes first."""
    session = shell.session
    shell.clients += 1
    shell.last_seen = time.time()
    cursor = 0
    inbound = asyncio.create_task(_from_browser(ws, shell))
    try:
        while True:
            data, cursor = session.since(cursor)
            if data:
                await ws.send_bytes(data)
                await asyncio.sleep(_POLL_BUSY_S)
                continue
            if not session.alive:
                await ws.send_json({"type": "exit", "code": session.exit_code})
                break
            if inbound.done():
                break
            await asyncio.sleep(_POLL_QUIET_S)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        inbound.cancel()
        shell.clients = max(0, shell.clients - 1)
        shell.last_seen = time.time()
        if not session.alive:
            _close(shell)
