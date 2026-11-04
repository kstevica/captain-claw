"""Flight Deck's process management on Windows (proc_compat), checked on Unix.

On Windows ``import fcntl`` fails, so the first registry save of a new agent
ended ``POST /fd/spawn-process`` in a bare 500; ``os.kill(pid, 0)`` there ends
the process it asks about; and Python reads the home folder from USERPROFILE,
never HOME. Windows is simulated here: no fcntl, ``IS_WINDOWS`` on, the kernel32
query and taskkill stubbed. No real agent is started.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from captain_claw.flight_deck import proc_compat, server


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch):
    """Windows as Flight Deck sees it: no fcntl, no signal-0 probe, taskkill."""
    alive: set[int] = set()
    taskkills: list[list[str]] = []

    def _no_signals(pid, sig):  # os.kill(pid, 0) would END the process there
        raise AssertionError(f"os.kill({pid}, {sig}) on Windows")

    def _run(args, **kw):
        taskkills.append(list(args))
        alive.discard(int(args[2]))
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(proc_compat, "_fcntl", None)
    monkeypatch.setattr(proc_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(proc_compat, "_win_pid_alive", lambda pid: pid in alive)
    monkeypatch.setattr(proc_compat.os, "kill", _no_signals)
    monkeypatch.setattr(proc_compat.subprocess, "run", _run)
    return alive, taskkills


# ── proc_compat ──────────────────────────────────────────────────────


def test_no_fcntl_module_is_not_an_error():
    """`import fcntl` failing, as on Windows, leaves the locks as no-ops."""
    code = ("import sys; sys.modules['fcntl'] = None\n"
            "from captain_claw.flight_deck import proc_compat as p\n"
            "assert p._fcntl is None\n"
            "import tempfile\n"
            "with tempfile.TemporaryFile('w') as f: p.lock_shared(f); p.lock_exclusive(f)\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_pid_alive_on_unix():
    assert proc_compat.pid_alive(os.getpid())
    for pid in (None, 0, -1):
        assert not proc_compat.pid_alive(pid)  # never a process group
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not proc_compat.pid_alive(child.pid)


def test_windows_pid_check_sends_no_signal(windows):
    alive, _ = windows
    alive.add(4242)
    assert proc_compat.pid_alive(4242)
    assert not proc_compat.pid_alive(4243)
    assert not proc_compat.reaped(4242)  # no zombies, no waitpid


def test_windows_stop_ends_the_whole_tree(windows):
    alive, taskkills = windows
    alive.add(4242)
    proc_compat.terminate(4242)
    proc_compat.kill(4242)
    assert taskkills[0] == ["taskkill", "/pid", "4242", "/t", "/f"]
    assert not proc_compat.pid_alive(4242)


def test_kill_pid_ends_a_real_child_quickly():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    started = time.monotonic()
    server._kill_pid(child.pid, timeout=5.0)
    assert time.monotonic() - started < 3  # reaped, not waited out as a zombie
    assert child.wait(timeout=5) is not None


def test_kill_pid_on_windows(windows):
    alive, taskkills = windows
    alive.add(5151)
    server._kill_pid(5151, timeout=1.0)
    assert taskkills == [["taskkill", "/pid", "5151", "/t", "/f"]]
    server._kill_pid(5151, timeout=1.0)  # already gone: nothing sent
    assert len(taskkills) == 1


# ── the registry and the agent's environment ─────────────────────────


def test_registry_round_trip_without_fcntl(tmp_path, monkeypatch, windows):
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", tmp_path / ".processes.json")
    server._save_process_registry({"deep-researcher": {"name": "Deep Researcher — ü", "pid": None}})
    assert server._load_process_registry()["deep-researcher"]["name"] == "Deep Researcher — ü"
    assert server._load_process_registry_strict()["deep-researcher"]["pid"] is None


def test_process_is_alive_on_windows(tmp_path, monkeypatch, windows):
    alive, _ = windows
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", tmp_path / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    server._save_process_registry({"a": {"pid": 7001}, "b": {"pid": 7002}, "c": {"pid": None}})
    alive.add(7001)
    assert server._process_is_alive("a")
    assert not server._process_is_alive("b")
    assert not server._process_is_alive("c")
    assert not server._process_is_alive("missing")


def test_windows_agent_home_and_utf8(monkeypatch):
    env = {"HOME": "/agents/x/data/home-config-parent"}
    server._pin_windows_home(env)
    assert env == {"HOME": "/agents/x/data/home-config-parent"}  # Unix: untouched

    monkeypatch.setattr(proc_compat, "IS_WINDOWS", True)
    env = {"HOME": r"C:\fd-data\x\data\home-config-parent", "USERPROFILE": r"C:\Users\me"}
    server._pin_windows_home(env)
    assert env["USERPROFILE"] == env["HOME"]  # Path.home() there reads USERPROFILE
    assert env["PYTHONUTF8"] == "1"
    env = {"HOME": "h", "PYTHONUTF8": "0"}
    server._pin_windows_home(env)
    assert env["PYTHONUTF8"] == "0"  # an operator's own choice stays
