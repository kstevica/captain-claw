"""Custom tools, Phase 6: on Linux the launcher puts a tool in a Landlock sandbox
before it runs. These run wherever Landlock is (they skip elsewhere, macOS
included), and run standalone too, with only the standard library:

    python3 tests/test_flight_deck/test_xtool_sandbox.py

which is how they were checked in a Linux container from a Mac."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[2] / "captain_claw" / "flight_deck" / "xtool_launch.py"


def _launcher():
    spec = importlib.util.spec_from_file_location("xtool_launch", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ABI = _launcher().landlock_abi()

try:
    import pytest

    needs_landlock = pytest.mark.skipif(ABI < 1, reason="no Landlock on this host")
    needs_net = pytest.mark.skipif(ABI < 4, reason="Landlock here can't rule the network")
    needs_scope = pytest.mark.skipif(ABI < 6, reason="Landlock here can't scope signals")
except ImportError:  # run standalone
    def needs_landlock(f):
        return f
    needs_net = needs_scope = needs_landlock

PROBE = textwrap.dedent('''
    import json, os, socket, sys
    out = {}
    def tried(name, fn):
        try:
            fn()
            out[name] = "ok"
        except Exception as exc:
            out[name] = type(exc).__name__
    args = json.loads(sys.argv[1])
    tried("read_secret", lambda: open(args["secret"]).read())
    tried("list_outside", lambda: os.listdir(args["outside"]))
    tried("write_own_folder", lambda: open(os.path.join(args["version"], "planted.py"), "w").write("x"))
    tried("write_call_folder", lambda: open(os.path.join(args["call"], "out.txt"), "w").write("x"))
    tried("read_own_code", lambda: open(os.path.join(args["version"], "probe.py")).read())
    tried("own_proc", lambda: open("/proc/self/status").read())
    tried("others_proc", lambda: open("/proc/%d/environ" % args["parent"]).read())
    tried("signal_parent", lambda: os.kill(args["parent"], 0))
    def connect():
        s = socket.create_connection(("127.0.0.1", args["port"]), timeout=2)
        s.close()
    tried("connect", connect)
    tried("devnull", lambda: open("/dev/null", "w").write("x"))
    print(json.dumps(out))
''')


def _system_ro() -> list[str]:
    paths = ["/usr", "/lib", "/lib64", "/bin", "/etc", sys.base_prefix, os.path.dirname(os.path.realpath(sys.executable))]
    return list(dict.fromkeys(p for p in paths if p and os.path.exists(p)))


def _run(tmp: Path, *, no_net: bool, listener_port: int) -> dict:
    version, call, outside = tmp / "version", tmp / "call", tmp / "outside"
    for d in (version, call, outside):
        d.mkdir()
    (version / "probe.py").write_text(PROBE)
    secret = outside / "flight-deck.db"
    secret.write_text("secrets")
    args = {"secret": str(secret), "outside": str(outside), "version": str(version), "call": str(call),
            "parent": os.getpid(), "port": listener_port}
    cmd = [sys.executable, "-E", "-s", str(LAUNCHER), "--mem", "1024", "--cpu", "30"]
    for p in [str(version), *_system_ro(), "/proc/self"]:
        cmd += ["--ro", p]
    cmd += ["--rw", str(call)]
    if no_net:
        cmd.append("--no-net")
    cmd += ["--require-sandbox", "--", sys.executable, "-E", "-s", "-B", str(version / "probe.py"), json.dumps(args)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _listener() -> socket.socket:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    return srv


@needs_landlock
def test_a_tool_reaches_only_its_own_folders():
    srv = _listener()
    try:
        with tempfile.TemporaryDirectory() as d:
            out = _run(Path(d), no_net=True, listener_port=srv.getsockname()[1])
    finally:
        srv.close()
    assert out["read_secret"] == "PermissionError", out
    assert out["list_outside"] == "PermissionError", out
    assert out["write_own_folder"] == "PermissionError", out
    assert out["write_call_folder"] == "ok", out
    assert out["read_own_code"] == "ok", out
    assert out["own_proc"] == "ok", out
    assert out["others_proc"] == "PermissionError", out
    assert out["devnull"] == "ok", out


@needs_net
def test_no_network_unless_the_tool_says_so():
    srv = _listener()
    try:
        port = srv.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            closed = _run(Path(d), no_net=True, listener_port=port)
        with tempfile.TemporaryDirectory() as d:
            open_ = _run(Path(d), no_net=False, listener_port=port)
    finally:
        srv.close()
    assert closed["connect"] == "PermissionError", closed
    assert open_["connect"] == "ok", open_


@needs_scope
def test_a_tool_cant_signal_flight_deck():
    srv = _listener()
    try:
        with tempfile.TemporaryDirectory() as d:
            out = _run(Path(d), no_net=True, listener_port=srv.getsockname()[1])
    finally:
        srv.close()
    assert out["signal_parent"] == "PermissionError", out


LINE_PROBE = textwrap.dedent('''
    import json, os, socket
    sock = socket.socket(fileno=int(os.environ["XT_CAPTAIN_FD"]))
    sock.sendall(b'{"op": "tool", "name": "read"}\\n')
    print(sock.makefile("rb").readline().decode().strip())
''')


@needs_net
def test_the_line_to_flight_deck_works_in_the_sandbox_without_network():
    """Phase 7: a tool's line to the agent is an inherited socket pair, which
    Landlock leaves alone even with TCP shut off."""
    with tempfile.TemporaryDirectory() as d:
        version, call = Path(d) / "version", Path(d) / "call"
        version.mkdir()
        call.mkdir()
        (version / "line.py").write_text(LINE_PROBE)
        ours, theirs = socket.socketpair()
        cmd = [sys.executable, "-E", "-s", str(LAUNCHER), "--mem", "1024", "--cpu", "30"]
        for p in [str(version), *_system_ro(), "/proc/self"]:
            cmd += ["--ro", p]
        cmd += ["--rw", str(call), "--no-net", "--require-sandbox", "--", sys.executable, "-E", "-s", "-B",
                str(version / "line.py")]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(theirs.fileno(),),
                                env={"PATH": "/usr/bin:/bin", "XT_CAPTAIN_FD": str(theirs.fileno())})
        theirs.close()
        try:
            asked = ours.makefile("rb").readline()
            assert json.loads(asked) == {"op": "tool", "name": "read"}
            ours.sendall(b'{"ok": true, "content": "answered"}\n')
            out, err = proc.communicate(timeout=30)
        finally:
            ours.close()
        assert proc.returncode == 0, err
        assert json.loads(out) == {"ok": True, "content": "answered"}


def test_the_probe_says_what_this_host_offers():
    done = subprocess.run([sys.executable, "-E", "-s", str(LAUNCHER), "--probe"], capture_output=True, text=True, timeout=30)
    info = json.loads(done.stdout)
    assert info["landlock"] == ABI and info["files"] == (ABI >= 1) and info["network"] == (ABI >= 4)


def test_requiring_a_sandbox_where_there_is_none_runs_nothing():
    if ABI >= 1:
        return  # this host has one
    done = subprocess.run([sys.executable, "-E", "-s", str(LAUNCHER), "--require-sandbox", "--", sys.executable, "-c",
                           "print('ran')"], capture_output=True, text=True, timeout=30)
    assert done.returncode == 126 and "ran" not in done.stdout and "sandbox" in done.stderr


if __name__ == "__main__":
    print(f"Landlock ABI {ABI} on {sys.platform}")
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, exc)
    sys.exit(1 if failed else 0)
