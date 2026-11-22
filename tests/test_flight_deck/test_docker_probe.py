"""Docker probes on a host without Docker leave no file open, and the deck's
open-file limit is lifted at start.

cohort001 (2026-11-20): Docker isn't installed there, and every probe
(``/fd/health``, agent-proxied requests) made a new docker client whose failed
``connect()`` left an AF_UNIX socket open. The deck hit its 1024-file limit
within a day and refused connections.
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import docker
import pytest

from captain_claw.flight_deck import docker_probe, proc_compat
from captain_claw.flight_deck import server

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Unix sockets and rlimits")


def _open_files() -> int:
    for d in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(d))
        except OSError:
            continue
    pytest.skip("can't count open files here")


@pytest.fixture
def short_dir():
    # macOS caps a Unix socket path near 104 bytes; pytest's tmp_path is longer.
    d = Path(tempfile.mkdtemp(prefix="dp", dir="/tmp"))
    yield d
    for p in d.iterdir():
        p.unlink()
    d.rmdir()


@pytest.fixture
def fresh(monkeypatch):
    docker_probe.reset()
    monkeypatch.setattr(server, "_client", None)
    yield
    docker_probe.reset()


def _from_env_must_not_run():
    raise AssertionError("docker.from_env() ran while Docker was known to be gone")


class TestNoDocker:
    def test_missing_socket_raises_a_docker_exception_and_opens_nothing(self, fresh, short_dir, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", f"unix://{short_dir}/docker.sock")
        monkeypatch.setattr(server.docker, "from_env", _from_env_must_not_run)
        before = _open_files()
        for _ in range(25):
            with pytest.raises(docker.errors.DockerException):
                server.get_docker()
        assert _open_files() == before

    def test_a_dead_socket_is_closed_after_the_check(self, fresh, short_dir, monkeypatch):
        """The socket file is there but nobody listens: connect() is refused."""
        path = short_dir / "docker.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(path))
        srv.close()  # the file stays, nothing answers
        monkeypatch.setenv("DOCKER_HOST", f"unix://{path}")
        before = _open_files()
        assert docker_probe._answers(str(path)) is False
        assert _open_files() == before

    def test_gone_is_remembered_for_a_minute(self, fresh, short_dir, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", f"unix://{short_dir}/docker.sock")
        looked = []
        real = docker_probe._answers
        monkeypatch.setattr(docker_probe, "_answers", lambda p: looked.append(p) or real(p))
        for _ in range(5):
            with pytest.raises(docker_probe.DockerUnavailable):
                docker_probe.check()
        assert len(looked) == 1

        # A minute later it looks again.
        clock = [docker_probe.time.monotonic() + docker_probe.RECHECK_SECONDS + 1]
        monkeypatch.setattr(docker_probe.time, "monotonic", lambda: clock[0])
        with pytest.raises(docker_probe.DockerUnavailable):
            docker_probe.check()
        assert len(looked) == 2

    def test_health_still_answers(self, fresh, short_dir, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", f"unix://{short_dir}/docker.sock")
        monkeypatch.setattr(server.docker, "from_env", _from_env_must_not_run)
        assert server.health() == {"ok": True, "docker": False, "processes": True}

    def test_a_failed_client_is_not_retried_within_the_minute(self, fresh, monkeypatch):
        """A DOCKER_HOST the check can't look at (tcp://): the first failure is remembered."""
        monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:1")
        calls = []

        def failing():
            calls.append(1)
            raise docker.errors.DockerException("Error while fetching server API version")

        monkeypatch.setattr(server.docker, "from_env", failing)
        for _ in range(4):
            with pytest.raises(docker.errors.DockerException):
                server.get_docker()
        assert len(calls) == 1


class TestDocker:
    def test_a_listening_socket_gets_a_client(self, fresh, short_dir, monkeypatch):
        path = short_dir / "docker.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(path))
        srv.listen(4)
        try:
            monkeypatch.setenv("DOCKER_HOST", f"unix://{path}")
            client = object()
            monkeypatch.setattr(server.docker, "from_env", lambda: client)
            assert server.get_docker() is client
            assert server.get_docker() is client  # kept
        finally:
            srv.close()

    def test_default_socket_is_docker_pys(self, monkeypatch):
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        assert docker_probe.unix_socket() == "/var/run/docker.sock"
        monkeypatch.setenv("DOCKER_HOST", "ssh://me@box")
        assert docker_probe.unix_socket() is None


class TestOpenFileLimit:
    def _fake(self, monkeypatch, soft, hard, refuse_above=None):
        import resource

        state = {"soft": soft, "hard": hard}

        def getrlimit(which):
            return state["soft"], state["hard"]

        def setrlimit(which, limits):
            new_soft, new_hard = limits
            if refuse_above is not None and new_soft > refuse_above:
                raise ValueError("not allowed")
            state["soft"], state["hard"] = new_soft, new_hard

        monkeypatch.setattr(resource, "getrlimit", getrlimit)
        monkeypatch.setattr(resource, "setrlimit", setrlimit)
        return state

    def test_linux_default_is_lifted(self, monkeypatch):
        state = self._fake(monkeypatch, 1024, 524288)
        assert proc_compat.raise_open_file_limit() == (1024, 65536)
        assert state == {"soft": 65536, "hard": 524288}

    def test_never_past_the_hard_limit(self, monkeypatch):
        state = self._fake(monkeypatch, 1024, 4096)
        assert proc_compat.raise_open_file_limit() == (1024, 4096)
        assert state["soft"] == 4096

    def test_already_high_is_left_alone(self, monkeypatch):
        state = self._fake(monkeypatch, 100000, 524288)
        assert proc_compat.raise_open_file_limit() == (100000, 100000)
        assert state["soft"] == 100000

    def test_macos_cap_falls_back(self, monkeypatch):
        import resource

        state = self._fake(monkeypatch, 256, resource.RLIM_INFINITY, refuse_above=10240)
        assert proc_compat.raise_open_file_limit() == (256, 10240)
        assert state["soft"] == 10240
