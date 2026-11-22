"""Whether Docker can be reached, asked without leaving files open.

Docker isn't on every host Flight Deck runs on. docker-py reaches a missing
daemon by making an AF_UNIX socket and calling ``connect()``. When that fails
the socket lives on in the exception's frames until a garbage-collection pass,
and ``docker.from_env()`` makes a new client (and a new socket) on every call
while the daemon stays away. The deck asks on health polls and on agent-proxied
requests, so on a host without Docker it leaked about one open file a minute and
ran into its 1024-file limit within a day (cohort001, 2026-11-20: "Too many open
files", new connections refused).

So: "no Docker" is remembered for a minute, and the check before a client is made
uses a socket that is always closed. Lives in its own module because server.py
is loaded twice (``python -m`` and import) and the memory must be one.
"""

from __future__ import annotations

import os
import socket
import threading
import time

import docker

# How long "Docker can't be reached" is believed before looking again.
RECHECK_SECONDS = 60.0

_lock = threading.Lock()
_down_until = 0.0
_why = ""


class DockerUnavailable(docker.errors.DockerException):
    """Docker can't be reached from this deck (looked at no more than once a minute)."""


def unix_socket() -> str | None:
    """The Unix socket docker-py would talk to, or None when it would use
    something else (tcp://, ssh://, a Windows named pipe) that isn't checked here."""
    if os.name == "nt":
        return None
    host = os.environ.get("DOCKER_HOST", "").strip()
    if not host:
        return "/var/run/docker.sock"  # docker-py's own default
    if host.startswith("unix://"):
        return host[len("unix://"):] or None
    return None


def _answers(path: str) -> bool:
    """Whether something listens on the socket at *path*. Never leaves a file open."""
    if not os.path.exists(path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(path)
    except OSError:
        return False
    return True


def mark_down(why: str = "") -> None:
    """Believe Docker unreachable for the next minute (a client couldn't be made)."""
    global _down_until, _why
    with _lock:
        _down_until = time.monotonic() + RECHECK_SECONDS
        _why = why or "Docker isn't reachable from this deck"


def check() -> None:
    """Raise :class:`DockerUnavailable` when Docker can't be reached, without
    trying it again within a minute of finding it gone. Returns when it may be
    there: a Unix socket that answers, or a DOCKER_HOST this can't check."""
    with _lock:
        if time.monotonic() < _down_until:
            raise DockerUnavailable(_why)
    path = unix_socket()
    if path is not None and not _answers(path):
        mark_down(f"Docker isn't reachable at {path}")
        raise DockerUnavailable(_why)


def reset() -> None:
    """Forget what was found (tests)."""
    global _down_until, _why
    with _lock:
        _down_until = 0.0
        _why = ""
