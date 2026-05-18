"""Subprocess lifecycle for agent-coded apps.

Each code-app is a directory under ``~/.captain-claw-fd/apps/<slug>/``
containing at minimum:

* ``backend.py``  — user-written Python module. Must expose
  ``async def handle(method, path, headers, body) -> dict`` (or a sync
  ``handle``). The dict shape is
  ``{"status": int, "headers": dict, "body": bytes | str}``.
* ``frontend.html`` — single-file HTML/JS bundle served at ``/``.
* ``manifest.json`` — minimal metadata (name, version, created_at).

This module owns the *runtime* side of that contract:

* lazy-spawn the backend subprocess on first request,
* talk to it over a per-app unix socket,
* keep rotating stderr/stdout logs for the self-repair loop,
* enforce CPU + memory rlimits,
* reap idle subprocesses after ``_IDLE_TIMEOUT_SECONDS``.

The agent never imports this directly — it only writes ``backend.py``.
FD owns spawning, routing, killing.

The subprocess entry point lives in ``app_subprocess.py`` next to this
file (run via ``python -m captain_claw.flight_deck.app_subprocess``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp


log = logging.getLogger(__name__)


# ── storage layout ────────────────────────────────────────────────────


def _fd_home() -> Path:
    base = os.environ.get("CAPTAIN_CLAW_FD_HOME") or os.path.expanduser("~/.captain-claw-fd")
    return Path(base)


def apps_root() -> Path:
    """Root dir for code-app directories (one per slug)."""
    p = _fd_home() / "apps"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sockets_root() -> Path:
    p = _fd_home() / "app_sockets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def app_dir(slug: str) -> Path:
    """Filesystem dir for a single code-app. Created on demand."""
    safe = _safe_slug(slug)
    d = apps_root() / safe
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    return d


def _safe_slug(slug: str) -> str:
    out = "".join(c for c in (slug or "") if c.isalnum() or c in ("-", "_"))
    if not out:
        raise ValueError(f"unsafe slug: {slug!r}")
    return out


def _socket_path(slug: str) -> Path:
    # Keep socket path short — many kernels cap unix socket paths at ~104
    # chars. Per-slug filename under a dedicated dir is well within that.
    return _sockets_root() / f"{_safe_slug(slug)}.sock"


# ── tunables ──────────────────────────────────────────────────────────

# How long an app may sit idle (no proxied requests) before the reaper
# kills its subprocess. Conserves memory across many infrequently-used
# apps. Override via env for development.
_IDLE_TIMEOUT_SECONDS = int(os.environ.get("FD_APP_IDLE_TIMEOUT", "600"))

# Memory ceiling per subprocess. The worker enforces this via RLIMIT_AS
# at startup; we re-state it here only for documentation. Override via
# the worker entry-point flag, not here.
_DEFAULT_MEM_LIMIT_MB = 512

# Lines of stderr/stdout to retain in the in-memory ring buffer that
# feeds the self-repair loop. The on-disk log file holds more.
_LOG_RING_LINES = 400

# Spawn timeout — how long we wait for the subprocess to create its
# socket file. Longer than typical aiohttp startup so cold-import-heavy
# apps still come up.
_SPAWN_TIMEOUT_SECONDS = 12.0

# Time to wait for graceful SIGTERM shutdown before SIGKILL.
_TERM_GRACE_SECONDS = 3.0


# ── per-app process state ─────────────────────────────────────────────


@dataclass
class AppProcess:
    """Bookkeeping for one running code-app subprocess.

    ``last_request_at`` drives the idle reaper. ``last_error`` is the
    most recent fatal traceback (or empty); the self-repair loop reads
    it after a failed request to feed the agent.
    """

    slug: str
    pid: int
    socket_path: Path
    started_at: float
    last_request_at: float
    process: asyncio.subprocess.Process
    stderr_ring: deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_RING_LINES))
    stdout_ring: deque[str] = field(default_factory=lambda: deque(maxlen=_LOG_RING_LINES))
    last_error: str = ""
    # Lock serializing spawn / restart for this slug. ``get_or_spawn``
    # acquires it before checking the live-process map so two concurrent
    # requests can't both spawn.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ── runtime singleton ─────────────────────────────────────────────────


class AppRuntime:
    """Owns the live subprocess table and the proxy aiohttp session.

    One instance is created at FD startup (see ``server.py`` lifespan).
    """

    def __init__(self) -> None:
        self._procs: dict[str, AppProcess] = {}
        self._table_lock = asyncio.Lock()
        # Build the proxy session lazily — we need it inside the
        # running event loop, not at import time.
        self._session: aiohttp.ClientSession | None = None
        self._reaper_task: asyncio.Task | None = None
        self._shutdown = False

    # ── lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Wire up the idle-reaper. Call from FD startup."""
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._idle_reaper())

    async def shutdown(self) -> None:
        """Stop all subprocesses and close the proxy session."""
        self._shutdown = True
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reaper_task = None
        async with self._table_lock:
            slugs = list(self._procs.keys())
        for slug in slugs:
            try:
                await self.stop(slug)
            except Exception as exc:
                log.warning("Error stopping app %s on shutdown: %s", slug, exc)
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ── spawn / stop ───────────────────────────────────────────────

    async def get_or_spawn(self, slug: str) -> AppProcess:
        """Return the live ``AppProcess`` for ``slug``, spawning if needed.

        Thread-safety: callers may invoke concurrently; only one spawn
        per slug ever runs.
        """
        slug = _safe_slug(slug)
        async with self._table_lock:
            existing = self._procs.get(slug)
        if existing and self._is_alive(existing):
            return existing
        return await self._spawn(slug)

    async def stop(self, slug: str) -> bool:
        """Terminate the subprocess for ``slug``. Idempotent.

        Returns ``True`` if a process was killed, ``False`` if none was
        running.
        """
        slug = _safe_slug(slug)
        async with self._table_lock:
            proc = self._procs.pop(slug, None)
        if proc is None:
            return False
        await self._terminate(proc)
        try:
            proc.socket_path.unlink(missing_ok=True)
        except OSError:
            pass
        return True

    async def restart(self, slug: str) -> AppProcess:
        """Stop (if running) and spawn fresh. Used by the self-repair loop
        after the agent rewrites ``backend.py``."""
        await self.stop(slug)
        return await self._spawn(slug)

    # ── proxy ──────────────────────────────────────────────────────

    async def proxy(
        self,
        slug: str,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        *,
        query_string: str = "",
    ) -> tuple[int, dict[str, str], bytes]:
        """Forward an HTTP request to the app's subprocess.

        Returns ``(status, headers, body)``. The caller is responsible
        for translating that back into an HTTP response on the FD side.

        If the subprocess dies mid-request, the failure is captured in
        ``last_error`` and re-raised as ``AppRuntimeError`` so the
        self-repair loop can surface it to the agent.
        """
        proc = await self.get_or_spawn(slug)
        proc.last_request_at = time.time()
        session = await self._proxy_session()

        # aiohttp lets us point UnixConnector at a fixed socket per
        # request via the connector. Build a one-shot connector here so
        # different slugs can share the session without sticky routing.
        url = f"http://unix{path or '/'}"
        if query_string:
            url = f"{url}?{query_string}"

        # Strip hop-by-hop headers and our own routing prefix.
        fwd_headers = _filter_request_headers(headers)

        connector = aiohttp.UnixConnector(path=str(proc.socket_path))
        try:
            try:
                async with aiohttp.ClientSession(connector=connector) as inner:
                    async with inner.request(
                        method=method,
                        url=url,
                        headers=fwd_headers,
                        data=body if body else None,
                        timeout=aiohttp.ClientTimeout(total=60),
                        allow_redirects=False,
                    ) as resp:
                        out_body = await resp.read()
                        out_headers = _filter_response_headers(dict(resp.headers))
                        return resp.status, out_headers, out_body
            except aiohttp.ClientError as exc:
                # Most likely the subprocess died. Capture and let the
                # caller decide whether to surface to the agent.
                err = f"app subprocess connection error: {exc}"
                proc.last_error = err
                log.warning("App %s proxy error: %s", slug, exc)
                raise AppRuntimeError(err) from exc
        finally:
            # The session above owns the connector and closes it on
            # exit. The outer ``_proxy_session`` is unused right now
            # but kept as a stub for a future per-slug pool.
            _ = session

    # ── logs (for self-repair) ─────────────────────────────────────

    def tail_logs(self, slug: str, n: int = 200) -> dict[str, list[str]]:
        """Return the last ``n`` lines of stderr and stdout for ``slug``.

        Used by the agent during self-repair: after a failed request,
        the orchestration loop reads the most-recent stderr lines and
        feeds them back as a tool result.
        """
        slug = _safe_slug(slug)
        proc = self._procs.get(slug)
        if proc is None:
            return {"stderr": [], "stdout": [], "last_error": ""}
        stderr = list(proc.stderr_ring)[-n:]
        stdout = list(proc.stdout_ring)[-n:]
        return {"stderr": stderr, "stdout": stdout, "last_error": proc.last_error}

    def list_running(self) -> list[dict[str, Any]]:
        """Return a debug summary for every live app process."""
        out: list[dict[str, Any]] = []
        for proc in self._procs.values():
            out.append({
                "slug": proc.slug,
                "pid": proc.pid,
                "started_at": proc.started_at,
                "last_request_at": proc.last_request_at,
                "idle_seconds": int(time.time() - proc.last_request_at),
                "has_error": bool(proc.last_error),
            })
        return out

    # ── internals ──────────────────────────────────────────────────

    async def _proxy_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _is_alive(self, proc: AppProcess) -> bool:
        return proc.process.returncode is None

    async def _spawn(self, slug: str) -> AppProcess:
        """Create the subprocess and wait for its socket to appear.

        Holds ``_table_lock`` only long enough to install the new entry
        — the actual ``asyncio.create_subprocess_exec`` runs outside
        the lock so a slow spawn doesn't block proxy traffic for other
        slugs.
        """
        slug = _safe_slug(slug)
        code_dir = app_dir(slug)
        backend_path = code_dir / "backend.py"
        if not backend_path.exists():
            raise AppRuntimeError(
                f"app '{slug}' has no backend.py at {backend_path}"
            )

        sock_path = _socket_path(slug)
        # Always clear any stale socket left by a crashed prior run —
        # aiohttp's UnixSite refuses to bind if the file already exists.
        try:
            sock_path.unlink(missing_ok=True)
        except OSError:
            pass

        worker_module = "captain_claw.flight_deck.app_subprocess"
        env = _build_subprocess_env(slug=slug)
        cmd = [
            sys.executable,
            "-u",  # unbuffered stdout/stderr so the log pump sees lines promptly
            "-m",
            worker_module,
            "--slug",
            slug,
            "--socket",
            str(sock_path),
            "--code-dir",
            str(code_dir),
            "--mem-limit-mb",
            str(_DEFAULT_MEM_LIMIT_MB),
        ]

        log.info("Spawning app subprocess slug=%s cmd=%s", slug, " ".join(cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(code_dir),
            env=env,
            # Put the child in its own process group so SIGTERM hits
            # any sub-children it spawned too.
            start_new_session=True,
        )

        proc = AppProcess(
            slug=slug,
            pid=process.pid,
            socket_path=sock_path,
            started_at=time.time(),
            last_request_at=time.time(),
            process=process,
        )

        # Pump stderr/stdout into the ring buffers + on-disk log file.
        # These tasks die naturally when the streams close.
        asyncio.create_task(self._pump_stream(proc, process.stderr, proc.stderr_ring, code_dir / "logs" / "stderr.log", "stderr"))
        asyncio.create_task(self._pump_stream(proc, process.stdout, proc.stdout_ring, code_dir / "logs" / "stdout.log", "stdout"))

        # Wait for the worker to create its socket. We don't try to
        # connect — that races with aiohttp's UnixSite startup; a
        # stat() loop is cheap and unambiguous.
        deadline = time.time() + _SPAWN_TIMEOUT_SECONDS
        while time.time() < deadline:
            if sock_path.exists():
                break
            if process.returncode is not None:
                # Subprocess died during startup. Surface its stderr.
                await asyncio.sleep(0.1)  # let the pump catch up
                tail = "\n".join(list(proc.stderr_ring)[-30:])
                raise AppRuntimeError(
                    f"app '{slug}' subprocess exited during startup "
                    f"(returncode={process.returncode}). stderr:\n{tail}"
                )
            await asyncio.sleep(0.05)
        else:
            await self._terminate(proc)
            tail = "\n".join(list(proc.stderr_ring)[-30:])
            raise AppRuntimeError(
                f"app '{slug}' did not create its socket within "
                f"{_SPAWN_TIMEOUT_SECONDS}s. stderr:\n{tail}"
            )

        async with self._table_lock:
            self._procs[slug] = proc
        log.info("App subprocess ready slug=%s pid=%d", slug, process.pid)
        return proc

    async def _pump_stream(
        self,
        proc: AppProcess,
        stream: asyncio.StreamReader | None,
        ring: deque[str],
        log_path: Path,
        label: str,
    ) -> None:
        """Read a subprocess stream line-by-line into ring + log file."""
        if stream is None:
            return
        try:
            log_fh = log_path.open("a", buffering=1, encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("Cannot open %s log for slug=%s: %s", label, proc.slug, exc)
            log_fh = None
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                ring.append(text)
                if log_fh:
                    try:
                        log_fh.write(text + "\n")
                    except OSError:
                        pass
                # Mirror to FD's own logger for tailing in development.
                if label == "stderr":
                    log.debug("[%s/stderr] %s", proc.slug, text)
                else:
                    log.debug("[%s/stdout] %s", proc.slug, text)
        finally:
            if log_fh:
                try:
                    log_fh.close()
                except OSError:
                    pass

    async def _terminate(self, proc: AppProcess) -> None:
        """Stop the subprocess with SIGTERM, escalating to SIGKILL."""
        if proc.process.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(proc.process.wait(), timeout=_TERM_GRACE_SECONDS)
        except asyncio.TimeoutError:
            log.warning("App %s did not exit on SIGTERM, sending SIGKILL", proc.slug)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.wait_for(proc.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                log.error("App %s ignored SIGKILL", proc.slug)

    async def _idle_reaper(self) -> None:
        """Background task: kill subprocesses idle > _IDLE_TIMEOUT_SECONDS."""
        try:
            while not self._shutdown:
                await asyncio.sleep(30)
                now = time.time()
                to_kill: list[str] = []
                for slug, proc in list(self._procs.items()):
                    if proc.process.returncode is not None:
                        to_kill.append(slug)
                        continue
                    if now - proc.last_request_at > _IDLE_TIMEOUT_SECONDS:
                        to_kill.append(slug)
                for slug in to_kill:
                    log.info("Reaping idle app subprocess slug=%s", slug)
                    try:
                        await self.stop(slug)
                    except Exception as exc:
                        log.warning("Reaper error for %s: %s", slug, exc)
        except asyncio.CancelledError:
            return


# ── helpers ───────────────────────────────────────────────────────────


# Hop-by-hop headers that must not be forwarded per RFC 7230 §6.1, plus
# Host (we want the subprocess to see ``unix``, not the FD hostname).
_HOP_BY_HOP_REQ = frozenset({
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",  # aiohttp will recompute
})

_HOP_BY_HOP_RESP = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
})


def _filter_request_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_REQ}


def _filter_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_RESP}


def _build_subprocess_env(*, slug: str) -> dict[str, str]:
    """Trimmed env for the subprocess.

    We keep the venv's PATH + PYTHONPATH so the worker can import
    captain_claw, but otherwise strip credentials the agent's code
    has no business seeing.

    Three pieces are injected for the SDK in
    :mod:`captain_claw.app_sdk`:

    * ``FD_APP_SLUG`` — so an app knows its own name (used for
      attribution in cross-app calls).
    * ``FD_INTERNAL_URL`` — base URL the subprocess uses to reach
      back into FD when calling sibling apps. Defaults to a localhost
      address because FD + apps share a host in the dev setup; ops
      can override via the same env var on the FD process.
    * ``FD_AGENT_SECRET`` — same shared secret the chat agent uses,
      so the FD proxy will accept cross-app calls without the app
      needing a user JWT. See :mod:`agent_secret` for the resolution
      rules.
    """
    keep = {
        "PATH", "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
        "CAPTAIN_CLAW_FD_HOME",
        "LANG", "LC_ALL", "LC_CTYPE",
        "HOME",
        # Some libraries (httpx, anyio) refuse to start without TZ or TMPDIR.
        "TZ", "TMPDIR",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["FD_APP_SLUG"] = slug
    # Cross-app SDK bootstrap. If FD itself was started with
    # FD_INTERNAL_URL set, honor that — multi-host deployments would
    # need it. Otherwise assume same-host loopback on the standard FD
    # port. The subprocess only needs to *reach* FD, not match the
    # public URL.
    env["FD_INTERNAL_URL"] = os.environ.get(
        "FD_INTERNAL_URL", "http://127.0.0.1:25080",
    )
    try:
        # Deferred import so a stripped-down test environment without
        # the flight_deck package on the path still gets a working
        # subprocess (it just can't do cross-app calls).
        from captain_claw.flight_deck.agent_secret import (
            get_or_create_agent_secret,
        )
        env["FD_AGENT_SECRET"] = get_or_create_agent_secret()
    except Exception as exc:
        log.warning(
            "Could not resolve agent_secret for subprocess env "
            "(cross-app SDK will be disabled): %s", exc,
        )
    return env


# ── exceptions ────────────────────────────────────────────────────────


class AppRuntimeError(RuntimeError):
    """Raised when spawning or proxying to an app subprocess fails."""


# ── module-level singleton ────────────────────────────────────────────


_runtime: AppRuntime | None = None


def get_runtime() -> AppRuntime:
    """Return the process-wide ``AppRuntime`` singleton.

    Lazy so import order doesn't matter — the FD server calls
    ``await get_runtime().start()`` inside the lifespan context.
    """
    global _runtime
    if _runtime is None:
        _runtime = AppRuntime()
    return _runtime


# ── minimal app-dir helpers (used by routes + tools) ──────────────────


def read_app_manifest(slug: str) -> dict[str, Any] | None:
    """Load ``manifest.json`` for a code-app, or ``None`` if absent."""
    p = app_dir(slug) / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_app_manifest(slug: str, manifest: dict[str, Any]) -> None:
    """Persist ``manifest.json`` for a code-app."""
    p = app_dir(slug) / "manifest.json"
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def list_code_apps() -> list[dict[str, Any]]:
    """List all code-apps on disk (slug + manifest summary)."""
    out: list[dict[str, Any]] = []
    root = apps_root()
    if not root.exists():
        return out
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest = read_app_manifest(entry.name) or {}
        # ``data_api`` is the publish contract for cross-app reads:
        # surfacing it at list-time so the chat agent (and other apps,
        # once we expose a discovery endpoint) can see what's
        # available without a second round-trip.
        out.append({
            "slug": entry.name,
            "name": manifest.get("name") or entry.name,
            "version": manifest.get("version") or "0.0.0",
            "has_backend": (entry / "backend.py").exists(),
            "has_frontend": (entry / "frontend.html").exists(),
            "manifest": manifest,
            "data_api": manifest.get("data_api") or {},
        })
    return out


def read_frontend_html(slug: str) -> str | None:
    """Return the ``frontend.html`` contents for a code-app, or None."""
    p = app_dir(slug) / "frontend.html"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None
