"""PTY session daemon — the local "terminal app" on the Mac.

Holds long-lived pseudo-terminal sessions and exposes a tiny JSON/HTTP
API on localhost.  The agent's ``terminal`` tool talks to it; because the
daemon is a separate process, the sessions survive agent restarts (which
happen on every code change).

Run it::

    python -m captain_claw.terminal.daemon            # 127.0.0.1:23190
    CLAW_PTY_PORT=23191 CLAW_PTY_TOKEN=secret python -m captain_claw.terminal.daemon

Security model: binds to 127.0.0.1 only and, when ``CLAW_PTY_TOKEN`` is
set, requires that token in the ``X-Claw-Token`` header.  A PTY session is
arbitrary code execution on this machine — keep it local and tokened.

API (all POST, JSON in/out)::

    /open    {cmd?, cwd?, cols?, rows?, env?}      -> {session_id, pid}
    /input   {session_id, data?, key?, enter?}     -> {ok}
    /read    {session_id, wait?, settle?, cursor?} -> {output, cursor, alive, exit_code}
    /list    {}                                    -> {sessions: [...]}
    /resize  {session_id, cols, rows}              -> {ok}
    /close   {session_id}                          -> {ok}
    /health  {}                                    -> {ok, sessions}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pty
import signal
import struct
import sys
import time
import uuid

import fcntl
import termios

from aiohttp import web

log = logging.getLogger("claw.pty")

# Named keys → raw bytes written to the PTY.  Covers the common interactive
# keypresses an agent needs to "emulate"; arbitrary sequences go via ``data``.
KEYS: dict[str, bytes] = {
    "enter": b"\r",
    "tab": b"\t",
    "esc": b"\x1b",
    "escape": b"\x1b",
    "space": b" ",
    "backspace": b"\x7f",
    "up": b"\x1b[A",
    "down": b"\x1b[B",
    "right": b"\x1b[C",
    "left": b"\x1b[D",
    "home": b"\x1b[H",
    "end": b"\x1b[F",
    "pageup": b"\x1b[5~",
    "pagedown": b"\x1b[6~",
    "delete": b"\x1b[3~",
    "ctrl-c": b"\x03",
    "ctrl-d": b"\x04",
    "ctrl-z": b"\x1a",
    "ctrl-l": b"\x0c",
    "ctrl-u": b"\x15",
    "ctrl-a": b"\x01",
    "ctrl-e": b"\x05",
    "ctrl-r": b"\x12",
}

# Cap the per-session output buffer so a chatty process can't grow memory
# without bound.  Reads use absolute stream offsets, so trimming the front
# of the buffer is transparent to a reader that keeps its cursor.
_MAX_BUFFER = 512 * 1024

# Live mirror: echo PTY output to the daemon's own console so the operator
# watching the daemon window sees sessions as they happen. On by default;
# set CLAW_PTY_MIRROR=0 to silence it.
_MIRROR = os.environ.get("CLAW_PTY_MIRROR", "1").lower() not in ("0", "false", "no", "off")


def _mirror(data: bytes) -> None:
    if not _MIRROR:
        return
    try:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except Exception:
        pass


class PtySession:
    """One pseudo-terminal running a child process."""

    def __init__(self, cmd: list[str], cwd: str, env: dict[str, str], cols: int, rows: int):
        self.id = uuid.uuid4().hex[:12]
        self.cmd = cmd
        self.cwd = cwd
        self.cols = cols
        self.rows = rows
        self.created = time.time()
        self.pid: int | None = None
        self.master_fd: int | None = None
        self.alive = True
        self.exit_code: int | None = None
        # Absolute-offset output buffer: ``base`` is the stream position of
        # ``buffer[0]``; ``total`` is bytes ever produced.
        self._buffer = bytearray()
        self._base = 0
        self._total = 0
        self._last_output = time.time()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── lifecycle ────────────────────────────────────────────────────
    def spawn(self, env: dict[str, str]) -> None:
        pid, master_fd = pty.fork()
        if pid == 0:  # child
            try:
                os.chdir(self.cwd)
            except Exception:
                pass
            try:
                os.execvpe(self.cmd[0], self.cmd, env)
            except Exception as exc:  # pragma: no cover - child path
                os.write(2, f"exec failed: {exc}\n".encode())
                os._exit(127)
        # parent
        self.pid = pid
        self.master_fd = master_fd
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._set_winsize(self.rows, self.cols)

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        assert self.master_fd is not None
        loop.add_reader(self.master_fd, self._on_readable)

    def _set_winsize(self, rows: int, cols: int) -> None:
        if self.master_fd is None:
            return
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            pass

    def resize(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        self._set_winsize(rows, cols)

    # ── io ───────────────────────────────────────────────────────────
    def _on_readable(self) -> None:
        if self.master_fd is None:
            return
        try:
            while True:
                chunk = os.read(self.master_fd, 65536)
                if not chunk:
                    self._reap()
                    return
                self._append(chunk)
        except BlockingIOError:
            return
        except OSError:
            # EIO on macOS/Linux when the child has exited.
            self._reap()

    def _append(self, chunk: bytes) -> None:
        _mirror(chunk)
        self._buffer.extend(chunk)
        self._total += len(chunk)
        self._last_output = time.time()
        if len(self._buffer) > _MAX_BUFFER:
            drop = len(self._buffer) - _MAX_BUFFER
            del self._buffer[:drop]
            self._base += drop

    def write(self, data: bytes) -> None:
        if self.master_fd is None or not self.alive:
            raise RuntimeError("session is not alive")
        os.write(self.master_fd, data)

    async def read_new(self, cursor: int, wait: float, settle: float) -> tuple[str, int]:
        """Return output produced after *cursor*, advancing the stream offset.

        Waits up to *wait* seconds for output to appear, then keeps waiting
        while output is still actively arriving (quiescing for *settle*).
        """
        deadline = time.time() + max(0.0, wait)
        while self._total <= cursor and self.alive and time.time() < deadline:
            await asyncio.sleep(0.03)
        if settle > 0:
            last = self._total
            while time.time() < deadline:
                await asyncio.sleep(min(settle, max(0.0, deadline - time.time())))
                if self._total == last:
                    break
                last = self._total
        start = max(cursor, self._base)
        data = bytes(self._buffer[start - self._base:])
        return data.decode("utf-8", errors="replace"), self._total

    @property
    def total(self) -> int:
        return self._total

    def _reap(self) -> None:
        if not self.alive:
            return
        self.alive = False
        if self._loop is not None and self.master_fd is not None:
            try:
                self._loop.remove_reader(self.master_fd)
            except Exception:
                pass
        if self.pid is not None:
            try:
                _, status = os.waitpid(self.pid, os.WNOHANG)
                if os.WIFEXITED(status):
                    self.exit_code = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    self.exit_code = -os.WTERMSIG(status)
            except ChildProcessError:
                pass
            except Exception:
                pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except Exception:
                pass
            self.master_fd = None
        _mirror(f"\n\x1b[2m── session {self.id} ended (exit {self.exit_code}) ──\x1b[0m\n".encode())

    def kill(self) -> None:
        if self.pid is not None and self.alive:
            try:
                os.kill(self.pid, signal.SIGHUP)
            except ProcessLookupError:
                pass
            except Exception:
                pass
        self._reap()

    def info(self) -> dict:
        return {
            "session_id": self.id,
            "cmd": " ".join(self.cmd),
            "cwd": self.cwd,
            "pid": self.pid,
            "alive": self.alive,
            "exit_code": self.exit_code,
            "cols": self.cols,
            "rows": self.rows,
            "bytes": self._total,
            "age_seconds": round(time.time() - self.created, 1),
            "idle_seconds": round(time.time() - self._last_output, 1),
        }


OPS = ("open", "input", "read", "resize", "close", "list", "health")


class OpError(Exception):
    """A daemon op failed; carries an HTTP-style status for the caller."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Daemon:
    """Owns the PTY sessions. Transport-agnostic: the same ``dispatch`` core
    serves both the local HTTP API and the dial-out WebSocket relay."""

    def __init__(self, token: str | None):
        self.token = token
        self.sessions: dict[str, PtySession] = {}
        # Per-session read cursor so the caller gets "what's new since last
        # read" without having to track offsets itself.
        self.cursors: dict[str, int] = {}
        # Default working directory for new sessions: the folder the daemon
        # was launched from (so terminals open where the operator is working).
        self.base_cwd = os.getcwd()

    def _get(self, sid: str) -> PtySession:
        sess = self.sessions.get(sid)
        if sess is None:
            raise OpError(404, f"no session {sid!r}")
        return sess

    async def dispatch(self, op: str, body: dict) -> dict:
        """Execute one operation against the session pool.

        Returns a JSON-able body dict; raises ``OpError`` on failure. This is
        the single entry point both transports funnel through.
        """
        body = body if isinstance(body, dict) else {}
        if op == "open":
            return await self._open(body)
        if op == "input":
            return await self._input(body)
        if op == "read":
            return await self._read(body)
        if op == "resize":
            return await self._resize(body)
        if op == "close":
            return await self._close(body)
        if op == "list":
            return {"sessions": [s.info() for s in self.sessions.values()]}
        if op == "health":
            return {"ok": True, "sessions": len(self.sessions)}
        raise OpError(404, f"unknown op {op!r}")

    async def _open(self, body: dict) -> dict:
        cmd_raw = body.get("cmd")
        if isinstance(cmd_raw, str) and cmd_raw.strip():
            # A string is a shell command line: run it through a login shell so
            # shell syntax (cd, &&, pipes, env) and the user's PATH work, and
            # interactive programs (claude, python3) still get the pty. Pass an
            # explicit list instead if you want literal argv with no shell.
            shell = os.environ.get("SHELL") or "/bin/bash"
            cmd = [shell, "-lc", cmd_raw.strip()]
        elif isinstance(cmd_raw, list) and cmd_raw:
            cmd = [str(x) for x in cmd_raw]
        else:
            cmd = [os.environ.get("SHELL", "/bin/zsh"), "-i"]
        cwd = str(body.get("cwd") or self.base_cwd)
        cols = int(body.get("cols") or 120)
        rows = int(body.get("rows") or 40)
        env = dict(os.environ)
        env["TERM"] = env.get("TERM", "xterm-256color")
        for k, v in (body.get("env") or {}).items():
            env[str(k)] = str(v)
        sess = PtySession(cmd, cwd, env, cols, rows)
        _mirror(f"\n\x1b[2m── session {sess.id} · {' '.join(cmd)} · {cwd} ──\x1b[0m\n".encode())
        sess.spawn(env)
        sess.attach(asyncio.get_running_loop())
        self.sessions[sess.id] = sess
        self.cursors[sess.id] = 0
        log.info("opened session %s pid=%s cmd=%s cwd=%s", sess.id, sess.pid, " ".join(cmd), cwd)
        return {"session_id": sess.id, "pid": sess.pid}

    async def _input(self, body: dict) -> dict:
        sess = self._get(str(body.get("session_id", "")))
        data = body.get("data")
        key = body.get("key")
        payload = b""
        if isinstance(data, str):
            payload += data.encode("utf-8")
        if isinstance(key, str) and key.strip():
            mapped = KEYS.get(key.strip().lower())
            if mapped is None:
                raise OpError(400, f"unknown key {key!r}")
            payload += mapped
        if body.get("enter"):
            payload += b"\r"
        try:
            sess.write(payload)
        except RuntimeError as exc:
            raise OpError(409, str(exc))
        return {"ok": True}

    async def _read(self, body: dict) -> dict:
        sess = self._get(str(body.get("session_id", "")))
        cursor = body.get("cursor")
        if cursor is None:
            cursor = self.cursors.get(sess.id, 0)
        wait = float(body.get("wait", 1.0))
        settle = float(body.get("settle", 0.25))
        output, new_cursor = await sess.read_new(int(cursor), wait, settle)
        self.cursors[sess.id] = new_cursor
        return {
            "output": output,
            "cursor": new_cursor,
            "alive": sess.alive,
            "exit_code": sess.exit_code,
        }

    async def _resize(self, body: dict) -> dict:
        sess = self._get(str(body.get("session_id", "")))
        sess.resize(int(body.get("cols") or sess.cols), int(body.get("rows") or sess.rows))
        return {"ok": True}

    async def _close(self, body: dict) -> dict:
        sess = self._get(str(body.get("session_id", "")))
        sess.kill()
        self.sessions.pop(sess.id, None)
        self.cursors.pop(sess.id, None)
        return {"ok": True}

    def shutdown(self) -> None:
        for sess in list(self.sessions.values()):
            sess.kill()


# ── HTTP transport (local / reachable-network mode) ──────────────────


async def _json(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_app(token: str | None) -> web.Application:
    d = Daemon(token)

    def _check_auth(request: web.Request) -> None:
        if token and request.headers.get("X-Claw-Token") != token:
            raise web.HTTPUnauthorized(text="bad or missing X-Claw-Token")

    def _make(op: str):
        async def handler(request: web.Request) -> web.Response:
            if op != "health":
                _check_auth(request)
            body = await _json(request)
            try:
                return web.json_response(await d.dispatch(op, body))
            except OpError as exc:
                return web.json_response({"error": exc.message}, status=exc.status)
        return handler

    app = web.Application()
    app.add_routes([
        web.post("/open", _make("open")),
        web.post("/input", _make("input")),
        web.post("/read", _make("read")),
        web.post("/resize", _make("resize")),
        web.post("/close", _make("close")),
        web.post("/list", _make("list")),
        web.get("/health", _make("health")),
        web.post("/health", _make("health")),
    ])

    async def _shutdown(app: web.Application) -> None:
        d.shutdown()

    app.on_cleanup.append(_shutdown)
    return app


# ── Dial-out transport (NAT-friendly relay mode) ─────────────────────


async def run_relay(relay_url: str, worker: str, token: str | None) -> None:
    """Dial OUT to a Flight Deck relay and service requests over the socket.

    For the case where this machine is behind NAT and the agent/Flight Deck
    are elsewhere: instead of waiting for inbound connections, we open a
    persistent WebSocket to the relay, register as *worker*, and answer the
    ``request`` frames it forwards from the agent's terminal tool.

    Uses the ``websockets`` client rather than aiohttp's: reverse proxies
    (Cloudflare/nginx) in front of Flight Deck often add a second ``Server``
    handshake header, which aiohttp rejects with a 400 but ``websockets``
    tolerates.
    """
    from websockets.asyncio.client import connect

    daemon = Daemon(token=None)  # auth happens at connect time, not per-op
    backoff = 1.0
    while True:
        try:
            # max_size=None: PTY output frames can exceed the default 1 MiB cap.
            async with connect(relay_url, max_size=None, open_timeout=20) as ws:
                await ws.send(json.dumps({"type": "register", "worker": worker, "token": token}))
                log.info("relay connected: %s as worker %r", relay_url, worker)
                backoff = 1.0
                send_lock = asyncio.Lock()

                async def serve(req: dict) -> None:
                    rid = req.get("id")
                    op = str(req.get("op", ""))
                    try:
                        body = await daemon.dispatch(op, req.get("payload") or {})
                        resp = {"type": "response", "id": rid, "status": 200, "body": body}
                    except OpError as exc:
                        resp = {"type": "response", "id": rid, "status": exc.status,
                                "body": {"error": exc.message}}
                    except Exception as exc:  # pragma: no cover - defensive
                        resp = {"type": "response", "id": rid, "status": 500,
                                "body": {"error": str(exc)}}
                    async with send_lock:
                        await ws.send(json.dumps(resp))

                async for message in ws:
                    try:
                        data = json.loads(message)
                    except Exception:
                        continue
                    if isinstance(data, dict) and data.get("type") == "request":
                        # Per-request task so a slow `read` can't block other
                        # requests or the keepalive ping.
                        asyncio.create_task(serve(data))
            log.warning("relay connection closed; reconnecting")
        except Exception as exc:
            log.warning("relay connection failed: %s", exc)
        # Sessions deliberately survive a dropped socket — the PTYs are still
        # alive in this process; only the transport reconnects.
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    token = os.environ.get("CLAW_PTY_TOKEN") or None

    # Dial-out / relay mode: when CLAW_PTY_RELAY is set, this machine is behind
    # NAT and reaches Flight Deck by opening an outbound WebSocket instead of
    # serving an inbound port. The agent's terminal tool talks to FD's relay,
    # which tunnels each call down to us.
    relay_url = os.environ.get("CLAW_PTY_RELAY")
    if relay_url:
        worker = os.environ.get("CLAW_PTY_WORKER", "default")
        log.info("PTY daemon dialing out to %s as worker %r", relay_url, worker)
        asyncio.run(run_relay(relay_url, worker, token))
        return

    host = os.environ.get("CLAW_PTY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("CLAW_PTY_PORT", "23190"))

    # A PTY session is arbitrary code execution. When binding anywhere other
    # than loopback (i.e. exposing it to another machine), demand a token —
    # unless the operator explicitly opts out via CLAW_PTY_INSECURE=1.
    if host not in _LOOPBACK and not token and os.environ.get("CLAW_PTY_INSECURE") != "1":
        raise SystemExit(
            f"Refusing to bind {host}:{port} without CLAW_PTY_TOKEN — a PTY is "
            "remote code execution. Set a token (and put this behind TLS or a "
            "private network), or set CLAW_PTY_INSECURE=1 to override."
        )

    app = build_app(token)
    log.info(
        "PTY daemon on %s:%s (auth=%s)%s",
        host, port, "on" if token else "off",
        "" if host in _LOOPBACK else "  [exposed to network]",
    )
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
