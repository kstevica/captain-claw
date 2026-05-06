"""MCP transport abstractions used by :class:`MCPManager`.

Phase 2 introduced two transports:

* :class:`HttpTransport` — speaks JSON-RPC over the *Streamable HTTP*
  binding (POST + optional ``text/event-stream`` response, with an
  ``Mcp-Session-Id`` header for session continuity). This is what
  Phase 1 already supported.

* :class:`StdioTransport` — spawns a long-lived child process and
  speaks JSON-RPC over stdin/stdout, one JSON object per line. This is
  the de-facto standard for local MCP servers shipped via ``npx`` /
  ``uvx`` (filesystem, sqlite, github, etc.).

Both implement the same tiny interface:

* ``await t.request(method, params) -> dict``
* ``await t.notify(method, params) -> None``
* ``await t.close() -> None``

so the manager doesn't care which transport it's talking to.

Concurrency notes
-----------------

For HTTP the answer is trivial — every request is its own POST.

For stdio we run a single background reader task that reads lines from
stdout and dispatches each parsed JSON-RPC response to the matching
pending ``asyncio.Future`` (looked up by ``id``). That means many
concurrent requests can be in flight on a single subprocess, each
correlated by id, with the upstream server free to respond out of
order.

When the child process exits unexpectedly all pending futures are
failed with :class:`MCPServerError` so callers don't hang.
"""

from __future__ import annotations

import abc
import asyncio
import itertools
import json
import os
import shutil
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)


class MCPTransportError(RuntimeError):
    """Raised when transport-level I/O fails (network, subprocess, framing).

    The manager re-raises these as ``MCPServerError`` so callers see a
    consistent error type regardless of transport.
    """


# ── interface ───────────────────────────────────────────────────────


class Transport(abc.ABC):
    """Common surface for HTTP and stdio MCP clients.

    Subclasses may set :attr:`on_notification` to a callable that will
    receive any server-initiated JSON-RPC notification (a message with
    ``method`` but no ``id``). The manager uses this to fire
    ``notifications/tools/list_changed`` into the
    :mod:`captain_claw.flight_deck.mcp_events` bus so subscribed agents
    can hot-reload their proxy tools.

    HTTP transports never receive server-initiated notifications in the
    Phase 1 wire format (the response is a one-shot SSE), so this hook
    is wired for the stdio transport only — but exposing it on the ABC
    keeps the manager's wiring uniform.
    """

    #: Called for every JSON-RPC notification (no-id message) from the
    #: server. Stays ``None`` when the manager hasn't wired a handler.
    on_notification: Callable[[dict[str, Any]], None] | None = None

    @abc.abstractmethod
    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and return its parsed ``result`` dict.

        Raises :class:`MCPTransportError` on framing / I/O failure and
        the upstream's RPC errors are surfaced as the same type with
        the upstream message included.

        When ``on_progress`` is provided the transport opts the call
        into MCP progress notifications: the ``params`` dict is
        augmented with a ``_meta.progressToken`` and any
        ``notifications/progress`` from the server is delivered to the
        callback (each as the JSON-RPC ``params`` dict, e.g.
        ``{"progressToken": ..., "progress": 0.42, "total": 1.0,
        "message": "..."}``). The final ``result`` dict is still the
        function's return value — progress is purely a side channel.
        """

    @abc.abstractmethod
    async def notify(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (subprocess, sockets, etc.)."""


def _rpc_request(method: str, params: dict[str, Any] | None, id_: int) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params is not None:
        msg["params"] = params
    return msg


def _rpc_notification(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# ── HTTP transport ──────────────────────────────────────────────────


class HttpTransport(Transport):
    """JSON-RPC over Streamable HTTP (the Phase 1 transport).

    The transport keeps an :class:`httpx.AsyncClient` alive across
    calls, captures any ``Mcp-Session-Id`` header from the first
    response and replays it on subsequent requests, and parses
    ``text/event-stream`` responses transparently.

    OAuth ``client_credentials`` flow (when ``token_endpoint`` /
    ``client_id`` are configured) is owned by this transport so callers
    don't have to know whether auth applies.
    """

    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record
        self._client: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._session_id: str | None = None
        self._id_counter = itertools.count(1)
        self.last_status_code: int | None = None

    @property
    def url(self) -> str:
        return str(self._record.get("url") or "").rstrip("/")

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    def reset_session(self) -> None:
        self._access_token = None
        self._session_id = None

    # ── OAuth ────────────────────────────────────────────────────────

    def _resolve_token_endpoint(self) -> str:
        ep = str(self._record.get("token_endpoint") or "")
        if not ep:
            return ""
        if ep.startswith("http://") or ep.startswith("https://"):
            return ep
        parsed = urlparse(self.url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return base + "/" + ep.lstrip("/")

    async def _ensure_token(self) -> str:
        if self._access_token:
            return self._access_token
        token_endpoint = self._resolve_token_endpoint()
        client_id = str(self._record.get("client_id") or "")
        if client_id and not token_endpoint:
            raise MCPTransportError(
                "client_id is set but token_endpoint is empty — fill in the "
                "OAuth2 token URL (absolute or relative to the server, e.g. "
                "/api/mcp/oauth/token)"
            )
        if token_endpoint and not client_id:
            raise MCPTransportError(
                "token_endpoint is set but client_id is empty — provide the "
                "OAuth2 client_id, or clear the token_endpoint to skip OAuth"
            )
        if not token_endpoint or not client_id:
            return ""
        client_secret = str(self._record.get("client_secret") or "")
        client = await self._http()
        resp = await client.post(
            token_endpoint,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise MCPTransportError(
                f"OAuth token endpoint returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        token = str(data.get("access_token") or "").strip()
        if not token:
            raise MCPTransportError("OAuth response did not contain access_token")
        self._access_token = token
        return token

    async def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        for k, v in (self._record.get("headers") or {}).items():
            headers[str(k)] = str(v)
        token = await self._ensure_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    @staticmethod
    def _iter_sse_messages(text: str) -> list[dict[str, Any]]:
        """Yield every JSON object decoded from ``data:`` lines, in order.

        SSE bodies for streaming MCP responses contain a sequence of
        ``notifications/progress`` envelopes followed by the final
        JSON-RPC response. Older non-streaming bodies have a single
        envelope. The caller picks out progress vs result.
        """
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if not data_str:
                continue
            try:
                msg = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict):
                out.append(msg)
        return out

    @classmethod
    def _parse_sse(
        cls,
        text: str,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        *,
        notification_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Walk an SSE body, fan progress out to ``on_progress`` and
        return the final JSON-RPC result.

        ``notification_handler`` receives every non-progress
        notification (e.g. ``notifications/tools/list_changed``) so
        the manager's bus stays in the loop even on HTTP transports.
        """
        for msg in cls._iter_sse_messages(text):
            if "error" in msg:
                err = msg["error"]
                raise MCPTransportError(
                    f"MCP RPC error {err.get('code', '?')}: {err.get('message', str(err))}"
                )
            if "result" in msg:
                result = msg["result"]
                return result if isinstance(result, dict) else {"result": result}
            # No id + method ⇒ server-initiated notification.
            method = str(msg.get("method") or "")
            if method == "notifications/progress" and on_progress is not None:
                params = msg.get("params") or {}
                if isinstance(params, dict):
                    try:
                        on_progress(params)
                    except Exception:
                        log.debug(
                            "MCP HTTP: on_progress handler raised", exc_info=True
                        )
            elif method and notification_handler is not None:
                try:
                    notification_handler(msg)
                except Exception:
                    log.debug(
                        "MCP HTTP: notification_handler raised", exc_info=True
                    )
        raise MCPTransportError("MCP server returned SSE without a JSON-RPC result")

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        client = await self._http()
        headers = await self._build_headers()
        resp = await client.post(self.url, json=payload, headers=headers)
        self.last_status_code = resp.status_code
        sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        if sid and not self._session_id:
            self._session_id = sid
        return resp

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        msg_id = next(self._id_counter)
        # When the caller wants progress, opt into the MCP progress
        # mechanism by stamping a unique token into params._meta.
        # Servers that don't support progress will simply ignore it.
        out_params = params
        if on_progress is not None:
            out_params = dict(params or {})
            meta = dict(out_params.get("_meta") or {})
            meta["progressToken"] = f"progress-{msg_id}"
            out_params["_meta"] = meta
        payload = _rpc_request(method, out_params, msg_id)
        resp = await self._post(payload)
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            return self._parse_sse(
                resp.text,
                on_progress,
                notification_handler=self.on_notification,
            )
        if resp.status_code == 401:
            self._access_token = None
            raise MCPTransportError(
                "MCP server rejected request (401); token cleared"
            )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            err = body["error"]
            raise MCPTransportError(
                f"MCP RPC error {err.get('code', '?')}: {err.get('message', str(err))}"
            )
        result = body.get("result", body)
        return result if isinstance(result, dict) else {"result": result}

    async def notify(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        try:
            await self._post(_rpc_notification(method, params))
        except Exception:
            # Notifications are best-effort by spec; failures are logged
            # but don't propagate.
            log.debug("MCP notify failed (best-effort)", method=method, exc_info=True)


# ── stdio transport ─────────────────────────────────────────────────


class StdioTransport(Transport):
    """JSON-RPC over a child process's stdin/stdout.

    The framing is the standard "one JSON object per line" used by the
    MCP stdio binding (NDJSON). The transport spawns the child lazily
    on the first request, runs a background reader task that
    dispatches responses to pending futures by id, and forwards
    notifications fire-and-forget over stdin.

    Process lifecycle:

    * Spawn on first :meth:`request` / :meth:`notify`.
    * If the child exits, all pending futures are failed with
      :class:`MCPTransportError` and a fresh process is spawned on the
      next call (so a flaky child auto-recovers).
    * :meth:`close` terminates the child gracefully (SIGTERM with a
      2-second grace, then SIGKILL).
    """

    # Cap on stdout line length. MCP responses can be large (tool
    # outputs include base64 images, etc.) so we lift asyncio's default
    # 64 KiB to something generous. 16 MiB is enough for any sane tool
    # response without letting a misbehaving child OOM us.
    _STDOUT_LINE_LIMIT = 16 * 1024 * 1024

    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        # Per-call progress handlers, keyed by progressToken (string).
        # Populated for the lifetime of one ``request`` call so the
        # reader can dispatch ``notifications/progress`` to the right
        # caller. Cleaned up in the request's ``finally`` block.
        self._progress_handlers: dict[
            str, Callable[[dict[str, Any]], None]
        ] = {}
        self._id_counter = itertools.count(1)
        self._spawn_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        # Set when the reader detects the child has exited.  Used to
        # fail any future request immediately rather than blocking on
        # an unresponsive write.
        self._dead = False

    @property
    def command(self) -> str:
        return str(self._record.get("command") or "").strip()

    @property
    def args(self) -> list[str]:
        raw = self._record.get("args") or []
        return [str(a) for a in raw] if isinstance(raw, list) else []

    @property
    def env(self) -> dict[str, str]:
        raw = self._record.get("env") or {}
        return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

    # ── process lifecycle ────────────────────────────────────────────

    async def _ensure_proc(self) -> asyncio.subprocess.Process:
        async with self._spawn_lock:
            if self._proc is not None and self._proc.returncode is None:
                return self._proc
            # Either never spawned or the child died — spawn fresh.
            await self._cleanup_dead_proc()
            command = self.command
            if not command:
                raise MCPTransportError("stdio transport: 'command' is empty")
            resolved = shutil.which(command) or command
            full_env = os.environ.copy()
            for k, v in self.env.items():
                full_env[k] = v
            try:
                proc = await asyncio.create_subprocess_exec(
                    resolved,
                    *self.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=full_env,
                    limit=self._STDOUT_LINE_LIMIT,
                )
            except FileNotFoundError as exc:
                raise MCPTransportError(
                    f"stdio transport: cannot execute {command!r}: {exc}"
                ) from exc
            except Exception as exc:
                raise MCPTransportError(
                    f"stdio transport: failed to spawn {command!r}: {exc}"
                ) from exc
            self._proc = proc
            self._dead = False
            self._reader_task = asyncio.create_task(
                self._read_loop(proc), name=f"mcp-stdio-reader[{command}]"
            )
            self._stderr_task = asyncio.create_task(
                self._stderr_loop(proc), name=f"mcp-stdio-stderr[{command}]"
            )
            log.info(
                "MCP stdio child spawned",
                command=command,
                args=self.args,
                pid=proc.pid,
            )
            return proc

    async def _cleanup_dead_proc(self) -> None:
        """Reap a previously-spawned process / reader task."""
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()
        self._reader_task = None
        self._stderr_task = None
        # Fail any still-pending requests so callers don't hang.
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(
                    MCPTransportError("stdio MCP child died before responding")
                )
        self._pending.clear()
        self._proc = None

    async def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._dead = True
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
        await self._cleanup_dead_proc()

    # ── I/O loops ────────────────────────────────────────────────────

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        """Read JSON-RPC envelopes from stdout and dispatch by id."""
        assert proc.stdout is not None
        try:
            while True:
                try:
                    line = await proc.stdout.readline()
                except asyncio.LimitOverrunError as exc:
                    log.warning(
                        "MCP stdio: stdout line exceeded limit; skipping",
                        bytes=exc.consumed,
                    )
                    # Drain the oversized line so we re-sync.
                    await proc.stdout.readexactly(exc.consumed)
                    continue
                if not line:
                    # EOF — child closed stdout.
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    msg = json.loads(stripped)
                except json.JSONDecodeError:
                    log.warning(
                        "MCP stdio: skipping non-JSON line",
                        sample=stripped[:200].decode(errors="replace")
                        if isinstance(stripped, bytes)
                        else stripped[:200],
                    )
                    continue
                if not isinstance(msg, dict):
                    continue
                # Notifications (no id) are ignored for now; Phase 2.3
                # will hook into ``notifications/tools/list_changed``.
                msg_id = msg.get("id")
                if msg_id is None:
                    self._handle_notification(msg)
                    continue
                fut = self._pending.pop(int(msg_id), None)
                if fut is None or fut.done():
                    continue
                if "error" in msg:
                    err = msg["error"] or {}
                    fut.set_exception(
                        MCPTransportError(
                            f"MCP RPC error {err.get('code', '?')}: "
                            f"{err.get('message', str(err))}"
                        )
                    )
                else:
                    result = msg.get("result", {})
                    if not isinstance(result, dict):
                        result = {"result": result}
                    fut.set_result(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("MCP stdio reader crashed", error=str(exc))
        finally:
            self._dead = True
            # Wake up anyone still waiting.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(
                        MCPTransportError("stdio MCP child closed stdout")
                    )
            self._pending.clear()

    async def _stderr_loop(self, proc: asyncio.subprocess.Process) -> None:
        """Drain stderr, logging each line at debug level.

        Keeping stderr unread would eventually fill the pipe buffer and
        wedge the child. We log at debug rather than warning so a chatty
        child (e.g. ``npx`` printing progress) doesn't spam the FD log.
        """
        assert proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    log.debug("MCP stdio stderr", line=text, command=self.command)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _handle_notification(self, msg: dict[str, Any]) -> None:
        """Hook for incoming server-initiated notifications.

        ``notifications/progress`` is routed to whichever in-flight
        request advertised the matching ``progressToken``; everything
        else is forwarded to :attr:`on_notification` (the manager's
        bus hook). Errors in either handler are swallowed so a broken
        listener can't kill the reader loop.
        """
        method = str(msg.get("method") or "")
        if method:
            log.debug("MCP stdio notification", method=method, command=self.command)

        if method == "notifications/progress":
            params = msg.get("params") or {}
            if isinstance(params, dict):
                token = str(params.get("progressToken") or "")
                handler = self._progress_handlers.get(token) if token else None
                if handler is not None:
                    try:
                        handler(params)
                    except Exception:
                        log.warning(
                            "MCP stdio: progress handler raised",
                            command=self.command,
                            exc_info=True,
                        )
            return

        cb = self.on_notification
        if cb is not None:
            try:
                cb(msg)
            except Exception:
                log.warning(
                    "MCP stdio: on_notification handler raised",
                    command=self.command,
                    exc_info=True,
                )

    # ── public RPC ───────────────────────────────────────────────────

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        proc = await self._ensure_proc()
        if self._dead or proc.stdin is None or proc.stdin.is_closing():
            raise MCPTransportError("stdio MCP child is not writable")
        msg_id = next(self._id_counter)
        # Stamp a progressToken into params._meta when the caller wants
        # progress, then register the matching handler for the
        # lifetime of this call.
        progress_token: str | None = None
        out_params = params
        if on_progress is not None:
            out_params = dict(params or {})
            meta = dict(out_params.get("_meta") or {})
            progress_token = f"progress-{msg_id}"
            meta["progressToken"] = progress_token
            out_params["_meta"] = meta
            self._progress_handlers[progress_token] = on_progress
        envelope = _rpc_request(method, out_params, msg_id)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[msg_id] = fut
        try:
            data = (json.dumps(envelope, separators=(",", ":")) + "\n").encode("utf-8")
            async with self._write_lock:
                proc.stdin.write(data)
                await proc.stdin.drain()
        except Exception as exc:
            self._pending.pop(msg_id, None)
            if progress_token:
                self._progress_handlers.pop(progress_token, None)
            raise MCPTransportError(f"stdio write failed: {exc}") from exc
        try:
            return await fut
        finally:
            if progress_token:
                self._progress_handlers.pop(progress_token, None)

    async def notify(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        try:
            proc = await self._ensure_proc()
        except MCPTransportError:
            return  # best-effort
        if self._dead or proc.stdin is None or proc.stdin.is_closing():
            return
        envelope = _rpc_notification(method, params)
        try:
            data = (json.dumps(envelope, separators=(",", ":")) + "\n").encode("utf-8")
            async with self._write_lock:
                proc.stdin.write(data)
                await proc.stdin.drain()
        except Exception:
            log.debug("MCP stdio notify failed (best-effort)", method=method, exc_info=True)


# ── factory ─────────────────────────────────────────────────────────


def build_transport(record: dict[str, Any]) -> Transport:
    """Pick the right transport for a configured server record."""
    transport = str(record.get("transport") or "http").strip().lower()
    if transport == "stdio":
        return StdioTransport(record)
    return HttpTransport(record)
