"""HTTP routes for Flight Deck's centrally-managed MCP servers.

Two audiences hit these endpoints:

1. **Admin UI** (``/fd/mcp/servers``, ``/fd/mcp/servers/<name>``,
   ``/fd/mcp/servers/<name>/test``) — gated by the normal Flight Deck
   user auth. Returns secrets in masked form only.

2. **Captain-claw agents** (``/fd/mcp/<server>/tools``,
   ``/fd/mcp/<server>/call``) — gated by the same loopback / shared-secret
   pattern Codex / Google OAuth use. Agents proxy here instead of dialing
   the upstream MCP server themselves; that gives FD a single chokepoint
   for OAuth, rate-limiting and observability.

Phase 1 keeps the schema flat: name + URL + optional OAuth client
credentials + optional headers. Anything more (per-tenant scoping,
ACLs, audit log) is intentionally deferred.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from captain_claw.flight_deck import mcp_events, mcp_storage
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.mcp_manager import MCPServerError, get_manager
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/mcp", tags=["mcp"])


# ── agent auth (mirrors codex_oauth_routes) ─────────────────────────


def _agent_shared_secret() -> str:
    return os.environ.get("FD_AGENT_SHARED_SECRET", "").strip()


def _authorize_agent_call(request: Request) -> str:
    """Allow the call if the request comes from loopback OR carries the
    matching ``X-Agent-Secret`` header. Used for the proxy endpoints
    captain-claw agents call.

    Returns the calling agent's slug (from ``X-Agent-Slug``) when
    provided, or ``""`` when the caller didn't identify itself. The
    slug is used for per-agent ACL filtering downstream — an empty
    slug means "anonymous" and is only allowed access to servers with
    no allowlist configured.
    """
    secret = _agent_shared_secret()
    authorized = False
    if secret:
        provided = request.headers.get("X-Agent-Secret", "")
        if provided and secrets.compare_digest(provided, secret):
            authorized = True
    if not authorized:
        client_host = request.client.host if request.client else ""
        if client_host in ("127.0.0.1", "::1", "localhost"):
            authorized = True
    if not authorized:
        raise HTTPException(status_code=401, detail="Unauthorized agent call")
    return (request.headers.get("X-Agent-Slug") or "").strip()


# ── admin: list / add / remove servers ──────────────────────────────


@router.get("/servers")
async def list_servers(_user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Return every configured server with secrets masked, plus runtime
    status (initialised? tool count? last error?)."""
    manager = get_manager()
    return {"servers": manager.status_snapshot()}


@router.post("/servers")
async def add_or_update_server(
    payload: dict = Body(...),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Insert or update a server by ``name``. Existing client_secret is
    preserved when the new payload has an empty / masked client_secret."""
    name = str(payload.get("name") or "").strip()
    transport = str(payload.get("transport") or "http").strip().lower() or "http"
    url = str(payload.get("url") or "").strip()
    command = str(payload.get("command") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if transport == "http" and not url:
        raise HTTPException(status_code=400, detail="url is required for http transport")
    if transport == "stdio" and not command:
        raise HTTPException(
            status_code=400, detail="command is required for stdio transport"
        )

    incoming_secret = str(payload.get("client_secret") or "")
    # The list view returns a placeholder of "•••..." — preserve the
    # stored secret if the user re-submits without changing that field.
    looks_masked = incoming_secret and set(incoming_secret) <= {"•", "*"}
    if looks_masked or not incoming_secret:
        existing = mcp_storage.get_server(name)
        if existing is not None:
            incoming_secret = existing.get("client_secret", "")

    record = {
        "name": name,
        "transport": transport,
        "url": url,
        "command": command,
        "args": payload.get("args") or [],
        "env": payload.get("env") or {},
        "client_id": str(payload.get("client_id") or ""),
        "client_secret": incoming_secret,
        "token_endpoint": str(payload.get("token_endpoint") or ""),
        "headers": payload.get("headers") or {},
        "enabled": bool(payload.get("enabled", True)),
        "allowed_agents": payload.get("allowed_agents") or [],
    }
    existed_before = mcp_storage.get_server(name) is not None
    saved = await mcp_storage.upsert_server(record)
    # Drop any cached upstream session so the new config takes effect.
    # Awaited so a running stdio child gets terminated before we return —
    # otherwise the next call would race a brand-new state against the
    # still-shutting-down old one.
    await get_manager().forget_server(name)
    # Tell subscribed agents to refresh.  We publish *after* both
    # storage and manager state have settled so the subscriber's
    # follow-up GET observes the new world.
    if existed_before:
        mcp_events.publish_server_updated(name)
    else:
        mcp_events.publish_server_added(name)
    return {"ok": True, "server": mcp_storage._public_view(saved)}


@router.delete("/servers/{name}")
async def remove_server(
    name: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    deleted = await mcp_storage.delete_server(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    await get_manager().forget_server(name)
    mcp_events.publish_server_removed(name)
    return {"ok": True}


@router.post("/servers/{name}/test")
async def test_server(
    name: str,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Probe a configured server: re-init, list tools, surface any error."""
    if mcp_storage.get_server(name) is None:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    return await get_manager().test_server(name)


@router.post("/probe")
async def probe_transient(
    payload: dict = Body(...),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Probe an ad-hoc server config without persisting it.

    Used by the "Test connection" button before the user actually saves
    a server. The payload accepts the same shape as ``POST /servers``
    (``name``, ``url``, ``client_id``, ``client_secret``, ``token_endpoint``,
    ``headers``). When the submitted ``client_secret`` is empty/masked
    and a server with the same ``name`` already exists, the saved
    secret is used so users can re-test without re-entering it.
    """
    transport = str(payload.get("transport") or "http").strip().lower() or "http"
    url = str(payload.get("url") or "").strip()
    command = str(payload.get("command") or "").strip()
    if transport == "http" and not url:
        raise HTTPException(status_code=400, detail="url is required for http transport")
    if transport == "stdio" and not command:
        raise HTTPException(
            status_code=400, detail="command is required for stdio transport"
        )
    name = str(payload.get("name") or "_probe").strip() or "_probe"

    incoming_secret = str(payload.get("client_secret") or "")
    looks_masked = incoming_secret and set(incoming_secret) <= {"•", "*"}
    if (looks_masked or not incoming_secret) and name and name != "_probe":
        existing = mcp_storage.get_server(name)
        if existing is not None:
            incoming_secret = existing.get("client_secret", "")

    record = {
        "name": name,
        "transport": transport,
        "url": url,
        "command": command,
        "args": payload.get("args") or [],
        "env": payload.get("env") or {},
        "client_id": str(payload.get("client_id") or ""),
        "client_secret": incoming_secret,
        "token_endpoint": str(payload.get("token_endpoint") or ""),
        "headers": payload.get("headers") or {},
        "enabled": True,
        "allowed_agents": payload.get("allowed_agents") or [],
    }
    return await get_manager().probe_record(record)


# ── shared SSE helpers (used by both streaming routes below) ────────


# How often to send a comment-only "ping" through an idle SSE
# connection. Many proxies (and our own dev nginx) drop a TCP socket
# after ~60s of no bytes, so we send at half that.
_SSE_PING_INTERVAL_SECONDS = 25.0


def _sse_format(event: dict[str, Any]) -> bytes:
    """Encode a payload dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode("utf-8")


# ── agent-facing: discovery + tools/list + tools/call passthrough ───


@router.get("/agent/servers")
async def agent_list_servers(request: Request) -> dict[str, Any]:
    """Return enabled server names for an agent to enumerate.

    Filtered by the calling agent's slug against each server's
    ``allowed_agents`` list — servers with an empty allowlist are
    visible to every agent (Phase 1 behaviour preserved for
    backwards-compat).

    Secrets are *not* included — agents never need them, since every
    upstream call routes through this proxy. Use ``/fd/mcp/<name>/tools``
    for per-server discovery.
    """
    agent_slug = _authorize_agent_call(request)
    records = mcp_storage.load_servers()
    out: list[dict[str, Any]] = []
    for rec in records:
        if not rec.get("enabled", True):
            continue
        if not mcp_storage.is_agent_allowed(rec, agent_slug):
            continue
        out.append({"name": rec["name"]})
    return {"servers": out}


@router.get("/{name}/tools")
async def proxy_tools_list(
    name: str,
    request: Request,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return the cached ``tools/list`` for ``name``. Agents hit this
    instead of running their own MCP discovery, so OAuth + session is
    single-flighted across the fleet."""
    agent_slug = _authorize_agent_call(request)
    record = mcp_storage.get_server(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    if not mcp_storage.is_agent_allowed(record, agent_slug):
        # Match the 404 shape of "not configured" so disallowed servers
        # are indistinguishable from non-existent ones — don't leak the
        # existence of restricted servers to agents that can't use them.
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    try:
        tools = await get_manager().list_tools(name, force_refresh=refresh)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tools": tools}


@router.post("/{name}/call")
async def proxy_tool_call(
    name: str,
    request: Request,
    payload: dict = Body(...),
) -> dict[str, Any]:
    agent_slug = _authorize_agent_call(request)
    record = mcp_storage.get_server(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    if not mcp_storage.is_agent_allowed(record, agent_slug):
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="payload.tool is required")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")
    try:
        result = await get_manager().call_tool(name, tool_name, arguments)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tool": tool_name, "result": result}


# ── user-facing: tool list + call for the agent-app runtime ─────────


@router.get("/{name}/user_tools")
async def user_tools_list(
    name: str,
    refresh: bool = False,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """User-authed counterpart of ``/{name}/tools``. Used by the
    Flight-Deck app runtime so the browser can introspect a server
    without holding an agent shared secret."""
    record = mcp_storage.get_server(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    try:
        tools = await get_manager().list_tools(name, force_refresh=refresh)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tools": tools}


@router.post("/{name}/user_call")
async def user_tool_call(
    name: str,
    payload: dict = Body(...),
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """User-authed counterpart of ``/{name}/call``. Lets the app
    runtime in the browser invoke MCP tools as the logged-in user
    (no agent slug, no shared secret)."""
    record = mcp_storage.get_server(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="payload.tool is required")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")
    try:
        result = await get_manager().call_tool(name, tool_name, arguments)
    except MCPServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"server": name, "tool": tool_name, "result": result}


# ── agent-facing: streaming tool call (Phase 2.4) ───────────────────


@router.post("/{name}/call_stream")
async def proxy_tool_call_stream(
    name: str,
    request: Request,
    payload: dict = Body(...),
) -> StreamingResponse:
    """Streaming variant of ``/{name}/call``.

    Returns an SSE stream of two event flavours:

    * ``{"type": "progress", "params": {...}}`` — one frame per
      ``notifications/progress`` from the upstream MCP server.
    * ``{"type": "result",   "result": {...}}`` — exactly one final
      frame with the same ``result`` shape ``/{name}/call`` returns.
    * ``{"type": "error",    "error": "..."}`` — sent in place of
      ``result`` when the upstream call fails.

    Callers that don't care about progress should keep using
    ``/{name}/call``; this endpoint exists for UIs that want to
    surface "tool is running, 30%…" indicators while the model
    waits.
    """
    agent_slug = _authorize_agent_call(request)
    record = mcp_storage.get_server(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    if not mcp_storage.is_agent_allowed(record, agent_slug):
        raise HTTPException(status_code=404, detail=f"No MCP server named '{name}'")
    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    if not tool_name:
        raise HTTPException(status_code=400, detail="payload.tool is required")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")

    # Bridge the synchronous ``on_progress`` callback (called from the
    # transport reader) into an asyncio.Queue we can drain from the
    # streaming response. Using a queue rather than wiring directly
    # into the StreamingResponse generator keeps the transport's
    # callback invariants simple — it never has to await anything.
    progress_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
    sentinel: dict[str, Any] = {"_done": True}

    def _on_progress(params: dict[str, Any]) -> None:
        try:
            progress_queue.put_nowait(params)
        except asyncio.QueueFull:
            log.warning(
                "MCP call_stream: progress queue full, dropping frame",
                server=name,
                tool=tool_name,
            )

    async def event_iterator():
        # Run the upstream call as a background task so we can pull
        # progress frames off the queue while it's in flight.
        manager = get_manager()
        call_task = asyncio.create_task(
            manager.call_tool(
                name, tool_name, arguments, on_progress=_on_progress
            ),
            name=f"mcp-call[{name}/{tool_name}]",
        )
        # Wake the queue when the call finishes so we can break.
        call_task.add_done_callback(
            lambda _t: progress_queue.put_nowait(sentinel)
        )
        try:
            while True:
                if await request.is_disconnected():
                    call_task.cancel()
                    break
                try:
                    item = await asyncio.wait_for(
                        progress_queue.get(), timeout=_SSE_PING_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    yield _sse_format({"type": "ping"})
                    continue
                if item is sentinel:
                    break
                yield _sse_format({"type": "progress", "params": item})
            # Drain whatever the call returned (or its error).
            try:
                result = await call_task
                yield _sse_format(
                    {"type": "result", "server": name, "tool": tool_name, "result": result}
                )
            except asyncio.CancelledError:
                # Caller disconnected; nothing more to emit.
                return
            except MCPServerError as exc:
                yield _sse_format({"type": "error", "error": str(exc)})
            except Exception as exc:
                yield _sse_format({"type": "error", "error": str(exc)})
        finally:
            if not call_task.done():
                call_task.cancel()

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── agent-facing: hot-push event stream (Phase 2.3) ─────────────────


@router.get("/agent/events")
async def agent_event_stream(request: Request) -> StreamingResponse:
    """SSE stream of MCP state changes filtered by the caller's allowlist.

    Captain-claw agents subscribe on boot and on each event they
    receive they re-run :func:`register_mcp_tools` so the model sees
    fresh tool catalogues without a restart.

    Events are only delivered if the calling agent is allowed to use
    the affected server (see
    :func:`mcp_storage.is_agent_allowed`). That keeps restricted-server
    existence opaque to disallowed agents — same rule as the rest of
    the agent-facing surface.

    The connection is kept alive with a periodic ``ping`` event so it
    survives proxies / NATs that drop idle TCP. The client should
    treat ``ping`` as a no-op.
    """
    agent_slug = _authorize_agent_call(request)
    bus = mcp_events.get_event_bus()
    queue = await bus.subscribe()

    async def event_iterator():
        try:
            # Send an initial hello so the client knows the connection
            # is live before any real event arrives.
            yield _sse_format({"type": "hello", "agent": agent_slug})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=_SSE_PING_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    # No real event in the window; emit a ping to keep
                    # intermediaries from closing the socket.
                    yield _sse_format({"type": "ping"})
                    continue
                # Filter per-server events by the caller's allowlist.
                # Pings / hellos never have a ``server`` field and are
                # always forwarded.
                server_name = event.get("server")
                if server_name:
                    record = mcp_storage.get_server(str(server_name))
                    if record is not None and not mcp_storage.is_agent_allowed(
                        record, agent_slug
                    ):
                        continue
                    # ``server_removed`` fires *after* deletion, so
                    # ``record is None`` is normal there — forward
                    # anyway so the agent can drop its proxies.
                yield _sse_format(event)
        finally:
            await bus.unsubscribe(queue)

    return StreamingResponse(
        event_iterator(),
        media_type="text/event-stream",
        headers={
            # Prevent buffering by intermediaries (nginx in particular).
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
