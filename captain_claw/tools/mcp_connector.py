"""MCP tools for captain-claw agents.

In Phase 1 of the MCP-via-Flight-Deck design, all upstream MCP I/O lives
inside Flight Deck (see :mod:`captain_claw.flight_deck.mcp_manager`).
Agents do **not** open their own connections to MCP servers — instead
they enumerate FD's configured servers via ``/fd/mcp/agent/servers``,
fetch each server's tool catalogue from ``/fd/mcp/<name>/tools`` and
proxy each call through ``/fd/mcp/<name>/call``.

This module exposes:

* :class:`MCPProxyTool` — a captain-claw :class:`Tool` whose ``execute``
  forwards arguments to FD.
* :class:`MCPProxyConnector` — the per-server client wrapping the FD
  proxy endpoints.
* :func:`register_mcp_tools` — top-level helper that asks FD for its
  enabled server list and registers proxy tools for every advertised
  upstream tool.

When ``FD_URL`` is unset, :func:`register_mcp_tools` is a no-op. With
the new architecture, MCP servers are administered exclusively from
Flight Deck — there is no per-agent ``config.yaml`` ``mcp_servers``
list any more.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from captain_claw.fd_client import FDClient, is_under_flight_deck
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


# When the SSE event stream drops we reconnect with exponential
# backoff capped at this value. FD restarts and brief network blips are
# common in dev — keep the cap short so reconnects feel snappy.
_SSE_RECONNECT_INITIAL_SECONDS = 1.0
_SSE_RECONNECT_MAX_SECONDS = 30.0

# Per-process tracking of which proxy-tool names we registered for each
# upstream server, so we can ``unregister`` the right ones when an
# event tells us the server changed or went away. Keyed by server name
# → list of registered captain-claw tool names.
_registered_by_server: dict[str, list[str]] = {}


# ── proxy connector ─────────────────────────────────────────────────


class MCPProxyConnector:
    """Talks to a Flight-Deck-managed MCP server via FD's REST proxy.

    One instance per upstream server name. The connector owns a tiny
    :class:`FDClient` so multiple proxy tools on the same server share
    a connection pool.
    """

    def __init__(self, server_name: str, fd_client: FDClient | None = None) -> None:
        self.name = server_name
        self._fd = fd_client or FDClient()

    async def discover_tools(self) -> list[dict[str, Any]]:
        resp = await self._fd.get(f"/fd/mcp/{self.name}/tools")
        if resp.status_code != 200:
            raise RuntimeError(
                f"FD /fd/mcp/{self.name}/tools returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        body = resp.json()
        tools = body.get("tools", []) if isinstance(body, dict) else []
        return list(tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        resp = await self._fd.post(
            f"/fd/mcp/{self.name}/call",
            json={"tool": tool_name, "arguments": arguments},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"FD /fd/mcp/{self.name}/call returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        body = resp.json()
        result = (body or {}).get("result") if isinstance(body, dict) else None
        if not isinstance(result, dict):
            return json.dumps(body)

        content = result.get("content", [])
        is_error = result.get("isError", False)
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "image":
                    parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
                else:
                    parts.append(json.dumps(block))
            elif isinstance(block, str):
                parts.append(block)
        text = "\n".join(parts) if parts else json.dumps(result)
        if is_error:
            raise RuntimeError(f"MCP tool error: {text}")
        return text


# ── proxy tool ──────────────────────────────────────────────────────


class MCPProxyTool(Tool):
    """A captain-claw :class:`Tool` proxying execution to an upstream MCP tool."""

    def __init__(
        self,
        mcp_tool_name: str,
        mcp_description: str,
        mcp_input_schema: dict[str, Any],
        server_name: str,
        connector: MCPProxyConnector,
    ) -> None:
        # Anthropic restricts tool names to ^[a-zA-Z0-9_-]{1,128}$ so we
        # need to sanitise the upstream identifier.
        safe_name = mcp_tool_name.replace(".", "_").replace(" ", "_")
        safe_server = server_name.replace(".", "_").replace(" ", "_")
        self.name = f"mcp_{safe_server}_{safe_name}"
        # Use the upstream description verbatim.  An earlier version
        # prepended ``[MCP:<server>]`` for traceability, but the
        # ``MCP`` token in tool descriptions appeared to bias
        # gpt-5.3-codex into refusing to invoke them ("MCP execution is
        # blocked here").  The server is already encoded in the tool
        # name (``mcp_<server>_<tool>``), so the prefix added nothing.
        self.description = mcp_description or mcp_tool_name
        self.parameters = mcp_input_schema or {"type": "object", "properties": {}}
        self.timeout_seconds = 60.0
        self._mcp_tool_name = mcp_tool_name
        self._server_name = server_name
        self._connector = connector

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            declared = set((self.parameters.get("properties") or {}).keys())
            required = set(self.parameters.get("required") or [])
            clean_args: dict[str, Any] = {}
            for key, value in kwargs.items():
                if key not in declared:
                    continue
                if isinstance(value, Path):
                    value = str(value)
                # Drop "empty filler" values for *optional* fields.  Models
                # like gpt-5.3-codex defensively fill every declared
                # property — sending ``""`` or ``None`` for fields they
                # don't actually want to filter on.  Many MCP servers
                # validate optional strings with ``minLength: 1`` (so
                # ``""`` raises) or treat the literal value as a filter
                # (so ``"any"`` matches nothing).  Stripping these
                # makes the call equivalent to "field absent", which is
                # what the model actually meant.  Required fields are
                # forwarded as-is so upstream validation errors still
                # surface to the model.
                if key not in required:
                    if value is None:
                        continue
                    if isinstance(value, str) and value.strip() == "":
                        continue
                clean_args[key] = value
            text = await self._connector.call_tool(self._mcp_tool_name, clean_args)
            return ToolResult(success=True, content=text)
        except Exception as exc:
            log.error(
                "MCP tool execution failed",
                tool=self._mcp_tool_name,
                server=self._server_name,
                error=str(exc),
            )
            return ToolResult(success=False, error=str(exc))


# ── registration entrypoint ─────────────────────────────────────────


async def _register_one_server(
    registry: Any,
    server_name: str,
    fd: FDClient,
) -> list[str]:
    """Discover and register every tool advertised by one MCP server.

    Returns the captain-claw tool names that were registered. Errors
    are logged but never raised — a single broken upstream shouldn't
    take the whole agent down.
    """
    connector = MCPProxyConnector(server_name, fd_client=fd)
    try:
        mcp_tools = await connector.discover_tools()
    except Exception as exc:
        log.error(
            "Failed to fetch MCP tools from FD", server=server_name, error=str(exc)
        )
        return []
    registered: list[str] = []
    for tool_def in mcp_tools:
        if not isinstance(tool_def, dict):
            continue
        tool_name = str(tool_def.get("name") or "").strip()
        if not tool_name:
            continue
        proxy = MCPProxyTool(
            mcp_tool_name=tool_name,
            mcp_description=str(tool_def.get("description") or ""),
            mcp_input_schema=tool_def.get("inputSchema") or {},
            server_name=server_name,
            connector=connector,
        )
        registry.register(proxy)
        registered.append(proxy.name)
        log.info(
            "Registered MCP proxy tool",
            tool=proxy.name,
            mcp_tool=tool_name,
            server=server_name,
        )
    return registered


def _unregister_server(registry: Any, server_name: str) -> int:
    """Remove every captain-claw tool we previously registered for ``server_name``.

    Returns the number of tools dropped. Safe to call when no tools
    were ever registered for the server (returns 0).
    """
    names = _registered_by_server.pop(server_name, [])
    for tool_name in names:
        try:
            registry.unregister(tool_name)
        except Exception:
            log.debug(
                "registry.unregister raised; ignoring",
                tool=tool_name,
                exc_info=True,
            )
    if names:
        log.info(
            "Unregistered MCP proxy tools",
            server=server_name,
            count=len(names),
        )
    return len(names)


async def _refresh_server(registry: Any, server_name: str, fd: FDClient) -> None:
    """Drop and re-register every proxy tool for one server.

    Called when a ``server_added``, ``server_updated`` or
    ``tools_changed`` event arrives. Idempotent — safe to call when
    the server has no tools yet (just unregisters and registers an
    empty list).
    """
    _unregister_server(registry, server_name)
    registered = await _register_one_server(registry, server_name, fd)
    if registered:
        _registered_by_server[server_name] = registered


async def register_mcp_tools(registry: Any) -> list[str]:
    """Discover FD-managed MCP servers and register a proxy tool per upstream tool.

    Called once during agent boot. Returns the list of registered
    tool names (the same ones now in ``registry``).

    When the agent is not running under Flight Deck (``FD_URL`` unset),
    this returns immediately — MCP is exclusively a Flight Deck feature
    in the new architecture.
    """
    if not is_under_flight_deck():
        return []

    fd = FDClient()
    all_registered: list[str] = []
    try:
        resp = await fd.get("/fd/mcp/agent/servers")
        if resp.status_code != 200:
            log.warning(
                "FD /fd/mcp/agent/servers returned %s: %s",
                resp.status_code,
                resp.text[:200],
            )
            return []
        body = resp.json() if resp.content else {}
        servers = body.get("servers", []) if isinstance(body, dict) else []
    except Exception as exc:
        log.warning("Failed to enumerate MCP servers from FD: %s", exc)
        return []

    for srv in servers:
        if not isinstance(srv, dict):
            continue
        name = str(srv.get("name") or "").strip()
        if not name:
            continue
        registered = await _register_one_server(registry, name, fd)
        if registered:
            _registered_by_server[name] = registered
            all_registered.extend(registered)

    return all_registered


# ── hot-reload event subscriber (Phase 2.3) ─────────────────────────


async def _consume_event_stream(registry: Any, fd: FDClient) -> None:
    """One SSE connection. Returns when the stream ends (clean EOF).
    Raises on transport errors so the outer loop can reconnect.
    """
    async with fd.stream(
        "GET",
        "/fd/mcp/agent/events",
        headers={"Accept": "text/event-stream"},
        timeout=None,
    ) as response:
        if response.status_code != 200:
            # Drain enough to get a useful error line, then bail.
            preview = ""
            try:
                preview = (await response.aread()).decode(errors="replace")[:200]
            except Exception:
                pass
            raise RuntimeError(
                f"FD /fd/mcp/agent/events returned {response.status_code}: {preview}"
            )
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                log.debug("MCP event stream: skipping non-JSON line", line=payload[:200])
                continue
            await _handle_event(registry, fd, event)


async def _handle_event(registry: Any, fd: FDClient, event: dict[str, Any]) -> None:
    etype = str(event.get("type") or "")
    server = str(event.get("server") or "")
    if etype in ("ping", "hello"):
        return
    if not server:
        return
    if etype == "server_removed":
        _unregister_server(registry, server)
        return
    if etype in ("server_added", "server_updated", "tools_changed"):
        try:
            await _refresh_server(registry, server, fd)
        except Exception as exc:
            log.warning(
                "MCP event refresh failed",
                server=server,
                event_type=etype,
                error=str(exc),
            )
        return
    log.debug("MCP event stream: ignoring unknown event", event_type=etype)


async def watch_mcp_events(registry: Any) -> None:
    """Long-running task that keeps the agent's MCP tool list fresh.

    Subscribes to FD's SSE event stream and reconnects forever (with
    exponential backoff capped at 30s) so the agent picks up MCP
    changes hot — no restart required.

    No-op when the agent isn't running under Flight Deck.
    """
    if not is_under_flight_deck():
        return
    fd = FDClient()
    backoff = _SSE_RECONNECT_INITIAL_SECONDS
    while True:
        try:
            log.debug("MCP event stream connecting")
            await _consume_event_stream(registry, fd)
            # Clean EOF — server probably restarted.  Reset backoff so
            # the reconnect feels snappy.
            backoff = _SSE_RECONNECT_INITIAL_SECONDS
            log.debug("MCP event stream ended cleanly; reconnecting")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "MCP event stream error; will reconnect",
                error=str(exc),
                backoff_seconds=backoff,
            )
        # Sleep before reconnect.
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, _SSE_RECONNECT_MAX_SECONDS)
