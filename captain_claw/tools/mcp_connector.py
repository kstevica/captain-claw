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
import re
import secrets
from pathlib import Path
from typing import Any

from captain_claw import mail_authority
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

    def __init__(self, server_name: str, fd_client: FDClient | None = None,
                 calls_client: FDClient | None = None) -> None:
        self.name = server_name
        self._fd = fd_client or FDClient()
        # Tool calls are followed in requests of up to ~30 s on Flight
        # Deck's side (a call may wait for the person), so they get their
        # own, longer timeout.
        self._calls = calls_client or FDClient(timeout=_CALL_HTTP_TIMEOUT)

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

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], *, agent: Any = None,
                        session_id: str = "", stream: Any = None) -> str:
        """Call *tool_name* through Flight Deck and return its text. The
        call is for the person this turn is for: if the server asks them
        something first, Flight Deck puts it on their card and this waits
        (saying so on *stream*); stopping the turn cancels the call."""
        body = await self._call_followed(tool_name, arguments, agent=agent, session_id=session_id,
                                         stream=stream)
        return _result_text(body)

    async def _call_once(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """One request for the whole call: a Flight Deck from before
        followed calls."""
        resp = await self._calls.post(
            f"/fd/mcp/{self.name}/call",
            json={"tool": tool_name, "arguments": arguments},
        )
        return _body_or_raise(resp, f"/fd/mcp/{self.name}/call")

    async def _call_followed(self, tool_name: str, arguments: dict[str, Any], *, agent: Any,
                             session_id: str, stream: Any) -> dict[str, Any]:
        from captain_claw import speaker

        headers: dict[str, str] = {}
        params: dict[str, str] = {}
        nobody = False
        try:
            headers = speaker.grant_headers()
            params = speaker.grant_params()
        except speaker.SpeakerGrantMissing:
            # A member's turn without its grant: the call runs, but a
            # question from the server can't reach them, so it's cancelled.
            nobody = True
        call_id = "xm_" + secrets.token_hex(8)
        payload = {"server": self.name, "tool": tool_name, "arguments": arguments,
                   "origin": _turn_origin(agent), "session_id": session_id or "",
                   "nobody": nobody, "wait": 20, "call_id": call_id}
        finished = False
        try:
            resp = await self._calls.post("/fd/mcp/agent/calls", json=payload, headers=headers, params=params)
            if _route_missing(resp):
                finished = True
                return await self._call_once(tool_name, arguments)
            told = False
            while True:
                body = _body_or_raise(resp, "/fd/mcp/agent/calls")
                if not body.get("pending"):
                    finished = True
                    return body
                if body.get("asking") and not told and callable(stream):
                    told = True
                    try:
                        stream(WAITING_NOTE.format(server=self.name))
                    except Exception:  # noqa: BLE001
                        pass
                resp = await self._calls.post("/fd/mcp/agent/calls/follow", json={"id": call_id, "wait": 25})
        finally:
            if not finished:  # the turn was stopped (or the line failed): stop the call
                try:
                    await self._calls.post("/fd/mcp/agent/calls/cancel", json={"id": call_id})
                except BaseException:  # noqa: BLE001 — best effort while unwinding
                    pass


_CALL_HTTP_TIMEOUT = 45.0  # each request waits at most 30 s on Flight Deck's side
WAITING_NOTE = "Waiting for your answer in Flight Deck: {server} asks you before it goes on…\n"


def _turn_origin(agent: Any) -> str:
    origin = getattr(agent, "_turn_source", None)
    return str(origin[0]) if isinstance(origin, tuple) and origin else ""


def _route_missing(resp: Any) -> bool:
    """A Flight Deck from before followed calls doesn't have the route."""
    if resp.status_code not in (404, 405):
        return False
    try:
        detail = resp.json().get("detail")
    except Exception:  # noqa: BLE001
        return True
    return detail in ("Not Found", "Method Not Allowed")


def _body_or_raise(resp: Any, path: str) -> dict[str, Any]:
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail") or resp.text
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise RuntimeError(f"FD {path} returned {resp.status_code}: {str(detail)[:300]}")
    body = resp.json()
    return body if isinstance(body, dict) else {"result": body}


def _result_text(body: dict[str, Any]) -> str:
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


# Gmail-like MCP proxy tools that write mail (part of the automated-turn guard).
_MCP_MAIL_VERB = re.compile(
    r"^(create_draft|update_draft|send_draft|send_message|send_email|reply|reply_all|forward"
    r"|compose\w*|draft(?!s?_(?:list|get|read|search|fetch|find|count|query)(?:_|$))\w*)$",
    re.I,
)
_MCP_MAIL_HINT = re.compile(r"mail|gmail|outlook", re.I)
# Upstream names vary ("gmail_send_email", "sendEmail", "users.drafts.create",
# "create_gmail_draft", Microsoft Graph's "createReply" / "me.sendMail"):
# normalize to snake_case, drop a leading service prefix, and also accept a
# verb followed by more words. Read actions (list_drafts, get_draft …) never
# start with a write verb. Graph's createReply / createReplyAll /
# createForward and create_message (POST /me/messages) all create drafts.
_MCP_SERVICE_PREFIX = re.compile(
    r"^(?:(?:gmail|google|mail|email|outlook|ms|microsoft|o365|graph|users|user|me)_)+"
)
_MCP_MAIL_VERB_LOOSE = re.compile(
    r"^(?:send(?:_\w+)?|reply(?:_\w+)?|forward(?:_\w+)?|compose\w*"
    r"|draft(?!s?_(?:list|get|read|search|fetch|find|count|query)(?:_|$))\w*"
    r"|(?:create|update|save|upsert)_\w*draft\w*"
    r"|create_(?:reply(?:_all)?|forward|message|mail|email)(?:_draft)?"
    r"|create_\w*_(?:reply(?:_all)?|forward)"
    r"|(?:schedule|batch|bulk)_send\w*"
    r"|(?:drafts?|messages?)_(?:create|update|send|reply(?:_all)?|forward"
    r"|create_(?:reply(?:_all)?|forward)))$"
)


def _mcp_mail_write_name(name: str) -> bool:
    """Is this upstream MCP tool name a mail-writing verb?"""
    if _MCP_MAIL_VERB.match(name or ""):
        return True
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name or "")
    snake = re.sub(r"[^A-Za-z0-9]+", "_", snake).strip("_").lower()
    snake = _MCP_SERVICE_PREFIX.sub("", snake)
    return bool(_MCP_MAIL_VERB.match(snake) or _MCP_MAIL_VERB_LOOSE.match(snake))


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
        # Flight Deck bounds the call itself (it stops one that goes quiet,
        # mcp_calls.QUIET_LIMIT_S), and a call may wait for the person's
        # answer on a card (mcp_elicitation.WAIT_S): this is only a backstop.
        self.timeout_seconds = 3_700.0
        self._mcp_tool_name = mcp_tool_name
        self._server_name = server_name
        self._connector = connector

    async def execute(self, **kwargs: Any) -> ToolResult:
        # A Gmail-like MCP write in an automated turn obeys the same guard as
        # google_mail (recipients unknown → only "allow" / any-scope intent).
        if _mcp_mail_write_name(self._mcp_tool_name or "") and _MCP_MAIL_HINT.search(
            f"{self._server_name} {self.description}"
        ):
            _r = mail_authority.check_mail_write("mcp_mail", None)
            if _r:
                log.info(
                    "MCP mail write refused (automated turn)",
                    tool=self._mcp_tool_name, server=self._server_name,
                    kind=mail_authority.current().kind,
                )
                return ToolResult(success=False, error=_r)
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
            text = await self._connector.call_tool(
                self._mcp_tool_name, clean_args, agent=kwargs.get("_agent"),
                session_id=str(kwargs.get("_session_id") or ""), stream=kwargs.get("_stream_callback"))
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
