"""Centralised MCP client running inside Flight Deck.

In Phase 1, all real MCP I/O happens here — agents proxy to FD instead
of opening their own connections. This means:

* OAuth client_credentials flows happen exactly once per server
  (regardless of how many agents are connected).
* ``tools/list`` is cached for a short window so the chat panel /
  agent boot path don't pay a discovery round-trip every time.
* Session state (``Mcp-Session-Id`` for HTTP, the long-lived child
  process for stdio) is owned by FD, so an agent can issue
  ``tools/call`` without redoing the handshake.

Phase 2 split the wire protocol into pluggable
:class:`~captain_claw.flight_deck.mcp_transport.Transport`
implementations so the manager itself is now transport-agnostic — it
just speaks JSON-RPC method/params and lets the transport worry about
HTTP+OAuth+SSE vs subprocess+NDJSON.

Concurrency model: one :class:`asyncio.Lock` per server keeps
initialisation / re-initialisation single-flighted; ``tools/call``
runs without holding the lock so a slow tool call doesn't block other
tools on the same server.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any, Callable

from captain_claw.flight_deck import mcp_client_oauth, mcp_elicitation, mcp_events, mcp_storage
from captain_claw.flight_deck.mcp_transport import (
    METHOD_NOT_FOUND,
    HttpTransport,
    MCPRequestError,
    MCPTransportError,
    Transport,
    build_transport,
)
from captain_claw.logging import get_logger

log = get_logger(__name__)


# Tools list is cached this long. Discovery is cheap once the server
# is initialised, but capping it at a few seconds means the chat-panel
# refresh button still feels responsive without burning API quota.
_TOOLS_TTL_SECONDS = 30.0

# Same token we send to the upstream MCP server identifying ourselves.
# (The MCP revision we offer is the transport's: ``Transport.protocol_version``.)
_CLIENT_NAME = "captain-claw-flight-deck"


class MCPServerError(RuntimeError):
    """Raised when an upstream MCP server returns a non-recoverable error."""


class _ServerState:
    """Per-server runtime state. Not exposed outside this module."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.transport: Transport = build_transport(record)
        self.lock = asyncio.Lock()
        self.initialized: bool = False
        self.tools: list[dict[str, Any]] = []
        self.tools_loaded_at: float = 0.0
        self.last_error: str | None = None
        # Wire the upstream notification hook so a stdio server's
        # ``notifications/tools/list_changed`` invalidates our cache and
        # fans out a ``tools_changed`` event to subscribed agents.
        # Bind ``self`` explicitly because the transport stores the
        # callback by reference.
        self.transport.on_notification = self._on_upstream_notification
        # Requests the server sends us (an elicitation: it asks the person
        # before it goes on), answered for whoever its tool calls are for.
        self.transport.on_request = self._on_upstream_request
        # Tool calls in flight → whom each is for (None: nobody we can ask).
        self.calls: dict[int, mcp_elicitation.Asker | None] = {}
        self.call_ids = itertools.count(1)

    def _on_upstream_notification(self, msg: dict[str, Any]) -> None:
        method = str(msg.get("method") or "")
        if method == "notifications/tools/list_changed":
            # Force a reload on the next list_tools() call and tell
            # subscribers there's something new worth fetching.
            self.tools = []
            self.tools_loaded_at = 0.0
            mcp_events.publish_tools_changed(self.name)

    async def _on_upstream_request(self, msg: dict[str, Any]) -> dict[str, Any]:
        method = str(msg.get("method") or "")
        if method == "elicitation/create":
            return await mcp_elicitation.handle(
                self.name, msg.get("params"), list(self.calls.values()))
        raise MCPRequestError(METHOD_NOT_FOUND, f"Method not found: {method}")

    @property
    def name(self) -> str:
        return self.record["name"]

    @property
    def last_status_code(self) -> int | None:
        """HTTP status of the last upstream response, or ``None`` for
        stdio (where status codes don't apply). Read by the manager to
        decide when a transient error is worth dropping the session
        for."""
        if isinstance(self.transport, HttpTransport):
            return self.transport.last_status_code
        return None

    def reset_session(self) -> None:
        """Drop any session/token state so the next call re-handshakes.

        For HTTP this clears the cached OAuth token + ``Mcp-Session-Id``.
        For stdio the transport keeps the child running — a dead child
        is auto-respawned by the transport itself, so there's nothing
        to reset here.
        """
        self.initialized = False
        if isinstance(self.transport, HttpTransport):
            self.transport.reset_session()


class MCPManager:
    """Singleton owning all upstream MCP connections within Flight Deck."""

    def __init__(self) -> None:
        self._states: dict[str, _ServerState] = {}
        self._states_lock = asyncio.Lock()

    # ── lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        # Close every transport so subprocess children get SIGTERM and
        # HTTP clients release their connection pools.
        states = list(self._states.values())
        self._states.clear()
        for state in states:
            try:
                await state.transport.close()
            except Exception:
                log.debug("transport close failed", server=state.name, exc_info=True)

    # ── state lookup ─────────────────────────────────────────────────

    async def _state_for(self, name: str) -> _ServerState:
        """Return the state for ``name``, refreshing record from disk."""
        async with self._states_lock:
            record = mcp_storage.get_server(name)
            if record is None:
                raise KeyError(f"MCP server '{name}' is not configured")
            existing = self._states.get(name)
            if existing is None or existing.record != record:
                # Config changed (or first hit) — rebuild state. Drop any
                # cached tokens/subprocess since secrets/command may have
                # rotated.
                if existing is not None:
                    try:
                        await existing.transport.close()
                    except Exception:
                        log.debug(
                            "transport close failed",
                            server=existing.name,
                            exc_info=True,
                        )
                self._states[name] = _ServerState(record)
            return self._states[name]

    async def forget_server(self, name: str) -> None:
        """Remove cached state for ``name``. Called on delete / config change.

        Closes the underlying transport so any spawned subprocess /
        connection pool is released before the state is discarded —
        otherwise a stdio child would linger after the user changed its
        command and a fresh state was built next call.
        """
        state = self._states.pop(name, None)
        if state is None:
            return
        try:
            await state.transport.close()
        except Exception:
            log.debug("transport close failed", server=name, exc_info=True)

    # ── high-level operations ────────────────────────────────────────

    async def _initialize(self, state: _ServerState) -> None:
        if state.initialized:
            return
        async with state.lock:
            if state.initialized:
                return
            try:
                result = await state.transport.request(
                    "initialize",
                    {
                        "protocolVersion": state.transport.protocol_version,
                        "capabilities": state.transport.client_capabilities(),
                        "clientInfo": {
                            "name": _CLIENT_NAME,
                            "version": "0.4.23",
                        },
                    },
                )
                # Best-effort initialized notification (no id).
                await state.transport.notify("notifications/initialized")
                state.initialized = True
                state.last_error = None
                log.info(
                    "MCP server initialized",
                    server=state.name,
                    transport=state.record.get("transport", "http"),
                    server_info=(
                        result.get("serverInfo")
                        if isinstance(result, dict)
                        else {}
                    ),
                )
            except MCPTransportError as exc:
                state.last_error = str(exc)
                state.initialized = False
                raise MCPServerError(str(exc)) from exc
            except Exception as exc:
                state.last_error = str(exc)
                state.initialized = False
                raise

    async def list_tools(
        self,
        name: str,
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        state = await self._state_for(name)
        if not state.record.get("enabled", True):
            return []
        await self._initialize(state)
        now = time.time()
        if (
            not force_refresh
            and state.tools
            and (now - state.tools_loaded_at) < _TOOLS_TTL_SECONDS
        ):
            return state.tools
        try:
            result = await state.transport.request("tools/list")
        except MCPTransportError as exc:
            raise MCPServerError(str(exc)) from exc
        tools = result.get("tools", []) if isinstance(result, dict) else []
        state.tools = list(tools)
        state.tools_loaded_at = now
        log.info("MCP tools discovered", server=state.name, count=len(tools))
        return state.tools

    async def call_tool(
        self,
        name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        asker: mcp_elicitation.Asker | None = None,
    ) -> dict[str, Any]:
        """Call *tool_name*. *asker* is the person the call is for: if the
        server asks something while it runs (an elicitation), the question
        goes to them on a card; without one, the server's questions are
        cancelled (nobody we may ask)."""
        state = await self._state_for(name)
        if not state.record.get("enabled", True):
            raise MCPServerError(f"MCP server '{name}' is disabled")
        await self._initialize(state)
        key = next(state.call_ids)
        state.calls[key] = asker
        try:
            result = await state.transport.request(
                "tools/call",
                {"name": tool_name, "arguments": arguments or {}},
                on_progress=on_progress,
            )
        except MCPTransportError as exc:
            # If the session went bad mid-flight, drop session+token and
            # let the next caller redo the handshake. Status code is
            # only meaningful for HTTP; stdio transports report None
            # and we leave their state alone (the transport itself
            # handles dead-child recovery).
            if state.last_status_code in (400, 401, 404):
                state.reset_session()
            raise MCPServerError(str(exc)) from exc
        finally:
            state.calls.pop(key, None)
            if asker is not None:
                await mcp_elicitation.call_ended(asker)
        return result if isinstance(result, dict) else {"result": result}

    async def probe_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Run an end-to-end probe against a transient (non-persisted)
        server record. Never raises; never mutates ``self._states``.

        Always closes the transport before returning so a probe of a
        stdio record doesn't leak a child process.
        """
        if mcp_client_oauth.is_oauth(record) and not mcp_client_oauth.public_status(record)["connected"]:
            # Nothing to call with yet — check that sign-in can work instead.
            return await _probe_sign_in(record)
        try:
            state = _ServerState(record)
        except Exception as exc:
            return {"ok": False, "error": f"invalid record: {exc}"}
        try:
            await self._initialize(state)
            result = await state.transport.request("tools/list")
            tools = result.get("tools", []) if isinstance(result, dict) else []
            return {
                "ok": True,
                "tools_count": len(tools),
                "tool_names": [t.get("name", "") for t in tools],
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "status_code": state.last_status_code,
            }
        finally:
            try:
                await state.transport.close()
            except Exception:
                log.debug("probe transport close failed", exc_info=True)

    async def test_server(self, name: str) -> dict[str, Any]:
        """Run an end-to-end probe of a configured server.

        Returns a dict with: ``ok``, optional ``tools_count``, ``error``,
        ``status_code``. Never raises — used by the UI to render
        per-server health.
        """
        try:
            state = await self._state_for(name)
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        if mcp_client_oauth.is_oauth(state.record) and not mcp_client_oauth.public_status(state.record)["connected"]:
            return await _probe_sign_in(state.record)
        # Force a clean run so an old cached failure doesn't poison the
        # test. For HTTP this resets the session/token; for stdio the
        # existing child stays (it's still healthy if it responded last
        # time, and a dead child auto-respawns).
        state.reset_session()
        state.tools = []
        state.tools_loaded_at = 0.0
        try:
            tools = await self.list_tools(name, force_refresh=True)
            return {
                "ok": True,
                "tools_count": len(tools),
                "tool_names": [t.get("name", "") for t in tools],
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "status_code": state.last_status_code,
            }

    # ── status views (UI) ────────────────────────────────────────────

    def status_snapshot(self) -> list[dict[str, Any]]:
        """Cheap, non-network view of every configured server."""
        records = mcp_storage.list_servers_public()
        out: list[dict[str, Any]] = []
        for rec in records:
            state = self._states.get(rec["name"])
            entry = {
                **rec,
                "initialized": bool(state and state.initialized),
                "tools_count": len(state.tools) if state else 0,
                "last_error": state.last_error if state else None,
            }
            if mcp_client_oauth.is_oauth(rec):
                entry["oauth"] = mcp_client_oauth.public_status(rec)
            out.append(entry)
        return out


async def _probe_sign_in(record: dict[str, Any]) -> dict[str, Any]:
    """Probe for an OAuth server nobody has signed in to: run discovery and
    say whether Flight Deck can register itself there. Never raises."""
    try:
        found = await mcp_client_oauth.discover(
            str(record.get("url") or ""), headers=record.get("headers") or {},
        )
    except mcp_client_oauth.MCPOAuthError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"discovery failed: {exc}"}
    can_register = bool(found["registration_endpoint"]) or bool(record.get("client_id"))
    return {
        "ok": can_register,
        "needs_sign_in": True,
        "issuer": found["issuer"],
        "registration_supported": bool(found["registration_endpoint"]),
        "scope": found["scope"],
        **({} if can_register else {
            "error": (
                f"{found['issuer']} has no dynamic client registration — enter a "
                "client ID registered with it by hand"
            ),
        }),
    }


# ── module-level singleton ──────────────────────────────────────────


_manager: MCPManager | None = None


def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager


async def shutdown() -> None:
    global _manager
    if _manager is not None:
        await _manager.close()
        _manager = None
