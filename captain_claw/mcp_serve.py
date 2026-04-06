"""MCP Server — expose Captain Claw as an MCP-compatible server.

Runs over stdio using JSON-RPC 2.0, following the Model Context Protocol
(MCP) specification.  External clients (Claude Desktop, other MCP clients)
can discover and call tools to interact with Captain Claw sessions.

Usage:
    captain-claw-mcp              # stdio mode (for Claude Desktop config)
    python -m captain_claw.mcp_serve
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast for stdio handshake
# ---------------------------------------------------------------------------

_agent_instance: Any = None
_session_manager: Any = None


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# MCP protocol constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "captain-claw"
SERVER_VERSION = "0.4.16"

# ---------------------------------------------------------------------------
# Tool definitions (MCP format)
# ---------------------------------------------------------------------------

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "sessions_list",
        "description": (
            "List recent Captain Claw conversation sessions. "
            "Returns session IDs, names, timestamps, and message counts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of sessions to return (default 20).",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "session_get",
        "description": (
            "Get details about a specific Captain Claw session by ID or name."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID or name to look up.",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "messages_read",
        "description": (
            "Read messages from a Captain Claw session. "
            "Returns the conversation history with roles, content, "
            "tool calls, and timestamps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID or name.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return (default 50, from most recent).",
                    "default": 50,
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip this many messages from the end (for pagination).",
                    "default": 0,
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "message_send",
        "description": (
            "Send a message to Captain Claw and get a response. "
            "The agent will process the message using its full tool suite "
            "(web search, file ops, code execution, etc.) and return the result. "
            "Optionally target a specific session to continue a conversation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to Captain Claw.",
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Optional session ID to continue. "
                        "If omitted, uses the agent's current session."
                    ),
                },
            },
            "required": ["message"],
        },
    },
    {
        "name": "session_create",
        "description": "Create a new Captain Claw conversation session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new session.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "agent_status",
        "description": (
            "Get the current Captain Claw agent status: "
            "active session, model, provider, token usage, and capabilities."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _ensure_session_manager():
    """Lazily initialize the session manager."""
    global _session_manager
    if _session_manager is None:
        from captain_claw.session import get_session_manager
        _session_manager = get_session_manager()
    return _session_manager


async def _ensure_agent():
    """Lazily initialize the Agent instance."""
    global _agent_instance
    if _agent_instance is None:
        from captain_claw.agent import Agent
        _agent_instance = Agent()
        await _agent_instance.initialize()
    return _agent_instance


async def _handle_sessions_list(args: dict[str, Any]) -> list[dict[str, Any]]:
    sm = await _ensure_session_manager()
    limit = int(args.get("limit", 20))
    sessions = await sm.list_sessions(limit=limit)
    return [
        {
            "id": s.id,
            "name": s.name,
            "message_count": len(s.messages),
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in sessions
    ]


async def _handle_session_get(args: dict[str, Any]) -> dict[str, Any]:
    sm = await _ensure_session_manager()
    session_id = args.get("session_id", "")
    if not session_id:
        return {"error": "session_id is required"}

    session = await sm.select_session(session_id)
    if not session:
        return {"error": f"Session not found: {session_id}"}

    # Summarize roles
    role_counts: dict[str, int] = {}
    for msg in session.messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "id": session.id,
        "name": session.name,
        "message_count": len(session.messages),
        "role_counts": role_counts,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "metadata": session.metadata,
    }


async def _handle_messages_read(args: dict[str, Any]) -> list[dict[str, Any]]:
    sm = await _ensure_session_manager()
    session_id = args.get("session_id", "")
    if not session_id:
        return [{"error": "session_id is required"}]

    session = await sm.select_session(session_id)
    if not session:
        return [{"error": f"Session not found: {session_id}"}]

    limit = int(args.get("limit", 50))
    offset = int(args.get("offset", 0))

    messages = session.messages
    if offset > 0:
        messages = messages[:-offset] if offset < len(messages) else []
    messages = messages[-limit:] if limit < len(messages) else messages

    result = []
    for msg in messages:
        entry: dict[str, Any] = {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
        }
        if msg.get("timestamp"):
            entry["timestamp"] = msg["timestamp"]
        if msg.get("tool_name"):
            entry["tool_name"] = msg["tool_name"]
        if msg.get("tool_calls"):
            entry["tool_calls"] = msg["tool_calls"]
        if msg.get("message_id"):
            entry["message_id"] = msg["message_id"]
        result.append(entry)

    return result


async def _handle_message_send(args: dict[str, Any]) -> dict[str, Any]:
    message = args.get("message", "").strip()
    if not message:
        return {"error": "message is required"}

    agent = await _ensure_agent()

    # Optionally switch session
    target_session = args.get("session_id", "")
    if target_session and agent.session and agent.session.id != target_session:
        sm = await _ensure_session_manager()
        session = await sm.select_session(target_session)
        if session:
            agent.session = session
        else:
            return {"error": f"Session not found: {target_session}"}

    response = await agent.complete(message)
    return {
        "response": response,
        "session_id": agent.session.id if agent.session else None,
        "usage": agent.last_usage,
    }


async def _handle_session_create(args: dict[str, Any]) -> dict[str, Any]:
    name = args.get("name", "").strip()
    if not name:
        return {"error": "name is required"}

    sm = await _ensure_session_manager()
    from captain_claw.session import Session
    session = Session(
        id=uuid.uuid4().hex[:12],
        name=name,
    )
    await sm.save_session(session)
    return {
        "id": session.id,
        "name": session.name,
        "created_at": session.created_at,
    }


async def _handle_agent_status(args: dict[str, Any]) -> dict[str, Any]:
    from captain_claw.config import get_config
    cfg = get_config()

    status: dict[str, Any] = {
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "model": cfg.model.model,
        "provider": cfg.model.provider,
    }

    if _agent_instance is not None:
        status["initialized"] = _agent_instance._initialized
        if _agent_instance.session:
            status["active_session"] = {
                "id": _agent_instance.session.id,
                "name": _agent_instance.session.name,
                "messages": len(_agent_instance.session.messages),
            }
        status["total_usage"] = _agent_instance.total_usage
    else:
        status["initialized"] = False

    # List enabled tools
    status["tools_enabled"] = cfg.tools.enabled[:20]  # Cap for brevity
    status["tool_count"] = len(cfg.tools.enabled)

    return status


# Tool dispatch table
_TOOL_HANDLERS: dict[str, Any] = {
    "sessions_list": _handle_sessions_list,
    "session_get": _handle_session_get,
    "messages_read": _handle_messages_read,
    "message_send": _handle_message_send,
    "session_create": _handle_session_create,
    "agent_status": _handle_agent_status,
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 helpers
# ---------------------------------------------------------------------------

def _jsonrpc_result(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


# ---------------------------------------------------------------------------
# MCP request handlers
# ---------------------------------------------------------------------------

async def handle_initialize(id: Any, params: dict[str, Any]) -> dict[str, Any]:
    return _jsonrpc_result(id, {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"listChanged": False},
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
    })


async def handle_tools_list(id: Any, params: dict[str, Any]) -> dict[str, Any]:
    return _jsonrpc_result(id, {"tools": MCP_TOOLS})


async def handle_tools_call(id: Any, params: dict[str, Any]) -> dict[str, Any]:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return _jsonrpc_error(id, -32602, f"Unknown tool: {tool_name}")

    try:
        result = await handler(arguments)
        text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        return _jsonrpc_result(id, {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        })
    except Exception as exc:
        return _jsonrpc_result(id, {
            "content": [{"type": "text", "text": f"Error: {exc}"}],
            "isError": True,
        })


async def handle_ping(id: Any, params: dict[str, Any]) -> dict[str, Any]:
    return _jsonrpc_result(id, {})


# Method dispatch
_METHOD_HANDLERS: dict[str, Any] = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
    "ping": handle_ping,
}

# Notifications (no response expected)
_NOTIFICATIONS: set[str] = {
    "notifications/initialized",
    "notifications/cancelled",
}


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------

async def _read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read a single JSON-RPC message from stdin.

    MCP stdio transport sends newline-delimited JSON.
    """
    line = await reader.readline()
    if not line:
        return None  # EOF
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _write_message(msg: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout."""
    data = json.dumps(msg, ensure_ascii=False)
    sys.stdout.write(data + "\n")
    sys.stdout.flush()


async def run_stdio_server() -> None:
    """Main MCP server loop over stdio."""
    import os

    # Setup config first — log path depends on it
    from captain_claw.config import Config, set_config
    from captain_claw.logging import configure_logging
    cfg = Config.load()
    set_config(cfg)
    configure_logging()

    # Redirect stderr so agent log output doesn't corrupt the JSON-RPC stream
    log_path = Path(cfg.mcp_serve.log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    os.dup2(log_file.fileno(), 2)  # Redirect fd 2 (stderr)

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    _log(log_file, "MCP server started")

    try:
        while True:
            msg = await _read_message(reader)
            if msg is None:
                break  # EOF — client disconnected

            method = msg.get("method", "")
            msg_id = msg.get("id")
            params = msg.get("params", {})

            _log(log_file, f"<- {method} id={msg_id}")

            # Notifications: no response
            if method in _NOTIFICATIONS:
                continue

            # Find handler
            handler = _METHOD_HANDLERS.get(method)
            if handler is None:
                if msg_id is not None:
                    response = _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")
                    _write_message(response)
                continue

            # Handle request
            try:
                response = await handler(msg_id, params)
                _write_message(response)
                _log(log_file, f"-> response id={msg_id}")
            except Exception as exc:
                if msg_id is not None:
                    response = _jsonrpc_error(msg_id, -32603, str(exc))
                    _write_message(response)
                _log(log_file, f"!! error: {exc}")

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        _log(log_file, "MCP server stopped")
        log_file.close()


def _log(f: Any, msg: str) -> None:
    """Write a timestamped line to the log file."""
    f.write(f"[{_utcnow_iso()}] {msg}\n")
    f.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for captain-claw-mcp command."""
    asyncio.run(run_stdio_server())


if __name__ == "__main__":
    main()
