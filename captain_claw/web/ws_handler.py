"""WebSocket protocol handler for the web UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.agent import Agent
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def ws_handler(server: WebServer, request: web.Request) -> web.WebSocketResponse:
    """Handle WebSocket connections."""
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)
    server.clients.add(ws)

    # Send welcome payload
    from captain_claw.web_server import COMMANDS

    models = server.agent.get_allowed_models() if server.agent else []
    await server._send(ws, {
        "type": "welcome",
        "session": server._session_info(),
        "models": models,
        "commands": COMMANDS,
    })

    # Replay existing session messages for the connecting client
    if server.agent and server.agent.session:
        for msg in server.agent.session.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_name = msg.get("tool_name", "")
            timestamp = msg.get("timestamp", "")
            model = msg.get("model", "")
            if role in ("user", "assistant"):
                await server._send(ws, {
                    "type": "chat_message",
                    "role": role,
                    "content": content,
                    "replay": True,
                    "timestamp": timestamp,
                    "model": model,
                })
            elif role == "tool" and tool_name == "task_rephrase":
                # Replay task rephrase as a visible chat panel.
                await server._send(ws, {
                    "type": "chat_message",
                    "role": "rephrase",
                    "content": content,
                    "replay": True,
                })
            elif role == "tool" and tool_name and not Agent._is_monitor_only_tool_name(tool_name):
                await server._send(ws, {
                    "type": "monitor",
                    "tool_name": tool_name,
                    "arguments": msg.get("tool_arguments", {}),
                    "output": content,
                    "replay": True,
                })
        await server._send(ws, {"type": "replay_done"})

    try:
        async for raw_msg in ws:
            if raw_msg.type in (
                web.WSMsgType.TEXT,
            ):
                try:
                    data = json.loads(raw_msg.data)
                except json.JSONDecodeError:
                    await server._send(ws, {"type": "error", "message": "Invalid JSON"})
                    continue
                await handle_ws_message(server, ws, data)
            elif raw_msg.type == web.WSMsgType.ERROR:
                log.error("WebSocket error", error=str(ws.exception()))
    finally:
        server.clients.discard(ws)

    return ws


async def handle_ws_message(
    server: WebServer, ws: web.WebSocketResponse, data: dict
) -> None:
    """Dispatch incoming WebSocket messages."""
    msg_type = data.get("type", "")

    if msg_type == "chat":
        content = str(data.get("content", "")).strip()
        if not content:
            return
        if content.startswith("/"):
            from captain_claw.web.slash_commands import handle_command
            await handle_command(server, ws, content)
        else:
            from captain_claw.web.chat_handler import handle_chat
            await handle_chat(server, ws, content)

    elif msg_type == "command":
        command = str(data.get("command", "")).strip()
        if command:
            from captain_claw.web.slash_commands import handle_command
            await handle_command(server, ws, command)

    elif msg_type == "set_model":
        # Switch model for the current session via direct WebSocket message.
        selector = str(data.get("selector", "")).strip()
        if selector and server.agent:
            ok, msg = await server.agent.set_session_model(selector, persist=True)
            if ok:
                server._broadcast({"type": "session_info", **server._session_info()})
            await server._send(ws, {
                "type": "command_result",
                "command": "/session model",
                "content": msg,
            })

    elif msg_type == "cancel":
        if server.agent and hasattr(server.agent, "cancel_event"):
            server.agent.cancel_event.set()
            log.info("Cancel signal received via WebSocket")

    elif msg_type == "approval_response":
        pass
