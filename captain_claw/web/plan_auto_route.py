"""Auto-route a chat message through `/plan` then `/plan-execute`.

Activated when ``server.agent.plan_mode_auto`` is True. The user's free-text
message is treated as a plan request: a plan is generated, persisted, and
executed. Plan-progress UI updates flow through the existing ``plan_*``
WebSocket events emitted by ``PlanExecutor``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def handle_plan_auto_route(
    server: "WebServer", ws: web.WebSocketResponse, content: str,
) -> None:
    """Run /plan <content> then /plan-execute, broadcasting results to chat."""
    from captain_claw.web.plan_commands import (
        handle_plan_command,
        handle_plan_execute_command,
    )

    # Echo the user message so it appears in chat history (UI side only —
    # auto-routed messages don't go through the LLM session).
    server._broadcast({
        "type": "chat_message",
        "role": "user",
        "content": content,
        "timestamp": datetime.now(UTC).isoformat(),
    })
    server._broadcast({"type": "status", "status": "thinking"})

    try:
        plan_result = await handle_plan_command(server, content)
        await server._send(ws, {
            "type": "command_result",
            "command": f"/plan {content}",
            "content": plan_result,
        })

        execute_result = await handle_plan_execute_command(server, "")
        await server._send(ws, {
            "type": "command_result",
            "command": "/plan-execute",
            "content": execute_result,
        })
    except Exception as e:
        log.error("Plan auto-route failed", error=str(e), error_type=type(e).__name__)
        await server._send(ws, {
            "type": "command_result",
            "command": "/plan (auto)",
            "content": f"Plan auto-route failed: {e}",
        })
    finally:
        server._broadcast({"type": "status", "status": "ready"})
