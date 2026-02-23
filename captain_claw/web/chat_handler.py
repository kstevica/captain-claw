"""Chat message handler for the web UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def handle_chat(server: WebServer, ws: web.WebSocketResponse, content: str) -> None:
    """Process a chat message through the agent."""
    if not server.agent:
        await server._send(ws, {"type": "error", "message": "Agent not initialized"})
        return

    if server._busy:
        await server._send(ws, {
            "type": "error",
            "message": "Agent is busy processing another request. Please wait.",
        })
        return

    async with server._busy_lock:
        server._busy = True
        server._broadcast({"type": "status", "status": "thinking"})
        server._thinking_callback("Thinking\u2026", phase="reasoning")

        # Echo user message to all clients
        server._broadcast({
            "type": "chat_message",
            "role": "user",
            "content": content,
        })

        try:
            # Route /orchestrate requests to the orchestrator.
            stripped = content.strip()
            if stripped.lower().startswith("/orchestrate ") and server._orchestrator:
                orchestrate_input = stripped[len("/orchestrate "):].strip()
                if not orchestrate_input:
                    server._broadcast({
                        "type": "error",
                        "message": "Usage: /orchestrate <request>",
                    })
                else:
                    response = await server._orchestrator.orchestrate(orchestrate_input)
                    server._broadcast({
                        "type": "chat_message",
                        "role": "assistant",
                        "content": response,
                    })
            else:
                # Use complete() which handles tool calls and guards
                response = await server.agent.complete(content)

                server._broadcast({
                    "type": "chat_message",
                    "role": "assistant",
                    "content": response,
                })

            # Send updated usage/session info
            server._broadcast({
                "type": "usage",
                "last": server.agent.last_usage,
                "total": server.agent.total_usage,
            })
            server._broadcast({
                "type": "session_info",
                **server._session_info(),
            })

        except Exception as e:
            log.error("Chat error", error=str(e))
            server._broadcast({
                "type": "error",
                "message": f"Error: {str(e)}",
            })
        finally:
            server._busy = False
            server._broadcast({"type": "status", "status": "ready"})
