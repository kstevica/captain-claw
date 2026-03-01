"""Chat message handler for the web UI."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def handle_chat(
    server: WebServer,
    ws: web.WebSocketResponse,
    content: str,
    *,
    image_path: str | None = None,
    file_path: str | None = None,
) -> None:
    """Process a chat message through the agent.

    The actual work is launched as a background asyncio task so that the
    WebSocket read-loop stays free to process incoming messages (most
    importantly ``cancel`` signals) while the agent is running.
    """
    if not server.agent:
        await server._send(ws, {"type": "error", "message": "Agent not initialized"})
        return

    if server._busy:
        await server._send(ws, {
            "type": "error",
            "message": "Agent is busy processing another request. Please wait.",
        })
        return

    # When an image is attached, prepend the path context so the agent
    # knows a file is available for tools like image_ocr.
    effective_content = content
    if image_path:
        prefix = f"[Attached image: {image_path}]\n"
        effective_content = prefix + (content or "Please analyze this image.")

    # When a data file is attached, prepend the path context so the agent
    # knows a file is available. The user's message determines what to do
    # with it (datastore import, deep memory indexing, extraction, etc.).
    if file_path:
        prefix = f"[Attached file: {file_path}]\n"
        effective_content = prefix + (content or "I've attached a file.")

    # Mark busy *before* spawning the task so that a second chat message
    # arriving immediately is rejected.
    server._busy = True
    server._broadcast({"type": "status", "status": "thinking"})
    server._thinking_callback("Thinking\u2026", phase="reasoning")

    # Echo user message to all clients
    server._broadcast({
        "type": "chat_message",
        "role": "user",
        "content": effective_content,
        "timestamp": datetime.now(UTC).isoformat(),
    })

    # Launch the heavy work as a background task — do NOT await it here
    # so the caller (the WebSocket read-loop) can keep processing
    # incoming messages such as cancel/stop.
    task = asyncio.create_task(_run_agent(server, effective_content))

    # Store a reference so it isn't garbage-collected.
    server._active_task = task


async def _run_agent(server: WebServer, content: str) -> None:
    """Background coroutine that drives the agent and finalises the turn."""
    try:
        # Capture model details before running the agent.
        model_details = server.agent.get_runtime_model_details() if server.agent else {}
        model_label = f"{model_details.get('provider', '')}:{model_details.get('model', '')}" if model_details else ""

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
                    "timestamp": datetime.now(UTC).isoformat(),
                    "model": model_label,
                })
        else:
            # Use complete() which handles tool calls and guards
            response = await server.agent.complete(content)

            server._broadcast({
                "type": "chat_message",
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(UTC).isoformat(),
                "model": model_label,
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
        server._active_task = None
        server._broadcast({"type": "status", "status": "ready"})
