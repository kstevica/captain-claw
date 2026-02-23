"""REST handlers for config, sessions list, and commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.config import get_config

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def get_config_summary(server: WebServer, request: web.Request) -> web.Response:
    """Return a safe config summary (no secrets)."""
    cfg = get_config()
    details = {}
    if server.agent:
        details = server.agent.get_runtime_model_details()
    return web.json_response({
        "model": {
            "provider": details.get("provider", cfg.model.provider),
            "model": details.get("model", cfg.model.model),
            "temperature": details.get("temperature", cfg.model.temperature),
            "max_tokens": details.get("max_tokens", cfg.model.max_tokens),
        },
        "context": {
            "max_tokens": cfg.context.max_tokens,
            "compaction_threshold": cfg.context.compaction_threshold,
        },
        "tools": cfg.tools.enabled,
        "guards": {
            "input": cfg.guards.input.enabled,
            "output": cfg.guards.output.enabled,
            "script_tool": cfg.guards.script_tool.enabled,
        },
    })


async def list_sessions_api(server: WebServer, request: web.Request) -> web.Response:
    """List sessions via REST."""
    from captain_claw.session import get_session_manager

    sm = get_session_manager()
    sessions = await sm.list_sessions(limit=20)
    result = []
    for s in sessions:
        result.append({
            "id": s.id,
            "name": s.name,
            "message_count": len(s.messages),
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "description": (s.metadata or {}).get("description", ""),
        })
    return web.json_response(result)


async def get_commands_api(server: WebServer, request: web.Request) -> web.Response:
    """Return the available commands list."""
    from captain_claw.web_server import COMMANDS

    return web.json_response(COMMANDS)
