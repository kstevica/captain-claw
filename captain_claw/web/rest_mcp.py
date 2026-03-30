"""REST handlers for MCP connector management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def test_connection(server: WebServer, request: web.Request) -> web.Response:
    """Test connectivity to an MCP server: authenticate, initialize, and list tools."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    url = (body.get("url") or "").strip()
    if not url:
        return web.json_response({"ok": False, "error": "Server URL is required"}, status=400)

    from captain_claw.tools.mcp_connector import MCPConnector

    # Don't send masked secrets to the connector — treat as empty.
    SECRET_MASK = "\u2022" * 8
    client_secret = body.get("client_secret", "")
    if client_secret == SECRET_MASK:
        # Try to pull the real secret from saved config
        client_secret = _resolve_saved_secret(body.get("name", ""), url)

    connector = MCPConnector(
        name=body.get("name", "test"),
        server_url=url,
        client_id=body.get("client_id", ""),
        client_secret=client_secret,
        token_endpoint=body.get("token_endpoint", ""),
        headers=body.get("headers") or {},
    )

    try:
        init_result = await connector.initialize()
        server_info = init_result.get("serverInfo", {})
    except Exception as exc:
        return web.json_response({
            "ok": False,
            "error": f"Failed to initialize: {exc}",
        })

    try:
        tools = await connector.discover_tools()
    except Exception as exc:
        return web.json_response({
            "ok": False,
            "error": f"Connected but failed to list tools: {exc}",
            "server_info": server_info,
        })

    return web.json_response({
        "ok": True,
        "server_info": server_info,
        "instructions": init_result.get("instructions", ""),
        "tools": [
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
            }
            for t in tools
        ],
    })


def _resolve_saved_secret(name: str, url: str) -> str:
    """Look up the real client_secret from saved config for a matching server."""
    try:
        from captain_claw.config import get_config
        cfg = get_config()
        for srv in cfg.tools.mcp_servers:
            if (srv.name == name or srv.url == url) and srv.client_secret:
                return srv.client_secret
    except Exception:
        pass
    return ""
