"""REST handlers for MCP connector management — delegates to Flight Deck.

In Phase 1 of the MCP-via-Flight-Deck design, MCP servers are
administered exclusively from Flight Deck. This handler stays in
place so the existing per-agent settings page can still offer a
"Test connection" button — but it forwards the probe request to FD's
``/fd/mcp/probe`` endpoint instead of running the MCP handshake from
the agent process.

When the agent is not running under Flight Deck (``FD_URL`` unset)
the endpoint returns a clear error directing the user to manage MCP
through Flight Deck.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.fd_client import FDClient, is_under_flight_deck
from captain_claw.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def test_connection(server: "WebServer", request: web.Request) -> web.Response:
    """Forward a "Test connection" probe to Flight Deck."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    if not is_under_flight_deck():
        return web.json_response(
            {
                "ok": False,
                "error": (
                    "MCP servers are now managed by Flight Deck. Start "
                    "Flight Deck and add the server via Connections → MCP."
                ),
            },
            status=400,
        )

    fd = FDClient()
    try:
        resp = await fd.post("/fd/mcp/probe", json=body)
    except Exception as exc:
        log.error("Failed to forward MCP probe to FD", error=str(exc))
        return web.json_response(
            {"ok": False, "error": f"Could not reach Flight Deck: {exc}"},
            status=502,
        )
    finally:
        await fd.close()

    if resp.status_code != 200:
        try:
            payload = resp.json()
        except Exception:
            payload = {"ok": False, "error": resp.text[:500]}
        return web.json_response(payload, status=resp.status_code)

    return web.json_response(resp.json())
