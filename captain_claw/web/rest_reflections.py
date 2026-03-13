"""REST handlers for self-reflections management.

Endpoints
---------
GET    /api/reflections            – list reflections (newest first)
GET    /api/reflections/latest     – get the active (latest) reflection
POST   /api/reflections/generate   – trigger a new reflection
DELETE /api/reflections/{timestamp} – delete a reflection
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def list_reflections_api(server: WebServer, request: web.Request) -> web.Response:
    """List all reflections, newest first."""
    from captain_claw.reflections import list_reflections, reflection_to_dict

    limit = int(request.query.get("limit", "50"))
    refs = list_reflections(limit=limit)
    return web.json_response([reflection_to_dict(r) for r in refs])


async def get_latest_reflection(server: WebServer, request: web.Request) -> web.Response:
    """Return the latest (active) reflection."""
    from captain_claw.reflections import load_latest_reflection, reflection_to_dict

    r = load_latest_reflection()
    if not r:
        return web.json_response({"error": "No reflections yet"}, status=404)
    return web.json_response(reflection_to_dict(r))


async def trigger_reflection(server: WebServer, request: web.Request) -> web.Response:
    """Manually trigger a new self-reflection."""
    from captain_claw.reflections import generate_reflection, reflection_to_dict

    if not server.agent:
        return web.json_response({"error": "Agent not available"}, status=503)

    try:
        r = await generate_reflection(server.agent)
    except Exception as exc:
        log.warning("Reflection generation failed: %s", exc)
        return web.json_response(
            {"error": f"Reflection generation failed: {exc}"}, status=500
        )

    return web.json_response(reflection_to_dict(r))


async def delete_reflection_api(server: WebServer, request: web.Request) -> web.Response:
    """Delete a reflection by timestamp."""
    from captain_claw.reflections import delete_reflection

    timestamp = request.match_info["timestamp"]
    deleted = delete_reflection(timestamp)
    if not deleted:
        return web.json_response({"error": "Reflection not found"}, status=404)
    return web.json_response({"ok": True})
