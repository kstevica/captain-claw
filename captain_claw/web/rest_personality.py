"""REST handlers for the agent personality profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


async def get_personality(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/personality — return the personality as a JSON object."""
    from captain_claw.personality import load_personality, personality_to_dict

    p = load_personality()
    return web.json_response(personality_to_dict(p))


async def put_personality(server: WebServer, request: web.Request) -> web.Response:
    """PUT /api/personality — update personality fields from a JSON body.

    Accepted body keys: ``name``, ``description``, ``background``,
    ``expertise`` (list of strings or comma-separated string).
    Only provided keys are updated; omitted keys are left unchanged.
    """
    from captain_claw.personality import (
        load_personality,
        personality_to_dict,
        save_personality,
    )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    p = load_personality()

    # ── Merge provided fields ────────────────────────────────────────
    if "name" in body:
        name = str(body["name"]).strip()
        if not name:
            return web.json_response(
                {"error": "Name cannot be empty"}, status=400
            )
        p.name = name

    if "description" in body:
        p.description = str(body["description"]).strip()

    if "background" in body:
        p.background = str(body["background"]).strip()

    if "expertise" in body:
        expertise_raw = body["expertise"]
        if isinstance(expertise_raw, list):
            p.expertise = [str(e).strip() for e in expertise_raw if str(e).strip()]
        elif isinstance(expertise_raw, str):
            p.expertise = [e.strip() for e in expertise_raw.split(",") if e.strip()]

    save_personality(p)

    # Clear instruction caches so the next prompt build picks up changes.
    if server.agent:
        server.agent.instructions._cache.pop("system_prompt.md", None)
        server.agent.instructions._cache.pop("micro_system_prompt.md", None)

    return web.json_response(personality_to_dict(p))
