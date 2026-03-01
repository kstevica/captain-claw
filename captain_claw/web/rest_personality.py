"""REST handlers for the agent personality profile.

Provides endpoints for both the global (default) personality and
per-user (Telegram) personalities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer


# ── Helpers ──────────────────────────────────────────────────────────


def _merge_personality_fields(p: "Personality", body: dict) -> None:  # type: ignore[name-defined]
    """Apply body fields to personality *p* in-place."""
    if "name" in body:
        name = str(body["name"]).strip()
        if name:
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


def _clear_instruction_caches(server: "WebServer") -> None:
    """Clear instruction caches so prompts are rebuilt with new personality."""
    if server.agent:
        server.agent.instructions._cache.pop("system_prompt.md", None)
        server.agent.instructions._cache.pop("micro_system_prompt.md", None)
    # Also clear caches for all Telegram user agents.
    for agent in getattr(server, "_telegram_agents", {}).values():
        if hasattr(agent, "instructions") and hasattr(agent.instructions, "_cache"):
            agent.instructions._cache.pop("system_prompt.md", None)
            agent.instructions._cache.pop("micro_system_prompt.md", None)


# ── Global personality endpoints ────────────────────────────────────


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

    if "name" in body and not str(body["name"]).strip():
        return web.json_response(
            {"error": "Name cannot be empty"}, status=400
        )

    _merge_personality_fields(p, body)
    save_personality(p)
    _clear_instruction_caches(server)

    return web.json_response(personality_to_dict(p))


# ── Per-user personality endpoints ──────────────────────────────────


async def list_user_personalities(
    server: WebServer, request: web.Request
) -> web.Response:
    """GET /api/user-personalities — list all user personalities."""
    from captain_claw.personality import list_user_personalities as _list_all

    # Enrich with display labels from approved Telegram users.
    items = _list_all()
    approved = getattr(server, "_approved_telegram_users", {})
    for item in items:
        uid = item.get("user_id", "")
        info = approved.get(uid, {})
        item["username"] = str(info.get("username", "")).strip()
        item["first_name"] = str(info.get("first_name", "")).strip()
    return web.json_response(items)


async def get_user_personality(
    server: WebServer, request: web.Request
) -> web.Response:
    """GET /api/user-personalities/{user_id} — get one user personality."""
    from captain_claw.personality import load_user_personality, personality_to_dict

    user_id = request.match_info["user_id"]
    p = load_user_personality(user_id)
    if p is None:
        return web.json_response({"error": "Not found"}, status=404)
    d = personality_to_dict(p)
    d["user_id"] = user_id
    return web.json_response(d)


async def put_user_personality(
    server: WebServer, request: web.Request
) -> web.Response:
    """PUT /api/user-personalities/{user_id} — create or update a user personality."""
    from captain_claw.personality import (
        Personality,
        load_user_personality,
        personality_to_dict,
        save_user_personality,
    )

    user_id = request.match_info["user_id"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    p = load_user_personality(user_id) or Personality()
    _merge_personality_fields(p, body)
    save_user_personality(user_id, p)
    _clear_instruction_caches(server)

    d = personality_to_dict(p)
    d["user_id"] = user_id
    return web.json_response(d)


async def delete_user_personality(
    server: WebServer, request: web.Request
) -> web.Response:
    """DELETE /api/user-personalities/{user_id} — remove a user personality."""
    from captain_claw.personality import delete_user_personality as _delete

    user_id = request.match_info["user_id"]
    removed = _delete(user_id)
    if not removed:
        return web.json_response({"error": "Not found"}, status=404)
    _clear_instruction_caches(server)
    return web.json_response({"ok": True})


async def list_telegram_users(
    server: WebServer, request: web.Request
) -> web.Response:
    """GET /api/telegram-users — list approved Telegram users.

    Returns a minimal list so the UI can offer a user picker for
    creating new user personalities.
    """
    approved = getattr(server, "_approved_telegram_users", {})
    result = []
    for uid, info in approved.items():
        result.append({
            "user_id": uid,
            "username": str(info.get("username", "")).strip(),
            "first_name": str(info.get("first_name", "")).strip(),
        })
    return web.json_response(result)
