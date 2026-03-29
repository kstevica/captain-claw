"""Admin REST endpoints for managing public sessions.

Endpoints
---------
GET    /api/public/admin/sessions           – list all public sessions with stats
POST   /api/public/admin/sessions           – create a new pre-configured session
POST   /api/public/admin/sessions/bulk      – create multiple sessions at once
PATCH  /api/public/admin/sessions/{id}      – update session settings + lock
DELETE /api/public/admin/sessions/{id}      – delete a session

All endpoints require admin authentication.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


def _require_admin(request: web.Request, server: "WebServer") -> bool:
    from captain_claw.web.public_auth import _is_admin
    return _is_admin(request, server.config.web)


async def list_public_sessions(server: "WebServer", request: web.Request) -> web.Response:
    """List all public sessions with metadata and stats."""
    if not _require_admin(request, server):
        return web.json_response({"error": "forbidden"}, status=403)

    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    sessions = await sm.list_sessions(limit=500)

    result = []
    for s in sessions:
        meta = s.metadata or {}
        if not meta.get("public_code"):
            continue  # Skip non-public sessions.

        # Count user vs assistant messages.
        user_msgs = sum(1 for m in s.messages if m.get("role") == "user")
        assistant_msgs = sum(1 for m in s.messages if m.get("role") == "assistant")

        # Find last user activity timestamp.
        last_active = s.updated_at
        for m in reversed(s.messages):
            if m.get("role") == "user" and m.get("timestamp"):
                last_active = m["timestamp"]
                break

        result.append({
            "id": s.id,
            "code": meta.get("public_code", ""),
            "session_name": meta.get("session_display_name", ""),
            "session_description": meta.get("session_description", ""),
            "session_instructions": meta.get("session_instructions", ""),
            "locked": bool(meta.get("session_settings_locked", False)),
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "last_active": last_active,
            "message_count": len(s.messages),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
        })

    # Sort by last_active descending.
    result.sort(key=lambda x: x["last_active"] or "", reverse=True)
    return web.json_response(result)


async def create_public_session(server: "WebServer", request: web.Request) -> web.Response:
    """Create a new pre-configured public session."""
    if not _require_admin(request, server):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        body = await request.json()
    except Exception:
        body = {}

    from captain_claw.web.public_session import create_public_session as _create
    session, code = await _create(server)

    # Apply settings.
    if session.metadata is None:
        session.metadata = {}
    if body.get("session_name"):
        session.metadata["session_display_name"] = str(body["session_name"]).strip()
    if body.get("session_description"):
        session.metadata["session_description"] = str(body["session_description"]).strip()
    if body.get("session_instructions"):
        session.metadata["session_instructions"] = str(body["session_instructions"]).strip()
    if body.get("locked") is not None:
        session.metadata["session_settings_locked"] = bool(body["locked"])

    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    await sm.save_session(session)

    return web.json_response({
        "id": session.id,
        "code": code,
        "session_name": session.metadata.get("session_display_name", ""),
        "session_description": session.metadata.get("session_description", ""),
        "session_instructions": session.metadata.get("session_instructions", ""),
        "locked": bool(session.metadata.get("session_settings_locked", False)),
    })


async def bulk_create_sessions(server: "WebServer", request: web.Request) -> web.Response:
    """Create multiple pre-configured sessions at once.

    Body: { "sessions": [ { "session_name": ..., "session_description": ...,
                             "session_instructions": ..., "locked": true }, ... ] }
    """
    if not _require_admin(request, server):
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad_request"}, status=400)

    items = body.get("sessions", [])
    if not isinstance(items, list) or not items:
        return web.json_response({"error": "missing_sessions", "message": "Provide a 'sessions' array."}, status=400)

    from captain_claw.web.public_session import create_public_session as _create
    from captain_claw.session import get_session_manager
    sm = get_session_manager()

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        session, code = await _create(server)
        if session.metadata is None:
            session.metadata = {}
        if item.get("session_name"):
            session.metadata["session_display_name"] = str(item["session_name"]).strip()
        if item.get("session_description"):
            session.metadata["session_description"] = str(item["session_description"]).strip()
        if item.get("session_instructions"):
            session.metadata["session_instructions"] = str(item["session_instructions"]).strip()
        session.metadata["session_settings_locked"] = bool(item.get("locked", True))

        await sm.save_session(session)
        results.append({
            "id": session.id,
            "code": code,
            "session_name": session.metadata.get("session_display_name", ""),
            "locked": bool(session.metadata.get("session_settings_locked", False)),
        })

    return web.json_response({"created": results, "count": len(results)})


async def update_public_session(server: "WebServer", request: web.Request) -> web.Response:
    """Update session settings and/or lock state."""
    if not _require_admin(request, server):
        return web.json_response({"error": "forbidden"}, status=403)

    session_id = request.match_info["id"]
    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    session = await sm.load_session(session_id)
    if session is None:
        return web.json_response({"error": "not_found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "bad_request"}, status=400)

    if session.metadata is None:
        session.metadata = {}

    changed = False
    if "session_name" in body and isinstance(body["session_name"], str):
        session.metadata["session_display_name"] = body["session_name"].strip()
        changed = True
    if "session_description" in body and isinstance(body["session_description"], str):
        session.metadata["session_description"] = body["session_description"].strip()
        changed = True
    if "session_instructions" in body and isinstance(body["session_instructions"], str):
        session.metadata["session_instructions"] = body["session_instructions"].strip()
        changed = True
    if "locked" in body:
        session.metadata["session_settings_locked"] = bool(body["locked"])
        changed = True

    if changed:
        await sm.save_session(session)
        # Clear instruction cache for live agent if it exists.
        agent = server._public_agents.get(session_id)
        if agent and hasattr(agent, "instructions") and hasattr(agent.instructions, "_cache"):
            agent.instructions._cache.pop("system_prompt.md", None)
            agent.instructions._cache.pop("micro_system_prompt.md", None)
        # Update the live agent's session metadata.
        if agent and agent.session:
            agent.session.metadata = session.metadata

    return web.json_response({"ok": True})


async def delete_public_session(server: "WebServer", request: web.Request) -> web.Response:
    """Delete a public session."""
    if not _require_admin(request, server):
        return web.json_response({"error": "forbidden"}, status=403)

    session_id = request.match_info["id"]
    from captain_claw.session import get_session_manager
    sm = get_session_manager()
    deleted = await sm.delete_session(session_id)
    if not deleted:
        return web.json_response({"error": "not_found"}, status=404)
    # Clean up live agent if any.
    server._public_agents.pop(session_id, None)
    return web.json_response({"ok": True})
