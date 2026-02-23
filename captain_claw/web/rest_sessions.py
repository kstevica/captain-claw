"""REST handlers for session browsing and management.

Endpoints
---------
GET    /api/sessions              – list sessions (with optional ?q= search)
GET    /api/sessions/{id}         – full session detail with messages
PATCH  /api/sessions/{id}         – rename / set description
DELETE /api/sessions/{id}         – delete a session
POST   /api/sessions/{id}/auto-describe – LLM-generate a description
GET    /api/sessions/{id}/export  – download chat/monitor/all as markdown
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def list_sessions(server: WebServer, request: web.Request) -> web.Response:
    """List sessions, optionally filtered by ?q= search term."""
    from captain_claw.session import get_session_manager

    sm = get_session_manager()
    sessions = await sm.list_sessions(limit=100)

    # Optional client-side search filter
    query = (request.query.get("q") or "").strip().lower()
    result: list[dict[str, Any]] = []
    for s in sessions:
        desc = (s.metadata or {}).get("description", "")
        if query and query not in s.name.lower() and query not in desc.lower():
            continue
        result.append({
            "id": s.id,
            "name": s.name,
            "message_count": len(s.messages),
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "description": desc,
        })
    return web.json_response(result)


async def get_session_detail(server: WebServer, request: web.Request) -> web.Response:
    """Return full session detail including all messages."""
    from captain_claw.session import get_session_manager

    session_id = request.match_info["id"]
    sm = get_session_manager()
    session = await sm.load_session(session_id)
    if not session:
        return web.json_response({"error": "Session not found"}, status=404)

    return web.json_response({
        "id": session.id,
        "name": session.name,
        "description": (session.metadata or {}).get("description", ""),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": len(session.messages),
        "messages": session.messages,
    })


async def update_session(server: WebServer, request: web.Request) -> web.Response:
    """Update session name and/or description."""
    from captain_claw.session import get_session_manager

    session_id = request.match_info["id"]
    sm = get_session_manager()
    session = await sm.load_session(session_id)
    if not session:
        return web.json_response({"error": "Session not found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    changed = False
    if "name" in body and isinstance(body["name"], str) and body["name"].strip():
        session.name = body["name"].strip()
        changed = True
    if "description" in body and isinstance(body["description"], str):
        if session.metadata is None:
            session.metadata = {}
        session.metadata["description"] = body["description"].strip()
        changed = True

    if changed:
        await sm.save_session(session)

    return web.json_response({"ok": True})


async def delete_session(server: WebServer, request: web.Request) -> web.Response:
    """Delete a session by ID."""
    from captain_claw.session import get_session_manager

    session_id = request.match_info["id"]
    sm = get_session_manager()
    deleted = await sm.delete_session(session_id)
    if not deleted:
        return web.json_response({"error": "Session not found"}, status=404)
    return web.json_response({"ok": True})


async def auto_describe_session(server: WebServer, request: web.Request) -> web.Response:
    """Auto-generate a description for the session using the LLM."""
    from captain_claw.session import get_session_manager

    session_id = request.match_info["id"]
    sm = get_session_manager()
    session = await sm.load_session(session_id)
    if not session:
        return web.json_response({"error": "Session not found"}, status=404)

    if not server.agent:
        return web.json_response({"error": "Agent not available"}, status=503)

    try:
        description = await server.agent.generate_session_description(session, max_sentences=5)
    except Exception as exc:
        log.warning("Auto-describe failed for session %s: %s", session_id, exc)
        return web.json_response({"error": f"Description generation failed: {exc}"}, status=500)

    if not description:
        return web.json_response({"error": "Could not generate description"}, status=500)

    # Persist the generated description
    if session.metadata is None:
        session.metadata = {}
    session.metadata["description"] = description
    await sm.save_session(session)

    return web.json_response({"description": description})


async def bulk_delete_sessions(server: WebServer, request: web.Request) -> web.Response:
    """Delete multiple sessions by ID."""
    from captain_claw.session import get_session_manager

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    ids = body.get("ids")
    if not isinstance(ids, list) or not ids:
        return web.json_response({"error": "Missing or empty 'ids' list"}, status=400)

    sm = get_session_manager()
    deleted: list[str] = []
    failed: list[str] = []
    for session_id in ids:
        if not isinstance(session_id, str):
            continue
        ok = await sm.delete_session(session_id)
        if ok:
            deleted.append(session_id)
        else:
            failed.append(session_id)

    return web.json_response({"deleted": deleted, "failed": failed})


async def export_session(server: WebServer, request: web.Request) -> web.Response:
    """Export session history as a downloadable markdown file.

    Query params:
      mode=chat|monitor|all  (default: all)
    """
    from captain_claw.session import get_session_manager
    from captain_claw.session_export import (
        normalize_session_id,
        render_chat_export_markdown,
        render_monitor_export_markdown,
    )

    session_id = request.match_info["id"]
    mode = (request.query.get("mode") or "all").strip().lower()
    if mode not in {"chat", "monitor", "all"}:
        mode = "all"

    sm = get_session_manager()
    session = await sm.load_session(session_id)
    if not session:
        return web.json_response({"error": "Session not found"}, status=404)

    safe_name = normalize_session_id(session.name)
    messages: list[dict[str, object]] = [dict(m) for m in session.messages]

    if mode == "chat":
        content = render_chat_export_markdown(session.id, session.name, messages)
        filename = f"{safe_name}-chat.md"
    elif mode == "monitor":
        content = render_monitor_export_markdown(session.id, session.name, messages)
        filename = f"{safe_name}-monitor.md"
    else:
        chat_part = render_chat_export_markdown(session.id, session.name, messages)
        monitor_part = render_monitor_export_markdown(session.id, session.name, messages)
        content = chat_part + "\n---\n\n" + monitor_part
        filename = f"{safe_name}-export.md"

    return web.Response(
        body=content.encode("utf-8"),
        content_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
