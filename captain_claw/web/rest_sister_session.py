"""REST handlers for the Sister Session (proactive tasks and briefings)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger
from captain_claw.sister_session import get_sister_session_manager, get_session_sister_manager

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


def _resolve_manager(request: web.Request) -> Any:
    """Return the appropriate SisterSessionManager for this request."""
    from captain_claw.web.public_auth import get_request_session_id
    is_public, session_id = get_request_session_id(request)
    if is_public and session_id:
        return get_session_sister_manager(session_id)
    return get_sister_session_manager()


# ── Tasks ────────────────────────────────────────────────────────────


async def list_tasks(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/sister/tasks — list proactive tasks.

    Query params:
      session_id — filter by parent session (optional)
      status     — filter by status (optional)
      limit      — max results (default 50)
    """
    mgr = _resolve_manager(request)
    session_id = request.query.get("session_id", "").strip() or None
    status = request.query.get("status", "").strip() or None
    limit = min(int(request.query.get("limit", "50")), 200)

    items = await mgr.list_tasks(session_id, status=status, limit=limit)
    total = await mgr.count_tasks(session_id, status=status)

    return web.json_response(
        {"items": items, "total": total},
        dumps=_JSON_DUMPS,
    )


async def get_task(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/sister/tasks/{id} — get a single task."""
    mgr = _resolve_manager(request)
    task_id = request.match_info["id"]
    item = await mgr.get_task(task_id)
    if not item:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(item, dumps=_JSON_DUMPS)


async def create_task(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/sister/tasks — create a proactive task manually."""
    from captain_claw.sister_session import maybe_create_proactive_task

    body = await request.json()
    trigger_reason = str(body.get("trigger_reason", body.get("query", ""))).strip()
    if not trigger_reason:
        return web.json_response({"error": "'trigger_reason' or 'query' is required"}, status=400)

    # Get parent session from body or from the active agent
    parent_session_id = str(body.get("parent_session_id", "")).strip()
    if not parent_session_id:
        agent = getattr(server, "agent", None)
        if agent and agent.session:
            parent_session_id = str(agent.session.id)
        else:
            return web.json_response({"error": "No active session"}, status=400)

    priority = int(body.get("priority", 8))

    task_id = await maybe_create_proactive_task(
        parent_session_id=parent_session_id,
        source_type="manual",
        source_id=f"manual-{trigger_reason[:20]}",
        trigger_reason=trigger_reason,
        priority=priority,
    )
    if task_id:
        mgr = _resolve_manager(request)
        item = await mgr.get_task(task_id)
        return web.json_response(item, status=201, dumps=_JSON_DUMPS)
    return web.json_response(
        {"error": "Task not created — rate limit, budget cap, or duplicate"},
        status=429,
    )


async def delete_task(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/sister/tasks/{id} — cancel/delete a task."""
    mgr = _resolve_manager(request)
    task_id = request.match_info["id"]
    ok = await mgr.delete_task(task_id)
    if not ok:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response({"ok": True})


# ── Briefings ────────────────────────────────────────────────────────


async def list_briefings(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/briefings — list briefings.

    Query params:
      session_id — filter by parent session (optional)
      status     — filter by status (optional)
      limit      — max results (default 50)
    """
    mgr = _resolve_manager(request)
    session_id = request.query.get("session_id", "").strip() or None
    status = request.query.get("status", "").strip() or None
    limit = min(int(request.query.get("limit", "50")), 200)

    items = await mgr.list_briefings(session_id, status=status, limit=limit)
    total = await mgr.count_briefings(session_id, status=status)
    unread = await mgr.count_briefings(session_id, status="unread")

    return web.json_response(
        {"items": items, "total": total, "unread": unread},
        dumps=_JSON_DUMPS,
    )


async def get_briefing(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/briefings/{id} — get a single briefing (marks as read)."""
    mgr = _resolve_manager(request)
    briefing_id = request.match_info["id"]
    item = await mgr.get_briefing(briefing_id)
    if not item:
        return web.json_response({"error": "Not found"}, status=404)
    # Auto-mark as read on detail view
    if item.get("status") == "unread":
        await mgr.mark_read(briefing_id)
        item["status"] = "read"
    return web.json_response(item, dumps=_JSON_DUMPS)


async def update_briefing(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/briefings/{id} — update briefing status."""
    mgr = _resolve_manager(request)
    briefing_id = request.match_info["id"]
    body = await request.json()
    new_status = str(body.get("status", "")).strip()

    if new_status == "dismissed":
        ok = await mgr.dismiss(briefing_id)
    elif new_status == "read":
        ok = await mgr.mark_read(briefing_id)
    elif new_status == "acted":
        ok = await mgr.update_briefing(briefing_id, status="acted")
    else:
        ok = await mgr.update_briefing(briefing_id, **body)

    if not ok:
        return web.json_response({"error": "Not found or no changes"}, status=404)
    item = await mgr.get_briefing(briefing_id)
    return web.json_response(item, dumps=_JSON_DUMPS)


async def delete_briefing(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/briefings/{id} — delete a briefing."""
    mgr = _resolve_manager(request)
    briefing_id = request.match_info["id"]
    ok = await mgr.delete_briefing(briefing_id)
    if not ok:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response({"ok": True})


# ── Stats ────────────────────────────────────────────────────────────


async def get_stats(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/sister/stats — aggregate statistics."""
    mgr = _resolve_manager(request)
    session_id = request.query.get("session_id", "").strip() or None
    data = await mgr.stats(session_id)
    return web.json_response(data, dumps=_JSON_DUMPS)


# ── Watches ───────────────────────────────────────────────────────────


async def list_watches(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/sister/watches — list active watches.

    Query params:
      session_id — filter by parent session (optional)
    """
    mgr = _resolve_manager(request)
    session_id = request.query.get("session_id", "").strip() or None
    items = await mgr.list_watches(session_id)
    return web.json_response({"items": items, "total": len(items)}, dumps=_JSON_DUMPS)


async def create_watch(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/sister/watches — create a recurring watch."""
    mgr = _resolve_manager(request)
    body = await request.json()

    query = str(body.get("query", "")).strip()
    if not query:
        return web.json_response({"error": "'query' is required"}, status=400)

    parent_session_id = str(body.get("parent_session_id", "")).strip()
    if not parent_session_id:
        agent = getattr(server, "agent", None)
        if agent and agent.session:
            parent_session_id = str(agent.session.id)
        else:
            return web.json_response({"error": "No active session"}, status=400)

    interval_seconds = int(body.get("interval_seconds", 3600))

    watch_id = await mgr.create_watch(
        parent_session_id=parent_session_id,
        query=query,
        interval_seconds=interval_seconds,
    )
    watch = {"id": watch_id, "query": query, "interval_seconds": interval_seconds}
    return web.json_response(watch, status=201, dumps=_JSON_DUMPS)


async def delete_watch(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/sister/watches/{id} — delete a watch."""
    mgr = _resolve_manager(request)
    watch_id = request.match_info["id"]
    ok = await mgr.delete_watch(watch_id)
    if not ok:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response({"ok": True})
