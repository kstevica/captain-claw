"""REST handlers for Playbook CRUD."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.session import get_session_manager

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


async def list_playbooks(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/playbooks — list all playbooks, optionally filtered by task_type."""
    sm = get_session_manager()
    task_type = request.query.get("task_type") or None
    items = await sm.list_playbooks(limit=200, task_type=task_type)
    return web.json_response(
        [_playbook_dict(p) for p in items],
        dumps=_JSON_DUMPS,
    )


async def search_playbooks(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/playbooks/search?q=... — keyword search."""
    sm = get_session_manager()
    query = request.query.get("q", "").strip()
    task_type = request.query.get("task_type") or None
    if not query:
        return web.json_response({"error": "q parameter is required"}, status=400)
    items = await sm.search_playbooks(query, limit=50, task_type=task_type)
    return web.json_response(
        [_playbook_dict(p) for p in items],
        dumps=_JSON_DUMPS,
    )


async def get_playbook(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/playbooks/{id} — get one playbook by ID."""
    sm = get_session_manager()
    pid = request.match_info.get("id", "")
    item = await sm.load_playbook(pid)
    if not item:
        return web.json_response({"error": "Playbook not found"}, status=404)
    return web.json_response(_playbook_dict(item), dumps=_JSON_DUMPS)


async def create_playbook(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/playbooks — create a new playbook."""
    sm = get_session_manager()
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = str(body.get("name", "")).strip()
    task_type = str(body.get("task_type", "")).strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if not task_type:
        return web.json_response({"error": "task_type is required"}, status=400)

    do_pattern = str(body.get("do_pattern", "")).strip()
    dont_pattern = str(body.get("dont_pattern", "")).strip()
    if not do_pattern and not dont_pattern:
        return web.json_response(
            {"error": "At least one of do_pattern or dont_pattern is required"},
            status=400,
        )

    item = await sm.create_playbook(
        name=name,
        task_type=task_type,
        rating=str(body.get("rating", "good")).strip() or "good",
        do_pattern=do_pattern,
        dont_pattern=dont_pattern,
        trigger_description=str(body.get("trigger_description", "")).strip(),
        reasoning=str(body.get("reasoning", "")).strip() or None,
        tags=str(body.get("tags", "")).strip() or None,
        source_session=str(body.get("source_session", "")).strip() or None,
    )
    return web.json_response(_playbook_dict(item), status=201, dumps=_JSON_DUMPS)


async def update_playbook(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/playbooks/{id} — update a playbook."""
    sm = get_session_manager()
    pid = request.match_info.get("id", "")
    item = await sm.load_playbook(pid)
    if not item:
        return web.json_response({"error": "Playbook not found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    fields: dict[str, Any] = {}
    for key in (
        "name", "task_type", "rating", "do_pattern", "dont_pattern",
        "trigger_description", "reasoning", "tags",
    ):
        if key in body:
            val = body[key]
            fields[key] = str(val).strip() if val is not None else None

    if not fields:
        return web.json_response({"error": "No fields to update"}, status=400)

    ok = await sm.update_playbook(item.id, **fields)
    if not ok:
        return web.json_response({"error": "Update failed"}, status=500)

    updated = await sm.load_playbook(item.id)
    return web.json_response(
        _playbook_dict(updated) if updated else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_playbook(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/playbooks/{id} — delete a playbook."""
    sm = get_session_manager()
    pid = request.match_info.get("id", "")
    item = await sm.load_playbook(pid)
    if not item:
        return web.json_response({"error": "Playbook not found"}, status=404)
    ok = await sm.delete_playbook(item.id)
    if not ok:
        return web.json_response({"error": "Delete failed"}, status=500)
    return web.Response(status=204)


# ── Helpers ──────────────────────────────────────────────────────────


def _playbook_dict(p: Any) -> dict[str, Any]:
    """Convert a PlaybookEntry to a JSON-friendly dict."""
    return {
        "id": p.id,
        "name": p.name,
        "task_type": p.task_type,
        "rating": p.rating,
        "do_pattern": p.do_pattern,
        "dont_pattern": p.dont_pattern,
        "trigger_description": p.trigger_description,
        "reasoning": p.reasoning,
        "tags": p.tags,
        "use_count": p.use_count,
        "last_used_at": p.last_used_at,
        "source_session": p.source_session,
        "created_at": p.created_at,
        "updated_at": p.updated_at,
    }
