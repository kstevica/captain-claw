"""REST handlers for Browser Workflow CRUD (recorded interaction workflows)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.session import get_session_manager

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


async def list_workflows(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/browser-workflows — list all recorded workflows."""
    sm = get_session_manager()
    app_name = request.query.get("app_name") or None
    items = await sm.list_workflows(limit=200, app_name=app_name)
    return web.json_response(
        [_workflow_dict(w) for w in items],
        dumps=_JSON_DUMPS,
    )


async def get_workflow(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/browser-workflows/{id} — get one workflow by ID."""
    sm = get_session_manager()
    wid = request.match_info.get("id", "")
    item = await sm.load_workflow(wid)
    if not item:
        return web.json_response({"error": "Workflow not found"}, status=404)
    return web.json_response(_workflow_dict(item), dumps=_JSON_DUMPS)


async def create_workflow(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/browser-workflows — create a new workflow."""
    sm = get_session_manager()
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)

    steps = body.get("steps", [])
    variables = body.get("variables", [])

    item = await sm.create_workflow(
        name=name,
        description=str(body.get("description", "")).strip(),
        app_name=str(body.get("app_name", "")).strip(),
        start_url=str(body.get("start_url", "")).strip(),
        steps=json.dumps(steps) if isinstance(steps, list) else str(steps),
        variables=json.dumps(variables) if isinstance(variables, list) else str(variables),
    )
    return web.json_response(_workflow_dict(item), status=201, dumps=_JSON_DUMPS)


async def update_workflow(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/browser-workflows/{id} — update a workflow."""
    sm = get_session_manager()
    wid = request.match_info.get("id", "")
    item = await sm.load_workflow(wid)
    if not item:
        return web.json_response({"error": "Workflow not found"}, status=404)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    fields: dict[str, Any] = {}
    for key in ("name", "description", "app_name", "start_url"):
        if key in body:
            val = body[key]
            fields[key] = str(val).strip() if val is not None else ""
    if "steps" in body:
        steps = body["steps"]
        fields["steps"] = json.dumps(steps) if isinstance(steps, list) else str(steps)
    if "variables" in body:
        variables = body["variables"]
        fields["variables"] = json.dumps(variables) if isinstance(variables, list) else str(variables)

    if not fields:
        return web.json_response({"error": "No fields to update"}, status=400)

    ok = await sm.update_workflow(item.id, **fields)
    if not ok:
        return web.json_response({"error": "Update failed"}, status=500)

    updated = await sm.load_workflow(item.id)
    return web.json_response(
        _workflow_dict(updated) if updated else {"ok": True},
        dumps=_JSON_DUMPS,
    )


async def delete_workflow(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/browser-workflows/{id} — delete a workflow."""
    sm = get_session_manager()
    wid = request.match_info.get("id", "")
    item = await sm.load_workflow(wid)
    if not item:
        return web.json_response({"error": "Workflow not found"}, status=404)
    ok = await sm.delete_workflow(item.id)
    if not ok:
        return web.json_response({"error": "Delete failed"}, status=500)
    return web.Response(status=204)


# ── Helpers ──────────────────────────────────────────────────────────


def _workflow_dict(w: Any) -> dict[str, Any]:
    """Convert a WorkflowEntry to a JSON-friendly dict."""
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "app_name": w.app_name,
        "start_url": w.start_url,
        "steps": json.loads(w.steps) if isinstance(w.steps, str) else w.steps,
        "variables": json.loads(w.variables) if isinstance(w.variables, str) else w.variables,
        "use_count": w.use_count,
        "last_used_at": w.last_used_at,
        "created_at": w.created_at,
        "updated_at": w.updated_at,
    }
