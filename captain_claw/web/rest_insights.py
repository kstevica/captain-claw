"""REST handlers for the Insights browser UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.insights import get_insights_manager, get_session_insights_manager
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


def _resolve_manager(request: web.Request) -> Any:
    """Return the appropriate InsightsManager for this request."""
    from captain_claw.web.public_auth import get_request_session_id
    is_public, session_id = get_request_session_id(request)
    if is_public and session_id:
        return get_session_insights_manager(session_id)
    return get_insights_manager()


# ── List / Search ────────────────────────────────────────────────────


async def list_insights(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/insights — list or search insights.

    Query params:
      q        — FTS search query (optional)
      category — filter by category (optional)
      limit    — max results (default 50)
    """
    mgr = _resolve_manager(request)
    query = request.query.get("q", "").strip()
    category = request.query.get("category", "").strip() or None
    limit = min(int(request.query.get("limit", "50")), 200)

    if query:
        items = await mgr.search(query, category=category, limit=limit)
    else:
        items = await mgr.list_recent(limit=limit, category=category)

    total = await mgr.count()

    return web.json_response(
        {"items": items, "total": total},
        dumps=_JSON_DUMPS,
    )


async def get_insight(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/insights/{id} — get a single insight."""
    mgr = _resolve_manager(request)
    insight_id = request.match_info["id"]
    item = await mgr.get(insight_id)
    if not item:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(item, dumps=_JSON_DUMPS)


# ── Create ───────────────────────────────────────────────────────────


async def create_insight(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/insights — create a new insight."""
    mgr = _resolve_manager(request)
    body = await request.json()
    content = str(body.get("content", "")).strip()
    if not content:
        return web.json_response({"error": "'content' is required"}, status=400)

    insight_id = await mgr.add(
        content=content,
        category=str(body.get("category", "fact")).strip(),
        entity_key=body.get("entity_key") or None,
        importance=int(body.get("importance", 5)),
        source_tool="web_ui",
        tags=body.get("tags") or None,
        expires_at=body.get("expires_at") or None,
    )
    if insight_id:
        item = await mgr.get(insight_id)
        return web.json_response(item, status=201, dumps=_JSON_DUMPS)
    return web.json_response({"message": "Deduped — similar insight exists"}, status=200)


# ── Update ───────────────────────────────────────────────────────────


async def update_insight(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/insights/{id} — update an insight."""
    mgr = _resolve_manager(request)
    insight_id = request.match_info["id"]
    body = await request.json()

    ok = await mgr.update(insight_id, **body)
    if not ok:
        return web.json_response({"error": "Not found or no changes"}, status=404)
    item = await mgr.get(insight_id)
    return web.json_response(item, dumps=_JSON_DUMPS)


# ── Delete ───────────────────────────────────────────────────────────


async def delete_insight(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/insights/{id} — delete an insight."""
    mgr = _resolve_manager(request)
    insight_id = request.match_info["id"]
    ok = await mgr.delete(insight_id)
    if not ok:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response({"ok": True})
