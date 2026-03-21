"""REST handlers for the Nervous System (intuitions) browser UI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger
from captain_claw.nervous_system import get_nervous_system_manager, get_session_nervous_system_manager

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)


def _resolve_manager(request: web.Request) -> Any:
    """Return the appropriate NervousSystemManager for this request."""
    from captain_claw.web.public_auth import get_request_session_id
    is_public, session_id = get_request_session_id(request)
    if is_public and session_id:
        return get_session_nervous_system_manager(session_id)
    return get_nervous_system_manager()


# ── List / Search ────────────────────────────────────────────────────


async def list_intuitions(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/nervous-system — list or search intuitions.

    Query params:
      q             — FTS search query (optional)
      thread_type   — filter by type (optional)
      min_confidence — minimum confidence (optional, default 0.0)
      limit         — max results (default 50)
    """
    mgr = _resolve_manager(request)
    query = request.query.get("q", "").strip()
    thread_type = request.query.get("thread_type", "").strip() or None
    min_confidence = float(request.query.get("min_confidence", "0.0"))
    limit = min(int(request.query.get("limit", "50")), 200)

    if query:
        items = await mgr.search(query, thread_type=thread_type, limit=limit)
    else:
        items = await mgr.list_recent(limit=limit, thread_type=thread_type, min_confidence=min_confidence)

    total = await mgr.count()

    return web.json_response(
        {"items": items, "total": total},
        dumps=_JSON_DUMPS,
    )


async def get_intuition(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/nervous-system/{id} — get a single intuition."""
    mgr = _resolve_manager(request)
    intuition_id = request.match_info["id"]
    item = await mgr.get(intuition_id)
    if not item:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(item, dumps=_JSON_DUMPS)


# ── Create ───────────────────────────────────────────────────────────


async def create_intuition(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/nervous-system — create a new intuition manually."""
    mgr = _resolve_manager(request)
    body = await request.json()
    content = str(body.get("content", "")).strip()
    if not content:
        return web.json_response({"error": "'content' is required"}, status=400)

    intuition_id = await mgr.add(
        content=content,
        thread_type=str(body.get("thread_type", "association")).strip(),
        source_layers=body.get("source_layers") or [],
        source_ids=body.get("source_ids") or [],
        confidence=float(body.get("confidence", 0.5)),
        importance=int(body.get("importance", 5)),
        tags=body.get("tags") or None,
    )
    if intuition_id:
        item = await mgr.get(intuition_id)
        return web.json_response(item, status=201, dumps=_JSON_DUMPS)
    return web.json_response({"message": "Deduped — similar intuition exists"}, status=200)


# ── Update ───────────────────────────────────────────────────────────


async def update_intuition(server: WebServer, request: web.Request) -> web.Response:
    """PATCH /api/nervous-system/{id} — update an intuition."""
    mgr = _resolve_manager(request)
    intuition_id = request.match_info["id"]
    body = await request.json()

    ok = await mgr.update(intuition_id, **body)
    if not ok:
        return web.json_response({"error": "Not found or no changes"}, status=404)
    item = await mgr.get(intuition_id)
    return web.json_response(item, dumps=_JSON_DUMPS)


# ── Delete ───────────────────────────────────────────────────────────


async def delete_intuition(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/nervous-system/{id} — delete an intuition."""
    mgr = _resolve_manager(request)
    intuition_id = request.match_info["id"]
    ok = await mgr.delete(intuition_id)
    if not ok:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response({"ok": True})


# ── Dream trigger ────────────────────────────────────────────────────


async def trigger_dream(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/nervous-system/dream — manually trigger a dream cycle."""
    from captain_claw.nervous_system import dream

    agent = getattr(server, "agent", None)
    if not agent:
        return web.json_response({"error": "No active agent"}, status=503)

    try:
        results = await dream(agent)
        return web.json_response(
            {"stored": len(results), "intuitions": results},
            dumps=_JSON_DUMPS,
        )
    except Exception as exc:
        log.error("Manual dream trigger failed", error=str(exc))
        return web.json_response({"error": str(exc)}, status=500)


# ── Stats ────────────────────────────────────────────────────────────


async def get_stats(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/nervous-system/stats — aggregate statistics."""
    mgr = _resolve_manager(request)
    data = await mgr.stats()
    return web.json_response(data, dumps=_JSON_DUMPS)
