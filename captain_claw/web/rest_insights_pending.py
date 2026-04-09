"""REST endpoints for the pending-review insights queue.

Items land here only when ``import_items(stage_conflicts=True)`` detects a
conflict between an incoming decision/preference/workflow insight and an
existing one. A human approves or rejects each row; approval optionally
supersedes the conflicting live insight.

Endpoints
---------
GET    /api/insights/pending                — list staged conflicts
GET    /api/insights/pending/count          — count only (cheap poll)
POST   /api/insights/pending/{id}/approve   — promote to live insights
POST   /api/insights/pending/{id}/reject    — drop without promoting
DELETE /api/insights/pending                — clear queue (dangerous)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.insights import (
    get_insights_manager,
    get_session_insights_manager,
)
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


def _resolve_manager(request: web.Request) -> Any:
    """Same per-session resolution rule as the other insights endpoints."""
    from captain_claw.web.public_auth import get_request_session_id

    is_public, session_id = get_request_session_id(request)
    if is_public and session_id:
        return get_session_insights_manager(session_id)
    return get_insights_manager()


# ── List / count ────────────────────────────────────────────────────


async def list_pending(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/insights/pending — list staged conflicts, newest first.

    Optional query params:
      category — filter by insight category
      limit    — int, default 100
    """
    category = request.query.get("category") or None
    limit = int(request.query.get("limit", "100"))
    mgr = _resolve_manager(request)
    items = await mgr.list_pending(category=category, limit=limit)
    total = await mgr.count_pending()
    return web.json_response(
        {
            "items": items,
            "returned": len(items),
            "total": total,
        }
    )


async def count_pending(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/insights/pending/count — cheap poll for the UI badge."""
    mgr = _resolve_manager(request)
    total = await mgr.count_pending()
    return web.json_response({"total": total})


# ── Approve / reject ────────────────────────────────────────────────


async def approve_pending(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/insights/pending/{pending_id}/approve.

    Optional query param ``supersede`` (default 1): when true and the pending
    row was staged because of a collision with a live insight, that live
    insight is deleted first so the approved value replaces it.
    """
    pending_id = request.match_info.get("pending_id", "").strip()
    if not pending_id:
        return web.json_response({"error": "Missing pending_id"}, status=400)

    supersede = request.query.get("supersede", "1").lower() not in ("0", "false", "no")
    mgr = _resolve_manager(request)
    result = await mgr.approve_pending(pending_id, supersede=supersede)
    if result is None:
        return web.json_response({"error": "Pending insight not found"}, status=404)
    return web.json_response({"ok": True, "insight": result})


async def reject_pending(server: WebServer, request: web.Request) -> web.Response:
    """POST /api/insights/pending/{pending_id}/reject — drop a staged item."""
    pending_id = request.match_info.get("pending_id", "").strip()
    if not pending_id:
        return web.json_response({"error": "Missing pending_id"}, status=400)

    mgr = _resolve_manager(request)
    found = await mgr.reject_pending(pending_id)
    if not found:
        return web.json_response({"error": "Pending insight not found"}, status=404)
    return web.json_response({"ok": True})


async def clear_pending(server: WebServer, request: web.Request) -> web.Response:
    """DELETE /api/insights/pending — wipe the entire staging queue."""
    mgr = _resolve_manager(request)
    deleted = await mgr.clear_pending()
    return web.json_response({"ok": True, "deleted": deleted})
