"""REST handlers for conversation topics — the agent owns the topic store, so
the Flight Deck panel is a thin client that lists topics, opens one (summary +
message excerpts), and triggers a backfill over past messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.conversation_topics import backfill_topics, get_topics_manager
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def list_topics(server: "WebServer", request: web.Request) -> web.Response:
    """GET /api/topics — recent topics (optionally ?q= to search, ?limit=)."""
    mgr = get_topics_manager()
    q = (request.query.get("q") or "").strip()
    try:
        limit = int(request.query.get("limit") or 60)
    except (ValueError, TypeError):
        limit = 60
    topics = mgr.search_topics(q, limit=limit) if q else mgr.list_topics(limit=limit)
    return web.json_response({"topics": topics})


async def get_topic(server: "WebServer", request: web.Request) -> web.Response:
    """GET /api/topics/{topic_id} — one topic with its message excerpts."""
    mgr = get_topics_manager()
    topic_id = request.match_info.get("topic_id", "")
    try:
        limit = int(request.query.get("limit") or 60)
    except (ValueError, TypeError):
        limit = 60
    t = mgr.get_topic(topic_id, max_excerpts=limit)
    if not t:
        return web.json_response({"error": "topic not found"}, status=404)
    return web.json_response({"topic": t})


async def backfill(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/backfill — classify untagged past messages. Body/query:
    {hours} (0 or omitted = all history)."""
    if not server.agent or not server.agent.session:
        return web.json_response({"error": "no active session"}, status=503)
    hours = 0
    try:
        if request.body_exists:
            body = await request.json()
            hours = int((body or {}).get("hours") or 0)
        else:
            hours = int(request.query.get("hours") or 0)
    except Exception:
        hours = 0
    result = await backfill_topics(server.agent, hours=hours)
    return web.json_response(result)
