"""REST handlers for conversation topics — the agent owns the topic store, so
the Flight Deck panel is a thin client that lists topics, opens one (summary +
message excerpts), and triggers a backfill over past messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.conversation_topics import (
    backfill_topics,
    get_topics_manager,
    refresh_topic,
)
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)


async def list_topics(server: "WebServer", request: web.Request) -> web.Response:
    """GET /api/topics — topics (optionally ?q= to search, ?limit=, ?order=recent|alpha)."""
    mgr = get_topics_manager()
    q = (request.query.get("q") or "").strip()
    order = (request.query.get("order") or "recent").strip()
    group = (request.query.get("group") or "").strip()
    tags = [t for t in (request.query.get("tags") or "").split(",") if t.strip()]
    try:
        limit = int(request.query.get("limit") or 300)
    except (ValueError, TypeError):
        limit = 300
    if q:
        topics = mgr.search_topics(q, limit=limit, order=order, group=group, tags=tags)
    else:
        topics = mgr.list_topics(limit=limit, order=order, group=group, tags=tags)
    return web.json_response({"topics": topics, "total": len(topics)})


async def list_groups(server: "WebServer", request: web.Request) -> web.Response:
    """GET /api/topics/groups — all groups with member counts."""
    return web.json_response({"groups": get_topics_manager().list_groups()})


async def create_group(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/groups — create a group. Body: {name}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    g = get_topics_manager().create_group(str((body or {}).get("name") or ""))
    if not g:
        return web.json_response({"error": "name required"}, status=400)
    return web.json_response({"ok": True, "group": g})


async def delete_group(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/groups/{group_id}/delete — remove a group (not its topics)."""
    ok = get_topics_manager().delete_group(request.match_info.get("group_id", ""))
    return web.json_response({"ok": ok})


async def set_topic_groups(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/{topic_id}/groups — set a topic's groups. Body: {group_ids}."""
    topic_id = request.match_info.get("topic_id", "")
    try:
        body = await request.json()
    except Exception:
        body = {}
    gids = [str(g) for g in ((body or {}).get("group_ids") or [])]
    groups = get_topics_manager().set_topic_groups(topic_id, gids)
    return web.json_response({"ok": True, "groups": groups})


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


async def refresh(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/{topic_id}/refresh — re-pull full message text from the
    live session into this topic's stored excerpts."""
    if not server.agent or not server.agent.session:
        return web.json_response({"error": "no active session"}, status=503)
    topic_id = request.match_info.get("topic_id", "")
    return web.json_response(refresh_topic(server.agent, topic_id))


async def append_turn(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/{topic_id}/append — persist a chat turn (user + agent
    messages) into a topic. Body: {messages: [{role, content}]}.

    Uses the REAL agent session msg_ids for this turn whenever we can match
    them, so the periodic classifier later dedups against these entries instead
    of duplicating the same content under a fresh msg_id."""
    import time as _time
    from captain_claw.conversation_topics import _utcnow
    topic_id = request.match_info.get("topic_id", "")
    try:
        body = await request.json()
    except Exception:
        body = {}
    # Find the just-finished turn's real session msg_ids (latest user + latest
    # assistant). We're called from finishTurn() immediately after the agent
    # commits its reply, so the tail of the session IS this turn.
    sess_msgs = []
    if server.agent and server.agent.session:
        sess_msgs = server.agent.session.messages or []
    real_user_id = ""
    real_agent_id = ""
    for m in reversed(sess_msgs):
        if not real_agent_id and m.get("role") == "assistant" and str(m.get("content") or "").strip():
            real_agent_id = str(m.get("message_id") or "")
        elif not real_user_id and m.get("role") == "user":
            real_user_id = str(m.get("message_id") or "")
        if real_user_id and real_agent_id:
            break

    base = int(_time.time() * 1000)
    items = []
    for i, m in enumerate((body or {}).get("messages") or []):
        content = str((m or {}).get("content") or "").strip()
        if not content:
            continue
        role = "user" if (m or {}).get("role") == "user" else "agent"
        # Prefer the real session msg_id (dedups with the periodic classifier);
        # fall back to a synthetic id only when the session doesn't have it yet.
        mid = (real_user_id if role == "user" else real_agent_id) or f"chat-{base}-{i}"
        items.append({"role": role, "channel": "chat", "excerpt": content,
                      "msg_id": mid, "ts": _utcnow()})
    if not items:
        return web.json_response({"ok": True, "added": 0})
    mgr = get_topics_manager()
    mgr.add_messages(topic_id, items, cap=500)  # chat turns: keep a long tail
    return web.json_response({"ok": True, "added": len(items), "topic": mgr.get_topic(topic_id, max_excerpts=200)})


async def move_message(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topic-message/{message_id}/move — move one message to another
    topic. Body: {target_topic_id}."""
    msg_id = request.match_info.get("message_id", "")
    try:
        body = await request.json()
    except Exception:
        body = {}
    target = str((body or {}).get("target_topic_id") or "").strip()
    if not target:
        return web.json_response({"ok": False, "error": "target_topic_id required"}, status=400)
    return web.json_response(get_topics_manager().move_message(msg_id, target))


async def unclassify(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/{topic_id}/unclassify — drop the topic and free its
    messages so the next backfill/classifier pass can redistribute them."""
    topic_id = request.match_info.get("topic_id", "")
    return web.json_response(get_topics_manager().unclassify_topic(topic_id))


async def star(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/{topic_id}/star — pin/unpin a topic. Body: {starred: bool}."""
    topic_id = request.match_info.get("topic_id", "")
    try:
        body = await request.json()
    except Exception:
        body = {}
    starred = bool((body or {}).get("starred", True))
    ok = get_topics_manager().set_star(topic_id, starred)
    return web.json_response({"ok": ok, "starred": starred})


async def reset(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/reset — wipe topics + backfill progress. Body
    ``{preserve_ids: [...]}`` (optional) keeps those topics untouched."""
    body: dict = {}
    try:
        if request.body_exists:
            body = await request.json()
    except Exception:
        body = {}
    preserve_ids = [str(x) for x in ((body or {}).get("preserve_ids") or [])]
    return web.json_response({"ok": True, **get_topics_manager().reset_all(preserve_ids)})


async def combine(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/topics/combine — merge sources into a target. Body:
    {target_id, source_ids: [...]}."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    target_id = str((body or {}).get("target_id") or "").strip()
    source_ids = [str(s).strip() for s in ((body or {}).get("source_ids") or []) if str(s).strip()]
    if not target_id or not source_ids:
        return web.json_response({"error": "target_id and source_ids are required"}, status=400)
    merged = get_topics_manager().combine_topics(target_id, source_ids)
    if not merged:
        return web.json_response({"error": "target topic not found"}, status=404)
    return web.json_response({"ok": True, "topic": merged})
