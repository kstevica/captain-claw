"""REST handler for the Brain Graph visualization — aggregates all cognitive
data sources into a unified {nodes, links} structure for 3d-force-graph."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

_JSON_DUMPS = lambda obj: json.dumps(obj, default=str)

# ── Node colors ──────────────────────────────────────────────────────

COLORS = {
    "insight":   "#FFD700",  # gold
    "intuition": "#9B59B6",  # purple
    "tension":   "#E74C3C",  # red
    "task":      "#3498DB",  # blue
    "briefing":  "#2ECC71",  # green
    "todo":      "#1ABC9C",  # teal
    "contact":   "#F39C12",  # orange
    "session":   "#95A5A6",  # gray
    "event":     "#00BCD4",  # cyan
    "message":   "#7CB9E8",  # light blue
}

LINK_COLORS = {
    "spawned":    "#FFD700",
    "supersedes": "#E67E22",
    "resolves":   "#2ECC71",
    "triggers":   "#3498DB",
    "parent":     "#1ABC9C",
    "mentions":   "#F39C12",
    "contains":   "#95A5A6",
    "source":     "#9B59B6",
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ── Main handler ─────────────────────────────────────────────────────


def _resolve_public_session(request: web.Request) -> tuple[bool, str | None]:
    """Check if this is a public session request and return session_id."""
    try:
        from captain_claw.web.public_auth import get_request_session_id
        return get_request_session_id(request)
    except Exception:
        return False, None


async def get_graph_data(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/brain-graph — return full graph as {nodes, links}.

    Query params:
      limit  — max nodes per type (default 100, max 500)
      types  — comma-separated node types to include (default: all)
      since  — ISO timestamp, only items created after this time
    """
    limit = min(int(request.query.get("limit", "100")), 500)
    type_filter = request.query.get("types", "").strip()
    since = request.query.get("since", "").strip() or None
    want_types = set(type_filter.split(",")) if type_filter else None

    is_public, public_session_id = _resolve_public_session(request)

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    node_ids: set[str] = set()  # track existing node IDs for edge validation

    def _want(t: str) -> bool:
        return want_types is None or t in want_types

    # ── 1. Sessions (group containers) ────────────────────────────
    if _want("session"):
        try:
            from captain_claw.session import get_session_manager
            sm = get_session_manager()
            if is_public and public_session_id:
                # Public mode: only show this user's session.
                pub_session = await sm.load_session(public_session_id)
                sessions = [pub_session] if pub_session else []
            else:
                sessions = await sm.list_sessions(limit=limit)
            for s in sessions:
                nid = f"session_{s.id}"
                msgs = s.messages or []
                msg_count = len(msgs)
                nodes.append({
                    "id": nid,
                    "type": "session",
                    "label": s.name or s.id[:8],
                    "group": None,
                    "importance": 5,
                    "confidence": 1.0,
                    "status": "active",
                    "created_at": s.created_at,
                    "color": COLORS["session"],
                    "size": _clamp(2 + msg_count * 0.1, 2, 8),
                    "meta": {"message_count": msg_count},
                })
                node_ids.add(nid)

                # Add message nodes (user + assistant only, skip system/tool).
                prev_msg_nid: str | None = None
                for msg_idx, msg in enumerate(msgs):
                    role = msg.get("role", "")
                    if role not in ("user", "assistant"):
                        continue
                    mid = msg.get("message_id") or f"{s.id[:8]}_{msg_idx}"
                    msg_nid = f"msg_{mid}"
                    full_content = str(msg.get("content") or "")
                    label_preview = full_content[:80]
                    msg_color = "#E8D44D" if role == "user" else "#7CB9E8"
                    nodes.append({
                        "id": msg_nid,
                        "type": "message",
                        "label": f"[{role}] {label_preview}",
                        "group": nid,
                        "importance": 3,
                        "confidence": 1.0,
                        "status": role,
                        "created_at": msg.get("timestamp"),
                        "color": msg_color,
                        "size": 1.5,
                        "meta": {
                            "role": role,
                            "content": full_content,
                            "model": msg.get("model", ""),
                        },
                    })
                    node_ids.add(msg_nid)

                    # Edge: session contains message.
                    links.append({
                        "source": nid, "target": msg_nid,
                        "type": "contains", "color": LINK_COLORS["contains"],
                        "strength": 0.4,
                    })

                    # Edge: sequential message chain.
                    if prev_msg_nid:
                        links.append({
                            "source": prev_msg_nid, "target": msg_nid,
                            "type": "sequence", "color": "#555555",
                            "strength": 0.6,
                        })
                    prev_msg_nid = msg_nid
        except Exception as e:
            log.warning("Brain graph: sessions failed", error=str(e))

    # ── 2. Insights ───────────────────────────────────────────────
    if _want("insight"):
        try:
            from captain_claw.insights import get_insights_manager, get_session_insights_manager
            mgr = get_session_insights_manager(public_session_id) if is_public and public_session_id else get_insights_manager()
            items = await mgr.list_recent(limit=limit)
            for item in items:
                nid = f"insight_{item['id']}"
                imp = int(item.get("importance", 5))
                nodes.append({
                    "id": nid,
                    "type": "insight",
                    "label": (item.get("content") or "")[:80],
                    "group": f"session_{item['source_session']}" if item.get("source_session") else None,
                    "importance": imp,
                    "confidence": 1.0,
                    "status": item.get("category", "fact"),
                    "created_at": item.get("created_at"),
                    "color": COLORS["insight"],
                    "size": _clamp(2 + imp * 0.4, 2, 7),
                    "meta": {
                        "category": item.get("category"),
                        "entity_key": item.get("entity_key"),
                        "tags": item.get("tags"),
                        "content": item.get("content"),
                    },
                })
                node_ids.add(nid)

                # Edge: insight → session (spawned)
                if item.get("source_session"):
                    sid = f"session_{item['source_session']}"
                    links.append({
                        "source": sid, "target": nid,
                        "type": "contains", "color": LINK_COLORS["contains"],
                        "strength": 0.3,
                    })

                # Edge: insight supersedes older insight
                if item.get("supersedes_id"):
                    old_nid = f"insight_{item['supersedes_id']}"
                    links.append({
                        "source": nid, "target": old_nid,
                        "type": "supersedes", "color": LINK_COLORS["supersedes"],
                        "strength": 0.6,
                    })
        except Exception as e:
            log.warning("Brain graph: insights failed", error=str(e))

    # ── 3. Intuitions ─────────────────────────────────────────────
    if _want("intuition"):
        try:
            from captain_claw.nervous_system import get_nervous_system_manager, get_session_nervous_system_manager
            mgr = get_session_nervous_system_manager(public_session_id) if is_public and public_session_id else get_nervous_system_manager()
            items = await mgr.list_recent(limit=limit)
            for item in items:
                thread_type = item.get("thread_type", "association")
                is_tension = thread_type == "unresolved" or item.get("resolution_state") == "open"
                node_type = "tension" if is_tension else "intuition"
                nid = f"intuition_{item['id']}"
                conf = float(item.get("confidence", 0.5))
                imp = int(item.get("importance", 5))
                maturation = item.get("maturation_state", "mature")

                nodes.append({
                    "id": nid,
                    "type": node_type,
                    "label": (item.get("content") or "")[:80],
                    "group": f"session_{item['source_session']}" if item.get("source_session") else None,
                    "importance": imp,
                    "confidence": conf,
                    "status": maturation,
                    "created_at": item.get("created_at"),
                    "color": COLORS[node_type],
                    "size": _clamp(2 + conf * 4 + imp * 0.2, 2, 8),
                    "meta": {
                        "thread_type": thread_type,
                        "resolution_state": item.get("resolution_state"),
                        "maturation_state": maturation,
                        "dream_cycles_seen": item.get("dream_cycles_seen"),
                        "source_trigger": item.get("source_trigger"),
                        "content": item.get("content"),
                    },
                })
                node_ids.add(nid)

                # Edge: intuition → session (contains)
                if item.get("source_session"):
                    sid = f"session_{item['source_session']}"
                    links.append({
                        "source": sid, "target": nid,
                        "type": "contains", "color": LINK_COLORS["contains"],
                        "strength": 0.3,
                    })

                # Edge: resolved tension → source tension
                if item.get("resolved_from_id"):
                    src_nid = f"intuition_{item['resolved_from_id']}"
                    links.append({
                        "source": nid, "target": src_nid,
                        "type": "resolves", "color": LINK_COLORS["resolves"],
                        "strength": 0.8,
                    })

                # Edge: source_ids → linked insights/intuitions
                source_ids = item.get("source_ids") or []
                if isinstance(source_ids, str):
                    try:
                        source_ids = json.loads(source_ids)
                    except (json.JSONDecodeError, ValueError):
                        source_ids = []
                for sid in source_ids:
                    if sid:
                        # Could be insight or intuition — try both prefixes
                        links.append({
                            "source": f"insight_{sid}", "target": nid,
                            "type": "source", "color": LINK_COLORS["source"],
                            "strength": 0.5,
                        })
        except Exception as e:
            log.warning("Brain graph: intuitions failed", error=str(e))

    # ── 4. Sister Session Tasks ───────────────────────────────────
    if _want("task"):
        try:
            from captain_claw.sister_session import get_sister_session_manager, get_session_sister_manager
            ss_mgr = get_session_sister_manager(public_session_id) if is_public and public_session_id else get_sister_session_manager()
            tasks = await ss_mgr.list_tasks(parent_session_id=public_session_id if is_public and public_session_id else None, limit=limit)
            for t in tasks:
                nid = f"task_{t['id']}"
                priority = int(t.get("priority", 5))
                nodes.append({
                    "id": nid,
                    "type": "task",
                    "label": (t.get("trigger_reason") or t.get("prompt") or "")[:80],
                    "group": f"session_{t['parent_session_id']}" if t.get("parent_session_id") else None,
                    "importance": priority,
                    "confidence": float(t.get("confidence", 0.5)),
                    "status": t.get("status", "queued"),
                    "created_at": t.get("created_at"),
                    "color": COLORS["task"],
                    "size": _clamp(2 + priority * 0.3, 2, 6),
                    "meta": {
                        "source_type": t.get("source_type"),
                        "status": t.get("status"),
                        "result_summary": t.get("result_summary"),
                    },
                })
                node_ids.add(nid)

                # Edge: task → source insight/intuition (triggers)
                source_type = t.get("source_type", "")
                source_id = t.get("source_id", "")
                if source_id:
                    if source_type == "intuition":
                        src_nid = f"intuition_{source_id}"
                    elif source_type in ("insight", "deadline"):
                        src_nid = f"insight_{source_id}"
                    else:
                        src_nid = f"{source_type}_{source_id}"
                    links.append({
                        "source": src_nid, "target": nid,
                        "type": "triggers", "color": LINK_COLORS["triggers"],
                        "strength": 0.7,
                    })
        except Exception as e:
            log.warning("Brain graph: tasks failed", error=str(e))

    # ── 5. Briefings ──────────────────────────────────────────────
    if _want("briefing"):
        try:
            from captain_claw.sister_session import get_sister_session_manager, get_session_sister_manager
            ss_mgr = get_session_sister_manager(public_session_id) if is_public and public_session_id else get_sister_session_manager()
            briefings = await ss_mgr.list_briefings(parent_session_id=public_session_id if is_public and public_session_id else None, limit=limit)
            for b in briefings:
                nid = f"briefing_{b['id']}"
                nodes.append({
                    "id": nid,
                    "type": "briefing",
                    "label": (b.get("summary") or "")[:80],
                    "group": f"session_{b['parent_session_id']}" if b.get("parent_session_id") else None,
                    "importance": 5,
                    "confidence": float(b.get("confidence", 0.5)),
                    "status": b.get("status", "unread"),
                    "created_at": b.get("created_at"),
                    "color": COLORS["briefing"],
                    "size": 3,
                    "meta": {
                        "actionable": b.get("actionable"),
                        "summary": b.get("summary"),
                    },
                })
                node_ids.add(nid)
        except Exception as e:
            log.warning("Brain graph: briefings failed", error=str(e))

    # ── 6. Todos ──────────────────────────────────────────────────
    if _want("todo"):
        try:
            from captain_claw.session import get_session_manager
            sm = get_session_manager()
            todos = await sm.list_todos(limit=limit, session_filter=public_session_id if is_public and public_session_id else None)
            for td in todos:
                nid = f"todo_{td.id}"
                pri_map = {"low": 2, "normal": 5, "high": 7, "urgent": 9}
                pri_val = pri_map.get(td.priority, 5)
                nodes.append({
                    "id": nid,
                    "type": "todo",
                    "label": (td.content or "")[:80],
                    "group": f"session_{td.source_session}" if td.source_session else None,
                    "importance": pri_val,
                    "confidence": 1.0,
                    "status": td.status,
                    "created_at": td.created_at,
                    "color": COLORS["todo"],
                    "size": _clamp(2 + pri_val * 0.2, 2, 5),
                    "meta": {
                        "responsible": td.responsible,
                        "priority": td.priority,
                        "content": td.content,
                    },
                })
                node_ids.add(nid)

                # Edge: todo → parent todo
                if td.parent_id:
                    links.append({
                        "source": f"todo_{td.parent_id}", "target": nid,
                        "type": "parent", "color": LINK_COLORS["parent"],
                        "strength": 0.6,
                    })

                # Edge: todo → triggering insight/intuition
                if td.triggered_by_id:
                    links.append({
                        "source": td.triggered_by_id, "target": nid,
                        "type": "triggers", "color": LINK_COLORS["triggers"],
                        "strength": 0.5,
                    })
        except Exception as e:
            log.warning("Brain graph: todos failed", error=str(e))

    # ── 7. Contacts (skip in public mode — global, not session-scoped)
    if _want("contact") and not (is_public and public_session_id):
        try:
            from captain_claw.session import get_session_manager
            sm = get_session_manager()
            contacts = await sm.list_contacts(limit=limit)
            for c in contacts:
                nid = f"contact_{c.id}"
                nodes.append({
                    "id": nid,
                    "type": "contact",
                    "label": c.name or c.id[:8],
                    "group": None,
                    "importance": c.importance,
                    "confidence": 1.0,
                    "status": c.relation or "unknown",
                    "created_at": c.created_at,
                    "color": COLORS["contact"],
                    "size": _clamp(2 + c.mention_count * 0.3, 2, 7),
                    "meta": {
                        "organization": c.organization,
                        "position": c.position,
                        "email": c.email,
                        "mention_count": c.mention_count,
                    },
                })
                node_ids.add(nid)
        except Exception as e:
            log.warning("Brain graph: contacts failed", error=str(e))

    # ── 8. Cognitive Events ───────────────────────────────────────
    if _want("event"):
        try:
            from captain_claw.cognitive_metrics import get_cognitive_metrics_manager
            cm = get_cognitive_metrics_manager()
            events = await cm.query_events(
                session_id=public_session_id if is_public and public_session_id else None,
                limit=min(limit, 50),
            )
            for ev in events:
                nid = f"event_{ev['id']}"
                nodes.append({
                    "id": nid,
                    "type": "event",
                    "label": f"{ev.get('feature', '')}:{ev.get('event_type', '')}",
                    "group": f"session_{ev['session_id']}" if ev.get("session_id") else None,
                    "importance": 3,
                    "confidence": 1.0,
                    "status": ev.get("event_type", ""),
                    "created_at": ev.get("created_at"),
                    "color": COLORS["event"],
                    "size": 1.5,
                    "meta": {
                        "event_type": ev.get("event_type"),
                        "feature": ev.get("feature"),
                    },
                })
                node_ids.add(nid)

                # Edge: event → session
                if ev.get("session_id"):
                    sid = f"session_{ev['session_id']}"
                    links.append({
                        "source": sid, "target": nid,
                        "type": "contains", "color": LINK_COLORS["contains"],
                        "strength": 0.2,
                    })
        except Exception as e:
            log.warning("Brain graph: cognitive events failed", error=str(e))

    # ── Prune orphan links ────────────────────────────────────────
    # Remove links where source or target node doesn't exist.
    valid_links = [
        lnk for lnk in links
        if lnk["source"] in node_ids and lnk["target"] in node_ids
    ]

    # ── Stats ─────────────────────────────────────────────────────
    type_counts: dict[str, int] = {}
    for n in nodes:
        t = n["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return web.json_response(
        {
            "nodes": nodes,
            "links": valid_links,
            "stats": {
                "total_nodes": len(nodes),
                "total_links": len(valid_links),
                "type_counts": type_counts,
            },
        },
        dumps=_JSON_DUMPS,
    )


# ── WebSocket broadcast helper ───────────────────────────────────────


async def get_message_content(server: WebServer, request: web.Request) -> web.Response:
    """GET /api/brain-graph/message/{msg_id} — fetch full message content by message_id.

    Searches across all sessions for a message with the given message_id.
    Returns the full content for rendering in a modal.
    """
    msg_id = request.match_info.get("msg_id", "")
    if not msg_id:
        return web.json_response({"error": "msg_id required"}, status=400)

    is_public, public_session_id = _resolve_public_session(request)

    try:
        from captain_claw.session import get_session_manager
        sm = get_session_manager()

        # Public mode: only search the user's own session.
        if is_public and public_session_id:
            pub_session = await sm.load_session(public_session_id)
            sessions = [pub_session] if pub_session else []
        else:
            sessions = await sm.list_sessions(limit=50)

        for s in sessions:
            for msg in (s.messages or []):
                mid = msg.get("message_id") or ""
                # Also match fallback IDs like "sessionid_index"
                if mid == msg_id or f"msg_{mid}" == msg_id:
                    return web.json_response({
                        "content": str(msg.get("content") or ""),
                        "role": msg.get("role", ""),
                        "timestamp": msg.get("timestamp", ""),
                        "model": msg.get("model", ""),
                        "message_id": mid,
                        "session_name": s.name,
                    }, dumps=_JSON_DUMPS)

        # Fallback: try matching by fallback ID pattern "sessionprefix_index"
        for s in sessions:
            prefix = s.id[:8]
            for idx, msg in enumerate(s.messages or []):
                fallback_id = f"{prefix}_{idx}"
                if fallback_id == msg_id or f"msg_{fallback_id}" == msg_id:
                    return web.json_response({
                        "content": str(msg.get("content") or ""),
                        "role": msg.get("role", ""),
                        "timestamp": msg.get("timestamp", ""),
                        "model": msg.get("model", ""),
                        "message_id": fallback_id,
                        "session_name": s.name,
                    }, dumps=_JSON_DUMPS)

        return web.json_response({"error": "message not found"}, status=404)
    except Exception as e:
        log.warning("Brain graph: message fetch failed", error=str(e))
        return web.json_response({"error": str(e)}, status=500)


def broadcast_graph_nodes(
    agent: Any,
    stored: list[dict[str, Any]],
    *,
    node_type: str = "insight",
) -> None:
    """Broadcast new nodes to all connected Brain Graph clients via WebSocket.

    Called from insight/intuition storage hooks.  Non-blocking, fire-and-forget.
    """
    _broadcast = getattr(agent, "_broadcast_ws", None)
    if _broadcast is None:
        # Try to get it from the web server reference.
        ws_ref = getattr(agent, "_web_server", None)
        if ws_ref is not None:
            _broadcast = getattr(ws_ref, "_broadcast", None)
    if not callable(_broadcast):
        return  # No web server available (CLI mode).

    for item in stored:
        content = str(item.get("content", ""))
        item_id = item.get("id", "")
        color = COLORS.get(node_type, "#888")

        # Determine sizing.
        if node_type == "insight":
            imp = int(item.get("importance", 5))
            size = _clamp(2 + imp * 0.4, 2, 7)
        elif node_type in ("intuition", "tension"):
            conf = float(item.get("confidence", 0.5))
            imp = int(item.get("importance", 5))
            is_tension = item.get("thread_type") == "unresolved"
            if is_tension:
                node_type = "tension"
                color = COLORS["tension"]
            size = _clamp(2 + conf * 4 + imp * 0.2, 2, 8)
        else:
            size = 3

        node = {
            "id": f"{node_type}_{item_id}",
            "type": node_type,
            "label": content[:80],
            "group": None,
            "importance": int(item.get("importance", 5)),
            "confidence": float(item.get("confidence", 1.0)),
            "status": item.get("maturation_state") or item.get("category") or "",
            "created_at": item.get("created_at"),
            "color": color,
            "size": size,
            "meta": {"content": content},
        }

        _broadcast({
            "type": "brain_graph_update",
            "action": "add_node",
            "node": node,
            "links": [],
        })
