"""REST handlers for intentions — list + resolve pending decisions.

These are the deterministic surfaces (Phase 2b): the Flight Deck panel and
glasses card call them to render the list and to approve/decline/snooze with
a button, hitting the same ``follow_through`` path the agent uses for freeform
replies. The store lives in the agent process, so the agent owns these
endpoints and channels are thin clients.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from aiohttp import web

from captain_claw.intentions import (
    DECISION_RESOLUTIONS,
    follow_through,
    get_intentions_manager,
)
from captain_claw.logging import get_logger

if TYPE_CHECKING:
    from captain_claw.web_server import WebServer

log = get_logger(__name__)

# Accept friendly button verdicts in addition to canonical resolutions.
_VERDICT_ALIASES = {
    "yes": "approved", "approve": "approved",
    "no": "declined", "decline": "declined",
    "later": "snoozed", "snooze": "snoozed",
    "stop": "undone", "cancel": "undone",
}

# Statuses the Flight Deck panel can set directly on an intention (manual
# transitions — distinct from the decision/approval queue above). Friendly
# aliases map the button labels to canonical statuses.
_STATUS_ALIASES = {
    "resolve": "done", "resolved": "done", "complete": "done", "completed": "done",
    "cancel": "cancelled", "canceled": "cancelled",
    "activate": "active", "reopen": "active", "open": "active",
    "snooze": "snoozed", "decline": "declined", "dismiss": "declined",
}
_MANUAL_STATUSES = frozenset({"done", "cancelled", "active", "snoozed", "declined"})


async def list_intentions(server: "WebServer", request: web.Request) -> web.Response:
    """GET /api/intentions — open intentions (optionally ?status= / ?origin=)."""
    mgr = get_intentions_manager()
    status = (request.query.get("status") or "").strip() or None
    origin = (request.query.get("origin") or "").strip() or None
    try:
        limit = int(request.query.get("limit") or 50)
    except ValueError:
        limit = 50
    if status:
        items = await mgr.list(origin=origin, status=status, limit=limit)
    else:
        from captain_claw.intentions import OPEN_STATUSES
        items = await mgr.list(origin=origin, statuses=list(OPEN_STATUSES), limit=limit)
    return web.json_response({"intentions": items})


async def set_intention_status(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/intentions/{intention_id}/status — body {status, days?}.

    Directly transition an intention (done/cancelled/active/snoozed/declined)
    from the Flight Deck panel, without going through the approval queue.
    """
    intention_id = request.match_info.get("intention_id", "").strip()
    if not intention_id:
        return web.json_response({"error": "intention_id required"}, status=400)
    try:
        body = await request.json()
    except Exception:
        body = {}
    raw = str(body.get("status") or "").strip().lower()
    status = _STATUS_ALIASES.get(raw, raw)
    if status not in _MANUAL_STATUSES:
        return web.json_response(
            {"error": f"status must be one of {sorted(_MANUAL_STATUSES)} (or done/cancel/snooze/activate)"},
            status=400,
        )
    mgr = get_intentions_manager()
    cur = await mgr.get(intention_id)
    if not cur:
        return web.json_response({"error": "no such intention"}, status=404)
    extra: dict = {}
    now = datetime.now(UTC)
    if status == "snoozed":
        try:
            days = int(body.get("days") or 1)
        except (TypeError, ValueError):
            days = 1
        extra["next_surface_at"] = (now + timedelta(days=max(1, days))).isoformat()
    elif status in ("done", "cancelled", "declined"):
        extra["decided_at"] = now.isoformat()
    ok = await mgr.set_status(intention_id, status, **extra)
    if not ok:
        return web.json_response({"error": "update failed"}, status=500)
    return web.json_response({"ok": True, "id": intention_id, "status": status})


async def list_decisions(server: "WebServer", request: web.Request) -> web.Response:
    """GET /api/intentions/decisions — pending decisions awaiting a verdict."""
    mgr = get_intentions_manager()
    try:
        limit = int(request.query.get("limit") or 50)
    except ValueError:
        limit = 50
    decisions = await mgr.list_pending_decisions(limit=limit)
    return web.json_response({"decisions": decisions})


async def resolve_decision(server: "WebServer", request: web.Request) -> web.Response:
    """POST /api/intentions/decisions/{decision_id}/resolve — body {verdict, via}."""
    decision_id = request.match_info.get("decision_id", "").strip()
    if not decision_id:
        return web.json_response({"error": "decision_id required"}, status=400)
    try:
        body = await request.json()
    except Exception:
        body = {}
    raw = str(body.get("verdict") or body.get("resolution") or "").strip().lower()
    resolution = _VERDICT_ALIASES.get(raw, raw)
    if resolution not in DECISION_RESOLUTIONS:
        return web.json_response(
            {"error": "verdict must be approved/declined/snoozed/undone (or yes/no/later/stop)"},
            status=400,
        )
    via = str(body.get("via") or "flight_deck").strip()

    mgr = get_intentions_manager()
    dec = await mgr.resolve_decision(decision_id, resolution, via=via)
    if not dec:
        return web.json_response({"error": "no such pending decision"}, status=404)
    # source_waid: the decision may carry where it was surfaced.
    target = dec.get("target_hint") if isinstance(dec.get("target_hint"), dict) else {}
    source_waid = str((target or {}).get("waid") or "").strip()
    res = await follow_through(
        dec["intention_id"], resolution, source_waid=source_waid
    )
    status = 200 if res.get("ok") else 500
    return web.json_response({"resolution": resolution, **res}, status=status)
