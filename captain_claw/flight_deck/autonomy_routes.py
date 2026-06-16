"""Flight Deck Autonomous Work API — the cockpit for the closed autonomy loop.

Scoped to the logged-in user. Phase 1 surface:

  * GET/PUT /fd/autonomy/config       — per-user effective config + overrides
  * GET     /fd/autonomy/actions       — the action ledger (the page's feed)
  * POST    /fd/autonomy/actions/{id}/approve|reject — resolve a pending action
  * GET     /fd/autonomy/reliability   — learned per-kind weights
  * POST    /fd/autonomy/nudge         — force one arbiter pass (no-op until Phase 2)

Approve/reject already move ledger rows; wiring them through to ``follow_through``
(intentions) and dispatch lands with Topics 1–3.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, status

from captain_claw.flight_deck.auth import get_optional_user
from captain_claw.flight_deck.autonomy import (
    global_defaults,
    get_store,
    resolve_config,
    save_config,
)

router = APIRouter(prefix="/fd/autonomy", tags=["autonomy"])


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _user_id(request: Request) -> str:
    """Resolve the caller's user id; enforce auth when enabled, else local bucket."""
    uid = getattr(request.state, "user_id", "") or ""
    if _auth_enabled() and not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Not authenticated")
    return uid


@router.get("/config")
async def get_config_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Effective config for this user plus the global defaults (so the UI can
    show what's overridden) and the shipped autonomy ceiling."""
    uid = _user_id(request)
    effective = resolve_config(uid)
    return {
        "config": effective,
        "defaults": global_defaults(),
        "max_autonomy_level": effective.get("max_autonomy_level", "propose"),
    }


@router.put("/config")
async def put_config_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Save per-user overrides. Body is a partial config dict; unknown and
    server-owned keys (the ceiling) are ignored. Returns the new effective config."""
    uid = _user_id(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    overrides = body.get("config") if isinstance(body.get("config"), dict) else body
    effective = save_config(uid, overrides)
    return {"config": effective, "defaults": global_defaults()}


@router.get("/actions")
async def list_actions_route(
    request: Request,
    status_filter: str | None = None,
    limit: int = 100,
    _user: dict | None = Depends(get_optional_user),
):
    """The action ledger — what the loop considered, queued, dispatched, or did."""
    uid = _user_id(request)
    return {"actions": get_store().list_actions(uid, status=status_filter, limit=limit)}


@router.post("/actions/{action_id}/approve")
async def approve_action_route(
    action_id: str,
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Approve a pending action. Phase 1 moves it to 'queued'; later phases hand
    it to dispatch / follow_through."""
    uid = _user_id(request)
    store = get_store()
    action = store.get_action(action_id)
    if not action or action.get("user_id") not in (uid, "local"):
        raise HTTPException(status_code=404, detail="Action not found")
    return {"action": store.update_status(action_id, "queued")}


@router.post("/actions/{action_id}/reject")
async def reject_action_route(
    action_id: str,
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Reject a pending action — a strong negative training signal for Topic 3."""
    uid = _user_id(request)
    store = get_store()
    action = store.get_action(action_id)
    if not action or action.get("user_id") not in (uid, "local"):
        raise HTTPException(status_code=404, detail="Action not found")
    return {"action": store.update_status(
        action_id, "rejected", outcome="fail", outcome_note="rejected by user")}


@router.get("/reliability")
async def reliability_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Learned reliability weights per action kind/domain."""
    uid = _user_id(request)
    return {"reliability": get_store().list_reliability(uid)}


@router.post("/nudge")
async def nudge_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Force one arbiter pass now. No-op in Phase 1 (the Arbiter lands in Phase 2)."""
    uid = _user_id(request)
    cfg = resolve_config(uid)
    return {"ok": True, "ran": False, "reason": "arbiter not enabled yet",
            "autonomy_level": cfg.get("autonomy_level", "off")}
