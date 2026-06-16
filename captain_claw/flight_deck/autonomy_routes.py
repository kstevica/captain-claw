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
    record_human_feedback,
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
    """Approve a pending action: record the positive signal, then dispatch it to
    the user's strongest agent. Falls back to 'queued' if no agent is reachable."""
    uid = _user_id(request)
    store = get_store()
    action = store.get_action(action_id)
    if not action or action.get("user_id") not in (uid, "local"):
        raise HTTPException(status_code=404, detail="Action not found")
    learned = record_human_feedback(uid, action, True)

    from captain_claw.flight_deck.fd_dispatch import dispatch_action

    disp = await dispatch_action(uid, action)
    if not disp["ok"]:
        # No agent to run it — park as queued for a later pass.
        store.update_status(action_id, "queued", outcome_note=disp["note"])
    # On success dispatch_action already moved it to dispatched (and the async
    # judge will move it to done) — just return the current row.
    return {"action": store.get_action(action_id), "reliability": learned, "dispatch": disp}


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
    learned = record_human_feedback(uid, action, False)
    return {"action": store.update_status(
        action_id, "rejected", outcome="fail", outcome_note="rejected by user"),
        "reliability": learned}


@router.post("/actions/{action_id}/undo")
async def undo_action_route(
    action_id: str,
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Undo a completed reversible action by running its captured reverse call."""
    uid = _user_id(request)
    store = get_store()
    action = store.get_action(action_id)
    if not action or action.get("user_id") not in (uid, "local"):
        raise HTTPException(status_code=404, detail="Action not found")
    if not (action.get("payload") or {}).get("reverse"):
        raise HTTPException(status_code=400, detail="No reverse available for this action")
    from captain_claw.flight_deck.actions import undo_action
    res = await undo_action(uid, action)
    if res.get("ok"):
        store.update_status(action_id, "undone", outcome_note="undone by user")
    return {"action": store.get_action(action_id), "undo": res}


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
    """Force one heartbeat now (forces a reflection, which runs the Arbiter).
    Returns the arbiter outcome so the page can show what, if anything, it proposed."""
    uid = _user_id(request)
    cfg = resolve_config(uid)
    if not cfg.get("enabled"):
        return {"ok": True, "ran": False, "reason": "disabled",
                "autonomy_level": cfg.get("autonomy_level", "off")}
    store = get_store()
    try:
        from captain_claw.flight_deck.consciousness import pulse

        result = await pulse(uid, force=True)
    except Exception as exc:
        store.log(uid, "error: manual nudge failed", str(exc), "error")
        raise HTTPException(status_code=500, detail=f"pulse failed: {exc}") from exc
    arb = result.get("arbiter")
    # If the heartbeat bailed before the arbiter (no agents / nothing to think
    # with), the arbiter logs nothing — record it here so the nudge is explained.
    if not arb:
        store.log(uid, f"nudge: pulse {result.get('reason', '?')}",
                  "heartbeat returned before the arbiter ran (e.g. no running agent)", "warn")
    return {"ok": True, "pulse": result.get("reason"), "arbiter": arb,
            "autonomy_level": cfg.get("autonomy_level", "off")}


@router.get("/log")
async def log_route(
    request: Request,
    limit: int = 100,
    _user: dict | None = Depends(get_optional_user),
):
    """The live trace of what the loop did — arbiter passes, skips, dispatches,
    judge verdicts, and errors. Newest first."""
    uid = _user_id(request)
    return {"log": get_store().list_log(uid, limit=limit)}


@router.get("/plans")
async def list_plans_route(request: Request, status_filter: str | None = None,
                           limit: int = 50, _user: dict | None = Depends(get_optional_user)):
    """Active + past plans (#4) with their step progress."""
    uid = _user_id(request)
    from captain_claw.flight_deck.plans import get_store as plans_store
    return {"plans": plans_store().list_plans(uid, status=status_filter, limit=limit)}


@router.post("/plans")
async def create_plan_route(request: Request, _user: dict | None = Depends(get_optional_user)):
    """Decompose a goal into steps and create a plan. Body: {goal}."""
    uid = _user_id(request)
    body = await request.json()
    goal = str((body or {}).get("goal") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="goal is required")
    from captain_claw.flight_deck.plans import decompose_goal, get_store as plans_store
    steps = await decompose_goal(uid, goal)
    if not steps:
        raise HTTPException(status_code=422, detail="Could not decompose the goal into steps")
    plan = plans_store().create_plan(uid, goal, steps)
    return {"ok": True, "plan": plan}


@router.post("/plans/{plan_id}/advance")
async def advance_plan_route(plan_id: str, request: Request, _user: dict | None = Depends(get_optional_user)):
    """Run the plan's next step (manual advance = approval for that step)."""
    uid = _user_id(request)
    from captain_claw.flight_deck.plans import advance_one, get_store as plans_store
    res = await advance_one(uid, plan_id, auto=False)
    return {"result": res, "plan": plans_store().get_plan(plan_id)}


@router.post("/plans/{plan_id}/abandon")
async def abandon_plan_route(plan_id: str, request: Request, _user: dict | None = Depends(get_optional_user)):
    """Abandon a plan and roll back its completed reversible steps."""
    uid = _user_id(request)
    from captain_claw.flight_deck.plans import abandon_plan, get_store as plans_store
    res = await abandon_plan(uid, plan_id)
    return {"result": res, "plan": plans_store().get_plan(plan_id)}


@router.get("/catalog")
async def catalog_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """The action catalog (#1) — what the autonomous loop may do, with risk +
    reversibility. Phase 1: full catalog; grant-filtering lands with the grants UI."""
    from captain_claw.flight_deck.action_catalog import list_catalog
    _ = _user_id(request)
    return {"catalog": list_catalog()}


@router.post("/run-action")
async def run_action_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Phase-1 manual exerciser: run a catalog action directly. Body:
    {action_id, args}. Bypasses the arbiter — used to validate the rail before
    wiring it into autonomous dispatch."""
    uid = _user_id(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    action_id = str(body.get("action_id") or "").strip()
    args = body.get("args") if isinstance(body.get("args"), dict) else {}
    from captain_claw.flight_deck.actions import run_action
    result = await run_action(uid, action_id, args)
    return result
