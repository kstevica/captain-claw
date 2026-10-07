"""Flight Deck Autonomous Work API — the cockpit for the closed autonomy loop.

Scoped to the logged-in user. Phase 1 surface:

  * GET/PUT /fd/autonomy/config       — per-user effective config + overrides
  * GET     /fd/autonomy/actions       — the action ledger (the page's feed)
  * POST    /fd/autonomy/actions/{id}/approve|reject — resolve a pending action
  * GET     /fd/autonomy/reliability   — learned per-kind weights
  * POST    /fd/autonomy/nudge         — force one arbiter pass (no-op until Phase 2)
  * GET/PUT /fd/autonomy/whatsapp      — this user's WhatsApp nudge number (+ POST …/test)

Approve/reject already move ledger rows; wiring them through to ``follow_through``
(intentions) and dispatch lands with Topics 1–3.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from captain_claw.flight_deck.auth import get_optional_user
from captain_claw.flight_deck.autonomy import (
    _norm_user,
    global_defaults,
    get_store,
    record_human_feedback,
    resolve_config,
    save_config,
)

router = APIRouter(prefix="/fd/autonomy", tags=["autonomy"])
_log = logging.getLogger(__name__)


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _user_id(request: Request) -> str:
    """Resolve the caller's user id; enforce auth when enabled, else local bucket."""
    uid = getattr(request.state, "user_id", "") or ""
    if _auth_enabled() and not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Not authenticated")
    return uid


def _public_defaults() -> dict[str, Any]:
    """The global defaults as shown to a user. With auth on a deck-wide WhatsApp
    number is someone's personal phone and applies to nobody (see
    ``resolve_config``), so it is blanked rather than shown to everyone."""
    defaults = global_defaults()
    if _auth_enabled():
        defaults["notify_waid"] = ""
    return defaults


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
        "defaults": _public_defaults(),
        "max_autonomy_level": effective.get("max_autonomy_level", "propose"),
    }


@router.put("/config")
async def put_config_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Save per-user overrides. Body is a partial config dict; unknown and
    server-owned keys (the ceiling) are ignored. Returns the new effective config.

    The WhatsApp keys are owned by PUT /fd/autonomy/whatsapp (validated there):
    a whole-config save from the Autonomous Work page — its Reset, or a stale
    tab — keeps the stored number and switch instead of wiping or overwriting them."""
    uid = _user_id(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    overrides = dict(body.get("config") if isinstance(body.get("config"), dict) else body)
    stored = get_store().get_overrides(uid)
    for key in _WHATSAPP_KEYS:
        overrides.pop(key, None)
        if key in stored:
            overrides[key] = stored[key]
    effective = save_config(uid, overrides)
    return {"config": effective, "defaults": _public_defaults()}


# ── WhatsApp nudge delivery (the Connections card) ──────────────────

_WHATSAPP_KEYS = ("notify_waid", "nudge_to_whatsapp")
_MAX_NOTIFY_WAIDS = 3
_TEST_COOLDOWN_S = 60.0
_last_test_at: dict[str, float] = {}  # user → monotonic time of their last test send
# Number changes per user per hour — each attempt answers "can this number be
# used?", so cap how fast anyone can ask.
_MAX_NUMBER_CHANGES_PER_HOUR = 10
_number_changes: dict[str, list[float]] = {}

# One message for "not on the allowlist" and "linked to another account", so the
# form can't be used to tell the two apart.
_UNAVAILABLE = ("That number can't be used for your nudges — it isn't on this deck's "
                "WhatsApp allowlist, or it's linked to another account. Ask an admin.")


def _parse_waids(raw: Any) -> list[str]:
    """Digits-only numbers from a comma list (``"+385 91 123-4567"`` → ``"385911234567"``)."""
    out: list[str] = []
    for part in str(raw or "").split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits and digits not in out:
            out.append(digits)
    return out


def _validated_waids(raw: Any) -> list[str]:
    """``_parse_waids`` for user input: a part with no digits is an error, not
    silently dropped (which would clear the binding), and the list is capped."""
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="notify_waid must be a string")
    for part in raw.split(","):
        if part.strip() and not any(ch.isdigit() for ch in part):
            raise HTTPException(status_code=400, detail=f"'{part.strip()[:40]}' isn't a phone number.")
    waids = _parse_waids(raw)
    if len(waids) > _MAX_NOTIFY_WAIDS:
        raise HTTPException(status_code=400, detail=f"At most {_MAX_NOTIFY_WAIDS} numbers.")
    return waids


async def _holder_of(uid: str, waids: list[str]) -> str:
    """Another current user who already receives nudges on one of ``waids`` ('' if
    none). The auth-off ``local`` bucket and deleted users don't hold numbers."""
    from captain_claw.flight_deck.auth import get_db

    me = _norm_user(uid)
    wanted = set(waids)
    for other, ov in get_store().all_overrides().items():
        if other in (me, "local") or not wanted & set(_parse_waids(ov.get("notify_waid"))):
            continue
        try:
            if await get_db().get_user_by_id(other):
                return other
        except Exception:  # noqa: BLE001 — can't tell: err on the side of holding
            return other
    return ""


def _number_change_allowed(uid: str) -> bool:
    """Sliding one-hour window of number changes for this user."""
    key = _norm_user(uid)
    now = time.monotonic()
    recent = [t for t in _number_changes.get(key, []) if now - t < 3600.0]
    if len(recent) >= _MAX_NUMBER_CHANGES_PER_HOUR:
        _number_changes[key] = recent
        return False
    _number_changes[key] = recent + [now]
    return True


def _whatsapp_state(uid: str) -> dict[str, Any]:
    """Where this user's nudges go right now — never the deck's allowlist itself."""
    from captain_claw.flight_deck.fd_dispatch import _nudge_waids
    from captain_claw.flight_deck.whatsapp_bridge import _allowed_waids, _env, _send_url

    cfg = resolve_config(uid)
    recipients, issue = _nudge_waids(cfg)
    return {
        # Without send credentials a push silently no-ops, so "configured"
        # needs them as well as the allowlist.
        "bridge_configured": bool(_allowed_waids() and _env("WHATSAPP_ACCESS_TOKEN") and _send_url()),
        "auth_enabled": _auth_enabled(),
        "autonomy_enabled": bool(cfg.get("enabled")),
        # The arbiter only runs (and so only nudges) with all three.
        "autonomy_active": bool(cfg.get("enabled") and cfg.get("arbiter_on_pulse")
                                and str(cfg.get("autonomy_level") or "off") != "off"),
        "nudge_to_whatsapp": bool(cfg.get("nudge_to_whatsapp", True)),
        "notify_waid": str(cfg.get("notify_waid") or ""),
        "recipients": recipients,
        "issue": issue,
    }


@router.get("/whatsapp")
async def get_whatsapp_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """This user's WhatsApp nudge binding and where nudges would be delivered."""
    return _whatsapp_state(_user_id(request))


@router.put("/whatsapp")
async def put_whatsapp_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Set ``notify_waid`` and/or ``nudge_to_whatsapp``, merged into the user's
    other overrides. Each number must be on the bridge allowlist and not already
    another user's (with auth on)."""
    from captain_claw.flight_deck.whatsapp_bridge import _allowed_waids

    uid = _user_id(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    overrides = get_store().get_overrides(uid)
    if "notify_waid" in body:
        waids = _validated_waids(body.get("notify_waid"))
        if ",".join(waids) != str(overrides.get("notify_waid") or ""):
            if not _number_change_allowed(uid):
                raise HTTPException(status_code=429,
                                    detail="Too many number changes — try again in an hour.")
            allowed = _allowed_waids()
            if waids and not allowed:
                raise HTTPException(status_code=400, detail="WhatsApp isn't set up on this deck.")
            if any(w not in allowed for w in waids):
                raise HTTPException(status_code=400, detail=_UNAVAILABLE)
            holder = await _holder_of(uid, waids) if (_auth_enabled() and waids) else ""
            if holder:
                _log.warning("WhatsApp nudge number refused for %s: already linked to user %s",
                             _norm_user(uid), holder)
                raise HTTPException(status_code=400, detail=_UNAVAILABLE)
        overrides["notify_waid"] = ",".join(waids)
    if "nudge_to_whatsapp" in body:
        if not isinstance(body.get("nudge_to_whatsapp"), bool):
            raise HTTPException(status_code=400, detail="nudge_to_whatsapp must be true or false")
        overrides["nudge_to_whatsapp"] = body["nudge_to_whatsapp"]
    save_config(uid, overrides)
    return _whatsapp_state(uid)


@router.post("/whatsapp/test")
async def test_whatsapp_route(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Send one test message to wherever this user's nudges go, and report what
    WhatsApp said for each number. At most one test a minute per user."""
    from captain_claw.flight_deck.whatsapp_bridge import send_text_checked

    uid = _user_id(request)
    state = _whatsapp_state(uid)
    recipients = state["recipients"]
    if not state["bridge_configured"]:
        raise HTTPException(status_code=400, detail="WhatsApp isn't set up on this deck.")
    if not recipients:
        raise HTTPException(status_code=400,
                            detail=state["issue"] or "No WhatsApp number to send to.")
    key = _norm_user(uid)
    now = time.monotonic()
    last = _last_test_at.get(key)
    if last is not None and now - last < _TEST_COOLDOWN_S:
        wait = int(_TEST_COOLDOWN_S - (now - last)) + 1
        raise HTTPException(status_code=429, detail=f"Wait {wait}s before sending another test.")
    _last_test_at[key] = now
    results = []
    for waid in recipients:
        ok, why = await send_text_checked(
            waid, "Captain Claw: test nudge — autonomous nudges will reach you here.")
        results.append({"to": waid, "ok": ok, "error": why})
    return {"sent": sum(1 for r in results if r["ok"]), "total": len(results), "results": results}


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
    the user's strongest agent. Falls back to 'queued' if no agent is reachable.

    Compare-and-set (J21): only a row still ``awaiting_approval`` or ``queued`` is
    approved — it is claimed (``dispatched``) atomically BEFORE learning and
    dispatch, so a double click can't dispatch twice or double-count; anything
    else is a 409."""
    uid = _user_id(request)
    store = get_store()
    action = store.get_action(action_id)
    if not action or action.get("user_id") not in (uid, "local"):
        raise HTTPException(status_code=404, detail="Action not found")
    if not store.claim_action(action_id, ("awaiting_approval", "queued"), "dispatched"):
        cur = store.get_action(action_id) or action
        raise HTTPException(
            status_code=409,
            detail=f"action is {cur.get('status') or 'unknown'}, not awaiting approval",
        )
    learned = record_human_feedback(uid, action, True)

    from captain_claw.flight_deck.fd_dispatch import dispatch_action

    try:
        disp = await dispatch_action(uid, action, approved_by_human=True)
    except Exception as exc:
        # The row was claimed: don't strand it in 'dispatched' (that would also
        # hold its email thread "open" forever) — park it for a retry.
        if store.claim_action(action_id, ("dispatched",), "queued"):
            store.update_status(action_id, "queued", outcome_note=f"dispatch error: {exc}"[:500])
        raise
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


@router.get("/follow-ups")
async def list_follow_ups_route(request: Request, status_filter: str | None = None,
                                limit: int = 100, _user: dict | None = Depends(get_optional_user)):
    """Tracked open loops — soft reminders/requests the arbiter is holding for you
    ('waiting on you'). Open ones first, soonest-due first."""
    uid = _user_id(request)
    from captain_claw.flight_deck.events import get_store as events_store
    return {"follow_ups": events_store().list_follow_ups(uid, status=status_filter, limit=limit)}


@router.post("/follow-ups/{follow_up_id}/done")
async def done_follow_up_route(follow_up_id: str, request: Request,
                               _user: dict | None = Depends(get_optional_user)):
    """Mark a tracked follow-up resolved — it stops resurfacing."""
    uid = _user_id(request)
    from captain_claw.flight_deck.events import get_store as events_store
    es = events_store()
    fu = es.get_follow_up(follow_up_id)
    if not fu or fu.get("user_id") not in (uid, "local"):
        raise HTTPException(status_code=404, detail="follow-up not found")
    es.mark_follow_up(follow_up_id, "done")
    _learn_from_follow_up(uid, fu, worthwhile=True)
    get_store().log(uid, "follow-up done", fu.get("summary", "")[:120])
    return {"ok": True}


@router.post("/follow-ups/{follow_up_id}/dismiss")
async def dismiss_follow_up_route(follow_up_id: str, request: Request,
                                  _user: dict | None = Depends(get_optional_user)):
    """Dismiss a tracked follow-up — drop it without acting."""
    uid = _user_id(request)
    from captain_claw.flight_deck.events import get_store as events_store
    es = events_store()
    fu = es.get_follow_up(follow_up_id)
    if not fu or fu.get("user_id") not in (uid, "local"):
        raise HTTPException(status_code=404, detail="follow-up not found")
    es.mark_follow_up(follow_up_id, "dismissed")
    _learn_from_follow_up(uid, fu, worthwhile=False)
    get_store().log(uid, "follow-up dismissed", fu.get("summary", "")[:120])
    return {"ok": True}


def _learn_from_follow_up(uid: str, fu: dict, *, worthwhile: bool) -> None:
    """A done/dismissed follow-up is a learning signal on the 'track' kind for
    that source: keep tracking what the user resolves, suppress what they keep
    dismissing. Gated by learning_enabled. Best-effort."""
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        cfg = resolve_config(uid)
        if not cfg.get("learning_enabled"):
            return
        get_store().record_outcome(
            uid, "track", str(fu.get("source") or "general"), bool(worthwhile),
            seed=float(cfg.get("reliability_seed", 0.6)),
        )
    except Exception:
        pass


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
    return {"catalog": list_catalog(user_id=_user_id(request))}


@router.get("/agent-tools")
async def agent_tools_route(request: Request, _user: dict | None = Depends(get_optional_user)):
    """Discover the user's agent's live tools + skills — the menu the Tools &
    Sources panel (Theme A) promotes into custom actions / sources."""
    from captain_claw.flight_deck.actions import list_agent_tools
    return await list_agent_tools(_user_id(request))


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
    # A person drove this run by hand — it counts as their approval.
    result = await run_action(uid, action_id, args, approved_by_human=True)
    return result
