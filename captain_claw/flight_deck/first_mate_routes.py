"""First Mate routes: the caller's factories, their versions and their runs.

See docs/first-mate-plan.md §11 and §14.

* ``GET  /fd/first-mate``                          may the caller use it (and why not), their
                                                   preferences, the deck's settings, counts
* ``PUT  /fd/first-mate/prefs``                    the caller's preferences
* ``GET  /fd/first-mate/templates``                the starter factories
* ``GET  /fd/first-mate/factories``                the caller's factories, each with its live
                                                   version, its draft and its latest run
* ``POST /fd/first-mate/factories``                a new factory, as a draft (from a template,
                                                   fields, or both)
* ``GET  /fd/first-mate/factories/{id}``           one factory: versions, recent runs, history
* ``PATCH /fd/first-mate/factories/{id}``          name, budget share, priority, Course goal
* ``DELETE /fd/first-mate/factories/{id}``         a draft or a retired factory, for good
* ``POST /fd/first-mate/factories/{id}/draft``     a new draft from the live version
* ``PATCH /fd/first-mate/factories/{id}/draft``    change the open draft
* ``POST /fd/first-mate/factories/{id}/submit``    send the draft for approval
* ``POST /fd/first-mate/factories/{id}/approve``   ``{version, sha}``: the version as the owner saw it
* ``POST /fd/first-mate/factories/{id}/decline``   ``{version, reason}``
* ``POST /fd/first-mate/factories/{id}/state``     ``{state, reason}``: running · paused · stopped · retired
* ``POST /fd/first-mate/factories/{id}/trust``     ``{level, pin}``
* ``POST /fd/first-mate/factories/{id}/run``       Run now, ``{note}``
* ``GET  /fd/first-mate/factories/{id}/runs``      its runs, going ones brought up to date
* ``POST /fd/first-mate/runs/{run_id}/stop``       stop a going run
* ``GET/PUT /fd/admin/first-mate/settings``        the deck switch, admins only, the caps (admin)

Who may use it: the deck switch must be on, and while it's set to admins only
(through Phase 3), only admins. ``GET /fd/first-mate`` answers either way, so
the page can say why; everything else is a 403 for a caller who may not.

Plain ``get_current_user``, as on the Course routes: an admin's ``X-FD-Act-As``
doesn't reach another user's factories, and someone else's factory or run is a
404. There is no admin view of a user's factories.

The rules (versions, approval, trust, runs) live in :mod:`first_mate_store`;
starting and following runs in :mod:`first_mate_runs`.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fms
from captain_claw.flight_deck import first_mate_templates as templates
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.first_mate_store import FirstMateError

router = APIRouter(tags=["first-mate"])

# The states an owner may set by hand. paused_budget is the deck's, and a
# factory leaves draft only through approval.
OWNER_STATES = ("running", "paused", "stopped", "retired")
_CREATE_FIELDS = ("template", "title", "charter", "runner", "runner_config", "triggers", "grants",
                  "budget", "checks", "goal_id", "share", "priority", "note")
_VERSION_FIELDS = ("charter", "runner", "runner_config", "triggers", "grants", "budget", "checks")
_FACTORY_FIELDS = ("title", "share", "priority", "goal_id")


def _uid(user: dict) -> str:
    return fms.norm_user(str(user.get("id") or ""))


def _is_admin(user: dict) -> bool:
    return user.get("role") == "admin"


def _err(exc: Exception) -> HTTPException:
    text = str(exc)
    return HTTPException(status_code=404 if text.startswith("no such") else 400, detail=text)


def _object(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    return body


def _only(body: dict[str, Any], allowed: tuple[str, ...]) -> None:
    extra = sorted(set(body) - set(allowed))
    if extra:
        raise HTTPException(400, f"unknown fields: {', '.join(extra)}")


def access(user: dict) -> tuple[bool, str]:
    """May ``user`` use First Mate on this deck, and if not, why not."""
    settings = fms.get_store().settings()
    if not settings["enabled"]:
        return False, "First Mate is off on this deck. An admin can turn it on in Settings."
    if settings["admins_only"] and not _is_admin(user):
        return False, "First Mate is open to admins only on this deck for now."
    return True, ""


async def allowed_user(user: dict = Depends(get_current_user)) -> dict:
    ok, why = access(user)
    if not ok:
        raise HTTPException(403, why)
    return user


def _mine(factory_id: str, user: dict) -> dict[str, Any]:
    store = fms.get_store()
    factory = store.get_owned(factory_id, _uid(user))
    if factory is None:
        raise HTTPException(404, "no such factory")
    return store.view(factory_id)


def _check_goal(goal_id: Any, user: dict) -> None:
    """A Course goal a factory may serve: the caller's own or a deck-wide one."""
    if not goal_id:
        return
    from captain_claw.flight_deck import course
    goal = course.get_store().get(str(goal_id))
    if goal is None or not (goal["scope"] == "instance" or goal["owner_id"] == _uid(user)):
        raise HTTPException(400, "that Course goal isn't one of yours")


def _week(store: fms.FirstMateStore, owner: str, now: datetime) -> dict[str, Any]:
    """What a card needs from the owner's whole week: when it started, each
    factory's share, and the events waiting in each factory's batch."""
    from captain_claw.flight_deck import first_mate_events, first_mate_loop
    factories = store.list_factories(owner)
    evs = first_mate_events._events_store()
    pending = Counter(str((e.get("metadata") or {}).get("fm_factory") or "")
                      for e in (evs.list_claimed(owner) if evs is not None else []))
    return {"start": first_mate_loop.week_start(now),
            "shares": first_mate_loop.shares(store.prefs(owner), factories),
            "pending": pending}


def _next_at(store: fms.FirstMateStore, view: dict[str, Any], now: datetime) -> str | None:
    """When its schedules next fire (the loop's saved time, or worked out for
    one it hasn't seen yet). Past = due now."""
    from captain_claw.flight_deck import first_mate_loop
    live = view.get("live")
    schedules = [t for t in (live or {}).get("triggers", []) if t.get("kind") == "schedule"]
    if not schedules:
        return None
    saved = store.trigger_state(view["id"])
    times = []
    for t in schedules:
        st = saved.get(first_mate_loop.trigger_key(t))
        nxt = st["next_at"] if st else first_mate_loop.next_fire(t["schedule"], t.get("tz", ""), now)
        if nxt:
            times.append(nxt)
    return datetime.fromtimestamp(min(times), UTC).isoformat() if times else None


def _card(store: fms.FirstMateStore, view: dict[str, Any], week: dict[str, Any], now: datetime) -> dict[str, Any]:
    latest = store.list_runs(view["id"], limit=1)
    return {**view, "last_run": latest[0] if latest else None,
            "next_at": _next_at(store, view, now),
            "week_tokens": store.tokens_since(view["id"], week["start"]),
            "share_tokens": week["shares"].get(view["id"], 0),
            "pending_events": week["pending"].get(view["id"], 0)}


# ── status and preferences ─────────────────────────────────────────────

@router.get("/fd/first-mate")
async def status(user: dict = Depends(get_current_user)):
    store = fms.get_store()
    ok, why = access(user)
    uid = _uid(user)
    factories = store.list_factories(uid) if ok else []
    from captain_claw.flight_deck import first_mate_loop
    start = first_mate_loop.week_start(fms._utcnow())
    prefs = store.prefs(uid)
    return {
        "available": ok,
        "reason": why,
        "is_admin": _is_admin(user),
        "settings": store.settings(),
        "prefs": prefs,
        "counts": {"factories": len(factories),
                   "running": sum(1 for f in factories if f["state"] == "running"),
                   "runs_going": len(store.active_runs(uid)) if ok else 0},
        "week": {"start": start.isoformat(), "budget": prefs["weekly_tokens"],
                 "used": sum(store.tokens_since(f["id"], start) for f in factories)},
    }


@router.put("/fd/first-mate/prefs")
async def put_prefs(body: Any = Body(...), user: dict = Depends(allowed_user)):
    """The caller's preferences. Turning ``on`` off pauses their running
    factories and stops their going runs; turning it on again resumes the
    ones the switch paused."""
    from captain_claw.flight_deck import first_mate_loop

    store = fms.get_store()
    uid = _uid(user)
    was_on = store.prefs(uid)["on"]
    try:
        saved = store.set_prefs(uid, _object(body))
    except FirstMateError as exc:
        raise _err(exc) from None
    if was_on and not saved["on"]:
        await first_mate_loop.switch_off(uid)
    elif saved["on"] and not was_on:
        await first_mate_loop.switch_on(uid)
    return saved


@router.get("/fd/first-mate/templates")
async def list_templates(user: dict = Depends(allowed_user)):
    return {"templates": templates.listing()}


# ── factories ──────────────────────────────────────────────────────────

@router.get("/fd/first-mate/factories")
async def list_factories(include_retired: bool = False, user: dict = Depends(allowed_user)):
    uid = _uid(user)
    await runs.refresh_owner(uid)
    store = fms.get_store()
    now = fms._utcnow()
    week = _week(store, uid, now)
    return {"factories": [_card(store, store.view(f["id"]), week, now)
                          for f in store.list_factories(uid, include_retired=include_retired)]}


@router.post("/fd/first-mate/factories")
async def create_factory(body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, _CREATE_FIELDS)
    fields: dict[str, Any] = {}
    if body.get("template"):
        base = templates.fields_for(str(body["template"]))
        if base is None:
            raise HTTPException(400, "no such template")
        fields.update(base)
    fields.update({k: v for k, v in body.items() if k != "template" and v is not None})
    _check_goal(fields.get("goal_id"), user)
    try:
        view = fms.get_store().create(_uid(user), created_by="owner", **fields)
    except TypeError as exc:  # a required field is missing
        raise HTTPException(400, "title and charter are required") from exc
    except FirstMateError as exc:
        raise _err(exc) from None
    return view


@router.get("/fd/first-mate/factories/{factory_id}")
async def get_factory(factory_id: str, user: dict = Depends(allowed_user)):
    view = _mine(factory_id, user)
    store = fms.get_store()
    for run in store.list_runs(factory_id, limit=5):
        if run["status"] in fms.ACTIVE_RUN:
            await runs.refresh(run["id"])
    return {**view,
            "versions": store.versions(factory_id),
            "runs": store.list_runs(factory_id, limit=20),
            "events": store.events(factory_id, limit=50),
            "satisfaction": store.satisfaction(factory_id)}


@router.patch("/fd/first-mate/factories/{factory_id}")
async def patch_factory(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, _FACTORY_FIELDS)
    _mine(factory_id, user)
    if "goal_id" in body:
        _check_goal(body["goal_id"], user)
    try:
        return fms.get_store().update_factory(factory_id, "owner", **body)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.delete("/fd/first-mate/factories/{factory_id}")
async def delete_factory(factory_id: str, user: dict = Depends(allowed_user)):
    _mine(factory_id, user)
    try:
        return {"ok": fms.get_store().delete(factory_id, _uid(user))}
    except FirstMateError as exc:
        raise _err(exc) from None


# ── versions ───────────────────────────────────────────────────────────

@router.post("/fd/first-mate/factories/{factory_id}/draft")
async def new_draft(factory_id: str, body: Any = Body(default={}), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, (*_VERSION_FIELDS, "note"))
    _mine(factory_id, user)
    try:
        return fms.get_store().new_draft(factory_id, "owner", **body)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.patch("/fd/first-mate/factories/{factory_id}/draft")
async def edit_draft(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, (*_VERSION_FIELDS, "note"))
    _mine(factory_id, user)
    try:
        return fms.get_store().edit_draft(factory_id, "owner", **body)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.post("/fd/first-mate/factories/{factory_id}/submit")
async def submit(factory_id: str, user: dict = Depends(allowed_user)):
    _mine(factory_id, user)
    try:
        return fms.get_store().submit(factory_id, "owner")
    except FirstMateError as exc:
        raise _err(exc) from None


@router.post("/fd/first-mate/factories/{factory_id}/approve")
async def approve(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, ("version", "sha"))
    _mine(factory_id, user)
    try:
        version = int(body.get("version") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "version must be a number") from None
    try:
        return fms.get_store().approve(factory_id, version, str(body.get("sha") or ""), _uid(user))
    except FirstMateError as exc:
        raise _err(exc) from None


@router.post("/fd/first-mate/factories/{factory_id}/decline")
async def decline(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, ("version", "reason"))
    _mine(factory_id, user)
    try:
        version = int(body.get("version") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "version must be a number") from None
    try:
        return fms.get_store().decline(factory_id, version, _uid(user), str(body.get("reason") or ""))
    except FirstMateError as exc:
        raise _err(exc) from None


# ── state and trust ────────────────────────────────────────────────────

@router.post("/fd/first-mate/factories/{factory_id}/state")
async def set_state(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, ("state", "reason"))
    _mine(factory_id, user)
    state = str(body.get("state") or "")
    if state not in OWNER_STATES:
        raise HTTPException(400, f"state must be one of: {', '.join(OWNER_STATES)}")
    try:
        view = fms.get_store().set_state(factory_id, state, "owner", str(body.get("reason") or ""))
    except FirstMateError as exc:
        raise _err(exc) from None
    if state == "running":
        from captain_claw.flight_deck import first_mate_line
        await first_mate_line.factory_resumed(_uid(user), factory_id)  # its "paused" item leaves Needs you
    return view


@router.post("/fd/first-mate/factories/{factory_id}/trust")
async def set_trust(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, ("level", "pin"))
    _mine(factory_id, user)
    pin = body.get("pin")
    if pin is not None and not isinstance(pin, bool):
        raise HTTPException(400, "pin must be true or false")
    try:
        return fms.get_store().set_trust(factory_id, str(body.get("level") or ""), "owner", pin=pin)
    except FirstMateError as exc:
        raise _err(exc) from None


# ── runs ───────────────────────────────────────────────────────────────

@router.post("/fd/first-mate/factories/{factory_id}/run")
async def run_now(factory_id: str, body: Any = Body(default={}), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, ("note",))
    _mine(factory_id, user)
    note = body.get("note") or ""
    if not isinstance(note, str):
        raise HTTPException(400, "note must be text")
    try:
        return await runs.start(factory_id, _uid(user), trigger="manual", note=note)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.get("/fd/first-mate/factories/{factory_id}/runs")
async def list_runs(factory_id: str, limit: int = 50, user: dict = Depends(allowed_user)):
    _mine(factory_id, user)
    store = fms.get_store()
    for run in store.list_runs(factory_id, limit=max(1, min(limit, 200))):
        if run["status"] in fms.ACTIVE_RUN:
            await runs.refresh(run["id"])
    return {"runs": store.list_runs(factory_id, limit=max(1, min(limit, 200)))}


@router.post("/fd/first-mate/runs/{run_id}/stop")
async def stop_run(run_id: str, user: dict = Depends(allowed_user)):
    try:
        return await runs.stop(run_id, _uid(user))
    except FirstMateError as exc:
        raise _err(exc) from None


# ── the deck's settings (admins) ───────────────────────────────────────

@router.get("/fd/admin/first-mate/settings")
async def get_settings(admin: dict = Depends(require_admin)):
    return {**fms.get_store().settings(), "defaults": dict(fms.DEFAULT_DECK)}


@router.put("/fd/admin/first-mate/settings")
async def put_settings(body: Any = Body(...), admin: dict = Depends(require_admin)):
    try:
        return fms.get_store().set_settings(_object(body))
    except FirstMateError as exc:
        raise _err(exc) from None
