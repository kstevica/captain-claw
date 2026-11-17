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
from pydantic import BaseModel, Field

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


def _with_intents(store: fms.FirstMateStore, runs_: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs, each rehearsal with what it would have done (the words, not the arguments)."""
    return [{**r, "thumbs": store.thumbs(r["id"]),
             "intents": [{**{k: i[k] for k in ("id", "kind", "tool", "what", "status")},
                          "detail": runs.intent_detail(i)}
                         for i in store.intents(r["id"])] if r["kind"] == "rehearsal" else []}
            for r in runs_]


def _card(store: fms.FirstMateStore, view: dict[str, Any], week: dict[str, Any], now: datetime) -> dict[str, Any]:
    latest = store.list_runs(view["id"], limit=1, kinds=fms.WORK_KINDS)
    waiting = store.open_proposals(view["id"])
    proposal = None
    if waiting:
        count = sum(1 for i in store.intents(waiting[0]) if i["status"] == "open")
        proposal = {"run_id": waiting[0], "count": count}
    from captain_claw.flight_deck import first_mate_score
    card = first_mate_score.scorecard(store, view, now)
    return {**view, "last_run": latest[0] if latest else None, "proposal": proposal,
            "health": card["health"], "health_why": card["health_why"], "satisfaction": card["satisfaction"],
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
        "calibration": _calibration(store, uid) if ok else None,
    }


def _calibration(store: fms.FirstMateStore, uid: str) -> dict[str, Any]:
    from captain_claw.flight_deck import first_mate_trust
    cal = first_mate_trust.calibration(store, uid)
    return {k: cal[k] for k in ("n", "kappa", "agreement", "ready", "why", "labels_bar", "kappa_bar")}


@router.put("/fd/first-mate/prefs")
async def put_prefs(body: Any = Body(...), user: dict = Depends(allowed_user)):
    """The caller's preferences. Turning ``on`` off pauses their running
    factories and stops their going runs; turning it on again resumes the
    ones the switch paused."""
    from captain_claw.flight_deck import first_mate_loop

    store = fms.get_store()
    uid = _uid(user)
    was_on = store.prefs(uid)["on"]
    body = _object(body)
    try:
        if body.get("agent"):
            from captain_claw.flight_deck import first_mate_agent
            first_mate_agent.check_pick(uid, str(body["agent"]))
        saved = store.set_prefs(uid, body)
    except FirstMateError as exc:
        raise _err(exc) from None
    if was_on and not saved["on"]:
        await first_mate_loop.switch_off(uid)
    elif saved["on"] and not was_on:
        await first_mate_loop.switch_on(uid)
    return saved


@router.get("/fd/first-mate/my-agent")
async def my_agent(user: dict = Depends(allowed_user)):
    """The agent that is the caller's First Mate, and the agents they may pick."""
    from captain_claw.flight_deck import first_mate_agent
    uid = _uid(user)
    return {"current": first_mate_agent.current(fms.get_store(), uid), "agents": first_mate_agent.mine(uid)}


@router.post("/fd/first-mate/my-agent")
async def make_my_agent(user: dict = Depends(allowed_user)):
    """Make the caller an agent called First Mate and point First Mate at it."""
    from captain_claw.flight_deck import first_mate_agent
    try:
        made = await first_mate_agent.create(fms.get_store(), user)
    except FirstMateError as exc:
        raise _err(exc) from None
    return {**made, "current": first_mate_agent.current(fms.get_store(), _uid(user))}


@router.post("/fd/first-mate/draft")
async def draft_factory(body: Any = Body(...), user: dict = Depends(allowed_user)):
    """A new factory drafted by a model from the owner's words, or from one of
    their Course goals. It waits for their approval, with a card on the Line."""
    from captain_claw.flight_deck import first_mate_drafting
    body = _object(body)
    _only(body, ("words", "goal_id"))
    words = body.get("words") or ""
    if not isinstance(words, str):
        raise HTTPException(400, "words must be text")
    goal = None
    if body.get("goal_id"):
        _check_goal(body["goal_id"], user)
        from captain_claw.flight_deck import course
        goal = course.get_store().get(str(body["goal_id"]))
    try:
        return await first_mate_drafting.draft(fms.get_store(), _uid(user), words, goal=goal, by="owner")
    except FirstMateError as exc:
        raise _err(exc) from None


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
    from captain_claw.flight_deck import first_mate_score
    view = _mine(factory_id, user)
    store = fms.get_store()
    for run in store.list_runs(factory_id, limit=5):
        if run["status"] in fms.ACTIVE_RUN:
            await runs.refresh(run["id"])
    return {**view,
            "versions": store.versions(factory_id),
            "runs": _with_intents(store, store.list_runs(factory_id, limit=20, kinds=fms.WORK_KINDS)),
            "check": store.active_check(factory_id),
            "events": store.events(factory_id, limit=50),
            "satisfaction": store.satisfaction(factory_id),
            "scorecard": first_mate_score.scorecard(store, view, fms._utcnow()),
            "ladder": _ladder(store, view),
            "signals": store.signals(factory_id, limit=20),
            "drafts": [{k: d[k] for k in ("id", "run_id", "subject", "status", "similarity", "created_at")}
                       for d in store.drafts(factory_id, limit=20)]}


def _ladder(store: fms.FirstMateStore, view: dict[str, Any]) -> dict[str, Any]:
    """The factory's record at its trust level, and the owner's calibration."""
    from captain_claw.flight_deck import first_mate_trust
    cal = first_mate_trust.calibration(store, view["owner_id"])
    rec = first_mate_trust.record(store, view, cal)
    return {**{k: rec[k] for k in ("level", "next", "counted", "needed", "value", "bar", "doubts", "pinned",
                                   "eligible", "why")},
            "calibration": {k: cal[k] for k in ("n", "kappa", "ready", "why", "labels_bar", "kappa_bar")}}


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
    view = _mine(factory_id, user)
    before = (view.get("draft") or {}).get("sha", "")
    try:
        after = fms.get_store().edit_draft(factory_id, "owner", **body)
    except FirstMateError as exc:
        raise _err(exc) from None
    if before and (after.get("draft") or {}).get("sha") != before:
        from captain_claw.flight_deck import first_mate_line
        await first_mate_line.charter_settled(_uid(user), factory_id, before)  # its card shows the old draft
    return after


@router.post("/fd/first-mate/factories/{factory_id}/submit")
async def submit(factory_id: str, user: dict = Depends(allowed_user)):
    _mine(factory_id, user)
    try:
        return fms.get_store().submit(factory_id, "owner")
    except FirstMateError as exc:
        raise _err(exc) from None


@router.post("/fd/first-mate/factories/{factory_id}/approve")
async def approve(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    """Approve the waiting draft as the owner saw it (``sha``). A factory that
    is live and has cases checks the draft on them first: it goes live when the
    check passes (``check`` in the answer), unless ``skip_check``."""
    from captain_claw.flight_deck import first_mate_cases
    body = _object(body)
    _only(body, ("version", "sha", "skip_check"))
    view = _mine(factory_id, user)
    try:
        version = int(body.get("version") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "version must be a number") from None
    store = fms.get_store()
    sha = str(body.get("sha") or "")
    draft = store.version(factory_id, version)
    gated = (view["version"] and not body.get("skip_check") and draft is not None
             and draft["status"] in fms.OPEN_VERSION and version == view["draft_version"] and sha == draft["sha"]
             and view["state"] != "retired" and first_mate_cases.cases(store, factory_id))
    from captain_claw.flight_deck import first_mate_line
    try:
        if gated:
            check = first_mate_cases.start_check(store, factory_id, version=version, purpose="approve")
            await first_mate_line.charter_settled(_uid(user), factory_id, sha)
            return {**store.view(factory_id), "check": check}
        approved = store.approve(factory_id, version, sha, _uid(user))
    except FirstMateError as exc:
        raise _err(exc) from None
    await first_mate_line.charter_settled(_uid(user), factory_id, sha)
    return approved


@router.post("/fd/first-mate/factories/{factory_id}/decline")
async def decline(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    body = _object(body)
    _only(body, ("version", "reason"))
    view = _mine(factory_id, user)
    try:
        version = int(body.get("version") or 0)
    except (TypeError, ValueError):
        raise HTTPException(400, "version must be a number") from None
    sha = (view.get("draft") or {}).get("sha", "")
    try:
        declined = fms.get_store().decline(factory_id, version, _uid(user), str(body.get("reason") or ""))
    except FirstMateError as exc:
        raise _err(exc) from None
    if sha:
        from captain_claw.flight_deck import first_mate_line
        await first_mate_line.charter_settled(_uid(user), factory_id, sha)
    return declined


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
    store = fms.get_store()
    try:
        view = store.set_trust(factory_id, str(body.get("level") or ""), "owner", pin=pin)
    except FirstMateError as exc:
        raise _err(exc) from None
    # Setting trust by hand is the owner's answer to a doubtful carry-out.
    from captain_claw.flight_deck import first_mate_line
    doubted = [r["id"] for r in store.list_runs(factory_id, limit=100)
               if r["kind"] == "carry_out" and (r.get("proof") or {}).get("state") == "doubt"]
    await first_mate_line.proofs_seen(_uid(user), doubted)
    return view


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


# ── the weekly report ──────────────────────────────────────────────────

@router.get("/fd/first-mate/report")
async def week_report(week: str = "last", user: dict = Depends(allowed_user)):
    """The owner's week: ``last`` (the one the Monday report is about) or ``this`` so far."""
    from captain_claw.flight_deck import first_mate_report
    if week not in ("last", "this"):
        raise HTTPException(400, "week is last or this")
    store = fms.get_store()
    now = fms._utcnow()
    start, end = first_mate_report.period(now, week)
    report = first_mate_report.build(store, _uid(user), start, end, now)
    title, text = first_mate_report.text(report)
    return {**report, "title": title, "text": text, "sent": first_mate_report.last_sent(store, _uid(user))}


# ── improvement runs (Captain Swarm) ───────────────────────────────────

_IMPROVE_FIELDS = ("budget_tokens", "budget_hours", "budget_max_experiments", "per_round", "parallel", "repeat",
                   "holdout_repeat")


@router.post("/fd/first-mate/factories/{factory_id}/improve")
async def improve_factory(factory_id: str, body: Any = Body(default={}), user: dict = Depends(allowed_user)):
    """An improvement run of the factory: Captain Swarm tries better charters
    and steps on its cases. Its best becomes a draft only on the owner's say."""
    from captain_claw.flight_deck import captain_swarm_loop, first_mate_improve
    from captain_claw.flight_deck.captain_swarm_objective import ObjectiveError
    body = _object(body)
    _only(body, _IMPROVE_FIELDS)
    _mine(factory_id, user)
    if captain_swarm_loop.get_store() is None:
        raise HTTPException(503, "Captain Swarm isn't running on this deck.")
    try:
        run = await first_mate_improve.start(user, factory_id, body)
    except ObjectiveError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"run_id": run["id"], "status": run["status"], "title": run["title"]}


@router.get("/fd/first-mate/factories/{factory_id}/improvements")
async def list_improvements(factory_id: str, user: dict = Depends(allowed_user)):
    """The factory's improvement runs (Captain Swarm), newest first, whether
    each one's best became a draft, and whether one can start now."""
    from captain_claw.flight_deck import captain_swarm_routes as csr
    from captain_claw.flight_deck import first_mate_improve
    _mine(factory_id, user)
    store = fms.get_store()
    drafted = {str((e.get("detail") or {}).get("run") or "") for e in store.events(factory_id, limit=500)
               if e["event"] == "improvement_drafted"}
    def iso(t: Any) -> str | None:  # Captain Swarm keeps epoch seconds; the page reads ISO
        return datetime.fromtimestamp(float(t), UTC).isoformat() if isinstance(t, (int, float)) and t > 0 else None

    runs_ = []
    for r in await first_mate_improve.runs_for(csr._uid(user), factory_id):
        summary = csr._run_summary(r)
        runs_.append({**summary, "created_at": iso(summary["created_at"]), "started_at": iso(summary["started_at"]),
                      "finished_at": iso(summary["finished_at"]), "drafted": r["id"] in drafted,
                      "going": r["status"] in first_mate_improve.ACTIVE})
    ok, why = await first_mate_improve.can_improve(user, factory_id)
    from captain_claw.flight_deck import captain_swarm_loop
    try:
        await csr.admin_user(user)
        available = captain_swarm_loop.get_store() is not None
    except HTTPException:
        available = False  # not an admin, or the desktop app: Captain Swarm isn't theirs here
    return {"runs": runs_, "can": {"ok": ok, "why": why}, "available": available}


@router.post("/fd/first-mate/factories/{factory_id}/improvements/{run_id}/draft")
async def improvement_draft(factory_id: str, run_id: str, user: dict = Depends(allowed_user)):
    """A finished improvement run's best as the factory's new draft, when it
    couldn't become one as the run ended (a draft was open then)."""
    from captain_claw.flight_deck import captain_swarm_loop, first_mate_improve, first_mate_line
    _mine(factory_id, user)
    cs = captain_swarm_loop.get_store()
    run = await cs.get_run(run_id) if cs is not None else None
    subj = ((run or {}).get("objective") or {}).get("subject") or {}
    if run is None or fms.norm_user(run["owner_id"]) != _uid(user) or subj.get("factory_id") != factory_id:
        raise HTTPException(404, "no such improvement run")
    if run["status"] in ("awaiting_go", "baseline", "running"):
        raise HTTPException(409, "that improvement run is still going")
    try:
        view = await first_mate_improve.draft_from(run)
    except FirstMateError as exc:
        raise _err(exc) from None
    kept = [e["hypothesis"] for e in await cs.list_experiments(run_id) if e.get("status") == "kept"]
    await first_mate_line.improvement_drafted(view, run, kept)
    return view


# ── cases and checks ───────────────────────────────────────────────────

_CASE_FIELDS = ("title", "input", "expect", "sealed")


@router.get("/fd/first-mate/factories/{factory_id}/cases")
async def list_cases(factory_id: str, user: dict = Depends(allowed_user)):
    from captain_claw.flight_deck import first_mate_cases
    _mine(factory_id, user)
    store = fms.get_store()
    return {"cases": first_mate_cases.cases(store, factory_id), "checks": store.checks(factory_id, limit=20),
            "max": first_mate_cases.CASES_MAX}


@router.post("/fd/first-mate/factories/{factory_id}/cases")
async def add_case(factory_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    from captain_claw.flight_deck import first_mate_cases
    body = _object(body)
    _only(body, (*_CASE_FIELDS, "from_run"))
    _mine(factory_id, user)
    store = fms.get_store()
    from_run = str(body.get("from_run") or "")
    if from_run:
        run = store.get_run(from_run)
        if run is None or run["factory_id"] != factory_id:
            raise HTTPException(404, "no such run")
    try:
        return first_mate_cases.add_case(store, factory_id, title=body.get("title"), input=body.get("input"),
                                         expect=body.get("expect"), sealed=body.get("sealed", True),
                                         origin="run" if from_run else "owner", from_run=from_run)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.patch("/fd/first-mate/factories/{factory_id}/cases/{case_id}")
async def update_case(factory_id: str, case_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    from captain_claw.flight_deck import first_mate_cases
    body = _object(body)
    _only(body, _CASE_FIELDS)
    _mine(factory_id, user)
    try:
        return first_mate_cases.update_case(fms.get_store(), factory_id, case_id, **body)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.delete("/fd/first-mate/factories/{factory_id}/cases/{case_id}")
async def delete_case(factory_id: str, case_id: str, user: dict = Depends(allowed_user)):
    from captain_claw.flight_deck import first_mate_cases
    _mine(factory_id, user)
    try:
        first_mate_cases.delete_case(fms.get_store(), factory_id, case_id)
    except FirstMateError as exc:
        raise _err(exc) from None
    return {"ok": True}


@router.get("/fd/first-mate/runs/{run_id}/case-draft")
async def case_draft(run_id: str, user: dict = Depends(allowed_user)):
    """A case started from a past run: its fixed input and its contract."""
    from captain_claw.flight_deck import first_mate_cases
    store = fms.get_store()
    run = store.get_run(run_id)
    if run is None or run["owner_id"] != _uid(user) or run["kind"] == "check":
        raise HTTPException(404, "no such run")
    return first_mate_cases.case_draft(store, run)


@router.post("/fd/first-mate/factories/{factory_id}/check")
async def check_now(factory_id: str, user: dict = Depends(allowed_user)):
    """Check the live version on every case now."""
    from captain_claw.flight_deck import first_mate_cases
    view = _mine(factory_id, user)
    if not view["version"]:
        raise HTTPException(400, "a factory is checked once its charter is approved")
    try:
        return first_mate_cases.start_check(fms.get_store(), factory_id, version=view["version"], purpose="manual")
    except FirstMateError as exc:
        raise _err(exc) from None


@router.delete("/fd/first-mate/factories/{factory_id}/check")
async def stop_check(factory_id: str, user: dict = Depends(allowed_user)):
    """Stop the check going; a draft it was checking stays waiting."""
    _mine(factory_id, user)
    store = fms.get_store()
    going = store.active_check(factory_id)
    if going is None:
        raise HTTPException(404, "nothing is being checked")
    finished = store.finish_check(going["id"], "void", note="stopped by you")
    for res in going["results"].values():
        run = store.get_run(res.get("run_id") or "")
        if run is not None and run["status"] in fms.ACTIVE_RUN:
            try:
                await runs.stop(run["id"], _uid(user))
            except FirstMateError:
                pass
    return finished


@router.get("/fd/first-mate/factories/{factory_id}/runs")
async def list_runs(factory_id: str, limit: int = 50, user: dict = Depends(allowed_user)):
    _mine(factory_id, user)
    store = fms.get_store()
    for run in store.list_runs(factory_id, limit=max(1, min(limit, 200))):
        if run["status"] in fms.ACTIVE_RUN:
            await runs.refresh(run["id"])
    return {"runs": _with_intents(store, store.list_runs(factory_id, limit=max(1, min(limit, 200)),
                                                         kinds=fms.WORK_KINDS))}


@router.post("/fd/first-mate/runs/{run_id}/approve")
async def approve_run(run_id: str, body: Any = Body(default={}), user: dict = Depends(allowed_user)):
    """Carry out what a rehearsal would have done: all of it, or ``ids``. One
    carry-out run, with exactly the charter's grants; the rest is rejected."""
    body = _object(body)
    _only(body, ("ids",))
    ids = body.get("ids")
    if ids is not None and (not isinstance(ids, list) or not all(isinstance(i, int) for i in ids)):
        raise HTTPException(400, "ids must be a list of intent ids")
    try:
        return await runs.carry_out(run_id, _uid(user), ids)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.post("/fd/first-mate/runs/{run_id}/reject")
async def reject_run(run_id: str, user: dict = Depends(allowed_user)):
    try:
        return await runs.reject(run_id, _uid(user))
    except FirstMateError as exc:
        raise _err(exc) from None


@router.post("/fd/first-mate/runs/{run_id}/thumbs")
async def thumbs(run_id: str, body: Any = Body(...), user: dict = Depends(allowed_user)):
    """The owner's 👍/👎 on a run's result (``{kind: up | down | ""}``): one per run."""
    body = _object(body)
    _only(body, ("kind",))
    kinds = {"up": "thumbs_up", "down": "thumbs_down", "": ""}
    kind = body.get("kind")
    if kind not in kinds:
        raise HTTPException(400, "kind must be up, down or empty")
    store = fms.get_store()
    run = store.get_run(run_id)
    if run is None or run["owner_id"] != _uid(user):
        raise HTTPException(404, "no such run")
    if run["status"] not in fms.TERMINAL_RUN:
        raise HTTPException(400, "a run is rated once it has finished")
    try:
        store.set_thumbs(run["factory_id"], run_id, kinds[kind])
    except FirstMateError as exc:
        raise _err(exc) from None
    return {"thumbs": store.thumbs(run_id), "satisfaction": store.satisfaction(run["factory_id"])}


@router.get("/fd/first-mate/runs/{run_id}/progress")
async def run_progress(run_id: str, since: int = 0, user: dict = Depends(allowed_user)):
    """What one of the owner's runs is doing: its steps, its events from
    ``since`` (``next`` is where the following read starts), the agents working
    for it now, and its files' folder."""
    run = fms.get_store().get_run(run_id)
    if run is None or run["owner_id"] != _uid(user):
        raise HTTPException(404, "no such run")
    try:
        return await runs.progress(run, since=since)
    except FirstMateError as exc:
        raise _err(exc) from None


@router.post("/fd/first-mate/runs/{run_id}/stop")
async def stop_run(run_id: str, user: dict = Depends(allowed_user)):
    try:
        return await runs.stop(run_id, _uid(user))
    except FirstMateError as exc:
        raise _err(exc) from None


# ── a rehearsal's intents (the worker's guard calls this) ─────────────

INTENTS_PER_RUN = 300


class AgentIntentReq(BaseModel):
    source_port: int = 0
    web_auth: str = ""
    owner_id: str = ""
    tag: str = Field("", max_length=80)
    kind: str = Field("intent", max_length=10)
    tool: str = Field("", max_length=80)
    what: str = Field("", max_length=500)
    args: dict[str, Any] = Field(default_factory=dict)


@router.post("/fd/first-mate/agent/intent")
async def agent_intent(body: AgentIntentReq) -> dict:
    """A rehearsing worker's intent or commit point (rehearsal_turn): kept with
    its run. Only the run's own owner's agents can add to it, only while the
    run goes, only for a rehearsal. The guard has already refused the call
    itself; this only keeps what it would have been."""
    from captain_claw.flight_deck.basna_routes import _AgentReq, _resolve_owner
    from captain_claw.flight_deck.captain_swarm_routes import _agent_slug
    if body.kind not in ("intent", "commit"):
        return {"ok": False, "why": "not an intent"}
    owner = fms.norm_user(_resolve_owner(_AgentReq(source_port=body.source_port, web_auth=body.web_auth,
                                                   owner_id=body.owner_id)) or "local")
    store = fms.get_store()
    run = store.get_run(body.tag)
    if run is None or run["owner_id"] != owner or run["kind"] not in ("rehearsal", "check"):
        return {"ok": False, "why": "no such rehearsal"}
    if run["status"] not in fms.ACTIVE_RUN:
        return {"ok": False, "why": "that run is over"}
    if len(store.intents(run["id"])) >= INTENTS_PER_RUN:
        return {"ok": False, "why": "too many intents for one run"}
    try:
        iid = store.add_intent(run["id"], kind=body.kind, tool=body.tool, what=body.what, args=body.args,
                               agent=_agent_slug(body.source_port))
    except FirstMateError as exc:
        return {"ok": False, "why": str(exc)}
    return {"ok": True, "id": iid}


# ── the first_mate tool (an agent's turn for its owner) ─────────────────

AGENT_ACTIONS = ("list", "status", "run", "pause", "resume", "report", "draft")
_STATE_WORDS = {"draft": "a draft", "awaiting_charter": "waiting for your approval", "running": "running",
                "paused": "paused", "paused_budget": "over budget", "stopped": "stopped"}


class AgentToolReq(BaseModel):
    source_port: int = 0
    web_auth: str = ""
    owner_id: str = ""
    action: str = Field("", max_length=20)
    factory: str = Field("", max_length=200)
    note: str = Field("", max_length=500)
    week: str = Field("last", max_length=10)
    words: str = Field("", max_length=2000)


def find_factory(store: fms.FirstMateStore, owner: str, name: str) -> dict[str, Any]:
    """One of the owner's factories by id or by name: exact, then the only one
    whose name holds it. Refused with the names when it isn't one."""
    mine = store.list_factories(owner)
    want = (name or "").strip()
    if not want:
        raise FirstMateError("say which factory")
    for f in mine:
        if f["id"] == want or f["title"].casefold() == want.casefold():
            return f
    hits = [f for f in mine if want.casefold() in f["title"].casefold()]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(f["title"] for f in (hits or mine)) or "none yet"
    if hits:
        raise FirstMateError(f"more than one factory fits \"{want}\": {names}")
    raise FirstMateError(f"no factory called \"{want}\"; yours: {names}")


def _line_of(store: fms.FirstMateStore, card: dict[str, Any]) -> str:
    from captain_claw.flight_deck import first_mate_report as report
    bits = [_STATE_WORDS.get(card["state"], card["state"].replace("_", " ")),
            report._TRUST_WORDS.get(card["trust"], card["trust"]),
            report._HEALTH_WORDS.get(card["health"], card["health"])]
    last = card.get("last_run")
    if last:
        bits.append(f"last run {last['status']} {str(last['created_at'])[:16].replace('T', ' ')} UTC")
    if card.get("next_at"):
        bits.append(f"next {str(card['next_at'])[:16].replace('T', ' ')} UTC")
    if card.get("proposal"):
        n = card["proposal"]["count"]
        bits.append(f"{n} {'action' if n == 1 else 'actions'} waiting for your yes")
    return f"- {card['title']}: " + " · ".join(bits)


def _status_text(store: fms.FirstMateStore, card: dict[str, Any], now: datetime) -> str:
    from captain_claw.flight_deck import (
        first_mate_cases,
        first_mate_report,
        first_mate_score,
        first_mate_trust,
    )
    f = store.view(card["id"])
    live = f.get("live") or {}
    sc = first_mate_score.scorecard(store, f, now)
    lines = [f"{f['title']} (version {f['version'] or 'none yet'}, {f['state'].replace('_', ' ')}, "
             f"{first_mate_report._TRUST_WORDS.get(f['trust'], f['trust'])})"]
    if f["state_reason"]:
        lines.append(f"Why: {f['state_reason']}")
    if live:
        lines.append("Charter: " + live["charter"][:600])
        sched = [t["schedule"] for t in live.get("triggers") or [] if t.get("kind") == "schedule"]
        evs = [t.get("source", "") for t in live.get("triggers") or [] if t.get("kind") == "events"]
        lines.append("Starts: " + (", ".join(sched + [f"on new {e}" for e in evs]) or "only when you press Run now"))
    if f.get("draft"):
        lines.append(f"Draft version {f['draft']['version']} waits for your approval on First Mate.")
    why = sc["health_why"].rstrip(".")
    lines.append(f"Health: {first_mate_report._HEALTH_WORDS.get(sc['health'], sc['health'])}"
                 + (f" ({why[:1].lower() + why[1:]})" if why else "") + ".")
    w = sc["week"]
    lines.append(f"This week: {w['runs']} runs, {w['done']} done, {first_mate_report.tokens_text(card['week_tokens'])} "
                 f"of its {first_mate_report.tokens_text(card['share_tokens'])} share."
                 + (f" Satisfaction {sc['satisfaction']:.0%} ({sc['signals']} signals)." if sc["satisfaction"] is not None
                    else ""))
    rec = first_mate_trust.record(store, f, first_mate_trust.calibration(store, f["owner_id"]))
    if rec["why"]:
        lines.append(f"Trust: {rec['why']}")
    n = len(first_mate_cases.cases(store, f["id"]))
    if n:
        k = sc.get("check")
        lines.append(f"Cases: {n}" + (f"; latest check {k['score']:.0%} ({k['purpose']})" if k and k.get("score") is not None
                                      else "") + ".")
    last = card.get("last_run")
    if last:
        lines.append(f"Last run: {last['status']} at {str(last['created_at'])[:16].replace('T', ' ')} UTC"
                     + (f". {str(last.get('result') or last.get('error') or '')[:400]}" if last.get("result") or last.get("error")
                        else "."))
    for rid in store.open_proposals(f["id"])[:3]:
        waiting = [i["what"] for i in store.intents(rid) if i["status"] == "open"]
        lines.append("Waiting for your yes on First Mate or the Line: " + "; ".join(waiting[:6])
                     + (f"; and {len(waiting) - 6} more" if len(waiting) > 6 else ""))
    return "\n".join(lines)


@router.post("/fd/first-mate/agent/tool")
async def agent_tool(body: AgentToolReq) -> dict:
    """The ``first_mate`` tool: an agent, for its own owner, lists their
    factories, reads one, runs, pauses or resumes it, or reads the week. It
    never approves, carries out or raises trust: those are the owner's own."""
    from captain_claw.flight_deck import crew_runner
    from captain_claw.flight_deck.basna_routes import _AgentReq, _resolve_owner
    owner = fms.norm_user(_resolve_owner(_AgentReq(source_port=body.source_port, web_auth=body.web_auth,
                                                   owner_id=body.owner_id)) or "local")
    user = await crew_runner._user_of(owner)
    if not user:
        raise HTTPException(403, "this agent's owner could not be resolved")
    ok, why = access(user)
    if not ok:
        raise HTTPException(403, why)
    if body.action not in AGENT_ACTIONS:
        raise HTTPException(400, f"action is one of: {', '.join(AGENT_ACTIONS)}")
    store = fms.get_store()
    now = fms._utcnow()
    try:
        if body.action == "report":
            from captain_claw.flight_deck import first_mate_report
            week = body.week if body.week in ("last", "this") else "last"
            rep = first_mate_report.build(store, owner, *first_mate_report.period(now, week), now)
            title, text = first_mate_report.text(rep)
            return {"ok": True, "text": f"{title}\n\n{text}"}
        if body.action == "draft":
            from captain_claw.flight_deck import first_mate_drafting
            made = await first_mate_drafting.draft(store, owner, body.words or body.note, by="first_mate")
            d = made.get("draft") or {}
            starts = ", ".join([t["schedule"] for t in d.get("triggers") or [] if t.get("kind") == "schedule"]
                               + ["on new mail" for t in d.get("triggers") or [] if t.get("kind") == "events"])
            return {"ok": True, "id": made["id"],
                    "text": (f"Drafted \"{made['title']}\". It waits for your approval on the Line and on First Mate; "
                             f"nothing runs before that.\nWhat it does: {d.get('charter', '')[:600]}\n"
                             f"Starts: {starts or 'only when you press Run now'}."
                             + (f"\nIt came with {made['cases']} sealed test cases." if made.get("cases") else ""))}
        week = _week(store, owner, now)
        if body.action == "list":
            cards = [_card(store, f, week, now) for f in store.list_factories(owner)]
            if not cards:
                return {"ok": True, "text": "You have no factories yet. Set one up on First Mate.", "factories": []}
            on = store.prefs(owner)["on"]
            head = (f"You have {len(cards)} {'factory' if len(cards) == 1 else 'factories'}. Running on their own: "
                    f"{'on' if on else 'off (they run only when you press Run now)'}.")
            return {"ok": True, "text": "\n".join([head] + [_line_of(store, c) for c in cards]),
                    "factories": [{"id": c["id"], "title": c["title"], "state": c["state"]} for c in cards]}
        f = find_factory(store, owner, body.factory)
        if body.action == "status":
            return {"ok": True, "text": _status_text(store, _card(store, f, week, now), now), "id": f["id"]}
        if body.action == "run":
            run = await runs.start(f["id"], owner, trigger="manual", note=body.note)
            how = ("a rehearsal: nothing outside happens until you approve what it would do"
                   if run["kind"] == "rehearsal" else "it acts directly")
            return {"ok": True, "text": f"Started a run of {f['title']} ({how}). It shows on First Mate.",
                    "run_id": run["id"]}
        if body.action == "pause":
            store.set_state(f["id"], "paused", "owner", body.note or "paused by you, through your agent")
            return {"ok": True, "text": f"Paused {f['title']}. It won't start on its own until you resume it."}
        store.set_state(f["id"], "running", "owner", body.note)
        from captain_claw.flight_deck import first_mate_line
        await first_mate_line.factory_resumed(owner, f["id"])
        return {"ok": True, "text": f"{f['title']} is running again."}
    except FirstMateError as exc:
        return {"ok": False, "why": str(exc)}


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
