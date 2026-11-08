"""Course routes — the goals people set for the deck, themselves and their agents.

* ``GET  /fd/course``                      the caller's whole picture: the deck's goals,
                                           their own, each of their agents', the admin's
                                           settings, and flags (due, stale, retired horizon,
                                           serves nothing above it)
* ``GET  /fd/course/preview``              what one of the caller's agents receives
                                           (``?agent_ref=``; none = a new agent of theirs)
* ``POST /fd/course/goals``                a new goal: ``scope`` user, agent (one of the
                                           caller's agents) or instance (admin only)
* ``GET  /fd/course/goals/{id}``           one goal, with its links and history
* ``PATCH /fd/course/goals/{id}``          edit it
* ``POST /fd/course/goals/{id}/{action}``  pause · resume · achieve · drop · reopen ·
                                           review · accept · decline
* ``POST /fd/course/goals/{id}/links``,
  ``DELETE /fd/course/goals/{id}/links/{kind}/{ref_id}``  the evidence
* ``PUT  /fd/course/goals/{id}/value``     the current value of a goal measured by a
                                           number a person enters
* ``POST /fd/course/conflicts/{id}/dismiss``  "not a conflict"
* ``GET/PUT /fd/admin/course/settings``    the horizons each scope uses, the caps, the
                                           suggestion rules and what Flight Deck does on
                                           its own (admin only)

Plain ``get_current_user``, as on the profile routes: an admin's ``X-FD-Act-As``
does not reach another user's Course, and someone else's goal is a 404 — there is
no admin view of a user's goals. Deck-wide goals are visible to everyone and
changed only by an admin (403 otherwise).

Every change rewrites the course files of the agents it reaches
(``course_carrier.refresh_for_goal``) and says how many it updated. The rules
themselves (horizons, caps, the tree) live in :mod:`course`. A goal that is
added, accepted or reworded is checked for conflicts in the background
(:mod:`course_conflicts`).
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request

from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import course_carrier as carrier
from captain_claw.flight_deck import course_conflicts as conflicts
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(tags=["course"])

# What the page shows: everything but replaced and declined proposals.
VISIBLE = ("proposed", "active", "paused", "achieved", "dropped")

_ACTIONS: dict[str, str] = {
    "pause": "paused", "resume": "active", "achieve": "achieved", "drop": "dropped",
    "reopen": "active", "accept": "active", "decline": "declined",
}


def _uid(user: dict) -> str:
    return c.norm_user(str(user.get("id") or ""))


def _is_admin(user: dict) -> bool:
    return user.get("role") == "admin"


def _actor(user: dict) -> str:
    return f"user:{_uid(user)}"


def _bad(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


async def _my_agents(user: dict) -> list[dict[str, Any]]:
    """The caller's agents (every agent on an auth-off deck)."""
    owner = _uid(user) if carrier._auth_enabled() else None
    agents = await asyncio.to_thread(carrier.list_agents, owner)
    return [a for a in agents if a.get("ref")]


def _flags(goal: dict[str, Any], settings: dict[str, Any], visible_ids: set[str],
           now: datetime) -> dict[str, bool]:
    parent = goal.get("parent_id") or ""
    return {
        "due": c.is_due(goal, now),
        "stale": c.is_stale(goal, now),
        "horizon_retired": goal["horizon"] not in settings["horizons"].get(goal["scope"], []),
        "serves_nothing": (goal["horizon"] != "strategic" and not parent
                           and goal["status"] in ("active", "paused", "proposed")),
        "parent_gone": bool(parent) and parent not in visible_ids,
    }


def _view(goal: dict[str, Any], *, can_edit: bool, flags: dict[str, bool],
          moved: str | None = None, names: dict[str, str] | None = None,
          off: str = "", clashes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    proposer = ""
    if goal.get("origin") == "deck":
        proposer = "Flight Deck"
    elif goal.get("origin") == "agent":
        proposer = (names or {}).get(str(goal.get("proposed_by") or ""), "") or "An agent"
    return {**goal, "can_edit": can_edit, "flags": {**flags, "off_course": bool(off)},
            "last_moved": moved, "proposer": proposer, "off_course_reason": off,
            "conflicts": clashes or [],
            "measure": c.measure(goal) if goal.get("status") in ("active", "paused") else None,
            "grace_days": c.grace_days(int(goal.get("review_every_days") or 30))}


def _clashes(uid: str, admin: bool, visible: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Goal id → the open conflicts the caller is told about that it's part of,
    each naming the other goal (only between goals the caller can see)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for cf in c.get_store().conflicts(uid, admin=admin):
        a, b = visible.get(cf["goal_id"]), visible.get(cf["other_id"])
        if a is None or b is None:
            continue
        for mine, other in ((a, b), (b, a)):
            out.setdefault(mine["id"], []).append(
                {"id": cf["id"], "other_id": other["id"], "other_title": other["title"],
                 "other_scope": other["scope"], "reason": cf["reason"], "at": cf["at"]})
    return out


async def _visible_goal(goal_id: str, user: dict, *, edit: bool) -> dict[str, Any]:
    """The goal, if the caller may see it (and, with ``edit``, change it):
    someone else's is a 404, a deck-wide one a non-admin may only read (403)."""
    goal = c.get_store().get(goal_id)
    if goal is None:
        raise HTTPException(404, "Goal not found")
    if goal["scope"] == "instance":
        if edit and not _is_admin(user):
            raise HTTPException(403, "Only an admin changes the deck's goals")
        return goal
    if goal["owner_id"] != _uid(user):
        raise HTTPException(404, "Goal not found")
    if goal["scope"] == "agent":
        refs = {a["ref"] for a in await _my_agents(user)}
        if goal["agent_ref"] not in refs:
            raise HTTPException(404, "Goal not found")
    return goal


async def _refresh(goal: dict[str, Any]) -> int:
    try:
        return await carrier.refresh_for_goal(get_db(), goal)
    except Exception:  # noqa: BLE001 — a save never fails over the files
        return 0


@router.get("/fd/course")
async def get_course(user: dict = Depends(get_current_user)):
    store = c.get_store()
    uid = _uid(user)
    admin = _is_admin(user)
    settings = store.settings()
    agents = await _my_agents(user)
    refs = [a["ref"] for a in agents]
    deck = store.list_goals(scope="instance", statuses=VISIBLE)
    mine = store.list_goals(scope="user", owner_id=uid, statuses=VISIBLE)
    theirs = store.list_goals(scope="agent", owner_id=uid, agent_refs=refs, statuses=VISIBLE)
    visible = {g["id"] for g in deck + mine + theirs}
    names = {a["ref"]: a["name"] for a in agents}
    moved = store.last_moved([g["id"] for g in mine + theirs])
    moved.update(store.last_moved([g["id"] for g in deck], owner_id=uid))
    now = datetime.now(UTC)
    off = c.off_course(deck + mine + theirs, moved, now=now)
    clashes = _clashes(uid, admin, {g["id"]: g for g in deck + mine + theirs})

    def view(g: dict[str, Any], can_edit: bool) -> dict[str, Any]:
        return _view(g, can_edit=can_edit, flags=_flags(g, settings, visible, now),
                     moved=moved.get(g["id"]), names=names, off=off.get(g["id"], ""),
                     clashes=clashes.get(g["id"]))

    by_agent: dict[str, list[dict[str, Any]]] = {r: [] for r in refs}
    for g in theirs:
        by_agent.setdefault(g["agent_ref"], []).append(view(g, True))
    return {
        "settings": settings,
        "horizons": list(c.HORIZONS),
        "horizon_labels": dict(c.HORIZON_LABELS),
        "review_days": dict(c.REVIEW_DAYS),
        "text_caps": {k: c.TEXT_CAPS[k] for k in ("title", "detail", "success")},
        "is_admin": admin,
        "auth_enabled": carrier._auth_enabled(),
        "prefs": store.prefs(uid),
        "recommendations": settings["proposals"]["recommendations"],
        "agents": [{"ref": a["ref"], "name": a["name"], "slug": a["slug"],
                    "runtime": a["runtime"]} for a in agents],
        "metric_kinds": dict(c.METRIC_KIND_LABELS),
        "goals": {
            "deck": [view(g, admin) for g in deck],
            "mine": [view(g, True) for g in mine],
            "agents": by_agent,
        },
    }


@router.get("/fd/course/preview")
async def get_preview(agent_ref: str = Query(""), user: dict = Depends(get_current_user)):
    """What an agent of the caller's receives — the owner's-turn block and the
    member's-turn block — exactly as it is written to its files."""
    owner = _uid(user)
    if agent_ref:
        match = [a for a in await _my_agents(user) if a["ref"] == agent_ref]
        if not match:
            raise HTTPException(404, "Unknown agent")
        owner = match[0]["owner"]
    texts = await carrier.compose_for_agent(get_db(), owner, agent_ref)
    return {"owner": {"full": texts["owner"][0], "compact": texts["owner"][1]},
            "member": {"full": texts["deck"][0], "compact": texts["deck"][1]}}


@router.post("/fd/course/goals")
async def create_goal(body: Any = Body(...), user: dict = Depends(get_current_user)):
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    scope = body.get("scope")
    owner_id, agent_ref, origin = _uid(user), "", "user"
    if scope == "instance":
        if not _is_admin(user):
            raise HTTPException(403, "Only an admin sets the deck's goals")
        owner_id, origin = "", "admin"
    elif scope == "agent":
        agent_ref = str(body.get("agent_ref") or "")
        if agent_ref not in {a["ref"] for a in await _my_agents(user)}:
            raise HTTPException(404, "Unknown agent")
    elif scope != "user":
        raise HTTPException(400, "scope must be instance, user or agent")
    parent_id = str(body.get("parent_id") or "")
    if parent_id:
        await _visible_goal(parent_id, user, edit=False)  # someone else's: 404
    try:
        goal = c.get_store().create(
            scope=scope, horizon=body.get("horizon"), title=body.get("title"),
            owner_id=owner_id, agent_ref=agent_ref, detail=body.get("detail"),
            success=body.get("success"), parent_id=parent_id, due_at=body.get("due_at"),
            review_every_days=body.get("review_every_days"), origin=origin,
            actor=_actor(user), metric=body.get("metric"),
        )
    except c.CourseError as exc:
        raise _bad(exc) from None
    conflicts.check_soon(get_db(), goal, thinker=_uid(user))
    return {"goal": goal, "agents_updated": await _refresh(goal)}


def _links_owner(goal: dict[str, Any], user: dict) -> str | None:
    """Whose evidence the caller sees on ``goal``: everyone's on their own goals,
    only their own on a deck-wide goal (other users' work is theirs)."""
    return _uid(user) if goal["scope"] == "instance" else None


@router.get("/fd/course/goals/{goal_id}")
async def get_goal(goal_id: str, user: dict = Depends(get_current_user)):
    goal = await _visible_goal(goal_id, user, edit=False)
    store = c.get_store()
    events = store.events(goal_id)
    if goal["scope"] == "instance":
        # A link event names someone's work: the caller sees only their own.
        events = [e for e in events if e["event"] != "linked"
                  or str(e["detail"].get("owner_id") or "") == _uid(user)]
    clashes = []
    for cf in store.conflicts(_uid(user), admin=_is_admin(user)):
        if goal_id not in (cf["goal_id"], cf["other_id"]):
            continue
        other = store.get(cf["other_id"] if cf["goal_id"] == goal_id else cf["goal_id"])
        if other is not None:
            clashes.append({"id": cf["id"], "other_id": other["id"], "other_title": other["title"],
                            "other_scope": other["scope"], "reason": cf["reason"], "at": cf["at"]})
    return {"goal": goal, "links": store.links(goal_id, owner_id=_links_owner(goal, user)),
            "events": events, "can_edit": goal["scope"] != "instance" or _is_admin(user),
            "measure": c.measure(goal), "conflicts": clashes}


@router.patch("/fd/course/goals/{goal_id}")
async def edit_goal(goal_id: str, body: Any = Body(...), user: dict = Depends(get_current_user)):
    goal = await _visible_goal(goal_id, user, edit=True)
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    parent_id = str(body.get("parent_id") or "")
    if parent_id and parent_id != goal.get("parent_id"):
        await _visible_goal(parent_id, user, edit=False)
    try:
        updated = c.get_store().update(goal_id, body, actor=_actor(user))
    except c.CourseError as exc:
        raise _bad(exc) from None
    except c.CourseNotFoundError:
        raise HTTPException(404, "Goal not found") from None
    if any(updated.get(f) != goal.get(f) for f in ("title", "success", "detail")):
        conflicts.check_soon(get_db(), updated, thinker=_uid(user))
    return {"goal": updated, "agents_updated": await _refresh(updated)}


# What a person may link from the UI, and how their ownership of it is checked.
PEOPLE_LINK_KINDS = ("plan", "project", "schedule", "swarm_run", "task", "autonomy_action")


async def _owns_item(kind: str, ref_id: str, user: dict) -> bool:
    """The caller's own plan, scheduled task, Swarm run, Taskboard task or
    Autonomous Work action (projects belong to the whole deck). False when the
    item can't be found or read."""
    uid = _uid(user)
    try:
        if kind == "project":
            return True
        if kind == "plan":
            from captain_claw.flight_deck.plans import get_store as plans_store
            plan = plans_store().get_plan(ref_id)
            return bool(plan) and c.norm_user(plan.get("user_id")) == uid
        if kind == "autonomy_action":
            from captain_claw.flight_deck.autonomy import get_store as autonomy_store
            row = autonomy_store().get_action(ref_id)
            return bool(row) and c.norm_user(row.get("user_id")) == uid
        if kind == "schedule":
            from captain_claw.flight_deck.fd_scheduler import get_store as scheduler_store
            job = scheduler_store().get(ref_id)
            if not job:
                return False
            owner = str(job.get("owner_id") or "")
            return c.norm_user(owner) == uid or (not owner and _is_admin(user))
        if kind == "swarm_run":
            from captain_claw.flight_deck import captain_swarm_loop
            store = captain_swarm_loop.get_store()
            run = await store.get_run(ref_id) if store is not None else None
            return bool(run) and c.norm_user(run.get("owner_id")) == uid
        if kind == "task":
            from captain_claw.flight_deck.taskboard import store as taskboard_store
            return taskboard_store().get_task(uid, ref_id) is not None
    except Exception:  # noqa: BLE001 — can't tell: not theirs
        return False
    return False


async def _check_item(kind: str, ref_id: str, user: dict) -> None:
    if kind not in PEOPLE_LINK_KINDS:
        raise HTTPException(400, f"kind must be one of {', '.join(PEOPLE_LINK_KINDS)}")
    if not ref_id or not await _owns_item(kind, ref_id, user):
        raise HTTPException(404, "Not found")


async def _linkable_goal(goal_id: str, user: dict) -> dict[str, Any]:
    """A goal the caller may link their own work to: any goal they can see —
    a deck-wide one too (only they see their links on it)."""
    return await _visible_goal(goal_id, user, edit=False)


@router.post("/fd/course/goals/{goal_id}/links")
async def add_link(goal_id: str, body: Any = Body(...), user: dict = Depends(get_current_user)):
    await _linkable_goal(goal_id, user)
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    kind, ref_id = str(body.get("kind") or ""), str(body.get("ref_id") or "")
    await _check_item(kind, ref_id, user)
    try:
        link = c.get_store().add_link(goal_id, kind, ref_id, note=body.get("note"),
                                      actor=_actor(user), owner_id=_uid(user))
    except c.CourseError as exc:
        raise _bad(exc) from None
    return {"link": link}


@router.delete("/fd/course/goals/{goal_id}/links/{kind}/{ref_id}")
async def remove_link(goal_id: str, kind: str, ref_id: str,
                      user: dict = Depends(get_current_user)):
    goal = await _linkable_goal(goal_id, user)
    owner = _links_owner(goal, user)
    if not c.get_store().remove_link(goal_id, kind, ref_id, owner_id=owner):
        raise HTTPException(404, "Link not found")
    return {"ok": True}


# ── "Serves a goal": one item's goal, from the item's own page ──────────


@router.get("/fd/course/served")
async def get_served(kind: str = Query(...), ref_id: str = Query(...),
                     user: dict = Depends(get_current_user)):
    """The goal the caller linked this item to ({goal: null} when none)."""
    await _check_item(kind, ref_id, user)
    uid = _uid(user)
    store = c.get_store()
    for link in store.links_for(kind, ref_id):
        if link["owner_id"] != uid:
            continue
        goal = store.get(link["goal_id"])
        if goal and goal["status"] in VISIBLE:
            return {"goal": {"id": goal["id"], "title": goal["title"], "scope": goal["scope"],
                             "horizon": goal["horizon"], "status": goal["status"]}}
    return {"goal": None}


@router.put("/fd/course/served")
async def put_served(body: Any = Body(...), user: dict = Depends(get_current_user)):
    """Set (``goal_id``) or clear ("") the one goal an item serves: the caller's
    earlier link for it is replaced. ``note`` names the item (its title)."""
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    kind, ref_id = str(body.get("kind") or ""), str(body.get("ref_id") or "")
    await _check_item(kind, ref_id, user)
    goal_id = str(body.get("goal_id") or "")
    goal = await _linkable_goal(goal_id, user) if goal_id else None
    if goal is not None and goal["status"] not in ("active", "paused"):
        raise HTTPException(400, "That goal isn't active")
    uid = _uid(user)
    store = c.get_store()
    for link in store.links_for(kind, ref_id):
        if link["owner_id"] == uid and link["goal_id"] != goal_id:
            store.remove_link(link["goal_id"], kind, ref_id, owner_id=uid)
    if goal is None:
        return {"goal": None}
    try:
        store.add_link(goal_id, kind, ref_id, note=body.get("note"), actor=_actor(user),
                       owner_id=uid)
    except c.CourseError as exc:
        raise _bad(exc) from None
    return {"goal": {"id": goal["id"], "title": goal["title"], "scope": goal["scope"],
                     "horizon": goal["horizon"], "status": goal["status"]}}


@router.post("/fd/course/goals/{goal_id}/{action}")
async def goal_action(goal_id: str, action: str, user: dict = Depends(get_current_user)):
    if action != "review" and action not in _ACTIONS:
        raise HTTPException(404, "Unknown action")
    goal = await _visible_goal(goal_id, user, edit=True)
    store = c.get_store()
    if action in ("accept", "decline") and goal["status"] != "proposed":
        raise HTTPException(409, f"This goal is {goal['status']}, not a proposal")
    try:
        if action == "review":
            updated = store.review(goal_id, actor=_actor(user))
        elif action == "accept":
            updated = store.accept(goal_id, actor=_actor(user))
        else:
            updated = store.set_status(goal_id, _ACTIONS[action], actor=_actor(user))
    except c.CourseError as exc:
        raise _bad(exc) from None
    except c.CourseNotFoundError:
        raise HTTPException(404, "Goal not found") from None
    # Its "Still on course?" / "Goal proposed" items are answered, whatever was decided.
    await carrier.resolve_reviews(get_db(), goal)
    if action in ("accept", "decline"):
        from captain_claw.flight_deck import course_proposals

        await course_proposals.resolve(get_db(), goal)
    if action in ("achieve", "drop"):
        await conflicts.resolve_for_goal(get_db(), goal)
    if action == "accept":
        if goal.get("supersedes_id"):
            replaced = store.get(goal["supersedes_id"])
            if replaced is not None:
                await conflicts.resolve_for_goal(get_db(), replaced)
        conflicts.check_soon(get_db(), updated, thinker=_uid(user))
    return {"goal": updated, "agents_updated": await _refresh(updated)}


@router.put("/fd/course/goals/{goal_id}/value")
async def set_value(goal_id: str, body: Any = Body(...), user: dict = Depends(get_current_user)):
    """The current value of a goal measured by a number a person enters."""
    await _visible_goal(goal_id, user, edit=True)
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    try:
        updated = c.get_store().set_metric_value(goal_id, body.get("value"), actor=_actor(user))
    except c.CourseError as exc:
        raise _bad(exc) from None
    return {"goal": updated, "measure": c.measure(updated)}


@router.post("/fd/course/conflicts/{conflict_id}/dismiss")
async def dismiss_conflict(conflict_id: str, user: dict = Depends(get_current_user)):
    """"Not a conflict": only someone it was told to; the pair isn't flagged again."""
    store = c.get_store()
    cf = store.get_conflict(conflict_id)
    allowed = cf is not None and (
        cf["owner_id"] == _uid(user) or (not cf["owner_id"] and _is_admin(user)))
    if not allowed:
        raise HTTPException(404, "Not found")
    done = store.dismiss_conflict(conflict_id, actor=_actor(user))
    await conflicts.resolve(get_db(), done)
    return {"conflict": done}


# ── The caller's own preferences ───────────────────────────────────────


@router.get("/fd/course/prefs")
async def get_prefs(user: dict = Depends(get_current_user)):
    return c.get_store().prefs(_uid(user))


@router.put("/fd/course/prefs")
async def put_prefs(body: Any = Body(...), user: dict = Depends(get_current_user)):
    try:
        return c.get_store().set_prefs(_uid(user), body)
    except c.CourseError as exc:
        raise _bad(exc) from None


# ── The admin's settings ───────────────────────────────────────────────


@router.get("/fd/admin/course/settings")
async def get_settings(admin: dict = Depends(require_admin)):
    return {
        **c.get_store().settings(),
        "defaults": {"horizons": c.DEFAULT_HORIZONS, "caps": c.DEFAULT_CAPS,
                     "proposals": c.DEFAULT_PROPOSALS, "automation": c.DEFAULT_AUTOMATION},
        "cap_range": list(c.CAP_RANGE),
        "per_week_range": list(c.PER_WEEK_RANGE),
        "scopes": list(c.SCOPES),
        "horizons_all": list(c.HORIZONS),
        "horizon_labels": dict(c.HORIZON_LABELS),
    }


@router.put("/fd/admin/course/settings")
async def put_settings(body: Any = Body(...), admin: dict = Depends(require_admin)):
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    try:
        saved = c.get_store().set_settings(horizons=body.get("horizons"), caps=body.get("caps"),
                                           proposals=body.get("proposals"),
                                           automation=body.get("automation"))
    except c.CourseError as exc:
        raise _bad(exc) from None
    # No agent's files change: goals at a horizon a scope no longer uses are
    # kept and still reach the agents; only new ones are refused.
    try:
        await get_db().log_usage(str(admin.get("id") or ""), "course_settings_update",
                                 json.dumps(saved))
    except Exception:  # noqa: BLE001 — never fail the save over its audit row
        pass
    return saved


# ── The agent's side: the course tool (course_client) ──────────────────
#
# Owner only: an agent reads and reports on its owner's course, never a
# member's (it sends no speaker headers here). Who is calling is this deck's own
# record of the X-Agent-Auth token it issued — never the agent's word.

AGENT_LINK_KINDS = ("session", "schedule", "project", "task", "plan")


async def _agent_caller(request: Request) -> tuple[str, str, str]:
    """``(owner, agent_ref, name)`` of the calling agent, or 401 / 403."""
    from captain_claw.flight_deck import google_oauth_routes as _g
    from captain_claw.flight_deck.server import _find_agent_by_auth

    _g._authorize_agent_call(request)
    matched, owner, slug = _find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
    if not matched or not slug:
        raise HTTPException(403, "This Flight Deck doesn't know this agent")
    if carrier._auth_enabled():
        owner = await _g._agent_owner(request, service="Course")
    agents = await asyncio.to_thread(carrier.list_agents, None)
    match = next((a for a in agents if a["slug"] == slug and a["runtime"] == "process"), None) \
        or next((a for a in agents if a["slug"] == slug), None)
    return c.norm_user(owner), (match or {}).get("ref", ""), (match or {}).get("name", slug)


def _agent_goals(owner: str, ref: str) -> list[dict[str, Any]]:
    """What the agent may see: the deck's goals, its owner's and its own —
    active or paused."""
    store = c.get_store()
    live = ("active", "paused")
    goals = store.list_goals(scope="instance", statuses=live)
    goals += store.list_goals(scope="user", owner_id=owner, statuses=live)
    if ref:
        goals += store.list_goals(scope="agent", owner_id=owner, agent_refs=[ref], statuses=live)
    return goals


def _agent_goal(owner: str, ref: str, goal_id: Any) -> dict[str, Any]:
    gid = c.goal_ref(goal_id)
    for g in _agent_goals(owner, ref):
        if g["id"] == gid:
            return g
    raise HTTPException(404, "No such goal on your course")


@router.post("/fd/course/agent/view")
async def agent_view(request: Request) -> dict:
    owner, ref, _name = await _agent_caller(request)
    goals = _agent_goals(owner, ref)
    store = c.get_store()
    mine = [g["id"] for g in goals if g["scope"] != "instance"]
    moved = store.last_moved(mine)
    moved.update(store.last_moved([g["id"] for g in goals if g["scope"] == "instance"],
                                  owner_id=owner))
    now = datetime.now(UTC)
    return {"goals": [{
        "id": g["id"], "scope": g["scope"], "horizon": g["horizon"], "title": g["title"],
        "success": g["success"], "detail": g["detail"][:400], "status": g["status"],
        "parent_id": g["parent_id"], "due_at": g["due_at"], "review_at": g["review_at"],
        "progress": g["progress"], "progress_note": g["progress_note"],
        "last_moved": moved.get(g["id"]), "stale": c.is_stale(g, now),
        "own": g["scope"] == "agent", "measure": c.measure(g),
    } for g in goals]}


@router.post("/fd/course/agent/link")
async def agent_link(request: Request, body: dict = Body(default={})) -> dict:
    owner, ref, name = await _agent_caller(request)
    goal = _agent_goal(owner, ref, body.get("goal_id"))
    kind, ref_id = str(body.get("kind") or ""), str(body.get("ref_id") or "").strip()
    if kind not in AGENT_LINK_KINDS:
        raise HTTPException(400, f"kind must be one of {', '.join(AGENT_LINK_KINDS)}")
    if kind != "session" and not await _owns_item(kind, ref_id, {"id": owner, "role": "user"}):
        raise HTTPException(404, f"No such {kind} of your owner's")
    try:
        link = c.get_store().add_link(goal["id"], kind, ref_id, note=body.get("note"),
                                      actor=f"agent:{ref}", owner_id=owner)
    except c.CourseError as exc:
        raise _bad(exc) from None
    return {"link": link, "goal": goal["title"]}


@router.post("/fd/course/agent/report")
async def agent_report(request: Request, body: dict = Body(default={})) -> dict:
    """A progress estimate on the owner's goal or the agent's own. A deck-wide
    goal is everyone's: no single agent reports on it. ``looks_achieved`` asks
    the owner on the Line (at most once a day per goal); it never marks it done."""
    owner, ref, name = await _agent_caller(request)
    goal = _agent_goal(owner, ref, body.get("goal_id"))
    if goal["scope"] == "instance":
        raise HTTPException(400, "A deck-wide goal belongs to everyone: report on your "
                                 "owner's goals or your own")
    try:
        updated = c.get_store().set_progress(goal["id"], body.get("progress"), body.get("note"),
                                             actor=f"agent:{ref}")
    except c.CourseError as exc:
        raise _bad(exc) from None
    asked = False
    if body.get("looks_achieved") is True:
        note = " ".join(str(body.get("note") or "").split())[:400]
        asked = await c.ask_looks_achieved(
            goal, source=name, trust="agent",
            body=f"{name} thinks this goal is done." + (f"\n\n{note}" if note else ""))
    return {"goal": {"id": updated["id"], "title": updated["title"],
                     "progress": updated["progress"]}, "asked_owner": asked}


_PROPOSE_FIELDS = ("horizon", "title", "success", "detail", "parent_id", "due_at", "rationale")


@router.post("/fd/course/agent/propose")
async def agent_propose(request: Request, body: dict = Body(default={})) -> dict:
    """A goal the agent suggests — for itself (``scope="agent"``) or for its
    owner (``scope="user"``). It waits for the owner (unless they let their
    agents add their own short-term goals)."""
    owner, ref, name = await _agent_caller(request)
    scope = str(body.get("scope") or "agent")
    if scope not in ("agent", "user"):
        raise HTTPException(400, "scope must be agent (your own) or user (your owner's)")
    if scope == "agent" and not ref:
        raise HTTPException(403, "This Flight Deck can't tell which agent you are")
    parent = c.goal_ref(body.get("parent_id"))
    if parent:
        _agent_goal(owner, ref, parent)   # one it can see: the deck's, its owner's, its own
    from captain_claw.flight_deck import course_proposals

    fields = {k: body.get(k) for k in _PROPOSE_FIELDS if k in body}
    fields["parent_id"] = parent
    try:
        out = await course_proposals.submit(owner_id=owner, origin="agent", proposed_by=ref,
                                            proposer=name, scope=scope,
                                            agent_ref=ref if scope == "agent" else "", **fields)
    except c.CourseError as exc:
        raise _bad(exc) from None
    g = out["goal"]
    return {"goal": {"id": g["id"], "title": g["title"], "status": g["status"]},
            "auto_accepted": out["auto_accepted"]}


@router.post("/fd/course/agent/propose_change")
async def agent_propose_change(request: Request, body: dict = Body(default={})) -> dict:
    """A change to one of the owner's goals or the agent's own, waiting for the owner."""
    owner, ref, name = await _agent_caller(request)
    goal = _agent_goal(owner, ref, body.get("goal_id"))
    if goal["scope"] == "instance":
        raise HTTPException(400, "Only an admin changes the deck's goals")
    changes = body.get("changes") if isinstance(body.get("changes"), dict) else {}
    if "parent_id" in changes:
        changes = {**changes, "parent_id": c.goal_ref(changes["parent_id"])}
        if changes["parent_id"]:
            _agent_goal(owner, ref, changes["parent_id"])
    from captain_claw.flight_deck import course_proposals

    try:
        out = await course_proposals.submit(owner_id=owner, origin="agent", proposed_by=ref,
                                            proposer=name, change_of=goal["id"], changes=changes,
                                            rationale=body.get("rationale", ""))
    except c.CourseError as exc:
        raise _bad(exc) from None
    except c.CourseNotFoundError:
        raise HTTPException(404, "No such goal on your course") from None
    g = out["goal"]
    return {"goal": {"id": g["id"], "title": g["title"], "status": g["status"]},
            "auto_accepted": out["auto_accepted"]}
