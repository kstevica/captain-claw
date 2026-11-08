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
* ``GET/PUT /fd/admin/course/settings``    the horizons each scope uses and the caps
                                           (admin only)

Plain ``get_current_user``, as on the profile routes: an admin's ``X-FD-Act-As``
does not reach another user's Course, and someone else's goal is a 404 — there is
no admin view of a user's goals. Deck-wide goals are visible to everyone and
changed only by an admin (403 otherwise).

Every change rewrites the course files of the agents it reaches
(``course_carrier.refresh_for_goal``) and says how many it updated. The rules
themselves (horizons, caps, the tree) live in :mod:`course`.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from captain_claw.flight_deck import course as c
from captain_claw.flight_deck import course_carrier as carrier
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


def _view(goal: dict[str, Any], *, can_edit: bool, flags: dict[str, bool]) -> dict[str, Any]:
    return {**goal, "can_edit": can_edit, "flags": flags,
            "grace_days": c.grace_days(int(goal.get("review_every_days") or 30))}


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
    now = datetime.now(UTC)
    by_agent: dict[str, list[dict[str, Any]]] = {r: [] for r in refs}
    for g in theirs:
        by_agent.setdefault(g["agent_ref"], []).append(
            _view(g, can_edit=True, flags=_flags(g, settings, visible, now)))
    return {
        "settings": settings,
        "horizons": list(c.HORIZONS),
        "horizon_labels": dict(c.HORIZON_LABELS),
        "review_days": dict(c.REVIEW_DAYS),
        "text_caps": {k: c.TEXT_CAPS[k] for k in ("title", "detail", "success")},
        "is_admin": admin,
        "auth_enabled": carrier._auth_enabled(),
        "agents": [{"ref": a["ref"], "name": a["name"], "slug": a["slug"],
                    "runtime": a["runtime"]} for a in agents],
        "goals": {
            "deck": [_view(g, can_edit=admin, flags=_flags(g, settings, visible, now))
                     for g in deck],
            "mine": [_view(g, can_edit=True, flags=_flags(g, settings, visible, now))
                     for g in mine],
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
            actor=_actor(user),
        )
    except c.CourseError as exc:
        raise _bad(exc) from None
    return {"goal": goal, "agents_updated": await _refresh(goal)}


@router.get("/fd/course/goals/{goal_id}")
async def get_goal(goal_id: str, user: dict = Depends(get_current_user)):
    goal = await _visible_goal(goal_id, user, edit=False)
    store = c.get_store()
    return {"goal": goal, "links": store.links(goal_id), "events": store.events(goal_id),
            "can_edit": goal["scope"] != "instance" or _is_admin(user)}


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
    return {"goal": updated, "agents_updated": await _refresh(updated)}


@router.post("/fd/course/goals/{goal_id}/links")
async def add_link(goal_id: str, body: Any = Body(...), user: dict = Depends(get_current_user)):
    await _visible_goal(goal_id, user, edit=True)
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    try:
        link = c.get_store().add_link(goal_id, str(body.get("kind") or ""), body.get("ref_id"),
                                      note=body.get("note"), actor=_actor(user))
    except c.CourseError as exc:
        raise _bad(exc) from None
    return {"link": link}


@router.delete("/fd/course/goals/{goal_id}/links/{kind}/{ref_id}")
async def remove_link(goal_id: str, kind: str, ref_id: str,
                      user: dict = Depends(get_current_user)):
    await _visible_goal(goal_id, user, edit=True)
    if not c.get_store().remove_link(goal_id, kind, ref_id):
        raise HTTPException(404, "Link not found")
    return {"ok": True}


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
        else:
            updated = store.set_status(goal_id, _ACTIONS[action], actor=_actor(user))
    except c.CourseError as exc:
        raise _bad(exc) from None
    except c.CourseNotFoundError:
        raise HTTPException(404, "Goal not found") from None
    # Its "Still on course?" items are answered, whatever was decided.
    await carrier.resolve_reviews(get_db(), goal)
    return {"goal": updated, "agents_updated": await _refresh(updated)}


# ── The admin's settings ───────────────────────────────────────────────


@router.get("/fd/admin/course/settings")
async def get_settings(admin: dict = Depends(require_admin)):
    return {
        **c.get_store().settings(),
        "defaults": {"horizons": c.DEFAULT_HORIZONS, "caps": c.DEFAULT_CAPS},
        "cap_range": list(c.CAP_RANGE),
        "scopes": list(c.SCOPES),
        "horizons_all": list(c.HORIZONS),
        "horizon_labels": dict(c.HORIZON_LABELS),
    }


@router.put("/fd/admin/course/settings")
async def put_settings(body: Any = Body(...), admin: dict = Depends(require_admin)):
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    try:
        saved = c.get_store().set_settings(horizons=body.get("horizons"), caps=body.get("caps"))
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
