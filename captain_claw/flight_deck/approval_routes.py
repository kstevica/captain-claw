"""Approval cards (approvals.py): the routes.

An agent, before a call of a kind that may need approval (its guard,
``approvals_client``), as the person its turn is for (the owner, or on a
shared agent the member whose turn it is, proved by the turn's grant):

- ``GET  /fd/approvals/agent/policy`` — what each kind needs on this deck;
- ``POST /fd/approvals/agent`` — the decision, or ``{"pending": id}`` while
  the card waits for the person;
- ``POST /fd/approvals/agent/follow`` — wait some more;
- ``POST /fd/approvals/agent/cancel`` — the turn was stopped.

The person: ``GET /fd/approvals/mine`` (what they always allow, per agent) and
``DELETE /fd/approvals/mine/{agent}/{kind}``. Everybody signed in reads the
deck's policy at ``GET /fd/approvals/policy``; an admin sets it with ``PUT``.
The card itself is answered as any question (``POST /fd/asks/<id>``).
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.approval_kinds import KIND_LABELS, KINDS
from captain_claw.flight_deck import approvals
from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/approvals", tags=["approvals"])
agent_router = APIRouter(prefix="/fd/approvals/agent", tags=["approvals"])

# The agent routes a member's turn reaches with its grant (speaker_grants.GRANT_AWARE_PATHS).
GRANT_AWARE = frozenset({"/fd/approvals/agent", "/fd/approvals/agent/follow", "/fd/approvals/agent/cancel"})


def _uid(user: dict) -> str:
    uid = str(user.get("id") or "")
    if not uid:
        raise HTTPException(401, "Sign in first.")
    return uid


def _policy_view(policy: dict[str, str]) -> dict[str, Any]:
    return {"policy": policy, "kinds": [{"kind": k, "label": KIND_LABELS[k]} for k in KINDS],
            "modes": [{"mode": m, "label": approvals.MODE_LABELS[m]} for m in approvals.MODES]}


# ── The deck's policy ─────────────────────────────────────────────────


class PolicyBody(BaseModel):
    policy: dict[str, str]


@router.get("/policy")
async def read_policy(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    _uid(user)
    return _policy_view(await approvals.load_policy(get_db()))


@router.put("/policy")
async def write_policy(body: PolicyBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if str(user.get("role") or "") != "admin":
        raise HTTPException(403, "Only an admin sets what needs approval on this deck.")
    try:
        policy = await approvals.save_policy(get_db(), body.policy)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return _policy_view(policy)


# ── What the person always allows ─────────────────────────────────────


@router.get("/mine")
async def mine(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    uid = _uid(user)
    allowed = await approvals.always_allowed(get_db(), uid)
    from captain_claw.flight_deck.agent_names import display_name

    out = []
    for agent, kinds in sorted(allowed.items()):
        name = await display_name(agent, uid)
        out.extend({"agent": agent, "agent_name": name, "kind": k, "kind_label": KIND_LABELS[k]} for k in kinds)
    return {"always": out}


@router.delete("/mine/{agent}/{kind}")
async def forget(agent: str, kind: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    if kind not in KINDS:
        raise HTTPException(404, "No such kind.")
    await approvals.set_always(get_db(), _uid(user), agent, kind, False)
    return {"ok": True}


# ── An agent asks ─────────────────────────────────────────────────────


class AgentBody(BaseModel):
    kind: str = Field(..., max_length=20)
    tool: str = Field("", max_length=128)
    summary: str = Field("", max_length=2000)
    origin: str = Field("", max_length=60)
    session_id: str = Field("", max_length=200)
    wait: float = Field(20, ge=0, le=30)
    wait_s: float = Field(approvals.DEFAULT_WAIT_S, ge=5, le=asks.MAX_WAIT_S)
    ask_id: str = Field("", pattern=r"^(xa_[0-9a-f]{16})?$")


class FollowBody(BaseModel):
    id: str = Field(..., max_length=40)
    wait: float = Field(25, ge=0, le=30)


async def _runner(request: Request):
    from captain_claw.flight_deck.ask_routes import _runner as ask_runner

    return await ask_runner(request)


@agent_router.get("/policy")
async def agent_policy(request: Request) -> dict[str, Any]:
    from captain_claw.flight_deck.agent_identity import require_agent_caller

    require_agent_caller(request, what="reading what needs approval")
    return {"policy": await approvals.load_policy(get_db())}


def _pending(a: asks.Ask) -> dict[str, Any]:
    return {"pending": a.id, "seconds_left": max(0, int(a.deadline - time.time()))}


@agent_router.post("")
async def agent_request(body: AgentBody, request: Request) -> dict[str, Any]:
    if body.kind not in KINDS:
        raise HTTPException(422, "Unknown kind.")
    runner = await _runner(request)
    db = get_db()
    from captain_claw.flight_deck.agent_names import display_name

    name = await display_name(runner.agent, runner.user_id) or runner.agent
    summary = body.summary.strip() or body.tool
    try:
        decided, a = await approvals.request(
            db, user_id=runner.user_id, agent=runner.agent, agent_name=name, kind=body.kind,
            summary=summary, origin=body.origin, session_id=body.session_id, ask_id=body.ask_id,
            wait_s=body.wait_s)
    except asks.AskError as exc:
        return approvals._deny(f"The card couldn't be shown: {exc}", scope="error")
    if decided is not None:
        return decided
    done = await asks.wait_answer(a, body.wait)
    return await approvals.outcome(db, a.id, done) if done is not None else _pending(a)


@agent_router.post("/follow")
async def agent_follow(body: FollowBody, request: Request) -> dict[str, Any]:
    runner = await _runner(request)
    try:
        done = await asks.follow(body.id, runner.user_id, body.wait)
    except KeyError:
        raise HTTPException(404, "There's no such card (it ended a while ago).") from None
    if done.get("pending"):
        return done
    return await approvals.outcome(get_db(), body.id, done)


@agent_router.post("/cancel")
async def agent_cancel(body: FollowBody, request: Request) -> dict[str, Any]:
    runner = await _runner(request)
    cancelled = await asks.cancel(body.id, runner.user_id)
    approvals._PENDING.pop(body.id, None)
    return {"cancelled": cancelled}
