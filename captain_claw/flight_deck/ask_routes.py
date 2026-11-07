"""Questions for the person (person_asks): the routes.

The browser (every signed-in tab, the ``PersonAsks`` dialog):

- ``GET /fd/asks?wait&known`` — the person's open questions, at once when they
  differ from *known* (the ids the tab shows), else after *wait* seconds;
- ``POST /fd/asks/<id>`` — the answers (checked), or ``{"declined": true}``.

An agent (its ``ask_user`` tool), as the person its turn is for — the agent's
owner on Flight Deck's own record, or on a shared agent the member whose turn
it is (the turn's speaker grant, ``external_tools_routes.runner_of``):

- ``POST /fd/asks/agent`` — ask, and wait a little: the outcome, or
  ``{"pending": id}`` to follow;
- ``POST /fd/asks/agent/follow`` — wait some more;
- ``POST /fd/asks/agent/cancel`` — the turn was stopped: the question closes
  (also when the cancel arrives before the question opened).

``agent_router`` is included before ``router`` in server.py, so
``/fd/asks/agent`` never reads as a question's id.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck import person_asks as asks
from captain_claw.flight_deck.auth import get_current_user

router = APIRouter(prefix="/fd/asks", tags=["asks"])
agent_router = APIRouter(prefix="/fd/asks/agent", tags=["asks"])

# The agent routes a member's turn reaches with its grant (speaker_grants.GRANT_AWARE_PATHS).
GRANT_AWARE = frozenset({"/fd/asks/agent", "/fd/asks/agent/follow", "/fd/asks/agent/cancel"})


def _uid(user: dict) -> str:
    uid = str(user.get("id") or "")
    if not uid:
        raise HTTPException(401, "Sign in first.")
    return uid


# ── The person's tabs ────────────────────────────────────────────────────


class AnswerBody(BaseModel):
    answers: dict[str, Any] | None = None
    declined: bool = False


@router.get("")
async def my_asks(wait: float = Query(0, ge=0, le=30), known: str = Query("", max_length=2000),
                  user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """The questions waiting for this person: a custom tool's, or an agent's."""
    seen = {k for k in known.split(",") if k}
    return {"asks": await asks.wait_for_change(_uid(user), seen, wait)}


@router.post("/{ask_id}")
async def answer(ask_id: str, body: AnswerBody, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    try:
        asks.answer(ask_id, _uid(user), answers=body.answers, declined=body.declined)
    except KeyError:
        raise HTTPException(404, "That question isn't open any more.") from None
    except asks.AskError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"ok": True}


# ── An agent asks (its ask_user tool) ────────────────────────────────────


class AgentAskBody(BaseModel):
    questions: list[dict[str, Any]] | dict[str, Any]
    wait_s: float = Field(asks.DEFAULT_WAIT_S, ge=1, le=asks.MAX_WAIT_S)
    # Where the agent's turn came from (a turn source): nobody to ask on a
    # schedule; elsewhere than Flight Deck the agent asks in its reply.
    origin: str = Field("", max_length=60)
    # How long to wait before answering "pending"; the agent then follows it.
    wait: float = Field(20, ge=0, le=30)
    # The id the agent will follow (and can cancel) it by, chosen by the agent.
    ask_id: str = Field("", pattern=r"^(xa_[0-9a-f]{16})?$")


class FollowBody(BaseModel):
    id: str = Field(..., max_length=40)
    wait: float = Field(25, ge=0, le=30)


async def _runner(request: Request):
    """Who the agent's turn is for (as custom tools decide it): the member of a
    shared agent's turn, proved by its grant, else the agent's owner."""
    from captain_claw.flight_deck import speaker_grants
    from captain_claw.flight_deck.agent_identity import require_agent_caller
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    from captain_claw.flight_deck.external_tools_routes import Runner

    if speaker_grants.member_request(request):
        m = await speaker_grants.acting_member(request)
        if m is None:  # not reached: a marked request returns a member or raises
            raise HTTPException(403, speaker_grants.NO_TURN_DETAIL)
        return Runner(m.user_id, m.slug, True)
    caller = require_agent_caller(request, what="asking the person")
    owner = caller.owner or ("" if _fd_auth_enabled() else "local")
    if not owner:
        raise HTTPException(403, "This agent has no owner on record, so it has nobody to ask.")
    return Runner(owner, caller.slug, False)


def _agent_name(slug: str) -> str:
    from captain_claw.flight_deck import server as srv

    entry = srv._load_process_registry().get(slug) if slug else None
    return str((entry or {}).get("name") or slug or "Your agent")


@agent_router.post("")
async def agent_ask(body: AgentAskBody, request: Request) -> dict[str, Any]:
    """Ask the person this turn is for; their answer, or ``{"pending": id}``."""
    runner = await _runner(request)
    if not asks.attended(body.origin):
        return {"ok": False, "no_answer": True, "nobody": True,
                "error": "Nobody is here to answer: this turn started on its own (a schedule or another "
                         "automatic start)."}
    if not asks.in_flight_deck(body.origin):
        return {"ok": False, "no_answer": True, "elsewhere": True,
                "error": f"They're talking to you in {body.origin.split(':', 1)[0]}, where this dialog doesn't "
                         "open: ask in your reply instead."}
    try:
        questions = asks.check_questions(body.questions)
        name = _agent_name(runner.agent)
        a = await asks.open_ask(runner.user_id, source="agent", title=name, agent_name=name,
                                questions=questions, wait_s=body.wait_s, ask_id=body.ask_id)
    except asks.AskError as exc:
        raise HTTPException(422, str(exc)) from None
    done = await asks.wait_answer(a, body.wait)
    return done if done is not None else {"pending": a.id, "seconds_left": max(0, int(a.deadline - time.time()))}


@agent_router.post("/follow")
async def agent_follow(body: FollowBody, request: Request) -> dict[str, Any]:
    runner = await _runner(request)
    try:
        return await asks.follow(body.id, runner.user_id, body.wait)
    except KeyError:
        raise HTTPException(404, "There's no such question (it ended a while ago).") from None


@agent_router.post("/cancel")
async def agent_cancel(body: FollowBody, request: Request) -> dict[str, Any]:
    runner = await _runner(request)
    return {"cancelled": await asks.cancel(body.id, runner.user_id)}
