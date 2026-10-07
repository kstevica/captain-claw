"""Shared-agent usage route (PR D) — see ``shared_usage``.

Agent (no JWT): ``POST /fd/shared-agents/agent/members`` — the calling agent's
live member roster (body ``{"user_id": ""}``), plus one member's published
context (``{"user_id": "<fd id>"}``). For the OWNER's agent only: the route is
not grant-aware (``GrantGuardMiddleware`` refuses the member markers before it
runs) and refuses a member request itself as well. Never logs a name, an
email, a user id or profile text.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import context_packs as cp
from captain_claw.flight_deck import shared_usage as su
from captain_claw.flight_deck import speaker_grants
from captain_claw.flight_deck.auth import get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["shared-usage"])


class MembersBody(BaseModel):
    user_id: str = ""


@router.post(su.SHARED_USAGE_ROUTE)
async def agent_members(body: MembersBody, request: Request) -> dict:
    """The calling owner's agent's current members (part 0 §5.1)."""
    if not sharing.sharing_active():
        raise HTTPException(403, cp.SHARING_OFF_DETAIL)
    if speaker_grants.member_request(request):            # GrantGuard already 403s; belt and braces
        raise HTTPException(403, su.OWNER_ONLY_DETAIL)
    acting, rec = await cp.caller_agent(request)           # browser 403 / transport 401 / 403 NO_AGENT
    if acting is not None:
        raise HTTPException(403, su.OWNER_ONLY_DETAIL)
    uid = body.user_id or ""
    if uid and not su.USER_ID_RE.fullmatch(uid):
        raise HTTPException(400, su.BAD_USER_DETAIL)
    db = get_db()
    members, truncated = await su.roster(db, rec)
    context = None
    if uid:
        # Only an id from the (capped) roster: a member past MAX_ROSTER is 404 too.
        m = next((x for x in members if x.user_id == uid), None)
        if m is None:
            raise HTTPException(404, su.MEMBER_GONE_DETAIL)
        context = await su.member_context(db, rec, m)
    log.info("Shared-agent roster read", agent=rec.slug, members=len(members), detail=bool(uid))
    return {"agent": {"name": rec.name, "runtime": rec.runtime},
            "members": [su.member_payload(m) for m in members],
            "truncated": truncated, "context": context}
