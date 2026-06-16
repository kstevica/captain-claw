"""Flight Deck consciousness API — the read-only Observatory window plus the
manual *nudge* trigger.

Everything here is scoped to the logged-in user: you can only ever observe your
own consciousness, and a nudge only beats over your own agents. There is no
endpoint to *speak to* it — by design it is observe-only.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, status

from captain_claw.flight_deck.auth import get_optional_user
from captain_claw.flight_deck.consciousness import (
    _agent_rank,
    _user_agents,
    agent_name_for_slug,
    get_store,
    pulse,
)

router = APIRouter(prefix="/fd/consciousness", tags=["consciousness"])

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


def _auth_enabled() -> bool:
    return os.environ.get("FD_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")


def _user_id(request: Request) -> str:
    """Resolve the caller's user id. Enforces auth when enabled; falls back to
    the local single-user bucket when auth is disabled (desktop/standalone)."""
    uid = getattr(request.state, "user_id", "") or ""
    if _auth_enabled() and not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Not authenticated")
    return uid


@router.get("")
async def get_consciousness(
    request: Request,
    limit: int = 80,
    _user: dict | None = Depends(get_optional_user),
):
    """Everything the Observatory needs in one shot: state, standing
    intentions, and the most recent stream of inner life."""
    uid = _user_id(request)
    store = get_store()
    state = store.get_state(uid)
    # Don't leak the raw cursor internals to the UI.
    state.pop("cursor", None)
    # The agents this user could think through, strongest first.
    agents = sorted(_user_agents(uid), key=_agent_rank, reverse=True)
    agent_dicts = [
        {"slug": a["slug"], "name": a["name"], "model": a.get("model", ""), "offline": False}
        for a in agents
    ]
    narrator = store.get_narrator(uid)
    # Surface a pinned-but-not-running narrator so the saved choice stays
    # visible in the UI (e.g. right after restart, before agents re-attach).
    if narrator and narrator not in {a["slug"] for a in agents}:
        agent_dicts.append({
            "slug": narrator,
            "name": agent_name_for_slug(narrator),
            "model": "",
            "offline": True,
        })
    return {
        "state": state,
        "intentions": store.list_intentions(uid),
        "journal": store.list_journal(uid, limit=max(1, min(500, limit))),
        "agents": agent_dicts,
        "narrator": narrator,
    }


@router.post("/narrator")
async def set_narrator(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Choose which agent the consciousness thinks through. An empty slug means
    auto (most capable available)."""
    uid = _user_id(request)
    body = await request.json()
    slug = str((body or {}).get("slug") or "").strip()
    get_store().set_narrator(uid, slug)
    return {"ok": True, "narrator": slug}


@router.get("/journal")
async def get_journal(
    request: Request,
    limit: int = 80,
    before: str | None = None,
    _user: dict | None = Depends(get_optional_user),
):
    """Paginated journal feed (descending). Pass ``before`` (an ISO timestamp,
    e.g. the oldest entry you already have) to page back in time."""
    uid = _user_id(request)
    return {
        "journal": get_store().list_journal(
            uid, limit=max(1, min(500, limit)), before=before,
        ),
    }


@router.post("/nudge")
async def nudge(
    request: Request,
    _user: dict | None = Depends(get_optional_user),
):
    """Force one heartbeat right now, regardless of how little has changed.
    Reflects as long as there's a running agent to think with."""
    uid = _user_id(request)
    try:
        result = await pulse(uid, force=True)
    except Exception as exc:  # surfaced to the UI as a clean error
        raise HTTPException(status_code=500, detail=f"pulse failed: {exc}") from exc
    return result
