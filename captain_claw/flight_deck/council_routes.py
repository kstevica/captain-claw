"""Council of Agents REST endpoints for Flight Deck."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/council", tags=["council"])


# ── Request models ───────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: str
    topic: str
    session_type: str = "brainstorm"
    verbosity: str = "message"
    max_rounds: int = Field(default=5, ge=1, le=20)
    moderator_mode: str = "round-robin"
    moderator_agent: str = ""
    agents: str = "[]"
    config: str = "{}"


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    topic: str | None = None
    status: str | None = None
    current_round: int | None = None
    moderator_mode: str | None = None
    moderator_agent: str | None = None
    agents: str | None = None
    pinned_ids: str | None = None
    config: str | None = None


class AddMessagesRequest(BaseModel):
    messages: list[dict]


class AddVotesRequest(BaseModel):
    votes: list[dict]


class UpsertArtifactRequest(BaseModel):
    kind: str
    agent_id: str = ""
    agent_name: str = ""
    content: str = ""


# ── Session endpoints ────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.list_council_sessions(user["id"])


@router.post("/sessions")
async def create_session(body: CreateSessionRequest, user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.create_council_session(
        user_id=user["id"], title=body.title, topic=body.topic,
        session_type=body.session_type, verbosity=body.verbosity,
        max_rounds=body.max_rounds, moderator_mode=body.moderator_mode,
        moderator_agent=body.moderator_agent, agents=body.agents,
        config=body.config,
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    sess = await db.get_council_session(session_id, user["id"])
    if not sess:
        raise HTTPException(status_code=404, detail="Council session not found")
    return sess


@router.put("/sessions/{session_id}")
async def update_session(
    session_id: str, body: UpdateSessionRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    ok = await db.update_council_session(session_id, user["id"], **fields)
    if not ok:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    deleted = await db.delete_council_session(session_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}


# ── Message endpoints ────────────────────────────────────────────

@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str, round: int | None = None, limit: int = 500,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_council_messages(session_id, user["id"], round_num=round, limit=limit)


@router.post("/sessions/{session_id}/messages")
async def add_messages(
    session_id: str, body: AddMessagesRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ids = await db.add_council_messages(session_id, user["id"], body.messages)
    if not ids:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "ids": ids}


@router.put("/sessions/{session_id}/messages/{message_id}/pin")
async def toggle_pin(
    session_id: str, message_id: int,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ok = await db.toggle_council_pin(session_id, user["id"], message_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}


# ── Vote endpoints ───────────────────────────────────────────────

@router.post("/sessions/{session_id}/votes")
async def add_votes(
    session_id: str, body: AddVotesRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ids = await db.add_council_votes(session_id, user["id"], body.votes)
    if not ids:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "ids": ids}


@router.get("/sessions/{session_id}/votes")
async def get_votes(
    session_id: str, round: int | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_council_votes(session_id, user["id"], round_num=round)


# ── Artifact endpoints ──────────────────────────────────────────

@router.get("/sessions/{session_id}/artifacts")
async def get_artifacts(
    session_id: str, kind: str | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_council_artifacts(session_id, user["id"], kind=kind)


@router.post("/sessions/{session_id}/artifacts")
async def upsert_artifact(
    session_id: str, body: UpsertArtifactRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    art_id = await db.upsert_council_artifact(
        session_id, user["id"],
        kind=body.kind, agent_id=body.agent_id,
        agent_name=body.agent_name, content=body.content,
    )
    if not art_id:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True, "id": art_id}


@router.delete("/sessions/{session_id}/artifacts")
async def delete_artifacts(
    session_id: str, kind: str | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ok = await db.delete_council_artifacts(session_id, user["id"], kind=kind)
    if not ok:
        raise HTTPException(status_code=404, detail="Council session not found")
    return {"ok": True}
