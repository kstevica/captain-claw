"""Chat persistence REST endpoints for Flight Deck."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/chat", tags=["chat"])


class UpsertSessionRequest(BaseModel):
    id: str
    agent_id: str = ""
    agent_name: str = ""


class AddMessagesRequest(BaseModel):
    messages: list[dict]


@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.list_chat_sessions(user["id"])


@router.post("/sessions")
async def upsert_session(body: UpsertSessionRequest, user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.upsert_chat_session(
        session_id=body.id, user_id=user["id"],
        agent_id=body.agent_id, agent_name=body.agent_name,
    )


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str, limit: int = 100, before: int | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_chat_messages(session_id, user["id"], limit=limit, before_id=before)


@router.post("/sessions/{session_id}/messages")
async def add_messages(
    session_id: str, body: AddMessagesRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ids = await db.add_chat_messages(session_id, user["id"], body.messages)
    if not ids:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"ok": True, "ids": ids}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    deleted = await db.delete_chat_session(session_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"ok": True}
