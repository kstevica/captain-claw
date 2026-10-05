"""Chat persistence REST endpoints for Flight Deck."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/chat", tags=["chat"])

# Chats with a shared agent are keyed by the agent (``shared:<agent_ref>`` or
# ``shared:<agent_ref>::<lane>``), which every member uses alike — but
# ``chat_sessions.id`` is global. Each member's is stored under a server-derived
# ``…@<user id>``, so nobody can pre-create, read or overwrite someone else's.
# Clients never send (or see) the suffix.
SHARED_PREFIX = "shared:"


def _store_id(session_id: str, user_id: str) -> str:
    if session_id.startswith(SHARED_PREFIX):
        return f"{session_id}@{user_id}"
    return session_id


def _client_id(store_id: str, user_id: str) -> str:
    suffix = f"@{user_id}"
    if store_id.startswith(SHARED_PREFIX) and store_id.endswith(suffix):
        return store_id[: -len(suffix)]
    return store_id


class UpsertSessionRequest(BaseModel):
    id: str
    agent_id: str = ""
    agent_name: str = ""


class AddMessagesRequest(BaseModel):
    messages: list[dict]


@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    db = get_db()
    rows = await db.list_chat_sessions(user["id"])
    out = []
    for row in rows:
        sid = str(row.get("id") or "")
        client_sid = _client_id(sid, user["id"])
        out.append({**row, "id": client_sid} if client_sid != sid else row)
    return out


@router.post("/sessions")
async def upsert_session(body: UpsertSessionRequest, user: dict = Depends(get_current_user)):
    db = get_db()
    row = await db.upsert_chat_session(
        session_id=_store_id(body.id, user["id"]), user_id=user["id"],
        agent_id=body.agent_id, agent_name=body.agent_name,
    )
    if isinstance(row, dict) and "id" in row:
        row = {**row, "id": _client_id(str(row["id"]), user["id"])}
    return row


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str, limit: int = 100, before: int | None = None,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    return await db.get_chat_messages(
        _store_id(session_id, user["id"]), user["id"], limit=limit, before_id=before)


@router.post("/sessions/{session_id}/messages")
async def add_messages(
    session_id: str, body: AddMessagesRequest,
    user: dict = Depends(get_current_user),
):
    db = get_db()
    ids = await db.add_chat_messages(_store_id(session_id, user["id"]), user["id"], body.messages)
    if not ids:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"ok": True, "ids": ids}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    deleted = await db.delete_chat_session(_store_id(session_id, user["id"]), user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"ok": True}
