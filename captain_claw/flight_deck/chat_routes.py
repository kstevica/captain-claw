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
    session_id: str, limit: int = 100, before: int | None = None, since_id: int | None = None,
    user: dict = Depends(get_current_user),
):
    """The newest page of the lane's log (or the page before ``before``), or
    with ``since_id`` the log from that row on — oldest first either way."""
    db = get_db()
    rows = await db.get_chat_messages(
        _store_id(session_id, user["id"]), user["id"],
        limit=limit, before_id=before, since_id=since_id)
    return [_with_origin(r) for r in rows]


def _with_origin(row: dict) -> dict:
    """A log row, with ``origin`` on a user row the runtime wrote (a guard's
    "[system] STOP …", a relayed result, a fleet notice) that an earlier
    client stored as the user's own message: the chat shows it as a note, or
    not at all, instead of as something the user typed."""
    if row.get("role") != "user":
        return row
    from captain_claw import msg_origin

    origin = msg_origin.transcript_origin(str(row.get("content") or ""))
    return {**row, "origin": origin} if origin else row


@router.get("/sessions/{session_id}/topics")
async def list_topics(session_id: str, user: dict = Depends(get_current_user)):
    """The lane's topic dividers, oldest first, each with its first user
    message and how many turns followed before the next divider."""
    db = get_db()
    return await db.list_chat_topics(_store_id(session_id, user["id"]), user["id"])


class TopicTitleRequest(BaseModel):
    title: str = ""


@router.patch("/sessions/{session_id}/topics/{row_id}")
async def set_topic_title(
    session_id: str, row_id: int, body: TopicTitleRequest,
    user: dict = Depends(get_current_user),
):
    """Name a topic: the agent's generated title, or one the user typed."""
    db = get_db()
    ok = await db.set_chat_topic_title(_store_id(session_id, user["id"]), user["id"], row_id, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Topic not found")
    return {"ok": True, "title": " ".join(body.title.split())[:120]}


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


@router.delete("/sessions/{session_id}/messages/{row_id}")
async def delete_message(session_id: str, row_id: int, user: dict = Depends(get_current_user)):
    """Withdraw one of the user's own rows (a composer echo of a "new topic:"
    line that the agent re-sent trimmed). Replies and dividers stay."""
    db = get_db()
    ok = await db.delete_chat_message(_store_id(session_id, user["id"]), user["id"], row_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    deleted = await db.delete_chat_session(_store_id(session_id, user["id"]), user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"ok": True}
