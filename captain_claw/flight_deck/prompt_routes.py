"""Prompt Builder REST endpoints for Flight Deck."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/prompts", tags=["prompts"])


class PromptCreate(BaseModel):
    title: str = ""
    content: str = ""
    files: list[str] = []
    tags: list[str] = []


class PromptUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    files: list[str] | None = None
    tags: list[str] | None = None


@router.get("")
async def list_prompts(user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.list_prompts(user["id"])


@router.get("/{prompt_id}")
async def get_prompt(prompt_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    p = await db.get_prompt(prompt_id, user["id"])
    if not p:
        raise HTTPException(404, "Prompt not found")
    return p


@router.post("")
async def create_prompt(body: PromptCreate, user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.create_prompt(
        user["id"],
        title=body.title,
        content=body.content,
        files=json.dumps(body.files),
        tags=json.dumps(body.tags),
    )


@router.put("/{prompt_id}")
async def update_prompt(prompt_id: str, body: PromptUpdate, user: dict = Depends(get_current_user)):
    db = get_db()
    fields = {}
    if body.title is not None:
        fields["title"] = body.title
    if body.content is not None:
        fields["content"] = body.content
    if body.files is not None:
        fields["files"] = json.dumps(body.files)
    if body.tags is not None:
        fields["tags"] = json.dumps(body.tags)
    result = await db.update_prompt(prompt_id, user["id"], **fields)
    if not result:
        raise HTTPException(404, "Prompt not found")
    return result


@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    ok = await db.delete_prompt(prompt_id, user["id"])
    if not ok:
        raise HTTPException(404, "Prompt not found")
    return {"ok": True}
