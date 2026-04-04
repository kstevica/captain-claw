"""User settings REST endpoints for Flight Deck."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/settings", tags=["settings"])

PROVIDER_KEYS_SETTING = "fd:provider-keys"


class SettingsUpdate(BaseModel):
    """Partial settings update — key-value pairs to merge."""
    settings: dict[str, str]


@router.get("")
async def get_settings(user: dict = Depends(get_current_user)):
    db = get_db()
    return await db.get_all_settings(user["id"])


@router.put("")
async def put_settings(body: SettingsUpdate, user: dict = Depends(get_current_user)):
    db = get_db()
    await db.set_settings(user["id"], body.settings)
    return {"ok": True, "count": len(body.settings)}


@router.delete("/{key:path}")
async def delete_setting(key: str, user: dict = Depends(get_current_user)):
    db = get_db()
    deleted = await db.delete_setting(user["id"], key)
    if not deleted:
        return {"ok": False, "detail": "Setting not found"}
    return {"ok": True}


@router.get("/provider-keys")
async def get_system_provider_keys(user: dict = Depends(get_current_user)):
    """Get system-level provider API keys (set by admin). Available to all authenticated users."""
    db = get_db()
    raw = await db.get_system_setting(PROVIDER_KEYS_SETTING)
    if not raw:
        return {"keys": {}}
    try:
        return {"keys": json.loads(raw)}
    except (json.JSONDecodeError, TypeError):
        return {"keys": {}}
