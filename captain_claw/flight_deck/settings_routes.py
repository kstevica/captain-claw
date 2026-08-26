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
    """Which providers have a system-level key configured (set by admin).

    Returns presence + a non-usable last-4 hint per provider — NEVER the raw
    secret. Any authenticated user may learn that, say, an Anthropic key exists
    so the Spawner can offer "use the system key" (which sends the ``@system``
    sentinel, resolved server-side at spawn time). Previously this returned the
    admin's plaintext keys to every logged-in user.
    """
    db = get_db()
    raw = await db.get_system_setting(PROVIDER_KEYS_SETTING)
    if not raw:
        return {"keys": {}}
    try:
        stored = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"keys": {}}
    masked: dict[str, dict] = {}
    if isinstance(stored, dict):
        for provider, key in stored.items():
            if not key:
                continue
            s = str(key)
            masked[provider] = {"configured": True, "hint": f"····{s[-4:]}" if len(s) >= 4 else "····"}
    return {"keys": masked}
