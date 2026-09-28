"""User settings REST endpoints for Flight Deck."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/settings", tags=["settings"])

PROVIDER_KEYS_SETTING = "fd:provider-keys"


# Server-owned keys that share the per-user store but are never the browser's
# to read or write. ``google_oauth:*`` is the user's Google refresh/access token
# and connected identity: only the OAuth routes write it (/fd/google/callback,
# a token refresh, /fd/google/logout). Readable here, any script running as the
# user would get the refresh token; writable, a user could forge a Google
# identity (e.g. to make another user's Disconnect skip the revoke) or clear
# their connection without the revoke.
_SERVER_OWNED_PREFIXES = ("google_oauth:",)


def _server_owned(key: str) -> bool:
    return str(key).strip().lower().startswith(_SERVER_OWNED_PREFIXES)


def _refuse_server_owned(keys) -> None:
    bad = sorted(k for k in keys if _server_owned(k))
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Setting(s) managed by Flight Deck, not writable here: {', '.join(bad)}",
        )


class SettingsUpdate(BaseModel):
    """Partial settings update — key-value pairs to merge."""
    settings: dict[str, str]


@router.get("")
async def get_settings(user: dict = Depends(get_current_user)):
    db = get_db()
    settings = await db.get_all_settings(user["id"])
    return {k: v for k, v in settings.items() if not _server_owned(k)}


@router.put("")
async def put_settings(body: SettingsUpdate, user: dict = Depends(get_current_user)):
    _refuse_server_owned(body.settings)
    db = get_db()
    await db.set_settings(user["id"], body.settings)
    return {"ok": True, "count": len(body.settings)}


@router.delete("/{key:path}")
async def delete_setting(key: str, user: dict = Depends(get_current_user)):
    _refuse_server_owned([key])
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


SHARED_TIER_SETS_SETTING = "fd:shared-tier-sets"


@router.get("/shared-tier-sets")
async def get_shared_tier_sets(user: dict = Depends(get_current_user)):
    """Team-default tier sets published by an admin. Available to every authed
    user; each tier's api_key is already the ``@system`` sentinel (no secrets
    leave the box), resolved server-side at run time from the org key store."""
    db = get_db()
    raw = await db.get_system_setting(SHARED_TIER_SETS_SETTING)
    if not raw:
        return {"sets": [], "defaultSetId": None}
    try:
        blob = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"sets": [], "defaultSetId": None}
    return blob if isinstance(blob, dict) else {"sets": [], "defaultSetId": None}
