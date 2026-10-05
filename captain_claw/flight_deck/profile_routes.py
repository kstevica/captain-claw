"""Owner profile routes — what a user's agents know about them.

* ``GET``/``PUT /fd/profile`` — the signed-in user's own profile (about me,
  company, standing preferences), the deck defaults it merges with, the caps
  and a preview of what the agents receive. Plain `get_current_user`: an
  admin's ``X-FD-Act-As`` does not reach another user's profile.
* ``GET``/``PUT /fd/admin/profile-defaults`` — the deck-wide company and
  instructions (admin only; on an auth-off deck the local user is the admin).

A save rewrites the agents' context files (`tenant_profile.refresh_agents`) and
says how many agents it updated. Text that can't be encoded as UTF-8 (a lone
surrogate) is refused with 400, like an over-cap or non-string field. Storage,
merge and rendering live in :mod:`captain_claw.flight_deck.tenant_profile`.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(tags=["profile"])


def _validated(body: Any, fields: tuple[str, ...]) -> dict[str, str]:
    """The fields ``body`` sets (absent / null = unchanged), each a string of
    valid text within its cap — 400 otherwise."""
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected a JSON object")
    out: dict[str, str] = {}
    for field in fields:
        value = body.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise HTTPException(400, f"{field} must be a string")
        try:
            # JSON lets a lone surrogate (\ud800) through; it can't be written
            # to the agents' UTF-8 files, so it never gets stored either.
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise HTTPException(400, f"{field} contains characters that aren't valid text")
        value = value.strip()
        if len(value) > tp.CAPS[field]:
            raise HTTPException(
                400, f"{field} is too long ({len(value)} characters, at most {tp.CAPS[field]})")
        out[field] = value
    return out


async def _profile_payload(db, user_id: str) -> dict:
    full, compact = await tp.compose_for_owner(db, user_id)
    return {
        "profile": await tp.load_profile(db, user_id),
        "deck": await tp.load_deck(db),
        "caps": dict(tp.CAPS),
        "preview": {"full": full, "compact": compact},
    }


@router.get("/fd/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    return await _profile_payload(get_db(), str(user["id"]))


@router.put("/fd/profile")
async def put_profile(body: Any = Body(...), user: dict = Depends(get_current_user)):
    updates = _validated(body, tp.PROFILE_FIELDS)
    db = get_db()
    uid = str(user["id"])
    profile = await tp.load_profile(db, uid)
    profile.update(updates)
    await tp.save_profile(db, uid, profile)
    updated = await tp.refresh_agents(db, uid)
    return {**await _profile_payload(db, uid), "agents_updated": updated}


@router.get("/fd/admin/profile-defaults")
async def get_profile_defaults(admin: dict = Depends(require_admin)):
    deck = await tp.load_deck(get_db())
    return {**deck, "caps": {f: tp.CAPS[f] for f in tp.DECK_FIELDS}}


@router.put("/fd/admin/profile-defaults")
async def put_profile_defaults(body: Any = Body(...), admin: dict = Depends(require_admin)):
    updates = _validated(body, tp.DECK_FIELDS)
    db = get_db()
    deck = await tp.load_deck(db)
    deck.update(updates)
    await tp.save_deck(db, deck)
    updated = await tp.refresh_agents(db, None)
    try:
        await db.log_usage(str(admin.get("id") or ""), "profile_defaults_update", json.dumps({
            "fields": sorted(updates),
            "company_chars": len(deck["company"]),
            "instructions_chars": len(deck["instructions"]),
            "agents_updated": updated,
        }))
    except Exception:  # never fail the save over its own audit row
        pass
    return {"company": deck["company"], "instructions": deck["instructions"],
            "agents_updated": updated}
