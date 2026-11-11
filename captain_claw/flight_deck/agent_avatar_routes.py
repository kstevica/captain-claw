"""Pictures a person gives their agents (``/fd/agent-avatars``).

Which agent wears which picture lives in the person's own settings
(``fd:agent-avatars``: agent id → an emoji, or the id of a picture here), so it
follows them from device to device like their agents' names. The pictures
themselves are kept apart, one setting each under ``agent-avatar:<id>``: a
server-owned prefix, so ``GET /fd/settings`` — read on every load and on
coming back to a tab — never carries them, and ``PUT /fd/settings`` can't
write one without the checks below.

A picture is small and checked: PNG, JPEG or WebP only (never SVG, which can
carry script), at most ``MAX_BYTES``, its bytes matching its type. The browser
scales it down before it uploads. Its id is a hash of its bytes, so the same
picture on two agents is kept once, and each person reads only their own.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/agent-avatars", tags=["agent-avatars"])

SETTING_PREFIX = "agent-avatar:"
# The browser sends ~192px squares: a few KB as WebP. Room for a PNG of one.
MAX_BYTES = 256 * 1024

_DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,(.+)$", re.DOTALL)
_ID = re.compile(r"^[0-9a-f]{32}$")


def _looks_like(mime: str, raw: bytes) -> bool:
    if mime == "image/png":
        return raw.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/jpeg":
        return raw.startswith(b"\xff\xd8\xff")
    if mime == "image/webp":
        return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    return False


def _avatar_id(avatar_id: str) -> str:
    if not _ID.match(avatar_id or ""):
        raise HTTPException(status_code=404, detail="No such picture")
    return avatar_id


class AvatarUpload(BaseModel):
    data_url: str


@router.post("")
async def upload_avatar(body: AvatarUpload, user: dict = Depends(get_current_user)):
    # Twice the cap covers the base64 of the largest picture let in.
    if len(body.data_url) > MAX_BYTES * 2:
        raise HTTPException(status_code=413, detail="That picture is too large")
    m = _DATA_URL.match(body.data_url.strip())
    if not m:
        raise HTTPException(status_code=400, detail="A PNG, JPEG or WebP picture, please")
    mime, b64 = m.group(1), re.sub(r"\s+", "", m.group(2))
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="That picture could not be read")
    if not raw:
        raise HTTPException(status_code=400, detail="That picture is empty")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="That picture is too large")
    if not _looks_like(mime, raw):
        raise HTTPException(status_code=400, detail="That file is not the picture it says it is")
    avatar_id = hashlib.sha256(raw).hexdigest()[:32]
    await get_db().set_settings(user["id"], {
        SETTING_PREFIX + avatar_id: f"{mime};{base64.b64encode(raw).decode('ascii')}",
    })
    return {"id": avatar_id}


@router.get("/{avatar_id}")
async def get_avatar(avatar_id: str, user: dict = Depends(get_current_user)):
    stored = await get_db().get_setting(user["id"], SETTING_PREFIX + _avatar_id(avatar_id))
    if not stored or ";" not in stored:
        raise HTTPException(status_code=404, detail="No such picture")
    mime, b64 = stored.split(";", 1)
    try:
        raw = base64.b64decode(b64)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=404, detail="No such picture")
    return Response(
        content=raw,
        media_type=mime if mime in ("image/png", "image/jpeg", "image/webp") else "application/octet-stream",
        # Its id is its content: it never changes under that id.
        headers={"Cache-Control": "private, max-age=31536000, immutable", "X-Content-Type-Options": "nosniff"},
    )


@router.delete("/{avatar_id}")
async def delete_avatar(avatar_id: str, user: dict = Depends(get_current_user)):
    deleted = await get_db().delete_setting(user["id"], SETTING_PREFIX + _avatar_id(avatar_id))
    return {"ok": True, "deleted": bool(deleted)}
