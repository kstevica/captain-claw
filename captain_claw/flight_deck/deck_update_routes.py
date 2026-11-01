"""Console → Updates: admins only, but for the notice everyone gets. The engine and its order are in
``deck_update.py``.

``GET  /fd/system/update/notice`` anyone signed in: is an update coming, and how long until the restart
``GET  /fd/system/update``        where the deck is, what the last fetch brought, the update under way
``POST /fd/system/update/check``  ask the remote
``POST /fd/system/update/start``  ``{target, wait}``: update to that commit
``PUT  /fd/system/update/mode``   ``{mode}``: off · notify · install
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from captain_claw.flight_deck import deck_update
from captain_claw.flight_deck.admin_routes import require_admin
from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/system/update", tags=["system"])


class StartBody(BaseModel):
    target: str
    wait: bool = False


class ModeBody(BaseModel):
    mode: str


async def _status(*, fetch: bool = False) -> dict[str, Any]:
    return {**await deck_update.status(fetch=fetch), "mode": await deck_update.get_mode(get_db())}


@router.get("/notice")
async def update_notice(_: dict = Depends(get_current_user)) -> dict[str, Any]:
    return deck_update.notice()


@router.get("")
async def update_status(_: dict = Depends(require_admin)) -> dict[str, Any]:
    return await _status()


@router.post("/check")
async def update_check(_: dict = Depends(require_admin)) -> dict[str, Any]:
    return await _status(fetch=True)


@router.post("/start", status_code=202)
async def update_start(body: StartBody, user: dict = Depends(require_admin)) -> dict[str, Any]:
    try:
        return await deck_update.start(by=user.get("email") or user.get("display_name") or "an admin", target=body.target,
                                       wait=body.wait, user_id=user.get("id") or "")
    except deck_update.RefusedError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.put("/mode")
async def update_mode(body: ModeBody, _: dict = Depends(require_admin)) -> dict[str, str]:
    try:
        return {"mode": await deck_update.set_mode(get_db(), body.mode)}
    except deck_update.RefusedError as exc:
        raise HTTPException(400, str(exc)) from exc
