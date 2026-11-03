"""Persistent in-app notifications (the bell).

The bell is the Line's Needs-you list (line.py): what waits for the user or
failed, picked up by the bell's poll. Everything else the deck says on its
own is on the Line page only. All routes are scoped to the calling user.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from captain_claw.flight_deck.auth import get_current_user, get_db

router = APIRouter(prefix="/fd/notifications", tags=["notifications"])


@router.get("")
async def list_notifications(
    limit: int = 50, unread_only: bool = False,
    user: dict = Depends(get_current_user),
):
    """This user's items that need them (newest first) + their unread count.
    Every one still waiting comes along, however old: the bell lists it on
    top. Proposals waiting that the Line doesn't hold yet are added first."""
    from captain_claw.flight_deck.line import resolve_answered
    from captain_claw.flight_deck.owner_notify import bell_backfill

    await bell_backfill(user["id"])
    await resolve_answered(user["id"])
    db = get_db()
    items = await db.list_notifications(user["id"], limit=limit, unread_only=unread_only, needs_you=True)
    unread = await db.count_unread_notifications(user["id"], needs_you=True)
    return {"notifications": items, "unread": unread}


@router.post("/{notif_id}/read")
async def mark_read(notif_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    ok = await db.mark_notification_read(notif_id, user["id"])
    return {"ok": ok}


@router.post("/read-all")
async def mark_all_read(user: dict = Depends(get_current_user)):
    db = get_db()
    n = await db.mark_all_notifications_read(user["id"], needs_you=True)
    return {"ok": True, "marked": n}
