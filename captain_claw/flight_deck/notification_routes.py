"""Persistent in-app notifications (the bell).

A durable counterpart to the in-memory toast store: a share you receive or a
web-origin run that finishes while your tab is closed lands here and is picked
up by the bell's poll. All routes are scoped to the calling user.
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
    """This user's notifications (newest first) + unread count. Proposals
    waiting for them that the bell doesn't hold yet are added first, and a
    waiting one older than the newest `limit` comes along after them: the
    bell answers it (and lists it on top)."""
    from captain_claw.flight_deck.owner_notify import bell_backfill, waiting_bells

    await bell_backfill(user["id"])
    db = get_db()
    items = await db.list_notifications(user["id"], limit=limit, unread_only=unread_only)
    items += await waiting_bells(user["id"], {n["id"] for n in items}, unread_only=unread_only)
    unread = await db.count_unread_notifications(user["id"])
    return {"notifications": items, "unread": unread}


@router.post("/{notif_id}/read")
async def mark_read(notif_id: str, user: dict = Depends(get_current_user)):
    db = get_db()
    ok = await db.mark_notification_read(notif_id, user["id"])
    return {"ok": ok}


@router.post("/read-all")
async def mark_all_read(user: dict = Depends(get_current_user)):
    db = get_db()
    n = await db.mark_all_notifications_read(user["id"])
    return {"ok": True, "marked": n}
