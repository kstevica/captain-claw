"""The Line page's routes (line.py): the calling user's own Line only."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.line import KIND_LABELS, resolve_answered, view

router = APIRouter(prefix="/fd/line", tags=["line"])


async def _counts(user_id: str) -> dict:
    db = get_db()
    return {
        "unread": await db.count_unread_line_items(user_id),
        "waiting": len(await db.list_unresolved_line_items(user_id)),
    }


@router.get("")
async def list_line(
    limit: int = 50, before: str = "", kind: str = "", needs_you: bool | None = None,
    user: dict = Depends(get_current_user),
):
    """The newest items (before `before`, a created_at, to page), optionally
    one `kind` or only what needs the user; with the unread and still-waiting
    counts and the kinds the page names."""
    await resolve_answered(user["id"])
    rows = await get_db().list_line_items(
        user["id"], limit=limit, before=before, kinds=[kind] if kind else (), needs_you=needs_you)
    return {"items": [view(r) for r in rows], **(await _counts(user["id"])), "kinds": KIND_LABELS}


@router.get("/unread")
async def line_unread(user: dict = Depends(get_current_user)):
    """The Line's badge: how many are unread, and how many still wait."""
    await resolve_answered(user["id"])
    return await _counts(user["id"])


@router.post("/{item_id}/read")
async def mark_read(item_id: str, user: dict = Depends(get_current_user)):
    return {"ok": await get_db().mark_line_item_read(item_id, user["id"])}


@router.post("/read-all")
async def mark_all_read(user: dict = Depends(get_current_user)):
    return {"ok": True, "marked": await get_db().mark_all_line_items_read(user["id"])}
