"""The Line's routes (line.py): the page's, for the calling user's own Line,
and the agents', for their owner's (the line tool, line_client)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.autonomy import _norm_user
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


# ── The agent's side: the line tool and the per-turn count (line_client) ──
#
# Owner only: an agent reads its owner's Line, never a member's (the agent
# sends no speaker headers here). Who is calling is this deck's own record of
# the X-Agent-Auth token it issued — never the agent's word.

_FIRST_LOOK = timedelta(days=1)
_NEWS_MAX = 100


async def _agent(request: Request) -> tuple[str, str, str]:
    """``(owner, slug, name)`` of the calling agent, or 401 / 403."""
    from captain_claw.flight_deck import google_oauth_routes as _g
    from captain_claw.flight_deck.auth import _fd_auth_enabled
    from captain_claw.flight_deck.line import agent_name
    from captain_claw.flight_deck.server import _find_agent_by_auth

    _g._authorize_agent_call(request)
    matched, owner, slug = _find_agent_by_auth(request.headers.get("X-Agent-Auth", ""))
    if not matched or not slug:
        raise HTTPException(status_code=403, detail="This Flight Deck doesn't know this agent")
    if _fd_auth_enabled():
        owner = await _g._agent_owner(request, service="Line")
    return _norm_user(owner), slug, agent_name(slug)


def _since(value: str) -> str:
    """created_at bound from ``since`` (an ISO time; "" = none)."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"since: not a time — {text!r}") from None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).isoformat()


@router.post("/agent/news")
async def agent_news(request: Request) -> dict:
    """What came on the Line since this agent was last told — counts by kind
    and what needs the user, no text; what it posted itself left out. The
    telling moves its ``told_at`` (the first time: the last day)."""
    owner, slug, name = await _agent(request)
    db = get_db()
    told = (await db.get_line_cursor(owner, slug))["told_at"]
    after = told or (datetime.now(timezone.utc) - _FIRST_LOOK).isoformat()
    rows = await db.list_line_items(owner, after=after, limit=500, oldest_first=True)
    if rows:
        await db.set_line_cursor(owner, slug, told_at=rows[-1]["created_at"])
    elif not told:
        await db.set_line_cursor(owner, slug, told_at=after)
    news = [r for r in rows if not (name and r.get("source") == name)]
    kinds: dict[str, int] = {}
    for r in news:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"connected": True, "new": len(news), "kinds": kinds, "first_look": not told,
            "needs_you": sum(1 for r in news if r["needs_you"] and not r["resolved_at"])}


@router.post("/agent/unread")
async def agent_unread(request: Request, body: dict = Body(default={})) -> dict:
    """The items this agent hasn't read yet, oldest first (``limit``, default
    20); reading them moves its ``read_at`` (the first time: the last day)."""
    owner, slug, _name = await _agent(request)
    db = get_db()
    seen = (await db.get_line_cursor(owner, slug))["read_at"]
    after = seen or (datetime.now(timezone.utc) - _FIRST_LOOK).isoformat()
    try:
        limit = max(1, min(int(body.get("limit") or 20), _NEWS_MAX))
    except (TypeError, ValueError):
        limit = 20
    rows = await db.list_line_items(owner, after=after, limit=limit + 1, oldest_first=True)
    more, rows = len(rows) > limit, rows[:limit]
    if rows:
        await db.set_line_cursor(owner, slug, read_at=rows[-1]["created_at"])
    elif not seen:
        await db.set_line_cursor(owner, slug, read_at=after)
    return {"items": [view(r) for r in rows], "more": more, "first_look": not seen}


@router.post("/agent/items")
async def agent_items(request: Request, body: dict = Body(default={})) -> dict:
    """The newest items, by ``kind``, ``needs_you``, ``query`` (every word in
    the title, text or source) and ``since``; ``limit`` (default 20, max 100).
    Moves no cursor."""
    owner, _slug, _name = await _agent(request)
    try:
        limit = max(1, min(int(body.get("limit") or 20), _NEWS_MAX))
    except (TypeError, ValueError):
        limit = 20
    kind = str(body.get("kind") or "").strip()
    needs = body.get("needs_you")
    rows = await get_db().list_line_items(
        owner, limit=limit, kinds=[kind] if kind else (), after=_since(str(body.get("since") or "")),
        needs_you=None if needs is None else bool(needs), words=str(body.get("query") or "").split()[:8],
    )
    return {"items": [view(r) for r in rows], "kinds": KIND_LABELS}


@router.post("/agent/item")
async def agent_item(request: Request, body: dict = Body(default={})) -> dict:
    """One item, whole."""
    owner, _slug, _name = await _agent(request)
    row = await get_db().get_line_item(owner, str(body.get("id") or "").strip())
    if row is None:
        raise HTTPException(status_code=404, detail="No such item on your user's Line")
    return {"item": view(row)}
