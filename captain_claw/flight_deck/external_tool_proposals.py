"""Custom tools an agent proposes, waiting for its owner (Phase 5 of
docs/external-tools-plan.md; the agent side is captain_claw/tool_proposals.py).

An agent drafts a tool when a job looked like code's (or when asked to, with
``custom_tools action=propose``). Here the draft waits on its owner's Custom
tools page. Building it makes the tool's repository and starts a Crew task as
that agent, so the agent hears when it's done; nothing is built without the
owner's click. One unread bell per owner, settled once nothing waits.

Stored per owner beside their tools: ``external_tools/<owner>/.proposals.json``.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.flight_deck.auth import get_db
from captain_claw.logging import get_logger

log = get_logger(__name__)

# Fixed paths: included in server.py BEFORE external_tools_routes.router.
router = APIRouter(prefix="/fd/external-tools", tags=["external-tools"])
agent_router = APIRouter(prefix="/fd/external-tools/agent", tags=["external-tools"])

WAITING_MAX = 20
KEPT = 60
BELL_REF = "custom_tool_proposal"
_LOCK = asyncio.Lock()


def _file(owner: str):
    return xt.root() / xt._check_owner(owner) / ".proposals.json"


def load(owner: str) -> list[dict[str, Any]]:
    data = xt._read_json(_file(owner))
    return [p for p in data if isinstance(p, dict)] if isinstance(data, list) else []


def _save(owner: str, items: list[dict[str, Any]]) -> None:
    waiting = [p for p in items if p.get("state") == "waiting"]
    settled = sorted((p for p in items if p.get("state") != "waiting"), key=lambda p: p.get("settled") or 0)
    xt._write_json(_file(owner), waiting + settled[-max(0, KEPT - len(waiting)):])


def _same(a: str, b: str) -> bool:
    from captain_claw.retrieval import same_fact

    return " ".join(a.lower().split()) == " ".join(b.lower().split()) or same_fact(a, b)


def known(owner: str, items: list[dict[str, Any]], name: str, what: str) -> str:
    """Why a draft isn't kept ("" when it is new)."""
    try:
        if xt.exists(owner, name):
            return "exists"
    except xt.ToolError:
        return "bad_name"
    for p in items:
        if p.get("name") == name or _same(what, str(p.get("what") or "")):
            if p.get("state") == "waiting":
                return "waiting"
            if p.get("state") == "dismissed":
                return "dismissed"
    if sum(1 for p in items if p.get("state") == "waiting") >= WAITING_MAX:
        return "full"
    return ""


def shown(p: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "agent", "name", "title", "what", "why", "said", "source", "created", "state", "settled", "task_id", "tool")
    return {k: p.get(k) for k in keys if k in p}


async def _ring(owner: str, title: str, body: str) -> None:
    db = get_db()
    try:
        rows = await db.list_notifications_by_ref(owner, BELL_REF, [owner])
        if any(not r.get("read") and not r.get("resolved") for r in rows):
            return
        await db.add_notification(owner, "custom_tool", title[:200], body[:300], ref_type=BELL_REF, ref_id=owner)
    except Exception as exc:  # noqa: BLE001 — a bell must not lose the proposal
        log.warning("Couldn't ring for a tool proposal", error=str(exc))


async def _settle_bell(owner: str, items: list[dict[str, Any]]) -> None:
    if any(p.get("state") == "waiting" for p in items):
        return
    try:
        await get_db().resolve_line_items(owner, BELL_REF, [owner])
    except Exception:
        pass


async def _agent_name(slug: str) -> str:
    try:
        from captain_claw.flight_deck import lesson_proposals

        return lesson_proposals._name(slug)
    except Exception:
        return slug or "Your agent"


# ── Agents ───────────────────────────────────────────────────────────────


class ProposeBody(BaseModel):
    name: str = Field(..., max_length=60)
    title: str = Field("", max_length=200)
    what: str = Field(..., max_length=4000)
    why: str = Field("", max_length=800)
    said: str = Field("", max_length=4000)
    channel: str = Field("", max_length=20)
    source: str = Field("noticed", max_length=20)


@agent_router.post("/propose")
async def propose(body: ProposeBody, request: Request) -> dict[str, Any]:
    """An agent's draft for its owner. Owners only: a shared agent's member
    doesn't propose tools for the agent's owner."""
    runner = await xr._agent_enabled(request)
    if runner.member:
        raise HTTPException(403, "Only the agent's owner gets tool proposals.")
    if not await xr._can_build({"id": runner.user_id}):
        raise HTTPException(403, xr.NEEDS_CREW)
    name = body.name.strip().lower()
    try:
        name = xt.check_slug(name)
    except xt.ToolError as exc:
        raise HTTPException(400, exc.message) from None
    if name in xt.RESERVED_SLUGS:
        raise HTTPException(400, f"“{name}” is a name Flight Deck uses itself: pick another.")
    what = " ".join(body.what.split())
    if len(what) < 20:
        raise HTTPException(422, "Say what the tool should do: its input, what it returns, an example.")
    owner = runner.user_id
    async with _LOCK:
        items = load(owner)
        why_not = known(owner, items, name, what)
        if why_not:
            return {"ok": True, "kept": False, "why": why_not}
        item = {"id": secrets.token_hex(6), "agent": runner.agent, "name": name,
                "title": " ".join(body.title.split())[:80] or name, "what": what[:2000],
                "why": " ".join(body.why.split())[:400], "said": body.said[:3000], "channel": body.channel,
                "source": body.source if body.source in ("noticed", "asked") else "noticed",
                "created": time.time(), "state": "waiting"}
        items.append(item)
        _save(owner, items)
    who = await _agent_name(runner.agent)
    await _ring(owner, f"{who} suggests a custom tool: {item['title']}",
                f"{item['why'] or item['what'][:200]} Build it or dismiss it under Custom tools.")
    return {"ok": True, "kept": True, "id": item["id"]}


# ── The owner ────────────────────────────────────────────────────────────


class BuildProposalBody(BaseModel):
    harness: str = Field(..., max_length=40)
    name: str = Field("", max_length=48)
    title: str = Field("", max_length=200)
    what: str = Field("", max_length=8000)
    model: str = Field("", max_length=200)


async def _find(owner: str, pid: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = load(owner)
    item = next((p for p in items if p.get("id") == pid), None)
    if item is None:
        raise HTTPException(404, "There's no such proposal.")
    return items, item


@router.get("/proposals")
async def proposals(user: dict = Depends(xr._enabled)) -> dict[str, Any]:
    owner = xr._uid(user)
    items = load(owner)
    out = []
    for p in items:
        out.append({**shown(p), "agent_name": await _agent_name(str(p.get("agent") or ""))})
    waiting = [p for p in out if p.get("state") == "waiting"]
    recent = sorted((p for p in out if p.get("state") != "waiting"), key=lambda p: p.get("settled") or 0, reverse=True)[:10]
    return {"waiting": sorted(waiting, key=lambda p: p.get("created") or 0, reverse=True), "recent": recent}


@router.post("/proposals/{pid}/build")
async def build_proposal(pid: str, body: BuildProposalBody, user: dict = Depends(xr._enabled)) -> dict[str, Any]:
    """Build it (as drafted, or with the owner's edits): the tool's repository and a
    Crew task as the agent that proposed it, which hears when it's done."""
    from captain_claw.flight_deck import crew_routes, crew_runner

    await xr._require_builder(user)
    owner = xr._uid(user)
    async with _LOCK:
        items, item = await _find(owner, pid)
        if item.get("state") != "waiting":
            raise HTTPException(409, "It's already been decided.")
        name = (body.name or item["name"]).strip().lower()
        title = " ".join((body.title or item.get("title") or name).split())[:80]
        what = (body.what or item["what"]).strip()
        slug, made = await xr._build_target(owner, name, new=True, title=title, what=what)
        try:
            row, grant = await crew_routes._approved(user)
            task = await crew_runner.dispatch(row, grant, {
                "harness": body.harness, "brief": xr.build_brief(slug, title or slug, what), "title": f"Build custom tool {slug}",
                "kind": "ship", "deliver": "branch", "tool": slug, "model": body.model,
                "agent_slug": item.get("agent") or "", "source_host": "localhost",
            })
        except BaseException:
            await xr._drop_if_made(owner, slug, made)
            raise
        item.update(state="built", settled=time.time(), task_id=task.get("id"), tool=slug)
        _save(owner, items)
        await _settle_bell(owner, items)
    return {"proposal": shown(item), "tool": xt.summary(owner, slug), "task": xr._task_out(crew_runner.view(task, full=True))}


@router.post("/proposals/{pid}/dismiss")
async def dismiss_proposal(pid: str, user: dict = Depends(xr._enabled)) -> dict[str, Any]:
    owner = xr._uid(user)
    async with _LOCK:
        items, item = await _find(owner, pid)
        if item.get("state") != "waiting":
            raise HTTPException(409, "It's already been decided.")
        item.update(state="dismissed", settled=time.time())
        _save(owner, items)
        await _settle_bell(owner, items)
    return {"proposal": shown(item)}
