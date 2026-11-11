"""Lessons to approve — what the owner's agents noticed, in one place.

An agent that reads a correction or a standing rule in its owner's message
("too long", "always CC Marko") drafts one lesson with its own model
(captain_claw/lesson_proposals.py) and hands it here. It waits beside what
members of a shared agent suggested (agent_onboarding), in the agent's
``lesson-suggestions.json`` — ``origin: auto``, drafted already, with the
owner's words it came from — and the bell tells the owner (one unread bell
per agent at a time).

The owner sees every agent's waiting lessons at once (kiosk "Lessons to
approve") and approves one as it is, edits it first, or dismisses it. An
approved lesson is taught like one added by hand (``taught_by: noticed``,
or ``member``) — it counts from the agent's next message — and stays listed
under "Just learned" for a while, where Undo takes it away again and puts
the proposal back. A proposal the agent has as a lesson already, one from a
chat message its owner taught a lesson from already, or one like a waiting
or dismissed one, is never kept; one waiting is settled when its owner
teaches the same from the chat (agent_onboarding ``said_key``).

    POST /fd/lesson-proposals/agent                 an agent proposes one (its own X-Agent-Auth)
    GET  /fd/lesson-proposals                       every agent of the caller's: waiting, and just learned
    POST /fd/lesson-proposals/{slug}/{id}/approve   teach it (kind, text, always: as edited, else as drafted)
    POST /fd/lesson-proposals/{slug}/{id}/dismiss   not a lesson
    POST /fd/lesson-proposals/{slug}/{id}/undo      take an approved one's lesson away; it waits again
"""

from __future__ import annotations

import asyncio
import secrets
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw import retrieval
from captain_claw.flight_deck import agent_lessons
from captain_claw.flight_deck import agent_onboarding as ob
from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck.agent_identity import require_agent_caller
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/lesson-proposals", tags=["lesson-proposals"])

WAITING_MAX = 30                 # an agent's own proposals waiting at once; more are dropped
LEARNED_DAYS = 14                # how long an approved one stays under "Just learned"
_CHANNELS = ("web", "whatsapp", "telegram", "device", "app", "voice", "slack", "discord", "")


def _name(slug: str) -> str:
    from captain_claw.flight_deck import server as srv
    return str((srv._load_process_registry().get(slug) or {}).get("name") or slug)


def _dirs(slug: str) -> tuple[Path, Path]:
    """An agent's folder and workspace, from the deck's records."""
    from captain_claw.flight_deck import server as srv
    entry = srv._load_process_registry().get(slug) or {}
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    return agent_dir, workspace


def _same(a: str, b: str) -> bool:
    return ob._flat(a) == ob._flat(b) or retrieval.same_fact(a, b)


def known(record: dict, items: list[dict], kind: str, text: str, said: str = "") -> str:
    """Why a proposal isn't kept — the agent has that lesson, or one was
    taught from the same chat message (*said*) already (``known``), one like
    it waits (``waiting``) or was dismissed (``dismissed``) — or ""."""
    if ob._taken(record, kind, text) or any(_same(text, x["text"]) for x in agent_lessons.from_record(record)):
        return "known"
    if said and ob.said_key(said) in ob.stamped(record):
        return "known"
    for s in items:
        if s.get("origin") != "auto" or not s.get("text") or not _same(text, str(s["text"])):
            continue
        if s.get("state") == "waiting":
            return "waiting"
        if s.get("state") == "declined":
            return "dismissed"
    return ""


class ProposeBody(BaseModel):
    kind: str = Field("rule", max_length=20)
    text: str = Field(..., max_length=2000)
    always: bool = False
    quote: str = Field("", max_length=1000)      # the owner's words it comes from
    said: str = Field("", max_length=4000)       # their whole message
    reply: str = Field("", max_length=8000)      # the agent's answer they wrote after
    channel: str = Field("", max_length=20)
    session_id: str = Field("", max_length=200)


async def _ring(owner: str, slug: str, text: str) -> None:
    """The bell for a new proposal — unless one about this agent is still unread."""
    from captain_claw.flight_deck.auth import get_db

    db = get_db()
    try:
        rows = await db.list_notifications_by_ref(owner, "agent_lessons", [slug])
        if any(not r["read"] and not r["resolved"] for r in rows):
            return
        await db.add_notification(owner, "info", f"{_name(slug)} noticed a lesson — approve it?"[:200], text[:300],
                                  ref_type="agent_lessons", ref_id=slug)
    except Exception:
        log.debug("lesson proposals: bell failed", exc_info=True)


async def _settle_bell(owner: str, slug: str, agent_dir: Path) -> None:
    """Nothing waits for this agent any more: its bells are answered."""
    if not owner or await asyncio.to_thread(ob._waiting, agent_dir):
        return
    from captain_claw.flight_deck.auth import get_db
    try:
        await get_db().resolve_line_items(owner, "agent_lessons", [slug])
    except Exception:
        log.debug("lesson proposals: resolving the bell failed", exc_info=True)


@router.post("/agent")
async def propose(body: ProposeBody, request: Request) -> dict:
    """An agent proposes a lesson it noticed in its owner's message. ``kept``
    false (with ``why``) when it isn't: known, waiting or dismissed already,
    or too many wait."""
    caller = require_agent_caller(request, what="lesson proposals")
    slug = caller.slug
    if not slug:
        raise HTTPException(403, "Flight Deck can't tell which agent this is.")
    kind = body.kind if body.kind in agent_lessons.KINDS else "rule"
    text = " ".join(body.text.split())[:ob._TEXT_MAX]
    if not text:
        raise HTTPException(400, "A proposal needs its lesson.")
    agent_dir, workspace = _dirs(slug)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    async with ob._SUGGESTION_LOCK:
        items = await asyncio.to_thread(ob._load_suggestions, agent_dir)
        why = known(record, items, kind, text, body.said)
        if not why and sum(1 for s in items if s.get("origin") == "auto" and s.get("state") == "waiting") >= WAITING_MAX:
            why = "full"
        if why:
            log.info("lesson proposals: one not kept", agent=slug, why=why)
            return {"ok": True, "kept": False, "why": why}
        item = {"id": secrets.token_hex(6), "origin": "auto", "kind": kind, "text": text,
                "always": bool(body.always) and kind != "fact", "quote": " ".join(body.quote.split())[:300],
                "said": body.said.strip()[:4000], "reply": body.reply.strip()[:6000],
                "channel": body.channel if body.channel in _CHANNELS else "", "session_id": body.session_id[:200],
                "created": time.time(), "state": "waiting"}
        await asyncio.to_thread(ob._write_suggestions, agent_dir, [item, *items])
    if caller.owner:
        await _ring(caller.owner, slug, text)
    log.info("lesson proposals: one waits for its owner", agent=slug, kind=kind)
    return {"ok": True, "kept": True, "id": item["id"]}


def _mine(user: dict) -> list[tuple[str, dict]]:
    """The caller's agents (every agent, on a deck without sign-in)."""
    from captain_claw.flight_deck import server as srv
    owner_id = ap._owner(user)
    return [(slug, e) for slug, e in srv._load_process_registry().items()
            if isinstance(e, dict) and (not owner_id or e.get("owner") == owner_id)]


def overview_for(agents: list[tuple[str, dict]], now: float | None = None) -> dict:
    """Every agent's waiting proposals and just-learned lessons — those with
    any, the most waiting first."""
    from captain_claw.flight_deck import server as srv
    now = time.time() if now is None else now
    out = []
    for slug, entry in agents:
        items = ob._load_suggestions(srv.DATA_DIR / slug)
        waiting = [ob.shown(s) for s in items if s.get("state") == "waiting"]
        learned = [ob.shown(s) for s in items if s.get("state") == "taken" and s.get("lesson")
                   and float(s.get("settled") or 0) >= now - LEARNED_DAYS * 86400]
        if waiting or learned:
            learned.sort(key=lambda s: -float(s.get("settled") or 0))
            out.append({"slug": slug, "name": str(entry.get("name") or slug), "waiting": waiting, "learned": learned})
    out.sort(key=lambda a: (-len(a["waiting"]), a["name"].casefold()))
    return {"agents": out, "waiting": sum(len(a["waiting"]) for a in out)}


@router.get("")
async def overview(user: dict = Depends(get_managed_user)) -> dict:
    """Lessons to approve: every agent of the caller's that has one waiting,
    or learned one lately."""
    return await asyncio.to_thread(overview_for, _mine(user))


async def _get(agent_dir: Path, pid: str, state: str) -> dict:
    items = await asyncio.to_thread(ob._load_suggestions, agent_dir)
    got = next((s for s in items if s["id"] == pid and s.get("state") == state), None)
    if got is None:
        raise HTTPException(404, "That proposal isn't waiting any more." if state == "waiting"
                            else "That lesson isn't among those just learned.")
    return got


async def _mark(agent_dir: Path, pid: str, **changes: object) -> dict:
    async with ob._SUGGESTION_LOCK:
        items = await asyncio.to_thread(ob._load_suggestions, agent_dir)
        got = next((s for s in items if s["id"] == pid), None)
        if got is None:
            raise HTTPException(404, "That proposal is gone.")
        got.update(changes)
        await asyncio.to_thread(ob._write_suggestions, agent_dir, items)
        return got


class ApproveBody(BaseModel):
    kind: str | None = Field(None, max_length=20)
    text: str | None = Field(None, max_length=2000)
    always: bool | None = None


@router.post("/{slug}/{pid}/approve")
async def approve(slug: str, pid: str, body: ApproveBody, request: Request,
                  user: dict = Depends(get_managed_user)) -> dict:
    """Teach it — as the owner edited it, else as drafted. One the agent
    has already is taken as that lesson. Returns the lessons' change (for
    the chat to send its instructions on) and the proposal."""
    owner_id, agent_dir, _ = ob._place(slug, user)
    s = await _get(agent_dir, pid, "waiting")
    kind = body.kind or str(s.get("kind") or "rule")
    text = " ".join(str(body.text if body.text is not None else s.get("text") or "").split())
    if not text:
        raise HTTPException(400, "Write the lesson first.")
    always = bool(body.always if body.always is not None else s.get("always")) and kind != "fact"
    taught_by = "noticed" if s.get("origin") == "auto" else "member"
    # The owner's chat message it was noticed in: the lesson keeps its key, so the chat doesn't offer it again.
    key = ob.said_key(str(s.get("said") or "")) if s.get("origin") == "auto" else ""
    changed: dict = {}
    try:
        changed = await ob.teach_one(slug, user, request, kind, text, always, taught_by, said_key=key,
                                     proposal=True)
    except HTTPException as exc:
        if exc.status_code != 409:
            raise
    lesson = agent_lessons.lesson_id(kind, ob._clean(text))
    got = await _mark(agent_dir, pid, state="taken", kind=kind, text=ob._clean(text), always=always,
                      lesson=lesson, settled=time.time())
    if got.get("by"):
        from captain_claw.flight_deck.auth import get_db
        try:
            await get_db().add_notification(str(got["by"]), "success", f"{_name(slug)} learned what you suggested"[:200],
                                            str(got.get("said") or "")[:300])
        except Exception:
            log.debug("lesson proposals: bell for a taken suggestion failed", exc_info=True)
    await _settle_bell(owner_id, slug, agent_dir)
    log.info("lesson proposals: approved", agent=slug, origin=got.get("origin") or "member")
    return {"changed": changed or None, "proposal": ob.shown(got)}


@router.post("/{slug}/{pid}/dismiss")
async def dismiss(slug: str, pid: str, user: dict = Depends(get_managed_user)) -> dict:
    """Not a lesson. A proposal like it isn't kept again."""
    owner_id, agent_dir, _ = ob._place(slug, user)
    await _get(agent_dir, pid, "waiting")
    got = await _mark(agent_dir, pid, state="declined", settled=time.time())
    await _settle_bell(owner_id, slug, agent_dir)
    return {"proposal": ob.shown(got)}


@router.post("/{slug}/{pid}/undo")
async def undo(slug: str, pid: str, request: Request, user: dict = Depends(get_managed_user)) -> dict:
    """An approved one was a mistake: its lesson goes, and it waits again."""
    _, agent_dir, _ = ob._place(slug, user)
    s = await _get(agent_dir, pid, "taken")
    changed: dict = {}
    try:
        changed = await ob.remove_one(slug, user, request, str(s.get("lesson") or ""))
    except HTTPException as exc:
        if exc.status_code != 404:   # gone already: nothing to take away
            raise
    got = await _mark(agent_dir, pid, state="waiting", lesson="", settled=0)
    return {"changed": changed or None, "proposal": ob.shown(got)}
