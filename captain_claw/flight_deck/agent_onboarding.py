"""An agent's lessons, seen and changed by its owner — the Onboarding page's
Lessons tab (kiosk OnboardingDialog).

What a setup read and what an exam taught (agent_lessons) is listed here,
each with where it came from and who taught it; the owner adds a lesson by
hand, rewords one, switches whether it is always in mind, or takes it away.
Every change writes the agent's record, its lessons (read from its next
message) and knowledge/owner-rules.md again, and — when what is always in
mind changed — its instructions' block, returned for the chat to send.

    GET    /fd/agent-lessons/{slug}         its lessons, and which always-in-mind ones its block has no room for
    POST   /fd/agent-lessons/{slug}         add one (kind, text, always)
    PATCH  /fd/agent-lessons/{slug}/{id}    reword it, or switch always
    DELETE /fd/agent-lessons/{slug}/{id}    take it away
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck import agent_lessons
from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-lessons", tags=["agent-lessons"])

_GROUP = {"rule": "rules", "ask_first": "escalation"}     # where a setup record keeps these kinds
_TEXT_MAX = 400


def _place(slug: str, user: dict) -> tuple[str, Path, Path]:
    """The caller's own agent: its owner, folder and workspace (404 for anyone else's)."""
    from captain_claw.flight_deck import server as srv
    owner_id = ap._owner(user)
    entry = ap._own_process(slug, owner_id)
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    return owner_id, agent_dir, workspace


def _flat(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def view(record: dict, block: str) -> dict:
    """The lessons as the page lists them, and the always-in-mind ones the
    block had no room for (they still reach the agent as lessons)."""
    lessons = agent_lessons.from_record(record)
    flat = _flat(block)
    return {"lessons": lessons, "counts": agent_lessons.counts(record),
            "not_in_block": [x["id"] for x in lessons if x["always"] and x["kind"] != "fact"
                             and _flat(x["text"]) not in flat]}


def find(record: dict, lesson_id: str) -> tuple[str, int] | None:
    """Where a lesson is in the record: its group and index."""
    for kind, group in _GROUP.items():
        for i, r in enumerate(record.get(group) or []):
            if isinstance(r, dict) and agent_lessons.lesson_id(kind, str(r.get("text") or "")) == lesson_id:
                return group, i
    for i, x in enumerate(record.get("lessons") or []):
        if isinstance(x, dict) and agent_lessons.lesson_id(str(x.get("kind") or ""), str(x.get("text") or "")) == lesson_id:
            return "lessons", i
    return None


def _clean(text: str) -> str:
    out = " ".join(str(text or "").split())[:_TEXT_MAX]
    if not out:
        raise HTTPException(400, "Say what the lesson is.")
    return out


def _taken(record: dict, kind: str, text: str, but: str = "") -> bool:
    new = agent_lessons.lesson_id(kind, text)
    return any(x["id"] == new and x["id"] != but for x in agent_lessons.from_record(record))


async def _save(slug: str, owner_id: str, agent_dir: Path, workspace: Path, record: dict) -> dict:
    blocks = await asyncio.to_thread(ap.rewrite_lessons, agent_dir, workspace, record)
    stored, instructions, block = await ap.instruct(slug, owner_id, blocks["block"], blocks["block_all"])
    return {**view(record, block), "stored": stored, "instructions": instructions,
            "block": block and f"{ap._BLOCK_OPEN}\n{block}\n{ap._BLOCK_CLOSE}"}


@router.get("/{slug}")
async def list_lessons(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    _, agent_dir, workspace = _place(slug, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    preset = ap.normalise({g: record.get(g) or [] for g in ap._KEPT_GROUPS}, ap.TOTALS)
    facts = sum(1 for x in agent_lessons.from_record(record) if x["kind"] == "fact")
    block = ap.rules_block(preset, record.get("knowledge") or [], record.get("skills") or [],
                           tables=record.get("tables") or [], facts=facts)
    return view(record, block)


class LessonBody(BaseModel):
    kind: str = Field("rule", max_length=20)
    text: str = Field(..., max_length=2000)
    always: bool = False


@router.post("/{slug}")
async def add_lesson(slug: str, body: LessonBody, user: dict = Depends(get_managed_user)) -> dict:
    """A lesson its owner gives by hand (``taught_by: you``)."""
    owner_id, agent_dir, workspace = _place(slug, user)
    if body.kind not in agent_lessons.KINDS:
        raise HTTPException(400, "A lesson is a rule, something to ask you first, or a fact.")
    text = _clean(body.text)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    if _taken(record, body.kind, text):
        raise HTTPException(409, "It has that lesson already.")
    if body.kind in _GROUP:
        item = {"text": text, "always": body.always, "source": {"doc": "instructions", "section": ""},
                "taught_by": "you", "added": time.time()}
        record = {**record, _GROUP[body.kind]: [item, *(record.get(_GROUP[body.kind]) or [])]}
    else:
        record, _ = agent_lessons.taught(record, "fact", text, taught_by="you")
    log.info("agent lessons: one added", agent=slug, kind=body.kind)
    return await _save(slug, owner_id, agent_dir, workspace, record)


class LessonPatch(BaseModel):
    text: str | None = Field(None, max_length=2000)
    always: bool | None = None


@router.patch("/{slug}/{lesson_id}")
async def change_lesson(slug: str, lesson_id: str, body: LessonPatch, user: dict = Depends(get_managed_user)) -> dict:
    """Reword a lesson, or switch whether it is always in mind (a rule or
    when-to-ask: in its instructions, for every conversation)."""
    owner_id, agent_dir, workspace = _place(slug, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    at = find(record, lesson_id)
    if at is None:
        raise HTTPException(404, "That lesson isn't there any more.")
    group, i = at
    items = [dict(x) for x in record.get(group) or []]
    item = items[i]
    kind = str(item.get("kind") or "") if group == "lessons" else ("rule" if group == "rules" else "ask_first")
    if body.text is not None:
        text = _clean(body.text)
        if _taken(record, kind, text, but=lesson_id):
            raise HTTPException(409, "It has that lesson already.")
        item["text"] = text
    if body.always is not None:
        if kind == "fact":
            raise HTTPException(400, "A fact reaches it when a message is about it; only rules are always in mind.")
        item["always"] = body.always
    items[i] = item
    return await _save(slug, owner_id, agent_dir, workspace, {**record, group: items})


@router.delete("/{slug}/{lesson_id}")
async def remove_lesson(slug: str, lesson_id: str, user: dict = Depends(get_managed_user)) -> dict:
    owner_id, agent_dir, workspace = _place(slug, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    at = find(record, lesson_id)
    if at is None:
        raise HTTPException(404, "That lesson isn't there any more.")
    group, i = at
    items = list(record.get(group) or [])
    del items[i]
    log.info("agent lessons: one taken away", agent=slug, group=group)
    return await _save(slug, owner_id, agent_dir, workspace, {**record, group: items})
