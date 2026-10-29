"""An agent's lessons, seen and changed by its owner — the Onboarding page's
Lessons tab (kiosk OnboardingDialog).

What a setup read and what an exam taught (agent_lessons) is listed here,
each with where it came from and who taught it; the owner adds a lesson by
hand, rewords one, switches whether it is always in mind, or takes it away.
Every change writes the agent's record, its lessons (read from its next
message) and knowledge/owner-rules.md again, and — when what is always in
mind changed — its instructions' block, returned for the chat to send.

What an admin taught it — setting it up for its user, or in the archetype it
was made from — is its base (``layer: base``): its user adds lessons of their
own, but can't reword, pin or take away the base's (``base_locked``). An
admin can.

On a shared agent a member can't teach it; they suggest a lesson from their
chat instead, which waits for its owner (a bell) to take — drafted and
checked as their own correction would be — or decline. Nothing is spent on
the owner's models until they draft it.

    GET    /fd/agent-lessons/{slug}         its lessons, which always-in-mind ones its block has no room for, and the
                                            members' suggestions waiting
    POST   /fd/agent-lessons/{slug}         add one (kind, text, always; taught in a chat, a practice run, from a
                                            member's suggestion)
    POST   /fd/agent-lessons/{slug}/draft   a lesson drafted from the owner's words (a correction in a chat,
                                            what a practice answer should have been), by the agent's model
    PATCH  /fd/agent-lessons/{slug}/{id}    reword it, or switch always
    DELETE /fd/agent-lessons/{slug}/{id}    take it away
    POST   /fd/agent-lessons/shared/suggest                   a member suggests one (agent_ref, said, reply)
    POST   /fd/agent-lessons/{slug}/suggestions/{id}          its owner took it, or declined it
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.flight_deck import agent_lessons
from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck.auth import get_current_user, get_managed_user
from captain_claw.flight_deck.file_library import reply_json
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-lessons", tags=["agent-lessons"])

_GROUP = {"rule": "rules", "ask_first": "escalation"}     # where a setup record keeps these kinds
_TEXT_MAX = 400
_TAUGHT_BY = ("you", "chat", "practice", "member")        # who teaches by hand: the page, a chat, a practice run,
                                                          # a member's suggestion its owner took
_DRAFT_PROMPT = Path(__file__).parent.parent / "instructions" / "agent_lesson_draft_prompt.md"
_DRAFT_TOKENS = 600
_SUGGESTIONS = "lesson-suggestions.json"
_SUGGESTIONS_KEPT = 50
_SUGGESTIONS_WAITING = 20                                 # waiting from one member at a time
_LOCKED = ("This one is part of its base — taught by an admin, or with the archetype it was made from — so it stays "
           "as it was set. Add a lesson of your own on top.")
_SUGGESTION_LOCK = asyncio.Lock()


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


def view(record: dict, block: str, base_locked: bool = False) -> dict:
    """The lessons as the page lists them, and the always-in-mind ones the
    block had no room for (they still reach the agent as lessons);
    ``base_locked``: the caller can't change the base's."""
    lessons = agent_lessons.from_record(record)
    flat = _flat(block)
    return {"lessons": lessons, "counts": agent_lessons.counts(record), "base_locked": base_locked,
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


def _teaching(request: Request, user: dict) -> tuple[str, bool]:
    """(layer, base locked) for the caller — nothing is locked on a deck without sign-in."""
    return agent_lessons.teaching(request, user) if ap._auth_enabled() else ("", False)


async def _save(slug: str, owner_id: str, agent_dir: Path, workspace: Path, record: dict,
                base_locked: bool = False) -> dict:
    blocks = await asyncio.to_thread(ap.rewrite_lessons, agent_dir, workspace, record)
    stored, instructions, block = await ap.instruct(slug, owner_id, blocks["block"], blocks["block_all"])
    return {**view(record, block, base_locked), "stored": stored, "instructions": instructions,
            "block": block and f"{ap._BLOCK_OPEN}\n{block}\n{ap._BLOCK_CLOSE}",
            "suggestions": await asyncio.to_thread(_waiting, agent_dir)}


@router.get("/{slug}")
async def list_lessons(slug: str, request: Request, user: dict = Depends(get_managed_user)) -> dict:
    _, agent_dir, workspace = _place(slug, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    preset = ap.normalise({g: record.get(g) or [] for g in ap._KEPT_GROUPS}, ap.TOTALS)
    facts = sum(1 for x in agent_lessons.from_record(record) if x["kind"] == "fact")
    block = ap.rules_block(preset, record.get("knowledge") or [], record.get("skills") or [],
                           tables=record.get("tables") or [], facts=facts)
    return {**view(record, block, _teaching(request, user)[1]),
            "suggestions": await asyncio.to_thread(_waiting, agent_dir)}


class LessonBody(BaseModel):
    kind: str = Field("rule", max_length=20)
    text: str = Field(..., max_length=2000)
    always: bool = False
    taught_by: str = Field("you", max_length=20)


@router.post("/{slug}")
async def add_lesson(slug: str, body: LessonBody, request: Request, user: dict = Depends(get_managed_user)) -> dict:
    """A lesson its owner gives by hand — on this page (``taught_by: you``),
    from a chat, after a practice answer, or from a member's suggestion. An
    admin giving it for its user gives the base."""
    owner_id, agent_dir, workspace = _place(slug, user)
    layer, locked = _teaching(request, user)
    if body.kind not in agent_lessons.KINDS:
        raise HTTPException(400, "A lesson is a rule, something to ask you first, or a fact.")
    taught_by = body.taught_by if body.taught_by in _TAUGHT_BY else "you"
    text = _clean(body.text)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    if _taken(record, body.kind, text):
        raise HTTPException(409, "It has that lesson already.")
    if body.kind in _GROUP:
        item = {"text": text, "always": body.always, "source": {"doc": "instructions", "section": ""},
                "taught_by": taught_by, "added": time.time(), **({"layer": layer} if layer else {})}
        record = {**record, _GROUP[body.kind]: [item, *(record.get(_GROUP[body.kind]) or [])]}
    else:
        record, _ = agent_lessons.taught(record, "fact", text, taught_by=taught_by, layer=layer)
    log.info("agent lessons: one added", agent=slug, kind=body.kind, layer=layer or "own")
    return await _save(slug, owner_id, agent_dir, workspace, record, locked)


class LessonPatch(BaseModel):
    text: str | None = Field(None, max_length=2000)
    always: bool | None = None


@router.patch("/{slug}/{lesson_id}")
async def change_lesson(slug: str, lesson_id: str, body: LessonPatch, request: Request,
                        user: dict = Depends(get_managed_user)) -> dict:
    """Reword a lesson, or switch whether it is always in mind (a rule or
    when-to-ask: in its instructions, for every conversation). The base's
    only by an admin."""
    owner_id, agent_dir, workspace = _place(slug, user)
    _, locked = _teaching(request, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    at = find(record, lesson_id)
    if at is None:
        raise HTTPException(404, "That lesson isn't there any more.")
    group, i = at
    items = [dict(x) for x in record.get(group) or []]
    item = items[i]
    if locked and item.get("layer") == "base":
        raise HTTPException(403, _LOCKED)
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
    return await _save(slug, owner_id, agent_dir, workspace, {**record, group: items}, locked)


@router.delete("/{slug}/{lesson_id}")
async def remove_lesson(slug: str, lesson_id: str, request: Request, user: dict = Depends(get_managed_user)) -> dict:
    owner_id, agent_dir, workspace = _place(slug, user)
    _, locked = _teaching(request, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    at = find(record, lesson_id)
    if at is None:
        raise HTTPException(404, "That lesson isn't there any more.")
    group, i = at
    items = list(record.get(group) or [])
    if locked and isinstance(items[i], dict) and items[i].get("layer") == "base":
        raise HTTPException(403, _LOCKED)
    del items[i]
    log.info("agent lessons: one taken away", agent=slug, group=group)
    return await _save(slug, owner_id, agent_dir, workspace, {**record, group: items}, locked)


class DraftBody(BaseModel):
    said: str = Field(..., max_length=4000)      # the owner's words
    reply: str = Field("", max_length=8000)      # what the agent had answered
    task: str = Field("", max_length=2000)       # the practice task it answered


@router.post("/{slug}/draft")
async def draft_lesson(slug: str, body: DraftBody, user: dict = Depends(get_managed_user)) -> dict:
    """One lesson drafted from the owner's words, for them to check before
    it's taught: by the agent's own model (thinking off, as reads), with the
    answer it corrects. Without a model — or one that can't say — their words
    as a rule (``drafted`` false); ``text`` empty when they teach nothing."""
    from captain_claw.flight_deck.agent_exam import ExamJob

    owner_id, _, _ = _place(slug, user)
    said = " ".join(body.said.split())
    if not said:
        raise HTTPException(400, "Say what it should learn.")
    plain = {"kind": "rule", "text": said[:_TEXT_MAX], "always": False, "drafted": False}
    if await ap._quota_spent(owner_id):
        return plain
    reader, _ = await ap.choose_reader(owner_id, slug)
    if not reader:
        return plain
    job = ExamJob(id="draft", agent=slug, owner_id=owner_id, owner=ap.owner_key(owner_id))
    prompt = "\n\n".join(x for x in (
        body.task.strip() and f"## The task it was given\n{body.task.strip()}",
        body.reply.strip() and f"## What the agent answered\n{body.reply.strip()[:6000]}",
        f"## What the owner said\n{said}") if x)
    try:
        got = reply_json((await ap._call(reader, _DRAFT_PROMPT.read_text(encoding="utf-8"), prompt, owner_id, job,
                                         _DRAFT_TOKENS)).text)
    except Exception as exc:
        log.info("agent lessons: no draft", agent=slug, error=str(exc)[:200])
        return plain
    kind = str(got.get("kind") or "").strip().lower()
    kind = kind if kind in agent_lessons.KINDS else "rule"
    text = " ".join(str(got.get("text") or "").split())[:_TEXT_MAX]
    return {"kind": kind, "text": text, "always": got.get("always") is True and kind != "fact" and bool(text),
            "drafted": True}


# ── Members' suggestions ─────────────────────────────────────────────


def _load_suggestions(agent_dir: Path) -> list[dict]:
    try:
        data = json.loads((agent_dir / _SUGGESTIONS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [s for s in data if isinstance(s, dict) and s.get("id")] if isinstance(data, list) else []


def _write_suggestions(agent_dir: Path, items: list[dict]) -> None:
    # The waiting ones are all kept; of the settled, the newest.
    waiting = [s for s in items if s.get("state") == "waiting"]
    settled = [s for s in items if s.get("state") != "waiting"][:max(0, _SUGGESTIONS_KEPT - len(waiting))]
    agent_dir.mkdir(parents=True, exist_ok=True)
    tmp = agent_dir / f".{_SUGGESTIONS}.tmp"
    tmp.write_text(json.dumps(waiting + settled, ensure_ascii=False), encoding="utf-8")
    tmp.replace(agent_dir / _SUGGESTIONS)


def _waiting(agent_dir: Path) -> list[dict]:
    """What members suggested that its owner hasn't taken or declined, newest first."""
    return [{k: s.get(k) for k in ("id", "said", "reply", "by_name", "created")}
            for s in _load_suggestions(agent_dir) if s.get("state") == "waiting"]


class SuggestBody(BaseModel):
    agent_ref: str = Field(..., max_length=300)
    said: str = Field(..., max_length=4000)      # the member's words
    reply: str = Field("", max_length=8000)      # what the agent had answered them


@router.post("/shared/suggest")
async def suggest_lesson(body: SuggestBody, user: dict = Depends(get_current_user)) -> dict:
    """A member of someone's shared agent suggests a lesson from their chat
    with it — their words, and the answer they correct — for its owner to
    take or decline (a bell tells them). Only its members; nothing is drafted
    until its owner looks."""
    from captain_claw.flight_deck import agent_sharing as sharing
    from captain_claw.flight_deck import server as srv
    from captain_claw.flight_deck import tenant_profile
    from captain_claw.flight_deck.auth import get_db

    uid = str(user.get("id") or "")
    db = get_db()
    rec = sharing.resolve_agent_record(body.agent_ref) if sharing.sharing_active() else None
    if (rec is None or rec.runtime != "process" or rec.owner == uid
            or not await sharing.member_check(db, rec.ref, rec.owner, uid, max_age=0)):
        raise HTTPException(404, "That agent isn't shared with you.")
    said = " ".join(body.said.split())
    if not said:
        raise HTTPException(400, "Say what it should learn.")
    name = await tenant_profile.owner_name(db, uid) or "A member"
    agent_dir = srv.DATA_DIR / rec.slug
    async with _SUGGESTION_LOCK:
        items = await asyncio.to_thread(_load_suggestions, agent_dir)
        waiting = [s for s in items if s.get("state") == "waiting"]
        if any(_flat(s.get("said") or "") == _flat(said) for s in waiting):
            return {"ok": True, "already": True}
        if sum(1 for s in waiting if s.get("by") == uid) >= _SUGGESTIONS_WAITING:
            raise HTTPException(429, f"You have {_SUGGESTIONS_WAITING} suggestions waiting for its owner already.")
        item = {"id": secrets.token_hex(6), "said": said, "reply": body.reply.strip()[:6000], "by": uid,
                "by_name": name, "created": time.time(), "state": "waiting"}
        await asyncio.to_thread(_write_suggestions, agent_dir, [item, *items])
    try:
        await db.add_notification(rec.owner, "info", f"{name} suggested a lesson for {rec.name}"[:200], said[:300],
                                  ref_type="agent_lessons", ref_id=rec.slug)
    except Exception:
        log.debug("agent lessons: bell for a suggestion failed", exc_info=True)
    log.info("agent lessons: a member suggested one", agent=rec.slug)
    return {"ok": True, "already": False}


class SettleBody(BaseModel):
    state: str = Field(..., max_length=10)       # taken | declined
    lesson: str = Field("", max_length=40)       # the lesson taught from it


@router.post("/{slug}/suggestions/{suggestion_id}")
async def settle_suggestion(slug: str, suggestion_id: str, body: SettleBody,
                            user: dict = Depends(get_managed_user)) -> dict:
    """Its owner took a member's suggestion (taught as a lesson of theirs,
    ``taught_by: member``) or declined it; the member hears which it was."""
    _, agent_dir, _ = _place(slug, user)
    if body.state not in ("taken", "declined"):
        raise HTTPException(400, "A suggestion is taken, or declined.")
    async with _SUGGESTION_LOCK:
        items = await asyncio.to_thread(_load_suggestions, agent_dir)
        got = next((s for s in items if s["id"] == suggestion_id and s.get("state") == "waiting"), None)
        if got is None:
            raise HTTPException(404, "That suggestion isn't waiting any more.")
        got.update(state=body.state, lesson=body.lesson, settled=time.time())
        await asyncio.to_thread(_write_suggestions, agent_dir, items)
    if body.state == "taken" and got.get("by"):
        from captain_claw.flight_deck import server as srv
        from captain_claw.flight_deck.auth import get_db
        name = str((srv._load_process_registry().get(slug) or {}).get("name") or slug)
        try:
            await get_db().add_notification(str(got["by"]), "success", f"{name} learned what you suggested"[:200],
                                            str(got.get("said") or "")[:300])
        except Exception:
            log.debug("agent lessons: bell for a taken suggestion failed", exc_info=True)
    return {"suggestions": await asyncio.to_thread(_waiting, agent_dir)}
