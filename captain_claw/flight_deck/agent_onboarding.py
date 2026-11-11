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
    GET    /fd/agent-lessons/{slug}/from-chat  which of the owner's chat messages are lessons already, wait in
                                            Lessons to approve, or were dismissed there (by said_key)
    POST   /fd/agent-lessons/shared/suggest                   a member suggests one (agent_ref, said, reply)
    GET    /fd/agent-lessons/shared/suggested?agent_ref=…     what the caller suggested, by said_key
    POST   /fd/agent-lessons/{slug}/suggestions/{id}          its owner took it, or declined it

A chat offers "Save as a lesson?" under a message that reads like one. Once
it is a lesson, it doesn't ask again: the lesson keeps the message's
``said_key`` (``said_keys`` on its record item — a short hash of the words,
not the words), and the chat reads them back (``from-chat``). A lesson the
agent noticed in that message counts too, while it waits, once approved, or
when it was dismissed. The key goes with the lesson when it's reworded, and
with it when it's taken away, so the chat offers it again.
"""

from __future__ import annotations

import asyncio
import json
import re
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
_TAUGHT_BY = ("you", "chat", "practice", "member", "noticed")  # who teaches by hand: the page, a chat, a practice
                                                          # run, a member's suggestion its owner took, a lesson
                                                          # the agent noticed in a chat (lesson_proposals)
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


_SAID_KEY = re.compile(r"[0-9a-f]{9,16}")
_SAID_KEYS_MAX = 20                                       # chat messages one lesson remembers coming from


def said_key(text: str) -> str:
    """A chat message's key: FNV-1a (32 bits) of its words, whitespace
    collapsed, as UTF-8, then their length — the browser works out the same
    (flight-deck services/agentLessons.ts ``saidKey``). "" for no words."""
    data = " ".join(str(text or "").split()).encode("utf-8")
    if not data:
        return ""
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}{len(data):x}"


def clean_key(key: str) -> str:
    return key if _SAID_KEY.fullmatch(key or "") else ""


def stamped(record: dict) -> set[str]:
    """The keys of every chat message a lesson of the record was taught from."""
    out: set[str] = set()
    for group in (*_GROUP.values(), "lessons"):
        for x in record.get(group) or []:
            if isinstance(x, dict):
                out.update(k for k in x.get("said_keys") or [] if isinstance(k, str))
    return out


def _stamp(record: dict, lesson_id: str, key: str) -> dict:
    """*record* with *key* — the chat message it was taught from — on the lesson."""
    at = find(record, lesson_id) if key else None
    if at is None:
        return record
    group, i = at
    items = list(record.get(group) or [])
    item = dict(items[i])
    item["said_keys"] = [key, *(k for k in item.get("said_keys") or [] if k != key)][:_SAID_KEYS_MAX]
    items[i] = item
    return {**record, group: items}


_MARK_RANK = {"dismissed": 1, "waiting": 2, "taught": 3}


def chat_marks(record: dict, items: list[dict], *, by: str = "") -> dict[str, str]:
    """``{said_key: taught | waiting | dismissed}`` — the chat messages not
    to offer as a lesson again: one a lesson was taught from, one the agent
    noticed a lesson in (waiting, approved while its lesson lasts, or
    dismissed). With *by*: that member's own suggestions instead."""
    have = {x["id"] for x in agent_lessons.from_record(record)}
    marks: dict[str, str] = {}
    for s in items:
        if (s.get("by") != by) if by else (s.get("origin") != "auto"):
            continue
        key = clean_key(str(s.get("said_key") or "")) or said_key(str(s.get("said") or ""))
        state = s.get("state")
        mark = ("waiting" if state == "waiting" else "dismissed" if state == "declined"
                else "taught" if state == "taken" and (by or s.get("lesson") in have) else "")
        if key and mark and _MARK_RANK[mark] > _MARK_RANK.get(marks.get(key, ""), 0):
            marks[key] = mark
    if not by:
        marks.update(dict.fromkeys(stamped(record), "taught"))
    return marks


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


@router.get("/{slug}/from-chat")
async def from_chat(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    """The owner's chat messages not to offer as a lesson again, by
    ``said_key``: taught, waiting in Lessons to approve, or dismissed there."""
    _, agent_dir, workspace = _place(slug, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    items = await asyncio.to_thread(_load_suggestions, agent_dir)
    return {"marks": chat_marks(record, items)}


class LessonBody(BaseModel):
    kind: str = Field("rule", max_length=20)
    text: str = Field(..., max_length=2000)
    always: bool = False
    taught_by: str = Field("you", max_length=20)
    said_key: str = Field("", max_length=20)     # the chat message it's taught from (said_key)


@router.post("/{slug}")
async def add_lesson(slug: str, body: LessonBody, request: Request, user: dict = Depends(get_managed_user)) -> dict:
    """A lesson its owner gives by hand — on this page (``taught_by: you``),
    from a chat, after a practice answer, or from a member's suggestion. An
    admin giving it for its user gives the base."""
    return await teach_one(slug, user, request, body.kind, body.text, body.always, body.taught_by,
                           said_key=body.said_key)


async def teach_one(slug: str, user: dict, request: Request, kind: str, text: str, always: bool,
                    taught_by: str, *, said_key: str = "", proposal: bool = False) -> dict:
    """Teach the caller's agent one lesson (409 when it has it already) —
    by hand, or approved from Lessons to approve (lesson_proposals:
    *proposal*).

    From a chat message (*said_key*), the lesson keeps that message's key,
    so the chat doesn't offer it again; one it has already takes the key
    (``already``) rather than a 409. What waits in Lessons to approve that
    says what this lesson says is settled as taught — and, saved from the
    chat, whatever the agent noticed in that same message."""
    owner_id, agent_dir, workspace = _place(slug, user)
    layer, locked = _teaching(request, user)
    if kind not in agent_lessons.KINDS:
        raise HTTPException(400, "A lesson is a rule, something to ask you first, or a fact.")
    taught_by = taught_by if taught_by in _TAUGHT_BY else "you"
    text = _clean(text)
    key = clean_key(said_key)
    lesson = agent_lessons.lesson_id(kind, text)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    if _taken(record, kind, text):
        if not key:
            raise HTTPException(409, "It has that lesson already.")
        out = await _save(slug, owner_id, agent_dir, workspace, _stamp(record, lesson, key), locked)
        await _settle_taught(owner_id, slug, agent_dir, kind, text, lesson, "" if proposal else key)
        return {**out, "already": True}
    if kind in _GROUP:
        item = {"text": text, "always": always, "source": {"doc": "instructions", "section": ""},
                "taught_by": taught_by, "added": time.time(), **({"layer": layer} if layer else {})}
        record = {**record, _GROUP[kind]: [item, *(record.get(_GROUP[kind]) or [])]}
    else:
        record, _ = agent_lessons.taught(record, "fact", text, taught_by=taught_by, layer=layer)
    log.info("agent lessons: one added", agent=slug, kind=kind, layer=layer or "own", taught_by=taught_by)
    out = await _save(slug, owner_id, agent_dir, workspace, _stamp(record, lesson, key), locked)
    await _mirror_lessons(owner_id, slug, [(kind, text)], taught_by)
    await _settle_taught(owner_id, slug, agent_dir, kind, text, lesson, "" if proposal else key)
    return out


async def _settle_taught(owner_id: str, slug: str, agent_dir: Path, kind: str, text: str, lesson: str,
                         key: str) -> None:
    """A lesson was taught: what the agent noticed in the chat message it
    was saved from (*key*), or one saying the same, waits in Lessons to
    approve no more — it's listed as just learned, as this lesson. Never
    fails the teaching."""
    from captain_claw.flight_deck import lesson_proposals as lp
    try:
        async with _SUGGESTION_LOCK:
            items = await asyncio.to_thread(_load_suggestions, agent_dir)
            hits = [s for s in items if s.get("origin") == "auto" and s.get("state") == "waiting" and (
                (key and said_key(str(s.get("said") or "")) == key)
                or (s.get("text") and lp._same(text, str(s["text"]))))]
            for s in hits:
                s.update(state="taken", kind=kind, text=text, lesson=lesson, settled=time.time())
            if hits:
                await asyncio.to_thread(_write_suggestions, agent_dir, items)
        if hits:
            log.info("agent lessons: lessons to approve settled by one taught", agent=slug, settled=len(hits))
            await lp._settle_bell(owner_id, slug, agent_dir)
    except Exception:
        log.debug("agent lessons: settling lessons to approve failed", exc_info=True)


async def _mirror_lessons(owner_id: str, slug: str, lessons: list[tuple[str, str]], taught_by: str) -> None:
    """Copy newly taught lessons into the owner's external memory (when one
    is connected and lessons are mirrored). Never fails the teaching."""
    try:
        from datetime import UTC, datetime

        from captain_claw.flight_deck import server as srv
        from captain_claw.flight_deck.external_memory import service as em
        from captain_claw.flight_deck.external_memory.base import MemoryItem

        name = str((srv._load_process_registry().get(slug) or {}).get("name") or slug)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        labels = {"rule": "rule", "ask_first": "ask first", "fact": "fact"}
        for kind, text in lessons:
            await em.mirror_for_owner(owner_id, MemoryItem(
                kind="lesson",
                source_id=f"{slug}:{agent_lessons.lesson_id(kind, text)}",
                content=text,
                agent=name,
                created_at=now,
                meta={"category": labels.get(kind, kind), "source": f"taught by {taught_by}"},
            ))
    except Exception:
        log.debug("agent lessons: external memory mirror failed", exc_info=True)


async def teach_rules(slug: str, user: dict, request: Request, rules: list[str], *, source: str) -> dict:
    """Rules a Captain Swarm run found better, taught when their owner says so
    (``taught_by: swarm``). One the agent has already is skipped. Like rules
    taught by hand, they reach the agent with the messages they bear on."""
    owner_id, agent_dir, workspace = _place(slug, user)
    layer, locked = _teaching(request, user)
    record = await asyncio.to_thread(ap.earlier_setup, agent_dir, workspace)
    added = 0
    for raw in rules:
        text = _clean(raw)
        if not text or _taken(record, "rule", text):
            continue
        item = {"text": text, "always": False, "source": {"doc": "captain swarm", "section": source[:200]},
                "taught_by": "swarm", "added": time.time(), **({"layer": layer} if layer else {})}
        record = {**record, _GROUP["rule"]: [item, *(record.get(_GROUP["rule"]) or [])]}
        added += 1
    if not added:
        return {"added": 0}
    log.info("agent lessons: rules taught from a Captain Swarm run", agent=slug, added=added)
    out = {**await _save(slug, owner_id, agent_dir, workspace, record, locked), "added": added}
    taught = [("rule", str(i.get("text") or "")) for i in (record.get(_GROUP["rule"]) or [])[:added]]
    await _mirror_lessons(owner_id, slug, [t for t in taught if t[1]], "swarm")
    return out


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
    return await remove_one(slug, user, request, lesson_id)


async def remove_one(slug: str, user: dict, request: Request, lesson_id: str) -> dict:
    """Take one lesson away from the caller's agent (404 when it's gone; the
    base's only by an admin)."""
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


_SHOWN = ("id", "origin", "kind", "text", "always", "quote", "said", "reply", "channel", "by_name", "created",
          "state", "lesson", "settled")


def shown(s: dict) -> dict:
    """One suggestion as the pages list it: a member's (their words; drafted
    when its owner takes it) or one the agent noticed (``origin: auto``,
    drafted already, with the words it came from)."""
    return {**{k: s.get(k) for k in _SHOWN}, "origin": s.get("origin") or "member",
            "kind": s.get("kind") or "", "text": s.get("text") or "", "always": bool(s.get("always")),
            "quote": s.get("quote") or "", "channel": s.get("channel") or "", "by_name": s.get("by_name") or ""}


def _waiting(agent_dir: Path) -> list[dict]:
    """What members suggested, and the agent noticed, that its owner hasn't
    taken or declined, newest first."""
    return [shown(s) for s in _load_suggestions(agent_dir) if s.get("state") == "waiting"]


class SuggestBody(BaseModel):
    agent_ref: str = Field(..., max_length=300)
    said: str = Field(..., max_length=4000)      # the member's words
    reply: str = Field("", max_length=8000)      # what the agent had answered them
    said_key: str = Field("", max_length=20)     # the chat message they suggest it from (said_key)


async def _shared_with(agent_ref: str, uid: str):
    """The shared agent *agent_ref* — a process the caller is a member of
    (404 otherwise)."""
    from captain_claw.flight_deck import agent_sharing as sharing
    from captain_claw.flight_deck.auth import get_db

    rec = sharing.resolve_agent_record(agent_ref) if sharing.sharing_active() else None
    if (rec is None or rec.runtime != "process" or rec.owner == uid
            or not await sharing.member_check(get_db(), rec.ref, rec.owner, uid, max_age=0)):
        raise HTTPException(404, "That agent isn't shared with you.")
    return rec


@router.get("/shared/suggested")
async def suggested(agent_ref: str, user: dict = Depends(get_current_user)) -> dict:
    """The caller's chat messages with a shared agent they suggested a
    lesson from, by ``said_key`` — waiting for its owner, taught, or
    dismissed — so their chat doesn't offer them again."""
    from captain_claw.flight_deck import server as srv

    uid = str(user.get("id") or "")
    rec = await _shared_with(agent_ref, uid)
    items = await asyncio.to_thread(_load_suggestions, srv.DATA_DIR / rec.slug)
    return {"marks": chat_marks({}, items, by=uid)}


@router.post("/shared/suggest")
async def suggest_lesson(body: SuggestBody, user: dict = Depends(get_current_user)) -> dict:
    """A member of someone's shared agent suggests a lesson from their chat
    with it — their words, and the answer they correct — for its owner to
    take or decline (a bell tells them). Only its members; nothing is drafted
    until its owner looks."""
    from captain_claw.flight_deck import server as srv
    from captain_claw.flight_deck import tenant_profile
    from captain_claw.flight_deck.auth import get_db

    uid = str(user.get("id") or "")
    db = get_db()
    rec = await _shared_with(body.agent_ref, uid)
    said = " ".join(body.said.split())
    if not said:
        raise HTTPException(400, "Say what it should learn.")
    key = clean_key(body.said_key)
    name = await tenant_profile.owner_name(db, uid) or "A member"
    agent_dir = srv.DATA_DIR / rec.slug
    async with _SUGGESTION_LOCK:
        items = await asyncio.to_thread(_load_suggestions, agent_dir)
        waiting = [s for s in items if s.get("state") == "waiting"]
        if any(_flat(s.get("said") or "") == _flat(said)
               or (key and s.get("by") == uid and s.get("said_key") == key) for s in waiting):
            return {"ok": True, "already": True}
        if sum(1 for s in waiting if s.get("by") == uid) >= _SUGGESTIONS_WAITING:
            raise HTTPException(429, f"You have {_SUGGESTIONS_WAITING} suggestions waiting for its owner already.")
        item = {"id": secrets.token_hex(6), "said": said, "reply": body.reply.strip()[:6000], "by": uid,
                "by_name": name, "created": time.time(), "state": "waiting", **({"said_key": key} if key else {})}
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
