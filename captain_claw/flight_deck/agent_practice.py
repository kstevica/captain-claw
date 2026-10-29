"""Practice — the owner tries the agent on real tasks, and corrects it.

The Onboarding page's Practice tab: the owner gives the agent a task the way
a user would ("A customer writes: my parcel came damaged — what do you
reply?"), or takes one suggested from its documents (with the answer they
give, to judge by). The agent answers on its exam lane (exam_turn: in a
session of its own that keeps nothing, and can't send or change anything).
The owner marks the answer good, or says what it should have done — which
becomes a lesson (agent_onboarding: drafted by its model, checked by them,
``taught_by: practice``) — and has it try again.

Tasks are kept per agent (``practice.json`` in its folder, the newest 50),
each with its attempts.

    GET    /fd/agent-practice/{slug}                its tasks, newest first
    POST   /fd/agent-practice/{slug}                a new task, tried at once
    POST   /fd/agent-practice/{slug}/suggest        tasks written from its documents
    POST   /fd/agent-practice/{slug}/{id}/try       try it again
    POST   /fd/agent-practice/{slug}/{id}/mark      an attempt good, or to fix (with the lesson taught for it)
    DELETE /fd/agent-practice/{slug}/{id}           forget a task
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck import agent_exam as ex
from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck.auth import get_managed_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/agent-practice", tags=["agent-practice"])

_FILE = "practice.json"
_KEEP = 50                    # tasks kept per agent
_ATTEMPTS = 10                # attempts kept per task
_TASK_MAX = 2000
_SUGGEST = 3                  # tasks a suggestion writes
_RUNNING: dict[str, asyncio.Task] = {}   # "slug/task/attempt" → its answer under way
_LOCKS: dict[str, asyncio.Lock] = {}     # one writer of an agent's practice file at a time


def _path(slug: str) -> Path:
    from captain_claw.flight_deck import server as srv
    return srv.DATA_DIR / slug / _FILE


def load(slug: str) -> list[dict]:
    try:
        data = json.loads(_path(slug).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    tasks = data.get("tasks") if isinstance(data, dict) else None
    return [t for t in tasks or [] if isinstance(t, dict) and t.get("id")]


def _write(slug: str, tasks: list[dict]) -> None:
    path = _path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{_FILE}.tmp")
    tmp.write_text(json.dumps({"tasks": tasks[:_KEEP]}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _lock(slug: str) -> asyncio.Lock:
    return _LOCKS.setdefault(slug, asyncio.Lock())


async def _update(slug: str, task_id: str, fn: Any) -> dict | None:
    """Change one task under the agent's lock; the task as changed (None: gone)."""
    async with _lock(slug):
        tasks = load(slug)
        for t in tasks:
            if t["id"] == task_id:
                fn(t)
                _write(slug, tasks)
                return t
        return None


def view(slug: str, t: dict) -> dict:
    """A task as the page shows it: an attempt still being answered says so
    (a restart of Flight Deck under it: it stopped)."""
    out = {**t, "attempts": [dict(a) for a in t.get("attempts") or []]}
    for a in out["attempts"]:
        if a.get("state") == "answering" and f"{slug}/{t['id']}/{a['id']}" not in _RUNNING:
            a["state"], a["error"] = "error", a.get("error") or "Flight Deck restarted while it was answering."
    return out


async def _answer(slug: str, name: str, task_id: str, attempt_id: str, text: str) -> None:
    """One attempt: the agent answers on its exam lane; kept with the task."""
    job = ex.ExamJob(id=f"practice{task_id}", agent=slug, agent_name=name)
    got: dict[str, Any]
    try:
        if not await ex.can_take_exams(job):
            got = {"error": f"Restart {name} first: it was started before it could practise."}
        else:
            got = await ex.ask(job, {"id": attempt_id, "question": text})
    except ex._ExamError as exc:
        got = {"error": str(exc)}
    except Exception as exc:
        log.warning("agent practice: an attempt failed", agent=slug, exc_info=True)
        got = {"error": f"It couldn't answer: {exc}"}
    finally:
        _RUNNING.pop(f"{slug}/{task_id}/{attempt_id}", None)

    def done(t: dict) -> None:
        for a in t.get("attempts") or []:
            if a["id"] == attempt_id:
                a.update(state="error" if got.get("error") else "done", finished=time.time(),
                         answer=str(got.get("answer") or ""), error=str(got.get("error") or ""),
                         lookups=ex._kept(got.get("lookups") or []), tokens=int(got.get("tokens") or 0),
                         seconds=got.get("seconds"))

    await _update(slug, task_id, done)


def _start(slug: str, name: str, t: dict) -> dict:
    """A new attempt of *t*, answered in the background."""
    attempt = {"id": secrets.token_hex(4), "state": "answering", "started": time.time(), "verdict": ""}
    t["attempts"] = [*(t.get("attempts") or []), attempt][-_ATTEMPTS:]
    key = f"{slug}/{t['id']}/{attempt['id']}"
    _RUNNING[key] = asyncio.get_running_loop().create_task(_answer(slug, name, t["id"], attempt["id"], t["task"]))
    return attempt


def _place(slug: str, user: dict) -> tuple[str, dict, Path]:
    owner_id = ap._owner(user)
    entry = ap._own_process(slug, owner_id)
    from captain_claw.flight_deck import server as srv
    agent_dir = srv.DATA_DIR / slug
    workspace = Path(entry["workspace"]) if entry.get("workspace") else agent_dir / "data" / "workspace"
    return owner_id, entry, workspace


async def _can_ask(owner_id: str, slug: str) -> None:
    if await ap._quota_spent(owner_id):
        raise HTTPException(429, "Today's allowance is used up. Practise again tomorrow.")
    reader, _ = await ap.choose_reader(owner_id, slug)
    if not reader or not reader.get("agent"):
        raise HTTPException(409, "Switch it on first: it answers the task itself.")


@router.get("/{slug}")
async def list_tasks(slug: str, user: dict = Depends(get_managed_user)) -> dict:
    _place(slug, user)
    return {"tasks": [view(slug, t) for t in await asyncio.to_thread(load, slug)]}


class TaskBody(BaseModel):
    task: str = Field(..., max_length=_TASK_MAX)
    expected: str = Field("", max_length=2000)          # what its documents say (a suggested task)
    source: dict = Field(default_factory=dict)          # {doc, doc_title, title, path} of a suggested task


@router.post("/{slug}")
async def new_task(slug: str, body: TaskBody, user: dict = Depends(get_managed_user)) -> dict:
    """A task, as a user would give it; the agent tries it at once."""
    owner_id, entry, _ = _place(slug, user)
    text = body.task.strip()
    if not text:
        raise HTTPException(400, "Give it a task to try.")
    await _can_ask(owner_id, slug)
    src = {k: body.source.get(k) for k in ("doc", "doc_title", "title", "path") if body.source.get(k)}
    t = {"id": secrets.token_hex(6), "task": text, "expected": body.expected.strip(), "source": src,
         "created": time.time(), "attempts": []}
    async with _lock(slug):
        _start(slug, str(entry.get("name") or slug), t)
        _write(slug, [t, *load(slug)])
    return view(slug, t)


@router.post("/{slug}/{task_id}/try")
async def try_again(slug: str, task_id: str, user: dict = Depends(get_managed_user)) -> dict:
    owner_id, entry, _ = _place(slug, user)
    await _can_ask(owner_id, slug)
    t = await _update(slug, task_id, lambda t: _start(slug, str(entry.get("name") or slug), t))
    if t is None:
        raise HTTPException(404, "That task isn't here any more.")
    return view(slug, t)


class MarkBody(BaseModel):
    attempt: str = Field(..., max_length=20)
    verdict: str = Field(..., max_length=10)            # good | fix | "" (unmarked)
    note: str = Field("", max_length=2000)              # what it should have done
    lesson: str = Field("", max_length=40)              # the lesson taught for it


@router.post("/{slug}/{task_id}/mark")
async def mark(slug: str, task_id: str, body: MarkBody, user: dict = Depends(get_managed_user)) -> dict:
    _place(slug, user)
    if body.verdict not in ("good", "fix", ""):
        raise HTTPException(400, "An answer is good, or to fix.")
    found: list[bool] = []

    def set_verdict(t: dict) -> None:
        for a in t.get("attempts") or []:
            if a["id"] == body.attempt:
                a.update(verdict=body.verdict, note=body.note.strip(), lesson=body.lesson)
                found.append(True)

    t = await _update(slug, task_id, set_verdict)
    if t is None or not found:
        raise HTTPException(404, "That answer isn't here any more.")
    return view(slug, t)


@router.delete("/{slug}/{task_id}")
async def forget(slug: str, task_id: str, user: dict = Depends(get_managed_user)) -> dict:
    _place(slug, user)
    async with _lock(slug):
        tasks = load(slug)
        left = [t for t in tasks if t["id"] != task_id]
        if len(left) == len(tasks):
            raise HTTPException(404, "That task isn't here any more.")
        _write(slug, left)
    return {"ok": True}


class SuggestBody(BaseModel):
    focus: str = Field("", max_length=300)


@router.post("/{slug}/suggest")
async def suggest(slug: str, body: SuggestBody, user: dict = Depends(get_managed_user)) -> dict:
    """A few tasks to try, written from its documents as an exam's
    situations are (agent_exam), each with the answer its documents give."""
    owner_id, _, workspace = _place(slug, user)
    if await ap._quota_spent(owner_id):
        raise HTTPException(429, "Today's allowance is used up. Try again tomorrow.")
    reader, why = await ap.choose_reader(owner_id, slug)
    if not reader:
        raise HTTPException(409, why or "No model is set up to write tasks with.")
    srcs = await ex.sources(slug, workspace)
    if not srcs:
        raise HTTPException(409, "It has no documents to write tasks from. Give it some first.")
    picks = await asyncio.to_thread(ex.sample, srcs, _SUGGEST, body.focus.strip(), secrets.randbits(32))
    picks = sorted(picks, key=lambda p: -p["score"])[:_SUGGEST + 1]
    for p in picks:
        p["kind"] = "situation"
    job = ex.ExamJob(id="suggest", agent=slug, owner_id=owner_id, owner=ap.owner_key(owner_id))
    text_of = {s.file: s.text for s in srcs}
    try:
        reply = await ap._call(reader, ex._QUESTIONS_PROMPT.read_text(encoding="utf-8"),
                               ex.writing_prompt(picks, text_of), owner_id, job, ex._MAX_TOKENS)
    except Exception as exc:
        log.info("agent practice: no suggestions", agent=slug, error=str(exc)[:200])
        raise HTTPException(502, f"{reader.get('model') or 'Its model'} couldn't write tasks just now.") from exc
    got = ex.parse_questions(reply.text, picks)[:_SUGGEST]
    return {"tasks": [{"task": q["question"], "expected": q["expected"],
                       "source": {"doc": q["doc"], "doc_title": q["doc_title"], "title": q["title"], "path": q["path"]}}
                      for q in got]}
