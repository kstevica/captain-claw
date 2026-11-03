"""Captain Swarm — the owner's routes (Phase 1: practice sets and measurements).

docs/captain-swarm-plan.md. Phase 1 builds no swarm yet, only what it will be
measured by: a practice set frozen from the owner's own mail, and a
measurement of one of the owner's agents on it.

    GET    /fd/captain-swarm/practice                         the owner's sets
    POST   /fd/captain-swarm/practice/mail-triage             build one from their Gmail
    GET    /fd/captain-swarm/practice/{set}                   one set (and how its build goes)
    DELETE /fd/captain-swarm/practice/{set}                   delete it, mail copy and all
    POST   /fd/captain-swarm/practice/{set}/measure           measure an agent on it
    GET    /fd/captain-swarm/practice/{set}/measurements      what was measured
    GET    /fd/captain-swarm/measurements/{set}/{id}          one measurement
    POST   /fd/captain-swarm/measurements/{set}/{id}/stop     stop one under way

Everything is the caller's own: their Google account, their sets, their
agents. Building and measuring run in the background; the state is kept
with the set, so a restart shows where it stopped.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.flight_deck import captain_swarm_measure as measure
from captain_claw.flight_deck import captain_swarm_practice as practice
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/captain-swarm", tags=["captain-swarm"])

_TASKS: dict[str, asyncio.Task] = {}      # "build:<owner>/<set>" or "measure:<owner>/<set>/<id>"
_PROGRESS: dict[str, dict] = {}
_STOP: set[str] = set()
_MAX_MEASURE_CASES = 300
_MAX_REPEAT = 5


def _uid(user: dict) -> str:
    return str(user.get("id") or "local")


def _meta_or_404(owner: str, set_id: str) -> dict:
    meta = practice.get_meta(owner, set_id)
    if meta is None:
        raise HTTPException(404, "No such practice set.")
    return meta


def _view(owner: str, meta: dict) -> dict:
    key = f"build:{owner}/{meta['id']}"
    out = {**meta}
    if meta.get("state") == "building":
        out["progress"] = _PROGRESS.get(key, {})
        if key not in _TASKS:
            out["state"], out["error"] = "error", meta.get("error") or "Flight Deck restarted while it was building."
    return out


# ── practice sets ────────────────────────────────────────────────────


@router.get("/practice")
async def list_practice(user: dict = Depends(get_current_user)) -> dict:
    owner = _uid(user)
    return {"sets": [_view(owner, m) for m in await asyncio.to_thread(practice.list_sets, owner)]}


class TriageBody(BaseModel):
    window_days: int = Field(90, ge=14, le=365)
    reply_days: int = Field(3, ge=1, le=14)
    holdout_days: int = Field(21, ge=7, le=90)
    max_cases: int = Field(400, ge=20, le=1000)


async def _build(owner: str, set_id: str, body: TriageBody) -> None:
    key = f"build:{owner}/{set_id}"
    from captain_claw.flight_deck.google_oauth_routes import get_valid_google_access_token
    try:
        token = await get_valid_google_access_token(owner)
        if not token:
            raise RuntimeError("Connect Google first: the set is read from your own Gmail.")

        def progress(stage: str, done: int, total: int) -> None:
            _PROGRESS[key] = {"stage": stage, "done": done, "total": total}

        async with httpx.AsyncClient(timeout=30.0) as client:
            built = await practice.build_mail_triage(
                practice.gmail_fetch(client, token), now_ms=int(time.time() * 1000),
                window_days=body.window_days, reply_days=body.reply_days,
                holdout_days=body.holdout_days, max_cases=body.max_cases, progress=progress)
        if not built["open"] or not built["holdout"]:
            raise RuntimeError("Too little mail in that window to hold any back. Try a longer window.")
        await asyncio.to_thread(practice.save_set, owner, set_id, built, body.model_dump())
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning("captain swarm: building a practice set failed", owner=owner, exc_info=True)
        await asyncio.to_thread(practice.mark, owner, set_id, state="error", error=str(exc)[:300])
    finally:
        _TASKS.pop(key, None)
        _PROGRESS.pop(key, None)


@router.post("/practice/mail-triage")
async def build_mail_triage(body: TriageBody, user: dict = Depends(get_current_user)) -> dict:
    """Freeze a sample of the caller's received mail, each labelled by whether
    they wrote in its thread within *reply_days*. Read-only on Gmail."""
    owner = _uid(user)
    if any(k.startswith(f"build:{owner}/") for k in _TASKS):
        raise HTTPException(409, "A practice set is already being built.")
    set_id = practice.new_set_id()
    await asyncio.to_thread(practice.mark, owner, set_id, state="building", title="Inbox triage: did you reply?",
                            source={"params": body.model_dump()})
    _TASKS[f"build:{owner}/{set_id}"] = asyncio.get_running_loop().create_task(_build(owner, set_id, body))
    return _view(owner, _meta_or_404(owner, set_id))


@router.get("/practice/{set_id}")
async def get_practice(set_id: str, user: dict = Depends(get_current_user)) -> dict:
    owner = _uid(user)
    return _view(owner, _meta_or_404(owner, set_id))


@router.delete("/practice/{set_id}")
async def delete_practice(set_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Delete the set: the frozen mail and every measurement made on it."""
    owner = _uid(user)
    _meta_or_404(owner, set_id)
    if f"build:{owner}/{set_id}" in _TASKS:
        # Its last write runs in a thread a cancel can't stop: deleting now
        # could leave the mail copy behind after the folder is gone.
        raise HTTPException(409, "It's still being built. Delete it once it's done.")
    for key in [k for k in _TASKS if k.startswith(f"measure:{owner}/{set_id}/")]:
        _STOP.add(key)
        _TASKS[key].cancel()
    return {"deleted": await asyncio.to_thread(practice.delete_set, owner, set_id)}


# ── measurements ─────────────────────────────────────────────────────


class MeasureBody(BaseModel):
    agent: str = Field(..., min_length=1, max_length=120)
    split: str = Field("open", pattern="^(open|holdout)$")
    cases: int = Field(50, ge=5, le=_MAX_MEASURE_CASES)
    repeat: int = Field(3, ge=1, le=_MAX_REPEAT)


def _measure_path(owner: str, set_id: str, mid: str):
    if not practice._ID_RE.match(mid or ""):
        raise HTTPException(404, "No such measurement.")
    return practice.set_dir(owner, set_id) / "measurements" / f"{mid}.json"


def _read_measurement(owner: str, set_id: str, mid: str) -> dict | None:
    return practice._read_json(_measure_path(owner, set_id, mid))


def _measurement_view(owner: str, set_id: str, m: dict, *, full: bool = False) -> dict:
    key = f"measure:{owner}/{set_id}/{m['id']}"
    out = {k: v for k, v in m.items() if full or k != "result" or not isinstance(v, dict)}
    if not full and isinstance(m.get("result"), dict):
        out["result"] = {k: v for k, v in m["result"].items() if k != "cases"}
    if m.get("state") == "running":
        out["progress"] = _PROGRESS.get(key, {})
        if key not in _TASKS:
            out["state"], out["error"] = "error", m.get("error") or "Flight Deck restarted while it was measuring."
    return out


async def _ask_agent(owner: str, slug: str, name: str, mid: str):
    """An ask over the agent's exam lane: read-only, keeps nothing."""
    from captain_claw.flight_deck import agent_exam as ex
    job = ex.ExamJob(id=f"swarm{mid}", agent=slug, agent_name=name, owner_id=owner)
    if not await ex.can_take_exams(job):
        raise HTTPException(409, f"Restart {name} first: it was started before it could answer practice cases.")
    counter = 0

    async def ask(question: str) -> dict:
        nonlocal counter
        counter += 1
        try:
            return await ex.ask(job, {"id": f"c{counter}", "question": question})
        except ex._ExamError as exc:
            return {"error": str(exc)}
    return ask


async def _measure(owner: str, set_id: str, mid: str, record: dict, cases: list[dict], ask, owner_name: str) -> None:
    key = f"measure:{owner}/{set_id}/{mid}"
    path = _measure_path(owner, set_id, mid)
    try:
        try:
            result = await measure.run_triage(
                cases, ask=ask, repeat=record["repeat"], owner_name=owner_name,
                progress=lambda done, total: _PROGRESS.__setitem__(key, {"done": done, "total": total}),
                should_stop=lambda: key in _STOP)
            record.update(state="stopped" if result["stopped"] else "done", finished=time.time(), result=result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("captain swarm: a measurement failed", owner=owner, exc_info=True)
            record.update(state="error", finished=time.time(), error=str(exc)[:300])
        # Written while it still counts as running, so a reader never sees a
        # finished measurement as one a restart cut short.
        if practice.set_dir(owner, set_id).is_dir():
            await asyncio.to_thread(practice.write_json, path, record)
    finally:
        _TASKS.pop(key, None)
        _PROGRESS.pop(key, None)
        _STOP.discard(key)


@router.post("/practice/{set_id}/measure")
async def start_measurement(set_id: str, body: MeasureBody, user: dict = Depends(get_current_user)) -> dict:
    """Ask one of the caller's agents every chosen case, *repeat* times, on its
    exam lane, and score its YES/NO against what the caller actually did."""
    from captain_claw.flight_deck import agent_preset as ap
    owner = _uid(user)
    meta = _meta_or_404(owner, set_id)
    if meta.get("state") != "ready":
        raise HTTPException(409, "That set isn't ready.")
    entry = ap._own_process(body.agent, ap._owner(user))
    if any(k.startswith(f"measure:{owner}/") for k in _TASKS):
        raise HTTPException(409, "A measurement is already under way. Wait for it, or stop it.")
    if await ap._quota_spent(ap._owner(user)):
        raise HTTPException(429, "Today's allowance is used up.")
    cases = measure.pick_cases(await asyncio.to_thread(practice.load_cases, owner, set_id, body.split), body.cases)
    if not cases:
        raise HTTPException(409, "That part of the set has no cases.")
    name = str(entry.get("name") or body.agent)
    mid = f"m{time.strftime('%Y%m%d%H%M')}-{secrets.token_hex(2)}"
    ask = await _ask_agent(ap._owner(user), body.agent, name, mid)
    record = {"id": mid, "set": set_id, "agent": body.agent, "agent_name": name, "split": body.split,
              "cases": len(cases), "repeat": body.repeat, "state": "running", "started": time.time(),
              "scorer": "agreement", "question": "triage_v1"}
    await asyncio.to_thread(practice.write_json, _measure_path(owner, set_id, mid), record)
    owner_name = str(user.get("display_name") or user.get("name") or "")
    if owner_name.lower() == "local":
        owner_name = ""
    key = f"measure:{owner}/{set_id}/{mid}"
    _TASKS[key] = asyncio.get_running_loop().create_task(
        _measure(owner, set_id, mid, record, cases, ask, owner_name))
    return _measurement_view(owner, set_id, record)


@router.get("/practice/{set_id}/measurements")
async def list_measurements(set_id: str, user: dict = Depends(get_current_user)) -> dict:
    owner = _uid(user)
    _meta_or_404(owner, set_id)
    folder = practice.set_dir(owner, set_id) / "measurements"
    rows = []
    if folder.is_dir():
        for p in sorted(folder.glob("*.json"), reverse=True):
            m = practice._read_json(p)
            if m and m.get("id"):
                rows.append(_measurement_view(owner, set_id, m))
    return {"measurements": rows}


@router.get("/measurements/{set_id}/{mid}")
async def get_measurement(set_id: str, mid: str, user: dict = Depends(get_current_user)) -> dict:
    owner = _uid(user)
    _meta_or_404(owner, set_id)
    m = _read_measurement(owner, set_id, mid)
    if m is None:
        raise HTTPException(404, "No such measurement.")
    return _measurement_view(owner, set_id, m, full=True)


@router.post("/measurements/{set_id}/{mid}/stop")
async def stop_measurement(set_id: str, mid: str, user: dict = Depends(get_current_user)) -> dict:
    owner = _uid(user)
    key = f"measure:{owner}/{set_id}/{mid}"
    if key not in _TASKS:
        raise HTTPException(409, "It isn't running.")
    _STOP.add(key)
    return {"stopping": True}
