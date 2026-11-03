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
import os
import secrets
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
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


# ── runs (Phase 2: the loop, on code) ────────────────────────────────
#
#    GET    /fd/captain-swarm/templates                 what a run can start from, with this deck's defaults
#    POST   /fd/captain-swarm/runs                      start one (the owner's own: no gate)
#    GET    /fd/captain-swarm/runs                      the owner's runs
#    GET    /fd/captain-swarm/runs/{id}                 one run: objective, ledger, events
#    POST   /fd/captain-swarm/runs/{id}/cancel          stop it now
#    POST   /fd/captain-swarm/runs/{id}/go | /decline   an agent's run waits for one of these
#    POST   /fd/captain-swarm/runs/{id}/pr              open a pull request from the best commit
#    POST   /fd/captain-swarm/agent/{start,status,get,list,cancel}   the `captain_swarm` tool

_MAX_ACTIVE_RUNS = int(os.environ.get("FD_CAPTAIN_SWARM_MAX_RUNS", "1") or 1)


def _cs_store():
    from captain_claw.flight_deck import captain_swarm_loop as loop_mod
    store = loop_mod.get_store()
    if store is None:
        raise HTTPException(503, "Captain Swarm isn't running on this deck (FD_CAPTAIN_SWARM_DISABLED).")
    return store


def _source_repo() -> str:
    """This deck's own source checkout, when it is a git repository: the Mrav
    template improves the code the deck runs."""
    from pathlib import Path

    import captain_claw
    root = Path(captain_claw.__file__).resolve().parents[1]
    return str(root) if (root / ".git").exists() else ""


async def _local_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://127.0.0.1:11434/api/tags")
        return sorted(m["name"] for m in r.json().get("models") or [] if not str(m.get("name", "")).endswith("cloud"))
    except Exception:
        return []


def _default_holdout() -> str:
    path = practice.sealed_root() / "mrav" / "holdout.json"
    return str(path) if path.is_file() else ""


@router.get("/templates")
async def templates(user: dict = Depends(get_current_user)) -> dict:
    from captain_claw.flight_deck import captain_swarm_objective as om
    from captain_claw.flight_deck.bat_routes import _owner_has_shell
    sets = [{"id": m["id"], "title": m.get("title") or m["id"], "open": (m.get("open") or {}).get("cases", 0),
             "holdout": (m.get("holdout") or {}).get("cases", 0)}
            for m in await asyncio.to_thread(practice.list_sets, _uid(user)) if m.get("state") == "ready"]
    return {"templates": {"outcome": {
        "title": "Find the best result for a task",
        "about": "Several attempts at one real task — each a Bat run in rehearsal: it searches, reads and browses, "
                 "but buying, sending, booking and submitting are recorded instead of done, and the browser stops "
                 "at the commit point. Captain opens every link itself and counts only what the page bears out. "
                 "You carry the winner out once, through Bat's gate.",
        "defaults": {"label": "price", "direction": "min", "per_round": 2, "attempt_usd": 2.0,
                     "budget_usd": 10.0, "budget_hours": 4.0, "budget_max_experiments": 6},
    }, "triage": {
        "title": "Triage my inbox the way I would",
        "about": "Improve an agent's triage rules against what you actually did: every version is tried on the "
                 "agent's exam lane (it can't send or change anything) over a practice set from your own mail, "
                 "and the best rules reach the agent only when you say so.",
        "sets": sets,
        "defaults": {"cases": 40, "repeat": 2, "holdout_cases": 0, "per_round": 2,
                     "budget_usd": 20.0, "budget_hours": 8.0, "budget_max_experiments": 16},
    }, "mrav": {
        "title": "Mrav passes more of its eval",
        "about": "Improve Captain's small-model runtime (captain_claw/mrav/), measured by the Mrav eval on a "
                 "local model: 20 open tasks in the repo, held-out tasks only on this deck.",
        "defaults": {"repo_path": _source_repo(), "model": "gemma4:e2b", "base_url": "http://127.0.0.1:11434",
                     "harness": "claude", "holdout_file": _default_holdout(), "per_round": 2,
                     **{f"budget_{k}": v for k, v in om.DEFAULT_BUDGET.items()}},
        "local_models": await _local_models(),
    }}, "can_start": _owner_has_shell(user)}


class RunBody(BaseModel):
    template: str = "mrav"
    options: dict[str, Any] = Field(default_factory=dict)
    objective: dict[str, Any] | None = None


def _objective_from(body: RunBody, has_shell: bool, user: dict | None = None) -> dict:
    from captain_claw.flight_deck import captain_swarm_objective as om
    from captain_claw.flight_deck import captain_swarm_rules as rules
    if body.objective and body.objective.get("kind") == "outcome" or (not body.objective and body.template == "outcome"):
        from captain_claw.flight_deck import captain_swarm_outcome as outcome
        o = body.options or {}
        raw = body.objective or outcome.outcome_template(
            task=str(o.get("task") or ""), label=str(o.get("label") or "price"),
            direction=str(o.get("direction") or "min"),
            max_value=None if o.get("max_value") in (None, "") else o.get("max_value"),
            must_include=[w.strip() for w in str(o.get("must_include") or "").split(",") if w.strip()],
            currency=str(o.get("currency") or ""))
        if not body.objective:
            if o.get("per_round"):
                raw["lane"]["per_round"] = o["per_round"]
            if o.get("attempt_usd"):
                raw["lane"]["attempt_usd"] = o["attempt_usd"]
            for k in ("usd", "hours", "max_experiments"):
                if o.get(f"budget_{k}") not in (None, ""):
                    raw["budget"][k] = o[f"budget_{k}"]
        try:
            return outcome.normalize(raw)
        except om.ObjectiveError as exc:
            raise HTTPException(400, str(exc)) from exc
    if body.objective and body.objective.get("kind") == "practice":
        try:
            return rules.normalize(body.objective)
        except om.ObjectiveError as exc:
            raise HTTPException(400, str(exc)) from exc
    if not body.objective and body.template == "triage":
        o = body.options or {}
        owner_name = str((user or {}).get("display_name") or (user or {}).get("name") or "")
        raw = rules.triage_template(
            set_id=str(o.get("set_id") or ""), agent=str(o.get("agent") or ""),
            agent_name=str(o.get("agent_name") or ""), owner_name="" if owner_name.lower() == "local" else owner_name,
            cases=int(o.get("cases") or 40), repeat=int(o.get("repeat") or 2),
            holdout_cases=int(o.get("holdout_cases") or 0))
        if o.get("per_round"):
            raw["lane"]["per_round"] = o["per_round"]
        for k in ("usd", "hours", "max_experiments"):
            if o.get(f"budget_{k}") not in (None, ""):
                raw["budget"][k] = o[f"budget_{k}"]
        try:
            return rules.normalize(raw)
        except om.ObjectiveError as exc:
            raise HTTPException(400, str(exc)) from exc
    if body.objective:
        raw = body.objective
    else:
        if body.template not in om.TEMPLATES:
            raise HTTPException(400, f"No template called {body.template!r}.")
        o = body.options or {}
        raw = om.TEMPLATES[body.template](
            str(o.get("repo_path") or _source_repo()), model=str(o.get("model") or "gemma4:e2b"),
            base_url=str(o.get("base_url") or "http://127.0.0.1:11434"), harness=str(o.get("harness") or "claude"),
            holdout_file=str(o.get("holdout_file") or _default_holdout()))
        if o.get("lane_model"):
            raw["lane"]["model"] = str(o["lane_model"])
        if o.get("per_round"):
            raw["lane"]["per_round"] = o["per_round"]
        for k in ("usd", "hours", "max_experiments"):
            if o.get(f"budget_{k}") not in (None, ""):
                raw["budget"][k] = o[f"budget_{k}"]
        if o.get("target") not in (None, ""):
            raw["stop"]["target"] = o["target"]
    try:
        return om.normalize(raw, owner_has_shell=has_shell)
    except om.ObjectiveError as exc:
        raise HTTPException(400, str(exc)) from exc


async def _can_run(owner: str, objective: dict | None = None, user: dict | None = None) -> None:
    """What a run needs before it starts: a model to plan with, no other run of
    this owner's under way, and for code Crew open for a local repository; for
    a practice run, the owner's own ready set and their own agent."""
    from captain_claw.flight_deck import crew_access, crew_runner
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.basna_routes import _load_owner_tiers
    from captain_claw.flight_deck.bat_routes import _creds_for
    tiers, _ = await _load_owner_tiers(get_db(), owner)
    if not _creds_for(tiers):
        raise HTTPException(409, "Set up a model tier first: a run asks a model for ideas.")
    if (objective or {}).get("kind") == "outcome":
        from captain_claw.flight_deck import bat_loop
        if bat_loop.get_store() is None:
            raise HTTPException(503, "Bat isn't running on this deck, and every attempt is a Bat run.")
    elif (objective or {}).get("kind") == "practice":
        from captain_claw.flight_deck import agent_preset as ap
        subj = objective["subject"]
        meta = practice.get_meta(owner, subj["set_id"])
        if not meta or meta.get("state") != "ready":
            raise HTTPException(409, "That practice set isn't ready.")
        ap._own_process(subj["agent"], ap._owner(user) if user else (owner if ap._auth_enabled() else ""))
    else:
        grant = crew_access.grant_of(await crew_runner._user_of(owner))
        if not grant.allowed or not grant.trusted:
            raise HTTPException(403, "Captain Swarm's coding lane is Crew on a local repository, open to admins only.")
    active = [r for r in await _cs_store().list_runs(owner) if r["status"] in ("baseline", "running", "awaiting_go")]
    if len(active) >= _MAX_ACTIVE_RUNS:
        raise HTTPException(409, f"You already have {len(active)} Captain Swarm run(s) going (limit {_MAX_ACTIVE_RUNS}).")


def _run_summary(r: dict) -> dict:
    champ, base = r.get("champion") or {}, r.get("baseline") or {}

    def mean(point: dict, part: str) -> float | None:
        s = (point.get(part) or {}).get("stats")
        return round(s["mean"], 4) if s else None
    return {"id": r["id"], "kind": (r.get("objective") or {}).get("kind", "code"),
            "title": r["title"], "status": r["status"], "stopped_reason": r["stopped_reason"],
            "started_by": r["started_by"], "round": (r.get("state") or {}).get("round", 0),
            "experiments": r["experiments"], "spent_usd": round(r["spent_usd"], 4), "spent_tokens": r["spent_tokens"],
            "baseline_dev": mean(base, "dev"), "champion_dev": mean(champ, "dev"),
            "baseline_holdout": mean(base, "holdout"), "champion_holdout": mean(champ, "holdout"),
            "improved": bool(champ.get("exp")), "created_at": r["created_at"], "started_at": r["started_at"],
            "finished_at": r["finished_at"], "budget": (r.get("objective") or {}).get("budget")}


async def _start(owner: str, objective: dict, *, status: str, started_by: str) -> dict:
    import uuid
    from captain_claw.flight_deck import captain_swarm_loop as loop_mod
    store = _cs_store()
    run_id = f"cs_{uuid.uuid4().hex[:12]}"
    run = await store.create_run(run_id=run_id, owner_id=owner, title=objective["title"], objective=objective,
                                 status=status, started_by=started_by)
    await store.append_event(run_id, "start", f"started by {started_by}"
                             + (": waiting for the owner's Go" if status == "awaiting_go" else ""))
    if status == "baseline":
        await loop_mod.kick()
    return run


@router.post("/runs")
async def start_run(body: RunBody, user: dict = Depends(get_current_user)) -> dict:
    from captain_claw.flight_deck.bat_routes import _owner_has_shell
    owner = _uid(user)
    objective = _objective_from(body, _owner_has_shell(user), user)
    await _can_run(owner, objective, user)
    return _run_summary(await _start(owner, objective, status="baseline", started_by="the owner"))


@router.get("/runs")
async def list_runs(user: dict = Depends(get_current_user)) -> dict:
    return {"runs": [_run_summary(r) for r in await _cs_store().list_runs(_uid(user))]}


async def _own_run(run_id: str, user: dict) -> dict:
    run = await _cs_store().get_run(run_id)
    if run is None or run["owner_id"] != _uid(user):
        raise HTTPException(404, "No such run.")
    return run


@router.get("/runs/{run_id}")
async def get_run(run_id: str, since: int = 0, user: dict = Depends(get_current_user)) -> dict:
    run = await _own_run(run_id, user)
    store = _cs_store()
    obj = run["objective"]
    extra: dict[str, Any] = {}
    if obj.get("kind") == "practice":
        from captain_claw.flight_deck import captain_swarm_rules as rules
        extra["rules"] = {"baseline": await rules.rules_of((run.get("baseline") or {}).get("sha") or ""),
                          "champion": await rules.rules_of((run.get("champion") or {}).get("sha") or "")}
    if obj.get("kind") == "outcome":
        champ = run.get("champion") or {}
        checks = await store.get_version(f"{champ.get('sha')}_c") if champ.get("sha") else None
        extra["outcome"] = {"best": (champ.get("dev") or {}).get("best"),
                            "checks": ((checks or {}).get("content") or {}).get("checks") or [],
                            "carried_out": (run.get("state") or {}).get("carried_out") or ""}
        extra["intents"] = {}
        for intent in await store.list_intents(run_id):
            extra["intents"].setdefault(intent["exp_id"], []).append(
                {k: intent[k] for k in ("kind", "tool", "what", "created_at")})
    return {**_run_summary(run), **extra,
            "objective": {k: obj.get(k) for k in ("kind", "title", "goal", "subject", "metric", "guardrails", "noise",
                                                   "lane", "budget", "stop")}
            | {"holdout": {"file": "(sealed)"}},
            "baseline": run["baseline"], "champion": run["champion"],
            "report": (run.get("state") or {}).get("report"),
            "ledger": await store.list_experiments(run_id),
            "events": await store.list_events(run_id, since=since)}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str, user: dict = Depends(get_current_user)) -> dict:
    from captain_claw.flight_deck import captain_swarm_loop as loop_mod
    from captain_claw.flight_deck.captain_swarm_store import TERMINAL_STATES
    run = await _own_run(run_id, user)
    if run["status"] in TERMINAL_STATES:
        raise HTTPException(409, "It has already stopped.")
    await loop_mod.cancel_run(_cs_store(), run_id)
    return _run_summary(await _cs_store().get_run(run_id))


@router.post("/runs/{run_id}/go")
async def go(run_id: str, user: dict = Depends(get_current_user)) -> dict:
    """The owner's Go for a run an agent started. The checks a run needs are made
    now, as the owner — the agent's say-so isn't enough for any of them."""
    from captain_claw.flight_deck import captain_swarm_loop as loop_mod
    from captain_claw.flight_deck import captain_swarm_objective as om
    from captain_claw.flight_deck.bat_routes import _owner_has_shell
    run = await _own_run(run_id, user)
    if run["status"] != "awaiting_go":
        raise HTTPException(409, "It isn't waiting for a Go.")
    try:
        if run["objective"].get("kind") == "practice":
            from captain_claw.flight_deck import captain_swarm_rules as rules
            rules.normalize(run["objective"])
        elif run["objective"].get("kind") == "outcome":
            from captain_claw.flight_deck import captain_swarm_outcome as outcome
            outcome.normalize(run["objective"])
        else:
            om.normalize(run["objective"], owner_has_shell=_owner_has_shell(user))
    except om.ObjectiveError as exc:
        raise HTTPException(400, str(exc)) from exc
    store = _cs_store()
    await store.set_status(run_id, "declined", "")  # out of the count while it's checked
    try:
        await _can_run(_uid(user), run["objective"], user)
    except HTTPException:
        await store.set_status(run_id, "awaiting_go", "")
        raise
    await store.update_run(run_id, status="baseline", stopped_reason="", finished_at=0.0)
    await store.append_event(run_id, "start", "the owner said Go")
    await loop_mod.kick()
    return _run_summary(await store.get_run(run_id))


@router.post("/runs/{run_id}/decline")
async def decline(run_id: str, user: dict = Depends(get_current_user)) -> dict:
    run = await _own_run(run_id, user)
    if run["status"] != "awaiting_go":
        raise HTTPException(409, "It isn't waiting for a Go.")
    await _cs_store().set_status(run_id, "declined", "the owner declined it")
    await _cs_store().append_event(run_id, "stop", "the owner declined it")
    return _run_summary(await _cs_store().get_run(run_id))


@router.post("/runs/{run_id}/carry-out")
async def carry_out(run_id: str, user: dict = Depends(get_current_user)) -> dict:
    """The winner, carried out once: one ordinary Bat run that re-checks the page,
    asks the owner if anything changed, and acts through Bat's gate — the owner's
    Go for spending, mail and accounts. Only on the owner's click, only once."""
    import uuid

    from captain_claw.flight_deck import bat_loop
    from captain_claw.flight_deck import captain_swarm_outcome as outcome
    from captain_claw.flight_deck.captain_swarm_store import TERMINAL_STATES
    run = await _own_run(run_id, user)
    if run["objective"].get("kind") != "outcome":
        raise HTTPException(400, "Only a run for one task's best result is carried out.")
    if run["status"] not in TERMINAL_STATES:
        raise HTTPException(409, "Stop the run first, or wait for it to finish.")
    state = run.get("state") or {}
    if state.get("carried_out"):
        raise HTTPException(409, f"It's being carried out already: Bat run {state['carried_out']}.")
    champ = run.get("champion") or {}
    best = (champ.get("dev") or {}).get("best")
    if not champ.get("exp") or not best:
        raise HTTPException(409, "No attempt found a result Captain could verify, so there's nothing to carry out.")
    bat = bat_loop.get_store()
    if bat is None:
        raise HTTPException(503, "Bat isn't running on this deck.")
    store = _cs_store()
    intents = await store.list_intents(run_id, champ["exp"])
    bat_id = f"bat_{uuid.uuid4().hex[:12]}"
    await bat.create_run(run_id=bat_id, owner_id=run["owner_id"], title=f"Carry out: {run['title'][:80]}",
                         task=outcome.carry_out_task(run, best, intents),
                         config={"source": "captain_swarm_carry_out", "steps": []}, origin={},
                         llm_usd_cap=float(run["objective"]["lane"].get("attempt_usd") or 2.0) * 2, status="planning")
    await bat.append_event(bat_id, "note", f"carrying out the winner of Captain Swarm run {run_id}")
    await store.update_run(run_id, state={**state, "carried_out": bat_id})
    await store.append_event(run_id, "carry", f"carrying out the winner, once: Bat run {bat_id}")
    try:
        await bat_loop.kick()
    except Exception:
        log.warning("captain swarm: Bat kick failed (it starts on the next tick)", exc_info=True)
    return {"bat_run": bat_id}


@router.post("/runs/{run_id}/teach")
async def teach(run_id: str, request: Request, user: dict = Depends(get_current_user)) -> dict:
    """The rules the best version added, taught to the agent as lessons — only
    when the owner says so, and only once the run has stopped."""
    from captain_claw.flight_deck import agent_onboarding
    from captain_claw.flight_deck import captain_swarm_rules as rules
    from captain_claw.flight_deck.captain_swarm_store import TERMINAL_STATES
    run = await _own_run(run_id, user)
    if run["objective"].get("kind") != "practice":
        raise HTTPException(400, "Only a practice run has rules to teach.")
    if run["status"] not in TERMINAL_STATES:
        raise HTTPException(409, "Stop the run first, or wait for it to finish.")
    champ = run.get("champion") or {}
    if not champ.get("exp"):
        raise HTTPException(409, "No version beat the starting point, so there's nothing to teach.")
    start = await rules.rules_of((run.get("baseline") or {}).get("sha") or "")
    new = [r for r in await rules.rules_of(champ["sha"]) if r not in start]
    got = await agent_onboarding.teach_rules(run["objective"]["subject"]["agent"], user, request, new,
                                             source=run["title"])
    await _cs_store().append_event(run_id, "teach", f"{got.get('added', 0)} rule(s) taught to "
                                                    f"{run['objective']['subject'].get('agent_name')}")
    return {"added": got.get("added", 0), "rules": new}


@router.post("/runs/{run_id}/pr")
async def open_pr(run_id: str, user: dict = Depends(get_current_user)) -> dict:
    """A pull request from the best commit, through Crew (its branch holds every
    kept change). Captain never merges it."""
    from captain_claw.flight_deck import crew_access, crew_runner
    from captain_claw.flight_deck.captain_swarm_store import TERMINAL_STATES
    run = await _own_run(run_id, user)
    if run["status"] not in TERMINAL_STATES:
        raise HTTPException(409, "Stop the run first, or wait for it to finish.")
    exp_id = (run.get("champion") or {}).get("exp")
    exp = await _cs_store().get_experiment(exp_id) if exp_id else None
    if exp is None or not exp.get("child_id"):
        raise HTTPException(409, "No attempt beat the starting point, so there's nothing to propose.")
    row = await crew_runner._user_of(_uid(user))
    task = await crew_runner.open_pr(row, crew_access.grant_of(row), exp["child_id"])
    await _cs_store().append_event(run_id, "pr", f"pull request opened: {(task.get('delivery') or {}).get('url', '')}")
    return {"delivery": task.get("delivery") or {}}


# ── the `captain_swarm` tool ─────────────────────────────────────────


class AgentReq(BaseModel):
    source_port: int = 0
    web_auth: str = ""
    owner_id: str = ""
    run_id: str = ""
    limit: int = Field(default=20, ge=1, le=100)


class AgentStartReq(AgentReq):
    template: str = "mrav"
    options: dict[str, Any] = Field(default_factory=dict)
    objective: dict[str, Any] | None = None
    agent_name: str = ""


def _agent_owner(body: AgentReq) -> str:
    from captain_claw.flight_deck.basna_routes import _AgentReq, _resolve_owner
    return _resolve_owner(_AgentReq(source_port=body.source_port, web_auth=body.web_auth, owner_id=body.owner_id)) \
        or "local"


@router.post("/agent/start")
async def agent_start(body: AgentStartReq) -> dict:
    """An agent's run never starts by itself: it waits for its owner's Go, with
    the objective, the scorer and the budget in front of them."""
    from captain_claw.flight_deck import crew_runner
    from captain_claw.flight_deck.auth import get_db
    from captain_claw.flight_deck.bat_routes import _owner_has_shell
    owner = _agent_owner(body)
    user = await crew_runner._user_of(owner)
    try:
        objective = _objective_from(RunBody(template=body.template, options=body.options, objective=body.objective),
                                    _owner_has_shell(user), user)
    except HTTPException as exc:
        return {"status": "rejected", "reason": exc.detail}
    run = await _start(owner, objective, status="awaiting_go", started_by=f"agent {body.agent_name or 'unknown'}")
    try:
        await get_db().add_notification(owner, "captain_swarm", "A Captain Swarm run waits for your Go",
                                        objective["title"][:300], "captain_swarm_run", run["id"])
    except Exception:
        log.warning("captain swarm: the Go bell failed", owner=owner, exc_info=True)
    return {"status": "awaiting_go", "run_id": run["id"], "title": run["title"],
            "note": "It starts when your owner says Go on the Captain Swarm page."}


@router.post("/agent/status")
async def agent_status(body: AgentReq) -> dict:
    run = await _cs_store().get_run(body.run_id)
    if run is None or run["owner_id"] != _agent_owner(body):
        return {"found": False}
    return {"found": True, **_run_summary(run)}


@router.post("/agent/get")
async def agent_get(body: AgentReq) -> dict:
    run = await _cs_store().get_run(body.run_id)
    if run is None or run["owner_id"] != _agent_owner(body):
        return {"found": False}
    exps = await _cs_store().list_experiments(body.run_id)
    return {"found": True, **_run_summary(run), "report": (run.get("state") or {}).get("report"),
            "ledger": [{"round": e["round"], "n": e["n"], "status": e["status"], "hypothesis": e["hypothesis"][:300],
                        "verdict": e["verdict_reason"][:300], "lesson": e["lesson"]} for e in exps[-body.limit:]]}


@router.post("/agent/list")
async def agent_list(body: AgentReq) -> dict:
    runs = await _cs_store().list_runs(_agent_owner(body), limit=body.limit)
    return {"runs": [_run_summary(r) for r in runs]}


@router.post("/agent/cancel")
async def agent_cancel(body: AgentReq) -> dict:
    from captain_claw.flight_deck import captain_swarm_loop as loop_mod
    from captain_claw.flight_deck.captain_swarm_store import TERMINAL_STATES
    run = await _cs_store().get_run(body.run_id)
    if run is None or run["owner_id"] != _agent_owner(body):
        return {"found": False}
    if run["status"] in TERMINAL_STATES:
        return {"found": True, "status": run["status"]}
    if run["status"] == "awaiting_go":
        await _cs_store().set_status(body.run_id, "declined", "withdrawn by the agent")
    else:
        await loop_mod.cancel_run(_cs_store(), body.run_id)
    return {"found": True, "status": (await _cs_store().get_run(body.run_id))["status"]}


class IntentReq(AgentReq):
    tag: str = Field("", max_length=80)
    kind: str = Field("intent", max_length=10)
    tool: str = Field("", max_length=80)
    what: str = Field("", max_length=500)
    args: dict[str, Any] = Field(default_factory=dict)


def _agent_slug(port: int) -> str:
    if not port:
        return ""
    try:
        from captain_claw.flight_deck import server as srv
        return next((slug for slug, e in srv._load_process_registry().items()
                     if int(e.get("web_port") or 0) == int(port)), "")
    except Exception:
        return ""


@router.post("/agent/intent")
async def agent_intent(body: IntentReq) -> dict:
    """A rehearsal worker's intent or commit point: kept with its attempt. Only
    the attempt's own owner's agents can add to it, and only while it runs."""
    from captain_claw.flight_deck import captain_swarm_rehearsal as rehearsal
    return await rehearsal.record(_cs_store(), _agent_owner(body), tag=body.tag, kind=body.kind, tool=body.tool,
                                  what=body.what, args=body.args, agent=_agent_slug(body.source_port))
