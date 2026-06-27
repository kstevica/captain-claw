"""Dubina (Frontier Horizon) REST endpoints for Flight Deck.

Two sections over one engine (design §"one engine, two verifiers"):

* **Coder**     — ground-truth verifier (runs the project's tests).
* **Reasoning** — statistical verifier (self-consistency + diverse-lens critics).

Both expose the same controls: a user-selected ``base_tier`` (the rung the run
starts on), a ``max_tier`` ceiling, and a ``compute_budget``. Runs execute in the
background; the live view polls ``GET /runs/{track}/{id}`` for the per-attempt ladder
log. The store is injected via :func:`set_store` from the server lifespan (the same
pattern as the flow engine).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from captain_claw.config import get_config
from captain_claw.dubina import (
    CODER_LADDER,
    REASON_LADDER,
    Budget,
    CoderVerifier,
    EngineConfig,
    HorizonEngine,
    ReasoningJudge,
    ReasonVerifier,
    Step,
    Tier,
    any_pass_aggregator,
    ensure_tests,
    load_critic_modes,
    make_coder_generator,
    make_reasoning_generator,
    provider_for_tier_from_config,
    resolve_ladder,
    shell_command_runner,
)
from captain_claw.dubina.coder import (
    SOLUTION_PATH_KEY,
    TEST_COMMAND_KEY,
    WORKSPACE_KEY,
    Workspace,
)
from captain_claw.dubina.reasoning import DEFAULT_CRITIC_MODES
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.dubina_store import TRACKS, DubinaStore
from captain_claw.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/fd/dubina", tags=["dubina"])

# Injected from the server lifespan (mirrors flow_router.set_engine).
_STORE: DubinaStore | None = None


def set_store(store: DubinaStore) -> None:
    global _STORE
    _STORE = store


def _store() -> DubinaStore:
    if _STORE is None:
        raise HTTPException(503, "Dubina store not initialized")
    return _STORE


# ── Tier / ladder helpers ────────────────────────────────────────────

def _allowed_ids() -> dict[str, Any]:
    return {m.id: m for m in get_config().model.allowed}


def build_ladder(track: str, base_tier: str, max_tier: str, tiers: list[str] | None) -> list[Tier]:
    """Resolve the active escalation ladder (cheap→expensive) from the request.

    Priority: an explicit ordered ``tiers`` list; else the track's default ladder
    sliced ``base_tier``..``max_tier``; else a two-rung ``[base, max]`` ladder. All
    ids are validated against ``config.model.allowed``. Costs escalate by position.
    """
    allowed = _allowed_ids()
    if tiers:
        ids = list(dict.fromkeys(t for t in tiers if t))
    else:
        default = CODER_LADDER if track == "coder" else REASON_LADDER
        default_ids = [t.id for t in default]
        if base_tier in default_ids and max_tier in default_ids:
            sliced = resolve_ladder(default, base_tier, max_tier)
            ids = [t.id for t in sliced]
        else:
            ids = [base_tier] if base_tier == max_tier else [base_tier, max_tier]
    unknown = [i for i in ids if i not in allowed]
    if unknown:
        raise HTTPException(400, f"unknown tier(s): {unknown}")
    if not ids:
        raise HTTPException(400, "empty ladder")
    return [Tier(tid, float(2**i)) for i, tid in enumerate(ids)]


def _budget_value(compute_budget: float) -> float:
    return math.inf if not compute_budget or compute_budget <= 0 else float(compute_budget)


def _final_status(passed: bool, stopped_reason: str) -> str:
    if passed:
        return "passed"
    return {"budget": "budget", "step_failed": "failed"}.get(stopped_reason, "failed")


# ── Request models ───────────────────────────────────────────────────

class CoderRequest(BaseModel):
    task: str
    workspace: str                       # base project dir (the verifier runs here)
    test_command: str = "pytest -q"
    solution_path: str = "solution.py"
    test_path: str = ""
    spec: str = ""                       # if set + no tests, synthesize them first
    base_tier: str
    max_tier: str
    tiers: list[str] | None = None
    compute_budget: float = 0.0          # <=0 → unbounded
    max_step_samples: int = 3
    max_fix_attempts: int = 2


class ReasonRequest(BaseModel):
    task: str
    base_tier: str
    max_tier: str
    tiers: list[str] | None = None
    compute_budget: float = 0.0
    max_step_samples: int = 3
    max_fix_attempts: int = 1
    stakes: str = "normal"               # "high" forces critics even on agreement
    critic_modes: list[str] = Field(default_factory=lambda: list(DEFAULT_CRITIC_MODES))
    agreement_threshold: float = 0.6
    critic_cost: float = 1.0


# ── Execution (awaitable; injectable for tests) ──────────────────────

async def execute_coder(
    store: DubinaStore, run_id: str, req: CoderRequest,
    *, provider_factory=None, runner=None,
):
    try:
        provider_factory = provider_factory or provider_for_tier_from_config()
        runner = runner or shell_command_runner
        ladder = build_ladder("coder", req.base_tier, req.max_tier, req.tiers)
        config = EngineConfig(
            ladder=ladder, max_step_samples=req.max_step_samples,
            max_fix_attempts=req.max_fix_attempts,
            compute_budget=_budget_value(req.compute_budget),
        )
        if req.spec and req.test_path:
            await ensure_tests(req.spec, provider_factory(ladder[0].id), req.workspace, req.test_path)

        events: list[dict] = []
        engine = HorizonEngine(
            config, make_coder_generator(provider_factory, Workspace(req.workspace)),
            CoderVerifier(runner), aggregator=any_pass_aggregator,
            on_event=events.append,
        )
        step = Step(id="task", prompt=req.task, metadata={
            WORKSPACE_KEY: req.workspace, TEST_COMMAND_KEY: req.test_command,
            SOLUTION_PATH_KEY: req.solution_path, "test_path": req.test_path,
        })
        result = await engine.run(step, budget=Budget(config.compute_budget))
        await _persist(store, "coder", run_id, events, result, _coder_summary(result))
        return result
    except Exception as e:  # noqa: BLE001 — background task: record, don't crash the loop
        log.exception("dubina coder run failed", run_id=run_id)
        await store.finish_run("coder", run_id, status="error", passed=False,
                               stopped_reason="error", cost_spent=0.0, error=str(e))


async def execute_reason(
    store: DubinaStore, run_id: str, req: ReasonRequest, *, provider_factory=None,
):
    try:
        provider_factory = provider_factory or provider_for_tier_from_config()
        ladder = build_ladder("reason", req.base_tier, req.max_tier, req.tiers)
        config = EngineConfig(
            ladder=ladder, max_step_samples=req.max_step_samples,
            max_fix_attempts=req.max_fix_attempts,
            compute_budget=_budget_value(req.compute_budget),
        )
        budget = Budget(config.compute_budget)
        # Critics run on the base (cheapest) tier by default — cheap substrate first.
        critics = load_critic_modes(provider_factory(ladder[0].id), req.critic_modes)
        judge = ReasoningJudge(
            critics, agreement_threshold=req.agreement_threshold,
            budget=budget, critic_cost=req.critic_cost,
        )
        events: list[dict] = []
        engine = HorizonEngine(
            config, make_reasoning_generator(provider_factory), ReasonVerifier(),
            aggregator=judge, on_event=events.append,
        )
        step = Step(id="task", prompt=req.task, stakes=req.stakes)
        result = await engine.run(step, budget=budget)
        await _persist(store, "reason", run_id, events, result, _reason_summary(result))
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("dubina reason run failed", run_id=run_id)
        await store.finish_run("reason", run_id, status="error", passed=False,
                               stopped_reason="error", cost_spent=0.0, error=str(e))


async def _persist(store, track, run_id, events, result, summary):
    for i, e in enumerate(events):
        await store.append_step(run_id, track, i, e)
    await store.finish_run(
        track, run_id, status=_final_status(result.passed, result.stopped_reason),
        passed=result.passed, stopped_reason=result.stopped_reason,
        cost_spent=result.cost_spent, result=summary,
    )


def _coder_summary(result) -> dict:
    out = result.steps[-1] if result.steps else None
    return {
        "tier_used": out.tier_used if out else None,
        "rung_reached": out.rung_reached if out else None,
        "code": (out.candidate.content if out and out.candidate else ""),
    }


def _reason_summary(result) -> dict:
    out = result.steps[-1] if result.steps else None
    return {
        "tier_used": out.tier_used if out else None,
        "rung_reached": out.rung_reached if out else None,
        "confidence": (out.verdict.confidence if out and out.verdict else 0.0),
        "answer": (out.candidate.content if out and out.candidate else ""),
    }


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/tiers")
async def list_tiers(user: dict = Depends(get_current_user)):
    """Available model tiers + the default per-track ladders, for the UI selectors."""
    tiers = [
        {"id": m.id, "provider": m.provider, "model": m.model,
         "description": m.description, "reasoning_level": m.reasoning_level}
        for m in get_config().model.allowed if m.model_type == "llm"
    ]
    return {
        "tiers": tiers,
        "default_ladders": {
            "coder": [t.id for t in CODER_LADDER],
            "reason": [t.id for t in REASON_LADDER],
        },
    }


@router.post("/coder")
async def start_coder(req: CoderRequest, user: dict = Depends(get_current_user)):
    store = _store()
    run_id = await store.create_run(
        "coder", user["id"], req.task, req.base_tier, req.max_tier,
        _budget_value(req.compute_budget),
        config=req.model_dump(exclude={"task"}),
    )
    asyncio.create_task(execute_coder(store, run_id, req))
    return {"run_id": run_id, "track": "coder", "status": "running"}


@router.post("/reason")
async def start_reason(req: ReasonRequest, user: dict = Depends(get_current_user)):
    store = _store()
    run_id = await store.create_run(
        "reason", user["id"], req.task, req.base_tier, req.max_tier,
        _budget_value(req.compute_budget),
        config=req.model_dump(exclude={"task"}),
    )
    asyncio.create_task(execute_reason(store, run_id, req))
    return {"run_id": run_id, "track": "reason", "status": "running"}


@router.get("/runs/{track}/{run_id}")
async def get_run(track: str, run_id: str, user: dict = Depends(get_current_user)):
    if track not in TRACKS:
        raise HTTPException(404, f"unknown track {track!r}")
    run = await _store().get_run(track, run_id)
    if run is None or run.get("user_id") not in (user["id"], ""):
        raise HTTPException(404, "run not found")
    return run


@router.get("/runs/{track}")
async def list_runs(track: str, limit: int = 50, user: dict = Depends(get_current_user)):
    if track not in TRACKS:
        raise HTTPException(404, f"unknown track {track!r}")
    return {"runs": await _store().list_runs(track, user["id"], limit)}
