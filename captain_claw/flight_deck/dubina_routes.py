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

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from captain_claw.dubina import (
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
    shell_command_runner,
)
from captain_claw.dubina.coder import (
    SOLUTION_PATH_KEY,
    TEST_COMMAND_KEY,
    WORKSPACE_KEY,
    Workspace,
)
from captain_claw.dubina.reasoning import DEFAULT_CRITIC_MODES
from captain_claw.flight_deck.auth import get_current_user, get_db
from captain_claw.flight_deck.basna_routes import _load_owner_tiers
from captain_claw.flight_deck.dubina_agents import (
    ArchetypeRunner,
    make_agent_factory,
    resolve_agent_port_token,
)
from captain_claw.flight_deck.dubina_store import TRACKS, DubinaStore
from captain_claw.llm import create_provider
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


# ── Tier / ladder helpers (Library tiers) ────────────────────────────

# Flight Deck Library tiers, cheap → expensive. ``longctx`` is a special-purpose
# tier, not part of the default escalation ladders.
TIER_ORDER = ("fast", "balanced", "reason", "longctx")
LIBRARY_CODER_LADDER = ["fast", "balanced", "reason"]
LIBRARY_REASON_LADDER = ["fast", "balanced", "reason"]


async def _resolve_tiers(db, user_id: str) -> dict[str, dict]:
    """The user's configured Library tier map (name -> {provider, model, ...})."""
    tiers_map, _env = await _load_owner_tiers(db, user_id)
    return tiers_map or {}


def _library_provider_factory(tiers_map: dict[str, dict]):
    """A ``ProviderForTier`` backed by the user's Library tier configs."""
    def factory(tier: str):
        t = tiers_map.get(tier)
        if not t:
            raise ValueError(f"tier {tier!r} is not configured in your Library")
        return create_provider(
            provider=t.get("provider", "anthropic"), model=t.get("model", ""),
            base_url=t.get("base_url") or None, api_key=t.get("api_key") or None,
            temperature=0.7, max_tokens=int(t.get("output_ctx") or 0) or 8000,
        )
    return factory


def build_ladder(
    base_tier: str, max_tier: str, tiers: list[str] | None,
    *, default_ladder: list[str], allowed: set[str],
) -> list[Tier]:
    """Resolve the active escalation ladder (cheap→expensive) from the request.

    Priority: an explicit ordered ``tiers`` list; else the track's default ladder
    sliced ``base_tier``..``max_tier``; else a two-rung ``[base, max]`` ladder. All
    ids are validated against the user's configured Library tier names (``allowed``).
    Costs escalate by position.
    """
    if tiers:
        ids = list(dict.fromkeys(t for t in tiers if t))
    elif base_tier in default_ladder and max_tier in default_ladder:
        lo, hi = default_ladder.index(base_tier), default_ladder.index(max_tier)
        if lo > hi:
            raise HTTPException(400, f"base_tier {base_tier!r} is above max_tier {max_tier!r}")
        ids = default_ladder[lo:hi + 1]
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


class IntentRequest(BaseModel):
    """Run an arbitrary intent via an archetype or a live agent (statistical verifier)."""
    task: str
    target: str                          # "agent:<id>" | "archetype:<id>"
    base_tier: str
    max_tier: str
    tiers: list[str] | None = None
    compute_budget: float = 0.0
    max_step_samples: int = 3
    max_fix_attempts: int = 1
    stakes: str = "normal"
    critic_modes: list[str] = Field(default_factory=lambda: list(DEFAULT_CRITIC_MODES))
    agreement_threshold: float = 0.6
    critic_cost: float = 1.0


# ── Execution (awaitable; injectable for tests) ──────────────────────

async def execute_coder(
    store: DubinaStore, run_id: str, req: CoderRequest,
    *, provider_factory=None, runner=None, tiers_map: dict | None = None,
):
    try:
        tiers_map = tiers_map or {}
        allowed = set(tiers_map) or set(TIER_ORDER)
        ladder = build_ladder(req.base_tier, req.max_tier, req.tiers,
                              default_ladder=LIBRARY_CODER_LADDER, allowed=allowed)
        provider_factory = provider_factory or _library_provider_factory(tiers_map)
        runner = runner or shell_command_runner
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
    store: DubinaStore, run_id: str, req: ReasonRequest,
    *, provider_factory=None, tiers_map: dict | None = None,
):
    try:
        tiers_map = tiers_map or {}
        allowed = set(tiers_map) or set(TIER_ORDER)
        ladder = build_ladder(req.base_tier, req.max_tier, req.tiers,
                              default_ladder=LIBRARY_REASON_LADDER, allowed=allowed)
        provider_factory = provider_factory or _library_provider_factory(tiers_map)
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


async def execute_intent(
    store: DubinaStore, run_id: str, req: IntentRequest,
    *, provider_factory, critic_provider=None, dispose=None, allowed: set[str] | None = None,
):
    """Run an intent through the reasoning engine over a chosen run-target.

    ``provider_factory`` dispatches to the target (agent/archetype). Critics use a
    separate real-model ``critic_provider`` so the target isn't grading itself.
    """
    try:
        allowed = allowed or set(TIER_ORDER)
        ladder = build_ladder(req.base_tier, req.max_tier, req.tiers,
                              default_ladder=LIBRARY_REASON_LADDER, allowed=allowed)
        config = EngineConfig(
            ladder=ladder, max_step_samples=req.max_step_samples,
            max_fix_attempts=req.max_fix_attempts,
            compute_budget=_budget_value(req.compute_budget),
        )
        budget = Budget(config.compute_budget)
        critics = load_critic_modes(critic_provider, req.critic_modes) if critic_provider else []
        judge = ReasoningJudge(critics, agreement_threshold=req.agreement_threshold,
                               budget=budget, critic_cost=req.critic_cost)
        events: list[dict] = []
        engine = HorizonEngine(
            config, make_reasoning_generator(provider_factory), ReasonVerifier(),
            aggregator=judge, on_event=events.append,
        )
        step = Step(id="task", prompt=req.task, stakes=req.stakes)
        result = await engine.run(step, budget=budget)
        await _persist(store, "intent", run_id, events, result, _reason_summary(result))
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("dubina intent run failed", run_id=run_id)
        await store.finish_run("intent", run_id, status="error", passed=False,
                               stopped_reason="error", cost_spent=0.0, error=str(e))
    finally:
        if dispose is not None:
            try:
                await dispose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                log.warning("dubina intent dispose failed", run_id=run_id)


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
async def list_tiers(db=Depends(get_db), user: dict = Depends(get_current_user)):
    """The user's configured Library tiers + default per-track ladders, for the UI."""
    tiers_map = await _resolve_tiers(db, user["id"])
    ordered = ([n for n in TIER_ORDER if n in tiers_map]
               + [n for n in tiers_map if n not in TIER_ORDER])
    tiers = [
        {"id": n, "provider": tiers_map[n].get("provider", ""),
         "model": tiers_map[n].get("model", ""),
         "description": f"{tiers_map[n].get('provider', '')}/{tiers_map[n].get('model', '')}",
         "reasoning_level": ""}
        for n in ordered
    ]
    return {
        "tiers": tiers,
        "default_ladders": {"coder": LIBRARY_CODER_LADDER, "reason": LIBRARY_REASON_LADDER},
    }


@router.post("/coder")
async def start_coder(req: CoderRequest, db=Depends(get_db), user: dict = Depends(get_current_user)):
    store = _store()
    tiers_map = await _resolve_tiers(db, user["id"])
    run_id = await store.create_run(
        "coder", user["id"], req.task, req.base_tier, req.max_tier,
        _budget_value(req.compute_budget),
        config=req.model_dump(exclude={"task"}),
    )
    asyncio.create_task(execute_coder(store, run_id, req, tiers_map=tiers_map))
    return {"run_id": run_id, "track": "coder", "status": "running"}


@router.post("/reason")
async def start_reason(req: ReasonRequest, db=Depends(get_db), user: dict = Depends(get_current_user)):
    store = _store()
    tiers_map = await _resolve_tiers(db, user["id"])
    run_id = await store.create_run(
        "reason", user["id"], req.task, req.base_tier, req.max_tier,
        _budget_value(req.compute_budget),
        config=req.model_dump(exclude={"task"}),
    )
    asyncio.create_task(execute_reason(store, run_id, req, tiers_map=tiers_map))
    return {"run_id": run_id, "track": "reason", "status": "running"}


async def _find_archetype(db, user_id: str, arch_id: str) -> dict:
    from captain_claw.flight_deck.archetypes import merged_archetypes
    arch = next((a for a in await merged_archetypes(db, user_id) if a.get("id") == arch_id), None)
    if arch is None:
        raise HTTPException(404, f"archetype {arch_id!r} not found")
    return arch


@router.post("/intent")
async def start_intent(req: IntentRequest, request: Request,
                       db=Depends(get_db), user: dict = Depends(get_current_user)):
    """Run an intent via a live agent or a spawned archetype, verified statistically.

    Runs inline (so the Request stays valid for archetype spawn) and returns the
    finished run; archetype agents are disposed afterward.
    """
    store = _store()
    tiers_map = await _resolve_tiers(db, user["id"])
    allowed = set(tiers_map) or set(TIER_ORDER)
    # Critics use a real Library model (base tier) — never the target judging itself.
    try:
        critic_provider = _library_provider_factory(tiers_map)(req.base_tier)
    except Exception:  # noqa: BLE001 — no base-tier model configured → agreement-only
        critic_provider = None

    kind, _, ident = req.target.partition(":")
    dispose = None
    if kind == "agent":
        port, token = resolve_agent_port_token(ident)
        provider_factory = make_agent_factory(port, token, agent_name=ident)
    elif kind == "archetype":
        runner = ArchetypeRunner(await _find_archetype(db, user["id"], ident),
                                 request, user, tiers_map)
        provider_factory = runner.provider_for_tier()
        dispose = runner.dispose
    else:
        raise HTTPException(400, "target must be 'agent:<id>' or 'archetype:<id>'")

    run_id = await store.create_run(
        "intent", user["id"], req.task, req.base_tier, req.max_tier,
        _budget_value(req.compute_budget), config=req.model_dump(exclude={"task"}),
    )
    await execute_intent(store, run_id, req, provider_factory=provider_factory,
                         critic_provider=critic_provider, dispose=dispose, allowed=allowed)
    return await store.get_run("intent", run_id)


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
