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
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
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
from captain_claw.flight_deck.basna_routes import (
    _guess_mime,
    _is_texty,
    _load_owner_tiers,
    _safe_name,
)
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


# ── Live execution log (in-memory, polled by the UI; mirrors Basna's _PROGRESS) ──
# Keyed by run_id → {"events": [{i, ts, stage, message, ...}], "active": bool}. The
# engine's on_event fires per ladder attempt; we also bracket the run with start/
# done/error lines so the UI shows a live narration, not just the final steps table.
_PROGRESS: dict[str, dict] = {}

# Live background run tasks, keyed by run_id, so a run can be cancelled mid-flight.
_RUNNING: dict[str, asyncio.Task] = {}


def _track_task(run_id: str, coro) -> asyncio.Task:
    """Schedule a run's background task and register it so it can be stopped."""
    task = asyncio.create_task(coro)
    _RUNNING[run_id] = task
    task.add_done_callback(
        lambda t: _RUNNING.pop(run_id, None) if _RUNNING.get(run_id) is t else None)
    return task


def _progress(run_id: str, stage: str, message: str, **extra) -> None:
    p = _PROGRESS.get(run_id)
    if p is not None:
        p["events"].append({"i": len(p["events"]), "ts": time.time(),
                            "stage": stage, "message": message, **extra})


def _progress_done(run_id: str) -> None:
    p = _PROGRESS.get(run_id)
    if p is not None:
        p["active"] = False


def _attach_progress(run_id: str, events: list[dict], ladder: list[Tier]):
    """Start a run's live log and return an on_event sink for the engine.

    The returned callback both keeps the raw event (for ``_persist`` → DB steps) and
    appends a human-readable line to the live progress buffer.
    """
    _PROGRESS[run_id] = {"events": [], "active": True}
    _progress(run_id, "start", "Ladder: " + " → ".join(t.id for t in ladder))

    def on_event(e: dict) -> None:
        events.append(e)
        conf = float(e.get("confidence", 0.0) or 0.0)
        mark = "✓ passed" if e.get("passed") else "✗ escalate"
        _progress(
            run_id, "attempt",
            f"{e.get('tier', '?')} · {e.get('kind', '?')} · "
            f"{e.get('samples', 0)} sample(s) → conf {conf:.2f} · {mark}",
            tier=e.get("tier"), rung=e.get("rung"), kind=e.get("kind"),
            samples=e.get("samples"), passed=bool(e.get("passed")), confidence=conf,
        )

    return on_event


def _agent_callbacks(run_id: str, label: str):
    """``(on_action, on_usage)`` sinks that stream a target agent's live steps.

    Mirrors Basna / the agent desktop: the agent's monitor events arrive as
    ``{tool, detail}`` (tool calls + ``narration``), and ``turn_usage`` fires after
    **each internal LLM call** with the turn's running token counts — surfaced here
    as an ``llm`` line so you can see when (and how heavily) the model is called.
    """
    def on_action(act: dict) -> None:
        tool = str(act.get("tool") or "tool")
        detail = str(act.get("detail") or "")
        if tool == "narration":
            _progress(run_id, "narration", detail or "(thinking…)", agent=label, tool=tool)
        else:
            _progress(run_id, "action", detail or tool, agent=label, tool=tool, detail=detail)

    def on_usage(pt: int, ct: int, tt: int) -> None:
        _progress(run_id, "llm", f"{pt:,}→{ct:,} tok · {tt:,} total", agent=label,
                  prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)

    return on_action, on_usage


# ── Generated-file capture (target agents that answer by writing files) ───────

def _dubina_files_dir(run_id: str) -> Path:
    from captain_claw.flight_deck.server import DATA_DIR
    d = DATA_DIR / "dubina_files" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _capture_run_files(run_id: str, slugs: list[str]) -> list[dict]:
    """Copy any files the run's archetype agents wrote into the run's files dir.

    Mirrors Basna: a target agent may answer by writing a document. We snapshot each
    spawned agent's workspace before dispose tears it down, returning ``{name, mime,
    size}`` metadata persisted on the run and served by the download endpoint.
    """
    from captain_claw.flight_deck.server import DATA_DIR
    out: list[dict] = []
    seen: set[str] = set()
    dest: Path | None = None
    for slug in slugs:
        ws = DATA_DIR / slug / "data" / "workspace"
        if not ws.is_dir():
            continue
        for p in sorted(ws.rglob("*")):
            if not p.is_file():
                continue
            name = _safe_name(p.name)
            if name in seen:
                name = _safe_name(f"{slug}__{p.name}")
            if name in seen:
                continue
            try:
                if dest is None:
                    dest = _dubina_files_dir(run_id)
                shutil.copy2(p, dest / name)
            except OSError as e:  # noqa: PERF203
                log.warning("dubina file capture failed", file=p.name, error=str(e))
                continue
            seen.add(name)
            out.append({"name": name, "mime": _guess_mime(name), "size": p.stat().st_size})
    return out


def _text_fallback(run_id: str, files: list[dict], limit: int = 20000) -> str:
    """First texty generated file's content — used as the answer when the agent
    replied with an empty chat message (it answered by writing a document)."""
    for f in files:
        if _is_texty(f["name"], f.get("mime", "")):
            try:
                return (_dubina_files_dir(run_id) / f["name"]).read_text(errors="replace")[:limit]
            except OSError:
                continue
    return ""


def _finalize_summary(run_id: str, summary: dict, capture_slugs) -> dict:
    """Capture generated files into the summary; fall back to a file for the answer."""
    files = _capture_run_files(run_id, capture_slugs() if capture_slugs else [])
    if files:
        summary["files"] = files
        if not summary.get("answer") and not summary.get("code"):
            text = _text_fallback(run_id, files)
            if text:
                summary["answer"] = text
    return summary


# ── Tier / ladder helpers (Library tiers) ────────────────────────────

# Flight Deck Library tiers, cheap → expensive. ``longctx``, ``coding`` and
# ``vision`` are special-purpose tiers, not part of the default escalation ladders.
TIER_ORDER = ("fast", "balanced", "reason", "longctx", "coding", "vision")
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
    target: str = ""                     # "" → Library tier model; else agent:/archetype:


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
    target: str = ""                     # "" → Library tier model; else agent:/archetype:


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
    *, provider_factory=None, runner=None, tiers_map: dict | None = None, dispose=None,
    capture_slugs=None,
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
            on_event=_attach_progress(run_id, events, ladder),
        )
        step = Step(id="task", prompt=req.task, metadata={
            WORKSPACE_KEY: req.workspace, TEST_COMMAND_KEY: req.test_command,
            SOLUTION_PATH_KEY: req.solution_path, "test_path": req.test_path,
        })
        result = await engine.run(step, budget=Budget(config.compute_budget))
        _progress(run_id, "done", f"{result.stopped_reason or 'passed'} · {result.cost_spent:.0f}u")
        _progress_done(run_id)
        summary = _finalize_summary(run_id, _coder_summary(result), capture_slugs)
        await _persist(store, "coder", run_id, events, result, summary)
        return result
    except Exception as e:  # noqa: BLE001 — background task: record, don't crash the loop
        log.exception("dubina coder run failed", run_id=run_id)
        _progress(run_id, "error", str(e))
        _progress_done(run_id)
        await store.finish_run("coder", run_id, status="error", passed=False,
                               stopped_reason="error", cost_spent=0.0, error=str(e))
    finally:
        if dispose is not None:
            try:
                await dispose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                log.warning("dubina coder dispose failed", run_id=run_id)


# Sentinel: critics default to the generator's base-tier provider (the cheap-substrate
# path). A target run passes a real ``critic_provider`` (or None for agreement-only) so
# the target never grades itself.
_CRITICS_DEFAULT = object()


async def execute_reason(
    store: DubinaStore, run_id: str, req: ReasonRequest,
    *, provider_factory=None, tiers_map: dict | None = None,
    critic_provider=_CRITICS_DEFAULT, dispose=None, capture_slugs=None,
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
        # With a run-target generator, critics use a real Library model instead (or
        # none → agreement-only) so the target never judges its own answer.
        if critic_provider is _CRITICS_DEFAULT:
            critics = load_critic_modes(provider_factory(ladder[0].id), req.critic_modes)
        elif critic_provider is None:
            critics = []
        else:
            critics = load_critic_modes(critic_provider, req.critic_modes)
        judge = ReasoningJudge(
            critics, agreement_threshold=req.agreement_threshold,
            budget=budget, critic_cost=req.critic_cost,
        )
        events: list[dict] = []
        engine = HorizonEngine(
            config, make_reasoning_generator(provider_factory), ReasonVerifier(),
            aggregator=judge, on_event=_attach_progress(run_id, events, ladder),
        )
        step = Step(id="task", prompt=req.task, stakes=req.stakes)
        result = await engine.run(step, budget=budget)
        _progress(run_id, "done", f"{result.stopped_reason or 'passed'} · {result.cost_spent:.0f}u")
        _progress_done(run_id)
        summary = _finalize_summary(run_id, _reason_summary(result), capture_slugs)
        await _persist(store, "reason", run_id, events, result, summary)
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("dubina reason run failed", run_id=run_id)
        _progress(run_id, "error", str(e))
        _progress_done(run_id)
        await store.finish_run("reason", run_id, status="error", passed=False,
                               stopped_reason="error", cost_spent=0.0, error=str(e))
    finally:
        if dispose is not None:
            try:
                await dispose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                log.warning("dubina reason dispose failed", run_id=run_id)


async def execute_intent(
    store: DubinaStore, run_id: str, req: IntentRequest,
    *, provider_factory, critic_provider=None, dispose=None, allowed: set[str] | None = None,
    capture_slugs=None,
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
            aggregator=judge, on_event=_attach_progress(run_id, events, ladder),
        )
        step = Step(id="task", prompt=req.task, stakes=req.stakes)
        result = await engine.run(step, budget=budget)
        _progress(run_id, "done", f"{result.stopped_reason or 'passed'} · {result.cost_spent:.0f}u")
        _progress_done(run_id)
        summary = _finalize_summary(run_id, _reason_summary(result), capture_slugs)
        await _persist(store, "intent", run_id, events, result, summary)
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("dubina intent run failed", run_id=run_id)
        _progress(run_id, "error", str(e))
        _progress_done(run_id)
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
    pf = dispose = capture = None
    if req.target:
        # Generator dispatches to an agent/archetype (stub Request → background-safe).
        pf, dispose, capture = await _resolve_target(req.target, _stub_request(user["id"]),
                                                     db, user, tiers_map, run_id=run_id)
    _track_task(run_id, execute_coder(store, run_id, req, provider_factory=pf,
                                      tiers_map=tiers_map, dispose=dispose, capture_slugs=capture))
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
    pf = dispose = capture = None
    critic_provider = _CRITICS_DEFAULT  # no target → critics default to the base-tier model
    if req.target:
        # Generator dispatches to an agent/archetype (stub Request → background-safe).
        pf, dispose, capture = await _resolve_target(req.target, _stub_request(user["id"]),
                                                     db, user, tiers_map, run_id=run_id)
        # Critics use a real Library base-tier model — never the target judging itself.
        try:
            critic_provider = _library_provider_factory(tiers_map)(req.base_tier)
        except Exception:  # noqa: BLE001 — no base-tier model → agreement-only
            critic_provider = None
    _track_task(run_id, execute_reason(store, run_id, req, provider_factory=pf, tiers_map=tiers_map,
                                       critic_provider=critic_provider, dispose=dispose,
                                       capture_slugs=capture))
    return {"run_id": run_id, "track": "reason", "status": "running"}


async def _find_archetype(db, user_id: str, arch_id: str) -> dict:
    from captain_claw.flight_deck.archetypes import merged_archetypes
    arch = next((a for a in await merged_archetypes(db, user_id) if a.get("id") == arch_id), None)
    if arch is None:
        raise HTTPException(404, f"archetype {arch_id!r} not found")
    return arch


def _stub_request(user_id: str):
    """A lightweight stand-in Request for background spawns (mirrors Basna's pattern).

    ``spawn_process`` only reads ``request.state.user_id`` to stamp the owner, so a
    SimpleNamespace suffices — and unlike a real Request it stays valid after the HTTP
    response returns, letting target runs execute in the background and stream live.
    """
    import types
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=user_id))


async def _resolve_target(target: str, request, db, user: dict, tiers_map: dict,
                          run_id: str | None = None):
    """Resolve a target to ``(provider_factory, dispose, capture_slugs)``.

    The factory is the run's **generator** — it dispatches each step to the chosen
    live agent (tier ignored) or a freshly spawned archetype (re-spawned per tier).
    ``request`` is a :func:`_stub_request` so archetype spawn works in the background.
    When ``run_id`` is given, the agent's live steps stream into that run's log.
    ``capture_slugs`` lists the spawned-agent workspaces to snapshot for generated
    files — only archetypes (ephemeral, fresh workspace); live agents return none.
    """
    kind, _, ident = target.partition(":")
    if kind == "agent":
        port, token = resolve_agent_port_token(ident)
        oa, ou = _agent_callbacks(run_id, ident) if run_id else (None, None)
        pf = make_agent_factory(port, token, agent_name=ident, on_action=oa, on_usage=ou)
        return pf, None, None
    if kind == "archetype":
        arch = await _find_archetype(db, user["id"], ident)
        oa, ou = _agent_callbacks(run_id, arch.get("role") or ident) if run_id else (None, None)
        runner = ArchetypeRunner(arch, request, user, tiers_map, on_action=oa, on_usage=ou)
        return runner.provider_for_tier(), runner.dispose, runner.agent_slugs
    raise HTTPException(400, "target must be 'agent:<id>' or 'archetype:<id>'")


@router.post("/intent")
async def start_intent(req: IntentRequest, db=Depends(get_db), user: dict = Depends(get_current_user)):
    """Run an intent via a live agent or a spawned archetype, verified statistically.

    Runs in the background (a stub Request keeps archetype spawn valid) so the UI can
    poll the live execution log; archetype agents are disposed when the run ends.
    """
    store = _store()
    tiers_map = await _resolve_tiers(db, user["id"])
    allowed = set(tiers_map) or set(TIER_ORDER)
    # Critics use a real Library model (base tier) — never the target judging itself.
    try:
        critic_provider = _library_provider_factory(tiers_map)(req.base_tier)
    except Exception:  # noqa: BLE001 — no base-tier model configured → agreement-only
        critic_provider = None

    run_id = await store.create_run(
        "intent", user["id"], req.task, req.base_tier, req.max_tier,
        _budget_value(req.compute_budget), config=req.model_dump(exclude={"task"}),
    )
    provider_factory, dispose, capture = await _resolve_target(
        req.target, _stub_request(user["id"]), db, user, tiers_map, run_id=run_id)
    _track_task(run_id, execute_intent(store, run_id, req, provider_factory=provider_factory,
                                       critic_provider=critic_provider, dispose=dispose,
                                       allowed=allowed, capture_slugs=capture))
    return {"run_id": run_id, "track": "intent", "status": "running"}


@router.get("/runs/{track}/{run_id}/progress")
async def get_progress(track: str, run_id: str, user: dict = Depends(get_current_user)):
    """Live execution log for a run, polled by the UI while it runs (and after)."""
    if track not in TRACKS:
        raise HTTPException(404, f"unknown track {track!r}")
    return _PROGRESS.get(run_id) or {"events": [], "active": False}


@router.get("/runs/{track}/{run_id}/files/{name}")
async def download_file(track: str, run_id: str, name: str,
                        user: dict = Depends(get_current_user)):
    """Stream a file generated by the run's target agent (view or download)."""
    if track not in TRACKS:
        raise HTTPException(404, f"unknown track {track!r}")
    run = await _store().get_run(track, run_id)
    if run is None or run.get("user_id") not in (user["id"], ""):
        raise HTTPException(404, "run not found")
    safe = _safe_name(name)
    path = _dubina_files_dir(run_id) / safe
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=safe, media_type=_guess_mime(safe))


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


@router.post("/runs/{track}/{run_id}/stop")
async def stop_run(track: str, run_id: str, user: dict = Depends(get_current_user)):
    """Cancel a running run. Cancellation unwinds the task's ``finally`` (disposing
    any spawned archetype agents); we then mark the run stopped."""
    if track not in TRACKS:
        raise HTTPException(404, f"unknown track {track!r}")
    store = _store()
    run = await store.get_run(track, run_id)
    if run is None or run.get("user_id") not in (user["id"], ""):
        raise HTTPException(404, "run not found")

    task = _RUNNING.pop(run_id, None)
    if task is not None and not task.done():
        task.cancel()
    if run.get("status") == "running":
        await store.finish_run(track, run_id, status="stopped", passed=False,
                               stopped_reason="stopped", cost_spent=run.get("cost_spent") or 0.0)
    _progress(run_id, "done", "stopped by user")
    _progress_done(run_id)
    return {"ok": True, "status": "stopped"}


@router.post("/agents/cleanup")
async def cleanup_agents(user: dict = Depends(get_current_user)):
    """Stop any lingering Dubina-spawned archetype agents (``dubina-*``).

    A safety net for orphans left behind when a run crashed before its dispose ran.
    Scoped to the caller's own agents.
    """
    from captain_claw.flight_deck.server import _do_stop_process, _load_process_registry
    reg = _load_process_registry()
    stopped: list[str] = []
    for slug, entry in list(reg.items()):
        name = entry.get("name", "") or ""
        if not (slug.startswith("dubina-") or name.startswith("dubina-")):
            continue
        if entry.get("owner") and entry["owner"] != user["id"]:
            continue
        try:
            await _do_stop_process(slug)
            stopped.append(slug)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            log.warning("dubina agent cleanup failed", slug=slug)
    return {"stopped": stopped, "count": len(stopped)}
