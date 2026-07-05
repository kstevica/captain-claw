"""Horizon worker — drive one Basna/Vatra worker through the Dubina engine.

Basna is one-shot per agent: one prompt → one reply. This wraps a single worker
in the **Frontier Horizon** depth axes so each agent "thinks way longer" — the
test-time-compute scaffolding that simulates a stronger model:

    single pass → N-sample self-consistency vote → diverse-lens critics →
    feedback-driven fix loop

Phase 1 deliberately runs on a **single tier** (no model escalation): the depth
comes from sampling + critics + fix, not from climbing the ladder. Escalation
(re-spawn the archetype at a higher tier) is a later phase.

The hard part Dubina warns about — self-consistency needs *independent* rollouts —
is handled by spawning a **pool of N fresh archetype instances** and round-robining
the engine's samples across them, rather than re-prompting one agent (one chat =
shared context, correlated errors). Critics run on a **different** model than the
worker (``critic_provider``) so the worker never grades itself.

The engine, generator, judge and verifier are reused verbatim from
``captain_claw.dubina``; the only new thing here is the agent pool + the wiring.
``spawn``/``send``/``stop`` are injected so the whole path unit-tests with stubs.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from captain_claw.dubina import (
    DEFAULT_CRITIC_MODES,
    Budget,
    CriticVerdict,
    EngineConfig,
    HorizonEngine,
    ReasoningJudge,
    ReasonVerifier,
    Step,
    Tier,
    load_critic_modes,
    make_reasoning_generator,
)
from captain_claw.flight_deck.dubina_agents import (
    DispatchProvider,
    _real_send,
)
from captain_claw.llm import LLMProvider
from captain_claw.logging import get_logger

log = get_logger(__name__)


async def _default_stop(slug: str) -> None:
    """Minimal pool teardown. ``_do_stop_process`` is **sync** (Dubina's ``_real_stop``
    awaits it, which is a latent bug); Basna passes its own full-teardown stop."""
    from captain_claw.flight_deck.server import _do_stop_process
    _do_stop_process(slug)

# ``spawn(name_suffix) -> (port, token, slug)`` — the caller binds archetype/tier/
# config; the helper only needs a way to get one more fresh worker instance.
SpawnPoolMember = Callable[[str], Awaitable["tuple[int, str, str]"]]
# ``send(port, token, prompt, timeout, fleet_instructions, agent_name, on_action, on_usage) -> reply``
Send = Callable[..., Awaitable[str]]
# ``stop(slug) -> None``
Stop = Callable[[str], Awaitable[None]]


@dataclass
class HorizonConfig:
    """Per-run depth knobs (ride in the Basna/Vatra session ``config`` JSON)."""

    samples: int = 3            # N for the self-consistency vote (pool size)
    fix_attempts: int = 1       # feedback-driven retries after a failed vote
    critics: list[str] = field(default_factory=lambda: list(DEFAULT_CRITIC_MODES))
    stakes: str = "normal"      # "high" forces critics even when agreement is high
    agreement_threshold: float = 0.6
    critic_cost: float = 1.0
    compute_budget: float = 0.0  # <= 0 → unbounded (Tier.cost units)
    critic_tier: str = "reason"  # Library tier the critics run on (≠ the worker's)
    worker: bool = True          # Lever A: per-worker self-consistency depth
    close: bool = False          # Lever B: adversarially verify + revise the final

    @classmethod
    def from_dict(cls, d: dict | None) -> HorizonConfig:
        d = d or {}
        critics = d.get("critics")
        if not isinstance(critics, list) or not critics:
            critics = list(DEFAULT_CRITIC_MODES)
        return cls(
            samples=max(1, int(d.get("samples") or 3)),
            fix_attempts=max(0, int(d.get("fix_attempts") or 1)),
            critics=[str(c) for c in critics],
            stakes=str(d.get("stakes") or "normal"),
            agreement_threshold=float(d.get("agreement_threshold") or 0.6),
            critic_cost=float(d.get("critic_cost") or 1.0),
            compute_budget=float(d.get("compute_budget") or 0.0),
            critic_tier=str(d.get("critic_tier") or "reason"),
            worker=bool(d.get("worker", True)),
            close=bool(d.get("close", False)),
        )


def _pool_provider(
    port: int, token: str, send: Send, *, timeout: float,
    fleet_instructions: str, agent_name: str, on_action, on_usage,
) -> LLMProvider:
    """A provider whose ``complete`` dispatches to one specific pooled agent.

    Mirrors ``dubina_agents.make_agent_factory`` but pins a fixed (port, token) and
    carries this pool member's tagged callbacks, so the engine's N concurrent
    samples each land on a distinct agent.
    """
    async def dispatch(prompt: str) -> str:
        return await send(
            port, token, prompt, timeout,
            fleet_instructions=fleet_instructions, agent_name=agent_name,
            on_action=on_action, on_usage=on_usage,
        )

    return DispatchProvider(dispatch)


async def run_worker_horizon(
    *,
    spawn: SpawnPoolMember,
    tier: str,
    prompt: str,
    cfg: HorizonConfig,
    critic_provider: LLMProvider | None = None,
    fleet_instructions: str = "",
    agent_name: str = "",
    on_action: Callable[[dict], None] | None = None,
    on_usage: Callable[[int, int, int], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_spawn: Callable[[list[str]], None] | None = None,
    send: Send = _real_send,
    stop: Stop = _default_stop,
    timeout: float = 600.0,
) -> dict:
    """Drive a pool of ``cfg.samples`` archetype instances through the engine.

    Spawns the pool up front (independent rollouts), runs the reasoning track over
    it on a single-tier ladder, disposes the pool, and returns the best verified
    answer shaped like a Basna dispatch result::

        {output, ok, passed, confidence, actions, latency_ms,
         tier_used, rung_reached, samples_used, cost_spent, slugs}

    Never raises for an agent failure — returns ``ok=False`` with an ``error``.
    """
    started = time.monotonic()
    n = max(1, cfg.samples)
    actions: list[dict] = []
    slugs: list[str] = []
    # pool[i] = (port, token, slug)
    pool: list[tuple[int, str, str]] = []

    def _mk_on_action(sample: int):
        def _sink(act: dict) -> None:
            # Tag which rollout produced this action; bound the kept history.
            tagged = {**act, "sample": sample}
            if len(actions) < 400:
                actions.append(tagged)
            if on_action is not None:
                on_action(tagged)
        return _sink

    try:
        # 1) Spawn the pool of independent rollouts (best-effort; use what comes up).
        suffix = uuid.uuid4().hex[:6]
        spawn_out = await asyncio.gather(
            *[spawn(f"{suffix}-{i}") for i in range(n)], return_exceptions=True,
        )
        for i, item in enumerate(spawn_out):
            if isinstance(item, Exception):
                log.warning("horizon pool spawn failed", index=i, error=str(item))
                continue
            port, token, slug = item
            pool.append((port, token, slug))
            if slug:
                slugs.append(slug)
        if on_spawn is not None:
            on_spawn(list(slugs))
        if not pool:
            return {
                "output": "", "ok": False, "passed": False, "confidence": 0.0,
                "actions": actions, "latency_ms": int((time.monotonic() - started) * 1000),
                "tier_used": tier, "rung_reached": 0, "samples_used": 0,
                "cost_spent": 0.0, "slugs": slugs, "error": "no pool agent spawned",
            }

        # 2) Round-robin the engine's samples across distinct pool members.
        providers = [
            _pool_provider(
                port, token, send, timeout=timeout,
                fleet_instructions=fleet_instructions, agent_name=agent_name,
                on_action=_mk_on_action(i), on_usage=on_usage,
            )
            for i, (port, token, _slug) in enumerate(pool)
        ]
        cyc = itertools.cycle(providers)

        def provider_for_tier(_tier: str) -> LLMProvider:
            # Called synchronously at the start of each generate() coroutine, before
            # its first await — so concurrent vote samples advance the cycle in order.
            return next(cyc)

        budget = Budget(math.inf if cfg.compute_budget <= 0 else float(cfg.compute_budget))
        config = EngineConfig(
            ladder=[Tier(tier, 1.0)],  # single rung: depth, not escalation (Phase 1)
            max_step_samples=n,
            max_fix_attempts=cfg.fix_attempts,
            compute_budget=budget.total,
        )
        critics = load_critic_modes(critic_provider, cfg.critics) if critic_provider else []
        judge = ReasoningJudge(
            critics, agreement_threshold=cfg.agreement_threshold,
            budget=budget, critic_cost=cfg.critic_cost,
        )
        engine = HorizonEngine(
            config, make_reasoning_generator(provider_for_tier), ReasonVerifier(),
            aggregator=judge, on_event=on_event,
        )
        result = await engine.run(Step(id="task", prompt=prompt, stakes=cfg.stakes), budget=budget)

        out = result.steps[-1] if result.steps else None
        return {
            "output": (out.candidate.content if out and out.candidate else ""),
            "ok": bool(out and out.candidate and (out.candidate.content or "").strip()),
            "passed": result.passed,
            "confidence": (out.verdict.confidence if out and out.verdict else 0.0),
            "actions": actions,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "tier_used": (out.tier_used if out else tier),
            "rung_reached": (out.rung_reached if out else 0),
            "samples_used": (out.samples_used if out else 0),
            "cost_spent": result.cost_spent,
            "slugs": slugs,
            "stopped_reason": result.stopped_reason,
        }
    except Exception as e:  # noqa: BLE001 — a worker error must not crash the run
        log.exception("horizon worker failed", agent=agent_name)
        return {
            "output": "", "ok": False, "passed": False, "confidence": 0.0,
            "actions": actions, "latency_ms": int((time.monotonic() - started) * 1000),
            "tier_used": tier, "rung_reached": 0, "samples_used": 0,
            "cost_spent": 0.0, "slugs": slugs, "error": str(e),
        }
    finally:
        # 3) Dispose the pool — best-effort, mirrors Basna's hard teardown.
        for _port, _token, slug in pool:
            try:
                await stop(slug)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                log.warning("horizon pool dispose failed", slug=slug)


# ── Lever B: the closer (verify the final answer, revise once if refuted) ─────

_REVISE_SYSTEM = (
    "You are revising a draft answer that adversarial reviewers found flawed. "
    "Produce a corrected, complete final answer that fixes every valid objection "
    "while preserving what was already correct. Do not mention the review or that "
    "this is a revision — output only the improved answer."
)


def _triage_feedback(reasons: list[str]) -> str:
    """R3: distil raw critic objections into a deduped, ordered fix checklist.

    The closer otherwise pipes the objections in as one ``a | b | c`` blob; a
    numbered checklist of distinct issues is far more actionable for the revision
    (this is Code's triage shape applied to critic findings). Deterministic and
    free — no model call. Also makes a cleaner findings ledger for the caller."""
    seen: set[str] = set()
    items: list[str] = []
    for r in reasons:
        r = (r or "").strip()
        if not r:
            continue
        key = r.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        items.append(r)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return "Fix each of these distinct objections:\n" + "\n".join(
        f"{i}. {r}" for i, r in enumerate(items, 1))


async def _revise(question: str, answer: str, feedback: str, provider: LLMProvider) -> str:
    from captain_claw.llm import Message
    user = (f"## Task\n{question}\n\n## Draft answer\n{answer}\n\n"
            f"## Reviewer objections\n{feedback}\n\n## Your revised final answer")
    resp = await provider.complete(
        [Message(role="system", content=_REVISE_SYSTEM), Message(role="user", content=user)])
    return (resp.content or "").strip()


async def _with_heartbeat(awaitable, *, on_event, phase: str, interval: float = 15.0):
    """Await ``awaitable`` while emitting a ``heartbeat`` event every ``interval``
    seconds — so a slow model call (critics, revise) shows it's alive, not hung."""
    if on_event is None:
        return await awaitable
    done = asyncio.Event()

    async def beat():
        elapsed = 0.0
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=interval)
            except TimeoutError:
                elapsed += interval
                on_event({"stage": "heartbeat", "phase": phase, "elapsed": round(elapsed)})

    hb = asyncio.create_task(beat())
    try:
        return await awaitable
    finally:
        done.set()
        hb.cancel()


async def run_horizon_closer(
    *,
    question: str,
    answer: str,
    critic_provider: LLMProvider | None,
    revise_provider: LLMProvider | None = None,
    critics: list[str] | None = None,
    critic_timeout: float = 120.0,
    revise_timeout: float = 300.0,
    on_event: Callable[[dict], None] | None = None,
    triage_findings: bool = False,
) -> dict:
    """Adversarially verify a final answer; revise once if a majority refute it.

    The back-edge Basna/Vatra lack: a diverse-lens critic panel (run on a model
    *different* from whoever produced the answer) reviews the merged/assembled
    deliverable; if a strict majority refute it, one feedback-driven revision is
    produced. Returns::

        {answer, revised: bool, survived: int, total: int, feedback: str}

    No critic provider (or empty answer) → a no-op pass-through (answer unchanged).
    Every model call is **time-bounded** (a hung model can't stall the run), and a
    revision that **collapses** (empty, or a fragment far shorter than a substantial
    original — e.g. a reasoning model returning only a reasoning tail) is **rejected**
    so good content is never silently replaced by garbage.
    """
    modes = critics if critics else list(DEFAULT_CRITIC_MODES)
    if critic_provider is None or not (answer or "").strip():
        return {"answer": answer, "revised": False, "survived": 0, "total": 0, "feedback": ""}
    panel = load_critic_modes(critic_provider, modes)
    if not panel:
        return {"answer": answer, "revised": False, "survived": 0, "total": 0, "feedback": ""}
    total = len(panel)
    if on_event is not None:
        on_event({"stage": "verify_start", "total": total})

    async def _run_critic(i: int, mode: str, critic):
        try:
            r = await asyncio.wait_for(critic(question, answer), critic_timeout)
        except Exception:  # noqa: BLE001 — timeout/failure → abstain (don't refute on noise)
            r = CriticVerdict(refuted=False, reason="")
        if on_event is not None:
            on_event({"stage": "critic", "index": i, "total": total,
                      "mode": mode, "refuted": r.refuted})
        return r

    results = await _with_heartbeat(
        asyncio.gather(
            *(_run_critic(i, modes[i] if i < len(modes) else "", c) for i, c in enumerate(panel))),
        on_event=on_event, phase="verify")
    survived = sum(1 for r in results if not r.refuted)
    if on_event is not None:
        on_event({"stage": "verify", "survived": survived, "total": total})
    # Strict majority must survive (matches ReasoningJudge's critic gate).
    if survived * 2 > total:
        return {"answer": answer, "revised": False, "survived": survived,
                "total": total, "feedback": ""}
    refuted_reasons = [r.reason for r in results if r.refuted]
    # R3 (opt-in): an ordered, deduped checklist revises more precisely than a blob.
    feedback = (_triage_feedback(refuted_reasons) if triage_findings
                else " | ".join(refuted_reasons))
    try:
        revised = await asyncio.wait_for(
            _with_heartbeat(
                _revise(question, answer, feedback, revise_provider or critic_provider),
                on_event=on_event, phase="revise"),
            revise_timeout)
    except Exception:  # noqa: BLE001 — timeout/failure → keep the original answer
        revised = ""
    # Reject a collapsed revision: empty, or a fragment of a substantial original
    # (guards against a reasoning model surfacing only a short reasoning tail).
    collapsed = not revised.strip() or (len(answer) > 800 and len(revised) < 0.5 * len(answer))
    if collapsed:
        if on_event is not None:
            on_event({"stage": "revise_rejected", "survived": survived, "total": total})
        return {"answer": answer, "revised": False, "survived": survived,
                "total": total, "feedback": feedback}
    if on_event is not None:
        on_event({"stage": "revise", "survived": survived, "total": total})
    return {"answer": revised, "revised": True,
            "survived": survived, "total": total, "feedback": feedback}
