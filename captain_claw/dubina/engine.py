"""Dubina horizon engine — the substrate-agnostic escalation loop.

The engine knows nothing about LLMs, tests, or critics. It drives one principle:

    A weak model fails long tasks because per-step errors compound (p^n).
    Buy back the horizon with test-time compute — but only where a *verifier*
    can prove a step is good. Never let the model be its own ground truth.

It is parameterised by two plug-points:

* ``Generator`` — produces a ``Candidate`` for a step at a given tier, optionally
  given verifier ``feedback`` from a previous failed attempt.
* ``Verifier``  — returns a ``Verdict`` (passed / confidence / feedback) for a
  candidate. Coder track wires this to tests/typecheck (ground truth); reasoning
  track wires it to self-consistency + diverse-lens critics (statistical).

Per step, the engine climbs the **escalation ladder** (design §"escalation ladder"):

    1. base tier, single pass
    2.  └─ fail? → N-sample + vote                (parallel axis, same tier)
    3.      └─ still failing? → fix loop w/ feedback (sequential refine)
    4.          └─ still failing? → climb to the next tier (up to max_tier)

Everything is budget-bounded; on exhaustion the engine returns the best
verified-so-far with an explicit ``stopped_reason`` — never a silent truncation.

Pure orchestration: no I/O beyond the injected plug-points, fully unit-testable.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from captain_claw.logging import get_logger

log = get_logger(__name__)


# ── Core value types ─────────────────────────────────────────────────

@dataclass
class Tier:
    """One rung of the model ladder: a ``model.allowed`` id + its relative cost.

    ``cost`` is a unit-free weight (cheap draft ≈ 1, expensive frontier ≈ 8) used
    for budget accounting; Phase 4 replaces it with measured $/token.
    """

    id: str
    cost: float = 1.0


@dataclass
class Step:
    """A unit of work the engine drives to a verified result."""

    id: str
    prompt: str
    stakes: str = "normal"  # "normal" | "high" — high steps always run critics
    metadata: dict = field(default_factory=dict)


@dataclass
class Candidate:
    """One produced attempt at a step, at a particular tier."""

    step_id: str
    content: str
    tier: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Verdict:
    """A verifier's judgement of a candidate.

    ``confidence`` is the statistical signal (self-consistency agreement) the
    reasoning track leans on; for ground-truth verifiers it is 1.0/0.0.
    ``feedback`` is fed back into the next generation on the fix loop.
    """

    passed: bool
    confidence: float = 0.0
    feedback: str = ""


# Plug-points. Generator is a plain async callable; Verifier is a Protocol so the
# two track implementations can carry state (test runner, critic panel, etc.).
Generator = Callable[["Step", str, str, int], Awaitable["Candidate"]]
"""``async (step, tier_id, feedback, sample_index) -> Candidate``."""


class Verifier(Protocol):
    async def check(self, step: Step, candidate: Candidate) -> Verdict: ...


# Aggregator: collapse the N parallel samples of one attempt into a single winner.
# May be sync (coder: pick a passing solution) or async (reasoning: a self-
# consistency + conditional-critic judge that makes its own LLM calls).
Aggregator = Callable[
    [Sequence[Candidate], Sequence[Verdict]],
    "tuple[Candidate, Verdict] | Awaitable[tuple[Candidate, Verdict]]",
]


# ── Default tier ladders (illustrative costs; resolved from model.allowed) ────
# Cheap → expensive. The active slice is base_tier..max_tier (see resolve_ladder).
CODER_LADDER: list[Tier] = [
    Tier("gemini-flash", 1.0),   # draft
    Tier("gpt-5-mini", 3.0),     # light coding
    Tier("gpt-5.3-codex", 8.0),  # "extremely good for coding and deep reasoning"
]
REASON_LADDER: list[Tier] = [
    Tier("gemini-flash", 1.0),
    Tier("claude-sonnet", 3.0),
    Tier("claude-opus", 8.0),    # "the best model, ... but it's expensive"
]


def resolve_ladder(track_ladder: Sequence[Tier], base_tier: str, max_tier: str) -> list[Tier]:
    """Slice ``track_ladder`` to the active ``base_tier``..``max_tier`` window.

    ``base_tier == max_tier`` pins the run to a single user-chosen tier.
    Raises ``ValueError`` for unknown ids or an inverted window.
    """
    ids = [t.id for t in track_ladder]
    if base_tier not in ids:
        raise ValueError(f"base_tier {base_tier!r} not in ladder {ids}")
    if max_tier not in ids:
        raise ValueError(f"max_tier {max_tier!r} not in ladder {ids}")
    lo, hi = ids.index(base_tier), ids.index(max_tier)
    if lo > hi:
        raise ValueError(f"base_tier {base_tier!r} is above max_tier {max_tier!r}")
    return list(track_ladder[lo : hi + 1])


# ── Default aggregators ──────────────────────────────────────────────

def any_pass_aggregator(
    candidates: Sequence[Candidate], verdicts: Sequence[Verdict]
) -> tuple[Candidate, Verdict]:
    """Coder-friendly: one working solution is enough.

    Returns the first passing (candidate, verdict); else the highest-confidence one.
    """
    for cand, verdict in zip(candidates, verdicts):
        if verdict.passed:
            return cand, verdict
    best = max(range(len(verdicts)), key=lambda i: verdicts[i].confidence)
    return candidates[best], verdicts[best]


def majority_agreement_aggregator(
    candidates: Sequence[Candidate], verdicts: Sequence[Verdict]
) -> tuple[Candidate, Verdict]:
    """Reasoning-friendly: pass only if a majority of samples pass (self-consistency).

    The winning candidate is the highest-confidence passer (or highest overall if
    the majority did not pass). The returned verdict's ``passed`` reflects the vote.
    """
    passers = [i for i, v in enumerate(verdicts) if v.passed]
    won = len(passers) * 2 > len(verdicts)
    pool = passers if (won and passers) else range(len(verdicts))
    best = max(pool, key=lambda i: verdicts[i].confidence)
    verdict = verdicts[best]
    return candidates[best], Verdict(won, verdict.confidence, verdict.feedback)


# ── Budget ───────────────────────────────────────────────────────────

class Budget:
    """Cost meter for a run, in ``Tier.cost`` units (Phase 4 swaps in $/token)."""

    def __init__(self, total: float = math.inf):
        self.total = total
        self.spent = 0.0

    @property
    def remaining(self) -> float:
        return self.total - self.spent

    def can_afford(self, cost: float) -> bool:
        return self.spent + cost <= self.total

    def charge(self, cost: float) -> None:
        self.spent += cost


# ── Engine config & results ──────────────────────────────────────────

@dataclass
class EngineConfig:
    """Knobs for one run. ``ladder`` is the already-resolved base..max slice."""

    ladder: list[Tier]
    max_step_samples: int = 3   # N for the parallel vote on a step's first attempt
    max_fix_attempts: int = 2   # sequential feedback-driven retries per tier
    compute_budget: float = math.inf


@dataclass
class StepOutcome:
    step_id: str
    passed: bool
    candidate: Candidate | None
    verdict: Verdict | None
    tier_used: str | None
    rung_reached: int          # index into the active ladder (0 = base/cheapest)
    samples_used: int          # total generations spent on this step
    fix_attempts: int
    cost_spent: float
    stopped_reason: str = ""    # "" if resolved normally; "budget" if cut short


@dataclass
class RunResult:
    passed: bool
    steps: list[StepOutcome]
    cost_spent: float
    stopped_reason: str = ""    # "" | "budget" | "step_failed"


# A decompose hook turns a task into ordered steps (the sequential horizon).
Decompose = Callable[["Step"], Awaitable[list["Step"]]]


async def _identity_decompose(task: Step) -> list[Step]:
    return [task]


# ── The engine ───────────────────────────────────────────────────────

class HorizonEngine:
    """Drives a task to a verified result via the budget-bounded escalation ladder."""

    def __init__(
        self,
        config: EngineConfig,
        generator: Generator,
        verifier: Verifier,
        aggregator: Aggregator = any_pass_aggregator,
        decompose: Decompose = _identity_decompose,
        on_event: Callable[[dict], None] | None = None,
    ):
        if not config.ladder:
            raise ValueError("EngineConfig.ladder is empty")
        self.config = config
        self.generator = generator
        self.verifier = verifier
        self.aggregator = aggregator
        self.decompose = decompose
        self._on_event = on_event

    def _emit(self, **event) -> None:
        if self._on_event is not None:
            self._on_event(event)

    async def run(self, task: Step) -> RunResult:
        """Decompose, then drive each step in sequence — the horizon."""
        steps = await self.decompose(task)
        budget = Budget(self.config.compute_budget)
        outcomes: list[StepOutcome] = []

        for step in steps:
            outcome = await self._run_step(step, budget)
            outcomes.append(outcome)
            if outcome.stopped_reason == "budget":
                return RunResult(False, outcomes, budget.spent, "budget")
            if not outcome.passed:
                # A step that can't be verified even at max_tier hit the per-step
                # floor — continuing would just compound an unverified error.
                return RunResult(False, outcomes, budget.spent, "step_failed")

        return RunResult(True, outcomes, budget.spent, "")

    async def _run_step(self, step: Step, budget: Budget) -> StepOutcome:
        """Run the escalation ladder for one step.

        Within each tier the attempts run in design-rung order: a single pass, then
        (if it fails) an N-sample vote, then feedback-driven fix attempts — before
        climbing to the next tier.
        """
        start_spent = budget.spent
        samples_used = 0
        fix_attempts = 0
        best: Candidate | None = None
        best_verdict: Verdict | None = None

        for rung, tier in enumerate(self.config.ladder):
            feedback = best_verdict.feedback if best_verdict else ""

            # (n_samples, kind) attempts at this tier, cheapest first.
            plan: list[tuple[int, str]] = [(1, "single")]
            if self.config.max_step_samples > 1:
                plan.append((self.config.max_step_samples, "vote"))
            plan += [(1, "fix")] * self.config.max_fix_attempts

            for n, kind in plan:
                # The vote draws fresh samples; single/fix thread the feedback.
                fb = "" if kind == "vote" else feedback
                if math.isinf(budget.remaining):
                    affordable = n
                else:
                    affordable = min(n, int(budget.remaining // tier.cost))
                if affordable <= 0:
                    return self._stopped(
                        step, best, best_verdict, tier, rung,
                        samples_used, fix_attempts, budget.spent - start_spent,
                    )
                if kind == "fix":
                    fix_attempts += 1

                budget.charge(affordable * tier.cost)
                samples_used += affordable
                cands = await asyncio.gather(
                    *(self.generator(step, tier.id, fb, i) for i in range(affordable))
                )
                verdicts = await asyncio.gather(
                    *(self.verifier.check(step, c) for c in cands)
                )
                aggregated = self.aggregator(cands, verdicts)
                if inspect.isawaitable(aggregated):
                    aggregated = await aggregated
                cand, verdict = aggregated
                if best_verdict is None or verdict.confidence >= best_verdict.confidence:
                    best, best_verdict = cand, verdict

                self._emit(
                    step=step.id, tier=tier.id, rung=rung, kind=kind,
                    samples=affordable, passed=verdict.passed,
                    confidence=verdict.confidence,
                )
                if verdict.passed:
                    return StepOutcome(
                        step.id, True, cand, verdict, tier.id, rung,
                        samples_used, fix_attempts, budget.spent - start_spent, "",
                    )
                feedback = verdict.feedback  # carry into the next fix / tier

        # Ladder exhausted without a passing verdict.
        last_rung = len(self.config.ladder) - 1
        return StepOutcome(
            step.id, False, best, best_verdict,
            self.config.ladder[last_rung].id, last_rung,
            samples_used, fix_attempts, budget.spent - start_spent, "",
        )

    def _stopped(
        self, step, best, best_verdict, tier, rung, samples, fixes, cost
    ) -> StepOutcome:
        return StepOutcome(
            step.id, False, best, best_verdict, tier.id, rung,
            samples, fixes, cost, "budget",
        )
