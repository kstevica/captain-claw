"""Tests for the Dubina horizon engine (Phase 0).

Cover the escalation ladder transitions, the parallel sample+vote axis, the
sequential fix loop, budget exhaustion (best-so-far, no silent truncation), and
ladder resolution.
"""

from __future__ import annotations

import pytest

from captain_claw.dubina import (
    CODER_LADDER,
    Candidate,
    EngineConfig,
    HorizonEngine,
    Step,
    Verdict,
    any_pass_aggregator,
    majority_agreement_aggregator,
    resolve_ladder,
)

# ── Stub plug-points ─────────────────────────────────────────────────

def make_generator():
    """Generator that records calls and echoes tier/feedback/sample into content."""
    calls: list[dict] = []

    async def gen(step: Step, tier: str, feedback: str, sample: int) -> Candidate:
        calls.append({"tier": tier, "feedback": feedback, "sample": sample})
        return Candidate(step.id, content=f"{tier}|{feedback}|{sample}", tier=tier)

    gen.calls = calls  # type: ignore[attr-defined]
    return gen


class TierVerifier:
    """Passes once the candidate's tier is in ``pass_tiers``."""

    def __init__(self, pass_tiers: set[str]):
        self.pass_tiers = pass_tiers

    async def check(self, step: Step, candidate: Candidate) -> Verdict:
        ok = candidate.tier in self.pass_tiers
        return Verdict(passed=ok, confidence=1.0 if ok else 0.1,
                       feedback="" if ok else "try harder")


class SampleVerifier:
    """Passes only the sample whose index == ``winning_sample``."""

    def __init__(self, winning_sample: int):
        self.winning_sample = winning_sample

    async def check(self, step: Step, candidate: Candidate) -> Verdict:
        idx = int(candidate.content.rsplit("|", 1)[1])
        ok = idx == self.winning_sample
        return Verdict(passed=ok, confidence=0.9 if ok else 0.2,
                       feedback="" if ok else "wrong sample")


class FeedbackVerifier:
    """Passes only once feedback has been threaded in (simulates a fix)."""

    async def check(self, step: Step, candidate: Candidate) -> Verdict:
        had_feedback = candidate.content.split("|", 2)[1] != ""
        return Verdict(passed=had_feedback, confidence=1.0 if had_feedback else 0.1,
                       feedback="" if had_feedback else "needs fixing")


def cfg(**kw) -> EngineConfig:
    base = dict(ladder=list(CODER_LADDER), max_step_samples=3, max_fix_attempts=2)
    base.update(kw)
    return EngineConfig(**base)


def step() -> Step:
    return Step(id="s1", prompt="do the thing")


# ── Tests ────────────────────────────────────────────────────────────

async def test_passes_at_base_tier_no_climb():
    gen = make_generator()
    engine = HorizonEngine(cfg(), gen, TierVerifier({"gemini-flash"}))
    result = await engine.run(step())

    assert result.passed
    out = result.steps[0]
    assert out.tier_used == "gemini-flash"
    assert out.rung_reached == 0
    assert out.samples_used == 1            # passed on the first single attempt
    assert {c["tier"] for c in gen.calls} == {"gemini-flash"}  # never climbed


async def test_climbs_to_top_tier_when_cheap_fails():
    gen = make_generator()
    engine = HorizonEngine(cfg(), gen, TierVerifier({"gpt-5.3-codex"}))
    result = await engine.run(step())

    assert result.passed
    out = result.steps[0]
    assert out.tier_used == "gpt-5.3-codex"
    assert out.rung_reached == 2
    # Saw every rung on the way up.
    assert {c["tier"] for c in gen.calls} == {"gemini-flash", "gpt-5-mini", "gpt-5.3-codex"}


async def test_parallel_vote_rescues_without_climbing():
    # Only sample #2 passes; max_step_samples=3 means the first attempt covers it.
    gen = make_generator()
    engine = HorizonEngine(cfg(), gen, SampleVerifier(winning_sample=2),
                           aggregator=any_pass_aggregator)
    result = await engine.run(step())

    assert result.passed
    out = result.steps[0]
    assert out.rung_reached == 0            # stayed on base tier
    assert out.samples_used == 4            # 1 single pass + 3-sample vote


async def test_fix_loop_threads_feedback():
    gen = make_generator()
    # max_step_samples=1 so the win can only come from a feedback-bearing retry.
    engine = HorizonEngine(cfg(max_step_samples=1), gen, FeedbackVerifier())
    result = await engine.run(step())

    assert result.passed
    out = result.steps[0]
    assert out.rung_reached == 0
    assert out.fix_attempts >= 1            # needed at least one fix pass
    assert out.candidate is not None and out.candidate.content.split("|", 2)[1] != ""


async def test_budget_exhaustion_returns_best_so_far():
    gen = make_generator()
    # Nothing passes; budget only affords a couple of cheap generations.
    engine = HorizonEngine(
        cfg(compute_budget=2.5), gen, TierVerifier(set()),
    )
    result = await engine.run(step())

    assert not result.passed
    assert result.stopped_reason == "budget"
    out = result.steps[0]
    assert out.stopped_reason == "budget"
    assert out.candidate is not None        # best-so-far, not a silent drop
    assert result.cost_spent <= 2.5         # never overspent


async def test_step_failure_stops_the_horizon():
    gen = make_generator()
    engine = HorizonEngine(cfg(), gen, TierVerifier(set()))  # never passes
    result = await engine.run(step())

    assert not result.passed
    assert result.stopped_reason == "step_failed"
    assert result.steps[0].rung_reached == 2  # exhausted the whole ladder


async def test_multi_step_runs_in_sequence():
    gen = make_generator()

    async def decompose(task: Step) -> list[Step]:
        return [Step(id=f"s{i}", prompt=f"part {i}") for i in range(3)]

    engine = HorizonEngine(cfg(), gen, TierVerifier({"gemini-flash"}),
                           decompose=decompose)
    result = await engine.run(step())

    assert result.passed
    assert [o.step_id for o in result.steps] == ["s0", "s1", "s2"]


def test_resolve_ladder_slices_and_validates():
    full = list(CODER_LADDER)
    assert [t.id for t in resolve_ladder(full, "gemini-flash", "gpt-5-mini")] == [
        "gemini-flash", "gpt-5-mini",
    ]
    # Pin to a single tier.
    assert [t.id for t in resolve_ladder(full, "gpt-5-mini", "gpt-5-mini")] == ["gpt-5-mini"]
    with pytest.raises(ValueError):
        resolve_ladder(full, "gpt-5.3-codex", "gemini-flash")  # inverted window
    with pytest.raises(ValueError):
        resolve_ladder(full, "nope", "gpt-5-mini")             # unknown id


def test_majority_agreement_aggregator():
    cands = [Candidate("s", c, "t") for c in ("a", "b", "c")]
    # 2 of 3 pass -> majority -> winning verdict passes.
    won_c, won_v = majority_agreement_aggregator(
        cands, [Verdict(True, 0.6), Verdict(True, 0.9), Verdict(False, 0.3)]
    )
    assert won_v.passed and won_c.content == "b"
    # 1 of 3 -> no majority -> fails.
    _, lost_v = majority_agreement_aggregator(
        cands, [Verdict(True, 0.9), Verdict(False, 0.3), Verdict(False, 0.2)]
    )
    assert not lost_v.passed
