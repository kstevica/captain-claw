"""Tests for the Dubina reasoning track (Phase 2).

Cover answer extraction, agreement scoring, the cheap per-candidate verifier, the
mode critic parsing, the two-stage judge (agreement gate vs. conditional critics),
and an end-to-end engine run elevated by self-consistency.
"""

from __future__ import annotations

import pytest

from captain_claw.dubina import (
    CODER_LADDER,
    Candidate,
    EngineConfig,
    HorizonEngine,
    ReasoningJudge,
    ReasonVerifier,
    Step,
    agreement_score,
    extract_answer,
    make_mode_critic,
    make_reasoning_generator,
    resolve_ladder,
)
from captain_claw.dubina.reasoning import (
    PROMPT_KEY,
    STAKES_KEY,
    CriticVerdict,
)
from captain_claw.llm import LLMResponse

# ── Stubs ────────────────────────────────────────────────────────────

class StubProvider:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls += 1
        return LLMResponse(content=self.content, finish_reason="stop")


def critic_const(refuted: bool, reason="r"):
    async def critic(question, answer):
        return CriticVerdict(refuted=refuted, reason=reason)
    return critic


def critic_tripwire():
    """A critic that fails the test if it is ever called (proves it was skipped)."""
    async def critic(question, answer):
        raise AssertionError("critics should not run when agreement is high")
    return critic


def reason_cands(answers, stakes="normal", confidences=None):
    confidences = confidences or [0.8] * len(answers)
    cands = [
        Candidate("q", f"reasoning...\nAnswer: {a}", "gemini-flash",
                  {PROMPT_KEY: "what is 2+2?", STAKES_KEY: stakes})
        for a in answers
    ]
    from captain_claw.dubina import Verdict
    verdicts = [Verdict(True, c) for c in confidences]
    return cands, verdicts


# ── extract_answer / agreement ───────────────────────────────────────

def test_extract_answer_prefers_marker():
    assert extract_answer("lots of work\nFinal Answer: 42.") == "42"
    assert extract_answer("blah\nanswer:  Paris ") == "paris"


def test_extract_answer_falls_back_to_last_line():
    assert extract_answer("step 1\nstep 2\nthe result is seven") == "the result is seven"


def test_agreement_score():
    assert agreement_score(["42", "42", "7"]) == ("42", pytest.approx(2 / 3))
    assert agreement_score(["a", "a", "a"]) == ("a", 1.0)


# ── ReasonVerifier ───────────────────────────────────────────────────

async def test_reason_verifier_flags_empty_and_marks_answers():
    v = ReasonVerifier()
    empty = await v.check(Step("q", "p"), Candidate("q", "   ", "t"))
    assert not empty.passed

    marked = await v.check(Step("q", "p"), Candidate("q", "work\nAnswer: 42", "t"))
    bare = await v.check(Step("q", "p"), Candidate("q", "just some text", "t"))
    assert marked.confidence > bare.confidence   # marker -> higher pick priority


# ── Mode critic ──────────────────────────────────────────────────────

async def test_mode_critic_parses_refuted_and_sound():
    refuting = make_mode_critic(StubProvider("REFUTED: off by one"), "phrygian")
    sound = make_mode_critic(StubProvider("SOUND: checks out"), "aeolian")

    assert (await refuting("q", "a")).refuted is True
    assert (await sound("q", "a")).refuted is False


# ── Judge: agreement gate ────────────────────────────────────────────

async def test_judge_passes_on_high_agreement_without_critics():
    judge = ReasoningJudge([critic_tripwire()], agreement_threshold=0.6)
    cands, verdicts = reason_cands(["42", "42", "42"])
    cand, verdict = await judge(cands, verdicts)

    assert verdict.passed
    assert verdict.confidence == pytest.approx(1.0)   # critics never ran


async def test_judge_single_sample_defers_to_vote():
    judge = ReasoningJudge([critic_const(False)])
    cands, verdicts = reason_cands(["42"])
    _, verdict = await judge(cands, verdicts)

    assert not verdict.passed   # can't establish self-consistency from one sample


# ── Judge: conditional critics ───────────────────────────────────────

async def test_low_agreement_triggers_critics_majority_sound_passes():
    judge = ReasoningJudge(
        [critic_const(False), critic_const(False), critic_const(True)],
        agreement_threshold=0.9,   # 2/3 agreement is "low" -> critics fire
    )
    cands, verdicts = reason_cands(["42", "42", "7"])
    _, verdict = await judge(cands, verdicts)

    assert verdict.passed   # 2 of 3 critics found it sound


async def test_low_agreement_critics_majority_refute_fails():
    judge = ReasoningJudge(
        [critic_const(True, "bad1"), critic_const(True, "bad2"), critic_const(False)],
        agreement_threshold=0.9,
    )
    cands, verdicts = reason_cands(["42", "42", "7"])
    _, verdict = await judge(cands, verdicts)

    assert not verdict.passed
    assert "bad1" in verdict.feedback   # refuters' reasons threaded back


async def test_high_stakes_forces_critics_despite_agreement():
    judge = ReasoningJudge([critic_const(True, "subtle flaw")], agreement_threshold=0.6)
    cands, verdicts = reason_cands(["42", "42", "42"], stakes="high")
    _, verdict = await judge(cands, verdicts)

    assert not verdict.passed   # unanimous, but the critic caught a confident error


# ── End-to-end through the engine ─────────────────────────────────────

async def test_engine_elevates_via_self_consistency():
    provider = StubProvider("reasoning...\nAnswer: 42")
    judge = ReasoningJudge([critic_tripwire()], agreement_threshold=0.6)
    engine = HorizonEngine(
        EngineConfig(ladder=resolve_ladder(CODER_LADDER, "gemini-flash", "gpt-5.3-codex"),
                     max_step_samples=3, max_fix_attempts=1),
        make_reasoning_generator(lambda tier: provider),
        ReasonVerifier(),
        aggregator=judge,
    )

    result = await engine.run(Step("q", "what is 2+2?"))
    assert result.passed
    out = result.steps[0]
    assert out.rung_reached == 0          # cheap tier sufficed
    assert out.samples_used == 4          # 1 single (defers) + 3-sample vote
