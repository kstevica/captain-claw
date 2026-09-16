"""The done-gate blocks only on DETERMINISTIC hard findings, and quality scores.

Regression for the glasses "Story run 7" incident: a single PR (professional-procedure)
MODEL-pass finding, clamped to hard, false-blocked a finished, internally-airtight story
under gate_blocks_done. Model-pass judgments (C/F/I/J/CA/PR) must surface and drive
revision but never gate `done` on their own — only the fail-safe deterministic passes
(A/B/D/E/G/H) block. Plus: advisory 0-100 quality scores (integrity + continuity /
plausibility / grounding / craft).
"""

import pytest

from captain_claw.flight_deck import story_state as ss

_ALL = {"A", "B", "D", "E", "G", "H", "C", "F", "I", "J", "CA", "PR"}


def _f(pass_, severity, origin=None):
    d = {"pass": pass_, "severity": severity, "kind": "k", "reason": "r", "scene": "s"}
    if origin:
        d["origin"] = origin
    return d


# ── is_deterministic ──────────────────────────────────────────────────

def test_is_deterministic_by_origin():
    assert ss.is_deterministic(_f("A", "hard", "deterministic")) is True
    assert ss.is_deterministic(_f("PR", "hard", "model")) is False


def test_is_deterministic_fallback_by_pass_id():
    # No origin tag → classify by pass id (older/foreign findings).
    assert ss.is_deterministic(_f("G", "hard")) is True
    assert ss.is_deterministic(_f("F", "hard")) is False
    assert ss.is_deterministic(_f("PR", "hard")) is False


# ── blocking_findings: the gating set ─────────────────────────────────

def test_only_deterministic_hard_blocks():
    findings = [
        _f("G", "hard", "deterministic"),   # blocks
        _f("PR", "hard", "model"),           # does NOT block
        _f("F", "hard", "model"),            # does NOT block
        _f("A", "major", "deterministic"),   # major never blocks
    ]
    blocking = ss.blocking_findings(findings)
    assert len(blocking) == 1
    assert blocking[0]["pass"] == "G"


def test_all_model_hard_yields_no_blocker():
    # The exact run-7 shape: 1 PR hard + F majors + I softs, all model → nothing gates.
    findings = ([_f("PR", "hard", "model")]
                + [_f("F", "major", "model") for _ in range(6)]
                + [_f("I", "soft", "model") for _ in range(5)])
    assert ss.blocking_findings(findings) == []


# ── scores ────────────────────────────────────────────────────────────

def test_clean_scores_all_100():
    sc = ss.score([], passes_run=_ALL)
    assert sc["integrity"] == 100 and sc["grade"] == "clean"
    assert sc["continuity"] == 100 and sc["plausibility"] == 100


def test_penalties_per_dimension_and_floor():
    findings = [_f("PR", "hard", "model"),          # plausibility -25
                _f("F", "major", "model"),           # grounding -10
                _f("I", "soft", "model")]            # craft -3
    sc = ss.score(findings, passes_run=_ALL)
    assert sc["continuity"] == 100       # no continuity findings
    assert sc["plausibility"] == 75
    assert sc["grounding"] == 90
    assert sc["craft"] == 97
    # composite is continuity-dominant (0.7) + 0.1 each of the model dims
    assert sc["integrity"] == round(0.7 * 100 + 0.1 * 75 + 0.1 * 90 + 0.1 * 97)


def test_dimension_score_floors_at_zero():
    findings = [_f("F", "hard", "model") for _ in range(10)]  # grounding -250
    sc = ss.score(findings, passes_run=_ALL)
    assert sc["grounding"] == 0


def test_unchecked_dimension_is_null_not_100():
    # Only deterministic + PR ran → grounding (F/CA) and craft (I/J) were never checked.
    sc = ss.score([_f("PR", "hard", "model")], passes_run=ss._DETERMINISTIC_PASSES | {"PR"})
    assert sc["continuity"] == 100          # deterministic always runs
    assert sc["plausibility"] == 75         # PR ran
    assert sc["grounding"] is None          # F/CA never ran
    assert sc["craft"] is None              # I/J never ran


def test_grade_thresholds():
    assert ss._grade(85) == "clean"
    assert ss._grade(70) == "sound"
    assert ss._grade(50) == "caution"
    assert ss._grade(49) == "weak"


def test_continuity_backbone_survives_noisy_model_findings():
    # Airtight deterministic backbone, many noisy model complaints → still not "weak".
    findings = ([_f("F", "hard", "model") for _ in range(4)]
                + [_f("PR", "hard", "model") for _ in range(4)])
    sc = ss.score(findings, passes_run=_ALL)
    assert sc["continuity"] == 100
    assert sc["integrity"] >= 70    # continuity's 0.7 weight keeps the headline up


# ── summarize surfaces blocking + scores ──────────────────────────────

def test_summarize_reports_blocking_and_scores():
    findings = [_f("PR", "hard", "model"), _f("F", "major", "model")]
    result = {"findings": findings, "initial_findings": findings, "rounds": 1,
              "blocking": ss.blocking_findings(findings),
              "scores": ss.score(findings, passes_run=_ALL), **ss.bucket(findings)}
    s = ss.summarize(result)
    assert s["hard"] == 1          # a hard finding exists
    assert s["blocking"] == 0      # but nothing deterministic gates
    assert s["scores"]["grade"] in ("clean", "sound", "caution", "weak")


# ── run_validator integration ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_validator_model_hard_does_not_block():
    async def extract_fn(prompt):
        return '{"characters": [{"name": "Ana", "age": "34"}]}'  # clean store

    async def model_fn(prompt):
        # a single PR-pass hard objection (clamped to the PR ceiling = hard)
        return '[{"severity": "hard", "kind": "procedure", "reason": "r", "scene": "sc"}]'

    res = await ss.run_validator(
        "# Ch1\n\nAna was 34, a calm morning.", extract_fn=extract_fn,
        revise_fn=None, model_fn=model_fn, model_passes=("PR",))
    assert len(res["hard"]) == 1            # the model hard is reported
    assert res["blocking"] == []           # but it does NOT gate `done`
    assert res["scores"]["continuity"] == 100
    assert res["scores"]["plausibility"] == 75


@pytest.mark.asyncio
async def test_run_validator_deterministic_hard_blocks():
    async def extract_fn(prompt):
        return '{"characters": [{"name": "Ana", "age": "34"}, {"name": "Ana", "age": "41"}]}'

    res = await ss.run_validator("Ana 34 then 41.", extract_fn=extract_fn, revise_fn=None)
    assert len(res["hard"]) >= 1
    assert len(res["blocking"]) >= 1       # a deterministic age mismatch DOES gate
    assert all(ss.is_deterministic(f) for f in res["blocking"])


# ── composite must not bake in un-run dimensions (verifier finding 1) ──

def test_composite_excludes_unchecked_dimensions():
    # deterministic-only: the model dims never ran → integrity must equal continuity, not
    # a blend that counts their default 100.
    sc = ss.score([_f("A", "hard", "deterministic")], passes_run=ss._DETERMINISTIC_PASSES)
    assert sc["continuity"] == 75
    assert sc["plausibility"] is None and sc["grounding"] is None and sc["craft"] is None
    assert sc["integrity"] == 75      # == continuity, NOT an inflated 82
    assert sc["grade"] == "sound"


def test_integrity_null_when_nothing_checked():
    sc = ss.score([], passes_run=set())
    assert sc["integrity"] is None and sc["grade"] is None


# ── model-pass outage must score null, not a fabricated 100 (finding 2) ─

@pytest.mark.asyncio
async def test_run_validator_model_outage_scores_null_not_100():
    async def extract_fn(prompt):
        return '{"characters": [{"name": "Ana", "age": "34"}]}'   # clean store

    async def boom(prompt):
        raise RuntimeError("provider down")

    res = await ss.run_validator(
        "# Ch1\n\nAna was 34, a calm morning.", extract_fn=extract_fn, revise_fn=None,
        model_fn=boom, model_passes=("C", "F", "I", "J", "CA", "PR"))
    sc = res["scores"]
    assert sc["continuity"] == 100
    # every model pass failed → those dimensions are null, NOT a fabricated 100
    assert sc["plausibility"] is None
    assert sc["grounding"] is None
    assert sc["craft"] is None
    assert sc["integrity"] == 100     # renormalized to continuity only — honest


@pytest.mark.asyncio
async def test_run_validator_no_model_fn_scores_null_model_dims():
    async def extract_fn(prompt):
        return '{"characters": [{"name": "Ana", "age": "34"}]}'

    # model_passes requested but no model wired → the model dims never ran
    res = await ss.run_validator(
        "# Ch1\n\nAna was 34.", extract_fn=extract_fn, revise_fn=None,
        model_fn=None, model_passes=("C", "F", "I", "J", "CA", "PR"))
    sc = res["scores"]
    assert sc["plausibility"] is None and sc["grounding"] is None and sc["craft"] is None
    assert sc["integrity"] == sc["continuity"]


@pytest.mark.asyncio
async def test_run_validator_partial_model_outage_nulls_only_failed_dims():
    # C and PR (plausibility) time out; F/I/J/CA succeed → plausibility null, others scored.
    async def extract_fn(prompt):
        return '{"characters": [{"name": "Ana", "age": "34"}]}'

    async def model_fn(prompt):
        if "PHYSICAL-MECHANISM" in prompt or "PROFESSIONAL-PROCEDURE" in prompt:
            raise RuntimeError("timeout")
        return '{"findings": []}'   # ran clean

    res = await ss.run_validator(
        "# Ch1\n\nAna was 34, a calm morning.", extract_fn=extract_fn, revise_fn=None,
        model_fn=model_fn, model_passes=("C", "F", "I", "J", "CA", "PR"))
    sc = res["scores"]
    assert sc["plausibility"] is None      # C + PR both failed
    assert sc["grounding"] == 100          # F + CA ran clean
    assert sc["craft"] == 100              # I + J ran clean
