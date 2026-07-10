"""Tests for the cross-mode quality/cost governor.

The load-bearing guarantee is: an absent/empty ``quality`` config reproduces the
systems' current behaviour (every feature off, budget unbounded). If any of these
fail, the safety envelope is broken and features could regress the base systems.
"""

from __future__ import annotations

import math

from captain_claw.flight_deck.quality_profile import (
    QualityProfile,
    TokenBudget,
    build_quality_metrics,
    worker_produced_nothing,
)


# ── the safety property ──────────────────────────────────────────────

def test_empty_config_is_all_off():
    for cfg in (None, {}, {"profile": "off"}):
        p = QualityProfile.from_dict(cfg)
        assert p.profile == "off"
        assert not p.any_enabled
        for name in QualityProfile._BOOL_FLAGS:
            assert getattr(p, name) is False, name
        assert p.token_budget == 0  # unbounded → no cost change


def test_unknown_profile_degrades_to_off():
    p = QualityProfile.from_dict({"profile": "bogus"})
    assert p.profile == "off"
    assert not p.any_enabled


# ── presets ──────────────────────────────────────────────────────────

def test_balanced_enables_only_the_free_or_saving_levers():
    p = QualityProfile.from_dict({"profile": "balanced"})
    assert p.test_gate and p.acted_gate
    assert p.research_map and p.worker_escalate
    assert p.delta_rounds and p.critic_triage  # free savers
    assert p.judgment_ledger                    # R11, free
    # Paid / heavier levers stay off in balanced.
    assert not p.coverage_check
    assert not p.git_snapshots
    assert not p.source_corpus and not p.rubric_contract
    assert not p.intent_brief  # R12 costs a routing-time call → thorough, not balanced


def test_thorough_enables_the_wired_paid_levers():
    p = QualityProfile.from_dict({"profile": "thorough"})
    assert p.coverage_check and p.git_snapshots
    assert p.test_gate and p.acted_gate  # still inherits balanced's set
    assert p.delta_rounds and p.critic_triage and p.judgment_ledger
    assert p.source_corpus and p.rubric_contract  # R10, R9
    assert p.intent_brief  # R12


def test_paid_levers_are_never_preset_enabled():
    # deep_build (code) and claim_check (research) must be explicit opt-ins.
    for name in ("off", "balanced", "thorough"):
        p = QualityProfile.from_dict({"profile": name})
        assert not p.deep_build, name
        assert not p.claim_check, name


def test_claim_check_is_explicit_with_its_knob():
    p = QualityProfile.from_dict(
        {"profile": "thorough", "claim_check": True, "claim_check_max": 12,
         "token_budget": 400_000})
    assert p.claim_check and p.claim_check_max == 12 and p.token_budget == 400_000


def test_deep_build_is_explicit_opt_in_with_knobs():
    p = QualityProfile.from_dict(
        {"profile": "thorough", "deep_build": True, "deep_build_samples": 3,
         "token_budget": 500_000})
    assert p.deep_build
    assert p.deep_build_samples == 3
    assert p.token_budget == 500_000


def test_explicit_flag_overrides_preset_both_ways():
    on = QualityProfile.from_dict({"profile": "off", "test_gate": True})
    assert on.test_gate and on.any_enabled
    off = QualityProfile.from_dict({"profile": "thorough", "coverage_check": False})
    assert not off.coverage_check
    assert off.git_snapshots  # untouched flags keep the preset


def test_numeric_knobs_are_clamped_sane():
    p = QualityProfile.from_dict(
        {"deep_build_samples": 0, "deep_build_fix_attempts": -3, "escalate_max": -1,
         "token_budget": -50})
    assert p.deep_build_samples == 1
    assert p.deep_build_fix_attempts == 0
    assert p.escalate_max == 0
    assert p.token_budget == 0


# ── token budget ─────────────────────────────────────────────────────

def test_unbounded_budget_never_refuses():
    b = TokenBudget(0)
    assert b.unbounded
    b.add(10_000_000)
    assert b.can_afford(10_000_000) is True
    assert b.remaining() == math.inf
    assert not b.over()
    assert b.stopped_reason == ""


def test_bounded_budget_refuses_past_ceiling_without_interrupting():
    b = TokenBudget(1000)
    assert b.can_afford(600)
    b.add(600)
    assert b.remaining() == 400
    # A lever estimated to overshoot is refused *before* it starts.
    assert b.can_afford(500) is False
    assert "budget reached" in b.stopped_reason
    # Work already counted is never rolled back.
    assert b.spent() == 600
    # A cheaper lever that still fits is allowed.
    assert b.can_afford(300) is True


def test_over_flips_once_spent_reaches_total():
    b = TokenBudget(100)
    assert not b.over()
    b.add(100)
    assert b.over()


# ── R2 acted-gate helper ─────────────────────────────────────────────

def test_worker_with_text_is_not_a_noop():
    assert worker_produced_nothing({"output": "here is my answer", "actions": []}) is False


def test_worker_that_wrote_a_file_is_not_a_noop():
    d = {"output": "", "actions": [{"tool": "write", "detail": "report.md"}]}
    assert worker_produced_nothing(d) is False


def test_pure_narration_is_a_noop():
    d = {"output": "  ", "actions": [{"tool": "narration", "detail": "I'll analyse this"}]}
    assert worker_produced_nothing(d) is True


def test_missing_result_counts_as_noop():
    assert worker_produced_nothing(None) is True
    assert worker_produced_nothing({}) is True


# ── per-run quality metrics assembly (increment 3) ───────────────────

def test_metrics_empty_when_nothing_ran():
    # A run with no levers must not persist a metrics record at all.
    assert build_quality_metrics() == {}
    assert build_quality_metrics(budget=TokenBudget(0)) == {}  # no spend, no reason


def test_metrics_claim_tally_mirrors_verdict_semantics():
    findings = [
        {"verdict": "confirmed"},
        {"verdict": "refuted", "correction": "the right value"},
        {"verdict": "unverifiable", "hedge": "reportedly …"},
        {"verdict": "unverifiable", "hedge": ""},  # already-qualified → not hedged
    ]
    m = build_quality_metrics(claim_findings=findings)
    assert m == {"claims_checked": 4, "claims_confirmed": 1, "claims_refuted": 1,
                 "claims_unverifiable": 2, "claims_hedged": 1}


def test_metrics_distinguish_clean_run_from_no_run():
    # Ran-and-clean records zeros; never-ran records nothing.
    assert build_quality_metrics(claim_findings=[])["claims_checked"] == 0
    assert "claims_checked" not in build_quality_metrics(claim_findings=None)


def test_metrics_consistency_and_gaps_counts():
    m = build_quality_metrics(
        consistency={"critical": 0, "major": 1, "initial_critical": 2, "revised": True},
        gaps=[{"severity": "major"}, {"severity": "minor"}, {"severity": "minor"}])
    assert m["consistency_critical"] == 0
    assert m["consistency_initial_critical"] == 2
    assert m["consistency_revised"] is True
    assert m["gaps_major"] == 1 and m["gaps_minor"] == 2


def test_metrics_counters_and_budget():
    b = TokenBudget(1000)
    b.add(600)
    assert b.can_afford(500) is False  # trips stopped_reason
    m = build_quality_metrics(acted_retries=2, escalations=0, budget=b)
    assert m["acted_retries"] == 2 and m["escalations"] == 0
    assert m["budget_spent_tokens"] == 600
    assert "budget reached" in m["budget_stopped_reason"]
