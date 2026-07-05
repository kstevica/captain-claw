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
    # Paid / not-yet-wired levers stay off in balanced.
    assert not p.coverage_check
    assert not p.git_snapshots


def test_thorough_enables_the_wired_paid_levers():
    p = QualityProfile.from_dict({"profile": "thorough"})
    assert p.coverage_check and p.git_snapshots
    assert p.test_gate and p.acted_gate  # still inherits balanced's set


def test_no_preset_enables_the_expensive_or_reserved_levers():
    # deep_build (paid) must be explicit; delta_rounds (R4) is not wired yet.
    # Neither may be switched on silently by a preset.
    for name in ("off", "balanced", "thorough"):
        p = QualityProfile.from_dict({"profile": name})
        assert not p.delta_rounds, name
        assert not p.deep_build, name


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
