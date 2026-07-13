"""Iskra Constitution: stage gates, preset clamps, tier-weighted debit math."""

from __future__ import annotations

from captain_claw.flight_deck import being_constitution as c


def test_capabilities_are_cumulative():
    assert c.capabilities("egg") == frozenset()
    assert "chat" in c.capabilities("infant")
    assert "web_read" not in c.capabilities("infant")
    assert "web_read" in c.capabilities("child")
    assert "organ_runs" not in c.capabilities("child")
    assert "organ_runs" in c.capabilities("adolescent")
    assert "chat" in c.capabilities("adult")          # inherits everything
    assert "procreate" in c.capabilities("adult")
    assert not c.has_capability("adolescent", "procreate")


def test_tier_gates_follow_stage():
    assert c.tier_allowed("infant", "fast")
    assert not c.tier_allowed("infant", "balanced")
    assert not c.tier_allowed("child", "reason")
    assert c.tier_allowed("adolescent", "balanced")
    assert not c.tier_allowed("adolescent", "reason")
    assert c.tier_allowed("adult", "reason")


def test_clamp_preset_caps_by_stage():
    assert c.clamp_preset("infant", "50M") == "2M"
    assert c.clamp_preset("child", "2M") == "2M"        # under cap: untouched
    assert c.clamp_preset("child", "unlimited") == "5M"  # unlimited below adult
    assert c.clamp_preset("adolescent", "10M") == "10M"
    assert c.clamp_preset("adult", "unlimited") == "unlimited"


def test_savings_ceiling_days_times_allowance():
    assert c.savings_ceiling_tokens("infant", "2M") == 6_000_000     # 3 days
    assert c.savings_ceiling_tokens("child", "5M") == 35_000_000     # 7 days
    assert c.savings_ceiling_tokens("adolescent", "20M") == 600_000_000
    assert c.savings_ceiling_tokens("adult", "unlimited") is None


def test_weighted_tokens_tier_and_cache_aware():
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    assert c.weighted_tokens(usage, "fast") == 1500
    assert c.weighted_tokens(usage, "reason") == 15000
    cached = {"prompt_tokens": 0, "completion_tokens": 100,
              "cache_read_input_tokens": 10_000,
              "cache_creation_input_tokens": 1000}
    # 100 + 0.1*10000 + 1.25*1000 = 2350
    assert c.weighted_tokens(cached, "fast") == 2350
    assert c.weighted_tokens(cached, "balanced") == 7050
    assert c.weighted_tokens({}, "reason") == 0
    assert c.weighted_tokens(None, "fast") == 0


def test_unknown_tier_defaults_to_weight_one():
    assert c.weighted_tokens({"completion_tokens": 10}, "mystery") == 10


def test_metamorphosis_policy_per_stage():
    assert c.metamorphosis_policy("child") == "none"
    assert c.metamorphosis_policy("adolescent") == "cosign"
    assert c.metamorphosis_policy("adult") == "auto"


def test_constitution_text_carries_all_nine_invariants():
    text = c.constitution_text()
    assert len(c.INVARIANTS) == 9
    for title, _ in c.INVARIANTS:
        assert title in text
