"""Tests for the run cost / hourly-rate accounting (pure pricing layer)."""

from __future__ import annotations

from captain_claw.flight_deck import pricing


# ── rate lookup ──────────────────────────────────────────────────────

def test_exact_and_family_prefix_match():
    r = pricing.rate_for("claude-opus-4-8")
    assert r and r["input"] == 5.0 and r["output"] == 25.0
    # A dated variant prices off its family via longest-prefix match.
    r2 = pricing.rate_for("claude-opus-4-8-20260601")
    assert r2 == r


def test_provider_prefixed_id_is_normalised():
    assert pricing.rate_for("anthropic/claude-haiku-4-5")["input"] == 1.0
    assert pricing.rate_for("gemini/gemini-2.5-flash")["output"] == 2.5


def test_unknown_model_has_no_rate():
    assert pricing.rate_for("totally-made-up-model") is None
    assert pricing.rate_for("") is None


def test_tier_override_wins():
    r = pricing.rate_for("claude-opus-4-8", override={"input": 1.0, "output": 2.0})
    assert r["input"] == 1.0 and r["output"] == 2.0


# ── cost from usage ──────────────────────────────────────────────────

def test_cost_prices_input_output_and_cache_separately():
    usage = {
        "prompt_tokens": 1_000_000,          # $5 at opus input
        "completion_tokens": 1_000_000,      # $25 at opus output
        "cache_read_input_tokens": 1_000_000,   # $0.50 at opus cache-read
        "cache_creation_input_tokens": 1_000_000,  # $6.25 at opus cache-write
    }
    c = pricing.cost_from_usage("claude-opus-4-8", usage)
    assert c["priced"]
    assert c["input_usd"] == 5.0
    assert c["output_usd"] == 25.0
    assert c["cache_read_usd"] == 0.5
    assert c["cache_write_usd"] == 6.25
    assert c["usd"] == 36.75


def test_cache_read_is_far_cheaper_than_fresh_input():
    fresh = pricing.cost_from_usage("claude-opus-4-8", {"prompt_tokens": 1_000_000})
    cached = pricing.cost_from_usage("claude-opus-4-8", {"cache_read_input_tokens": 1_000_000})
    assert cached["usd"] < fresh["usd"]
    assert cached["usd"] == 0.5 and fresh["usd"] == 5.0


def test_unknown_model_is_unpriced_not_wrong():
    c = pricing.cost_from_usage("mystery-model", {"prompt_tokens": 1_000_000})
    assert c["priced"] is False
    assert c["usd"] is None


def test_empty_usage_costs_zero_but_is_priced():
    c = pricing.cost_from_usage("claude-haiku-4-5", {})
    assert c["priced"] is True and c["usd"] == 0.0


# ── hourly rate (the wage-comparable number) ─────────────────────────

def test_hourly_rate_is_spend_over_wallclock():
    # $2.00 spent in 4 minutes (240s) → $30/hour burn.
    assert pricing.hourly_rate(2.0, 240) == 30.0


def test_hourly_rate_none_without_cost_or_time():
    assert pricing.hourly_rate(None, 240) is None
    assert pricing.hourly_rate(2.0, 0) is None
    assert pricing.hourly_rate(2.0, None) is None


# ── run summary ──────────────────────────────────────────────────────

def test_summarize_aggregates_tokens_cost_and_hourly():
    agents = [
        {"model": "claude-opus-4-8", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}},   # $5
        {"model": "claude-haiku-4-5", "usage": {"prompt_tokens": 0, "completion_tokens": 1_000_000}},  # $5
    ]
    s = pricing.summarize(agents, elapsed_seconds=3600)  # exactly one hour
    assert s["priced"] is True
    assert s["usd"] == 10.0
    assert s["tokens"]["prompt_tokens"] == 1_000_000
    assert s["tokens"]["completion_tokens"] == 1_000_000
    assert s["hourly_usd"] == 10.0  # $10 in 1h → $10/hr
    assert set(s["per_model"]) == {"claude-opus-4-8", "claude-haiku-4-5"}
    assert s["per_model"]["claude-opus-4-8"]["usd"] == 5.0


def test_summarize_unpriced_when_no_model_known():
    s = pricing.summarize([{"model": "mystery", "usage": {"prompt_tokens": 999}}], elapsed_seconds=60)
    assert s["priced"] is False
    assert s["usd"] is None
    assert s["hourly_usd"] is None
    assert s["tokens"]["prompt_tokens"] == 999  # tokens still counted


def test_summarize_empty_run():
    s = pricing.summarize([], elapsed_seconds=None)
    assert s["usd"] is None and s["priced"] is False
    assert s["tokens"]["prompt_tokens"] == 0
