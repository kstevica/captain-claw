"""The Harbor adapter's pure parts. Needs Harbor installed (skipped otherwise):

    PYTHONPATH=evals/harbor python -m pytest evals/harbor/test_captain_harbor.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("harbor")

import captain_harbor as ch  # noqa: E402


def test_split_model_keeps_slashes_in_the_model():
    assert ch.split_model("anthropic/claude-haiku-5-5") == ("anthropic", "claude-haiku-5-5")
    assert ch.split_model("openrouter/deepseek/deepseek-v4-pro") == ("openrouter", "deepseek/deepseek-v4-pro")
    with pytest.raises(ValueError):
        ch.split_model("claude-haiku-5-5")


def test_runner_flags_deck_by_default_standalone_on_request():
    deck = ch.runner_flags(
        "anthropic", "m", ch.CaptainOptions(reasoning_effort="max", drive="ambitious", archetype="a"), "/logs/agent")
    assert deck[:4] == ["-p", "anthropic", "-m", "m"]
    assert ["--drive", "ambitious"] == deck[deck.index("--drive"):][:2]
    with pytest.raises(ValueError):
        ch.CaptainOptions(drive="frantic")
    assert "--standalone" not in deck and ["--archetype", "a"] == deck[deck.index("--archetype"):][:2]
    assert ["--effort", "max"] == deck[deck.index("--effort"):][:2]
    assert "/logs/agent/captain-result.json" in deck
    alone = ch.runner_flags("anthropic", "m", ch.CaptainOptions(standalone=True, archetype="a"), "/logs/agent")
    assert "--standalone" in alone and "--bench" in alone and "--archetype" not in alone


def test_context_from_result():
    ctx = ch.context_from_result({
        "ok": True, "mode": "deck", "cost_usd_estimate": 0.01, "tool_calls": 3,
        "usage": {"prompt_tokens": 100, "completion_tokens": 9, "cache_read_input_tokens": 80},
        "messages": [{"role": "user"}],
    })
    assert ctx["n_input_tokens"] == 100 and ctx["n_output_tokens"] == 9 and ctx["n_cache_tokens"] == 80
    assert ctx["cost_usd"] == 0.01
    assert ctx["metadata"]["mode"] == "deck" and "messages" not in ctx["metadata"]


def test_a_harbor_trial_runs_at_full_stretch_and_locked_by_default():
    flags = ch.runner_flags("anthropic", "m", ch.CaptainOptions(), "/logs/agent")
    assert ["--effort", "max"] == flags[flags.index("--effort"):][:2]
    assert ["--drive", "ambitious"] == flags[flags.index("--drive"):][:2]
    assert "--lock" in flags


def test_auto_leaves_a_level_to_captain_and_lock_can_be_turned_off():
    flags = ch.runner_flags("anthropic", "m", ch.CaptainOptions(reasoning_effort="auto", drive="persistent"), "/l")
    assert "--effort" not in flags and ["--drive", "persistent"] == flags[flags.index("--drive"):][:2]
    assert "--lock" in flags
    nothing = ch.runner_flags("anthropic", "m", ch.CaptainOptions(reasoning_effort="auto", drive="auto"), "/l")
    assert not {"--effort", "--drive", "--lock"} & set(nothing)
    loose = ch.runner_flags("anthropic", "m", ch.CaptainOptions(lock=False), "/l")
    assert "--effort" in loose and "--lock" not in loose
