"""Tests for provider usage extraction, incl. OpenAI-compatible caching."""

from captain_claw.llm import _extract_usage


def test_extract_usage_anthropic_shape_passthrough():
    u = _extract_usage({
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
        "cache_creation_input_tokens": 30, "cache_read_input_tokens": 40,
    })
    assert u["prompt_tokens"] == 100          # anthropic input is already net
    assert u["cache_read_input_tokens"] == 40
    assert u["cache_creation_input_tokens"] == 30


def test_extract_usage_openai_cached_tokens_de_overlapped():
    # OpenAI reports cached tokens INSIDE prompt_tokens — normalise so pricing
    # counts them once, at the cache rate.
    u = _extract_usage({
        "prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
        "prompt_tokens_details": {"cached_tokens": 800, "audio_tokens": 0},
    })
    assert u["cache_read_input_tokens"] == 800
    assert u["prompt_tokens"] == 200          # 1000 − 800 cached = net input


def test_extract_usage_anthropic_wins_when_both_present():
    # If the explicit anthropic field is set, don't also subtract the openai one.
    u = _extract_usage({
        "prompt_tokens": 500, "completion_tokens": 10,
        "cache_read_input_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 300},
    })
    assert u["cache_read_input_tokens"] == 100
    assert u["prompt_tokens"] == 500          # untouched — no double subtraction


def test_extract_usage_no_cache_is_zero():
    u = _extract_usage({"prompt_tokens": 10, "completion_tokens": 5})
    assert u["cache_read_input_tokens"] == 0
    assert u["prompt_tokens"] == 10
