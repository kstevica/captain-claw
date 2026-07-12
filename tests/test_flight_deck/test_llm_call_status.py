"""The dispatch loop surfaces an in-flight LLM call so a slow model call isn't mistaken
for a stall. `_llm_call_status` is the pure normaliser behind that: it recognises the
agent's "Calling LLM …" runtime-status broadcast and strips the ⚡ streaming prefix so a
call and its first-chunk repeat collapse to a single surfaced line."""

from __future__ import annotations

from captain_claw.flight_deck.basna_routes import _llm_call_status


def test_calling_llm_status_is_surfaced_verbatim():
    s = "Calling LLM (qwen2.5-coder) · 12,345 ctx tokens (30%)..."
    assert _llm_call_status(s) == s


def test_streaming_prefix_is_stripped_to_match_the_call_line():
    call = "Calling LLM (qwen2.5-coder) · 12,345 ctx tokens (30%)..."
    stream = "⚡ " + call
    # Both normalise to the same string, so the dedupe in _handle collapses them.
    assert _llm_call_status(stream) == call
    assert _llm_call_status(call) == _llm_call_status(stream)


def test_non_llm_statuses_are_ignored():
    for s in ("thinking", "Using read...", "waiting", "ready", "streaming", "", "  "):
        assert _llm_call_status(s) is None


def test_none_input_is_safe():
    assert _llm_call_status(None) is None  # type: ignore[arg-type]
