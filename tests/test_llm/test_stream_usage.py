"""Token usage survives stream collection, whatever chunk it arrives on.

With ``stream_options.include_usage`` OpenAI sends the counts on a final chunk
whose ``choices`` is empty, and LiteLLM 1.109+ passes that shape on for every
provider, Anthropic included. Up to 1.108 LiteLLM padded the chunk with an
empty choice. The collector used to skip choice-less chunks before reading
usage, so every streamed call recorded 0 tokens on current LiteLLM.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import captain_claw.llm as llm_mod
from captain_claw.llm import (
    LiteLLMProvider,
    Message,
    _extract_usage,
    _merge_stream_usage,
)


def _text(content, finish=None, usage=None):
    chunk = {"choices": [{"delta": {"content": content}, "finish_reason": finish}]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _usage_only(usage):
    return {"choices": [], "usage": usage}


async def _stream(chunks):
    for c in chunks:
        yield c


async def _collect(chunks):
    provider = LiteLLMProvider(provider="openai", model="m", api_key="x")
    return await provider._collect_streaming_response(_stream(chunks))


REAL = {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}
ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def test_usage_on_choiceless_final_chunk_is_kept():
    # The regression: OpenAI / LiteLLM 1.111 shape.
    collected = await _collect([
        _text("Hello"),
        _text("!", finish="stop"),
        _usage_only(REAL),
    ])
    assert collected["choices"][0]["message"]["content"] == "Hello!"
    u = _extract_usage(collected["usage"])
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (120, 8, 128)


async def test_usage_only_on_last_chunk_with_choices():
    # LiteLLM 1.83 shape: the usage chunk carries an empty choice.
    collected = await _collect([
        _text("Hi"),
        _text("", finish="stop", usage=REAL),
    ])
    assert _extract_usage(collected["usage"])["prompt_tokens"] == 120


async def test_early_counts_not_overwritten_by_zero_usage():
    collected = await _collect([
        _text("Hi", usage=REAL),
        _text(" there", finish="stop"),
        _usage_only(ZERO),
    ])
    u = _extract_usage(collected["usage"])
    assert (u["prompt_tokens"], u["completion_tokens"]) == (120, 8)


async def test_early_counts_not_overwritten_by_zero_usage_with_choices():
    collected = await _collect([
        _text("Hi", usage=REAL),
        _text("", finish="stop", usage=ZERO),
    ])
    assert _extract_usage(collected["usage"])["completion_tokens"] == 8


async def test_no_usage_at_all():
    collected = await _collect([_text("Hi"), _text("", finish="stop")])
    assert collected["usage"] is None
    assert _extract_usage(collected["usage"])["total_tokens"] == 0


async def test_running_totals_end_at_the_final_figure():
    # Gemini-style: every chunk resends the running totals.
    collected = await _collect([
        _text("a", usage={"prompt_tokens": 50, "completion_tokens": 1}),
        _text("b", usage={"prompt_tokens": 50, "completion_tokens": 2}),
        _text("c", finish="stop", usage={"prompt_tokens": 50, "completion_tokens": 3}),
    ])
    u = _extract_usage(collected["usage"])
    assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (50, 3, 53)


async def test_input_and_output_on_different_chunks_are_both_kept():
    collected = await _collect([
        _text("a", usage={"prompt_tokens": 300, "cache_read_input_tokens": 200}),
        _text("b", finish="stop", usage={"completion_tokens": 9}),
    ])
    u = _extract_usage(collected["usage"])
    assert u["prompt_tokens"] == 300
    assert u["completion_tokens"] == 9
    assert u["cache_read_input_tokens"] == 200
    assert u["total_tokens"] == 309


async def test_openai_cached_tokens_survive_a_merge():
    collected = await _collect([
        _text("a", usage={"prompt_tokens": 1000, "completion_tokens": 1,
                          "prompt_tokens_details": {"cached_tokens": 800}}),
        _usage_only({"prompt_tokens": 1000, "completion_tokens": 40}),
    ])
    u = _extract_usage(collected["usage"])
    assert u["cache_read_input_tokens"] == 800
    assert u["prompt_tokens"] == 200
    assert u["completion_tokens"] == 40


def test_merge_keeps_a_single_usage_object_untouched():
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    assert _merge_stream_usage(None, usage) is usage
    assert _merge_stream_usage(usage, None) is usage
    assert _merge_stream_usage(usage, SimpleNamespace(prompt_tokens=0)) is usage


def test_merge_with_litellm_usage_objects():
    litellm = pytest.importorskip("litellm")
    real = litellm.Usage(prompt_tokens=650, completion_tokens=9, total_tokens=659,
                         cache_creation_input_tokens=50, cache_read_input_tokens=400)
    zero = litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assert _merge_stream_usage(real, zero) is real
    later = litellm.Usage(prompt_tokens=650, completion_tokens=12, total_tokens=662)
    u = _extract_usage(_merge_stream_usage(real, later))
    assert u["completion_tokens"] == 12
    assert u["cache_read_input_tokens"] == 400
    assert u["cache_creation_input_tokens"] == 50


async def test_complete_with_callback_records_choiceless_usage(monkeypatch):
    # Through the public streamed path: usage lands on LLMResponse.usage.
    async def fake_acompletion(kwargs, provider):
        assert kwargs["stream_options"] == {"include_usage": True}
        return _stream([
            _text("Hello"),
            _text(" there", finish="stop"),
            _usage_only({**REAL, "cache_read_input_tokens": 30}),
        ])

    monkeypatch.setattr(llm_mod, "_acompletion_tolerant", fake_acompletion)
    provider = LiteLLMProvider(provider="openai", model="m", api_key="x")
    seen: list[str] = []
    resp = await provider.complete_with_callback(
        [Message(role="user", content="hi")], on_chunk=seen.append)
    assert resp.content == "Hello there"
    assert "".join(seen) == "Hello there"
    assert resp.usage["prompt_tokens"] == 120
    assert resp.usage["completion_tokens"] == 8
    assert resp.usage["cache_read_input_tokens"] == 30
