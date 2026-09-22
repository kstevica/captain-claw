"""gpt-6-luna on /v1/chat/completions refuses function tools unless reasoning is
off: "Function tools with reasoning_effort are not supported for gpt-6-luna …
set reasoning_effort to 'none'". Tiers can now pin effort via the model-name
suffix (``gpt-6-luna-none``), and the request self-heals the 400 by retrying
with ``reasoning_effort="none"`` — including when the 400 surfaces during
stream iteration rather than on the ``acompletion()`` call.
"""

from __future__ import annotations

import sys
import types

import captain_claw.llm as llm_mod
from captain_claw.llm import (
    LiteLLMProvider,
    Message,
    ToolDefinition,
    _acompletion_tolerant,
    _extract_reasoning_effort,
    _is_reasoning_with_tools_rejected_error,
)

_LUNA_400 = (
    "litellm.BadRequestError: OpenAIException - Function tools with "
    "reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions. "
    "To use function tools, use /v1/responses or set reasoning_effort to 'none'."
)
_TOOLS = [ToolDefinition(name="write", description="write a file", parameters={})]
_MSGS = [Message(role="user", content="hi")]


def test_none_is_a_recognised_effort_suffix():
    assert _extract_reasoning_effort("gpt-6-luna-none") == ("gpt-6-luna", "none")
    assert _extract_reasoning_effort("gpt-6-luna-high") == ("gpt-6-luna", "high")


def test_none_suffix_sends_reasoning_effort_none():
    p = LiteLLMProvider(provider="openai", model="gpt-6-luna-none", api_key="x")
    assert p.model.endswith("gpt-6-luna")
    kw = p._request_kwargs(_MSGS, tools=_TOOLS)
    assert kw["reasoning_effort"] == "none"


def test_deepseek_none_keeps_thinking_off():
    p = LiteLLMProvider(provider="deepseek", model="deepseek-reasoner-none", api_key="x")
    kw = p._request_kwargs(_MSGS)
    assert "reasoning_effort" not in kw
    assert "thinking" not in (kw.get("extra_body") or {})


def test_classifier():
    assert _is_reasoning_with_tools_rejected_error(_LUNA_400.lower())
    assert not _is_reasoning_with_tools_rejected_error("reasoning_effort must be one of low, medium, high")
    assert not _is_reasoning_with_tools_rejected_error("function tools are limited to 128")


async def test_tolerant_turns_reasoning_off_and_remembers(monkeypatch):
    seen: list[dict] = []

    async def fake_acompletion(**kwargs):
        seen.append(kwargs)
        if kwargs.get("reasoning_effort") != "none":
            raise Exception(_LUNA_400)
        return "ok"

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=fake_acompletion))
    monkeypatch.setattr(llm_mod, "_REASONING_OFF_WITH_TOOLS_MODELS", set())

    out = await _acompletion_tolerant(
        {"model": "openai/gpt-6-luna", "tools": [{"type": "function"}], "reasoning_effort": "high"}
    )
    assert out == "ok" and len(seen) == 2
    assert seen[1]["tools"] == [{"type": "function"}]
    # Learned: the next tool-carrying request goes out with reasoning off.
    p = LiteLLMProvider(provider="openai", model="gpt-6-luna", api_key="x")
    assert p._request_kwargs(_MSGS, tools=_TOOLS)["reasoning_effort"] == "none"
    # …but a tool-less call keeps the model's own default.
    assert "reasoning_effort" not in p._request_kwargs(_MSGS)


async def _raising_stream(exc: Exception):
    if False:
        yield  # pragma: no cover
    raise exc


async def _content_stream():
    yield {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


async def test_streaming_callback_heals_mid_stream_400(monkeypatch):
    monkeypatch.setattr(llm_mod, "_REASONING_OFF_WITH_TOOLS_MODELS", set())
    p = LiteLLMProvider(provider="openai", model="gpt-6-luna", api_key="x")
    calls: list[dict] = []

    async def fake_tolerant(kwargs, prov=None):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return _raising_stream(Exception(_LUNA_400))
        assert kwargs["reasoning_effort"] == "none"
        return _content_stream()

    monkeypatch.setattr(llm_mod, "_acompletion_tolerant", fake_tolerant)
    seen: list[str] = []
    resp = await p.complete_with_callback(_MSGS, tools=_TOOLS, on_chunk=seen.append)
    assert len(calls) == 2 and resp.content == "hello" and seen == ["hello"]


async def test_complete_streaming_heals_mid_stream_400(monkeypatch):
    monkeypatch.setattr(llm_mod, "_REASONING_OFF_WITH_TOOLS_MODELS", set())
    p = LiteLLMProvider(provider="openai", model="gpt-6-luna", api_key="x")
    calls: list[dict] = []

    async def fake_tolerant(kwargs, prov=None):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return _raising_stream(Exception(_LUNA_400))
        return _content_stream()

    monkeypatch.setattr(llm_mod, "_acompletion_tolerant", fake_tolerant)
    chunks = [c async for c in p.complete_streaming(_MSGS, tools=_TOOLS)]
    assert chunks == ["hello"] and calls[1]["reasoning_effort"] == "none"
