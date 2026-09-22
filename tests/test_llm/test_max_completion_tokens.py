"""OpenAI reasoning models (GPT-5 and later, o-series) reject ``max_tokens``
with a 400 "Unsupported parameter: 'max_tokens' is not supported with this
model. Use 'max_completion_tokens' instead." LiteLLM only translates the key
for models in its own registry, so a model newer than it (the gpt-6 family)
crashed every turn. The request builder now sends ``max_completion_tokens`` for
those families, and ``_acompletion_tolerant`` renames the key on that exact 400
for anything the static check misses.
"""

from __future__ import annotations

import sys
import types

import pytest

import captain_claw.llm as llm_mod
from captain_claw.llm import (
    LiteLLMProvider,
    Message,
    _acompletion_tolerant,
    _is_max_tokens_rejected_error,
    _normalize_temperature_for_model,
    _uses_max_completion_tokens,
)

_MAX_TOKENS_400 = (
    "litellm.badrequesterror: openaiexception - unsupported parameter: "
    "'max_tokens' is not supported with this model. use "
    "'max_completion_tokens' instead."
)


@pytest.mark.parametrize(
    "model",
    ["gpt-6", "gpt-6-mini", "gpt-6.1", "openai/gpt-6", "gpt-5", "gpt-5.2-codex",
     "o1", "o3-mini", "o4-mini"],
)
def test_reasoning_families_use_max_completion_tokens(model):
    assert _uses_max_completion_tokens("openai", model)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "omni-x"])
def test_older_openai_models_keep_max_tokens(model):
    assert not _uses_max_completion_tokens("openai", model)


def test_non_openai_provider_keeps_max_tokens():
    assert not _uses_max_completion_tokens("anthropic", "gpt-6")
    assert not _uses_max_completion_tokens("openrouter", "o3-mini")


def test_gpt6_temperature_pinned_like_gpt5():
    assert _normalize_temperature_for_model("openai", "gpt-6", 0.3) == 1.0


def test_request_kwargs_send_the_right_key():
    new = LiteLLMProvider(provider="openai", model="gpt-6", api_key="x")
    kw = new._request_kwargs([Message(role="user", content="hi")], max_tokens=500)
    assert kw["max_completion_tokens"] == 500 and "max_tokens" not in kw

    old = LiteLLMProvider(provider="openai", model="gpt-4o", api_key="x")
    kw = old._request_kwargs([Message(role="user", content="hi")], max_tokens=500)
    assert kw["max_tokens"] == 500 and "max_completion_tokens" not in kw


def test_classifier():
    assert _is_max_tokens_rejected_error(_MAX_TOKENS_400)
    assert not _is_max_tokens_rejected_error("max_tokens is too large: 99999")
    assert not _is_max_tokens_rejected_error("context length exceeded")


async def test_tolerant_renames_key_and_remembers(monkeypatch):
    seen: list[dict] = []

    async def fake_acompletion(**kwargs):
        seen.append(kwargs)
        if "max_tokens" in kwargs:
            raise Exception(_MAX_TOKENS_400)
        return "ok"

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=fake_acompletion))
    monkeypatch.setattr(llm_mod, "_MAX_COMPLETION_TOKENS_MODELS", set())

    out = await _acompletion_tolerant({"model": "custom/future-reasoner", "max_tokens": 700})
    assert out == "ok"
    assert len(seen) == 2
    assert seen[1]["max_completion_tokens"] == 700 and "max_tokens" not in seen[1]
    # Learned: later requests for this model send the right key up front.
    assert _uses_max_completion_tokens("custom", "future-reasoner")
