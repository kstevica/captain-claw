"""The chat box's reasoning-effort picker (llm/reasoning.py): which levels each
model offers on each transport, the request fields a level becomes, and the
heal when a server refuses them."""

from __future__ import annotations

import sys
import time
import types

import pytest

import captain_claw.llm as llm_mod
from captain_claw.llm import LiteLLMProvider, Message, ToolDefinition, _acompletion_tolerant, reasoning
from captain_claw.llm.reasoning import clamp, normalize_level, plan_for, profile_for, transport_for

_MSGS = [Message(role="user", content="hi")]
_TOOLS = [ToolDefinition(name="write", description="write a file", parameters={})]

DEEPSEEK = "https://api.deepseek.com"
MOONSHOT = "https://api.moonshot.ai/v1"
ZAI = "https://api.z.ai/api/paas/v4"
DASHSCOPE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
LOCAL = "http://127.0.0.1:8000/v1"
OLLAMA = "http://127.0.0.1:11434"


@pytest.fixture(autouse=True)
def _static_tables(monkeypatch):
    """No OpenRouter fetch in tests; no refusals carried between tests."""
    monkeypatch.setitem(reasoning._openrouter_state, "fetched_at", time.time())
    monkeypatch.setattr(reasoning, "_openrouter_meta", {})
    monkeypatch.setattr(reasoning, "_REFUSED", set())


def _levels(provider, model, base_url=None):
    prof = profile_for(provider, model, base_url)
    return (prof.levels, prof.default) if prof else None


# ── vocabulary ───────────────────────────────────────────────────────


def test_normalize_level_aliases():
    assert normalize_level("auto") is None and normalize_level("") is None
    assert normalize_level("none") == "off"
    assert normalize_level("Extra-High") == "xhigh"
    assert normalize_level("bogus") is None


def test_clamp_picks_nearest_and_cheaper_on_a_tie():
    prof = reasoning.Profile("x", ("low", "high", "max"), "high")
    assert clamp("medium", prof) == "low"  # tie between low and high → cheaper
    assert clamp("xhigh", prof) in ("high", "max")
    assert clamp("off", prof) == "low"
    toggle = reasoning.Profile("x", ("off", "on"), "on")
    assert clamp("max", toggle) == "on" and clamp("off", toggle) == "off"
    assert clamp("on", reasoning.Profile("x", ("low", "medium", "high"), "medium")) == "medium"


def test_transport_from_base_url():
    assert transport_for("openai", DEEPSEEK) == "deepseek"
    assert transport_for("openai", None) == "openai"
    assert transport_for("openai", LOCAL) == "local"
    assert transport_for("openrouter", OPENROUTER) == "openrouter"
    assert transport_for("ollama", OLLAMA) == "ollama"
    assert transport_for("openai", "https://generativelanguage.googleapis.com/v1beta/openai/") == "gemini-compat"
    assert transport_for("openai", "https://chatgpt.com/backend-api/codex/responses") == "codex"


# ── profiles ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", (("low", "medium", "high", "xhigh", "max"), "medium")),
    ("claude-fable-5-1", (("low", "medium", "high", "xhigh", "max"), "high")),
    ("claude-sonnet-5-5", (("off", "low", "medium", "high", "xhigh", "max"), "high")),
    ("claude-haiku-5-5", (("off", "low", "medium", "high", "xhigh", "max"), "medium")),
    ("claude-opus-4-8", (("off", "low", "medium", "high", "xhigh", "max"), "off")),
    ("claude-sonnet-4-6", (("off", "low", "medium", "high", "max"), "off")),
    ("claude-haiku-4-5-20251001", (("off", "low", "medium", "high", "max"), "off")),
    ("claude-3-5-haiku", None),
])
def test_claude_profiles(model, expected):
    assert _levels("anthropic", model) == expected


@pytest.mark.parametrize("model,expected", [
    ("gpt-5-mini", (("minimal", "low", "medium", "high"), "medium")),
    ("gpt-5.1", (("off", "low", "medium", "high"), "off")),
    ("gpt-5.4", (("off", "low", "medium", "high", "xhigh"), "off")),
    ("gpt-5.5", (("off", "low", "medium", "high", "xhigh"), "medium")),
    ("gpt-5.6-sol", (("off", "low", "medium", "high", "xhigh", "max"), "medium")),
    ("gpt-6-astra", (("low", "medium", "high", "xhigh", "max"), "medium")),
    ("gpt-5.5-pro", (("medium", "high", "xhigh"), "high")),
    ("o3", (("low", "medium", "high"), "medium")),
    ("gpt-4o", None),
    ("gpt-5-chat-latest", None),
])
def test_openai_profiles(model, expected):
    assert _levels("openai", model) == expected


@pytest.mark.parametrize("provider,model,base,expected", [
    ("openai", "deepseek-v4-flash", DEEPSEEK, (("off", "low", "high", "max"), "high")),
    ("openai", "kimi-k2.6", MOONSHOT, (("off", "on"), "on")),
    ("openai", "kimi-k3", MOONSHOT, (("low", "high", "max"), "max")),
    ("openai", "kimi-k2.7-code", MOONSHOT, None),
    ("openai", "glm-5.2", ZAI, (("off", "high", "max"), "max")),
    ("openai", "glm-5.3", ZAI, (("low", "high", "max"), "max")),
    ("openai", "qwen3.8-max", DASHSCOPE, (("off", "low", "medium", "xhigh"), "xhigh")),
    ("openai", "qwen-plus", DASHSCOPE, (("off", "low", "medium", "high", "max"), "off")),
    ("openai", "qwen-max", DASHSCOPE, None),
    ("openai", "Qwen/Qwen3-235B-A22B-Instruct-2507", LOCAL, None),
    ("openai", "mlx-community/Qwen3.6-35B-A3B-4bit", LOCAL, (("off", "on"), "on")),
    ("openai", "openai/gpt-oss-120b", LOCAL, (("low", "medium", "high"), "medium")),
    ("gemini", "gemini-2.5-pro", None, (("low", "medium", "high", "max"), "medium")),
    ("gemini", "gemini-3.6-flash", None, (("minimal", "low", "medium", "high"), "medium")),
    ("gemini", "gemini-4-argon", None, (("low", "medium", "high"), "medium")),
    ("xai", "grok-4.7", None, (("low", "medium", "high", "xhigh"), "high")),
    ("xai", "grok-4.20-0309-reasoning", None, None),
    ("ollama", "qwen3:8b", OLLAMA, (("off", "on"), "on")),
    ("ollama", "llama3.2", OLLAMA, None),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", OPENROUTER, None),
])
def test_other_profiles(provider, model, base, expected):
    assert _levels(provider, model, base) == expected


def test_openrouter_uses_its_published_levels(monkeypatch):
    monkeypatch.setattr(reasoning, "_openrouter_meta", {
        "deepseek/deepseek-v4-flash": {"mandatory": False, "supported_efforts": ["xhigh", "high", "none"],
                                       "default_effort": "high"},
        "google/gemma-4-31b-it": {"mandatory": False, "default_enabled": False},
        "minimax/minimax-m2.7": {"mandatory": True},
        "meta-llama/llama-3.3-70b-instruct": {"none": True},
    })
    assert _levels("openrouter", "deepseek/deepseek-v4-flash", OPENROUTER) == (("off", "high", "xhigh"), "high")
    assert _levels("openrouter", "google/gemma-4-31b-it:free", OPENROUTER) == (("off", "on"), "off")
    assert profile_for("openrouter", "minimax/minimax-m2.7", OPENROUTER) is None
    plan = plan_for("openrouter", "deepseek/deepseek-v4-flash", OPENROUTER, "max")
    assert plan.body == {"reasoning": {"effort": "xhigh"}}
    assert plan_for("openrouter", "google/gemma-4-31b-it", OPENROUTER, "off").body == {"reasoning": {"effort": "none"}}


# ── request fields ───────────────────────────────────────────────────


def _kwargs(provider, model, level, base_url=None, **kw):
    p = LiteLLMProvider(provider=provider, model=model, api_key="x", base_url=base_url)
    with reasoning.call_effort(level):
        return p._request_kwargs(_MSGS, tools=kw.get("tools"))


def test_claude_adaptive_effort_is_top_level():
    kw = _kwargs("anthropic", "claude-opus-5-5", "max")
    assert kw["output_config"] == {"effort": "max"}
    assert "thinking" not in kw and "extra_body" not in kw and "temperature" not in kw


def test_claude_off_modes():
    assert _kwargs("anthropic", "claude-sonnet-5-5", "off")["thinking"] == {"type": "between_tools"}
    assert _kwargs("anthropic", "claude-opus-5", "off")["thinking"] == {"type": "disabled"}
    kw = _kwargs("anthropic", "claude-opus-4-8", "off")
    assert "thinking" not in kw and "output_config" not in kw


def test_claude_explicit_adaptive_and_budget():
    kw = _kwargs("anthropic", "claude-opus-4-8", "high")
    assert kw["thinking"] == {"type": "adaptive"} and kw["output_config"] == {"effort": "high"}
    kw = _kwargs("anthropic", "claude-haiku-4-5", "medium")
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 8192}


def test_claude_budget_skipped_mid_tool_loop():
    p = LiteLLMProvider(provider="anthropic", model="claude-haiku-4-5", api_key="x")
    msgs = _MSGS + [
        Message(role="assistant", content="", tool_calls=[{"id": "1", "type": "function",
                                                           "function": {"name": "write", "arguments": "{}"}}]),
        Message(role="tool", content="ok", tool_call_id="1", tool_name="write"),
    ]
    with reasoning.call_effort("high"):
        kw = p._request_kwargs(msgs, tools=_TOOLS)
    assert "thinking" not in kw


def test_openai_effort_in_body_bypasses_litellm_map():
    kw = _kwargs("openai", "gpt-5.6-sol", "xhigh")
    assert kw["extra_body"]["reasoning_effort"] == "xhigh"
    assert "temperature" not in kw


def test_tools_refusing_model_still_gets_none(monkeypatch):
    monkeypatch.setattr(llm_mod, "_REASONING_OFF_WITH_TOOLS_MODELS", {"gpt-6-luna"})
    kw = _kwargs("openai", "gpt-6-luna", "high", tools=_TOOLS)
    assert kw["reasoning_effort"] == "none" and kw["extra_body"]["reasoning_effort"] == "none"


def test_deepseek_on_openai_base_url():
    kw = _kwargs("openai", "deepseek-v4-flash", "max", DEEPSEEK)
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}
    kw = _kwargs("openai", "deepseek-v4-flash", "off", DEEPSEEK)
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}


def test_kimi_drops_fixed_temperature():
    kw = _kwargs("openai", "kimi-k2.6", "off", MOONSHOT)
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}} and "temperature" not in kw


def test_qwen_dashscope_budget_and_38_effort():
    assert _kwargs("openai", "qwen3.6-plus", "low", DASHSCOPE)["extra_body"] == {
        "enable_thinking": True, "thinking_budget": 2048}
    assert _kwargs("openai", "qwen3.6-plus", "max", DASHSCOPE)["extra_body"] == {"enable_thinking": True}
    assert _kwargs("openai", "qwen3.8-max", "high", DASHSCOPE)["extra_body"] in (
        {"reasoning_effort": "medium"}, {"reasoning_effort": "xhigh"})


def test_local_templates():
    assert _kwargs("openai", "zai-org/GLM-5.1-FP8", "off", LOCAL)["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}
    assert _kwargs("openai", "Qwen/Qwen3.8-Flash-Next", "medium", LOCAL)["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium"}}
    # A Flight Deck setup read ("off") on a self-hosted Qwen 3.8, which thinks at xhigh unless told.
    assert _kwargs("openai", "LithosAI/Qwen3.8-27B-DSpark-NVFP4", "off", LOCAL)["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_gemini_thinking_config_passthrough():
    assert _kwargs("gemini", "gemini-3.6-flash", "minimal")["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert _kwargs("gemini", "gemini-2.5-flash", "off")["thinkingConfig"] == {"thinkingBudget": 0}
    assert _kwargs("gemini", "gemini-2.5-pro", "low")["thinkingConfig"] == {"thinkingBudget": 1024}


def test_auto_sends_nothing_and_suffix_still_works():
    kw = _kwargs("openai", "gpt-5.5", None)
    assert "extra_body" not in kw and "reasoning_effort" not in kw
    p = LiteLLMProvider(provider="openai", model="gpt-5.5-high", api_key="x")
    assert p._request_kwargs(_MSGS)["reasoning_effort"] == "high"


# ── scope ────────────────────────────────────────────────────────────


def test_turn_effort_needs_a_call_to_open_it():
    p = LiteLLMProvider(provider="anthropic", model="claude-opus-5-5", api_key="x")
    with reasoning.turn_effort("max"):
        assert "output_config" not in p._request_kwargs(_MSGS)  # housekeeping call
        with reasoning.call_effort(None):
            assert "output_config" not in p._request_kwargs(_MSGS)
        with reasoning.call_effort(reasoning.current_turn_effort()):
            assert p._request_kwargs(_MSGS)["output_config"] == {"effort": "max"}
    assert reasoning.current_turn_effort() is None  # turn over


def test_guard_exempt_labels_cover_housekeeping():
    from captain_claw.agent_guard_mixin import _REASONING_EXEMPT_LABELS

    assert {"task_rephrase", "compaction_summary", "session_title", "reflection"} <= _REASONING_EXEMPT_LABELS
    assert "turn_1" not in _REASONING_EXEMPT_LABELS and "conversation" not in _REASONING_EXEMPT_LABELS


# ── heal ─────────────────────────────────────────────────────────────


async def test_refused_reasoning_is_stripped_and_remembered(monkeypatch):
    seen: list[dict] = []

    async def fake_acompletion(**kwargs):
        seen.append(kwargs)
        if "chat_template_kwargs" in (kwargs.get("extra_body") or {}):
            raise Exception("BadRequestError: Unrecognized request argument supplied: chat_template_kwargs")
        return "ok"

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=fake_acompletion))
    kw = _kwargs("openai", "zai-org/GLM-5.1-FP8", "off", LOCAL)
    assert await _acompletion_tolerant(kw) == "ok"
    assert "extra_body" not in seen[-1]
    assert plan_for("openai", "zai-org/GLM-5.1-FP8", LOCAL, "off") is None


def test_classifier_leaves_other_heals_alone():
    refused = llm_mod._is_reasoning_field_refused_error
    assert refused("reasoning_effort must be one of low, medium, high")
    assert refused("output_config.effort: invalid value 'xhigh'")
    assert not refused("function tools with reasoning_effort are not supported for gpt-6-luna "
                       "in /v1/chat/completions. set reasoning_effort to 'none'")
    assert not refused("the reasoning_content in the thinking mode must be passed back to the api")
    assert not refused("rate limit exceeded")
