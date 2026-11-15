"""Housekeeping calls run with thinking off where the model can switch it off.

A thinking model on a self-hosted server (Qwen 3.8 on llama.cpp) counts its
thinking against the output budget. The agent's own small calls — a task name
(25 tokens), next-step pills (600), a reflection (1500), insight extraction
(1000), lesson proposals (400) — ran out while still thinking: the task name
came back empty, the pills and insights didn't parse, and the model's draft
notes were saved as its self-reflection. These calls now ask for thinking off
(``reasoning.quiet_call`` / ``reasoning.quiet_plan``); a model that can't
switch it off, or doesn't think unless asked, gets a request that says nothing.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

from captain_claw import reflections
from captain_claw.llm import LiteLLMProvider, LLMProvider, LLMResponse, Message, reasoning

_MSGS = [Message(role="user", content="hi")]
LOCAL = "http://127.0.0.1:8080/v1"


@pytest.fixture(autouse=True)
def _static_tables(monkeypatch):
    """No OpenRouter fetch in tests; no refusals carried between tests."""
    monkeypatch.setitem(reasoning._openrouter_state, "fetched_at", time.time())
    monkeypatch.setattr(reasoning, "_openrouter_meta", {})
    monkeypatch.setattr(reasoning, "_REFUSED", set())


def _quiet_kwargs(provider: str, model: str, base_url: str | None = None) -> dict:
    p = LiteLLMProvider(provider=provider, model=model, api_key="x", base_url=base_url)
    with reasoning.quiet_call():
        return p._request_kwargs(_MSGS)


# ── the plan ─────────────────────────────────────────────────────────


def test_a_self_hosted_qwen_38_gets_thinking_off():
    assert _quiet_kwargs("openai", "Qwen3.8-flash-next", LOCAL)["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}


def test_other_switchable_models_get_their_own_off():
    assert _quiet_kwargs("gemini", "gemini-2.5-flash")["thinkingConfig"] == {"thinkingBudget": 0}
    assert _quiet_kwargs("anthropic", "claude-opus-5")["thinking"] == {"type": "disabled"}


def test_a_model_that_cannot_switch_thinking_off_gets_nothing():
    kw = _quiet_kwargs("anthropic", "claude-opus-5-5")
    assert "thinking" not in kw and "output_config" not in kw


def test_a_model_that_thinks_only_when_asked_gets_nothing():
    # Thinking is already off by default: the request stays as it was.
    assert reasoning.quiet_plan("anthropic", "claude-opus-4-8", None) is None
    assert "extra_body" not in _quiet_kwargs("openai", "qwen-plus",
                                              "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")


def test_an_unknown_model_or_a_refusing_server_gets_nothing():
    assert reasoning.quiet_plan("openai", "some-house-model", LOCAL) is None
    reasoning.remember_refused("Qwen3.8-flash-next")
    assert "extra_body" not in _quiet_kwargs("openai", "Qwen3.8-flash-next", LOCAL)


def test_the_quiet_scope_ends_with_the_call():
    p = LiteLLMProvider(provider="openai", model="Qwen3.8-flash-next", api_key="x", base_url=LOCAL)
    with reasoning.quiet_call():
        pass
    assert "extra_body" not in p._request_kwargs(_MSGS)


# ── the callers ──────────────────────────────────────────────────────


class _EffortSpy(LLMProvider):
    provider = "openai"
    model = "stub"

    def __init__(self, content: str = "[]"):
        self.content = content
        self.seen: list[str | None] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.seen.append(reasoning.requested_effort())
        return LLMResponse(content=self.content)

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return 1


@pytest.mark.asyncio
async def test_guarded_housekeeping_labels_run_quiet():
    from captain_claw.agent import Agent
    from captain_claw.session import Session

    spy = _EffortSpy("- a reflection")
    agent = Agent(provider=spy)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    for label in ("reflection", "insight_extraction", "lesson_proposal", "conversation_topics"):
        await agent._complete_with_guards(messages=_MSGS, tools=None, interaction_label=label, max_tokens=100)
    assert spy.seen == [reasoning.QUIET] * 4


@pytest.mark.asyncio
async def test_next_steps_extraction_runs_quiet():
    from captain_claw.next_steps import extract_next_steps

    spy = _EffortSpy('[{"label": "Send it", "action": "send the draft"}]')
    reply = ("The follow-up draft for the five quiet leads is ready in the Drafts folder.\n\n"
             "Options:\n1. Send the draft\n2. Edit the subject line first")
    steps = await extract_next_steps(spy, reply)
    assert spy.seen == [reasoning.QUIET]
    assert [s.label for s in steps] == ["Send it"]


@pytest.mark.asyncio
async def test_task_naming_asks_for_thinking_off(monkeypatch):
    from captain_claw.web import chat_handler

    calls: list[dict] = []

    async def _acompletion(**kwargs):
        calls.append(kwargs)
        msg = types.SimpleNamespace(content="Outreach batch follow-up")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=_acompletion))
    name = await chat_handler._generate_task_name(
        "check the outreach replies", [], "openai/Qwen3.8-flash-next", api_key="x", base_url=LOCAL)

    assert name == "Outreach batch follow-up"
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_task_naming_retries_without_the_switch_a_server_refuses(monkeypatch):
    from captain_claw.web import chat_handler

    calls: list[dict] = []

    async def _acompletion(**kwargs):
        calls.append(dict(kwargs))
        if "extra_body" in kwargs:
            raise RuntimeError("400 Bad Request: chat_template_kwargs is not permitted (extra inputs)")
        msg = types.SimpleNamespace(content="Outreach replies")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=_acompletion))
    name = await chat_handler._generate_task_name(
        "check the outreach replies", [], "openai/Qwen3.8-flash-next", api_key="x", base_url=LOCAL)

    assert name == "Outreach replies"
    assert len(calls) == 2 and "extra_body" not in calls[1]
    assert reasoning.was_refused("Qwen3.8-flash-next")


# ── a reflection cut off at the limit is not saved ───────────────────


def _reflection_agent(response):
    async def _complete(messages, tools=None, interaction_label="", max_tokens=None):
        return response

    class _Instructions:
        def load(self, name):
            return (Path(reflections.__file__).parent / "instructions" / name).read_text(encoding="utf-8")

        def render(self, name, **values):
            return values["recent_messages"]

    return types.SimpleNamespace(
        session=types.SimpleNamespace(id="s1", messages=[{"role": "user", "content": "check the inbox"}]),
        instructions=_Instructions(),
        provider=types.SimpleNamespace(provider="stub", model="stub"),
        _complete_with_guards=_complete,
    )


class _NoUsage:
    async def record_llm_usage(self, **kwargs):
        return None


@pytest.mark.asyncio
async def test_a_cut_off_reflection_keeps_the_previous_one(monkeypatch):
    saved: list = []
    monkeypatch.setattr(reflections, "load_latest_reflection", lambda: None)
    monkeypatch.setattr(reflections, "save_reflection", saved.append)
    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: _NoUsage())
    draft = LLMResponse(content="Potential final 12 bullets:\n- Keep monitoring", finish_reason="length")

    with pytest.raises(RuntimeError, match="cut off"):
        await reflections.generate_reflection(_reflection_agent(draft))
    assert saved == []


@pytest.mark.asyncio
async def test_a_finished_reflection_is_saved(monkeypatch):
    saved: list = []
    monkeypatch.setattr(reflections, "load_latest_reflection", lambda: None)
    monkeypatch.setattr(reflections, "save_reflection", saved.append)
    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: _NoUsage())
    done = LLMResponse(content="- Check the live inbox before reporting.", finish_reason="stop")

    reflection = await reflections.generate_reflection(_reflection_agent(done))
    assert reflection.summary == "- Check the live inbox before reporting."
    assert len(saved) == 1
