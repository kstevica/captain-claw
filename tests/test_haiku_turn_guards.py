"""Turn-level guards that misfired on Claude Haiku (glasses staging, 2026-10-22/23):
eco mode hid every memory tool ("I don't have a memory tool in this session"),
the stall check forced a tool call on a clarifying question, and task naming
sent a temperature Haiku 5.5 rejects on every turn."""

from __future__ import annotations

import types

import pytest

from captain_claw import llm
from captain_claw.agent_orchestration_mixin import (
    _ECO_CORE_TOOLS,
    _eco_select_tools_by_intent,
    _looks_like_stall,
)
from captain_claw.web import chat_handler


def test_insights_schema_is_always_sent():
    assert "insights" in _ECO_CORE_TOOLS


@pytest.mark.parametrize(
    "message",
    [
        "Martina je moja supruga, zapamti to.",
        "Upamti da je sastanak u petak.",
        "remember that I prefer short answers",
        "Okay, so what's your memory capacity?",
        "Sjećaš li se što smo pričali o Svenu?",
    ],
)
def test_memory_requests_surface_the_memory_tools(message):
    matched = _eco_select_tools_by_intent(message)
    assert {"personality", "typesense", "topics"} <= matched


def test_unrelated_messages_do_not_surface_memory_tools():
    assert "typesense" not in _eco_select_tools_by_intent("what's the weather in Zagreb")


@pytest.mark.parametrize(
    "text",
    [
        # The staging reply that was forced into a calendar write.
        "Got it: **Tue 9 Feb 2027**. What time, and should it be all day? "
        "I'll set the location to Lobby.",
        "I'll need the date first — which day?",
        "Found two Svens. Which one do you mean? I'll look him up then.",
    ],
)
def test_a_question_to_the_user_is_not_a_stall(text):
    assert _looks_like_stall(text) is False


def test_a_url_with_a_query_string_is_still_a_stall():
    assert _looks_like_stall("Let me fetch https://ex.com/a?b=1 now.") is True


class _TempRejectedError(Exception):
    pass


@pytest.fixture
def fake_litellm(monkeypatch):
    calls: list[dict] = []

    async def acompletion(**kwargs):
        calls.append(dict(kwargs))
        if "temperature" in kwargs and "haiku" in kwargs["model"]:
            raise _TempRejectedError(
                'AnthropicException - {"type":"invalid_request_error",'
                '"message":"`temperature` is deprecated for this model."}')
        msg = types.SimpleNamespace(content="Battery report")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    import litellm
    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(llm, "_TEMPERATURE_UNSUPPORTED_MODELS", set())
    return calls


@pytest.mark.asyncio
async def test_task_naming_drops_a_rejected_temperature_and_remembers(fake_litellm):
    # A model the table says takes a temperature, rejecting it anyway.
    name = await chat_handler._generate_task_name(
        "Napravi report o bateriji", [], "anthropic/claude-haiku-4-5", api_key="k")
    assert name == "Battery report"
    assert [("temperature" in c) for c in fake_litellm] == [True, False]

    fake_litellm.clear()
    await chat_handler._generate_task_name(
        "Koliko je sati?", [], "anthropic/claude-haiku-4-5", api_key="k")
    assert [("temperature" in c) for c in fake_litellm] == [False]


@pytest.mark.asyncio
async def test_task_naming_sends_no_temperature_to_a_model_known_to_refuse_it(fake_litellm):
    """Haiku 5.5 is known up front (llm._claude_without_sampling): no failed first call."""
    name = await chat_handler._generate_task_name(
        "Napravi report o bateriji", [], "anthropic/claude-haiku-5-5", api_key="k")
    assert name == "Battery report"
    assert [("temperature" in c) for c in fake_litellm] == [False]


@pytest.mark.asyncio
async def test_task_naming_keeps_temperature_where_it_works(fake_litellm):
    await chat_handler._generate_task_name(
        "Check emails", [], "openai/deepseek-flash", api_key="k")
    assert fake_litellm[0]["temperature"] == 0.0
