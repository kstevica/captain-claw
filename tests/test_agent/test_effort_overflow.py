"""A main call whose thinking used the whole output budget and gave nothing
back (no text, no tool call) is run again one effort level lower — not forced
to call a tool, which only buys a quick call before the next one runs out the
same way. Seen with Haiku 5.5 at a locked max effort on a benchmark task."""

from __future__ import annotations

import pytest

from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, reasoning
from captain_claw.session import Session
from captain_claw.tools.registry import ToolRegistry

OUT_OF_ROOM = LLMResponse(content="", finish_reason="length")


class _Claude(LLMProvider):
    """A scripted adaptive-thinking Claude that notes each call's effort."""

    provider = "anthropic"
    model = "claude-haiku-5-5"
    base_url = None

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[list] = []
        self.efforts: list[str | None] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append(list(messages))
        self.efforts.append(reasoning.requested_effort())
        return self.responses.pop(0) if self.responses else LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text):
        return len(str(text).split()) or 1


class _SessionManager:
    async def save_session(self, session):
        return None


def _agent(provider: LLMProvider) -> Agent:
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = _SessionManager()
    agent.tools = ToolRegistry()
    agent._build_env_now_text = lambda: ""
    return agent


@pytest.fixture(autouse=True)
def _no_lock(monkeypatch):
    monkeypatch.delenv(reasoning.LOCK_ENV, raising=False)


def _main_efforts(provider: _Claude) -> list[str | None]:
    """The efforts of the main conversation calls (housekeeping calls run
    quiet or at the model's default)."""
    return [e for e in provider.efforts if e not in (None, reasoning.QUIET)]


@pytest.mark.asyncio
async def test_an_empty_out_of_room_call_is_retried_one_level_lower(monkeypatch):
    monkeypatch.setenv(reasoning.LOCK_ENV, "max")
    provider = _Claude([OUT_OF_ROOM, LLMResponse(content="Scheduled 11 of 15 meetings.", finish_reason="stop")])
    agent = _agent(provider)

    result = await agent.complete("schedule the meetings")

    assert result.startswith("Scheduled 11 of 15 meetings.")
    assert _main_efforts(provider)[:2] == ["max", "xhigh"]
    # Not treated as a stall: no "your last response was empty" scolding, no forced tool.
    assert not any("completely empty" in str(m.content) for m in provider.calls[1])
    assert getattr(provider, "_tool_choice_override", None) is None


@pytest.mark.asyncio
async def test_each_overflow_steps_down_again_and_a_new_turn_starts_over(monkeypatch):
    monkeypatch.setenv(reasoning.LOCK_ENV, "max")
    provider = _Claude([OUT_OF_ROOM, OUT_OF_ROOM, LLMResponse(content="Done.", finish_reason="stop")])
    agent = _agent(provider)

    await agent.complete("first task")
    assert _main_efforts(provider)[:3] == ["max", "xhigh", "high"]

    provider.efforts.clear()
    provider.responses = [LLMResponse(content="Second answer.", finish_reason="stop")]
    await agent.complete("second task")
    assert _main_efforts(provider)[0] == "max"


@pytest.mark.asyncio
async def test_with_no_lower_level_it_is_an_ordinary_empty_reply(monkeypatch):
    monkeypatch.setenv(reasoning.LOCK_ENV, "low")   # Haiku 5.5's lowest thinking level
    provider = _Claude([OUT_OF_ROOM, LLMResponse(content="Answer.", finish_reason="stop")])
    agent = _agent(provider)

    await agent.complete("quick question")

    assert _main_efforts(provider)[:2] == ["low", "low"]
    assert any("completely empty" in str(m.content) for m in provider.calls[1])


@pytest.mark.asyncio
async def test_a_cut_off_reply_with_text_is_still_continued_not_stepped_down(monkeypatch):
    monkeypatch.setenv(reasoning.LOCK_ENV, "max")
    provider = _Claude([
        LLMResponse(content="The schedule keeps ", finish_reason="length"),
        LLMResponse(content="all 30 events.", finish_reason="stop"),
    ])
    agent = _agent(provider)

    result = await agent.complete("write the schedule")

    assert result.startswith("The schedule keeps all 30 events.")
    assert _main_efforts(provider)[:2] == ["max", "max"]
