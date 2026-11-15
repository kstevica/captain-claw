"""A thank-you gets its answer, not a forced tool call.

A local model running an outbound-mail agent answered "ok, thank you" with
"You're welcome — the machine runs itself from here: follow-up pass and the
new-capped batch tomorrow morning, reply-watch every 3h. I'll report as things
land." The closing-sentence stall check read the short reply's last sentence,
"I'll …", as an announcement of work, hid the reply and forced a tool call —
and the agent answered the thanks with an inbox check nobody asked for.

Now a closing promise to report later ("I'll report as things land", "I'll
let you know when replies come in", "I'll keep you posted") signs a reply off
instead of making it a stall, and a reply to a message that only thanks is
never forced into a tool call (an empty one still gets its retry). Mailboxes
and names here are made up.
"""

from __future__ import annotations

from typing import Any

import pytest

from captain_claw.agent import Agent
from captain_claw.agent_orchestration_mixin import _looks_like_stall, _only_thanks
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

_SEEN_LIVE = (
    "You're welcome — the machine runs itself from here: follow-up pass and the new-capped "
    "batch tomorrow morning, reply-watch every 3h. I'll report as things land."
)

# ── the detector ────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    _SEEN_LIVE,
    "I'll report as things land.",
    "Glad it helped. I'll let you know when replies come in.",
    "Done — both jobs are scheduled. I'll keep you posted.",
    "Saved. I'll ping you if anything changes.",
    "All set for Monday. I'll report back tomorrow after the 10:30 batch.",
    "You're welcome! I’ll report as soon as they reply.",
])
def test_a_report_later_sign_off_is_not_a_stall(text):
    assert not _looks_like_stall(text)


@pytest.mark.parametrize("text", [
    # Work for now, with a report to follow: still a stall.
    "I'll report back once I've checked the inbox.",
    "Great data. I'll keep you posted while I dig through the logs.",
    "I'll keep you posted. Checking the live inbox now.",
    "Noted. I'll check the inbox and report as things land.",
    "Let me pull the lead table first.",
])
def test_work_announced_for_now_is_still_a_stall(text):
    assert _looks_like_stall(text)


@pytest.mark.parametrize("text", [
    "ok, thank you", "Thanks!", "thank you so much", "ok thanks a lot", "hvala", "Perfect, thanks.",
    "that's all for now, thanks",
])
def test_messages_that_only_thank(text):
    assert _only_thanks(text)


@pytest.mark.parametrize("text", [
    "ok", "great", "yes", "thanks, now send the batch", "thank you — what's left?",
    "", "Thanks for the list. Please draft the follow-ups for the five quiet leads now.",
])
def test_messages_that_ask_for_something(text):
    assert not _only_thanks(text)


# ── the loop ────────────────────────────────────────────────────────


class _DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class _ScriptedProvider(LLMProvider):
    provider = "openai"
    model = "stub"

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.tool_choice_seen: list[Any] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.tool_choice_seen.append(getattr(self, "_tool_choice_override", None))
        self._tool_choice_override = None
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


class _Inbox(Tool):
    name = "imap_mail"
    description = "Mailbox"
    parameters = {"type": "object", "properties": {"action": {"type": "string"}}}

    def __init__(self):
        self.calls = 0

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, content="10 messages, none new")


def _agent(responses: list[LLMResponse]) -> tuple[Agent, _ScriptedProvider, _Inbox]:
    provider = _ScriptedProvider(responses)
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = _DummySessionManager()
    inbox = _Inbox()
    registry = ToolRegistry()
    registry.register(inbox)
    agent.tools = registry
    agent._build_env_now_text = lambda: ""
    return agent, provider, inbox


@pytest.fixture
def _no_worker_env(monkeypatch: pytest.MonkeyPatch):
    for name in ("CLAW_BEING_WORKER", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER",
                 "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT", "CLAW_BAT_WORKER"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_the_thank_you_reply_seen_live_is_the_answer(_no_worker_env):
    agent, provider, inbox = _agent([LLMResponse(content=_SEEN_LIVE)])

    result = await agent.complete("ok, thank you")

    assert result == _SEEN_LIVE
    assert provider.tool_choice_seen == [None]
    assert inbox.calls == 0


@pytest.mark.asyncio
async def test_a_short_reply_to_thanks_is_never_forced_into_a_tool(_no_worker_env):
    reply = "Anytime. I'll check the inbox."
    agent, provider, inbox = _agent([LLMResponse(content=reply)])

    result = await agent.complete("thanks!")

    assert result == reply
    assert provider.tool_choice_seen == [None]
    assert inbox.calls == 0


@pytest.mark.asyncio
async def test_the_same_reply_to_a_request_is_still_retried(_no_worker_env):
    agent, provider, inbox = _agent([
        LLMResponse(content="Anytime. I'll check the inbox."),
        LLMResponse(content="", tool_calls=[
            ToolCall(id="c1", name="imap_mail", arguments={"action": "list_messages"})]),
        LLMResponse(content="Inbox: 10 messages, none new since the last check."),
    ])

    result = await agent.complete("check the inbox for replies")

    assert result.startswith("Inbox:")
    assert inbox.calls == 1
    assert provider.tool_choice_seen == [None, "required", None]


@pytest.mark.asyncio
async def test_an_empty_reply_to_thanks_still_gets_its_retry(_no_worker_env):
    agent, provider, inbox = _agent([
        LLMResponse(content=""),
        LLMResponse(content="You're welcome."),
    ])

    result = await agent.complete("ok, thank you")

    assert result == "You're welcome."
    assert len(provider.tool_choice_seen) == 2
