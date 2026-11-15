"""Replies that only announce work with a gerund are stalls.

A local model running an outbound-mail agent answered scheduled runs with a
single line — "Running the reply-watch — checking the live inbox now." — and
no tool call, run after run. The stall check only knew "let me / I'll /
starting" openers, and the narrated-lookup check only sees a progress verb
at the start of a sentence, so the line was finalized as the answer. Every
such reply left in the agent's one long chat session taught it to answer the
next run the same way.

These replies now get the "You announced intent without acting" retry with a
forced tool call — on scheduled (cron) turns as well as chat turns — while
nouns ("Nothing new", "Pricing is …"), statements about the work and real
answers stay answers. Names and mailboxes here are made up.
"""

from __future__ import annotations

from typing import Any

import pytest

from captain_claw import mail_authority
from captain_claw.agent import Agent
from captain_claw.agent_orchestration_mixin import _announces_work, _looks_like_stall
from captain_claw.cron_dispatch import wrap_cron_prompt
from captain_claw.fabrication_guard import narrated_lookups
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

# ── the detector ────────────────────────────────────────────────────

# The shapes the scheduled runs (and one chat turn) ended with.
_SEEN_LIVE = [
    "Sending the batch now — pulling the live queue, the inbox, and the Sent folder first.",
    "Running the reply-watch — checking the live inbox now.",
    "Running the follow-up pass — live table and mailbox first.",
    "Running the reply-watch — live inbox, drafts, and lead records first.",
    "Running the follow-up pass — querying the live table and mailbox now.",
    "Running the reply-watch — checking the live inbox and Mira Vale's lead record now.",
    "Running the reply-watch — checking the live inbox.",
    "Running the reply-watch — live inbox check first.",
    "Running the scheduled reply-watch.",
    "Running the reply-watch.",
]


@pytest.mark.parametrize("text", _SEEN_LIVE)
def test_the_live_announcements_are_stalls(text):
    assert _looks_like_stall(text) is True


@pytest.mark.parametrize("text", [
    "Running the reply-watch — checking the live inbox now.",
    "Sending the batch now — pulling the live queue, the inbox, and the Sent folder first.",
])
def test_the_lookup_check_does_not_see_them(text):
    """Why the stall check has to: the progress verb is not at a sentence
    start, so the narrated-lookup check (which runs first) lets these by."""
    assert narrated_lookups(text) == ()


@pytest.mark.parametrize("text", [
    # An opening report, then the announcement.
    "New mail detected in the inbox — pulling the messages first.",
    # An acknowledgement, then the announcement.
    "You're right — running the check now.",
    "You’re right — running the check now.",
    "Authorized — running the batch now.",
    "Sure — running it now.",
    "Doing it now.",
    "Now updating the lead table.",
    "No new mail since 09:00 — updating the lead table now.",
    # Markdown and emoji around it.
    "**Running** the reply-watch — checking the inbox now.",
    "\U0001F504 Running the reply-watch now.",
    "- Pulling the live queue first.",
])
def test_one_sentence_announcements_are_stalls(text):
    assert _looks_like_stall(text) is True


@pytest.mark.parametrize("text", [
    # A short paragraph after the announcement that delivers nothing.
    "New mail detected in the inbox — pulling the messages first.\n\nBack with a summary shortly.",
    # A short paragraph that announces more work.
    "You're right — running the check now.\n\nPulling the latest message and the lead record first.",
    "You're right — the lookups didn't happen. Running them now.",
    "Batch run at 10:30. Pulling the eligible queue and the pre-written drafts first.\n\n"
    "One email came back malformed — re-pulling the queue cleanly.",
    "Running the scheduled reply-watch.\n\nChecking the inbox for anything new since the last check.",
])
def test_short_multi_paragraph_announcements_are_stalls(text):
    assert _looks_like_stall(text) is True


@pytest.mark.parametrize("text", [
    # Leads into an answer.
    "Looking at the numbers, revenue grew 12% quarter over quarter.",
    "Checking the logs, I see the job failed at 03:00.",
    "Reading the room, I'd hold off on the follow-up.",
    # Nouns, not work.
    "Nothing new in the inbox.",
    "Pricing is €49 per seat per month.",
    "Meeting moved to 3pm.",
    "Booking confirmed for Tuesday 10:00.",
    "Building 4 is closed on Fridays.",
    "Running total: 42 leads contacted.",
    "Finding: the job failed because the token expired.",
    # Statuses and statements about the work.
    "Processing complete: 3 of 3 items processed.",
    "Testing passed on all 12 cases.",
    "Sent 12 of 12. Updating the lead table was the slow part.",
    "Searching the archive turned up nothing older than March.",
    # Idioms.
    "Checking in: everything is on track.",
    "Running late — back at 3pm.",
    "Getting there — 8 of 12 done.",
    # Work set for later or for a condition.
    "No new mail — checking again in 3 hours.",
    "Sending it once you confirm.",
    # A report under a job name.
    "Reply-watch at 11:49: no new mail since the last check.",
    "Running the reply-watch — no new mail since the last check.",
    "Done — the batch went out and the lead table is updated.",
    # Hand-backs and questions give control back to the user.
    "Updating the table now. Let me know if you want the CSV.",
    "Sending the batch now — want me to include the drafts?",
])
def test_answers_are_not_stalls(text):
    assert _looks_like_stall(text) is False


def test_an_announcement_followed_by_the_result_is_an_answer():
    """Whatever follows the opening announcement is what the check reads: a
    result of 40+ characters makes it an answer (true or not — made-up
    results are the fabrication check's)."""
    text = ("New mail detected in the inbox — pulling the messages first.\n\n"
            "One new message: Ana Petrić replied at 02:16 and asked for a demo.")
    assert _looks_like_stall(text) is False


def test_long_replies_are_not_judged():
    text = ("Running the reply-watch — checking the live inbox now. "
            + "The inbox holds the same five messages as at the last check, all handled. " * 3)
    assert len(text) > 200
    assert _announces_work(text) is False


def test_a_long_reply_closing_on_an_announcement_is_an_answer():
    """As for "I'll …": only a short reply that closes on an announcement is a stall."""
    text = ("Reply-watch at 14:47: no new mail since the last check. The two inbox "
            "messages are the ones already handled, and none of the five leads replied yet. "
            "Updating the log now.")
    assert 160 < len(text) <= 200
    assert _looks_like_stall(text) is False


# ── the loop: retried with a forced tool, on cron turns too ──────────


class _DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class _ScriptedProvider(LLMProvider):
    """Answers in order and records the tool_choice override each call saw
    (consumed, as the real providers consume it)."""

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


class _LeadTable(Tool):
    name = "datastore"
    description = "Lead table"
    parameters = {"type": "object", "properties": {"action": {"type": "string"}}}

    def __init__(self):
        self.calls = 0

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, content="| org | status |\n| Example Labs | contacted |")


def _agent(responses: list[LLMResponse]) -> tuple[Agent, _ScriptedProvider, _LeadTable]:
    provider = _ScriptedProvider(responses)
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="s1", name="default")
    agent.session_manager = _DummySessionManager()
    table = _LeadTable()
    registry = ToolRegistry()
    registry.register(table)
    agent.tools = registry
    agent._build_env_now_text = lambda: ""
    return agent, provider, table


@pytest.fixture
def _no_worker_env(monkeypatch: pytest.MonkeyPatch):
    for name in ("CLAW_BEING_WORKER", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER",
                 "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT", "CLAW_BAT_WORKER"):
        monkeypatch.delenv(name, raising=False)


_STALL_LEAD = "You announced intent without acting."
_TASK = "Follow-up pass: query the leads table for leads due a follow-up and report."


def _stall_correctives(agent: Agent) -> list[str]:
    return [m["content"] for m in agent.session.messages
            if m["role"] == "user" and str(m.get("content", "")).startswith(_STALL_LEAD)]


def _rejected(agent: Agent) -> list[str]:
    return [m["content"] for m in agent.session.messages
            if m["role"] == "assistant" and m.get("origin") == "rejected"]


@pytest.mark.asyncio
async def test_a_cron_turn_that_only_announces_work_gets_a_forced_tool_retry(_no_worker_env):
    stall = "Running the follow-up pass — live table and mailbox first."
    agent, provider, table = _agent([
        LLMResponse(content=stall),
        LLMResponse(content="", tool_calls=[
            ToolCall(id="c1", name="datastore", arguments={"action": "query"})]),
        LLMResponse(content="Follow-up pass: no lead is due a follow-up today."),
    ])

    with mail_authority.bound(mail_authority.automated("cron", _TASK, "intent")):
        result = await agent.complete(wrap_cron_prompt("6bed101c-job", _TASK), turn_origin="cron")

    assert result == "Follow-up pass: no lead is due a follow-up today."
    assert table.calls == 1
    # The retry call was forced to use a tool; the others were not.
    assert provider.tool_choice_seen == [None, "required", None]
    assert len(_stall_correctives(agent)) == 1
    # The announcement stays out of what later runs learn from.
    assert _rejected(agent) == [stall]
    assert all(m.get("content") != stall for m in agent.session.messages
               if m["role"] == "assistant" and m.get("origin") != "rejected")


@pytest.mark.asyncio
async def test_a_chat_turn_that_only_announces_work_gets_the_same_retry(_no_worker_env):
    stall = "Sending the batch now — pulling the live queue, the inbox, and the Sent folder first."
    agent, provider, table = _agent([
        LLMResponse(content=stall),
        LLMResponse(content="", tool_calls=[
            ToolCall(id="c1", name="datastore", arguments={"action": "query"})]),
        LLMResponse(content="Noted — the 0.9.1 points are saved for the next batch."),
    ])

    result = await agent.complete("remember these release notes for the next outreach batch")

    assert result.startswith("Noted")
    assert table.calls == 1
    assert provider.tool_choice_seen == [None, "required", None]
    assert len(_stall_correctives(agent)) == 1


@pytest.mark.asyncio
async def test_a_cron_turn_with_a_short_real_answer_is_not_retried(_no_worker_env):
    answer = "Reply-watch at 09:00: no new mail since the last check."
    agent, provider, table = _agent([LLMResponse(content=answer)])

    with mail_authority.bound(mail_authority.automated("cron", _TASK, "intent")):
        result = await agent.complete(wrap_cron_prompt("024f96c6-job", _TASK), turn_origin="cron")

    assert result == answer
    assert provider.tool_choice_seen == [None]
    assert _stall_correctives(agent) == []
