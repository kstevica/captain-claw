"""A scheduled run that called none of its tools: the retry must call one,
and a report that still shipped unchecked is never replayed as an example.

A self-hosted deck, 2026-11-18..22: a 3-hourly reply-watch on a small local model ran
39 times; 37 replies were turned down for calling no tool and 11 runs shipped
a made-up "live check" ("Inbox read this turn: 27 messages") with the note
that nothing was checked. The corrective asked in words only; the stall gate's
forced ``tool_choice`` made the same model call imap_mail at once. The
shipped reports stayed in the job's history and the next runs copied them.
"""

from __future__ import annotations

from typing import Any

import pytest

from captain_claw import fabrication_guard as fg
from captain_claw import msg_origin
from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

WATCH = (
    "DAILY WATCH: new tenders for the service.\n"
    "1) Run web_search on these queries: cleaning services tender, facility tender.\n"
    "2) Report new ones only."
)
COPIED = (
    "**Watch — live check.** Ran all nine queries just now and read each result.\n\n"
    "**Nothing new.** Notices 70001 and 70002 are the ones already reported; no new tender "
    "matches the service. Next check tomorrow at the same time."
)
REAL = "Checked with web_search: 2 results, both already reported — nothing new."


class _Session:
    async def save_session(self, session):
        return None


class _Recording(LLMProvider):
    """Scripted answers; records the forced ``tool_choice`` each call saw and
    consumes it, as the real providers do."""

    provider = "openai"
    model = "stub"

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.calls: list[list[Any]] = []
        self.forced: list[Any] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append(list(messages))
        self.forced.append(getattr(self, "_tool_choice_override", None))
        self._tool_choice_override = None
        return self.responses.pop(0) if self.responses else LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


class _Search(Tool):
    name = "web_search"
    description = "Search the web"
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}

    def __init__(self):
        self.calls = 0

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, content="1. Notice 70001\n2. Notice 70002")


def _agent(responses: list[LLMResponse]) -> tuple[Agent, _Recording, _Search]:
    provider = _Recording(responses)
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="job", name="⏰ DAILY WATCH")
    agent.session_manager = _Session()
    search = _Search()
    registry = ToolRegistry()
    registry.register(search)
    agent.tools = registry
    agent._build_env_now_text = lambda: ""
    return agent, provider, search


@pytest.fixture(autouse=True)
def _plain_agent(monkeypatch: pytest.MonkeyPatch):
    for name in ("CLAW_BEING_WORKER", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER",
                 "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT", "CLAW_BAT_WORKER"):
        monkeypatch.delenv(name, raising=False)


def _search_call() -> LLMResponse:
    return LLMResponse(content="", tool_calls=[
        ToolCall(id="w1", name="web_search", arguments={"query": "cleaning services tender"})])


@pytest.mark.asyncio
async def test_the_retry_must_call_a_tool_and_the_run_does_its_work():
    agent, provider, search = _agent([LLMResponse(content=COPIED), _search_call(),
                                      LLMResponse(content=REAL)])

    result = await agent.complete(WATCH, turn_origin="cron")

    assert search.calls == 1
    assert result.startswith(REAL) and fg.SKIPPED_TOOLS_BACKSTOP not in result
    # Forced on the retry only; the call after the tool chooses freely.
    assert provider.forced == [None, "required", None]
    turned_down = [m for m in agent.session.messages if m.get("origin") == "rejected"]
    assert [m["content"] for m in turned_down] == [COPIED]


@pytest.mark.asyncio
async def test_a_persons_turn_is_never_forced():
    """The lookup check still sends a person's narrated reply back in words."""
    narrated = "Checking the tender portal now.\n\nSearching the gazette too.\n\nNothing new."
    agent, provider, search = _agent([LLMResponse(content=narrated), LLMResponse(content=REAL)])

    await agent.complete("any new tenders? use web_search", turn_origin="human")

    assert len(provider.calls) == 2 and search.calls == 0
    assert provider.forced == [None, None]


@pytest.mark.asyncio
async def test_a_run_that_never_calls_one_ships_with_the_note_and_is_replayed_as_a_line():
    retries = fg.MAX_FABRICATION_RETRIES
    agent, provider, _ = _agent([LLMResponse(content=COPIED)] * (retries + 1))

    result = await agent.complete(WATCH, turn_origin="cron")

    assert result.rstrip().endswith(fg.SKIPPED_TOOLS_BACKSTOP)
    # Every retry was forced; nothing is left set after the last one.
    assert provider.forced == [None] + ["required"] * retries
    assert getattr(provider, "_tool_choice_override", None) is None
    shipped = [m for m in agent.session.messages
               if m["role"] == "assistant" and m.get("origin") != "rejected"]
    assert len(shipped) == 1 and msg_origin.is_unchecked_run_report(shipped[0])
    assert msg_origin.is_turned_down(shipped[0])   # not summarized, indexed or remembered

    # The next run sees one line where the made-up report was.
    provider.responses = [_search_call(), LLMResponse(content=REAL)]
    provider.calls.clear()
    await agent.complete(WATCH, turn_origin="cron")

    first_call = "\n".join(str(getattr(m, "content", "") or "") for m in provider.calls[0])
    assert "Ran all nine queries" not in first_call
    assert msg_origin.UNCHECKED_RUN_REPLAY in first_call


def test_the_note_is_one_string_everywhere():
    assert fg.SKIPPED_TOOLS_BACKSTOP == msg_origin.UNCHECKED_RUN_NOTE
    assert fg.has_backstop(f"{COPIED}\n\n{fg.SKIPPED_TOOLS_BACKSTOP}")


@pytest.mark.parametrize("msg, unchecked", [
    ({"role": "assistant", "content": f"{COPIED}\n\n{msg_origin.UNCHECKED_RUN_NOTE}"}, True),
    # Other backstop lines after it (made-up IDs) don't hide it.
    ({"role": "assistant", "content": f"{COPIED}\n\n{msg_origin.UNCHECKED_RUN_NOTE}\n"
                                      "Note: `70003` was not returned by any tool."}, True),
    ({"role": "assistant", "content": COPIED}, False),
    ({"role": "user", "content": f"why this? {msg_origin.UNCHECKED_RUN_NOTE}"}, False),
    ({"role": "assistant", "content": msg_origin.UNCHECKED_RUN_NOTE,
      "tool_calls": [{"id": "x", "function": {"name": "web_search", "arguments": "{}"}}]}, False),
    ("not a message", False),
])
def test_what_counts_as_an_unchecked_report(msg, unchecked):
    assert msg_origin.is_unchecked_run_report(msg) is unchecked
    assert msg_origin.is_turned_down(msg) is unchecked
