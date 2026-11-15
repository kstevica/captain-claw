"""Tool search: a full-mode agent past its tool-definitions budget sends only
some definitions in full, `search_tools` loads the rest, and a deferred tool
still runs when called by name. All tools and data here are made up."""

from __future__ import annotations

import json
from typing import Any

import pytest

from captain_claw import tool_search
from captain_claw.agent import Agent
from captain_claw.config import ToolSearchConfig, get_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


def _count(text: str) -> int:
    return len(text)


def _defn(name: str, desc: str = "", size: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "description": desc or f"The {name} tool.",
        "parameters": {"type": "object", "properties": {"pad": {"type": "string", "description": "x" * size}}},
    }


def _sizes(defs: list[dict[str, Any]]) -> dict[str, int]:
    return {d["name"]: tool_search.definition_tokens(d, _count) for d in defs}


# ── select ──────────────────────────────────────────────────────────────


def test_config_default_budget() -> None:
    cfg = ToolSearchConfig()
    assert cfg.enabled is True
    assert cfg.budget_tokens == 16000
    assert get_config().tools.tool_search.budget_tokens >= 0


def test_switch_file_beats_the_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flight Deck's switch applies from the next message; without it the
    config decides; the agent's own settings page clears it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cc = tmp_path / ".captain-claw"
    cc.mkdir()
    cfg = get_config().tools.tool_search
    assert tool_search.settings() == {"enabled": cfg.enabled and cfg.budget_tokens > 0,
                                      "budget_tokens": cfg.budget_tokens or 16000}

    (cc / tool_search.SWITCH_FILENAME).write_text(tool_search.render_switch(True, 8000))
    assert tool_search.settings() == {"enabled": True, "budget_tokens": 8000}
    assert tool_search.budget_tokens() == 8000

    (cc / tool_search.SWITCH_FILENAME).write_text(tool_search.render_switch(False, 8000))
    assert tool_search.budget_tokens() == 0

    tool_search.clear()
    assert not (cc / tool_search.SWITCH_FILENAME).exists()


@pytest.mark.parametrize("text, expected", [
    ('{"enabled": true, "budget_tokens": 24000}', {"enabled": True, "budget_tokens": 24000}),
    ('{"enabled": true, "budget_tokens": 0}', {"enabled": False, "budget_tokens": 16000}),
    ('{"enabled": "yes", "budget_tokens": 24000}', None),
    ('{"enabled": true}', None),
    ("on", None),
    ("", None),
])
def test_parse_switch(text: str, expected) -> None:
    assert tool_search.parse_switch(text) == expected


def test_under_budget_sends_everything_unchanged() -> None:
    defs = [_defn("read"), _defn("frobnicate", size=50)]
    sent, deferred, state = tool_search.select(defs, _sizes(defs), budget=10_000, state=None, count=_count)
    assert sent == defs
    assert deferred == []
    assert state is None


def test_over_budget_sends_definitions_under_budget() -> None:
    defs = [
        _defn("read"), _defn("write"),
        _defn("pptx_maker", "Builds slide decks.", size=900),
        _defn("bulk_resizer", "Resizes images.", size=700),
        _defn("note_pad", "Keeps notes.", size=40),
        _defn("cron", "Schedules things.", size=60),
    ]
    sizes = _sizes(defs)
    budget = 1500
    sent, deferred, state = tool_search.select(
        defs, sizes, budget=budget, state=None, core={"cron"}, count=_count)
    sent_names = [d["name"] for d in sent]
    # The basics and the core set go in full; the big ones are deferred.
    assert {"read", "write", "cron", "note_pad"} <= set(sent_names)
    assert set(deferred) == {"pptx_maker", "bulk_resizer"}
    # search_tools comes last and names what is deferred.
    assert sent_names[-1] == tool_search.SEARCH_TOOLS_NAME
    assert "pptx_maker" in sent[-1]["description"] and "bulk_resizer" in sent[-1]["description"]
    # Everything sent, search_tools included, stays under the budget.
    total = sum(tool_search.definition_tokens(d, _count) for d in sent)
    assert total <= budget
    assert state == {"budget": budget, "loaded": sorted(set(sent_names) - {tool_search.SEARCH_TOOLS_NAME})}


def test_intent_matched_tool_goes_in_full() -> None:
    defs = [_defn("read"), _defn("twitter", size=300), _defn("pptx_maker", size=1500)]
    sent, deferred, _ = tool_search.select(
        defs, _sizes(defs), budget=1400, state=None, intent={"twitter"}, count=_count)
    assert "twitter" in [d["name"] for d in sent]
    assert deferred == ["pptx_maker"]


def test_selection_is_kept_for_the_session() -> None:
    """A later message's intent doesn't change the set (the prompt cache
    holds); only a used tool is added."""
    defs = [_defn("read"), _defn("twitter", size=300), _defn("pptx_maker", size=300), _defn("zipper", size=300)]
    sizes = _sizes(defs)
    sent1, deferred1, state = tool_search.select(defs, sizes, budget=1400, state=None, intent={"twitter"},
                                                 count=_count)
    sent2, deferred2, state2 = tool_search.select(defs, sizes, budget=1400, state=state, intent={"pptx_maker"},
                                                  count=_count)
    assert sent2 == sent1 and deferred2 == deferred1 and state2 == state
    sent3, deferred3, state3 = tool_search.select(defs, sizes, budget=1400, state=state, used={"zipper"},
                                                  count=_count)
    assert "zipper" in [d["name"] for d in sent3]
    assert deferred3 == ["pptx_maker"]
    assert "zipper" in state3["loaded"]


def test_changed_budget_reselects() -> None:
    defs = [_defn("read"), _defn("pptx_maker", size=300), _defn("zipper", size=300)]
    sizes = _sizes(defs)
    _, deferred, state = tool_search.select(defs, sizes, budget=600, state=None, count=_count)
    assert deferred
    _, deferred2, state2 = tool_search.select(defs, sizes, budget=100_000, state=state, count=_count)
    assert deferred2 == []


# ── search ──────────────────────────────────────────────────────────────

_POOL = [
    _defn("read", "Read a file from disk."),
    _defn("pptx", "Create and edit PowerPoint slide decks."),
    _defn("image_manipulation", "Resize, crop, rotate and convert images."),
    _defn("google_mail", "Read, draft and send email through the connected Gmail account."),
    _defn("twitter", "Post a tweet."),
]


def test_search_by_exact_name() -> None:
    hits = tool_search.search(_POOL, "pptx")
    assert hits[0]["name"] == "pptx"


def test_search_by_several_names() -> None:
    names = {d["name"] for d in tool_search.search(_POOL, "pptx, twitter, google_mail", limit=1)}
    assert names == {"pptx", "twitter", "google_mail"}


def test_search_by_words() -> None:
    assert tool_search.search(_POOL, "resize an image")[0]["name"] == "image_manipulation"
    assert tool_search.search(_POOL, "slide decks")[0]["name"] == "pptx"
    assert tool_search.search(_POOL, "send an email")[0]["name"] == "google_mail"
    assert tool_search.search(_POOL, "quantum chromodynamics") == []


# ── The agent ───────────────────────────────────────────────────────────


class _FrobTool(Tool):
    name = "frobnicate"
    description = "Frobnicates widgets in bulk. " + "Long manual text. " * 40
    parameters = {
        "type": "object",
        "properties": {"widget": {"type": "string", "description": "The widget to frobnicate."}},
        "required": ["widget"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append({k: v for k, v in kwargs.items() if not k.startswith("_")})
        return ToolResult(success=True, content=f"frobnicated {kwargs.get('widget')}")


class _ReadTool(Tool):
    name = "read"
    description = "Read a file."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, content="file text")


class _ScriptedProvider(LLMProvider):
    """Plays back one response per call and records the tools each call got."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.tools_per_call: list[list[str]] = []
        self.messages_per_call: list[list[Message]] = []

    async def complete(self, messages: list[Message], tools: list[ToolDefinition] | None = None,
                       temperature: float | None = None, max_tokens: int | None = None) -> LLMResponse:
        self.tools_per_call.append([t["name"] if isinstance(t, dict) else t.name for t in (tools or [])])
        self.messages_per_call.append(messages)
        return self.responses.pop(0) if self.responses else LLMResponse(content="done")

    async def complete_streaming(self, messages: list[Message], tools: list[ToolDefinition] | None = None,
                                 temperature: float | None = None, max_tokens: int | None = None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


class _NoSave:
    async def save_session(self, session: Session) -> None:
        return None


def _agent(provider: LLMProvider, frob: _FrobTool, monkeypatch: pytest.MonkeyPatch) -> Agent:
    agent = Agent(provider=provider)
    agent._initialized = True
    agent.session = Session(id="ts1", name="default")
    agent.session_manager = _NoSave()
    agent.instructions.use_micro = False   # full mode
    # The default test model may be one the loop won't send tool results back
    # to (an Ollama ":cloud" model); these tests need the follow-up call.
    monkeypatch.setattr(agent, "_supports_tool_result_followup", lambda: True)
    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(frob)
    agent.tools = registry
    # Room for `read` and search_tools, not for frobnicate's long manual.
    monkeypatch.setattr(tool_search, "budget_tokens", lambda: 900)
    return agent


def _tool_result(provider: _ScriptedProvider, call: int) -> str:
    for msg in provider.messages_per_call[call]:
        if getattr(msg, "role", "") == "tool" and getattr(msg, "tool_name", "") == tool_search.SEARCH_TOOLS_NAME:
            return str(msg.content)
    return "".join(str(getattr(m, "content", "")) for m in provider.messages_per_call[call]
                   if getattr(m, "role", "") == "tool")


@pytest.mark.asyncio
async def test_search_tools_unlocks_a_deferred_tool_which_then_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ScriptedProvider([
        LLMResponse(content="", tool_calls=[ToolCall(
            id="s1", name=tool_search.SEARCH_TOOLS_NAME, arguments={"query": "frobnicate widgets"})]),
        LLMResponse(content="", tool_calls=[ToolCall(
            id="f1", name="frobnicate", arguments={"widget": "w-17"})]),
        LLMResponse(content="Frobnicated w-17."),
    ])
    agent = _agent(provider, frob, monkeypatch)

    result = await agent.complete("please frobnicate widget w-17")

    assert result == "Frobnicated w-17."
    # First call: frobnicate deferred, search_tools offered in its place.
    assert "frobnicate" not in provider.tools_per_call[0]
    assert tool_search.SEARCH_TOOLS_NAME in provider.tools_per_call[0]
    assert "read" in provider.tools_per_call[0]
    # search_tools returned the full definition…
    found = _tool_result(provider, 1)
    assert "Loaded 1 tool" in found
    payload = json.loads(found.splitlines()[1])
    assert payload["name"] == "frobnicate"
    assert payload["parameters"]["required"] == ["widget"]
    # …and the next call carries it.
    assert "frobnicate" in provider.tools_per_call[1]
    # The deferred tool ran through the registry.
    assert frob.calls == [{"widget": "w-17"}]
    # Loaded for the rest of the session.
    assert "frobnicate" in agent.session.metadata[tool_search.STATE_KEY]["loaded"]


@pytest.mark.asyncio
async def test_deferred_tool_called_by_name_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ScriptedProvider([
        LLMResponse(content="", tool_calls=[ToolCall(
            id="f1", name="frobnicate", arguments={"widget": "w-3"})]),
        LLMResponse(content="Done."),
    ])
    agent = _agent(provider, frob, monkeypatch)

    await agent.complete("frobnicate w-3")

    assert "frobnicate" not in provider.tools_per_call[0]
    assert frob.calls == [{"widget": "w-3"}]
    # Used, so it is in full from then on.
    assert "frobnicate" in provider.tools_per_call[1]


@pytest.mark.asyncio
async def test_tool_list_is_stable_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ScriptedProvider([LLMResponse(content="Hi."), LLMResponse(content="Sure.")])
    agent = _agent(provider, frob, monkeypatch)

    await agent.complete("hello")
    await agent.complete("now something about a tweet and a slide deck")

    assert provider.tools_per_call[0] == provider.tools_per_call[1]


@pytest.mark.asyncio
async def test_under_budget_no_search_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    agent = _agent(provider, frob, monkeypatch)
    monkeypatch.setattr(tool_search, "budget_tokens", lambda: 1_000_000)

    await agent.complete("hello")

    assert provider.tools_per_call[0] == ["read", "frobnicate"]


@pytest.mark.asyncio
async def test_eco_mode_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ScriptedProvider([LLMResponse(content="ok")])
    agent = _agent(provider, frob, monkeypatch)
    agent.instructions.use_micro = True

    await agent.complete("hello")

    assert tool_search.SEARCH_TOOLS_NAME not in provider.tools_per_call[0]


@pytest.mark.asyncio
async def test_search_tools_when_nothing_is_deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    agent = _agent(_ScriptedProvider([]), frob, monkeypatch)
    result = tool_search.run(agent, {"query": "frobnicate"})
    assert result.success and "already in your function list" in result.content
