"""Anthropic's own tool search: on a Claude model that has it, every tool
definition goes out (the deferred ones marked ``defer_loading``) with
Anthropic's search tool, the searches the model runs are kept on the assistant
message and sent back unchanged, and the tool list never changes. LiteLLM's
real Anthropic converters and parsers run here; nothing reaches the network.
All tools and data are made up."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from captain_claw import tool_search
from captain_claw.agent import Agent
from captain_claw.llm import (
    LiteLLMProvider,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
    _convert_messages_for_openai_style,
)
from captain_claw.llm import anthropic_tool_search as ats
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

SEARCH_ID = "srvtoolu_01TestSearch"


def _search_blocks(found: str = "frobnicate", search_id: str = SEARCH_ID) -> list[dict[str, Any]]:
    return [
        {"type": "server_tool_use", "id": search_id, "name": ats.SEARCH_TOOL_NAME,
         "input": {"query": "frobnicate widgets"}},
        {"type": "tool_search_tool_result", "tool_use_id": search_id,
         "content": {"type": "tool_search_tool_search_result",
                     "tool_references": [{"type": "tool_reference", "tool_name": found}]}},
    ]


# ── Which models ────────────────────────────────────────────────────────


@pytest.mark.parametrize("model, ok", [
    ("anthropic/claude-opus-5-5", True),
    ("claude-sonnet-5-5", True),
    ("claude-haiku-5-5", True),
    ("claude-fable-5-2", True),
    ("claude-mythos-preview", True),
    ("claude-opus-4-8", True),
    ("claude-sonnet-4-6", True),
    ("claude-haiku-4-5-20251001", True),
    ("claude-sonnet-4-5-20250929", True),
    ("claude-opus-4-1-20250805", False),
    ("claude-sonnet-4-20250514", False),
    ("claude-3-5-sonnet-20241022", False),
    ("gpt-5", False),
    ("", False),
])
def test_model_supported(model: str, ok: bool) -> None:
    assert ats.model_supported(model) is ok


def test_provider_support_needs_anthropic_api() -> None:
    assert LiteLLMProvider("anthropic", "claude-haiku-5-5", api_key="k").supports_native_tool_search()
    assert not LiteLLMProvider("anthropic", "claude-3-5-sonnet-20241022", api_key="k").supports_native_tool_search()
    # An Anthropic-style endpoint somewhere else may not have server tools.
    assert not LiteLLMProvider("anthropic", "claude-haiku-5-5", api_key="k",
                               base_url="https://proxy.example.test/anthropic").supports_native_tool_search()
    assert not LiteLLMProvider("openai", "gpt-5", api_key="k").supports_native_tool_search()


# ── select(native=True) ─────────────────────────────────────────────────


def _defn(name: str, size: int = 0) -> dict[str, Any]:
    return {"name": name, "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {"pad": {"type": "string", "description": "x" * size}}}}


def _sizes(defs: list[dict[str, Any]]) -> dict[str, int]:
    return {d["name"]: tool_search.definition_tokens(d, len) for d in defs}


def test_native_select_has_no_search_tools_and_does_not_grow() -> None:
    defs = [_defn("read"), _defn("twitter", 300), _defn("pptx_maker", 300), _defn("zipper", 300)]
    sizes = _sizes(defs)
    sent, deferred, state = tool_search.select(defs, sizes, budget=900, state=None, count=len, native=True)
    assert tool_search.SEARCH_TOOLS_NAME not in [d["name"] for d in sent]
    assert deferred
    # A deferred tool the model found and used stays deferred: the list holds.
    used = {deferred[0]}
    sent2, deferred2, state2 = tool_search.select(defs, sizes, budget=900, state=state, used=used,
                                                  count=len, native=True)
    assert deferred2 == deferred and state2 == state and sent2 == sent


# ── The agent ───────────────────────────────────────────────────────────


class _FrobTool(Tool):
    name = "frobnicate"
    description = "Frobnicates widgets in bulk. " + "Long manual text. " * 40
    parameters = {"type": "object", "properties": {"widget": {"type": "string"}}, "required": ["widget"]}

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


class _ClaudeLike(LLMProvider):
    """A scripted model that says it has Anthropic's tool search; records
    the tool definitions and messages of every call."""

    def __init__(self, responses: list[LLMResponse], native: bool = True) -> None:
        self.responses = list(responses)
        self.native = native
        self.tools_per_call: list[list[dict[str, Any]]] = []
        self.messages_per_call: list[list[Message]] = []

    def supports_native_tool_search(self) -> bool:
        return self.native

    async def complete(self, messages: list[Message], tools: list[ToolDefinition] | None = None,
                       temperature: float | None = None, max_tokens: int | None = None) -> LLMResponse:
        self.tools_per_call.append([dict(t) for t in (tools or [])])
        self.messages_per_call.append(messages)
        return self.responses.pop(0) if self.responses else LLMResponse(content="done")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
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
    agent.session = Session(id="ats1", name="default")
    agent.session_manager = _NoSave()
    agent.instructions.use_micro = False   # full mode
    monkeypatch.setattr(agent, "_supports_tool_result_followup", lambda: True)
    registry = ToolRegistry()
    registry.register(_ReadTool())
    registry.register(frob)
    agent.tools = registry
    monkeypatch.setattr(tool_search, "budget_tokens", lambda: 900)
    monkeypatch.setattr(tool_search, "native_enabled", lambda: True)
    return agent


def _names(defs: list[dict[str, Any]]) -> list[str]:
    return [str(d.get("name")) for d in defs]


@pytest.mark.asyncio
async def test_native_search_keeps_the_tool_list_and_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ClaudeLike([
        # The model searches on the server, finds frobnicate and calls it.
        LLMResponse(content="", tool_calls=[ToolCall(id="toolu_f1", name="frobnicate",
                                                     arguments={"widget": "w-17"})],
                    server_blocks=_search_blocks()),
        LLMResponse(content="Frobnicated w-17."),
        LLMResponse(content="Hi again."),
    ])
    agent = _agent(provider, frob, monkeypatch)

    result = await agent.complete("please frobnicate widget w-17")
    assert result == "Frobnicated w-17."

    first = provider.tools_per_call[0]
    # Every definition goes out, frobnicate marked deferred, Anthropic's
    # search tool last and no search_tools.
    assert _names(first) == ["read", "frobnicate", ats.SEARCH_TOOL_NAME]
    assert first[-1] == ats.search_tool()
    by_name = {d["name"]: d for d in first}
    assert by_name["frobnicate"].get("defer_loading") is True
    assert "defer_loading" not in by_name["read"]
    assert frob.calls == [{"widget": "w-17"}]
    # The call after the load sends the same tool list (the cache holds).
    assert provider.tools_per_call[1] == first

    # The search is stored on the assistant message with the call…
    stored = [m for m in agent.session.messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert stored and stored[-1]["server_blocks"] == _search_blocks()
    # …and sent back with it on the next call.
    sent = [m for m in provider.messages_per_call[1] if m.role == "assistant" and m.tool_calls]
    assert sent and sent[-1].server_blocks == _search_blocks()
    # No other assistant message picked the blocks up.
    assert sum(1 for m in agent.session.messages if m.get("server_blocks")) == 1

    # A later turn: the same list, frobnicate still deferred.
    await agent.complete("hello")
    assert provider.tools_per_call[2] == first


@pytest.mark.asyncio
async def test_native_off_uses_search_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ClaudeLike([LLMResponse(content="ok")])
    agent = _agent(provider, frob, monkeypatch)
    monkeypatch.setattr(tool_search, "native_enabled", lambda: False)

    await agent.complete("hello")

    names = _names(provider.tools_per_call[0])
    assert tool_search.SEARCH_TOOLS_NAME in names and "frobnicate" not in names
    assert ats.SEARCH_TOOL_NAME not in names


@pytest.mark.asyncio
async def test_unsupported_model_uses_search_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    frob = _FrobTool()
    provider = _ClaudeLike([LLMResponse(content="ok")], native=False)
    agent = _agent(provider, frob, monkeypatch)

    await agent.complete("hello")

    names = _names(provider.tools_per_call[0])
    assert tool_search.SEARCH_TOOLS_NAME in names and ats.SEARCH_TOOL_NAME not in names


# ── Requests, through LiteLLM's own Anthropic converters ────────────────


def _anthropic_payload(kwargs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    messages = [m for m in kwargs["messages"] if m.get("role") != "system"]
    tools, _ = AnthropicConfig()._map_tools(kwargs.get("tools") or [])
    return anthropic_messages_pt(messages, model="claude-haiku-5-5", llm_provider="anthropic"), tools


def _history() -> list[Message]:
    return [
        Message(role="system", content="You are a test agent."),
        Message(role="user", content="please frobnicate widget w-17"),
        Message(role="assistant", content="Searching.", server_blocks=_search_blocks(),
                tool_calls=[{"id": "toolu_f1", "type": "function",
                             "function": {"name": "frobnicate", "arguments": '{"widget": "w-17"}'}}]),
        Message(role="tool", content="frobnicated w-17", tool_call_id="toolu_f1", tool_name="frobnicate"),
    ]


def _tools(native: bool) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [_defn("read"), {**_defn("frobnicate"), "defer_loading": True}]
    return tools + [ats.search_tool()] if native else [_defn("read"), _defn("frobnicate")]


def test_request_sends_deferred_tools_search_tool_and_search_blocks() -> None:
    provider = LiteLLMProvider("anthropic", "claude-haiku-5-5", api_key="k")
    payload, tools = _anthropic_payload(provider._request_kwargs(_history(), tools=_tools(native=True)))

    by_name = {t["name"]: t for t in tools}
    assert by_name["frobnicate"]["defer_loading"] is True
    assert "defer_loading" not in by_name["read"]
    assert by_name[ats.SEARCH_TOOL_NAME]["type"] == ats.SEARCH_TOOL_TYPE

    assistant = [m for m in payload if m["role"] == "assistant"][-1]
    types = [b["type"] for b in assistant["content"]]
    assert types == ["text", "server_tool_use", "tool_search_tool_result", "tool_use"]
    server_use, result = assistant["content"][1], assistant["content"][2]
    assert {k: server_use[k] for k in ("type", "id", "name", "input")} == _search_blocks()[0]
    assert result["content"] == _search_blocks()[1]["content"]
    # The search answer never goes back as a tool_result.
    results = [b for m in payload if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"]
    assert [r["tool_use_id"] for r in results] == ["toolu_f1"]


def test_search_blocks_stay_home_without_the_search_tool() -> None:
    """A later call without Anthropic's search (switched model, tool search
    off) must not send its blocks: the API would refuse them."""
    provider = LiteLLMProvider("anthropic", "claude-3-5-sonnet-20241022", api_key="k")
    payload, tools = _anthropic_payload(provider._request_kwargs(_history(), tools=_tools(native=False)))
    assistant = [m for m in payload if m["role"] == "assistant"][-1]
    assert [b["type"] for b in assistant["content"]] == ["text", "tool_use"]
    assert all("defer_loading" not in t for t in tools)


def test_other_providers_never_get_the_search_tool() -> None:
    provider = LiteLLMProvider("openai", "gpt-5", api_key="k")
    kwargs = provider._request_kwargs(_history(), tools=_tools(native=True))
    assert [t["function"]["name"] for t in kwargs["tools"]] == ["read", "frobnicate"]
    assert all("defer_loading" not in t for t in kwargs["tools"])
    assistant = [m for m in kwargs["messages"] if m["role"] == "assistant"][-1]
    assert isinstance(assistant["content"], str)


def test_unpaired_blocks_are_not_sent() -> None:
    lone = [_search_blocks()[0]]
    out = _convert_messages_for_openai_style(
        [Message(role="assistant", content="x", server_blocks=lone)], server_blocks=True)
    assert out[0]["content"] == "x"


# ── Responses, through LiteLLM's own Anthropic parsers ──────────────────


def _anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [
            {"type": "text", "text": "Looking for a tool."},
            *_search_blocks(),
            {"type": "tool_use", "id": "toolu_f1", "name": "frobnicate", "input": {"widget": "w-17"}},
        ],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5000},
    }


@pytest.mark.asyncio
async def test_plain_response_keeps_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm import ModelResponse
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    assert ats.install()
    model_response = AnthropicConfig().transform_parsed_response(
        completion_response=_anthropic_response(),
        raw_response=httpx.Response(200, json=_anthropic_response()),
        model_response=ModelResponse(),
    )

    async def _fake_acompletion(kwargs, provider=None):
        return model_response

    monkeypatch.setattr("captain_claw.llm._acompletion_tolerant", _fake_acompletion)
    provider = LiteLLMProvider("anthropic", "claude-haiku-5-5", api_key="k")
    response = await provider.complete(_history()[:2], tools=_tools(native=True))

    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("toolu_f1", "frobnicate", {"widget": "w-17"})]
    assert response.server_blocks == _search_blocks()


def _sse(events: list[dict[str, Any]]):
    async def _lines():
        for event in events:
            yield f"data: {json.dumps(event)}"
    return _lines()


def _stream_events() -> list[dict[str, Any]]:
    search_use, search_result = _search_blocks()
    return [
        {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [], "stop_reason": None,
            "usage": {"input_tokens": 120, "output_tokens": 1, "cache_read_input_tokens": 5000,
                      "cache_creation_input_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Looking."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {**search_use, "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {
            "type": "input_json_delta", "partial_json": json.dumps(search_use["input"])}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": search_result},
        {"type": "content_block_stop", "index": 2},
        {"type": "content_block_start", "index": 3, "content_block": {
            "type": "tool_use", "id": "toolu_f1", "name": "frobnicate", "input": {}}},
        {"type": "content_block_delta", "index": 3, "delta": {
            "type": "input_json_delta", "partial_json": '{"widget": "w-17"}'}},
        {"type": "content_block_stop", "index": 3},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 40}},
        {"type": "message_stop"},
    ]


@pytest.mark.asyncio
async def test_streamed_response_keeps_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    assert ats.install()

    async def _fake_acompletion(kwargs, provider=None):
        return ModelResponseIterator(streaming_response=_sse(_stream_events()), sync_stream=False)

    monkeypatch.setattr("captain_claw.llm._acompletion_tolerant", _fake_acompletion)
    provider = LiteLLMProvider("anthropic", "claude-haiku-5-5", api_key="k")
    chunks: list[str] = []
    response = await provider.complete_with_callback(_history()[:2], tools=_tools(native=True),
                                                     on_chunk=chunks.append)

    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("toolu_f1", "frobnicate", {"widget": "w-17"})]
    assert response.server_blocks == _search_blocks()
    assert response.content == "Looking."


def test_search_without_a_kept_result_is_dropped() -> None:
    calls = [ToolCall(id="srvtoolu_never_seen", name=ats.SEARCH_TOOL_NAME, arguments={"query": "x"}),
             ToolCall(id="toolu_2", name="read", arguments={"path": "a.txt"})]
    client, blocks = ats.split_tool_calls(calls)
    assert [c.id for c in client] == ["toolu_2"]
    assert blocks == []
