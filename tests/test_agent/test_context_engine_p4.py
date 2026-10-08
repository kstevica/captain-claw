"""Context engine P4: accounting inside the tier's budget, and answers that
stop early are continued instead of shipped as complete."""

from __future__ import annotations

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import LiteLLMProvider, LLMProvider, LLMResponse, ToolCall
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


def _words(text: str) -> int:
    return len([part for part in (text or "").split() if part]) or 1


class _Stub(LLMProvider):
    provider = "openai"
    model = "stub"

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[list] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text):
        return _words(text)


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


# ── Answers cut off at the output limit ───────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["length", "interrupted"])
async def test_a_cut_off_answer_is_continued_and_joined(reason):
    provider = _Stub([
        LLMResponse(content="The report covers revenue, ", finish_reason=reason),
        LLMResponse(content="costs and the outlook.", finish_reason="stop"),
    ])
    agent = _agent(provider)

    result = await agent.complete("write the report")

    assert result.startswith("The report covers revenue, costs and the outlook.")
    # The continuation request carried the partial and the instruction.
    second = provider.calls[1]
    assert second[-2].content == "The report covers revenue, "
    assert "cut off before it was finished" in second[-1].content
    stored = [(m["role"], m.get("origin")) for m in agent.session.messages]
    assert ("assistant", "rejected") in stored and ("user", "corrective") in stored
    assert agent.session.messages[-1]["content"].startswith(
        "The report covers revenue, costs and the outlook."
    )


@pytest.mark.asyncio
async def test_continuations_are_bounded():
    provider = _Stub([LLMResponse(content=f"part {n} ", finish_reason="length") for n in range(5)])
    agent = _agent(provider)

    result = await agent.complete("go")

    # Two continuations, then the third piece is accepted as it is.
    assert result.startswith("part 0 part 1 part 2")
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_a_broken_stream_is_marked_interrupted():
    provider = LiteLLMProvider(provider="openai", model="gpt-test", api_key="sk-test")

    class _Chunk(dict):
        pass

    async def _stream():
        yield {"choices": [{"delta": {"content": "half an "}, "finish_reason": None}]}
        raise RuntimeError("Timeout on reading data from socket")

    result = await provider._collect_streaming_response(_stream())
    choice = result["choices"][0]
    assert choice["message"]["content"] == "half an "
    assert choice["finish_reason"] == "interrupted"


@pytest.mark.asyncio
async def test_a_cut_off_compaction_summary_is_retried_with_room():
    provider = _Stub([
        LLMResponse(content="Summary: the user", finish_reason="length"),
        LLMResponse(content="Summary: the user planned a trip to Split.", finish_reason="stop"),
    ])
    agent = _agent(provider)

    summary = await agent._summarize_for_compaction([
        {"role": "user", "content": "plan a trip to Split"},
        {"role": "assistant", "content": "Here is a plan."},
    ])

    assert summary == "Summary: the user planned a trip to Split."
    assert len(provider.calls) == 2


# ── Accounting inside the tier's budget ───────────────────────────────


class _ThinkingLiteLLM(LiteLLMProvider):
    """A LiteLLM provider that never calls out, with word-count tokens."""

    def count_tokens(self, text):
        return _words(text)


def test_replayed_reasoning_weighs_on_the_budget_only_where_it_is_sent():
    deepseek = Agent(provider=_ThinkingLiteLLM(provider="openai", model="deepseek-test", api_key="k"))
    anthropic = Agent(provider=_ThinkingLiteLLM(provider="anthropic", model="claude-test", api_key="k"))
    plain = Agent(provider=_Stub())
    msg = {"role": "assistant", "content": "one two", "reasoning_content": "think " * 30}

    assert deepseek._wire_token_count(dict(msg)) == 32
    assert anthropic._wire_token_count(dict(msg)) == 2
    assert plain._wire_token_count(dict(msg)) == 2

    deepseek.session = Session(id="s1", name="d")
    deepseek._build_env_now_text = lambda: ""
    deepseek.session.add_message("user", "q1", origin="human")
    deepseek.session.add_message("assistant", "a1", reasoning_content="think " * 30, origin="model")
    deepseek.session.add_message("user", "q2", origin="human")
    deepseek._build_messages(tool_messages_from_index=2, query="q2")
    window = deepseek.last_context_window
    assert window["reasoning_in_budget"] == 1
    assert window["sections"]["prior_history"] >= 30


def test_tool_schemas_come_off_the_history_budget():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 1000
    set_config(cfg)
    try:
        agent = Agent(provider=_Stub())
        agent.session = Session(id="s1", name="default")
        agent._build_env_now_text = lambda: ""
        agent._build_system_prompt = lambda: "sys"
        agent.session.add_message("user", "hello", origin="human")
        agent._build_messages(query="hello")
        before = agent.last_context_window["history_budget_tokens"]
        agent._note_tool_schema_tokens([{"name": "t", "description": "word " * 300, "parameters": {}}])
        agent._turn_system_prompt = None
        agent._build_messages(query="hello")
        after = agent.last_context_window["history_budget_tokens"]
        assert before - after == agent.last_context_window["tool_schema_tokens_budgeted"] > 0
    finally:
        set_config(old_cfg)


class _BigTool(Tool):
    name = "big"
    description = "Returns a lot of text"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="x" * 5000)


@pytest.mark.asyncio
async def test_tool_result_cap_is_off_by_default_and_applies_when_set():
    for cap, expected_len in ((0, 5000), (1000, None)):
        old_cfg = get_config().model_copy(deep=True)
        cfg = old_cfg.model_copy(deep=True)
        cfg.context.tool_result_max_chars = cap
        set_config(cfg)
        try:
            provider = _Stub([
                LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="big", arguments={})]),
                LLMResponse(content="done"),
            ])
            agent = _agent(provider)
            registry = ToolRegistry()
            registry.register(_BigTool())
            agent.tools = registry

            await agent.complete("fetch it")

            stored = next(m for m in agent.session.messages if m.get("tool_call_id") == "c1")
            if expected_len is not None:
                assert len(stored["content"]) == expected_len
            else:
                assert stored["content"].startswith("x" * 1000)
                assert "4000 more characters cut" in stored["content"]
        finally:
            set_config(old_cfg)


# ── Compaction weighs what the next turn would send ───────────────────


def _session_with_tool_output(agent: Agent, words_per_tool: int, tools: int) -> None:
    s = agent.session
    for n in range(tools):
        s.add_message("user", f"question {n}", origin="human")
        s.add_message("assistant", "", tool_calls=[{"id": f"c{n}", "type": "function",
                      "function": {"name": "web_fetch", "arguments": "{}"}}], origin="model")
        s.add_message("tool", "data " * words_per_tool, tool_call_id=f"c{n}", tool_name="web_fetch",
                      origin="tool")
        s.add_message("assistant", f"answer {n}", origin="model")
    s.add_message("user", "next question", origin="human")


@pytest.mark.asyncio
async def test_old_tool_output_alone_does_not_trigger_compaction():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 200        # threshold 160, storage guard 800
    set_config(cfg)
    try:
        agent = _agent(_Stub())
        _session_with_tool_output(agent, words_per_tool=60, tools=10)   # ~600 stored
        assert agent._session_token_count() < 160
        compacted, stats = await agent.compact_session(force=False, trigger="auto")
        assert compacted is False and stats["reason"] == "below_threshold"
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_storage_guard_compacts_when_stored_output_piles_up():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 200
    set_config(cfg)
    try:
        agent = _agent(_Stub([LLMResponse(content="earlier: fetched pages and answered")]))
        _session_with_tool_output(agent, words_per_tool=100, tools=10)  # ~1000 stored > 800
        compacted, _stats = await agent.compact_session(force=False, trigger="auto")
        assert compacted is True
    finally:
        set_config(old_cfg)


def test_send_weight_follows_the_history_rules():
    agent = Agent(provider=_ThinkingLiteLLM(provider="openai", model="deepseek-test", api_key="k"))
    agent.session = Session(id="s1", name="d")
    s = agent.session
    s.add_message("user", "q one", origin="human")                                    # 2
    s.add_message("assistant", "", tool_calls=[{"id": "c1", "type": "function",
                  "function": {"name": "x", "arguments": "{}"}}],
                  reasoning_content="r " * 50, origin="model")                        # stripped step
    s.add_message("tool", "big " * 500, tool_call_id="c1", tool_name="x", origin="tool")
    s.add_message("assistant", "Let me check:", reasoning_content="r " * 20, origin="model")
    s.add_message("assistant", "Answer one.", reasoning_content="r " * 10, origin="model")
    s.add_message("user", "[Flight Deck] Agent 'x' has joined the fleet on port 1.", origin="fleet_notice")
    s.add_message("user", "q two", origin="human")                                    # 2
    # q one (2) + "Let me check:" (3) + "Answer one." (2) + the run's last reasoning (10) + q two (2)
    assert agent._session_token_count() == 19


# ── Continuation edge cases (review round) ────────────────────────────


@pytest.mark.asyncio
async def test_a_fragment_answered_with_a_tool_call_is_not_glued_on_later():
    provider = _Stub([
        LLMResponse(content="Quarterly draft: revenue grew while costs", finish_reason="length"),
        LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="big", arguments={})]),
        LLMResponse(content="Final report: margin 60%.", finish_reason="stop"),
    ])
    agent = _agent(provider)
    registry = ToolRegistry()
    registry.register(_BigTool())
    agent.tools = registry

    result = await agent.complete("write the report")

    assert result.startswith("Final report: margin 60%.")
    assert "Quarterly draft" not in result


@pytest.mark.asyncio
async def test_a_reasoning_tail_is_not_continued_as_an_answer():
    thinking = "Plan A is cheaper.\n\nI should compare plan A and plan B on price, then"
    provider = _Stub([
        LLMResponse(content="I should compare plan A and plan B on price, then",
                    finish_reason="length", reasoning_content=thinking),
        LLMResponse(content="Plan A costs 10 EUR, plan B 12 EUR.", finish_reason="stop"),
    ])
    agent = _agent(provider)

    result = await agent.complete("compare the plans")

    assert result.startswith("Plan A costs 10 EUR")
    assert "I should compare" not in result
    assert "ran out of room while you were still thinking" in provider.calls[1][-1].content
    assert not any(m.get("origin") == "rejected" for m in agent.session.messages)


def test_the_seam_of_a_continued_answer():
    from captain_claw.agent_orchestration_mixin import _join_cut_off

    assert _join_cut_off(["The quick brown", "fox jumps."]) == "The quick brown fox jumps."
    assert _join_cut_off(["Intro paragraph.", "## Next"]) == "Intro paragraph.\n## Next"
    assert _join_cut_off(["| a | b |", "| c | d |"]) == "| a | b |\n| c | d |"
    assert _join_cut_off(["ends with space ", "next"]) == "ends with space next"
    # The model repeated the tail it had already written.
    assert _join_cut_off(["the totals for March were", "the totals for March were 42."]) == (
        "the totals for March were 42."
    )


def test_a_responses_reply_stopped_by_the_cap_reads_as_cut_off():
    from captain_claw.llm import ChatGPTResponsesProvider

    provider = ChatGPTResponsesProvider.__new__(ChatGPTResponsesProvider)
    provider.model = "gpt-test"
    event = {"type": "response.incomplete", "response": {
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "half"}]}],
    }}
    assert provider._parse_response_output(event).finish_reason == "length"


def test_replay_leaves_out_the_plumbing_of_a_continued_answer():
    from captain_claw.web.ws_handler import _build_replay_batch

    s = Session(id="s1", name="d")
    s.add_message("user", "write it", origin="human")
    s.add_message("assistant", "part one", origin="rejected", origin_detail="cut_off")
    s.add_message("user", "Your previous reply was cut off", origin="corrective",
                  origin_detail="continue_cut_off")
    s.add_message("assistant", "part one part two", origin="model")
    contents = [m["content"] for m in _build_replay_batch(s) if m.get("type") == "chat_message"]
    assert contents == ["write it", "part one part two"]


# ── Accounting edge cases (review round) ──────────────────────────────


def _thinking_turns(s: Session, turns: int, steps: int) -> None:
    for n in range(turns):
        s.add_message("user", f"question {n}", origin="human")
        for k in range(steps):
            s.add_message("assistant", "", tool_calls=[{"id": f"c{n}{k}", "type": "function",
                          "function": {"name": "x", "arguments": "{}"}}],
                          reasoning_content="think " * 200, origin="model")
            s.add_message("tool", "data", tool_call_id=f"c{n}{k}", tool_name="x", origin="tool")
        s.add_message("assistant", f"answer {n}", reasoning_content="r " * 5, origin="model")


@pytest.mark.asyncio
async def test_the_keep_window_weighs_what_is_sent_not_dropped_reasoning():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 400            # threshold 320, keep target ~ ratio x 400
    set_config(cfg)
    try:
        agent = _agent(_ThinkingLiteLLM(provider="deepseek", model="deepseek-test", api_key="k"))
        agent.provider.count_tokens = _words
        _thinking_turns(agent.session, turns=40, steps=3)
        agent.session.add_message("user", "next", origin="human")
        agent.provider.complete = _Stub([LLMResponse(content="summary")]).complete
        compacted, stats = await agent.compact_session(force=False, trigger="auto")
        assert compacted
        kept_turns = sum(1 for m in agent.session.messages if m.get("role") == "user")
        assert kept_turns > 5          # tool-step reasoning (600 words a turn) isn't sent
    finally:
        set_config(old_cfg)


@pytest.mark.asyncio
async def test_compaction_fires_on_the_room_history_really_has():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 1000
    set_config(cfg)
    try:
        agent = _agent(_Stub([LLMResponse(content="earlier: chatted")]))
        for n in range(30):
            agent.session.add_message("user", f"question number {n} " + "word " * 10, origin="human")
            agent.session.add_message("assistant", f"answer {n} " + "word " * 10, origin="model")
        assert 600 < agent._session_token_count() < 800          # under 0.8 x 1000
        compacted, stats = await agent.compact_session(force=False, trigger="auto")
        assert not compacted
        # The system prompt and tool schemas leave history 500 tokens.
        agent.last_context_window = {"history_budget_tokens": 500}
        compacted, stats = await agent.compact_session(force=False, trigger="auto")
        assert compacted
    finally:
        set_config(old_cfg)


def test_a_fresh_agent_budgets_with_the_sessions_last_schema_size():
    agent = Agent(provider=_Stub())
    agent.session = Session(id="s1", name="d")
    agent.session.metadata["context_window"] = {"tool_schema_tokens": 321}
    agent._build_env_now_text = lambda: ""
    agent.session.add_message("user", "hi", origin="human")
    agent._build_messages(query="hi")
    assert agent.last_context_window["tool_schema_tokens_budgeted"] == 321


@pytest.mark.asyncio
async def test_a_forced_compaction_reports_stored_sizes():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 400
    set_config(cfg)
    try:
        agent = _agent(_Stub([LLMResponse(content="a summary of what happened " * 20)]))
        _session_with_tool_output(agent, words_per_tool=100, tools=6)
        sent_before = agent._session_token_count()
        compacted, stats = await agent.compact_session(force=True, trigger="manual")
        # The summary outweighs the terse text it replaced, but six tool
        # outputs were folded: stored sizes say so.
        assert compacted and stats["after_tokens"] < stats["before_tokens"]
        assert agent._session_token_count() > sent_before - 30
    finally:
        set_config(old_cfg)


def test_reasoning_counts_only_for_providers_known_to_carry_it():
    msg = {"role": "assistant", "content": "one two", "reasoning_content": "think " * 30}
    for provider, model, counted in (("deepseek", "deepseek-chat", True), ("gemini", "gemini-pro", True),
                                     ("openrouter", "x-ai/grok-4", False), ("openai", "gpt-test", False),
                                     ("openrouter", "deepseek/deepseek-r1", True)):
        agent = Agent(provider=_ThinkingLiteLLM(provider=provider, model=model, api_key="k"))
        assert (agent._wire_token_count(dict(msg)) == 32) is counted, (provider, model)


def test_earlier_openers_weigh_without_their_surface_rules():
    agent = Agent(provider=_Stub())
    agent.session = Session(id="s1", name="d")
    surface = ("[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply.]\n"
               + "rule " * 50 + "\nUSER MESSAGE:\n")
    agent.session.add_message("user", surface + "plan it", origin="human")
    agent.session.add_message("assistant", "ok", origin="model")
    agent.session.add_message("user", "and now?", origin="human")
    assert agent._session_token_count() < 20
