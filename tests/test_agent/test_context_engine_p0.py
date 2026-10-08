"""Context engine P0: the clock out of the system prompt, a stable prompt
prefix, the per-section trace, and compaction that weighs what is sent."""

from __future__ import annotations

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolDefinition,
    _history_breakpoint_index,
    _inject_anthropic_cache_control,
)
from captain_claw.session import Session


class _StubProvider(LLMProvider):
    provider = "openai"
    model = "stub-model"

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len([part for part in text.split() if part]) or 1


def _agent() -> Agent:
    agent = Agent(provider=_StubProvider())
    agent.session = Session(id="s1", name="default")
    return agent


# ── The clock rides in the per-turn block ─────────────────────────────


def test_clock_is_in_the_turn_block_not_the_system_prompt():
    agent = _agent()
    agent._build_env_now_text = lambda: "System environment:\n- Date/time: CLOCK-MARKER"
    agent.session.add_message("user", "earlier")
    agent.session.add_message("assistant", "answer")
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", "what time is it")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="what time is it")

    assert "CLOCK-MARKER" not in messages[0].content
    holders = [m for m in messages if "CLOCK-MARKER" in m.content]
    assert len(holders) == 1
    assert holders[0] is messages[-1]
    assert messages[-1].content.endswith("what time is it")
    assert agent.last_context_window["env_note_used"] == 1
    assert agent.last_context_window["context_notes_used"] == 0


def test_clock_survives_a_budget_too_small_for_any_note():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 20
    set_config(cfg)
    try:
        agent = _agent()
        agent._build_system_prompt = lambda: "sys"
        agent._build_env_now_text = lambda: "Env: CLOCK-MARKER"
        agent._build_todo_context_note = lambda: "Pending todos: " + "filler " * 40
        agent.session.add_message("user", "go")

        messages = agent._build_messages(query="go")

        assert "CLOCK-MARKER" in messages[-1].content
        assert "filler" not in messages[-1].content
        assert agent.last_context_window["context_notes_used"] == 0
    finally:
        set_config(old_cfg)


def test_system_prompt_keeps_turn_varying_blocks_after_the_cache_split():
    agent = _agent()
    prompt = agent._build_system_prompt()
    static, marker, _dynamic = prompt.partition("<!-- CACHE_SPLIT -->")
    assert marker
    assert "System environment:" not in prompt
    # The reflection and peer roster used to sit at the very top.
    assert "Other Available Agents" not in static
    assert "Self-reflection (latest self-assessment" not in static


def test_previous_user_message_time_is_kept():
    agent = _agent()
    agent.session.metadata["timing"] = {"last_user_msg_at": "2026-01-01T08:00:00+00:00"}
    agent._record_timing_event("last_user_msg_at")
    timing = agent.session.metadata["timing"]
    assert timing["prev_user_msg_at"] == "2026-01-01T08:00:00+00:00"
    assert "Previous user message:" in agent._build_timing_block()


def test_host_facts_are_cached_between_turns(monkeypatch):
    import captain_claw.system_info as si

    calls = {"n": 0}

    def slow_ip():
        calls["n"] += 1
        return "203.0.113.9"

    monkeypatch.setattr(si, "_get_public_ip", slow_ip)
    first = si.build_system_info_block("normal")
    second = si.build_system_info_block("normal")
    assert "203.0.113.9" in first and "203.0.113.9" in second
    assert calls["n"] == 1


# ── Anthropic: a breakpoint at the end of prior history ───────────────


def test_prior_history_end_is_flagged_and_becomes_a_breakpoint():
    agent = _agent()
    agent._build_env_now_text = lambda: ""
    agent.session.add_message("user", "earlier question")
    agent.session.add_message("assistant", "earlier answer")
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", "next question")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="next question")

    flagged = [m for m in messages if m.cache_breakpoint]
    assert [m.content for m in flagged] == ["earlier answer"]

    idx = _history_breakpoint_index(messages)
    payload = [{"role": m.role, "content": m.content} for m in messages]
    result = _inject_anthropic_cache_control(payload, history_breakpoint=idx)
    marked = [i for i, m in enumerate(result)
              if isinstance(m["content"], list)
              and any(isinstance(b, dict) and b.get("cache_control") for b in m["content"])]
    # system, the prior-history end, and the last message.
    assert marked == [0, idx, len(result) - 1]


def test_history_breakpoint_index_skips_roles_the_converter_drops():
    messages = [
        Message(role="system", content="s"),
        Message(role="developer", content="dropped"),
        Message(role="assistant", content="a", cache_breakpoint=True),
        Message(role="user", content="u"),
    ]
    assert _history_breakpoint_index(messages) == 1


# ── Trace ─────────────────────────────────────────────────────────────


def test_trace_reports_sections_and_tool_schemas():
    agent = _agent()
    agent._build_env_now_text = lambda: "Env: now"
    agent.session.add_message("user", "earlier question")
    agent.session.add_message("assistant", "earlier answer", reasoning_content="thinking " * 10)
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", "next question")

    agent._build_messages(tool_messages_from_index=turn_start, query="next question")
    window = agent.last_context_window
    sections = window["sections"]
    assert sections["prior_history"] > 0
    assert sections["turn_message"] > 0
    assert sections["reasoning_prior"] == 10
    assert window["reasoning_tokens"] == 10
    assert sections["env_note"] == 2

    tools = [{"name": "dummy", "description": "d", "parameters": {"type": "object"}}]
    agent._note_tool_schema_tokens(tools)
    assert window["tool_schema_tokens"] > 0
    assert window["tool_count"] == 1
    assert window["prompt_tokens_with_tools"] == window["prompt_tokens"] + window["tool_schema_tokens"]
    assert agent.session.metadata["context_window"]["tool_schema_tokens"] == window["tool_schema_tokens"]


# ── Compaction weighs what the model is sent ──────────────────────────


def test_compaction_ignores_debug_echoes_and_replayed_reasoning():
    agent = _agent()
    agent.session.add_message("user", "one two three")
    agent.session.add_message("tool", "dump " * 50, tool_name="memory_select")
    agent._add_session_message("assistant", "four five", reasoning_content="think " * 30)

    assert agent._session_token_count() == 5
    # Reasoning is counted on its own (the trace reports it).
    assert agent.session.messages[-1]["token_count"] == 2
    assert agent.session.messages[-1]["reasoning_token_count"] == 30


@pytest.mark.parametrize("name", ["memory_select", "pipeline_trace", "task_rephrase"])
def test_compaction_summary_input_skips_debug_echoes(name):
    agent = _agent()
    text = agent._format_compaction_messages([
        {"role": "user", "content": "real question"},
        {"role": "tool", "tool_name": name, "content": "ECHO"},
    ])
    assert "real question" in text and "ECHO" not in text


def test_anthropic_never_gets_more_than_four_cache_breakpoints():
    payload = [
        {"role": "system", "content": "main"},
        {"role": "system", "content": "dispatch context"},
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    result = _inject_anthropic_cache_control(payload, history_breakpoint=4)
    blocks = sum(
        1 for m in result if isinstance(m["content"], list)
        for b in m["content"] if isinstance(b, dict) and b.get("cache_control")
    )
    assert blocks == 4


def test_pipeline_export_keeps_old_rows_before_the_metadata_log():
    from captain_claw.session_export import collect_pipeline_trace_entries

    rows = [{"role": "tool", "tool_name": "pipeline_trace", "tool_arguments": {"source": "old"},
             "timestamp": "t0"}]
    entries = collect_pipeline_trace_entries(
        "s1", "default", rows, metadata={"pipeline_trace": [{"source": "new", "timestamp": "t1"}]},
    )
    assert [(e["source"], e["seq"]) for e in entries] == [("old", 1), ("new", 2)]
