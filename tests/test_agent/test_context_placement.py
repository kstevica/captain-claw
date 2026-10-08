"""Where context notes sit in the request, and the shape of earlier turns.

Weak models misbehave once a session gets crowded. Three causes are pinned
here: context notes that trailed the user's question as assistant turns,
the turn start going stale on the turn where auto-compaction runs (the model
then lost every tool result of that turn), and earlier turns' tool steps
replayed as empty or announce-and-stop assistant messages.
"""

from __future__ import annotations

import pytest

from captain_claw.agent import Agent
from captain_claw.config import get_config, set_config
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolCall, ToolDefinition
from captain_claw.platform_adapter import effective_turn_start_idx
from captain_claw.session import Session
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult


class _StubProvider(LLMProvider):
    provider = "openai"
    model = "stub-model"

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="ok")

    async def complete_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len([part for part in text.split() if part]) or 1


class _AnthropicStubProvider(_StubProvider):
    provider = "anthropic"


class _DummySessionManager:
    async def save_session(self, session: Session) -> None:
        return None


class _DummyTool(Tool):
    name = "dummy"
    description = "Dummy test tool"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, content="dummy tool output")


def _call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "dummy", "arguments": "{\"value\": \"v\"}"},
    }


def _agent(provider: LLMProvider | None = None) -> Agent:
    agent = Agent(provider=provider or _StubProvider())
    agent.session = Session(id="s1", name="default")
    # The clock note is pinned in the block; leave it out of these budget
    # sums (it has its own tests in test_context_engine_p0.py).
    agent._build_env_now_text = lambda: ""
    return agent


def _followup_session(agent: Agent) -> int:
    """An earlier turn with a tool result, then the new question."""
    agent.session.add_message("user", "extract titles")
    agent.session.add_message("tool", "Title: Alpha", tool_name="web_fetch")
    agent.session.add_message("assistant", "Found Alpha.")
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", "details for Alpha")
    return turn_start


# ── Fix 1: notes ride on the turn's question ─────────────────────────


@pytest.mark.parametrize("provider_cls", [_StubProvider, _AnthropicStubProvider])
def test_background_notes_ride_on_the_question_and_request_ends_on_it(provider_cls):
    agent = _agent(provider_cls())
    agent._build_todo_context_note = lambda: "Pending todos: water the plants"
    turn_start = _followup_session(agent)

    messages = agent._build_messages(
        tool_messages_from_index=turn_start, query="details for Alpha",
    )

    last = messages[-1]
    assert last.role == "user"
    assert last.content.endswith("details for Alpha")
    assert last.content.startswith("[INTERNAL CONTEXT")
    assert "Continuity note from earlier tool outputs" in last.content
    assert "water the plants" in last.content
    assert last.content.index("[END INTERNAL CONTEXT]") < last.content.index("details for Alpha")
    assert "[System context]" not in last.content
    # One block, and nothing else carries the notes (the system prompt
    # names the marker in its own guidance, so it is left out).
    joined = "\n".join(m.content for m in messages[1:])
    assert joined.count("[INTERNAL CONTEXT") == 1
    assert sum("water the plants" in m.content for m in messages) == 1
    assert agent.last_context_window["context_notes_used"] == 2


def test_notes_ride_on_the_turn_opener_not_on_a_later_nudge():
    agent = _agent()
    agent._build_todo_context_note = lambda: "Pending todos: water the plants"
    agent.session.add_message("user", "earlier question")
    agent.session.add_message("assistant", "earlier answer")
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", "do the thing")
    agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
    agent.session.add_message("tool", "result one", tool_call_id="c1", tool_name="dummy")
    agent.session.add_message("user", "[system] STOP repeating that call")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")

    opener = next(m for m in messages if m.role == "user" and m.content.endswith("do the thing"))
    assert "water the plants" in opener.content
    nudge = next(m for m in messages if "STOP repeating" in m.content)
    assert "water the plants" not in nudge.content
    assert messages[-1] is nudge


_PIPELINE = {
    "tasks": [{"id": "task_1", "title": "Gather constraints", "status": "in_progress"}],
    "current_index": 0,
    "state": "active",
}


def test_live_task_state_rides_on_the_question_before_any_tool_call():
    agent = _agent()
    agent.session.add_message("user", "help me plan deployment")

    messages = agent._build_messages(query="deployment", planning_pipeline=_PIPELINE)

    last = messages[-1]
    assert last.role == "user"
    assert last.content.startswith("help me plan deployment")
    assert last.content.index("help me plan deployment") < last.content.index("Planning mode is ON")
    assert last.content.endswith("[END INTERNAL CONTEXT]")
    assert not any(m.role == "assistant" for m in messages)
    assert agent.last_context_window["planning_note_used"] == 1


def test_live_task_state_rides_on_the_last_tool_result():
    agent = _agent()
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", "do the thing")
    agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
    agent.session.add_message("tool", "result one", tool_call_id="c1", tool_name="dummy")

    messages = agent._build_messages(
        tool_messages_from_index=turn_start, query="do the thing", planning_pipeline=_PIPELINE,
    )

    # The request still ends on the tool result (no new user turn after the
    # chain), with the task state as its suffix.
    last = messages[-1]
    assert last.role == "tool"
    assert last.tool_call_id == "c1"
    assert last.content.startswith("result one")
    assert "Planning mode is ON" in last.content
    assert sum("Planning mode is ON" in m.content for m in messages) == 1
    # The session's tool message itself is untouched.
    assert agent.session.messages[-1]["content"] == "result one"


def test_background_notes_are_frozen_for_the_turn():
    agent = _agent()
    todo = ["Pending todos: first"]
    agent._build_todo_context_note = lambda: todo[0]
    agent.session.add_message("user", "earlier question")
    agent.session.add_message("assistant", "earlier answer")
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", "do the thing")

    first = agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")
    # Mid-turn: the tool loop runs and a note's source changes.
    todo[0] = "Pending todos: second"
    agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
    agent.session.add_message("tool", "result one", tool_call_id="c1", tool_name="dummy")
    second = agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")

    opener_1 = next(m for m in first if m.content.endswith("do the thing"))
    opener_2 = next(m for m in second if m.content.endswith("do the thing"))
    assert opener_1.content == opener_2.content
    assert "first" in opener_2.content
    # complete() appends advisories to the query mid-turn; the notes stay put.
    agent._turn_user_text = "do the thing"
    agent._turn_context_notes = None
    base = agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")
    todo[0] = "Pending todos: third"
    advised = agent._build_messages(
        tool_messages_from_index=turn_start, query="do the thing\n--- SCALE ADVISORY ---",
    )
    opener_base = next(m for m in base if m.content.endswith("do the thing"))
    opener_advised = next(m for m in advised if m.content.endswith("do the thing"))
    assert opener_base.content == opener_advised.content
    # A new turn (complete()/stream() reset the cache) renders it afresh.
    agent._turn_context_notes = None
    third = agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")
    assert "third" in next(m for m in third if m.content.endswith("do the thing")).content


def test_notes_outrank_the_tool_chain_and_drop_one_by_one():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 140
    set_config(cfg)
    try:
        agent = _agent()
        agent._build_system_prompt = lambda: "sys"
        agent._build_todo_context_note = lambda: "Pending todos: water the plants"
        agent._build_insights_context_note = lambda: "Persistent insights: " + "big " * 200
        agent.session.add_message("user", "old question")
        agent.session.add_message("assistant", "old answer")
        turn_start = len(agent.session.messages)
        agent.session.add_message("user", "do the thing")
        for n in range(1, 6):
            agent.session.add_message("assistant", "", tool_calls=[_call(f"c{n}")])
            agent.session.add_message("tool", f"result {n} " + "data " * 20,
                                      tool_call_id=f"c{n}", tool_name="dummy")

        messages = agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")

        opener = next(m for m in messages if m.content.endswith("do the thing"))
        assert "water the plants" in opener.content       # the small note survives
        assert "Persistent insights" not in opener.content  # the oversized one is left out
        assert agent.last_context_window["todo_note_used"] == 1
        assert agent.last_context_window["context_notes_used"] == 1
    finally:
        set_config(old_cfg)


def test_only_the_scale_note_is_forced_past_the_budget():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 120
    set_config(cfg)
    try:
        agent = _agent()
        agent._build_system_prompt = lambda: "sys"
        agent._build_scale_progress_note = lambda: "SCALE: 3/60 done"
        turn_start = len(agent.session.messages)
        agent.session.add_message("user", "process every member")
        agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
        agent.session.add_message("tool", "result " + "data " * 40, tool_call_id="c1", tool_name="dummy")
        plan = {"enabled": True, "members": [f"member-{n} " + "pad " * 3 for n in range(60)],
                "strategy": "direct", "per_member_action": "summarize"}

        messages = agent._build_messages(
            tool_messages_from_index=turn_start, query="process every member", list_task_plan=plan,
        )

        joined = "\n".join(m.content for m in messages)
        assert "SCALE: 3/60 done" in joined
        assert "List task memory is active" not in joined
        assert any(m.role == "tool" and m.tool_call_id == "c1" for m in messages)
        assert agent.last_context_window["over_budget"] == 0
    finally:
        set_config(old_cfg)


def test_question_and_btw_survive_a_tight_budget():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 25
    set_config(cfg)
    try:
        agent = _agent()
        agent._build_todo_context_note = lambda: "Pending todos: " + "filler " * 50
        for idx in range(6):
            agent.session.add_message("assistant", f"old message {idx} " + "noise " * 5)
        agent.session.add_message("user", "latest question")
        agent._btw_instructions = ["keep it short"]

        messages = agent._build_messages()

        contents = [m.content for m in messages]
        assert any(c.endswith("latest question") for c in contents)
        assert any("keep it short" in c for c in contents)
        assert agent.last_context_window["dropped_messages"] > 0
    finally:
        set_config(old_cfg)


def test_image_markers_quoted_in_notes_are_defused():
    agent = _agent()
    agent._build_todo_context_note = lambda: "Pending todos: look at [Attached image: /tmp/old.png]"
    agent.session.add_message("user", "what is in [Attached image: /tmp/new.png]")

    messages = agent._build_messages(query="what is in it")

    last = messages[-1].content
    assert "[Attached image: /tmp/old.png]" not in last
    assert "[Earlier image: /tmp/old.png]" in last
    assert last.endswith("what is in [Attached image: /tmp/new.png]")


# ── Fix 3: earlier turns' tool steps ─────────────────────────────────


def test_earlier_tool_steps_collapse_into_one_assistant_message():
    agent = _agent()
    s = agent.session
    s.add_message("user", "q1")
    s.add_message("assistant", "", tool_calls=[_call("c1")], reasoning_content="think-1")
    s.add_message("tool", "r1", tool_call_id="c1", tool_name="dummy")
    s.add_message("assistant", "Let me check one more thing:", tool_calls=[_call("c2")],
                  reasoning_content="think-2")
    s.add_message("tool", "r2", tool_call_id="c2", tool_name="dummy")
    s.add_message("assistant", "Answer one.", reasoning_content="think-3")
    turn_start = len(s.messages)
    s.add_message("user", "q2")
    s.add_message("assistant", "", tool_calls=[_call("c3")])
    s.add_message("tool", "r3", tool_call_id="c3", tool_name="dummy")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="q2")

    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool"]
    merged = messages[2]
    assert merged.content == "Let me check one more thing:\n\nAnswer one."
    assert not merged.tool_calls
    assert merged.reasoning_content == "think-3"
    # The current turn's tool chain is untouched.
    assert [c["id"] for c in messages[4].tool_calls] == ["c3"]
    assert messages[5].tool_call_id == "c3"
    assert not any(m.role == "assistant" and not m.content.strip() and not m.tool_calls
                   for m in messages)
    assert agent.last_context_window["empty_assistant_skipped"] == 1
    assert agent.last_context_window["historical_assistant_merged"] == 1
    # The session itself is not rewritten.
    assert s.messages[3]["content"] == "Let me check one more thing:"
    assert s.messages[3]["tool_calls"]


def test_compaction_summary_and_hints_are_not_lost_when_merging():
    agent = _agent()
    s = agent.session
    s.add_message("assistant", "Conversation summary of earlier messages", tool_name="compaction_summary")
    s.add_message("assistant", "Step one.", system_hint="(hint one)")
    s.add_message("assistant", "Step two.")
    turn_start = len(s.messages)
    s.add_message("user", "next")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="next")

    assistants = [m.content for m in messages if m.role == "assistant"]
    assert assistants == [
        "Conversation summary of earlier messages",
        "Step one.\n(hint one)\n\nStep two.",
    ]


def test_current_turn_assistant_messages_are_not_merged():
    agent = _agent()
    s = agent.session
    turn_start = len(s.messages)
    s.add_message("user", "go")
    s.add_message("assistant", "Working on it.")
    s.add_message("assistant", "Still working.")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="go")

    assert [m.content for m in messages if m.role == "assistant"] == ["Working on it.", "Still working."]


# ── Fix 2: turn start after compaction ───────────────────────────────


class _ToolThenAnswerProvider(_StubProvider):
    def __init__(self):
        self.tool_called = False
        self.followup_messages: list[Message] = []

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        if tools and not self.tool_called:
            self.tool_called = True
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="dummy", arguments={"value": "v"})],
            )
        if tools:
            self.followup_messages = list(messages)
            return LLMResponse(content="final")
        return LLMResponse(content="")


@pytest.mark.asyncio
async def test_compaction_turn_keeps_its_own_tool_results():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 200
    cfg.context.compaction_threshold = 0.8
    cfg.context.compaction_ratio = 0.4
    set_config(cfg)
    try:
        provider = _ToolThenAnswerProvider()
        agent = _agent(provider)
        agent._initialized = True
        agent.session_manager = _DummySessionManager()
        registry = ToolRegistry()
        registry.register(_DummyTool())
        agent.tools = registry
        agent._build_system_prompt = lambda: "You are a test agent."

        async def _summary(_messages):
            return "earlier stuff"

        agent._summarize_for_compaction = _summary
        # 240 counted tokens of history: over the 160-token threshold, so
        # the turn opens with an auto-compaction.
        for idx in range(8):
            role = "user" if idx % 2 == 0 else "assistant"
            agent.session.add_message(role, f"message {idx} " + "word " * 28)

        result = await agent.complete("run tool")

        assert result == "final"
        assert agent.session.metadata.get("compaction", {}).get("auto_count") == 1
        sent = provider.followup_messages
        tool_msgs = [m for m in sent if m.role == "tool"]
        assert [m.tool_call_id for m in tool_msgs] == ["c1"]
        prev = sent[sent.index(tool_msgs[0]) - 1]
        assert prev.role == "assistant"
        assert [c["id"] for c in prev.tool_calls] == ["c1"]
        start = agent.last_turn_start_idx
        assert agent.session.messages[start]["role"] == "user"
        assert agent.session.messages[start]["content"] == "run tool"
    finally:
        set_config(old_cfg)


def test_turn_start_after_compaction_finds_the_turn_message():
    agent = _agent()
    s = agent.session
    for idx in range(6):
        s.add_message("user" if idx % 2 == 0 else "assistant", f"m{idx}")
    captured = len(s.messages)
    s.add_message("user", "this turn")
    turn_msg = s.messages[-1]

    # No compaction: the captured start stands.
    assert agent._turn_start_after_compaction(turn_msg, captured) == captured

    summary = {"role": "assistant", "content": "summary", "tool_name": "compaction_summary"}
    s.messages = [summary, *s.messages[-3:]]
    assert agent._turn_start_after_compaction(turn_msg, captured) == 3

    # The turn's own message folded into the summary: the turn starts after
    # it, past leading tool results whose calls were folded with it.
    s.messages = [summary, {"role": "assistant", "content": "a"}]
    assert agent._turn_start_after_compaction(turn_msg, captured) == 1
    s.messages = [
        summary,
        {"role": "tool", "content": "r1", "tool_call_id": "c1"},
        {"role": "tool", "content": "r2", "tool_call_id": "c2"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c3")]},
        {"role": "tool", "content": "r3", "tool_call_id": "c3"},
    ]
    assert agent._turn_start_after_compaction(turn_msg, captured) == 3
    # captured_idx indexes the list before compaction: a fresh session's 0
    # must not pull the start back onto the orphans.
    assert agent._turn_start_after_compaction(turn_msg, 0) == 3


def test_effective_turn_start_idx_only_moves_back():
    class _A:
        last_turn_start_idx = None

    agent = _A()
    assert effective_turn_start_idx(agent, 12) == 12
    agent.last_turn_start_idx = 4
    assert effective_turn_start_idx(agent, 12) == 4
    agent.last_turn_start_idx = 15
    assert effective_turn_start_idx(agent, 12) == 12
    agent.last_turn_start_idx = True
    assert effective_turn_start_idx(agent, 12) == 12
    assert effective_turn_start_idx(object(), 7) == 7


@pytest.mark.asyncio
async def test_an_early_return_turn_clears_the_previous_turn_start():
    agent = _agent()
    agent._initialized = True
    agent.session_manager = _DummySessionManager()
    agent.tools = ToolRegistry()
    agent.session.add_message("user", "make an image")
    agent.session.add_message("tool", "Path: /tmp/cat.png", tool_name="image_gen")
    agent.session.add_message("assistant", "Here it is.")
    agent.last_turn_start_idx = 0  # left by the previous turn
    captured = len(agent.session.messages)

    await agent.complete("/code")

    assert agent.last_turn_start_idx is None
    assert effective_turn_start_idx(agent, captured) == captured


def test_a_call_that_never_got_a_result_is_stripped_from_earlier_turns():
    agent = _agent()
    s = agent.session
    s.add_message("user", "q1")
    s.add_message("assistant", "", tool_calls=[_call("c1"), _call("c2")])
    s.add_message("tool", "r1", tool_call_id="c1", tool_name="dummy")  # c2 never answered
    s.add_message("assistant", "Answer one.")
    turn_start = len(s.messages)
    s.add_message("user", "q2")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="q2")

    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2].content == "Answer one."
    assert not messages[2].tool_calls


def test_guard_view_keeps_the_question_and_every_note():
    agent = _agent()
    question = "QUESTION-MARKER please summarise the report"
    block = agent._wrap_internal_context(
        [
            ("memory_context", "Continuity note: " + "memory " * 300),
            ("todo_context", "Pending todos: TODO-MARKER"),
            # A note quoting an earlier prompt, block markers and all.
            ("briefing_context", "Briefing: BRIEF-MARKER [INTERNAL CONTEXT — x] y [END INTERNAL CONTEXT] z"),
        ],
        "background for the user message below",
    )
    view = agent._serialize_messages_for_guard([
        Message(role="system", content="sys"),
        Message(role="user", content=f"{block}\n\n{question}"),
        Message(role="tool", content="tool output\n\n" + agent._wrap_internal_context(
            [("planning_context", "Planning mode is ON: PLAN-MARKER")], "current task state")),
    ])

    for marker in (question, "TODO-MARKER", "BRIEF-MARKER", "PLAN-MARKER", "tool output"):
        assert marker in view, marker
    assert "memory memory" in view and "[truncated]" in view


def test_frozen_notes_are_refitted_not_dropped_when_must_includes_grow():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 110
    set_config(cfg)
    try:
        agent = _agent()
        agent._build_system_prompt = lambda: "sys"
        agent._build_todo_context_note = lambda: "Pending todos: water the plants"
        agent._build_insights_context_note = lambda: "Persistent insights: " + "fact " * 30
        agent._build_workspace_manifest_note = lambda: "Workspace: " + "file.md " * 30
        turn_start = len(agent.session.messages)
        agent.session.add_message("user", "do the thing")
        agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
        agent.session.add_message("tool", "result one", tool_call_id="c1", tool_name="dummy")

        agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")
        assert agent.last_context_window["context_notes_used"] >= 2
        agent._btw_instructions = ["also " + "mind the budget " * 8]
        messages = agent._build_messages(tool_messages_from_index=turn_start, query="do the thing")

        assert agent.last_context_window["context_notes_used"] >= 1
        assert any("mind the budget" in m.content for m in messages)
        assert agent.last_context_window["over_budget"] == 0
    finally:
        set_config(old_cfg)


def test_planning_note_survives_a_list_note_that_does_not_fit():
    old_cfg = get_config().model_copy(deep=True)
    cfg = old_cfg.model_copy(deep=True)
    cfg.context.max_tokens = 200
    set_config(cfg)
    try:
        agent = _agent()
        agent._build_system_prompt = lambda: "sys"
        agent.session.add_message("user", "plan and process every member")
        plan = {"enabled": True, "members": [f"member-{n} " + "pad " * 3 for n in range(60)],
                "strategy": "direct", "per_member_action": "summarize"}

        messages = agent._build_messages(
            query="plan", planning_pipeline=_PIPELINE, list_task_plan=plan,
        )

        joined = "\n".join(m.content for m in messages)
        assert "Planning mode is ON" in joined
        assert "List task memory is active" not in joined
    finally:
        set_config(old_cfg)


def test_block_with_no_user_message_opens_the_turn_instead_of_trailing_it():
    agent = _agent()
    agent._build_todo_context_note = lambda: "Pending todos: water the plants"
    s = agent.session
    s.messages.append({"role": "assistant", "content": "Conversation summary", "tool_name": "compaction_summary"})
    turn_start = len(s.messages)
    s.add_message("assistant", "", tool_calls=[_call("c3")])
    s.add_message("tool", "r3", tool_call_id="c3", tool_name="dummy")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query="go on")

    roles = [m.role for m in messages]
    assert roles == ["system", "assistant", "user", "assistant", "tool"]
    assert "water the plants" in messages[2].content
    assert messages[-1].tool_call_id == "c3"


class _QuarterCountProvider(_StubProvider):
    def count_tokens(self, text: str) -> int:
        return len(text) // 4


def test_fitted_notes_are_always_sent_with_a_coarse_token_counter():
    notes = ["Pending todos: " + "x" * 37, "Insights: " + "y" * 51, "Workspace: " + "z" * 63,
             "Contacts: " + "w" * 29, "Briefing: " + "v" * 45]
    for max_tokens in range(60, 260, 3):
        old_cfg = get_config().model_copy(deep=True)
        cfg = old_cfg.model_copy(deep=True)
        cfg.context.max_tokens = max_tokens
        set_config(cfg)
        try:
            agent = _agent(_QuarterCountProvider())
            agent._build_system_prompt = lambda: "sys"
            agent._collect_background_context_notes = (
                lambda *a, **k: [(f"n{i}", text) for i, text in enumerate(notes)]
            )
            agent.session.add_message("user", "go")
            agent._build_messages(query="go")
            frozen = agent._turn_context_notes[1]
            assert agent.last_context_window["context_notes_used"] == len(frozen), max_tokens
        finally:
            set_config(old_cfg)
