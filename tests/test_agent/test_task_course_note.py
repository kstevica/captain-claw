"""The task anchor in the request (task_course + _build_messages)."""

from __future__ import annotations

from captain_claw import task_course
from captain_claw.agent import Agent
from captain_claw.llm import LLMProvider, LLMResponse, Message, ToolDefinition
from captain_claw.session import Session

TASK = "Compare the three cheapest flights from Zagreb to Lisbon and write a table of prices."


class _StubProvider(LLMProvider):
    provider = "openai"
    model = "stub-model"

    async def complete(self, messages: list[Message], tools: list[ToolDefinition] | None = None,
                       temperature: float | None = None, max_tokens: int | None = None) -> LLMResponse:
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len([part for part in text.split() if part]) or 1


def _agent() -> Agent:
    agent = Agent(provider=_StubProvider())
    agent.session = Session(id="s1", name="default")
    agent._build_env_now_text = lambda: ""
    return agent


def _call(call_id: str) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": "dummy", "arguments": "{\"value\": \"v\"}"}}


def _start_turn(agent: Agent, text: str = TASK) -> int:
    turn_start = len(agent.session.messages)
    agent.session.add_message("user", text)
    agent._task_course_begin(text)
    return turn_start


def test_begin_turn_keeps_the_anchor_in_session_metadata():
    agent = _agent()
    _start_turn(agent)
    course = agent.session.metadata[task_course.KEY]
    assert course["anchor"] == TASK
    assert course["source"] == {"turn_msg": 0}
    agent.session.add_message("assistant", "Here is the table.")
    _start_turn(agent, "continue")
    assert agent.session.metadata[task_course.KEY]["anchor"] == TASK
    assert agent._task_course_start.relation == task_course.CONTINUES


def test_first_call_carries_no_note():
    agent = _agent()
    turn_start = _start_turn(agent)
    messages = agent._build_messages(tool_messages_from_index=turn_start, query=TASK)
    assert not any("Task (the person's request)" in m.content for m in messages)
    assert agent.last_context_window["task_course_note_used"] == 0


def test_after_a_tool_call_the_note_rides_on_the_tool_result():
    agent = _agent()
    turn_start = _start_turn(agent)
    agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
    agent.session.add_message("tool", "result one", tool_call_id="c1", tool_name="dummy")

    messages = agent._build_messages(tool_messages_from_index=turn_start, query=TASK)

    last = messages[-1]
    assert last.role == "tool"
    assert last.content.startswith("result one")
    assert f"Task (the person's request): {TASK}" in last.content
    assert sum("Task (the person's request)" in m.content for m in messages) == 1
    assert agent.last_context_window["task_course_note_used"] == 1
    # Never saved into the session.
    assert agent.session.messages[-1]["content"] == "result one"
    assert not any("Task (the person's request)" in str(m.get("content")) for m in agent.session.messages)


def test_note_comes_before_the_plan_note_and_carries_live_lines():
    agent = _agent()
    turn_start = _start_turn(agent)
    agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
    agent.session.add_message("tool", "result one", tool_call_id="c1", tool_name="dummy")
    agent._task_course_live = {"stage": "Stage: gathering (2 of 5)"}
    pipeline = {"tasks": [{"id": "t1", "title": "Gather prices", "status": "in_progress"}],
                "current_index": 0, "state": "active"}

    content = agent._build_messages(
        tool_messages_from_index=turn_start, query=TASK, planning_pipeline=pipeline,
    )[-1].content

    assert "Stage: gathering (2 of 5)" in content
    assert content.index("Task (the person's request)") < content.index("Planning mode is ON")


def test_anchor_off_sends_nothing(monkeypatch):
    agent = _agent()
    turn_start = _start_turn(agent)
    agent.session.add_message("assistant", "", tool_calls=[_call("c1")])
    agent.session.add_message("tool", "result one", tool_call_id="c1", tool_name="dummy")
    monkeypatch.setattr(task_course, "enabled", lambda: False)
    messages = agent._build_messages(tool_messages_from_index=turn_start, query=TASK)
    assert not any("Task (the person's request)" in m.content for m in messages)
