"""The course check inside the agent loop (_course_tick, relation, request)."""

from __future__ import annotations

import asyncio

import pytest

from captain_claw import jev_course, task_course
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
    agent._test_gates = []
    agent.tool_output_callback = lambda name, args, out: (
        agent._test_gates.append(out) if name == "completion_gate" else None)
    return agent


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.fixture
def jev_on(monkeypatch):
    calls: list[dict] = []
    answers: list[jev_course.Check | None] = []

    async def ready(switch=None) -> bool:
        return True

    async def ask_check(state, *, step, trigger):
        calls.append({"state": state, "step": step, "trigger": trigger})
        return answers.pop(0) if answers else None

    monkeypatch.setattr(jev_course, "enabled", lambda: True)
    monkeypatch.setattr(jev_course, "ready", ready)
    monkeypatch.setattr(jev_course, "ask_check", ask_check)
    return calls, answers


def _gates(agent: Agent) -> list[str]:
    return list(agent._test_gates)


async def _steps(agent: Agent, n: int, start: int, turn_start: int, fail: bool = False) -> int:
    count = start
    for i in range(n):
        agent.session.add_message("tool", "Error: nope" if fail else f"result {count}",
                                  tool_call_id=f"c{count}", tool_name="web_search")
        count += 1
        await agent._course_tick(count, count, turn_start)
    return count


@pytest.mark.asyncio
async def test_a_check_after_six_steps_adds_the_stage_and_one_line(jev_on):
    calls, answers = jev_on
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    answers.append(jev_course.Check(step=6, stage=3.0, stage_name="producing", on_track=0.1,
                                    next="refocus", next_confidence=0.9, trigger="steps"))
    await agent._course_tick(0, 0, 0)
    count = await _steps(agent, 6, 0, 0)
    await _settle()
    assert len(calls) == 1 and calls[0]["trigger"] == "steps" and calls[0]["step"] == 6
    assert calls[0]["state"]["task"] == TASK

    count = await _steps(agent, 1, count, 0)
    live = agent._task_course_live
    assert live["stage"] == "Stage: producing (3 of 5), checked 1 step ago"
    assert live["line"] == jev_course.LINES[jev_course.LINE_REFOCUS]
    checks = agent.session.metadata[task_course.KEY]["checks"]
    assert checks[-1]["line"] == "refocus" and checks[-1]["next"] == "refocus"
    assert any("step=jev_course\n" in g for g in _gates(agent))

    # The line is shown for one call; the stage stays.
    await _steps(agent, 1, count, 0)
    assert "line" not in agent._task_course_live
    assert agent._task_course_live["stage"].startswith("Stage: producing")


@pytest.mark.asyncio
async def test_no_jev_means_no_check_and_the_reminder_goes_on(monkeypatch):
    async def not_ready(switch=None) -> bool:
        return False

    asked: list = []

    async def ask_check(state, *, step, trigger):
        asked.append(step)

    monkeypatch.setattr(jev_course, "enabled", lambda: True)
    monkeypatch.setattr(jev_course, "ready", not_ready)
    monkeypatch.setattr(jev_course, "ask_check", ask_check)
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    count = await _steps(agent, 6, 0, 0)
    await _settle()
    await _steps(agent, 7, count, 0)
    await _settle()
    assert asked == []
    assert agent._course_watch.disabled is True
    assert agent._task_course_live is None
    note = task_course.note_for_turn(agent.session.metadata, agent.session.messages)
    assert note.startswith("Task (the person's request)")


@pytest.mark.asyncio
async def test_switch_off_starts_nothing(jev_on, monkeypatch):
    calls, _answers = jev_on
    monkeypatch.setattr(jev_course, "enabled", lambda: False)
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    await _steps(agent, 8, 0, 0)
    await _settle()
    assert calls == []


@pytest.mark.asyncio
async def test_three_failures_in_a_row_trigger_a_check(jev_on):
    calls, _answers = jev_on
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    await _steps(agent, 3, 0, 0, fail=True)
    await _settle()
    assert [c["trigger"] for c in calls] == ["failures"]


@pytest.mark.asyncio
async def test_a_fold_triggers_a_check(jev_on):
    calls, _answers = jev_on
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    count = await _steps(agent, 1, 0, 0)
    agent._course_mark_folded()
    agent.session.add_message("tool", "after fold", tool_call_id="cx", tool_name="read")
    await agent._course_tick(count + 1, count + 1, 0)
    await _settle()
    assert [c["trigger"] for c in calls] == ["fold"]


@pytest.mark.asyncio
async def test_jev_relation_replaces_the_guess(monkeypatch):
    async def ask_relation(state):
        assert state["new_message"] == "also include Porto"
        return task_course.NEW_TASK, 0.9, {"new_task": 0.9, "amends": 0.08}

    monkeypatch.setattr(jev_course, "enabled", lambda: True)
    monkeypatch.setattr(jev_course, "ask_relation", ask_relation)
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.session.add_message("assistant", "Here is the table.")
    agent.session.add_message("user", "also include Porto")
    agent._task_course_begin("also include Porto")
    assert agent._task_course_start.relation == task_course.AMENDS
    await agent._course_watch.relation_task
    assert agent.session.metadata[task_course.KEY]["anchor"] == "also include Porto"
    assert agent._task_course_start.relation == task_course.NEW_TASK
    await agent._course_tick(0, 0, 3)
    assert any("step=jev_course_relation" in g for g in _gates(agent))


@pytest.mark.asyncio
async def test_unsure_relation_keeps_the_guess(monkeypatch):
    async def ask_relation(state):
        return task_course.NEW_TASK, 0.4, {}

    monkeypatch.setattr(jev_course, "enabled", lambda: True)
    monkeypatch.setattr(jev_course, "ask_relation", ask_relation)
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.session.add_message("assistant", "Here is the table.")
    agent.session.add_message("user", "also include Porto")
    agent._task_course_begin("also include Porto")
    await agent._course_watch.relation_task
    assert agent.session.metadata[task_course.KEY]["anchor"] == TASK


@pytest.mark.asyncio
async def test_continue_words_skip_the_relation_question(monkeypatch):
    asked: list = []

    async def ask_relation(state):
        asked.append(state)
        return None, None, {}

    monkeypatch.setattr(jev_course, "enabled", lambda: True)
    monkeypatch.setattr(jev_course, "ask_relation", ask_relation)
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.session.add_message("assistant", "Done half.")
    agent.session.add_message("user", "continue")
    agent._task_course_begin("continue")
    assert agent._course_watch.relation_task is None
    assert asked == []


def test_judges_read_the_anchor_on_a_continue_turn():
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.session.add_message("assistant", "Half done.")
    agent.session.add_message("user", "continue")
    agent._task_course_begin("continue")
    assert agent._course_request("continue").startswith(f"Task (the person's request): {TASK}")
    agent.session.metadata.clear()
    assert agent._course_request("continue") == "continue"
    # A plain turn reads as it always did.
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    assert agent._course_request(TASK) == TASK


@pytest.mark.asyncio
async def test_finish_cancels_the_check_in_flight(jev_on):
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    pending = asyncio.ensure_future(asyncio.sleep(10))
    agent._course_watch.task = pending
    agent._task_course_live = {"stage": "x"}
    agent._course_finish()
    await asyncio.sleep(0)
    assert pending.cancelled()
    assert agent._task_course_live is None


def test_a_workers_share_of_the_run_cap():
    agent = _agent()
    agent._course_frame = {"max_checks": 3}
    assert agent._course_max_checks() == 3
    agent._course_frame = {"max_checks": 50}
    assert agent._course_max_checks() == jev_course.MAX_CHECKS_PER_TURN
    agent._course_frame = None
    assert agent._course_max_checks() == jev_course.MAX_CHECKS_PER_TURN


@pytest.mark.asyncio
async def test_a_workers_frame_sets_goal_part_switch_and_share(jev_on):
    calls, _answers = jev_on
    agent = _agent()
    prompt = "You are the Prices owner. Long instructions follow. " * 20
    agent.session.add_message("user", prompt)
    with task_course.turn_frame({"goal": "Plan the trip", "part": "Prices: collect fares",
                                 "check": True, "max_checks": 2}):
        agent._task_course_begin(prompt)
    course = agent.session.metadata[task_course.KEY]
    assert course["goal"] == "Plan the trip" and course["anchor"] == "Prices: collect fares"
    assert agent._task_course_start.relation == task_course.NEW_TASK
    assert agent._course_watch.max_checks == 2
    assert agent._course_watch.relation_task is None
    count = await _steps(agent, 6, 0, 0)
    await _settle()
    assert len(calls) == 1 and calls[0]["state"]["run_goal"] == "Plan the trip"


@pytest.mark.asyncio
async def test_a_runs_switch_off_beats_the_workers_own(jev_on):
    calls, _answers = jev_on
    agent = _agent()
    agent.session.add_message("user", "do your part")
    with task_course.turn_frame({"goal": "Plan the trip", "part": "Prices", "check": False}):
        agent._task_course_begin("do your part")
    await _steps(agent, 8, 0, 0)
    await _settle()
    assert calls == []


# ── per-turn records (phase 4) ───────────────────────────────────────


def _tool_msg(agent: Agent, path: str, n: int) -> None:
    agent.session.add_message("tool", "ok", tool_call_id=f"t{n}", tool_name="read",
                              tool_arguments={"path": path})


def test_a_tool_turn_leaves_a_record_with_the_follow_rate():
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.last_turn_start_idx = 0
    for i in range(3):
        _tool_msg(agent, "a.md", i)
    watch = agent._course_watch
    watch.tool_steps = 5
    watch.checks_done = 1
    watch.latest = jev_course.Check(step=3, stage=2.0, stage_name="gathering", on_track=0.1,
                                    next="refocus", next_confidence=0.9)
    watch.line_marks.append(("refocus", 3))
    for i in range(3, 5):
        _tool_msg(agent, "b.md", i)
    agent._course_finish("Here is the table.", True, {"total_tokens": 1234})
    (stats,) = agent.session.metadata[task_course.KEY]["turn_stats"]
    assert stats["outcome"] == "finished" and stats["tool_calls"] == 5 and stats["tokens"] == 1234
    assert stats["checks"] == 1 and stats["jev"] is True and stats["final_stage"] == 2.0
    assert stats["relation"] == task_course.NEW_TASK and stats["worker"] is False
    assert stats["lines"] == [{"line": "refocus", "at": 3, "followed": True}]
    assert agent.session.metadata[task_course.KEY]["followed"] == {"shown": 1, "followed": 1, "unknown": 0}
    assert any("step=course_turn" in g and "followed=1" in g for g in _gates(agent))


def test_a_plain_reply_leaves_no_record():
    agent = _agent()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.last_turn_start_idx = 0
    agent._course_finish("Sure.", True, {})
    assert "turn_stats" not in agent.session.metadata[task_course.KEY]


# ── Jev's decisions as narration ─────────────────────────────────────


def _narrations(agent: Agent) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []

    def cb(text, iteration=0, source=""):
        seen.append((source, text))

    agent.narration_callback = cb
    return seen


def test_jev_narration_reaches_only_callbacks_that_take_a_source():
    agent = _agent()
    seen = _narrations(agent)
    assert agent._emit_jev_narration("Jev checked the course.") is True
    assert seen == [("jev", "Jev checked the course.")]
    plain: list = []
    agent.narration_callback = lambda text, iteration=0: plain.append(text)  # Telegram's shape
    assert agent._emit_jev_narration("Jev checked the course.") is False
    assert plain == []
    agent.narration_callback = None
    assert agent._emit_jev_narration("x") is False


@pytest.mark.asyncio
async def test_a_course_check_is_narrated(jev_on):
    _calls, answers = jev_on
    agent = _agent()
    seen = _narrations(agent)
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    answers.append(jev_course.Check(step=6, stage=4.0, stage_name="checking", on_track=0.93, next="continue",
                                    next_confidence=0.61, trigger="steps"))
    count = await _steps(agent, 6, 0, 0)
    await _settle()
    await _steps(agent, 1, count, 0)
    jev_lines = [t for src, t in seen if src == "jev"]
    assert jev_lines == ["Jev checked the course after step 6: on track 93%; stage 4 of 5, checking; "
                         "next: keep going (61%). No reminder needed."]
    # Never saved into the session.
    assert not any("Jev checked" in str(m.get("content")) for m in agent.session.messages)


@pytest.mark.asyncio
async def test_the_relation_is_narrated(monkeypatch):
    async def ask_relation(state):
        return task_course.NEW_TASK, 0.9, {"new_task": 0.9, "amends": 0.08}

    monkeypatch.setattr(jev_course, "enabled", lambda: True)
    monkeypatch.setattr(jev_course, "ask_relation", ask_relation)
    agent = _agent()
    seen = _narrations(agent)
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.session.add_message("assistant", "Here is the table.")
    agent.session.add_message("user", "also include Porto")
    agent._task_course_begin("also include Porto")
    await agent._course_watch.relation_task
    await agent._course_tick(0, 0, 3)
    assert seen == [("jev", "Jev read your message as a new task (90%; an addition to the task 8%) "
                            "and updated the task reminder.")]


# ── the turn record is saved, not left in memory ─────────────────────


@pytest.mark.asyncio
async def test_the_turn_record_is_saved_right_after_it_is_written():
    saved: list[dict] = []

    class _Manager:
        async def save_session(self, session):
            saved.append(dict(session.metadata[task_course.KEY]))

    agent = _agent()
    agent.session_manager = _Manager()
    agent.session.add_message("user", TASK)
    agent._task_course_begin(TASK)
    agent.last_turn_start_idx = 0
    _tool_msg(agent, "a.md", 0)
    agent._course_finish("Done.", True, {"total_tokens": 10})
    await _settle()
    assert saved and saved[-1]["turn_stats"][0]["outcome"] == "finished"
