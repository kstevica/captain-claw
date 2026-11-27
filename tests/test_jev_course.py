"""The Jev course check (captain_claw/jev_course.py): answers, lines, cadence."""

from __future__ import annotations

import asyncio

import pytest

from captain_claw import jev_course as jc
from captain_claw import task_course

TASK = {"status": "open", "anchor": "Compare flights to Lisbon", "amendments": ["add Porto"],
        "latest": "economy", "requirements": ["prices table"]}


def _resp(stage: str = "producing", probs: dict | None = None, on_track: float = 0.9,
          nxt: str = "continue", conf: float = 0.9) -> dict:
    return {"answers": {
        jc.Q_STAGE: {"type": "choice", "choice": stage, "probabilities": probs or {stage: 1.0},
                     "confidence": 0.9},
        jc.Q_ON_TRACK: {"type": "noul", "noul": on_track},
        jc.Q_NEXT: {"type": "choice", "choice": nxt, "confidence": conf},
    }}


# ── reading answers ──────────────────────────────────────────────────


def test_parse_check_reads_stage_on_track_and_next():
    check = jc.parse_check(_resp(), step=6, latency_ms=120, trigger="steps")
    assert check.stage == 3.0 and check.stage_name == "producing"
    assert check.on_track == 0.9 and check.next == "continue" and check.next_confidence == 0.9
    rec = check.record()
    assert rec["p_on_track"] == 0.9 and rec["trigger"] == "steps" and "line" not in rec


def test_stage_is_the_probability_weighted_level():
    check = jc.parse_check(_resp(probs={"gathering": 0.5, "producing": 0.5}), step=1)
    assert check.stage == 2.5
    assert check.stage_name in ("gathering", "producing")


def test_empty_answers_give_no_check():
    assert jc.parse_check({"answers": {}}, step=1) is None
    assert jc.parse_check(None, step=1) is None


def test_choice_confidence_falls_back_to_its_probability():
    resp = {"answers": {jc.Q_NEXT: {"type": "choice", "choice": "finish", "probabilities": {"finish": 0.8}}}}
    check = jc.parse_check(resp, step=1)
    assert check.next == "finish" and check.next_confidence == 0.8


# ── corrective lines ─────────────────────────────────────────────────


def _check(**kw) -> jc.Check:
    base = dict(step=6, stage=3.0, stage_name="producing", on_track=0.9, next="continue", next_confidence=0.9)
    base.update(kw)
    return jc.Check(**base)


def test_refocus_needs_low_on_track_and_a_confident_refocus():
    assert jc.corrective(_check(on_track=0.1, next="refocus", next_confidence=0.8), []) == jc.LINE_REFOCUS
    assert jc.corrective(_check(on_track=0.4, next="refocus", next_confidence=0.8), []) == ""
    assert jc.corrective(_check(on_track=0.1, next="refocus", next_confidence=0.5), []) == ""


def test_finish_needs_a_late_stage_and_a_confident_finish():
    assert jc.corrective(_check(stage=4.8, next="finish", next_confidence=0.75), []) == jc.LINE_FINISH
    assert jc.corrective(_check(stage=4.0, next="finish", next_confidence=0.9), []) == ""


def test_ask_needs_a_confident_ask():
    assert jc.corrective(_check(next="ask_person", next_confidence=0.7), []) == jc.LINE_ASK
    assert jc.corrective(_check(next="ask_person", next_confidence=0.6), []) == ""


def test_at_most_two_lines_and_never_the_same_twice_in_a_row():
    refocus = _check(on_track=0.1, next="refocus", next_confidence=0.9)
    assert jc.corrective(refocus, [jc.LINE_REFOCUS]) == ""
    assert jc.corrective(refocus, [jc.LINE_ASK]) == jc.LINE_REFOCUS
    assert jc.corrective(refocus, [jc.LINE_ASK, jc.LINE_FINISH]) == ""


def test_stage_and_status_text():
    check = _check(stage=2.4, stage_name="gathering")
    assert jc.stage_text(check, 0) == "Stage: gathering (2 of 5), just checked"
    assert jc.stage_text(check, 1) == "Stage: gathering (2 of 5), checked 1 step ago"
    assert jc.stage_text(check, 4) == "Stage: gathering (2 of 5), checked 4 steps ago"
    assert jc.status_text(check) == "Stage 2 of 5 · gathering"
    assert jc.stage_text(_check(stage=None, stage_name=None), 0) == ""


def test_budget_verdict():
    assert jc.budget_verdict(_check(on_track=0.7, next="continue"))
    assert not jc.budget_verdict(_check(on_track=0.7, next="finish"))
    assert not jc.budget_verdict(_check(on_track=0.5, next="continue"))


# ── state ────────────────────────────────────────────────────────────


def test_check_state_carries_the_anchor_and_recent_steps():
    steps = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "Searching."},
        {"role": "tool", "content": "3 results", "tool_name": "web_search", "tool_arguments": {"q": "lisbon"}},
    ]
    state = jc.check_state(TASK, steps, steps_used=7)
    assert state["task"] == "Compare flights to Lisbon"
    assert state["added_since"] == ["add Porto"]
    assert state["latest_message"] == "economy"
    assert state["requirements"] == ["prices table"]
    assert state["steps_used"] == 7
    assert "web_search" in state["recent_steps"]


def test_check_state_for_a_worker_names_the_run_goal():
    state = jc.check_state({**TASK, "goal": "Plan the trip"}, [], steps_used=1)
    assert state["run_goal"] == "Plan the trip" and state["task"] == "Compare flights to Lisbon"


def test_relation_state():
    state = jc.relation_state(TASK, "also Madrid", "Which dates?")
    assert state == {"open_task": "Compare flights to Lisbon", "added_since": ["add Porto"],
                     "agent_last_reply": "Which dates?", "new_message": "also Madrid"}


# ── the switch and readiness ─────────────────────────────────────────


def test_switch_file_then_config(tmp_path, monkeypatch):
    monkeypatch.setattr(jc, "SETTING_PATH", tmp_path / jc.FILENAME)
    monkeypatch.setattr(jc, "config_enabled", lambda: True)
    assert jc.enabled() is True
    (tmp_path / jc.FILENAME).write_text("off\n")
    assert jc.enabled() is False
    (tmp_path / jc.FILENAME).write_text("on")
    monkeypatch.setattr(jc, "config_enabled", lambda: False)
    assert jc.enabled() is True
    jc.clear()
    assert jc.enabled() is False


def test_config_default_is_on():
    from captain_claw.config import JevToolConfig

    assert JevToolConfig().course_check is True


@pytest.mark.asyncio
async def test_not_ready_without_switch_member_or_key(monkeypatch):
    from captain_claw import jev_client

    monkeypatch.setattr(jc, "enabled", lambda: True)
    monkeypatch.setattr(jev_client, "_member_call", lambda: False)

    async def connected(force: bool = False) -> bool:
        return True

    monkeypatch.setattr(jev_client, "refresh_connected", connected)
    assert await jc.ready() is True
    monkeypatch.setattr(jev_client, "_member_call", lambda: True)
    assert await jc.ready() is False
    monkeypatch.setattr(jev_client, "_member_call", lambda: False)
    monkeypatch.setattr(jc, "enabled", lambda: False)
    assert await jc.ready() is False


@pytest.mark.asyncio
async def test_ask_check_and_relation_go_through_jev(monkeypatch):
    asked: list[dict] = []

    async def ready() -> bool:
        return True

    async def ask(state, questions):
        asked.append(questions)
        if jc.Q_RELATION in questions:
            return {"answers": {jc.Q_RELATION: {"type": "choice", "choice": "amends", "confidence": 0.8}}}
        return _resp(nxt="refocus", on_track=0.1)

    monkeypatch.setattr(jc, "ready", ready)
    monkeypatch.setattr(jc, "_ask", ask)
    check = await jc.ask_check({"task": "x"}, step=6, trigger="steps")
    assert check.next == "refocus" and check.step == 6
    assert await jc.ask_relation({"x": 1}) == ("amends", 0.8)
    assert set(asked[0]) == {jc.Q_STAGE, jc.Q_ON_TRACK, jc.Q_NEXT}


@pytest.mark.asyncio
async def test_jev_errors_mean_no_check(monkeypatch):
    async def ready() -> bool:
        return True

    async def boom(state, questions):
        raise RuntimeError("down")

    monkeypatch.setattr(jc, "ready", ready)
    monkeypatch.setattr(jc, "_ask", boom)
    assert await jc.ask_check({}, step=1, trigger="steps") is None
    assert await jc.ask_relation({}) == (None, None)


def test_questions_validate():
    from captain_claw import jev_client

    jev_client.normalize_questions(jc.CHECK_QUESTIONS)
    jev_client.normalize_questions(jc.RELATION_QUESTIONS)


# ── cadence ──────────────────────────────────────────────────────────


def test_due_every_six_steps_after_six():
    w = jc.TurnWatch()
    for _ in range(5):
        w.note_tool_step(0, 1)
        assert w.due() == ""
    w.note_tool_step(0, 1)
    assert w.due() == "steps"


def test_due_after_a_fold_or_three_failures():
    w = jc.TurnWatch()
    w.note_tool_step(0, 1)
    w.folded = True
    assert w.due() == "fold"
    w = jc.TurnWatch()
    w.note_tool_step(2, 0)
    assert w.due() == ""
    w.note_tool_step(1, 0)
    assert w.due() == "failures"
    w2 = jc.TurnWatch()
    w2.note_tool_step(2, 0)
    w2.note_tool_step(0, 1)   # a success resets the streak
    assert w2.fail_streak == 0


@pytest.mark.asyncio
async def test_one_check_at_a_time_and_the_turn_cap():
    w = jc.TurnWatch(max_checks=1)
    for _ in range(6):
        w.note_tool_step(0, 1)
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    w.started(asyncio.ensure_future(fut))
    w.note_tool_step(0, 1)
    w.folded = True
    assert w.due() == ""          # one in flight
    assert w.collect() is None    # still running
    fut.set_result(_check(step=6))
    await asyncio.sleep(0)
    got = w.collect()
    assert got is not None and w.latest is got
    assert w.due() == ""          # cap reached


@pytest.mark.asyncio
async def test_stale_results_are_dropped_and_fresh_ones_reused():
    w = jc.TurnWatch()
    for _ in range(6):
        w.note_tool_step(0, 1)

    async def answer():
        return _check(step=6)

    w.started(asyncio.ensure_future(answer()))
    await asyncio.sleep(0)
    for _ in range(jc.STALE_STEPS + 1):
        w.note_tool_step(0, 1)
    assert w.collect() is None and w.latest is None

    w2 = jc.TurnWatch()
    w2.tool_steps = 10
    w2.started(asyncio.ensure_future(answer()))
    await asyncio.sleep(0)
    kept = w2.collect()  # 4 steps old: within STALE_STEPS
    assert kept is not None and kept.step == 6


@pytest.mark.asyncio
async def test_fresh_window():
    w = jc.TurnWatch()
    w.tool_steps = 9
    w.latest = _check(step=6)
    assert w.fresh() is w.latest
    w.tool_steps = 10
    assert w.fresh() is None


@pytest.mark.asyncio
async def test_cancel_stops_the_check_but_not_the_relation():
    w = jc.TurnWatch()
    check_task = asyncio.ensure_future(asyncio.sleep(10))
    relation_task = asyncio.ensure_future(asyncio.sleep(10))
    w.task, w.relation_task = check_task, relation_task
    w.cancel()
    await asyncio.sleep(0)
    assert check_task.cancelled()
    assert not relation_task.cancelled()
    relation_task.cancel()


def test_failures_in_counts_tool_results_only():
    msgs = [
        {"role": "tool", "content": "Error: 404", "tool_name": "web_fetch"},
        {"role": "tool", "content": "ok", "tool_name": "read"},
        {"role": "tool", "content": "Error", "tool_name": "completion_gate"},
        {"role": "assistant", "content": "Error"},
    ]
    assert jc.failures_in(msgs) == (1, 1)


# ── redo guard (task_course) ─────────────────────────────────────────


def test_redo_is_skipped_once_a_later_turn_moved_the_anchor():
    meta: dict = {}
    task_course.begin_turn(meta, "Compare the cheapest flights from Zagreb to Lisbon next month please")
    start = task_course.begin_turn(meta, "also Porto", prev=task_course.PreviousTurn())
    task_course.begin_turn(meta, "and Madrid", prev=task_course.PreviousTurn())
    anchor_before = meta[task_course.KEY]["anchor"]
    assert task_course.redo_turn(meta, start, task_course.NEW_TASK) is start
    assert meta[task_course.KEY]["anchor"] == anchor_before


def test_redo_keeps_checks_recorded_during_the_turn():
    meta: dict = {}
    task_course.begin_turn(meta, "Compare the cheapest flights from Zagreb to Lisbon next month please")
    start = task_course.begin_turn(meta, "also Porto", prev=task_course.PreviousTurn())
    task_course.record_check(meta, {"step": 6})
    task_course.redo_turn(meta, start, task_course.CONTINUES)
    assert meta[task_course.KEY]["checks"] == [{"step": 6}]
    assert meta[task_course.KEY]["amendments"] == []
