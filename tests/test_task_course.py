"""The task anchor (captain_claw/task_course.py): rules, lifetime, the note."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from captain_claw import task_course as tc

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
LONG_TASK = (
    "Research the three cheapest flights from Zagreb to Lisbon next month, compare "
    "baggage rules and write a short table with prices and departure times."
)


def _open(metadata: dict, text: str = LONG_TASK) -> dict:
    tc.begin_turn(metadata, text, now=NOW)
    return metadata[tc.KEY]


def _prev(minutes: float = 1, answered: bool = True, asked: bool = False) -> tc.PreviousTurn:
    return tc.PreviousTurn(end_at=NOW - timedelta(minutes=minutes), answered=answered, asked=asked)


# ── rules ────────────────────────────────────────────────────────────


def test_first_message_opens_an_anchor_in_the_persons_words():
    meta: dict = {}
    start = tc.begin_turn(meta, f"  {LONG_TASK}\n\n ", now=NOW)
    assert start.relation == tc.NEW_TASK
    course = meta[tc.KEY]
    assert course["anchor"] == LONG_TASK
    assert course["status"] == "open"
    assert course["id"].startswith("tc_")


@pytest.mark.parametrize("word", ["continue", "Continue.", "go on please", "nastavi", "Dalje!", "weiter", "继续"])
def test_continue_words_keep_the_anchor_unchanged(word):
    meta: dict = {}
    _open(meta)
    start = tc.begin_turn(meta, word, prev=_prev(), now=NOW)
    assert start.relation == tc.CONTINUES
    assert meta[tc.KEY]["anchor"] == LONG_TASK
    assert meta[tc.KEY]["latest"] == ""
    assert meta[tc.KEY]["turns"] == 2


@pytest.mark.parametrize("word", ["thanks!", "ok", "hvala", "👍"])
def test_thanks_is_not_a_task(word):
    meta: dict = {}
    _open(meta)
    before = dict(meta[tc.KEY])
    start = tc.begin_turn(meta, word, prev=_prev(), now=NOW)
    assert start.relation == tc.NOT_A_TASK
    assert meta[tc.KEY] == before


def test_short_follow_up_adds_to_the_task():
    meta: dict = {}
    _open(meta)
    start = tc.begin_turn(meta, "also include Porto", prev=_prev(), now=NOW)
    assert start.relation == tc.AMENDS
    assert meta[tc.KEY]["amendments"] == ["also include Porto"]
    assert meta[tc.KEY]["latest"] == "also include Porto"


def test_short_answer_to_a_question_continues():
    meta: dict = {}
    _open(meta)
    start = tc.begin_turn(meta, "economy, one adult", prev=_prev(asked=True), now=NOW)
    assert start.relation == tc.CONTINUES
    assert meta[tc.KEY]["amendments"] == []
    assert meta[tc.KEY]["latest"] == "economy, one adult"


def test_long_unrelated_message_starts_a_new_task_and_closes_the_old_one():
    meta: dict = {}
    _open(meta)
    other = (
        "Draft a polite reminder email to the accountant about the quarterly invoices "
        "that still have not been booked into the ledger this autumn."
    )
    start = tc.begin_turn(meta, other, prev=_prev(), now=NOW)
    assert start.relation == tc.NEW_TASK
    assert meta[tc.KEY]["anchor"] == other
    closed = meta[tc.CLOSED_KEY]
    assert len(closed) == 1 and closed[0]["status"] == "done"
    assert "Zagreb" in closed[0]["anchor"]


def test_long_related_message_amends():
    meta: dict = {}
    _open(meta)
    more = (
        "For the flights to Lisbon please also compare baggage rules for cabin bags and "
        "add the departure times in local time to the table."
    )
    assert tc.begin_turn(meta, more, prev=_prev(), now=NOW).relation == tc.AMENDS


def test_two_hour_gap_after_an_answer_starts_a_new_task():
    meta: dict = {}
    _open(meta)
    start = tc.begin_turn(meta, "and Porto?", prev=_prev(minutes=180), now=NOW)
    assert start.relation == tc.NEW_TASK


def test_topic_switch_heuristic_starts_a_new_task():
    meta: dict = {}
    _open(meta)
    start = tc.begin_turn(
        meta, "what's the weather", prev=_prev(), now=NOW, topic_switch=lambda msg, anchor: True,
    )
    assert start.relation == tc.NEW_TASK


def test_unanswered_old_task_is_dropped_not_done():
    meta: dict = {}
    _open(meta)
    other = "Write a haiku about autumn rain falling softly over the old harbour walls tonight please."
    tc.begin_turn(meta, other, prev=_prev(answered=False), now=NOW)
    assert meta[tc.CLOSED_KEY][0]["status"] == "dropped"


def test_automated_turn_is_always_a_new_task_and_a_person_moves_on_from_it():
    meta: dict = {}
    _open(meta)
    tc.begin_turn(meta, "check the inbox", automated=True, now=NOW)
    assert meta[tc.KEY]["automated"] is True
    start = tc.begin_turn(meta, "also flag invoices", prev=_prev(), now=NOW)
    assert start.relation == tc.NEW_TASK
    assert "automated" not in meta[tc.KEY]


# ── caps and odd input ───────────────────────────────────────────────


def test_long_request_is_cut_with_a_pointer():
    meta: dict = {}
    tc.begin_turn(meta, "word " * 400, now=NOW)
    anchor = meta[tc.KEY]["anchor"]
    assert len(anchor) <= tc.ANCHOR_CHARS
    assert anchor.endswith("earlier in the conversation)")


def test_amendments_keep_the_newest_three():
    meta: dict = {}
    _open(meta)
    for i in range(5):
        tc.begin_turn(meta, f"also add item {i}", prev=_prev(), now=NOW)
    assert meta[tc.KEY]["amendments"] == ["also add item 2", "also add item 3", "also add item 4"]


def test_surface_block_is_stripped(monkeypatch):
    from captain_claw import msg_origin

    monkeypatch.setattr(msg_origin, "split_surface_block", lambda text: ("RULES", "the real ask"))
    meta: dict = {}
    tc.begin_turn(meta, "RULES the real ask", now=NOW)
    assert meta[tc.KEY]["anchor"] == "the real ask"


def test_empty_message_and_odd_metadata_change_nothing():
    meta: dict = {tc.KEY: "garbage"}
    start = tc.begin_turn(meta, "   ", now=NOW)
    assert start.relation == tc.NOT_A_TASK
    assert meta[tc.KEY] == "garbage"
    assert tc.current(meta) is None


def test_closed_list_keeps_ten():
    meta: dict = {}
    for i in range(14):
        tc.begin_turn(meta, f"completely different job number {i} about topic{i} zebra{i} " * 3,
                      prev=_prev(minutes=500), now=NOW)
    assert len(meta[tc.CLOSED_KEY]) == tc.MAX_CLOSED


# ── redo (a later, better decision) ──────────────────────────────────


def test_redo_turn_replaces_a_wrong_guess():
    meta: dict = {}
    _open(meta)
    start = tc.begin_turn(meta, "also include Porto", prev=_prev(), now=NOW)
    assert start.relation == tc.AMENDS
    redone = tc.redo_turn(meta, start, tc.NEW_TASK)
    assert redone.relation == tc.NEW_TASK
    assert meta[tc.KEY]["anchor"] == "also include Porto"
    assert len(meta[tc.CLOSED_KEY]) == 1

    back = tc.redo_turn(meta, redone, tc.CONTINUES)
    assert back.relation == tc.CONTINUES
    assert meta[tc.KEY]["anchor"] == LONG_TASK
    assert tc.CLOSED_KEY not in meta or meta[tc.CLOSED_KEY] == []


def test_redo_turn_agreeing_is_a_no_op():
    meta: dict = {}
    start = tc.begin_turn(meta, LONG_TASK, now=NOW)
    snapshot = dict(meta[tc.KEY])
    assert tc.redo_turn(meta, start, tc.NEW_TASK) is start
    assert meta[tc.KEY] == snapshot


# ── requirements and checks ──────────────────────────────────────────


def test_requirements_from_the_contract_are_short_and_capped():
    meta: dict = {}
    _open(meta)
    reqs = [{"id": f"r{i}", "title": f"requirement {i} " + "x" * 300} for i in range(8)]
    tc.set_requirements(meta, reqs)
    out = meta[tc.KEY]["requirements"]
    assert len(out) == tc.MAX_REQUIREMENTS
    assert all(len(r) <= tc.REQUIREMENT_CHARS for r in out)


def test_record_check_keeps_the_newest_twenty():
    meta: dict = {}
    _open(meta)
    for i in range(25):
        tc.record_check(meta, {"step": i})
    checks = meta[tc.KEY]["checks"]
    assert len(checks) == tc.MAX_CHECKS and checks[-1]["step"] == 24


# ── the note ─────────────────────────────────────────────────────────


def _tool_turn() -> list[dict]:
    return [
        {"role": "user", "content": LONG_TASK},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "result", "tool_name": "web_search"},
    ]


def test_no_note_before_the_first_tool_call():
    meta: dict = {}
    _open(meta)
    assert tc.note_for_turn(meta, [{"role": "user", "content": LONG_TASK}]) == ""
    # A monitor echo row is not a tool call.
    assert tc.note_for_turn(meta, [{"role": "tool", "content": "x", "tool_name": "llm_trace"}]) == ""


def test_note_after_a_tool_call_shows_anchor_additions_and_latest():
    meta: dict = {}
    _open(meta)
    tc.begin_turn(meta, "also include Porto", prev=_prev(), now=NOW)
    tc.begin_turn(meta, "economy only", prev=_prev(asked=True), now=NOW)
    note = tc.note_for_turn(meta, _tool_turn())
    assert note.startswith("Task (the person's request): Research the three cheapest")
    assert "Added since: also include Porto" in note
    assert "Their latest message: economy only" in note


def test_note_carries_live_lines_and_run_goal():
    course = {
        "status": "open", "anchor": "Collect prices", "goal": "Plan the trip",
        "amendments": [], "latest": "",
    }
    note = tc.note(course, {"stage": "Stage: gathering (2 of 5)", "line": "Re-read the task."})
    assert note.splitlines() == [
        "Run goal: Plan the trip",
        "Your part: Collect prices",
        "Stage: gathering (2 of 5)",
        "Re-read the task.",
    ]


def test_anchor_off_means_no_note(monkeypatch):
    meta: dict = {}
    _open(meta)
    monkeypatch.setattr(tc, "enabled", lambda: False)
    assert tc.note_for_turn(meta, _tool_turn()) == ""


def test_previous_turn_reads_the_end_of_the_session():
    ts = (NOW - timedelta(minutes=5)).isoformat()
    prev = tc.previous_turn([
        {"role": "user", "content": "q", "timestamp": ts},
        {"role": "assistant", "content": "Which city?", "timestamp": ts},
        {"role": "tool", "content": "trace", "tool_name": "llm_trace"},
    ])
    assert prev.answered and prev.asked
    assert prev.end_at == NOW - timedelta(minutes=5)
    assert tc.previous_turn([]) == tc.PreviousTurn()
