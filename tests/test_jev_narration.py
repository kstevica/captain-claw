"""Jev's decisions as narration lines (captain_claw/jev_narration.py)."""

from __future__ import annotations

from captain_claw import jev_course
from captain_claw import jev_narration as jn


def _check(**kw) -> jev_course.Check:
    base = dict(step=6, stage=3.95, stage_name="checking", on_track=0.93, next="continue",
                next_confidence=0.61, stage_probs={"checking": 0.62, "producing": 0.3, "gathering": 0.04},
                next_probs={"continue": 0.61, "finish": 0.25, "refocus": 0.1})
    base.update(kw)
    return jev_course.Check(**base)


def test_course_check_says_where_it_stands_and_what_jev_weighed():
    assert jn.course_check(_check()) == (
        "Jev checked the course after step 6: on track 93%; stage 4 of 5, checking (also producing 30%); "
        "next: keep going (61%, wrap up 25%, refocus 10%). No reminder needed.")


def test_course_check_with_a_line_says_what_the_agent_was_told():
    text = jn.course_check(_check(on_track=0.1, next="refocus", next_confidence=0.8, next_probs={}), "refocus")
    assert text.startswith("Jev checked the course after step 6: on track 10%;")
    assert "next: refocus (80%)" in text
    assert text.endswith("Reminded the agent to get back to the task.")


def test_course_check_with_little_to_say():
    assert jn.course_check(jev_course.Check(step=0)) == "Jev checked the course. No reminder needed."


def test_relation_lines():
    probs = {"amends": 0.82, "new_task": 0.12}
    assert jn.relation("amends", 0.82, probs, "continues", True, 0.6) == (
        "Jev read your message as an addition to the task (82%; a new task 12%) and updated the task reminder.")
    assert jn.relation("continues", 0.9, {}, "continues", False, 0.6) == (
        "Jev read your message as the same task (90%), as the usual rules did.")
    assert jn.relation("new_task", 0.45, {"new_task": 0.45, "amends": 0.4}, "amends", False, 0.6) == (
        "Jev wasn't sure how your message relates to the open task (a new task 45%; an addition to the task "
        "40%), so the usual rules decide: an addition to the task.")


def test_budget_lines():
    assert jn.budget(0.9, True, 12) == "Step budget used up. Jev says the work is still moving (90%), so it gets more steps (up to 12)."
    assert jn.budget(0.2, False) == ("Step budget used up. Jev says the work has stalled (only 20% that it's "
                                     "still moving), so the turn stops here.")
    assert jn.budget(0.85, True, 30, reused=True) == (
        "Step budget used up. Jev's last course check says the work is on track (85%), so it gets more steps (up to 30).")
    assert "doesn't say keep going" in jn.budget(0.4, False, reused=True)
    assert jn.budget(None, False) == ""


def test_block_review_lines():
    assert jn.block_review("read", "verify", 0.7, applied=True, released=False) == (
        "A repeated read call was held back; Jev picked what comes next: read the file once before deciding (70%).")
    assert "let it run once" in jn.block_review("read", "repeat_ok", 0.85, applied=False, released=True)
    assert "the usual message stands" in jn.block_review("web_fetch", "finish", 0.4, applied=False, released=False)


def test_the_web_server_marks_jevs_narration_frames():
    from types import SimpleNamespace

    from captain_claw.web_server import WebServer

    sent: list[dict] = []
    fake = SimpleNamespace(_broadcast=sent.append)
    WebServer._narration_callback(fake, "Jev checked the course.", 0, source="jev")
    WebServer._narration_callback(fake, "Reading the sources.", 2)
    assert sent[0]["source"] == "jev" and sent[0]["type"] == "narration"
    assert "source" not in sent[1] and sent[1]["iteration"] == 2
