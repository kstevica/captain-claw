"""The caveat check (answer_caveats + agent_completion_mixin): an answer whose
reply or written report qualifies its own result goes back once to settle on
one reading; an answer that doesn't is delivered as it was."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from captain_claw import answer_caveats as ac
from captain_claw import drive

# The delivery check's stub agent and finalize helper (tests/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "_delivery_check_tests", Path(__file__).with_name("test_drive_delivery_check.py"))
_dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dc)
_quiet = _dc._quiet          # the same config/drive isolation (autouse fixture)

PASSES = json.dumps({"complete": True, "gaps": []})


def _call(name: str, result: str, **args) -> dict:
    """A tool message: `result` is what the tool returned, `args` what it was called with."""
    return {"role": "tool", "tool_name": name, "tool_arguments": args, "content": result}


# ── Finding them ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Gap to the threshold: 1,000 − 820 = 180. (Read strictly as 'more than 1,000', it would be 181.)",
    "All 7 conditions met (at the product level).",
    "If both on one unit is a hard requirement, no model meets every condition.",
    "Under a stricter reading of the rule, the answer would be different.",
    "(If the stricter \"more than 1,000\" reading is used, the gap is 181.)",
    "Technically the 1 TB model counts as having 512 GB.",
    "若把两项放在同一台机器上当硬性要求，则没有机型满足全部条件。",
    "7/7 全部满足（在机型层面）。",
    "严格来说，差距是 181。",
    "Strogo gledano, razlika je 181.",
    "Ako je to obavezan uvjet, nijedan model ne odgovara.",
    "Streng genommen fehlen 181.",
])
def test_a_sentence_that_qualifies_the_result_is_found(text):
    assert ac.find_caveats([text])


@pytest.mark.parametrize("text", [
    "Final answer: 181. The rule says more than 1,000, so 1,001 is the first qualifying count.",
    "No model meets all seven conditions; the closest two are listed below.",
    "The deal depends on whether the audit passes. 合同金额取决于审计结果。",
    "A report built on that claim would be misleading, so this file instead holds the checked record.",
    "```python\nif strict_reading: technically = True\n```",
    "| Model | Battery |\n|---|---|\n| A | 5,000 mAh |",
    "",
])
def test_plain_answers_are_left_alone(text):
    assert ac.find_caveats([text]) == []


def test_each_sentence_is_quoted_once_and_the_list_is_bounded():
    one = "Technically it passes."
    assert ac.find_caveats([one, one]) == [one]
    many = " ".join(f"Technically item {i} passes." for i in range(9))
    found = ac.find_caveats([many])
    assert len(found) == ac.MAX_QUOTED
    assert len(ac.find_caveats(["Technically " + "x" * 900])[0]) == ac.MAX_SENTENCE_CHARS


def test_the_feedback_quotes_them_and_asks_for_one_reading():
    text = ac.feedback(["Technically it passes."])
    assert '- "Technically it passes."' in text
    assert "every file you wrote" in text and "Do not ask the person a question" in text


# ── In the completion gate ───────────────────────────────────────────

REPORT_HEDGED = "Target: CEC\n\nFinal answer: 180\n\n(Read strictly as 'more than 1,000', it would be 181.)"


async def test_a_hedged_report_goes_back_once_then_is_delivered():
    messages = [_call("web_search", "found"),
                _call("write", "Written 80 chars", path="/w/results/results.md", content=REPORT_HEDGED)]
    stub = _dc._Stub(messages, PASSES)
    with drive.turn_drive("ambitious"):
        finalized, text, _ok, feedback, _pw = await _dc._finalize(stub, "Saved to results.md.")
        assert (finalized, text) == (False, "")
        assert "Read strictly as 'more than 1,000'" in feedback
        assert stub.emitted[-1] == {"step": "caveat_check", "passed": False, "caveats": 1}
        assert stub.checker_calls == []          # the model check comes after, on the next reply
        # Sent back once: the next reply goes on to the delivery check and is delivered.
        finalized, text, *_ = await _dc._finalize(stub, "Final answer: 181.", iteration=5)
    assert finalized is True and text == "Final answer: 181."
    assert len(stub.checker_calls) == 1


async def test_a_hedge_in_the_reply_itself_is_found():
    stub = _dc._Stub([_call("web_search", "found")], PASSES)
    with drive.turn_drive("persistent"):
        finalized, _t, _ok, feedback, _pw = await _dc._finalize(
            stub, "All 7 conditions met (at the product level).")
    assert finalized is False and "at the product level" in feedback


@pytest.mark.parametrize("message", [
    _call("write", "Error: foreign_overwrite_refused", path="/w/r.md", content=REPORT_HEDGED),
    _call("write", "Written 80 chars", path="/w/check.py", content="# technically strict reading\n"),
    _call("read", REPORT_HEDGED, path="/w/source.md"),
])
async def test_refused_writes_code_and_what_was_only_read_do_not_count(message):
    stub = _dc._Stub([_call("web_search", "found"), message], PASSES)
    with drive.turn_drive("ambitious"):
        finalized, *_ = await _dc._finalize(stub, "Final answer: 181.")
    assert finalized is True


@pytest.mark.parametrize("level", ["relaxed", "diligent", None])
async def test_lower_levels_never_check(level):
    stub = _dc._Stub([_call("web_search", "found")], PASSES)
    with drive.turn_drive(level):
        finalized, *_ = await _dc._finalize(stub, "Technically it passes.")
    assert finalized is True


async def test_words_alone_and_the_last_step_are_not_checked():
    stub = _dc._Stub([], PASSES)
    with drive.turn_drive("ambitious"):
        assert (await _dc._finalize(stub, "Technically it passes."))[0] is True
    stub = _dc._Stub([_call("web_search", "found")], PASSES)
    with drive.turn_drive("ambitious"):
        assert (await _dc._finalize(stub, "Technically it passes.", iteration=79, hard=80))[0] is True


async def test_an_edit_counts_by_its_new_text():
    edit = _call("edit", "Edited /w/r.md", path="/w/r.md", old_string="180", new_string="180 (technically 181)")
    stub = _dc._Stub([_call("web_search", "found"), edit], PASSES)
    with drive.turn_drive("ambitious"):
        finalized, *_ = await _dc._finalize(stub, "Done.")
    assert finalized is False
