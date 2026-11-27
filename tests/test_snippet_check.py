"""The snippet check (agent_completion_mixin): at Drive Persistent and
Ambitious, an answer from a turn that searched the web but opened no page goes
back once to open the pages its facts come from."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from captain_claw import drive

# The delivery check's stub agent and finalize helper (tests/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "_delivery_check_tests_snippet", Path(__file__).with_name("test_drive_delivery_check.py"))
_dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dc)
_quiet = _dc._quiet          # the same config/drive isolation (autouse fixture)

PASSES = json.dumps({"complete": True, "gaps": []})
SEARCH = {"role": "tool", "tool_name": "web_search", "tool_arguments": {"query": "threshold"},
          "content": "[SEARCH ENGINE: Brave] 1. Report — a threshold of 1,000 or more …"}
FETCH = {"role": "tool", "tool_name": "web_fetch", "tool_arguments": {"url": "https://example.org/law"},
         "content": "Section 2: more than 1,000 …"}


def _stub(messages, tools=("web_search", "web_fetch")):
    stub = _dc._Stub(list(messages), PASSES)
    stub.tools = SimpleNamespace(has_tool=lambda name: name in tools)
    return stub


async def test_an_answer_from_unopened_search_results_goes_back_once():
    stub = _stub([SEARCH])
    with drive.turn_drive("ambitious"):
        finalized, text, _ok, feedback, _pw = await _dc._finalize(stub, "Final answer: 180.")
        assert (finalized, text) == (False, "")
        assert "opened no page" in feedback and "read the original they cite" in feedback
        assert stub.emitted[-1] == {"step": "snippet_check", "passed": False}
        # Once per turn: the next answer goes on to the other checks and is delivered.
        finalized, text, *_ = await _dc._finalize(stub, "Final answer: 181.", iteration=5)
    assert finalized is True and text == "Final answer: 181."


@pytest.mark.parametrize("messages", [
    [SEARCH, FETCH],                                   # it opened a page
    [SEARCH, {"role": "tool", "tool_name": "shell", "content": "saved",
              "tool_arguments": {"command": "curl -sL -o paper.pdf https://example.org/paper.pdf"}}],
    [SEARCH, {"role": "tool", "tool_name": "pdf_extract", "content": "# paper.pdf", "tool_arguments": {}}],
    [dict(SEARCH, content="Error: search failed")],    # the search gave nothing
    [FETCH],                                           # no search at all
    [],
])
async def test_no_check_when_a_page_was_opened_or_nothing_was_found(messages):
    with drive.turn_drive("ambitious"):
        finalized, *_ = await _dc._finalize(_stub(messages), "Final answer: 181.")
    assert finalized is True


async def test_no_check_without_a_tool_that_opens_pages():
    with drive.turn_drive("ambitious"):
        finalized, *_ = await _dc._finalize(_stub([SEARCH], tools=("web_search",)), "Answer.")
    assert finalized is True


@pytest.mark.parametrize("level", ["relaxed", "diligent", None])
async def test_lower_levels_never_check(level):
    with drive.turn_drive(level):
        finalized, *_ = await _dc._finalize(_stub([SEARCH]), "Answer.")
    assert finalized is True


async def test_no_check_on_the_last_step():
    with drive.turn_drive("persistent"):
        finalized, *_ = await _dc._finalize(_stub([SEARCH]), "Answer.", iteration=79, hard=80)
    assert finalized is True
