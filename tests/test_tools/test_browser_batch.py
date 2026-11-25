"""The browser tool's batch (browser_batch.py): several steps in one call.

Pinned here:

* the steps are checked before any runs — a list of browser calls, at most
  10, known actions only, at most 3 pictures; none that waits for the person
  or runs steps of its own; none of a kind the person may approve first (a
  click on "Pay now", storing a password); internal ``_`` arguments a model
  slips into a step are dropped;
* each step goes through the agent's own way of calling a tool (where the
  guard, approval cards, the privacy gate and the register are), in order,
  with the batch's target when the step names none;
* the first failure stops it: the error holds what the steps before said and
  which steps didn't run; a step that moved the work to the person's Chrome
  ends it too;
* without an agent it doesn't run; the time it gets is its steps' time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from captain_claw.tools import browser as browser_mod
from captain_claw.tools import browser_batch
from captain_claw.tools import browser_chrome as bc
from captain_claw.tools.browser import BrowserTool
from captain_claw.tools.registry import ToolResult

ACTIONS = frozenset(BrowserTool.parameters["properties"]["action"]["enum"])


class FakeAgent:
    """Calls the tool the way Agent._execute_tool_with_guard ends up doing, and keeps a list."""

    def __init__(self, tool: BrowserTool, base: Path, answers: dict[str, ToolResult] | None = None) -> None:
        self.tool = tool
        self.base = base
        self.answers = answers or {}
        self.calls: list[dict[str, Any]] = []

    async def _execute_tool_with_guard(self, name: str, arguments: dict[str, Any], interaction_label: str,
                                       session_id_override: str | None = None, **_: Any) -> ToolResult:
        assert name == "browser" and interaction_label == "tool_call:browser"
        self.calls.append(dict(arguments))
        action = arguments["action"]
        if action in self.answers:
            return self.answers[action]
        rest = {k: v for k, v in arguments.items() if k != "action"}
        return await self.tool.execute(action, **rest, _agent=self, _session_id=session_id_override or "s1",
                                       _runtime_base_path=str(self.base))


@pytest.fixture
def deck(monkeypatch):
    calls: list[tuple[str, dict, float]] = []

    async def _call(action, args, wait):
        calls.append((action, args, wait))
        return {"ok": True, "content": f"did {action}"}

    monkeypatch.setattr(bc, "call_deck", _call)
    monkeypatch.setattr(browser_mod, "_TARGETS", {})
    return calls


def test_the_steps_are_checked_before_any_runs():
    ok, why = browser_batch.check('[{"action": "navigate", "url": "https://x.com", "_session_id": "evil", '
                                  '"steps": [1]}, {"action": "PAGE_TEXT"}]', ACTIONS)
    assert not why and ok == [{"action": "navigate", "url": "https://x.com"}, {"action": "page_text"}]
    for raw, says in (
        (None, "needs steps"),
        ([], "needs steps"),
        ("not json", "needs steps"),
        ([{"action": "wait"}] * 11, "at most 10"),
        (["navigate"], "isn't a browser call"),
        ([{"url": "https://x.com"}], "has no action"),
        ([{"action": "fly"}], "isn't a browser action"),
        ([{"action": "batch", "steps": []}], "can't be part of a batch"),
        ([{"action": "extension_install", "extension_id": "captain"}], "can't be part of a batch"),
        ([{"action": "login", "app_name": "jira"}], "can't be part of a batch"),
        ([{"action": "navigate", "url": "https://shop"}, {"action": "click", "text": "Pay now"}], "approve first"),
        ([{"action": "credentials_store", "app_name": "x"}], "approve first"),
        ([{"action": "screenshot"}, {"action": "zoom"}, {"action": "observe"}, {"action": "screenshot"}],
         "at most 3 pictures"),
    ):
        steps, why = browser_batch.check(raw, ACTIONS)
        assert steps == [] and says in why, (raw, why)
    # A plain click on an ordinary button is fine.
    assert browser_batch.check([{"action": "click", "text": "Next"}], ACTIONS)[1] == ""


async def test_each_step_goes_through_the_agent_in_order(deck, tmp_path):
    tool = BrowserTool()
    agent = FakeAgent(tool, tmp_path)
    r = await tool.execute("batch", target="chrome", steps=[
        {"action": "navigate", "url": "https://example.com"},
        {"action": "form_input", "ref": "ref_3", "value": "Ana"},
        {"action": "page_text"},
    ], _agent=agent, _session_id="s1", _runtime_base_path=str(tmp_path))
    assert r.success, r.error
    assert [c["action"] for c in agent.calls] == ["navigate", "form_input", "page_text"]
    assert all(c["target"] == "chrome" for c in agent.calls), "the batch's target goes to steps that name none"
    assert [c[0] for c in deck] == ["navigate", "form_input", "page_text"]
    assert "Step 1/3 (navigate):\ndid navigate" in r.content and "Step 3/3 (page_text):" in r.content
    assert bc.WEB_CONTENT_NOTE in r.content


async def test_the_first_failure_stops_it(deck, tmp_path):
    tool = BrowserTool()
    agent = FakeAgent(tool, tmp_path, answers={"click": ToolResult(success=False, error="ref_9 isn't on the page")})
    r = await tool.execute("batch", target="chrome", steps=[
        {"action": "navigate", "url": "https://example.com"},
        {"action": "click", "ref": "ref_9"},
        {"action": "page_text"},
        {"action": "screenshot"},
    ], _agent=agent, _session_id="s1", _runtime_base_path=str(tmp_path))
    assert not r.success
    assert [c["action"] for c in agent.calls] == ["navigate", "click"], "nothing after the failure runs"
    assert "Step 1/4 (navigate):\ndid navigate" in r.error
    assert "Step 2/4 (click) failed: ref_9 isn't on the page Steps 3–4 weren't run." in r.error
    one_left = await tool.execute("batch", target="chrome", steps=[
        {"action": "click", "ref": "ref_9"}, {"action": "page_text"},
    ], _agent=agent, _session_id="s1", _runtime_base_path=str(tmp_path))
    assert one_left.error.endswith("Step 2 wasn't run.")


async def test_a_step_a_guard_refuses_is_a_failure(tmp_path):
    class Refusing(FakeAgent):
        async def _execute_tool_with_guard(self, name, arguments, interaction_label, **kw):
            raise RuntimeError("Guard blocked script_tool: not on this deck")

    tool = BrowserTool()
    r = await tool.execute("batch", steps=[{"action": "navigate", "url": "https://x.com"}],
                           _agent=Refusing(tool, tmp_path), _session_id="s1")
    assert not r.success and "Step 1/1 (navigate) failed: Guard blocked" in r.error


async def test_moving_to_the_persons_chrome_ends_it(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_mod, "_TARGETS", {})

    class Walled(FakeAgent):
        async def _execute_tool_with_guard(self, name, arguments, interaction_label, **kw):
            self.calls.append(dict(arguments))
            if arguments["action"] == "navigate":
                browser_mod._TARGETS["s1"] = "chrome"   # what a wall does (BrowserTool._to_their_chrome)
                return ToolResult(success=True, content="x.com blocks your headless browser, so it opened in Chrome")
            return ToolResult(success=True, content="read it")

    tool = BrowserTool()
    agent = Walled(tool, tmp_path)
    r = await tool.execute("batch", steps=[
        {"action": "navigate", "url": "https://x.com"}, {"action": "read_page"}, {"action": "click", "ref": "ref_1"},
    ], _agent=agent, _session_id="s1")
    assert r.success and [c["action"] for c in agent.calls] == ["navigate"]
    assert "moved to the person's Chrome at step 1" in r.content and "Steps 2–3 weren't run." in r.content
    # Asked for Chrome on purpose: that's no move, the batch goes on.
    browser_mod._TARGETS.clear()
    agent.calls.clear()
    r = await tool.execute("batch", steps=[
        {"action": "navigate", "url": "https://x.com", "target": "chrome"}, {"action": "read_page"},
    ], _agent=agent, _session_id="s1")
    assert r.success and [c["action"] for c in agent.calls] == ["navigate", "read_page"]


async def test_without_an_agent_or_with_a_secret_on_it_it_doesnt_run(tmp_path):
    tool = BrowserTool()
    r = await tool.execute("batch", steps=[{"action": "page_text"}], _session_id="s1")
    assert not r.success and "agent's turn" in r.error
    r = await tool.execute("batch", secret_ref="sec_1", steps=[{"action": "type", "ref": "ref_2"}],
                           _agent=FakeAgent(tool, tmp_path), _session_id="s1")
    assert not r.success and "step that types it" in r.error


def test_the_time_it_gets_is_its_steps_time():
    tool = BrowserTool()
    assert tool.timeout_for({"action": "batch", "steps": [{"action": "click"}, {"action": "page_text"}]}) == 135.0
    assert tool.timeout_for({"action": "batch", "steps": [{"action": "click"}] * 10}) == 600.0
    assert tool.timeout_for({"action": "batch", "steps": [{"action": "click"}] * 50}) == 600.0
    assert tool.timeout_for({"action": "batch", "steps": "nonsense"}) == 60.0
    assert tool.timeout_for({"action": "read", "urls": ["a", "b", "c"]}) == 115.0
    assert tool.timeout_for({"action": "read", "url": "a"}) == 75.0
