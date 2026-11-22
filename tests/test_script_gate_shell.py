"""The "run your script" completion gate asks only an agent that can run it.

cohort001 (2026-11-21): a non-admin's agent on a multi-user deck (no shell:
deck_confine reserves it for admins' agents) wrote ``css_audit.py``, said it
had no shell, and ``script_execution_enforcement`` held its finished answer for
60+ steps, each time saying "You MUST run the script via shell".
"""

from __future__ import annotations

import types

import pytest

from captain_claw.agent_completion_mixin import AgentCompletionMixin
from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin

SCRIPT = "saved/scripts/s1/css_audit.py"


class _Tools:
    def __init__(self, names=("shell", "write", "read")):
        self.names = set(names)

    def has_tool(self, name):
        return name in self.names


def _session(*tool_rows):
    msgs = [{"role": "user", "content": "check the css"}]
    for name, args, content in tool_rows:
        msgs.append({"role": "tool", "tool_name": name, "tool_arguments": args, "content": content})
    return types.SimpleNamespace(messages=msgs, id="s1")


class _Stub(AgentCompletionMixin, AgentToolLoopMixin):
    _scale_progress = None

    def __init__(self, session, tools=None):
        self.session = session
        self.tools = tools or _Tools()
        self._empty_answer_retries = 0
        self.emitted = []

    async def _maybe_auto_write_requested_output(self, **kw):
        return kw.get("output_text", "")

    def _collect_turn_tool_output(self, _idx):
        return ""

    def _active_task_tool_policy_payload(self, _p):
        return None

    def _emit_tool_output(self, name, args, output):
        self.emitted.append(args.get("step"))

    def _maybe_append_playbook_hint(self, final_response, *a, **kw):
        return final_response

    async def _persist_assistant_response(self, *a, **kw):
        return None

    def __getattr__(self, name):
        if name.startswith(("_verify_", "_scale_")):
            return lambda *a, **kw: False
        if name.startswith("_auto_capture_"):
            async def _nothing(*a, **kw):
                return None
            return _nothing
        if name.startswith(("_update_", "_emit_", "_set_", "_refresh_")):
            return lambda *a, **kw: None
        raise AttributeError(name)


@pytest.fixture(autouse=True)
def _no_confine(monkeypatch):
    monkeypatch.delenv("CLAW_CONFINE", raising=False)


def _written():
    return _session(("write", {"path": SCRIPT, "content": "print(1)"}, "Wrote 8 bytes"))


class TestShellUsable:
    def test_usable_by_default(self):
        assert _Stub(_written())._shell_usable(0) is True

    def test_not_for_a_non_admins_agent_on_a_multi_user_deck(self, monkeypatch):
        monkeypatch.setenv("CLAW_CONFINE", "user")
        assert _Stub(_written())._shell_usable(0) is False

    def test_an_admins_agent_keeps_it(self, monkeypatch):
        monkeypatch.setenv("CLAW_CONFINE", "admin")
        assert _Stub(_written())._shell_usable(0) is True

    def test_not_without_the_tool(self):
        assert _Stub(_written(), _Tools(("write", "read")))._shell_usable(0) is False

    def test_not_after_a_refused_shell_call_this_turn(self):
        session = _session(("write", {"path": SCRIPT}, "Wrote"),
                           ("shell", {"command": "python3 x.py"}, "Error: Tool 'shell' blocked: Blocked by tool policy chain"))
        assert _Stub(session)._shell_usable(0) is False
        # An earlier turn's refusal doesn't count.
        assert _Stub(session)._shell_usable(3) is True


class TestOncePerScript:
    def test_asks_once_then_lets_the_answer_go(self):
        stub = _Stub(_written())
        assert stub._script_gate_may_ask(0, SCRIPT) is True
        assert stub._script_gate_may_ask(0, SCRIPT) is False
        assert stub._script_gate_may_ask(0, "saved/scripts/s1/other.py") is True

    def test_a_new_turn_may_ask_again(self):
        stub = _Stub(_written())
        assert stub._script_gate_may_ask(0, SCRIPT) is True
        assert stub._script_gate_may_ask(7, SCRIPT) is True


async def _finalize(stub, text="Here is the new CSS."):
    return await stub._attempt_finalize_response(
        output_text=text, iteration=4, hard_turn_iterations=161, finish_success=True,
        effective_user_input="check the css", user_input="check the css", turn_start_idx=0,
        turn_usage={}, session_tool_policy=None, planning_pipeline=None, list_task_plan={},
        task_contract=None, completion_requirements=[], completion_feedback="",
        enforce_python_worker_mode=False, python_worker_attempted=False,
    )


class TestFinalize:
    async def test_no_shell_the_answer_goes(self, monkeypatch):
        monkeypatch.setenv("CLAW_CONFINE", "user")
        stub = _Stub(_written())
        finalized, text, _ok, feedback, _pw = await _finalize(stub)
        assert finalized is True and text == "Here is the new CSS."
        assert "script_execution_enforcement" not in stub.emitted

    async def test_with_a_shell_it_asks_once(self):
        stub = _Stub(_written())
        finalized, _text, _ok, feedback, _pw = await _finalize(stub)
        assert finalized is False and "MUST run the script via shell" in feedback
        finalized, text, _ok, _feedback, _pw = await _finalize(stub)
        assert finalized is True and text == "Here is the new CSS."
