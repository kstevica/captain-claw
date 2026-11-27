"""Drive's check before delivering (agent_completion_mixin): at Persistent and
above, a turn that did work with tools has its reply compared with what was
asked, once, and is sent back when something asked for is missing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from captain_claw import drive
from captain_claw.agent_completion_mixin import AgentCompletionMixin


def _tool(name: str, content: str = "ok", **args) -> dict:
    return {"role": "tool", "tool_name": name, "tool_arguments": args, "content": content}


class _Stub(AgentCompletionMixin):
    _scale_progress = None

    def __init__(self, messages: list[dict], verdict: object):
        self.session = SimpleNamespace(messages=messages)
        self.verdict = verdict                      # what the checker "answers", or an exception
        self.checker_calls: list[list] = []
        self.checker_labels: list[tuple] = []
        self.emitted: list[dict] = []
        self.persisted: list[str] = []
        self.instructions = SimpleNamespace(
            load=lambda name: f"<{name}>",
            render=lambda name, **kw: json.dumps(kw),
        )

    async def _complete_with_guards(self, *, messages, tools, interaction_label, turn_usage, max_tokens):
        assert tools is None and interaction_label in ("delivery_check", "delivery_check_quiet")
        self.checker_calls.append(messages)
        self.checker_labels.append((interaction_label, max_tokens))
        verdict = self.verdict.pop(0) if isinstance(self.verdict, list) else self.verdict
        if isinstance(verdict, Exception):
            raise verdict
        content, finish = verdict if isinstance(verdict, tuple) else (verdict, "stop")
        return SimpleNamespace(content=content, finish_reason=finish)

    @staticmethod
    def _extract_json_object(raw: str):
        try:
            value = json.loads(raw)
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _is_monitor_only_tool_name(name: str) -> bool:
        return name == "task_status"

    async def _maybe_auto_write_requested_output(self, *, output_text, **kw):
        return output_text

    def _collect_turn_tool_output(self, _idx):
        return ""

    def _active_task_tool_policy_payload(self, _p):
        return None

    def _turn_has_unexecuted_script(self, _idx):
        return False, ""

    def _emit_tool_output(self, _tool_name, payload, _text):
        self.emitted.append(payload)

    def _update_clarification_state(self, **kw):
        pass

    async def _persist_assistant_response(self, text):
        self.persisted.append(text)

    async def _auto_capture_todos(self, *a):
        pass

    _auto_capture_contacts = _auto_capture_scripts = _auto_capture_apis = _auto_capture_todos

    def _maybe_append_playbook_hint(self, text, **kw):
        return text

    def __getattr__(self, name):
        if name.startswith(("_turn_", "_verify_", "_scale_", "_maybe_")):
            return lambda *a, **kw: False
        raise AttributeError(name)


async def _finalize(stub: _Stub, reply: str = "Here is the report on A and B.", *, iteration: int = 3,
                    hard: int = 80):
    return await stub._attempt_finalize_response(
        output_text=reply, iteration=iteration, hard_turn_iterations=hard, finish_success=True,
        effective_user_input="Report on A, B and C.", user_input="Report on A, B and C.",
        turn_start_idx=0, turn_usage={}, session_tool_policy=None, planning_pipeline=None,
        list_task_plan={}, task_contract=None,
        completion_requirements=[], completion_feedback="",
        enforce_python_worker_mode=False, python_worker_attempted=False,
    )


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    cfg = SimpleNamespace(drive=SimpleNamespace(default=""), cognitive_mode=SimpleNamespace(enabled=False),
                          model=SimpleNamespace(max_tokens=4000))
    monkeypatch.setattr("captain_claw.config.get_config", lambda: cfg)
    monkeypatch.setattr(drive, "SETTING_PATH", tmp_path / "drive.txt")
    monkeypatch.setattr("captain_claw.exam_turn.active", lambda agent: False)
    monkeypatch.setattr("captain_claw.member_privacy.private_turn", lambda agent: False)
    monkeypatch.setattr("captain_claw.member_privacy.output_level", lambda agent: "")


WORK = [_tool("web_search", "A: found. B: found.", query="A B C"), _tool("task_status", "watching")]
GAPS = json.dumps({"complete": False, "gaps": ["Nothing on C, which the request asked for.", "  "]})


def test_only_persistent_and_ambitious_check():
    assert [lvl.name for lvl in drive.list_levels() if lvl.delivery_check] == ["persistent", "ambitious"]


async def test_a_gap_sends_the_reply_back_once_and_the_second_reply_is_delivered():
    stub = _Stub(list(WORK), GAPS)
    with drive.turn_drive("persistent"):
        finalized, text, _ok, feedback, _pw = await _finalize(stub)
        assert (finalized, text) == (False, "")
        assert "Nothing on C" in feedback and "Do not ask the person a question" in feedback
        assert stub.persisted == [] and stub.emitted[-1] == {"step": "delivery_check", "passed": False, "gaps": 1}
        # The checker saw the request, the work (not the tool that only watches) and the reply.
        seen = json.loads(stub.checker_calls[0][1].content)
        assert seen["user_input"] == "Report on A, B and C." and "report on A and B" in seen["candidate_response"]
        assert "web_search(" in seen["work_log"] and "task_status" not in seen["work_log"]
        # Sent back: the next reply is delivered without another check.
        finalized, text, *_ = await _finalize(stub, "A, B, and C could not be found.", iteration=5)
    assert finalized is True and text == "A, B, and C could not be found."
    assert len(stub.checker_calls) == 1 and stub.persisted == [text]


async def test_a_complete_reply_is_delivered_after_one_check():
    stub = _Stub(list(WORK), json.dumps({"complete": True, "gaps": []}))
    with drive.turn_drive("ambitious"):
        finalized, text, *_ = await _finalize(stub)
    assert finalized is True and len(stub.checker_calls) == 1
    assert stub.emitted[-1] == {"step": "delivery_check", "passed": True, "gaps": 0}


@pytest.mark.parametrize("level", ["relaxed", "diligent", None])
async def test_lower_levels_never_check(level):
    stub = _Stub(list(WORK), GAPS)
    with drive.turn_drive(level):
        finalized, *_ = await _finalize(stub)
    assert finalized is True and stub.checker_calls == []


async def test_a_reply_made_of_words_alone_is_not_checked():
    for messages in ([], [_tool("task_status", "watching")], [{"role": "assistant", "content": "hi"}]):
        stub = _Stub(messages, GAPS)
        with drive.turn_drive("persistent"):
            finalized, *_ = await _finalize(stub, "You're welcome.")
        assert finalized is True and stub.checker_calls == []


async def test_no_check_on_the_last_step():
    stub = _Stub(list(WORK), GAPS)
    with drive.turn_drive("persistent"):
        finalized, *_ = await _finalize(stub, iteration=79, hard=80)
    assert finalized is True and stub.checker_calls == []


async def test_no_second_opinion_when_a_contract_already_judged_the_reply():
    stub = _Stub(list(WORK), GAPS)

    async def critic(**kw):                       # the contract's own critic passes the reply
        return {"complete": True, "checks": []}

    stub._evaluate_contract_completion = critic
    with drive.turn_drive("persistent"):
        finalized, *_ = await stub._attempt_finalize_response(
            output_text="Report.", iteration=3, hard_turn_iterations=80, finish_success=True,
            effective_user_input="x", user_input="x", turn_start_idx=0, turn_usage={},
            session_tool_policy=None, planning_pipeline=None, list_task_plan={},
            task_contract={"requirements": [{"id": "r1"}]}, completion_requirements=[{"id": "r1"}],
            completion_feedback="", enforce_python_worker_mode=False, python_worker_attempted=False,
        )
    assert finalized is True and stub.checker_calls == []


@pytest.mark.parametrize("verdict", [
    RuntimeError("model down"), "not json", json.dumps({"complete": True, "gaps": ["ignored"]}),
    json.dumps({"complete": False}), json.dumps({"complete": False, "gaps": "C"}), json.dumps({"gaps": ["C"]}),
])
async def test_a_checker_that_fails_or_answers_badly_never_blocks(verdict):
    stub = _Stub(list(WORK), verdict)
    with drive.turn_drive("persistent"):
        finalized, *_ = await _finalize(stub)
    assert finalized is True and len(stub.checker_calls) == 1


def test_the_work_log_is_bounded():
    from captain_claw import agent_completion_mixin as m

    stub = _Stub([_tool("read", "x" * 5000, path="f" * 900) for _ in range(m._DELIVERY_LOG_CALLS + 7)], "")
    log = stub._delivery_work_log(0)
    lines = log.splitlines()
    assert lines[0] == "(7 earlier tool call(s) not shown)" and len(lines) == m._DELIVERY_LOG_CALLS + 1
    assert max(len(line) for line in lines) < m._DELIVERY_LOG_ARGS + m._DELIVERY_LOG_RESULT + 40
    assert _Stub([], "")._delivery_work_log(0) == "(no tool calls)"


def test_the_prompts_exist_and_take_their_fields():
    from pathlib import Path

    from captain_claw.instructions import InstructionLoader

    loader = InstructionLoader(base_dir=Path(drive.__file__).resolve().parent / "instructions", use_micro=False)
    system = loader.load("delivery_check_system_prompt.md")
    assert '"complete"' in system
    # A deliberate, explained refusal is not a gap: the check must never push toward it.
    assert "declined on purpose" in system and "never ask for that part to be done" in system
    text = loader.render("delivery_check_user_prompt.md", user_input="ASK {x}", work_log="LOG", candidate_response="REPLY")
    assert "ASK {x}" in text and "LOG" in text and "REPLY" in text


async def test_a_check_that_runs_out_thinking_is_asked_again_with_thinking_off():
    stub = _Stub(list(WORK), [("Potential gap: nothing on C, which", "length"), (GAPS, "stop")])
    with drive.turn_drive("persistent"):
        finalized, _t, _ok, feedback, _pw = await _finalize(stub)
    assert finalized is False and "Nothing on C" in feedback
    # First with room to think (capped by the model's max_tokens), then thinking off.
    assert stub.checker_labels == [("delivery_check", 4000), ("delivery_check_quiet", 1500)]


async def test_a_check_cut_off_twice_still_never_blocks():
    stub = _Stub(list(WORK), [("Potential gap", "length"), ("Still thinking", "length")])
    with drive.turn_drive("persistent"):
        finalized, *_ = await _finalize(stub)
    assert finalized is True and len(stub.checker_calls) == 2


def test_the_quiet_retry_runs_with_thinking_off():
    from captain_claw.agent_guard_mixin import _REASONING_EXEMPT_LABELS

    assert "delivery_check_quiet" in _REASONING_EXEMPT_LABELS
    assert "delivery_check" not in _REASONING_EXEMPT_LABELS
