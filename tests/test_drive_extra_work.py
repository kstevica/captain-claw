"""Drive's extra-work step (Ambitious, where the agent's switch is on): once
the asked work has passed every gate, the agent may add one obvious next
step with reading and file-writing tools only; the answer stays as it was and
what was added is shown under it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from captain_claw import drive
from captain_claw.agent_completion_mixin import AgentCompletionMixin
from captain_claw.agent_guard_mixin import AgentGuardMixin

ANSWER = "Here is the report on A, B and C."


def _tool(name: str, content: str = "ok", **args) -> dict:
    return {"role": "tool", "tool_name": name, "tool_arguments": args, "content": content}


class _Stub(AgentCompletionMixin):
    _scale_progress = None
    max_iterations = 40
    _turn_origin = ("human", "", None)

    def __init__(self, messages=None):
        self.session = SimpleNamespace(messages=list(messages if messages is not None else [_tool("web_search")]))
        self._extra_work = None
        self._delivery_check_done = True            # the check is another test's subject
        self.persisted: list[str] = []
        self.emitted: list[dict] = []
        self.auto_written: list[str] = []
        self.instructions = SimpleNamespace(render=lambda name, **kw: f"<{name} {json.dumps(kw)}>")

    @staticmethod
    def _is_monitor_only_tool_name(name: str) -> bool:
        return False

    async def _maybe_auto_write_requested_output(self, *, output_text, **kw):
        self.auto_written.append(output_text)
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

    def _add_session_message(self, role, text):
        self.session.messages.append({"role": role, "content": text})

    async def _auto_capture_todos(self, *a):
        pass

    _auto_capture_contacts = _auto_capture_scripts = _auto_capture_apis = _auto_capture_todos

    def _maybe_append_playbook_hint(self, text, **kw):
        return text

    def __getattr__(self, name):
        if name.startswith(("_turn_", "_verify_", "_scale_", "_maybe_")):
            return lambda *a, **kw: False
        raise AttributeError(name)


async def _finalize(stub: _Stub, reply: str = ANSWER, *, iteration: int = 3, hard: int = 160):
    return await stub._attempt_finalize_response(
        output_text=reply, iteration=iteration, hard_turn_iterations=hard, finish_success=True,
        effective_user_input="Report on A, B and C.", user_input="Report on A, B and C.",
        turn_start_idx=0, turn_usage={}, session_tool_policy=None, planning_pipeline=None,
        list_task_plan={}, task_contract=None, completion_requirements=[], completion_feedback="",
        enforce_python_worker_mode=False, python_worker_attempted=False,
    )


@pytest.fixture()
def cfg(monkeypatch, tmp_path):
    c = SimpleNamespace(drive=SimpleNamespace(default="", initiative=True),
                        cognitive_mode=SimpleNamespace(enabled=False), model=SimpleNamespace(max_tokens=4000))
    monkeypatch.setattr("captain_claw.config.get_config", lambda: c)
    monkeypatch.setattr(drive, "SETTING_PATH", tmp_path / "drive.txt")
    monkeypatch.setattr(drive, "INITIATIVE_PATH", tmp_path / "drive_initiative.txt")
    monkeypatch.setattr("captain_claw.exam_turn.active", lambda agent: False)
    monkeypatch.setattr("captain_claw.member_privacy.private_turn", lambda agent: False)
    monkeypatch.setattr("captain_claw.member_privacy.output_level", lambda agent: "")
    return c


# ── The step itself ───────────────────────────────────────────────────


def test_only_ambitious_has_it_and_it_is_on_until_switched_off(cfg):
    assert [lvl.name for lvl in drive.list_levels() if lvl.initiative] == ["ambitious"]
    assert drive.LEVELS["ambitious"].initiative_share == 0.25
    assert drive.initiative_enabled() is True
    cfg.drive.initiative = False
    assert drive.initiative_enabled() is False
    from captain_claw.config import Config

    assert Config().drive.initiative is True


def test_the_setting_file_wins_over_the_config_and_applies_at_once(cfg):
    drive.INITIATIVE_PATH.write_text("off", encoding="utf-8")
    assert drive.initiative_enabled() is False          # config says on
    cfg.drive.initiative = False
    drive.INITIATIVE_PATH.write_text("on\n", encoding="utf-8")
    assert drive.initiative_enabled() is True
    drive.INITIATIVE_PATH.write_text("maybe", encoding="utf-8")
    assert drive.initiative_enabled() is False          # unreadable words: the config decides
    assert (drive.render_switch(True), drive.render_switch(False)) == ("on", "off")


async def test_the_answer_is_held_the_step_is_offered_and_what_it_added_goes_under_the_answer(cfg):
    stub = _Stub()
    with drive.turn_drive("ambitious"):
        finalized, text, _ok, feedback, _pw = await _finalize(stub)
        assert (finalized, text) == (False, "") and stub.persisted == []
        assert "drive_extra_work_prompt.md" in feedback and '"limit": 10' in feedback
        assert stub._extra_work == {"active": True, "reply": ANSWER, "calls": 0, "refused": 0, "limit": 10}
        stub._extra_work["calls"] = 2                       # it read a source and wrote a file
        finalized, text, ok, feedback, _pw = await _finalize(stub, "I added a sources section to report.md.", iteration=6)
    assert finalized is True and ok is True and feedback == ""
    assert text == f"{ANSWER}\n\n---\n**Also done (not asked for)**\n\nI added a sources section to report.md."
    assert stub.persisted == [text]
    # The step's own words were never auto-written to a file as if they were the answer.
    assert stub.auto_written == [ANSWER]
    assert stub.emitted[-1]["phase"] == "end" and stub.emitted[-1]["added"] is True


@pytest.mark.parametrize("extra, calls", [
    ("NOTHING TO ADD", 0), ("Nothing to add.", 3), ("", 2), ("   ", 0),
    ("You could also look at D and E next.", 0),            # words alone are not work
])
async def test_when_nothing_was_added_the_answer_goes_out_alone(cfg, extra, calls):
    stub = _Stub()
    with drive.turn_drive("ambitious"):
        await _finalize(stub)
        stub._extra_work["calls"] = calls
        finalized, text, *_ = await _finalize(stub, extra, iteration=5)
    assert finalized is True and text == ANSWER and stub.persisted == [ANSWER]


async def test_it_is_offered_once_per_turn(cfg):
    stub = _Stub()
    with drive.turn_drive("ambitious"):
        await _finalize(stub)
        stub._extra_work["calls"] = 1
        await _finalize(stub, "Added a chart.", iteration=5)
        finalized, text, *_ = await _finalize(stub, "Another reply.", iteration=7)   # a later finalize in the turn
    assert finalized is True and text == "Another reply."


@pytest.mark.parametrize("change", ["level", "switch", "words_only", "automation", "member", "no_room"])
async def test_when_it_is_not_offered(cfg, change):
    stub = _Stub()
    level, iteration, hard = "ambitious", 3, 160
    if change == "level":
        level = "persistent"
    elif change == "switch":
        cfg.drive.initiative = False
    elif change == "words_only":
        stub = _Stub(messages=[])
    elif change == "automation":
        stub._turn_origin = ("automation", "cron", None)     # a scheduled run: nobody would see it
    elif change == "member":
        stub._speaker_scoped = True                           # a shared agent's member: not their workspace
    elif change == "no_room":
        iteration, hard = 150, 160
    with drive.turn_drive(level):
        finalized, text, *_ = await _finalize(stub, iteration=iteration, hard=hard)
    assert finalized is True and text == ANSWER and stub._extra_work is None


# ── What may run during it ────────────────────────────────────────────


@pytest.mark.parametrize("tool, args, allowed", [
    ("read", {}, True), ("grep", {}, True), ("web_search", {}, True), ("web_fetch", {}, True),
    ("write", {}, True), ("edit", {}, True), ("docx", {}, True), ("xlsx", {}, True),
    ("google_mail", {"action": "list_messages"}, True), ("cron", {"action": "list"}, True),
    ("shell", {}, False), ("terminal", {}, False), ("browser", {}, False), ("scripts", {}, False),
    ("send_mail", {}, False), ("google_mail", {"action": "send"}, False), ("google_mail", {}, False),
    ("whatsapp_web", {}, False), ("public_link", {}, False), ("hosting", {}, False),
    ("cron", {"action": "add"}, False), ("google_calendar", {"action": "create"}, False),
    ("datastore", {"action": "sql"}, False), ("payments", {}, False), ("spend", {}, False),
    ("bat", {}, False), ("vatra", {}, False), ("crew", {}, False), ("flight_deck", {}, False),
    ("mcp_tracker_create_issue", {}, False), ("some_new_tool", {}, False),
    ("ask_user", {}, False), ("ask_human", {}, False), ("request_secret", {}, False),
])
def test_only_reading_and_file_writing_tools(tool, args, allowed):
    assert drive.extra_work_allows(tool, args) is allowed


class _GuardStub(AgentGuardMixin):
    def __init__(self):
        self.tools = SimpleNamespace(has_tool=lambda name: True)
        self.refused: list[tuple[str, str]] = []
        self._extra_work = {"active": True, "reply": ANSWER, "calls": 0, "refused": 0, "limit": 2}

    def _audit_refused_tool(self, name, arguments, reason, session_id_override):
        self.refused.append((name, reason))

    async def _enforce_guard(self, **kw):
        raise RuntimeError("reached the real guard")       # an allowed call goes on as usual


async def test_the_guard_refuses_other_tools_and_stops_at_the_limit():
    stub = _GuardStub()
    result = await stub._execute_tool_with_guard("send_mail", {"to": "x"}, "chat")
    assert result.success is False and "not available for extra work" in result.error
    assert stub._extra_work["refused"] == 1 and stub._extra_work["calls"] == 0
    for _ in range(2):                                       # two allowed calls pass the extra-work check
        with pytest.raises(RuntimeError, match="reached the real guard"):
            await stub._execute_tool_with_guard("read", {"path": "a"}, "chat")
    assert stub._extra_work["calls"] == 2
    result = await stub._execute_tool_with_guard("read", {"path": "b"}, "chat")
    assert result.success is False and "used up" in result.error
    assert [name for name, _ in stub.refused] == ["send_mail", "read"]
    # Outside the step nothing is refused here.
    stub._extra_work["active"] = False
    with pytest.raises(RuntimeError, match="reached the real guard"):
        await stub._execute_tool_with_guard("send_mail", {"to": "x"}, "chat")


# ── A turn that ends while the step is open ───────────────────────────


def test_a_turn_cut_off_during_the_step_still_delivers_the_answer(cfg):
    stub = _Stub()
    stub._extra_work = {"active": True, "reply": ANSWER, "calls": 4, "refused": 0, "limit": 10}
    assert stub._extra_work_rescue("I wasn't able to fully complete this request.", False) == (ANSWER, True)
    assert stub.session.messages[-1] == {"role": "assistant", "content": ANSWER}
    assert stub._extra_work["active"] is False
    # Not in the step (or already closed): the turn's own ending stands.
    assert stub._extra_work_rescue("Something failed.", False) == ("Something failed.", False)
    stub._extra_work = None
    assert stub._extra_work_rescue("Done.", True) == ("Done.", True)


def test_the_prompt_takes_its_limit():
    from pathlib import Path

    from captain_claw.instructions import InstructionLoader

    loader = InstructionLoader(base_dir=Path(drive.__file__).resolve().parent / "instructions", use_micro=False)
    text = loader.render("drive_extra_work_prompt.md", limit=10)
    assert "At most 10 tool calls" in text and drive.NOTHING_TO_ADD in text
