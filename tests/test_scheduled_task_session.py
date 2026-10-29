"""Flight Deck scheduled tasks on the agent: each task's turn runs in the
task's own session (found by id, else by name, else created), the lane goes
back to its own session afterwards, and the reply / mirror name the task."""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from captain_claw import mail_authority
from captain_claw.config import get_config
from captain_claw.session import Session
from captain_claw.web import chat_handler
from captain_claw.web_server import WebServer


class FakeWS:
    def __init__(self, lane: str | None = None):
        self.closed = False
        self.sent: list[str] = []
        if lane is not None:
            self._lane = lane

    async def send_str(self, data: str):
        self.sent.append(data)

    def frames(self) -> list[dict]:
        return [json.loads(d) for d in self.sent]


class FakeSM:
    """An in-memory session manager."""

    def __init__(self, *sessions: Session):
        self.rows = {s.id: s for s in sessions}
        self.saved: list[str] = []
        self.deleted: list[str] = []
        self.created = 0

    async def load_session(self, sid):
        s = self.rows.get(sid)
        return Session(id=s.id, name=s.name, metadata=dict(s.metadata), messages=list(s.messages)) if s else None

    async def load_session_by_name(self, name):
        for s in reversed(list(self.rows.values())):
            if s.name == name:
                return await self.load_session(s.id)
        return None

    async def create_session(self, name="default", metadata=None):
        self.created += 1
        s = Session(id=f"new-{self.created}", name=name, metadata=metadata or {})
        self.rows[s.id] = s
        return s

    async def save_session(self, session):
        self.saved.append(session.id)
        self.rows[session.id] = session

    async def delete_session(self, sid):
        self.deleted.append(sid)
        return self.rows.pop(sid, None) is not None


class TaskAgent:
    def __init__(self, sm: FakeSM, reply: str = "Brief: 3 meetings."):
        self.session_manager = sm
        self.session = Session(id="lane-AUTO", name="lane-AUTO")
        self.reply = reply
        self.seen_session = ""
        self.last_usage: dict = {}
        self.total_usage: dict = {}
        self.last_context_window: dict = {}
        self.provider = object()
        self.tools = types.SimpleNamespace(set_session_policy=lambda *a: None,
                                           clear_session_policy=lambda *a: None)

    def refresh_session_runtime_flags(self):
        pass

    def get_runtime_model_details(self):
        return {"provider": "p", "model": "m"}

    def _current_session_slug(self):
        return self.session.id

    async def complete(self, content):
        self.seen_session = self.session.id
        self.session.messages.append({"role": "assistant", "content": self.reply})
        return self.reply


def _server(main=None):
    s = WebServer.__new__(WebServer)
    s.agent = main or types.SimpleNamespace(session=Session(id="main", name="default"))
    s.clients = set()
    s._lane_agents = {}
    s._lane_locks = {}
    s._lane_sockets = {}
    s._public_agents = {}
    s._speaker_agents = {}
    s._busy = False
    s._active_task = None
    s._orchestrator = None
    return s


@pytest.fixture(autouse=True)
def sync_sends(monkeypatch):
    def _send(ws, data):
        ws.sent.append(data)
    monkeypatch.setattr("captain_claw.web_server.fire_and_forget_send", _send)
    monkeypatch.setattr("captain_claw.web.chat_handler.fire_and_forget_send", _send)


@pytest.fixture
def quiet_post_turn(monkeypatch):
    async def _noop(*a, **k):
        return None
    for target in ("captain_claw.reflections.maybe_auto_reflect",
                   "captain_claw.insights.maybe_extract_insights",
                   "captain_claw.nervous_system.maybe_dream",
                   "captain_claw.conversation_topics.maybe_classify_topics",
                   "captain_claw.intentions_generator.maybe_auto_propose"):
        monkeypatch.setattr(target, _noop)
    monkeypatch.setattr(get_config().ui, "next_steps", False)


def _task(**over):
    return chat_handler.task_marker({"id": "job_ab12", "name": "Morning briefing", **over})


# ── the marker ──

def test_task_marker_is_cleaned():
    assert chat_handler.task_marker(None) is None
    assert chat_handler.task_marker({"name": "no id"}) is None
    t = chat_handler.task_marker({"id": "job_1; DROP", "name": "  Morning \n brief ", "mirror": False})
    assert t == {"id": "job_1DROP", "name": "Morning brief", "session_id": "", "mirror": False}
    assert chat_handler.task_marker({"id": "j"})["name"] == "Scheduled task"
    assert chat_handler.task_marker({"id": "j"})["mirror"] is True


# ── finding the session ──

async def test_a_first_run_creates_the_tasks_session_and_the_lane_goes_back():
    sm = FakeSM()
    agent = TaskAgent(sm)
    prev = await chat_handler.enter_task_session(_server(), agent, _task())
    assert agent.session.name == "⏰ Morning briefing"
    assert agent.session.metadata["scheduled_task"] == "job_ab12"
    assert prev.id == "lane-AUTO"
    await chat_handler.leave_task_session(agent, prev)
    assert agent.session.id == "lane-AUTO" and sm.saved == ["new-1"]


async def test_a_known_session_is_reused_and_renamed_with_the_task():
    old = Session(id="s-1", name="⏰ Old name", metadata={"scheduled_task": "job_ab12"})
    sm = FakeSM(old)
    agent = TaskAgent(sm)
    await chat_handler.enter_task_session(_server(), agent, _task(session_id="s-1"))
    assert agent.session.id == "s-1" and agent.session.name == "⏰ Morning briefing"
    assert sm.created == 0


async def test_found_by_name_when_the_id_is_not_known_yet():
    sm = FakeSM(Session(id="s-9", name="⏰ Morning briefing", metadata={"scheduled_task": "job_ab12"}))
    agent = TaskAgent(sm)
    await chat_handler.enter_task_session(_server(), agent, _task())
    assert agent.session.id == "s-9" and sm.created == 0


async def test_only_a_session_marked_as_this_tasks_is_adopted():
    # The user's own session with the id / name, and a member's private one.
    sm = FakeSM(Session(id="s-user", name="⏰ Morning briefing", metadata={}),
                Session(id="s-member", name="x", metadata={"scheduled_task": "job_ab12", "speaker_id": "u2"}))
    agent = TaskAgent(sm)
    await chat_handler.enter_task_session(_server(), agent, _task(session_id="s-user"))
    assert agent.session.id == "new-1"
    agent = TaskAgent(sm)
    await chat_handler.enter_task_session(_server(), agent, _task(session_id="s-member"))
    assert agent.session.id not in ("s-member", "s-user")


async def test_a_session_open_in_the_main_chat_is_shared_not_copied():
    live = Session(id="s-1", name="⏰ Morning briefing", metadata={"scheduled_task": "job_ab12"})
    main = types.SimpleNamespace(session=live)
    agent = TaskAgent(FakeSM(Session(id="s-1", name="⏰ Morning briefing",
                                     metadata={"scheduled_task": "job_ab12"})))
    await chat_handler.enter_task_session(_server(main), agent, _task(session_id="s-1"))
    assert agent.session is live


# ── a whole turn ──

@pytest.mark.parametrize("mirror", [True, False])
async def test_a_task_turn_runs_in_its_session_names_it_and_mirrors_by_choice(
        quiet_post_turn, mirror):
    main_sm = types.SimpleNamespace(saved=0)

    async def _save(session):
        main_sm.saved += 1
    main_sm.save_session = _save
    main = types.SimpleNamespace(session=Session(id="main", name="default"), session_manager=main_sm)
    s = _server(main)
    viewer_a, caller = FakeWS(), FakeWS()
    s.clients = {viewer_a, caller}
    s._lane_sockets = {"AUTO": {caller}}
    caller._claw_borrowed_lane = "AUTO"
    sm = FakeSM()
    agent = TaskAgent(sm)
    task = _task(mirror=mirror)
    task = {**task, "previous": await chat_handler.enter_task_session(s, agent, task)}

    await chat_handler._run_agent(
        s, caller, agent, "[Automated turn] brief", None, lane="AUTO", no_flow=True,
        automation=mail_authority.Authority(mode="automated", kind="fd_scheduler"),
        rerouted=True, task=task)
    for _ in range(10):
        await asyncio.sleep(0)

    assert agent.seen_session == "new-1"                  # ran in the task's session
    assert agent.session.id == "lane-AUTO"                # and the lane is back on its own
    assert "new-1" in sm.saved
    (reply,) = [f for f in caller.frames() if f.get("role") == "assistant"]
    assert reply["task_session"] == {"id": "new-1", "name": "⏰ Morning briefing"}
    mirrored = [f for f in viewer_a.frames() if f.get("automation_lane") == "AUTO"]
    notes = [m for m in main.session.messages if m.get("origin_detail") == "automation_result"]
    if mirror:
        assert mirrored[0]["task"] == {"id": "job_ab12", "name": "Morning briefing", "session_id": "new-1"}
        assert notes and notes[0]["content"].startswith('[Scheduled task "Morning briefing" — ran in its own')
    else:
        assert not mirrored and not notes


# ── an onboarding exam's question (exam_turn) ──

async def test_an_exam_question_keeps_nothing_and_its_session_goes(monkeypatch):
    from captain_claw import exam_turn

    jobs: list[str] = []
    for target in ("captain_claw.reflections.maybe_auto_reflect",
                   "captain_claw.insights.maybe_extract_insights",
                   "captain_claw.nervous_system.maybe_dream",
                   "captain_claw.conversation_topics.maybe_classify_topics",
                   "captain_claw.intentions_generator.maybe_auto_propose"):
        async def _record(*a, _name=target, **k):
            jobs.append(_name)
        monkeypatch.setattr(target, _record)
    monkeypatch.setattr(get_config().ui, "next_steps", False)

    class ExamAgent(TaskAgent):
        seen_content = ""
        active = False

        async def complete(self, content):
            self.seen_content = content
            self.active = exam_turn.active(self)
            exam_turn.note_call(self, "knowledge", {"action": "search", "query": "refund", "_agent": self},
                                True, "[1] Handbook › Refunds\n" + "x" * 9000)
            self.last_usage = {"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000}
            return await super().complete(content)

    s = _server()
    caller = FakeWS()
    s._lane_sockets = {exam_turn.LANE: {caller}}
    sm = FakeSM()
    agent = ExamAgent(sm, reply="14 days.")
    task = chat_handler.task_marker({"id": "exam-ab-q1", "name": "Onboarding exam", "mirror": False})
    task = {**task, "previous": await chat_handler.enter_task_session(s, agent, task)}

    await chat_handler._run_agent(
        s, caller, agent, "What is our refund window?", None, lane=exam_turn.LANE, no_flow=True,
        no_next_steps=True, no_rephrase=True,
        automation=mail_authority.Authority(mode="automated", kind=exam_turn.KIND), fd_delivers=True, task=task)
    for _ in range(10):
        await asyncio.sleep(0)

    assert agent.seen_content == "What is our refund window?"    # as a user would ask it: no automated-turn line
    assert agent.active and not exam_turn.active(agent)           # an exam turn while it ran, not after
    assert agent.seen_session == "new-1" and agent.session.id == "lane-AUTO"
    assert sm.deleted == ["new-1"] and "new-1" not in sm.saved and "new-1" not in sm.rows
    (reply,) = [f for f in caller.frames() if f.get("role") == "assistant"]
    assert reply["content"] == "14 days." and reply["exam"]["usage"]["total_tokens"] == 1000
    (call,) = reply["exam"]["lookups"]
    assert call["tool"] == "knowledge" and call["args"] == {"action": "search", "query": "refund"}
    assert call["ok"] and len(call["result"]) == 4000
    assert jobs == []                                             # no reflection, insights, dreams, topics


def test_the_exam_kind_is_known():
    from captain_claw import turn_source

    assert mail_authority.from_wire({"kind": "exam"}, default_kind="unknown").kind == "exam"
    assert turn_source.from_automation("exam") == "exam"
    assert turn_source.label_for("exam") == "Onboarding exams"
