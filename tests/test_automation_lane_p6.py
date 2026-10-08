"""Context engine P6: automated traffic on its own lane, results mirrored into
the main chat, and the new-session cue."""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from captain_claw import mail_authority
from captain_claw.config import get_config
from captain_claw.session import Session
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


@pytest.fixture(autouse=True)
def sync_sends(monkeypatch):
    def _send(ws, data):
        ws.sent.append(data)
    monkeypatch.setattr("captain_claw.web_server.fire_and_forget_send", _send)
    monkeypatch.setattr("captain_claw.web.chat_handler.fire_and_forget_send", _send)


def _server(main_agent=None):
    s = WebServer.__new__(WebServer)
    s.agent = main_agent or types.SimpleNamespace(name="main")
    s.clients = set()
    s._lane_agents = {}
    s._lane_locks = {}
    s._lane_sockets = {}
    s._public_agents = {}
    s._busy = False
    s._active_task = None
    s._orchestrator = None
    return s


def _auth(kind: str):
    return mail_authority.Authority(mode="automated", kind=kind)


# ── Which turns move ───────────────────────────────────────────────────


@pytest.mark.parametrize("kind,moves", [
    ("fd_scheduler", True), ("autonomy", True), ("plan", True), ("peer", True), ("mcp_task", True),
    ("peer_relay", False), ("flow", False), ("unknown", False),
])
def test_which_automated_turns_move_to_the_automation_lane(kind, moves):
    from captain_claw.web.chat_handler import automation_lane_for

    assert (automation_lane_for(_server(), _auth(kind), "A", False) == "AUTO") is moves


def test_human_public_and_side_lane_turns_stay(monkeypatch):
    from captain_claw.web.chat_handler import automation_lane_for

    s = _server()
    assert automation_lane_for(s, None, "A", False) == ""
    assert automation_lane_for(s, _auth("fd_scheduler"), "A", True) == ""
    assert automation_lane_for(s, _auth("fd_scheduler"), "B", False) == ""
    monkeypatch.setattr(get_config().session, "automation_lane", "")
    assert automation_lane_for(s, _auth("fd_scheduler"), "A", False) == ""
    monkeypatch.setattr(get_config().session, "automation_lane", "AUTO")
    monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    assert automation_lane_for(s, _auth("fd_scheduler"), "A", False) == ""


async def test_the_automation_lane_agent_leaves_the_global_tools_alone():
    s = _server()
    built = []

    async def fake_build(session, send, **kwargs):
        built.append((session.name, kwargs))
        return types.SimpleNamespace(session=session)

    async def fake_session(lane):
        return types.SimpleNamespace(id=lane, name=f"lane-{lane}")

    s._build_scoped_agent = fake_build
    s._lane_session = fake_session
    await s._get_lane_agent("AUTO")
    await s._get_lane_agent("B")
    assert built == [("lane-AUTO", {"register_tools": False}), ("lane-B", {})]


# ── A rerouted turn ────────────────────────────────────────────────────


class _AutoAgent:
    def __init__(self, reply: str):
        self.reply = reply
        self.session = types.SimpleNamespace(id="lane-AUTO", messages=[])
        self.last_usage: dict = {}
        self.total_usage: dict = {}
        self.last_context_window: dict = {}
        self.provider = object()
        self.tools = types.SimpleNamespace(set_session_policy=lambda *a: None,
                                           clear_session_policy=lambda *a: None)

    def get_runtime_model_details(self):
        return {"provider": "p", "model": "m"}

    def _current_session_slug(self):
        return "lane-AUTO"

    async def complete(self, content):
        return self.reply


class _SessionManager:
    def __init__(self):
        self.saved = 0

    async def save_session(self, session):
        self.saved += 1


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


@pytest.mark.parametrize("kind,mirrored", [("fd_scheduler", True), ("peer", False)])
async def test_a_rerouted_turn_reaches_its_caller_and_mirrors_user_facing_results(
        quiet_post_turn, monkeypatch, kind, mirrored):
    from captain_claw.web import chat_handler

    main = types.SimpleNamespace(session=Session(id="main", name="default"),
                                 session_manager=_SessionManager())
    s = _server(main)
    viewer_a, caller, auto_viewer = FakeWS(), FakeWS(), FakeWS("AUTO")
    s.clients = {viewer_a, caller}
    s._lane_sockets = {"AUTO": {auto_viewer, caller}}
    caller._claw_borrowed_lane = "AUTO"
    agent = _AutoAgent("Morning brief: 3 meetings today.")

    await chat_handler._run_agent(s, caller, agent, "[Automated turn] brief", None, lane="AUTO",
                                  no_flow=True, automation=_auth(kind), rerouted=True)
    for _ in range(10):
        await asyncio.sleep(0)

    replies = [f for f in caller.frames() if f.get("type") == "chat_message" and f.get("role") == "assistant"]
    assert [f["content"] for f in replies] == ["Morning brief: 3 meetings today."]   # once, not twice
    assert any(f.get("type") == "status" and f.get("status") == "ready" for f in caller.frames())
    assert caller not in s._lane_sockets["AUTO"]                                   # handed back
    assert any(f.get("type") == "chat_message" for f in auto_viewer.frames())
    mirror = [f for f in viewer_a.frames() if f.get("automation_lane") == "AUTO"]
    notes = [m for m in main.session.messages if m.get("origin_detail") == "automation_result"]
    if mirrored:
        assert mirror and mirror[0]["content"] == "Morning brief: 3 meetings today."
        assert len(notes) == 1 and notes[0]["role"] == "assistant"
        assert "Morning brief" in notes[0]["content"] and main.session_manager.saved == 1
    else:
        assert not mirror and not notes


async def test_the_mirror_waits_for_a_busy_main_chat(monkeypatch):
    from captain_claw.web import chat_handler

    main = types.SimpleNamespace(session=Session(id="main", name="default"),
                                 session_manager=_SessionManager())
    s = _server(main)
    s._busy = True
    task = asyncio.create_task(chat_handler._mirror_automation_result(
        s, None, _auth("autonomy"), "Nudge: reply to Marko", "AUTO"))
    await asyncio.sleep(0)
    assert main.session.messages == []
    s._busy = False
    await asyncio.wait_for(task, timeout=3)
    assert main.session.messages[-1]["origin"] == "system_note"


# ── The new-session cue ────────────────────────────────────────────────


@pytest.mark.parametrize("text,rest", [
    ("Nova tema: blog post ideas", "blog post ideas"),
    ("nova tema", ""),
    ("NEW TOPIC\nhow do I…", "how do I…"),
    ("new topic - quick q", "quick q"),
    ("Novi razgovor. Kako si?", "Kako si?"),
    ("Nova tema za blog", None),
    ("what is the new topic?", None),
])
def test_rotation_cue(text, rest):
    from captain_claw.web.slash_commands import rotation_cue

    assert rotation_cue(text) == rest


async def test_a_cue_starts_a_new_session_then_runs_the_rest(monkeypatch):
    from captain_claw.web import ws_handler

    s = _server(types.SimpleNamespace(plan_mode_auto=False))
    calls: list = []

    async def fake_command(server, ws, raw):
        calls.append(("command", raw))

    async def fake_chat(server, ws, content, **kwargs):
        calls.append(("chat", content))

    async def resolve(ws):
        return s.agent

    s.resolve_agent = resolve
    monkeypatch.setattr("captain_claw.web.slash_commands.handle_command", fake_command)
    monkeypatch.setattr("captain_claw.web.chat_handler.handle_chat", fake_chat)
    await ws_handler.handle_ws_message(s, FakeWS(), {"type": "chat", "content": "Nova tema: plan the trip"})
    assert calls == [("command", "/new"), ("chat", "plan the trip")]

    calls.clear()
    public = FakeWS()
    public._public_session_id = "pub-1"
    await ws_handler.handle_ws_message(s, public, {"type": "chat", "content": "Nova tema: hi"})
    assert calls == [("chat", "Nova tema: hi")]                 # never for a public visitor


# ── Fleet notices ──────────────────────────────────────────────────────


async def test_a_fleet_notice_goes_to_the_automation_lane_and_its_event_stays(monkeypatch):
    from captain_claw.web import ws_handler

    main = types.SimpleNamespace(session=Session(id="main", name="default"),
                                 session_manager=_SessionManager())
    auto = types.SimpleNamespace(session=Session(id="lane-AUTO", name="lane-AUTO"),
                                 session_manager=_SessionManager())
    s = _server(main)

    async def lane_agent(lane):
        assert lane == "AUTO"
        return auto

    async def resolve(ws):
        return main

    s._get_lane_agent = lane_agent
    s.resolve_agent = resolve
    notice = "[Flight Deck] Agent 'researcher' has joined the fleet on port 24001. Current fleet: researcher, writer."
    await ws_handler.handle_ws_message(s, FakeWS(), {"type": "notification", "content": notice})

    assert main.session.messages == []
    assert auto.session.messages[-1]["origin"] == "fleet_notice"
    assert main.session.metadata["fleet_events"][-1]["text"] == notice

    # The main chat's next turn still gets the one-line fleet note.
    from captain_claw.agent import Agent

    class _Words:
        provider = "openai"
        model = "stub"

        def count_tokens(self, text):
            return len(str(text or "").split()) or 1

    agent = Agent(provider=_Words())
    agent.session = main.session
    agent._build_env_now_text = lambda: ""
    agent.session.add_message("user", "who is around?", origin="human")
    agent._turn_user_text = "who is around?"
    messages = agent._build_messages(tool_messages_from_index=len(agent.session.messages) - 1,
                                     query="who is around?")
    block = next(m.content for m in messages if "[INTERNAL CONTEXT" in str(m.content) and m.role == "user")
    assert "'researcher' joined" in block


# ── The compaction digest ──────────────────────────────────────────────


def _digest_agent(tmp_path, monkeypatch):
    import captain_claw.conversation_topics as ct
    from captain_claw.agent import Agent

    mgr = ct.ConversationTopicsManager(tmp_path / "topics.db")
    monkeypatch.setattr(ct, "_MANAGER", mgr)

    class _Words:
        provider = "openai"
        model = "stub"

        def count_tokens(self, text):
            return len(str(text or "").split()) or 1

    agent = Agent(provider=_Words())
    agent.session = Session(id="s1", name="d")
    return agent, mgr


def test_the_digest_groups_exchanges_by_topic(tmp_path, monkeypatch):
    agent, mgr = _digest_agent(tmp_path, monkeypatch)
    s = agent.session
    s.add_message("user", "book the Munich hotel", origin="human")
    s.add_message("assistant", "Booked Hotel Adler.", origin="model")
    s.add_message("user", "[SCHEDULED TASK — cron job 1] brief", origin="cron")
    s.add_message("assistant", "Brief sent.", origin="model")
    s.add_message("user", "write a haiku", origin="human")
    s.add_message("assistant", "Autumn leaves fall…", origin="model")
    tid = mgr.upsert_topic("Munich trip", summary="the trip")
    mgr.add_messages(tid, [{"role": "user", "excerpt": "x", "msg_id": s.messages[0]["message_id"]}])

    digest = agent._digest_for_compaction(s.messages)
    assert "Munich trip [munich-trip] — 1 exchange(s)" in digest
    assert 'last reply: "Booked Hotel Adler."' in digest
    assert '"write a haiku" → "Autumn leaves fall…"' in digest
    assert "1 cron turn(s)" in digest
    mgr._conn.close()


async def test_compaction_uses_the_digest_by_default(tmp_path, monkeypatch):
    from captain_claw.config import set_config

    agent, mgr = _digest_agent(tmp_path, monkeypatch)
    agent.session_manager = _SessionManager()
    called = []

    async def no_llm(messages):
        called.append(1)
        return "model summary"

    agent._summarize_for_compaction = no_llm
    old = get_config().model_copy(deep=True)
    cfg = old.model_copy(deep=True)
    cfg.context.max_tokens = 60
    set_config(cfg)
    try:
        for n in range(12):
            agent.session.add_message("user", f"question {n} " + "word " * 6, origin="human")
            agent.session.add_message("assistant", f"answer {n} " + "word " * 6, origin="model")
        compacted, _stats = await agent.compact_session(force=False, trigger="auto")
        assert compacted and called == []
        assert agent.session.messages[0]["content"].count("Digest of the earlier part") == 1
    finally:
        set_config(old)
        mgr._conn.close()


async def test_a_cancel_from_a_rerouted_turn_stops_that_turn_not_lane_a():
    from captain_claw.web import ws_handler

    main = types.SimpleNamespace(cancel_event=asyncio.Event())
    auto = types.SimpleNamespace(cancel_event=asyncio.Event())
    s = _server(main)
    s._lane_agents = {"AUTO": auto}

    async def resolve(ws):
        return main

    s.resolve_agent = resolve
    async def resolve(ws):
        return auto if getattr(ws, "_lane", "A") == "AUTO" else main

    s.resolve_agent = resolve
    caller = FakeWS()
    from captain_claw.web.chat_handler import _borrow_socket

    _borrow_socket(s, caller, "AUTO")
    await ws_handler.handle_ws_message(s, caller, {"type": "cancel"})
    assert auto.cancel_event.is_set() and not main.cancel_event.is_set()


# ── Review round ───────────────────────────────────────────────────────


def test_a_borrowed_socket_hears_its_lane_only(monkeypatch):
    from captain_claw.web.chat_handler import _borrow_socket, _return_socket

    monkeypatch.setattr(get_config().web, "public_run", "")
    s = _server()
    caller, user_a = FakeWS("A"), FakeWS("A")
    s.clients = {caller, user_a}
    s._lane_sockets = {"A": {caller, user_a}}
    _borrow_socket(s, caller, "AUTO")
    s._broadcast({"type": "chat_message", "role": "assistant", "content": "lane A reply"})
    s._lane_send("AUTO")({"type": "chat_message", "role": "assistant", "content": "job result"})
    assert [f["content"] for f in caller.frames()] == ["job result"]
    assert [f["content"] for f in user_a.frames()] == ["lane A reply"]
    _return_socket(s, caller, "AUTO")
    assert caller._lane == "A" and caller in s._lane_sockets["A"] and caller not in s._lane_sockets["AUTO"]


def test_the_automation_agent_takes_the_main_agents_context():
    from captain_claw.web.chat_handler import _sync_automation_agent

    main_session = Session(id="main", name="default")
    main_session.metadata.update(whatsapp_waid="385...", origin={"kind": "whatsapp"}, fd_url="http://fd")
    main = types.SimpleNamespace(session=main_session, _fleet_identity={"name": "Olga"},
                                 _fleet_instructions="Be brief.", _peer_agents=[{"name": "r"}],
                                 _fd_url="http://fd", memory="MEM",
                                 provider=types.SimpleNamespace(provider="deepseek", model="v4"))
    auto = types.SimpleNamespace(session=Session(id="lane-AUTO", name="lane-AUTO"), memory=None,
                                 provider=types.SimpleNamespace(provider="deepseek", model="v3"))
    _sync_automation_agent(_server(main), auto)
    assert auto._fleet_identity == {"name": "Olga"} and auto._peer_agents == [{"name": "r"}]
    assert auto.session.metadata["whatsapp_waid"] == "385..." and auto.memory == "MEM"
    assert auto.provider.model == "v4"


async def test_results_wait_while_lane_a_is_busy_and_go_in_before_its_next_turn():
    from captain_claw.web import chat_handler

    main = types.SimpleNamespace(session=Session(id="main", name="default"),
                                 session_manager=_SessionManager())
    s = _server(main)
    s._busy = True
    monkey_ticks = chat_handler._MIRROR_WAIT_TICKS
    chat_handler._MIRROR_WAIT_TICKS = 1
    try:
        await chat_handler._mirror_automation_result(s, None, _auth("autonomy"), "Nudge: reply to Marko", "AUTO")
    finally:
        chat_handler._MIRROR_WAIT_TICKS = monkey_ticks
    assert main.session.messages == []                                   # never mid-turn
    assert main.session.metadata[chat_handler.PENDING_RESULTS_KEY][0]["text"] == "Nudge: reply to Marko"
    s._busy = False
    assert await chat_handler.flush_automation_results(s) == 1
    assert main.session.messages[-1]["origin_detail"] == "automation_result"
    assert chat_handler.PENDING_RESULTS_KEY not in main.session.metadata


async def test_the_cue_waits_for_a_busy_agent_and_reads_past_the_surface_block(monkeypatch):
    from captain_claw.web import ws_handler

    main = types.SimpleNamespace(plan_mode_auto=False)
    s = _server(main)
    calls: list = []

    async def fake_command(server, ws, raw):
        calls.append(("command", raw))

    async def fake_chat(server, ws, content, **kwargs):
        calls.append(("chat", content))

    async def resolve(ws):
        return main

    s.resolve_agent = resolve
    sent: list = []

    async def fake_send(ws, msg):
        sent.append(msg)

    s._send = fake_send
    monkeypatch.setattr("captain_claw.web.slash_commands.handle_command", fake_command)
    monkeypatch.setattr("captain_claw.web.chat_handler.handle_chat", fake_chat)

    s._busy = True
    await ws_handler.handle_ws_message(s, FakeWS(), {"type": "chat", "content": "Nova tema: hi"})
    assert calls == [] and sent[-1]["type"] == "error"

    s._busy = False
    block = ("[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply.]\n"
             "Keep replies short.\nUSER MESSAGE:\n")
    await ws_handler.handle_ws_message(s, FakeWS(), {"type": "chat", "content": block + "nova tema: plan the trip"})
    assert calls == [("command", "/new"), ("chat", block + "plan the trip")]
    assert sent[-1] == {"type": "chat_message", "role": "user", "content": "plan the trip", "rotation_cue": True}


async def test_a_lanes_new_session_keeps_the_lane_name():
    from captain_claw.web.slash_commands import _create_new_session
    from captain_claw.web_server import _LaneServerView

    created = []

    class _SM:
        async def create_session(self, name=None, metadata=None):
            created.append(name)
            return Session(id="new", name=name)

    lane_agent = types.SimpleNamespace(session=Session(id="old", name="lane-B"), session_manager=_SM())
    s = _server(types.SimpleNamespace(session=Session(id="main", name="default"), session_manager=_SM()))
    await _create_new_session(_LaneServerView(s, lane_agent, lambda m: None, lane="B"), None)
    await _create_new_session(s, None)
    assert created == ["lane-B", "web-session"]


def test_switching_into_a_session_another_agent_holds_is_refused():
    from captain_claw.web.slash_commands import _held_elsewhere
    from captain_claw.web_server import _LaneServerView

    main = types.SimpleNamespace(session=Session(id="main", name="default"))
    lane_b = types.SimpleNamespace(session=Session(id="b", name="lane-B"))
    s = _server(main)
    s._lane_agents = {"B": lane_b}
    s._speaker_agents = {}
    view = _LaneServerView(s, lane_b, lambda m: None, lane="B")
    assert _held_elsewhere(view, "main") is True
    assert _held_elsewhere(view, "b") is False
    assert _held_elsewhere(view, "elsewhere") is False


async def test_public_visitors_get_no_commands():
    from captain_claw.web.slash_commands import handle_command

    s = _server(types.SimpleNamespace(session=Session(id="main", name="default")))
    sent: list = []

    async def fake_send(ws, msg):
        sent.append(msg)

    s._send = fake_send
    visitor = FakeWS()
    visitor._public_session_id = "pub-1"
    await handle_command(s, visitor, "/clear")
    assert sent[-1]["content"] == "Commands aren't available in this session."


def test_the_digest_carries_the_earlier_summary_and_the_last_reply(tmp_path, monkeypatch):
    agent, mgr = _digest_agent(tmp_path, monkeypatch)
    s = agent.session
    s.messages.append({"role": "assistant", "tool_name": "compaction_summary", "origin": "system_note",
                       "origin_detail": "compaction",
                       "content": "Conversation summary of earlier messages (compacted memory):\n"
                                  "Digest of the earlier part of this conversation (…):\n"
                                  "- Booking code AX-77 for Hotel Adler"})
    s.add_message("user", "research the hotels", origin="human")
    s.add_message("assistant", "I've asked the researcher.", origin="model")
    s.add_message("user", "[Delegated result from researcher] Adler.", origin="delegated_result")
    s.add_message("assistant", "The researcher says Adler is best.", origin="model")
    digest = agent._digest_for_compaction(s.messages)
    assert "Before that: - Booking code AX-77 for Hotel Adler" in digest
    assert '→ "The researcher says Adler is best."' in digest
    mgr._conn.close()


def test_a_slice_nobody_typed_in_goes_to_the_model(tmp_path, monkeypatch):
    agent, mgr = _digest_agent(tmp_path, monkeypatch)
    s = agent.session
    s.add_message("user", "[Autonomous nudge] check mail", origin="autonomy")
    s.add_message("assistant", "Nothing new.", origin="model")
    assert agent._digest_for_compaction(s.messages) == ""
    mgr._conn.close()
