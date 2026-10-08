"""Shared-agent member sockets (contract part 2 §2-§4, part 2b §5-§6).

Flight Deck connects a member with a signed ``X-FD-Speaker`` header. The
agent then serves that member from their own instance and private session:
never in ``clients`` / ``_lane_sockets``, welcome (with ``speaker_ack``)
before anything else, a frame and slash allowlist, and exactly ONE
``status/ready`` frame with ``turn_end`` for every chat frame — whatever
path the frame takes.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import types
from unittest.mock import AsyncMock

import aiohttp.web
import pytest

from captain_claw import speaker
from captain_claw.config import get_config
from captain_claw.speaker import (
    NOT_ALLOWED_MESSAGE,
    SPEAKER_MODE_NOTE,
    SPEAKER_TOOL_ALLOWLIST,
    Principal,
    sign_assertion,
    speaker_ack_for,
    speaker_commands,
)
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult
from captain_claw.web.speaker_ws import speaker_gate
from captain_claw.web.ws_handler import handle_ws_message, ws_handler
from captain_claw.web_server import SpeakerCapacityError, WebServer, _LaneServerView

WEB_AUTH = "test-web-auth"


# ── fakes ────────────────────────────────────────────────────────────


class FakeWS:
    """Enough of a WebSocketResponse: records sends/close, replays fed frames."""

    def __init__(self):
        self.closed = False
        self.close_code: int | None = None
        self.sent: list[str] = []
        self._inbox: asyncio.Queue = asyncio.Queue()

    async def prepare(self, request):
        return None

    async def send_str(self, data: str):
        self.sent.append(data)

    async def close(self, code: int = 1000, message: bytes = b""):
        self.closed = True
        self.close_code = code
        self._inbox.put_nowait(None)

    def exception(self):
        return None

    def feed(self, obj) -> None:
        self._inbox.put_nowait(types.SimpleNamespace(
            type=aiohttp.web.WSMsgType.TEXT, data=json.dumps(obj),
        ))

    def hangup(self) -> None:
        self._inbox.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self._inbox.get()
        if msg is None:
            raise StopAsyncIteration
        return msg

    def frames(self) -> list[dict]:
        return [json.loads(s) for s in self.sent]

    def of_type(self, t: str) -> list[dict]:
        return [f for f in self.frames() if f.get("type") == t]

    def turn_ends(self, tid: str) -> int:
        return sum(1 for f in self.frames() if f.get("type") == "status"
                   and f.get("status") == "ready" and f.get("turn_end") == tid)


class _Rec(Tool):
    def __init__(self, name):
        self.name = name
        self.description = name
        self.parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return ToolResult(success=True)


class FakeAgent:
    """A speaker instance: real session, the process-global registry."""

    def __init__(self, session, tools, sm):
        self.session = session
        self.tools = tools
        self.session_manager = sm
        self.cancel_event = asyncio.Event()
        self.last_usage: dict = {}
        self.total_usage: dict = {}
        self.last_context_window: dict = {}
        self.provider = types.SimpleNamespace(model="m", provider="p")
        self.instructions = types.SimpleNamespace(
            _cache={"system_prompt.md": "x", "micro_system_prompt.md": "y", "other.md": "z"},
        )
        self.complete_impl = None
        self.completed: list[str] = []

    def _current_session_slug(self):
        return self.session.id

    def get_runtime_model_details(self):
        return {"provider": "p", "model": "m"}

    async def complete(self, content):
        self.completed.append(content)
        if self.complete_impl is not None:
            return await self.complete_impl(self, content)
        return f"reply to {content}"

    def archive_current_session_to_history(self):
        return 0

    def refresh_session_runtime_flags(self):
        return None

    def _sync_runtime_flags_from_session(self):
        return None

    def _empty_usage(self):
        return {}

    async def _refresh_insights_context_cache(self):
        return None

    async def _refresh_nervous_system_cache(self):
        return None


class MainAgent:
    """The owner's lane-A agent. Any touch from a member frame is a bug."""

    def __init__(self, tools, sm, session):
        self.tools = tools
        self.session_manager = sm
        self.session = session
        self.approval_callback = None
        self._force_script_mode = False
        self._active_personality_id = None
        self._peer_agents = ["owner-peer"]
        self._fleet_identity = {"name": "Helper"}
        self._fleet_instructions = "Be kind."
        self._fd_url = "http://localhost:25080"
        self.set_session_model = AsyncMock(return_value=(True, "ok"))
        self._execute_tool_with_guard = AsyncMock()
        self.cancel_event = asyncio.Event()
        self.instructions = types.SimpleNamespace(_cache={})

    def get_runtime_model_details(self):
        return {"provider": "p", "model": "m"}

    def get_allowed_models(self):
        return [{"id": "x"}]


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def sync_sends(monkeypatch):
    def _send(ws, data):
        ws.sent.append(data)
    monkeypatch.setattr("captain_claw.web_server.fire_and_forget_send", _send)
    monkeypatch.setattr("captain_claw.web.chat_handler.fire_and_forget_send", _send)


@pytest.fixture(autouse=True)
def quiet_background(monkeypatch):
    """Post-turn background jobs and task naming: no LLM, no DB."""
    async def _noop(*a, **k):
        return None

    async def _no_name(*a, **k):
        return ""

    for target in (
        "captain_claw.reflections.maybe_auto_reflect",
        "captain_claw.insights.maybe_extract_insights",
        "captain_claw.nervous_system.maybe_dream",
        "captain_claw.conversation_topics.maybe_classify_topics",
    ):
        monkeypatch.setattr(target, _noop)
    monkeypatch.setattr("captain_claw.web.chat_handler._generate_task_name", _no_name)


@pytest.fixture(autouse=True)
def fresh_nonces():
    speaker._NONCES.clear()
    yield
    speaker._NONCES.clear()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Nothing here may reach the real ~/.captain-claw or a real FD data dir
    (real Agents are built below): HOME, FD_DATA_DIR, every config DB path and
    the global session / topic managers point at tmp first."""
    import captain_claw.conversation_topics as _ct
    from captain_claw import session as _session

    home = tmp_path / "isolated-home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    for var in ("CLAW_VFS_ROOT", "CLAW_VFS_USER", "FD_OWNER_ID", "FD_URL"):
        monkeypatch.delenv(var, raising=False)
    cfg = get_config()
    for section, attr in (
        ("memory", "path"), ("session", "path"), ("insights", "db_path"),
        ("conversation_topics", "db_path"), ("nervous_system", "db_path"),
        ("sister_session", "db_path"), ("cognitive_metrics", "db_path"),
        ("datastore", "path"), ("autonomous_work", "db_path"),
    ):
        monkeypatch.setattr(getattr(cfg, section), attr, str(home / ".captain-claw" / f"{section}.db"))
    monkeypatch.setattr(_session, "_manager", _session.SessionManager(home / ".captain-claw" / "s.db"))
    monkeypatch.setattr(_ct, "_MANAGER", None)
    monkeypatch.setattr(speaker, "_IN_FLIGHT", 0)
    monkeypatch.setattr(speaker, "_MAIN_LOOP", None)
    return home


@pytest.fixture(autouse=True)
def agent_config(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg.web, "auth_token", WEB_AUTH)
    monkeypatch.setattr(cfg.web, "public_run", False)
    return cfg


@pytest.fixture
async def sm(tmp_path, monkeypatch):
    from captain_claw.session import SessionManager

    manager = SessionManager(tmp_path / "sessions.db")
    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: manager)
    yield manager
    await manager.close()


@pytest.fixture
async def server(sm, monkeypatch):
    s = WebServer.__new__(WebServer)          # skip __init__'s heavy wiring
    s.clients = set()
    s._lane_agents = {}
    s._lane_locks = {}
    s._lane_sockets = {}
    s._public_agents = {}
    s._pending_playbook_approvals = {}
    s._busy = False
    s._orchestrator = None
    s._inbound_queue = asyncio.Queue()
    s._init_speaker_state()
    registry = ToolRegistry()
    for name in ["shell", "history", "read", *sorted(SPEAKER_TOOL_ALLOWLIST)]:
        registry.register(_Rec(name))
    s._registry = registry
    owner_session = await sm.create_session(name="owner-main")
    owner_session.add_message("user", "OWNER PRIVATE MESSAGE")
    await sm.save_session(owner_session)
    s.agent = MainAgent(registry, sm, owner_session)
    s.built: list[dict] = []

    async def fake_build(session, send, *, register_tools=True,
                         warm_owner_caches=True, approval_owner=None):
        s.built.append({"register_tools": register_tools,
                        "warm_owner_caches": warm_owner_caches,
                        "approval_owner": approval_owner})
        return FakeAgent(session, registry, sm)

    s._build_scoped_agent = fake_build
    s.test_sockets: list[FakeWS] = []
    yield s
    for ws in s.test_sockets:
        ws.hangup()
    for ws in s.test_sockets:
        try:
            await asyncio.wait_for(ws.task, 5)
        except BaseException:
            pass
    for agent in list(s._speaker_agents.values()):
        task = getattr(agent, "_public_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass


def make_header(sub="u-member", lane="A", name="Ana", web_auth=WEB_AUTH, **over) -> str:
    now = int(time.time())
    payload = {
        "v": 1, "sub": sub, "name": name, "owner": "u-owner", "owner_name": "Olga",
        "ref": "process:helper:0123456789abcdef", "lane": lane,
        "conn": secrets.token_hex(8), "iat": now, "exp": now + 60,
        "nonce": secrets.token_hex(8),
    }
    payload.update(over)
    return sign_assertion(payload, web_auth)


async def wait_for(cond, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met in time")


async def connect(server, monkeypatch, header=None, **hdr):
    """Open a member socket through the real ws_handler dispatch."""
    ws = FakeWS()
    header = header or make_header(**hdr)
    monkeypatch.setattr(aiohttp.web, "WebSocketResponse", lambda *a, **k: ws)
    request = types.SimpleNamespace(headers={"X-FD-Speaker": header}, query={}, cookies={})
    task = asyncio.create_task(ws_handler(server, request))
    ws.task = task
    server.test_sockets.append(ws)

    def _ready():
        key = getattr(ws, "_speaker_key", None)
        return ws.closed or (key is not None and ws in server._speaker_sockets.get(key, ()))

    await wait_for(_ready)
    ws.task = task
    ws.header = header
    return ws


async def disconnect(ws):
    ws.hangup()
    await asyncio.wait_for(ws.task, 5)


async def chat(ws, content, tid=None, **extra):
    tid = tid or secrets.token_hex(8)
    ws.feed({"type": "chat", "content": content, "_fd_turn": tid, "no_next_steps": True, **extra})
    return tid


async def settle(server, ws, tid):
    """Wait for the turn's end, then for its task, then a beat for strays."""
    await wait_for(lambda: ws.turn_ends(tid) >= 1)
    agent = server._speaker_agents.get(ws._speaker_key)
    task = getattr(agent, "_public_task", None)
    if task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(task), 5)
        except BaseException:
            pass
    await asyncio.sleep(0.02)


# ── handshake ────────────────────────────────────────────────────────


async def test_speaker_header_on_a_public_run_agent_is_4400(server, monkeypatch, agent_config):
    monkeypatch.setattr(agent_config.web, "public_run", True)
    ws = await connect(server, monkeypatch)
    await asyncio.wait_for(ws.task, 5)
    assert ws.close_code == 4400 and ws.sent == []
    assert server._speaker_agents == {} and ws not in server.clients


@pytest.mark.parametrize("bad", ["forged", "expired", "garbage"])
async def test_bad_header_is_4401(server, monkeypatch, bad):
    header = {
        "forged": make_header(web_auth="someone-elses-token"),
        "expired": make_header(iat=int(time.time()) - 200, exp=int(time.time()) - 140),
        "garbage": "v1.not.valid",
    }[bad]
    ws = await connect(server, monkeypatch, header=header)
    await asyncio.wait_for(ws.task, 5)
    assert ws.close_code == 4401 and ws.sent == []
    assert server._speaker_agents == {}


async def test_replayed_header_is_4401(server, monkeypatch):
    first = await connect(server, monkeypatch)
    assert first.close_code is None
    replay = await connect(server, monkeypatch, header=first.header)
    await asyncio.wait_for(replay.task, 5)
    assert replay.close_code == 4401 and replay.sent == []
    await disconnect(first)


async def test_instance_cap_is_4429(server, monkeypatch):
    monkeypatch.setattr(speaker, "SPEAKER_MAX_INSTANCES", 1)
    first = await connect(server, monkeypatch, sub="u-one")
    second = await connect(server, monkeypatch, sub="u-two")
    await asyncio.wait_for(second.task, 5)
    assert first.close_code is None and second.close_code == 4429
    await disconnect(first)


async def test_per_member_cap_is_4429(server, monkeypatch):
    monkeypatch.setattr(speaker, "SPEAKER_MAX_PER_USER", 1)
    first = await connect(server, monkeypatch, lane="A")
    second = await connect(server, monkeypatch, lane="B")
    await asyncio.wait_for(second.task, 5)
    assert second.close_code == 4429
    await disconnect(first)


async def test_without_the_header_nothing_changes():
    """The owner path is untouched: no header → no speaker branch."""
    import inspect

    from captain_claw.web import ws_handler as mod

    src = inspect.getsource(mod.ws_handler)
    assert src.index('request.headers.get("X-FD-Speaker")') < src.index("if public_mode:")
    assert src.index("speaker_ws_session(") < src.index("server.clients.add(ws)")


# ── isolation ────────────────────────────────────────────────────────


async def test_speaker_socket_is_never_a_client_or_lane_socket(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    assert ws not in server.clients
    assert all(ws not in socks for socks in server._lane_sockets.values())
    before = len(ws.sent)
    server._broadcast({"type": "chat_message", "role": "assistant", "content": "owner turn"})
    assert len(ws.sent) == before
    # Even if something put it in `clients`, a broadcast still skips it.
    server.clients.add(ws)
    server._broadcast({"type": "chat_message", "role": "assistant", "content": "owner turn"})
    assert len(ws.sent) == before
    server.clients.discard(ws)
    await disconnect(ws)
    assert ws not in server._speaker_sockets.get(ws._speaker_key, set())


async def test_resolve_agent_and_lane_view_never_reach_the_owner(server, monkeypatch):
    ws = await connect(server, monkeypatch, lane="A")
    agent = await server.resolve_agent(ws)
    assert agent is not server.agent and agent._speaker_scoped is True
    view = server.lane_view(ws, agent)
    assert isinstance(view, _LaneServerView) and view.agent is agent   # not `server`
    await disconnect(ws)


async def test_instances_are_built_without_owner_wiring(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    assert server.built == [{"register_tools": False, "warm_owner_caches": False,
                             "approval_owner": ("u-member", "A")}]
    agent = server._speaker_agents[("u-member", "A")]
    assert agent._peer_agents == [] and agent._active_personality_id is None
    assert agent._user_id is None and agent._speaker_profile == ("", "")
    assert agent._fleet_identity == {"name": "Helper"}
    # Its session keys are registered as member keys in the global registry.
    assert server._registry._is_speaker_call(agent.session.id)
    assert not server._registry._is_speaker_call(server.agent.session.id)
    await disconnect(ws)


async def test_two_members_get_separate_sessions_and_streams(server, monkeypatch):
    ana = await connect(server, monkeypatch, sub="u-ana", name="Ana")
    bo = await connect(server, monkeypatch, sub="u-bo", name="Bo")
    a_agent = server._speaker_agents[("u-ana", "A")]
    b_agent = server._speaker_agents[("u-bo", "A")]
    assert a_agent is not b_agent and a_agent.session.id != b_agent.session.id
    assert a_agent.session.metadata["speaker_id"] == "u-ana"

    t1 = await chat(ana, "ANA-SECRET-QUESTION")
    t2 = await chat(bo, "BO-QUESTION")
    await settle(server, ana, t1)
    await settle(server, bo, t2)
    assert "ANA-SECRET-QUESTION" not in "".join(bo.sent)
    assert "BO-QUESTION" not in "".join(ana.sent)
    assert ana.turn_ends(t1) == 1 and bo.turn_ends(t2) == 1
    await disconnect(ana)
    await disconnect(bo)


async def test_member_session_mapping_survives_a_restart(server, sm):
    p1 = Principal("u-ana", "Ana", "Olga", "A", "process:helper:0123456789abcdef")
    p2 = Principal("u-bo", "Bo", "Olga", "A", "process:helper:0123456789abcdef")
    s1 = await server._speaker_session(p1)
    s2 = await server._speaker_session(p2)
    s1b = await server._speaker_session(Principal("u-ana", "Ana", "Olga", "B", ""))
    assert len({s1.id, s2.id, s1b.id}) == 3
    assert await sm.get_app_state("speaker_session:u-ana|A") == s1.id

    restarted = WebServer.__new__(WebServer)
    restarted._init_speaker_state()
    assert (await restarted._speaker_session(p1)).id == s1.id
    assert (await restarted._speaker_session(p2)).id == s2.id


async def test_a_mapping_to_someone_elses_session_is_replaced(server, sm):
    p = Principal("u-ana", "Ana", "Olga", "A", "")
    owner = server.agent.session
    await sm.set_app_state("speaker_session:u-ana|A", owner.id)
    session = await server._speaker_session(p)
    assert session.id != owner.id and session.metadata["speaker_id"] == "u-ana"


# ── welcome / replay / ordering ──────────────────────────────────────


async def test_welcome_ack_and_member_only_replay(server, monkeypatch, sm):
    p = Principal("u-member", "Ana", "Olga", "A", "")
    mine = await server._speaker_session(p)
    mine.add_message("user", "MEMBER EARLIER QUESTION")
    mine.add_message("assistant", "MEMBER EARLIER ANSWER")
    await sm.save_session(mine)

    ws = await connect(server, monkeypatch)
    frames = ws.frames()
    assert [f["type"] for f in frames[:3]] == ["welcome", "replay_batch", "replay_done"]
    welcome = frames[0]
    assert welcome["speaker_ack"] == speaker_ack_for(ws.header)
    assert welcome["models"] == [] and welcome["personalities"] == []
    assert welcome["commands"] == speaker_commands()
    assert welcome["is_public"] is False
    assert welcome["speaker"] == {"id": "u-member", "name": "Ana",
                                  "owner_name": "Olga", "lane": "A"}
    assert welcome["session"]["id"] == mine.id
    # A2: the member's own allowlist (process agent: + their file tools) ∩
    # what is registered — never the owner's roster.
    registered = {"shell", "history", "read", *SPEAKER_TOOL_ALLOWLIST}
    p_ws = ws._speaker_principal
    assert set(welcome["session"]["tools"]) == speaker.allowed_tools(p_ws) & registered
    assert "shell" not in welcome["session"]["tools"]
    assert "history" not in welcome["session"]["tools"]
    assert welcome["session"]["skills"] == []
    replayed = json.dumps(frames[1])
    assert "MEMBER EARLIER QUESTION" in replayed
    assert "OWNER PRIVATE MESSAGE" not in "".join(ws.sent)
    await disconnect(ws)


async def test_empty_session_still_gets_replay_done(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    assert [f["type"] for f in ws.frames()] == ["welcome", "replay_done"]
    await disconnect(ws)


async def test_speaker_commands_are_only_the_allowlist():
    cmds = [c["command"] for c in speaker_commands()]
    assert "/help" in cmds and "/new [name]" in cmds and "/session" in cmds
    assert "/session rename <name>" in cmds
    for refused in ("/sessions", "/nuke", "/session switch <id|name|#N>", "/session list",
                    "/session export [chat|monitor|all]", "/session model", "/cron list",
                    "/config", "/models", "/planning on|off"):
        assert refused not in cmds


async def test_socket_is_registered_only_after_replay_done(server, monkeypatch, sm):
    """A second tab: a frame fanned out to the member while the new socket is
    still being welcomed must not reach it (FD would drop it pre-welcome)."""
    tab1 = await connect(server, monkeypatch)
    key = tab1._speaker_key
    real_list = sm.list_playbooks
    fired = []

    async def _during_welcome(*a, **k):
        if not fired:
            fired.append(1)
            server._speaker_send(key)({"type": "intruder"})
        return await real_list(*a, **k)

    monkeypatch.setattr(sm, "list_playbooks", _during_welcome)
    tab2 = await connect(server, monkeypatch)
    assert fired
    assert tab1.of_type("intruder") and not tab2.of_type("intruder")
    types_seen = [f["type"] for f in tab2.frames()]
    assert types_seen[0] == "welcome" and types_seen[-1] == "replay_done"
    # …and once registered, live frames do reach it.
    server._speaker_send(key)({"type": "after"})
    assert tab2.of_type("after")
    await disconnect(tab1)
    await disconnect(tab2)


# ── frame allowlist ──────────────────────────────────────────────────


@pytest.mark.parametrize("frame", [
    {"type": "run_tool", "tool": "shell", "args": {"command": "id"}, "req_id": "1"},
    {"type": "peer_agents", "agents": [{"name": "x"}], "fd_url": "http://evil"},
    {"type": "set_model", "selector": "gpt-x"},
    {"type": "set_personality", "personality_id": "boss"},
    {"type": "notification", "content": "do it", "trigger_response": True},
    {"type": "command", "command": "/nuke"},
    {"type": "set_force_script", "enabled": True},
    {"type": "set_byok", "provider": "x", "model": "y", "api_key": "z"},
    {"type": "something_new"},
    {"type": None},
])
async def test_owner_only_frames_are_refused(server, monkeypatch, frame):
    ws = await connect(server, monkeypatch)
    before = len(ws.sent)
    ws.feed(frame)
    await wait_for(lambda: len(ws.sent) > before)
    await asyncio.sleep(0.02)
    new = ws.frames()[before:]
    assert new == [{"type": "error", "code": "not_allowed", "message": NOT_ALLOWED_MESSAGE}]
    main = server.agent
    main._execute_tool_with_guard.assert_not_called()
    main.set_session_model.assert_not_called()
    assert main._force_script_mode is False and main._active_personality_id is None
    assert main._peer_agents == ["owner-peer"] and main._fd_url == "http://localhost:25080"
    assert server._inbound_queue.empty()
    assert server._speaker_agents[ws._speaker_key]._peer_agents == []
    await disconnect(ws)


async def test_non_object_frames_are_refused(server):
    ws = FakeWS()
    ws._speaker_key = ("u-member", "A")
    await handle_ws_message(server, ws, ["chat"])
    assert ws.frames() == [{"type": "error", "code": "invalid", "message": "Invalid frame"}]


async def test_fd_speaker_context_is_accepted_once(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    ws.feed({"type": "fd_speaker_context", "profile_full": "F" * 30_000, "profile_compact": "C"})
    await wait_for(lambda: agent._speaker_profile != ("", ""))
    assert agent._speaker_profile == ("F" * 20_000, "C")
    assert "system_prompt.md" not in agent.instructions._cache
    assert "micro_system_prompt.md" not in agent.instructions._cache
    before = len(ws.sent)
    ws.feed({"type": "fd_speaker_context", "profile_full": "EVIL", "profile_compact": "EVIL"})
    await wait_for(lambda: len(ws.sent) > before)
    assert ws.frames()[-1]["code"] == "not_allowed"
    assert agent._speaker_profile == ("F" * 20_000, "C")
    await disconnect(ws)


async def test_fd_speaker_context_after_a_chat_is_refused(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    tid = await chat(ws, "hello")
    await settle(server, ws, tid)
    before = len(ws.sent)
    ws.feed({"type": "fd_speaker_context", "profile_full": "LATE", "profile_compact": ""})
    await wait_for(lambda: len(ws.sent) > before)
    assert ws.frames()[-1]["code"] == "not_allowed"
    assert server._speaker_agents[ws._speaker_key]._speaker_profile == ("", "")
    await disconnect(ws)


async def test_approval_response_only_resolves_the_members_own_requests(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    mine, theirs = asyncio.Event(), asyncio.Event()
    server._pending_playbook_approvals = {"r-mine": (mine, [True]), "r-theirs": (theirs, [True])}
    server._speaker_approval_ids = {"r-mine": ws._speaker_key, "r-theirs": ("u-other", "A")}
    ws.feed({"type": "approval_response", "id": "r-theirs", "approved": False})
    ws.feed({"type": "approval_response", "id": "r-owner", "approved": False})
    ws.feed({"type": "approval_response", "id": "r-mine", "approved": False})
    await wait_for(mine.is_set)
    assert not theirs.is_set()
    assert server._pending_playbook_approvals["r-mine"][1] == [False]
    await disconnect(ws)


async def test_set_playbook_reports_to_the_member_only(server, monkeypatch):
    owner_ws = FakeWS()
    server.clients.add(owner_ws)
    ws = await connect(server, monkeypatch)
    ws.feed({"type": "set_playbook", "playbook_id": "__none__"})
    await wait_for(lambda: ws.of_type("command_result"))
    agent = server._speaker_agents[ws._speaker_key]
    assert agent._playbook_override == "__none__"
    assert ws.of_type("session_info") and owner_ws.sent == []
    assert getattr(server.agent, "_playbook_override", None) is None
    await disconnect(ws)


async def test_cancel_reaches_only_the_members_instance(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    ws.feed({"type": "cancel"})
    agent = server._speaker_agents[ws._speaker_key]
    await wait_for(agent.cancel_event.is_set)
    assert not server.agent.cancel_event.is_set()
    await disconnect(ws)


# ── slash commands ───────────────────────────────────────────────────


@pytest.mark.parametrize("cmd", [
    "/sessions", "/session switch x", "/session list", "/session load x",
    "/session export", "/session model gpt", "/session protect on", "/session new x",
    "/nuke", "/model", "/models", "/planning on", "/config", "/cron list", "/todo",
    "/orchestrate do things", "/basna x", "/skill y", "/approve user telegram t",
])
async def test_refused_slash_commands(server, monkeypatch, cmd):
    ws = await connect(server, monkeypatch)
    tid = await chat(ws, cmd)
    await settle(server, ws, tid)
    results = ws.of_type("command_result")
    assert results == [{"type": "command_result", "command": cmd, "content": NOT_ALLOWED_MESSAGE}]
    assert ws.turn_ends(tid) == 1
    assert server._speaker_agents[ws._speaker_key].completed == []
    await disconnect(ws)


async def test_help_lists_only_member_commands(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    tid = await chat(ws, "/help")
    await settle(server, ws, tid)
    help_text = ws.of_type("command_result")[0]["content"]
    assert "/clear" in help_text and "/nuke" not in help_text and "/sessions" not in help_text
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_new_rebinds_the_member_and_never_moves_last_active(server, monkeypatch, sm):
    spy = AsyncMock(return_value=True)
    monkeypatch.setattr(sm, "set_last_active_session", spy)
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    agent.session.add_message("user", "earlier chat")       # not empty → a new session
    old = agent.session.id
    tid = await chat(ws, "/new fresh start")
    await settle(server, ws, tid)
    assert agent.session.id != old
    assert agent.session.metadata["speaker_id"] == "u-member"
    assert await sm.get_app_state("speaker_session:u-member|A") == agent.session.id
    assert server._registry._is_speaker_call(agent.session.id)
    spy.assert_not_called()
    assert server.agent.session.name == "owner-main"
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_lane_a_member_clear_clears_their_session_not_the_owners(server, monkeypatch, sm):
    ws = await connect(server, monkeypatch, lane="A")
    agent = server._speaker_agents[ws._speaker_key]
    agent.session.add_message("user", "member chatter")
    tid = await chat(ws, "/clear")
    await settle(server, ws, tid)
    assert agent.session.messages == []
    assert agent.session.metadata["speaker_id"] == "u-member"       # mapping kept
    assert agent.session.metadata["speaker_lane"] == "A"
    owner = await sm.load_session(server.agent.session.id)
    assert any("OWNER PRIVATE MESSAGE" in m["content"] for m in owner.messages)
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_a_lane_view_replies_through_the_real_per_socket_send(server, sm):
    """The view's lane sender used to shadow `server._send(ws, msg)`, so every
    slash command on lanes B/C (and every member command) died at its reply."""
    from captain_claw.web.slash_commands import handle_command

    lane_ws = FakeWS()
    lane_ws._lane = "B"
    lane_agent = FakeAgent(await sm.create_session(name="lane-B"), server._registry, sm)
    lane_agent.session.add_message("user", "lane b question")
    view = server.lane_view(lane_ws, lane_agent)
    assert isinstance(view, _LaneServerView)
    await handle_command(view, lane_ws, "/history")
    result = lane_ws.of_type("command_result")
    assert result and "lane b question" in result[0]["content"]


async def test_owner_cannot_switch_into_a_member_session(server, sm):
    from captain_claw.web.slash_commands import handle_session_subcommand

    member = await server._speaker_session(Principal("u-ana", "Ana", "Olga", "A", ""))
    owner_session = server.agent.session
    out = await handle_session_subcommand(server, f"switch {member.id}")
    assert out == "That's a member's private conversation on this shared agent."
    assert server.agent.session is owner_session


async def test_member_cross_session_reference_is_unresolved():
    from captain_claw.agent_context_mixin import AgentContextMixin

    class _SM:
        async def select_session(self, sel):
            return types.SimpleNamespace(
                id="owner-sid", name="Owner chat", metadata={},
                messages=[{"role": "assistant", "content": "OWNER SECRET"}],
            )

    agent = types.SimpleNamespace(
        session=types.SimpleNamespace(id="mine"), session_manager=_SM(), memory=None,
        _speaker_scoped=True,
        _speaker_principal=Principal("u-ana", "Ana", "Olga", "A", ""),
        _extract_session_references=lambda q: ["#1"],
    )
    note = await AgentContextMixin._resolve_cross_session_context(agent, "see session #1")
    assert note is None or "OWNER SECRET" not in note


# ── exactly one turn_end ─────────────────────────────────────────────


async def test_turn_end_once_for_a_normal_turn(server, monkeypatch):
    propose = AsyncMock()
    monkeypatch.setattr("captain_claw.intentions_generator.maybe_auto_propose", propose)
    flows = AsyncMock(side_effect=AssertionError("flows never run for members"))
    monkeypatch.setattr("captain_claw.web.chat_handler._maybe_run_flow", flows)
    owner_ws = FakeWS()
    server.clients.add(owner_ws)
    ws = await connect(server, monkeypatch)
    tid = await chat(ws, "hello there")
    await settle(server, ws, tid)
    assert ws.turn_ends(tid) == 1
    replies = [f for f in ws.of_type("chat_message") if f.get("role") == "assistant"]
    assert replies and replies[0]["content"] == "reply to hello there"
    assert ws.of_type("session_info")
    assert owner_ws.sent == []                       # nothing reached the owner
    propose.assert_not_called()
    flows.assert_not_called()
    agent = server._speaker_agents[ws._speaker_key]
    assert agent._lane_busy is False
    assert speaker.current() is None                  # the binding didn't leak
    await disconnect(ws)


async def test_turn_end_once_when_busy(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    release = asyncio.Event()

    async def _slow(a, content):
        await release.wait()
        return "done"

    agent.complete_impl = _slow
    try:
        t1 = await chat(ws, "first")
        await wait_for(lambda: agent.completed)
        t2 = await chat(ws, "second")
        await wait_for(lambda: ws.turn_ends(t2) == 1)
        busy = [f for f in ws.of_type("error") if f.get("code") == "busy"]
        assert busy and busy[0]["message"] == "Your previous message is still being answered."
    finally:
        release.set()
    await settle(server, ws, t1)
    assert ws.turn_ends(t1) == 1 and ws.turn_ends(t2) == 1
    assert agent.completed == ["first"]
    await disconnect(ws)


async def test_back_to_back_frames_cannot_start_two_turns(server, monkeypatch):
    """The busy claim is synchronous: no window between check and launch."""
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    release = asyncio.Event()

    async def _slow(a, content):
        await release.wait()
        return "done"

    agent.complete_impl = _slow
    t1, t2 = secrets.token_hex(8), secrets.token_hex(8)
    try:
        await speaker_gate(server, ws, {"type": "chat", "content": "one", "_fd_turn": t1,
                                        "no_next_steps": True})
        await speaker_gate(server, ws, {"type": "chat", "content": "two", "_fd_turn": t2,
                                        "no_next_steps": True})
        assert ws.turn_ends(t2) == 1
    finally:
        release.set()
    await settle(server, ws, t1)
    assert agent.completed == ["one"] and ws.turn_ends(t1) == 1
    await disconnect(ws)


@pytest.mark.parametrize("content", ["", "   "])
async def test_turn_end_once_for_empty_content(server, monkeypatch, content):
    ws = await connect(server, monkeypatch)
    tid = await chat(ws, content)
    await settle(server, ws, tid)
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_turn_end_once_for_oversized_content(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    tid = await chat(ws, "x" * (speaker.MAX_CHAT_CONTENT + 1))
    await settle(server, ws, tid)
    assert ws.turn_ends(tid) == 1
    assert ws.of_type("error")[-1]["code"] == "invalid"
    assert server._speaker_agents[ws._speaker_key].completed == []
    await disconnect(ws)


async def test_turn_end_once_for_an_allowed_slash(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    tid = await chat(ws, "/history")
    await settle(server, ws, tid)
    assert ws.turn_ends(tid) == 1 and ws.of_type("command_result")
    await disconnect(ws)


@pytest.mark.parametrize("error", ["instance", "capacity"])
async def test_turn_end_once_when_the_instance_fails(server, monkeypatch, error):
    ws = await connect(server, monkeypatch)

    async def _fail(p):
        if error == "capacity":
            raise SpeakerCapacityError("full")
        raise RuntimeError("boom /owner/secret/path")

    monkeypatch.setattr(server, "_get_speaker_agent", _fail)
    tid = await chat(ws, "hello")
    await settle(server, ws, tid)
    assert ws.turn_ends(tid) == 1
    err = ws.of_type("error")[-1]
    if error == "capacity":
        assert err["code"] == "capacity"
    assert "/owner/secret/path" not in err["message"]
    await disconnect(ws)


async def test_turn_end_once_when_the_turn_raises(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]

    async def _raise(a, content):
        raise RuntimeError("model exploded")

    agent.complete_impl = _raise
    tid = await chat(ws, "hello")
    await settle(server, ws, tid)
    assert ws.turn_ends(tid) == 1
    errors = ws.of_type("error")
    assert errors and ws.frames().index(errors[-1]) < [
        i for i, f in enumerate(ws.frames()) if f.get("turn_end") == tid][0]
    assert agent._lane_busy is False
    await disconnect(ws)


async def test_turn_end_once_when_cancelled_by_the_member(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]

    async def _until_cancel(a, content):
        await a.cancel_event.wait()
        return "stopped"

    agent.complete_impl = _until_cancel
    tid = await chat(ws, "long task")
    try:
        await wait_for(lambda: agent.completed)
        ws.feed({"type": "cancel"})
        await settle(server, ws, tid)
    finally:
        agent.cancel_event.set()
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_turn_end_once_when_the_task_is_cancelled(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    hang = asyncio.Event()

    async def _forever(a, content):
        await hang.wait()

    agent.complete_impl = _forever
    tid = await chat(ws, "long task")
    try:
        await wait_for(lambda: agent.completed)
        agent._public_task.cancel()
        await settle(server, ws, tid)
    finally:
        hang.set()
    assert ws.turn_ends(tid) == 1
    assert agent._lane_busy is False
    await disconnect(ws)


async def test_member_turn_binds_the_principal_for_tools(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    seen = []

    async def _probe(a, content):
        seen.append(speaker.current())
        return "ok"

    agent.complete_impl = _probe
    tid = await chat(ws, "hi")
    await settle(server, ws, tid)
    assert seen and seen[0] is not None and seen[0].speaker_id == "u-member"
    await disconnect(ws)


# ── eviction ─────────────────────────────────────────────────────────


def _parked(server, key, *, busy=False, socket=False, last_used=0.0):
    agent = types.SimpleNamespace(
        _lane_busy=busy, tools=server._registry, _speaker_registry_keys={f"sess-{key[0]}"},
    )
    server._registry.register_speaker_session(f"sess-{key[0]}")
    server._speaker_agents[key] = agent
    server._speaker_last_used[key] = last_used
    if socket:
        server._speaker_sockets[key] = {FakeWS()}
    return agent


async def test_eviction_never_drops_busy_or_attached_instances(server, monkeypatch):
    monkeypatch.setattr(speaker, "SPEAKER_MAX_INSTANCES", 2)
    _parked(server, ("u-busy", "A"), busy=True)
    _parked(server, ("u-attached", "A"), socket=True)
    server._evict_speaker_agents()
    assert set(server._speaker_agents) == {("u-busy", "A"), ("u-attached", "A")}
    with pytest.raises(SpeakerCapacityError):
        await server._get_speaker_agent(Principal("u-new", "New", "Olga", "A", ""))


async def test_lru_eviction_at_the_cap(server, monkeypatch):
    monkeypatch.setattr(speaker, "SPEAKER_MAX_INSTANCES", 3)
    now = time.monotonic()
    _parked(server, ("u-old", "A"), last_used=now - 30)
    _parked(server, ("u-mid", "A"), last_used=now - 20)
    _parked(server, ("u-new", "A"), last_used=now - 10)
    agent = await server._get_speaker_agent(Principal("u-fresh", "F", "Olga", "A", ""))
    assert ("u-old", "A") not in server._speaker_agents
    assert {("u-mid", "A"), ("u-new", "A"), ("u-fresh", "A")} == set(server._speaker_agents)
    assert not server._registry._is_speaker_call("sess-u-old")     # keys unregistered
    assert server._registry._is_speaker_call("sess-u-mid")
    assert agent._speaker_scoped is True


async def test_idle_instances_are_evicted_below_the_cap(server):
    _parked(server, ("u-idle", "A"), last_used=time.monotonic() - speaker.SPEAKER_IDLE_EVICT_S - 5)
    _parked(server, ("u-recent", "A"), last_used=time.monotonic())
    server._evict_speaker_agents()
    assert set(server._speaker_agents) == {("u-recent", "A")}


async def test_closed_sockets_do_not_pin_an_instance(server, monkeypatch):
    monkeypatch.setattr(speaker, "SPEAKER_MAX_INSTANCES", 1)
    _parked(server, ("u-gone", "A"), socket=True)
    next(iter(server._speaker_sockets[("u-gone", "A")])).closed = True
    await server._get_speaker_agent(Principal("u-next", "N", "Olga", "A", ""))
    assert set(server._speaker_agents) == {("u-next", "A")}


# ── prompt and owner data (real Agent) ───────────────────────────────


def _real_agent(monkeypatch, tmp_path):
    from pathlib import Path

    import captain_claw.agent_context_mixin as acm
    from captain_claw.agent import Agent
    from captain_claw.instructions import InstructionLoader
    from captain_claw.llm import LLMProvider, LLMResponse
    from captain_claw.session import Session

    class P(LLMProvider):
        async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
            return LLMResponse(content="ok")

        async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
            if False:
                yield ""

        def count_tokens(self, text):
            return len(text.split()) or 1

    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    agent = Agent(provider=P())
    agent.session = Session(id="s-member", name="spk-ana-A")
    agent.instructions = InstructionLoader(
        base_dir=Path(acm.__file__).resolve().parent / "instructions",
        personal_dir=tmp_path / "personal",
    )
    agent._build_skills_system_prompt_section = lambda: ""
    agent._build_playbook_context_note_sync = lambda q: ""
    return agent


@pytest.fixture
def owner_profile(monkeypatch, tmp_path):
    import captain_claw.tenant_context as tc

    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAW_VFS_PROJECT", raising=False)
    monkeypatch.delenv("CLAW_BEING_WORKER", raising=False)
    tc._cache.clear()
    (home / ".captain-claw" / tc.FULL_FILENAME).write_text("## Your owner\nOWNER-OLGA-PROFILE")
    (home / ".captain-claw" / tc.COMPACT_FILENAME).write_text("OWNER-OLGA-COMPACT")
    docs = tmp_path / "owner-private-docs"
    docs.mkdir()
    (docs / "salaries.xlsx").write_text("x")
    cfg = get_config()
    monkeypatch.setattr(cfg.tools.read, "extra_dirs", [str(docs)])
    yield docs
    tc._cache.clear()


async def test_member_prompt_has_their_profile_and_the_mode_note(server, monkeypatch, tmp_path,
                                                                 owner_profile):
    agent = _real_agent(monkeypatch, tmp_path)
    agent.tools.register(_Rec("browser"))          # a tool the member can't use
    owner_prompt = agent._build_system_prompt()
    assert "OWNER-OLGA-PROFILE" in owner_prompt and str(owner_profile) in owner_prompt
    assert "MANDATORY browser policy" in owner_prompt

    p = Principal("u-member", "Ana", "Olga", "A", "")
    agent._speaker_scoped = True
    agent._speaker_principal = p
    agent._speaker_profile = ("", "")
    server._speaker_agents[("u-member", "A")] = agent
    ws = FakeWS()
    ws._speaker_key = ("u-member", "A")
    ws._speaker_principal = p
    await speaker_gate(server, ws, {
        "type": "fd_speaker_context",
        "profile_full": "## About the member\nANA-FROM-ACME", "profile_compact": "ANA-COMPACT",
    })
    prompt = agent._build_system_prompt()
    assert "ANA-FROM-ACME" in prompt and SPEAKER_MODE_NOTE in prompt
    assert prompt.index("ANA-FROM-ACME") < prompt.index("<!-- CACHE_SPLIT -->")
    assert "OWNER-OLGA-PROFILE" not in prompt and "OWNER-OLGA-COMPACT" not in prompt
    assert str(owner_profile) not in prompt and "salaries.xlsx" not in prompt
    assert "MANDATORY browser policy" not in prompt


async def test_member_prompt_without_a_profile_still_has_the_note(monkeypatch, tmp_path,
                                                                  owner_profile):
    agent = _real_agent(monkeypatch, tmp_path)
    agent._speaker_scoped = True
    agent._speaker_profile = ("", "")
    prompt = agent._build_system_prompt()
    assert SPEAKER_MODE_NOTE in prompt and "OWNER-OLGA-PROFILE" not in prompt


def test_member_messages_skip_owner_notes(monkeypatch, tmp_path):
    agent = _real_agent(monkeypatch, tmp_path)
    markers = {
        "_build_todo_context_note": "OWNER-TODO",
        "_build_contacts_context_note": "OWNER-CONTACTS",
        "_build_scripts_context_note": "OWNER-SCRIPTS",
        "_build_apis_context_note": "OWNER-APIS",
        "_build_datastore_context_note": "OWNER-DATASTORE",
        "_build_intentions_context_note": "OWNER-INTENTIONS",
        "_build_briefing_context_note": "OWNER-BRIEFING",
        "_build_insights_context_note": "COMMONS-INSIGHTS",
        "_build_nervous_system_context_note": "COMMONS-INTUITIONS",
    }
    for attr, text in markers.items():
        setattr(agent, attr, (lambda t: (lambda *a, **k: t))(text))

    owner_text = " ".join(str(m.content) for m in agent._build_messages(query="hello"))
    for text in markers.values():
        assert text in owner_text

    agent._speaker_scoped = True
    agent._turn_system_prompt = None
    member_text = " ".join(str(m.content) for m in agent._build_messages(query="hello"))
    for attr, text in markers.items():
        if text.startswith("OWNER-"):
            assert text not in member_text, attr
        else:
            assert text in member_text, attr


async def test_member_turn_skips_owner_cache_refresh_and_google(monkeypatch):
    """agent_orchestration_mixin: owner caches + Google probe are owner-only.

    A2: a member's per-iteration Google refresh is their own status from
    Flight Deck (speaker_status), in the `is True` branch; the owner's
    is_connected() sits in its `else`.
    """
    import inspect

    from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

    src = inspect.getsource(AgentOrchestrationMixin)
    gate = 'if getattr(self, "_speaker_scoped", False) is not True:'
    block = src[src.index(gate):]
    assert block.index("_refresh_todo_context_cache") < block.index("_refresh_datastore_context_cache")
    google = src.index("GoogleOAuthManager(self.session_manager).is_connected()")
    member_gate = 'if getattr(self, "_speaker_scoped", False) is True:'
    member = src.rfind(member_gate, 0, google)
    assert member > src.rfind("def ", 0, google)
    status = src.index("GoogleOAuthManager(self.session_manager).speaker_status()", member)
    assert member < status < google and "else:" in src[status:google]


# ── a member session can never become the owner's lane / default session ──


@pytest.mark.parametrize("name", ["lane-B", "LANE-c", "default"])
async def test_member_cannot_rename_their_session_to_a_reserved_name(server, monkeypatch, name):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    before = agent.session.name
    tid = await chat(ws, f"/session rename {name}")
    await settle(server, ws, tid)
    assert agent.session.name == before
    assert "reserved" in ws.of_type("command_result")[-1]["content"]
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


@pytest.mark.parametrize("name", ["lane-B", "default"])
async def test_member_new_never_takes_a_reserved_name(server, monkeypatch, name):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    agent.session.add_message("user", "earlier chat")
    old = agent.session.id
    tid = await chat(ws, f"/new {name}")
    await settle(server, ws, tid)
    assert agent.session.id != old
    assert agent.session.name == "spk-ana-A"
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_member_can_still_rename_to_an_ordinary_name(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    tid = await chat(ws, "/session rename trip planning")
    await settle(server, ws, tid)
    assert agent.session.name == "trip planning"
    await disconnect(ws)


async def test_lane_session_never_adopts_a_member_session(server, sm):
    """Even a member session already named `lane-B` (e.g. renamed before this
    guard) is never picked up as the owner's lane B — that lane agent would
    write the owner's turns into the member's replayed transcript."""
    member = await server._speaker_session(Principal("u-ana", "Ana", "Olga", "B", ""))
    member.name = "lane-B"
    member.add_message("user", "MEMBER PRIVATE")
    await sm.save_session(member)
    lane = await server._lane_session("B")
    assert lane.id != member.id
    assert not (lane.metadata or {}).get("speaker_id")
    assert lane.name == "lane-B"
    # And an owner lane session that already exists is still rejoined.
    assert (await server._lane_session("B")).id == lane.id


def test_member_prompt_has_the_clock_but_not_the_owners_machine(monkeypatch, tmp_path):
    """System info for a member is the date/time only — never the host's
    name, local/public IP, memory, disk, load or uptime."""
    import captain_claw.system_info as si

    monkeypatch.setattr(si.socket, "gethostname", lambda: "OWNER-HOSTNAME")
    monkeypatch.setattr(si, "_get_local_ip", lambda: "10.9.8.7")
    monkeypatch.setattr(si, "_get_public_ip", lambda: "203.0.113.77")
    agent = _real_agent(monkeypatch, tmp_path)
    # The clock and host lines ride in the per-turn context block now.
    owner_env = agent._build_env_now_text()
    assert "OWNER-HOSTNAME" in owner_env and "203.0.113.77" in owner_env
    assert "OWNER-HOSTNAME" not in agent._build_system_prompt()

    agent._speaker_scoped = True
    agent._speaker_profile = ("", "")
    member_env = agent._build_env_now_text()
    member_prompt = agent._build_system_prompt()
    for leak in ("OWNER-HOSTNAME", "10.9.8.7", "203.0.113.77", "Uptime", "Disk free"):
        assert leak not in member_env, leak
        assert leak not in member_prompt, leak
    assert "UTC" in member_env and SPEAKER_MODE_NOTE in member_prompt


# ── error frames on the member wire ──────────────────────────────────


async def test_member_turn_failure_never_shows_the_exception(server, monkeypatch):
    """The owner's provider errors (base URLs, key fragments) never reach a
    member: a fixed message with a contract code, then the one turn_end."""
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]

    async def _raise(a, content):
        raise RuntimeError("provider exploded at http://10.0.0.5:11434 key sk-abc")

    agent.complete_impl = _raise
    tid = await chat(ws, "hello")
    await settle(server, ws, tid)
    assert ws.of_type("error") == [{
        "type": "error", "code": "invalid", "message": speaker.TURN_FAILED_MESSAGE,
    }]
    assert "10.0.0.5" not in "".join(ws.sent) and "sk-abc" not in "".join(ws.sent)
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_owner_turn_failure_keeps_the_detail(server):
    """Owner / lane sockets still get the exception text."""
    from captain_claw.web.chat_handler import _run_agent

    ws = FakeWS()
    agent = FakeAgent(server.agent.session, server._registry, server.agent.session_manager)

    async def _raise(a, content):
        raise RuntimeError("provider exploded at http://10.0.0.5:11434")

    agent.complete_impl = _raise
    await _run_agent(server, ws, agent, "hello", no_flow=True, no_broadcast=True)
    assert ws.of_type("error") == [{
        "type": "error", "message": "Error: provider exploded at http://10.0.0.5:11434",
    }]


async def test_every_member_error_frame_carries_a_contract_code(server, monkeypatch, sm):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]

    # Invalid JSON, a non-object frame, a refused frame, an oversized chat.
    ws._inbox.put_nowait(types.SimpleNamespace(type=aiohttp.web.WSMsgType.TEXT, data="{nope"))
    ws.feed(["chat"])
    ws.feed({"type": "run_tool", "tool": "shell"})
    t_long = await chat(ws, "x" * (speaker.MAX_CHAT_CONTENT + 1))
    await wait_for(lambda: ws.turn_ends(t_long) == 1)

    # Busy, then btw past its cap, while a turn runs.
    release = asyncio.Event()

    async def _slow(a, content):
        await release.wait()
        return "done"

    agent.complete_impl = _slow
    t1 = await chat(ws, "first")
    try:
        await wait_for(lambda: agent.completed)
        t2 = await chat(ws, "second")
        await wait_for(lambda: ws.turn_ends(t2) == 1)
        for i in range(speaker.MAX_BTW_INSTRUCTIONS + 1):
            ws.feed({"type": "btw", "content": f"note {i}"})
        ws.feed({"type": "btw", "content": "y" * (speaker.MAX_CHAT_CONTENT + 1)})
        await wait_for(lambda: sum(1 for f in ws.frames() if f.get("command") == "/btw")
                       == speaker.MAX_BTW_INSTRUCTIONS)
        await asyncio.sleep(0.05)
    finally:
        release.set()
    await settle(server, ws, t1)

    # The session-settings lock, then a failing turn.
    agent.session.metadata["session_settings_locked"] = True
    ws.feed({"type": "session_settings", "session_name": "mine now"})
    async def _raise(a, content):
        raise RuntimeError("boom")

    agent.complete_impl = _raise
    t3 = await chat(ws, "again")
    await settle(server, ws, t3)

    errors = ws.of_type("error")
    codes = [e.get("code") for e in errors]
    assert len(errors) >= 9, errors
    assert all(c in speaker.ERROR_CODES for c in codes), errors
    assert {"invalid", "busy", "not_allowed"} <= set(codes)
    await disconnect(ws)


# ── btw and session settings are bounded for a member ────────────────


async def test_btw_on_an_idle_member_instance_is_dropped(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    for _ in range(5):
        ws.feed({"type": "btw", "content": "x" * 1000})
    await wait_for(lambda: len(ws.of_type("command_result")) >= 5)
    assert not getattr(agent, "_btw_instructions", None)
    assert all("Nothing is running" in f["content"] for f in ws.of_type("command_result"))
    await disconnect(ws)


async def test_btw_during_a_member_turn_is_capped_and_cleared(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    release = asyncio.Event()
    seen: list[list[str]] = []

    async def _slow(a, content):
        await release.wait()
        seen.append(list(getattr(a, "_btw_instructions", [])))
        return "done"

    agent.complete_impl = _slow
    tid = await chat(ws, "long task")
    try:
        await wait_for(lambda: agent.completed)
        ws.feed({"type": "btw", "content": "z" * (speaker.MAX_CHAT_CONTENT + 1)})
        await wait_for(lambda: ws.of_type("error"))
        assert ws.of_type("error")[0]["message"] == "Message is too long."
        for i in range(speaker.MAX_BTW_INSTRUCTIONS + 5):
            ws.feed({"type": "btw", "content": f"note {i}"})
        await wait_for(lambda: len(ws.of_type("error")) >= 6)
    finally:
        release.set()
    await settle(server, ws, tid)
    assert seen == [[f"note {i}" for i in range(speaker.MAX_BTW_INSTRUCTIONS)]]
    assert all(e["code"] == "invalid" for e in ws.of_type("error"))
    assert agent._btw_instructions == []
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)


async def test_member_session_settings_are_capped(server, monkeypatch, sm):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    ws.feed({"type": "session_settings", "session_name": "n" * 5000,
             "session_description": "d" * 50_000, "session_instructions": "i" * 90_000})
    await wait_for(lambda: ws.of_type("session_settings_saved"))
    meta = agent.session.metadata
    limits = speaker.SESSION_SETTINGS_LIMITS
    assert meta["session_display_name"] == "n" * limits["session_name"]
    assert meta["session_description"] == "d" * limits["session_description"]
    assert meta["session_instructions"] == "i" * limits["session_instructions"]
    stored = (await sm.load_session(agent.session.id)).metadata
    assert len(stored["session_instructions"]) == limits["session_instructions"]
    await disconnect(ws)


async def test_owner_session_settings_are_not_capped(server, sm):
    ws = FakeWS()
    await handle_ws_message(server, ws, {"type": "session_settings",
                                         "session_instructions": "i" * 9000})
    assert server.agent.session.metadata["session_instructions"] == "i" * 9000


# ── /new: no empty duplicates, a per-member cap, the owner's lanes kept ──


async def _member_rows(sm, speaker_id="u-member"):
    return [s for s in await sm.list_sessions(limit=500)
            if (s.metadata or {}).get("speaker_id") == speaker_id]


async def test_member_new_reuses_an_empty_session(server, monkeypatch, sm):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    old = agent.session.id
    for _ in range(3):
        tid = await chat(ws, "/new trip ideas")
        await settle(server, ws, tid)
        assert ws.turn_ends(tid) == 1
    assert agent.session.id == old and agent.session.name == "trip ideas"
    assert [s.id for s in await _member_rows(sm)] == [old]
    await disconnect(ws)


async def test_member_sessions_are_capped(server, monkeypatch, sm):
    monkeypatch.setattr(speaker, "MAX_SESSIONS_PER_SPEAKER", 3)
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]
    first = agent.session.id
    for _ in range(5):
        agent.session.add_message("user", "some chat")
        tid = await chat(ws, "/new")
        await settle(server, ws, tid)
        assert ws.turn_ends(tid) == 1
    assert len(await _member_rows(sm)) == 3
    errors = ws.of_type("error")
    assert len(errors) == 3 and all(e["code"] == "not_allowed" for e in errors)
    stuck = agent.session.id
    # Room again once a session is gone (e.g. the owner deleted it).
    await sm.delete_session(first)
    tid = await chat(ws, "/new")
    await settle(server, ws, tid)
    assert agent.session.id != stuck and len(await _member_rows(sm)) == 3
    await disconnect(ws)


async def test_the_session_cap_spans_a_members_lanes(server, monkeypatch, sm):
    monkeypatch.setattr(speaker, "MAX_SESSIONS_PER_SPEAKER", 2)
    a = await connect(server, monkeypatch, lane="A")
    b = await connect(server, monkeypatch, lane="B")
    assert len(await _member_rows(sm)) == 2          # one per lane, never refused
    agent = server._speaker_agents[a._speaker_key]
    agent.session.add_message("user", "some chat")
    tid = await chat(a, "/new")
    await settle(server, a, tid)
    assert a.of_type("error")[-1]["code"] == "not_allowed"
    await disconnect(a)
    await disconnect(b)


async def test_many_newer_sessions_never_lose_the_owners_lane(server, sm):
    lane = await sm.create_session(name="lane-B")
    for i in range(205):
        await sm.create_session(name=f"spk-m-{i}", metadata={"speaker_id": f"u-{i % 10}"})
    assert (await server._lane_session("B")).id == lane.id


async def test_member_command_errors_never_show_the_exception(server, monkeypatch):
    ws = await connect(server, monkeypatch)
    agent = server._speaker_agents[ws._speaker_key]

    async def _boom(**kw):
        raise RuntimeError("compaction failed at http://10.0.0.5:11434 key sk-abc")

    agent.compact_session = _boom
    tid = await chat(ws, "/compact")
    await settle(server, ws, tid)
    assert "10.0.0.5" not in "".join(ws.sent) and "sk-abc" not in "".join(ws.sent)
    assert ws.of_type("command_result")[-1]["content"] == "That command couldn't be completed — try again."
    await disconnect(ws)


async def test_owner_session_list_and_index_skip_member_sessions(server, sm):
    from captain_claw.web.slash_commands import handle_command, handle_session_subcommand

    member = await server._speaker_session(Principal("u-ana", "Ana", "Olga", "A", ""))
    member.add_message("user", "MEMBER PRIVATE")
    await sm.save_session(member)                     # the newest session overall
    other = await sm.create_session(name="owner-side")
    owner_session = server.agent.session
    server.agent.session = other
    server._session_info = lambda agent=None: {}
    server._broadcast = lambda msg: None
    server.agent._sync_runtime_flags_from_session = lambda: None
    out = await handle_session_subcommand(server, "list")
    assert "spk-" not in out and "owner-main" in out
    # `#2` is the owner's second-newest session, never the member's.
    assert (await handle_session_subcommand(server, "switch #2")).startswith("Switched")
    assert server.agent.session.id == owner_session.id
    ws = FakeWS()
    await handle_command(server, ws, "/sessions")
    assert "spk-" not in ws.of_type("command_result")[0]["content"]



@pytest.mark.parametrize("failure", ["instance", "no_agent"])
async def test_member_setup_errors_carry_a_contract_code(server, monkeypatch, failure):
    ws = await connect(server, monkeypatch)
    if failure == "instance":
        async def _fail(p):
            raise RuntimeError("boom /owner/secret/path")

        monkeypatch.setattr(server, "_get_speaker_agent", _fail)
    else:
        monkeypatch.setattr(server, "agent", None)
    tid = await chat(ws, "hello")
    await settle(server, ws, tid)
    errors = ws.of_type("error")
    assert errors and all(e.get("code") == "invalid" for e in errors), errors
    assert "/owner/secret/path" not in "".join(ws.sent)
    assert ws.turn_ends(tid) == 1
    await disconnect(ws)
