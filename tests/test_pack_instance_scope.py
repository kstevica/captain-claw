"""PR B (r3): which agent instances use context packs — and the prompt block.

The user's decision (contract b part 0 J1, part 2 §7): packs apply on EVERY
turn of an allowed agent instance — the owner's Flight Deck chats on any lane
AND the owner's channels and automations (WhatsApp/glasses pumps on lane A,
Telegram per-user agents, the API pool, cron, peer relays, slash and hotkey
turns) and every member instance. There is no per-turn origin check, no
socket marker and no lane rule: ``pack_access.packs_allowed(agent)`` is the
only gate. Public sessions / ``public_run``, BotPort dispatch agents, Iskra
bodies and ``CLAW_VFS_SCOPE`` processes never get packs.

Real ``Agent`` objects drive real turns here (a scripted provider asks for
``read vfs:@ana-notes/a.md`` / a deep-memory search); Flight Deck is a
patched ``FDClient.post``. HOME, FD_DATA_DIR, every config DB path and the
global session / topic managers point at tmp before anything is built.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import types
import uuid
from pathlib import Path

import httpx
import pytest

from captain_claw import pack_access, speaker
from captain_claw.config import get_config
from captain_claw.exceptions import ToolBlockedError
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.pack_access import PACK_RESOLVE_PATH, PACKS_NOT_HERE_MESSAGE
from captain_claw.speaker import SHARED_CONTEXT_MEMBER_NOTE, SPEAKER_MODE_NOTE_FULL, Principal

WEB_AUTH = "test-web-auth"
ALIAS = "ana-notes"
LABEL = "“Ana” (a member)"
SEARCH_PATH = "/fd/deep-memory/agent/search"
SPLIT = "<!-- CACHE_SPLIT -->"
PRINCIPAL = Principal("u-member", "Ana", "Olga", "A", "process:helper:0123456789abcdef")
SHARED_FULL = ("## Shared context on this agent\nPeople who use this agent shared this.\n\n"
               "### Shared profiles\nAbout “Ana” (a member), shared by them:\n"
               "> SHARED-FULL-MARKER")
SHARED_COMPACT = "## Shared context on this agent\nSHARED-COMPACT-MARKER"
TENANT_FULL = "## Your owner\nOWNER-PROFILE-MARKER"
TENANT_COMPACT = "OWNER-COMPACT-MARKER"

_DB_FIELDS = (
    ("memory", "path"), ("session", "path"), ("insights", "db_path"),
    ("conversation_topics", "db_path"), ("nervous_system", "db_path"),
    ("sister_session", "db_path"), ("cognitive_metrics", "db_path"),
    ("datastore", "path"), ("autonomous_work", "db_path"),
)
_ENV_CLEARED = (
    "CLAW_VFS_ROOT", "CLAW_VFS_PROJECT", "CLAW_VFS_SCOPE", "CLAW_WRITE_DIRECT",
    "FD_INTERNAL_URL", "FD_AGENT_SHARED_SECRET", "FD_AGENT_SLUG", "CLAW_AGENT_LABEL",
    "CLAW_VATRA_OWNER", "CLAW_BEING_WORKER", "CLAW_BEING_CAPS", "CLAW_CODE_AGENT",
)


# ── isolation and fakes ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    import captain_claw.conversation_topics as _ct
    import captain_claw.tenant_context as tc
    from captain_claw import session as _session

    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    fd_data = tmp_path / "fd-data"
    (fd_data / "vfs").mkdir(parents=True)
    monkeypatch.setenv("FD_DATA_DIR", str(fd_data))
    for var in _ENV_CLEARED:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLAW_VFS_USER", "owner")
    monkeypatch.setenv("FD_OWNER_ID", "owner")
    monkeypatch.setenv("FD_URL", "http://fd.test")
    cfg = get_config()
    for section, attr in _DB_FIELDS:
        monkeypatch.setattr(getattr(cfg, section), attr, str(home / ".captain-claw" / f"{section}.db"))
    monkeypatch.setattr(cfg.tools.read, "extra_dirs", [])
    monkeypatch.setattr(cfg.web, "auth_token", WEB_AUTH)
    monkeypatch.setattr(cfg.web, "public_run", "")
    monkeypatch.setattr(cfg.ui, "next_steps", False)
    monkeypatch.setattr(cfg.ui, "streaming", False)
    sm = _session.SessionManager(home / ".captain-claw" / "s.db")
    monkeypatch.setattr(_session, "_manager", sm)
    monkeypatch.setattr(_ct, "_MANAGER", None)
    monkeypatch.setattr(speaker, "_IN_FLIGHT", 0)
    monkeypatch.setattr(speaker, "_MAIN_LOOP", None)
    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    tc._cache.clear()
    yield types.SimpleNamespace(home=home, sm=sm, fd_data=fd_data.resolve())
    tc._cache.clear()


_DB_MODULES = ("nervous_system", "sister_session", "intentions", "cognitive_metrics", "insights",
               "datastore")


@pytest.fixture(autouse=True)
async def close_dbs(isolated, monkeypatch):
    """Every process-global DB manager starts empty (its DB under the tmp
    HOME) and is closed afterwards — an open aiosqlite worker thread would
    keep the test process alive."""
    import functools
    import importlib
    import threading

    import aiosqlite.core

    # A connection opened by a fire-and-forget post-turn task after the
    # managers below were closed must not keep the process alive either.
    monkeypatch.setattr(aiosqlite.core, "Thread", functools.partial(threading.Thread, daemon=True))
    mods = [importlib.import_module(f"captain_claw.{m}") for m in _DB_MODULES]
    for mod in mods:
        monkeypatch.setattr(mod, "_manager", None)
    yield
    # Let the turn's background tasks (metrics, caches) finish first.
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        _done, still = await asyncio.wait(pending, timeout=5)
        for task in still:
            task.cancel()
        if still:
            await asyncio.gather(*still, return_exceptions=True)
    for mod in mods:
        manager = getattr(mod, "_manager", None)
        if manager is not None and hasattr(manager, "close"):
            try:
                await manager.close()
            except Exception:
                pass
    from captain_claw import datastore

    for closer in ("close_vfs_datastore_managers", "close_session_datastore_managers"):
        try:
            await getattr(datastore, closer)()
        except Exception:
            pass
    try:
        await isolated.sm.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """No post-turn LLM jobs, no flows, no channel delivery, no skills scan."""
    from captain_claw.agent import Agent

    async def _noop(*a, **k):
        return None

    async def _no_name(*a, **k):
        return ""

    async def _delivered(*a, **k):
        return True

    for target in (
        "captain_claw.reflections.maybe_auto_reflect",
        "captain_claw.insights.maybe_extract_insights",
        "captain_claw.nervous_system.maybe_dream",
        "captain_claw.conversation_topics.maybe_classify_topics",
        "captain_claw.intentions_generator.maybe_auto_propose",
        "captain_claw.web.chat_handler._maybe_run_flow",
    ):
        monkeypatch.setattr(target, _noop)
    monkeypatch.setattr("captain_claw.web.chat_handler._generate_task_name", _no_name)
    monkeypatch.setattr("captain_claw.delivery.deliver_to_origin", _delivered)
    monkeypatch.setattr("captain_claw.web_server.fire_and_forget_send",
                        lambda ws, data: ws.sent.append(data))
    monkeypatch.setattr("captain_claw.web.chat_handler.fire_and_forget_send",
                        lambda ws, data: ws.sent.append(data))
    async def _no_network(self, request, *a, **k):
        raise httpx.ConnectError(f"no network in tests: {request.url.host}", request=request)

    # Flight Deck is the patched FDClient.post; anything else (Google token
    # fetches, peer probes) must never leave the process.
    monkeypatch.setattr(httpx.AsyncClient, "send", _no_network)
    monkeypatch.setattr(Agent, "_build_skills_system_prompt_section", lambda self, *a, **k: "")
    monkeypatch.setattr(Agent, "_build_playbook_context_note_sync", lambda self, *a, **k: "")

    def _few_tools(self):
        from captain_claw.tools.glob import GlobTool
        from captain_claw.tools.read import ReadTool
        from captain_claw.tools.typesense import TypesenseTool

        for tool in (ReadTool(), GlobTool(), TypesenseTool()):
            self.tools.register(tool)

    # Lane / Telegram / API-pool / public / BotPort agents call this; the real
    # one registers ~80 tools. The construction path is otherwise unchanged.
    monkeypatch.setattr(Agent, "_register_default_tools", _few_tools)


@pytest.fixture
def shared_files(isolated):
    import captain_claw.tenant_context as tc

    d = isolated.home / ".captain-claw"
    (d / tc.FULL_FILENAME).write_text(TENANT_FULL)
    (d / tc.COMPACT_FILENAME).write_text(TENANT_COMPACT)
    (d / tc.SHARED_FULL_FILENAME).write_text(SHARED_FULL)
    (d / tc.SHARED_COMPACT_FILENAME).write_text(SHARED_COMPACT)
    return d


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


@pytest.fixture
def fd(monkeypatch, isolated, shared_files):
    pack = isolated.fd_data / "vfs" / "ana" / "notes"
    pack.mkdir(parents=True)
    (pack / "a.md").write_text("ANA ALPHA CONTENT\n")
    state = types.SimpleNamespace(calls=[], pack=pack.resolve())

    async def post(self, path, *, json=None, params=None, headers=None):  # noqa: A002
        await asyncio.sleep(0)
        state.calls.append({"path": path, "json": json, "params": dict(params or {}),
                            "headers": dict(headers or {})})
        if path == PACK_RESOLVE_PATH:
            return FakeResp(200, {"packs": [{"alias": ALIAS, "owner_name": LABEL,
                                             "project": "notes", "root": str(state.pack)}]})
        if path == SEARCH_PATH:
            return FakeResp(200, {"results": []})
        return FakeResp(404, {"detail": "nope"})

    monkeypatch.setattr("captain_claw.fd_client.FDClient.post", post)
    state.resolves = lambda: [c for c in state.calls if c["path"] == PACK_RESOLVE_PATH]
    state.searches = lambda: [c for c in state.calls if c["path"] == SEARCH_PATH]
    return state


def _role(m):
    return getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")


def _content(m):
    return str((getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")) or "")


class ScriptProvider(LLMProvider):
    """Answers "done"; once per turn, a user message carrying READPACK asks
    for `read vfs:@ana-notes/a.md`, and SEARCHMEM for a deep-memory search."""

    def __init__(self, name="p"):
        self.model = f"fake-{name}"
        self.provider = "fake"
        self.calls: list[tuple[list, object]] = []
        self.fired = False
        self.reads = 0

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        await asyncio.sleep(0)
        self.calls.append((list(messages), tools))
        users = [i for i, m in enumerate(messages) if _role(m) == "user"]
        # The turn's user message (internal-context notes may follow it), with
        # no tool result after it yet.
        pending = bool(users) and not any(
            _role(m) == "tool" for m in messages[users[-1] + 1:])
        if tools and pending and not self.fired:
            text = _content(messages[users[-1]])
            if "READPACK" in text:
                self.fired = True
                self.reads += 1          # distinct args: no duplicate-call short-cut
                return LLMResponse(content="", tool_calls=[ToolCall(
                    id=f"c-{uuid.uuid4().hex[:8]}", name="read",
                    arguments={"path": f"vfs:@{ALIAS}/a.md", "limit": 1000 + self.reads})])
            if "SEARCHMEM" in text:
                self.fired = True
                return LLMResponse(content="", tool_calls=[ToolCall(
                    id=f"c-{uuid.uuid4().hex[:8]}", name="typesense",
                    arguments={"action": "search", "query": "plans"})])
        return LLMResponse(content="done")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text):
        return len(str(text).split()) or 1

    # what the tests read back
    def turn_prompts(self) -> list[str]:
        return [_content(msgs[0]) for msgs, tools in self.calls if tools and msgs]

    def tool_results(self) -> str:
        return "\n".join(_content(m) for msgs, _t in self.calls for m in msgs if _role(m) == "tool")


def _instructions(tmp_path):
    import captain_claw.agent_context_mixin as acm
    from captain_claw.instructions import InstructionLoader

    return InstructionLoader(
        base_dir=Path(acm.__file__).resolve().parent / "instructions",
        personal_dir=tmp_path / "personal",
    )


async def _main_agent(provider, tmp_path, sm, name="default"):
    from captain_claw.agent import Agent

    agent = Agent(provider=provider)
    agent.session = await sm.create_session(name=name)
    agent.session_manager = sm
    agent.instructions = _instructions(tmp_path)
    agent._initialized = True
    agent.memory = None
    agent._register_default_tools()
    return agent


class FakeWS:
    def __init__(self, lane: str | None = "A"):
        self.closed = False
        self.sent: list[str] = []
        self._is_admin = True
        if lane is not None:
            self._lane = lane

    async def send_str(self, data: str):
        self.sent.append(data)

    async def prepare(self, request):
        return None


@pytest.fixture
async def server(isolated, fd, tmp_path):
    from captain_claw.web_server import WebServer

    s = WebServer.__new__(WebServer)          # skip __init__'s heavy wiring
    s.config = get_config()
    s.clients = set()
    s._lane_agents = {}
    s._lane_locks = {}
    s._lane_sockets = {}
    s._public_agents = {}
    s._public_agent_locks = {}
    s._public_active_ws = {}
    s._pending_playbook_approvals = {}
    s._busy = False
    s._active_task = None
    s._orchestrator = None
    s._inbound_queue = asyncio.Queue()
    s._telegram_agents = {}
    s._telegram_user_sessions = {}
    s._telegram_user_locks = {}
    s._approved_telegram_users = {}
    s._telegram_bridge = None
    s._init_speaker_state()
    s.provider = ScriptProvider("main")
    s.scoped_provider = ScriptProvider("scoped")
    s._scoped_provider = lambda: s.scoped_provider
    s.agent = await _main_agent(s.provider, tmp_path, isolated.sm)
    yield s
    for agent in list(s._lane_agents.values()) + list(s._public_agents.values()):
        task = getattr(agent, "_public_task", None)
        if task is not None and not task.done():
            task.cancel()


def _connect(server, lane="A") -> FakeWS:
    ws = FakeWS(lane)
    server.clients.add(ws)
    server._lane_sockets.setdefault(lane, set()).add(ws)
    return ws


async def _settle(server, agent=None):
    task = getattr(agent, "_public_task", None) if agent is not None else server._active_task
    if task is not None:
        await asyncio.wait_for(task, 20)


def _assert_packs_used(provider, fd, *, posts=1, compact=False):
    prompts = provider.turn_prompts()
    assert prompts, "no turn reached the provider"
    marker = "SHARED-COMPACT-MARKER" if compact else "SHARED-FULL-MARKER"
    assert all(marker in p for p in prompts)
    assert prompts[0].index(marker) < prompts[0].index(SPLIT)
    assert len(fd.resolves()) == posts
    assert fd.resolves()[-1]["json"] == {"aliases": [ALIAS]}
    assert "ANA ALPHA CONTENT" in provider.tool_results()
    assert f"[shared by {LABEL}" in provider.tool_results()


def _assert_no_packs(provider, fd):
    prompts = provider.turn_prompts()
    assert prompts, "no turn reached the provider"
    for p in prompts:
        assert "SHARED-FULL-MARKER" not in p and "SHARED-COMPACT-MARKER" not in p
        assert SHARED_CONTEXT_MEMBER_NOTE not in p
    assert fd.resolves() == []
    assert PACKS_NOT_HERE_MESSAGE in provider.tool_results()
    assert "ANA ALPHA CONTENT" not in provider.tool_results()


# ── channel and automation turns DO get packs ────────────────────────


@pytest.mark.parametrize("extra", [
    {"whatsapp_waid": "385991234567", "origin": {"kind": "whatsapp", "address": "385991234567"}},
    {"origin": {"kind": "glasses", "address": "glasses-1"}},
    {"origin": {"kind": "web", "address": "fd"}},
    {},
])
async def test_lane_a_channel_frames_use_packs(server, fd, extra):
    """The shape the WhatsApp/glasses pump opens: a plain `/ws?token=` socket
    on lane A (no FD marker of any kind), a chat frame with its origin."""
    from captain_claw.web.ws_handler import handle_ws_message

    ws = _connect(server, "A")
    await handle_ws_message(server, ws, {"type": "chat", "content": "READPACK please", **extra})
    await _settle(server)
    _assert_packs_used(server.provider, fd)
    if extra.get("whatsapp_waid"):
        assert server.agent.session.metadata.get("whatsapp_waid") == "385991234567"


async def test_lane_a_with_an_fd_chat_and_a_pump_connected(server, fd):
    """No listener rule: both sockets on lane A, either one's turn uses packs."""
    from captain_claw.web.ws_handler import handle_ws_message

    fd_chat, pump = _connect(server, "A"), _connect(server, "A")
    await handle_ws_message(server, pump, {
        "type": "chat", "content": "READPACK from whatsapp",
        "whatsapp_waid": "385991234567", "origin": {"kind": "whatsapp", "address": "385991234567"},
    })
    await _settle(server)
    _assert_packs_used(server.provider, fd)
    server.provider.fired = False
    await handle_ws_message(server, fd_chat, {"type": "chat", "content": "READPACK from fd"})
    await _settle(server)
    _assert_packs_used(server.provider, fd, posts=2)
    assert any('"chat_message"' in s for s in fd_chat.sent) and any(
        '"chat_message"' in s for s in pump.sent)


async def test_lane_b_socket_uses_packs(server, fd):
    from captain_claw.web.ws_handler import handle_ws_message

    ws = _connect(server, "B")
    await handle_ws_message(server, ws, {"type": "chat", "content": "READPACK on lane B"})
    lane_agent = server._lane_agents["B"]
    await _settle(server, lane_agent)
    assert lane_agent is not server.agent and pack_access.packs_allowed(lane_agent)
    _assert_packs_used(server.scoped_provider, fd)
    assert server.provider.calls == []


async def test_inbound_peer_relay_uses_packs(server, fd):
    _connect(server, "A")
    consumer = asyncio.create_task(server._inbound_queue_consumer())
    try:
        await server._inbound_queue.put("[Delegated result from helper-2] READPACK relay")
        for _ in range(400):
            if server._active_task is not None or server.provider.calls:
                break
            await asyncio.sleep(0.01)
        await _settle(server)
    finally:
        consumer.cancel()
        try:
            await consumer
        except BaseException:
            pass
    _assert_packs_used(server.provider, fd)


async def test_hotkey_style_turn_uses_packs(server, fd):
    """The hotkey daemon ends in `handle_chat(server, ws, user_content, …)`
    on the first connected socket; that call, on the main agent."""
    from captain_claw.web import hotkey_daemon
    from captain_claw.web.chat_handler import handle_chat

    src = inspect.getsource(hotkey_daemon._do_activation)
    assert "ws = next(iter(server.clients), None)" in src
    assert "await handle_chat(server, ws, user_content, image_path=image_path)" in src
    _connect(server, "A")
    ws = next(iter(server.clients))
    await handle_chat(server, ws, "Here is some text I selected. READPACK", image_path=None)
    await _settle(server)
    _assert_packs_used(server.provider, fd)


async def test_slash_command_turn_uses_packs(server, fd):
    from captain_claw.web.ws_handler import handle_ws_message

    ws = _connect(server, "A")
    await handle_ws_message(server, ws, {"type": "chat", "content": "/orchestrate READPACK slash"})
    await _settle(server)
    _assert_packs_used(server.provider, fd)


async def test_cron_job_searches_with_packs(server, fd, isolated):
    from captain_claw.cron_dispatch import execute_cron_job

    sm = isolated.sm
    job = await sm.create_cron_job(
        kind="prompt", payload={"text": "SEARCHMEM what are the plans"},
        schedule={"type": "interval", "interval": 60, "unit": "minutes"},
        session_id=server.agent.session.id, next_run_at="2026-10-06T00:00:00+00:00",
    )
    ctx = server._get_web_runtime_context()
    assert ctx.agent is server.agent
    await execute_cron_job(ctx, job, trigger="manual")
    searches = fd.searches()
    assert len(searches) == 1 and searches[0]["json"]["packs"] is True
    assert all("SHARED-FULL-MARKER" in p for p in server.provider.turn_prompts())


async def test_telegram_user_agent_uses_packs(server, fd):
    from captain_claw.web.telegram import _tg_get_or_create_agent

    message = types.SimpleNamespace(user_id=4242, chat_id=4242, text="hi", username="tg-user")
    agent = await _tg_get_or_create_agent(server, message)
    assert agent is not server.agent and pack_access.packs_allowed(agent)
    agent.instructions = _instructions(Path(str(fd.pack)).parent)
    await agent.complete("READPACK from telegram")
    _assert_packs_used(server.provider, fd)        # Telegram agents share the main provider


async def test_api_pool_agent_uses_packs(server, fd):
    from captain_claw.agent_pool import AgentPool
    from captain_claw.tools.typesense import TypesenseTool

    provider = ScriptProvider("api")
    pool = AgentPool(provider=provider, session_name_prefix="api")
    agent = await pool.get_or_create("api-session-1")
    assert pack_access.packs_allowed(agent)
    agent.instructions = _instructions(Path(str(fd.pack)).parent)
    await agent.complete("READPACK from the API")
    _assert_packs_used(provider, fd)
    await TypesenseTool().execute(action="search", query="q", _agent=agent)
    assert fd.searches()[-1]["json"]["packs"] is True


# ── public, BotPort, Iskra and scope-restricted instances DON'T ──────


async def _refused_everywhere(agent, provider, fd):
    from captain_claw.tools.typesense import TypesenseTool

    assert pack_access.packs_allowed(agent) is False
    await agent.complete("READPACK please")
    _assert_no_packs(provider, fd)
    with pytest.raises(ToolBlockedError) as info:
        await agent.tools.execute("read", {"path": f"vfs:@{ALIAS}/a.md", "_agent": agent})
    assert info.value.reason == PACKS_NOT_HERE_MESSAGE
    assert fd.resolves() == []
    await TypesenseTool().execute(action="search", query="q", _agent=agent)
    assert fd.searches()[-1]["json"]["packs"] is False


async def test_public_session_agent_never_uses_packs(server, fd, isolated):
    session = await isolated.sm.create_session(name="public-visitor")
    agent = await server._get_public_agent(session.id)
    assert agent._public_scoped is True
    await _refused_everywhere(agent, server.scoped_provider, fd)
    assert "OWNER-PROFILE-MARKER" not in server.scoped_provider.turn_prompts()[0]


async def test_botport_dispatch_agent_never_uses_packs(server, fd):
    from captain_claw.botport_client import BotPortClient

    client = BotPortClient.__new__(BotPortClient)
    provider = ScriptProvider("botport")
    client._provider = provider
    client._dispatch_sessions = {}
    client._tool_output_callback = None
    client._status_callback = None
    client._thinking_callback = None
    client._send_activity = lambda *a, **k: None
    client._track_file_creation = lambda *a, **k: None
    agent = await client._spawn_dispatch_agent(
        "c" * 32, "do the task", {}, "remote-instance", "")
    assert agent._tenant_hidden is True
    await _refused_everywhere(agent, provider, fd)
    src = inspect.getsource(BotPortClient._spawn_dispatch_agent)
    assert "agent._tenant_hidden = True" in src


@pytest.mark.parametrize("setup", ["public_run", "being", "scope"])
async def test_refused_processes_never_use_packs(server, fd, monkeypatch, setup):
    if setup == "public_run":
        monkeypatch.setattr(get_config().web, "public_run", "chat")
    elif setup == "being":
        monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    else:
        monkeypatch.setenv("CLAW_VFS_SCOPE", "being-x,commons")
    await _refused_everywhere(server.agent, server.provider, fd)


async def test_public_turn_beside_a_main_turn(server, fd, isolated):
    """Same process, concurrent turns: the main agent's prompt carries the
    block and its read works; the public agent's never does."""
    session = await isolated.sm.create_session(name="public-visitor")
    public = await server._get_public_agent(session.id)
    await asyncio.gather(
        server.agent.complete("READPACK main"),
        public.complete("READPACK public"),
    )
    _assert_packs_used(server.provider, fd)
    for p in server.scoped_provider.turn_prompts():
        assert "SHARED-FULL-MARKER" not in p and "SHARED-COMPACT-MARKER" not in p
    assert PACKS_NOT_HERE_MESSAGE in server.scoped_provider.tool_results()
    assert len(fd.resolves()) == 1


async def test_no_turn_gate_in_the_signatures():
    from captain_claw.web import chat_handler, ws_handler

    for fn in (chat_handler.handle_chat, chat_handler._run_agent, ws_handler.ws_handler,
               ws_handler.handle_ws_message):
        params = inspect.signature(fn).parameters
        for gone in ("fd_ui", "fd_ui_turn", "packs_turn"):
            assert gone not in params, (fn.__name__, gone)
    src = inspect.getsource(ws_handler) + inspect.getsource(chat_handler)
    assert "_fd_ui" not in src and "set_turn_packs" not in src


# ── the prompt block ─────────────────────────────────────────────────


async def _prompt_agent(provider, tmp_path, sm):
    agent = await _main_agent(provider, tmp_path, sm, name=f"s-{uuid.uuid4().hex[:6]}")
    return agent


async def test_owner_prompt_order(isolated, shared_files, tmp_path):
    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    prompt = agent._build_system_prompt()
    t, s, c = prompt.index("OWNER-PROFILE-MARKER"), prompt.index("SHARED-FULL-MARKER"), prompt.index(SPLIT)
    assert t < s < c
    assert prompt.index("## Shared context on this agent") < s
    assert "SHARED-COMPACT-MARKER" not in prompt


async def test_shared_block_without_an_owner_profile(isolated, shared_files, tmp_path):
    import captain_claw.tenant_context as tc

    (shared_files / tc.FULL_FILENAME).unlink()
    (shared_files / tc.COMPACT_FILENAME).unlink()
    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    prompt = agent._build_system_prompt()
    assert "OWNER-PROFILE-MARKER" not in prompt
    assert prompt.index("SHARED-FULL-MARKER") < prompt.index(SPLIT)


async def test_telegram_style_prompt_has_the_block(server, fd):
    from captain_claw.web.telegram import _tg_get_or_create_agent

    agent = await _tg_get_or_create_agent(
        server, types.SimpleNamespace(user_id=7, chat_id=7, text="x", username="u"))
    agent.instructions = _instructions(Path(str(fd.pack)).parent)
    prompt = agent._build_system_prompt()
    assert prompt.index("SHARED-FULL-MARKER") < prompt.index(SPLIT)


async def test_member_prompt(isolated, shared_files, tmp_path):
    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    agent._speaker_scoped = True
    agent._speaker_principal = PRINCIPAL
    agent._speaker_profile = ("## About the member\nANA-MEMBER-PROFILE", "ANA-COMPACT")
    prompt = agent._build_system_prompt()
    note = f"{SPEAKER_MODE_NOTE_FULL} {SHARED_CONTEXT_MEMBER_NOTE}"
    assert note in prompt
    assert prompt.index("ANA-MEMBER-PROFILE") < prompt.index(note)
    assert f"{note}\n\n## Shared context on this agent" in prompt
    assert prompt.index("SHARED-FULL-MARKER") < prompt.index(SPLIT)
    assert "OWNER-PROFILE-MARKER" not in prompt


@pytest.mark.parametrize("kind", ["public", "hidden", "being"])
async def test_refused_instances_have_neither_block_nor_note(isolated, shared_files, tmp_path,
                                                             monkeypatch, kind):
    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    if kind == "public":
        agent._public_scoped = True
    elif kind == "hidden":
        agent._tenant_hidden = True
    else:
        monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    prompt = agent._build_system_prompt()
    for marker in ("SHARED-FULL-MARKER", "SHARED-COMPACT-MARKER", SHARED_CONTEXT_MEMBER_NOTE,
                   "## Shared context on this agent"):
        assert marker not in prompt, marker
    if kind == "being":
        # A being body that is also a member instance: no block, no note.
        agent._speaker_scoped = True
        agent._speaker_principal = PRINCIPAL
        agent._speaker_profile = ("", "")
        prompt = agent._build_system_prompt()
        assert SHARED_CONTEXT_MEMBER_NOTE not in prompt and "SHARED-FULL-MARKER" not in prompt


@pytest.mark.parametrize("flag", ["use_micro", "use_nano"])
async def test_micro_and_nano_use_the_compact_file(isolated, shared_files, tmp_path, flag):
    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    setattr(agent.instructions, flag, True)
    prompt = agent._build_system_prompt()
    assert "SHARED-COMPACT-MARKER" in prompt and "SHARED-FULL-MARKER" not in prompt
    if SPLIT in prompt:
        assert prompt.index("SHARED-COMPACT-MARKER") < prompt.index(SPLIT)


async def test_rewriting_the_file_changes_the_next_build(isolated, shared_files, tmp_path):
    import captain_claw.tenant_context as tc

    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    assert "SHARED-FULL-MARKER" in agent._build_system_prompt()
    (shared_files / tc.SHARED_FULL_FILENAME).write_text(
        "## Shared context on this agent\nREWRITTEN-MARKER and more text")
    prompt = agent._build_system_prompt()
    assert "REWRITTEN-MARKER" in prompt and "SHARED-FULL-MARKER" not in prompt


async def test_no_file_means_the_a2_prompt(isolated, tmp_path):
    import captain_claw.tenant_context as tc

    d = isolated.home / ".captain-claw"
    (d / tc.FULL_FILENAME).write_text(TENANT_FULL)
    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    before = agent._build_system_prompt()
    assert "Shared context on this agent" not in before
    (d / tc.SHARED_FULL_FILENAME).write_text(SHARED_FULL)
    assert "SHARED-FULL-MARKER" in agent._build_system_prompt()
    (d / tc.SHARED_FULL_FILENAME).unlink()
    after = agent._build_system_prompt()
    assert after.split(SPLIT)[0] == before.split(SPLIT)[0]
    agent._speaker_scoped = True
    agent._speaker_principal = PRINCIPAL
    agent._speaker_profile = ("", "")
    member = agent._build_system_prompt()
    assert SHARED_CONTEXT_MEMBER_NOTE not in member
    assert f"{SPEAKER_MODE_NOTE_FULL}\n\n{SPLIT}" in member


async def test_turn_cache_rerenders_when_packs_turn_off(isolated, shared_files, tmp_path,
                                                        monkeypatch):
    """A frozen turn prompt is reused only while the pack permission holds."""
    agent = await _prompt_agent(ScriptProvider(), tmp_path, isolated.sm)
    agent._turn_system_prompt = None
    first = agent._build_messages(query="hello")[0].content
    assert "SHARED-FULL-MARKER" in first
    again = agent._build_messages(query="hello")[0].content
    assert again == first                                     # cached within the turn
    monkeypatch.setattr(get_config().web, "public_run", "chat")
    third = agent._build_messages(query="hello")[0].content
    assert "SHARED-FULL-MARKER" not in third
