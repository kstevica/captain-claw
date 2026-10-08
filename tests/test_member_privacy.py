"""PR D: a turn that read members' private data feeds no shared learnings (contract 2b §2-§3, 2c §2, 2d §2, §4).

A real owner ``Agent`` runs real turns against a scripted provider: the first
turn asks ``shared_agent_usage read_conversation`` for Ana's conversation (FD
is a patched ``pack_access._fd_client``; her seeded session holds
``ANA-SECRET-42``) and replies quoting it. Every message of that turn after
the user's own is flagged ``member_private``; every reader that feeds shared
learnings, summaries or indexes leaves flagged messages out; the post-turn
jobs and auto-capture skip the turn; text the turn hands to another session
or agent carries the header. Member instances are never flagged.

HOME, FD_DATA_DIR, the workspace, every config DB path and the global
session / topic / datastore managers point at tmp before anything is built.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import types
import uuid
from pathlib import Path

import httpx
import pytest

from captain_claw import member_privacy, pack_access, shared_usage, speaker
from captain_claw import saved_attribution as sa
from captain_claw.config import get_config
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.member_privacy import (
    COMPACTION_PRIVATE_NOTE,
    FLAG,
    MEMBER_DATA_HEADER,
    MEMBER_SESSION_RATE_MESSAGE,
    PRIVATE_HEADER,
    TURN_INPUT,
)
from captain_claw.session import Session
from captain_claw.speaker import Principal
from captain_claw.tools.shared_agent_usage import SharedAgentUsageTool

WEB_AUTH = "test-web-auth"
REF = "process:helper:0123456789abcdef"
SHARED_AT = "2026-10-01T10:00:00+00:00"
SECRET = "ANA-SECRET-42"
OWNER_MARK = "OWNER-NOTE-77"
TOOL = member_privacy.TOOL_NAME
MEMBERS_FILE = "## People this agent is shared with\nYour owner shares this agent with “Ana” (a member)."
ANA_P = Principal("u-ana", "Ana", "Olga", "A", REF)


def _key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:8]


def _member(uid, name, *, shared_at=SHARED_AT):
    return {"user_id": uid, "key": _key(uid), "name": name, "label": f"“{name}” (a member)",
            "shared_at": shared_at, "google_enabled": False, "packs": []}


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
    "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER", "CLAW_COUNCIL_WORKER",
)
_DB_MODULES = ("nervous_system", "sister_session", "intentions", "cognitive_metrics", "insights",
               "datastore")


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
    monkeypatch.setattr(cfg.ui, "monitor_trace_llm", False)
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
    yield types.SimpleNamespace(home=home, sm=sm, ws=ws, cfg_dir=home / ".captain-claw")
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
    """A tool under a real name that records whether (and how) it ran."""
    from captain_claw.tools.registry import Tool, ToolResult

    class _Stub(Tool):
        async def execute(self, **kw):
            log.append((self.name, {k: v for k, v in kw.items() if not k.startswith("_")}))
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

    def _few_tools(self):
        from captain_claw.tools.read import ReadTool

        self.tools.register(ReadTool())
        self.tools.register(SharedAgentUsageTool(), metadata={"requires_shared_members": True})
        for name in ("insights", "web_fetch", "send_mail"):
            self.tools.register(stub_tool(name, ran))

    monkeypatch.setattr(Agent, "_register_default_tools", _few_tools)


class FakeFD:
    def __init__(self):
        self.calls: list[dict] = []
        self.members = [_member("u-ana", "Ana")]
        self.down = False

    async def post(self, path, *, json=None, params=None, headers=None):  # noqa: A002
        await asyncio.sleep(0)
        self.calls.append({"path": path, "json": json})
        if self.down:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"agent": {"name": "Helper", "runtime": "process"},
                                         "members": self.members, "truncated": False,
                                         "context": None})


@pytest.fixture
def fd(monkeypatch, isolated):
    fake = FakeFD()
    monkeypatch.setattr(pack_access, "_fd_client", lambda: fake)
    return fake


def msg(role, content, ts, **extra):
    m = {"message_id": uuid.uuid4().hex[:12], "role": role, "content": content,
         "tool_call_id": None, "tool_name": None, "tool_calls": None, "tool_arguments": None,
         "token_count": 1, "timestamp": ts}
    m.update(extra)
    return m


def at(day: int, hour: int = 9, minute: int = 0) -> str:
    return f"2026-10-{day:02d}T{hour:02d}:{minute:02d}:00+00:00"


async def seed(sm, uid, msgs, *, name="chat"):
    meta = {"speaker_id": uid, "speaker_lane": "A", "speaker_name": uid} if uid else {}
    session = Session(id=str(uuid.uuid4()), name=name, metadata=meta)
    session.messages = list(msgs)
    await sm.save_session(session)
    return session


@pytest.fixture
async def ana_session(isolated, fd):
    return await seed(isolated.sm, "u-ana", [
        msg("user", f"My code is {SECRET}", at(2, 9), turn_input=True),
        msg("assistant", "Noted.", at(2, 9, 1))])


def _role(m):
    return getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")


def _content(m):
    return str((getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")) or "")


def read_call(**args):
    return LLMResponse(content="", tool_calls=[ToolCall(
        id=f"c-{uuid.uuid4().hex[:8]}", name=TOOL,
        arguments={"action": "read_conversation", "member": "Ana", **args})])


def tool_call(name, **args):
    return LLMResponse(content="", tool_calls=[ToolCall(
        id=f"c-{uuid.uuid4().hex[:8]}", name=name, arguments=args)])


class ScriptProvider(LLMProvider):
    """Scripted responses for tool-enabled calls (then "done"); every prompt recorded."""

    def __init__(self, script=()):
        self.model = "fake"
        self.provider = "fake"
        self.calls: list[tuple[list, object]] = []
        self.script = list(script)
        self.on_call = None

    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        await asyncio.sleep(0)
        self.calls.append((list(messages), tools))
        if self.on_call is not None:
            self.on_call(messages, tools)
        if tools and self.script:
            return self.script.pop(0)
        return LLMResponse(content="done")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text):
        return len(str(text).split()) or 1

    def prompts_since(self, n: int) -> str:
        return "\n".join(_content(m) for msgs, _t in self.calls[n:] for m in msgs)


def _instructions(tmp_path):
    import captain_claw.agent_context_mixin as acm
    from captain_claw.instructions import InstructionLoader

    return InstructionLoader(
        base_dir=Path(acm.__file__).resolve().parent / "instructions",
        personal_dir=tmp_path / "personal",
    )


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
async def private_turn(isolated, fd, ana_session, tmp_path):
    """An owner agent after one plain turn (OWNER_MARK) and one private turn."""
    provider = ScriptProvider()
    agent = await make_agent(isolated.sm, tmp_path, provider)
    await agent.complete(f"{OWNER_MARK} remember my plain note")
    plain_end = len(agent.session.messages)
    provider.script = [read_call(), LLMResponse(content=f"Ana wrote {SECRET}.")]
    reply = await agent.complete("What did Ana say?")
    return types.SimpleNamespace(agent=agent, provider=provider, reply=reply, plain_end=plain_end)


# ── 1. flag_new_message / carry-over / turn input (units) ────────────


def _fake_agent(messages=None, sid="s1"):
    return types.SimpleNamespace(session=types.SimpleNamespace(id=sid, messages=messages or []))


def _add(agent, role, content):
    msg = {"role": role, "content": content}
    agent.session.messages.append(msg)
    member_privacy.flag_new_message(agent, msg)
    return msg


def test_turn_level_flags_every_message():
    a = _fake_agent()
    member_privacy.begin_turn(a)
    assert member_privacy.turn_level(a) == ""
    first = _add(a, "user", "plain question")
    assert FLAG not in first
    member_privacy.mark_private_read(a, "data")
    assert member_privacy.turn_level(a) == "data" and member_privacy.private_turn(a)
    assert not member_privacy.content_turn(a)
    for role in ("assistant", "tool", "user"):
        assert _add(a, role, "x")[FLAG] is True
    member_privacy.mark_private_read(a, "content")
    member_privacy.mark_private_read(a, "data")                # never lowered
    assert member_privacy.turn_level(a) == "content"
    member_privacy.mark_private_read(a, "bogus")
    assert member_privacy.turn_level(a) == "content"
    member_privacy.mark_private_read(None, "content")          # no-op, no error


@pytest.mark.parametrize("role", ["user", "assistant", "tool", "system"])
@pytest.mark.parametrize("header,level", [(PRIVATE_HEADER, "content"),
                                          (MEMBER_DATA_HEADER, "data")])
def test_header_in_any_role_flags_and_sets_the_level(role, header, level):
    a = _fake_agent()
    member_privacy.begin_turn(a)
    msg = _add(a, role, f"Response from peer:\n\n{header}\nbody")
    assert msg[FLAG] is True and member_privacy.turn_level(a) == level
    b = _fake_agent()
    member_privacy.begin_turn(b)
    late = _add(b, role, "x" * member_privacy.HEADER_SCAN + header)
    assert FLAG not in late and member_privacy.turn_level(b) == ""


def test_content_header_beats_data():
    a = _fake_agent()
    member_privacy.begin_turn(a)
    _add(a, "tool", PRIVATE_HEADER)
    _add(a, "tool", MEMBER_DATA_HEADER)
    assert member_privacy.turn_level(a) == "content"
    assert member_privacy.header_level(MEMBER_DATA_HEADER + PRIVATE_HEADER) == "content"
    assert member_privacy.header_for("content") == PRIVATE_HEADER
    assert member_privacy.header_for("data") == MEMBER_DATA_HEADER
    assert member_privacy.header_for("") == "" and member_privacy.header_level(None) == ""


def test_carry_over_flags_non_user_roles_while_a_flagged_message_remains(monkeypatch):
    a = _fake_agent()
    member_privacy.begin_turn(a)
    _add(a, "tool", PRIVATE_HEADER + "\nsecret")
    member_privacy.begin_turn(a)                               # the next turn
    user = _add(a, "user", "next question")
    assert FLAG not in user
    for role in ("assistant", "tool", "system"):
        assert _add(a, role, "restated")[FLAG] is True           # llm_trace / guard entries too
    assert FLAG not in _add(a, "user", "another")
    # Compaction removes the flagged messages and clears the cache.
    a.session.messages = [m for m in a.session.messages if not member_privacy.is_private(m)]
    setattr(a, member_privacy.CARRY_ATTR, None)
    assert FLAG not in _add(a, "assistant", "fresh")
    monkeypatch.setattr(member_privacy, "CARRY_OVER", False)
    b = _fake_agent()
    member_privacy.begin_turn(b)
    _add(b, "tool", PRIVATE_HEADER)
    member_privacy.begin_turn(b)
    assert FLAG not in _add(b, "assistant", "restated")


def test_carry_cache_scans_once_per_session(monkeypatch):
    scans = []
    real = member_privacy._scan_private

    def _spy(msgs):
        scans.append(len(msgs))
        return real(msgs)

    monkeypatch.setattr(member_privacy, "_scan_private", _spy)
    a = _fake_agent([{"role": "tool", "content": "x", FLAG: True}] + [
        {"role": "user", "content": str(i)} for i in range(50)], sid="s1")
    member_privacy.begin_turn(a)
    for _ in range(5):
        assert _add(a, "assistant", "r")[FLAG] is True
    assert len(scans) == 1
    a.session = types.SimpleNamespace(id="s2", messages=[{"role": "user", "content": "q"}])
    for _ in range(3):
        assert FLAG not in _add(a, "assistant", "r")
    assert len(scans) == 2
    # A flagged message landing through Session.add_message (no agent) is still seen.
    a.session.messages.append({"role": "user", "content": PRIVATE_HEADER, FLAG: True})
    assert _add(a, "assistant", "r")[FLAG] is True
    assert len(scans) == 2
    a.session.messages.clear()                                  # /clear: shorter → rescan
    assert FLAG not in _add(a, "assistant", "r")
    assert len(scans) == 3


def test_carry_over_raises_the_output_level_not_the_turn_level():
    """A later turn's reply flagged by carry-over goes out marked (the peer
    relay keeps the header) at the level the session's flagged data has; the
    turn level — and so the pause — stays off."""
    a = _fake_agent()
    member_privacy.begin_turn(a)
    _add(a, "tool", MEMBER_DATA_HEADER + "\nroster")
    assert member_privacy.output_level(a) == "data"
    member_privacy.begin_turn(a)                               # the next turn
    assert member_privacy.output_level(a) == ""
    _add(a, "user", "and again?")
    assert member_privacy.output_level(a) == ""
    assert _add(a, "assistant", "restated")[FLAG] is True
    assert member_privacy.turn_level(a) == "" and member_privacy.output_level(a) == "data"
    member_privacy.begin_turn(a)
    _add(a, "tool", PRIVATE_HEADER + "\nconversation")         # a content read this turn
    member_privacy.begin_turn(a)
    _add(a, "user", "q")
    _add(a, "assistant", "restated")
    assert member_privacy.turn_level(a) == "" and member_privacy.output_level(a) == "content"
    b = _fake_agent()                                          # flagged, no header left: content
    member_privacy.begin_turn(b)
    member_privacy.mark_private_read(b, "data")
    _add(b, "assistant", "reply of a data turn")
    member_privacy.begin_turn(b)
    _add(b, "assistant", "restated")
    assert member_privacy.output_level(b) == "content"
    c = _fake_agent()                                          # nothing flagged: no level
    member_privacy.begin_turn(c)
    _add(c, "assistant", "plain")
    assert member_privacy.output_level(c) == ""


def test_member_instances_are_never_flagged():
    a = _fake_agent()
    a._speaker_scoped = True
    member_privacy.begin_turn(a)
    msg = _add(a, "assistant", PRIVATE_HEADER + "\nquoted")
    assert FLAG not in msg and member_privacy.turn_level(a) == ""


def test_never_raises_on_junk():
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError(name)

        def __setattr__(self, name, value):
            raise RuntimeError(name)

    for agent in (None, Boom(), 5, "x"):
        member_privacy.begin_turn(agent)
        member_privacy.flag_new_message(agent, {"role": "assistant", "content": PRIVATE_HEADER})
        member_privacy.flag_new_message(agent, "not a dict")
        member_privacy.mark_turn_input(agent, {"role": "user"})
        assert member_privacy.turn_level(agent) == ""
        assert member_privacy.tool_block("write", {"action": None}, agent) is None
    assert member_privacy.learnable(5) == [] and member_privacy.visible_user(5) == set()
    assert member_privacy.header_level(object()) == ""


def test_turn_input_and_visible_user():
    a = _fake_agent()
    member_privacy.begin_turn(a)
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "nudge"}]
    member_privacy.mark_turn_input(a, msgs[0])
    member_privacy.mark_turn_input(a, msgs[1])
    member_privacy.mark_turn_input(a, msgs[2])
    assert msgs[0].get(TURN_INPUT) is True and TURN_INPUT not in msgs[1] and TURN_INPUT not in msgs[2]
    member_privacy.begin_turn(a)
    msgs.append({"role": "user", "content": "q2"})
    member_privacy.mark_turn_input(a, msgs[3])
    assert msgs[3][TURN_INPUT] is True
    assert member_privacy.visible_user(msgs) == {0, 3}
    legacy = [{"role": "user"}, {"role": "assistant"}, {"role": "user"}]
    assert member_privacy.visible_user(legacy) == {0, 2}
    mixed = [{"role": "user"}, {"role": "user"}, {"role": "user", TURN_INPUT: True},
             {"role": "user"}, {"role": "user", TURN_INPUT: True}]
    assert member_privacy.visible_user(mixed) == {0, 1, 2, 4}


# ── 2. an end-to-end owner turn ──────────────────────────────────────


async def test_private_turn_flags_everything_after_the_users_message(private_turn, isolated):
    agent = private_turn.agent
    msgs = agent.session.messages[private_turn.plain_end:]
    user = msgs[0]
    assert user["role"] == "user" and user["content"] == "What did Ana say?"
    assert FLAG not in user and user.get(TURN_INPUT) is True
    # The tool result and everything after it (the assistant message that
    # only asked for the tool came before anything was read).
    first = next(i for i, m in enumerate(msgs) if m["role"] == "tool" and m.get("tool_name") == TOOL)
    rest = msgs[first:]
    assert all(m.get(FLAG) is True for m in rest), [(m["role"], m.get(FLAG)) for m in rest]
    assert all(SECRET not in _content(m) for m in msgs[:first])
    assert SECRET in private_turn.reply
    for m in agent.session.messages[:private_turn.plain_end]:
        assert FLAG not in m
    assert member_privacy.turn_level(agent) == "content"
    await isolated.sm.save_session(agent.session)
    loaded = await isolated.sm.load_session(agent.session.id)
    assert [m.get(FLAG) for m in loaded.messages] == [m.get(FLAG) for m in agent.session.messages]
    assert loaded.messages[private_turn.plain_end].get(TURN_INPUT) is True


async def test_next_turn_resets_the_level_and_carries_over(private_turn, monkeypatch):
    agent, provider = private_turn.agent, private_turn.provider
    monkeypatch.setattr(get_config().ui, "monitor_trace_llm", True)
    agent.monitor_trace_llm = True
    levels = []
    provider.on_call = lambda msgs, tools: levels.append(member_privacy.turn_level(agent))
    start = len(agent.session.messages)
    await agent.complete("thanks, anything else?")
    assert levels and set(levels) == {""}
    new = agent.session.messages[start:]
    user = next(m for m in new if m["role"] == "user")
    assert FLAG not in user and user.get(TURN_INPUT) is True
    reply = [m for m in new if m["role"] == "assistant"]
    assert reply and all(m.get(FLAG) is True for m in reply)
    traces = [m for m in new if m.get("tool_name") == "llm_trace"]
    assert traces and all(m.get(FLAG) is True for m in traces)
    assert member_privacy.turn_level(agent) == ""


async def test_complete_with_a_header_input_gets_the_level(isolated, fd, tmp_path):
    agent = await make_agent(isolated.sm, tmp_path)
    await agent.complete(PRIVATE_HEADER + "\nRelayed: Ana said hi")
    assert member_privacy.turn_level(agent) == "content"
    first = next(m for m in agent.session.messages if m["role"] == "user")
    assert first[FLAG] is True


# ── 3. readers drop flagged messages ─────────────────────────────────


def _check(text: str):
    assert SECRET not in text
    assert OWNER_MARK in text


async def test_insights_reflections_dreams_intentions(private_turn):
    from captain_claw import insights, intentions_generator, nervous_system, reflections

    agent, provider = private_turn.agent, private_turn.provider
    for run in (lambda: insights.extract_insights(agent),
                lambda: reflections.generate_reflection(agent),
                lambda: nervous_system.dream(agent),
                lambda: intentions_generator._generate(agent, 2)):
        n = len(provider.calls)
        await run()
        assert len(provider.calls) > n
        _check(provider.prompts_since(n))


async def test_playbook_distillers(private_turn, isolated, monkeypatch):
    agent, provider = private_turn.agent, private_turn.provider
    await isolated.sm.save_session(agent.session)
    n = len(provider.calls)
    await agent._distill_playbook_from_session(agent.session.id, "good")
    _check(provider.prompts_since(n))

    seen = []

    class _Recorder:
        def __init__(self, cfg):
            pass

        async def complete(self, messages, tools=None, max_tokens=None):
            seen.append("\n".join(_content(m) for m in messages))
            return LLMResponse(content="{}")

    monkeypatch.setattr("captain_claw.llm.LLMProvider", _Recorder)
    from captain_claw.tools.playbooks import PlaybooksTool

    res = await PlaybooksTool().execute(action="rate", rating="good",
                                        _session_id=agent.session.id, _agent=agent)
    assert res.success and seen
    _check(seen[-1])


async def test_topics_collect_backfill_and_refresh(private_turn):
    from captain_claw import conversation_topics as ct

    agent, provider = private_turn.agent, private_turn.provider
    items, _ = ct._collect_new_messages(agent, 0, 100)
    _check("\n".join(it["excerpt"] for it in items))
    n = len(provider.calls)
    await ct.backfill_topics(agent)
    _check(provider.prompts_since(n))
    mgr = ct.get_topics_manager()
    flagged = next(m for m in agent.session.messages
                   if m.get(FLAG) and SECRET in str(m.get("content")))
    tid = mgr.upsert_topic("Refresh me")
    mgr.add_messages(tid, [{"role": "agent", "excerpt": "OLD-EXCERPT",
                            "msg_id": flagged["message_id"], "ts": at(5)}])
    ct.refresh_topic(agent, tid)
    excerpts = [m["excerpt"] for m in mgr.get_topic(tid)["messages"]]
    assert excerpts == ["OLD-EXCERPT"]


async def test_semantic_memory_session_documents(private_turn, isolated):
    from captain_claw.semantic_memory import SemanticMemoryIndex

    agent = private_turn.agent
    _check(SemanticMemoryIndex._format_messages_as_text(agent.session.messages))
    await isolated.sm.save_session(agent.session)
    idx = SemanticMemoryIndex.__new__(SemanticMemoryIndex)
    idx.session_db_path = Path(isolated.sm.db_path)
    docs = idx._collect_session_documents()
    own = next(d for d in docs if d.reference == agent.session.id)
    _check(own.text)
    ana_doc = [d for d in docs if SECRET in d.text]
    assert ana_doc and all(d.reference != agent.session.id for d in ana_doc)   # NB6: member's own


async def test_compaction_summarizes_only_learnable_messages(private_turn, monkeypatch):
    from captain_claw.semantic_memory import SemanticMemoryIndex

    agent = private_turn.agent
    # Two more turns, so the compacted window holds the private turn.
    await agent.complete("next one")
    await agent.complete("and another")
    monkeypatch.setattr(get_config().context, "max_tokens", 10)
    monkeypatch.setattr(get_config().context, "compaction_ratio", 0.05)
    seen, archived = [], []

    async def _summ(messages):
        seen.append("\n".join(_content(m) for m in messages))
        return "SUMMARY"

    monkeypatch.setattr(agent, "_summarize_for_compaction", _summ)
    agent.memory = types.SimpleNamespace(
        archive_session_history=lambda **kw: archived.append(
            SemanticMemoryIndex._format_messages_as_text(kw["messages"])),
        record_message=lambda *a: None)
    before = list(agent.session.messages)
    ok, info = await agent.compact_session(force=True)
    assert ok and seen
    old = before[:info["compacted_messages"]]
    assert any(member_privacy.is_private(m) and SECRET in _content(m) for m in old)
    _check(seen[0])
    _check(archived[0])
    assert getattr(agent, member_privacy.CARRY_ATTR) is None

    only_private = [m for m in agent.session.messages]
    for m in only_private:
        m[FLAG] = True
    compacted, _ = await agent._compact_messages_snapshot(only_private + [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}])
    assert len(seen) == 1                                     # nothing learnable → no LLM
    assert COMPACTION_PRIVATE_NOTE in compacted[0]["content"]


def test_record_narration_skips_a_private_turn(monkeypatch):
    from captain_claw import conversation_topics as ct
    from captain_claw.config import get_config

    monkeypatch.setattr(get_config().conversation_topics, "include_narration", True)
    a = types.SimpleNamespace()
    member_privacy.begin_turn(a)
    ct.record_narration(a, "kept blurb")
    member_privacy.mark_private_read(a, "data")
    ct.record_narration(a, "dropped blurb")
    assert getattr(a, ct._ATTR_NARRATION) == ["kept blurb"]


# ── 5. chat_handler: post-turn jobs and the final frame ──────────────


class _OwnerTurnAgent:
    """An owner instance: complete() taints the turn as the test says."""

    def __init__(self, level):
        self.level = level
        self.session = types.SimpleNamespace(id="own-session", messages=[])
        self.last_usage: dict = {}
        self.total_usage: dict = {}
        self.last_context_window: dict = {}
        self.provider = object()
        self.tools = types.SimpleNamespace(set_session_policy=lambda *a: None,
                                           clear_session_policy=lambda *a: None)

    def get_runtime_model_details(self):
        return {"provider": "p", "model": "m"}

    def _current_session_slug(self):
        return "own-session"

    async def complete(self, content):
        member_privacy.begin_turn(self)
        if self.level:
            member_privacy.mark_private_read(self, self.level)
        return "reply"


class _Server:
    LANE_MAIN = "A"

    def __init__(self, agent):
        self.agent = agent
        self.sent: list[dict] = []
        self._busy = True
        self._active_task = None
        self._orchestrator = None

    def _broadcast(self, msg):
        self.sent.append(msg)

    def _session_info(self, agent):
        return {}


@pytest.fixture
def post_turn(monkeypatch):
    calls = []

    def _spy(name):
        async def _job(*a, **k):
            calls.append(name)
        return _job

    for target, name in (
        ("captain_claw.reflections.maybe_auto_reflect", "reflect"),
        ("captain_claw.insights.maybe_extract_insights", "insights"),
        ("captain_claw.nervous_system.maybe_dream", "dream"),
        ("captain_claw.conversation_topics.maybe_classify_topics", "topics"),
        ("captain_claw.intentions_generator.maybe_auto_propose", "intentions"),
    ):
        monkeypatch.setattr(target, _spy(name))
    return calls


@pytest.mark.parametrize("level", ["content", "data", ""])
async def test_run_agent_owner_turn(post_turn, monkeypatch, level):
    from captain_claw.web import chat_handler

    monkeypatch.setattr(get_config().ui, "next_steps", True)

    async def _reset_level(provider, response):
        member_privacy.begin_turn(agent)             # a concurrent turn resetting it
        return []

    monkeypatch.setattr(chat_handler, "extract_next_steps", _reset_level)
    agent = _OwnerTurnAgent(level)
    server = _Server(agent)
    await chat_handler._run_agent(server, None, agent, "hello", None, lane="A", no_flow=True)
    for _ in range(5):
        await asyncio.sleep(0)
    final = [f for f in server.sent if f.get("type") == "chat_message"]
    assert len(final) == 1
    if level:
        assert final[0]["member_private"] == level
        assert post_turn == []
    else:
        assert "member_private" not in final[0]
        assert sorted(post_turn) == ["dream", "insights", "intentions", "reflect", "topics"]


async def test_carry_over_reply_marks_the_relay_frame(private_turn, post_turn):
    """Turn 2 answers from context without reading again: the stored reply is
    flagged by carry-over, so the final frame carries member_private (FD's
    relay adds the header) while the pause stays lifted."""
    from captain_claw.web import chat_handler

    agent, provider = private_turn.agent, private_turn.provider
    provider.script = [LLMResponse(content=f"From earlier: Ana wrote {SECRET}.")]
    server = _Server(agent)
    await chat_handler._run_agent(server, None, agent, "remind me what Ana said", None,
                                  lane="A", no_flow=True)
    for _ in range(5):
        await asyncio.sleep(0)
    final = [f for f in server.sent if f.get("type") == "chat_message"]
    assert len(final) == 1 and SECRET in final[0]["content"]
    reply = [m for m in agent.session.messages if m.get("role") == "assistant"][-1]
    assert member_privacy.is_private(reply)
    assert member_privacy.turn_level(agent) == ""
    assert final[0]["member_private"] == "content"
    assert post_turn == []


async def test_orchestrate_branch_marks_a_private_result(post_turn, monkeypatch):
    from captain_claw.web import chat_handler

    agent = _OwnerTurnAgent("")
    server = _Server(agent)

    async def _orchestrate(text):
        return PRIVATE_HEADER + "\nresult"

    server._orchestrator = types.SimpleNamespace(orchestrate=_orchestrate)
    await chat_handler._run_agent(server, None, agent, "/orchestrate do it", None, lane="A",
                                  no_flow=True)
    final = [f for f in server.sent if f.get("type") == "chat_message"]
    assert final[0]["member_private"] == "content" and post_turn == []


async def test_level_blocks_tools_only_while_a_turn_runs(private_turn, ran):
    """The level a finished turn leaves behind (still read by the final frame,
    sister and orchestrator output) never refuses a tool run outside
    complete()/stream() — a cron script, a flow tool step, the action rail."""
    agent, provider = private_turn.agent, private_turn.provider
    assert member_privacy.turn_level(agent) == "content"
    assert getattr(agent, member_privacy.ACTIVE_ATTR) == 0
    await agent.tools.execute("send_mail", {"to": "me", "_agent": agent})
    assert [n for n, _a in ran] == ["send_mail"]
    assert member_privacy.tool_block("send_mail", {}, agent) is None
    # Within a reading turn it is still refused.
    ran.clear()
    provider.script = [read_call(), tool_call("send_mail", to="x"), LLMResponse(content="ok")]
    await agent.complete("What did Ana say? Mail it.")
    assert ran == []
    refusals = [m["content"] for m in agent.session.messages
                if m["role"] == "tool" and m.get("tool_name") == "send_mail"]
    assert refusals and member_privacy.CONTENT_PAUSED_MESSAGE.format(tool="send_mail") in refusals[-1]
    # stream() scopes it the same way.
    provider.script = [read_call(), LLMResponse(content="ok")]
    async for _chunk in agent.stream("Read Ana again"):
        pass
    assert member_privacy.turn_level(agent) == "content"
    assert getattr(agent, member_privacy.ACTIVE_ATTR) == 0
    await agent.tools.execute("send_mail", {"to": "me", "_agent": agent})
    assert [n for n, _a in ran] == ["send_mail"]


def test_turn_running_counter():
    a = types.SimpleNamespace()
    assert member_privacy.turn_running(a)                      # never ran a turn: fail closed
    member_privacy.enter_turn(a)
    member_privacy.enter_turn(a)
    member_privacy.exit_turn(a)
    assert member_privacy.turn_running(a)
    member_privacy.exit_turn(a)
    member_privacy.exit_turn(a)
    assert not member_privacy.turn_running(a) and getattr(a, member_privacy.ACTIVE_ATTR) == 0
    member_privacy.mark_private_read(a, "content")
    assert member_privacy.tool_block("send_mail", {}, a) is None
    member_privacy.enter_turn(a)
    assert member_privacy.tool_block("send_mail", {}, a) == \
        member_privacy.CONTENT_PAUSED_MESSAGE.format(tool="send_mail")


# ── 6. auto-capture ──────────────────────────────────────────────────


async def test_auto_capture_skips_a_private_turn(isolated, fd, ana_session, tmp_path, monkeypatch):
    from captain_claw.agent import Agent

    calls = []

    def _rec(name):
        async def _f(self, *a, **k):
            calls.append(name)
        return _f

    for name in ("_auto_capture_todos", "_auto_capture_contacts", "_auto_capture_scripts",
                 "_auto_capture_apis", "_auto_capture_contacts_from_tool_call",
                 "_auto_capture_scripts_from_tool_call", "_auto_capture_apis_from_tool_call",
                 "_maybe_extract_insights_from_tool"):
        monkeypatch.setattr(Agent, name, _rec(name))
    provider = ScriptProvider([read_call(), LLMResponse(content=f"Ana wrote {SECRET}")])
    agent = await make_agent(isolated.sm, tmp_path, provider)
    await agent.complete("What did Ana say?")
    assert calls == []
    provider.script = [tool_call("web_fetch", url="https://example.com")]
    await agent.complete("fetch it")
    assert "_auto_capture_todos" in calls and "_maybe_extract_insights_from_tool" in calls


async def test_carry_over_reply_is_not_scanned_for_todos(private_turn, monkeypatch):
    """A later turn's reply that carry-over flagged (it restates member text
    from context) feeds no auto-captured to-do — those land in every later
    prompt; the owner's own words still do."""
    from captain_claw.agent import Agent

    seen = []

    async def _rec(self, user_message, assistant_response):
        seen.append((user_message, assistant_response))

    monkeypatch.setattr(Agent, "_auto_capture_todos", _rec)
    agent, provider = private_turn.agent, private_turn.provider
    reply = f"Once you send the file, {SECRET} gets mailed."
    provider.script = [LLMResponse(content=reply)]
    await agent.complete("remind me to call mom")
    assert member_privacy.turn_level(agent) == "" and member_privacy.output_level(agent)
    assert seen and seen[-1][1] == "" and "remind me to call mom" in seen[-1][0]


def test_insights_hook_returns_first():
    """The hook's first statement skips the tool and any private turn (no
    trigger is wired today, so this pins the guard's place)."""
    import inspect

    import captain_claw.agent_context_mixin as acm

    src = inspect.getsource(acm.AgentContextMixin._maybe_extract_insights_from_tool)
    guard = "if tool_name == member_privacy.TOOL_NAME or member_privacy.private_turn(self):"
    assert guard in src and src.index(guard) < src.index("cfg = get_config()")


# ── 7. the owner's topics get ────────────────────────────────────────


async def test_owner_topics_get(isolated, fd, tmp_path):
    from captain_claw.conversation_topics import get_topics_manager
    from captain_claw.tools.conversation_topics import TopicsTool

    mgr = get_topics_manager()
    await seed(isolated.sm, "u-ana", [
        msg("user", "ANA-OLD-EXCERPT", at(1, 8), message_id="m-ana-old", turn_input=True),
        msg("user", "ANA-EXCERPT", at(3), message_id="m-ana-new", turn_input=True)])
    tid = mgr.upsert_topic("Mixed")
    mgr.add_messages(tid, [
        {"role": "user", "excerpt": "ANA-EXCERPT", "ts": at(3), "speaker": "u-ana",
         "msg_id": "m-ana-new"},
        {"role": "user", "excerpt": "ANA-OLD-EXCERPT", "ts": at(1, 8), "speaker": "u-ana",
         "msg_id": "m-ana-old"},
        {"role": "user", "excerpt": "GONE-EXCERPT", "ts": at(3), "speaker": "u-gone"},
        {"role": "user", "excerpt": "OWNER-EXCERPT", "ts": at(3), "speaker": ""}])
    assert {m["speaker"] for m in mgr.get_topic(tid)["messages"]} == {"u-ana", "u-gone", ""}
    fd.members = [_member("u-ana", "Ana", shared_at="2026-10-01T12:00:00+00:00")]
    agent = await make_agent(isolated.sm, tmp_path)
    res = await TopicsTool().execute(action="get", topic=tid, _agent=agent)
    assert res.content.startswith(PRIVATE_HEADER + "\n")
    assert "ANA-EXCERPT" in res.content and "OWNER-EXCERPT" in res.content
    assert "GONE-EXCERPT" not in res.content and "ANA-OLD-EXCERPT" not in res.content
    assert member_privacy.turn_level(agent) == "content"

    shared_usage._reset_cache()
    fd.down = True
    agent2 = await make_agent(isolated.sm, tmp_path)
    res = await TopicsTool().execute(action="get", topic=tid, _agent=agent2)
    assert not res.content.startswith(PRIVATE_HEADER)
    assert "OWNER-EXCERPT" in res.content and "ANA-EXCERPT" not in res.content
    assert member_privacy.turn_level(agent2) == ""

    own = mgr.upsert_topic("Owner only")
    mgr.add_messages(own, [{"role": "user", "excerpt": "MINE", "ts": at(3), "speaker": ""}])
    calls = len(fd.calls)
    res = await TopicsTool().execute(action="get", topic=own, _agent=agent2)
    assert "MINE" in res.content and not res.content.startswith(PRIVATE_HEADER)
    assert len(fd.calls) == calls and member_privacy.turn_level(agent2) == ""

    member = await make_agent(isolated.sm, tmp_path)
    member._speaker_scoped = True
    member._speaker_principal = ANA_P
    res = await TopicsTool().execute(action="get", topic=tid, _agent=member)
    assert "ANA-EXCERPT" in res.content and "ANA-OLD-EXCERPT" in res.content
    assert "OWNER-EXCERPT" not in res.content and "GONE-EXCERPT" not in res.content
    assert not res.content.startswith(PRIVATE_HEADER)
    assert member_privacy.turn_level(member) == ""


async def test_owner_topics_get_shows_only_what_read_conversation_shows(isolated, fd, tmp_path):
    """A member's topic excerpts follow read_conversation's view: no reply
    grounded in their Google / private deep memory, no follow-up prompt the
    agent added itself, no narration, nothing whose message can't be found."""
    from captain_claw.conversation_topics import get_topics_manager
    from captain_claw.tools.conversation_topics import TopicsTool

    s = await seed(isolated.sm, "u-ana", [
        msg("user", "CAL-Q summarize my calendar", at(3, 9), turn_input=True, message_id="q1"),
        msg("assistant", "", at(3, 9, 1), message_id="c1", tool_calls=[
            {"id": "g", "function": {"name": "google_calendar", "arguments": "{}"}}]),
        msg("tool", "events", at(3, 9, 2), tool_name="google_calendar", message_id="t1"),
        msg("user", "Your previous response was empty. NUDGE-TEXT", at(3, 9, 3), message_id="n1"),
        msg("assistant", "Tomorrow: CAL-SECRET dentist", at(3, 9, 4), message_id="g1"),
        msg("user", "PLAIN-Q thanks", at(3, 10), turn_input=True, message_id="q2"),
        msg("assistant", "PLAIN-A you're welcome", at(3, 10, 1), message_id="a2"),
    ])
    assert s.messages
    mgr = get_topics_manager()
    tid = mgr.upsert_topic("Calendar")
    row = {"ts": at(3, 9), "speaker": "u-ana"}
    mgr.add_messages(tid, [
        {**row, "role": "user", "excerpt": "CAL-Q summarize my calendar", "msg_id": "q1"},
        {**row, "role": "user", "excerpt": "Your previous response was empty. NUDGE-TEXT",
         "msg_id": "n1"},
        {**row, "role": "agent", "excerpt": "Tomorrow: CAL-SECRET dentist", "msg_id": "g1"},
        {**row, "role": "user", "excerpt": "PLAIN-Q thanks", "msg_id": "q2"},
        {**row, "role": "agent", "excerpt": "PLAIN-A you're welcome", "msg_id": "a2"},
        {**row, "role": "narration", "excerpt": "NARRATION-TEXT checking calendar"},
        {**row, "role": "agent", "excerpt": "NO-ID-TEXT"},
        {**row, "role": "agent", "excerpt": "UNKNOWN-ID-TEXT", "msg_id": "zz"},
        {"role": "user", "excerpt": "OWNER-EXCERPT", "ts": at(3), "speaker": ""}])
    agent = await make_agent(isolated.sm, tmp_path)
    res = await TopicsTool().execute(action="get", topic=tid, _agent=agent)
    assert res.success and res.content.startswith(PRIVATE_HEADER + "\n")
    for shown in ("CAL-Q", "PLAIN-Q", "PLAIN-A", "OWNER-EXCERPT"):
        assert shown in res.content, shown
    for hidden in ("CAL-SECRET", "NUDGE-TEXT", "NARRATION-TEXT", "NO-ID-TEXT", "UNKNOWN-ID-TEXT"):
        assert hidden not in res.content, hidden
    assert member_privacy.turn_level(agent) == "content"
    # Only hidden member excerpts: no header, no level, the owner's still show.
    only = mgr.upsert_topic("Grounded only")
    mgr.add_messages(only, [
        {**row, "role": "agent", "excerpt": "Tomorrow: CAL-SECRET dentist", "msg_id": "g1"},
        {"role": "user", "excerpt": "OWNER-EXCERPT", "ts": at(3), "speaker": ""}])
    agent2 = await make_agent(isolated.sm, tmp_path)
    res = await TopicsTool().execute(action="get", topic=only, _agent=agent2)
    assert "CAL-SECRET" not in res.content and "OWNER-EXCERPT" in res.content
    assert not res.content.startswith(PRIVATE_HEADER)
    assert member_privacy.turn_level(agent2) == ""


# ── 8. member instances ──────────────────────────────────────────────


async def test_member_turn_never_flags(isolated, fd, tmp_path):
    provider = ScriptProvider([LLMResponse(content=PRIVATE_HEADER + "\nhaha")])
    member = await make_agent(isolated.sm, tmp_path, provider)
    member.session.metadata["speaker_id"] = "u-ana"
    member._speaker_scoped = True
    member._speaker_principal = ANA_P
    await member.complete("hello")
    assert member_privacy.turn_level(member) == ""
    assert not any(member_privacy.is_private(m) for m in member.session.messages)
    s = Session(id="m", name="m", metadata={"speaker_id": "u-ana"})
    s.add_message("user", PRIVATE_HEADER + "\nx")
    assert FLAG not in s.messages[-1]


# ── 9. the owner's history ───────────────────────────────────────────


class FakeHistory:
    def __init__(self, snaps):
        self.snaps = {s["history_id"]: s for s in snaps}

    def list_history(self, limit=20):
        return [{**s, "preview": s["text"][:200]} for s in self.snaps.values()][:limit]

    def get_history(self, history_id):
        return self.snaps.get(history_id)

    def search_history(self, query, max_results=None):
        return [types.SimpleNamespace(reference=h, snippet=s["text"], updated_at=s["created_at"],
                                      score=1.0) for h, s in self.snaps.items()
                if query.lower() in s["text"].lower()]


async def test_owner_history(isolated, fd, ana_session, tmp_path, monkeypatch):
    from captain_claw.tools.session_history import SessionHistoryTool

    gone = await seed(isolated.sm, "u-gone", [msg("user", "x", at(3))])
    agent = await make_agent(isolated.sm, tmp_path)
    snaps = [
        {"history_id": "h-ana", "session_id": ana_session.id, "session_name": "a",
         "text": "[user] ANA-SNAP word", "message_count": 1, "created_at": at(3)},
        {"history_id": "h-ana-old", "session_id": ana_session.id, "session_name": "a",
         "text": "[user] ANA-OLD-SNAP word", "message_count": 1, "created_at": at(1, 8)},
        {"history_id": "h-gone", "session_id": gone.id, "session_name": "g",
         "text": "[user] GONE-SNAP word", "message_count": 1, "created_at": at(3)},
        {"history_id": "h-own", "session_id": agent.session.id, "session_name": "o",
         "text": "[user] OWN-SNAP word", "message_count": 1, "created_at": at(3)},
    ]
    fd.members = [_member("u-ana", "Ana", shared_at="2026-10-01T12:00:00+00:00")]
    agent.memory = FakeHistory(snaps)
    tool = SessionHistoryTool()
    tool._agent = agent
    for action, kw in (("list", {}), ("search", {"query": "word"})):
        member_privacy.begin_turn(agent)
        res = await tool.execute(action=action, _agent=agent, **kw)
        assert res.content.startswith(PRIVATE_HEADER + "\n"), action
        assert "h-ana]" in res.content and "h-own]" in res.content
        assert "h-gone" not in res.content and "h-ana-old" not in res.content
        assert member_privacy.turn_level(agent) == "content"
    member_privacy.begin_turn(agent)
    res = await tool.execute(action="get", history_id="h-own", _agent=agent)
    assert "OWN-SNAP" in res.content and not res.content.startswith(PRIVATE_HEADER)
    assert member_privacy.turn_level(agent) == ""
    for hid in ("h-gone", "h-ana-old"):
        res = await tool.execute(action="get", history_id=hid, _agent=agent)
        assert not res.success and "No snapshot found" in res.error
    res = await tool.execute(action="get", history_id="h-ana", _agent=agent)
    assert res.content.startswith(PRIVATE_HEADER) and "ANA-SNAP" in res.content

    async def _boom(ids):
        raise RuntimeError("db")

    monkeypatch.setattr(shared_usage, "speakers_of", _boom)
    res = await tool.execute(action="list", _agent=agent)
    assert "h-own" not in res.content and "h-ana" not in res.content


# ── 10. playbooks rate on a member's session ─────────────────────────


async def test_owner_cannot_rate_a_member_session(isolated, fd, ana_session, monkeypatch):
    from captain_claw.tools.playbooks import PlaybooksTool

    llm = []

    class _Recorder:
        def __init__(self, cfg):
            pass

        async def complete(self, messages, tools=None, max_tokens=None):
            llm.append(1)
            return LLMResponse(content="{}")

    monkeypatch.setattr("captain_claw.llm.LLMProvider", _Recorder)
    res = await PlaybooksTool().execute(action="rate", rating="good", session_id=ana_session.id)
    assert not res.success and res.error == MEMBER_SESSION_RATE_MESSAGE
    assert llm == []
    loaded = await isolated.sm.load_session(ana_session.id)
    assert "playbook_rating" not in loaded.metadata
    tok = speaker.bind(ANA_P)
    try:
        res = await PlaybooksTool().execute(action="rate", rating="good",
                                            session_id=ana_session.id)
    finally:
        speaker.reset(tok)
    assert res.success and llm == [1]
    assert (await isolated.sm.load_session(ana_session.id)).metadata["playbook_rating"] == "good"


# ── 11. propagation ──────────────────────────────────────────────────


def test_session_add_message_flags_headers_on_owner_sessions_only():
    own = Session(id="o", name="o")
    own.add_message("user", PRIVATE_HEADER + "\nrelayed")
    own.add_message("user", "plain")
    own.add_message("tool", "Response from helper:\n\n" + MEMBER_DATA_HEADER)
    assert [m.get(FLAG) for m in own.messages] == [True, None, True]
    member = Session(id="m", name="m", metadata={"speaker_id": "u-ana"})
    member.add_message("user", PRIVATE_HEADER)
    assert FLAG not in member.messages[0]


async def test_silent_notification_lands_flagged(isolated, tmp_path):
    from captain_claw.web.ws_handler import handle_ws_message

    agent = await make_agent(isolated.sm, tmp_path)

    async def _resolve(ws):
        return agent

    server = types.SimpleNamespace(_broadcast=lambda m: None, resolve_agent=_resolve,
                                   _busy=False, _telegram_agents={})
    ws = types.SimpleNamespace()
    await handle_ws_message(server, ws, {
        "type": "notification",
        "content": f"[Delegated result from helper-2] {PRIVATE_HEADER}\nAna said hi"})
    assert agent.session.messages[-1][FLAG] is True
    await handle_ws_message(server, ws, {"type": "notification", "content": "plain update"})
    assert FLAG not in agent.session.messages[-1]


class _Pool:
    """The orchestrator's AgentPool: one scripted worker per task session."""

    def __init__(self, behave):
        self.behave = behave
        self.prompts: dict[str, list[str]] = {}

    async def get_or_create(self, session_id, **kw):
        pool = self

        class _Worker:
            def __init__(self):
                self.session = Session(id=session_id, name=session_id)
                self.max_iterations = 10
                self.last_usage = {}
                self.last_context_window = {}
                self._last_complete_success = True

            async def complete(self, prompt):
                member_privacy.begin_turn(self)
                pool.prompts.setdefault(session_id, []).append(prompt)
                return await pool.behave(self, session_id, prompt)

        return _Worker()

    async def evict_idle(self):
        return None


async def _orchestrate(behave):
    from captain_claw.session_orchestrator import SessionOrchestrator
    from captain_claw.task_graph import OrchestratorTask, TaskGraph

    orch = SessionOrchestrator(provider=ScriptProvider())
    graph = TaskGraph()
    graph.add_tasks([OrchestratorTask(id="t1", title="Read", description="read it", session_id="s-a"),
                     OrchestratorTask(id="t2", title="Use", description="use it", depends_on=["t1"],
                                      session_id="s-b")])
    graph.refresh()
    orch._graph = graph
    orch._pool = _Pool(behave)

    async def _noop(*a, **k):
        return None

    orch._assign_sessions = _noop
    orch._save_run_output = _noop
    result = await asyncio.wait_for(orch.execute(skip_synthesize=True), 30)
    return orch, result


async def test_orchestrator_propagates_the_header(isolated, fd, ana_session):
    async def _reads(worker, sid, prompt):
        if sid == "s-a":
            res = await shared_usage.run("read_conversation", worker, member="Ana")
            assert res.success
            return f"Ana wrote {SECRET}"
        return "used it"

    orch, result = await _orchestrate(_reads)
    assert result.startswith(PRIVATE_HEADER + "\n")
    assert orch._pool.prompts["s-b"][0].startswith(PRIVATE_HEADER + "\n")
    assert not orch._pool.prompts["s-a"][0].startswith(PRIVATE_HEADER)
    assert orch._graph.get_task("t1").result["output"].startswith(PRIVATE_HEADER + "\n")

    async def _clean(worker, sid, prompt):
        return "fine"

    orch, result = await _orchestrate(_clean)
    assert not result.startswith(PRIVATE_HEADER)
    assert not orch._pool.prompts["s-b"][0].startswith(PRIVATE_HEADER)


async def test_sister_session(isolated, fd, ana_session, tmp_path, monkeypatch):
    from captain_claw import sister_session as ss

    parent = await seed(isolated.sm, "", [
        msg("user", "PARENT-PLAIN", at(4)),
        msg("tool", PRIVATE_HEADER + f"\n{SECRET}", at(4, 9, 1), member_private=True),
        msg("assistant", f"restated {SECRET}", at(4, 9, 2), member_private=True)], name="parent")
    prompt = await ss._build_investigation_prompt({"trigger_reason": "x"}, parent.id)
    assert "PARENT-PLAIN" in prompt and SECRET not in prompt

    created = []

    async def _sister(parent_session_id, thinking_callback=None):
        session = await isolated.sm.create_session(name="sister")

        class _Sister:
            def __init__(self):
                self.session = session
                self.session_manager = isolated.sm

            async def complete(self, full_prompt):
                member_privacy.begin_turn(self)
                res = await shared_usage.run("read_conversation", self, member="Ana")
                assert res.success
                return json.dumps({"summary": "Looked at Ana", "actionable": False,
                                   "confidence": 0.5, "tags": []})

        created.append(_Sister())
        return created[-1]

    monkeypatch.setattr(ss, "_create_sister_agent", _sister)
    mgr = ss.get_sister_session_manager()
    task_id = await mgr.create_task(parent_session_id=parent.id, source_type="insight",
                                    source_id="i1", trigger_reason="check")
    out = await ss.execute_task(await mgr.get_task(task_id))
    assert out is not None
    briefing = await mgr.get_briefing(task_id)
    assert briefing["body"].startswith(PRIVATE_HEADER + "\n")
    last = created[0].session.messages[-1]
    assert last["content"].startswith(PRIVATE_HEADER + "\n[SISTER] Task completed:")
    assert last[FLAG] is True


async def test_sister_carry_over_marks_the_briefing(isolated, fd, monkeypatch):
    """A sister's session is persistent (linked to its parent): an investigation
    that restates what an earlier one read from members — carry-over flags its
    reply — still hands the parent a briefing that starts with the header."""
    from captain_claw import sister_session as ss

    parent = await seed(isolated.sm, "", [msg("user", "PARENT-PLAIN", at(4))], name="parent")
    session = await isolated.sm.create_session(name="sister")
    session.messages.append({"role": "tool", "content": PRIVATE_HEADER + f"\n{SECRET}",
                             FLAG: True})                  # an earlier investigation's read

    class _Sister:
        def __init__(self):
            self.session = session
            self.session_manager = isolated.sm

        async def complete(self, full_prompt):
            member_privacy.begin_turn(self)
            text = json.dumps({"summary": f"Ana wrote {SECRET}", "actionable": False,
                               "confidence": 0.5, "tags": []})
            _add(self, "assistant", text)                   # from context, no new read
            assert member_privacy.turn_level(self) == ""
            return text

    async def _sister(parent_session_id, thinking_callback=None):
        return _Sister()

    monkeypatch.setattr(ss, "_create_sister_agent", _sister)
    mgr = ss.get_sister_session_manager()
    task_id = await mgr.create_task(parent_session_id=parent.id, source_type="insight",
                                    source_id="i2", trigger_reason="again")
    assert await ss.execute_task(await mgr.get_task(task_id)) is not None
    briefing = await mgr.get_briefing(task_id)
    assert briefing["body"].startswith(PRIVATE_HEADER + "\n")


async def test_orchestrator_carry_over_worker_keeps_the_header(isolated, fd, ana_session):
    """A worker whose session already holds a flagged read (an earlier run on
    the same session) and answers from it: its output and every dependent
    prompt carry the header."""
    async def _from_context(worker, sid, prompt):
        if sid == "s-a":
            worker.session.messages.append(
                {"role": "tool", "content": PRIVATE_HEADER + f"\n{SECRET}", FLAG: True})
            _add(worker, "assistant", f"Ana wrote {SECRET}")
            assert member_privacy.turn_level(worker) == ""
            return f"Ana wrote {SECRET}"
        return "used it"

    orch, result = await _orchestrate(_from_context)
    assert orch._graph.get_task("t1").result["output"].startswith(PRIVATE_HEADER + "\n")
    assert orch._pool.prompts["s-b"][0].startswith(PRIVATE_HEADER + "\n")
    assert result.startswith(PRIVATE_HEADER + "\n")


async def test_private_briefing_never_enters_the_prompt(isolated, monkeypatch):
    """A briefing from a sister turn that read members' data (body starts with
    a header) is not injected into the parent's prompt: its summary would be
    restated by an untainted turn whose replies feed shared learnings."""
    import types as _types

    from captain_claw import sister_session as ss
    from captain_claw.agent_context_mixin import AgentContextMixin
    from captain_claw.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.sister_session, "enabled", True)
    monkeypatch.setattr(cfg.sister_session, "briefing_inject_in_context", True)
    mgr = ss.get_sister_session_manager()
    await mgr.create_briefing(id="b-private", parent_session_id="p1", source_type="insight",
                              trigger_reason="x", summary=f"PRIVATE-SUMMARY {SECRET}",
                              body=PRIVATE_HEADER + "\nfound things", actionable=False,
                              confidence=0.5)
    await mgr.create_briefing(id="b-plain", parent_session_id="p1", source_type="insight",
                              trigger_reason="x", summary="PLAIN-SUMMARY", body="plain body",
                              actionable=False, confidence=0.5)
    holder = _types.SimpleNamespace(session=_types.SimpleNamespace(id="p1"))
    await AgentContextMixin._refresh_briefing_context_cache(holder)
    note = AgentContextMixin._build_briefing_context_note(holder)
    assert "PLAIN-SUMMARY" in note
    assert "PRIVATE-SUMMARY" not in note and SECRET not in note


async def test_owner_history_drops_snapshots_of_sessions_begun_before_reshare(
        isolated, fd, tmp_path):
    """J16: a snapshot taken after a re-share (leave → re-add) from a session
    that began during the EARLIER membership may hold that membership's
    messages (snapshots carry no per-message times) — dropped; one from a
    session begun since the current shared_at is kept."""
    from captain_claw.tools.session_history import SessionHistoryTool

    old = Session(id=str(uuid.uuid4()), name="old", created_at=at(1, 8),
                  metadata={"speaker_id": "u-ana", "speaker_lane": "A", "speaker_name": "Ana"})
    old.messages = [msg("user", "x", at(1, 8))]
    await isolated.sm.save_session(old)
    new = Session(id=str(uuid.uuid4()), name="new", created_at=at(2),
                  metadata={"speaker_id": "u-ana", "speaker_lane": "A", "speaker_name": "Ana"})
    new.messages = [msg("user", "y", at(2))]
    await isolated.sm.save_session(new)
    agent = await make_agent(isolated.sm, tmp_path)
    fd.members = [_member("u-ana", "Ana", shared_at="2026-10-01T12:00:00+00:00")]
    agent.memory = FakeHistory([
        {"history_id": "h-span", "session_id": old.id, "session_name": "old",
         "text": "[user] BEFORE-LEAVE word", "message_count": 1, "created_at": at(3)},
        {"history_id": "h-new", "session_id": new.id, "session_name": "new",
         "text": "[user] SINCE-RESHARE word", "message_count": 1, "created_at": at(3)},
    ])
    tool = SessionHistoryTool()
    tool._agent = agent
    member_privacy.begin_turn(agent)
    res = await tool.execute(action="list", _agent=agent)
    assert "h-new]" in res.content and "h-span" not in res.content
    res = await tool.execute(action="get", history_id="h-span", _agent=agent)
    assert not res.success and "No snapshot found" in res.error
