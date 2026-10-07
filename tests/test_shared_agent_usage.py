"""PR D: the owner's agent looks into how its members use it (contract part 2, 2b §1, 2c §1).

``shared_agent_usage`` is owner-only: the gate (``shared_usage.usage_allowed``)
refuses member turns and public / BotPort / Iskra / ``CLAW_VFS_SCOPE``
instances, which don't even list it. Every action starts from Flight Deck's
live roster (a patched ``pack_access._fd_client``) and only those members —
and only what they did since their CURRENT ``shared_at`` — are queried
locally. Member text reaches the model quoted, under a header, and taints
the turn (``member_privacy``).

HOME, FD_DATA_DIR, the workspace, every config DB path and the global
session / topic / datastore managers point at tmp before anything is built;
nothing here reaches ~/.captain-claw, a real Flight Deck or a real LLM.
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
from captain_claw.exceptions import ToolBlockedError
from captain_claw.llm import LLMProvider, LLMResponse
from captain_claw.member_privacy import MEMBER_DATA_HEADER, PRIVATE_HEADER
from captain_claw.session import Session
from captain_claw.shared_usage import (
    COMMONS_HEADER,
    GROUNDED_REPLY_TEXT,
    NOT_HERE_MESSAGE,
    SHARED_USAGE_ROUTE,
    UNAVAILABLE_MESSAGE,
)
from captain_claw.speaker import Principal
from captain_claw.tools.shared_agent_usage import SharedAgentUsageTool

WEB_AUTH = "test-web-auth"
SPLIT = "<!-- CACHE_SPLIT -->"
REF = "process:helper:0123456789abcdef"
SHARED_AT = "2026-10-01T10:00:00+00:00"
TOOL = member_privacy.TOOL_NAME
MEMBERS_FILE = ("## People this agent is shared with\nYour owner shares this agent with "
                "“Ana Kovač” (a member) and “Marko” (a member) in Flight Deck. ROSTER-MARKER")
SHARED_FULL = "## Shared context on this agent\nSHARED-FULL-MARKER"
TENANT_FULL = "## Your owner\nOWNER-PROFILE-MARKER"
ANA_P = Principal("u-ana", "Ana Kovač", "Olga", "A", REF)


def _key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:8]


ANA_KEY, MARKO_KEY, GONE_KEY = _key("u-ana"), _key("u-marko"), _key("u-gone")


def _member(uid, name, label, *, shared_at=SHARED_AT, google=False, packs=()):
    return {"user_id": uid, "key": _key(uid), "name": name, "label": label,
            "shared_at": shared_at, "google_enabled": google, "packs": list(packs)}


def ana(**kw):
    return _member("u-ana", "Ana Kovač", "“Ana Kovač” (a member)", packs=[
        {"kind": "profile"}, {"kind": "vfs", "alias": "ana-notes", "project": "notes"},
        {"kind": "deep_memory", "tags": ["domain:legal"]}], **kw)


def marko(**kw):
    return _member("u-marko", "Marko", "“Marko” (a member)", **kw)


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


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """No post-turn LLM jobs, no network, few tools (incl. shared_agent_usage)."""
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

    monkeypatch.setattr(Agent, "_register_default_tools", _few_tools)


@pytest.fixture
def members_file(isolated):
    import captain_claw.tenant_context as tc

    p = isolated.cfg_dir / tc.SHARED_MEMBERS_FILENAME
    p.write_text(MEMBERS_FILE)
    return p


class FakeFD:
    """Flight Deck's roster route: records every call, answers from its state."""

    def __init__(self):
        self.calls: list[dict] = []
        self.members = [ana(), marko()]
        self.status = 200
        self.raw: object = None          # a body to send verbatim
        self.exc: Exception | None = None
        self.truncated = False
        self.context = {"profile": {"about_me": "ANA-ABOUT line1\nline2", "company": "Kovač d.o.o."},
                        "folders": [{"alias": "ana-notes", "project": "notes"}],
                        "deep_memory": {"tags": ["domain:legal"]}}

    async def post(self, path, *, json=None, params=None, headers=None):  # noqa: A002
        await asyncio.sleep(0)
        self.calls.append({"path": path, "json": json, "params": dict(params or {}),
                           "headers": dict(headers or {})})
        if self.exc is not None:
            raise self.exc
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "nope"})
        if self.raw is not None:
            return httpx.Response(200, content=self.raw if isinstance(self.raw, bytes)
                                  else __import__("json").dumps(self.raw).encode())
        uid = (json or {}).get("user_id") or ""
        ids = {m["user_id"] for m in self.members}
        if uid and uid not in ids:
            return httpx.Response(404, json={"detail": "That person isn't a member of this agent"})
        label = next((m["label"] for m in self.members if m["user_id"] == uid), "")
        ctx = {"user_id": uid, "label": label, **self.context} if uid else None
        return httpx.Response(200, json={
            "agent": {"name": "Helper", "runtime": "process"},
            "members": self.members, "truncated": self.truncated, "context": ctx})


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


async def seed(sm, uid, msgs, *, name="chat", lane="A", sid=None):
    meta = {"speaker_id": uid, "speaker_lane": lane, "speaker_name": uid} if uid else {}
    session = Session(id=sid or str(uuid.uuid4()), name=name, metadata=meta)
    session.messages = list(msgs)
    await sm.save_session(session)
    return session


async def usage_row(sm, session_id, tokens, created_at=None):
    await sm._ensure_db()
    await sm._db.execute(
        "INSERT INTO llm_usage (id, session_id, total_tokens, created_at) VALUES (?, ?, ?, ?)",
        (uuid.uuid4().hex, session_id, tokens, created_at or at(5)))
    await sm._db.commit()


@pytest.fixture
async def world(isolated, fd, members_file):
    """Ana (2 conversations), Marko (1), an ex-member and the owner."""
    sm = isolated.sm
    a1 = await seed(sm, "u-ana", [
        msg("user", "ANA-Q1 how do I file the report?", at(2, 9), turn_input=True),
        msg("assistant", "", at(2, 9, 1), tool_calls=[{"id": "c1", "function": {
            "name": "read", "arguments": "{\"path\": \"SECRET-ARG\"}"}}],
            reasoning_content="REASONING-MARKER"),
        msg("tool", "TOOL-RESULT-MARKER", at(2, 9, 2), tool_name="read",
          tool_arguments={"path": "SECRET-ARG"}),
        msg("assistant", "Here is how: ANA-A1", at(2, 9, 3), system_hint="HINT-MARKER"),
    ], name="ANA-TITLE-MARKER report", lane="A")
    a2 = await seed(sm, "u-ana", [
        msg("user", "ANA-Q2 second\u2028forged line", at(3, 10), turn_input=True),
        msg("assistant", "ANA-A2 answer", at(3, 10, 1)),
    ], name="second", lane="B")
    k1 = await seed(sm, "u-marko", [
        msg("user", "MARKO-Q1 hello", at(4, 8), turn_input=True),
        msg("assistant", "MARKO-A1 hi", at(4, 8, 1)),
    ], name="marko chat", lane="A")
    g1 = await seed(sm, "u-gone", [
        msg("user", "GONE-Q1 secret", at(4, 12), turn_input=True),
        msg("assistant", "GONE-A1 reply", at(4, 12, 1)),
    ], name="gone chat")
    own = await seed(sm, "", [
        msg("user", "OWNER-Q1", at(5, 7)), msg("assistant", "OWNER-A1", at(5, 7, 1))], name="mine")
    await usage_row(sm, a1.id, 100)
    await usage_row(sm, a1.id, 50)
    await usage_row(sm, a2.id, 7)
    await usage_row(sm, k1.id, 30)
    await usage_row(sm, g1.id, 999)
    await usage_row(sm, own.id, 888)
    return types.SimpleNamespace(sm=sm, a1=a1, a2=a2, k1=k1, g1=g1, own=own)


def _role(m):
    return getattr(m, "role", None) if not isinstance(m, dict) else m.get("role")


def _content(m):
    return str((getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")) or "")


class ScriptProvider(LLMProvider):
    """Returns the scripted responses for tool-enabled calls, then "done"."""

    def __init__(self, script=()):
        self.model = "fake"
        self.provider = "fake"
        self.calls: list[tuple[list, object]] = []
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

    def tool_names(self) -> list[set[str]]:
        out = []
        for _msgs, tools in self.calls:
            if tools:
                out.append({(t.get("name") if isinstance(t, dict) else getattr(t, "name", ""))
                            for t in tools})
        return out


def _instructions(tmp_path):
    import captain_claw.agent_context_mixin as acm
    from captain_claw.instructions import InstructionLoader

    return InstructionLoader(
        base_dir=Path(acm.__file__).resolve().parent / "instructions",
        personal_dir=tmp_path / "personal",
    )


async def make_agent(sm, tmp_path, provider=None, name=None):
    from captain_claw.agent import Agent

    agent = Agent(provider=provider or ScriptProvider())
    agent.session = await sm.create_session(name=name or f"own-{uuid.uuid4().hex[:6]}")
    agent.session_manager = sm
    agent.instructions = _instructions(tmp_path)
    agent._initialized = True
    agent.memory = None
    agent._register_default_tools()
    return agent


async def call(agent, **args):
    return await SharedAgentUsageTool().execute(**args, _agent=agent)


# ── 1. gates ─────────────────────────────────────────────────────────


async def test_owner_agent_is_allowed(isolated, fd, members_file, tmp_path):
    agent = await make_agent(isolated.sm, tmp_path)
    assert shared_usage.usage_instance_allowed(agent) is True
    assert shared_usage.usage_allowed(agent) is True
    res = await call(agent, action="list_members")
    assert res.success and res.content.startswith(MEMBER_DATA_HEADER)


@pytest.mark.parametrize("case", ["speaker_instance", "bound_principal", "identity_lost",
                                  "public", "hidden", "being", "scope", "public_run", "none"])
async def test_gate_refuses(isolated, fd, members_file, tmp_path, monkeypatch, case):
    agent = await make_agent(isolated.sm, tmp_path)
    tok = None
    if case == "speaker_instance":
        agent._speaker_scoped = True
        agent._speaker_principal = ANA_P
    elif case == "bound_principal":
        tok = speaker.bind(ANA_P)
    elif case == "identity_lost":
        monkeypatch.setattr(speaker, "_IN_FLIGHT", 1)
        monkeypatch.setattr(speaker, "_MAIN_LOOP", object())
        assert speaker.identity_lost() is True
    elif case == "public":
        agent._public_scoped = True
    elif case == "hidden":
        agent._tenant_hidden = True
    elif case == "being":
        monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    elif case == "scope":
        monkeypatch.setenv("CLAW_VFS_SCOPE", "being-x,commons")
    elif case == "public_run":
        monkeypatch.setattr(get_config().web, "public_run", "chat")
    else:
        agent = None
    try:
        assert shared_usage.usage_allowed(agent) is False
        res = await call(agent, action="list_members")
        assert not res.success and res.error == NOT_HERE_MESSAGE
        assert fd.calls == []
    finally:
        if tok is not None:
            speaker.reset(tok)


async def test_member_calls_are_blocked_in_the_registry(isolated, fd, members_file, tmp_path):
    agent = await make_agent(isolated.sm, tmp_path)
    tok = speaker.bind(ANA_P)
    try:
        with pytest.raises(ToolBlockedError):
            await agent.tools.execute(TOOL, {"action": "list_members", "_agent": agent})
    finally:
        speaker.reset(tok)
    member = await make_agent(isolated.sm, tmp_path)
    member._speaker_scoped = True
    member._speaker_principal = ANA_P
    with pytest.raises(ToolBlockedError):
        await member.tools.execute(TOOL, {"action": "list_members", "_agent": member})
    assert fd.calls == []
    assert TOOL not in speaker.SPEAKER_TOOL_ALLOWLIST_MAX
    assert TOOL not in speaker.SPEAKER_PATH_MAP
    _args, err = speaker.apply_tool_rules(TOOL, {"action": "list_members"}, member)
    assert err == speaker.NOT_ALLOWED_MESSAGE


async def test_listed_only_while_the_members_file_exists(isolated, fd, tmp_path, monkeypatch):
    import captain_claw.tenant_context as tc

    agent = await make_agent(isolated.sm, tmp_path)
    assert TOOL not in agent.tools.list_tools()
    with pytest.raises(ToolBlockedError):
        await agent.tools.execute(TOOL, {"action": "list_members", "_agent": agent})
    (isolated.cfg_dir / tc.SHARED_MEMBERS_FILENAME).write_text(MEMBERS_FILE)
    assert TOOL in agent.tools.list_tools()

    def _boom():
        raise RuntimeError("x")

    monkeypatch.setattr(tc, "load_shared_members", _boom)
    assert TOOL not in agent.tools.list_tools()
    assert "read" in agent.tools.list_tools()


def test_registered_with_the_metadata(monkeypatch):
    import captain_claw.agent_context_mixin as acm
    from captain_claw.config import Config
    from captain_claw.tools.registry import ToolRegistry

    cfg = Config()
    cfg.tools.enabled = ["read"]
    monkeypatch.setattr(acm, "get_config", lambda: cfg)
    rec = types.SimpleNamespace(tools=ToolRegistry(), _register_plugin_tools=lambda: None)
    acm.AgentContextMixin._register_default_tools(rec)
    assert rec.tools.has_tool(TOOL)
    assert rec.tools.get_tool_metadata(TOOL) == {"requires_shared_members": True}


@pytest.mark.parametrize("case", ["owner", "public", "hidden", "being", "scope", "none"])
async def test_drop_unusable(isolated, members_file, tmp_path, monkeypatch, case):
    agent = await make_agent(isolated.sm, tmp_path)
    if case == "public":
        agent._public_scoped = True
    elif case == "hidden":
        agent._tenant_hidden = True
    elif case == "being":
        monkeypatch.setenv("CLAW_BEING_WORKER", "1")
    elif case == "scope":
        monkeypatch.setenv("CLAW_VFS_SCOPE", "being-x")
    defs = agent.tools.get_definitions()
    assert TOOL in {d["name"] for d in defs}
    kept = {d["name"] for d in shared_usage.drop_unusable(defs, None if case == "none" else agent)}
    assert (TOOL in kept) is (case == "owner")
    assert "read" in kept


@pytest.mark.parametrize("public", [False, True])
async def test_definitions_sent_to_the_provider(isolated, fd, members_file, tmp_path, public):
    provider = ScriptProvider()
    agent = await make_agent(isolated.sm, tmp_path, provider)
    if public:
        agent._public_scoped = True
    await agent.complete("hello there")
    names = provider.tool_names()
    assert names, "no tool-enabled call reached the provider"
    assert all((TOOL in n) is (not public) for n in names)
    prompt = next(_content(msgs[0]) for msgs, tools in provider.calls if tools)
    assert (TOOL in prompt) is (not public)


async def test_mrav_toolpack_follows_the_rule(isolated, members_file, tmp_path):
    from captain_claw.mrav.runtime import MravRuntime

    agent = await make_agent(isolated.sm, tmp_path)
    rt = MravRuntime.__new__(MravRuntime)
    rt.tools = agent.tools
    rt.session_id = None
    rt.board = types.SimpleNamespace(pinned_tools=[])
    rt.agent = agent
    assert TOOL in rt._toolpack().all_names
    rt.agent = None
    assert TOOL not in rt._toolpack().all_names
    agent._public_scoped = True
    rt.agent = agent
    assert TOOL not in rt._toolpack().all_names


# ── 2. the Flight Deck call ──────────────────────────────────────────


async def test_fd_call_shape_and_cache(isolated, fd, members_file, tmp_path, monkeypatch):
    agent = await make_agent(isolated.sm, tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(shared_usage, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    await call(agent, action="list_members")
    assert len(fd.calls) == 1
    c = fd.calls[0]
    assert c["path"] == SHARED_USAGE_ROUTE
    assert c["json"] == {"user_id": ""}
    assert c["headers"].get("X-Agent-Auth") == WEB_AUTH
    assert speaker.GRANT_HEADER not in c["headers"]
    assert speaker.MEMBER_MARKER_PARAM not in c["params"]
    clock[0] += 5
    await call(agent, action="list_members")
    assert len(fd.calls) == 1                       # cached
    clock[0] += 6
    await call(agent, action="list_members")
    assert len(fd.calls) == 2                       # older than 10 s
    await call(agent, action="member_shared_context", member="Ana")
    await call(agent, action="member_shared_context", member="Ana")
    with_uid = [x for x in fd.calls if x["json"]["user_id"]]
    assert len(with_uid) == 2 and with_uid[0]["json"] == {"user_id": "u-ana"}


@pytest.mark.parametrize("failure", ["403", "500", "garbage", "notalist", "exception"])
async def test_fd_failure_is_unavailable_and_nothing_local_runs(
        isolated, fd, members_file, tmp_path, monkeypatch, failure):
    agent = await make_agent(isolated.sm, tmp_path)
    if failure == "403":
        fd.status = 403
    elif failure == "500":
        fd.status = 500
    elif failure == "garbage":
        fd.raw = b"<html>not json"
    elif failure == "notalist":
        fd.raw = {"members": {"u-ana": 1}}
    else:
        fd.exc = httpx.ConnectError("down")
    ran = []

    async def _spy(*a, **k):
        ran.append(a)
        return []

    monkeypatch.setattr(shared_usage, "load_member_sessions", _spy)
    for action in shared_usage.ACTIONS:
        res = await call(agent, action=action, member="Ana", query="hello")
        assert not res.success and res.error == UNAVAILABLE_MESSAGE, action
    assert ran == []
    assert member_privacy.turn_level(agent) == ""
    assert shared_usage._CACHE is None


async def test_failure_clears_the_cache(isolated, fd, members_file, tmp_path):
    agent = await make_agent(isolated.sm, tmp_path)
    assert (await call(agent, action="list_members")).success
    assert shared_usage._CACHE is not None
    fd.status = 500
    await shared_usage.fetch_roster("u-ana")
    assert shared_usage._CACHE is None


async def test_member_gone_on_context(isolated, fd, members_file, tmp_path, monkeypatch):
    agent = await make_agent(isolated.sm, tmp_path)
    assert (await call(agent, action="list_members")).success
    fd.members = [marko()]          # Ana left between the two calls (cached roster still has her)
    res = await call(agent, action="member_shared_context", member="Ana")
    assert not res.success
    assert res.error.startswith("No current member matches “Ana”.")


async def test_not_under_flight_deck(isolated, fd, members_file, tmp_path, monkeypatch):
    monkeypatch.delenv("FD_URL", raising=False)
    agent = await make_agent(isolated.sm, tmp_path)
    res = await call(agent, action="list_members")
    assert res.error == UNAVAILABLE_MESSAGE and fd.calls == []


async def test_roster_validation(isolated, fd, members_file):
    bad = [
        {**ana(), "key": "XYZ"},                                   # key not hex
        {**marko(), "user_id": ""},
        {**_member("u-x", "X", "“X” (a member)"), "packs": [{"kind": "vfs", "alias": "../etc"}]},
        {**_member("u-y", "Y", "“Y” (a member)"), "shared_at": "x" * 41},
    ]
    fd.members = bad + [_member("u-ok", "Ok\u202eName", "“Ok” <b>(a member)",
                                google="yes", packs=[{"kind": "deep_memory", "tags": ["a"] * 12}])]
    roster = await shared_usage.fetch_roster()
    assert [m.user_id for m in roster.members] == ["u-ok"]
    m = roster.members[0]
    assert m.google_enabled is False and m.name == "OkName" and "<" not in m.label
    assert m.packs == ({"kind": "deep_memory", "tags": ["a"] * 10},)
    assert roster.agent_name == "Helper" and roster.runtime == "process"


# ── 3. ex-members and membership scope (J16) ─────────────────────────


async def test_ex_member_never_appears(world, tmp_path, monkeypatch):
    agent = await make_agent(world.sm, tmp_path)
    seen_params: list = []
    real_execute = world.sm._db.execute

    def _spy(sql, params=()):
        seen_params.append((sql, list(params)))
        return real_execute(sql, params)

    monkeypatch.setattr(world.sm._db, "execute", _spy)
    outputs = []
    for args in ({"action": "list_members"}, {"action": "member_activity", "member": "Ana"},
                 {"action": "search_conversations", "query": "secret"},
                 {"action": "search_conversations", "query": "GONE"},
                 {"action": "read_conversation", "member": "Ana"}):
        res = await call(agent, **args)
        outputs.append(res.content or res.error)
    text = "\n".join(outputs)
    for marker in ("GONE-Q1", "GONE-A1", "gone chat", world.g1.id):
        assert marker not in text
    res = await call(agent, action="read_conversation", conversation=world.g1.id)
    assert not res.success and "has no conversation" in res.error
    res = await call(agent, action="read_conversation", conversation=world.a1.id, member="Marko")
    assert not res.success
    assert res.error == shared_usage.NO_CONVERSATION_MESSAGE.format(
        label="“Marko” (a member)", c=world.a1.id)
    member_sql = [(s, p) for s, p in seen_params if "FROM sessions" in s and "speaker_id" in s]
    assert member_sql
    assert all("u-gone" not in p for _s, p in member_sql)
    assert all("u-gone" not in p for _s, p in seen_params)


async def test_membership_scope_since_shared_at(world, tmp_path, fd):
    """Leave → re-add: only what happened since the CURRENT shared_at shows."""
    sm = world.sm
    old = await seed(sm, "u-ana", [
        msg("user", "OLD-ANA-Q before the re-share", at(1, 8), turn_input=True),
        msg("assistant", "OLD-ANA-A", at(1, 8, 1))], name="OLD-SESSION")
    span = await seed(sm, "u-ana", [
        msg("user", "SPAN-OLD-Q", at(1, 9), turn_input=True),
        msg("assistant", "SPAN-OLD-A", at(1, 9, 1)),
        msg("user", "SPAN-NEW-Q", at(6, 9), turn_input=True),
        msg("assistant", "SPAN-NEW-A", at(6, 9, 1))], name="spanning")
    await usage_row(sm, span.id, 5000, created_at=at(1, 9, 30))
    await usage_row(sm, span.id, 11, created_at=at(6, 9, 30))
    from captain_claw.conversation_topics import get_topics_manager

    tm = get_topics_manager()
    tid = tm.upsert_topic("Old Topic")
    tm.add_messages(tid, [{"role": "user", "excerpt": "x", "ts": at(1, 8), "speaker": "u-ana"}])
    tid2 = tm.upsert_topic("Fresh Topic")
    tm.add_messages(tid2, [{"role": "user", "excerpt": "y", "ts": at(6, 9), "speaker": "u-ana"}])
    fd.members = [ana(shared_at="2026-10-01T12:00:00+00:00"), marko()]
    agent = await make_agent(sm, tmp_path)

    act = (await call(agent, action="member_activity", member="Ana")).content
    assert old.id not in act and span.id in act
    assert "Fresh Topic" in act and "Old Topic" not in act
    assert "messages: 1 from them, 1 from the agent · 11 tokens" in act
    lst = (await call(agent, action="list_members")).content
    assert "5000" not in lst and "5011" not in lst
    res = await call(agent, action="search_conversations", query="OLD-ANA")
    assert res.content.startswith("No matches")
    res = await call(agent, action="search_conversations", query="SPAN-OLD")
    assert res.content.startswith("No matches")
    res = await call(agent, action="read_conversation", member="Ana", conversation=old.id)
    assert not res.success
    read = (await call(agent, action="read_conversation", member="Ana", conversation=span.id)).content
    assert "SPAN-NEW-Q" in read and "SPAN-OLD" not in read
    assert "messages 1–2 of 2" in read

    # Owner change = FD hands back a new shared_at → same rule.
    shared_usage._reset_cache()
    fd.members = [ana(shared_at=at(6, 10)), marko()]
    res = await call(agent, action="member_activity", member="Ana")
    assert res.content == shared_usage.NO_CONVERSATIONS_MESSAGE.format(label="“Ana Kovač” (a member)")

    # Unparseable shared_at → nothing of hers.
    shared_usage._reset_cache()
    member_privacy.begin_turn(agent)
    fd.members = [ana(shared_at="not a date"), marko()]
    lst = (await call(agent, action="list_members")).content
    ana_line = next(ln for ln in lst.splitlines() if "Ana" in ln)
    assert "0 conversation(s), 0 message(s), last active never, 0 tokens" in ana_line
    res = await call(agent, action="search_conversations", member="Ana", query="ANA")
    assert res.content.startswith("No matches")


# ── 4. list_members / member_activity ────────────────────────────────


async def test_list_members_totals(world, tmp_path):
    sid = "tg-Ana Kovač 1"
    await seed(world.sm, "u-ana", [
        msg("user", "TG-Q", at(6, 9), turn_input=True), msg("assistant", "TG-A", at(6, 9, 1))],
        sid=sid, name="telegram")
    from captain_claw.agent_file_ops_mixin import AgentFileOpsMixin
    from captain_claw.tools.write import WriteTool

    slug = AgentFileOpsMixin._normalize_session_slug(sid)
    assert slug != sid and slug != WriteTool._normalize_session_id(sid)
    await usage_row(world.sm, sid, 3)
    await usage_row(world.sm, slug, 4)
    agent = await make_agent(world.sm, tmp_path)
    res = await call(agent, action="list_members")
    text = res.content
    lines = text.splitlines()
    assert lines[0] == MEMBER_DATA_HEADER
    assert lines[1] == "“Helper” is shared with 2 member(s):"
    assert lines[2] == (
        f"- “Ana Kovač” (a member) · key {ANA_KEY} · member since 2026-10-01 · Google off · "
        "shared: profile, folder vfs:@ana-notes, deep memory (tags: domain:legal) · "
        "3 conversation(s), 6 message(s), last active 2026-10-06 09:01, 164 tokens")
    assert lines[3] == (
        f"- “Marko” (a member) · key {MARKO_KEY} · member since 2026-10-01 · Google off · "
        "shared: nothing · 1 conversation(s), 2 message(s), last active 2026-10-04 08:01, 30 tokens")
    assert lines[-1].startswith("Use member_activity, member_items")
    for marker in ("ANA-Q1", "ANA-A1", "ANA-TITLE-MARKER", "MARKO-Q1", "OWNER-Q1", "TG-Q", "telegram"):
        assert marker not in text
    assert member_privacy.turn_level(agent) == "data"


async def test_list_members_paging_and_truncation(world, tmp_path, fd):
    fd.members = [ana(), marko()] + [_member(f"u-m{i}", f"M{i}", f"“M{i}” (a member)")
                                     for i in range(3)]
    fd.truncated = True
    agent = await make_agent(world.sm, tmp_path)
    text = (await call(agent, action="list_members", offset=1, limit=2)).content
    assert "is shared with 5 member(s):" in text
    assert "“Marko”" in text and "“M0”" in text and "“Ana" not in text and "“M1”" not in text
    assert "(2 more: offset=3.)" in text
    assert "(More than 200 members — the rest aren't shown.)" in text


async def test_no_members_means_no_taint(isolated, fd, members_file, tmp_path):
    fd.members = []
    agent = await make_agent(isolated.sm, tmp_path)
    res = await call(agent, action="list_members")
    assert res.success and res.content == shared_usage.NO_MEMBERS_MESSAGE
    assert member_privacy.turn_level(agent) == ""
    res = await call(agent, action="member_activity", member="Ana")
    assert not res.success and res.error == shared_usage.NO_MEMBERS_MESSAGE


async def test_member_activity(world, tmp_path):
    from captain_claw.conversation_topics import get_topics_manager

    tm = get_topics_manager()
    for i in range(10):
        tid = tm.upsert_topic(f"Topic {i}")
        tm.add_messages(tid, [{"role": "user", "excerpt": "e", "ts": at(3), "speaker": "u-ana"}])
    tm.add_messages(tm.upsert_topic("Marko Only"), [{"role": "user", "excerpt": "e",
                                                     "ts": at(3), "speaker": "u-marko"}])
    agent = await make_agent(world.sm, tmp_path)
    text = (await call(agent, action="member_activity", member=ANA_KEY)).content
    lines = text.splitlines()
    assert lines[0] == MEMBER_DATA_HEADER
    assert lines[1] == f"“Ana Kovač” (a member) · key {ANA_KEY} — 2 conversation(s) on this agent:"
    assert lines[2] == (f"- {world.a2.id} · lane B · started {world.a2.created_at[:16].replace('T', ' ')}"
                        " · last active 2026-10-03 10:01 · messages: 1 from them, 1 from the agent "
                        "· 7 tokens")
    assert lines[3].startswith(f"- {world.a1.id} · lane A") and lines[3].endswith("· 150 tokens")
    assert "messages: 1 from them, 1 from the agent" in lines[3]
    topics = lines[4]
    assert topics.startswith("Topics they talked about: Topic 9, Topic 8")
    assert topics.count("Topic ") == 8 and "Marko Only" not in topics
    assert lines[5].startswith("Message contents aren't shown here")
    for marker in ("ANA-Q1", "ANA-A1", "TOOL-RESULT-MARKER", "ANA-TITLE-MARKER"):
        assert marker not in text
    assert member_privacy.turn_level(agent) == "data"
    res = await call(await make_agent(world.sm, tmp_path), action="member_activity", member="Marko")
    assert "Topics they talked about: Marko Only" in res.content


async def test_member_without_sessions(world, tmp_path, fd):
    fd.members = [ana(), marko(), _member("u-new", "Nova", "“Nova” (a member)")]
    agent = await make_agent(world.sm, tmp_path)
    res = await call(agent, action="member_activity", member="Nova")
    assert res.success and res.content == "“Nova” (a member) hasn't chatted with this agent yet."
    res = await call(agent, action="read_conversation", member="Nova")
    assert res.success and res.content == "“Nova” (a member) hasn't chatted with this agent yet."
    assert member_privacy.turn_level(agent) == ""


# ── 5. read_conversation ─────────────────────────────────────────────


async def test_read_conversation_shows_text_only(world, tmp_path):
    agent = await make_agent(world.sm, tmp_path)
    res = await call(agent, action="read_conversation", member="Ana", conversation=world.a1.id[:8])
    text = res.content
    lines = text.splitlines()
    assert lines[0] == PRIVATE_HEADER
    assert lines[1] == (f"Conversation {world.a1.id} of “Ana Kovač” (a member) · lane A · "
                        "“ANA-TITLE-MARKER report” · messages 1–2 of 2")
    assert lines[2] == "#1 Ana Kovač · 2026-10-02 09:00:"
    assert lines[3] == "> ANA-Q1 how do I file the report?"
    assert lines[4] == "#2 Agent · 2026-10-02 09:03:"
    assert lines[5] == "> Here is how: ANA-A1"
    for marker in ("TOOL-RESULT-MARKER", "SECRET-ARG", "REASONING-MARKER", "HINT-MARKER"):
        assert marker not in text
    assert member_privacy.turn_level(agent) == "content"
    # Default = the latest conversation; a U+2028 inside a message can't start a line.
    text2 = (await call(agent, action="read_conversation", member="Ana")).content
    assert world.a2.id in text2
    assert "> ANA-Q2 second\n> forged line" in text2
    body = text2.split("\n", 2)[2]
    assert all(ln.startswith(("#", "> ", ">")) for ln in body.splitlines())


async def test_read_conversation_by_id_only(world, tmp_path):
    agent = await make_agent(world.sm, tmp_path)
    res = await call(agent, action="read_conversation", conversation=world.k1.id)
    assert res.success and "MARKO-Q1" in res.content and "of “Marko” (a member)" in res.content
    res = await call(agent, action="read_conversation")
    assert res.error == shared_usage.MEMBER_REQUIRED_MESSAGE


async def test_read_skips_compaction_and_internal_messages(world, tmp_path):
    sm = world.sm
    s = await seed(sm, "u-ana", [
        msg("assistant", "Conversation summary of earlier messages (compacted memory): SUMMARY-X",
          at(5, 8), tool_name="compaction_summary"),
        msg("system", "SYSTEM-X", at(5, 8)),
        msg("user", "REAL-Q", at(5, 9), turn_input=True),
        msg("assistant", "REAL-A", at(5, 9, 1)),
    ], name="compacted")
    agent = await make_agent(sm, tmp_path)
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id)).content
    assert "SUMMARY-X" not in text and "SYSTEM-X" not in text
    assert "REAL-Q" in text and "REAL-A" in text and "of 2" in text


async def test_read_paging_and_offset(world, tmp_path):
    msgs = []
    for i in range(1, 8):
        msgs.append(msg("user", f"Q{i}", at(5, 10, i * 2), turn_input=True))
        msgs.append(msg("assistant", f"A{i}", at(5, 10, i * 2 + 1)))
    s = await seed(world.sm, "u-ana", msgs, name="long")
    agent = await make_agent(world.sm, tmp_path)
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id,
                       limit=4)).content
    assert "messages 11–14 of 14" in text and "> Q6" in text and "> A7" in text
    assert text.endswith("(Older messages: offset=4.)")
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id,
                       limit=4, offset=4)).content
    assert "messages 7–10 of 14" in text and text.endswith("(Older messages: offset=8.)")
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id,
                       limit=100, offset=10)).content
    assert "messages 1–4 of 14" in text and "Older messages" not in text
    res = await call(agent, action="read_conversation", member="Ana", conversation=s.id, offset=14)
    assert not res.success
    assert res.error == "That conversation has 14 message(s) — use a smaller offset."


async def test_read_output_max_trims_oldest(world, tmp_path, monkeypatch):
    msgs = []
    for i in range(1, 11):
        msgs.append(msg("user", f"Q{i} " + "x" * 3000, at(5, 11, i * 2), turn_input=True))
        msgs.append(msg("assistant", f"A{i}", at(5, 11, i * 2 + 1)))
    s = await seed(world.sm, "u-ana", msgs, name="big")
    agent = await make_agent(world.sm, tmp_path)
    monkeypatch.setattr(shared_usage, "OUTPUT_MAX", 9000)
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id,
                       limit=20)).content
    assert len(text) <= 9000
    assert "> Q10 " + "x" * (shared_usage.MESSAGE_CLIP - 4) + "…\n" in text   # MESSAGE_CLIP
    import re

    first = int(re.search(r"messages (\d+)–20 of 20", text).group(1))
    assert f"#{first} " in text and f"#{first - 1} " not in text
    assert text.endswith(f"(Older messages: offset={20 - (first - 1)}.)")


async def test_grounded_replies_are_hidden(world, tmp_path, monkeypatch):
    sm = world.sm
    s = await seed(sm, "u-ana", [
        msg("assistant", "ORPHAN-REPLY before any member message", at(5, 7)),
        msg("user", "MAIL-Q check my inbox", at(5, 8), turn_input=True),
        msg("assistant", "", at(5, 8, 1), tool_calls=[{"id": "g", "function": {
            "name": "google_mail", "arguments": "{}"}}]),
        msg("tool", "inbox", at(5, 8, 2), tool_name="google_mail"),
        msg("assistant", "You have mail: GMAIL-SECRET-7", at(5, 8, 3)),
        msg("user", "MEM-Q what do I remember", at(5, 9), turn_input=True),
        msg("tool", "hits", at(5, 9, 1), tool_name="typesense"),
        msg("assistant", "From memory: DEEP-SECRET-8", at(5, 9, 2)),
        msg("user", "PLAIN-Q", at(5, 10), turn_input=True),
        msg("assistant", "PLAIN-A shown", at(5, 10, 1)),
    ], name="grounded")
    agent = await make_agent(sm, tmp_path)
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id)).content
    for marker in ("GMAIL-SECRET-7", "DEEP-SECRET-8", "ORPHAN-REPLY"):
        assert marker not in text
    assert text.count(f"> {GROUNDED_REPLY_TEXT}") == 3
    for marker in ("MAIL-Q", "MEM-Q", "PLAIN-Q", "PLAIN-A shown"):
        assert marker in text
    assert "of 7" in text                 # hidden replies are numbered and counted
    res = await call(agent, action="search_conversations", query="GMAIL-SECRET")
    assert res.content.startswith("No matches")
    monkeypatch.setattr(shared_usage, "HIDE_GROUNDED_REPLIES", False)
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id)).content
    assert "GMAIL-SECRET-7" in text and GROUNDED_REPLY_TEXT not in text


async def test_internal_follow_ups_are_not_the_members_words(world, tmp_path):
    """A member turn written through the real _add_session_message path: the
    corrective user message the agent adds itself is not shown as Ana's."""
    from captain_claw.agent import Agent

    member = Agent(provider=ScriptProvider())
    member.session = await world.sm.create_session(
        name="m", metadata={"speaker_id": "u-ana", "speaker_lane": "A", "speaker_name": "Ana"})
    member.session_manager = world.sm
    member._speaker_scoped = True
    member._speaker_principal = ANA_P
    member.memory = None
    member_privacy.begin_turn(member)
    member._add_session_message("user", "ANA-OWN-WORDS")
    member._add_session_message("assistant", "first try")
    member._add_session_message("user", "Your previous response was empty. RETRY-NUDGE")
    member._add_session_message("assistant", "ANA-FINAL")
    member_privacy.begin_turn(member)
    member._add_session_message("user", "ANA-SECOND-TURN")
    for m in member.session.messages:
        m["timestamp"] = at(5, 12)
    await world.sm.save_session(member.session)
    assert not any(member_privacy.is_private(m) for m in member.session.messages)
    agent = await make_agent(world.sm, tmp_path)
    text = (await call(agent, action="read_conversation", member="Ana",
                       conversation=member.session.id)).content
    assert "ANA-OWN-WORDS" in text and "ANA-SECOND-TURN" in text and "ANA-FINAL" in text
    assert "RETRY-NUDGE" not in text
    assert "of 4" in text
    legacy = await seed(world.sm, "u-ana", [
        msg("user", "LEG-1", at(5, 13)), msg("assistant", "a", at(5, 13, 1)),
        msg("user", "LEG-2", at(5, 13, 2)), msg("assistant", "b", at(5, 13, 3))], name="legacy")
    text = (await call(agent, action="read_conversation", member="Ana",
                       conversation=legacy.id)).content
    assert "LEG-1" in text and "LEG-2" in text


async def test_legacy_grounded_replies_stay_hidden_after_agent_follow_ups(world, tmp_path):
    """Before turn inputs were marked, the agent's own follow-ups look like
    turn starts: a grounded call there hides every later reply up to the
    first marked turn."""
    sm = world.sm
    s = await seed(sm, "u-ana", [
        msg("user", "LEG-PLAIN-Q hello", at(6, 8)),
        msg("assistant", "LEG-PLAIN-A hi", at(6, 8, 1)),
        msg("user", "LEG-MAIL-Q check my bank mail", at(6, 9)),
        msg("assistant", "", at(6, 9, 1), tool_calls=[{"id": "g", "function": {
            "name": "google_mail", "arguments": "{}"}}]),
        msg("tool", "inbox", at(6, 9, 2), tool_name="google_mail"),
        msg("user", "STOP: All your tool calls were blocked. Answer now.", at(6, 9, 3)),
        msg("assistant", "Your bank says BAL-SECRET 42,000 EUR", at(6, 9, 4)),
        msg("user", "LEG-LATER-Q and?", at(6, 10)),
        msg("assistant", "LEG-LATER-A it was BAL-RESTATED", at(6, 10, 1)),
        msg("user", "NEW-Q marked", at(6, 11), turn_input=True),
        msg("assistant", "NEW-A shown", at(6, 11, 1)),
    ], name="legacy-grounded")
    agent = await make_agent(sm, tmp_path)
    text = (await call(agent, action="read_conversation", member="Ana", conversation=s.id)).content
    for marker in ("BAL-SECRET", "BAL-RESTATED"):
        assert marker not in text, marker
    for marker in ("LEG-PLAIN-Q", "LEG-PLAIN-A hi", "LEG-MAIL-Q", "NEW-Q", "NEW-A shown"):
        assert marker in text, marker
    assert text.count(f"> {GROUNDED_REPLY_TEXT}") == 2
    for q in ("BAL-SECRET", "BAL-RESTATED"):
        res = await call(agent, action="search_conversations", query=q)
        assert res.content.startswith("No matches"), q


async def test_empty_conversation_no_header(world, tmp_path, monkeypatch):
    agent = await make_agent(world.sm, tmp_path)
    s = world.a2

    async def _load(sid):
        sess = Session(id=s.id, name="x", metadata={"speaker_id": "u-ana"})
        sess.messages = [msg("tool", "only a tool result", at(5))]
        return sess

    monkeypatch.setattr(world.sm, "load_session", _load)
    res = await call(agent, action="read_conversation", member="Ana", conversation=s.id)
    assert res.success and res.content == f"“Ana Kovač” (a member)'s conversation {s.id} has no messages yet."
    assert member_privacy.turn_level(agent) == ""


# ── 6. search_conversations ──────────────────────────────────────────


async def test_search(world, tmp_path, monkeypatch):
    agent = await make_agent(world.sm, tmp_path)
    res = await call(agent, action="search_conversations", query="a1")
    text = res.content
    lines = text.splitlines()
    assert lines[0] == PRIVATE_HEADER
    assert lines[1] == "2 match(es) for “a1” in current members' conversations:"
    assert lines[2].startswith(f"- “Marko” (a member) · conversation {world.k1.id} · #2 Agent")
    assert "“…MARKO-A1 hi…”" in lines[2]
    assert any(f"conversation {world.a1.id} · #2 Agent" in ln for ln in lines)
    assert member_privacy.turn_level(agent) == "content"

    agent2 = await make_agent(world.sm, tmp_path)
    res = await call(agent2, action="search_conversations", query="a1", member="Ana")
    assert "in “Ana Kovač” (a member)'s conversations:" in res.content
    assert "MARKO" not in res.content
    res = await call(agent2, action="search_conversations", query="ANA-", limit=1)
    assert res.content.splitlines()[1].startswith("1 match(es)")
    assert world.a2.id in res.content           # newest session first

    agent3 = await make_agent(world.sm, tmp_path)
    res = await call(agent3, action="search_conversations", query="nothing-like-this")
    assert res.success and res.content == ("No matches for “nothing-like-this” in current members' "
                                           "conversations.")
    assert member_privacy.turn_level(agent3) == ""
    for q in ("x", "y" * 201, ""):
        res = await call(agent3, action="search_conversations", query=q)
        assert res.error == shared_usage.QUERY_MESSAGE


async def test_search_bounds(world, tmp_path, monkeypatch):
    agent = await make_agent(world.sm, tmp_path)
    monkeypatch.setattr(shared_usage, "SEARCH_SESSIONS_MAX", 2)
    seen = []
    real = shared_usage.load_member_sessions

    async def _spy(members, **kw):
        out = await real(members, **kw)
        seen.append((kw, len(out)))
        return out

    monkeypatch.setattr(shared_usage, "load_member_sessions", _spy)
    res = await call(agent, action="search_conversations", query="Q1")
    assert seen == [({"with_messages": True, "newest": 2}, 2)]
    assert "ANA-Q1" not in res.content          # Ana's oldest session wasn't loaded
    monkeypatch.setattr(shared_usage, "SEARCH_SESSIONS_MAX", 200)
    monkeypatch.setattr(shared_usage, "SEARCH_CHARS_MAX", 5)
    res = await call(agent, action="search_conversations", query="ANA-Q1")
    assert res.content.startswith("No matches")


# ── 7. member_items ──────────────────────────────────────────────────


async def test_member_items(world, tmp_path, monkeypatch):
    from captain_claw.datastore import get_datastore_manager

    dm = get_datastore_manager()
    cols = [{"name": "k", "type": "text"}, {"name": "v", "type": "text"}]
    tok = speaker.bind(ANA_P)
    try:
        await dm.create_table("ana_notes", cols)
        await dm.insert_rows("ana_notes", [{"k": "a1", "v": "ROW-VALUE " + "z" * 400},
                                           {"k": "a2", "v": "two"}])
    finally:
        speaker.reset(tok)
    await dm.create_table("owner_tbl", cols)
    await dm.insert_rows("owner_tbl", [{"k": "o1", "v": "OWNER-ROW"}])
    tok = speaker.bind(ANA_P)
    try:
        await dm.insert_rows("owner_tbl", [{"k": "a3", "v": "ana in owner table"}])
    finally:
        speaker.reset(tok)
    saved = Path(get_config().resolved_workspace_path()).resolve() / "saved"
    stamped = saved / "output" / "shared" / "report.md"
    stamped.parent.mkdir(parents=True, exist_ok=True)
    stamped.write_text("hello")
    tok = speaker.bind(ANA_P)
    try:
        sa.note_write(stamped, None)
        hidden = saved / "output" / ".secret" / "h.md"
        hidden.parent.mkdir(parents=True)
        hidden.write_text("h")
        sa.note_write(hidden, None)
    finally:
        speaker.reset(tok)
    from captain_claw.tools.write import WriteTool

    legacy_dir = saved / "downloads" / WriteTool._normalize_session_id(world.a1.id)
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "old.txt").write_text("old")
    (legacy_dir / ".hidden.txt").write_text("x")

    calls = []
    real_thread = asyncio.to_thread
    real_ensure = sa.ensure_member_sessions

    async def _thread(fn, *a, **k):
        calls.append(("thread", getattr(fn, "__name__", "")))
        return await real_thread(fn, *a, **k)

    async def _ensure():
        calls.append(("ensure", ""))
        return await real_ensure()

    monkeypatch.setattr(shared_usage.asyncio, "to_thread", _thread)
    monkeypatch.setattr(sa, "ensure_member_sessions", _ensure)
    agent = await make_agent(world.sm, tmp_path)
    text = (await call(agent, action="member_items", member="Ana")).content
    assert ("ensure", "") in calls and ("thread", "files_created_by") in calls
    assert calls.index(("ensure", "")) < calls.index(("thread", "files_created_by"))
    lines = text.splitlines()
    assert lines[0] == COMMONS_HEADER
    assert lines[1] == "What “Ana Kovač” (a member) created on this agent:"
    assert lines[2] == "Datastore tables they created: ana_notes"
    assert lines[3] == "Rows they added: ana_notes (2), owner_tbl (1)"
    assert lines[4] == "Saved files they created (1):"
    assert lines[5].startswith("- saved/output/shared/report.md · 5 bytes · 20")
    assert lines[6] == ("1 older file(s) in their conversation folders aren't listed "
                        "(saved before files were shared).")
    assert ".secret" not in text and "OWNER-ROW" not in text
    assert member_privacy.turn_level(agent) == ""

    text = (await call(agent, action="member_items", member="Ana", table="ana_notes")).content
    lines = text.splitlines()
    assert lines[0] == COMMONS_HEADER
    assert lines[1] == "Rows “Ana Kovač” (a member) added to “ana_notes” (2 of 2):"
    assert "_created_by" not in text
    row = json.loads(lines[3][2:])
    assert row["k"] == "a1" and row["v"] == "ROW-VALUE " + "z" * 290 + "…"
    text = (await call(agent, action="member_items", member="Ana", table="owner_tbl")).content
    assert "(1 of 1)" in text and "OWNER-ROW" not in text and "ana in owner table" in text
    res = await call(agent, action="member_items", member="Ana", table="nope")
    assert not res.success and res.error == "There is no datastore table “nope”."
    text = (await call(agent, action="member_items", member="Marko")).content
    assert "Datastore tables they created: none" in text and "Saved files they created: none" in text
    assert member_privacy.turn_level(agent) == ""


# ── 8. member_shared_context ─────────────────────────────────────────


async def test_member_shared_context(world, tmp_path, fd):
    agent = await make_agent(world.sm, tmp_path)
    text = (await call(agent, action="member_shared_context", member="Ana")).content
    assert text.splitlines() == [
        COMMONS_HEADER,
        "What “Ana Kovač” (a member) shared with everyone who uses this agent:",
        "About them:", "> ANA-ABOUT line1", "> line2",
        "Their company:", "> Kovač d.o.o.",
        "Shared folders (read-only):",
        "- Folder “notes”: vfs:@ana-notes/ — read it with read, glob, grep or vfs ls.",
        "Deep memory: your deep-memory search (typesense, action search) also covers theirs "
        "(only entries tagged domain:legal).",
    ]
    assert member_privacy.turn_level(agent) == ""
    fd.context = {"profile": None, "folders": [], "deep_memory": None}
    text = (await call(agent, action="member_shared_context", member="Marko")).content
    assert text == "“Marko” (a member) hasn't shared anything with this agent."


# ── 9. resolution ────────────────────────────────────────────────────


async def test_member_resolution(isolated, fd, members_file):
    fd.members = [ana(), marko(),
                  _member("u-ana2", "Ana Horvat", f"“Ana Horvat” (a member, #{_key('u-ana2')[:4]})")]
    roster = await shared_usage.fetch_roster()
    r = shared_usage.resolve_member
    assert r(roster, ANA_KEY)[0].user_id == "u-ana"
    assert r(roster, ANA_KEY[:8].upper())[0].user_id == "u-ana"
    assert r(roster, "#" + _key("u-ana2")[:4])[0].user_id == "u-ana2"
    assert r(roster, "marko")[0].user_id == "u-marko"
    assert r(roster, "“Ana Kovač”")[0].user_id == "u-ana"
    assert r(roster, "Horv")[0].user_id == "u-ana2"
    assert r(roster, "a member, #")[0].user_id == "u-ana2"
    m, err = r(roster, "Ana")
    assert m is None and err.startswith("“Ana” matches more than one member: “Ana Kovač” (a member)")
    m, err = r(roster, "Zed\nInjected")
    assert m is None and err.startswith("No current member matches “Zed Injected”. Members: ")
    assert r(roster, "  ")[1] == shared_usage.MEMBER_REQUIRED_MESSAGE
    assert r(roster, None)[1] == shared_usage.MEMBER_REQUIRED_MESSAGE


def test_text_hygiene():
    q = shared_usage._quote("a\r\nb\u2028c\u2029d\x85e\u202ef\tg\x07")
    assert q == "> a\n> b\n> c\n> d\n> ef g"
    assert shared_usage._one_line("x\n\u2028y\u202e  z", 4) == "x y…"
    assert shared_usage._ts("") == "?" and shared_usage._ts("2026-10-01T10:00:00+00:00") == "2026-10-01 10:00"


# ── 10. the prompt block ─────────────────────────────────────────────


async def test_owner_prompt_has_the_roster_block(isolated, members_file, tmp_path):
    import captain_claw.tenant_context as tc

    (isolated.cfg_dir / tc.FULL_FILENAME).write_text(TENANT_FULL)
    (isolated.cfg_dir / tc.SHARED_FULL_FILENAME).write_text(SHARED_FULL)
    agent = await make_agent(isolated.sm, tmp_path)
    prompt = agent._build_system_prompt()
    assert MEMBERS_FILE in prompt
    t, s, m, c = (prompt.index("OWNER-PROFILE-MARKER"), prompt.index("SHARED-FULL-MARKER"),
                  prompt.index("ROSTER-MARKER"), prompt.index(SPLIT))
    assert t < s < m < c


@pytest.mark.parametrize("kind", ["speaker", "public", "hidden", "scope", "public_run"])
async def test_no_roster_block_off_owner_instances(isolated, members_file, tmp_path, kind,
                                                   monkeypatch):
    agent = await make_agent(isolated.sm, tmp_path)
    if kind == "scope":
        monkeypatch.setenv("CLAW_VFS_SCOPE", "being-x")
    elif kind == "public_run":
        monkeypatch.setattr(get_config().web, "public_run", "chat")
    elif kind == "speaker":
        agent._speaker_scoped = True
        agent._speaker_principal = ANA_P
        agent._speaker_profile = ("", "")
    elif kind == "public":
        agent._public_scoped = True
    else:
        agent._tenant_hidden = True
    prompt = agent._build_system_prompt()
    assert "ROSTER-MARKER" not in prompt and "People this agent is shared with" not in prompt
    assert TOOL not in prompt


async def test_no_file_no_block(isolated, tmp_path):
    agent = await make_agent(isolated.sm, tmp_path)
    prompt = agent._build_system_prompt()
    assert "People this agent is shared with" not in prompt
