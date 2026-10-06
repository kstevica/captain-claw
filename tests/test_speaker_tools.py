"""Shared-agent member tool enforcement (contract part 2b §7; A2 part 2 §3).

A member turn may only run ``speaker.allowed_tools(principal)`` — A1's
``SPEAKER_TOOL_ALLOWLIST`` for a docker / unverified member, up to
``SPEAKER_TOOL_ALLOWLIST_MAX`` (their own Google, deep memory and files, with
a grant and confined paths) for a verified member of a process agent —
enforced inside ``ToolRegistry.execute`` (so the classic loop,
``run_tool``-style paths and Mrav all hit it) from three independent signals:
the speaker contextvar, a registered speaker session key, and a
speaker-scoped ``_agent``. Each signal alone is enough, and no session/task
policy can widen the allowlist.
"""

from __future__ import annotations

import asyncio
import functools
import types
from unittest.mock import AsyncMock, MagicMock

import httpcore
import pytest

from captain_claw import speaker
from captain_claw.exceptions import ToolBlockedError
from captain_claw.llm import LLMProvider, LLMResponse
from captain_claw.speaker import (
    SPEAKER_TOOL_ALLOWLIST,
    SPEAKER_TOOL_ALLOWLIST_MAX,
    Principal,
)
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

PRINCIPAL = Principal("u-member", "Ana", "Olga", "A", "process:helper:0123456789abcdef")
DOCKER = Principal("u-member", "Ana", "Olga", "A", "docker:helper:0123456789abcdef")
SPK_SESSION = "spk-session-1"
GRANT = "g" * 43

# Never for a member, under any signal or widening policy (A1 and A2).
BLOCKED = ["shell", "history", "flight_deck", "cron", "mcp_x_y", "web_get", "web_fetch_batch"]
# A2: a process member's own Google / deep memory / files — still blocked for
# a docker or unverified member, and without a grant or confined paths.
A2_TOOLS = ["read", "write", "vfs", "google_mail", "typesense"]
SIGNALS = ["contextvar", "session_key", "agent"]

_DB_FIELDS = (
    ("memory", "path"), ("session", "path"), ("insights", "db_path"),
    ("conversation_topics", "db_path"), ("nervous_system", "db_path"),
    ("sister_session", "db_path"), ("cognitive_metrics", "db_path"),
    ("datastore", "path"), ("autonomous_work", "db_path"),
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Nothing here may reach the real ~/.captain-claw or a real FD data dir
    (a real Agent is built below): HOME, FD_DATA_DIR, every config DB path and
    the global session / topic managers point at tmp first."""
    import captain_claw.conversation_topics as _ct
    from captain_claw import session as _session
    from captain_claw.config import get_config

    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    for var in ("CLAW_VFS_ROOT", "CLAW_VFS_USER", "FD_OWNER_ID", "FD_URL"):
        monkeypatch.delenv(var, raising=False)
    cfg = get_config()
    for section, attr in _DB_FIELDS:
        monkeypatch.setattr(getattr(cfg, section), attr, str(home / ".captain-claw" / f"{section}.db"))
    monkeypatch.setattr(_session, "_manager", _session.SessionManager(home / ".captain-claw" / "s.db"))
    monkeypatch.setattr(_ct, "_MANAGER", None)
    monkeypatch.setattr(speaker, "_IN_FLIGHT", 0)
    monkeypatch.setattr(speaker, "_MAIN_LOOP", None)
    return home


class _Rec(Tool):
    """A tool that records what it was called with and never does anything."""

    def __init__(self, name: str):
        self.name = name
        self.description = name
        self.parameters = {"type": "object", "properties": {}, "required": []}
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(success=True, content=f"{self.name} ran")


def _speaker_agent(session=None, principal=PRINCIPAL, grant=""):
    return types.SimpleNamespace(
        _speaker_scoped=True, _speaker_principal=principal, _turn_grant=grant,
        session=session or types.SimpleNamespace(id=SPK_SESSION),
    )


def _registry(names) -> tuple[ToolRegistry, dict[str, _Rec]]:
    reg = ToolRegistry()
    tools = {n: _Rec(n) for n in names}
    for t in tools.values():
        reg.register(t)
    return reg, tools


async def _call(reg: ToolRegistry, name: str, signal: str, args: dict | None = None,
                principal: Principal = PRINCIPAL, grant: str = "", **kw):
    """Run *name* through the registry as a member, by exactly ONE signal."""
    arguments = dict(args or {})
    session_id = kw.pop("session_id", "owner-session")
    tok = gtok = None
    if signal == "contextvar":
        tok = speaker.bind(principal)
        gtok = speaker.bind_grant(grant)
    elif signal == "session_key":
        reg.register_speaker_session(SPK_SESSION)
        session_id = SPK_SESSION
    elif signal == "agent":
        arguments["_agent"] = _speaker_agent(principal=principal, grant=grant)
    try:
        return await reg.execute(name, arguments, session_id=session_id, **kw)
    finally:
        if gtok is not None:
            speaker.reset_grant(gtok)
        if tok is not None:
            speaker.reset(tok)


def _expected_names(signal: str, principal: Principal = PRINCIPAL) -> frozenset[str]:
    """What a member may run by this signal alone: the session key carries no
    principal (A1's set, fail closed); the contextvar / `_agent` carry one."""
    return speaker.allowed_tools(None if signal == "session_key" else principal)


# ── the allowlist itself ─────────────────────────────────────────────


def test_allowlist_is_exactly_the_contract():
    assert SPEAKER_TOOL_ALLOWLIST == frozenset(
        {"insights", "playbooks", "topics", "web_search", "web_fetch"}
    )
    assert SPEAKER_TOOL_ALLOWLIST_MAX == SPEAKER_TOOL_ALLOWLIST | {
        "google_mail", "google_drive", "google_calendar", "typesense", "read", "write",
        "edit", "glob", "grep", "vfs", "pdf_extract", "docx_extract", "xlsx_extract",
        "pptx_extract", "datastore",
    }


@pytest.mark.parametrize("principal", [PRINCIPAL, DOCKER], ids=["process", "docker"])
@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("name", BLOCKED)
async def test_blocked_tools_are_refused_for_a_member(name, signal, principal):
    reg, tools = _registry(BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST))
    with pytest.raises(ToolBlockedError):
        await _call(reg, name, signal, principal=principal, grant=GRANT)
    assert tools[name].calls == []


@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("name", A2_TOOLS)
async def test_a2_tools_are_refused_for_docker_and_unverified_members(name, signal):
    """Docker / unknown-runtime members stay A1 even with a grant; the session
    key alone is an unverified member (A1)."""
    reg, tools = _registry(BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST))
    with pytest.raises(ToolBlockedError):
        await _call(reg, name, signal, {"action": "search", "path": "vfs:p/a.md"},
                    principal=DOCKER, grant=GRANT)
    assert tools[name].calls == []


@pytest.mark.parametrize("signal", ["contextvar", "agent"])
@pytest.mark.parametrize("name,why", [
    ("google_mail", speaker.NO_GRANT_MESSAGE), ("typesense", speaker.NO_GRANT_MESSAGE),
    ("read", speaker.FILES_UNAVAILABLE_MESSAGE), ("write", speaker.FILES_UNAVAILABLE_MESSAGE),
    ("vfs", speaker.FILES_UNAVAILABLE_MESSAGE),
])
async def test_a2_tools_need_a_grant_or_confined_paths(name, why, signal):
    """A process member without a grant (Google / deep memory) or without
    their own instance + session (files) is refused before the tool runs."""
    reg, tools = _registry(BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST))
    args = {"action": "ls" if name == "vfs" else "search", "path": "vfs:p/a.md",
            "content": "x", "query": "q"}
    with pytest.raises(ToolBlockedError) as info:
        await _call(reg, name, signal, args, principal=PRINCIPAL, grant="")
    assert info.value.reason == why
    assert tools[name].calls == []


@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("widen", ["task_also_allow", "session_policy_arg", "per_turn_policy"])
@pytest.mark.parametrize("name", BLOCKED)
async def test_no_policy_can_widen_the_member_allowlist(name, signal, widen):
    reg, tools = _registry(BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST))
    kw: dict = {}
    if widen == "task_also_allow":
        kw["task_policy"] = {"also_allow": [name]}
    elif widen == "session_policy_arg":
        kw["session_policy"] = {"allow": [name], "also_allow": [name]}
    else:
        for sid in ("owner-session", SPK_SESSION):
            reg.set_session_policy(sid, {"allow": [name], "also_allow": [name]})
    with pytest.raises(ToolBlockedError):
        await _call(reg, name, signal, **kw)
    assert tools[name].calls == []


@pytest.mark.parametrize("signal", SIGNALS)
async def test_the_chain_step_alone_also_blocks(signal):
    """Even past the top-of-execute check, the policy chain carries the
    principal step last (list_tools / get_definitions use it)."""
    registered = BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST)
    reg, _ = _registry(registered)
    if signal == "agent":
        pytest.skip("listing carries no arguments; covered by execute()")
    tok = speaker.bind(PRINCIPAL) if signal == "contextvar" else None
    if signal == "session_key":
        reg.register_speaker_session(SPK_SESSION)
    try:
        names = reg.list_tools(
            session_id=SPK_SESSION,
            task_policy={"also_allow": ["shell", "vfs"]},
            session_policy={"also_allow": ["history"]},
        )
    finally:
        if tok is not None:
            speaker.reset(tok)
    assert set(names) == _expected_names(signal) & set(registered)
    assert not set(names) & set(BLOCKED)


@pytest.mark.parametrize("signal", SIGNALS)
async def test_allowlisted_tools_still_run_for_a_member(signal):
    reg, tools = _registry(BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST))
    result = await _call(reg, "web_search", signal, {"query": "weather"})
    assert result.success and len(tools["web_search"].calls) == 1


async def test_without_any_signal_nothing_changes():
    reg, tools = _registry(BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST))
    assert reg._is_speaker_call("owner-session", {}) is False
    result = await reg.execute("read", {}, session_id="owner-session")
    assert result.success and len(tools["read"].calls) == 1
    assert "read" in reg.list_tools(session_id="owner-session")


async def test_a_mocked_agent_is_not_a_speaker():
    """A MagicMock `_agent` has every attribute — the flag must be `is True`."""
    reg, tools = _registry(["read"])
    result = await reg.execute("read", {"_agent": MagicMock()}, session_id="s")
    assert result.success and len(tools["read"].calls) == 1


def test_speaker_session_keys_unregister():
    reg = ToolRegistry()
    reg.register_speaker_session(SPK_SESSION)
    assert reg._is_speaker_call(SPK_SESSION) is True
    reg.unregister_speaker_session(SPK_SESSION)
    assert reg._is_speaker_call(SPK_SESSION) is False


def test_definitions_for_a_member_session_are_only_the_allowlist():
    reg, _ = _registry(BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST))
    reg.register_speaker_session(SPK_SESSION)
    names = {d["name"] for d in reg.get_definitions(session_id=SPK_SESSION)}
    assert names == set(SPEAKER_TOOL_ALLOWLIST)
    # The owner's session in the same process is untouched.
    owner = {d["name"] for d in reg.get_definitions(session_id="owner-session")}
    assert "shell" in owner and "history" in owner


@pytest.mark.parametrize("principal", [PRINCIPAL, DOCKER], ids=["process", "docker"])
def test_definitions_under_the_contextvar_are_only_the_allowlist(principal):
    registered = BLOCKED + A2_TOOLS + sorted(SPEAKER_TOOL_ALLOWLIST)
    reg, _ = _registry(registered)
    tok = speaker.bind(principal)
    try:
        names = {d["name"] for d in reg.get_definitions()}
    finally:
        speaker.reset(tok)
    assert names == speaker.allowed_tools(principal) & set(registered)
    if principal is DOCKER:
        assert names == set(SPEAKER_TOOL_ALLOWLIST)


# ── every tool there is ──────────────────────────────────────────────


def _every_tool_name() -> list[str]:
    import captain_claw.tools as pkg

    names: set[str] = set()
    for attr in dir(pkg):
        obj = getattr(pkg, attr)
        if isinstance(obj, type) and issubclass(obj, Tool) and obj is not Tool:
            n = getattr(obj, "name", "")
            if isinstance(n, str) and n:
                names.add(n)
    names |= {"mcp_x_y", "mcp_github_create_issue", "flight_deck", "consult_peer"}
    return sorted(names)


@pytest.mark.parametrize("principal", [PRINCIPAL, DOCKER], ids=["process", "docker"])
@pytest.mark.parametrize("signal", SIGNALS)
async def test_every_known_tool_outside_the_allowlist_is_blocked(signal, principal):
    names = _every_tool_name()
    assert len(names) > 40
    reg, tools = _registry(names)
    allowed = _expected_names(signal, principal)
    for name in names:
        if name in allowed:
            continue
        with pytest.raises(ToolBlockedError):
            await _call(reg, name, signal, principal=principal, grant=GRANT,
                        task_policy={"also_allow": [name]})
        assert tools[name].calls == [], name


class _Provider(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) or 1


async def test_every_default_registered_tool_is_blocked(monkeypatch):
    """The real registry an agent builds: every registered tool outside the
    allowlist is refused for a member before its execute() is reached."""
    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    from captain_claw.agent import Agent

    agent = Agent(provider=_Provider())
    agent._register_default_tools()
    reg = agent.tools
    registered = list(reg._tools.items())
    assert registered
    ran: list[str] = []
    for name, tool in registered:
        async def _never(_n=name, **kwargs):
            ran.append(_n)
            return ToolResult(success=True)
        object.__setattr__(tool, "execute", _never)
    for name, _tool in registered:
        for principal in (PRINCIPAL, DOCKER):
            for signal in SIGNALS:
                if name in _expected_names(signal, principal):
                    continue
                with pytest.raises(ToolBlockedError):
                    await _call(reg, name, signal, principal=principal, grant=GRANT,
                                task_policy={"also_allow": [name]})
    assert ran == []


# ── classic runtime only ─────────────────────────────────────────────


def test_speaker_instances_never_use_mrav(monkeypatch):
    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    monkeypatch.setattr("captain_claw.agent._mrav_flag_state", lambda path=None: True)
    from captain_claw.agent import Agent

    agent = Agent(provider=_Provider())
    assert agent._mrav_enabled() is True           # the flag is on…
    agent._speaker_scoped = True
    assert agent._mrav_enabled() is False          # …and a member never gets it


async def test_speaker_complete_takes_the_classic_path(monkeypatch):
    monkeypatch.setattr("captain_claw.tools.registry._registry", None)
    monkeypatch.setattr("captain_claw.agent._mrav_flag_state", lambda path=None: True)
    from captain_claw.agent import Agent
    from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

    agent = Agent(provider=_Provider())
    agent._speaker_scoped = True
    agent._mrav_complete = AsyncMock(side_effect=AssertionError("mrav used"))
    classic = AsyncMock(return_value="classic reply")
    monkeypatch.setattr(AgentOrchestrationMixin, "complete", classic, raising=False)
    assert await agent.complete("hi") == "classic reply"


# ── per-tool rules ───────────────────────────────────────────────────


@pytest.mark.parametrize("action", ["update", "delete", "UPDATE"])
async def test_insights_update_and_delete_are_owner_only(action):
    reg, tools = _registry(["insights"])
    with pytest.raises(ToolBlockedError) as info:
        await _call(reg, "insights", "agent", {"action": action, "insight_id": "i1"})
    assert "owner" in str(info.value)
    assert tools["insights"].calls == []


@pytest.mark.parametrize("action", ["add", "search", "list"])
async def test_insights_read_and_add_are_open(action):
    reg, tools = _registry(["insights"])
    result = await _call(reg, "insights", "agent", {"action": action, "content": "x", "query": "x"})
    assert result.success and len(tools["insights"].calls) == 1


async def test_insights_unknown_action_is_refused():
    reg, tools = _registry(["insights"])
    with pytest.raises(ToolBlockedError):
        await _call(reg, "insights", "contextvar", {"action": "purge"})
    assert tools["insights"].calls == []


@pytest.mark.parametrize("action", ["update", "remove"])
async def test_playbooks_update_and_remove_are_owner_only(action):
    reg, tools = _registry(["playbooks"])
    with pytest.raises(ToolBlockedError):
        await _call(reg, "playbooks", "agent", {"action": action, "playbook_id": "p1"})
    assert tools["playbooks"].calls == []


async def test_topics_only_reads():
    reg, tools = _registry(["topics"])
    for action in ("list", "search", "get"):
        assert (await _call(reg, "topics", "agent", {"action": action})).success
    with pytest.raises(ToolBlockedError):
        await _call(reg, "topics", "agent", {"action": "merge"})
    assert len(tools["topics"].calls) == 3


async def test_web_fetch_deep_fetch_is_forced_off():
    reg, tools = _registry(["web_fetch"])
    await _call(reg, "web_fetch", "agent", {"url": "https://example.com", "deep_fetch": True})
    assert tools["web_fetch"].calls[0]["deep_fetch"] is False


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x", "gopher://x", "http://user:pw@example.com/",
    "https://token@example.com/", "http:///nohost", "", None,
])
async def test_web_fetch_rejects_non_public_url_shapes(url):
    reg, tools = _registry(["web_fetch"])
    with pytest.raises(ToolBlockedError):
        await _call(reg, "web_fetch", "agent", {"url": url})
    assert tools["web_fetch"].calls == []


def test_an_allowlisted_name_without_a_rule_is_refused():
    _, err = speaker.apply_tool_rules("brand_new_tool", {}, _speaker_agent())
    assert err


# ── playbooks: own session only, no source session ───────────────────


@pytest.fixture
async def sm(tmp_path, monkeypatch):
    from captain_claw.session import SessionManager

    manager = SessionManager(tmp_path / "sessions.db")
    monkeypatch.setattr("captain_claw.tools.playbooks.get_session_manager", lambda: manager)
    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: manager)
    yield manager
    await manager.close()


async def test_playbooks_rate_only_writes_the_members_own_session(sm, monkeypatch):
    from captain_claw.tools.playbooks import PlaybooksTool

    monkeypatch.setattr(
        "captain_claw.tools.playbooks._distill_session_standalone", AsyncMock(return_value=None),
    )
    owner = await sm.create_session(name="owner-chat")
    member = await sm.create_session(name="spk-ana-A", metadata={"speaker_id": "u-member"})
    reg = ToolRegistry()
    reg.register(PlaybooksTool())

    result = await reg.execute(
        "playbooks",
        {"action": "rate", "rating": "bad", "session_id": owner.id,
         "_agent": _speaker_agent(session=member)},
        session_id="anything",
    )
    assert result.success
    assert "playbook_rating" not in (await sm.load_session(owner.id)).metadata
    assert (await sm.load_session(member.id)).metadata["playbook_rating"] == "bad"


async def test_playbooks_rate_without_a_member_session_is_refused(sm):
    from captain_claw.tools.playbooks import PlaybooksTool

    reg = ToolRegistry()
    reg.register(PlaybooksTool())
    tok = speaker.bind(PRINCIPAL)          # contextvar only, no `_agent`
    try:
        with pytest.raises(ToolBlockedError):
            await reg.execute("playbooks", {"action": "rate", "rating": "good",
                                            "session_id": "owner"}, session_id="x")
    finally:
        speaker.reset(tok)


async def test_playbooks_info_hides_the_source_session_from_a_member(sm):
    from captain_claw.tools.playbooks import PlaybooksTool

    item = await sm.create_playbook(
        name="Research flow", task_type="web-research", do_pattern="search then fetch",
        source_session="owner-private-session-42",
    )
    tool = PlaybooksTool()
    as_member = await tool.execute(action="info", playbook_id=item.id, _agent=_speaker_agent())
    as_owner = await tool.execute(action="info", playbook_id=item.id, _agent=None)
    assert as_member.success and "Research flow" in as_member.content
    assert "Source session" not in as_member.content
    assert "owner-private-session-42" not in as_member.content
    assert "Source session: owner-private-session-42" in as_owner.content


# ── topics: commons labels, private excerpts ─────────────────────────


class _Topics:
    """A1 hid every excerpt from members; A2 shows a member only THEIR OWN
    (`speaker=` narrows the rows, as ConversationTopicsManager.get_topic)."""

    MESSAGES = [
        {"ts": "2026-10-01T10:00", "role": "user", "speaker": "",
         "excerpt": "SECRET EXCERPT from someone's chat"},
        {"ts": "2026-10-01T10:05", "role": "user", "speaker": "u-member",
         "excerpt": "ANA OWN EXCERPT"},
    ]

    def get_topic(self, topic, max_excerpts=40, speaker=None):
        rows = [m for m in self.MESSAGES if speaker is None or m["speaker"] == speaker]
        return {
            "id": "t1", "label": "Munich trip", "summary": "Planning the Munich trip",
            "keywords": "travel,munich", "msg_count": 2,
            "messages": rows,
        }


@pytest.mark.parametrize("member_via", ["agent", "contextvar"])
async def test_topics_get_shows_no_excerpts_to_a_member(monkeypatch, member_via):
    from captain_claw.tools.conversation_topics import TopicsTool

    monkeypatch.setattr("captain_claw.tools.conversation_topics.get_topics_manager", _Topics)
    tool = TopicsTool()
    tok = speaker.bind(PRINCIPAL) if member_via == "contextvar" else None
    try:
        kwargs = {"_agent": _speaker_agent()} if member_via == "agent" else {}
        res = await tool.execute(action="get", topic="t1", **kwargs)
    finally:
        if tok is not None:
            speaker.reset(tok)
    assert res.success
    assert "Munich trip" in res.content and "Planning the Munich trip" in res.content
    assert "SECRET EXCERPT" not in res.content
    assert "ANA OWN EXCERPT" in res.content            # A2: their own excerpts


async def test_topics_get_still_shows_excerpts_to_the_owner(monkeypatch):
    from captain_claw.tools.conversation_topics import TopicsTool

    monkeypatch.setattr("captain_claw.tools.conversation_topics.get_topics_manager", _Topics)
    res = await TopicsTool().execute(action="get", topic="t1")
    assert "SECRET EXCERPT" in res.content


# ── SSRF-safe web_fetch ──────────────────────────────────────────────


class _Stream(httpcore.AsyncNetworkStream):
    def __init__(self, chunks: list[bytes], log: list):
        self._chunks = list(chunks)
        self._log = log

    async def read(self, max_bytes, timeout=None):
        return self._chunks.pop(0) if self._chunks else b""

    async def write(self, buffer, timeout=None):
        return None

    async def aclose(self):
        return None

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self._log.append(("tls", server_hostname))
        return self

    def get_extra_info(self, info):
        return None


class _FakeNet(httpcore.AsyncNetworkBackend):
    """Inner backend: records every connect and replays canned responses."""

    def __init__(self, responses: list[list[bytes]]):
        self.responses = list(responses)
        self.log: list = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.log.append(("tcp", host, port))
        return _Stream(self.responses.pop(0) if self.responses else [], self.log)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise AssertionError("unix socket")

    async def sleep(self, seconds):
        return None


def _http(status: str, headers: dict[str, str], body: bytes = b"") -> list[bytes]:
    head = f"HTTP/1.1 {status}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    head += f"Content-Length: {len(body)}\r\n\r\n"
    return [head.encode() + body]


@pytest.fixture
def no_deep_fetch(monkeypatch):
    async def _boom(*a, **k):
        raise AssertionError("a member fetch must never launch a browser")
    monkeypatch.setattr("captain_claw.tools.web_fetch._deep_fetch", _boom)


@pytest.fixture
def fake_net(monkeypatch):
    """Route member fetches through a recording inner backend; public DNS
    for public.example / mixed.example, nothing else resolves."""
    net = _FakeNet([])
    real = speaker.make_public_http_client
    monkeypatch.setattr(speaker, "make_public_http_client",
                        functools.partial(real, network_backend=net))

    async def _dns(host, port):
        table = {
            "public.example": ["93.184.216.34"],
            "mixed.example": ["93.184.216.34", "10.0.0.5"],
            "rebind.example": ["127.0.0.1"],
        }
        if host not in table:
            raise OSError("no such host")
        return table[host]

    monkeypatch.setattr(speaker, "_getaddrinfo", _dns)
    return net


class _NoSharedClient:
    """The tool's shared (unrestricted) client must never serve a member — and
    if a regression routes one there, fail fast instead of touching the network."""

    async def get(self, url, *a, **k):
        raise AssertionError("member fetch went through the shared client")


async def _member_fetch(url: str, **extra):
    from captain_claw.tools.web_fetch import WebFetchTool

    reg = ToolRegistry()
    tool = WebFetchTool()
    tool.client = _NoSharedClient()
    reg.register(tool)
    return await reg.execute(
        "web_fetch", {"url": url, "deep_fetch": True, "_agent": _speaker_agent(), **extra},
        session_id="x",
    )


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:25080/", "http://10.0.0.1/",
    "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
    "http://[::ffff:127.0.0.1]/", "http://0.0.0.0:8080/", "http://192.168.1.1/",
    "http://172.16.0.1/", "http://[fc00::1]/", "http://100.64.0.1/",
    "http://sub.localhost/", "http://rebind.example/", "http://mixed.example/",
])
async def test_member_web_fetch_refuses_private_targets(url, fake_net, no_deep_fetch):
    result = await _member_fetch(url)
    assert result.success is False
    assert "not a public address" in (result.error or "") or "non-public" in (result.error or "")
    assert [e for e in fake_net.log if e[0] == "tcp"] == []      # never connected


async def test_member_web_fetch_refuses_a_redirect_to_loopback(fake_net, no_deep_fetch):
    fake_net.responses = [_http("302 Found", {"Location": "http://127.0.0.1:25080/fd/agents"})]
    result = await _member_fetch("http://public.example/start")
    assert result.success is False
    assert "127.0.0.1" in (result.error or "")
    # Exactly one connection: the public hop. The loopback hop never opened.
    assert [e for e in fake_net.log if e[0] == "tcp"] == [("tcp", "93.184.216.34", 80)]


async def test_member_web_fetch_reads_a_public_page(fake_net, no_deep_fetch):
    body = b"<html><head><title>Hello</title></head><body><p>Public text</p></body></html>"
    fake_net.responses = [_http("200 OK", {"Content-Type": "text/html"}, body)]
    result = await _member_fetch("http://public.example/")
    assert result.success is True
    assert "Public text" in result.content and "[Mode: text]" in result.content
    assert fake_net.log == [("tcp", "93.184.216.34", 80)]


async def test_member_https_keeps_the_hostname_for_sni(fake_net, no_deep_fetch):
    fake_net.responses = [_http("200 OK", {"Content-Type": "text/html"}, b"<p>ok</p>")]
    result = await _member_fetch("https://public.example/")
    assert result.success is True
    assert fake_net.log[0] == ("tcp", "93.184.216.34", 443)    # connects to the checked IP
    assert ("tls", "public.example") in fake_net.log             # …but TLS names the host


async def test_member_fetch_never_writes_the_corpus(fake_net, no_deep_fetch, monkeypatch):
    monkeypatch.setenv("CLAW_SOURCE_CORPUS", "1")
    saved = []
    monkeypatch.setattr("captain_claw.tools.web_fetch._save_source_to_corpus",
                        lambda url, content: saved.append(url) or None)
    fake_net.responses = [_http("200 OK", {"Content-Type": "text/html"}, b"<p>ok</p>")]
    assert (await _member_fetch("http://public.example/")).success
    assert saved == []


@pytest.mark.parametrize("addr,public", [
    ("8.8.8.8", True), ("2606:4700::1111", True),
    ("127.0.0.1", False), ("10.1.2.3", False), ("172.31.255.255", False),
    ("192.168.0.1", False), ("169.254.169.254", False), ("100.64.0.1", False),
    ("0.0.0.0", False), ("255.255.255.255", False), ("224.0.0.1", False),
    ("::1", False), ("::", False), ("fc00::1", False), ("fe80::1", False),
    ("::ffff:127.0.0.1", False), ("::ffff:8.8.8.8", True),
    ("64:ff9b::7f00:1", False), ("2002:7f00:1::1", False),
])
def test_public_address_classification(addr, public):
    import ipaddress

    assert speaker._is_public_ip(ipaddress.ip_address(addr)) is public


async def test_public_client_settings():
    client = speaker.make_public_http_client()
    try:
        assert client.trust_env is False
        assert client.max_redirects == 5
        assert client.follow_redirects is True
        assert isinstance(client._transport._pool._network_backend, speaker._PublicOnlyBackend)
    finally:
        await client.aclose()


async def test_public_backend_refuses_unix_sockets():
    with pytest.raises(httpcore.ConnectError):
        await speaker._PublicOnlyBackend(_FakeNet([])).connect_unix_socket("/tmp/x.sock")


async def test_owner_web_fetch_is_unchanged(monkeypatch):
    """Without a member signal the shared client (and deep fetch) are used."""
    from captain_claw.tools.web_fetch import WebFetchTool

    called = {}

    async def _deep(url, *a, **k):
        called["deep"] = url
        return "<p>deep</p>"

    monkeypatch.setattr("captain_claw.tools.web_fetch._deep_fetch", _deep)
    monkeypatch.setattr(speaker, "make_public_http_client",
                        lambda **k: (_ for _ in ()).throw(AssertionError("member client")))
    result = await WebFetchTool().execute(url="http://127.0.0.1:9/", deep_fetch=True)
    assert result.success and called["deep"] == "http://127.0.0.1:9/"


# ── every signal reaches the tool's own member rules ─────────────────


class _Probe(Tool):
    """Records whether the tool task saw a member principal."""

    def __init__(self, name: str):
        self.name = name
        self.description = name
        self.parameters = {"type": "object", "properties": {}, "required": []}
        self.seen: list = []

    async def execute(self, **kwargs):
        self.seen.append(speaker.principal_for(kwargs.get("_agent")))
        return ToolResult(success=True)


@pytest.mark.parametrize("signal", SIGNALS)
async def test_the_tool_task_sees_a_member_on_every_signal(signal):
    """The registry binds a principal for the tool task whichever signal fired,
    so a tool's own member rules (public-only fetch, hidden excerpts/sources)
    hold even when the contextvar was lost."""
    reg = ToolRegistry()
    probe = _Probe("topics")
    reg.register(probe)
    await _call(reg, "topics", signal, {"action": "list"})
    assert probe.seen and probe.seen[0] is not None
    assert speaker.current() is None                  # nothing leaks out of the call


async def test_an_owner_tool_task_sees_no_member():
    reg = ToolRegistry()
    probe = _Probe("topics")
    reg.register(probe)
    await reg.execute("topics", {"action": "list"}, session_id="owner-session")
    assert probe.seen == [None]


async def test_topics_get_by_session_key_alone_shows_no_excerpts(monkeypatch):
    from captain_claw.tools.conversation_topics import TopicsTool

    monkeypatch.setattr("captain_claw.tools.conversation_topics.get_topics_manager", _Topics)
    reg = ToolRegistry()
    reg.register(TopicsTool())
    reg.register_speaker_session(SPK_SESSION)
    res = await reg.execute("topics", {"action": "get", "topic": "t1"}, session_id=SPK_SESSION)
    assert res.success and "Munich trip" in res.content
    assert "SECRET EXCERPT" not in res.content


async def test_member_web_fetch_by_session_key_alone_stays_public_only(fake_net, no_deep_fetch):
    """No contextvar, no `_agent` — only the registered session key. The fetch
    must still go through the public-only client, never the shared one."""
    from captain_claw.tools.web_fetch import WebFetchTool

    reg = ToolRegistry()
    tool = WebFetchTool()
    tool.client = _NoSharedClient()
    reg.register(tool)
    reg.register_speaker_session(SPK_SESSION)
    result = await reg.execute(
        "web_fetch", {"url": "http://127.0.0.1:25080/fd/agents", "deep_fetch": True},
        session_id=SPK_SESSION,
    )
    assert result.success is False
    assert "not a public address" in (result.error or "")
    assert fake_net.log == []


@pytest.mark.parametrize("addr", [
    "::127.0.0.1", "::a00:1", "::a9fe:a9fe",      # IPv4-compatible (deprecated) wrappers
    "fec0::1",                                    # site-local (deprecated)
    "64:ff9b:1::a00:1",                           # local-use NAT64 (RFC 8215)
])
def test_deprecated_and_local_v6_forms_are_not_public(addr):
    import ipaddress

    assert speaker._is_public_ip(ipaddress.ip_address(addr)) is False


async def test_member_fetch_refuses_an_ipv4_compatible_loopback(fake_net, no_deep_fetch):
    result = await _member_fetch("http://[::127.0.0.1]:25080/")
    assert result.success is False and "not a public address" in (result.error or "")
    assert fake_net.log == []


async def test_playbooks_add_records_the_members_own_session(sm):
    from captain_claw.tools.playbooks import PlaybooksTool

    member = await sm.create_session(name="spk-ana-A", metadata={"speaker_id": "u-member"})
    reg = ToolRegistry()
    reg.register(PlaybooksTool())
    result = await reg.execute(
        "playbooks",
        {"action": "add", "name": "Flow", "task_type": "web-research", "do_pattern": "x",
         "session_id": "owner-private-session", "_agent": _speaker_agent(session=member)},
        session_id="anything",
    )
    assert result.success
    items = await sm.list_playbooks(limit=10)
    assert [i.source_session for i in items] == [member.id]


async def test_playbooks_add_by_a_member_links_no_scripts(sm):
    """A member's playbook can't link the owner's scripts — they would be
    injected (names, paths, purposes) next to it in every later prompt."""
    from captain_claw.tools.playbooks import PlaybooksTool

    member = await sm.create_session(name="spk-ana-A", metadata={"speaker_id": "u-member"})
    script = await sm.create_script(name="backup_db", file_path="/Users/olga/backup_db.py")
    reg = ToolRegistry()
    reg.register(PlaybooksTool())
    result = await reg.execute(
        "playbooks",
        {"action": "add", "name": "Flow", "task_type": "web-research", "do_pattern": "x",
         "script_ids": script.id, "_agent": _speaker_agent(session=member)},
        session_id="anything",
    )
    assert result.success
    assert [i.script_ids for i in await sm.list_playbooks(limit=10)] in ([None], [""])


def test_playbooks_add_rule_drops_script_ids():
    args, err = speaker.apply_tool_rules(
        "playbooks", {"action": "add", "name": "x", "script_ids": "s1,s2"}, _speaker_agent(),
    )
    assert err is None and "script_ids" not in args


# ── member web_fetch: bounded body, bounded time ─────────────────────


class _EndlessBody(httpcore.AsyncNetworkStream):
    """A 200 response with a 16 MB body served as fast as it is read (or,
    with *stall*, a server that sends the headers and then nothing)."""

    CHUNK = 64 * 1024
    SIZE = 16 * 1024 * 1024

    def __init__(self, *, stall: bool = False):
        self.served = 0
        self.stall = stall
        self._head_sent = False

    async def read(self, max_bytes, timeout=None):
        if not self._head_sent:
            self._head_sent = True
            return (b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                    b"Content-Length: %d\r\n\r\n" % self.SIZE)
        if self.stall:
            await asyncio.Event().wait()          # a server that never sends more
        n = min(max_bytes, self.CHUNK, self.SIZE - self.served)
        self.served += n
        return b"a" * n

    async def write(self, buffer, timeout=None):
        return None

    async def aclose(self):
        return None

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        return self

    def get_extra_info(self, info):
        return None


class _EndlessNet(_FakeNet):
    def __init__(self, stream):
        super().__init__([])
        self.stream = stream

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.log.append(("tcp", host, port))
        return self.stream


@pytest.fixture
def endless(fake_net, monkeypatch):
    def _install(**kw):
        stream = _EndlessBody(**kw)
        net = _EndlessNet(stream)
        monkeypatch.setattr(speaker, "make_public_http_client", functools.partial(
            speaker.make_public_http_client.func, network_backend=net,
        ))
        return stream
    return _install


async def test_member_fetch_reads_at_most_the_byte_cap(endless, no_deep_fetch, monkeypatch):
    from captain_claw.tools import web_fetch as wf

    monkeypatch.setattr(wf, "MEMBER_FETCH_MAX_BYTES", 1024 * 1024)
    stream = endless()
    result = await _member_fetch("http://public.example/huge.bin", max_chars=50)
    assert result.success is True
    assert stream.served < 2 * 1024 * 1024                 # stopped near 1 MB, not 16
    assert "[Size: 1048576 chars]" in result.content
    assert "[Body: only the first 1 MB was read]" in result.content


async def test_member_fetch_has_a_total_deadline(endless, no_deep_fetch, monkeypatch):
    from captain_claw.tools import web_fetch as wf

    monkeypatch.setattr(wf, "MEMBER_FETCH_DEADLINE_S", 0.2)
    endless(stall=True)
    # The stalled server never trips httpx's per-read timeout here; only the
    # tool's own total deadline ends the fetch (10 s guard so a regression
    # fails instead of hanging).
    result = await asyncio.wait_for(_member_fetch("http://public.example/slow"), 10)
    assert result.success is False and "abandoned" in (result.error or "")


def test_member_fetch_limits_are_the_agreed_values():
    from captain_claw.tools import web_fetch as wf

    assert wf.MEMBER_FETCH_MAX_BYTES == 8 * 1024 * 1024
    assert wf.MEMBER_FETCH_DEADLINE_S == 45.0

