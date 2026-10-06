"""A2: shared-agent members act with their OWN credentials (contract a2 part 2 / 2b §3-§4, §6).

Flight Deck mints a per-turn grant for a member's chat message. It travels
chat frame → ``_run_agent`` → ``speaker.TURN_GRANT`` / ``agent._turn_grant``,
is cleared before the post-turn jobs and again in ``finally`` (before the lane
is freed), and is sent — together with the ``fd_member=1`` marker — only on
the member's own Google and deep-memory calls. No grant, or a thread that lost
the speaker context while member work is live, means a LOCAL refusal: never a
request made as the owner, never the agent's local (owner's) tokens or key.

Every test runs with HOME, FD_DATA_DIR and the session / topic stores pointed
at a tmp dir (nothing here may reach ~/.captain-claw or a real FD data dir).
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import inspect
import json
import threading
import time
import types
from pathlib import Path

import httpx
import pytest

from captain_claw import speaker
from captain_claw.config import get_config
from captain_claw.exceptions import ToolBlockedError
from captain_claw.speaker import (
    GRANT_HEADER,
    NO_GRANT_MESSAGE,
    SPEAKER_MODE_NOTE,
    SPEAKER_MODE_NOTE_FULL,
    SPEAKER_TOOL_ALLOWLIST,
    SPEAKER_TOOL_ALLOWLIST_MAX,
    UNKNOWN_PRINCIPAL,
    Principal,
    SpeakerGrantMissing,
)
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

WEB_AUTH = "web-auth-of-this-agent"
GRANT = "G" * 40 + "_-1"
GRANT_B = "b" * 43
PRINCIPAL = Principal("u-member", "Ana", "Olga", "A", "process:helper:0123456789abcdef")
OTHER = Principal("u-other", "Bo", "Olga", "B", "process:helper:0123456789abcdef")
DOCKER = Principal("u-member", "Ana", "Olga", "A", "docker:helper:0123456789abcdef")
SPK_SESSION = "spk-session-1"
FD = "http://fd.test"
_RealAsyncClient = httpx.AsyncClient

_DB_FIELDS = (
    ("memory", "path"), ("session", "path"), ("insights", "db_path"),
    ("conversation_topics", "db_path"), ("nervous_system", "db_path"),
    ("sister_session", "db_path"), ("cognitive_metrics", "db_path"),
    ("datastore", "path"), ("autonomous_work", "db_path"),
)
_ENV_CLEARED = (
    "CLAW_VFS_ROOT", "CLAW_VFS_USER", "FD_OWNER_ID", "CLAW_VFS_PROJECT", "CLAW_VFS_SCOPE",
    "CLAW_WRITE_DIRECT", "FD_URL", "FD_AGENT_SHARED_SECRET", "FD_AGENT_SLUG",
    "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER", "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT",
    "CLAW_BEING_WORKER",
)


# ── isolation ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """HOME, FD_DATA_DIR, every config DB path and the global session/topic
    managers → tmp, BEFORE anything is created; module state reset."""
    home = tmp_path / "home"
    (home / ".captain-claw").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    fd_data = tmp_path / "fd-data"
    (fd_data / "vfs").mkdir(parents=True)
    monkeypatch.setenv("FD_DATA_DIR", str(fd_data))
    for var in _ENV_CLEARED:
        monkeypatch.delenv(var, raising=False)
    cfg = get_config()
    for section, attr in _DB_FIELDS:
        monkeypatch.setattr(getattr(cfg, section), attr, str(home / ".captain-claw" / f"{section}.db"))
    monkeypatch.setattr(cfg.web, "auth_token", WEB_AUTH)
    monkeypatch.setattr(cfg.web, "public_run", False)
    monkeypatch.setattr(cfg.google_oauth, "flight_deck_url", "")
    monkeypatch.setattr(cfg.google_oauth, "flight_deck_secret", "deck-secret")
    from captain_claw import session as _session

    monkeypatch.setattr(_session, "_manager", _session.SessionManager(home / ".captain-claw" / "s.db"))
    import captain_claw.conversation_topics as _ct

    monkeypatch.setattr(_ct, "_MANAGER", None)
    import captain_claw.google_oauth_manager as gom

    monkeypatch.setattr(gom, "_GOOGLE_CONNECTED", {})
    monkeypatch.setattr(speaker, "_IN_FLIGHT", 0)
    monkeypatch.setattr(speaker, "_MAIN_LOOP", None)
    monkeypatch.setattr(speaker, "_MEMBER_GOOGLE", {})
    return home


# ── helpers ──────────────────────────────────────────────────────────


async def wait_for(cond, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met in time")


class _Bound:
    """``with _Bound(p, grant):`` — bind a principal (and grant) for a block."""

    def __init__(self, p, grant=""):
        self.p, self.grant = p, grant

    def __enter__(self):
        self._t1 = speaker.bind(self.p)
        self._t2 = speaker.bind_grant(self.grant)
        return self

    def __exit__(self, *exc):
        speaker.reset_grant(self._t2)
        speaker.reset(self._t1)


def _member_agent(p=PRINCIPAL, grant="", session_id=SPK_SESSION):
    return types.SimpleNamespace(
        _speaker_scoped=True, _speaker_principal=p, _turn_grant=grant,
        session=types.SimpleNamespace(id=session_id),
        _current_session_slug=lambda: session_id,
    )


class _Probe(Tool):
    """Records what the tool task saw; never does anything."""

    def __init__(self, name):
        self.name = name
        self.description = name
        self.parameters = {"type": "object", "properties": {}, "required": []}
        self.seen: list[dict] = []

    async def execute(self, **kwargs):
        self.seen.append({
            "principal": speaker.current(),
            "grant": speaker.current_grant(),
            "file_registry": kwargs.get("_file_registry"),
            "lost": speaker.identity_lost(),
            # A thread that copies the tool context (asyncio.to_thread).
            "lost_in_thread": await asyncio.to_thread(speaker.identity_lost),
        })
        return ToolResult(success=True, content=f"{self.name} ran")


class _HTTP:
    """httpx.AsyncClient factory over a MockTransport that records requests."""

    def __init__(self, handler):
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def factory(self, *a, **kw):
        kw.pop("transport", None)

        def _h(request):
            self.requests.append(request)
            return self.handler(request)

        return _RealAsyncClient(transport=httpx.MockTransport(_h), **kw)


@pytest.fixture
def http(monkeypatch):
    def install(handler=None):
        rec = _HTTP(handler or (lambda r: httpx.Response(500, json={"detail": "unexpected"})))
        monkeypatch.setattr(httpx, "AsyncClient", rec.factory)
        return rec
    return install


# ── the chat frame: _fd_grant on speaker sockets only ────────────────


class _GateServer:
    def __init__(self):
        self.sent = []
        self.agent = types.SimpleNamespace(plan_mode_auto=False)

    async def _send(self, ws, msg):
        self.sent.append(msg)


@pytest.mark.parametrize("raw,expected", [
    (GRANT, GRANT),
    ("short", ""),
    ("x" * 44, ""),
    ("G" * 42 + "!", ""),
    (GRANT + "\n", ""),
    (12345, ""),
    (None, ""),
    ({"g": GRANT}, ""),
])
async def test_speaker_chat_frame_passes_the_sanitized_grant(monkeypatch, raw, expected):
    from captain_claw.web import speaker_ws

    calls = []

    async def _handle_chat(server, ws, content, **kw):
        calls.append(kw)
        return True

    monkeypatch.setattr("captain_claw.web.chat_handler.handle_chat", _handle_chat)
    ws = types.SimpleNamespace(_speaker_key=("u-member", "A"))
    frame = {"type": "chat", "content": "hello", "_fd_turn": "t1"}
    if raw is not None:
        frame["_fd_grant"] = raw
    await speaker_ws._speaker_chat(_GateServer(), ws, frame)
    assert calls and calls[0]["speaker_grant"] == expected


async def test_slash_commands_ignore_the_grant(monkeypatch):
    from captain_claw.web import speaker_ws

    handled, chats = [], []

    async def _cmd(view, ws, content):
        handled.append(content)

    async def _handle_chat(*a, **k):
        chats.append(k)
        return True

    monkeypatch.setattr("captain_claw.web.slash_commands.handle_command", _cmd)
    monkeypatch.setattr("captain_claw.web.chat_handler.handle_chat", _handle_chat)
    server = _GateServer()

    async def _resolve(ws):
        return object()

    server.resolve_agent = _resolve
    server.lane_view = lambda ws, agent: server
    ws = types.SimpleNamespace(_speaker_key=("u-member", "A"))
    await speaker_ws._speaker_chat(server, ws, {"type": "chat", "content": "/help",
                                                "_fd_turn": "t1", "_fd_grant": GRANT})
    assert handled == ["/help"] and chats == []


async def test_owner_sockets_ignore_a_grant_field(monkeypatch):
    from captain_claw.web.ws_handler import handle_ws_message

    calls = []

    async def _handle_chat(server, ws, content, **kw):
        calls.append(kw)
        return True

    monkeypatch.setattr("captain_claw.web.chat_handler.handle_chat", _handle_chat)
    owner_ws = types.SimpleNamespace()          # no _speaker_key: the owner's socket
    await handle_ws_message(_GateServer(), owner_ws,
                            {"type": "chat", "content": "hi", "_fd_grant": GRANT})
    assert calls and "speaker_grant" not in calls[0]


@pytest.mark.parametrize("given,expected", [(GRANT, GRANT), (None, "")])
async def test_handle_chat_forwards_the_grant_to_the_member_turn(monkeypatch, given, expected):
    from captain_claw.web import chat_handler

    seen = []

    async def _speaker_chat(server, ws, content, key, **kw):
        seen.append(kw)
        return True

    monkeypatch.setattr(chat_handler, "_handle_speaker_chat", _speaker_chat)
    server = types.SimpleNamespace(agent=object())
    ws = types.SimpleNamespace(_speaker_key=("u-member", "A"))
    assert await chat_handler.handle_chat(server, ws, "hi", speaker_turn="t", speaker_grant=given)
    assert seen[0]["speaker_grant"] == expected


class _SpeakerServer:
    """Enough of WebServer for _handle_speaker_chat / _run_agent."""

    LANE_MAIN = "A"

    def __init__(self, agent):
        self._agent = agent
        self.sent: list[dict] = []
        self.errors: list[dict] = []
        self._busy = False
        self._orchestrator = None
        self.agent = types.SimpleNamespace(_fleet_identity=None, _fleet_instructions="", _fd_url="")

    async def _get_speaker_agent(self, p):
        return self._agent

    def _speaker_send(self, key):
        return self.sent.append

    async def _send(self, ws, msg):
        self.errors.append(msg)

    def _session_info(self, agent):
        return {}


async def test_a_busy_second_chat_leaves_the_running_grant(monkeypatch):
    from captain_claw.web import chat_handler

    launched = []

    async def _run_agent(*a, **kw):
        launched.append(kw)

    monkeypatch.setattr(chat_handler, "_run_agent", _run_agent)
    monkeypatch.setattr(chat_handler, "_start_task_naming", lambda *a, **k: None)
    agent = _member_agent(grant=GRANT)
    agent._lane_busy = True
    server = _SpeakerServer(agent)
    ws = types.SimpleNamespace(_speaker_principal=PRINCIPAL)
    ok = await chat_handler._handle_speaker_chat(
        server, ws, "second", ("u-member", "A"), rewind_to=None, no_next_steps=True,
        no_rephrase=False, speaker_turn="t2", speaker_grant=GRANT_B,
    )
    assert ok is False and launched == []
    assert agent._turn_grant == GRANT                     # the running turn's grant
    assert server.errors and server.errors[0]["code"] == "busy"

    # Not busy: the new grant goes to the launched turn (not set here).
    agent._lane_busy = False
    ok = await chat_handler._handle_speaker_chat(
        server, ws, "third", ("u-member", "A"), rewind_to=None, no_next_steps=True,
        no_rephrase=False, speaker_turn="t3", speaker_grant=GRANT_B,
    )
    await asyncio.sleep(0)
    assert ok is True and launched[-1]["speaker_grant"] == GRANT_B


# ── _run_agent: bind, clear before post-turn jobs, clear in finally ──


class _TurnAgent:
    """A member instance whose `_lane_busy` setter records the grant left on
    it at the moment the lane is freed."""

    def __init__(self, complete_impl=None):
        self._speaker_scoped = True
        self._speaker_principal = PRINCIPAL
        self._turn_grant = ""
        self._lb = False
        self.freed_with: list[str] = []
        self.session = types.SimpleNamespace(id=SPK_SESSION, messages=[])
        self.last_usage: dict = {}
        self.total_usage: dict = {}
        self.last_context_window: dict = {}
        self._speaker_cache_refreshed_at = time.monotonic()
        self.seen: list[tuple] = []
        self.complete_impl = complete_impl

    @property
    def _lane_busy(self):
        return self._lb

    @_lane_busy.setter
    def _lane_busy(self, value):
        if not value:
            self.freed_with.append(self._turn_grant)
        self._lb = value

    def _current_session_slug(self):
        return SPK_SESSION

    def get_runtime_model_details(self):
        return {"provider": "p", "model": "m"}

    async def complete(self, content):
        self.seen.append((speaker.current(), speaker.current_grant(), self._turn_grant,
                          speaker.turns_in_flight()))
        if self.complete_impl is not None:
            return await self.complete_impl(self)
        return "ok"


@pytest.fixture
def post_turn(monkeypatch):
    """Spies for the four post-turn jobs; each blocks until released."""
    rec = types.SimpleNamespace(calls=[], release=None)

    def _spy(name):
        async def _job(*a, **k):
            if rec.release is None:
                rec.release = asyncio.Event()
            rec.calls.append((name, speaker.current(), speaker.current_grant(),
                              speaker.turns_in_flight()))
            await rec.release.wait()
        return _job

    for target, name in (
        ("captain_claw.reflections.maybe_auto_reflect", "reflect"),
        ("captain_claw.insights.maybe_extract_insights", "insights"),
        ("captain_claw.nervous_system.maybe_dream", "dream"),
        ("captain_claw.conversation_topics.maybe_classify_topics", "topics"),
    ):
        monkeypatch.setattr(target, _spy(name))

    async def _never(*a, **k):
        raise AssertionError("intentions never run for a member turn")

    monkeypatch.setattr("captain_claw.intentions_generator.maybe_auto_propose", _never)
    return rec


def _launch(agent, server, grant=GRANT):
    from captain_claw.web.chat_handler import _run_agent

    agent._lane_busy = True
    return asyncio.create_task(_run_agent(
        server, None, agent, "hello", None, lane="A", no_flow=True, no_next_steps=True,
        speaker_key=("u-member", "A"), speaker_turn="t1", speaker_grant=grant,
    ))


def _ready_frames(server):
    return [f for f in server.sent if f.get("type") == "status" and f.get("status") == "ready"]


async def test_run_agent_binds_the_grant_for_the_turn_and_clears_it(post_turn):
    agent = _TurnAgent()
    server = _SpeakerServer(agent)
    await _launch(agent, server)
    principal, grant, on_agent, counted = agent.seen[0]
    assert principal == PRINCIPAL and grant == GRANT and on_agent == GRANT and counted == 1

    # The post-turn jobs carry the principal but NOT the grant, and count.
    await wait_for(lambda: len(post_turn.calls) == 4)
    assert {c[0] for c in post_turn.calls} == {"reflect", "insights", "dream", "topics"}
    for _name, p, g, n in post_turn.calls:
        assert p == PRINCIPAL and g == "" and n >= 1
    assert speaker.turns_in_flight() == 4                 # the turn ended; its jobs still run
    assert agent._turn_grant == "" and agent.freed_with == [""]
    assert _ready_frames(server) == [{"type": "status", "status": "ready", "turn_end": "t1"}]
    post_turn.release.set()
    await wait_for(lambda: speaker.turns_in_flight() == 0)


async def test_run_agent_error_path_clears_the_grant(post_turn):
    async def _boom(agent):
        raise RuntimeError("provider error with sk-OWNERKEY")

    agent = _TurnAgent(_boom)
    server = _SpeakerServer(agent)
    await _launch(agent, server)
    assert agent.seen[0][1] == GRANT
    assert agent._turn_grant == "" and agent.freed_with == [""]
    assert speaker.turns_in_flight() == 0 and post_turn.calls == []
    assert len(_ready_frames(server)) == 1
    assert "OWNERKEY" not in json.dumps(server.sent)


async def test_run_agent_cancel_path_clears_the_grant(post_turn):
    started = asyncio.Event()

    async def _hang(agent):
        started.set()
        await asyncio.Event().wait()

    agent = _TurnAgent(_hang)
    server = _SpeakerServer(agent)
    task = _launch(agent, server)
    await asyncio.wait_for(started.wait(), 5)
    assert agent._turn_grant == GRANT and speaker.turns_in_flight() == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent._turn_grant == "" and agent.freed_with == [""]
    assert speaker.turns_in_flight() == 0
    assert len(_ready_frames(server)) == 1


async def test_a_malformed_grant_never_reaches_the_turn(post_turn):
    agent = _TurnAgent()
    await _launch(agent, _SpeakerServer(agent), grant="not-a-grant")
    assert agent.seen[0][1] == "" and agent.seen[0][2] == ""
    post_turn.release.set()


async def test_open_commons_jobs_still_run_after_a_google_turn(post_turn):
    """Part 0 G-A2-4/N8 (open commons): a member turn that used their Google
    still schedules insights, dreaming and topic classification — each
    tracked as member work while it runs, none holding the grant."""
    reg = ToolRegistry()
    mail = _Probe("google_mail")
    reg.register(mail)

    async def _use_google(agent):
        result = await reg.execute("google_mail", {"action": "list_messages", "_agent": agent},
                                   session_id=SPK_SESSION)
        assert result.success
        return "done"

    agent = _TurnAgent(_use_google)
    agent.tools = reg
    await _launch(agent, _SpeakerServer(agent))
    assert mail.seen and mail.seen[0]["grant"] == GRANT
    await wait_for(lambda: {"insights", "dream", "topics"} <= {c[0] for c in post_turn.calls})
    for name, p, g, n in post_turn.calls:
        assert p == PRINCIPAL and g == "" and n > 0, name
    post_turn.release.set()
    await wait_for(lambda: speaker.turns_in_flight() == 0)


# ── the registry: grant on every signal, none → local refusal ────────


async def test_tool_task_sees_the_grant_via_the_contextvar():
    reg = ToolRegistry()
    probe = _Probe("typesense")
    reg.register(probe)
    with _Bound(PRINCIPAL, GRANT):
        assert (await reg.execute("typesense", {"action": "search", "query": "q"},
                                  session_id="x")).success
    assert probe.seen[0]["principal"] == PRINCIPAL and probe.seen[0]["grant"] == GRANT


@pytest.mark.parametrize("with_key", [False, True])
async def test_tool_task_sees_the_grant_from_the_member_instance(with_key):
    """Nothing bound (a lost contextvar): `_agent` (and the session key) still
    identify the member, and the instance's `_turn_grant` is the grant."""
    reg = ToolRegistry()
    probe = _Probe("typesense")
    reg.register(probe)
    if with_key:
        reg.register_speaker_session(SPK_SESSION)
    result = await reg.execute(
        "typesense", {"action": "search", "query": "q", "_agent": _member_agent(grant=GRANT)},
        session_id=SPK_SESSION,
    )
    assert result.success
    assert probe.seen[0]["principal"] == PRINCIPAL and probe.seen[0]["grant"] == GRANT
    assert speaker.current() is None and speaker.current_grant() == ""   # nothing leaks out


@pytest.mark.parametrize("name", ["google_mail", "google_drive", "google_calendar", "typesense"])
@pytest.mark.parametrize("signal", ["contextvar", "agent"])
async def test_google_and_deep_memory_without_a_grant_are_refused_locally(name, signal):
    reg = ToolRegistry()
    probe = _Probe(name)
    reg.register(probe)
    args = {"action": "search", "query": "q"}
    if signal == "agent":
        args["_agent"] = _member_agent(grant="")
        ctx = _Bound(None)
    else:
        ctx = _Bound(PRINCIPAL, "")
    with ctx, pytest.raises(ToolBlockedError) as info:
        await reg.execute(name, args, session_id=SPK_SESSION)
    assert NO_GRANT_MESSAGE in str(info.value)
    assert probe.seen == []


async def test_a_member_instance_owned_grant_is_ignored_for_a_non_speaker_agent():
    reg = ToolRegistry()
    probe = _Probe("typesense")
    reg.register(probe)
    reg.register_speaker_session(SPK_SESSION)
    owner_agent = types.SimpleNamespace(_speaker_scoped=False, _turn_grant=GRANT)
    with pytest.raises(ToolBlockedError):
        await reg.execute("typesense", {"action": "search", "query": "q", "_agent": owner_agent},
                          session_id=SPK_SESSION)
    assert probe.seen == []


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    note = ws / "saved" / "tmp" / SPK_SESSION / "note.md"
    note.parent.mkdir(parents=True)
    note.write_text("member note")
    return ws


async def test_member_file_calls_get_no_file_registry(workspace):
    reg = ToolRegistry()
    probe = _Probe("read")
    reg.register(probe)
    sentinel = object()
    # A process member's read recognised by `_agent` only (nothing bound, the
    # loop thread): the policy chain does NOT block it, and no file registry.
    result = await reg.execute(
        "read", {"path": f"saved/tmp/{SPK_SESSION}/note.md", "_agent": _member_agent()},
        session_id=SPK_SESSION, runtime_base_path=workspace, file_registry=sentinel,
    )
    assert result.success
    assert probe.seen[0]["file_registry"] is None and probe.seen[0]["principal"] == PRINCIPAL
    # The owner's call keeps its registry.
    await reg.execute("read", {"path": "x"}, session_id="owner", file_registry=sentinel)
    assert probe.seen[1]["file_registry"] is sentinel and probe.seen[1]["principal"] is None


async def test_session_key_only_never_lists_google_tools_even_when_the_owner_is_connected():
    import captain_claw.google_oauth_manager as gom

    reg = ToolRegistry()
    reg.register(_Probe("google_mail"), metadata={"requires_google": True})
    reg.register(_Probe("web_search"))
    gom._mark_google_connected(True)                      # the OWNER's flag (nothing bound)
    assert "google_mail" in reg.list_tools(session_id="owner")
    reg.register_speaker_session(SPK_SESSION)
    assert "google_mail" not in reg.list_tools(session_id=SPK_SESSION)
    assert "web_search" in reg.list_tools(session_id=SPK_SESSION)


# ── docker members stay A1 ───────────────────────────────────────────


@pytest.mark.parametrize("name", ["google_mail", "typesense", "read", "vfs"])
@pytest.mark.parametrize("signal", ["contextvar", "agent"])
async def test_docker_members_get_no_a2_tools(name, signal, workspace):
    reg = ToolRegistry()
    probe = _Probe(name)
    reg.register(probe)
    args = {"action": "search", "path": f"saved/tmp/{SPK_SESSION}/note.md"}
    if signal == "agent":
        args["_agent"] = _member_agent(DOCKER, grant=GRANT)
        ctx = _Bound(None)
    else:
        args["_agent"] = _member_agent(DOCKER, grant=GRANT)
        ctx = _Bound(DOCKER, GRANT)
    with ctx, pytest.raises(ToolBlockedError):
        await reg.execute(name, args, session_id=SPK_SESSION, runtime_base_path=workspace)
    assert probe.seen == []


def test_docker_and_unknown_members_stay_at_a1():
    assert speaker.allowed_tools(DOCKER) == SPEAKER_TOOL_ALLOWLIST
    assert speaker.allowed_tools(UNKNOWN_PRINCIPAL) == SPEAKER_TOOL_ALLOWLIST
    assert speaker.allowed_tools(None) == SPEAKER_TOOL_ALLOWLIST
    assert speaker.allowed_tools(Principal("u", "x", "o", "A", "process:bad ref")) == SPEAKER_TOOL_ALLOWLIST
    assert speaker.allowed_tools(PRINCIPAL) == SPEAKER_TOOL_ALLOWLIST_MAX
    assert speaker.speaker_mode_note(DOCKER) == SPEAKER_MODE_NOTE
    assert speaker.speaker_mode_note(UNKNOWN_PRINCIPAL) == SPEAKER_MODE_NOTE
    assert speaker.speaker_mode_note(PRINCIPAL) == SPEAKER_MODE_NOTE_FULL
    assert speaker.prompt_tools(PRINCIPAL) == SPEAKER_TOOL_ALLOWLIST_MAX - speaker.SPEAKER_GOOGLE_TOOLS
    assert speaker.runtime_of(DOCKER) == "docker" and speaker.runtime_of(PRINCIPAL) == "process"


def test_a2_constants_are_the_contract():
    assert speaker.GRANT_HEADER == "X-FD-Speaker-Grant"
    assert speaker.MEMBER_MARKER_PARAM == "fd_member"
    assert speaker.CHAT_GRANT_FIELD == "_fd_grant"
    assert speaker.GRANT_TOKEN_RE == r"^[A-Za-z0-9_-]{43}$"
    assert speaker.SPEAKER_GOOGLE_TOOLS == {"google_mail", "google_drive", "google_calendar"}
    assert speaker.SPEAKER_DEEP_MEMORY_TOOLS == {"typesense"}
    assert speaker.SPEAKER_FILE_TOOLS == {"read", "write", "edit", "glob", "grep", "vfs",
                                          "pdf_extract", "docx_extract", "xlsx_extract",
                                          "pptx_extract"}
    assert SPEAKER_TOOL_ALLOWLIST == {"insights", "playbooks", "topics", "web_search", "web_fetch"}
    assert speaker.VFS_RESERVED_NAMES == {".vfs-links.json", ".vfs-meta.jsonl",
                                          ".drive-manifest.json", ".drive-cache"}
    assert speaker.SPEAKER_GOOGLE_STATUS_REFRESH_S == 30
    assert speaker.SPEAKER_GOOGLE_STATUS_MAX_AGE_S == 120


# ── identity_lost: fail closed off the speaker context ───────────────


def _lost_report() -> dict:
    """identity_lost + what the A2 clients do with it, from wherever this runs."""
    out: dict = {"lost": speaker.identity_lost()}
    try:
        speaker.grant_headers()
        out["headers"] = "ok"
    except SpeakerGrantMissing:
        out["headers"] = "raised"
    from captain_claw import vfs

    try:
        out["vfs_user"] = vfs.vfs_user()
    except PermissionError:
        out["vfs_user"] = "raised"
    return out


@pytest.fixture
def owner_env(monkeypatch):
    monkeypatch.setenv("CLAW_VFS_USER", "owner-id")
    monkeypatch.setenv("FD_OWNER_ID", "owner-id")


async def test_identity_lost_in_a_bare_executor_thread(owner_env):
    loop = asyncio.get_running_loop()
    speaker.turn_started()
    try:
        report = await loop.run_in_executor(None, _lost_report)
    finally:
        speaker.turn_ended()
    assert report == {"lost": True, "headers": "raised", "vfs_user": "raised"}


async def test_identity_lost_inside_asyncio_run_in_a_thread(owner_env):
    async def _inner():
        return _lost_report()

    loop = asyncio.get_running_loop()
    speaker.turn_started()
    try:
        report = await loop.run_in_executor(None, lambda: asyncio.run(_inner()))
    finally:
        speaker.turn_ended()
    assert report == {"lost": True, "headers": "raised", "vfs_user": "raised"}


async def test_identity_lost_in_a_thread_spawned_by_a_tracked_post_turn_task(owner_env):
    speaker.turn_started()                     # the turn
    out: dict = {}
    go = asyncio.Event()

    async def _job():
        await go.wait()
        t = threading.Thread(target=lambda: out.update(_lost_report()))
        t.start()
        await asyncio.get_running_loop().run_in_executor(None, t.join)

    task = speaker.track_member_task(asyncio.create_task(_job()))
    speaker.turn_ended()                       # the turn is over; its job is not
    assert speaker.turns_in_flight() == 1
    go.set()
    await task
    assert out == {"lost": True, "headers": "raised", "vfs_user": "raised"}
    assert speaker.turns_in_flight() == 0


async def test_identity_is_not_lost_where_the_contextvar_is_authoritative(owner_env):
    loop = asyncio.get_running_loop()
    speaker.turn_started()
    try:
        # The main loop thread with nothing bound: the owner.
        assert _lost_report() == {"lost": False, "headers": "ok", "vfs_user": "owner-id"}
        # A copy_context().run thread from a member's context: the member.
        with _Bound(PRINCIPAL, GRANT):
            ctx = contextvars.copy_context()
        report = await loop.run_in_executor(None, ctx.run, _lost_report)
        assert report == {"lost": False, "headers": "ok", "vfs_user": "u-member"}
        # A copy_context().run thread from a registry tool context (owner).
        tool_ctx = contextvars.copy_context()
        tool_ctx.run(speaker.mark_tool_context)
        report = await loop.run_in_executor(None, tool_ctx.run, _lost_report)
        assert report == {"lost": False, "headers": "ok", "vfs_user": "owner-id"}
    finally:
        speaker.turn_ended()


async def test_registry_tool_tasks_are_never_lost(owner_env):
    reg = ToolRegistry()
    probe = _Probe("topics")
    reg.register(probe)
    speaker.turn_started()
    try:
        await reg.execute("topics", {"action": "list"}, session_id="owner")
    finally:
        speaker.turn_ended()
    assert probe.seen[0]["lost"] is False and probe.seen[0]["principal"] is None
    # The registry marks every tool context (owner calls too): an owner tool's
    # to_thread worker keeps its identity while member work is live.
    assert probe.seen[0]["lost_in_thread"] is False


async def test_nothing_counted_means_nothing_lost(owner_env):
    loop = asyncio.get_running_loop()
    assert speaker.turns_in_flight() == 0
    report = await loop.run_in_executor(None, _lost_report)
    assert report == {"lost": False, "headers": "ok", "vfs_user": "owner-id"}


def test_turn_counter_never_goes_negative():
    speaker.turn_ended()
    speaker.turn_ended()
    assert speaker.turns_in_flight() == 0
    speaker.turn_started()
    assert speaker.turns_in_flight() == 1
    speaker.turn_ended()
    assert speaker.turns_in_flight() == 0


# ── grant_headers / grant_params ─────────────────────────────────────


def test_grant_headers_and_params():
    assert speaker.grant_headers() == {} and speaker.grant_params() == {}
    with _Bound(PRINCIPAL, GRANT):
        assert speaker.grant_headers() == {GRANT_HEADER: GRANT}
        assert speaker.grant_params() == {"fd_member": "1"}
    with _Bound(PRINCIPAL, ""):
        with pytest.raises(SpeakerGrantMissing):
            speaker.grant_headers()
        with pytest.raises(SpeakerGrantMissing):
            speaker.grant_params()
    with _Bound(UNKNOWN_PRINCIPAL, ""):
        with pytest.raises(SpeakerGrantMissing):
            speaker.grant_headers()


async def test_grant_headers_raise_when_identity_is_lost():
    loop = asyncio.get_running_loop()
    speaker.turn_started()
    try:
        def _try():
            try:
                speaker.grant_params()
                return "ok"
            except SpeakerGrantMissing:
                return "raised"
        assert await loop.run_in_executor(None, _try) == "raised"
    finally:
        speaker.turn_ended()


def test_sanitize_and_bind_grant():
    assert speaker.sanitize_grant(GRANT) == GRANT
    assert speaker.sanitize_grant(" " + GRANT) == ""
    tok = speaker.bind_grant("junk")
    try:
        assert speaker.current_grant() == ""
    finally:
        speaker.reset_grant(tok)
    speaker.reset_grant(None)                    # never raises
    assert speaker.current_grant() == ""


# ── glob / grep run their scans with the member bound ────────────────


@pytest.fixture
def member_tree(tmp_path):
    root = tmp_path / "fd-data" / "vfs" / "u-member" / "p"
    root.mkdir(parents=True)
    (root / "a.md").write_text("alpha line\n")
    return root


async def test_glob_executor_sees_the_member(monkeypatch, member_tree, workspace):
    import glob as _stdglob

    real = _stdglob.glob
    seen = []

    def _recording(*a, **k):
        seen.append((speaker.current(), threading.current_thread() is threading.main_thread()))
        return real(*a, **k)

    monkeypatch.setattr(_stdglob, "glob", _recording)
    from captain_claw.tools.glob import GlobTool

    reg = ToolRegistry()
    reg.register(GlobTool())
    result = await reg.execute("glob", {"pattern": "vfs:p/*.md", "_agent": _member_agent()},
                               session_id=SPK_SESSION, runtime_base_path=workspace)
    assert result.success and "vfs:p/a.md" in result.content
    assert seen and seen[0] == (PRINCIPAL, False)


async def test_grep_executor_sees_the_member(monkeypatch, member_tree, workspace):
    from captain_claw.tools.grep import GrepTool

    real = GrepTool._scan
    seen = []

    def _recording(files, rx, rel_base, limit):
        seen.append((speaker.current(), threading.current_thread() is threading.main_thread()))
        return real(files, rx, rel_base, limit)

    monkeypatch.setattr(GrepTool, "_scan", staticmethod(_recording))
    reg = ToolRegistry()
    reg.register(GrepTool())
    result = await reg.execute("grep", {"pattern": "alpha", "path": "vfs:p", "_agent": _member_agent()},
                               session_id=SPK_SESSION, runtime_base_path=workspace)
    assert result.success and "alpha line" in result.content
    assert seen and seen[0] == (PRINCIPAL, False)


# ── AST guard: member-reachable modules never drop the context ───────

_EXTRA_MODULES = (
    "captain_claw.vfs", "captain_claw.vfs_drive", "captain_claw.drive_client",
    "captain_claw.google_oauth_manager", "captain_claw.fd_client",
    "captain_claw.file_tree_builder",
)


def _tool_modules() -> dict[str, str]:
    import captain_claw.tools as pkg

    found: dict[str, str] = {}
    for attr in dir(pkg):
        obj = getattr(pkg, attr)
        if isinstance(obj, type) and issubclass(obj, Tool) and obj is not Tool:
            name = getattr(obj, "name", "")
            if name in SPEAKER_TOOL_ALLOWLIST_MAX:
                found[name] = obj.__module__
    return found


def _is_copy_context(node) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "copy_context") or (
        isinstance(f, ast.Attribute) and f.attr == "copy_context")


def executor_violations(source: str) -> list[str]:
    """`run_in_executor` whose callable isn't `<copy_context()>.run`, and any
    `threading.Thread(` / `Thread(` / `ThreadPoolExecutor(` / `.submit(`."""
    tree = ast.parse(source)
    ctx_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_copy_context(node.value):
            ctx_names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        if isinstance(node, ast.AnnAssign) and _is_copy_context(node.value) \
                and isinstance(node.target, ast.Name):
            ctx_names.add(node.target.id)
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if fname == "run_in_executor":
            fn = node.args[1] if len(node.args) >= 2 else None
            ok = (isinstance(fn, ast.Attribute) and fn.attr == "run" and (
                (isinstance(fn.value, ast.Name) and fn.value.id in ctx_names)
                or _is_copy_context(fn.value)))
            if not ok:
                bad.append(f"run_in_executor at line {node.lineno}")
        elif fname in ("Thread", "ThreadPoolExecutor"):
            bad.append(f"{fname}( at line {node.lineno}")
        elif fname == "submit" and isinstance(f, ast.Attribute):
            bad.append(f".submit( at line {node.lineno}")
    return bad


def test_the_ast_guard_catches_what_it_should():
    bad = executor_violations(
        "import asyncio, threading, contextvars\n"
        "async def f(loop, fn):\n"
        "    await loop.run_in_executor(None, fn)\n"
        "    await loop.run_in_executor(None, lambda: fn())\n"
        "    threading.Thread(target=fn).start()\n"
        "    pool.submit(fn)\n"
        "    ctx = contextvars.copy_context()\n"
        "    await loop.run_in_executor(None, ctx.run, fn)\n"
        "    await loop.run_in_executor(None, contextvars.copy_context().run, fn)\n"
    )
    assert len(bad) == 4, bad


def test_member_reachable_modules_propagate_the_context():
    import importlib

    tools = _tool_modules()
    assert set(tools) == set(SPEAKER_TOOL_ALLOWLIST_MAX), set(SPEAKER_TOOL_ALLOWLIST_MAX) - set(tools)
    modules = sorted(set(tools.values()) | set(_EXTRA_MODULES))
    problems = {}
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        bad = executor_violations(Path(inspect.getsourcefile(mod)).read_text())
        if bad:
            problems[mod_name] = bad
    assert problems == {}
    # asyncio.run( is allowed (identity_lost treats a foreign loop as lost);
    # the known site is vfs_drive.materialize_sync.
    import captain_claw.vfs_drive as vd

    assert "asyncio.run(" in inspect.getsource(vd.materialize_sync)


# ── Google: the member's own account through Flight Deck, or nothing ──


def _fd(monkeypatch):
    monkeypatch.setenv("FD_URL", FD)


async def test_member_without_grant_gets_no_google_request(monkeypatch, http):
    from captain_claw.google_oauth_manager import FlightDeckRefused, GoogleOAuthManager

    _fd(monkeypatch)
    rec = http()
    with _Bound(PRINCIPAL, ""):
        with pytest.raises(FlightDeckRefused) as info:
            await GoogleOAuthManager(object()).get_tokens()
    assert info.value.status == 403 and NO_GRANT_MESSAGE in info.value.detail
    assert rec.requests == []


async def test_member_access_token_carries_grant_and_marker(monkeypatch, http):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    _fd(monkeypatch)
    rec = http(lambda r: httpx.Response(200, json={
        "access_token": "tok-member", "expires_at": time.time() + 3600, "scope": "gmail",
    }))
    with _Bound(PRINCIPAL, GRANT):
        tokens = await GoogleOAuthManager(object()).get_tokens()
    assert tokens.access_token == "tok-member"
    req = rec.requests[0]
    assert req.url.path == "/fd/google/access_token"
    assert req.headers["X-Agent-Auth"] == WEB_AUTH
    assert req.headers[GRANT_HEADER] == GRANT
    assert req.url.params["fd_member"] == "1"


async def test_owner_access_token_has_no_grant_or_marker(monkeypatch, http):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    _fd(monkeypatch)
    rec = http(lambda r: httpx.Response(200, json={"access_token": "tok-owner",
                                                   "expires_at": time.time() + 3600}))
    tokens = await GoogleOAuthManager(object()).get_tokens()
    assert tokens.access_token == "tok-owner"
    req = rec.requests[0]
    assert GRANT_HEADER not in req.headers and "fd_member" not in req.url.params
    assert req.headers["X-Agent-Auth"] == WEB_AUTH


async def test_cached_tokens_are_never_reused_across_principals(monkeypatch, http):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    _fd(monkeypatch)
    rec = http(lambda r: httpx.Response(200, json={
        "access_token": "tok-member" if GRANT_HEADER in r.headers else "tok-owner",
        "expires_at": time.time() + 3600,
    }))
    mgr = GoogleOAuthManager(object())
    assert (await mgr.get_tokens()).access_token == "tok-owner"
    with _Bound(PRINCIPAL, GRANT):
        assert (await mgr.get_tokens()).access_token == "tok-member"
    assert (await mgr.get_tokens()).access_token == "tok-owner"
    assert len(rec.requests) == 3


async def test_vertex_credentials_stay_the_owners(monkeypatch, http):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    _fd(monkeypatch)
    rec = http(lambda r: httpx.Response(200, json={"credentials": {"type": "authorized_user"}}))
    with _Bound(PRINCIPAL, GRANT):
        data = await GoogleOAuthManager(object())._fd_get_credentials()
    assert data["credentials"]["type"] == "authorized_user"
    req = rec.requests[0]
    assert req.url.path == "/fd/google/credentials"
    assert GRANT_HEADER not in req.headers and "fd_member" not in req.url.params


async def test_member_is_connected_asks_only_agent_status(monkeypatch, http):
    from captain_claw.google_oauth_manager import GoogleOAuthManager, is_google_connected_cached

    _fd(monkeypatch)
    rec = http(lambda r: httpx.Response(200, json={"connected": True, "enabled": True}))
    with _Bound(PRINCIPAL, GRANT):
        assert await GoogleOAuthManager(object()).is_connected() is True
        assert is_google_connected_cached() is True
        assert speaker.member_google_enabled() is True
    assert [r.url.path for r in rec.requests] == ["/fd/google/agent_status"]
    assert rec.requests[0].headers[GRANT_HEADER] == GRANT
    assert rec.requests[0].url.params["fd_member"] == "1"
    assert is_google_connected_cached() is False          # the owner's flag is untouched


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(200, json=["connected"]),
    httpx.Response(403, json={"detail": "No active shared-agent turn for this request"}),
    httpx.Response(404, json={"detail": "Not Found"}),
])
async def test_member_status_fails_closed(monkeypatch, http, response):
    from captain_claw.google_oauth_manager import GoogleOAuthManager, is_google_connected_cached

    _fd(monkeypatch)
    http(lambda r: response)
    with _Bound(PRINCIPAL, GRANT):
        assert await GoogleOAuthManager(object()).speaker_status() is False
        assert is_google_connected_cached() is False
        assert speaker.member_google_enabled() is False


async def test_member_status_without_grant_or_fd_makes_no_request(monkeypatch, http):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    rec = http()
    with _Bound(PRINCIPAL, GRANT):                       # no FD URL: not an FD client
        assert await GoogleOAuthManager(object()).speaker_status() is False
    _fd(monkeypatch)
    with _Bound(PRINCIPAL, ""):                          # FD, but no grant
        assert await GoogleOAuthManager(object()).speaker_status() is False
    with _Bound(DOCKER, ""):
        assert await GoogleOAuthManager(object()).is_connected() is False
    assert rec.requests == []


async def test_google_caches_are_per_principal(monkeypatch, http):
    import captain_claw.google_oauth_manager as gom
    from captain_claw.google_oauth_manager import GoogleOAuthManager, is_google_connected_cached

    _fd(monkeypatch)
    http(lambda r: httpx.Response(200, json={
        "connected": r.headers.get(GRANT_HEADER) == GRANT, "enabled": True,
    }))
    with _Bound(PRINCIPAL, GRANT):
        assert await GoogleOAuthManager(object()).speaker_status() is True
    with _Bound(OTHER, GRANT_B):
        assert await GoogleOAuthManager(object()).speaker_status() is False
    with _Bound(PRINCIPAL, GRANT):
        assert is_google_connected_cached() is True
    with _Bound(OTHER, GRANT_B):
        assert is_google_connected_cached() is False
    assert is_google_connected_cached() is False         # owner
    with _Bound(UNKNOWN_PRINCIPAL, ""):
        gom._mark_google_connected(True)                 # never written for UNKNOWN
        assert is_google_connected_cached() is False
    assert set(gom._GOOGLE_CONNECTED) == {"spk:u-member", "spk:u-other"}

    # The registry's Google gate follows: only A sees Google tools.
    reg = ToolRegistry()
    reg.register(_Probe("google_mail"), metadata={"requires_google": True})
    reg.register(_Probe("web_search"))
    with _Bound(PRINCIPAL, GRANT):
        assert "google_mail" in reg.list_tools(session_id="x")
    with _Bound(OTHER, GRANT_B):
        assert "google_mail" not in reg.list_tools(session_id="x")
    assert "google_mail" not in reg.list_tools(session_id="owner")


# ── local mode: a member never reads the agent's (owner's) tokens ────


class _AppStateSM:
    def __init__(self):
        self.reads: list[str] = []

    async def get_app_state(self, key):
        self.reads.append(key)
        return json.dumps({"access_token": "owner-at", "refresh_token": "owner-rt",
                           "token_type": "Bearer", "expires_at": time.time() + 3600,
                           "scope": "https://www.googleapis.com/auth/gmail.send"})

    async def set_app_state(self, key, value):
        raise AssertionError("no writes expected")


async def test_local_mode_member_never_reads_app_state(monkeypatch, http):
    from captain_claw.google_oauth_manager import FlightDeckRefused, GoogleOAuthManager

    sm = _AppStateSM()
    rec = http()
    with _Bound(PRINCIPAL, GRANT):
        with pytest.raises(FlightDeckRefused):
            await GoogleOAuthManager(sm).get_tokens()
        assert await GoogleOAuthManager(sm).is_connected() is False
    assert sm.reads == [] and rec.requests == []
    # The owner's path is unchanged.
    from captain_claw.google_oauth import STATE_KEY_TOKENS

    tokens = await GoogleOAuthManager(sm).get_tokens()
    assert tokens.access_token == "owner-at" and sm.reads == [STATE_KEY_TOKENS]


async def test_local_mode_drive_token_provider_refuses_a_member(monkeypatch):
    from captain_claw.drive_client import DriveNotConnected, global_token_provider

    sm = _AppStateSM()
    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: sm)
    with _Bound(PRINCIPAL, GRANT):
        with pytest.raises(DriveNotConnected):
            await global_token_provider()
    assert sm.reads == []
    assert (await global_token_provider())[0] == "owner-at"     # owner unchanged


async def test_local_mode_member_send_never_happens(monkeypatch, http):
    from captain_claw.tools.google_mail import GoogleMailTool

    sm = _AppStateSM()
    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: sm)
    monkeypatch.setattr(get_config().tools.google_mail, "allow_send", True)
    rec = http()
    tool = GoogleMailTool()
    with _Bound(PRINCIPAL, GRANT):
        result = await tool.execute(action="send", to="bob@example.com", subject="s", body="b")
    assert result.success is False and result.error.startswith("Email not sent")
    assert NO_GRANT_MESSAGE in result.error
    assert rec.requests == [] and sm.reads == []


async def test_fd_mode_member_send_carries_grant_and_marker(monkeypatch, http):
    from captain_claw.tools.google_mail import GoogleMailTool

    _fd(monkeypatch)
    rec = http(lambda r: httpx.Response(200, json={
        "to": "bob@example.com", "subject": "s", "message_id": "m1", "thread_id": "t1",
    }))
    tool = GoogleMailTool()
    with _Bound(PRINCIPAL, GRANT):
        result = await tool.execute(action="send", to="bob@example.com", subject="s", body="b")
    assert result.success, result.error
    req = rec.requests[0]
    assert req.url.path == "/fd/google/gmail/send"
    assert req.headers[GRANT_HEADER] == GRANT and req.url.params["fd_member"] == "1"
    assert req.headers["X-Agent-Auth"] == WEB_AUTH


async def test_fd_mode_member_send_without_grant_sends_nothing(monkeypatch, http):
    from captain_claw.tools.google_mail import GoogleMailTool

    _fd(monkeypatch)
    rec = http()
    tool = GoogleMailTool()
    with _Bound(PRINCIPAL, ""):
        result = await tool.execute(action="send", to="bob@example.com", subject="s", body="b")
    assert result.success is False and result.error == f"Email not sent: {NO_GRANT_MESSAGE}"
    assert rec.requests == []


async def test_fd_mode_owner_send_is_unchanged(monkeypatch, http):
    from captain_claw.tools.google_mail import GoogleMailTool

    _fd(monkeypatch)
    rec = http(lambda r: httpx.Response(200, json={"message_id": "m1"}))
    result = await GoogleMailTool().execute(action="send", to="bob@example.com", subject="s", body="b")
    assert result.success
    assert GRANT_HEADER not in rec.requests[0].headers
    assert "fd_member" not in rec.requests[0].url.params


# ── deep memory: the member's own pool through FD, by reference ──────


def _deep_ok(r):
    if r.url.path.endswith("/search"):
        return httpx.Response(200, json={"results": []})
    if r.url.path.endswith("/delete"):
        return httpx.Response(200, json={"deleted": 1})
    return httpx.Response(200, json={"chunks": 1, "reference": "r"})


async def test_member_deep_memory_carries_grant_and_marker(monkeypatch, http):
    from captain_claw.tools.typesense import TypesenseTool

    _fd(monkeypatch)
    rec = http(_deep_ok)
    tool = TypesenseTool()
    with _Bound(PRINCIPAL, GRANT):
        assert (await tool.execute(action="search", query="q")).success
        assert (await tool.execute(action="index", text="hello", reference="note-1")).success
        assert (await tool.execute(action="delete", reference="note-1")).success
    assert [r.url.path for r in rec.requests] == [
        "/fd/deep-memory/agent/search", "/fd/deep-memory/agent/index",
        "/fd/deep-memory/agent/delete",
    ]
    for r in rec.requests:
        assert r.headers[GRANT_HEADER] == GRANT and r.url.params["fd_member"] == "1"
        assert r.headers["X-Agent-Auth"] == WEB_AUTH


async def test_member_index_reference_is_never_a_host_path(monkeypatch, http, tmp_path):
    from captain_claw.tools.typesense import TypesenseTool

    _fd(monkeypatch)
    rec = http(_deep_ok)
    doc = tmp_path / "fd-data" / "vfs" / "u-member" / "p" / "a.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("member notes")
    elsewhere = tmp_path / "workspace" / "saved" / "tmp" / SPK_SESSION / "x.md"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("saved notes")
    tool = TypesenseTool()
    with _Bound(PRINCIPAL, GRANT):
        assert (await tool.execute(action="index", file_path=str(doc))).success
        assert (await tool.execute(action="index", file_path=str(elsewhere))).success
    assert (await tool.execute(action="index", file_path=str(elsewhere))).success   # owner
    refs = [json.loads(r.content)["reference"] for r in rec.requests]
    assert refs == ["vfs:p/a.md", "x.md", str(elsewhere)]


async def test_member_deep_memory_without_grant_is_refused_locally(monkeypatch, http):
    from captain_claw.tools.typesense import TypesenseTool

    _fd(monkeypatch)
    rec = http(_deep_ok)
    with _Bound(PRINCIPAL, ""):
        result = await TypesenseTool().execute(action="search", query="q")
    assert result.success is False and result.error == NO_GRANT_MESSAGE
    assert rec.requests == []


async def test_member_known_only_by_agent_never_reaches_fd_as_the_owner(monkeypatch, http):
    """Called directly with a member's `_agent` and nothing bound (no registry
    to bind the principal + grant): with no principal in context the request
    would carry no grant — i.e. go out as the OWNER — so it is refused locally,
    even when the instance holds a grant."""
    from captain_claw.tools.typesense import TypesenseTool

    _fd(monkeypatch)
    rec = http(_deep_ok)
    for agent in (_member_agent(grant=GRANT), _member_agent(grant="")):
        result = await TypesenseTool().execute(action="search", query="q", _agent=agent)
        assert result.success is False and result.error == NO_GRANT_MESSAGE
    assert rec.requests == []


async def test_grant_missing_inside_a_proxied_call_is_the_no_grant_message(monkeypatch, http):
    from captain_claw.tools.typesense import TypesenseTool

    _fd(monkeypatch)
    http(_deep_ok)
    tool = TypesenseTool()

    async def _raise(**kw):
        raise SpeakerGrantMissing("lost")

    tool._fd_search = _raise
    result = await tool.execute(action="search", query="q")
    assert result.error == NO_GRANT_MESSAGE and "unavailable" not in result.error


async def test_member_deep_memory_outside_fd_never_uses_the_local_key(monkeypatch, http):
    from captain_claw.tools.typesense import TypesenseTool

    rec = http()
    monkeypatch.setattr(get_config().tools.typesense, "api_key", "OWNER-LOCAL-KEY")
    tool = TypesenseTool()
    with _Bound(PRINCIPAL, GRANT):
        result = await tool.execute(action="search", query="q")
    assert result.success is False and "shared chats" in result.error
    result = await tool.execute(action="search", query="q", _agent=_member_agent(grant=GRANT))
    assert result.success is False and "shared chats" in result.error
    assert rec.requests == []


async def test_member_filter_delete_is_refused_locally(monkeypatch, http):
    from captain_claw.tools.typesense import TypesenseTool

    _fd(monkeypatch)
    rec = http(_deep_ok)
    reg = ToolRegistry()
    reg.register(TypesenseTool())
    with _Bound(PRINCIPAL, GRANT):
        for args in ({"action": "delete", "filter_by": "source:=x"},
                     {"action": "delete", "filter_by": "source:=x", "reference": "r1"},
                     {"action": "delete"}):
            with pytest.raises(ToolBlockedError) as info:
                await reg.execute("typesense", dict(args), session_id="x")
            assert speaker.MEMBER_DELETE_MESSAGE in str(info.value)
        # The tool itself refuses too (defence in depth).
        direct = await TypesenseTool().execute(action="delete", filter_by="source:=x")
        assert direct.success is False and direct.error == speaker.MEMBER_DELETE_MESSAGE
    assert rec.requests == []


async def test_owner_deep_memory_is_unchanged(monkeypatch, http):
    from captain_claw.tools.typesense import TypesenseTool

    _fd(monkeypatch)
    rec = http(_deep_ok)
    assert (await TypesenseTool().execute(action="delete", filter_by="source:=x")).success
    req = rec.requests[0]
    assert GRANT_HEADER not in req.headers and "fd_member" not in req.url.params
    assert json.loads(req.content)["filter_by"] == "source:=x"


# ── prompt / orchestration ───────────────────────────────────────────


def test_member_google_refresh_never_fetches_a_token():
    """agent_orchestration_mixin: a member's per-iteration refresh is
    speaker_status() (throttled), the owner's is_connected() is unchanged."""
    from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

    src = inspect.getsource(AgentOrchestrationMixin)
    status = src.index("GoogleOAuthManager(self.session_manager).speaker_status()")
    gate = src.rfind('if getattr(self, "_speaker_scoped", False) is True:', 0, status)
    owner = src.index("GoogleOAuthManager(self.session_manager).is_connected()", status)
    assert gate != -1 and gate > src.rfind("def ", 0, status)
    assert "google_cache_fresh(SPEAKER_GOOGLE_STATUS_REFRESH_S)" in src[gate:status]
    assert "else:" in src[status:owner]
