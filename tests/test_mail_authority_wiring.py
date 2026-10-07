"""PR E (agent part): the mail-write guard is wired into every entry point.

* google_mail / send_mail / Gmail-like MCP proxy tools refuse an automated
  turn's unrequested email BEFORE any token fetch or HTTP call;
* the automation marker is received on WS chat / run_tool frames, the
  ``/api/tool`` body, the inbound relay queue, cron runs, sister / BotPort /
  Telegram relays, and bound for the turn;
* stored (cron) and forwarded (peer) intent comes from the bound human text;
* the tool-avoidance nudge and the stall nag never push an email nobody asked
  for; the prompts say so.

No network: Gmail / Flight Deck answer through httpx.MockTransport; no real
session DB (fakes), HOME and the FD home are tmp dirs.
"""

from __future__ import annotations

import asyncio
import json
import time
import types
from pathlib import Path

import httpx
import pytest

from captain_claw import mail_authority as ma

_WORKER_ENVS = (
    "CLAW_BEING_WORKER", "CLAW_BASNA_WORKER", "CLAW_VATRA_WORKER",
    "CLAW_COUNCIL_WORKER", "CLAW_CODE_AGENT",
)
_RealAsyncClient = httpx.AsyncClient
FD = "http://fd.test"


# ── fixtures (copied, not imported: tests/ has no __init__.py) ───────


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    from captain_claw.flight_deck import agent_secret

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    yield
    agent_secret.reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _no_worker_env(monkeypatch):
    for name in _WORKER_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    import captain_claw.config as config_mod
    from captain_claw.config import Config

    cfg = Config()
    monkeypatch.setattr(config_mod, "_config", cfg)
    return cfg


@pytest.fixture
def _no_real_session_manager(monkeypatch):
    # GoogleOAuthManager(get_session_manager()) — never touch a DB. Not
    # autouse: the cron tests use their own fakes.
    import captain_claw.session as session_mod

    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())


# ── Gmail (as tests/test_gmail_send_tool.py) ─────────────────────────

_G = "https://www.googleapis.com/auth/"
READ = _G + "gmail.readonly"
COMPOSE = _G + "gmail.compose"


def _set_mode(monkeypatch, *, fd: bool, scope: str = f"{READ} {COMPOSE}") -> dict:
    from captain_claw.google_oauth import GoogleOAuthTokens
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    calls = {"get_tokens": 0}
    monkeypatch.setattr(
        GoogleOAuthManager, "_flight_deck_base",
        staticmethod(lambda: "http://localhost:25080" if fd else ""),
    )

    async def _get_tokens(self):
        calls["get_tokens"] += 1
        return GoogleOAuthTokens(access_token="T", refresh_token="",
                                 expires_at=time.time() + 3300, scope=scope)

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)
    return calls


class _Gmail:
    def __init__(self, routes: dict):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.split("/gmail/v1/users/me", 1)[-1]
        status, body = self.routes.get((request.method, path), (404, {"error": {"message": "nope"}}))
        return httpx.Response(status, json=body)


def _gmail_tool(gmail: _Gmail):
    from captain_claw.tools.google_mail import GoogleMailTool

    tool = GoogleMailTool()
    tool._client = _RealAsyncClient(transport=httpx.MockTransport(gmail))
    return tool


_DRAFT_OK = (200, {"id": "r-9", "message": {"id": "m9", "threadId": "t9"}})


@pytest.mark.usefixtures("_no_real_session_manager")
class TestGoogleMailGuard:
    @pytest.mark.parametrize("action", ["create_draft", "update_draft", "send", "send_draft"])
    async def test_refused_without_any_token_or_http(self, monkeypatch, _config, action):
        _config.tools.google_mail.allow_send = True
        calls = _set_mode(monkeypatch, fd=False)
        gmail = _Gmail({("POST", "/drafts"): _DRAFT_OK})
        with ma.bound(ma.automated("cron", "summarize my inbox")):
            res = await _gmail_tool(gmail).execute(
                action, to="bob@x.co", subject="s", body="b", draft_id="d1",
            )
        assert not res.success
        assert res.error.startswith("[not-authorized: mail-write]")
        assert gmail.requests == [] and calls["get_tokens"] == 0

    @pytest.mark.parametrize("action", ["send", "send_draft"])
    async def test_fd_sends_refused_without_an_fd_post(self, monkeypatch, action):
        calls = _set_mode(monkeypatch, fd=True)
        seen: list = []
        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kw: _RealAsyncClient(
                transport=httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200, json={})),
                **kw),
        )
        with ma.bound(ma.automated("autonomy_tool", "", "deny")):
            res = await _gmail_tool(_Gmail({})).execute(action, to="bob@x.co", subject="s",
                                                        body="b", draft_id="d1")
        assert not res.success and ma.is_refusal(res.error)
        assert "Autonomous Work" in res.error
        assert seen == [] and calls["get_tokens"] == 0

    async def test_job_text_that_asks_reaches_gmail(self, monkeypatch, _config):
        _config.tools.google_mail.repeat_check_days = 0
        _set_mode(monkeypatch, fd=False)
        gmail = _Gmail({("POST", "/drafts"): _DRAFT_OK})
        with ma.bound(ma.automated("cron", "draft a reply to bob@x.co")):
            res = await _gmail_tool(gmail).execute("create_draft", to="bob@x.co", subject="s", body="b")
        assert res.success, res.error
        assert [(r.method, r.url.path.rsplit("/", 1)[-1]) for r in gmail.requests] == [("POST", "drafts")]

    async def test_self_scope_only_to_the_owner(self, monkeypatch, _config):
        _config.tools.google_mail.repeat_check_days = 0
        _set_mode(monkeypatch, fd=False)
        routes = {("GET", "/profile"): (200, {"emailAddress": "Me@x.co"}),
                  ("POST", "/drafts"): _DRAFT_OK}
        job = ma.automated("cron", "email me the inbox summary every morning")

        gmail = _Gmail(routes)
        with ma.bound(job):
            res = await _gmail_tool(gmail).execute("create_draft", to="Ana <ana@x.co>",
                                                   subject="s", body="b")
        assert not res.success and res.error == ma.refusal_text_self("cron", "created")
        assert not [r for r in gmail.requests if r.method == "POST"]

        gmail = _Gmail(routes)
        with ma.bound(job):
            res = await _gmail_tool(gmail).execute("create_draft", to="me@x.co", subject="s", body="b")
        assert res.success, res.error
        assert [r.method for r in gmail.requests if r.method == "POST"] == ["POST"]

        gmail = _Gmail(routes)
        with ma.bound(job):
            res = await _gmail_tool(gmail).execute("create_draft", to="me@x.co", body="b",
                                                   reply_to_message_id="m1")
        assert not res.success and ma.is_refusal(res.error)
        assert not [r for r in gmail.requests if r.method == "POST"]

        gmail = _Gmail({("GET", "/profile"): (500, {"error": {"message": "boom"}}),
                        ("POST", "/drafts"): _DRAFT_OK})
        with ma.bound(job):
            res = await _gmail_tool(gmail).execute("create_draft", to="me@x.co", subject="s", body="b")
        assert not res.success and ma.is_refusal(res.error)
        assert not [r for r in gmail.requests if r.method == "POST"]

    async def test_self_scope_reaches_an_address_the_job_names(self, monkeypatch, _config):
        """ "email me at stevica@company.com …" — not only the Gmail primary."""
        _config.tools.google_mail.repeat_check_days = 0
        _set_mode(monkeypatch, fd=False)
        routes = {("GET", "/profile"): (200, {"emailAddress": "kstevica@gmail.com"}),
                  ("POST", "/drafts"): _DRAFT_OK}
        job = ma.automated("cron", "Every morning email me at stevica@company.com the inbox summary")
        for to in ("stevica@company.com", "kstevica@gmail.com"):
            gmail = _Gmail(routes)
            with ma.bound(job):
                res = await _gmail_tool(gmail).execute("create_draft", to=to, subject="s", body="b")
            assert res.success, (to, res.error)
            assert [r.method for r in gmail.requests if r.method == "POST"] == ["POST"]
        gmail = _Gmail(routes)
        with ma.bound(job):
            res = await _gmail_tool(gmail).execute("create_draft", to="ana@x.co", subject="s", body="b")
        assert not res.success and res.error == ma.refusal_text_self("cron", "created")
        assert not [r for r in gmail.requests if r.method == "POST"]

    async def test_own_address_is_cached_per_token(self, monkeypatch, _config):
        # One tool instance can serve several mailboxes (a shared agent's
        # members): the owner's address must never be reused for another token.
        from captain_claw.google_oauth import GoogleOAuthTokens
        from captain_claw.google_oauth_manager import GoogleOAuthManager

        _set_mode(monkeypatch, fd=False)
        current_token = {"v": "T-owner"}

        async def _get_tokens(self):
            return GoogleOAuthTokens(access_token=current_token["v"], refresh_token="",
                                     expires_at=time.time() + 3300, scope=f"{READ} {COMPOSE}")

        monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)

        def _profile(request: httpx.Request) -> httpx.Response:
            who = {"Bearer T-owner": "owner@x.co", "Bearer T-member": "member@x.co"}
            return httpx.Response(200, json={"emailAddress": who[request.headers["Authorization"]]})

        from captain_claw.tools.google_mail import GoogleMailTool

        tool = GoogleMailTool()
        tool._client = _RealAsyncClient(transport=httpx.MockTransport(_profile))
        assert await tool._own_addresses() == {"owner@x.co"}
        current_token["v"] = "T-member"
        assert await tool._own_addresses() == {"member@x.co"}

    async def test_human_turn_unaffected(self, monkeypatch, _config):
        _config.tools.google_mail.repeat_check_days = 0
        _set_mode(monkeypatch, fd=False)
        gmail = _Gmail({("POST", "/drafts"): _DRAFT_OK})
        with ma.bound(ma.human("summarize my inbox")):
            res = await _gmail_tool(gmail).execute("create_draft", to="bob@x.co", subject="s", body="b")
        assert res.success, res.error
        res = await _gmail_tool(gmail).execute("create_draft", to="bob@x.co", subject="s2", body="b")
        assert res.success, res.error

    def test_followup_hint_reply_bullet_only_when_writes_allowed(self):
        from captain_claw.tools import google_mail as gm

        assert gm._followup_hint() == gm._FOLLOWUP_HINT
        with ma.bound(ma.automated("cron", "summarize my inbox")):
            assert gm._followup_hint() == gm._FOLLOWUP_HINT_NO_REPLY
        with ma.bound(ma.automated("cron", "email me the summary")):
            assert gm._followup_hint() == gm._FOLLOWUP_HINT_NO_REPLY
        with ma.bound(ma.automated("cron", "draft a reply to Ana")):
            assert gm._followup_hint() == gm._FOLLOWUP_HINT


# ── send_mail ────────────────────────────────────────────────────────


class TestSendMailGuard:
    def _tool(self, monkeypatch, _config, sent: list):
        from captain_claw.tools.send_mail import SendMailTool

        _config.tools.send_mail.provider = "smtp"
        _config.tools.send_mail.from_address = "agent@x.co"
        tool = SendMailTool()

        async def _smtp(self, mail_cfg, from_header, from_address, to, cc, bcc, *a, **kw):
            sent.append((list(to), list(cc), list(bcc)))
            return "ok"

        monkeypatch.setattr(SendMailTool, "_send_smtp", _smtp)
        return tool

    async def test_refused_under_deny_without_sending(self, monkeypatch, _config):
        sent: list = []
        tool = self._tool(monkeypatch, _config, sent)
        with ma.bound(ma.automated("sister", "", "deny")):
            res = await tool.execute(to=["bob@x.co"], subject="s", body="b")
        assert not res.success and ma.is_refusal(res.error)
        assert "a background task" in res.error
        assert sent == []

    async def test_self_scope_one_recipient(self, monkeypatch, _config):
        sent: list = []
        tool = self._tool(monkeypatch, _config, sent)
        with ma.bound(ma.automated("cron", "email me the summary every morning")):
            ok = await tool.execute(to=["me@x.co"], subject="s", body="b")
            two = await tool.execute(to=["me@x.co", "ana@x.co"], subject="s", body="b")
            cc = await tool.execute(to=["me@x.co"], cc=["ana@x.co"], subject="s", body="b")
        assert ok.success, ok.error
        assert not two.success and ma.is_refusal(two.error)
        assert not cc.success and ma.is_refusal(cc.error)
        assert sent == [(["me@x.co"], [], [])]

    async def test_human_unaffected(self, monkeypatch, _config):
        sent: list = []
        tool = self._tool(monkeypatch, _config, sent)
        res = await tool.execute(to=["a@x.co", "b@x.co"], subject="s", body="b")
        assert res.success, res.error


# ── MCP proxy ────────────────────────────────────────────────────────


class _Connector:
    def __init__(self):
        self.calls: list = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return "done"


def _mcp(tool_name: str, server: str, description: str = ""):
    from captain_claw.tools.mcp_connector import MCPProxyTool

    conn = _Connector()
    tool = MCPProxyTool(tool_name, description, {"type": "object", "properties": {
        "to": {"type": "string"}}}, server, conn)
    return tool, conn


class TestMcpMailGuard:
    async def test_gmail_create_draft_refused_when_automated(self):
        for auth in (ma.automated("cron", "", "deny"),
                     ma.automated("cron", "email me the summary every morning")):
            tool, conn = _mcp("create_draft", "gmail", "Create a Gmail draft")
            with ma.bound(auth):
                res = await tool.execute(to="me@x.co")
            assert not res.success and ma.is_refusal(res.error)
            assert conn.calls == []

    async def test_proceeds_in_a_human_turn(self):
        tool, conn = _mcp("create_draft", "gmail", "Create a Gmail draft")
        res = await tool.execute(to="me@x.co")
        assert res.success and conn.calls == [("create_draft", {"to": "me@x.co"})]

    async def test_any_scope_proceeds(self):
        tool, conn = _mcp("send_email", "outlook", "")
        with ma.bound(ma.automated("cron", "draft a reply to Ana")):
            res = await tool.execute(to="ana@x.co")
        assert res.success and conn.calls

    async def test_non_mail_server_unaffected(self):
        tool, conn = _mcp("send_message", "slack", "Post a message to a channel")
        with ma.bound(ma.automated("cron", "", "deny")):
            res = await tool.execute(to="#general")
        assert res.success and conn.calls

    @pytest.mark.parametrize("name", [
        "gmail_send_email", "sendEmail", "users.drafts.create", "create_gmail_draft",
        "send_gmail_message", "replyToEmail",
        # Microsoft Graph (Outlook MCP servers): these create drafts / send
        "createReply", "createReplyAll", "createForward", "create_reply", "create_message",
        "sendMail", "me.sendMail", "me.messages.createReply", "schedule_send", "batch_send",
    ])
    async def test_prefixed_and_camel_case_mail_writes_refused(self, name):
        tool, conn = _mcp(name, "gmail", "Gmail")
        with ma.bound(ma.automated("cron", "", "deny")):
            res = await tool.execute(to="ana@x.co")
        assert not res.success and ma.is_refusal(res.error) and conn.calls == []

    @pytest.mark.parametrize("name", ["message_send", "message_reply", "message_create",
                                      "message_forward", "save_draft", "saveDraft", "upsert_draft"])
    async def test_singular_and_save_draft_names_refused(self, name):
        tool, conn = _mcp(name, "outlook", "Outlook mail")
        with ma.bound(ma.automated("cron", "", "deny")):
            res = await tool.execute(to="ana@x.co")
        assert not res.success and ma.is_refusal(res.error) and conn.calls == []

    @pytest.mark.parametrize("name", ["list_drafts", "get_draft", "search_threads", "get_thread",
                                      "listMessages", "createMailFolder", "createMessageRule",
                                      "draft_list", "drafts_get", "Drafts_List"])
    async def test_mail_reads_still_run(self, name):
        tool, conn = _mcp(name, "gmail", "Gmail")
        with ma.bound(ma.automated("cron", "", "deny")):
            res = await tool.execute(to="x")
        assert res.success and conn.calls


# ── ws_handler ───────────────────────────────────────────────────────


class _GuardAgent:
    def __init__(self):
        self.seen: list = []
        self.plan_mode_auto = False
        self.session_manager = object()

    async def _execute_tool_with_guard(self, tool, args, *a, **kw):
        self.seen.append(ma.current())
        return types.SimpleNamespace(success=True, content="ok", error=None)


def _ws_server(agent=None):
    sent: list = []

    async def _send(ws, msg):
        sent.append(msg)

    return types.SimpleNamespace(
        agent=agent or _GuardAgent(), _send=_send, sent=sent,
        _broadcast=lambda msg: None, _telegram_agents={}, _telegram_user_locks={},
        _telegram_bridge=None,
    )


@pytest.fixture
def _no_google_refresh(monkeypatch):
    from captain_claw.google_oauth_manager import GoogleOAuthManager

    async def _connected(self):
        return True

    monkeypatch.setattr(GoogleOAuthManager, "is_connected", _connected)


@pytest.mark.usefixtures("_no_google_refresh")
class TestWsHandler:
    async def test_run_tool_without_marker_is_deny(self):
        from captain_claw.web.ws_handler import handle_ws_message

        server = _ws_server()
        await handle_ws_message(server, object(), {"type": "run_tool", "tool": "google_mail",
                                                   "args": {"action": "create_draft"}})
        a = server.agent.seen[-1]
        assert (a.mode, a.kind, a.mail_write) == ("automated", "autonomy_tool", "deny")
        assert ma.current() == ma.HUMAN

    async def test_run_tool_allow(self):
        from captain_claw.web.ws_handler import handle_ws_message

        server = _ws_server()
        await handle_ws_message(server, object(), {
            "type": "run_tool", "tool": "google_mail", "args": {},
            "automation": {"kind": "autonomy_tool", "mail_write": "allow"}})
        assert server.agent.seen[-1].mail_write == "allow"

    def _patch_routes(self, monkeypatch) -> dict:
        from captain_claw.web import chat_handler, plan_auto_route, slash_commands

        got: dict = {"chat": [], "command": [], "plan": []}

        async def _chat(server, ws, content, **kw):
            got["chat"].append((content, kw.get("automation")))

        async def _command(server, ws, content):
            got["command"].append(content)

        async def _plan(server, ws, content):
            got["plan"].append((content, ma.current()))

        monkeypatch.setattr(chat_handler, "handle_chat", _chat)
        monkeypatch.setattr(slash_commands, "handle_command", _command)
        monkeypatch.setattr(plan_auto_route, "handle_plan_auto_route", _plan)
        return got

    async def test_chat_frame_marker_reaches_handle_chat(self, monkeypatch):
        from captain_claw.web.ws_handler import handle_ws_message

        got = self._patch_routes(monkeypatch)
        server = _ws_server()
        await handle_ws_message(server, object(), {
            "type": "chat", "content": "check mail",
            "automation": {"kind": "fd_scheduler", "job_text": "summarize", "mail_write": "intent"}})
        await handle_ws_message(server, object(), {"type": "chat", "content": "hello"})
        assert got["chat"][0] == ("check mail", ma.Authority("automated", "fd_scheduler",
                                                             "summarize", "intent"))
        assert got["chat"][1] == ("hello", None)

    async def test_automated_slash_goes_to_chat(self, monkeypatch):
        from captain_claw.web.ws_handler import handle_ws_message

        got = self._patch_routes(monkeypatch)
        server = _ws_server()
        await handle_ws_message(server, object(), {
            "type": "chat", "content": "/new", "automation": {"kind": "flow"}})
        await handle_ws_message(server, object(), {"type": "chat", "content": "/new"})
        assert [c for c, _ in got["chat"]] == ["/new"]
        assert got["command"] == ["/new"]

    async def test_plan_auto_only_for_human_frames(self, monkeypatch):
        from captain_claw.web.ws_handler import handle_ws_message

        got = self._patch_routes(monkeypatch)
        server = _ws_server()
        server.agent.plan_mode_auto = True
        await handle_ws_message(server, object(), {
            "type": "chat", "content": "do it", "automation": {"kind": "autonomy",
                                                               "mail_write": "deny"}})
        await handle_ws_message(server, object(), {"type": "chat", "content": "plan this"})
        assert [c for c, _ in got["chat"]] == ["do it"]
        assert len(got["plan"]) == 1
        content, auth = got["plan"][0]
        assert content == "plan this" and auth.mode == "human" and auth.job_text == "plan this"
        assert ma.current() == ma.HUMAN

    async def test_plan_auto_in_a_worker_keeps_the_deny_default(self, monkeypatch):
        from captain_claw.web.ws_handler import handle_ws_message

        monkeypatch.setenv("CLAW_VATRA_WORKER", "1")
        got = self._patch_routes(monkeypatch)
        server = _ws_server()
        server.agent.plan_mode_auto = True
        await handle_ws_message(server, object(), {"type": "chat", "content": "draft the emails"})
        _content, auth = got["plan"][0]
        assert (auth.mode, auth.kind, auth.mail_write) == ("automated", "fd_worker", "deny")

    async def test_telegram_delegate_result_is_peer_relay_deny(self, monkeypatch):
        from captain_claw.web import telegram as tg
        from captain_claw.web.ws_handler import handle_ws_message

        async def _no_send(*a, **kw):
            return None

        monkeypatch.setattr(tg, "_tg_send", _no_send)
        seen: list = []
        done = asyncio.Event()

        async def _complete(content):
            seen.append(ma.current())
            done.set()
            return ""

        tg_agent = types.SimpleNamespace(session=None, complete=_complete)
        server = _ws_server()
        server._telegram_agents["u1"] = tg_agent
        await handle_ws_message(server, object(), {
            "type": "notification", "content": "[Delegated result from X] done",
            "trigger_response": True, "origin_platform": "telegram",
            "origin_user_id": "u1", "origin_chat_id": 5})
        await asyncio.wait_for(done.wait(), 5)
        assert (seen[0].kind, seen[0].mail_write) == ("peer_relay", "deny")


# ── chat_handler._run_agent ──────────────────────────────────────────


class _ChatAgent:
    def __init__(self):
        self.seen: list = []
        self.session = None

    def get_runtime_model_details(self):
        return {}

    async def complete(self, content):
        self.seen.append((ma.current(), content))
        return "reply"


class TestRunAgent:
    def _setup(self, monkeypatch):
        from captain_claw.web import chat_handler

        flows: list = []
        monkeypatch.setattr(chat_handler, "fire_and_forget_send", lambda ws, data: None)

        async def _flow(agent, text, **kw):
            flows.append(kw.get("automated"))
            return None

        monkeypatch.setattr(chat_handler, "_maybe_run_flow", _flow)
        server = types.SimpleNamespace(LANE_MAIN="A", _busy=True, _active_task=None)
        return chat_handler, server, flows

    async def test_human_turn_binds_the_message(self, monkeypatch):
        chat_handler, server, flows = self._setup(monkeypatch)
        agent = _ChatAgent()
        await chat_handler._run_agent(server, object(), agent, "hello there", no_broadcast=True)
        auth, content = agent.seen[0]
        assert auth.mode == "human" and auth.job_text == "hello there"
        assert content == "hello there"
        assert flows == [""]
        assert ma.current() == ma.HUMAN

    async def test_automated_turn_binds_and_prefixes(self, monkeypatch):
        chat_handler, server, flows = self._setup(monkeypatch)
        agent = _ChatAgent()
        auto = ma.automated("fd_scheduler", "summarize my inbox", "intent")
        await chat_handler._run_agent(server, object(), agent, "summarize my inbox",
                                      no_broadcast=True, automation=auto)
        auth, content = agent.seen[0]
        assert auth == auto
        assert content == (
            "[Automated turn — a scheduled job. Not a live message from the user.]\n"
            "summarize my inbox"
        )
        assert flows == ["fd_scheduler"]
        assert ma.current() == ma.HUMAN

    @pytest.mark.parametrize("env,kind", [
        ("CLAW_VATRA_WORKER", "fd_worker"), ("CLAW_BASNA_WORKER", "fd_worker"),
        ("CLAW_COUNCIL_WORKER", "fd_worker"), ("CLAW_CODE_AGENT", "fd_worker"),
        ("CLAW_BEING_WORKER", "being"),
    ])
    async def test_worker_plain_frame_keeps_the_deny_default(self, monkeypatch, env, kind):
        """J7: FD sends worker prompts and being ticks as plain chat frames —
        they must not become a human turn (that would let them write Gmail)."""
        from captain_claw.web import chat_handler

        monkeypatch.setenv(env, "1")
        flows: list = []
        monkeypatch.setattr(chat_handler, "fire_and_forget_send", lambda ws, data: None)

        async def _flow(agent, text, **kw):
            flows.append((kw.get("automated"), kw.get("automated_mail_write")))
            return None

        monkeypatch.setattr(chat_handler, "_maybe_run_flow", _flow)
        server = types.SimpleNamespace(LANE_MAIN="A", _busy=True, _active_task=None)
        seen: list = []

        class _Worker(_ChatAgent):
            async def complete(self, content):
                seen.append((ma.current(), content,
                             ma.check_mail_write("google_mail", "create_draft"),
                             ma.intent_source_text(self), ma.recent_intent_text(self)))
                return "reply"

        await chat_handler._run_agent(server, object(), _Worker(),
                                      "Draft emails to every investor about Q3", no_broadcast=True)
        auth, content, refusal, source, recent = seen[0]
        assert (auth.mode, auth.kind, auth.mail_write) == ("automated", kind, "deny")
        assert ma.is_refusal(refusal)
        assert source == "" and recent == ""
        assert content == "Draft emails to every investor about Q3"
        assert flows == [(kind, "deny")]
        assert ma.current() == auth  # the process default, nothing left bound

    async def test_flow_evaluate_body_marks_automated(self, monkeypatch):
        from captain_claw.web import chat_handler

        posts: list = []

        class _Resp:
            status_code = 200

            def json(self):
                return {"matched": False}

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, **kw):
                posts.append(json)
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        monkeypatch.setenv("FD_URL", FD)
        agent = types.SimpleNamespace(session=types.SimpleNamespace(metadata={}))
        await chat_handler._maybe_run_flow(agent, "digest", is_public=False, automated="cron",
                                           automated_mail_write="intent")
        await chat_handler._maybe_run_flow(agent, "digest", is_public=False)
        await chat_handler._maybe_run_flow(agent, "digest", is_public=False, automated="autonomy",
                                           automated_mail_write="deny")
        assert (posts[0]["automated"], posts[0]["automated_mail_write"]) == ("cron", "intent")
        assert "automated" not in posts[1] and "automated_mail_write" not in posts[1]
        assert (posts[2]["automated"], posts[2]["automated_mail_write"]) == ("autonomy", "deny")
        # a caller that doesn't say (the /flow slash command) in a worker or a
        # being: the process default goes with it
        for env, kind in (("CLAW_VATRA_WORKER", "fd_worker"), ("CLAW_BEING_WORKER", "being")):
            monkeypatch.setenv(env, "1")
            await chat_handler._maybe_run_flow(agent, "/flow status", is_public=False)
            monkeypatch.delenv(env)
            assert (posts[-1]["automated"], posts[-1]["automated_mail_write"]) == (kind, "deny")

    async def test_run_agent_sends_the_turns_mail_write_to_flows(self, monkeypatch):
        chat_handler, server, _flows = self._setup(monkeypatch)
        got: list = []

        async def _flow(agent, text, **kw):
            got.append((kw.get("automated"), kw.get("automated_mail_write")))
            return None

        monkeypatch.setattr(chat_handler, "_maybe_run_flow", _flow)
        for auto in (ma.automated("autonomy", "", "deny"), ma.automated("fd_scheduler", "x", "intent"),
                     None):
            await chat_handler._run_agent(server, object(), _ChatAgent(), "hi", no_broadcast=True,
                                          automation=auto)
        assert got == [("autonomy", "deny"), ("fd_scheduler", "intent"), ("", "")]


# ── web_server ───────────────────────────────────────────────────────


class TestWebServer:
    async def test_api_tool_binds(self):
        from captain_claw.web_server import WebServer

        agent = _GuardAgent()
        fake = types.SimpleNamespace(agent=agent)

        class _Req:
            def __init__(self, body):
                self._body = body
                self.headers = {}

            async def json(self):
                return self._body

        await WebServer._api_tool(fake, _Req({"tool": "google_mail", "args": {}}))
        await WebServer._api_tool(fake, _Req({"tool": "google_mail", "args": {},
                                              "automation": {"kind": "flow_tool",
                                                             "mail_write": "allow"}}))
        assert (agent.seen[0].kind, agent.seen[0].mail_write) == ("flow_tool", "deny")
        assert (agent.seen[1].kind, agent.seen[1].mail_write) == ("flow_tool", "allow")
        assert ma.current() == ma.HUMAN

    async def test_inbound_queue_relays_as_peer_relay_deny(self, monkeypatch):
        from captain_claw.web import chat_handler
        from captain_claw.web_server import WebServer

        got: list = []
        done = asyncio.Event()

        async def _chat(server, ws, content, **kw):
            got.append((content, kw.get("automation")))
            done.set()

        monkeypatch.setattr(chat_handler, "handle_chat", _chat)
        fake = types.SimpleNamespace(_inbound_queue=asyncio.Queue(), _busy=False,
                                     clients=[object()], agent=None)
        task = asyncio.create_task(WebServer._inbound_queue_consumer(fake))
        fake._inbound_queue.put_nowait("[Delegated result from X] hi")
        await asyncio.wait_for(done.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        content, auth = got[0]
        assert content.startswith("[Delegated result")
        assert (auth.kind, auth.mail_write) == ("peer_relay", "deny")


# ── cron ─────────────────────────────────────────────────────────────


class _Ui:
    def can_capture_escape(self):
        return False

    def __getattr__(self, name):
        return lambda *a, **kw: None


class _CronAgent:
    def __init__(self, jobs: dict):
        self.seen: list = []
        self.session = None
        self._turn_is_automated = False
        self.session_manager = types.SimpleNamespace(load_cron_job=self._load)
        self._jobs = jobs

    async def _load(self, job_id):
        return self._jobs.get(job_id)

    def _record_timing_event(self, kind):
        return None

    async def complete(self, prompt):
        self.seen.append(ma.current())
        return ""

    async def stream(self, prompt):
        self.seen.append(ma.current())
        if False:
            yield ""


@pytest.fixture
def _quiet_cron(monkeypatch):
    from captain_claw import cron_dispatch

    async def _noop(*a, **kw):
        return None

    monkeypatch.setattr(cron_dispatch, "cron_chat_event", _noop)
    monkeypatch.setattr(cron_dispatch, "cron_monitor_event", _noop)


@pytest.mark.usefixtures("_quiet_cron")
class TestCron:
    async def _run(self, agent, prompt, **kw):
        from captain_claw.prompt_execution import run_prompt_in_active_session

        ctx = types.SimpleNamespace(agent=agent, ui=_Ui(), on_cron_output=None, last_next_steps=[])
        await run_prompt_in_active_session(ctx, prompt, queue=False, **kw)
        assert ma.current() == ma.HUMAN

    async def test_cron_run_judged_on_stored_job_text(self):
        job = types.SimpleNamespace(payload={"text": "draft a reply to Ana"})
        agent = _CronAgent({"J": job})
        await self._run(agent, "draft a reply to Ana", cron_job_id="J")
        a = agent.seen[0]
        assert (a.mode, a.kind, a.job_text, a.mail_write) == (
            "automated", "cron", "draft a reply to Ana", "intent")

    async def test_agent_written_job_uses_narrower_text(self):
        job = types.SimpleNamespace(payload={
            "text": "Summarize my inbox daily", "author": "agent",
            "mail_intent_text": "draft a reply to Ana and set up a daily inbox summary"})
        agent = _CronAgent({"J": job})
        await self._run(agent, "Summarize my inbox daily", cron_job_id="J")
        assert agent.seen[0].job_text == ""
        assert ma.check_mail_write("google_mail", "create_draft") is None  # back to human
        with ma.bound(agent.seen[0]):
            assert ma.is_refusal(ma.check_mail_write("google_mail", "create_draft"))

    async def test_missing_job_is_deny(self):
        agent = _CronAgent({})
        await self._run(agent, "draft a reply to Ana", cron_job_id="gone")
        assert (agent.seen[0].kind, agent.seen[0].mail_write) == ("cron", "deny")

    async def test_no_cron_id_binds_the_human_prompt(self):
        agent = _CronAgent({})
        await self._run(agent, "napravi draft za Anu")
        a = agent.seen[0]
        assert a.mode == "human" and a.job_text == "napravi draft za Anu"

    async def test_no_cron_id_in_a_worker_keeps_the_deny_default(self, monkeypatch):
        monkeypatch.setenv("CLAW_BASNA_WORKER", "1")
        from captain_claw.prompt_execution import run_prompt_in_active_session

        agent = _CronAgent({})

        ctx = types.SimpleNamespace(agent=agent, ui=_Ui(), on_cron_output=None, last_next_steps=[])
        await run_prompt_in_active_session(ctx, "draft a reply to Ana", queue=False)
        a = agent.seen[0]
        assert (a.mode, a.kind, a.mail_write) == ("automated", "fd_worker", "deny")

    def _cron_tool(self, monkeypatch, stored: list):
        from captain_claw.tools.cron_tool import CronTool

        async def _create(**kw):
            stored.append(kw["payload"])
            return types.SimpleNamespace(id="J1", next_run_at="2026-10-08T09:00:00Z")

        tool = CronTool()
        tool._agent = types.SimpleNamespace(
            session=types.SimpleNamespace(id="s1"), _turn_user_text="draft a reply to Bob")
        monkeypatch.setattr(tool, "_get_session_manager",
                            lambda: types.SimpleNamespace(create_cron_job=_create))
        return tool

    async def test_cron_tool_stores_provenance(self, monkeypatch):
        stored: list = []
        tool = self._cron_tool(monkeypatch, stored)
        with ma.bound(ma.human("napravi draft za Anu svaki petak")):
            res = await tool.execute("create", schedule="weekly fri 09:00",
                                     task="Draft a reply to Ana every Friday")
        assert res.success, res.error
        assert stored[0] == {"text": "Draft a reply to Ana every Friday", "author": "agent",
                             "mail_intent_text": "napravi draft za Anu svaki petak"}
        with ma.bound(ma.automated("autonomy_tool", "", "deny")):
            await tool.execute("create", schedule="daily 09:00", task="Draft replies")
        assert stored[1]["mail_intent_text"] == ""
        # unbound human: the attribute is never used
        await tool.execute("create", schedule="daily 09:00", task="Draft replies")
        assert stored[2]["mail_intent_text"] == ""

    async def test_cron_tool_keeps_the_request_behind_a_confirmation(self, monkeypatch):
        """ "Every Friday email Ana the KPI report" → "Should I set it up?" →
        "yes, set it up": the stored intent keeps the request's words."""
        stored: list = []
        tool = self._cron_tool(monkeypatch, stored)
        tool._agent.session = types.SimpleNamespace(id="s1", messages=[
            {"role": "user", "content": "Every Friday email Ana the KPI report", "turn_input": True},
            {"role": "assistant", "content": "Should I set it up for Fridays at 9:00?"},
            {"role": "user", "content": "yes, set it up", "turn_input": True},
        ])
        with ma.bound(ma.human("yes, set it up")):
            res = await tool.execute("create", schedule="weekly fri 09:00",
                                     task="Every Friday email Ana the KPI report")
        assert res.success, res.error
        assert stored[0]["mail_intent_text"] == (
            "yes, set it up\nEvery Friday email Ana the KPI report")
        assert "can't write email" not in res.content
        with ma.bound(ma.automated("cron", ma.cron_job_text(stored[0]), "intent")):
            assert ma.check_mail_write("google_mail", "create_draft") is None

    async def test_cron_tool_says_when_the_job_cant_write_email(self, monkeypatch):
        stored: list = []
        tool = self._cron_tool(monkeypatch, stored)
        with ma.bound(ma.human("yes, set it up")):
            res = await tool.execute("create", schedule="weekly fri 09:00",
                                     task="Every Friday email Ana the KPI report")
        assert res.success, res.error
        assert stored[0]["mail_intent_text"] == "yes, set it up"
        assert "this job can't write email when it runs" in res.content
        with ma.bound(ma.human("summarize my inbox daily")):
            res = await tool.execute("create", schedule="daily 09:00", task="Summarize my inbox")
        assert "can't write email" not in res.content

    async def test_legacy_warning(self, monkeypatch):
        from captain_claw import cron_dispatch

        warned: list = []
        monkeypatch.setattr(cron_dispatch, "_LEGACY_MAIL_WARNED", False)
        monkeypatch.setattr(cron_dispatch, "log", types.SimpleNamespace(
            warning=lambda msg, **kw: warned.append((msg, kw)),
            debug=lambda *a, **kw: None, info=lambda *a, **kw: None))

        def _job(i, payload, kind="prompt"):
            return types.SimpleNamespace(id=i, kind=kind, payload=payload)

        async def _list(limit=200, active_only=False):
            return [_job("a", {"text": "draft replies to unanswered mail"}),
                    _job("b", {"text": "summarize my inbox"}),
                    _job("c", {"text": "draft replies to unanswered mail", "author": "agent"}),
                    _job("d", {"path": "x.sh"}, kind="script")]

        ctx = types.SimpleNamespace(agent=types.SimpleNamespace(
            session_manager=types.SimpleNamespace(list_cron_jobs=_list)))
        assert await cron_dispatch.warn_legacy_mail_cron_jobs(ctx) == 1
        assert await cron_dispatch.warn_legacy_mail_cron_jobs(ctx) == 0
        assert warned == [("legacy cron job may write email",
                           {"job_id": "a", "text": "draft replies to unanswered mail"})]

    def test_orchestrate_cron_is_bound_deny(self):
        import inspect

        from captain_claw import cron_dispatch

        src = inspect.getsource(cron_dispatch.execute_cron_job)
        assert 'mail_authority.automated("cron", "", "deny")' in src


# ── sister / BotPort / Telegram (source pins: their callers need a full runtime) ──


def test_sister_botport_telegram_bind():
    import inspect

    from captain_claw import botport_client, sister_session
    from captain_claw.web import telegram

    assert 'mail_authority.automated("sister", "", "deny")' in inspect.getsource(sister_session)
    assert inspect.getsource(botport_client).count('mail_authority.automated("botport", "", "deny")') == 2
    assert "mail_authority.bound(mail_authority.interactive(text))" in inspect.getsource(telegram)


# ── peer tools forward the narrower intent ───────────────────────────


@pytest.fixture
def _fd_peer(monkeypatch, _config):
    from captain_claw.tools import flight_deck as fd_tool

    _config.web.auth_token = "my-web-auth"
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("FD_DATA_DIR", raising=False)
    monkeypatch.setenv("FD_URL", FD)
    monkeypatch.delenv("FD_INTERNAL_URL", raising=False)
    monkeypatch.delenv("CLAW_GOOGLE_OAUTH__FLIGHT_DECK_URL", raising=False)
    fd_tool._UNPINNED_LOGGED.clear()
    bodies: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/fd/fleet":
            return httpx.Response(200, json=[{"name": "peer", "kind": "process",
                                              "host": "localhost", "port": 24300,
                                              "status": "running"}])
        if path == "/fd/consult-peer":
            bodies.setdefault(path, []).append(json.loads(request.content))
            line = json.dumps({"done": True, "ok": True, "response": "hi"}) + "\n"
            return httpx.Response(200, content=line.encode())
        if path == "/fd/delegate-peer":
            bodies.setdefault(path, []).append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **kw: _RealAsyncClient(transport=httpx.MockTransport(_handler),
                                                          timeout=5.0))
    return bodies


@pytest.mark.parametrize("question,human,expect", [
    ("Draft a reply to Ana re Q3", "draft a reply to Ana", "Draft a reply to Ana re Q3"),
    ("Check her inbox", "draft a reply to Ana", ""),
    ("Draft a reply to Ana re Q3", None, ""),
])
async def test_peer_tools_forward_narrower_intent(_fd_peer, question, human, expect):
    from captain_claw.tools.consult_peer import ConsultPeerTool
    from captain_claw.tools.flight_deck import FlightDeckTool

    agent = types.SimpleNamespace(_turn_user_text="draft a reply to Ana")
    session = types.SimpleNamespace(metadata={
        "fd_url": FD, "peer_agents": [{"name": "peer", "host": "localhost", "port": 24300}]})
    auth = ma.human(human) if human is not None else None
    with ma.bound(auth):
        r1 = await ConsultPeerTool().execute("peer", question, _agent=agent, _session=session)
        r2 = await FlightDeckTool()._consult(FD, "peer", question, _agent=agent)
        r3 = await FlightDeckTool()._delegate(FD, "peer", question, _agent=agent)
    assert r1.success, r1.error
    assert r2.success, r2.error
    assert r3.success, r3.error
    consults = _fd_peer["/fd/consult-peer"]
    assert [b["mail_intent_text"] for b in consults] == [expect, expect]
    assert _fd_peer["/fd/delegate-peer"][0]["mail_intent_text"] == expect


# ── turn-loop nudges ─────────────────────────────────────────────────


class _Loop:
    """Just enough agent for the nudge / stall helpers."""

    def __init__(self, turn_user_text: str = ""):
        self._turn_user_text = turn_user_text
        self.session = types.SimpleNamespace(messages=[])

    def _turn_has_mail_write(self, idx):
        return False


def _nudge(loop, text, tools=frozenset({"google_mail"})):
    from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

    return AgentOrchestrationMixin._should_nudge_mail(loop, text, set(tools), 0)


_TWO_DRAFTS = ("**To:** ana@x.co\n**Subject:** Q3\nHi Ana\n\n"
               "**To:** bob@x.co\n**Subject:** Q3\nHi Bob")
_DIGEST = ("1. From: Ana <ana@x.co>\nSubject: Q3 numbers\n\n"
           "2. From: Bob <bob@x.co>\nSubject: Lunch\n")
_SUBJECTS = "Subject: Q3 numbers — Ana\nSubject: Lunch — Bob\n"


class TestToolAvoidanceNudge:
    def test_human_who_did_not_ask_gets_no_nudge(self):
        with ma.bound(ma.human("what's new in my inbox?")):
            assert _nudge(_Loop(), _SUBJECTS) is False

    def test_human_who_asked_gets_the_nudge(self):
        with ma.bound(ma.human("draft emails to Ana and Bob")):
            assert _nudge(_Loop(), _TWO_DRAFTS) is True
            assert _nudge(_Loop(), _TWO_DRAFTS, tools={"read"}) is False

    def test_a_yes_to_the_agents_offer_gets_the_nudge(self):
        loop = _Loop()
        loop.session.messages = [
            {"role": "user", "content": "anything from Ana or Bob?", "turn_input": True},
            {"role": "assistant", "content": "Both are waiting for a reply. Want me to draft them?"},
            {"role": "user", "content": "yes do it", "turn_input": True},
        ]
        with ma.bound(ma.human("yes do it")):
            assert _nudge(loop, _TWO_DRAFTS) is True
        loop.session.messages[1]["content"] = "Both wrote about Q3. Anything else?"
        with ma.bound(ma.human("yes do it")):
            assert _nudge(loop, _TWO_DRAFTS) is False

    def test_received_mail_digest_is_not_a_dodge(self):
        with ma.bound(ma.human("draft emails to Ana and Bob")):
            assert _nudge(_Loop(), _DIGEST) is False

    def test_never_in_automated_turns(self):
        for auth in (ma.automated("cron", "email me the summary every morning"),
                     ma.automated("autonomy_tool", "", "allow"),
                     ma.automated("cron", "draft a reply to Ana"),
                     ma.automated("cron", "", "deny")):
            with ma.bound(auth):
                assert _nudge(_Loop("draft emails to Ana and Bob"), _SUBJECTS) is False
                assert _nudge(_Loop("draft emails to Ana and Bob"), _TWO_DRAFTS) is False


class TestStallNag:
    def _instr(self, loop, text):
        from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

        return AgentOrchestrationMixin._stall_retry_instruction(loop, text, True)

    def test_no_mail_variant_when_nobody_asked(self):
        from captain_claw.agent_orchestration_mixin import (
            _STALL_MAIL_NEUTRAL_INSTRUCTION,
            _STALL_NO_MAIL_INSTRUCTION,
        )

        # human turn (U3): neutral — never "nobody asked"
        with ma.bound(ma.human("At the moment you have a hard gate")):
            instr, force = self._instr(_Loop(), "I'll create the draft now…")
        assert instr == _STALL_MAIL_NEUTRAL_INSTRUCTION and force is False
        assert "nobody asked" not in instr
        assert "If the user asked for this email in this conversation" in instr
        # automated turn: the strict variant
        with ma.bound(ma.automated("cron", "", "deny")):
            instr, force = self._instr(_Loop(), "I'll create the draft now…")
        assert instr == _STALL_NO_MAIL_INSTRUCTION and force is False

    @pytest.mark.parametrize("yes", ["yes do it", "draft it", "odgovori mu", "da, napravi",
                                     "odgovori joj da može", "go ahead and draft the reply"])
    def test_a_yes_to_the_agents_offer_forces_the_tool(self, yes):
        loop = _Loop()
        loop.session.messages = [
            {"role": "user", "content": "what did Ana write?", "turn_input": True},
            {"role": "assistant", "content": "Ana asks about the contract. Shall I draft a reply?"},
            {"role": "user", "content": yes, "turn_input": True},
        ]
        with ma.bound(ma.human(yes)):
            instr, force = self._instr(loop, "I'll draft the reply to Ana now.")
        assert "Call the appropriate tool now" in instr and force is True

    def test_the_request_two_messages_back_counts(self):
        loop = _Loop()
        loop.session.messages = [
            {"role": "user", "content": "draft a reply to Ana about the contract", "turn_input": True},
            {"role": "assistant", "content": "Formal or casual?"},
            {"role": "user", "content": "casual", "turn_input": True},
        ]
        with ma.bound(ma.human("casual")):
            instr, force = self._instr(loop, "I'll draft the reply to Ana now.")
        assert "Call the appropriate tool now" in instr and force is True

    @pytest.mark.parametrize("human,stall", [
        ("Fix the API so the server responds with 404", "I'll update the handler so it responds with a 404."),
        ("Create a Google Doc with a project proposal draft", "Let me draft the proposal in a Google Doc."),
        ("Summarise report.pdf into summary.md", "Let me read it and reply with the summary."),
        ("Configure the mailer settings in config.yaml", "I'll update the mailer config now."),
        ("what's new in my inbox?", "Let me check your email now."),
        ("summarize the thread with Bob", "Let me open the email thread now."),
    ])
    def test_ordinary_stalls_still_force_a_tool(self, human, stall):
        with ma.bound(ma.human(human)):
            instr, force = self._instr(_Loop(), stall)
        assert "Call the appropriate tool now" in instr and force is True

    @pytest.mark.parametrize("stall", ["Let me draft chapter 3 now.",
                                       "I'll write the draft of section 2 to the file.",
                                       "Let me fix how the endpoint responds."])
    def test_worker_stalls_that_arent_email_still_force_a_tool(self, monkeypatch, stall):
        monkeypatch.setenv("CLAW_VATRA_WORKER", "1")
        instr, force = self._instr(_Loop(), stall)
        assert "Call the appropriate tool now" in instr and force is True

    def test_old_variant_when_asked_or_not_mail(self):
        with ma.bound(ma.human("draft a reply to Nataša")):
            instr, force = self._instr(_Loop(), "I'll create the draft now…")
        assert "Call the appropriate tool now" in instr and force is True
        with ma.bound(ma.automated("cron", "draft a reply to Ana")):
            instr, force = self._instr(_Loop(), "I'll create the draft now…")
        assert "Call the appropriate tool now" in instr and force is True
        with ma.bound(ma.human("make me a chart")):
            instr, force = self._instr(_Loop(), "I'll build the chart now…")
        assert "Call the appropriate tool now" in instr and force is True

    def test_stall_loop_uses_the_helper_and_force_flag(self):
        import inspect

        from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin

        src = inspect.getsource(AgentOrchestrationMixin)
        assert "self._stall_retry_instruction(" in src
        assert "if _force_tool:" in src
        assert "self._should_nudge_mail(" in src


# ── prompt pins ──────────────────────────────────────────────────────

_INSTR = Path(ma.__file__).resolve().parent / "instructions"


def test_prompt_pins():
    import captain_claw.agent_context_mixin as acm
    from captain_claw.tools import google_mail as gm
    from captain_claw.tools.cron_tool import CronTool

    section = (_INSTR / "section_google.md").read_text(encoding="utf-8")
    assert "ONLY when the user asked for that email" in section
    assert "`create_draft` by default" not in section
    micro = (_INSTR / "micro_section_google.md").read_text(encoding="utf-8")
    assert "ONLY when the user asked" in micro and "create_draft by default" not in micro

    assert "only when the user asked" in acm._TOOL_PROMPT_DESCRIPTIONS["google_mail"]
    assert "only when the user asked" in acm._TOOL_PROMPT_DESCRIPTIONS_MICRO["google_mail"]

    d = gm.GoogleMailTool.description
    assert "WRITE ONLY WHEN ASKED" in d
    assert "create_draft is the DEFAULT" not in d and "MANDATORY: when the user asks" not in d
    assert "ONLY if the user asked you to reply" in gm._FOLLOWUP_HINT
    assert "Reply to one" not in gm._FOLLOWUP_HINT_NO_REPLY

    for name in ("system_prompt.md", "micro_system_prompt.md"):
        text = (_INSTR / name).read_text(encoding="utf-8")
        assert '"[Automated turn"' in text, name
        assert "never count as the user asking you to write an email" in text, name
    assert "Email is never yours to start" in (_INSTR / "system_prompt.md").read_text(encoding="utf-8")

    assert "scheduled runs may only write email when the task itself says to" in (
        CronTool.parameters["properties"]["task"]["description"])

    data = json.loads((_INSTR / "archetypes.json").read_text(encoding="utf-8"))
    by_id = {a["id"]: a for a in data["archetypes"]}
    assert "never draft or send a reply on your own" in by_id["inbox-manager"]["fleet_instructions"]
    assert "drafts replies only when asked" in by_id["inbox-manager"]["description"]
    assert "never start an email on your own" in by_id["comms-outbound"]["fleet_instructions"]


def test_turn_user_text_is_set_by_complete_and_stream():
    import inspect

    from captain_claw.agent import Agent

    assert Agent._turn_user_text == ""
    for fn in (Agent.complete, Agent.stream):
        assert 'self._turn_user_text = str(user_input or "")' in inspect.getsource(fn)
