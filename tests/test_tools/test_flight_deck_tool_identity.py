"""The agent-side ``flight_deck`` tool proves WHICH agent is calling Flight Deck.

FD resolves an unauthenticated spawn's owner ONLY from the calling agent's
``X-Agent-Auth`` (its own web_auth), and FD_LOCKDOWN additionally requires the
per-deck ``X-Agent-Secret``. So every FD call the tool makes carries both, from
the same sources google_oauth_manager / tools.basna use; an auth refusal on
spawn is reported plainly instead of being retried against the Docker endpoint.

Those two headers (and basna's body ``web_auth``) only ever go to the FD URL
pinned in the agent's environment / config — never to a session ``fd_url``,
which any websocket client (public-run sockets included) can set.

No FD, no network: httpx is routed to a MockTransport that records requests.
"""

from __future__ import annotations

import json
import types

import httpx
import pytest

from captain_claw import config as config_mod
from captain_claw.config import Config
from captain_claw.flight_deck import agent_secret
from captain_claw.tools import flight_deck as fd_tool
from captain_claw.tools.basna import BasnaTool
from captain_claw.tools.flight_deck import FlightDeckTool

FD = "http://fd.test"
EVIL = "http://evil.test"


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    c = Config()
    c.web.auth_token = "my-web-auth"
    monkeypatch.setattr(config_mod, "_config", c)
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("FD_DATA_DIR", raising=False)
    # The deck FD pinned at spawn.
    monkeypatch.setenv("FD_URL", FD)
    monkeypatch.delenv("FD_INTERNAL_URL", raising=False)
    monkeypatch.delenv("CLAW_GOOGLE_OAUTH__FLIGHT_DECK_URL", raising=False)
    fd_tool._UNPINNED_LOGGED.clear()
    # Never touch the real ~/.captain-claw-fd secret file.
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    yield c
    agent_secret.reset_cache_for_tests()


class TestHeaders:
    def test_sends_own_web_auth_and_the_deck_secret_file(self, cfg):
        h = fd_tool._fd_agent_headers(FD)
        assert h["X-Agent-Auth"] == "my-web-auth"
        assert h["X-Agent-Secret"] == agent_secret.get_or_create_agent_secret()

    def test_env_secret_beats_the_file(self, cfg, monkeypatch):
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "from-env")
        assert fd_tool._fd_agent_headers(FD)["X-Agent-Secret"] == "from-env"

    def test_configured_flight_deck_secret_beats_everything(self, cfg, monkeypatch):
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "from-env")
        cfg.google_oauth.flight_deck_secret = "from-config"
        assert fd_tool._fd_agent_headers(FD)["X-Agent-Secret"] == "from-config"

    def test_no_web_auth_no_identity_header(self, cfg):
        cfg.web.auth_token = ""
        assert "X-Agent-Auth" not in fd_tool._fd_agent_headers(FD)


class TestOnlyThePinnedFdUrlGetsTheIdentity:
    def test_unpinned_url_gets_no_headers_and_is_logged_once(self, cfg, monkeypatch):
        seen: list[dict] = []
        monkeypatch.setattr(fd_tool.log, "warning", lambda *a, **k: seen.append(k))
        assert fd_tool._fd_agent_headers(EVIL) == {}
        assert fd_tool._fd_agent_headers(EVIL + "/") == {}
        assert [k["fd_url"] for k in seen] == [EVIL]

    def test_no_pinned_url_at_all_means_no_headers(self, cfg, monkeypatch):
        monkeypatch.delenv("FD_URL")
        assert fd_tool._fd_agent_headers(FD) == {}

    @pytest.mark.parametrize("pin", ["env_internal", "config"])
    def test_other_pinned_sources_count(self, cfg, monkeypatch, pin):
        monkeypatch.delenv("FD_URL")
        if pin == "env_internal":
            monkeypatch.setenv("FD_INTERNAL_URL", FD + "/")
        else:
            cfg.google_oauth.flight_deck_url = FD
        assert fd_tool._fd_agent_headers(FD)["X-Agent-Auth"] == "my-web-auth"
        # Trailing slash / host case don't matter; a different port does.
        assert fd_tool._fd_agent_headers("HTTP://FD.test/")["X-Agent-Auth"] == "my-web-auth"
        assert fd_tool._fd_agent_headers("http://fd.test:9999") == {}

    def test_pinned_url_outranks_a_websocket_supplied_one(self, cfg):
        session = types.SimpleNamespace(metadata={"fd_url": EVIL})
        agent = types.SimpleNamespace(_fd_url=EVIL)
        assert FlightDeckTool()._get_fd_url(_session=session, _agent=agent) == FD
        assert BasnaTool()._get_fd_url(_session=session, _agent=agent) == FD
        assert FlightDeckTool()._get_fd_url() == FD

    def test_session_alias_of_the_pinned_deck_wins(self, cfg, monkeypatch):
        # A Docker agent: env FD_URL says localhost (unreachable from inside the
        # container), the session carries the reachable host.docker.internal.
        monkeypatch.setenv("FD_URL", "http://localhost:25080")
        session = types.SimpleNamespace(metadata={"fd_url": "http://host.docker.internal:25080"})
        assert FlightDeckTool()._get_fd_url(_session=session) == "http://host.docker.internal:25080"
        assert BasnaTool()._get_fd_url(_session=session) == "http://host.docker.internal:25080"
        other_port = types.SimpleNamespace(metadata={"fd_url": "http://host.docker.internal:25081"})
        assert FlightDeckTool()._get_fd_url(_session=other_port) == "http://localhost:25080"

    @pytest.mark.parametrize("pinned,other,same", [
        ("http://localhost:25080", "http://host.docker.internal:25080", True),
        ("http://127.0.0.1:25080/", "http://LOCALHOST:25080", True),
        ("http://localhost:25080", "http://host.docker.internal:25081", False),
        ("http://localhost:25080", "https://localhost:25080", False),
        ("http://localhost:25080", "http://evil.test:25080", False),
        ("http://fd.test", "http://host.docker.internal", False),
    ])
    def test_local_host_aliases_are_the_same_pinned_deck(self, cfg, monkeypatch, pinned, other, same):
        from captain_claw.tools.flight_deck import _is_pinned_fd_url

        monkeypatch.setenv("FD_URL", pinned)
        assert _is_pinned_fd_url(other) is same

    def test_session_url_is_still_used_when_nothing_is_pinned(self, cfg, monkeypatch):
        monkeypatch.delenv("FD_URL")
        session = types.SimpleNamespace(metadata={"fd_url": EVIL})
        assert FlightDeckTool()._get_fd_url(_session=session) == EVIL
        assert BasnaTool()._get_fd_url(_session=session) == EVIL


class _FD:
    """Records requests; answers like Flight Deck would."""

    def __init__(self, spawn_status: int = 200):
        self.requests: list[httpx.Request] = []
        self.spawn_status = spawn_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/fd/fleet":
            return httpx.Response(200, json=[
                {"name": "peer", "kind": "process", "host": "localhost",
                 "port": 24300, "status": "running"}])
        if path == "/fd/consult-peer":
            line = json.dumps({"done": True, "ok": True, "response": "hi"}) + "\n"
            return httpx.Response(200, content=line.encode())
        if path == "/fd/delegate-peer":
            return httpx.Response(200, json={"ok": True})
        if path in ("/fd/spawn-process", "/fd/spawn"):
            if self.spawn_status != 200:
                return httpx.Response(self.spawn_status, json={"detail": "Not authenticated"})
            return httpx.Response(200, json={"ok": True, "slug": "kid"})
        if path.startswith("/fd/basna/agent/"):
            return httpx.Response(200, json={"sessions": []})
        return httpx.Response(404)


@pytest.fixture
def fd(monkeypatch):
    server = _FD()
    real_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        return real_client(transport=httpx.MockTransport(server), timeout=5.0)

    monkeypatch.setattr(httpx, "AsyncClient", _client)
    return server


class TestEveryCallCarriesTheAgentsIdentity:
    async def test_list_consult_delegate_and_spawn(self, cfg, fd):
        tool = FlightDeckTool()
        assert (await tool._list_agents(FD)).success
        assert (await tool._consult(FD, "peer", "q?")).success
        assert (await tool._delegate(FD, "peer", "do it")).success
        assert (await tool._spawn_agent(FD, "kid")).success
        paths = [r.url.path for r in fd.requests]
        assert {"/fd/fleet", "/fd/consult-peer", "/fd/delegate-peer", "/fd/spawn-process"} <= set(paths)
        secret = agent_secret.get_or_create_agent_secret()
        for r in fd.requests:
            assert r.headers.get("X-Agent-Auth") == "my-web-auth", r.url.path
            assert r.headers.get("X-Agent-Secret") == secret, r.url.path


class TestSpawnBaseline:
    """The child inherits this agent's working model as a baseline — its key
    and endpoint only together with its provider."""

    GW = "https://gw.example/v1"

    def _agent(self):
        return types.SimpleNamespace(provider=types.SimpleNamespace(
            provider="openai", model="swift", api_key="gw-key", base_url=self.GW))

    async def _body(self, fd, overrides: dict | None = None) -> dict:
        res = await FlightDeckTool()._spawn_agent(
            FD, "kid", json.dumps(overrides) if overrides is not None else "", _agent=self._agent())
        assert res.success, res.error
        return json.loads(fd.requests[-1].content)

    async def test_no_overrides_inherits_the_whole_model(self, cfg, fd):
        body = await self._body(fd)
        assert (body["provider"], body["model"], body["provider_api_key"], body["base_url"]) == (
            "openai", "swift", "gw-key", self.GW)

    async def test_another_model_on_the_same_provider_stays_on_our_endpoint(self, cfg, fd):
        body = await self._body(fd, {"model": "swift-mini"})
        assert (body["model"], body["provider_api_key"], body["base_url"]) == ("swift-mini", "gw-key", self.GW)

    async def test_another_provider_gets_neither_our_key_nor_our_endpoint(self, cfg, fd):
        body = await self._body(fd, {"provider": "anthropic", "model": "claude"})
        assert (body["provider"], body["provider_api_key"], body["base_url"]) == ("anthropic", "", "")

    @pytest.mark.parametrize("base_url", ["http://third-party.example/v1", ""])
    async def test_another_endpoint_on_our_provider_does_not_get_our_key(self, cfg, fd, base_url):
        body = await self._body(fd, {"model": "local", "base_url": base_url})
        assert (body["provider"], body["provider_api_key"], body["base_url"]) == ("openai", "", base_url)

    async def test_our_own_endpoint_written_differently_is_still_ours(self, cfg, fd):
        body = await self._body(fd, {"base_url": self.GW.upper() + "/"})
        assert body["provider_api_key"] == "gw-key"

    @pytest.mark.parametrize("alias", ["chatgpt", "OpenAI"])
    async def test_an_alias_of_our_provider_is_not_another_provider(self, cfg, fd, alias):
        body = await self._body(fd, {"provider": alias, "model": "swift-mini"})
        assert (body["provider"], body["provider_api_key"], body["base_url"]) == ("openai", "gw-key", self.GW)

    async def test_our_own_key_override_stays_on_our_endpoint(self, cfg, fd):
        body = await self._body(fd, {"api_key": "another-gw-key"})
        assert (body["provider_api_key"], body["base_url"]) == ("another-gw-key", self.GW)

    async def _spawn(self, fd, parent, overrides: dict):
        agent = types.SimpleNamespace(provider=types.SimpleNamespace(**parent))
        res = await FlightDeckTool()._spawn_agent(FD, "kid", json.dumps(overrides), _agent=agent)
        assert res.success, res.error
        return json.loads(fd.requests[-1].content), res.content

    @pytest.mark.parametrize("parent_url, override_url", [
        ("", "https://api.openai.com/v1"),                      # our own endpoint, spelled out
        ("https://api.openai.com/v1/", ""),
        ("http://localhost:1234/v1", "http://127.0.0.1:1234/v1"),
    ])
    async def test_our_endpoint_is_recognised_the_way_flight_deck_recognises_it(
            self, cfg, fd, parent_url, override_url):
        parent = dict(provider="openai", model="m", api_key="our-key", base_url=parent_url)
        body, _text = await self._spawn(fd, parent, {"model": "m2", "base_url": override_url})
        assert (body["provider_api_key"], body["base_url"]) == ("our-key", parent_url)

    async def test_a_chatgpt_sign_in_model_does_not_go_to_our_gateway(self, cfg, fd):
        """GPT-5 / Codex models sign in through ChatGPT: on our gateway the
        child would post the ChatGPT token there."""
        body = await self._body(fd, {"model": "gpt-5.2"})
        assert (body["model"], body["provider_api_key"], body["base_url"]) == ("gpt-5.2", "", "")

    async def test_a_keyed_model_does_not_go_to_our_chatgpt_endpoint(self, cfg, fd):
        parent = dict(provider="openai", model="gpt-5.2", api_key="",
                      base_url="https://chatgpt.com/backend-api/codex/responses")
        body, _text = await self._spawn(fd, parent, {"model": "gpt-4.1"})
        assert (body["model"], body["base_url"]) == ("gpt-4.1", "")
        body, _text = await self._spawn(fd, parent, {"model": "gpt-5.2-codex"})   # still ChatGPT: stays
        assert body["base_url"] == parent["base_url"]

    async def test_the_agent_is_told_when_its_key_was_not_passed_along(self, cfg, fd):
        agent = self._agent()
        withheld = await FlightDeckTool()._spawn_agent(
            FD, "kid", json.dumps({"model": "local", "base_url": "http://localhost:1234/v1"}), _agent=agent)
        assert "your own API key was not passed along" in withheld.content
        for overrides in ({}, {"model": "swift-mini"},
                          {"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"}):
            res = await FlightDeckTool()._spawn_agent(FD, "kid", json.dumps(overrides), _agent=agent)
            assert "not passed along" not in res.content

    async def test_another_provider_with_its_own_key_and_endpoint_keeps_them(self, cfg, fd):
        body = await self._body(fd, {"provider": "anthropic", "model": "claude", "api_key": "sk-ant",
                                     "base_url": "https://proxy.example"})
        assert (body["provider_api_key"], body["base_url"]) == ("sk-ant", "https://proxy.example")


class TestSpawnRefusal:
    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_refusal_is_reported_not_retried_on_docker(self, cfg, fd, status):
        fd.spawn_status = status
        res = await FlightDeckTool()._spawn_agent(FD, "kid")
        assert not res.success
        assert f"HTTP {status}" in res.error and "Not authenticated" in res.error
        assert [r.url.path for r in fd.requests] == ["/fd/spawn-process"]

    async def test_other_failures_still_fall_back_to_docker(self, cfg, fd):
        fd.spawn_status = 500
        res = await FlightDeckTool()._spawn_agent(FD, "kid")
        assert not res.success
        assert [r.url.path for r in fd.requests] == ["/fd/spawn-process", "/fd/spawn"]


class TestWebsocketSuppliedUrlNeverGetsTheIdentity:
    """The review's scenario: a websocket client (a public-run socket too) sends
    ``peer_agents`` with its own ``fd_url``; the next fleet / spawn / basna call
    used to carry X-Agent-Secret + X-Agent-Auth (+ basna's body web_auth) there."""

    def _session(self):
        return types.SimpleNamespace(metadata={"fd_url": EVIL})

    async def test_flight_deck_tool_calls_the_pinned_deck_instead(self, cfg, fd):
        res = await FlightDeckTool().execute(action="list_agents", _session=self._session())
        assert res.success, res.error
        assert {r.url.host for r in fd.requests} == {"fd.test"}
        assert all(r.headers.get("X-Agent-Auth") == "my-web-auth" for r in fd.requests)

    async def test_flight_deck_tool_sends_identity_to_the_pinned_deck(self, cfg, fd):
        res = await FlightDeckTool().execute(action="list_agents")
        assert res.success, res.error
        assert {r.url.host for r in fd.requests} == {"fd.test"}
        assert all(r.headers.get("X-Agent-Auth") == "my-web-auth" for r in fd.requests)

    async def test_flight_deck_tool_sends_nothing_to_an_unpinned_session_url(self, cfg, fd, monkeypatch):
        monkeypatch.delenv("FD_URL")
        tool = FlightDeckTool()
        assert (await tool.execute(action="list_agents", _session=self._session())).success
        await tool.execute(action="spawn_agent", agent_name="kid", _session=self._session())
        assert {r.url.host for r in fd.requests} == {"evil.test"}
        for r in fd.requests:
            assert "X-Agent-Auth" not in r.headers and "X-Agent-Secret" not in r.headers

    async def test_basna_calls_the_pinned_deck_with_its_identity(self, cfg, fd):
        res = await BasnaTool().execute(action="list")
        assert res.success, res.error
        (req,) = fd.requests
        assert req.url.host == "fd.test"
        assert req.headers.get("X-Agent-Secret") == agent_secret.get_or_create_agent_secret()
        assert json.loads(req.content)["web_auth"] == "my-web-auth"

    async def test_basna_sends_no_secret_or_web_auth_to_an_unpinned_url(self, cfg, fd, monkeypatch):
        monkeypatch.delenv("FD_URL")
        await BasnaTool().execute(action="list", _session=self._session())
        (req,) = fd.requests
        assert req.url.host == "evil.test"
        assert "X-Agent-Secret" not in req.headers
        assert json.loads(req.content)["web_auth"] == ""

    def test_basna_legacy_no_arg_secret_header_is_unchanged(self, cfg):
        """tools.vatra still calls it without a target."""
        from captain_claw.tools.basna import _agent_secret_headers

        assert _agent_secret_headers() == {
            "X-Agent-Secret": agent_secret.get_or_create_agent_secret()}
        assert _agent_secret_headers(EVIL) == {}
