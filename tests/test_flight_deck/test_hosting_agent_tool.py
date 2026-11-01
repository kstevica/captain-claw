"""The `hosting` and `code` agent tools, through Flight Deck's real guard.

Publishing from a browser chat was refused with "agent routes require a bearer
token, loopback, or X-Agent-Secret". The tools preferred the session's ``fd_url``
(the browser's origin) over the deck FD pinned at spawn, so the call left through
the public proxy and came back looking remote; and they sent no X-Agent-Secret,
which FD_LOCKDOWN refuses even from loopback. A publish that got through then
built its link from the request's host, which for an agent is localhost.

No network: `_Network` routes httpx into the real app, and where a request lands
decides who FD thinks is calling.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import httpx
import pytest

from captain_claw import config as config_mod
from captain_claw.config import Config
from captain_claw.flight_deck import agent_secret
from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import get_db, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.tools.basna import BasnaTool
from captain_claw.tools.code_session import CodeSessionTool
from captain_claw.tools.hosting import HostingTool
from captain_claw.vfs import user_root_of

DECK = "http://localhost:25080"
PUBLIC = "https://deck.example.test"
EVIL = "http://evil.test"
OWNER = "u-owner"
TOKEN = "my-web-auth"
PUBLISH = {"action": "publish", "name": "demo", "kind": "static",
           "project": "crew", "subdir": "web/task/files"}


class _Network(httpx.AsyncBaseTransport):
    """Local names reach FD from loopback; the public name reaches it through the
    proxy, as a remote client; anything else is not this deck at all."""

    def __init__(self, app) -> None:
        self.requests: list[httpx.Request] = []
        self._loopback = httpx.ASGITransport(app=app, client=("127.0.0.1", 40001))
        self._proxied = httpx.ASGITransport(app=app, client=("203.0.113.9", 40001))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "localhost":
            return await self._loopback.handle_async_request(request)
        if request.url.host == "deck.example.test":
            return await self._proxied.handle_async_request(request)
        return httpx.Response(404)


@pytest.fixture
async def network(tmp_path: Path, monkeypatch):
    from captain_claw.flight_deck import auth as fd_auth

    prev_db = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)

    monkeypatch.setattr(fd_server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.setenv("FD_URL", DECK)
    monkeypatch.setenv("FD_PUBLIC_URL", PUBLIC)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for name in ("CLAW_VFS_ROOT", "FD_INTERNAL_URL", "CLAW_GOOGLE_OAUTH__FLIGHT_DECK_URL",
                 "FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET", "FD_ALLOWED_HOSTS"):
        monkeypatch.delenv(name, raising=False)
    agent_secret.reset_cache_for_tests()

    cfg = Config()
    cfg.web.auth_token = TOKEN
    monkeypatch.setattr(config_mod, "_config", cfg)
    monkeypatch.setattr(fd_server, "_resolve_agent_owner_by_auth",
                        lambda token: OWNER if token == TOKEN else None)

    site = user_root_of(OWNER) / "crew" / "web" / "task" / "files"
    site.mkdir(parents=True)
    (site / "index.html").write_text("<h1>crew page</h1>")

    net = _Network(fd_server.app)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=net, timeout=5.0))
    try:
        yield net
    finally:
        agent_secret.reset_cache_for_tests()
        set_auth_db(prev_db)
        await db.close()


def _session(url: str = PUBLIC):
    return types.SimpleNamespace(metadata={"fd_url": url})


class TestBrowserChat:
    """The session carries the browser's origin: the deck's public address."""

    async def test_publish_stays_on_the_pinned_deck_and_links_publicly(self, network):
        res = await HostingTool().execute(**PUBLISH, _session=_session())
        assert res.success, res.error
        assert f"{PUBLIC}/vfs/demo/" in res.content
        assert "localhost" not in res.content
        assert {r.url.host for r in network.requests} == {"localhost"}

    async def test_the_published_page_is_served(self, network):
        assert (await HostingTool().execute(**PUBLISH, _session=_session())).success
        async with httpx.AsyncClient() as client:
            page = await client.get(f"{DECK}/vfs/demo/")
        assert page.status_code == 200
        assert "crew page" in page.text

    async def test_code_tool_gets_through_too(self, network):
        res = await CodeSessionTool().execute(action="list", _session=_session())
        assert res.success, res.error
        assert {r.url.host for r in network.requests} == {"localhost"}


class TestLockdown:
    """FD_LOCKDOWN refuses bare loopback: the deck secret has to come along."""

    async def test_publish(self, network, monkeypatch):
        monkeypatch.setenv("FD_LOCKDOWN", "1")
        res = await HostingTool().execute(**PUBLISH)
        assert res.success, res.error
        assert network.requests[0].headers["X-Agent-Secret"] == agent_secret.get_or_create_agent_secret()

    async def test_code_tool(self, network, monkeypatch):
        monkeypatch.setenv("FD_LOCKDOWN", "1")
        res = await CodeSessionTool().execute(action="list")
        assert res.success, res.error

    async def test_basna_file_fetch_carries_the_secret(self, network, monkeypatch):
        monkeypatch.setenv("FD_LOCKDOWN", "1")
        await BasnaTool()._get_file(DECK, session_id="s", name="report.md")
        (req,) = network.requests
        assert req.headers["X-Agent-Secret"] == agent_secret.get_or_create_agent_secret()


class TestLinks:
    async def test_every_action_gives_the_public_link(self, network):
        tool = HostingTool()
        assert (await tool.execute(**PUBLISH)).success
        for action in ("list", "status"):
            res = await tool.execute(action=action, name="demo")
            assert res.success, res.error
            assert f"{PUBLIC}/vfs/demo/" in res.content, action

    async def test_a_browser_origin_stands_in_for_a_missing_public_url(self, network, monkeypatch):
        monkeypatch.delenv("FD_PUBLIC_URL")
        await get_db().set_public_link_origin(OWNER, PUBLIC)
        res = await HostingTool().execute(**PUBLISH)
        assert res.success, res.error
        assert f"{PUBLIC}/vfs/demo/" in res.content
        assert "no public address" not in res.content

    async def test_no_public_address_is_said_plainly(self, network, monkeypatch):
        monkeypatch.delenv("FD_PUBLIC_URL")
        res = await HostingTool().execute(**PUBLISH)
        assert res.success, res.error
        assert f"{DECK}/vfs/demo/" in res.content
        assert "no public address" in res.content


@pytest.mark.parametrize("tool", [HostingTool, CodeSessionTool])
class TestCredentialsOnlyGoToThePinnedDeck:
    """A session URL is whatever a websocket client sent."""

    async def test_a_foreign_session_url_is_ignored(self, network, tool):
        await tool().execute(action="list", _session=_session(EVIL))
        assert {r.url.host for r in network.requests} == {"localhost"}

    async def test_nothing_pinned_means_no_credentials(self, network, monkeypatch, tool):
        monkeypatch.delenv("FD_URL")
        await tool().execute(action="list", _session=_session(EVIL))
        (req,) = network.requests
        assert req.url.host == "evil.test"
        assert "X-Agent-Secret" not in req.headers
        assert "X-Agent-Auth" not in req.headers
        assert json.loads(req.content)["web_auth"] == ""
