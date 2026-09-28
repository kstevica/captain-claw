"""Browser guard for Flight Deck: Host allowlist, cross-site refusal, locked CORS,
and agent identity on the Codex / MCP agent endpoints.

The threat: on an auth-disabled deck (the Electron desktop build) every route is
open, and FD trusts loopback — and a web page the user visits runs on loopback
too. Before this change any site could (a) call the API cross-origin (CORS was
``*`` with credentials, which Starlette answers by mirroring any origin), (b)
open a WebSocket to the agent proxy (CORS never covers WebSockets), (c) do the
same same-origin via DNS rebinding, and (d) pull the ChatGPT/Codex token or
drive the MCP proxy from loopback with no identity at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.websockets import WebSocketDisconnect

from captain_claw.flight_deck import codex_oauth_routes, mcp_storage, origin_guard
from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)
REMOTE = ("203.0.113.9", 40001)
FD = "http://localhost:25080"
EVIL = "https://evil.example"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path: Path):
    for var in ("FD_ALLOWED_HOSTS", "FD_PUBLIC_URL", "FD_CORS_ORIGINS",
                "FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_MCP_PATH", str(tmp_path / "mcp_servers.json"))
    # Only this deck's registry; never the developer's real Docker daemon.
    monkeypatch.setattr(fd_server, "_load_process_registry", lambda: {
        "alpha": {"name": "Alpha", "web_auth": "tok-alpha", "owner": "",
                  "web_port": 24001, "pid": None},
    })

    def _no_docker():
        raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(fd_server, "get_docker", _no_docker)
    origin_guard._last_logged.clear()


def _client(client_addr=LOOPBACK, base_url=FD) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fd_server.app, client=client_addr),
        base_url=base_url)


# ── Host allowlist (DNS rebinding) ───────────────────────────────────


@pytest.mark.parametrize("host", [
    "localhost", "localhost:25080", "app.localhost:25080", "127.0.0.1:25080",
    "[::1]:25080", "192.168.1.20:25080", "10.0.0.7", "host.docker.internal:25080",
])
async def test_default_hosts_allowed(host):
    """Loopback, any IP literal (LAN access by IP — an IP can't be rebound) and
    Docker's host alias work with no configuration."""
    async with _client() as c:
        r = await c.get("/fd/auth/status", headers={"Host": host})
    assert r.status_code == 200, (host, r.text)


async def test_machine_hostname_allowed():
    """LAN access by this machine's own name keeps working unconfigured."""
    name = sorted(origin_guard._machine_hosts())[0]
    async with _client() as c:
        r = await c.get("/fd/auth/status", headers={"Host": f"{name}:25080"})
    assert r.status_code == 200


async def test_dns_rebinding_host_refused_and_logged(caplog):
    """A rebinding page is same-origin in the browser's eyes (no Origin at all);
    only its Host gives it away."""
    caplog.set_level(logging.WARNING, logger="flight_deck.origin_guard")
    async with _client() as c:
        r = await c.get("/fd/processes", headers={"Host": "evil.example:25080"})
    assert r.status_code == 403
    assert "FD_ALLOWED_HOSTS" in r.json()["detail"]
    assert "access-control-allow-origin" not in r.headers
    msg = " ".join(rec.getMessage() for rec in caplog.records)
    assert "evil.example:25080" in msg and "FD_ALLOWED_HOSTS" in msg


@pytest.mark.parametrize("host", ["evil.example@localhost", "local host", "localhost/x"])
async def test_malformed_host_refused(host):
    async with _client() as c:
        r = await c.get("/fd/auth/status", headers={"Host": host})
    assert r.status_code == 403


async def test_configured_hosts_allowed(monkeypatch):
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.example.com, .corp.example")
    monkeypatch.setenv("FD_PUBLIC_URL", "https://glasses.example.org")
    async with _client() as c:
        for host in ("fd.example.com", "fd.example.com:8765", "corp.example",
                     "deck.corp.example", "glasses.example.org"):
            r = await c.get("/fd/auth/status", headers={"Host": host})
            assert r.status_code == 200, host
        r = await c.get("/fd/auth/status", headers={"Host": "notcorp.example"})
        assert r.status_code == 403


async def test_host_check_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "*")
    async with _client() as c:
        r = await c.get("/fd/auth/status", headers={"Host": "anything.example"})
    assert r.status_code == 200


# ── cross-site requests + CORS ───────────────────────────────────────


async def test_cross_origin_fetch_refused_without_cors_headers():
    async with _client() as c:
        r = await c.get("/fd/processes", headers={"Origin": EVIL})
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers


async def test_same_origin_fetch_allowed(monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", False)
    async with _client() as c:
        r = await c.get("/fd/processes", headers={"Origin": FD, "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 200


async def test_lan_ip_same_origin_allowed(monkeypatch):
    """The SPA opened at http://<lan-ip>:25080 posts with that Origin."""
    async with _client(base_url="http://192.168.1.20:25080") as c:
        r = await c.get("/fd/auth/status", headers={"Origin": "http://192.168.1.20:25080"})
    assert r.status_code == 200


async def test_cross_site_spawn_never_reaches_handler(monkeypatch):
    """The RCE path: a page POSTs a JSON spawn (its preflight used to be answered
    with the page's own origin mirrored back). Refused before the handler runs,
    whatever the page's origin."""
    calls: list = []

    async def _record(config, request, user):
        calls.append(config.name)
        return fd_server.ProcessActionResult(ok=True, slug="x")

    monkeypatch.setattr(fd_server, "_spawn_process_locked", _record)
    spawn = {"name": "pwn", "tools": ["shell"]}
    async with _client() as c:
        r = await c.post("/fd/spawn-process", json=spawn,
                         headers={"Origin": EVIL, "Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403
        # Opaque-origin (sandboxed iframe / data: URL) pages are refused too.
        r = await c.post("/fd/spawn-process", json=spawn, headers={"Origin": "null"})
        assert r.status_code == 403
        # Another local page (e.g. an agent's own web UI port) is not FD either.
        r = await c.post("/fd/spawn-process", json=spawn,
                         headers={"Origin": "http://localhost:24001"})
        assert r.status_code == 403
        assert calls == []
        # Positive control: a non-browser caller still reaches the handler.
        r = await c.post("/fd/spawn-process", json=spawn)
    assert r.status_code == 200
    assert calls == ["pwn"]


async def test_cross_site_without_origin_refused_but_navigation_allowed():
    async with _client() as c:
        # <img>/<script>/no-cors GET from another site.
        r = await c.get("/fd/processes", headers={
            "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Dest": "image"})
        assert r.status_code == 403
        # A link / OAuth redirect landing on FD.
        r = await c.get("/fd/auth/status", headers={
            "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document"})
        assert r.status_code == 200


async def test_preflight_from_other_site_refused():
    async with _client() as c:
        r = await c.options("/fd/spawn-process", headers={
            "Origin": EVIL, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers


async def test_configured_cors_origin_gets_explicit_allow(monkeypatch):
    """FD_CORS_ORIGINS still serves a frontend on another origin — echoed
    exactly, never '*'."""
    app_origin = "https://app.example.com"
    monkeypatch.setenv("FD_CORS_ORIGINS", app_origin)
    async with _client() as c:
        r = await c.options("/fd/auth/status", headers={
            "Origin": app_origin, "Access-Control-Request-Method": "GET"})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == app_origin
        r = await c.get("/fd/auth/status", headers={"Origin": app_origin})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == app_origin
        r = await c.get("/fd/auth/status", headers={"Origin": EVIL})
        assert r.status_code == 403


async def test_public_url_origin_trusted_behind_host_rewriting_proxy(monkeypatch):
    """nginx's default proxy_pass rewrites Host to the upstream; the browser's
    Origin is the public URL. FD_PUBLIC_URL makes that FD's own origin."""
    monkeypatch.setenv("FD_PUBLIC_URL", "https://deck.example.org")
    async with _client(base_url="http://127.0.0.1:8765") as c:
        r = await c.get("/fd/auth/status", headers={"Origin": "https://deck.example.org"})
        assert r.status_code == 200
        # …and a proxy that forwards X-Forwarded-Host also works unconfigured.
        monkeypatch.delenv("FD_PUBLIC_URL")
        r = await c.get("/fd/auth/status", headers={
            "Origin": "https://other.example.org",
            "X-Forwarded-Host": "other.example.org"})
        assert r.status_code == 200


def test_vite_dev_origin_trusted():
    """vite.config.ts proxies /fd with changeOrigin: Host rewritten, Origin kept."""
    assert origin_guard.origin_trusted("http://localhost:5173", host_header="localhost:25080")
    assert not origin_guard.origin_trusted("http://localhost:24080", host_header="localhost:25080")


# ── WebSockets (CORS never applies) ──────────────────────────────────
# (TestClient ignores base_url for websockets and sends Host: testserver, so
# these use absolute ws:// URLs to control the Host header.)

WS_FD = "ws://localhost:25080"


def _ws_code(client: TestClient, url: str, headers: dict) -> int:
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(url, headers=headers):
            pass
    return exc.value.code


def test_cross_site_websocket_refused(monkeypatch):
    """A page opening ws://localhost:25080/fd/agent-ws/... would chat with an
    agent that has a shell (FD injects the agent's token itself)."""
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", False)
    client = TestClient(fd_server.app)
    assert _ws_code(client, f"{WS_FD}/fd/agent-ws/localhost/24001", {"Origin": EVIL}) == 4403
    # Terminal relay: a page must not register as a PTY worker either.
    assert _ws_code(client, f"{WS_FD}/fd/pty/connect", {"Origin": EVIL}) == 4403


def test_rebinding_websocket_refused(monkeypatch):
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", False)
    client = TestClient(fd_server.app)
    code = _ws_code(client, "ws://evil.example:25080/fd/agent-ws/localhost/24001",
                    {"Origin": "http://evil.example:25080"})
    assert code == 4403


def test_same_origin_websocket_passes_guard(monkeypatch):
    """Own origin gets through the guard to the handler's own check (auth on:
    no fd_token → the handler's 4001, not the guard's 4403)."""
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", True)
    client = TestClient(fd_server.app)
    code = _ws_code(client, f"{WS_FD}/fd/agent-ws/localhost/24001", {"Origin": FD})
    assert code == 4001


# ── agent web_auth only to FD's own pages ────────────────────────────


async def test_web_auth_withheld_from_other_origins_even_with_wildcard_cors(monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", False)
    monkeypatch.setenv("FD_CORS_ORIGINS", "*")  # legacy opt-out
    async with _client() as c:
        r = await c.get("/fd/processes", headers={"Origin": EVIL, "Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == EVIL
        assert [p["web_auth"] for p in r.json()] == [""]
        # FD's own page and non-browser callers (agents, BFF) still get it.
        r = await c.get("/fd/processes", headers={"Sec-Fetch-Site": "same-origin"})
        assert [p["web_auth"] for p in r.json()] == ["tok-alpha"]
        r = await c.get("/fd/processes")
        assert [p["web_auth"] for p in r.json()] == ["tok-alpha"]


def test_may_expose_agent_secrets():
    ok = origin_guard.may_expose_agent_secrets
    assert ok(Headers({}))
    assert ok(Headers({"sec-fetch-site": "same-origin"}))
    assert ok(Headers({"host": "localhost:25080", "origin": FD}))
    assert not ok(Headers({"sec-fetch-site": "same-site"}))
    assert not ok(Headers({"host": "localhost:25080", "origin": "http://localhost:24080"}))


# ── Codex token endpoint: agent identity required ────────────────────


@pytest.fixture
def codex_tokens(monkeypatch):
    fake = SimpleNamespace(access_token="chatgpt-at", account_id="acct",
                           expires_at=0.0, email="me@example.com", plan="pro")
    monkeypatch.setattr(codex_oauth_routes, "load_tokens_from_disk", lambda: fake)
    return fake


async def test_codex_token_refused_to_anonymous_loopback(codex_tokens):
    """Loopback alone used to be enough — any page in the user's browser is on
    loopback."""
    async with _client() as c:
        r = await c.get("/fd/codex/access_token")
    assert r.status_code == 403
    assert "X-Agent-Auth" in r.json()["detail"]


async def test_codex_token_refused_to_browsers(codex_tokens):
    async with _client() as c:
        # Same-origin fetch passes the guard but is still a browser.
        for headers in ({"Origin": FD, "X-Agent-Auth": "tok-alpha"},
                        {"Sec-Fetch-Site": "same-origin", "X-Agent-Auth": "tok-alpha"}):
            r = await c.get("/fd/codex/access_token", headers=headers)
            assert r.status_code == 403
            assert "not browsers" in r.json()["detail"]


async def test_codex_token_refused_to_unknown_agent(codex_tokens):
    async with _client() as c:
        r = await c.get("/fd/codex/access_token", headers={"X-Agent-Auth": "someone-else"})
    assert r.status_code == 403
    assert "not spawned by this Flight Deck" in r.json()["detail"]


async def test_codex_token_remote_needs_transport_secret(codex_tokens):
    async with _client(REMOTE) as c:
        r = await c.get("/fd/codex/access_token", headers={"X-Agent-Auth": "tok-alpha"})
    assert r.status_code == 401


async def test_codex_token_served_to_own_agent(codex_tokens):
    async with _client() as c:
        r = await c.get("/fd/codex/access_token", headers={"X-Agent-Auth": "tok-alpha"})
    assert r.status_code == 200
    assert r.json()["access_token"] == "chatgpt-at"


# ── MCP agent proxy: identity from FD's records, not X-Agent-Slug ────


async def test_mcp_agent_proxy_requires_identity_and_ignores_claimed_slug():
    await mcp_storage.upsert_server({"name": "open", "url": "https://mcp.example/a"})
    await mcp_storage.upsert_server({"name": "beta-only", "url": "https://mcp.example/b",
                                     "allowed_agents": ["beta"]})
    async with _client() as c:
        r = await c.get("/fd/mcp/agent/servers")
        assert r.status_code == 403
        r = await c.get("/fd/mcp/agent/servers", headers={"Origin": FD})
        assert r.status_code == 403  # a browser, even FD's own page
        # tok-alpha is agent "alpha"; claiming to be "beta" doesn't unlock
        # beta's server.
        r = await c.get("/fd/mcp/agent/servers",
                        headers={"X-Agent-Auth": "tok-alpha", "X-Agent-Slug": "beta"})
    assert r.status_code == 200
    assert r.json() == {"servers": [{"name": "open"}]}


def test_resolve_agent_by_auth():
    assert fd_server._find_agent_by_auth("tok-alpha") == (True, "", "alpha")
    assert fd_server._resolve_agent_identity_by_auth("tok-alpha") == (True, "")
    assert fd_server._resolve_agent_identity_by_auth("nope") == (False, "")
    assert fd_server._resolve_agent_identity_by_auth("") == (False, "")
    # Non-ASCII header junk must not raise out of the constant-time compare.
    assert fd_server._resolve_agent_identity_by_auth("tök") == (False, "")


def test_agent_sends_its_identity(monkeypatch):
    from captain_claw import config as cc_config
    from captain_claw.fd_client import flight_deck_headers

    monkeypatch.setattr(cc_config, "get_config",
                        lambda: SimpleNamespace(web=SimpleNamespace(auth_token="tok-alpha")))
    assert flight_deck_headers()["X-Agent-Auth"] == "tok-alpha"


# ── accounts on an auth-disabled deck ────────────────────────────────


@pytest.fixture
async def fd_db(tmp_path: Path):
    from captain_claw.flight_deck import auth as fd_auth
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


async def test_register_refused_when_auth_disabled(fd_db, monkeypatch):
    """With the DB now open on desktop decks, nobody may plant the first (admin)
    account there — it would become real if auth is switched on later."""
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client() as c:
        r = await c.post("/fd/auth/register",
                         json={"email": "mallory@example.com", "password": "hunter22"})
    assert r.status_code == 403
    assert await fd_db.count_users() == 0


async def test_register_bootstrap_unchanged_when_auth_enabled(fd_db, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    async with _client() as c:
        r = await c.post("/fd/auth/register",
                         json={"email": "owner@example.com", "password": "hunter22"})
    assert r.status_code == 200
    assert r.json()["user"]["role"] == "admin"


async def test_fd_db_opened_regardless_of_auth(tmp_path, monkeypatch):
    """_init_fd_db backs Connections (Google/Codex/MCP settings) on the
    auth-disabled desktop deck too."""
    from captain_claw.flight_deck import auth as fd_auth

    monkeypatch.setattr(fd_server, "DATA_DIR", tmp_path)
    prev = fd_auth._db
    try:
        db = await fd_server._init_fd_db()
        assert fd_auth.get_db() is db
        assert (tmp_path / "flight-deck.db").is_file()
        await db.close()
    finally:
        fd_auth._db = prev
