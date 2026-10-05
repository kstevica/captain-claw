"""One Google account per tenant — through the real /fd/google routes.

Tenant = an FD user on a deck (only their own Google account, and their agents
only ever get their token) and a deck on a shared host (no Google state leaks
to another deck's agents). Each class pins one hole the audit found:

* the agent endpoints failed OPEN to the primary owner for any caller they
  couldn't identify (incl. another deck's agent) — now 403, like deep memory,
  and for any browser request (a web page must never read tokens /
  client_secret); checked once more against the REAL registry resolver;
* an auth-disabled (desktop) deck has no DB and can't tell callers apart —
  every route there says "not available" (as on main, minus the 500);
* the agent gate ignored FD_LOCKDOWN / the deck's agent secret;
* the Gmail/Calendar pollers used the primary owner's token for every user,
  and the ``requires_google`` gate read a flag FD never sets;
* ``/login`` bound an expired/missing session to the primary owner;
* the OAuth state wasn't bound to the browser, and ``/login?fd_token=`` could
  be forwarded to another user (login-CSRF) — now a single-use connect ticket
  that only the browser holding its cookie can use;
* ``/callback`` reflected ``error_description`` unescaped (XSS) and posted to '*';
* consent skipped the account chooser;
* ``/logout`` revoked a Google grant another user of the deck still relied on
  (matched by Google ``sub`` only — a forgeable email never skips the revoke).

Google is never contacted: exchange / userinfo / refresh / revoke are mocked.
The app is a bare FastAPI with just the Google router, driven by TestClient
from the loopback address agents use.
"""

import json
import re
import time
from urllib.parse import parse_qs, urlsplit

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import captain_claw.flight_deck.event_sources as es
import captain_claw.flight_deck.event_sources_google as esg
import captain_claw.flight_deck.google_oauth_routes as gr
from captain_claw.flight_deck import agent_secret, auth
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.google_oauth import GoogleOAuthTokens, build_authorization_url

ALICE = "user-alice"  # admin → the deck's primary owner
BOB = "user-bob"
CLOUD = "https://www.googleapis.com/auth/cloud-platform"


def _tokens(access, refresh=None):
    # Far-future expiry so _refresh_if_needed never makes a network call.
    return GoogleOAuthTokens(access_token=access, refresh_token=refresh or f"{access}-refresh",
                             token_type="Bearer", expires_at=time.time() + 3600,
                             scope="openid email")


@pytest.fixture()
async def db(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_PUBLIC_URL", "FD_AGENT_SHARED_SECRET", "FD_COOKIE_SECURE"):
        monkeypatch.delenv(var, raising=False)
    # Keep any agent_secret file this deck might mint inside the test's tmp dir.
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, email, "h", uid.title(), role, now, now),
        )
    await d._db.commit()
    # A configured deployment OAuth client, Vertex-capable so /credentials would
    # hand out refresh_token + client_secret if the owner check let anyone in.
    await d.set_system_setting(gr._K_CLIENT_ID, "cid")
    await d.set_system_setting(gr._K_CLIENT_SECRET, "csecret")
    await d.set_system_setting(gr._K_PROJECT_ID, "proj")
    await d.set_system_setting(gr._K_SCOPES, json.dumps(["openid", "email", CLOUD]))
    gr._pending_oauth.clear()
    gr._connect_tickets.clear()
    gr._primary_owner_cache.update(id=None, at=0.0)
    try:
        yield d
    finally:
        gr._pending_oauth.clear()
        gr._connect_tickets.clear()
        gr._primary_owner_cache.update(id=None, at=0.0)
        agent_secret.reset_cache_for_tests()
        auth.set_auth_db(prev_db)  # don't leave later tests a closed DB
        await d.close()


@pytest.fixture()
def agents(monkeypatch):
    """This deck's agent registry: web_auth token → recorded owner ("" = this
    deck issued the token but recorded no owner; "local" = recorded by the deck
    while auth was off; a deleted user's id = an owner that no longer exists).
    Stands in for ``server._resolve_agent_identity_by_auth`` (the contract:
    (matched, owner) for a token THIS deck issued, else (False, "")) — which
    must exist: the routes fail closed if it goes missing, so a rename would
    otherwise pass here while every agent got 403. TestRealAgentIdentity runs
    the real one."""
    import captain_claw.flight_deck.server as srv
    registry = {
        "tok-alice": ALICE, "tok-bob": BOB, "tok-orphan": "", "tok-local": "local",
        "tok-ghost": "user-deleted",
    }
    monkeypatch.setattr(
        srv, "_resolve_agent_identity_by_auth",
        lambda t: (t in registry, registry.get(t, "")) if t else (False, ""),
    )
    return registry


@pytest.fixture()
def google(monkeypatch):
    """Mock every Google network call the routes make; record what happened."""
    calls = {"exchange": [], "revoke": []}
    identity = {"sub": "g-bob", "email": "bob@gmail.com"}

    async def exchange(**kw):
        calls["exchange"].append(kw)
        return _tokens("fresh-access", "fresh-refresh")

    async def userinfo(access_token):
        return dict(identity)

    async def refresh(**kw):
        raise AssertionError("tokens are fresh; no refresh expected")

    async def revoke(token):
        calls["revoke"].append(token)
        return True

    monkeypatch.setattr(gr, "exchange_code_for_tokens", exchange)
    monkeypatch.setattr(gr, "fetch_user_info", userinfo)
    monkeypatch.setattr(gr, "refresh_access_token", refresh)
    monkeypatch.setattr(gr, "revoke_token", revoke)
    calls["identity"] = identity
    return calls


def _client(host="127.0.0.1"):
    app = FastAPI()
    app.include_router(gr.router)
    return TestClient(app, client=(host, 50123), follow_redirects=False)


def _bearer(uid, role="user"):
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


def _fd_token(uid):
    return auth.create_access_token(uid, "user")


def _ticket(client, uid, headers=None):
    """The SPA's first step: mint a connect ticket with the session. The
    client's cookie jar keeps the ticket cookie, as the browser does."""
    r = client.post("/fd/google/connect-ticket", headers={**_bearer(uid), **(headers or {})})
    assert r.status_code == 200, r.text
    return r.json()["ticket"]


def _start(client, uid, **kw):
    """Connect ticket, then the popup's /login — the SPA's Connect Google."""
    return client.get(f"/fd/google/login?ticket={_ticket(client, uid)}", **kw)


def _state_of(location):
    return parse_qs(urlsplit(location).query)["state"][0]


# ── agent endpoints: identity from FD's records only, else 403 ─────────────


class TestAgentEndpointsFailClosed:
    async def test_no_agent_identity_gets_403_not_the_primary_owner(self, db, agents):
        await gr._store_tokens(db, ALICE, _tokens("ALICE-ACCESS", "ALICE-REFRESH"))
        c = _client()
        for path in ("/fd/google/access_token", "/fd/google/credentials"):
            r = c.get(path)  # e.g. a teammate's agent shell: curl localhost:25080/...
            assert r.status_code == 403, (path, r.text)
            assert "ALICE" not in r.text and "csecret" not in r.text

    async def test_unknown_agent_token_gets_403(self, db, agents):
        # Also the fd_instance case: an agent of ANOTHER deck on this host
        # presents a web_auth token this deck never issued.
        await gr._store_tokens(db, ALICE, _tokens("ALICE-ACCESS", "ALICE-REFRESH"))
        c = _client()
        for path in ("/fd/google/access_token", "/fd/google/credentials"):
            r = c.get(path, headers={"X-Agent-Auth": "deck-b-agent-token"})
            assert r.status_code == 403
            assert "ALICE" not in r.text

    async def test_each_agent_gets_only_its_owners_account(self, db, agents):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await gr._store_tokens(db, BOB, _tokens("bob-access"))
        c = _client()
        r = c.get("/fd/google/access_token", headers={"X-Agent-Auth": "tok-bob"})
        assert r.status_code == 200 and r.json()["access_token"] == "bob-access"
        r = c.get("/fd/google/credentials", headers={"X-Agent-Auth": "tok-alice"})
        assert r.status_code == 200
        assert r.json()["credentials"]["refresh_token"] == "alice-access-refresh"

    async def test_agent_of_unconnected_owner_gets_404_not_someone_else(self, db, agents):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        r = _client().get("/fd/google/access_token", headers={"X-Agent-Auth": "tok-bob"})
        assert r.status_code == 404

    async def test_agent_of_a_deleted_user_gets_403(self, db, agents):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        r = _client().get("/fd/google/access_token", headers={"X-Agent-Auth": "tok-ghost"})
        assert r.status_code == 403
        assert "no longer a user" in r.json()["detail"]
        assert "alice" not in r.text

    async def test_identity_lookup_error_fails_closed(self, db, agents, monkeypatch):
        import captain_claw.flight_deck.server as srv

        def boom(token):
            raise RuntimeError("docker down")

        monkeypatch.setattr(srv, "_resolve_agent_identity_by_auth", boom)
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        r = _client().get("/fd/google/access_token", headers={"X-Agent-Auth": "tok-alice"})
        assert r.status_code == 403


class TestAgentEndpointsRefuseBrowsers:
    """Agents' httpx sends no Origin / Sec-Fetch-*; every browser request does.
    A web page must never read a user's token, refresh token or client_secret
    (e.g. through a DNS-rebound loopback name)."""

    BROWSER_HEADERS = [
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Mode": "cors"},
        {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"},
    ]

    @pytest.mark.parametrize("extra", BROWSER_HEADERS)
    async def test_browser_request_is_refused_even_with_a_valid_agent_token(
        self, db, agents, extra
    ):
        await gr._store_tokens(db, ALICE, _tokens("OWNER-ACCESS", "OWNER-REFRESH"))
        c = _client()
        for path in ("/fd/google/access_token", "/fd/google/credentials"):
            r = c.get(path, headers={"X-Agent-Auth": "tok-alice", **extra})
            assert r.status_code == 403, (path, extra, r.text)
            for secret in ("ACCESS", "REFRESH", "csecret"):
                assert secret not in r.text

    async def test_the_same_agent_without_browser_headers_passes(self, db, agents):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        r = _client().get("/fd/google/access_token", headers={"X-Agent-Auth": "tok-alice"})
        assert r.status_code == 200 and r.json()["access_token"] == "alice-access"


class TestOwnerlessAgents:
    """Agents THIS deck issued a token to but recorded no owner for (spawned
    while auth was off, pre-owner records): the sole user on a single-user deck,
    a distinct "respawn" 403 otherwise — never the primary owner."""

    async def _drop_bob(self, db):
        await db._db.execute("DELETE FROM users WHERE id = ?", (BOB,))
        await db._db.commit()

    @pytest.mark.parametrize("token", ["tok-orphan", "tok-local"])
    async def test_single_user_deck_serves_its_only_user(self, db, agents, token):
        await self._drop_bob(db)
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        c = _client()
        r = c.get("/fd/google/access_token", headers={"X-Agent-Auth": token})
        assert r.status_code == 200 and r.json()["access_token"] == "alice-access"
        r = c.get("/fd/google/credentials", headers={"X-Agent-Auth": token})
        assert r.status_code == 200

    @pytest.mark.parametrize("token", ["tok-orphan", "tok-local"])
    async def test_multi_user_deck_refuses_with_a_respawn_reason(self, db, agents, token):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))  # the primary owner
        r = _client().get("/fd/google/access_token", headers={"X-Agent-Auth": token})
        assert r.status_code == 403
        detail = r.json()["detail"]
        assert "can't attribute this agent to a user" in detail and "respawn" in detail
        assert "alice-access" not in r.text

    async def test_unknown_and_missing_tokens_have_their_own_reasons(self, db, agents):
        c = _client()
        unknown = c.get("/fd/google/access_token", headers={"X-Agent-Auth": "deck-b-token"})
        missing = c.get("/fd/google/access_token")
        assert unknown.status_code == missing.status_code == 403
        assert "not spawned by this Flight Deck" in unknown.json()["detail"]
        assert "X-Agent-Auth" in missing.json()["detail"]


class TestAgentGate:
    """Same transport rule as server._agent_caller_ok (the PR #21 guard)."""

    async def test_off_host_caller_without_secret_is_refused(self, db, agents):
        await gr._store_tokens(db, BOB, _tokens("bob-access"))
        r = _client("10.1.2.3").get("/fd/google/access_token",
                                    headers={"X-Agent-Auth": "tok-bob"})
        assert r.status_code == 401

    async def test_off_host_caller_with_the_decks_secret_passes(self, db, agents, monkeypatch):
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "deck-secret")
        await gr._store_tokens(db, BOB, _tokens("bob-access"))
        c = _client("10.1.2.3")
        ok = c.get("/fd/google/access_token",
                   headers={"X-Agent-Auth": "tok-bob", "X-Agent-Secret": "deck-secret"})
        assert ok.status_code == 200 and ok.json()["access_token"] == "bob-access"
        bad = c.get("/fd/google/access_token",
                    headers={"X-Agent-Auth": "tok-bob", "X-Agent-Secret": "other-deck"})
        assert bad.status_code == 401

    async def test_lockdown_makes_the_secret_mandatory_even_from_loopback(
        self, db, agents, monkeypatch
    ):
        monkeypatch.setenv("FD_LOCKDOWN", "1")
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "deck-secret")
        await gr._store_tokens(db, BOB, _tokens("bob-access"))
        c = _client()  # loopback — e.g. a same-host TLS proxy
        assert c.get("/fd/google/access_token",
                     headers={"X-Agent-Auth": "tok-bob"}).status_code == 401
        r = c.get("/fd/google/access_token",
                  headers={"X-Agent-Auth": "tok-bob", "X-Agent-Secret": "deck-secret"})
        assert r.status_code == 200 and r.json()["access_token"] == "bob-access"


# ── the agent path against the REAL identity resolver (no stub) ────────────


class _Container:
    def __init__(self, labels: dict, status: str = "running", name: str = "cc-box"):
        self.labels = labels
        self.status = status
        self.name = name  # docker-py always has one; the lookup derives the slug from it


class _DockerHost:
    """The slice of docker-py the deck's label lookups use."""

    def __init__(self, items: list[_Container]):
        self.containers = self
        self._items = items

    def list(self, all: bool = False, filters: dict | None = None):
        label = (filters or {}).get("label")
        return [c for c in self._items
                if (all or c.status == "running") and (not label or label in c.labels)]


OTHER_DECK = "0123456789abcdef"


@pytest.fixture()
def real_identity(db, monkeypatch, tmp_path):
    """This deck's own records, kept the way ``server`` keeps them: a process
    registry file under a tmp DATA_DIR, and a Docker host shared with ANOTHER
    deck whose container names Alice as owner with a token of its choosing.
    ``server._resolve_agent_identity_by_auth`` itself is NOT replaced."""
    import captain_claw.flight_deck.server as srv

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(srv, "DATA_DIR", data)
    monkeypatch.setattr(srv, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(srv, "_process_is_alive", lambda slug: True)
    srv._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "web_port": 24101, "web_auth": "reg-bob",
                      "owner": BOB},
        "alice-agent": {"slug": "alice-agent", "web_port": 24102, "web_auth": "reg-alice",
                        "owner": ALICE},
    })

    def labels(token, owner, deck):
        return {srv.CONTAINER_LABEL: "true", srv.OWNER_LABEL: owner,
                "flight-deck.web-auth": token, srv.DECK_LABEL: deck}

    monkeypatch.setattr(srv, "get_docker", lambda: _DockerHost([
        _Container(labels("foreign-tok", ALICE, OTHER_DECK)),
        _Container(labels("box-bob", BOB, srv._deck_id())),
    ]))
    return srv


class TestRealAgentIdentity:
    async def _connect_both(self, db):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await gr._store_tokens(db, BOB, _tokens("bob-access"))

    async def test_each_registered_agent_gets_its_owners_account(self, db, real_identity):
        await self._connect_both(db)
        c = _client()
        for token, access in (("reg-bob", "bob-access"), ("reg-alice", "alice-access"),
                              ("box-bob", "bob-access")):  # this deck's own container
            r = c.get("/fd/google/access_token", headers={"X-Agent-Auth": token})
            assert r.status_code == 200 and r.json()["access_token"] == access, token
        r = c.get("/fd/google/credentials", headers={"X-Agent-Auth": "reg-bob"})
        assert r.json()["credentials"]["refresh_token"] == "bob-access-refresh"

    async def test_another_decks_container_and_unknown_tokens_get_nothing(
        self, db, real_identity
    ):
        await self._connect_both(db)
        c = _client()
        for token in ("foreign-tok", "not-issued-here"):
            for path in ("/fd/google/access_token", "/fd/google/credentials"):
                r = c.get(path, headers={"X-Agent-Auth": token})
                assert r.status_code == 403, (token, path)
                assert "not spawned by this Flight Deck" in r.json()["detail"]
                assert "alice-access" not in r.text

    async def test_the_real_agent_client_works_under_lockdown(
        self, db, real_identity, monkeypatch
    ):
        # FD_LOCKDOWN with no FD_AGENT_SHARED_SECRET in the env: the deck
        # relies on its per-deck agent_secret file, which the agent's Google
        # client must send (it used to send only X-Agent-Auth → 401).
        import httpx

        import captain_claw.config as config_mod
        import captain_claw.google_oauth_manager as gom
        from captain_claw.config import Config

        monkeypatch.setenv("FD_LOCKDOWN", "1")
        await self._connect_both(db)
        app = FastAPI()
        app.include_router(gr.router)
        assert TestClient(app, client=("127.0.0.1", 50123)).get(
            "/fd/google/access_token", headers={"X-Agent-Auth": "reg-bob"}).status_code == 401

        monkeypatch.setattr(config_mod, "_config", Config(
            web={"auth_token": "reg-bob"},
            google_oauth={"flight_deck_url": "http://fd.test"},
        ))
        real = httpx.AsyncClient
        monkeypatch.setattr(gom.httpx, "AsyncClient", lambda **kw: real(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50123)), **kw))
        tokens = await gom.GoogleOAuthManager(object()).get_tokens()
        assert tokens.access_token == "bob-access"


# ── FD-side consumers: the event-spine pollers ─────────────────────────────


class _RecordingClient:
    """Stands in for httpx.AsyncClient inside the Google pollers."""

    instances: list["_RecordingClient"] = []

    def __init__(self, *a, **kw):
        self.headers_seen = []
        _RecordingClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        self.headers_seen.append(dict(headers or {}))

        class _R:
            status_code = 200

            @staticmethod
            def json():
                return {"items": [], "messages": []}

        return _R()


class _FakeStore:
    def __init__(self):
        self.state = {}

    def get_poll_state(self, user_id, source):
        return self.state.get((user_id, source), {})

    def set_poll_state(self, user_id, source, *, last_poll_at, cursor=None):
        self.state[(user_id, source)] = {"last_poll_at": last_poll_at, "cursor": cursor}

    def add_event(self, user_id, **kw):
        return {"id": 1}


class TestPollersArePerUser:
    async def test_user_id_is_required_and_never_defaults_to_the_primary_owner(self, db):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        with pytest.raises(TypeError):
            await gr.get_valid_google_access_token()  # type: ignore[call-arg]
        assert await gr.get_valid_google_access_token("") is None
        assert await gr.get_valid_google_access_token(BOB) is None  # not alice's
        assert await gr.get_valid_google_access_token(ALICE) == "alice-access"

    async def test_unconnected_users_poll_never_reads_the_primary_owners_mail(
        self, db, monkeypatch
    ):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        _RecordingClient.instances = []
        monkeypatch.setattr(esg.httpx, "AsyncClient", _RecordingClient)
        assert await esg.poll_gmail(BOB, "") == ([], "")
        assert await esg.poll_calendar(BOB, "c0") == ([], "c0")
        assert _RecordingClient.instances == []  # no Google call at all

    async def test_each_users_poll_uses_their_own_token(self, db, monkeypatch):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await gr._store_tokens(db, BOB, _tokens("bob-access"))
        _RecordingClient.instances = []
        monkeypatch.setattr(esg.httpx, "AsyncClient", _RecordingClient)
        await esg.poll_gmail(BOB, "")
        await esg.poll_calendar(ALICE, "")
        seen = [h["Authorization"] for c in _RecordingClient.instances for h in c.headers_seen]
        assert seen[0] == "Bearer bob-access"
        assert set(seen[1:]) == {"Bearer alice-access"}

    def test_google_adapters_are_gated_on_google(self):
        gated = {a.name: a.requires_google for a in es._ADAPTERS}
        assert gated["gmail"] is True and gated["calendar"] is True

    async def test_requires_google_gate_is_per_user(self, db, monkeypatch):
        # Previously read an agent-side process flag FD never sets → never polled.
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        polled = []

        async def poll(uid, cursor):
            polled.append(uid)
            return [], cursor

        monkeypatch.setattr(es, "_ADAPTERS", [es.Adapter(
            name="g", interval_seconds=0, poll=poll, enabled=lambda u: True,
            requires_google=True)])
        monkeypatch.setattr(es, "get_store", _FakeStore)

        async def no_custom(*a):
            return 0

        monkeypatch.setattr(es, "_poll_custom_sources", no_custom)
        await es.poll_user(ALICE)
        await es.poll_user(BOB)
        assert polled == [ALICE]

    async def test_custom_requires_google_source_polls_only_connected_users(
        self, db, monkeypatch
    ):
        from captain_claw.flight_deck import actions, autonomy, fd_dispatch
        await gr._store_tokens(db, BOB, _tokens("bob-access"))
        ran = []

        async def run_tool(agent, tool, args):
            ran.append(agent["owner"])
            return {"ok": True, "content": ""}

        monkeypatch.setattr(autonomy, "resolve_config", lambda uid: {"custom_sources": [
            {"name": "gm", "tool": "google_mail", "enabled": True, "requires_google": True}]})
        monkeypatch.setattr(fd_dispatch, "_strongest_agent", lambda uid: {"owner": uid})
        monkeypatch.setattr(actions, "run_tool_on_agent", run_tool)
        await es._poll_custom_sources(ALICE, time.time(), _FakeStore())
        await es._poll_custom_sources(BOB, time.time(), _FakeStore())
        assert ran == [BOB]


# ── /login: who is connecting (a connect ticket, never a URL JWT) ──────────


class TestConnectTicket:
    """``POST /connect-ticket`` (the SPA, with its session) → a single-use
    ticket, also set as a cookie; ``/login?ticket=`` needs both. A /login link
    forwarded to another browser therefore starts nothing there."""

    def _expired(self, uid):
        past = int(time.time()) - 3600
        return jwt.encode({"sub": uid, "role": "user", "iat": past - 900, "exp": past,
                           "type": "access"}, auth.get_jwt_secret(), algorithm=auth.ALGORITHM)

    async def test_minting_needs_a_session(self, db):
        c = _client()
        for headers in ({}, {"Authorization": f"Bearer {self._expired(BOB)}"},
                        {"Authorization": "Bearer not-a-jwt"}):
            r = c.post("/fd/google/connect-ticket", headers=headers)
            assert r.status_code == 401, headers
        assert gr._connect_tickets == {}

    async def test_ticket_cookie_is_bound_to_the_browser(self, db):
        r = _client().post("/fd/google/connect-ticket", headers=_bearer(BOB))
        ticket = r.json()["ticket"]
        assert gr._connect_tickets[ticket]["owner"] == BOB
        assert r.headers["cache-control"] == "no-store"
        cookie = r.headers["set-cookie"].lower()
        assert cookie.startswith(f"{gr._TICKET_COOKIE}={ticket}".lower())
        for attr in ("httponly", "samesite=strict", "path=/fd/google", "max-age=120"):
            assert attr in cookie
        assert "secure" not in cookie  # plain http here
        r = _client().post("/fd/google/connect-ticket", headers={
            **_bearer(BOB), "X-Forwarded-Proto": "https"})
        assert "secure" in r.headers["set-cookie"].lower()

    async def test_ticket_starts_a_flow_for_its_user_once(self, db):
        c = _client()
        ticket = _ticket(c, BOB)
        r = c.get(f"/fd/google/login?ticket={ticket}")
        assert r.status_code == 302
        loc = r.headers["location"]
        assert loc.startswith("https://accounts.google.com/")
        q = parse_qs(urlsplit(loc).query)
        assert q["prompt"] == ["select_account consent"]
        assert q["access_type"] == ["offline"]
        assert gr._pending_oauth[q["state"][0]]["owner"] == BOB
        cookies = r.headers.get_list("set-cookie")
        state = next(v.lower() for v in cookies if v.startswith(f"{gr._STATE_COOKIE}="))
        for attr in ("httponly", "samesite=lax", "path=/fd/google", "max-age=600"):
            assert attr in state
        assert "secure" not in state  # plain http here
        spent = next(v.lower() for v in cookies if v.startswith(f"{gr._TICKET_COOKIE}="))
        assert "max-age=0" in spent
        # Single-use: the same ticket can't start a second flow.
        assert gr._connect_tickets == {}
        c.cookies.set(gr._TICKET_COOKIE, ticket, path="/fd/google")
        again = c.get(f"/fd/google/login?ticket={ticket}")
        assert again.status_code == 401 and len(gr._pending_oauth) == 1

    async def test_forwarded_login_link_does_nothing_in_another_browser(self, db):
        # The review's login-CSRF: Bob gets Alice to open HIS /login link. Her
        # browser doesn't hold his ticket — whether it holds none, or her own
        # (she's mid-connect) — so nothing binds her consent to his account.
        bob, alice = _client(), _client()
        _ticket(alice, ALICE)  # her own, valid ticket cookie
        for alices_browser in (alice, _client()):
            link = f"/fd/google/login?ticket={_ticket(bob, BOB)}"
            r = alices_browser.get(link)
            assert r.status_code == 401 and "Reload Flight Deck" in r.text
            assert gr._STATE_COOKIE not in alices_browser.cookies
            # The forwarded ticket is spent: Bob can't use it afterwards either.
            assert bob.get(link).status_code == 401
        assert gr._pending_oauth == {}

    async def test_a_jwt_in_the_url_no_longer_starts_a_flow(self, db):
        r = _client().get(f"/fd/google/login?fd_token={_fd_token(BOB)}")
        assert r.status_code == 401
        assert gr._pending_oauth == {} and gr._STATE_COOKIE not in r.cookies

    @pytest.mark.parametrize("case", ["none", "unknown", "expired", "deleted-user"])
    async def test_no_usable_ticket_is_refused_not_bound_to_the_primary_owner(self, db, case):
        c = _client()
        ticket = _ticket(c, BOB)
        if case == "expired":
            gr._connect_tickets[ticket]["ts"] -= gr._TICKET_TTL + 1
        elif case == "deleted-user":
            await db._db.execute("DELETE FROM users WHERE id = ?", (BOB,))
            await db._db.commit()
        url = {"none": "/fd/google/login", "unknown": "/fd/google/login?ticket=forged"}.get(
            case, f"/fd/google/login?ticket={ticket}")
        r = c.get(url)
        assert r.status_code == 401
        assert "Reload Flight Deck and click Connect again" in r.text
        assert gr._pending_oauth == {}  # nothing to complete
        assert gr._STATE_COOKIE not in r.cookies

    async def test_a_newer_ticket_replaces_the_users_older_one(self, db):
        c = _client()
        first = _ticket(c, BOB)
        second = _ticket(c, BOB)
        _ticket(_client(), ALICE)
        assert first not in gr._connect_tickets and second in gr._connect_tickets
        assert len(gr._connect_tickets) == 2  # bob's newest + alice's
        assert c.get(f"/fd/google/login?ticket={second}").status_code == 302

    async def test_non_ascii_cookie_or_ticket_is_a_refusal_not_a_500(self, db):
        ticket = _ticket(_client(), BOB)
        raw = {"Cookie": f"{gr._TICKET_COOKIE}=t\u00efcket".encode("latin-1")}
        other = _client()
        assert other.get(f"/fd/google/login?ticket={ticket}", headers=raw).status_code == 401
        assert other.get("/fd/google/login?ticket=%C3%AF").status_code == 401


class TestAuthDisabledDeck:
    """FD_AUTH_ENABLED=false (desktop): FD keeps no DB there and can't tell its
    callers — or a web page on its loopback — apart, so Google via FD is not
    available (as on main), and says so instead of a 500. Even with a DB left
    over from an auth-enabled past, nothing is served."""

    UI = [("get", "/fd/google/status"), ("get", "/fd/google/probe"),
          ("get", "/fd/google/config"), ("post", "/fd/google/config"),
          ("get", "/fd/google/scope_catalog"), ("post", "/fd/google/connect-ticket"),
          ("get", "/fd/google/login"), ("get", "/fd/google/callback?code=c&state=s"),
          ("post", "/fd/google/logout")]
    AGENT = ["/fd/google/access_token", "/fd/google/credentials"]

    def _assert_refused(self, c):
        for method, path in self.UI:
            r = getattr(c, method)(path, **({"json": {"clear": True}} if method == "post" else {}))
            assert r.status_code == 503, (path, r.text)
            assert "auth is disabled" in r.json()["detail"]
        for path in self.AGENT:
            r = c.get(path, headers={"X-Agent-Auth": "tok-orphan"})
            assert r.status_code == 403, path  # the agent's tools surface 401/403
            assert "auth is disabled" in r.json()["detail"]
            assert "LOCAL" not in r.text and "csecret" not in r.text
            # The machine-readable "auth is off on this single-tenant deck"
            # marker (the retired gws tool kept its own credentials on it).
            assert r.headers.get(gr.AUTH_OFF_HEADER) == gr.AUTH_OFF_VALUE == "auth-disabled"

    async def test_every_route_says_not_available(self, db, agents, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        await db.set_system_setting(gr._K_TOKENS, json.dumps(
            _tokens("LOCAL-ACCESS", "LOCAL-REFRESH").to_dict()))
        self._assert_refused(_client())
        # Nothing was changed: /config's clear=True never ran.
        assert await db.get_system_setting(gr._K_CLIENT_ID) == "cid"
        assert gr._connect_tickets == {} and gr._pending_oauth == {}

    async def test_no_db_is_a_clear_answer_not_a_500(self, db, agents, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        auth._db = None  # what an auth-off deck has (the fixture restores it)
        try:
            self._assert_refused(_client())
        finally:
            auth._db = db

    async def test_agent_owner_refuses_even_when_called_directly(self, db, agents, monkeypatch):
        from fastapi import HTTPException

        from captain_claw.flight_deck import server as srv
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        req = type("R", (), {"headers": {"X-Agent-Auth": "tok-bob"}, "client": None})()
        assert srv._resolve_agent_identity_by_auth("tok-bob") == (True, BOB)
        with pytest.raises(HTTPException) as ei:
            await gr._agent_owner(req)
        assert ei.value.status_code == 403

    async def test_pollers_get_nothing(self, db, monkeypatch):
        await gr._store_tokens(db, ALICE, _tokens("alice-access"))
        await db.set_system_setting(gr._K_TOKENS, json.dumps(_tokens("LOCAL").to_dict()))
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        for uid in (ALICE, "local", "some-registry-owner"):
            assert await gr.get_valid_google_access_token(uid) is None
            assert await gr.is_google_connected(uid) is False


# ── OAuth state bound to the initiating browser (login-CSRF) ───────────────


class TestStateBoundToBrowser:
    async def test_callback_in_another_browser_is_refused(self, db, google):
        bob = _client()
        state = _state_of(_start(bob, BOB).headers["location"])
        # Bob forwards the Google URL; ALICE consents in her own browser.
        alice = _client()
        r = alice.get(f"/fd/google/callback?code=alice-code&state={state}")
        assert r.status_code == 400
        assert "not started in this browser" in r.text
        assert google["exchange"] == []  # alice's code was never redeemed
        assert await gr._load_tokens(db, BOB) is None
        # Single-use: Bob can't finish it afterwards either.
        assert "Invalid or expired state" in bob.get(
            f"/fd/google/callback?code=x&state={state}").text

    async def test_same_browser_completes_and_the_cookie_is_cleared(self, db, google):
        bob = _client()
        state = _state_of(_start(bob, BOB).headers["location"])
        r = bob.get(f"/fd/google/callback?code=bob-code&state={state}")
        assert r.status_code == 200 and "Connected" in r.text
        assert (await gr._load_tokens(db, BOB)).access_token == "fresh-access"
        assert await gr._load_tokens(db, ALICE) is None
        assert (await gr._load_user(db, BOB))["sub"] == "g-bob"
        assert f'{gr._STATE_COOKIE}=""' in r.headers["set-cookie"] or \
            "max-age=0" in r.headers["set-cookie"].lower()
        # Replay of the same state is dead.
        assert "Invalid or expired state" in bob.get(
            f"/fd/google/callback?code=bob-code&state={state}").text

    async def test_success_page_names_the_flight_deck_account(self, db, google):
        # A mis-bind (someone else's FD account) is visible on the spot.
        bob = _client()
        state = _state_of(_start(bob, BOB).headers["location"])
        r = bob.get(f"/fd/google/callback?code=c&state={state}")
        assert "Linked bob@gmail.com to Flight Deck user bob@x.co" in r.text
        m = re.search(r"var payload = (.*?);\n", r.text)
        assert json.loads(m.group(1))["email"] == "bob@gmail.com"

    async def test_non_ascii_state_cookie_is_a_refusal_not_a_500(self, db, google):
        state = _state_of(_start(_client(), BOB).headers["location"])
        raw = {"Cookie": f"{gr._STATE_COOKIE}=br\u00efwser".encode("latin-1")}
        r = _client().get(f"/fd/google/callback?code=c&state={state}", headers=raw)
        assert r.status_code == 400 and google["exchange"] == []

    async def test_stale_state_is_expired(self, db, google):
        bob = _client()
        state = _state_of(_start(bob, BOB).headers["location"])
        gr._pending_oauth[state]["ts"] -= gr._PENDING_TTL + 1
        r = bob.get(f"/fd/google/callback?code=c&state={state}")
        assert "Invalid or expired state" in r.text and google["exchange"] == []

    async def test_cookie_is_secure_behind_a_tls_proxy(self, db):
        tls = {"X-Forwarded-Proto": "https"}
        c = _client()
        ticket = _ticket(c, BOB, headers=tls)
        # httpx withholds a Secure cookie over http; the proxy would send it.
        r = c.get(f"/fd/google/login?ticket={ticket}",
                  headers={**tls, "Cookie": f"{gr._TICKET_COOKIE}={ticket}"})
        assert r.status_code == 302
        state = next(v for v in r.headers.get_list("set-cookie")
                     if v.startswith(f"{gr._STATE_COOKIE}="))
        assert "secure" in state.lower()

    async def test_cookie_follows_the_browsers_scheme_not_the_deck_policy(
        self, db, monkeypatch
    ):
        # FD_LOCKDOWN makes FD's own refresh cookie Secure, but a Secure cookie
        # sent to a plain-http browser is dropped — the callback could never
        # match it. The browser's scheme decides.
        monkeypatch.setenv("FD_LOCKDOWN", "1")
        r = _start(_client(), BOB)
        assert r.status_code == 302
        cookie = r.headers["set-cookie"].lower()
        assert f"{gr._STATE_COOKIE}=" in cookie and "secure" not in cookie


class TestPublicUrl:
    """With FD_PUBLIC_URL set Google returns to THAT origin, and the state
    cookie is only sent back there. /login doesn't pre-check the origin (a
    reverse proxy may forward any Host); a flow started elsewhere just can't
    complete, and /callback says where to go."""

    PUBLIC = "https://claw.example.com"

    @pytest.fixture(autouse=True)
    def _public(self, db, monkeypatch):  # after db, which clears FD_PUBLIC_URL
        monkeypatch.setenv("FD_PUBLIC_URL", self.PUBLIC)

    async def test_on_the_public_origin_the_flow_completes(self, db, google):
        browser = _client()  # one cookie jar, absolute URLs: a real browser
        base = f"{self.PUBLIC}/fd/google"
        ticket = browser.post(f"{base}/connect-ticket", headers=_bearer(BOB)).json()["ticket"]
        r = browser.get(f"{base}/login?ticket={ticket}")
        assert r.headers["location"].startswith("https://accounts.google.com/")
        assert "redirect_uri=https%3A%2F%2Fclaw.example.com%2Ffd%2Fgoogle%2Fcallback" \
            in r.headers["location"]
        assert "secure" in r.headers["set-cookie"].lower()
        state = _state_of(r.headers["location"])
        done = browser.get(f"{base}/callback?code=c&state={state}")
        assert done.status_code == 200 and "Connected" in done.text
        assert (await gr._load_tokens(db, BOB)).access_token == "fresh-access"

    @pytest.mark.parametrize("headers", [
        {"Host": "127.0.0.1:25080"},  # nginx default: proxy_set_header Host upstream
        {"Host": "127.0.0.1:25080", "X-Forwarded-Proto": "http"},  # inner hop says http
        {"Host": "claw.example.com"},  # TLS proxy forwarding Host but no scheme
    ])
    async def test_behind_a_proxy_on_the_public_origin_connect_works(self, db, google, headers):
        # The browser is on the public origin; whatever Host FD sees, the
        # browser keeps the cookie for the host IT visited.
        browser = _client()
        ticket = browser.post(f"{self.PUBLIC}/fd/google/connect-ticket",
                              headers=_bearer(BOB)).json()["ticket"]
        r = browser.get(f"{self.PUBLIC}/fd/google/login?ticket={ticket}", headers=headers)
        assert r.headers["location"].startswith("https://accounts.google.com/")
        done = browser.get(f"{self.PUBLIC}/fd/google/callback?code=c&state="
                           f"{_state_of(r.headers['location'])}")
        assert done.status_code == 200 and "Connected" in done.text
        assert (await gr._load_tokens(db, BOB)).access_token == "fresh-access"

    async def test_started_off_the_public_origin_cannot_complete(self, db, google):
        browser = _client()  # e.g. a LAN IP: the cookie stays on that host
        ticket = _ticket(browser, BOB)
        r = browser.get(f"http://testserver/fd/google/login?ticket={ticket}")
        state = _state_of(r.headers["location"])
        done = browser.get(f"{self.PUBLIC}/fd/google/callback?code=c&state={state}")
        assert done.status_code == 400
        assert "not started in this browser" in done.text
        assert self.PUBLIC in done.text  # names the address to use
        assert await gr._load_tokens(db, BOB) is None


# ── callback page: no reflected XSS, no postMessage to '*' ─────────────────


class TestCallbackPageEscaping:
    PAYLOAD = "<img src=x onerror=alert(1)></script><script>alert(2)</script><!--"

    def test_error_description_is_escaped_everywhere(self, db):
        r = _client().get("/fd/google/callback",
                          params={"error": "x", "error_description": self.PAYLOAD})
        body = r.text
        assert "<img" not in body and "<script>alert" not in body
        assert "&lt;img src=x onerror=alert(1)&gt;" in body
        # The JS literal round-trips to the exact string, with no raw '<'.
        m = re.search(r"var payload = (.*?);\n", body)
        assert m and "<" not in m.group(1)
        assert json.loads(m.group(1))["detail"] == self.PAYLOAD

    def test_result_goes_only_to_the_same_origin_under_a_nonce_csp(self, db):
        r = _client().get("/fd/google/callback", params={"error": "access_denied"})
        assert "'*'" not in r.text
        assert "postMessage(payload, origin)" in r.text
        csp = r.headers["content-security-policy"]
        nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
        assert f'<script nonce="{nonce}">' in r.text
        assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp


def test_consent_always_shows_the_account_chooser():
    q = parse_qs(urlsplit(build_authorization_url("cid", "http://x/cb", state="s")).query)
    assert q["prompt"] == ["select_account consent"]  # not a silent same-account bind
    assert q["access_type"] == ["offline"]  # still returns a refresh_token


# ── /logout: don't revoke a grant another user still holds ─────────────────


class TestLogoutSharedAccount:
    async def _connect(self, db, uid, access, info):
        await gr._store_tokens(db, uid, _tokens(access))
        if info is not None:
            await gr._store_user(db, uid, info)

    async def test_same_google_account_elsewhere_skips_the_revoke(self, db, google):
        await self._connect(db, ALICE, "alice", {"sub": "g-ops", "email": "ops@co.com"})
        await self._connect(db, BOB, "bob", {"sub": "g-ops", "email": "ops@co.com"})
        r = _client().post("/fd/google/logout", headers=_bearer(BOB))
        assert r.json() == {"disconnected": True}
        assert google["revoke"] == []
        assert await gr._load_tokens(db, BOB) is None  # bob's local state cleared
        assert (await gr._load_tokens(db, ALICE)).access_token == "alice"

    async def test_email_alone_never_skips_the_revoke(self, db, google):
        # An identity blob without Google's stable ``sub`` (e.g. one forged
        # before /fd/settings refused google_oauth:* writes) can't claim to
        # share BOB's account: his Disconnect still revokes.
        await self._connect(db, ALICE, "alice", {"email": "Ops@Co.com"})
        await self._connect(db, BOB, "bob", {"sub": "g-ops", "email": "ops@co.com"})
        _client().post("/fd/google/logout", headers=_bearer(BOB))
        assert google["revoke"] == ["bob-refresh"]

    async def test_missing_sub_on_the_leaving_user_still_revokes(self, db, google):
        await self._connect(db, ALICE, "alice", {"sub": "g-ops", "email": "ops@co.com"})
        await self._connect(db, BOB, "bob", {"email": "ops@co.com"})
        _client().post("/fd/google/logout", headers=_bearer(BOB))
        assert google["revoke"] == ["bob-refresh"]

    async def test_own_account_is_revoked(self, db, google):
        await self._connect(db, ALICE, "alice", {"sub": "g-alice", "email": "a@co.com"})
        await self._connect(db, BOB, "bob", {"sub": "g-bob", "email": "b@co.com"})
        _client().post("/fd/google/logout", headers=_bearer(BOB))
        assert google["revoke"] == ["bob-refresh"]
        assert await gr._load_tokens(db, BOB) is None

    async def test_disconnected_holder_does_not_block_the_revoke(self, db, google):
        # Alice once connected the same account but no longer holds tokens.
        await gr._store_user(db, ALICE, {"sub": "g-ops"})
        await self._connect(db, BOB, "bob", {"sub": "g-ops"})
        _client().post("/fd/google/logout", headers=_bearer(BOB))
        assert google["revoke"] == ["bob-refresh"]

    async def test_unknown_identity_still_revokes(self, db, google):
        await self._connect(db, ALICE, "alice", {"sub": "g-ops"})
        await self._connect(db, BOB, "bob", None)
        _client().post("/fd/google/logout", headers=_bearer(BOB))
        assert google["revoke"] == ["bob-refresh"]


class TestPrimaryOwnerCache:
    async def test_stale_cache_is_recomputed(self, db):
        gr._primary_owner_cache.update(id="user-demoted", at=time.time() - 3600)
        assert await gr._primary_owner(db) == ALICE

    async def test_auth_disabled_primary_owner_is_the_local_user(self, db, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        assert await gr._primary_owner(db) == "local"
