"""Spawn-time ownership integrity — "one Google account per tenant".

An agent's tenant is the owner FD records for it at spawn: its web_auth maps
back to that owner (``_resolve_agent_owner_by_auth``), and that decides whose
Google account, VFS root, deep-memory pool and Library keys it acts with. These
tests pin that the recorded owner can only come from a verified identity or
FD's own records, never from a request body or an unrelated agent:

* ``_resolve_spawn_owner`` — JWT / in-process stub / X-Agent-Auth (the ONLY
  identity an unauthenticated HTTP spawn can have: no owner_hint, no sole-user
  fallback, no browser Origin) / auth-disabled branches, the transport guard,
  web_auth pinning.
* ``POST /fd/spawn-process`` end to end (Popen faked): the audited forgery,
  web_auth collisions, and the child env (owner + FD_URL pinned to THIS deck,
  FD-only secrets and ambient gws credentials dropped).
* Deck scoping: Docker labels are host-global, so every label-based identity /
  ownership lookup ignores containers another deck on the host spawned
  (``_resolve_agent_identity_by_auth`` and friends; Docker faked).
* The agent-side ``flight_deck`` tool's spawn, through ASGI into FD.
* Restart (``_start_registered_process``) re-pins owner + FD_URL.
* ``clone_process`` keeps the owner and mints the clone its own web_auth.
* ``_resolve_primary_owner`` is the genuinely oldest admin past 100 users.
* ``_fd_self_url`` / ``main()`` record the deck's real bound port.
* ``_init_fd_db`` opens the settings DB only with auth enabled (as on main):
  an auth-disabled deck has no /fd/auth/register a web page could use.
* Non-agent subprocesses (hosted VFS apps, stdio MCP servers) don't inherit
  FD's own secrets either.

No real process, container or server is ever started.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import docker
import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request as StarletteRequest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

# Imported up front: importing the tools package runs subprocesses, which the
# spawn fixtures fake.
from captain_claw.tools.flight_deck import FlightDeckTool

ALICE = "user-alice"   # admin, owns an agent
BOB = "user-bob"       # regular user, owns an agent
CAROL = "user-carol"   # regular user, owns no agent
LOOPBACK = ("127.0.0.1", 40001)
REMOTE = ("203.0.113.9", 40001)


# ── fixtures ─────────────────────────────────────────────────────────


async def _insert_users(db: FlightDeckDB, rows: list[tuple[str, str, str]]) -> None:
    for i, (uid, role, created) in enumerate(rows):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (uid, f"{uid}-{i}@x.co", "h", uid, role, created, created),
        )
    await db._db.commit()


@pytest.fixture
async def fd_db(tmp_path: Path):
    """A real FlightDeckDB wired into the auth module (restored afterwards)."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


@pytest.fixture
async def users(fd_db):
    await _insert_users(fd_db, [
        (ALICE, "admin", "2026-01-01T00:00:00Z"),
        (BOB, "user", "2026-01-02T00:00:00Z"),
        (CAROL, "user", "2026-01-03T00:00:00Z"),
    ])
    return fd_db


@pytest.fixture
def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate the deck: tmp DATA_DIR + registry, no Docker, clean env."""
    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)

    def _no_docker():
        raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server, "get_docker", _no_docker)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("FD_INTERNAL_URL", raising=False)
    monkeypatch.setenv("FD_PORT", "25999")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")  # origin_guard's Host allowlist
    registry = {
        "alice-agent": {"slug": "alice-agent", "name": "alice-agent", "web_port": 24101,
                        "web_auth": "alice-agent-token", "owner": ALICE, "pid": None},
        "bob-agent": {"slug": "bob-agent", "name": "bob-agent", "web_port": 24102,
                      "web_auth": "bob-agent-token", "owner": BOB, "pid": None},
    }
    server._save_process_registry(registry)
    return data


def _http(client=LOOPBACK, headers: dict | None = None, uid: str = "") -> StarletteRequest:
    """A real Starlette Request (what the HTTP route receives)."""
    scope = {
        "type": "http", "method": "POST", "path": "/fd/spawn-process", "scheme": "http",
        "query_string": b"", "server": ("127.0.0.1", 25999), "client": client,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    req = StarletteRequest(scope)
    if uid:
        req.state.user_id = uid
    return req


def _stub(uid: str = ""):
    """The in-process stub Request Basna/Vatra/Dubina/flows/beings pass."""
    return types.SimpleNamespace(state=types.SimpleNamespace(user_id=uid))


def _cfg(**kw) -> server.AgentConfig:
    return server.AgentConfig(name=kw.pop("name", "child"), **kw)


# ── _resolve_spawn_owner ─────────────────────────────────────────────


class TestResolveSpawnOwner:
    async def test_verified_jwt_user_wins_over_body_hint(self, deck, users):
        cfg = _cfg(owner_hint=ALICE, web_auth_token="custom")
        owner = await server._resolve_spawn_owner(cfg, _http(uid=BOB))
        assert owner == BOB
        assert cfg.owner_hint == BOB            # normalised for _resolve_archetype
        assert cfg.web_auth_token == "custom"   # a signed-in user may pick one

    async def test_remote_unauthenticated_spawn_refused(self, deck, users):
        """The off-machine half of the audited hole: no JWT, not loopback, no
        secret → nothing is spawned, whatever owner the body names."""
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(owner_hint=ALICE), _http(client=REMOTE))
        assert e.value.status_code == 401

    async def test_agent_auth_resolves_the_calling_agents_owner(self, deck, users):
        cfg = _cfg(web_auth_token="attacker-picked")
        req = _http(headers={"X-Agent-Auth": "bob-agent-token"})
        assert await server._resolve_spawn_owner(cfg, req) == BOB
        # An unverified caller never chooses the child's identity token.
        assert cfg.web_auth_token == ""

    async def test_agent_auth_with_a_foreign_hint_is_refused(self, deck, users):
        req = _http(headers={"X-Agent-Auth": "bob-agent-token"})
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(owner_hint=ALICE), req)
        assert e.value.status_code == 403

    async def test_unknown_agent_auth_is_refused(self, deck, users):
        """E.g. a token from another deck on the same host — never a fallback."""
        req = _http(headers={"X-Agent-Auth": "some-other-decks-token"})
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), req)
        assert e.value.status_code == 403

    async def test_owner_hint_alone_never_attributes_a_spawn(self, deck, users):
        """The old tool shape (loopback + owner_hint only) proved nothing: user
        ids are listed to every user, so B could spawn an agent recorded as
        Alice's — with B's botport/env config — that then acts with Alice's
        Google. Without the calling agent's X-Agent-Auth it is refused, even for
        a real agent owner."""
        for hint in (BOB, ALICE, CAROL, "made-up-uuid"):
            with pytest.raises(HTTPException) as e:
                await server._resolve_spawn_owner(_cfg(owner_hint=hint), _http())
            assert e.value.status_code == 401

    async def test_agent_auth_whose_owner_is_not_a_user_here_is_refused(self, deck, users):
        """A registry owner that isn't a user of this deck (e.g. copied from
        another deck) doesn't count."""
        reg = server._load_process_registry()
        reg["ghost"] = {"slug": "ghost", "web_port": 24103, "web_auth": "g", "owner": "foreign"}
        server._save_process_registry(reg)
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), _http(headers={"X-Agent-Auth": "g"}))
        assert e.value.status_code == 403

    async def test_no_identity_in_multi_user_deck_is_401_not_an_arbitrary_owner(self, deck, users):
        """The old code inherited the FIRST registry owner (here Alice) — the
        agent then silently used Alice's Google account."""
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), _http())
        assert e.value.status_code == 401

    async def test_no_identity_in_single_user_deck_is_401_too(self, deck, fd_db):
        """No sole-user fallback for a real HTTP request: any web page open on
        the FD host can POST here (CORS '*'), and would otherwise get an agent
        acting with the owner's Google."""
        await _insert_users(fd_db, [(ALICE, "admin", "2026-01-01T00:00:00Z")])
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), _http())
        assert e.value.status_code == 401

    async def test_browser_origin_is_refused_even_with_an_agent_token(self, deck, users):
        """A page in a browser must sign in; only FD-spawned agents (no Origin)
        spawn on their own identity. 401 also sends an expired SPA to refresh."""
        req = _http(headers={"X-Agent-Auth": "bob-agent-token", "Origin": "http://evil.test"})
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), req)
        assert e.value.status_code == 401

    async def test_ownerless_agent_of_this_deck_belongs_to_the_sole_user(self, deck, fd_db):
        """E.g. spawned while auth was off: FD recorded no owner. The token is
        this deck's own, so on a single-user deck the owner is unambiguous."""
        await _insert_users(fd_db, [(ALICE, "admin", "2026-01-01T00:00:00Z")])
        reg = server._load_process_registry()
        reg["old"] = {"slug": "old", "web_port": 24104, "web_auth": "old-token", "owner": ""}
        server._save_process_registry(reg)
        assert await server._resolve_spawn_owner(
            _cfg(), _http(headers={"X-Agent-Auth": "old-token"})) == ALICE

    async def test_ownerless_agent_on_a_multi_user_deck_is_refused(self, deck, users):
        reg = server._load_process_registry()
        reg["old"] = {"slug": "old", "web_port": 24104, "web_auth": "old-token", "owner": ""}
        server._save_process_registry(reg)
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), _http(headers={"X-Agent-Auth": "old-token"}))
        assert e.value.status_code == 403

    async def test_synthetic_local_owner_counts_as_ownerless(self, deck, fd_db):
        """Recorded as 'local' while the deck ran with auth off (get_current_user
        stubs, _legacy_inherited_owner), its FD_OWNER_ID hint 'local' too: after
        auth is switched on, a single-user deck's sole user owns it — as its
        Google calls already resolve — rather than a 403."""
        await _insert_users(fd_db, [(ALICE, "admin", "2026-01-01T00:00:00Z")])
        reg = server._load_process_registry()
        reg["was-off"] = {"slug": "was-off", "web_port": 24105, "web_auth": "off-token",
                          "owner": "local"}
        server._save_process_registry(reg)
        req = _http(headers={"X-Agent-Auth": "off-token"})
        for hint in ("", "local", ALICE):
            cfg = _cfg(owner_hint=hint)
            assert await server._resolve_spawn_owner(cfg, req) == ALICE
            assert cfg.owner_hint == ALICE
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(owner_hint="someone-else"), req)
        assert e.value.status_code == 403

    async def test_synthetic_local_owner_on_a_multi_user_deck_is_refused(self, deck, users):
        reg = server._load_process_registry()
        reg["was-off"] = {"slug": "was-off", "web_port": 24105, "web_auth": "off-token",
                          "owner": "local"}
        server._save_process_registry(reg)
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(
                _cfg(owner_hint="local"), _http(headers={"X-Agent-Auth": "off-token"}))
        assert e.value.status_code == 403

    async def test_local_hint_does_not_override_a_real_recorded_owner(self, deck, users):
        req = _http(headers={"X-Agent-Auth": "bob-agent-token"})
        assert await server._resolve_spawn_owner(_cfg(owner_hint="local"), req) == BOB

    async def test_lockdown_requires_the_secret_even_from_loopback(self, deck, users, monkeypatch):
        monkeypatch.setenv("FD_LOCKDOWN", "1")
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(owner_hint=BOB), _http())
        assert e.value.status_code == 401
        ok = _http(headers={"X-Agent-Secret": "shh", "X-Agent-Auth": "bob-agent-token"})
        assert await server._resolve_spawn_owner(_cfg(), ok) == BOB

    async def test_in_process_stub_owner_is_authoritative(self, deck, users):
        assert await server._resolve_spawn_owner(_cfg(owner_hint=BOB), _stub(BOB)) == BOB
        # beings pass the owner as a hint on a stub; FD code set it, so it holds.
        assert await server._resolve_spawn_owner(_cfg(owner_hint=CAROL), _stub()) == CAROL

    async def test_in_process_stub_without_owner_never_borrows_a_tenant(self, deck, users):
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), _stub())
        assert e.value.status_code == 403

    async def test_auth_disabled_ignores_the_body_hint(self, deck, users, monkeypatch):
        """Single tenant: the calling agent's recorded owner when it proves one,
        else the first recorded owner — never the body's hint. A caller-chosen
        web_auth stays (the desktop Spawner can set one)."""
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        cfg = _cfg(owner_hint="whoever", web_auth_token="mine")
        assert await server._resolve_spawn_owner(cfg, _http(client=REMOTE)) == ALICE
        assert cfg.web_auth_token == "mine"
        assert await server._resolve_spawn_owner(_cfg(), _http()) == ALICE
        by_agent = _http(headers={"X-Agent-Auth": "bob-agent-token"})
        assert await server._resolve_spawn_owner(_cfg(owner_hint=ALICE), by_agent) == BOB
        # In-process callers are FD code: their hint still holds.
        assert await server._resolve_spawn_owner(_cfg(owner_hint=CAROL), _stub()) == CAROL


class TestWebAuthCollision:
    def test_detects_another_agents_token_but_not_its_own_slug(self, deck):
        assert server._web_auth_in_use("bob-agent-token", "new-agent")
        assert not server._web_auth_in_use("bob-agent-token", "bob-agent")
        assert not server._web_auth_in_use("fresh", "new-agent")
        assert not server._web_auth_in_use("", "new-agent")


# ── POST /fd/spawn-process end to end ────────────────────────────────


class _FakePopen:
    """Stands in for the agent process. Records the child env and simulates the
    child announcing a (drifted) port, which keeps the spawn path from
    scheduling its deferred drift re-check."""

    calls: list[dict] = []

    def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None, start_new_session=None):
        slug = Path(cwd).name
        _FakePopen.calls.append({"slug": slug, "env": dict(env or {})})
        self.pid = 424242
        reg = server._load_process_registry()
        if slug in reg:
            reg[slug]["web_port"] = int(reg[slug]["web_port"]) + 1
            server._save_process_registry(reg)
        if stdout is not None:
            stdout.close()

    def poll(self):
        return None


@pytest.fixture
def spawn_env(deck, monkeypatch):
    from captain_claw.flight_deck import rate_limiter

    _FakePopen.calls = []
    # Per-user spawn/API windows are process-global: start each test fresh.
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(server.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(server, "_is_port_available", lambda port: True)
    monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)
    monkeypatch.setenv("FD_SPAWN_SETTLE_S", "0")
    return deck


def _client(client_addr) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=client_addr),
        base_url="http://fd.test")


class TestSpawnProcessRoute:
    async def test_audited_forgery_no_longer_yields_a_known_web_auth(self, spawn_env, users):
        """Audit: a loopback caller POSTs {owner_hint: A, web_auth_token: 't'} and
        then uses X-Agent-Auth 't' to pull A's Google token. A hint alone is now
        refused outright; an agent spawning on its OWN identity gets a child of
        its own owner with a server-minted token, never the chosen one."""
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={
                "name": "forged", "owner_hint": ALICE, "web_auth_token": "t", "web_port": 24500})
            assert r.status_code == 401
            assert "forged" not in server._load_process_registry()
            r = await c.post("/fd/spawn-process", json={
                "name": "child", "owner_hint": BOB, "web_auth_token": "t", "web_port": 24500},
                headers={"X-Agent-Auth": "bob-agent-token"})
        assert r.status_code == 200, r.text
        entry = server._load_process_registry()["child"]
        assert entry["owner"] == BOB
        assert entry["web_auth"] and entry["web_auth"] != "t"
        assert server._resolve_agent_owner_by_auth("t") == ""
        assert server._resolve_agent_owner_by_auth(entry["web_auth"]) == BOB

    async def test_remote_forgery_refused_before_anything_is_written(self, spawn_env, users):
        async with _client(REMOTE) as c:
            r = await c.post("/fd/spawn-process", json={
                "name": "forged", "owner_hint": ALICE, "web_auth_token": "t"})
        assert r.status_code == 401
        assert "forged" not in server._load_process_registry()
        assert not (spawn_env / "forged").exists()
        assert _FakePopen.calls == []

    async def test_another_users_agent_name_is_refused(self, spawn_env, users):
        """alice-agent is stopped; re-spawning it would reuse its data dir. Bob
        (e.g. a kiosk account picking an archetype's default name) gets a 409,
        and nothing of Alice's is touched."""
        before = dict(server._load_process_registry()["alice-agent"])
        async with _client(REMOTE) as c:
            r = await c.post("/fd/spawn-process", json={"name": "Alice-Agent", "web_port": 24530},
                             headers={"Authorization": f"Bearer {create_access_token(BOB)}"})
        assert r.status_code == 409
        assert "already exists" in r.json()["detail"]
        assert server._load_process_registry()["alice-agent"] == before
        assert _FakePopen.calls == []

    async def test_own_stopped_agent_can_be_respawned(self, spawn_env, users):
        async with _client(REMOTE) as c:
            r = await c.post("/fd/spawn-process", json={"name": "alice-agent", "web_port": 24531},
                             headers={"Authorization": f"Bearer {create_access_token(ALICE)}"})
        assert r.status_code == 200, r.text
        assert server._load_process_registry()["alice-agent"]["owner"] == ALICE

    async def test_jwt_user_cannot_be_reassigned_by_hint(self, spawn_env, users):
        tok = create_access_token(BOB)
        async with _client(REMOTE) as c:
            r = await c.post("/fd/spawn-process", json={"name": "mine", "owner_hint": ALICE,
                                                        "web_port": 24510},
                             headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        assert server._load_process_registry()["mine"]["owner"] == BOB

    async def test_colliding_web_auth_gets_a_fresh_one(self, spawn_env, users):
        """A signed-in user requesting another agent's token (here Alice's) is
        not given it — nor refused: the agent gets a fresh one, returned."""
        tok = create_access_token(BOB)
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={
                "name": "copycat", "web_auth_token": "alice-agent-token", "web_port": 24520},
                headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        entry = server._load_process_registry()["copycat"]
        assert entry["web_auth"] and entry["web_auth"] != "alice-agent-token"
        assert r.json()["web_auth"] == entry["web_auth"]
        assert "already used by another agent" in r.json()["message"]
        assert "alice-agent-token" not in (spawn_env / "copycat" / "config.yaml").read_text()
        assert server._resolve_agent_owner_by_auth("alice-agent-token") == ALICE
        assert server._resolve_agent_owner_by_auth(entry["web_auth"]) == BOB

    async def test_spawner_preset_with_a_fixed_token_spawns_twice(self, spawn_env, users):
        """SpawnerPage presets carry the whole config, webAuthToken included.
        (Carol: owns no agent yet, so the free plan's agent limit allows two.)"""
        tok = create_access_token(CAROL)
        async with _client(LOOPBACK) as c:
            first = await c.post("/fd/spawn-process", json={
                "name": "preset-a", "web_auth_token": "preset-pw", "web_port": 24521},
                headers={"Authorization": f"Bearer {tok}"})
            second = await c.post("/fd/spawn-process", json={
                "name": "preset-b", "web_auth_token": "preset-pw", "web_port": 24523},
                headers={"Authorization": f"Bearer {tok}"})
        assert first.status_code == 200 and second.status_code == 200, second.text
        reg = server._load_process_registry()
        assert reg["preset-a"]["web_auth"] == "preset-pw"
        assert first.json()["web_auth"] == ""   # kept as requested: nothing to report
        assert reg["preset-b"]["web_auth"] not in ("", "preset-pw")
        assert second.json()["web_auth"] == reg["preset-b"]["web_auth"]

    async def test_child_env_pins_owner_and_this_decks_fd_url(self, spawn_env, users, monkeypatch):
        # Inherited from the shell / a CWD .env: another deck's URL + a global
        # VFS user + a Google-URL override. The caller's env_vars try too.
        monkeypatch.setenv("FD_URL", "http://localhost:25080")
        monkeypatch.setenv("CLAW_VFS_USER", ALICE)
        monkeypatch.setenv("CLAW_GOOGLE_OAUTH__FLIGHT_DECK_URL", "http://localhost:25080")
        tok = create_access_token(BOB)
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={
                "name": "pinned", "web_port": 24530,
                "env_vars": [{"key": "FD_URL", "value": "http://localhost:25080"}]},
                headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        env = _FakePopen.calls[-1]["env"]
        assert env["FD_URL"] == "http://localhost:25999"
        assert "CLAW_GOOGLE_OAUTH__FLIGHT_DECK_URL" not in env
        assert env["FD_OWNER_ID"] == BOB
        assert env["CLAW_VFS_USER"] == BOB

    async def test_fd_internal_url_is_the_explicit_override(self, spawn_env, users, monkeypatch):
        monkeypatch.setenv("FD_INTERNAL_URL", "http://127.0.0.1:26000/")
        tok = create_access_token(BOB)
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={"name": "internal", "web_port": 24540},
                             headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        assert _FakePopen.calls[-1]["env"]["FD_URL"] == "http://127.0.0.1:26000"

    async def test_docker_spawn_refused_before_docker_is_touched(self, spawn_env, users):
        """/fd/spawn resolves the owner first — an unattributable caller can't
        remove a stopped container by name or create one."""
        async with _client(REMOTE) as c:
            r = await c.post("/fd/spawn", json={"name": "x", "owner_hint": ALICE})
        assert r.status_code == 401


# ── restart path ─────────────────────────────────────────────────────


class TestRestartRepinsOwnerAndFdUrl:
    def test_restart_pins_owner_and_overrides_a_stale_fd_url(self, spawn_env, monkeypatch):
        agent_dir = spawn_env / "bob-agent"
        agent_dir.mkdir()
        # The agent's own .env froze another port's URL at spawn time.
        (agent_dir / ".env").write_text("FD_URL=http://localhost:25080\nFOO=bar\n")
        monkeypatch.setenv("CLAW_VFS_USER", ALICE)   # lifespan's primary-owner bind
        monkeypatch.delenv("FD_OWNER_ID", raising=False)
        entry = server._load_process_registry()["bob-agent"]
        assert server._start_registered_process("bob-agent", entry)
        env = _FakePopen.calls[-1]["env"]
        assert env["FD_URL"] == "http://localhost:25999"
        assert env["FD_OWNER_ID"] == BOB
        assert env["CLAW_VFS_USER"] == BOB
        assert env["FOO"] == "bar"


# ── clone ────────────────────────────────────────────────────────────


class TestClone:
    async def test_clone_keeps_owner_and_gets_its_own_web_auth(self, deck, monkeypatch):
        src = deck / "bob-agent"
        (src / "data" / "home-config").mkdir(parents=True)
        cfg_text = "web:\n  port: 24102\n  auth_token: bob-agent-token\n"
        (src / "config.yaml").write_text(cfg_text)
        (src / "data" / "home-config" / "config.yaml").write_text(cfg_text)
        reg = server._load_process_registry()
        reg["bob-agent"].update({"grid_tags": ["domain:legal"], "grid_recall": "domain",
                                 "tier": "reason"})
        server._save_process_registry(reg)
        monkeypatch.setattr(server, "_find_available_port", lambda start: 24777)

        res = await server.clone_process(
            "bob-agent", server.CloneRequest(new_name="bob agent 2"), _stub(BOB), None)
        assert res.ok
        clone = server._load_process_registry()["bob-agent-2"]
        assert clone["owner"] == BOB
        assert clone["grid_tags"] == ["domain:legal"] and clone["grid_recall"] == "domain"
        assert clone["web_auth"] and clone["web_auth"] != "bob-agent-token"
        # Each token resolves to exactly one agent — and to the right owner.
        assert server._resolve_agent_owner_by_auth(clone["web_auth"]) == BOB
        for p in (deck / "bob-agent-2" / "config.yaml",
                  deck / "bob-agent-2" / "data" / "home-config" / "config.yaml"):
            text = p.read_text()
            assert clone["web_auth"] in text and "bob-agent-token" not in text
            assert "port: 24777" in text

    async def test_clone_survives_removal_of_its_source(self, deck, monkeypatch):
        (deck / "bob-agent").mkdir()
        monkeypatch.setattr(server, "_find_available_port", lambda start: 24778)
        await server.clone_process(
            "bob-agent", server.CloneRequest(new_name="solo"), _stub(BOB), None)
        reg = server._load_process_registry()
        reg.pop("bob-agent")
        server._save_process_registry(reg)
        assert server._resolve_agent_owner_by_auth(reg["solo"]["web_auth"]) == BOB


# ── primary owner ────────────────────────────────────────────────────


class TestPrimaryOwner:
    async def test_oldest_admin_even_past_the_newest_100_users(self, fd_db):
        rows = [("boss", "admin", "2025-01-01T00:00:00Z")]
        rows += [(f"u{i:03d}", "user", f"2026-02-01T00:{i // 60:02d}:{i % 60:02d}Z")
                 for i in range(150)]
        await _insert_users(fd_db, rows)
        assert await server._resolve_primary_owner(fd_db) == "boss"

    @pytest.mark.parametrize("total,admins,expected", [
        # Only admin sits in the second-oldest page (walk continues backwards).
        (450, [300], 300),
        # Two admins in one page: the older one, not the first one seen.
        (450, [120, 180], 120),
        # Last step overlaps pages already scanned (total not a page multiple).
        (250, [220, 240], 220),
    ])
    async def test_oldest_admin_across_pages(self, fd_db, total, admins, expected):
        rows = [(f"u{i:03d}", "user", f"2026-02-01T{i // 60:02d}:{i % 60:02d}:00Z")
                for i in range(total)]
        for i in admins:
            rows[i] = (f"admin{i:03d}", "admin", rows[i][2])
        await _insert_users(fd_db, rows)
        assert await server._resolve_primary_owner(fd_db) == f"admin{expected:03d}"

    async def test_no_admin_falls_back_to_the_oldest_user(self, fd_db):
        rows = [(f"u{i:03d}", "user", f"2026-02-01T00:{i // 60:02d}:{i % 60:02d}Z")
                for i in range(130)]
        await _insert_users(fd_db, rows)
        assert await server._resolve_primary_owner(fd_db) == "u000"

    async def test_single_user_and_empty(self, fd_db):
        assert await server._resolve_primary_owner(fd_db) == ""
        await _insert_users(fd_db, [("only", "user", "2026-01-01T00:00:00Z")])
        assert await server._resolve_primary_owner(fd_db) == "only"


# ── this deck's own URL / port ───────────────────────────────────────


class TestSelfUrl:
    def test_self_url_prefers_fd_internal_url_then_fd_port(self, monkeypatch):
        monkeypatch.delenv("FD_INTERNAL_URL", raising=False)
        monkeypatch.setenv("FD_PORT", "25181")
        assert server._fd_self_url() == "http://localhost:25181"
        monkeypatch.setenv("FD_INTERNAL_URL", "http://10.0.0.5:25181/")
        assert server._fd_self_url() == "http://10.0.0.5:25181"

    def test_main_records_the_bound_port_over_an_inherited_one(self, monkeypatch):
        import uvicorn
        seen: dict = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
        monkeypatch.setattr(sys, "argv", ["fd", "--port", "25181", "--dev"])
        monkeypatch.setenv("FD_PORT", "25080")   # leaked from another deck's shell
        server.main()
        assert seen["port"] == 25181
        assert server.os.environ["FD_PORT"] == "25181"


# ── settings DB: auth-enabled decks only (as on main) ────────────────


class TestSettingsDbOnlyWithAuth:
    async def test_auth_off_deck_opens_the_db_but_refuses_register(self, deck, monkeypatch):
        """Auth-disabled decks open their DB for connector settings (origin_guard
        keeps web pages off the API), but /fd/auth/register refuses there — no
        one can plant the deck's first admin, who would later become primary
        owner (legacy Google tokens included) once auth is switched on."""
        monkeypatch.setattr(server, "AUTH_ENABLED", False)
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        prev = fd_auth._db
        try:
            db = await server._init_fd_db()
            try:
                assert db is not None and (deck / "flight-deck.db").exists()
                async with _client(LOOPBACK) as c:
                    r = await c.post("/fd/auth/register",
                                     json={"email": "evil@x.co", "password": "secret123"})
                assert r.status_code == 403
                assert "access_token" not in r.text
                assert await db.list_users(limit=10) == []
            finally:
                await db.close()
        finally:
            fd_auth._db = prev

    async def test_auth_on_deck_opens_and_wires_the_db(self, deck):
        prev = fd_auth._db
        try:
            db = await server._init_fd_db()
            try:
                assert db is not None and fd_auth.get_db() is db
                assert (deck / "flight-deck.db").is_file()   # per-deck file
            finally:
                await db.close()
        finally:
            fd_auth._db = prev


# ── deck scoping of Docker labels ────────────────────────────────────


class _FakeContainer:
    def __init__(self, name: str, labels: dict, status: str = "running"):
        self.name = name
        self.id = f"id-{name}"
        self.short_id = name[:12]
        self.labels = dict(labels)
        self.status = status
        self.attrs = {"NetworkSettings": {"Ports": {}}, "Created": ""}
        self.image = types.SimpleNamespace(tags=["img:latest"], short_id="img")
        self.removed = False

    def remove(self, force: bool = False) -> None:
        self.removed = True


class _FakeContainers:
    """The slice of docker-py's container API FD's lookups use."""

    def __init__(self, items: list[_FakeContainer]):
        self.items = list(items)
        self.run_calls: list[dict] = []

    def list(self, all: bool = False, filters: dict | None = None):
        label = (filters or {}).get("label")
        return [c for c in self.items
                if not c.removed and (all or c.status == "running")
                and (not label or label in c.labels)]

    def get(self, name: str):
        for c in self.items:
            if c.name == name and not c.removed:
                return c
        raise docker.errors.NotFound("no such container")

    def run(self, **kw):
        self.run_calls.append(kw)
        c = _FakeContainer(kw["name"], kw.get("labels") or {})
        self.items.append(c)
        return c


def _labels(name: str, owner: str, token: str, *, deck: str | None, port: str) -> dict:
    lb = {server.CONTAINER_LABEL: "true", server.OWNER_LABEL: owner,
          "flight-deck.web-auth": token, "flight-deck.agent-name": name,
          "flight-deck.web-port": port}
    if deck is not None:
        lb[server.DECK_LABEL] = deck
    return lb


OTHER_DECK = "0123456789abcdef"


@pytest.fixture
def docker_host(deck, monkeypatch):
    """A Docker daemon this deck shares with another deck on the host. The
    other deck's container comes FIRST and names Alice as its owner with a
    token of its choosing — the review's scenario: an auth-disabled deck Y
    honours any owner_hint, so its labels are attacker-chosen."""
    fake = types.SimpleNamespace(containers=_FakeContainers([
        _FakeContainer("foreign", _labels("foreign", ALICE, "forged", deck=OTHER_DECK, port="24203")),
        _FakeContainer("legacy", _labels("legacy", CAROL, "legacy-token", deck=None, port="24202")),
        _FakeContainer("mine", _labels("mine", BOB, "mine-token", deck=server._deck_id(), port="24201")),
    ]))
    monkeypatch.setattr(server, "get_docker", lambda: fake)
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: False)
    return fake


def _container(host, name: str) -> _FakeContainer:
    return next(c for c in host.containers.items if c.name == name)


class TestDeckScopedIdentity:
    def test_deck_id_is_stable_and_per_data_dir(self, deck, monkeypatch, tmp_path):
        mine = server._deck_id()
        assert mine == server._deck_id() and len(mine) == 16
        monkeypatch.setattr(server, "DATA_DIR", tmp_path / "another-deck")
        assert server._deck_id() != mine

    def test_identity_contract(self, docker_host):
        ident = server._resolve_agent_identity_by_auth
        assert ident("alice-agent-token") == (True, ALICE)   # this deck's registry
        assert ident("mine-token") == (True, BOB)            # this deck's label
        assert ident("legacy-token") == (True, CAROL)        # unlabelled: legacy, as before
        assert ident("forged") == (False, "")                # another deck's: ignored
        assert ident("unknown") == (False, "")
        assert ident("") == (False, "")
        # The legacy str API is a thin wrapper.
        assert server._resolve_agent_owner_by_auth("forged") == ""
        assert server._resolve_agent_owner_by_auth("mine-token") == BOB

    def test_ownerless_match_is_told_apart_from_an_unknown_token(self, docker_host):
        reg = server._load_process_registry()
        reg["old"] = {"slug": "old", "web_port": 24104, "web_auth": "old-token", "owner": ""}
        server._save_process_registry(reg)
        assert server._resolve_agent_identity_by_auth("old-token") == (True, "")
        assert server._resolve_agent_owner_by_auth("old-token") == ""

    def test_a_recorded_owner_wins_over_a_legacy_ownerless_duplicate(self, docker_host):
        """Old clones copied their source's token and recorded no owner."""
        reg = server._load_process_registry()
        server._save_process_registry({
            "old-clone": {"slug": "old-clone", "web_port": 24105,
                          "web_auth": "bob-agent-token", "owner": ""},
            **reg,
        })
        assert server._resolve_agent_identity_by_auth("bob-agent-token") == (True, BOB)

    async def test_another_decks_forged_label_cannot_spawn_here(self, docker_host, users):
        with pytest.raises(HTTPException) as e:
            await server._resolve_spawn_owner(_cfg(), _http(headers={"X-Agent-Auth": "forged"}))
        assert e.value.status_code == 403
        assert await server._resolve_spawn_owner(
            _cfg(), _http(headers={"X-Agent-Auth": "mine-token"})) == BOB

    def test_port_and_grid_lookups_ignore_other_decks(self, docker_host):
        assert server._resolve_agent_owner(24203) == ""
        assert server._resolve_agent_auth(24203) == ""
        assert server._resolve_agent_grid_by_auth("forged") == ([], "")
        assert server._resolve_agent_owner(24201) == BOB
        assert server._resolve_agent_auth(24201) == "mine-token"
        assert server._resolve_agent_auth(24202) == "legacy-token"

    def test_collision_check_and_legacy_owner_ignore_other_decks(self, docker_host):
        assert server._web_auth_in_use("mine-token", "new-agent")
        assert not server._web_auth_in_use("forged", "new-agent")
        server._save_process_registry({})
        # Auth-disabled inheritance: first recorded owner — never the foreign one
        # that is listed first.
        assert server._legacy_inherited_owner(include_docker=True) == CAROL

    async def test_container_list_and_actions_ignore_other_decks(self, docker_host, monkeypatch):
        # Before: Alice's list showed the foreign container — web_auth included.
        assert await server.list_containers(_http(uid=ALICE), None) == []
        monkeypatch.setattr(server, "AUTH_ENABLED", False)
        names = {i.name for i in await server.list_containers(_http(), None)}
        assert names == {"mine", "legacy"}
        with pytest.raises(HTTPException) as e:
            server._find_container("foreign")
        assert e.value.status_code == 404
        assert server._find_container("mine").name == "mine"

    async def test_fleet_ignores_other_decks(self, docker_host, monkeypatch):
        monkeypatch.setattr(server, "AUTH_ENABLED", False)
        names = {a.name for a in await server.get_fleet(_http(), None)}
        assert "foreign" not in names
        assert {"mine", "legacy", "alice-agent", "bob-agent"} <= names

    def test_glasses_default_agent_is_never_another_decks(self, docker_host):
        """It used to pick the FIRST managed container host-wide — here the
        foreign one, handing its web_auth to this deck's glasses channel."""
        from captain_claw.flight_deck import glasses_bridge

        assert glasses_bridge._resolve_default_agent() == ("localhost", 24202, "legacy-token")

    def test_inbound_mcp_fleet_ignores_other_decks(self, docker_host):
        """The foreign container names Alice as its owner."""
        from captain_claw.flight_deck import mcp_server_routes

        names = {a["name"] for a in mcp_server_routes._user_agents(ALICE)}
        assert names == {"alice-agent"}
        assert {a["name"] for a in mcp_server_routes._user_agents(BOB)} == {"mine", "bob-agent"}

    def test_project_fleet_ports_ignore_other_decks(self, docker_host):
        from captain_claw.flight_deck import project_routes

        ports = project_routes._resolve_fleet_ports()
        assert "foreign" not in ports
        assert ports["mine"] == 24201 and ports["legacy"] == 24202


class TestDockerSpawnDeckLabel:
    @pytest.fixture
    def docker_spawn(self, docker_host, monkeypatch):
        monkeypatch.setattr(server, "_is_port_available", lambda port: True)
        monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)

        async def _sys_cfg():
            return {"docker_spawn_enabled": True}

        monkeypatch.setattr(server, "_get_system_config", _sys_cfg)
        return docker_host

    async def test_spawn_stamps_this_decks_label_and_mints_an_identity(self, docker_spawn, users):
        res = await server.spawn_agent(_cfg(name="boxed", web_enabled=False), _stub(BOB), None)
        assert res.ok
        labels = docker_spawn.containers.run_calls[-1]["labels"]
        assert labels[server.DECK_LABEL] == server._deck_id()
        assert labels[server.OWNER_LABEL] == BOB
        token = labels["flight-deck.web-auth"]
        assert token   # minted even with the web UI off: it is the agent's identity
        assert server._resolve_agent_identity_by_auth(token) == (True, BOB)

    async def test_spawn_never_removes_another_decks_container(self, docker_spawn, users):
        foreign = _container(docker_spawn, "foreign")
        foreign.status = "exited"   # a stopped same-name container used to be removed
        with pytest.raises(HTTPException) as e:
            await server.spawn_agent(_cfg(name="foreign"), _stub(BOB), None)
        assert e.value.status_code == 409
        assert not foreign.removed
        assert docker_spawn.containers.run_calls == []

    async def test_colliding_web_auth_gets_a_fresh_one(self, docker_spawn, users):
        res = await server.spawn_agent(
            _cfg(name="copy", web_auth_token="mine-token"), _stub(BOB), None)
        assert res.ok
        token = docker_spawn.containers.run_calls[-1]["labels"]["flight-deck.web-auth"]
        assert token not in ("", "mine-token")
        assert res.web_auth == token and "already used by another agent" in res.message
        assert server._resolve_agent_identity_by_auth("mine-token") == (True, BOB)
        # A free requested token is kept, and not echoed back.
        res = await server.spawn_agent(
            _cfg(name="fixed", web_auth_token="free-token"), _stub(BOB), None)
        assert docker_spawn.containers.run_calls[-1]["labels"]["flight-deck.web-auth"] == "free-token"
        assert res.web_auth == ""


class TestDockerClone:
    async def test_clone_gets_its_own_web_auth(self, docker_host, monkeypatch):
        """Two containers sharing a web_auth are one identity to FD. The clone's
        token is set where the container reads it at start: the label FD checks,
        its own (host-side, bind-mounted) config copies, and its env."""
        src = docker_host.containers.items[2]   # "mine", Bob's
        assert src.name == "mine"
        src.attrs.update({"Config": {"Env": ["CLAW_WEB__AUTH_TOKEN=mine-token", "KEEP=1"]},
                          "Mounts": [], "HostConfig": {}})
        cfg_text = "web:\n  port: 24201\n  auth_token: mine-token\n"
        (server.DATA_DIR / "mine" / "data" / "home-config").mkdir(parents=True)
        (server.DATA_DIR / "mine" / "config.yaml").write_text(cfg_text)
        (server.DATA_DIR / "mine" / "data" / "home-config" / "config.yaml").write_text(cfg_text)
        monkeypatch.setattr(server, "_find_available_port", lambda start: 24290)

        res = await server.clone_container("mine", server.CloneRequest(new_name="mine two"),
                                           _stub(BOB), None)
        assert res.ok
        run = docker_host.containers.run_calls[-1]
        token = run["labels"]["flight-deck.web-auth"]
        assert token not in ("", "mine-token")
        assert run["labels"][server.OWNER_LABEL] == BOB
        assert run["environment"] == {"CLAW_WEB__AUTH_TOKEN": token, "KEEP": "1"}
        new_dir = server.DATA_DIR / "mine-two"
        for p in (new_dir / "config.yaml", new_dir / "data" / "home-config" / "config.yaml"):
            text = p.read_text()
            assert token in text and "mine-token" not in text
        # The source keeps its own.
        assert "mine-token" in (server.DATA_DIR / "mine" / "config.yaml").read_text()
        assert server._resolve_agent_identity_by_auth("mine-token") == (True, BOB)


# ── the agent's environment ──────────────────────────────────────────


class TestAgentEnvironment:
    async def test_spawn_env_drops_fd_only_secrets(self, spawn_env, users, monkeypatch):
        tok = create_access_token(BOB)   # before FD_JWT_SECRET is set: never cache ours
        monkeypatch.setenv("FD_JWT_SECRET", "deck-jwt-secret")
        monkeypatch.setenv("FD_EVENTS_WEBHOOK_TOKEN", "hook-token")
        monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "OPERATOR-ACCESS")
        monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "/operator/creds.json")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/operator/adc.json")
        monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
        monkeypatch.setenv("FD_DATA_DIR", "./fd-data")   # relative, as an operator may set it
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={
                "name": "clean", "web_port": 24550,
                # A credential the user hands THEIR agent deliberately stays.
                "env_vars": [{"key": "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE",
                              "value": "/bob/own.json"}]},
                headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        env = _FakePopen.calls[-1]["env"]
        for gone in ("FD_JWT_SECRET", "FD_EVENTS_WEBHOOK_TOKEN", "GOOGLE_WORKSPACE_CLI_TOKEN"):
            assert gone not in env, gone
        assert env["GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"] == "/bob/own.json"
        assert env["FD_AGENT_SHARED_SECRET"] == "shh"          # agents send it
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/operator/adc.json"  # left, reported
        # Absolute: the agent's cwd is its own dir, where ./fd-data is elsewhere.
        assert env["FD_DATA_DIR"] == str(spawn_env)

    def test_restart_env_drops_fd_only_secrets_too(self, spawn_env, monkeypatch):
        (spawn_env / "bob-agent").mkdir()
        monkeypatch.setenv("FD_JWT_SECRET", "deck-jwt-secret")
        monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "OPERATOR-ACCESS")
        monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "/operator/creds.json")
        monkeypatch.setenv("SOME_UNRELATED_VAR", "kept")
        entry = server._load_process_registry()["bob-agent"]
        assert server._start_registered_process("bob-agent", entry)
        env = _FakePopen.calls[-1]["env"]
        for gone in ("FD_JWT_SECRET", "GOOGLE_WORKSPACE_CLI_TOKEN",
                     "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"):
            assert gone not in env, gone
        assert env["SOME_UNRELATED_VAR"] == "kept"

    async def test_web_disabled_agent_still_gets_an_identity(self, spawn_env, users):
        """Without a web_auth an agent can never prove who it is (X-Agent-Auth):
        no owner's Google, no spawning children, and its always-running
        captain-claw-web port is unauthenticated."""
        tok = create_access_token(BOB)
        async with _client(LOOPBACK) as c:
            r = await c.post("/fd/spawn-process", json={"name": "quiet", "web_enabled": False},
                             headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        entry = server._load_process_registry()["quiet"]
        assert entry["web_auth"]
        assert server._resolve_agent_identity_by_auth(entry["web_auth"]) == (True, BOB)
        assert entry["web_auth"] in (spawn_env / "quiet" / "config.yaml").read_text()


class TestNonAgentSubprocessEnv:
    """Hosted VFS apps (a tenant's start command, shell=True) and stdio MCP
    servers used to get os.environ whole — FD_JWT_SECRET included, enough to
    mint a session for any user of the deck."""

    @pytest.fixture
    def fd_secrets(self, monkeypatch):
        monkeypatch.setenv("FD_JWT_SECRET", "deck-jwt-secret")
        monkeypatch.setenv("FD_EVENTS_WEBHOOK_TOKEN", "hook-token")
        monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_TOKEN", "OPERATOR-ACCESS")
        monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "/operator/creds.json")
        monkeypatch.setenv("SOME_UNRELATED_VAR", "kept")

    def _assert_scrubbed(self, env: dict) -> None:
        for gone in server._FD_ONLY_ENV_VARS:
            assert gone not in env, gone
        assert env["SOME_UNRELATED_VAR"] == "kept"

    def test_hosted_vfs_app(self, deck, fd_secrets, monkeypatch, tmp_path):
        from captain_claw.flight_deck import vfs_hosting as vh

        seen: dict = {}

        class _Popen:
            pid = 31337

            def __init__(self, cmd, shell=None, cwd=None, env=None, **kw):
                seen["env"] = dict(env or {})

            def poll(self):
                return None

        monkeypatch.setattr(vh, "load_registry",
                            lambda: {"site": {"kind": "app", "start_cmd": "npm start"}})
        monkeypatch.setattr(vh, "save_registry", lambda reg: None)
        monkeypatch.setattr(vh, "entry_dir", lambda ent: tmp_path)
        monkeypatch.setattr(vh, "app_is_alive", lambda name: False)
        monkeypatch.setattr(vh, "_procs", {})
        monkeypatch.setattr(vh.subprocess, "Popen", _Popen)
        monkeypatch.setattr(server, "_find_available_port", lambda start: 26123)
        ok, _msg = vh.start_app("site")
        assert ok
        self._assert_scrubbed(seen["env"])
        assert seen["env"]["PORT"] == "26123"

    async def test_stdio_mcp_server(self, fd_secrets, monkeypatch):
        from captain_claw.flight_deck import mcp_transport

        seen: dict = {}

        async def _exec(*args, env=None, **kw):
            seen["env"] = dict(env or {})
            raise RuntimeError("not really spawning")

        monkeypatch.setattr(mcp_transport.asyncio, "create_subprocess_exec", _exec)
        t = mcp_transport.StdioTransport({"command": "some-mcp-server",
                                          "env": {"FD_JWT_SECRET": "admin-chose-this"}})
        with pytest.raises(mcp_transport.MCPTransportError):
            await t._ensure_proc()
        env = seen["env"]
        assert env["SOME_UNRELATED_VAR"] == "kept"
        for gone in ("FD_EVENTS_WEBHOOK_TOKEN", "GOOGLE_WORKSPACE_CLI_TOKEN",
                     "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"):
            assert gone not in env, gone
        # The server's own configured env still wins (an admin's deliberate choice).
        assert env["FD_JWT_SECRET"] == "admin-chose-this"


# ── the agent-side flight_deck tool, through ASGI into FD ────────────


class TestFlightDeckToolSpawn:
    """Old Man / any agent with the ``flight_deck`` tool on a team deck
    (FD_LOCKDOWN=1): the tool's own request must spawn a child of ITS owner."""

    @pytest.fixture
    def agent(self, spawn_env, monkeypatch, tmp_path):
        from captain_claw import config as config_mod
        from captain_claw.config import Config
        from captain_claw.flight_deck import agent_secret

        cfg = Config()
        cfg.web.auth_token = "bob-agent-token"   # this agent IS bob-agent
        monkeypatch.setattr(config_mod, "_config", cfg)
        # FD and the agent share the per-deck secret FILE (no env override).
        monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
        monkeypatch.setenv("FD_LOCKDOWN", "1")
        monkeypatch.setenv("FD_OWNER_ID", BOB)
        # The deck FD pinned at spawn: only it gets the agent's identity.
        monkeypatch.setenv("FD_URL", "http://fd.test")
        agent_secret.reset_cache_for_tests()
        real_client = httpx.AsyncClient

        def _to_fd(*args, **kwargs):
            return real_client(transport=httpx.ASGITransport(app=server.app, client=LOOPBACK),
                               timeout=30.0)

        monkeypatch.setattr(httpx, "AsyncClient", _to_fd)
        yield cfg
        agent_secret.reset_cache_for_tests()

    async def test_tool_spawn_resolves_to_the_calling_agents_owner(self, agent, users):
        res = await FlightDeckTool()._spawn_agent("http://fd.test", "kid")
        assert res.success, res.error
        entry = server._load_process_registry()["kid"]
        assert entry["owner"] == BOB
        assert entry["web_auth"] and entry["web_auth"] != "bob-agent-token"

    async def test_tool_spawn_without_its_identity_is_refused(self, agent, users):
        agent.web.auth_token = ""
        res = await FlightDeckTool()._spawn_agent("http://fd.test", "kid")
        assert not res.success
        assert "401" in res.error
        assert "kid" not in server._load_process_registry()
        assert _FakePopen.calls == []

    async def test_lockdown_still_needs_the_right_deck_secret(self, agent, users):
        """The configured google_oauth.flight_deck_secret outranks the deck
        file (as google_oauth_manager sends it); a wrong one is refused."""
        agent.google_oauth.flight_deck_secret = "not-this-decks-secret"
        res = await FlightDeckTool()._spawn_agent("http://fd.test", "kid")
        assert not res.success
        assert "kid" not in server._load_process_registry()

    async def test_tool_spawn_via_a_websocket_supplied_url_carries_no_identity(self, agent, users):
        """A session fd_url that isn't the pinned deck gets no X-Agent-Auth /
        X-Agent-Secret — so FD (here the same app) refuses the spawn (401 from
        the agent gate, or 403 from origin_guard's Host allowlist first)."""
        tool = FlightDeckTool()
        res = await tool._spawn_agent("http://evil.test", "kid")
        assert not res.success and ("401" in res.error or "403" in res.error)
        assert "kid" not in server._load_process_registry()
