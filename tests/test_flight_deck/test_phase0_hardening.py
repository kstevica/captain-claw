"""Phase 0 hardening + read endpoints (docs/research-desk-product-plan.md).

Covers, all Captain-generic and default-off:

* ``/fd/{basna,vatra}/agent/*`` now require loopback or ``X-Agent-Secret``
  (previously reachable from anywhere with an attacker-chosen ``owner_id``).
* ``FD_LOCKDOWN=1`` makes the secret mandatory even from loopback and
  disables the host-filesystem surfaces (``/fd/vfs/browse-fs``, ``POST
  /fd/vfs/links``, ``/fd/projects/*``).
* ``GET /fd/basna/sessions/{id}/facts`` — the run's facts ledger as JSON.
* ``GET /fd/costs`` — the previously write-only ``cost_ledger``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import facts_ledger
from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import create_access_token, get_current_user, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)
REMOTE = ("203.0.113.9", 40001)
_LOCKDOWN_DETAIL = "disabled by FD_LOCKDOWN"


def _client(app, client_addr) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=client_addr),
        base_url="http://fd.test")


@pytest.fixture
async def fd_db(tmp_path: Path):
    """A real FlightDeckDB wired into the auth module (restored afterwards)."""
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


# ── 0a: agent-route guard ────────────────────────────────────────────


async def test_agent_route_remote_denied_without_secret(fd_db, monkeypatch):
    """The core fix: a remote caller can no longer act as an arbitrary owner."""
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/basna/agent/sessions", json={"owner_id": "victim"})
    assert r.status_code == 403
    assert "X-Agent-Secret" in r.json()["detail"]


async def test_agent_route_loopback_unchanged(fd_db, monkeypatch):
    """Single-user local mode: a loopback agent's owner_id hint still resolves
    (the transport guard lets loopback through unchanged)."""
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(fd_server.app, LOOPBACK) as c:
        r = await c.post("/fd/basna/agent/sessions", json={"owner_id": "u-loop"})
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


async def test_agent_route_secret_authorizes_remote(fd_db, monkeypatch):
    """The shared secret lets a remote caller past the transport guard; in
    single-user mode its owner_id hint then resolves. A wrong secret is blocked
    at the guard."""
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(fd_server.app, REMOTE) as c:
        ok = await c.post("/fd/basna/agent/sessions", json={"owner_id": "u-r"},
                          headers={"X-Agent-Secret": "shh"})
        bad = await c.post("/fd/basna/agent/sessions", json={"owner_id": "u-r"},
                           headers={"X-Agent-Secret": "nope"})
    assert ok.status_code == 200
    assert bad.status_code == 403


async def test_agent_route_secret_holder_cannot_forge_owner_in_multitenant(fd_db, monkeypatch):
    """Residual closed: in a multi-tenant (auth-on) deployment a remote
    shared-secret holder passes the transport guard but can NOT act as an
    arbitrary owner_id — the owner comes from FD's spawn records, not the body."""
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/basna/agent/sessions", json={"owner_id": "victim"},
                         headers={"X-Agent-Secret": "shh"})
    assert r.status_code == 403
    assert "could not resolve" in r.json()["detail"]


async def test_agent_route_loopback_cannot_forge_owner_in_multitenant(fd_db, monkeypatch):
    """A loopback caller (e.g. one FD-spawned agent) likewise cannot claim
    another owner_id when auth is enabled — closes the loopback path too."""
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, LOOPBACK) as c:
        r = await c.post("/fd/basna/agent/sessions", json={"owner_id": "victim"})
    assert r.status_code == 403
    assert "could not resolve" in r.json()["detail"]


async def test_vatra_agent_route_owner_forgery_also_closed(fd_db, monkeypatch):
    """Vatra shares basna's _resolve_owner, so the same protection applies to
    /fd/vatra/agent/*."""
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/vatra/agent/blackboard", json={"owner_id": "victim"},
                         headers={"X-Agent-Secret": "shh"})
    assert r.status_code == 403
    assert "could not resolve" in r.json()["detail"]


async def test_agent_route_vatra_prefix_also_guarded(fd_db, monkeypatch):
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/vatra/agent/blackboard", json={"owner_id": "victim"})
    assert r.status_code == 403


async def test_lockdown_requires_secret_even_from_loopback(fd_db, monkeypatch):
    """A same-host TLS proxy must not launder remote callers into 'loopback'."""
    monkeypatch.setenv("FD_LOCKDOWN", "1")
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(fd_server.app, LOOPBACK) as c:
        denied = await c.post("/fd/basna/agent/sessions", json={"owner_id": "x"})
        allowed = await c.post("/fd/basna/agent/sessions", json={"owner_id": "x"},
                               headers={"X-Agent-Secret": "shh"})
    assert denied.status_code == 403
    assert allowed.status_code == 200


# ── 0a: code / hosting agent guard — verified bearer OR loopback/secret ──
#
# The /fd/code/agent/* and /fd/hosting/agent/* routes run real shell/git or
# host apps as the resolved owner, so they carry the same transport guard as
# basna/vatra — but they ALSO accept a verified bearer (e.g. Captain Spark),
# which the handler binds to the caller's own identity (a mismatched owner_id
# → 403). Without a bearer they still require loopback or the shared secret,
# closing the unauthenticated owner_id/source_port impersonation vector.


async def test_code_hosting_agent_remote_denied_without_bearer_or_secret(fd_db, monkeypatch):
    """A remote caller with no bearer and no secret can no longer act as an
    arbitrary owner on the code/hosting agent routes."""
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        for path in ("/fd/code/agent/list", "/fd/hosting/agent/list"):
            r = await c.post(path, json={"owner_id": "victim"})
            assert r.status_code == 403, path
            assert "bearer token" in r.json()["detail"], path


async def test_code_agent_remote_allowed_with_bearer(fd_db, monkeypatch):
    """Captain Spark's service-account bearer passes the transport guard and
    the handler resolves the caller to its own verified identity."""
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    u = await fd_db.create_user("spark@svc.local", "x")
    tok = create_access_token(u["id"])
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/code/agent/list", json={},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert "projects" in r.json()


async def test_code_agent_bearer_cannot_impersonate_other_owner(fd_db, monkeypatch):
    """A valid bearer + a mismatched owner_id is rejected — the caller may act
    only as itself (the core of the code_routes fix)."""
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    u = await fd_db.create_user("spark2@svc.local", "x")
    tok = create_access_token(u["id"])
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/code/agent/list", json={"owner_id": "someone-else"},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    assert "does not match" in r.json()["detail"]


async def test_hosting_agent_bearer_cannot_impersonate_other_owner(fd_db, monkeypatch):
    """Hosting now benefits from the same bearer authority (previously the
    resolver was called without auth_user, so this branch never ran)."""
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    u = await fd_db.create_user("spark3@svc.local", "x")
    tok = create_access_token(u["id"])
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/hosting/agent/list", json={"owner_id": "someone-else"},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    assert "does not match" in r.json()["detail"]


async def test_code_agent_loopback_unchanged(fd_db, monkeypatch):
    """Locally spawned agents (loopback, owner_id hint, no bearer) still work."""
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, LOOPBACK) as c:
        r = await c.post("/fd/code/agent/list", json={"owner_id": "u-loop"})
    assert r.status_code == 200
    assert "projects" in r.json()


async def test_code_agent_secret_authorizes_remote(fd_db, monkeypatch):
    """A remote secret-holder (e.g. an agent behind a same-host proxy) is
    admitted; a wrong secret is blocked at the guard."""
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        ok = await c.post("/fd/hosting/agent/list", json={"owner_id": "u-r"},
                          headers={"X-Agent-Secret": "shh"})
        bad = await c.post("/fd/hosting/agent/list", json={"owner_id": "u-r"},
                           headers={"X-Agent-Secret": "nope"})
    assert ok.status_code == 200
    assert bad.status_code == 403
    assert "bearer token" in bad.json()["detail"]


async def test_code_agent_lockdown_requires_secret_even_from_loopback(fd_db, monkeypatch):
    """Under FD_LOCKDOWN, bare loopback is not enough — a bearer or the secret
    is required, matching the basna/vatra lockdown semantics."""
    monkeypatch.setenv("FD_LOCKDOWN", "1")
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    async with _client(fd_server.app, LOOPBACK) as c:
        denied = await c.post("/fd/code/agent/list", json={"owner_id": "x"})
        allowed = await c.post("/fd/code/agent/list", json={"owner_id": "x"},
                               headers={"X-Agent-Secret": "shh"})
    assert denied.status_code == 403
    assert "bearer token" in denied.json()["detail"]
    assert allowed.status_code == 200


async def test_code_agent_cancel_remote_denied_without_bearer(fd_db, monkeypatch):
    """The cancel endpoint (Spark's stall/timeout stop) is behind the same
    transport guard as the rest of /fd/code/agent/*."""
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/code/agent/cancel",
                         json={"owner_id": "victim", "project": "p", "session_id": "s"})
    assert r.status_code == 403
    assert "bearer token" in r.json()["detail"]


async def test_code_agent_cancel_bearer_cannot_impersonate(fd_db, monkeypatch):
    """Cancel resolves the owner the same way as the other agent routes: a valid
    bearer + a mismatched owner_id is rejected before any session lookup (proves
    the route is wired with the authoritative-bearer resolver)."""
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    u = await fd_db.create_user("spark-cancel@svc.local", "x")
    tok = create_access_token(u["id"])
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/code/agent/cancel",
                         json={"owner_id": "someone-else", "project": "p", "session_id": "s"},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403
    assert "does not match" in r.json()["detail"]


# ── 0a: FD_LOCKDOWN host-filesystem surfaces ─────────────────────────


async def test_lockdown_blocks_host_fs_surfaces(fd_db, monkeypatch):
    monkeypatch.setenv("FD_LOCKDOWN", "1")
    # Auth on, so non-blocked control paths answer 401 instead of running
    # their real handlers (gdrive would probe Google, links would read disk).
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    async with _client(fd_server.app, LOOPBACK) as c:
        for method, path in (("GET", "/fd/vfs/browse-fs"),
                             ("POST", "/fd/vfs/links"),
                             ("GET", "/fd/projects"),
                             ("GET", "/fd/projects/anything/status")):
            r = await c.request(method, path, json={} if method == "POST" else None)
            assert r.status_code == 403, f"{method} {path} not blocked"
            assert r.json()["detail"] == _LOCKDOWN_DETAIL
        # The Drive link route is per-user OAuth, not a host-fs mount — the
        # middleware must not block it (it proceeds to normal auth/validation).
        r = await c.post("/fd/vfs/links/gdrive", json={})
        assert r.json().get("detail") != _LOCKDOWN_DETAIL
        # Listing existing links stays readable too.
        r = await c.get("/fd/vfs/links")
        assert r.json().get("detail") != _LOCKDOWN_DETAIL


async def test_no_lockdown_leaves_surfaces_reachable(fd_db, monkeypatch):
    """Without the env, the middleware is inert — requests reach normal
    auth/routing (asserted via responses that don't run the heavy handlers:
    401 from enforced auth, 404 from an unrouted sub-path)."""
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    async with _client(fd_server.app, LOOPBACK) as c:
        r = await c.get("/fd/vfs/browse-fs")
        assert r.status_code == 401  # reached auth, not middleware-blocked
        r = await c.get("/fd/projects/x/y/z/unrouted")
        assert r.status_code == 404  # reached the router, not middleware-blocked


# ── 0b: facts ledger endpoint ────────────────────────────────────────


@pytest.fixture
async def two_users(fd_db) -> tuple[str, str]:
    """basna_sessions.user_id has a FK to users — create real rows."""
    u1 = await fd_db.create_user("u1@test.local", "x")
    u2 = await fd_db.create_user("u2@test.local", "x")
    return u1["id"], u2["id"]


def _basna_app(monkeypatch, tmp_path: Path, user_id: str) -> FastAPI:
    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(fd_server, "DATA_DIR", tmp_path, raising=False)
    app = FastAPI()
    app.include_router(basna_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": user_id}
    return app


async def test_session_facts_returns_ledger_rows(fd_db, two_users, monkeypatch, tmp_path):
    u1, _ = two_users
    sess = await fd_db.create_basna_session(u1, "size the EU market")
    sid = sess["id"]
    vfs_dir = tmp_path / "vfs" / u1 / f"basna-{sid[:8]}"
    vfs_dir.mkdir(parents=True)
    facts_ledger.upsert(vfs_dir, "eu.market.size", "42", unit="B EUR",
                        status="verified", updated_by="worker-a")
    facts_ledger.upsert(vfs_dir, "eu.market.size", "57")  # conflict, not saved

    app = _basna_app(monkeypatch, tmp_path, u1)
    async with _client(app, LOOPBACK) as c:
        r = await c.get(f"/fd/basna/sessions/{sid}/facts")
    assert r.status_code == 200
    body = r.json()
    assert body["project"] == f"basna-{sid[:8]}"
    assert body["count"] == 1
    fact = body["facts"][0]
    assert fact["key"] == "eu_market_size"  # norm_key folds separators
    assert fact["status"] == "verified"
    assert len(body["conflicts"]) == 1


async def test_session_facts_honors_vfs_project_override(fd_db, two_users, monkeypatch, tmp_path):
    """Continuation chains pin config.vfs_project — the ledger follows it."""
    u1, _ = two_users
    cfg = json.dumps({"mode": "vatra", "vfs_project": "stream-alpha"})
    sess = await fd_db.create_basna_session(u1, "round 2", config=cfg)
    vfs_dir = tmp_path / "vfs" / u1 / "stream-alpha"
    vfs_dir.mkdir(parents=True)
    facts_ledger.upsert(vfs_dir, "round.count", "2")

    app = _basna_app(monkeypatch, tmp_path, u1)
    async with _client(app, LOOPBACK) as c:
        r = await c.get(f"/fd/basna/sessions/{sess['id']}/facts")
    body = r.json()
    assert body["project"] == "stream-alpha"
    assert body["count"] == 1


async def test_session_facts_empty_without_ledger(fd_db, two_users, monkeypatch, tmp_path):
    u1, _ = two_users
    sess = await fd_db.create_basna_session(u1, "no ledger yet")
    app = _basna_app(monkeypatch, tmp_path, u1)
    async with _client(app, LOOPBACK) as c:
        r = await c.get(f"/fd/basna/sessions/{sess['id']}/facts")
    assert r.status_code == 200
    assert r.json() == {"project": f"basna-{sess['id'][:8]}", "facts": [],
                        "conflicts": [], "count": 0}


async def test_session_facts_foreign_session_404(fd_db, two_users, monkeypatch, tmp_path):
    u1, u2 = two_users
    sess = await fd_db.create_basna_session(u2, "someone else's run")
    app = _basna_app(monkeypatch, tmp_path, u1)
    async with _client(app, LOOPBACK) as c:
        r = await c.get(f"/fd/basna/sessions/{sess['id']}/facts")
    assert r.status_code == 404


def test_session_vfs_folder_resolution():
    """The folder rule the facts endpoint rides on (continuation-aware)."""
    from captain_claw.flight_deck.basna_routes import _session_vfs_folder
    sid = "abcdefgh-rest"
    assert _session_vfs_folder(
        {"id": sid, "config": '{"vfs_project": "stream-a"}'}) == "stream-a"
    assert _session_vfs_folder(
        {"id": sid, "config": '{"mode": "vatra"}'}) == "vatra-abcdefgh"
    assert _session_vfs_folder({"id": sid, "config": "{}"}) == "basna-abcdefgh"


# ── 0b: costs endpoint ───────────────────────────────────────────────


def _costs_app(user_id: str = "u1") -> FastAPI:
    from captain_claw.flight_deck import costs_routes
    app = FastAPI()
    app.include_router(costs_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"id": user_id}
    return app


async def test_costs_lists_own_rows_with_totals(fd_db):
    cost = {"tokens": {"prompt_tokens": 1000}, "usd": 0.10, "elapsed_seconds": 9.0}
    await fd_db.log_run_cost("u1", "basna", "s1", cost)
    await fd_db.log_run_cost("u1", "vatra", "s2", {"tokens": {}, "usd": None})
    await fd_db.log_run_cost("u1", "being_tick", "t1",
                             {"tokens": {}, "usd": 0.05},
                             owner_type="being", owner_ref="iskra-x")
    await fd_db.log_run_cost("u2", "code", "p/s", cost)

    async with _client(_costs_app(), LOOPBACK) as c:
        r = await c.get("/fd/costs")
        body = r.json()
        assert r.status_code == 200
        assert body["count"] == 3          # u2's row invisible
        assert body["priced"] == 2
        assert body["total_usd"] == pytest.approx(0.15)
        assert all(isinstance(row["usage"], dict) for row in body["costs"])

        by_kind = (await c.get("/fd/costs", params={"run_kind": "basna"})).json()
        assert by_kind["count"] == 1
        assert by_kind["costs"][0]["usd"] == pytest.approx(0.10)

        by_ref = (await c.get("/fd/costs", params={"ref": "iskra-x"})).json()
        assert by_ref["count"] == 1
        assert by_ref["costs"][0]["run_kind"] == "being_tick"


async def test_costs_since_filter_and_empty(fd_db):
    await fd_db.log_run_cost("u1", "basna", "s1", {"tokens": {}, "usd": 0.1})
    async with _client(_costs_app(), LOOPBACK) as c:
        future = (await c.get("/fd/costs", params={"since": "9999-01-01"})).json()
        assert future == {"costs": [], "count": 0, "priced": 0, "total_usd": None}
        past = (await c.get("/fd/costs", params={"since": "2000-01-01"})).json()
        assert past["count"] == 1
