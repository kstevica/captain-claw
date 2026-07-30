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
from captain_claw.flight_deck.auth import get_current_user, set_auth_db
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
    """Default behavior for locally spawned agents is byte-identical."""
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, LOOPBACK) as c:
        r = await c.post("/fd/basna/agent/sessions", json={"owner_id": "u-loop"})
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


async def test_agent_route_secret_authorizes_remote(fd_db, monkeypatch):
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        ok = await c.post("/fd/basna/agent/sessions", json={"owner_id": "u-r"},
                          headers={"X-Agent-Secret": "shh"})
        bad = await c.post("/fd/basna/agent/sessions", json={"owner_id": "u-r"},
                           headers={"X-Agent-Secret": "nope"})
    assert ok.status_code == 200
    assert bad.status_code == 403


async def test_agent_route_vatra_prefix_also_guarded(fd_db, monkeypatch):
    monkeypatch.delenv("FD_AGENT_SHARED_SECRET", raising=False)
    async with _client(fd_server.app, REMOTE) as c:
        r = await c.post("/fd/vatra/agent/blackboard", json={"owner_id": "victim"})
    assert r.status_code == 403


async def test_lockdown_requires_secret_even_from_loopback(fd_db, monkeypatch):
    """A same-host TLS proxy must not launder remote callers into 'loopback'."""
    monkeypatch.setenv("FD_LOCKDOWN", "1")
    monkeypatch.setenv("FD_AGENT_SHARED_SECRET", "shh")
    async with _client(fd_server.app, LOOPBACK) as c:
        denied = await c.post("/fd/basna/agent/sessions", json={"owner_id": "x"})
        allowed = await c.post("/fd/basna/agent/sessions", json={"owner_id": "x"},
                               headers={"X-Agent-Secret": "shh"})
    assert denied.status_code == 403
    assert allowed.status_code == 200


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
