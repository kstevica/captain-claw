"""Captain Swarm is open to the deck's admins only: every page route, and the
`captain_swarm` tool's routes for agents whose owner is not an admin."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from captain_claw.flight_deck import captain_swarm_routes as routes
from captain_claw.flight_deck import crew_runner
from captain_claw.flight_deck.auth import get_current_user


def _app(user: dict) -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app)


def test_every_page_route_needs_an_admin():
    """A route added later can't forget the lock: all but the agents' go through admin_user."""
    for route in routes.router.routes:
        if "/agent/" in route.path:
            continue
        calls = {d.call for d in route.dependant.dependencies}
        assert routes.admin_user in calls, route.path


@pytest.mark.parametrize("method,path", [
    ("get", "/fd/captain-swarm/templates"),
    ("get", "/fd/captain-swarm/runs"),
    ("post", "/fd/captain-swarm/runs"),
    ("get", "/fd/captain-swarm/practice"),
    ("post", "/fd/captain-swarm/practice/mail-triage"),
    ("get", "/fd/captain-swarm/lessons"),
    ("post", "/fd/captain-swarm/runs/cs_0123456789ab/go"),
])
def test_a_member_is_refused(method, path):
    with _app({"id": "u2", "role": "user"}) as c:
        r = getattr(c, method)(path, **({"json": {}} if method == "post" else {}))
    assert r.status_code == 403 and r.json()["detail"] == routes.ADMINS_ONLY


def test_an_admin_gets_in():
    with _app({"id": "u1", "role": "admin"}) as c:
        assert c.get("/fd/captain-swarm/lessons").status_code == 200


@pytest.mark.parametrize("path", ["start", "status", "get", "list", "cancel"])
def test_an_agent_of_a_member_is_refused(monkeypatch, path):
    monkeypatch.setattr(routes, "_agent_owner", lambda body: "u2")

    async def user_of(uid):
        return {"id": uid, "role": "user"}
    monkeypatch.setattr(crew_runner, "_user_of", user_of)
    with _app({"id": "u1", "role": "admin"}) as c:
        r = c.post(f"/fd/captain-swarm/agent/{path}", json={"run_id": "cs_0123456789ab", "template": "document"})
    assert r.status_code == 403 and r.json()["detail"] == routes.ADMINS_ONLY


def test_an_agent_of_an_admin_or_of_a_deck_with_sign_in_off_gets_in(monkeypatch):
    real_user_of = crew_runner._user_of
    monkeypatch.setattr(routes, "_agent_owner", lambda body: "u1")

    async def admin_of(uid):
        return {"id": uid, "role": "admin"}
    monkeypatch.setattr(crew_runner, "_user_of", admin_of)
    with _app({"id": "u1", "role": "admin"}) as c:
        assert c.post("/fd/captain-swarm/agent/list", json={}).json() == {"runs": []}
        assert c.post("/fd/captain-swarm/agent/status", json={"run_id": "cs_none"}).json() == {"found": False}
    # Sign-in off: the deck's one user is its local admin, and so its agents' owner.
    monkeypatch.setattr(crew_runner, "_user_of", real_user_of)
    monkeypatch.setattr("captain_claw.flight_deck.auth._fd_auth_enabled", lambda: False)
    monkeypatch.setattr(routes, "_agent_owner", lambda body: "local")
    with _app({"id": "local", "role": "admin"}) as c:
        assert c.post("/fd/captain-swarm/agent/list", json={}).status_code == 200


@pytest.fixture(autouse=True)
def store(tmp_path):
    import asyncio

    from captain_claw.flight_deck import captain_swarm_loop as loop
    from captain_claw.flight_deck.captain_swarm_store import CaptainSwarmStore
    s = CaptainSwarmStore(tmp_path / "cs.db")
    loop.set_store(s)
    yield s
    loop.set_store(None)
    asyncio.run(s.close())
