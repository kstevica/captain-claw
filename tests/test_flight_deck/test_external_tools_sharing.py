# ruff: noqa: F811 — the `data` and `deck` fixtures are imported from test_external_tools
"""Custom tools, Phase 3: offering a tool to a person or to everybody (an admin
approves that, unless they turned it off), and the person's consent (Add)
before any agent of theirs runs it, as them, with their secrets."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    admin_routes,
    agent_identity,
    settings_routes,
    share_routes,
    speaker_grants,
)
from captain_claw.flight_deck import external_tools as xt
from captain_claw.flight_deck import external_tools_routes as xr
from captain_claw.flight_deck.auth import get_current_user

from .test_external_tools import LOOPBACK, _commit, _manifest, _switch_on, data, deck  # noqa: F401

WHO = "import json,os,sys\nd=json.load(sys.stdin)\nprint(json.dumps({'key': os.environ.get('KEY'), 'n': d.get('n', 0)}))\n"


def _app(user: dict | None) -> FastAPI:
    app = FastAPI()
    app.include_router(xr.agent_router)
    app.include_router(xr.catalog_router)
    app.include_router(xr.router)
    app.include_router(share_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(settings_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return app


def _as(user: dict | None) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(user), client=LOOPBACK), base_url="http://deck.test")


async def _tool(uid: str, slug: str = "who", **manifest) -> None:
    await xt.create_tool(uid, slug, title=slug.title(), description=f"The {slug} tool.")
    _commit(uid, slug, {"main.py": WHO, "tool.json": _manifest(slug, secrets=[{"name": "KEY", "why": "for tests"}],
                                                                **manifest), "tests/test_main.py": None})
    state = await xt.publish(uid, slug, by=uid, secret_values={"KEY": "publisher"})
    assert state["state"] == "passed", state


@pytest.fixture
async def team(deck):
    """A (user) owns a tool, B (other) is a teammate, the admin decides."""
    await _switch_on(deck)
    await _tool(deck.user["id"])
    deck.a, deck.b = deck.user, deck.other
    return deck


async def _share(team, grantee: dict) -> None:
    async with _as(team.a) as c:
        resp = await c.post("/fd/shares", json={"resource_type": "external_tool",
                                                 "resource_id": f"{team.a['id']}/who", "grantee_id": grantee["id"]})
    assert resp.status_code == 200, resp.text


async def test_an_offer_to_a_person_needs_their_add(team):
    a, b = team.a["id"], team.b["id"]
    async with _as(team.b) as c:
        assert (await c.get("/fd/external-tools/shared")).json()["tools"] == []
        assert (await c.post(f"/fd/external-tools/who/run?owner={a}", json={"input": {}})).status_code == 404
    await _share(team, team.b)
    async with _as(team.b) as c:
        [offer] = (await c.get("/fd/external-tools/shared")).json()["tools"]
        assert offer["slug"] == "who" and offer["offered"] == "person" and offer["added"] is False
        assert offer["declares"]["secrets"] == ["KEY"] and offer["owner_person"]["email"] == "u@example.com"
        # Offered isn't added: nothing runs yet.
        assert (await c.post(f"/fd/external-tools/who/run?owner={a}", json={"input": {}})).status_code == 404
        assert (await c.post("/fd/external-tools/added", json={"owner": a, "slug": "who"})).status_code == 200
        resp = await c.post(f"/fd/external-tools/who/run?owner={a}", json={"input": {}})
        assert resp.status_code == 400 and "KEY isn't set" in resp.json()["detail"]
        put = await c.put(f"/fd/external-tools/who/secrets?owner={a}", json={"values": {"KEY": "bee"}})
        assert put.status_code == 200 and put.json()["secrets"][0]["set"] is True
        out = (await c.post(f"/fd/external-tools/who/run?owner={a}", json={"input": {"n": 3}})).json()
        assert out["ok"] and out["output"] == {"key": "bee", "n": 3}
        detail = (await c.get(f"/fd/external-tools/who?owner={a}")).json()
        assert detail["added"] is True and "versions" not in detail and "publish" not in detail
        # Use, never change.
        assert (await c.post(f"/fd/external-tools/who/publish?owner={a}", json={})).status_code == 404
        for path in ("/fd/external-tools/who/sharing", "/fd/external-tools/who/usage"):
            assert (await c.get(path)).status_code == 404
        assert (await c.put("/fd/settings", json={"settings": {f"xtool-added:{a}/who": "x"}})).status_code == 400
    async with _as(team.a) as c:
        sharing = (await c.get("/fd/external-tools/who/sharing")).json()
        assert [p["grantee_id"] for p in sharing["people"]] == [b] and sharing["added_by"] == 1
        usage = (await c.get("/fd/external-tools/who/usage")).json()
        # The run refused for a missing secret never started, so it isn't a call.
        assert usage["calls"] == 1 and usage["failed"] == 0 and usage["people"] == 1
        assert "bee" not in str(usage)
        assert (await c.delete(f"/fd/shares?resource_type=external_tool&resource_id={a}/who&grantee_id={b}")).status_code == 200
    async with _as(team.b) as c:
        assert (await c.post(f"/fd/external-tools/who/run?owner={a}", json={"input": {}})).status_code == 404


async def test_everybody_waits_for_an_admin(team):
    a = team.a["id"]
    async with _as(team.a) as c:
        resp = await c.put("/fd/external-tools/who/everybody", json={"on": True})
        assert resp.status_code == 200 and resp.json()["everybody"]["status"] == "pending"
        assert resp.json()["everybody"]["live"] is False
    bells = await team.db.list_notifications(team.admin["id"])
    assert any("asks to offer the custom tool" in n["title"] for n in bells)
    async with _as(team.b) as c:
        assert (await c.get("/fd/external-tools/catalog")).json()["tools"] == []
        assert (await c.get("/fd/external-tools/approvals")).status_code == 403
    async with _as(team.admin) as c:
        [waiting] = (await c.get("/fd/external-tools/approvals")).json()["approvals"]
        assert waiting["slug"] == "who" and waiting["manifest"]["secrets"][0]["name"] == "KEY"
        assert (await c.post(f"/fd/external-tools/approvals/{a}/who", json={"approve": True})).status_code == 200
        assert (await c.post(f"/fd/external-tools/approvals/{a}/who", json={"approve": True})).status_code == 409
    async with _as(team.b) as c:
        [listed] = (await c.get("/fd/external-tools/catalog")).json()["tools"]
        assert listed["offered"] == "everybody"
        assert (await c.post("/fd/external-tools/added", json={"owner": a, "slug": "who"})).status_code == 200
        await c.put(f"/fd/external-tools/who/secrets?owner={a}", json={"values": {"KEY": "k"}})
        assert (await c.post(f"/fd/external-tools/who/run?owner={a}", json={"input": {}})).json()["ok"]
    async with _as(team.a) as c:
        assert (await c.get("/fd/external-tools/catalog")).json()["tools"] == []  # not your own
        await c.put("/fd/external-tools/who/everybody", json={"on": False})
    async with _as(team.b) as c:
        assert (await c.post(f"/fd/external-tools/who/run?owner={a}", json={"input": {}})).status_code == 404


async def test_with_the_approval_off_everybody_is_offered_it_at_once(team):
    async with _as(team.admin) as c:
        await c.put("/fd/admin/config", json={"external_tools_everybody_approval": False})
    async with _as(team.a) as c:
        resp = await c.put("/fd/external-tools/who/everybody", json={"on": True})
        assert resp.json()["everybody"]["status"] == "approved"
    async with _as(team.b) as c:
        assert [t["slug"] for t in (await c.get("/fd/external-tools/catalog")).json()["tools"]] == ["who"]


async def test_a_rejected_offer_reaches_nobody(team):
    a = team.a["id"]
    async with _as(team.a) as c:
        await c.put("/fd/external-tools/who/everybody", json={"on": True})
    async with _as(team.admin) as c:
        await c.post(f"/fd/external-tools/approvals/{a}/who", json={"approve": False})
    async with _as(team.b) as c:
        assert (await c.get("/fd/external-tools/catalog")).json()["tools"] == []
        assert (await c.post("/fd/external-tools/added", json={"owner": a, "slug": "who"})).status_code == 404
    assert any("wasn't approved" in n["title"] for n in await team.db.list_notifications(a))


async def test_an_agent_sees_the_tools_its_person_added(team, monkeypatch):
    a, b = team.a["id"], team.b["id"]
    await _share(team, team.b)
    async with _as(team.b) as c:
        await c.post("/fd/external-tools/added", json={"owner": a, "slug": "who"})
        await c.put(f"/fd/external-tools/who/secrets?owner={a}", json={"values": {"KEY": "bees-key"}})
    monkeypatch.setattr(agent_identity, "require_agent_caller",
                        lambda request, what="": agent_identity.AgentCaller(slug="bee", owner=b))
    monkeypatch.setattr(speaker_grants, "member_request", lambda request: False)
    async with _as(None) as agent:
        tools = (await agent.get("/fd/external-tools/agent/tools")).json()["tools"]
        assert [(t["ref"], t.get("by")) for t in tools] == [("who", "u@example.com")]
        out = (await agent.post("/fd/external-tools/agent/run", json={"tool": "who", "input": {}})).json()
        assert out["output"]["key"] == "bees-key"
    # B makes a tool of the same name: theirs keeps the name, A's is told apart.
    await _tool(b)
    async with _as(None) as agent:
        refs = sorted(t["ref"] for t in (await agent.get("/fd/external-tools/agent/tools")).json()["tools"])
        assert refs == ["u/who", "who"]
        theirs = (await agent.post("/fd/external-tools/agent/run", json={"tool": "u/who", "input": {}})).json()
        assert theirs["output"]["key"] == "bees-key"


async def test_a_new_version_that_asks_for_more_tells_who_added_it(team):
    a, b = team.a["id"], team.b["id"]
    await _share(team, team.b)
    async with _as(team.b) as c:
        await c.post("/fd/external-tools/added", json={"owner": a, "slug": "who"})
    _commit(a, "who", {"tool.json": _manifest("who", secrets=[{"name": "KEY", "why": ""}], timeout_s=20)})
    assert (await xt.publish(a, "who", by=a, secret_values={"KEY": "p"}))["state"] == "passed"
    assert not [n for n in await team.db.list_notifications(b) if "updated" in n["title"]]
    _commit(a, "who", {"tool.json": _manifest("who", secrets=[{"name": "KEY", "why": ""}], network=True)})
    assert (await xt.publish(a, "who", by=a, secret_values={"KEY": "p"}))["state"] == "passed"
    [bell] = [n for n in await team.db.list_notifications(b) if "updated" in n["title"]]
    assert "it now asks to reach the internet" in bell["title"]


async def test_deleting_a_tool_takes_its_offers_and_consents(team):
    a, b = team.a["id"], team.b["id"]
    await _share(team, team.b)
    async with _as(team.b) as c:
        await c.post("/fd/external-tools/added", json={"owner": a, "slug": "who"})
        await c.put(f"/fd/external-tools/who/secrets?owner={a}", json={"values": {"KEY": "k"}})
    async with _as(team.a) as c:
        await c.put("/fd/external-tools/who/everybody", json={"on": True})
        assert (await c.delete("/fd/external-tools/who")).status_code == 200
    settings = await team.db.get_all_settings(b)
    assert not [k for k in settings if k.startswith(("xtool:", "xtool-added:"))]
    assert await team.db.get_tool_listing(a, "who") is None
    assert await team.db.list_shares_for_grantee(b, "external_tool") == []


async def test_only_a_published_tool_is_offered_to_everybody(team):
    await xt.create_tool(team.a["id"], "draft", title="Draft", description="d")
    async with _as(team.a) as c:
        assert (await c.put("/fd/external-tools/draft/everybody", json={"on": True})).status_code == 409


def test_route_names_are_reserved():
    for name in ("shared", "catalog", "approvals", "agent", "build", "added"):
        assert name in xt.RESERVED_SLUGS
