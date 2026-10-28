"""/fd/agent-insights/{host}/{port}: the kiosk's view of what an agent
remembers — listed and searched, added to, edited and forgotten through the
agent's own /api/insights. Only the owner's fields pass, only an insight id
reaches the agent's path, and like every port-addressed proxy only the
agent's owner (or an admin) gets through."""

from __future__ import annotations

import types
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

REMOTE = ("203.0.113.9", 40001)
PORT = 24121
_RealAsyncClient = httpx.AsyncClient  # the routes' client is patched; ours is not


def _client() -> httpx.AsyncClient:
    return _RealAsyncClient(transport=httpx.ASGITransport(app=server.app, client=REMOTE),
                            base_url="http://localhost:25080")


def _bearer(user: dict) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user['id'], user.get('role', 'user'))}"}


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch):
    from captain_claw.flight_deck import auth as fd_auth

    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "users.db")
    await db.init()
    set_auth_db(db)
    alice = await db.create_user("alice@x.test", "h", "Alice")
    bob = await db.create_user("bob@x.test", "h", "Bob")
    registry = {"helpdesk": {"name": "Helpdesk", "web_port": PORT, "web_auth": "tok-a", "owner": alice["id"]}}
    (tmp_path / "fd-data").mkdir()
    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setattr(server, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(server, "_process_is_alive", lambda slug: True)
    def no_docker():
        raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server, "get_docker", no_docker)
    try:
        yield types.SimpleNamespace(alice=alice, bob=bob)
    finally:
        await db.close()
        fd_auth._db = prev


@pytest.fixture
def agent(monkeypatch) -> list[dict]:
    """The agent's side: every request it gets; it answers like /api/insights does."""
    got: list[dict] = []

    class _Agent:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, method, url, *, params=None, json=None, **kwargs):
            got.append({"method": method, "url": httpx.URL(url).copy_merge_params(params or {}), "json": json})
            answer = ({"items": [{"id": "abc123def456", "content": "Shipping is free over 50 EUR.",
                                  "source_file": "price-list.pdf"}], "total": 1} if method == "GET"
                      else {"ok": True} if method == "DELETE" else {"id": "abc123def456", **(json or {})})
            return httpx.Response(201 if method == "POST" else 200, json=answer,
                                  request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "AsyncClient", _Agent)
    return got


async def test_the_owner_lists_and_searches_what_it_remembers(deck, agent):
    async with _client() as c:
        r = await c.get(f"/fd/agent-insights/localhost/{PORT}", headers=_bearer(deck.alice),
                        params={"q": " shipping ", "category": "fact"})
    assert r.status_code == 200 and r.json()["items"][0]["source_file"] == "price-list.pdf"
    (sent,) = agent
    assert sent["method"] == "GET" and sent["url"].path == "/api/insights"
    assert dict(sent["url"].params) == {"limit": "200", "q": "shipping", "category": "fact", "token": "tok-a"}


async def test_another_user_cant_reach_it(deck, agent):
    async with _client() as c:
        for method, path in (("GET", ""), ("POST", ""), ("PATCH", "/abc123def456"), ("DELETE", "/abc123def456")):
            r = await c.request(method, f"/fd/agent-insights/localhost/{PORT}{path}", headers=_bearer(deck.bob),
                                json={"content": "x"} if method in ("POST", "PATCH") else None)
            assert r.status_code == 403, (method, r.text)
    assert agent == []


async def test_teaching_it_passes_only_the_owners_fields(deck, agent):
    async with _client() as c:
        r = await c.post(f"/fd/agent-insights/localhost/{PORT}", headers=_bearer(deck.alice), json={
            "content": "  Ana is the finance lead.  ", "category": "Contact", "importance": 42, "tags": "team",
            "source_tool": "spoofed", "id": "zzz", "project_id": "other"})
        assert r.status_code == 201
        assert agent[0]["json"] == {"content": "Ana is the finance lead.", "category": "contact",
                                    "importance": 10, "tags": "team"}
        r = await c.post(f"/fd/agent-insights/localhost/{PORT}", headers=_bearer(deck.alice), json={"content": " "})
        assert r.status_code == 400 and len(agent) == 1
        r = await c.post(f"/fd/agent-insights/localhost/{PORT}", headers=_bearer(deck.alice),
                         json={"content": "A fact.", "category": "nonsense"})
        assert agent[-1]["json"] == {"content": "A fact."}           # an unknown category isn't passed on


async def test_editing_and_forgetting_reach_only_an_insight(deck, agent):
    async with _client() as c:
        r = await c.patch(f"/fd/agent-insights/localhost/{PORT}/abc123def456", headers=_bearer(deck.alice),
                          json={"importance": 3, "why": "not editable here"})
        assert r.status_code == 200 and agent[-1]["json"] == {"importance": 3}
        assert agent[-1]["method"] == "PATCH" and agent[-1]["url"].path == "/api/insights/abc123def456"
        r = await c.delete(f"/fd/agent-insights/localhost/{PORT}/abc123def456", headers=_bearer(deck.alice))
        assert r.status_code == 200 and agent[-1]["method"] == "DELETE"
        n = len(agent)
        # "pending" is the agent's review queue (DELETE clears all of it): never an insight id.
        for bad in ("pending", "..", "ABC123DEF456!"):
            r = await c.delete(f"/fd/agent-insights/localhost/{PORT}/{bad}", headers=_bearer(deck.alice))
            assert r.status_code in (400, 404, 405), bad
        r = await c.patch(f"/fd/agent-insights/localhost/{PORT}/abc123def456", headers=_bearer(deck.alice),
                          json={"source_tool": "x"})
        assert r.status_code == 400
    assert len(agent) == n
