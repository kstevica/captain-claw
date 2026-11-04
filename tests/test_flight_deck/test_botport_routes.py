"""FD routes for the BotPort link: remote consult/delegate, fleet, admin config."""

from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import botport_link as bl
from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

LOOPBACK = ("127.0.0.1", 40001)
REMOTE = ("203.0.113.9", 40001)
ALICE_PORT = 24111


def _client(addr) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=fd_server.app, client=addr),
                             base_url="http://localhost:25080")


def _bearer(user: dict) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user['id'], user.get('role', 'user'))}"}


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch):
    from captain_claw.flight_deck import auth as fd_auth
    from captain_claw.flight_deck import fd_crypto

    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "users.db")
    await db.init()
    set_auth_db(db)
    alice = await db.create_user("alice@x.test", "h", "Alice")
    admin = await db.create_user("admin@x.test", "h", "Admin", role="admin")
    registry = {"alice-agent": {"name": "Planner", "web_port": ALICE_PORT, "web_auth": "tok-alice",
                                "owner": alice["id"]}}
    (tmp_path / "fd-data").mkdir()

    def _no_docker():
        raise RuntimeError("no docker in tests")

    monkeypatch.setattr(fd_server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setattr(fd_server, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(fd_server, "_process_is_alive", lambda slug: True)
    monkeypatch.setattr(fd_server, "get_docker", _no_docker)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    for var in bl._ENV.values():
        monkeypatch.delenv(var, raising=False)
    fd_crypto.reset_for_tests()

    link = bl.BotPortLink()
    # Every user's agents may reach other decks here; the admin-only default
    # has its own tests (test_other_decks_are_the_admins_by_default).
    link.settings = bl.LinkSettings.from_dict({"enabled": True, "url": "hub:23180", "deck_name": "vps1",
                                               "outbound_users": True})
    link.state, link._ws = "connected", object()
    link.network = {"decks": [{"name": "pi", "label": "Kitchen Pi", "status": "connected", "agents": [
        {"name": "Researcher", "slug": "researcher", "description": "Finds things", "status": "running"}]}],
        "agents": []}
    calls: list[dict] = []

    async def fake_consult(**kw):
        calls.append(kw)
        yield {"event": "status", "data": {"status": "Routed"}}
        yield {"ok": True, "done": True, "response": f"remote says hi to {kw['from_agent']}"}

    link.consult = fake_consult  # type: ignore[method-assign]
    applied: list[dict] = []

    async def fake_apply(stored):
        applied.append(stored)
        link.settings = link.effective(stored)

    link.apply = fake_apply  # type: ignore[method-assign]
    monkeypatch.setattr(bl, "_LINK", link)

    delivered: list[dict] = []

    async def fake_deliver(req, source_port, source_auth, peer_display, response_text, relay_level=""):
        delivered.append({"port": source_port, "auth": source_auth, "peer": peer_display, "text": response_text})

    monkeypatch.setattr(fd_server, "_deliver_delegate_result", fake_deliver)
    try:
        yield types.SimpleNamespace(db=db, alice=alice, admin=admin, link=link, calls=calls,
                                    applied=applied, delivered=delivered)
    finally:
        await db.close()
        fd_auth._db = prev
        fd_crypto.reset_for_tests()


async def test_agent_consults_a_remote_agent_through_fd(deck):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "port": 0, "target_deck": "pi", "target_agent": "researcher", "message": "hello",
            "source_name": "spoofed-name"})
    assert r.status_code == 200
    lines = [json.loads(x) for x in r.text.splitlines() if x.strip()]
    assert lines[-1] == {"ok": True, "done": True, "response": "remote says hi to Planner"}
    # The name the remote deck sees comes from FD's records, not the body.
    assert deck.calls[0]["from_agent"] == "Planner" and deck.calls[0]["target_deck"] == "pi"


async def test_remote_consult_keeps_the_peer_route_guard(deck):
    async with _client(REMOTE) as c:
        r = await c.post("/fd/consult-peer", json={"target_deck": "pi", "target_agent": "researcher",
                                                   "message": "x"})
    assert r.status_code == 403 and deck.calls == []


async def test_remote_consult_refuses_attachments_and_bad_addresses(deck):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "target_deck": "pi", "target_agent": "researcher", "message": "x",
            "attach_path": "/etc/passwd"})
        r2 = await c.post("/fd/consult-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "target_agent": "researcher", "message": "x"})
    assert r.status_code == 400 and "cannot be attached" in r.text
    assert r2.status_code == 400 and deck.calls == []


async def test_agent_at_deck_form_is_accepted(deck):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/consult-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "target_agent": "researcher@pi", "message": "x"})
    assert r.status_code == 200 and deck.calls[0]["target_agent"] == "researcher"


async def test_remote_delegate_delivers_back_to_the_caller(deck):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/delegate-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "target_deck": "pi", "target_agent": "researcher", "message": "dig"})
    assert r.status_code == 200 and r.json()["ok"] is True
    for _ in range(50):
        if deck.delivered:
            break
        await asyncio.sleep(0.02)
    assert deck.delivered == [{"port": ALICE_PORT, "auth": "tok-alice", "peer": "Researcher@pi",
                               "text": "remote says hi to Planner"}]
    assert deck.calls[0]["mode"] == "delegate"


async def test_remote_delegate_to_an_unknown_agent_says_so(deck):
    async with _client(LOOPBACK) as c:
        r = await c.post("/fd/delegate-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "target_deck": "pi", "target_agent": "ghost", "message": "dig"})
    body = r.json()
    assert body["ok"] is False and "Researcher@pi" in body["message"]


async def test_fleet_lists_remote_agents_only_on_request(deck):
    async with _client(LOOPBACK) as c:
        plain = (await c.get("/fd/fleet", headers=_bearer(deck.alice))).json()
        full = (await c.get("/fd/fleet?include_remote=1", headers=_bearer(deck.alice))).json()
    assert all(a["kind"] != "remote" for a in plain)
    async with _client(REMOTE) as c:
        anon = (await c.get("/fd/fleet?include_remote=1")).json()
    async with _client(LOOPBACK) as c:
        as_agent = (await c.get("/fd/fleet?include_remote=1", headers={"X-Agent-Auth": "tok-alice"})).json()
    assert all(a["kind"] != "remote" for a in anon), "no roster for an anonymous caller"
    assert any(a["kind"] == "remote" for a in as_agent)
    remote = [a for a in full if a["kind"] == "remote"]
    assert remote == [{"name": "Researcher@pi", "slug": "researcher", "kind": "remote", "host": "",
                       "port": 0, "status": "running", "description": "Finds things", "deck": "pi",
                       "deck_label": "Kitchen Pi"}]


async def test_config_is_admin_only_and_secret_never_returned(deck):
    async with _client(LOOPBACK) as c:
        denied = await c.put("/fd/botport/config", headers=_bearer(deck.alice), json={"enabled": True})
        bad = await c.put("/fd/botport/config", headers=_bearer(deck.admin), json={"deck_name": "has space"})
        bad_url = await c.put("/fd/botport/config", headers=_bearer(deck.admin), json={"url": "ftp://x"})
        ok = await c.put("/fd/botport/config", headers=_bearer(deck.admin), json={
            "enabled": True, "url": "pi.local:23180", "deck_name": "vps1", "key": "k",
            "secret": "very-secret-value", "expose": ["researcher"], "outbound_users": True})
        got = await c.get("/fd/botport/config", headers=_bearer(deck.admin))
        keep = await c.put("/fd/botport/config", headers=_bearer(deck.admin), json={"secret": ""})
        status_user = await c.get("/fd/botport/status", headers=_bearer(deck.alice))
    assert denied.status_code == 403 and bad.status_code == 400 and bad_url.status_code == 400
    assert ok.status_code == 200
    body = got.json()
    assert "very-secret-value" not in got.text and body["secret_set"] is True
    assert body["normalized_url"] == "ws://pi.local:23180/ws" and body["problems"] == []
    assert keep.json()["secret_set"] is True, "an empty secret keeps the stored one"
    raw = await deck.db.get_system_setting(bl.SETTING_KEY)
    assert "very-secret-value" not in raw
    assert deck.applied[-1]["secret"] == "very-secret-value"
    assert set(status_user.json()) == {"enabled", "state", "deck_name", "remote_agents", "decks"}
    # A user sees the decks online by the names the hub gives them, no more.
    assert status_user.json()["decks"] == [{"name": "pi", "label": "Kitchen Pi"}]


async def test_sign_in_off_deck_takes_link_settings_only_from_this_machine(deck, monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client(REMOTE) as c:
        far = await c.put("/fd/botport/config", json={"enabled": True})
        far_get = await c.get("/fd/botport/config")
    async with _client(LOOPBACK) as c:
        near = await c.put("/fd/botport/config", json={"deck_name": "pi"})
    assert far.status_code == 403 and far_get.status_code == 403
    assert "from this machine" in far.json()["detail"]
    assert near.status_code == 200 and near.json()["deck_name"] == "pi"


# ── Users share their own agents ─────────────────────────────────


@pytest.fixture
async def fleet(deck, monkeypatch):
    """Alice's and Bob's agents, plus a Flight Deck worker, wired into the link."""
    bob = await deck.db.create_user("bob@x.test", "h", "Bob")
    registry = {
        "alice-agent": {"name": "Planner", "web_port": ALICE_PORT, "web_auth": "tok-alice", "owner": deck.alice["id"]},
        "bob-agent": {"name": "Coder", "web_port": 24112, "web_auth": "tok-bob", "owner": bob["id"]},
        "dubina-run-1": {"name": "Worker", "web_port": 24113, "web_auth": "tok-w", "owner": deck.alice["id"],
                         "description": "Dubina ephemeral worker"},
    }
    monkeypatch.setattr(fd_server, "_load_process_registry", lambda: registry)
    deck.link._fleet = fd_server._fleet_snapshot
    sent: list[dict] = []

    async def fake_send(frame):
        sent.append(frame)

    deck.link._send = fake_send  # type: ignore[method-assign]
    return types.SimpleNamespace(bob=bob, registry=registry, sent=sent)


async def _ads(link) -> list[str]:
    return sorted(a["name"] for a in await link._ads())


async def test_a_user_shares_their_own_agent_on_the_network(deck, fleet):
    async with _client(REMOTE) as c:
        r = await c.get("/fd/botport/shares", headers=_bearer(deck.alice))
        assert r.status_code == 200
        view = r.json()
        assert view["can_share"] and view["user_sharing"] and view["deck_name"] == "vps1"
        assert [a["slug"] for a in view["agents"]] == ["alice-agent", "dubina-run-1"], "only her own agents"
        assert all(a["mine"] for a in view["agents"]) and view["admin"] is False
        assert not view["agents"][0]["exposed"]
        assert not view["agents"][1]["shareable"], "a Flight Deck worker is never shareable"

        r = await c.put("/fd/botport/shares/alice-agent", headers=_bearer(deck.alice), json={"shared": True})
        assert r.status_code == 200, r.text
        mine = r.json()["agents"][0]
        assert mine["exposed"] and mine["by_owner"] and not mine["by_admin"]
    assert await _ads(deck.link) == ["Planner"]
    await asyncio.sleep(0)  # the link re-advertises in the background
    assert fleet.sent and fleet.sent[-1]["type"] == "deck_agents"
    assert [a["name"] for a in fleet.sent[-1]["agents"]] == ["Planner"]
    stored = bl.parse_shares(await deck.db.get_system_setting(bl.SHARES_KEY))
    assert stored[f"process:alice-agent|{deck.alice['id']}"]["owner"] == deck.alice["id"]

    async with _client(REMOTE) as c:
        r = await c.put("/fd/botport/shares/alice-agent", headers=_bearer(deck.alice), json={"shared": False})
    assert r.status_code == 200 and not r.json()["agents"][0]["exposed"]
    assert await _ads(deck.link) == []


async def test_users_cannot_share_what_is_not_theirs(deck, fleet):
    async with _client(REMOTE) as c:
        r = await c.put("/fd/botport/shares/bob-agent", headers=_bearer(deck.alice), json={"shared": True})
        assert r.status_code == 403
        r = await c.put("/fd/botport/shares/dubina-run-1", headers=_bearer(deck.alice), json={"shared": True})
        assert r.status_code == 400
        r = await c.put("/fd/botport/shares/nope", headers=_bearer(deck.alice), json={"shared": True})
        assert r.status_code == 404
        # An admin offers others' agents through the Exposed list, not as their owner...
        r = await c.put("/fd/botport/shares/bob-agent", headers=_bearer(deck.admin), json={"shared": True})
        assert r.status_code == 403 and "Exposed agents" in r.json()["detail"]
        # ...but may stop an owner's share.
        assert (await c.put("/fd/botport/shares/bob-agent", headers=_bearer(fleet.bob),
                            json={"shared": True})).status_code == 200
        assert await _ads(deck.link) == ["Coder"]
        view = (await c.get("/fd/botport/shares", headers=_bearer(deck.admin))).json()
        coder = next(a for a in view["agents"] if a["slug"] == "bob-agent")
        assert view["admin"] and coder["by_owner"] and not coder["mine"]
        r = await c.put("/fd/botport/shares/bob-agent", headers=_bearer(deck.admin), json={"shared": False})
        assert r.status_code == 200
        # Alice cannot stop Bob's share either.
        await c.put("/fd/botport/shares/bob-agent", headers=_bearer(fleet.bob), json={"shared": True})
        r = await c.put("/fd/botport/shares/bob-agent", headers=_bearer(deck.alice), json={"shared": False})
        assert r.status_code == 403
    assert await _ads(deck.link) == ["Coder"]


async def test_the_admin_can_turn_sharing_by_users_off(deck, fleet):
    async with _client(REMOTE) as c:
        assert (await c.put("/fd/botport/shares/alice-agent", headers=_bearer(deck.alice),
                            json={"shared": True})).status_code == 200
        r = await c.put("/fd/botport/config", headers=_bearer(deck.admin), json={
            "enabled": True, "url": "hub:23180", "deck_name": "vps1", "user_sharing": False})
        assert r.status_code == 200 and r.json()["user_sharing"] is False
        assert await _ads(deck.link) == [], "existing shares stop counting"
        r = await c.put("/fd/botport/shares/alice-agent", headers=_bearer(deck.alice), json={"shared": True})
        assert r.status_code == 403
        view = (await c.get("/fd/botport/shares", headers=_bearer(deck.alice))).json()
        assert not view["can_share"]
        # The admin's own agents too: with sharing by users off, the Exposed list decides.
        deck.link._fleet = lambda: [{"name": "Ops", "slug": "ops", "kind": "process", "status": "running",
                                     "owner": deck.admin["id"]}]
        r = await c.put("/fd/botport/shares/ops", headers=_bearer(deck.admin), json={"shared": True})
        assert r.status_code == 403 and "Exposed agents" in r.json()["detail"]
        deck.link._fleet = fd_server._fleet_snapshot
        # The admin's own list still works next to it.
        await c.put("/fd/botport/config", headers=_bearer(deck.admin), json={"expose": ["Planner"]})
        mine = (await c.get("/fd/botport/shares", headers=_bearer(deck.alice))).json()["agents"][0]
        assert mine["exposed"] and mine["by_admin"]


async def test_a_share_lapses_when_the_agent_changes_hands_or_is_removed(deck, fleet):
    async with _client(REMOTE) as c:
        await c.put("/fd/botport/shares/alice-agent", headers=_bearer(deck.alice), json={"shared": True})
    assert await _ads(deck.link) == ["Planner"]
    fleet.registry["alice-agent"]["owner"] = fleet.bob["id"]
    assert await _ads(deck.link) == [], "Bob never agreed to share it"
    fleet.registry["alice-agent"]["owner"] = deck.alice["id"]
    assert await _ads(deck.link) == ["Planner"]

    from captain_claw.flight_deck.botport_routes import forget_share

    await forget_share("process", "alice-agent", fleet.bob["id"])
    assert await _ads(deck.link) == ["Planner"], "only that owner's share goes"
    await forget_share("process", "alice-agent", deck.alice["id"])
    assert await _ads(deck.link) == []
    assert bl.parse_shares(await deck.db.get_system_setting(bl.SHARES_KEY)) == {}


async def test_sharing_needs_a_connected_network(deck, fleet):
    deck.link.settings = bl.LinkSettings.from_dict({"enabled": False})
    async with _client(REMOTE) as c:
        r = await c.put("/fd/botport/shares/alice-agent", headers=_bearer(deck.alice), json={"shared": True})
        assert r.status_code == 409
        assert not (await c.get("/fd/botport/shares", headers=_bearer(deck.alice))).json()["can_share"]


async def test_an_exact_slug_wins_over_another_agents_name(deck, fleet):
    fleet.registry["alice-agent"]["name"] = "beta"   # renamed: its name is another agent's slug
    fleet.registry["beta"] = {"name": "Beta", "web_port": 24114, "web_auth": "tok-beta", "owner": deck.alice["id"]}
    async with _client(REMOTE) as c:
        r = await c.put("/fd/botport/shares/beta?kind=process", headers=_bearer(deck.alice), json={"shared": True})
    assert r.status_code == 200
    assert await _ads(deck.link) == ["Beta"]
    assert list(bl.parse_shares(await deck.db.get_system_setting(bl.SHARES_KEY))) == [f"process:beta|{deck.alice['id']}"]


# ── Other decks are the admins' unless they say otherwise (cohort001) ──


async def test_other_decks_are_the_admins_by_default(deck, monkeypatch):
    deck.link.settings.outbound_users = False
    registry = fd_server._load_process_registry()
    registry["admin-agent"] = {"name": "Ops", "web_port": 24112, "web_auth": "tok-admin",
                               "owner": deck.admin["id"]}
    monkeypatch.setattr(fd_server, "_owner_is_admin", lambda owner: owner == deck.admin["id"])
    async with _client(LOOPBACK) as c:
        users = (await c.get("/fd/fleet?include_remote=1", headers=_bearer(deck.alice))).json()
        users_agent = (await c.get("/fd/fleet?include_remote=1", headers={"X-Agent-Auth": "tok-alice"})).json()
        admins = (await c.get("/fd/fleet?include_remote=1", headers=_bearer(deck.admin))).json()
        admins_agent = (await c.get("/fd/fleet?include_remote=1", headers={"X-Agent-Auth": "tok-admin"})).json()
        consult = await c.post("/fd/consult-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "target_deck": "pi", "target_agent": "researcher", "message": "hi"})
        delegate = await c.post("/fd/delegate-peer", headers={"X-Agent-Auth": "tok-alice"}, json={
            "target_deck": "pi", "target_agent": "researcher", "message": "dig"})
        admin_consult = await c.post("/fd/consult-peer", headers={"X-Agent-Auth": "tok-admin"}, json={
            "target_deck": "pi", "target_agent": "researcher", "message": "hi"})
        status_user = (await c.get("/fd/botport/status", headers=_bearer(deck.alice))).json()
    assert all(a["kind"] != "remote" for a in users + users_agent)
    assert any(a["kind"] == "remote" for a in admins) and any(a["kind"] == "remote" for a in admins_agent)
    assert consult.status_code == 403 and delegate.status_code == 403
    assert "admins only" in consult.text
    assert admin_consult.status_code == 200
    assert status_user["remote_agents"] == [] and status_user["decks"] == []


async def test_an_agent_sees_only_its_owners_agents(deck, monkeypatch):
    registry = fd_server._load_process_registry()
    registry["bob-agent"] = {"name": "Bob's", "web_port": 24113, "web_auth": "tok-bob", "owner": "bob-id"}
    async with _client(LOOPBACK) as c:
        as_alice_agent = (await c.get("/fd/fleet", headers={"X-Agent-Auth": "tok-alice"})).json()
        as_bob_agent = (await c.get("/fd/fleet", headers={"X-Agent-Auth": "tok-bob"})).json()
        nameless = (await c.get("/fd/fleet")).json()
        forged = (await c.get("/fd/fleet", headers={"X-Agent-Auth": "not-a-token"})).json()
    async with _client(REMOTE) as c:
        # Off loopback without the agent secret the token alone doesn't count.
        remote_alice = (await c.get("/fd/fleet", headers={"X-Agent-Auth": "tok-alice"})).json()
    assert [a["name"] for a in as_alice_agent] == ["Planner"]
    assert [a["name"] for a in as_bob_agent] == ["Bob's"]
    assert nameless == [] and forged == [] and remote_alice == []
