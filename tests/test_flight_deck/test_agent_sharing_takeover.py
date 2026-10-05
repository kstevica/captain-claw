"""A1 — a name is a directory: nobody takes over another user's stopped agent.

A spawn or clone that reuses an agent's name reuses ``DATA_DIR/<slug>`` — its
sessions (members' private chats with a shared agent among them), memory and
workspace. Pinned here:

* a process clone may replace only the caller's own stopped entry (409 for
  another user's or an unowned one, before saying whether it runs);
* a Docker spawn or clone may remove only the caller's own stopped container;
* whatever replaces an agent drops that agent's shares and closes its members'
  sockets (the replacement is a new agent with a new ref).

Same deck as ``test_agent_sharing`` (real FlightDeckDB + process registry in
tmp dirs, Docker faked).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from captain_claw.flight_deck import agent_sharing as sharing
from captain_claw.flight_deck import server
from test_flight_deck import test_agent_sharing as base
from test_flight_deck.test_agent_sharing import (
    DOCKER_INST,
    DOCKER_REF,
    MEMBER,
    OTHER,
    OWNER,
    FakeContainer,
    _client,
    _hdr,
    _ref,
)

deck = base.deck  # the same deck fixture
SLEEPY_REF = _ref("sleepy", "4444444444444444")
REMOVED = (None, 4404, "Agent removed")


@pytest.fixture
def spy(monkeypatch):
    calls: list = []

    async def fake_close(ref, user_id=None, *, code=4403, reason="Access removed"):
        calls.append((ref, user_id, code, reason))
        return 0

    monkeypatch.setattr(sharing, "close_member_sockets", fake_close)
    return calls


@pytest.fixture
def dock(deck, monkeypatch):
    """Docker as the spawn and clone routes use it, over the deck's containers."""
    import docker as docker_mod

    from captain_claw.flight_deck import rate_limiter

    runs: list[dict] = []

    def _get(name):
        for c in deck.containers:
            if c.name == name:
                return c
        raise docker_mod.errors.NotFound("no such container")

    def _run(**kw):
        runs.append(kw)
        labels = kw["labels"]
        box = FakeContainer(deck.containers, kw["name"], labels[server.OWNER_LABEL],
                            labels["flight-deck.web-auth"],
                            int(labels["flight-deck.web-port"] or 0),
                            instance=labels.get(sharing.INSTANCE_LABEL))
        deck.containers.append(box)
        return box

    client = SimpleNamespace(containers=SimpleNamespace(
        get=_get, run=_run, list=lambda **kw: list(deck.containers)))
    monkeypatch.setattr(server, "get_docker", lambda: client)

    async def _sys_cfg():
        return {"docker_spawn_enabled": True}

    async def _no_cap(user, count):
        return None

    monkeypatch.setattr(server, "_get_system_config", _sys_cfg)
    monkeypatch.setattr(server, "check_agent_count_limit", _no_cap)
    monkeypatch.setattr(server, "_is_port_available", lambda port: True)
    monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)
    monkeypatch.setattr(server.app.state, "fd_db", deck.db, raising=False)
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fd-env-key")
    return SimpleNamespace(runs=runs, containers=deck.containers)


def _container(dock, name: str):
    return next((c for c in dock.containers if c.name == name), None)


class TestProcessClone:
    async def test_onto_another_users_stopped_agent_is_refused(self, deck, spy):
        await deck.db.create_share("agent", SLEEPY_REF, OWNER, MEMBER, "view")
        (deck.data / "sleepy" / "member-chat.txt").write_text("private")
        async with _client() as c:
            r = await c.post("/fd/processes/others/clone", json={"new_name": "Sleepy"},
                             headers=_hdr(OTHER))
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == "An agent named 'sleepy' already exists. Choose a different name."
        entry = server._load_process_registry()["sleepy"]
        assert (entry["owner"], entry["instance_id"]) == (OWNER, "4444444444444444")
        assert sharing.resolve_agent_record(SLEEPY_REF) is not None
        assert await deck.db.is_agent_member(SLEEPY_REF, OWNER, MEMBER)
        assert spy == []

    async def test_running_or_unowned_entries_are_refused_too(self, deck, spy):
        async with _client() as c:
            # another user's RUNNING agent: 409, not "already running"
            running = await c.post("/fd/processes/others/clone", json={"new_name": "helper"},
                                   headers=_hdr(OTHER))
            unowned = await c.post("/fd/processes/legacy/clone", json={"new_name": "orphan"},
                                   headers=_hdr(OWNER))
            own_running = await c.post("/fd/processes/legacy/clone", json={"new_name": "helper"},
                                       headers=_hdr(OWNER))
        assert running.status_code == 409, running.text
        assert unowned.status_code == 409, unowned.text
        assert own_running.status_code == 400, own_running.text  # unchanged
        registry = server._load_process_registry()
        assert registry["helper"]["owner"] == OWNER and registry["orphan"]["owner"] == ""
        assert spy == []

    async def test_onto_own_stopped_agent_drops_its_shares(self, deck, spy):
        await deck.db.create_share("agent", SLEEPY_REF, OWNER, MEMBER, "view")
        async with _client() as c:
            r = await c.post("/fd/processes/legacy/clone", json={"new_name": "Sleepy"},
                             headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        entry = server._load_process_registry()["sleepy"]
        assert entry["owner"] == OWNER and entry["instance_id"] != "4444444444444444"
        assert sharing.resolve_agent_record(SLEEPY_REF) is None
        assert await deck.db.list_agent_members(SLEEPY_REF, OWNER) == []
        assert spy == [(SLEEPY_REF, *REMOVED)]

    async def test_a_fresh_name_touches_no_shares(self, deck, spy):
        await deck.db.create_share("agent", SLEEPY_REF, OWNER, MEMBER, "view")
        async with _client() as c:
            r = await c.post("/fd/processes/sleepy/clone", json={"new_name": "Sleepy Two"},
                             headers=_hdr(OWNER))
        assert r.status_code == 200, r.text
        assert await deck.db.is_agent_member(SLEEPY_REF, OWNER, MEMBER)
        assert spy == []


class TestDockerSpawn:
    @pytest.mark.parametrize("status", ["exited", "running"])
    async def test_over_another_users_container_is_refused(self, deck, dock, spy, status):
        helper = _container(dock, "helper")  # OTHER's
        helper.status = status
        async with _client() as c:
            r = await c.post("/fd/spawn", json={"name": "Helper"}, headers=_hdr(OWNER))
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == "An agent named 'helper' already exists. Choose a different name."
        assert _container(dock, "helper") is helper and dock.runs == []
        assert spy == []

    async def test_over_own_stopped_container_drops_its_shares(self, deck, dock, spy):
        _container(dock, "helper").status = "exited"
        await deck.db.create_share("agent", DOCKER_REF, OTHER, MEMBER, "view")
        async with _client() as c:
            r = await c.post("/fd/spawn", json={"name": "Helper"}, headers=_hdr(OTHER))
        assert r.status_code == 200, r.text
        new = _container(dock, "helper")
        assert new.labels[sharing.INSTANCE_LABEL] not in ("", DOCKER_INST)
        assert sharing.resolve_agent_record(DOCKER_REF) is None
        assert await deck.db.list_agent_members(DOCKER_REF, OTHER) == []
        assert spy == [(DOCKER_REF, *REMOVED)]


class TestDockerClone:
    async def test_over_another_users_stopped_container_is_refused(self, deck, dock, spy):
        helper = _container(dock, "helper")  # OTHER's
        helper.status = "exited"
        dock.containers.append(FakeContainer(dock.containers, "mine", OWNER, "mine-tok", 24992,
                                             instance="abababababababab"))
        async with _client() as c:
            r = await c.post("/fd/containers/sid-mine/clone", json={"new_name": "Helper"},
                             headers=_hdr(OWNER))
        assert r.status_code == 409, r.text
        assert _container(dock, "helper") is helper and dock.runs == []
        assert spy == []

    async def test_over_own_stopped_container_drops_its_shares(self, deck, dock, spy):
        _container(dock, "helper").status = "exited"
        dock.containers.append(FakeContainer(dock.containers, "spare", OTHER, "spare-tok", 24993,
                                             instance="cdcdcdcdcdcdcdcd"))
        await deck.db.create_share("agent", DOCKER_REF, OTHER, MEMBER, "view")
        async with _client() as c:
            r = await c.post("/fd/containers/sid-spare/clone", json={"new_name": "Helper"},
                             headers=_hdr(OTHER))
        assert r.status_code == 200, r.text
        assert _container(dock, "helper").labels[sharing.INSTANCE_LABEL] not in ("", DOCKER_INST)
        assert await deck.db.list_agent_members(DOCKER_REF, OTHER) == []
        assert spy == [(DOCKER_REF, *REMOVED)]

