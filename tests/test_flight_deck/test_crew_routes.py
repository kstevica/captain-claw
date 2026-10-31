"""Crew's boundary: an admin approves it per user, Flight Deck checks on every call,
and an agent can ask for a merge but only the person's Approve does it."""

from __future__ import annotations

import asyncio
import json
import stat
import subprocess
import time
import types
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import admin_routes, agent_identity
from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import autonomy, crew_access, crew_routes, crew_runner, crew_store, fd_crypto, fd_dispatch
from captain_claw.flight_deck import settings_routes
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.tools.crew import CrewTool

from .test_crew_runner import FAKE_CLI

LOOPBACK = ("127.0.0.1", 40001)
GIT = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com"]


@pytest.fixture
async def deck(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_DATA_KEY_FILE", str(tmp_path / "data_key"))
    fd_crypto.reset_for_tests()
    crew_store.set_root(tmp_path / "crew")
    fake = tmp_path / "bin" / "claude"
    fake.parent.mkdir()
    fake.write_text(FAKE_CLI)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FD_CREW_CLAUDE_BIN", str(fake))
    monkeypatch.setattr("captain_claw.flight_deck.server.DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))

    async def notify(task, *, stalled=False):
        return None

    monkeypatch.setattr(crew_runner, "_notify", notify)
    monkeypatch.setattr(crew_runner, "_sync_board", lambda task: None)
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    monkeypatch.setattr(auth_mod, "_db", db)
    admin = await db.create_user("admin@example.com", "x", role="admin")
    user = await db.create_user("u@example.com", "x", role="user")
    other = await db.create_user("o@example.com", "x", role="user")
    yield types.SimpleNamespace(db=db, admin=admin, user=user, other=other, tmp=tmp_path)
    for task in crew_store.get_store().with_status(*crew_store.ACTIVE):
        await crew_runner._terminate(task)
    crew_store.get_store().close()
    crew_store.set_root(None)
    crew_runner._PROCS.clear()
    crew_runner._LOCKS.clear()
    await db.close()


def _client(user: dict | None) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(crew_routes.router)
    app.include_router(admin_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=LOOPBACK), base_url="http://deck.test")


async def _approve(deck, user: dict, **over) -> None:
    body = {"approved": True, "harnesses": ["claude"], "max_parallel": 2, **over}
    async with _client(deck.admin) as admin:
        resp = await admin.put(f"/fd/admin/users/{user['id']}", json={"crew": body})
    assert resp.status_code == 200, resp.text


async def _settle(uid: str, task_id: str) -> dict:
    end = time.monotonic() + 20
    while time.monotonic() < end:
        await crew_runner.tick()
        fresh = crew_store.get_store().get(uid, task_id)
        if fresh["status"] not in crew_store.ACTIVE:
            return fresh
        await asyncio.sleep(0.05)
    raise AssertionError(fresh["status"])


# ── Who may ──────────────────────────────────────────────────────────────


def _row(role="user", **meta) -> dict:
    return {"id": "u1", "role": role, "metadata": json.dumps(meta)}


def test_nobody_is_approved_by_default():
    assert not crew_access.grant_of(_row()).allowed
    assert not crew_access.grant_of(None).allowed


def test_the_grant_rules():
    admin = crew_access.grant_of(_row("admin"))
    assert admin.allowed and admin.trusted and admin.why == "admin"
    good = crew_access.grant_of(_row(crew={"approved": True, "harnesses": ["pi", "codex", "nonsense"], "max_parallel": 3}))
    assert good.allowed and not good.trusted
    assert set(good.harnesses) == {"pi", "codex"}
    assert good.max_parallel == 3
    assert crew_access.grant_of(_row(crew={"approved": True, "harnesses": ["pi"], "max_parallel": 999})).max_parallel \
        == crew_access.MAX_PARALLEL
    for bad in ({"approved": "yes", "harnesses": ["pi"]}, {"approved": True}, {"approved": True, "harnesses": []},
                {"approved": False, "harnesses": ["pi"]}, "true", 1):
        assert not crew_access.grant_of(_row(crew=bad)).allowed, bad
    assert not crew_access.grant_of({"id": "u", "role": "user", "metadata": "not json"}).allowed


def test_a_deck_without_sign_in_is_its_one_trusted_user(monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    grant = crew_access.grant_of({"id": "local", "role": "admin"})
    assert grant.allowed and grant.why == "local" and grant.trusted


def test_the_deck_can_turn_it_off(monkeypatch):
    monkeypatch.setenv("FD_CREW", "off")
    assert not crew_access.grant_of(_row("admin")).allowed


def test_approving_names_at_least_one_cli():
    with pytest.raises(ValueError):
        crew_access.clean_entry({"approved": True, "harnesses": []}, admin_id="a")
    assert crew_access.clean_entry({"approved": False}, admin_id="a") is None
    first = crew_access.clean_entry({"approved": True, "harnesses": ["pi"]}, admin_id="a")
    assert first["by"] == "a" and first["at"]
    # Changing what's approved doesn't rewrite who approved it first.
    later = crew_access.clean_entry({"approved": True, "harnesses": ["codex"]}, admin_id="b", previous=first)
    assert later["by"] == "a" and later["at"] == first["at"] and later["harnesses"] == ["codex"]


# ── The boundary ─────────────────────────────────────────────────────────


ROUTES = [
    ("GET", "/fd/crew/repos", None), ("GET", "/fd/crew/tasks", None),
    ("POST", "/fd/crew/tasks", {"harness": "claude", "brief": "x", "project": "p"}),
    ("GET", "/fd/crew/tasks/crw_x", None), ("GET", "/fd/crew/tasks/crw_x/events", None),
    ("GET", "/fd/crew/tasks/crw_x/diff", None), ("POST", "/fd/crew/tasks/crw_x/steer", {"text": "x"}),
    ("POST", "/fd/crew/tasks/crw_x/pr", None), ("POST", "/fd/crew/tasks/crw_x/merge", None),
    ("POST", "/fd/crew/tasks/crw_x/close", None), ("PUT", "/fd/crew/logins/claude", {"secret": "x"}),
    ("DELETE", "/fd/crew/logins/claude", None),
]


async def test_every_route_refuses_someone_who_is_not_approved(deck):
    async with _client(deck.user) as client:
        for method, path, body in ROUTES:
            resp = await client.request(method, path, json=body)
            assert resp.status_code == 403, f"{method} {path}: {resp.status_code}"
            assert "approved" in resp.json()["detail"]
        me = await client.get("/fd/crew/me")
    assert me.status_code == 200
    assert me.json()["allowed"] is False
    assert "harnesses" not in me.json()


async def test_only_an_admin_can_approve(deck):
    async with _client(deck.user) as client:
        resp = await client.put(f"/fd/admin/users/{deck.user['id']}", json={
            "crew": {"approved": True, "harnesses": ["claude"]}})
    assert resp.status_code == 403
    fresh = await deck.db.get_user_by_id(deck.user["id"])
    assert not crew_access.grant_of(fresh).allowed


async def test_approval_is_per_user_and_shows_at_once(deck):
    await _approve(deck, deck.user, harnesses=["claude", "pi"])
    async with _client(deck.user) as client:
        me = (await client.get("/fd/crew/me")).json()
    assert me["allowed"] and me["trusted"] is False
    assert {h["id"] for h in me["harnesses"]} == {"claude", "pi"}
    assert me["max_parallel"] == 2
    async with _client(deck.other) as client:
        assert (await client.get("/fd/crew/me")).json()["allowed"] is False
    stored = crew_access.stored(await deck.db.get_user_by_id(deck.user["id"]))
    assert stored["by"] == deck.admin["id"]


async def test_an_empty_approval_is_refused(deck):
    async with _client(deck.admin) as admin:
        resp = await admin.put(f"/fd/admin/users/{deck.user['id']}", json={"crew": {"approved": True, "harnesses": []}})
    assert resp.status_code == 400


async def test_withdrawing_stops_their_workers_and_closes_the_door(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        assert (await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})).status_code == 200
        made = await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:slow", "project": "p"})
    assert made.status_code == 201, made.text
    assert made.json()["status"] == "running"
    async with _client(deck.admin) as admin:
        await admin.put(f"/fd/admin/users/{deck.user['id']}", json={"crew": {"approved": False}})
    assert crew_store.get_store().get(deck.user["id"], made.json()["id"])["status"] == "stopped"
    async with _client(deck.user) as client:
        assert (await client.get("/fd/crew/tasks")).status_code == 403
        # Their own worker they can always stop, though.
        assert (await client.post(f"/fd/crew/tasks/{made.json()['id']}/stop")).status_code == 200


async def test_narrowing_the_approval_stops_only_what_it_no_longer_covers(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:slow", "project": "p"})).json()
    await _approve(deck, deck.user, harnesses=["pi"])
    assert crew_store.get_store().get(deck.user["id"], made["id"])["status"] == "stopped"


async def test_demoting_an_admin_stops_their_workers(deck):
    repo = deck.tmp / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q", "."], cwd=repo, check=True)
    subprocess.run([*GIT, "commit", "-q", "--allow-empty", "-m", "base"], cwd=repo, check=True)
    second = await deck.db.create_user("a2@example.com", "x", role="admin")
    async with _client(second) as client:
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:slow",
                                                          "repo_path": str(repo)})).json()
    assert made["status"] == "running"
    async with _client(deck.admin) as admin:
        await admin.put(f"/fd/admin/users/{second['id']}", json={"role": "user"})
    assert crew_store.get_store().get(second["id"], made["id"])["status"] == "stopped"


async def test_deleting_a_user_stops_their_workers_and_removes_their_folder(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:slow", "project": "p"})).json()
    task = crew_store.get_store().get(deck.user["id"], made["id"])
    await crew_runner.forget_user(deck.user["id"])
    assert not crew_runner._alive(task)
    assert not (crew_store.root() / crew_store.safe_uid(deck.user["id"])).exists()


async def test_one_users_tasks_are_nothing_to_another(deck):
    await _approve(deck, deck.user)
    await _approve(deck, deck.other)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "project": "p"})).json()
    await _settle(deck.user["id"], made["id"])
    async with _client(deck.other) as client:
        assert (await client.get("/fd/crew/tasks")).json()["tasks"] == []
        for method, suffix in (("GET", ""), ("GET", "/events"), ("GET", "/diff"), ("POST", "/merge"),
                               ("POST", "/close"), ("POST", "/stop")):
            resp = await client.request(method, f"/fd/crew/tasks/{made['id']}{suffix}")
            assert resp.status_code == 404, suffix


async def test_a_worktree_path_is_not_shown_to_a_non_admin(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:nothing", "project": "p"})
        got = await client.get(f"/fd/crew/tasks/{made.json()['id']}")
    assert "worktree" not in made.json()
    assert "worktree" not in got.json()


# ── Logins ───────────────────────────────────────────────────────────────


async def test_logins_are_sealed_and_never_come_back(deck):
    await _approve(deck, deck.user)
    secret = "sk-ant-oat01-very-secret-value"
    async with _client(deck.user) as client:
        saved = await client.put("/fd/crew/logins/claude", json={"secret": secret})
        assert saved.status_code == 200
        assert secret not in saved.text
        me = await client.get("/fd/crew/me")
        assert secret not in me.text
        assert me.json()["logins"]["claude"]["hint"] == "…alue"
        assert next(h for h in me.json()["harnesses"] if h["id"] == "claude")["login"]["ok"]
        assert (await client.put("/fd/crew/logins/nonsense", json={"secret": "x"})).status_code == 404
    raw = await deck.db.get_setting(deck.user["id"], "crew:logins")
    assert secret not in raw
    async with _client(deck.user) as client:
        assert (await client.delete("/fd/crew/logins/claude")).status_code == 200
        me = await client.get("/fd/crew/me")
    assert next(h for h in me.json()["harnesses"] if h["id"] == "claude")["login"]["ok"] is False


def test_crew_settings_are_server_owned():
    assert settings_routes._server_owned("crew:logins")
    assert settings_routes._server_owned("crew:meta")


# ── Agents ───────────────────────────────────────────────────────────────


@pytest.fixture
def as_agent(monkeypatch):
    """Requests from the agent of whoever ``state['owner']`` names."""
    state = {"owner": ""}

    def caller(request, *, what):
        return agent_identity.AgentCaller(slug="helper", owner=state["owner"])

    monkeypatch.setattr(agent_identity, "require_agent_caller", caller)
    return state


async def _agent_task(deck, as_agent, brief="SCENARIO:work") -> dict:
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
    as_agent["owner"] = deck.user["id"]
    async with _client(None) as agent:
        made = await agent.post("/fd/crew/agent/dispatch", json={"harness": "claude", "brief": brief, "project": "p"})
    assert made.status_code == 200, made.text
    return made.json()


async def test_an_agent_acts_for_its_owner_only_while_they_are_approved(deck, as_agent):
    as_agent["owner"] = deck.user["id"]
    async with _client(None) as agent:
        assert (await agent.get("/fd/crew/agent/status")).json() == {"usable": False}
        refused = await agent.post("/fd/crew/agent/dispatch", json={"harness": "claude", "brief": "x", "project": "p"})
        assert refused.status_code == 403
    made = await _agent_task(deck, as_agent)
    assert made["from_agent"] == "helper"
    async with _client(None) as agent:
        assert (await agent.get("/fd/crew/agent/status")).json() == {"usable": True}
    as_agent["owner"] = deck.other["id"]
    async with _client(None) as agent:
        assert (await agent.post("/fd/crew/agent/task", json={"task_id": made["id"]})).status_code == 403


async def test_a_browser_is_not_an_agent(deck):
    async with _client(None) as client:
        resp = await client.get("/fd/crew/agent/status", headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403


async def test_an_agent_has_no_way_to_merge_or_discard(deck, as_agent):
    made = await _agent_task(deck, as_agent)
    await _settle(deck.user["id"], made["id"])
    async with _client(None) as agent:
        for path in ("merge", "close", "discard"):
            resp = await agent.post(f"/fd/crew/agent/{path}", json={"task_id": made["id"]})
            assert resp.status_code in (404, 405), path
    assert "merge" not in {p["enum"][0] for p in [CrewTool.parameters["properties"]["action"]]}
    assert not {"merge", "close", "discard"} & set(CrewTool.parameters["properties"]["action"]["enum"])


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    store = autonomy.AutonomyStore(tmp_path / "autonomy.db")
    monkeypatch.setattr(autonomy, "_store", store, raising=False)
    monkeypatch.setattr(autonomy, "get_store", lambda: store)
    monkeypatch.setattr(fd_dispatch, "get_store", lambda: store)
    from captain_claw.flight_deck import owner_notify

    async def quiet(user_id, action):
        return {}

    monkeypatch.setattr(owner_notify, "notify_proposal", quiet)
    return store


async def test_a_merge_an_agent_asks_for_waits_for_the_persons_approve(deck, as_agent, ledger):
    made = await _agent_task(deck, as_agent)
    done = await _settle(deck.user["id"], made["id"])
    async with _client(None) as agent:
        asked = await agent.post("/fd/crew/agent/request_merge", json={"task_id": made["id"]})
        again = await agent.post("/fd/crew/agent/request_merge", json={"task_id": made["id"]})
    assert asked.json()["status"] == "waiting_for_approval"
    assert again.json()["status"] == "already_waiting"
    waiting = ledger.waiting_actions(deck.user["id"])
    assert len(waiting) == 1
    action = waiting[0]
    assert action["kind"] == "crew_merge" and action["status"] == "awaiting_approval"
    # Nothing's merged, and a later automatic pass can't merge it either.
    assert crew_store.get_store().get(deck.user["id"], made["id"])["status"] == "ready"
    auto = await fd_dispatch.dispatch_action(deck.user["id"], action)
    assert auto["ok"] is False
    assert crew_store.get_store().get(deck.user["id"], made["id"])["status"] == "ready"
    # The Approve.
    out = await fd_dispatch.dispatch_action(deck.user["id"], action, approved_by_human=True)
    assert out["ok"] is True
    fresh = crew_store.get_store().get(deck.user["id"], made["id"])
    assert fresh["status"] == "delivered"
    assert not Path(done["worktree"]).exists()
    assert ledger.get_action(action["id"])["status"] == "done"


async def test_an_approve_covers_only_the_commits_it_was_asked_about(deck, as_agent, ledger):
    made = await _agent_task(deck, as_agent)
    await _settle(deck.user["id"], made["id"])
    async with _client(None) as agent:
        await agent.post("/fd/crew/agent/request_merge", json={"task_id": made["id"]})
    action = ledger.waiting_actions(deck.user["id"])[0]
    # The agent has the worker add something after asking.
    async with _client(None) as agent:
        await agent.post("/fd/crew/agent/steer", json={"task_id": made["id"], "text": "one more commit"})
    done = await _settle(deck.user["id"], made["id"])
    assert done["evidence"]["commits"] > 1
    out = await fd_dispatch.dispatch_action(deck.user["id"], action, approved_by_human=True)
    assert "changed since" in out["note"]
    assert crew_store.get_store().get(deck.user["id"], made["id"])["status"] == "ready"
    assert ledger.get_action(action["id"])["status"] == "failed"
    # Asking again asks about what's there now.
    async with _client(None) as agent:
        again = await agent.post("/fd/crew/agent/request_merge", json={"task_id": made["id"]})
    assert again.json()["status"] == "waiting_for_approval"
    fresh = ledger.get_action(again.json()["action_id"])
    assert fresh["payload"]["head"] == done["evidence"]["head"]
    out = await fd_dispatch.dispatch_action(deck.user["id"], fresh, approved_by_human=True)
    assert crew_store.get_store().get(deck.user["id"], made["id"])["status"] == "delivered", out


async def test_asking_again_after_new_commits_replaces_the_old_question(deck, as_agent, ledger):
    made = await _agent_task(deck, as_agent)
    await _settle(deck.user["id"], made["id"])
    async with _client(None) as agent:
        first = await agent.post("/fd/crew/agent/request_merge", json={"task_id": made["id"]})
        await agent.post("/fd/crew/agent/steer", json={"task_id": made["id"], "text": "one more commit"})
    await _settle(deck.user["id"], made["id"])
    async with _client(None) as agent:
        second = await agent.post("/fd/crew/agent/request_merge", json={"task_id": made["id"]})
    assert second.json()["action_id"] != first.json()["action_id"]
    waiting = ledger.waiting_actions(deck.user["id"])
    assert [a["id"] for a in waiting] == [second.json()["action_id"]]


async def test_an_approve_cannot_merge_for_someone_no_longer_approved(deck, as_agent, ledger):
    made = await _agent_task(deck, as_agent)
    await _settle(deck.user["id"], made["id"])
    async with _client(None) as agent:
        await agent.post("/fd/crew/agent/request_merge", json={"task_id": made["id"]})
    action = ledger.waiting_actions(deck.user["id"])[0]
    async with _client(deck.admin) as admin:
        await admin.put(f"/fd/admin/users/{deck.user['id']}", json={"crew": {"approved": False}})
    out = await fd_dispatch.dispatch_action(deck.user["id"], action, approved_by_human=True)
    assert out["ok"] is True
    assert "approved" in out["note"]
    assert crew_store.get_store().get(deck.user["id"], made["id"])["status"] == "ready"
    assert ledger.get_action(action["id"])["status"] == "failed"


async def test_a_person_can_merge_from_the_page(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "project": "p"})).json()
    await _settle(deck.user["id"], made["id"])
    async with _client(deck.user) as client:
        events = (await client.get(f"/fd/crew/tasks/{made['id']}/events")).json()["events"]
        assert events and events[0]["n"] == 1
        later = (await client.get(f"/fd/crew/tasks/{made['id']}/events", params={"after": 1})).json()["events"]
        assert later[0]["n"] == 2
        diff = (await client.get(f"/fd/crew/tasks/{made['id']}/diff")).json()
        assert "work.txt" in diff["committed"]
        merged = await client.post(f"/fd/crew/tasks/{made['id']}/merge")
    assert merged.status_code == 200, merged.text
    assert merged.json()["status"] == "delivered"


# ── Projects and results ──────────────────────────────────────────────────


async def test_a_project_can_be_made_before_any_task_and_shows_its_vfs_path(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        made = await client.post("/fd/crew/projects", json={"name": "Game"})
        assert made.status_code == 201, made.text
        assert made.json() == {"project": "Game", "key": "project:Game", "path": "vfs:Game/Game/"}
        again = await client.post("/fd/crew/projects", json={"name": "Game"})
        assert again.status_code == 201
        repos = (await client.get("/fd/crew/repos")).json()
    assert repos["projects"] == [{"project": "Game", "key": "project:Game", "folders": ["Game"],
                                  "paths": {"Game": "vfs:Game/Game/"}}]


async def test_a_task_shows_its_project_and_its_files_and_a_link_comes_from_the_clean_copy(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:page", "project": "Game"})).json()
        assert made["project_key"] == "project:Game" and made["project_name"] == "Game"
    await _settle(deck.user["id"], made["id"])
    async with _client(deck.user) as client:
        listed = (await client.get("/fd/crew/tasks")).json()["tasks"][0]
        assert listed["results"]["publish"]["project"] == "crew"
        files = (await client.post(f"/fd/crew/tasks/{made['id']}/files")).json()
        assert "site/index.html" in files["files"] and files["project_path"] == "vfs:Game/Game/"
        shown = (await client.get(f"/fd/crew/tasks/{made['id']}/file", params={"path": "site/index.html"})).json()
        assert shown["text"].startswith("<html>") and not shown["binary"]
        link = await client.post(f"/fd/crew/tasks/{made['id']}/link")
        assert link.status_code == 200, link.text
        assert link.json()["name"] == f"game-{made['id'][-4:]}" and link.json()["url"]
        assert (await client.post(f"/fd/crew/tasks/{made['id']}/link")).json() == link.json()
    from captain_claw.flight_deck import vfs_hosting

    entry = vfs_hosting.load_registry()[link.json()["name"]]
    assert entry["owner"] == deck.user["id"] and entry["project"] == "crew"
    assert vfs_hosting.resolve_static_file(entry, "index.html").read_text().startswith("<html>")
    # Someone else's task doesn't exist, here as everywhere.
    await _approve(deck, deck.other)
    async with _client(deck.other) as client:
        for call in (client.post(f"/fd/crew/tasks/{made['id']}/files"), client.post(f"/fd/crew/tasks/{made['id']}/link"),
                     client.get(f"/fd/crew/tasks/{made['id']}/file", params={"path": "site/index.html"})):
            assert (await call).status_code == 404


async def test_no_index_means_no_page_to_link(deck):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "project": "Game"})).json()
    await _settle(deck.user["id"], made["id"])
    async with _client(deck.user) as client:
        assert (await client.post(f"/fd/crew/tasks/{made['id']}/link")).status_code == 409


def test_a_published_folder_never_serves_a_repository(tmp_path, monkeypatch):
    from captain_claw.flight_deck import vfs_hosting

    site = tmp_path / "site"
    (site / ".git").mkdir(parents=True)
    (site / ".git" / "config").write_text("secret")
    (site / ".env").write_text("secret")
    (site / "index.html").write_text("page")
    monkeypatch.setattr(vfs_hosting, "entry_dir", lambda entry: site)
    assert vfs_hosting.resolve_static_file({}, "index.html") == (site / "index.html").resolve()
    for path in (".git/config", ".git", "./.git/config", ".env", "a/../.git/config"):
        assert vfs_hosting.resolve_static_file({}, path) is None, path


async def test_tidying_is_for_the_owner_of_the_tasks(deck):
    await _approve(deck, deck.user)
    await _approve(deck, deck.other)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:nothing", "kind": "scout", "project": "Game"})).json()
    await _settle(deck.user["id"], made["id"])
    async with _client(deck.other) as client:
        assert (await client.post("/fd/crew/projects/tidy", json={"key": "project:Game"})).json() == {"cleared": 0, "kept": 0}
    async with _client(deck.user) as client:
        assert (await client.post("/fd/crew/projects/tidy", json={"key": "project:Game"})).json() == {"cleared": 1, "kept": 0}
        assert (await client.get("/fd/crew/tasks")).json()["tasks"] == []


class _FakeClient:
    """What the tool asks Flight Deck, answered."""

    def __init__(self, **answers):
        self.answers = answers
        self.asked: list[tuple[str, dict | None]] = []

    async def call(self, method, path, body=None):
        self.asked.append((path, body))
        return 200, self.answers[path]


async def test_the_tool_points_at_the_vfs_rather_than_at_a_worker():
    options = _FakeClient(options={"me": {"harnesses": []}, "repos": {"projects": [
        {"project": "Game", "folders": ["client"], "paths": {"client": "vfs:Game/client/"}}]}})
    out = await CrewTool()._do_options(options)
    assert "client (vfs:Game/client/)" in out.content and "read it yourself" in out.content
    files = _FakeClient(files={"files_path": "vfs:crew/Game/t/files/", "count": 2, "whole": True, "files": ["a", "b"],
                               "report": "vfs:crew/Game/t/report.md", "landed": True, "project_path": "vfs:Game/client/",
                               "publish": {"project": "crew", "subdir": "Game/t/files"}, "uncommitted": 1})
    out = await CrewTool()._do_files(files, task_id="t", whole=True)
    assert files.asked == [("files", {"task_id": "t", "whole": True})]
    for piece in ("vfs:Game/client/", "vfs:crew/Game/t/files/", "vfs:crew/Game/t/report.md", "subdir=Game/t/files",
                  "1 uncommitted"):
        assert piece in out.content, piece


async def test_the_tool_warns_of_work_a_new_task_will_not_see():
    client = _FakeClient(dispatch={"id": "t", "status": "running", "harness_label": "Claude Code", "repo_label": "Game/client",
                                   "branch": "crew/x", "base_branch": "main", "unlanded": ["Add the board"]})
    out = await CrewTool()._do_dispatch(client, harness="claude", brief="look")
    assert "starts from main" in out.content and "Add the board" in out.content and "end your turn" in out.content
    tasks = _FakeClient(tasks={"tasks": [{"id": "a", "project_name": "Game", "title": "one"},
                                         {"id": "b", "project_name": "Other", "title": "two"}]})
    out = await CrewTool()._do_list(tasks, project="game")
    assert "one" in out.content and "two" not in out.content


# ── The tool ─────────────────────────────────────────────────────────────


async def test_the_registry_shows_the_tool_only_while_approved(monkeypatch):
    from captain_claw import crew_client
    from captain_claw.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(CrewTool(), metadata={"requires_crew": True})
    monkeypatch.setattr(crew_client, "_USABLE", (False, 0.0))
    assert "crew" not in {t["name"] for t in registry.get_definitions()}
    monkeypatch.setattr(crew_client, "_USABLE", (True, time.time()))
    assert "crew" in {t["name"] for t in registry.get_definitions()}
    # A stale yes is a no.
    monkeypatch.setattr(crew_client, "_USABLE", (True, time.time() - 10_000))
    assert "crew" not in {t["name"] for t in registry.get_definitions()}


def test_eco_mode_keeps_the_tool_for_those_the_registry_shows_it_to():
    from captain_claw.agent_orchestration_mixin import _ECO_CORE_TOOLS

    assert "crew" in _ECO_CORE_TOOLS


async def test_the_tool_refuses_a_shared_agents_member_and_workers(monkeypatch):
    from captain_claw import crew_client

    monkeypatch.setattr(crew_client, "under_flight_deck", lambda: True)
    monkeypatch.setattr(crew_client, "member_call", lambda: True)
    out = await CrewTool().execute(action="list")
    assert not out.success and "member" in out.error
    monkeypatch.setattr(crew_client, "member_call", lambda: False)
    monkeypatch.setenv("CLAW_CODE_AGENT", "1")
    out = await CrewTool().execute(action="list")
    assert not out.success and "inside" in out.error


async def test_the_tool_asks_rather_than_merges(monkeypatch):
    from captain_claw import crew_client

    calls = []

    async def call(method, path, body=None):
        calls.append((method, path, body))
        return 200, {"status": "waiting_for_approval"}

    monkeypatch.setattr(crew_client, "under_flight_deck", lambda: True)
    monkeypatch.setattr(crew_client, "member_call", lambda: False)
    monkeypatch.setattr(crew_client, "call", call)
    monkeypatch.delenv("CLAW_CODE_AGENT", raising=False)
    out = await CrewTool().execute(action="request_merge", task_id="crw_1")
    assert out.success and "isn't merged" in out.content
    assert calls == [("POST", "request_merge", {"task_id": "crw_1"})]
    refused = await CrewTool().execute(action="merge", task_id="crw_1")
    assert not refused.success


def test_the_tool_counts_as_running_code():
    from captain_claw.flight_deck.tool_risk import risk_for

    assert risk_for("crew", {"action": "dispatch"}) == "exec"
    assert risk_for("crew", {"action": "status"}) == "read"
    assert risk_for("crew", {"action": "list"}) == "read"
