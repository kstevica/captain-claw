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
from fastapi import FastAPI, HTTPException

from captain_claw.flight_deck import admin_routes, agent_identity
from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import autonomy, crew_access, crew_logins, crew_routes, crew_runner, crew_store, fd_crypto, fd_dispatch
from captain_claw.flight_deck import settings_routes
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.tools.crew import CrewTool

from .test_crew_runner import FAKE_CLI

LOOPBACK = ("127.0.0.1", 40001)
GIT = ["git", "-c", "user.name=T", "-c", "user.email=t@example.com"]


@pytest.fixture
async def deck(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Crew asks a CLI its version from here: never from a real checkout
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
    # A lock belongs to the loop it was first waited on in, and each test has its own.
    monkeypatch.setattr(crew_runner, "_QUEUE_LOCK", asyncio.Lock())
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    monkeypatch.setattr(auth_mod, "_db", db)
    admin = await db.create_user("admin@example.com", "x", role="admin")
    user = await db.create_user("u@example.com", "x", role="user")
    other = await db.create_user("o@example.com", "x", role="user")
    yield types.SimpleNamespace(db=db, admin=admin, user=user, other=other, tmp=tmp_path)
    jobs = list(crew_runner._PREPARING.values())
    for job in jobs:
        job.cancel()
    await asyncio.gather(*jobs, return_exceptions=True)
    crew_runner._PREPARING.clear()
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


async def test_a_task_link_can_be_named_and_follows_the_site(deck):
    from captain_claw.flight_deck import vfs_hosting

    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:page", "project": "Game"})).json()
    await _settle(deck.user["id"], made["id"])
    url = f"/fd/crew/tasks/{made['id']}"
    registry = vfs_hosting.load_registry()
    registry["taken"] = {"kind": "static", "owner": deck.other["id"], "project": "x"}
    vfs_hosting.save_registry(registry)
    async with _client(deck.user) as client:
        assert (await client.post(f"{url}/link", json={"name": "Not A Name!"})).status_code == 400
        assert (await client.post(f"{url}/link", json={"name": "taken"})).status_code == 409
        named = (await client.post(f"{url}/link", json={"name": "my-game"})).json()
        assert named == {"name": "my-game", "url": "/vfs/my-game/", "stopped": False}
        assert (await client.post(f"{url}/files")).json()["link"]["stopped"] is False

        # Stopped from Live sites: the task says so, and keeps the address.
        vfs_hosting.stop_app("my-game")
        assert (await client.post(f"{url}/files")).json()["link"] == {"name": "my-game", "url": "/vfs/my-game/", "stopped": True}
        assert (await client.post(f"{url}/link")).json()["stopped"] is True

        # Removed: the task no longer claims a link, and can make a new one.
        registry = vfs_hosting.load_registry()
        del registry["my-game"]
        vfs_hosting.save_registry(registry)
        assert "link" not in (await client.post(f"{url}/files")).json()
        assert (await client.post(f"{url}/link", json={"name": "again"})).json()["name"] == "again"


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


async def test_the_tool_says_a_push_is_not_a_workers_job():
    made = {"id": "t", "status": "running", "harness_label": "Claude Code", "repo_label": "p", "branch": "crew/x"}
    # A brief that asks for a pull request gets one from Captain.
    for brief in ("Add it and open a pull request.", "Add it, then create a PR.", "Commit, then run gh pr create."):
        client = _FakeClient(dispatch=made)
        out = await CrewTool()._do_dispatch(client, harness="claude", brief=brief)
        assert client.asked[0][1]["deliver"] == "pr", brief
        assert "Captain opens its pull request" in out.content
    # Mentioning one isn't asking for one, and pushing isn't either.
    for brief in ("Fix what the review of pull request 12 found.", "Add a dark theme."):
        client = _FakeClient(dispatch=made)
        await CrewTool()._do_dispatch(client, harness="claude", brief=brief)
        assert "deliver" not in client.asked[0][1], brief
    pushing = await CrewTool()._do_dispatch(_FakeClient(dispatch=made), harness="claude", brief="Add it, then git push it.")
    assert "crew action=pr" in pushing.content
    # A scout changes nothing, so it has nothing to open one for.
    client = _FakeClient(dispatch=made)
    await CrewTool()._do_dispatch(client, harness="claude", kind="scout", brief="How would you open a pull request here?")
    assert "deliver" not in client.asked[0][1]
    for text in ("Push branch crew/x, then gh pr create.", "GH_TOKEN is set now, try again", "git push origin crew/x"):
        out = await CrewTool()._do_steer(_FakeClient(steer={"id": "t"}), task_id="t", text=text)
        assert "crew action=pr" in out.content, text
    out = await CrewTool()._do_steer(_FakeClient(steer={"id": "t"}), task_id="t", text="Use the blue one.")
    assert "crew action=pr" not in out.content


async def test_usage_is_each_users_own_and_a_new_login_forgets_the_old_account(deck):
    await _approve(deck, deck.user)
    await _approve(deck, deck.other)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "project": "Game"})).json()
    await _settle(deck.user["id"], made["id"])
    async with _client(deck.user) as client:
        row = (await client.get("/fd/crew/usage")).json()["harnesses"][0]
        assert row["model"] == "claude-test-model" and row["limits"]["seven_day"]["used"] == 0.44
        assert (await client.get(f"/fd/crew/tasks/{made['id']}")).json()["model_used"] == "claude-test-model"
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-another-account"})
        row = (await client.get("/fd/crew/usage")).json()["harnesses"][0]
        assert row["limits"] == {} and row["model"] == "" and row["tasks"] == 1
    async with _client(deck.other) as client:
        assert (await client.get("/fd/crew/usage")).json() == {"harnesses": []}


@pytest.fixture
def opencode_too(deck, monkeypatch):
    """A stand-in OpenCode beside the stand-in Claude: it records how it's run."""
    fake = deck.tmp / "bin" / "opencode"
    fake.write_text(r"""#!/bin/sh
case "$1" in --version) echo "stand-in 1.0"; exit 0;; esac
case "$*" in *--help*) exit 0;; esac
if [ "$1" = "models" ]; then printf 'opencode/deepseek-v4.1-flash-free\nopencode/big-pickle\n'; exit 0; fi
printf '%s\n' "$*" >> "$OC_LOG"
echo work > free.txt; git add free.txt; git -c user.name=T -c user.email=t@example.com commit -qm free
printf '%s\n' '{"type":"text","sessionID":"s1","part":{"type":"text","text":"Done.\n\nCREW-STATUS: done made it"}}'
""")
    fake.chmod(0o755)
    log = deck.tmp / "oc.log"
    monkeypatch.setenv("FD_CREW_OPENCODE_BIN", str(fake))
    monkeypatch.setattr(crew_runner, "_MODELS", {})
    real = crew_runner.crew_logins.worker_env

    async def with_log(*args, **kwargs):
        return {**await real(*args, **kwargs), "OC_LOG": str(log)}

    monkeypatch.setattr(crew_runner.crew_logins, "worker_env", with_log)
    return log


async def _approve_opencode(deck, user):
    async with _client(deck.admin) as client:
        made = await client.put(f"/fd/admin/users/{user['id']}", json={"crew": {"approved": True, "harnesses": ["opencode"], "max_parallel": 2}})
    assert made.status_code == 200, made.text


async def test_opencode_needs_no_key_but_someones_code_needs_a_chosen_model(deck, opencode_too):
    await _approve_opencode(deck, deck.user)
    brief = {"harness": "opencode", "brief": "Add it.", "project": "Game"}
    async with _client(deck.user) as client:
        h = (await client.get("/fd/crew/me")).json()["harnesses"][0]
        assert h["login"] == {"ok": True, "how": "OpenCode's free models", "free": True}
        assert h["default_model"] == ""
        assert (await client.get("/fd/crew/models")).json()["opencode"] == ["opencode/deepseek-v4.1-flash-free", "opencode/big-pickle"]
        # Left to itself it would pick one: not with someone's repository.
        refused = await client.post("/fd/crew/tasks", json=brief)
        assert refused.status_code == 400 and "opencode.ai/docs/zen" in refused.text
        assert not opencode_too.exists()
        # Named on the task.
        named = await client.post("/fd/crew/tasks", json={**brief, "model": "opencode/big-pickle"})
        assert named.status_code == 201, named.text
    done = await _settle(deck.user["id"], named.json()["id"])
    assert done["status"] == "ready" and done["evidence"]["commits"] == 1 and done["model_used"] == "opencode/big-pickle"
    assert "--model opencode/big-pickle" in opencode_too.read_text()
    async with _client(deck.user) as client:
        # Chosen as the default: what the person picked, wherever they picked it.
        assert (await client.put("/fd/crew/models/opencode", json={"model": "opencode/deepseek-v4.1-flash-free"})).status_code == 200
        assert (await client.get("/fd/crew/me")).json()["harnesses"][0]["default_model"] == "opencode/deepseek-v4.1-flash-free"
        chosen = await client.post("/fd/crew/tasks", json=brief)
        assert chosen.status_code == 201, chosen.text
        assert chosen.json()["model"] == "opencode/deepseek-v4.1-flash-free"
    await _settle(deck.user["id"], chosen.json()["id"])
    assert "--model opencode/deepseek-v4.1-flash-free" in opencode_too.read_text()
    async with _client(deck.user) as client:
        row = (await client.get("/fd/crew/usage")).json()["harnesses"][0]
        assert row["model"] == "opencode/deepseek-v4.1-flash-free"
        # A task can still name another, and clearing the default brings the question back.
        await client.put("/fd/crew/models/opencode", json={"model": ""})
        assert (await client.post("/fd/crew/tasks", json=brief)).status_code == 400


async def test_a_model_opencode_doesnt_know_is_refused_before_it_costs_a_worktree(deck, opencode_too, monkeypatch):
    # From a real log: a made-up provider, and only "Unexpected server error" four times over.
    await _approve_opencode(deck, deck.user)
    brief = {"harness": "opencode", "brief": "Add it.", "project": "Game"}
    async with _client(deck.user) as client:
        refused = await client.post("/fd/crew/tasks", json={**brief, "model": "homeserver/qwen"})
        assert refused.status_code == 400
        detail = refused.json()["detail"]
        assert "homeserver/qwen" in detail and "opencode/deepseek-v4.1-flash-free" in detail
        assert "Your own model" in detail  # why one that works in their terminal isn't here
        assert (await client.get("/fd/crew/tasks")).json()["tasks"] == []
        assert not opencode_too.exists()
        # Just set up: the list is asked again rather than read from the cache.
        real = crew_runner.crew_harness.list_models
        monkeypatch.setattr(crew_runner.crew_harness, "list_models", lambda *a: [*real(*a), "home/qwen"])
        added = await client.post("/fd/crew/tasks", json={**brief, "model": "home/qwen"})
        assert added.status_code == 201, added.text
    await _settle(deck.user["id"], added.json()["id"])


async def test_the_page_doesnt_wait_for_a_cli_to_list_its_models(deck, opencode_too, monkeypatch):
    await _approve_opencode(deck, deck.user)
    asked = []
    monkeypatch.setattr(crew_runner.crew_harness, "list_models", lambda *a: asked.append(a) or ["opencode/a-free"])
    async with _client(deck.user) as client:
        me = (await client.get("/fd/crew/me")).json()
        assert not asked and me["harnesses"][0]["models"] == []
        assert (await client.get("/fd/crew/models")).json() == {"opencode": ["opencode/a-free"]}
        assert len(asked) == 1
        # Known now, so it's on the page straight away, with nobody asked again.
        assert (await client.get("/fd/crew/me")).json()["harnesses"][0]["models"] == ["opencode/a-free"]
        assert len(asked) == 1
    async with _client(deck.other) as client:
        assert (await client.get("/fd/crew/models")).status_code == 403


async def test_a_task_runs_on_opencode_2_while_someone_elses_service_holds_its_port(deck, opencode_too, monkeypatch):
    """OpenCode 2 as it is on a computer where someone already uses it: no --dir, one port for its
    background service, and a worker's own home can't use the service that's running."""
    fake = deck.tmp / "bin" / "opencode"
    fake.write_text(r"""#!/bin/sh
case "$*" in *--help*) printf 'FLAGS\n  --auto\nGLOBAL FLAGS\n  --standalone  Run with a private server\n'; exit 0;; esac
case "$*" in
  *--standalone*) ;;
  *) echo 'Error: Managed service port 49374 on 127.0.0.1 is already in use' >&2; exit 1;;
esac
if [ "$1" = "models" ]; then printf 'opencode/deepseek-v4.1-flash-free\n'; exit 0; fi
case "$*" in *--dir*|*--variant*) echo USAGE >&2; exit 1;; esac
printf '%s\n' "$*" >> "$OC_LOG"
pwd >> "$OC_LOG"
echo work > free.txt; git add free.txt; git -c user.name=T -c user.email=t@example.com commit -qm free
printf '%s\n' '{"type":"text","sessionID":"s1","part":{"type":"text","text":"Done.\n\nCREW-STATUS: done made it"}}'
""")
    await _approve_opencode(deck, deck.user)
    model = "opencode/deepseek-v4.1-flash-free"
    async with _client(deck.user) as client:
        assert (await client.get("/fd/crew/models")).json() == {"opencode": [model]}
        made = await client.post("/fd/crew/tasks", json={"harness": "opencode", "brief": "Add it.", "project": "Game", "model": model, "effort": "high"})
        assert made.status_code == 201, made.text
    done = await _settle(deck.user["id"], made.json()["id"])
    assert done["status"] == "ready", done.get("note")
    said = opencode_too.read_text()
    assert "--standalone" in said and f"--model {model}#high" in said
    assert Path(said.splitlines()[-1]).resolve() == Path(done["worktree"]).resolve()  # where it worked, with no --dir


async def test_opencode_that_cant_list_its_models_isnt_stopped_by_that(deck, opencode_too, monkeypatch):
    await _approve_opencode(deck, deck.user)
    monkeypatch.setattr(crew_runner.crew_harness, "list_models", lambda *a: [])
    async with _client(deck.user) as client:
        made = await client.post("/fd/crew/tasks", json={"harness": "opencode", "brief": "Add it.", "project": "Game", "model": "home/anything"})
        assert made.status_code == 201, made.text
    await _settle(deck.user["id"], made.json()["id"])


async def _own_model_server(seen):
    """A real little OpenAI-style server on a local port: what it serves, and who asked."""
    async def handle(reader, writer):
        request = (await reader.read(4096)).decode("latin-1")
        line = request.split("\r\n")[0]
        auth = next((h.split(": ", 1)[1] for h in request.split("\r\n") if h.lower().startswith("authorization:")), "")
        seen.append((line, auth))
        if "/v1/models" in line and auth == "Bearer the-right-key":
            body, status = json.dumps({"data": [{"id": "qwen:3"}, {"id": "bad name!"}]}).encode(), "200 OK"
        elif "/v1/models" in line:
            body, status = b"{}", "401 Unauthorized"
        else:
            body, status = b"{}", "404 Not Found"
        writer.write(f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"


async def test_your_own_model_runs_on_a_sealed_key_that_never_comes_back(deck, opencode_too, monkeypatch):
    await _approve_opencode(deck, deck.user)
    await _approve_opencode(deck, deck.other)
    secret = "the-right-key"
    async with _client(deck.user) as client:
        saved = await client.put("/fd/crew/own-model", json={"url": "http://192.168.1.20:8080", "key": secret, "models": "qwen:3, other", "context": 240000})
        assert saved.status_code == 200, saved.text
        view = saved.json()
        # No path means the usual one.
        assert view["url"] == "http://192.168.1.20:8080/v1" and view["names"] == ["own/qwen:3", "own/other"] and view["has_key"] is True
        me = (await client.get("/fd/crew/me")).json()
        assert secret not in saved.text and secret not in json.dumps(me)
        h = me["harnesses"][0]
        assert h["login"] == {"ok": True, "how": "your own model"} and h["default_model"] == "own/qwen:3"
        assert h["models"][:2] == ["own/qwen:3", "own/other"]
    raw = await deck.db.get_setting(deck.user["id"], "crew:own_model")
    assert raw and secret not in raw
    # The worker gets it in its environment, and in no file.
    user = await deck.db.get_user_by_id(deck.user["id"])
    env = await crew_logins.worker_env(deck.db, user, crew_access.grant_of(user), "opencode", str(deck.tmp / "bin" / "opencode"))
    assert env["CREW_OWN_MODEL_KEY"] == secret and secret not in env["OPENCODE_CONFIG_CONTENT"]
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["own"]
    assert config["options"] == {"baseURL": "http://192.168.1.20:8080/v1", "apiKey": "{env:CREW_OWN_MODEL_KEY}"}
    assert config["models"]["qwen:3"]["limit"] == {"context": 240000, "output": 60000}
    # Someone else's stays theirs.
    other = await deck.db.get_user_by_id(deck.other["id"])
    env = await crew_logins.worker_env(deck.db, other, crew_access.grant_of(other), "opencode", str(deck.tmp / "bin" / "opencode"))
    assert "CREW_OWN_MODEL_KEY" not in env and "OPENCODE_CONFIG_CONTENT" not in env
    async with _client(deck.other) as client:
        assert (await client.get("/fd/crew/me")).json()["own_model"] is None
    real = crew_runner.crew_harness.list_models
    monkeypatch.setattr(crew_runner.crew_harness, "list_models", lambda *a: [*real(*a), "own/qwen:3", "own/other"])
    async with _client(deck.user) as client:
        # Named nowhere, a task runs on it, not on OpenCode's own pick.
        made = await client.post("/fd/crew/tasks", json={"harness": "opencode", "brief": "Add it.", "project": "Game"})
        assert made.status_code == 201, made.text
        assert made.json()["model"] == "own/qwen:3"
    await _settle(deck.user["id"], made.json()["id"])
    assert "--model own/qwen:3" in opencode_too.read_text()
    async with _client(deck.user) as client:
        # Gone means gone, including from the choices.
        assert (await client.delete("/fd/crew/own-model")).status_code == 200
        me = (await client.get("/fd/crew/me")).json()
        assert me["own_model"] is None and me["harnesses"][0]["default_model"] == ""
        assert (await client.post("/fd/crew/tasks", json={"harness": "opencode", "brief": "Add it.", "project": "Game"})).status_code == 400


@pytest.mark.parametrize(("body", "why"), [
    ({"url": "ftp://x/v1", "models": "a"}, "address"), ({"url": "file:///etc/passwd", "models": "a"}, "address"),
    ({"url": "http://user:pw@x/v1", "models": "a"}, "address"), ({"url": "http://x/{file:~/.ssh/id}", "models": "a"}, "characters"),
    ({"url": "http://x/v1", "models": ""}, "at least one"), ({"url": "http://x/v1", "models": "{env:SECRET}"}, "model names"),
    ({"url": "http://x/v1", "models": "--flag"}, "model names"), ({"url": "http://x/v1", "models": "a", "context": "lots"}, "context"),
])
async def test_your_own_model_is_checked_before_it_is_kept(deck, body, why):
    await _approve_opencode(deck, deck.user)
    async with _client(deck.user) as client:
        refused = await client.put("/fd/crew/own-model", json=body)
        assert refused.status_code == 400 and why in refused.json()["detail"], refused.text
        assert (await client.get("/fd/crew/me")).json()["own_model"] is None


async def test_a_saved_key_goes_only_to_the_server_it_was_saved_for(deck):
    await _approve_opencode(deck, deck.user)
    seen: list = []
    server, url = await _own_model_server(seen)
    other_seen: list = []
    other_server, other_url = await _own_model_server(other_seen)
    async with server, other_server, _client(deck.user) as client:
        # No key yet: it's there, and says no.
        got = (await client.post("/fd/crew/own-model/check", json={"url": url})).json()
        assert got["ok"] is False and "refused" in got["error"] and got["url"] == f"{url}/v1"
        got = (await client.post("/fd/crew/own-model/check", json={"url": url, "key": "the-right-key"})).json()
        assert got["ok"] is True and got["models"] == ["qwen:3"]  # a name that couldn't be used isn't offered
        await client.put("/fd/crew/own-model", json={"url": url, "key": "the-right-key", "models": "qwen:3"})
        # Saved: a blank key means that one, for that server.
        assert (await client.post("/fd/crew/own-model/check", json={"url": url})).json()["ok"] is True
        # Another server gets nothing of it, whether checked or saved.
        got = (await client.post("/fd/crew/own-model/check", json={"url": other_url})).json()
        assert got["ok"] is False and all(auth == "" for _, auth in other_seen)
        kept = await client.put("/fd/crew/own-model", json={"url": other_url, "models": "qwen:3"})
        assert kept.json()["has_key"] is False
        unreachable = (await client.post("/fd/crew/own-model/check", json={"url": "http://127.0.0.1:9/v1"})).json()
        assert unreachable["ok"] is False and "Can't reach" in unreachable["error"]


async def test_pi_runs_on_your_own_model_and_keeps_the_rest_of_its_file(deck):
    async with _client(deck.admin) as client:
        await client.put(f"/fd/admin/users/{deck.user['id']}", json={"crew": {"approved": True, "harnesses": ["pi"], "max_parallel": 1}})
    user = await deck.db.get_user_by_id(deck.user["id"])
    grant = crew_access.grant_of(user)
    home = crew_logins.worker_home(user["id"])
    file = home / ".pi" / "agent" / "models.json"
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps({"theirs": 1, "providers": {"mine": {"baseUrl": "x"}}}))
    async with _client(deck.user) as client:
        assert (await client.get("/fd/crew/me")).json()["harnesses"][0]["login"]["ok"] is False
        await client.put("/fd/crew/own-model", json={"url": "http://x:1/v1", "key": "the-right-key", "models": "qwen:3", "context": 8192})
        h = (await client.get("/fd/crew/me")).json()["harnesses"][0]
        assert h["login"] == {"ok": True, "how": "your own model"} and h["default_model"] == "own/qwen:3" and h["models"] == ["own/qwen:3"]
    env = await crew_logins.worker_env(deck.db, user, grant, "pi", "/bin/pi")
    assert env["CREW_OWN_MODEL_KEY"] == "the-right-key"
    written = file.read_text()
    assert "the-right-key" not in written
    parsed = json.loads(written)
    assert parsed["theirs"] == 1 and parsed["providers"]["mine"] == {"baseUrl": "x"}
    assert parsed["providers"]["own"]["apiKey"] == "$CREW_OWN_MODEL_KEY"
    assert parsed["providers"]["own"]["models"] == [{"id": "qwen:3", "contextWindow": 8192, "maxTokens": 2048}]
    # Removed from the page, it leaves the file at the next run, and what's theirs stays.
    crew_logins._sync_pi_models(home, {})
    parsed = json.loads(file.read_text())
    assert "own" not in parsed["providers"] and parsed["providers"]["mine"] == {"baseUrl": "x"} and parsed["theirs"] == 1


async def test_a_default_model_is_checked_and_each_users_own(deck, opencode_too):
    await _approve_opencode(deck, deck.user)
    await _approve_opencode(deck, deck.other)
    async with _client(deck.user) as client:
        assert (await client.put("/fd/crew/models/opencode", json={"model": "--danger"})).status_code == 400
        assert (await client.put("/fd/crew/models/claude", json={"model": "x"})).status_code == 404  # not approved for it
        assert (await client.put("/fd/crew/models/opencode", json={"model": "opencode/big-pickle"})).status_code == 200
    async with _client(deck.other) as client:
        assert (await client.get("/fd/crew/me")).json()["harnesses"][0]["default_model"] == ""


async def test_pi_still_needs_a_key(deck):
    async with _client(deck.admin) as client:
        await client.put(f"/fd/admin/users/{deck.user['id']}", json={"crew": {"approved": True, "harnesses": ["pi"], "max_parallel": 1}})
    async with _client(deck.user) as client:
        assert (await client.get("/fd/crew/me")).json()["harnesses"][0]["login"]["ok"] is False


async def test_the_tool_says_a_free_model_is_the_users_choice():
    options = _FakeClient(options={"repos": {}, "usage": {"harnesses": []}, "me": {"harnesses": [
        {"id": "opencode", "label": "OpenCode", "installed": True, "login": {"ok": True, "how": "OpenCode's free models", "free": True},
         "default_model": "", "models": ["opencode/deepseek-v4.1-flash-free"]}]}})
    out = (await CrewTool()._do_options(options)).content
    assert "the user's choice" in out and "opencode/deepseek-v4.1-flash-free" in out
    options.answers["options"]["me"]["harnesses"][0]["default_model"] = "opencode/deepseek-v4.1-flash-free"
    out = (await CrewTool()._do_options(options)).content
    assert "Tasks use opencode/deepseek-v4.1-flash-free" in out and "ask them" not in out


async def test_the_tool_answers_which_model_and_how_much_of_the_limit():
    options = _FakeClient(options={"repos": {}, "me": {"harnesses": [
        {"id": "claude", "label": "Claude Code", "installed": True, "login": {"ok": True, "how": "your saved login"}},
        {"id": "codex", "label": "Codex", "installed": True, "login": {"ok": True, "how": "a key"}},
        {"id": "pi", "label": "Pi", "installed": False, "login": {"ok": False}},
    ]}, "usage": {"harnesses": [{"id": "claude", "model": "claude-sonnet-5-5", "tasks": 3, "tokens": 480000, "seen_at": time.time() - 600,
                                 "limits": {"seven_day": {"used": 0.44, "resets_at": 1793811600, "status": "allowed"}}}]}})
    out = (await CrewTool()._do_options(options)).content
    for piece in ("Last ran on claude-sonnet-5-5", "The weekly limit: 44% used, resets", "10 minutes ago", "3, 480,000 tokens"):
        assert piece in out, piece
    # A CLI that says nothing is said to say nothing, not guessed at.
    assert out.count("not reported yet") == 1
    status = _FakeClient(task={"id": "t", "status": "ready", "title": "x", "model_used": "claude-haiku-5-5", "model": "haiku"})
    assert "Model: claude-haiku-5-5" in (await CrewTool()._do_status(status, task_id="t")).content


async def test_status_gives_the_next_steps_and_pr_gives_the_link():
    status = _FakeClient(task={"id": "t", "status": "waiting", "title": "x", "evidence": {"commits": 1},
                               "next_steps": ["To push this and open a pull request: crew action=pr."]})
    out = await CrewTool()._do_status(status, task_id="t")
    assert "crew action=pr" in out.content
    pr = await CrewTool()._do_pr(_FakeClient(pr={"note": "Pull request ready: https://github.com/a/b/pull/7"}), task_id="t")
    assert "https://github.com/a/b/pull/7" in pr.content and "Give the user that link" in pr.content


async def test_a_merge_from_an_address_is_refused_before_the_person_is_asked(deck, monkeypatch):
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = (await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "project": "Game"})).json()
    done = await _settle(deck.user["id"], made["id"])
    crew_store.get_store().update(done["id"], repo_kind="url")
    task = crew_runner.view(crew_store.get_store().get(deck.user["id"], done["id"]), full=True)
    assert not any("request_merge" in s for s in task["next_steps"]) and any("crew action=pr" in s for s in task["next_steps"])
    from captain_claw.flight_deck import agent_identity

    caller = types.SimpleNamespace(owner=deck.user["id"], slug="a", is_shared=False)
    monkeypatch.setattr(agent_identity, "require_agent_caller", lambda request, what="": caller)
    with pytest.raises(HTTPException) as err:
        await crew_routes.agent_request_merge(crew_routes.AgentTaskBody(task_id=done["id"]), types.SimpleNamespace())
    assert err.value.status_code == 409 and "crew action=pr" in err.value.detail


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


async def test_an_agent_asks_for_a_deck_update_and_only_its_admin_approves(deck, as_agent, ledger, monkeypatch):
    from captain_claw.flight_deck import deck_update

    new = {"available": True, "active": False, "behind": 2, "target": "a" * 40, "upstream": "origin/main", "blockers": [],
           "head": "b" * 40, "head_title": "old", "commits": [{"title": "one"}, {"title": "two"}]}
    started: list[dict] = []

    async def status(*, fetch=False):
        assert fetch  # an agent's ask is about what's on the remote now
        return new

    async def start(**kw):
        started.append(kw)
        return {"from": new["head"]}

    monkeypatch.setattr(deck_update, "status", status)
    monkeypatch.setattr(deck_update, "start", start)

    # Not for an approved non-admin's agent.
    await _approve(deck, deck.user)
    as_agent["owner"] = deck.user["id"]
    async with _client(None) as agent:
        refused = await agent.post("/fd/crew/agent/update_deck", json={})
    assert refused.status_code == 403 and "admin" in refused.json()["detail"]
    assert ledger.waiting_actions(deck.user["id"]) == []

    as_agent["owner"] = deck.admin["id"]
    async with _client(None) as agent:
        asked = await agent.post("/fd/crew/agent/update_deck", json={"origin_platform": "telegram", "origin_chat_id": 7})
        again = await agent.post("/fd/crew/agent/update_deck", json={})
    assert asked.json()["status"] == "waiting_for_approval" and again.json()["status"] == "already_waiting"
    action = ledger.waiting_actions(deck.admin["id"])[0]
    assert action["kind"] == "deck_update" and action["payload"]["target"] == "a" * 40
    assert "- one" in action["rationale"]

    # A later automatic pass can't start it, and the Approve does, for that commit.
    assert (await fd_dispatch.dispatch_action(deck.admin["id"], action))["ok"] is False
    assert started == []
    out = await fd_dispatch.dispatch_action(deck.admin["id"], action, approved_by_human=True)
    assert out["ok"] is True and ledger.get_action(action["id"])["status"] == "done"
    assert started[0]["target"] == "a" * 40 and started[0]["wait"] is True and started[0]["user_id"] == deck.admin["id"]
    assert started[0]["agent"] == {"slug": "helper", "origin": {"platform": "telegram", "user_id": "", "chat_id": 7}}


async def test_a_newer_version_replaces_the_deck_update_waiting(deck, as_agent, ledger, monkeypatch):
    from captain_claw.flight_deck import deck_update

    new = {"available": True, "active": False, "behind": 1, "target": "a" * 40, "upstream": "origin/main", "blockers": [],
           "commits": [{"title": "one"}]}

    async def status(*, fetch=False):
        return new

    monkeypatch.setattr(deck_update, "status", status)
    as_agent["owner"] = deck.admin["id"]
    async with _client(None) as agent:
        first = (await agent.post("/fd/crew/agent/update_deck", json={})).json()
        new["target"] = "c" * 40
        second = (await agent.post("/fd/crew/agent/update_deck", json={})).json()
    assert first["action_id"] != second["action_id"]
    waiting = ledger.waiting_actions(deck.admin["id"])
    assert [w["payload"]["target"] for w in waiting] == ["c" * 40]


async def test_a_deck_update_an_agent_cant_have_says_why(deck, as_agent, ledger, monkeypatch):
    from captain_claw.flight_deck import deck_update

    answers = iter([
        {"available": False, "why": "no checkout"},
        {"available": True, "active": True},
        {"available": True, "active": False, "fetch_error": "can't read it"},
        {"available": True, "active": False, "behind": 1, "blockers": ["files in the way"]},
        {"available": True, "active": False, "behind": 0, "head": "b" * 40, "head_title": "old"},
    ])

    async def status(*, fetch=False):
        return next(answers)

    monkeypatch.setattr(deck_update, "status", status)
    as_agent["owner"] = deck.admin["id"]
    async with _client(None) as agent:
        for expect in ("no checkout", "already under way", "can't read it", "files in the way"):
            resp = await agent.post("/fd/crew/agent/update_deck", json={})
            assert resp.status_code == 409 and expect in resp.json()["detail"]
        assert (await agent.post("/fd/crew/agent/update_deck", json={})).json()["status"] == "up_to_date"
    assert ledger.waiting_actions(deck.admin["id"]) == []


def _a_repository(deck, name: str = "theirs") -> str:
    repo = deck.tmp / name
    repo.mkdir()
    for args in (("init", "-q", "-b", "main", "."), ("commit", "-q", "--allow-empty", "-m", "base")):
        subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True)
    return str(repo)


async def test_an_agent_gets_a_repository_with_no_worker(deck, as_agent, monkeypatch):
    monkeypatch.setattr(crew_runner, "_check_url", lambda url, grant: None)
    url = _a_repository(deck)
    await _approve(deck, deck.user)
    as_agent["owner"] = deck.user["id"]
    async with _client(None) as agent:
        made = await agent.post("/fd/crew/agent/clone", json={"repo_url": url, "origin_platform": "telegram", "origin_user_id": "7", "origin_chat_id": 7})
    assert made.status_code == 200, made.text
    task = made.json()
    assert task["status"] == "preparing" and task["kind"] == "clone" and task["from_agent"] == "helper"
    assert task["harness_label"] == "Captain" and task["project_name"] == "theirs" and task["repo_url"] == url
    done = await _settle(deck.user["id"], task["id"])
    assert done["status"] == "ready" and done["origin"]["chat_id"] == 7
    async with _client(deck.user) as client:
        assert [t["id"] for t in (await client.get("/fd/crew/tasks")).json()["tasks"]] == [task["id"]]
        recent = (await client.get("/fd/crew/repos")).json()["recent"]
        assert [r["repo_url"] for r in recent] == [url]
    async with _client(deck.other) as client:
        await _approve(deck, deck.other)
        assert (await client.get("/fd/crew/tasks")).json()["tasks"] == []


async def test_the_page_opens_a_task_on_an_address_straight_away(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_check_url", lambda url, grant: None)
    url = _a_repository(deck)
    await _approve(deck, deck.user)
    async with _client(deck.user) as client:
        await client.put("/fd/crew/logins/claude", json={"secret": "sk-ant-oat01-abc"})
        made = await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "SCENARIO:work", "repo_url": url})
        assert made.status_code == 201, made.text
        assert made.json()["status"] == "preparing"
        both = await client.post("/fd/crew/tasks", json={"harness": "claude", "brief": "x", "repo_url": url, "project": "other"})
        assert both.status_code == 400 and "not project and repo_url" in both.json()["detail"]
    done = await _settle(deck.user["id"], made.json()["id"])
    assert done["status"] == "ready" and done["evidence"]["commits"] == 1


async def test_the_tool_says_a_repository_is_on_its_way_and_what_comes_next():
    url = "https://github.com/k/captain-claw"
    made = {"id": "crw_1", "status": "preparing", "harness_label": "OpenCode", "repo_label": "captain-claw", "branch": "crew/x-abc123"}
    out = await CrewTool()._do_dispatch(_FakeClient(dispatch=made), harness="opencode", brief="b", repo_url=url)
    assert "getting its repository first" in out.content and "nothing is done yet" in out.content and "crw_1" in out.content
    client = _FakeClient(clone={"id": "crw_2", "status": "preparing", "repo_label": "captain-claw"})
    out = await CrewTool()._do_clone(client, repo_url=url)
    assert client.asked[0][0] == "clone" and client.asked[0][1]["repo_url"] == url
    assert "crw_2" in out.content and "with no worker" in out.content and f"crew action=dispatch repo_url={url}" in out.content
    assert not (await CrewTool()._do_clone(_FakeClient())).success
    options = _FakeClient(options={"me": {"harnesses": []}, "repos": {"projects": [], "recent": [
        {"repo": "/deck/crew/u/repos/a", "repo_label": "a", "repo_kind": "url", "repo_url": url},
        {"repo": "/deck/crew/u/repos/b", "repo_label": "b", "repo_kind": "url", "repo_url": ""},
        {"repo": "/home/me/c", "repo_label": "c", "repo_kind": "path", "repo_url": ""}]}})
    shown = (await CrewTool()._do_options(options)).content
    assert url in shown and "/home/me/c" in shown and "/deck/crew" not in shown


async def test_a_call_that_times_out_says_so_rather_than_nothing(monkeypatch):
    """httpx's timeout has empty text: the agent got "Crew request failed: " and guessed the rest."""
    from captain_claw import crew_client

    monkeypatch.setattr(crew_client, "under_flight_deck", lambda: True)
    monkeypatch.setattr(crew_client, "member_call", lambda: False)

    async def slow(*args, **kwargs):
        raise httpx.ReadTimeout("")

    monkeypatch.setattr(crew_client, "call", slow)
    out = await CrewTool().execute(action="list")
    assert not out.success and "may still be going" in out.error and "crew action=list" in out.error

    async def blank(*args, **kwargs):
        raise RuntimeError

    monkeypatch.setattr(crew_client, "call", blank)
    assert (await CrewTool().execute(action="list")).error == "Crew request failed: RuntimeError"
