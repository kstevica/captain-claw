"""Flow tenancy — flows are owned, and a run only reaches its owner's agents.

Before: every /fd/flows route served every authenticated user (list / read /
edit / delete / run anyone's flow), and the runner dispatched a step to any
agent in the deck — `tool shell on name:<someone else's agent>` ran a shell
command on another user's agent with that agent's own token.

Now: flows carry an owner (stamped on create; legacy rows are admin-only),
non-admins see and run only their own, and every step that addresses an
existing agent is checked against the run's owner using FD's spawn records.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import httpx
import pytest

from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.flow_runner import FlowRunner
from captain_claw.flight_deck.flows_store import FlowStore

ALICE_PORT, BOB_PORT, ORPHAN_PORT = 24611, 24621, 24631


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fd_server.app, client=("127.0.0.1", 40001)),
        base_url="http://localhost:25080")


def _bearer(user: dict) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user['id'], user.get('role', 'user'))}"}


def _tool_flow(name: str, on: str) -> dict:
    return {
        "name": name,
        "trigger": {"on": "message", "channel": "any", "match": {"kind": "always"}},
        "steps": [{"id": "sh", "type": "tool", "tool": "shell",
                   "args": {"command": "id"}, "on": on}],
        "output": {"channel": "log"},
    }


def _agent_flow(name: str, on: str) -> dict:
    return {
        "name": name,
        "trigger": {"on": "message", "channel": "any", "match": {"kind": "always"}},
        "steps": [{"id": "ask", "type": "agent", "prompt": "{{trigger.text}}", "on": on}],
        "output": {"channel": "log"},
    }


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch):
    """Two users' process agents + an ownerless one, a real FlowStore, and the
    runner wired like the server wires it; agent HTTP and consults recorded."""
    from captain_claw.flight_deck import auth as fd_auth

    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "users.db")
    await db.init()
    set_auth_db(db)
    alice = await db.create_user("alice@x.test", "h", "Alice")
    bob = await db.create_user("bob@x.test", "h", "Bob")
    admin = await db.create_user("admin@x.test", "h", "Admin", role="admin")

    registry = {
        "alice-agent": {"name": "alice-agent", "web_port": ALICE_PORT, "web_auth": "tok-alice",
                        "owner": alice["id"]},
        "bob-agent": {"name": "bob-agent", "web_port": BOB_PORT, "web_auth": "tok-bob",
                      "owner": bob["id"]},
        "orphan": {"name": "orphan", "web_port": ORPHAN_PORT, "web_auth": "tok-orphan", "owner": ""},
    }

    def _no_docker():
        raise RuntimeError("no docker in tests")

    monkeypatch.setattr(fd_server, "_load_process_registry", lambda: registry)
    monkeypatch.setattr(fd_server, "_process_is_alive", lambda slug: True)
    monkeypatch.setattr(fd_server, "get_docker", _no_docker)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)

    # Agent-bound HTTP (tool RPC, chat push) — recorded, never sent. Clients
    # built with an explicit transport (the ASGI test client) pass through.
    hits: list[dict] = []
    real_client = httpx.AsyncClient

    def _handler(request: httpx.Request) -> httpx.Response:
        hits.append({"port": request.url.port, "path": request.url.path})
        return httpx.Response(200, json={"success": True, "content": "uid=501"})

    def _client_factory(*a, **kw):
        kw.setdefault("transport", httpx.MockTransport(_handler))
        return real_client(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

    consults: list[int] = []

    async def fake_consult(host, port, auth, message, **kw):
        consults.append(int(port))
        yield {"ok": True, "done": True, "response": f"answered by {port}"}

    store = FlowStore(tmp_path / "flows.db")
    runner = FlowRunner(
        store,
        get_agents=fd_server._running_agents,
        resolve_auth=fd_server._resolve_agent_auth,
        fd_self_base="http://localhost:1",
        consult_peer=fake_consult,
        resolve_owner=fd_server._resolve_agent_owner,
        user_is_admin=fd_server._flow_user_is_admin,
        enforce_owner=True,
    )
    monkeypatch.setattr(fd_server.app.state, "flow_store", store, raising=False)
    monkeypatch.setattr(fd_server.app.state, "flow_runner", runner, raising=False)
    try:
        yield types.SimpleNamespace(
            store=store, runner=runner, hits=hits, consults=consults,
            alice=alice, bob=bob, admin=admin)
    finally:
        if store._db is not None:
            await store._db.close()
        await db.close()
        fd_auth._db = prev


async def _create(c: httpx.AsyncClient, user: dict, spec: dict) -> str:
    r = await c.post("/fd/flows", headers=_bearer(user), json=spec)
    assert r.status_code == 200, r.text
    return r.json()["id"]


async def _wait_run(store: FlowStore, run_id: str) -> dict:
    for _ in range(200):
        d = await store.get_run(run_id)
        if d and d["run"]["status"] not in ("running",):
            return d
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish")


# ── the reported exploit ─────────────────────────────────────────────


async def test_tool_step_on_another_users_agent_is_refused(deck):
    async with _client() as c:
        fid = await _create(c, deck.bob, _tool_flow("pwn", "name:alice-agent"))
        r = await c.post(f"/fd/flows/{fid}/run", headers=_bearer(deck.bob), json={})
    assert r.status_code == 200
    detail = await _wait_run(deck.store, r.json()["run_id"])
    assert deck.hits == []                     # nothing reached Alice's agent
    out = detail["steps"][0]["output_text"]
    assert "belongs to another user" in out


async def test_tool_step_on_own_agent_runs(deck):
    async with _client() as c:
        fid = await _create(c, deck.bob, _tool_flow("mine", "name:bob-agent"))
        r = await c.post(f"/fd/flows/{fid}/run", headers=_bearer(deck.bob), json={})
    detail = await _wait_run(deck.store, r.json()["run_id"])
    assert detail["run"]["status"] == "done"
    assert deck.hits == [{"port": BOB_PORT, "path": "/api/tool"}]
    assert detail["steps"][0]["output_text"] == "uid=501"


async def test_dry_test_route_is_checked_too(deck):
    """/test runs steps for real (only persistence is skipped)."""
    async with _client() as c:
        fid = await _create(c, deck.bob, _tool_flow("pwn", "name:alice-agent"))
        r = await c.post(f"/fd/flows/{fid}/test", headers=_bearer(deck.bob), json={})
    assert r.status_code == 200
    assert deck.hits == []
    assert "belongs to another user" in r.json()["steps"][0]["output"]


async def test_origin_payload_naming_another_users_agent_is_refused(deck):
    """`on: origin` (the default) addressed by a caller-supplied payload."""
    async with _client() as c:
        fid = await _create(c, deck.bob, _tool_flow("pwn", "origin"))
        r = await c.post(f"/fd/flows/{fid}/run", headers=_bearer(deck.bob),
                         json={"payload": {"origin_port": ALICE_PORT}})
    detail = await _wait_run(deck.store, r.json()["run_id"])
    assert deck.hits == []
    assert detail["run"]["status"] == "error"
    assert "belongs to another user" in (detail["run"]["error"] or "")


async def test_any_selector_picks_only_the_owners_agents(deck):
    async with _client() as c:
        fid = await _create(c, deck.bob, _agent_flow("ask", "any"))
        r = await c.post(f"/fd/flows/{fid}/test", headers=_bearer(deck.bob),
                         json={"payload": {"text": "hi"}})
    assert r.status_code == 200
    assert deck.consults == [BOB_PORT]


async def test_ownerless_agent_is_not_reachable_by_a_user_flow(deck):
    async with _client() as c:
        fid = await _create(c, deck.bob, _tool_flow("orph", "name:orphan"))
        r = await c.post(f"/fd/flows/{fid}/test", headers=_bearer(deck.bob), json={})
    assert deck.hits == []
    assert "belongs to another user" in r.json()["steps"][0]["output"]


async def test_admin_run_may_target_any_agent(deck):
    async with _client() as c:
        fid = await _create(c, deck.admin, _tool_flow("ops", "name:alice-agent"))
        r = await c.post(f"/fd/flows/{fid}/test", headers=_bearer(deck.admin), json={})
    assert r.status_code == 200
    assert deck.hits == [{"port": ALICE_PORT, "path": "/api/tool"}]


# ── flows are owner-scoped ───────────────────────────────────────────


async def test_flows_are_scoped_to_their_owner(deck):
    async with _client() as c:
        fid = await _create(c, deck.alice, _tool_flow("alice-flow", "name:alice-agent"))
        bob = _bearer(deck.bob)
        listed = (await c.get("/fd/flows", headers=bob)).json()["flows"]
        assert [f["id"] for f in listed] == []
        assert (await c.get(f"/fd/flows/{fid}", headers=bob)).status_code == 404
        assert (await c.put(f"/fd/flows/{fid}", headers=bob,
                            json=_tool_flow("hijack", "name:bob-agent"))).status_code == 404
        assert (await c.post(f"/fd/flows/{fid}/enable", headers=bob,
                             json={"enabled": False})).status_code == 404
        assert (await c.post(f"/fd/flows/{fid}/run", headers=bob, json={})).status_code == 404
        assert (await c.post(f"/fd/flows/{fid}/test", headers=bob, json={})).status_code == 404
        assert (await c.get(f"/fd/flows/{fid}/runs", headers=bob)).status_code == 404
        assert (await c.delete(f"/fd/flows/{fid}", headers=bob)).status_code == 404

        # The owner still has it, untouched; the admin sees everyone's.
        mine = (await c.get(f"/fd/flows/{fid}", headers=_bearer(deck.alice))).json()
        assert mine["name"] == "alice-flow" and mine["enabled"] is True
        assert mine["owner_id"] == deck.alice["id"]
        all_ids = [f["id"] for f in (await c.get("/fd/flows", headers=_bearer(deck.admin))).json()["flows"]]
        assert fid in all_ids
    assert deck.hits == []


async def test_owner_is_stamped_from_the_caller_not_the_body(deck):
    async with _client() as c:
        spec = {**_tool_flow("sneaky", "name:bob-agent"), "owner_id": deck.alice["id"]}
        fid = await _create(c, deck.bob, spec)
        flow = (await c.get(f"/fd/flows/{fid}", headers=_bearer(deck.bob))).json()
        assert flow["owner_id"] == deck.bob["id"]
        # An update can't move it to someone else either.
        await c.put(f"/fd/flows/{fid}", headers=_bearer(deck.bob), json=spec)
        flow = (await c.get(f"/fd/flows/{fid}", headers=_bearer(deck.bob))).json()
        assert flow["owner_id"] == deck.bob["id"]


async def test_legacy_unowned_flows_are_admin_only(deck):
    fid = await deck.store.create_flow(_tool_flow("legacy", "name:alice-agent"))
    async with _client() as c:
        assert (await c.get(f"/fd/flows/{fid}", headers=_bearer(deck.bob))).status_code == 404
        assert fid not in [f["id"] for f in
                           (await c.get("/fd/flows", headers=_bearer(deck.bob))).json()["flows"]]
        assert (await c.get(f"/fd/flows/{fid}", headers=_bearer(deck.admin))).status_code == 200


async def test_run_detail_and_control_are_scoped(deck):
    async with _client() as c:
        fid = await _create(c, deck.alice, _tool_flow("a", "name:alice-agent"))
        r = await c.post(f"/fd/flows/{fid}/run", headers=_bearer(deck.alice), json={})
        run_id = r.json()["run_id"]
        await _wait_run(deck.store, run_id)
        bob = _bearer(deck.bob)
        assert (await c.get(f"/fd/flows/runs/{run_id}", headers=bob)).status_code == 404
        for verb in ("pause", "resume", "stop"):
            assert (await c.post(f"/fd/flows/runs/{run_id}/{verb}", headers=bob,
                                 json={})).status_code == 404
        assert (await c.get(f"/fd/flows/runs/{run_id}",
                            headers=_bearer(deck.alice))).status_code == 200


async def test_gosub_resolves_only_the_owners_flows(deck):
    """A same-named flow of another user is not a subroutine of yours."""
    async with _client() as c:
        await _create(c, deck.alice, {**_tool_flow("helper", "name:alice-agent"), "enabled": False})
        caller = {
            "name": "caller",
            "trigger": {"on": "message", "match": {"kind": "always"}},
            "steps": [{"id": "g", "type": "gosub", "flow": "helper"}],
            "output": {"channel": "log"},
        }
        fid = await _create(c, deck.bob, caller)
        r = await c.post(f"/fd/flows/{fid}/test", headers=_bearer(deck.bob), json={})
    assert deck.hits == []
    assert "no flow named 'helper'" in r.json()["steps"][0]["output"]


async def test_gosub_args_cannot_repoint_the_origin(deck):
    """A gosub child shares the root, so run()'s origin check doesn't see its
    payload: a `wait` keyed on Alice's agent would swallow her next message."""
    from captain_claw.flight_deck import flow_router

    def _caller(port: int) -> dict:
        return {"name": f"caller-{port}",
                "trigger": {"on": "message", "match": {"kind": "always"}},
                "steps": [{"id": "g", "type": "gosub", "flow": "listen",
                           "args": {"origin_port": port}}],
                "output": {"channel": "log"}}

    await deck.store.create_flow({
        "name": "listen", "trigger": {"on": "message", "match": {"kind": "always"}},
        "steps": [{"id": "w", "type": "wait", "until": "contains hello", "timeout": 1}],
        "output": {"channel": "log"}}, owner_id=deck.bob["id"])
    pwn = await deck.store.get_flow(
        await deck.store.create_flow(_caller(ALICE_PORT), owner_id=deck.bob["id"]))
    task = asyncio.create_task(deck.runner.run(pwn, {}, owner_id=deck.bob["id"]))
    await asyncio.sleep(0.05)
    assert not flow_router.has_pending_input(channel="", origin_port=ALICE_PORT)
    res = await task
    assert "belongs to another user" in res["steps"][0]["output"]

    # Re-pointing at the owner's own agent is fine: the child runs (and waits).
    mine = await deck.store.get_flow(
        await deck.store.create_flow(_caller(BOB_PORT), owner_id=deck.bob["id"]))
    task = asyncio.create_task(deck.runner.run(mine, {}, owner_id=deck.bob["id"]))
    await asyncio.sleep(0.05)
    assert flow_router.deliver_pending_input(channel="", origin_port=BOB_PORT, text="hello")
    res = await task
    assert res["status"] == "done"


async def test_archetype_env_comes_only_from_the_run_owners_agent(deck, tmp_path, monkeypatch):
    """A spawned specialist inherits the origin agent's .env — only when that
    agent is the run owner's (the runner stamps the owner as user_id)."""
    for slug, key in (("alice-agent", "ALICE_SECRET"), ("bob-agent", "BOB_KEY")):
        (tmp_path / slug).mkdir()
        (tmp_path / slug / ".env").write_text(f"{key}=v\n")
    monkeypatch.setattr(fd_server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", True)
    bob = deck.bob["id"]

    def keys(payload: dict) -> list[str]:
        return [e["key"] for e in fd_server._flow_origin_env(payload)]

    # Another user's agent — named (even stopped: not in the live pool) or by port.
    assert keys({"origin_name": "alice-agent", "user_id": bob}) == []
    assert keys({"origin_port": ALICE_PORT, "origin_name": "bob-agent", "user_id": bob}) == ["BOB_KEY"]
    assert keys({"origin_port": BOB_PORT, "user_id": bob}) == ["BOB_KEY"]
    # Unowned (legacy) runs and auth off: unchanged.
    assert keys({"origin_name": "alice-agent"}) == ["ALICE_SECRET"]
    monkeypatch.setattr(fd_server, "AUTH_ENABLED", False)
    assert keys({"origin_name": "alice-agent", "user_id": bob}) == ["ALICE_SECRET"]


async def test_unavailable_agent_message_names_no_other_users_agent(deck, monkeypatch):
    """`any` with none of the owner's agents up must not reveal whose are."""
    monkeypatch.setattr(fd_server, "_process_is_alive", lambda slug: slug != "bob-agent")
    fid = await deck.store.create_flow(_agent_flow("ask", "any"), owner_id=deck.bob["id"])
    res = await deck.runner.run(await deck.store.get_flow(fid), {"text": "hi"}, dry=True)
    out = res["steps"][0]["output"]
    assert "no agent" in out and "alice" not in out and "orphan" not in out
    assert deck.consults == []


# ── message triggers ─────────────────────────────────────────────────


async def test_trigger_only_matches_flows_of_the_origin_agents_owner(deck):
    """Alice's catch-all flow doesn't fire on (or read) Bob's agent's messages."""
    from captain_claw.flight_deck import flow_router

    monkey_store, monkey_runner = flow_router._STORE, flow_router._RUNNER
    flow_router.set_engine(deck.store, deck.runner)
    try:
        async with _client() as c:
            await _create(c, deck.alice, _agent_flow("snoop", "name:alice-agent"))
        bob_msg = flow_router.classify_payload(channel="web", text="secret", origin_port=BOB_PORT)
        assert await flow_router.match_flow(bob_msg) is None
        alice_msg = flow_router.classify_payload(channel="web", text="hi", origin_port=ALICE_PORT)
        matched = await flow_router.match_flow(alice_msg)
        assert matched is not None and matched["name"] == "snoop"
        # Even if a caller skips the match filter, the run refuses up front.
        snoop = await deck.store.get_flow_by_name("snoop")
        res = await deck.runner.run(snoop, bob_msg, dry=True)
        assert res["status"] == "error" and "belongs to another user" in res["error"]
        assert deck.consults == []
    finally:
        flow_router.set_engine(monkey_store, monkey_runner)


async def test_explicit_flow_run_command_picks_the_origin_owners_flow(deck):
    """'/flow run helper' from Bob's agent runs Bob's 'helper', not Alice's."""
    from captain_claw.flight_deck import flow_router

    def _echo(name: str, text: str) -> dict:
        return {"name": name, "enabled": False,
                "trigger": {"on": "message", "match": {"kind": "always"}},
                "steps": [{"id": "e", "type": "emit", "channel": "log", "body": text}],
                "output": {"channel": "log"}}

    prev = flow_router._STORE, flow_router._RUNNER
    flow_router.set_engine(deck.store, deck.runner)
    try:
        async with _client() as c:
            await _create(c, deck.alice, {**_echo("helper", "alice's"), "priority": 90})
            await _create(c, deck.bob, _echo("helper", "bob's"))
            r = await c.post("/fd/flows/evaluate", json={
                "channel": "web", "text": "/flow run helper", "origin_port": BOB_PORT})
        assert r.status_code == 200
        assert r.json()["output"] == "bob's"
    finally:
        flow_router.set_engine(*prev)


# ── caller-supplied owner context (scheduler wiring) ─────────────────


async def test_explicit_owner_context_refuses_another_users_flow(deck):
    fid = await deck.store.create_flow(_tool_flow("alice-flow", "name:alice-agent"),
                                       owner_id=deck.alice["id"])
    flow = await deck.store.get_flow(fid)
    res = await deck.runner.run(flow, {}, dry=True, owner_id=deck.bob["id"])
    assert res["status"] == "error" and "another user" in res["error"]
    assert deck.hits == []
    ok = await deck.runner.run(flow, {}, dry=True, owner_id=deck.alice["id"])
    assert ok["status"] == "done"
    assert deck.hits == [{"port": ALICE_PORT, "path": "/api/tool"}]


async def test_flow_without_caller_context_runs_as_its_owner(deck):
    """A scheduler job (no owner passed yet) runs the flow as the flow's owner."""
    fid = await deck.store.create_flow(_tool_flow("x", "name:alice-agent"), owner_id=deck.bob["id"])
    res = await deck.runner.run(await deck.store.get_flow(fid), {"channel": "scheduler"}, dry=True)
    assert deck.hits == []
    assert "belongs to another user" in res["steps"][0]["output"]


async def test_archetype_spawn_runs_as_the_run_owner_not_the_payload(deck):
    """payload.user_id used to pick whose Library keys a spawned specialist got."""
    seen: list[str] = []

    async def load(payload, aid):
        seen.append(str(payload.get("user_id") or ""))
        return None

    deck.runner.load_archetype = load
    deck.runner.spawn_archetype = lambda *a, **k: None
    fid = await deck.store.create_flow(_agent_flow("spec", "archetype:fact-checker"),
                                       owner_id=deck.bob["id"])
    await deck.runner.run(await deck.store.get_flow(fid), {"user_id": deck.alice["id"]}, dry=True)
    assert seen == [deck.bob["id"]]


# ── single-user (auth off) unchanged ─────────────────────────────────


async def test_auth_disabled_runner_is_unscoped(deck):
    deck.runner.enforce_owner = False
    fid = await deck.store.create_flow(_tool_flow("x", "name:alice-agent"), owner_id=deck.bob["id"])
    res = await deck.runner.run(await deck.store.get_flow(fid), {}, dry=True, owner_id=deck.bob["id"])
    assert res["status"] == "done"
    assert deck.hits == [{"port": ALICE_PORT, "path": "/api/tool"}]


async def test_auth_disabled_routes_see_all_flows(deck, monkeypatch):
    fid = await deck.store.create_flow(_tool_flow("legacy", "name:alice-agent"))
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    async with _client() as c:
        listed = (await c.get("/fd/flows")).json()["flows"]
        assert fid in [f["id"] for f in listed]
        new_id = await _create(c, {"id": "x", "role": "admin"}, _tool_flow("n", "name:bob-agent"))
        assert (await c.get(f"/fd/flows/{new_id}")).json()["owner_id"] == ""


# ── store migration ──────────────────────────────────────────────────


async def test_store_migrates_a_pre_owner_db(tmp_path: Path):
    import aiosqlite

    path = tmp_path / "flows.db"
    async with aiosqlite.connect(str(path)) as db:
        await db.execute("""CREATE TABLE flows (id TEXT PRIMARY KEY, name TEXT NOT NULL,
            description TEXT, enabled INTEGER NOT NULL DEFAULT 1, priority INTEGER NOT NULL DEFAULT 50,
            trigger_json TEXT, steps_json TEXT, guardrails_json TEXT, output_json TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        await db.execute("INSERT INTO flows (id, name, created_at, updated_at) VALUES ('old', 'Old', 'x', 'x')")
        await db.commit()
    store = FlowStore(path)
    try:
        old = await store.get_flow("old")
        assert old["owner_id"] == ""
        assert [f["id"] for f in await store.list_flows(owner_id="u1")] == []
        assert [f["id"] for f in await store.list_flows()] == ["old"]
    finally:
        await store._db.close()
