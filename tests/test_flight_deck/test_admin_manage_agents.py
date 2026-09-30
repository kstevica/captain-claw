"""An admin administers other users' agents — through the ordinary agent routes.

Every agent route is owner-scoped: before this an admin could only see and
control their own agents. ``X-FD-Act-As: <user id>`` lets an ADMIN run an
agent-management request for that user — list, create, start, stop, configure,
remove — exactly as if the owner had made it (their ownership, plan, Library).

Pinned here:

* only an admin may act for someone else (403 otherwise — refused, not ignored),
  and only for a real user (404);
* the header is honoured on the agent-management routes ONLY — it is not a
  login-as: the target's settings stay out of reach;
* each operation lands on the target's agent and nobody else's;
* an agent the admin creates belongs to the target, and its archetype
  instructions are recorded in the target's settings (the chat reads them there);
* what the admin does is attributed to the admin (rate limits, usage log), not
  charged to the target, and the admin gets no agent tokens back;
* instruction writes are per agent, never lose the rest of the map on a failed
  read, are capped, and go away with the agent.

Real FlightDeckDB + process registry in tmp dirs; the agent process is faked.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import captain_claw.flight_deck.archetypes as arch_mod
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = "user-admin"
BOB = "user-bob"
CAROL = "user-carol"
REMOTE = ("203.0.113.9", 40001)
LOOPBACK = ("127.0.0.1", 40001)
INSTR = "fd:process-fleet-instructions"
_ARCH = {"id": "analyst", "role": "Data Analyst", "tier": "balanced",
         "tools": ["read"], "fleet_instructions": "You are a careful data analyst."}


class _FakePopen:
    calls: list[str] = []

    def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None, start_new_session=None):
        _FakePopen.calls.append(Path(cwd).name)
        self.pid = 424242
        if stdout is not None:
            stdout.close()

    def poll(self):
        return None


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck: an admin and two users; Bob and Carol own one agent each."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid, role in ((ADMIN, "admin"), (BOB, "user"), (CAROL, "user")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid, role))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.setenv("FD_PORT", "25999")
    monkeypatch.setenv("FD_SPAWN_SETTLE_S", "0")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    # The archetype's tier is the registry's (Anthropic) and nobody configured a
    # key: this deck has it in its environment, which process agents inherit —
    # an archetype agent with no model key at all is refused.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fd-env-key")

    def _no_docker():
        raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server, "get_docker", _no_docker)
    _FakePopen.calls = []
    monkeypatch.setattr(server.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(server, "_is_port_available", lambda port: True)
    monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())

    async def fake_merged(_db, _uid):
        return [dict(_ARCH)]

    monkeypatch.setattr(arch_mod, "merged_archetypes", fake_merged)

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
        "carol-agent": {"slug": "carol-agent", "name": "Carol's agent", "web_port": 24102,
                        "web_auth": "carol-tok", "owner": CAROL, "pid": None},
    })
    for slug in ("bob-agent", "carol-agent"):
        (data / slug).mkdir()
        (data / slug / "config.yaml").write_text(f"model: {slug}-model\n")
        (data / slug / ".env").write_text("KEY=1\n")
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client(addr=REMOTE) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=addr), base_url="http://fd.test")


async def _instructions(db, uid) -> dict:
    raw = await db.get_setting(uid, INSTR)
    return json.loads(raw) if raw else {}


async def _usage(db, uid) -> list[tuple[str, dict]]:
    cur = await db._db.execute(
        "SELECT event_type, detail FROM usage_logs WHERE user_id = ? ORDER BY id", (uid,))
    return [(r[0], json.loads(r[1])) for r in await cur.fetchall()]


def _hdr(caller: str, act_as: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as is not None:
        h[ACT_AS_HEADER] = act_as
    return h


# ── who may act for whom ────────────────────────────────────────────────────


class TestWhoMayActForWhom:
    async def test_admin_lists_a_users_agents(self, deck):
        async with _client() as c:
            r = await c.get("/fd/processes", headers=_hdr(ADMIN, BOB))
        assert r.status_code == 200
        assert [p["slug"] for p in r.json()] == ["bob-agent"]

    async def test_without_the_header_an_admin_sees_only_their_own(self, deck):
        async with _client() as c:
            mine = await c.get("/fd/processes", headers=_hdr(ADMIN))
            stop = await c.post("/fd/processes/bob-agent/stop", headers=_hdr(ADMIN))
        assert mine.json() == [] and stop.status_code == 404

    async def test_a_non_admin_naming_someone_else_is_refused(self, deck):
        async with _client() as c:
            r = await c.get("/fd/processes", headers=_hdr(CAROL, BOB))
            stop = await c.post("/fd/processes/bob-agent/stop", headers=_hdr(CAROL, BOB))
        assert r.status_code == 403 and stop.status_code == 403
        assert "Admin access required" in r.json()["detail"]

    async def test_naming_yourself_is_not_acting_for_anyone(self, deck):
        async with _client() as c:
            r = await c.get("/fd/processes", headers=_hdr(CAROL, CAROL))
        assert r.status_code == 200 and [p["slug"] for p in r.json()] == ["carol-agent"]

    async def test_unknown_target_is_a_404(self, deck):
        async with _client() as c:
            r = await c.get("/fd/processes", headers=_hdr(ADMIN, "no-such-user"))
        assert r.status_code == 404

    async def test_acting_for_bob_never_reaches_carols_agent(self, deck):
        async with _client() as c:
            r = await c.post("/fd/processes/carol-agent/stop", headers=_hdr(ADMIN, BOB))
        assert r.status_code == 404

    async def test_it_is_not_a_login_as(self, deck):
        """The header means nothing outside agent management: the admin still
        reads and writes THEIR OWN settings, never the target's."""
        await deck.set_settings(BOB, {"fd:theme": "bobs"})
        async with _client() as c:
            got = await c.get("/fd/settings", headers=_hdr(ADMIN, BOB))
            put = await c.put("/fd/settings", headers=_hdr(ADMIN, BOB),
                              json={"settings": {"fd:theme": "admins"}})
        assert got.status_code == 200 and "bobs" not in got.text
        assert put.status_code == 200
        assert await deck.get_setting(BOB, "fd:theme") == "bobs"
        assert await deck.get_setting(ADMIN, "fd:theme") == "admins"


# ── start / stop / remove / config, on the target's agent ───────────────────


class TestOperations:
    async def test_stop_and_start_reach_the_targets_agent(self, deck, monkeypatch):
        seen: list[tuple[str, str]] = []

        def _ok(kind):
            def fn(slug):
                seen.append((kind, slug))
                return server.ProcessActionResult(ok=True, slug=slug, message=kind)
            return fn

        monkeypatch.setattr(server, "_do_stop_process", _ok("stop"))
        monkeypatch.setattr(server, "_do_start_process", _ok("start"))
        async with _client() as c:
            for action in ("stop", "start"):
                r = await c.post(f"/fd/processes/bob-agent/{action}", headers=_hdr(ADMIN, BOB))
                assert r.status_code == 200, r.text
        assert seen == [("stop", "bob-agent"), ("start", "bob-agent")]

    async def test_remove_takes_it_off_the_owners_list_only(self, deck):
        async with _client() as c:
            r = await c.delete("/fd/processes/bob-agent", headers=_hdr(ADMIN, BOB))
        assert r.status_code == 200
        assert set(server._load_process_registry()) == {"carol-agent"}

    async def test_read_and_change_config(self, deck):
        async with _client() as c:
            got = await c.get("/fd/agent-config/process/bob-agent", headers=_hdr(ADMIN, BOB))
            assert got.json() == {"config_yaml": "model: bob-agent-model\n", "env": "KEY=1\n"}
            put = await c.put("/fd/agent-config/process/bob-agent", headers=_hdr(ADMIN, BOB),
                              json={"config_yaml": "model: new\n", "env": "KEY=2\n"})
            assert put.status_code == 200
            other = await c.get("/fd/agent-config/process/carol-agent", headers=_hdr(ADMIN, BOB))
        assert (server.DATA_DIR / "bob-agent" / "config.yaml").read_text() == "model: new\n"
        assert (server.DATA_DIR / "bob-agent" / ".env").read_text() == "KEY=2\n"
        assert other.status_code == 404
        assert (server.DATA_DIR / "carol-agent" / "config.yaml").read_text() == "model: carol-agent-model\n"

    async def test_rename_via_identity(self, deck):
        async with _client() as c:
            r = await c.post("/fd/processes/bob-agent/identity", headers=_hdr(ADMIN, BOB),
                             json={"name": "Renamed", "description": "by admin"})
        assert r.status_code == 200
        entry = server._load_process_registry()["bob-agent"]
        assert entry["name"] == "Renamed" and entry["owner"] == BOB

    async def test_instructions_live_in_the_owners_settings(self, deck):
        async with _client() as c:
            put = await c.put("/fd/agent-instructions/process/bob-agent", headers=_hdr(ADMIN, BOB),
                              json={"instructions": "Answer in Croatian."})
            got = await c.get("/fd/agent-instructions/process/bob-agent", headers=_hdr(ADMIN, BOB))
            mine = await c.get("/fd/agent-instructions/process/bob-agent", headers=_hdr(BOB))
            nope = await c.put("/fd/agent-instructions/process/carol-agent", headers=_hdr(ADMIN, BOB),
                               json={"instructions": "x"})
        assert put.status_code == 200 and nope.status_code == 404
        assert got.json() == mine.json() == {"instructions": "Answer in Croatian."}
        stored = json.loads(await deck.get_setting(BOB, "fd:process-fleet-instructions"))
        assert stored == {"bob-agent": "Answer in Croatian."}
        assert await deck.get_setting(ADMIN, "fd:process-fleet-instructions") is None

    async def test_clearing_instructions_removes_the_entry(self, deck):
        await deck.set_settings(BOB, {"fd:process-fleet-instructions": json.dumps(
            {"bob-agent": "old", "other": "kept"})})
        async with _client() as c:
            await c.put("/fd/agent-instructions/process/bob-agent", headers=_hdr(ADMIN, BOB),
                        json={"instructions": "  "})
        assert json.loads(await deck.get_setting(BOB, "fd:process-fleet-instructions")) == {"other": "kept"}


# ── creating an agent for a user ────────────────────────────────────────────


class TestCreateForAUser:
    async def test_the_new_agent_belongs_to_the_target(self, deck):
        async with _client() as c:
            r = await c.post("/fd/spawn-process", headers=_hdr(ADMIN, BOB), json={
                "name": "Analyst", "archetype": "analyst", "botport_enabled": False,
                "web_enabled": True, "web_port": 24200})
        assert r.status_code == 200, r.text
        entry = server._load_process_registry()["analyst"]
        assert entry["owner"] == BOB
        assert _FakePopen.calls == ["analyst"]
        async with _client() as c:
            bobs = await c.get("/fd/processes", headers=_hdr(BOB))
            admins = await c.get("/fd/processes", headers=_hdr(ADMIN))
        assert "analyst" in [p["slug"] for p in bobs.json()] and admins.json() == []

    async def test_archetype_instructions_are_recorded_for_the_owner(self, deck):
        async with _client() as c:
            await c.post("/fd/spawn-process", headers=_hdr(ADMIN, BOB), json={
                "name": "Analyst", "archetype": "analyst", "botport_enabled": False,
                "web_enabled": True, "web_port": 24200})
        stored = json.loads(await deck.get_setting(BOB, "fd:process-fleet-instructions"))
        assert stored == {"analyst": "You are a careful data analyst."}
        assert await deck.get_setting(ADMIN, "fd:process-fleet-instructions") is None

    async def test_the_owners_own_archetype_spawn_records_them_too(self, deck):
        async with _client() as c:
            r = await c.post("/fd/spawn-process", headers=_hdr(CAROL), json={
                "name": "Mine", "archetype": "analyst", "botport_enabled": False,
                "web_enabled": True, "web_port": 24201})
        assert r.status_code == 200, r.text
        stored = json.loads(await deck.get_setting(CAROL, "fd:process-fleet-instructions"))
        assert stored == {"mine": "You are a careful data analyst."}

    async def test_the_targets_plan_limit_applies(self, deck, monkeypatch):
        async def _refuse(user, owned):
            raise server.HTTPException(403, f"limit for {user['id']}")

        monkeypatch.setattr(server, "check_agent_count_limit", _refuse)
        async with _client() as c:
            r = await c.post("/fd/spawn-process", headers=_hdr(ADMIN, BOB), json={
                "name": "Analyst", "archetype": "analyst", "web_port": 24200})
        assert r.status_code == 403 and r.json()["detail"] == f"limit for {BOB}"

    async def test_a_non_admin_cannot_create_for_someone_else(self, deck):
        async with _client() as c:
            r = await c.post("/fd/spawn-process", headers=_hdr(CAROL, BOB), json={
                "name": "Sneaky", "archetype": "analyst", "web_port": 24200})
        assert r.status_code == 403
        assert "sneaky" not in server._load_process_registry() and _FakePopen.calls == []

    async def test_the_picker_lists_the_targets_gallery(self, deck, monkeypatch):
        import captain_claw.flight_deck.archetype_routes as ar

        seen: list = []

        async def fake_registry(_db, uid):
            seen.append(uid)
            return {"tiers": {}, "archetypes": []}

        monkeypatch.setattr(ar, "merged_registry", fake_registry)
        async with _client() as c:
            await c.get("/fd/archetypes", headers=_hdr(ADMIN, BOB))
            await c.get("/fd/archetypes", headers=_hdr(ADMIN))
            r = await c.get("/fd/archetypes", headers=_hdr(CAROL, BOB))
        assert seen == [BOB, ADMIN] and r.status_code == 403



# ── who it is attributed to, and what comes back ────────────────────────────


class TestAttribution:
    async def test_changes_are_recorded_under_the_admin_reads_are_not(self, deck, monkeypatch):
        monkeypatch.setattr(server, "_do_stop_process",
                            lambda slug: server.ProcessActionResult(ok=True, slug=slug, message="x"))
        async with _client() as c:
            await c.get("/fd/processes", headers=_hdr(ADMIN, BOB))
            await c.post("/fd/processes/bob-agent/stop", headers=_hdr(ADMIN, BOB))
        assert await _usage(deck, ADMIN) == [("admin_act_as", {
            "target_user": BOB, "method": "POST", "path": "/fd/processes/bob-agent/stop"})]

    async def test_a_spawn_names_the_admin_and_spends_the_admins_rate_budget(self, deck, monkeypatch):
        charged: list[str] = []
        capped: list[str] = []
        monkeypatch.setattr(server, "check_spawn_rate_limit", lambda u: charged.append(u["id"]))
        monkeypatch.setattr(server, "check_api_rate_limit", lambda u: charged.append(u["id"]))

        async def _count(user, owned):
            capped.append(user["id"])

        monkeypatch.setattr(server, "check_agent_count_limit", _count)
        async with _client() as c:
            r = await c.post("/fd/spawn-process", headers=_hdr(ADMIN, BOB), json={
                "name": "Analyst", "archetype": "analyst", "web_port": 24200})
        assert r.status_code == 200, r.text
        assert charged == [ADMIN, ADMIN] and capped == [BOB]   # budget: admin; cap: owner
        spawn = [d for e, d in await _usage(deck, BOB) if e == "agent_spawn"]
        assert spawn and spawn[0]["acting_admin"] == ADMIN

    async def test_the_admin_gets_the_list_not_the_agents_tokens(self, deck):
        async with _client() as c:
            mine = await c.get("/fd/processes", headers=_hdr(BOB))
            theirs = await c.get("/fd/processes", headers=_hdr(ADMIN, BOB))
        assert [p["web_auth"] for p in mine.json()] == ["bob-tok"]
        assert [p["web_auth"] for p in theirs.json()] == [""]

    async def test_an_admins_rename_is_not_hidden_by_the_owners_own_labels(self, deck):
        await deck.set_settings(BOB, {
            "fd:process-names": json.dumps({"bob-agent": "Bob's label", "x": "keep"}),
            "fd:process-descriptions": json.dumps({"bob-agent": "old"})})
        async with _client() as c:
            await c.post("/fd/processes/bob-agent/identity", headers=_hdr(ADMIN, BOB),
                         json={"name": "Renamed"})
        assert json.loads(await deck.get_setting(BOB, "fd:process-names")) == {"x": "keep"}
        # description wasn't part of the change: the owner's label for it stays
        assert json.loads(await deck.get_setting(BOB, "fd:process-descriptions")) == {"bob-agent": "old"}

    async def test_the_owners_own_rename_leaves_their_labels_alone(self, deck):
        await deck.set_settings(BOB, {"fd:process-names": json.dumps({"bob-agent": "mine"})})
        async with _client() as c:
            await c.post("/fd/processes/bob-agent/identity", headers=_hdr(BOB), json={"name": "N"})
        assert json.loads(await deck.get_setting(BOB, "fd:process-names")) == {"bob-agent": "mine"}


# ── instructions: one entry at a time, never at the map's expense ───────────


class TestInstructionWrites:
    async def test_the_agent_list_carries_them_for_the_owners_flight_deck(self, deck):
        await deck.set_settings(BOB, {INSTR: json.dumps({"bob-agent": "Be brief."})})
        async with _client() as c:
            r = await c.get("/fd/processes", headers=_hdr(BOB))
        assert [(p["slug"], p["fleet_instructions"]) for p in r.json()] == [("bob-agent", "Be brief.")]

    async def test_a_failed_read_aborts_the_write_instead_of_wiping_the_map(self, deck, monkeypatch):
        await deck.set_settings(BOB, {INSTR: json.dumps({"bob-agent": "old", "other": "kept"})})
        real = deck.get_setting

        async def flaky(uid, key):
            if key == INSTR:
                raise RuntimeError("database is locked")
            return await real(uid, key)

        monkeypatch.setattr(deck, "get_setting", flaky)
        async with _client() as c:
            r = await c.put("/fd/agent-instructions/process/bob-agent", headers=_hdr(ADMIN, BOB),
                            json={"instructions": "new"})
        monkeypatch.setattr(deck, "get_setting", real)
        assert r.status_code == 500 and "nothing was changed" in r.json()["detail"]
        assert await _instructions(deck, BOB) == {"bob-agent": "old", "other": "kept"}

    async def test_instructions_are_capped(self, deck):
        async with _client() as c:
            put = await c.put("/fd/agent-instructions/process/bob-agent", headers=_hdr(BOB),
                              json={"instructions": "x" * 64_001})
            spawn = await c.post("/fd/spawn-process", headers=_hdr(BOB), json={
                "name": "Big", "fleet_instructions": "x" * 64_001, "web_port": 24200})
        assert put.status_code == 422 and spawn.status_code == 422
        assert await _instructions(deck, BOB) == {}

    async def test_removing_an_agent_removes_its_instructions(self, deck):
        await deck.set_settings(BOB, {INSTR: json.dumps({"bob-agent": "old", "other": "kept"})})
        async with _client() as c:
            await c.delete("/fd/processes/bob-agent", headers=_hdr(BOB))
        assert await _instructions(deck, BOB) == {"other": "kept"}

    async def test_an_agent_cannot_plant_its_own_text_in_its_owners_settings(self, deck):
        """An agent (no user session, identified by X-Agent-Auth) spawning a
        child: text from its request body is not stored — only an archetype's."""
        async with _client(LOOPBACK) as c:
            plain = await c.post("/fd/spawn-process", headers={"X-Agent-Auth": "bob-tok"}, json={
                "name": "Child", "fleet_instructions": "PLANTED", "web_port": 24200})
            arch = await c.post("/fd/spawn-process", headers={"X-Agent-Auth": "bob-tok"}, json={
                "name": "Kid", "archetype": "analyst", "fleet_instructions": "PLANTED",
                "web_port": 24201})
        assert plain.status_code == 200 and arch.status_code == 200, (plain.text, arch.text)
        assert server._load_process_registry()["child"]["owner"] == BOB
        assert await _instructions(deck, BOB) == {"kid": "You are a careful data analyst."}

    async def test_without_accounts_the_route_says_so(self, deck, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        async with _client() as c:
            r = await c.put("/fd/agent-instructions/process/bob-agent", headers=_hdr(BOB),
                            json={"instructions": "x"})
        assert r.status_code == 400
