"""Owner profile routes: /fd/profile and /fd/admin/profile-defaults.

Pinned here:

* /fd/profile is the signed-in user's own profile — 401 without a session, and
  an admin's ``X-FD-Act-As`` is ignored (it is not a login-as);
* a PUT is a partial merge, each field a string of valid (UTF-8 encodable)
  text within its cap (400 otherwise), and rewrites that owner's agents' context
  files only — in the one location each agent's runtime reads;
* the deck defaults are admin-only, audited, and reach every agent;
* auth-off decks: the local user's profile lives in system_settings and the
  local user may set the deck defaults;
* the generic settings routes can't read or write ``fd:tenant-profile``;
* deleting a user removes their agents' files; a display-name change rewrites
  them; a spawn (process or Docker) and a clone write the new agent's, replacing
  whatever an earlier agent of that slug left.

Real FlightDeckDB + process registry in tmp dirs; the agent process is faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck import tenant_profile as tp
from captain_claw.flight_deck.auth import ACT_AS_HEADER, create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

ADMIN = "user-admin"
BOB = "user-bob"
CAROL = "user-carol"
REMOTE = ("203.0.113.9", 40001)
EMPTY = {"about_me": "", "company": "", "instructions": ""}


class _FakePopen:
    def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None, start_new_session=None):
        self.pid = 434343
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
    for uid, role, name in ((ADMIN, "admin", "Ada Admin"), (BOB, "user", "Bob Builder"),
                            (CAROL, "user", "Carol")):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", name, role))
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fd-env-key")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)

    def _no_docker():
        raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server, "get_docker", _no_docker)
    monkeypatch.setattr(server.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(server, "_is_port_available", lambda port: True)
    monkeypatch.setattr(server, "_schedule_fleet_notify", lambda *a, **k: None)
    monkeypatch.setattr(rate_limiter, "_spawn_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
        "carol-agent": {"slug": "carol-agent", "name": "Carol's agent", "web_port": 24102,
                        "web_auth": "carol-tok", "owner": CAROL, "pid": None},
    })
    for slug in ("bob-agent", "carol-agent"):
        (data / slug).mkdir()
    try:
        yield db
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=REMOTE), base_url="http://fd.test")


def _hdr(caller: str, act_as: str | None = None) -> dict:
    h = {"Authorization": f"Bearer {create_access_token(caller)}"}
    if act_as is not None:
        h[ACT_AS_HEADER] = act_as
    return h


def _context(slug: str, compact: bool = False) -> dict[str, str]:
    """The agent's context file in each runtime's location, by runtime
    (missing ones left out)."""
    name = tp.COMPACT_FILE if compact else tp.FULL_FILE
    out = {}
    for runtime in tp.RUNTIMES:
        path = tp.context_dir(server.DATA_DIR / slug, runtime) / name
        if path.is_file():
            out[runtime] = path.read_text(encoding="utf-8")
    return out


# ── /fd/profile ─────────────────────────────────────────────────────────────


class TestProfile:
    async def test_needs_a_session(self, deck):
        async with _client() as c:
            got = await c.get("/fd/profile")
            put = await c.put("/fd/profile", json={"about_me": "x"})
        assert got.status_code == 401 and put.status_code == 401

    async def test_empty_to_start_with(self, deck):
        async with _client() as c:
            r = await c.get("/fd/profile", headers=_hdr(BOB))
        assert r.status_code == 200
        assert r.json() == {
            "profile": EMPTY,
            "deck": {"company": "", "instructions": ""},
            "caps": {"about_me": 1500, "company": 4000, "instructions": 2000},
            "preview": {"full": "", "compact": ""},
        }

    async def test_save_previews_and_reaches_the_owners_agents_only(self, deck):
        async with _client() as c:
            r = await c.put("/fd/profile", headers=_hdr(BOB), json={
                "about_me": "  I build bridges.  ", "company": "Bridges Ltd.",
                "instructions": "Use metric units."})
            got = await c.get("/fd/profile", headers=_hdr(BOB))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["agents_updated"] == 1
        assert body["profile"] == {"about_me": "I build bridges.", "company": "Bridges Ltd.",
                                   "instructions": "Use metric units."}
        assert "You work for Bob Builder" in body["preview"]["full"]
        assert "Use metric units." in body["preview"]["compact"]
        assert got.json()["profile"] == body["profile"]
        assert "agents_updated" not in got.json()
        assert _context("bob-agent") == {"process": body["preview"]["full"]}
        assert _context("bob-agent", compact=True) == {"process": body["preview"]["compact"]}
        assert _context("carol-agent") == {}
        stored = json.loads(await deck.get_setting(BOB, tp.PROFILE_SETTING))
        assert stored["company"] == "Bridges Ltd."

    async def test_a_put_is_a_partial_merge(self, deck):
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": "Me.", "company": "Co."})
            r = await c.put("/fd/profile", headers=_hdr(BOB), json={"company": "", "instructions": None})
        assert r.json()["profile"] == {"about_me": "Me.", "company": "", "instructions": ""}

    async def test_clearing_everything_removes_the_files(self, deck):
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": "Me."})
            assert _context("bob-agent")
            r = await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": ""})
        assert r.json()["agents_updated"] == 1 and r.json()["preview"] == {"full": "", "compact": ""}
        assert _context("bob-agent") == {} and _context("bob-agent", compact=True) == {}

    @pytest.mark.parametrize("payload", [
        {"about_me": "x" * 1501},
        {"company": "x" * 4001},
        {"instructions": "x" * 2001},
        {"about_me": 5},
        {"company": ["x"]},
    ])
    async def test_bad_fields_are_a_400_and_change_nothing(self, deck, payload):
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": "Kept."})
            r = await c.put("/fd/profile", headers=_hdr(BOB), json={"instructions": "New.", **payload})
        assert r.status_code == 400
        assert await tp.load_profile(deck, BOB) == {**EMPTY, "about_me": "Kept."}

    async def test_text_that_cant_be_encoded_is_a_400(self, deck):
        """JSON lets a lone surrogate through; the agents' files are UTF-8, so
        it would fail every write (and used to abort the save with a 500)."""
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": "Kept."})
            r = await c.put("/fd/profile", headers={**_hdr(BOB), "Content-Type": "application/json"},
                            content=b'{"instructions": "Hi \\ud800 there"}')
            d = await c.put("/fd/admin/profile-defaults",
                            headers={**_hdr(ADMIN), "Content-Type": "application/json"},
                            content=b'{"company": "\\udfff"}')
        assert r.status_code == 400 and "instructions" in r.json()["detail"]
        assert d.status_code == 400 and "company" in d.json()["detail"]
        assert await tp.load_profile(deck, BOB) == {**EMPTY, "about_me": "Kept."}
        assert await tp.load_deck(deck) == {"company": "", "instructions": ""}
        assert "Kept." in _context("bob-agent")["process"]

    async def test_at_the_cap_is_fine(self, deck):
        async with _client() as c:
            r = await c.put("/fd/profile", headers=_hdr(BOB), json={
                "about_me": "a" * 1500, "company": "c" * 4000, "instructions": "i" * 2000})
        assert r.status_code == 200
        assert len(r.json()["preview"]["compact"]) <= tp.COMPACT_MAX

    async def test_act_as_is_ignored(self, deck):
        """An admin's act-as header reaches agent management, not a profile:
        the admin reads and writes their own."""
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": "Bob's secret."})
            got = await c.get("/fd/profile", headers=_hdr(ADMIN, BOB))
            put = await c.put("/fd/profile", headers=_hdr(ADMIN, BOB), json={"about_me": "Admin."})
        assert got.status_code == 200 and got.json()["profile"] == EMPTY
        assert "Bob's secret." not in got.text
        assert put.status_code == 200 and put.json()["agents_updated"] == 0
        assert (await tp.load_profile(deck, BOB))["about_me"] == "Bob's secret."
        assert (await tp.load_profile(deck, ADMIN))["about_me"] == "Admin."

    async def test_a_non_admins_act_as_is_ignored_too(self, deck):
        async with _client() as c:
            r = await c.get("/fd/profile", headers=_hdr(CAROL, BOB))
        assert r.status_code == 200 and r.json()["profile"] == EMPTY


# ── /fd/admin/profile-defaults ──────────────────────────────────────────────


class TestDeckDefaults:
    async def test_admin_only(self, deck):
        async with _client() as c:
            got = await c.get("/fd/admin/profile-defaults", headers=_hdr(BOB))
            put = await c.put("/fd/admin/profile-defaults", headers=_hdr(BOB), json={"company": "x"})
            anon = await c.get("/fd/admin/profile-defaults")
        assert got.status_code == 403 and put.status_code == 403 and anon.status_code == 401
        assert await tp.load_deck(deck) == {"company": "", "instructions": ""}

    async def test_save_audits_and_reaches_every_agent(self, deck):
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(CAROL), json={"company": "Carol's Shop"})
            put = await c.put("/fd/admin/profile-defaults", headers=_hdr(ADMIN), json={
                "company": "Deck Co.", "instructions": "Never share credentials."})
            got = await c.get("/fd/admin/profile-defaults", headers=_hdr(ADMIN))
            bob = await c.get("/fd/profile", headers=_hdr(BOB))
        assert put.status_code == 200, put.text
        assert put.json() == {"company": "Deck Co.", "instructions": "Never share credentials.",
                              "agents_updated": 2}
        assert got.json() == {"company": "Deck Co.", "instructions": "Never share credentials.",
                              "caps": {"company": 4000, "instructions": 2000}}
        assert bob.json()["deck"] == {"company": "Deck Co.", "instructions": "Never share credentials."}
        assert "### About their company\n> Deck Co." in bob.json()["preview"]["full"]
        bob_file = _context("bob-agent")["process"]
        carol_file = _context("carol-agent")["process"]
        assert "Deck Co." in bob_file and "Never share credentials." in bob_file
        assert "Carol's Shop" in carol_file and "Deck Co." not in carol_file
        assert "Never share credentials." in carol_file
        rows = await deck.get_usage_logs(user_id=ADMIN, event_type="profile_defaults_update")
        assert len(rows) == 1
        assert json.loads(rows[0]["detail"]) == {
            "fields": ["company", "instructions"], "company_chars": 8,
            "instructions_chars": 24, "agents_updated": 2}

    async def test_partial_merge_and_caps(self, deck):
        async with _client() as c:
            await c.put("/fd/admin/profile-defaults", headers=_hdr(ADMIN), json={"company": "Deck Co."})
            r = await c.put("/fd/admin/profile-defaults", headers=_hdr(ADMIN), json={"instructions": "Hi."})
            big = await c.put("/fd/admin/profile-defaults", headers=_hdr(ADMIN),
                              json={"company": "c" * 4001})
        assert r.json()["company"] == "Deck Co." and r.json()["instructions"] == "Hi."
        assert big.status_code == 400
        assert await tp.load_deck(deck) == {"company": "Deck Co.", "instructions": "Hi."}


# ── auth-off decks ──────────────────────────────────────────────────────────


class TestAuthOff:
    async def test_the_local_user_profile_is_a_system_setting(self, deck, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        async with _client() as c:
            r = await c.put("/fd/profile", json={"about_me": "Solo developer."})
            got = await c.get("/fd/profile")
        assert r.status_code == 200, r.text
        assert r.json()["agents_updated"] == 2  # one tenant: every agent is theirs
        assert got.json()["profile"]["about_me"] == "Solo developer."
        assert "You work for your owner," in got.json()["preview"]["full"]
        raw = await deck.get_system_setting(tp.LOCAL_PROFILE_SETTING)
        assert json.loads(raw)["about_me"] == "Solo developer."
        assert await deck.get_all_settings("local") == {}
        assert "Solo developer." in _context("carol-agent")["process"]

    async def test_the_local_user_sets_the_deck_defaults(self, deck, monkeypatch):
        monkeypatch.setenv("FD_AUTH_ENABLED", "false")
        async with _client() as c:
            put = await c.put("/fd/admin/profile-defaults", json={"company": "Home Lab"})
            got = await c.get("/fd/admin/profile-defaults")
        assert put.status_code == 200 and put.json()["agents_updated"] == 2
        assert got.json()["company"] == "Home Lab"
        rows = await deck.get_usage_logs(user_id="local", event_type="profile_defaults_update")
        assert len(rows) == 1


# ── the generic settings routes ─────────────────────────────────────────────


class TestSettingsRoutesRefuse:
    async def test_cannot_write_read_or_delete_it(self, deck):
        await tp.save_profile(deck, BOB, {"about_me": "Me."})
        async with _client() as c:
            put = await c.put("/fd/settings", headers=_hdr(BOB), json={"settings": {
                "fd:tenant-profile": json.dumps({"about_me": "x" * 9000})}})
            put_local = await c.put("/fd/settings", headers=_hdr(BOB), json={"settings": {
                "FD:Tenant-Profile:local": "{}"}})
            got = await c.get("/fd/settings", headers=_hdr(BOB))
            dele = await c.delete("/fd/settings/fd:tenant-profile", headers=_hdr(BOB))
        assert put.status_code == 400 and put_local.status_code == 400 and dele.status_code == 400
        assert got.status_code == 200 and "fd:tenant-profile" not in got.json()
        assert (await tp.load_profile(deck, BOB))["about_me"] == "Me."


# ── user deletion and spawn ─────────────────────────────────────────────────


class TestLifecycle:
    async def test_deleting_a_user_removes_their_agents_files(self, deck):
        async with _client() as c:
            await c.put("/fd/admin/profile-defaults", headers=_hdr(ADMIN), json={"company": "Deck Co."})
            await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": "Me."})
            assert _context("bob-agent") and _context("carol-agent")
            r = await c.delete(f"/fd/admin/users/{BOB}", headers=_hdr(ADMIN))
        assert r.status_code == 200
        assert _context("bob-agent") == {} and _context("bob-agent", compact=True) == {}
        assert _context("carol-agent")  # nobody else's

    async def test_a_display_name_change_rewrites_the_owners_agents(self, deck, monkeypatch):
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"about_me": "Me."})
            await c.put("/fd/profile", headers=_hdr(CAROL), json={"about_me": "Carol."})
            me = await c.put("/fd/auth/me", headers=_hdr(BOB), json={"display_name": "Robert"})
            assert me.status_code == 200, me.text
            assert "You work for Robert," in _context("bob-agent")["process"]
            assert "You work for Robert" in _context("bob-agent", compact=True)["process"]
            adm = await c.put(f"/fd/admin/users/{BOB}", headers=_hdr(ADMIN),
                              json={"display_name": "Bobby"})
            assert adm.status_code == 200, adm.text
            assert "You work for Bobby," in _context("bob-agent")["process"]
            assert "You work for Carol," in _context("carol-agent")["process"]  # untouched

            calls: list = []

            async def spy(db, owner_id=None):
                calls.append(owner_id)
                return 0

            monkeypatch.setattr(tp, "refresh_agents", spy)
            await c.put("/fd/auth/me", headers=_hdr(BOB), json={"display_name": "Bobby"})
            await c.put(f"/fd/admin/users/{BOB}", headers=_hdr(ADMIN), json={"role": "user"})
            assert calls == []  # unchanged name: nothing to rewrite
            await c.put(f"/fd/admin/users/{BOB}", headers=_hdr(ADMIN), json={"display_name": "B."})
            assert calls == [BOB]

    async def test_a_failing_refresh_never_fails_a_rename(self, deck, monkeypatch):
        async def boom(db, owner_id=None):
            raise RuntimeError("registry unreadable")

        monkeypatch.setattr(tp, "refresh_agents", boom)
        async with _client() as c:
            me = await c.put("/fd/auth/me", headers=_hdr(BOB), json={"display_name": "Robert"})
            adm = await c.put(f"/fd/admin/users/{CAROL}", headers=_hdr(ADMIN),
                              json={"display_name": "Caroline"})
        assert me.status_code == 200 and me.json()["display_name"] == "Robert"
        assert adm.status_code == 200
        assert (await deck.get_user_by_id(CAROL))["display_name"] == "Caroline"

    async def test_a_spawn_writes_the_new_agents_files(self, deck):
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"instructions": "Answer in Croatian."})
            r = await c.post("/fd/spawn-process", headers=_hdr(BOB), json={
                "name": "Fresh", "web_port": 24300})
            mine = await c.post("/fd/spawn-process", headers=_hdr(CAROL), json={
                "name": "Plain", "web_port": 24301})
        assert r.status_code == 200, r.text
        assert mine.status_code == 200, mine.text
        assert set(_context("fresh")) == {"process"}  # only where a process agent reads it
        assert "Answer in Croatian." in _context("fresh")["process"]
        assert "Answer in Croatian." in _context("fresh", compact=True)["process"]
        assert _context("plain") == {}  # Carol has no profile

    async def test_a_spawn_replaces_what_an_earlier_agent_of_the_slug_left(self, deck):
        _stale("plain", "process")
        async with _client() as c:
            r = await c.post("/fd/spawn-process", headers=_hdr(CAROL), json={
                "name": "Plain", "web_port": 24301})
        assert r.status_code == 200, r.text
        assert _context("plain") == {} and _context("plain", compact=True) == {}

    async def test_a_process_clone_gets_its_owners_profile(self, deck):
        _stale("bob-two", "process")
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"instructions": "Answer in Croatian."})
            r = await c.post("/fd/processes/bob-agent/clone", headers=_hdr(BOB),
                             json={"new_name": "Bob two"})
            await c.put("/fd/profile", headers=_hdr(CAROL), json={"about_me": "Carol."})
            _stale("carol-two", "process", text="Somebody else's profile.")
            await c.put("/fd/profile", headers=_hdr(CAROL), json={"about_me": ""})
            plain = await c.post("/fd/processes/carol-agent/clone", headers=_hdr(CAROL),
                                 json={"new_name": "Carol two"})
        assert r.status_code == 200, r.text
        assert plain.status_code == 200, plain.text
        clone = _context("bob-two")
        assert set(clone) == {"process"}
        assert "Answer in Croatian." in clone["process"] and "Carol" not in clone["process"]
        assert "You work for Bob Builder" in _context("bob-two", compact=True)["process"]
        assert _context("carol-two") == {}  # no profile: the stale one is gone too

    async def test_a_docker_spawn_and_clone_write_where_the_container_reads(self, deck, monkeypatch):
        boxes = _docker(monkeypatch, deck)
        _stale("boxed-two", "docker")
        async with _client() as c:
            await c.put("/fd/profile", headers=_hdr(BOB), json={"instructions": "Answer in Croatian."})
            r = await c.post("/fd/spawn", headers=_hdr(BOB), json={"name": "Boxed", "web_port": 24310})
            assert r.status_code == 200, r.text
            cl = await c.post("/fd/containers/boxed/clone", headers=_hdr(BOB),
                              json={"new_name": "Boxed two"})
        assert cl.status_code == 200, cl.text
        assert [b.name for b in boxes.items] == ["boxed", "boxed-two"]
        for slug in ("boxed", "boxed-two"):
            files = _context(slug)
            assert set(files) == {"docker"}  # home-config: the container's ~/.captain-claw
            assert "Answer in Croatian." in files["docker"] and "Carol" not in files["docker"]
            assert "Answer in Croatian." in _context(slug, compact=True)["docker"]


def _stale(slug: str, runtime: str, text: str = "Carol's profile.") -> None:
    """Files an earlier agent of ``slug`` (another owner's) left behind."""
    tp.write_for_agent_dir(server.DATA_DIR / slug, runtime, text, text)


class _Box:
    def __init__(self, name: str, labels: dict):
        self.name, self.id, self.short_id = name, f"id-{name}", name[:12]
        self.labels, self.status = dict(labels), "running"
        self.attrs = {"Config": {"Env": []}, "Mounts": [], "HostConfig": {},
                      "NetworkSettings": {"Ports": {}}}

    def remove(self, force: bool = False) -> None:
        pass


class _Boxes:
    """The slice of docker-py's container API the spawn and clone routes use."""

    def __init__(self):
        self.items: list[_Box] = []

    def list(self, all: bool = False, filters: dict | None = None):
        label = (filters or {}).get("label")
        return [b for b in self.items if not label or label in b.labels]

    def get(self, name: str):
        for b in self.items:
            if b.name == name:
                return b
        raise server.docker.errors.NotFound("no such container")

    def run(self, **kw):
        box = _Box(kw["name"], kw.get("labels") or {})
        self.items.append(box)
        return box


def _docker(monkeypatch, db) -> _Boxes:
    boxes = _Boxes()
    monkeypatch.setattr(server, "get_docker", lambda: SimpleNamespace(containers=boxes))

    async def _sys_cfg():
        return {"docker_spawn_enabled": True}

    monkeypatch.setattr(server, "_get_system_config", _sys_cfg)
    monkeypatch.setattr(server.app.state, "fd_db", db, raising=False)
    return boxes
