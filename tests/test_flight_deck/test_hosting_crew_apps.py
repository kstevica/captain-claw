"""A hosted app for an approved Crew user who isn't an admin.

Running an app (a published folder whose start command runs as a server, behind
/vfs-apps/<name>/) used to be an admin's alone on a sign-in deck. Now an admin can
tick "Hosted apps" on someone's Crew approval. What their app gets is what their
Crew worker already has, and no more: the deck's OS user, their own Crew home, a
short list of variables, never the deck's secrets. It runs on its *owner's*
approval, whoever presses Start, an admin included; within limits (how many at
once, a port range, memory, running time); and when the approval goes, so does
the app, keeping its address. A stopped app is a 404 to the public, as a stopped
page is.

Real processes: each app here is a few lines of Python that says what it can see.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import socket
import sys
import time
import types
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import admin_routes, crew_access, crew_logins, crew_runner, crew_store, hosting_routes, vfs_hosting
from captain_claw.flight_deck import auth as auth_mod
from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import get_current_user, get_optional_user
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.vfs import user_root_of

DECK_SECRET = "sk-deck-made-up-0000"
JWT_SECRET = "jwt-made-up-0000"

APP = """
import http.server, json, os
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            "home": os.environ.get("HOME", ""), "key": os.environ.get("OPENAI_API_KEY", ""),
            "jwt": os.environ.get("FD_JWT_SECRET", ""), "data_dir": os.environ.get("FD_DATA_DIR", ""),
            "nice": os.nice(0), "cwd": os.getcwd(), "pid": os.getpid(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
"""
START = f"{shlex.quote(sys.executable)} server.py"
_ENV = ("FD_CREW", "FD_CREW_APPS", "FD_CREW_APPS_PER_USER", "FD_CREW_APPS_MAX_RUNNING", "FD_CREW_APP_MEMORY_MB",
        "FD_CREW_APP_HOURS", "VFS_APPS_PORT_SPAN", "CLAW_VFS_ROOT", "FD_LOCKDOWN", "FD_ALLOWED_HOSTS")


def _free_base(span: int = 20) -> int:
    """A run of ports nothing here listens on, so a real app can have one."""
    for base in range(47100, 49000, span):
        ok = True
        for port in range(base, base + span):
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    ok = False
                    break
        if ok:
            return base
    raise RuntimeError("no free ports")


def _crew(apps: bool) -> dict:
    return {"crew": {"approved": True, "harnesses": ["opencode"], "max_parallel": 2, "apps": apps}}


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fd_server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_PUBLIC_URL", "https://deck.example.test")
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VFS_APPS_PORT_BASE", str(_free_base()))
    monkeypatch.setenv("VFS_APPS_PORT_SPAN", "20")
    # The deck's own secrets, which a non-admin's app must never see.
    monkeypatch.setenv("OPENAI_API_KEY", DECK_SECRET)
    monkeypatch.setenv("FD_JWT_SECRET", JWT_SECRET)
    crew_store.set_root(tmp_path / "crew")
    monkeypatch.setattr(crew_runner, "_QUEUE_LOCK", asyncio.Lock())
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    monkeypatch.setattr(auth_mod, "_db", db)
    admin = await db.create_user("admin@example.com", "x", role="admin")
    maker = await db.create_user("maker@example.com", "x", metadata=_crew(apps=True))
    coder = await db.create_user("coder@example.com", "x", metadata=_crew(apps=False))
    guest = await db.create_user("guest@example.com", "x")
    second = await db.create_user("second@example.com", "x", metadata=_crew(apps=True))
    for user in (admin, maker, coder, guest, second):
        folder = user_root_of(user["id"]) / "web"
        folder.mkdir(parents=True)
        (folder / "server.py").write_text(APP)
        (folder / "index.html").write_text("<h1>page</h1>")
    yield types.SimpleNamespace(db=db, admin=admin, maker=maker, coder=coder, guest=guest, second=second, tmp=tmp_path)
    for name in list(vfs_hosting.load_registry()):
        vfs_hosting.stop_app(name)
    crew_store.get_store().close()
    crew_store.set_root(None)
    await db.close()


def _client(user: dict | None = None) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(hosting_routes.router)
    app.include_router(admin_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_optional_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 40001)),
                             base_url="http://deck.test", timeout=30)


async def _publish(client: httpx.AsyncClient, name: str, **over) -> httpx.Response:
    return await client.post("/fd/hosting", json={"name": name, "kind": "app", "project": "web", "start_cmd": START, **over})


async def _seen(public: httpx.AsyncClient, name: str) -> dict:
    """What the app says it sees, once it's up."""
    end = time.monotonic() + 15
    while time.monotonic() < end:
        resp = await public.get(f"/vfs-apps/{name}/")
        if resp.status_code == 200:
            return resp.json()
        await asyncio.sleep(0.1)
    raise AssertionError(f"{name} never answered: {resp.status_code} {resp.text}\n{vfs_hosting.read_app_log(name)}")


async def _listed(client: httpx.AsyncClient, name: str) -> dict:
    return next(e for e in (await client.get("/fd/hosting")).json()["entries"] if e["name"] == name)


# ── The approval ─────────────────────────────────────────────────────────


def _row(role: str = "user", **meta) -> dict:
    return {"id": "u1", "role": role, "metadata": json.dumps(meta)}


def test_hosted_apps_is_a_tick_on_the_crew_approval(monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.delenv("FD_CREW_APPS", raising=False)
    monkeypatch.delenv("FD_CREW", raising=False)
    assert crew_access.grant_of(_row(**_crew(apps=True))).apps is True
    assert crew_access.grant_of(_row(**_crew(apps=False))).apps is False
    assert crew_access.grant_of(_row(crew={"approved": True, "harnesses": ["pi"], "apps": "yes"})).apps is False
    # Hosted apps without Crew approves nothing: it's the worker's reach that's lent.
    assert crew_access.app_reach(_row(crew={"approved": False, "apps": True})) == ""
    assert crew_access.app_reach(_row(crew={"approved": True, "harnesses": [], "apps": True})) == ""
    assert crew_access.app_reach(_row(**_crew(apps=True))) == "crew"
    assert crew_access.app_reach(_row(**_crew(apps=False))) == ""
    assert crew_access.app_reach(_row()) == ""
    assert crew_access.app_reach(None) == ""
    # An admin's apps run as Flight Deck's own processes do, as they always did.
    assert crew_access.app_reach(_row(role="admin")) == "deck"
    stored = crew_access.clean_entry({"approved": True, "harnesses": ["pi"], "apps": True}, admin_id="a")
    assert stored["apps"] is True
    assert crew_access.clean_entry({"approved": True, "harnesses": ["pi"], "apps": 1}, admin_id="a")["apps"] is False


def test_the_deck_can_take_hosted_apps_away_from_everyone(monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_CREW_APPS", "off")
    assert crew_access.grant_of(_row(**_crew(apps=True))).apps is False
    assert crew_access.app_reach(_row(**_crew(apps=True))) == ""
    assert crew_access.app_reach(_row(role="admin")) == "deck"
    monkeypatch.delenv("FD_CREW_APPS")
    monkeypatch.setenv("FD_CREW", "off")
    assert crew_access.app_reach(_row(**_crew(apps=True))) == ""


def test_a_deck_without_sign_in_runs_anyones_app_as_before(monkeypatch):
    monkeypatch.setenv("FD_AUTH_ENABLED", "false")
    assert crew_access.app_reach(None) == "deck"
    assert crew_access.app_reach(_row()) == "deck"


def test_the_limits_come_from_the_deck(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    assert crew_access.app_limits() == crew_access.AppLimits(per_user=2, deck=6, memory_mb=1024, hours=0)
    monkeypatch.setenv("FD_CREW_APPS_PER_USER", "0")  # at least one, or the tick means nothing
    monkeypatch.setenv("FD_CREW_APPS_MAX_RUNNING", "nonsense")
    monkeypatch.setenv("FD_CREW_APP_MEMORY_MB", "256")
    monkeypatch.setenv("FD_CREW_APP_HOURS", "12")
    assert crew_access.app_limits() == crew_access.AppLimits(per_user=1, deck=6, memory_mb=256, hours=12)


# ── Who may ──────────────────────────────────────────────────────────────


async def test_only_an_admin_or_someone_approved_for_hosted_apps_can_publish_one(deck):
    for who in (deck.guest, deck.coder):
        async with _client(who) as client:
            refused = await _publish(client, f"app-{who['email'][:4]}")
            assert refused.status_code == 403 and "Hosted apps" in refused.json()["detail"]
            # A page stays anyone's own.
            page = await client.post("/fd/hosting", json={"name": f"page-{who['email'][:4]}", "kind": "static", "project": "web"})
            assert page.status_code == 200
            # Not by the back door either: a page edited into an app.
            edit = await client.put(f"/fd/hosting/page-{who['email'][:4]}", json={"kind": "app", "project": "web", "start_cmd": START})
            assert edit.status_code == 403
    for who in (deck.maker, deck.admin):
        async with _client(who) as client:
            assert (await _publish(client, f"app-{who['email'][:4]}")).status_code == 200


async def test_an_app_runs_on_its_owners_approval_whoever_presses_start(deck):
    registry = vfs_hosting.load_registry()
    registry["theirs"] = {"kind": "app", "owner": deck.guest["id"], "project": "web", "start_cmd": START, "stopped": True}
    vfs_hosting.save_registry(registry)
    async with _client(deck.admin) as admin, _client(deck.guest) as guest:
        # Started by an admin it would run as that admin's say-so: refused, for the admin too.
        for client in (admin, guest):
            refused = await client.post("/fd/hosting/theirs/start")
            assert refused.status_code == 403 and "Hosted apps" in refused.json()["detail"]
        assert not vfs_hosting.app_is_alive("theirs")
        assert (await admin.post("/fd/hosting/theirs/restart")).status_code == 403


# ── What a non-admin's app gets ─────────────────────────────────────────


async def test_a_non_admins_app_sees_what_their_crew_worker_sees_and_no_deck_secret(deck):
    async with _client(deck.maker) as maker, _client() as public:
        assert (await _publish(maker, "mine")).status_code == 200
        started = await maker.post("/fd/hosting/mine/start")
        assert started.status_code == 200, started.text
        seen = await _seen(public, "mine")
        assert seen["home"] == str(crew_logins.worker_home(deck.maker["id"]))
        assert seen["key"] == "" and seen["jwt"] == "" and seen["data_dir"] == ""
        assert seen["nice"] >= 10  # below the deck in the CPU's queue
        assert Path(seen["cwd"]).resolve() == (user_root_of(deck.maker["id"]) / "web").resolve()
        entry = vfs_hosting.load_registry()["mine"]
        assert entry["reach"] == "crew" and entry["port"] in vfs_hosting.port_range()


async def test_an_admins_app_keeps_the_decks_environment_minus_its_own_secrets(deck):
    async with _client(deck.admin) as admin, _client() as public:
        await _publish(admin, "deckapp")
        assert (await admin.post("/fd/hosting/deckapp/start")).status_code == 200
        seen = await _seen(public, "deckapp")
        assert seen["key"] == DECK_SECRET and seen["jwt"] == "" and seen["nice"] == os.nice(0)
        assert vfs_hosting.load_registry()["deckapp"]["reach"] == "deck"


async def test_an_admin_starting_someones_app_runs_it_as_them_not_as_the_admin(deck):
    async with _client(deck.maker) as maker, _client(deck.admin) as admin, _client() as public:
        await _publish(maker, "for-me")
        assert (await admin.post("/fd/hosting/for-me/start")).status_code == 200
        seen = await _seen(public, "for-me")
        assert seen["home"] == str(crew_logins.worker_home(deck.maker["id"])) and seen["key"] == ""


# ── Start, stop, restart, remove ────────────────────────────────────────


async def test_stop_start_restart_and_remove_and_a_stopped_app_is_a_404_at_the_same_address(deck):
    async with _client(deck.maker) as maker, _client() as public:
        await _publish(maker, "cycle")
        await maker.post("/fd/hosting/cycle/start")
        first = (await _seen(public, "cycle"))["pid"]

        assert (await maker.post("/fd/hosting/cycle/stop")).json()["ok"] is True
        assert (await public.get("/vfs-apps/cycle/")).status_code == 404
        assert (await public.get("/vfs-apps/cycle/anything")).status_code == 404
        listed = await _listed(maker, "cycle")
        assert listed["running"] is False and listed["url"] == "/vfs-apps/cycle/"

        assert (await maker.post("/fd/hosting/cycle/start")).status_code == 200
        second = (await _seen(public, "cycle"))["pid"]
        assert second != first and (await _listed(maker, "cycle"))["running"] is True

        assert (await maker.post("/fd/hosting/cycle/restart")).status_code == 200
        third = (await _seen(public, "cycle"))["pid"]
        assert third != second

        assert (await maker.delete("/fd/hosting/cycle")).json()["ok"] is True
        assert (await public.get("/vfs-apps/cycle/")).status_code == 404
        assert all(e["name"] != "cycle" for e in (await maker.get("/fd/hosting")).json()["entries"])


async def test_a_public_request_is_a_404_for_what_isnt_a_running_app(deck):
    async with _client(deck.maker) as maker, _client() as public:
        assert (await public.get("/vfs-apps/nobody-has-this/")).status_code == 404
        await maker.post("/fd/hosting", json={"name": "apage", "kind": "static", "project": "web"})
        assert (await public.get("/vfs-apps/apage/")).status_code == 404
        # Published and never started, or ended by itself: its card says Stopped, and so does this.
        await _publish(maker, "never")
        assert (await public.get("/vfs-apps/never/")).status_code == 404
        await _publish(maker, "crashy", start_cmd="exit 3")
        await maker.post("/fd/hosting/crashy/start")
        for _ in range(50):
            if not vfs_hosting.app_is_alive("crashy"):
                break
            await asyncio.sleep(0.05)
        crashed = await public.get("/vfs-apps/crashy/")
        assert crashed.status_code == 404 and "crashy" not in crashed.text
        assert (await _listed(maker, "crashy"))["running"] is False


async def test_someone_elses_app_stays_theirs(deck):
    async with _client(deck.maker) as maker, _client(deck.second) as other:
        await _publish(maker, "private")
        for action in ("start", "stop", "restart"):
            assert (await other.post(f"/fd/hosting/private/{action}")).status_code == 403
        assert (await other.delete("/fd/hosting/private")).status_code == 403


# ── Limits ──────────────────────────────────────────────────────────────


async def test_how_many_run_at_once_for_one_person_and_for_the_deck(deck, monkeypatch):
    monkeypatch.setenv("FD_CREW_APPS_PER_USER", "1")
    monkeypatch.setenv("FD_CREW_APPS_MAX_RUNNING", "2")
    async with _client(deck.maker) as maker, _client(deck.second) as second, _client(deck.admin) as admin:
        await _publish(maker, "one")
        await _publish(maker, "two")
        assert (await maker.post("/fd/hosting/one/start")).status_code == 200
        busy = await maker.post("/fd/hosting/two/start")
        assert busy.status_code == 409 and "one" in busy.json()["detail"]
        # Starting the one that's running isn't a second one.
        assert (await maker.post("/fd/hosting/one/start")).json()["message"] == "already running"
        assert (await maker.post("/fd/hosting/one/restart")).status_code == 200
        await maker.post("/fd/hosting/one/stop")
        assert (await maker.post("/fd/hosting/two/start")).status_code == 200

        await _publish(second, "theirs")
        assert (await second.post("/fd/hosting/theirs/start")).status_code == 200
        await _publish(second, "more")
        await second.post("/fd/hosting/theirs/stop")
        await _publish(admin, "admins")
        assert (await admin.post("/fd/hosting/admins/start")).status_code == 200  # an admin's isn't counted
        assert (await second.post("/fd/hosting/more/start")).status_code == 200
        await _publish(maker, "three")
        await maker.post("/fd/hosting/two/stop")
        await _publish(second, "fourth")
        monkeypatch.setenv("FD_CREW_APPS_PER_USER", "5")
        assert (await maker.post("/fd/hosting/three/start")).status_code == 200
        full = await second.post("/fd/hosting/fourth/start")
        assert full.status_code == 409 and "as many apps as it allows" in full.json()["detail"]


async def test_ports_come_from_one_range(deck, monkeypatch):
    base = int(os.environ["VFS_APPS_PORT_BASE"])
    monkeypatch.setenv("VFS_APPS_PORT_SPAN", "1")
    async with _client(deck.maker) as maker:
        await _publish(maker, "inrange")
        with socket.socket() as holder:
            holder.bind(("127.0.0.1", base))
            holder.listen()
            refused = await maker.post("/fd/hosting/inrange/start")
            assert refused.status_code == 400 and "no free port" in refused.json()["detail"]
        assert (await maker.post("/fd/hosting/inrange/start")).status_code == 200
        assert vfs_hosting.load_registry()["inrange"]["port"] == base


async def test_the_watcher_stops_an_app_past_its_memory_or_running_time(deck):
    async with _client(deck.maker) as maker, _client(deck.admin) as admin, _client() as public:
        await _publish(maker, "big")
        await _publish(admin, "admins")
        await maker.post("/fd/hosting/big/start")
        await admin.post("/fd/hosting/admins/start")
        await _seen(public, "big")
        await _seen(public, "admins")
        stopped = vfs_hosting.police(memory_mb=1, hours=0)
        assert [name for name, _ in stopped] == ["big"]  # an admin's app has no such limit
        assert "[Flight Deck] Stopped: it used" in vfs_hosting.read_app_log("big")
        assert vfs_hosting.load_registry()["big"]["stopped"] is True
        assert (await public.get("/vfs-apps/big/")).status_code == 404
        assert vfs_hosting.app_is_alive("admins")

        await maker.post("/fd/hosting/big/start")
        await _seen(public, "big")
        assert vfs_hosting.police(memory_mb=0, hours=1) == []
        registry = vfs_hosting.load_registry()
        registry["big"]["started_at"] = int(time.time()) - 3700
        vfs_hosting.save_registry(registry)
        assert [name for name, _ in vfs_hosting.police(memory_mb=0, hours=1)] == ["big"]
        assert "ran for 1 hour" in vfs_hosting.read_app_log("big")


# ── When the approval changes ───────────────────────────────────────────


async def test_withdrawing_hosted_apps_stops_them_and_they_keep_their_address(deck):
    async with _client(deck.maker) as maker, _client(deck.admin) as admin, _client() as public:
        await _publish(maker, "goes")
        await maker.post("/fd/hosting/goes/start")
        await _seen(public, "goes")
        resp = await admin.put(f"/fd/admin/users/{deck.maker['id']}",
                               json={"crew": {"approved": True, "harnesses": ["opencode"], "apps": False}})
        assert resp.status_code == 200, resp.text
        assert not vfs_hosting.app_is_alive("goes")
        assert (await public.get("/vfs-apps/goes/")).status_code == 404
        assert (await _listed(maker, "goes"))["url"] == "/vfs-apps/goes/"
        assert "approval for hosted apps changed" in vfs_hosting.read_app_log("goes")
        assert (await maker.post("/fd/hosting/goes/start")).status_code == 403


async def test_an_admin_made_a_user_has_their_app_stopped_and_started_again_with_less(deck):
    async with _client(deck.admin) as admin, _client() as public:
        await _publish(admin, "was-admin")
        await admin.post("/fd/hosting/was-admin/start")
        assert (await _seen(public, "was-admin"))["key"] == DECK_SECRET
    await deck.db.update_user(deck.admin["id"], role="user", metadata=json.dumps(_crew(apps=True)))
    assert await hosting_routes.enforce_apps(deck.admin["id"]) == ["was-admin"]
    async with _client({**deck.admin, "role": "user"}) as now_user, _client() as public:
        assert (await now_user.post("/fd/hosting/was-admin/start")).status_code == 200
        assert (await _seen(public, "was-admin"))["key"] == ""


async def test_the_check_at_start_up_stops_what_an_approval_no_longer_covers(deck, monkeypatch):
    async with _client(deck.maker) as maker, _client() as public:
        await _publish(maker, "survivor")
        await maker.post("/fd/hosting/survivor/start")
        await _seen(public, "survivor")
    monkeypatch.setenv("FD_CREW_APPS", "off")  # changed while the deck was down
    await hosting_routes.enforce_app_owners()
    assert not vfs_hosting.app_is_alive("survivor")


async def test_a_deleted_account_has_its_apps_stopped(deck):
    async with _client(deck.maker) as maker, _client() as public:
        await _publish(maker, "orphan")
        await maker.post("/fd/hosting/orphan/start")
        await _seen(public, "orphan")
    await crew_runner.forget_user(deck.maker["id"])
    assert not vfs_hosting.app_is_alive("orphan")


# ── Their agent, through the `hosting` tool's routes ────────────────────


async def test_their_agent_can_publish_restart_stop_and_start_an_app(deck):
    async with _client(deck.maker) as agent, _client() as public:
        made = (await agent.post("/fd/hosting/agent/publish", json={
            "name": "agentapp", "kind": "app", "project": "web", "start_cmd": START})).json()
        assert made["started"] is True and made["url_abs"] == "https://deck.example.test/vfs-apps/agentapp/"
        first = (await _seen(public, "agentapp"))["pid"]
        again = (await agent.post("/fd/hosting/agent/restart", json={"name": "agentapp"})).json()
        assert again["started"] is True
        assert (await _seen(public, "agentapp"))["pid"] != first
        assert (await agent.post("/fd/hosting/agent/stop", json={"name": "agentapp"})).json()["ok"] is True
        assert (await public.get("/vfs-apps/agentapp/")).status_code == 404
        assert (await agent.post("/fd/hosting/agent/start", json={"name": "agentapp"})).json()["started"] is True


async def test_an_agent_whose_owner_isnt_approved_gets_flight_decks_refusal(deck):
    async with _client(deck.coder) as agent:
        refused = await agent.post("/fd/hosting/agent/publish", json={
            "name": "nope", "kind": "app", "project": "web", "start_cmd": START})
        assert refused.status_code == 403 and "Hosted apps" in refused.json()["detail"]
        assert "nope" not in vfs_hosting.load_registry()


async def test_an_agent_over_the_limit_publishes_and_is_told_why_it_didnt_start(deck, monkeypatch):
    monkeypatch.setenv("FD_CREW_APPS_PER_USER", "1")
    async with _client(deck.maker) as agent:
        first = await agent.post("/fd/hosting/agent/publish", json={"name": "a1", "kind": "app", "project": "web", "start_cmd": START})
        assert first.json()["started"] is True
        second = (await agent.post("/fd/hosting/agent/publish", json={
            "name": "a2", "kind": "app", "project": "web", "start_cmd": START})).json()
        assert second["started"] is False and "a1" in second["message"]
        refused = await agent.post("/fd/hosting/agent/start", json={"name": "a2"})
        assert refused.status_code == 409


# ── What the tools say ──────────────────────────────────────────────────


def test_the_crew_tool_says_how_to_run_what_a_task_made_only_when_the_owner_may():
    from captain_claw.tools.crew import CrewTool

    results = {"files_path": "vfs:crew/Game/fix-ab12cd/files/", "count": 3, "whole": False}
    assert not any("kind=app" in line for line in CrewTool._where_lines(results))
    lines = "\n".join(CrewTool._where_lines(results, apps=True))
    assert "hosting action=publish kind=app project=crew subdir=Game/fix-ab12cd/files" in lines
    assert "whole=true" in lines
    assert "whole=true" not in "\n".join(CrewTool._where_lines({**results, "whole": True}, apps=True))


def test_the_hosting_tool_can_restart():
    from captain_claw.tools.hosting import HostingTool

    assert "restart" in HostingTool.parameters["properties"]["action"]["enum"]
    assert "Users → Crew" in HostingTool.description
