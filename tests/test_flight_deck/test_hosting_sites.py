"""Stopping and starting a hosted site, a page or an app alike.

A static site used to have no such thing: `stop` said "stopped" and went on serving.
Now a stopped page answers 404 to the public, keeps its address and its place in the
owner's list, and comes back on `start`. Running an app stays an admin's; a page is
anyone's own.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import hosting_routes, vfs_hosting
from captain_claw.flight_deck import server as fd_server
from captain_claw.flight_deck.auth import get_current_user
from captain_claw.vfs import user_root_of

OWNER = {"id": "u-owner", "role": "user"}
OTHER = {"id": "u-other", "role": "user"}
ADMIN = {"id": "u-admin", "role": "admin"}


@pytest.fixture
def deck(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fd_server, "DATA_DIR", tmp_path / "fd-data")
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path / "fd-data"))
    monkeypatch.delenv("CLAW_VFS_ROOT", raising=False)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    folder = user_root_of(OWNER["id"]) / "site"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<h1>hello</h1>")


def _client(user: dict | None = None) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(hosting_routes.router)
    if user is not None:
        app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 40001)), base_url="http://deck.test")


async def _publish(client: httpx.AsyncClient, **over) -> dict:
    resp = await client.post("/fd/hosting", json={"name": "demo", "kind": "static", "project": "site", **over})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _listed(client: httpx.AsyncClient) -> dict:
    return (await client.get("/fd/hosting")).json()["entries"][0]


async def test_a_stopped_page_is_offline_and_comes_back_at_the_same_address(deck):
    async with _client(OWNER) as owner, _client() as public:
        made = await _publish(owner)
        assert made["running"] is True and made["missing"] is False
        assert (await public.get("/vfs/demo/")).status_code == 200

        assert (await owner.post("/fd/hosting/demo/stop")).json()["ok"] is True
        assert (await public.get("/vfs/demo/")).status_code == 404
        assert (await public.get("/vfs/demo/index.html")).status_code == 404
        entry = await _listed(owner)
        assert entry["running"] is False and entry["url"] == "/vfs/demo/"

        assert (await owner.post("/fd/hosting/demo/start")).status_code == 200
        page = await public.get("/vfs/demo/")
        assert page.status_code == 200 and "hello" in page.text
        assert (await _listed(owner))["running"] is True


async def test_the_list_counts_visits_and_says_when_the_folder_is_gone(deck):
    async with _client(OWNER) as owner, _client() as public:
        await _publish(owner)
        before = (await _listed(owner))["visit_count"]
        await public.get("/vfs/demo/")
        assert (await _listed(owner))["visit_count"] == before + 1
        (user_root_of(OWNER["id"]) / "site" / "index.html").unlink()
        (user_root_of(OWNER["id"]) / "site").rmdir()
        assert (await _listed(owner))["missing"] is True


async def test_someone_elses_site_stays_theirs(deck):
    async with _client(OWNER) as owner, _client(OTHER) as other:
        await _publish(owner)
        for action in ("start", "stop"):
            assert (await other.post(f"/fd/hosting/demo/{action}")).status_code == 403
        assert (await other.delete("/fd/hosting/demo")).status_code == 403
        assert (await other.get("/fd/hosting")).json()["entries"] == []
        assert (await _listed(owner))["running"] is True


async def test_starting_an_app_stays_an_admins_and_a_page_is_anyones_own(deck, monkeypatch):
    started: list[str] = []
    monkeypatch.setattr(vfs_hosting, "start_app", lambda name: (started.append(name), (True, "started"))[1])
    registry = {
        "page": {"kind": "static", "owner": OWNER["id"], "project": "site", "stopped": True},
        "tool": {"kind": "app", "owner": OWNER["id"], "project": "site", "start_cmd": "x", "stopped": True},
    }
    vfs_hosting.save_registry(registry)
    async with _client(OWNER) as owner, _client(ADMIN) as admin:
        assert (await owner.post("/fd/hosting/page/start")).status_code == 200
        refused = await owner.post("/fd/hosting/tool/start")
        assert refused.status_code == 403 and "reserved for admins" in refused.json()["detail"]
        assert started == ["page"]
        assert (await admin.post("/fd/hosting/tool/start")).status_code == 200
    assert started == ["page", "tool"]


async def test_publishing_over_a_stopped_page_brings_it_back(deck):
    async with _client(OWNER) as owner, _client() as public:
        await _publish(owner)
        await owner.post("/fd/hosting/demo/stop")
        edited = await owner.put("/fd/hosting/demo", json={"kind": "static", "project": "site"})
        assert edited.status_code == 200 and edited.json()["running"] is True
        assert (await public.get("/vfs/demo/")).status_code == 200


async def test_no_site_can_be_called_agent(deck):
    """/fd/hosting/agent/… are the routes agents call."""
    async with _client(OWNER) as owner:
        resp = await owner.post("/fd/hosting", json={"name": "agent", "kind": "static", "project": "site"})
    assert resp.status_code == 400


def test_starting_a_name_nobody_has_says_so(deck):
    assert vfs_hosting.start_app("nothing") == (False, "not found")
