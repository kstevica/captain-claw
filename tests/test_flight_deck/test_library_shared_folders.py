"""The File and Data libraries hold the person's shared folders too.

A shared folder (vfs:<folder>/) is a source beside the agents: a scan walks
its files like an agent's workspace and its datastore (.datastore/store.db)
like an agent's, so the Library and Data dialogs list, open, search, filter
and edit them the same way. The UI knows a folder as ``vfs:<folder>``; two
people may each have a folder of that name, and neither sees the other's.
Linked folders, Drive mounts and Flight Deck's own dot folders stay out, and
a folder that is gone is forgotten by the next scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from captain_claw.datastore import DatastoreManager
from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import data_library as dl
from captain_claw.flight_deck import file_library as fl
from captain_claw.flight_deck import rate_limiter, server
from captain_claw.flight_deck.auth import create_access_token, set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB

BOB = "user-bob"
CAROL = "user-carol"


@pytest.fixture
async def deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Auth-enabled deck: Bob owns a stopped agent with one file, a shared
    folder "reports" (a file, a hidden one, a datastore with a table), a
    linked folder and a Drive mount tree; Carol has a "reports" of her own."""
    prev = fd_auth._db
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    set_auth_db(db)
    for uid in (BOB, CAROL):
        await db._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, 'h', ?, 'user', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (uid, f"{uid}@x.co", uid))
    await db._db.commit()

    data = tmp_path / "fd-data"
    data.mkdir()
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "PROCESS_REGISTRY_FILE", data / ".processes.json")
    monkeypatch.setattr(server, "_processes", {})
    monkeypatch.setattr(server, "AUTH_ENABLED", True)
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    monkeypatch.setenv("FD_ALLOWED_HOSTS", "fd.test")
    monkeypatch.delenv("FD_LOCKDOWN", raising=False)
    monkeypatch.setattr(rate_limiter, "_api_limiter", rate_limiter._SlidingWindow())
    monkeypatch.setattr(fl, "_STORE", fl.LibraryStore(tmp_path / "file_library.db"))
    monkeypatch.setattr(dl, "_STORE", dl.DataStore(tmp_path / "data_library.db"))

    # No model: a scan lists what is there and stops before summaries.
    async def no_tier(*_a, **_kw):
        return None

    async def no_media(owner, tier):
        return fl.MediaReach(owner=owner, tier=tier)

    monkeypatch.setattr(fl, "_resolve_tier", no_tier)
    monkeypatch.setattr(dl, "_resolve_tier", no_tier)
    monkeypatch.setattr(fl, "media_reach", no_media)

    server._save_process_registry({
        "bob-agent": {"slug": "bob-agent", "name": "Bob's agent", "web_port": 24101,
                      "web_auth": "bob-tok", "owner": BOB, "pid": None},
    })
    notes = data / "bob-agent" / "data" / "workspace" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("# Notes\n")

    bob = data / "vfs" / BOB
    (bob / "reports" / "q3").mkdir(parents=True)
    (bob / "reports" / "q3" / "summary.md").write_text("# Q3\n\nRevenue grew.\n")
    (bob / "reports" / ".hidden.md").write_text("not listed\n")
    mgr = DatastoreManager(db_path=bob / "reports" / ".datastore" / "store.db")
    await mgr.create_table("leads", [{"name": "org", "type": "text"}, {"name": "score", "type": "integer"}])
    await mgr.insert_rows("leads", [{"org": "Acme", "score": 7}, {"org": "Globex", "score": 3}])
    await mgr.close()
    # A folder linked in from this computer, and a Drive mount's tree.
    outside = tmp_path / "repo"
    outside.mkdir()
    (outside / "main.py").write_text("print('hi')\n")
    (bob / ".vfs-links.json").write_text(json.dumps({"repo": {"path": str(outside), "mode": "ro"}}))
    (bob / ".drive" / "gd").mkdir(parents=True)
    (bob / ".drive" / "gd" / "doc.md").write_text("drive\n")

    carol = data / "vfs" / CAROL / "reports"
    carol.mkdir(parents=True)
    (carol / "carol.md").write_text("# Carol's\n")
    # Each added their "reports" to the File library (a folder is in it only once added).
    fl.get_store().add_pick(fl._vfs_owner(BOB), "reports")
    fl.get_store().add_pick(fl._vfs_owner(CAROL), "reports")
    try:
        yield data
    finally:
        await db.close()
        fd_auth._db = prev


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, client=("203.0.113.9", 40001)), base_url="http://fd.test")


def _auth(uid: str) -> dict:
    return {"Authorization": f"Bearer {create_access_token(uid)}"}


async def _scan(uid: str) -> None:
    user = {"id": uid}
    await fl.run_scan(user, fl.ScanJob())
    await dl.run_scan(user, fl.ScanJob())


async def test_a_folders_files_are_in_the_library_beside_the_agents(deck: Path):
    await _scan(BOB)
    async with _client() as c:
        r = await c.get("/fd/file-library/files", headers=_auth(BOB))
    assert r.status_code == 200, r.text
    body = r.json()
    by_name = {i["filename"]: i for i in body["items"]}
    # The agent's file, the folder's file — not the hidden one, the linked
    # repository or the Drive tree.
    assert set(by_name) == {"notes.md", "summary.md"}
    item = by_name["summary.md"]
    assert item["source"] == "folder" and item["folder"] == "reports"
    assert item["agent_slug"] == "vfs:reports" and item["agent_name"] == "reports"
    assert item["agent_running"] is False
    assert item["rel_path"] == "q3/summary.md"
    assert item["logical"] == "vfs:reports/q3/summary.md"
    assert Path(item["physical"]).read_text().startswith("# Q3")
    assert by_name["notes.md"]["source"] == "agent" and "folder" not in by_name["notes.md"]
    # Agents and folders are offered as filters apart.
    assert [f["id"] for f in body["facets"]["agents"]] == ["bob-agent"]
    assert body["facets"]["folders"] == [{"id": "vfs:reports", "label": "reports", "count": 1}]


async def test_filtering_to_a_folder_and_opening_its_file(deck: Path):
    await _scan(BOB)
    async with _client() as c:
        r = await c.get("/fd/file-library/files", params={"agent": "vfs:reports"}, headers=_auth(BOB))
        [item] = r.json()["items"]
        view = await c.get(f"/fd/file-library/files/{item['id']}/view", headers=_auth(BOB))
        status = (await c.get("/fd/file-library/status", headers=_auth(BOB))).json()
    assert item["filename"] == "summary.md"
    assert view.status_code == 200 and view.text.startswith("# Q3")
    assert status["agents"] == 1 and status["folders"] == 1
    assert status["counts"]["total"] == 2


async def test_a_folder_of_the_same_name_is_each_persons_own(deck: Path):
    await _scan(BOB)
    await _scan(CAROL)
    async with _client() as c:
        bob = (await c.get("/fd/file-library/files", params={"agent": "vfs:reports"}, headers=_auth(BOB))).json()
        carol = (await c.get("/fd/file-library/files", params={"agent": "vfs:reports"}, headers=_auth(CAROL))).json()
        [bobs] = bob["items"]
        nope = await c.get(f"/fd/file-library/files/{bobs['id']}/view", headers=_auth(CAROL))
    assert [i["filename"] for i in carol["items"]] == ["carol.md"]
    assert nope.status_code == 404


async def test_a_folder_that_is_gone_is_forgotten(deck: Path):
    import shutil

    await _scan(BOB)
    await _scan(CAROL)
    shutil.rmtree(deck / "vfs" / BOB / "reports")
    await _scan(BOB)
    async with _client() as c:
        bob = (await c.get("/fd/file-library/files", headers=_auth(BOB))).json()
        carol = (await c.get("/fd/file-library/files", headers=_auth(CAROL))).json()
        tables = (await c.get("/fd/data-library/tables", headers=_auth(BOB))).json()
    assert [i["filename"] for i in bob["items"]] == ["notes.md"]
    assert tables["items"] == []
    # Carol's folder of the same name is untouched.
    assert [i["filename"] for i in carol["items"]] == ["carol.md"]


async def test_a_folders_datastore_is_in_the_data_library(deck: Path):
    await _scan(BOB)
    async with _client() as c:
        r = await c.get("/fd/data-library/tables", headers=_auth(BOB))
        body = r.json()
        [t] = body["items"]
        rows = await c.get(f"/fd/data-library/tables/{t['id']}/rows", params={"sort": "score"}, headers=_auth(BOB))
        edit = await c.patch(f"/fd/data-library/tables/{t['id']}/rows/1", headers=_auth(BOB),
                             json={"values": {"score": 9}, "expect": {"score": 7}})
        csv = await c.get(f"/fd/data-library/tables/{t['id']}/export", params={"format": "csv"}, headers=_auth(BOB))
        carol = await c.get(f"/fd/data-library/tables/{t['id']}/rows", headers=_auth(CAROL))
        mine = (await c.get("/fd/data-library/tables", params={"agent": "vfs:reports"}, headers=_auth(BOB))).json()
    assert r.status_code == 200, r.text
    assert t["source"] == "folder" and t["agent_slug"] == "vfs:reports" and t["folder"] == "reports"
    assert t["table_name"] == "leads" and t["row_count"] == 2
    assert body["facets"]["folders"] == [{"id": "vfs:reports", "label": "reports", "count": 1}]
    assert [row["org"] for row in rows.json()["rows"]] == ["Globex", "Acme"]
    assert edit.status_code == 200, edit.text
    assert edit.json()["row"]["score"] == 9
    assert "Acme,9" in csv.text
    assert carol.status_code == 404
    assert [x["id"] for x in mine["items"]] == [t["id"]]


async def test_forgetting_removed_agents_keeps_the_folders(deck: Path):
    await _scan(BOB)
    store = fl.get_store()
    store.drop_agents_except(set())
    dl.get_store().drop_agents_except(set())
    async with _client() as c:
        files = (await c.get("/fd/file-library/files", headers=_auth(BOB))).json()
        tables = (await c.get("/fd/data-library/tables", headers=_auth(BOB))).json()
    assert [i["filename"] for i in files["items"]] == ["summary.md"]
    assert [t["table_name"] for t in tables["items"]] == ["leads"]


def test_a_folder_key_never_reaches_outside(deck: Path):
    assert fl.folder_of("vfs:user-bob/reports") == ("user-bob", "reports")
    for bad in ("vfs:user-bob/../x", "vfs:user-bob/a/b", "vfs:/reports", "vfs:user-bob/", "bob-agent"):
        assert fl.folder_of(bad) is None
