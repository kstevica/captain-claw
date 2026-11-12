"""Anonymizing files where they are: a VFS file, a VFS folder (a job), a Library file."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.flight_deck import (
    agent_secret,
    anonymize_targets,
    anonymize_vault,
    auth,
    fd_crypto,
)
from captain_claw.flight_deck.db import FlightDeckDB

ALICE = "user-alice"
BOB = "user-bob"
LETTER = "Name: Ivan Horvat\nOIB: 52601815906\nmail ivan.horvat@acme.hr\n"


@pytest.fixture()
async def deck(monkeypatch, tmp_path):
    import captain_claw.flight_deck.server as srv

    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "data")
    agent_secret.reset_cache_for_tests()
    fd_crypto.reset_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev_db = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email in [(ALICE, "alice@x.co"), (BOB, "bob@x.co")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'user', ?, ?)", (uid, email, "h", uid.title(), now, now))
    await d._db.commit()
    store = anonymize_vault.VaultStore(tmp_path / "anonymize.db")
    anonymize_vault.set_store(store)
    anonymize_targets._JOBS.clear()
    try:
        yield tmp_path
    finally:
        anonymize_targets._JOBS.clear()
        anonymize_vault.set_store(None)
        await store.close()
        agent_secret.reset_cache_for_tests()
        fd_crypto.reset_for_tests()
        auth.set_auth_db(prev_db)
        await d.close()


def _client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(anonymize_targets.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://deck")


def _as(uid: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth.create_access_token(uid, 'user')}"}


def _vfs(tmp: Path, uid: str = ALICE) -> Path:
    return tmp / "data" / "vfs" / uid


def _put(root: Path, rel: str, data: str | bytes) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        p.write_text(data, encoding="utf-8")
    else:
        p.write_bytes(data)
    return p


async def _finish() -> None:
    job = anonymize_targets._JOBS[ALICE]
    if job.task is not None:
        await job.task


class TestVfsFile:
    async def test_copy_beside_the_original(self, deck):
        src = _put(_vfs(deck), "Clients/Ivan Horvat/ugovor.txt", LETTER)
        async with _client() as c:
            r = await c.post("/fd/anonymize/vfs/file", headers=_as(ALICE),
                             json={"project": "Clients", "path": "Ivan Horvat/ugovor.txt"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["beside"] is True and data["vault"]["name"] == "Clients"  # the folder names the vault
        assert data["written"] == {"project": "Clients", "owner": "", "path": "Ivan Horvat/ugovor.anonymized.txt"}
        out = (_vfs(deck) / "Clients" / data["written"]["path"]).read_text()
        assert out == "Name: ⟦PERSON_1⟧\nOIB: ⟦OIB_1⟧\nmail ⟦EMAIL_1⟧\n"
        assert src.read_text() == LETTER  # the original is untouched
        # Again: a second copy, never over the first.
        async with _client() as c:
            again = (await c.post("/fd/anonymize/vfs/file", headers=_as(ALICE),
                                  json={"project": "Clients", "path": "Ivan Horvat/ugovor.txt"})).json()
        assert again["written"]["path"] == "Ivan Horvat/ugovor.anonymized (2).txt"

    async def test_from_a_read_only_link_into_own_anonymized_folder(self, deck):
        external = deck / "external"
        _put(external, "Ivan Horvat/cv.txt", LETTER)
        root = _vfs(deck)
        root.mkdir(parents=True)
        (root / ".vfs-links.json").write_text(json.dumps({"Archive": {"path": str(external), "mode": "ro"}}))
        async with _client() as c:
            r = await c.post("/fd/anonymize/vfs/file", headers=_as(ALICE),
                             json={"project": "Archive", "path": "Ivan Horvat/cv.txt"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["beside"] is False and data["written"]["project"] == "Anonymized"
        # The folder named after a person is tokenized in the copy's path.
        assert data["written"]["path"] == "Archive/PERSON_1/cv.anonymized.txt"
        assert "Ivan" not in (root / "Anonymized" / data["written"]["path"]).read_text()
        assert not (external / "Ivan Horvat" / "cv.anonymized.txt").exists()

    async def test_someone_elses_folder_is_not_found(self, deck):
        _put(_vfs(deck), "Clients/a.txt", LETTER)
        async with _client() as c:
            r = await c.post("/fd/anonymize/vfs/file", headers=_as(BOB),
                             json={"project": "Clients", "path": "a.txt", "owner": ALICE})
        assert r.status_code == 404


class TestVfsFolder:
    async def test_whole_folder_into_a_new_folder_beside_it(self, deck):
        root = _vfs(deck)
        _put(root, "Clients/Ivan Horvat/ugovor.txt", LETTER)
        _put(root, "Clients/notes.md", "Sastanak s Ivanom Horvatom u utorak.")
        _put(root, "Clients/photo.png", b"\x89PNG\r\n\x1a\n")
        _put(root, "Clients/broken.txt", b"\x00\x01binary")
        _put(root, "Clients/old.anonymized.txt", "⟦PERSON_1⟧ again")  # an earlier copy
        async with _client() as c:
            r = await c.post("/fd/anonymize/vfs/folder", headers=_as(ALICE), json={"project": "Clients"})
            assert r.status_code == 200, r.text
            assert r.json()["phase"] == "anonymizing" and r.json()["dest"] == {}  # named once the files ran
            await _finish()
            job = (await c.get("/fd/anonymize/vfs/job", headers=_as(ALICE))).json()
        assert job["running"] is False and job["total"] == 3 and job["done"] == 2
        assert sorted(s["path"] for s in job["skipped"]) == ["old.anonymized.txt", "photo.png"]  # never copied as is
        assert [f["path"] for f in job["failed"]] == ["broken.txt"]
        assert job["dest"] == {"project": "Clients_anonymized", "owner": "", "path": ""} and job["beside"]
        dest = root / "Clients_anonymized"
        names = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
        assert names == ["PERSON_1/ugovor.anonymized.txt", "notes.anonymized.md"]
        assert (dest / "notes.anonymized.md").read_text() == "Sastanak s ⟦PERSON_1⟧ u utorak."
        assert job["entities"]["PERSON"] >= 2
        # A second run goes to a new folder.
        async with _client() as c:
            await c.post("/fd/anonymize/vfs/folder", headers=_as(ALICE), json={"project": "Clients"})
            await _finish()
            again = (await c.get("/fd/anonymize/vfs/job", headers=_as(ALICE))).json()
        assert again["dest"]["project"] == "Clients_anonymized-2"

    async def test_a_sub_folder_beside_itself(self, deck):
        root = _vfs(deck)
        _put(root, "Clients/Ivan Horvat/ugovor.txt", LETTER)
        async with _client() as c:
            await c.post("/fd/anonymize/vfs/folder", headers=_as(ALICE),
                         json={"project": "Clients", "path": "Ivan Horvat", "mode": "one_way"})
            await _finish()
            job = (await c.get("/fd/anonymize/vfs/job", headers=_as(ALICE))).json()
        assert job["dest"] == {"project": "Clients", "owner": "", "path": "PERSON_1_anonymized"}
        assert job["vault"] == ""  # one way keeps nothing
        assert (root / "Clients" / "PERSON_1_anonymized" / "ugovor.anonymized.txt").is_file()

    async def test_one_job_at_a_time_and_stop(self, deck):
        root = _vfs(deck)
        for i in range(5):
            _put(root, f"Big/f{i}.txt", LETTER)
        async with _client() as c:
            await c.post("/fd/anonymize/vfs/folder", headers=_as(ALICE), json={"project": "Big"})
            busy = await c.post("/fd/anonymize/vfs/folder", headers=_as(ALICE), json={"project": "Big"})
            stopped = (await c.post("/fd/anonymize/vfs/job/cancel", headers=_as(ALICE))).json()
            await _finish()
            job = (await c.get("/fd/anonymize/vfs/job", headers=_as(ALICE))).json()
        assert busy.status_code == 409 and stopped["cancelled"] is True
        assert job["done"] < 5 and job["message"] == "Stopped."
        # What it finished before the stop is placed.
        placed = list((_vfs(deck) / job["dest"]["project"]).rglob("*.txt")) if job["done"] else []
        assert len(placed) == job["done"]


class TestLibrary:
    async def test_copy_beside_it_in_the_agent_workspace(self, deck, monkeypatch):
        from captain_claw.flight_deck import file_library as fl

        workspace = deck / "data" / "alice-bot" / "data" / "workspace"
        _put(workspace, "notes/Ivan_Horvat_CV.txt", LETTER)
        store = fl.LibraryStore(deck / "file_library.db")
        monkeypatch.setattr(fl, "_STORE", store)
        monkeypatch.setattr(fl, "caller_agents", lambda user: {"alice-bot": {"name": "Alice bot", "running": False}}
                            if user.get("id") == ALICE else {})
        store.sync_agent("alice-bot", fl.walk_workspace("alice-bot"))
        file_id = store.file_id("alice-bot", "notes/Ivan_Horvat_CV.txt")
        async with _client() as c:
            nope = await c.post(f"/fd/anonymize/library/{file_id}", headers=_as(BOB), json={})
            r = await c.post(f"/fd/anonymize/library/{file_id}", headers=_as(ALICE), json={})
        assert nope.status_code == 404
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["vault"]["name"] == "Alice bot"
        assert data["written"]["rel_path"] == "notes/PERSON_1_CV.anonymized.txt"
        assert "Ivan" not in (workspace / data["written"]["rel_path"]).read_text()
        # Listed in the Library right away.
        assert store.get(store.file_id("alice-bot", "notes/PERSON_1_CV.anonymized.txt")) is not None


class TestAgentTool:
    @pytest.fixture()
    def tool_env(self, deck, monkeypatch):
        """Bob's agent; its VFS is the deck's (the same root on disk)."""
        import captain_claw.flight_deck.server as srv
        from captain_claw import fd_client
        from captain_claw.tools import anonymize as tool_mod

        def find(token):
            return (True, BOB, "bob-bot") if token == "tok-bob" else (False, "", "")

        monkeypatch.setattr(srv, "_find_agent_by_auth", find)
        monkeypatch.setattr(srv, "_resolve_agent_identity_by_auth", lambda t: find(t)[:2])
        monkeypatch.setattr(srv, "_load_process_registry", lambda: {})
        monkeypatch.setenv("CLAW_VFS_ROOT", str(deck / "data" / "vfs"))
        monkeypatch.setenv("CLAW_VFS_USER", BOB)
        app = FastAPI()
        app.include_router(anonymize_targets.router)
        real = httpx.AsyncClient

        def factory(*a, **kw):
            kw["transport"] = httpx.ASGITransport(app=app, client=("127.0.0.1", 50123))
            kw.setdefault("base_url", "http://fd.test")
            return real(*a, **kw)

        monkeypatch.setattr(httpx, "AsyncClient", factory)
        monkeypatch.setattr(fd_client, "flight_deck_base", lambda: "http://fd.test")
        monkeypatch.setattr(fd_client, "flight_deck_headers", lambda: {"X-Agent-Auth": "tok-bob"})
        monkeypatch.setattr(tool_mod, "FOLDER_POLL_SECONDS", 0.05)
        work = deck / "workspace"
        (work / "saved").mkdir(parents=True)
        return {"_runtime_base_path": str(work), "_saved_base_path": str(work / "saved"), "_session_id": "s-1"}

    async def test_a_vfs_folder_through_the_tool(self, deck, tool_env):
        from captain_claw.tools.anonymize import AnonymizeTool

        root = _vfs(deck, BOB)
        _put(root, "Clients/Ivan Horvat/ugovor.txt", LETTER)
        _put(root, "Clients/Ivan Horvat/scan.png", b"\x89PNG\r\n\x1a\n")
        res = await AnonymizeTool().execute(action="anonymize", path="vfs:Clients/Ivan Horvat", **tool_env)
        assert res.success, res.error
        assert "Anonymized 1 of 1 files in vfs:Clients/Ivan Horvat." in res.content
        assert "Copies: vfs:Clients/PERSON_1_anonymized/" in res.content and "Left out 1" in res.content
        assert "Ivan" not in res.content.replace("vfs:Clients/Ivan Horvat", "")  # nothing private comes back
        out = root / "Clients" / "PERSON_1_anonymized" / "ugovor.anonymized.txt"
        assert out.read_text().startswith("Name: ⟦PERSON_1⟧")

