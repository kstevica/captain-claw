"""Data rooms (docs/data-room-plan.md P1): intake (ZIP checks, names, folders),
the run (anonymize, folder names, final sweep, text, leak check), release,
search, and that a room is its owner's alone. Code lane only (no trusted
model in tests). Made-up data."""

from __future__ import annotations

import asyncio
import io
import zipfile

import httpx
import pytest
from fastapi import FastAPI

from captain_claw.anonymize import engine
from captain_claw.flight_deck import (
    agent_secret,
    anonymize_vault,
    auth,
    fd_crypto,
    rooms_intake,
    rooms_job,
    rooms_routes,
    rooms_store,
)
from captain_claw.flight_deck.db import FlightDeckDB

ALICE, BOB = "user-alice", "user-bob"
EMAIL = "ana.kovac@primjer-firma.hr"
IBAN = "HR1210010051863000160"


@pytest.fixture()
async def deck(monkeypatch, tmp_path):
    monkeypatch.setenv("FD_AUTH_ENABLED", "true")
    for var in ("FD_LOCKDOWN", "FD_AGENT_SHARED_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CAPTAIN_CLAW_FD_HOME", str(tmp_path / "fd-home"))
    agent_secret.reset_cache_for_tests()
    fd_crypto.reset_for_tests()
    d = FlightDeckDB(str(tmp_path / "fd.db"))
    await d.init()
    prev = auth._db
    auth.set_auth_db(d)
    now = "2026-01-01T00:00:00Z"
    for uid, email, role in [(ALICE, "alice@x.co", "admin"), (BOB, "bob@x.co", "user")]:
        await d._db.execute(
            "INSERT INTO users (id, email, password_hash, display_name, role, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)", (uid, email, "h", uid.title(), role, now, now))
    await d._db.commit()
    vaults = anonymize_vault.VaultStore(tmp_path / "anonymize.db")
    anonymize_vault.set_store(vaults)
    rooms = rooms_store.RoomStore(tmp_path / "data" / "rooms.db")
    rooms_store.set_store(rooms)
    monkeypatch.setattr(rooms_job, "_RUN_SLOT", asyncio.Semaphore(1))
    try:
        yield tmp_path
    finally:
        for task in list(rooms_job._TASKS.values()):
            task.cancel()
        rooms_store.set_store(None)
        anonymize_vault.set_store(None)
        await rooms.close()
        await vaults.close()
        fd_crypto.reset_for_tests()
        auth.set_auth_db(prev)
        await d.close()


def _client() -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(rooms_routes.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 50123)),
                             base_url="http://deck.test")


def _as(uid: str, role: str = "user") -> dict[str, str]:
    return {"Authorization": f"Bearer {auth.create_access_token(uid, role)}"}


ALICE_H = _as(ALICE, "admin")


def _zip(files: dict[str, bytes | str], *, dos_names: dict[str, bytes] | None = None) -> bytes:
    """A ZIP; *dos_names* are written the way old Windows tools did — a Croatian
    code page's bytes, no UTF-8 flag (Python's zipfile would set the flag, so
    an ASCII stand-in of the same length is written and its bytes swapped)."""
    buf = io.BytesIO()
    swaps: list[tuple[bytes, bytes]] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data.encode() if isinstance(data, str) else data)
        for i, (raw, data) in enumerate((dos_names or {}).items()):
            stand_in = f"DOS{i}".ljust(len(raw), "_").encode()
            swaps.append((stand_in, raw))
            z.writestr(stand_in.decode(), data.encode() if isinstance(data, str) else data)
    blob = buf.getvalue()
    for stand_in, raw in swaps:
        blob = blob.replace(stand_in, raw)
    return blob


def _inner_zip() -> bytes:
    return _zip({"deep.txt": f"Kontakt: {EMAIL}"})


async def _new_room(c: httpx.AsyncClient, codename: str = "Project Falcon", headers=None) -> dict:
    r = await c.post("/fd/rooms", headers=headers or ALICE_H, json={"codename": codename, "name": "Sokol prodaja"})
    assert r.status_code == 200, r.text
    return r.json()


async def _wait(c: httpx.AsyncClient, room_id: str, headers=None) -> dict:
    for _ in range(400):
        r = await c.get(f"/fd/rooms/{room_id}", headers=headers or ALICE_H)
        room = r.json()
        if not room["job"]["running"] and room["state"] != "running":
            return room
        await asyncio.sleep(0.05)
    raise AssertionError("the room never finished")


async def _files(c: httpx.AsyncClient, room_id: str) -> dict[str, dict]:
    r = await c.get(f"/fd/rooms/{room_id}/files", headers=ALICE_H)
    return {f["path"]: f for f in r.json()["files"]}


# ── names and checks ────────────────────────────────────────────────


def test_dos_names_are_decoded_by_the_code_page_with_real_letters():
    raw = "Ugovor Kovač Šimić.txt".encode("cp852")
    info = zipfile.ZipInfo(raw.decode("cp437"))
    assert rooms_intake.decode_name(info) == "Ugovor Kovač Šimić.txt"
    utf = zipfile.ZipInfo("Ugovor Kovač.txt")
    utf.flag_bits |= 0x800
    assert rooms_intake.decode_name(utf) == "Ugovor Kovač.txt"


@pytest.mark.parametrize("name,ok", [
    ("a/b.txt", "a/b.txt"), ("../evil.txt", None), ("/etc/passwd", None), ("C:/x.txt", None),
    ("__MACOSX/a/._b.txt", None), ("a/.DS_Store", None), ("a/~$lock.docx", None), ("./a/./b.txt", "a/b.txt"),
])
def test_safe_paths(name, ok):
    assert rooms_intake.safe_rel(name) == ok


def test_leak_check_finds_real_values_whatever_the_case_and_accents():
    state = engine.VaultState([engine.Entity("⟦PERSON_1⟧", "PERSON", "ana kovac", "Ana Kovač"),
                               engine.Entity("⟦ORG_2⟧", "ORG", "sokol", "Sokol Softver d.o.o."),
                               engine.Entity("⟦PERSON_3⟧", "PERSON", "ivo", "Ivo")])
    needle_list = rooms_job.needles(state)
    assert ("⟦PERSON_3⟧", "ivo") not in needle_list  # too short to look for
    assert rooms_job.leaks("Sastanak: ANA KOVAC i ⟦ORG_2⟧", needle_list) == ["⟦PERSON_1⟧"]
    assert rooms_job.leaks("sokol softver d.o.o. je rekao", needle_list) == ["⟦ORG_2⟧"]
    assert rooms_job.leaks("Anakovačević", needle_list) == []  # a whole word only


def test_seed_terms():
    assert rooms_job.seed_terms({"orgs": ["Sokol Softver d.o.o."], "people": ["Ana Kovač", " "],
                                 "ids": ["12345678901"]}) == [
        {"text": "Sokol Softver d.o.o.", "kind": "ORG"}, {"text": "Ana Kovač", "kind": "PERSON"},
        {"text": "12345678901", "kind": "ID_NUMBER"}]


# ── the whole way through ───────────────────────────────────────────


async def test_zip_in_run_release_search(deck):
    letter = f"Ime i prezime: Ana Kovač\nE-mail: {EMAIL}\nIBAN: {IBAN}\n"
    archive = _zip({
        "Klijenti/Ana Kovač/ugovor.txt": letter,
        "Klijenti/a-sazetak.md": "Sastanak s Ana Kovač prošli tjedan.",  # before the file that names her
        "inner.zip": _inner_zip(),
        "podaci.bin": b"\x00\x01",
        "__MACOSX/._ugovor.txt": b"junk",
    }, dos_names={"Klijenti/Bilješka Šimić.txt".encode("cp852"): f"Kontakt: {EMAIL}"})
    async with _client() as c:
        room = await _new_room(c)
        r = await c.put(f"/fd/rooms/{room['id']}/zip", headers={**ALICE_H, "Content-Type": "application/zip"},
                        content=archive)
        assert r.status_code == 200, r.text
        report = r.json()["report"]
        assert report["added"] == 4 and report["nested_zips"] == 1
        assert [u["path"] for u in report["unread"]] == ["podaci.bin"]
        assert report["skipped_count"] == 1
        files = await _files(c, room["id"])
        assert "Klijenti/Bilješka Šimić.txt" in files and "inner/deep.txt" in files

        # Originals are encrypted at rest; original paths are sealed in rooms.db.
        raw_dir = deck / "data" / "rooms" / room["id"] / "raw"
        assert all(EMAIL.encode() not in p.read_bytes() for p in raw_dir.iterdir())
        assert b"Ana Kova" not in (deck / "data" / "rooms.db").read_bytes()

        r = await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        assert r.status_code == 200, r.text
        done = await _wait(c, room["id"])
        assert done["state"] == "review" and done["job"]["phase"] == "done"
        files = await _files(c, room["id"])
        assert {f["state"] for p, f in files.items() if p != "podaci.bin"} == {"done"}

        for f in files.values():
            if f["state"] != "done":
                continue
            assert "Ana" not in f["clean_path"] and "Kova" not in f["clean_path"]
            t = (await c.get(f"/fd/rooms/{room['id']}/files/{f['id']}/text", headers=ALICE_H)).json()
            assert EMAIL not in t["clean"] and IBAN not in t["clean"] and "Ana Kovač" not in t["clean"], f["path"]
        # The summary named her before the contract did: the final sweep caught it.
        summary = files["Klijenti/a-sazetak.md"]
        t = (await c.get(f"/fd/rooms/{room['id']}/files/{summary['id']}/text?original=1", headers=ALICE_H)).json()
        assert "⟦PERSON_" in t["clean"] and "Ana Kovač" in t["original"]
        assert done["job"].get("sweep_changed", 0) >= 1

        r = await c.post(f"/fd/rooms/{room['id']}/release", headers=ALICE_H)
        assert r.status_code == 200 and r.json()["released_now"] == 4
        assert r.json()["room"]["state"] == "released"
        hits = (await c.get(f"/fd/rooms/{room['id']}/search", params={"q": "IBAN"}, headers=ALICE_H)).json()
        assert hits["results"] and all(IBAN not in h["snippet"] for h in hits["results"])
        assert (await c.get(f"/fd/rooms/{room['id']}/search", params={"q": "Kovač"}, headers=ALICE_H)).json()[
            "results"] == []

        clean = (await c.get(f"/fd/rooms/{room['id']}/files/{files['inner/deep.txt']['id']}/clean",
                             headers=ALICE_H))
        assert clean.status_code == 200 and EMAIL.encode() not in clean.content


async def test_a_blocked_file_is_never_released(deck):
    async with _client() as c:
        room = await _new_room(c)
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({"a.txt": f"Mail: {EMAIL}",
                                                                                   "b.txt": "Ništa osobno."}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await _wait(c, room["id"])
        files = await _files(c, room["id"])
        await rooms_store.store().set_file(room["id"], files["a.txt"]["id"], state="blocked", error="test")
        r = await c.post(f"/fd/rooms/{room['id']}/release", headers=ALICE_H)
        assert r.json()["released_now"] == 1
        files = await _files(c, room["id"])
        assert files["a.txt"]["released"] is False and files["b.txt"]["released"] is True


async def test_seed_terms_reach_the_anonymizer(deck):
    async with _client() as c:
        room = await _new_room(c)
        r = await c.patch(f"/fd/rooms/{room['id']}", headers=ALICE_H,
                          json={"seed": {"orgs": ["Sokol Softver"], "people": ["Petra Babić"]}})
        assert r.status_code == 200 and r.json()["seed"]["orgs"] == ["Sokol Softver"]
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H,
                    content=_zip({"memo.txt": "Sokol Softver i Petra Babić dogovorili su cijenu."}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await _wait(c, room["id"])
        f = (await _files(c, room["id"]))["memo.txt"]
        text = (await c.get(f"/fd/rooms/{room['id']}/files/{f['id']}/text", headers=ALICE_H)).json()["clean"]
        assert "Sokol" not in text and "Babić" not in text and "⟦ORG_" in text and "⟦PERSON_" in text


async def test_a_role_that_names_a_real_value_is_refused(deck):
    async with _client() as c:
        room = await _new_room(c)
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({"a.txt": f"Mail: {EMAIL}"}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await _wait(c, room["id"])
        r = await c.patch(f"/fd/rooms/{room['id']}", headers=ALICE_H,
                          json={"roles": {"⟦EMAIL_1⟧": f"CFO's address {EMAIL}", "⟦ORG_1⟧": "Target"}})
        assert r.status_code == 422 and "⟦EMAIL_1⟧" in r.json()["detail"] and EMAIL not in r.json()["detail"]
        assert (await c.get(f"/fd/rooms/{room['id']}", headers=ALICE_H)).json()["roles"] == {}  # nothing saved
        ok = await c.patch(f"/fd/rooms/{room['id']}", headers=ALICE_H,
                           json={"roles": {"⟦EMAIL_1⟧": "CFO's address", "⟦ORG_1⟧": "Target"}})
        assert ok.status_code == 200 and ok.json()["roles"] == {"⟦EMAIL_1⟧": "CFO's address", "⟦ORG_1⟧": "Target"}
        bad = await c.patch(f"/fd/rooms/{room['id']}", headers=ALICE_H, json={"roles": {"ORG 1": "x"}})
        assert bad.status_code == 422


async def test_folder_source_is_admin_only_and_read_in_place(deck):
    src = deck / "vdr"
    (src / "Pravno").mkdir(parents=True)
    (src / "Pravno" / "spor.txt").write_text(f"Tužitelj: {EMAIL}", encoding="utf-8")
    (src / ".hidden.txt").write_text("x")
    async with _client() as c:
        room = await _new_room(c)
        bob_room = await _new_room(c, headers=_as(BOB))
        r = await c.post(f"/fd/rooms/{bob_room['id']}/folder", headers=_as(BOB), json={"path": str(src)})
        assert r.status_code == 403
        r = await c.post(f"/fd/rooms/{room['id']}/folder", headers=ALICE_H, json={"path": str(deck / "data")})
        assert r.status_code == 422  # the deck's own data folder
        r = await c.post(f"/fd/rooms/{room['id']}/folder", headers=ALICE_H, json={"path": str(src)})
        assert r.status_code == 200, r.text
        assert r.json()["report"]["added"] == 1
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await _wait(c, room["id"])
        f = (await _files(c, room["id"]))["Pravno/spor.txt"]
        assert f["state"] == "done" and f["origin"] == "folder"
        assert (src / "Pravno" / "spor.txt").read_text(encoding="utf-8") == f"Tužitelj: {EMAIL}"  # never written
        assert not list((deck / "data" / "rooms" / room["id"] / "raw").iterdir())  # nothing copied


async def test_zip_limits(deck, monkeypatch):
    async with _client() as c:
        room = await _new_room(c)
        bomb = _zip({"zeros.txt": b"0" * (3 * 1024 * 1024)})
        r = await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=bomb)
        assert r.status_code == 422 and "bomb" in r.json()["detail"]
        r = await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=b"not a zip")
        assert r.status_code == 422
        monkeypatch.setattr(rooms_intake, "MAX_ZIP_BYTES", 100)
        r = await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({"a.txt": "x" * 500}))
        assert r.status_code == 413


async def test_a_room_is_its_owners_alone(deck):
    async with _client() as c:
        room = await _new_room(c)
        assert (await c.get(f"/fd/rooms/{room['id']}", headers=_as(BOB))).status_code == 404
        assert (await c.get("/fd/rooms", headers=_as(BOB))).json()["rooms"] == []
        acting = {**ALICE_H, auth.ACT_AS_HEADER: BOB}
        assert (await c.get("/fd/rooms", headers=acting)).status_code == 403
        assert (await c.get(f"/fd/rooms/{room['id']}", headers={})).status_code == 401


async def test_delete_room_and_its_vault(deck):
    async with _client() as c:
        room = await _new_room(c)
        r = await c.delete(f"/fd/rooms/{room['id']}", params={"vault": "true"}, headers=ALICE_H)
        assert r.json() == {"deleted": True, "vault_deleted": True}
        assert not (deck / "data" / "rooms" / room["id"]).exists()
        assert await anonymize_vault.store().get(ALICE, room["vault_id"]) is None


async def test_a_room_that_was_running_resumes_at_deck_start(deck):
    async with _client() as c:
        room = await _new_room(c)
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({"a.txt": f"Mail: {EMAIL}"}))
        await rooms_store.store().update(room["id"], state="running", job={"phase": "anonymizing"})
        assert await rooms_job.resume_all() == 1
        done = await _wait(c, room["id"])
        assert done["state"] == "review" and (await _files(c, room["id"]))["a.txt"]["state"] == "done"
