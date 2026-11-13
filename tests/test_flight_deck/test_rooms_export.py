"""“Download for Claude” (docs/data-room-plan.md P6): a released room as a ZIP
for a Claude Project — clean copies, roles and a README, every file checked
once more and left out when it still holds a real value; one download; logged.
Made-up data."""

from __future__ import annotations

import io
import zipfile

import docx
import pytest

from captain_claw.flight_deck import rooms_store
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE_H,
    BOB,
    EMAIL,
    IBAN,
    _as,
    _files,
    deck,
)
from test_flight_deck.test_rooms_mcp import _client, _released_room


def _letter() -> bytes:
    d = docx.Document()
    d.add_paragraph("Ime i prezime: Ana Kovač")
    d.add_paragraph(f"E-mail: {EMAIL}")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


async def _room(c):
    room = await _released_room(c, files={
        "Pravno/ugovor.txt": f"Ime i prezime: Ana Kovač\nE-mail: {EMAIL}\nIBAN: {IBAN}\n",
        "Pravno/pismo.docx": _letter(),
    })
    await c.patch(f"/fd/rooms/{room['id']}", headers=ALICE_H, json={"roles": {"⟦PERSON_1⟧": "CFO of the target"}})
    return room


def _all_bytes(z: zipfile.ZipFile) -> bytes:
    """Every member, and every member of an Office file inside, as bytes."""
    out = []
    for name in z.namelist():
        data = z.read(name)
        out.append(data)
        if name.endswith((".docx", ".xlsx", ".pptx")):
            inner = zipfile.ZipFile(io.BytesIO(data))
            out += [inner.read(n) for n in inner.namelist()]
    return b"\n".join(out)


@pytest.mark.usefixtures("deck")
async def test_the_zip_has_clean_files_roles_and_a_readme_and_downloads_once():
    async with _client() as c:
        room = await _room(c)
        r = await c.post(f"/fd/rooms/{room['id']}/export", headers=ALICE_H, json={})
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["files"] == 2 and info["held_back"] == []
        got = await c.get(f"/fd/rooms/{room['id']}/export/{info['token']}", headers=ALICE_H)
        assert got.status_code == 200 and "Project%20Falcon%20-%20for%20Claude.zip" in got.headers["content-disposition"]
        z = zipfile.ZipFile(io.BytesIO(got.content))
        names = z.namelist()
        assert "Project Falcon/ROLES.md" in names and "Project Falcon/README-for-Claude.md" in names
        assert any(n.endswith(".docx") for n in names)  # Office files as files
        assert "| ⟦PERSON_1⟧ | CFO of the target |" in z.read("Project Falcon/ROLES.md").decode()
        assert "brackets included" in z.read("Project Falcon/README-for-Claude.md").decode()
        everything = _all_bytes(z)
        assert b"Ana Kova" not in everything and EMAIL.encode() not in everything and IBAN.encode() not in everything
        # One download only; the export shows with the room's reads.
        again = await c.get(f"/fd/rooms/{room['id']}/export/{info['token']}", headers=ALICE_H)
        assert again.status_code == 404
        reads = (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()["reads"]
        assert reads[0]["tool"] == "export" and reads[0]["app"] == "You"


@pytest.mark.usefixtures("deck")
async def test_text_only_turns_office_files_into_markdown():
    async with _client() as c:
        room = await _room(c)
        info = (await c.post(f"/fd/rooms/{room['id']}/export", headers=ALICE_H, json={"text_only": True})).json()
        z = zipfile.ZipFile(io.BytesIO((await c.get(f"/fd/rooms/{room['id']}/export/{info['token']}",
                                                    headers=ALICE_H)).content))
        assert not any(n.endswith(".docx") for n in z.namelist())
        md = next(n for n in z.namelist() if n.endswith(".docx.md"))
        assert "⟦PERSON_1⟧" in z.read(md).decode() and "as text" in z.read("Project Falcon/README-for-Claude.md").decode()


@pytest.mark.usefixtures("deck")
async def test_a_file_still_holding_a_real_value_is_left_out():
    async with _client() as c:
        room = await _room(c)
        files = await _files(c, room["id"])
        f = files["Pravno/ugovor.txt"]
        clean = rooms_store.store().clean_file(room["id"], f["id"], ".txt")
        clean.write_text(clean.read_text(encoding="utf-8") + "\nPotpis: Ana Kovač", encoding="utf-8")
        info = (await c.post(f"/fd/rooms/{room['id']}/export", headers=ALICE_H, json={})).json()
        assert info["held_back"] == [f["clean_path"]] and info["files"] == 1
        z = zipfile.ZipFile(io.BytesIO((await c.get(f"/fd/rooms/{room['id']}/export/{info['token']}",
                                                    headers=ALICE_H)).content))
        assert not any(n.endswith(f["clean_path"]) for n in z.namelist())
        assert b"Ana Kova" not in _all_bytes(z)
        alarms = (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()["alarms"]
        assert alarms[0]["tool"] == "export (held back)" and alarms[0]["tokens"] == ["⟦PERSON_1⟧"]


@pytest.mark.usefixtures("deck")
async def test_nothing_released_and_someone_elses_room():
    async with _client() as c:
        empty = (await c.post("/fd/rooms", headers=ALICE_H, json={"codename": "Empty"})).json()
        assert (await c.post(f"/fd/rooms/{empty['id']}/export", headers=ALICE_H, json={})).status_code == 409
        room = await _room(c)
        assert (await c.post(f"/fd/rooms/{room['id']}/export", headers=_as(BOB), json={})).status_code == 404
        info = (await c.post(f"/fd/rooms/{room['id']}/export", headers=ALICE_H, json={})).json()
        assert (await c.get(f"/fd/rooms/{room['id']}/export/{info['token']}", headers=_as(BOB))).status_code == 404
        assert (await c.get(f"/fd/rooms/{room['id']}/export/not-a-token", headers=ALICE_H)).status_code == 404
