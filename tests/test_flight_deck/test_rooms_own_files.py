"""Files Claude made itself (docs/data-room-plan.md P5): sent as base64 (in
parts too) or dropped in by the person, the names put back inside; a file
without tokens flagged; room_get_file hands Claude a released anonymized copy
— never one that still holds a real value. Made-up data."""

from __future__ import annotations

import base64
import io

import docx
import openpyxl
import pytest

from captain_claw.flight_deck import auth, rooms_store
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE,
    ALICE_H,
    EMAIL,
    _files,
    deck,
)
from test_flight_deck.test_rooms_mcp import _client, _released_room, _rooms_token, _rpc, _tool


async def _send(c, t, **args):
    """room_deliver_file (its `name` argument collides with _tool's own)."""
    import json

    r = await _rpc(c, t, "tools/call", {"name": "room_deliver_file", "arguments": args})
    result = r.json()["result"]
    text = result["content"][0]["text"]
    return {"error": text} if result["isError"] else json.loads(text)


async def _room(c, **kw):
    room = await _released_room(c, **kw)
    await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
    return room


def _xlsx(rows, sheet="Model") -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.mark.usefixtures("deck")
async def test_claudes_own_workbook_gets_the_names_back_inside():
    async with _client() as c:
        room = await _room(c)
        blob = _xlsx([["CFO", "⟦PERSON_1⟧"], ["Mail", "⟦EMAIL_1⟧"], ["Prihod", 13980]], sheet="⟦PERSON_1⟧ model")
        receipt = await _send(c, _rooms_token(), room=room["id"], name="model ⟦PERSON_1⟧.xlsx",
                              content_base64=base64.b64encode(blob).decode())
        assert receipt["delivered"] == "#1" and receipt["file"] == "Excel workbook (.xlsx)"
        assert receipt["restored"] >= 3 and "Ana" not in str(receipt)
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["report"]["kind"] == "file" and d["report"]["source"] == "claude" and d["title"] == "model Ana Kovač.xlsx"
        got = await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "file"}, headers=ALICE_H)
        wb = openpyxl.load_workbook(io.BytesIO(got.content))
        ws = wb["Ana Kovač model"]
        assert ws["B1"].value == "Ana Kovač" and ws["B2"].value == EMAIL and ws["B3"].value == 13980
        assert "model Ana Kov" in got.headers["content-disposition"] or "model%20Ana" in got.headers["content-disposition"]


@pytest.mark.usefixtures("deck")
async def test_base64_in_parts_and_a_file_without_tokens():
    async with _client() as c:
        room = await _room(c)
        t = _rooms_token()
        data = base64.b64encode("# Bilješka\n\n⟦PERSON_1⟧ potvrđuje.\n".encode()).decode()
        a, b = data[:10], data[10:]
        first = await _send(c, t, room=room["id"], name="note.md", content_base64=b, part=2, of=2)
        assert first["waiting_for_parts"] == [1]
        done = await _send(c, t, room=room["id"], name="note.md", content_base64=a, part=1, of=2)
        assert done["delivered"] == "#1" and done["parts_joined"] == 2
        body = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()["body"]
        assert "Ana Kovač potvrđuje" in body

        plain = await _send(c, t, room=room["id"], name="summary.md",
                            content_base64=base64.b64encode(b"Project Falcon: the target grows.").decode())
        assert "No tokens were found" in plain["note"]
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/2", headers=ALICE_H)).json()
        assert d["report"]["no_tokens"] is True

        assert "base64" in (await _send(c, t, room=room["id"], name="x.md",
                                        content_base64="@@not base64@@"))["error"]
        assert "PDF" in (await _send(c, t, room=room["id"], name="x.pdf",
                                     content_base64="AAAA"))["error"]


@pytest.mark.usefixtures("deck")
async def test_the_person_drops_in_a_file_claude_made():
    async with _client() as c:
        room = await _room(c)
        d = docx.Document()
        d.add_paragraph("Memo: ⟦PERSON_1⟧ potpisuje ugovor.")
        buf = io.BytesIO()
        d.save(buf)
        r = await c.put(f"/fd/rooms/{room['id']}/deliveries/file", params={"name": "memo.docx"},
                        headers=ALICE_H, content=buf.getvalue())
        assert r.status_code == 200 and r.json()["report"]["source"] == "owner"
        listing = (await c.get(f"/fd/rooms/{room['id']}/deliveries", headers=ALICE_H)).json()["deliveries"]
        assert listing[0]["app"] == "Added by you"
        got = await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "file"}, headers=ALICE_H)
        text = "\n".join(p.text for p in docx.Document(io.BytesIO(got.content)).paragraphs)
        assert "Ana Kovač potpisuje" in text
        notes = await auth._db.list_notifications(ALICE, limit=10)
        assert not any(n["type"] == "data_room_delivery" for n in notes)  # no bell for your own drop
        bad = await c.put(f"/fd/rooms/{room['id']}/deliveries/file", params={"name": "scan.pdf"},
                          headers=ALICE_H, content=b"%PDF-1.4")
        assert bad.status_code == 422


@pytest.mark.usefixtures("deck")
async def test_room_get_file_hands_out_clean_copies_only():
    async with _client() as c:
        room = await _room(c)
        files = await _files(c, room["id"])
        t = _rooms_token()
        path = files["Pravno/ugovor.txt"]["clean_path"]
        got = await _tool(c, t, "room_get_file", room=room["id"], path=path)
        raw = base64.b64decode(got["base64"]).decode()
        assert got["bytes"] == len(raw.encode()) and "⟦PERSON_1⟧" in raw and "Ana Kovač" not in raw
        reads = (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()["reads"]
        assert reads[0]["tool"] == "room_get_file" and reads[0]["path"] == path

        # A copy that somehow still holds a real value is held back, and the owner told.
        clean = rooms_store.store().clean_file(room["id"], files["Pravno/ugovor.txt"]["id"], ".txt")
        clean.write_text(raw + "\nPotpis: Ana Kovač", encoding="utf-8")
        held = await _tool(c, t, "room_get_file", room=room["id"], path=path)
        assert "can't be handed out" in held["error"] and "Ana" not in held["error"]
        alarms = (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()["alarms"]
        assert alarms[0]["tool"] == "room_get_file (held back)" and alarms[0]["tokens"] == ["⟦PERSON_1⟧"]

        await rooms_store.store().set_file(room["id"], files["Pravno/ugovor.txt"]["id"], state="blocked",
                                           released=False)
        assert "error" in await _tool(c, t, "room_get_file", room=room["id"], path=path)
