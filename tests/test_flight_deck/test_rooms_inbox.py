"""The Inbox (docs/data-room-plan.md P3): room_deliver puts the real names back
on the deck, for the owner only. Claude's receipt has counts and tokens, never a
value — not even that it guessed one right. Parts join in order. Downloads take
no pictures. Made-up data."""

from __future__ import annotations

import io
import json
import zipfile

import docx
import pytest

from captain_claw.flight_deck import auth, rooms_store
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE,
    ALICE_H,
    BOB,
    EMAIL,
    _as,
    deck,
)
from test_flight_deck.test_rooms_mcp import _client, _released_room, _rooms_token, _tool


async def _granted_room(c, **kw):
    room = await _released_room(c, **kw)
    await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
    return room


@pytest.mark.usefixtures("deck")
async def test_a_delivery_is_restored_for_the_owner_and_claude_gets_a_receipt():
    async with _client() as c:
        room = await _granted_room(c)
        receipt = await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Memo o ⟦PERSON_1⟧",
                              content="## Sažetak\n\n⟦PERSON_1⟧ je CFO; piše na ⟦EMAIL_1⟧. ORG_77 i ⟦ORG_99⟧ "
                                      "nisu poznati.\n")
        assert receipt["delivered"] == "#1" and receipt["restored"] >= 3
        assert receipt["not_in_room"] == ["⟦ORG_99⟧"]
        assert "Ana" not in json.dumps(receipt) and EMAIL not in json.dumps(receipt)

        listing = (await c.get(f"/fd/rooms/{room['id']}/deliveries", headers=ALICE_H)).json()["deliveries"]
        assert listing[0]["title"] == "Memo o Ana Kovač" and listing[0]["seen"] is False
        assert listing[0]["report"]["real_values"] == []
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert "Ana Kovač je CFO" in d["body"] and EMAIL in d["body"] and "⟦ORG_99⟧" in d["body"]
        assert "⟦PERSON_1⟧" in d["body_tokens"] and d["title_tokens"] == "Memo o ⟦PERSON_1⟧"
        assert (await c.get(f"/fd/rooms/{room['id']}", headers=ALICE_H)).json()["inbox"] == {"total": 1, "unseen": 0}

        # Encrypted at rest; the owner's bell says nothing of the content.
        rooms_db = rooms_store.store().path.read_bytes()
        assert EMAIL.encode() not in rooms_db and "Ana Kovač".encode() not in rooms_db
        notes = await auth._db.list_notifications(ALICE, limit=10)
        note = next(n for n in notes if n["type"] == "data_room_delivery")
        assert "Ana" not in note["title"] + note["body"] and "Memo" not in note["body"]


@pytest.mark.usefixtures("deck")
async def test_parts_join_in_order():
    async with _client() as c:
        room = await _granted_room(c)
        t = _rooms_token()
        first = await _tool(c, t, "room_deliver", room=room["id"], title="Dugi memo", content="DRUGI ", part=2, of=3)
        assert first["waiting_for_parts"] == [1, 3]
        second = await _tool(c, t, "room_deliver", room=room["id"], title="Dugi memo", content="PRVI ⟦PERSON_1⟧ ",
                             part=1, of=3)
        assert second["waiting_for_parts"] == [3]
        done = await _tool(c, t, "room_deliver", room=room["id"], title="Dugi memo", content="TREĆI", part=3, of=3)
        assert done["delivered"] == "#1" and done["parts_joined"] == 3
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["body"] == "PRVI Ana Kovač DRUGI TREĆI" and d["report"]["parts"] == 3


@pytest.mark.usefixtures("deck")
async def test_a_real_value_claude_wrote_is_kept_from_claude_and_raised_for_the_owner():
    async with _client() as c:
        room = await _granted_room(c)
        receipt = await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Pretpostavka",
                              content="Mislim da je ⟦PERSON_1⟧ zapravo Ana Kovač.")
        # The same receipt as any other: nothing tells Claude its guess was right.
        assert set(receipt) == {"delivered", "restored", "characters", "note"} and "Ana" not in json.dumps(receipt)
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["report"]["real_values"] == ["⟦PERSON_1⟧"]
        alarms = (await c.get(f"/fd/rooms/{room['id']}/reads", headers=ALICE_H)).json()["alarms"]
        assert alarms[0]["tool"] == "room_deliver" and alarms[0]["tokens"] == ["⟦PERSON_1⟧"]


@pytest.mark.usefixtures("deck")
async def test_downloads_have_the_names_and_no_pictures():
    async with _client() as c:
        room = await _granted_room(c)
        await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Memo ⟦PERSON_1⟧",
                    content="# Naslov\n\n⟦PERSON_1⟧ potpisuje.\n\n![logo](/etc/hosts)\n\n![](https://example.com/x.png)\n")
        md = await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "md"}, headers=ALICE_H)
        assert md.status_code == 200 and "Ana Kovač potpisuje" in md.text
        assert "attachment" in md.headers["content-disposition"]
        word = await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "docx"}, headers=ALICE_H)
        assert word.status_code == 200
        names = zipfile.ZipFile(io.BytesIO(word.content)).namelist()
        assert not any("/media/" in n for n in names)  # no picture went in, local or remote
        text = "\n".join(p.text for p in docx.Document(io.BytesIO(word.content)).paragraphs)
        assert "Ana Kovač potpisuje" in text and "[picture: logo]" in text
        assert (await c.get(f"/fd/rooms/{room['id']}/deliveries/1/download", params={"as": "pdf"},
                            headers=ALICE_H)).status_code == 422


@pytest.mark.usefixtures("deck")
async def test_only_the_owner_and_only_granted_rooms():
    async with _client() as c:
        room = await _released_room(c)  # no grant
        out = await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="x", content="y")
        assert "error" in out
        await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
        await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="x", content="y")
        assert (await c.get(f"/fd/rooms/{room['id']}/deliveries", headers=_as(BOB))).status_code == 404
        assert (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=_as(BOB))).status_code == 404
        acting = {**ALICE_H, auth.ACT_AS_HEADER: BOB}
        assert (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=acting)).status_code == 403
        assert (await c.delete(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json() == {"deleted": True}


@pytest.mark.usefixtures("deck")
async def test_limits():
    async with _client() as c:
        room = await _granted_room(c)
        t = _rooms_token()
        too_big = await _tool(c, t, "room_deliver", room=room["id"], title="x", content="a" * 200_001)
        assert "parts" in too_big["error"]
        assert "title" in (await _tool(c, t, "room_deliver", room=room["id"], title=" ", content="y"))["error"]
        assert "between" in (await _tool(c, t, "room_deliver", room=room["id"], title="x", content="y",
                                         part=4, of=3))["error"]
