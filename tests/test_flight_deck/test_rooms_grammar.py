"""Case endings after restoring (docs/data-room-plan.md P7): for a Croatian
delivery the trusted model puts each restored person's name in the case its
sentence needs; a guard takes a form only when it is the same name with a
case ending. The plain restore stays; the owner picks either. Made-up data,
and a stand-in for the trusted model."""

from __future__ import annotations

import io
import json

import openpyxl
import pytest

from captain_claw.anonymize import engine
from captain_claw.flight_deck import anonymize_trusted, rooms_grammar, rooms_store
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE,
    ALICE_H,
    deck,
)
from test_flight_deck.test_rooms_mcp import _client, _released_room, _rooms_token, _tool
from test_flight_deck.test_rooms_own_files import _xlsx

# ── the guard and the language, alone ───────────────────────────────

@pytest.mark.parametrize("old,new", [
    ("Ivan Horvat", "Ivanom Horvatom"), ("Ivan Horvat", "Ivana Horvata"), ("Ana Kovač", "Anom Kovač"),
    ("Petar Jurić", "Petrom Jurićem"), ("Pavao Babić", "Pavla Babića"), ("Luka Novak", "Luci Novaku"),
    ("Juraj Perkovac", "Jurja Perkovca"), ("Mario Bašić", "Marija Bašića"), ("Hrvoje Matić", "Hrvojem Matićem"),
    ("Marija Kovačević-Horvat", "Mariji Kovačević-Horvat"), ("I. Horvat", "I. Horvatom"),
    ("IVAN HORVAT", "IVANOM HORVATOM"),
])
def test_the_guard_takes_case_endings(old, new):
    assert rooms_grammar.guard(old, new)


@pytest.mark.parametrize("old,new", [
    ("Ivan Horvat", "Ivica Horvat"),        # another name
    ("Marko Ilić", "Marta Ilić"),
    ("Ivan Horvat", "Ivanom"),              # a word dropped
    ("Ivan Horvat", "Horvatom Ivanom"),     # reordered
    ("Ivan Horvat", "ivanom horvatom"),     # capitals lost
    ("I. Horvat", "Ivanom Horvatom"),       # an initial filled in
    ("Ivan Horvat", "Ivan Horvat d.o.o."),
    ("Ivan Horvat", "Ivanom Horvat2"),
    ("Ivan", "Ivna"),
    ("Ivan Horvat", None),
])
def test_the_guard_refuses_anything_else(old, new):
    assert not rooms_grammar.guard(old, new)


def test_the_language_is_told_from_claudes_own_words():
    assert rooms_grammar.language("Sastanak s ⟦PERSON_1⟧ održan je u ⟦ORG_2⟧ prema ugovoru.") == "hr"
    # Croatian names restored into an English memo don't make it Croatian: only Claude's text counts.
    assert rooms_grammar.language("The meeting with ⟦PERSON_1⟧ was held at ⟦ORG_2⟧ and the amount is agreed.") == "en"
    assert rooms_grammar.language("OK.") == ""


def _state() -> engine.VaultState:
    return engine.VaultState([engine.Entity("⟦PERSON_1⟧", "PERSON", "ana kovac", "Ana Kovač"),
                              engine.Entity("⟦PERSON_2⟧", "PERSON", "petar juric", "Petar Jurić"),
                              engine.Entity("⟦ORG_1⟧", "ORG", "sokol", "Sokol Softver d.o.o."),
                              engine.Entity("⟦PERSON_3⟧", "PERSON", "ana kovac slug", "ana-kovac")])


FORMS = {"s": {"Ana Kovač": "Anom Kovač", "Petar Jurić": "Petrom Jurićem"},
         "o": {"Ana Kovač": "Ani Kovač", "Petar Jurić": "Petru Juriću"},
         "za": {"Ana Kovač": "Ivani Kovač"}}  # a model that drifts: the guard says no


def _model(seen: list | None = None):
    async def call(system: str, user: str) -> tuple[str, int]:
        assert "Croatian" in system
        out = {}
        for item in json.loads(user):
            if seen is not None:
                seen.append(item)
            before = item["sentence"].split("«", 1)[0].split()
            prep = before[-1].lower() if before else ""
            out[str(item["id"])] = FORMS.get(prep, {}).get(item["name"], item["name"])
        return json.dumps(out, ensure_ascii=False), 120
    return call


async def test_names_take_their_case_and_only_people_names():
    seen: list = []
    texts = ["Sastanak s ⟦PERSON_1⟧ i s ⟦PERSON_2⟧ u ⟦ORG_1⟧. ⟦PERSON_1⟧ je CFO.\n"
             "Razgovarali smo o ⟦PERSON_2⟧. Pismo za ⟦PERSON_1⟧. Link: https://x.example/⟦PERSON_3⟧",
             "Memo o ⟦PERSON_1⟧"]
    r = await rooms_grammar.inflect(texts, _state(), _model(seen))
    body, title = rooms_grammar.apply(texts[0], r.edits[0]), rooms_grammar.apply(texts[1], r.edits[1])
    assert body == ("Sastanak s Anom Kovač i s Petrom Jurićem u Sokol Softver d.o.o.. Ana Kovač je CFO.\n"
                    "Razgovarali smo o Petru Juriću. Pismo za Ana Kovač. Link: https://x.example/ana-kovac")
    assert title == "Memo o Ani Kovač"
    assert r.changed == 4 and r.refused == 1 and r.names == 6 and r.tokens > 0
    # The model sees each name in its own sentence, marked; never the company or the link's slug.
    assert all(i["name"] in ("Ana Kovač", "Petar Jurić") for i in seen)
    assert any(i["sentence"].startswith("Sastanak s «Ana Kovač» i s Petar Jurić") for i in seen)


# ── on the deck ─────────────────────────────────────────────────────

@pytest.fixture
def trusted_model(monkeypatch):
    """Switches the stand-in on (after the room's own run, which it isn't for)."""
    model = anonymize_trusted.TrustedModel(provider="ollama", model="stand-in", basis="local")

    async def usable(_db):
        return model

    def switch_on(call=None):
        monkeypatch.setattr(anonymize_trusted, "usable", usable)
        monkeypatch.setattr(anonymize_trusted, "completer", lambda _m: call or _model())

    return switch_on


async def _granted_room(c, switch_on=None):
    room = await _released_room(c)
    await rooms_store.store().set_grant(ALICE, "cc_oc_test", [room["id"]], False)
    if switch_on is not None:
        switch_on()
    return room


async def _settled(room_id: str, seq: int) -> None:
    task = rooms_grammar._TASKS.get((room_id, seq))
    if task is not None:
        await task


@pytest.mark.usefixtures("deck")
async def test_a_croatian_delivery_gets_its_case_endings_and_keeps_the_plain_one(trusted_model):
    async with _client() as c:
        room = await _granted_room(c, trusted_model)
        receipt = await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Memo o ⟦PERSON_1⟧",
                              content="Sastanak s ⟦PERSON_1⟧ održan je prema ugovoru. ⟦PERSON_1⟧ je CFO.")
        assert "grammar" not in receipt and "Anom" not in json.dumps(receipt)  # Claude learns nothing of it
        await _settled(room["id"], 1)
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["body_fixed"] == "Sastanak s Anom Kovač održan je prema ugovoru. Ana Kovač je CFO."
        assert d["body"] == "Sastanak s Ana Kovač održan je prema ugovoru. Ana Kovač je CFO."
        assert d["title_fixed"] == "Memo o Ani Kovač"
        g = d["report"]["grammar"]
        assert g["state"] == "done" and g["changed"] == 2 and g["language"] == "hr" and g["model"] == "ollama/stand-in"
        listing = (await c.get(f"/fd/rooms/{room['id']}/deliveries", headers=ALICE_H)).json()["deliveries"]
        assert listing[0]["title"] == "Memo o Ani Kovač"

        url = f"/fd/rooms/{room['id']}/deliveries/1/download"
        assert "s Anom Kovač" in (await c.get(url, params={"as": "md"}, headers=ALICE_H)).text
        assert "s Ana Kovač" in (await c.get(url, params={"as": "md", "plain": 1}, headers=ALICE_H)).text
        # Both versions are encrypted at rest.
        assert "Anom Kovač".encode() not in rooms_store.store().path.read_bytes()


@pytest.mark.usefixtures("deck")
async def test_english_is_left_alone_and_the_owner_can_ask_anyway(trusted_model):
    async with _client() as c:
        room = await _granted_room(c, trusted_model)
        await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Memo",
                    content="The meeting with ⟦PERSON_1⟧ was held and the amount is agreed for the company.")
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert "grammar" not in d["report"] and d["body_fixed"] is None
        r = await c.post(f"/fd/rooms/{room['id']}/deliveries/1/grammar", headers=ALICE_H)
        assert r.status_code == 200 and r.json()["grammar"]["state"] == "waiting"
        await _settled(room["id"], 1)
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["report"]["grammar"]["state"] == "done" and d["report"]["grammar"]["changed"] == 0
        assert d["body_fixed"] is None  # nothing changed: nothing extra kept


@pytest.mark.usefixtures("deck")
async def test_without_a_trusted_model_names_stay_as_written():
    async with _client() as c:
        room = await _granted_room(c)
        await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Memo",
                    content="Sastanak s ⟦PERSON_1⟧ održan je prema ugovoru.")
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["report"]["grammar"] == {"state": "unavailable", "language": "hr"}
        assert (await c.post(f"/fd/rooms/{room['id']}/deliveries/1/grammar", headers=ALICE_H)).status_code == 409


@pytest.mark.usefixtures("deck")
async def test_a_failed_model_keeps_the_plain_restore(trusted_model):
    async def broken(_s, _u):
        raise RuntimeError("ConnectError")

    async with _client() as c:
        room = await _granted_room(c, lambda: trusted_model(broken))
        await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Memo",
                    content="Sastanak s ⟦PERSON_1⟧ održan je prema ugovoru.")
        await _settled(room["id"], 1)
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["report"]["grammar"]["state"] == "failed" and d["report"]["grammar"]["error"] == "RuntimeError"
        assert d["body_fixed"] is None and "s Ana Kovač" in d["body"]


@pytest.mark.usefixtures("deck")
async def test_a_file_gets_a_fixed_copy_next_to_the_plain_one(trusted_model):
    async with _client() as c:
        room = await _granted_room(c, trusted_model)
        r = await c.put(f"/fd/rooms/{room['id']}/deliveries/file", params={"name": "pregled.xlsx"},
                        headers=ALICE_H, content=_xlsx([["Napomena"], ["Dogovor s ⟦PERSON_1⟧ je potpisan prema ugovoru, a iznos je za društvo."],
                                                        ["Kontakt: ⟦PERSON_1⟧"]]))
        seq = r.json()["seq"]
        await _settled(room["id"], seq)
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/{seq}", headers=ALICE_H)).json()
        assert d["report"]["grammar"]["changed"] == 1
        url = f"/fd/rooms/{room['id']}/deliveries/{seq}/download"

        def cells(blob: bytes) -> str:
            ws = openpyxl.load_workbook(io.BytesIO(blob)).active
            return "\n".join(str(c.value) for row in ws.iter_rows() for c in row if c.value)

        fixed = cells((await c.get(url, params={"as": "file"}, headers=ALICE_H)).content)
        plain = cells((await c.get(url, params={"as": "file", "plain": 1}, headers=ALICE_H)).content)
        assert "Dogovor s Anom Kovač" in fixed and "Kontakt: Ana Kovač" in fixed
        assert "Dogovor s Ana Kovač" in plain
        assert rooms_store.store().room_dir(room["id"]).joinpath("inbox", f"{seq}.fixed.xlsx.enc").exists()
        assert (await c.delete(f"/fd/rooms/{room['id']}/deliveries/{seq}", headers=ALICE_H)).status_code == 200
        assert not list(rooms_store.store().room_dir(room["id"]).joinpath("inbox").glob(f"{seq}.*"))


@pytest.mark.usefixtures("deck")
async def test_waiting_deliveries_carry_on_after_a_restart(trusted_model):
    async with _client() as c:
        room = await _granted_room(c, trusted_model)
        await _tool(c, _rooms_token(), "room_deliver", room=room["id"], title="Memo",
                    content="Sastanak s ⟦PERSON_1⟧ održan je prema ugovoru.")
        await _settled(room["id"], 1)
        # As if the deck stopped before the model answered.
        d = await rooms_store.store().delivery(room["id"], 1)
        await rooms_store.store().update_delivery(room["id"], 1, report={**d["report"], "grammar": {
            "state": "waiting", "language": "hr"}})
        rooms_grammar._TASKS.clear()
        assert await rooms_store.store().grammar_waiting() == [(room["id"], 1)]
        assert await rooms_grammar.resume_all() == 1
        await _settled(room["id"], 1)
        assert await rooms_store.store().grammar_waiting() == []
        d = (await c.get(f"/fd/rooms/{room['id']}/deliveries/1", headers=ALICE_H)).json()
        assert d["report"]["grammar"]["state"] == "done" and "s Anom Kovač" in d["body_fixed"]
