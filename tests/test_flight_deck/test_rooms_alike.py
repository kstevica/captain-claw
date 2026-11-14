"""Tokens that may be one party: code lists them, the person says, and "the
same" makes the second token the first one's spelling in the vault and in
every copy in the room, without a new run. Made-up data."""

from __future__ import annotations

import io
import zipfile

import pytest

from captain_claw.anonymize import engine
from captain_claw.flight_deck import anonymize_vault, rooms_alike
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE,
    ALICE_H,
    _client,
    _files,
    _new_room,
    _wait,
    _zip,
    deck,
)
from test_flight_deck.test_rooms_leftovers import _stub_trusted_model

E = engine.Entity


# ── the engine: one party from two ──────────────────────────────────


def test_merge_makes_the_other_token_a_spelling_and_points_to_the_kept_one():
    state = engine.VaultState([E("⟦PERSON_1⟧", "PERSON", "darko simunic", "Darko Šimunić"),
                               E("⟦PERSON_2⟧", "PERSON", "darko simunec", "Darko Šimunec", spellings=["ŠIMUNEC D."])])
    mapping = state.merge(state.by_token["⟦PERSON_1⟧"], state.by_token["⟦PERSON_2⟧"])
    assert mapping == {"⟦PERSON_2⟧": "⟦PERSON_1:2⟧", "⟦PERSON_2:2⟧": "⟦PERSON_1:3⟧"}
    # Old tokens still restore; the new ones too.
    assert engine.restore("⟦PERSON_2⟧ ⟦PERSON_1:2⟧ ⟦PERSON_1:3⟧", state)[0] == "Darko Šimunec Darko Šimunec ŠIMUNEC D."
    # What finds the other one gives the kept one from now on.
    opts = engine.Options(extra=(("Darko Šimunec", "PERSON"),), one_word_names=False)
    assert engine.anonymize_text("Plaća Darko Šimunec", state, opts)[0] == "Plaća ⟦PERSON_1:2⟧"
    assert engine.anonymize_text("Plaća Darko Šimunec", state, engine.Options(one_word_names=False))[0] \
        == "Plaća ⟦PERSON_1:2⟧"


def test_a_merged_company_found_by_its_old_name_is_the_kept_one():
    state = engine.VaultState([E("⟦ORG_1⟧", "ORG", "sokol dom", "Sokol Dom d.o.o."),
                               E("⟦ORG_2⟧", "ORG", "sokol", "Sokol d.o.o.")])
    state.merge(state.by_token["⟦ORG_1⟧"], state.by_token["⟦ORG_2⟧"])
    out = engine.anonymize_text("Zajam od Sokol d.o.o.", state, engine.Options(one_word_names=False))[0]
    assert out == "Zajam od ⟦ORG_1:2⟧"
    copy = state.snapshot()
    assert copy.by_token["⟦ORG_2⟧"].same_as == "⟦ORG_1⟧"


def test_rewrite_changes_tokens_in_office_parts_and_text():
    mapping = {"⟦PERSON_2⟧": "⟦PERSON_1:2⟧"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("xl/sharedStrings.xml", "<sst><si><t>⟦PERSON_2⟧ i ⟦PERSON_22⟧</t></si></sst>")
        z.writestr("xl/media/a.png", b"\x89PNG")
    out = rooms_alike.rewrite_blob("a.xlsx", buf.getvalue(), mapping)
    z = zipfile.ZipFile(io.BytesIO(out))
    assert z.read("xl/sharedStrings.xml").decode() == "<sst><si><t>⟦PERSON_1:2⟧ i ⟦PERSON_22⟧</t></si></sst>"
    assert z.namelist()[0] == "[Content_Types].xml" and z.read("xl/media/a.png") == b"\x89PNG"
    assert rooms_alike.rewrite_blob("a.txt", "x ⟦PERSON_2⟧".encode(), mapping) == "x ⟦PERSON_1:2⟧".encode()
    assert rooms_alike.rewrite_blob("a.bin", b"\xff\xfe" + "⟦PERSON_2⟧".encode(), mapping) is None


# ── suggestions ─────────────────────────────────────────────────────


def _suggest(entities: list[engine.Entity], files: dict[str, set[str]], not_same=()) -> list[tuple[str, str, str]]:
    out = rooms_alike.suggest(engine.VaultState(entities), files, not_same)
    return [(s["a"]["token"], s["b"]["token"], s["why"]) for s in out]


def test_suggestions_typo_name_alone_changed_surname_company_address():
    ents = [
        E("⟦PERSON_1⟧", "PERSON", "", "Darko Šimunić"), E("⟦PERSON_2⟧", "PERSON", "", "Darko Šimunč"),
        E("⟦PERSON_3⟧", "PERSON", "", "MARIBEL SANTOS REYES"), E("⟦PERSON_4⟧", "PERSON", "", "MARIBEL"),
        E("⟦PERSON_5⟧", "PERSON", "", "Jelena Kovač"), E("⟦PERSON_6⟧", "PERSON", "", "Jelena Babić"),
        E("⟦PERSON_7⟧", "PERSON", "", "Petra Horvat"), E("⟦PERSON_8⟧", "PERSON", "", "Petra Lučić"),
        E("⟦ORG_1⟧", "ORG", "", "Sokol Dom d.o.o."), E("⟦ORG_2⟧", "ORG", "", "direktor sokol doo"),
        E("⟦ADDRESS_1⟧", "ADDRESS", "", "Kovačka cesta 14, Gornje Selo"),
        E("⟦ADDRESS_2⟧", "ADDRESS", "", "Kovačka cesta 14, Velika Gorica"),
    ]
    # The register and the payroll list mostly the same people…
    staff = [E(f"⟦PERSON_{20 + i}⟧", "PERSON", "", name) for i, name in enumerate(
        ["Toma Zec", "Nera Vuk", "Lovro Sova", "Iva Lisica", "Bruno Ris", "Mara Jež"])]
    ents += staff
    files = {e.token: {"reg"} for e in ents}
    files.update({e.token: {"reg", "pay"} for e in staff})
    files["⟦PERSON_6⟧"] = {"pay"}            # …and Jelena is never on both: a married name?
    files["⟦PERSON_8⟧"] = {"reg", "pay"}     # Petra Lučić shares a file with Petra Horvat: two people
    files["⟦PERSON_7⟧"] = {"reg", "pay"}
    files["⟦ORG_3⟧"] = {"gl"}
    got = _suggest(ents, files)
    pairs = {frozenset((a, b)): why for a, b, why in got}
    assert pairs[frozenset(("⟦PERSON_1⟧", "⟦PERSON_2⟧"))] == "One letter apart"
    assert pairs[frozenset(("⟦PERSON_3⟧", "⟦PERSON_4⟧"))] == "A name alone"
    assert pairs[frozenset(("⟦PERSON_5⟧", "⟦PERSON_6⟧"))] == "Share a name, never on the same list"
    assert frozenset(("⟦PERSON_7⟧", "⟦PERSON_8⟧")) not in pairs
    assert frozenset(("⟦ORG_1⟧", "⟦ORG_2⟧")) in pairs
    assert pairs[frozenset(("⟦ADDRESS_1⟧", "⟦ADDRESS_2⟧"))] == "Same street and number, another town"
    # An answered pair isn't asked again; a token not in any file isn't asked about.
    again = _suggest(ents, files, [rooms_alike.pair_key("⟦PERSON_2⟧", "⟦PERSON_1⟧")])
    assert all(frozenset((a, b)) != frozenset(("⟦PERSON_1⟧", "⟦PERSON_2⟧")) for a, b, _w in again)
    files.pop("⟦PERSON_4⟧")
    assert all("⟦PERSON_4⟧" not in (a, b) for a, b, _w in _suggest(ents, files))
    # A ledger isn't a list of the same people: a Jelena only there is no one's other name.
    files["⟦PERSON_6⟧"] = {"gl"}
    assert frozenset(("⟦PERSON_5⟧", "⟦PERSON_6⟧")) not in {frozenset((a, b)) for a, b, _w in _suggest(ents, files)}


def test_two_letters_off_in_two_words_is_asked():
    ents = [E("⟦PERSON_1⟧", "PERSON", "", "LORENA MAY DIZEN"), E("⟦PERSON_2⟧", "PERSON", "", "Lorena Mae Dizon")]
    got = _suggest(ents, {"⟦PERSON_1⟧": {"reg"}, "⟦PERSON_2⟧": {"pay"}})
    assert [w for _a, _b, w in got] == ["A letter or two apart"]


def test_a_card_payment_s_cents_are_not_the_company_s():
    spans = [s for s in engine.d.detect("3.11/EUR/41,30SOKOL TRGOVINA D.O.O. kartica", ner=False) if s.kind == "ORG"]
    text = "3.11/EUR/41,30SOKOL TRGOVINA D.O.O. kartica"
    assert [text[s.start:s.end] for s in spans] == ["SOKOL TRGOVINA D.O.O."]


# ── a room, end to end ──────────────────────────────────────────────


async def _vault_has(room: dict, values: list[tuple[str, str]]) -> None:
    store = anonymize_vault.store()
    row = await store.get(ALICE, room["vault_id"])
    async with store.lock(row["id"]):
        state = await store.load_state(row)
        for kind, value in values:
            state.add(kind, engine.normalize(kind, value), value).count = 1
        await store.save_run(row, state)


@pytest.mark.usefixtures("deck")
async def test_same_rewrites_the_copies_and_not_the_same_is_remembered(monkeypatch):
    _stub_trusted_model(monkeypatch)
    async with _client() as c:
        room = await _new_room(c)
        rid = room["id"]
        staff = ["Toma Zec", "Nera Vuk", "Lovro Sova", "Iva Lisica", "Bruno Ris"]  # on both lists
        await _vault_has(room, [("PERSON", "Darko Šimunić"), ("PERSON", "Darko Šimunč"), ("PERSON", "Jelena Kovač"),
                                ("PERSON", "Jelena Babić"), *(("PERSON", s) for s in staff)])
        both = "".join(f" Radnik {s}." for s in staff)
        await c.put(f"/fd/rooms/{rid}/zip", headers=ALICE_H, content=_zip({
            "registar.txt": f"Radnik Darko Šimunić, kuhar. Radnica Jelena Kovač.{both}\n",
            "place.txt": f"Isplata Darko Šimunč 1.200. Isplata Jelena Babić 900.{both}\n"}))
        await c.post(f"/fd/rooms/{rid}/run", headers=ALICE_H, json={})
        await _wait(c, rid)
        r = await c.get(f"/fd/rooms/{rid}/alike", headers=ALICE_H)
        assert r.status_code == 200, r.text
        pairs = {frozenset((s["a"]["token"], s["b"]["token"])): s for s in r.json()["suggestions"]}
        typo = pairs[frozenset(("⟦PERSON_1⟧", "⟦PERSON_2⟧"))]
        assert {typo["a"]["text"], typo["b"]["text"]} == {"Darko Šimunić", "Darko Šimunč"}
        assert frozenset(("⟦PERSON_3⟧", "⟦PERSON_4⟧")) in pairs

        r = await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H, json={"keep": "⟦PERSON_1⟧", "other": "⟦PERSON_2⟧"})
        assert r.status_code == 200, r.text
        assert r.json()["rewritten"] == 1 and r.json()["requeued"] == 0
        files = await _files(c, rid)
        t = (await c.get(f"/fd/rooms/{rid}/files/{files['place.txt']['id']}/text", headers=ALICE_H)).json()
        assert "⟦PERSON_1:2⟧ 1.200" in t["clean"] and "⟦PERSON_2⟧" not in t["clean"]
        hits = (await c.get(f"/fd/rooms/{rid}/search", headers=ALICE_H, params={"q": "PERSON_1"})).json()["results"]
        assert {h["file_id"] for h in hits} >= {files["place.txt"]["id"], files["registar.txt"]["id"]}
        again = (await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H,
                              json={"keep": "⟦PERSON_1⟧", "other": "⟦PERSON_2⟧"}))
        assert again.status_code == 409

        r = await c.post(f"/fd/rooms/{rid}/not-same", headers=ALICE_H, json={"a": "⟦PERSON_4⟧", "b": "⟦PERSON_3⟧"})
        assert r.json() == {"answered": 1}
        r = (await c.get(f"/fd/rooms/{rid}/alike", headers=ALICE_H)).json()
        assert r["suggestions"] == [] and r["answered"] == 1
        assert r["merged"] == [{"token": "⟦PERSON_2⟧", "text": "Darko Šimunč", "into": "⟦PERSON_1⟧"}]

        # A check again keeps them one (the vault says so).
        await c.post(f"/fd/rooms/{rid}/run", headers=ALICE_H, json={})
        await _wait(c, rid)
        t = (await c.get(f"/fd/rooms/{rid}/files/{files['place.txt']['id']}/text", headers=ALICE_H)).json()
        assert "⟦PERSON_1:2⟧" in t["clean"] and "⟦PERSON_2⟧" not in t["clean"]
        assert (await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H,
                             json={"keep": "⟦PERSON_1:2⟧", "other": "⟦PERSON_3⟧"})).status_code == 422
