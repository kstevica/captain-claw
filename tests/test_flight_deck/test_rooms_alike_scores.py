"""Code's score for a "Same party?" pair — how sure it is the two are one —
and answering many pairs at once: all at a score and above are one, all
under it aren't. Made-up data."""

from __future__ import annotations

import re

import pytest

from captain_claw.anonymize import engine
from captain_claw.flight_deck import rooms_alike
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE_H,
    _client,
    _files,
    _new_room,
    _wait,
    _zip,
    deck,
)
from test_flight_deck.test_rooms_alike import _vault_has
from test_flight_deck.test_rooms_leftovers import _stub_trusted_model

E = engine.Entity


def _sim(kind: str, a: str, b: str) -> int:
    return round(rooms_alike.similarity(kind, a, b)[0] * 100)


def test_a_cut_off_name_or_a_diacritic_is_near_sure():
    cut = rooms_alike.similarity("ORG", "Pekara i slastičarnica Sokol, vl. Ana Lisac",
                                 "Pekara i slastičarnica Sokol, vl. Ana")
    assert cut == (0.97, "One is the start of the other")
    assert _sim("ORG", "SOKOL GRADNJA D.O.O", "Sokol Gradnja d.o.o.") >= 99
    assert _sim("PERSON", "Darko Šimunić", "Darko Šimunč") >= 90


def test_a_short_name_or_a_short_word_a_letter_off_is_never_sure():
    sim, note = rooms_alike.similarity("ORG", "Sokol Nekretnine d.o.o.", "Sokol d.o.o.")
    assert 0.8 <= sim < 0.9 and note == "A short name"  # or a sister company
    assert _sim("PERSON", "Ivan Marić", "Ivan Matić") < 90  # two real surnames
    assert _sim("ORG", "OLA-TRANS d.o.o.", "ALA-TRANS d.o.o.") < 80


def test_two_legal_forms_are_likelier_two_companies():
    sim, note = rooms_alike.similarity("ORG", "Elektro Lisac d.o.o.", "LISAC, obrt")
    assert sim < 0.7 and note == "One is a d.o.o., the other an obrt"
    assert rooms_alike._forms("Sokol Hrvatska d.o.o. k.d.") == {"doo", "kd"}
    assert rooms_alike._forms("ADD studio") == set() and rooms_alike._forms("Sokol j.d.o.o.") == {"jdoo"}


def _scored(entities, files, counts=None) -> dict[frozenset, dict]:
    out = rooms_alike.suggest(engine.VaultState(entities), files, (), counts)
    return {frozenset((s["a"]["token"], s["b"]["token"])): s for s in out}


def test_how_often_each_is_written_moves_a_typo():
    ents = [E("⟦PERSON_1⟧", "PERSON", "", "Darko Šimunić"), E("⟦PERSON_2⟧", "PERSON", "", "Darko Šimunč")]
    files = {"⟦PERSON_1⟧": {"gl"}, "⟦PERSON_2⟧": {"gl"}}
    pair = frozenset(("⟦PERSON_1⟧", "⟦PERSON_2⟧"))
    plain = _scored(ents, files)[pair]["score"]
    rare = _scored(ents, files, {"⟦PERSON_1⟧": 40, "⟦PERSON_2⟧": 1})[pair]
    often = _scored(ents, files, {"⟦PERSON_1⟧": 12, "⟦PERSON_2⟧": 14})[pair]
    assert rare["score"] > plain > often["score"]
    assert rare["notes"] == ["Written 40× and 1×: a typo?"]
    assert often["notes"] == ["Each written often (14× and 12×): two people?"]  # the kept one first
    assert rare["a"]["token"] == "⟦PERSON_1⟧"  # the one written more often is kept


def test_a_name_alone_that_fits_two_people_is_split():
    ents = [E("⟦PERSON_1⟧", "PERSON", "", "Maribel Santos Reyes"), E("⟦PERSON_2⟧", "PERSON", "", "MARIBEL")]
    files = {e.token: {"reg"} for e in ents}
    one = _scored(ents, files)[frozenset(("⟦PERSON_1⟧", "⟦PERSON_2⟧"))]
    assert one["score"] == 60 and one["notes"] == []
    ents.append(E("⟦PERSON_3⟧", "PERSON", "", "Maribel Cruz"))
    files["⟦PERSON_3⟧"] = {"reg"}
    two = _scored(ents, files)
    first = two[frozenset(("⟦PERSON_1⟧", "⟦PERSON_2⟧"))]
    assert first["score"] == 30 and first["notes"] == ["Also like “Maribel Cruz”"]


def test_a_cut_off_company_is_found_and_listed_first():
    ents = [E("⟦ORG_1⟧", "ORG", "", "Pekara i slastičarnica Sokol, vl. Ana Lisac"),
            E("⟦ORG_2⟧", "ORG", "", "Pekara i slastičarnica Sokol, vl. A"),
            E("⟦PERSON_1⟧", "PERSON", "", "Jelena Kovač"), E("⟦PERSON_2⟧", "PERSON", "", "Jelena")]
    out = rooms_alike.suggest(engine.VaultState(ents), {e.token: {"gl"} for e in ents})
    assert [(s["why"], s["score"]) for s in out] == [("One is the start of the other", 97), ("A name alone", 60)]


# ── many at once ────────────────────────────────────────────────────


@pytest.mark.usefixtures("deck")
async def test_many_pairs_are_made_one_at_once_and_many_not(monkeypatch):
    _stub_trusted_model(monkeypatch)
    async with _client() as c:
        room = await _new_room(c)
        rid = room["id"]
        await _vault_has(room, [("PERSON", "Darko Šimunić"), ("PERSON", "Darko Šimunč"), ("PERSON", "Darko Šimnuić"),
                                ("PERSON", "Ivana Horvatić"), ("PERSON", "Ivana Hrovatić"),
                                ("PERSON", "Jelena Kovač"), ("PERSON", "Jelena")])
        await c.put(f"/fd/rooms/{rid}/zip", headers=ALICE_H, content=_zip({
            "registar.txt": "Radnik Darko Šimunić, kuhar. Radnica Ivana Horvatić. Radnica Jelena Kovač.\n",
            "ugovor.txt": "Ugovor: Darko Šimunić i Ivana Horvatić.\n",
            "place.txt": "Isplata Darko Šimunč 1.200. Isplata Ivana Hrovatić 900. Bolovanje Jelena.\n",
            "gl.txt": "Putni nalog Darko Šimnuić.\n"}))
        await c.post(f"/fd/rooms/{rid}/run", headers=ALICE_H, json={})
        await _wait(c, rid)
        got = (await c.get(f"/fd/rooms/{rid}/alike", headers=ALICE_H)).json()["suggestions"]
        assert [s["score"] for s in got] == sorted((s["score"] for s in got), reverse=True)
        sure = [s for s in got if s["score"] >= 90]
        assert {(s["a"]["text"], s["b"]["text"]) for s in sure} == {  # the one in more files kept
            ("Darko Šimunić", "Darko Šimunč"), ("Darko Šimunić", "Darko Šimnuić"), ("Ivana Horvatić", "Ivana Hrovatić")}

        # All at 90% and more, with a pair the vault hasn't and one already one at the end.
        pairs = [{"keep": s["a"]["token"], "other": s["b"]["token"]} for s in sure]
        pairs += [{"keep": "⟦PERSON_1⟧", "other": "⟦PERSON_99⟧"}, pairs[0]]
        r = await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H, json={"pairs": pairs})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["merged"] == 3 and body["requeued"] == 0 and body["rewritten"] == 2  # place.txt, gl.txt
        assert [s["why"] for s in body["skipped"]] == [
            "This room's vault doesn't have that token.", f"{pairs[0]['other']} is already one with ⟦PERSON_1⟧."]
        files = await _files(c, rid)
        place = (await c.get(f"/fd/rooms/{rid}/files/{files['place.txt']['id']}/text", headers=ALICE_H)).json()["clean"]
        gl = (await c.get(f"/fd/rooms/{rid}/files/{files['gl.txt']['id']}/text", headers=ALICE_H)).json()["clean"]
        assert re.search(r"⟦PERSON_1:[23]⟧ 1\.200", place) and "⟦PERSON_4:2⟧ 900" in place
        assert re.search(r"⟦PERSON_1:[23]⟧", gl) and "⟦PERSON_3⟧" not in gl

        # All under: not the same, never asked again.
        left = (await c.get(f"/fd/rooms/{rid}/alike", headers=ALICE_H)).json()
        assert left["suggestions"] and all(s["score"] < 90 for s in left["suggestions"])
        assert len(left["merged"]) == 3
        r = await c.post(f"/fd/rooms/{rid}/not-same", headers=ALICE_H, json={
            "pairs": [{"a": s["a"]["token"], "b": s["b"]["token"]} for s in left["suggestions"]]})
        assert r.json() == {"answered": len(left["suggestions"])}
        assert (await c.get(f"/fd/rooms/{rid}/alike", headers=ALICE_H)).json()["suggestions"] == []

        # One answer stays one answer: the old body still works, and a bad list is said.
        assert (await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H,
                             json={"keep": "⟦PERSON_1⟧", "other": "⟦PERSON_2⟧"})).status_code == 409
        assert (await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H, json={"pairs": []})).status_code == 422
        assert (await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H,
                             json={"pairs": [{"keep": "x", "other": "⟦PERSON_2⟧"}]})).status_code == 422
