"""Leftovers: real values the final sweep leaves on purpose (a surname alone,
a company in lower case) that the leak check then finds. The check hides
them with the token it named; what it can't hide blocks the file, and a
spelling that isn't a name here can be left as it is. Made-up data."""

from __future__ import annotations

import pytest

from captain_claw.anonymize import engine
from captain_claw.flight_deck import anonymize_trusted, anonymize_vault, rooms_job, rooms_store
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

E = engine.Entity


# ── the engine: pinned spellings ────────────────────────────────────


def _state() -> engine.VaultState:
    return engine.VaultState([E("⟦PERSON_2⟧", "PERSON", "horvat", "Horvat"),
                              E("⟦ORG_4⟧", "ORG", "sokol softver", "Sokol Softver d.o.o."),
                              E("⟦PERSON_7⟧", "PERSON", "~kovacic", "Kovačić")])


def test_a_pinned_spelling_gets_the_token_it_is_pinned_to():
    state = _state()
    opts = engine.Options(one_word_names=False, pinned=(
        ("Horvat", "⟦PERSON_2⟧"), ("sokol softver d.o.o.", "⟦ORG_4⟧"), ("KOVACIC", "⟦PERSON_7⟧"),
        ("Nikola", "⟦PERSON_99⟧")))  # a token the vault doesn't know: ignored
    out, _r = engine.anonymize_text("Uplata Horvat; dobavljač sokol  softver d.o.o.; KOVACIC; Nikola", state, opts)
    assert out == "Uplata ⟦PERSON_2⟧; dobavljač ⟦ORG_4⟧; ⟦PERSON_7:2⟧; Nikola"
    back, _n, _u = engine.restore(out, state)
    assert back == "Uplata Horvat; dobavljač Sokol Softver d.o.o.; KOVACIC; Nikola"


def test_pinned_spellings_are_whole_words_only():
    state = _state()
    opts = engine.Options(one_word_names=False, pinned=(("Horvat", "⟦PERSON_2⟧"),))
    out, _r = engine.anonymize_text("Horvatić nije Horvat, a ni horvatski.", state, opts)
    assert out == "Horvatić nije ⟦PERSON_2⟧, a ni horvatski."
    without, _r = engine.anonymize_text("Uplata Horvat.", _state(), engine.Options(one_word_names=False))
    assert without == "Uplata Horvat."  # the sweep leaves a surname alone on purpose


# ── the check: where, as written ────────────────────────────────────


def test_leftovers_are_found_as_written_with_counts():
    needle_list = rooms_job.needles(_state())
    text = "Uplata Horvat.\nHORVAT opet, i Horvat.\ndobavljač sokol softver d.o.o.\nKovačićeva ulica\nKovacic"
    found = rooms_job.find_leftovers(text, needle_list)
    by = {(x["token"], x["text"]): x["count"] for x in found}
    assert by[("⟦PERSON_2⟧", "Horvat")] == 3  # any case: one spelling, as first written
    assert ("⟦PERSON_2⟧", "HORVAT") not in by
    assert by[("⟦ORG_4⟧", "sokol softver d.o.o")] == 1
    assert by[("⟦PERSON_7⟧", "Kovacic")] == 1 and not any(t == "Kovačićeva" for _k, t in by)
    assert rooms_job.find_leftovers("Ništa ovdje.", needle_list) == []


def test_a_name_inside_a_longer_one_counts_once_for_the_longer():
    state = engine.VaultState([E("⟦ORG_1⟧", "ORG", "galeb softver", "Galeb Softver d.o.o."),
                               E("⟦ORG_2⟧", "ORG", "softver", "Softver d.o.o.")])
    found = rooms_job.find_leftovers("platio galeb softver d.o.o.; i softver d.o.o. sam", rooms_job.needles(state))
    assert sorted((x["token"], x["text"], x["count"]) for x in found) == [
        ("⟦ORG_1⟧", "galeb softver d.o.o", 1), ("⟦ORG_2⟧", "softver d.o.o", 1)]


def test_allowed_spellings_are_not_looked_for():
    state = engine.VaultState([E("⟦PERSON_1⟧", "PERSON", "ured", "Ured"),
                               E("⟦ORG_2⟧", "ORG", "plava rijeka capital", "Plava Rijeka Capital")])
    plain = rooms_job.needles(state)
    assert rooms_job.leaks("Ured je zatvoren. Plava Rijeka PRC Capital.", plain) == ["⟦PERSON_1⟧", "⟦ORG_2⟧"]
    allowed = rooms_job.needles(state, ["URED", "plava rijeka prc capital"])
    assert rooms_job.leaks("Ured je zatvoren. Plava Rijeka PRC Capital.", allowed) == []
    assert rooms_job.leaks("Plava Rijeka Investment Capital ili Plava Rijeka – Capital", allowed) == ["⟦ORG_2⟧"]


# ── a room, end to end ──────────────────────────────────────────────


def _stub_trusted_model(monkeypatch) -> None:
    """A trusted model that finds nothing: names are then its call, so code
    takes no surname alone — as on a deck with a model set up."""
    model = anonymize_trusted.TrustedModel(provider="ollama", model="stand-in", basis="local")

    async def usable(_db):
        return model

    async def complete(system, user):
        return '{"spans": []}', 1

    monkeypatch.setattr(anonymize_trusted, "usable", usable)
    monkeypatch.setattr(anonymize_trusted, "completer", lambda _m: complete)


async def _vault_knows(room: dict, entities: list[engine.Entity]) -> None:
    store = anonymize_vault.store()
    row = await store.get(ALICE, room["vault_id"])
    async with store.lock(row["id"]):
        state = await store.load_state(row)
        for e in entities:
            state.add(e.kind, e.key, e.original).count = 1  # a run stores what it used
        await store.save_run(row, state)


@pytest.mark.usefixtures("deck")
async def test_the_check_hides_what_the_sweep_left_with_the_token_it_named(monkeypatch):
    _stub_trusted_model(monkeypatch)
    async with _client() as c:
        room = await _new_room(c)
        await _vault_knows(room, [E("", "PERSON", "horvat", "Horvat"), E("", "ORG", "sokol softver",
                                                                         "Sokol Softver d.o.o.")])
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({
            "a.txt": "Uplata Horvat, račun 12.\n", "b.txt": "Dobavljač sokol softver d.o.o. je platio.\n"}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        done = await _wait(c, room["id"])
        assert done["job"]["blocked"] == 0 and done["job"]["hidden"] == 2
        files = await _files(c, room["id"])
        assert {f["state"] for f in files.values()} == {"done"}
        assert files["a.txt"]["leftovers"] == {"hidden": 1, "left": 0}  # counts only in the list

        t = (await c.get(f"/fd/rooms/{room['id']}/files/{files['a.txt']['id']}/text", headers=ALICE_H)).json()
        assert "Horvat" not in t["clean"] and "⟦PERSON_1⟧" in t["clean"]
        assert t["leftovers"]["hidden"] == [{"token": "⟦PERSON_1⟧", "text": "Horvat", "count": 1}]
        t = (await c.get(f"/fd/rooms/{room['id']}/files/{files['b.txt']['id']}/text", headers=ALICE_H)).json()
        assert "sokol" not in t["clean"] and "⟦ORG_1⟧" in t["clean"]

        # Check again: nothing new to hide, and what was hidden stays listed.
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        again = await _wait(c, room["id"])
        assert again["job"]["hidden"] == 0
        assert (await _files(c, room["id"]))["a.txt"]["leftovers"] == {"hidden": 1, "left": 0}

    db = rooms_store.store().path
    stored = b"".join(p.read_bytes() for p in db.parent.glob(f"{db.name}*"))
    assert stored
    assert b"Horvat" not in stored and b"sokol" not in stored  # the spellings are sealed


@pytest.mark.usefixtures("deck")
async def test_not_a_name_here_leaves_it_and_the_check_lets_it_be(monkeypatch):
    _stub_trusted_model(monkeypatch)
    async with _client() as c:
        room = await _new_room(c)
        await _vault_knows(room, [E("", "PERSON", "ured", "Ured")])  # the model's mistake: an office, not a person
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({
            "a.txt": "Ured je zatvoren u petak.\n", "b.txt": "Ništa osobno.\n"}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await _wait(c, room["id"])
        files = await _files(c, room["id"])
        assert files["a.txt"]["leftovers"]["hidden"] == 1  # hidden after the check, as the vault says

        r = await c.post(f"/fd/rooms/{room['id']}/allowed", headers=ALICE_H,
                         json={"text": "Ured", "tokens": ["⟦PERSON_1⟧"], "rerun": True})
        assert r.status_code == 200, r.text
        assert r.json()["requeued"] == 1 and [a["text"] for a in r.json()["room"]["allowed"]] == ["Ured"]
        assert (await _files(c, room["id"]))["a.txt"]["state"] == "new"

        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await _wait(c, room["id"])
        files = await _files(c, room["id"])
        assert files["a.txt"]["state"] == "done" and files["a.txt"]["leftovers"] == {"hidden": 0, "left": 0}
        t = (await c.get(f"/fd/rooms/{room['id']}/files/{files['a.txt']['id']}/text", headers=ALICE_H)).json()
        assert t["clean"].startswith("Ured je zatvoren")

        # Looked for again: the next check hides it again.
        r = await c.post(f"/fd/rooms/{room['id']}/allowed/remove", headers=ALICE_H, json={"text": "URED"})
        assert r.json()["room"]["allowed"] == []
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await _wait(c, room["id"])
        t = (await c.get(f"/fd/rooms/{room['id']}/files/{files['a.txt']['id']}/text", headers=ALICE_H)).json()
        assert "Ured" not in t["clean"]

        assert (await c.post(f"/fd/rooms/{room['id']}/allowed", headers=ALICE_H,
                             json={"text": "Ivo"})).status_code == 422  # too short to be looked for anyway


@pytest.mark.usefixtures("deck")
async def test_what_the_check_cant_hide_still_blocks_and_says_what(monkeypatch):
    _stub_trusted_model(monkeypatch)

    async def nothing_hidden(room_id, job, f, path, blob, text, hideable, data, needle_list, owner, actor):
        return await __import__("asyncio").to_thread(rooms_job.find_leftovers, text, needle_list), [], blob, text

    monkeypatch.setattr(rooms_job, "_hide_leftovers", nothing_hidden)
    async with _client() as c:
        room = await _new_room(c)
        await _vault_knows(room, [E("", "PERSON", "horvat", "Horvat")])
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({"a.txt": "Uplata Horvat.\n"}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        done = await _wait(c, room["id"])
        assert done["job"]["blocked"] == 1
        f = (await _files(c, room["id"]))["a.txt"]
        assert f["state"] == "blocked" and f["leftovers"] == {"hidden": 0, "left": 1}
        assert "⟦PERSON_1⟧" in f["error"] and "Horvat" not in f["error"]
        t = (await c.get(f"/fd/rooms/{room['id']}/files/{f['id']}/text", headers=ALICE_H)).json()
        assert t["leftovers"]["left"] == [{"token": "⟦PERSON_1⟧", "text": "Horvat", "count": 1}]

        await c.post(f"/fd/rooms/{room['id']}/allowed", headers=ALICE_H, json={"text": "Horvat"})
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})  # Check again
        await _wait(c, room["id"])
        assert (await _files(c, room["id"]))["a.txt"]["state"] == "done"
