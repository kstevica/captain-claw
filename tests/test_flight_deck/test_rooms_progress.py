"""How far a data room's run is, for big files: the step a file is at, the
trusted model's chunk n of N, and how big each file is (rows and sheets for a
table). Made-up data."""

from __future__ import annotations

import io

import openpyxl
import pytest

from captain_claw.anonymize import llm_lane as L
from captain_claw.flight_deck import anonymize_routes as ar
from captain_claw.flight_deck import rooms_intake
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE_H,
    _client,
    _new_room,
    _zip,
    deck,
)


def _xlsx() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Plaće"
    ws.append(["Ime", "Iznos"])
    for i in range(1, 41):
        ws.append([f"Djelatnik {i}", i * 1000])
    ws["A60"].number_format = "0.00"  # formatted, empty: not a row
    second = wb.create_sheet("Sažetak")
    second.append(["Ukupno", "=SUM(Plaće!B2:B41)"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_table_info_counts_rows_with_something_in_them():
    assert rooms_intake.table_info("place.xlsx", _xlsx()) == {"rows": 42, "sheets": 2}
    assert rooms_intake.table_info("a.csv", b"ime;iznos\nAna;1\n\nIvo;2\n") == {"rows": 3}
    assert rooms_intake.table_info("a.docx", b"whatever") == {}
    assert rooms_intake.table_info("broken.xlsx", b"not a zip") == {}


async def test_the_model_lane_tells_how_far_it_is(monkeypatch):
    monkeypatch.setattr(L, "CHUNK_CHARS", 40)
    seen: list[tuple[int, int]] = []

    async def model(system, user):
        return '{"spans": []}', 1

    async def progress(done, total):
        seen.append((done, total))

    text = "\n\n".join(f"Paragraph {i} about the harbour and the boats." for i in range(5))
    result = await L.detect(text, model, on_progress=progress)
    n = result.chunks
    assert n >= 3 and seen[0] == (0, n) and seen[-1] == (n, n) and len(seen) == n + 1
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


async def test_anonymize_blob_reports_its_steps(monkeypatch):
    from captain_claw.flight_deck import anonymize_trusted

    monkeypatch.setattr(L, "CHUNK_CHARS", 60)
    model = anonymize_trusted.TrustedModel(provider="ollama", model="stand-in", basis="local")

    async def usable(_db):
        return model

    async def complete(system, user):
        return '{"spans": []}', 1

    monkeypatch.setattr(anonymize_trusted, "usable", usable)
    monkeypatch.setattr(anonymize_trusted, "completer", lambda _m: complete)
    monkeypatch.setattr(ar, "get_db", lambda: None)
    events: list[dict] = []

    async def hook(e):
        events.append(dict(e))

    token = ar.PROGRESS.set(hook)
    try:
        text = "\n\n".join(f"Stavka {i}: Sastanak je održan u utorak prema dogovoru." for i in range(40)).encode()
        await ar.anonymize_blob("user-progress", "memo.txt", text, {"mode": "one_way", "lane": "both"},
                                actor="test", source="test")
    finally:
        ar.PROGRESS.reset(token)
    import asyncio

    await asyncio.sleep(0.05)  # the engine's ticks come from its worker thread through the loop
    steps = [e["step"] for e in events]
    assert steps[0] == "reading" and "model" in steps and steps.index("model") < steps.index("replacing")
    assert "writing" in steps and steps.index("replacing") < steps.index("writing")
    chunks = [e for e in events if e["step"] == "model"]
    assert chunks[0]["chunk"] == 0 and chunks[-1]["chunk"] == chunks[-1]["chunks"] >= 2
    start = next(e for e in events if e["step"] == "replacing")
    assert start["unit"] == "parts" and start["pieces"] >= 1
    passes = {(e["round"], e["pass"]) for e in events if "pass" in e}
    assert (1, "looking") in passes and (1, "replacing") in passes


@pytest.mark.usefixtures("deck")
async def test_stop_stops_mid_file_and_says_so(monkeypatch):
    """Stop cancels the run at once, even in the middle of a file the trusted
    model is reading; that file stays "not run yet" and the room says so."""
    import asyncio

    from captain_claw.flight_deck import anonymize_trusted, rooms_job

    model = anonymize_trusted.TrustedModel(provider="ollama", model="stand-in", basis="local")
    started = asyncio.Event()

    async def usable(_db):
        return model

    async def slow(system, user):
        started.set()
        await asyncio.sleep(30)  # a local model on a long chunk
        return '{"spans": []}', 1

    monkeypatch.setattr(anonymize_trusted, "usable", usable)
    monkeypatch.setattr(anonymize_trusted, "completer", lambda _m: slow)
    async with _client() as c:
        room = await _new_room(c)
        text = "\n\n".join(f"Stavka {i}: sastanak je održan prema dogovoru." for i in range(50))
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H,
                    content=_zip({"a.txt": text, "b.txt": text + " kraj"}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        await asyncio.wait_for(started.wait(), 10)
        busy = (await c.get(f"/fd/rooms/{room['id']}", headers=ALICE_H)).json()
        assert busy["job"]["step"] == "model" and busy["job"]["file_index"] == 1 and busy["job"]["chunks"] >= 1

        t0 = asyncio.get_running_loop().time()
        r = await c.post(f"/fd/rooms/{room['id']}/stop", headers=ALICE_H)
        assert r.json()["stopping"] is True and asyncio.get_running_loop().time() - t0 < 5
        after = (await c.get(f"/fd/rooms/{room['id']}", headers=ALICE_H)).json()
        assert not rooms_job.running(room["id"]) and after["state"] != "running"
        assert after["job"]["phase"] == "done" and "step" not in after["job"]
        assert after["job"]["message"].startswith("Stopped by you — 2 files not run yet")
        files = (await c.get(f"/fd/rooms/{room['id']}/files", headers=ALICE_H)).json()["files"]
        assert {f["state"] for f in files} == {"new"}
        assert (await c.post(f"/fd/rooms/{room['id']}/stop", headers=ALICE_H)).json()["stopping"] is False


@pytest.mark.usefixtures("deck")
async def test_files_show_rows_and_sheets_right_after_the_upload():
    async with _client() as c:
        room = await _new_room(c)
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H,
                    content=_zip({"Plaće/place.xlsx": _xlsx(), "Plaće/popis.csv": "ime\nAna\nIvo\n", "a.txt": "x"}))
        files = {f["path"]: f for f in (await c.get(f"/fd/rooms/{room['id']}/files", headers=ALICE_H)).json()["files"]}
        assert files["Plaće/place.xlsx"]["info"] == {"rows": 42, "sheets": 2}
        assert files["Plaće/popis.csv"]["info"] == {"rows": 3}
        assert files["a.txt"]["info"] == {} and files["a.txt"]["size"] == 1


@pytest.mark.usefixtures("deck")
async def test_the_sweep_and_the_check_tell_how_far_they_are(monkeypatch):
    """The final sweep and the leftover check say which file of how many they
    are on, its size and the step it is at — and tidy it all away when done."""
    from captain_claw.flight_deck import rooms_job
    from test_flight_deck.test_rooms import _wait

    seen: list[dict] = []
    real_set_job = rooms_job._set_job

    async def spy(room_id, job):
        seen.append(dict(job))
        await real_set_job(room_id, job)

    monkeypatch.setattr(rooms_job, "_set_job", spy)
    letter = "Ime i prezime: Ana Kovač\nE-mail: ana.kovac@primjer-firma.hr\n"
    async with _client() as c:
        room = await _new_room(c)
        await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip({
            "Klijenti/a-sazetak.txt": "Sastanak s Ana Kovač prošli tjedan.",  # before the file that names her
            "Klijenti/ugovor.txt": letter}))
        await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
        done = await _wait(c, room["id"])

    sweep = [j for j in seen if j.get("phase") == "sweeping" and j.get("current")]
    assert sweep and {j["file_count"] for j in sweep} == {2} and {j["file_index"] for j in sweep} == {1, 2}
    assert all(j["current_size"] > 0 and j["bytes_total"] >= j["current_size"] and j["sweep_rounds"] == 3
               for j in sweep)
    assert {"reading", "replacing", "writing"} <= {j.get("step") for j in sweep}
    assert any(j.get("round_found", 0) >= 1 and j.get("round_changed") == 1 for j in sweep)  # the summary, fixed

    check = [j for j in seen if j.get("phase") == "checking" and j.get("current")]
    assert {j["file_index"] for j in check} == {1, 2} and all(j["needles"] >= 1 for j in check)
    assert {"reading", "text", "leaks", "indexing"} <= {j.get("step") for j in check}
    assert any(j.get("step") == "leaks" and j.get("chars", 0) > 0 for j in check)
    assert any(j.get("step") == "indexing" and j.get("pages", 0) >= 1 for j in check)

    job = done["job"]
    assert job["phase"] == "done" and job["blocked"] == 0 and job["sweep_changed"] >= 1
    assert not {"current", "step", "file_index", "file_count", "needles", "chars", "pages", "round_found",
                "phase_started_at"} & set(job)


def test_the_leak_check_finds_the_same_with_its_word_filter_and_fast():
    """Values whose words aren't all in the text are skipped without a search:
    the same answers as searching for each one, in a fraction of the time."""
    import random
    import re
    import time

    from captain_claw.anonymize import engine
    from captain_claw.flight_deck import rooms_job

    def by_search(text, needle_list):  # the check before the filter
        hay = rooms_job._fold_loose(text)
        hits = []
        for token, needle in needle_list:
            if token in hits:
                continue
            if isinstance(needle, rooms_job.Loose):
                found = needle.pattern.search(hay)
            else:
                found = needle in hay and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay)
            if found:
                hits.append(token)
        return hits

    rnd = random.Random(7)
    syl = ["ka", "ro", "vi", "ne", "ma", "lu", "to", "sa", "ri", "go", "če", "ša", "đu"]

    def word(n):
        return "".join(rnd.choice(syl) for _ in range(n)).capitalize()

    entities = []
    for i in range(1200):
        first, last = word(2), word(3)
        entities.append(engine.Entity(f"⟦PERSON_{i}⟧", "PERSON", f"{first} {last}".casefold(), f"{first} {last}",
                                      forms=[f"{first}a {last}a"]))
    for i in range(400):
        name = f"{word(2)} {word(3)} d.o.o."
        entities.append(engine.Entity(f"⟦ORG_{i}⟧", "ORG", name.casefold(), name))
    needle_list = rooms_job.needles(engine.VaultState(entities))
    rows = [f"| {r} | ⟦PERSON_{r % 900}⟧ račun {word(2)} | {r * 7},00 |" for r in range(15000)]
    leftovers = [entities[5].original.upper(), entities[40].forms[0],
                 "Text in the picture: " + entities[1300].original.replace(" ", " prc ", 1).replace(" d.o.o.", ""),
                 "".join(entities[77].original.split())]  # glued together: not a whole word
    text = "\n".join(rows + leftovers)

    t0 = time.perf_counter()
    fast = rooms_job.leaks(text, needle_list)
    took = time.perf_counter() - t0
    assert fast == by_search(text, needle_list)
    assert set(fast) >= {"⟦PERSON_5⟧", "⟦PERSON_40⟧", "⟦ORG_100⟧"} and "⟦PERSON_77⟧" not in fast
    assert took < 1.5, took
