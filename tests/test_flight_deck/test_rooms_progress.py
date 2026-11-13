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
        text = "\n\n".join(f"Stavka {i}: Sastanak je održan u utorak prema dogovoru." for i in range(6)).encode()
        await ar.anonymize_blob("user-progress", "memo.txt", text, {"mode": "one_way", "lane": "both"},
                                actor="test", source="test")
    finally:
        ar.PROGRESS.reset(token)
    steps = [e["step"] for e in events]
    assert steps[0] == "reading" and steps[-1] == "replacing" and "model" in steps
    chunks = [e for e in events if e["step"] == "model"]
    assert chunks[0]["chunk"] == 0 and chunks[-1]["chunk"] == chunks[-1]["chunks"] >= 2


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
