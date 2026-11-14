"""What each file of a data room is about: the trusted model's one line,
paragraph, summary, keywords and topics after a run, in tokens and checked
for real values; describing again; and the connector's tools that find and
summarize files with them. Made-up data."""

from __future__ import annotations

import asyncio
import io
import re
import zipfile

import pytest

from captain_claw.flight_deck import anonymize_trusted, rooms_describe, rooms_store
from test_flight_deck.test_rooms import (  # noqa: F401 — the deck fixture
    ALICE,
    ALICE_H,
    _files,
    _wait,
    _zip,
    deck,
)
from test_flight_deck.test_rooms_alike import _vault_has
from test_flight_deck.test_rooms_mcp import _client, _rooms_token, _tool

rd = rooms_describe


# ── reading the model's answer ──────────────────────────────────────


def test_the_answer_is_read_in_sections_or_as_json():
    got = rd.parse("<think>hmm</think>\n**ONE LINE:** A lease for ⟦ORG_2⟧'s office.\n"
                   "PARAGRAPH: The lease runs\nfrom 2024 to 2029.\n"
                   "SUMMARY:\nFirst part.\n\n\n Second part.\n"
                   "KEYWORDS: lease; ⟦ORG_2⟧; office rent; Lease\nTOPICS:\n- real estate\n- leases")
    assert got == {"one": "A lease for ⟦ORG_2⟧'s office.", "paragraph": "The lease runs from 2024 to 2029.",
                   "summary": "First part.\n\nSecond part.", "keywords": ["lease", "⟦ORG_2⟧", "office rent"],
                   "topics": ["real estate", "leases"]}
    js = rd.parse('Here: {"one_line": "Payroll, March", "summary": ["a", "b"], "keywords": ["pay", "staff"]}')
    assert js and js["one"] == "Payroll, March" and js["summary"] == "a\n\nb" and js["keywords"] == ["pay", "staff"]
    assert rd.parse("I can't describe this file.") is None
    many = rd.parse("ONE LINE: x\nKEYWORDS: " + "; ".join(f"word{i}" for i in range(15)))
    assert many and len(many["keywords"]) == rd.MAX_TERMS


def test_tokens_are_written_as_the_room_writes_them_and_strangers_go():
    about = {"one": "Contract of [ORG_1] with PERSON_2 and ⟦PERSON_9⟧.", "paragraph": "", "summary": "",
             "keywords": ["⟦PERSON_9⟧", "[[ORG_1]]", "HTTP_200"], "topics": ["sales"]}
    out, dropped = rd.tidy_tokens(about, {"⟦ORG_1⟧", "⟦PERSON_2⟧"})
    assert out["one"] == "Contract of ⟦ORG_1⟧ with ⟦PERSON_2⟧ and [?]."
    assert out["keywords"] == ["⟦ORG_1⟧", "HTTP_200"] and dropped == 2  # ⟦PERSON_9⟧ wasn't in what it read


def test_a_long_file_is_shown_by_parts_spread_over_it():
    short = "Kratki tekst."
    assert rd.excerpt(short) == (short, False)
    sheets = "\n\n".join(f"## Sheet S{i}\n\n| a | b |\n" + "| 1 | 2 |\n" * 400 for i in range(60))
    body, cut = rd.excerpt(sheets, 8000)
    assert cut and len(body) < 9500 and "## Sheet S0" in body and "## Sheet S59" in body and "more left out" in body
    plain = "početak " * 1000 + "SREDINA " * 300 + "kraj " * 1600
    body, cut = rd.excerpt(plain, 6000)
    assert cut and body.startswith("početak") and "SREDINA" in body and body.rstrip().endswith("kraj")


def test_finding_files_takes_any_word_and_forgives_endings():
    assert rooms_store.find_match("the employment contracts of ⟦PERSON_3⟧") == \
        '"employme"* OR "contrac"* OR "⟦PERSON_3⟧"'
    assert rooms_store.find_match("plaće 2024") == '"plać"* OR "2024"'
    assert rooms_store.find_match("of the") == ""


# ── a room's files described ────────────────────────────────────────

FILES = {
    "Pravno/ugovor.txt": "Ugovor o kupoprodaji udjela. Prodavatelj: Ana Kovač. Kupac: Ivan Horvat. "
                         "Cijena: 42 milijuna EUR, plativo do 30. lipnja 2025.\n",
    "Financije/place.txt": "Isplata plaća za ožujak 2025. Radnik Ivan Horvat, neto 1.200 EUR.\n",
    "Financije/prazno.txt": "\n",
}


def _answer(user: str, *, real: str = "") -> str:
    path = re.search(r"^File: (.+)$", user, re.M).group(1)
    toks = list(dict.fromkeys(re.findall(r"⟦[A-Z_]+_\d+(?::\w+)?⟧", user.split("---", 1)[1])))
    who = toks[0] if toks else "nobody"
    if "ugovor" in path:
        return (f"ONE LINE: A share sale contract: {who} sells to {toks[-1]} for EUR 42 million{real}.\n"
                f"PARAGRAPH: {who} sells its shares to {toks[-1]}. The price is EUR 42 million.\n"
                "SUMMARY:\nThe seller and the buyer agree a sale.\n\nPayment is due by 30 June 2025.\n\nNo more terms.\n"
                f"KEYWORDS: share sale; purchase price; {who}; [PERSON_99]\n"
                "TOPICS: share purchase; pricing")
    return (f"ONE LINE: Payroll for March 2025, one worker ({who}).\n"
            "PARAGRAPH: Net pay of EUR 1,200 for March 2025.\nSUMMARY:\nOne payment.\n"
            "KEYWORDS: payroll; net pay; March 2025\nTOPICS: payroll; employment")


def _stub(monkeypatch, *, real: str = "") -> list[str]:
    """A trusted model: no names of its own when anonymizing, and a description
    when asked for one. Returns the files it was asked to describe."""
    model = anonymize_trusted.TrustedModel(provider="ollama", model="stand-in", basis="local")
    asked: list[str] = []

    async def usable(_db):
        return model

    async def complete(system, user):
        if system != rd.SYSTEM:
            return '{"spans": []}', 1
        asked.append(re.search(r"^File: (.+)$", user, re.M).group(1))
        return _answer(user, real=real), 7

    monkeypatch.setattr(anonymize_trusted, "usable", usable)
    monkeypatch.setattr(anonymize_trusted, "completer", lambda _m: complete)
    return asked


async def _described(c, rid: str) -> dict:
    for _ in range(400):
        room = (await c.get(f"/fd/rooms/{rid}", headers=ALICE_H)).json()
        n = room["counts"]
        if not room["job"]["running"] and room["state"] != "running" and room["describing"] is None \
                and n["describe_waiting"] == 0 and n["described"] + n["describe_failed"] >= n["done"]:
            return room
        await asyncio.sleep(0.05)
    raise AssertionError("describing never finished")


async def _room(c, files=FILES, names=("Ana Kovač", "Ivan Horvat")) -> dict:
    room = (await c.post("/fd/rooms", headers=ALICE_H, json={"codename": "Project Heron"})).json()
    await _vault_has(room, [("PERSON", n) for n in names])
    await c.put(f"/fd/rooms/{room['id']}/zip", headers=ALICE_H, content=_zip(files))
    await c.post(f"/fd/rooms/{room['id']}/run", headers=ALICE_H, json={})
    await _wait(c, room["id"])
    return room


@pytest.mark.usefixtures("deck")
async def test_a_run_describes_its_files_and_claude_finds_them(monkeypatch):
    asked = _stub(monkeypatch)
    async with _client() as c:
        room = await _room(c)
        rid = room["id"]
        done = await _described(c, rid)
        assert done["counts"]["described"] == 3 and done["counts"]["describe_needed"] == 0
        assert sorted(asked) == sorted(f["clean_path"] for f in (await _files(c, rid)).values()
                                       if "prazno" not in f["path"])  # an empty file isn't sent
        files = await _files(c, rid)
        contract = files["Pravno/ugovor.txt"]
        assert contract["about"]["one"].startswith("A share sale contract: ⟦PERSON_")
        assert files["Financije/prazno.txt"]["about"]["one"] == rd.EMPTY_ONE

        # The owner's page: in tokens as Claude reads it, and with the names put back.
        text = (await c.get(f"/fd/rooms/{rid}/files/{contract['id']}/text", headers=ALICE_H)).json()
        about = text["about"]
        assert "Ana Kovač" not in str({k: v for k, v in about.items() if k != "real"})
        assert "Ana Kovač" in about["real"]["one"] and "Ivan Horvat" in about["real"]["paragraph"]
        assert about["keywords"][-1].startswith("⟦PERSON_") and "[PERSON_99]" not in about["keywords"]
        assert about["dropped"] == 1 and not about["stale"]  # [PERSON_99]: no such party in what it read

        # Claude, through the connector: one line in the list, files found by what they're about.
        await c.post(f"/fd/rooms/{rid}/release", headers=ALICE_H)
        await rooms_store.store().set_grant(ALICE, "cc_oc_test", [rid], False)
        t = _rooms_token()
        listing = await _tool(c, t, "room_list_files", room=rid)
        assert {f["path"]: f.get("about", "") for f in listing["files"]}[contract["clean_path"]] == about["one"]
        found = await _tool(c, t, "room_find_files", room=rid, query="salaries payroll")
        assert found["files"][0]["path"] == files["Financije/place.txt"]["clean_path"]
        assert found["files"][0]["matched_in"] == "description" and found["files"][0]["topics"] == ["payroll",
                                                                                                    "employment"]
        found = await _tool(c, t, "room_find_files", room=rid, query="lipnja")  # only in the text
        assert [f["matched_in"] for f in found["files"]] == ["text"]
        sums = await _tool(c, t, "room_file_summaries", room=rid, paths=[contract["clean_path"], "nope.txt"])
        assert sums["files"][0]["summary"].count("\n\n") == 2 and "error" in sums["files"][1]
        overview = await _tool(c, t, "room_overview", room=rid)
        assert overview["described_files"] == 3 and overview["topics"]["payroll"] == 1
        reads = (await c.get(f"/fd/rooms/{rid}/reads", headers=ALICE_H)).json()
        assert {"room_find_files", "room_file_summaries"} <= {r["tool"] for r in reads["reads"]}
        assert not reads["alarms"]

        # A ZIP for a Claude Project says the same in FILES.md.
        info = (await c.post(f"/fd/rooms/{rid}/export", headers=ALICE_H, json={})).json()
        blob = (await c.get(f"/fd/rooms/{rid}/export/{info['token']}", headers=ALICE_H)).content
        index = next(zipfile.ZipFile(io.BytesIO(blob)).read(n).decode() for n in zipfile.ZipFile(io.BytesIO(blob)).namelist()
                     if n.endswith("/FILES.md"))
        assert f"## {contract['clean_path']}" in index and about["one"] in index and "Topics: payroll" in index
        assert "Ana Kovač" not in index

        # Again, for every file — and for one.
        asked.clear()
        r = await c.post(f"/fd/rooms/{rid}/describe", headers=ALICE_H, json={"again": True})
        assert r.status_code == 200 and r.json()["waiting"] == 3
        await _described(c, rid)
        assert len(asked) == 2
        asked.clear()
        r = await c.post(f"/fd/rooms/{rid}/describe", headers=ALICE_H, json={"files": [contract["id"]], "again": True})
        await _described(c, rid)
        assert asked == [contract["clean_path"]]
        assert (await c.post(f"/fd/rooms/{rid}/describe", headers=ALICE_H, json={})).json()["waiting"] == 0


@pytest.mark.usefixtures("deck")
async def test_a_description_with_a_real_name_is_held_back(monkeypatch):
    _stub(monkeypatch, real=" (Ana Kovač)")
    async with _client() as c:
        room = await _room(c)
        rid = room["id"]
        done = await _described(c, rid)
        assert done["counts"]["describe_failed"] == 1
        files = await _files(c, rid)
        contract = files["Pravno/ugovor.txt"]
        assert contract["about"]["state"] == "held" and contract["about"]["one"] == ""
        await c.post(f"/fd/rooms/{rid}/release", headers=ALICE_H)
        await rooms_store.store().set_grant(ALICE, "cc_oc_test", [rid], False)
        t = _rooms_token()
        listing = await _tool(c, t, "room_list_files", room=rid)
        assert "Ana Kovač" not in str(listing)
        sums = await _tool(c, t, "room_file_summaries", room=rid, paths=[contract["clean_path"]])
        assert sums["files"][0]["note"].startswith("No description yet")


@pytest.mark.usefixtures("deck")
async def test_no_trusted_model_no_descriptions(monkeypatch):
    async with _client() as c:
        room = await _room(c)
        rid = room["id"]
        got = (await c.get(f"/fd/rooms/{rid}", headers=ALICE_H)).json()
        assert got["describing"] is None and got["counts"]["described"] == 0 and got["counts"]["describe_needed"] == 3
        r = await c.post(f"/fd/rooms/{rid}/describe", headers=ALICE_H, json={})
        assert r.status_code == 409 and "trusted model" in r.json()["detail"]


@pytest.mark.usefixtures("deck")
async def test_same_party_rewrites_a_description_in_place(monkeypatch):
    asked = _stub(monkeypatch)
    async with _client() as c:
        room = await _room(c, {"Pravno/ugovor.txt": FILES["Pravno/ugovor.txt"],
                               "Financije/place.txt": "Isplata plaća za ožujak 2025. Radnik Ivan Hrovat, neto 1.200 EUR.\n"},
                         names=("Ana Kovač", "Ivan Horvat", "Ivan Hrovat"))
        rid = room["id"]
        await _described(c, rid)
        files = await _files(c, rid)
        pay = files["Financije/place.txt"]
        typo = re.search(r"⟦PERSON_\d+⟧", pay["about"]["one"]).group(0)
        right = re.findall(r"⟦PERSON_\d+⟧", files["Pravno/ugovor.txt"]["about"]["one"])[-1]
        assert typo != right
        asked.clear()
        r = await c.post(f"/fd/rooms/{rid}/same", headers=ALICE_H, json={"keep": right, "other": typo})
        assert r.status_code == 200, r.text
        pay = (await _files(c, rid))["Financije/place.txt"]
        assert typo not in pay["about"]["one"] and right[:-1] in pay["about"]["one"]
        assert not pay["about"]["stale"]
        room = (await c.get(f"/fd/rooms/{rid}", headers=ALICE_H)).json()
        assert room["counts"]["describe_needed"] == 0 and asked == []  # rewritten, not described again
