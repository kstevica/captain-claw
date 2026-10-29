"""Section maps of an agent's knowledge — token-free: the sections come from
the headings (a PDF's pages grouped), each gets a gist from its own text, and
the map says the file and lines that hold it — lines that really do start
at the section's heading in the files written. A PDF made into Markdown is
tidied: chapters named by their number and title, their bold subheadings
found, page heads told apart; read tracking keeps the headings as written."""

from __future__ import annotations

import re

import numpy as np

from captain_claw.flight_deck import agent_preset as ap
from captain_claw.flight_deck import agent_preset_docs as pdocs
from captain_claw.flight_deck import agent_preset_outline as ol


def _offline(texts):
    raise RuntimeError("no local model here")


def _doc(chapters: int = 6, subs: int = 4, para: int = 3) -> str:
    out = []
    for c in range(1, chapters + 1):
        out.append(f"# Chapter {c}: Valuation part {c}\n\nThis chapter covers valuation topic {c} in depth. "
                   f"It explains why the discount rate matters for topic {c}.\n")
        for s in range(1, subs + 1):
            body = "\n\n".join(f"Section {c}.{s} paragraph {p} explains cash flows, growth and the cost of "
                               f"capital for case {c}.{s}. The terminal value follows from stable growth here."
                               for p in range(para))
            out.append(f"## {c}.{s} Topic {c}.{s}\n\n{body}\n\n```python\n# a comment, not a heading\nx = {s}\n```\n")
    return "\n".join(out)


def test_sections_come_from_headings_and_never_from_code():
    text = _doc(chapters=2, subs=2)
    secs = ol.outline(text)
    assert [s["title"] for s in secs] == ["Chapter 1: Valuation part 1", "1.1 Topic 1.1", "1.2 Topic 1.2",
                                          "Chapter 2: Valuation part 2", "2.1 Topic 2.1", "2.2 Topic 2.2"]
    assert [s["level"] for s in secs] == [1, 2, 2, 1, 2, 2]
    ch1 = secs[0]
    assert text[ch1["start"]:].startswith("# Chapter 1") and ch1["end"] == secs[3]["start"]   # its subsections in it
    assert ch1["own"] == secs[1]["start"]                                                      # its own text stops there
    assert ol.outline("Just prose, no headings at all.") == []


def test_levels_are_kept_while_they_fit():
    secs = ol.outline(_doc(chapters=10, subs=10), max_sections=50)
    assert {s["level"] for s in secs} == {1} and len(secs) == 10      # 10 + 100 wouldn't fit: chapters only
    assert len(ol.outline(_doc(chapters=10, subs=10), max_sections=200)) == 110


def test_a_pdfs_pages_are_grouped():
    text = "# report.pdf\n\n" + "".join(f"## Page {n}\n\nText of page {n}.\n\n" for n in range(1, 26))
    titles = [s["title"] for s in ol.outline(text)]
    assert titles == ["Pages 1–10", "Pages 11–20", "Pages 21–25"]


def test_a_gist_is_the_sections_most_central_sentence(monkeypatch):
    text = ("# Rates\n\nBananas are yellow and sweet in summer.\n"
            "The discount rate reflects the risk of the cash flows.\n"
            "A higher discount rate lowers the value of the cash flows.\n"
            "Risk in the cash flows raises the discount rate.\n")
    secs = ol.outline(text)

    def fake(texts):   # a word-overlap embedding: "discount rate" sentences agree, bananas don't
        vocab = ["discount", "rate", "cash", "flows", "risk", "bananas", "yellow"]
        v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype="float32") + 1e-6
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(ol, "_encode", fake)
    assert "discount rate" in ol.add_gists(text, secs)[0]["gist"]

    monkeypatch.setattr(ol, "_encode", _offline)
    assert ol.add_gists(text, ol.outline(text))[0]["gist"].startswith("Bananas")       # the first, then


def _entries(map_text: str) -> list[tuple[str, str, int, int]]:
    """(file, title, first line, last line) of every entry in a map."""
    out, file = [], ""
    for line in map_text.splitlines():
        if line.startswith("## "):
            file = line[3:].split(" · ")[0]
        m = re.match(r"\s*- \*\*(.+?)\*\* — lines (\d+)–(\d+)", line)
        if m:
            out.append((file, m[1], int(m[2]), int(m[3])))
    return out


def test_the_map_points_at_the_lines_that_hold_each_section(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "_PART_BYTES", 6_000)          # many parts
    monkeypatch.setattr(ap, "_MAP_MIN_CHARS", 1_000)
    monkeypatch.setattr(ol, "_encode", _offline)
    text = _doc(chapters=6, subs=4)
    preset = ap.normalise({"knowledge": [{"doc": "manual.md", "title": "Valuation manual", "summary": "How to value."}]})
    written = ap.write_knowledge(tmp_path, preset, [{"name": "manual.md", "text": text, "description": "",
                                                     "usage": "reference"}])
    (w,) = written
    assert w["parts"] >= 3 and w["map"] == "manual.map.md" and w["sections"] == 30
    assert "manual.map.md" in w["files"]
    entries = _entries((tmp_path / "manual.map.md").read_text())
    assert len(entries) == 30
    for file, title, first, last in entries:
        lines = (tmp_path / file).read_text().splitlines()
        assert lines[first - 1].lstrip("#").strip() == title, (file, title, first)
        assert first <= last <= len(lines)
    assert "section map: manual.map.md" in (tmp_path / "INDEX.md").read_text()
    assert "manual.map.md" in (tmp_path / "manual.md").read_text()                 # the overview points at it


def test_a_short_document_gets_no_map(tmp_path):
    preset = ap.normalise({"knowledge": [{"doc": "note.md", "title": "Note", "summary": ""}]})
    (w,) = ap.write_knowledge(tmp_path, preset, [{"name": "note.md", "text": "# A\n\nx\n\n# B\n\ny\n\n# C\n\nz\n",
                                                  "description": "", "usage": "auto"}])
    assert w["map"] == "" and w["files"] == ["note.md"] and not (tmp_path / "note.map.md").exists()


def test_a_big_map_drops_gists_then_levels(monkeypatch):
    text = _doc(chapters=20, subs=12)
    secs = ol.outline(text)
    for s in secs:
        s["gist"] = "g" * ol.GIST_CHARS
    files = [{"name": "big.md", "label": "", "start": 0, "raw": text, "body": text, "prefix": "# Big\n\n---\n\n"}]
    full = ol.map_md("Big", "big.md", files, secs)
    monkeypatch.setattr(ol, "MAX_MAP_BYTES", len(full.encode()) // 2)
    small = ol.map_md("Big", "big.md", files, secs)
    assert len(small.encode()) <= ol.MAX_MAP_BYTES
    assert "**1.1 Topic 1.1**" in small and "· ggg" in small          # level 2 kept, only top gists left
    monkeypatch.setattr(ol, "MAX_MAP_BYTES", 4_000)
    tiny = ol.map_md("Big", "big.md", files, secs)
    assert "**1.1 Topic 1.1**" not in tiny and "**Chapter 1: Valuation part 1**" in tiny


def test_the_agent_is_told_about_maps_and_a_setup_reads_back_with_them(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "_MAP_MIN_CHARS", 1_000)
    monkeypatch.setattr(ol, "_encode", _offline)
    preset = ap.normalise({"knowledge": [{"doc": "manual.md", "title": "Manual", "summary": "How to value."}]})
    written = ap.write_knowledge(tmp_path / "knowledge", preset,
                                 [{"name": "manual.md", "text": _doc(), "description": "", "usage": "auto"}])
    block = ap.rules_block(preset, written, [])
    assert "manual.map.md" not in block and "section map (<name>.map.md)" in block
    back = ap._record_from_files(tmp_path / "skills", tmp_path / "knowledge")
    (k,) = back["knowledge"]
    assert k["summary"] == "How to value." and k["map"] == "manual.map.md" and "manual.map.md" in k["files"]


def test_staging_counts_the_sections(tmp_path):
    meta = pdocs.stage(tmp_path, "manual.md", _doc(chapters=3, subs=2), "upload")
    assert meta["sections"] == 9


async def test_a_running_agent_is_asked_to_index_its_knowledge(monkeypatch):
    import httpx
    seen: list[httpx.Request] = []

    async def ready(port_of, token, timeout):
        return port_of()

    monkeypatch.setattr(ap, "_wait_ready", ready)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(
        transport=httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(200, json={"ok": True}))[1]), **kw))
    assert await ap.ask_to_index("helpdesk", lambda: 24555, "tok") is True
    (r,) = seen
    assert r.method == "POST" and r.url.path == "/api/semantic-memory/sync" and r.url.port == 24555
    assert r.url.params["token"] == "tok"

    async def never(port_of, token, timeout):
        return 0

    monkeypatch.setattr(ap, "_wait_ready", never)
    assert await ap.ask_to_index("helpdesk", lambda: 24555, "tok") is False and len(seen) == 1


async def test_the_agents_sync_route_indexes_in_the_background():
    import json
    import types

    from captain_claw.web import rest_semantic_memory as rsm
    reasons: list[str] = []
    semantic = types.SimpleNamespace(schedule_sync=reasons.append)
    server = types.SimpleNamespace(agent=types.SimpleNamespace(memory=types.SimpleNamespace(semantic=semantic)))
    resp = await rsm.sync(server, None)
    assert resp.status == 200 and json.loads(resp.text) == {"ok": True} and reasons == ["requested"]
    off = types.SimpleNamespace(agent=types.SimpleNamespace(memory=types.SimpleNamespace(semantic=None)))
    assert (await rsm.sync(off, None)).status == 503


# ── A PDF made into Markdown ──────────────────────────────────────────

_PROSE = ("Divesting a business unit creates value when other owners believe the unit is worth more to them, "
          "and the parent can use the proceeds well.")


def _converted_chapter(n: int) -> str:
    """A book chapter as a PDF's conversion leaves it: a number's heading and
    a title's, then subheadings only in bold, page heads, an exhibit's labels."""
    body = "\n\n".join(_PROSE for _ in range(12))
    return "\n".join([
        f"# **{n}**", "", f"##### Creating Value {n}", "", body, "",
        f"**{470 + n}** **CREATING VALUE THROUGH DIVESTITURES**", "",
        "**VALUE CREATION FROM DIVESTITURES**", "", body, "",
        "**Evidence for Value from Divestitures**", "", body, "",
        "**Use of proceeds**", "", "**Debt**", "", "2,126", "",
        "**Why Executives Shy Away**", "", body, "",
        "**SUMMARY**", "", body, ""])


def test_a_converted_books_chapters_are_named_and_split_by_their_bold_subheadings(monkeypatch):
    monkeypatch.setattr(ol, "LONG_STRETCH", 2_000)
    text = _converted_chapter(22) + "\n" + _converted_chapter(23)
    secs = ol.outline(text)
    assert [(s["level"], s["title"]) for s in secs[:5]] == [
        (1, "22 Creating Value 22"), (2, "VALUE CREATION FROM DIVESTITURES"),
        (3, "Evidence for Value from Divestitures"), (3, "Why Executives Shy Away"), (2, "SUMMARY")]
    assert [s["title"] for s in secs].count("SUMMARY") == 2 and len(secs) == 10
    assert not any(t in s["title"] for s in secs for t in ("Use of proceeds", "Debt", "472", "CREATING VALUE THROUGH"))
    assert secs[0]["end"] == secs[5]["start"] and text[secs[0]["start"]:].startswith("# **22**")
    # Read tracking keeps the headings as written: the keys a setup recorded still match.
    raw = ol.outline(text, tidy=False)
    assert [s["title"] for s in raw] == ["**22**", "Creating Value 22", "**23**", "Creating Value 23"]
    from captain_claw.flight_deck import agent_preset_select as psel
    assert psel.spans(text)[1]["key"] == "**22** › Creating Value 22"


def test_a_short_stretch_and_a_documents_own_headings_are_left_as_they_are():
    text = "# Guide\n\n**Note**\n\n" + _PROSE + "\n\n## 1.1 Returns\n\n" + _PROSE + "\n"
    assert [s["title"] for s in ol.outline(text)] == ["Guide", "1.1 Returns"]
    assert [s["title"] for s in ol.outline("# **Pricing** `v2`\n\ntext\n")] == ["Pricing v2"]


def test_page_heads_and_numbers_are_told_from_text():
    for line in ("**470** **CREATING VALUE THROUGH DIVESTITURES**", "**VALUE CREATION FROM DIVESTITURES** **471**",
                 "**469**", "478** **CREATING VALUE THROUGH DIVESTITURES**", "**12** **Valuation of Companies Today**",
                 "**xvi** **ACKNOWLEDGMENTS**", "**PREFACE** **xvii**"):
        assert ol.page_furniture(line), line
    for line in ("**Mergers and acquisitions**", "**2004** revenue was up", "**Total** **125**", "2,126",
                 "Plain prose 12", "**Exhibit 22.1** Divestitures by type", "**Tim** **Koller** is a partner in"):
        assert not ol.page_furniture(line), line


def test_a_sections_path_is_named_as_the_outline_names_it():
    assert ol.tidy_path(["**22**"], "Creating Value") == ([], "22 Creating Value")
    assert ol.tidy_path(["**VALUATION**", "**B**"], "Economic Profit") == (["VALUATION"], "B Economic Profit")
    assert ol.tidy_path(["Guide"], "**Returns**") == (["Guide"], "Returns")
