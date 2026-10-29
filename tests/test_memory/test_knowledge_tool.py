"""The knowledge tool — finding things in the documents an owner's setup gave
the agent (knowledge/, as agent_preset.write_knowledge writes it), with no
model:

* search reaches the knowledge alone (not other workspace files, not the
  section maps), and each hit is a passage with its document, section path
  and lines, lines that hold it in the file named;
* an exact code comes first; a quoted phrase is required; ``doc`` keeps to
  one document; without the index the files are read instead;
* a section is read whole by its title, across the parts it spans, a long
  one in pages; outline and list say what there is;
* the tool shows only when there are documents, and a shared agent's
  member gets nothing;
* a book made from a PDF is read by its subsections, a title its chapters
  share by its path, without the PDF's page heads; the owner's rules and
  the glossary give the items that match; a document is found however its
  name is written; a quoted phrase nowhere gives the closest, said so;
* before a turn, the sections a message bears on are named for the agent
  to look up (``pointers``).
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from captain_claw import speaker
from captain_claw.flight_deck import agent_preset as ap
from captain_claw.semantic_memory import (
    SemanticMemoryIndex,
    _EmbeddingProviderChain,
    _LocalHashEmbeddingProvider,
)
from captain_claw.tools.knowledge import KnowledgeTool
from captain_claw.tools.registry import ToolRegistry


def _manual() -> str:
    out = ["# Valuation manual\n\nHow we value companies, chapter by chapter.\n"]
    for c in range(1, 7):
        out.append(f"## Chapter {c}: Topic {c}\n\nChapter {c} opens with an overview of topic {c}.\n")
        for s in range(1, 4):
            body = "\n\n".join(f"Paragraph {p} of section {c}.{s} on cash flows and growth for case {c}.{s}."
                               for p in range(12))
            out.append(f"### {c}.{s} Part {c}.{s}\n\n{body}\n")
    out.append("### 6.4 Cost of equity for banks\n\nThe cost of equity for banks uses a beta of regulated peers. "
               "Return code RMA-2291 marks a refund that needs approval.\n")
    return "\n".join(out)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace as a setup leaves it: a long manual in parts with a map, a
    short FAQ, and a note elsewhere that says the same words."""
    monkeypatch.setattr(ap, "_PART_BYTES", 6_000)
    monkeypatch.setattr(ap, "_MAP_MIN_CHARS", 1_000)
    ws = tmp_path / "workspace"
    preset = ap.normalise({"knowledge": [
        {"doc": "manual.md", "title": "Valuation manual", "summary": "How we value companies."},
        {"doc": "faq.md", "title": "Client FAQ", "summary": "Questions clients ask."}]})
    ap.write_knowledge(ws / "knowledge", preset, [
        {"name": "manual.md", "text": _manual(), "description": "", "usage": "reference"},
        {"name": "faq.md", "text": "# FAQ\n\nRefunds take 14 days. A refund with code RMA-1000 is automatic.\n",
         "description": "", "usage": "auto"}])
    (ws / "notes").mkdir()
    (ws / "notes" / "scratch.md").write_text("The cost of equity for banks, my own scratch note.\n")
    return ws


def _index(tmp_path: Path, ws: Path) -> SemanticMemoryIndex:
    session_db = tmp_path / "sessions.db"
    session_db.touch()
    index = SemanticMemoryIndex(db_path=tmp_path / "memory.db", session_db_path=session_db, workspace_path=ws,
                                index_workspace=True, index_sessions=False, auto_sync_on_search=False,
                                layered_summaries=False,
                                embedding_chain=_EmbeddingProviderChain([_LocalHashEmbeddingProvider()]))
    index._sync_once()
    return index


def _tool(ws: Path, semantic=None) -> KnowledgeTool:
    tool = KnowledgeTool()
    tool._agent = types.SimpleNamespace(memory=types.SimpleNamespace(semantic=semantic), workspace_base_path=ws)
    return tool


async def test_search_finds_the_passage_and_says_where_it_is(tmp_path, workspace):
    index = _index(tmp_path, workspace)
    try:
        out = (await _tool(workspace, index).execute(action="search", query="cost of equity for banks")).content
        assert out.startswith("1 passage" ) or "passages for 'cost of equity for banks'" in out
        first = out.split("\n[1] ", 1)[1]
        assert first.startswith("Valuation manual › Valuation manual › Chapter 6: Topic 6 › 6.4 Cost of equity for banks")
        file, lines = first.split("\n")[1].strip().split(", lines ")
        a, b = (int(x) for x in lines.split("–"))
        held = (workspace / file).read_text().splitlines()[a - 1:b]
        assert any("beta of regulated peers" in line for line in held)
        assert "scratch note" not in out and ".map.md" not in out              # the knowledge alone
        assert "action=section, doc='Valuation manual', section='6.4 Cost of equity for banks'" in out
    finally:
        index.close()


async def test_an_exact_code_comes_first_and_a_quoted_phrase_is_required(tmp_path, workspace):
    index = _index(tmp_path, workspace)
    try:
        tool = _tool(workspace, index)
        out = (await tool.execute(action="search", query="refund code RMA-2291")).content
        assert "RMA-2291" in out.split("\n[2] ")[0]                          # the passage with the code is [1]
        out = (await tool.execute(action="search", query='"beta of regulated peers"')).content
        assert out.count("\n[") == 1 and "regulated peers" in out
        # Words in quotes the documents never use (a model's guess at them): the closest, said so.
        near = (await tool.execute(action="search", query='"cost of equity for insurers" banks')).content
        assert near.startswith("No passage has the exact words “cost of equity for insurers”; the closest are below.")
        assert "Cost of equity for banks" in near.split("\n[2] ")[0]
        faq = (await tool.execute(action="search", query="refund code", doc="Client FAQ")).content
        assert "(Client FAQ)" in faq and "Valuation manual ›" not in faq
    finally:
        index.close()


async def test_without_the_index_the_files_are_read(workspace):
    out = (await _tool(workspace).execute(action="search", query="RMA-2291 approval")).content
    assert "6.4 Cost of equity for banks" in out.split("\n[2] ")[0]
    assert "scratch note" not in out


async def test_a_section_is_read_whole_across_its_parts_and_in_pages(workspace, monkeypatch):
    tool = _tool(workspace)
    whole = _manual()
    want = whole[whole.index("## Chapter 3: Topic 3"):whole.index("## Chapter 4: Topic 4")].strip()
    parts = sorted(p.name for p in (workspace / "knowledge").glob("manual-part-*.md"))
    out = (await tool.execute(action="section", doc="manual", section="Chapter 3: Topic 3")).content
    head, text = out.split("\n\n", 1)
    assert head.startswith("Valuation manual › Valuation manual › Chapter 3: Topic 3 — knowledge/manual-part-")
    assert " ".join(text.split()) == " ".join(want.split())                   # all of it, though parts cut it
    assert len({p for p in parts if p in head}) == 1
    monkeypatch.setattr("captain_claw.tools.knowledge.SECTION_CHARS", 1500)
    first = (await tool.execute(action="section", doc="manual", section="chapter 3")).content
    assert "page 1 of " in first and first.endswith("(Continues: page=2.)")
    second = (await tool.execute(action="section", doc="manual", section="Chapter 3: Topic 3", page=2)).content
    assert "page 2 of " in second
    missing = (await tool.execute(action="section", doc="manual", section="Nothing like it")).content
    assert missing.startswith("No section of Valuation manual matches")
    # A part's own file, as a search hit names it, finds its document (an onboarding exam saw an agent try).
    by_part = (await tool.execute(action="section", doc=parts[-1], section="Chapter 3: Topic 3")).content
    assert by_part.startswith(head)


async def test_outline_and_list_say_what_there_is(workspace):
    tool = _tool(workspace)
    listed = (await tool.execute(action="list")).content
    assert "- Valuation manual — knowledge/manual.md (" in listed and "parts): How we value companies." in listed
    assert "- Client FAQ — knowledge/faq.md: Questions clients ask." in listed
    outline = (await tool.execute(action="outline", doc="Valuation manual", section="Chapter 2: Topic 2")).content
    assert outline.splitlines()[1].startswith("- Chapter 2: Topic 2 (manual-part-")
    assert "  - 2.3 Part 2.3 (" in outline and "Chapter 3" not in outline
    gone = await tool.execute(action="outline", doc="nope")
    assert not gone.success and "No document matches 'nope'" in gone.error


async def test_the_tool_shows_only_with_documents_and_never_to_a_member(tmp_path, workspace, monkeypatch):
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(_tool(workspace), metadata={"requires_knowledge": True})
    assert "knowledge" in registry.list_tools()
    empty = tmp_path / "empty"
    (empty / "knowledge").mkdir(parents=True)
    bare = ToolRegistry(base_path=tmp_path)
    bare.register(_tool(empty), metadata={"requires_knowledge": True})
    assert "knowledge" not in bare.list_tools()
    monkeypatch.setattr(speaker, "member_bound", lambda: True)
    refused = await _tool(workspace).execute(action="list")
    assert not refused.success and "aren't available" in refused.error


def test_the_setup_tells_the_agent_about_the_tool():
    preset = ap.normalise({"knowledge": [{"doc": "m.md", "title": "Manual", "summary": ""}]})
    block = ap.rules_block(preset, [{"title": "Manual", "map": "m.map.md"}], [])
    assert "the knowledge tool searches all of it" in block and "section map (<name>.map.md)" in block


# ── add: keeping a file the user gave ─────────────────────────────────


def _deck(monkeypatch, reply) -> list:
    """Run as Flight Deck's agent; Flight Deck answers with *reply(request)*. The requests sent."""
    import httpx

    monkeypatch.setenv("FD_URL", "http://fd.test")
    sent: list = []
    real = httpx.AsyncClient

    def handler(request):
        sent.append(request)
        return reply(request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return sent


async def test_add_sends_a_document_and_says_what_was_kept(tmp_path, monkeypatch):
    import httpx

    sent = _deck(monkeypatch, lambda r: httpx.Response(200, json={
        "doc": "prices.xlsx", "title": "Prices", "chars": 1234, "parts": 1, "sections": 0,
        "tables": ["prices_basic"], "replaced": False, "warnings": []}))
    doc = tmp_path / "prices-20261028-215500-a1b2c3.xlsx"            # as a chat upload is saved
    doc.write_bytes(b"PK fake workbook")
    tool = _tool(tmp_path)
    got = await tool.execute(action="add", path=str(doc), note="Our price list", use="reference")
    assert got.success, got.error
    assert "Kept “Prices” (prices.xlsx) in your knowledge" in got.content and "prices_basic" in got.content
    assert "isn't read for rules" in got.content
    (req,) = sent
    assert req.url.path == "/fd/agent-knowledge/agent/add"
    body = req.content
    assert b'name="file"; filename="prices.xlsx"' in body and b"PK fake workbook" in body
    sent.clear()
    await tool.execute(action="add", path=str(doc), name="price-list-2026.md")   # a name the agent chose
    assert b'filename="prices.xlsx"' in sent[0].content and b"price-list-2026.md" in sent[0].content
    assert b"Our price list" in body and b"reference" in body

    sent.clear()
    got = await tool.execute(action="add", text="Total 42 EUR", name="receipt.md")
    from urllib.parse import parse_qs

    form = parse_qs(sent[0].content.decode())
    assert got.success and form["text"] == ["Total 42 EUR"] and form["name"] == ["receipt.md"]


async def test_add_asks_for_a_pictures_text_and_turns_a_scan_into_pages(tmp_path, monkeypatch):
    import httpx
    import pypdf

    sent = _deck(monkeypatch, lambda r: httpx.Response(422, json={"detail": {"reason": "scan", "detail": "no text"}}))
    tool = _tool(tmp_path)
    shot = tmp_path / "screenshot.png"
    shot.write_bytes(b"\x89PNG")
    got = await tool.execute(action="add", path=str(shot))
    assert not got.success and "image_ocr" in got.error and 'name="screenshot.md"' in got.error and sent == []

    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    scan = tmp_path / "Old contract.pdf"
    with scan.open("wb") as fh:
        writer.write(fh)
    got = await tool.execute(action="add", path=str(scan), _runtime_base_path=str(tmp_path))
    assert not got.success and "is a scan" in got.error and 'name="Old-contract.md"' in got.error
    assert "scans/Old-contract/page-03.png" in got.error and (tmp_path / "scans/Old-contract/page-01.png").is_file()


async def test_add_needs_flight_deck_and_the_tool_shows_there(tmp_path, monkeypatch):
    monkeypatch.delenv("FD_URL", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    tool = _tool(empty)
    assert not tool.has_knowledge()                                     # nothing to search, no Flight Deck
    got = await tool.execute(action="add", text="x", name="x.md")
    assert not got.success and "Flight Deck" in got.error
    monkeypatch.setenv("FD_URL", "http://fd.test")
    assert tool.has_knowledge()                                         # under it: always — it takes files in


# ── A book from a PDF, the owner's lists, names written any way ───────

_PROSE = ("Divesting a business unit creates value when other owners believe the unit is worth more to them, "
          "and the parent can use the proceeds well.")


def _chapter(n: int) -> str:
    body = "\n\n".join(f"{_PROSE} Case {n}.{p}." for p in range(14))
    return "\n".join([f"# **{n}**", "", f"##### Creating Value {n}", "", body, "",
                      f"**{470 + n}** **CREATING VALUE THROUGH DIVESTITURES**", "",
                      "**Why Divestitures Create Value**", "", body, "",
                      "**SUMMARY**", "", f"Chapter {n} sums up: divest early, from strength, chapter {n}.", "",
                      body, ""])


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """knowledge/ with a converted book (chapters 22 and 23), the owner's
    rules and a glossary — as a setup writes them."""
    from captain_claw.flight_deck import agent_preset_outline as ol

    monkeypatch.setattr(ol, "LONG_STRETCH", 2_000)
    ws = tmp_path / "ws"
    k = ws / "knowledge"
    k.mkdir(parents=True)
    (k / "INDEX.md").write_text("# Knowledge\n\n- [Valuation](valuation-5ed-mckinsey.md) — the book\n")
    (k / "valuation-5ed-mckinsey.md").write_text("# Valuation\n\nThe book.\n\n---\n\n" + _chapter(22) + _chapter(23))
    rules = [f"- Always check item {n} of the closing list before filing report {n}." for n in range(1, 25)]
    rules[11] = "- Refunds are allowed within a refund window of 14 days of delivery."
    (k / "owner-rules.md").write_text("# Your owner's rules\n\n## Rules\n" + "\n".join(rules) + "\n")
    (k / "glossary.md").write_text("# Glossary\n\n" + "\n".join(
        f"- **Term {n}** — what term {n} means in the closing list." for n in range(1, 20)) + "\n")
    return ws


async def test_a_books_subsections_are_read_and_a_shared_title_by_its_path(library):
    tool = _tool(library)
    outline = (await tool.execute(action="outline", doc="valuation")).content
    assert "- 22 Creating Value 22 (" in outline and "  - Why Divestitures Create Value (" in outline
    first = (await tool.execute(action="section", doc="Valuation", section="SUMMARY")).content
    assert "Chapter 22 sums up" in first and "Chapter 23 sums up" not in first
    assert "1 other section is titled 'SUMMARY' too: 23 Creating Value 23 › SUMMARY." in first
    assert "CREATING VALUE THROUGH DIVESTITURES" not in first                   # a page head isn't text
    second = (await tool.execute(action="section", doc="Valuation", section="23 Creating Value 23 › SUMMARY")).content
    assert "Chapter 23 sums up" in second and "other section" not in second
    found = (await tool.execute(action="search", query="Chapter 23 sums up divest early")).content
    assert "section='23 Creating Value 23 › SUMMARY'" in found                    # the hint names it by its path


async def test_the_owners_rules_and_the_glossary_give_the_items_that_match(library):
    tool = _tool(library)
    out = (await tool.execute(action="search", query="refund window days", doc="owner-rules")).content
    hit = out.split("\n[2] ")[0]
    assert "[1] owner-rules › Rules" in hit and "refund window of 14 days" in hit
    assert hit.count("\n- ") <= 4 and "report 1." not in hit                     # its items, not the whole list


async def test_a_document_is_found_however_its_name_is_written(library):
    tool = _tool(library)
    for name in ("Valuation 5ed McKinsey.md", "valuation_5ed_mckinsey", "VALUATION-5ED-MCKINSEY"):
        out = await tool.execute(action="search", query="divest early", doc=name)
        assert out.success and "(Valuation)" in out.content, name
    rules = await tool.execute(action="outline", doc="owner-rules")
    assert rules.success and "Rules" in rules.content


async def test_before_a_turn_the_sections_a_message_bears_on_are_named(tmp_path, workspace, monkeypatch):
    from captain_claw.tools import knowledge as kn

    index = _index(tmp_path, workspace)
    try:
        tool = _tool(workspace, index)
        monkeypatch.setattr(kn, "POINTER_MIN", 0.0)
        got = await tool.pointers("What beta do we use for the cost of equity for banks?")
        assert 1 <= len(got) <= kn.POINTERS and all(p["doc"] for p in got)
        assert len({p["where"] for p in got}) == len(got)                          # each section once
        note = kn.pointer_note(got)
        assert note.startswith("Your owner's documents look like they cover this message.")
        assert f"(doc={got[0]['doc']!r}, section=" in note
        assert await tool.pointers("thanks!") == []                                 # too short to bear on anything
        monkeypatch.setattr(kn, "POINTER_MIN", 99.0)
        assert await tool.pointers("What beta do we use for the cost of equity for banks?") == []
    finally:
        index.close()
    assert await _tool(workspace).pointers("What beta do we use for banks?") == []  # no index: no score to trust
    assert kn.pointer_note([]) == ""


async def test_the_turn_gets_the_note_only_with_the_knowledge_tool():
    from captain_claw.agent_context_mixin import AgentContextMixin

    class Pointing(KnowledgeTool):
        async def pointers(self, message: str, limit: int = 2) -> list[dict]:
            return [{"doc": "Valuation", "where": "Valuation › 28 Capitalized Expenses", "section": "28 Capitalized Expenses",
                     "ref": "28 Capitalized Expenses", "score": 0.9}]

    registry = ToolRegistry()
    registry.register(Pointing())
    agent = types.SimpleNamespace(tools=registry)
    await AgentContextMixin._refresh_knowledge_context_cache(agent, "Does cutting R&D lift ROIC this year?")
    note = AgentContextMixin._build_knowledge_context_note(agent)
    assert "- 28 Capitalized Expenses (doc='Valuation', section='28 Capitalized Expenses')" in note
    bare = types.SimpleNamespace(tools=ToolRegistry())
    await AgentContextMixin._refresh_knowledge_context_cache(bare, "Does cutting R&D lift ROIC this year?")
    assert AgentContextMixin._build_knowledge_context_note(bare) == ""             # no knowledge tool, no note
