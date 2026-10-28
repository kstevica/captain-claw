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
  member gets nothing.
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
        none = (await tool.execute(action="search", query='"no such phrase anywhere"')).content
        assert none.startswith("Nothing in your knowledge")
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
