"""Insights remember where they came from: the file (a document's name, a
path, a URL) and the section or page in it — kept through add, search,
import and the REST route, shown to the agent with each surfaced insight,
and taken from the insight extractor only when the conversation shows it."""

from __future__ import annotations

import json
import types

import aiosqlite
import pytest

from captain_claw import insights
from captain_claw.agent_context_mixin import AgentContextMixin
from captain_claw.insights import InsightsManager, seen_source, source_note


@pytest.fixture
async def mgr(tmp_path):
    m = InsightsManager(tmp_path / "insights.db")
    yield m
    await m.close()


async def test_an_insight_keeps_its_file_and_section(mgr):
    iid = await mgr.add(content="Shipping is free over 50 EUR.", category="fact", importance=7,
                        source_file="price-list.pdf", source_section="page 4")
    got = await mgr.get(iid)
    assert got["source_file"] == "price-list.pdf" and got["source_section"] == "page 4"
    assert (await mgr.list_recent(limit=1))[0]["source_file"] == "price-list.pdf"
    found = await mgr.search("shipping free")
    assert found[0]["source_file"] == "price-list.pdf" and isinstance(found[0]["rank"], float)
    assert await mgr.update(iid, source_section="page 5")
    assert (await mgr.get(iid))["source_section"] == "page 5"
    plain = await mgr.get(await mgr.add(content="The owner likes short answers.", category="preference"))
    assert plain["source_file"] is None and plain["source_section"] is None


async def test_a_database_from_before_gets_the_columns(tmp_path):
    path = tmp_path / "old.db"
    async with aiosqlite.connect(path) as db:      # the table as the first release made it
        await db.execute("""CREATE TABLE insights (id TEXT PRIMARY KEY, content TEXT NOT NULL,
            category TEXT NOT NULL, entity_key TEXT, importance INTEGER NOT NULL DEFAULT 5,
            source_tool TEXT, source_session TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            expires_at TEXT, tags TEXT)""")
        await db.execute("INSERT INTO insights VALUES ('old1', 'An old fact.', 'fact', NULL, 5, NULL, NULL,"
                         " '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', NULL, NULL)")
        await db.commit()
    m = InsightsManager(path)
    try:
        assert (await m.get("old1"))["source_file"] is None
        iid = await m.add(content="A new fact from a file.", category="fact", source_file="notes.md")
        assert (await m.get(iid))["source_file"] == "notes.md"
    finally:
        await m.close()


async def test_an_import_keeps_where_each_insight_came_from(mgr):
    stats = await mgr.import_items([{"content": "Returns take 30 days.", "category": "fact", "importance": 6,
                                     "source_file": "policy.pdf", "source_section": "Returns"}])
    assert stats["stored"] == 1
    assert (await mgr.list_recent(limit=1))[0]["source_file"] == "policy.pdf"


def test_the_extractor_names_only_a_source_the_conversation_shows():
    convo = '[tool] read: /workspace/knowledge/price-list.md\n[assistant] Shipping is free over 50 EUR.'
    assert seen_source("price-list.md", convo) == "price-list.md"
    assert seen_source("/workspace/knowledge/PRICE-LIST.md", convo) == "/workspace/knowledge/PRICE-LIST.md"
    assert seen_source("catalogue.pdf", convo) is None            # never in the conversation: invented
    assert seen_source("null", convo) is None and seen_source(None, convo) is None
    assert seen_source("https://x.co/p", "", "fetched https://x.co/p today") == "https://x.co/p"


def test_a_source_reads_as_a_short_note():
    assert source_note({"source_file": "price-list.pdf", "source_section": "page 4"}) == " (from price-list.pdf — page 4)"
    assert source_note({"source_file": "notes.md"}) == " (from notes.md)"
    assert source_note({"source_section": "page 4"}) == ""
    assert source_note({}) == ""


def test_the_agent_sees_where_a_surfaced_insight_came_from():
    agent = types.SimpleNamespace(_insights_context_cache=[
        {"category": "fact", "importance": 7, "content": "Shipping is free over 50 EUR.",
         "source_file": "price-list.pdf", "source_section": "page 4"},
        {"category": "preference", "importance": 5, "content": "Short answers."}])
    note = AgentContextMixin._build_insights_context_note(agent)
    assert "- [fact] (imp:7) Shipping is free over 50 EUR. (from price-list.pdf — page 4)" in note
    assert "- [preference] (imp:5) Short answers.\n" in note + "\n"
    block = AgentContextMixin._build_insights_block(agent)
    assert 'marked "(from <file>)"' in block


async def test_the_insights_tool_records_and_shows_a_source(mgr, monkeypatch):
    from captain_claw.tools import insights as tool_mod
    monkeypatch.setattr(tool_mod, "_resolve_manager", lambda session_id: mgr)
    tool = tool_mod.InsightsTool()
    assert "source" in tool.parameters["properties"]
    added = await tool.execute(action="add", content="Support hours are 8-16 CET.", category="fact",
                               source="hours.docx", source_section="Contact")
    assert added.success
    listed = await tool.execute(action="list")
    assert "Support hours are 8-16 CET. (from hours.docx — Contact)" in listed.content
    await tool.execute(action="add", content="A section alone means nothing.", source_section="p. 2")
    assert "A section alone means nothing.  id=" in (await tool.execute(action="list")).content


async def test_the_rest_route_takes_a_source(mgr, monkeypatch):
    from captain_claw.web import rest_insights

    class _Req:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body

    monkeypatch.setattr(rest_insights, "_resolve_manager", lambda request: mgr)
    resp = await rest_insights.create_insight(None, _Req({
        "content": "Shipping is free over 50 EUR.", "category": "fact", "importance": 7, "tags": "owner-setup",
        "source_file": "price-list.pdf", "source_section": "page 4"}))
    data = json.loads(resp.text)
    assert resp.status == 201 and data["source_file"] == "price-list.pdf" and data["source_section"] == "page 4"


def test_the_extraction_prompt_asks_for_the_source():
    text = (insights.Path(insights.__file__).parent / "instructions" / "insight_extraction_system_prompt.md").read_text()
    assert '"source"' in text and '"source_section"' in text
