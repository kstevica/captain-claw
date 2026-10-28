"""Intuition rows are read by column name, whatever order the DB holds them in.

The source_trigger migration was added above the older maturation one, so a
DB made before it holds maturation_state … dream_cycles_seen ahead of
source_trigger and a new DB the other way round. A fixed column list
mislabelled one of them: a new DB's get() said maturation_state="manual" and
source_trigger=0.

Every test uses a tmp DB, default config and a stub metrics store (nothing
here may reach ~/.captain-claw).
"""

from __future__ import annotations

import sqlite3

import pytest

# The table of a DB made before the source_trigger migration, with the later
# migrations (source_trigger, source_message_id) appended as they would be.
_OLD_LAYOUT = """CREATE TABLE intuitions (
    id TEXT PRIMARY KEY, content TEXT NOT NULL, thread_type TEXT NOT NULL,
    source_layers TEXT NOT NULL, source_ids TEXT, source_session TEXT,
    confidence REAL NOT NULL DEFAULT 0.5, importance INTEGER NOT NULL DEFAULT 5,
    access_count INTEGER NOT NULL DEFAULT 0, last_accessed TEXT,
    validated INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, decayed_at TEXT, tags TEXT,
    resolution_state TEXT DEFAULT NULL, resolved_from_id TEXT DEFAULT NULL,
    maturation_state TEXT DEFAULT 'mature', matured_at TEXT DEFAULT NULL,
    dream_cycles_seen INTEGER DEFAULT 0,
    source_trigger TEXT DEFAULT 'dream', source_message_id TEXT DEFAULT NULL)"""


class _Metrics:
    async def record_event(self, *args, **kwargs) -> None:
        pass


@pytest.fixture(params=["new", "made_before_source_trigger"])
async def ns(request, tmp_path, monkeypatch):
    from captain_claw import cognitive_metrics, nervous_system
    from captain_claw.config import Config

    monkeypatch.setattr(nervous_system, "get_config", Config)
    monkeypatch.setattr(cognitive_metrics, "get_cognitive_metrics_manager", _Metrics)
    db = tmp_path / "intuitions.db"
    if request.param == "made_before_source_trigger":
        conn = sqlite3.connect(db)
        conn.execute(_OLD_LAYOUT)
        conn.commit()
        conn.close()
    mgr = nervous_system.NervousSystemManager(db)
    yield mgr
    await mgr.close()


@pytest.mark.asyncio
async def test_the_two_layouts_really_differ(ns, request):
    await ns.count()
    trigger_first = ns._columns.index("source_trigger") < ns._columns.index("maturation_state")
    assert trigger_first == (request.node.callspec.params["ns"] == "new")


@pytest.mark.asyncio
async def test_get_reads_each_field_from_its_own_column(ns):
    iid = await ns.add(content="Board calls move to Thursdays when Vesna travels.", thread_type="pattern",
                       source_layers=["insights", "deep"], importance=9,
                       source_trigger="manual", source_message_id="msg-7")
    got = await ns.get(iid)
    assert got["content"] == "Board calls move to Thursdays when Vesna travels."
    assert (got["maturation_state"], got["source_trigger"], got["dream_cycles_seen"],
            got["source_message_id"], got["resolution_state"]) == ("mature", "manual", 0, "msg-7", None)
    assert got["source_layers"] == ["insights", "deep"]


@pytest.mark.asyncio
async def test_search_similar_and_lists_read_by_name_too(ns):
    text = "Invoices go out late when the accountant is away."
    iid = await ns.add(content=text, thread_type="hypothesis", importance=9, source_trigger="idle_dream")
    [hit] = await ns.search("invoices accountant")
    [similar] = await ns.find_similar(text)
    [recent] = await ns.list_recent()
    for item in (hit, similar, recent):
        assert (item["id"], item["source_trigger"], item["maturation_state"]) == (iid, "idle_dream", "mature")
    assert "rank" in hit and "rank" in similar
