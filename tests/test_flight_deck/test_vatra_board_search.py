"""Increment 5: tokenised board search + per-subtask exclusion."""

import tempfile
from pathlib import Path

import pytest

from captain_claw.flight_deck.db import FlightDeckDB


@pytest.fixture()
async def db():
    tmp = tempfile.mkdtemp()
    d = FlightDeckDB(str(Path(tmp) / "fd.db"))
    await d.init()
    # vatra_board.session_id has an FK to basna_sessions(id); create the parents.
    u = await d.create_user("t@x.co", "h")
    sess = await d.create_basna_session(u["id"], "write a book")
    d._test_sid = sess["id"]
    yield d
    await d.close()


@pytest.mark.asyncio
async def test_multiword_query_matches_via_tokens(db):
    sid = db._test_sid
    # an OUTPUT row from s4 whose body scatters the query words
    await db.add_vatra_board(sid, "editor-writer", "s4", "output", "Story Bible",
                             "clue register for the sealed space and the culprit reveal")
    # a NARRATION row with the same words must NOT satisfy the wait
    await db.add_vatra_board(sid, "editor-writer", "s5", "narration", "",
                             "thinking about the clue register sealed space culprit")
    rows = await db.search_vatra_board(sid, "Story Bible clue register sealed space culprit")
    assert len(rows) == 1
    assert rows[0]["from_subtask"] == "s4"
    assert rows[0]["kind"] == "output"


@pytest.mark.asyncio
async def test_exact_phrase_still_matches(db):
    sid = db._test_sid
    await db.add_vatra_board(sid, "o1", "s4", "note", "Bible", "the exact phrase here verbatim")
    rows = await db.search_vatra_board(sid, "exact phrase here")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_exclude_subtask_keeps_same_archetype_teammate(db):
    sid = db._test_sid
    # two writers share the SAME archetype id but different subtasks
    await db.add_vatra_board(sid, "editor-writer", "s4", "output", "Part One", "chapter content one")
    await db.add_vatra_board(sid, "editor-writer", "s5", "output", "Part Two", "chapter content two")
    # s5 searching, excluding its OWN subtask → still sees s4 (same archetype!)
    rows = await db.search_vatra_board(sid, "chapter content", exclude_subtask="s5")
    subs = {r["from_subtask"] for r in rows}
    assert "s4" in subs and "s5" not in subs
    # by contrast excluding by owner would hide both (the bug this fixes)
    rows_owner = await db.search_vatra_board(sid, "chapter content", exclude_owner="editor-writer")
    assert rows_owner == []


@pytest.mark.asyncio
async def test_list_board_exclude_subtask(db):
    sid = db._test_sid
    await db.add_vatra_board(sid, "editor-writer", "s4", "output", "P1", "one")
    await db.add_vatra_board(sid, "editor-writer", "s5", "output", "P2", "two")
    rows = await db.list_vatra_board(sid, exclude_subtask="s5")
    assert {r["from_subtask"] for r in rows} == {"s4"}
