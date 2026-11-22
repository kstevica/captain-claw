"""A turn the chat heals from the agent's replay is not written twice.

cohort001 (2026-11): a slip in the chat's replay merge saved whole agent
sessions to the Flight Deck chat log a second time (613 of 2,403 rows). The
chat now marks the turns it adds from a replay (``"healed": true``); the log
skips one it already holds word for word and hands back the existing row id.
"""

from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck.db import FlightDeckDB

LONG = "A reply long enough that the log would never hold it twice by chance."
HEALED = json.dumps({"healed": True})


@pytest.fixture
async def chat(tmp_path):
    db = FlightDeckDB(tmp_path / "fd.db")
    await db.init()
    user = await db.create_user("a@example.com", "x")
    await db.upsert_chat_session("proc-ana", user["id"], agent_id="ana")
    try:
        yield db, user["id"]
    finally:
        await db.close()


async def _rows(db, uid):
    return [(m["role"], m["content"]) for m in await db.get_chat_messages("proc-ana", uid, limit=100)]


async def test_a_healed_turn_the_log_has_is_not_written_again(chat):
    db, uid = chat
    [first] = await db.add_chat_messages("proc-ana", uid, [{"role": "assistant", "content": LONG}])
    known: list[int] = []
    ids = await db.add_chat_messages("proc-ana", uid, [
        {"role": "assistant", "content": LONG, "metadata": HEALED},
        {"role": "user", "content": "A question the log really is missing, asked on the phone.", "metadata": HEALED},
    ], known=known)
    assert ids[0] == first and known == [0]
    assert ids[1] != first
    assert await _rows(db, uid) == [
        ("assistant", LONG), ("user", "A question the log really is missing, asked on the phone.")]


async def test_live_rows_and_short_lines_are_always_written(chat):
    db, uid = chat
    await db.add_chat_messages("proc-ana", uid, [
        {"role": "assistant", "content": LONG}, {"role": "user", "content": "da"}])
    known: list[int] = []
    await db.add_chat_messages("proc-ana", uid, [
        {"role": "assistant", "content": LONG},                       # live: a reply can repeat
        {"role": "user", "content": "da", "metadata": HEALED},        # short: the chat's call
        {"role": "assistant", "content": LONG, "metadata": "{not json"},
    ], known=known)
    assert known == []
    assert len(await _rows(db, uid)) == 5


async def test_another_role_or_session_does_not_count(chat):
    db, uid = chat
    await db.upsert_chat_session("proc-bob", uid, agent_id="bob")
    await db.add_chat_messages("proc-bob", uid, [{"role": "assistant", "content": LONG}])
    await db.add_chat_messages("proc-ana", uid, [{"role": "user", "content": LONG}])
    known: list[int] = []
    await db.add_chat_messages("proc-ana", uid, [{"role": "assistant", "content": LONG, "metadata": HEALED}], known=known)
    assert known == []
    assert await _rows(db, uid) == [("user", LONG), ("assistant", LONG)]
