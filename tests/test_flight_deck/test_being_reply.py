"""Iskra — the parent writing back: reply reaches the being's next tick."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


class FakeDB:
    async def list_chat_sessions(self, user_id):
        return []

    async def upsert_chat_session(self, *a, **k):
        return {}

    async def add_chat_messages(self, *a, **k):
        return [1]

    async def log_run_cost(self, *a, **k):
        pass


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, name="Lada", stage="child", allowance="5M"):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset=allowance, now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    bb = store.get(OWNER, b["slug"])
    await life.build_home(bb)
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "s", "journal_entry": "I read it.",
         "served_drive": "connect", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "warm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def test_empty_and_dead_are_refused(store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    with pytest.raises(BeingError, match="empty"):
        store.send_parent_message(OWNER, b["slug"], "   ", now=NOW)
    store.set_state(OWNER, b["slug"], "dead", now=NOW)
    with pytest.raises(BeingError, match="dead"):
        store.send_parent_message(OWNER, b["slug"], "hello", now=NOW)


async def test_reply_reaches_next_tick_once_and_feeds_connect(store):
    db = FakeDB()
    b = await _being(store)
    store.send_parent_message(
        OWNER, b["slug"],
        "I got your letter, Lada. I'm proud of you. Keep growing.", now=NOW)
    # it's unread until a tick delivers it
    assert len(store.unread_parent_messages(b["id"])) == 1
    seen = {}

    async def send(being, prompt):
        seen["prompt"] = prompt
        return _reply(message_to_parent="Thank you. That means everything.")

    before = store.get(OWNER, b["slug"])["drives"]["connect"]["satisfaction"]
    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    # surfaced prominently in the prompt
    assert "YOUR PARENT WROTE TO YOU: I got your letter" in seen["prompt"]
    # delivered once — now read
    assert store.unread_parent_messages(b["id"]) == []
    # connection was fed
    after = store.get(OWNER, b["slug"])["drives"]["connect"]["satisfaction"]
    assert after > before
    # a parent_message event exists for the timeline
    assert "parent_message" in [e["kind"] for e in
                                store.events(OWNER, b["slug"])]

    # next tick no longer re-delivers it
    async def send2(being, prompt):
        seen["p2"] = prompt
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send2,
                    usage_fn=_usage)
    assert "YOUR PARENT WROTE TO YOU" not in seen["p2"]


async def test_message_thread_merges_both_sides_chronologically(store):
    db = FakeDB()
    b = await _being(store)
    store.send_parent_message(OWNER, b["slug"], "First hello.", now=NOW)

    async def send(being, prompt):
        return _reply(message_to_parent="Hello back, parent.")

    await life.tick(db, store, b, now=NOW + timedelta(minutes=1),
                    send_fn=send, usage_fn=_usage)
    store.send_parent_message(OWNER, b["slug"], "Second note.",
                              now=NOW + timedelta(minutes=2))
    thread = store.message_thread(OWNER, b["slug"])
    assert [t["from"] for t in thread] == ["parent", "being", "parent"]
    assert thread[0]["body"] == "First hello." and thread[0]["read"] is True
    assert thread[1]["body"] == "Hello back, parent."
    assert thread[2]["body"] == "Second note." and thread[2]["read"] is False


async def test_reading_is_free_reply_still_spends_attention(store):
    db = FakeDB()
    b = await _being(store)
    store.send_parent_message(OWNER, b["slug"], "How are you?", now=NOW)
    credits0 = store.get(OWNER, b["slug"])["attention_credits"]

    async def send(being, prompt):
        return _reply(message_to_parent="I'm well, thank you for asking.")

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    # the reply spent exactly one credit (reading itself was free)
    assert store.get(OWNER, b["slug"])["attention_credits"] == credits0 - 1
