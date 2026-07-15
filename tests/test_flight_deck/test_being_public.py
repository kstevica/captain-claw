"""Iskra §9 — the public square: strangers' notes, threads, tick integration.

The square is the one un-gated surface. These tests pin its two guarantees:
a stranger can only ever reach a PUBLIC being (never private data), and a
visitor's note is a suggestion the being MAY weigh — never parenting it must
obey. They also lock the anti-flood / anti-leak rules and the tick plumbing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck.beings import (
    PUBLIC_MSG_MAX_CHARS,
    BeingError,
    BeingsStore,
)

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
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


async def _being(store, name="Zephyr", stage="child", public=True):
    b = store.conceive(OWNER, name, preset="artist",
                       allowance_preset="5M", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=NOW)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    await life.build_home(store.get(OWNER, b["slug"]))
    if public:
        store.set_public(OWNER, b["slug"], True, now=NOW)
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "s", "journal_entry": "I mused.",
         "served_drive": "explore", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "curious"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 100, "completion_tokens": 100,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── Visibility: a private being is not reachable through the public door ──

def test_private_being_hidden_public_flag_reveals(store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(
        _being(store, public=False))
    with pytest.raises(BeingError):
        store.get_public(b["slug"])
    assert store.public_beings() == []
    store.set_public(OWNER, b["slug"], True, now=NOW)
    assert store.get_public(b["slug"])["slug"] == b["slug"]
    assert [x["slug"] for x in store.public_beings()] == [b["slug"]]
    assert store.vitals(OWNER, b["slug"])["public"] is True
    # …and can be closed again
    store.set_public(OWNER, b["slug"], False, now=NOW)
    assert store.public_beings() == []


# ── Posting: name required, empty refused, 64-char cap, dead can't answer ──

async def test_post_validation_and_trim(store):
    b = await _being(store)
    slug = b["slug"]
    with pytest.raises(BeingError, match="name"):
        store.post_public_message(slug, "  ", "hello", now=NOW)
    with pytest.raises(BeingError, match="empty"):
        store.post_public_message(slug, "Mira", "   ", now=NOW)
    long = "x" * 200
    r = store.post_public_message(slug, "Mira", long, now=NOW)
    th = store.public_thread(slug, r["thread_id"])
    assert len(th["messages"][0]["body"]) == PUBLIC_MSG_MAX_CHARS


async def test_dead_being_refuses_notes(store):
    b = await _being(store)
    store.set_state(OWNER, b["slug"], "dead", now=NOW)
    # still public + viewable, but no new notes
    store.set_public(OWNER, b["slug"], True, now=NOW)
    with pytest.raises(BeingError, match="died"):
        store.post_public_message(b["slug"], "Mira", "hi", now=NOW)


# ── Threads: one unseen note per thread, follow-up after a tick sees it ──

async def test_thread_flood_guard_and_followup(store):
    b = await _being(store)
    slug = b["slug"]
    r = store.post_public_message(slug, "Mira", "first", now=NOW)
    tid = r["thread_id"]
    # a second note before a tick has SEEN the first is refused
    with pytest.raises(BeingError, match="tick"):
        store.post_public_message(slug, "Mira", "second", thread_id=tid,
                                  now=NOW)
    # a tick sees it → the follow-up is now allowed
    unseen = store.unread_public_messages(b["id"])
    store.mark_public_messages_read([m["id"] for m in unseen], now=NOW)
    store.post_public_message(slug, "Mira", "second", thread_id=tid, now=NOW)
    assert len(store.public_thread(slug, tid)["messages"]) == 2


async def test_stats_and_parent_overview_and_answer(store):
    b = await _being(store)
    slug = b["slug"]
    r1 = store.post_public_message(slug, "Mira", "paint rain", now=NOW)
    store.post_public_message(slug, "Ivo", "why blue?", now=NOW)
    # the being answers Mira
    store.answer_public_message(b["id"], r1["thread_id"], "I will try.",
                                now=NOW)
    stats = store.public_stats(b["id"])
    assert stats == {"messages": 2, "threads": 2, "answered": 1}
    threads = store.public_threads_for(OWNER, slug)
    assert len(threads) == 2
    mira = next(t for t in threads if t["sender_name"] == "Mira")
    assert [m["role"] for m in mira["messages"]] == ["public", "being"]
    assert mira["messages"][0]["answered_at"] is not None


# ── Tick: visitor notes are surfaced as NOT-parent, then not re-surfaced ──

async def test_tick_surfaces_visitors_not_as_parent_and_replies(store):
    db = FakeDB()
    b = await _being(store)
    slug = b["slug"]
    r = store.post_public_message(slug, "Mira", "what does the tide teach?",
                                  now=NOW)
    tid = r["thread_id"]
    seen = {}

    async def send(being, prompt):
        seen["prompt"] = prompt
        return _reply(public_replies=[
            {"thread_id": tid[:8], "reply": "Patience, mostly."}])

    before = store.get(OWNER, slug)["drives"]["connect"]["satisfaction"]
    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    p = seen["prompt"]
    assert "VISITORS" in p and "NOT from your parent" in p
    assert "Mira" in p and "what does the tide teach?" in p
    # framed as a suggestion, never an order
    assert "SEEDS, never orders" in p
    # connection was gently fed
    after = store.get(OWNER, slug)["drives"]["connect"]["satisfaction"]
    assert after > before
    # the optional reply landed in the thread and marked the note answered
    th = store.public_thread(slug, tid)
    assert [m["role"] for m in th["messages"]] == ["public", "being"]
    assert th["messages"][1]["body"] == "Patience, mostly."
    assert th["messages"][0]["answered_at"] is not None

    # a later tick no longer re-surfaces the (now seen) note
    async def send2(being, prompt):
        seen["p2"] = prompt
        return _reply()

    await life.tick(db, store, store.get(OWNER, slug),
                    now=NOW + timedelta(hours=1), send_fn=send2,
                    usage_fn=_usage)
    assert "what does the tide teach?" not in seen["p2"]


async def test_non_public_being_never_hears_the_square(store):
    db = FakeDB()
    b = await _being(store, public=False)
    # a note can't even be posted to a private being, but even if the table
    # held one, a private tick must not surface it — assert the prompt is clean
    seen = {}

    async def send(being, prompt):
        seen["prompt"] = prompt
        return _reply()

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    assert "VISITORS" not in seen["prompt"]
