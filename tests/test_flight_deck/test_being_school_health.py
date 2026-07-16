"""Roadmap T2.12-13 — reading lists (verified learning) and illness as
consequence (fever/confusion from the real ledger, never dice)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
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


async def _being(store, name="Zvjezdana", stage="child", now=NOW):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=now)
    store.hatch(OWNER, b["slug"], now=now)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=now)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    await life.build_home(store.get(OWNER, b["slug"]))
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "s", "journal_entry": "small.",
         "served_drive": "grow", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ═══ Reading lists (T2.12) ═══════════════════════════════════════════════

async def test_reading_assigned_surfaces_every_wake(store):
    b = await _being(store, now=NOW - timedelta(days=1))
    store.add_reading(OWNER, b["slug"], "https://example.org/maps",
                      note="old maps", fee_tokens=100_000, now=NOW)
    bb = store.get(OWNER, b["slug"])
    assert bb["reading_list"][0]["ref"].endswith("/maps")
    percepts = life.percepts_since(store, bb)
    reading = [p for p in percepts if p.startswith("READING")]
    assert reading and "old maps" in reading[0] \
        and '"reading_report"' in reading[0]


async def test_verified_report_pays_and_feeds_grow(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=1))
    bb = store.get(OWNER, b["slug"])
    store.add_reading(OWNER, b["slug"], "the red atlas", fee_tokens=100_000,
                      now=NOW - timedelta(hours=2))
    item = store.get(OWNER, b["slug"])["reading_list"][0]
    balance0 = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]

    async def send(being, prompt):
        p = life._home_path(being, "garden/reports/red-atlas.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# What the atlas taught me\n", encoding="utf-8")
        return _reply(act_kind="create", served_drive="grow",
                      reading_report={"item_id": item["id"],
                                      "path": "garden/reports/red-atlas.md"})

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    after = store.get(OWNER, b["slug"])
    done = after["reading_list"][0]
    assert done["done_at"] and done["report_path"].endswith("red-atlas.md")
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "reading_done" in kinds
    assert store.wallet_view(after)["balance_tokens"] > balance0  # fee landed
    assert any(e["data"].get("name") == "first_report"
               for e in store.events(OWNER, b["slug"])
               if e["kind"] == "milestone")
    # …and the grow drive was truly served (verified learning is food)
    assert (tick := next(e for e in store.events(OWNER, b["slug"])
                         if e["kind"] == "tick"))
    assert tick["data"]["drives"]["grow"] > 0.7
    del bb


async def test_unwritten_report_is_refused(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=1))
    store.add_reading(OWNER, b["slug"], "the red atlas", fee_tokens=100_000,
                      now=NOW - timedelta(hours=2))
    item = store.get(OWNER, b["slug"])["reading_list"][0]

    async def send(being, prompt):
        return _reply(reading_report={"item_id": item["id"],
                                      "path": "garden/reports/ghost.md"})

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    after = store.get(OWNER, b["slug"])
    assert after["reading_list"][0]["done_at"] is None      # still open
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "reading_refused" in kinds and "reading_done" not in kinds


async def test_bogus_reading_id_is_flagged(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=1))

    async def send(being, prompt):
        p = life._home_path(being, "garden/reports/x.md")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
        return _reply(reading_report={"item_id": "deadbeef",
                                      "path": "garden/reports/x.md"})

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert "reading_claim_invalid" in [
        e["kind"] for e in store.events(OWNER, b["slug"])]


def test_reading_add_remove_validation(store):
    b = store.conceive(OWNER, "Mala", preset="explorer", now=NOW)
    store.hatch(OWNER, b["slug"], now=NOW)
    with pytest.raises(BeingError, match="something to read"):
        store.add_reading(OWNER, b["slug"], "   ", now=NOW)
    store.add_reading(OWNER, b["slug"], "a small book", fee_tokens=10**9,
                      now=NOW)
    item = store.get(OWNER, b["slug"])["reading_list"][0]
    assert item["fee_tokens"] == store.READING_MAX_FEE_TOKENS   # clamped
    store.remove_reading(OWNER, b["slug"], item["id"], now=NOW)
    assert store.get(OWNER, b["slug"])["reading_list"] == []


# ═══ Illness (T2.13) ═════════════════════════════════════════════════════

async def test_fever_from_collapse_floors_cadence_and_colors_affect(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    store.set_tick_interval(OWNER, b["slug"], 2, now=NOW)   # parent-pinned
    store.record_event(b["id"], "collapsed_exhausted", {},
                       now=NOW - timedelta(hours=3))
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    out = await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                          send_fn=send, usage_fn=_usage)
    assert "YOU ARE UNWELL" in prompts[0]
    woke = datetime.fromisoformat(out["next_wake"])
    assert woke >= NOW + timedelta(minutes=world.FEVER_MIN_WAKE_MINUTES)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "fever" in kinds and "resting_fever" in kinds
    tick_ev = next(e for e in store.events(OWNER, b["slug"])
                   if e["kind"] == "tick")
    assert tick_ev["data"]["mood_engine"] == "feverish"
    # a second tick within the day repeats the percept but not the onset
    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=2), send_fn=send,
                    usage_fn=_usage)
    assert [e["kind"] for e in store.events(OWNER, b["slug"])
            ].count("fever") == 1


async def test_fever_from_timeouts_and_recovery(store):
    b = await _being(store, now=NOW - timedelta(days=2))
    for i in range(3):
        store.record_event(b["id"], "tick_timeout", {},
                           now=NOW - timedelta(hours=1 + i))
    assert world.fever_state(store, store.get(OWNER, b["slug"]),
                             NOW) is not None
    # a day later the events age out — health returns on its own
    assert world.fever_state(store, store.get(OWNER, b["slug"]),
                             NOW + timedelta(hours=26)) is None


async def test_confusion_surfaces_self_exam_at_dream_only(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    for i in range(3):
        store.record_event(b["id"], "narration_mismatch", {},
                           now=NOW - timedelta(hours=1 + i))
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply(act_kind="dream")

    await life.tick(db, store, store.get(OWNER, b["slug"]), kind="dream",
                    now=NOW, send_fn=send, usage_fn=_usage)
    assert "SELF-EXAMINATION" in prompts[0]
    assert "confusion" in [e["kind"] for e in store.events(OWNER, b["slug"])]

    async def send_wake(being, prompt):
        prompts.append(prompt)
        return _reply()

    prompts.clear()
    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send_wake,
                    usage_fn=_usage)
    assert "SELF-EXAMINATION" not in prompts[0]     # wakes stay ordinary
