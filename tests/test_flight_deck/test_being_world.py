"""Roadmap Tier 1 + first Tier 2 pair — the umwelt: a world a being feels.

Calendar texture (weekday/season/daylight + seasonal drive lean +
month-birthdays), the machine as felt body, boredom → sleep-in, dream
recombination, relationship memory, and life projects. Everything traces to
a real source: the host, the calendar, the ledger, the disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)   # a Thursday, July
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


async def _being(store, name="Zvjezdana", stage="child", home=True, now=NOW):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=now)
    store.hatch(OWNER, b["slug"], now=now)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=now)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    if home:
        await life.build_home(store.get(OWNER, b["slug"]))
    return store.get(OWNER, b["slug"])


def _mk(being, rel, text=""):
    p = life._home_path(being, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text or f"# {rel}\n", encoding="utf-8")


def _reply(**over):
    d = {"act_kind": "journal", "summary": "a small day",
         "journal_entry": "I kept things small and true.",
         "served_drive": "grow", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ── The calendar (T1.2 + T1.3) ───────────────────────────────────────────

def test_season_and_daylight_follow_the_real_calendar(monkeypatch):
    monkeypatch.setattr(world, "_tz_name", lambda: "Europe/Zagreb")
    assert world.season(NOW) == "summer"
    assert world.season(datetime(2026, 1, 10, tzinfo=timezone.utc)) == "winter"
    assert "long" in world._daylight_phrase(NOW)
    # the same July day is winter south of the equator
    monkeypatch.setattr(world, "_tz_name", lambda: "Australia/Sydney")
    assert world.season(NOW) == "winter"


def test_seasonal_lean_shifts_explore_and_create(monkeypatch):
    monkeypatch.setattr(world, "_tz_name", lambda: "")
    assert world.seasonal_weight_shift("explore", NOW) == 0.05   # summer
    assert world.seasonal_weight_shift("create", NOW) == 0.0
    jan = datetime(2026, 1, 10, tzinfo=timezone.utc)
    assert world.seasonal_weight_shift("create", jan) == 0.05    # winter
    assert world.seasonal_weight_shift("survive", NOW) == 0.0
    # and the lean reaches the real pressure ranking
    drives = {"explore": {"weight": 0.9, "satisfaction": 0.5}}
    ranked = dict(life.drive_pressures(drives, now=NOW))
    assert ranked["explore"] == pytest.approx((0.9 + 0.05) * 0.5)


def test_world_note_speaks_the_morning(monkeypatch, store):
    import asyncio
    monkeypatch.setattr(world, "_tz_name", lambda: "")
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    note = world.world_note(b, NOW)
    assert "THE WORLD TODAY" in note and "Thursday" in note
    assert "summer" in note and "July" in note
    saturday = world.world_note(b, datetime(2026, 7, 18, 12, 0,
                                            tzinfo=timezone.utc))
    assert "weekend" in saturday


# ── The body (T1.1) ──────────────────────────────────────────────────────

def test_body_note_only_speaks_under_real_strain(monkeypatch, store):
    import asyncio
    b = asyncio.get_event_loop().run_until_complete(_being(store))
    monkeypatch.setattr(world, "_body_readings", lambda: [])
    assert world.body_note(b) is None                     # health is silent
    monkeypatch.setattr(
        world, "_body_readings",
        lambda: ["the machine is under heavy load (load 12.0 on 8 cores)"])
    note = world.body_note(b)
    assert "YOUR BODY" in note and "heavy load" in note


# ── Month-birthdays (T1.3) ───────────────────────────────────────────────

async def test_month_birthday_fires_once_at_that_days_dream(store):
    b = await _being(store, now=NOW - timedelta(days=31))   # hatched June 15
    later = NOW - timedelta(days=1) + timedelta(days=31)    # same day-of-month
    bb = store.get(OWNER, b["slug"])
    assert world._is_monthiversary(bb, datetime(2026, 7, 15,
                                                tzinfo=timezone.utc))
    note = world.anniversary_note(store, bb,
                                  datetime(2026, 7, 15, tzinfo=timezone.utc),
                                  "dream")
    assert note and "1 MONTH" in note and "journal/2026-06-15.md" in note
    names = [e["data"].get("name") for e in store.events(OWNER, b["slug"])
             if e["kind"] == "milestone"]
    assert "turned_1_months" in names
    # a wake tick that day says nothing; an ordinary day says nothing
    assert world.anniversary_note(
        store, bb, datetime(2026, 7, 15, tzinfo=timezone.utc), "wake") is None
    assert world.anniversary_note(
        store, bb, datetime(2026, 7, 20, tzinfo=timezone.utc), "dream") is None
    del later


# ── Dream recombination (T1.6) ───────────────────────────────────────────

async def test_dreams_tangle_two_real_artifacts(store):
    b = await _being(store)
    _mk(b, "garden/red-map.md")
    _mk(b, "garden/quiet-gate.md")
    note = world.dream_tangle(store.get(OWNER, b["slug"]), NOW, "dream")
    assert note and "tangle" in note
    assert note.count("garden/") + note.count("skills/") >= 2
    assert world.dream_tangle(store.get(OWNER, b["slug"]), NOW, "wake") is None
    lonely = await _being(store, name="Prazna", home=False)
    assert world.dream_tangle(lonely, NOW, "dream") is None


# ── Relationship memory (T2.7) ───────────────────────────────────────────

async def test_real_exchanges_nudge_the_relationships_file(store):
    b = await _being(store)
    assert world.relationships_nudge(store, b, NOW, "dream") is None
    store.record_event(b["id"], "letter_sent",
                       {"to": "iskra-lada-1", "preview": "hi"},
                       now=NOW - timedelta(hours=2))
    note = world.relationships_nudge(store, b, NOW, "dream")
    assert note and "iskra-lada-1" in note and "RELATIONSHIPS.md" in note
    assert world.relationships_nudge(store, b, NOW, "wake") is None
    # yesterday's exchange has already had its dream
    assert world.relationships_nudge(
        store, b, NOW + timedelta(days=2), "dream") is None


# ── Life projects (T2.11) ────────────────────────────────────────────────

async def test_project_offered_then_checked_in_weekly(store):
    baby = await _being(store, name="Beba", stage="infant")
    assert world.project_note(store, baby, NOW, "dream") is None   # too young
    b = await _being(store)
    offer = world.project_note(store, b, NOW, "dream")
    assert offer and "self/PROJECT.md" in offer and "declare" in offer
    assert world.project_note(store, b, NOW, "wake") is None
    _mk(b, "self/PROJECT.md", "# A book of small poems\nStep one: one poem.")
    checkin = world.project_note(store, b, NOW, "dream")
    assert checkin and "reread self/PROJECT.md" in checkin
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "project_checkin" in kinds
    # revisited this week already → silence until next week
    assert world.project_note(store, b, NOW + timedelta(days=2),
                              "dream") is None
    assert world.project_note(store, b, NOW + timedelta(days=8),
                              "dream") is not None


# ── Boredom → sleep-in (T1.5) ────────────────────────────────────────────

async def test_a_quiet_day_stretches_the_next_sleep(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    sated = {n: {"weight": d["weight"], "satisfaction": 0.97}
             for n, d in b["drives"].items()}
    store.tick_bookkeeping(b["id"], drives=sated,
                           next_wake_at=NOW - timedelta(minutes=5),
                           now=NOW - timedelta(hours=1))

    async def send(being, prompt):
        return _reply(next_wake_minutes=60)

    out = await life.tick(db, store, store.get(OWNER, b["slug"]),
                          now=NOW, send_fn=send, usage_fn=_usage)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "slept_in" in kinds
    woke = datetime.fromisoformat(out["next_wake"])
    assert woke == NOW + timedelta(minutes=120)          # 60 → doubled


async def test_a_called_day_never_sleeps_in(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    sated = {n: {"weight": d["weight"], "satisfaction": 0.97}
             for n, d in b["drives"].items()}
    store.tick_bookkeeping(b["id"], drives=sated,
                           next_wake_at=NOW - timedelta(minutes=5),
                           now=NOW - timedelta(hours=1))
    store.post_chore(OWNER, b["slug"], "water the garden", 1000, now=NOW)

    async def send(being, prompt):
        return _reply(next_wake_minutes=60)

    out = await life.tick(db, store, store.get(OWNER, b["slug"]),
                          now=NOW, send_fn=send, usage_fn=_usage)
    assert "slept_in" not in [e["kind"] for e in
                              store.events(OWNER, b["slug"])]
    assert datetime.fromisoformat(out["next_wake"]) == NOW + \
        timedelta(minutes=60)


# ── The umwelt reaches the tick prompt ───────────────────────────────────

async def test_morning_prompt_carries_the_world(store):
    db = FakeDB()
    b = await _being(store)
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    assert "THE WORLD TODAY" in prompts[0]            # first tick of the day
