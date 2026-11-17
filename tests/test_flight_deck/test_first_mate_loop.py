"""First Mate's supervisor: the loop that runs factories on their own.

Pinned here:

* a schedule waits for its first due time, then fires once, as a schedule run,
  and moves on;
* nothing starts with the deck's or the owner's switch off, or a factory not running;
* a run still going skips the schedule; no free Bat slot keeps it due; one due
  more than six hours ago is skipped and noted;
* a version that changes a schedule starts it afresh; two schedules due at once
  make one run;
* budget: a factory that spent its share is parked until it has tokens again,
  and a run's cap is never above what's left;
* a failure streak pauses a factory; runs before it last resumed, and stopped
  runs, don't count;
* the owner's switch pauses and stops, and resumes only what it paused;
* overlapping scans start nothing twice; a deck that never used First Mate gets no file.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from captain_claw.flight_deck import bat_loop
from captain_claw.flight_deck import first_mate_loop as loop
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm

BOB, CAROL = "user-bob", "user-carol"
HOUR = timedelta(hours=1)


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        await asyncio.sleep(0)  # like the real store, give way mid-scan
        self.runs[run_id] = {"id": run_id, "owner_id": owner_id, "title": title, "task": task,
                             "config": config or {}, "status": status, "cumulative_tokens": 0,
                             "cumulative_usd": 0.0, "truth": "", "stopped_reason": ""}
        return run_id

    async def append_event(self, run_id, stage, message="", **data):
        return 1

    async def get_run(self, run_id):
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def list_runs(self, owner_id, limit=50):
        return [dict(r) for r in self.runs.values() if r["owner_id"] == owner_id][:limit]

    async def set_status(self, run_id, status, **kw):
        self.runs[run_id].update(status=status, **kw)


@pytest.fixture
def store(tmp_path, monkeypatch):
    from captain_claw.flight_deck import events
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))  # never the real one
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True})
    s.set_prefs(BOB, {"on": True})
    return s


@pytest.fixture
def bat(monkeypatch):
    fake = FakeBat()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", fake)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    return fake


@pytest.fixture
def sup():
    return loop.Supervisor()


def _live(store, owner=BOB, schedule="every 1h", **kw):
    kw.setdefault("title", "Price watch")
    kw.setdefault("charter", "Check the price of the item and report the three best offers.")
    if schedule is not None:
        kw.setdefault("triggers", [{"kind": "schedule", "schedule": schedule}])
    f = store.create(owner, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


def _runs(store, f):
    return store.list_runs(f["id"])


async def _due_now(store, sup, f):
    """First scan sees the trigger; the next, an hour on, finds it due."""
    t0 = fm._utcnow()
    await sup.tick(t0)
    return t0 + HOUR + timedelta(minutes=1)


# ── pure helpers ───────────────────────────────────────────────────────

def test_week_start_and_shares():
    assert loop.week_start(datetime(2026, 11, 18, 15, 30, tzinfo=UTC)) == datetime(2026, 11, 16, tzinfo=UTC)
    assert loop.week_start(datetime(2026, 11, 16, 0, 0, tzinfo=UTC)) == datetime(2026, 11, 16, tzinfo=UTC)
    fs = [{"id": "a", "version": 1, "state": "running", "share": 1},
          {"id": "b", "version": 2, "state": "paused", "share": 2},
          {"id": "c", "version": 0, "state": "draft", "share": 5},
          {"id": "d", "version": 1, "state": "retired", "share": 5}]
    assert loop.shares({"weekly_tokens": 30_000_000}, fs) == {"a": 10_000_000, "b": 20_000_000}
    assert loop.shares({"weekly_tokens": 1}, []) == {}
    assert loop.trigger_key({"kind": "schedule", "schedule": "x", "tz": ""}) == \
        loop.trigger_key({"tz": "", "schedule": "x", "kind": "schedule"})


# ── schedules ──────────────────────────────────────────────────────────

async def test_a_schedule_waits_then_fires_once(store, bat, sup):
    f = _live(store)
    t0 = fm._utcnow()
    assert (await sup.tick(t0))["started"] == 0  # first sight: never at approval
    [st] = store.trigger_state(f["id"]).values()
    assert st["next_at"] == pytest.approx((t0 + HOUR).timestamp(), abs=2)
    assert (await sup.tick(t0 + timedelta(minutes=30)))["started"] == 0
    t1 = t0 + HOUR + timedelta(minutes=1)
    assert (await sup.tick(t1))["started"] == 1
    [run] = _runs(store, f)
    assert run["trigger"] == "schedule" and run["status"] == "running"
    [st] = store.trigger_state(f["id"]).values()
    assert st["last_at"] == pytest.approx(t1.timestamp()) and st["next_at"] == pytest.approx((t1 + HOUR).timestamp(), abs=2)
    assert (await sup.tick(t1 + timedelta(minutes=5)))["started"] == 0


async def test_switches_and_states_stop_it(store, bat, sup):
    f = _live(store)
    t1 = await _due_now(store, sup, f)
    store.set_prefs(BOB, {"on": False})
    assert (await sup.tick(t1))["started"] == 0
    store.set_prefs(BOB, {"on": True})
    store.set_settings({"enabled": False})
    assert (await sup.tick(t1))["started"] == 0
    store.set_settings({"enabled": True})
    store.set_state(f["id"], "paused", "owner")
    assert (await sup.tick(t1))["started"] == 0
    store.set_state(f["id"], "running", "owner")
    assert (await sup.tick(t1))["started"] == 1


async def test_a_run_still_going_skips_the_schedule(store, bat, sup):
    f = _live(store)
    t1 = await _due_now(store, sup, f)
    await runs.start(f["id"], BOB, watch=False)  # the owner's own run
    out = await sup.tick(t1)
    assert out["started"] == 0 and out["skipped"] == 1
    assert store.events(f["id"])[0]["detail"]["reason"] == "a run was still going"
    [st] = store.trigger_state(f["id"]).values()
    assert st["next_at"] > t1.timestamp() and st["last_at"] is None


async def test_no_free_slot_keeps_it_due(store, bat, sup):
    f = _live(store)
    t1 = await _due_now(store, sup, f)
    for i in range(2):
        await bat.create_run(run_id=f"bat_own{i}", owner_id=BOB, title="mine", task="t", status="running")
    out = await sup.tick(t1)
    assert out["started"] == 0 and out["waiting"] == 1
    bat.runs["bat_own0"]["status"] = "done"
    assert (await sup.tick(t1 + timedelta(minutes=1)))["started"] == 1


async def test_a_schedule_missed_by_hours_is_skipped(store, bat, sup):
    f = _live(store)
    t0 = fm._utcnow()
    await sup.tick(t0)
    out = await sup.tick(t0 + HOUR + loop.LATE_GRACE + timedelta(minutes=1))
    assert out["started"] == 0 and out["skipped"] == 1 and _runs(store, f) == []
    assert "6 hours" in store.events(f["id"])[0]["detail"]["reason"]


async def test_a_changed_schedule_starts_afresh_and_two_due_make_one_run(store, bat, sup):
    f = _live(store)
    t0 = fm._utcnow()
    await sup.tick(t0)
    old = set(store.trigger_state(f["id"]))
    f = store.new_draft(f["id"], "owner", triggers=[{"kind": "schedule", "schedule": "every 2h"},
                                                     {"kind": "schedule", "schedule": "every 3h"}])
    store.approve(f["id"], 2, f["draft"]["sha"], BOB)
    await sup.tick(t0 + timedelta(minutes=1))
    new = set(store.trigger_state(f["id"]))
    assert len(new) == 2 and not (old & new)
    out = await sup.tick(t0 + timedelta(hours=4))  # both are due now
    assert out["started"] == 1 and len(_runs(store, f)) == 1
    assert all(st["next_at"] > (t0 + timedelta(hours=4)).timestamp() for st in store.trigger_state(f["id"]).values())


async def test_priority_goes_first_when_slots_are_few(store, bat, sup):
    low = _live(store, title="Low")
    high = _live(store, title="High")
    store.update_factory(high["id"], "owner", priority=1)
    store.update_factory(low["id"], "owner", priority=5)
    t1 = await _due_now(store, sup, low)
    await bat.create_run(run_id="bat_own", owner_id=BOB, title="mine", task="t", status="running")  # one slot left
    assert (await sup.tick(t1))["started"] == 1
    assert _runs(store, high) and not _runs(store, low)


# ── budget ─────────────────────────────────────────────────────────────

async def test_a_factory_that_spent_its_share_waits_for_tokens(store, bat, sup):
    store.set_prefs(BOB, {"weekly_tokens": 2_000_000})
    f = _live(store)
    run = store.start_run(f["id"])
    store.finish_run(run["id"], "done", tokens=2_000_000)
    now = fm._utcnow()
    out = await sup.tick(now)
    assert out["paused"] == 1
    assert store.get(f["id"])["state"] == "paused_budget"
    out = await sup.tick(now + timedelta(days=8))  # a new week
    assert out["resumed"] == 1 and store.get(f["id"])["state"] == "running"


async def test_a_runs_cap_is_never_above_whats_left(store, bat, sup):
    store.set_prefs(BOB, {"weekly_tokens": 4_000_000})
    f = _live(store)
    spent = store.start_run(f["id"])
    store.finish_run(spent["id"], "done", tokens=3_000_000)
    t1 = await _due_now(store, sup, f)
    assert (await sup.tick(t1))["started"] == 1
    newest = _runs(store, f)[0]
    assert bat.runs[newest["child_id"]]["config"]["token_cap"] == 1_000_000


# ── failure streaks ────────────────────────────────────────────────────

def _finish(store, f, *statuses):
    for status in statuses:
        r = store.start_run(f["id"])
        store.finish_run(r["id"], status)


async def test_a_failure_streak_pauses_the_factory(store, bat, sup):
    f = _live(store, schedule=None)
    _finish(store, f, "failed", "cancelled", "failed")
    await sup.tick(fm._utcnow())
    assert store.get(f["id"])["state"] == "running"  # two failures; a stop doesn't count
    _finish(store, f, "failed")
    await sup.tick(fm._utcnow())
    g = store.get(f["id"])
    assert g["state"] == "paused" and g["state_reason"] == loop.STREAK_REASON
    store.set_state(f["id"], "running", "owner")  # the owner tries again
    await sup.tick(fm._utcnow())
    assert store.get(f["id"])["state"] == "running"  # old failures count no more
    _finish(store, f, "failed", "done", "failed", "failed")
    await sup.tick(fm._utcnow())
    assert store.get(f["id"])["state"] == "running"  # a done broke it
    _finish(store, f, "failed")
    await sup.tick(fm._utcnow())
    assert store.get(f["id"])["state"] == "paused"


# ── the owner's switch ─────────────────────────────────────────────────

async def test_switch_off_pauses_and_stops_switch_on_resumes_its_own(store, bat):
    a, b, c = _live(store, title="A"), _live(store, title="B"), _live(store, title="C")
    store.set_state(b["id"], "paused", "owner", "holiday")
    run = await runs.start(a["id"], BOB, watch=False)
    keep = _live(store, CAROL)
    assert await loop.switch_off(BOB) == 2
    assert store.get(a["id"])["state_reason"] == loop.SWITCH_REASON
    assert store.get_run(run["id"])["status"] == "cancelled"
    assert bat.runs[run["child_id"]]["status"] == "cancelled"
    assert store.get(keep["id"])["state"] == "running"
    assert await loop.switch_on(BOB) == 2
    assert [store.get(x["id"])["state"] for x in (a, b, c)] == ["running", "paused", "running"]
    assert store.get(b["id"])["state_reason"] == "holiday"


# ── the scan ───────────────────────────────────────────────────────────

async def test_a_scan_finishes_runs_even_with_the_deck_off(store, bat, sup):
    f = _live(store, schedule=None)
    run = await runs.start(f["id"], BOB, watch=False)
    bat.runs[run["child_id"]].update(status="done", truth="Three offers.")
    store.set_settings({"enabled": False})
    assert (await sup.tick(fm._utcnow()))["refreshed"] == 1
    assert store.get_run(run["id"])["status"] == "done"


async def test_overlapping_scans_start_nothing_twice(store, bat, sup):
    f = _live(store)
    t1 = await _due_now(store, sup, f)
    first, second = await asyncio.gather(sup.tick(t1), sup.tick(t1))
    assert sorted([first.get("started", 0), second.get("started", 0)]) == [0, 1]
    assert {"busy": 1} in (first, second)
    assert len(_runs(store, f)) == 1


async def test_a_deck_that_never_used_first_mate_gets_no_file(tmp_path, monkeypatch):
    from captain_claw.flight_deck import events
    monkeypatch.setattr(events, "_STORE", None)
    monkeypatch.setattr(fm, "_STORE", None)
    monkeypatch.setenv("FD_DATA_DIR", str(tmp_path))
    out = await loop.Supervisor().tick()
    assert out["started"] == 0 and not (tmp_path / "first_mate.db").exists() and fm._STORE is None


async def test_the_loop_stops_when_told(monkeypatch):
    calls = []

    async def fake_tick(self, now=None):
        calls.append(1)
        return {}

    monkeypatch.setattr(loop.Supervisor, "tick", fake_tick)
    monkeypatch.setattr(loop, "TICK_SECONDS", 0.01)
    stop = asyncio.Event()
    task = asyncio.create_task(loop.supervisor_loop(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert len(calls) >= 2
