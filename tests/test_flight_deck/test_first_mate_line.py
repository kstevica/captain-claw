"""First Mate on the Line: one thread per factory.

Pinned here:

* a finished run posts once, on its factory's thread, as First Mate, not
  "needs you" and not to phones; a failed one says why; a stopped one says nothing;
* an automatic run that couldn't start is told; a manual one isn't (its owner saw it);
* a failure-streak pause is "needs you" and goes to phones; resuming settles it;
* a budget pause is a plain item;
* Bat says nothing itself about a First Mate run (each run is told once),
  and still tells about its own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, bat_routes, events, line
from captain_claw.flight_deck import first_mate_line as fml
from captain_claw.flight_deck import first_mate_loop as loop
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.db import FlightDeckDB
from captain_claw.flight_deck.first_mate_store import FirstMateError

BOB = "user-bob"


class FakeBat:
    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.fail_create = False

    async def create_run(self, *, run_id, owner_id, title, task, config=None, origin=None,
                         source_host="localhost", source_port=0, llm_usd_cap=0.0, status="planning"):
        if self.fail_create:
            raise RuntimeError("bat.db is locked")
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


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    """A rehearsal's leak check reads the tool-call register; a test never opens one
    (importing the deck server points it at a data folder)."""
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
async def db(tmp_path: Path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        await d.close()
        fd_auth._db = prev


@pytest.fixture
def store(tmp_path, monkeypatch, db):
    monkeypatch.setattr(events, "_STORE", events.EventsStore(tmp_path / "events.db"))
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
def phones(monkeypatch):
    """What reached the fan-out, with its to_phones."""
    sent: list[tuple[str, bool]] = []

    async def fake_fan_out(user_id, item, to_phones=True, rung=False):
        sent.append((item.get("title", ""), to_phones))
        return {}

    monkeypatch.setattr(line, "fan_out", fake_fan_out)
    monkeypatch.setattr(line, "_fan_out_on", True)
    return sent


def _live(store, **kw):
    kw.setdefault("title", "Weekly report")
    kw.setdefault("charter", "Write one page on last week.")
    f = store.create(BOB, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


async def _items(db):
    await asyncio.sleep(0)
    return await db.list_line_items(BOB, limit=50)


async def test_a_finished_run_posts_once_on_its_thread(store, bat, db, phones):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bat.runs[run["child_id"]].update(status="done", truth="## Week\nAll quiet.")
    await runs.refresh(run["id"])
    await runs.refresh(run["id"])
    [item] = await _items(db)
    assert item["title"] == "Weekly report: rehearsed" and "All quiet." in item["body"]  # trust: propose
    assert "It needed nothing outside." in item["body"]
    assert item["thread_id"] == f"fm:{f['id']}" and item["source"] == "First Mate"
    assert item["ref_type"] == "first_mate_run" and item["ref_id"] == run["id"] and not item["needs_you"]
    assert phones == [("Weekly report: rehearsed", False)]


async def test_a_failed_run_says_why_and_a_stopped_one_says_nothing(store, bat, db, phones):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bat.runs[run["child_id"]].update(status="error", stopped_reason="the reviewers wouldn't pass it")
    await runs.refresh(run["id"])
    other = await runs.start(f["id"], BOB, watch=False)
    await runs.stop(other["id"], BOB)
    [item] = await _items(db)
    assert item["title"] == "Weekly report: the run failed" and "reviewers" in item["body"]
    assert not item["needs_you"]


async def test_only_an_automatic_run_that_couldnt_start_is_told(store, bat, db):
    f = _live(store)
    bat.fail_create = True
    with pytest.raises(FirstMateError):
        await runs.start(f["id"], BOB, watch=False)
    assert await _items(db) == []
    with pytest.raises(FirstMateError):
        await runs.start(f["id"], BOB, trigger="schedule", watch=False)
    [item] = await _items(db)
    assert "couldn't start" in item["body"]


async def test_a_streak_pause_needs_you_and_resuming_settles_it(store, bat, db, phones):
    f = _live(store)
    for _ in range(loop.FAILURE_STREAK):
        r = store.start_run(f["id"])
        store.finish_run(r["id"], "failed")
    await loop.Supervisor().tick(fm._utcnow())
    [item] = await _items(db)
    assert item["title"] == "Weekly report is paused" and item["needs_you"]
    assert item["ref_type"] == "first_mate_factory" and item["thread_id"] == f"fm:{f['id']}"
    assert phones == [("Weekly report is paused", True)]
    await fml.factory_resumed(BOB, f["id"])
    [item] = await _items(db)
    assert item["resolved_at"]


async def test_a_budget_pause_is_a_plain_item(store, bat, db):
    store.set_prefs(BOB, {"weekly_tokens": 1_000_000})
    f = _live(store)
    r = store.start_run(f["id"])
    store.finish_run(r["id"], "done", tokens=1_000_000)
    await loop.Supervisor().tick(fm._utcnow())
    [item] = await _items(db)
    assert item["title"] == "Weekly report is paused" and not item["needs_you"]
    assert "tokens" in item["body"]


async def test_bat_stays_quiet_about_a_first_mate_run(db, monkeypatch):
    async def no_cost(run):
        return None

    monkeypatch.setattr(bat_routes, "_log_own_cost", no_cost)
    base = {"owner_id": BOB, "status": "done", "title": "Weekly report", "task": "t", "truth": "ok",
            "origin": {}, "source_host": "localhost", "source_port": 0}
    await bat_routes._bat_on_finish({**base, "id": "bat_fm", "config": {"source": "first_mate"}})
    assert await _items(db) == []
    await bat_routes._bat_on_finish({**base, "id": "bat_own", "config": {"source": "ui"}})
    [item] = await _items(db)
    assert item["ref_id"] == "bat_own"
