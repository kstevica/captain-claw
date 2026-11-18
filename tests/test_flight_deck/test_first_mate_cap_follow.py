"""A going run follows its factory's cap, and Bat's deck settings come from the
admin console.

Pinned here:

* a factory run records the cap setting it starts under; when the owner changes
  the factory's cap (or their default), its going run moves with it: base plus
  the new cap, or no cap; a weekly-lowered run keeps its lower cap until the
  setting changes; an improvement run keeps its own budget; a run from before
  this (no recorded setting) is only ever raised;
* a continued run counts its cap from what it had used when it went on;
* Bat reads the cap again right before stopping on it, so a raise in between
  keeps the run going;
* changing a factory's cap through the page moves its going run at once;
* Bat's deck settings: kept in the deck, checked, in force at once (steps at the
  same time, a worker turn's ceiling, the quiet that ends a step), with the
  FD_BAT_* values as defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw.flight_deck import auth as fd_auth
from captain_claw.flight_deck import bat_loop, bat_routes
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.auth import set_auth_db
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_store import BatStore
from captain_claw.flight_deck.db import FlightDeckDB

BOB = "user-bob"
ME = {"id": BOB, "role": "user"}


@pytest.fixture(autouse=True)
def _no_register(monkeypatch):
    from captain_claw.flight_deck import tool_audit_routes
    monkeypatch.setattr(tool_audit_routes, "_data_dir", None)
    monkeypatch.setattr(tool_audit_routes, "_store", None)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = fm.FirstMateStore(tmp_path / "first_mate.db")
    monkeypatch.setattr(fm, "_STORE", s)
    s.set_settings({"enabled": True, "admins_only": False})
    return s


@pytest.fixture
async def bat(tmp_path, monkeypatch):
    b = BatStore(tmp_path / "bat.db")
    await b.init()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", b)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    yield b
    await b.close()


def _live(store):
    f = store.create(BOB, title="Lead list", charter="Collect public leads with a reason they fit.")
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


async def _cfg(bat, bid):
    return (await bat.get_run(bid))["config"]


async def test_a_going_run_follows_its_factorys_cap(store, bat):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bid = run["child_id"]
    cfg = await _cfg(bat, bid)
    assert (cfg["token_cap"], cfg["cap_setting"], cfg["cap_base"]) == (3_000_000, 3_000_000, 0)
    assert await runs.sync_cap(await bat.get_run(bid)) is False  # nothing changed
    store.update_factory(f["id"], "owner", run_tokens=20_000_000)
    assert await runs.sync_cap(await bat.get_run(bid)) is True
    assert (await _cfg(bat, bid))["token_cap"] == 20_000_000
    store.update_factory(f["id"], "owner", run_tokens=fm.UNLIMITED)
    assert await runs.sync_cap(await bat.get_run(bid)) is True
    assert (await _cfg(bat, bid))["token_cap"] == 0  # no cap
    notes = [e["message"] for e in await bat.list_events(bid) if e["stage"] == "note"]
    assert any("no token cap now" in n for n in notes)


async def test_what_keeps_its_own_cap(store, bat):
    f = _live(store)
    lowered = await runs.start(f["id"], BOB, watch=False, max_tokens=1_000_000)  # its weekly share was left
    assert (await _cfg(bat, lowered["child_id"]))["token_cap"] == 1_000_000
    assert await runs.sync_cap(await bat.get_run(lowered["child_id"])) is False  # setting unchanged: stays lowered
    swarm = {"id": "bat_s", "config": {"factory_id": f["id"], "source": "captain_swarm", "token_cap": 500}}
    assert await runs.sync_cap(swarm) is False
    # a run from before caps were followed: only ever raised
    await bat.create_run(run_id="bat_old", owner_id=BOB, title="t", task="t", status="running",
                         config={"factory_id": f["id"], "token_cap": 5_000_000})
    store.update_factory(f["id"], "owner", run_tokens=2_000_000)
    assert await runs.sync_cap(await bat.get_run("bat_old")) is False
    assert (await _cfg(bat, "bat_old"))["token_cap"] == 5_000_000
    store.update_factory(f["id"], "owner", run_tokens=fm.UNLIMITED)
    assert await runs.sync_cap(await bat.get_run("bat_old")) is True
    assert (await _cfg(bat, "bat_old"))["token_cap"] == 0


async def test_a_continued_run_counts_from_where_it_went_on(store, bat):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    bid = run["child_id"]
    await bat.upsert_step(bid, "step-1", title="Collect", status="pending", seq=0)
    await bat.bump_cost(bid, 0.0, 3_000_100)
    await bat.set_status(bid, "error", stopped_reason="llm_token_cap_reached")
    await runs.refresh(run["id"])
    await runs.continue_run(run["id"], BOB)
    runs._unwatch(run["id"])
    cfg = await _cfg(bat, bid)
    assert (cfg["token_cap"], cfg["cap_base"], cfg["cap_setting"]) == (6_000_100, 3_000_100, 3_000_000)
    store.update_factory(f["id"], "owner", run_tokens=10_000_000)
    await runs.sync_cap(await bat.get_run(bid))
    assert (await _cfg(bat, bid))["token_cap"] == 3_000_100 + 10_000_000


async def test_bat_reads_the_cap_again_before_stopping(store, bat):
    f = _live(store)
    await bat.create_run(run_id="bat_c", owner_id=BOB, title="t", task="t", status="running",
                         config={"factory_id": f["id"], "token_cap": 3_000_000, "cap_setting": 3_000_000,
                                 "cap_base": 0, "steps": []})
    await bat.upsert_step("bat_c", "a", title="A", status="pending", seq=0)
    await bat.bump_cost("bat_c", 0.0, 3_000_500)  # over its cap already
    store.update_factory(f["id"], "owner", run_tokens=fm.UNLIMITED)  # the owner took the cap off meanwhile
    bat_loop.set_cap_refresh(bat_routes._cap_refresh)

    async def runner(_run, _step):
        return {"ok": True, "output": "done it"}

    async def judge(_run, steps):
        return {"done": True, "reason": "good"}

    try:
        assert await BatDriver(bat, attempt_runner=runner, judge=judge).drive("bat_c") == "done"
    finally:
        bat_loop.set_cap_refresh(bat_routes._cap_refresh)
    bat_loop.set_cap_refresh(None)
    await bat.create_run(run_id="bat_d", owner_id=BOB, title="t", task="t", status="running",
                         config={"token_cap": 100, "steps": []})
    await bat.upsert_step("bat_d", "a", title="A", status="pending", seq=0)
    await bat.bump_cost("bat_d", 0.0, 500)
    assert await BatDriver(bat, attempt_runner=runner, judge=judge).drive("bat_d") == "error"
    bat_loop.set_cap_refresh(bat_routes._cap_refresh)


async def test_changing_the_cap_on_the_page_moves_the_going_run(store, bat):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    await routes.patch_factory(f["id"], {"run_tokens": 15_000_000}, user=ME)
    assert (await _cfg(bat, run["child_id"]))["token_cap"] == 15_000_000
    await routes.patch_factory(f["id"], {"run_tokens": 0}, user=ME)  # back to the default
    assert (await _cfg(bat, run["child_id"]))["token_cap"] == 3_000_000


# ── Bat's deck settings ────────────────────────────────────────────────

@pytest.fixture
async def deck_db(tmp_path: Path):
    prev = fd_auth._db
    d = FlightDeckDB(tmp_path / "fd.db")
    await d.init()
    set_auth_db(d)
    try:
        yield d
    finally:
        bat_routes._apply_settings({})
        await d.close()
        fd_auth._db = prev


async def test_bats_deck_settings(deck_db, monkeypatch):
    monkeypatch.delenv("FD_BAT_MAX_PARALLEL", raising=False)
    assert bat_routes.deck_settings() == {"max_parallel": 3, "step_ceiling": bat_routes._BAT_STEP_CEILING,
                                          "idle_minutes": int(bat_routes._BAT_IDLE_S // 60)}
    got = await bat_routes.save_settings({"max_parallel": 1, "idle_minutes": 30})
    assert got["max_parallel"] == 1 and got["idle_minutes"] == 30
    assert bat_loop._max_parallel() == 1 and bat_routes._idle_s() == 1800.0
    env = {e["key"]: e["value"] for e in bat_routes._bat_env("bat_x", "step-1", BOB)}
    assert env["CLAW_FLOATING_BUDGET"] == str(bat_routes._BAT_STEP_CEILING)
    await bat_routes.save_settings({"step_ceiling": 20_000})
    env = {e["key"]: e["value"] for e in bat_routes._bat_env("bat_x", "step-1", BOB)}
    assert env["CLAW_FLOATING_BUDGET"] == "20000"
    for bad in ({"max_parallel": 0}, {"max_parallel": 9}, {"idle_minutes": "15"}, {"nope": 1}, {"max_parallel": True}):
        with pytest.raises(ValueError):
            await bat_routes.save_settings(bad)
    # kept in the deck: a restart takes them up again
    bat_routes._apply_settings({})
    assert bat_loop._max_parallel() == 3
    assert (await bat_routes.load_settings())["max_parallel"] == 1 and bat_loop._max_parallel() == 1
    view = await bat_routes.get_deck_settings(admin={"id": "admin", "role": "admin"})
    assert view["defaults"]["max_parallel"] == 3 and view["ranges"]["max_parallel"] == [1, 8]
