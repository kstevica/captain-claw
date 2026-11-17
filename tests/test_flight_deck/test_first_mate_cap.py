"""A factory's cap per run: a setting on the factory, changed without approval.

Pinned here:

* the cap in force is the factory's own, else its charter's, else the owner's
  default, and the page is told which and what applies when none is set;
* setting it changes no version and needs no approval; 0 clears it; it stays
  within 100k–50M tokens, or is no cap at all (-1);
* a new run and a Continue both use it;
* the week may have no limit either: no share to run out of, a factory parked
  over budget runs again, and the report says so.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from captain_claw.flight_deck import bat_loop
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.bat_store import BatStore
from captain_claw.flight_deck.first_mate_store import FirstMateError

BOB = "user-bob"
ME = {"id": BOB, "role": "user"}
DEFAULT = 3_000_000


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
    yield b
    await b.close()


def _live(store, **kw):
    f = store.create(BOB, title="Lead list", charter="Collect public leads with a reason they fit.", **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], BOB)


def test_whose_cap_is_in_force(store):
    f = _live(store)
    cap = runs.run_cap(store, store.get(f["id"]), store.live(f["id"]))
    assert cap == {"tokens": DEFAULT, "source": "default", "inherited": DEFAULT, "inherited_from": "default"}
    g = _live(store, budget={"run_tokens": 1_000_000})
    cap = runs.run_cap(store, store.get(g["id"]), store.live(g["id"]))
    assert cap == {"tokens": 1_000_000, "source": "charter", "inherited": 1_000_000, "inherited_from": "charter"}
    store.update_factory(g["id"], "owner", run_tokens=20_000_000)
    cap = runs.run_cap(store, store.get(g["id"]), store.live(g["id"]))
    assert cap == {"tokens": 20_000_000, "source": "factory", "inherited": 1_000_000, "inherited_from": "charter"}


async def test_setting_it_needs_no_approval(store):
    f = _live(store)
    got = await routes.patch_factory(f["id"], {"run_tokens": 15_000_000}, user=ME)
    assert got["run_tokens"] == 15_000_000 and got["version"] == 1 and got["draft_version"] == 0
    detail = await routes.get_factory(f["id"], user=ME)
    assert detail["run_cap"]["tokens"] == 15_000_000 and detail["run_cap"]["source"] == "factory"
    assert store.events(f["id"])[0]["event"] == "updated"
    assert (await routes.patch_factory(f["id"], {"run_tokens": 0}, user=ME))["run_tokens"] == 0
    for bad in (50_000, 60_000_000):
        with pytest.raises(HTTPException) as no:
            await routes.patch_factory(f["id"], {"run_tokens": bad}, user=ME)
        assert no.value.status_code == 400
    with pytest.raises(FirstMateError):
        store.update_factory(f["id"], "owner", run_tokens=-2)  # -1 is "no cap"; nothing else below 100k


async def test_runs_and_continue_use_it(store, bat):
    f = _live(store)
    store.update_factory(f["id"], "owner", run_tokens=10_000_000)
    run = await runs.start(f["id"], BOB, watch=False)
    assert (await bat.get_run(run["child_id"]))["config"]["token_cap"] == 10_000_000
    # stopped by that cap, then given more
    await bat.upsert_step(run["child_id"], "step-1", title="Collect", status="pending", seq=0)
    await bat.bump_cost(run["child_id"], 0.0, 10_000_100)
    await bat.set_status(run["child_id"], "error", stopped_reason="llm_token_cap_reached")
    await runs.refresh(run["id"])
    store.update_factory(f["id"], "owner", run_tokens=30_000_000)
    await runs.continue_run(run["id"], BOB)
    runs._unwatch(run["id"])
    assert (await bat.get_run(run["child_id"]))["config"]["token_cap"] == 10_000_100 + 30_000_000


# ── no limit ───────────────────────────────────────────────────────────

def test_no_limit_is_a_choice_for_the_week_and_per_run(store):
    store.set_prefs(BOB, {"weekly_tokens": fm.UNLIMITED, "run_tokens": fm.UNLIMITED})
    assert store.prefs(BOB)["weekly_tokens"] == fm.UNLIMITED and store.prefs(BOB)["run_tokens"] == fm.UNLIMITED
    for bad in ({"weekly_tokens": -2}, {"run_tokens": 5}):
        with pytest.raises(FirstMateError):
            store.set_prefs(BOB, bad)
    f = _live(store)
    cap = runs.run_cap(store, store.get(f["id"]), store.live(f["id"]))
    assert cap["tokens"] == fm.UNLIMITED and cap["inherited"] == fm.UNLIMITED  # the owner's default: no cap
    store.update_factory(f["id"], "owner", run_tokens=5_000_000)
    assert runs.token_cap_for(store, store.get(f["id"]), store.live(f["id"])) == 5_000_000
    store.update_factory(f["id"], "owner", run_tokens=fm.UNLIMITED)
    assert store.get(f["id"])["run_tokens"] == fm.UNLIMITED


async def test_a_run_with_no_cap(store, bat):
    f = _live(store)
    store.update_factory(f["id"], "owner", run_tokens=fm.UNLIMITED)
    run = await runs.start(f["id"], BOB, watch=False)
    assert (await bat.get_run(run["child_id"]))["config"]["token_cap"] == 0  # Bat: no cap
    # a weekly share still lowers it to what's left
    g = _live(store)
    store.update_factory(g["id"], "owner", run_tokens=fm.UNLIMITED)
    other = await runs.start(g["id"], BOB, watch=False, max_tokens=4_000_000)
    assert (await bat.get_run(other["child_id"]))["config"]["token_cap"] == 4_000_000
    # and Continue goes on with no cap
    await bat.upsert_step(run["child_id"], "step-1", title="Collect", status="pending", seq=0)
    await bat.set_status(run["child_id"], "cancelled", stopped_reason="owner_cancel")
    await runs.refresh(run["id"])
    await runs.continue_run(run["id"], BOB)
    runs._unwatch(run["id"])
    assert (await bat.get_run(run["child_id"]))["config"]["token_cap"] == 0


async def test_a_week_with_no_limit(store, monkeypatch):
    from captain_claw.flight_deck import first_mate_loop as loop
    from captain_claw.flight_deck import first_mate_report as report
    from captain_claw.flight_deck import line

    async def fan_out(user_id, item, to_phones=True, rung=False):
        return {}
    monkeypatch.setattr(line, "fan_out", fan_out)
    f = _live(store)
    store.set_prefs(BOB, {"weekly_tokens": fm.UNLIMITED})
    assert loop.shares(store.prefs(BOB), store.list_factories(BOB)) == {f["id"]: fm.UNLIMITED}
    assert loop._lower_to(fm.UNLIMITED) is None and loop._lower_to(0) is None and loop._lower_to(7) == 7
    # a factory parked over budget runs again once the week has no limit
    store.set_state(f["id"], "paused_budget", "first_mate", "it used its share")
    out = {"paused": 0, "resumed": 0}
    now = fm._utcnow()
    left = await loop.Supervisor()._budget(store, store.get(f["id"]), fm.UNLIMITED, loop.week_start(now), out)
    assert left == fm.UNLIMITED and out["resumed"] == 1 and store.get(f["id"])["state"] == "running"
    week = report.build(store, BOB, loop.week_start(now), now)
    title, body = report.text(week)
    assert "(no weekly limit)" in body and not any("of the week's" in x for x in week["look"])
