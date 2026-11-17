"""Continue: a stopped run goes on where it stopped.

Pinned here:

* Bat reopens a run its token (or dollar) cap or its owner stopped: finished
  steps stay finished, only what's left runs, with the new cap. A run that
  stopped on its own, finished, has nothing left, or still has a live driver
  doesn't reopen;
* First Mate's Continue does that for the factory's own runs and rehearsals,
  with the live version's cap per run on top of what it used (so raising the
  cap first gives it more), notes it in the history, and says in words why a
  cap stopped it;
* it refuses a carry-out, a charter that changed since (a new cap alone is
  fine), a factory with another run going, and other people's runs, leaving
  the Bat run as it was;
* the run list says whether the newest stopped run can go on.
"""

from __future__ import annotations

import time

import pytest
from fastapi import HTTPException

from captain_claw.flight_deck import bat_loop
from captain_claw.flight_deck import first_mate_routes as routes
from captain_claw.flight_deck import first_mate_runs as runs
from captain_claw.flight_deck import first_mate_store as fm
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_store import BatStore
from captain_claw.flight_deck.first_mate_store import FirstMateError

BOB, CAROL = "user-bob", "user-carol"
CAP = 3_000_000


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
    kicks: list[int] = []

    async def kick():
        kicks.append(1)
        return 0

    saved = (bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE)
    bat_loop.set_gate_check(None)
    bat_loop._PLANNER = bat_loop._default_planner
    bat_loop._JUDGE = bat_loop._default_judge
    monkeypatch.setattr(bat_loop, "_STORE", b)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    b.kicks = kicks
    yield b
    bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE = saved
    await b.close()


STEPS = [("step-1", "Find sources", "done"), ("step-2", "Collect the list", "done"),
         ("step-3", "Check each one", "pending"), ("step-4", "Write the list", "pending")]


async def _capped(bat, bid, *, reason="llm_token_cap_reached", status="error", used=CAP + 5_000, steps=STEPS):
    """A Bat run stopped the way the cap stops one: some steps finished."""
    for n, (key, title, st) in enumerate(steps):
        await bat.upsert_step(bid, key, title=title, status=st, seq=n, output="done it" if st == "done" else None)
    await bat.bump_cost(bid, 0.0, used)
    await bat.set_status(bid, status, stopped_reason=reason)


def _live(store, owner=BOB, **kw):
    kw.setdefault("title", "Lead list")
    kw.setdefault("charter", "Collect public leads with a name, a public contact and a reason they fit.")
    f = store.create(owner, **kw)
    return store.approve(f["id"], 1, f["draft"]["sha"], owner)


async def _stopped_run(store, bat, **kw):
    f = _live(store)
    run = await runs.start(f["id"], BOB, watch=False)
    await _capped(bat, run["child_id"], **kw)
    return f, await runs.refresh(run["id"])


# ── Bat ────────────────────────────────────────────────────────────────

async def test_bat_reopens_a_capped_run_where_it_stopped(bat):
    await bat.create_run(run_id="bat_a", owner_id=BOB, title="t", task="t", config={"token_cap": CAP},
                         status="running")
    await _capped(bat, "bat_a")
    assert await bat_loop.why_not_reopen(bat, "bat_a") == ""
    run = await bat_loop.reopen_run(bat, "bat_a", token_cap=2 * CAP)
    assert (run["status"], run["stopped_reason"], run["config"]["token_cap"]) == ("running", "", 2 * CAP)
    assert bat.kicks == [1]
    ran: list[str] = []

    async def runner(run, step):
        ran.append(step["step_key"])
        return {"ok": True, "output": "did it"}

    assert await BatDriver(bat, attempt_runner=runner).drive("bat_a") == "done"
    assert sorted(ran) == ["step-3", "step-4"]  # the finished steps stay finished


async def test_bat_wont_reopen_what_would_stop_again(bat):
    for rid, status, reason, steps in (
            ("bat_own", "error", "the reviewers wouldn't pass it", STEPS),
            ("bat_done", "done", "", STEPS),
            ("bat_all", "error", "llm_token_cap_reached", [(k, t, "done") for k, t, _ in STEPS])):
        await bat.create_run(run_id=rid, owner_id=BOB, title="t", task="t", config={}, status="running")
        await _capped(bat, rid, status=status, reason=reason, steps=steps)
    assert "stopped on its own" in await bat_loop.why_not_reopen(bat, "bat_own")
    assert await bat_loop.why_not_reopen(bat, "bat_done") == "it finished"
    assert await bat_loop.why_not_reopen(bat, "bat_all") == "it has no step left to do"
    with pytest.raises(ValueError, match="stopped on its own"):
        await bat_loop.reopen_run(bat, "bat_own", token_cap=CAP)
    assert (await bat.get_run("bat_own"))["status"] == "error"
    # stopped by its owner: it may go on, once its driver has let go
    await bat.create_run(run_id="bat_stop", owner_id=BOB, title="t", task="t", config={}, status="running")
    await _capped(bat, "bat_stop", status="cancelled", reason="owner_cancel")
    assert await bat.claim_lease("bat_stop", pid=4242, host="elsewhere", now=time.time())
    assert "winding down" in await bat_loop.why_not_reopen(bat, "bat_stop")
    await bat.release_lease("bat_stop", pid=4242, host="elsewhere")
    assert await bat_loop.why_not_reopen(bat, "bat_stop") == ""


# ── First Mate ─────────────────────────────────────────────────────────

async def test_continue_gives_it_the_cap_again_on_top_of_what_it_used(store, bat):
    f, run = await _stopped_run(store, bat)
    assert run["status"] == "failed" and run["error"] == runs.STOP_WORDS["llm_token_cap_reached"]
    [listed] = (await routes.list_runs(f["id"], user={"id": BOB, "role": "user"}))["runs"]
    assert listed["can_continue"] is True
    [shown] = (await routes.get_factory(f["id"], user={"id": BOB, "role": "user"}))["runs"]  # the drawer's
    assert shown["can_continue"] is True
    got = await runs.continue_run(run["id"], BOB)
    runs._unwatch(run["id"])
    assert (got["status"], got["error"], got["result"]) == ("running", "", "")
    b = await bat.get_run(run["child_id"])
    assert b["status"] == "running" and b["config"]["token_cap"] == CAP + 5_000 + CAP
    assert store.events(f["id"])[0]["event"] == "run_continued"
    # it is going again: the list says so, and a second Continue is refused
    with pytest.raises(FirstMateError, match="only a run that stopped"):
        await runs.continue_run(run["id"], BOB)


async def test_a_higher_cap_first_gives_it_more(store, bat):
    f, run = await _stopped_run(store, bat)
    d = store.new_draft(f["id"], BOB, budget={"run_tokens": 10_000_000})
    store.approve(f["id"], 2, d["draft"]["sha"], BOB)
    await runs.continue_run(run["id"], BOB)
    runs._unwatch(run["id"])
    assert (await bat.get_run(run["child_id"]))["config"]["token_cap"] == CAP + 5_000 + 10_000_000


async def test_what_continue_refuses(store, bat):
    # the charter changed since it ran
    f, run = await _stopped_run(store, bat)
    d = store.new_draft(f["id"], BOB, charter="Collect public leads in Canada only, with a reason they fit.")
    store.approve(f["id"], 2, d["draft"]["sha"], BOB)
    with pytest.raises(FirstMateError, match="charter changed"):
        await runs.continue_run(run["id"], BOB)
    assert (await bat.get_run(run["child_id"]))["status"] == "error"  # left as it was
    # it stopped on its own
    g, run2 = await _stopped_run(store, bat, reason="the reviewers wouldn't pass it")
    assert "stopped on its own" in await runs.why_not_continue(run2)
    # another run of the factory is going
    h, run3 = await _stopped_run(store, bat)
    going = store.start_run(h["id"])
    with pytest.raises(FirstMateError, match="already has a run going"):
        await runs.continue_run(run3["id"], BOB)
    assert (await bat.get_run(run3["child_id"]))["status"] == "error"
    store.finish_run(going["id"], "cancelled")
    # someone else's run
    with pytest.raises(FirstMateError, match="no such run"):
        await runs.continue_run(run3["id"], CAROL)
    with pytest.raises(HTTPException) as no:
        await routes.continue_run(run3["id"], user={"id": CAROL, "role": "user"})
    assert no.value.status_code == 404
    # a carry-out never goes on
    carry = {**run3, "kind": "carry_out"}
    assert "only a factory's own run or rehearsal" in await runs.why_not_continue(carry)
