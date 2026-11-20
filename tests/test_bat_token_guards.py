"""Bounds on a run's tokens that hold when its owner set no cap.

A list funnel run whose factory had "no cap" went to 54.7M tokens overnight for 20
rows: its finishing agent used ~6M and was retried after timing out, and fix rounds
ran agent steps of 3 to 16M each, retrying the ones that came back short. Pinned here:

* a run with no token cap stops at the deck's safety ceiling, as a cap would (so it
  can be continued), and a run continued with no cap counts the ceiling from there;
  a run with a cap of its own is held to that cap only;
* an agent step that uses its share of tokens is stopped, counted, and done with
  what it made; the run goes on;
* a list funnel's agent steps (its finishing step, a fix round's) are never run again
  after timing out or coming back short: what they made is kept.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

import captain_claw.flight_deck.bat_loop as _bl
from captain_claw.flight_deck import bat_nested
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_store import BatStore


@pytest.fixture(autouse=True)
def _default_handlers(monkeypatch):
    saved = (_bl._ATTEMPT_RUNNER, _bl._PLANNER, _bl._JUDGE, _bl._ON_FINISH, _bl._GATE_CHECK,
             _bl._REPLANNER, _bl._POLL_SPEND, _bl._STOP_CHILDREN)
    _bl._ATTEMPT_RUNNER = _bl._default_attempt_runner
    _bl._PLANNER = _bl._default_planner
    _bl._JUDGE = _bl._default_judge
    _bl._ON_FINISH = None
    _bl._GATE_CHECK = None
    _bl._REPLANNER = _bl._POLL_SPEND = _bl._STOP_CHILDREN = None
    monkeypatch.setattr(_bl, "_STEP_CHECK", None)
    monkeypatch.setattr(_bl, "_CAP_REFRESH", None)
    monkeypatch.setattr(_bl, "_WATCH_INTERVAL_S", 0.05)
    monkeypatch.delenv("CLAW_BAT_TOKEN_CEILING", raising=False)
    yield
    (_bl._ATTEMPT_RUNNER, _bl._PLANNER, _bl._JUDGE, _bl._ON_FINISH, _bl._GATE_CHECK,
     _bl._REPLANNER, _bl._POLL_SPEND, _bl._STOP_CHILDREN) = saved


@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    yield s
    await s.close()


async def _make_run(store, steps, **config) -> str:
    rid = f"bat_{uuid.uuid4().hex[:10]}"
    await store.create_run(run_id=rid, owner_id="u1", title="t", task="do the thing",
                           config={"steps": steps, **config}, llm_usd_cap=0.0, status="planning")
    return rid


def _runner(script):
    calls: list[str] = []

    async def run(run, step):
        key = step["step_key"]
        calls.append(key)
        results = script.get(key, [{"ok": True, "output": f"{key} ok"}])
        got = results[min(sum(1 for c in calls if c == key) - 1, len(results) - 1)]
        return await got(run, step) if callable(got) else got

    run.calls = calls  # type: ignore[attr-defined]
    return run


async def _notes(store, rid) -> list[str]:
    return [e["message"] for e in await store.list_events(rid) if e["stage"] == "note"]


async def test_a_run_with_no_cap_stops_at_the_safety_ceiling(store, monkeypatch):
    monkeypatch.setenv("CLAW_BAT_TOKEN_CEILING", "500")
    rid = await _make_run(store, ["a", "b"])
    runner = _runner({"a": [{"ok": True, "output": "a", "tokens": 600}]})
    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "error"
    run = await store.get_run(rid)
    assert run["stopped_reason"] == "llm_token_cap_reached"  # continuable, as a cap is
    assert runner.calls == ["a"]
    assert any("no token cap, so it stopped at the deck's safety ceiling of 500 tokens" in n
               for n in await _notes(store, rid))
    # a cap of its own: held to that, not to the ceiling
    capped = await _make_run(store, ["a", "b"], token_cap=10_000)
    assert await BatDriver(store, attempt_runner=_runner({"a": [{"ok": True, "output": "a", "tokens": 600}]})
                           ).drive(capped) == "done"
    # the ceiling switched off
    monkeypatch.setenv("CLAW_BAT_TOKEN_CEILING", "0")
    off = await _make_run(store, ["a", "b"])
    assert await BatDriver(store, attempt_runner=_runner({"a": [{"ok": True, "output": "a", "tokens": 600}]})
                           ).drive(off) == "done"


def test_the_ceiling_counts_from_where_a_run_was_continued(monkeypatch):
    monkeypatch.setenv("CLAW_BAT_TOKEN_CEILING", "1000")
    assert _bl.ceiling_for({"config": {}}) == 1000
    assert _bl.ceiling_for({"config": {"cap_base": 4000}}) == 5000
    assert _bl.ceiling_for({"config": {"token_cap": 3000}}) == 0  # its own cap applies


async def test_continuing_a_run_with_no_cap_restarts_the_ceiling(store, monkeypatch):
    monkeypatch.setenv("CLAW_BAT_TOKEN_CEILING", "500")
    rid = await _make_run(store, ["a", "b"])
    await BatDriver(store, attempt_runner=_runner({"a": [{"ok": True, "output": "a", "tokens": 600}]})).drive(rid)

    async def no_kick():
        return 0

    monkeypatch.setattr(_bl, "kick", no_kick)
    monkeypatch.setattr(_bl, "LEASE_STALE_SECONDS", 0)
    run = await _bl.reopen_run(store, rid)
    assert run["config"]["cap_base"] == 600 and _bl.ceiling_for(run) == 1100


def test_each_kind_of_run_gets_its_step_share(monkeypatch):
    monkeypatch.setenv("CLAW_BAT_TOKEN_CEILING", "21000000")
    assert _bl.step_token_limit({"config": {"step_token_limit": 1234}}) == 1234
    assert _bl.step_token_limit({"config": {"engine": "funnel", "token_cap": 3_000_000}}) == _bl.FUNNEL_STEP_TOKENS
    assert _bl.step_token_limit({"config": {}}) == 7_000_000  # no cap: a third of the ceiling
    assert _bl.step_token_limit({"config": {"token_cap": 3_000_000}}) == 0  # a capped Bat run: its cap bounds it


async def test_a_step_past_its_share_is_stopped_with_its_work_kept(store):
    rid = await _make_run(store, ["a", "b"], step_token_limit=1000)

    async def long_turn(run, step):
        bat_nested.set_live(run["id"], step["step_key"], 0.0, 5000)  # what its turn has used so far
        try:
            await asyncio.sleep(30)
        finally:
            bat_nested.worker_ended(run["id"], step["step_key"])
        return {"ok": True, "output": "never gets here"}

    runner = _runner({"a": [long_turn]})
    assert await asyncio.wait_for(BatDriver(store, attempt_runner=runner).drive(rid), 10) == "done"
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["a"]["status"] == "done" and "share of 1,000 tokens (used 5,000)" in steps["a"]["output"]
    assert steps["b"]["status"] == "done" and runner.calls == ["a", "b"]  # the run went on; a wasn't run again


async def test_a_funnel_runs_agent_step_keeps_what_it_made(store, monkeypatch):
    rid = await _make_run(store, ["work", "finish", "merge"], engine="funnel",
                          funnel_steps={"work": {"kind": "work"}, "finish": {"kind": "finish"},
                                        "merge": {"kind": "merge"}})
    runner = _runner({"finish": [{"ok": False, "error": "timed out", "output": "found 1 row"}]})
    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert runner.calls == ["work", "finish", "merge"]  # never a second try at the same cost
    finish = next(s for s in await store.list_steps(rid) if s["step_key"] == "finish")
    assert finish["status"] == "done" and finish["output"].startswith("Stopped (timed out); what it made so far is kept")
    # a step check that would send it round again keeps it instead
    redo = await _make_run(store, ["finish", "merge"], engine="funnel",
                           funnel_steps={"finish": {"kind": "finish"}, "merge": {"kind": "merge"}})

    async def short(run, step, started_at, last):
        return {"redo": True, "error": "came back short: 1 of the 10"} if step["step_key"] == "finish" else None

    monkeypatch.setattr(_bl, "_STEP_CHECK", short)
    runner2 = _runner({})
    assert await BatDriver(store, attempt_runner=runner2).drive(redo) == "done"
    assert runner2.calls == ["finish", "merge"]
    assert any("kept as it is" in n for n in await _notes(store, redo))
    # any other run's steps still try again
    monkeypatch.setattr(_bl, "_STEP_CHECK", None)
    plain = await _make_run(store, ["finish"])
    runner3 = _runner({"finish": [{"ok": False, "error": "timed out"}, {"ok": True, "output": "ok"}]})
    assert await BatDriver(store, attempt_runner=runner3).drive(plain) == "done" and runner3.calls == ["finish", "finish"]
