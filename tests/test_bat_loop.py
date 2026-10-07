"""Bat Phase 2 — durable store + supervisor/driver.

Proves the durability contract with injected handlers (no real agents): plan →
checkpointed steps → retry → judge → terminal status, plus owner cancel, the
LLM-spend cap, lease takeover, and — the point of Phase 2 — a run resuming from
its checkpoints after a simulated crash/restart instead of redoing finished work.
"""

from __future__ import annotations

import uuid

import pytest

import captain_claw.flight_deck.bat_loop as _bl
from captain_claw.flight_deck.bat_loop import BatDriver, BatSupervisor, cancel_run
from captain_claw.flight_deck.bat_store import BatStore


@pytest.fixture(autouse=True)
def _default_handlers():
    """Pin bat_loop's handler seams to their defaults for these tests, so that
    importing bat_routes elsewhere in the session (which registers the real
    handlers) can't leak in and change the driver's default planner/judge."""
    saved = (_bl._ATTEMPT_RUNNER, _bl._PLANNER, _bl._JUDGE, _bl._ON_FINISH)
    _bl._ATTEMPT_RUNNER = _bl._default_attempt_runner
    _bl._PLANNER = _bl._default_planner
    _bl._JUDGE = _bl._default_judge
    _bl._ON_FINISH = None
    yield
    (_bl._ATTEMPT_RUNNER, _bl._PLANNER, _bl._JUDGE, _bl._ON_FINISH) = saved


@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    yield s
    await s.close()


def _rid() -> str:
    return f"bat_{uuid.uuid4().hex[:10]}"


async def _make_run(store, steps, *, cap: float = 0.0, status: str = "planning") -> str:
    rid = _rid()
    await store.create_run(
        run_id=rid, owner_id="u1", title="t", task="do the thing",
        config={"steps": steps}, llm_usd_cap=cap, status=status,
    )
    return rid


def _runner(script):
    """script: dict step_key -> list of result dicts consumed per attempt (last
    one repeats). Records the order of attempted step_keys in `.calls`."""
    calls: list[str] = []

    async def run(run, step):
        key = step["step_key"]
        calls.append(key)
        results = script.get(key, [{"ok": True, "output": f"{key} ok"}])
        idx = min(sum(1 for c in calls if c == key) - 1, len(results) - 1)
        return results[idx]

    run.calls = calls  # type: ignore[attr-defined]
    return run


# ── store ───────────────────────────────────────────────────────────────

async def test_store_run_step_event_roundtrip(store):
    rid = await _make_run(store, ["a", "b"])
    run = await store.get_run(rid)
    assert run and run["owner_id"] == "u1" and run["config"]["steps"] == ["a", "b"]
    assert await store.list_runs("u1")

    await store.seed_steps(rid, [{"step_key": "a", "title": "A", "seq": 0},
                                 {"step_key": "b", "title": "B", "seq": 1}])
    await store.seed_steps(rid, [{"step_key": "a", "title": "A", "seq": 0}])  # idempotent
    steps = await store.list_steps(rid)
    assert [s["step_key"] for s in steps] == ["a", "b"]
    assert all(s["status"] == "pending" for s in steps)

    await store.upsert_step(rid, "a", status="done", output="hi")
    assert (await store.list_steps(rid))[0]["status"] == "done"

    i0 = await store.append_event(rid, "phase", "running")
    i1 = await store.append_event(rid, "step", "A", agent="a", ok=True)
    assert (i0, i1) == (0, 1)
    evs = await store.list_events(rid, since=1)
    assert len(evs) == 1 and evs[0]["data"]["agent"] == "a"

    total = await store.bump_cost(rid, 0.25, 1000)
    assert total == pytest.approx(0.25)
    assert (await store.get_run(rid))["cumulative_tokens"] == 1000


async def test_lease_takeover_rules(store):
    # Distinct hostnames keep the cross-host path deterministic (the same-host
    # os.kill liveness probe is exercised separately by the restart test).
    rid = await _make_run(store, ["a"])
    assert await store.claim_lease(rid, pid=1000, host="deck-a", now=100.0)
    # a different, fresh holder on another deck is refused
    assert not await store.claim_lease(rid, pid=2000, host="deck-b", now=150.0)
    # the same holder re-claims
    assert await store.claim_lease(rid, pid=1000, host="deck-a", now=160.0)
    # once stale, anyone takes over
    assert await store.claim_lease(rid, pid=2000, host="deck-b", now=160.0 + 10_000)

    assert await store.heartbeat_lease(rid, pid=2000, host="deck-b", now=100_000.0)
    assert not await store.heartbeat_lease(rid, pid=9999, host="deck-b", now=100_001.0)

    await store.release_lease(rid, pid=2000, host="deck-b")
    adopt = await store.adoptable_runs(now=100_002.0)
    assert any(r["id"] == rid for r in adopt)  # empty lease ⇒ adoptable


async def test_adoptable_excludes_fresh_lease(store):
    rid = await _make_run(store, ["a"])
    await store.claim_lease(rid, pid=1234, host="h", now=500.0)
    assert not any(r["id"] == rid for r in await store.adoptable_runs(now=520.0))
    assert any(r["id"] == rid for r in await store.adoptable_runs(now=500.0 + 10_000))


async def test_demote_running_steps(store):
    rid = await _make_run(store, ["a"])
    await store.seed_steps(rid, [{"step_key": "a"}])
    await store.upsert_step(rid, "a", status="running", attempt=1)
    assert await store.demote_running_steps(rid) == 1
    assert (await store.list_steps(rid))[0]["status"] == "pending"


# ── driver ────────────────────────────────────────────────────────────

async def test_driver_happy_path_sums_cost_and_assembles_truth(store):
    rid = await _make_run(store, ["a", "b"])
    runner = _runner({
        "a": [{"ok": True, "output": "alpha", "usd": 0.10, "tokens": 100}],
        "b": [{"ok": True, "output": "beta", "usd": 0.20, "tokens": 200}],
    })
    status = await BatDriver(store, attempt_runner=runner).drive(rid)
    assert status == "done"
    run = await store.get_run(rid)
    assert run["status"] == "done"
    assert run["cumulative_usd"] == pytest.approx(0.30)
    assert run["cumulative_tokens"] == 300
    assert "alpha" in run["truth"] and "beta" in run["truth"]
    assert runner.calls == ["a", "b"]


async def test_driver_retries_then_succeeds(store):
    rid = await _make_run(store, ["a", "b"])
    runner = _runner({
        "a": [{"ok": False, "error": "boom"}, {"ok": False, "error": "boom"}, {"ok": True, "output": "a-ok"}],
        "b": [{"ok": True, "output": "b-ok"}],
    })
    status = await BatDriver(store, attempt_runner=runner).drive(rid)
    assert status == "done"
    assert runner.calls.count("a") == 3  # two fails + a success
    assert runner.calls.count("b") == 1  # done on round 1, never retried
    assert (await store.get_run(rid))["status"] == "done"


async def test_driver_permanent_failure_errors(store):
    rid = await _make_run(store, ["a"])
    runner = _runner({"a": [{"ok": False, "error": "nope"}]})
    status = await BatDriver(store, attempt_runner=runner, max_step_attempts=3).drive(rid)
    assert status == "error"
    assert runner.calls.count("a") == 3
    run = await store.get_run(rid)
    assert run["status"] == "error" and run["stopped_reason"]


async def test_driver_stops_on_cancel_before_next_step(store):
    rid = await _make_run(store, ["a", "b"])

    async def runner(run, step):
        if step["step_key"] == "a":
            await cancel_run(store, run["id"])  # owner presses stop during step a
            return {"ok": True, "output": "a done but cancelled after"}
        pytest.fail("step b must not run after cancel")

    status = await BatDriver(store, attempt_runner=runner).drive(rid)
    assert status == "cancelled"
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["a"]["status"] == "done" and steps["b"]["status"] == "pending"


async def test_driver_stops_when_llm_cap_reached(store):
    rid = await _make_run(store, ["a", "b"], cap=0.05)
    runner = _runner({"a": [{"ok": True, "output": "a", "usd": 0.10}]})  # one step blows the cap

    status = await BatDriver(store, attempt_runner=runner).drive(rid)
    assert status == "error"
    run = await store.get_run(rid)
    assert run["stopped_reason"] == "llm_usd_cap_reached"
    assert runner.calls == ["a"]  # b never attempted


# ── restart recovery (the point of Phase 2) ───────────────────────────

async def test_run_resumes_from_checkpoints_after_crash(store):
    """Simulate: a run got step 'a' done, then the FD died mid-'b' (step 'b'
    left 'running', lease held by a now-dead pid). A fresh supervisor must adopt
    it, demote the orphaned step, and finish WITHOUT re-running 'a'."""
    rid = await _make_run(store, ["a", "b", "c"], status="running")
    await store.seed_steps(rid, [{"step_key": "a", "seq": 0},
                                 {"step_key": "b", "seq": 1},
                                 {"step_key": "c", "seq": 2}])
    await store.upsert_step(rid, "a", status="done", output="a-from-before")
    await store.upsert_step(rid, "b", status="running", attempt=1)  # orphaned by the crash
    # a dead driver's stale lease
    await store.claim_lease(rid, pid=999_999, host="dead-host", now=1.0)

    runner = _runner({
        "b": [{"ok": True, "output": "b-fresh"}],
        "c": [{"ok": True, "output": "c-fresh"}],
    })
    sup = BatSupervisor(store, BatDriver(store, attempt_runner=runner))
    launched = await sup.tick()           # adopt the stale run
    assert launched == 1
    await sup.drain(timeout=5.0)

    run = await store.get_run(rid)
    assert run["status"] == "done"
    assert "a" not in runner.calls, "finished step 'a' must not be re-run"
    assert set(runner.calls) == {"b", "c"}
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["a"]["output"] == "a-from-before"  # checkpoint preserved
    assert "a-from-before" in run["truth"] and "b-fresh" in run["truth"]


async def test_supervisor_skips_a_freshly_leased_run(store):
    rid = await _make_run(store, ["a"], status="running")
    # Fresh lease held by another deck (heartbeat at real `now`, so tick's
    # real-time staleness check sees it as warm).
    await store.claim_lease(rid, pid=4242, host="other-deck")
    sup = BatSupervisor(store, BatDriver(store, attempt_runner=_runner({})))
    assert await sup.tick() == 0  # not adoptable while the lease is warm
