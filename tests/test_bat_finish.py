"""Bat finishes: a refusal becomes new work, the plan runs in parallel where it can,
a check Captain runs itself decides, and the budget covers the whole run.

The driver tests use injected handlers (no agents), like test_bat_loop. The route
tests stub the model and the database, and use a real folder for the checks.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

import captain_claw.flight_deck.bat_loop as bl
from captain_claw.flight_deck import bat_loop, bat_nested, bat_routes, human_ask
from captain_claw.flight_deck.bat_loop import BatDriver, cancel_run
from captain_claw.flight_deck.bat_store import BatStore


@pytest.fixture(autouse=True)
def _handlers(monkeypatch):
    """No real handlers unless a test sets them; look up often so tests are quick."""
    saved = (bl._ATTEMPT_RUNNER, bl._PLANNER, bl._JUDGE, bl._ON_FINISH, bl._GATE_CHECK,
             bl._REPLANNER, bl._POLL_SPEND, bl._STOP_CHILDREN, human_ask._NOTIFY)
    bl._ATTEMPT_RUNNER = bl._default_attempt_runner
    bl._PLANNER = bl._default_planner
    bl._JUDGE = bl._default_judge
    bl._ON_FINISH = bl._GATE_CHECK = bl._REPLANNER = bl._POLL_SPEND = bl._STOP_CHILDREN = None

    async def quiet(_ask):
        return None

    human_ask.set_notifier(quiet)
    monkeypatch.setattr(bl, "_WATCH_INTERVAL_S", 0.01)
    monkeypatch.delenv("FD_BAT_MAX_PARALLEL", raising=False)
    bat_nested._WORKERS.clear()
    bat_nested._LIVE.clear()
    yield
    (bl._ATTEMPT_RUNNER, bl._PLANNER, bl._JUDGE, bl._ON_FINISH, bl._GATE_CHECK,
     bl._REPLANNER, bl._POLL_SPEND, bl._STOP_CHILDREN, notify) = saved
    human_ask.set_notifier(notify)


@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    bat_loop.set_store(s)
    yield s
    await s.close()


async def _run(store, *, steps=None, cap=0.0, task="do the thing", owner="u1", token_cap=0):
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    config = {"contract": [], **({"token_cap": token_cap} if token_cap else {})}
    await store.create_run(run_id=rid, owner_id=owner, title="t", task=task,
                           config=config, llm_usd_cap=cap, status="running")
    if steps:
        await store.seed_steps(rid, steps)
    return rid


def _step(key, seq, after=None, status="pending", **more):
    return {"step_key": key, "title": key.upper(), "seq": seq, "depends_on": after, "status": status, **more}


def _ok_runner(calls=None):
    async def run(_run, step):
        if calls is not None:
            calls.append(step["step_key"])
        return {"ok": True, "output": f"{step['step_key']} ok"}
    return run


# ── the plan's shape ──────────────────────────────────────────────────

def test_no_recorded_deps_means_every_earlier_step():
    steps = [_step("a", 0), _step("b", 1), _step("c", 2)]
    assert bl.effective_deps(steps) == {"a": [], "b": ["a"], "c": ["a", "b"]}


def test_a_dep_on_a_step_the_plan_never_had_does_not_block():
    steps = [_step("a", 0, after=["ghost"])]
    assert [s["step_key"] for s in bl.ready_steps(steps, 3)] == ["a"]


def test_ready_steps_wait_for_their_dependencies():
    steps = [_step("a", 0, after=[]), _step("b", 1, after=[]), _step("c", 2, after=["a", "b"])]
    assert [s["step_key"] for s in bl.ready_steps(steps, 3)] == ["a", "b"]
    steps[0]["status"] = "done"
    assert [s["step_key"] for s in bl.ready_steps(steps, 3)] == ["b"]
    steps[1]["status"] = "done"
    assert [s["step_key"] for s in bl.ready_steps(steps, 3)] == ["c"]


def test_a_failed_step_holds_back_what_needs_it_until_attempts_run_out():
    steps = [_step("a", 0, after=[], status="failed", attempt=1), _step("b", 1, after=["a"])]
    assert [s["step_key"] for s in bl.ready_steps(steps, 3)] == ["a"]
    steps[0]["attempt"] = 3
    assert bl.ready_steps(steps, 3) == []


def test_fix_steps_only_wait_on_earlier_steps_of_their_own_round():
    made = bl.fix_steps([{"title": "x", "after": [1, 2, 9]}, {"title": "y", "after": [1]}], 2, 5)
    assert [m["step_key"] for m in made] == ["fix-2-1", "fix-2-2"]
    assert made[0]["depends_on"] == []  # nothing before the first
    assert made[1]["depends_on"] == ["fix-2-1"]
    assert (made[0]["seq"], made[1]["round"]) == (5, 2)


def test_rewire_hands_dependents_of_a_failed_step_to_the_fix():
    steps = [_step("a", 0, after=[], status="failed", attempt=3), _step("b", 1, after=[], status="done"),
             _step("c", 2, after=["a", "b"]), _step("d", 3, after=["b"])]
    skip, rewired = bl.rewire_failed(steps, ["fix-1-1"])
    assert skip == ["a"]
    assert rewired == {"c": ["b", "fix-1-1"]}  # d never needed a


# ── parallel ──────────────────────────────────────────────────────────

async def test_independent_steps_really_run_at_the_same_time(store):
    rid = await _run(store, steps=[_step("a", 0, after=[]), _step("b", 1, after=[])])
    both_started = asyncio.Barrier(2)

    async def runner(_run, step):
        await asyncio.wait_for(both_started.wait(), timeout=2)  # alone, this times out
        return {"ok": True, "output": step["step_key"]}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"


async def test_parallel_limit_is_honoured_and_order_kept(store, monkeypatch):
    monkeypatch.setenv("FD_BAT_MAX_PARALLEL", "2")
    rid = await _run(store, steps=[_step("a", 0, after=[]), _step("b", 1, after=[]), _step("c", 2, after=[]),
                                   _step("z", 3, after=["a", "b", "c"])])
    live, peak, order = 0, 0, []

    async def runner(_run, step):
        nonlocal live, peak
        order.append(step["step_key"])
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return {"ok": True, "output": "x"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert peak == 2 and order[-1] == "z"


async def test_a_plain_list_still_runs_one_at_a_time(store):
    rid = await _run(store, steps=[_step("a", 0), _step("b", 1)])
    live, peak = 0, 0

    async def runner(_run, step):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return {"ok": True, "output": "x"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert peak == 1


async def test_a_retry_is_told_why_the_last_attempt_failed(store):
    rid = await _run(store, steps=[_step("a", 0)])
    seen = []

    async def runner(_run, step):
        seen.append((step["attempt"], step["error"]))
        return {"ok": False, "error": "the page was gone"} if len(seen) == 1 else {"ok": True, "output": "x"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert seen == [(0, ""), (1, "the page was gone")]


async def test_no_worker_available_waits_its_turn_without_burning_an_attempt(store):
    rid = await _run(store, steps=[_step("a", 0, after=[]), _step("b", 1, after=[])])
    b_tries = 0
    a_finish = asyncio.Event()

    async def runner(_run, step):
        nonlocal b_tries
        if step["step_key"] == "a":
            await a_finish.wait()
            return {"ok": True, "output": "a"}
        b_tries += 1
        if b_tries == 1:
            await asyncio.sleep(0)
            a_finish.set()  # a keeps running a moment longer than this refusal
            return {"ok": False, "error": "no room", "spawn_refused": True}
        return {"ok": True, "output": "b"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["b"]["status"] == "done" and steps["b"]["attempt"] == 1


# ── a refusal is new work ─────────────────────────────────────────────

def _judge_after(fixed_when, *, failing=("a check",)):
    calls = []

    async def judge(_run, steps):
        calls.append([s["step_key"] for s in steps])
        if fixed_when(steps):
            return {"done": True, "reason": "good"}
        return {"done": False, "reason": "still wrong", "failing": list(failing)}

    judge.calls = calls
    return judge


async def test_a_refusal_adds_fix_steps_and_the_run_finishes(store):
    rid = await _run(store, steps=[_step("a", 0)])
    calls = []
    plans = []

    async def replan(_run, steps, verdict, round_no):
        plans.append((verdict["reason"], round_no))
        return [{"title": "fix it", "brief": "what was wrong", "after": []}]

    bat_loop.set_replanner(replan)
    judge = _judge_after(lambda steps: any(s["step_key"].startswith("fix-") for s in steps))
    assert await BatDriver(store, attempt_runner=_ok_runner(calls), judge=judge).drive(rid) == "done"
    assert calls == ["a", "fix-1-1"]
    assert plans == [("still wrong", 1)]
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["fix-1-1"]["brief"] == "what was wrong" and steps["fix-1-1"]["round"] == 1
    assert (await store.get_run(rid))["config"]["rounds"] == 1


async def test_a_step_out_of_attempts_is_replaced_and_the_plan_carries_on(store):
    rid = await _run(store, steps=[_step("a", 0, after=[]), _step("b", 1, after=["a"])])
    calls = []

    async def runner(_run, step):
        calls.append(step["step_key"])
        return {"ok": False, "error": "nope"} if step["step_key"] == "a" else {"ok": True, "output": "fine"}

    async def replan(_run, steps, _verdict, _round):
        return [{"title": "another way", "after": []}]

    async def judge(_run, steps):
        if all(s["status"] in ("done", "skipped") for s in steps):
            return {"done": True, "reason": "good"}
        return {"done": False, "reason": "failed: a", "failing": ["all planned steps completed"]}

    bat_loop.set_replanner(replan)
    assert await BatDriver(store, attempt_runner=runner, judge=judge, max_step_attempts=2).drive(rid) == "done"
    assert calls == ["a", "a", "fix-1-1", "b"]  # b waited for the fix, not the dead step
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["a"]["status"] == "skipped" and steps["a"]["error"] == "nope"
    assert steps["b"]["depends_on"] == ["fix-1-1"]


async def test_the_same_failure_after_two_fix_rounds_is_no_progress(store):
    rid = await _run(store, steps=[_step("a", 0)])
    made = []

    async def replan(_run, _steps, _verdict, round_no):
        made.append(round_no)
        return [{"title": f"try {round_no}", "after": []}]

    bat_loop.set_replanner(replan)
    judge = _judge_after(lambda _steps: False)
    assert await BatDriver(store, attempt_runner=_ok_runner(), judge=judge).drive(rid) == "error"
    run = await store.get_run(rid)
    assert run["stopped_reason"].startswith("no progress")
    assert made == [1, 2]  # judged three times, fixed twice


async def test_changing_failures_run_until_the_round_ceiling(store):
    rid = await _run(store, steps=[_step("a", 0)])
    n = 0

    async def replan(_run, _steps, _verdict, round_no):
        return [{"title": f"try {round_no}", "after": []}]

    async def judge(_run, _steps):
        nonlocal n
        n += 1
        return {"done": False, "reason": f"new problem {n}", "failing": [f"check {n}"]}

    bat_loop.set_replanner(replan)
    assert await BatDriver(store, attempt_runner=_ok_runner(), judge=judge, max_rounds=4).drive(rid) == "error"
    assert n == 4
    assert "no more rounds" in (await store.get_run(rid))["stopped_reason"]


async def test_the_round_count_survives_a_restart(store):
    rid = await _run(store, steps=[_step("a", 0)])
    cfg = (await store.get_run(rid))["config"]
    await store.set_config(rid, {**cfg, "rounds": 3})

    async def replan(_run, _steps, _verdict, _round):
        pytest.fail("no rounds left, so nothing to plan")

    async def judge(_run, _steps):
        return {"done": False, "reason": "no", "failing": ["c"]}

    bat_loop.set_replanner(replan)
    assert await BatDriver(store, attempt_runner=_ok_runner(), judge=judge, max_rounds=4).drive(rid) == "error"


async def test_no_new_work_ends_the_run_honestly(store):
    rid = await _run(store, steps=[_step("a", 0)])

    async def replan(*_a):
        return []

    bat_loop.set_replanner(replan)
    judge = _judge_after(lambda _steps: False)
    assert await BatDriver(store, attempt_runner=_ok_runner(), judge=judge).drive(rid) == "error"
    run = await store.get_run(rid)
    assert run["stopped_reason"] == "still wrong"
    assert "a ok" in run["truth"]  # the work is kept


async def test_an_overridden_verdict_says_so_in_the_result(store):
    rid = await _run(store, steps=[_step("a", 0)])

    async def judge(_run, _steps):
        return {"done": True, "overridden": True, "reason": "model veto (disagree) overridden"}

    assert await BatDriver(store, attempt_runner=_ok_runner(), judge=judge).drive(rid) == "done"
    assert "Reviewers still objected" in (await store.get_run(rid))["truth"]


# ── asking the human, from a tool ─────────────────────────────────────

async def test_a_step_that_asked_through_the_tool_runs_again_with_the_answer(store):
    rid = await _run(store, steps=[_step("login", 0), _step("after", 1)])
    calls = []

    async def runner(_run, step):
        calls.append(step["step_key"])
        if step["step_key"] == "login" and calls.count("login") == 1:
            # what /fd/bat/agent/ask does while the worker's turn is running
            await human_ask.raise_ask(store, run_id=rid, owner="u1", kind="input",
                                      question="which account?", step_key="login")
            await store.set_status(rid, "awaiting_human")
            return {"ok": True, "output": "Asked the owner and paused."}
        return {"ok": True, "output": "done"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "awaiting_human"
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["login"]["status"] == "pending" and steps["after"]["status"] == "pending"

    ask = await store.latest_open_ask(rid)
    await human_ask.answer(store, ask["id"], "the work one", via="ui")
    await store.set_status(rid, "running")
    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert calls == ["login", "login", "after"]


async def test_a_question_from_one_step_leaves_the_others_finished(store):
    rid = await _run(store, steps=[_step("asks", 0, after=[]), _step("works", 1, after=[])])
    asked = asyncio.Event()

    async def runner(_run, step):
        if step["step_key"] == "asks":
            await human_ask.raise_ask(store, run_id=rid, owner="u1", kind="input",
                                      question="?", step_key="asks")
            await store.set_status(rid, "awaiting_human")
            asked.set()
            return {"ok": True, "output": "paused"}
        await asked.wait()
        return {"ok": True, "output": "real work"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "awaiting_human"
    steps = {s["step_key"]: s for s in await store.list_steps(rid)}
    assert steps["asks"]["status"] == "pending"
    assert steps["works"]["status"] == "done" and steps["works"]["output"] == "real work"


# ── stopping work in flight ───────────────────────────────────────────

async def _hanging_runner(started: asyncio.Event, ended: list):
    async def runner(_run, step):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            ended.append(step["step_key"])  # where _bat_attempt tears its worker down
        return {"ok": True, "output": "never"}
    return runner


async def test_cancel_ends_a_running_step_instead_of_waiting_for_it(store):
    rid = await _run(store, steps=[_step("a", 0)])
    started, ended = asyncio.Event(), []
    runner = await _hanging_runner(started, ended)
    drive = asyncio.create_task(BatDriver(store, attempt_runner=runner).drive(rid))
    await started.wait()
    await cancel_run(store, rid)
    assert await asyncio.wait_for(drive, timeout=2) == "cancelled"
    assert ended == ["a"]


async def test_a_turn_that_spends_past_the_cap_is_stopped_mid_turn(store):
    rid = await _run(store, steps=[_step("a", 0)], cap=1.0)
    started, ended = asyncio.Event(), []
    runner = await _hanging_runner(started, ended)

    async def poll(_store, run):
        return bat_nested.live_usd(run["id"])

    bat_loop.set_spend_watch(poll, None)
    drive = asyncio.create_task(BatDriver(store, attempt_runner=runner).drive(rid))
    await started.wait()
    bat_nested.set_live(rid, "a", 1.5)
    assert await asyncio.wait_for(drive, timeout=2) == "error"
    run = await store.get_run(rid)
    assert run["stopped_reason"] == "llm_usd_cap_reached" and ended == ["a"]
    assert (await store.list_steps(rid))[0]["status"] == "pending"  # not left "running"


async def test_a_turn_that_uses_more_tokens_than_the_cap_is_stopped_mid_turn(store):
    """A cap in tokens (config.token_cap) — Captain Swarm's attempts — holds like a dollar one."""
    rid = await _run(store, steps=[_step("a", 0)], token_cap=1_000_000)
    started, ended = asyncio.Event(), []
    runner = await _hanging_runner(started, ended)

    async def poll(_store, run):
        return bat_nested.live_usd(run["id"])

    bat_loop.set_spend_watch(poll, None)
    drive = asyncio.create_task(BatDriver(store, attempt_runner=runner).drive(rid))
    await started.wait()
    bat_nested.set_live(rid, "a", 0.0, 400_000)          # an unpriced model: no dollars, still tokens
    await asyncio.sleep(0.05)
    assert not drive.done()
    bat_nested.set_live(rid, "a", 0.0, 1_200_000)
    assert await asyncio.wait_for(drive, timeout=2) == "error"
    run = await store.get_run(rid)
    assert run["stopped_reason"] == "llm_token_cap_reached" and ended == ["a"]


# ── runs a worker starts ──────────────────────────────────────────────

def _fake_child(monkeypatch, states):
    """Scripted readings per child id; the last one repeats."""
    seen: dict[str, int] = {}
    stopped: list[str] = []

    async def read(_kind, child_id, _owner):
        i = seen.get(child_id, 0)
        seen[child_id] = i + 1
        return states[child_id][min(i, len(states[child_id]) - 1)]

    async def stop_child(_kind, child_id, _owner):
        stopped.append(child_id)

    monkeypatch.setattr(bat_nested, "_read", read)
    monkeypatch.setattr(bat_nested, "_stop_child", stop_child)
    bat_loop.set_spend_watch(bat_nested.poll, bat_nested.stop)
    return stopped


async def test_adopt_only_ties_a_run_to_a_known_worker_of_the_same_owner(store):
    rid = await _run(store)
    await bat_nested.adopt("vatra", "stranger", "u1", port=999)
    bat_nested.worker_started(rid, "a", port=1234, slug="w")
    await bat_nested.adopt("vatra", "wrong-owner", "u2", port=1234)
    await bat_nested.adopt("vatra", "by-port", "u1", port=1234)
    await bat_nested.adopt("crew", "by-slug", "u1", slug="w")
    assert [c["child_id"] for c in await store.list_children(rid)] == ["by-port", "by-slug"]
    bat_nested.worker_ended(rid, "a", port=1234, slug="w")
    assert bat_nested.worker_of(port=1234) is None and bat_nested.worker_of(slug="w") is None


async def test_a_worker_of_a_spent_run_cannot_start_another(store):
    rid = await _run(store, cap=1.0)
    bat_nested.worker_started(rid, "a", port=1234)
    assert await bat_nested.refusal("u1", port=1234) == ""
    await store.bump_cost(rid, 1.0)
    assert "budget" in await bat_nested.refusal("u1", port=1234)
    assert await bat_nested.refusal("u1", port=4321) == ""  # not a Bat worker


async def test_a_worker_of_a_run_out_of_tokens_cannot_start_another(store):
    rid = await _run(store, token_cap=500_000)
    bat_nested.worker_started(rid, "a", port=1234)
    assert await bat_nested.refusal("u1", port=1234) == ""
    await store.bump_cost(rid, 0.0, 500_000)
    assert "500,000 tokens" in await bat_nested.refusal("u1", port=1234)


async def test_a_childs_spend_counts_and_the_judge_waits_for_its_result(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0)])
    _fake_child(monkeypatch, {"kid": [
        {"usd": 0.5, "tokens": 100, "open": True, "result": ""},
        {"usd": 2.0, "tokens": 400, "open": False, "result": "the team's answer"},
    ]})
    judged = []

    async def runner(_run, step):
        await store.add_child(rid, "kid", "vatra", step["step_key"])
        return {"ok": True, "output": "started a team", "usd": 0.25}

    async def judge(_run, steps):
        judged.append(steps[0]["output"])
        return {"done": True, "reason": "good"}

    assert await BatDriver(store, attempt_runner=runner, judge=judge).drive(rid) == "done"
    assert "the team's answer" in judged[0]
    run = await store.get_run(rid)
    assert run["cumulative_usd"] == pytest.approx(2.25)  # the child's total once, not its readings
    assert run["cumulative_tokens"] == 400
    assert (await store.list_children(rid))[0]["status"] == "closed"


async def test_a_step_that_needs_a_childs_result_waits_for_it(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0, after=[]), _step("b", 1, after=["a"])])
    _fake_child(monkeypatch, {"kid": [
        {"usd": 0.0, "tokens": 0, "open": True, "result": ""},
        {"usd": 0.0, "tokens": 0, "open": True, "result": ""},
        {"usd": 0.0, "tokens": 0, "open": False, "result": "the team's answer"},
    ]})
    seen = {}

    async def runner(_run, step):
        if step["step_key"] == "a":
            await store.add_child(rid, "kid", "vatra", "a")
            return {"ok": True, "output": "started a team"}
        seen["a_output"] = next(s for s in await store.list_steps(rid) if s["step_key"] == "a")["output"]
        return {"ok": True, "output": "used it"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert "the team's answer" in seen["a_output"]


async def test_a_childs_spend_stops_the_run_at_the_cap_and_ends_the_child(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0)], cap=1.0)
    stopped = _fake_child(monkeypatch, {"kid": [{"usd": 5.0, "tokens": 9, "open": True, "result": ""}]})

    async def runner(_run, step):
        await store.add_child(rid, "kid", "vatra", step["step_key"])
        return {"ok": True, "output": "started a team"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "error"
    run = await store.get_run(rid)
    assert run["stopped_reason"] == "llm_usd_cap_reached"
    assert stopped == ["kid"]
    assert (await store.list_children(rid))[0]["status"] == "closed"


async def test_a_child_is_ended_when_the_owner_cancels(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0)])
    stopped = _fake_child(monkeypatch, {"kid": [{"usd": 0.0, "tokens": 0, "open": True, "result": ""}]})

    async def runner(_run, step):
        await store.add_child(rid, "kid", "crew", step["step_key"])
        await cancel_run(store, rid)
        return {"ok": True, "output": "x"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "cancelled"
    assert stopped == ["kid"]


async def test_a_cancel_during_the_judgement_wins_over_its_verdict(store):
    rid = await _run(store, steps=[_step("a", 0)])

    async def judge(_run, _steps):
        await cancel_run(store, rid)
        return {"done": True, "reason": "looks good"}

    assert await BatDriver(store, attempt_runner=_ok_runner(), judge=judge).drive(rid) == "cancelled"
    assert (await store.get_run(rid))["status"] == "cancelled"


async def test_cancelling_a_parked_run_ends_its_children(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0)])
    stopped = _fake_child(monkeypatch, {"kid": [{"usd": 0.0, "tokens": 0, "open": True, "result": ""}]})
    await store.add_child(rid, "kid", "vatra", "a")
    await store.set_status(rid, "awaiting_human")
    assert await cancel_run(store, rid)
    assert stopped == ["kid"]


async def test_a_bug_in_one_step_leaves_no_other_step_running(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0, after=[]), _step("b", 1, after=[])])
    ended, b_started = [], asyncio.Event()
    real = store.upsert_step

    async def flaky(run_id, key, **kw):
        if key == "a" and kw.get("status") == "done":
            raise RuntimeError("disk full")
        return await real(run_id, key, **kw)

    monkeypatch.setattr(store, "upsert_step", flaky)

    async def runner(_run, step):
        if step["step_key"] == "a":
            await b_started.wait()
            return {"ok": True, "output": "a"}
        b_started.set()
        try:
            await asyncio.sleep(30)
        finally:
            ended.append("b")

    assert await asyncio.wait_for(BatDriver(store, attempt_runner=runner).drive(rid), timeout=2) == "error"
    assert ended == ["b"]


async def test_a_stuck_child_is_stopped_so_the_judge_is_not_held_forever(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0)])
    stopped = _fake_child(monkeypatch, {"kid": [{"usd": 0.0, "tokens": 0, "open": True, "result": ""}]})
    monkeypatch.setattr(bat_nested, "_CHILD_WAIT_S", -1)

    async def runner(_run, step):
        await store.add_child(rid, "kid", "vatra", step["step_key"])
        return {"ok": True, "output": "x"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert stopped == ["kid"]
    assert "stopped for taking too long" in (await store.list_steps(rid))[0]["output"]


async def test_a_parked_run_ends_its_children_once_the_cap_is_gone(store, monkeypatch):
    rid = await _run(store, steps=[_step("a", 0)], cap=1.0)
    stopped = _fake_child(monkeypatch, {"kid": [{"usd": 2.0, "tokens": 9, "open": True, "result": ""}]})
    await store.add_child(rid, "kid", "vatra", "a")
    await BatDriver(store, attempt_runner=_ok_runner()).watch_parked(await store.get_run(rid))
    assert stopped == ["kid"]
    assert (await store.get_run(rid))["cumulative_usd"] == pytest.approx(2.0)


async def test_reading_a_real_team_child(store, monkeypatch):
    from captain_claw.flight_deck import auth, basna_routes

    class Db:
        async def run_cost_total(self, _owner, _id):
            return 3.0, 300

        async def get_basna_session(self, _id, _owner):
            return {"truth": "final words", "status": "done"}

    monkeypatch.setattr(auth, "get_db", lambda: Db())
    monkeypatch.setitem(basna_routes._RUN_USAGE, "live", [{"model": "unknown-model", "usage": {"total_tokens": 7}}])
    monkeypatch.setitem(basna_routes._active_agent_runs, "u1", {"live"})
    live = await bat_nested._read_team("live", "u1")
    assert live["open"] and live["tokens"] == 7 and live["result"] == ""
    ended = await bat_nested._read_team("gone", "u1")
    assert not ended["open"] and ended["usd"] == 3.0 and ended["result"] == "final words"


async def test_reading_a_crew_child(monkeypatch):
    from captain_claw.flight_deck import crew_runner

    class Store:
        def get(self, _owner, task_id):
            return {"running": {"status": "running", "cost_usd": 0.5, "tokens_in": 1, "tokens_out": 2},
                    "over": {"status": "delivered", "final": "merged it", "cost_usd": 1.5,
                             "results": {"files_path": "vfs:crew/p/t/files/"}}}[task_id]

    monkeypatch.setattr(crew_runner, "_store", lambda: Store())
    running = await bat_nested._read_crew("running", "u1")
    assert running["open"] and running["tokens"] == 3
    over = await bat_nested._read_crew("over", "u1")
    assert not over["open"] and "merged it" in over["result"] and "vfs:crew/p/t/files/" in over["result"]


# ── plans ─────────────────────────────────────────────────────────────

def test_parse_plan_reads_dependencies_and_keeps_old_shapes_sequential():
    steps = bat_routes._parse_plan(
        '[{"step": "a", "after": []}, {"step": "b", "after": []}, {"step": "c", "after": [1, 2, 7]}, {"step": "d"}]', "t")
    assert [s["depends_on"] for s in steps] == [[], [], ["step-1", "step-2"], None]
    assert bat_routes._parse_plan('["x", "y"]', "t")[1]["depends_on"] is None


def test_parse_plan_can_not_wait_on_itself_or_a_later_step():
    steps = bat_routes._parse_plan('[{"step": "a", "after": [1, 2]}, {"step": "b", "after": [2, 1]}]', "t")
    assert steps[0]["depends_on"] == [] and steps[1]["depends_on"] == ["step-1"]


def test_parse_archetype_plan_reads_dependencies():
    steps, arch = bat_routes._parse_archetype_plan(
        '[{"step": "a", "archetype": "x", "after": []}, {"step": "b", "archetype": "x", "after": [1]}]', {"x"})
    assert steps[1]["depends_on"] == ["step-1"] and arch == {"step-1": "x", "step-2": "x"}


def test_parse_fix_plan():
    fixes = bat_routes._parse_fix_plan('[{"step": "a"}, "b", {"step": "c", "after": [1, "x"]}, {"nope": 1}]')
    assert fixes == [{"title": "a", "after": []}, {"title": "b", "after": []}, {"title": "c", "after": [1]}]
    assert bat_routes._parse_fix_plan("not json") == []
    assert len(bat_routes._parse_fix_plan(str(["s"] * 9).replace("'", '"'))) == bat_routes._MAX_FIX_STEPS


def test_final_deliverable_of_a_parallel_plan_is_every_branch_and_every_fix():
    steps = [_step("a", 0, after=[], status="done", output="research notes " * 30),
             _step("b", 1, after=["a"], status="done", output="BRANCH ONE " * 20),
             _step("c", 2, after=["a"], status="done", output="BRANCH TWO " * 20),
             _step("fix-1-1", 3, after=[], status="done", output="THE FIX " * 20)]
    out = bat_routes._final_deliverable(steps)
    assert "BRANCH ONE" in out and "BRANCH TWO" in out and "THE FIX" in out
    assert "research notes" not in out


def test_step_prompt_carries_the_brief_the_failure_the_checks_and_the_company():
    run = {"id": "bat_12345678", "task": "t", "config": {"contract": [{"text": "report.md exists"}]}}
    step = {"step_key": "fix-1-1", "title": "fix", "brief": "- Reviewer: section 3 is empty",
            "attempt": 1, "error": "the page was gone"}
    prompt = bat_routes._build_step_prompt(run, step, [{"step_key": "other", "status": "running"}])
    assert "section 3 is empty" in prompt
    assert "the page was gone" in prompt and "different approach" in prompt
    assert "report.md exists" in prompt and "vfs:bat-bat_1234/" in prompt
    assert "same time" in prompt
    assert "deleted when your step ends" in prompt
    plain = bat_routes._build_step_prompt({"task": "t", "config": {}}, {"step_key": "a", "attempt": 0}, [])
    assert "same time" not in plain and "failed" not in plain and "checked" not in plain


def test_default_cap(monkeypatch):
    monkeypatch.delenv("FD_BAT_DEFAULT_LLM_USD", raising=False)
    assert bat_routes._llm_cap(0) == bat_routes._DEFAULT_LLM_USD
    assert bat_routes._llm_cap(7.5) == 7.5
    monkeypatch.setenv("FD_BAT_DEFAULT_LLM_USD", "0")
    assert bat_routes._llm_cap(0) == 0.0


def test_panel_prefers_a_model_the_workers_do_not_run_on(monkeypatch):
    from captain_claw.flight_deck import basna_routes
    monkeypatch.setattr(basna_routes, "_effective_key", lambda _p, k, _b=None: k, raising=True)
    tiers = {"reason": {"model": "big"}, "balanced": {"model": "big"}, "fast": {"model": "small"}}
    creds, independent = bat_routes._panel_creds(tiers)
    assert creds["model"] == "small" and independent
    creds, independent = bat_routes._panel_creds({"reason": {"model": "only"}})
    assert creds["model"] == "only" and not independent


# ── the fix round's gate ──────────────────────────────────────────────

async def test_a_fix_step_beyond_what_was_approved_asks_again(store):
    rid = await _run(store, task="research the market", steps=[_step("a", 0, status="done")])
    assert await bat_routes._gate_check(store, await store.get_run(rid)) == "proceed"
    await store.seed_steps(rid, [_step("fix-1-1", 1, after=[]) | {"title": "Email the supplier for the figures"}])
    assert await bat_routes._gate_check(store, await store.get_run(rid)) == "wait"
    run = await store.get_run(rid)
    assert run["status"] == "awaiting_plan" and "send email" in run["config"]["gate_reason"]
    ask = await store.latest_open_ask(rid)
    assert "Email the supplier" in ask["question"]
    await human_ask.answer(store, ask["id"], "approve", via="ui")
    assert await bat_routes._gate_check(store, await store.get_run(rid)) == "proceed"
    assert (await store.get_run(rid))["config"]["email_allowed"] is True


async def test_an_approved_plan_needs_no_new_approval_for_more_of_the_same(store):
    rid = await _run(store, task="email the supplier", steps=[_step("a", 0)])
    cfg = (await store.get_run(rid))["config"]
    await store.set_config(rid, {**cfg, "plan_approved": True, "email_allowed": True})
    await store.seed_steps(rid, [_step("fix-1-1", 1, after=[]) | {"title": "Email them again with the right file"}])
    assert await bat_routes._gate_check(store, await store.get_run(rid)) == "proceed"


async def test_a_new_kind_of_action_after_approval_asks_again(store):
    rid = await _run(store, task="email the supplier", steps=[_step("a", 0)])
    cfg = (await store.get_run(rid))["config"]
    await store.set_config(rid, {**cfg, "plan_approved": True, "email_allowed": True})
    await store.seed_steps(rid, [_step("fix-1-1", 1, after=[]) | {"title": "Buy the report from the supplier"}])
    assert await bat_routes._gate_check(store, await store.get_run(rid)) == "wait"
    assert "spend money" in (await store.latest_open_ask(rid))["question"]


async def test_the_driver_stops_for_that_gate_before_running_the_step(store):
    rid = await _run(store, task="research the market", steps=[_step("a", 0)])
    calls = []

    async def replan(*_a):
        return [{"title": "Email the supplier", "after": []}]

    async def judge(_run, _steps):
        return {"done": False, "reason": "missing figures", "failing": []}

    bat_loop.set_replanner(replan)
    bat_loop.set_gate_check(bat_routes._gate_check)
    assert await BatDriver(store, attempt_runner=_ok_runner(calls), judge=judge).drive(rid) == "awaiting_plan"
    assert calls == ["a"]


# ── the judge's real checks ───────────────────────────────────────────

class _Prov:
    def __init__(self, votes, log, audit="[]"):
        self.votes, self.log, self.audit = votes, log, audit

    async def complete(self, messages, **_kw):
        if messages[0].content == bat_routes._AUDIT_SYSTEM:
            self.log.append("AUDIT " + messages[-1].content)
            return type("R", (), {"content": self.audit, "usage": None, "model": "m"})()
        self.log.append(messages[-1].content)
        votes_so_far = sum(not p.startswith("AUDIT ") for p in self.log)
        vote = self.votes[min(votes_so_far - 1, len(self.votes) - 1)]
        return type("R", (), {"content": f"VOTE: {vote}\nREASON: {vote} because section 3 is empty",
                              "usage": None, "model": "m"})()


@pytest.fixture
def judge_env(monkeypatch, tmp_path):
    """A real folder for the run, a stubbed database and model."""
    from captain_claw.flight_deck import auth, basna_routes

    class Db:
        async def get_user_by_id(self, _id):
            return {"id": "u1", "role": "user"}

    prompts: list[str] = []
    env = {"votes": ["AGREE"], "prompts": prompts, "folder": tmp_path / "folder", "audit": "[]"}
    env["folder"].mkdir()

    async def tiers(_db, _owner):
        return {"reason": {"model": "big"}, "fast": {"model": "small"}}, []

    monkeypatch.setattr(auth, "get_db", lambda: Db())
    monkeypatch.setattr(auth, "_fd_auth_enabled", lambda: True)
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(basna_routes, "_effective_key", lambda _p, k, _b=None: k)
    monkeypatch.setattr(basna_routes, "_provider_call", lambda *_a, **_k: (_Prov(env["votes"], prompts, env["audit"]), 256))
    monkeypatch.setattr(bat_routes, "_run_folder", lambda _run: env["folder"])
    bat_loop.set_replanner(bat_routes._bat_replan)
    return env


async def _judged_run(store, contract, *, cfg=None):
    rid = await _run(store, steps=[_step("a", 0, status="done", output="the whole answer " * 20)])
    await store.upsert_step(rid, "a", status="done", output="the whole answer " * 20)
    await store.set_config(rid, {"contract": contract, **(cfg or {})})
    return await store.get_run(rid), await store.list_steps(rid)


def _rule(text, check, severity="critical"):
    return {"id": "a1", "text": text, "severity": severity, "check": check}


async def test_a_failed_file_check_blocks_done_without_asking_the_panel(store, judge_env):
    run, steps = await _judged_run(store, [_rule("report.md exists", {"type": "file_exists", "path": "report.md"})])
    verdict = await bat_routes._bat_judge(run, steps)
    assert not verdict["done"]
    assert [p[:5] for p in judge_env["prompts"]] == ["AUDIT"]  # a second look at the check, no votes
    assert verdict["failing"] == ["acceptance: report.md exists"]
    assert "does not exist" in verdict["det_criticals"][0]


async def test_a_check_at_fault_goes_to_the_reviewers_and_stays_there(store, judge_env):
    """What a real run did: correct files, and a command that fails on macOS."""
    (judge_env["folder"] / "primes.txt").write_text("2\n3\n5\n7\n11\n")
    judge_env["audit"] = '[{"id": "a1", "verdict": "pass", "note": "wc -l pads its number here"}]'
    run, steps = await _judged_run(store, [_rule("primes.txt has five lines", {"type": "command", "cmd": "exit 1"})])
    run = {**run, "config": {**run["config"]}}
    with_shell = bat_routes._owner_has_shell
    bat_routes._owner_has_shell = lambda _u: True
    try:
        verdict = await bat_routes._bat_judge(run, steps)
    finally:
        bat_routes._owner_has_shell = with_shell
    assert verdict["done"] and not verdict["overridden"]
    votes = [p for p in judge_env["prompts"] if not p.startswith("AUDIT")]
    assert "primes.txt has five lines" in votes[0]  # the reviewers had to agree it holds
    fresh = await store.get_run(run["id"])
    assert fresh["config"]["contract"][0]["check"] == {"type": "judge"}
    assert "was at fault" in " ".join(str(e["message"]) for e in await store.list_events(run["id"]))


async def test_a_check_at_fault_still_needs_the_reviewers_to_agree(store, judge_env):
    judge_env["audit"] = '[{"id": "a1", "verdict": "pass", "note": "quirk"}]'
    judge_env["votes"] = ["DISAGREE"]
    run, steps = await _judged_run(store, [_rule("report.md exists", {"type": "file_exists", "path": "report.md"})])
    verdict = await bat_routes._bat_judge(run, steps)
    assert not verdict["done"] and not verdict["det_criticals"] and not verdict.get("parked")


async def test_an_unclear_or_failed_audit_leaves_the_failure_standing(store, judge_env, monkeypatch):
    rule = _rule("report.md exists", {"type": "file_exists", "path": "report.md"})
    judge_env["audit"] = '[{"id": "a1", "verdict": "unclear", "note": "can not tell"}]'
    run, steps = await _judged_run(store, [rule])
    assert (await bat_routes._bat_judge(run, steps))["failing"] == ["acceptance: report.md exists"]
    judge_env["audit"] = "the model rambled instead"
    assert (await bat_routes._bat_judge(run, steps))["failing"] == ["acceptance: report.md exists"]


async def test_patterns_read_line_by_line_like_grep(judge_env, store):
    (judge_env["folder"] / "total.txt").write_text("note\n28\nend\n")
    run, _ = await _judged_run(store, [
        _rule("has the sum", {"type": "file_contains", "path": "total.txt", "pattern": "^28$"})])
    checks, _, _ = await bat_routes._contract_checks(run, has_shell=False)
    assert [c.passed for c in checks] == [True]


async def test_a_failed_check_is_never_overridden_even_with_no_rounds_left(store, judge_env):
    run, steps = await _judged_run(store, [_rule("report.md exists", {"type": "file_exists", "path": "report.md"})],
                                   cfg={"rounds": 99, "veto_rounds": 99})
    assert not (await bat_routes._bat_judge(run, steps))["done"]


async def test_passing_checks_go_to_the_panel_with_what_is_really_in_the_folder(store, judge_env):
    (judge_env["folder"] / "report.md").write_text("# The real report\nfigures here")
    run, steps = await _judged_run(store, [
        _rule("report.md exists", {"type": "file_exists", "path": "report.md"}),
        _rule("it reads well", {"type": "judge"}),
    ])
    verdict = await bat_routes._bat_judge(run, steps)
    assert verdict["done"] and not verdict["overridden"]
    shown = judge_env["prompts"][0]
    assert "VERIFIED" in shown and "# The real report" in shown  # read from the folder
    assert "it reads well" in shown  # the rule only a reviewer can decide
    assert "report.md exists" not in shown  # already settled


async def test_a_panel_refusal_is_work_while_rounds_remain_then_the_owner_decides(store, judge_env):
    judge_env["votes"] = ["DISAGREE"]
    run, steps = await _judged_run(store, [])
    first = await bat_routes._bat_judge(run, steps)
    assert not first["done"] and not first.get("parked")
    assert "section 3 is empty" in bat_routes._findings(first)
    run = await store.get_run(run["id"])
    assert run["config"]["veto_rounds"] == 1  # kept on the run, so a restart can't reset it
    assert not (await bat_routes._bat_judge(run, steps)).get("parked")
    third = await bat_routes._bat_judge(await store.get_run(run["id"]), steps)
    assert third["parked"] and not third["done"]
    fresh = await store.get_run(run["id"])
    assert fresh["status"] == "awaiting_human"
    ask = await store.latest_open_ask(run["id"])
    assert ask["kind"] == "tie_break" and "section 3 is empty" in ask["question"]
    assert ask["options"] == [bat_routes._TIE_BREAK_ACCEPT, bat_routes._TIE_BREAK_STOP]


async def test_a_rehearsal_never_asks_the_owner_to_break_a_stalemate(store, judge_env):
    """A Captain Swarm attempt: its run checks every result's page itself, so the
    reviewers' objection rides along with the result instead of parking it."""
    judge_env["votes"] = ["DISAGREE"]
    run, steps = await _judged_run(store, [], cfg={"rounds": bat_loop._DEFAULT_MAX_ROUNDS - 1,
                                                   "rehearsal": "cs_0123456789ab:exp_abcdef123456"})
    verdict = await bat_routes._bat_judge(run, steps)
    assert verdict["done"] and verdict["overridden"] and not verdict.get("parked")
    assert await store.latest_open_ask(run["id"]) is None


async def test_no_rounds_left_asks_the_owner_too(store, judge_env):
    judge_env["votes"] = ["DISAGREE"]
    run, steps = await _judged_run(store, [], cfg={"rounds": bat_loop._DEFAULT_MAX_ROUNDS - 1})
    assert (await bat_routes._bat_judge(run, steps))["parked"]


async def _stalemate(store, judge_env):
    """A run the reviewers keep refusing, driven until it asks the owner."""
    judge_env["votes"] = ["DISAGREE"]
    bat_loop.set_gate_check(bat_routes._gate_check)
    rid = await _run(store, steps=[_step("a", 0)])

    async def replan(*_a):
        return [{"title": "try again", "after": []}]

    bat_loop.set_replanner(replan)
    driver = BatDriver(store, attempt_runner=_ok_runner(), judge=bat_routes._bat_judge)
    assert await driver.drive(rid) == "awaiting_human"
    return rid, driver


async def test_accepting_a_stalemate_finishes_it_and_says_so(store, judge_env):
    rid, driver = await _stalemate(store, judge_env)
    votes_before = len(judge_env["prompts"])
    ask = await store.latest_open_ask(rid)
    await human_ask.answer(store, ask["id"], bat_routes._TIE_BREAK_ACCEPT, via="ui")
    assert await driver.drive(rid) == "done"
    assert len(judge_env["prompts"]) == votes_before  # no point asking the reviewers again
    run = await store.get_run(rid)
    assert "you accepted it anyway" in run["truth"] and "section 3 is empty" in run["truth"]


async def test_stopping_a_stalemate_ends_it_as_not_done_and_keeps_the_work(store, judge_env):
    rid, driver = await _stalemate(store, judge_env)
    ask = await store.latest_open_ask(rid)
    await human_ask.answer(store, ask["id"], bat_routes._TIE_BREAK_STOP, via="ui")
    assert await driver.drive(rid) == "error"
    run = await store.get_run(rid)
    assert run["status"] == "error" and "section 3 is empty" in run["stopped_reason"]
    assert "a ok" in run["truth"]


async def test_nothing_delivered_is_never_called_done_by_itself(store, judge_env):
    """What a real run did: every step "finished", the files went to a folder that's
    deleted, and the reviewers kept saying so."""
    rid, _driver = await _stalemate(store, judge_env)
    assert (await store.get_run(rid))["status"] == "awaiting_human"


async def test_a_step_left_failed_is_a_failure_and_skipped_is_not(store, judge_env):
    rid = await _run(store, steps=[_step("a", 0), _step("b", 1)])
    await store.upsert_step(rid, "a", status="failed", error="x", attempt=3)
    await store.upsert_step(rid, "b", status="done", output="y" * 300)
    verdict = await bat_routes._bat_judge(await store.get_run(rid), await store.list_steps(rid))
    assert not verdict["done"] and "failed: a (x)" in verdict["det_criticals"][0]
    assert verdict["failing"] == ["step failed: x"]  # the same error from a new step is no progress
    await store.upsert_step(rid, "a", status="skipped")
    assert (await bat_routes._bat_judge(await store.get_run(rid), await store.list_steps(rid)))["done"]


async def test_nothing_produced_is_not_a_deliverable(store, judge_env):
    rid = await _run(store, steps=[_step("a", 0)])
    await store.upsert_step(rid, "a", status="done", output="")
    verdict = await bat_routes._bat_judge(await store.get_run(rid), await store.list_steps(rid))
    assert not verdict["done"] and "no deliverable" in verdict["reason"]


async def test_files_alone_can_be_the_deliverable(store, judge_env):
    (judge_env["folder"] / "site.html").write_text("<h1>hello</h1>")
    rid = await _run(store, steps=[_step("a", 0)])
    await store.upsert_step(rid, "a", status="done", output="")
    assert (await bat_routes._bat_judge(await store.get_run(rid), await store.list_steps(rid)))["done"]


async def test_a_command_rule_goes_to_the_reviewers_for_someone_without_a_shell(judge_env, store):
    run, _ = await _judged_run(store, [_rule("it builds", {"type": "command", "cmd": "touch ran.txt"})])
    checks, open_rules, _ = await bat_routes._contract_checks(run, has_shell=False)
    assert checks == [] and [r["text"] for r in open_rules] == ["it builds"]
    assert not (judge_env["folder"] / "ran.txt").exists()


async def test_a_command_rule_runs_for_someone_with_a_shell(judge_env, store):
    run, _ = await _judged_run(store, [_rule("it passes", {"type": "command", "cmd": "exit 3"})])
    checks, _, failed = await bat_routes._contract_checks(run, has_shell=True)
    assert [(c.name, c.passed) for c in checks] == [("acceptance: it passes", False)]
    assert [f["text"] for f in failed] == ["it passes"]


async def test_a_passing_command_rule_runs_in_the_runs_folder(judge_env, store):
    (judge_env["folder"] / "here.txt").write_text("x")
    run, _ = await _judged_run(store, [_rule("it's there", {"type": "command", "cmd": "test -f here.txt"})])
    checks, _, _ = await bat_routes._contract_checks(run, has_shell=True)
    assert [(c.name, c.passed) for c in checks] == [("acceptance: it's there", True)]


async def test_a_projects_own_tests_count_for_someone_with_a_shell(judge_env, store):
    (judge_env["folder"] / "package.json").write_text('{"scripts": {"test": "exit 1"}}')
    run, _ = await _judged_run(store, [])
    checks, _, _ = await bat_routes._contract_checks(run, has_shell=True)
    assert [(c.name, c.passed) for c in checks] == [("the project's tests pass", False)]
    assert (await bat_routes._contract_checks(run, has_shell=False))[0] == []


async def test_the_harm_floor_covers_a_contract_command():
    ok, said = await bat_routes._floor_runner("rm -rf /", "/tmp")
    assert not ok and "hard floor" in said


def test_evidence_lists_real_files_and_skips_hidden_ones_and_links(tmp_path):
    (tmp_path / "report.md").write_text("real words")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret")
    (tmp_path / ".env").write_text("KEY=secret")
    os.symlink("/etc/hosts", tmp_path / "link.md")
    out = bat_routes._evidence_block(tmp_path)
    assert "report.md" in out and "real words" in out and "data.bin" in out
    assert "secret" not in out and "link.md" not in out


def test_evidence_says_so_when_the_folder_is_empty(tmp_path):
    assert bat_routes._NO_FILES in bat_routes._evidence_block(tmp_path / "missing")


async def test_replan_falls_back_to_the_findings_when_the_model_fails(store, judge_env, monkeypatch):
    from captain_claw.flight_deck import basna_routes

    def boom(*_a, **_k):
        raise RuntimeError("model down")

    monkeypatch.setattr(basna_routes, "_provider_call", boom)
    run, steps = await _judged_run(store, [])
    plan = await bat_routes._bat_replan(run, steps, {"reason": "r", "det_criticals": ["acceptance: x"]}, 1)
    assert len(plan) == 1 and "acceptance: x" in plan[0]["brief"]


async def test_contract_is_pinned_once_and_a_bad_reply_means_none(store, judge_env, monkeypatch):
    from captain_claw.flight_deck import basna_routes

    class Reply:
        async def complete(self, messages, **_kw):
            assert "command" not in messages[0].content  # no shell, no command rules offered
            return type("R", (), {"content": '{"constraints": [{"text": "a thing", "severity": "major"}]}'})()

    monkeypatch.setattr(basna_routes, "_provider_call", lambda *_a, **_k: (Reply(), 256))
    rid = await _run(store)
    await store.set_config(rid, {})
    await bat_routes._ensure_contract(await store.get_run(rid), {"model": "m"}, has_shell=False)
    run = await store.get_run(rid)
    assert [r["text"] for r in run["config"]["contract"]] == ["a thing"]

    class Boom:
        async def complete(self, *_a, **_k):
            pytest.fail("derived twice")

    monkeypatch.setattr(basna_routes, "_provider_call", lambda *_a, **_k: (Boom(), 256))
    await bat_routes._ensure_contract(run, {"model": "m"}, has_shell=False)


# ── a worker's turn, on a model with a price ──────────────────────────

@pytest.fixture
def attempt_env(monkeypatch):
    """_bat_attempt with the worker spawn and teardown stubbed; the turn is a test's."""
    from captain_claw.flight_deck import auth, basna_routes, pricing

    class Db:
        async def get_user_by_id(self, _id):
            return {"id": "u1", "role": "user"}

    async def tiers(_db, _owner):
        return {"reason": {"model": "priced-model"}}, []

    async def spawn(*_a, **_k):
        return {"ok": True, "port": 4567, "auth": "x", "slug": "worker-1", "message": ""}

    env = {"torn_down": []}
    monkeypatch.setattr(auth, "get_db", lambda: Db())
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(bat_routes, "_spawn_bat_worker", spawn)
    monkeypatch.setattr(bat_routes, "_teardown", lambda slugs: env["torn_down"].extend(slugs))
    # $1 per million tokens each way, so the numbers below are easy to read.
    monkeypatch.setitem(pricing._table(), "priced-model", {"input": 1.0, "output": 1.0, "cache_read": 1.0})
    return env


async def test_a_turn_reports_spend_while_it_runs_and_is_counted_once(store, attempt_env, monkeypatch):
    from captain_claw.flight_deck import basna_routes
    rid = await _run(store, steps=[_step("a", 0)], cap=100.0)
    seen = {}

    async def dispatch(_port, _auth, _prompt, _timeout, *, on_usage, usage_sink, **_k):
        assert bat_nested.worker_of(port=4567) == (rid, "a")  # a run started now is tied to this one
        usage_sink.update(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000, model="priced-model")
        on_usage(1_000_000, 1_000_000, 2_000_000)
        seen["live"] = bat_nested.live_usd(rid)
        basna_routes._record_run_usage("priced-model", dict(usage_sink))
        return {"ok": True, "output": "the answer", "model": "priced-model"}

    monkeypatch.setattr(basna_routes, "_dispatch_one", dispatch)
    res = await bat_routes._bat_attempt(await store.get_run(rid), (await store.list_steps(rid))[0])
    assert seen["live"] == pytest.approx(2.0)
    assert res["ok"] and res["usd"] == pytest.approx(2.0) and res["tokens"] == 2_000_000
    assert bat_nested.live_usd(rid) == 0.0  # it's the step's to report now, not live
    assert bat_nested.worker_of(port=4567) is None and attempt_env["torn_down"] == ["worker-1"]


async def test_steps_at_the_same_time_do_not_count_each_others_spend(store, attempt_env, monkeypatch):
    from captain_claw.flight_deck import basna_routes
    rid = await _run(store, steps=[_step("a", 0, after=[]), _step("b", 1, after=[])])
    both = asyncio.Barrier(2)

    async def dispatch(_port, _auth, _prompt, _timeout, *, agent_name, **_k):
        await both.wait()  # each records while the other's window is open
        tokens = 1_000_000 if agent_name == "a" else 3_000_000
        basna_routes._record_run_usage("priced-model", {"prompt_tokens": tokens, "total_tokens": tokens})
        await both.wait()
        return {"ok": True, "output": agent_name}

    monkeypatch.setattr(basna_routes, "_dispatch_one", dispatch)
    run = await store.get_run(rid)
    a, b = await store.list_steps(rid)
    ra, rb = await asyncio.gather(bat_routes._bat_attempt(run, a), bat_routes._bat_attempt(run, b))
    assert (ra["usd"], rb["usd"]) == (pytest.approx(1.0), pytest.approx(3.0))


async def test_a_turn_stopped_midway_still_counts_what_it_spent(store, attempt_env, monkeypatch):
    from captain_claw.flight_deck import basna_routes
    rid = await _run(store, steps=[_step("a", 0)])
    in_turn = asyncio.Event()

    async def dispatch(_port, _auth, _prompt, _timeout, *, usage_sink, **_k):
        usage_sink.update(prompt_tokens=2_000_000, total_tokens=2_000_000, model="priced-model")
        in_turn.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(basna_routes, "_dispatch_one", dispatch)
    t = asyncio.create_task(bat_routes._bat_attempt(await store.get_run(rid), (await store.list_steps(rid))[0]))
    await in_turn.wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert (await store.get_run(rid))["cumulative_usd"] == pytest.approx(2.0)
    assert attempt_env["torn_down"] == ["worker-1"]


async def test_a_full_deck_is_reported_as_no_worker_not_a_failed_step(store, attempt_env, monkeypatch):
    from fastapi import HTTPException

    async def full(*_a, **_k):
        raise HTTPException(429, "agent limit reached")

    monkeypatch.setattr(bat_routes, "_spawn_bat_worker", full)
    rid = await _run(store, steps=[_step("a", 0)])
    res = await bat_routes._bat_attempt(await store.get_run(rid), (await store.list_steps(rid))[0])
    assert res["spawn_refused"] and "agent limit" in res["error"]


async def test_the_ledger_gets_bats_own_spend_not_its_childrens(store, monkeypatch):
    from captain_claw.flight_deck import auth
    logged = []

    class Db:
        async def log_run_cost(self, owner, kind, run_id, cost):
            logged.append((owner, kind, run_id, cost["usd"], cost["tokens"]["total_tokens"]))

    monkeypatch.setattr(auth, "get_db", lambda: Db())
    rid = await _run(store)
    await store.bump_cost(rid, 3.0, 300)
    await store.add_child(rid, "kid", "vatra", "a")
    await store.update_child(rid, "kid", counted_usd=2.0, counted_tokens=200)
    await bat_routes._log_own_cost(await store.get_run(rid))
    assert logged == [("u1", "bat", rid, 1.0, 100)]


async def test_a_failed_spawn_leaves_nothing_behind(monkeypatch):
    from captain_claw.flight_deck import server
    torn_down = []

    async def spawn(*_a, **_k):
        return type("R", (), {"ok": False, "slug": "bat-x-a", "message": "captain-claw-web not found"})()

    monkeypatch.setattr(server, "spawn_process", spawn)
    monkeypatch.setattr(server, "_load_process_registry", lambda: {"bat-x-a": {"slug": "bat-x-a"}})
    monkeypatch.setattr(bat_routes, "_teardown", lambda slugs: torn_down.extend(slugs))
    res = await bat_routes._spawn_bat_worker(None, {"id": "u1"}, name="bat-x-a", tier="", tiers={}, env_vars=[], extra_env=[])
    assert not res["ok"] and "not found" in res["message"]
    assert torn_down == ["bat-x-a"]


def test_a_cancel_sweeps_workers_that_were_still_spawning(monkeypatch, tmp_path):
    from captain_claw.flight_deck import basna_routes, server
    reg = {"basna-abcd1234-x-writer": {"owner": "u1"}, "vatra-abcd1234-x-lead": {"owner": "u1"},
           "basna-abcd1234-x-theirs": {"owner": "u2"}, "basna-ffff0000-x-other": {"owner": "u1"},
           "my-own-agent": {"owner": "u1"}}
    (tmp_path / "basna-abcd1234-x-writer").mkdir()
    stopped = []
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "_load_process_registry", lambda: dict(reg))
    monkeypatch.setattr(server, "_do_stop_process", stopped.append)

    def save(new):
        reg.clear()
        reg.update(new)

    monkeypatch.setattr(server, "_save_process_registry", save)
    assert basna_routes._sweep_run_workers("abcd1234ffff", "u1") == 2
    assert sorted(stopped) == ["basna-abcd1234-x-writer", "vatra-abcd1234-x-lead"]
    assert sorted(reg) == ["basna-abcd1234-x-theirs", "basna-ffff0000-x-other", "my-own-agent"]
    assert not (tmp_path / "basna-abcd1234-x-writer").exists()


def test_one_dropped_brace_does_not_cost_every_rule():
    """A real reply from a local model: strict JSON parsing threw all five away."""
    reply = ('```json\n{"constraints": [{"id": "a1", "text": "primes.txt exists", "severity": "critical", "check": '
             '{"type": "file_exists", "path": "primes.txt"}}, {"id": "a2", "text": "primes.txt contains all five primes", '
             '"severity": "critical", "check": {"type": "command", "cmd": "awk \'{c++} END{exit !(c==5)}\' primes.txt"}, '
             '{"id": "a3", "text": "total.txt exists", "severity": "critical", "check": {"type": "file_exists", '
             '"path": "total.txt"}}, {"id": "a4", "text": "total.txt contains the sum 28", "severity": "critical", '
             '"check": {"type": "file_contains", "path": "total.txt", "pattern": "^28$"}, {"id": "a5", "text": '
             '"sum matches", "severity": "critical", "check": {"type": "command", "cmd": "grep -q 28 total.txt"}}]}\n```')
    rules = bat_routes._read_rules(reply)
    assert [r["id"] for r in rules] == ["a1", "a2", "a3", "a4", "a5"]
    assert rules[1]["check"]["type"] == "command" and "awk" in rules[1]["check"]["cmd"]
    assert rules[3]["check"] == {"type": "file_contains", "path": "total.txt", "pattern": "^28$"}


def test_reading_rules_leaves_good_and_hopeless_replies_alone():
    good = '{"constraints": [{"id": "a1", "text": "x", "severity": "major"}]}'
    assert [r["id"] for r in bat_routes._read_rules(good)] == ["a1"]
    assert bat_routes._read_rules('{"constraints": []}') == []
    assert bat_routes._read_rules("I would suggest checking the files.") == []
    assert bat_routes._read_rules("") == []


def test_a_worker_gets_bats_rules_with_every_step():
    prompt = bat_routes._build_step_prompt({"task": "t", "config": {}}, {"step_key": "a", "attempt": 0}, [])
    assert "ask_human" in prompt and "never claim a step is done" in prompt and "CAPTCHA" in prompt


def test_a_bat_workers_prompt_never_trips_the_basna_relay(monkeypatch):
    from captain_claw.agent_orchestration_mixin import _detect_basna_run
    prompt = bat_routes._build_step_prompt(
        {"task": "t", "config": {}}, {"step_key": "a", "title": "Run a Basna on the Q3 market", "attempt": 0}, [])
    monkeypatch.delenv("CLAW_BAT_WORKER", raising=False)
    assert _detect_basna_run("run a basna on the Q3 market")  # anyone else's request still relays
    monkeypatch.setenv("CLAW_BAT_WORKER", "1")
    assert _detect_basna_run(prompt) is None

