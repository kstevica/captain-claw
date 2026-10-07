"""Bat Phase 4 — the human-in-the-loop gate, end to end at the FD layer.

Exercises the REAL bat_routes._gate_check wired into the driver (no LLM / no
spawned workers): the start gate pausing a mail/$/account run until approved,
approve/reject, a non-gated run proceeding untouched, and a step that asks for a
human value pausing and resuming with the answer injected.
"""

from __future__ import annotations

import uuid

import pytest

from captain_claw.flight_deck import bat_loop, bat_routes, human_ask
from captain_claw.flight_deck.bat_loop import BatDriver, BatSupervisor
from captain_claw.flight_deck.bat_store import BatStore


@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    yield s
    await s.close()


@pytest.fixture(autouse=True)
def _wire():
    """Real gate_check + default planner/judge; a recording notifier (so we
    don't hit the bell/WhatsApp). Returns the captured asks."""
    notes: list[dict] = []

    async def rec(ask):
        notes.append(ask)

    saved = (bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, human_ask._NOTIFY)
    bat_loop.set_gate_check(bat_routes._gate_check)
    bat_loop._PLANNER = bat_loop._default_planner
    bat_loop._JUDGE = bat_loop._default_judge
    human_ask.set_notifier(rec)
    human_ask._SECRETS.clear()
    bat_routes._SECRET_ANSWERS.clear()
    yield notes
    bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, _old_notify = saved
    human_ask.set_notifier(_old_notify)


async def _mk(store, task, steps, status="planning"):
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id="u1", title="t", task=task,
                           config={"steps": steps}, status=status)
    return rid


# ── start gate ─────────────────────────────────────────────────────────

async def test_mail_run_pauses_until_approved(store, _wire):
    rid = await _mk(store, "email the quarterly report to the team",
                    ["draft the report", "email it to the team"])
    calls = []

    async def runner(run, step):
        calls.append(step["step_key"])
        return {"ok": True, "output": f"did {step['step_key']}"}

    drv = BatDriver(store, attempt_runner=runner)
    assert await drv.drive(rid) == "awaiting_plan"
    assert calls == []  # nothing acted before approval
    ask = await store.latest_open_ask(rid)
    assert ask and ask["kind"] == "plan_approval"
    assert any(a["kind"] == "plan_approval" for a in _wire)  # owner was notified

    await human_ask.answer(store, ask["id"], "approve", via="ui")
    assert await drv.drive(rid) == "done"
    assert calls  # steps ran only after approval
    assert (await store.get_run(rid))["config"]["plan_approved"] is True


async def test_mail_run_rejected_is_cancelled(store, _wire):
    rid = await _mk(store, "reply to that email", ["write reply", "send it"])
    drv = BatDriver(store, attempt_runner=lambda r, s: pytest.fail("must not run"))
    assert await drv.drive(rid) == "awaiting_plan"
    ask = await store.latest_open_ask(rid)
    await human_ask.answer(store, ask["id"], "cancel", via="ui")
    assert await drv.drive(rid) == "cancelled"
    assert (await store.get_run(rid))["stopped_reason"] == "plan_rejected"


async def test_plain_run_is_not_gated(store, _wire):
    rid = await _mk(store, "summarize the latest AI research", ["read sources", "write summary"])

    async def runner(run, step):
        return {"ok": True, "output": "ok"}

    assert await BatDriver(store, attempt_runner=runner).drive(rid) == "done"
    assert await store.latest_open_ask(rid) is None  # no approval ask raised


async def test_spend_and_account_tasks_are_gated(store, _wire):
    for task in ("buy a domain for the project", "sign up for a mailchimp account"):
        rid = await _mk(store, task, ["do it"])
        assert await BatDriver(store, attempt_runner=lambda r, s: pytest.fail()).drive(rid) == "awaiting_plan"


# ── step-level ask (awaiting_human) ────────────────────────────────────

async def test_step_asks_for_input_then_resumes_with_answer(store, _wire):
    rid = await _mk(store, "log into the dashboard and read the metrics", ["log in and read"])
    state = {"asked": False}

    async def runner(run, step):
        if not state["asked"]:
            state["asked"] = True
            return {"ask": {"kind": "input", "question": "what's the login email?"}}
        answer = ((run.get("config") or {}).get("answers") or {}).get(step["step_key"], "")
        return {"ok": True, "output": f"logged in as {answer}"}

    drv = BatDriver(store, attempt_runner=runner)
    assert await drv.drive(rid) == "awaiting_human"
    ask = await store.latest_open_ask(rid)
    assert ask and ask["kind"] == "input" and ask["step_key"]
    # the step was returned to pending so it re-runs after the answer
    assert (await store.list_steps(rid))[0]["status"] == "pending"

    await human_ask.answer(store, ask["id"], "me@example.com", via="ui")
    assert await drv.drive(rid) == "done"
    out = (await store.list_steps(rid))[0]["output"]
    assert "me@example.com" in out


async def test_expired_ask_fails_the_run(store, _wire):
    rid = await _mk(store, "do a thing", ["main"])

    async def runner(run, step):
        return {"ask": {"kind": "input", "question": "need a value", "expires_at": 100.0}}

    drv = BatDriver(store, attempt_runner=runner)
    assert await drv.drive(rid) == "awaiting_human"
    # time passes, the ask expires, the sweep marks it; re-driving errors the run
    assert await store.expire_asks(now=10_000.0) == [rid]
    assert await drv.drive(rid) == "error"
    assert (await store.get_run(rid))["stopped_reason"] == "human_ask_timeout"


# ── supervisor: don't spin on a waiting run; resume on answer ──────────

async def test_supervisor_skips_waiting_then_resumes_on_answer(store, _wire):
    rid = await _mk(store, "email the team the update", ["send it"])
    calls = []

    async def runner(run, step):
        calls.append(step["step_key"])
        return {"ok": True, "output": "sent"}

    sup = BatSupervisor(store, BatDriver(store, attempt_runner=runner))
    assert await sup.tick() == 1          # adopt → drives to awaiting_plan
    await sup.drain()
    assert (await store.get_run(rid))["status"] == "awaiting_plan"
    assert await sup.tick() == 0          # still waiting on the open ask → not re-adopted
    assert calls == []

    await human_ask.answer(store, (await store.latest_open_ask(rid))["id"], "approve")
    assert await sup.tick() == 1          # ask answered → resume
    await sup.drain()
    assert (await store.get_run(rid))["status"] == "done" and calls


async def test_supervisor_expires_overdue_ask_and_errors_run(store, _wire):
    rid = await _mk(store, "do a thing", ["main"])

    async def runner(run, step):
        return {"ask": {"kind": "input", "question": "need it", "expires_at": 100.0}}  # already past

    sup = BatSupervisor(store, BatDriver(store, attempt_runner=runner))
    await sup.tick()
    await sup.drain()
    assert (await store.get_run(rid))["status"] == "awaiting_human"
    # next tick expires the overdue ask, re-adopts, and errors the run
    assert await sup.tick() == 1
    await sup.drain()
    run = await store.get_run(rid)
    assert run["status"] == "error" and run["stopped_reason"] == "human_ask_timeout"
