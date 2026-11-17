"""A First Mate factory's Bat run never stops to ask its owner.

Pinned here:

* the start gate lets a factory's run through, whatever its plan says, and
  never widens its grants: mail it wasn't granted stays refused by the
  workers' mail guard (``mail_write: deny``); other runs still ask;
* a worker's ``ask_human`` on a factory run is declined, not raised: the run
  keeps going and the worker is told nobody answers;
* an ask left open on a factory run (raised before factory runs stopped
  asking) is answered for it by First Mate, and approving its plan, by anyone,
  grants nothing beyond the charter;
* a factory run's tie-break ends the way such a run ends now, without
  "accepted by the owner";
* its workers are told nobody answers, instead of to call ``ask_human``.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from captain_claw.flight_deck import bat_loop, bat_routes, human_ask
from captain_claw.flight_deck.bat_loop import BatDriver
from captain_claw.flight_deck.bat_routes import _AskReq, _build_step_prompt, agent_ask
from captain_claw.flight_deck.bat_store import BatStore

LEADS = ("Collect 500 leads with name, location and a reason. Find public email addresses for each "
         "verified candidate, then send them the list.")
FACTORY = {"source": "first_mate", "factory_id": "fac_0123456789abcdef", "fm_run": "fmr_0123456789abcdef"}


@pytest.fixture
async def store(tmp_path):
    s = BatStore(tmp_path / "bat.db")
    await s.init()
    yield s
    await s.close()


@pytest.fixture(autouse=True)
def _wire(monkeypatch):
    rung: list[dict] = []

    async def rec(ask):
        rung.append(ask)

    saved = (bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, human_ask._NOTIFY)
    bat_loop.set_gate_check(bat_routes._gate_check)
    bat_loop._PLANNER = bat_loop._default_planner
    bat_loop._JUDGE = bat_loop._default_judge
    human_ask.set_notifier(rec)
    human_ask._SECRETS.clear()
    bat_routes._SECRET_ANSWERS.clear()
    monkeypatch.setattr(bat_routes, "_resolve", lambda body: "u1")
    yield rung
    bat_loop._GATE_CHECK, bat_loop._PLANNER, bat_loop._JUDGE, _old = saved
    human_ask.set_notifier(_old)


@pytest.fixture
def kicks(monkeypatch):
    seen: list[int] = []

    async def kick():
        seen.append(1)

    monkeypatch.setattr(bat_loop, "kick", kick)
    return seen


async def _run(store, config, status="planning", task=LEADS):
    rid = f"bat_{uuid.uuid4().hex[:8]}"
    await store.create_run(run_id=rid, owner_id="u1", title="Leads", task=task,
                           config={"steps": ["Find public email addresses", "Send the list"], **config},
                           status=status)
    return rid


async def _done(run, step):
    return {"ok": True, "output": "did it"}


# ── the start gate ─────────────────────────────────────────────────────

async def test_a_factory_run_goes_through_the_gate_on_its_charters_grants(store, _wire):
    rid = await _run(store, FACTORY)
    assert await BatDriver(store, attempt_runner=_done).drive(rid) == "done"
    run = await store.get_run(rid)
    assert await store.latest_ask(rid) is None and _wire == []  # nobody was asked, nothing rang
    assert not run["config"].get("email_allowed") and "gate_reason" not in run["config"]
    assert bat_routes._bat_automation(run)["mail_write"] == "deny"  # sending stays refused
    other = await _run(store, {})
    assert await BatDriver(store, attempt_runner=_done).drive(other) == "awaiting_plan"  # others still ask


async def test_a_carry_out_keeps_exactly_the_charters_grants(store):
    rid = await _run(store, {**FACTORY, "plan_approved": True, "email_allowed": True},
                     task="Send the reply email to Ana. Then buy the upgrade.")
    assert await BatDriver(store, attempt_runner=_done).drive(rid) == "done"
    cfg = (await store.get_run(rid))["config"]
    assert cfg["email_allowed"] is True and not cfg.get("spend_allowed")  # the purchase is refused, not asked
    assert bat_routes._spend_policy(await store.get_run(rid))["enabled"] is False


# ── a worker's question ────────────────────────────────────────────────

async def test_a_workers_question_is_declined_and_the_run_goes_on(store, _wire):
    bat_loop.set_store(store)
    rid = await _run(store, FACTORY, status="running")
    got = await agent_ask(_AskReq(owner_id="u1", session_id=rid, step_key="s1", kind="secret",
                                  question="what is the 2FA code?"))
    assert got == {"status": "declined", "message": bat_routes.FACTORY_NO_ANSWER}
    assert (await store.get_run(rid))["status"] == "running"
    assert await store.latest_ask(rid) is None and _wire == []
    other = await _run(store, {}, status="running")
    got = await agent_ask(_AskReq(owner_id="u1", session_id=other, step_key="s1", question="which site?"))
    assert got["status"] == "requested" and (await store.get_run(other))["status"] == "awaiting_human"


async def test_the_tool_tells_the_worker_nobody_answers(monkeypatch):
    from captain_claw.tools import bat_ask
    real = httpx.AsyncClient

    def answer(request):
        return httpx.Response(200, json={"status": "declined", "message": bat_routes.FACTORY_NO_ANSWER})

    monkeypatch.setenv("CLAW_BAT_SESSION", "bat_x")
    monkeypatch.setattr(bat_ask.AskHumanTool, "_get_fd_url", lambda self, **kw: "http://fd.test")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(answer), **kw))
    res = await bat_ask.AskHumanTool().execute(question="which site?")
    assert not res.success and res.error == bat_routes.FACTORY_NO_ANSWER


# ── asks left open from before ─────────────────────────────────────────

async def _park(store, rid, kind, **cfg):
    run = await store.get_run(rid)
    await store.set_config(rid, {**run["config"], **cfg})
    ask = await human_ask.raise_ask(store, run_id=rid, owner="u1", kind=kind, question="?",
                                    step_key="s1" if kind in ("input", "secret") else "")
    await store.set_status(rid, "awaiting_plan" if kind == "plan_approval" else "awaiting_human")
    return ask


async def test_first_mate_answers_a_parked_plan_and_it_runs_on_the_charter(store, kicks):
    # Parked the way Bat parked a factory's run before: planned, then asked.
    rid = await _run(store, {})
    assert await BatDriver(store, attempt_runner=_done).drive(rid) == "awaiting_plan"
    run = await store.get_run(rid)
    await store.set_config(rid, {**run["config"], **FACTORY})
    assert "send email" in run["config"]["gate_reason"]
    ask = (await store.latest_open_ask(rid))["id"]
    assert await bat_routes.release_factory_ask(store, await store.get_run(rid)) is True and kicks == [1]
    got = await store.get_ask(ask)
    assert (got["status"], got["answer"]) == ("answered", "approve")
    assert await BatDriver(store, attempt_runner=_done).drive(rid) == "done"
    cfg = (await store.get_run(rid))["config"]
    assert cfg["plan_approved"] is True and not cfg.get("email_allowed")  # the gate's reason grants nothing
    assert await bat_routes.release_factory_ask(store, await store.get_run(rid)) is False  # nothing left


async def test_a_parked_question_gets_no_answer_and_the_step_runs_again(store, kicks):
    rid = await _run(store, FACTORY)
    await _park(store, rid, "secret")
    assert await bat_routes.release_factory_ask(store, await store.get_run(rid)) is True
    assert await bat_routes._gate_check(store, await store.get_run(rid)) == "proceed"
    assert (rid, "s1") not in bat_routes._SECRET_ANSWERS


async def test_a_parked_tie_break_ends_without_the_owner(store, kicks):
    rid = await _run(store, FACTORY)
    await _park(store, rid, "tie_break")
    assert await bat_routes.release_factory_ask(store, await store.get_run(rid)) is True
    assert await bat_routes._gate_check(store, await store.get_run(rid)) == "proceed"
    assert "tie_break" not in (await store.get_run(rid))["config"]  # never "accepted by the owner"


async def test_other_runs_keep_waiting(store, kicks):
    rid = await _run(store, {})
    ask = await _park(store, rid, "plan_approval", gate_reason="send email")
    assert await bat_routes.release_factory_ask(store, await store.get_run(rid)) is False and kicks == []
    assert (await store.get_ask(ask))["status"] == "open"


# ── what its workers are told ──────────────────────────────────────────

def test_its_workers_are_told_nobody_answers():
    step = {"step_key": "s1", "title": "Find public email addresses"}
    mine = _build_step_prompt({"task": LEADS, "config": FACTORY}, step, [])
    assert "ask_human" not in mine and "nobody answers" in mine.lower()
    assert "`ask_human`" in _build_step_prompt({"task": LEADS, "config": {}}, step, [])
