"""When Bat's planner or the shaping of a factory's checks gives nothing.

On the Pi both timed out on a thinking model: the run went on as one step holding
the whole job, every done-check stayed with the reviewers, and nothing said so.
Pinned here:

* the planner is asked again; a plan on the second try says so;
* no plan after that: a note says why, and a factory's run (or a big task) gets
  three steps (conventions, the work segment by segment, putting it together)
  instead of one; a small task still runs as one step;
* shaping that fails says why, isn't marked done, and is tried again (before the
  next check), up to three times; a reply with no checks counts as a failure;
* planning's writes to a run's config never undo each other (the limit on steps
  at once used to be written from a copy read before the checks were shaped).
"""

from __future__ import annotations

import json

import pytest

from captain_claw.flight_deck import bat_course, bat_loop, bat_routes
from captain_claw.flight_deck.bat_store import BatStore


@pytest.fixture
async def bat(tmp_path, monkeypatch):
    b = BatStore(tmp_path / "bat.db")
    await b.init()

    async def kick():
        return 0

    monkeypatch.setattr(bat_loop, "_STORE", b)
    monkeypatch.setattr(bat_loop, "kick", kick)
    monkeypatch.setattr(bat_loop, "_STOP_CHILDREN", None)
    from captain_claw.flight_deck import vfs_routes
    monkeypatch.setattr(vfs_routes, "_user_root", lambda owner: tmp_path / "vfs" / owner)
    yield b
    await b.close()


class _Reply:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def model(monkeypatch):
    """Answers in turn from ``state["answers"]``: a string, or an exception to raise."""
    state = {"answers": [], "asked": 0}

    class Prov:
        async def complete(self, messages, **kw):
            state["asked"] += 1
            got = state["answers"].pop(0) if state["answers"] else ""
            if isinstance(got, Exception):
                raise got
            return _Reply(got)

    from captain_claw.flight_deck import auth, basna_routes

    async def tiers(_db, _owner):
        return {"reason": {"provider": "x", "model": "m"}}, None

    monkeypatch.setattr(auth, "get_db", lambda: None)
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(bat_routes, "think_call", lambda creds, **kw: (Prov(), 1000))
    return state


async def _run(bat, rid="bat_plan00001", task="Find 100 small US firms with a reason they fit.", **cfg) -> dict:
    await bat.create_run(run_id=rid, owner_id="u1", title="t", task=task, status="planning", config=cfg)
    return await bat.get_run(rid)


async def _notes(bat, rid="bat_plan00001") -> list[str]:
    return [e["message"] for e in await bat.list_events(rid) if e["stage"] == "note"]


PLAN = json.dumps([{"step": "Conventions", "after": []}, {"step": "Segment A", "after": [1]},
                   {"step": "Merge", "after": [2]}])


async def test_the_planner_is_asked_again(bat, model):
    run = await _run(bat)
    model["answers"] = [TimeoutError("timed out"), PLAN]
    steps = await bat_routes._plan_steps(run)
    assert [s["title"] for s in steps] == ["Conventions", "Segment A", "Merge"] and model["asked"] == 2
    assert any("second try (TimeoutError: timed out the first time)" in n for n in await _notes(bat))


async def test_no_plan_a_factory_run_still_gets_three_steps(bat, model):
    run = await _run(bat, factory_id="fac_1")
    model["answers"] = ["", "not json at all"]
    steps = await bat_routes._plan_steps(run)
    assert [s["step_key"] for s in steps] == ["step-1", "step-2", "step-3"]
    assert steps[1]["depends_on"] == ["step-1"] and steps[0]["depends_on"] == []
    assert "shared/conventions.md" in steps[0]["title"] and "segment by segment" in steps[1]["title"]
    note = (await _notes(bat))[-1]
    assert note.startswith("The planner gave no plan (the answer wasn't a plan)") and "three-step plan" in note


async def test_no_plan_a_small_task_is_one_step(bat, model):
    run = await _run(bat, rid="bat_small0001", task="Say hello.")
    model["answers"] = [RuntimeError("down"), RuntimeError("down")]
    steps = await bat_routes._plan_steps(run)
    assert [s["step_key"] for s in steps] == ["main"]
    assert any("RuntimeError: down" in n and "one step" in n for n in await _notes(bat, "bat_small0001"))


RULES = [{"id": "fm1", "text": "At least 100 rows", "severity": "major", "check": {"type": "judge"}}]


async def test_shaping_that_fails_says_so_and_is_tried_again(bat, model):
    await _run(bat, factory_id="fac_1", contract=RULES)
    model["answers"] = [TimeoutError("timed out"), "{}"]
    for _ in range(2):
        await bat_routes._shape_contract(await bat.get_run("bat_plan00001"), {"provider": "x"}, False)
    cfg = (await bat.get_run("bat_plan00001"))["config"]
    assert not cfg.get("contract_shaped") and cfg["contract_shape_tries"] == 2
    notes = await _notes(bat)
    assert any("TimeoutError: timed out" in n and "trying again before the next check" in n for n in notes)
    assert any("the answer had no checks in it" in n for n in notes)
    # the third try works
    model["answers"] = [json.dumps({"constraints": [{"id": "fm1", "text": "x", "check": {"type": "table", "min_rows": 100}}]})]
    await bat_course._shape_again(await bat.get_run("bat_plan00001"))
    cfg = (await bat.get_run("bat_plan00001"))["config"]
    assert cfg["contract_shaped"] is True and cfg["contract"][0]["check"]["type"] == "table"
    assert cfg["contract"][0]["text"] == "At least 100 rows"


async def test_shaping_stops_trying_after_three(bat, model):
    await _run(bat, factory_id="fac_1", contract=RULES, contract_shape_tries=bat_routes.SHAPE_TRIES)
    await bat_routes._shape_contract(await bat.get_run("bat_plan00001"), {"provider": "x"}, False)
    assert model["asked"] == 0


async def test_planning_writes_dont_undo_each_other(bat, model):
    stale = await _run(bat, factory_id="fac_1", contract=RULES)
    model["answers"] = [json.dumps({"constraints": [{"id": "fm1", "text": "x", "check": {"type": "table", "min_rows": 100}}]})]
    await bat_routes._shape_contract(stale, {"provider": "x"}, False)
    # a copy read before the shaping, as the planner held it
    await bat_routes._one_at_a_time(stale, {"fast": {"provider": "ollama", "model": "m", "base_url": "http://x.example"}})
    cfg = (await bat.get_run("bat_plan00001"))["config"]
    assert cfg["max_parallel"] == 1 and cfg["contract_shaped"] is True
    assert cfg["contract"][0]["check"]["type"] == "table"


async def test_shaping_asks_for_no_thinking_and_a_short_answer(bat, monkeypatch):
    """A local Qwen thought 137k characters on the shaping call, looped and answered nothing."""
    from captain_claw.flight_deck import auth, basna_routes
    from captain_claw.llm import reasoning
    seen = {}

    class Prov:
        async def complete(self, messages, **kw):
            seen.update(effort=reasoning.requested_effort(), max_tokens=kw.get("max_tokens"))
            return _Reply(json.dumps({"constraints": [{"id": "fm1", "text": "x", "check": {"type": "table", "min_rows": 100}}]}))

    async def tiers(_db, _owner):
        return {"reason": {"provider": "x", "model": "m"}}, None

    monkeypatch.setattr(auth, "get_db", lambda: None)
    monkeypatch.setattr(basna_routes, "_load_owner_tiers", tiers)
    monkeypatch.setattr(bat_routes, "think_call", lambda creds, **kw: (Prov(), 32768))
    await _run(bat, factory_id="fac_1", contract=RULES)
    await bat_routes._shape_contract(await bat.get_run("bat_plan00001"), {"provider": "x"}, False)
    assert seen == {"effort": "off", "max_tokens": bat_routes.SHAPE_TOKENS}
    assert (await bat.get_run("bat_plan00001"))["config"]["contract_shaped"] is True
