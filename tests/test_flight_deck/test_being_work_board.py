"""The work board (docs/being-work-board-plan.md): the analytical mind
assigns a board of tasks; the impulsive feet actively pick the one that
suits them, work it, and mark it done / active / refused-with-reason; the
mind reviews the board each tick and adds or drops tasks. This is the
two-way loop between the two brains — the mind assigns, the feet answer.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_instinct as instinct
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, name="Iskra", preset="explorer", stage="adolescent",
                 now=NOW):
    b = store.conceive(OWNER, name, preset=preset, allowance_preset="20M",
                       now=now)
    store.hatch(OWNER, b["slug"], now=now)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=now)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    await life.build_home(store.get(OWNER, b["slug"]))
    world.ensure_village(store, OWNER, now=now)
    return store.get(OWNER, b["slug"])


def _at(store, being, place):
    store._update(being["id"], NOW, location=json.dumps({"at": place}))
    return store.get(OWNER, being["slug"])


# ═══ The board: the store layer ══════════════════════════════════════════

async def test_a_build_task_carries_its_object_kind(store):
    b = await _being(store)
    added = store.add_plan_steps(
        b["id"], [{"kind": "build", "target": "square", "detail": "bench"}],
        now=NOW)
    assert added and added[0]["kind"] == "build"
    steps = store.open_plan_steps(b["id"], now=NOW)
    assert len(steps) == 1
    assert steps[0]["kind"] == "build" and steps[0]["detail"] == "bench"
    assert steps[0]["state"] == "open"


async def test_claim_moves_open_to_active_still_actionable(store):
    b = await _being(store)
    tid = store.add_plan_steps(
        b["id"], [{"kind": "go", "target": "library"}], now=NOW)[0]["id"]
    store.claim_plan_step(b["id"], tid, now=NOW)
    steps = store.open_plan_steps(b["id"], now=NOW)   # open + active
    assert len(steps) == 1 and steps[0]["state"] == "active"
    assert steps[0]["claimed_at"]


async def test_fulfill_links_the_stake_and_leaves_the_board(store):
    b = await _being(store)
    tid = store.add_plan_steps(
        b["id"], [{"kind": "build", "target": "square", "detail": "bench"}],
        now=NOW)[0]["id"]
    store.fulfill_plan_step(b["id"], tid, object_id="obj-42", now=NOW)
    assert store.open_plan_steps(b["id"], now=NOW) == []
    summary = store.board_summary(b["id"], NOW - timedelta(hours=1), now=NOW)
    assert summary["done"] and summary["done"][0]["object_id"] == "obj-42"


async def test_refuse_records_a_reason_the_mind_can_read(store):
    b = await _being(store)
    tid = store.add_plan_steps(
        b["id"], [{"kind": "go", "target": "well"}], now=NOW)[0]["id"]
    store.refuse_plan_step(b["id"], tid, "too far right now", now=NOW)
    assert store.open_plan_steps(b["id"], now=NOW) == []
    refused = store.board_summary(b["id"], NOW - timedelta(hours=1),
                                  now=NOW)["refused"]
    assert refused and refused[0]["note"] == "too far right now"


async def test_the_mind_drops_by_target_or_id(store):
    b = await _being(store)
    a = store.add_plan_steps(b["id"], [
        {"kind": "go", "target": "library"},
        {"kind": "build", "target": "square", "detail": "bench"},
    ], now=NOW)
    # drop one by target name, one by id
    dropped = store.drop_plan_steps(b["id"], ["library", a[1]["id"]], now=NOW)
    assert len(dropped) == 2
    assert store.open_plan_steps(b["id"], now=NOW) == []


async def test_board_is_capped_and_never_doubles(store):
    b = await _being(store)
    many = [{"kind": "go", "target": p} for p in
            ("library", "well", "square", "meadow", "garden", "workshop",
             "old-bench")]
    store.add_plan_steps(b["id"], many, now=NOW)
    assert len(store.open_plan_steps(b["id"], now=NOW)) \
        == constitution.PLAN_STEPS_MAX
    # an identical open task never doubles
    before = len(store.open_plan_steps(b["id"], now=NOW))
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "library"}],
                         now=NOW)
    assert len(store.open_plan_steps(b["id"], now=NOW)) == before


async def test_board_view_keeps_a_short_memory(store):
    b = await _being(store)
    ids = store.add_plan_steps(b["id"], [
        {"kind": "go", "target": "library"},
        {"kind": "go", "target": "well"},
    ], now=NOW)
    store.fulfill_plan_step(b["id"], ids[0]["id"], now=NOW)
    view = store.board_view(b["id"], now=NOW)
    assert len(view["open"]) == 1 and view["open"][0]["target"] == "well"
    assert any(r["state"] == "done" for r in view["recent"])


# ═══ The mind: it plans and edits the board ══════════════════════════════

def test_the_mind_can_plan_a_build_task():
    d = life._normalize_digest(
        {"plan": [{"build": "bench", "at": "the plaza"},
                  {"go": "library"}]})
    kinds = {(s["kind"], s.get("detail")) for s in d["plan"]}
    assert ("build", "bench") in kinds
    assert ("go", "") in kinds
    # the build task's target is the "at" place
    build = next(s for s in d["plan"] if s["kind"] == "build")
    assert build["target"] == "the plaza"


def test_the_mind_can_drop_tasks():
    d = life._normalize_digest({"plan_drop": ["the mill", "square"]})
    assert d["plan_drop"] == ["the mill", "square"]
    # a bare string is accepted too
    d2 = life._normalize_digest({"plan_drop": "the mill"})
    assert d2["plan_drop"] == ["the mill"]
    # junk is dropped
    assert life._normalize_digest({"plan_drop": 5})["plan_drop"] is None


# ═══ The feet: they select, work, or refuse ══════════════════════════════

def test_the_feet_parse_do_and_refuse():
    assert instinct.parse_feet_act('{"act":"do","task":"t1"}') == \
        {"act": "do", "task": "t1"}
    assert instinct.parse_feet_act(
        '{"act":"refuse","task":"t2","why":"too far"}') == \
        {"act": "refuse", "task": "t2", "why": "too far"}
    # aliases fold in; a do with no task is nothing
    assert instinct.parse_feet_act('{"act":"take","task":"t1"}')["act"] == "do"
    assert instinct.parse_feet_act('{"act":"do"}') is None


async def test_feet_do_a_go_task_departs_and_claims_it(store):
    b = _at(store, await _being(store), "home")
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "library"}],
                         now=NOW)
    out = instinct._apply_act(store, b, {"act": "do", "task": "t1"}, now=NOW)
    assert out["act"] == "go" and out["to"] == "library"
    steps = store.open_plan_steps(b["id"], now=NOW)
    assert steps and steps[0]["state"] == "active"
    assert (store.get(OWNER, b["slug"])["location"] or {}).get("to") \
        == "library"


async def test_feet_break_ground_on_a_build_task_at_its_spot(store):
    # a DELIBERATE being (scholar, low impulse) still carries out an
    # assigned build — on_task bypasses the impulse floor.
    b = _at(store, await _being(store, preset="scholar"), "square")
    tid = store.add_plan_steps(
        b["id"], [{"kind": "build", "target": "square", "detail": "bench"}],
        now=NOW)[0]["id"]
    out = instinct._apply_act(store, b, {"act": "do", "task": "t1"}, now=NOW)
    assert out["act"] == "build" and out["kind"] == "bench"
    stakes = store.village_objects(OWNER, state="staked")
    assert len(stakes) == 1 and stakes[0]["id"] == out["id"]
    # the task is done and linked to the real stake
    done = store.board_summary(b["id"], NOW - timedelta(hours=1),
                               now=NOW)["done"]
    assert done and done[0]["object_id"] == out["id"]


async def test_feet_walk_toward_a_far_build_task_before_breaking_ground(store):
    b = _at(store, await _being(store), "home")
    store.add_plan_steps(
        b["id"], [{"kind": "build", "target": "library", "detail": "bench"}],
        now=NOW)
    out = instinct._apply_act(store, b, {"act": "do", "task": "t1"}, now=NOW)
    assert out["act"] == "go" and out["to"] == "library"
    # nothing built yet — the feet must arrive first
    assert store.village_objects(OWNER, state="staked") == []
    assert store.open_plan_steps(b["id"], now=NOW)[0]["state"] == "active"


async def test_feet_refuse_a_task_with_a_reason(store):
    b = _at(store, await _being(store), "home")
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "well"}], now=NOW)
    out = instinct._apply_act(
        store, b, {"act": "refuse", "task": "t1", "why": "too far"}, now=NOW)
    assert out["act"] == "refuse" and out["why"] == "too far"
    refused = store.board_summary(b["id"], NOW - timedelta(hours=1),
                                  now=NOW)["refused"]
    assert refused and refused[0]["note"] == "too far"


# ═══ The interrupt: a task may stop a walk ════════════════════════════════

async def test_a_task_can_stir_a_decision_mid_walk(store):
    b = await _being(store, now=NOW - timedelta(days=2))
    store.depart(OWNER, b["slug"], "library", now=NOW, by="feet")  # mid-walk
    mid = store.get(OWNER, b["slug"])
    assert not (mid["location"] or {}).get("at")            # truly on the road
    # no task on the road → the road decides (None)
    assert instinct.wants_decision(store, mid, NOW + timedelta(minutes=10)) \
        is None
    # a build task waiting → the feet may interrupt the walk
    store.add_plan_steps(
        mid["id"], [{"kind": "build", "target": "square", "detail": "bench"}],
        now=NOW)
    assert instinct.wants_decision(
        store, mid, NOW + timedelta(minutes=10)) == "task"


async def test_the_interrupt_is_rate_limited(store):
    b = _at(store, await _being(store, now=NOW - timedelta(days=2)), "home")
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "library"}],
                         now=NOW)
    # a feet decision just happened — the board must not re-fire immediately
    store.record_event(b["id"], "instinct", {"act": "linger"},
                       now=NOW - timedelta(minutes=2))
    assert instinct.wants_decision(store, store.get(OWNER, b["slug"]), NOW) \
        != "task"


# ═══ The mind sees the board ═════════════════════════════════════════════

async def test_the_mind_meets_its_board_each_wake(store):
    b = await _being(store)
    ids = store.add_plan_steps(b["id"], [
        {"kind": "go", "target": "library"},
        {"kind": "go", "target": "well"},
        {"kind": "build", "target": "square", "detail": "bench"},
    ], now=NOW)
    store.fulfill_plan_step(b["id"], ids[0]["id"], now=NOW)          # done
    store.refuse_plan_step(b["id"], ids[1]["id"], "too far", now=NOW)  # refused
    lines = world.board_percept(store, b, NOW, "wake", True)
    text = " ".join(lines)
    assert "WORK BOARD" in text
    assert "finished" in text and "refused" in text and "too far" in text
    assert "plan_drop" in text          # it teaches the edit syntax
    # silent when nothing is on the board
    b2 = await _being(store, name="Empty")
    assert world.board_percept(store, b2, NOW, "wake", True) == []
