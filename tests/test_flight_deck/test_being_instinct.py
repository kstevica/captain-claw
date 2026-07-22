"""Body-brain plan Phase 1 — the reflex layer: an instincts toggle, plans
the mind writes for its feet, settle/encounter/fever reflexes riding the
60s loop poll (pure Python, $0), and the planned-milestone gate — a first
visit mints by choice or by plan, never by wandering feet."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_constitution as constitution
from captain_claw.flight_deck import being_instinct as instinct
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society as society
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck import beings_loop
from captain_claw.flight_deck.beings import BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)      # a Wednesday
OWNER = "user-1"


class FakeDB:
    async def list_chat_sessions(self, user_id):
        return []

    async def upsert_chat_session(self, *a, **k):
        return {}

    async def add_chat_messages(self, *a, **k):
        return [1]

    async def log_run_cost(self, *a, **k):
        pass

    async def get_user_llm_tiers(self, *a, **k):
        return {}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


async def _being(store, name="Zvjezdana", stage="child", now=NOW):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=now)
    store.hatch(OWNER, b["slug"], now=now)
    if stage != "infant":
        store.set_stage(OWNER, b["slug"], stage, now=now)
    bb = store.get(OWNER, b["slug"])
    store.set_agent(bb["id"], bb["slug"], 1234, "tok")
    await life.build_home(store.get(OWNER, b["slug"]))
    return store.get(OWNER, b["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "s", "journal_entry": "small.",
         "served_drive": "grow", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(being, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


def _events(store, slug, kind):
    return [e for e in store.events(OWNER, slug) if e["kind"] == kind]


async def _settle_at(store, slug, place, depart_at, **kw):
    store.depart(OWNER, slug, place, now=depart_at, **kw)
    store.settle_location(store.get(OWNER, slug),
                          now=depart_at + timedelta(hours=6))
    return store.get(OWNER, slug)


# ═══ The toggle ═══════════════════════════════════════════════════════════

async def test_instincts_toggle_and_roster(store):
    b = await _being(store)
    assert store.vitals(OWNER, b["slug"])["instincts"] is False
    assert store.instinct_beings() == []
    store.set_instincts(OWNER, b["slug"], True, now=NOW)
    assert store.vitals(OWNER, b["slug"])["instincts"] is True
    assert _events(store, b["slug"], "instincts_set")
    assert [x["slug"] for x in store.instinct_beings()] == [b["slug"]]
    # the roster covers only the LIVING — a paused body has no reflexes
    store.set_state(OWNER, b["slug"], "paused", now=NOW)
    assert store.instinct_beings() == []


# ═══ Plans: lifecycle, caps, lapse ════════════════════════════════════════

async def test_plan_steps_cap_dedup_and_fulfill(store):
    b = await _being(store)
    steps = [{"kind": "go", "target": f"p{i}"} for i in range(7)]
    added = store.add_plan_steps(b["id"], steps, now=NOW)
    assert len(added) == constitution.PLAN_STEPS_MAX
    # an identical open step is never doubled; junk kinds never land
    again = store.add_plan_steps(
        b["id"], [{"kind": "go", "target": "p0"},
                  {"kind": "dance", "target": "p9"}], now=NOW)
    assert again == []
    open_now = store.open_plan_steps(b["id"], now=NOW)
    assert [s["target"] for s in open_now] \
        == [f"p{i}" for i in range(constitution.PLAN_STEPS_MAX)]
    store.fulfill_plan_step(b["id"], open_now[0]["id"], now=NOW)
    assert len(store.open_plan_steps(b["id"], now=NOW)) \
        == constitution.PLAN_STEPS_MAX - 1


async def test_stale_plan_steps_lapse_quietly(store):
    b = await _being(store)
    old = NOW - timedelta(days=constitution.PLAN_LAPSE_DAYS + 1)
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "square"}],
                         now=old)
    store.add_plan_steps(b["id"], [{"kind": "meet", "target": "ada"}],
                         now=NOW)
    open_now = store.open_plan_steps(b["id"], now=NOW)
    assert [s["kind"] for s in open_now] == ["meet"]   # the old walk lapsed


# ═══ Digest: normalization + handlers ═════════════════════════════════════

def test_normalize_digest_plan_and_intend():
    d = life._normalize_digest({
        "plan": [{"go": "Library"}, {"attend": "square"}, {"meet": "Ada"},
                 "junk", {"x": 1}],
        "intend": {"stay": True, "avoid": ["market", ""], "junk": 1}})
    assert d["plan"] == [{"kind": "go", "target": "Library", "detail": ""},
                         {"kind": "go", "target": "square", "detail": ""},
                         {"kind": "meet", "target": "Ada", "detail": ""}]
    assert d["intend"] == {"stay": True, "avoid": ["market"]}
    d2 = life._normalize_digest({"intend": {"junk": 1}, "plan": "walk"})
    assert d2["plan"] is None and d2["intend"] is None


async def test_tick_plans_land_and_junk_is_refused(store):
    db = FakeDB()
    a = await _being(store, name="Ana")
    bb = await _being(store, name="Bura")
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))

    async def send(being, prompt):
        return _reply(plan=[{"go": "library"}, {"meet": "Bura"},
                            {"go": "atlantis"}],
                      intend={"stay": True})

    await life.tick(db, store, store.get(OWNER, a["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    open_now = store.open_plan_steps(store.get(OWNER, a["slug"])["id"],
                                     now=NOW)
    assert {(s["kind"], s["target"]) for s in open_now} \
        == {("go", "library"), ("meet", bb["slug"])}
    refused = [e for e in _events(store, a["slug"], "society_refused")
               if e["data"].get("what") == "plan"]
    assert len(refused) == 1 and "atlantis" in refused[0]["data"]["to"]
    assert store.get(OWNER, a["slug"])["intent"] == {"stay": True}
    assert _events(store, a["slug"], "plan_set")
    assert _events(store, a["slug"], "intent_set")


# ═══ Settle fulfills a planned walk ═══════════════════════════════════════

async def test_settle_fulfills_go_plan_and_marks_arrival(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "library"}],
                         now=NOW - timedelta(hours=7))
    await _settle_at(store, b["slug"], "library", NOW - timedelta(hours=6))
    assert store.open_plan_steps(b["id"], now=NOW) == []
    done = _events(store, b["slug"], "plan_fulfilled")
    assert done and done[0]["data"]["target"] == "library"
    arrived = _events(store, b["slug"], "arrived")[0]       # newest first
    assert arrived["data"]["planned"] is True
    assert arrived["data"]["by"] == "mind"


# ═══ The planned-milestone gate ═══════════════════════════════════════════

async def test_feet_wandering_never_mints_first_visit(store):
    db = FakeDB()
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    await _settle_at(store, b["slug"], "square", NOW - timedelta(hours=6),
                     by="feet")

    async def send(being, prompt):
        return _reply(served_drive="")

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    names = [e["data"].get("name")
             for e in _events(store, b["slug"], "milestone")]
    assert "first_visit_square" not in names


async def test_planned_feet_walk_still_counts_as_discovery(store):
    db = FakeDB()
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "meadow"}],
                         now=NOW - timedelta(hours=7))
    await _settle_at(store, b["slug"], "meadow", NOW - timedelta(hours=6),
                     by="feet")

    async def send(being, prompt):
        return _reply(served_drive="")

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    names = [e["data"].get("name")
             for e in _events(store, b["slug"], "milestone")]
    assert "first_visit_meadow" in names
    assert store.get(OWNER, b["slug"])["drives"]["explore"]["last_served"]


# ═══ Reflexes: settle, encounters, fever ══════════════════════════════════

async def test_reflex_pass_settles_a_finished_walk(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.depart(OWNER, b["slug"], "square", now=NOW - timedelta(hours=6))
    acted = world.reflex_pass(store, store.get(OWNER, b["slug"]), NOW)
    assert acted >= 1
    assert store.get(OWNER, b["slug"])["location"] == {"at": "square"}


async def test_reflex_encounters_between_ticks_once_per_day(store):
    a = await _being(store, name="Ana")
    b = await _being(store, name="Bura")
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.add_plan_steps(a["id"], [{"kind": "meet", "target": b["slug"]}],
                         now=NOW - timedelta(hours=7))
    await _settle_at(store, a["slug"], "square", NOW - timedelta(hours=6))
    await _settle_at(store, b["slug"], "square", NOW - timedelta(hours=6))
    acted = world.reflex_pass(store, store.get(OWNER, a["slug"]), NOW)
    assert acted == 1
    assert len(_events(store, a["slug"], "crossed_paths")) == 1
    assert len(_events(store, b["slug"], "crossed_paths")) == 1
    done = _events(store, a["slug"], "plan_fulfilled")
    assert done and done[0]["data"]["kind"] == "meet"
    # a minute later: the pair already crossed today — nothing new lands
    assert world.reflex_pass(store, store.get(OWNER, a["slug"]),
                             NOW + timedelta(minutes=1)) == 0
    assert len(_events(store, a["slug"], "crossed_paths")) == 1


async def test_reflex_fever_turns_the_body_home(store, monkeypatch):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    await _settle_at(store, b["slug"], "square", NOW - timedelta(hours=6))
    monkeypatch.setattr(world, "fever_state", lambda *a, **k: "collapse")
    acted = world.reflex_pass(store, store.get(OWNER, b["slug"]), NOW)
    assert acted == 1
    departed = _events(store, b["slug"], "departed")[0]     # newest first
    assert departed["data"]["to"] == "home"
    assert departed["data"]["by"] == "feet"
    assert departed["data"]["reason"] == "fever"
    # fevered and homeward: no mingling on the way
    assert _events(store, b["slug"], "crossed_paths") == []


# ═══ The loop pass: roster + quiet hours ══════════════════════════════════

async def test_instinct_pass_respects_quiet_hours(store, monkeypatch):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.set_instincts(OWNER, b["slug"], True, now=NOW)
    store.depart(OWNER, b["slug"], "square", now=NOW - timedelta(hours=6))
    monkeypatch.setattr(beings_loop, "get_store", lambda: store)
    monkeypatch.setattr(beings_loop, "_quiet_window", lambda o: (0, 23))
    assert await beings_loop._instinct_pass(FakeDB(), now=NOW) == 0
    assert store.get(OWNER, b["slug"])["location"].get("to") == "square"
    monkeypatch.setattr(beings_loop, "_quiet_window", lambda o: (22, 8))
    assert await beings_loop._instinct_pass(FakeDB(), now=NOW) >= 1
    assert store.get(OWNER, b["slug"])["location"] == {"at": "square"}


# ═══ The mind hears what the body did ═════════════════════════════════════

async def test_fulfilled_plan_surfaces_as_percept(store):
    db = FakeDB()
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))

    async def send(being, prompt):
        return _reply(served_drive="")

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW - timedelta(hours=3), send_fn=send,
                    usage_fn=_usage)
    store.add_plan_steps(b["id"], [{"kind": "go", "target": "library"}],
                         now=NOW - timedelta(hours=2))
    store.depart(OWNER, b["slug"], "library", now=NOW - timedelta(hours=2))
    # streets made the course longer (world plan Phase 2) — settle well
    # after any honest pace has arrived
    world.reflex_pass(store, store.get(OWNER, b["slug"]),
                      NOW + timedelta(hours=8))
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any(ln.startswith("AS YOU PLANNED") for ln in lines)


async def test_a_fresh_walk_makes_the_mind_speak_of_the_place(store):
    """The mandate half of the arrived-trigger rate limit: the FIRST wake
    after a walk is made to comment on where the feet landed — and only that
    wake (a later tick sees no fresh arrival, so staying on costs no further
    words). Home is the one arrival with no mandate — it is terminal rest."""
    b = await _being(store, now=NOW - timedelta(hours=3))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    bid = b["id"]
    # the mind last woke an hour ago; then the feet walked to the Well
    store.tick_bookkeeping(bid, drives={}, next_wake_at=NOW,
                           now=NOW - timedelta(hours=1))
    store.record_event(bid, "arrived",
                       {"place": "well", "name": "the Well", "by": "feet"},
                       now=NOW - timedelta(minutes=5))
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any("MUST BE ABOUT the Well" in ln and "not optional" in ln
               for ln in lines)
    # the mind witnesses it — a later wake carries no fresh arrival, so
    # staying another tick demands no further comment
    store.tick_bookkeeping(bid, drives={}, next_wake_at=NOW,
                           now=NOW - timedelta(minutes=2))
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert not any("MUST BE ABOUT" in ln for ln in lines)
    # coming HOME is terminal rest — the walk is felt, but nothing is demanded
    store.record_event(bid, "arrived", {"place": "home", "by": "feet"},
                       now=NOW - timedelta(minutes=1))
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any("home" in ln for ln in lines)
    assert not any("MUST BE ABOUT" in ln for ln in lines)


async def test_the_wake_after_a_walk_carries_the_mandate_into_the_prompt(store):
    """End to end: the feet carry the body somewhere, the walk settles, and
    the very next mind tick is handed the speak-of-the-place mandate in its
    actual prompt — not just in percepts_since."""
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=1))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))

    async def send0(being, prompt):
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send0, usage_fn=_usage)          # grounds "last woke"
    store.depart(OWNER, b["slug"], "library", now=NOW + timedelta(minutes=1),
                 by="feet")
    world.reflex_pass(store, store.get(OWNER, b["slug"]),
                      NOW + timedelta(hours=8))               # the walk finishes
    seen: list[str] = []

    async def send(being, prompt):
        seen.append(prompt)
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=9), send_fn=send, usage_fn=_usage)
    assert seen and "MUST BE ABOUT the Library" in seen[0]


async def test_morning_teaches_the_plan_fields_when_instincts_on(store):
    db = FakeDB()
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.set_instincts(OWNER, b["slug"], True, now=NOW)
    seen: list[str] = []

    async def send(being, prompt):
        seen.append(prompt)
        return _reply(served_drive="")

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert any("YOUR FEET WORK A BOARD" in p for p in seen)


# ═══════════════════ Phase 2 — the tiny decision brain ════════════════════

def _garden_file(being, rel="garden/poem.md", text="# a poem\nsmall.\n"):
    p = life._home_path(being, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return rel


def _mark(store, bid, minutes_ago, act="linger"):
    """Pin the feet's last decision — the anchor every trigger gap uses."""
    store.record_event(bid, "instinct", {"act": act},
                       now=NOW - timedelta(minutes=minutes_ago))


async def test_wants_decision_rests_the_feet_until_the_mind_witnesses_a_walk(
        store):
    """The arrived-trigger rate limit: once the feet set the body down
    somewhere new it STAYS there until the next mind tick, which is made to
    speak of the place. Before this a fresh arrival re-fired the feet and a
    being paced the village all day (staging: ~50 walks between one pair of
    hourly ticks). Home is terminal rest — no gate."""
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    bid = b["id"]
    # a long quiet gap would ordinarily stir the feet
    _mark(store, bid, instinct.FEET_IDLE_MINUTES * 2)
    assert instinct.wants_decision(store, b, NOW) == "restless"
    # but a fresh arrival RESTS them — the body stays where it landed
    store.record_event(bid, "arrived", {"place": "square", "by": "feet"},
                       now=NOW - timedelta(minutes=5))
    assert instinct.wants_decision(store, store.get(OWNER, b["slug"]),
                                   NOW) is None
    # even the mind's own go_to lands the same way — reach it, wait to be seen
    store.record_event(bid, "arrived",
                       {"place": "well", "by": "mind", "planned": True},
                       now=NOW - timedelta(minutes=4))
    assert instinct.wants_decision(store, store.get(OWNER, b["slug"]),
                                   NOW) is None
    # the mind ticks (and is made to speak of the place) — the gate lifts and
    # the freed, still long-quiet feet stir again
    store.tick_bookkeeping(bid, drives={},
                           next_wake_at=NOW + timedelta(hours=1),
                           now=NOW - timedelta(minutes=3))
    assert instinct.wants_decision(store, store.get(OWNER, b["slug"]),
                                   NOW) == "restless"
    # arriving HOME is terminal rest — no gate holds them there
    store.record_event(bid, "arrived", {"place": "home", "by": "feet"},
                       now=NOW - timedelta(minutes=1))
    assert instinct.wants_decision(store, store.get(OWNER, b["slug"]),
                                   NOW) == "restless"


async def test_wants_decision_company_still_stirs(store):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    bid = b["id"]
    _mark(store, bid, 10)                       # a fresh decision quiets
    assert instinct.wants_decision(store, b, NOW) is None
    # company crossing the path stirs them; deciding consumes it
    store.record_event(bid, "crossed_paths", {"with": "x", "name": "X"},
                       now=NOW - timedelta(minutes=2))
    assert instinct.wants_decision(store, store.get(OWNER, b["slug"]),
                                   NOW) == "company"
    _mark(store, bid, 1)
    assert instinct.wants_decision(store, store.get(OWNER, b["slug"]),
                                   NOW) is None


async def test_wants_decision_plan_restless_stay_and_road(store):
    p = await _being(store, name="Planka", now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    # an actionable go/build task presses as "task" after FEET_TASK_MINUTES
    # (the active board supersedes the old passive plan nudge for it)
    store.add_plan_steps(p["id"], [{"kind": "go", "target": "library"}],
                         now=NOW - timedelta(hours=1))
    _mark(store, p["id"], instinct.FEET_TASK_MINUTES + 1)
    assert instinct.wants_decision(store, p, NOW) == "task"
    _mark(store, p["id"], 2)                       # too soon — rate-limited
    assert instinct.wants_decision(store, p, NOW) is None
    # a meet-only board is world-fulfilled: it presses gently as "plan"
    m = await _being(store, name="Meetka", now=NOW - timedelta(hours=2))
    store.add_plan_steps(m["id"], [{"kind": "meet", "target": "ada"}],
                         now=NOW - timedelta(hours=1))
    _mark(store, m["id"], instinct.FEET_PLAN_MINUTES + 1)
    assert instinct.wants_decision(store, m, NOW) == "plan"
    # plain restlessness needs the longer gap (no plan on this one)
    r = await _being(store, name="Mirna", now=NOW - timedelta(hours=2))
    _mark(store, r["id"], instinct.FEET_IDLE_MINUTES + 1)
    assert instinct.wants_decision(store, r, NOW) == "restless"
    _mark(store, r["id"], 10)                     # a fresh decision quiets
    assert instinct.wants_decision(store, r, NOW) is None
    later = NOW + timedelta(minutes=instinct.FEET_IDLE_MINUTES)
    assert instinct.wants_decision(store, r, later) == "restless"
    # the mind's stay pin rests the feet entirely
    store.set_intent(r["id"], {"stay": True}, now=later)
    assert instinct.wants_decision(store, store.get(OWNER, r["slug"]),
                                   later) is None
    store.set_intent(r["id"], None, now=later)
    # mid-walk there is nothing to decide — the road is the decision
    store.depart(OWNER, r["slug"], "square", now=later)
    assert instinct.wants_decision(store, store.get(OWNER, r["slug"]),
                                   later) is None


async def test_fevered_feet_never_decide(store, monkeypatch):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    monkeypatch.setattr(world, "fever_state", lambda *a, **k: "collapse")
    assert instinct.wants_decision(store, b, NOW) is None


async def test_decide_walks_the_feet_and_meters_the_spend(store):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.credit_allowance(b["id"], now=NOW)
    b = store.get(OWNER, b["slug"])
    prompts: list[str] = []

    async def send(prompt):
        prompts.append(prompt)
        return '{"act": "go", "to": "library"}'

    res = await instinct.decide(None, store, b, now=NOW, send_fn=send)
    assert res["act"] == "go" and res["to"] == "library"
    assert res["trigger"] == "restless"
    assert "One line of JSON now." in prompts[0]
    assert store.get(OWNER, b["slug"])["location"].get("to") == "library"
    dep = _events(store, b["slug"], "departed")[0]
    assert dep["data"]["by"] == "feet"
    ev = _events(store, b["slug"], "instinct")[0]
    assert ev["data"]["act"] == "go" and ev["data"]["tokens"] > 0
    assert any(r.get("note") == "instinct"
               for r in store.ledger(OWNER, b["slug"]))


async def test_decide_honors_the_avoid_pin(store):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.credit_allowance(b["id"], now=NOW)
    store.set_intent(b["id"], {"avoid": ["library"]}, now=NOW)
    b = store.get(OWNER, b["slug"])

    async def send(prompt):
        return '{"act": "go", "to": "library"}'

    res = await instinct.decide(None, store, b, now=NOW, send_fn=send)
    assert res["act"] == "none" and "avoid" in res["note"]
    assert store.get(OWNER, b["slug"])["location"] == {"at": "home"}
    assert _events(store, b["slug"], "instinct")     # journaled, quietly


async def test_junk_output_leaves_the_feet_still(store):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.credit_allowance(b["id"], now=NOW)
    b = store.get(OWNER, b["slug"])

    async def send(prompt):
        return "I believe I shall wander the meadow at dusk!"

    res = await instinct.decide(None, store, b, now=NOW, send_fn=send)
    assert res["act"] == "none" and res["note"] == "unparsed"
    assert store.get(OWNER, b["slug"])["location"] == {"at": "home"}
    refused = [e for e in _events(store, b["slug"], "society_refused")]
    assert refused == []                             # never nags the mind


async def test_decide_browse_reads_the_stalls(store):
    seller = await _being(store, name="Zvj", now=NOW - timedelta(days=1))
    rel = _garden_file(store.get(OWNER, seller["slug"]))
    society.market_sell(store, store.get(OWNER, seller["slug"]), rel,
                        "A Sea Poem", 3, now=NOW - timedelta(hours=1))
    b = await _being(store, name="Kupac", now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.credit_allowance(b["id"], now=NOW)
    b = store.get(OWNER, b["slug"])

    async def send(prompt):
        return '{"act": "browse"}'

    res = await instinct.decide(None, store, b, now=NOW, send_fn=send)
    assert res["act"] == "browse" and res["stalls"] == 1
    browsed = _events(store, b["slug"], "browsed")[0]
    assert browsed["data"]["titles"] == ["A Sea Poem"]
    lines = life.percepts_since(store, store.get(OWNER, b["slug"]))
    assert any(ln.startswith("Your feet idled past the stalls")
               for ln in lines)


async def test_broke_feet_do_not_think(store):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    # drain the wallet — no headroom above the reserve, feet too hungry
    view = store.wallet_view(b)
    store.burn(b["id"], view["balance_tokens"], reason="adjust",
               note="drain", now=NOW)
    called: list[int] = []

    async def send(prompt):
        called.append(1)
        return '{"act": "linger"}'

    res = await instinct.decide(None, store, store.get(OWNER, b["slug"]),
                                now=NOW, send_fn=send)
    assert res is None and called == []


async def test_feet_prompt_respects_the_hard_cap(store, monkeypatch):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    system, user = instinct.feet_prompt(store, b, NOW, "restless")
    cap_chars = instinct.FEET_CONTEXT_CAP * 4
    assert len(system) + len(user) <= cap_chars      # default cap holds
    monkeypatch.setattr(instinct, "FEET_CONTEXT_CAP", 300)
    system, user = instinct.feet_prompt(store, b, NOW, "restless")
    assert len(system) + len(user) <= 300 * 4        # tightened cap holds


async def test_instinct_pass_reaches_the_decision_brain(store, monkeypatch):
    b = await _being(store, now=NOW - timedelta(hours=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.set_instincts(OWNER, b["slug"], True, now=NOW)
    monkeypatch.setattr(beings_loop, "get_store", lambda: store)
    monkeypatch.setattr(beings_loop, "_quiet_window", lambda o: (22, 8))
    seen: list[str] = []

    async def fake_decide(db, store_, being, now=None):
        seen.append(being["slug"])
        return {"act": "linger"}

    monkeypatch.setattr(instinct, "decide", fake_decide)
    acted = await beings_loop._instinct_pass(FakeDB(), now=NOW)
    assert seen == [b["slug"]] and acted >= 1
