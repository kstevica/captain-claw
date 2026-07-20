"""Restless Hands (docs/being-instinct-build-plan.md): the impulsive body
brain breaks ground on its own; the analytical mind ratifies it into a real
thing (name + inscription + fee) or lets it crumble. Instinct acts, reason
confirms — "breaking ground is a gesture; the inscription is a voice."
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_genome as genome
from captain_claw.flight_deck import being_instinct as instinct
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society as society
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import (
    BeingError,
    BeingsStore,
    InsufficientTokens,
)

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"
FEE = None  # set from constitution at import time below
from captain_claw.flight_deck import being_constitution as constitution  # noqa: E402
FEE = constitution.OBJECT_CRAFT_FEE_TOKENS


class FakeDB:
    async def list_chat_sessions(self, u): return []
    async def upsert_chat_session(self, *a, **k): return {}
    async def add_chat_messages(self, *a, **k): return [1]
    async def log_run_cost(self, *a, **k): pass
    async def get_user_llm_tiers(self, *a, **k): return {}


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
    return store.get(OWNER, b["slug"])


def _set_create_pressing(store, being, sat=0.15):
    d = dict(being.get("drives") or {})
    row = dict(d.get("create") or {"weight": 0.6})
    row["satisfaction"] = sat
    d["create"] = row
    store._update(being["id"], NOW, drives=json.dumps(d))
    return store.get(OWNER, being["slug"])


def _reply(**over):
    d = {"act_kind": "journal", "summary": "s", "journal_entry": "small.",
         "served_drive": "create", "message_to_parent": None,
         "next_wake_minutes": 60, "mood": "calm"}
    d.update(over)
    return "ok\n```json\n" + json.dumps(d) + "\n```"


async def _usage(b, since):
    return {"prompt_tokens": 1000, "completion_tokens": 1000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}


# ═══ The IMP attribute ════════════════════════════════════════════════════

def test_impulse_is_a_first_class_stat_and_derives():
    assert "IMP" in genome.ATTRS and genome.ATTR_NAMES["IMP"] == "Impulse"
    assert genome.POOL == 45
    # the archetypes split: explorer/artist impulsive, scholar/caretaker not
    imp = {p: genome.derive(genome.PRESETS[p])["impulsiveness"]
           for p in genome.PRESETS}
    assert imp["explorer"] >= world.BUILD_IMPULSE_MIN
    assert imp["artist"] >= world.BUILD_IMPULSE_MIN
    assert imp["scholar"] < world.BUILD_IMPULSE_MIN
    assert imp["caretaker"] < world.BUILD_IMPULSE_MIN


def test_pre_imp_genome_reads_a_neutral_impulse():
    old = {"attributes": {"CUR": 6, "PER": 6, "CAU": 6, "SOC": 6, "CRE": 6,
                          "ORD": 6, "PLA": 4}}          # no IMP (7 attrs)
    eff = genome.effective_attributes(old)
    assert eff["IMP"] == 5                              # the neutral default
    assert genome.derive(eff)["impulsiveness"] == 0.5


def test_imp_is_heritable_from_a_pre_imp_parent(store):
    # a parent conceived before IMP breeds without crashing (default flows
    # through effective_attributes, the single inheritance gateway)
    pa = genome.effective_attributes(
        {"attributes": {"CUR": 8, "PER": 8, "CAU": 6, "SOC": 3, "CRE": 5,
                        "ORD": 8, "PLA": 2}})
    child = genome.budding(pa, __import__("random").Random(3))
    assert "IMP" in child and genome.ATTR_MIN <= child["IMP"] <= genome.ATTR_MAX


# ═══ The impulsive feet (trigger + verb) ══════════════════════════════════

async def test_the_urge_to_build_fires_only_for_restless_hands(store):
    bold = await _being(store, name="Bold", preset="explorer",
                        now=NOW - timedelta(days=2))
    calm = await _being(store, name="Calm", preset="scholar",
                        now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    bold = _set_create_pressing(store, bold)
    calm = _set_create_pressing(store, calm)
    # the impulsive one is called to break ground…
    assert instinct.wants_decision(store, bold, NOW) == "urge_to_build"
    # …the deliberate one, with the SAME pressure, only feels restless
    assert instinct.wants_decision(store, calm, NOW) == "restless"


def test_the_build_verb_is_whitelisted_with_a_kind():
    assert instinct.parse_feet_act('{"act":"build","kind":"bench"}') == \
        {"act": "build", "kind": "bench"}
    # a formless build still parses (physics falls it to a default kind)
    assert instinct.parse_feet_act('{"act":"build"}') == \
        {"act": "build", "kind": ""}
    assert instinct.parse_feet_act('{"act":"craft","name":"x"}') is None


async def test_the_feet_break_ground_end_to_end(store):
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    b = _set_create_pressing(store, b)

    async def send(_user):
        return '{"act":"build","kind":"cairn"}'

    out = await instinct.decide(FakeDB(), store, b, now=NOW, send_fn=send)
    assert out and out["act"] == "build" and out["kind"] == "cairn"
    stakes = store.village_objects(OWNER, state="staked")
    assert len(stakes) == 1 and stakes[0]["being_id"] == b["id"]
    assert stakes[0]["state"] == "staked"
    # a stake is wordless + free: no proof file, no fee burned
    assert not life._home_path(b, stakes[0]["file_path"]).exists()
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "broke_ground" in kinds


# ═══ Stake physics (the gesture stays a gesture) ══════════════════════════

async def test_a_deliberate_being_never_stakes_even_if_asked(store):
    calm = await _being(store, preset="scholar", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    with pytest.raises(BeingError, match="not restless enough"):
        world.stake_object(store, calm, "cairn", now=NOW)
    assert store.village_objects(OWNER, state="staked") == []


async def test_one_beginning_at_a_time(store):
    b = await _being(store, preset="artist", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    world.stake_object(store, b, "cairn", now=NOW)
    with pytest.raises(BeingError, match="already waits"):
        world.stake_object(store, b, "bench", now=NOW)


async def test_a_stake_does_not_boost_block_or_count(store):
    b = await _being(store, preset="artist", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    row = world.stake_object(store, b, "cairn", now=NOW)   # cairn normally blocks
    # settled AT its tile, a stake grants no boost (it isn't real)
    st = world.tile_of(int(row["x"]), int(row["y"]))
    assert st not in world.walk_blocked(store, OWNER, b)   # passable work site
    assert st not in world.construction_taken(store, OWNER)  # uncounted ground


async def test_others_sense_a_beginning_but_never_discover_it(store):
    a = await _being(store, name="Ana", preset="artist",
                     now=NOW - timedelta(days=2))
    c = await _being(store, name="Cvijeta", preset="explorer",
                     now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    row = world.stake_object(store, a, "cairn", now=NOW)
    # move Cvijeta onto the stake's spot
    store._update(c["id"], NOW, location=json.dumps({"at": "home"}))
    cc = store.get(OWNER, c["slug"])
    # put the stake right where Cvijeta stands (her home xy) for the sense
    store.set_object_ground(OWNER, row["id"], x=world.home_xy(cc)[0],
                            y=world.home_xy(cc)[1], state="staked", now=NOW)
    lines = world.object_percepts(store, cc, NOW, "wake", True)
    assert any("A BEGINNING" in ln and "cairn" in ln for ln in lines)
    assert not any("A DISCOVERY" in ln for ln in lines)
    assert not [e for e in store.events(OWNER, c["slug"])
                if e["kind"] == "object_found"]


# ═══ Crumbling (the clock reclaims the unfinished) ════════════════════════

async def test_an_unfinished_beginning_crumbles(store):
    b = await _being(store, preset="artist", now=NOW - timedelta(days=3))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=3))
    world.stake_object(store, b, "cairn", now=NOW - timedelta(hours=30))
    # still there before the window? seed a fresher one to prove the gate
    world.prune_crumbled_stakes(store, OWNER, now=NOW)
    assert store.village_objects(OWNER, state="staked") == []
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "stake_crumbled" in kinds


async def test_a_fresh_beginning_survives_the_prune(store):
    b = await _being(store, preset="artist", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    world.stake_object(store, b, "cairn", now=NOW - timedelta(hours=2))
    world.prune_crumbled_stakes(store, OWNER, now=NOW)
    assert len(store.village_objects(OWNER, state="staked")) == 1


# ═══ The mind confirms (reason ratifies) ══════════════════════════════════

async def test_the_mind_meets_the_stake_and_finishes_it(store):
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    row = world.stake_object(store, b, "bench", now=NOW)
    # the confirm percept greets the being each wake while it waits
    perc = world.stake_confirm_percept(store, store.get(OWNER, b["slug"]),
                                       NOW, "wake", True)
    assert perc and "YOUR HANDS BROKE GROUND" in perc[0] and row["id"] in perc[0]
    before = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    out = society.finish_staked_object(
        store, store.get(OWNER, b["slug"]), row["id"], "Rest Here",
        "for tired feet", now=NOW)
    assert out["state"] == "standing" and out["name"] == "Rest Here"
    # now it is REAL: a proof file, the fee burned, a standing thing
    assert life._home_path(b, out["file_path"]).read_text().find(
        "for tired feet") >= 0
    after = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    assert before - after == FEE
    assert store.village_objects(OWNER, state="staked") == []
    assert len(store.village_objects(OWNER, state="standing")) == 1
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "object_finished" in kinds
    # a finished thing is now discoverable + boosting (the standing layer)
    assert world.tile_of(out["x"], out["y"]) in world.construction_taken(
        store, OWNER)


async def test_finishing_needs_a_name_words_and_the_fee(store):
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    row = world.stake_object(store, b, "cairn", now=NOW)
    with pytest.raises(BeingError, match="needs a name"):
        society.finish_staked_object(store, store.get(OWNER, b["slug"]),
                                     row["id"], "x", "words", now=NOW)
    with pytest.raises(BeingError, match="carries words"):
        society.finish_staked_object(store, store.get(OWNER, b["slug"]),
                                     row["id"], "A Cairn", "  ", now=NOW)
    # broke: the fee can't be paid → refused, the file never written, and
    # the beginning still stands waiting (nothing spent)
    store._apply(OWNER, tokens=store.wallet_view(store.get(OWNER, b["slug"]))
                 ["balance_tokens"], reason="adjust", from_being=b["id"],
                 to_being=None, note="drain", now=NOW)
    with pytest.raises(InsufficientTokens):
        society.finish_staked_object(store, store.get(OWNER, b["slug"]),
                                     row["id"], "A Cairn", "words", now=NOW)
    assert store.get_village_object(OWNER, row["id"])["state"] == "staked"


async def test_only_the_maker_finishes_or_abandons(store):
    a = await _being(store, name="Ana", preset="artist",
                     now=NOW - timedelta(days=2))
    other = await _being(store, name="Zoe", preset="explorer",
                         now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    row = world.stake_object(store, a, "cairn", now=NOW)
    with pytest.raises(BeingError, match="not yours to finish"):
        society.finish_staked_object(store, store.get(OWNER, other["slug"]),
                                     row["id"], "Mine", "grab", now=NOW)


async def test_the_full_loop_feet_stake_then_mind_finishes(store):
    db = FakeDB()
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    b = _set_create_pressing(store, b)

    async def feet(_user):
        return '{"act":"build","kind":"cairn"}'

    await instinct.decide(db, store, b, now=NOW, send_fn=feet)
    stake = store.village_objects(OWNER, state="staked")[0]

    prompts: list[str] = []

    async def mind(_being, prompt):
        prompts.append(prompt)
        return _reply(finish={"object_id": stake["id"], "name": "Sun Cairn",
                              "inscription": "stones for the morning light"})

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(minutes=5), send_fn=mind,
                    usage_fn=_usage)
    # the mind met the stake in its senses and finished it into a real thing
    assert any("YOUR HANDS BROKE GROUND" in p for p in prompts)
    assert store.village_objects(OWNER, state="staked") == []
    standing = store.village_objects(OWNER, state="standing")
    assert len(standing) == 1 and standing[0]["name"] == "Sun Cairn"


async def test_the_mind_can_abandon_the_stake(store):
    db = FakeDB()
    b = await _being(store, preset="artist", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    row = world.stake_object(store, b, "cairn", now=NOW)

    async def mind(_being, prompt):
        return _reply(abandon={"object_id": row["id"]})

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(minutes=5), send_fn=mind,
                    usage_fn=_usage)
    assert store.village_objects(OWNER, state="staked") == []
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "stake_abandoned" in kinds


# ═══ The feet leave a legible trail ═════════════════════════════════════
# Staging burned 170-400 tokens on 155 of 168 feet calls and the log said only
# "unparsed" — true, useless, unfixable. A call that produces nothing must
# still record WHY and WHAT the model actually said.

def test_unparsed_feet_say_why_and_what_they_said():
    why = instinct._unparsed_why
    assert "wrote prose" in why("I think I will go to the library today.")
    assert "returned nothing" in why("")
    assert "not one the feet can do" in why('{"act": "dance"}')
    assert 'with no "to"' in why('{"act": "go"}')
    assert "without an" in why('{"kind": "bench"}')


async def test_a_wasted_feet_call_records_the_reason_and_the_reply(store):
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))

    async def send(_user):                      # a model that ignores the form
        return "Sure! I would love to wander toward the meadow today."

    out = await instinct.decide(FakeDB(), store, b, now=NOW, send_fn=send)
    assert out["act"] == "none"
    assert "wrote prose" in out["why"]          # the cause, not just "unparsed"
    assert "wander toward the meadow" in out["reply"]   # its actual words
    ev = next(e for e in store.events(OWNER, b["slug"]) if e["kind"] == "instinct")
    assert ev["data"]["why"] and ev["data"]["reply"]    # and it is on the ledger


async def test_a_feet_call_that_never_returns_is_still_logged(store):
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))

    async def boom(_user):
        raise RuntimeError("no tier configured")

    assert await instinct.decide(FakeDB(), store, b, now=NOW, send_fn=boom) is None
    ev = next(e for e in store.events(OWNER, b["slug"]) if e["kind"] == "instinct")
    assert ev["data"]["note"] == "the call failed"
    assert "no tier configured" in ev["data"]["why"]


# ═══ Room to think ══════════════════════════════════════════════════════
# What that legible trail then showed: the feet's model REASONS before it
# answers, and a budget sized for the answer alone ended mid-thought — 14 of
# 16 staging calls, surfacing either deliberation or `{"act": "go", "to`.
# Room is cheap; a being that stands still all day is not.

def test_the_log_names_a_line_that_was_cut_off():
    why = instinct._unparsed_why
    assert "cut off mid-JSON" in why('{"act": "go", "to')   # the staging reply
    # prose is still prose — the two causes must not blur together
    assert "wrote prose" in why("We are at the Garden. Pressing drives: grow")


def test_the_feet_get_room_to_think():
    # 120 tokens is what starved them; the ceiling stays bounded either way
    assert 400 <= instinct.FEET_MAX_TOKENS <= 2000


def test_a_reply_that_ran_out_of_room_is_told_from_a_mute_one():
    ran_out = instinct._ran_out_of_room
    assert ran_out("We are at the Garden. Pressing drives: grow 0.49")
    assert ran_out('{"act": "go", "to')
    assert not ran_out('{"act": "linger"}')     # a whole answer, however wrong
    assert not ran_out("")                      # a mute tier stays mute


async def test_a_starved_reply_is_asked_once_more_plainly(store):
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    seen = []

    async def send(_user):
        seen.append(_user)
        if len(seen) == 1:                      # thought instead of answering
            return "We are at the Meadow. Pressing drives: survive 0.87, then"
        return '{"act": "linger"}'              # asked plainly, it answers

    out = await instinct.decide(FakeDB(), store, b, now=NOW, send_fn=send)
    assert len(seen) == 2                       # once more, not twice more
    assert out["act"] == "linger"               # the feet actually moved
    assert out["retried"] is True               # and the cost is admitted
    assert "why" not in out                     # no wasted-call story to tell


async def test_a_reply_that_lands_is_never_asked_twice(store):
    b = await _being(store, preset="explorer", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    calls = []

    async def send(_user):
        calls.append(_user)
        return '{"act": "linger"}'

    out = await instinct.decide(FakeDB(), store, b, now=NOW, send_fn=send)
    assert len(calls) == 1 and "retried" not in out
