"""Space plan Phase 1 — the ground: a village founded once (deterministic
default; the LLM architect may redraw it), position as a pure function of
the location row and the clock, ``go_to`` as the one digest field that moves
a body, arrivals settled on read at their REAL time, infants toddling."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
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


def _place(pid, x=500, y=500, aff=("read",)):
    return {"id": pid, "name": pid.title(), "x": x, "y": y,
            "affordances": list(aff), "description": "a place"}


# ═══ The ground exists (default village + MAP.md) ═════════════════════════

def test_default_village_is_founded_once_and_mapped(store):
    world.ensure_village(store, OWNER, now=NOW)
    places = store.village_places(OWNER)
    assert world.VILLAGE_MIN_PLACES <= len(places) <= world.VILLAGE_MAX_PLACES
    for p in places:
        assert 40 <= p["x"] <= 960 and 40 <= p["y"] <= 960
        assert p["affordances"] and all(a in world.AFFORDANCES
                                        for a in p["affordances"])
    world.ensure_village(store, OWNER, now=NOW)          # idempotent
    assert len(store.village_places(OWNER)) == len(places)
    text = being_society._commons_path(OWNER, "village/MAP.md").read_text()
    assert "go_to" in text and "the Square" in text and "min" in text
    assert "the Architect" in text


def test_save_village_validates_hard(store):
    with pytest.raises(BeingError, match="holds"):
        store.save_village(OWNER, [_place("a"), _place("b")], now=NOW)
    base = [_place("a", 200, 200), _place("b", 300, 300),
            _place("c", 600, 300), _place("d", 300, 600)]
    with pytest.raises(BeingError, match="reserved"):
        store.save_village(OWNER, base[:3] + [_place("home")], now=NOW)
    with pytest.raises(BeingError, match="duplicate"):
        store.save_village(OWNER, base[:3] + [_place("a", 700, 700)], now=NOW)
    with pytest.raises(BeingError, match="off the plot"):
        store.save_village(OWNER, base[:3] + [_place("e", 990, 500)], now=NOW)
    with pytest.raises(BeingError, match="vocabulary"):
        store.save_village(
            OWNER, base[:3] + [_place("e", 500, 700, aff=("teleport",))],
            now=NOW)
    saved = store.save_village(OWNER, base, now=NOW)
    assert sorted(p["id"] for p in saved) == ["a", "b", "c", "d"]


def test_architect_draft_parses_and_the_store_is_the_gate(store):
    raw = ('Here is the village.\n```json\n'
           + json.dumps({"places": [_place("spring-court", 480, 470,
                                           ("gather", "trade")),
                         _place("reading-room", 200, 300, ("read",)),
                         _place("mudworks", 700, 650, ("create",)),
                         _place("long-grass", 820, 200, ("play",))]})
           + '\n```')
    places = world.parse_architect_places(raw)
    saved = store.save_village(OWNER, places, now=NOW)
    assert len(saved) == 4
    with pytest.raises(BeingError, match="not json"):
        world.parse_architect_places("the village is nice")
    bad = world.parse_architect_places(
        '```json\n' + json.dumps({"places": [
            _place("a", 200, 200), _place("b", 300, 300),
            _place("c", 600, 300),
            _place("spa", 500, 700, ("teleport",))]}) + '\n```')
    with pytest.raises(BeingError, match="vocabulary"):
        store.save_village(OWNER, bad, now=NOW)
    assert sorted(p["id"] for p in store.village_places(OWNER)) == \
        sorted(p["id"] for p in saved)                   # nothing changed


# ═══ Movement: go_to, the walk between ticks, arrival on read ═════════════

async def test_morning_names_the_ground_and_everyone_starts_home(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, b, now=NOW, send_fn=send, usage_fn=_usage)
    assert "YOUR GROUND: you are at home" in prompts[0]
    assert '"go_to"' in prompts[0] and "MAP.md" in prompts[0]
    assert store.get(OWNER, b["slug"])["location"] == {"at": "home"}


async def test_go_to_walks_between_ticks_and_arrival_settles(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))

    async def send_go(being, prompt):
        return _reply(go_to="library")

    await life.tick(db, store, b, now=NOW, send_fn=send_go, usage_fn=_usage)
    bb = store.get(OWNER, b["slug"])
    assert bb["location"].get("to") == "library"
    deps = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "departed"]
    assert deps and deps[0]["to"] == "library" and deps[0]["minutes"] > 0
    mid = world.position_of(
        store, bb, NOW + timedelta(minutes=deps[0]["minutes"] / 2))
    assert mid["to"] == "library" and mid["minutes_left"] > 0
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    later = NOW + timedelta(minutes=deps[0]["minutes"] + 5)
    await life.tick(db, store, store.get(OWNER, b["slug"]), now=later,
                    send_fn=send, usage_fn=_usage)
    bb = store.get(OWNER, b["slug"])
    assert bb["location"] == {"at": "library"}
    arr = [e for e in store.events(OWNER, b["slug"]) if e["kind"] == "arrived"]
    assert arr and arr[0]["data"]["place"] == "library"
    assert arr[0]["at"] < later.isoformat()              # the REAL arrival time
    assert any("You reached the Library" in p for p in prompts)


async def test_the_road_is_felt_on_a_midwalk_wake(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    prompts: list[str] = []

    async def send_go(being, prompt):
        prompts.append(prompt)
        return _reply(go_to="library")

    await life.tick(db, store, b, now=NOW, send_fn=send_go, usage_fn=_usage)
    deps = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "departed"]
    prompts.clear()

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    mid = NOW + timedelta(minutes=max(2, deps[0]["minutes"] // 2))
    await life.tick(db, store, store.get(OWNER, b["slug"]), now=mid,
                    send_fn=send, usage_fn=_usage)
    assert any("on the road to the Library" in p for p in prompts)


async def test_unknown_ground_is_refused_loudly_and_echoes(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))

    async def send_go(being, prompt):
        return _reply(go_to="the moon")

    await life.tick(db, store, b, now=NOW, send_fn=send_go, usage_fn=_usage)
    bb = store.get(OWNER, b["slug"])
    assert bb["location"] == {"at": "home"}
    refs = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "go_to" and "MAP.md" in r["reason"] for r in refs)
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, bb, now=NOW + timedelta(hours=1),
                    send_fn=send, usage_fn=_usage)
    assert any("PHYSICS SAID NO" in p for p in prompts)


async def test_reroute_leaves_from_the_road_and_same_place_is_a_noop(store):
    b = await _being(store, now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    assert store.depart(OWNER, b["slug"], "home", now=NOW)["location"] == \
        {"at": "home"}
    assert [e for e in store.events(OWNER, b["slug"])
            if e["kind"] == "departed"] == []            # no theater recorded
    store.depart(OWNER, b["slug"], "library", now=NOW)
    bb = store.get(OWNER, b["slug"])
    deps = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "departed"]
    half = NOW + timedelta(minutes=deps[0]["minutes"] / 2)
    road_xy = world.position_of(store, bb, half)["xy"]
    bb2 = store.depart(OWNER, b["slug"], "the Square", now=half)
    assert bb2["location"]["to"] == "square"             # name resolved to id
    assert bb2["location"]["origin"] == [road_xy[0], road_xy[1]]
    assert len([e for e in store.events(OWNER, b["slug"])
                if e["kind"] == "departed"]) == 2


async def test_infants_toddle_the_same_ground(store):
    child = await _being(store, name="Brzi", stage="child",
                         now=NOW - timedelta(days=2))
    baby = await _being(store, name="Spori", stage="infant",
                        now=NOW - timedelta(days=2))
    c, i = store.get(OWNER, child["slug"]), store.get(OWNER, baby["slug"])
    assert world.speed_for(i) == pytest.approx(
        world.WALK_SPEED * world.INFANT_SPEED_FACTOR)
    a, z = (100, 100), (500, 500)
    assert world.travel_minutes(i, a, z) == pytest.approx(
        world.travel_minutes(c, a, z) / world.INFANT_SPEED_FACTOR)
    # and nothing gates the destination — infants roam free (user-locked)
    world.ensure_village(store, OWNER, now=NOW)
    far = store.depart(OWNER, baby["slug"], "meadow", now=NOW)
    assert far["location"]["to"] == "meadow"


async def test_fever_turns_the_body_home(store):
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    store.depart(OWNER, b["slug"], "library", now=NOW - timedelta(hours=3))
    for i in range(3):
        store.record_event(b["id"], "tick_timeout", {},
                           now=NOW - timedelta(hours=2, minutes=-i))
    prompts: list[str] = []

    async def send_go(being, prompt):
        prompts.append(prompt)
        return _reply(go_to="meadow")

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send_go, usage_fn=_usage)
    bb = store.get(OWNER, b["slug"])
    assert bb["location"].get("to") == "home" or \
        bb["location"] == {"at": "home"}
    deps = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "departed"]
    assert any(d.get("reason") == "fever" for d in deps)
    refs = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "go_to" and "fever" in r["reason"] for r in refs)
    assert any("turned for home" in p for p in prompts)


async def test_position_is_a_pure_function_of_the_clock(store):
    b = await _being(store, now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    store.depart(OWNER, b["slug"], "square", now=NOW)
    bb = store.get(OWNER, b["slug"])
    assert bb["location"].get("to") == "square"
    # the course was plotted once at depart (world plan Phase 2): position
    # at any instant is the same pure read — halfway in time is halfway
    # along the STORED path, not the beeline
    loc = bb["location"]
    total = float(loc["minutes"])
    pm = world.position_of(store, bb, NOW + timedelta(minutes=total / 2))
    assert pm == world.position_of(store, bb,
                                   NOW + timedelta(minutes=total / 2))
    pts = [(float(p[0]), float(p[1])) for p in loc["path"]]
    assert tuple(pm["xy"]) == world._along(pts, 0.5)
    pe = world.position_of(store, bb, NOW + timedelta(minutes=total + 1))
    assert pe["at"] == "square" and pe["minutes_left"] == 0.0
    # a pure read — the row itself still says "on the road"
    assert store.get(OWNER, b["slug"])["location"].get("to") == "square"
    assert store.vitals(OWNER, b["slug"])["location"].get("to") == "square"
