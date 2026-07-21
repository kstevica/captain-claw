"""World-shaping plan Phase 1 — made things: a being crafts a real thing
(a proof file in its home + a burned token fee) and places it on open
ground. The commons and its ring refuse an explicit ask (the law, taught);
an at-your-feet placement slides out instead; occupied ground snaps; the
area-scaled cap and the per-being share bound the clutter; a standing
thing is walkable ground ('object:<id>') by its own name."""

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
from captain_claw.flight_deck.beings import (
    BeingError,
    BeingsStore,
    InsufficientTokens,
)

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"
FEE = constitution.OBJECT_CRAFT_FEE_TOKENS


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


def _craft(store, b, kind="cairn", name="Sun Cairn",
           words="stones for the morning light", now=NOW):
    return society.craft_object(store, store.get(OWNER, b["slug"]),
                                kind, name, words, now=now)


def _legal_asks(store, being, n):
    """Deterministic open-ground tile centers an explicit ask may name —
    outside the commons ring, the lane, and every cottage."""
    civic = world._civic_zone(store, OWNER)
    homes: set = set()
    for r in store.list(OWNER):
        homes |= set(world.home_tiles(r))
    out = []
    for ty in range(3, world.GRID_H - 3, 4):
        for tx in range(world.HOME_LANE_TX + 3, world.GRID_W - 3, 4):
            if (tx, ty) in civic or (tx, ty) in homes:
                continue
            out.append(world.tile_center(tx, ty))
            if len(out) >= n:
                return out
    return out


# ═══ Crafting (the making is a real file + a burned fee) ══════════════════

async def test_craft_makes_a_real_file_and_burns_the_fee(store):
    b = await _being(store)
    before = store.wallet_view(b)["balance_tokens"]
    row = _craft(store, b)
    assert row["state"] == "held" and row["kind"] == "cairn"
    assert row["file_path"] == f"garden/works/{row['id']}.md"
    assert row["affordance"] == "remember"
    p = life._home_path(b, row["file_path"])
    text = p.read_text(encoding="utf-8")
    assert "# Sun Cairn" in text and "stones for the morning light" in text
    assert b["slug"] in text                       # provenance names the maker
    after = store.wallet_view(store.get(OWNER, b["slug"]))["balance_tokens"]
    assert before - after == FEE
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "object_crafted" in kinds
    assert any(e["kind"] == "milestone"
               and e["data"].get("name") == "first_craft"
               for e in store.events(OWNER, b["slug"]))


async def test_craft_validation_is_the_law(store):
    b = await _being(store)
    with pytest.raises(BeingError, match="vocabulary is fixed"):
        _craft(store, b, kind="tower")
    with pytest.raises(BeingError, match="needs a name"):
        _craft(store, b, name="x")
    with pytest.raises(BeingError, match="carries words"):
        _craft(store, b, words="   ")
    baby = await _being(store, name="Beba", stage="infant")
    with pytest.raises(BeingError, match="childhood"):
        _craft(store, baby)
    assert store.village_objects(OWNER) == []      # nothing slipped through


async def test_craft_refused_when_broke(store, monkeypatch):
    b = await _being(store)
    monkeypatch.setattr(constitution, "OBJECT_CRAFT_FEE_TOKENS",
                        10**12)
    with pytest.raises(InsufficientTokens, match="craft fee"):
        _craft(store, b)
    assert store.village_objects(OWNER) == []


# ═══ Placing (geometry decides: refuse, snap, cap) ════════════════════════

async def test_place_at_your_feet_from_home_lands_in_the_yard(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    out = world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                             now=NOW)
    assert out["state"] == "standing"
    t = world.tile_of(out["x"], out["y"])
    hx, hy = world.home_xy(b)
    assert abs(t[0] - hx // world.TILE) <= world.OBJECT_SNAP_TILES + 1
    assert t not in world._civic_zone(store, OWNER)
    assert t not in set(world.home_tiles(b))       # beside the cottage, not on
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "object_placed" in kinds


async def test_place_at_your_feet_at_a_civic_place_slides_out(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    store.depart(OWNER, b["slug"], "meadow", now=NOW - timedelta(hours=6))
    bb = store.get(OWNER, b["slug"])
    store.settle_location(bb, now=NOW)
    row = _craft(store, b)
    out = world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                             now=NOW)
    assert out["state"] == "standing"              # not refused — slid out
    assert world.tile_of(out["x"], out["y"]) not in \
        world._civic_zone(store, OWNER)


async def test_an_explicit_ask_for_the_commons_is_refused(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    sq = next(p for p in store.village_places(OWNER) if p["id"] == "square")
    row = _craft(store, b)
    with pytest.raises(BeingError, match="commons isn't yours"):
        world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                           x=sq["x"], y=sq["y"], now=NOW)
    # the buffer ring refuses too — one tile past the footprint edge
    tiles = world.footprint_tiles(sq)
    edge = (min(t[0] for t in tiles) - 1, tiles[0][1])
    ex, ey = world.tile_center(*edge)
    with pytest.raises(BeingError, match="commons isn't yours"):
        world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                           x=ex, y=ey, now=NOW)
    # the lane belongs to the commons as well
    lx, ly = world.tile_center(world.HOME_LANE_TX, world.GRID_H // 2)
    with pytest.raises(BeingError, match="commons isn't yours"):
        world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                           x=lx, y=ly, now=NOW)
    assert store.get_village_object(OWNER, row["id"])["state"] == "held"


async def test_anothers_yard_is_refused(store):
    b = await _being(store)
    other = await _being(store, name="Lada")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    civic = world._civic_zone(store, OWNER)
    yard = next(t for t in world.home_tiles(other) if t not in civic)
    yx, yy = world.tile_center(*yard)
    with pytest.raises(BeingError, match="another's yard"):
        world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                           x=yx, y=yy, now=NOW)


async def test_occupied_ground_snaps_to_the_nearest_open_tile(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    meta = store.get_village_meta(OWNER)
    roads = {(int(t[0]), int(t[1])) for t in (meta.get("roads") or [])}
    civic = world._civic_zone(store, OWNER)
    homes = set(world.home_tiles(b))
    road = next(t for t in sorted(roads)
                if t not in civic and t not in homes
                and 3 <= t[0] <= world.GRID_W - 4
                and 3 <= t[1] <= world.GRID_H - 4)
    rx, ry = world.tile_center(*road)
    row = _craft(store, b)
    out = world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                             x=rx, y=ry, now=NOW)
    t = world.tile_of(out["x"], out["y"])
    assert t != road and t not in roads            # slid off the street
    assert t not in civic
    # and the taken ground now guards future construction
    assert t in world.construction_taken(store, OWNER)


async def test_the_village_cap_holds_and_unplacing_frees(store, monkeypatch):
    monkeypatch.setattr(constitution, "OBJECT_AREA_PER_SLOT", 250_000)  # cap 4
    monkeypatch.setattr(constitution, "OBJECT_MIN_PER_BEING", 5)  # share ≥ cap
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    asks = _legal_asks(store, b, 5)
    rows = []
    for i in range(4):
        r = _craft(store, b, name=f"Mark {i + 1}", words="a mark")
        world.place_object(store, store.get(OWNER, b["slug"]), r["id"],
                           x=asks[i][0], y=asks[i][1], now=NOW)
        rows.append(r)
    fifth = _craft(store, b, name="One Too Many", words="a mark")
    with pytest.raises(BeingError, match="holds 4 made things"):
        world.place_object(store, store.get(OWNER, b["slug"]), fifth["id"],
                           x=asks[4][0], y=asks[4][1], now=NOW)
    # moving a STANDING thing never re-checks the cap
    moved = world.place_object(store, store.get(OWNER, b["slug"]),
                               rows[0]["id"], x=asks[4][0], y=asks[4][1],
                               now=NOW)
    assert moved["state"] == "standing"
    # unplacing frees the slot
    world.unplace_object(store, store.get(OWNER, b["slug"]), rows[1]["id"],
                         now=NOW)
    out = world.place_object(store, store.get(OWNER, b["slug"]), fifth["id"],
                             x=asks[1][0], y=asks[1][1], now=NOW)
    assert out["state"] == "standing"


async def test_the_per_being_share_holds(store, monkeypatch):
    monkeypatch.setattr(constitution, "OBJECT_AREA_PER_SLOT", 250_000)  # cap 4
    monkeypatch.setattr(constitution, "OBJECT_MIN_PER_BEING", 1)
    b = await _being(store)
    await _being(store, name="Lada")                # roster of 2 → share 2
    world.ensure_village(store, OWNER, now=NOW)
    asks = _legal_asks(store, b, 3)
    for i in range(2):
        r = _craft(store, b, name=f"Mark {i + 1}", words="a mark")
        world.place_object(store, store.get(OWNER, b["slug"]), r["id"],
                           x=asks[i][0], y=asks[i][1], now=NOW)
    third = _craft(store, b, name="Third", words="a mark")
    with pytest.raises(BeingError, match="your hands keep 2"):
        world.place_object(store, store.get(OWNER, b["slug"]), third["id"],
                           x=asks[2][0], y=asks[2][1], now=NOW)


async def test_unplace_is_yours_alone_and_honest(store):
    b = await _being(store)
    other = await _being(store, name="Lada")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    with pytest.raises(BeingError, match="already in your hands"):
        world.unplace_object(store, store.get(OWNER, b["slug"]), row["id"],
                             now=NOW)
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW)
    with pytest.raises(BeingError, match="not yours to take up"):
        world.unplace_object(store, store.get(OWNER, other["slug"]),
                             row["id"], now=NOW)
    out = world.unplace_object(store, store.get(OWNER, b["slug"]),
                               row["id"], now=NOW)
    assert out["state"] == "held"
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "object_removed" in kinds


# ═══ A standing thing is walkable ground ══════════════════════════════════

async def test_go_to_a_made_thing_by_its_name(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW)
    assert store.resolve_place_ref(OWNER, "the Sun Cairn") == \
        f"object:{row['id']}"
    bb = store.depart(OWNER, b["slug"], "Sun Cairn", now=NOW)
    assert bb["location"]["to"] == f"object:{row['id']}"
    mid = world.position_of(store, bb, NOW + timedelta(minutes=1))
    assert mid["to"] == f"object:{row['id']}"
    settled = store.settle_location(store.get(OWNER, b["slug"]),
                                    now=NOW + timedelta(hours=8))
    assert settled and settled["name"] == "Sun Cairn"
    arrived = [e for e in store.events(OWNER, b["slug"])
               if e["kind"] == "arrived"]
    assert arrived and arrived[0]["data"]["name"] == "Sun Cairn"
    # …and a place NAME always wins a collision with an object's
    assert store.resolve_place_ref(OWNER, "meadow") == "meadow"


async def test_a_made_things_id_survives_being_resolved_twice(store):
    """A walk resolves its destination TWICE — once where the being's words
    are read, once again inside depart — so an id this layer MINTS has to
    come back through it whole. It didn't: 'object:' only ever encoded, and
    every being who set out for a made thing was told the thing wasn't
    there. Zvjezdana learned the name of Lada's hearth from a letter, walked
    for it, and the village answered `no place called 'object:tiho-svjetlo'`
    — the resolved id, right there in the refusal."""
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW)
    pid = store.resolve_place_ref(OWNER, "Sun Cairn")
    assert store.resolve_place_ref(OWNER, pid) == pid          # idempotent
    assert store.depart(OWNER, b["slug"], pid, now=NOW)["location"]["to"] \
        == pid
    # Decoding the namespace is not a licence to skip the ground's own law:
    # a thing that no longer stands is no longer walkable, by either word.
    world.unplace_object(store, store.get(OWNER, b["slug"]), row["id"],
                         now=NOW)
    assert store.resolve_place_ref(OWNER, pid) is None
    assert store.resolve_place_ref(OWNER, "Sun Cairn") is None
    assert store.resolve_place_ref(OWNER, "object:no-such-thing") is None


async def test_the_mind_walks_to_a_made_thing_it_only_heard_named(store):
    """The whole staging path end to end: the tick names a thing in words,
    being_life resolves it, depart resolves it again — and the feet leave
    the ground. A refusal here means the round trip broke."""
    db = FakeDB()
    b = await _being(store, now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW)

    async def send_go(being, prompt):
        return _reply(go_to="Sun Cairn")

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send_go, usage_fn=_usage)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "society_refused" not in kinds
    assert store.get(OWNER, b["slug"])["location"]["to"] \
        == f"object:{row['id']}"


async def test_two_beings_at_one_made_thing_cross_paths(store):
    """Co-presence works on object ground out of the box — a cairn in the
    wilds is a meeting place the moment two beings walk to it."""
    a = await _being(store, name="Ana")
    c = await _being(store, name="Cvijeta")
    world.ensure_village(store, OWNER, now=NOW - timedelta(days=1))
    row = _craft(store, a, now=NOW - timedelta(hours=20))
    ask = _legal_asks(store, a, 1)[0]
    world.place_object(store, store.get(OWNER, a["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW - timedelta(hours=20))
    for s in (a, c):
        store.depart(OWNER, s["slug"], "Sun Cairn",
                     now=NOW - timedelta(hours=10))
        store.settle_location(store.get(OWNER, s["slug"]), now=NOW)
    lines = world.encounters(store, store.get(OWNER, a["slug"]), NOW, "wake")
    assert any("Cvijeta is here at Sun Cairn" in ln for ln in lines)
    kinds = [e["kind"] for e in store.events(OWNER, a["slug"])]
    assert "crossed_paths" in kinds


async def test_a_lifted_thing_is_broken_ground(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW)
    store.depart(OWNER, b["slug"], "Sun Cairn", now=NOW)
    world.unplace_object(store, store.get(OWNER, b["slug"]), row["id"],
                         now=NOW + timedelta(minutes=1))
    # the walk's ground vanished mid-road — the being resolves home,
    # the pattern every broken walk already follows
    pos = world.position_of(store, store.get(OWNER, b["slug"]),
                            NOW + timedelta(hours=8))
    assert pos["at"] == "home"


# ═══ The tick (digest → physics, refusals loud) ═══════════════════════════

async def test_the_tick_crafts_places_and_refuses_loudly(store):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    b = await _being(store, now=born)
    world.ensure_village(store, OWNER, now=born)

    async def send_craft(being, prompt):
        return _reply(craft={"kind": "bench", "name": "Rest Here",
                             "inscription": "for tired feet"})

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send_craft, usage_fn=_usage)
    objs = store.village_objects(OWNER)
    assert len(objs) == 1 and objs[0]["state"] == "held"
    assert life._home_path(b, objs[0]["file_path"]).exists()

    prompts: list[str] = []

    async def send_place(being, prompt):
        prompts.append(prompt)
        return _reply(place={"object_id": objs[0]["id"]})

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send_place,
                    usage_fn=_usage)
    placed = store.get_village_object(OWNER, objs[0]["id"])
    assert placed["state"] == "standing"
    # the held work was offered back to the being before it placed
    assert any('"place": {"object_id"' in p for p in prompts)
    assert any('"craft": {"kind"' in p for p in prompts)

    async def send_junk(being, prompt):
        return _reply(craft={"kind": "tower", "name": "Babel",
                             "inscription": "up"})

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=2), send_fn=send_junk,
                    usage_fn=_usage)
    refs = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "craft" and "vocabulary" in r["reason"]
               for r in refs)


async def test_arriving_at_an_object_never_burns_a_place_milestone(store):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    b = await _being(store, now=born)
    world.ensure_village(store, OWNER, now=born)
    row = _craft(store, b, now=born + timedelta(hours=1))
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=born + timedelta(hours=2))
    store.depart(OWNER, b["slug"], "Sun Cairn",
                 now=NOW - timedelta(hours=9))

    async def send(being, prompt):
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    names = [e["data"].get("name") for e in store.events(OWNER, b["slug"])
             if e["kind"] == "milestone"]
    assert not any(str(n).startswith("first_visit_object:") for n in names)


def test_feet_cannot_craft():
    assert instinct.parse_feet_act(
        '{"act": "craft", "kind": "bench", "name": "No"}') is None
    assert instinct.parse_feet_act('{"act": "place", "to": "square"}') is None


# ═══ Phase 2 — function (boosts) + discovery (see / sense / urge) ═════════

def _legal_ask_near(store, tx, ty):
    """The nearest legal tile center to a WANTED tile — directional
    placement for sense-line tests, same filters the snap applies."""
    civic = world._civic_zone(store, OWNER)
    homes: set = set()
    for r in store.list(OWNER):
        homes |= set(world.home_tiles(r))
    for ring in range(0, 14):
        for dy in range(-ring, ring + 1):
            for dx in range(-ring, ring + 1):
                if max(abs(dx), abs(dy)) != ring:
                    continue
                t = (tx + dx, ty + dy)
                if not (2 <= t[0] <= world.GRID_W - 3
                        and 2 <= t[1] <= world.GRID_H - 3):
                    continue
                if t in civic or t in homes:
                    continue
                return world.tile_center(*t)
    raise AssertionError("no legal tile near the wanted spot")


def _set_drive(store, being, name, satisfaction):
    d = dict(being.get("drives") or {})
    row = dict(d.get(name) or {"weight": 0.5})
    row["satisfaction"] = satisfaction
    d[name] = row
    store._update(being["id"], NOW, drives=json.dumps(d))
    return store.get(OWNER, being["slug"])


def _settle_at_object(store, b, name, now=NOW):
    store.depart(OWNER, b["slug"], name, now=now - timedelta(hours=6))
    store.settle_location(store.get(OWNER, b["slug"]), now=now)
    return store.get(OWNER, b["slug"])


async def test_boost_factors_full_for_others_reduced_for_own(store):
    a = await _being(store, name="Ana")
    c = await _being(store, name="Cvijeta")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, a, kind="bench", name="Rest Here", words="sit")
    ask = _legal_asks(store, a, 1)[0]
    world.place_object(store, store.get(OWNER, a["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW - timedelta(hours=7))
    cc = _settle_at_object(store, c, "Rest Here")
    assert world.drive_boost_factors(store, cc, NOW).get("grow") \
        == world.PLACE_BOOST                      # another's work: full
    aa = _settle_at_object(store, a, "Rest Here")
    assert world.drive_boost_factors(store, aa, NOW).get("grow") \
        == world.OBJECT_OWN_BOOST                 # your own: reduced
    lada = await _being(store, name="Lada")       # far away at home
    assert "grow" not in world.drive_boost_factors(
        store, store.get(OWNER, lada["slug"]), NOW)


async def test_the_boost_reaches_the_serve(store):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    a = await _being(store, name="Ana", now=born)
    c = await _being(store, name="Cvijeta", now=born)
    ctl = await _being(store, name="Kontrola", now=born)
    world.ensure_village(store, OWNER, now=born)
    row = _craft(store, a, kind="bench", name="Rest Here", words="sit",
                 now=born + timedelta(hours=1))
    ask = _legal_asks(store, a, 1)[0]
    world.place_object(store, store.get(OWNER, a["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=born + timedelta(hours=2))
    _settle_at_object(store, c, "Rest Here")
    for b in (c, ctl):                             # identical starting sat
        _set_drive(store, store.get(OWNER, b["slug"]), "grow", 0.5)

    async def send(being, prompt):
        return _reply(served_drive="grow")

    for b in (c, ctl):
        await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                        send_fn=send, usage_fn=_usage)
    sat_bench = store.get(OWNER, c["slug"])["drives"]["grow"]["satisfaction"]
    sat_home = store.get(OWNER, ctl["slug"])["drives"]["grow"]["satisfaction"]
    assert sat_bench > sat_home                    # 1.5× damp landed


async def test_discovery_fires_once_reads_the_face_and_serves_explore(store):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    a = await _being(store, name="Ana", now=born)
    c = await _being(store, name="Cvijeta", now=born)
    world.ensure_village(store, OWNER, now=born)
    row = _craft(store, a, now=born + timedelta(hours=1))
    ask = _legal_asks(store, a, 1)[0]
    world.place_object(store, store.get(OWNER, a["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=born + timedelta(hours=2))
    _settle_at_object(store, c, "Sun Cairn")
    _set_drive(store, store.get(OWNER, c["slug"]), "explore", 0.5)
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, store.get(OWNER, c["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    assert any("A DISCOVERY: a cairn stands here" in p
               and '"Sun Cairn"' in p and "Ana's work" in p
               and "stones for the morning light" in p for p in prompts)
    ev = [e for e in store.events(OWNER, c["slug"])
          if e["kind"] == "object_found"]
    assert len(ev) == 1 and ev[0]["data"]["id"] == row["id"]
    sat = store.get(OWNER, c["slug"])["drives"]["explore"]["satisfaction"]
    assert sat > 0.55                              # the landmark paid explore
    prompts.clear()
    await life.tick(db, store, store.get(OWNER, c["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send,
                    usage_fn=_usage)
    assert not any("A DISCOVERY" in p for p in prompts)   # once per life
    assert len([e for e in store.events(OWNER, c["slug"])
                if e["kind"] == "object_found"]) == 1


async def test_your_own_work_is_silent_ground(store):
    a = await _being(store, name="Ana")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, a)
    ask = _legal_asks(store, a, 1)[0]
    world.place_object(store, store.get(OWNER, a["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW - timedelta(hours=7))
    aa = _settle_at_object(store, a, "Sun Cairn")
    assert world.object_percepts(store, aa, NOW, "wake", True) == []
    assert not [e for e in store.events(OWNER, a["slug"])
                if e["kind"] == "object_found"]


async def test_sense_lines_point_and_bound_and_stay_nameless(store):
    a = await _being(store, name="Ana")
    c = await _being(store, name="Cvijeta")
    world.ensure_village(store, OWNER, now=NOW)
    hx, hy = world.home_xy(c)
    east = _legal_ask_near(store, world.GRID_W - 6, hy // world.TILE)
    for i, ask in enumerate([east,
                             _legal_ask_near(store, 20, 42),
                             _legal_ask_near(store, 34, 6)]):
        r = _craft(store, a, name=f"Mark {i + 1}", words="a mark")
        world.place_object(store, store.get(OWNER, a["slug"]), r["id"],
                           x=ask[0], y=ask[1], now=NOW - timedelta(hours=7))
    cc = _set_drive(store, store.get(OWNER, c["slug"]), "explore", 0.95)
    lines = world.object_percepts(store, cc, NOW, "wake", True)
    assert lines and len(lines) <= world.OBJECT_SENSE_LINES
    assert all("Mark" not in ln for ln in lines)   # nameless texture
    assert any("east" in ln for ln in lines)       # the nearest points east
    assert all(("short walk" in ln) or ("good walk" in ln)
               or ("far across" in ln) for ln in lines)
    # not a morning → no pulls; dreams never survey
    assert world.object_percepts(store, cc, NOW, "wake", False) == []
    assert world.object_percepts(store, cc, NOW, "dream", True) == []


async def test_the_urge_needs_a_hungry_explore_and_offers_the_road(store):
    a = await _being(store, name="Ana")
    c = await _being(store, name="Cvijeta")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, a)
    ask = _legal_asks(store, a, 1)[0]
    world.place_object(store, store.get(OWNER, a["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW - timedelta(hours=7))
    sated = _set_drive(store, store.get(OWNER, c["slug"]), "explore", 0.95)
    calm = world.object_percepts(store, sated, NOW, "wake", True)
    assert calm and not any("AN URGE" in ln for ln in calm)
    hungry = _set_drive(store, store.get(OWNER, c["slug"]), "explore", 0.2)
    urged = world.object_percepts(store, hungry, NOW, "wake", True)
    assert any("AN URGE" in ln and '"Sun Cairn"' in ln and '"go_to"' in ln
               for ln in urged)
    # the hearsay name truly walks — the urge's road is real ground
    assert store.resolve_place_ref(OWNER, "Sun Cairn") == \
        f"object:{row['id']}"


async def test_found_things_stop_pulling(store):
    a = await _being(store, name="Ana")
    c = await _being(store, name="Cvijeta")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, a)
    ask = _legal_asks(store, a, 1)[0]
    world.place_object(store, store.get(OWNER, a["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW - timedelta(hours=7))
    store.milestone(c["id"], f"found_object_{row['id']}",
                    {"object": row["id"]}, now=NOW)
    cc = _set_drive(store, store.get(OWNER, c["slug"]), "explore", 0.2)
    assert world.object_percepts(store, cc, NOW, "wake", True) == []


async def test_blocking_kinds_block_but_the_goal_stays_enterable(store):
    a = await _being(store, name="Ana")
    c = await _being(store, name="Cvijeta")
    world.ensure_village(store, OWNER, now=NOW)
    cairn = _craft(store, a, kind="cairn", name="Stone Stack", words="stone")
    bench = _craft(store, a, kind="bench", name="Soft Seat", words="sit")
    asks = _legal_asks(store, a, 2)
    world.place_object(store, store.get(OWNER, a["slug"]), cairn["id"],
                       x=asks[0][0], y=asks[0][1], now=NOW)
    world.place_object(store, store.get(OWNER, a["slug"]), bench["id"],
                       x=asks[1][0], y=asks[1][1], now=NOW)
    crow = store.get_village_object(OWNER, cairn["id"])
    brow = store.get_village_object(OWNER, bench["id"])
    ct = world.tile_of(crow["x"], crow["y"])
    blocked = world.walk_blocked(store, OWNER, store.get(OWNER, c["slug"]))
    assert ct in blocked                                    # stone stands
    assert world.tile_of(brow["x"], brow["y"]) not in blocked
    # a straight line through the stone bends around it…
    tiles = world._astar(blocked, set(), (ct[0] - 3, ct[1]),
                         (ct[0] + 3, ct[1]))
    assert tiles and ct not in tiles
    # …while a walk TO the stone still ends before it
    bb = store.depart(OWNER, c["slug"], "Stone Stack", now=NOW)
    assert bb["location"]["to"] == f"object:{cairn['id']}"
    settled = store.settle_location(store.get(OWNER, c["slug"]),
                                    now=NOW + timedelta(hours=8))
    assert settled and settled["place"] == f"object:{cairn['id']}"


# ═══ Phase 4 — home as your canvas ════════════════════════════════════════

async def test_naming_your_home_is_ungated_and_daily(store):
    baby = await _being(store, name="Beba", stage="infant")
    world.ensure_village(store, OWNER, now=NOW)
    b = store.set_home_name(OWNER, baby["slug"], "Mala Koliba", now=NOW)
    assert b["home_name"] == "Mala Koliba"          # an infant may keep house
    kinds = [e["kind"] for e in store.events(OWNER, baby["slug"])]
    assert "home_named" in kinds
    assert any(e["kind"] == "milestone"
               and e["data"].get("name") == "named_home"
               for e in store.events(OWNER, baby["slug"]))
    with pytest.raises(BeingError, match="already bears"):
        store.set_home_name(OWNER, baby["slug"], "Mala Koliba", now=NOW)
    with pytest.raises(BeingError, match="live with it a day"):
        store.set_home_name(OWNER, baby["slug"], "Druga Koliba",
                            now=NOW + timedelta(hours=2))
    b = store.set_home_name(OWNER, baby["slug"], "Druga Koliba",
                            now=NOW + timedelta(days=1))
    assert b["home_name"] == "Druga Koliba"
    with pytest.raises(BeingError, match="2–40"):
        store.set_home_name(OWNER, baby["slug"], "x",
                            now=NOW + timedelta(days=2))


async def test_dressing_your_home_speaks_the_vocab(store):
    b = await _being(store)
    out = store.set_home_look(OWNER, b["slug"], "moss", "sage", now=NOW)
    assert out["home_look"] == {"roof": "moss", "wall": "sage"}
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "home_styled" in kinds
    with pytest.raises(BeingError, match="roofs come in"):
        store.set_home_look(OWNER, b["slug"], "thatch", "sage", now=NOW)
    with pytest.raises(BeingError, match="walls come in"):
        store.set_home_look(OWNER, b["slug"], "ember", "marble", now=NOW)


async def test_the_tick_names_and_dresses_the_home(store):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    b = await _being(store, now=born)
    world.ensure_village(store, OWNER, now=born)
    prompts: list[str] = []

    async def send(being, prompt):
        prompts.append(prompt)
        return _reply(home_name="Kuća od Vjetra",
                      home_look={"roof": "dusk"})       # wall fills plaster

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    fresh = store.get(OWNER, b["slug"])
    assert fresh["home_name"] == "Kuća od Vjetra"
    assert fresh["home_look"] == {"roof": "dusk", "wall": "plaster"}
    # the offer taught it while unnamed…
    assert any('"home_name"' in p for p in prompts)
    prompts.clear()

    async def send_quiet(being, prompt):
        prompts.append(prompt)
        return _reply()

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=send_quiet,
                    usage_fn=_usage)
    # …and falls silent once the house is kept
    assert not any('"home_name"' in p for p in prompts)
    # junk vocab through the tick refuses on the record
    async def send_junk(being, prompt):
        return _reply(home_look={"roof": "thatch"})

    await life.tick(db, store, store.get(OWNER, b["slug"]),
                    now=NOW + timedelta(hours=2), send_fn=send_junk,
                    usage_fn=_usage)
    refs = [e["data"] for e in store.events(OWNER, b["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "home_look" and "roofs come in" in r["reason"]
               for r in refs)


async def test_the_yard_is_cap_exempt_but_never_lawless(store, monkeypatch):
    monkeypatch.setattr(constitution, "OBJECT_AREA_PER_SLOT", 500_000)  # cap 2
    monkeypatch.setattr(constitution, "OBJECT_MIN_PER_BEING", 5)
    b = await _being(store)
    other = await _being(store, name="Lada")
    world.ensure_village(store, OWNER, now=NOW)
    asks = _legal_asks(store, b, 3)
    for i in range(2):                                  # commons full at 2
        r = _craft(store, b, name=f"Mark {i + 1}", words="a mark")
        world.place_object(store, store.get(OWNER, b["slug"]), r["id"],
                           x=asks[i][0], y=asks[i][1], now=NOW)
    blocked = _craft(store, b, name="Third", words="a mark")
    with pytest.raises(BeingError, match="holds 2 made things"):
        world.place_object(store, store.get(OWNER, b["slug"]), blocked["id"],
                           x=asks[2][0], y=asks[2][1], now=NOW)
    # …but the same thing SETS DOWN AT HOME freely (at-feet, from home)
    out = world.place_object(store, store.get(OWNER, b["slug"]),
                             blocked["id"], now=NOW)
    assert out["state"] == "standing"
    assert world.tile_of(out["x"], out["y"]) in world.home_yard_tiles(b)
    # yard works never count against the commons — a sibling still
    # gets refused by the full commons, not by yard clutter
    lr = _craft(store, other, name="Lada Mark", words="a mark")
    with pytest.raises(BeingError, match="holds 2 made things"):
        world.place_object(store, store.get(OWNER, other["slug"]), lr["id"],
                           x=asks[2][0], y=asks[2][1], now=NOW)
    # and another's yard stays another's — the law outranks the freedom
    civic = world._civic_zone(store, OWNER)
    yard = next(t for t in world.home_tiles(b) if t not in civic)
    yx, yy = world.tile_center(*yard)
    with pytest.raises(BeingError, match="another's yard"):
        world.place_object(store, store.get(OWNER, other["slug"]), lr["id"],
                           x=yx, y=yy, now=NOW)


async def test_home_rides_every_surface(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    store.set_home_name(OWNER, b["slug"], "Mala Koliba", now=NOW)
    store.set_home_look(OWNER, b["slug"], "slate", "timber", now=NOW)
    v = store.vitals(OWNER, b["slug"])
    assert v["home_name"] == "Mala Koliba"
    assert v["home_look"] == {"roof": "slate", "wall": "timber"}
    payload = world.village_map_payload(store, OWNER, now=NOW)
    me = next(e for e in payload["beings"] if e["slug"] == b["slug"])
    assert me["home_name"] == "Mala Koliba"
    assert me["home_look"]["roof"] == "slate"
    prof = life.public_profile(store, store.get(OWNER, b["slug"]))
    assert prof["home_name"] == "Mala Koliba"
    assert prof["place"] == {"kind": "home", "name": "Mala Koliba"}


# ═══ Phase 5 — the civic hand (steward + parent) ══════════════════════════

async def _adult(store, name, now=NOW):
    b = await _being(store, name=name, stage="adolescent", now=now)
    return b


async def test_the_steward_raises_a_public_work_on_the_commons(store):
    b = await _adult(store, "Vlada")               # sole adult → the steward
    world.ensure_village(store, OWNER, now=NOW)
    assert world.current_steward(store, OWNER, NOW) == b["slug"]
    sq = next(p for p in store.village_places(OWNER) if p["id"] == "square")
    row = _craft(store, b, kind="lantern", name="Square Light",
                 words="so the square is never dark")
    # a normal placement AT the square would refuse…
    with pytest.raises(BeingError, match="commons isn't yours"):
        world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                           x=sq["x"], y=sq["y"], now=NOW)
    # …but the STEWARD raises it as a public work
    out = world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                             x=sq["x"], y=sq["y"], steward=True, now=NOW)
    assert out["state"] == "standing" and int(out["civic"]) == 1
    assert world.tile_of(out["x"], out["y"]) in world._civic_zone(store, OWNER)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "civic_placed" in kinds
    payload = world.village_map_payload(store, OWNER, now=NOW)
    entry = next(o for o in payload["objects"] if o["id"] == row["id"])
    assert entry["civic"] is True


async def test_a_stewards_own_yard_work_is_not_a_public_work(store):
    b = await _adult(store, "Vlada")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b, name="Home Bench", words="mine")
    # steward=True opens the commons but a thing set down AT HOME is not civic
    out = world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                             steward=True, now=NOW)
    assert int(out["civic"]) == 0
    assert world.tile_of(out["x"], out["y"]) in world.home_yard_tiles(b)
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "object_placed" in kinds and "civic_placed" not in kinds


async def test_civic_works_stand_outside_the_being_cap(store, monkeypatch):
    monkeypatch.setattr(constitution, "OBJECT_AREA_PER_SLOT", 1_000_000)  # cap1
    monkeypatch.setattr(constitution, "OBJECT_MIN_PER_BEING", 1)
    b = await _adult(store, "Vlada")
    world.ensure_village(store, OWNER, now=NOW)
    # fill the whole open-ground cap with one private work
    r1 = _craft(store, b, name="Lone Mark", words="a mark")
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), r1["id"],
                       x=ask[0], y=ask[1], now=NOW)
    # a second private work is refused (cap 1, full)…
    r2 = _craft(store, b, name="Second", words="a mark")
    asks = _legal_asks(store, b, 3)
    with pytest.raises(BeingError, match="holds 1 made"):
        world.place_object(store, store.get(OWNER, b["slug"]), r2["id"],
                           x=asks[1][0], y=asks[1][1], now=NOW)
    # …but a STEWARD public work on the commons ignores the cap
    sq = next(p for p in store.village_places(OWNER) if p["id"] == "square")
    out = world.place_object(store, store.get(OWNER, b["slug"]), r2["id"],
                             x=sq["x"], y=sq["y"], steward=True, now=NOW)
    assert int(out["civic"]) == 1


async def test_the_steward_renames_a_place_keeping_its_id(store, monkeypatch):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    b = await _being(store, name="Vlada", stage="adolescent", now=born)
    world.ensure_village(store, OWNER, now=born)
    monkeypatch.setattr(world, "current_steward",
                        lambda *a, **k: b["slug"])

    async def send(being, prompt):
        return _reply(rename_place={"place": "the Square", "name": "the Heart",
                                    "why": "it beats at the center"})

    await life.tick(db, store, store.get(OWNER, b["slug"]), now=NOW,
                    send_fn=send, usage_fn=_usage)
    p = store.get_place(OWNER, "square")
    assert p["id"] == "square" and p["name"] == "the Heart"   # id preserved
    kinds = [e["kind"] for e in store.events(OWNER, b["slug"])]
    assert "place_renamed" in kinds
    mapmd = society._commons_path(OWNER, "village/MAP.md").read_text()
    assert "the Heart" in mapmd and "the Square" not in mapmd
    # the new name is walkable by both its id and its new name
    assert store.resolve_place_ref(OWNER, "the Heart") == "square"


async def test_the_steward_redescribes_but_a_non_steward_cannot(store,
                                                                monkeypatch):
    db = FakeDB()
    born = NOW - timedelta(days=2)
    steward = await _being(store, name="Aaa", stage="adolescent", now=born)
    other = await _being(store, name="Zzz", stage="adolescent", now=born)
    world.ensure_village(store, OWNER, now=born)
    monkeypatch.setattr(world, "current_steward",
                        lambda *a, **k: steward["slug"])

    async def redescribe(being, prompt):
        return _reply(redescribe_place={"place": "the Library",
                                        "description": "where the quiet lives"})

    await life.tick(db, store, store.get(OWNER, steward["slug"]), now=NOW,
                    send_fn=redescribe, usage_fn=_usage)
    assert "quiet lives" in store.get_place(OWNER, "library")["description"]
    # a NON-steward's civic edit is refused, not applied
    async def other_try(being, prompt):
        return _reply(rename_place={"place": "the Library", "name": "MINE",
                                    "why": "grab"})

    await life.tick(db, store, store.get(OWNER, other["slug"]),
                    now=NOW + timedelta(hours=1), send_fn=other_try,
                    usage_fn=_usage)
    assert store.get_place(OWNER, "library")["name"] == "the Library"
    refs = [e["data"] for e in store.events(OWNER, other["slug"])
            if e["kind"] == "society_refused"]
    assert any(r["what"] == "rename_place" and "steward" in r["reason"]
               for r in refs)


async def test_update_place_gates_and_leaves_the_ground_alone(store):
    world.ensure_village(store, OWNER, now=NOW)
    before = store.get_place(OWNER, "garden")
    out = store.update_place(OWNER, "garden", name="the Green",
                             description="patient rows", now=NOW)
    assert out["id"] == "garden" and out["name"] == "the Green"
    assert out["x"] == before["x"] and out["y"] == before["y"]
    assert out["affordances"] == before["affordances"]          # untouched
    with pytest.raises(BeingError, match="2–60"):
        store.update_place(OWNER, "garden", name="x", now=NOW)
    with pytest.raises(BeingError, match="nothing to change"):
        store.update_place(OWNER, "garden", now=NOW)
    with pytest.raises(Exception):
        store.update_place(OWNER, "no-such-place", name="Ghost", now=NOW)


# ═══ The parent's own hand (parent-build) ═════════════════════════════════

async def test_the_parent_places_a_thing_anywhere(store):
    await _being(store, name="Ana")
    world.ensure_village(store, OWNER, now=NOW)
    sq = next(p for p in store.village_places(OWNER) if p["id"] == "square")
    # the parent may place ON the commons (the keeper tends everything)
    o = world.place_parent_object(store, OWNER, "fountain", "Keeper's Spring",
                                  "drink and rest", sq["x"], sq["y"], now=NOW)
    assert o["state"] == "standing" and o["being_id"] == "parent"
    assert world.tile_of(o["x"], o["y"]) in world._civic_zone(store, OWNER)
    assert world._object_face(store, o) == "drink and rest"   # from commons
    payload = world.village_map_payload(store, OWNER, now=NOW)
    e = next(x for x in payload["objects"] if x["id"] == o["id"])
    assert e["parent"] is True and e["by_name"] == "the village's keeper"
    with pytest.raises(BeingError, match="vocabulary is fixed"):
        world.place_parent_object(store, OWNER, "tower", "X", "", 500, 500,
                                  now=NOW)


async def test_beings_discover_and_are_boosted_by_a_parent_object(store):
    b = await _being(store, name="Cvijeta", now=NOW - timedelta(days=2))
    world.ensure_village(store, OWNER, now=NOW)
    ask = _legal_asks(store, b, 1)[0]
    o = world.place_parent_object(store, OWNER, "bench", "A Quiet Seat",
                                  "for weary feet", ask[0], ask[1], now=NOW)
    _settle_at_object(store, b, "A Quiet Seat")
    # a parent bench (someone else's) grants the FULL boost to a being on it
    assert world.drive_boost_factors(
        store, store.get(OWNER, b["slug"]), NOW).get("grow") \
        == world.PLACE_BOOST
    # and it is discovered like any standing thing
    lines = world.object_percepts(store, store.get(OWNER, b["slug"]),
                                  NOW, "wake", True)
    assert any("A DISCOVERY" in ln and "A Quiet Seat" in ln
               and "keeper" in ln for ln in lines)
    assert o["id"] in [e["data"]["id"] for e in store.events(OWNER, b["slug"])
                       if e["kind"] == "object_found"]


async def test_a_beings_own_work_is_not_the_parents_to_lift(store):
    b = await _being(store, name="Ana")
    world.ensure_village(store, OWNER, now=NOW)
    row = _craft(store, b)
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW)
    got = store.get_village_object(OWNER, row["id"])
    assert got["being_id"] != "parent"     # the route guards on this being_id


# ═══ Road-building (the parent paints streets) ════════════════════════════

async def test_parent_paints_and_lifts_a_road(store):
    world.ensure_village(store, OWNER, now=NOW)
    carved = len(store.get_village_meta(OWNER)["roads"])
    store.toggle_manual_road(OWNER, 30, 10, now=NOW)
    store.toggle_manual_road(OWNER, 30, 11, now=NOW)
    meta = store.get_village_meta(OWNER)
    assert meta["roads_manual"] == [[30, 10], [30, 11]]
    eff = world.effective_roads(meta)
    assert (30, 10) in eff and (30, 11) in eff
    assert len(eff) == carved + 2                      # carved ∪ painted
    # a painted road is real ground: it feeds the walk grid + build-guard
    assert (30, 10) in world.construction_taken(store, OWNER)
    # lifting toggles it off
    store.toggle_manual_road(OWNER, 30, 10, now=NOW)
    assert (30, 10) not in world.effective_roads(store.get_village_meta(OWNER))


async def test_a_painted_road_survives_a_recarve(store):
    world.ensure_village(store, OWNER, now=NOW)
    store.toggle_manual_road(OWNER, 32, 40, now=NOW)
    world.refresh_layout(store, OWNER, now=NOW)         # the auto-carve reruns
    assert (32, 40) in world.effective_roads(store.get_village_meta(OWNER))
    # …and it rides the map payload for the client to draw
    payload = world.village_map_payload(store, OWNER, now=NOW)
    assert [32, 40] in payload["roads"]


# ═══ Grow map (the parent enlarges the plot) ══════════════════════════════

async def test_grow_the_plot_scales_grid_and_room(store):
    world.ensure_village(store, OWNER, now=NOW)
    assert world.grid_dims(store, OWNER) == (50, 50)
    out = store.set_plot_size(OWNER, 1800, now=NOW)
    assert out["plot_w"] == 1800
    assert world.grid_dims(store, OWNER) == (90, 90)
    assert world.village_map_payload(store, OWNER, now=NOW)["plot"] == 1800
    # the new room is buildable: the keeper places far out east
    o = world.place_parent_object(store, OWNER, "bench", "Far Bench",
                                  "out east", 1600, 1600, now=NOW)
    assert world.tile_of(o["x"], o["y"])[0] >= 60      # beyond the old grid


async def test_grow_is_clamped_snapped_and_grow_only(store):
    world.ensure_village(store, OWNER, now=NOW)
    assert store.set_plot_size(OWNER, 99999, now=NOW)["plot_w"] \
        == world.PLOT_MAX                              # clamped up
    # grow-only: a smaller ask floors at the standard plot (never shrinks
    # below), and any size snaps to a whole tile grid
    assert store.set_plot_size(OWNER, 100, now=NOW)["plot_w"] \
        == world.PLOT_MIN
    got = store.set_plot_size(OWNER, 1333, now=NOW)["plot_w"]
    assert got % world.TILE == 0 and got == 1320


async def test_homes_stay_valid_on_a_grown_plot(store):
    b = await _being(store, name="Ana")
    world.ensure_village(store, OWNER, now=NOW)
    store.set_plot_size(OWNER, 2400, now=NOW)
    hx, hy = world.home_xy(store.get(OWNER, b["slug"]))
    assert 0 <= hx <= 2400 and 0 <= hy <= 2400         # still on the plot
    # position resolves cleanly (no crash from the bigger grid)
    pos = world.position_of(store, store.get(OWNER, b["slug"]), NOW)
    assert pos["at"] == "home"


# ═══ The whitelist, the offers, the map ═══════════════════════════════════

def test_normalize_digest_object_shapes():
    d = life._normalize_digest({
        "craft": {"kind": "  BENCH ", "name": " Rest Here ",
                  "inscription": "for tired feet"},
        "place": {"object_id": "rest-here", "x": "312", "y": 480.0},
        "unplace": {"object_id": "old-mark"},
    })
    assert d["craft"] == {"kind": "bench", "name": "Rest Here",
                          "inscription": "for tired feet"}
    assert d["place"] == {"object_id": "rest-here", "x": 312, "y": 480}
    assert d["unplace"] == {"object_id": "old-mark"}
    junk = life._normalize_digest({
        "craft": "a bench", "place": {"x": 10}, "unplace": ["old-mark"]})
    assert junk["craft"] is None and junk["place"] is None \
        and junk["unplace"] is None
    coords = life._normalize_digest(
        {"place": {"object_id": "m", "x": "abc"}})
    assert coords["place"] == {"object_id": "m", "x": None, "y": None}


def test_offers_gate_honestly():
    being = {"stage": "child", "genome": {}, "public": False}
    none = life.society_prompt_fields(being, None, None)
    assert not any('"craft"' in f for f in none)
    offered = life.society_prompt_fields(being, None, None, can_craft=True,
                                         held_objects=["sun-cairn"])
    assert any('"craft": {"kind"' in f and "commons" in f for f in offered)
    assert any('"place": {"object_id": "sun-cairn"' in f for f in offered)


async def test_map_payload_carries_standing_objects(store):
    b = await _being(store)
    world.ensure_village(store, OWNER, now=NOW)
    held = _craft(store, b, name="In Hand", words="not yet")
    row = _craft(store, b, name="Sun Cairn", words="stones")
    ask = _legal_asks(store, b, 1)[0]
    world.place_object(store, store.get(OWNER, b["slug"]), row["id"],
                       x=ask[0], y=ask[1], now=NOW)
    payload = world.village_map_payload(store, OWNER, now=NOW)
    ids = [o["id"] for o in payload["objects"]]
    assert row["id"] in ids and held["id"] not in ids
    entry = next(o for o in payload["objects"] if o["id"] == row["id"])
    assert entry["kind"] == "cairn" and entry["by"] == b["slug"]
    assert entry["xy"] == [entry["xy"][0], entry["xy"][1]]
