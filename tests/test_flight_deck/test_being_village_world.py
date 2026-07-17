"""Village-world plan Phase 1 — the ground gets a body: a 50×50 tile grid
over the same plot, footprints with doors, deterministically carved
streets, pure seeded props, and in-place upgrades that never move an
anchor a being remembers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from captain_claw import vfs
from captain_claw.flight_deck import being_life as life
from captain_claw.flight_deck import being_society as society
from captain_claw.flight_deck import being_world as world
from captain_claw.flight_deck.beings import BeingError, BeingsStore

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
OWNER = "user-1"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    getattr(vfs.vfs_base, "cache_clear", lambda: None)()
    return BeingsStore(db_path=tmp_path / "beings.db")


def _being(store, name="Zvjezdana", stage=None):
    b = store.conceive(OWNER, name, preset="explorer",
                       allowance_preset="5M", now=NOW - timedelta(days=2))
    store.hatch(OWNER, b["slug"], now=NOW - timedelta(days=2))
    if stage:
        store.set_stage(OWNER, b["slug"], stage, now=NOW - timedelta(days=2))
    return store.get(OWNER, b["slug"])


def _events(store, slug, kind):
    return [e for e in store.events(OWNER, slug) if e["kind"] == kind]


def _buildings(places):
    return [p for p in places if p["kind"] == "building"]


def _road_set(store):
    return {(int(t[0]), int(t[1]))
            for t in store.get_village_meta(OWNER)["roads"]}


# ═══ Founding lays out the whole body ══════════════════════════════════════

def test_founding_lays_out_the_ground(store):
    world.ensure_village(store, OWNER, now=NOW)
    places = store.village_places(OWNER)
    assert places
    for p in places:
        assert p["w"] >= 1 and p["h"] >= 1
        assert p["kind"] in ("building", "grounds")
    for p in _buildings(places):
        door = (p["door_x"], p["door_y"])
        assert door[0] is not None
        tiles = set(world.footprint_tiles(p))
        assert door in tiles                       # the door is ON the body
    meta = store.get_village_meta(OWNER)
    assert meta["plot_w"] == 1000 and meta["plot_h"] == 1000
    assert meta["tile_size"] == world.TILE == 20
    assert meta["terrain"]["default_elevation"] == 0   # the 3D hook, flat v1
    assert meta["roads"]                               # streets exist


def test_upgrade_dresses_in_place_and_never_moves_an_anchor(store):
    world.ensure_village(store, OWNER, now=NOW)
    anchors = {p["id"]: (p["x"], p["y"])
               for p in store.village_places(OWNER)}
    # simulate a village from before the world model: strip the layout
    c = store._c()
    c.execute("UPDATE village_places SET w=0, h=0, kind='',"
              " door_x=NULL, door_y=NULL WHERE owner_id = ?", (OWNER,))
    c.execute("UPDATE village_meta SET roads='' WHERE owner_id = ?",
              (OWNER,))
    c.commit()
    world.ensure_village(store, OWNER, now=NOW)        # dresses it again
    after = store.village_places(OWNER)
    assert {p["id"]: (p["x"], p["y"]) for p in after} == anchors
    assert all(p["kind"] in ("building", "grounds") for p in after)
    assert _road_set(store)


def test_layout_is_deterministic(store, tmp_path):
    world.ensure_village(store, OWNER, now=NOW)
    other = BeingsStore(db_path=tmp_path / "beings2.db")
    world.ensure_village(other, OWNER, now=NOW)
    a = [(p["id"], p["x"], p["y"], p["w"], p["h"], p["kind"],
          p["door_x"], p["door_y"]) for p in store.village_places(OWNER)]
    b = [(p["id"], p["x"], p["y"], p["w"], p["h"], p["kind"],
          p["door_x"], p["door_y"]) for p in other.village_places(OWNER)]
    assert a == b
    assert store.get_village_meta(OWNER)["roads"] \
        == other.get_village_meta(OWNER)["roads"]
    assert world.village_props(store, OWNER) \
        == world.village_props(other, OWNER)


# ═══ Streets ═══════════════════════════════════════════════════════════════

def test_roads_reach_every_place_and_hold_together(store):
    world.ensure_village(store, OWNER, now=NOW)
    roads = _road_set(store)
    places = store.village_places(OWNER)
    for p in places:
        if p["kind"] == "building":
            assert (p["door_x"], p["door_y"]) in roads
        else:
            assert world.tile_of(p["x"], p["y"]) in roads
    # one connected web: flood from any road tile covers all of them
    seen = set()
    stack = [next(iter(roads))]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in roads and nxt not in seen:
                stack.append(nxt)
    assert seen == roads


def test_footprints_homes_and_streets_never_overlap(store):
    a = _being(store, "Ana")
    b = _being(store, "Bura")
    world.ensure_village(store, OWNER, now=NOW)
    places = store.village_places(OWNER)
    homes = set(world.home_tiles(a)) | set(world.home_tiles(b))
    lane = {(world.HOME_LANE_TX, ty) for ty in range(3, world.GRID_H - 3)}
    prints = {p["id"]: set(world.footprint_tiles(p)) for p in places}
    ids = sorted(prints)
    for i, one in enumerate(ids):
        for two in ids[i + 1:]:
            assert not (prints[one] & prints[two]), (one, two)
        assert not (prints[one] & homes)
        assert not (prints[one] & lane)
    roads = _road_set(store)
    walls = set()
    for p in _buildings(places):
        walls |= set(world.footprint_tiles(p))
        walls.discard((p["door_x"], p["door_y"]))
    assert not (roads & walls)              # streets go around, never through


# ═══ Props ═════════════════════════════════════════════════════════════════

def test_props_are_pure_and_respect_the_ground(store):
    world.ensure_village(store, OWNER, now=NOW)
    props = world.village_props(store, OWNER)
    assert props == world.village_props(store, OWNER)   # pure — twice equal
    roads = _road_set(store)
    prints = set()
    for p in store.village_places(OWNER):
        prints |= set(world.footprint_tiles(p))
    kinds = {p["kind"] for p in props}
    assert "tree" in kinds and "lamp" in kinds
    for pr in props:
        t = tuple(pr["tile"])
        if pr["kind"] == "lamp":
            assert t in roads                            # lamps line streets
        else:
            assert t not in roads and t not in prints
            assert t[0] > world.HOME_LANE_TX             # gardens stay east


def test_a_new_building_never_reshuffles_a_distant_tree(store):
    world.ensure_village(store, OWNER, now=NOW)
    before = {tuple(p["tile"]): p["kind"]
              for p in world.village_props(store, OWNER)
              if p["kind"] != "lamp"}
    spot = world.commission_spot(store, OWNER, "fund-1",
                                 affordance="create")
    place = store.add_place(OWNER, {
        "id": "the-kiln", "name": "the Kiln", "x": spot[0], "y": spot[1],
        "affordances": ["create"], "description": "clay and heat"},
        now=NOW)
    cleared = set(world.footprint_tiles(store.get_place(OWNER,
                                                        place["id"])))
    roads_now = _road_set(store)
    after = {tuple(p["tile"]): p["kind"]
             for p in world.village_props(store, OWNER)
             if p["kind"] != "lamp"}
    for t, kind in before.items():
        if t in cleared or t in roads_now:
            continue                       # its ground was built or paved
        assert after.get(t) == kind        # every distant tree still stands


# ═══ Commissions on the new ground ═════════════════════════════════════════

def test_commission_spot_fits_its_footprint(store):
    world.ensure_village(store, OWNER, now=NOW)
    spot = world.commission_spot(store, OWNER, "fund-2",
                                 affordance="create")
    assert spot == world.commission_spot(store, OWNER, "fund-2",
                                         affordance="create")
    tiles = set(world._tiles_at(spot[0], spot[1], 2, 2))
    assert not (tiles & world.construction_taken(store, OWNER))


def test_add_place_raises_a_connected_building(store):
    world.ensure_village(store, OWNER, now=NOW)
    spot = world.commission_spot(store, OWNER, "fund-3",
                                 affordance="read")
    place = store.add_place(OWNER, {
        "id": "the-annex", "name": "the Annex", "x": spot[0], "y": spot[1],
        "affordances": ["read"], "description": "more shelves"}, now=NOW)
    p = store.get_place(OWNER, place["id"])
    assert (p["w"], p["h"], p["kind"]) == (3, 2, "building")
    assert (p["door_x"], p["door_y"]) in _road_set(store)   # connected


# ═══ Meta + the map file ═══════════════════════════════════════════════════

def test_meta_upsert_keeps_the_world_columns(store):
    world.ensure_village(store, OWNER, now=NOW)
    roads = store.get_village_meta(OWNER)["roads"]
    assert roads
    store.set_steward_stipend(OWNER, 3, now=NOW)     # unrelated upsert write
    meta = store.get_village_meta(OWNER)
    assert meta["roads"] == roads                    # the gotcha regression
    assert meta["tile_size"] == 20
    assert meta["steward_stipend_coins"] == 3


def test_map_md_tells_the_streets(store):
    world.ensure_village(store, OWNER, now=NOW)
    path = society._commons_path(OWNER, "village/MAP.md")
    text = path.read_text(encoding="utf-8")
    assert "Streets run from every door" in text
    assert "— the Architect" in text


# ═══ The parent nudge + the observer map ═══════════════════════════════════

def test_the_map_payload_carries_the_whole_world(store):
    a = _being(store, "Ana", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    store.depart(OWNER, a["slug"], "library", now=NOW)
    m = world.village_map_payload(store, OWNER, now=NOW)
    assert m["grid"]["tile_size"] == 20 and m["roads"] and m["props"]
    assert {p["id"] for p in m["places"]} >= {"square", "library"}
    me = next(b for b in m["beings"] if b["slug"] == a["slug"])
    assert me["to"] == "library" and me["path"] and me["avatar"]
    assert me["departed_at"] and me["total_minutes"]


def test_only_slugs_restricts_who_is_drawn(store):
    a = _being(store, "Ana", stage="adult")
    _being(store, "Bura", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    m = world.village_map_payload(store, OWNER, now=NOW,
                                  only_slugs={a["slug"]})
    assert [b["slug"] for b in m["beings"]] == [a["slug"]]


def test_a_parent_nudge_walks_an_alive_being(store):
    a = _being(store, "Ada", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    store.depart(OWNER, a["slug"], "library", now=NOW, by="nudge")
    loc = store.get(OWNER, a["slug"])["location"]
    assert loc["to"] == "library" and loc["by"] == "nudge" and loc["path"]
    dep = _events(store, a["slug"], "departed")[0]
    assert dep["data"]["by"] == "nudge"
    # the being feels it honestly on arrival
    store.settle_location(store.get(OWNER, a["slug"]),
                          now=NOW + timedelta(hours=8))
    lines = life.percepts_since(store, store.get(OWNER, a["slug"]))
    assert any("Your parent walked you to" in ln for ln in lines)


def test_the_dead_and_paused_refuse_the_nudge(store):
    a = _being(store, "Mrtva", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    store.set_state(OWNER, a["slug"], "paused", now=NOW)
    with pytest.raises(BeingError):
        store.depart(OWNER, a["slug"], "library", now=NOW, by="nudge")


def test_a_fevered_being_is_only_nudged_home(store):
    """The route gate (mirrors the being's own go_to refusal): while
    fevered, a nudge anywhere but home is refused — the body would only
    turn back. Home is always allowed. Tests the exact decision the route
    makes (resolve + fever_state)."""
    a = _being(store, "Grozna", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    # a real breakdown in the last day → fever holds (not dice)
    store.record_event(a["id"], "collapsed_exhausted", {"weighted": 9},
                       now=NOW - timedelta(hours=2))
    being = store.get(OWNER, a["slug"])
    assert world.fever_state(store, being, NOW)          # she is fevered

    def gate(dest: str) -> bool:                          # the route's rule
        pid = store.resolve_place_ref(OWNER, dest)
        return bool(pid and pid != "home"
                    and world.fever_state(store, being, NOW))

    assert gate("library") is True                       # refused elsewhere
    assert gate("the Meadow") is True
    assert gate("home") is False                          # home always ok
    # and once it passes, every road opens again
    store.record_event(a["id"], "collapsed_exhausted", {"weighted": 9},
                       now=NOW - timedelta(days=3))       # aged out
    well = store.get(OWNER, a["slug"])
    assert world.fever_state(store, well,
                             NOW + timedelta(days=2)) is None


def test_public_owner_and_map_show_only_public_beings(store):
    pub = _being(store, "Javna", stage="adult")
    _being(store, "Tajna", stage="adult")           # stays private
    world.ensure_village(store, OWNER, now=NOW)
    store.set_public(OWNER, pub["slug"], True, now=NOW)
    assert store.public_village_owner() == OWNER
    slugs = {b["slug"] for b in store.public_beings()
             if b.get("owner_id") == OWNER}
    m = world.village_map_payload(store, OWNER, now=NOW, only_slugs=slugs)
    assert [b["slug"] for b in m["beings"]] == [pub["slug"]]
    assert m["places"]                               # the ground still shows


# ═══ Phase 2 — plotted courses ═════════════════════════════════════════════

def test_astar_routes_around_walls_and_prefers_streets():
    # a wall forces the way around it, never through
    wall = {(12, y) for y in range(5, 15)}
    path = world._astar(wall, set(), (10, 10), (14, 10))
    assert path and path[0] == (10, 10) and path[-1] == (14, 10)
    assert not (set(path) & wall)
    # a street detour that is LONGER in tiles wins on cost (0.6 vs 1.0)
    roads = {(x, 11) for x in range(9, 18)}
    path2 = world._astar(set(), roads, (10, 10), (16, 10))
    assert (12, 11) in path2                   # walked the street
    # determinism: the same ask plots the same course
    assert path2 == world._astar(set(), roads, (10, 10), (16, 10))


def test_position_is_pure_and_follows_the_course(store):
    b = _being(store, "Putnik", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    store.depart(OWNER, b["slug"], "library", now=NOW)
    b = store.get(OWNER, b["slug"])
    loc = b["location"]
    assert len(loc["path"]) > 2                # it turns — real streets
    t1 = NOW + timedelta(minutes=loc["minutes"] / 2)
    assert world.position_of(store, b, t1) \
        == world.position_of(store, b, t1)     # pure: same clock, same point
    mid = world.position_of(store, b, t1)["xy"]
    o, d = loc["path"][0], loc["path"][-1]
    assert tuple(mid) != ((o[0] + d[0]) // 2, (o[1] + d[1]) // 2)
    done = world.position_of(
        store, b, NOW + timedelta(minutes=loc["minutes"] + 1))
    assert done["at"] == "library"
    eta_min = (done["arrived_at"] - NOW).total_seconds() / 60.0
    assert abs(eta_min - loc["minutes"]) < 0.51
    arr = store.settle_location(store.get(OWNER, b["slug"]),
                                now=NOW + timedelta(hours=6))
    assert arr and arr["place"] == "library"


def test_rows_from_before_the_world_model_still_walk(store):
    import json as _json
    b = _being(store, "Stara")
    world.ensure_village(store, OWNER, now=NOW)
    old = {"to": "library", "from": "home",
           "origin": list(world.home_xy(b)),
           "departed_at": (NOW - timedelta(minutes=5)).isoformat()}
    c = store._c()
    c.execute("UPDATE beings SET location = ? WHERE id = ?",
              (_json.dumps(old), b["id"]))
    c.commit()
    b = store.get(OWNER, b["slug"])
    pos = world.position_of(store, b, NOW)
    assert pos["to"] == "library" and pos["minutes_left"] > 0
    arr = store.settle_location(b, now=NOW + timedelta(hours=8))
    assert arr and arr["place"] == "library"   # the straight-line fallback


def test_buildings_are_entered_by_the_door(store):
    b = _being(store, "Gost", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    lib = store.get_place(OWNER, "library")
    store.depart(OWNER, b["slug"], "library", now=NOW)
    path = store.get(OWNER, b["slug"])["location"]["path"]
    assert tuple(path[-1]) == world.tile_center(lib["door_x"],
                                                lib["door_y"])
    store.settle_location(store.get(OWNER, b["slug"]),
                          now=NOW + timedelta(hours=6))
    store.depart(OWNER, b["slug"], "meadow", now=NOW + timedelta(hours=6))
    path2 = store.get(OWNER, b["slug"])["location"]["path"]
    med = store.get_place(OWNER, "meadow")
    assert tuple(path2[-1]) == (med["x"], med["y"])   # grounds: the heart


def test_infants_toddle_the_same_course_slower(store):
    a = _being(store, "Odrasla", stage="adult")
    i = _being(store, "Bebica")                       # infant
    world.ensure_village(store, OWNER, now=NOW)
    pa, ma = world.plot_course(store, a, (500, 500), "library")
    pi, mi = world.plot_course(store, i, (500, 500), "library")
    assert pa == pi                                    # same streets
    assert mi == pytest.approx(ma / world.INFANT_SPEED_FACTOR, rel=1e-6)


def test_your_own_door_is_open_others_are_not(store):
    a = _being(store, "Ana")
    bb = _being(store, "Bura")
    world.ensure_village(store, OWNER, now=NOW)
    blocked_for_a = world.walk_blocked(store, OWNER, a)
    assert not (set(world.home_tiles(a)) & blocked_for_a)
    assert set(world.home_tiles(bb)) <= blocked_for_a


# ═══ Phase 3 — the look ════════════════════════════════════════════════════

def test_every_iskra_has_a_stable_default_look(store):
    b = _being(store, "Svjetla")
    av = world.default_avatar(b)
    assert 1 <= av["c"] <= world.AVATAR_CHARACTERS
    assert av["p"] in world.AVATAR_PALETTES
    assert av == world.default_avatar(b)               # stable forever
    assert store.vitals(OWNER, b["slug"])["avatar"] == av


def test_the_parent_picks_a_look(store):
    b = _being(store, "Svjetla")
    store.set_avatar(OWNER, b["slug"], 7, "sea", now=NOW)
    assert store.vitals(OWNER, b["slug"])["avatar"] == {"c": 7, "p": "sea"}
    ev = [e for e in store.events(OWNER, b["slug"])
          if e["kind"] == "avatar_set"]
    assert ev and ev[0]["data"] == {"c": 7, "p": "sea"}
    with pytest.raises(BeingError):
        store.set_avatar(OWNER, b["slug"], 11, "sea")
    with pytest.raises(BeingError):
        store.set_avatar(OWNER, b["slug"], 3, "plaid")


def test_a_commissioned_building_is_walkable(store):
    b = _being(store, "Graditelj", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    spot = world.commission_spot(store, OWNER, "fund-9",
                                 affordance="create")
    place = store.add_place(OWNER, {
        "id": "the-kiln", "name": "the Kiln", "x": spot[0], "y": spot[1],
        "affordances": ["create"], "description": "clay"}, now=NOW)
    store.depart(OWNER, b["slug"], place["id"], now=NOW)
    loc = store.get(OWNER, b["slug"])["location"]
    p = store.get_place(OWNER, place["id"])
    assert tuple(loc["path"][-1]) == world.tile_center(p["door_x"],
                                                       p["door_y"])
    arr = store.settle_location(store.get(OWNER, b["slug"]),
                                now=NOW + timedelta(hours=8))
    assert arr and arr["place"] == place["id"]
