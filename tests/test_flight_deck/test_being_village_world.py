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


# ═══ Standing spots — two Iskre never occupy one point ════════════════════

def _park(store, slug, place):
    """Send a being to a place a day ago so, read now, its walk has ended and
    it reports PARKED at that place (position_of settles on read)."""
    store.depart(OWNER, slug, place, now=NOW - timedelta(days=1))


def test_two_beings_at_one_place_get_distinct_spots(store):
    a = _being(store, "Ada", stage="adult")
    b = _being(store, "Bela", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    _park(store, a["slug"], "square")
    _park(store, b["slug"], "square")
    m = world.village_map_payload(store, OWNER, now=NOW)
    ea = next(x for x in m["beings"] if x["slug"] == a["slug"])
    eb = next(x for x in m["beings"] if x["slug"] == b["slug"])
    assert ea["at"] == "square" and eb["at"] == "square" and not ea["to"]
    assert tuple(ea["xy"]) != tuple(eb["xy"])          # never the same pixel
    # and both stand near the place, not flung across the plot
    sq = next(p for p in m["places"] if p["id"] == "square")
    for e in (ea, eb):
        assert abs(e["xy"][0] - sq["x"]) <= 3 * world.TILE
        assert abs(e["xy"][1] - sq["y"]) <= 3 * world.TILE


def test_many_beings_at_one_place_all_distinct(store):
    slugs = []
    for nm in ("Ana", "Bura", "Cvita", "Dora", "Eva"):
        slugs.append(_being(store, nm, stage="adult")["slug"])
    world.ensure_village(store, OWNER, now=NOW)
    for s in slugs:
        _park(store, s, "meadow")
    m = world.village_map_payload(store, OWNER, now=NOW)
    pts = [tuple(e["xy"]) for e in m["beings"] if e["at"] == "meadow"]
    assert len(pts) == len(slugs) == len(set(pts))     # all seated apart


def test_a_spot_is_stable_while_the_room_is_unchanged(store):
    a = _being(store, "Ada", stage="adult")
    b = _being(store, "Bela", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    _park(store, a["slug"], "library")
    _park(store, b["slug"], "library")
    m1 = world.village_map_payload(store, OWNER, now=NOW)
    m2 = world.village_map_payload(store, OWNER, now=NOW + timedelta(minutes=5))
    for slug in (a["slug"], b["slug"]):
        p1 = next(e["xy"] for e in m1["beings"] if e["slug"] == slug)
        p2 = next(e["xy"] for e in m2["beings"] if e["slug"] == slug)
        assert p1 == p2                                # seat doesn't wander


def test_a_lone_occupant_keeps_the_anchor(store):
    a = _being(store, "Sama", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    _park(store, a["slug"], "square")
    m = world.village_map_payload(store, OWNER, now=NOW)
    e = next(x for x in m["beings"] if x["slug"] == a["slug"])
    sq = next(p for p in m["places"] if p["id"] == "square")
    assert tuple(e["xy"]) == (sq["x"], sq["y"])        # no needless offset


def test_beings_at_a_building_stand_outside_its_walls(store):
    """A building can't be entered, so a being parked at one is seated on
    walkable ground OUT FRONT — never rendered inside its footprint (the
    reported bug: iskre drawn inside the Well pavilion). Holds for a lone
    occupant and a crowd, on the host village's own map."""
    world.ensure_village(store, OWNER, now=NOW)
    solid = world._building_tiles(store, OWNER)
    assert solid                                        # the village has walls
    # the Library is a 3x2 building; the Well a 1x1 — both solid
    lone = _being(store, "Sama", stage="adult")
    _park(store, lone["slug"], "well")
    a = _being(store, "Ada", stage="adult")
    b = _being(store, "Bela", stage="adult")
    _park(store, a["slug"], "library")
    _park(store, b["slug"], "library")
    m = world.village_map_payload(store, OWNER, now=NOW)
    at_bldg = [e for e in m["beings"] if e["at"] in ("well", "library")]
    assert len(at_bldg) == 3
    for e in at_bldg:
        assert world.tile_of(e["xy"][0], e["xy"][1]) not in solid   # not in a wall
    # the two co-occupants still get distinct spots
    lib = [tuple(e["xy"]) for e in at_bldg if e["at"] == "library"]
    assert len(lib) == 2 and lib[0] != lib[1]


def test_walking_beings_are_never_seated(store):
    a = _being(store, "Ada", stage="adult")
    b = _being(store, "Bela", stage="adult")
    world.ensure_village(store, OWNER, now=NOW)
    # both still ON THE ROAD to the same place — seats must not touch them
    store.depart(OWNER, a["slug"], "library", now=NOW)
    store.depart(OWNER, b["slug"], "library", now=NOW)
    m = world.village_map_payload(store, OWNER, now=NOW)
    for slug in (a["slug"], b["slug"]):
        e = next(x for x in m["beings"] if x["slug"] == slug)
        assert e["to"] == "library"                    # walking, not parked
        # xy is the live path position, i.e. the walker's origin at t0
        assert e.get("path")


def test_spots_stay_inside_the_plot(store):
    offs = world._spot_offsets(4, 4)
    seats = world.standing_spots((world.PLOT_SIZE - 5, world.PLOT_SIZE - 5),
                                 (4, 4, "grounds"),
                                 [f"iskra-edge-{i}" for i in range(8)])
    assert len(offs) == world.SPOT_TOTAL
    for (x, y) in seats.values():
        assert world.TILE <= x <= world.PLOT_SIZE - world.TILE
        assert world.TILE <= y <= world.PLOT_SIZE - world.TILE


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


# ═══ Signs & the felt ghost (FPV plan Phase 3) ═════════════════════════════

def test_a_sign_is_planted_bounded_and_pulled(store):
    world.ensure_village(store, OWNER, now=NOW)
    n = store.add_village_note(OWNER, 500, 500, "  be kind today  ",
                               author="parent", author_kind="parent",
                               now=NOW)
    assert n["text"] == "be kind today"
    assert store.village_notes(OWNER)[0]["id"] == n["id"]
    with pytest.raises(BeingError):        # words required
        store.add_village_note(OWNER, 500, 500, "   ")
    with pytest.raises(BeingError):        # bounded text
        store.add_village_note(OWNER, 500, 500, "x" * 281)
    with pytest.raises(BeingError):        # on the plot
        store.add_village_note(OWNER, 1200, 500, "off the map")
    with pytest.raises(BeingError):        # a visitor signs with a name
        store.add_village_note(OWNER, 1, 1, "hi",
                               author="x" * 25, author_kind="visitor")
    assert store.remove_village_note(OWNER, n["id"])
    assert not store.remove_village_note(OWNER, n["id"])
    assert store.village_notes(OWNER) == []


def test_the_grass_holds_only_so_many_signs(store):
    world.ensure_village(store, OWNER, now=NOW)
    for i in range(store.MAX_VILLAGE_NOTES):
        store.add_village_note(OWNER, 10 + i, 10, f"sign {i}", now=NOW)
    with pytest.raises(BeingError):
        store.add_village_note(OWNER, 900, 900, "one too many", now=NOW)


def test_a_being_finds_a_near_sign_once(store):
    b = _being(store, "Nalaznik")
    world.ensure_village(store, OWNER, now=NOW)
    home = world.home_xy(b)
    near = store.add_village_note(OWNER, home[0] + 10, home[1], "hello you",
                                  author="parent", author_kind="parent",
                                  now=NOW)
    store.add_village_note(OWNER, (home[0] + 500) % 1000, home[1],
                           "far away", now=NOW)
    assert world.discover_notes(store, b, NOW) == 1
    ev = _events(store, b["slug"], "note_found")
    assert len(ev) == 1 and ev[0]["data"]["text"] == "hello you"
    assert b["slug"] in store.village_notes(OWNER)[0]["read_by"]
    # once means once — a second pass finds nothing new
    assert world.discover_notes(store, b, NOW + timedelta(minutes=5)) == 0
    # the reflex pass carries discovery too (no double-count after read)
    assert near["id"]  # (sanity: the near sign existed)


def test_a_visitor_sign_reads_signed(store):
    b = _being(store, "Citac")
    world.ensure_village(store, OWNER, now=NOW)
    home = world.home_xy(b)
    store.add_village_note(OWNER, home[0], home[1] + 8, "lijep pozdrav",
                           author="Mira", author_kind="visitor", now=NOW)
    world.discover_notes(store, b, NOW)
    senses = life.percepts_since(store, b)
    line = next(s for s in senses if "sign planted in the grass" in s)
    assert "a visitor, Mira" in line and "lijep pozdrav" in line


def test_presence_is_felt_near_once_per_cooldown(store):
    b = _being(store, "Osjetljiva")
    world.ensure_village(store, OWNER, now=NOW)
    home = world.home_xy(b)
    felt = world.presence_felt(store, OWNER, home[0] + 20, home[1],
                               author="parent", author_kind="parent",
                               now=NOW)
    assert felt == [b["name"]]
    # cooldown: passing again within the hour is weather, not an alarm
    felt2 = world.presence_felt(store, OWNER, home[0], home[1],
                                author="parent", author_kind="parent",
                                now=NOW + timedelta(minutes=10))
    assert felt2 == []
    # …and past the cooldown it lands again
    felt3 = world.presence_felt(store, OWNER, home[0], home[1],
                                author="parent", author_kind="parent",
                                now=NOW + timedelta(hours=2))
    assert felt3 == [b["name"]]
    # far away is nothing at all
    felt4 = world.presence_felt(store, OWNER, (home[0] + 500) % 1000,
                                home[1], author="parent",
                                author_kind="parent",
                                now=NOW + timedelta(hours=5))
    assert felt4 == []


def test_a_visitor_wake_only_touches_public_beings(store):
    pub = _being(store, "Javna")
    priv = _being(store, "Skrita")
    world.ensure_village(store, OWNER, now=NOW)
    hp, hv = world.home_xy(pub), world.home_xy(priv)
    # both are close enough — but the visitor's wake is scoped
    felt = world.presence_felt(store, OWNER, hp[0], hp[1],
                               author="Mira", author_kind="visitor",
                               now=NOW, only_slugs={pub["slug"]})
    assert felt == [pub["name"]]
    assert _events(store, priv["slug"], "presence") == []
    line = next(s for s in life.percepts_since(store, pub)
                if "visitor" in s and "Mira" in s)
    assert "roamed the village" in line
    assert hv  # (sanity)


def test_the_map_payload_carries_signs_and_redacts_readers(store):
    b = _being(store, "Znak")
    world.ensure_village(store, OWNER, now=NOW)
    n = store.add_village_note(OWNER, 500, 500, "vidimo se", now=NOW)
    store.mark_note_read(OWNER, n["id"], b["slug"])
    full = world.village_map_payload(store, OWNER, now=NOW)
    assert full["notes"][0]["found"] == 1
    assert full["notes"][0]["read_by"] == [b["slug"]]
    pub = world.village_map_payload(store, OWNER, now=NOW,
                                    only_slugs=set())
    assert pub["notes"][0]["found"] == 1
    assert pub["notes"][0]["read_by"] == []


# ═══ The living ghost roster (FPV plan Phase 5) ════════════════════════════

def test_ghosts_in_one_village_see_each_other(store):
    world._ghost_roster.clear()
    # the parent heartbeats first — alone, sees no one
    assert world.ghost_heartbeat(OWNER, "p1", kind="parent", name="parent",
                                 x=100, y=100) == []
    # a visitor arrives — sees the parent
    seen = world.ghost_heartbeat(OWNER, "v1", kind="visitor", name="Mira",
                                 x=200, y=200)
    assert len(seen) == 1
    assert seen[0]["kind"] == "parent" and seen[0]["xy"] == [100, 100]
    # the parent's next beat now sees the visitor, by name
    seen2 = world.ghost_heartbeat(OWNER, "p1", kind="parent", name="parent",
                                  x=100, y=100)
    assert [g["name"] for g in seen2] == ["Mira"]
    # a second visitor sees BOTH the parent and the first visitor
    seen3 = world.ghost_heartbeat(OWNER, "v2", kind="visitor", name="Vjetar",
                                  x=300, y=300)
    assert {g["name"] for g in seen3} == {"parent", "Mira"}


def test_a_ghost_fades_on_silence(store, monkeypatch):
    world._ghost_roster.clear()
    t = {"now": 1000.0}
    monkeypatch.setattr(world.time, "monotonic", lambda: t["now"])
    world.ghost_heartbeat(OWNER, "p1", kind="parent", name="parent",
                          x=10, y=10)
    # a visitor beats, sees the parent
    assert len(world.ghost_heartbeat(OWNER, "v1", kind="visitor",
                                     name="Mira", x=20, y=20)) == 1
    # time passes beyond the TTL; the parent went silent
    t["now"] += world.GHOST_TTL_S + 1
    # the visitor's next beat prunes the stale parent — no one else here
    assert world.ghost_heartbeat(OWNER, "v1", kind="visitor", name="Mira",
                                 x=25, y=20) == []


def test_ghosts_never_cross_villages(store):
    world._ghost_roster.clear()
    world.ghost_heartbeat("village-a", "p1", kind="parent", name="parent",
                          x=1, y=1)
    seen = world.ghost_heartbeat("village-b", "v1", kind="visitor",
                                 name="Mira", x=2, y=2)
    assert seen == []           # a different village is a different room


def test_a_ghost_can_depart_before_the_ttl(store):
    world._ghost_roster.clear()
    world.ghost_heartbeat(OWNER, "p1", kind="parent", name="parent",
                          x=5, y=5)
    world.ghost_depart(OWNER, "p1")
    assert world.ghost_heartbeat(OWNER, "v1", kind="visitor", name="Mira",
                                 x=6, y=6) == []
