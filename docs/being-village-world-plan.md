# The Village World — tiles, streets, footprints, and a game-worthy look

Sibling to `docs/being-village-space-plan.md` (the ground) and
`docs/being-body-brain-plan.md` (the feet). Status: ALL 4 PHASES SHIPPED
2026-07-16 — the ground's body, plotted courses, the art, and the
isometric renderer; live-verified end to end in both themes.

## Intent

The village grows a real body: a tile grid with width, height, and an
elevation model (flat in v1, 3D-ready), buildings with multi-tile
footprints and doors, streets that connect them, and the props that make
a village a village — trees, lamps, hedges. Iskre stop beelining: every
walk plots a course over the grid (streets preferred, shortcuts allowed)
and the map animates them along it. On top, a warm storybook-flat
isometric look: ~10 building sprites keyed by affordance, and 10 Iskra
characters × 4 palettes (40 avatar looks) the parent picks from.

Everything stays deterministic and $0 — no LLM anywhere in this arc; the
one-shot architect keeps naming places, the layout engine does the rest.

## Locked decisions (2026-07-16)

1. **Isometric 2.5D** view — buildings drawn with facades and roofs on a
   2:1 diamond grid; closest feel to the future 3D version.
2. **Warm storybook flat** art direction — soft rounded shapes, warm
   muted palette, gentle outlines; fits the Iskre fiction.
3. **Streets preferred, shortcuts allowed** — A* over all walkable tiles
   with road tiles cheaper (≈0.6 vs 1.0), buildings and trees blocking;
   beings follow streets when sensible, cut across the meadow when
   clearly shorter.
4. **Sprite per affordance** — one building SVG per affordance kind +
   home cottage + a generic; commissioned places automatically look
   right the day they rise.

## Standing invariants (must survive every phase)

- **Position is a pure function of the location row + the clock.** The
  course is computed ONCE at `depart()` and stored as a polyline; no
  scheduler, settle-on-read unchanged, the map extrapolates client-side
  at zero poll cost.
- Existing villages upgrade in place: place ids, names, and x/y anchors
  are preserved (guestbooks, MAP.md references, and everything a being
  remembers stay true). Footprints, doors, roads, and props are dressed
  around the existing coordinates deterministically.
- Old-format location rows (straight-line walks in flight at deploy)
  settle correctly via a fallback interpolation.
- Mind walks (`go_to`), feet walks (instinct `go`), and fever homing all
  travel the same plotted courses at the same speeds (infant ×0.35).

## World model

### Grid

- The plot stays **1000×1000 units**; `TILE = 20` units → **50×50
  tiles**. Unit space remains authoritative (WALK_SPEED = 10 units/min,
  all stored x/y); the grid is the derived overlay for footprints,
  pathing, and rendering. Tile (tx, ty) = (x // 20, y // 20).
- `village_meta` gains `plot_w`, `plot_h`, `tile_size` (defaults
  1000/1000/20) and `terrain` (JSON, `{"default_elevation": 0,
  "elevation": {}}` — the 3D hook, empty in v1). GOTCHA:
  `_upsert_village_meta` hardcodes its column list — extend both INSERT
  and UPDATE.

### Footprints and doors

- `village_places` gains `w`, `h` (in tiles) and `kind`:
  `building` (blocks walking except its **door tile**) or `grounds`
  (walkable area — square plaza, meadow, garden). Anchor x/y stays the
  CENTER of the footprint.
- Default sizes by affordance (verify actual affordance keys at impl):
  square 4×4 grounds, library 3×2, workshop 2×2, garden 3×3 grounds,
  meadow 4×3 grounds, well 1×1, bench 1×1, commissioned places by their
  affordance, generic 2×2.
- The **door** is the footprint-edge tile facing the nearest road
  (deterministic, stored). Walks to a building end at its door; grounds
  are entered anywhere.
- **Homes**: computed 2×2 cottages at the existing crc32 west-lane
  `home_xy`, with doors facing the lane. Not stored — same pure function
  everywhere.

### Streets

- Deterministic carving, recomputed on `save_village` / `add_place` and
  stored in `village_meta.roads` (JSON tile list): a lane along the
  homes' west side, then an L-shaped (Manhattan) road from each door to
  the nearest already-carved road tile, starting from the square.
  Dedup → the road tile set. Commission builds re-carve so the new
  place is connected the day it stands.

### Props

- A **pure seeded function** (crc32 of owner + village seed), NOT
  stored: trees and bushes scattered on tiles that are not road, not
  footprint, not home (density ≈8%; trees block walking, bushes and
  flowers don't), lamps every ~6 road tiles. The same function feeds
  the path cost grid and the renderer, so collision and picture can
  never disagree. Zero migration, stable forever.

## Pathfinding

- Cost grid: building/tree tiles ∞ (doors open), road 0.6, everything
  else 1.0. 4-directional A*, collinear runs collapsed into a waypoint
  polyline (unit coords at tile centers).
- `depart()`: settle first (as today) → resolve dest → dest door (or
  nearest grounds tile) → A* from the current tile → store in the
  location row: `{"to", "from", "origin", "departed_at", "by",
  "path": [[x,y], …], "minutes": total}`. Waypoints ~20–30 after
  collapse; JSON stays small.
- `position_of`: walk the polyline by elapsed × the being's speed —
  still a pure clock read. `travel_minutes` = path length / speed, so
  every "~N min walk" percept and UI line stays honest (paths are a
  little longer than the old beeline — walks feel realer).
- Fallbacks: no path found (disconnected grid — shouldn't happen) or an
  old-format row → straight-line interpolation (the existing math),
  logged, never a crash.
- `commission_spot` upgrades from max-min-distance scatter to a search
  over FREE tile rectangles (footprint-aware), still seeded and
  deterministic.

## The art (SVG, storybook flat, isometric 2:1)

- **Projection**: tile → 2:1 diamond (64×32 px base); painter's order
  by (tx + ty). Ground diamonds in 2–3 grass tones, road diamonds in
  warm dirt, subtle edge blades.
- **Buildings** (~10 sprites, one per affordance + home + generic):
  TSX components in `flight-deck/src/components/village/sprites/`,
  drawn to sit on their w×h diamond footprint — facade + roof, warm
  palette, soft outline, a small glow at dusk (dark theme).
- **Iskre**: 10 characters × 4 palettes from 10 SVGs — each character
  uses CSS custom properties for its palette slots (`--c1` cloth,
  `--c2` accent, `--c3` hair/spark, `--c4` glow), and 4 named palettes
  (ember, meadow, sea, dusk) apply via a wrapper class → 40 looks, 10
  files. Distinct silhouettes (hood, braids, scarf, hat, round, tall,
  curly, spark-antenna, cloak, apron), 3/4 iso stance, ~48×64 viewBox,
  readable at map size.
- **Avatar selection**: `beings.avatar` JSON `{"c": 1–10, "p":
  "ember"}`; default derived deterministically from the genome (every
  Iskra has a stable look before the parent ever picks). POST
  `/beings/{slug}/avatar`; Care-drawer picker (10 characters × 4
  swatches); used on the map, the card, and the public page.

## Phases

### Phase 1 — the ground gets a body (backend, $0) — SHIPPED 2026-07-16

As built:

- **Columns**: `village_meta` + plot_w/plot_h/tile_size/terrain/roads
  (get/upsert both extended — the hardcoded-columns gotcha covered by a
  regression test); `village_places` + w/h/kind/door_x/door_y (0/'' =
  not laid out yet). New store writers: `set_place_layout`,
  `set_village_roads`.
- **being_world**: TILE=20, GRID 50×50, HOME_LANE_TX=7;
  `_ID_FOOTPRINTS` for the 7 default places (square 4×4 grounds …
  well 1×1 building) + `_AFF_FOOTPRINTS` fallback by first affordance
  for architect drafts and commissions; `tile_of` / `tile_center` /
  `_tiles_at` / `footprint_tiles` / `home_tiles` (2×2 cottages, pure);
  `_door_for` (edge tile facing the square); `_grid_path` (deterministic
  BFS, N/E/S/W fixed order — Phase 2 walking reuses it with costs).
- **`refresh_layout`**: anchors never move; already-assigned places keep
  their footprints; a colliding newcomer shrinks toward 1×1; doors face
  the square; streets carve as home lane → square → each place, BFS
  around buildings and homes; persisted via the store. Triggered from
  `save_village`, `add_place`, and once per pre-world village through
  `ensure_village`'s cheap check.
- **`village_props`**: pure per-tile crc32 (owner+tile) — trees (block),
  bushes/flowers (dressing), lamps every 6th road tile; never stored, so
  a new building clears its own ground without reshuffling one distant
  tree, and pathing + rendering read the same function.
- **`commission_spot(…, affordance=)`**: footprint-aware — candidates
  colliding with anything standing, homes, the lane, or streets are
  rejected; deterministic; falls back to the old scatter rather than
  refusing to build. `add_place` re-carves so the new door is connected
  the day it stands.
- MAP.md now tells beings the streets exist and that their legs follow
  them. ASCII-render eyeballed: lane + cottages west, streets radiating
  from the square (crossing its plaza — grounds are walkable), every
  door on a street, roads never through walls, props clear of it all.
- Tests: 11 new (`test_being_village_world.py`) — founding layout,
  in-place upgrade preserves anchors, cross-store determinism, road
  connectivity (single component), no overlaps, prop purity + stability
  under construction, commission fit, connected `add_place`, meta-upsert
  regression, MAP.md. FD suite 805 + 8 known.

### Phase 2 — plotted courses (backend, $0) — SHIPPED 2026-07-16

As built:

- **`_astar`**: weighted A* over the tile grid — stepping onto a street
  costs ROAD_COST=0.6, open ground 1.0, blocked never; the GOAL tile is
  always enterable (a being boxed in by new construction can still come
  home); deterministic (fixed neighbor order + insertion tie-break),
  admissible heuristic (manhattan × 0.6).
- **`walk_blocked(store, owner, being)`**: building walls minus their
  doors, trees (the same pure prop function the renderer reads), and
  OTHER beings' cottages — your own home lets you in.
- **`plot_course`**: plotted ONCE at `depart()` — collapse collinear
  tiles to waypoints, exact unit-space endpoints, minutes at THIS
  being's pace (infants toddle the same course at ×0.35); falls back to
  the straight line rather than refusing to walk. Stored in the location
  row as `path` + `minutes` alongside origin/departed_at/by.
- **`walk_target_xy`**: buildings are walked to their DOOR; grounds to
  their heart; home to your own doorstep. (At rest the being reports the
  place anchor as before — stepping through the door is the settle.)
- **`position_of`**: walks the stored polyline by elapsed time
  (`_along`) — still a pure clock read; rows from before the world model
  (no path) fall back to the old straight-line math and settle fine.
- All walks — mind `go_to`, feet decisions, fever homing — flow through
  the same `depart`, so every one follows the streets; the departed
  event's "~N min" is now path-honest.
- **village-map payload** gains `grid` {plot_w, plot_h, tile_size},
  `terrain`, `roads`, `props`, per-place footprints/doors (already in
  the rows), and each walker's `path`/`departed_at`/`total_minutes` so
  Phase 4 animates along real streets at zero poll. (Until Phase 4 the
  existing map still linear-extrapolates between refreshes — a small
  visual drift, corrected on every 60s snapshot.)
- Tests: 7 new (18 total in the file) — synthetic A* wall/street/
  determinism, purity + course-following + exact ETA + settle, pre-world
  rows walk and settle (fallback), door-vs-grounds endpoints, infant
  pace on identical course, own-home-open/others-blocked, commissioned
  building reachable end-to-end. Two older tests updated to the new
  physics (beeline-midpoint → along-the-stored-path; a longer settle
  window). FD suite 812 + 8 known.
- ASCII-verified live: home lane north → main street east → through the
  Square's plaza → the Library door; 21/25 position samples on street
  tiles; 94 min vs the 71-min beeline.

### Phase 3 — the art — SHIPPED 2026-07-16

As built:

- **`avatars.tsx`**: 10 storybook-flat characters (the Hood, the Braids,
  the Scarf, the Hat, the Little Round, the Tall, the Curly, the Spark,
  the Cape, the Apron), each drawn once (viewBox 0 0 48 64) and dressed
  by 4 palettes (ember/meadow/sea/dusk) through CSS custom properties —
  slots `--c1` cloth, `--c2` trim, `--c3` hair & spark, `--c4` the face
  glow (they are sparks; the face IS the light). 40 looks, 10 drawings.
  `IskraAvatar {c, p, size}` wrapper.
- **`buildings.tsx`** (for Phase 4's renderer, compile-checked now):
  isometric 2:1 sprites in local iso coords (footprint N corner at 0,0;
  east tile = +32,+16) — cottage, library, workshop, well, bench, stall,
  pavilion, cairn + grounds decals plaza/garden/meadow/pond + props
  tree/conifer/bush/flowers/lamp. `spriteForPlace`: bespoke look for the
  7 default ids, affordance fallback for anything the village raises
  later. Village fixtures wear fixed warm hues; only Iskre are
  palette-dressed.
- **Backend**: `beings.avatar` column (JSON {"c","p"}), `set_avatar`
  (validated, `avatar_set` event), `default_avatar` slug-hash stable
  default (AVATAR_CHARACTERS=10, AVATAR_PALETTES in being_world),
  `_avatar_view` never empty; vitals + village-map beings payload carry
  it; POST `/beings/{slug}/avatar`.
- **UI**: card header shows the avatar beside the name; Care drawer
  "look" row — current character · palette label opening a picker (all
  10 characters rendered in the current palette + 4 two-dot palette
  swatches).
- Live-verified on an isolated FD: defaults flowed (slug-hash gave the
  seeded pair distinct looks), picker rendered all 10 × 4, two clicks
  round-tripped UI → POST → store ({"c":5,"p":"ember"}) with
  `avatar_set` events, both themes. 2 backend tests (default stability,
  pick + validation); suite 814 + 8 known. Bundle js → BkWE4ZWA.

### Phase 4 — the village, playable — SHIPPED 2026-07-16

As built:

- **`IsoScene.tsx`**: the isometric 2:1 scene fed entirely by the
  village-map payload. Projection `iso(x,y) = ((x−y)·1.6, (x+y)·0.8)`;
  replicates the backend footprint/home clamps exactly (`footNW`,
  `homeNW`) so picture and physics agree tile-for-tile. Layers: ground
  diamond with a two-tone `<pattern>` checker → street tiles → then
  buildings, home cottages (Cottage sprite at 0.72), props (trees vary
  Tree/Conifer by tile hash), and Iskre, all depth-sorted by base sy;
  place labels float above with paint-order stroke halos.
- **Walking**: `posOf` walks the stored course client-side —
  `departed_at` is absolute, so one snapshot animates the whole walk
  along real streets (the 1 Hz beat only re-renders); `minutes_left`
  comes from `total_minutes` − elapsed. Pre-world rows keep the old
  dead-reckoning fallback. Selecting a walker draws its remaining
  course as a dashed violet polyline.
- **Interactions**: click a building/grounds → the existing place panel
  (description, here-now, stalls, guestbook); click an Iskra → the being
  panel with its road countdown; the commission card and stipend knob
  untouched. Wheel zoom, drag pan, double-click reset.
- **Evening**: dark theme swaps ground/street tones cooler and the
  lamps come on (warm halos); building windows glow in both.
- Live-verified on an isolated FD, both themes: streets radiate from
  the plaza fountain, Ada (the Braids · sea) walked her plotted course
  mid-street with a live countdown (57→55 min across the pass), Beba
  (the Little Round · ember) toddled the north road, Zvjezdana (the
  Cape · dusk) stood at the Library and appeared in its "here now"
  panel; selection, path highlight, and panels all confirmed. Bundle
  pair → CN1xb14b.js / CMjWNanm.css.

Deferred within Phase 4: the PUBLIC page still shows text-only place
lines — a public iso map needs a public village-map endpoint (public
beings only) first; noted below.

## Phase 5 — the parent nudge + the public observer map — SHIPPED 2026-07-16

As built (follow-up to the four phases):

- **Shared payload builder** `being_world.village_map_payload(store,
  owner, *, now, only_slugs=None)` — the authed `/village-map` route and
  the public one both call it; `only_slugs` restricts which beings are
  drawn.
- **Parent nudge**: `POST /fd/beings/{slug}/go {dest}` → `depart(...,
  by="nudge")` — plots the same A* course as any walk; only the living
  walk (paused/dead refuse loudly, surfaced to the UI); unknown place →
  400. The being feels it honestly: arrival percept "Your parent walked
  you to X." UI: a "send … to" select in the map's being panel (every
  place + home), disabled with a note when the being isn't alive; the
  map reloads and the orb starts down the street. `nudgeBeing` service.
- **Public observer map**: `GET /fd/public/village/map` (un-gated) —
  resolves the fronting village via `store.public_village_owner()`
  (extracted from `public_village`), draws its ground + streets + props
  and ONLY its public beings (`public_beings` filtered to that owner).
  A private being never appears. `PublicVillageMap` on `/village`
  reuses the exact `IsoScene` + shared `walk.ts` math, read-only (no
  nudge); a being's observer panel links to `/b/<slug>`.
- **Refactor**: the walk-extrapolation math (`posOf`/`destOf`/
  `alongPath`/`minutesLeft`/`statusOf`) moved to
  `components/village/walk.ts` so the parent map and the public map
  extrapolate identically from one source.
- Tests: 5 new (30 total in the world file) — payload carries the world,
  `only_slugs` filter, nudge walks an alive being + honest percept,
  dead/paused refuse, public owner + public-only map. Live-verified on
  an isolated FD (loop disabled): authed map showed all three, public
  map excluded the private Ada; nudged Ada from the UI (garden, 4-way
  course, `by=nudge`, fresh departed event); `/village` rendered the
  evening observer map, no nudge, being panel linked out. FD suite 819
  + 8 known. Bundle pair → CyvSlPZN.js / DSo3Dyqh.css.

## Deferred

- Real elevation (hills, bridges) — the model carries it from day one.
- Day/night cycle tied to real village time; window lights when a being
  is home.
- Interiors, seasons' visual dress (snow on roofs), a stream + bridge.
- The avatar rite — a being proposing its own look (parent blesses),
  epigenetics like the chosen name.
- The 3D renderer (three.js) consuming the same tiles/footprints/
  elevation model.
- Road wear — much-walked paths visibly widen (walk counts exist on the
  ledger already).
