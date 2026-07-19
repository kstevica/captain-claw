# Roads + a growing plot — the parent shapes the ground itself

Sibling to `docs/being-world-shaping-plan.md` (objects, the civic hand,
parent-build). Status: SHIPPED 2026-07-19, live-verified.

Two direct-hand tools for the parent, on the same iso map surface as
parent-build:

1. **Road-building** — paint streets tile by tile.
2. **Grow map** — enlarge the (square) plot; grow-only, keep everything.

## Locked decisions (user, 2026-07-19)

- Roads: **paint tile-by-tile** (each click toggles that tile; click a
  painted tile again to lift it) — full freedom over any shape.
- Map size: **grow-only** (≥ the standard 1000, up to 2400) — homes stay
  on the west edge and every existing position stays valid; shrinking
  (which would need rescaling the coordinate core) is out of scope.
- On resize: **keep everything** — places, homes, objects and roads keep
  their coordinates; the map just gains open room. (A separate "Redraw"
  respreads places if wanted.)

## Roads (as built)

- A second street layer `village_meta.roads_manual` (JSON tiles), UNIONed
  with the carved `roads` at read time by `being_world.effective_roads`
  — one set every consumer reads (`village_props` lamps + prop exclusion,
  `plot_course`/`_astar` ROAD_COST, `construction_taken`, `_object_taken`,
  the map payload). So a hand-drawn road **renders as a street, speeds
  walking on it, and keeps beings from building on it** exactly like a
  carved one — and it **survives every re-carve** (a redraw or resize
  never wipes it, because `refresh_layout` only touches `roads`).
- `store.toggle_manual_road(owner, tx, ty)` adds/lifts one tile.
  `POST /fd/beings/village-map/road {x, y}` (units → `tile_of`).
- Frontend: a **"roads"** toggle in the build block (amber when armed).
  In road mode a map click paints the clicked tile (via the same
  `onGround` unprojection parent-build uses); clicking a painted tile
  lifts it. Mutually exclusive with the object-kind palette.

## Grow map (as built)

- The plot's real size lives in `village_meta.plot_w/plot_h` (already
  there; now authoritative). `store.set_plot_size(owner, size)` — a
  SQUARE plot, clamped `[PLOT_MIN=1000, PLOT_MAX=2400]`, snapped to a
  whole tile grid — then re-carves streets/props for the new grid.
  `POST /fd/beings/village-map/size {size}`.
- `being_world` grew per-owner helpers `plot_dims`/`grid_dims` (read from
  meta) and a generous `GRID_MAX = PLOT_MAX//TILE` (120). The change
  splits cleanly:
  - **Pure clamps** (`tile_of`, `_tiles_at`, `home_tiles`, the two path
    bounds) clamp to `GRID_MAX` — safe for any plot; real coords are
    always in-bounds so the clamp only guards garbage, and this matches
    the frontend.
  - **Iteration/bounds** (`refresh_layout`, `village_props`,
    `_civic_zone`, `construction_taken`, `object_spot`, `commission_spot`,
    `standing_spots`) read the REAL per-owner grid/plot — streets, props,
    the civic ring, and buildable bounds all fill the true plot.
  - The architect prompt now names the real plot size, so a **redraw
    after growing** spreads places across the whole plot.
- Homes are UNCHANGED (grow-only keeps `home_xy`'s west band valid — the
  homes stay on the west edge, which is what they are). This is why
  shrinking is out of scope: a smaller plot would push homes off it.
- Frontend: `IsoScene` derives the ground diamond, the home viewBox, and
  the zoom ceiling from `data.grid.plot_w` (the iso projection itself is
  plot-agnostic); a resize re-frames the view. `GRID` became the generous
  120 to match the backend. A **"Map size"** row (standard / large / huge
  / vast) posts the size; smaller-than-current is disabled (grow-only).

## Tests

`test_being_objects.py` +5 — road paint/lift + effective union + build-
guard, a painted road survives a re-carve + rides the payload; grow scales
grid + room + payload, clamp/snap/grow-only, homes stay valid on a 2400
plot. Full FD suite green minus the 8 pre-existing mcp/vfs failures; the
`standing_spots` clamp was made plot-aware (a test caught it).

## FPV: a thing built inside appears in place (fix)

The FPV voxel world is built ONCE from a snapshot; a thing built (or a road
painted) from inside used to appear only after leaving and re-entering,
because static geometry lives in the mesh, not in the walker/sign updates
the 60s refetch pushes. Fixed: the engine's world became re-meshable —
`createFPV` tracks the solid/glow meshes mutably and exposes
`rebuild(world)` (dispose old meshes → swap `world` → re-mesh → re-add), and
`VillageFPV.build()` refetches the payload and calls `handle.rebuild(
buildWorld(m))` after a placement. Live-verified: a shrine built with the
FPV **B** / hammer button appeared at the ghost's feet without leaving.

## Deferred

- **FPV plot-scaling.** The first-person voxel world is a fixed
  200-block-per-side extent (`worldgen` W/D/H are module constants the
  engine imports). On a grown plot the FPV renders the standard extent and
  clips content beyond it (the block setter bounds-guards, so it never
  crashes) — the iso map is the full-resize surface. Making the voxel
  world dynamic (allocating + meshing a larger grid, threading W/D through
  the engine) is a real engine refactor, noted for later.
- Non-square plots (independent width/height).
- Shrinking the plot (needs `home_xy` to become plot-aware).
- Two-point road auto-routing (v1 is freeform tile paint).
