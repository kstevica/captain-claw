# The Ghost in the Village — first-person view (FPV) plan

The parent enters the village in first person and roams it like a quiet
ghost: walking the same streets the Iskre walk, seeing them mid-journey,
leaving small signs in the grass they will find later. No direct
communication — they can only *sense* the presence, and read what was left
behind.

Engine: the attached single-file three.js voxel engine (VOXELHEIM) is the
skeleton. We keep its FPV camera + pointer lock, WASD physics with
sub-stepped AABB collision, chunk meshing, procedural canvas texture atlas,
and day/night sky. We strip everything survival: mining, placing, mobs,
hearts, hotbar, drops, death.

## Ground rules (locked)

- **World is NOT noise-generated.** It is built deterministically from the
  existing village layout: the same 50×50 tile grid, footprints, roads, and
  seeded props that feed the 2D isometric map. FPV and 2D map can never
  disagree.
- **Scale:** 1 block = 5 village units → 1 tile = 4×4 blocks → the plot is
  a 200×200-block world. Buildings (2×2..4×4 tiles) become 8–16 block
  houses. Flat elevation in v1 (per the village-world decision).
- **Iskre are paper cutouts** (locked): full SVG avatar rasterized to a
  canvas texture on a thin slab — front + back drawn, edges tinted `--c2`,
  name tag above. Slab faces its walking direction; turns toward the ghost
  when sensing. Infants at ~0.75 scale.
- **Ghost physics** (locked): solid walker by default (WASD + jump, walls
  and trees are real, doors work), `F` toggles **phase mode** — fly
  (Space/Shift) and pass through everything.
- **Sensing** (locked): pause + percept. Near the ghost an iskra stops and
  turns to look (client-side, free); a rate-limited "felt a presence"
  percept (per-being cooldown) lands at their next mind tick. No LLM call,
  no realtime cost, no event-log flooding.
- **Notes** (locked): signpost anywhere. Plant a sign at your feet; it
  exists in the world; each iskra that wanders near it later finds it once,
  and the text lands as a percept.
- **Two ghosts** (expanded at Phase 3, user decision): the PARENT (authed,
  wears the violet "parent" pill, may pull any sign) and PUBLIC VISITORS
  (un-gated /village, choose a name that becomes their amber pill and
  signs their notes; may plant, never pull; their presence wake touches
  PUBLIC beings only). Entered from either map's "Enter the village"
  button, rendered as a lazy-loaded fullscreen overlay so three.js never
  weighs down the main bundle.
- three.js as an npm dependency (no CDN — bundle stays self-contained).
- Day/night follows the real clock (sun angle from local hour), not the
  engine's 6-minute cycle.
- Positions come from the same zero-poll walk data as the 2D map
  (`/fd/beings/village/map`): client animates polylines from
  `departed_at` + minutes; refetch ~every 60s. WALK_SPEED stays untouched —
  Iskre move at their real living pace; the ghost is simply faster.

## Phase 1 — the world in 3D (walk it alone)

Frontend only. `three` added to flight-deck; new
`src/components/village/fpv/` with a lazy `VillageFPV` overlay mounted from
the map header button.

- **World build from the map payload:** grass ground plane of blocks;
  roads → packed-earth path blocks; `building` footprints → walls in a
  per-kind material (library = timber + tall roof, workshop = planks,
  well = stone ring, remember = old stone) with the door tile left open and
  a simple hip roof; `grounds` (square, garden, meadow, play) → low fences,
  flower/grass dressing; every being's home → a small 2×2-tile cottage with
  door; props → trunk+canopy trees, bushes, flower tufts, lamps that glow
  at night.
- **Textures:** the engine's procedural 16px atlas approach, recolored to
  the warm storybook palette of the 2D map (same hues as IsoScene day
  theme).
- **Physics:** walker (collision, jump) + `F` phase toggle (fly, no
  collision). Esc unlocks pointer → pause overlay with "Leave the village".
- **HUD:** crosshair, a location chip ("near the Library"), a clock chip,
  controls hint on entry. Nothing else.
- Verify live in the isolated FD (seeded village), both themes.

## Phase 2 — the Iskre walk it with you

- Rasterize `IskraAvatar` SVG → canvas texture per being (cached by
  character+palette); thin-slab paper figure + name tag sprite.
- Animate along the same `path` polylines the 2D map uses (client-side,
  zero poll; refetch every 60s to pick up new walks). Position converted
  village-units → blocks. Idle figures stand at their place; walking
  figures inch along at true pace with a gentle paper bob.
- **Sensing, visible half:** ghost within ~3 tiles → the figure pauses its
  walk animation and turns to face you for a few seconds, then resumes.
  Purely client-side this phase.
- Buildings' interiors reachable through doors (or phase mode), so you can
  stand in the Library beside a reading iskra.

## Phase 3 — the ghost touches the world (backend)

- **Presence percept:** `POST /fd/beings/village/presence` `{x, y}` —
  client sends while roaming (throttled, ~every 10s of movement). Server
  computes proximity to each living being via `position_of`; within radius
  and past the per-being cooldown (1h) it records a `presence` event whose
  text lands at the next mind tick via `percepts_since` ("The air went
  still around you for a moment — something kind passed close by.").
- **Notes:** new `village_notes` (owner, x, y, text, created_at, read_by).
  `POST/DELETE /fd/beings/village/notes`; notes included in the map
  payload. In FPV: press `E` to plant a sign at your feet (pointer unlocks,
  small text dialog); signs render as little wooden posts with a paper
  slip; look + click to read or pull out your own sign.
- **Discovery:** at each tick/feet pass, a being checks unread notes within
  ~2 tiles of its current position; a hit records a percept ("You found a
  small sign planted in the grass — your parent's handwriting: …") and
  marks it read for that being. Each being finds each note once.
- **2D map parity:** notes drawn on the isometric map (tiny sign icon) so
  the parent sees their signs from above too.
- Tests: presence cooldown + radius, note CRUD + per-being read-once
  discovery, fevered/egg/dead beings never sense.

## Phase 4 — read the buildings in first person (shipped)

- A **reading stand** (lectern: post + tilted paper board + soft glow)
  stands inside each building whose Iskre keep work in a folder — derived
  from `folderFor(place)`, so the FPV and the 2D map agree on which places
  are readable (Library → reports, Garden → garden pages, Workshop →
  skills, and `remember` places → self). Placed in worldgen alongside the
  labels; the block position feeds both the prop and the proximity check.
- Walk within ~2.4 blocks → HUD prompt "R — read the reading room in the
  Library"; **R** exits pointer lock and opens `BuildingReader`.
- `BuildingReader` is the SAME per-iskra file browser the 2D map's
  `MapPlaceCard` uses — `folderFor`/`shortName`/`isBoilerplate` + GFM
  markdown — with the file source injected: the PARENT reads authed
  self-files (`getSelfFiles`/`getSelfFile`), a PUBLIC visitor reads the
  un-gated public files (`getPublicFiles`/`getPublicFile`). No backend
  work — those endpoints already existed.
- Closing (X / Esc) re-locks and returns to walking. No new backend, no
  new tests (pure reuse of existing endpoints + verified live).

## Phase 5 — ghosts see each other (shipped)

The parent and public visitors roam ONE village together and see each
other — the parent sees the visitors, the visitors see the parent and each
other.

- **A live in-memory roster** per village owner (`being_world._ghost_roster`
  + `ghost_heartbeat` / `ghost_depart`): every roaming client heartbeats
  its spot every ~2s and gets back the OTHER ghosts here right now. Pure
  Python, $0, no DB, no percepts — this is the render layer of company.
  Entries expire on silence (`GHOST_TTL_S` = 8s), so a paused or departed
  ghost fades within the TTL; a leave-beacon vanishes it at once. One
  shared roster per owner means the parent and the visitors to THAT
  village share a room; other villages never bleed in.
- Endpoints: authed `POST /beings/village-map/ghost` (+ `/ghost/leave`),
  public `POST /fd/public/village/ghost` (+ `/ghost/leave`). The authed
  side keys by `user["id"]`, the public side by `public_village_owner()` —
  same owner on a single-village machine, so parent and visitors merge.
- Frontend `Ghosts` manager: a soft translucent spectral figure per other
  ghost (NOT a paper Iskra) — violet-tinted for the parent, amber for a
  visitor — each wearing its identity pill ("parent" / the visitor's
  name). Positions arrive on the 2s beat, so each figure LERPS toward its
  latest spot. The heartbeat runs only WHILE ROAMING (a stable per-session
  ghost id); pausing/reading/leaving stops it and you fade for the others.
- Tests: roster cross-visibility, TTL fade, cross-village isolation,
  explicit depart. Verified live both ways (parent saw two visitor ghosts;
  a visitor saw the violet parent + another visitor).

## Phase 6 — plays on a phone (shipped)

- **Keyboard no longer leaks into text fields.** The engine's document
  keydown used to `preventDefault` Space (and capture every letter) even
  while a note/name field was focused — so a space jumped instead of
  typing. `onKeyDown`/`onKeyUp` now bail when an editable element
  (`input`/`textarea`/`select`/contenteditable) has focus.
- **Touch controls** (`MobileControls`, shown when the primary pointer is
  coarse, or `?touch=1`): a left thumb-stick to walk, a drag-anywhere zone
  to look, and round action buttons — fly, jump, note, read. Pointer
  events (not touch events) so each finger is captured independently:
  walk and look at once. Read dims when no stand is in reach.
- **Landscape nudge:** a portrait phone gets a "turn your phone sideways"
  overlay; it clears on rotate.
- **Gyro look:** a top toggle switches the view control from drag to the
  phone's own orientation (device-orientation, baselined on enable so it
  doesn't snap; iOS permission-gated on the tap). Best-effort — the tilt
  mapping is device-dependent; drag-to-look is always there as the
  fallback.
- Engine gained a small touch API (`enterTouch`/`setMove`/`look`/`jump`/
  `toggleFly`/`note`/`read`/`setGyro`); the actions are shared by keys and
  buttons so a tap and a keypress are the same thing. On touch, entry
  skips pointer lock entirely.

## Deferred (explicitly not in v1)

- Elevation, water, weather.
- Ghost-to-ghost communication (they see each other; they don't talk).
- Analog walk speed from the joystick (it's full-speed directional now).
- The Iskre *reacting in the body brain* to presence in real time (feet
  detours toward/away from the ghost).
- Multi-visitor ghosts (other FD users with view-shares).
