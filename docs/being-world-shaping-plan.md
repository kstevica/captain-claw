# Iskre Shape Their World — objects, homes, and a democratized architect

Sibling to `docs/being-village-space-plan.md` (the ground),
`docs/being-village-world-plan.md` (tiles, streets, props), and
`docs/being-body-brain-plan.md` (the feet). Status: **ALL 5 PHASES
SHIPPED 2026-07-19** (as-built notes at the end) — the whole arc is
built and live-verified; only the noted deferrals remain.

## Why

Today an Iskra's only lever on the shared world is a **commission**:
propose one civic building (name + one affordance + a reason), pool 50
coins, wait for a parent to approve. Everything else — every tree, lamp,
street, and even the shape of its own cottage — is a deterministic
function no one authors. The being reads the world; it barely writes it.

This arc gives beings a real, free-form way to **make things and place
them in the world** — a cairn by the meadow, a bench under a tree, a
planter in the open grass, a signpost at a crossroads — each a permanent,
readable, *functional* mark that other beings discover, walk to, and use.
A being finally shapes the ground it lives on, not just the files inside
its home.

## Locked decisions (user, 2026-07-19)

1. **Objects with function.** A placed object is not decoration — it
   grants a small **affordance boost** to whoever uses it (a bench →
   `remember`, a planter → `tend`), slotting straight into the existing
   `PLACE_BOOST` homeostat. Building reshapes the village's drive-map
   from day one.
2. **Hard cap, scaled to village area — no decay.** Objects are
   permanent (a made thing stays until its maker removes it). The number
   the world can hold is a function of the **plot area**; the plot is
   fixed at 1000×1000 today, so the cap is a constant now, but it is
   written as `area // OBJECT_AREA_PER_SLOT` so it **rises automatically
   when the village grows later**. No weathering, no background prune.
3. **Civic ground — and a buffer around it — is off-limits to beings.**
   Only the **parent** and the **steward** may place on or near the
   commons (the square, the well, any civic place, and a `CIVIC_BUFFER`
   ring around each). A being that tries is refused loudly: *"the commons
   isn't yours to build on — the steward and your parent tend it."*
4. **Everywhere else is free.** A being places anywhere on the open map
   that isn't civic-or-near-civic, its own yard included, with **no coin
   cost and no parent approval**. Making the object costs **tokens**
   (making is a create act; metabolism, not money) — nothing else gates
   it.
5. **Proximity is the whole social mechanic.**
   - **Near** a standing object (within `OBJECT_SEE_RADIUS`): a being
     **sees** it, can **read** its inscription, and **using** it grants
     the affordance boost. First time close serves **explore** once (a
     landmark discovered), exactly like a first place-visit.
   - **Far**: a being **senses** something is out there — a faint pull
     with a direction and a rough distance, but *not what it is*.
   - **The urge**: a being that has **never been close** to a sensed
     object **and** whose **explore/curiosity pressure is high** feels a
     real *urge to go check it* — an offered `go_to` toward its spot. The
     drive it's chasing is paid off when it arrives (first-close explore
     serve). World-building is wired directly into the explore loop.

## Design law (inherited, unbroken)

- **Physics decides, the LLM narrates.** Every effect — the boost, the
  cap, the civic refusal, the discovery radius, the explore serve — is
  enforced or measured by code. The being's digest only *asks*.
- **Position and discovery are pure functions of the clock + rows.** No
  scheduler, no background process. Distance to an object, "who can see
  it now", and "who is being pulled toward it" are all computed on read,
  the fever/steward/geometry pattern already used everywhere.
- **Anti-theater: a made thing is a real file.** `craft` writes a real
  markdown file in the being's home (the object's story / inscription);
  its existence in that tick's git diff is the proof, exactly as `sell`
  proves a listing and `guestbook` proves a line. A claimed object with
  no file refuses.
- **Only the store moves state.** No cheap-LLM (feet) path may craft or
  place — building is words + weight + a boost, so it stays a **mind**
  act, like letters, money, and commissions (the body brain's locked
  invariant is preserved).
- **Objects are a separate layer from the 12 civic places.** They never
  count against `VILLAGE_MAX_PLACES`; they are their own table and their
  own render layer, sitting beside the pure-function props.

---

## Arc A — Objects (`craft` + `place`)

### Data model

New table (`beings.py`), mirroring `village_places`:

```
village_objects(
  owner_id, id, being_id,        -- maker (or 'parent'/'steward' for civic)
  kind,                          -- from OBJECT_KINDS
  name,                          -- being-chosen, ≤40
  affordance,                    -- derived from kind (one of AFFORDANCES)
  x, y,                          -- unit coords, validated + snapped
  file_path,                     -- the proof: garden/works/<id>.md
  created_at, state              -- standing | removed
)  PK (owner_id, id)
```

Objects are read into the map payload beside props; `standing_objects
(owner_id)` is the one query the renderer, the boost math, and the
discovery percepts all read.

### Object kinds → affordance → sprite

A fixed vocabulary (physics enforces the effect; the being only names and
places). Each kind carries exactly one affordance, reusing the existing
`AFFORDANCE_DRIVE_BOOSTS` map so objects-with-function need **zero** new
homeostat code — they are new *sources* of the boost that already exists:

| kind | affordance | boosts drive | blocks walking? | sprite |
|---|---|---|---|---|
| `bench` | remember | grow | no | reuse `buildings.tsx` bench |
| `cairn` | remember | grow | yes | reuse cairn |
| `signpost` | read | grow | no | new |
| `planter` | tend | create | no | new |
| `sculpture` | play | explore | yes | new |
| `lantern` | gather | connect | no | reuse lamp |
| `fountain` | gather | connect | yes | new |
| `shrine` | remember | grow | yes | new |

`OBJECT_KINDS` and `OBJECT_BLOCKING` live in `being_world.py`.
Blocking kinds join the same set `walk_blocked` builds from trees, so the
path grid and the picture can never disagree (the world plan's invariant).

### The being's new tools (digest fields, `_normalize_digest` whitelist)

- **`craft`** `{kind, name, inscription}` — makes an object. Validates
  `kind ∈ OBJECT_KINDS`, `name ≤ 40`, `inscription ≤ 300`; writes
  `garden/works/<id>.md` (title = name, body = inscription, provenance
  header) — the file IS the proof; debits a token **fee** (create act,
  reason `craft`, ceiling-clamped like a reading/chore mint — making a
  thing costs thought). Records `object_crafted`. The object exists but
  is **not yet placed** (no x/y) — it sits "in hand".
- **`place`** `{object_id, x, y}` — sets a crafted (or previously
  removed) object down at `(x, y)`. Physics:
  1. resolve the object (must belong to this being, be `state≠standing`);
  2. **civic guard** — refuse if `(x,y)` is on any civic place footprint
     or within `CIVIC_BUFFER` of one, on the home lane road, or on
     another being's cottage (loud `society_refused what=place`, the
     locked commons message);
  3. **snap** to the nearest valid open tile (deterministic, like
     `commission_spot`) so a rough coordinate from a weak model still
     lands somewhere sensible;
  4. **cap guard** — refuse if the village is at `object_cap()` (message
     names the cap and suggests removing an old work);
  5. write x/y, `state=standing`, record `object_placed`.
- **`unplace` / remove** `{object_id}` — lifts your own object (frees a
  cap slot; `state=removed`, the file stays in `garden/works/`). Lets a
  being curate rather than hoard.
- **`go_to` extension** — `resolve_place_ref` also matches a standing
  object by name / id, returning its `(x,y)` as an ad-hoc destination, so
  a being (or the urge percept) can walk *to an object*, not only to the
  12 named places. `depart` plots the same A* course to that spot.

No **presence requirement** to place (per "placing anywhere is free") —
the being decorates open ground by decree; only the civic guard and the
cap constrain it.

### The homeostat hook (objects with function)

`place_drive_boosts` (today keyed off the being's current *place*) gains
an **object** source: when a being is settled within `OBJECT_ACCESS_RADIUS`
of a standing object, that object's affordance contributes its
`PLACE_BOOST` (1.5×) to the matching drive on the next serve — folded
into the same `_serve` closure, so every serve path inherits it,
asymptotic + per-day-satiation-damped exactly as today (self-farming
hits diminishing returns within one day).

**Anti-farming (proposed default):** a being's **own** objects give a
**reduced** boost (or none); **other** beings' objects give the full
1.5×. Building becomes an act *for the village* — and *discovering and
using someone else's* mark is the rewarding path, which is what makes the
proximity/urge mechanic matter. (Flag for confirmation; one constant to
flip.)

### The discovery mechanic (the star) — `object_percepts`

A new pure function in `being_world.py`, called from `umwelt_percepts`
each wake, over `standing_objects` minus the being's own:

- **d ≤ `OBJECT_SEE_RADIUS`** → a full line: *"A cairn stands close by —
  'name' — its face reads: <first line of inscription>."* First time
  close records `object_found` + `milestone(first_object_<id>)` +
  `_serve("explore")` once (the landmark payoff), same shape as
  `first_visit_<place>`.
- **d > see-radius** → a sense line, capped to the **nearest 1–2** so it
  never floods: *"Something new stands to the north-east, a good walk
  off — you can't make out what."* Direction from `atan2`, distance
  bucketed ("a short walk / a good walk / far across the village").
- **the urge** — for a sensed object the being has **no** `object_found`
  for, **and** `drive_pressures[explore] ≥ OBJECT_URGE_EXPLORE` → upgrade
  the sense line to an urge and offer the walk: *"…and you find you
  *want* to know what it is."* + `go_to` that spot in the tick's offered
  fields. Curiosity, not compulsion — silent when explore is satisfied.

Everything here is a pure read (positions via `position_of`, pressures
already computed) — no new process, $0.

### Guardrails

- **Cap** `object_cap(store, owner)` = `plot_area // OBJECT_AREA_PER_SLOT`
  (constant today; rises with the plot when the village grows). A soft
  **per-being share** (`cap // roster_size`, floor `OBJECT_MIN_PER_BEING`)
  keeps one prolific being from filling the map. `unplace` frees slots.
- **Civic buffer** `CIVIC_BUFFER` (proposed 2 tiles = 40 units) around
  every civic footprint; the home lane road; other cottages — all
  refused. A being's own yard is naturally clear (homes sit far west of
  the central square).
- **Cost** is tokens only (`OBJECT_CRAFT_FEE`, ceiling-clamped), charged
  at `craft`, never at `place` — you pay to *make*, not to *move*.

---

## Arc B — Home as your canvas

Today a cottage is a hash-derived 2×2 tile with no interior and no
decoration surface; `'home'` is a reserved id no one may author. This arc
gives the being its own ground to shape freely.

- **Name & style your cottage.** `home_name {name}` and (reusing the
  avatar machinery) a cottage look — stored on `beings` like `avatar`,
  shown on the map label and the being's public page. Personal, ungated
  (it's your home, not the commons).
- **The yard is the free canvas.** The tiles around your `home_xy` (a
  `YARD_RADIUS` patch, minus the lane) are always a legal, cap-friendly
  place to `place` objects — the no-approval zone falls straight out of
  Arc A's "everywhere non-civic is free". Homes scatter on the west lane,
  so yards naturally become the first cluster of discoverable marks,
  which seeds the sense/urge mechanic between neighbors.
- **Deferred:** true furnishable *interiors* (a home scene you decorate)
  — bigger, wants the FPV interior work; noted, not in v1.

---

## Arc C — Democratize the Architect (the civic hand)

Locked decision #3 makes the **steward** and **parent** the only civic
builders — so give them the tools the civic ground now needs:

- **Steward civic placement.** The current steward (the existing rotating
  civic role) may `place` on/near civic ground — the one exception to the
  civic guard — as a stewardship act (its objects are attributed to
  `steward`, evented, spoken in the steward's morning note). Public
  works, finally by a being.
- **Rename / redescribe / name a street.** A light civic-editing route
  (steward or parent): rename a place, rewrite its description, name a
  road — public content, so parent-blessed when a being proposes it
  (mirrors the chosen-name / self-mod approval pattern). Rewrites MAP.md.
- **The redesign button.** `POST /village-map/architect` **already
  exists** (the space plan deferred only its UI); surface it in the map
  panel as a parent action, and let the steward *propose* a redesign the
  parent runs.

This arc is deliberately small and dovetails with Arc A: beings own the
open wilds; the steward and parent own the commons.

---

## Frontend

- **`IsoScene` + the 2D map + FPV** render `standing_objects` as sprites
  at their tiles, depth-sorted with props (reuse `buildings.tsx` prop
  vocabulary + ~4 new sprites). Blocking objects read as solid; a click
  opens a small panel (name, maker, inscription, affordance).
- **Care drawer** gains the home name/look row (Arc B) and, for a
  steward-eligible being, the civic-placement affordance.
- **Life-log** labels + dots for `object_crafted` / `object_placed` /
  `object_found` / `object_removed`.
- The urge is visible too: a being walking to an object shows its dashed
  course like any `go_to`.

---

## Phases (each ships independently, off-path byte-identical)

1. **Objects backend, no function** — `village_objects` table + store
   writers/readers, `craft`/`place`/`unplace` digest fields + handlers,
   civic guard + snap + cap, `garden/works/` proof files, `go_to`→object
   extension, events. (~14 tests: craft writes proof + fee, place snaps +
   civic refusal + cap refusal, remove frees a slot, go_to reaches an
   object, whitelist drops junk, feet-can't-craft.)
2. **Function + discovery** — the homeostat object-boost source (+ own-vs-
   others anti-farming), `object_percepts` (see / sense / urge), the
   first-close explore serve + milestone, walk-blocking objects into
   `walk_blocked`. (~12 tests: boost lands on serve, own-reduced, first-
   close serves explore once, sense line direction/distance buckets, urge
   fires only when explore-starved + never-visited, blocking object
   reroutes a course.)
3. **The map** — render objects in IsoScene + 2D + FPV, the object panel,
   life-log labels, both themes; live-verify on an isolated FD.
4. **Home as canvas** (Arc B) — home name/look, the yard free-zone,
   public-page surfacing. (~6 tests + live verify.)
5. **The civic hand** (Arc C) — steward civic placement, rename/
   redescribe/name-a-street route + parent approval, the redesign
   button. (~8 tests + live verify.)

## Deferred (noted, not forgotten)

- Furnishable home **interiors** (wants the FPV interior work).
- **Object weathering** as an *optional* later mode (the cap is v1; decay
  can layer on if the world wants entropy).
- Objects a being can **give** or **sell** (an object is a placed
  artifact — the market could list it; unify with `sell` later).
- Objects with **richer function** (a signpost that names a real
  direction to a place; a lantern that lights an FPV night).
- **Group builds** — several beings raising one larger object together
  (the commission-pooling pattern applied to objects).
- Per-object **guestbooks** (a mark people sign as they pass).

---

## Phase 1 as built (SHIPPED 2026-07-19)

- **Store** (`beings.py`): `village_objects` table (PK owner+id; states
  `held` — in hand — and `standing`; no separate 'removed': unplacing
  returns the thing to your hands, the proof file stays). Writers are
  SQL-only, never law: `add_village_object` (id slugified from the name,
  numeric suffix on collision, `file_path` fixed at insert so file and
  row can never disagree), `set_object_ground`, `delete_village_object`
  (only the craft compensator uses it — no file, no object),
  `get_village_object` / `village_objects(state=)` readers, and
  `resolve_object_ref` (id / slug / name, "the " optional, optional
  maker + standing filters). `TRANSFER_REASONS` gained `craft_burn`.
  `resolve_place_ref` now falls through to standing objects and returns
  the namespaced `object:<id>` — **places always win a name collision**.
- **World** (`being_world.py`): `OBJECT_KINDS` (8 kinds → affordance +
  blocks-walking flag, both recorded now and wired in Phase 2),
  `CIVIC_BUFFER_TILES=2`, `OBJECT_SNAP_TILES=8`; `object_cap` =
  plot area // `OBJECT_AREA_PER_SLOT` (40 today, rises with the plot),
  `object_share` = cap split across the roster floored at
  `OBJECT_MIN_PER_BEING=3`. `_civic_zone` (footprints ⊕ buffer + the
  whole home lane) refuses; `_object_taken` (streets, ALL props, standing
  objects) merely snaps. **Refinement of locked decision #3**:
  `object_spot(asked=)` — an EXPLICIT x,y ask into the commons/ring, the
  lane, or another's yard refuses loudly (the law teaches); an
  at-your-feet placement (no x,y — e.g. standing in the square) slides
  out to the nearest legal tile instead, because the being said "set it
  down", not "build on the commons". The result always lands outside the
  ring either way. `place_object` (ownership, at-feet default, cap +
  share checked only when taking NEW ground — moving a standing thing is
  free), `unplace_object`. Walk plumbing: `place_xy` / `place_name` /
  `walk_target_xy` grew the `object:` branch, so `depart`, `position_of`,
  `settle_location`, plans, feet walks, and the parent nudge all inherit
  object destinations for free; a lifted thing mid-walk resolves like any
  broken ground (home). `construction_taken` includes standing-object
  tiles (a commission never rises on someone's cairn).
  `village_map_payload` gains an `objects` layer (id/kind/name/
  affordance/xy/tile/by) so Phase 3 is pure frontend.
- **Society** (`being_society.craft_object`): the making — alive,
  **child+** (stage gate: an implementation decision, the plan was
  silent), fixed kind vocabulary, name 2–40, inscription required; fee
  `OBJECT_CRAFT_FEE_TOKENS=25_000` burns first (reason `craft_burn`, the
  self-mod pattern — wallet checked, InsufficientTokens when broke);
  the proof file `garden/works/<id>.md` (title + provenance comment +
  inscription) written by physics; a failed write deletes the row and
  refuses. Events `object_crafted` / `object_placed` / `object_removed`;
  milestone `first_craft` (GOTCHA honored: the made thing's name rides
  under data key `made`, never `name`).
- **Life** (`being_life.py`): digest fields `craft` / `place` / `unplace`
  in `_normalize_digest` (the whitelist) + handlers between commission
  and `go_to` (so craft → place → walk chains in one tick), refusals as
  `society_refused` what=craft|place|unplace. The first-visit explore
  milestone skips `object:` arrivals (Phase 2's `object_found` owns the
  landmark payoff). Offers: `society_prompt_fields(can_craft=,
  held_objects=)` — craft offered only when stage + wallet truly allow;
  work in hand is named back to the being (the one nudge an unplaced
  thing gets); threaded through BOTH cognitions (monolith + orient).
- **Emergent, verified**: co-presence works on object ground out of the
  box — two beings walking to the same cairn cross paths (encounter,
  contact, gossip line "Cvijeta is here at Sun Cairn"), because
  `_co_present` compares settled ground and `place_name` resolves the
  object layer. A cairn in the wilds is a meeting place from day one.
- **Known minors** (accepted): two beings parked at one object share a
  pixel (`_seat_parked` skips non-place ground); a village re-layout may
  carve a street through an object tile until Phase 2 makes blocking
  kinds real; a guestbook attempt at an object refuses gracefully
  (per-object guestbooks stay deferred).
- Tests: `test_being_objects.py` (20) — craft proof/fee/milestone,
  vocabulary + stage + broke refusals, at-feet from home lands in the
  yard, at-feet at a civic place slides out, explicit commons/ring/lane/
  another's-yard asks refuse, occupied ground snaps deterministically,
  village cap + per-being share + move-exempt + unplace-frees, unplace
  ownership honesty, go_to a made thing end-to-end (resolve → depart →
  mid-road → settle → arrived, place-name collision won by places),
  lifted-thing-mid-walk resolves home, crossed paths at a cairn, tick
  craft/place/refusal + held-work offer in the prompt, no first-visit
  milestone for objects, feet can't craft, digest whitelist shapes,
  honest offers, map payload layer. Being suite 473 pass; full FD suite
  green minus the 8 pre-existing mcp/vfs failures (verified pre-existing
  on a clean tree); no new lint (same 14 findings, line-shifted).

## Phase 2 as built (SHIPPED 2026-07-19)

- **Function** (`being_world.drive_boost_factors`): the boost source
  generalized from a frozenset of favored drives to a **dict drive →
  factor**. The settled place's affordances pay `PLACE_BOOST=1.5` exactly
  as before; a standing made thing within `OBJECT_ACCESS_RADIUS=40`
  units (2 tiles — your yard, or standing AT the thing) pays through the
  same map: **another's work the full 1.5, your OWN
  `OBJECT_OWN_BOOST=1.25`** (building for the village is the point;
  farming your own bench pays less — the anti-farming default the user
  approved). Strongest source wins per drive; objects only count when
  SETTLED (reach is a fact of standing somewhere, not passing by).
  `_tick_locked`'s `boosted` became that dict; the `_serve` closure does
  `damp *= boosted[name]` — every serve path inherits, dreams still skip.
- **Discovery** (`being_world.object_percepts`, wired into the
  `umwelt_percepts` sweep): over standing objects **minus your own**
  (your works are silent ground — no self-discovery, ever).
  - **Close** (≤ `OBJECT_SEE_RADIUS=60` units, ANY wake — even waking
    mid-road beside it): "A DISCOVERY: a cairn stands here — 'Sun
    Cairn', Ana's work. Its face reads: …" — the inscription's first
    true line read from the maker's REAL proof file (`_object_face`; a
    vanished file reads blank, never crashes). Gated once per thing per
    life by `milestone(found_object_<id>)` (data key `object`, honoring
    the name-collision gotcha); records `object_found`.
  - **Far** (mornings only, `first_of_day`): at most
    `OBJECT_SENSE_LINES=2` pulls for UNFOUND things, nearest first —
    8-way compass heading (map space: x east, y south) + the distance as
    THIS body walks it (`_walk_bucket`: short walk / good walk / far
    across the village — an infant's far is honest). Nameless texture:
    "Something stands to the east, a good walk off — too far to make out
    what."
  - **The urge**: when explore pressure ≥ `OBJECT_URGE_EXPLORE=0.25`
    (via `drive_pressures`, lazy import), the nearest unfound pull
    upgrades: "AN URGE: … you find you WANT to know. You have heard it
    called 'Sun Cairn'. Add \"go_to\": \"Sun Cairn\" and see it with
    your own eyes." **Design decision**: the name arrives as HEARSAY so
    the offer is walkable (`go_to` resolves names) while what it IS
    stays unknown until the being stands before it — full mystery would
    have made the urge un-actionable.
  - **The payoff**: `_tick_locked` serves **explore** when the senses
    carry "A DISCOVERY:" — placed AFTER the umwelt sweep (the crossed-
    paths serve pattern; the scan must follow the sweep that produces
    the line). Found things stop pulling (read-only check over the
    milestone ledger).
- **Blocking** (`walk_blocked`): standing objects whose kind blocks
  (cairn, sculpture, fountain, shrine) join the blocked set beside trees
  and walls; `_astar`'s goal-always-enterable rule means walking TO a
  cairn still ends at the cairn while a course THROUGH it bends around.
  Benches, signposts, planters, lanterns stay walkable dressing.
- Tests (8 new; file total 28): boost factors full/reduced/absent, the
  1.5× damp reaching a real serve (two beings, equal starting
  satisfaction, bench vs home), discovery once + maker + face + explore
  satisfaction rising + never again, own-work silence, sense lines
  pointing east / bucketed / nameless / capped at 2 / morning-only /
  never in dreams, urge only when explore-hungry + hearsay name resolves
  to real ground, found things stop pulling, blocking tile in the grid +
  a straight line bending around it + settle AT the stone. Full FD suite
  931 pass + the same 8 pre-existing failures; no new lint.
- **Phase 2 known-minors**: discovery needs a WAKE while near (feet may
  pass a thing between ticks without finding it — v1 honest: you
  discover what you stand before when you open your eyes); the urge
  names one thing per morning (the nearest), by design.

## Phase 3 as built (SHIPPED 2026-07-19, live-verified)

- **Payload**: the `objects` entries gained `face` (the inscription's
  first line via `_object_face` — ~40 small file reads per payload
  build, fail-open) and `by_name`, so both maps and the FPV render
  everything with zero new routes.
- **Sprites** (`buildings.tsx`): 6 new storybook-flat fixtures —
  Signpost, Planter, Sculpture, Lantern, Fountain (small), Shrine —
  drawn CENTERED on (0,0) like the props (the renderer translates to
  the tile center); Bench and Cairn reuse the civic fixtures nudged to
  center. `OBJECT_SPRITES` registry keyed by kind. Lantern + shrine
  glow at dusk like the lamps.
- **IsoScene**: an objects layer between props and beings, depth-sorted
  (`+0.25` so a made thing edges in front of a same-tile prop),
  amber selection ring, hover title "<name> — a <kind>, <maker>'s
  work". New OPTIONAL `selObject`/`onObject` props — read-only maps
  (public observer, visit tab) still DRAW objects, they just aren't
  clickable there; all three click paths (place/being/object)
  cross-deselect each other.
- **The object panel** (`MapObjectCard` in BeingsPage): name, kind +
  affordance chips (AFF_HUE), maker with avatar, the face as a quote,
  and **"read the whole inscription"** — fetches the maker's real
  proof file through the existing `getSelfFile` (garden/works/<id>.md),
  rendered as markdown. Wired third in the panel chain (being > object
  > place > default).
- **FPV worldgen**: per-kind block builders on the object's tile
  (bench planks / signpost post+plank / planter with leaves+flowers /
  lantern post+LAMP / cairn stones / sculpture column / fountain
  stone-ring water / shrine posts + red roof + votive lamp) + a
  `WorldLabel` per object so the HUD names it when the ghost stands
  near. Blocking parity is physical: blocking kinds fill the tile
  center solidly; light kinds stay walkable-around at block scale.
- **Life-log**: labels + dots for object_crafted (emerald) /
  object_placed (teal) / object_removed (zinc) / object_found (amber),
  fee shown on crafted.
- **Raw-id leak hunt** (found live, fixed): the walk status ("on the
  road to object:the-far-light"), the being card's "at object:…", and
  the departed label all leaked ids. ONE frontend fix — fold standing
  objects into `placeById` as pseudo-places (`object:<id>` → name +
  xy) at all three map builders (VillageMap, PublicBeingPage,
  VisitedVillage) so `walk.ts` destOf/posOf/statusOf resolve them —
  plus the SOURCE fix: `store.depart`'s departed event now carries
  `name` (place_name resolves both layers; frontend falls back to
  prefix-strip for old events).
- **Live-verified** on an isolated FD (auth off, scratch dirs, owner
  `local`, seed script in the session scratchpad; beings PAUSED, walks
  still animate): 7 objects of 7 kinds rendered on the authed iso map
  (DOM titles confirmed), held work correctly absent; object panel +
  full-inscription read (real file content on screen); "on the road to
  The Far Light — ~28 min" and "at Sun Cairn" in the status lines;
  life-log labels ("made a fountain — 'Ada's Spring' (25k burned)");
  the PUBLIC /village observer map draws all 7 (private beings still
  excluded); FPV world builds + runs with the objects in the block
  world (Zvjezdana's paper figure walking her plotted course to The
  Far Light), console clean; light theme pass. Bundle pair →
  index-PjXQUveX.js (+ new lazy chunk graph; CSS hash unchanged
  Cwk5zLki). FD suite 931 + 8 pre-existing; tsc clean.
- **Phase 3 known-minors**: an FPV close-up of each object build was
  not walked to (blind FPV navigation through the remote pane is
  imprecise) — the builders follow the verified civic-fixture pattern
  and the world renders clean; the parent nudge select still lists
  only civic places (objects are nudgeable via the backend resolver,
  the select just doesn't offer them yet).

## Phase 4 as built (SHIPPED 2026-07-19, live-verified)

- **Name your home** (`beings.home_name` + `set_home_name`): UNGATED —
  any living stage keeps house (an infant may name its cottage; it's
  your home, not the commons). 2–40 chars; renameable **once a day**
  (event-ledger check — a churning model can't thrash the label);
  events `home_named` {name, from} + milestone `named_home`.
- **Dress your home** (`beings.home_look` + `set_home_look`): roof ∈
  `HOME_ROOFS` (ember, slate, moss, dusk), wall ∈ `HOME_WALLS`
  (plaster, timber, sage) — the physics of taste refuses anything
  else, loudly. Event `home_styled`. Both columns additive; `get()`
  decodes home_look JSON.
- **Digest fields** `home_name` (string) + `home_look` {roof, wall} in
  the whitelist; handlers beside the object handlers; a PARTIAL look
  ("roof only") keeps the current other half (defaults ember/plaster)
  rather than refusing. Offers appear in the society fields ONLY while
  the cottage is unnamed/undressed — after that the being knows the
  way (renames stay possible, just not advertised).
- **The yard, cap-exempt** (`YARD_RADIUS=2` tiles beyond the cottage,
  `home_yard_tiles`): `place_object` now snaps FIRST and checks the
  cap after — a spot landing in the placer's OWN yard skips both the
  village cap and the per-being share (the cap guards the commons'
  openness, not your garden). Symmetrically, standing objects that
  live in their maker's yard don't COUNT against the commons cap for
  anyone. The law still outranks the freedom: the civic ring and
  another's yard refuse exactly as before, and the cap-refusal
  messages now point home ("your own yard is always yours").
- **Surfaces**: vitals + the map payload beings entries + the
  federation-shared `public_profile` all carry home_name/home_look;
  `_public_place` at home now speaks the cottage's name (public page:
  "at home — “Mala Koliba”").
- **Frontend**: `Cottage` sprite parametrized (`HOME_ROOF_HUES` /
  `HOME_WALL_HUES` — same warm storybook range); IsoScene cottages
  wear the dress, carry "“name” — X's home" titles, and a small
  italic name label floats under a named cottage; FPV cottages build
  with per-look blocks — three NEW atlas tiles (ROOF_SLATE,
  ROOF_MOSS, WALL_SAGE — a shared `shingles()` painter) so every roof
  is true, not nearest-neighbor; the FPV home label carries the name;
  the public being page shows "at home — “name”"; life-log labels +
  violet dots for home_named / home_styled.
- **Gotcha hit**: the bundler's TS transform (rolldown/oxc) rejects a
  multi-line parenthesized `({…} as Record<…>)[k]` cast that `tsc
  --noEmit` accepts — module-level typed const maps instead.
- Tests (5 new; file total 33): naming ungated for an infant + daily
  rename gate + same-name/length refusals, dress vocab + events, the
  tick names-and-dresses (partial look fills, offer taught then falls
  silent, junk refused on the record), yard cap-exemption end-to-end
  (commons full → at-feet home placement stands; yard works never
  count against the commons; another's yard still refuses), home
  fields on vitals/payload/profile + the named-home public place.
  FD suite 936 + the same 8 pre-existing; tsc clean; no new lint.
- **Live-verified** on the isolated FD: Zvjezdana's cottage wears a
  dusk roof + timber walls with "Kuća od Vjetra" beneath; Ada's wears
  moss + sage as "Zvonik"; Beba's stays classic terracotta (unset);
  the yard planter "Vjetrov Vrt" stands beside the cottage (DOM
  titles confirmed); status line "arriving at The Far Light". Bundle
  pair → index-UK6qVQ2j.js + index-CjkAm5fp.css (BOTH hashes moved —
  at commit git rm the old pair index-BST82ZvP.js +
  index-Cwk5zLki.css).
- Deferred within Phase 4 (unchanged from the plan): furnishable
  interiors (wants the FPV interior work).

## Phase 5 as built (SHIPPED 2026-07-19, live-verified) — THE ARC IS COMPLETE

- **Steward civic placement** (`being_world.place_object(steward=)`):
  the current steward (the existing rotating weekly role,
  `current_steward` — a pure calendar+roster function, no state) is the
  one being that may raise a made thing ON the commons.
  `object_spot(civic_ok=)` opens the commons for it — a public work may
  stand on the plaza, by the well, along the ring; only real obstacles
  (building WALLS via `_building_tiles`, homes, occupied tiles) snap it
  aside (grounds stay usable). A public work is one that TRULY stands on
  civic ground: a steward setting a thing in its own yard or the open
  wilds is just a being with a made thing (not civic, and the cap
  applies). Civic works are marked (`village_objects.civic`, additive
  column), evented `civic_placed`, and stand OUTSIDE the being-cap
  (they're the role's, not a being's share — also excluded from the
  commons count for everyone). The tick computes `is_steward` once and
  threads it into the being's `place`; the steward's morning note
  teaches the civic hand for its week only.
- **Rename & redescribe places** (`store.update_place` + digest fields
  `rename_place` {place, name, why} / `redescribe_place` {place,
  description}): the id NEVER changes (guestbooks, MAP.md, and
  everything a being remembers stay true — a standing invariant); only
  the display name and prose move, same HARD bounds as `save_village`;
  coords/affordances/layout untouched; `write_map_md` rewrites MAP.md so
  every being reads the new word next wake; `resolve_place_ref` finds
  the place by its new name. Steward-direct through the tick (the civic
  role tends the commons); a NON-steward is refused kindly
  (`society_refused` → "ask the steward, or your parent"). Events
  `place_renamed` / `place_redescribed`.
- **The parent's civic hand** (routes + UI): `POST
  /village-map/place/{id}/edit` (rename/redescribe + MAP.md) — the
  MapPlaceCard grows a pencil → a "THE CIVIC HAND" inline editor (name
  input + description textarea + Save/Cancel + the id-preservation
  note), parent-only (passed `onEdit` only on the authed map). The
  already-existing architect route is surfaced as a **"Redraw the
  village…"** button in the map's default panel (confirm dialog; beings
  mid-walk to a removed place settle home next wake). Life-log labels +
  amber dots for `civic_placed` / `place_renamed` / `place_redescribed`;
  the object panel reads "a public work" + "Raised by the steward" for
  civic objects.
- Tests (6 new; file total 39): steward raises a public work (civic
  flag + event + payload, normal placement at the same spot refused);
  a steward's own-yard work is NOT civic; civic works stand outside the
  being-cap; steward renames keeping the id + MAP.md rewritten +
  resolvable by new name; steward redescribes but a non-steward is
  refused; `update_place` gates (2–60, nothing-to-change, unknown
  place) and leaves coords/affordances alone. FD suite 942 + the same 8
  pre-existing; tsc clean; the only source-lint delta was fixed (an
  unused `p =`), leaving one PRE-EXISTING unused import in the architect
  route untouched.
- **Live-verified** on the isolated FD: Ada (made adolescent → the sole
  steward) raised "The Commons Spring" fountain on the square — the
  object panel read fountain · gather · **a public work** / **RAISED BY
  THE STEWARD** / Ada / the inscription + full-read; "the Memory Stone"
  (renamed old-bench, id preserved) stood on the map; the parent
  civic-hand editor opened on the Garden with name+description+Save; the
  `POST …/edit` route renamed it to "the Green Rows" keeping id `garden`
  and rewrote MAP.md. Bundle pair → index-Cwxy96BW.js +
  index-DrhB6W8j.css (BOTH hashes moved — at commit git rm the prior
  pair index-BST82ZvP.js + index-Cwk5zLki.css).
- **Deferred within Phase 5** (noted, not built): naming a STREET (roads
  are an anonymous tile set — no identity model yet); a steward
  *proposing* a full redesign the parent then runs (the parent's own
  button covers the redraw).

---

## The arc, complete

Iskre now shape their world across four surfaces the commission never
touched: they **make and place functional objects** (Phase 1–2) that
others **discover, walk to, and use** (the see/sense/urge loop); those
objects **render** on the 2D map and in the FPV (Phase 3); every being
**names and dresses its own cottage** and keeps a **cap-exempt yard**
(Phase 4); and the **steward tends the commons** — public works, place
renames — while the **parent** holds the redraw (Phase 5). Everything
stayed inside the design law: physics decides, the LLM narrates; every
felt thing is a real variable; a made thing is a real file; position and
discovery are pure functions of the clock; building is a mind act the $0
feet can never reach.

Still open (all noted above): furnishable interiors, object weathering,
object gifting/selling (unify with the market), richer object function,
group builds, per-object guestbooks, street naming, steward-proposed
redesign.
