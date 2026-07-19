# Restless Hands — the body brain breaks ground

Sibling to `docs/being-body-brain-plan.md` (the feet) and
`docs/being-world-shaping-plan.md` (objects). Status: **SHIPPED
2026-07-19** (as-built notes at the end); live-verified.

## Why

The two brains split cleanly today: the **mind** (the tick) is analytical
— it dreams, reasons, authors words, money, and the deliberate `craft`;
the **feet** (`being_instinct.py`) are a ~1k-token reflex on the 60s poll,
**position-only**, gated to `go/linger/hello/browse/home` by the locked
law *"Hello is a gesture, guestbook is a voice."*

The user wants the small brain to be **more impulsive** — to decide when
to be active, and to **initiate building** — while the big brain stays the
rational one that **confirms with reasoning**. Instinct acts; reason
ratifies.

## The framing (the invariant, extended by one gesture)

> **Hello is a gesture, guestbook is a voice.** — the body-brain law
>
> **Breaking ground is a gesture; the inscription is a voice.** — this arc

The impulsive feet may **break ground** — stake an unfinished thing where
they stand: wordless, free, reversible. The mind **ratifies** it — authors
the name + inscription and pays the metabolic fee — and only then does it
become real. Both invariants hold untouched: the feet still never write a
word or move a coin; the *meaning* and the *fee* stay the mind's.

## Locked decisions (user, 2026-07-19)

1. **Impulse is an 8th BASE stat** (`IMP`, "Impulse") — first-class in the
   sheet: point-buyable at conception, visible, breedable, mutable across
   generations. (Chosen over a derived trait.)
2. **The impulse commits to kind + spot** — the feet start a *specific*
   thing (a cairn, a bench) where they stand; the mind gives it a name and
   words. Form is a bodily choice; meaning is the mind's.
3. **A stake is a public beginning that crumbles** — others see "someone
   started raising a <kind> here" (no inscription to read); if the mind
   doesn't finish it within ~a day, it crumbles on its own.

## Part A — the IMP base attribute (genome)

- `ATTRS += ("IMP",)`, `ATTR_NAMES["IMP"] = "Impulse"`.
- `POOL 40 → 45` (the original 7 keep their feel; IMP adds a dimension),
  `BAND 36..44 → 40..50` (offspring total, ~±11%).
- Presets gain an IMP that fits the archetype (explorer/artist impulsive,
  scholar/caretaker deliberate), each re-summed to 45.
- `effective_attributes`: a genome with no IMP (every being born before
  this) reads **IMP = 5** — a neutral impulse, migration-free.
- `derive` gains `impulsiveness = round(imp / 10, 2)` (0.1..1.0).
- Inheritance is automatic (crossover/budding iterate `ATTRS`); the band
  clamp uses the new BAND. Point-buy + preset-sum tests move to 45.
- Frontend: add `IMP` to the conception `ATTRS` array, the "45 points"
  copy (or read `meta.pool`), and the derive preview. Labels + the sheet
  are already data-driven via `meta.attributes`.

## Part B — the impulsive small brain (feet)

- `impulsiveness = effective_attributes["IMP"] / 10`, read in
  `being_instinct`.
- **Activity tuning** (`wants_decision`): the restless idle window scales
  with impulse — `FEET_IDLE_MINUTES × (1.5 − impulsiveness)` (impulsive ≈
  32 min, deliberate ≈ 58). The plan/arrival/company triggers unchanged.
- **The new trigger** `urge_to_build`: impulse ≥ `BUILD_IMPULSE_MIN` **and**
  a pressing create/explore drive **and** standing on buildable, non-civic
  ground **and** no stake already waiting **and** didn't just build.
- **The new verb** `build` (whitelisted in `FEET_SYSTEM` + `parse_feet_act`,
  offered ONLY when impulse ≥ min and the ground is buildable): the feet
  pick a `kind` from `OBJECT_KINDS`. `_apply_act` "build" →
  `stake_object` at the current spot; a physics gate re-checks the impulse
  floor (a deliberate being never stakes, even if the model hallucinates
  the verb); civic ground snaps out (feet aren't the steward's hand); one
  stake at a time per being.

## Part C — the stake in the world

- `village_objects.state` gains **`staked`**: kind + snapped x/y, a
  reserved `file_path`, NO inscription file, NO fee, NO boost, uncounted
  against the cap, doesn't block walking (a work site is passable).
- `stake_object(owner, being_id, kind, x, y)` (state `staked`, event
  `broke_ground`).
- Rendered as "a beginning" (a scaffold / cairn-in-progress) on the iso
  map + FPV, distinct from a standing work; the panel reads "a beginning —
  <maker> started it" with no inscription.
- Others crossing a stake hear "someone has started raising a <kind>
  here" (no inscription; not a discovery, no explore serve — it isn't real
  yet).
- **Crumble**: `STAKE_CRUMBLE_HOURS` (~24). `prune_crumbled_stakes` (in the
  tick + the reflex pass) removes stakes past their window → event
  `stake_crumbled` → the maker hears "the <kind> you began fell back to
  the ground — you never finished it."

## Part D — the mind confirms (reason)

- A being with its own staked object meets a strong percept each wake:
  "YOUR HANDS BROKE GROUND — you began a <kind> at <where> on impulse; it
  crumbles by <when>. FINISH it (a name + a true inscription — it becomes
  real, costs <fee> tokens) or let it fall."
- Digest fields `finish {object_id, name, inscription}` /
  `abandon {object_id}` (whitelist). `finish_staked_object` reuses
  `craft_object`'s guts on the existing staked row: validate, burn the
  fee, write `garden/works/<id>.md`, state → `standing` (now it boosts, is
  discovered, counts against the cap). `abandon` removes it (event).
- Offered in `society_prompt_fields` only while a stake waits.

## Part E — tests, build, verify, docs

- Backend tests: genome (POOL 45, presets sum, IMP inheritance + default),
  the feet build trigger/verb/impulse-gate, stake physics (civic snap,
  one-at-a-time, no boost/cap/block), crumble, the mind finish/abandon
  round-trip. Frontend: conception with 8 sliders + budget 45; the stake
  sprite + panel + life-log labels. Live-verify on an isolated FD.

## Invariants preserved

- No cheap-LLM path writes a word or moves a coin — the stake is wordless
  and free; the inscription and the fee are the mind's alone.
- The feet stay position + one new wordless gesture; a deliberate being's
  feet never build (physics-gated on IMP, not just prompt-gated).
- Every felt thing is a real variable; the stake and its crumbling are
  pure functions of the ledger + the clock.

## Deferred

- The feet choosing WHERE to build beyond their current spot (they stake
  where they stand; walking-to-build is the mind's `go_to` + deliberate
  craft).
- Impulse tuning the mind itself (faster cadence, bolder acts) — this arc
  keeps IMP a body-brain lever; the mind reads the sheet as ever.
- Group stakes (several beings finishing one beginning).

---

## As built (SHIPPED 2026-07-19, live-verified)

- **Part A — IMP** (`being_genome`): `ATTRS` gained `IMP` ("Impulse") as a
  first-class 8th stat; `POOL 40→45`, `BAND 40..50`; presets carry an IMP
  that fits the archetype (explorer 7 / artist 8 / scholar 3 / caretaker
  4); `ATTR_DEFAULTS = {"IMP": 5}` makes `effective_attributes` read a
  neutral impulse for every pre-IMP genome (the single inheritance gateway,
  so old parents breed IMP with no migration); `derive` gained
  `impulsiveness = imp/10`. Frontend: `IMP` in the conception `ATTRS`, "45
  points" copy, and the derive preview — labels/budget were already
  data-driven via `meta`. Tests: `test_being_genome`/`test_beings_store`
  moved to POOL 45 + IMP.
- **Part B — the impulsive feet** (`being_instinct`): the restless idle
  window scales with impulse (`FEET_IDLE_MINUTES × (1.5 − impulsiveness)`);
  a new `urge_to_build` trigger fires for impulse ≥ `BUILD_IMPULSE_MIN`
  (0.55) + a pressing create/explore drive (≥0.35) + open non-civic footing
  + no beginning already waiting; a new whitelisted verb `build` (offered
  only to restless hands on open ground) whose kind is a bodily choice; a
  physics floor in `stake_object` re-checks the impulse so a deliberate
  being NEVER stakes even if the model hallucinates the verb.
- **Part C — the stake** (`being_world` + store): `village_objects.state`
  gained `staked` (kind + snapped spot, reserved file path, no file/fee/
  boost, uncounted, non-blocking). `stake_object` (event `broke_ground`),
  `staked_object_of`, `prune_crumbled_stakes` (`STAKE_CRUMBLE_HOURS=24`,
  event `stake_crumbled`) — called in the tick AND the reflex pass. The map
  payload carries stakes with `staked:true`; `object_percepts` gained "A
  BEGINNING" sensing for a close neighbour's stake (no milestone, no explore
  serve — it isn't real). Render: iso draws it faint + dashed + greyscaled
  ("a beginning" title); FPV lays a walkable soil work-site (never a solid
  block — parity with `walk_blocked`); the object panel shows "a beginning
  · unfinished · on impulse / Started by <maker>".
- **Part D — the mind confirms** (`being_society` + `being_life`):
  `stake_confirm_percept` greets the being every wake while a stake waits
  ("YOUR HANDS BROKE GROUND … finish it or let it fall, it crumbles in
  ~Nh"). Digest fields `finish {object_id, name, inscription}` (→
  `finish_staked_object`: check→write→burn→stand, so a broke being or a
  failed write costs nothing; reuses the craft fee + proof file on the
  staked row; events `object_finished` + milestone `first_finish`) and
  `abandon {object_id}` (removes it, event `stake_abandoned`). A finished
  thing joins the standing layer — it now boosts, is discovered, counts.
- **Invariant kept**: the feet's stake is wordless + free; the inscription
  and the fee are the mind's alone. A deliberate being's feet never build
  (physics-gated on IMP, not just prompt-gated). The personality split is
  real: explorer/artist break ground on impulse; scholar/caretaker leave it
  to the mind's deliberate craft.
- Tests: `test_being_instinct_build.py` (17) — IMP first-class + derive +
  pre-IMP default + heritable; urge fires only for restless hands; the
  build verb; feet break ground end-to-end (wordless + free); deliberate
  never stakes; one-at-a-time; no boost/block/count; others sense a
  beginning but never discover it; crumble + survive; the mind meets +
  finishes (fee + file + standing + becomes real); finish needs name/words/
  fee; only the maker finishes/abandons; the full feet→mind loop; abandon.
  Full FD suite 959 pass + the same 8 pre-existing failures; tsc clean; the
  one new source-lint (an unused import) fixed. Live-verified: the "a
  beginning" render + panel on an isolated FD, and the conception meta
  serving pool 45 + the 8th Impulse stat.
- Bundle pair → index-DCjHHh2n.js + index--u1USjrv.css (BOTH moved; at
  commit git rm the prior pair index-Cwxy96BW.js + index-DrhB6W8j.css from
  02aa6cd).
