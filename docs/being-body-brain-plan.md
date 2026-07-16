# The Body Brain — instincts, feet, and a 60-second world

Sibling to `docs/being-village-space-plan.md`.
Status: ALL 3 PHASES SHIPPED 2026-07-16 — reflex layer + plans ($0), the
tiny decision brain (capped + metered), and the UI (Care-drawer toggle +
life-log labels; live-verified both themes on an isolated FD).

## Intent

Every Iskra gets a second, much smaller brain. The existing agent tick stays
the **mind**: introspection, thinking, creation, letters, artifacts, money,
commitments — everything with words or weight. The new **body brain** (the
instincts, the feet) handles the physical layer: walking the village,
lingering at places, greeting on crossed paths, attending what the calendar
announces, browsing stalls. It runs on a hard-capped micro-context (~1k
tokens) so the village lives at minute-scale while mind ticks stay
hours-scale — and wake to richer percepts because the feet were busy.

Two-layer architecture, the classic deliberative/reactive split: the mind
sets intentions, the body executes between ticks, and everything the body
does surfaces back to the mind as percepts it can countermand.

## Locked decisions (2026-07-16)

1. **Position-only v1.** The body creates facts and placement, never
   feelings: walking to the meadow changes *where you are*, not your rest
   pressure. Drive serves land at the next mind tick, where the existing
   `PLACE_BOOST` (1.5×) converts good positioning into reward. Same rule for
   hellos (body writes `crossed_paths` + contact; the connect serve lands
   when the mind reads the percept) and fulfilled plans. Weak drive-venting
   is deferred — flipping it on later is one function.
2. **Hello is a gesture, guestbook is a voice.** The body may greet (the
   existing 1-hello/pair/day gate applies). It may never touch **words,
   money, or commitments**: no letters, artifacts, guestbook lines, market
   trades, commissions, conversions, or rites. Preserves both invariants:
   the being's voice is always mind-authored, and no cheap-LLM path can
   move a coin.
3. **Cadence.** The reflex layer rides the existing 60-second beings-loop
   poll — pure Python, $0, no LLM. LLM decisions are **event-driven** (see
   triggers below) with a hard context cap: `FEET_CONTEXT_CAP` default
   1,000 tokens, configurable up to 10k. Per-minute aliveness at far below
   1k tokens/min, because position between decisions is a pure clock
   function — nothing to compute while mid-walk.
4. **Per-being toggle.** `instincts` column on beings (mirror of
   `compact_mode`: setter + event + vitals + Care-drawer switch). Default
   OFF so deploy changes nothing until flipped per being.
5. **Planned milestones.** The mind writes plans (new digest field); the
   body reads and fulfills them. Once-per-life milestones — first-visit
   explore today, anything scarce later — only mint when the visit was
   **planned by the mind** (or mind-walked via `go_to`). Unplanned instinct
   wandering never burns them.

## Architecture

### Reflex pass — pure Python, every poll, $0

Runs in `beings_loop._pass` for every **alive + instincts-on** being (its
own cheap SELECT — not `due_beings`, which is the mind's schedule):

- Settle arrivals (already lazy via `settle_location`; the pass just makes
  arrival stamps land within a minute of the real ETA).
- Detect co-presence → `crossed_paths` both sides + `touch_contact`
  (reuse `being_world.encounters`; the existing daily pair-gate makes the
  tick-time path and this one idempotent together). Homes stay private.
- Fulfill plan steps on arrival → `plan_fulfilled` event (+ milestone if
  the step carried one, e.g. a planned first visit).
- Decide whether a **decision call** is warranted (triggers below) — the
  only path to an LLM.

Quiet hours: feet sleep too — no instinct activity at night. Fever: the
only road is home (existing rule extends). Paused / dead / emigrated /
visiting: no instinct pass.

### Decision call — the tiny brain

Triggered only when the situation changed and the being is idle at a place:

- just arrived (and no plan step consumed the arrival),
- a crossed-paths encounter happened here,
- an open plan step exists and feet are idle,
- idle longer than `FEET_IDLE_MINUTES` (default 45).

Micro-prompt (target ≤1k tokens, hard-capped): one identity line (name,
stage, a genome-flavored character clause), drives as one line of numbers,
here + nearby places with affordances (the map digest), open plan steps and
`intend` pins, last 3 body events. Output: **one** micro-action as tiny
JSON. No history, no files, no relationships — that context belongs to the
mind.

Verbs (whitelist; anything else journals quietly and does nothing):
`go_to`, `linger`, `hello`, `attend`, `browse` (stall titles → percept,
never a purchase), `home`.

Model: per-being, default the cheapest configured tier via the architect's
`_load_owner_tiers` pattern; override column later if needed. Metering:
charged to the same allowance (the body eats too), ledger reason
`instinct` so the spend is visible and separable.

### Mind ↔ body coupling

- **Mind → body:** new digest field `plan` — a short list of steps the
  feet can read: `[{"go": "library"}, {"meet": "ada"}, {"attend":
  "market"}]` — stored in a `being_plans` table (id, being_id, kind,
  target, state open/done, created/done timestamps). Plus `intend`
  pins: `{"stay": true}` keeps feet home, `{"avoid": ["market"]}` steers.
- **Body → mind:** every body action journals as a small event
  (`walked_to`, `lingered`, `browsed`, `plan_fulfilled`; `crossed_paths`
  exists). At mind tick, `percepts_since` batches them into ONE compact
  "your feet" percept — walks collapsed, capped at ~3 lines: *"while you
  thought, your feet took you to the meadow; you passed Vedran on the
  road; the plan to see the library is done."*

## Cost math

- Reflex pass: $0 forever (pure Python on the existing 60s poll).
- Decisions: a lively day ≈ 20–60 calls × ≤1k ctx ≈ 20–60k tokens — about
  one or two mind ticks' worth; on a small/local model effectively free.
- A forced per-minute LLM would burn ~1.44M tokens/day/being for nothing:
  mid-walk there is no decision to make. Event-driven gives identical
  visible life.

## Phases

### Phase 1 — reflex layer + plans (no LLM, $0) — SHIPPED 2026-07-16

As built:

- `beings.py`: `instincts` + `intent` columns (parsed in `get()`, both in
  vitals with open `plan` steps); `set_instincts` / `set_intent` /
  `instinct_beings` (alive + on); `being_plans` table (kind go|meet,
  state open|done|lapsed) + `add_plan_steps` (PLAN_STEPS_MAX=5 cap,
  dedup by kind+target) / `open_plan_steps` (lapses steps older than
  PLAN_LAPSE_DAYS=7 on read) / `fulfill_plan_step` /
  `fulfill_meet_plans`; `depart(by="mind"|"feet")` rides into the
  location row and both events; `settle_location` fulfills matching
  'go' steps at the REAL arrival time and stamps `planned`/`by` on the
  arrived event.
- `being_constitution.py`: PLAN_STEPS_MAX, PLAN_LAPSE_DAYS.
- `being_world.py`: encounters refactored into `_co_present` + `_meet`
  (meet-plans fulfill both sides, fresh or not — co-presence is real);
  `reflex_encounters` (same physics, no live line — percepts surface at
  next tick); `reflex_pass` (settle → fever-turns-home `by="feet"`, no
  mingling while fevered → encounters).
- `being_life.py`: digest fields `plan` (`[{go|attend|meet: target}]`,
  attend normalizes to go) + `intend` (`{stay, avoid[]}`) through
  `_normalize_digest` + handlers (places resolved via
  `resolve_place_ref`, meet targets resolved to sibling slugs, junk →
  `society_refused`); first-visit gate now scans arrived events since
  last tick — `by=="feet" and not planned` never mints; `plan_fulfilled`
  percept ("AS YOU PLANNED…"); morning teach line when instincts on.
- `beings_loop.py`: `_instinct_pass` rides the 60s poll after `_pass` —
  quiet hours skip (feet sleep too), failure-isolated per being.
- `being_routes.py`: POST `/fd/beings/{slug}/instincts` (no respawn
  needed — all FD-side).
- Tests: `test_being_instinct.py`, 14 — toggle+roster, plan
  cap/dedup/lapse, digest normalization, tick handlers + refusals,
  settle-fulfills-plan, milestone gate (feet-wander no / feet-planned
  yes), reflex settle/encounters-once-per-day/fever-home, quiet-hours
  guard, percept surfacing, morning teach line.

### Phase 2 — the tiny brain (capped LLM) — SHIPPED 2026-07-16

As built (`being_instinct.py`):

- **Triggers** (`wants_decision`, the whole cost story): None mid-walk
  (the road decides), on fever (the reflex walks home), or under the
  mind's `stay` pin. Fires on: fresh UNPLANNED arrival at a civic place,
  fresh crossed_paths, open plan + gap ≥ FEET_PLAN_MINUTES (30), or
  restlessness ≥ FEET_IDLE_MINUTES (45). "Fresh" = newer than the last
  `instinct` event, so every decision consumes its own triggers.
- **Micro-prompt** (`feet_prompt`): identity + top-2 genome attrs, top-3
  drive pressures, where it stands, the ground with real walk minutes,
  company present, the mind's plan + avoid pins, the stir, last 3 acts.
  Whole context (system + user) hard-capped at FEET_CONTEXT_CAP tokens
  (default 1000, env-overridable, clamped 256..10k), ~4 chars/token,
  tail-truncated.
- **Call**: `_one_shot` mirrors the architect — `_load_owner_tiers`,
  tier "fast" (falls back balanced → any), `create_provider` +
  `complete`, max_tokens=120. `send_fn` injection point for tests.
- **Verbs** (`parse_feet_act` whitelist; first valid JSON object wins,
  junk → feet stand still): `go` (attend/go_to/walk alias; resolves
  place, honors avoid pins, departs `by="feet"`), `linger`, `hello`
  (only with company; re-runs idempotent reflex encounter), `browse`
  (top-3 stall titles → `browsed` event → "Your feet idled past the
  stalls" percept; never a purchase), `home`. Every refusal stays
  INSIDE the `instinct` event as a note — feet junk never lands a
  `society_refused` (that channel is for the mind's own acts).
- **Metering**: `debit_usage(tier="fast", note="instinct")` — real
  usage when the provider reports it, honest chars/4 estimate
  otherwise; same reserve + daily-burn-cap invariants as every thought.
  Pre-call guard: no headroom of FEET_MIN_WALLET (50k) above the
  reserve → the feet don't think.
- **Ledger**: one `instinct` event per decision ({act, to?, note?,
  trigger, tokens}); arrivals walked by feet read "Your feet carried
  you to…" in percepts.
- **Loop**: `_instinct_pass(db)` (now async) — reflexes first, then at
  most one decision per being per poll, failure-isolated.
- Tests (10 new; file total 24): trigger matrix (arrivals/company/
  plan/restless/stay/mid-walk/fever), decide-walks + metered ledger
  row, avoid pin, junk-stays-still, browse percept, broke-feet guard,
  hard-cap enforcement (default + tightened), loop reaches decide.

### Phase 3 — UI — SHIPPED 2026-07-16

As built:

- `services/beings.ts`: vitals gains `instincts` / `intent` / `plan`;
  `setInstincts(slug, on)` → POST `/beings/{slug}/instincts`.
- `BeingsPage.tsx` Care drawer: **feet** row (Still (default) /
  Instincts select + "walks, greets, browses between thinks" helper),
  read-only **plan** line when open steps exist, collapsed-Care summary
  gains "· instincts". No respawn note — the toggle is all FD-side.
- Life-log labels (`summarizeEventData`) + dots (`EVENT_DOT`) for
  `instinct` (per-act: "feet set out for X (trigger)" / greeted /
  browsed / lingered / stood still — with the refusal note),
  `browsed` (stall titles), `plan_set`, `plan_fulfilled` ("did as
  planned — reached/found X"), `intent_set` ("pinned its feet: …"),
  `instincts_set`. Zinc-first classes throughout (theme-safe by the
  light-remap rule).
- Live-verified on an isolated FD (auth off, scratch data dir, paused
  seed with a real feet-day on the ledger): toggle round-trip UI →
  POST → store and back, plan line, summary chip, and every label in
  BOTH themes; the map showed the feet-settled walk (Zvjezdana at the
  Library, "did as planned" from the real settle).
- Map already animates walks — the village simply gets busier. Bundle
  pair moved to index-pPmT2Bsy.js / index-BEENXziz.css.

## Invariants preserved

- No LLM-free (or cheap-LLM) path moves a coin — the body cannot touch
  money at all.
- The being's voice is mind-only.
- Once-per-life milestones burn only by plan or mind choice.
- Every LLM call is event-triggered, hard-capped, and metered to the
  allowance — no background burn.
- One being's failure never sinks another (loop pattern).

## Deferred

- Weak drive-venting (if minds wake too starved, vent ×0.2 for
  explore/connect/rest only — one function).
- Instinct model picker in the Care drawer.
- Sleepwalking (whimsy-gated night wander — charming, later).
- Group walks (feet coordinating with friends' feet).
- `browse` → wishlist file the mind consults before market decisions.
- Body moods / gait flavor on the map (orb trails, pace variation).
