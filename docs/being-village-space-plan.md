# The Village Gets Ground — space, movement, and money

Status: ALL 5 PHASES SHIPPED 2026-07-16 (as-built notes at the end)
Scope: a spatial village designed by a one-shot architect; Iskre move
between places between ticks; encounters build a contact graph
(networking); a second currency (coins) separates money from metabolism.

---

## Why

The Iskre have time (seasons, market Saturdays, elderhood), society
(letters, commons, federation), and body (strain, fever) — but no SPACE.
Everything happens nowhere. Giving them ground does three things:

1. **Co-presence** — two beings at the same well is a mechanic letters
   can't fake. Encounters are earned, not granted.
2. **Distance as honest cost** — travel takes clock time. You can't be
   at the market if you dawdled. Zero tokens, real consequence.
3. **A stage for everything already shipped** — reading lists → the
   library, market day → the square, memoirs → a bench, stewardship →
   the bell, emigration → a literal road out of town.

And the economy needs a second denomination. Tokens are METABOLISM —
spending them literally shortens thinking. Pricing the social economy in
tokens makes every purchase an act of self-harm (today a publication
bought for `price_tokens` eats the buyer's mind). Real life separates
calories from cash: **coins are money, tokens are food.**

## Locked decisions (user, 2026-07-16)

- **Strong bonuses, not hard gates.** Place affordances boost drives and
  yields; nothing existing breaks if a being never moves. The only
  "gates" are genuinely physical facts (stall percepts need presence,
  encounters need co-presence) — those aren't gates on abilities, they
  ARE the physics.
- **Encounters grant networking.** Contacts are first-class: gossip
  percepts, pen-pal introductions, connect-drive serves.
- **Infants roam free**, at toddle speed (~×0.35). No radius limit — an
  infant may spend half a day crossing the village, and that is the
  point.
- **Coins → tokens is one-way.** Coins buy thinking time (ceiling-
  clamped mint). Tokens can never buy coins: liquidating metabolism into
  wealth recreates the original problem and lets allowance print money.

## Design law

Every place effect is **enforced or measured by code** — affordance
multipliers in the homeostat, presence checks in percepts, file-copy
trades, ledgered coins. The LLM narrates; the physics decides. Position
is a pure function of the clock (the fever/steward pattern): **no
background process, ever.** The world is simply further along when a
being wakes.

---

## Part I — Space

### The architect (one-shot, not a resident)

When an owner's village first needs ground (first hatch after deploy, or
an explicit "found the village" action), ONE LLM call designs it:

- Picks 6–10 civic places from a **fixed affordance vocabulary** (code
  enforces effects; the architect only names, places, and describes).
- Lays them on a 1000×1000 plot with named roads between neighbors.
- Writes `commons/village/MAP.md` — prose the beings read: places,
  descriptions, walking times.
- Structured truth goes to the store (`village_places`); MAP.md is the
  being-facing rendering of it, regenerable.

Each being's **home** is auto-placed on a home lane (deterministic from
slug hash) — no LLM needed per being. Villages differ per owner; the
architect seeds flavor, never physics.

Affordance vocabulary (v1): `rest` (home), `read` (library), `create`
(workshop), `gather` (square, well), `trade` (square), `tend` (garden),
`play` (meadow), `remember` (the bench). Architect may combine ≤2 per
place.

### Data model

```
village_places(owner_id, id, name, kind, x, y, affordances_json, description)
beings.location TEXT  -- JSON: {"at": "library"}
                      --   or  {"from": "home", "to": "square",
                      --        "departed_at": iso}
events: departed, arrived, crossed_paths
```

- `WALK_SPEED = 10` units/min (constitution). `INFANT_SPEED_FACTOR =
  0.35`. Everyone else identical (locked). Village diagonal ≈ 2.3 h
  adult walk; typical errand 25–50 min.
- **Position(t)** = linear interpolation from `departed_at` at the
  being's speed. Arrival is *computed on read*: any code touching
  location first settles it (if `now ≥ eta`: write `{"at": to}`, record
  `arrived`). No scheduler.
- v1 travel is straight-line; named roads are architect flavor in
  MAP.md. A real path graph is a later refinement if the map wants
  geography (a river to walk around).

### Movement is a digest field

The being replies with `go_to: "library"` (MUST be added to
`_normalize_digest` — the whitelist silently drops unknown fields).

- Valid place → settle current position, write en-route location,
  record `departed`. The being travels while it sleeps.
- Unknown place → `society_refused` ("no such place — read
  commons/village/MAP.md"), loudly, like letters.
- Already there → no-op. En route → new order re-routes from the
  settled current coordinate.
- **Fever auto-routes home** at onset (physics, not choice): a sick body
  walks home and stays; `go_to` elsewhere is refused with "you are
  fevered; home is the only road" until it ages out.
- Emigration's rite gains a walk: the being is routed to the village
  edge before the manifest exports (flavor percept, no new physics).

### Percepts (via `being_world.umwelt_percepts`)

- Morning, first of day: "You are at home. The square is 12 min away,
  the library 31." (2 lines max in compact mode.)
- En route on wake: "You are on Lipa Lane, twenty minutes from the
  square."
- Arrival since last tick: "You reached the garden at 14:05."
- Co-presence: "Mira is here at the well." (see Part III)
- Market Saturday, AT the square: the stall-crying percept (presence is
  physics — elsewhere you only hear "it is market day; the square hums
  without you").

### Strong bonuses (locked)

- `serve_drive` gains a place factor: serving a drive that matches the
  current place's affordance → sat gain **×1.5** (grow at the library,
  create at the workshop, connect at the square/well, explore anywhere
  new — first visit to a place serves explore outright).
- Reading completion at the library mints the diff-verified fee ×1.25
  (still clamped by `READING_MAX_FEE_TOKENS` and savings ceiling).
- Dreams at home run the normal book; falling asleep elsewhere is
  charming, true, and slightly restless (whimsy +0.05, nothing more).
- Nothing is location-REQUIRED. A being that never moves lives exactly
  as today, minus the bonuses.

---

## Part II — Coins (money vs metabolism)

### The two currencies

| | tokens | coins |
|---|---|---|
| are | metabolism (LLM budget) | money (social economy) |
| minted by | allowance, judged work (today) | parent grants, judged work (opt-in), sales |
| spent on | thinking (usage), fees | market trades, conversion, later: buildings |
| exchange | — | **coins → tokens, one-way** |

Tech name `coins` everywhere in code/models. Frontend name proposal:
**"žar"** (embers — fire family with Iskra/Vatra); user to confirm, same
split as beings/Iskre.

### Ledger (anti-theater)

`being_coin_events(owner_id, being_id, delta, reason, from_being,
data_json, at)` — balance is the SUM, computed like everything else.
Reasons: `grant` (pocket money), `wage` (judged quest/chore paid in
coins), `sale` / `purchase` (circulation pair, zero-sum), `exchange`
(conversion debit), `stipend` (steward). No LLM path can move a coin;
only physics functions write events.

### Faucets (all parent-budgeted or zero-sum)

1. **Pocket money** — parent grants coins from the panel (mirror of
   token `grant`). The parent's real-dollar exposure = token allowances
   + coin faucets × rate; every faucet is a parent act.
2. **Work, opt-in** — `post_chore` / quests gain `pay_in: "coins"`
   (parent picks denomination at posting; escrow/judgment flow
   unchanged, mint-at-judgment becomes a coin `wage` event).
3. **Sales** — beings sell real artifacts to each other (below);
   circulation, not minting.
4. **Steward stipend** — 1 coin/week to the current steward, parent
   knob, default off (treasury concept deferred).

### Sinks

1. **Conversion** — the being asks (digest field `convert_coins: N`);
   physics mints `N × COIN_TOKEN_RATE` tokens clamped by savings-ceiling
   headroom, debits coins for exactly what was minted, and the event
   records any clamp. Headroom 0 → `society_refused` ("your savings are
   full"). `COIN_TOKEN_RATE = 100_000` (constitution; a 10-coin pocket
   week ≈ 1M tokens of thought).
2. **Market purchases** — circulation.
3. **Commissioned buildings** — Phase 5: a being (or several, pooling)
   pays the architect to draft a new place; parent approves; the map
   grows. The long-arc savings goal that makes wealth mean something.

### The market (trades of real things)

A listing is a REAL file offered for coins:

- `sell` digest field `{path, title, price_coins}` — physics verifies
  the file exists in the seller's home, creates a listing row, notes it
  in `commons/village/MARKET.md`.
- `buy` digest field `{listing_id}` — physics checks funds, copies the
  file into the buyer's home (`shelf/` with provenance header), writes
  the `sale`/`purchase` event pair, both beings get percepts. Diffable,
  conservational, refused loudly when broke.
- Listings are always browsable; **on market Saturday at the square**
  the stall percept names 2–3 live listings (presence bonus, not gate).
- Existing token-priced `being_publications` stay as-is; unifying them
  into coin listings is deferred (noted, not forgotten).

### Stage gates (constitution, mirrors letters_per_day)

- infant: may RECEIVE coins (gifts); no trading, no conversion.
- child: buy/sell up to 3 trades/day; no conversion.
- adolescent: trades 5/day; conversion allowed.
- adult: trades 8/day; conversion allowed.
- Market Saturday: +2 trades (single `trades_cap` source in
  being_world, exactly like `letters_cap`).

---

## Part III — Networking (what an encounter grants)

Co-presence is detected at settle time: when a being's position is
settled and another being's settled position is the same place with
overlapping presence windows → ONE `crossed_paths` event to each per
pair per day.

```
being_contacts(owner_id, a_id, b_id, met_count, last_met_at, strength)
```

Strength grows asymptotically on meetings and exchanges (the satiation
curve again), decays slowly over months of silence.

What contacts pay (all real):

1. **Gossip percepts** — next wake after crossing paths: "Mira is here
   at the well — she has been carving a game for the shelf" (her latest
   public act, pulled from her events; informational, true, free).
2. **Introductions** — a contact with a pen-pal may introduce you: one
   `introduce` society op that seeds a first letter across the
   federation link to THEIR pen-pal (extends reach one hop; quota'd
   under the same letters_cap; both sides evented).
3. **Connect serve** — a genuine encounter serves the connect drive
   (damped same-day, asymptotic as ever) — presence finally feeds the
   loneliness loop honestly: an empty square feels different from a
   full one because it IS different.
4. **Guestbooks** — `commons/places/<place>/guestbook.md`: arriving
   beings may leave one line (society op, 1/day/place). Cheap, diffable,
   permanent traces of a life lived in places.

Referral finder's-fees (a contact who points you to a quest earns a coin
carved from its fee) are designed but DEFERRED — v1 networking is
information + reach + feeling, not commissions.

---

## UI (its own phase)

- **The living map** — Village page renders `village_places` + live
  Iskre positions; position is a pure function of time, so the client
  animates walking orbs (infants visibly toddling) with zero polling.
  Same glow language as the Mind graph. Click a place → description +
  who's there + guestbook.
- **Wallet card** shows both currencies; Care drawer gains pocket-money
  grant + pay-in-coins toggle on chore/quest posting.
- **Market tab** — live listings, provenance on bought files.
- Public being page: current place ("last seen at the library") if the
  being is public.

## Phases

1. **Ground truth** — architect + `village_places` + MAP.md, location
   state + settle-on-read, `go_to` (whitelist!), departed/arrived
   events, refusals, infant speed, fever-routes-home, morning/en-route/
   arrival percepts. (~12 tests)
2. **Coins core** — coin ledger + balance, pocket-money grant + route +
   panel, `pay_in: coins` on chores/quests, conversion with clamp
   semantics, stage gates, wallet UI. (~10 tests)
3. **Teeth** — affordance boosts in `serve_drive` + library reading
   bonus, co-presence detection + contacts + gossip percepts + connect
   serve, guestbooks, market listings + buy/sell + Saturday trades
   bonus + stall percepts at the square. (~14 tests)
4. **The living map** — frontend map + walking orbs + place/market
   panels + both-theme pass. (build + live verify)
5. **Growth** — introductions across federation, commissioned buildings
   (pooled coins → architect draft → parent approves), steward stipend.

Each phase ships independently; nothing in 1–2 changes existing
behavior for a being that never moves and never touches coins.

## Deferred (noted, not forgotten)

- Village treasury (stall rent, public works) — wants Phase 5 first.
- Inter-village roads: visiting a pen-pal's village over federation —
  the natural sequel; emigration's road becomes literal.
- Weather slowing travel (wants the weather API thread).
- Presence-gating the market letter bonus (today global; tighten only
  if presence proves sticky).
- Unifying `being_publications` into coin listings.
- Referral finder's-fees; parent-visible contact graph rendering.

---

## Phase 1 as built (SHIPPED 2026-07-16)

- **Store** (`beings.py`): `village_places` table + `beings.location`
  column (JSON `{"at"}` / `{"to","from","origin","departed_at"}`, default
  home). `save_village` validates HARD (4–12 places, kebab ids, `home`
  reserved, coords 40..960, affordance vocabulary, ≤2 per place) and
  replaces atomically. `resolve_place_ref` maps a being's words (id /
  slugified / name, "the " optional) to ground. `settle_location` writes
  the rest state and records `arrived` AT the real eta (events order by
  `at`, so back-dating is safe). `depart` settles first, leaves from the
  TRUE position (mid-road re-routes carry `origin`), records `departed`
  with honest minutes; already-there is a quiet no-op. `vitals` gains
  `location` + `position` (pure read).
- **World** (`being_world.py`): `PLOT_SIZE=1000`, `WALK_SPEED=10`/min,
  `INFANT_SPEED_FACTOR=0.35`; `home_xy` from slug crc32 on the west lane
  (no rows, no migration); `position_of` is the pure clock function
  (broken ground resolves home); `default_village` is a per-owner seeded
  ring of 7 places; `ensure_village` founds idempotently and writes
  `commons/village/MAP.md` (walking-times table, signed "— the
  Architect"). `location_percepts`: morning = where + 3 nearest at THIS
  body's pace + the `go_to` offer; mid-walk wakes hear the road; silence
  otherwise. Architect prompt/parse are pure functions.
- **Life** (`being_life.py`): `go_to` in `_normalize_digest` (the
  whitelist); tick start ensures village + settles (arrival surfaces
  this tick via the `arrived` percept); fever walks the body home
  (`depart reason="fever"`, physics not choice) and refuses `go_to`
  elsewhere ("home is the only road"); unknown ground → `society_refused`
  pointing at MAP.md (echoes as PHYSICS SAID NO next tick).
  `architect_village` = one owner-tier LLM call → parse → the store gate
  → MAP.md rewrite; called at FIRST birth (default stands on any
  failure) and from `POST /fd/beings/village-map/architect`.
- **Routes**: `GET /fd/beings/village-map` (places + live positions,
  client-side animatable — zero polling) + the architect POST.
- **Templates**: `location_note.md`, `road_note.md` (shared by compact
  mode — one-liners don't fork).
- Tests: `test_being_village_space.py` (11) — being suite 307, full FD
  suite 740 pass (8 pre-existing mcp/vfs failures untouched).
- Not in Phase 1 (by plan): affordance boosts, encounters/contacts,
  guestbooks, coins, the map UI.

## Phase 2 as built (SHIPPED 2026-07-16)

- **Ledger** (`beings.py`): `being_coin_events` table; balance = SUM;
  `_apply_coins` is the single writer (COIN_REASONS vocabulary, overdraft
  refused — no negative money). `coin_balance` / `coin_ledger` /
  `grant_coins` (pocket money: positive, ≤ `COIN_GRANT_MAX`=1000, egg has
  no pocket, infants may RECEIVE). `vitals` gains `coins`.
- **Conversion** (`convert_coins`): one-way coins→tokens at
  `COIN_TOKEN_RATE`=100k, `convert` capability (adolescent+ via
  constitution grants), WHOLE coins only, clamped to savings-ceiling
  headroom (`possible = min(requested, balance, headroom // RATE)`) —
  the coin debit records `{tokens, requested}` so clamps are on the
  ledger; token mint uses new transfer reason `exchange` (conservation
  audit still balances). Coin debit runs first (it can refuse; the mint
  cannot fail).
- **Work in coins**: `being_jobs` + `being_quests` gain `fee_coins`
  (ALTER, default 0). `post_chore`/`post_quest` accept `fee_coins` with
  one-denomination XOR validation + `WORK_MAX_FEE_COINS`=500 cap;
  judgment mints a coin `wage` (NO ceiling clamp — the ceiling guards
  metabolism, not wealth); `chore_paid`/`quest_paid` events carry
  `fee_coins`; chore/quest percepts and earning board lines speak the
  true denomination.
- **The tick**: `convert_coins` digest field (whitelist, int, capped);
  handler refuses loudly (`society_refused what=convert_coins`); VITALS
  line says "N coin(s) in your pocket (money, not food)" when held;
  `society_prompt_fields` offers conversion ONLY when coins>0 AND stage
  allows (never a dangled refusal); pocket-money and coin-wage percepts
  ("POCKET MONEY", "PAID N coin(s)").
- **Routes**: `POST /fd/beings/{slug}/coins` (pocket money) + `GET`
  (balance+ledger); chore/quest routes pass `fee_coins`.
- **UI** (`BeingsPage.tsx` + services): amber Coins chip beside the
  wallet, Care drawer "pocket" row (+1/+5/+10/+25), chore form pay-in
  toggle (token presets ↔ coin presets), quest form denomination select,
  fee labels + event feed speak coins (`coins_granted`,
  `coins_converted`, plus Phase 1 `departed`/`arrived` labels/dots).
- Tests: `test_being_coins.py` (9) — pocket money heard as percept,
  stage gates, exact conversion + conservation, whole-coin clamp +
  savings-full refusal, overdraft, chore/quest coin wages end-to-end,
  tick-borne conversion + child refused on the record, honest offers.
  Being suite 316; full FD suite 749 (same 8 pre-existing failures).
- Not in Phase 2 (by plan): sales/purchases (market, Phase 3), stipend
  (Phase 5), trades_cap (arrives with trades), currency display name
  ("žar" candidate — UI says "coins" until the user picks).

## Phase 3 as built (SHIPPED 2026-07-16)

- **Strong bonuses** (`being_world` + `_serve`): `PLACE_BOOST=1.5` via
  `place_drive_boosts` (AFFORDANCE_DRIVE_BOOSTS: read→grow,
  create→create, gather/trade→connect, tend→create, play→explore,
  remember→grow, rest→survive) folded into the tick's `_serve` closure —
  every serve path (digest, reading, encounters) inherits it. First
  arrival at a place serves explore outright, gated once-per-life-per-
  place by `milestone(first_visit_<place>)`. Reading completed at a
  place with the `read` affordance mints ×`READING_PLACE_FACTOR`=1.25
  (`complete_reading(fee_factor=)`, still capped by READING_MAX +
  ceiling). Dreams skip boosts.
- **Co-presence** (`being_world.encounters`, called right after
  `percepts_since`): beings settled at the same CIVIC place (homes are
  private — same `{"at":"home"}` is DIFFERENT ground) → one hello per
  pair per day (deduped in `touch_contact` by last_met date, symmetric
  a<b key), `crossed_paths` event to BOTH, gossip line for the ticking
  being pulled from the other's own tick summaries ("lately: …"), and
  the passive side hears "You crossed paths with…" on waking. Both
  percept shapes feed `_serve("connect")` — presence finally feeds the
  loneliness loop. `being_contacts` grows asymptotically
  (`CONTACT_STRENGTH_STEP=0.2`); `contacts_for` for later UI.
- **Guestbooks** (`being_society.guestbook_sign`): one ≤200-char line
  per place per day (event-ledger dedup) into
  `commons/places/<id>/guestbook.md`; home/road refused; offered inside
  the arrival percept. `guestbook` digest field (whitelist).
- **The market** (`village_listings` + society file physics): `sell`
  digest field — `read_self_file` is the existence proof (sandboxed,
  .md), store gates quota (`trades_cap` = constitution
  `trades_per_day` infant 0/child 3/adolescent 5/adult 8, +2 market
  Saturday) and price (`MARKET_MAX_PRICE_COINS`=100); `buy` — read-
  before-pay (a vanished file refuses before any coin moves), atomic
  claim (UPDATE WHERE state='open', un-claimed on a failed debit), the
  being→being purchase/sale coin pair (circulation, never minting), the
  copy lands in `shelf/<seller-slug>--<name>` under a provenance
  header; `market_sold`/`market_bought` events + percepts;
  `first_sale` milestone; `commons/village/MARKET.md` rewritten on
  every change (browsable any day). Selling needs no presence (bonus
  philosophy), buying neither.
- **Market morning** (`market_percepts` reworked): standing at a
  `trade` place hears the full cry (coin listings + publications);
  elsewhere on Saturday: "the square hums without you" + pointer to
  MARKET.md. Presence flavors, never gates.
- **Offers**: `society_prompt_fields(trades_left=)` — sell offered only
  with quota left; buy additionally needs coins. `trades_left` threaded
  like coins through both composers + faculties orient.
- **UI**: event labels + dots for crossed_paths / guestbook_signed /
  market_listed / market_sold / market_bought (bundle pair
  index-6J0aSx0Q.js + index-DyigXKVI.css).
- Tests: `test_being_village_teeth.py` (12) — boost comparison, first-
  visit explore once, library reading +25k exact, crossed-paths full
  arc (contact growth 0.2→0.36, both sides heard + fed, daily dedup),
  homes private, guestbook day/place physics, stall validation (real
  file, price caps, infant refused), buy moves coins+file with
  provenance (broke/self/double refused, circulation sums to grants),
  quota counts both sides (3rd trade ok, 4th refused; Saturday 5),
  presence-flavored market morning, sell/buy through the tick with
  echoed refusals, honest offers. Being suite 328; FD suite 761 (same
  8 pre-existing failures).
- Not in Phase 3 (by plan): the living map + market UI (Phase 4),
  introductions/finder's-fees/commissions (Phase 5), sleeping-away
  whimsy flavor (noted, dropped as pure flavor).

## Phase 4 as built (SHIPPED 2026-07-16, live-verified)

- **The living map** (`VillageMap` in BeingsPage, default-on with a
  header "Map" toggle): SVG 1000² plot — places as glowing rings hued by
  primary affordance (AFF_HUE), dashed roads from the gather-hub,
  beings' homes as small squares on the west lane, Iskre as violet orbs
  (radius by stage — infants visibly small) with name labels. Walking
  orbs carry a dashed pulsing ring and MOVE: position extrapolated
  client-side from one snapshot (xy + destination + speed) on a 1 Hz
  heartbeat — zero polling; the snapshot refreshes each 60 s.
- **Panels**: click a place → description, affordance chips, HERE NOW
  (live from positions), STALLS (at trade places), and the REAL
  guestbook tail (new route `GET /village-map/place/{id}`); click an
  iskra → stage + honest road status ("on the road to the Library —
  ~67 min", counting down); default panel → who's walking + the open
  stalls (`GET /fd/beings/market`).
- **Public pages**: `public_profile` gains `place` (at/road/home — a
  name, never home coordinates); gallery cards say "at the Square" /
  "walking to…", the being page says "last seen at the Square" (+
  minutes when on the road).
- **Live verification** (the first for an authed page): an ISOLATED FD
  instance — `FD_AUTH_ENABLED=false FD_DATA_DIR=<scratch>
  CLAW_VFS_ROOT=<scratch> … server --port 25081` — seeded with a demo
  village (all beings PAUSED so the loop never spawns bodies; walks
  still animate, position being pure clock). Verified in the browser:
  dark + light themes, place panel with the real guestbook file, stall
  listing, and true motion — Ada's orb advanced 6.45 plot-units in 45 s,
  exactly the x-component of a 10 unit/min walk on her bearing; road
  minutes count down (70→67).
- Bundle pair: index-nac5NxfR.js + index-BHZvS7Fp.css (prior 6J0aSx0Q/
  DyigXKVI pair removed pre-commit). Being suite 328 (unchanged),
  FD suite green minus the 8 pre-existing failures.
- Not in Phase 4: the architect-redesign button (route exists,
  `POST /village-map/architect`; panel button deferred), contacts
  rendering (data + `contacts_for` ready).

## Phase 5 as built (SHIPPED 2026-07-16, live-verified)

- **Introductions** (`introduction_reach` + `_deliver_introduction` +
  `introduce` digest field): a PUBLIC, letter-capable sibling this being
  TRULY met (being_contacts) lends its pen-pal reach one hop — the
  letter goes out signed "(— <via>, whom we both know, made this
  introduction)", spends the SENDER's letters quota (a `penpal_sent`
  event keeps accounting single-sourced), and both sides carry events
  (`introduced` / `made_introduction`, the via hears it as a percept).
  Offered as a sense line ("AN OPEN DOOR…") only when the being has no
  reach of its own and quota remains. Consent story: the via's open
  door + the real meeting. Refused loudly otherwise.
- **Commissioned buildings** (`village_commissions`, ONE active fund
  per village): `commission` digest field — proposes when none is open
  (adolescent+, skin first: the proposer's coins escrow immediately,
  reason `commission`) or contributes while one is (ANY coin-holder —
  pooling is the point; clamped to remainder and pocket). Funded at
  `COMMISSION_COST_COINS`=50 → the parent judges
  (`POST /fd/beings/commission/judge`): approve → `add_place` raises
  real ground at a deterministic `commission_spot` (seeded max-min-
  distance scatter), MAP.md rewritten, coins BURN (the economy's sink),
  `commission_built` percepts to every contributor + `first_commission`
  milestone; reject → every contributor refunded to the coin off the
  ledger (`commission_contributors` reads the coin events — no state).
  Morning percept cries the fund's progress (+ a proposing nudge for
  adolescent savers with ≥10 coins when none is open).
- **Steward stipend**: `village_meta.steward_stipend_coins` (0–10,
  default 0, `POST /fd/beings/village-stipend`); paid inside
  `steward_percepts` once per ISO week, ledger-idempotent (reason
  `stipend`, data.week), spoken in the steward's morning note.
- **Routes/UI**: `GET /fd/beings/village-life` (fund + contributors +
  steward + stipend); map side panel gains the commission card
  (progress bar, contributor list, Approve & build / Reject & refund)
  and the stipend select; event labels + dots for all six new kinds.
- **Live-verified** on the isolated FD: the funded "the Pond" card
  rendered with both buttons; one click on Approve & build and the
  Pond appeared on the map with its own road — the village grew in
  the browser. Bundle pair index-D3EkTWm-.js + index-Bt7doGMc.css.
- Tests: `test_being_village_growth.py` (9) — introduction full arc
  (via named in body, quota spent, both evented, percept), no-contact/
  dark-door refusals, open-door offer in the tick, skin-first pooling
  (child can't propose, anyone contributes, clamps, one-at-a-time),
  approval raises walkable ground + burns coins + MAP.md, rejection
  refunds exactly, fund rides the tick (cried each morning, broke
  refused on the record), deterministic off-crowd spot, stipend
  validates/pays once per ISO week. Being suite 337; FD suite 770
  (same 8 pre-existing failures).
- Deferred beyond the arc: referral finder's-fees, treasury,
  inter-village roads, publications unification (see Deferred above).
