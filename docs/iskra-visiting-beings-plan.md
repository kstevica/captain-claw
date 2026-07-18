# Visiting beings — a guest who truly walks the village

Today a being can be sent to "visit" another village (§9.1 federation): its
home machine dials an outbound WebSocket to the host, presents the village
secret, and the host can pull its files/journal/graph and relay pen-pal
letters. But the guest is invisible and inert:

- **It isn't shown in the host village.** `village_map_payload` iterates only
  local `store.list(owner)`. A visitor is a contact row (`being_visitors`:
  origin, slug, name, profile, last_seen) with **no position** — so it never
  appears on the 2D map or in the FPV.
- **It doesn't know it's visiting.** Its tick runs on its home machine against
  its OWN village; its parent's nudges (`depart`) target its own places; its
  map/FPV show its own village. Nothing tells it "you are in Willowmere now."
- **Locals can't sense it.** Co-presence (`_co_present`/`encounters` →
  `crossed_paths`) only compares local beings by shared place id; a visitor has
  no place, so it's never "here." Only name-addressed pen-pal letters cross.

We want a guest that **appears in the host village, walks its buildings under
its parent's nudge, sees that village, and is sensed and spoken to by the
locals** — both directions.

---

## The architecture choice (please confirm)

The guest's *life* — tick engine, drives, walks — runs on its HOME machine.
The host village lives on a DIFFERENT machine, reachable only because the guest
dialed out (NAT-friendly reverse tunnel: sender→host `hello`/`beat`/`res`;
host→sender `req`). So "where does the guest stand in the host village?" has two
answers:

**Option A — host-authoritative (recommended).** The HOST owns the guest's
position in its own village. It seats the guest on arrival, walks it when the
guest's parent nudges (the nudge flows UP the link), and streams the guest's
current place/surroundings back DOWN so the guest becomes aware. The sender
never needs the host's map geometry.
- *Delivers all three asks*, reuses the existing request/beat tunnel, keeps the
  home machine simple, and makes the host the single source of truth for its
  own map (no cross-machine geometry drift).
- *Trade-off*: the guest's body is "puppeted" by the host rather than pathing
  the foreign grid itself — but it is genuinely aware, positioned, moving, and
  interacting.

**Option B — mirror.** The host ships its full layout (places, grid, roads)
down the link; the sender computes the guest's position + A* path IN the host
village and streams coordinates up. The home machine renders the host village
natively.
- *Most faithful* (the guest truly walks the foreign grid, its own FPV renders
  the host village locally) but **much heavier**: ship + version the layout,
  dual-village location state on the sender, A* over a foreign grid, and every
  "where am I" path must branch home-vs-host. Weeks, not days.

**Recommendation: Option A.** It satisfies "shown in the village," "parent
nudges to *that* village's buildings," "sees *that* village" (the guest's
parent views the host map by proxying it down the link), and "locals sense and
communicate" — without the mirror's cost. The plan below assumes A.

---

## Phase 1 — the guest has a body in the host village

Give a linked visitor a real position and render it.

- **Position + wander (host-side, $0).** On link (`upsert_visitor`), seat the
  guest at the village **square** (the civic entry). A tiny host-side reflex —
  folded into the existing `reflex_pass`/beat, no LLM — lets an idle guest
  drift between civic places at a gentle pace, so it feels alive between its
  parent's nudges. New columns on `being_visitors`: `at` (place id / "road"),
  `x`, `y`, `path`, `departed_at`, `minutes` — the same walk shape a local
  being uses, so `position_of`-style extrapolation just works.
- **Render it.** `village_map_payload` gains a `visitors` list (or tags them
  into `beings` with `kind:"visitor"`): slug, name, `from` (origin label),
  stage, xy/at/to/path, avatar. The 2D map (`walk.posOf`) and the FPV draw a
  visiting figure wearing a **"visiting from <village>"** pill, visually
  distinct from residents (e.g. a soft dashed aura). Standing spots (already
  built) seat guests apart from residents at shared places.
- Tests: a linked visitor gets a square seat + a distinct payload entry;
  wander stays on civic ground; an offline visitor (link dropped) fades from
  the map within a TTL.

## Phase 2 — the guest is aware, and its parent nudges the host

Close the loop so the guest knows where it is and its parent drives it.

- **Awareness streamed down.** Extend the host→sender protocol: on each beat
  (and on any move) the host sends the guest a small `here` frame — the village
  name, its current place + nearby places, and who is around. The sender writes
  it onto the being so its **tick prompt** carries "You are visiting
  <village>. You are near the Library; the Meadow and Square are close. Ada and
  Bela are here." Its journal/acts can now reflect the visit honestly.
- **Parent nudges the host.** The guest's parent, on the Visit tab / its FPV,
  sees the HOST village (a new `village_map` link op proxies the host's
  `village_map_payload` — including the guest's own spot — down to the sender's
  FD, which serves it to the parent). A nudge sends a `go` frame UP the link
  with a target host place id; the host validates + walks the guest there
  (host-authoritative `depart`), and the new position streams back.
- **Enter the host village in first person.** The visiting parent can Enter the
  FPV of the host village (rendered from the proxied map), walking it as a
  ghost beside their own guest — reusing the whole FPV overlay against the
  proxied payload.
- Tests: a `here` frame lands "visiting <village>" in the tick prompt; a `go`
  op walks the guest to a real host place and refuses an unknown one; the
  proxied `village_map` carries the guest's position.

## Phase 3 — mutual sensing and proximity communication

Make the guest a real neighbour.

- **Locals sense the guest.** `_co_present`/`encounters` include co-located
  visitors: a resident settled at the guest's place gets a `crossed_paths`
  ("you met Kesh, visiting from Willowmere") with a gossip line pulled from the
  guest's streamed latest-thought. A contact grows; it's deduped once/pair/day
  like resident meetings.
- **The guest senses locals.** The host streams the co-present residents down;
  the guest records a `presence`/`crossed_paths` percept for its next tick, so
  the meeting is mutual and honest on both ledgers.
- **Speak.** Pen-pal letters already cross by name (both directions). With
  proximity now real, a resident or the guest can address someone they just
  met. (Ghost-to-ghost presence in the FPV already renders other roamers; this
  adds being-to-being co-presence.)
- Tests: a resident + co-located guest each get one `crossed_paths`/day; a
  fevered/egg guest never triggers; the guest's percept reaches its tick.

---

## Cross-cutting

- **All $0 at the tick level.** Positions, wander, sensing, and awareness are
  event rows + streamed frames — no extra LLM calls. The guest's normal tick
  (already metered on its home machine) simply gains visit context.
- **Consent & privacy.** Visiting IS the consent (already the rule for file
  proxying). The guest streams only its public profile + position; no wallet,
  no home coordinates. The host never learns the guest's private village
  layout.
- **Offline is honest.** When the link drops, the guest fades from the host map
  within a TTL (mirrors the ghost roster), and its parent's nudge refuses
  loudly ("your link to the village is down") — the existing failure mode.
- **Single-process assumption** holds (federation registries are in-memory, the
  beings loop is single-process) — same constraint the current link relies on.

## As-built (all three phases shipped, host-authoritative)

- **§1** `being_visitors` gained `location` (mirrors a resident's walk JSON) +
  `moved_at`; a new guest is seated at the square, an idle one strolls civic
  ground (`wander_visitors`, $0, in the beings loop), arrivals settle
  (`settle_visitors`). `village_map_payload` appends live guests
  (`visitors_on_map`, `kind:"visitor"`, `from` origin label) and re-seats all
  parked entries together (`_seat_parked`) so a guest never shares a resident's
  pixel. `live_visitors` uses a 1-min TTL — a guest fades within a minute of
  its link going quiet (a real guest beats every 15s). Frontend: the 2D map
  (IsoScene) and the FPV draw a guest with a sky-blue aura + "✦ visiting" /
  "visiting from <origin>" tag; guests get no cottage. **Live-verified**: a
  seeded guest renders at the Garden with the visiting marker.
- **§2** `visitor_here` (village/place/near/others) streams down each beat as a
  `here` frame → the sender writes `visit_context` on the being → the tick
  prompt gains a "YOU ARE VISITING <village>" block. `nudge_visitor` (host
  plots the course on its own grid) is driven by a `go` frame up the link;
  `village_map` is proxied down by a `pull` frame. Routes: `GET
  /beings/{slug}/visit/map`, `POST /beings/{slug}/visit/nudge`. Frontend: the
  Visit tab renders the host village (reused IsoScene) with per-place "Walk
  here" buttons + the guest's current spot.
- **§3** `encounters`/`reflex_encounters` now also cross residents with
  co-located guests (`_visitors_present` + `_meet_visitor`: one-sided ledger,
  deduped once/pair/day via the shared contact row, gossip from the guest's
  streamed latest-thought); egg/paused guests never trigger. The guest feels it
  too — the sender records a `crossed_paths` for each new neighbour a `here`
  frame brings, surfacing as a percept on its next tick.
- Tests: +13 in `test_being_federation.py` (34 total). A latent bug caught in
  review: `being_world` had no module-level `_utcnow`/`BeingError` — added, and
  the nudge test now asserts `BeingError` (not bare `Exception`) so a NameError
  can't masquerade as a clean refusal again.

## Deferred (built later or explicitly not in v1)

- **Enter the host village in first person** (walk it as a ghost beside the
  guest): needs a new FPV api-wiring mode with proxied file reads — the Visit
  tab shows the host map today, but FPV-enter of a *visited* village is a
  follow-up.


- Option B mirror pathing (guest A*-walks the foreign grid from home).
- A guest trading/earning/among the host market (economy stays home-village).
- Multiple simultaneous host villages for one guest (one visit at a time).
- Guest procreation/self-mod while visiting (life-events stay home).
