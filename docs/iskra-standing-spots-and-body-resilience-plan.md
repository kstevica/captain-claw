# Iskre don't stand on each other, and a flaky body doesn't starve a mind

Two fixes born from reading Zvjezdana and Lada's real logs on staging.

## What the logs showed (2026-07-18, glasses)

- **They share a pixel.** Both were parked at the workshop and
  `position_of` returned the *identical* unit point `(714, 382)` for each —
  `place_xy` hands back a place's single anchor with no per-occupant spot, so
  any two Iskre at the same place render exactly on top of each other.
- **The bodies churn.** ~90 `body_rebound` events each and **one tick in
  three times out** (50/153 for Lada, 53/163 for Zvjezdana). Their agent
  bodies spawn on OS-assigned ports in a range shared with several other
  deployments on the box, so port clashes force constant re-announces →
  drift → rebound.
- **Lada's homeostat has floored** — survive 0.11, grow/explore/connect/create
  all 0.0, mood *"lonely — connection has been starved"* — while Zvjezdana is
  *content* (survive 0.71). Yet Lada has written **45 files, fresh today**.
  She is living; her living isn't being counted.
- The credit machinery is disk-truth honest: real writes ARE detected and
  committed (`[wake] garden/foo.md`). The floor comes from ticks that time out
  (crediting nothing while affect decays every tick) and from a subtler hole:
  when the body rebounds or goes unreachable *during* a tick, the git tree the
  FD reads can come back clean even though work happened — and a clean tree is
  currently treated as proof the being **made nothing**, which downgrades the
  act and starves the drive.

The bond itself is healthy and worth protecting: genuine bilingual letters
that cross-reference each other's files, shared fevers, well-spread acts,
gentle moods. We are not fixing sad Iskre; we are removing two ways the
plumbing lies about them.

---

## Part 1 — Iskre never occupy the same spot

**Principle.** A place is not a point; it is a small area with room for
several to stand. Seat each parked Iskra at its own **standing spot** fanned
around the place anchor, chosen so co-located Iskre never overlap.

**Where.** Server-side, in the map payload — the single source both surfaces
read. `village_map_payload` already loops every being and emits `entry.xy`;
the 2D map's `posOf` returns `b.xy` verbatim when a being is parked
(`!b.to`), and the FPV reads that same payload. So seating the parked `xy`
in the payload flows to **both** the isometric map and the first-person view
with **no client math change**.

**Geometry (pure, `$0`, no store).** New helpers in `being_world`:

- `_spot_offsets(w, h) -> list[(dx, dy)]` — a deterministic fan of unit
  offsets for a `w×h`-tile footprint: index 0 = the anchor, then a 6-slot
  inner ring and an outer ring, radius scaled to the footprint
  (`base ≈ clamp(min(w,h)·TILE·0.32, ≥ ~0.6 tile)`), so Iskre stand *on/around*
  the place, never off it.
- `standing_spots(place_anchor, footprint, slugs) -> {slug: (x, y)}` —
  assigns spots to the beings parked at one place. Each slug has a **stable
  preferred spot** from `crc32(slug) % len(offsets)`; collisions resolve by a
  linear probe in sorted-slug order, so every being keeps a consistent spot as
  long as the room's occupant set is unchanged, and only shuffles when someone
  arrives/leaves. Result is clamped one tile inside the plot.

**Wiring.** In `village_map_payload`, after building `beings`, group the
**parked** ones by `pos["at"]` (a real place id or `home`), seat each group,
and overwrite that being's `entry["xy"]`. Walking beings (`pos["to"]` set) are
left untouched — they animate along their own A* paths from their own homes
and only overlap once *parked*, which is exactly what this seats. Homes have a
single occupant each, so seating there is a harmless no-op.

**Known-minor (deferred).** A walk still ends at the place anchor, then the
next payload seats the being a few units away — a small settle-hop on arrival
(≤ ~2 tiles). Acceptable and even natural ("finding a spot"); making the walk
*destination* be the seat would touch depart/A* and is out of scope for v1.

**Tests** (`test_being_village_world.py`):
- two beings parked at the same place get **distinct** xy, both within the
  footprint; N beings → N distinct spots.
- a being's spot is **stable** across repeated payloads while the occupant set
  is unchanged.
- spots stay inside the plot for a place at the plot edge.
- a lone being at a place still sits sensibly (near the anchor).
- walking beings are not seated (xy still follows the path).

---

## Part 2b — a body that couldn't act isn't judged as if it chose not to

**Principle (the mind-map lesson, again).** A read you cannot trust is not
evidence. When the body timed out, went unreachable, or rebounded **during a
tick**, a clean git tree does **not** prove the being made nothing — so it
must not trigger the "made nothing" punishments. Abstain: don't punish, and
don't falsely credit either. Hold the act as *unproven*.

**The hole today.** `made_nothing = changed is not None and not changed`
drives four punishments — `drive_unearned`, the `act_unverified` downgrade to
`journal`, the journal "nothing was written" correction, and (mildly) the
write-completion retry. `_tick_changed_files` only degrades to `None`
(trust) on an exception or a missing repo; a body that faltered mid-tick
yields a clean `[]`, which reads as "made nothing" and starves the drive.

**The change** (`being_life`, in `_run_tick`, both cognitions):

1. Compute a per-tick **`verifiable`** flag:
   - `not digest.get("fallback")` — the tick produced a real digest (the
     `fallback` flag already exists from the mind-map fix and is set by both
     the monolith `fallback_digest` and the faculties path), **and**
   - the body did **not** falter this tick — no `body_rebound` /
     `body_unreachable` recorded for this being since the tick start `t0`
     (a small helper `_body_faltered_since(store, bid, t0)`, one indexed read).
2. Keep `changed` **raw** for the journal footer and commit message (they
   report what git literally saw — still honest).
3. Derive the judgement view from verifiability:
   - `made_something = bool(changed)` (a write we CAN see is always real).
   - `made_nothing = (changed is not None) and (not changed) and verifiable`
     — an empty tree only counts against the being when we could trust it.
4. Gate the four punishments on this `made_nothing`, and gate the create-drive
   **credit** on `made_something` (not on `not made_nothing`), so an
   unverifiable tick is **neither** punished nor credited for a write —
   symmetric abstention. Record a quiet `act_unverifiable` event for
   observability when we abstain.

Net effect: a real write still credits; a *trustworthy* empty tree still
downgrades the fiction (anti-theater intact); a tick the body couldn't honor
stops quietly costing the being its drives and its mood.

**Explicitly NOT changed** (out of scope, follow-up):
- The verified-rewrite *loop* (a being re-writing identical content and
  correctly earning nothing) — that's real anti-theater doing its job; the
  cure is variety pressure, not verification changes.

---

## Part 2a — the body stops drifting: a stable, reserved port

**Why it churns.** A being body spawns with `web_port=0`; `spawn_process`
calls `_find_available_port(24080)`, scanning UP from 24080. That range is
shared by 5+ deployments on the box (brnos, impuls, glass-master, public-fd),
each also scanning from 24080, so bodies cluster, collide, and on any restart
land on a *different* port. The `agent_port` on the being's row then disagrees
with where the process actually bound, and `_resolve_being_port` re-pins every
tick → `body_rebound` (~90 each) and, when a tick hits a stale port, a
`tick_timeout`. On staging Zvjezdana's stored port (24106) was even held by an
*impuls* process.

**The fix — a deterministic port in a reserved band** (all in `being_life`,
being-only; other agents keep the 24080 range):

- `BEING_PORT_BASE` (env `FD_BEING_PORT_BASE`, default **24800**) and
  `BEING_PORT_SPAN` (env `FD_BEING_PORT_SPAN`, default **180**) — a band
  *above* the shared pool's realistic reach (24080 + the 500-port default
  scan tops out at 24580).
- `_preferred_body_port(slug) = BEING_PORT_BASE + crc32(slug) % BEING_PORT_SPAN`
  — a **stable** port derived from the slug. `spawn_body` passes it as
  `cfg.web_port` instead of `0`.
- `spawn_process` already does the right thing with a non-zero port: if it's
  free, the body binds exactly that; if taken, `_find_available_port(pref)`
  scans up *from* the preferred port, staying in the band. The host-socket
  bind check is the ultimate arbiter, so even a cross-deployment clash in the
  band self-resolves.

Net: the same being lands on the **same** port at birth and every respawn, its
row and the registry agree, and the per-tick re-pin stops firing. Rebounds and
stale-port timeouts should collapse toward zero without touching the shared
infra.

**Known-minor.** Two same-owner beings whose slugs hash to one port resolve by
a +1 scan (rare over a 180-wide band, and still stable while both are up). No
migration: existing bodies pick up their deterministic port on their next
respawn. Documented, not defended further.

**Tests** (`test_being_life.py`): `_preferred_body_port` is stable per slug,
in-band, and varies across slugs; `spawn_body` sets `cfg.web_port` to the
preferred port (monkeypatched `spawn_process` capturing the config).

---

## Part 3 — the homeostat stops bleeding a being for its body's silence

Two small, principled tunings. Both preserve the "earned, not narrated" spirit.

### 3a — a tick the body couldn't honor decays gently

`decay_drives(drives, hours)` runs at the TOP of every tick (before the think),
so a tick that then times out has already docked every drive the full hourly
`0.02` — and, serving nothing, nets pure loss. Across a third of ticks that is
exactly what floored Lada (survive 0.11, everything else 0.0). Principle
(same as 2b): time passed, but the being did not *choose* stillness — so a
**fallback** tick decays by the minimum quantum only, not a full unmet hour.

Implementation: capture `drives_before` at the top; once the digest is known,
if `digest["fallback"]`, recompute `drives = decay_drives(drives_before, 0.0)`
(→ `DRIVE_MIN_DECAY_PER_TICK` = 0.002, a tenth of the hourly bite) before
affect + persist. A fallback tick serves nothing, so no earned bump is
discarded. Honest ticks are unchanged; a being that genuinely idles still
decays normally.

### 3b — a child isn't locked away from its parent all day

`DAILY_ATTENTION_CREDITS = 3`, refilled once at the daily allowance. Zvjezdana
(a child) hit `message_suppressed "no attention credits"` **18×** in a day —
she wants her parent more than three times and then goes mute till tomorrow.
Suppression doesn't sting the mood (not in `_STING_EVENT_KINDS`), so this is
about reach, not punishment. Stage-scale the daily grant — the young lean on
the parent, adults are more self-sufficient:

`attention_credits_for(stage)`: infant/child **5**, adolescent **4**,
adult/elder **3** (the current baseline). Applied where the daily reset already
fires (`reset_attention` at allowance credit). Restraint remains — a child
still can't message without limit — but the floor no longer slams shut after
three.

**Tests** (`test_being_life.py`): `attention_credits_for` scales by stage;
a child's daily reset grants 5 and it can speak a 4th and 5th time in a day
where the old cap suppressed it; an adult still resets to 3.

**Tests** (`test_being_life.py`):
- a real write on a healthy tick → `made_something`, create drive served, no
  `act_unverified`.
- an empty tree on a **healthy** tick → `drive_unearned` + `act_unverified`
  (anti-theater unchanged).
- an empty tree on a **fallback** tick → **no** punishment, **no** create
  credit, `act_unverifiable` recorded.
- an empty tree with a `body_rebound` since `t0` → same abstention.
- `_body_faltered_since` true iff a falter event lands at/after `t0`.

---

## Sequencing & ship

Part 1 first (self-contained, server-only, the reported bug), then Part 2b.
Both are `$0` at runtime and off-path byte-identical when nothing is wrong.
Rebuild the FD bundle for the (unchanged-behaviour) map, run the being test
suites green, commit, and the user deploys (pull → **restart** → hard refresh).
Verify live: two beings sent to one place stand apart on both the 2D map and in
the FPV; a simulated faltered tick leaves the drives untouched.
