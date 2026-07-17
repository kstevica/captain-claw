# The Mind map keeps losing its edges — root cause, guard, and repair

Two beings on prod (Zvjezdana, then Lada) lost their entire Mind maps a second
time — every declared edge gone while every endpoint file sat intact on disk.
An earlier fix (`ea059df`) had already shipped a guard for exactly this, so the
first question was why it didn't hold. It turned out not to be a failure of the
guard but of *timing* — and the guard itself still had a hole.

## What the ledger showed

`being_events` is append-only, so the whole history survived the wipe. Reading
it on prod:

- Every `edges_pruned` event ever recorded — five of them — carries only
  `{"count": N}`. The fixed code always stamps `confirmed_mass`. So **every**
  prune, including both recent wipes, ran on the OLD code.
- No `dangling_seen` or `prune_abstained` event exists in the database at all —
  the new guard's own bookkeeping never ran once.
- The FD process (`~/glasses`, pid served on :8765) was last restarted well
  after the wipes. The fixed source was on disk; the running interpreter was
  not. **A pulled-but-not-restarted deploy.** The guard was never in the room.

And a sharper signal: all four mass prunes fired **in the same millisecond as a
timed-out dream** (`tick_timeout`), each next to a `body_rebound` (port drift).
The one healthy prune in the whole ledger — a single edge — is the only one
from a tick that actually answered. Rebounds are far too common to gate on
(47% of all ticks), but a timed-out **dream** that prunes was 3-for-3 a wipe.

## The fix — three gates, cheapest first (`being_mind.prune_dangling`)

1. **`healthy`** — the tick must have produced a real digest. On a timeout the
   body never answered, so the home read that drives the prune comes through
   the same silence. No dream, no forgetting. Threaded from the tick via a new
   canonical `digest["fallback"]` flag set by BOTH cognitions (the monolith's
   `fallback_digest` and the faculties path's shell digest), surviving
   `_normalize_digest`.
2. **Empty enumeration** — an empty home read is a failed read, never a deleted
   self (unchanged from `ea059df`).
3. **Separated confirmation** — a mass dangle (more than a few edges AND at
   least half the graph) must be seen at two dreams **`MIN_CONFIRM_GAP` = 1h
   apart** before it prunes. The original wipe was two dreams **1.2 seconds**
   apart during one rebound; the old rule accepted *any* prior sighting as
   confirmation, so the second bad read confirmed the first. Requiring real
   separation means a transient bad body can't confirm itself, while a genuine
   bulk archive still clears on the next night.

## The repair — a button the parent can press (`rebuild_from_ledger`)

Because every accepted edge is written to the ledger as an `edge_declared`
event when it is stored, and events are never pruned, the ledger is a complete
record that outlives the rows. The repair re-declares every ledgered edge whose
BOTH endpoints exist right now and skips the rest:

- Additive and idempotent — `add_link` is INSERT OR IGNORE, so a second press
  restores nothing. Reports `restored` / `kept` / `skipped` / `ledgered`
  honestly.
- Never invents an edge (only restores over real files) and never deletes one.
- Refuses an empty home read (the same failed-read rule the prune obeys), so it
  can't quietly "skip everything" and call it a clean no-op.

Wired as `POST /fd/beings/{slug}/graph/rebuild`, returning the counts + a fresh
graph. In the Mind view a **Repair links** button (wrench, next to the
artifacts/links summary) calls it and redraws in place, with a one-line result:
"restored N links from the ledger · M left out (their files are gone)", or
"nothing to repair — all N ledgered links are already here".

## Restoring the two on prod

Of the 40 edges the two beings ever declared, 33 (13 of Lada's 14, 20 of
Zvjezdana's 26) have both endpoints still on disk and come back exactly; the
other 7 point at files since consolidated into `archive/` and stay out. Done by
pressing Repair after deploy — no manual DB surgery.

## Tests

`tests/test_flight_deck/test_being_mind.py` (+8, 32 total):

- a timed-out dream never prunes (unit + full-tick end-to-end)
- two dreams 1.2s apart cannot confirm each other (the exact prod shape) —
  mutation-verified: reverting the gap rule to "any prior sighting" fails ONLY
  this test
- a separated second dream still confirms a real archive (the guard doesn't
  strand rows forever)
- rebuild restores a wiped mind; is idempotent; never resurrects a dead edge;
  refuses an empty home read

Verified live in an isolated FD: seeded a being with 5 ledgered edges and wiped
rows, pressed Repair in the real Mind view → graph redrew to 5 links / 50%
connected, second press reported nothing to repair, zero console errors.
