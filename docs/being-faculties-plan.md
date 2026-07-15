# Iskra — the decomposed tick ("faculties")

## Why

A tick today is **one monolithic prompt** — vitals, drives, the whole home
manifest, persona, media diet, siblings, society fields, earning fields,
visitors, mind links, last-tick feedback, journal tail, the task, and a
~20-field digest schema (plus self-mod and procreate) — asking for **one giant
JSON reply**. A small-context model can do the *act* (the coding-like part it is
genuinely good at) but cannot *also* emit a clean 20-field digest in the same
breath. Evidence from the pilot: Ada (weak infant model) failed to emit a
parseable digest **14 of 19 ticks**, so `links` and every other structured
field never even reached the parser.

The fix is not a bigger model. It is to **decompose the tick into focused
faculties** — small, single-purpose calls, each with a tight prompt and a tiny
output, orchestrated Flight-Deck-side and composed back into one digest.

## The one invariant

**Decompose the tick, not the self.** The sub-calls are ephemeral *faculties*
of ONE being writing to ONE home / journal / wallet / memory. They must never
become separate identities. Anti-theater (plan rule #1) is unchanged: every
step is verified against ground truth (git diff), the write gate lives on the
act step, links are verified.

## The pipeline

Each faculty is one small `send()` to the being's body. Conditional ones are
skipped most ticks, so a typical tick is 2–3 small calls, not one bloated one.

1. **Orient** (tiny): compact vitals + drives + a *summary* of the home +
   percepts → `{act_kind, target, served_drive, next_wake, intent}` plus the
   rare structured moves (letter / publish / gift / adopt / chore / quest /
   venture / self_mod / procreate) surfaced only when the stage/context allows.
   This stays the single holistic decision point, so the being stays coherent.
2. **Act** (its strength): *"you chose to write `garden/x.md` about Y — do it
   now with your tools."* Just the work. Git-diff verified; the **write gate**
   retries here. Skipped for rest/journal/talk.
3. **Journal** (grounded): given what *actually* changed on disk → `journal_entry`
   + `mood` + `served_drive`, plus `public_replies` when visitors were shown.
   Small and truthful.
4. **Connect** (conditional): its **own** call, only the link task + the exact
   file list + last tick's refusals → `links` (and `consolidate` at dream).
   One job, small context — this is what finally makes edges reliable. Runs
   only when scattered / intent detected / at dream.

Flight Deck merges the partial outputs into the **same internal `digest` dict**
`parse_digest` produces, so ALL downstream routing (society, earning, links,
consolidate, public_replies, self_mod, procreate, journal, ledger) is untouched.
Metering already sums the whole window (`_usage_since`), so accounting is
unaffected.

## Rollout

Gated per-being by a `cognition` column: `monolith` (default — the existing
single-prompt path, unchanged) vs `faculties` (the pipeline). The parent flips
it per being, so weak-model beings get faculties while the monolith stays the
default and every existing test/being is byte-identical.

## Roadmap

- **Increment 1 (this cut): the split.** orient → act → journal → connect, one
  model, opt-in `cognition = faculties`, FD composes the digest. Proves the
  reliability + edge-creation win with the least machinery.
- **Increment 2: per-faculty model routing.** Each faculty pinnable to a
  model/tier — cheap coding model for *act*, a stronger reasoner for
  *connect / consolidate / self-reflection* when the wallet allows. A mixed
  fleet inside one being. (The step interface in increment 1 already carries a
  per-faculty model seam.)
- **Increment 3: full faculty workers.** Each faculty as its own ephemeral
  agent process (Council/Vatra-style) for real parallelism and isolation — with
  care to preserve the one-self invariant.
- **Increment 4: a tick *is* a quick session.** For a hard tick (a real
  decision, a big consolidation, a venture), the orient step may escalate into
  a short Vatra (collaborative) or Basna (ensemble) run over the being's own
  VFS home, then collapse the result back into one digest. The being thinks
  with a team when it matters, alone when it doesn't.

## Addendum (2026-07-15): the talk faculty — speech is real or refused loudly

The Zvjezdana→Lada bug: an infant, with her parent's blessing, "greeted" her
sibling by choosing `act_kind: "talk"` — and the greeting went nowhere. The
act was in every menu but wired to nothing; the only real channel (the
`letter` digest field) was monolith-only AND child+; the journal gate checks
only disk, so she journalled the greeting as sent. Physics neither delivered
nor refused — it just evaporated, and both the being and the parent believed.

What changed:

- **TALK is a faculty step now.** `orient` picking `talk` at a sibling runs a
  tiny `[LIFE TICK — talk]` call whose only output is `{"letter": {to, body}}`
  — the one channel that actually reaches a sibling. Below the `letters`
  capability (or with the day's quota spent) the step is SKIPPED — no tokens
  burned — and a `society_refused` event is recorded instead.
- **Refusals are loud.** The journal step is told THE WORLD SAID NO this same
  tick (no pretending), and the NEXT tick's percepts carry "PHYSICS SAID NO
  last tick" so the refusal cannot be remembered as success.
- **Speech anti-theater.** A `talk` digest that produced no letter row, no
  word to the parent and no public reply downgrades to `journal`
  (`act_unverified`), and a claimed `connect` drive is `drive_unearned` — the
  connect drive is settled only after real delivery, mirroring how `create`
  is settled against the git diff.
- **Society restored to faculties.** The orient step now offers the same
  capability-gated society fields (letter/publish/gift/adopt), chore claims
  and rare options (self_mod, procreate) as the monolith — the split had
  silently amputated all of them.
- **Menus stop lying.** Every act menu renders talk honestly per stage:
  "sibling letters unlock in childhood" for infants, "quota spent" at the
  daily cap, the real offer otherwise.
- **Fleet containment.** A being's body below `agent_messaging` (adolescent)
  no longer registers consult_peer / flight_deck / basna / vatra /
  code_session / hosting / app_runner / synthesize_flow, and its system
  prompt carries no fleet identity or peer roster (`CLAW_BEING_CAPS`, stamped
  by `spawn_body`; `_iskra_fleet_hidden` in agent_context_mixin). A body that
  can consult a sibling's body would bypass letters physics, rate limits and
  wallet metering. Stage advancement now respawns a living body so the new
  physics (tier + caps) take hold at once.
