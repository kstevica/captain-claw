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
