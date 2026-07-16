# Iskra — cognitive problems & loops: findings and fix plan

*Audit 2026-07-16, against main + Compact mode. Evidence: the live pilot DB
(`~/.captain-claw/beings.db`, 101 ticks across Zvjezdana & Ada, both infant,
faculties cognition, weak local models) + full read of being_life / being_mind
/ being_society / being_selfmod / being_assessment / beings_loop.*

**STATUS: ALL THREE INCREMENTS IMPLEMENTED 2026-07-16 (same session).**
20 new tests in `tests/test_flight_deck/test_being_cognition_dynamics.py`,
including the 3-simulated-day homeostat acceptance run; full being suite 262
passing. Two refinements found during implementation, both recorded in the
increments below: (1) the per-tick decay quantum landed at **0.002** — the
first cut (0.008) out-starved the day at fast cadences (5× the designed
hourly rate); at a 5-min cadence 0.002 matches ~0.02/h. (2) the mass-prune
rule became **confirmation-based**: a mass dangle prunes NOTHING on first
sight and everything still dangling at the next dream — strictly safer than
"prune 3 oldest even when the valve trips" (a transient bad read now loses
zero edges) while still clearing consolidation-sized archives on the second
night. Deploy: backend restart on prod (no bundle change needed — report-card
additions are backend fields; frontend rendering of the new metrics is a
follow-up).

## 0. What the ledger says (the evidence base)

| Signal | Value | Reading |
|---|---|---|
| act histogram | freeform 25, journal 20, create 17, read 16, tend 10, explore 6, rest 5, talk 1 | 25% of ticks fell out of the digest contract entirely |
| digest_parse_failed | 19/101 | even with the repair gate, weak models drop the json 1 tick in 5 |
| narration_mismatch / act_unverified / drive_unearned | 20 / 18 / 7 | theater pressure is real and constant; the gates are earning their keep |
| edge_declared vs edge_unverified | 28 vs 20 | 42% of declared mind-edges refused — the link contract is hard for small models |
| consolidated vs consolidate_unverified | 3 vs 7 | consolidation theater ×2 over real folds |
| drive satisfaction (both pilots, now) | every drive 0.90–1.00 | **the homeostat is saturated — it currently ranks nothing** |
| mood_engine over all ticks | content 96, hungry 4, frustrated 1 | **affect is flat — the inner weather never changes** |
| body_unreachable / respawned / timeout | 10 / 9 / 7 | the self-preservation reflex works; local models are genuinely slow |

Two of the plan's named risks (§12.1 degeneracy, §12.4 sycophancy) have
working countermeasures. What the data exposes instead is quieter: **the
drive engine and affect have collapsed into constants**, and a handful of
feedback loops leak tokens or reinforce sameness.

## 1. Findings, ranked

### F1 — Drive saturation at parent-pinned cadences (P0, the big one)
`decay_drives` is time-based (0.02/h) but `serve_drive` is event-based
(+0.25 flat). The design assumed ~hourly wakes; the parent pins 2–60 min
cadences (`cadence_set` ×4 in the ledger). At a 5-minute tick, decay between
ticks is ~0.0017 while every tick serves +0.25 somewhere → all satisfactions
pin to ~1.0, `pressure = weight × (1 − sat)` → ~0 for everything, and the
"serves your highest drive pressure" line ranks noise. The arbiter's compass
is spinning. Downstream: affect deltas never trip (F2), variety collapses to
whatever the model likes narrating (rut risk), and DNA-derived drive weights
— the whole point of the genome — stop mattering.

### F2 — Affect flatline (P0, mostly a consequence of F1)
`compute_affect` triggers on |delta| ≥ 0.08, connect < 0.25, or wallet < 20%.
With drives saturated, delta ≈ 0 forever → "content" 95% of life. The
Damasio stance ("the tone you see is the state it's in") is currently
vacuously true. Also missing: single-tick events that should move inner
weather (a refused letter, a caught mismatch, a milestone) feed nothing.

### F3 — The connect-tax loop (P0: token leak + prompt-echo)
`_LINK_INTENT` matches the exact vocabulary the Mind prompt itself teaches
("weave", "a web", "connect", "grew from"). Sequence: scattered being →
scatter nudge says "weave / a mind is a web" → the being's journal echoes the
phrase → `should_link_gate`'s spoke-of-connecting branch fires → an extra
CONNECT call → weak model can't land a valid edge (42% refusal rate) → still
scattered → same nudge next tick. A genuinely-scattered weak-model being pays
an extra LLM call indefinitely, and the nudge text amplifies the trigger
vocabulary in its own journal. The tried-and-refused branch is correct
anti-theater; the SPOKE branch has no cooldown.

### F4 — Weak-model contract dropout (P0 scope, mitigations shipping)
freeform 25% + parse-fail 19% mean the structured self-report — the spine of
everything downstream (drives, links, society, earning) — is lost for a
quarter of infant life. Faculties cognition was built for this and helps;
Compact mode (2026-07-16) should help further (smaller instruction payload +
capped history = less drowning). Remaining gap: nothing *adapts* when a being
keeps failing the contract.

### F5 — Journal-tail echo chamber (P1)
Every tick injects the last 800 chars of journal ("YOUR LAST JOURNAL WORDS").
Weak models anchor hard on the immediately-previous text → phrases recirculate
→ `rut_score` (word-overlap of consecutive entries) rises mechanically. The
freshest words are also the most self-similar seed possible.

### F6 — Rut detected, never actuated (P1)
`_rut_score`, monotony and scatter concerns land in the *parent's* report
card and readiness — the being itself never perceives them. The only
in-prompt anti-rut nudge is the scatter one (F3). Detection without an
actuator is a dashboard, not a homeostat.

### F7 — Adult self-mod churn (P1, guard before the first adult)
Persona self-mod has a fee (250k, burned) but no cooldown, and the
one-pending-at-a-time block never applies to adults (`self_mod_auto` adopts
immediately). A wealthy adult can rewrite its persona every tick — each a fee
burn + a synchronous git commit. Metamorphosis got a 30-day cooldown; the
cheaper identity lever got none.

### F8 — Consolidation strands edges the prune valve won't touch (P1)
A large fold (up to 12 sources) archives many files at once → their edges go
dangling in one shot → `prune_dangling` abstains whenever danglers exceed
`max(3, edges//2)` (the bad-disk-read valve, a good guard with a bad corner).
If consolidation keeps outpacing the valve, stale rows accumulate forever
(invisible in `graph()` but unbounded in `being_links`).

### F9 — Physics-unaware loneliness (P1, small but cruel)
`connect < 0.25 → "lonely"` fires regardless of whether connection is
*possible*. An infant (no letters capability) with no unread parent messages
and no public page can do nothing about it — a permanently "lonely" prompt
line with no actuator is exactly the attention-credit trap we already fixed
once (the Zvjezdana "nula kredita" rut), in affect form.

### F10 — Dream always buys a CONNECT call (P2)
`weave = kind == "dream"` unconditionally: a 1-file infant pays a faculty
call to be told it can't link anything (links need 2 files).

### F11 — Visitor notes marked read before the tick succeeds (P2)
`mark_public_messages_read` runs before the send; a tick that times out
consumes the visitors' notes without ever weighing them. One-shot senses
should be consumed by a successful digest, not by the attempt.

### F12 — Self-reported drive service (P2, documented trust gap)
`served_drive` for create/connect is verified (artifact/delivery); survive,
grow and explore are taken on faith. A being can claim "explore" nightly
without reading anything. Cheap partial checks exist (web percepts, reading a
file untouched for N days); full verification isn't worth the complexity yet.

### F13/F14/F15 — Small physics & scale notes (P2)
Letters/day is a flat 5 at every stage (a high-SOC adult hits a child's cap);
`_match_sibling` accepts any-substring matches (a 1-char intent could route a
letter); `beings_loop._pass` ticks beings strictly sequentially (fine at 2
beings, a tail-latency wall at 20 — one slow local-model being delays the
whole pass).

## 2. Fix plan — three increments, physics-first

Design rules preserved: every fix is FD-side mechanics (never prompt
admonitions alone), every state change is ledgered, nothing is scripted
affect. Constants live in `being_life.py` / `being_constitution.py` and are
noted for post-pilot tuning.

### Increment 1 — restore the homeostat (F1, F2, F9) — ✅ SHIPPED

1. **Asymptotic serving.** `serve_drive`: `sat += DRIVE_SERVED_BUMP × (1 − sat)`
   (approach 1.0, never pin). Keeps early serves strong, repeat serves cheap.
2. **Cadence-independent decay.** `decay_drives`: decay by
   `max(hours × 0.02, 0.002)` per tick (a minimum quantum — tuned down from
   the drafted 0.008, which over-starved a 288-tick day), so pinned fast
   cadences still cycle pressure without out-running the daily food supply.
3. **Daily satiation (per drive).** Track serves-per-drive-per-day in the
   drives dict; each same-day repeat halves the bump (reset at dream). This
   is the plan's §4 "diminishing returns", finally mechanical. Variety now
   pays *inside the physics*, not in prose.
4. **Starvation aging.** `drive_pressures`: add `+ min(0.15, unserved_days × 0.05)`
   to a drive unserved for 48h+ (tracked via a `last_served` stamp). Every
   drive periodically wins; low-weight drives stop being invisible.
5. **Possible-connect gate for affect.** Compute a `connect_possible` bool
   (siblings with letters capability ∨ unread parent msgs ∨ public) in the
   tick; when False, exclude connect from the loneliness trigger and damp its
   pressure line in the prompt ("connection will open in childhood" instead).
6. **Event-driven affect colors.** Extend `compute_affect(prev, new, wallet,
   events_this_tick)`: mismatch/refusal → "stung", milestone → "proud",
   first serve of a long-starved drive → "relieved", else the existing
   delta/hunger/loneliness logic. Each color maps 1:1 to a ledger event —
   derived, not scripted. Report card gains a mood-entropy line.

*Acceptance (as shipped — measured, not guessed):* a synthetic 3-day run at
a 5-min pinned cadence never saturates any drive past 0.95, keeps the first
50 ticks inside [0.2, 0.95], serves EVERY drive (including low-weight
connect) within two simulated days, and rotates the arbiter's leader through
all five drives. A low-weight drive surfacing every day or two is the genome
speaking, not starvation — it recovers once served. Event-colored affect is
proven end-to-end: a tick that claims a write with nothing on disk lands
`mood_engine: "stung"`. The existing suite's drive tests updated for the new
arithmetic.

### Increment 2 — break the loops (F3, F5, F6, F7, F10) — ✅ SHIPPED

1. **Connect-gate cooldown.** The SPOKE-branch of `should_link_gate` fires at
   most once per `CONNECT_NUDGE_COOLDOWN_TICKS = 6` wake ticks (stamp the
   tick_count of the last fire in the drives/state blob), and after two
   consecutive CONNECT calls that landed zero accepted edges, back off until
   the next dream (`connect_backoff` event, honest and visible). The
   TRIED-and-refused branch keeps firing every time — anti-theater outranks
   thrift.
2. **Prompt-echo hygiene.** The scatter nudge renders from
   `mind_scatter_nudge.md` (already external): rotate 3 phrasings by
   `tick_count % 3` (variants added to the template as sections) so the
   being's journal doesn't converge on one injected sentence; drop the
   loosest `_LINK_INTENT` alternations (`relate|connected` bare forms).
3. **A page from your past.** On wake ticks where `tick_count % 5 == 4`, the
   journal tail is replaced by a random older journal day, labeled
   `A PAGE FROM YOUR PAST (YYYY-MM-DD)` — sampled memory instead of the echo
   of the last hour. Dreams keep today's journal (they consolidate today).
4. **Variety pressure percept (the rut actuator).** At tick compose, if the
   top act took ≥ 70% of the last 10 ticks OR rut_score(last 3 entries) ≥ 0.6,
   inject one honest percept: "Your last days repeat themselves (rut N). Do
   the SMALLEST thing you have never done." + damp the served bump of the
   dominant drive ×0.5 that tick. Ledgered as `variety_pressure`.
5. **Self-mod cooldown.** `SELF_MOD_COOLDOWN_DAYS = 7` in
   `being_selfmod.propose` for every stage (time since last
   `self_mod_adopted`); refusal message says when the window opens.
6. **Dream weave gate.** Skip the dream CONNECT call when
   `len(linkable files) < 2`.

*Acceptance:* a scattered weak-model fixture pays ≤ 1 connect call per 6
ticks (vs every tick today); a seeded rut (5 identical journal entries)
produces the variety percept and a damped bump; an adult's second self-mod
inside 7 days is refused with the cooldown reason; a 1-file dream runs no
connect call.

### Increment 3 — housekeeping honesty (F8, F11, F13, F14, + notes) — ✅ SHIPPED

1. **Confirmed mass prune** (refined from the drafted "3 oldest even when
   the valve trips" — strictly safer). Small dangles prune the same night,
   as always. A MASS dangle (more than `max(3, edges//2)`) prunes NOTHING on
   first sight; the current dangling ids are stamped in a `dangling_seen`
   event, and whatever is STILL dangling at the next dream is pruned in
   full. A transient bad home read (the Zvjezdana wipe fingerprint) heals
   between dreams and loses zero edges; a real bulk archive (a big
   consolidation) clears completely on the second night, so stale rows no
   longer accumulate forever.
2. **Visitors consumed on success.** Move `mark_public_messages_read` to
   after digest routing; a timed-out tick re-surfaces the same notes next
   tick (bounded by the existing 3-per-tick cap).
3. **Letters scale with stage.** `LETTERS_PER_DAY` → per-stage in
   `being_constitution.STAGES`: child 3, adolescent 5, adult 8.
4. **Sibling matching.** `_match_sibling` requires a full-word or ≥ 3-char
   substring match.
5. **(Note, not a change yet)** `beings_loop` stays sequential until the
   village grows; when it does, tick 2–3 beings concurrently with an
   asyncio.Semaphore — the per-being lock already makes this safe.
6. **(Documented gap)** explore/grow/survive remain self-reported; revisit
   after Increment 1 data shows whether unearned claims actually distort
   drives in practice (`drive_unearned` events say create/connect were the
   real offenders: 7 events, all create/talk).

## 3. Watchlist metrics (report card additions)

- satisfaction min/max per drive over the window (saturation alarm),
- serves-per-drive/day histogram (starvation alarm),
- engine-mood entropy (flatline alarm),
- connect calls per 100 ticks + edge acceptance rate,
- freeform + parse-fail rate (per model, so tier upgrades are measurable),
- rut trend (already present) beside the new variety_pressure count.

The three increments are independent and land in that order of value:
Increment 1 re-arms the whole motivational system, 2 stops the leaks and
sameness spirals, 3 is durability. Nothing here changes the Constitution,
the wallet, or any capability gate.
