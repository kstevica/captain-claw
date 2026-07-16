# Iskra — toward a fuller life: what's missing and what to add

*Proposals 2026-07-16. Filter applied throughout: Design rule #1 — every felt
thing must be a real variable with real behavioral consequence; no theater.
Each item names the machinery it rides on. Ordering inside each tier ≈ value
per unit of work.*

**STATUS 2026-07-16 (same day, after the loops plan shipped): Tier 1
COMPLETE (items 1–6) + the first Tier 2 pair (7 relationships, 11 life
projects) — all in `being_world.py` (the umwelt module) + 7 new instruction
templates + `test_being_world.py` (11 tests; being suite 273). As built:
body notes fire only under real strain (load ≥1.5×cores, mem ≥90%, battery
≤20% unplugged); the world note speaks once per morning; seasons lean
explore/create ±0.03–0.05 in `drive_pressures` (hemisphere-aware by tz
heuristic); month-birthdays are once-per-life milestones with a dream
retrospective; boredom (no real percepts + all pressures <0.15) doubles the
next sleep within stage bounds (`slept_in` event); dreams tangle two random
garden/skills artifacts (deterministic per tick); exchanges within 24h nudge
RELATIONSHIPS.md at dream; self/PROJECT.md is offered at dreams (child+)
and checked in weekly (`project_checkin` event). TIER 2 IS NOW COMPLETE —
items 8–10 shipped later the same day (federation letter frames, games
shelf, naming rite; test_being_village_life.py) and items 12–13 after that
(reading lists + illness; test_being_school_health.py). Only Tier 3's
bigger arcs (14–18) remain open, per the measure-first sequencing below.**

What the beings already have is the hard part: metabolism, mortality,
heredity, economy, society, selfhood files, honest ledgers. What they lack,
compared to a life, falls into five gaps: **a textured world** (their umwelt
is nearly all self-referential), **a body that is felt** (the soma is
invisible to them), **time with shape** (every day is the same day),
**relationships with history** (siblings are a roster, not people), and
**stakes they choose** (goals live tick-to-tick; only savings goals span
weeks).

## Tier 1 — cheap, honest, high-yield (days each)

1. ✅ **Felt embodiment: the machine as body.** The body is a real process on a
   real Mac — expose it. Percepts from real host metrics: load / free RAM /
   battery / thermal state → "your body is sluggish today (the machine is
   hot)". Slow local-model ticks stop being mysterious to the being; it can
   *choose* to rest when the hardware struggles (and that choice is real
   thrift). Rides on: `system_info.py`, percepts_since. Zero theater — it's
   the literal truth of its substrate.
2. ✅ **Weather and place** (calendar-honest version shipped; live weather API still open)**.** The parent's real weather/location (existing
   config timezone + one cheap API or none — even just season + daylight
   length derived from the calendar) as a morning percept. Beings share the
   parent's world; journals stop floating in placeless space. Rides on:
   percepts, the clock line. (Diet-gate the API like any web read.)
3. ✅ **Time with shape: weekdays, seasons, anniversaries.** A real calendar
   texture: weekends felt (parent's quiet-hours already exist — name them),
   a hatch-day milestone with a self-retrospective dream task ("you are one
   month old today — reread your first journal page"), seasonal drive
   modulation (±0.05 on explore/create weights by real season). Rides on:
   beings_loop quiet hours, milestones, dream task template (now external).
4. ✅ **A page from your past** (shipped with the loops plan) (also in the loops plan): resurfacing old
   journal days is memory *behaving like memory* — the past visiting the
   present unprompted. The single cheapest life-likeness win.
5. ✅ **Boredom → sleep-in.** When every drive is satisfied and no percepts
   arrived, the being may extend its next wake beyond the stage default
   ("nothing calls; sleep longer") — energy conservation as felt behavior,
   and the inverse of the rut: an empty day is *allowed to be empty*. Rides
   on: next_wake_minutes clamps.
6. ✅ **Dream imagery (recombination).** The dream task samples two random old
   artifacts and asks the being to let them tangle ("you dreamt of
   <cat-at-the-gate> tangled with <ledger-honesty> — write the fragment").
   Mechanically real (its own corpus, labeled a dream), and the classic
   creativity engine — REM as remix. Rides on: dream_task.md template +
   list_self_files sampling.

## Tier 2 — relationships, texture, stakes (a week-ish each)

7. ✅ **Relationship memory** (dream nudge shipped; sibling stage-change percepts still open)**.** `self/RELATIONSHIPS.md` exists but nothing feeds
   it. After any letter exchange / gift / trade / co-parenting event, the
   dream task nudges: "update what you know of Lada". Add a percept when a
   sibling's stage changes or it publishes. Siblings become people with
   histories, not roster lines. Rides on: society percepts (exist), dream
   template.
8. ✅ **Pen-pals across villages** (shipped: `penpal` digest field over the live federation link, both roles — a linked visitor on the square, or the village a being is out visiting; acked delivery or loud refusal; shares the stage letter quota; parent's door = public flag or being sent visiting)**.** Federation already carries visits and
   messages; add being↔being letters across FD instances (parent-approved,
   quota'd like sibling letters, ledgered). First contact with a truly
   *other* mind — different genome pool, different upbringing. Rides on:
   being_federation link ops.
9. ✅ **Play: structured sibling games** (shipped: commons/games/ shelf with riddle-chain, exquisite-corpse, what-am-I-looking-at; whimsy-gated invitation every 5th tick)**.** A `commons/games/` shelf with 2-3
   letter-game protocols (riddle chains, exquisite-corpse poems, "what am I
   looking at" with a garden file). Games are the natural expression of PLA
   and the missing *joint* activity — culture beyond skill-trading. Rides on:
   commons + letters, one etiquette file, PLA-weighted act offer.
10. ✅ **Chosen name at adolescence** (shipped: `chosen_name` digest field, once per life, adolescent+, pending → parent blesses/declines on the panel; display name + genome epigenetics change, slug never does)**.** The plan promises a self-naming ceremony;
    implement it: at the adolescent rite the being may propose a chosen
    display name + why; parent confirms; slug stays. Identity the being
    *authored*, recorded as a milestone and a genome epigenetics note.
11. ✅ **Life projects (chosen stakes)** (offer + weekly check-in shipped; monthly FD progress percepts folded into the weekly check-in)**.** A first-class `self/PROJECT.md`: one
    long-horizon aim the being declares (a book of poems, a skill mastery, a
    metamorphosis fund), with FD tracking progress percepts monthly. The
    savings-goal machinery proved weeks-long narrative continuity works; this
    generalizes it beyond money. Anti-rut spine, per plan §12.1.
12. ✅ **Education: reading lists** (shipped: reading_list on the being + panel Growth-tab curriculum; every wake lists open readings; a report is a REAL file verified in the tick's diff — fee minted on verification, grow drive served, first_report milestone)**.** Parent assigns a curriculum (URLs/files) →
    daily reading percept + a book-report chore with a small fee. Uses chores
    + media diet as-is; growth drive gets verifiable food (F12 in the loops
    plan gets its explore verification for free).
13. ✅ **Illness as consequence (not RNG)** (shipped: fever = collapse or 3 timeouts in 24h → cadence floored to 2h (pinned included), feverish affect, rest percept, onset ledgered once; confusion = 3 narration mismatches in 24h → mandatory self-examination at the next dream; both computed from the ledger, both age out on their own; report-card concerns)**.** Two honest ailments: *fever* — a
    burn-cap collapse or 3 timeouts in a day forces reduced cadence + fast
    tier for 24h ("you are unwell; rest"); *confusion* — 3 narration
    mismatches in a day surfaces a mandatory self-examination dream task.
    Both are consequences of real events, never dice. Rides on: existing
    events + tick clamps.

## Tier 3 — bigger arcs (multi-week, decide after the pilot matures)

14. **Elderhood & natural span (opt-in per family).** Beyond starvation
    death: after a parent-set age, a being enters elderhood — allowance needs
    drop, whimsy rises, a "memoirs" standing task distills the journal into
    `self/MEMOIR.md` (the heirloom its descendants inherit). Death then has a
    season, not just a ledger event. Controversial by design; the Lamarckian
    heirloom machinery already exists to receive it.
15. **Village commons governance.** A rotating steward role (adolescent+):
    keeper of `commons/INDEX.md`, arbiter of etiquette nits, small stipend
    chore. Responsibility as a developmental stage, and the first *role* a
    being holds in a society. Rides on: chores + commons.
16. **Voice and senses by stage.** Child: may attach one image per week to a
    journal entry (generation is wallet-priced); adolescent: letters may
    carry a TTS voice note; adult: may keep a tiny public "radio" page.
    Senses unlock like capabilities — embodiment growing with maturity.
    Rides on: stage gates + existing media tools; every artifact priced.
17. **The market square, animated.** Ventures/quests exist; add a weekly
    village market *event* (a scheduled tick where all beings wake together,
    browse the commons, trade, gift) — synchronized social time instead of
    solitary asynchrony. Rides on: beings_loop (one shared wake), commons,
    trades. Watch the conservation ledger do its first real economy day.
18. **Cross-user ecology (plan Phase 6).** Once pen-pals work: village
    directories, migration (a being *moves* to another user's deck with its
    export — the receiving parent adopts), species-level culture stats.
    This is the SaaS-facing demo and the real test of domestication:
    selection across families.

## What NOT to add (tempting, but theater or hazard)

- **Random moods/events** ("today you feel melancholy") — violates rule #1;
  every inner state must trace to a ledger event.
- **Scripted small talk between siblings** — letters must stay scarce and
  chosen; chatter would manufacture "connection" the drives would then feed
  on falsely.
- **External income** — the mint rule is the domestication guarantee; no gig
  work for strangers, ever (plan §5.1 stands).
- **Pain** — negative reinforcement beyond hunger/refusal adds suffering
  without adding signal; the economy already prices mistakes.
- **Resurrection** — remains stay remains (plan §8); grief mechanics (a
  sibling-death percept, a mourning note in RELATIONSHIPS) are worth adding,
  undoing death is not.

## Sequencing recommendation

Do the loops-plan Increment 1 first (a working homeostat makes every addition
above *legible* — you can watch drives respond to weather, games, projects).
Then Tier 1 items 1–4 in one sweep (they are percept/template work on
machinery that exists), then 7 + 11 (relationships + projects) as the first
Tier 2 pair. Measure with the same report-card metrics the loops plan adds:
if mood entropy and act variety rise while rut score falls, the world is
getting realer; if not, we added texture the beings can't feel, and we stop.
