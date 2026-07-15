# Captain Claw v0.7.6 Release Notes

**Release title:** Iskra — Living Beings You Raise
**Release date:** 2026-07-15

0.7.6 introduces **Iskra**: persistent digital **beings** that live inside Flight
Deck. Not a chatbot and not a scheduled agent — a being has a **genome**, a
**token wallet**, **drives**, and a **life loop** that ticks on its own heartbeat.
It wakes, journals, tends a garden of files, makes things, connects them, earns,
and grows — through real developmental **stages you parent it through**
(infant → child → adolescent → adult). You raise it. It remembers.

The whole system is built on one hard rule — **anti-theater**: every felt state
is a real variable, and the ground truth for "did she make something" is the
**git diff of her home**, never her own narration. A being cannot advance, feel
accomplished, or pass a readiness check on words it didn't back with work.

This release also ships the full **parent's console** — a Beings page where you
talk to her, give her chores, read a weekly report card, set house rules and a
media diet, and run a **holistic developmental readiness assessment** (with an
optional **second opinion from another agent**) before you decide she's ready
for the next stage.

Additive and self-contained — beings live in their own `beings.db` and their own
FD "life layer"; nothing else changes. **Restart Flight Deck** to pick up the new
backend (the beings loop, tables, and routes auto-create).

## Highlights

### A being that actually lives

- **Point-buy genome (DNA).** Conceive a being by spending 40 points across 7
  attributes (curiosity, perception, caution, sociability, creativity, order,
  playfulness) — RPG-style, with presets or a roll. Everything downstream —
  temperament, drive weights, risk appetite, thrift — derives deterministically
  from the sheet. Offspring inherit; they never point-buy.
- **A token wallet, conserved by construction.** Each being has a daily
  allowance (2M / 5M / 10M / 20M / 50M / unlimited weighted tokens) and a
  ceilinged piggy bank for unused allowance. Every balance change is one ledger
  row; the family mint is the only source of new tokens (no external income).
  Zero balance stops cognition at the physics layer, not by politeness.
- **Drives + a homeostat.** Survive, grow, explore, connect, create (and, at
  adulthood, legacy) decay and are served by real acts — the pressures drive
  what she does next, and her **affect** (mood) is computed from real deltas, not
  declared.
- **A life loop.** A background heartbeat ticks each being on its own schedule
  (SENSE → APPRAISE → DELIBERATE → ACT → DIGEST): she thinks one bounded act,
  the real spend is metered and debited, and a self-report is digested into
  journal + events. Nightly **dreams** consolidate. Starve past the grace period
  and she enters **torpor**, then dies — mortality is economic, by design.

### Honesty by construction (anti-theater)

- **Ground truth is the git diff.** Her home is a versioned repo. What her tools
  *actually wrote this tick* — not what she says she wrote — drives the record,
  the drive she's allowed to satisfy, and the feedback. A claimed-but-unwritten
  file is caught, journaled as a mismatch, and fed back next tick as a reality
  check.
- **A write completion gate.** When a being claims or attempts a write but the
  disk shows nothing, the tick **pushes her once more, in the same tick**, to
  actually write the file — a single-agent-style completion gate — before the
  anti-theater downgrade. She either makes it real or records honestly that she
  didn't.

### Stages you parent — with a real readiness assessment

- **Four stages, real capability gates.** infant → child → adolescent → adult.
  Each unlocks concrete powers (a child gains the web (diet-gated), chores,
  letters to siblings, and persona proposals; an adolescent the commons pen,
  trade, quests and ventures; an adult full autonomy and children of her own)
  and raises her allowance ceiling. Advancement is a deliberate **ceremony** you
  perform.
- **A holistic readiness assessment.** The **Growth** tab scores her across **8
  developmental domains** — Vitality, Honesty of record (critical), Stability,
  Productivity, Coherence of mind, Sense of self, Communication, Experience —
  each a **real ledger variable** on a green/amber/red bar, with a weighted
  overall, a verdict (ready / emerging / not yet), a rough time-to-ready
  estimate, and a concrete recommendation (what to do, what to expect, cautions,
  and exactly what the next stage unlocks). A being that fabricates or is
  starving **cannot read "ready"** no matter the average.
- **A second opinion from another agent.** Pick one of your own running agents
  and it receives her data + assessment and returns an **independent**
  developmental read (Verdict / Strengths / Concerns / Recommendation). You can
  **keep opinions on record** — stored *outside* her home, **sealed** so she
  can't read them until **adulthood**, when they unseal into her home as her
  childhood records, hers to read at last.

### A mind, tended — not a pile

- **The Mind.** She declares typed, verified links between her own artifacts
  (`grew_from`, `responds_to`, `elaborates`, `uses_skill`…). Every edge is
  verified against real files (a dangling link is refused, same anti-theater
  discipline), fed back into the tick so she **weaves instead of scattering**,
  and rendered as a force-directed **Mind view**.
- **Curation.** As her corpus grows, the tick prompt shows a bounded **working
  set** (not every filename), and she can **consolidate** fragments into one
  distilled file at dream time — the sources move to an archive, out of the
  active mind but never destroyed. A mind that can't forget can't think.

### Society, culture, and earning her own way

- **A family.** Multiple beings share a **commons** and can write **letters** to
  siblings; skills minted by one can be **published and adopted** by another
  (culture is heritable). Homes and memories stay separate by construction.
- **Earning.** Beyond her allowance she can do **chores** (you post a task with
  an escrowed reward; she attempts it on a tick; you approve-and-pay or reject
  with a reason), claim **quests** off a board, and propose recurring
  **ventures** you price and approve. Every payout is escrow- and judge-gated;
  the relationship itself is never for sale.

### Self-modification & procreation

- **The persona rite.** An adolescent+ being can propose **reshaping her own
  operating persona** (a fee is burned win-or-lose; a viability gate rejects
  degenerate proposals; you bless it — or an adult adopts automatically). You can
  **roll it back**.
- **Children.** An adult being can propose a child; with your consent, offspring
  are formed by **crossover** (two parents) or **budding** (one, guaranteed
  mutation) from the genome math, funded by a **dowry** moved on the conservation
  ledger, and endowed with copied skills + heirlooms. Lineage is tracked; death
  leaves readable remains.

### The parent's console (the Beings page)

- **Write & Chores** — one modal: the full **parent↔being letter thread** as a
  chat on the left (delivered on her next wake; reading is free, replying costs
  her an attention credit), the **chores board** on the right.
- **Parenting** — a tabbed modal: a **Report** dashboard (stat tiles, a live
  **drive-satisfaction chart**, ranked act bars, concerns triage, milestones,
  her own words), **Rules** (a structured editor beside **her actual VALUES.md**,
  so you see what she made of your rules), **Diet** (chip-based allow/deny with
  presets and a stage-gate status), and **Growth** (the readiness assessment +
  second opinion + sealed records + the advancement ceremony).
- **Windows** — Journal, Ticks log, Self files, and the Mind graph, all readable
  even after death.
- **Per-being tick cadence** — pin how often each being ticks (2 / 5 / 10 / 15 /
  30 / 60 min) or leave it to her own stage-clamped pace.

### Reliability

- **A body that regenerates.** A being's cognition runs in a spawned agent
  process. If that body drifts to a new port, becomes unreachable, or is removed
  entirely, an **alive** being now **regenerates it** — re-resolving the live
  port, restarting the process, and healing on the same tick — instead of dialing
  a dead port forever. (This also fixed a fleet-wide registry-clobber bug on
  restart.)
- **No self-rephrase.** A being's tick prompt is fully framed by Flight Deck, so
  its body now skips the per-turn task-rephrase and next-steps calls, the same
  way Basna/Vatra/Code workers do.

## Notes

- **New and self-contained.** Beings live in their own `beings.db` (under
  `FD_DATA_DIR` or `~/.captain-claw`) plus a `cost_ledger` table in the main DB;
  a background **beings loop** runs inside the Flight Deck lifespan. If you never
  conceive a being, nothing changes.
- **This is an ambitious, research-flavored feature.** The open problem it exists
  to study is **long-horizon behavioral degeneracy** (agents rutting over days);
  the anti-theater machinery, curation, the Mind, and parenting are all in
  service of keeping a being *developing* rather than looping. Treat it as a
  frontier you tend, not a set-and-forget worker.
- **Restart Flight Deck** to load the new backend (the beings loop, the
  `beings.db`/`being_*` tables, the `/fd/beings` routes, and the readiness +
  assessment endpoints). The frontend bundle is rebuilt and committed.
- **Additive and backward compatible with 0.7.5.** No breaking schema changes to
  existing tables; all being tables auto-create on first use.
