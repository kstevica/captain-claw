# Captain Claw v0.6.4 Release Notes

**Release title:** Think way ahead, way long — Frontier Horizon comes to Basna & Vatra
**Release date:** 2026-06-28

A release about **depth**. Until now Basna and Vatra parallelized *breadth* — a fleet of
specialists answering once and merging. This release gives them an optional **Frontier
Horizon** mode that spends test-time compute, gated by verifiers, to reach frontier-grade
quality on cheaper models: each worker can **self-consistency vote, get critiqued by a
diverse panel, and revise**; the team's final answer can be **adversarially verified and
rewritten**; and a whole task can be driven as a **verify-gated, re-planning multi-step
horizon** where each step is itself a single model, a full Basna ensemble, or a full Vatra
team. Plus the shared VFS now records **who wrote each file and when**, and the Basna/Vatra
controls have been **redesigned** around a single Effort choice. Additive and backward
compatible with 0.6.3 — everything new is **off by default**.

---

## What's new

### Deep mode — frontier-grade depth on any model

A new **Deep** toggle on the Basna/Vatra page turns on the depth axes of the Frontier
Horizon engine. It reconstructs, in the harness, the three things frontier models get "for
free" — low per-step error, error recovery, and self-correction:

- **Per-worker depth (Basna).** Instead of one shot per agent, each worker spawns a small
  pool of **independent rollouts**, **self-consistency-votes** across them, runs the leading
  answer past a panel of **diverse-lens critics** (the `phrygian` / `aeolian` / `locrian`
  cognitive modes, as adversarial refuters), and **fixes** with the critique if it fails.
  Set the **samples** width to control the vote.
- **Per-owner depth (Vatra).** A team can't spawn N copies of each specialist without
  flooding its shared blackboard, so Vatra instead verifies **each owner's slice** with the
  critic panel and revises it once if a majority refute it — the same depth, blackboard-safe.
- **The closer (both).** After the answer is merged (Basna) or the deliverable assembled
  (Vatra), a critic panel reviews the **final** result and rewrites it once if it's refuted —
  the self-correction step a single pass lacks.

Critics always run on a model **different** from the one that produced the answer (never
self-grading), and the whole thing is **budget-bounded**. Deep is much slower and costlier
than a standard run — use it when quality matters more than speed.

### Plan mode — think way ahead, way long

The headline lever. A new **Plan** toggle decomposes a task into **ordered steps**, drives
each to a **verified** result before the next begins, **re-plans** the remainder when a step
can't be verified, and **synthesizes** the deliverable from the verified steps — the
long-horizon behavior frontier models get from goal coherence, reconstructed externally so
the system never compounds an unverified step.

- **Three step engines.** Each step can run as a single fast model (*simple*), a full
  **Basna ensemble**, or a full **Vatra team** (*complex*) — chosen by the mode card above,
  so the card means the same thing everywhere: *who staffs the work*. When a step runs an
  ensemble or team, its **live agent activity is mirrored into the plan log** so you watch
  the agents work, not just a step summary.
- **Parallel plans.** Tick **parallel** and the planner emits a **dependency graph** instead
  of a linear list — independent steps run concurrently in dependency waves, and each step
  sees only the outputs it depends on.
- **Forced teams.** Pick a team and it staffs **every** step's ensemble/team.
- **Bounded and honest.** Steps, fix attempts, and re-plans are all capped; the run ends with
  an explicit reason and never silently drops an unverified step.

New endpoint: `POST /fd/basna/plan` (runs in the background; the UI polls the live log like
any other run). Plan-step child runs are real Basna/Vatra sessions — their archetype-
reliability learning still closes — and are hidden from the session list under their parent.

### VFS provenance — who wrote what, when

The shared cross-agent filesystem viewer now shows:

- **Timestamps** on every file and folder (and the last-modified time per project).
- **Authorship** — an append-only `.vfs-meta.jsonl` sidecar records the agent that wrote each
  file, surfaced as a `✎ <agent>` label. Concurrency-safe; the main interactive agent writes
  anonymously, so there's no noise.
- **Run titles** — a `basna-…` / `vatra-…` project shows the human title of the run that
  created it, with a colored mode badge.

Authorship is recorded going forward, so files written before upgrading show a timestamp but
no author until they're next written.

### A redesigned Basna/Vatra control panel

The scattered row of selects, number boxes, and checkboxes is gone. The controls are now
organized around a single **Effort** choice — **Standard · Deep · Plan** — with
**progressive disclosure**: only the options for the chosen effort are shown, in a panel that
explains what it does. Router tier and team size move behind a **Tuning** toggle. Deep and
Plan are available in **both** Basna and Vatra modes, and a **Help** button opens an in-app
explainer of all three.

### Reliability & feedback polish

- All Horizon model calls (critics, revise, planner, verifier, synthesizer, each step) are
  **time-bounded**, so a slow or hung reasoning model can't stall a run.
- A revision that **collapses** a substantial answer to a fragment (e.g. a reasoning model
  returning only a reasoning tail) is **rejected** — good content is never silently replaced.
- The closer streams **per-critic** progress plus a **heartbeat** so the long verification
  phase shows it's alive.

---

## Compatibility & upgrade notes

- **Fully additive, off by default.** Standard Basna/Vatra runs are unchanged. Deep, Plan,
  and the VFS sidecar only activate when you opt in.
- **No schema changes.** Deep/Plan settings ride in the existing session `config` JSON; the
  VFS authorship sidecar is a per-project file alongside your tree.
- **New endpoint:** `POST /fd/basna/plan`. The closer, per-owner, and plan paths reuse your
  existing Library tiers — point your **critic / reason tier at a model that returns real
  content** (not a reasoning-only model that returns empty `content`), or critic verdicts and
  revisions degrade safely but won't be as useful.
- Rebuild the Flight Deck bundle (`cd flight-deck && npm run build`) if you run from source.

Backward compatible with 0.6.3.
