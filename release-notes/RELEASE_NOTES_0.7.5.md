# Captain Claw v0.7.5 Release Notes

**Release title:** The Long Horizon Planner — a Coordination Plan You Approve Before the Team Runs
**Release date:** 2026-07-12

0.7.5 gives **Vatra** a brain before the hands. A permanent **Group 0 — Long
Horizon Planner** now runs first and drafts a structured, **per-agent
coordination plan** — for every teammate: its mandate, the artifact it produces,
which teammates it consumes from, and the hand-off notes downstream agents need.
The run then **pauses at a mandatory gate** where you review, edit, and either
**Execute** or **Cancel** — the same plan → approve flow Code already uses. Each
worker then runs with its own slice of the plan injected into its prompt, so the
team executes against **one shared plan** instead of eight agents improvising.

The planner can also **ask you clarifying questions**, you can **re-group** any
agent and have the plan **re-generated** for the new order, and the dispatch log
now shows **which model is being called** while it's thinking. Additive and
backward compatible with 0.7.4 — **restart Flight Deck** to pick up the new
backend.

## Highlights

### Group 0 — the Long Horizon Planner + a mandatory plan gate

Every Vatra run now opens with a planning pre-phase and pauses for your sign-off:

- **A real planning agent.** The `long-horizon-planner` runs as a live Group 0
  agent (its own card), reads the task, attached files, and the whole roster
  (each agent's role + resolved phase), and drafts the coordination plan.
- **A structured, per-agent plan.** For each teammate: a **mandate** (what it must
  accomplish), what it **produces**, which teammates it **consumes from**, and
  **hand-off notes** — plus a short team **overview**. At run time each worker gets
  its own slice injected, so it knows what's expected of it *and* what to expect
  from the others.
- **A mandatory gate.** The run stops at `awaiting_plan` — **nothing spawns** until
  you **Execute** (run Group A) or **Cancel** (discard; nothing runs). Mirrors the
  Code plan gate.
- **Never a dead end.** If the planner fails or times out, the run falls back to a
  pass-through plan derived from the decomposition rather than stalling. Headless
  runs (agent-started, continuations) **auto-approve**; **resume** neither re-plans
  nor re-gates.

### A coordination plan you actually read

The gate is a **master–detail** view — the agent list on the left, the selected
agent's fields on the right:

- **Reads as text by default.** Mandate, produces, hand-off, and the overview
  render as plain text; group shows as a badge, dependencies as static chips. An
  **Edit** toggle flips the panel into editors only when you want to change
  something — no wall of textareas to scroll past.
- **Phase-aware.** Each agent's **group (A/B/C/D)** is an editable selector; the
  list **sorts by phase** and re-sorts live as you re-group, keeping your selection.
- **Your grouping is absolute.** The group you set at the gate now **overrides the
  archetype floor and dependency repair** — an agent runs exactly in the phase you
  put it (the shared board / wait bridges any ordering you create).

### Re-plan when you change the shape

Because a group change (or a clarification answer) makes the planner's mandates
and hand-offs stale, editing the shape surfaces a **Re-plan** action: it re-runs
the planner with your new grouping and answers folded in, regenerates the whole
coordination for the new order, and lands you back at the gate to review before
Execute.

### The planner can ask *you* questions

When a genuinely blocking decision would change the plan, the planner surfaces up
to **9 clarifying questions** as a **dynamic form** — a Questions tab beside the
plan (same master–detail layout):

- Each question offers **1–4 suggested answers plus a free-form "Other"**, as
  **single- or multiple-select**.
- Answer what matters, hit **Re-plan**, and your answers are folded back into the
  planner so the coordination reflects your decisions. Unanswered questions fall
  back to the planner's defaults.

### See the model while it's thinking

A worker's LLM call is usually the slowest part of a turn, and it used to show
nothing until the tokens landed — a long local-model call looked stuck. The
dispatch log now streams a live **`Calling LLM (<model>) · … ctx tokens`** line at
the start of every call, deduped to one line per call. It's wired across **Vatra**
(owners + planner) and **Basna** (ensemble, fact-check, and deep/Horizon) — and it
immediately paid for itself by surfacing a bug where the approve gate dropped your
tier config and workers silently fell back to the default model. Fixed: the gate
now carries your **tiers/keys** through (with an owner-tiers fallback), so workers
run on the models you configured.

## Notes

- **Additive and backward compatible with 0.7.4.** No breaking schema changes.
  Group 0 is a distinct pre-phase, not a real execution group, so the A→B→C→D
  machinery is untouched; absent a Group 0 plan, Basna and any un-gated path are
  byte-identical.
- **Every interactive Vatra run now pauses at the gate** (including the one-shot
  "Run team" path) — that's the point of a permanent, mandatory Group 0. Only
  headless agent-started runs and continuations auto-approve.
- **Restart Flight Deck** to load the new backend (the planner pre-phase, the
  `awaiting_plan`/`planning` statuses, the plan approve/cancel/re-plan endpoints,
  the absolute group lock, and the dispatch-status plumbing). The frontend bundle
  is rebuilt and committed.
