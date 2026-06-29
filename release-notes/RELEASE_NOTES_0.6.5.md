# Captain Claw v0.6.5 Release Notes

**Release title:** Teams that wait for each other — Vatra rendezvous, specialist tiers, and a tidier deck
**Release date:** 2026-06-29

A polish-and-plumbing release on top of 0.6.4's Frontier Horizon. The headline is a
**`wait` rendezvous** for Vatra: a specialist that genuinely depends on a teammate's
artifact now **blocks until it's ready** instead of improvising a guess. Around it: the
Library gains dedicated **coding** and **vision** tiers (so code and image work route to the
right model), Basna/Vatra/Plan runs get a **sticky "what phase is running now" pill** plus
**wait events** in the timeline, Plan mode's live agent cards finally show **per-agent token
counts**, and a new **Agent Folders** admin page lets you inspect and clean up `fd-data`
subfolders. Fully additive and backward compatible with 0.6.4 — nothing changes for existing
runs unless you reach for it.

---

## What's new

### Vatra `wait` — agents block on a teammate's artifact instead of guessing

Agents on a Vatra team often look for a piece a teammate is supposed to produce — a VFS file
or a board post — don't find it yet, and **improvise a guess**. A new blocking **`wait`**
action lets a specialist that truly depends on a specific artifact **pause until it's ready**.

- **Two conditions.** `wait` takes a `path` (`vfs:<project>/<file>`) or a `query` (board
  keywords). The `/agent/wait` endpoint **long-polls** the VFS file or the blackboard every
  second.
- **Bounded, never hangs a run.** Capped at **90s** (tool timeout raised to 120s), which is
  below the dispatch timeout, so a wait can't stall the run. On timeout it returns a **board
  digest** plus a "proceed deliberately" nudge — so the agent does the right *something else*
  rather than erroring out. Sibling owners keep working while one awaits (they're concurrent
  coroutines).
- **Visible in the timeline.** A wait emits a `wait` stage event when it **starts** (⏳ … ≤Ns),
  when it **resolves** (✓ got `<file>` / board match), and on **timeout** (⌛ not ready) — so a
  run shows exactly who is blocked on what instead of a waiting agent just looking slow.
- **Steered, not forced.** Worker prompts now point agents to `wait` when *genuinely* blocked,
  and no longer say "don't stall waiting" unconditionally — don't busy-poll, but wait once
  when you really depend on something. There's no forced `depends_on` ordering; the tool is
  the whole mechanism. Owner+project file resolution is sandbox-checked (`resolve_under`), so
  the multi-user Flight Deck process serves the right run's files and rejects escapes.

### Coding & Vision tiers in the Library

Two new model tiers join the Library so work routes to a model suited for it:

- **`coding`** (default `opus-4-8`, 65 536 output context) for multi-file diffs and debugging.
- **`vision`** (default `sonnet-4-6`) for image / screenshot / document understanding.

Both are wired through **every** touchpoint — the `archetypes.json` tier table, the backend
`_VALID_TIERS`, Dubina's `TIER_ORDER`, consciousness's `_TIER_RANK`, and the frontend
`TIER_ORDER`. A **`backfillTierMap`** makes existing saved tier sets gain the new cards
(inheriting their provider/key) instead of rendering empty. Code-producing archetypes
(`code-implementer`, `debugger`, `simplifier`) are reassigned from *balanced* to **coding**,
and a new **"Vision & Multimodal"** family (`visual-extractor`, `ui-reviewer`,
`brand-design-reviewer`) plus vision routing lines in the project-coordinator and concierge
rosters round it out.

### Sticky phase pill + per-agent tokens in Plan mode

Runs emitted plenty of detail (action / usage / narration) but no clear **"what phase is
running now"** signal — under a long event stream the stage markers just scrolled away.

- **Explicit phase events** are now emitted at every transition and surfaced **stickily** in
  the Progress header (with a spinner while running), and phase rows are accented in the log:
  - **Vatra:** Planning → Spawning → Intro → Main → Review → Verifying slices → Synthesizing →
    Verifying deliverable → Learning → Done.
  - **Basna:** Routing → Spawning → Generating → Merging → Verifying → Done.
  - **Plan:** `Step x/y: <goal>` per step, plus Planning / Synthesizing / Done.
- **Per-agent token counts in Plan mode.** The plan-step progress mirror previously dropped
  `usage` events, so Plan-mode agent cards showed activity but no running token count. Usage
  events (prompt / completion / total) are now forwarded into the parent plan log, so the
  cards render the per-agent token header like a normal Basna run.

### Agent Folders — inspect and clean up `fd-data`

A new **admin-gated** Flight Deck page to inspect and clean up agent subfolders under
`fd-data`.

- **What it shows.** Each folder with its on-disk **size**, **file counts**, **last-modified**,
  and **desktop presence** — *running* / *on-desktop* / **orphaned**, where orphaned means a
  folder whose slug is no longer in the process registry.
- **Backend** (`/fd/agentfs`, admin-gated): list folders, list and view/download workspace
  files, and **delete a folder** (refused if the agent is still running). Path-sandboxed to
  direct children of `fd-data`.
- **Frontend:** `AgentFoldersPage` + `agentFsStore`, wired into routing and the sidebar.
  Light-theme-friendly colors, filters (running / on-desktop / orphaned / name search), and a
  rich file viewer that renders **HTML** (sandboxed iframe) and **Markdown** with GFM tables.

---

## Upgrade notes

- **No schema changes** and **no new required configuration.** Everything above is additive.
- **Library tiers.** Existing saved tier configs are **auto-backfilled** with the new
  `coding` / `vision` cards (they inherit the provider/key of your existing tiers) the next
  time the Library loads — review them if you want code/vision work pointed at different
  models.
- **New endpoints:** `POST /fd/<run>/agent/wait` (Vatra) and the `/fd/agentfs/*` router
  (admin-gated). The `wait` action is available to Vatra workers automatically.
- **Frontend bundle rebuilt** for the phase pill, Plan token cards, tier UI, and Agent
  Folders page.

Backward compatible with 0.6.4.
