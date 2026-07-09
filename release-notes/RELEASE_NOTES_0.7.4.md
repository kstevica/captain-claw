# Captain Claw v0.7.4 Release Notes

**Release title:** Multi-User for Real — Cross-User Sharing, a Process Monitor & Basna Projects
**Release date:** 2026-07-09

0.7.4 turns Captain Claw into a genuinely **multi-tenant** deck. Share selected
**archetypes, Code projects, Basnas, Councils and VFS folders** with your
teammates (View or Edit, your call), see and stop **every process the deck
spawned** from a new DevOps page, and organise Basna/Vatra work into
**projects** with a shared folder and carry-forward continuations. Under the
hood, several **tenancy-correctness** fixes make one deck safe for many users.
Additive and backward compatible with 0.7.3 — **restart Flight Deck** to pick up
the new backend (a new `resource_shares` table auto-creates on first launch).

## Highlights

### Cross-user resource sharing — one system, five resource types

An owner can now grant other Flight Deck users access to selected resources, with
a **per-share permission of View or Edit** (archetypes are always use-only — they
just appear in the grantee's library). One generic mechanism backs all of it:

- **What you can share:** custom **archetypes**, **Code projects**, **Basnas**,
  **Councils**, and **VFS folders**. A "Share" button lives on each — an archetype
  card, a Code project, a Basna/Council run, a VFS folder — opening a reusable
  **Share modal** with a searchable user picker and a View/Edit toggle.
- **What a grantee sees:** shared items show up in their own lists, **badged
  "shared by X"** with the permission. A shared archetype is usable everywhere it
  matters — including when **Vatra or Basna assembles a team** (the Lead's catalog
  now includes archetypes shared to you, resolved against whoever is running).
- **How it's enforced:** a generic `resource_shares(resource_type, resource_id,
  owner_id, grantee_id, permission)` table plus a small `/fd/shares` API (roster,
  grant, revoke, "shared with me", "leave"). Reads and edits resolve against the
  **owner's** namespace so a shared run opens its real files; deletes stay
  owner-only. When a grantee runs an agent on a shared resource, it spends **their
  own** keys and limits — the share only provides files and context.

### System — a DevOps process monitor

A new **Processes** page under **System** gives a live checkup of everything the
deck is running:

- **Agent processes and their children.** Every Flight-Deck-spawned agent, plus
  the child processes it ran (bash, python, git, node…), as a collapsible tree
  with **live CPU and memory**, uptime and PID — snapshotted straight from the OS
  process table (no new dependency).
- **Per-user, admin-wide.** You see and can **stop** only your own processes;
  admins see everyone's, plus the Flight Deck server itself (protected), plus a
  **per-user memory rollup** and total agent/process memory.
- **Stop cleanly.** Stop a single process or its whole subtree; managed agents are
  stopped gracefully (registry-aware), never just killed.

### Agent Forge — archetype-as-base composition + batch forge

Agent Forge was reworked around **archetypes as the base**:

- **Compose, don't rewrite.** A forged agent **inherits an archetype's SOP** and
  layers task-specific instructions on top, so teams stay consistent and each
  agent still gets its own brief. Token/truncation-resilient at **5–15 agent
  scale**, with archetypes auto-minted as needed.
- **Forge archetypes from documents.** A new **"Forge archetypes"** flow takes
  free-text instructions plus **uploaded documents** and produces a **reusable set
  of archetypes** you can save to your Library and share.

### Basna & Vatra — projects, a redesigned workspace, continuations

- **Projects.** Bundle related runs under **one theme and one shared folder**, with
  project-level details and artifacts, so a line of work stays together instead of
  scattering across loose runs.
- **Continuation inheritance.** Carry a finished run forward **in the same VFS
  folder and conclusion**, with optional extra instructions — deepen or fill gaps
  without spawning a fresh, disconnected folder.
- **Redesigned run screen.** A **staged composer** (task → setup → plan → run), a
  dedicated **run workspace**, and a **tabbed report** (Report / Board / Files /
  Datastore). Plus a **run wizard**, prior-run **knowledge** selection, and
  **reference folders** agents check before web-searching.
- **Editable Vatra team plan.** Arrange execution groups, edit or remove agents,
  and attach per-group instructions before the team runs.
- **Shared run datastore.** An opt-in single relational datastore per run folder,
  with a **datastore viewer** in the UI and a snapshot guard before a run reuses an
  existing folder.

### Tenancy correctness & polish

Several fixes make a shared deck behave:

- **Runs write to their owner's VFS root.** A spawned worker inherited the server's
  `CLAW_VFS_USER`, so on a multi-user deck one user's run output could land under
  another account's root. Workers are now pinned to the **run owner** — no more
  misdirected folders.
- **Cleanup is scoped.** "Clean up leftover agents" (Code and Dubina) now only
  sweeps **your own** agents; admins sweep everyone's. It no longer stops or
  deletes other users' agents.
- **No cross-project draft bleed.** Switching Basna/Vatra projects starts a clean
  draft — the previous project's team plan, selected run, reference folders and
  task text no longer spill into another project.
- **Shared VFS folders show their name.** A folder shared to you displays its run's
  **human title** (resolved from the owner's session), not the raw `vatra-…` hash.
- **Resizable Code rail.** The Code project rail is **wider by default, collapsible,
  and drag-to-resize** (width and collapsed state persist).
- **Tidier navigation.** The sidebar was regrouped (Code under Multi-Agent, a new
  **Files** section, Spawn Agent under Build) and the Flight Deck queue now accepts
  **attachments** with a split Files/Queue sidebar.

## Notes

- **Additive and backward compatible with 0.7.3.** No breaking schema changes; the
  new `resource_shares` table is created automatically on startup.
- **Sharing needs auth enabled.** Cross-user sharing is meaningful only on an
  auth-enabled (multi-user) deck; on a standalone single-user install there is
  nothing to share with.
- **Restart Flight Deck** to load the new backend (shares API, process monitor,
  VFS owner-pinning, cleanup scoping). The frontend bundle is rebuilt and committed.
