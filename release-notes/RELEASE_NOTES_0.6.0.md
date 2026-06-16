# Captain Claw v0.6.0 Release Notes

**Release title:** Agentic Basna & per-tenant archetypes — agents that start their own ensembles, and a library you can extend
**Release date:** 2026-06-16

A Flight Deck release that makes **Basna agent-native** and **archetypes per-tenant**. Agents can now read *and* start Basna runs as a tool — kick off an autonomous multi-agent research/analysis from any channel and get the result relayed back when it finishes. Separately, the curated archetype set becomes a base you build on: every user can create their own archetypes (by hand or from a prompt) and they show up everywhere base ones do — the Library gallery, Agent Forge, and Basna routing. Additive and backward compatible with 0.5.7.

---

## What's new

### Per-tenant agent archetypes
The curated `archetypes.json` is now the **base** set; each user can extend it.

- **Create your own archetypes** on the **Library** page — manually (role, family, description, cognitive mode, tier, tools, keywords, fleet instructions) or **generated from a prompt** (describe the agent; an LLM drafts a complete archetype you review before saving).
- **Override or add.** A custom archetype with the same id **shadows** the base one for that user; a new id is added alongside. Custom cards are badged and are edit/deletable; deleting an override restores the base.
- **Available everywhere base ones are.** Your archetypes appear in the Library gallery (one-click spawn), bias **Agent Forge** team composition, and are selectable by the **Basna** router — including per-user learned reliability.
- **Tools multiselect** in the editor (toggle chips for every known tool) instead of free-text.

New table `user_archetypes` (migrates in place). New endpoints under `/fd/archetypes/*` (CRUD + `generate`); `GET /fd/archetypes` now returns the merged base+user registry.

### Basna as an agent tool — read *and* run
A new always-available **`basna`** tool lets an agent work with its owner's Basna sessions like a datastore, and start new runs.

- **Read** — `list` / `get` / `agents` / `output` / `truth` / `analysis` / `files` / `get_file`: browse and search sessions, pull the compiled truth, the cross-agent analysis, each agent's output and tool activity, and fetch generated files into the agent's workspace. Owner-scoped, no token plumbing.
- **Start (v2)** — `start` kicks off a **new autonomous Basna run** on a task you hand it. Flight Deck auto-titles, routes the minimal team, and **executes the ensemble fully server-side** (no UI), then **reports completion back to the originating agent** so it relays the result on the user's channel. Fire-and-forget — works from **web, WhatsApp, glasses, or API**. Uses the owner's saved Library tiers (falling back to built-in defaults) and is capped at **2 concurrent** agent-started runs per owner.
- **Deterministic relay.** When the user explicitly says "run/execute a Basna…", the agent hands the task straight to the tool rather than relying on the model to pick it — so weak models don't go off and do the research themselves.

New agent-identity endpoints under `/fd/basna/agent/*` (read + `start`). Agents resolve to their owner by their unique auth token.

### Basna session titles
Every Basna session now has a **title** — type one, or it's auto-generated from the task (the router LLM returns a concise title; a heuristic covers the fallback path). Titles are editable inline and shown in the run list (with the task as a subline).

### Basna run monitoring
Agent-started runs surface in the **same** Basna run list, **badged** with their origin channel (e.g. `agent·whatsapp`), with an optional **"agent" filter**. The list **polls while runs are in flight** so background runs update live, and the confidence value is **colour-coded** (green / amber / rose by score).

---

## Fixes & hardening

- **Agent tool inventory is now complete.** The system-prompt tool list previously only included tools present in a hardcoded description table, silently dropping the rest (so an agent asked "what tools do you have?" couldn't list them all). It now falls back to each tool's own description, listing every registered tool.
- **No more web-search hijack.** A guard meant to catch a model that *claims* web research without calling a tool was matching ordinary replies that merely **describe** web tools or **ask** what to search — derailing weak models into a forced `web_search` loop. The matcher is tightened to genuine past-tense completion claims.
- **Plan-aware autonomous spawns.** Agent-started runs now pass the owner's full account (and a request carrying the owner) into execution, so plan/quota checks see the real plan (not the free default) and spawned agents are correctly owned; `spawn_process` reads request state null-safely.
- **Basna routing/merge regression fixed.** Restored the registry load that tier-credential resolution depends on in the router and executor.

---

## Upgrade notes

- **Migrations are automatic** — the `user_archetypes` table and the `basna_sessions.title` column are added in place on first run. No manual steps.
- **Restart to apply.** The agent-runtime changes (the `basna` tool, the deterministic relay, the tool-list and web-claim fixes) take effect when **agents restart**; the new endpoints when the **Flight Deck server** restarts.
- Backward compatible with 0.5.7.
