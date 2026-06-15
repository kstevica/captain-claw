# Captain Claw v0.5.7 Release Notes

**Release title:** Basna — a network-source ensemble that routes, runs, and merges a fleet
**Release date:** 2026-06-15

A Flight Deck release introducing **Basna**, a new ensemble mode that sits parallel to Council. Where Council is a multi-round deliberation *among* agents, Basna is one-shot and selective: a **router** picks the smallest set of specialist archetypes for the task, spawns them fresh, runs them **blind and in parallel**, and **merges** their outputs weighted by each archetype's learned reliability — calling a synthesizer only when they genuinely disagree, and **learning** which archetypes are good at what so routing improves over time. This release also **extracts model tiers and the archetype gallery out of Agent Forge into a dedicated Library page**. Additive and backward compatible with 0.5.6.

---

## What's new

### Basna — network-source ensemble mode
A new Flight Deck page (sidebar: **Basna**). You describe a task; Basna does the rest:

- **Router → minimal team.** A router (running on a tier you pick, default **Reasoning**) classifies the task (`domain` / `difficulty` / `converge|diverge`) and selects the **smallest** archetype subset that covers it — scaled to difficulty, capped by a Max-agents control. Per-query selection is the efficiency engine: easy tasks get one agent, hard ones get a handful. Falls back to deterministic keyword matching if the router LLM is unavailable.
- **Selective ephemeral spawn.** The chosen archetypes spawn fresh as process agents for the run and are **fully removed afterward** (no lingering "Stopped" rows). Their outputs and activity are persisted to the session, so nothing is lost.
- **Blind parallel dispatch.** Agents run independently and cannot see each other — independence is what makes the merge meaningful (the opposite of Council's deliberation).
- **Weighted merge, conflict-gated synthesis.** Outputs are combined weighted by each archetype's learned reliability. For convergent tasks that agree, the highest-weighted answer wins (no extra LLM call); only on genuine disagreement is a reasoning-tier **synthesizer** invoked. Divergent tasks are deduped and kept. Every run reports its `method` and a confidence.
- **Reliability learning.** Each agent's contribution is scored against the compiled truth and folded into per-archetype, per-domain reliability — so the next route's priors reflect what actually worked. A 👍/👎 on any agent is a first-class override that revises the weight without double-counting.
- **Live progress log.** A timestamped log streams every stage as it happens — route → spawn → per-agent dispatch → merge → learn → done — including each agent's **tool calls, narration (emphasized), and LLM usage (model + tokens)**. Tail-30 + scrollable, auto-scrolls to the newest line, and is **persisted** (reopen a session to see it). Export the whole log or any single agent's activity.
- **File attachments.** Attach files or **paste images** into a run. They're stored on the session and copied into **every spawned agent's workspace**, so agents work with them via their `read` / `pdf_extract` / `xlsx_extract` / `image_vision` tools. Drag-drop, click-to-attach, and paste are all supported.
- **Generated-file capture.** Files the agents *create* are scanned out of their workspaces **before teardown** and saved back onto the session — downloadable from a **Generated files** panel, so generated content is never lost.
- **Edit each agent before the run.** Every routed agent expands into an editor: a **tier picker** (resets the model fields to that tier), **provider** and **cognitive-mode** dropdowns, **model**, **API key**, **base URL**, **input/output context**, the **fleet instructions (system prompt)**, **extra task instructions**, and the role. Overrides persist on the session route and take precedence at spawn/dispatch (per-agent override → Library tier → registry default).
- **Markdown everywhere.** The compiled truth and each agent's answer render as markdown (GFM tables included), with **fullscreen** and **export-.md** buttons.

New endpoints under `/fd/basna`: `route`, `execute`, session CRUD, `sessions/{id}/runs`, `sessions/{id}/progress`, `sessions/{id}/files` (upload/download/delete), and `runs/{id}/feedback`. New tables: `basna_sessions`, `basna_runs`, `archetype_reliability` (migrate in place).

### Library — model tiers + archetype gallery, now their own page
The **Model Tiers** editor and the **archetype gallery** moved out of Agent Forge into a dedicated **Library** page (sidebar: **Library**):

- **Model Tiers** — the per-tier model editor (Reasoning / Balanced / Fast / Long context: provider, model, key, base URL, in/out context) plus the "Additional API Keys" env vars, all persisted per-user. Forge now *consumes* this saved config (it shows a "Configure in Library" hint), and Basna's router/agents/merge resolve their models and keys from it.
- **Archetype gallery** — the curated catalog, where clicking a card **spawns that archetype directly** as an agent (resolving its tier to a concrete model).

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild the frontend only if you build locally:
npm --prefix flight-deck run build
```

- **Restart the Flight Deck server** for the `/fd/basna/*` endpoints, the new tables/migrations, and the Library routes.
- **Restart the agent** (`captain-claw-web`) is not required for Basna, but recommended after a pull.
- **Hard-reload Flight Deck** for the Basna page, the Library page, and the moved tier/gallery UI.
- **Desktop app:** `./build-desktop.sh` — everything above is in the rebuilt bundle.

Backward compatible with 0.5.6 — Basna and Library are additive; the Basna tables migrate in place and no existing configuration or API changes.
