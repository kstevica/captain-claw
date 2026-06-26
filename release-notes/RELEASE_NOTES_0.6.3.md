# Captain Claw v0.6.3 Release Notes

**Release title:** Teams that build themselves — auto-assembled Councils & Vatra
**Release date:** 2026-06-26

A release about **multi-agent teams that compose themselves to the task**. Council — until now a deliberation among agents you'd already started — can now **auto-assemble its own panel**: a router reads your topic, picks a diverse set of specialist archetypes, spawns each as a fresh agent briefed for that exact discussion, and runs the rounds over them. And this is the first release to fully document **Vatra**, the collaborative sibling of Basna: a Lead decomposes the task, specialists each own one piece on a shared blackboard, and a dedicated reporter assembles a single deliverable. Additive and backward compatible with 0.6.2.

---

## What's new

### Council auto-assemble — a panel modeled to the topic

Council had one way in: pick from agents you'd already spawned. That made it powerful but heavy — you had to stand up the right specialists by hand before you could deliberate. **Auto-assemble** removes that step and brings Council in line with Basna and Vatra, which already spawn purpose-built agents from archetypes.

On the new-session screen there's now a **Auto-assemble ↔ Pick agents** toggle (auto is the default):

- **Auto-assemble** — give a topic, pick a panel size (2–6), and Start. A **Council Assembler** router (fast tier + the merged archetype catalog) selects a *diverse* set of archetypes — optimizing for complementary, sometimes opposing perspectives rather than the minimal single-answer team Basna routes — then spawns each as a fresh ephemeral agent. Every panelist arrives **briefed for this exact council**: its archetype persona plus a tailored "your seat on this council" charge (the topic, the session type, and the specific angle it should bring and who it tensions against). The deliberation then runs exactly as before.
- **Pick agents** — the classic flow, unchanged: choose from your already-running agents, Old Man → moderator mode, etc.
- **Hand-pick the specialists (optional).** In auto mode you can open a specialist list (your full archetype gallery, base + custom) and choose exactly which archetypes form the panel. Selected archetypes are honored **in full** — the panel-size slider steps aside and every one you pick gets a seat. Leave it empty to let the router compose the panel for you.

A new instruction prompt, `instructions/council/router.md`, encodes the "diversity of viewpoint, deliberate tension, session-type aware" selection policy. If the router LLM is unavailable it falls back to deterministic keyword matching, so assembly always returns a panel.

### Temporary agents, and a clean way to dispose them

Auto-assembled panelists are **ephemeral by design** — spawned for one council. The session now knows that:

- It records the spawned agents' slugs (`spawnedSlugs`) in its config and survives every config rewrite (memory, auto-advance, extend, pass/delegation, conclude) and reloads.
- The sidebar shows a **Temporary** badge on the panel and a **temp** tag on each spawned agent, so it's never ambiguous which agents are throwaway.
- A **Dispose agents** button appears at the natural moment — in the concluded controls bar — and any time in the sidebar Agents header. It disconnects the sockets, tears the panel down via `POST /fd/council/teardown`, and marks the session disposed (the badge flips to **Disposed**, and a disposed session never tries to reconnect dead agents). The transcript, synthesis, votes, and TL;DRs are all kept — only the agents are removed.
- **Deleting a session** still tears its temporary panel down automatically, so nothing leaks even if you never press Dispose.

Manual (picked-agent) councils show none of this — disposal is only offered for panels Council itself spawned.

New endpoints: `POST /fd/council/assemble` (route + spawn → council-ready agent defs) and `POST /fd/council/teardown` (stop + remove). No schema changes.

### Vatra — the collaborative ensemble (now documented)

**Vatra** is the collaborative sibling of Basna, surfaced as a **compose mode on the Basna page** (toggle: *Basna — independent ensemble* vs *Vatra — collaborative team*). Where Basna spawns agents that each answer the *whole* task blind and then **merges** their uncorrelated outputs, Vatra runs a **team that divides the work**:

- **Lead decomposition.** A Lead reads the task and splits it into the smallest set of **complementary, owner-assigned subtasks** — each with a title, an owner archetype, a brief, and `depends_on` links — plus a **shared context** contract every piece must honor. Press **Plan team** to produce and review this decomposition before anything spawns (it persists as a prepared session). You can also **pre-pick the archetypes** the Lead must use as owners.
- **Parallel ownership on a blackboard.** Press **Run** and each subtask owner spawns fresh and works **in parallel**, but — unlike Basna — they **collaborate through a shared blackboard**. An optional **intro (prep) round** lets each specialist post groundwork (facts, sources, an outline) before writing its full piece, so the team starts from shared footing.
- **Delegation by asking.** An owner that needs something from a teammate posts an **ask** to the board; a background **coordinator** spawns a short-lived helper to answer it, with hard termination guarantees — a per-run **ask ceiling (12)**, **ask depth cap (2)** so an answer that itself asks can't cascade forever, and at most **3 concurrent helpers**. The live **blackboard panel** shows asks and the shared board in real time.
- **Review round.** The Lead gathers an exec summary of the whole team's work and sends it back to each owner so they can revise their piece against what everyone else produced.
- **The reporter assembles.** A dedicated **reporter** archetype reads every piece and the full blackboard and writes **one coherent deliverable** — there is no weighted merge and no single "winner". The result renders as the **Final report** (shown as *assembled*, not *confidence*), with the generated files captured for download.
- **Learning.** Vatra scores not just the subtask owners but also the **helpers**, the **Lead's decomposition**, and the **reporter's assembly** (the last two as pseudo-archetypes), folding outcomes into per-archetype, per-domain reliability — so future decompositions and reporting improve.

Operational guarantees mirror Basna: a Vatra **worker can never start another run** (the run-starting tools are stripped from spawned agents and a `CLAW_VATRA_WORKER` env marker is double-checked), there's a **per-agent Skip** button on the live panel to drop a stuck specialist's turn and move on, and the deliverable is **persisted the moment the reporter finishes** so a late failure can't lose it.

**When to use which:** Basna for a single best answer from independent voices merged by reliability; Vatra for a *composed* deliverable whose parts are interdependent (a report, a plan, a spec) built by a collaborating team; Council for a multi-round *deliberation* you watch and steer — now with a panel it can assemble for you.

---

## Migration & compatibility

Additive — **backward compatible with 0.6.2**. Nothing changes unless you use the new flows.

- **Council auto-assemble** reuses your active **Library tier set** to choose the panelists' models (falling back to the registry tier defaults + the provider env var when a tier isn't configured), exactly like Basna and Vatra. Restart Flight Deck to pick up the two new `/fd/council/*` endpoints.
- **Existing councils** are untouched: the new fields default to "no temporary agents", so old sessions load and behave exactly as before.
- **Vatra** needs no setup beyond the same Library tiers Basna already uses; switch the compose mode on the Basna page.
- No databases removed, no schema migrations required.

See [release-notes/RELEASE_NOTES_0.6.2.md](release-notes/RELEASE_NOTES_0.6.2.md) for the previous release, or the [release-notes/](release-notes/) folder for the full history.
