# Captain Claw v0.5.6 Release Notes

**Release title:** Archetypes & durable councils — model tiers, action points, and sessions that don't lose work
**Release date:** 2026-06-14

A Flight Deck release focused on **Agent Forge** and **Agent Council**. Forge gains a curated **archetype library** and a per-tier **model configuration** (so model choices live in one place). Council becomes something you can run for real work: it **extracts per-agent action points** you can push into each agent's todos/intentions, lets you **restart a round** and **recover an interrupted session**, surfaces the model's reasoning in the activity log — and a persistence rewrite means a long session no longer silently loses messages when the access token rotates. Additive and backward compatible with 0.5.5.

---

## What's new

### Agent Forge — archetype library + model tiers
- **Archetype gallery.** ~20 curated, ready-to-spawn agents (Deep Researcher, Code Reviewer, Software Architect, Project Coordinator, Deal Screener, …) grouped by family, each a tuned role + cognitive mode + toolset + tier + fleet-instructions. Start a team from templates, or let the generator adapt them — the decomposition prompt is biased toward the catalog. Served from `instructions/archetypes.json` via `GET /fd/archetypes`.
- **Model tiers.** LLM Settings is now a per-tier editor — **Reasoning / Balanced / Fast / Long context** — each with its own provider, model, API key, base URL, and input/output context length. Pick which tier runs the decomposition. Every agent is assigned a tier and resolves to that concrete model at spawn, so a model release or reprice is a single edit instead of touching every agent. Settings persist per-user (multi-tenant) via `/fd/settings`.

### Agent Council — action points
After synthesis, each agent extracts **its own outstanding next steps** — scoped to its part of the discussion (no full-transcript dump), so the context size never matters. Each point is a self-contained brief (Context / Task / Done-when / Refs), classified **todo** or **intent**. **Send** records it into that agent's `todo`/`intentions` with the full detail preserved, and the button stays **Recorded** across reloads so you can't double-send.

### Agent Council — restart & recover
- **Restart round** — abort the current round, discard its messages, and re-run it from the top (handy when an agent goes off the rails).
- **Interrupted-session recovery** — a session reopened mid-round shows a clear *"Round N was interrupted"* banner (Restart / Synthesize / Conclude) instead of a stale "in session" state that isn't actually running.

### Agent Council — visibility & control
- **Narration in the activity log** — the model's between-step narration and reasoning are surfaced live, for debugging long or looping turns.
- **Allow delegation toggle** (off by default) — agents can't use orchestration tools (`flight_deck` / `task_contract` / `consult_peer`) during a turn, since the council itself is the coordination layer; turn on for planning sessions that farm out subtasks.
- **Stuck auto-nudge** — if an agent returns the canned *"I got stuck"* reply, the council nudges it to continue (up to 3×) before moving on, and holds the next speaker until it does.
- **Longer agent turns** — the per-turn wait is now generous (60 min) for slow/reasoning models doing real tool work; Restart round is the escape hatch.
- Four more session types — **Interview, Troubleshoot, Critique, Freeform**.

### Other
- **Group/role filter modal** — the Agent Desktop filter is now a searchable modal (Groups + Roles) instead of an overflowing chip row.
- **Light-theme readability** — Council Synthesis, Action Points, and the filter modal render correctly in light mode.

---

## Reliability fixes

- **No more silent data loss mid-council.** Council writes used the access token with no refresh-retry and swallowed failures — so once the token expired during a long session, every message/vote/status update was dropped (round 1 saved, then nothing). All council writes now refresh-and-retry on 401 and fall back to an **in-memory retry queue**; failures are logged, not silent.
- **Synthesis can't get stuck.** `requestSynthesis` reconnects agents first and only enters the `synthesizing` state once a synthesizer is reachable — and persists `concluded` on completion, so a finished session survives a reload.
- **Council connections don't flap.** Agents now connect using their **current** token/port (read live from the container/process store), not the values snapshotted when the session was created — fixing the connect/disconnect storm and "no agents reachable" on reopened sessions. A dropped socket is revived instead of skipped.
- **Action-point extraction can't run agents away.** Extraction turns are sent with tools disabled, so an agent describes its action points instead of delegating/executing them.
- **Stuck-marker single source.** The agent's canned "give-up" messages live in one place (`captain_claw/agent_stuck.py`) and the council fetches the detection markers via `GET /fd/council/stuck-markers`, instead of duplicating the strings.

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild the frontend only if you build locally:
npm --prefix flight-deck run build
```

- **Restart the agent** (`captain-claw-web`) for the per-turn `no_tools`/`deny_tools` handling, the stuck-markers endpoint, and the agent-loop changes.
- **Restart the Flight Deck server** for `GET /fd/archetypes`, the council restart/extract message routes, and the stuck-markers route.
- **Hard-reload Flight Deck** for the archetype gallery, model tiers, action points, restart/recover, and the light-theme fixes.
- **Desktop app:** `./build-desktop.sh` — everything above is in the rebuilt bundle.

Backward compatible with 0.5.5 — everything here is additive or a fix; no configuration or API changes.
