# Jarvis — Phase 6: Open the platform + quality

The closed loop works end-to-end (notice → decide → act → judge → learn), and
delivery reaches web + WhatsApp. What's left is **reach and quality**, in four
themes. The first one — making tools/senses/actions/sources *pluggable* — is the
keystone: it's the agent-native move, and the other three get easier once it
exists (more real actions to prepare, more sources to learn from and de-noise).

Today the **action catalog is a hardcoded Python dict** (`action_catalog.py`,
4 reversible + 4 human-only) and the **senses are two hardcoded pollers**
(`event_sources_google.py`: calendar, gmail). To be the platform, those must be
open — extended by configuration, not code.

---

## Theme A — Pluggable tools / senses / actions / sources  *(keystone)*

**Goal:** add new autonomous **actions** (hands) and **event sources** (senses)
without editing code, by *promoting the agent's already-connected tools*. The
agent (captain-claw-web) already holds authenticated tools — Google
Calendar/Mail/Drive, Granola, etc. Those become the menu.

- **A1 · Tool discovery.** FD fetches an agent's live tool list (the orchestrator
  endpoint already returns `{tools}` from `server.agent.tools.list_tools()`),
  cached per agent. This is the menu the UI promotes from.
- **A2 · Pluggable actions.** A per-user `custom_actions` store, catalog-shaped
  (`{id, label, tool, arg_map/required, risk, reversibility, reverse_tool,
  grant, human_only, enabled}`). `list_catalog()` merges built-in + custom, so
  the arbiter and trust ladder treat them identically. UI "Actions" panel:
  promote a tool → action (set label, risk, reversibility, optional reverse
  tool, human-only). **Safety:** the hard-exclude list (shell / browser /
  social / payments) stays enforced centrally; a user-added action defaults to
  *propose / human-only* and can only auto-fire after it graduates via the
  existing trust ladder. Never a bypass.
- **A3 · Pluggable senses.** A per-user `custom_sources` store
  (`{id, label, tool, args, interval_seconds, dedup_field, summary_template,
  enabled}`). A generic **tool-poller adapter** in `event_sources.py` calls a
  read/list tool on a cadence (via the existing `run_tool_on_agent` rail) and
  maps each result row to an `external_event` (dedup_key from a chosen id field,
  summary from a template). UI "Sources" panel: promote a read tool → sense
  (args/query, interval, which field is the id, which fields form the summary).
  Sources default **off**.
- **A4 · Guardrails.** One central allow/deny policy: hard-excludes, the
  reversibility/risk requirement for auto-fire, and a cap on custom sources so a
  cheap poller can't hammer an API.

## Theme B — Prepare real actions  *(depends on A)*

Right now the loop narrates (nudges + self-notes) more than it *does*. Bias the
arbiter toward concrete catalog actions:
- important email needing a reply → propose `mail.draft` (a drafted reply, human
  sends) instead of just a nudge;
- a gap before/after a meeting → `calendar.hold`;
- a stated commitment → `reminder.schedule`.
Done via arbiter prompt + a couple of worked examples, so a useful action lands
*prepared* for one-tap approval, not just announced.

## Theme C — Cut the noise + learn

- **Stop self-referential busywork.** The arbiter is writing `note.write`
  "observation" notes about its own quiet/stability ("Document stability signal
  observation"). Tighten the prompt: `note.write` is only for user-relevant
  facts, never journaling the system's own state.
- **Gmail filter tuning.** Drop automated / no-reply senders (e.g. Google Docs
  comment notifications) before they ever become candidates.
- **Learn from dismissals.** A follow-up *dismiss* or a nudge *reject* feeds
  reliability, so the loop stops resurfacing the kinds/sources you keep dropping
  — the negative signal it currently ignores.

## Theme D — Reliability polish

- **Replan-on-failure (#4).** A failed plan step re-decomposes the remainder
  instead of failing the whole plan.
- **Webhook push (#2).** Sub-minute event latency via a push receiver
  (Gmail push / PubSub, or a generic webhook ingest) instead of 5-min polling.
- **Guardrail hardening** and audit of the new pluggable surface.

---

## Sequencing

**A → B → C (interleavable quick wins) → D.** A is the foundation; B is the
biggest user-visible payoff and needs A; C is small and improves daily feel
immediately; D makes it sturdy. Build per-phase, verify on prod between phases.

---

## STATUS — all four themes shipped (on main)

- **A** — `CustomAction`/`CustomSource` on the per-user config; `resolve_catalog`
  merges built-ins + custom (never shadows a built-in, drops hard-excluded);
  generic tool-poller source; agent-tool discovery; **table-driven grounding**
  (built-ins + custom fetch contract); "Tools & Sources" UI panel.
- **B** — arbiter biased to *prepare* the action (draft/hold/reminder) over a
  nudge; `mail.draft` enriched with the event's real sender/subject.
- **C** — no self-journaling notes; Gmail no-reply/automated filter; follow-up
  done/dismiss feeds reliability.
- **D** — replan-on-failure (bounded); token-gated `/fd/events/webhook` push;
  custom-source interval floor + count cap.
