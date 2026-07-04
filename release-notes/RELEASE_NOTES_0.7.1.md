# Captain Claw v0.7.1 Release Notes

**Release title:** Quick Chat, guided setup, and a tidier Flight Deck
**Release date:** 2026-07-04

A focused Flight Deck UX release on top of 0.7.0's Code studio. Everything is additive and
backward compatible with 0.7.0 — no schema changes, no backend changes.

## Highlights

- **Quick Chat (new Workspace page).** Pick an archetype and start talking to it immediately.
  The agent spawns **hidden** from the Agent Desktop; you chat with it through the normal chat
  panel (plan mode, next steps, attachments — the full experience), and a single **Promote to
  desktop** button reveals it on the canvas when you want it there. The chat only opens after a
  server-side readiness probe (`/fd/probe`), so a not-yet-listening agent no longer greets you
  with a "Connection failed" bubble.

- **Library setup wizard.** A modal that takes one model — provider, model id, API key, base
  URL — plus **context presets** (input 128k–1M, output 16k–256k; power-of-2 based, defaulting
  to 256k in / 32k out) and applies it to **every tier** in a named tier set. It **auto-opens on
  a fresh install** (when no tier set is configured yet), and is always reachable from a **Setup
  wizard** button. Finishing the first-launch wizard drops you straight onto Quick Chat.

- **Reorganized sidebar.** The flat navigation is grouped into meaningful sections — Workspace,
  Build, Multi-Agent, Automation, Knowledge, Experimental, System, and Play — so the growing
  page list is scannable again.

- **Library, redesigned.** Split into **Model Tiers** and **Archetypes** tabs. Each tier card is
  reorganized into identity / capacity / connection groups, and the Tier Sets and Additional API
  Keys blocks share one consistent icon-chip + accent style.

- **Light-theme polish.** Contrast fixes across the Code page and Library — primary-button hover
  states, input borders, and segmented controls now read correctly in light mode.

## Notes

- Frontend-only: two small stores were added (Quick Chat tray, a deterministic desktop
  hide/reveal), reusing the existing agent-spawn and chat plumbing.
- Backward compatible with 0.7.0.
