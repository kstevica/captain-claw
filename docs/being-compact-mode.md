# Iskra — Compact mode (per-being lean instructions + lean body)

*Shipped 2026-07-16. Panel toggle: Beings page → being card → "prompts: Full / Compact".*

## What the 40k actually was

Measured on the pilots (llm_usage, Zvjezdana on deepseek-v4-flash, Ada on Qwen3.6-35B):

| Component | Fresh being | After 3 days |
|---|---|---|
| Body system prompt + tool schemas | ~6–7k tokens | same |
| Accumulated session history | ~0 | **~25–28k** |
| Tick prompt (all data + instructions) | ~1–1.5k | ~1.5–2.5k |
| **Per-call input** | **~9k** | **~34–36k** |

A faculties tick is 2–5 such calls (orient → act × gates → talk → journal → connect), so a mature being's tick runs 40k–100k+ input tokens. The instruction *text* is a small slice; the weight is the general-assistant system prompt and, above all, **history replayed into every call** — redundant by design, because a being's continuity lives in its home files (journal tail + manifest are re-injected fresh each tick).

## What Compact does (three levers, one toggle)

1. **Compact instruction set** — every being-facing instruction now lives in
   `captain_claw/instructions/beings/*.md` (full set, verbatim original prose)
   with `compact_*.md` siblings (same narrative beats, same physics and
   honesty rules, fewer words). `being_prompts.render()` picks the set from
   `beings.compact_mode`; a missing compact variant falls back to the full
   file. Files are mtime-cached — edits land on the next tick, no restart.
2. **Lean body (eco flag)** — toggling Compact writes/removes the body's
   `eco_mode.txt`, so the agent builds its system prompt from the existing
   micro instruction set (~4KB template vs ~30KB) with lazy tools. Runtime
   file, applies on the body's next prompt build.
3. **Capped context** — the body respawns with `max_context = 24_000`
   (`COMPACT_BODY_MAX_CONTEXT`), which bounds the history slice per call and
   keeps compaction ahead of drift. Applied at spawn; a toggle respawns an
   alive body (torpid ones converge on their next respawn).

Expected per-call input in Compact at steady state: **~18–22k vs ~36k** (and
no unbounded growth toward the 160k default). Template-only savings are
honest but small (3–12% of the tick prompt — most of it is live data, not
prose); the eco flag and the context cap carry the bulk.

## Wiring

- DB: `beings.compact_mode` (additive migration, default 0 = Full).
- Store: `set_compact_mode()` → event `compact_set`; surfaced in `vitals`.
- Route: `POST /fd/beings/{slug}/compact {"on": bool}` → store + eco flag +
  respawn-if-alive (same pattern as `/stage` and `/body-archetype`).
- Engine: `being_life.spawn_body` applies the cap + flag per the being;
  `being_prompts.py` is the loader; `being_life.py` / `being_mind.py` compose
  every prompt from the external files for BOTH modes.
- Frontend: `BeingCard` "prompts" selector → `setCompactMode()`
  (`flight-deck/src/services/beings.ts`); bundle rebuilt.
- Tests: `tests/test_flight_deck/test_being_compact.py` (9) — store/vitals,
  external-file loading, fallback, brace-safe rendering, same-contract
  smaller prompts, gates intact. Full being suite: 242 passing.

## Tuning the narrative

Both sets are plain markdown with `{placeholder}` slots, personal-override
friendly by design review (`~/.captain-claw/instructions/` is NOT consulted —
FD reads the repo folder directly, same convention as vatra/basna/code).
To tune a being's voice pressure-free: edit the file, watch the next tick.
Key files: `wake_task.md`, `dream_task.md`, `digest_contract.md`,
`orient_task.md`, `act_task.md`, `talk_task.md`, `journal_*.md`,
`write_gate.md`, `visitors_frame.md`, `mind_*.md`, `self_mod_offer.md`,
`procreate_offer.md`.
