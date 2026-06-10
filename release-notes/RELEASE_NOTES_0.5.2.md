# Captain Claw v0.5.2 Release Notes

**Release title:** Present from your glasses — deck control + a real file editor
**Release date:** 2026-06-07

A Flight-Deck-centric release. Two headline capabilities — **live presentation
control** (drive an agent-built HTML deck from your glasses, a phone, or
WhatsApp) and a **real in-browser file editor** (syntax highlighting, line
numbers, find, cursor memory) — plus **live narration**, a faster path for
trivial turns, and glasses polish. Fully additive and backward compatible.

---

## What's new

### Present a deck from anywhere
Ask an agent to build an HTML slide deck, then drive it live — no PDF, no
export. The deck, the Flight Deck big screen, the glasses view, a phone remote,
and WhatsApp all share one in-memory **channel**, so whatever you tap moves
every surface in lock-step.

- **Cast it** — in the Flight Deck file list, the new **⧉ deck-view** button
  opens a controllable presentation on a named channel (the channel is a live
  field in the green panel; the URL updates as you type). Files only cast on an
  explicit tap — selecting a file never auto-loads a deck.
- **Glasses** — a `⧉ ◀ ▶` present bar; the arrows advance both the big screen
  **and** the glasses' own view.
- **Phone remote** (`/deck/remote?c=<channel>`) — big prev/next, first/last,
  and a **go-to-slide** number box (clamped to the live total).
- **WhatsApp** — `next slide`, `previous slide`, `first slide`, `last slide`,
  and **`go to slide N`** (also `goto N`, `slide N`, `jump to slide N`); the
  reply tells you the current slide (`→ Slide 3 / 20`). Built for a live talk
  controlled hands-free from Meta Ray-Ban glasses.
- The deck's own on-page nav chrome is hidden in deck view; switching decks no
  longer reload-loops.

### A real file editor in Flight Deck
Agent text files (md, html, json, scripts, …) are no longer preview-only — the
file list gains an **Edit** (pencil) button that opens an editor:

- **Syntax highlighting** for markdown / HTML / CSS / JS / TS / JSON / Python /
  Bash / YAML — **and the Flow DSL**, with its own grammar.
- **Line numbers**, a scroll-synced gutter.
- **Find** (`⌘/Ctrl+F`) — case-insensitive, match count, `Enter` / `Shift+Enter`
  to step matches.
- **Cursor memory** — reopen a file and the caret returns to the **start of the
  row** you left off on.
- **Save** with `⌘/Ctrl+S`. The same editor now powers the **Flow builder's**
  code view.
- File-list rows were de-cluttered: the row actions (pin · view · deck · edit ·
  download) moved to their own line under the path.

### Live narration
The between-step "blurbs" an agent emits during a long task ("Now applying the
edits…") now **stream live to the channel the request came from** — web chat,
WhatsApp, and glasses — instead of only showing up in the final answer. You can
watch the work happen. (Gated by `STREAM_NARRATION`, on by default.)

### Faster trivial turns
- **Fast path** — a one-line request like *"change $2.2M to $4.5M"* now skips
  the contract → planner → planning-DAG → completion-gate pipeline (~10s of LLM
  round-trips) when it's a single, unambiguous edit or lookup. Multi-part
  requests still get the full pipeline.
- **Batch edits** — the `edit` tool applies multiple changes in one call and
  returns a closest-match hint when an `old_string` isn't found.
- **Fix** — `edit` infers `replace_string` when the action is omitted but
  `old_string`/`new_string` are present (removes a loop the model could fall
  into).

### Glasses polish
- Record/file content cards and the chat/datastore/files mode switcher are now
  focus-walkable, so everything is reachable hands-free.
- `/flow` control commands work from the web chat (no longer "Unknown command").

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild only if you build locally:
npm --prefix flight-deck run build
```

- **Restart the agent** — activates the fast path, the `edit` tool changes, and
  live narration.
- **Restart Flight Deck** — activates the deck channel bus and the WhatsApp
  slide commands.
- Static deck/glasses pages and the committed UI just need a hard reload
  (`Cmd+Shift+R`).

Backward compatible with 0.5.1 — everything here is additive.
