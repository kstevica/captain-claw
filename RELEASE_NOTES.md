# Captain Claw v0.4.4 Release Notes

**Release title:** Insights, Nervous System (Dreaming), Computer Insights

**Release date:** 2026-03-21

## Highlights

This release introduces three major cognitive features: **Insights** — a persistent knowledge base auto-extracted from conversations; **Nervous System** — an autonomous "dreaming" layer that proactively synthesizes patterns across all memory types; and **Computer Insights** — insights integration in the Computer workspace. Together, these features give the agent durable factual memory and subconscious pattern recognition that bleeds across sessions.

## New Features

### Insights System

Persistent knowledge base auto-extracted from conversations — facts, contacts, decisions, deadlines, and other durable information:

- **Auto-extraction** — after every 8 messages, the agent reviews recent conversation and extracts structured insights with entity keys, categories, and importance ratings
- **Deduplication** — new insights are checked against existing ones via entity key matching and BM25 text similarity
- **FTS5 search** — full-text search powered by SQLite FTS5 virtual tables
- **Context injection** — relevant insights are automatically injected into the system prompt to inform future conversations
- **Web browser** — new dashboard at `/insights` with searchable, filterable cards, category badges, and detail editing
- **Slash commands** — `/insight`, `/insight search`, `/insight add`, `/insight delete`, `/insight stats`
- **REST API** — full CRUD at `/api/insights` with search, stats, and session isolation
- **Settings integration** — configurable via Settings UI (enabled, auto-extract, interval, cooldown, max items)
- **Session isolation** — in public mode, each session uses a separate database file
- **Home page card** — new Insights card on the homepage

### Nervous System (Dreaming)

Autonomous "dreaming" layer that proactively synthesizes across all memory types — working memory, semantic memory, deep memory, insights, and reflections:

- **Dream cycles** — background process samples ~2000 tokens across all memory layers, sends to LLM for pattern recognition, stores up to 3 intuitions per cycle
- **Four thread types** — connection (links between unrelated info), pattern (recurring themes), hypothesis (speculative inferences), association (thematic groupings)
- **Confidence scoring** — each intuition has a 0.0-1.0 confidence score with color-coded bars (green/yellow/red) in the UI
- **Importance ratings** — 1-10 scale affecting context injection priority
- **Decay system** — unvalidated intuitions lose 0.05 confidence per day after 7 days of inactivity; deleted below 0.1
- **Validation** — manually validate intuitions to permanently protect from decay, boost confidence +0.2 and importance +1
- **Session bleeding** — in admin mode, intuitions bleed across sessions with source session tracking and confidence bonuses; fully isolated in public mode
- **Context injection** — high-confidence intuitions surfaced in system prompt and per-turn context notes
- **Web browser** — new dashboard at `/intuitions` with stats bar, searchable list, confidence bars, dream trigger button, and detail modal
- **Slash commands** — `/intuition`, `/intuition dream`, `/intuition add`, `/intuition validate`, `/intuition delete`, `/intuition stats`
- **REST API** — full CRUD at `/api/nervous-system` with dream trigger, stats, and session isolation
- **Settings integration** — 14 configurable parameters via Settings UI (enabled, auto-dream, intervals, cooldowns, decay rates, thresholds)
- **Cost control** — disabled by default, 5-minute cooldown, 12-message interval, 800-token output cap, 200 intuition hard cap
- **Home page card** — new Nervous System card on the homepage

### Computer Insights

- **Insights in Computer** — insights auto-extraction now runs in the Computer workspace alongside chat
- **Automatic insights memory** — Computer sessions build up their own insights knowledge base

### Todo and Datastore in Public Computer

- **Session isolation** — todo lists and datastore tables are now fully isolated per session in public computer mode
- **Public access** — public users can use todo and datastore features without affecting other sessions

### Swarm Improvements

- **Todo and datastore session isolation** — swarm tasks respect session boundaries for todo and datastore operations

## Version Changes

- `captain_claw` — 0.4.3 → 0.4.4
- `botport` — 0.4.3 → 0.4.4
- `desktop` — 0.4.3 → 0.4.4

## Internal

### New Files
- `captain_claw/insights.py` — Insights manager with SQLite+FTS5, auto-extraction, dedup, context injection
- `captain_claw/nervous_system.py` — Nervous system manager with dream cycles, decay, validation, session bleeding
- `captain_claw/web/rest_insights.py` — REST API handlers for insights
- `captain_claw/web/rest_nervous_system.py` — REST API handlers for nervous system
- `captain_claw/instructions/dreaming_system_prompt.md` — LLM system prompt for dream synthesis
- `captain_claw/instructions/dreaming_user_prompt.md` — LLM user prompt template for dreams
- `captain_claw/instructions/micro_dreaming_system_prompt.md` — Compressed variant system prompt
- `captain_claw/instructions/micro_dreaming_user_prompt.md` — Compressed variant user prompt
- `captain_claw/web/static/insights.html` — Insights browser page
- `captain_claw/web/static/insights.js` — Insights browser JavaScript
- `captain_claw/web/static/insights.css` — Insights browser styles
- `captain_claw/web/static/intuitions.html` — Nervous system browser page
- `captain_claw/web/static/intuitions.js` — Nervous system browser JavaScript
- `captain_claw/web/static/intuitions.css` — Nervous system browser styles

### Modified Files
- `pyproject.toml` — Version 0.4.3 → 0.4.4
- `captain_claw/__init__.py` — Version bump
- `botport/pyproject.toml` — Version bump
- `botport/botport/__init__.py` — Version bump
- `desktop/package.json` — Version bump
- `captain_claw/config.py` — Added `InsightsConfig` and `NervousSystemConfig` classes with fields on `Config`
- `captain_claw/agent_context_mixin.py` — Added insights and nervous system cache refresh, context notes, system prompt blocks
- `captain_claw/instructions/system_prompt.md` — Added `{insights_block}` and `{nervous_system_block}` placeholders
- `captain_claw/instructions/micro_system_prompt.md` — Added same placeholders
- `captain_claw/web/chat_handler.py` — Added `maybe_extract_insights` and `maybe_dream` post-turn hooks
- `captain_claw/web/slash_commands.py` — Added `/insight`, `/insights`, `/intuition`, `/intuitions`, `/dream` command routing
- `captain_claw/web_server.py` — Added insights and nervous system REST wrappers, route registrations, COMMANDS entries, static page serving
- `captain_claw/web/static_pages.py` — Added `serve_insights` and `serve_intuitions` handlers
- `captain_claw/web/rest_settings.py` — Added insights and nervous_system sections to settings schema
- `captain_claw/web/static/home.html` — Added Insights and Nervous System cards
- `README.md` — Added insights and nervous system to feature table, advanced features, architecture table
- `USAGE.md` — Added Insights and Nervous System documentation sections, config reference, slash commands, dashboard pages
