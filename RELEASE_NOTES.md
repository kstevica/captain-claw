# Captain Claw v0.4.5 Release Notes

**Release title:** Brain Graph, Process of Thoughts, Nuke Reset

**Release date:** 2026-03-22

## Highlights

This release introduces the **Brain Graph** — a 3D interactive visualization of the agent's entire cognitive topology, rendered with Three.js and force-directed physics. Every insight, intuition, task, todo, contact, session, and message is a node; every provenance link is an edge. Combined with the new **Process of Thoughts** traceability layer (message-level IDs and cross-system lineage), you can now see and traverse how a conversation message became an insight, triggered a dream cycle intuition, spawned a sister session task, and produced a briefing — all in an interactive 3D graph with live WebSocket updates.

## New Features

### Brain Graph Visualization

Interactive 3D force-directed graph at `/brain-graph`, built on Three.js + 3d-force-graph (CDN-loaded, no build step):

- **9 node types** with distinct 3D shapes — sessions (wireframe spheres), messages (tetrahedrons), insights (spheres), intuitions (spheres), tensions (icosahedrons), tasks (boxes), briefings (cones), todos (octahedrons), contacts (dodecahedrons), cognitive events (small spheres)
- **7 edge types** with directional arrows — contains, sequence, supersedes, resolves, triggers, parent, source
- **Dynamic session spheres** — wireframe spheres auto-resize every frame to enclose their furthest child node with 20% padding
- **WebSocket live updates** — new insights and intuitions appear in real-time as they're created during conversation
- **Detail panel** with prev/next navigation — click a node to see metadata, connections, and step through linked nodes with arrow keys
- **Connections list** — clickable list of all incoming/outgoing edges for the selected node
- **Full content modal** — "Show full content" button fetches complete message text from the session DB via REST API and renders as markdown
- **Search and filters** — text search highlights matching nodes; per-type checkboxes toggle visibility; node limit slider (20-500)
- **Deep linking from chat/computer** — brain button on assistant messages opens `/brain-graph?focus_ts=<timestamp>`, auto-selecting and zooming to that node
- **Keyboard navigation** — left/right arrows or `[`/`]` step through connected nodes, Escape closes panels
- **Public mode** — fully supported with session-isolated insights and intuitions
- **REST API** — `GET /api/brain-graph` (full graph data), `GET /api/brain-graph/message/{msg_id}` (full message content)
- **Home page card** — new Brain Graph card with web icon on the homepage

### Process of Thoughts (PoT) Traceability

Full lineage tracking across all cognitive subsystems via provenance IDs:

- **Message IDs** — every message in `Session.add_message()` gets a unique 12-char hex `message_id`; method now returns the ID
- **Insight provenance** — new `source_message_id` column tracks which user message triggered extraction; new `supersedes_id` column tracks insight evolution when entity_key dedup fires (creates new insight linking to predecessor instead of silent update)
- **Intuition provenance** — new `source_message_id` column tracks which user message was active during the dream cycle
- **Todo hierarchy** — new `parent_id` column for subtask relationships; new `triggered_by_id` column linking to the insight or intuition that created the todo
- **Schema migrations** — all new columns use safe `ALTER TABLE ADD COLUMN` with try/except pattern, fully backward compatible (existing data gets NULL)

### Nuke Command — Full Cognitive Reset

The `/nuke` command now clears all cognitive subsystems in addition to workspace files, sessions, and entities:

- **Insights** — `InsightsManager.clear_all()` deletes all insights
- **Intuitions** — `NervousSystemManager.clear_all()` deletes all intuitions
- **Cognitive metrics** — `CognitiveMetricsManager.clear_all()` deletes all events and snapshots
- **Sister sessions** — `SisterSessionManager.clear_all()` deletes all proactive tasks, briefings, watches, and daily budget
- **Updated confirmation message** — lists all new items being cleared
- **Per-section error handling** — each subsystem reports its own status

### Brain Button in Chat and Computer

- **Chat** — brain icon button added next to like/dislike on every assistant message; opens Brain Graph in a new tab focused on that message
- **Computer** — same brain button in the answer action bar next to copy/like/dislike

## Version Changes

- `captain_claw` — 0.4.4 → 0.4.5
- `botport` — 0.4.4 → 0.4.5
- `desktop` — 0.4.4 → 0.4.5

## Internal

### New Files
- `captain_claw/web/rest_brain_graph.py` — Brain Graph REST API handler aggregating all cognitive data sources into unified graph format with WebSocket broadcast helper
- `captain_claw/web/static/brain-graph.html` — Brain Graph page with CDN imports (Three.js, 3d-force-graph, marked.js)
- `captain_claw/web/static/brain-graph.css` — Dark theme styling for graph viewport, controls panel, detail panel, content modal, navigation buttons
- `captain_claw/web/static/brain-graph.js` — 3D force-directed graph initialization, data fetching, node rendering with typed shapes, WebSocket live updates, prev/next navigation, content modal, search/filter, dynamic session sphere sizing, auto-focus from URL params

### Modified Files
- `pyproject.toml` — Version 0.4.4 → 0.4.5
- `captain_claw/__init__.py` — Version bump
- `botport/pyproject.toml` — Version bump
- `botport/botport/__init__.py` — Version bump
- `desktop/package.json` — Version bump
- `captain_claw/session/__init__.py` — Added `message_id` to Message dataclass and `add_message()` (returns ID); added `parent_id` and `triggered_by_id` to TodoItem with schema migration, updated `_TODO_COLS`, `from_row()`, `to_dict()`, `create_todo()`
- `captain_claw/insights.py` — Added `source_message_id` and `supersedes_id` columns with schema migration; updated `add()` signature; changed entity_key dedup to create new insight with `supersedes_id` instead of silent update; threaded `source_message_id` through `extract_insights()`; added `clear_all()` method; updated `_row_to_dict()` columns; added Brain Graph broadcast hook
- `captain_claw/nervous_system.py` — Added `source_message_id` column with schema migration; updated `add()` signature; threaded `source_message_id` through `dream()`; added `clear_all()` method; updated `_row_to_dict()` columns; added Brain Graph broadcast hook
- `captain_claw/cognitive_metrics.py` — Added `clear_all()` method for events and snapshots
- `captain_claw/sister_session.py` — Added `clear_all()` method for tasks, briefings, watches, and daily budget
- `captain_claw/web/slash_commands.py` — Updated nuke confirmation message; added clearing of insights, intuitions, cognitive metrics, and sister session data in `_execute_nuke()`
- `captain_claw/web/static_pages.py` — Added `serve_brain_graph()` handler
- `captain_claw/web_server.py` — Added `_serve_brain_graph()`, `_bg_get_data()`, `_bg_get_message()` methods; registered `/brain-graph`, `/api/brain-graph`, `/api/brain-graph/message/{msg_id}` routes
- `captain_claw/web/static/app.js` — Added brain graph button to assistant message feedback row
- `captain_claw/web/static/computer.js` — Added brain graph button to answer action bar
- `captain_claw/web/static/home.html` — Added Brain Graph card with web icon
- `README.md` — Added Brain Graph and Process of Thoughts to feature table, Web UI line, Advanced Features, and Architecture table
- `USAGE.md` — Added Brain Graph and Process of Thoughts documentation sections with node/edge type tables, API reference, and thought chain examples
