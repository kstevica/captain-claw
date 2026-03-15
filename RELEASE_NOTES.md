# Captain Claw v0.4.0 Release Notes

**Release title:** Computer — Retro-Themed Research Workspace, Live Instructions, Personality Editor

**Release date:** 2026-03-15

## Highlights

This release introduces **Computer**, a retro-themed research workspace at `/computer` with visual generation, exploration trees, 14 built-in themes, and a custom theme engine. A new `/btw` command lets you inject live instructions while tasks are running across Chat, Computer, and Telegram. The personality and personas editor moves to its own dedicated page, and reflections become editable.

## New Features

### Computer — Research Workspace (`/computer`)

A three-panel workspace designed for extended research sessions and visual exploration:

- **Three-panel layout** — Resizable input area + activity log (left) and tabbed output (right: Answer, Blueprint, Files, Visual, Map)
- **14 built-in themes** — Amiga Workbench, Atari ST GEM, C64 GEOS, Classic Mac, Windows 3.1, Hacker Terminal, Modern, Windows 11, macOS, iPhone, Android, Nokia 7110, Nokia Communicator, plus a default theme. Each has unique boot sequences and CSS variable styling (39 custom properties)
- **Custom theme engine** — Download JSON template, edit colors/fonts/boot sequence, upload to create your own theme
- **Visual generation** — LLM generates themed HTML output matching the active theme (Amiga bevels, hacker terminal glow, etc.). Token tier selector controls generation budget (4K–32K tokens)
- **Exploration tree** — Click explore links in generated visuals to branch into multi-turn research. Nodes persist to SQLite, visualized in the Map tab with zoom controls
- **Model selector modal** — Grid of available models with provider icons, descriptions, and pricing. Selected model applies to all LLM operations (chat, visual generation, exploration), not just visual generation
- **Folder browser** — Modal with Local and Google Drive tabs. Browse directories, add folders for agent file access, drive selector for Windows
- **Image/file attachments** — Paste, drag-drop, or file picker. Images auto-resize to 1024px. Supports PNG, JPG, WEBP, GIF, BMP, CSV, XLSX
- **Panel resize persistence** — Left panel width saved to localStorage and restored on load
- **Activity log** — Real-time timestamped entries with type icons (system, user, /btw, tool, thinking, error). Max 200 entries with auto-trim
- **Input decomposition** — Automatic analysis showing identified actions, targets, and complexity

### Live Instructions (`/btw` Command)

Inject additional context or course corrections while the agent is working on a task:

```
/btw use bullet points for the summary
btw also include error counts
```

- Works in **Chat**, **Computer**, and **Telegram**
- Multiple `/btw` messages accumulate during a task
- Instructions are applied to all remaining subtasks
- Cleared automatically when the task completes
- In Telegram, processed before the per-user lock so it works even while the agent is busy

### Personality Editor Page (`/personality`)

- Dedicated full-screen page for editing agent personality and per-user profiles
- Extracted from the home page into a standalone route
- Split-pane layout: persona list (left) and editor form (right)
- Full CRUD for agent personality and user personas
- Rephrase & Enrich buttons on textarea fields
- Home page card added after Computer card for navigation

### Editable Reflections

- Reflections on the `/reflections` page are now editable
- Edit button on each reflection card opens inline editor with textarea for summary and input for topics
- Save persists changes to the markdown file and invalidates cache
- New `PUT /api/reflections/{timestamp}` REST endpoint

## Improvements

### Model Selection Scope
- Selected model in Computer now applies to **all** LLM operations (chat, visual generation, exploration), not just visual generation
- Model is re-applied on WebSocket reconnect via `set_model` message
- Provider is correctly logged in LLM usage records from Computer operations

### Computer Activity Log
- Type-specific icons: ● system, ▶ user, 💡 /btw, ⚙ tool, 💭 thinking, ✖ error
- Timestamps in locale-aware HH:MM:SS format
- Auto-trim at 200 entries

## REST API Changes

### New Endpoints
- `PUT /api/reflections/{timestamp}` — Update a reflection's summary and/or topics
- `POST /api/computer/visualize` — Generate themed HTML from prompt + result
- `POST /api/computer/exploration` — Save exploration tree node
- `GET /api/computer/exploration` — List exploration nodes for session
- `GET /api/computer/exploration/{id}` — Get single exploration node
- `PUT /api/computer/exploration/{id}/visual` — Update visual HTML for node
- `DELETE /api/computer/exploration/{id}` — Delete exploration node

### New Routes
- `GET /computer` — Computer workspace page
- `GET /personality` — Personality editor page

## Web UI Changes

- New `/computer` page — full retro-themed research workspace
- New `/personality` page — dedicated personality and personas editor
- Home page: Computer card added, Personality card added after Computer, Reflections card moved after Personality
- Reflections page: edit button with inline editor for summary and topics
- Dashboard pages table updated with Computer, Personality, and Reflections entries

## Configuration Changes

No new configuration keys. Computer uses existing `model.allowed` for the model selector and existing folder/GDrive APIs for the folder browser.

## Internal

### New Files
- `captain_claw/web/static/computer.html` — Computer workspace HTML
- `captain_claw/web/static/computer.js` — Computer client-side logic (~3,100 lines)
- `captain_claw/web/static/computer.css` — Theme engine and Computer styling (~2,950 lines)
- `captain_claw/web/static/personality.html` — Standalone personality editor page
- `captain_claw/web/rest_computer.py` — Visual generation and exploration REST handlers
- `captain_claw/instructions/computer_visualize_system_prompt.md` — Visual generation system prompt
- `captain_claw/instructions/computer_visualize_user_prompt.md` — Visual generation user prompt

### Modified Files
- `web_server.py` — Computer routes, personality route, reflection update route, `/btw` WebSocket handler
- `ws_handler.py` — `btw` message type handler
- `static_pages.py` — `serve_computer()`, `serve_personality()`
- `rest_reflections.py` — `update_reflection_api()` function
- `telegram.py` — `/btw` command handler (pre-lock), btw cleanup in finally block
- `home.html` — Computer and Personality navigation cards, personality editor removed
- `home.css` — Personality section styles removed (moved to personality.html)
- `reflections.html` — Edit UI with textarea, topic input, save/cancel buttons
- `app.js` — `/btw` detection and WebSocket send
- `pyproject.toml` — Version 0.3.5 → 0.4.0
- `captain_claw/__init__.py` — Version 0.3.5 → 0.4.0
