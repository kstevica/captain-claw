# Captain Claw v0.4.1 Release Notes

**Release title:** BotPort Swarm, PDF Export, Persona Selector, Extended File Attachments

**Release date:** 2026-03-16

## Highlights

This release introduces **BotPort Swarm** — DAG-based multi-agent orchestration through BotPort. Decompose complex goals into task graphs with dependencies, route tasks to specialist agents, and execute with approval gates, retry policies, checkpoints, and file transfer. The Computer workspace gains **PDF export** via WeasyPrint (preserving full CSS styling), a **persona selector**, and support for **additional file attachments** (PDF, DOCX, XLSX, PPTX, MD, TXT, CSV alongside images). Theme consistency improvements ensure all input action buttons match the active theme.

## New Features

### BotPort Swarm

DAG-based multi-agent orchestration built on top of BotPort's routing layer:

- **Task decomposition** — LLM-powered decomposer breaks goals into task graphs with dependencies
- **Agent designer** — LLM analyzes each task and assigns optimal persona + model tier (fast/mid/premium)
- **Swarm engine** — Advances DAG, launches tasks whose dependencies are satisfied, routes via BotPort concerns
- **Error policies** — `fail_fast`, `continue_on_error`, `manual_review`
- **Approval gates** — Require human approval before executing selected tasks
- **Retry with backoff** — Configurable retry count, backoff duration, and fallback persona
- **Timeout escalation** — Three-stage system: warning at 80%, extension at 100% (+50%), failure after extension
- **Checkpointing** — Save and restore swarm state for re-execution or rollback
- **File transfer** — Inter-agent file transfer over WebSocket (gzip + base64, up to 50 MB per file), with SHA-256 hashes
- **Cron scheduling** — Recurring swarms via cron expressions (standard 5-field format)
- **Audit logging** — Complete event trail for every swarm action
- **Projects** — Organize swarms into named projects
- **Visual dashboard** — DAG canvas with status colors and dependency arrows, task monitoring, file manager, approval UI

New files: `botport/swarm/` (engine, store, dag, decomposer, agent_designer, scheduler, file_manager, models), `botport/dashboard/swarm_routes.py`, `botport/dashboard/static/swarm.js`

### Computer — PDF Export

- **Visual tab PDF button** — Export the generated visual HTML to PDF via WeasyPrint, preserving all CSS styling (backgrounds, fonts, colors, tables, code blocks, emojis)
- **File preview PDF button** — HTML files previewed in the file modal now have a PDF export button in the title bar
- Print-friendly CSS injected automatically: A4 page size, color-adjust, page-break avoidance, overflow prevention
- PDF filename derived from the task prompt (e.g., `analyze-quarterly-results.pdf`) or HTML filename
- Replaces previous fpdf2 implementation — no more font/Unicode/layout issues

### Computer — Persona Selector

- New 👤 persona button in the Computer title bar (alongside Theme, Model, Tier)
- Modal grid showing all available personas (agent personality + per-user profiles)
- Persona selection persisted to localStorage

### Computer — Extended File Attachments

- File picker now accepts PDF, DOCX, XLSX, PPTX, MD, TXT, CSV files in addition to images
- All supported file types can be attached to prompts via the 📎 button, drag-drop, or paste

## Improvements

### Theme-Consistent Input Buttons

- 📎 Attach and 📁 Folder buttons now match `#send-btn` styling across all themes
- Base style updated to use theme variables (`--bevel`, `--chrome-hi`, `--chrome-lo`, `--radius`) matching Send button
- Per-theme overrides added for: Hacker Terminal (green glow + border), Modern (rounded, no border, scale effect), Windows 11 (rounded, no border), macOS (rounded, font-weight 500), iPhone (pill shape, border-radius 18px), Android (pill shape, border-radius 20px)

### Agent & BotPort Optimizations

- Agent and BotPort performance optimizations and bug fixes
- Swarm file transfer protocol for inter-agent file sharing

## Dependencies

- Added `weasyprint>=60.0` (replaces `fpdf2>=2.8.0`)
- WeasyPrint requires system libraries: `pango`, `cairo`, `gdk-pixbuf` (install via `brew install pango` on macOS)

## REST API Changes

### New Endpoints (BotPort)

- `GET /api/swarm/projects` — List all projects
- `POST /api/swarm/projects` — Create a project
- `GET/PUT/DELETE /api/swarm/projects/{id}` — Project CRUD
- `GET /api/swarm/swarms` — List swarms
- `POST /api/swarm/swarms` — Create a swarm
- `GET/PUT/DELETE /api/swarm/swarms/{id}` — Swarm CRUD
- `POST /api/swarm/swarms/{id}/start` — Start a swarm
- `POST /api/swarm/swarms/{id}/pause` — Pause a swarm
- `POST /api/swarm/swarms/{id}/resume` — Resume a swarm
- `POST /api/swarm/swarms/{id}/cancel` — Cancel a swarm
- `POST /api/swarm/swarms/{id}/decompose` — Decompose goal into tasks
- `POST /api/swarm/swarms/{id}/design-agents` — Design agent specs
- `POST /api/swarm/swarms/{id}/checkpoints` — Create checkpoint
- `POST /api/swarm/swarms/{id}/checkpoints/{cp}/restore` — Restore checkpoint
- `POST /api/swarm/tasks/{id}/approve` — Approve pending task
- `POST /api/swarm/tasks/{id}/reject` — Reject pending task
- `GET /api/swarm/swarms/{id}/files` — List swarm files
- `POST /api/swarm/swarms/{id}/files` — Upload file
- `GET /api/swarm/swarms/{id}/files/{name}` — Download file
- `GET /api/swarm/swarms/{id}/audit` — Get audit log

### Existing Endpoints

- `POST /api/computer/export-pdf` — Now uses WeasyPrint instead of fpdf2

## Version Changes

- `captain_claw` — 0.4.0.1 → 0.4.1
- `botport` — 0.3.4.1 → 0.4.1 (synced with captain_claw)
- `desktop` — 0.3.41 → 0.4.1

## Internal

### New Files (BotPort Swarm)
- `botport/botport/swarm/__init__.py` — Swarm module
- `botport/botport/swarm/models.py` — Data models (Swarm, SwarmTask, SwarmEdge, SwarmProject, etc.)
- `botport/botport/swarm/engine.py` — Swarm orchestration engine
- `botport/botport/swarm/store.py` — Async SQLite persistence
- `botport/botport/swarm/dag.py` — DAG validation, cycle detection, topological sort
- `botport/botport/swarm/decomposer.py` — LLM-based task decomposition
- `botport/botport/swarm/agent_designer.py` — LLM-based agent spec generation
- `botport/botport/swarm/scheduler.py` — Cron-based swarm scheduler
- `botport/botport/swarm/file_manager.py` — File storage and transfer
- `botport/botport/dashboard/swarm_routes.py` — Swarm REST API routes
- `botport/botport/dashboard/static/swarm.js` — Swarm dashboard UI

### Modified Files
- `pyproject.toml` — Version 0.4.0.1 → 0.4.1, weasyprint replaces fpdf2
- `captain_claw/__init__.py` — Version bump
- `botport/botport/__init__.py` — Version bump (synced to 0.4.1)
- `desktop/package.json` — Version bump
- `captain_claw/web/rest_computer.py` — PDF export rewritten with WeasyPrint
- `captain_claw/web/static/computer.html` — PDF button in file preview modal, persona selector
- `captain_claw/web/static/computer.js` — File preview PDF export, persona selector, extended file types
- `captain_claw/web/static/computer.css` — Theme-consistent attach/folder buttons across all themes
- `botport/botport/server.py` — Swarm engine and scheduler integration
- `botport/botport/protocol.py` — File transfer message types
