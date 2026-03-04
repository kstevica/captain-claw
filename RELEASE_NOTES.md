# Captain Claw v0.3.3 Release Notes

**Release title:** Playbooks — Learn from Past Sessions

**Release date:** 2026-03-03

## Highlights

Feature release — the agent can now learn from past sessions via **playbooks**, a persistent cross-session orchestration pattern memory. Rate sessions as good or bad to auto-distill reusable do/don't pseudo-code patterns that are injected into planning context when similar tasks are detected. Includes a web UI editor, chat header override selector, and full REST API.

## New Features

### Playbooks — Cross-Session Orchestration Pattern Memory
- Rate sessions with "rate good" or "rate bad" to auto-distill a reusable playbook from the session's tool trace and message history
- Standalone LLM call extracts a compact session summary (max 2000 chars) and ordered tool trace (max 30 entries), then produces structured do/don't pseudo-code patterns
- 9 task type classifications: batch-processing, web-research, code-generation, document-processing, data-transformation, orchestration, interactive, file-management, other
- Auto-injection: when a new task arrives, the agent classifies it by task type using keyword heuristics and retrieves up to 2 matching playbooks into the planning context
- Usage tracking: each playbook tracks retrieval count and last-used timestamp
- Manual creation via the `playbooks` tool `add` action or REST API
- 7 tool actions: `add`, `list`, `search`, `info`, `update`, `remove`, `rate`

### Playbooks Web UI Editor
- New `/playbooks` page linked from the homepage
- Browse all playbooks with task type badges, usage counts, and ratings
- Create, edit, and delete playbooks from the UI
- View full do/don't patterns, trigger descriptions, and reasoning

### Playbook Override Selector
- New dropdown in the chat header (next to model selector)
- Three modes: Auto (default — system selects based on task type), None (disable injection), or pick a specific playbook
- Override persists for the session and is reflected in session info

### Playbook Rating Hint
- After complex tasks (3+ tool calls with task contracts, scale loop, or pipeline), the agent appends a one-time hint suggesting the user rate the session
- Shown once per session to avoid noise

### Playbooks REST API
- `GET /api/playbooks` — list all playbooks (up to 200, optional `?task_type=` filter)
- `GET /api/playbooks/search` — keyword search (`?q=` required, optional `?task_type=` filter)
- `GET /api/playbooks/{id}` — get one playbook by ID
- `POST /api/playbooks` — create a new playbook
- `PATCH /api/playbooks/{id}` — partial update
- `DELETE /api/playbooks/{id}` — delete a playbook

## Other Changes

- Tool count: 29 built-in tools (was 28)
- Playbooks tool is always enabled (alongside personality and botport)
- Playbook context injected into task contract planner and scale loop planning
- Session metadata tracks playbook ratings

## Internal

- New `agent_playbook_mixin.py` — playbook retrieval, distillation, context injection, and override handling
- New `tools/playbooks.py` — playbooks tool with 7 actions
- New `web/rest_playbooks.py` — playbooks REST API endpoints
- New `web/static/playbooks.html` — playbooks editor page
- New `web/static/playbooks-editor.js` — playbooks editor JavaScript
- New `web/static/playbooks.css` — playbooks editor styles
- New `instructions/playbook_distill_system_prompt.md` — LLM distillation system prompt
- New `instructions/playbook_distill_user_prompt.md` — LLM distillation user prompt template
- Updated `agent.py` — added `AgentPlaybookMixin` to Agent class MRO
- Updated `agent_completion_mixin.py` — playbook rating hint after complex tasks
- Updated `agent_context_mixin.py` — playbook context injection into message assembly, tool registration
- Updated `agent_orchestration_mixin.py` — playbook injection into orchestrator planning
- Updated `agent_reasoning_mixin.py` — playbook injection into scale loop planning
- Updated `config.py` — added `playbooks` to default enabled tools and always-enabled set
- Updated `session/__init__.py` — PlaybookEntry model, CRUD operations, search, usage tracking
- Updated `web/ws_handler.py` — playbook list in welcome message, `set_playbook` message handler
- Updated `web/static/app.js` — playbook override selector UI logic
- Updated `web/static/index.html` — playbook badge and dropdown in chat header
- Updated `web/static/style.css` — playbook selector styles
- Updated `web/static/home.html` — playbooks card on homepage
- Updated `web/static_pages.py` — serve playbooks page
- Updated `web_server.py` — playbook REST routes, session info playbook fields
- Updated `instructions/task_contract_planner_user_prompt.md` — playbook block injection
- Updated `README.md` and `USAGE.md` — documentation for playbooks feature
- 22 files changed, 787 insertions, 10 deletions
