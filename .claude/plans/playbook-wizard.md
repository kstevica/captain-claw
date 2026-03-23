# Playbook Wizard — LLM-Driven Conversational UI

## Goal
Create a new `/playbook-wizard` page that guides users through creating a playbook via a multi-step LLM-driven conversation. The wizard asks for the desired outcome first, then decomposes it into follow-up questions, and finally produces a complete playbook ready to save.

## Architecture

### New Files
1. **`captain_claw/web/static/playbook-wizard.html`** — Wizard page HTML
2. **`captain_claw/web/static/playbook-wizard.css`** — Wizard styling
3. **`captain_claw/web/static/playbook-wizard.js`** — Client-side wizard logic
4. **`captain_claw/web/rest_playbook_wizard.py`** — Backend endpoint for LLM wizard steps

### Modified Files
5. **`captain_claw/web/static_pages.py`** — Add `serve_playbook_wizard` handler
6. **`captain_claw/web_server.py`** — Register routes for wizard page + API endpoint

## How It Works

### Frontend: Chat-style Wizard UI
- Clean single-column conversational layout (not split-panel like the editor)
- Messages alternate between assistant (LLM) and user
- Each wizard step shows a question/prompt from the LLM and an input area for the user
- At the end, a playbook preview card is shown with a "Save Playbook" button
- A progress indicator shows the current phase (Outcome → Details → Review → Save)

### Backend: `POST /api/playbook-wizard/step`

Single stateless endpoint. The frontend sends the full conversation history each time:

```json
{
  "messages": [
    {"role": "assistant", "content": "What outcome do you want this playbook to achieve?"},
    {"role": "user", "content": "I want to batch-process CSV files..."},
    {"role": "assistant", "content": "...follow-up question..."},
    {"role": "user", "content": "...answer..."}
  ]
}
```

The backend wraps these in a system prompt that instructs the LLM to either:
- Ask a follow-up question (if more info needed) — returns `{"type": "question", "content": "..."}`
- Produce the final playbook JSON (if enough info gathered) — returns `{"type": "playbook", "playbook": {...}}`

The LLM decides when it has enough information to produce the playbook (typically 2-4 exchanges). The system prompt guides it through phases:
1. **Outcome** — understand what the user wants to accomplish
2. **Decomposition** — ask about tools, patterns, edge cases, what to avoid
3. **Generation** — produce the complete playbook fields

### LLM Integration Pattern
Follow the same pattern as `rest_visualization_style.py:_analyze_with_provider`:
- Use `server.agent.provider` if available, fall back to `get_provider()`
- Use `provider.complete()` with `Message` objects
- Parse JSON from response, strip markdown fences

### System Prompt for the Wizard LLM
A dedicated instruction file: **`captain_claw/instructions/playbook_wizard_system_prompt.md`**

Tells the LLM it's a playbook creation wizard. It should:
- Start by understanding the desired outcome
- Ask 2-4 targeted follow-up questions (one at a time) about: tools involved, recommended approach, common mistakes, trigger conditions, examples
- When ready, output a JSON block with all playbook fields
- Response format: always JSON with `{"type": "question"|"playbook", ...}`

## UI Design

### Layout
- Full-width centered column (max ~720px), dark theme matching existing pages
- Top: header with logo, title "Playbook Wizard", nav links to /playbooks and /chat
- Middle: scrollable conversation area with message bubbles
- Bottom: sticky input area with textarea + send button
- Right side of header: phase indicator pills (Outcome → Details → Review)

### Message Bubbles
- Assistant messages: left-aligned, subtle border, `--bg-secondary` background
- User messages: right-aligned, `--accent` tinted background
- Playbook preview: special card with all fields displayed, matching the detail view style from the existing playbooks page
- Each field in the preview is editable inline before saving

### Flow
1. Page loads → shows welcome message from assistant asking for the outcome
2. User types outcome → sent to backend → LLM asks follow-up
3. 2-4 rounds of Q&A
4. LLM produces playbook → shown as a preview card
5. User can edit fields inline, then click "Save Playbook"
6. Save calls `POST /api/playbooks` (existing endpoint)
7. Success → toast + option to create another or go to /playbooks

## Implementation Steps

1. Create the system prompt markdown file
2. Create the backend REST handler (`rest_playbook_wizard.py`)
3. Wire routes in `web_server.py` and `static_pages.py`
4. Create the HTML page
5. Create the CSS
6. Create the JS (conversation state, API calls, rendering, save flow)
