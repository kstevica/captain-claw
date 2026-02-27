# Onboarding Redesign Plan

## Current State

**TUI onboarding** (`captain_claw/onboarding.py`): Works but has issues:
- Outdated default models (gpt-5-mini, claude-3-5-sonnet-latest)
- No API key validation (user can finish with a bad key)
- No multi-model setup (the main selling point of Captain Claw)
- Only covers Telegram, not Slack/Discord
- No Google OAuth setup
- No email setup
- Linear flow with no way to skip sections

**Web onboarding**: Does not exist. First-time web users land on a blank dashboard with no model configured. They have to find `/settings` on their own.

**Web settings page** (`settings.html/js`): Already very comprehensive with 12 config groups, secret masking, wizards for email and messaging, allowed-model card editor, and hot-reload on save. This is the "power user" page — onboarding should be the "quick start" funnel that leads here.

## Design Principles

1. **Two surfaces, one flow** — Web gets a step-by-step wizard page (`/onboarding`); TUI gets the same steps via rich prompts. Both call the same backend logic.
2. **Get to a working agent fast** — The only hard requirement is one working model. Everything else is "nice to have" and skippable.
3. **Validate eagerly** — Test the API key before moving on. A user who finishes onboarding should have a confirmed-working setup.
4. **Don't repeat settings** — Onboarding covers first-time essentials. Everything else lives in `/settings`. The final step should point users there.

## Proposed Flow

### Step 1 — Welcome

What: Splash with logo and one-liner ("Let's get Captain Claw ready").
Action: "Get Started" button / Enter.

### Step 2 — Config Location (TUI only)

What: Global (`~/.captain-claw/config.yaml`) vs local (`./config.yaml`).
Why TUI-only: Web always writes to global (the server needs a fixed config path).
Default: Global.

### Step 3 — Model Provider

What: Pick primary provider — OpenAI, Anthropic, Gemini, Ollama.
UI: Card grid (web) / numbered table (TUI).
Each card shows: provider name, example models, what env var to set.
Default: OpenAI.

### Step 4 — Model Name

What: Text input with provider-specific default pre-filled.
Defaults:
- OpenAI → `gpt-4.1-mini`
- Anthropic → `claude-sonnet-4-20250514`
- Gemini → `gemini-2.5-flash-preview-05-20`
- Ollama → `llama3.2`

### Step 5 — API Key

What: Masked input for the API key. Ollama: base URL instead.
Option: "I'll use an environment variable instead" checkbox that skips input.
Env var hint shown next to input (e.g. `OPENAI_API_KEY`).

### Step 6 — Validate Connection

What: Automatic — fire a lightweight completion request (`"Say OK"`, max_tokens=5).
Success: Green checkmark, proceed.
Failure: Show error, let user re-enter key or skip validation.
Ollama: Test `GET /api/tags` endpoint instead.

### Step 7 — Additional Models (optional, collapsed by default)

What: "Want to add more models for multi-session use?"
If yes: Repeat provider/model/key for up to 3 additional models.
These populate `model.allowed[]` in config.
If no: Skip — user can add later in `/settings`.

### Step 8 — Web Search (optional)

What: "Enable web search? Requires a Brave Search API key."
If yes: Masked key input.
If no: Skip.
Note: Show link to get a free key.

### Step 9 — Safety Guards (optional)

What: "Enable safety guards? They check input, output, and tool calls before execution."
Toggle: On/Off. Default: Off (matches current default).
If on: All three guards enabled with `ask_for_approval` level.

### Step 10 — Summary & Save

What: Review table showing all selected values.
Actions: "Save & Launch" / "Back" to edit.
After save: Mark onboarding completed, redirect to chat (web) or start agent (TUI).
Footer: "You can change any of these settings later in /settings (web) or config.yaml."

## Web UI Implementation

### New Files
- `captain_claw/web/static/onboarding.html` — Wizard page
- `captain_claw/web/static/onboarding.js` — Step logic, validation, API calls
- `captain_claw/web/static/onboarding.css` — Wizard styling (or inline in the HTML)
- `captain_claw/web/rest_onboarding.py` — REST endpoints for onboarding

### REST Endpoints
- `GET /api/onboarding/status` — Returns `{ "needed": bool, "completed": bool }`
- `POST /api/onboarding/validate` — Test provider connection `{ provider, model, api_key, base_url }` → `{ ok, error }`
- `POST /api/onboarding/save` — Save config and mark completed `{ ...all_values }` → `{ ok, config_path }`

### Auto-Redirect
In the web server startup or the home/chat page:
- If onboarding not completed and no config exists → redirect to `/onboarding`
- `captain-claw --onboarding` → open browser to `/onboarding` (web mode) or run TUI wizard

### Wizard UI
- Progress bar at top (Step 3 of 10)
- One step visible at a time
- Back/Next buttons
- Skip links for optional steps
- Inline validation (green check / red error)
- Mobile-friendly (single column)

## TUI Implementation

### Changes to `onboarding.py`
- Update `_PROVIDER_DEFAULT_MODELS` with current model names
- Add Step 6 (validation): Use `captain_claw/llm/` provider abstraction to fire a test completion
- Add Step 7 (additional models): Optional loop to add `model.allowed[]` entries
- Remove Telegram-specific setup (move to "configure later in settings" note)
- Add post-save hint about `/settings` and `config.yaml`

### Shared Logic
- Extract validation logic into a shared module (`captain_claw/onboarding_utils.py` or keep in `onboarding.py`)
- Both web and TUI call the same `validate_provider_connection()` and `save_onboarding_config()` functions

## What Gets Removed from Onboarding

These are better handled in `/settings` after first launch:
- Telegram/Slack/Discord setup (complex, requires external bot creation first)
- Google OAuth setup (requires GCP console setup first)
- Email provider setup (already has a settings wizard)
- Memory/RAG configuration (advanced)
- Deep memory / Typesense (advanced)
- Scale loop / orchestrator settings (advanced)

## Migration

- Existing `onboarding_state.json` format stays the same (backward compatible)
- Users who already completed onboarding are unaffected
- `--onboarding` flag works as before (forces wizard re-run)
