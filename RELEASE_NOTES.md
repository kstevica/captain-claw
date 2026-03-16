# Captain Claw v0.4.2 Release Notes

**Release title:** Public Computer Mode, OpenRouter Provider, Suggested Next Steps

**Release date:** 2026-03-16

## Highlights

This release introduces **Public Computer Mode** — a secure, session-isolated deployment mode that exposes only the Computer research workspace to anonymous visitors while locking down all other pages and APIs. Each public user gets a unique 6-character access code, and all files, uploads, and exploration trees are fully isolated per session. Also new: **OpenRouter** as a first-class LLM provider (access 200+ models through a single API key), and **suggested next-step buttons** that appear after each agent response in Computer for one-click follow-up actions.

## New Features

### Public Computer Mode

Deploy Captain Claw's Computer workspace as a public-facing research tool with built-in session isolation and route lockdown:

- **Session management** — Landing page with "New Session" (generates 6-character access code) and "Resume Session" (enter existing code) tabs
- **Per-session file isolation** — All file operations (list, view, download, upload, media) are scoped to the user's session directory
- **Route lockdown** — Only Computer-related routes are accessible; all other pages and APIs return 403 Forbidden
- **Admin bypass** — Users authenticated via `auth_token` have unrestricted access to the full UI
- **HMAC-signed session cookies** — Secure, tamper-proof session identification for all REST endpoints
- **Landing page** — Branded Captain Claw landing with public mode notice and GitHub link

**Configuration:**
```yaml
web:
  auth_token: "your-admin-password"
  public_run: "computer"
```

New files: `captain_claw/web/public_auth.py`, `captain_claw/web/public_session.py`, `captain_claw/web/static/public_landing.html`

### OpenRouter Provider

OpenRouter is now a first-class LLM provider, enabling access to 200+ models from multiple providers through a single API key:

- Provider name: `openrouter`
- Automatic LiteLLM routing with `openrouter/` prefix handling
- Model IDs with slashes (e.g., `nvidia/llama-3.1-nemotron-ultra-253b-v1`) handled correctly
- Added to onboarding flow alongside OpenAI, Anthropic, Gemini, and Ollama
- Environment variable: `OPENROUTER_API_KEY`

**Configuration:**
```yaml
model:
  provider: openrouter
  model: nvidia/llama-3.1-nemotron-ultra-253b-v1
```

### Suggested Next Steps (Computer)

After each agent response, Computer automatically extracts suggested next steps and presents them as clickable buttons:

- Lightweight follow-up LLM call analyzes the response for explicit suggestions
- Heuristic pre-filter avoids unnecessary LLM calls on responses without bullet/numbered lists
- Buttons appear below the Answer tab content with the suggestion label
- Clicking a button populates the input and sends the action automatically
- Maximum 6 suggestions per response, with 30-character button labels

New file: `captain_claw/next_steps.py`

## Improvements

### Computer UI Enhancements

- **Inline explore buttons** — Explore and Cancel buttons now appear inline in the input actions row (alongside 📎, 📁, Send) instead of a separate full-width bar, improving mobile usability
- **Image rendering fix** — Raw `<img>` HTML tags in LLM responses are pre-processed to markdown before escaping, preventing broken image rendering
- **Classic Mac theme fix** — Active tab labels are now visible (z-index fix for title bar stripe overlay)
- **Mobile layout** — Input panel scrollable on small screens, explore buttons compact on narrow viewports

### Chat Handler Improvements

- Refactored WebSocket chat handler for public session support
- Session-aware message routing for public vs admin users

## REST API Changes

### New Endpoints

- `POST /api/public/session/new` — Create a new public session (returns access code)
- `GET /api/public/session/enter?code=XXXXXX` — Enter a session (sets cookie, redirects to `/computer`)
- `GET /api/public/session/info` — Get current public session info

### Modified Endpoints

- `GET /api/files`, `GET /api/files/{session_id}` — Now enforce session isolation for public users
- `GET /api/files/content`, `GET /api/files/download/{path}`, `GET /api/files/view/{path}` — Session ownership check for public users
- `POST /api/file/upload`, `POST /api/image/upload` — Uploads scoped to public session directory
- `GET /api/media/{path}` — Session ownership check for public users

## Configuration Changes

New `web` config fields:

| Field | Default | Description |
|---|---|---|
| `auth_token` | `""` | Set to enable authentication; empty = auth disabled |
| `auth_cookie_max_age` | `90` | Days until auth cookie expires |
| `public_run` | `""` | Set to `"computer"` to expose only Computer to anonymous visitors |

## Version Changes

- `captain_claw` — 0.4.1 → 0.4.2
- `botport` — 0.3.4 → 0.4.2 (synced with captain_claw)
- `desktop` — 0.4.1 → 0.4.2

## Internal

### New Files
- `captain_claw/web/public_auth.py` — Route lockdown middleware and session identification for public mode
- `captain_claw/web/public_session.py` — Public session management (access codes, HMAC-signed cookies)
- `captain_claw/web/static/public_landing.html` — Public Computer landing page with session management UI
- `captain_claw/next_steps.py` — Extract suggested next steps from LLM responses

### Modified Files
- `pyproject.toml` — Version 0.4.1 → 0.4.2
- `captain_claw/__init__.py` — Version bump
- `botport/pyproject.toml` — Version 0.3.4 → 0.4.2 (synced)
- `botport/botport/__init__.py` — Version bump
- `desktop/package.json` — Version bump
- `captain_claw/llm/__init__.py` — OpenRouter provider support, model ID handling
- `captain_claw/config.py` — `auth_token`, `auth_cookie_max_age`, `public_run` fields
- `captain_claw/main.py` — OpenRouter in CLI onboarding
- `captain_claw/web_server.py` — Public mode middleware, session routes, public landing page serving
- `captain_claw/web/static_pages.py` — Public landing page redirect logic
- `captain_claw/web/chat_handler.py` — Public session-aware message routing
- `captain_claw/web/ws_handler.py` — Public session WebSocket handling
- `captain_claw/web/rest_files.py` — Session isolation for all file endpoints
- `captain_claw/web/rest_file_upload.py` — Public session upload scoping
- `captain_claw/web/rest_image_upload.py` — Public session upload scoping
- `captain_claw/web/rest_computer.py` — Next steps extraction endpoint
- `captain_claw/web/rest_settings.py` — Public mode config exposure
- `captain_claw/web/static/computer.js` — Next steps buttons, inline explore buttons, image rendering fix
- `captain_claw/web/static/computer.css` — Inline explore button styles, Classic Mac z-index fix, mobile layout improvements
- `captain_claw/web/static/computer.html` — Hamburger menu button
- `captain_claw/web/static/onboarding.js` — OpenRouter in onboarding flow
- `captain_claw/web/static/app.js` — Public mode UI adjustments
- `README.md` — OpenRouter in badges, feature table, API keys, Computer feature descriptions
- `USAGE.md` — Public mode docs, next steps docs, web config fields, ToC updates
