# Summary: browser.py

# browser.py Summary

## Overview
Comprehensive browser automation tool for Captain Claw that provides headless Playwright-based web interaction with persistent sessions, credential management, network traffic capture, API replay, and workflow recording/playback capabilities. Supports multi-app isolated contexts, intelligent page analysis via vision LLM, and automated login flows with cookie persistence.

## Purpose
Solves the problem of automating complex web application interactions including:
- Multi-step browser workflows (navigate → observe → interact → capture → replay)
- Credential-secured login automation with cookie persistence
- Network traffic interception and API pattern discovery
- Direct API execution via captured endpoints (bypassing browser)
- Workflow recording and parameterized replay for repeatable tasks
- Intelligent page understanding via accessibility trees + vision LLM analysis
- Multi-application session isolation within a single browser instance

## Architecture & Key Components

### Core Session Management
- **Module-level stores**: `_BROWSER_SESSIONS` and `_WORKFLOW_RECORDERS` persist across tool re-instantiation (critical for Telegram agents)
- **Session keying**: Supports multi-app contexts via `_session_id` parameter; defaults to "default"
- **Lazy initialization**: Sessions created on first use via `_get_session()`

### Action Dispatch System
- **Single execute() method** routes 34+ actions via `_DISPATCH` dictionary to handler methods
- **Consistent error handling**: Playwright dependency check, timeout detection, helpful error messages
- **Async/await throughout**: All handlers are async for non-blocking I/O

### Key Dependencies
- `BrowserSession`: Low-level Playwright wrapper (page management, navigation, clicks, typing)
- `AccessibilityExtractor`: Semantic page structure extraction + interactive element discovery
- `BrowserVision`: Vision LLM integration for goal-aware page analysis
- `CredentialStore`: Encrypted credential storage (Fernet or Base64 obfuscation)
- `NetworkInterceptor`: XHR/fetch capture and API pattern summarization
- `ApiReplayEngine`: HTTP execution of captured APIs with auth header resolution
- `WorkflowRecorder` / `WorkflowReplayEngine`: Step recording and parameterized replay
- `SessionManager`: Persistent storage for APIs and workflows

---

## Most Important Functions/Classes/Procedures

### 1. **execute(action, **kwargs) → ToolResult**
**Purpose**: Central dispatcher for all 34 browser actions.
- Validates Playwright installation
- Normalizes action name (lowercase, stripped)
- Routes to handler via `_DISPATCH` dictionary
- Wraps exceptions with context-aware error messages
- Logs all calls with action name and result status
- **Critical for**: Ensuring consistent error handling and action routing across the entire tool

### 2. **_observe() → ToolResult**
**Purpose**: Rich, multi-signal page analysis combining screenshot + vision LLM + accessibility tree + interactive elements.
- Takes optional `goal` parameter to focus vision analysis
- Captures viewport screenshot (registers with file registry)
- Runs vision model with goal-aware prompt if configured
- Extracts full accessibility tree (max 6 levels, 150 lines)
- Lists 40 interactive elements with suggested selectors
- Detects login forms and suggests stored credentials if available
- **Critical for**: Understanding page state before acting; primary action for multi-step workflows

### 3. **_act(goal) → ToolResult**
**Purpose**: Goal-directed page analysis with actionable recommendations (lighter than observe).
- **Requires** `goal` parameter (e.g., "find Sprint 23 report")
- Returns: current state + goal-relevant elements + recommended next action + blockers
- Generates action hints based on page state (login detection, error states, element matching)
- Compact accessibility tree (4 levels, 80 lines) vs. full observe
- **Critical for**: Driving multi-step observe-think-act workflows; agent loop calls this to decide next action

### 4. **_login(app_name) → ToolResult**
**Purpose**: Automated login with two-strategy fallback (cookies → form fill).
- **Strategy 1**: Restore saved cookies, navigate, verify URL didn't redirect to login
- **Strategy 2**: Navigate to login URL, find form fields via accessibility tree, fill username/password, submit, wait, save cookies
- Field detection: CSS selectors first (most reliable), then ARIA role-based
- Saves cookies for next time if `cookie_persistence` enabled
- Detects login page via URL heuristics + password field presence
- **Critical for**: Automating authentication; enables multi-app workflows without manual login

### 5. **_network_capture() → ToolResult**
**Purpose**: Analyze captured XHR/fetch traffic and store discovered API patterns in SessionManager.
- Summarizes captured requests into API groups (by base URL)
- Extracts: base_url, endpoints, auth type, sample requests
- Creates API entries in SessionManager for later replay
- Tags as "browser-captured,auto-discovered"
- Increments API usage counter on successful replay
- **Critical for**: Converting browser interactions into reusable APIs; enables api_replay to skip browser

### 6. **_api_replay(api_id, endpoint, method, query_params, body_json) → ToolResult**
**Purpose**: Execute a captured/stored API directly via HTTP, bypassing the browser.
- Looks up API in SessionManager (supports direct ID, #N index, fuzzy name)
- Resolves auth headers: API entry credentials → browser credential store → no auth
- Parses optional JSON query_params and body_json
- Executes via `ApiReplayEngine.replay()`
- Increments API usage counter on success
- Returns: status code + response body (truncated) + helpful error context (401/403 hints)
- **Critical for**: Speed optimization; replays API calls without browser overhead

### 7. **_workflow_record_start/stop/save/run/delete**
**Purpose**: Record, parameterize, and replay browser interaction sequences.
- **record_start**: Attaches WorkflowRecorder to current page, starts capturing clicks/typing/navigation
- **record_stop**: Stops recording, shows summary (step count, actions)
- **save**: Converts recorded steps to workflow entry, applies variable placeholders (e.g., `{{query}}`), stores in SessionManager
- **run**: Loads workflow, substitutes variables, replays steps via WorkflowReplayEngine, increments usage
- **list/show/delete**: CRUD operations on saved workflows
- **Critical for**: Automating repeatable multi-step tasks; enables parameterized workflows (e.g., search with different queries)

### 8. **_click(selector, text, role, nth) → ToolResult**
**Purpose**: Click an element with flexible targeting (CSS selector, visible text, ARIA role).
- **Priority**: role-based (best for React/SPA) → text-based → CSS selector
- **nth parameter**: Zero-based index for disambiguating multiple matches (essential for React apps with duplicate labels)
- Provides helpful error messages if element not found (suggests observe/find_element)
- **Critical for**: Interacting with dynamic pages; nth support is key for React/SPA apps

### 9. **_type(selector, role, text, nth) → ToolResult**
**Purpose**: Type text into form fields with flexible targeting.
- Supports CSS selector or ARIA role targeting
- **nth parameter**: For multiple matching fields
- Validates text parameter is provided
- Helpful timeout errors suggest using observe/find_element
- **Critical for**: Form filling; nth support handles duplicate input fields

### 10. **_credentials_store/list/delete(app_name, url, username, password) → ToolResult**
**Purpose**: Manage encrypted credential storage for web apps.
- **store**: Encrypts password (Fernet or Base64), stores with app_name/url/username
- **list**: Shows all stored credentials with passwords masked
- **delete**: Removes credential entry (including saved cookies)
- Credentials keyed by app_name for easy lookup during login
- **Critical for**: Enabling automated login without hardcoding passwords; supports multi-app workflows

---

## System Integration Points

1. **SessionManager** (`captain_claw.session`): Stores APIs, workflows, credentials persistently
2. **FileRegistry** (`_file_registry` kwarg): Registers screenshots for downstream vision/OCR tools
3. **Config** (`captain_claw.config`): Browser headless mode, login wait times, cookie persistence settings
4. **Logging** (`captain_claw.logging`): Structured logging for all actions (action name, success, error, timing)
5. **Vision LLM**: Optional integration for goal-aware page analysis (observe/act)

## Notable Design Patterns

- **Lazy initialization**: Sessions and recorders created on first use
- **Module-level state**: Survives tool re-instantiation (Telegram agent use case)
- **Flexible targeting**: Multiple ways to select elements (CSS, text, role, nth)
- **Two-strategy fallback**: Login tries cookies first, falls back to form fill
- **Helpful errors**: Timeout/not-found errors suggest next steps (observe, find_element)
- **Workflow parameterization**: Variable placeholders (`{{name}}`) enable reusable workflows
- **Multi-app isolation**: Each app gets isolated browser context with separate cookies/storage