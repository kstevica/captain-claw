# Summary: browser_session.py

# browser_session.py Summary

**Summary:**
Manages a single Playwright browser instance with full async lifecycle support, persisting cookies and state between tool calls. Supports multi-app sessions where each named application (e.g., Jira, Confluence) gets isolated BrowserContext + Page with independent cookies. Integrates network traffic interception and provides comprehensive page interaction methods (click, type, scroll, screenshot).

**Purpose:**
Solves the problem of maintaining a persistent, stateful browser automation session across multiple tool invocations within a Captain Claw AI agent session. Enables seamless switching between isolated application contexts while preserving authentication state, cookies, and navigation history for each app independently.

**Most Important Functions/Classes/Procedures:**

1. **`BrowserSession.__init__()` & lifecycle methods (`start()`, `close()`)**
   - Initializes session with config, creates NetworkInterceptor, manages Playwright browser launch/shutdown. `start()` launches headless Chromium with viewport/user-agent config and optionally auto-records network traffic. `close()` gracefully tears down all resources with error aggregation.

2. **`ensure_page()` & `navigate(url, wait_until)`**
   - `ensure_page()` returns active page, auto-starting browser if dead. `navigate()` performs goto with configurable wait state (domcontentloaded/networkidle), returns dict with status/title/url. Includes post-navigation delay for dynamic content.

3. **Page interaction methods (`click()`, `click_by_text()`, `click_by_role()`, `type_text()`, `type_by_role()`, `press_key()`, `scroll()`, `wait_for_load()`)**
   - Playwright locator-based interactions supporting CSS selectors, text matching (exact/substring), ARIA roles, and nth-element selection. All include logging and brief post-action delays. Support timeout configuration and handle React/dynamic app patterns.

4. **Multi-app session management (`create_app_context()`, `switch_app()`, `switch_to_default()`, `list_app_sessions()`)**
   - Creates isolated BrowserContext per app with independent cookies/storage. `switch_app()` swaps active context, detaches/reattaches network interceptor, preserves URL state. `list_app_sessions()` returns active session inventory with active status indicator.

5. **Cookie & state persistence (`save_cookies()`, `load_cookies()`) + Network interception (`network` property)**
   - Exposes context-level cookie management for session persistence across restarts. NetworkInterceptor attachment/detachment on context switches enables request/response capture with configurable filtering and body size limits.

**Architecture & Dependencies:**
- **Core dependency:** Playwright async API (optional, guarded by `_HAS_PLAYWRIGHT` flag)
- **Internal dependencies:** `BrowserToolConfig` (configuration), `NetworkInterceptor` (traffic capture), `get_logger()` (structured logging)
- **State model:** Single active context/page pair (`_context`, `_page`) plus dict of named app contexts (`_app_contexts`). Default context keyed as `"_default"`.
- **Async-first design:** All lifecycle and interaction methods are async; integrates with asyncio event loop
- **Error resilience:** Graceful degradation in `close()` with error aggregation; exception handling in content retrieval methods

**Role in System:**
Central stateful component for browser automation within Captain Claw agent. Bridges AI tool calls to Playwright, maintaining session continuity across multiple interactions. Enables multi-tenant app automation (e.g., agent switching between Jira and Confluence without re-authentication). Network interception integration supports observability of HTTP traffic during automation.