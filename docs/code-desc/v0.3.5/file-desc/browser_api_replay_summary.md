# Summary: browser_api_replay.py

# Summary: browser_api_replay.py

## Overview
A direct API replay engine that executes previously captured HTTP API calls via httpx, bypassing the browser entirely. This module is the final stage of a browser automation pipeline that captures API patterns during web app navigation and replays them at millisecond speeds instead of browser-based seconds. Implements intelligent auth resolution (explicit headers → stored credentials → credential store → anonymous) and handles token refresh scenarios via browser re-login.

## Purpose
Solves the performance bottleneck in browser automation by decoupling API execution from browser navigation. Instead of repeatedly driving a browser through UI interactions to trigger the same API calls, this engine replays captured requests directly via HTTP, achieving 100-1000x speed improvements while maintaining full request/response fidelity including headers, status codes, and body content.

## Most Important Functions/Classes

### 1. **ApiReplayResult** (class)
Data container for API execution outcomes. Encapsulates success status, HTTP metadata (status code, URL, method), response content (body, headers), timing (elapsed_ms), and error messages. Provides `to_summary()` method for human-readable formatting with JSON pretty-printing support and response truncation for large payloads.

### 2. **ApiReplayEngine.replay()** (async classmethod)
Core execution function that constructs and sends HTTP requests via httpx.AsyncClient. Accepts base_url + endpoint path, HTTP method, headers (including auth), query parameters, request body (raw or JSON), timeout, and response size limits. Returns ApiReplayResult with comprehensive metadata. Handles three exception types (TimeoutException, ConnectError, generic Exception) with graceful error reporting and elapsed time tracking via perf_counter.

### 3. **ApiReplayEngine.resolve_auth_headers()** (staticmethod)
Auth header builder supporting multiple authentication schemes: bearer tokens, API keys (X-API-Key header), basic auth, cookies, and custom header formats. Normalizes input (strips "Bearer " prefix if present) and returns a dict of headers to merge into requests. Fallback behavior treats unknown auth types as Authorization header values.

### 4. **ApiReplayEngine.find_endpoint_in_api()** (staticmethod)
Endpoint discovery utility that searches a JSON-encoded endpoints list for matching definitions. Supports regex-based path matching with parameterized routes (e.g., `/users/{id}` matches `/users/123`). Returns the first matching endpoint dict containing method, path, params, request_body, and response_sample metadata.

### 5. **ApiReplayEngine.format_endpoints_list()** (staticmethod)
Display formatter for endpoint documentation. Parses endpoints JSON and renders a human-readable list with HTTP method, path, and optional description. Truncates output to max_items (default 20) with "... and N more" indicator for large endpoint sets.

## Architecture & Dependencies

**Key Dependencies:**
- `httpx` — async HTTP client for direct API calls
- `json` — endpoint definition parsing and response formatting
- `re` — regex-based parameterized path matching
- `time.perf_counter()` — high-resolution timing for elapsed_ms tracking
- `urllib.parse.urljoin` — URL construction from base_url + endpoint

**Design Patterns:**
- **Stateless execution** — fresh httpx.AsyncClient per request (no connection pooling across calls)
- **Auth resolution hierarchy** — explicit > stored > credential store > anonymous
- **Graceful degradation** — JSON pretty-printing falls back to raw text on parse failure
- **Size-bounded responses** — max_response_bytes prevents memory exhaustion on large payloads

**Role in System:**
Acts as the performance optimization layer in a three-phase browser automation pipeline: Phase 1 (browser navigation) → Phase 2 (API capture via SessionManager) → Phase 3 (direct replay via this module). Bridges captured API metadata (stored in ApiEntry objects) with execution, enabling rapid iteration and testing without browser overhead.