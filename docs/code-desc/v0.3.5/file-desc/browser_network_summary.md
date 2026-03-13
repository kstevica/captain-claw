# Summary: browser_network.py

# browser_network.py Summary

## Summary
A network traffic interceptor for Playwright-based browser automation that captures XHR/fetch requests while intelligently filtering static assets (images, CSS, fonts, JS bundles). Transforms captured network patterns into structured API summaries with endpoint deduplication, authentication detection, and path parameterization for later API replay and integration with the Captain Claw APIs memory system.

## Purpose
Solves the problem of extracting API patterns from browser-based user interactions without manual documentation. Enables automated discovery of backend APIs by observing network traffic during web application usage, with intelligent filtering to focus only on meaningful API calls while ignoring asset loading. Bridges the gap between browser automation and API testing by converting observed traffic into reusable API specifications.

## Most Important Functions/Classes/Procedures

1. **NetworkInterceptor (class)**
   - Core orchestrator that attaches to Playwright pages and manages request/response lifecycle. Maintains recording state, enforces capture limits, and provides query/summary interfaces. Handles the async event loop integration with Playwright's request/response events.

2. **summarize_apis() → list[ApiSummary]**
   - Transforms raw captured requests into deduplicated API summaries grouped by base URL. Performs endpoint deduplication using parameterized paths, detects dominant authentication patterns, and generates human-readable API names and descriptions suitable for storage in Captain Claw's API memory system.

3. **_parameterize_path(path: str) → str**
   - Intelligently replaces dynamic path segments (numeric IDs, UUIDs, hash-like strings) with `{id}` placeholders. Enables endpoint deduplication by normalizing URLs like `/api/users/12345` and `/api/users/67890` into the same `/api/users/{id}` pattern.

4. **_is_static_request(url: str, content_type: str) → bool**
   - Multi-layer static asset filter using file extensions, content-type prefixes, and regex path patterns. Prevents noise from bundled JS, CSS, images, fonts, and service workers while allowing API responses through.

5. **_infer_auth_type(headers: dict[str, str]) → tuple[str, str]**
   - Detects authentication mechanism from request headers (Bearer tokens, Basic auth, API keys, cookies). Returns both auth type classification and the actual credential value for API replay scenarios.

## Architecture & Dependencies

**Key Data Structures:**
- `CapturedRequest`: Dataclass holding complete request/response pair with metadata (method, URL, headers, bodies, status, timing)
- `ApiSummary`: Aggregated API specification with endpoints, auth details, and discovery metadata

**Dependencies:**
- `playwright`: Page event attachment for request/response interception
- `urllib.parse`: URL parsing and component extraction
- `captain_claw.logging`: Structured logging integration
- Standard library: `json`, `re`, `time`, `collections.defaultdict`

**System Role:**
Acts as a middleware layer between Playwright browser automation and the Captain Claw API memory system. Captures raw network traffic, applies intelligent filtering and normalization, then produces structured API specifications that can be persisted and replayed without browser interaction. Designed for integration with SessionManager's `create_api()` method for seamless API discovery workflows.

**Configuration Parameters:**
- `max_captures` (default 500): Memory limit to prevent unbounded growth
- `max_body_bytes` (default 10KB): Truncation threshold for large response bodies
- `filter_static` (default True): Toggle for asset filtering behavior