# Summary: web_search.py

# web_search.py Summary

**Summary:** A web search tool that integrates with the Brave Search API to perform web queries and return ranked results. The module provides a configurable, async-capable search interface with support for pagination, regional filtering, language selection, and content freshness controls.

**Purpose:** Solves the problem of enabling AI agents and applications to search the web in real-time using Brave Search as the backend provider. Handles API authentication, parameter validation, response parsing, and error management while maintaining clean output formatting for downstream consumption.

**Most Important Functions/Classes:**

1. **WebSearchTool (class)** - Main Tool subclass implementing the web search capability. Inherits from the tool registry system and exposes a standardized `execute()` method with JSON schema parameter definitions. Manages async HTTP client lifecycle and API integration.

2. **execute() (async method)** - Core execution method that validates the search query, retrieves configuration and API credentials, constructs Brave API request parameters, handles HTTP communication, parses JSON responses, and formats results into human-readable text output. Implements comprehensive error handling for HTTP failures and malformed responses.

3. **_clean_text() (static method)** - Utility function that normalizes whitespace in text strings and enforces maximum character limits (default 500 chars) with truncation indicators. Prevents output bloat and ensures consistent formatting across titles, descriptions, and error messages.

**Architecture & Dependencies:**

- **External Dependencies:** `httpx` (async HTTP client), `captain_claw.config` (configuration management), `captain_claw.logging` (structured logging), `captain_claw.tools.registry` (Tool base class and ToolResult)
- **Configuration:** Reads from `cfg.tools.web_search` for provider type, API key, base URL, timeout, max results, and default SafeSearch level. Falls back to `BRAVE_API_KEY` environment variable for API authentication.
- **API Integration:** Communicates with Brave Search API v1 (`/res/v1/web/search`) using async HTTP GET requests with subscription token authentication via `X-Subscription-Token` header.
- **Parameter Handling:** Supports optional filtering by country code, search language, content freshness (pd/pw/pm/py), and SafeSearch levels (off/moderate/strict). Implements bounds checking (1-20 results max) and pagination via offset.
- **Response Format:** Returns structured `ToolResult` objects containing formatted text with search metadata (engine, query, result count) followed by numbered result entries with title, URL, and snippet.
- **Error Handling:** Gracefully handles missing API keys, invalid providers, HTTP errors (with status codes and response bodies), malformed JSON, and network timeouts. Logs errors with context for debugging.