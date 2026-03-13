# Summary: web_fetch.py

# web_fetch.py Summary

This module provides HTTP-based web content retrieval tools with dual modes: clean text extraction and raw HTML fetching. It integrates with the Captain Claw tool registry and includes intelligent blocking of Google Drive URLs that require authentication, redirecting users to the `gws` tool instead.

## Purpose

Solves the problem of retrieving and processing web page content in a format suitable for AI agents—either as human-readable text (default) or raw HTML for DOM analysis and scraping. Prevents authentication failures by detecting and blocking Google Drive/Docs URLs when the `gws` CLI tool is available.

## Most Important Functions/Classes

1. **`_extract_readable_text(html: str, base_url: str | None = None) -> str`**
   - Converts raw HTML to clean, readable text using BeautifulSoup. Removes script/style/canvas tags, preserves links with absolute URLs, normalizes whitespace, and prepends page title. Core text extraction engine.

2. **`WebFetchTool` (class)**
   - Primary tool for fetching URLs and returning clean readable text. Implements async `execute()` method that handles HTTP requests, text extraction, character truncation, and error handling. Always returns text mode regardless of parameters. Includes Google Drive URL blocking logic.

3. **`WebGetTool` (class)**
   - Secondary tool for fetching raw HTML markup when DOM inspection or CSS selector-based scraping is needed. Mirrors WebFetchTool architecture but returns unprocessed HTML. Also blocks Google Drive URLs.

4. **`_is_google_drive_url(url: str) -> bool`**
   - Utility function that detects Google Drive/Docs/Sheets/Slides URLs by parsing hostname against a whitelist of Google domains. Returns boolean for conditional blocking.

5. **`_make_http_client() -> httpx.AsyncClient`**
   - Factory function creating a configured async HTTP client with 30-second timeout, redirect following, and Captain Claw user agent. Used by both tool classes.

## Architecture & Dependencies

- **HTTP Client**: Uses `httpx.AsyncClient` for async HTTP operations with automatic redirect handling
- **HTML Parsing**: BeautifulSoup for DOM manipulation and text extraction
- **Integration**: Extends `Tool` base class from `captain_claw.tools.registry`; uses `ToolResult` for standardized response format
- **Configuration**: Reads `max_chars` limits from `captain_claw.config`
- **Logging**: Integrates with `captain_claw.logging` for request/error tracking
- **System Integration**: Uses `shutil.which()` to detect `gws` CLI availability for conditional Google Drive blocking

## Key Design Patterns

- **Dual-mode architecture**: Separate tools for text vs. HTML modes to prevent accidental HTML returns
- **Defensive parameter handling**: `WebFetchTool` explicitly removes `extract_mode` kwargs to enforce text-only behavior
- **Graceful degradation**: Character truncation with `[truncated]` marker rather than failure
- **Metadata wrapping**: Both tools prepend URL, HTTP status, mode, and size information to output
- **Async-first**: All I/O operations are async-compatible for integration with async agent frameworks