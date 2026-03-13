# Summary: static_pages.py

## Summary

This module provides HTTP request handlers for serving static HTML pages in a web UI built with aiohttp. It implements cache-busting functionality for dynamic asset versioning and routes requests to 23 different HTML page templates representing various features of the Captain Claw system (chat, orchestration, workflows, memory management, settings, etc.).

## Purpose

Solves the problem of efficiently serving static HTML pages while preventing browser caching issues when frontend assets are updated. Provides a centralized routing layer for all static page endpoints in the web server, enabling the UI to navigate between different feature modules without requiring manual cache invalidation.

## Most Important Functions/Classes

1. **`_cache_bust(html_path: Path) -> web.Response`**
   - Core utility function that reads HTML files and dynamically injects cache-busting query parameters into static asset references (app.js, style.css). Uses file modification time (mtime) as the version identifier to ensure browsers fetch fresh assets when files change.

2. **`serve_home()` and `serve_chat()`**
   - Primary entry point handlers that use `_cache_bust()` for dynamic cache invalidation. These are the main pages requiring aggressive cache management since they load frequently-updated JavaScript and CSS bundles.

3. **`serve_favicon()`**
   - Specialized handler with fallback logic—returns favicon.svg if present, otherwise returns a 204 No Content response. Demonstrates defensive file handling patterns.

4. **Generic page handlers (`serve_orchestrator()`, `serve_instructions()`, `serve_cron()`, `serve_workflows()`, etc.)**
   - 19 additional handlers serving feature-specific HTML pages (memory, settings, sessions, datastore, playbooks, skills, etc.). Each returns a direct FileResponse without cache-busting, indicating these pages have static content or handle versioning differently.

## Architecture Notes

- **Dependency**: Requires aiohttp web framework and WebServer instance (though WebServer parameter is unused in all handlers, suggesting legacy API design)
- **Static asset location**: All files resolved relative to module location via `STATIC_DIR = Path(__file__).resolve().parent / "static"`
- **Pattern**: Follows async handler pattern required by aiohttp routing system
- **Scalability consideration**: Cache-busting only applies to home/chat pages; other pages may need similar treatment if they load dynamic assets