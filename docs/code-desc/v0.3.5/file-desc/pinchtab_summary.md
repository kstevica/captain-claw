# Summary: pinchtab.py

# PinchTab Browser Automation Tool

## Summary

PinchTab is a token-efficient browser automation tool for Captain Claw that controls a headless Chrome browser via HTTP API instead of expensive screenshots. It uses accessibility tree snapshots (~800 tokens per page) rather than visual screenshots (~2K+ tokens), making it significantly more economical for AI agents. The tool manages persistent browser profiles, supports stealth mode, handles multi-instance orchestration, and requires no Python runtime dependency beyond the standalone Go binary.

## Purpose

Solves the token efficiency problem in browser automation for AI agents. Traditional screenshot-based tools consume 2K+ tokens per page view, while PinchTab's accessibility tree snapshots consume ~800 tokens. Enables persistent login state across restarts, provides anti-bot evasion through stealth modes, and allows agents to interact with pages through element references (e0, e5, e12) rather than coordinates. Coexists with the Playwright-based `browser` tool—agents choose based on task requirements (PinchTab for most browsing, browser for vision analysis or workflow recording).

## Most Important Functions/Classes/Procedures

### 1. **`_PinchTabSession` class**
Manages connection state to a PinchTab server instance. Handles HTTP client lifecycle, server auto-start, instance provisioning, and tab tracking. Key methods: `request()` (HTTP calls to PinchTab API), `ensure_server()` (auto-start if configured), `ensure_instance()` (provision/reuse Chrome instance), `_probe_instance()` (verify CDP connection readiness). Maintains module-level state dictionary keyed by session ID for multi-session support.

### 2. **`_click()` action handler**
Sophisticated click implementation with multi-level fallback logic. Attempts direct CDP click, detects navigation by comparing URLs before/after, handles new tab opens (target="_blank"), and implements alternative activation methods: hover+Enter (bypasses cookie overlays), and href extraction via JavaScript. Includes `_find_nearby_link()` helper to resolve non-focusable elements (e.g., headings inside links) to parent link refs, and `_activate_link()` for keyboard-based activation.

### 3. **`_snapshot()` action handler**
Retrieves page accessibility tree with configurable filtering. Supports `filter=interactive` (buttons/links/inputs only—most token-efficient) or `filter=all` (complete tree). Returns compact format with element references (e0, e5, etc.) that agents use for subsequent actions. Includes optional CSS selector scoping and returns page title/URL context.

### 4. **`_find()` action handler**
Natural language element discovery with fallback chain. Tries dedicated `/find` endpoint first (returns confidence scores and ranked matches), then falls back to interactive snapshot text search with role-based filtering. Only matches clickable roles (link, button, textbox, etc.) to avoid "Element is not focusable" errors. Implements word-matching scoring with role priority (links > buttons > others).

### 5. **`_links()` action handler**
Extracts all navigable links from the page. Attempts JavaScript evaluate to get href attributes and text content, filtering out javascript: URLs. Falls back to interactive snapshot listing link elements with refs when evaluate is unavailable (404). Supports optional query filtering and returns up to 40 links with markdown-style formatting.

### 6. **`execute()` dispatcher**
Central entry point routing actions to handlers via dispatch dictionary. Validates tool is enabled, catches PinchTabError and generic exceptions, logs action completion with content length. Supports 24 actions: navigate, snapshot, click, wait, type, fill, press, scroll, hover, select, text, find, links, screenshot, pdf, eval, tabs, tab_open, tab_close, cookies, profiles, profile_create, health, status, close.

## Architecture & Dependencies

**HTTP Client**: Uses `aiohttp.ClientSession` with configurable timeout, automatic header injection (Content-Type, Bearer token auth), and connection pooling via base_url.

**Process Management**: Spawns PinchTab binary as background subprocess via `asyncio.create_subprocess_exec`, with environment variable configuration (PINCHTAB_PORT, PINCHTAB_BIND, PINCHTAB_TOKEN). Implements graceful shutdown with terminate→wait→kill sequence.

**Session State**: Module-level `_PINCHTAB_SESSIONS` dictionary (keyed by session_id) maintains persistent connections across tool invocations. Each session tracks: HTTP client, server process, instance ID, active tab ID, and startup timestamp.

**Configuration**: Integrates with Captain Claw's config system (`PinchTabConfig`, `get_config()`). Reads from config.yaml: enabled flag, host/port, binary path, timeout, auto_start, headless mode, default profile, stealth mode, allow_evaluate.

**Error Handling**: Custom `PinchTabError` exception for API/connection failures. Implements retry logic for transient Chrome errors ("context canceled", "target closed") by restarting instance and retrying navigate.

**Response Parsing**: Helper methods `_as_dict()` and `_as_list()` normalize PinchTab API responses (which return dict, list, or scalar) into consistent shapes for safe `.get()` access.

**Tab Management**: Tracks active tab ID per session. Supports multi-tab workflows: detects new tabs opened by clicks, switches active tab, maintains separate tab paths for server-mode API calls (`/tabs/{id}/action`).

**Fallback Chains**: Multi-level fallbacks for robustness—e.g., click tries direct CDP, then hover+Enter, then href extraction; find tries dedicated endpoint, then snapshot search; links tries evaluate, then snapshot refs.

## Role in System

Provides primary browser automation interface for AI agents in Captain Claw. Replaces expensive screenshot-based tools for token-constrained scenarios. Enables persistent login workflows (profiles survive restarts), supports stealth browsing (anti-bot evasion), and orchestrates multi-instance Chrome management. Works alongside Playwright-based `browser` tool—agents select based on task (PinchTab for navigation/form-filling, browser for vision analysis). Integrates with Captain Claw's tool registry, config system, and logging infrastructure.