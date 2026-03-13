# Summary: rest_personality.py

# rest_personality.py Summary

Manages REST API endpoints for agent personality profiles, supporting both global default personalities and per-user (Telegram) customizations. Handles CRUD operations on personality attributes (name, description, background, expertise, instructions) with automatic cache invalidation and LLM-powered field rephrasing capabilities.

## Purpose

Solves the problem of dynamically configuring and personalizing AI agent behavior through HTTP endpoints, enabling:
- Global personality management for the default agent
- Per-user personality overrides for Telegram users
- Real-time personality updates with automatic instruction cache clearing
- AI-assisted enrichment of personality fields via LLM rephrasing

## Most Important Functions/Classes/Procedures

1. **`_merge_personality_fields(p, body)`**
   - Applies JSON body fields to a Personality object in-place with type coercion and validation. Handles flexible expertise input (list or comma-separated string). Core utility for both global and per-user personality updates.

2. **`_clear_instruction_caches(server)`**
   - Invalidates cached system prompts across the main agent and all Telegram user agents after personality changes. Ensures prompt regeneration reflects updated personality data without server restart.

3. **`put_personality(server, request)` & `put_user_personality(server, request)`**
   - HTTP PUT handlers for updating global and per-user personalities respectively. Validate input (non-empty name), merge fields, persist changes, and clear caches. Return updated personality as JSON.

4. **`rephrase_personality_field(server, request)`**
   - POST endpoint that uses an LLM provider to intelligently rewrite/enrich personality fields (description, background, expertise, instructions). Generates field-specific system prompts and handles timeouts gracefully, falling back to original text on failure.

5. **`list_user_personalities(server, request)` & `get_user_personality(server, request)` & `delete_user_personality(server, request)`**
   - Complete CRUD operations for per-user personalities. List endpoint enriches results with Telegram user metadata (username, first_name). Get/delete handle 404 cases appropriately.

6. **`list_telegram_users(server, request)`**
   - Exposes approved Telegram users from server state for UI user picker when creating new per-user personalities. Minimal data payload (user_id, username, first_name).

## Architecture & Dependencies

- **Type hints**: Uses `TYPE_CHECKING` guard for circular import avoidance with `WebServer`
- **Async/await pattern**: All handlers are async, compatible with aiohttp framework
- **Lazy imports**: Personality and LLM modules imported within handlers to minimize startup overhead
- **Cache invalidation**: Directly manipulates `_cache` dict on instruction objects (assumes dict-based caching)
- **LLM integration**: Dynamically obtains provider from `server.agent.provider` or falls back to `get_provider()` factory
- **Error handling**: Graceful degradation (returns original text if LLM fails), JSON validation with 400 status codes, 404 for missing resources