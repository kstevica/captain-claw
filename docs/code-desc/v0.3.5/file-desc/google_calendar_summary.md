# Summary: google_calendar.py

# Google Calendar Tool Summary

## Summary
A comprehensive Google Calendar integration tool that provides full CRUD operations for calendar events via the Google Calendar REST API v3. Uses OAuth2 Bearer token authentication managed by GoogleOAuthManager and httpx for async HTTP requests, with no additional Google SDK dependencies required.

## Purpose
Solves the problem of programmatically managing Google Calendar events within the Captain Claw system, enabling creation, retrieval, searching, updating, and deletion of calendar events with support for complex features like recurring events, attendees, reminders, timezones, and all-day events.

## Most Important Functions/Classes/Procedures

### 1. **execute(action: str, **kwargs) → ToolResult**
   - Main dispatcher method that routes action requests to appropriate handlers
   - Handles token acquisition, error management, and action validation
   - Supports 7 distinct actions: list_events, search_events, get_event, create_event, update_event, delete_event, list_calendars
   - Implements comprehensive exception handling for HTTP and runtime errors

### 2. **_build_event_body(...) → dict[str, Any]**
   - Constructs properly formatted event JSON payloads for Google Calendar API
   - Intelligently handles all-day vs. timed events with different date/datetime formats
   - Manages timezone handling, attendees, reminders, recurrence rules, and color IDs
   - Provides sensible defaults (1-hour duration for timed events, 1-day for all-day events)

### 3. **_get_access_token() → str**
   - Retrieves valid OAuth2 access tokens from GoogleOAuthManager
   - Validates that Calendar scope is granted in the OAuth token
   - Raises descriptive RuntimeError if Google account not connected or Calendar permissions missing
   - Provides user-friendly error messages directing to reconnection flow

### 4. **_action_create_event(...) → ToolResult**
   - Creates new calendar events with full parameter support (summary, start, end, timezone, description, location, attendees, reminders, recurrence, color)
   - Validates required fields (summary, start)
   - Returns created event details including event ID, summary, timing, and Google Calendar link

### 5. **_format_event(ev: dict) → str** and **_format_event_detail(ev: dict) → str**
   - Compact and detailed event formatting for human-readable output
   - Handles both all-day and timed events with intelligent time display
   - Includes location, status, attendees, recurrence, reminders, creator/organizer info
   - Provides emoji-enhanced formatting for calendar UI presentation

## Architecture & Dependencies

**Key Dependencies:**
- `httpx`: Async HTTP client for API requests
- `captain_claw.google_oauth_manager.GoogleOAuthManager`: OAuth token management
- `captain_claw.session`: Session manager for token persistence
- `captain_claw.tools.registry.Tool`: Base tool class for integration
- Standard library: `datetime`, `timezone`, `json`

**System Role:**
- Implements the Tool interface for integration into Captain Claw's tool registry
- Operates as an async-first service with 120-second timeout
- Manages OAuth2 authentication lifecycle and token validation
- Provides structured error handling with user-friendly messages for common API failures (401 auth, 403 permissions, 404 not found, 429 rate limits)

**Key Design Patterns:**
- Action-based dispatch pattern with handler mapping
- Async/await throughout for non-blocking operations
- RFC 3339 datetime normalization for API compatibility
- Scope validation to ensure Calendar permissions are granted
- Graceful degradation with informative error messages