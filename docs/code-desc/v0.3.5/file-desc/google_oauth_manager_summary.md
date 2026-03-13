# Summary: google_oauth_manager.py

# Summary: google_oauth_manager.py

## Overview
This module manages the complete lifecycle of Google OAuth tokens for a system integrating with Google's Vertex AI via LiteLLM. It handles token persistence in SQLite, automatic refresh of expired tokens, credential generation for downstream services, and graceful disconnection with token revocation.

## Purpose
Solves the problem of maintaining valid Google OAuth credentials across application sessions while abstracting token management complexity from the rest of the codebase. Provides a single source of truth for OAuth state, handles token expiration transparently, and ensures proper cleanup on disconnect.

## Architecture & Dependencies
- **Storage Layer**: Persists tokens via `SessionManager` to SQLite's `app_state` table as JSON
- **Token Model**: Uses `GoogleOAuthTokens` dataclass with expiration tracking and format conversion
- **External Integration**: Wraps low-level OAuth operations (`refresh_access_token`, `revoke_token`, `fetch_user_info`) from `captain_claw.google_oauth`
- **Caching Strategy**: In-memory cache (`_cached_tokens`) reduces database hits for non-expired tokens
- **Configuration**: Reads OAuth credentials from centralized config via `get_config()`

## Most Important Functions/Classes

1. **`GoogleOAuthManager.__init__(session_manager: SessionManager)`**
   - Initializes manager with session backend and empty token cache
   - Single dependency injection point for storage layer

2. **`async get_tokens() → GoogleOAuthTokens | None`**
   - Core retrieval method: checks cache, loads from DB, auto-refreshes if expired
   - Returns None on missing tokens or failed refresh; handles deserialization errors gracefully
   - Implements lazy refresh pattern to minimize token API calls

3. **`async get_vertex_credentials() → dict[str, Any] | None`**
   - Transforms stored tokens into LiteLLM-compatible `authorized_user` JSON format
   - Validates OAuth client credentials before credential generation
   - Primary interface for downstream Vertex AI integration

4. **`async disconnect() → None`**
   - Implements graceful OAuth revocation with fallback strategy (refresh token → access token)
   - Clears all persisted state (tokens and user info) and resets cache
   - Best-effort approach tolerates revocation failures to ensure local cleanup

5. **`async _try_refresh(tokens: GoogleOAuthTokens) → GoogleOAuthTokens | None`**
   - Internal refresh orchestration: validates preconditions, calls refresh API, persists fresh tokens
   - Opportunistically updates cached user info during refresh
   - Preserves tokens on transient failures (network errors) but logs warnings

## Key Design Patterns
- **Defensive Null Handling**: All public methods return `None` on failure rather than raising exceptions
- **Layered Validation**: Checks token existence, expiration, and config completeness at appropriate levels
- **Non-Critical Fallbacks**: User info refresh failures don't block token refresh; cached data is acceptable
- **Audit Logging**: Info/warning logs track token lifecycle events for troubleshooting