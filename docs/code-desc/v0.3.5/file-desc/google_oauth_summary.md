# Summary: google_oauth.py

# google_oauth.py Summary

Implements Google OAuth2 authentication flow handlers for a web server, managing authorization, token exchange, credential storage, and integration with an LLM provider (Gemini/Vertex AI). Handles the complete OAuth lifecycle including PKCE-based security, state validation, token management, and credential injection into downstream services.

## Purpose

Solves the problem of securely authenticating users via Google OAuth2 while maintaining PKCE security standards, managing token lifecycle, and seamlessly integrating Google credentials into an AI provider (Gemini/Vertex AI) for authenticated API calls. Provides endpoints for login initiation, callback handling, status checking, and logout/revocation.

## Most Important Functions/Classes

1. **auth_google_login(server, request)**
   - Initiates OAuth2 authorization flow by generating PKCE pair (verifier/challenge), creating secure state token, and redirecting to Google's authorization endpoint. Implements stale state cleanup (10-minute TTL) and merges default scopes with user-configured ones to prevent scope regression in config updates.

2. **auth_google_callback(server, request)**
   - Handles OAuth2 callback from Google, validates state token against pending OAuth store, exchanges authorization code for tokens using PKCE verifier, fetches user info, stores credentials via oauth_manager, and injects credentials into the provider. Returns HTML redirect response with user email confirmation.

3. **inject_oauth_into_provider(server)**
   - Retrieves stored Vertex AI credentials from oauth_manager and injects them into the active LiteLLMProvider (if Gemini provider), enabling authenticated calls to Google Cloud APIs. Bridges OAuth credential storage with downstream LLM provider configuration.

4. **auth_google_status(server, request)**
   - Returns JSON status of OAuth connection including enabled flag, connection state, and authenticated user info. Checks configuration validity and oauth_manager availability to determine overall connectivity status.

5. **auth_google_logout(server, request)**
   - Revokes OAuth tokens via oauth_manager and clears Vertex credentials from the provider, completely disconnecting the Google OAuth session and removing authentication from downstream services.

## Architecture & Dependencies

**Key Dependencies:**
- `aiohttp.web` - HTTP request/response handling
- `captain_claw.google_oauth` - Core OAuth utilities (PKCE generation, URL building, token exchange, user info fetching)
- `captain_claw.config` - Configuration management for OAuth settings
- `captain_claw.llm.LiteLLMProvider` - LLM provider integration point
- `secrets` - Cryptographic token generation
- `time` - State expiration tracking

**System Role:**
- Acts as HTTP endpoint handler layer for OAuth flows
- Manages in-memory state store (`server._pending_oauth`) for PKCE verification
- Bridges configuration, OAuth manager, and LLM provider components
- Implements security best practices: PKCE flow, state validation, token revocation, credential isolation

**Data Flow:**
1. User initiates login → state + PKCE pair generated → stored in server memory → redirect to Google
2. Google callback → state validated → code exchanged for tokens → user info fetched → credentials stored
3. Credentials injected into Gemini provider for authenticated API access
4. Logout revokes tokens and clears provider credentials