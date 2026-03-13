# Summary: auth.py

# auth.py Summary

Token-based authentication middleware for aiohttp web applications that enforces secure session management through signed cookies and query parameter tokens. Implements HMAC-SHA256 signing for cookie validation, automatic token-to-cookie conversion with redirect, and TLS detection for secure cookie flags.

## Purpose

Solves the problem of protecting web UI endpoints from unauthorized access while providing a seamless authentication flow. Enables both persistent cookie-based sessions and one-time token sharing via URL parameters (useful for sharing authenticated links), with automatic cleanup of sensitive token parameters from URLs.

## Most Important Functions/Classes

1. **`create_auth_middleware(config: WebConfig) -> Callable`**
   - Factory function that returns an aiohttp middleware enforcing token-based authentication. Implements three-stage request processing: validates existing cookies, converts query tokens to cookies with redirect, or returns 401 with appropriate response format (JSON for APIs, HTML for browsers).

2. **`_validate_cookie(value: str, auth_token: str, max_age_days: int) -> bool`**
   - Validates cookie integrity and expiration by verifying HMAC-SHA256 signature and checking age against max_age_days threshold. Returns False for malformed, expired, or tampered cookies using constant-time comparison to prevent timing attacks.

3. **`_make_cookie_value(auth_token: str) -> str`**
   - Creates signed cookie values in format `timestamp:hmac_hex` using current Unix timestamp and HMAC-SHA256 signature. Enables server-side validation without storing session state.

4. **`_strip_token_param(url: str) -> str`**
   - Removes sensitive `token` query parameter from URLs before redirecting, preventing token exposure in browser history, referrer headers, or logs. Uses urllib parsing to safely reconstruct URLs.

5. **`_is_behind_tls(request: web.Request) -> bool`**
   - Detects HTTPS connections via `X-Forwarded-Proto` header for reverse proxy scenarios. Determines whether to set `secure` flag on cookies to prevent transmission over unencrypted connections.

## Architecture & Dependencies

- **Framework**: aiohttp (async web framework)
- **Security**: hashlib, hmac (cryptographic signing), time (expiration tracking)
- **URL handling**: urllib.parse (query parameter manipulation)
- **Configuration**: Expects `WebConfig` with `auth_token` and `auth_cookie_max_age` properties
- **Cookie strategy**: HttpOnly + SameSite=Lax + Secure (when behind TLS) flags for defense-in-depth
- **Response differentiation**: Content negotiation via Accept headers and path patterns to serve JSON for APIs and HTML for browsers