"""Direct API replay engine for the browser tool.

Replays previously captured API calls directly via httpx — bypassing the
browser entirely.  This is the endgame of the browser automation pipeline:

1. Browser navigates web apps → captures API calls (Phase 2)
2. API patterns stored in APIs memory (``SessionManager.create_api()``)
3. This module replays those APIs directly — milliseconds instead of seconds

Auth resolution order:
1. Explicit ``auth_header`` passed to ``replay()``
2. ``credentials`` stored on the ApiEntry itself (captured during browsing)
3. Browser credential store (``CredentialStore``) for the matching domain
4. No auth (anonymous request)

If a token is expired (401/403), ``refresh_auth()`` can re-login via
the browser to capture a fresh token.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# ApiReplayResult
# ---------------------------------------------------------------------------

class ApiReplayResult:
    """Result of an API replay execution."""

    __slots__ = (
        "success", "status_code", "url", "method",
        "response_body", "response_headers", "elapsed_ms", "error",
    )

    def __init__(
        self,
        *,
        success: bool,
        status_code: int = 0,
        url: str = "",
        method: str = "",
        response_body: str = "",
        response_headers: dict[str, str] | None = None,
        elapsed_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        self.success = success
        self.status_code = status_code
        self.url = url
        self.method = method
        self.response_body = response_body
        self.response_headers = response_headers or {}
        self.elapsed_ms = elapsed_ms
        self.error = error

    def to_summary(self, max_body: int = 2000) -> str:
        """Format a human-readable summary of the result."""
        if not self.success:
            return f"API replay FAILED: {self.error}"

        body_preview = self.response_body[:max_body]
        if len(self.response_body) > max_body:
            body_preview += f"\n... ({len(self.response_body):,} chars total, showing first {max_body})"

        content_type = self.response_headers.get("content-type", "unknown")

        lines = [
            f"API replay: {self.method} {self.url}",
            f"Status: {self.status_code}",
            f"Content-Type: {content_type}",
            f"Elapsed: {self.elapsed_ms:.0f}ms",
            f"Response size: {len(self.response_body):,} chars",
            "",
        ]

        # Try to pretty-print JSON
        if "json" in content_type:
            try:
                parsed = json.loads(self.response_body)
                body_preview = json.dumps(parsed, indent=2)[:max_body]
                if len(json.dumps(parsed, indent=2)) > max_body:
                    body_preview += "\n... (truncated)"
            except (json.JSONDecodeError, TypeError):
                pass

        lines.append("--- Response Body ---")
        lines.append(body_preview)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ApiReplayEngine
# ---------------------------------------------------------------------------

class ApiReplayEngine:
    """Execute captured APIs directly via httpx.

    Stateless — creates a fresh httpx client per request.
    Auth resolution is handled by the caller (BrowserTool) which
    passes the appropriate headers.
    """

    _USER_AGENT = "Captain Claw/0.1.0 (API Replay)"

    @classmethod
    async def replay(
        cls,
        *,
        base_url: str,
        endpoint: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        body: str | None = None,
        body_json: Any | None = None,
        timeout: float = 30.0,
        max_response_bytes: int = 500_000,
    ) -> ApiReplayResult:
        """Execute a single API call.

        Args:
            base_url: The API base URL (e.g. ``https://jira.company.com``).
            endpoint: The endpoint path (e.g. ``/rest/api/2/search``).
            method: HTTP method (GET, POST, PUT, DELETE, PATCH).
            headers: Request headers (including auth headers).
            query_params: URL query parameters.
            body: Raw request body (string).
            body_json: JSON request body (will be serialized).
            timeout: Request timeout in seconds.
            max_response_bytes: Truncate response body at this size.

        Returns:
            ApiReplayResult with status, body, headers, timing.
        """
        method = method.upper()
        full_url = urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))

        # Build headers
        effective_headers: dict[str, str] = {
            "User-Agent": cls._USER_AGENT,
        }
        if headers:
            effective_headers.update(headers)

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "method": method,
            "url": full_url,
            "headers": effective_headers,
            "timeout": timeout,
            "follow_redirects": True,
        }

        if query_params:
            request_kwargs["params"] = query_params
        if body_json is not None:
            request_kwargs["json"] = body_json
        elif body is not None:
            request_kwargs["content"] = body.encode("utf-8")

        log.info(
            "API replay",
            method=method,
            url=full_url,
            has_auth="authorization" in {k.lower() for k in effective_headers},
        )

        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(**request_kwargs)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Read response body (with size limit)
            response_text = response.text
            if len(response_text) > max_response_bytes:
                response_text = response_text[:max_response_bytes]

            response_headers = dict(response.headers)

            log.info(
                "API replay complete",
                method=method,
                url=full_url,
                status=response.status_code,
                elapsed_ms=round(elapsed_ms, 1),
                body_size=len(response_text),
            )

            return ApiReplayResult(
                success=response.is_success,
                status_code=response.status_code,
                url=full_url,
                method=method,
                response_body=response_text,
                response_headers=response_headers,
                elapsed_ms=elapsed_ms,
            )

        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ApiReplayResult(
                success=False,
                url=full_url,
                method=method,
                elapsed_ms=elapsed_ms,
                error=f"Request timed out after {timeout}s",
            )
        except httpx.ConnectError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ApiReplayResult(
                success=False,
                url=full_url,
                method=method,
                elapsed_ms=elapsed_ms,
                error=f"Connection failed: {e}",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return ApiReplayResult(
                success=False,
                url=full_url,
                method=method,
                elapsed_ms=elapsed_ms,
                error=f"Request failed: {e}",
            )

    @staticmethod
    def resolve_auth_headers(
        auth_type: str | None,
        credentials: str | None,
    ) -> dict[str, str]:
        """Build auth headers from API entry's auth_type + credentials.

        Supports:
            - ``bearer``: ``Authorization: Bearer <token>``
            - ``api_key``: ``X-API-Key: <key>``  (also tries Authorization header)
            - ``basic``: ``Authorization: Basic <base64>``
            - ``custom``: Treat credentials as ``Header-Name: value`` format
            - ``none`` / empty: No auth headers

        Returns:
            Dict of headers to merge into the request.
        """
        if not auth_type or not credentials or auth_type.lower() == "none":
            return {}

        auth_type_lower = auth_type.lower()

        if auth_type_lower == "bearer":
            # Strip "Bearer " prefix if already present
            token = credentials.strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
            return {"Authorization": f"Bearer {token}"}

        if auth_type_lower == "api_key":
            return {"X-API-Key": credentials.strip()}

        if auth_type_lower == "basic":
            return {"Authorization": f"Basic {credentials.strip()}"}

        if auth_type_lower == "custom":
            # Try to parse "Header-Name: value" format
            if ": " in credentials:
                name, _, value = credentials.partition(": ")
                return {name.strip(): value.strip()}
            # Fallback: treat as Authorization value
            return {"Authorization": credentials.strip()}

        # Unknown auth type — try as Authorization header
        return {"Authorization": credentials.strip()}

    @staticmethod
    def find_endpoint_in_api(
        endpoints_json: str | None,
        endpoint_path: str | None = None,
        method: str | None = None,
    ) -> dict[str, Any] | None:
        """Find a matching endpoint definition from the API's endpoints list.

        The endpoints JSON (from ``ApiEntry.endpoints``) is a list of dicts,
        each with at least ``method``, ``path``, and optionally ``params``,
        ``request_body``, ``response_sample``.

        Returns the first matching endpoint dict, or None.
        """
        if not endpoints_json:
            return None

        try:
            endpoints = json.loads(endpoints_json)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(endpoints, list):
            return None

        for ep in endpoints:
            if not isinstance(ep, dict):
                continue

            ep_path = ep.get("path", "")
            ep_method = ep.get("method", "GET").upper()

            # Match by endpoint path
            if endpoint_path:
                # Support parameterized paths: /users/{id} matches /users/123
                pattern = re.sub(r"\{[^}]+\}", r"[^/]+", ep_path)
                if re.fullmatch(pattern, endpoint_path) or ep_path == endpoint_path:
                    if method and ep_method != method.upper():
                        continue
                    return ep

            # Match by method only (if no path specified)
            if not endpoint_path and method and ep_method == method.upper():
                return ep

        return None

    @staticmethod
    def format_endpoints_list(endpoints_json: str | None, max_items: int = 20) -> str:
        """Format the endpoints list for display."""
        if not endpoints_json:
            return "(no endpoints defined)"

        try:
            endpoints = json.loads(endpoints_json)
        except (json.JSONDecodeError, TypeError):
            return "(invalid endpoints JSON)"

        if not isinstance(endpoints, list) or not endpoints:
            return "(no endpoints)"

        lines: list[str] = []
        for i, ep in enumerate(endpoints[:max_items]):
            if not isinstance(ep, dict):
                continue
            method = ep.get("method", "?").upper()
            path = ep.get("path", "?")
            desc = ep.get("description", "")
            line = f"  {method:6s} {path}"
            if desc:
                line += f"  — {desc[:60]}"
            lines.append(line)

        if len(endpoints) > max_items:
            lines.append(f"  ... and {len(endpoints) - max_items} more")

        return "\n".join(lines) if lines else "(no endpoints)"
