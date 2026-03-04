"""Network interceptor for the browser tool.

Attaches to a Playwright page and captures XHR/fetch traffic.  Filters out
static assets (images, CSS, fonts, JS bundles) and groups captured requests
by base URL + path pattern.  Captured API patterns can be stored in the
existing Captain Claw APIs memory for later direct replay.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from captain_claw.logging import get_logger

log = get_logger(__name__)

# File extensions and content-types treated as static assets (ignored).
_STATIC_EXTENSIONS: frozenset[str] = frozenset({
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".map", ".webp",
    ".mp4", ".webm", ".mp3", ".wav",
})

_STATIC_CONTENT_TYPES: frozenset[str] = frozenset({
    "text/css", "text/javascript", "application/javascript",
    "image/", "font/", "audio/", "video/",
})

# URL path patterns that are almost always static asset loaders.
_STATIC_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.(chunk|bundle)\.(js|css)$", re.IGNORECASE),
    re.compile(r"/__webpack", re.IGNORECASE),
    re.compile(r"/static/(js|css|media)/", re.IGNORECASE),
    re.compile(r"/assets/(js|css|fonts|images)/", re.IGNORECASE),
    re.compile(r"/_next/static/", re.IGNORECASE),
    re.compile(r"/favicon", re.IGNORECASE),
    re.compile(r"/manifest\.json$", re.IGNORECASE),
    re.compile(r"/robots\.txt$", re.IGNORECASE),
    re.compile(r"/service-?worker", re.IGNORECASE),
)


# ---------- data classes -----------------------------------------------------


@dataclass
class CapturedRequest:
    """A single intercepted HTTP request/response pair."""

    method: str
    url: str
    path: str
    base_url: str  # scheme + host
    request_headers: dict[str, str] = field(default_factory=dict)
    request_body: str | None = None
    status: int = 0
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: str | None = None
    content_type: str = ""
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def summary_line(self) -> str:
        """One-line summary for display."""
        body_hint = ""
        if self.response_body:
            body_hint = f" ({len(self.response_body)} chars)"
        return (
            f"{self.method:6s} {self.status:3d}  {self.path}"
            f"  [{self.content_type.split(';')[0]}]{body_hint}"
        )


@dataclass
class ApiSummary:
    """A grouped API discovered from captured traffic."""

    name: str
    base_url: str
    endpoints: list[dict[str, str]]  # [{method, path, description}]
    auth_type: str  # bearer, api_key, basic, cookie, none
    auth_header_value: str  # e.g. the actual Bearer token (for API replay)
    description: str
    sample_requests: int


# ---------- helpers ----------------------------------------------------------


def _is_static_request(url: str, content_type: str) -> bool:
    """Return True if the request is likely a static asset."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Extension check
    for ext in _STATIC_EXTENSIONS:
        if path.endswith(ext):
            return True

    # Content-type check
    ct_lower = content_type.lower()
    for prefix in _STATIC_CONTENT_TYPES:
        if ct_lower.startswith(prefix):
            return True

    # Path pattern check
    for pattern in _STATIC_PATH_PATTERNS:
        if pattern.search(path):
            return True

    return False


def _infer_auth_type(headers: dict[str, str]) -> tuple[str, str]:
    """Detect auth type and value from request headers.

    Returns (auth_type, auth_value).
    """
    auth_header = headers.get("authorization", headers.get("Authorization", ""))
    if auth_header:
        if auth_header.lower().startswith("bearer "):
            return "bearer", auth_header
        if auth_header.lower().startswith("basic "):
            return "basic", auth_header
        return "api_key", auth_header

    # Check for API key in common header names
    for key in ("x-api-key", "X-API-Key", "x-api-token", "X-Api-Token", "api-key", "apikey"):
        if key in headers:
            return "api_key", headers[key]

    # Cookie-based auth (if cookie header present)
    if headers.get("cookie") or headers.get("Cookie"):
        return "cookie", ""

    return "none", ""


def _parameterize_path(path: str) -> str:
    """Replace likely dynamic path segments with {id} placeholders.

    Examples:
        /api/users/12345       → /api/users/{id}
        /api/v1/posts/abc-def  → /api/v1/posts/{id}
        /api/items/42/comments → /api/items/{id}/comments
    """
    segments = path.strip("/").split("/")
    result: list[str] = []
    for seg in segments:
        # UUID pattern
        if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", seg, re.I):
            result.append("{id}")
        # Numeric ID
        elif re.match(r"^\d+$", seg):
            result.append("{id}")
        # Hash-like string (8+ hex chars)
        elif re.match(r"^[0-9a-f]{8,}$", seg, re.I) and not seg.startswith("v"):
            result.append("{id}")
        else:
            result.append(seg)
    return "/" + "/".join(result)


def _group_endpoints(
    captures: list[CapturedRequest],
) -> dict[str, list[CapturedRequest]]:
    """Group captures by base_url."""
    groups: dict[str, list[CapturedRequest]] = defaultdict(list)
    for cap in captures:
        groups[cap.base_url].append(cap)
    return dict(groups)


# ---------- NetworkInterceptor -----------------------------------------------


class NetworkInterceptor:
    """Attaches to a Playwright page and captures XHR/fetch traffic.

    Filters out static assets.  Groups requests by base_url + path pattern.
    """

    def __init__(
        self,
        max_captures: int = 500,
        max_body_bytes: int = 10_000,
        filter_static: bool = True,
    ) -> None:
        self._max_captures = max_captures
        self._max_body_bytes = max_body_bytes
        self._filter_static = filter_static
        self._captures: list[CapturedRequest] = []
        self._pending: dict[str, CapturedRequest] = {}  # url → partial capture
        self._is_recording: bool = False
        self._page: Any | None = None

    # -- attach / detach ------------------------------------------------------

    async def attach(self, page: Any) -> None:
        """Register request/response event handlers on the page."""
        self._page = page
        page.on("request", self._on_request)
        page.on("response", self._on_response)
        log.info("Network interceptor attached")

    async def detach(self) -> None:
        """Remove event handlers."""
        if self._page is not None:
            try:
                self._page.remove_listener("request", self._on_request)
                self._page.remove_listener("response", self._on_response)
            except Exception:
                pass
            self._page = None
        log.info("Network interceptor detached")

    # -- recording control ----------------------------------------------------

    def start_recording(self) -> None:
        """Start capturing network traffic."""
        self._is_recording = True
        log.info("Network recording started")

    def stop_recording(self) -> None:
        """Stop capturing (keeps existing data)."""
        self._is_recording = False
        log.info("Network recording stopped", captures=len(self._captures))

    def clear(self) -> None:
        """Discard all captured data."""
        self._captures.clear()
        self._pending.clear()
        log.info("Network captures cleared")

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def capture_count(self) -> int:
        return len(self._captures)

    # -- event handlers -------------------------------------------------------

    def _on_request(self, request: Any) -> None:
        """Handle a Playwright request event."""
        if not self._is_recording:
            return
        if len(self._captures) >= self._max_captures:
            return

        url = request.url
        method = request.method
        parsed = urlparse(url)

        # Quick skip for data: and blob: URLs
        if parsed.scheme in ("data", "blob", "chrome-extension"):
            return

        # Build partial capture
        headers = {}
        try:
            headers = dict(request.headers)
        except Exception:
            pass

        body = None
        try:
            post_data = request.post_data
            if post_data and len(post_data) <= self._max_body_bytes:
                body = post_data
        except Exception:
            pass

        cap = CapturedRequest(
            method=method,
            url=url,
            path=parsed.path or "/",
            base_url=f"{parsed.scheme}://{parsed.netloc}",
            request_headers=headers,
            request_body=body,
            timestamp=time.time(),
        )

        # Store as pending until we get the response
        self._pending[url] = cap

    async def _on_response(self, response: Any) -> None:
        """Handle a Playwright response event."""
        if not self._is_recording:
            return

        url = response.url
        cap = self._pending.pop(url, None)
        if cap is None:
            return

        cap.status = response.status
        cap.duration_ms = (time.time() - cap.timestamp) * 1000

        # Get content type
        content_type = ""
        try:
            headers = dict(response.headers)
            cap.response_headers = headers
            content_type = headers.get("content-type", "")
            cap.content_type = content_type
        except Exception:
            pass

        # Filter static assets
        if self._filter_static and _is_static_request(url, content_type):
            return

        # Try to capture response body for API-like responses
        if self._is_api_like(content_type):
            try:
                body_text = await response.text()
                if len(body_text) <= self._max_body_bytes:
                    cap.response_body = body_text
                else:
                    cap.response_body = body_text[: self._max_body_bytes] + "... [truncated]"
            except Exception:
                pass

        self._captures.append(cap)
        log.debug(
            "Captured request",
            method=cap.method,
            status=cap.status,
            path=cap.path,
            base=cap.base_url,
        )

    @staticmethod
    def _is_api_like(content_type: str) -> bool:
        """Return True if the content type suggests an API response."""
        ct = content_type.lower()
        return any(
            hint in ct
            for hint in ("json", "xml", "text/plain", "graphql", "protobuf")
        )

    # -- query captured data --------------------------------------------------

    def get_captures(self, *, filter_static: bool = True) -> list[CapturedRequest]:
        """Return captured requests, optionally filtering static assets."""
        if not filter_static:
            return list(self._captures)
        return [
            c for c in self._captures
            if not _is_static_request(c.url, c.content_type)
        ]

    def format_capture_list(self, max_items: int = 50) -> str:
        """Return a formatted string listing captured API calls."""
        captures = self.get_captures()
        if not captures:
            return "No API calls captured."

        lines: list[str] = [f"Captured {len(captures)} API call(s):\n"]
        for i, cap in enumerate(captures[:max_items], 1):
            lines.append(f"  {i:3d}. {cap.summary_line()}")

        if len(captures) > max_items:
            lines.append(f"\n  ... and {len(captures) - max_items} more.")

        return "\n".join(lines)

    def summarize_apis(self) -> list[ApiSummary]:
        """Group captured requests by base_url and extract API summaries.

        Returns a list of ApiSummary objects suitable for storage in the
        Captain Claw APIs memory system via SessionManager.create_api().
        """
        captures = self.get_captures()
        if not captures:
            return []

        groups = _group_endpoints(captures)
        summaries: list[ApiSummary] = []

        for base_url, caps in groups.items():
            # Deduplicate endpoints by (method, parameterized_path)
            seen_endpoints: dict[tuple[str, str], dict[str, str]] = {}
            auth_types: dict[str, int] = defaultdict(int)
            auth_values: dict[str, str] = {}

            for cap in caps:
                param_path = _parameterize_path(cap.path)
                key = (cap.method, param_path)

                if key not in seen_endpoints:
                    # Build a short description from the status and content type
                    desc_parts: list[str] = []
                    if cap.status:
                        desc_parts.append(f"HTTP {cap.status}")
                    ct_short = cap.content_type.split(";")[0].strip() if cap.content_type else ""
                    if ct_short:
                        desc_parts.append(ct_short)
                    seen_endpoints[key] = {
                        "method": cap.method,
                        "path": param_path,
                        "description": " | ".join(desc_parts) if desc_parts else "",
                    }

                # Track auth patterns
                auth_type, auth_value = _infer_auth_type(cap.request_headers)
                auth_types[auth_type] += 1
                if auth_value and auth_type not in auth_values:
                    auth_values[auth_type] = auth_value

            # Pick dominant auth type
            dominant_auth = max(auth_types, key=auth_types.get) if auth_types else "none"
            dominant_auth_value = auth_values.get(dominant_auth, "")

            endpoints = list(seen_endpoints.values())

            # Generate a name from the domain
            parsed = urlparse(base_url)
            domain_parts = parsed.netloc.split(".")
            if len(domain_parts) >= 2:
                api_name = domain_parts[-2].capitalize() + " API"
            else:
                api_name = parsed.netloc + " API"

            summaries.append(
                ApiSummary(
                    name=api_name,
                    base_url=base_url,
                    endpoints=endpoints,
                    auth_type=dominant_auth,
                    auth_header_value=dominant_auth_value,
                    description=(
                        f"API calls captured from browsing {parsed.netloc}. "
                        f"{len(endpoints)} unique endpoint(s) discovered."
                    ),
                    sample_requests=len(caps),
                )
            )

        return summaries
