"""A structured, paginated, retrying Google Drive REST client.

The ``google_drive`` *tool* answers an LLM in prose; a filesystem mount needs
data. This module is the shared client both can sit on: it returns dicts, pages
through large folders, retries rate limits, and accepts any of the three Drive
scopes rather than demanding full read/write.

It is deliberately auth-agnostic. Tokens arrive through a ``token_provider``
callback, so the same client works with today's global Google connection and
with a future per-user one — only the provider changes, never this code.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from captain_claw.logging import get_logger

log = get_logger(__name__)

_DRIVE_API = "https://www.googleapis.com/drive/v3"

# Any of these grants read access; the tool used to demand the first one only,
# which made a read-only mount impossible.
DRIVE_SCOPES = (
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
)

_FILE_FIELDS = (
    "id,name,mimeType,size,modifiedTime,createdTime,parents,"
    "webViewLink,md5Checksum,trashed"
)

FOLDER_MIME = "application/vnd.google-apps.folder"

# Native Google types have no bytes of their own — a read is really an export.
# Standardised on the higher-quality targets (Sheet -> xlsx, Slides -> pptx),
# which the existing _extract_*_markdown helpers then render far better than
# Drive's flat CSV/plain-text exports.
GOOGLE_EXPORT: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": ("text/markdown", ".md"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    "application/vnd.google-apps.drawing": ("image/svg+xml", ".svg"),
}


class DriveError(RuntimeError):
    """A Drive call failed in a way the caller should surface, not retry."""


class DriveNotConnected(DriveError):
    """No usable token — Google isn't connected, or lacks a Drive scope."""


@dataclass
class DriveFile:
    """One Drive entry, normalised to the fields a mount cares about."""

    id: str
    name: str
    mime_type: str
    size: int | None = None
    modified_time: str = ""
    created_time: str = ""
    md5: str = ""
    web_view_link: str = ""
    parents: list[str] = field(default_factory=list)

    @property
    def is_folder(self) -> bool:
        return self.mime_type == FOLDER_MIME

    @property
    def is_google_native(self) -> bool:
        return self.mime_type.startswith("application/vnd.google-apps.")

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "DriveFile":
        raw_size = d.get("size")
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            mime_type=str(d.get("mimeType", "")),
            size=int(raw_size) if raw_size not in (None, "") else None,
            modified_time=str(d.get("modifiedTime", "")),
            created_time=str(d.get("createdTime", "")),
            md5=str(d.get("md5Checksum", "")),
            web_view_link=str(d.get("webViewLink", "")),
            parents=list(d.get("parents", []) or []),
        )


# A coroutine returning a bearer token (and its granted scope string). Kept as a
# callback so per-user token resolution can be swapped in without touching this
# module.
TokenProvider = Callable[[], Awaitable[tuple[str, str]]]


def escape_query_value(value: str) -> str:
    """Escape a value for a Drive ``q`` string literal.

    Drive query literals are single-quoted; a backslash or apostrophe in a
    folder id or name closes the literal early (or worse). The tool's list
    action interpolated ``folder_id`` raw — a crafted id could rewrite the
    query.
    """
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


class DriveClient:
    """Async Drive REST client returning structured data."""

    def __init__(
        self,
        token_provider: TokenProvider,
        *,
        timeout: float = 60.0,
        max_retries: int = 4,
        max_page_size: int = 1000,
    ) -> None:
        self._token_provider = token_provider
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_page_size = max_page_size
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": "Captain Claw (Drive VFS)"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _headers(self) -> dict[str, str]:
        token, scope = await self._token_provider()
        if not token:
            raise DriveNotConnected(
                "Google account is not connected. Connect it in "
                "Flight Deck → Connections → Google."
            )
        # An empty scope string means the provider couldn't tell us; allow it
        # rather than second-guessing (the API will 403 if truly missing).
        if scope:
            granted = set(scope.split())
            if not granted.intersection(DRIVE_SCOPES):
                raise DriveNotConnected(
                    "The Google connection has no Drive scope. Reconnect and "
                    "grant Drive access (read-only is enough)."
                )
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Core request with backoff
    # ------------------------------------------------------------------

    async def _request(
        self, method: str, url: str, *, sleep=asyncio.sleep, **kwargs: Any
    ) -> httpx.Response:
        """One Drive call, retrying rate limits and transient 5xx.

        429 and 5xx are the only retryable statuses; ``Retry-After`` is honoured
        when present, otherwise exponential backoff. Everything else raises
        immediately — a 404 or 403 will not fix itself.
        """
        headers = await self._headers()
        headers.update(kwargs.pop("headers", {}) or {})
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._http().request(method, url, headers=headers, **kwargs)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    raise DriveError(f"Drive request failed: {exc}") from exc
                await sleep(min(2**attempt, 8))
                continue

            if resp.status_code < 400:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                delay = self._retry_after(resp) or min(2**attempt, 8)
                log.debug(
                    "Drive %s retrying", resp.status_code, url=url, attempt=attempt, delay=delay
                )
                await sleep(delay)
                continue
            self._raise_for_status(resp)

        # Only reached if every attempt was a connect/timeout error.
        raise DriveError(f"Drive request failed after retries: {last_exc}")

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        status = resp.status_code
        try:
            message = resp.json().get("error", {}).get("message", "")
        except Exception:
            message = resp.text[:200]
        if status == 401:
            raise DriveNotConnected("Google authentication expired. Reconnect the account.")
        if status == 404:
            raise DriveError("Not found (check the file or folder id).")
        raise DriveError(f"Drive API error ({status}): {message}")

    # ------------------------------------------------------------------
    # Listing (paginated)
    # ------------------------------------------------------------------

    async def list_folder(
        self,
        folder_id: str = "root",
        *,
        order_by: str = "folder,name",
        max_files: int | None = None,
        sleep=asyncio.sleep,
    ) -> tuple[list[DriveFile], bool]:
        """Return a folder's direct children, following pagination.

        Returns ``(files, truncated)`` — ``truncated`` is True when *max_files*
        cut the listing short, so a caller can warn instead of silently showing
        a partial tree. The old tool requested ``nextPageToken`` and never read
        it, so any folder over 100 children was quietly incomplete.
        """
        q = f"'{escape_query_value(folder_id)}' in parents and trashed = false"
        files: list[DriveFile] = []
        page_token: str | None = None
        truncated = False

        while True:
            page_size = self._max_page_size
            if max_files is not None:
                remaining = max_files - len(files)
                if remaining <= 0:
                    truncated = True
                    break
                page_size = min(page_size, remaining)
            params: dict[str, Any] = {
                "q": q,
                "fields": f"nextPageToken,files({_FILE_FIELDS})",
                "pageSize": page_size,
                "orderBy": order_by,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            resp = await self._request("GET", f"{_DRIVE_API}/files", params=params, sleep=sleep)
            data = resp.json()
            files.extend(DriveFile.from_api(f) for f in data.get("files", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return files, truncated

    async def get_metadata(self, file_id: str, *, sleep=asyncio.sleep) -> DriveFile:
        resp = await self._request(
            "GET",
            f"{_DRIVE_API}/files/{file_id}",
            params={"fields": _FILE_FIELDS, "supportsAllDrives": "true"},
            sleep=sleep,
        )
        return DriveFile.from_api(resp.json())

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    async def download(self, file_id: str, *, sleep=asyncio.sleep) -> bytes:
        """Raw bytes of a binary file. Not valid for Google-native types."""
        resp = await self._request(
            "GET",
            f"{_DRIVE_API}/files/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
            sleep=sleep,
        )
        return resp.content

    async def export(self, file_id: str, export_mime: str, *, sleep=asyncio.sleep) -> bytes:
        """Export a Google-native doc to *export_mime*."""
        resp = await self._request(
            "GET",
            f"{_DRIVE_API}/files/{file_id}/export",
            params={"mimeType": export_mime},
            sleep=sleep,
        )
        return resp.content

    async def fetch(self, f: DriveFile, *, sleep=asyncio.sleep) -> tuple[bytes, str]:
        """Fetch a file's content, exporting if it is Google-native.

        Returns ``(bytes, effective_extension)`` — the extension reflects the
        export target for native docs (e.g. ``.xlsx`` for a Sheet), so callers
        route the bytes to the right converter.
        """
        if f.is_google_native:
            mapping = GOOGLE_EXPORT.get(f.mime_type)
            if mapping is None:
                raise DriveError(f"No export path for Google type {f.mime_type}")
            export_mime, ext = mapping
            return await self.export(f.id, export_mime, sleep=sleep), ext
        from pathlib import Path

        return await self.download(f.id, sleep=sleep), Path(f.name).suffix.lower()


# ---------------------------------------------------------------------------
# Token providers
# ---------------------------------------------------------------------------


async def global_token_provider() -> tuple[str, str]:
    """Resolve a token from the deployment-wide Google connection.

    This is the connection every Google tool uses today. It is correct for a
    single-operator deployment; a per-user provider (resolving the mount
    owner's own token) is the multi-tenant successor and is the ONLY piece that
    has to change to get there — the client and the mount stay as they are.
    """
    from captain_claw.google_oauth_manager import GoogleOAuthManager
    from captain_claw.session import get_session_manager

    mgr = GoogleOAuthManager(get_session_manager())
    tokens = await mgr.get_tokens()
    if not tokens or not tokens.access_token:
        return "", ""
    return tokens.access_token, (tokens.scope or "")


def make_client(token_provider: TokenProvider | None = None) -> DriveClient:
    """A DriveClient over the given provider, or the global connection."""
    return DriveClient(token_provider or global_token_provider)
