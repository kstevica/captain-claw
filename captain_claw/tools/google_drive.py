"""Google Drive tool for listing, searching, reading, and writing files.

Uses the Google Drive REST API v3 via httpx with OAuth2 Bearer tokens
managed by :class:`~captain_claw.google_oauth_manager.GoogleOAuthManager`.
No additional Google SDK dependencies are required.
"""

from __future__ import annotations

import asyncio
import contextvars
import io
import json
import mimetypes
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.drive_client import FOLDER_MIME, GOOGLE_EXPORT
from captain_claw.google_ids import (
    drive_id_from_url,
    drive_resource_key,
    is_google_drive_url,
)
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRIVE_API = "https://www.googleapis.com/drive/v3"
_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
# Any Drive scope grants read; write actions additionally need the check below.
_DRIVE_READ_SCOPES = frozenset({
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
})
_DRIVE_WRITE_SCOPES = frozenset({
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
})
_WRITE_ACTIONS = frozenset({"upload", "create", "update"})
# Scopes that reach a file this app did not create — a link shared with the
# user, a colleague's Doc. drive.file alone sees only files the app created
# or the user picked, so a pasted link 404s under it.
_DRIVE_LINK_SCOPES = frozenset({
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.readonly",
})

# Fields to request from the files endpoint.
_FILE_FIELDS = "id,name,mimeType,size,modifiedTime,createdTime,parents,webViewLink,owners"
_LIST_FIELDS = f"nextPageToken,files({_FILE_FIELDS})"

# list / search: Drive's largest page, the most one call returns, and a
# bound on pages followed (Drive may return short pages).
_MAX_PAGE_SIZE = 1000
_MAX_LIST_RESULTS = 1000
_MAX_LIST_PAGES = 50

# Google Workspace MIME types and their export mappings.
_GOOGLE_EXPORT_MAP: dict[str, tuple[str, str]] = {
    # mime_type → (export_mime, file_extension)
    "application/vnd.google-apps.document": ("text/markdown", ".md"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
    "application/vnd.google-apps.drawing": ("image/svg+xml", ".svg"),
}

# Binary MIME types that can be handled by existing extract tools.
_EXTRACTABLE_MIMES: dict[str, str] = {
    "application/pdf": "pdf_extract",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx_extract",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx_extract",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx_extract",
}

# Maximum content size for read operations (bytes).
_MAX_READ_BYTES = 500_000  # 500 KB
# Rows per tab a Sheet read renders (the read stays under _MAX_READ_BYTES).
_SHEET_READ_MAX_ROWS = 1000
# Largest file read (download + extract) or download will fetch.
_MAX_DOWNLOAD_BYTES = 50_000_000  # 50 MB

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# The extension each extract tool insists on — the temp file of a read and
# a download whose Drive name lacks it (an uploaded PDF titled "Contract").
_SUFFIX_BY_MIME = {
    "application/pdf": ".pdf", _DOCX_MIME: ".docx",
    _XLSX_MIME: ".xlsx", _PPTX_MIME: ".pptx",
}

# read: Sheets and Slides export as XLSX / PPTX and go through the extract
# tools — Drive's CSV export holds only a Sheet's first tab, and its text
# export flattens a deck. The flat export stays as the fallback.
_READ_VIA_OFFICE_EXPORT = frozenset({
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
})

# download: export formats an output_path extension may ask for, per Google
# type. Anything else gets the drive_client.GOOGLE_EXPORT default (Doc → .md,
# Sheet → .xlsx, Slides → .pptx), which the read/extract tools render best.
_EXPORT_BY_SUFFIX: dict[str, dict[str, str]] = {
    "application/vnd.google-apps.document": {
        ".md": "text/markdown", ".txt": "text/plain", ".html": "text/html",
        ".docx": _DOCX_MIME, ".pdf": "application/pdf",
    },
    "application/vnd.google-apps.spreadsheet": {
        ".xlsx": _XLSX_MIME, ".csv": "text/csv", ".pdf": "application/pdf",
    },
    "application/vnd.google-apps.presentation": {
        ".pptx": _PPTX_MIME, ".txt": "text/plain", ".pdf": "application/pdf",
    },
    "application/vnd.google-apps.drawing": {
        ".svg": "image/svg+xml", ".png": "image/png", ".pdf": "application/pdf",
    },
}
_TEXT_EXPORTS = frozenset({"text/markdown", "text/plain", "text/html"})

# The write tool's saved/ categories; a relative download path outside them
# is filed under downloads/.
_SAVED_CATEGORIES = frozenset({
    "downloads", "media", "output", "scripts", "showcase",
    "skills", "summaries", "tmp", "tools",
})

# Which tool reads a downloaded file, by extension (default: read).
_READER_BY_SUFFIX = {
    ".pdf": "pdf_extract", ".docx": "docx_extract",
    ".xlsx": "xlsx_extract", ".pptx": "pptx_extract",
    ".png": "image_vision", ".jpg": "image_vision", ".jpeg": "image_vision",
}

# Inline base64 images in a Docs markdown export (``![](data:image/png;base64,…)``)
# run to megabytes and would bury the text — replaced with "[image]".
_BASE64_IMG_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+")


def _strip_base64_images(text: str) -> str:
    """Remove inline base64 image data from text to prevent context bloat."""
    cleaned = _BASE64_IMG_RE.sub("[image]", text)
    if len(cleaned) < len(text):
        log.debug(
            "stripped base64 images",
            original_len=len(text),
            cleaned_len=len(cleaned),
        )
    return cleaned


# "<file id>/<resource key>" for the file this call works on, from a pasted
# URL's ?resourcekey= — older link-shared files 404 without it. Per call
# (a context variable), as tool calls may run concurrently.
_RESOURCE_KEY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "google_drive_resource_key", default="",
)


def agent_offers_google_drive(agent: Any) -> bool:
    """Whether the calling agent's model can actually call google_drive.

    The web tools only redirect a Drive link to google_drive when it can:
    a sister session never registers it, and nano mode cuts it from the tool
    list the model sees. Pointing those agents at a tool they don't have
    would strand a link-shared Doc that an anonymous fetch could still read.
    ``agent`` is the ``_agent`` every guarded tool call carries; None (a
    direct call) keeps the redirect.
    """
    if agent is None:
        return True
    try:
        if not agent.tools.has_tool("google_drive"):
            return False
    except Exception:
        return True
    instructions = getattr(agent, "instructions", None)
    return not bool(getattr(instructions, "use_nano", False))


async def google_drive_reads_links() -> bool:
    """True when google_drive can open a Drive link the user pastes.

    That needs Google connected with a scope that reaches files this app did
    not create (drive / drive.readonly). Under drive.file only, a pasted
    link-shared Doc 404s, so the web tools let their anonymous fetch run
    instead of redirecting here. An unreported scope counts as yes — the
    tool doesn't refuse on it either.
    """
    from captain_claw.google_oauth_manager import (
        GoogleOAuthManager,
        is_google_connected_cached,
    )
    from captain_claw.session import get_session_manager

    if not is_google_connected_cached():
        return False
    try:
        tokens = await GoogleOAuthManager(get_session_manager()).get_tokens()
    except Exception as exc:  # FD refused / unreachable: google_drive would fail too
        log.debug("google_drive link check failed", error=str(exc))
        return False
    if not tokens:
        return False
    granted = set(tokens.scope.split()) if tokens.scope else set()
    return not granted or bool(granted & _DRIVE_LINK_SCOPES)


def _missing_suffix(name: str, mime: str) -> str:
    """The extension a downloaded file needs for its reader, or "".

    Extract tools require theirs exactly (``Contract v2.1`` → ``.pdf``
    appended); any other type gets one only when the name has none.
    """
    current = Path(name).suffix.lower()
    wanted = _SUFFIX_BY_MIME.get(mime)
    if wanted:
        return "" if current == wanted else wanted
    if current or not mime or mime == "application/octet-stream":
        return ""
    return mimetypes.guess_extension(mime) or ""


def _tab_hit_row_cap(markdown: str, max_rows: int) -> bool:
    """True if a tab in xlsx_extract output stopped at its row cap.

    The extractor drops rows past the cap without saying so; a tab rendered
    with exactly *max_rows* rows (header + separator + body lines) may
    continue.
    """
    rows = 0
    for line in markdown.splitlines():
        if line.startswith("## Sheet:"):
            rows = 0
        elif line.startswith("| "):
            rows += 1
            if rows - 1 >= max_rows:  # one line is the | --- | separator
                return True
    return False


def _resolve_drive_ref(value: str) -> str | None:
    """A file_id / folder_id argument as a bare id.

    Drive/Docs/Sheets/Slides URLs give up their id; any other URL is not a
    Drive reference (None). Everything else passes through unchanged — the
    API is the judge of a bare id, and ``root`` must keep working.
    """
    ref = value.strip()
    if is_google_drive_url(ref):
        return drive_id_from_url(ref)
    if "://" in ref:
        return None
    return ref


def _safe_filename(name: str, fallback: str) -> str:
    """A Drive file name as one safe path component."""
    cleaned = re.sub(r"[\x00-\x1f/\\]+", "_", name or "").strip().lstrip(".").strip()
    return cleaned[:200] or fallback


class _DownloadPathError(ValueError):
    """output_path points somewhere a download may not write."""


class GoogleDriveTool(Tool):
    """Interact with Google Drive: list, search, read, download, upload, create, and update files."""

    name = "google_drive"
    description = (
        "Read and write the user's Google Drive, including Docs, Sheets, Slides "
        "and shared drives. Actions: list (folder contents), search (find files "
        "by name or content), read (content inline: Docs as markdown, Sheets/"
        "Slides exported, PDF/DOCX/XLSX/PPTX extracted), info (metadata), "
        "download (save a local copy and return its path), upload (send a local "
        "file to Drive), create (new file on Drive), update (replace a file's "
        "content). file_id and folder_id also accept a full Drive/Docs/Sheets/"
        "Slides URL."
    )
    timeout_seconds = 120.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list", "search", "read", "info", "download",
                    "upload", "create", "update",
                ],
                "description": "The action to perform.",
            },
            "file_id": {
                "type": "string",
                "description": (
                    "Google Drive file ID or the file's Drive/Docs/Sheets/Slides "
                    "URL (for read, info, download, update actions)."
                ),
            },
            "folder_id": {
                "type": "string",
                "description": (
                    "Folder ID or Drive folder URL to list or upload into. "
                    "Defaults to 'root'."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "download only: where to save, inside the saved/ area "
                    "(default saved/downloads/<session>/<file name>). For "
                    "Google Docs/Sheets/Slides the extension picks the export "
                    "format (.md/.docx/.pdf/.txt/.html for Docs, .xlsx/.csv/.pdf "
                    "for Sheets, .pptx/.pdf/.txt for Slides); default Docs → .md, "
                    "Sheets → .xlsx, Slides → .pptx."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search query text (for search action).",
            },
            "max_results": {
                "type": "number",
                "description": (
                    "Maximum number of results for list (default 100) or search "
                    "(default 20); up to 1000."
                ),
            },
            "page_token": {
                "type": "string",
                "description": (
                    "list/search: continue from the page_token a previous "
                    "result ended with (it says when more files exist)."
                ),
            },
            "local_path": {
                "type": "string",
                "description": "Local file path (for upload action or update from file).",
            },
            "name": {
                "type": "string",
                "description": "File name (for upload or create actions).",
            },
            "content": {
                "type": "string",
                "description": "Text content (for create or update actions).",
            },
            "mime_type": {
                "type": "string",
                "description": (
                    "MIME type for create action. Use 'application/vnd.google-apps.document' "
                    "for Google Doc, 'application/vnd.google-apps.spreadsheet' for Google Sheet, "
                    "or 'text/plain' for plain text."
                ),
            },
            "order_by": {
                "type": "string",
                "description": (
                    "Sort order for list/search (e.g. 'modifiedTime desc', "
                    "'name', 'createdTime desc'). Default: 'modifiedTime desc'."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Google Drive Tool)"},
        )

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        # Pop injected kwargs that tools receive from the registry; download
        # needs the saved-area ones to place its file.
        runtime = {
            key: kwargs.pop(key, None)
            for key in (
                "_runtime_base_path", "_saved_base_path", "_session_id",
                "_file_registry", "_task_id",
            )
        }
        kwargs.pop("_abort_event", None)

        # A pasted Drive/Docs URL works wherever an id is expected.
        resource_key = ""
        for key in ("file_id", "folder_id"):
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                resolved = _resolve_drive_ref(value)
                if not resolved:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Could not find a Google Drive id in {key}={value!r}. "
                            "Pass the id (the part after /d/, ?id= or /folders/ "
                            "in a Drive link) or a Drive/Docs URL that contains "
                            "one; published (/d/e/...) links carry no file id."
                        ),
                    )
                if key == "file_id" and is_google_drive_url(value.strip()):
                    rkey = drive_resource_key(value)
                    resource_key = f"{resolved}/{rkey}" if rkey else ""
                kwargs[key] = resolved

        handlers = {
            "list": self._action_list,
            "search": self._action_search,
            "read": self._action_read,
            "info": self._action_info,
            "download": self._action_download,
            "upload": self._action_upload,
            "create": self._action_create,
            "update": self._action_update,
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use one of: {', '.join(handlers)}",
            )
        if action == "download":
            kwargs["runtime"] = runtime

        try:
            token = await self._get_access_token(write=action in _WRITE_ACTIONS)
        except RuntimeError as e:
            return ToolResult(success=False, error=str(e))

        key_token = _RESOURCE_KEY.set(resource_key)
        try:
            return await handler(token, **kwargs)
        except httpx.HTTPStatusError as exc:
            return self._handle_http_error(exc)
        except httpx.HTTPError as exc:
            log.error("Google Drive HTTP error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"HTTP error: {exc}")
        except Exception as exc:
            log.error("Google Drive tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))
        finally:
            _RESOURCE_KEY.reset(key_token)

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    async def _get_access_token(self, *, write: bool = False) -> str:
        """Retrieve a valid Google OAuth access token.

        Raises RuntimeError if Google is not connected, or if the granted
        scopes don't cover the operation. Read actions accept any Drive scope
        including ``drive.readonly``; only write actions require a writable one.
        The tool previously demanded full read/write ``drive`` for *everything*,
        so a read-only connection couldn't even list a folder.
        """
        from captain_claw.google_oauth_manager import GoogleOAuthManager
        from captain_claw.session import get_session_manager

        mgr = GoogleOAuthManager(get_session_manager())
        tokens = await mgr.get_tokens()
        if not tokens:
            if mgr._is_flight_deck_client():
                # Under Flight Deck the agent-local OAuth flow's tokens are
                # discarded — the owner connects THEIR account in Flight Deck.
                raise RuntimeError(
                    "Google account is not connected. Connect your Google "
                    "account in Flight Deck → Connections → Google."
                )
            raise RuntimeError(
                "Google account is not connected. "
                "Please connect via the web UI (Settings > Google OAuth) or "
                "navigate to /auth/google/login in your browser."
            )

        granted = set(tokens.scope.split()) if tokens.scope else set()
        # An empty scope string means the token endpoint didn't report scopes
        # (Flight Deck client mode sometimes omits them); don't block on that —
        # the API returns 403 if the scope is genuinely missing.
        if granted:
            needed = _DRIVE_WRITE_SCOPES if write else _DRIVE_READ_SCOPES
            if not granted.intersection(needed):
                raise RuntimeError(
                    "Google Drive "
                    + ("write " if write else "")
                    + "scope not granted. Reconnect your Google account and "
                    + ("grant Drive edit access." if write else "grant Drive access (read-only is enough).")
                )

        return tokens.access_token

    def _auth_headers(self, token: str) -> dict[str, str]:
        """Build authorization headers (plus the file's resource key, if any)."""
        headers = {"Authorization": f"Bearer {token}"}
        resource_key = _RESOURCE_KEY.get()
        if resource_key:
            headers["X-Goog-Drive-Resource-Keys"] = resource_key
        return headers

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_http_error(exc: httpx.HTTPStatusError) -> ToolResult:
        """Convert HTTP status errors into user-friendly messages."""
        status = exc.response.status_code
        try:
            body = exc.response.json()
            message = body.get("error", {}).get("message", str(exc))
        except Exception:
            message = str(exc)

        if status == 401:
            return ToolResult(
                success=False,
                error="Google authentication expired. Please reconnect your Google account.",
            )
        elif status == 403:
            return ToolResult(
                success=False,
                error=f"Permission denied: {message}",
            )
        elif status == 404:
            return ToolResult(
                success=False,
                error="File not found. Please check the file ID.",
            )
        elif status == 429:
            return ToolResult(
                success=False,
                error="Google Drive rate limit exceeded. Please try again in a moment.",
            )
        else:
            return ToolResult(
                success=False,
                error=f"Google Drive API error ({status}): {message}",
            )

    # ------------------------------------------------------------------
    # Action: list
    # ------------------------------------------------------------------

    async def _action_list(
        self,
        token: str,
        folder_id: str = "root",
        max_results: int | float | None = None,
        order_by: str | None = None,
        page_token: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """List files in a Google Drive folder."""
        limit = max(1, min(int(max_results or 100), _MAX_LIST_RESULTS))
        order = order_by or "modifiedTime desc"

        from captain_claw.drive_client import escape_query_value

        q = f"'{escape_query_value(folder_id)}' in parents and trashed = false"
        params = {
            "q": q,
            "fields": _LIST_FIELDS,
            "orderBy": order,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "corpora": "allDrives",
        }
        files, next_token = await self._list_files(token, params, limit, page_token)
        if not files:
            folder_label = f"folder '{folder_id}'" if folder_id != "root" else "root folder"
            return ToolResult(
                success=True,
                content=f"No files found in {folder_label}." + self._more_hint(files, next_token),
            )

        lines = [f"Files in {'root' if folder_id == 'root' else folder_id} ({len(files)} results):\n"]
        for f in files:
            size = f.get("size", "")
            size_str = f" ({self._format_size(int(size))})" if size else ""
            modified = f.get("modifiedTime", "")[:10] if f.get("modifiedTime") else ""
            is_folder = f.get("mimeType") == "application/vnd.google-apps.folder"
            type_icon = "[folder]" if is_folder else "[file]"

            lines.append(
                f"  {type_icon} {f['name']}{size_str}"
                f"\n    ID: {f['id']}"
                f"  |  Type: {f.get('mimeType', 'unknown')}"
                f"  |  Modified: {modified}"
            )

        return ToolResult(success=True, content="\n".join(lines) + self._more_hint(files, next_token))

    async def _list_files(
        self, token: str, params: dict[str, Any], limit: int, page_token: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Up to *limit* files for a files.list query, following nextPageToken.

        Returns the files and the token for what is left (None once the
        listing is complete), so a cut listing can say so.
        """
        files: list[dict[str, Any]] = []
        next_token = (page_token or "").strip() or None
        for _ in range(_MAX_LIST_PAGES):
            page_params = {**params, "pageSize": min(limit - len(files), _MAX_PAGE_SIZE)}
            if next_token:
                page_params["pageToken"] = next_token
            resp = await self._client.get(
                f"{_DRIVE_API}/files",
                params=page_params,
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            data = resp.json()
            files.extend(data.get("files", []))
            next_token = data.get("nextPageToken") or None
            if not next_token or len(files) >= limit:
                break
        return files[:limit], next_token

    @staticmethod
    def _more_hint(files: list[dict[str, Any]], next_token: str | None) -> str:
        """The closing line of a listing cut short (empty when complete)."""
        if not next_token:
            return ""
        return (
            f"\n\nMore files exist — showing {len(files)}. Repeat this call with "
            f"page_token='{next_token}' for the next page, or a higher max_results "
            f"(up to {_MAX_LIST_RESULTS})."
        )

    # ------------------------------------------------------------------
    # Action: search
    # ------------------------------------------------------------------

    async def _action_search(
        self,
        token: str,
        query: str = "",
        max_results: int | float | None = None,
        order_by: str | None = None,
        page_token: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Search for files across Google Drive."""
        if not query:
            return ToolResult(success=False, error="Search query is required.")

        limit = max(1, min(int(max_results or 20), _MAX_LIST_RESULTS))
        order = order_by or "relevance"

        # Build search query — search by name and full text.
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        q = f"(name contains '{escaped}' or fullText contains '{escaped}') and trashed = false"

        params = {
            "q": q,
            "fields": _LIST_FIELDS,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            # Default corpus is My Drive only; shared-drive files never matched.
            "corpora": "allDrives",
        }
        # 'relevance' is only valid without orderBy (it's the default).
        if order != "relevance":
            params["orderBy"] = order

        files, next_token = await self._list_files(token, params, limit, page_token)
        if not files:
            return ToolResult(
                success=True,
                content=f"No files found matching '{query}'." + self._more_hint(files, next_token),
            )

        lines = [f"Search results for '{query}' ({len(files)} found):\n"]
        for f in files:
            size = f.get("size", "")
            size_str = f" ({self._format_size(int(size))})" if size else ""
            modified = f.get("modifiedTime", "")[:10] if f.get("modifiedTime") else ""
            is_folder = f.get("mimeType") == "application/vnd.google-apps.folder"
            type_icon = "[folder]" if is_folder else "[file]"

            lines.append(
                f"  {type_icon} {f['name']}{size_str}"
                f"\n    ID: {f['id']}"
                f"  |  Type: {f.get('mimeType', 'unknown')}"
                f"  |  Modified: {modified}"
            )

        return ToolResult(success=True, content="\n".join(lines) + self._more_hint(files, next_token))

    # ------------------------------------------------------------------
    # Action: read
    # ------------------------------------------------------------------

    async def _action_read(
        self,
        token: str,
        file_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Read/export a file's content from Google Drive."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for read action.")

        # First, get file metadata to determine type.
        meta = await self._get_file_metadata(token, file_id)
        mime = meta.get("mimeType", "")
        name = meta.get("name", file_id)

        # Sheet / Slides → XLSX / PPTX export, extracted: every tab, slide by
        # slide. Drive caps exports at 10 MB — past that, the flat export.
        if mime in _READ_VIA_OFFICE_EXPORT:
            export_mime = GOOGLE_EXPORT[mime][0]
            try:
                result = await self._download_and_extract(
                    token, file_id, name, mime, export_mime=export_mime,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (400, 403):
                    raise
                result = ToolResult(success=False, error=f"export failed ({exc.response.status_code})")
            if result.success:
                return result
            log.info("Office export read failed, using flat export", file_id=file_id, error=result.error)
            note = (
                "the XLSX export failed (Drive caps exports at 10 MB), so this "
                "is Drive's CSV export: the FIRST TAB ONLY."
                if mime == "application/vnd.google-apps.spreadsheet" else ""
            )
            return await self._export_google_file(token, file_id, name, mime, note=note)

        # Google Workspace file → export.
        if mime in _GOOGLE_EXPORT_MAP:
            return await self._export_google_file(token, file_id, name, mime)

        # Binary file with an extract tool → download + extract.
        if mime in _EXTRACTABLE_MIMES:
            return await self._download_and_extract(token, file_id, name, mime)

        # Folder → list contents instead.
        if mime == "application/vnd.google-apps.folder":
            return await self._action_list(token, folder_id=file_id)

        # Plain text / code / unknown → direct download as text.
        return await self._download_as_text(token, file_id, name, mime)

    async def _export_google_file(
        self, token: str, file_id: str, name: str, mime: str, *, note: str = "",
    ) -> ToolResult:
        """Export a Google Workspace file (Docs, Sheets, Slides) as text."""
        export_mime, ext = _GOOGLE_EXPORT_MAP[mime]

        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}/export",
            params={"mimeType": export_mime, "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )

        # Markdown export might not be supported for all docs — fall back to plain text.
        if resp.status_code == 400 and export_mime == "text/markdown":
            resp = await self._client.get(
                f"{_DRIVE_API}/files/{file_id}/export",
                params={"mimeType": "text/plain", "supportsAllDrives": "true"},
                headers=self._auth_headers(token),
            )
            ext = ".txt"

        resp.raise_for_status()
        content = _strip_base64_images(resp.text)

        if len(content) > _MAX_READ_BYTES:
            content = content[:_MAX_READ_BYTES] + "\n\n... [content truncated]"

        header = f"[Google Drive: {name}]\n[Type: {mime} → exported as {export_mime}]\n[Size: {len(resp.content)} bytes]\n"
        if note:
            header += f"[Note: {note}]\n"
        return ToolResult(success=True, content=header + "\n" + content)

    async def _download_and_extract(
        self, token: str, file_id: str, name: str, mime: str, *, export_mime: str = "",
    ) -> ToolResult:
        """Download a binary file — or export a Google file as *export_mime*
        (XLSX/PPTX) — and extract its text with the matching extract tool."""
        source_mime = export_mime or mime
        tool_name = _EXTRACTABLE_MIMES[source_mime]

        if export_mime:
            url = f"{_DRIVE_API}/files/{file_id}/export"
            params = {"mimeType": export_mime, "supportsAllDrives": "true"}
        else:
            url = f"{_DRIVE_API}/files/{file_id}"
            params = {"alt": "media", "supportsAllDrives": "true"}
        body = await self._fetch_capped(token, url, params)
        if body is None:
            return ToolResult(
                success=False,
                error=f"File '{name}' is too large (over 50 MB). Max 50 MB.",
            )

        # Write to a temp file and run the appropriate extract tool. The
        # suffix comes from the type: the extract tools refuse any other, and
        # a Drive name often has none.
        with tempfile.NamedTemporaryFile(suffix=_SUFFIX_BY_MIME[source_mime], delete=False) as tmp:
            tmp.write(body)
            tmp_path = tmp.name

        try:
            extract_tool = self._get_extract_tool(tool_name)
            if extract_tool is None:
                return ToolResult(
                    success=False,
                    error=f"Extract tool '{tool_name}' not available. Cannot read '{name}'.",
                )

            # Room for what the flat exports returned, and for more of a
            # workbook's rows than the extractor's 200-per-tab default.
            limits: dict[str, Any] = {}
            if export_mime or tool_name == "xlsx_extract":
                limits["max_chars"] = _MAX_READ_BYTES
            if tool_name == "xlsx_extract":
                limits["max_rows"] = _SHEET_READ_MAX_ROWS
            result = await extract_tool.execute(path=tmp_path, **limits)
            if result.success:
                kind = f"{mime} → exported as {export_mime}" if export_mime else mime
                header = f"[Google Drive: {name}]\n[Type: {kind}]\n[Size: {self._format_size(len(body))}]\n"
                if tool_name == "xlsx_extract" and _tab_hit_row_cap(result.content, _SHEET_READ_MAX_ROWS):
                    header += (
                        f"[Note: a tab may run past the {_SHEET_READ_MAX_ROWS} rows shown; "
                        "download the file (.xlsx or .csv) for the full data]\n"
                    )
                content = result.content
                if len(content) > _MAX_READ_BYTES:
                    content = content[:_MAX_READ_BYTES] + "\n\n... [content truncated]"
                return ToolResult(success=True, content=header + "\n" + content)
            return result
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    async def _download_as_text(
        self, token: str, file_id: str, name: str, mime: str,
    ) -> ToolResult:
        """Download a file and return as text."""
        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()

        # Check size.
        if len(resp.content) > _MAX_READ_BYTES:
            try:
                content = resp.content[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
                content += "\n\n... [content truncated]"
            except Exception:
                return ToolResult(
                    success=False,
                    error=f"File '{name}' is too large and not text-readable.",
                )
        else:
            try:
                content = resp.text
            except Exception:
                return ToolResult(
                    success=False,
                    error=f"File '{name}' appears to be a binary file (MIME: {mime}). Cannot display as text.",
                )

        header = f"[Google Drive: {name}]\n[Type: {mime}]\n[Size: {self._format_size(len(resp.content))}]\n\n"
        return ToolResult(success=True, content=header + content)

    # ------------------------------------------------------------------
    # Action: info
    # ------------------------------------------------------------------

    async def _action_info(
        self,
        token: str,
        file_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Get detailed metadata for a file."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for info action.")

        fields = (
            "id,name,mimeType,size,modifiedTime,createdTime,parents,"
            "webViewLink,webContentLink,owners,shared,sharingUser,"
            "description,starred,trashed"
        )
        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"fields": fields, "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        meta = resp.json()

        lines = [f"File: {meta.get('name', 'unknown')}"]
        lines.append(f"  ID: {meta['id']}")
        lines.append(f"  Type: {meta.get('mimeType', 'unknown')}")
        if meta.get("size"):
            lines.append(f"  Size: {self._format_size(int(meta['size']))}")
        lines.append(f"  Created: {meta.get('createdTime', '?')}")
        lines.append(f"  Modified: {meta.get('modifiedTime', '?')}")
        if meta.get("parents"):
            lines.append(f"  Parent folders: {', '.join(meta['parents'])}")
        if meta.get("webViewLink"):
            lines.append(f"  Web link: {meta['webViewLink']}")
        if meta.get("owners"):
            owners = [o.get("displayName", o.get("emailAddress", "?")) for o in meta["owners"]]
            lines.append(f"  Owner(s): {', '.join(owners)}")
        if meta.get("description"):
            lines.append(f"  Description: {meta['description']}")
        lines.append(f"  Shared: {'yes' if meta.get('shared') else 'no'}")
        lines.append(f"  Starred: {'yes' if meta.get('starred') else 'no'}")
        lines.append(f"  Trashed: {'yes' if meta.get('trashed') else 'no'}")

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: download
    # ------------------------------------------------------------------

    async def _action_download(
        self,
        token: str,
        file_id: str = "",
        output_path: str | None = None,
        runtime: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Save a Drive file locally; Google Docs/Sheets/Slides are exported."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for download action.")
        runtime = runtime or {}

        meta = await self._get_file_metadata(token, file_id)
        mime = meta.get("mimeType", "")
        name = meta.get("name") or file_id

        if mime == FOLDER_MIME:
            return ToolResult(
                success=False,
                error=(
                    f"'{name}' is a folder. Use list with folder_id='{file_id}' "
                    "to see its files, then download them one at a time."
                ),
            )
        native = mime.startswith("application/vnd.google-apps.")
        if native and mime not in GOOGLE_EXPORT:
            return ToolResult(
                success=False,
                error=(
                    f"'{name}' ({mime}) has no downloadable content. "
                    "Use info for its metadata and web link."
                ),
            )
        size = int(meta.get("size") or 0)
        if size > _MAX_DOWNLOAD_BYTES:
            return ToolResult(
                success=False,
                error=f"File '{name}' is too large ({self._format_size(size)}). Max 50 MB.",
            )

        # Export format: the output_path extension when this type supports
        # it, else drive_client's default for the type.
        export_mime, ext = "", ""
        if native:
            export_mime, ext = GOOGLE_EXPORT[mime]
            wanted = Path(str(output_path or "").strip()).suffix.lower()
            chosen = _EXPORT_BY_SUFFIX.get(mime, {}).get(wanted)
            if chosen:
                export_mime, ext = chosen, wanted

        default_name = _safe_filename(name, file_id)
        # A binary's Drive name may lack its extension, which picks the reader.
        suffix = ext or _missing_suffix(default_name, mime)
        if suffix and not default_name.lower().endswith(suffix):
            default_name += suffix
        try:
            dest = self._download_dest(output_path, default_name, ext, runtime)
        except _DownloadPathError as exc:
            return ToolResult(success=False, error=str(exc))

        if native:
            url = f"{_DRIVE_API}/files/{file_id}/export"
            params = {"mimeType": export_mime, "supportsAllDrives": "true"}
        else:
            url = f"{_DRIVE_API}/files/{file_id}"
            params = {"alt": "media", "supportsAllDrives": "true"}
        try:
            body = await self._fetch_capped(token, url, params)
        except httpx.HTTPStatusError as exc:
            # Markdown export isn't available for every Doc — plain text is.
            if not (exc.response.status_code == 400 and export_mime == "text/markdown"):
                raise
            export_mime = "text/plain"
            body = await self._fetch_capped(token, url, {**params, "mimeType": export_mime})
        if body is None:
            return ToolResult(
                success=False,
                error=f"File '{name}' is too large (over 50 MB). Max 50 MB.",
            )
        if export_mime in _TEXT_EXPORTS:
            text = body.decode("utf-8", errors="replace")
            stripped = _strip_base64_images(text)
            if len(stripped) < len(text):
                body = stripped.encode("utf-8")

        # PR C: a member never downloads over a saved/ file someone else
        # created; the creator is recorded after the write.
        from captain_claw import saved_attribution
        from captain_claw import speaker as _speaker

        prior = saved_attribution.prior_creator(dest)
        member = _speaker.current()
        if (member is not None and dest.exists()
                and not saved_attribution.member_may_change(dest, member.speaker_id)):
            return ToolResult(
                success=False,
                error=_speaker.PATH_REFUSED_PREFIX + _speaker.FILE_NOT_YOURS_WHY,
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(dest.write_bytes, body)
        saved_attribution.note_write(dest, prior)

        registry = runtime.get("_file_registry")
        if registry is not None:
            try:
                registry.register(
                    logical_path=str(output_path or "").strip() or dest.name,
                    physical_path=str(dest),
                    task_id=str(runtime.get("_task_id") or ""),
                )
            except Exception:
                pass  # non-critical

        kind = f"{mime} → exported as {export_mime}" if native else (mime or "unknown")
        reader = _READER_BY_SUFFIX.get(dest.suffix.lower(), "read")
        return ToolResult(
            success=True,
            content=(
                f"Downloaded '{name}' from Google Drive.\n"
                f"  Path: {dest}\n"
                f"  Type: {kind}\n"
                f"  Size: {self._format_size(len(body))}\n"
                f"Use {reader}(path=\"{dest}\") to view it."
            ),
        )

    @staticmethod
    def _download_dest(
        output_path: str | None, default_name: str, export_ext: str, runtime: dict[str, Any],
    ) -> Path:
        """Where a download lands: always inside the saved/ area.

        Default ``saved/downloads/<session>/<name>``. A relative output_path is
        scoped like the write tool's (category + session); an absolute one must
        already sit under saved/. ``..`` and anything resolving outside
        (symlinks included) raise :class:`_DownloadPathError`.
        """
        from captain_claw.tools.write import WriteTool

        saved_root = WriteTool._resolve_saved_root(runtime)
        session_id = WriteTool._normalize_session_id(str(runtime.get("_session_id") or ""))
        requested = str(output_path or "").strip()

        if not requested:
            dest = saved_root / "downloads" / session_id / default_name
        else:
            req = Path(requested).expanduser()
            if ".." in req.parts:
                raise _DownloadPathError(
                    f"output_path may not contain '..' (got {requested!r}). "
                    f"Omit it to save under saved/downloads/{session_id}/."
                )
            names_dir = requested.endswith(("/", "\\"))
            if req.is_absolute():
                dest = req
            else:
                # Scoped like the write tool: <category>/<session>/...,
                # anything outside a category under downloads/.
                parts = [p for p in req.parts if p not in ("", ".")]
                if parts and parts[0].lower() == "saved":
                    parts = parts[1:]
                if not parts or parts[0] not in _SAVED_CATEGORIES:
                    parts = ["downloads", *parts]
                if len(parts) < 2 or parts[1] != session_id:
                    parts = [parts[0], session_id, *parts[1:]]
                names_dir = names_dir or len(parts) == 2  # no file name given
                dest = saved_root.joinpath(*parts)
            if names_dir or dest.is_dir():
                dest = dest / default_name
            elif export_ext and dest.suffix.lower() != export_ext:
                dest = dest.with_name(dest.name + export_ext)

        resolved = dest.resolve()
        try:
            resolved.relative_to(saved_root)
        except ValueError:
            raise _DownloadPathError(
                f"output_path must be inside the saved area ({saved_root}); got "
                f"{requested!r}. Omit it to save under saved/downloads/{session_id}/."
            ) from None
        return resolved

    async def _fetch_capped(
        self, token: str, url: str, params: dict[str, str],
    ) -> bytes | None:
        """GET *url* as a stream; None once it passes the download cap.

        Exports carry no size in their metadata, so the cap is enforced on
        the bytes as they arrive rather than after buffering them all.
        """
        async with self._client.stream(
            "GET", url, params=params, headers=self._auth_headers(token),
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()  # so the error handler can read the API message
                resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    return None
                chunks.append(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Action: upload
    # ------------------------------------------------------------------

    async def _action_upload(
        self,
        token: str,
        local_path: str = "",
        name: str | None = None,
        folder_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Upload a local file to Google Drive."""
        if not local_path:
            return ToolResult(success=False, error="local_path is required for upload action.")

        file_path = Path(local_path).expanduser().resolve()
        if not file_path.exists():
            return ToolResult(success=False, error=f"Local file not found: {local_path}")
        if not file_path.is_file():
            return ToolResult(success=False, error=f"Not a file: {local_path}")

        file_name = name or file_path.name
        file_bytes = file_path.read_bytes()

        # Guess MIME type.
        import mimetypes as mt
        content_type = mt.guess_type(str(file_path))[0] or "application/octet-stream"

        metadata: dict[str, Any] = {"name": file_name}
        if folder_id:
            metadata["parents"] = [folder_id]

        result = await self._multipart_upload(token, metadata, file_bytes, content_type)
        uploaded_id = result.get("id", "?")
        uploaded_name = result.get("name", file_name)
        link = result.get("webViewLink", "")

        msg = f"Uploaded '{uploaded_name}' to Google Drive.\n  ID: {uploaded_id}"
        if link:
            msg += f"\n  Link: {link}"
        return ToolResult(success=True, content=msg)

    # ------------------------------------------------------------------
    # Action: create
    # ------------------------------------------------------------------

    async def _action_create(
        self,
        token: str,
        name: str = "",
        content: str = "",
        mime_type: str | None = None,
        folder_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Create a new file on Google Drive with the given content."""
        if not name:
            return ToolResult(success=False, error="name is required for create action.")

        target_mime = mime_type or "application/vnd.google-apps.document"

        # For Google Docs/Sheets, create with conversion.
        is_google_type = target_mime.startswith("application/vnd.google-apps.")

        if is_google_type:
            # Create by uploading plain text and converting.
            upload_mime = "text/plain"
            if "spreadsheet" in target_mime:
                upload_mime = "text/csv"

            metadata: dict[str, Any] = {"name": name, "mimeType": target_mime}
            if folder_id:
                metadata["parents"] = [folder_id]

            content_bytes = (content or "").encode("utf-8")
            result = await self._multipart_upload(
                token, metadata, content_bytes, upload_mime,
                convert=True,
            )
        else:
            # Plain file — upload as-is.
            metadata = {"name": name}
            if folder_id:
                metadata["parents"] = [folder_id]
            content_bytes = (content or "").encode("utf-8")
            result = await self._multipart_upload(
                token, metadata, content_bytes, target_mime,
            )

        created_id = result.get("id", "?")
        created_name = result.get("name", name)
        link = result.get("webViewLink", "")

        msg = f"Created '{created_name}' on Google Drive.\n  ID: {created_id}\n  Type: {target_mime}"
        if link:
            msg += f"\n  Link: {link}"
        return ToolResult(success=True, content=msg)

    # ------------------------------------------------------------------
    # Action: update
    # ------------------------------------------------------------------

    async def _action_update(
        self,
        token: str,
        file_id: str = "",
        content: str | None = None,
        local_path: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Update an existing file's content."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for update action.")

        if content is not None:
            body = content.encode("utf-8")
            upload_mime = "text/plain"
        elif local_path:
            path = Path(local_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                return ToolResult(success=False, error=f"Local file not found: {local_path}")
            body = path.read_bytes()
            import mimetypes as mt
            upload_mime = mt.guess_type(str(path))[0] or "application/octet-stream"
        else:
            return ToolResult(
                success=False,
                error="Either 'content' or 'local_path' is required for update action.",
            )

        resp = await self._client.patch(
            f"{_UPLOAD_API}/files/{file_id}",
            params={"uploadType": "media", "supportsAllDrives": "true"},
            headers={
                **self._auth_headers(token),
                "Content-Type": upload_mime,
            },
            content=body,
        )
        resp.raise_for_status()
        result = resp.json()

        updated_name = result.get("name", file_id)
        return ToolResult(
            success=True,
            content=f"Updated '{updated_name}' on Google Drive.\n  ID: {file_id}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_file_metadata(self, token: str, file_id: str) -> dict[str, Any]:
        """Fetch file metadata from the Drive API."""
        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"fields": _FILE_FIELDS, "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        return resp.json()

    async def _multipart_upload(
        self,
        token: str,
        metadata: dict[str, Any],
        content: bytes,
        content_type: str,
        convert: bool = False,
    ) -> dict[str, Any]:
        """Upload a file using multipart upload."""
        boundary = f"captain_claw_{uuid.uuid4().hex[:16]}"

        # Build multipart body per Google Drive API spec.
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(b"Content-Type: application/json; charset=UTF-8\r\n\r\n")
        body.write(json.dumps(metadata).encode("utf-8"))
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.write(content)
        body.write(f"\r\n--{boundary}--\r\n".encode())

        params: dict[str, str] = {
            "uploadType": "multipart",
            "supportsAllDrives": "true",
            "fields": "id,name,mimeType,webViewLink",
        }

        resp = await self._client.post(
            f"{_UPLOAD_API}/files",
            params=params,
            headers={
                **self._auth_headers(token),
                "Content-Type": f"multipart/related; boundary={boundary}",
            },
            content=body.getvalue(),
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _get_extract_tool(tool_name: str) -> Any | None:
        """Get an instance of an extract tool by name."""
        try:
            if tool_name == "pdf_extract":
                from captain_claw.tools.document_extract import PdfExtractTool
                return PdfExtractTool()
            elif tool_name == "docx_extract":
                from captain_claw.tools.document_extract import DocxExtractTool
                return DocxExtractTool()
            elif tool_name == "xlsx_extract":
                from captain_claw.tools.document_extract import XlsxExtractTool
                return XlsxExtractTool()
            elif tool_name == "pptx_extract":
                from captain_claw.tools.document_extract import PptxExtractTool
                return PptxExtractTool()
        except Exception:
            return None
        return None

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size into human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
