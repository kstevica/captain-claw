"""Google Drive tool for listing, searching, reading, and writing files.

Uses the Google Drive REST API v3 via httpx with OAuth2 Bearer tokens
managed by :class:`~captain_claw.google_oauth_manager.GoogleOAuthManager`.
No additional Google SDK dependencies are required.
"""

from __future__ import annotations

import io
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRIVE_API = "https://www.googleapis.com/drive/v3"
_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"

# Fields to request from the files endpoint.
_FILE_FIELDS = "id,name,mimeType,size,modifiedTime,createdTime,parents,webViewLink,owners"
_LIST_FIELDS = f"nextPageToken,files({_FILE_FIELDS})"

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


class GoogleDriveTool(Tool):
    """Interact with Google Drive: list, search, read, upload, create, and update files."""

    name = "google_drive"
    description = (
        "Interact with Google Drive. Actions: list (browse folders), "
        "search (find files by name or content), read (get file contents), "
        "info (get file metadata), upload (send a local file to Drive), "
        "create (create a new file on Drive), update (update existing file content)."
    )
    timeout_seconds = 120.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "search", "read", "info", "upload", "create", "update"],
                "description": "The action to perform.",
            },
            "file_id": {
                "type": "string",
                "description": "Google Drive file ID (for read, info, update actions).",
            },
            "folder_id": {
                "type": "string",
                "description": "Folder ID to list or upload into. Defaults to 'root'.",
            },
            "query": {
                "type": "string",
                "description": "Search query text (for search action).",
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results to return (default 20, max 100).",
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
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Google Drive Tool)"},
        )

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        # Pop injected kwargs that tools receive from the registry.
        kwargs.pop("_runtime_base_path", None)
        kwargs.pop("_saved_base_path", None)
        kwargs.pop("_session_id", None)
        kwargs.pop("_abort_event", None)
        kwargs.pop("_file_registry", None)
        kwargs.pop("_task_id", None)

        try:
            token = await self._get_access_token()
        except RuntimeError as e:
            return ToolResult(success=False, error=str(e))

        handlers = {
            "list": self._action_list,
            "search": self._action_search,
            "read": self._action_read,
            "info": self._action_info,
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

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """Retrieve a valid Google OAuth access token.

        Raises RuntimeError if Google is not connected or tokens are
        expired and cannot be refreshed.
        """
        from captain_claw.google_oauth_manager import GoogleOAuthManager
        from captain_claw.session import get_session_manager

        mgr = GoogleOAuthManager(get_session_manager())
        tokens = await mgr.get_tokens()
        if not tokens:
            raise RuntimeError(
                "Google account is not connected. "
                "Please connect via the web UI (Settings > Google OAuth) or "
                "navigate to /auth/google/login in your browser."
            )

        # Check if the Drive scope is present.
        granted = set(tokens.scope.split()) if tokens.scope else set()
        if _DRIVE_SCOPE not in granted:
            raise RuntimeError(
                "Google Drive scope not granted. Your current OAuth connection "
                "does not include Drive access. Please disconnect and reconnect "
                "your Google account to grant Drive permissions."
            )

        return tokens.access_token

    def _auth_headers(self, token: str) -> dict[str, str]:
        """Build authorization headers."""
        return {"Authorization": f"Bearer {token}"}

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
        **kwargs: Any,
    ) -> ToolResult:
        """List files in a Google Drive folder."""
        limit = min(int(max_results or 20), 100)
        order = order_by or "modifiedTime desc"

        q = f"'{folder_id}' in parents and trashed = false"
        params = {
            "q": q,
            "fields": _LIST_FIELDS,
            "pageSize": limit,
            "orderBy": order,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        resp = await self._client.get(
            f"{_DRIVE_API}/files",
            params=params,
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        files = data.get("files", [])
        if not files:
            folder_label = f"folder '{folder_id}'" if folder_id != "root" else "root folder"
            return ToolResult(success=True, content=f"No files found in {folder_label}.")

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

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: search
    # ------------------------------------------------------------------

    async def _action_search(
        self,
        token: str,
        query: str = "",
        max_results: int | float | None = None,
        order_by: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Search for files across Google Drive."""
        if not query:
            return ToolResult(success=False, error="Search query is required.")

        limit = min(int(max_results or 20), 100)
        order = order_by or "relevance"

        # Build search query — search by name and full text.
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        q = f"(name contains '{escaped}' or fullText contains '{escaped}') and trashed = false"

        params = {
            "q": q,
            "fields": _LIST_FIELDS,
            "pageSize": limit,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        # 'relevance' is only valid without orderBy (it's the default).
        if order != "relevance":
            params["orderBy"] = order

        resp = await self._client.get(
            f"{_DRIVE_API}/files",
            params=params,
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        files = data.get("files", [])
        if not files:
            return ToolResult(success=True, content=f"No files found matching '{query}'.")

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

        return ToolResult(success=True, content="\n".join(lines))

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
        self, token: str, file_id: str, name: str, mime: str,
    ) -> ToolResult:
        """Export a Google Workspace file (Docs, Sheets, Slides)."""
        export_mime, ext = _GOOGLE_EXPORT_MAP[mime]

        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}/export",
            params={"mimeType": export_mime},
            headers=self._auth_headers(token),
        )

        # Markdown export might not be supported for all docs — fall back to plain text.
        if resp.status_code == 400 and export_mime == "text/markdown":
            resp = await self._client.get(
                f"{_DRIVE_API}/files/{file_id}/export",
                params={"mimeType": "text/plain"},
                headers=self._auth_headers(token),
            )
            ext = ".txt"

        resp.raise_for_status()
        content = resp.text

        if len(content) > _MAX_READ_BYTES:
            content = content[:_MAX_READ_BYTES] + "\n\n... [content truncated]"

        header = f"[Google Drive: {name}]\n[Type: {mime} → exported as {export_mime}]\n[Size: {len(resp.content)} bytes]\n\n"
        return ToolResult(success=True, content=header + content)

    async def _download_and_extract(
        self, token: str, file_id: str, name: str, mime: str,
    ) -> ToolResult:
        """Download a binary file and extract text using existing tools."""
        tool_name = _EXTRACTABLE_MIMES[mime]

        # Download to a temporary file.
        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"alt": "media"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()

        # Check size before processing.
        if len(resp.content) > 50_000_000:  # 50 MB
            return ToolResult(
                success=False,
                error=f"File '{name}' is too large ({self._format_size(len(resp.content))}). Max 50 MB.",
            )

        # Write to a temp file and run the appropriate extract tool.
        suffix = Path(name).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        try:
            extract_tool = self._get_extract_tool(tool_name)
            if extract_tool is None:
                return ToolResult(
                    success=False,
                    error=f"Extract tool '{tool_name}' not available. Cannot read '{name}'.",
                )

            result = await extract_tool.execute(path=tmp_path)
            if result.success:
                header = f"[Google Drive: {name}]\n[Type: {mime}]\n[Size: {self._format_size(len(resp.content))}]\n\n"
                content = result.content
                if len(content) > _MAX_READ_BYTES:
                    content = content[:_MAX_READ_BYTES] + "\n\n... [content truncated]"
                return ToolResult(success=True, content=header + content)
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
            params={"alt": "media"},
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
