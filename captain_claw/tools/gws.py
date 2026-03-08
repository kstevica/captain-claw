"""Google Workspace CLI (gws) tool for Drive, Docs, Gmail, and Calendar.

Wraps the ``gws`` CLI (https://github.com/googleworkspace/cli) which must
be installed and authenticated separately (``gws auth setup && gws auth login``).
The tool shells out to the ``gws`` binary with ``--format json`` and returns
the JSON (or formatted text) output to the agent.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# Maximum output length returned to the agent.
_MAX_OUTPUT_CHARS = 60_000

# Default timeout for gws commands (seconds).
_DEFAULT_TIMEOUT = 120

# Pattern matching inline base64-encoded images (can be hundreds of KB in exported markdown).
_BASE64_IMG_RE = re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+')


def _strip_base64_images(text: str) -> str:
    """Remove inline base64 image data from text to prevent context bloat."""
    cleaned = _BASE64_IMG_RE.sub('[image]', text)
    if len(cleaned) < len(text):
        log.debug(
            "stripped base64 images",
            original_len=len(text),
            cleaned_len=len(cleaned),
        )
    return cleaned


class GwsTool(Tool):
    """Google Workspace CLI — access Drive, Docs, Gmail, and Calendar via the ``gws`` binary."""

    name = "gws"
    description = (
        "Google Workspace CLI tool. "
        "Actions: drive_list (list files in Drive), drive_search (find files by name/content), "
        "drive_download (download/export a file locally), drive_info (file metadata), "
        "drive_create (create a new Google Doc/Sheet/file on Drive), "
        "docs_read (read a Google Doc — returns the full document text INLINE in the "
        "tool result; do NOT attempt to read a file after docs_read, the content is "
        "already in the response), docs_append (append text to a Doc), "
        "mail_list (list recent emails), mail_search (search emails), "
        "mail_read (read a specific email), "
        "calendar_list (list upcoming events), calendar_search (search events), "
        "calendar_create (create a calendar event), calendar_agenda (show agenda). "
        "Requires the gws CLI to be installed and authenticated (gws auth login)."
    )
    timeout_seconds = 180.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "drive_list",
                    "drive_search",
                    "drive_download",
                    "drive_info",
                    "drive_create",
                    "docs_read",
                    "docs_append",
                    "mail_list",
                    "mail_search",
                    "mail_read",
                    "calendar_list",
                    "calendar_search",
                    "calendar_create",
                    "calendar_agenda",
                    "raw",
                ],
                "description": "The action to perform.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Search query text. Used by drive_search (file name/content), "
                    "mail_search (Gmail search syntax like 'from:alice subject:report'), "
                    "and calendar_search (event text search)."
                ),
            },
            "file_id": {
                "type": "string",
                "description": "Google Drive file ID (for drive_download, drive_info, docs_read, docs_append).",
            },
            "folder_id": {
                "type": "string",
                "description": "Drive folder ID for drive_list or drive_create. Defaults to root.",
            },
            "name": {
                "type": "string",
                "description": "File/document name (for drive_create).",
            },
            "content": {
                "type": "string",
                "description": "Text content (for drive_create initial content, or docs_append text).",
            },
            "mime_type": {
                "type": "string",
                "description": (
                    "MIME type for drive_create. Use 'application/vnd.google-apps.document' "
                    "for Google Doc, 'application/vnd.google-apps.spreadsheet' for Sheet, "
                    "or omit for auto-detection."
                ),
            },
            "output_path": {
                "type": "string",
                "description": "Local path to save downloaded file (for drive_download). Optional.",
            },
            "message_id": {
                "type": "string",
                "description": "Gmail message ID (for mail_read).",
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results to return (default varies by action).",
            },
            "label": {
                "type": "string",
                "description": "Gmail label for mail_list (e.g. INBOX, SENT, DRAFT). Default: INBOX.",
            },
            "summary": {
                "type": "string",
                "description": "Event title/summary (for calendar_create).",
            },
            "start": {
                "type": "string",
                "description": (
                    "Event start time in ISO 8601 format, e.g. '2026-03-10T14:00:00'. "
                    "For all-day events use date only: '2026-03-10'. (for calendar_create)."
                ),
            },
            "end": {
                "type": "string",
                "description": "Event end time in ISO 8601 format (for calendar_create). Optional.",
            },
            "attendees": {
                "type": "string",
                "description": "Comma-separated attendee email addresses (for calendar_create).",
            },
            "calendar_id": {
                "type": "string",
                "description": "Calendar ID (default: primary).",
            },
            "days": {
                "type": "number",
                "description": "Number of days to look ahead for calendar_list/calendar_agenda (default: 7).",
            },
            "raw_args": {
                "type": "string",
                "description": (
                    "Raw arguments passed directly to gws CLI (for 'raw' action). "
                    "Example: 'drive files list --params {\"pageSize\": 5}'"
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._binary: str | None = None

    # ------------------------------------------------------------------
    # Execution dispatch
    # ------------------------------------------------------------------

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        # Pop injected kwargs from the registry.
        runtime_base = kwargs.pop("_runtime_base_path", None)
        saved_base = kwargs.pop("_saved_base_path", None)
        kwargs.pop("_session_id", None)
        kwargs.pop("_abort_event", None)
        kwargs.pop("_file_registry", None)
        kwargs.pop("_task_id", None)

        # Resolve the gws binary path.
        binary = self._resolve_binary()
        if binary is None:
            return ToolResult(
                success=False,
                error=(
                    "The 'gws' CLI binary was not found. "
                    "Please install it: npm install -g @googleworkspace/cli  "
                    "Then authenticate: gws auth setup && gws auth login"
                ),
            )

        handlers: dict[str, Any] = {
            "drive_list": self._drive_list,
            "drive_search": self._drive_search,
            "drive_download": self._drive_download,
            "drive_info": self._drive_info,
            "drive_create": self._drive_create,
            "docs_read": self._docs_read,
            "docs_append": self._docs_append,
            "mail_list": self._mail_list,
            "mail_search": self._mail_search,
            "mail_read": self._mail_read,
            "calendar_list": self._calendar_list,
            "calendar_search": self._calendar_search,
            "calendar_create": self._calendar_create,
            "calendar_agenda": self._calendar_agenda,
            "raw": self._raw,
        }

        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use one of: {', '.join(handlers)}",
            )

        try:
            return await handler(binary, saved_base=saved_base, runtime_base=runtime_base, **kwargs)
        except asyncio.TimeoutError:
            return ToolResult(success=False, error="gws command timed out.")
        except Exception as exc:
            log.error("gws tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------

    def _resolve_binary(self) -> str | None:
        """Find the gws binary."""
        if self._binary and shutil.which(self._binary):
            return self._binary

        # Check config for a custom path.
        try:
            cfg = get_config()
            custom = getattr(cfg.tools, "gws", None)
            if custom and hasattr(custom, "binary_path") and custom.binary_path:
                p = Path(custom.binary_path).expanduser()
                if p.exists():
                    self._binary = str(p)
                    return self._binary
        except Exception:
            pass

        # Try PATH.
        found = shutil.which("gws")
        if found:
            self._binary = found
            return self._binary

        return None

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    async def _run_gws(
        self,
        binary: str,
        args: list[str],
        timeout: float = _DEFAULT_TIMEOUT,
        json_output: bool = True,
    ) -> ToolResult:
        """Run a gws command and return the result."""
        cmd = [binary] + args
        if json_output and "--format" not in args:
            cmd.extend(["--format", "json"])

        log.debug("Running gws command", cmd=" ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(success=False, error="gws command timed out.")

        stdout_str = stdout.decode("utf-8", errors="replace").strip()
        stderr_str = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            error_msg = stderr_str or stdout_str or f"gws exited with code {proc.returncode}"
            # Check for common auth errors.
            if "no credentials" in error_msg.lower() or "token" in error_msg.lower():
                error_msg += "\n\nHint: Run 'gws auth login' to authenticate."
            return ToolResult(success=False, error=error_msg)

        output = stdout_str
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n\n... [output truncated]"

        return ToolResult(success=True, content=output)

    # ------------------------------------------------------------------
    # Drive actions
    # ------------------------------------------------------------------

    async def _drive_list(
        self, binary: str, folder_id: str = "", max_results: int | float | None = None, **kw: Any,
    ) -> ToolResult:
        """List files in a Google Drive folder."""
        page_size = min(int(max_results or 20), 100)
        params: dict[str, Any] = {"pageSize": page_size}

        if folder_id:
            escaped = folder_id.replace("'", "\\'")
            params["q"] = f"'{escaped}' in parents and trashed = false"
        else:
            params["q"] = "'root' in parents and trashed = false"

        params["orderBy"] = "modifiedTime desc"
        params["fields"] = "files(id,name,mimeType,size,modifiedTime)"
        # Support shared drives.
        params["corpora"] = "allDrives"
        params["supportsAllDrives"] = "true"
        params["includeItemsFromAllDrives"] = "true"

        args = ["drive", "files", "list", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _drive_search(
        self, binary: str, query: str = "", max_results: int | float | None = None, **kw: Any,
    ) -> ToolResult:
        """Search for files across Google Drive."""
        if not query:
            return ToolResult(success=False, error="query is required for drive_search.")

        page_size = min(int(max_results or 20), 100)
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        q = f"(name contains '{escaped}' or fullText contains '{escaped}') and trashed = false"

        params: dict[str, Any] = {
            "pageSize": page_size,
            "q": q,
            "fields": "files(id,name,mimeType,size,modifiedTime,webViewLink)",
            # Support shared drives.
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        args = ["drive", "files", "list", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _drive_download(
        self,
        binary: str,
        file_id: str = "",
        output_path: str = "",
        saved_base: Any = None,
        runtime_base: Any = None,
        **kw: Any,
    ) -> ToolResult:
        """Download/export a file from Google Drive."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for drive_download.")

        # First get file metadata to determine type and name.
        meta_args = [
            "drive", "files", "get",
            "--params", json.dumps({"fileId": file_id, "fields": "id,name,mimeType", "supportsAllDrives": "true"}),
        ]
        meta_result = await self._run_gws(binary, meta_args)
        if not meta_result.success:
            return meta_result

        file_name = file_id
        mime_type = ""
        try:
            meta = json.loads(meta_result.content)
            file_name = meta.get("name", file_id)
            mime_type = meta.get("mimeType", "")
        except (json.JSONDecodeError, TypeError):
            pass

        # For Google Workspace native types, use export.
        # Sheets → XLSX and Presentations → PPTX to preserve all
        # sheets/slides (CSV/TXT only capture the first one).
        google_export_map = {
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

        if output_path:
            _out = Path(output_path).expanduser()
            if _out.is_absolute():
                dest = _out.resolve()
            elif runtime_base:
                # Resolve relative output paths against workspace root
                # so they are consistent with the read tool's path resolution.
                # (e.g. "saved/tmp/{session}/file.md" → workspace/saved/tmp/…)
                dest = (Path(str(runtime_base)) / _out).resolve()
            else:
                dest = _out.resolve()
        elif saved_base:
            dest = Path(str(saved_base)) / file_name
        else:
            dest = Path.cwd() / file_name

        # Ensure parent directory exists.
        dest.parent.mkdir(parents=True, exist_ok=True)

        if mime_type in google_export_map:
            export_mime, ext = google_export_map[mime_type]
            if not dest.suffix:
                dest = dest.with_suffix(ext)

            args = [
                "drive", "files", "export",
                "--params", json.dumps({
                    "fileId": file_id,
                    "mimeType": export_mime,
                    "supportsAllDrives": "true",
                }),
                "--output", str(dest),
            ]
        else:
            args = [
                "drive", "files", "get",
                "--params", json.dumps({"fileId": file_id, "alt": "media", "supportsAllDrives": "true"}),
                "--output", str(dest),
            ]

        result = await self._run_gws(binary, args, json_output=False)
        if result.success:
            # Strip base64 images from exported markdown/text files to prevent bloat.
            if dest.exists() and dest.suffix in (".md", ".txt") and mime_type in google_export_map:
                try:
                    raw = dest.read_text(encoding="utf-8", errors="replace")
                    cleaned = _strip_base64_images(raw)
                    if len(cleaned) < len(raw):
                        dest.write_text(cleaned, encoding="utf-8")
                except OSError:
                    pass
            # Clean up gws side-effect download.bin file.
            _stale = Path.cwd() / "download.bin"
            if _stale.exists():
                try:
                    _stale.unlink()
                except OSError:
                    pass
            abs_dest = str(dest.resolve())
            return ToolResult(
                success=True,
                content=(
                    f"Downloaded '{file_name}' to {abs_dest}\n"
                    f"Use read(path=\"{abs_dest}\") to view the file contents."
                ),
            )
        return result

    async def _drive_info(
        self, binary: str, file_id: str = "", **kw: Any,
    ) -> ToolResult:
        """Get metadata for a Drive file."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for drive_info.")

        params = {
            "fileId": file_id,
            "fields": "id,name,mimeType,size,modifiedTime,createdTime,parents,webViewLink,owners,shared,description",
            "supportsAllDrives": "true",
        }
        args = ["drive", "files", "get", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _drive_create(
        self,
        binary: str,
        name: str = "",
        content: str = "",
        mime_type: str = "",
        folder_id: str = "",
        **kw: Any,
    ) -> ToolResult:
        """Create a new file on Google Drive."""
        if not name:
            return ToolResult(success=False, error="name is required for drive_create.")

        body: dict[str, Any] = {"name": name}
        if mime_type:
            body["mimeType"] = mime_type
        if folder_id:
            body["parents"] = [folder_id]

        if content:
            # Use the +upload helper for content.
            args = ["drive", "+upload"]
            # Write content to a temp file.
            import tempfile
            suffix = ".txt"
            if mime_type and "spreadsheet" in mime_type:
                suffix = ".csv"
            elif mime_type and "document" in mime_type:
                suffix = ".md"

            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                args.extend(["--file", tmp_path, "--name", name])
                if mime_type:
                    args.extend(["--mime-type", mime_type])
                if folder_id:
                    args.extend(["--parent", folder_id])
                return await self._run_gws(binary, args)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            # Create empty file via API.
            args = ["drive", "files", "create", "--json", json.dumps(body)]
            return await self._run_gws(binary, args)

    # ------------------------------------------------------------------
    # Docs actions
    # ------------------------------------------------------------------

    # Google Workspace types that need binary export (XLSX / PPTX) followed
    # by local extraction rather than plain-text export.
    _BINARY_EXPORT_MAP: dict[str, tuple[str, str]] = {
        "application/vnd.google-apps.spreadsheet": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xlsx",
        ),
        "application/vnd.google-apps.presentation": (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".pptx",
        ),
    }

    async def _docs_read(
        self, binary: str, file_id: str = "", saved_base: Any = None, **kw: Any,
    ) -> ToolResult:
        """Read a Google Doc/Sheet/Slides content."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for docs_read.")

        # gws export writes a `download.bin` side-effect file in cwd; remove stale copies first.
        _stale = Path.cwd() / "download.bin"
        if _stale.exists():
            try:
                _stale.unlink()
            except OSError:
                pass

        # First get file metadata for name and MIME type.
        meta_params = {
            "fileId": file_id,
            "fields": "name,mimeType",
            "supportsAllDrives": "true",
        }
        meta_result = await self._run_gws(
            binary,
            ["drive", "files", "get", "--params", json.dumps(meta_params)],
        )
        doc_name = file_id
        file_mime = ""
        if meta_result.success:
            try:
                _meta = json.loads(meta_result.content)
                doc_name = _meta.get("name", file_id)
                file_mime = _meta.get("mimeType", "")
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Spreadsheets / Presentations: export as XLSX/PPTX, extract locally ──
        if file_mime in self._BINARY_EXPORT_MAP:
            return await self._docs_read_binary_export(
                binary, file_id, doc_name, file_mime,
            )

        # ── Docs / default: export as markdown, fallback to plain text ──
        params = {
            "fileId": file_id,
            "mimeType": "text/markdown",
            "supportsAllDrives": "true",
        }
        args = ["drive", "files", "export", "--params", json.dumps(params)]
        result = await self._run_gws(binary, args, json_output=False)

        if not result.success:
            # Fall back to plain text (catches any export error, not just "markdown").
            params["mimeType"] = "text/plain"
            args = ["drive", "files", "export", "--params", json.dumps(params)]
            result = await self._run_gws(binary, args, json_output=False)

        # Clean up the `download.bin` side-effect file that gws export creates.
        if _stale.exists():
            try:
                _stale.unlink()
            except OSError:
                pass

        # Strip inline base64 image data that can bloat the content (hundreds of KB).
        if result.success and result.content:
            cleaned = _strip_base64_images(result.content)
            result = ToolResult(success=True, content=cleaned)

        return result

    async def _docs_read_binary_export(
        self,
        binary: str,
        file_id: str,
        doc_name: str,
        file_mime: str,
    ) -> ToolResult:
        """Export a Google Sheet/Presentation to XLSX/PPTX, extract content,
        and return the text inline (same contract as ``docs_read``)."""
        import tempfile

        export_mime, suffix = self._BINARY_EXPORT_MAP[file_mime]

        # Export to a temp file.
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            params = {
                "fileId": file_id,
                "mimeType": export_mime,
                "supportsAllDrives": "true",
            }
            args = [
                "drive", "files", "export",
                "--params", json.dumps(params),
                "--output", str(tmp_path),
            ]
            result = await self._run_gws(binary, args, json_output=False)
            if not result.success:
                return result

            # Verify the exported file has content.
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                return ToolResult(
                    success=False,
                    error=f"gws export produced empty file for {doc_name!r}",
                )

            # Extract content using the appropriate extractor.
            from captain_claw.tools.document_extract import (
                _extract_xlsx_markdown,
                _extract_pptx_markdown,
            )

            if suffix == ".xlsx":
                content = await asyncio.to_thread(
                    _extract_xlsx_markdown, tmp_path, 200,
                )
            else:
                content = await asyncio.to_thread(
                    _extract_pptx_markdown, tmp_path, 200,
                )

            return ToolResult(success=True, content=content)

        except Exception as e:
            log.error("docs_read binary export failed", file_id=file_id, error=str(e))
            return ToolResult(success=False, error=str(e))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Clean up gws side-effect download.bin file.
            _stale = Path.cwd() / "download.bin"
            if _stale.exists():
                try:
                    _stale.unlink()
                except OSError:
                    pass

    async def _docs_append(
        self, binary: str, file_id: str = "", content: str = "", **kw: Any,
    ) -> ToolResult:
        """Append text to a Google Doc."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for docs_append.")
        if not content:
            return ToolResult(success=False, error="content is required for docs_append.")

        args = ["docs", "+write", "--id", file_id, "--text", content]
        return await self._run_gws(binary, args)

    # ------------------------------------------------------------------
    # Gmail actions
    # ------------------------------------------------------------------

    async def _mail_list(
        self,
        binary: str,
        label: str = "INBOX",
        max_results: int | float | None = None,
        **kw: Any,
    ) -> ToolResult:
        """List recent emails."""
        limit = min(int(max_results or 10), 50)
        params: dict[str, Any] = {
            "userId": "me",
            "maxResults": limit,
            "labelIds": label.upper(),
        }
        args = ["gmail", "users", "messages", "list", "--params", json.dumps(params)]
        result = await self._run_gws(binary, args)

        if not result.success:
            return result

        # Try to fetch details for each message.
        try:
            data = json.loads(result.content)
            messages = data.get("messages", [])
            if not messages:
                return ToolResult(success=True, content=f"No messages found in {label}.")

            details = []
            for msg in messages[:limit]:
                msg_id = msg.get("id", "")
                if not msg_id:
                    continue
                detail_result = await self._run_gws(binary, [
                    "gmail", "users", "messages", "get",
                    "--params", json.dumps({
                        "userId": "me",
                        "id": msg_id,
                        "format": "metadata",
                        "metadataHeaders": "From,To,Subject,Date",
                    }),
                ])
                if detail_result.success:
                    details.append(detail_result.content)

            if details:
                return ToolResult(success=True, content="\n---\n".join(details))
        except (json.JSONDecodeError, TypeError):
            pass

        return result

    async def _mail_search(
        self,
        binary: str,
        query: str = "",
        max_results: int | float | None = None,
        **kw: Any,
    ) -> ToolResult:
        """Search emails using Gmail search syntax."""
        if not query:
            return ToolResult(success=False, error="query is required for mail_search.")

        limit = min(int(max_results or 10), 50)
        params: dict[str, Any] = {
            "userId": "me",
            "maxResults": limit,
            "q": query,
        }
        args = ["gmail", "users", "messages", "list", "--params", json.dumps(params)]
        result = await self._run_gws(binary, args)

        if not result.success:
            return result

        # Fetch details for each message.
        try:
            data = json.loads(result.content)
            messages = data.get("messages", [])
            if not messages:
                return ToolResult(success=True, content=f"No messages found for: {query}")

            details = []
            for msg in messages[:limit]:
                msg_id = msg.get("id", "")
                if not msg_id:
                    continue
                detail_result = await self._run_gws(binary, [
                    "gmail", "users", "messages", "get",
                    "--params", json.dumps({
                        "userId": "me",
                        "id": msg_id,
                        "format": "metadata",
                        "metadataHeaders": "From,To,Subject,Date",
                    }),
                ])
                if detail_result.success:
                    details.append(detail_result.content)

            if details:
                return ToolResult(success=True, content="\n---\n".join(details))
        except (json.JSONDecodeError, TypeError):
            pass

        return result

    async def _mail_read(
        self, binary: str, message_id: str = "", **kw: Any,
    ) -> ToolResult:
        """Read a specific email by message ID."""
        if not message_id:
            return ToolResult(success=False, error="message_id is required for mail_read.")

        params = {
            "userId": "me",
            "id": message_id,
            "format": "full",
        }
        args = ["gmail", "users", "messages", "get", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    # ------------------------------------------------------------------
    # Calendar actions
    # ------------------------------------------------------------------

    async def _calendar_list(
        self,
        binary: str,
        calendar_id: str = "primary",
        max_results: int | float | None = None,
        days: int | float | None = None,
        **kw: Any,
    ) -> ToolResult:
        """List upcoming calendar events."""
        from datetime import datetime, timedelta, timezone

        limit = min(int(max_results or 10), 100)
        look_ahead = int(days or 7)

        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=look_ahead)

        params: dict[str, Any] = {
            "calendarId": calendar_id,
            "maxResults": limit,
            "singleEvents": True,
            "orderBy": "startTime",
            "timeMin": now.isoformat(),
            "timeMax": time_max.isoformat(),
        }
        args = ["calendar", "events", "list", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _calendar_search(
        self,
        binary: str,
        query: str = "",
        calendar_id: str = "primary",
        max_results: int | float | None = None,
        days: int | float | None = None,
        **kw: Any,
    ) -> ToolResult:
        """Search calendar events by text."""
        if not query:
            return ToolResult(success=False, error="query is required for calendar_search.")

        from datetime import datetime, timedelta, timezone

        limit = min(int(max_results or 10), 100)
        look_ahead = int(days or 30)

        now = datetime.now(timezone.utc)
        time_max = now + timedelta(days=look_ahead)

        params: dict[str, Any] = {
            "calendarId": calendar_id,
            "maxResults": limit,
            "singleEvents": True,
            "orderBy": "startTime",
            "timeMin": now.isoformat(),
            "timeMax": time_max.isoformat(),
            "q": query,
        }
        args = ["calendar", "events", "list", "--params", json.dumps(params)]
        return await self._run_gws(binary, args)

    async def _calendar_create(
        self,
        binary: str,
        summary: str = "",
        start: str = "",
        end: str = "",
        attendees: str = "",
        calendar_id: str = "primary",
        **kw: Any,
    ) -> ToolResult:
        """Create a new calendar event."""
        if not summary:
            return ToolResult(success=False, error="summary is required for calendar_create.")
        if not start:
            return ToolResult(success=False, error="start is required for calendar_create.")

        # Build event via the helper command.
        args = ["calendar", "+insert", "--summary", summary, "--start", start]
        if end:
            args.extend(["--end", end])
        if attendees:
            args.extend(["--attendees", attendees])
        if calendar_id and calendar_id != "primary":
            args.extend(["--calendar-id", calendar_id])

        return await self._run_gws(binary, args)

    async def _calendar_agenda(
        self, binary: str, days: int | float | None = None, calendar_id: str = "primary", **kw: Any,
    ) -> ToolResult:
        """Show calendar agenda."""
        look_ahead = int(days or 7)
        args = ["calendar", "+agenda", "--days", str(look_ahead)]
        if calendar_id and calendar_id != "primary":
            args.extend(["--calendar-id", calendar_id])
        return await self._run_gws(binary, args, json_output=False)

    # ------------------------------------------------------------------
    # Raw passthrough
    # ------------------------------------------------------------------

    async def _raw(
        self, binary: str, raw_args: str = "", **kw: Any,
    ) -> ToolResult:
        """Run an arbitrary gws command."""
        if not raw_args:
            return ToolResult(success=False, error="raw_args is required for raw action.")

        # Split the raw args string into tokens.
        import shlex
        try:
            tokens = shlex.split(raw_args)
        except ValueError as e:
            return ToolResult(success=False, error=f"Failed to parse raw_args: {e}")

        return await self._run_gws(binary, tokens)
