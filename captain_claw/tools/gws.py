"""Google Workspace CLI (gws) tool for Drive, Docs, Gmail, and Calendar.

Wraps the ``gws`` CLI (https://github.com/googleworkspace/cli) which must
be installed and authenticated separately (``gws auth setup && gws auth login``).
The tool shells out to the ``gws`` binary with ``--format json`` and returns
the JSON (or formatted text) output to the agent.
"""

from __future__ import annotations

import asyncio
import base64
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


# ---------------------------------------------------------------------------
# Gmail payload cleaner — strip base64, decode text, save images
# ---------------------------------------------------------------------------

_HEADER_KEYS = ("From", "To", "Cc", "Bcc", "Subject", "Date", "Reply-To")


def _decode_base64url(data: str) -> bytes:
    """Decode Gmail-style base64url data (padding-tolerant)."""
    # Gmail uses URL-safe base64 without padding.
    padded = data + "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def _html_to_text(html: str) -> str:
    """Best-effort HTML-to-text for email bodies."""
    t = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<script[^>]*>.*?</script>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"</(?:p|div|tr|li|h[1-6])>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", "", t)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                          ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        t = t.replace(entity, char)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _clean_gmail_message(
    msg_data: dict[str, Any],
    save_dir: Path | None = None,
) -> str:
    """Convert a Gmail API message (format=full) to clean LLM-friendly text.

    * Decodes text/plain and text/html body parts into readable text.
    * Strips all base64-encoded data from the output.
    * Saves image attachments to *save_dir* (if provided).
    """
    payload = msg_data.get("payload", {})

    # ── Headers ──
    headers: dict[str, str] = {}
    for h in payload.get("headers", []):
        name = h.get("name", "")
        if name in _HEADER_KEYS:
            headers[name] = h.get("value", "")

    # ── Walk MIME parts ──
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict[str, Any]] = []

    def _walk(part: dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        filename = part.get("filename", "")
        body = part.get("body", {})
        data = body.get("data", "")
        size = body.get("size", 0)

        # Recurse into multipart containers.
        for sub in part.get("parts", []):
            _walk(sub)

        # Text body parts (no filename → inline content, not attachment).
        if mime == "text/plain" and data and not filename:
            try:
                text_parts.append(_decode_base64url(data).decode("utf-8", errors="replace"))
            except Exception:
                pass
            return
        if mime == "text/html" and data and not filename:
            try:
                html_parts.append(_decode_base64url(data).decode("utf-8", errors="replace"))
            except Exception:
                pass
            return

        # Skip multipart containers themselves.
        if mime.startswith("multipart/"):
            return

        # Attachments (files, images, etc.).
        if not filename and not data and not body.get("attachmentId"):
            return
        att_info: dict[str, Any] = {
            "filename": filename or "(unnamed)",
            "type": mime,
            "size": size,
        }

        # Save images to workspace.
        if data and mime.startswith("image/") and save_dir:
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                ext = mime.split("/")[-1].split(";")[0]
                safe_name = filename or f"image.{ext}"
                # Avoid overwriting: append counter if needed.
                dest = save_dir / safe_name
                counter = 1
                while dest.exists():
                    stem = Path(safe_name).stem
                    suffix = Path(safe_name).suffix
                    dest = save_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                dest.write_bytes(_decode_base64url(data))
                att_info["saved_to"] = str(dest)
            except Exception as exc:
                log.debug("failed to save mail image", error=str(exc))

        attachments.append(att_info)

    _walk(payload)

    # ── Build clean output ──
    lines: list[str] = []
    msg_id = msg_data.get("id", "")
    thread_id = msg_data.get("threadId", "")
    if msg_id:
        lines.append(f"Message ID: {msg_id}")
    if thread_id:
        lines.append(f"Thread ID: {thread_id}")
    for key in _HEADER_KEYS:
        if key in headers:
            lines.append(f"{key}: {headers[key]}")
    lines.append("")

    if text_parts:
        lines.append("".join(text_parts).strip())
    elif html_parts:
        lines.append(_html_to_text("".join(html_parts)))
    else:
        snippet = msg_data.get("snippet", "")
        if snippet:
            lines.append(f"[snippet] {snippet}")

    if attachments:
        lines.append("")
        lines.append(f"Attachments ({len(attachments)}):")
        for att in attachments:
            saved = att.get("saved_to")
            if saved:
                lines.append(
                    f"  - {att['filename']} ({att['type']}, {att['size']} bytes)"
                    f" → saved: {saved}"
                )
            else:
                lines.append(f"  - {att['filename']} ({att['type']}, {att['size']} bytes)")

    return "\n".join(lines)


def _clean_gmail_thread(
    thread_data: dict[str, Any],
    save_dir: Path | None = None,
) -> str:
    """Convert a Gmail API thread (format=full) to clean LLM-friendly text."""
    messages = thread_data.get("messages", [])
    if not messages:
        return "Empty thread."

    # Thread-level info from the first message.
    first = messages[0]
    first_headers: dict[str, str] = {}
    for h in first.get("payload", {}).get("headers", []):
        name = h.get("name", "")
        if name in ("Subject",):
            first_headers[name] = h.get("value", "")

    parts: list[str] = []
    thread_id = thread_data.get("id", first.get("threadId", ""))
    parts.append(f"Thread ID: {thread_id}")
    if "Subject" in first_headers:
        parts.append(f"Subject: {first_headers['Subject']}")
    parts.append(f"Messages: {len(messages)}")
    parts.append("")

    for idx, msg in enumerate(messages, 1):
        parts.append(f"── Message {idx}/{len(messages)} ──")
        parts.append(_clean_gmail_message(msg, save_dir=save_dir))
        parts.append("")

    return "\n".join(parts)


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
        "mail_threads (list recent email threads — grouped conversations), "
        "mail_read_thread (read all messages in an email thread by thread_id), "
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
                    "mail_threads",
                    "mail_read_thread",
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
            "thread_id": {
                "type": "string",
                "description": "Gmail thread ID (for mail_read_thread).",
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results to return (default varies by action).",
            },
            "output_file": {
                "type": "string",
                "description": (
                    "When set, write results directly to this file path instead of "
                    "returning raw data. Use for drive_list and drive_search when the "
                    "user wants a file listing saved to disk. The tool handles "
                    "pagination, formatting, and writing automatically. Returns a "
                    "summary (file count, path) instead of raw JSON."
                ),
            },
            "recursive": {
                "type": "boolean",
                "description": (
                    "For drive_list: when true, recursively traverse ALL folders "
                    "and subfolders, listing every file with its full folder path. "
                    "Must be used with output_file — results are written directly "
                    "to disk. Format: <folder/path/filename>, <link>, <size>, <type>."
                ),
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
        stream_cb = kwargs.pop("_stream_callback", None)
        self._stream_callback = stream_cb

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
            "mail_threads": self._mail_threads,
            "mail_read_thread": self._mail_read_thread,
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
        stream_cb = getattr(self, "_stream_callback", None)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def _read_stream(
            stream: asyncio.StreamReader, collected: list[str], prefix: str = "",
        ) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                collected.append(text)
                if stream_cb:
                    try:
                        stream_cb(prefix + text)
                    except Exception:
                        pass

        async def _collect() -> None:
            await asyncio.gather(
                _read_stream(proc.stdout, stdout_chunks),
                _read_stream(proc.stderr, stderr_chunks),
            )
            await proc.wait()

        try:
            await asyncio.wait_for(_collect(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(success=False, error="gws command timed out.")

        stdout_str = "".join(stdout_chunks).strip()
        stderr_str = "".join(stderr_chunks).strip()

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
        self, binary: str, folder_id: str = "", max_results: int | float | None = None,
        output_file: str = "", recursive: bool = False, **kw: Any,
    ) -> ToolResult:
        """List files in a Google Drive folder with auto-pagination."""
        # ── Recursive mode: flat-fetch everything, build tree ─────
        if recursive:
            if not output_file:
                return ToolResult(
                    success=False,
                    error="recursive=true requires output_file to be set.",
                )
            return await self._drive_recursive_list(binary, output_file, int(max_results or 10000))

        limit = int(max_results or 100)
        page_size = min(limit, 1000)
        params: dict[str, Any] = {"pageSize": page_size}

        if folder_id:
            escaped = folder_id.replace("'", "\\'")
            params["q"] = f"'{escaped}' in parents and trashed = false"
        else:
            params["q"] = "'root' in parents and trashed = false"

        params["orderBy"] = "modifiedTime desc"
        params["fields"] = "nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink,parents)"
        params["corpora"] = "allDrives"
        params["supportsAllDrives"] = "true"
        params["includeItemsFromAllDrives"] = "true"

        return await self._drive_paginated_list(binary, params, limit, output_file)

    async def _drive_search(
        self, binary: str, query: str = "", max_results: int | float | None = None,
        output_file: str = "", **kw: Any,
    ) -> ToolResult:
        """Search for files across Google Drive with auto-pagination."""
        if not query:
            return ToolResult(success=False, error="query is required for drive_search.")

        limit = int(max_results or 100)
        page_size = min(limit, 1000)
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        q = f"(name contains '{escaped}' or fullText contains '{escaped}') and trashed = false"

        params: dict[str, Any] = {
            "pageSize": page_size,
            "q": q,
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink)",
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        return await self._drive_paginated_list(binary, params, limit, output_file)

    # ------------------------------------------------------------------
    # Drive pagination + direct file writing
    # ------------------------------------------------------------------

    @staticmethod
    def _format_file_size(size_str: str) -> str:
        """Convert byte string to human-readable size."""
        try:
            size = int(size_str)
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if size < 1024:
                    return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
                size /= 1024
            return f"{size:.1f} PB"
        except (ValueError, TypeError):
            return str(size_str) if size_str else "—"

    @staticmethod
    def _readable_mime(mime: str) -> str:
        """Convert MIME type to a short human-readable label."""
        _MAP = {
            "application/vnd.google-apps.folder": "Folder",
            "application/vnd.google-apps.document": "Google Doc",
            "application/vnd.google-apps.spreadsheet": "Google Sheet",
            "application/vnd.google-apps.presentation": "Google Slides",
            "application/vnd.google-apps.form": "Google Form",
            "application/vnd.google-apps.drawing": "Google Drawing",
            "application/vnd.google-apps.site": "Google Site",
            "application/pdf": "PDF",
            "text/csv": "CSV",
            "text/plain": "Text",
            "application/json": "JSON",
        }
        if mime in _MAP:
            return _MAP[mime]
        if "image/" in mime:
            return "Image"
        if "video/" in mime:
            return "Video"
        if "audio/" in mime:
            return "Audio"
        if "officedocument" in mime or "msword" in mime:
            return "Office Doc"
        return mime.split("/")[-1].upper() if "/" in mime else mime or "Unknown"

    @staticmethod
    def _format_drive_file_line(f: dict[str, Any]) -> str:
        """Format a single Drive file dict into a text line."""
        name = f.get("name", "Unnamed")
        link = f.get("webViewLink", "—")
        size = GwsTool._format_file_size(f.get("size", ""))
        ftype = GwsTool._readable_mime(f.get("mimeType", ""))
        return f"{name}, {link}, {size}, {ftype}"

    async def _drive_paginated_list(
        self,
        binary: str,
        params: dict[str, Any],
        limit: int,
        output_file: str = "",
    ) -> ToolResult:
        """Paginate through Drive file list results.

        When *output_file* is set the tool writes each file line directly
        to disk as it is fetched, streams progress via ``_stream_callback``,
        and returns a short summary.  When *output_file* is empty the raw
        JSON is returned (capped by ``_MAX_OUTPUT_CHARS``).
        """
        stream_cb = getattr(self, "_stream_callback", None)
        all_files: list[dict[str, Any]] = []
        page_token: str | None = None
        _max_pages = 50
        page_num = 0

        # ── Prepare output file if requested ──────────────────────
        out_fh = None
        out_path: Path | None = None
        if output_file:
            out_path = Path(output_file).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_fh = open(out_path, "w", encoding="utf-8")  # noqa: SIM115
            if stream_cb:
                stream_cb(f"Writing results to {out_path}\n")

        try:
            for _ in range(_max_pages):
                page_num += 1
                page_params = dict(params)
                if page_token:
                    page_params["pageToken"] = page_token

                if stream_cb:
                    stream_cb(f"Fetching page {page_num}...\n")

                args = ["drive", "files", "list", "--params", json.dumps(page_params)]
                result = await self._run_gws(binary, args)

                if not result.success:
                    if all_files:
                        break
                    return result

                try:
                    data = json.loads(result.content)
                except (json.JSONDecodeError, TypeError):
                    if all_files:
                        break
                    return result

                files = data.get("files", [])
                if not files:
                    break

                # ── Write / accumulate ────────────────────────────
                for f in files:
                    if len(all_files) >= limit:
                        break
                    all_files.append(f)
                    if out_fh:
                        out_fh.write(self._format_drive_file_line(f) + "\n")

                if stream_cb:
                    stream_cb(f"  ↳ {len(all_files)} files collected so far\n")

                page_token = data.get("nextPageToken")
                if not page_token or len(all_files) >= limit:
                    break
        finally:
            if out_fh:
                out_fh.close()

        # ── Build result ──────────────────────────────────────────
        if out_path:
            summary = (
                f"✓ {len(all_files)} files written to {out_path}\n"
                f"  Format: <filename>, <google docs link>, <size>, <type>\n"
                f"  Pages fetched: {page_num}"
            )
            if stream_cb:
                stream_cb(f"\n{summary}\n")
            return ToolResult(success=True, content=summary)

        # No output_file — return JSON payload.
        all_files = all_files[:limit]
        output = json.dumps({"files": all_files, "totalFiles": len(all_files)}, indent=2)
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n\n... [output truncated]"
        return ToolResult(success=True, content=output)

    # ------------------------------------------------------------------
    # Recursive drive listing (flat-fetch + tree reconstruction)
    # ------------------------------------------------------------------

    async def _drive_recursive_list(
        self,
        binary: str,
        output_file: str,
        limit: int = 10000,
    ) -> ToolResult:
        """Fetch ALL Drive files in one sweep, reconstruct folder paths, write to file.

        Strategy:
        1. Paginate through the entire Drive (no parent filter) collecting
           every file and folder.
        2. Build a folder-id → name lookup.
        3. Reconstruct full paths by walking parent chains.
        4. Write each file as ``<path/name>, <link>, <size>, <type>`` to *output_file*.

        This is far more efficient than recursive per-folder API calls.
        """
        stream_cb = getattr(self, "_stream_callback", None)

        # ── Phase 1: Fetch everything ─────────────────────────────
        if stream_cb:
            stream_cb("Phase 1: Fetching all files from Google Drive...\n")

        params: dict[str, Any] = {
            "pageSize": 1000,
            "q": "trashed = false",
            "fields": "nextPageToken,files(id,name,mimeType,size,webViewLink,parents)",
            "corpora": "allDrives",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }

        all_items: list[dict[str, Any]] = []
        page_token: str | None = None
        page_num = 0

        for _ in range(100):  # safety cap
            page_num += 1
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token

            if stream_cb:
                stream_cb(f"  Fetching page {page_num}...\n")

            args = ["drive", "files", "list", "--params", json.dumps(page_params)]
            result = await self._run_gws(binary, args)

            if not result.success:
                if all_items:
                    if stream_cb:
                        stream_cb(f"  ⚠ Page {page_num} failed, continuing with {len(all_items)} items\n")
                    break
                return result

            try:
                data = json.loads(result.content)
            except (json.JSONDecodeError, TypeError):
                if all_items:
                    break
                return result

            files = data.get("files", [])
            if not files:
                break

            all_items.extend(files)
            if stream_cb:
                stream_cb(f"  ↳ {len(all_items)} items collected\n")

            if len(all_items) >= limit:
                all_items = all_items[:limit]
                break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        if not all_items:
            return ToolResult(success=True, content="No files found in Google Drive.")

        # ── Phase 2: Build folder tree ────────────────────────────
        if stream_cb:
            stream_cb(f"\nPhase 2: Building folder tree from {len(all_items)} items...\n")

        FOLDER_MIME = "application/vnd.google-apps.folder"
        folder_map: dict[str, str] = {}  # id → name
        for item in all_items:
            if item.get("mimeType") == FOLDER_MIME:
                folder_map[item["id"]] = item.get("name", "Unnamed")

        # Resolve full path for a given parent list (with memoization).
        _path_cache: dict[str, str] = {}

        def _resolve_path(parent_id: str) -> str:
            if parent_id in _path_cache:
                return _path_cache[parent_id]
            if parent_id not in folder_map:
                # Root or shared drive root.
                _path_cache[parent_id] = "My Drive"
                return "My Drive"
            # Walk up — find this folder's own parents.
            folder_name = folder_map[parent_id]
            # Find the folder item to get its parents.
            for item in all_items:
                if item.get("id") == parent_id:
                    parents = item.get("parents", [])
                    if parents:
                        parent_path = _resolve_path(parents[0])
                        full = f"{parent_path}/{folder_name}"
                    else:
                        full = folder_name
                    _path_cache[parent_id] = full
                    return full
            _path_cache[parent_id] = folder_name
            return folder_name

        # ── Phase 3: Write output file ────────────────────────────
        if stream_cb:
            stream_cb("Phase 3: Writing output file...\n")

        out_path = Path(output_file).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        file_count = 0
        folder_count = 0
        folders_seen: set[str] = set()

        with open(out_path, "w", encoding="utf-8") as fh:
            for item in all_items:
                name = item.get("name", "Unnamed")
                mime = item.get("mimeType", "")
                parents = item.get("parents", [])
                parent_path = _resolve_path(parents[0]) if parents else "My Drive"

                # Track unique folders.
                if parents:
                    folders_seen.add(parents[0])

                full_path = f"{parent_path}/{name}"
                link = item.get("webViewLink", "—")
                size = self._format_file_size(item.get("size", ""))
                ftype = self._readable_mime(mime)

                if mime == FOLDER_MIME:
                    folder_count += 1
                else:
                    file_count += 1

                fh.write(f"{full_path}, {link}, {size}, {ftype}\n")

        summary = (
            f"✓ Recursive listing complete → {out_path}\n"
            f"  Total items: {len(all_items)} ({file_count} files, {folder_count} folders)\n"
            f"  Unique folders traversed: {len(folders_seen)}\n"
            f"  Pages fetched: {page_num}\n"
            f"  Format: <full/path/name>, <link>, <size>, <type>"
        )
        if stream_cb:
            stream_cb(f"\n{summary}\n")
        return ToolResult(success=True, content=summary)

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

    @staticmethod
    def _mail_attachment_dir() -> Path | None:
        """Return the workspace directory for saving mail attachments."""
        try:
            cfg = get_config()
            workspace = cfg.resolved_workspace_path()
            return workspace / "saved" / "mail_attachments"
        except Exception:
            return None

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
        result = await self._run_gws(binary, args)
        if not result.success:
            return result

        try:
            msg_data = json.loads(result.content)
            save_dir = self._mail_attachment_dir()
            return ToolResult(success=True, content=_clean_gmail_message(msg_data, save_dir=save_dir))
        except (json.JSONDecodeError, TypeError):
            # Fallback: strip any raw base64 from the output.
            return ToolResult(success=True, content=_strip_base64_images(result.content))

    async def _mail_threads(
        self,
        binary: str,
        label: str = "INBOX",
        query: str = "",
        max_results: int | float | None = None,
        **kw: Any,
    ) -> ToolResult:
        """List recent email threads (grouped conversations)."""
        limit = min(int(max_results or 10), 50)
        params: dict[str, Any] = {
            "userId": "me",
            "maxResults": limit,
        }
        if label:
            params["labelIds"] = label.upper()
        if query:
            params["q"] = query
        args = ["gmail", "users", "threads", "list", "--params", json.dumps(params)]
        result = await self._run_gws(binary, args)

        if not result.success:
            return result

        # Fetch summary for each thread (metadata of first message).
        try:
            data = json.loads(result.content)
            threads = data.get("threads", [])
            if not threads:
                return ToolResult(success=True, content=f"No threads found in {label or 'mailbox'}.")

            summaries: list[str] = []
            for thread in threads[:limit]:
                thread_id = thread.get("id", "")
                if not thread_id:
                    continue
                # Get thread with metadata only (lightweight).
                thread_result = await self._run_gws(binary, [
                    "gmail", "users", "threads", "get",
                    "--params", json.dumps({
                        "userId": "me",
                        "id": thread_id,
                        "format": "metadata",
                        "metadataHeaders": "From,To,Subject,Date",
                    }),
                ])
                if not thread_result.success:
                    continue

                try:
                    thread_data = json.loads(thread_result.content)
                except (json.JSONDecodeError, TypeError):
                    summaries.append(thread_result.content)
                    continue

                messages = thread_data.get("messages", [])
                msg_count = len(messages)
                # Extract headers from the first message.
                first_msg = messages[0] if messages else {}
                headers = {}
                for h in (first_msg.get("payload", {}).get("headers", [])):
                    name = h.get("name", "")
                    if name in ("From", "To", "Subject", "Date"):
                        headers[name] = h.get("value", "")

                # Collect unique senders across the thread.
                senders: list[str] = []
                seen: set[str] = set()
                for m in messages:
                    for h in (m.get("payload", {}).get("headers", [])):
                        if h.get("name") == "From":
                            val = h.get("value", "")
                            if val and val not in seen:
                                senders.append(val)
                                seen.add(val)

                snippet = first_msg.get("snippet", "")
                summary_parts = [
                    f"Thread ID: {thread_id}",
                    f"Subject: {headers.get('Subject', '(no subject)')}",
                    f"From: {headers.get('From', '?')}",
                    f"Date: {headers.get('Date', '?')}",
                    f"Messages: {msg_count}",
                ]
                if len(senders) > 1:
                    summary_parts.append(f"Participants: {', '.join(senders)}")
                if snippet:
                    summary_parts.append(f"Preview: {snippet[:120]}")
                summaries.append("\n".join(summary_parts))

            if summaries:
                return ToolResult(success=True, content="\n---\n".join(summaries))
        except (json.JSONDecodeError, TypeError):
            pass

        return result

    async def _mail_read_thread(
        self, binary: str, thread_id: str = "", **kw: Any,
    ) -> ToolResult:
        """Read all messages in an email thread by thread ID."""
        if not thread_id:
            return ToolResult(success=False, error="thread_id is required for mail_read_thread.")

        params = {
            "userId": "me",
            "id": thread_id,
            "format": "full",
        }
        args = ["gmail", "users", "threads", "get", "--params", json.dumps(params)]
        result = await self._run_gws(binary, args)
        if not result.success:
            return result

        try:
            thread_data = json.loads(result.content)
            save_dir = self._mail_attachment_dir()
            return ToolResult(success=True, content=_clean_gmail_thread(thread_data, save_dir=save_dir))
        except (json.JSONDecodeError, TypeError):
            return ToolResult(success=True, content=_strip_base64_images(result.content))

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
