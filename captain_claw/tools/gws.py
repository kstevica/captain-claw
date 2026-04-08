"""Google Workspace CLI (gws) tool for Drive, Docs, and Calendar.

Wraps the ``gws`` CLI (https://github.com/googleworkspace/cli) which must
be installed and authenticated separately (``gws auth setup && gws auth login``).

Gmail is intentionally NOT handled here — use the :class:`GoogleMailTool`
instead, which uses the centralized OAuth token manager and supports
sending, drafting, replying, and label modification. This tool focuses
on Drive/Docs/Calendar, where the gws CLI's binary exports and Docs
write helpers are genuinely useful.
"""

from __future__ import annotations

import asyncio
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools._gws_calendar import GwsCalendarMixin
from captain_claw.tools._gws_docs import GwsDocsMixin
from captain_claw.tools._gws_drive import GwsDriveMixin
from captain_claw.tools._gws_runtime import GwsRuntimeMixin
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


class GwsTool(
    Tool,
    GwsRuntimeMixin,
    GwsDriveMixin,
    GwsDocsMixin,
    GwsCalendarMixin,
):
    """Google Workspace CLI — Drive, Docs, and Calendar via the ``gws`` binary."""

    name = "gws"
    description = (
        "Google Workspace CLI tool (Drive / Docs / Calendar). "
        "Actions: drive_list (list files in Drive), drive_search (find files by name/content), "
        "drive_download (download/export a file locally), drive_info (file metadata), "
        "drive_create (create a new Google Doc/Sheet/file on Drive), "
        "docs_read (read a Google Doc — returns the full document text INLINE in the "
        "tool result; do NOT attempt to read a file after docs_read, the content is "
        "already in the response), docs_append (append text to a Doc), "
        "calendar_list (list upcoming events), calendar_search (search events), "
        "calendar_create (create a calendar event), calendar_agenda (show agenda). "
        "For Gmail, use the 'google_mail' tool instead — it supports reading, sending, "
        "drafting, replying, and label modification. "
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
                    "Search query text. Used by drive_search (file name/content) "
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
        GwsRuntimeMixin.__init__(self)

    # ------------------------------------------------------------------
    # Execution dispatch
    # ------------------------------------------------------------------

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        runtime_base = kwargs.pop("_runtime_base_path", None)
        saved_base = kwargs.pop("_saved_base_path", None)
        kwargs.pop("_session_id", None)
        kwargs.pop("_abort_event", None)
        kwargs.pop("_file_registry", None)
        kwargs.pop("_task_id", None)
        self._stream_callback = kwargs.pop("_stream_callback", None)

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
            "calendar_list": self._calendar_list,
            "calendar_search": self._calendar_search,
            "calendar_create": self._calendar_create,
            "calendar_agenda": self._calendar_agenda,
            "raw": self._raw,
        }

        handler = handlers.get(action)
        if handler is None:
            # Provide a helpful redirect if the agent still uses the old
            # gmail actions that now live in GoogleMailTool.
            if action in {"mail_list", "mail_search", "mail_read", "mail_threads", "mail_read_thread"}:
                return ToolResult(
                    success=False,
                    error=(
                        f"Action '{action}' has moved. Gmail is now handled by the "
                        "'google_mail' tool — use google_mail with equivalent actions: "
                        "list_messages, search, read_message, get_thread."
                    ),
                )
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
    # Raw passthrough
    # ------------------------------------------------------------------

    async def _raw(
        self, binary: str, raw_args: str = "", **kw: Any,
    ) -> ToolResult:
        """Run an arbitrary gws command."""
        if not raw_args:
            return ToolResult(success=False, error="raw_args is required for raw action.")

        import shlex
        try:
            tokens = shlex.split(raw_args)
        except ValueError as e:
            return ToolResult(success=False, error=f"Failed to parse raw_args: {e}")

        return await self._run_gws(binary, tokens)
