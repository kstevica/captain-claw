"""Google Mail (Gmail) tool for reading emails (read-only).

Uses the Gmail REST API v1 via httpx with OAuth2 Bearer tokens
managed by :class:`~captain_claw.google_oauth_manager.GoogleOAuthManager`.
Only ``gmail.readonly`` scope is required — no send/modify/delete operations.
"""

from __future__ import annotations

import base64
import email.utils
import html as html_module
import re
from typing import Any

import httpx

from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GMAIL_API = "https://gmail.googleapis.com/gmail/v1"
_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

# Max body length returned to the agent.
_MAX_BODY_CHARS = 30_000


class GoogleMailTool(Tool):
    """Read emails from Gmail (read-only). Actions: list_messages, search,
    read_message, list_labels, get_thread."""

    name = "google_mail"
    description = (
        "Read emails from Gmail (read-only). Actions: "
        "list_messages (list recent emails from inbox or a label), "
        "search (find emails by Gmail search query like 'from:alice subject:report'), "
        "read_message (get full email content by message ID), "
        "get_thread (get all messages in a thread), "
        "list_labels (list available Gmail labels/folders)."
    )
    timeout_seconds = 120.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_messages",
                    "search",
                    "read_message",
                    "get_thread",
                    "list_labels",
                ],
                "description": "The action to perform.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Gmail search query (for search action). Supports all Gmail "
                    "search operators: from:, to:, subject:, has:attachment, "
                    "after:2026/01/01, before:, is:unread, label:, etc."
                ),
            },
            "message_id": {
                "type": "string",
                "description": "Message ID (for read_message action).",
            },
            "thread_id": {
                "type": "string",
                "description": "Thread ID (for get_thread action).",
            },
            "label": {
                "type": "string",
                "description": (
                    "Label/folder to list from (for list_messages). "
                    "Common values: INBOX, SENT, DRAFT, STARRED, UNREAD, SPAM, TRASH. "
                    "Defaults to INBOX."
                ),
            },
            "max_results": {
                "type": "number",
                "description": "Maximum number of results (default 10, max 50).",
            },
            "include_body": {
                "type": "boolean",
                "description": (
                    "Whether to include message body in list/search results. "
                    "Default false for list/search (headers only), always true for read_message."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Gmail Tool)"},
        )

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
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
            "list_messages": self._action_list_messages,
            "search": self._action_search,
            "read_message": self._action_read_message,
            "get_thread": self._action_get_thread,
            "list_labels": self._action_list_labels,
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
            log.error("Gmail HTTP error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"HTTP error: {exc}")
        except Exception as exc:
            log.error("Gmail tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        """Retrieve a valid Google OAuth access token."""
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

        granted = set(tokens.scope.split()) if tokens.scope else set()
        if _GMAIL_SCOPE not in granted:
            raise RuntimeError(
                "Gmail readonly scope not granted. Your current OAuth connection "
                "does not include Gmail access. Please disconnect and reconnect "
                "your Google account to grant Gmail permissions."
            )

        return tokens.access_token

    def _auth_headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_http_error(exc: httpx.HTTPStatusError) -> ToolResult:
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
            return ToolResult(success=False, error=f"Permission denied: {message}")
        elif status == 404:
            return ToolResult(success=False, error="Message or thread not found.")
        elif status == 429:
            return ToolResult(
                success=False,
                error="Gmail rate limit exceeded. Please try again in a moment.",
            )
        else:
            return ToolResult(
                success=False,
                error=f"Gmail API error ({status}): {message}",
            )

    # ------------------------------------------------------------------
    # Action: list_labels
    # ------------------------------------------------------------------

    async def _action_list_labels(
        self, token: str, **kwargs: Any,
    ) -> ToolResult:
        """List all Gmail labels."""
        resp = await self._client.get(
            f"{_GMAIL_API}/users/me/labels",
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        labels = data.get("labels", [])
        if not labels:
            return ToolResult(success=True, content="No labels found.")

        system_labels = []
        user_labels = []
        for lab in labels:
            name = lab.get("name", lab.get("id", "?"))
            lab_id = lab.get("id", "?")
            lab_type = lab.get("type", "user")
            entry = f"  {name}  (id: {lab_id})"
            if lab_type == "system":
                system_labels.append(entry)
            else:
                user_labels.append(entry)

        lines = [f"Gmail labels ({len(labels)} total):\n"]
        if system_labels:
            lines.append("System labels:")
            lines.extend(sorted(system_labels))
        if user_labels:
            lines.append("\nUser labels:")
            lines.extend(sorted(user_labels))

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: list_messages
    # ------------------------------------------------------------------

    async def _action_list_messages(
        self,
        token: str,
        label: str = "INBOX",
        max_results: int | float | None = None,
        include_body: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """List recent messages from a label."""
        limit = min(int(max_results or 10), 50)

        params: dict[str, Any] = {
            "maxResults": limit,
            "labelIds": label.upper(),
        }

        resp = await self._client.get(
            f"{_GMAIL_API}/users/me/messages",
            params=params,
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        message_stubs = data.get("messages", [])
        if not message_stubs:
            return ToolResult(success=True, content=f"No messages found in {label}.")

        messages = await self._fetch_message_batch(
            token, [m["id"] for m in message_stubs], include_body=include_body,
        )

        lines = [f"Messages in {label} ({len(messages)} shown):\n"]
        for msg in messages:
            lines.append(self._format_message_summary(msg, include_body=include_body))

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: search
    # ------------------------------------------------------------------

    async def _action_search(
        self,
        token: str,
        query: str = "",
        max_results: int | float | None = None,
        include_body: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Search messages using Gmail search syntax."""
        if not query:
            return ToolResult(success=False, error="Search query is required.")

        limit = min(int(max_results or 10), 50)

        resp = await self._client.get(
            f"{_GMAIL_API}/users/me/messages",
            params={"q": query, "maxResults": limit},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        message_stubs = data.get("messages", [])
        if not message_stubs:
            return ToolResult(success=True, content=f"No messages found for: {query}")

        messages = await self._fetch_message_batch(
            token, [m["id"] for m in message_stubs], include_body=include_body,
        )

        lines = [f"Search results for '{query}' ({len(messages)} found):\n"]
        for msg in messages:
            lines.append(self._format_message_summary(msg, include_body=include_body))

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: read_message
    # ------------------------------------------------------------------

    async def _action_read_message(
        self,
        token: str,
        message_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Read the full content of a single message."""
        if not message_id:
            return ToolResult(success=False, error="message_id is required.")

        msg = await self._fetch_full_message(token, message_id)
        return ToolResult(success=True, content=self._format_message_detail(msg))

    # ------------------------------------------------------------------
    # Action: get_thread
    # ------------------------------------------------------------------

    async def _action_get_thread(
        self,
        token: str,
        thread_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Get all messages in a thread."""
        if not thread_id:
            return ToolResult(success=False, error="thread_id is required.")

        resp = await self._client.get(
            f"{_GMAIL_API}/users/me/threads/{thread_id}",
            params={"format": "full"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        messages = data.get("messages", [])
        if not messages:
            return ToolResult(success=True, content="Thread is empty.")

        lines = [f"Thread {thread_id} ({len(messages)} messages):\n"]
        for i, msg in enumerate(messages, 1):
            parsed = self._parse_message(msg)
            lines.append(f"--- Message {i}/{len(messages)} ---")
            lines.append(self._format_message_detail(parsed))
            lines.append("")

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Message fetching helpers
    # ------------------------------------------------------------------

    async def _fetch_message_batch(
        self, token: str, message_ids: list[str], include_body: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch metadata (or full) for a batch of messages."""
        fmt = "full" if include_body else "metadata"
        meta_headers = "From,To,Subject,Date"

        messages: list[dict[str, Any]] = []
        for msg_id in message_ids:
            try:
                params: dict[str, str] = {"format": fmt}
                if fmt == "metadata":
                    params["metadataHeaders"] = meta_headers
                resp = await self._client.get(
                    f"{_GMAIL_API}/users/me/messages/{msg_id}",
                    params=params,
                    headers=self._auth_headers(token),
                )
                resp.raise_for_status()
                messages.append(self._parse_message(resp.json()))
            except Exception as exc:
                log.debug("Failed to fetch message %s: %s", msg_id, exc)
                messages.append({"id": msg_id, "error": str(exc)})

        return messages

    async def _fetch_full_message(
        self, token: str, message_id: str,
    ) -> dict[str, Any]:
        """Fetch full message content."""
        resp = await self._client.get(
            f"{_GMAIL_API}/users/me/messages/{message_id}",
            params={"format": "full"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        return self._parse_message(resp.json())

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_message(raw: dict[str, Any]) -> dict[str, Any]:
        """Parse a Gmail API message into a clean dict."""
        msg_id = raw.get("id", "?")
        thread_id = raw.get("threadId", "")
        label_ids = raw.get("labelIds", [])
        snippet = raw.get("snippet", "")
        internal_date = raw.get("internalDate", "")

        # Extract headers
        payload = raw.get("payload", {})
        headers_raw = payload.get("headers", [])
        headers: dict[str, str] = {}
        for h in headers_raw:
            name = h.get("name", "").lower()
            if name in ("from", "to", "cc", "bcc", "subject", "date", "reply-to"):
                headers[name] = h.get("value", "")

        # Extract body
        body_text = ""
        body_parts: dict[str, list[str]] = {"text": [], "html": []}
        att_list: list[dict[str, str]] = []
        GoogleMailTool._extract_parts(payload, body_parts=body_parts, attachments=att_list)

        if body_parts["text"]:
            body_text = "\n".join(body_parts["text"])
        elif body_parts["html"]:
            body_text = GoogleMailTool._html_to_text("\n".join(body_parts["html"]))

        is_unread = "UNREAD" in label_ids
        is_starred = "STARRED" in label_ids

        return {
            "id": msg_id,
            "thread_id": thread_id,
            "labels": label_ids,
            "snippet": snippet,
            "date": headers.get("date", ""),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "cc": headers.get("cc", ""),
            "subject": headers.get("subject", "(no subject)"),
            "body": body_text[:_MAX_BODY_CHARS] if body_text else "",
            "attachments": att_list,
            "is_unread": is_unread,
            "is_starred": is_starred,
        }

    @staticmethod
    def _extract_parts(
        payload: dict[str, Any],
        body_parts: dict[str, list[str]],
        attachments: list[dict[str, str]],
    ) -> None:
        """Recursively extract text/html bodies and attachment info from MIME parts."""
        mime_type = payload.get("mimeType", "")
        body = payload.get("body", {})
        filename = payload.get("filename", "")

        # Attachment
        if filename and body.get("attachmentId"):
            attachments.append({
                "filename": filename,
                "mime_type": mime_type,
                "size": str(body.get("size", "?")),
            })
            return

        # Leaf part with data
        data = body.get("data", "")
        if data:
            try:
                decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            except Exception:
                decoded = ""
            if "text/plain" in mime_type and decoded:
                body_parts["text"].append(decoded)
            elif "text/html" in mime_type and decoded:
                body_parts["html"].append(decoded)

        # Recurse into sub-parts (multipart/*)
        for part in payload.get("parts", []):
            GoogleMailTool._extract_parts(part, body_parts, attachments)

    @staticmethod
    def _html_to_text(html_str: str) -> str:
        """Best-effort HTML to plain text conversion."""
        text = html_str
        # Replace common block tags with newlines
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|tr|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<(hr)\s*/?>", "\n---\n", text, flags=re.IGNORECASE)
        # Strip all remaining tags
        text = re.sub(r"<[^>]+>", "", text)
        # Decode HTML entities
        text = html_module.unescape(text)
        # Collapse whitespace
        lines = [line.rstrip() for line in text.splitlines()]
        # Remove excessive blank lines
        cleaned: list[str] = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append(line)
            else:
                blank_count = 0
                cleaned.append(line)
        return "\n".join(cleaned).strip()

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_message_summary(msg: dict[str, Any], include_body: bool = False) -> str:
        """Format a message for list/search output (compact)."""
        if "error" in msg:
            return f"  [error] {msg['id']}: {msg['error']}"

        unread = " [NEW]" if msg.get("is_unread") else ""
        starred = " ★" if msg.get("is_starred") else ""
        att_count = len(msg.get("attachments", []))
        att_str = f"  📎{att_count}" if att_count else ""

        from_addr = msg.get("from", "?")
        # Shorten from: "John Doe <john@example.com>" → "John Doe"
        parsed = email.utils.parseaddr(from_addr)
        from_short = parsed[0] if parsed[0] else parsed[1]

        date_str = msg.get("date", "")
        # Shorten date to just date + time
        try:
            parsed_date = email.utils.parsedate_to_datetime(date_str)
            date_short = parsed_date.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_short = date_str[:20] if date_str else ""

        lines = [
            f"  {'📩' if msg.get('is_unread') else '📧'} {msg.get('subject', '(no subject)')}{unread}{starred}{att_str}",
            f"    From: {from_short}  |  Date: {date_short}",
            f"    ID: {msg['id']}  |  Thread: {msg.get('thread_id', '')}",
        ]

        if include_body and msg.get("body"):
            # Show first ~300 chars of body in summary mode
            preview = msg["body"][:300]
            if len(msg["body"]) > 300:
                preview += "…"
            lines.append(f"    Preview: {preview}")
        elif msg.get("snippet"):
            lines.append(f"    Preview: {msg['snippet']}")

        return "\n".join(lines)

    @staticmethod
    def _format_message_detail(msg: dict[str, Any]) -> str:
        """Format a full message for read_message output."""
        if "error" in msg:
            return f"Error reading message {msg['id']}: {msg['error']}"

        lines = [
            f"Subject: {msg.get('subject', '(no subject)')}",
            f"From: {msg.get('from', '?')}",
            f"To: {msg.get('to', '?')}",
        ]
        if msg.get("cc"):
            lines.append(f"CC: {msg['cc']}")
        lines.extend([
            f"Date: {msg.get('date', '?')}",
            f"ID: {msg['id']}  |  Thread: {msg.get('thread_id', '')}",
            f"Labels: {', '.join(msg.get('labels', []))}",
        ])

        if msg.get("attachments"):
            lines.append(f"\nAttachments ({len(msg['attachments'])}):")
            for att in msg["attachments"]:
                lines.append(f"  📎 {att['filename']} ({att['mime_type']}, {att['size']} bytes)")

        if msg.get("body"):
            lines.append(f"\n{'─' * 60}")
            lines.append(msg["body"])
        else:
            lines.append("\n(no body content)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
