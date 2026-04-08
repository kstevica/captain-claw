"""Google Mail (Gmail) tool — read and draft.

Uses the Gmail REST API v1 via httpx with OAuth2 Bearer tokens managed
by :class:`~captain_claw.google_oauth_manager.GoogleOAuthManager`. This
is the canonical Gmail integration for captain-claw — the gws CLI tool
no longer handles Gmail.

Required OAuth scopes:

* ``gmail.readonly`` — list, search, read messages and threads.
* ``gmail.compose``  — create drafts.

Both are requested as part of the standard Google OAuth login flow.
This tool intentionally does NOT expose send / reply / label / trash
actions — users review drafts in Gmail and send them manually.
"""

from __future__ import annotations

import base64
import email.policy
import email.utils
import html as html_module
import re
from email.message import EmailMessage
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

# Read operations accept gmail.readonly or the broader gmail.modify
# (in case a legacy connection still has it). Draft creation requires
# gmail.compose (gmail.modify also suffices).
_GMAIL_READ_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
)
_GMAIL_COMPOSE_SCOPES = (
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
)

# Max body length returned to the agent.
_MAX_BODY_CHARS = 30_000

# Appended to list / search output so the LLM always sees the routing
# reminder next to the IDs it should pass back in. Helps prevent the
# model from reaching for filesystem read/glob when the user asks to
# "read" or "open" one of the listed emails.
_FOLLOWUP_HINT = (
    "\nNext steps — use google_mail actions only (never filesystem tools):\n"
    "  • Read full email body: google_mail action=read_message "
    "message_id=<Newest msg ID above>\n"
    "  • Read whole conversation: google_mail action=get_thread "
    "thread_id=<Thread ID above>\n"
    "  • Reply to one: google_mail action=create_draft "
    "reply_to_message_id=<Newest msg ID above> body=...\n"
    "  • Narrow the list: google_mail action=search query='from:... is:unread'"
)


class GoogleMailTool(Tool):
    """Read Gmail and create drafts. Actions: list_messages, search,
    read_message, get_thread, list_labels, create_draft."""

    name = "google_mail"
    description = (
        "Gmail — read AND create drafts (no sending; the user reviews and sends drafts in Gmail). "
        "ROUTING: any user request that refers to an email, message, inbox, thread, "
        "conversation, sender, subject line, or Gmail — including phrases like "
        "'read the one from X', 'open that email', 'show me the Fil Rouge email', "
        "'what does Alice's message say', 'reply to this' — MUST be handled with "
        "google_mail actions (read_message / get_thread / search / create_draft). "
        "NEVER use filesystem tools (read, glob, grep) to try to fulfill email "
        "requests — emails do not live on disk. If you just ran list_messages or "
        "search and the user refers to one of the items by sender/subject, call "
        "read_message with the `Newest msg:` ID (or get_thread with the `Thread:` ID) "
        "printed next to that item. "
        "DEFAULT SCOPE: unless the user explicitly asks for another folder / category / "
        "archived mail / spam / trash / all-mail, reads and searches are restricted to "
        "the INBOX **Primary** tab (i.e. `in:inbox category:primary`) so Promotions, "
        "Social, Updates, and Forums clutter are excluded. "
        "THREAD-CENTRIC: list_messages and search are grouped by conversation by "
        "default — one item per thread, showing the newest activity, unread count, "
        "and participants. When the user asks 'how many new emails' or 'what's new', "
        "count threads, not raw messages (a 4-reply unread conversation is ONE new "
        "item, not four). If the user explicitly wants every individual message, pass "
        "`group_by_thread=false`. "
        "For list_messages, leave `label` unset (defaults to INBOX Primary). Only set it "
        "when the user names a specific folder like SENT/DRAFT/STARRED. "
        "For search, if the query does not already contain `in:`, `label:`, or `category:`, "
        "`in:inbox category:primary` is auto-prepended. Pass an explicit operator "
        "(e.g. `category:promotions`, `in:anywhere`, `label:work`) to broaden. "
        "Actions: "
        "list_messages (list recent Primary-tab threads, newest first), "
        "search (Gmail search query — e.g. 'from:alice subject:report'; Primary-scoped, thread-grouped), "
        "read_message (get full email content by message ID), "
        "get_thread (get all messages in a thread), "
        "list_labels (list available Gmail labels/folders), "
        "create_draft (save a draft — never sends; user reviews in Gmail and sends manually). "
        "When drafting a REPLY to an existing email, always pass `reply_to_message_id` "
        "(the original message's ID) so the draft is threaded under the original "
        "conversation in Gmail, with proper In-Reply-To/References headers and a "
        "`Re:` subject."
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
                    "create_draft",
                ],
                "description": "The action to perform.",
            },
            "to": {
                "type": "string",
                "description": (
                    "Recipient address(es) for create_draft. "
                    "Comma-separated for multiple recipients."
                ),
            },
            "cc": {
                "type": "string",
                "description": "CC recipients (comma-separated). Optional.",
            },
            "bcc": {
                "type": "string",
                "description": "BCC recipients (comma-separated). Optional.",
            },
            "subject": {
                "type": "string",
                "description": "Subject line for create_draft.",
            },
            "body": {
                "type": "string",
                "description": "Message body (plain text) for create_draft.",
            },
            "html_body": {
                "type": "string",
                "description": "Optional HTML body. When set, saved as an HTML alternative.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Gmail search query (for search action). Supports all Gmail "
                    "search operators: from:, to:, subject:, has:attachment, "
                    "after:2026/01/01, before:, is:unread, label:, etc. "
                    "IMPORTANT: searches are auto-scoped to the INBOX Primary tab "
                    "(`in:inbox category:primary` is prepended) unless the query "
                    "already contains an `in:`, `label:`, or `category:` operator, "
                    "or the user explicitly asks you to search all mail / archive / "
                    "spam / trash / another category."
                ),
            },
            "message_id": {
                "type": "string",
                "description": "Message ID (for read_message action).",
            },
            "reply_to_message_id": {
                "type": "string",
                "description": (
                    "For create_draft: the Gmail message ID you are replying to. "
                    "When set, the draft is created inside the same thread and "
                    "appears nested under the original email in Gmail. The tool "
                    "automatically pulls the original sender, subject (with a "
                    "`Re:` prefix), Message-ID, and References headers — you do "
                    "NOT need to set `to` or `subject` yourself unless you want "
                    "to override them."
                ),
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
                    "Defaults to INBOX (auto-narrowed to the Primary category). "
                    "Only override when the user explicitly asks for a different folder."
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
            "group_by_thread": {
                "type": "boolean",
                "description": (
                    "For list_messages and search: when true (DEFAULT), results "
                    "are grouped by conversation — one entry per thread showing "
                    "the newest activity, unread count, and participants. Set to "
                    "false only if the user explicitly asks for every individual "
                    "message. This is how 'how many new emails do I have' should "
                    "be answered: count threads, not raw messages."
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

        compose_actions = {"create_draft"}
        need_compose = action in compose_actions

        try:
            token = await self._get_access_token(need_compose=need_compose)
        except RuntimeError as e:
            return ToolResult(success=False, error=str(e))

        handlers = {
            "list_messages": self._action_list_messages,
            "search": self._action_search,
            "read_message": self._action_read_message,
            "get_thread": self._action_get_thread,
            "list_labels": self._action_list_labels,
            "create_draft": self._action_create_draft,
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

    async def _get_access_token(self, need_compose: bool = False) -> str:
        """Retrieve a valid Google OAuth access token.

        When *need_compose* is True, verifies that a draft-capable scope
        (``gmail.compose`` or ``gmail.modify``) was granted. Otherwise a
        read scope (``gmail.readonly`` or ``gmail.modify``) is required.
        """
        from captain_claw.google_oauth_manager import GoogleOAuthManager
        from captain_claw.session import get_session_manager

        mgr = GoogleOAuthManager(get_session_manager())
        tokens = await mgr.get_tokens()
        if not tokens:
            raise RuntimeError(
                "Google account is not connected. "
                "Please connect via the Flight Deck Connections page."
            )

        granted = set(tokens.scope.split()) if tokens.scope else set()

        if need_compose:
            if not any(s in granted for s in _GMAIL_COMPOSE_SCOPES):
                raise RuntimeError(
                    "Gmail compose scope not granted. Your current OAuth "
                    "connection doesn't allow draft creation. Please "
                    "disconnect and reconnect your Google account."
                )
        else:
            if not any(s in granted for s in _GMAIL_READ_SCOPES):
                raise RuntimeError(
                    "Gmail read scope not granted. Your current OAuth connection "
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
        group_by_thread: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """List recent conversations (threads) from a label.

        By default this action is **thread-centric**: it returns one
        entry per conversation (newest activity first) so a thread with
        four unread replies counts as a single item, not four. Set
        ``group_by_thread=False`` to get individual messages instead.

        When *label* is left at its default (``INBOX``) the listing is
        further narrowed to the Primary category so Promotions / Social /
        Updates / Forums clutter is excluded.
        """
        limit = min(int(max_results or 10), 50)

        label_norm = (label or "INBOX").strip().upper() or "INBOX"
        default_primary = label_norm == "INBOX"

        params: dict[str, Any] = {"maxResults": limit}
        if default_primary:
            params["q"] = self._DEFAULT_SCOPE  # in:inbox category:primary
        else:
            params["labelIds"] = label_norm

        where_label = "INBOX (Primary)" if default_primary else label_norm

        if group_by_thread:
            # Use the threads endpoint so one conversation == one item,
            # regardless of how many replies are unread.
            resp = await self._client.get(
                f"{_GMAIL_API}/users/me/threads",
                params=params,
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            data = resp.json()
            thread_stubs = data.get("threads", [])
            if not thread_stubs:
                return ToolResult(
                    success=True,
                    content=f"No threads found in {where_label}.",
                )

            thread_summaries = await self._fetch_thread_summaries(
                token,
                [t["id"] for t in thread_stubs],
                include_body=include_body,
            )

            lines = [
                f"Threads in {where_label} ({len(thread_summaries)} shown, "
                f"grouped by conversation):\n"
            ]
            for ts in thread_summaries:
                lines.append(self._format_thread_summary(ts, include_body=include_body))
            lines.append(_FOLLOWUP_HINT)
            return ToolResult(success=True, content="\n".join(lines))

        # Flat message listing (group_by_thread=False)
        resp = await self._client.get(
            f"{_GMAIL_API}/users/me/messages",
            params=params,
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        message_stubs = data.get("messages", [])
        if not message_stubs:
            return ToolResult(success=True, content=f"No messages found in {where_label}.")

        messages = await self._fetch_message_batch(
            token, [m["id"] for m in message_stubs], include_body=include_body,
        )

        lines = [f"Messages in {where_label} ({len(messages)} shown, flat):\n"]
        for msg in messages:
            lines.append(self._format_message_summary(msg, include_body=include_body))

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: search
    # ------------------------------------------------------------------

    # Operators that already express a folder/container scope. If any of
    # these are present in the query, we leave it alone — otherwise we
    # auto-prepend ``in:inbox`` so searches don't silently pull archived
    # / spam / trashed mail the user never asked for.
    _SCOPE_OPERATORS = ("in:", "label:", "category:")

    _DEFAULT_SCOPE = "in:inbox category:primary"

    @classmethod
    def _scope_query_to_inbox(cls, query: str) -> str:
        """Prepend the default inbox/primary scope unless *query* already
        contains an explicit scope operator (``in:``, ``label:``, or
        ``category:``)."""
        q = query.strip()
        if not q:
            return cls._DEFAULT_SCOPE
        q_lower = q.lower()
        for op in cls._SCOPE_OPERATORS:
            if op in q_lower:
                return q
        return f"{cls._DEFAULT_SCOPE} {q}"

    async def _action_search(
        self,
        token: str,
        query: str = "",
        max_results: int | float | None = None,
        include_body: bool = False,
        group_by_thread: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Search conversations (or messages) using Gmail search syntax."""
        if not query:
            return ToolResult(success=False, error="Search query is required.")

        effective_query = self._scope_query_to_inbox(query)
        limit = min(int(max_results or 10), 50)

        scope_note = ""
        if effective_query != query:
            scope_note = (
                " (auto-scoped to INBOX Primary; pass an explicit in:/label:/"
                "category: operator to broaden)"
            )

        if group_by_thread:
            resp = await self._client.get(
                f"{_GMAIL_API}/users/me/threads",
                params={"q": effective_query, "maxResults": limit},
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            data = resp.json()
            thread_stubs = data.get("threads", [])
            if not thread_stubs:
                return ToolResult(
                    success=True,
                    content=f"No threads found for: {effective_query}{scope_note}",
                )

            thread_summaries = await self._fetch_thread_summaries(
                token,
                [t["id"] for t in thread_stubs],
                include_body=include_body,
            )
            lines = [
                f"Search results for '{effective_query}'{scope_note} "
                f"({len(thread_summaries)} threads):\n"
            ]
            for ts in thread_summaries:
                lines.append(self._format_thread_summary(ts, include_body=include_body))
            lines.append(_FOLLOWUP_HINT)
            return ToolResult(success=True, content="\n".join(lines))

        # Flat message search
        resp = await self._client.get(
            f"{_GMAIL_API}/users/me/messages",
            params={"q": effective_query, "maxResults": limit},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()

        message_stubs = data.get("messages", [])
        if not message_stubs:
            return ToolResult(
                success=True,
                content=f"No messages found for: {effective_query}{scope_note}",
            )

        messages = await self._fetch_message_batch(
            token, [m["id"] for m in message_stubs], include_body=include_body,
        )

        lines = [
            f"Search results for '{effective_query}'{scope_note} "
            f"({len(messages)} messages, flat):\n"
        ]
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
    # Action: create_draft
    # ------------------------------------------------------------------

    async def _action_create_draft(
        self,
        token: str,
        to: str = "",
        cc: str = "",
        bcc: str = "",
        subject: str = "",
        body: str = "",
        html_body: str = "",
        reply_to_message_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Save a draft message.

        When ``reply_to_message_id`` is provided, the draft is created
        inside the original message's thread with In-Reply-To /
        References headers set, so Gmail nests it under the conversation.
        """
        thread_id: str = ""
        in_reply_to: str = ""
        references: str = ""

        if reply_to_message_id:
            # Pull the headers we need to thread the reply correctly.
            try:
                meta_resp = await self._client.get(
                    f"{_GMAIL_API}/users/me/messages/{reply_to_message_id}",
                    params={
                        "format": "metadata",
                        "metadataHeaders": [
                            "From", "To", "Cc", "Subject",
                            "Message-ID", "References", "Reply-To",
                        ],
                    },
                    headers=self._auth_headers(token),
                )
                meta_resp.raise_for_status()
                orig = meta_resp.json()
            except httpx.HTTPStatusError as exc:
                return self._handle_http_error(exc)
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Failed to load reply_to_message_id={reply_to_message_id}: {exc}",
                )

            thread_id = orig.get("threadId", "") or ""
            orig_headers: dict[str, str] = {}
            for h in orig.get("payload", {}).get("headers", []):
                name = (h.get("name") or "").lower()
                orig_headers[name] = h.get("value", "") or ""

            orig_msg_id = orig_headers.get("message-id", "")
            orig_refs = orig_headers.get("references", "")
            orig_subject = orig_headers.get("subject", "")
            orig_from = orig_headers.get("reply-to") or orig_headers.get("from", "")

            if orig_msg_id:
                in_reply_to = orig_msg_id
                references = (orig_refs + " " + orig_msg_id).strip() if orig_refs else orig_msg_id

            # Default the recipient to the original sender when caller
            # didn't override it.
            if not to and orig_from:
                to = orig_from

            # Default subject to `Re: <original>` unless caller set one.
            if not subject and orig_subject:
                if orig_subject.lower().startswith("re:"):
                    subject = orig_subject
                else:
                    subject = f"Re: {orig_subject}"

        if not reply_to_message_id and not to and not subject and not body:
            return ToolResult(
                success=False,
                error="create_draft requires at least one of: to, subject, body (or reply_to_message_id).",
            )

        raw = self._build_raw_message(
            to=to, cc=cc, bcc=bcc, subject=subject,
            body=body, html_body=html_body,
            in_reply_to=in_reply_to, references=references,
        )
        draft_message: dict[str, Any] = {"raw": raw}
        if thread_id:
            draft_message["threadId"] = thread_id

        resp = await self._client.post(
            f"{_GMAIL_API}/users/me/drafts",
            headers={**self._auth_headers(token), "Content-Type": "application/json"},
            json={"message": draft_message},
        )
        resp.raise_for_status()
        data = resp.json()
        threaded_note = ""
        if thread_id:
            threaded_note = f"\n  Threaded under: {thread_id} (reply to {reply_to_message_id})"
        return ToolResult(
            success=True,
            content=(
                f"Draft created.{threaded_note}\n"
                f"  Draft ID: {data.get('id', '?')}\n"
                f"  Message ID: {data.get('message', {}).get('id', '?')}"
            ),
        )

    # ------------------------------------------------------------------
    # Raw RFC-822 message builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_raw_message(
        to: str = "",
        cc: str = "",
        bcc: str = "",
        subject: str = "",
        body: str = "",
        html_body: str = "",
        in_reply_to: str = "",
        references: str = "",
    ) -> str:
        """Build a base64url-encoded RFC-822 message for the Gmail API.

        Construct with the SMTP policy from the start so every header
        fold and every line ending is RFC 5322-compliant CRLF. Building
        with the default compat32 policy and only applying SMTP at
        serialization time has been observed to produce drafts whose
        body Gmail's web UI fails to render in threaded reply drafts.
        """
        msg = EmailMessage(policy=email.policy.SMTP)
        if to:
            msg["To"] = to
        if cc:
            msg["Cc"] = cc
        if bcc:
            msg["Bcc"] = bcc
        if subject:
            msg["Subject"] = subject
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
        if references:
            msg["References"] = references

        plain = body or ""

        # Always attach an HTML alternative. Gmail's web compose UI is
        # HTML-first — for threaded reply drafts specifically, it only
        # reliably renders the ``text/html`` part; drafts with just a
        # ``text/plain`` part have been observed to show an empty body
        # in the Gmail UI even though the raw message has the text.
        effective_html = html_body
        if not effective_html:
            effective_html = GoogleMailTool._text_to_html(plain)
        if not plain and html_body:
            plain = GoogleMailTool._html_to_text(html_body)

        msg.set_content(plain or " ")
        msg.add_alternative(effective_html, subtype="html")

        raw_bytes = msg.as_bytes()
        return base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")

    @staticmethod
    def _text_to_html(text: str) -> str:
        """Convert a plain text body to a simple HTML equivalent.

        Newlines become ``<br>`` and characters are HTML-escaped. This
        is intentionally minimal — Gmail will reformat it when the user
        opens the draft anyway, we just need a non-empty HTML part so
        the compose UI renders the body correctly.
        """
        if not text:
            return "<div></div>"
        escaped = html_module.escape(text)
        return "<div>" + escaped.replace("\n", "<br>") + "</div>"

    # ------------------------------------------------------------------
    # Message fetching helpers
    # ------------------------------------------------------------------

    async def _fetch_thread_summaries(
        self, token: str, thread_ids: list[str], include_body: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch one summary dict per thread.

        Each summary includes the thread id, total message count, unread
        count, the newest message (headers + snippet/body), and the list
        of unique participants seen across the thread.
        """
        summaries: list[dict[str, Any]] = []
        for thread_id in thread_ids:
            try:
                # Thread endpoint doesn't accept metadataHeaders, so use
                # 'metadata' for list views (snippets + headers) or
                # 'full' when include_body=True.
                fmt = "full" if include_body else "metadata"
                resp = await self._client.get(
                    f"{_GMAIL_API}/users/me/threads/{thread_id}",
                    params={"format": fmt},
                    headers=self._auth_headers(token),
                )
                resp.raise_for_status()
                raw = resp.json()
                msgs = raw.get("messages", [])
                if not msgs:
                    summaries.append({"thread_id": thread_id, "error": "empty thread"})
                    continue

                parsed_msgs = [self._parse_message(m) for m in msgs]
                # Sort by internalDate if available, newest first.
                def _sort_key(m: dict[str, Any]) -> int:
                    try:
                        return int(m.get("_internal_date") or 0)
                    except Exception:
                        return 0
                parsed_msgs.sort(key=_sort_key, reverse=True)

                newest = parsed_msgs[0]
                unread_count = sum(1 for m in parsed_msgs if m.get("is_unread"))
                participants: list[str] = []
                seen: set[str] = set()
                for m in parsed_msgs:
                    addr = m.get("from", "")
                    if addr and addr not in seen:
                        seen.add(addr)
                        participants.append(addr)

                summaries.append({
                    "thread_id": thread_id,
                    "message_count": len(parsed_msgs),
                    "unread_count": unread_count,
                    "participants": participants,
                    "newest": newest,
                    "is_unread": unread_count > 0,
                })
            except Exception as exc:
                log.debug("Failed to fetch thread %s: %s", thread_id, exc)
                summaries.append({"thread_id": thread_id, "error": str(exc)})

        return summaries

    async def _fetch_message_batch(
        self, token: str, message_ids: list[str], include_body: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch metadata (or full) for a batch of messages."""
        fmt = "full" if include_body else "metadata"
        # Gmail's ``metadataHeaders`` is a *repeated* query parameter —
        # it must be sent as multiple ``metadataHeaders=From&
        # metadataHeaders=To&...`` pairs, NOT a single comma-separated
        # value. httpx handles this automatically when we pass a list.
        meta_headers = ["From", "To", "Cc", "Subject", "Date", "Reply-To"]

        messages: list[dict[str, Any]] = []
        for msg_id in message_ids:
            try:
                params: dict[str, Any] = {"format": fmt}
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
            "_internal_date": internal_date,
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

    @classmethod
    def _format_thread_summary(
        cls, ts: dict[str, Any], include_body: bool = False,
    ) -> str:
        """Format a thread summary entry for list output."""
        if "error" in ts:
            return f"  [error] thread {ts.get('thread_id', '?')}: {ts['error']}"

        newest = ts.get("newest", {})
        msg_count = ts.get("message_count", 1)
        unread_count = ts.get("unread_count", 0)
        participants = ts.get("participants", [])

        # Thread badges
        unread_badge = f" [NEW ×{unread_count}]" if unread_count else ""
        if unread_count and msg_count > 1:
            unread_badge = f" [{unread_count} NEW / {msg_count} msgs]"
        elif msg_count > 1:
            unread_badge += f" [{msg_count} msgs]"
        starred = " ★" if newest.get("is_starred") else ""

        att_count = len(newest.get("attachments", []))
        att_str = f"  📎{att_count}" if att_count else ""

        # Participants (first 3 unique senders, shortened)
        parts_short: list[str] = []
        for addr in participants[:3]:
            p = email.utils.parseaddr(addr)
            parts_short.append(p[0] or p[1] or addr)
        if len(participants) > 3:
            parts_short.append(f"+{len(participants) - 3}")
        parts_str = ", ".join(parts_short) if parts_short else "?"

        # Newest date
        date_str = newest.get("date", "")
        try:
            parsed_date = email.utils.parsedate_to_datetime(date_str)
            date_short = parsed_date.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_short = date_str[:20] if date_str else ""

        icon = "📩" if unread_count else "📧"
        lines = [
            f"  {icon} {newest.get('subject', '(no subject)')}{unread_badge}{starred}{att_str}",
            f"    Participants: {parts_str}  |  Latest: {date_short}",
            f"    Thread: {ts.get('thread_id', '')}  |  Newest msg: {newest.get('id', '')}",
        ]

        if include_body and newest.get("body"):
            preview = newest["body"][:300]
            if len(newest["body"]) > 300:
                preview += "…"
            lines.append(f"    Preview: {preview}")
        elif newest.get("snippet"):
            lines.append(f"    Preview: {newest['snippet']}")

        return "\n".join(lines)

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
