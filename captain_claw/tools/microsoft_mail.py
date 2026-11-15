"""Microsoft (Outlook / Microsoft 365) Mail tool — read, search, draft and organize.

The Outlook twin of ``google_mail``, through Microsoft Graph with the owner's
Microsoft connection (:mod:`captain_claw.microsoft_oauth_manager`):

* Read: list_messages, search (Outlook KQL), read_message, get_thread (an
  Outlook conversation), list_folders (``folder`` lists one's subfolders),
  get_attachment (saved under ``saved/downloads/<session>/mail-<id>/`` with
  the tool that reads it).
* Write: create_draft (new or a reply that stays in its conversation),
  update_draft, list_drafts — both write actions take ``attachments`` (local
  files, checked by :mod:`captain_claw.mail_attachments` before the draft is
  touched; under 3 MB in one request, bigger through a Graph upload session),
  and update_draft ``remove_attachments``. **No sending** — the agent never sends Outlook
  mail; the user reviews the draft and sends it from Outlook. (Gmail sends go
  through Flight Deck's per-user policy, audit and limits; Outlook has no such
  gate yet, so no send action exists.)
* Every draft create_draft writes, and update_draft's new body, ends with
  the owner's agentic signature (Connections → Microsoft — see
  :mod:`captain_claw.mail_signature`).
* Mail writes in automated turns go through :mod:`captain_claw.mail_authority`
  exactly like google_mail's, and create_draft refuses a repeat (same
  recipient + subject already in Drafts, or in Sent Items within
  ``tools.microsoft_mail.repeat_check_days``; for a reply, a draft or newer
  reply in the conversation) unless ``allow_repeat`` — the refusal carries
  :data:`gmail_compose.REPEAT_MARK` so the loop and Flight Deck treat it as
  "already done".
* Organize: archive (to the mailbox's Archive folder), move (to a folder by
  name, 'Parent/Child' path or ID — never Deleted Items, Junk, a folder inside
  either, Drafts, Sent Items or Outbox), create_folder, add_categories /
  remove_categories. Nothing is written, sent or deleted. Targets are
  thread_ids (a conversation's messages, minus those in Sent Items / Drafts /
  Deleted Items / Junk / Outbox for a move) and/or message_ids, at most 100
  per call. A move gives each message a NEW id — the answer prints old → new.
  An automated turn may organize only when its own text asks to
  (:func:`mail_authority.check_mailbox_change`, before any token fetch).

Scopes: ``Mail.Read`` to read, ``Mail.ReadWrite`` to draft and to organize.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx

from captain_claw import gmail_compose, mail_attachments, mail_authority, mail_signature
from captain_claw.config import get_config
from captain_claw.logging import get_logger
from captain_claw.microsoft_graph import (
    GRAPH_API,
    MAIL_READ,
    MAIL_WRITE,
    flag,
    graph_error,
    graph_message,
    graph_token,
    limit,
    odata_string,
    pop_runtime,
)
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_WRITE_ACTIONS = frozenset({"create_draft", "update_draft"})
# Organizing the mailbox — moves, folders and categories, never an email
# write. Gated by mail_authority.check_mailbox_change; needs Mail.ReadWrite.
_ORGANIZE_ACTIONS = frozenset({"archive", "move", "create_folder", "add_categories", "remove_categories"})

# One organize call names at most this many emails (thread_ids + message_ids)…
_MAX_ORGANIZE_IDS = 100
# … and changes at most this many messages once conversations are expanded.
_MAX_ORGANIZE_MESSAGES = 500
# Concurrent Graph requests while organizing (Outlook allows 4 per mailbox).
_ORGANIZE_CONCURRENCY = 4
# An organize answer lists this many IDs, then "… and N more".
_MAX_LISTED_IDS = 20
# Well-known folders a move never targets (deleting / junking is out of scope)
# and whose messages a conversation's move leaves where they are.
_STAY_PUT = ("drafts", "sentitems", "deleteditems", "junkemail", "outbox")
# … and the two whose subfolders are just as out of bounds: mail filed inside
# Deleted Items / Junk Email goes when they are emptied.
_BINS = ("deleteditems", "junkemail")
# How far up a folder's parents a move looks for one of _BINS.
_MAX_FOLDER_DEPTH = 32
# The well-known folders a move — or a check that a cached folder wasn't
# deleted since — reads (once per token): _STAY_PUT, the Inbox (for the undo)
# and the mailbox root (where a walk up the parents ends).
_MOVE_KNOWN = (*_STAY_PUT, "inbox", "msgfolderroot")
_FOLDER_SELECT = "id,displayName,parentFolderId"
_FOLDER_LABELS = {
    "inbox": "Inbox", "drafts": "Drafts", "sentitems": "Sent Items", "deleteditems": "Deleted Items",
    "junkemail": "Junk Email", "outbox": "Outbox", "archive": "Archive",
}

_MAX_BODY_CHARS = 30_000
_MAX_ATTACHMENT_BYTES = 30 * 1024 * 1024
_MAX_ATTACHMENTS_PER_CALL = 10
# Attaching: Graph takes a file "under 3 MB" in one POST (base64 in a request
# capped near 4 MB) and refuses an upload session for one that small; bigger
# files go through a session in chunks under 4 MB (a multiple of 320 KiB).
# Which MB Graph means is undocumented: at 3 MB (decimal) and up a session is
# tried, and one refused as too small falls back to the POST.
_SMALL_ATTACHMENT_BYTES = 3_000_000
_UPLOAD_CHUNK_BYTES = 10 * 320 * 1024
_SESSION_TOO_SMALL = "ErrorAttachmentSizeShouldNotBeLessThanMinimumSize"

_MSG_SELECT = (
    "id,conversationId,subject,from,toRecipients,ccRecipients,receivedDateTime,"
    "sentDateTime,isRead,isDraft,hasAttachments,bodyPreview,webLink"
)
_FULL_SELECT = _MSG_SELECT + ",bccRecipients,replyTo,body,lastModifiedDateTime"

# Friendly folder names → Graph well-known folder names.
_WELL_KNOWN = {
    "inbox": "inbox", "drafts": "drafts", "draft": "drafts", "sent": "sentitems",
    "sentitems": "sentitems", "sent items": "sentitems", "deleted": "deleteditems",
    "deleteditems": "deleteditems", "deleted items": "deleteditems", "trash": "deleteditems",
    "junk": "junkemail", "junkemail": "junkemail", "spam": "junkemail", "archive": "archive",
    "outbox": "outbox",
}

_FOLLOWUP_HINT = (
    "\nNext steps — use microsoft_mail actions only (never filesystem tools):\n"
    "  • Read full email body: microsoft_mail action=read_message message_id=<Newest msg ID above>\n"
    "  • Read whole conversation: microsoft_mail action=get_thread thread_id=<Conversation ID above>\n"
    "  • Reply to one — ONLY if the user asked you to reply: first check get_thread / "
    "list_drafts for a reply already drafted or sent (never make a second one), then "
    "microsoft_mail action=create_draft reply_to_message_id=<Newest msg ID above> body=...\n"
    "  • Narrow the list: microsoft_mail action=search query='from:alice subject:report'"
)
_FOLLOWUP_HINT_NO_REPLY = (
    "\nNext steps — use microsoft_mail actions only (never filesystem tools):\n"
    "  • Read full email body: microsoft_mail action=read_message message_id=<Newest msg ID above>\n"
    "  • Read whole conversation: microsoft_mail action=get_thread thread_id=<Conversation ID above>\n"
    "  • Narrow the list: microsoft_mail action=search query='from:alice subject:report'"
)
# Appended when this turn may change the mailbox (a human turn, or a job whose
# own text asks to organize email).
_FOLLOWUP_ORGANIZE = (
    "\n  • Archive / move / categorize — ONLY if the user asked: microsoft_mail action=archive "
    "thread_ids=[<Conversation ID above>] (also move, create_folder, add_categories, remove_categories)"
)


def _followup_hint() -> str:
    if mail_authority.check_mail_write("microsoft_mail", "create_draft") is None:
        hint = _FOLLOWUP_HINT
    else:
        hint = _FOLLOWUP_HINT_NO_REPLY
    if mail_authority.check_mailbox_change("microsoft_mail", "archive") is None:
        hint += _FOLLOWUP_ORGANIZE
    return hint


def _seg(value: str) -> str:
    return quote(str(value).strip(), safe="=-_.!~")


def _ids(*values: Any) -> list[str]:
    """IDs from *values* — lists, or strings holding a JSON list or comma /
    whitespace separated IDs — de-duplicated, in order."""
    out: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        for item in value if isinstance(value, (list, tuple)) else [value]:
            for tok in re.split(r"[\s,]+", "" if item is None else str(item)):
                tok = tok.strip("[]'\"")
                if tok and tok not in out:
                    out.append(tok)
    return out


def _names(value: Any) -> list[str]:
    """Category names — a list, or a string holding one (see
    :func:`_name_list`) or comma-separated names (they may have spaces) —
    de-duplicated in any case."""
    if isinstance(value, str):
        value = _name_list(value.strip())
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        name = "" if item is None else str(item).strip()
        if name and name.casefold() not in {o.casefold() for o in out}:
            out.append(name)
    return out


def _name_list(text: str) -> list[Any]:
    """The names in *text*. Only a string that is one list as a whole loses
    its brackets — a JSON list, a ['Python', 'style'] one, or bracketed
    comma-separated names ('[Receipts, "Red category"]'). Anything else is
    split on commas, so bracketed names keep their brackets ('[EXT]',
    '[A], [B]'). A quoted name loses its own pair of quotes."""
    if text.startswith("[") and text.endswith("]"):
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(text)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
            if isinstance(parsed, list):
                return parsed
        inner = text[1:-1]
        parts = inner.split(",")
        quoted = any(_unquoted(p) != p.strip() for p in parts)
        if "[" not in inner and "]" not in inner and (len(parts) > 1 or quoted):
            return [_unquoted(p) for p in parts]
    return [_unquoted(p) for p in text.split(",")]


def _unquoted(text: str) -> str:
    """*text* stripped, without one pair of matching quotes around it."""
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def _folder_parts(name: Any) -> list[str]:
    """A folder name or 'Parent/Child' path as its parts; [] when a part is empty."""
    parts = [p.strip() for p in str(name or "").strip().strip("/").split("/")]
    return parts if all(parts) else []


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _listed(ids: list[str]) -> str:
    more = f", … and {len(ids) - _MAX_LISTED_IDS} more" if len(ids) > _MAX_LISTED_IDS else ""
    return ", ".join(ids[:_MAX_LISTED_IDS]) + more


def _recipients(field: str) -> list[dict[str, Any]]:
    """A To / Cc / Bcc string as Graph recipients (display names kept)."""
    import email.utils

    out: list[dict[str, Any]] = []
    for name, addr in email.utils.getaddresses([field or ""]):
        addr = addr.strip()
        if addr:
            entry: dict[str, Any] = {"address": addr}
            if name.strip():
                entry["name"] = name.strip()
            out.append({"emailAddress": entry})
    return out


def _addr_list(recips: list[dict[str, Any]] | None) -> list[str]:
    return [str((r.get("emailAddress") or {}).get("address") or "").lower()
            for r in recips or [] if (r.get("emailAddress") or {}).get("address")]


def _who(recip: dict[str, Any] | None) -> str:
    em = (recip or {}).get("emailAddress") or {}
    name, addr = em.get("name") or "", em.get("address") or ""
    return f"{name} <{addr}>" if name and name != addr else (addr or name or "?")


def _who_list(recips: list[dict[str, Any]] | None) -> str:
    return ", ".join(_who(r) for r in recips or []) or "(none)"


def _stamp(value: str | None) -> str:
    return str(value or "")[:16].replace("T", " ")


class _FolderPlaceError(Exception):
    """Where a folder sits (inside Deleted Items / Junk Email or not) can't be told."""


class MicrosoftMailTool(Tool):
    """Read Outlook mail, create drafts and organize it (never sends)."""

    name = "microsoft_mail"
    description = (
        "Outlook / Microsoft 365 mail — read, search, create drafts and organize (archive / move / "
        "folders / categories). It NEVER sends: drafts are "
        "saved in the user's Outlook Drafts folder for them to review and send; if the user asks you "
        "to send, create the draft and tell them it's waiting in Outlook Drafts. "
        "WRITE ONLY WHEN ASKED: create_draft / update_draft only when the user asked for that email "
        "in this conversation (or a scheduled job's own text explicitly says to draft it). Reading, "
        "listing, summarizing or triaging mail NEVER implies drafting replies — say who is waiting "
        "and offer to draft. If this tool answers '[not-authorized: mail-write]', don't retry: tell "
        "the user. When the user asks for emails, call create_draft once per recipient they named — "
        "do NOT output email text for them to copy. "
        "NO REPEATS: before create_draft check whether the email already exists — list_drafts with "
        "query='to:<recipient>', and for a reply get_thread on its conversation. If the same email "
        "is already drafted or sent, don't create another: report it (update_draft revises a draft). "
        "create_draft refuses a repeat itself ('Not created — repeat of …'): that is not a failure, "
        "don't retry; pass allow_repeat=true only if the user explicitly wants another copy. "
        "A new edition of a recurring email (a scheduled or weekly report) is not a repeat: give "
        "it its date in the subject (e.g. 'Weekly report — <today's date>') so it never repeats "
        "the last one. "
        "Replies pass reply_to_message_id (reply_all=true for reply-all) so the draft stays in the "
        "conversation. ATTACHMENTS: read_message lists them numbered; get_attachment saves them and "
        "names the tool that reads each. ATTACHING FILES: create_draft / update_draft take "
        "attachments=[paths] — files on disk as the file tools name them (the path a tool printed, "
        "a saved/ path, or vfs:<project>/<path>), up to 25 MB per email; for bigger files share a "
        "OneDrive link in the body. update_draft keeps the draft's files and adds yours (a "
        "same-named file replaces the old one); remove_attachments=[names or numbers, or 'all'] "
        "drops some; with only file changes the body can be left out. A file is attached only when "
        "the tool's answer lists it under 'Attachments:'. "
        "SIGNATURE: the user's agentic signature (set in Flight Deck) is added to the end of "
        "every draft automatically — never write it into the body yourself. "
        "ORGANIZE ONLY WHEN ASKED: archive / move / categories / create_folder only when the user "
        "asked to organize those emails (or a scheduled job's own text says so). Reading, "
        "summarizing or triaging mail never implies it — suggest it instead. Find the emails "
        "first (list_messages / search), then pass their IDs (thread_ids = whole conversations); "
        "report what changed using the IDs and counts the tool printed. If it answers "
        "'[not-authorized: mail-write]', don't retry: tell the user. Moving an Outlook message "
        "changes its Message ID — use the new ID the tool prints. archive files emails in the "
        "mailbox's Archive folder; move folder=<name, 'Parent/Child' path or ID> files them there "
        "(never Deleted Items, Junk, Drafts, Sent Items or Outbox — a conversation's messages in "
        "those stay put); add_categories / remove_categories take categories=[names]; "
        "create_folder makes a folder only when the user asked for one ('Inbox/Receipts' makes it "
        "inside the Inbox). At most 100 emails per call. "
        "ROUTING: any request about an Outlook email, inbox, "
        "conversation, sender or subject MUST use microsoft_mail — never filesystem tools. "
        "Actions: list_messages (newest first; folder default Inbox; grouped by conversation), "
        "search (Outlook search: from:, to:, subject:, hasattachments:true, received>=2026-01-01, "
        "free text; folder default Inbox, folder='all' for every folder), read_message, "
        "get_thread (whole conversation), list_folders (folder=<name> lists its subfolders), "
        "create_draft, update_draft (replace a draft's body; unset to/cc/bcc/subject keep the "
        "draft's own), list_drafts, get_attachment, archive (to the Archive folder — thread_ids "
        "and/or message_ids), move (to folder=<destination>), create_folder (folder=<name or "
        "'Parent/Child'>), add_categories / remove_categories (categories=[...] on thread_ids / "
        "message_ids)."
    )
    # Uploading 25 MB of attachments can take a while.
    timeout_seconds = 300.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_messages", "search", "read_message", "get_thread", "list_folders",
                    "create_draft", "update_draft", "list_drafts", "get_attachment",
                    "archive", "move", "create_folder", "add_categories", "remove_categories",
                ],
                "description": "The action to perform.",
            },
            "folder": {
                "type": "string",
                "description": (
                    "list_messages / search: Inbox (default), Sent, Drafts, Archive, Deleted, Junk, a "
                    "folder name, 'Parent/Child' path or ID from list_folders; search also takes "
                    "'all'. list_folders: the folder whose subfolders to list (default: the top "
                    "level). move: the destination (not Deleted, Junk, Drafts, Sent or Outbox). "
                    "create_folder: the new folder's name — 'Parent/Child' (e.g. 'Inbox/Receipts') "
                    "makes it inside Parent, creating missing parents."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "search: Outlook search text (from:alice subject:report hasattachments:true "
                    "received>=2026-01-01 …). list_drafts: 'to:<address>' or words in the subject."
                ),
            },
            "unread_only": {"type": "boolean", "description": "list_messages: only unread mail."},
            "focused_only": {
                "type": "boolean",
                "description": "list_messages: only the Focused Inbox (when the user uses Focused Inbox).",
            },
            "message_id": {
                "type": "string",
                "description": (
                    "Message ID (read_message, get_attachment; for archive / move / add_categories "
                    "/ remove_categories, one email — see message_ids)."
                ),
            },
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "archive / move / add_categories / remove_categories: Message IDs of single "
                    "emails. Moving changes a message's ID — use the new one the tool prints. At "
                    "most 100 emails per call (thread_ids + message_ids)."
                ),
            },
            "thread_id": {
                "type": "string",
                "description": (
                    "Conversation ID (get_thread) — the 'Conversation:' ID list/search print; for "
                    "archive / move / add_categories / remove_categories, one conversation — see "
                    "thread_ids."
                ),
            },
            "thread_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "archive / move / add_categories / remove_categories: Conversation IDs — each "
                    "acts on the whole conversation (a move leaves its messages in Sent Items, "
                    "Drafts, Deleted Items, Junk and Outbox where they are). At most 100 per call "
                    "(thread_ids + message_ids)."
                ),
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "add_categories / remove_categories: Outlook category names, e.g. "
                    "['Receipts', 'Red category']. A name not in the user's category list still "
                    "applies (shown without a colour)."
                ),
            },
            "attachment": {
                "type": "string",
                "description": "get_attachment: number from read_message ('1', '2', …), name, or 'all'.",
            },
            "to": {"type": "string", "description": "Recipients for create_draft / update_draft (comma-separated)."},
            "cc": {"type": "string", "description": "CC recipients (comma-separated)."},
            "bcc": {"type": "string", "description": "BCC recipients (comma-separated)."},
            "subject": {"type": "string", "description": "Subject for create_draft / update_draft."},
            "body": {
                "type": "string",
                "description": (
                    "Plain-text body (create_draft; update_draft: the full new text, it replaces "
                    "the draft's). For a reply it goes above the quoted original."
                ),
            },
            "html_body": {"type": "string", "description": "Optional HTML body instead of body (new drafts)."},
            "reply_to_message_id": {
                "type": "string",
                "description": (
                    "create_draft: the message you are replying to. The draft lands in its "
                    "conversation with 'RE:' and the original quoted; recipients default to the "
                    "sender (or, for your own sent email, its recipients)."
                ),
            },
            "reply_all": {"type": "boolean", "description": "create_draft reply: include everyone (reply all)."},
            "draft_id": {"type": "string", "description": "update_draft: the Draft ID create_draft / list_drafts printed."},
            "attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "create_draft / update_draft: local files to attach — paths as the file tools "
                    "take them (the path a tool printed, a saved/ path, or vfs:<project>/<path>). "
                    "At most 20 files, 25 MB together."
                ),
            },
            "remove_attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "update_draft: the draft's attachments to drop — names or numbers from "
                    "read_message on the draft, or 'all'."
                ),
            },
            "allow_repeat": {
                "type": "boolean",
                "description": (
                    "create_draft: true ONLY when the user explicitly asked for another copy of an "
                    "email already drafted or sent. Default false. Not for the next edition of a "
                    "recurring report — date its subject instead."
                ),
            },
            "max_results": {"type": "number", "description": "Maximum results (default 10, max 50)."},
            "include_body": {"type": "boolean", "description": "list_messages / search: add each preview text."},
            "group_by_thread": {
                "type": "boolean",
                "description": (
                    "list_messages / search: group by conversation (default true) — count conversations, "
                    "not messages, for 'how many new emails'."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Microsoft Mail Tool)"},
        )
        self._own_cache: dict[str, set[str]] = {}
        self._folder_cache: dict[str, dict[str, str]] = {}
        self._known_cache: dict[str, dict[str, dict[str, Any] | None]] = {}

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        runtime = pop_runtime(kwargs)
        runtime["_workflow_run_dir"] = kwargs.pop("_workflow_run_dir", None)
        if action == "get_attachment":
            kwargs["_runtime"] = runtime
        handlers = {
            "list_messages": self._action_list_messages,
            "search": self._action_search,
            "read_message": self._action_read_message,
            "get_thread": self._action_get_thread,
            "list_folders": self._action_list_folders,
            "create_draft": self._action_create_draft,
            "update_draft": self._action_update_draft,
            "list_drafts": self._action_list_drafts,
            "get_attachment": self._action_get_attachment,
            "archive": self._action_archive,
            "move": self._action_move,
            "create_folder": self._action_create_folder,
            "add_categories": self._action_add_categories,
            "remove_categories": self._action_remove_categories,
        }
        if action in ("send", "send_draft", "reply", "forward"):
            return ToolResult(success=False, error=(
                "microsoft_mail doesn't send email. Create a draft with action=create_draft "
                "and tell the user it is waiting in their Outlook Drafts folder to send."
            ))
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use one of: {', '.join(handlers)}",
            )

        # Mail writes in an automated turn need the job's own explicit words
        # (or a human's approval) — checked before any token fetch.
        if mail_authority.is_mail_write("microsoft_mail", action):
            if mail_authority.needs_recipient_check():
                rcpt = (
                    None if action != "create_draft" or kwargs.get("reply_to_message_id")
                    else mail_authority.parse_recipients(kwargs.get("to"), kwargs.get("cc"), kwargs.get("bcc"))
                )
                refusal = mail_authority.check_mail_write(
                    "microsoft_mail", action, recipients=rcpt, own_addresses=await self._own_addresses(),
                )
            else:
                refusal = mail_authority.check_mail_write("microsoft_mail", action)
            if refusal:
                log.info("microsoft_mail write refused (automated turn)",
                         action=action, kind=mail_authority.current().kind)
                return ToolResult(success=False, error=mail_authority.for_turn(refusal))

        # Archive / move / categories / a new folder: an automated turn needs
        # its own text to ask to organize email — also before any token fetch.
        if mail_authority.is_mailbox_change(self.name, action):
            refusal = mail_authority.check_mailbox_change(self.name, action)
            if refusal:
                log.info("microsoft_mail mailbox change refused (automated turn)",
                         action=action, kind=mail_authority.current().kind)
                return ToolResult(success=False, error=mail_authority.for_turn(refusal))

        # Which emails to organize: checked before the token too.
        problem = self._take_targets(action, kwargs)
        if problem:
            return ToolResult(success=False, error=problem)

        # Files to attach: all checked and read before the draft is touched.
        problem = await self._take_attachments(action, kwargs, runtime)
        if problem:
            return ToolResult(success=False, error=problem)

        if action in _ORGANIZE_ACTIONS:
            need, what = MAIL_WRITE, "mail organizing access (Mail.ReadWrite)"
        elif action in _WRITE_ACTIONS:
            need, what = MAIL_WRITE, "mail draft access (Mail.ReadWrite)"
        else:
            need, what = MAIL_READ, "mail access (Mail.Read)"
        try:
            token, _granted = await graph_token(need, what)
        except RuntimeError as exc:
            return ToolResult(success=False, error=str(exc))
        try:
            return await handler(token, **kwargs)
        except httpx.HTTPStatusError as exc:
            return graph_error(exc, "Outlook mail", not_found="Message, draft or folder not found. Check the ID.")
        except httpx.HTTPError as exc:
            log.error("Outlook mail HTTP error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"HTTP error: {exc}")
        except Exception as exc:
            log.error("Outlook mail tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))

    async def _take_attachments(
        self, action: str, kwargs: dict[str, Any], runtime: dict[str, Any],
    ) -> str:
        """Swap ``attachments`` / ``remove_attachments`` in *kwargs* for the
        loaded files (``_attachments``) and the names to drop
        (``_remove_attachments``); the refusal, or ""."""
        nothing = "No draft was created." if action == "create_draft" else "Nothing was changed."
        raw = kwargs.pop("attachments", None)
        raw_remove = kwargs.pop("remove_attachments", None)
        if action not in _WRITE_ACTIONS:
            return ""
        try:
            values = mail_attachments.parse_list(raw)
            remove = mail_attachments.parse_list(raw_remove)
        except mail_attachments.AttachmentError as exc:
            return f"{exc} {nothing}"
        if remove and action != "update_draft":
            return f"remove_attachments works only with update_draft. {nothing}"
        kwargs["_remove_attachments"] = remove
        if values:
            try:
                kwargs["_attachments"] = await mail_attachments.load(
                    values, runtime, max_total_bytes=mail_attachments.OUTLOOK_MAX_TOTAL_BYTES,
                    limit_label="attachments are limited to 25 MB per email",
                )
            except mail_attachments.AttachmentError as exc:
                return f"{exc} {nothing}"
        return ""

    @staticmethod
    def _take_targets(action: str, kwargs: dict[str, Any]) -> str:
        """Swap thread_ids / thread_id / message_ids / message_id in *kwargs*
        for the de-duplicated ``_threads`` / ``_messages`` of an organize
        action; the refusal, or ""."""
        if action not in _ORGANIZE_ACTIONS or action == "create_folder":
            return ""
        threads = _ids(kwargs.pop("thread_ids", None), kwargs.pop("thread_id", None))
        messages = _ids(kwargs.pop("message_ids", None), kwargs.pop("message_id", None))
        if not threads and not messages:
            return (
                "Which emails? Pass thread_ids (whole conversations — the 'Conversation:' IDs) "
                "and/or message_ids; find them with list_messages or search first. Nothing was "
                "changed."
            )
        if len(threads) + len(messages) > _MAX_ORGANIZE_IDS:
            return (f"At most {_MAX_ORGANIZE_IDS} emails per call — call again for the rest. "
                    "Nothing was changed.")
        kwargs["_threads"], kwargs["_messages"] = threads, messages
        return ""

    # ── HTTP helpers ────────────────────────────────────────────────

    @staticmethod
    def _headers(token: str, *, text_body: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {token}"}
        if text_body:
            headers["Prefer"] = 'outlook.body-content-type="text"'
        return headers

    async def _get(self, token: str, path: str, params: dict[str, Any] | None = None,
                   *, text_body: bool = False) -> dict[str, Any]:
        url = path if path.startswith("https://") else f"{GRAPH_API}{path}"
        resp = await self._client.get(url, params=params, headers=self._headers(token, text_body=text_body))
        resp.raise_for_status()
        return resp.json()

    async def _collect(self, token: str, path: str, params: dict[str, Any], want: int,
                       *, text_body: bool = False) -> list[dict[str, Any]]:
        """Up to *want* items, following @odata.nextLink."""
        items: list[dict[str, Any]] = []
        url: str | None = path
        p: dict[str, Any] | None = params
        while url and len(items) < want:
            data = await self._get(token, url, p, text_body=text_body)
            p = None
            items.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
        return items[:want]

    async def _own_addresses(self) -> set[str]:
        """The mailbox's own addresses (mail + UPN); empty on any error."""
        try:
            token, _ = await graph_token(MAIL_READ, "mail access (Mail.Read)")
        except Exception:
            return set()
        if token in self._own_cache:
            return set(self._own_cache[token])
        try:
            me = await self._get(token, "/me", {"$select": "mail,userPrincipalName"})
        except Exception as exc:  # noqa: BLE001 — fail closed
            log.info("Outlook profile lookup failed", error=str(exc))
            return set()
        addrs = {str(me.get(k) or "").strip().lower() for k in ("mail", "userPrincipalName")} - {""}
        self._own_cache.clear()  # tokens rotate; keep the latest only
        self._own_cache[token] = addrs
        return set(addrs)

    @staticmethod
    def _token_cache(caches: dict[str, dict[str, Any]], token: str) -> dict[str, Any]:
        if token not in caches:
            caches.clear()  # tokens rotate; keep the latest only
            caches[token] = {}
        return caches[token]

    async def _folder_call(
        self, token: str, folder: str | None, call: Any, *, seen: dict[str, str] | None = None,
    ) -> Any:
        """``await call(ref)`` with the ref of the folder *folder* names (see
        :meth:`_folder_ref`; *seen* as for :meth:`_bin_of`). A ref only assumed
        that Graph rejects — a cached ID since deleted, a well-known folder this
        mailbox lacks (no Archive), an ID-shaped name that is really a long
        display name — is looked up by display name once more and the call
        retried; when that finds no other folder, Graph's error stands.
        LookupError when there is no such folder."""
        ref, how = await self._folder_ref(token, folder, seen=seen)
        try:
            return await call(ref)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            rejected = status == 404 or (status == 400 and how == "id")  # 400: no ID after all
            if how not in ("cache", "known", "id") or not rejected:
                raise
            try:
                again = (await self._folder_ref(token, folder, fresh=True))[0]
            except LookupError:
                again = ref
            if again == ref:
                raise
        return await call(again)

    async def _folder_ref(
        self, token: str, folder: str | None, *, fresh: bool = False,
        seen: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """``(ref, how)``: the ID — or Graph well-known name — of the folder
        *folder* names (Inbox when empty) and how it was found: "inbox", "cache"
        (a name resolved earlier under this token), "known" (a well-known name:
        Sent, Archive …), "id" (an ID-shaped name, taken as the folder ID) or
        "lookup" (a display name or 'Parent/Child' path, see :meth:`_find_folder`).
        *fresh*: only a lookup (the cached / assumed ref was just rejected). A
        cached folder deleted since — moved into Deleted Items, ID kept (or into
        Junk Email) — is looked up again, as a new session would (*seen* as for
        :meth:`_bin_of`). LookupError when there is no such folder."""
        name = str(folder or "").strip()
        if not name:
            return "inbox", "inbox"
        parts = _folder_parts(name)
        cache = self._token_cache(self._folder_cache, token)
        key = "/".join(parts).casefold() if parts else name.casefold()
        if fresh:
            cache.pop(key, None)
        elif key in cache:
            if not await self._binned_since(token, str(cache[key]), seen):
                return str(cache[key]), "cache"
            cache.pop(key, None)
        elif _WELL_KNOWN.get(name.lower()):
            return _WELL_KNOWN[name.lower()], "known"
        elif len(name) > 40 and " " not in name and "/" not in name:  # a folder ID
            return name, "id"
        found = await self._find_folder(token, name)
        if not found:
            raise LookupError(
                f"No mail folder named '{name}'. list_folders shows the folders (folder=<name> "
                "lists one's subfolders); name a subfolder as 'Parent/Child'."
            )
        cache[key] = found
        return found, "lookup"

    async def _find_folder(self, token: str, name: str) -> str | None:
        """The ID of the folder display name *name* names, or None. One name: a
        top-level folder, else one in the Inbox. A 'Parent/Child' path: walked
        down from its first part (:meth:`_walk`); one that leads nowhere is
        tried as a single name with '/' in it."""
        parts = _folder_parts(name)
        if len(parts) > 1:
            ids, _how = await self._walk(token, parts)
            if len(ids) == len(parts):
                return ids[-1]
        elif parts:
            name = parts[0]
        return (await self._child_id(token, None, name)
                or await self._child_id(token, "inbox", name))

    async def _walk(
        self, token: str, parts: list[str], *, known_ids: bool = False,
    ) -> tuple[list[str], str]:
        """The IDs along the path *parts* as far as its folders exist (all of
        them when the whole path does) and how its first part was found. That
        part is tried, until the whole path is found, as a single name is: a
        well-known folder ("known" — its name, or its ID with *known_ids*; one
        this mailbox lacks leads nowhere), a top-level folder ("top"), one in
        the Inbox ("inbox"); the deepest walk wins ("" when none exists)."""
        best: list[str] = []
        best_how = ""
        known = _WELL_KNOWN.get(parts[0].lower())
        for how in ("known", "top", "inbox"):
            if how != "known":
                root = await self._child_id(token, None if how == "top" else "inbox", parts[0])
            elif known and known_ids:
                info = (await self._known_folders(token, (known,)))[known]
                root = str(info["id"]) if info else None
            else:
                root = known
            if not root:
                continue
            ids = [root]
            for part in parts[1:]:
                child = await self._child_id(token, ids[-1], part)
                if not child:
                    break
                ids.append(child)
            if len(ids) > len(best):
                best, best_how = ids, how
            if len(ids) == len(parts):
                break
        return best, best_how

    async def _child_id(self, token: str, parent: str | None, name: str) -> str | None:
        """The ID of *parent*'s subfolder called *name* (a top-level folder when
        *parent* is None); None when there is none, or no such parent."""
        path = "/me/mailFolders" if parent is None else f"/me/mailFolders/{_seg(parent)}/childFolders"
        try:
            data = await self._get(token, path, {
                "$filter": f"displayName eq '{odata_string(name)}'", "$select": "id,displayName",
            })
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        hits = data.get("value", [])
        return str(hits[0]["id"]) if hits else None

    async def _known_folders(
        self, token: str, names: tuple[str, ...],
    ) -> dict[str, dict[str, Any] | None]:
        """Well-known folders' ``{id, displayName, parentFolderId}`` by name —
        None for one this mailbox doesn't have (Graph promises only Inbox,
        Drafts, Sent Items and Deleted Items). Cached per token; folder IDs
        never change."""
        cache = self._token_cache(self._known_cache, token)
        missing = [n for n in names if n not in cache]
        for name, info, exc in await self._each(missing, lambda n: self._folder_info(token, n)):
            if exc is not None:
                raise exc
            cache[name] = info
        return {n: cache[n] for n in names}

    async def _bin_of(
        self, token: str, folder_id: str, parent: str | None = None,
        *, seen: dict[str, str] | None = None,
    ) -> str:
        """"deleteditems" / "junkemail" when the folder *folder_id* is Deleted
        Items / Junk Email or sits inside one at any depth, else "". Follows
        parentFolderId up to the mailbox root (*parent*: the folder's own, just
        read), reading each folder on the way afresh — the user can delete
        (move) one in Outlook at any time — so only the well-known folders' IDs
        are cached per token. What one call learns (folder ID → its bin, or "")
        goes in *seen*: pass the same dict throughout one tool call, never
        across calls. :class:`_FolderPlaceError` when the walk can't finish (a
        parent gone while the mailbox root is known, a loop, deeper than
        :data:`_MAX_FOLDER_DEPTH`) — never taken to mean "not inside one"."""
        known = await self._known_folders(token, _MOVE_KNOWN)
        where = {} if seen is None else seen
        for kind, info in known.items():
            if info:
                where.setdefault(str(info["id"]), kind if kind in _BINS else "")
        root_known = bool(known.get("msgfolderroot"))
        chain: list[str] = []
        fid, up = folder_id, parent
        if up is not None:
            chain.append(fid)
            fid = up
        while fid and fid not in where:
            if fid in chain or len(chain) >= _MAX_FOLDER_DEPTH:
                raise _FolderPlaceError("its parent folders loop or go too deep")
            chain.append(fid)
            info = await self._folder_info(token, fid)
            if info is None:
                if root_known:
                    raise _FolderPlaceError("one of its parent folders is gone")
                break  # no root to end at: an unreadable parent is taken as the top
            fid = str(info.get("parentFolderId") or "")
        # The walk ended at a folder it knows, or at the top.
        kind = where.get(fid, "")
        for f in chain:
            where[f] = kind
        return kind

    async def _binned_since(self, token: str, folder_id: str, seen: dict[str, str] | None) -> bool:
        """Whether the cached folder *folder_id* now sits inside Deleted Items /
        Junk Email (deleting a folder in Outlook moves it there, ID kept) — not
        when it is one of them. When that can't be read, False: the call made
        with it tells."""
        try:
            known = await self._known_folders(token, _MOVE_KNOWN)
            if any(known[kind] and str(known[kind]["id"]) == folder_id for kind in _BINS):
                return False
            return bool(await self._bin_of(token, folder_id, seen=seen))
        except _FolderPlaceError:
            return False
        except httpx.HTTPError as exc:
            log.info("Outlook folder check failed", error=str(exc))
            return False

    @staticmethod
    async def _each(items: list[Any], fn: Any) -> list[tuple[Any, Any, httpx.HTTPError | None]]:
        """``(item, result, error)`` for ``await fn(item)`` on each item, in
        order, at most :data:`_ORGANIZE_CONCURRENCY` at a time."""
        gate = asyncio.Semaphore(_ORGANIZE_CONCURRENCY)

        async def run(item: Any) -> tuple[Any, Any, httpx.HTTPError | None]:
            async with gate:
                try:
                    return item, await fn(item), None
                except httpx.HTTPError as exc:
                    return item, None, exc

        return list(await asyncio.gather(*(run(i) for i in items)))

    # ── list / search ───────────────────────────────────────────────

    async def _action_list_messages(
        self, token: str, folder: str | None = None, max_results: Any = None,
        unread_only: Any = None, focused_only: Any = None, include_body: Any = None,
        group_by_thread: Any = None, **kwargs: Any,
    ) -> ToolResult:
        n = limit(max_results, 10, 50)
        grouped = flag(group_by_thread, True)
        # Graph wants the $orderby property first in $filter (else
        # InefficientFilter), hence the always-true receivedDateTime clause.
        filters = []
        if flag(unread_only):
            filters.append("isRead eq false")
        if flag(focused_only):
            filters.append("inferenceClassification eq 'focused'")
        params: dict[str, Any] = {
            "$select": _MSG_SELECT, "$orderby": "receivedDateTime desc",
            "$top": min(n * 4 if grouped else n, 100),
        }
        if filters:
            params["$filter"] = " and ".join(["receivedDateTime ge 1900-01-01T00:00:00Z", *filters])
        try:
            msgs = await self._folder_call(token, folder, lambda ref: self._collect(
                token, f"/me/mailFolders/{_seg(ref)}/messages", params, n * 4 if grouped else n))
        except LookupError as exc:
            return ToolResult(success=False, error=str(exc))
        label = str(folder or "Inbox")
        if not msgs:
            return ToolResult(success=True, content=f"No messages in {label}.")
        return ToolResult(success=True, content=self._render(
            msgs, n, grouped, flag(include_body), f"{label} — newest first",
        ) + _followup_hint())

    async def _action_search(
        self, token: str, query: str = "", folder: str | None = None, max_results: Any = None,
        include_body: Any = None, group_by_thread: Any = None, **kwargs: Any,
    ) -> ToolResult:
        q = str(query or "").strip()
        if not q:
            return ToolResult(success=False, error="query is required for search.")
        n = limit(max_results, 10, 50)
        grouped = flag(group_by_thread, True)
        escaped = q.replace("\\", "\\\\").replace('"', '\\"')
        params = {"$search": f'"{escaped}"', "$select": _MSG_SELECT, "$top": min(n * 3 if grouped else n, 100)}
        everywhere = str(folder or "").strip().lower() in ("all", "anywhere", "*")
        if everywhere:
            msgs = await self._collect(token, "/me/messages", params, n * 3 if grouped else n)
        else:
            try:
                msgs = await self._folder_call(token, folder, lambda ref: self._collect(
                    token, f"/me/mailFolders/{_seg(ref)}/messages", params, n * 3 if grouped else n))
            except LookupError as exc:
                return ToolResult(success=False, error=str(exc))
        if not msgs:
            where = "all folders" if everywhere else str(folder or "Inbox")
            return ToolResult(success=True, content=f"No messages found for '{q}' in {where}.")
        msgs.sort(key=lambda m: str(m.get("receivedDateTime") or ""), reverse=True)
        return ToolResult(success=True, content=self._render(
            msgs, n, grouped, flag(include_body), f"Search '{q}'",
        ) + _followup_hint())

    def _render(self, msgs: list[dict[str, Any]], n: int, grouped: bool,
                include_body: bool, title: str) -> str:
        if not grouped:
            shown = msgs[:n]
            lines = [f"{title} ({len(shown)} messages):\n"]
            lines.extend(self._format_summary(m, include_body) for m in shown)
            return "\n".join(lines)
        groups: dict[str, list[dict[str, Any]]] = {}
        for m in msgs:  # newest first, so each group's first is its newest
            groups.setdefault(str(m.get("conversationId") or m.get("id")), []).append(m)
        convs = list(groups.items())[:n]
        unread = sum(1 for _cid, ms in convs if any(not x.get("isRead") for x in ms))
        lines = [f"{title} ({len(convs)} conversations, {unread} with unread mail):\n"]
        for cid, ms in convs:
            newest = ms[0]
            people = []
            for x in ms:
                w = (x.get("from") or {}).get("emailAddress", {}).get("name") or \
                    (x.get("from") or {}).get("emailAddress", {}).get("address") or "?"
                if w not in people:
                    people.append(w)
            n_unread = sum(1 for x in ms if not x.get("isRead"))
            mark = "🔵 " if n_unread else ""
            lines.append(
                f"  {mark}{newest.get('subject') or '(no subject)'}"
                f"\n    From: {_who(newest.get('from'))}  |  {_stamp(newest.get('receivedDateTime'))}"
                + (f"  |  {len(ms)} msgs here, {n_unread} unread" if len(ms) > 1 or n_unread else "")
                + ("  |  📎" if any(x.get("hasAttachments") for x in ms) else "")
                + (f"\n    Participants: {', '.join(people[:6])}" if len(people) > 1 else "")
                + f"\n    Newest msg: {newest.get('id')}\n    Conversation: {cid}"
            )
            if include_body and newest.get("bodyPreview"):
                lines.append(f"    Preview: {str(newest['bodyPreview'])[:300]}")
        return "\n".join(lines)

    @staticmethod
    def _format_summary(m: dict[str, Any], include_body: bool) -> str:
        mark = "🔵 " if not m.get("isRead") else ""
        out = (
            f"  {mark}{m.get('subject') or '(no subject)'}"
            f"\n    From: {_who(m.get('from'))}  |  {_stamp(m.get('receivedDateTime'))}"
            + ("  |  📎" if m.get("hasAttachments") else "")
            + ("  |  DRAFT" if m.get("isDraft") else "")
            + f"\n    Message: {m.get('id')}\n    Conversation: {m.get('conversationId')}"
        )
        if include_body and m.get("bodyPreview"):
            out += f"\n    Preview: {str(m['bodyPreview'])[:300]}"
        return out

    # ── read ────────────────────────────────────────────────────────

    async def _attachments(self, token: str, message_id: str) -> list[dict[str, Any]]:
        data = await self._get(token, f"/me/messages/{_seg(message_id)}/attachments", {
            "$select": "id,name,contentType,size,isInline",
        })
        return data.get("value", [])

    async def _action_read_message(self, token: str, message_id: str = "", **kwargs: Any) -> ToolResult:
        if not message_id:
            return ToolResult(success=False, error="message_id is required.")
        m = await self._get(token, f"/me/messages/{_seg(message_id)}", {"$select": _FULL_SELECT}, text_body=True)
        atts = await self._attachments(token, message_id) if m.get("hasAttachments") else []
        return ToolResult(success=True, content=self._format_detail(m, atts))

    async def _action_get_thread(self, token: str, thread_id: str = "", **kwargs: Any) -> ToolResult:
        cid = str(thread_id or kwargs.get("conversation_id") or "").strip()
        if not cid:
            return ToolResult(success=False, error="thread_id (the Conversation ID) is required.")
        msgs = await self._collect(token, "/me/messages", {
            "$filter": f"conversationId eq '{odata_string(cid)}'", "$select": _FULL_SELECT, "$top": 50,
        }, 50, text_body=True)
        if not msgs:
            return ToolResult(success=True, content="Conversation not found or empty.")
        msgs.sort(key=lambda m: str(m.get("receivedDateTime") or m.get("lastModifiedDateTime") or ""))
        lines = [f"Conversation {cid} ({len(msgs)} messages, oldest first):\n"]
        for i, m in enumerate(msgs, 1):
            lines.append(f"--- Message {i}/{len(msgs)} ---")
            lines.append(self._format_detail(m, None))
            lines.append("")
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    def _format_detail(m: dict[str, Any], atts: list[dict[str, Any]] | None) -> str:
        lines = [
            f"Subject: {m.get('subject') or '(no subject)'}" + ("  [DRAFT]" if m.get("isDraft") else ""),
            f"From: {_who(m.get('from'))}",
            f"To: {_who_list(m.get('toRecipients'))}",
        ]
        if m.get("ccRecipients"):
            lines.append(f"Cc: {_who_list(m.get('ccRecipients'))}")
        if m.get("isDraft") and m.get("bccRecipients"):
            lines.append(f"Bcc: {_who_list(m.get('bccRecipients'))}")
        when = m.get("receivedDateTime") or m.get("lastModifiedDateTime")
        lines.append(f"Date: {_stamp(when)} UTC")
        lines.append(f"Message ID: {m.get('id')}")
        if m.get("isDraft"):
            lines.append(f"Draft ID: {m.get('id')}")
        lines.append(f"Conversation: {m.get('conversationId')}")
        if atts:
            lines.append(f"Attachments ({len(atts)}) — get_attachment saves them:")
            for i, a in enumerate(atts, 1):
                kind = " (inline)" if a.get("isInline") else ""
                lines.append(f"  [{i}] {a.get('name') or 'attachment'}{kind} — "
                             f"{a.get('contentType') or '?'}, {a.get('size') or 0} bytes")
        body = str(((m.get("body") or {}).get("content")) or "").strip()
        if (m.get("body") or {}).get("contentType") == "html":
            body = gmail_compose.html_to_text(body)
        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + "\n\n... [body truncated]"
        lines.append("")
        lines.append(body or "(empty body)")
        return "\n".join(lines)

    async def _action_list_folders(
        self, token: str, folder: str | None = None, **kwargs: Any,
    ) -> ToolResult:
        """The top-level folders, or the subfolders of *folder*."""
        parent = str(folder or "").strip()
        params = {"$top": 100, "$select": "id,displayName,totalItemCount,unreadItemCount,childFolderCount"}
        if parent:
            try:
                data = await self._folder_call(token, parent, lambda ref: self._get(
                    token, f"/me/mailFolders/{_seg(ref)}/childFolders", params))
            except LookupError as exc:
                return ToolResult(success=False, error=str(exc))
        else:
            data = await self._get(token, "/me/mailFolders", params)
        folders = data.get("value", [])
        if not folders:
            return ToolResult(success=True, content=(
                f"No subfolders in '{parent}'." if parent else "No mail folders found."
            ))
        lines = [f"Subfolders of '{parent}' ({len(folders)}):\n" if parent
                 else f"Mail folders ({len(folders)}):\n"]
        for f in folders:
            lines.append(
                f"  {f.get('displayName')}  ({f.get('unreadItemCount', 0)} unread / "
                f"{f.get('totalItemCount', 0)} total"
                + (f", {f['childFolderCount']} subfolders" if f.get("childFolderCount") else "")
                + f")\n    ID: {f.get('id')}"
            )
        if any(f.get("childFolderCount") for f in folders):
            lines.append("\nlist_folders folder=<name> lists a folder's subfolders; name one as "
                         "'Parent/Child'.")
        return ToolResult(success=True, content="\n".join(lines))

    # ── drafts ──────────────────────────────────────────────────────

    async def _repeat_problem(
        self, token: str, *, to: list[str], subject: str, conversation_id: str = "",
        after: str = "",
    ) -> str:
        """The refusal when this email already exists, else "" (never raises)."""
        try:
            matches = await self._find_repeats(token, to=to, subject=subject,
                                               conversation_id=conversation_id, after=after)
        except Exception as exc:  # noqa: BLE001 — the check never blocks a draft on its own failure
            log.info("Outlook repeat check failed", error=str(exc))
            return ""
        if not matches:
            return ""
        first = matches[0]
        what = f'"{first.get("subject") or "(no subject)"}" to {", ".join(_addr_list(first.get("toRecipients"))) or "?"}'
        if first.get("isDraft"):
            desc = f"Draft ID {first.get('id')} ({what})"
            todo = f" To change it use microsoft_mail action=update_draft draft_id={first.get('id')}."
        else:
            desc = f"the email sent on {_stamp(first.get('sentDateTime'))} UTC, Message ID {first.get('id')} ({what})"
            todo = ""
        more = f" and {len(matches) - 1} more like it" if len(matches) > 1 else ""
        return (
            f"Not created {gmail_compose.REPEAT_MARK}{desc}{more}. This email is already drafted "
            f"or sent, so no new copy was made. Tell the user, with that Draft ID or sent date.{todo}"
            + gmail_compose.repeat_way_past(reply=bool(conversation_id))
        )

    async def _find_repeats(
        self, token: str, *, to: list[str], subject: str, conversation_id: str, after: str,
    ) -> list[dict[str, Any]]:
        sel = "id,subject,toRecipients,ccRecipients,isDraft,sentDateTime,receivedDateTime,from"
        wanted = set(to)
        if conversation_id:
            # A reply: a draft in the conversation, or a reply already sent after the original.
            own = await self._own_addresses()
            msgs = await self._collect(token, "/me/messages", {
                "$filter": f"conversationId eq '{odata_string(conversation_id)}'", "$select": sel, "$top": 50,
            }, 50)
            out = []
            for m in msgs:
                sender = str(((m.get("from") or {}).get("emailAddress") or {}).get("address") or "").lower()
                if m.get("isDraft"):
                    out.append(m)
                elif (sender in own and str(m.get("sentDateTime") or "") > after
                      and (not wanted or wanted & set(_addr_list(m.get("toRecipients"))))):
                    out.append(m)
            return out
        if not wanted:
            return []
        out = []
        drafts = await self._collect(token, "/me/mailFolders/drafts/messages", {
            "$select": sel, "$top": 50, "$orderby": "lastModifiedDateTime desc",
        }, 50)
        days = get_config().tools.microsoft_mail.repeat_check_days
        sent: list[dict[str, Any]] = []
        if days > 0:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            sent = await self._collect(token, "/me/mailFolders/sentitems/messages", {
                "$select": sel, "$top": 100, "$filter": f"sentDateTime ge {since}",
                "$orderby": "sentDateTime desc",
            }, 200)
        for m in drafts + sent:
            if wanted & set(_addr_list(m.get("toRecipients")) + _addr_list(m.get("ccRecipients"))) \
                    and gmail_compose.subjects_match(subject, str(m.get("subject") or "")):
                out.append(m)
        return out

    async def _action_create_draft(
        self, token: str, to: str = "", cc: str = "", bcc: str = "", subject: str = "",
        body: str = "", html_body: str = "", reply_to_message_id: str = "",
        reply_all: Any = None, allow_repeat: Any = None,
        _attachments: list[mail_attachments.Attachment] | None = None, **kwargs: Any,
    ) -> ToolResult:
        atts = list(_attachments or [])
        bad = gmail_compose.invalid_recipients(to, cc, bcc)
        if bad:
            return ToolResult(success=False, error=gmail_compose.invalid_recipients_message(bad)
                              .replace("Nothing was sent.", "No draft was created."))
        if reply_to_message_id:
            return await self._reply_draft(
                token, reply_to_message_id, to=to, cc=cc, bcc=bcc, subject=subject,
                body=body or gmail_compose.html_to_text(html_body or ""),
                reply_all=flag(reply_all), allow_repeat=flag(allow_repeat), atts=atts,
            )
        if not to and not subject and not body and not html_body and not atts:
            return ToolResult(success=False, error=(
                "create_draft requires at least one of: to, subject, body (or reply_to_message_id)."
            ))
        if not flag(allow_repeat):
            problem = await self._repeat_problem(
                token, to=gmail_compose.parse_addresses(to, cc, bcc), subject=subject,
            )
            if problem:
                return ToolResult(success=False, error=problem)
        signature = await self._signature()
        body, html_body = mail_signature.apply(body, html_body, signature or "")
        message: dict[str, Any] = {
            "subject": subject,
            "body": ({"contentType": "html", "content": html_body} if html_body
                     else {"contentType": "text", "content": body}),
            "toRecipients": _recipients(to),
        }
        if cc:
            message["ccRecipients"] = _recipients(cc)
        if bcc:
            message["bccRecipients"] = _recipients(bcc)
        resp = await self._client.post(f"{GRAPH_API}/me/messages", json=message, headers=self._headers(token))
        resp.raise_for_status()
        draft = resp.json()
        return ToolResult(success=True, content=await self._saved_with_files(
            token, draft, "Draft created", atts,
        ) + self._signature_line(signature))

    async def _reply_draft(
        self, token: str, original_id: str, *, to: str, cc: str, bcc: str, subject: str,
        body: str, reply_all: bool, allow_repeat: bool,
        atts: list[mail_attachments.Attachment] | None = None,
    ) -> ToolResult:
        orig = await self._get(token, f"/me/messages/{_seg(original_id)}", {
            "$select": "id,subject,from,replyTo,toRecipients,ccRecipients,conversationId,receivedDateTime,sentDateTime",
        })
        own = await self._own_addresses()
        sender = str(((orig.get("from") or {}).get("emailAddress") or {}).get("address") or "").lower()
        if to:
            to_r = _recipients(to)
        elif sender and sender in own:
            to_r = orig.get("toRecipients") or []  # replying to your own sent email
        else:
            to_r = orig.get("replyTo") or ([orig["from"]] if orig.get("from") else [])
        cc_r = _recipients(cc) if cc else []
        if reply_all and not cc:
            mine = set(own) | set(_addr_list(to_r))
            cc_r = [r for r in (orig.get("toRecipients") or []) + (orig.get("ccRecipients") or [])
                    if _addr_list([r]) and _addr_list([r])[0] not in mine]
        if not allow_repeat:
            problem = await self._repeat_problem(
                token, to=_addr_list(to_r), subject=subject or str(orig.get("subject") or ""),
                conversation_id=str(orig.get("conversationId") or ""),
                after=str(orig.get("receivedDateTime") or orig.get("sentDateTime") or ""),
            )
            if problem:
                return ToolResult(success=False, error=problem)
        message: dict[str, Any] = {"toRecipients": to_r}
        if cc_r:
            message["ccRecipients"] = cc_r
        if bcc:
            message["bccRecipients"] = _recipients(bcc)
        if subject:
            message["subject"] = subject
        verb = "createReplyAll" if reply_all else "createReply"
        # The reply's own text (Outlook quotes the original below it) ends
        # with the signature.
        signature = await self._signature()
        comment = mail_signature.apply(body or "", "", signature or "")[0]
        resp = await self._client.post(
            f"{GRAPH_API}/me/messages/{_seg(original_id)}/{verb}",
            json={"comment": comment, "message": message}, headers=self._headers(token),
        )
        resp.raise_for_status()
        return ToolResult(success=True, content=await self._saved_with_files(
            token, resp.json(), f"Reply draft created in conversation {orig.get('conversationId')}",
            list(atts or []),
        ) + self._signature_line(signature))

    @staticmethod
    async def _signature() -> str | None:
        """The owner's agentic signature for a draft ("" = none; None when
        Flight Deck couldn't say — the draft is made without it)."""
        from captain_claw.microsoft_oauth_manager import MicrosoftOAuthManager

        return await MicrosoftOAuthManager().mail_signature()

    @staticmethod
    def _signature_line(signature: str | None) -> str:
        if signature is None:
            return ("\n  Signature: NOT added — Flight Deck didn't answer; tell the user the "
                    "draft is without their agentic signature.")
        return "\n  Signature: added at the end" if signature else ""

    @staticmethod
    def _draft_saved(d: dict[str, Any], lead: str) -> str:
        return (
            f"{lead} (saved in Outlook Drafts — not sent).\n"
            f"  Draft ID: {d.get('id', '?')}\n"
            f"  To: {_who_list(d.get('toRecipients'))}\n"
            f"  Subject: {d.get('subject') or '(no subject)'}"
            + (f"\n  Open in Outlook: {d['webLink']}" if d.get("webLink") else "")
        )

    async def _action_update_draft(
        self, token: str, draft_id: str = "", to: str = "", cc: str = "", bcc: str = "",
        subject: str = "", body: str = "", html_body: str = "",
        _attachments: list[mail_attachments.Attachment] | None = None,
        _remove_attachments: list[str] | None = None, **kwargs: Any,
    ) -> ToolResult:
        """Revise a draft in place: the body (the full new text), any of
        to / cc / bcc / subject given, and its files — *_attachments* added
        (one with the name of a file already there replaces it, once the new
        one is in), *_remove_attachments* dropped. With only file changes the
        body can be left out."""
        new_atts = list(_attachments or [])
        remove = list(_remove_attachments or [])
        draft_id = str(draft_id or "").strip()
        if not draft_id:
            return ToolResult(success=False, error="update_draft requires draft_id (from create_draft or list_drafts).")
        has_body = bool(str(body or "").strip() or str(html_body or "").strip())
        if not has_body and not new_atts and not remove:
            return ToolResult(success=False, error=(
                "update_draft replaces the draft's text — pass the full new body. read_message "
                "with the Draft ID shows the current one. (To change only its files, pass "
                "attachments / remove_attachments.) Nothing was changed."
            ))
        bad = gmail_compose.invalid_recipients(to, cc, bcc)
        if bad:
            return ToolResult(success=False, error=gmail_compose.invalid_recipients_message(bad)
                              .replace("Nothing was sent.", "Nothing was changed."))
        try:
            current = await self._get(token, f"/me/messages/{_seg(draft_id)}", {
                "$select": "id,isDraft,subject,toRecipients,webLink",
            })
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(success=False, error=(
                    f"Draft {draft_id} not found — it may have been sent or deleted. "
                    "list_drafts shows the current Draft IDs."
                ))
            raise
        if not current.get("isDraft"):
            return ToolResult(success=False, error=f"{draft_id} is not a draft (it was sent) — nothing was changed.")
        existing = await self._attachments(token, draft_id) if (new_atts or remove) else []
        drop_ids, problem = self._removal_ids(existing, remove)
        if problem:
            return ToolResult(success=False, error=problem)

        patch: dict[str, Any] = {}
        signature: str | None = ""
        if has_body:
            signature = await self._signature()
            body, html_body = mail_signature.apply(body, html_body, signature or "")
            patch["body"] = ({"contentType": "html", "content": html_body} if html_body
                             else {"contentType": "text", "content": body})
        if subject:
            patch["subject"] = subject
        if to:
            patch["toRecipients"] = _recipients(to)
        if cc:
            patch["ccRecipients"] = _recipients(cc)
        if bcc:
            patch["bccRecipients"] = _recipients(bcc)
        draft = current
        if patch:
            resp = await self._client.patch(
                f"{GRAPH_API}/me/messages/{_seg(draft_id)}", json=patch, headers=self._headers(token),
            )
            resp.raise_for_status()
            draft = resp.json()
        if not new_atts and not remove:
            return ToolResult(success=True, content=self._draft_saved(
                draft, "Draft updated (same draft — no new one was created)",
            ) + self._signature_line(signature))

        failed = await self._attach_all(token, draft_id, new_atts)
        failed_names = {a.filename.casefold() for a, _why in failed}
        replacing = {a.filename.casefold() for a in new_atts} - failed_names
        not_removed: list[str] = []
        for att in existing:
            name = str(att.get("name") or "")
            explicit = str(att.get("id")) in drop_ids
            # An old same-named file goes only once its replacement is in; and
            # when any new file failed, nothing is removed — the draft never
            # ends up with neither the old file nor its replacement.
            if explicit and failed:
                not_removed.append(f"{name} (kept because a new file failed to attach)")
                continue
            if explicit or (not att.get("isInline") and name.casefold() in replacing):
                try:
                    resp = await self._client.delete(
                        f"{GRAPH_API}/me/messages/{_seg(draft_id)}/attachments/{_seg(str(att.get('id')))}",
                        headers=self._headers(token),
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    not_removed.append(f"{name} ({self._why(exc)})")
        now = await self._attachments(token, draft_id)
        lines = [self._draft_saved(draft, "Draft updated (same draft — no new one was created)")]
        lines.append(self._files_line(now) or "  Attachments: none")
        lines += self._failure_lines(draft_id, failed)
        if not_removed:
            lines.append(f"  NOT removed (still on the draft): {', '.join(not_removed)}")
        return ToolResult(success=True, content="\n".join(lines) + self._signature_line(signature))

    @staticmethod
    def _removal_ids(existing: list[dict[str, Any]], remove: list[str]) -> tuple[set[str], str]:
        """The ids of the draft's attachments *remove* names (a name, a number
        from read_message, or 'all' — every file but the text's inline images,
        e.g. a signature logo); ``(ids, refusal)``."""
        ids: set[str] = set()
        names = [str(a.get("name") or "") for a in existing]
        listing = ", ".join(f"[{i}] {n}" for i, n in enumerate(names, 1)) or "none"
        for item in remove:
            key = item.strip()
            if key.lower() == "all":
                ids.update(str(a.get("id")) for a in existing if not a.get("isInline"))
            elif key.isdigit() and 1 <= int(key) <= len(existing):
                ids.add(str(existing[int(key) - 1].get("id")))
            else:
                hits = {str(a.get("id")) for a, n in zip(existing, names) if n.casefold() == key.casefold()}
                if not hits:
                    return set(), (f"The draft has no attachment '{key}' (it has: {listing}). "
                                   "Nothing was changed.")
                ids |= hits
        return ids, ""

    # ── attaching files ─────────────────────────────────────────────

    async def _saved_with_files(
        self, token: str, draft: dict[str, Any], lead: str,
        atts: list[mail_attachments.Attachment],
    ) -> str:
        """The new draft's summary, after attaching *atts* to it."""
        text = self._draft_saved(draft, lead)
        if not atts:
            return text
        draft_id = str(draft.get("id") or "")
        failed = await self._attach_all(token, draft_id, atts)
        failed_ids = {id(f) for f, _why in failed}
        added = [a for a in atts if id(a) not in failed_ids]
        lines = [text]
        if added:
            lines.append(f"  Attachments: {mail_attachments.describe(added)}")
        lines += self._failure_lines(draft_id, failed)
        return "\n".join(lines)

    @staticmethod
    def _failure_lines(
        draft_id: str, failed: list[tuple[mail_attachments.Attachment, str]],
    ) -> list[str]:
        if not failed:
            return []
        return [
            "  NOT attached: " + ", ".join(f"{a.filename} ({why})" for a, why in failed)
            + f". The draft is saved without them — try again with update_draft "
            f"draft_id={draft_id} attachments=[…], or tell the user.",
        ]

    @staticmethod
    def _files_line(atts: list[dict[str, Any]]) -> str:
        if not atts:
            return ""
        return "  Attachments: " + ", ".join(
            f"{a.get('name') or 'attachment'} ({'inline image, ' if a.get('isInline') else ''}"
            f"{mail_attachments.human_size(int(a.get('size') or 0))})"
            for a in atts
        )

    @staticmethod
    def _why(exc: httpx.HTTPError) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            code, message = graph_message(exc)
            detail = message or code
            return f"Outlook HTTP {exc.response.status_code}" + (f": {detail}" if detail else "")
        return type(exc).__name__

    async def _attach_all(
        self, token: str, message_id: str, atts: list[mail_attachments.Attachment],
    ) -> list[tuple[mail_attachments.Attachment, str]]:
        """Attach each file to the draft; the ones that failed, with why."""
        failed: list[tuple[mail_attachments.Attachment, str]] = []
        for att in atts:
            try:
                await self._add_attachment(token, message_id, att)
            except httpx.HTTPError as exc:
                log.info("Outlook attachment failed", name=att.filename, error=str(exc))
                failed.append((att, self._why(exc)))
        return failed

    async def _add_attachment(
        self, token: str, message_id: str, att: mail_attachments.Attachment,
    ) -> None:
        """One file onto a draft: a single POST under 3 MB, else an upload
        session in chunks. Raises :class:`httpx.HTTPError`."""
        url = f"{GRAPH_API}/me/messages/{_seg(message_id)}/attachments"
        size = len(att.data)

        async def _post_whole() -> None:
            resp = await self._client.post(url, json={
                "@odata.type": "#microsoft.graph.fileAttachment",
                "name": att.filename,
                "contentType": att.mime_type,
                "contentBytes": base64.b64encode(att.data).decode("ascii"),
            }, headers=self._headers(token))
            resp.raise_for_status()

        if size < _SMALL_ATTACHMENT_BYTES:
            await _post_whole()
            return
        resp = await self._client.post(f"{url}/createUploadSession", json={"AttachmentItem": {
            "attachmentType": "file", "name": att.filename, "size": size,
            "contentType": att.mime_type,
        }}, headers=self._headers(token))
        if resp.status_code == 400 and graph_message(
            httpx.HTTPStatusError("", request=resp.request, response=resp),
        )[0] == _SESSION_TOO_SMALL:
            await _post_whole()
            return
        resp.raise_for_status()
        upload_url = str((resp.json() or {}).get("uploadUrl") or "")
        if not upload_url.startswith("https://"):
            raise httpx.HTTPStatusError(
                "Outlook did not return an upload session",
                request=resp.request, response=httpx.Response(502, request=resp.request),
            )
        for start in range(0, size, _UPLOAD_CHUNK_BYTES):
            chunk = att.data[start:start + _UPLOAD_CHUNK_BYTES]
            # The session URL carries its own credential: no Authorization header.
            put = await self._client.put(upload_url, content=chunk, headers={
                "Content-Type": "application/octet-stream",
                "Content-Range": f"bytes {start}-{start + len(chunk) - 1}/{size}",
            })
            put.raise_for_status()

    async def _action_list_drafts(
        self, token: str, query: str = "", max_results: Any = None, **kwargs: Any,
    ) -> ToolResult:
        n = limit(max_results, 10, 50)
        drafts = await self._collect(token, "/me/mailFolders/drafts/messages", {
            "$select": "id,subject,toRecipients,ccRecipients,lastModifiedDateTime,conversationId,bodyPreview",
            "$orderby": "lastModifiedDateTime desc", "$top": 100,
        }, 200)
        q = str(query or "").strip().lower()
        if q.startswith("to:"):
            addr = q[3:].strip()
            drafts = [d for d in drafts
                      if any(addr in a for a in _addr_list(d.get("toRecipients")) + _addr_list(d.get("ccRecipients")))]
        elif q:
            drafts = [d for d in drafts if q in f"{d.get('subject') or ''} {d.get('bodyPreview') or ''}".lower()]
        drafts = drafts[:n]
        if not drafts:
            return ToolResult(success=True, content="No drafts found" + (f" for '{query}'." if q else "."))
        lines = [f"Drafts ({len(drafts)}):\n"]
        for d in drafts:
            lines.append(
                f"  {d.get('subject') or '(no subject)'}\n    To: {_who_list(d.get('toRecipients'))}"
                f"  |  edited {_stamp(d.get('lastModifiedDateTime'))} UTC\n    Draft ID: {d.get('id')}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    # ── organizing: archive / move / create_folder / categories ─────
    #
    # Moves, folders and categories (Mail.ReadWrite) — nothing is written,
    # sent or deleted. execute() has already run check_mailbox_change and
    # parsed the targets into _threads / _messages.

    @classmethod
    def _why_not(cls, exc: httpx.HTTPError) -> str:
        """Why one email wasn't changed (the "NOT archived: <id> (…)" reason)."""
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            return "not found"
        return cls._why(exc)

    @staticmethod
    def _not_lines(verb: str, failed: list[tuple[str, str]]) -> list[str]:
        lines = [f"  NOT {verb}: {target} ({why})" for target, why in failed[:_MAX_LISTED_IDS]]
        if len(failed) > _MAX_LISTED_IDS:
            lines.append(f"  … and {len(failed) - _MAX_LISTED_IDS} more not {verb}.")
        return lines

    @classmethod
    def _nothing_changed(cls, verb: str, failed: list[tuple[str, str]], skipped: str = "") -> ToolResult:
        """No email changed and every one tried failed (the *skipped* ones
        needed no change): the refusal, listing why."""
        lines = [f"Nothing was {verb}:", *cls._not_lines(verb, failed)]
        if skipped:
            lines.append(f"  Skipped: {skipped}.")
        if all(why.endswith("not found") for _target, why in failed):
            lines.append("Check the IDs — list_messages / search print them.")
        return ToolResult(success=False, error="\n".join(lines))

    async def _conversations(
        self, token: str, threads: list[str], select: str,
    ) -> list[tuple[str, list[dict[str, Any]] | str]]:
        """Each conversation's messages (their *select* fields), or why it
        couldn't be read."""
        async def one(cid: str) -> list[dict[str, Any]]:
            return await self._collect(token, "/me/messages", {
                "$filter": f"conversationId eq '{odata_string(cid)}'", "$select": select, "$top": 100,
            }, _MAX_ORGANIZE_MESSAGES + 1)

        return [
            (cid, self._why_not(exc) if exc is not None else (msgs or "conversation not found"))
            for cid, msgs, exc in await self._each(threads, one)
        ]

    async def _folder_info(self, token: str, ref: str) -> dict[str, Any] | None:
        """A folder's ``{id, displayName, parentFolderId}``; None when there is
        no such folder."""
        try:
            return await self._get(token, f"/me/mailFolders/{_seg(ref)}", {"$select": _FOLDER_SELECT})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    async def _action_archive(
        self, token: str, _threads: list[str] | None = None, _messages: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Move conversations / messages to the mailbox's Archive folder."""
        archive = (await self._known_folders(token, ("archive",)))["archive"]
        if not archive:
            return ToolResult(success=False, error=(
                "This mailbox has no Archive folder. Create one with action=create_folder "
                "folder=Archive and move there, or tell the user. Nothing was moved."
            ))
        return await self._move(token, list(_threads or []), list(_messages or []), archive,
                                verb="archived", folder_arg="Archive")

    async def _action_move(
        self, token: str, folder: str = "", _threads: list[str] | None = None,
        _messages: list[str] | None = None, **kwargs: Any,
    ) -> ToolResult:
        """Move conversations / messages to *folder*."""
        dest = await self._destination(token, folder)
        if isinstance(dest, str):
            return ToolResult(success=False, error=dest)
        return await self._move(token, list(_threads or []), list(_messages or []), dest,
                                verb="moved", folder_arg=str(folder).strip())

    async def _destination(self, token: str, folder: Any) -> dict[str, Any] | str:
        """The move destination's ``{id, displayName, parentFolderId, inbox}``,
        or the refusal."""
        name = str(folder or "").strip()
        if not name:
            return (
                "move needs folder=<destination> — a folder name, 'Parent/Child' path or ID from "
                "list_folders (action=archive files emails in the Archive folder). Nothing was moved."
            )
        if _WELL_KNOWN.get(name.lower()) in _STAY_PUT:
            return self._no_move_to(_WELL_KNOWN[name.lower()])
        seen: dict[str, str] = {}  # where its parents are — read afresh for every move
        try:
            # A well-known folder this mailbox lacks (e.g. no Archive) falls
            # back to one the user named so.
            info = await self._folder_call(token, name, lambda ref: self._get(
                token, f"/me/mailFolders/{_seg(ref)}", {"$select": _FOLDER_SELECT}), seen=seen)
        except LookupError as exc:
            return f"{exc} Nothing was moved."
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            return f"No mail folder '{name}' — list_folders shows the folders. Nothing was moved."
        fid = str(info.get("id") or "")
        known = await self._known_folders(token, _MOVE_KNOWN)
        # Its display name or ID may hide one of them ('Junk Email', an ID …) …
        for kind in _STAY_PUT:
            if known[kind] and str(known[kind].get("id")) == fid:
                return self._no_move_to(kind)
        # … or it may sit inside Deleted Items / Junk Email ('Deleted Items/Old').
        try:
            inside = await self._bin_of(token, fid, str(info.get("parentFolderId") or ""), seen=seen)
        except _FolderPlaceError as exc:
            return (
                f"Couldn't confirm that '{info.get('displayName') or name}' isn't inside Deleted "
                f"Items or Junk Email ({exc}), so nothing was moved. Try again, or pick another "
                "folder from list_folders."
            )
        if inside:
            return self._no_move_to(inside, str(info.get("displayName") or name))
        inbox = known["inbox"]
        is_inbox = str(inbox.get("id")) == fid if inbox else _WELL_KNOWN.get(name.lower()) == "inbox"
        return {**info, "inbox": is_inbox}

    @staticmethod
    def _no_move_to(kind: str, inside: str = "") -> str:
        """The refusal to move to the well-known folder *kind* — or to the
        folder *inside* it."""
        where = f"'{inside}' (it's inside {_FOLDER_LABELS[kind]})" if inside else _FOLDER_LABELS[kind]
        return (
            f"microsoft_mail doesn't move email to {where} — deleting, junking and "
            "filing into Drafts / Sent Items / Outbox are out of scope (the user can do that in "
            "Outlook). Nothing was moved."
        )

    async def _move(
        self, token: str, threads: list[str], messages: list[str], dest: dict[str, Any],
        *, verb: str, folder_arg: str,
    ) -> ToolResult:
        """Move *messages* and the conversations' messages to *dest* — a
        conversation's messages already there, or in Drafts / Sent Items /
        Deleted Items / Junk / Outbox (or a folder inside Deleted Items / Junk),
        stay where they are."""
        dest_id = str(dest.get("id") or "")
        plan = list(messages)
        planned = set(plan)
        came_from: dict[str, str] = {}  # a conversation's message → its folder's ID
        inbox_id = ""
        failed: list[tuple[str, str]] = []
        already = 0
        left: dict[str, int] = {}
        if threads:
            known = await self._known_folders(token, _MOVE_KNOWN)
            stay = {str(info["id"]): kind for kind, info in known.items() if info and kind in _STAY_PUT}
            inbox_id = str((known["inbox"] or {}).get("id") or "")
            convs = await self._conversations(token, threads, "id,parentFolderId")
            # Mail filed inside Deleted Items / Junk Email stays put too; mail
            # in a folder whose place couldn't be read isn't moved.
            places = {str(m.get("parentFolderId") or "") for _cid, msgs in convs
                      if not isinstance(msgs, str) for m in msgs} - set(stay) - {dest_id, ""}
            seen: dict[str, str] = {}
            unchecked: dict[str, str] = {}  # folder ID → why its place is unknown

            async def bin_of(f: str) -> str:
                try:
                    return await self._bin_of(token, f, seen=seen)
                except _FolderPlaceError as exc:
                    return f"?{exc}"

            for place, kind, exc in await self._each(sorted(places), bin_of):
                if exc is not None:
                    unchecked[place] = self._why(exc)
                elif kind.startswith("?"):
                    unchecked[place] = kind[1:]
                elif kind:
                    stay[place] = kind
            for cid, msgs in convs:
                if isinstance(msgs, str):
                    failed.append((cid, msgs))
                    continue
                for m in msgs:
                    where, mid = str(m.get("parentFolderId") or ""), str(m.get("id") or "")
                    if where == dest_id:
                        already += 1
                    elif where in stay:
                        left[stay[where]] = left.get(stay[where], 0) + 1
                    elif mid and mid not in planned:
                        planned.add(mid)
                        if where in unchecked:
                            failed.append((mid, f"couldn't check its folder — {unchecked[where]}"))
                            continue
                        plan.append(mid)
                        came_from[mid] = where
        if len(plan) > _MAX_ORGANIZE_MESSAGES:
            return ToolResult(success=False, error=(
                f"These conversations hold {len(plan)} messages to move — at most "
                f"{_MAX_ORGANIZE_MESSAGES} per call. Pass fewer thread_ids. Nothing was moved."
            ))

        async def move_one(mid: str) -> str:
            resp = await self._client.post(
                f"{GRAPH_API}/me/messages/{_seg(mid)}/move",
                json={"destinationId": dest_id}, headers=self._headers(token),
            )
            resp.raise_for_status()
            try:
                return str((resp.json() or {}).get("id") or "?")
            except ValueError:
                return "?"

        moved: list[tuple[str, str]] = []
        for mid, new_id, exc in await self._each(plan, move_one):
            if exc is None:
                moved.append((mid, new_id))
            else:
                failed.append((mid, self._why_not(exc)))
        name = str(dest.get("displayName") or folder_arg)
        skipped = []
        if already:
            skipped.append(f"{_plural(already, 'message')} already in '{name}'")
        if left:
            skipped.append("left where they are: " + ", ".join(
                f"{n} in {_FOLDER_LABELS[kind]}" for kind, n in left.items()))
        # Nothing moved: a success only when nothing needed to (all skipped).
        if not moved and (failed or not skipped):
            return self._nothing_changed(verb, failed, "; ".join(skipped))

        if moved:
            lines = [
                f"{verb.capitalize()} {_plural(len(moved), 'message')} to '{name}'.",
                "Moving changes an Outlook message's ID — use the new one (old → new):",
                *(f"  {old} → {new}" for old, new in moved[:_MAX_LISTED_IDS]),
            ]
            if len(moved) > _MAX_LISTED_IDS:
                lines.append(f"  … and {len(moved) - _MAX_LISTED_IDS} more — list_messages "
                             f"folder='{folder_arg}' shows their IDs.")
        else:
            lines = [f"Nothing was {verb}."]
        if skipped:
            lines.append(f"  Skipped: {'; '.join(skipped)}.")
        lines += self._not_lines(verb, failed)
        # Undo by the folder they came from — known for a conversation's
        # messages ("" for message_ids: not read before the move).
        back: dict[str, list[str]] = {}
        for old, new in moved:
            back.setdefault(came_from.get(old, ""), []).append(new)

        def undo(source: str, new_ids: str) -> str:
            if source:
                return (f"microsoft_mail action=move folder={'Inbox' if source == inbox_id else source} "
                        f"{new_ids}")
            if dest.get("inbox"):
                return f"microsoft_mail action=move folder=<the folder they came from> {new_ids}"
            return (f"microsoft_mail action=move folder=Inbox {new_ids} (or the folder they came "
                    "from, if not the Inbox)")

        if len(moved) <= _MAX_LISTED_IDS:
            lines += [f"Undo: {undo(source, 'message_ids=[' + ', '.join(ids) + ']')}"
                      for source, ids in back.items()]
        else:  # too many to list: how many came from where, and where to find them
            lines.append(f"Undo — their new IDs aren't all listed here; list_messages "
                         f"folder='{folder_arg}' shows them. By the folder they came from:")
            for source, ids in back.items():
                origin = f" from {'Inbox' if source == inbox_id else source}" if source else ""
                lines.append(f"  {_plural(len(ids), 'message')}{origin}: "
                             f"{undo(source, 'message_ids=[their new IDs]')}")
        return ToolResult(success=True, content="\n".join(lines))

    async def _action_create_folder(self, token: str, folder: str = "", **kwargs: Any) -> ToolResult:
        """Create a mail folder; 'Parent/Child' creates each missing level. A
        path's first part is found as a single name is — a well-known folder
        ('Inbox/Receipts'), a top-level folder, else one in the Inbox — and is
        made at the top level only when it is none of those. Nothing is made
        inside Deleted Items or Junk Email."""
        parts = _folder_parts(folder)
        if not parts:
            return ToolResult(success=False, error=(
                "create_folder needs folder=<name> — 'Parent/Child' (e.g. 'Inbox/Receipts') makes "
                "it inside Parent; no empty parts. Nothing was created."
            ))
        full = "/".join(parts)
        have, how = await self._walk(token, parts, known_ids=True)
        if have and len(parts) > 1:
            for kind, info in (await self._known_folders(token, _BINS)).items():
                if info and str(info["id"]) == have[0]:
                    return ToolResult(success=False, error=(
                        f"microsoft_mail doesn't make folders inside {_FOLDER_LABELS[kind]} — mail "
                        "filed there goes when it is emptied, and deleting / junking is out of scope. "
                        "Nothing was created."
                    ))
        where = " in the Inbox" if how == "inbox" else ""
        parent: str | None = None  # None: the top level
        created: list[tuple[str, str]] = []  # (path, ID)
        for i, part in enumerate(parts):
            here = "/".join(parts[:i + 1])
            if i < len(have):
                found = have[i]
            else:
                try:
                    found, new = await self._post_folder(token, parent, part)
                except httpx.HTTPError as exc:
                    done = (
                        "Created before that: " + ", ".join(f"'{p}' (ID: {f})" for p, f in created) + "."
                    ) if created else "Nothing was created."
                    return ToolResult(success=False, error=(
                        f"Couldn't create folder '{here}' ({self._why(exc)}). {done}"
                    ))
                if new:
                    created.append((here, found))
            parent = found
        # Only the whole path is cached: a lookup of it now finds this folder.
        # Its first part needn't be what that name alone resolves to (a
        # top-level 'Projects' beside Inbox/Projects/Acme) — nor a shorter part
        # of the path; a well-known name stays one ('Inbox' → inbox).
        if len(parts) > 1 or how != "known":
            self._token_cache(self._folder_cache, token)[full.casefold()] = parent
        made_leaf = bool(created) and created[-1][0] == full
        if made_leaf:
            lines = [f"Created folder '{full}'{where}.", f"  Folder ID: {parent}"]
        else:
            lines = [f"Folder '{full}' already exists{where} (ID: {parent}) — nothing created."]
        parents = created[:-1] if made_leaf else created
        if parents:
            lines.append(f"  Also created its parent folder{'s' if len(parents) > 1 else ''}: "
                         + ", ".join(f"'{p}' (ID: {f})" for p, f in parents))
        if made_leaf:
            lines.append(f"File emails there only when the user asked: microsoft_mail action=move "
                         f"folder='{full}' thread_ids=[…]")
            lines.append("Undo: delete the folder in Outlook (this tool doesn't delete folders).")
        return ToolResult(success=True, content="\n".join(lines))

    async def _post_folder(self, token: str, parent: str | None, name: str) -> tuple[str, bool]:
        """Create *name* in *parent* (the top level when None): ``(ID, created)``.
        Outlook's 409 (ErrorFolderExists — made meanwhile) finds the existing one."""
        url = f"{GRAPH_API}/me/mailFolders" + ("" if parent is None else f"/{_seg(parent)}/childFolders")
        resp = await self._client.post(url, json={"displayName": name}, headers=self._headers(token))
        if resp.status_code == 409:
            found = await self._child_id(token, parent, name)
            if found:
                return found, False
        resp.raise_for_status()
        return str((resp.json() or {}).get("id") or "?"), True

    async def _action_add_categories(
        self, token: str, categories: Any = None, _threads: list[str] | None = None,
        _messages: list[str] | None = None, **kwargs: Any,
    ) -> ToolResult:
        """Add categories to messages / conversations (keeping theirs)."""
        return await self._categorize(token, categories, list(_threads or []),
                                      list(_messages or []), adding=True)

    async def _action_remove_categories(
        self, token: str, categories: Any = None, _threads: list[str] | None = None,
        _messages: list[str] | None = None, **kwargs: Any,
    ) -> ToolResult:
        """Remove categories from messages / conversations."""
        return await self._categorize(token, categories, list(_threads or []),
                                      list(_messages or []), adding=False)

    async def _categorize(
        self, token: str, categories: Any, threads: list[str], messages: list[str], *, adding: bool,
    ) -> ToolResult:
        """Merge *categories* into (or out of) each message's own, any case —
        a PATCH replaces the whole list, so only a message that changes gets
        one (it keeps its ID). A conversation is every message in it."""
        names = _names(categories)
        if not names:
            return ToolResult(success=False, error=(
                f"{'add' if adding else 'remove'}_categories needs categories=[names], e.g. "
                "categories=[\"Receipts\"]. Nothing was changed."
            ))
        have: dict[str, list[str] | None] = {m: None for m in messages}  # None: not read yet
        failed: list[tuple[str, str]] = []
        convs = await self._conversations(token, threads, "id,categories") if threads else []
        for cid, msgs in convs:
            if isinstance(msgs, str):
                failed.append((cid, msgs))
                continue
            for m in msgs:
                mid = str(m.get("id") or "")
                if mid and have.get(mid) is None:
                    have[mid] = [str(c) for c in m.get("categories") or []]
        if len(have) > _MAX_ORGANIZE_MESSAGES:
            return ToolResult(success=False, error=(
                f"These conversations hold {len(have)} messages — at most {_MAX_ORGANIZE_MESSAGES} "
                "per call. Pass fewer thread_ids. Nothing was changed."
            ))
        folded = {n.casefold() for n in names}

        async def recategorize(mid: str) -> list[str]:
            """The categories added to / removed from *mid* ([] when it needs no change)."""
            current = have[mid]
            if current is None:
                data = await self._get(token, f"/me/messages/{_seg(mid)}", {"$select": "categories"})
                current = [str(c) for c in data.get("categories") or []]
            if adding:
                own = {c.casefold() for c in current}
                delta = [n for n in names if n.casefold() not in own]
                new = current + delta
            else:
                delta = [c for c in current if c.casefold() in folded]
                new = [c for c in current if c.casefold() not in folded]
            if delta:
                resp = await self._client.patch(
                    f"{GRAPH_API}/me/messages/{_seg(mid)}", json={"categories": new},
                    headers=self._headers(token),
                )
                resp.raise_for_status()
            return delta

        changed: list[tuple[str, list[str]]] = []
        same = 0
        for mid, delta, exc in await self._each(list(have), recategorize):
            if exc is not None:
                failed.append((mid, self._why_not(exc)))
            elif delta:
                changed.append((mid, delta))
            else:
                same += 1
        what = ("categories " if len(names) > 1 else "category ") + ", ".join(f"'{n}'" for n in names)
        it = "them" if len(names) > 1 else "it"
        had = "already had" if adding else "didn't have"
        # Nothing changed: a success only when nothing needed to.
        if not changed and (failed or not same):
            return self._nothing_changed(
                "changed", failed, f"{_plural(same, 'message')} {had} {it}" if same else "")
        if changed:
            count = _plural(len(changed), "message")
            lines = [f"Added {what} to {count}." if adding else f"Removed {what} from {count}.",
                     f"  Message IDs: {_listed([mid for mid, _delta in changed])}"]
        else:
            lines = ["Nothing to change."]
        if same:
            lines.append(f"  {_plural(same, 'message')} {had} {it} — unchanged.")
        lines += self._not_lines("changed", failed)
        # Undo exactly what changed: messages grouped by the categories they gained / lost.
        groups: dict[tuple[str, ...], list[str]] = {}
        for mid, delta in changed:
            groups.setdefault(tuple(delta), []).append(mid)
        for cats, ids in groups.items():
            target = (f"message_ids=[{', '.join(ids)}]" if len(ids) <= _MAX_LISTED_IDS
                      else "with the same message IDs")
            lines.append(
                f"Undo: microsoft_mail action={'remove' if adding else 'add'}_categories "
                f"categories={json.dumps(list(cats), ensure_ascii=False)} {target}"
            )
        return ToolResult(success=True, content="\n".join(lines))

    # ── attachments ─────────────────────────────────────────────────

    async def _action_get_attachment(
        self, token: str, message_id: str = "", attachment: str = "",
        _runtime: dict[str, Any] | None = None, **kwargs: Any,
    ) -> ToolResult:
        from captain_claw.tools.google_mail import GoogleMailTool

        if not message_id:
            return ToolResult(success=False, error="message_id is required.")
        atts = await self._attachments(token, message_id)
        if not atts:
            return ToolResult(success=False, error=f"Message {message_id} has no attachments.")
        listing = ", ".join(f"[{i}] {a.get('name') or 'attachment'}" for i, a in enumerate(atts, 1))
        wanted = str(attachment or "").strip()
        if not wanted and len(atts) > 1:
            return ToolResult(success=False, error=(
                f"The message has {len(atts)} attachments ({listing}) — pass attachment=<number, name or 'all'>."
            ))
        name = wanted.lower()
        if not wanted or name == "all":
            chosen = list(enumerate(atts))[:_MAX_ATTACHMENTS_PER_CALL]
        elif wanted.isdigit() and 1 <= int(wanted) <= len(atts):
            chosen = [(int(wanted) - 1, atts[int(wanted) - 1])]
        else:
            exact = [(i, a) for i, a in enumerate(atts) if str(a.get("name") or "").lower() == name]
            partial = [] if wanted.isdigit() else [
                (i, a) for i, a in enumerate(atts) if name in str(a.get("name") or "").lower()]
            chosen = (exact or partial)[:1]
        if not chosen:
            return ToolResult(success=False, error=f"No attachment '{wanted}' — the message has {listing}.")

        kinds = [str(a.get("@odata.type") or "") for a in atts]
        names = GoogleMailTool._attachment_save_names([
            {"filename": (a.get("name") or "") + (".eml" if "itemAttachment" in k and not
                                                  str(a.get("name") or "").lower().endswith(".eml") else ""),
             "mime_type": a.get("contentType") or ""}
            for a, k in zip(atts, kinds)
        ])
        lines: list[str] = []
        saved = 0
        for index, att in chosen:
            ok, line = await self._save_attachment(token, message_id, att, kinds[index], names[index], _runtime or {})
            lines.append(line)
            saved += ok
        if not saved:
            return ToolResult(success=False, error="\n".join(lines))
        if len(atts) > len(chosen) and name == "all":
            lines.append(f"({len(atts) - len(chosen)} more not saved — ask for them by number.)")
        return ToolResult(success=True, content="\n".join(lines))

    async def _save_attachment(
        self, token: str, message_id: str, att: dict[str, Any], kind: str, save_name: str,
        runtime: dict[str, Any],
    ) -> tuple[bool, str]:
        from captain_claw import saved_attribution
        from captain_claw import speaker as _speaker
        from captain_claw.tools.google_drive import _READER_BY_SUFFIX, GoogleDriveTool, _DownloadPathError

        name = att.get("name") or save_name
        if "referenceAttachment" in kind:
            return False, (f"'{name}' is a link to a cloud file, not an attached file — open it "
                           "with microsoft_onedrive (its share URL) instead.")
        size = int(att.get("size") or 0)
        if size > _MAX_ATTACHMENT_BYTES:
            return False, f"'{name}' is too large ({size} bytes)."
        try:
            resp = await self._client.get(
                f"{GRAPH_API}/me/messages/{_seg(message_id)}/attachments/{_seg(att['id'])}/$value",
                headers=self._headers(token),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return False, f"'{name}' could not be fetched from Outlook (HTTP {exc.response.status_code})."
        except httpx.HTTPError as exc:
            return False, f"'{name}' could not be fetched from Outlook ({type(exc).__name__})."
        blob = resp.content
        if not blob:
            return False, f"'{name}' is empty."
        if len(blob) > _MAX_ATTACHMENT_BYTES:
            return False, f"'{name}' is too large ({len(blob)} bytes)."
        # Graph message ids share long mailbox prefixes: name the folder by a hash.
        folder = "mail-" + hashlib.sha1(message_id.encode("utf-8")).hexdigest()[:16]
        try:
            dest = GoogleDriveTool._download_dest(None, f"{folder}/{save_name}", "", runtime)
        except _DownloadPathError as exc:
            return False, str(exc)
        prior = saved_attribution.prior_creator(dest)
        member = _speaker.current()
        if (member is not None and dest.exists()
                and not saved_attribution.member_may_change(dest, member.speaker_id)):
            return False, _speaker.PATH_REFUSED_PREFIX + _speaker.FILE_NOT_YOURS_WHY
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(dest.write_bytes, blob)
        except OSError as exc:
            return False, f"'{name}' could not be saved ({exc.strerror or type(exc).__name__})."
        saved_attribution.note_write(dest, prior)
        registry = runtime.get("_file_registry")
        if registry is not None:
            try:
                registry.register(logical_path=dest.name, physical_path=str(dest),
                                  task_id=str(runtime.get("_task_id") or ""))
            except Exception:
                pass
        reader = _READER_BY_SUFFIX.get(dest.suffix.lower(), "read")
        # "Saved to:", not "Path:" — chat bridges auto-send files named on a
        # "Path:" line, and an email's attachment is an input.
        return True, (
            f"Saved attachment '{name}' ({att.get('contentType') or 'unknown'}, {len(blob)} bytes)\n"
            f"  Saved to: {dest}\n"
            f"Use {reader}(path=\"{dest}\") to view it."
        )

    async def close(self) -> None:
        await self._client.aclose()
