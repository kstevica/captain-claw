"""imap_mail — the owner's IMAP/SMTP inboxes (any mail provider), several at once.

The third mailbox tool beside google_mail and microsoft_mail. The user connects
inboxes with a username and password in Flight Deck → Connections → Email
(IMAP); Flight Deck keeps the passwords and does the IMAP / SMTP work
(``flight_deck.imap_routes``), so this tool only relays (see
:mod:`captain_claw.imap_mail_manager`).

* Read: list_accounts, list_messages, search (Gmail-like: from:, to:,
  subject:, since:, before:, is:unread, has:attachment, free text),
  read_message, get_thread, list_folders, list_drafts, get_attachment (saved
  under ``saved/downloads/<session>/mail-<hash>/``). ``account=all`` reads
  every inbox at once.
* Write: create_draft, update_draft (a new Draft ID each time — IMAP replaces
  the draft), send, send_draft. Sends go through Flight Deck's per-inbox gate
  (sending is OFF until the user turns it on; allowlist, daily limit,
  duplicates, repeats, audit and bell). Every draft and send ends with the
  inbox's agentic signature (Flight Deck adds it).
* Organize: archive, move, mark_read, mark_unread, create_folder — never to
  Trash or Junk; nothing is deleted.
* Mail writes and mailbox changes in automated turns go through
  :mod:`captain_claw.mail_authority`, checked before any request.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from typing import Any

import httpx

from captain_claw import mail_attachments, mail_authority
from captain_claw.imap_mail_manager import (
    NOT_CONNECTED,
    WHERE,
    ImapMailClient,
    ImapRefused,
    cached_accounts,
)
from captain_claw.logging import get_logger
from captain_claw.microsoft_graph import flag, limit, pop_runtime
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

_WRITE_ACTIONS = frozenset({"create_draft", "update_draft", "send", "send_draft"})
_SEND_ACTIONS = frozenset({"send", "send_draft"})
_ORGANIZE_ACTIONS = frozenset({"archive", "move", "mark_read", "mark_unread", "create_folder"})
_FILE_ACTIONS = frozenset({"create_draft", "update_draft", "send"})
_READ_ACTIONS = ("list_accounts", "list_messages", "search", "read_message", "get_thread", "list_folders",
                 "list_drafts", "get_attachment")
_ACTIONS = _READ_ACTIONS + ("create_draft", "update_draft", "send", "send_draft",
                            "archive", "move", "mark_read", "mark_unread", "create_folder")
_MAX_ATTACHMENT_BYTES = 30 * 1024 * 1024


def _human(n: int) -> str:
    return mail_attachments.human_size(int(n or 0))


def _write_account(kwargs: dict[str, Any]) -> str:
    """The inbox id a write goes from: the one an id names (a reply's message,
    a draft), else ``account`` matched to a connected inbox by id, address or
    label; "" when it can't be told. Only the mail guard reads it (an email's
    turn may send from its own inbox): Flight Deck still picks and checks."""
    for key in ("reply_to_message_id", "draft_id"):
        ref = str(kwargs.get(key) or "")
        if ":" in ref:
            return ref.split(":", 1)[0].strip().lower()
    spec = str(kwargs.get("account") or "").strip().casefold()
    if not spec:
        accounts = cached_accounts()
        return str(accounts[0].get("id") or "").lower() if len(accounts) == 1 else ""
    for acct in cached_accounts():
        if spec in {str(acct.get(k) or "").casefold() for k in ("id", "email", "label")}:
            return str(acct.get("id") or "").lower()
    return spec


class ImapMailTool(Tool):
    """The owner's IMAP/SMTP inboxes, through Flight Deck."""

    name = "imap_mail"
    description = (
        "Email in the user's own IMAP/SMTP inboxes (any provider: company mail server, Fastmail, "
        "iCloud, Yahoo, Zoho, GMX, a hosting provider's mail …). The user may have SEVERAL inboxes: "
        "each has an id (list_accounts shows them, with which can send). Pass account=<id, email or "
        "label> to pick one; account='all' lists or searches every inbox at once. Message, Draft and "
        "Thread IDs carry their inbox ('work:1700000000:4521:INBOX'), so an action on an ID needs no "
        "account. "
        "WRITE ONLY WHEN ASKED: create_draft / update_draft / send / send_draft only when the user "
        "asked for that email in this conversation (or a scheduled job's own text explicitly says "
        "to). Reading, summarizing or triaging NEVER implies drafting or sending — say who is "
        "waiting and offer. If the tool answers '[not-authorized: mail-write]', don't retry: tell the "
        "user. SENDING: send / send_draft work only when the user turned sending on for that inbox "
        "(list_accounts shows 'sending on'); otherwise create_draft and tell the user it waits in "
        "Drafts. Send only when the user asked to SEND, not just to write. NO REPEATS: before writing "
        "check list_drafts query='to:<recipient>' (and get_thread for a reply); create_draft / send "
        "refuse a repeat themselves ('— repeat of …'): that is not a failure, don't retry; "
        "allow_repeat=true only if the user explicitly wants another copy. A new edition of a "
        "recurring email is not a repeat: date its subject. Replies pass reply_to_message_id "
        "(reply_all=true for everyone) — the inbox, recipients, 'Re:' subject and threading follow "
        "from it. SIGNATURE: the inbox's signature is added automatically — never write it yourself. "
        "ATTACHMENTS: read_message lists them numbered; get_attachment saves them. To attach files "
        "pass attachments=[paths] (paths a tool printed, saved/ paths or vfs:<project>/<path>), up "
        "to 25 MB. update_draft gives the draft a NEW Draft ID (use the one it prints). "
        "ORGANIZE ONLY WHEN ASKED: archive / move / mark_read / mark_unread / create_folder only "
        "when the user asked; find the emails first and pass message_ids (or thread_ids = the whole "
        "conversation's Inbox messages). A move gives each email a NEW ID (printed). Never to Trash, "
        "Junk, Drafts or Sent. ROUTING: any request about these inboxes MUST use imap_mail — never "
        "filesystem tools, shell or send_mail. "
        "Actions: list_accounts, list_messages (newest first; folder default Inbox; unread_only), "
        "search (from:ana to:x subject:\"Q3 report\" since:2026-01-01 before: newer_than:7d is:unread "
        "is:flagged has:attachment -from:noreply plus free text; folder default Inbox, folder='all' "
        "for every folder), read_message, get_thread, list_folders, list_drafts (query: 'to:<addr>' "
        "or subject words), get_attachment, create_draft, update_draft, send, send_draft, archive, "
        "move (folder=<destination>), mark_read, mark_unread, create_folder (folder='Parent/Child')."
    )
    timeout_seconds = 300.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS), "description": "The action to perform."},
            "account": {
                "type": "string",
                "description": (
                    "Which inbox: its id, email or label from list_accounts; 'all' for list_messages / "
                    "search / list_drafts / list_folders across every inbox. Needed for writes when the "
                    "user has several inboxes (not when an ID or reply_to_message_id says which)."
                ),
            },
            "folder": {
                "type": "string",
                "description": (
                    "list_messages / search: Inbox (default), Sent, Drafts, Archive, Junk, a folder name "
                    "or path from list_folders; search also takes 'all'. move: the destination. "
                    "create_folder: the new folder ('Clients/Acme' makes it inside Clients)."
                ),
            },
            "query": {"type": "string", "description": "search: the query. list_drafts: 'to:<address>' or subject words."},
            "unread_only": {"type": "boolean", "description": "list_messages: only unread mail."},
            "message_id": {"type": "string", "description": "Message ID (read_message, get_attachment)."},
            "message_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "archive / move / mark_read / mark_unread: Message IDs (at most 100).",
            },
            "thread_id": {"type": "string", "description": "get_thread: the Thread ID list/search/read printed."},
            "thread_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "archive / move / mark_read / mark_unread: Thread IDs — each acts on that "
                               "conversation's messages in the Inbox.",
            },
            "attachment": {"type": "string", "description": "get_attachment: number from read_message, name, or 'all'."},
            "to": {"type": "string", "description": "Recipients (comma-separated)."},
            "cc": {"type": "string", "description": "CC recipients (comma-separated)."},
            "bcc": {"type": "string", "description": "BCC recipients (comma-separated)."},
            "subject": {"type": "string", "description": "Subject (a reply defaults to 'Re: <original>')."},
            "body": {"type": "string", "description": "Plain-text body (update_draft: the full new text)."},
            "html_body": {"type": "string", "description": "Optional HTML body instead of body."},
            "reply_to_message_id": {
                "type": "string",
                "description": "create_draft / send: the Message ID you reply to (threads the reply, "
                               "fills recipients and subject).",
            },
            "reply_all": {"type": "boolean", "description": "With reply_to_message_id: reply to everyone."},
            "draft_id": {"type": "string", "description": "update_draft / send_draft: the Draft ID."},
            "attachments": {
                "type": "array", "items": {"type": "string"},
                "description": "create_draft / update_draft / send: files to attach (paths). At most 20, 25 MB.",
            },
            "remove_attachments": {
                "type": "array", "items": {"type": "string"},
                "description": "update_draft: attachments to drop — names, numbers, or 'all'.",
            },
            "allow_repeat": {
                "type": "boolean",
                "description": "create_draft / send: true ONLY when the user explicitly asked for another "
                               "copy of an email already drafted or sent.",
            },
            "max_results": {"type": "number", "description": "Maximum results (default 10, max 50)."},
            "include_body": {"type": "boolean", "description": "list_messages / search: add a preview of each."},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._client = ImapMailClient()

    # ── entry ───────────────────────────────────────────────────────

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        runtime = pop_runtime(kwargs)
        runtime["_workflow_run_dir"] = kwargs.pop("_workflow_run_dir", None)
        action = str(action or "").strip()
        if action in ("reply", "forward", "delete", "trash"):
            return ToolResult(success=False, error=(
                "imap_mail has no such action. Reply with create_draft (or send) and "
                "reply_to_message_id; imap_mail never deletes email."))
        if action not in _ACTIONS:
            return ToolResult(success=False, error=f"Unknown action '{action}'. Use one of: {', '.join(_ACTIONS)}")

        if action in _WRITE_ACTIONS:
            account = _write_account(kwargs)
            if mail_authority.needs_recipient_check():
                rcpt = (None if action not in ("create_draft", "send") or kwargs.get("reply_to_message_id")
                        else mail_authority.parse_recipients(kwargs.get("to"), kwargs.get("cc"), kwargs.get("bcc")))
                own = {str(a.get("email") or "").lower() for a in cached_accounts()}
                refusal = mail_authority.check_mail_write(self.name, action, recipients=rcpt,
                                                          own_addresses=own, account=account)
            else:
                refusal = mail_authority.check_mail_write(self.name, action, account=account)
            if refusal:
                log.info("imap_mail write refused (automated turn)", action=action,
                         kind=mail_authority.current().kind)
                return ToolResult(success=False, error=refusal)
        if action in _ORGANIZE_ACTIONS:
            refusal = mail_authority.check_mailbox_change(self.name, action)
            if refusal:
                log.info("imap_mail mailbox change refused (automated turn)", action=action,
                         kind=mail_authority.current().kind)
                return ToolResult(success=False, error=refusal)

        payload = {k: v for k, v in kwargs.items() if not str(k).startswith("_") and v is not None}
        raw_atts = payload.pop("attachments", None)
        if action in _FILE_ACTIONS:
            try:
                values = mail_attachments.parse_list(raw_atts)
                if action == "update_draft":
                    payload["remove_attachments"] = mail_attachments.parse_list(payload.get("remove_attachments"))
                if values:
                    atts = await mail_attachments.load(
                        values, runtime, max_total_bytes=mail_attachments.OUTLOOK_MAX_TOTAL_BYTES,
                        limit_label="attachments are limited to 25 MB per email")
                    payload["attachments"] = mail_attachments.to_wire(atts)
            except mail_attachments.AttachmentError as exc:
                nothing = "Nothing was sent." if action == "send" else "Nothing was saved."
                return ToolResult(success=False, error=f"{exc} {nothing}")
        for key in ("max_results",):
            if key in payload:
                payload[key] = limit(payload[key], 10, 50)
        for key in ("unread_only", "include_body", "reply_all", "allow_repeat"):
            if key in payload:
                payload[key] = flag(payload[key])

        try:
            status, data = await self._client.call(
                action, payload, timeout=240 if payload.get("attachments") or action == "get_attachment" else 120)
        except ImapRefused as exc:
            return ToolResult(success=False, error=str(exc))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            return ToolResult(success=False, error=f"Could not reach Flight Deck ({exc}). Nothing was changed.")
        except httpx.HTTPError as exc:
            if action in _SEND_ACTIONS:
                return ToolResult(success=False, error=(
                    f"Flight Deck did not answer the send ({type(exc).__name__}) — the email may or may "
                    "not have been sent. Do not resend blindly: check the inbox's Sent folder "
                    "(imap_mail action=list_messages folder=Sent) first."))
            return ToolResult(success=False, error=f"Flight Deck did not answer ({type(exc).__name__}). Try again.")

        if status != 200:
            detail = data.get("detail")
            detail = detail if isinstance(detail, str) and detail else f"HTTP {status}"
            if action in _SEND_ACTIONS:
                if status == 504:
                    return ToolResult(success=False, error=f"Email may or may not have been sent: {detail}")
                return ToolResult(success=False, error=f"Email not sent: {detail}")
            if status == 404 and "No email inbox" in detail:
                return ToolResult(success=False, error=NOT_CONNECTED)
            return ToolResult(success=False, error=detail)
        try:
            render = getattr(self, f"_render_{action}")
            if action == "get_attachment":
                return await render(data, kwargs, runtime)
            return render(data, kwargs)
        except Exception as exc:  # a formatting slip must not hide what happened
            log.error("imap_mail render failed", action=action, error=str(exc))
            return ToolResult(success=True, content=str(data)[:20_000])

    # ── rendering ───────────────────────────────────────────────────

    @staticmethod
    def _errors(data: dict[str, Any]) -> list[str]:
        return [f"  (inbox {e.get('account')} couldn't be read: {e.get('error')})" for e in data.get("errors") or []]

    def _render_list_accounts(self, data: dict[str, Any], kw: dict[str, Any]) -> ToolResult:
        accts = data.get("accounts") or []
        if not accts:
            return ToolResult(success=False, error=NOT_CONNECTED)
        lines = [f"{len(accts)} inbox{'es' if len(accts) != 1 else ''}:"]
        for a in accts:
            if not a.get("can_send"):
                send = "receive-only"
            elif a.get("sending_enabled"):
                rest = (f", only to {', '.join(a['allowed_recipients'][:5])}" if a.get("allowed_recipients") else "")
                send = f"sending on (up to {a.get('daily_limit')}/day{rest})"
            else:
                send = "drafts only (sending off)"
            label = f" — {a['label']}" if a.get("label") else ""
            lines.append(f"• account={a['id']}: {a.get('email')}{label} · {send}")
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    def _summary(m: dict[str, Any], i: int, many: bool) -> str:
        tags = [t for t, on in (("unread", m.get("unread")), ("flagged", m.get("flagged")),
                                ("replied", m.get("answered")), ("attachments", m.get("has_attachments"))) if on]
        head = f"{i}. {m.get('date', '')} — From: {m.get('from') or '?'}"
        if m.get("folder", "").casefold() in ("sent", "drafts") or m.get("draft"):
            head += f" — To: {m.get('to') or '?'}"
        lines = [head + (f" [{', '.join(tags)}]" if tags else ""),
                 f"   Subject: {m.get('subject') or '(no subject)'}",
                 f"   Message ID: {m.get('id')}" + (f" · Inbox: {m.get('account')}" if many else "")]
        if m.get("thread_id"):
            lines.append(f"   Thread ID: {m['thread_id']}")
        if m.get("preview"):
            lines.append(f"   Preview: {m['preview']}")
        return "\n".join(lines)

    def _render_messages(self, data: dict[str, Any], what: str, hint: str) -> ToolResult:
        msgs = data.get("messages") or []
        many = len(data.get("accounts") or []) > 1
        where = data.get("folder") or "Inbox"
        inbox = "" if many else f" ({(data.get('accounts') or ['?'])[0]})"
        if not msgs:
            body = f"No {what} in {where}{inbox}."
            errs = self._errors(data)
            return ToolResult(success=True, content="\n".join([body, *errs]))
        lines = [f"{what.capitalize()} in {where}{inbox}: {len(msgs)} shown of {data.get('total', len(msgs))}."]
        lines += [self._summary(m, i, many) for i, m in enumerate(msgs, 1)]
        lines += self._errors(data)
        lines.append(hint)
        return ToolResult(success=True, content="\n".join(lines))

    def _followup(self) -> str:
        hint = ("\nNext: imap_mail action=read_message message_id=<Message ID> (full text), "
                "action=get_thread thread_id=<Thread ID> (the conversation).")
        if mail_authority.check_mail_write(self.name, "create_draft") is None:
            hint += (" Reply ONLY if the user asked: create_draft (or send, when sending is on) "
                     "reply_to_message_id=<Message ID> body=… — check list_drafts first.")
        if mail_authority.check_mailbox_change(self.name, "archive") is None:
            hint += " Archive / move / mark_read ONLY if the user asked."
        return hint

    def _render_list_messages(self, data, kw):
        return self._render_messages(data, "emails", self._followup())

    def _render_search(self, data, kw):
        return self._render_messages(data, f"emails matching {kw.get('query')!r}", self._followup())

    def _render_list_drafts(self, data, kw):
        return self._render_messages(
            data, "drafts", "\nThe Message ID of a draft is its Draft ID (update_draft / send_draft draft_id=…).")

    def _render_list_folders(self, data, kw):
        folders = data.get("folders") or []
        many = len(data.get("accounts") or []) > 1
        lines = [f"{len(folders)} folders:"]
        for f in folders:
            role = f" ({f['role']})" if f.get("role") else ""
            acct = f" · Inbox: {f.get('account')}" if many else ""
            lines.append(f"• {f['name']}{role}{'' if f.get('selectable', True) else ' [container]'}{acct}")
        lines += self._errors(data)
        return ToolResult(success=True, content="\n".join(lines))

    @staticmethod
    def _detail(m: dict[str, Any], *, full: bool = True) -> list[str]:
        flags = [t for t, on in (("unread", m.get("unread")), ("flagged", m.get("flagged")),
                                 ("replied", m.get("answered")), ("draft", m.get("draft"))) if on]
        lines = [f"Message ID: {m.get('id')}", f"Inbox: {m.get('account')} · Folder: {m.get('folder')}"]
        if m.get("thread_id"):
            lines.append(f"Thread ID: {m['thread_id']}")
        lines += [f"From: {m.get('from')}", f"To: {m.get('to')}"]
        for label, key in (("Cc", "cc"), ("Bcc", "bcc"), ("Reply-To", "reply_to")):
            if m.get(key):
                lines.append(f"{label}: {m[key]}")
        lines += [f"Date: {m.get('date')}", f"Subject: {m.get('subject') or '(no subject)'}"]
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")
        atts = m.get("attachments") or []
        if atts:
            lines.append("Attachments:")
            lines += [f"  {a['index']}. {a['name']} ({'inline image, ' if a.get('inline') else ''}"
                      f"{a.get('mime_type')}, {_human(a.get('size', 0))})" for a in atts]
        if full and m.get("list_unsubscribe"):
            lines.append(f"List-Unsubscribe: {m['list_unsubscribe']}")
        lines.append("Body:" + (" (from HTML)" if m.get("body_from_html") else ""))
        lines.append(m.get("body") or "(empty)")
        if m.get("body_truncated"):
            lines.append("[… body truncated]")
        return lines

    def _render_read_message(self, data, kw):
        m = data.get("message") or {}
        lines = self._detail(m)
        if m.get("attachments"):
            lines.append("\nSave attachments with imap_mail action=get_attachment message_id=<Message ID> "
                         "attachment=<number or 'all'>.")
        return ToolResult(success=True, content="\n".join(lines))

    def _render_get_thread(self, data, kw):
        msgs = data.get("messages") or []
        out = [f"Thread {data.get('thread_id')}: {len(msgs)} message{'s' if len(msgs) != 1 else ''}, oldest first."]
        for i, m in enumerate(msgs, 1):
            out.append(f"\n── {i} ──")
            out += self._detail(m, full=False)
        return ToolResult(success=True, content="\n".join(out))

    @staticmethod
    def _files_line(files: list[dict[str, Any]]) -> str:
        if not files:
            return ""
        return "Attachments: " + ", ".join(f"{f['filename']} ({_human(f.get('size', 0))})" for f in files)

    def _render_create_draft(self, data, kw):
        lines = [f"Draft saved in {data.get('account')}'s Drafts — not sent.",
                 f"Draft ID: {data.get('draft_id') or '(the server gave none — list_drafts shows it)'}",
                 f"To: {data.get('to')}"]
        if data.get("cc"):
            lines.append(f"Cc: {data['cc']}")
        if data.get("bcc"):
            lines.append(f"Bcc: {data['bcc']}")
        lines.append(f"Subject: {data.get('subject')}")
        files = self._files_line(data.get("attachments") or [])
        if files:
            lines.append(files)
        lines.append("Tell the user it's waiting in Drafts. To send it (only if asked, and only with "
                     "sending on for this inbox): imap_mail action=send_draft draft_id=<Draft ID>.")
        return ToolResult(success=True, content="\n".join(lines))

    def _render_update_draft(self, data, kw):
        lines = ["Draft updated — not sent.",
                 f"New Draft ID: {data.get('draft_id')} (the old ID {data.get('old_draft_id')} is gone — use the new one)",
                 f"To: {data.get('to')}", f"Subject: {data.get('subject')}"]
        files = self._files_line(data.get("attachments") or [])
        if files:
            lines.append(files)
        if data.get("removed"):
            lines.append(f"Removed: {', '.join(data['removed'])}")
        if data.get("note"):
            lines.append(f"Note: {data['note']}")
        return ToolResult(success=True, content="\n".join(lines))

    def _render_send(self, data, kw):
        lines = [f"Email sent. From: {data.get('from')} (inbox {data.get('account')})", f"To: {data.get('to')}"]
        if data.get("cc"):
            lines.append(f"Cc: {data['cc']}")
        if data.get("bcc"):
            lines.append(f"Bcc: {data['bcc']}")
        lines += [f"Subject: {data.get('subject')}", f"Message-ID: {data.get('message_id')}"]
        files = self._files_line(data.get("attachments") or [])
        if files:
            lines.append(files)
        lines.append(f"Sent from this inbox in the last 24h: {data.get('sent_last_24h')} / {data.get('daily_limit')}")
        for note in data.get("notes") or []:
            lines.append(f"Note: {note}")
        return ToolResult(success=True, content="\n".join(lines))

    _render_send_draft = _render_send

    def _render_organize(self, data: dict[str, Any], verb: str) -> ToolResult:
        if "moved" in data:
            moved = data.get("moved") or []
            if not moved and not data.get("skipped"):
                return ToolResult(success=True, content="Nothing matched — no email was moved.")
            lines = [f"{verb} {len(moved)} email{'s' if len(moved) != 1 else ''} to {data.get('destination')}."]
            for old, new in moved[:20]:
                lines.append(f"  {old} → {new or '(new ID unknown — list_messages in that folder)'}")
            if len(moved) > 20:
                lines.append(f"  … and {len(moved) - 20} more")
            if data.get("skipped"):
                lines.append(f"Already there / not moved: {', '.join(data['skipped'][:10])}")
            lines.append("Moved emails have NEW Message IDs (above).")
            return ToolResult(success=True, content="\n".join(lines))
        changed = data.get("changed") or []
        return ToolResult(success=True, content=f"{verb} {len(changed)} email{'s' if len(changed) != 1 else ''}.")

    def _render_archive(self, data, kw):
        return self._render_organize(data, "Archived")

    def _render_move(self, data, kw):
        return self._render_organize(data, "Moved")

    def _render_mark_read(self, data, kw):
        return self._render_organize(data, "Marked as read:")

    def _render_mark_unread(self, data, kw):
        return self._render_organize(data, "Marked as unread:")

    def _render_create_folder(self, data, kw):
        if data.get("created"):
            return ToolResult(success=True, content=f"Created folder {data.get('folder')}.")
        return ToolResult(success=True, content=f"Folder {data.get('folder')} already exists — nothing created.")

    async def _render_get_attachment(self, data: dict[str, Any], kw: dict[str, Any],
                                     runtime: dict[str, Any]) -> ToolResult:
        from captain_claw import saved_attribution
        from captain_claw import speaker as _speaker
        from captain_claw.tools.google_drive import (
            _READER_BY_SUFFIX,
            GoogleDriveTool,
            _DownloadPathError,
        )
        from captain_claw.tools.google_mail import GoogleMailTool

        files = [f for f in data.get("files") or [] if isinstance(f, dict)]
        if not files:
            return ToolResult(success=False, error="No attachment came back.")
        names = GoogleMailTool._attachment_save_names(
            [{"filename": f.get("filename") or "", "mime_type": f.get("mime_type") or ""} for f in files])
        folder = "mail-" + hashlib.sha1(str(kw.get("message_id") or "").encode("utf-8")).hexdigest()[:16]
        lines: list[str] = []
        saved = 0
        for f, save_name in zip(files, names):
            try:
                blob = base64.b64decode(f.get("data") or "", validate=True)
            except Exception:
                lines.append(f"'{f.get('filename')}' came back unreadable.")
                continue
            if not blob or len(blob) > _MAX_ATTACHMENT_BYTES:
                lines.append(f"'{f.get('filename')}' is empty or too large ({_human(len(blob))}).")
                continue
            try:
                dest = GoogleDriveTool._download_dest(None, f"{folder}/{save_name}", "", runtime)
            except _DownloadPathError as exc:
                lines.append(str(exc))
                continue
            prior = saved_attribution.prior_creator(dest)
            member = _speaker.current()
            if (member is not None and dest.exists()
                    and not saved_attribution.member_may_change(dest, member.speaker_id)):
                lines.append(_speaker.PATH_REFUSED_PREFIX + _speaker.FILE_NOT_YOURS_WHY)
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(dest.write_bytes, blob)
            except OSError as exc:
                lines.append(f"'{f.get('filename')}' could not be saved ({exc.strerror or type(exc).__name__}).")
                continue
            saved_attribution.note_write(dest, prior)
            registry = runtime.get("_file_registry")
            if registry is not None:
                try:
                    registry.register(logical_path=dest.name, physical_path=str(dest),
                                      task_id=str(runtime.get("_task_id") or ""))
                except Exception:
                    pass
            reader = _READER_BY_SUFFIX.get(dest.suffix.lower(), "read")
            saved += 1
            # "Saved to:", not "Path:" — chat bridges auto-send files named on a "Path:" line.
            lines.append(f"Saved attachment '{f.get('filename')}' ({f.get('mime_type') or 'unknown'}, "
                         f"{len(blob)} bytes)\n  Saved to: {dest}\nUse {reader}(path=\"{dest}\") to view it.")
        if not saved:
            return ToolResult(success=False, error="\n".join(lines) or "Nothing was saved.")
        return ToolResult(success=True, content="\n".join(lines))


__all__ = ["ImapMailTool", "WHERE"]
