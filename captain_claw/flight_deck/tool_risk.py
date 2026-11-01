"""How risky a tool call is, for the tool-call register's filters and colours.

Flight Deck decides this itself from the tool's name and its ``action`` (never
from anything the agent says), so a reader can ask for "everything that ran
code or sent something out in the last day":

* ``read``  — looks at something: files, the web, a mailbox, a table.
* ``write`` — changes something the user keeps: files, tables, drafts, events,
  memory, schedules.
* ``exec``  — runs code or drives a computer: shell, terminal, desktop, browser.
* ``send``  — reaches someone or something outside: email, WhatsApp, posts,
  other agents, published sites, outside APIs and MCP servers.
* ``money`` — payments and spending.
* ``other`` — anything not listed (a new tool lands here until it is).
"""

from __future__ import annotations

import re
from typing import Any

RISKS = ("read", "write", "exec", "send", "money", "other")

RISK_LABELS = {
    "read": "Read", "write": "Write", "exec": "Run code", "send": "Sends out",
    "money": "Money", "other": "Other",
}

# A tool's class when it has no action, or its action isn't a read.
_TOOL_RISK: dict[str, str] = {
    # look
    "read": "read", "glob": "read", "grep": "read", "web_search": "read", "web_fetch": "read",
    "web_fetch_batch": "read", "web_get": "read", "pdf_extract": "read", "docx_extract": "read",
    "pptx_extract": "read", "xlsx_extract": "read", "image_ocr": "read", "image_vision": "read",
    "video_vision": "read", "summarize_files": "read", "history": "read", "knowledge": "read", "sessions": "read",
    "workspace_read": "read", "codemap": "read", "cv": "read", "screen_capture": "read",
    "telegram_inbox": "read", "self_check": "read", "activity": "read", "researchmap": "read", "watch_background": "read",
    "audio": "read",
    # keep
    "write": "write", "edit": "write", "workspace_write": "write", "datastore": "write",
    "google_drive": "write", "microsoft_onedrive": "write", "google_calendar": "write",
    "microsoft_calendar": "write", "contacts": "write", "cron": "write", "todo": "write",
    "apis": "write", "typesense": "write", "insights": "write", "playbooks": "write",
    "personality": "write", "facts": "write", "intentions": "write", "project_memory": "write",
    "vfs": "write", "docx": "write", "xlsx": "write", "topics": "write", "image_gen": "write", "visualize": "write",
    "synthesize_flow": "write", "pocket_tts": "write", "clipboard": "write",
    "reasoning_effort": "write",
    # run
    "shell": "exec", "terminal": "exec", "desktop_action": "exec", "scripts": "exec",
    "app_runner": "exec", "code": "exec", "termux": "exec", "pinchtab": "exec", "browser": "exec",
    "vatra": "exec", "basna": "exec", "bat": "exec", "crew": "exec",
    # out
    "send_mail": "send", "google_mail": "write", "microsoft_mail": "write",
    "whatsapp_send_file": "send", "whatsapp_web": "send", "linked_system": "send",
    "botport": "send", "consult_peer": "send", "flight_deck": "send", "twitter": "send",
    "hosting": "write", "public_link": "send", "direct_api": "send", "jev": "send",
    "ask_human": "send",
    # money
    "payments": "money", "spend": "money",
}

# Actions that only look, whatever the tool.
_READ_ACTIONS = frozenset({
    "list", "search", "read", "get", "info", "describe", "query", "status", "show", "view",
    "me", "timeline", "mentions", "thread", "chats", "recent", "contact", "models", "observe",
    "screenshot", "download", "attachment", "get_attachment", "read_message", "get_thread",
    "list_messages", "list_labels", "list_drafts", "list_folders", "list_events", "search_events",
    "get_event", "list_calendars", "list_tables", "list_protections", "list_agents", "doc_read",
    "sheet_read", "changes", "export", "wait",
})
_READ_PREFIXES = ("list_", "get_", "search_", "read_")

# Actions that reach outside on a tool that is otherwise a "write".
_SEND_ACTIONS: dict[str, frozenset[str]] = {
    "google_mail": frozenset({"send", "send_draft", "reply", "forward"}),
    "microsoft_mail": frozenset({"send", "send_draft", "reply", "forward"}),
    "hosting": frozenset({"publish"}),
    "google_calendar": frozenset(),  # invites go out with a create, but it is the user's calendar
}

_MCP_WRITE_WORDS = frozenset({
    "send", "post", "create", "update", "delete", "write", "pay", "transfer", "publish", "reply",
    "forward", "trash", "remove", "upload", "add", "set", "put", "patch", "draft", "invite",
})
_MCP_READ_WORDS = frozenset({"get", "list", "search", "read", "fetch", "query", "find", "describe", "help"})


def _action(params: Any) -> str:
    if not isinstance(params, dict):
        return ""
    value = params.get("action")
    return str(value).strip().lower() if isinstance(value, str) else ""


def _is_read_action(action: str) -> bool:
    return action in _READ_ACTIONS or action.startswith(_READ_PREFIXES)


def risk_for(tool: str, params: Any = None) -> str:
    """The risk class of a call to *tool* with *params* (masked is fine)."""
    name = str(tool or "").strip().lower()
    if name.startswith("mcp_"):
        words = set(re.split(r"[^a-z0-9]+", name[4:]))
        if words & _MCP_WRITE_WORDS:
            return "send"
        return "read" if words & _MCP_READ_WORDS else "send"
    base = _TOOL_RISK.get(name, "other")
    action = _action(params)
    if not action:
        return base
    if _is_read_action(action):
        return "read"
    if action in _SEND_ACTIONS.get(name, frozenset()):
        return "send"
    if name == "datastore" and action == "sql":
        return "write"
    return base
