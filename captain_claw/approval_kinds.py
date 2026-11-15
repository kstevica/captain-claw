"""Which tool calls are a kind a person may want to approve first.

Six kinds: paying, sending a message to someone, posting publicly, sharing,
deleting, and storing or changing a password or key. A call's kind comes from
the tool's name and its ``action`` (and, for a browser click, the label of
the button it presses) — never from what the agent says about it or what the
message says, so nothing the agent writes can talk a call out of its kind.

Flight Deck decides what each kind needs on a deck (``flight_deck/approvals``);
the agent asks before such a call (``approvals_client``). Pure Python, so both
sides import it.
"""

from __future__ import annotations

import json
import re
from typing import Any

KINDS = ("pay", "send", "post", "share", "delete", "credentials")

KIND_LABELS = {
    "pay": "Pay or spend money",
    "send": "Send a message to someone",
    "post": "Post publicly",
    "share": "Share files, links or sites",
    "delete": "Delete things",
    "credentials": "Store or remove passwords",
}

# "<agent> wants to …", on the card.
KIND_VERBS = {
    "pay": "pay",
    "send": "send a message",
    "post": "post publicly",
    "share": "share something",
    "delete": "delete something",
    "credentials": "store or remove a password",
}

_SEND_ALWAYS = frozenset({"send_mail", "whatsapp_send_file"})
_SEND_ACTIONS = {
    "google_mail": frozenset({"send", "send_draft", "reply", "forward"}),
    "microsoft_mail": frozenset({"send", "send_draft", "reply", "forward"}),
    "imap_mail": frozenset({"send", "send_draft", "reply", "forward"}),
    "whatsapp_web": frozenset({"send"}),
    "twitter": frozenset({"dm"}),
}
_POST_ACTIONS = {"twitter": frozenset({"post", "reply", "retweet"})}
_SHARE_ACTIONS = {"public_link": frozenset({"share"}), "hosting": frozenset({"publish"})}
_PAY_ACTIONS = {"payments": frozenset({"purchase", "checkout_pay"}), "spend": frozenset({"authorize", "settle"})}
_DELETE_ACTIONS = {
    "microsoft_mail": frozenset({"trash", "delete"}),
    "google_mail": frozenset({"trash", "delete"}),
    "imap_mail": frozenset({"trash", "delete"}),
    "vfs": frozenset({"rm", "delete"}),
    "google_calendar": frozenset({"delete_event"}),
    "microsoft_calendar": frozenset({"delete_event"}),
    "contacts": frozenset({"remove"}),
    "datastore": frozenset({"delete", "drop_table", "drop_column"}),
    "google_drive": frozenset({"delete", "trash"}),
    "microsoft_onedrive": frozenset({"delete", "trash"}),
}
_CREDENTIAL_ACTIONS = {"browser": frozenset({"credentials_store", "credentials_delete"})}

_SQL_DELETE = re.compile(r"^\s*(?:delete|drop|truncate)\b", re.IGNORECASE)

# A browser click on a button with one of these labels. Whole label, so
# "Send feedback form" or "Delete filter" don't count, "Send" and "Pay now" do.
_BUTTON_KINDS = (
    ("pay", re.compile(r"^(?:pay(?: now)?|place (?:your )?order|buy(?: now)?|complete purchase|confirm (?:and )?pay(?:ment)?|checkout|submit payment)$", re.I)),
    ("send", re.compile(r"^(?:send|send (?:it|now|message|email|reply))$", re.I)),
    ("post", re.compile(r"^(?:post|publish|tweet|share post)$", re.I)),
    ("delete", re.compile(r"^(?:delete|remove|delete (?:it|forever|permanently|account)|empty trash)$", re.I)),
)
_CLICK_ACTIONS = frozenset({"click", "act"})
_LABEL_KEYS = ("text", "label", "name", "button")

# Words in an MCP tool's name (mcp_<server>_<tool>), first match wins.
_MCP_WORDS = (
    ("pay", frozenset({"pay", "payment", "charge", "transfer", "purchase", "checkout"})),
    ("delete", frozenset({"delete", "trash", "remove", "destroy", "purge", "drop"})),
    ("share", frozenset({"share", "invite", "permission", "permissions", "publish"})),
    ("post", frozenset({"post", "tweet", "toot"})),
    ("send", frozenset({"send", "reply", "forward", "message", "dm", "sms", "email"})),
)
_MCP_READ = frozenset({"get", "list", "search", "read", "fetch", "query", "find", "describe", "help"})


def _action(params: Any) -> str:
    if not isinstance(params, dict):
        return ""
    value = params.get("action")
    return value.strip().lower() if isinstance(value, str) else ""


def _button_label(params: dict[str, Any]) -> str:
    for key in _LABEL_KEYS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split()).strip(" .!")
    return ""


def kind_for(tool: str, params: Any = None) -> str:
    """The kind of a call to *tool* with *params*, or "" when it's none of them."""
    name = str(tool or "").strip().lower()
    args = params if isinstance(params, dict) else {}
    action = _action(args)
    if name.startswith("mcp_"):
        words = set(re.split(r"[^a-z0-9]+", name[4:])) - {""}
        if words & _MCP_READ and not any(words & w for _k, w in _MCP_WORDS):
            return ""
        for kind, vocab in _MCP_WORDS:
            if words & vocab:
                return kind
        return ""
    if name in _SEND_ALWAYS and action not in {"list", "status"}:
        return "send"
    for kind, table in (("pay", _PAY_ACTIONS), ("send", _SEND_ACTIONS), ("post", _POST_ACTIONS),
                        ("share", _SHARE_ACTIONS), ("delete", _DELETE_ACTIONS),
                        ("credentials", _CREDENTIAL_ACTIONS)):
        if action and action in table.get(name, frozenset()):
            return kind
    if name == "datastore" and action == "sql":
        statement = args.get("sql") or args.get("query") or args.get("statement") or ""
        if isinstance(statement, str) and _SQL_DELETE.match(statement):
            return "delete"
    if name in {"browser", "pinchtab"} and action in _CLICK_ACTIONS:
        label = _button_label(args)
        for kind, pattern in _BUTTON_KINDS:
            if label and pattern.match(label):
                return kind
    return ""


# Arguments worth showing on a card, in this order.
_SHOW_KEYS = (
    "to", "recipient", "recipients", "cc", "bcc", "chat", "contact", "phone", "subject", "title", "name",
    "amount", "currency", "merchant", "description", "path", "file", "file_path", "url", "site", "table",
    "event_id", "message_id", "id", "text", "body", "message", "content", "sql", "query", "label", "button",
)
_SHOW_MAX = 600


def _short(value: Any, limit: int = 160) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe(tool: str, params: Any = None) -> str:
    """What the call will do, in a few lines for the person: the tool, its
    action and the arguments that say who or what (passwords masked)."""
    from captain_claw.tool_audit import mask_params

    args = mask_params(params if isinstance(params, dict) else {})
    action = _action(args)
    head = f"{tool} · {action}" if action else str(tool)
    lines = [head]
    for key in _SHOW_KEYS:
        if key in args and args[key] not in (None, "", [], {}):
            lines.append(f"{key}: {_short(args[key])}")
    if len(lines) == 1:
        rest = {k: v for k, v in args.items() if k != "action"}
        if rest:
            lines.append(_short(rest, 300))
    text = "\n".join(lines)
    return text if len(text) <= _SHOW_MAX else text[: _SHOW_MAX - 1] + "…"
