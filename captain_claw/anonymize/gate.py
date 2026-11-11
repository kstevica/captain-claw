"""The privacy gate: what it covers (shared by Flight Deck and the agent).

A person picks sources — their mail, chats, calendar, contacts, files, data,
memory — and, when the gate applies (only for an agent whose model isn't
trusted, or always), what those sources give an agent is anonymized before
the agent's model sees it: tool results go through the engine into the
person's "Privacy gate" vault. On the way out, tokens in a message the agent
sends (mail, WhatsApp) are put back just before it leaves, so the recipient
reads real names while the model never does.

This module only says which tools belong to which source and which tools send
things out; Flight Deck keeps the setting and decides
(:mod:`captain_claw.flight_deck.anonymize_gate`), the agent applies it
(:mod:`captain_claw.anonymize_gate_client`).
"""

from __future__ import annotations

GATE_VAULT = "Privacy gate"

# Source → the tools whose results come from it.
GROUPS: dict[str, frozenset[str]] = {
    "mail": frozenset({"google_mail", "microsoft_mail", "imap_mail"}),
    "chats": frozenset({"telegram_inbox", "whatsapp_web"}),
    "calendar": frozenset({"google_calendar", "microsoft_calendar"}),
    "contacts": frozenset({"contacts"}),
    "files": frozenset({"read", "grep", "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
                        "summarize_files", "image_ocr", "ocr_pdf", "ocr_image", "google_drive",
                        "microsoft_onedrive", "workspace_read", "knowledge"}),
    "data": frozenset({"datastore"}),
    "memory": frozenset({"external_memory", "typesense"}),
}

GROUP_LABELS = {
    "mail": "Mail (Gmail, Outlook, IMAP)",
    "chats": "Chats (Telegram, WhatsApp)",
    "calendar": "Calendar",
    "contacts": "Contacts",
    "files": "Files (what agents read; or only some folders)",
    "data": "Data tables",
    "memory": "Memory (external and deep memory)",
}

DEFAULT_GROUPS = ("mail", "chats", "calendar", "contacts")

# Source → the background notes an agent's prompt gets from it at the start
# of a turn (recall, briefings, file lists); while the gate covers the source
# they are left out — they would carry real values past the gated tools.
NOTE_GROUPS: dict[str, frozenset[str]] = {
    "mail": frozenset({"briefing_context"}),
    "calendar": frozenset({"briefing_context"}),
    "contacts": frozenset({"contacts_context"}),
    "files": frozenset({"knowledge_context", "workspace_manifest"}),
    "data": frozenset({"datastore_context"}),
    "memory": frozenset({"deep_memory_context", "external_memory_context", "external_memory_main"}),
}

# Tools that send something to people outside: tokens in their arguments are
# put back before they run (their own result is then anonymized again).
EGRESS_TOOLS = frozenset({
    "send_mail", "google_mail", "microsoft_mail", "imap_mail", "whatsapp_web", "whatsapp_send_file",
})

# Arguments of the files group that name what is read (for the folder filter).
PATH_ARGS = ("path", "file_path", "paths", "folder", "file", "local_path")

WHEN = ("untrusted", "always")
LANES = ("deterministic", "both")


def gated(name: str, groups: frozenset[str] | set[str]) -> bool:
    return any(name in GROUPS.get(g, ()) for g in groups)


def path_values(arguments: dict) -> list[str]:
    out: list[str] = []
    for key in PATH_ARGS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
        elif isinstance(value, list):
            out.extend(str(v).strip() for v in value if str(v or "").strip())
    return out


def under_folders(paths: list[str], folders: list[str]) -> bool:
    """Whether any of *paths* lies under one of *folders* (plain prefixes,
    as the person wrote them: ``vfs:clients/``, ``/Users/me/Clients``)."""
    norm = [f.rstrip("/") for f in folders if f.strip()]
    for p in paths:
        for f in norm:
            if p == f or p.startswith(f + "/"):
                return True
    return False
