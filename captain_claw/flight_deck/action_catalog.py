"""The Action Catalog — the curated, named set of real-world actions the
autonomous loop may take (see docs/jarvis-actions-events-plan.md, #1).

This is NOT every tool. It's a vetted subset, each entry carrying the metadata
the loop needs to act safely:

  * ``risk``           — low | normal | high
  * ``reversibility``  — read_only | reversible | irreversible
  * ``reverse``        — how to undo a reversible action (built from the result)
  * ``human_only``     — never auto-dispatched, by grant or by earned trust
                          (irreversible / outward-facing / any email write)
  * ``proposable``     — the arbiter / plans may PROPOSE a ``human_only``
                          action (human approval, never auto). Without it a
                          ``human_only`` action is never proposed either.
  * ``grant``          — which per-user grant must be enabled to use it

Hard rule: raw ``shell``, ``browser`` form-submit, social posting, and anything
payment-like are deliberately ABSENT — those stay in normal agent chat and the
autonomous loop can never reach them.

Each entry maps an ``action_id`` to a concrete tool call: ``base_args`` (fixed,
e.g. the tool's action sub-type) merged with the validated user/agent args.
"""

from __future__ import annotations

import re
from typing import Any

# Both the cron and calendar tools print the created id as "ID: <value>", so one
# extractor pulls the reverse handle from a ToolResult's text content.
_ID_RE = re.compile(r"\bID:\s*(\S+)")

# action_id → spec
CATALOG: dict[str, dict[str, Any]] = {
    # ── Reversible / staging · auto-eligible (low risk) ──────────────────
    "note.write": {
        "label": "Write a note to the workspace",
        "home": "agent",
        "tool": "write",
        "base_args": {},
        "required": ["path", "content"],
        "optional": ["append"],
        "risk": "low",
        "reversibility": "reversible",   # prior content is backed up before write (Phase 2)
        "reverse": None,
        # Grounded verification (#5): read the file back to confirm it's really there.
        "verify": {"tool": "read", "arg_from_args": {"path": "path"}},
        "grant": "notes",
    },
    "calendar.hold": {
        "label": "Create a (tentative) calendar hold",
        "home": "agent",
        "tool": "google_calendar",
        "base_args": {"action": "create_event"},
        "required": ["summary", "start", "end"],
        "optional": ["description", "location", "calendar_id"],
        "risk": "low",
        "reversibility": "reversible",
        # The created event id comes back in the ToolResult; the reverse deletes it.
        "reverse": {"tool": "google_calendar", "base_args": {"action": "delete_event"},
                    "args_from_result": {"event_id": "id"}},
        # Read the created event back by id to confirm it actually landed.
        "verify": {"tool": "google_calendar", "base_args": {"action": "get_event"},
                   "arg_from_result": {"event_id": "id"}},
        "grant": "calendar",
    },
    "reminder.schedule": {
        "label": "Schedule a reminder / recurring nudge",
        "home": "agent",
        "tool": "cron",
        "base_args": {"action": "create"},
        "required": ["schedule", "task"],
        "optional": [],
        "risk": "low",
        "reversibility": "reversible",
        "reverse": {"tool": "cron", "base_args": {"action": "remove"},
                    "args_from_result": {"job_id": "id"}},
        "grant": "reminders",
    },

    # ── In catalog but HUMAN-ONLY (irreversible / outward-facing / email) ─
    # Never auto-dispatched — not by a grant, not by earned trust (#3). Only an
    # entry that is also ``proposable`` may be proposed for the user's approval.
    "mail.draft": {
        "label": "Create an email reply draft (not sent) — always needs your approval",
        "home": "agent", "tool": "google_mail", "base_args": {"action": "create_draft"},
        "required": ["to", "subject", "body"],
        "optional": ["cc", "bcc", "html_body", "reply_to_message_id"],
        "risk": "low", "reversibility": "reversible", "reverse": None,
        "grant": "mail", "human_only": True, "proposable": True,
    },
    "mail.send": {
        "label": "Send an email", "home": "agent", "tool": "google_mail",
        "base_args": {"action": "send"}, "required": ["to", "subject", "body"],
        "optional": ["cc", "bcc", "html_body", "reply_to_message_id"],
        "risk": "high", "reversibility": "irreversible", "reverse": None,
        "grant": "mail", "human_only": True,
    },
    "calendar.invite": {
        "label": "Create a calendar event with attendees", "home": "agent",
        "tool": "google_calendar", "base_args": {"action": "create_event"},
        "required": ["summary", "start", "end", "attendees"], "optional": ["description", "location"],
        "risk": "high", "reversibility": "reversible",
        "reverse": {"tool": "google_calendar", "base_args": {"action": "delete_event"},
                    "args_from_result": {"event_id": "id"}},
        "grant": "calendar", "human_only": True,   # notifies others → human-gated
    },
    "calendar.delete": {
        "label": "Delete a calendar event", "home": "agent", "tool": "google_calendar",
        "base_args": {"action": "delete_event"}, "required": ["event_id"], "optional": ["calendar_id"],
        "risk": "high", "reversibility": "irreversible", "reverse": None,
        "grant": "calendar", "human_only": True,
    },
    "message.send": {
        "label": "Send a message to a contact", "home": "agent", "tool": "whatsapp_send_file",
        "base_args": {"action": "send_text"}, "required": ["to", "text"], "optional": [],
        "risk": "high", "reversibility": "irreversible", "reverse": None,
        "grant": "messaging", "human_only": True,
    },
    # (No drive.delete: the google_drive tool has no delete action, so the
    # entry that used to sit here dispatched to a nonexistent action and always
    # errored. Drive access is read-oriented; deletion isn't offered.)
}


# google_mail sub-actions that write email (create/change/send a draft or mail).
MAIL_WRITE_ACTIONS = frozenset({"create_draft", "update_draft", "send", "send_draft"})


# A Gmail-like MCP proxy tool (``mcp_<server>_<tool>``, a user-promoted custom
# action) that writes email. These verbs mean email on their own …
_MCP_MAIL_VERBS = ("create_draft", "update_draft", "send_draft", "send_email")
# … these only when the tool's name also says mail.
_MCP_MAYBE_MAIL_VERBS = ("send_message", "reply", "forward", "compose", "draft")
_MCP_MAIL_WORDS = ("mail", "outlook")


def _is_mcp_mail_write(tool: str) -> bool:
    """Best-effort FD-side twin of the agent's MCP mail guard (part 0 J18): FD
    knows only the tool's name, so a borderline one just needs approval."""
    if not tool.startswith("mcp_"):
        return False
    name = tool[4:]
    if any(v in name for v in _MCP_MAIL_VERBS):
        return True
    return (any(w in name for w in _MCP_MAIL_WORDS)
            and any(v in name for v in _MCP_MAYBE_MAIL_VERBS))


def is_mail_write(spec: dict[str, Any] | None) -> bool:
    """True for any spec that writes email: send_mail, or google_mail whose fixed
    action is a write — or a google_mail spec with no fixed action (the action
    would come from args, so assume a write). A custom action over a Gmail-like
    MCP proxy tool (``mcp_gmail_create_draft`` …) counts too."""
    if not spec:
        return False
    tool = str(spec.get("tool") or "").strip().lower()
    if tool == "send_mail":
        return True
    if _is_mcp_mail_write(tool):
        return True
    if tool != "google_mail":
        return False
    action = str((spec.get("base_args") or {}).get("action") or "").strip().lower()
    return not action or action in MAIL_WRITE_ACTIONS


def may_propose(spec: dict[str, Any] | None) -> bool:
    """Whether the arbiter / a plan may propose this action at all: any non-
    ``human_only`` action, or a ``human_only`` one flagged ``proposable`` (it
    then always waits for the user's approval)."""
    return bool(spec) and (not spec.get("human_only") or bool(spec.get("proposable")))


def _is_excluded(tool: str) -> bool:
    """A tool that the autonomous loop may NEVER drive (the one hard wall)."""
    from captain_claw.config import AUTONOMY_HARD_EXCLUDE
    t = (tool or "").lower()
    return any(x in t for x in AUTONOMY_HARD_EXCLUDE)


def _custom_to_spec(ca: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a user CustomAction (Theme A) into a catalog spec. Returns None
    if disabled, missing a tool, or the tool hits the hard-exclude wall."""
    tool = str(ca.get("tool") or "").strip()
    if not tool or not ca.get("enabled", True) or _is_excluded(tool):
        return None
    spec: dict[str, Any] = {
        "label": ca.get("label") or ca.get("id"),
        "home": "agent",
        "tool": tool,
        "base_args": dict(ca.get("base_args") or {}),
        "required": list(ca.get("required") or []),
        "optional": list(ca.get("optional") or []),
        "risk": ca.get("risk") or "normal",
        "reversibility": ca.get("reversibility") or "irreversible",
        "reverse": None,
        "grant": ca.get("grant") or "custom",
        "human_only": bool(ca.get("human_only", True)),
        "custom": True,
    }
    rt = str(ca.get("reverse_tool") or "").strip()
    if rt and not _is_excluded(rt):
        spec["reverse"] = {"tool": rt, "base_args": {}, "args_from_result": {"id": "id"}}
    if is_mail_write(spec):
        # A user can't widen a custom email action: it never auto-fires and is
        # never proposed (proposable stays False).
        spec["human_only"] = True
    return spec


def resolve_catalog(user_id: str = "") -> dict[str, dict[str, Any]]:
    """The built-in catalog merged with ``user_id``'s enabled custom actions.
    Custom actions can never shadow a built-in id, and any hitting the hard-
    exclude wall are dropped. With no user_id, just the built-ins."""
    cat = dict(CATALOG)
    if not user_id:
        return cat
    try:
        from captain_claw.flight_deck.autonomy import resolve_config
        for ca in resolve_config(user_id).get("custom_actions") or []:
            aid = str(ca.get("id") or "").strip()
            if not aid or aid in cat:   # never shadow a built-in
                continue
            spec = _custom_to_spec(ca)
            if spec:
                cat[aid] = spec
    except Exception:
        pass
    return cat


def get_action(action_id: str, user_id: str = "") -> dict[str, Any] | None:
    return resolve_catalog(user_id).get(str(action_id or "").strip())


def list_catalog(*, granted: set[str] | None = None, user_id: str = "") -> list[dict[str, Any]]:
    """Catalog entries (id + safe metadata) for the UI / arbiter prompt. When
    ``granted`` is given (the user's enabled grants), only those are returned.
    Includes the user's enabled custom actions when ``user_id`` is supplied."""
    out: list[dict[str, Any]] = []
    for aid, spec in resolve_catalog(user_id).items():
        if granted is not None and spec.get("grant") not in granted:
            continue
        out.append({
            "id": aid, "label": spec["label"], "risk": spec["risk"],
            "reversibility": spec["reversibility"], "grant": spec.get("grant", ""),
            "human_only": bool(spec.get("human_only", False)),
            "proposable": bool(spec.get("proposable")),
            "mail_write": is_mail_write(spec),
            "args": spec["required"] + spec.get("optional", []),
            "required": spec["required"],
            "custom": bool(spec.get("custom", False)),
        })
    return out


def validate_args(spec: dict[str, Any], args: dict[str, Any]) -> tuple[bool, str]:
    """All required args present and non-empty. Returns (ok, error)."""
    if not isinstance(args, dict):
        return False, "args must be an object"
    for key in spec.get("required", []):
        v = args.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False, f"missing required arg: {key}"
    return True, ""


def build_tool_call(spec: dict[str, Any], args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Merge the fixed ``base_args`` with the recognised user args into a concrete
    ``(tool_name, tool_args)`` call. Unknown args are dropped."""
    merged: dict[str, Any] = dict(spec.get("base_args") or {})
    for key in spec.get("required", []) + spec.get("optional", []):
        if key in args and args[key] is not None:
            merged[key] = args[key]
    return spec["tool"], merged


def build_reverse(spec: dict[str, Any], result_content: str) -> dict[str, Any] | None:
    """Build the concrete reverse call ``{tool, args}`` for an action that just
    ran, extracting the id from its ToolResult ``result_content``. Returns None
    when the action has no reverse or the id can't be found."""
    rev = spec.get("reverse")
    if not rev:
        return None
    m = _ID_RE.search(result_content or "")
    if not m:
        return None
    extracted_id = m.group(1).strip().rstrip(".")
    args: dict[str, Any] = dict(rev.get("base_args") or {})
    for arg_name, _src in (rev.get("args_from_result") or {}).items():
        args[arg_name] = extracted_id  # only "id" is supported as the source today
    return {"tool": rev["tool"], "args": args}


def build_verify(spec: dict[str, Any], result_content: str, in_args: dict[str, Any]) -> dict[str, Any] | None:
    """Build the read-back call ``{tool, args}`` that confirms an action's side
    effect — from the created id (``arg_from_result``) and/or an input arg
    (``arg_from_args``). Returns None when the action has no verify spec or a
    needed value is missing."""
    v = spec.get("verify")
    if not v:
        return None
    args: dict[str, Any] = dict(v.get("base_args") or {})
    for arg_name, src in (v.get("arg_from_args") or {}).items():
        val = (in_args or {}).get(src)
        if val is None:
            return None
        args[arg_name] = val
    if v.get("arg_from_result"):
        m = _ID_RE.search(result_content or "")
        if not m:
            return None
        ext = m.group(1).strip().rstrip(".")
        for arg_name in v["arg_from_result"]:
            args[arg_name] = ext
    return {"tool": v["tool"], "args": args}
