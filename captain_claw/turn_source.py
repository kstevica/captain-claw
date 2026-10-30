"""Where a user message came from — the *source* recorded on it.

Every user message an agent stores carries its source: the web chat, the
Captain Claw app, smart glasses, WhatsApp, Telegram, a handheld device (and
which kind), a scheduled task, a flow, another agent … with a label a person
would read ("M5Stack StickS3 (stick)"). It sits beside the message's
``channel``, which stays the chat *surface* whose rules the turn renders
(glasses / whatsapp / messenger / device): the Android app talks over the
glasses surface, but its source is ``app``.

A source is ``<channel>`` or ``<channel>:<kind>`` (``device:sticks3``,
``glasses:meta-display``). The agent's per-channel reasoning effort is keyed
the same way and looked up most specific first (see
``llm.reasoning.configured_effort``).
"""

from __future__ import annotations

import re
from typing import Any

# The channels, as the settings list them.
SOURCE_LABELS: dict[str, str] = {
    "web": "Web chat",
    "app": "Captain Claw app",
    "glasses": "Smart glasses",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
    "messenger": "Messenger",
    "discord": "Discord",
    "device": "Device",
    "scheduled": "Scheduled tasks",
    "flow": "Flows",
    "agent": "Other agents",
    "autonomy": "Autonomous work",
    "taskboard": "Taskboard",
    "api": "API and MCP",
    "link": "Agent links",
    "exam": "Onboarding exams",
    "public": "Public link",
    "member": "Shared-agent members",
}

# Kinds with a name of their own.
KIND_LABELS: dict[str, str] = {
    "device:sticks3": "M5Stack StickS3",
    "device:stopwatch": "M5Stack StopWatch",
    "device:papercolor": "M5Stack PaperColor",
    "device:epaper154": "Waveshare ePaper 1.54",
    "device:epaper154_touch": "Waveshare Touch ePaper 1.54",
    "glasses:meta-display": "Meta Ray-Ban Display",
    "glasses:rokid-lumen": "Rokid glasses",
    "app:android": "Captain Claw app (Android)",
    "app:browser": "Captain Claw app (browser)",
}

# The groups the agent settings show, in order: (title, sources). Web chat is
# not among them — its effort is the chat box's picker.
SETTINGS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Messaging", ("whatsapp", "telegram", "messenger", "discord")),
    ("Apps and devices", ("app", "glasses", "device:sticks3", "device:stopwatch", "device:papercolor",
                          "device:epaper154", "device:epaper154_touch")),
    ("Automated", ("scheduled", "flow", "agent", "autonomy", "taskboard", "api", "link")),
    ("Other people", ("member", "public")),
)

# A turn's automation kind (mail_authority.AUTOMATION_KINDS) → its source.
_AUTOMATION_SOURCES = {
    "fd_scheduler": "scheduled", "cron": "scheduled",
    "flow": "flow", "flow_tool": "flow",
    "peer": "agent", "peer_relay": "agent", "sister": "agent", "botport": "agent",
    "fd_worker": "agent", "being": "agent",
    "autonomy": "autonomy", "autonomy_tool": "autonomy", "plan": "autonomy", "bat": "autonomy",
    "mcp_task": "api",
    "link": "link",
    "exam": "exam",
    "taskboard": "taskboard",
}
# A turn's channel (the surface or the entry point) → its source.
_CHANNEL_SOURCES = {
    "web": "web", "whatsapp": "whatsapp", "telegram": "telegram", "messenger": "messenger",
    "glasses": "glasses", "device": "device", "discord": "discord", "public": "public",
    "member": "member", "api": "api", "mcp": "api",
}
# A turn's origin (msg_origin) → its source, when nothing else says.
_ORIGIN_SOURCES = {
    "cron": "scheduled", "worker_task": "agent", "delegated_result": "agent", "fleet_notice": "agent",
}

_KIND_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_LABEL_MAX = 80


def normalize(source: Any) -> str:
    """A well-formed source (known channel, optional ``:kind``), else ""."""
    text = str(source or "").strip().lower()
    channel, _, kind = text.partition(":")
    if channel not in SOURCE_LABELS:
        return ""
    if not kind:
        return channel
    return f"{channel}:{kind}" if _KIND_RE.match(kind) else channel


def channel_of(source: str) -> str:
    return str(source or "").partition(":")[0]


def label_for(source: str, given: Any = "") -> str:
    """How a person names *source*: the label a bridge sent, else the kind's
    or the channel's own name."""
    text = " ".join(str(given or "").split())[:_LABEL_MAX]
    if text:
        return text
    if source in KIND_LABELS:
        return KIND_LABELS[source]
    return SOURCE_LABELS.get(channel_of(source), "")


def from_automation(kind: Any) -> str:
    return _AUTOMATION_SOURCES.get(str(kind or "").strip().lower(), "")


def for_frame(
    *,
    frame_source: Any = "",
    automation_kind: Any = "",
    surface: str | None = None,
    whatsapp: bool = False,
    public: bool = False,
) -> str:
    """The source of a chat frame's turn: an automated turn's kind first (a
    scheduled job delivering to WhatsApp is the job), then what the frame
    says of itself (the device bridge, the app), then its surface; a frame
    from WhatsApp without one, a public visitor, else the web chat."""
    if automation_kind:
        found = from_automation(automation_kind)
        if found:
            return found
    found = normalize(frame_source)
    if found:
        return found
    if surface:
        found = _CHANNEL_SOURCES.get(str(surface).strip().lower(), "")
        if found:
            return found
    if whatsapp:
        return "whatsapp"
    if public:
        return "public"
    return "web"


def resolve(*, hinted: Any = "", channel: Any = "", origin: Any = "", origin_detail: Any = "") -> str:
    """A turn's source as the agent records it: the caller's hint, else the
    automation kind behind it, its channel, or its origin; "" when nothing
    says (a CLI turn, an internal call)."""
    found = normalize(hinted)
    if found:
        return found
    found = from_automation(origin_detail)
    if found:
        return found
    found = _CHANNEL_SOURCES.get(str(channel or "").strip().lower(), "")
    if found:
        return found
    return _ORIGIN_SOURCES.get(str(origin or "").strip().lower(), "")


def settings_key(source: str) -> str:
    """The row of the agent's settings a person changes for *source*: the
    source itself when the settings list it, else its channel when they list
    that, else the source; "web" for the web chat (the chat box's picker)
    and "default" when nothing says where the turn came from."""
    found = normalize(source)
    if not found:
        return "default"
    channel = channel_of(found)
    if channel == "web":
        return "web"
    listed = {key for _title, keys in SETTINGS_GROUPS for key in keys}
    if found in listed:
        return found
    return channel if channel in listed else found


def settings_catalog() -> list[dict[str, Any]]:
    """The sources the agent settings offer, grouped, with their names."""
    return [
        {"title": title, "sources": [{"key": key, "label": label_for(key)} for key in keys]}
        for title, keys in SETTINGS_GROUPS
    ]
