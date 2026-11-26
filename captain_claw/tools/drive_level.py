"""drive_level tool: the agent reads and changes how far it pushes a turn.

It changes what a person changes in Flight Deck (``drive.py``). A web chat
turn's level is the chat box's Drive picker: Flight Deck keeps it and sends it
with every message, so the agent tells the open chats to move it (a ``drive``
frame). Every other turn runs at the agent's own default (``drive.txt``, the
"How far it pushes" setting), which the agent writes. Either way the change
applies at once, to the rest of the turn; ``scope: "turn"`` changes only that.
The extra-work switch (what Ambitious may add unasked) is set here too.

Named ``drive_level``, not ``drive``: "save it to Drive" is about files.
"""

from __future__ import annotations

from typing import Any

from captain_claw import drive, turn_source
from captain_claw.tools.registry import Tool, ToolResult

AUTO = "auto"
_LEVEL_CHOICES = (AUTO, *drive.LEVELS)
_LEVEL_HELP = "auto, relaxed, diligent, persistent or ambitious"

# Turns that aren't the owner's never see or change how far the owner's agent
# pushes (the owner pays for it).
_NOT_OWNER_CHANNELS = frozenset({"public", "member", "link"})
_CHAT_TARGETS = frozenset({"chat", "web", "chat box", "chatbox", "web chat"})
_DEFAULT_TARGETS = frozenset({"default", "agent", "all", "everywhere"})
_HERE_TARGETS = frozenset({"", "here", "this", "current"})


def _turn_source_of(agent: Any) -> str:
    src = getattr(agent, "_turn_source", None)
    return str(src[0] or "") if isinstance(src, tuple) and src else ""


def _label(level: str | None) -> str:
    return drive.LEVELS[level].label if level in drive.LEVELS else "Auto"


def _tell_chat_box(agent: Any, level: str) -> bool:
    """Move the chat box's Drive picker in every Flight Deck chat open on this
    agent; False when there is no chat to tell."""
    broadcast = getattr(agent, "ws_broadcast", None)
    if not callable(broadcast):
        return False
    try:
        broadcast({"type": "drive", "level": level})
    except Exception:
        return False
    return True


def _fallback_text(agent: Any) -> str:
    """What Auto means for this agent: its default, its mode's level, or Diligent."""
    own = drive.agent_default()
    if own:
        return f"your default, {_label(own)}"
    suggested = drive.mode_suggestion(agent)
    if suggested:
        return f"what your cognitive mode suggests, {_label(suggested)}"
    return f"{_label(drive.DEFAULT)}, the everyday level"


def _report(agent: Any, here: str) -> str:
    now = drive.current_name(agent)
    override = getattr(agent, "_turn_drive_override", None)
    picked = drive.current_turn_drive()
    if isinstance(override, str) and override:
        why = "you set it for the rest of this turn"
    elif picked:
        why = "the chat box's pick"
    else:
        why = _fallback_text(agent)
    origin = turn_source.label_for(here) if here else "no particular source"
    lines = [f"This message ({origin}): {_label(now)} — {why}."]
    lines.append("Levels:")
    for lvl in drive.list_levels():
        lines.append(f"- {lvl.label} ({lvl.name}): {lvl.summary}")
    own = drive.agent_default()
    lines.append(f"Your default (every turn nobody picked a level for): "
                 f"{_label(own) if own else 'not set — ' + _fallback_text(agent)}.")
    lines.append("Extra work at Ambitious (one obvious next step nobody asked for, reading and "
                 f"writing files only): {'on' if drive.initiative_enabled() else 'off'}.")
    lines.append("You can change these from any channel, this one included: drive_level set with a "
                 "level (where: chat = the chat box's picker, default = your own default), or with "
                 "extra_work on/off. Flight Deck shows the same in the chat box and under the "
                 "agent's settings → How far it pushes.")
    return "\n".join(lines)


class DriveLevelTool(Tool):
    """Read or change how far the agent pushes a turn, as the user would in Flight Deck."""

    name = "drive_level"
    timeout_seconds = 10.0
    description = (
        "Read or change your Drive: how far you push to finish a turn (relaxed · diligent · "
        "persistent · ambitious; auto = your default). Relaxed takes the first complete answer, "
        "fastest and cheapest; Persistent keeps going longer and checks the result; Ambitious is "
        "the most thorough and may add one obvious next step nobody asked for. This is not "
        "reasoning effort (how hard the model thinks) and has nothing to do with Google Drive. "
        "It works from EVERY channel — the web chat, WhatsApp, Telegram, the app, glasses, voice "
        "devices: when the user asks you to be more or less thorough, more ambitious or "
        "persistent, quicker and cheaper, or to set the Drive level, call set from wherever "
        "they asked. Never tell them it can only be changed in Flight Deck. Without 'where' it "
        "changes the level for where this message came from (the web chat: the chat box's "
        "picker; anywhere else: your own default) — at once (the rest of this turn) and for "
        "later messages. On your own, when a task plainly needs much more or much less than "
        "your current level, use scope 'turn' and tell the user; higher levels cost more. "
        "extra_work on/off turns the Ambitious extra step on or off. 'get' shows the level in "
        "effect, every level, your default and the extra-work switch."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set"],
                "description": "get: the level in effect, the levels and the settings. set: change them.",
            },
            "level": {
                "type": "string",
                "enum": list(_LEVEL_CHOICES),
                "description": "For set. auto: no pick — your default (else what your cognitive "
                               "mode suggests, else diligent).",
            },
            "scope": {
                "type": "string",
                "enum": ["setting", "turn"],
                "description": "setting (default): change the setting, as the user would in Flight "
                               "Deck — the rest of this turn and later messages. turn: only the "
                               "rest of this turn; the next message goes back to the setting.",
            },
            "where": {
                "type": "string",
                "enum": ["here", "chat", "default"],
                "description": "Which setting to change when it isn't the one for where this "
                               "message came from: 'chat' (the web chat box's picker) or "
                               "'default' (your own default, for every turn nobody picked a "
                               "level for: other channels, scheduled work).",
            },
            "extra_work": {
                "type": "string",
                "enum": ["on", "off"],
                "description": "For set: whether Ambitious may add one obvious next step nobody "
                               "asked for (reading and writing files only). Can be given alone.",
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "get", level: str | None = None,
                      scope: str | None = None, where: str | None = None,
                      extra_work: str | None = None, **kwargs: Any) -> ToolResult:
        agent = kwargs.get("_agent")
        if agent is None:
            return ToolResult(success=False, error="drive_level needs a running agent.")
        from captain_claw.speaker import is_speaker_agent

        here = _turn_source_of(agent)
        if is_speaker_agent(agent) or turn_source.channel_of(here) in _NOT_OWNER_CHANNELS:
            return ToolResult(success=False, error=(
                "Drive is your owner's setting — it can't be read or changed from this conversation."))
        act = str(action or "get").strip().lower()
        forced = drive.locked()
        if forced and act in ("get", "set"):
            note = (f"Drive is locked at {drive.LEVELS[forced].label} for this run "
                    "(set from outside, e.g. by a benchmark harness); it can't be changed.")
            if act == "set":
                return ToolResult(success=False, error=note)
            return ToolResult(content=f"{note}\n\n{_report(agent, here)}")
        if act == "get":
            return ToolResult(content=_report(agent, here))
        if act != "set":
            return ToolResult(success=False, error=f"Unknown action {action!r} — use get or set.")

        said: list[str] = []
        switch_text = str(extra_work or "").strip().lower()
        if switch_text:
            switch = drive.parse_switch(switch_text)
            if switch is None:
                return ToolResult(success=False, error=f"Unknown extra_work {extra_work!r} — use on or off.")
            try:
                drive.INITIATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
                drive.INITIATIVE_PATH.write_text(drive.render_switch(switch), encoding="utf-8")
            except OSError as exc:
                return ToolResult(success=False, error=f"Couldn't save the extra-work switch: {exc}")
            said.append(
                "Extra work at Ambitious is on: once the asked work is done you may add one obvious "
                "next step, reading and writing files only." if switch else
                "Extra work at Ambitious is off: Ambitious works like Persistent.")

        text = str(level or "").strip().lower()
        if not text:
            if said:
                return ToolResult(content=" ".join(said))
            return ToolResult(success=False, error=f"Give a level ({_LEVEL_HELP}) or extra_work on/off.")
        chosen = AUTO if text in (AUTO, "default", "none") else drive.normalize(text)
        if chosen is None:
            return ToolResult(success=False, error=f"Unknown level {level!r} — use {_LEVEL_HELP}.")
        mode = str(scope or "setting").strip().lower()
        if mode not in ("setting", "turn"):
            return ToolResult(success=False, error=f"Unknown scope {scope!r} — use setting or turn.")
        auto_note = f" Auto means {_fallback_text(agent)}." if chosen == AUTO else ""

        if mode == "turn":
            agent._turn_drive_override = chosen
            said.append(f"Drive for the rest of this turn: {_label(drive.current_name(agent))}."
                        f"{auto_note} The next message goes back to the setting.")
            return ToolResult(content=" ".join(said))

        here_is_web = turn_source.channel_of(here) == "web"
        target_text = " ".join(str(where or "").strip().lower().split())
        if target_text in _HERE_TARGETS:
            target = "chat" if here_is_web else "default"
        elif target_text in _CHAT_TARGETS:
            target = "chat"
        elif target_text in _DEFAULT_TARGETS:
            target = "default"
        else:
            return ToolResult(success=False, error=f"Unknown where {where!r} — use here, chat or default.")

        if target == "chat":
            told = _tell_chat_box(agent, chosen)
            if here_is_web:
                agent._turn_drive_override = chosen
            if not told:
                now = " It still applies to the rest of this turn." if here_is_web else ""
                return ToolResult(success=False, error=(
                    "No Flight Deck chat is connected, so the chat box's Drive picker can't be "
                    f"moved — ask the user to pick {_label(chosen) if chosen != AUTO else 'Auto'} there.{now}"))
            when = ("from now on, starting with the rest of this turn" if here_is_web
                    else "from the next web chat message")
            said.append(f"Drive in the web chat set to {_label(chosen) if chosen != AUTO else 'Auto'} — "
                        f"the chat box's picker in Flight Deck moved to it; it applies {when}.{auto_note}")
            return ToolResult(content=" ".join(said))

        try:
            drive.SETTING_PATH.parent.mkdir(parents=True, exist_ok=True)
            drive.SETTING_PATH.write_text(drive.render(None if chosen == AUTO else chosen), encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, error=f"Couldn't save the Drive default: {exc}")
        # The default decides this turn when nothing was picked for it (a
        # level set for this turn only no longer does).
        applies_here = drive.current_turn_drive() is None
        if applies_here:
            agent._turn_drive_override = None
        when = ("from now on, starting with the rest of this turn" if applies_here
                else "for every turn nobody picks a level for (this chat keeps the chat box's pick)")
        cleared = (f"Your Drive default is cleared — {_fallback_text(agent)} applies" if chosen == AUTO
                   else f"Your Drive default is set to {_label(chosen)} — it applies")
        said.append(f"{cleared} {when}. Flight Deck shows it under the agent's settings → How far it pushes.")
        return ToolResult(content=" ".join(said))
