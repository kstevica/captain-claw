"""reasoning_effort tool: the agent reads and changes how hard it thinks.

It changes what a person changes in Flight Deck (``llm/reasoning.py``). A web
chat turn's level is the chat box's picker: Flight Deck keeps it and sends it
with every message, so the agent tells the open chats to move it (a
``reasoning_effort`` frame). Every other source's level is in the agent's own
``reasoning_effort.json`` (Settings → Reasoning effort), which the agent
writes. Either way the change applies at once, to the rest of the turn;
``scope: "turn"`` changes only that.
"""

from __future__ import annotations

from typing import Any

from captain_claw import turn_source
from captain_claw.llm import reasoning
from captain_claw.tools.registry import Tool, ToolResult

LEVEL_LABELS: dict[str, str] = {
    "off": "Off", "minimal": "Minimal", "low": "Low", "medium": "Medium",
    "high": "High", "xhigh": "Extra high", "max": "Max", "on": "On",
}
_LEVEL_CHOICES = (reasoning.AUTO, *reasoning.LEVELS, reasoning.ON)
_LEVEL_HELP = "auto, off, minimal, low, medium, high, xhigh, max or on"

# Turns that aren't the owner's never see or change the owner's thinking
# (the owner pays for it).
_NOT_OWNER_CHANNELS = frozenset({"public", "member", "link"})
_WEB_TARGETS = frozenset({"web", "chat", "chat box", "chatbox", "web chat"})
_DEFAULT_TARGETS = frozenset({"default", "all", "everywhere"})
_HERE_TARGETS = frozenset({"", "here", "this", "current"})


def _turn_source_of(agent: Any) -> str:
    src = getattr(agent, "_turn_source", None)
    return str(src[0] or "") if isinstance(src, tuple) and src else ""


def _model_of(agent: Any) -> tuple[str, str]:
    """(provider, model) the agent runs on now."""
    try:
        details = agent.get_runtime_model_details() or {}
    except Exception:
        details = {}
    provider_obj = getattr(agent, "provider", None)
    provider = str(details.get("provider") or getattr(provider_obj, "provider", "") or "")
    model = str(details.get("model") or getattr(provider_obj, "model", "") or "")
    return provider, model


def _level_text(level: str | None, profile: reasoning.Profile | None) -> str:
    """A level as a person reads it on this model."""
    if level in (None, "", reasoning.AUTO):
        return f"Auto (the model's default, {LEVEL_LABELS[profile.default]})" if profile else "Auto (the model's default)"
    label = LEVEL_LABELS.get(level, level)
    if profile is not None and level not in profile.levels:
        near = reasoning.clamp(level, profile)
        if near:
            return f"{label} (runs at {LEVEL_LABELS[near]} on this model)"
    return label


def _where(key: str) -> str:
    if key == "web":
        return "the web chat"
    if key == "default":
        return "every source without a level of its own (the default)"
    return turn_source.label_for(key) or key


def _target(source: Any, here: str) -> str | None:
    """The setting *source* names ("web", "default" or a source key), or None."""
    text = " ".join(str(source or "").strip().lower().split())
    if text in _HERE_TARGETS:
        return turn_source.settings_key(here)
    if text in _WEB_TARGETS:
        return "web"
    if text in _DEFAULT_TARGETS:
        return "default"
    found = turn_source.normalize(text)
    return found if found and found != "web" else None


def _tell_chat_box(agent: Any, level: str) -> bool:
    """Move the chat box's picker in every Flight Deck chat open on this
    agent; False when there is no chat to tell."""
    broadcast = getattr(agent, "ws_broadcast", None)
    if not callable(broadcast):
        return False
    try:
        broadcast({"type": "reasoning_effort", "level": level})
    except Exception:
        return False
    return True


def _report(agent: Any, here: str, profile: reasoning.Profile | None) -> str:
    provider, model = _model_of(agent)
    name = f"{provider} / {model}" if provider else model or "unknown"
    if profile is None:
        return (f"Model: {name} — it has no reasoning-effort controls, so it always thinks "
                "its own way; there is nothing to change.")
    levels = ", ".join(LEVEL_LABELS[lv] for lv in profile.levels)
    lines = [f"Model: {name} — levels: {levels}; with nothing sent it runs at "
             f"{LEVEL_LABELS[profile.default]}."]
    if profile.note:
        lines.append(f"Note: {profile.note}")
    settings = reasoning.load_settings()
    origin = turn_source.label_for(here) if here else "no particular source"
    override = getattr(agent, "_turn_effort_override", None)
    key = reasoning.governing_key(here, settings)
    if override is not None:
        why = "you set it for the rest of this turn"
        now = override
    elif key == "web":
        now = reasoning.current_turn_effort()
        why = "the chat box's pick" if now else "the chat box is on Auto"
    else:
        now = settings.get("default") if key == "default" else settings["channels"].get(key)
        why = (f"your setting for {_where(key)}" if now
               else "no setting for it — the model's default")
    lines.append(f"This message ({origin}): {_level_text(now, profile)} — {why}.")
    lines.append("Levels by channel (the web chat's is the chat box's picker):")
    lines.append(f"- Default: {_level_text(settings.get('default'), profile) if settings.get('default') else 'Auto (not set)'}")
    for src, value in sorted(settings["channels"].items()):
        lines.append(f"- {_where(src)} ({src}): {_level_text(value, profile)}")
    lines.append("You can change any of them from any channel, this one included: reasoning_effort "
                 "set with a level (and a source for another channel: web, default, whatsapp, "
                 "device:sticks3 …). Flight Deck shows the same levels in the chat box and under "
                 "Settings → Reasoning effort.")
    if reasoning.was_refused(model):
        lines.append("The model's server refused reasoning fields earlier in this run, so it runs "
                     "at its default whatever is set, until the agent restarts.")
    return "\n".join(lines)


class ReasoningEffortTool(Tool):
    """Read or change the agent's reasoning effort, as the user would in Flight Deck."""

    name = "reasoning_effort"
    timeout_seconds = 10.0
    description = (
        "Read or change your reasoning effort: how hard your model thinks before answering "
        "(off · minimal · low · medium · high · xhigh · max, or on; auto = the model's own "
        "default). It works from EVERY channel — the web chat, WhatsApp, Telegram, Messenger, "
        "the app, glasses, voice devices: when the user asks you to set, raise or lower it, to "
        "think harder or less, or to turn thinking off or on, call set from wherever they asked. "
        "Never tell them it can only be changed in Flight Deck. Without a source it changes the "
        "level for where this message came from (the web chat: the chat box's picker; anywhere "
        "else: that channel's level) — what the user would change in Flight Deck — at once (the "
        "rest of this turn) and for later messages. On your own, when a task needs much deeper "
        "thinking than you have now (a hard proof, tricky debugging, a long plan) or much less, "
        "use scope 'turn' and tell the user. 'get' shows the current level, the levels your "
        "model offers and every channel's level."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set"],
                "description": "get: the current level, the levels your model offers and the "
                               "settings. set: change it.",
            },
            "level": {
                "type": "string",
                "enum": list(_LEVEL_CHOICES),
                "description": "For set. A level your model doesn't offer goes to the nearest one "
                               "it does; auto sends nothing (the model's own default).",
            },
            "scope": {
                "type": "string",
                "enum": ["setting", "turn"],
                "description": "setting (default): change the setting, as the user would in Flight "
                               "Deck — the rest of this turn and later messages. turn: only the "
                               "rest of this turn; the next message goes back to the setting.",
            },
            "source": {
                "type": "string",
                "description": "Which setting to change when it isn't the one for where this "
                               "message came from: 'web' (the chat box), 'default' (every source "
                               "without a level of its own), or a source — whatsapp, telegram, "
                               "messenger, discord, app, glasses, device:sticks3, "
                               "device:stopwatch, device:papercolor, device:epaper154, "
                               "device:epaper154_touch, scheduled, flow, agent, "
                               "autonomy, api, link, member, public.",
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "get", level: str | None = None,
                      scope: str | None = None, source: str | None = None,
                      **kwargs: Any) -> ToolResult:
        agent = kwargs.get("_agent")
        if agent is None:
            return ToolResult(success=False, error="reasoning_effort needs a running agent.")
        from captain_claw.speaker import is_speaker_agent

        here = _turn_source_of(agent)
        if is_speaker_agent(agent) or turn_source.channel_of(here) in _NOT_OWNER_CHANNELS:
            return ToolResult(success=False, error=(
                "Reasoning effort is your owner's setting — it can't be read or changed "
                "from this conversation."))
        profile = reasoning.profile_for_provider(getattr(agent, "provider", None))
        act = str(action or "get").strip().lower()
        if act == "get":
            return ToolResult(content=_report(agent, here, profile))
        if act != "set":
            return ToolResult(success=False, error=f"Unknown action {action!r} — use get or set.")

        text = str(level or "").strip().lower()
        if not text:
            return ToolResult(success=False, error=f"Give a level: {_LEVEL_HELP}.")
        wanted = reasoning.AUTO if text in (reasoning.AUTO, "default") else reasoning.normalize_level(text)
        if wanted is None:
            return ToolResult(success=False, error=f"Unknown level {level!r} — use {_LEVEL_HELP}.")
        if profile is None:
            provider, model = _model_of(agent)
            return ToolResult(success=False, error=(
                f"Your model ({model or provider or 'unknown'}) has no reasoning-effort controls "
                "— there is nothing to change."))
        chosen = wanted if wanted == reasoning.AUTO else reasoning.clamp(wanted, profile) or wanted
        moved = (f" {LEVEL_LABELS[wanted]} isn't offered by this model; the nearest is "
                 f"{LEVEL_LABELS[chosen]}." if chosen != wanted else "")
        mode = str(scope or "setting").strip().lower()
        if mode not in ("setting", "turn"):
            return ToolResult(success=False, error=f"Unknown scope {scope!r} — use setting or turn.")

        if mode == "turn":
            agent._turn_effort_override = chosen
            return ToolResult(content=(
                f"Reasoning effort for the rest of this turn: {_level_text(chosen, profile)}.{moved} "
                "The next message goes back to the setting."))

        target = _target(source, here)
        if target is None:
            return ToolResult(success=False, error=(
                f"Unknown source {source!r} — use web, default or one of: "
                + ", ".join(k for _t, keys in turn_source.SETTINGS_GROUPS for k in keys) + "."))
        here_is_web = turn_source.channel_of(here) == "web"

        if target == "web":
            told = _tell_chat_box(agent, chosen)
            if here_is_web:
                agent._turn_effort_override = chosen
            if not told:
                now = " It still applies to the rest of this turn." if here_is_web else ""
                return ToolResult(success=False, error=(
                    "No Flight Deck chat is connected, so the chat box's picker can't be moved "
                    f"— ask the user to pick {_level_text(chosen, profile)} there.{now}"))
            when = ("from now on, starting with the rest of this turn" if here_is_web
                    else "from the next web chat message")
            return ToolResult(content=(
                f"Reasoning effort in the web chat set to {_level_text(chosen, profile)} — the chat "
                f"box's picker in Flight Deck moved to it; it applies {when}.{moved}"))

        current = reasoning.load_settings()
        settings = {**current, "channels": dict(current.get("channels") or {})}
        if target == "default":
            settings["default"] = chosen
        else:
            settings["channels"][target] = chosen
        try:
            saved = reasoning.save_settings(settings)
        except OSError as exc:
            return ToolResult(success=False, error=f"Couldn't save the reasoning settings: {exc}")
        applies_here = not here_is_web and reasoning.governing_key(here, saved) == target
        if applies_here:
            # The setting decides the rest of this turn (a level set for this
            # turn only no longer does).
            agent._turn_effort_override = None
        when = ("from now on, starting with the rest of this turn" if applies_here
                else "from the next message that uses it")
        return ToolResult(content=(
            f"Reasoning effort for {_where(target)} set to {_level_text(chosen, profile)} — it "
            f"applies {when}. Flight Deck shows it under Settings → Reasoning effort.{moved}"))
