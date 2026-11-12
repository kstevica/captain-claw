"""model_switch tool: the agent reads and changes the model it runs on.

Only among the models of its own provider at its own endpoint — the key and
the address stay the same (``llm/model_switch.py`` says which models those
are). It is what the user does with the chat box's model picker in Flight
Deck: ``scope: "setting"`` switches this conversation from now on (kept with
it, so a restart, /clear and /new keep it), ``scope: "turn"`` switches only
the rest of this turn.

When the user asks, it always may. On its own initiative — a cheaper model
for simple bulk work, a smarter one for a hard task — only while its owner
turned that on (``~/.captain-claw/model_switch.txt``, Flight Deck → agent
settings).
"""

from __future__ import annotations

from typing import Any

from captain_claw import turn_source
from captain_claw.llm import model_switch, reasoning
from captain_claw.tools.registry import Tool, ToolResult

# Turns that aren't the owner's never see or change the owner's model (the
# owner pays for it).
_NOT_OWNER_CHANNELS = frozenset({"public", "member", "link"})


def _turn_source_of(agent: Any) -> str:
    src = getattr(agent, "_turn_source", None)
    return str(src[0] or "") if isinstance(src, tuple) and src else ""


def _recent_user_text(agent: Any) -> str:
    fn = getattr(agent, "_recent_user_text", None)
    if callable(fn):
        try:
            return str(fn() or "")
        except Exception:
            pass
    return str(getattr(agent, "_turn_user_text", "") or "")


def _choice_line(opt: dict[str, Any], running: str) -> str:
    bits = [f"{opt['name']} ({opt['model']})"]
    what = ", ".join(x for x in (opt.get("label"), opt.get("note")) if x)
    if what:
        bits.append(f"— {what}")
    price = model_switch.price_text(opt.get("price"))
    if price:
        bits.append(f"· {price}")
    tags = []
    if opt.get("default"):
        tags.append("configured default")
    if model_switch.is_model(running, opt["model"]):
        tags.append("running now")
    if tags:
        bits.append(f"[{', '.join(tags)}]")
    return "- " + " ".join(bits)


def _report(agent: Any) -> str:
    options = agent.model_switch_choices()
    running = agent._running_model()
    provider, home_model, base_url = agent._switch_home()
    where = f"{provider}" + (f" at {base_url}" if base_url else "")
    lines = [f"Provider: {where} (switching keeps it and its key; only the model changes)."]
    now = next((o for o in options if model_switch.is_model(running, o["model"])), None)
    now_name = f"{now['name']} ({now['model']})" if now else running or "unknown"
    turn_only = getattr(agent, "_turn_model_restore", None) is not None
    lines.append(f"Running now: {now_name}" + (" — for this turn only" if turn_only else "") + ".")
    if len(options) < 2:
        lines.append("There is no other model to switch to here. An admin can list more models for "
                     "this endpoint in Flight Deck (Console → Models).")
        return "\n".join(lines)
    lines.append("Models you can switch to:")
    lines.extend(_choice_line(o, running) for o in options)
    if model_switch.self_switch_enabled():
        lines.append("Switching on your own is ON: when a task clearly needs a smarter model (a hard "
                     "proof, tricky debugging, a long plan) or would do as well on a cheaper one "
                     "(simple bulk work), you may switch with scope 'turn' and tell the user.")
    else:
        lines.append("Switching on your own is OFF (the owner turns it on in Flight Deck → Agent "
                     "settings): switch only when the user asks.")
    lines.append("You can switch from any channel, this one included: model_switch set with a model "
                 "(a name, part of one like 'opus', 'default', 'smartest' or 'cheapest'). Flight Deck "
                 "shows the same choices in the chat box's model picker.")
    return "\n".join(lines)


def _tell_chat_box(agent: Any) -> None:
    """Move the model picker (and the effort picker, whose levels are the
    model's) in every Flight Deck chat open on this agent."""
    broadcast = getattr(agent, "ws_broadcast", None)
    if not callable(broadcast):
        return
    try:
        profile = reasoning.profile_for_provider(getattr(agent, "provider", None))
        broadcast({
            "type": "model_switch",
            "model_switch": agent.model_switch_wire(),
            "reasoning": profile.to_wire() if profile else None,
        })
    except Exception:
        pass


class ModelSwitchTool(Tool):
    """Read or change the model the agent runs on, among its provider's models."""

    name = "model_switch"
    timeout_seconds = 20.0
    description = (
        "Read or change the AI model you run on — only among your provider's models (same key, "
        "same endpoint), e.g. a smarter one for a hard task or a cheaper one for simple work. It "
        "works from EVERY channel (web chat, WhatsApp, Telegram, the app, glasses, devices): when "
        "the user asks you to switch, use a smarter/cheaper/faster model, or go back to the "
        "default, call set from wherever they asked — never say only Flight Deck can change it. "
        "scope 'setting' (default) switches this conversation from now on, as the chat box's model "
        "picker would; 'turn' only the rest of this turn. On your own initiative only while your "
        "owner allowed it (get says) — then use scope 'turn' and tell the user why. 'get' lists "
        "the models, what each is for and costs, and which one runs now."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set"],
                "description": "get: the models you can switch to and the one running now. set: switch.",
            },
            "model": {
                "type": "string",
                "description": "For set: a model id or name from get, part of one ('opus', 'flash'), "
                               "'default' (your configured model), 'smartest' or 'cheapest'.",
            },
            "scope": {
                "type": "string",
                "enum": ["setting", "turn"],
                "description": "setting (default when the user asks): this conversation from now on. "
                               "turn (default on your own): only the rest of this turn.",
            },
            "reason": {
                "type": "string",
                "enum": ["user_asked", "own"],
                "description": "For set: user_asked — the user asked for this switch in their "
                               "message; own — your own choice (allowed only when get says so).",
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str = "get", model: str | None = None,
                      scope: str | None = None, reason: str | None = None,
                      **kwargs: Any) -> ToolResult:
        agent = kwargs.get("_agent")
        if agent is None or not callable(getattr(agent, "switch_model", None)):
            return ToolResult(success=False, error="model_switch needs a running agent.")
        here = _turn_source_of(agent)
        if turn_source.channel_of(here) in _NOT_OWNER_CHANNELS:
            return ToolResult(success=False, error=(
                "The model is your owner's setting — it can't be read or changed from this conversation."))
        blocked = agent._model_switch_blocked()
        if blocked:
            return ToolResult(success=False, error=blocked)
        act = str(action or "get").strip().lower()
        if act == "get":
            # A fresh copy of the admin's list for this endpoint, when Flight Deck answers.
            provider, _home_model, base_url = agent._switch_home()
            try:
                await model_switch.refresh_custom(provider, base_url)
            except Exception:
                pass
            return ToolResult(content=_report(agent))
        if act != "set":
            return ToolResult(success=False, error=f"Unknown action {action!r} — use get or set.")
        wanted = str(model or "").strip()
        if not wanted:
            return ToolResult(success=False, error="Give a model — model_switch get lists them.")

        options = agent.model_switch_choices()
        why = str(reason or "").strip().lower()
        asked = model_switch.talks_about_models(_recent_user_text(agent), options, loose=True)
        own = why == "own" or (why != "user_asked" and not asked)
        if own and not model_switch.self_switch_enabled():
            return ToolResult(success=False, error=(
                "Switching models on your own is off for this agent. Carry on with the current "
                "model, or ask the user — they can say which model to use, or turn on switching on "
                "your own in Flight Deck → Agent settings."))
        if why == "user_asked" and not asked and not model_switch.self_switch_enabled():
            return ToolResult(success=False, error=(
                "The user's message doesn't ask for a model change, and switching on your own is off "
                "for this agent. Ask the user first."))

        mode = str(scope or "").strip().lower() or ("turn" if own else "setting")
        if mode not in ("setting", "turn"):
            return ToolResult(success=False, error=f"Unknown scope {scope!r} — use setting or turn.")
        ok, message, choice = await agent.switch_model(wanted, scope=mode)
        if not ok:
            return ToolResult(success=False, error=message)
        _tell_chat_box(agent)
        lines = [message]
        price = model_switch.price_text((choice or {}).get("price"))
        if price:
            lines.append(f"It costs {price}.")
        if mode == "turn":
            lines.append("The next message goes back to the model before it.")
        else:
            lines.append("The chat box's model picker in Flight Deck shows it.")
        if own:
            lines.append("Tell the user you switched, and why.")
        return ToolResult(content=" ".join(lines))
