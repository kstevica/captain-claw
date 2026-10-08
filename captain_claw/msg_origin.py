"""Where a session message came from, and what the model should see of it.

Stdlib-only, so session, memory and Flight Deck code can use it without
importing the agent.
"""

from __future__ import annotations

import re
from typing import Any

# Tool rows written for monitors and debug views (the UI's trace panes), never
# meant for the model. The FD transcript replay hides these too.
MONITOR_ONLY_TOOL_NAMES = frozenset({
    "llm_trace",
    "planning",
    "task_contract",
    "task_rephrase",
    "completion_gate",
    "pipeline_trace",
    "telegram",
    "memory_select",
    "memory_semantic_select",
    "memory_deep_select",
})

# Hidden from the model's prompt but still replayed to the user as monitor
# cards: the scale loop's own progress rows (its state reaches the model
# through the scale-progress note).
MODEL_HIDDEN_TOOL_NAMES = MONITOR_ONLY_TOOL_NAMES | {"scale_micro_loop"}


def _tool_name(msg: Any) -> str:
    return str((msg.get("tool_name") if isinstance(msg, dict) else "") or "").strip().lower()


def is_monitor_only_tool_name(tool_name: str) -> bool:
    """Monitor/debug-only tool output (hidden from the model and the replay)."""
    return str(tool_name or "").strip().lower() in MONITOR_ONLY_TOOL_NAMES


def is_monitor_only(msg: Any) -> bool:
    """A persisted monitor/debug echo row."""
    return (
        isinstance(msg, dict)
        and str(msg.get("role", "")).strip().lower() == "tool"
        and _tool_name(msg) in MONITOR_ONLY_TOOL_NAMES
    )


def is_model_hidden_tool(msg: Any) -> bool:
    """A tool row the model's prompt never carries."""
    return (
        isinstance(msg, dict)
        and str(msg.get("role", "")).strip().lower() == "tool"
        and _tool_name(msg) in MODEL_HIDDEN_TOOL_NAMES
    )


# ── Provenance ───────────────────────────────────────────────────────
#
# Every session message carries ``origin`` (where it came from); old messages
# are classified at read time from code-literal prefixes, never rewritten.

# role=user, opening a turn.
USER_OPENER_ORIGINS = frozenset({
    "human", "cron", "autonomy", "flow", "peer", "delegated_result", "mcp_task",
    "life_tick", "worker_task", "automated",
})
# role=user, injected mid-turn or outside any turn.
USER_INJECTED_ORIGINS = frozenset({"corrective", "fleet_notice", "notification"})
ASSISTANT_ORIGINS = frozenset({"model", "rejected", "command", "system_note"})
TOOL_ORIGINS = frozenset({"tool", "guard", "code_tool", "debug"})
ALL_ORIGINS = (
    USER_OPENER_ORIGINS | USER_INJECTED_ORIGINS | ASSISTANT_ORIGINS | TOOL_ORIGINS | {"system"}
)

# Automation kinds (mail_authority.AUTOMATION_KINDS) → turn origin.
_AUTOMATION_ORIGIN = {
    "autonomy": "autonomy", "autonomy_tool": "autonomy", "plan": "autonomy",
    "fd_scheduler": "cron", "cron": "cron",
    "flow": "flow", "flow_tool": "flow",
    "peer": "peer", "botport": "peer",
    "peer_relay": "delegated_result",
    "sister": "worker_task", "fd_worker": "worker_task", "bat": "worker_task",
    "being": "life_tick",
    "mcp_task": "mcp_task",
    "unknown": "automated",
}


def from_automation_kind(kind: str) -> str:
    """The turn origin for an automation marker kind."""
    return _AUTOMATION_ORIGIN.get(str(kind or "").strip(), "automated")


def _automated_label_origins() -> dict[str, str]:
    try:
        from captain_claw.mail_authority import KIND_LABELS
    except Exception:  # pragma: no cover - mail_authority is stdlib-only
        return {}
    labels: dict[str, str] = {}
    for kind, label in KIND_LABELS.items():
        labels.setdefault(label, from_automation_kind(kind))
    return labels


# "[Automated turn — <label>. Not a live message from the user.]"
_AUTOMATED_PREFIX_RE = re.compile(r"^\[Automated turn — (?P<label>[^\]]+?)\.\s[^\]]*\]\s*")
# Optional member-privacy header line in front of relayed text.
_PRIVACY_HEADER_RE = re.compile(r"^\[(?:Members' private conversations|About the members of this agent)[^\]]*\]\s*")
# The glasses/WhatsApp/Messenger rules block Flight Deck prepends to a
# binding's first message; the user's words follow the marker.
SURFACE_BLOCK_RE = re.compile(
    r"\[SYSTEM CONTEXT — do not echo, quote, or acknowledge this block in your reply\.\].*?USER MESSAGE:\n",
    re.DOTALL,
)
# The scheduler's clock preamble on a fired job (stale on any later turn).
TIME_ANCHOR_RE = re.compile(
    r"^RIGHT NOW: [^\n]*Anchor every date/deadline judgement on this\.\n.*?"
    r"Do not repeat a nudge you have no fresh reason to send again\.\n\n",
    re.DOTALL,
)

# Ordered (pattern, origin, detail): code-literal openings of synthetic text.
_LITERAL_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(
    (re.compile(pattern, re.DOTALL), origin, detail)
    for pattern, origin, detail in (
        # Correctives, matched on their full code templates (a person can
        # open a message with "You said you delegated…" too).
        (r"^STOP: All your tool calls were blocked as duplicates\. You already have the data you need",
         "corrective", "all_blocked"),
        (r"^\[system\] STOP — you have already written this file\. The file is saved and complete\.",
         "corrective", "write_loop"),
        (r"^\[system\] Scale advisory: \d+ items discovered via glob", "corrective", "scale_advisory"),
        (r"^STOP\. (?:The user asked you for this email and you have|You have) the google_mail tool",
         "corrective", "mail_nudge"),
        (r"^You announced intent without acting\. Do NOT narrate", "corrective", "stall"),
        (r"^You claimed you searched/fetched the web, but you did NOT call", "corrective", "false_web_claim"),
        (r"^You said you delegated/sent the task to a peer, but you", "corrective", "false_delegate_claim"),
        (r"^Your last response was completely empty\. (?:Call one tool now|Write the answer now)",
         "corrective", "empty"),
        (r"^Your last tool call was cut off by the output limit before its arguments finished",
         "corrective", "length_truncation"),
        (r"^That was internal planning data \(a task/contract object\), NOT a", "corrective", "plan_leak"),
        (r"^That was your reasoning, not the answer\. Give the user the complete", "corrective", "preamble"),
        (r"^\[Flight Deck\] Agent '[^']*' has \w+ the fleet", "fleet_notice", ""),
        (r"^\[Delegated result from ", "delegated_result", "delegate"),
        (r"^\[(?:Basna|Vatra|coding session) '.*?' (?:finished successfully|ran into an error)\]",
         "delegated_result", "run_result"),
        (r"^\[Autonomous (?:nudge|task)\]", "autonomy", "nudge"),
        (r"^\[Plan (?:step|task)\]", "autonomy", "plan"),
        (r"^\[SCHEDULED TASK — cron job", "cron", "cron"),
        (r"^RIGHT NOW: [^\n]*Anchor every date/deadline judgement on this", "cron", "fd_scheduler"),
        (r"^\[LIFE TICK — ", "life_tick", "tick"),
        (r"^STOP — reality check before this tick closes", "life_tick", "gate"),
        (r"^Your last reply had NO valid self-report", "life_tick", "gate"),
        (r"^## Project: ", "worker_task", "project"),
        (r"^You are an autonomous research assistant running in the background\.",
         "worker_task", "sister"),
        (r"^Summarize the most important details in one sentence\.\s*\n\s*Content:",
         "automated", "vision"),
    )
)


def detect_literal(text: str) -> tuple[str, str] | None:
    """(origin, detail) when *text* opens like a known synthetic message."""
    body = str(text or "")
    body = _PRIVACY_HEADER_RE.sub("", body, count=1)
    match = _AUTOMATED_PREFIX_RE.match(body)
    if match:
        rest = body[match.end():]
        for pattern, origin, detail in _LITERAL_RULES:
            if pattern.match(rest):
                return origin, detail
        origin = _automated_label_origins().get(match.group("label").strip(), "automated")
        return origin, "automated_prefix"
    for pattern, origin, detail in _LITERAL_RULES:
        if pattern.match(body):
            return origin, detail
    return None


def classify_legacy(msg: Any) -> str:
    """Origin of a message stored before origins were recorded."""
    if not isinstance(msg, dict):
        return "human"
    role = str(msg.get("role", "")).strip().lower()
    if role == "user":
        found = detect_literal(str(msg.get("content", "") or ""))
        return found[0] if found else "human"
    if role == "assistant":
        return "system_note" if _tool_name(msg) == "compaction_summary" else "model"
    if role == "tool":
        return "debug" if is_monitor_only(msg) else "tool"
    return "system" if role == "system" else "human"


def origin_of(msg: Any) -> str:
    """The message's recorded origin, else its read-time classification."""
    if isinstance(msg, dict):
        recorded = str(msg.get("origin", "") or "").strip()
        if recorded in ALL_ORIGINS:
            return recorded
    return classify_legacy(msg)


def is_human_input(msg: Any) -> bool:
    """A message a person typed (or sent through a chat surface)."""
    return (
        isinstance(msg, dict)
        and str(msg.get("role", "")).strip().lower() == "user"
        and origin_of(msg) == "human"
    )


def is_turn_opener(msg: Any) -> bool:
    return (
        isinstance(msg, dict)
        and str(msg.get("role", "")).strip().lower() == "user"
        and origin_of(msg) in USER_OPENER_ORIGINS
    )


def split_surface_block(text: str) -> tuple[str, str]:
    """(rules_block, text_without_it) for a message carrying the surface
    rules block — at the start, or after attachment lines; ("", text)
    otherwise."""
    body = str(text or "")
    match = SURFACE_BLOCK_RE.search(body)
    if not match:
        return "", body
    return match.group(0), body[: match.start()] + body[match.end():]


def surface_rules_text(block: str) -> str:
    """The rules inside a surface block, without its wrapper lines."""
    lines = str(block or "").splitlines()
    kept = [
        line for line in lines
        if not line.startswith("[SYSTEM CONTEXT")
        and line.strip() not in ("---", "USER MESSAGE:")
    ]
    return "\n".join(kept).strip()


def model_view_text(msg: Any) -> str:
    """A historical user message's content as the model should re-read it:
    without the one-off surface rules block or a stale scheduler clock."""
    content = str((msg.get("content") if isinstance(msg, dict) else "") or "")
    _block, rest = split_surface_block(content)
    if _block:
        content = rest
    stripped = TIME_ANCHOR_RE.sub("", content, count=1)
    if stripped != content:
        content = stripped
    else:
        match = _AUTOMATED_PREFIX_RE.match(content)
        if match:
            tail = TIME_ANCHOR_RE.sub("", content[match.end():], count=1)
            content = content[: match.end()] + tail
    return content


def resolve_turn_origin(
    user_input: str,
    explicit: str | None = None,
    explicit_detail: str | None = None,
) -> tuple[str, str]:
    """(origin, detail) for a turn's opening message.

    The caller's explicit origin wins; then the code-literal shape of the
    text (life ticks, nudges, scheduler prompts…); then an automation marker
    bound for this turn; in a multi-agent worker process an unmarked turn is
    the orchestrator's task. Everything else is a person. (The mail-authority
    process default is not used for beings: a parent chatting with a being
    is a human turn.)
    """
    if explicit and explicit in ALL_ORIGINS:
        return explicit, str(explicit_detail or "")
    found = detect_literal(user_input)
    # An opener is never a code-injected corrective or notice; those shapes on
    # a turn's input are a person's words.
    if found and found[0] not in USER_INJECTED_ORIGINS:
        return found
    try:
        from captain_claw import mail_authority

        bound = mail_authority.bound_authority()
        default = mail_authority.process_default()
        if bound is not None and bound.mode == "automated":
            if not (default.mode == "automated" and bound.kind == default.kind):
                return from_automation_kind(bound.kind), bound.kind
        if default.mode == "automated" and default.kind == "fd_worker":
            return "worker_task", "fd_worker"
    except Exception:
        pass
    return "human", ""


def default_origin(role: str, *, tool_name: str = "", tool_call_id: str = "") -> str:
    """The origin a message written without one gets (non-opener writes)."""
    role = str(role or "").strip().lower()
    if role == "assistant":
        return "model"
    if role == "tool":
        if is_monitor_only_tool_name(tool_name):
            return "debug"
        return "tool" if str(tool_call_id or "").strip() else "code_tool"
    if role == "system":
        return "system"
    return "corrective"


def hint_turn_provenance(
    agent: Any,
    *,
    turn_origin: str | None = None,
    channel: str | None = None,
    origin_detail: str | None = None,
) -> None:
    """Tell *agent*'s next ``complete()``/``stream()`` where the turn came
    from. A hint rather than call arguments, so any agent object — including
    test doubles with a plain ``complete(content)`` — can be driven."""
    try:
        agent._turn_provenance_hint = (turn_origin, origin_detail, channel)
    except Exception:
        pass
