"""Mrav step protocol — response schemas, lenient parsing, action validation.

Small models get grammar-constrained decoding where the provider supports it
(Ollama ``format``), but grammar guarantees syntax, not semantics — so every
response still goes through validation, and parsing stays junk-tolerant for
providers without constrained output (generalizes ``parse_feet_act``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Flat object with an action discriminator — deliberately NOT a oneOf union:
# flat + enum is far more robust for 2-4B models and for grammar engines.
ACT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool", "open_tool", "final", "give_up"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "name": {"type": "string"},
        "text": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action"],
}

PLAN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 6,
        },
    },
    "required": ["plan"],
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_thinking(text: str) -> str:
    """Drop <think>…</think> blocks some models leak despite think=off."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    # Unclosed think block: everything after the opener is reasoning.
    idx = cleaned.lower().find("<think>")
    if idx >= 0:
        cleaned = cleaned[:idx]
    return cleaned.strip()


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced {...} object, string-aware."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Lenient JSON-object parse ladder: strict → fences → balanced-scan → repairs."""
    if not text:
        return None
    cleaned = strip_thinking(text)

    candidates: list[str] = [cleaned]
    fence = _FENCE_RE.search(cleaned)
    if fence:
        candidates.append(fence.group(1).strip())
    first = _first_json_object(cleaned)
    if first:
        candidates.append(first)

    for candidate in candidates:
        if not candidate:
            continue
        for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                obj = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
    return None


@dataclass
class StepAction:
    """A validated ACT decision."""

    kind: str  # "tool" | "open_tool" | "final" | "give_up"
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    text: str = ""
    reason: str = ""


def validate_action(
    obj: dict[str, Any] | None,
    visible_tools: set[str],
    all_tools: set[str],
) -> tuple[StepAction | None, str]:
    """Validate a parsed ACT object; returns (action, "") or (None, error).

    The error string is written back into the retry prompt, so it is phrased
    for the model, not for a stack trace.
    """
    if obj is None:
        return None, "Response was not a single JSON object. Reply with exactly one JSON object."

    kind = str(obj.get("action", "")).strip().lower()
    if kind == "tool":
        tool = str(obj.get("tool", "")).strip()
        if not tool:
            return None, 'action "tool" needs a "tool" name from TOOLS.'
        if tool not in visible_tools:
            if tool in all_tools:
                return None, (
                    f'Tool "{tool}" is in INDEX, not TOOLS. '
                    f'Use {{"action":"open_tool","name":"{tool}"}} first.'
                )
            return None, f'Unknown tool "{tool}". Use an exact name from TOOLS.'
        args = obj.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None, '"args" must be a JSON object.'
        return StepAction(kind="tool", tool=tool, args=args), ""

    if kind == "open_tool":
        name = str(obj.get("name", "") or obj.get("tool", "")).strip()
        if not name:
            return None, 'action "open_tool" needs a "name" from INDEX.'
        if name not in all_tools:
            return None, f'Unknown tool "{name}". Use an exact name from INDEX.'
        # Opening an already-visible tool is accepted as a no-op: small
        # models routinely "open" a core tool before calling it, and
        # rejecting that traps them in a repeat loop (seen live with
        # Gemma 4 E2B). The runtime answers with a nudge observation.
        return StepAction(kind="open_tool", name=name), ""

    if kind == "final":
        text = str(obj.get("text", "") or obj.get("answer", "")).strip()
        if not text:
            return None, 'action "final" needs non-empty "text" with the complete answer.'
        return StepAction(kind="final", text=text), ""

    if kind == "give_up":
        reason = str(obj.get("reason", "")).strip() or "no reason given"
        return StepAction(kind="give_up", reason=reason), ""

    return None, 'Field "action" must be one of: tool, open_tool, final, give_up.'


def parse_plan(text_or_obj: str | dict[str, Any] | None) -> list[str]:
    """Parse a PLAN response into a bounded list of short step strings."""
    obj = text_or_obj if isinstance(text_or_obj, dict) else parse_json_object(text_or_obj or "")
    if not isinstance(obj, dict):
        return []
    plan = obj.get("plan")
    if not isinstance(plan, list):
        return []
    steps = [str(item).strip() for item in plan if str(item).strip()]
    return steps[:6]
