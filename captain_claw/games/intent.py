"""Intents — the structured action a player submits each tick.

The verb set is closed: anything not in `VERBS` is rejected by the engine.
Free-text from human players must be parsed into one of these by the
input adapter (web layer); the engine never sees raw text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Closed verb set. Add to this list deliberately, never dynamically.
VERBS: tuple[str, ...] = (
    "wait",      # no-op, advances tick
    "look",      # re-emit own view
    "move",      # args: {"direction": "north"} OR {"room_id": "..."}
    "take",      # args: {"entity_id": "..."}
    "drop",      # args: {"entity_id": "..."}
    "say",       # args: {"text": "..."} — public, heard by chars in same room
    "talk",      # args: {"target": "<char_id>", "text": "..."} — private direct message, cross-room
    "use",       # args: {"item_id": "...", "target_id": "..."} — use item on target
    "examine",   # args: {"entity_id": "..."} — examine an entity for detail
    "give",      # args: {"entity_id": "...", "target_id": "..."} — give item to character
)


@dataclass(frozen=True)
class Intent:
    actor: str                       # character id submitting this intent
    verb: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"actor": self.actor, "verb": self.verb, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intent":
        return cls(
            actor=str(data["actor"]),
            verb=str(data["verb"]),
            args=dict(data.get("args", {})),
        )


def validate_shape(raw: dict[str, Any]) -> tuple[Intent | None, str | None]:
    """Surface-level validation of an intent dict. Returns (intent, error)."""
    if not isinstance(raw, dict):
        return None, "intent must be an object"
    actor = raw.get("actor")
    verb = raw.get("verb")
    if not isinstance(actor, str) or not actor:
        return None, "missing actor"
    if not isinstance(verb, str) or verb not in VERBS:
        return None, f"unknown verb '{verb}' (allowed: {', '.join(VERBS)})"
    args = raw.get("args", {}) or {}
    if not isinstance(args, dict):
        return None, "args must be an object"
    return Intent(actor=actor, verb=verb, args=args), None
