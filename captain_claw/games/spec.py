"""WorldSpec — the user's high-level description of a game.

A spec is intentionally tiny: it's everything the user must provide.
The generator fills in the rest. Specs are stored alongside generated
worlds for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldSpec:
    title: str
    goal: str                                # one-line objective
    seats: int = 2                           # number of player characters
    genre: str = "exploration"               # cozy-mystery | exploration | escape | ...
    tone: str = "neutral"                    # short adjective list — drives narration
    size: str = "small"                      # small | medium | large | xl | huge | epic
    constraints: list[str] = field(default_factory=list)
    seat_mode: str = "party"                 # party | solo | mixed
    summary: str = ""                        # optional longer paragraph
    carry_reflections: list[str] = field(default_factory=list)  # opt-in per design §14.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "goal": self.goal,
            "seats": self.seats,
            "genre": self.genre,
            "tone": self.tone,
            "size": self.size,
            "constraints": list(self.constraints),
            "seat_mode": self.seat_mode,
            "summary": self.summary,
            "carry_reflections": list(self.carry_reflections),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldSpec":
        return cls(
            title=str(data.get("title", "Untitled")).strip() or "Untitled",
            goal=str(data.get("goal", "")).strip(),
            seats=int(data.get("seats", 2)),
            genre=str(data.get("genre", "exploration")),
            tone=str(data.get("tone", "neutral")),
            size=str(data.get("size", "small")),
            constraints=[str(c) for c in (data.get("constraints") or [])],
            seat_mode=str(data.get("seat_mode", "party")),
            summary=str(data.get("summary", "")),
            carry_reflections=[str(r) for r in (data.get("carry_reflections") or [])],
        )

    def validate(self) -> str | None:
        if not self.goal:
            return "spec.goal is required"
        if self.seats < 1 or self.seats > 6:
            return "spec.seats must be 1..6"
        if self.size not in {"small", "medium", "large", "xl", "huge", "epic"}:
            return "spec.size must be small | medium | large | xl | huge | epic"
        if self.seat_mode not in {"party", "solo", "mixed"}:
            return "spec.seat_mode must be party | solo | mixed"
        return None


# Size hints used by the generator to bound the world's complexity.
SIZE_HINTS: dict[str, dict[str, int]] = {
    "small":  {"rooms": 4,   "entities": 4,   "talkable_npcs": 0},
    "medium": {"rooms": 7,   "entities": 8,   "talkable_npcs": 2},
    "large":  {"rooms": 12,  "entities": 14,  "talkable_npcs": 4},
    "xl":     {"rooms": 37,  "entities": 27,  "talkable_npcs": 4},
    "huge":   {"rooms": 75,  "entities": 55,  "talkable_npcs": 4},
    "epic":   {"rooms": 135, "entities": 90,  "talkable_npcs": 4},
}

# Batch size limits for multi-pass generation (rooms/entities per LLM call)
BATCH_ROOMS = 20
BATCH_ENTITIES = 25
