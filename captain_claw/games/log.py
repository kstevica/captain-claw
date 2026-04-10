"""Append-only game log — the source of truth for replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from captain_claw.games.intent import Intent


class GameLog:
    """Append-only JSONL log of (tick, intents, events) tuples.

    Replay is `for entry in log: state, _ = engine.resolve(state, entry.intents, rng)`.
    Narrations are stored alongside intents on the same line so a replay
    never re-calls the LLM (relevant from M2 onward).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(
        self,
        tick: int,
        intents: list[Intent],
        events: list[dict[str, Any]],
        narration: dict[str, str] | None = None,
    ) -> None:
        record = {
            "tick": tick,
            "intents": [i.to_dict() for i in intents],
            "events": list(events),
            "narration": narration or {},
        }
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def truncate_to(self, tick: int) -> None:
        """Drop all entries with tick > `tick`. Used by `fork`."""
        records = [r for r in self.read_all() if r["tick"] <= tick]
        with self.path.open("w", encoding="utf-8") as fp:
            for r in records:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
