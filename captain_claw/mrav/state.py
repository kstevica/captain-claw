"""Mrav blackboard — all agent state lives here, outside the model.

Each step re-renders its prompt from this state; the model never sees raw
history. Persisted as JSON per session (survives restarts) with an
append-only JSONL trace next to it for observability.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.mrav.ledger import estimate_tokens

log = get_logger(__name__)


@dataclass
class Observation:
    """One observed result (tool output, digest, or runtime note)."""

    step: int
    kind: str  # "tool" | "note" | "error"
    label: str  # e.g. "read(path=...)"
    text: str
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = estimate_tokens(self.text)


@dataclass
class Blackboard:
    """Task-scoped working state for one Mrav session."""

    task: str = ""
    plan: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    summary: str = ""  # rolling conversation/task summary, carried across tasks
    pinned_tools: list[str] = field(default_factory=list)  # open_tool LRU
    step: int = 0
    consecutive_failures: int = 0
    tasks_completed: int = 0

    def new_task(self, task: str) -> None:
        """Start a new task; keep cross-task continuity (summary + facts)."""
        self.task = task
        self.plan = []
        self.observations = []
        self.pinned_tools = []
        self.step = 0
        self.consecutive_failures = 0

    def add_observation(self, kind: str, label: str, text: str) -> Observation:
        obs = Observation(step=self.step, kind=kind, label=label, text=text)
        self.observations.append(obs)
        return obs

    def observation_tokens(self) -> int:
        return sum(obs.tokens for obs in self.observations)

    def pin_tool(self, name: str, max_pinned: int) -> None:
        """LRU-pin an opened tool schema into the toolpack."""
        if name in self.pinned_tools:
            self.pinned_tools.remove(name)
        self.pinned_tools.append(name)
        while len(self.pinned_tools) > max(0, max_pinned):
            self.pinned_tools.pop(0)

    # ── persistence ──

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Blackboard:
        observations = [
            Observation(**obs) for obs in data.get("observations", []) if isinstance(obs, dict)
        ]
        return cls(
            task=str(data.get("task", "")),
            plan=[str(p) for p in data.get("plan", [])],
            facts=[str(f) for f in data.get("facts", [])],
            observations=observations,
            summary=str(data.get("summary", "")),
            pinned_tools=[str(t) for t in data.get("pinned_tools", [])],
            step=int(data.get("step", 0)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            tasks_completed=int(data.get("tasks_completed", 0)),
        )

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        except Exception as exc:
            log.warning("mrav blackboard save failed", path=str(path), error=str(exc))

    @classmethod
    def load(cls, path: Path) -> Blackboard:
        try:
            if path.is_file():
                return cls.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            log.warning("mrav blackboard load failed — starting fresh", path=str(path), error=str(exc))
        return cls()


class TraceWriter:
    """Append-only JSONL step trace (one file per session)."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def write(self, event: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "event": event, **fields}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            log.debug("mrav trace write failed", error=str(exc))
