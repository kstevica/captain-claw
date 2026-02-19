"""Layered memory primitives for Captain Claw."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.semantic_memory import SemanticMemoryIndex, create_semantic_memory_index

log = get_logger(__name__)


@dataclass
class WorkingMemorySnapshot:
    """Serializable working-memory snapshot."""

    summary: str
    messages: list[dict[str, Any]]


class WorkingMemory:
    """In-turn memory that keeps a summary + recent detailed window."""

    def __init__(self, max_tokens: int = 100_000):
        self.max_tokens = max_tokens
        self._messages: list[dict[str, Any]] = []
        self._summary: str = ""

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_token_count(self) -> int:
        summary_tokens = self.estimate_tokens(self._summary) if self._summary else 0
        message_tokens = sum(
            self.estimate_tokens(f"{msg.get('role', '')}: {msg.get('content', '')}")
            for msg in self._messages
        )
        return summary_tokens + message_tokens

    def should_compact(self, threshold: float = 0.8) -> bool:
        return self.get_token_count() > (self.max_tokens * threshold)

    def compact(self, keep_recent_ratio: float = 0.4) -> None:
        """Compact by summarizing older messages and retaining recent details."""
        if len(self._messages) <= 1:
            return
        ratio = min(max(float(keep_recent_ratio), 0.05), 0.95)
        keep_count = max(1, int(len(self._messages) * ratio))
        dropped = self._messages[:-keep_count]
        recent = self._messages[-keep_count:]
        if dropped:
            summary_lines = []
            for msg in dropped[-10:]:
                role = str(msg.get("role", "")).strip().lower() or "unknown"
                content = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
                if not content:
                    continue
                snippet = content[:180] + ("..." if len(content) > 180 else "")
                summary_lines.append(f"- {role}: {snippet}")
            new_summary = "Earlier context summary:\n" + "\n".join(summary_lines) if summary_lines else ""
            if self._summary and new_summary:
                self._summary = f"{self._summary}\n\n{new_summary}".strip()
            elif new_summary:
                self._summary = new_summary
        self._messages = recent
        log.info(
            "Working memory compacted",
            dropped=len(dropped),
            kept=len(self._messages),
            has_summary=bool(self._summary),
        )

    def snapshot(self, include_summary_message: bool = True) -> WorkingMemorySnapshot:
        messages = list(self._messages)
        if include_summary_message and self._summary:
            messages = [
                {
                    "role": "assistant",
                    "content": f"Conversation summary from older context:\n{self._summary}",
                    "tool_name": "working_memory_summary",
                },
                *messages,
            ]
        return WorkingMemorySnapshot(summary=self._summary, messages=messages)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)


class LayeredMemory:
    """Three-layer memory facade:
    1) Working memory (summary + recent turn window),
    2) Session memory (managed by SessionManager),
    3) Semantic memory (SQLite hybrid retrieval).
    """

    def __init__(
        self,
        *,
        working_memory: WorkingMemory,
        semantic_memory: SemanticMemoryIndex | None = None,
    ):
        self.working = working_memory
        self.semantic = semantic_memory
        self.active_session_id: str | None = None

    def set_active_session(self, session_id: str | None) -> None:
        self.active_session_id = (session_id or "").strip() or None
        if self.semantic is not None:
            self.semantic.set_active_session(self.active_session_id)

    def record_message(self, role: str, content: str) -> None:
        self.working.add_message(role, content)
        if self.semantic is not None and role in {"user", "assistant"}:
            self.semantic.schedule_sync("message")

    def compact_working_memory(self, keep_recent_ratio: float = 0.4) -> None:
        self.working.compact(keep_recent_ratio=keep_recent_ratio)

    def schedule_background_sync(self, reason: str = "manual") -> None:
        if self.semantic is not None:
            self.semantic.schedule_sync(reason)

    def search_semantic(self, query: str, max_results: int | None = None):
        if self.semantic is None:
            return []
        return self.semantic.search(query=query, max_results=max_results)

    def build_semantic_note(
        self,
        query: str,
        *,
        max_items: int = 3,
        max_snippet_chars: int = 360,
    ) -> tuple[str, str]:
        if self.semantic is None:
            return "", ""
        return self.semantic.build_context_note(
            query=query,
            max_items=max_items,
            max_snippet_chars=max_snippet_chars,
        )

    def close(self) -> None:
        if self.semantic is not None:
            self.semantic.close()


# Backward-compatible alias:
Memory = WorkingMemory


def create_layered_memory(
    *,
    config: Any,
    session_db_path: Path,
    workspace_path: Path,
) -> LayeredMemory:
    """Create layered memory from runtime config."""
    working = WorkingMemory(max_tokens=int(getattr(config.context, "max_tokens", 100_000)))
    memory_cfg = getattr(config, "memory", None)
    if memory_cfg is None or not bool(getattr(memory_cfg, "enabled", True)):
        return LayeredMemory(working_memory=working, semantic_memory=None)

    try:
        semantic = create_semantic_memory_index(
            memory_cfg=memory_cfg,
            session_db_path=session_db_path,
            workspace_path=workspace_path,
        )
    except Exception as exc:
        log.warning("Semantic memory disabled due to initialization failure", error=str(exc))
        semantic = None
    return LayeredMemory(working_memory=working, semantic_memory=semantic)
