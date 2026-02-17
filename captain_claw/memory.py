"""Memory management for Captain Claw."""

from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


class Memory:
    """Simple in-memory context tracking."""

    def __init__(self, max_tokens: int = 100000):
        """Initialize memory.
        
        Args:
            max_tokens: Maximum tokens in context
        """
        self.max_tokens = max_tokens
        self._messages: list[dict[str, Any]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to memory."""
        self._messages.append({
            "role": role,
            "content": content,
        })

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages."""
        return self._messages.copy()

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4

    def get_token_count(self) -> int:
        """Get total estimated token count."""
        total = 0
        for msg in self._messages:
            total += self.estimate_tokens(f"{msg['role']}: {msg['content']}")
        return total

    def should_compact(self, threshold: float = 0.8) -> bool:
        """Check if memory should be compacted."""
        return self.get_token_count() > (self.max_tokens * threshold)

    def compact(self, ratio: float = 0.4) -> None:
        """Compact memory by keeping a portion of messages.
        
        Args:
            ratio: Ratio of messages to keep (from start)
        """
        if not self._messages:
            return
        
        keep_count = max(1, int(len(self._messages) * ratio))
        self._messages = self._messages[:keep_count]
        
        log.info("Compacted memory", kept=keep_count, total=len(self._messages))
