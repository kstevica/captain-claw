"""Structured tracing for orchestrator observability.

Emits trace spans for LLM calls, tool executions, task lifecycle,
and orchestration events with timing, token usage, and correlation IDs.

Inspired by Open Multi-Agent's onTrace callback pattern.

Usage:
    ctx = TraceContext(trace_id="run-123", callback=my_handler)
    span_id = ctx.start_span("task", "Implement auth", task_id="t1")
    ...
    ctx.end_span(span_id, status="completed", tokens=1234)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from captain_claw.logging import get_logger

log = get_logger(__name__)


@dataclass
class TraceSpan:
    """A single trace span representing an operation in the orchestration."""

    span_id: str
    trace_id: str                       # orchestration run correlation ID
    parent_span_id: str = ""            # for hierarchy (task → llm_call)
    span_type: str = ""                 # orchestration | task | llm_call |
                                        # tool_execution | validation |
                                        # workspace_op | decompose | synthesize
    name: str = ""                      # human-readable label
    started_at: float = 0.0             # monotonic timestamp
    ended_at: float = 0.0               # 0 if still running
    status: str = "running"             # running | completed | failed
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds (0 if still running)."""
        if self.ended_at <= 0:
            return 0.0
        return (self.ended_at - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON transport / broadcast."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "span_type": self.span_type,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status,
            "attributes": self.attributes,
        }


TraceCallback = Callable[[TraceSpan], None]


class TraceContext:
    """Manages a hierarchy of trace spans for one orchestration run.

    Accumulates spans in memory and optionally emits each span event
    through a callback (e.g. for WebSocket broadcast to Flight Deck).

    Thread-safe via simple dict operations — only the orchestration's
    async event loop writes spans.
    """

    def __init__(
        self,
        trace_id: str,
        callback: TraceCallback | None = None,
    ) -> None:
        self.trace_id = trace_id
        self._callback = callback
        self._spans: dict[str, TraceSpan] = {}

    # ------------------------------------------------------------------
    # Span lifecycle
    # ------------------------------------------------------------------

    def start_span(
        self,
        span_type: str,
        name: str,
        *,
        parent_span_id: str = "",
        **attributes: Any,
    ) -> str:
        """Start a new span and return its ID."""
        span_id = str(uuid.uuid4())[:12]
        span = TraceSpan(
            span_id=span_id,
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            span_type=span_type,
            name=name,
            started_at=time.monotonic(),
            status="running",
            attributes=dict(attributes),
        )
        self._spans[span_id] = span
        self._emit(span)
        return span_id

    def end_span(
        self,
        span_id: str,
        status: str = "completed",
        **extra_attributes: Any,
    ) -> None:
        """End a span, updating its status and duration."""
        span = self._spans.get(span_id)
        if span is None:
            log.warning("end_span: unknown span_id", span_id=span_id)
            return
        span.ended_at = time.monotonic()
        span.status = status
        if extra_attributes:
            span.attributes.update(extra_attributes)
        self._emit(span)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_spans(self) -> list[TraceSpan]:
        """All accumulated spans (completed + running)."""
        return list(self._spans.values())

    def get_spans_by_type(self, span_type: str) -> list[TraceSpan]:
        return [s for s in self._spans.values() if s.span_type == span_type]

    def get_child_spans(self, parent_span_id: str) -> list[TraceSpan]:
        return [s for s in self._spans.values() if s.parent_span_id == parent_span_id]

    def get_summary(self) -> dict[str, Any]:
        """Aggregated statistics for the trace."""
        spans = list(self._spans.values())
        total_duration_ms = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        by_type: dict[str, int] = {}

        for s in spans:
            by_type[s.span_type] = by_type.get(s.span_type, 0) + 1
            if s.ended_at > 0:
                total_duration_ms += s.duration_ms
            total_input_tokens += s.attributes.get("input_tokens", 0)
            total_output_tokens += s.attributes.get("output_tokens", 0)

        return {
            "trace_id": self.trace_id,
            "span_count": len(spans),
            "by_type": by_type,
            "total_duration_ms": round(total_duration_ms, 1),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "completed": sum(1 for s in spans if s.status == "completed"),
            "failed": sum(1 for s in spans if s.status == "failed"),
            "running": sum(1 for s in spans if s.status == "running"),
        }

    def to_list(self) -> list[dict[str, Any]]:
        """All spans as serialisable dicts."""
        return [s.to_dict() for s in self._spans.values()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all spans."""
        self._spans.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, span: TraceSpan) -> None:
        """Fire the callback if registered."""
        if self._callback:
            try:
                self._callback(span)
            except Exception:
                pass
