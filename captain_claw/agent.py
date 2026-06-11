"""Agent orchestration for Captain Claw."""

import asyncio
import json
from pathlib import Path
from typing import Any
from typing import Callable

from captain_claw.agent_chunked_processing_mixin import AgentChunkedProcessingMixin
from captain_claw.agent_completion_mixin import AgentCompletionMixin
from captain_claw.agent_context_mixin import AgentContextMixin
from captain_claw.agent_file_ops_mixin import AgentFileOpsMixin
from captain_claw.agent_guard_mixin import AgentGuardMixin
from captain_claw.agent_model_mixin import AgentModelMixin
from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin
from captain_claw.agent_playbook_mixin import AgentPlaybookMixin
from captain_claw.agent_pipeline_mixin import AgentPipelineMixin
from captain_claw.agent_reasoning_mixin import AgentReasoningMixin
from captain_claw.agent_research_mixin import AgentResearchMixin
from captain_claw.agent_scale_detection_mixin import AgentScaleDetectionMixin
from captain_claw.agent_scale_loop_mixin import AgentScaleLoopMixin
from captain_claw.agent_session_mixin import AgentSessionMixin
from captain_claw.agent_skills_mixin import AgentSkillsMixin
from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin
from captain_claw.config import get_config
from captain_claw.file_registry import FileRegistry
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider, Message
from captain_claw.llm_session_logger import get_llm_session_logger
from captain_claw.session import Session, get_session_manager
from captain_claw.tools import get_tool_registry


class Agent(
    AgentOrchestrationMixin,
    AgentCompletionMixin,
    AgentFileOpsMixin,
    AgentContextMixin,
    AgentGuardMixin,
    AgentModelMixin,
    AgentPipelineMixin,
    AgentReasoningMixin,
    AgentResearchMixin,
    AgentScaleDetectionMixin,
    AgentScaleLoopMixin,
    AgentChunkedProcessingMixin,
    AgentSessionMixin,
    AgentSkillsMixin,
    AgentPlaybookMixin,
    AgentToolLoopMixin,
):
    """Main agent orchestrator."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        status_callback: Callable[[str], None] | None = None,
        tool_output_callback: Callable[[str, dict[str, Any], str], None] | None = None,
        approval_callback: Callable[[str], bool] | None = None,
        thinking_callback: Callable[[str, str, str], None] | None = None,
        tool_stream_callback: Callable[[str], None] | None = None,
        narration_callback: Callable[[str, int], None] | None = None,
    ):
        """Initialize the agent.

        Args:
            provider: Optional LLM provider override
            status_callback: Optional runtime status callback
            approval_callback: Optional callback for guard approval prompts
            thinking_callback: Optional callback for inline thinking/reasoning updates
            tool_stream_callback: Optional callback for streaming tool output chunks
        """
        self.provider = provider
        self.status_callback = status_callback
        self.tool_output_callback = tool_output_callback
        self.approval_callback = approval_callback
        self.playbook_approval_callback: Callable[[str], Any] | None = None
        self.thinking_callback = thinking_callback
        self._tool_stream_callback = tool_stream_callback
        self.narration_callback = narration_callback
        self.response_stream_callback: Callable[[str], None] | None = None
        self.tools = get_tool_registry()
        self.tools.set_approval_callback(self.approval_callback)
        self.runtime_base_path = Path.cwd().resolve()
        cfg = get_config()
        self.workspace_base_path = cfg.resolved_workspace_path(self.runtime_base_path)
        self.tools.set_runtime_base_path(self.workspace_base_path)
        self.session_manager = get_session_manager()
        self.session: Session | None = None
        self._initialized = False
        self.max_iterations = 40  # Max tool calls per message
        self.last_usage: dict[str, int] = self._empty_usage()
        self.total_usage: dict[str, int] = self._empty_usage()
        self.last_context_window: dict[str, int | float] = {}
        self._last_memory_debug_signature: str | None = None
        self.pipeline_mode: str = "loop"  # "loop" (fast/simple) | "contracts" (planner+critic)
        self.planning_enabled: bool = False
        self.plan_mode_auto: bool = False  # /planning on: auto-route chat → /plan + /plan-execute
        self.plan_mode_level: str = "plain"  # plain | enriched | insightful | complete
        self._is_worker: bool = False  # Set True for orchestrator worker agents
        self._last_complete_success: bool = True  # Updated by finish() in complete()
        self.monitor_trace_llm: bool = bool(getattr(cfg.ui, "monitor_trace_llm", False))
        self.monitor_trace_pipeline: bool = bool(getattr(cfg.ui, "monitor_trace_pipeline", True))
        self.instructions = InstructionLoader()
        self._provider_override = provider is not None
        self._runtime_model_details: dict[str, Any] = {}
        self._skills_snapshot_cache = None
        self.memory = None
        self._last_semantic_memory_debug_signature: str | None = None
        # File registry for cross-task/cross-session file path resolution.
        # Set per orchestration run or per session for single-agent mode.
        self._file_registry: FileRegistry | None = None
        # External cancellation event — set by the UI layer (TUI/web) when
        # the user presses Ctrl+C / ESC or sends a cancel WS message.
        # The main iteration loop checks this at the top of every iteration
        # and breaks cleanly rather than running to completion.
        self.cancel_event: asyncio.Event = asyncio.Event()
        # LLM session logger — writes every LLM call to logs/<session>/session_log.md
        self.llm_session_logging: bool = bool(getattr(cfg.logging, "llm_session_logging", False))
        self._llm_logger = get_llm_session_logger(self.workspace_base_path / "logs")
        self._refresh_runtime_model_details(source="startup")

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        """Create an empty usage bucket."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    @staticmethod
    def _is_monitor_only_tool_name(tool_name: str) -> bool:
        """Whether tool output is monitor-only and should not feed model context."""
        normalized = str(tool_name or "").strip().lower()
        return normalized in {
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
        }

    # ── Activity timing ──────────────────────────────────────────────
    # We stamp a few coarse timestamps into session metadata so the agent
    # always knows, in addition to the current clock, when things last
    # happened: the last real user message, its own last reply, and the
    # last automated (cron/scheduler) run. Injected into the system prompt
    # each turn so the model can reason about idle gaps and recency.

    def _record_timing_event(self, kind: str) -> None:
        """Stamp a timing event (UTC ISO) into ``session.metadata['timing']``."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return
        timing = self.session.metadata.get("timing")
        if not isinstance(timing, dict):
            timing = {}
            self.session.metadata["timing"] = timing
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        timing[kind] = now_iso
        # First activity we ever see for this session doubles as its start.
        timing.setdefault("session_started_at", now_iso)

    @staticmethod
    def _humanize_age(seconds: float) -> str:
        """Compact relative-age string: '12s', '5m', '3h', '2d' ago."""
        s = int(max(0, seconds))
        if s < 60:
            return f"{s}s ago" if s else "just now"
        m = s // 60
        if m < 60:
            return f"{m}m ago"
        h = m // 60
        if h < 24:
            return f"{h}h ago"
        return f"{h // 24}d ago"

    @staticmethod
    def _humanize_until(seconds: float) -> str:
        """Compact forward-looking string: 'in 5m', 'in 3h', 'in 2d'.

        Rounds to the nearest unit (not floor) so an ETA a few microseconds
        under 2h reads 'in 2h', not a misleading 'in 1h'."""
        s = max(0.0, float(seconds))
        if s < 60:
            return "in <1m"
        minutes = s / 60
        if minutes < 60:
            return f"in {round(minutes)}m"
        hours = minutes / 60
        if hours < 24:
            return f"in {round(hours)}h"
        return f"in {round(hours / 24)}d"

    @staticmethod
    def _part_of_day(hour: int) -> str:
        """Map an hour (0-23) to a coarse part-of-day label."""
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"

    def _build_timing_block(self, detail_level: str = "normal") -> str:
        """Render the activity-timing lines for the system prompt.

        Returns ``""`` for nano prompts or when nothing has been recorded."""
        if detail_level == "nano":
            return ""
        if not self.session or not isinstance(self.session.metadata, dict):
            return ""
        timing = self.session.metadata.get("timing")
        if not isinstance(timing, dict) or not timing:
            return ""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        def _ago(iso: str) -> str | None:
            try:
                dt = datetime.fromisoformat(str(iso))
            except (ValueError, TypeError):
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return self._humanize_age((now - dt).total_seconds())

        # Next scheduled/cron run ETA (forward-looking), from the cache the
        # async refresh populated before this turn.
        next_cron = getattr(self, "_next_cron_cache", None)
        next_run_str = ""
        if isinstance(next_cron, dict) and next_cron.get("eta_iso"):
            try:
                dt = datetime.fromisoformat(str(next_cron["eta_iso"]))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                until = self._humanize_until((dt - now).total_seconds())
                label = str(next_cron.get("label", "") or "").strip()
                next_run_str = f"{until} ({label})" if label else until
            except (ValueError, TypeError):
                next_run_str = ""

        # Part-of-day + weekday, from LOCAL time (the absolute clock is already
        # in the system-info block; this is a derived tone hint).
        local_now = datetime.now()
        weekday = local_now.strftime("%A")
        is_weekend = local_now.weekday() >= 5
        part = self._part_of_day(local_now.hour)
        when_str = f"{weekday} {part}" + (" (weekend)" if is_weekend else "")

        # Conversation cadence: user+assistant message count this session.
        msg_count = 0
        try:
            msg_count = sum(
                1 for m in (self.session.messages or [])
                if str(m.get("role", "")).strip().lower() in ("user", "assistant")
            )
        except Exception:
            msg_count = 0

        # (key, full label, compact label)
        rows = (
            ("last_user_msg_at", "Last user message", "user"),
            ("last_assistant_at", "Last reply", "reply"),
            ("last_cron_at", "Last scheduled/cron run", "cron"),
            ("session_started_at", "Session started", "started"),
        )
        if detail_level == "micro":
            parts = [
                f"{compact} {_ago(timing[key])}"
                for key, _full, compact in rows
                if timing.get(key) and _ago(timing[key])
            ]
            parts.append(when_str.lower())
            if msg_count:
                parts.append(f"{msg_count} msgs")
            if next_run_str:
                parts.append(f"next {next_run_str}")
            return ("Activity: " + " | ".join(parts)) if parts else ""
        lines = []
        for key, full, _compact in rows:
            iso = timing.get(key)
            if not iso:
                continue
            ago = _ago(iso)
            if not ago:
                continue
            lines.append(f"- {full}: {ago}")
        lines.append(f"- Local time: {when_str}")
        if msg_count:
            lines.append(f"- Conversation: {msg_count} messages this session")
        if next_run_str:
            lines.append(f"- Next scheduled run: {next_run_str}")
        if not lines:
            return ""
        return "Activity timing:\n" + "\n".join(lines)

    @staticmethod
    def _accumulate_usage(target: dict[str, int], usage: dict[str, int] | None) -> None:
        """Add usage values into target totals."""
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        total = int(usage.get("total_tokens", prompt + completion))
        target["prompt_tokens"] += prompt
        target["completion_tokens"] += completion
        target["total_tokens"] += total
        # Cache tokens (Anthropic via LiteLLM)
        target.setdefault("cache_creation_input_tokens", 0)
        target.setdefault("cache_read_input_tokens", 0)
        target["cache_creation_input_tokens"] += int(usage.get("cache_creation_input_tokens", 0))
        target["cache_read_input_tokens"] += int(usage.get("cache_read_input_tokens", 0))

    def _finalize_turn_usage(self, turn_usage: dict[str, int]) -> None:
        """Persist usage for the last turn and aggregate global totals."""
        self.last_usage = turn_usage
        self._accumulate_usage(self.total_usage, turn_usage)

    def _set_runtime_status(self, status: str) -> None:
        """Forward runtime status updates when callback is configured."""
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception:
                pass

    def _emit_thinking(self, text: str, tool: str = "", phase: str = "tool") -> None:
        """Forward inline thinking/reasoning updates to the UI."""
        if self.thinking_callback:
            try:
                self.thinking_callback(text, tool, phase)
            except Exception:
                pass

    def _emit_narration(self, text: str, iteration: int = 0) -> bool:
        """Forward an intermediate narration blurb — the short text the model
        emits alongside its tool calls (e.g. "Now let me apply the edits…") — to
        the UI and channel bridges as a LIVE progress event, so the user sees
        what's happening during a long task instead of only at the end.

        Returns True when a callback is wired, so the caller can skip folding
        the same text into the final combined answer (avoids showing it twice)."""
        cb = getattr(self, "narration_callback", None)
        if not cb:
            return False
        try:
            cb(text, iteration)
        except Exception:
            pass
        return True

    def _emit_tool_output(self, tool_name: str, arguments: dict[str, Any], output: str) -> None:
        """Forward raw tool output to UI callback when configured."""
        if self.session and tool_name in {"planning", "task_contract", "task_rephrase", "completion_gate", "scale_micro_loop", "memory_select", "memory_semantic_select", "memory_deep_select"}:
            self._add_session_message(
                role="tool",
                content=str(output or ""),
                tool_name=tool_name,
                tool_arguments=arguments if isinstance(arguments, dict) else {},
            )
            if self.monitor_trace_pipeline:
                trace_payload = self._build_pipeline_trace_payload(
                    source_tool=tool_name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
                trace_text = json.dumps(trace_payload, ensure_ascii=True, sort_keys=True)
                self._add_session_message(
                    role="tool",
                    content=trace_text,
                    tool_name="pipeline_trace",
                    tool_arguments=trace_payload,
                )
        if not self.tool_output_callback:
            return
        try:
            self.tool_output_callback(tool_name, arguments, output)
        except Exception:
            pass

    def _log_llm_call(
        self,
        interaction_label: str,
        messages: list[Message],
        response: Any,
        tools_enabled: bool = False,
        max_tokens: int | None = None,
        instruction_files: list[str] | None = None,
    ) -> None:
        """Write an LLM call entry to the file-based session log."""
        if not self.llm_session_logging:
            return
        try:
            # Ensure logger is pointed at the current session
            slug = self._current_session_slug()
            self._llm_logger.set_session(slug)

            # Drain recently-loaded instruction files if not explicitly provided.
            if not instruction_files:
                instruction_files = self.instructions.drain_recent_files()

            model = str(getattr(self.provider, "model", "") or "")
            self._llm_logger.log_call(
                interaction_label=interaction_label,
                model=model,
                messages=messages,
                response=response,
                instruction_files=instruction_files,
                tools_enabled=tools_enabled,
                max_tokens=max_tokens,
            )
        except Exception:
            pass
