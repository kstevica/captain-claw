"""Agent orchestration for Captain Claw."""

import json
from pathlib import Path
from typing import Any
from typing import Callable

from captain_claw.agent_context_mixin import AgentContextMixin
from captain_claw.agent_file_ops_mixin import AgentFileOpsMixin
from captain_claw.agent_guard_mixin import AgentGuardMixin
from captain_claw.agent_model_mixin import AgentModelMixin
from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin
from captain_claw.agent_pipeline_mixin import AgentPipelineMixin
from captain_claw.agent_reasoning_mixin import AgentReasoningMixin
from captain_claw.agent_session_mixin import AgentSessionMixin
from captain_claw.agent_skills_mixin import AgentSkillsMixin
from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin
from captain_claw.config import get_config
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import LLMProvider
from captain_claw.session import Session, get_session_manager
from captain_claw.tools import get_tool_registry


class Agent(
    AgentOrchestrationMixin,
    AgentFileOpsMixin,
    AgentContextMixin,
    AgentGuardMixin,
    AgentModelMixin,
    AgentPipelineMixin,
    AgentReasoningMixin,
    AgentSessionMixin,
    AgentSkillsMixin,
    AgentToolLoopMixin,
):
    """Main agent orchestrator."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        status_callback: Callable[[str], None] | None = None,
        tool_output_callback: Callable[[str, dict[str, Any], str], None] | None = None,
        approval_callback: Callable[[str], bool] | None = None,
    ):
        """Initialize the agent.
        
        Args:
            provider: Optional LLM provider override
            status_callback: Optional runtime status callback
            approval_callback: Optional callback for guard approval prompts
        """
        self.provider = provider
        self.status_callback = status_callback
        self.tool_output_callback = tool_output_callback
        self.approval_callback = approval_callback
        self.tools = get_tool_registry()
        self.tools.set_approval_callback(self.approval_callback)
        self.runtime_base_path = Path.cwd().resolve()
        cfg = get_config()
        self.workspace_base_path = cfg.resolved_workspace_path(self.runtime_base_path)
        self.tools.set_runtime_base_path(self.workspace_base_path)
        self.session_manager = get_session_manager()
        self.session: Session | None = None
        self._initialized = False
        self.max_iterations = 10  # Max tool calls per message
        self.last_usage: dict[str, int] = self._empty_usage()
        self.total_usage: dict[str, int] = self._empty_usage()
        self.last_context_window: dict[str, int | float] = {}
        self._last_memory_debug_signature: str | None = None
        self.pipeline_mode: str = "loop"  # "loop" (fast/simple) | "contracts" (planner+critic)
        self.planning_enabled: bool = False
        self.monitor_trace_llm: bool = bool(getattr(cfg.ui, "monitor_trace_llm", False))
        self.monitor_trace_pipeline: bool = bool(getattr(cfg.ui, "monitor_trace_pipeline", True))
        self.instructions = InstructionLoader()
        self._provider_override = provider is not None
        self._runtime_model_details: dict[str, Any] = {}
        self._skills_snapshot_cache = None
        self.memory = None
        self._last_semantic_memory_debug_signature: str | None = None
        self._refresh_runtime_model_details(source="startup")

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        """Create an empty usage bucket."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _is_monitor_only_tool_name(tool_name: str) -> bool:
        """Whether tool output is monitor-only and should not feed model context."""
        normalized = str(tool_name or "").strip().lower()
        return normalized in {
            "llm_trace",
            "planning",
            "task_contract",
            "completion_gate",
            "pipeline_trace",
            "telegram",
        }

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

    def _emit_tool_output(self, tool_name: str, arguments: dict[str, Any], output: str) -> None:
        """Forward raw tool output to UI callback when configured."""
        if self.session and tool_name in {"planning", "task_contract", "completion_gate"}:
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
