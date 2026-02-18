"""Agent orchestration for Captain Claw."""

import asyncio
import copy
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any, AsyncIterator
from typing import Callable

from captain_claw.config import get_config
from captain_claw.exceptions import GuardBlockedError
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import (
    LLMProvider,
    Message,
    ToolCall,
    ToolDefinition,
    create_provider,
    get_provider,
    set_provider,
)
from captain_claw.logging import get_logger
from captain_claw.tools import ToolRegistry, get_tool_registry
from captain_claw.session import Session, get_session_manager

log = get_logger(__name__)


class Agent:
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
        return normalized in {"llm_trace", "planning", "task_contract", "completion_gate", "pipeline_trace"}

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

    @staticmethod
    def _build_pipeline_trace_payload(
        source_tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Build compact pipeline-only trace payload without large content bodies."""
        payload: dict[str, Any] = {"source": str(source_tool or "").strip().lower()}
        args = arguments if isinstance(arguments, dict) else {}
        src = payload["source"]

        if src == "planning":
            for key in (
                "event",
                "mode",
                "enabled",
                "current_index",
                "current_task_id",
                "leaf_tasks",
                "leaf_index",
                "leaf_remaining",
                "current_path",
                "eta_seconds",
                "eta_text",
            ):
                if key in args:
                    payload[key] = args.get(key)
            raw_scopes = args.get("scope_progress")
            compact_scopes: list[dict[str, Any]] = []
            if isinstance(raw_scopes, list):
                for scope in raw_scopes:
                    if not isinstance(scope, dict):
                        continue
                    compact_scopes.append({
                        "level": scope.get("level"),
                        "path": scope.get("path"),
                        "index": scope.get("index"),
                        "siblings_total": scope.get("siblings_total"),
                        "siblings_remaining": scope.get("siblings_remaining"),
                        "scope_leaf_total": scope.get("scope_leaf_total"),
                        "scope_leaf_remaining": scope.get("scope_leaf_remaining"),
                        "eta_seconds": scope.get("eta_seconds"),
                        "eta_text": scope.get("eta_text"),
                    })
            payload["scope_progress"] = compact_scopes
            return payload

        if src == "completion_gate":
            for key in (
                "step",
                "passed",
                "failed_count",
                "base_limit",
                "effective_limit",
                "hard_limit",
                "previous_limit",
                "new_limit",
                "soft_limit",
                "recent_progress",
                "remaining_work",
                "stagnant_iterations",
                "iteration",
            ):
                if key in args:
                    payload[key] = args.get(key)
            return payload

        if src == "task_contract":
            for key in ("step", "missing"):
                if key in args:
                    payload[key] = args.get(key)
            return payload

        for key, value in args.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return payload

    def _emit_llm_trace(
        self,
        interaction_label: str,
        response: Any,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
    ) -> None:
        """Emit/export full intermediate LLM response for process analysis."""
        if not self.monitor_trace_llm:
            return
        tool_calls: list[dict[str, Any]] = []
        for call in list(getattr(response, "tool_calls", []) or []):
            tool_calls.append({
                "id": str(getattr(call, "id", "")),
                "name": str(getattr(call, "name", "")),
                "arguments": getattr(call, "arguments", {}),
            })

        model_name = str(getattr(response, "model", "") or "")
        usage = getattr(response, "usage", {}) or {}
        args = {
            "interaction": interaction_label,
            "model": model_name,
            "messages": len(messages),
            "tools_enabled": bool(tools),
            "max_tokens": int(max_tokens) if isinstance(max_tokens, int) else None,
            "tool_calls": len(tool_calls),
            "usage": usage if isinstance(usage, dict) else {},
        }
        response_text = str(getattr(response, "content", "") or "")
        output_lines = [
            f"interaction={interaction_label}",
            f"model={model_name or '(unknown)'}",
            f"messages={len(messages)}",
            f"tools_enabled={bool(tools)}",
            f"max_tokens={max_tokens if isinstance(max_tokens, int) else '(default)'}",
            f"tool_calls={len(tool_calls)}",
            "",
            "[assistant_response]",
            response_text if response_text else "(empty)",
        ]
        if tool_calls:
            output_lines.extend([
                "",
                "[tool_calls]",
                json.dumps(tool_calls, ensure_ascii=True, indent=2),
            ])
        output = "\n".join(output_lines).rstrip()
        self._emit_tool_output("llm_trace", args, output)
        self._add_session_message(
            role="tool",
            content=output,
            tool_name="llm_trace",
            tool_arguments=args,
        )

    @staticmethod
    def _truncate_guard_text(text: str, max_chars: int = 12000) -> str:
        """Trim guard payloads to bounded size."""
        cleaned = (text or "").strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "\n...[truncated for guard evaluation]"

    def _guard_settings(self, guard_type: str) -> tuple[bool, str]:
        """Return (enabled, level) for a guard type."""
        cfg = get_config()
        guards = getattr(cfg, "guards", None)
        if guards is None:
            return False, "stop_suspicious"
        raw = getattr(guards, guard_type, None)
        if raw is None:
            return False, "stop_suspicious"
        enabled = bool(getattr(raw, "enabled", False))
        level = str(getattr(raw, "level", "stop_suspicious") or "stop_suspicious").strip().lower()
        if level not in {"stop_suspicious", "ask_for_approval"}:
            level = "stop_suspicious"
        return enabled, level

    def guards_enabled(self) -> bool:
        """Whether any guard type is enabled."""
        return any(self._guard_settings(kind)[0] for kind in ("input", "output", "script_tool"))

    def _serialize_messages_for_guard(self, messages: list[Message], max_chars: int = 12000) -> str:
        """Serialize outbound prompt messages for input guard checks."""
        lines: list[str] = []
        for idx, msg in enumerate(messages, start=1):
            role = str(getattr(msg, "role", "")).strip().lower() or "unknown"
            content = re.sub(r"\s+", " ", str(getattr(msg, "content", "")).strip())
            if len(content) > 800:
                content = content[:800].rstrip() + "... [truncated]"
            lines.append(f"{idx}. {role}: {content}")
        return self._truncate_guard_text("\n".join(lines), max_chars=max_chars)

    @staticmethod
    def _parse_guard_decision(raw_text: str) -> dict[str, Any]:
        """Parse guard classifier output into a normalized decision payload."""
        text = (raw_text or "").strip()
        if not text:
            return {"allow": False, "reason": "Guard model returned empty output."}

        payload: dict[str, Any] | None = None
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                payload = value
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    value = json.loads(match.group(0))
                    if isinstance(value, dict):
                        payload = value
                except Exception:
                    payload = None

        if payload is None:
            lowered = text.lower()
            if "allow" in lowered and not any(word in lowered for word in ("suspicious", "malicious", "deny", "block")):
                return {"allow": True, "reason": "Allowed by non-JSON guard output."}
            return {"allow": False, "reason": "Could not parse guard output as JSON."}

        verdict = str(payload.get("verdict") or payload.get("decision") or "").strip().lower()
        reason = str(payload.get("reason") or payload.get("explanation") or "").strip()
        if verdict in {"allow", "safe", "ok", "pass"}:
            return {"allow": True, "reason": reason or "Guard allowed."}
        if verdict in {"suspicious", "block", "blocked", "deny", "denied", "malicious"}:
            return {"allow": False, "reason": reason or "Guard flagged suspicious content."}

        # Conservative fallback when model output shape is unexpected.
        return {"allow": False, "reason": reason or "Guard decision was inconclusive."}

    async def _run_guard_decision(
        self,
        guard_type: str,
        interaction_label: str,
        content: str,
        turn_usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Evaluate guard decision using prompt templates."""
        system_template = f"guard_{guard_type}_system_prompt.md"
        user_template = f"guard_{guard_type}_user_prompt.md"
        rendered_content = self._truncate_guard_text(content)
        messages = [
            Message(role="system", content=self.instructions.load(system_template)),
            Message(
                role="user",
                content=self.instructions.render(
                    user_template,
                    interaction_label=interaction_label,
                    content=rendered_content,
                ),
            ),
        ]
        try:
            response = await self.provider.complete(
                messages=messages,
                tools=None,
                max_tokens=400,
            )
            if turn_usage is not None:
                self._accumulate_usage(turn_usage, response.usage or {})
            parsed = self._parse_guard_decision(response.content or "")
            parsed["raw"] = response.content or ""
            return parsed
        except Exception as e:
            return {"allow": False, "reason": f"Guard evaluation failed: {e}", "raw": ""}

    def _request_guard_approval(self, question: str) -> bool:
        """Request user approval when guard level is ask_for_approval."""
        if not self.approval_callback:
            return False
        try:
            return bool(self.approval_callback(question))
        except Exception:
            return False

    async def _enforce_guard(
        self,
        guard_type: str,
        interaction_label: str,
        content: str,
        turn_usage: dict[str, int] | None = None,
    ) -> tuple[bool, str]:
        """Run one guard type and enforce configured policy."""
        enabled, level = self._guard_settings(guard_type)
        if not enabled:
            return True, ""
        if guard_type == "output" and not (content or "").strip():
            return True, ""

        decision = await self._run_guard_decision(
            guard_type=guard_type,
            interaction_label=interaction_label,
            content=content,
            turn_usage=turn_usage,
        )
        allow = bool(decision.get("allow", False))
        reason = str(decision.get("reason", "")).strip() or "Suspicious content detected."
        raw = str(decision.get("raw", "")).strip()

        if allow:
            self._emit_tool_output(
                f"guard_{guard_type}",
                {"interaction": interaction_label, "decision": "allow", "level": level},
                reason,
            )
            return True, ""

        if level == "ask_for_approval":
            question = (
                f"{guard_type} guard flagged suspicious content for {interaction_label}. "
                f"Reason: {reason} Approve anyway?"
            )
            approved = self._request_guard_approval(question)
            self._emit_tool_output(
                f"guard_{guard_type}",
                {
                    "interaction": interaction_label,
                    "decision": "suspicious",
                    "level": level,
                    "approved": approved,
                },
                reason if not raw else f"{reason}\nRaw guard output: {raw}",
            )
            if approved:
                return True, ""
            return False, f"Blocked by {guard_type} guard (approval denied): {reason}"

        self._emit_tool_output(
            f"guard_{guard_type}",
            {"interaction": interaction_label, "decision": "suspicious", "level": level},
            reason if not raw else f"{reason}\nRaw guard output: {raw}",
        )
        return False, f"Blocked by {guard_type} guard: {reason}"

    async def _complete_with_guards(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        interaction_label: str = "conversation",
        turn_usage: dict[str, int] | None = None,
        max_tokens: int | None = None,
    ):
        """Run guarded LLM completion (input + output guards)."""
        guard_payload = self._serialize_messages_for_guard(messages)
        allowed_input, input_error = await self._enforce_guard(
            guard_type="input",
            interaction_label=interaction_label,
            content=guard_payload,
            turn_usage=turn_usage,
        )
        if not allowed_input:
            raise GuardBlockedError("input", input_error)

        response = await self.provider.complete(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        if turn_usage is not None:
            self._accumulate_usage(turn_usage, response.usage or {})

        allowed_output, output_error = await self._enforce_guard(
            guard_type="output",
            interaction_label=interaction_label,
            content=str(response.content or ""),
            turn_usage=turn_usage,
        )
        if not allowed_output:
            raise GuardBlockedError("output", output_error)

        self._emit_llm_trace(
            interaction_label=interaction_label,
            response=response,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

        return response

    async def _execute_tool_with_guard(
        self,
        name: str,
        arguments: dict[str, Any],
        interaction_label: str,
        turn_usage: dict[str, int] | None = None,
    ):
        """Execute a tool after script/tool guard policy check."""
        guard_payload = json.dumps(
            {
                "tool_name": name,
                "arguments": arguments,
                "interaction": interaction_label,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        allowed, guard_error = await self._enforce_guard(
            guard_type="script_tool",
            interaction_label=interaction_label,
            content=guard_payload,
            turn_usage=turn_usage,
        )
        if not allowed:
            raise GuardBlockedError("script_tool", guard_error)
        return await self.tools.execute(
            name=name,
            arguments=arguments,
            session_id=self._current_session_slug(),
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens with provider support and safe fallback."""
        if not text:
            return 0
        if self.provider:
            try:
                return max(0, int(self.provider.count_tokens(text)))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _ensure_message_token_count(self, msg: dict[str, Any]) -> int:
        """Ensure persisted token_count exists on a session message."""
        value = msg.get("token_count")
        if isinstance(value, int) and value >= 0:
            return value
        count = self._count_tokens(str(msg.get("content", "")))
        msg["token_count"] = count
        return count

    def _session_token_count(self, messages: list[dict[str, Any]] | None = None) -> int:
        """Count total tokens for session messages."""
        source = messages if messages is not None else (self.session.messages if self.session else [])
        total = 0
        for msg in source:
            total += self._ensure_message_token_count(msg)
        return total

    @staticmethod
    def _compact_role(role: str) -> str:
        """Normalize role label for summaries."""
        normalized = (role or "").strip().lower()
        return normalized if normalized else "unknown"

    def _format_compaction_messages(
        self,
        messages: list[dict[str, Any]],
        max_total_chars: int = 24000,
        max_item_chars: int = 600,
    ) -> str:
        """Format messages for compaction summarization prompt."""
        lines: list[str] = []
        consumed = 0
        for idx, msg in enumerate(messages, start=1):
            role = self._compact_role(str(msg.get("role", "")))
            tool_name = str(msg.get("tool_name", "")).strip()
            label = f"{idx}. {role}"
            if tool_name:
                label += f"({tool_name})"
            content = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
            if len(content) > max_item_chars:
                content = content[:max_item_chars].rstrip() + "... [truncated]"
            line = f"{label}: {content}"
            if consumed + len(line) > max_total_chars:
                lines.append("[... older conversation excerpt truncated for compaction ...]")
                break
            lines.append(line)
            consumed += len(line)
        return "\n".join(lines)

    def _fallback_compaction_summary(self, messages: list[dict[str, Any]]) -> str:
        """Fallback summary when LLM-based compaction summarization fails."""
        if not messages:
            return "No prior messages available for summary."
        highlights: list[str] = []
        for msg in messages[-8:]:
            role = self._compact_role(str(msg.get("role", "")))
            content = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
            if not content:
                continue
            snippet = content[:180].rstrip()
            if len(content) > 180:
                snippet += "..."
            highlights.append(f"- {role}: {snippet}")
        if not highlights:
            return "Prior conversation compacted."
        return "Key points from earlier conversation:\n" + "\n".join(highlights)

    async def _summarize_for_compaction(self, messages: list[dict[str, Any]]) -> str:
        """Summarize older messages for long-session compaction."""
        if not messages:
            return "No prior messages available for summary."

        formatted = self._format_compaction_messages(messages)
        if not formatted.strip():
            return self._fallback_compaction_summary(messages)

        prompt = self.instructions.render(
            "compaction_summary_user_prompt.md",
            formatted=formatted,
        )
        rewrite_messages = [
            Message(
                role="system",
                content=self.instructions.load("compaction_summary_system_prompt.md"),
            ),
            Message(role="user", content=prompt),
        ]
        try:
            max_tokens = min(2048, int(get_config().model.max_tokens))
            response = await self._complete_with_guards(
                messages=rewrite_messages,
                tools=None,
                interaction_label="compaction_summary",
                max_tokens=max_tokens,
            )
            summary = (response.content or "").strip()
            if summary:
                return summary
        except Exception as e:
            log.warning("Compaction summarization failed, using fallback", error=str(e))
        return self._fallback_compaction_summary(messages)

    @staticmethod
    def _limit_description_sentences(text: str, max_sentences: int = 5) -> str:
        """Normalize description text and cap sentence count."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return ""
        if max_sentences <= 0:
            return cleaned
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
        if not parts:
            return cleaned
        return " ".join(parts[:max_sentences]).strip()

    def sanitize_session_description(self, text: str, max_sentences: int = 5) -> str:
        """Public helper to normalize session descriptions."""
        return self._limit_description_sentences(text, max_sentences=max_sentences)

    def _fallback_session_description(self, target_session: Session) -> str:
        """Build deterministic description when LLM description generation fails."""
        user_messages = [
            re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
            for msg in target_session.messages
            if str(msg.get("role", "")).strip().lower() == "user"
        ]
        tool_names = [
            str(msg.get("tool_name", "")).strip()
            for msg in target_session.messages
            if str(msg.get("role", "")).strip().lower() == "tool" and str(msg.get("tool_name", "")).strip()
        ]

        pieces: list[str] = []
        if user_messages:
            latest = user_messages[-1]
            if len(latest) > 180:
                latest = latest[:180].rstrip() + "..."
            pieces.append(f"Active focus: {latest}")
        if tool_names:
            unique_tools: list[str] = []
            for tool_name in tool_names:
                if tool_name in unique_tools:
                    continue
                unique_tools.append(tool_name)
            pieces.append(f"Common tools used: {', '.join(unique_tools[:5])}.")
        pieces.append(
            f'Session "{target_session.name}" has {len(target_session.messages)} messages and captures an ongoing task thread.'
        )

        raw = " ".join(pieces).strip()
        return self._limit_description_sentences(raw, max_sentences=5)

    async def generate_session_description(
        self,
        target_session: Session | None = None,
        max_sentences: int = 5,
    ) -> str:
        """Generate a short session description from session context and tasks."""
        session = target_session or self.session
        if not session:
            return ""
        if not session.messages:
            return self._limit_description_sentences(
                f'Session "{session.name}" has no conversation history yet.',
                max_sentences=max_sentences,
            )

        formatted = self._format_compaction_messages(
            session.messages,
            max_total_chars=18000,
            max_item_chars=500,
        )
        if not formatted.strip():
            return self._fallback_session_description(session)

        messages = [
            Message(
                role="system",
                content=self.instructions.load("session_description_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "session_description_user_prompt.md",
                    session_name=session.name,
                    max_sentences=max_sentences,
                    conversation_excerpt=formatted,
                ),
            ),
        ]
        try:
            max_tokens = min(400, int(get_config().model.max_tokens))
            self._set_runtime_status("thinking")
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="session_description",
                max_tokens=max_tokens,
            )
            generated = self._limit_description_sentences(
                response.content or "",
                max_sentences=max_sentences,
            )
            if generated:
                return generated
        except Exception as e:
            log.warning("Session description generation failed, using fallback", error=str(e))

        return self._fallback_session_description(session)

    async def compact_session(
        self,
        force: bool = False,
        trigger: str = "manual",
    ) -> tuple[bool, dict[str, Any]]:
        """Compact session by summarizing older messages and keeping recent context."""
        if not self.session:
            return False, {"reason": "no_session"}

        cfg = get_config()
        messages = self.session.messages
        if len(messages) < 3:
            return False, {"reason": "too_few_messages", "message_count": len(messages)}

        max_tokens = max(1, int(cfg.context.max_tokens))
        threshold_tokens = max(1, int(max_tokens * float(cfg.context.compaction_threshold)))
        total_tokens = self._session_token_count(messages)
        if not force and total_tokens <= threshold_tokens:
            return False, {
                "reason": "below_threshold",
                "total_tokens": total_tokens,
                "threshold_tokens": threshold_tokens,
            }

        ratio = float(cfg.context.compaction_ratio)
        ratio = min(max(ratio, 0.05), 0.95)
        target_recent_tokens = max(1, int(max_tokens * ratio))

        max_keep_if_compacting = max(1, len(messages) - 1)
        min_keep_messages = min(4, max_keep_if_compacting)
        keep_count = 0
        kept_tokens = 0
        for msg in reversed(messages):
            token_count = self._ensure_message_token_count(msg)
            if keep_count < min_keep_messages or kept_tokens + token_count <= target_recent_tokens:
                keep_count += 1
                kept_tokens += token_count
                continue
            break

        if keep_count >= len(messages):
            if force and len(messages) > 1:
                keep_count = len(messages) - 1
            else:
                return False, {"reason": "nothing_to_compact", "message_count": len(messages)}

        old_messages = messages[:-keep_count]
        recent_messages = messages[-keep_count:]
        summary_text = await self._summarize_for_compaction(old_messages)
        summary_content = (
            "Conversation summary of earlier messages (compacted memory):\n"
            f"{summary_text.strip()}"
        )
        now_iso = datetime.now(UTC).isoformat()
        summary_message = {
            "role": "assistant",
            "content": summary_content,
            "tool_call_id": None,
            "tool_name": "compaction_summary",
            "tool_arguments": {
                "trigger": trigger,
                "compacted_messages": len(old_messages),
                "kept_messages": len(recent_messages),
            },
            "token_count": self._count_tokens(summary_content),
            "timestamp": now_iso,
        }

        self.session.messages = [summary_message, *recent_messages]
        self.session.updated_at = now_iso

        after_tokens = self._session_token_count(self.session.messages)
        compact_meta = self.session.metadata.setdefault("compaction", {})
        compact_meta["count"] = int(compact_meta.get("count", 0)) + 1
        compact_meta[f"{trigger}_count"] = int(compact_meta.get(f"{trigger}_count", 0)) + 1
        compact_meta["last_trigger"] = trigger
        compact_meta["last_before_tokens"] = total_tokens
        compact_meta["last_after_tokens"] = after_tokens
        compact_meta["last_compacted_messages"] = len(old_messages)
        compact_meta["last_kept_messages"] = len(recent_messages)
        compact_meta["last_compacted_at"] = now_iso

        await self.session_manager.save_session(self.session)

        monitor_output = (
            f"Compaction trigger={trigger}\n"
            f"before_tokens={total_tokens}\n"
            f"after_tokens={after_tokens}\n"
            f"compacted_messages={len(old_messages)}\n"
            f"kept_messages={len(recent_messages)}"
        )
        self._emit_tool_output(
            "compaction",
            {
                "trigger": trigger,
                "force": force,
                "before_tokens": total_tokens,
                "after_tokens": after_tokens,
            },
            monitor_output,
        )

        return True, {
            "before_tokens": total_tokens,
            "after_tokens": after_tokens,
            "compacted_messages": len(old_messages),
            "kept_messages": len(recent_messages),
            "trigger": trigger,
        }

    async def _compact_messages_snapshot(
        self,
        messages: list[dict[str, Any]],
        trigger: str = "procreate",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compact a copied message list without mutating source session messages."""
        snapshot = copy.deepcopy(messages)
        if len(snapshot) < 3:
            return snapshot, {"reason": "too_few_messages", "message_count": len(snapshot)}

        cfg = get_config()
        max_tokens = max(1, int(cfg.context.max_tokens))
        ratio = float(cfg.context.compaction_ratio)
        ratio = min(max(ratio, 0.05), 0.95)
        target_recent_tokens = max(1, int(max_tokens * ratio))

        max_keep_if_compacting = max(1, len(snapshot) - 1)
        min_keep_messages = min(4, max_keep_if_compacting)
        keep_count = 0
        kept_tokens = 0
        for msg in reversed(snapshot):
            token_count = self._ensure_message_token_count(msg)
            if keep_count < min_keep_messages or kept_tokens + token_count <= target_recent_tokens:
                keep_count += 1
                kept_tokens += token_count
                continue
            break

        if keep_count >= len(snapshot):
            if len(snapshot) > 1:
                keep_count = len(snapshot) - 1
            else:
                return snapshot, {"reason": "nothing_to_compact", "message_count": len(snapshot)}

        old_messages = snapshot[:-keep_count]
        recent_messages = snapshot[-keep_count:]
        summary_text = await self._summarize_for_compaction(old_messages)
        summary_content = (
            "Conversation summary of earlier messages (compacted memory):\n"
            f"{summary_text.strip()}"
        )
        now_iso = datetime.now(UTC).isoformat()
        summary_message = {
            "role": "assistant",
            "content": summary_content,
            "tool_call_id": None,
            "tool_name": "compaction_summary",
            "tool_arguments": {
                "trigger": trigger,
                "compacted_messages": len(old_messages),
                "kept_messages": len(recent_messages),
            },
            "token_count": self._count_tokens(summary_content),
            "timestamp": now_iso,
        }
        compacted_messages = [summary_message, *recent_messages]
        return compacted_messages, {
            "compacted_messages": len(old_messages),
            "kept_messages": len(recent_messages),
            "before_messages": len(snapshot),
            "after_messages": len(compacted_messages),
            "trigger": trigger,
        }

    def is_session_memory_protected(self) -> bool:
        """Return whether current session blocks memory reset operations."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return False

        protection_meta = self.session.metadata.get("memory_protection")
        if isinstance(protection_meta, dict):
            return bool(protection_meta.get("enabled", False))

        legacy_value = self.session.metadata.get("memory_protected")
        if isinstance(legacy_value, bool):
            return legacy_value
        return False

    async def set_session_memory_protection(
        self,
        enabled: bool,
        persist: bool = True,
    ) -> tuple[bool, str]:
        """Enable or disable memory reset protection for the active session."""
        if not self.session:
            return False, "No active session"

        protection_meta = self.session.metadata.setdefault("memory_protection", {})
        protection_meta["enabled"] = bool(enabled)
        protection_meta["updated_at"] = datetime.now(UTC).isoformat()
        if persist:
            await self.session_manager.save_session(self.session)
        if enabled:
            return True, "Session memory protection enabled"
        return True, "Session memory protection disabled"

    async def procreate_sessions(
        self,
        parent_one: Session,
        parent_two: Session,
        new_name: str,
        persist: bool = True,
    ) -> tuple[Session, dict[str, Any]]:
        """Create a child session by merging compacted memory from two parent sessions."""
        if parent_one.id == parent_two.id:
            raise ValueError("Choose two different parent sessions for /session procreate")

        name = new_name.strip()
        if not name:
            raise ValueError("Usage: /session procreate <id|name|#index> <id|name|#index> <new-name>")

        self._emit_tool_output(
            "session_procreate",
            {
                "step": "start",
                "parent_one_id": parent_one.id,
                "parent_two_id": parent_two.id,
                "new_name": name,
            },
            (
                "step=start\n"
                f'parent_one="{parent_one.name}" ({parent_one.id})\n'
                f'parent_two="{parent_two.name}" ({parent_two.id})\n'
                f'child_name="{name}"'
            ),
        )
        self._emit_tool_output(
            "session_procreate",
            {"step": "compact_parent_one", "session_id": parent_one.id, "session_name": parent_one.name},
            f'step=compact_parent_one\nworking_on="{parent_one.name}" ({parent_one.id})',
        )
        compacted_one, stats_one = await self._compact_messages_snapshot(
            parent_one.messages,
            trigger="procreate_parent_one",
        )
        self._emit_tool_output(
            "session_procreate",
            {"step": "compact_parent_one_done", "session_id": parent_one.id},
            (
                "step=compact_parent_one_done\n"
                f"compacted_messages={int(stats_one.get('compacted_messages', 0))}\n"
                f"kept_messages={int(stats_one.get('kept_messages', 0))}"
            ),
        )
        self._emit_tool_output(
            "session_procreate",
            {"step": "compact_parent_two", "session_id": parent_two.id, "session_name": parent_two.name},
            f'step=compact_parent_two\nworking_on="{parent_two.name}" ({parent_two.id})',
        )
        compacted_two, stats_two = await self._compact_messages_snapshot(
            parent_two.messages,
            trigger="procreate_parent_two",
        )
        self._emit_tool_output(
            "session_procreate",
            {"step": "compact_parent_two_done", "session_id": parent_two.id},
            (
                "step=compact_parent_two_done\n"
                f"compacted_messages={int(stats_two.get('compacted_messages', 0))}\n"
                f"kept_messages={int(stats_two.get('kept_messages', 0))}"
            ),
        )

        self._emit_tool_output(
            "session_procreate",
            {"step": "merge_memory"},
            "step=merge_memory\nstatus=combining_compacted_parent_memories",
        )
        merged_messages = [*compacted_one, *compacted_two]
        now_iso = datetime.now(UTC).isoformat()
        metadata = {
            "procreate": {
                "created_at": now_iso,
                "parent_one": {"id": parent_one.id, "name": parent_one.name, "stats": stats_one},
                "parent_two": {"id": parent_two.id, "name": parent_two.name, "stats": stats_two},
                "merged_messages": len(merged_messages),
            }
        }

        self._emit_tool_output(
            "session_procreate",
            {"step": "create_child_session", "new_name": name},
            f'step=create_child_session\nchild_name="{name}"',
        )
        child = await self.session_manager.create_session(name=name, metadata=metadata)
        child.messages = merged_messages
        child.updated_at = now_iso
        if persist:
            self._emit_tool_output(
                "session_procreate",
                {"step": "save_child_session", "session_id": child.id},
                f'step=save_child_session\nsession_id="{child.id}"',
            )
            await self.session_manager.save_session(child)

        self._emit_tool_output(
            "session_procreate",
            {"step": "done", "session_id": child.id, "merged_messages": len(merged_messages)},
            (
                "step=done\n"
                f'session_id="{child.id}"\n'
                f"merged_messages={len(merged_messages)}"
            ),
        )
        return child, {
            "parent_one_compacted": int(stats_one.get("compacted_messages", 0)),
            "parent_two_compacted": int(stats_two.get("compacted_messages", 0)),
            "merged_messages": len(merged_messages),
        }

    async def _auto_compact_if_needed(self) -> None:
        """Compact session automatically when context usage exceeds threshold."""
        compacted, stats = await self.compact_session(force=False, trigger="auto")
        if compacted:
            log.info(
                "Auto compaction completed",
                before_tokens=stats.get("before_tokens"),
                after_tokens=stats.get("after_tokens"),
                compacted_messages=stats.get("compacted_messages"),
            )

    def _sync_runtime_flags_from_session(self) -> None:
        """Load runtime feature flags from active session metadata."""
        cfg = get_config()
        pipeline_mode = "loop"
        monitor_trace_llm = bool(getattr(cfg.ui, "monitor_trace_llm", False))
        monitor_trace_pipeline = bool(getattr(cfg.ui, "monitor_trace_pipeline", True))
        if self.session and isinstance(self.session.metadata, dict):
            planning_meta = self.session.metadata.get("planning")
            if isinstance(planning_meta, dict):
                raw_mode = str(planning_meta.get("mode", "")).strip().lower()
                if raw_mode in {"loop", "contracts"}:
                    pipeline_mode = raw_mode
                elif bool(planning_meta.get("enabled", False)):
                    # Backward compatibility with older session metadata.
                    pipeline_mode = "contracts"
            monitor_meta = self.session.metadata.get("monitor")
            if isinstance(monitor_meta, dict) and "trace_llm" in monitor_meta:
                monitor_trace_llm = bool(monitor_meta.get("trace_llm", False))
            if isinstance(monitor_meta, dict) and "trace_pipeline" in monitor_meta:
                monitor_trace_pipeline = bool(monitor_meta.get("trace_pipeline", True))
        self.pipeline_mode = pipeline_mode
        self.planning_enabled = self.pipeline_mode == "contracts"
        self.monitor_trace_llm = monitor_trace_llm
        self.monitor_trace_pipeline = monitor_trace_pipeline
        selection = self._session_model_selection()
        if selection:
            model_id = str(selection.get("id", "")).strip()
            try:
                self._apply_model_option(selection, source="session", model_id=model_id)
                return
            except Exception as e:
                log.warning(
                    "Failed to apply session model selection, using default config",
                    error=str(e),
                )
        self._apply_default_config_model_if_needed()

    def refresh_session_runtime_flags(self) -> None:
        """Public helper to reload runtime flags after session switch."""
        self._sync_runtime_flags_from_session()

    async def set_pipeline_mode(self, mode: str, persist: bool = True) -> None:
        """Set execution pipeline mode for current session/runtime."""
        normalized = str(mode or "").strip().lower()
        if normalized not in {"loop", "contracts"}:
            raise ValueError("Invalid pipeline mode. Use 'loop' or 'contracts'.")
        self.pipeline_mode = normalized
        self.planning_enabled = self.pipeline_mode == "contracts"
        if not self.session:
            return
        planning_meta = self.session.metadata.setdefault("planning", {})
        planning_meta["mode"] = self.pipeline_mode
        planning_meta["enabled"] = self.planning_enabled  # backward-compat mirror
        planning_meta["updated_at"] = datetime.now(UTC).isoformat()
        if persist:
            await self.session_manager.save_session(self.session)

    async def set_planning_mode(self, enabled: bool, persist: bool = True) -> None:
        """Backward-compatible alias for setting pipeline mode."""
        await self.set_pipeline_mode("contracts" if bool(enabled) else "loop", persist=persist)

    async def set_monitor_trace_llm(self, enabled: bool, persist: bool = True) -> None:
        """Enable or disable full intermediate LLM tracing in monitor history."""
        self.monitor_trace_llm = bool(enabled)
        if not self.session:
            return
        monitor_meta = self.session.metadata.setdefault("monitor", {})
        monitor_meta["trace_llm"] = self.monitor_trace_llm
        monitor_meta["updated_at"] = datetime.now(UTC).isoformat()
        if persist:
            await self.session_manager.save_session(self.session)

    async def set_monitor_trace_pipeline(self, enabled: bool, persist: bool = True) -> None:
        """Enable or disable compact pipeline trace logging in session history."""
        self.monitor_trace_pipeline = bool(enabled)
        if not self.session:
            return
        monitor_meta = self.session.metadata.setdefault("monitor", {})
        monitor_meta["trace_pipeline"] = self.monitor_trace_pipeline
        monitor_meta["updated_at"] = datetime.now(UTC).isoformat()
        if persist:
            await self.session_manager.save_session(self.session)

    @staticmethod
    def _iter_pipeline_nodes(tasks: list[dict[str, Any]]) -> Any:
        """Yield all task nodes in depth-first order."""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            yield task
            children = task.get("children")
            if isinstance(children, list) and children:
                yield from Agent._iter_pipeline_nodes(children)

    @staticmethod
    def _iter_pipeline_leaves(tasks: list[dict[str, Any]]) -> Any:
        """Yield leaf task nodes in depth-first order."""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            children = task.get("children")
            if isinstance(children, list) and children:
                yield from Agent._iter_pipeline_leaves(children)
                continue
            yield task

    @staticmethod
    def _set_all_pipeline_status(tasks: list[dict[str, Any]], status: str) -> None:
        """Set status recursively for all nodes in task tree."""
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task["status"] = status
            children = task.get("children")
            if isinstance(children, list) and children:
                Agent._set_all_pipeline_status(children, status)

    @staticmethod
    def _rollup_pipeline_status(tasks: list[dict[str, Any]]) -> None:
        """Roll up parent statuses from children."""
        def _node_status(node: dict[str, Any]) -> str:
            children = node.get("children")
            if not isinstance(children, list) or not children:
                return str(node.get("status", "pending")).strip().lower() or "pending"

            child_statuses = [_node_status(child) for child in children if isinstance(child, dict)]
            if not child_statuses:
                node["status"] = "pending"
                return "pending"
            if any(status == "failed" for status in child_statuses):
                node["status"] = "failed"
                return "failed"
            if any(status == "in_progress" for status in child_statuses):
                node["status"] = "in_progress"
                return "in_progress"
            if all(status == "completed" for status in child_statuses):
                node["status"] = "completed"
                return "completed"
            if any(status == "completed" for status in child_statuses):
                node["status"] = "in_progress"
                return "in_progress"
            node["status"] = "pending"
            return "pending"

        for task in tasks:
            if isinstance(task, dict):
                _node_status(task)

    @staticmethod
    def _refresh_pipeline_task_order(pipeline: dict[str, Any]) -> list[str]:
        """Refresh leaf-task execution order for pipeline state."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            pipeline["task_order"] = []
            pipeline["current_task_id"] = ""
            pipeline["current_index"] = 0
            return []

        order: list[str] = []
        for leaf in Agent._iter_pipeline_leaves(tasks):
            leaf_id = str(leaf.get("id", "")).strip()
            if not leaf_id:
                continue
            order.append(leaf_id)

        pipeline["task_order"] = order
        if not order:
            pipeline["current_task_id"] = ""
            pipeline["current_index"] = 0
            return []
        current_index = int(pipeline.get("current_index", 0))
        bounded = max(0, min(current_index, len(order) - 1))
        pipeline["current_index"] = bounded
        pipeline["current_task_id"] = order[bounded]
        return order

    @staticmethod
    def _find_pipeline_leaf_path(
        nodes: list[dict[str, Any]],
        target_leaf_id: str,
        prefix: list[int] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Locate path metadata from root to a specific leaf task id."""
        base_prefix = list(prefix or [])
        for idx, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            path_prefix = [*base_prefix, idx]
            path_label = ".".join(str(part) for part in path_prefix)
            entry = {
                "node": node,
                "scope_nodes": nodes,
                "index": idx,
                "siblings_total": len(nodes),
                "path": path_label,
            }
            children = node.get("children")
            if isinstance(children, list) and children:
                child_path = Agent._find_pipeline_leaf_path(children, target_leaf_id, path_prefix)
                if child_path:
                    return [entry, *child_path]
                continue

            node_id = str(node.get("id", "")).strip()
            if node_id and node_id == target_leaf_id:
                return [entry]
        return None

    @staticmethod
    def _format_eta_seconds(value: float | int | None) -> str:
        """Format ETA seconds into concise human-readable text."""
        if value is None:
            return "unknown"
        seconds = max(0, int(round(float(value))))
        if seconds < 60:
            return f"{seconds}s"
        minutes, rem_seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {rem_seconds}s"
        hours, rem_minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {rem_minutes}m"
        days, rem_hours = divmod(hours, 24)
        return f"{days}d {rem_hours}h"

    @staticmethod
    def _build_pipeline_progress_details(pipeline: dict[str, Any]) -> dict[str, Any]:
        """Build progress details for monitor visibility across nested scopes."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return {
                "leaf_index": 0,
                "leaf_total": 0,
                "leaf_remaining": 0,
                "current_path": "",
                "scope_progress": [],
                "eta_seconds": None,
                "eta_text": "unknown",
                "elapsed_seconds": 0,
                "completed_leaves": 0,
            }

        task_order = pipeline.get("task_order")
        if not isinstance(task_order, list):
            task_order = Agent._refresh_pipeline_task_order(pipeline)
        total_leaves = len(task_order)
        if total_leaves <= 0:
            return {
                "leaf_index": 0,
                "leaf_total": 0,
                "leaf_remaining": 0,
                "current_path": "",
                "scope_progress": [],
                "eta_seconds": None,
                "eta_text": "unknown",
                "elapsed_seconds": 0,
                "completed_leaves": 0,
            }

        current_index = int(pipeline.get("current_index", 0))
        bounded = max(0, min(current_index, total_leaves - 1))
        current_task_id = str(pipeline.get("current_task_id", "")).strip() or str(task_order[bounded])
        leaf_index = bounded + 1
        leaf_remaining = max(0, total_leaves - leaf_index)
        completed_leaves = sum(
            1
            for leaf in Agent._iter_pipeline_leaves(tasks)
            if str(leaf.get("status", "")).strip().lower() == "completed"
        )
        created_at_raw = str(pipeline.get("created_at", "")).strip()
        elapsed_seconds = 0.0
        if created_at_raw:
            try:
                created_at = datetime.fromisoformat(created_at_raw)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                elapsed_seconds = max(0.0, (datetime.now(UTC) - created_at.astimezone(UTC)).total_seconds())
            except Exception:
                elapsed_seconds = 0.0

        avg_seconds_per_leaf: float | None = None
        eta_seconds: float | None
        if leaf_remaining <= 0:
            eta_seconds = 0.0
        elif completed_leaves > 0 and elapsed_seconds > 0:
            avg_seconds_per_leaf = elapsed_seconds / max(1, completed_leaves)
            eta_seconds = avg_seconds_per_leaf * leaf_remaining
        else:
            eta_seconds = None

        path_entries = Agent._find_pipeline_leaf_path(tasks, current_task_id) or []
        scopes: list[dict[str, Any]] = []
        for level, entry in enumerate(path_entries, start=1):
            scope_nodes = entry.get("scope_nodes")
            if not isinstance(scope_nodes, list):
                continue
            scope_leaf_ids: list[str] = []
            for leaf in Agent._iter_pipeline_leaves(scope_nodes):
                leaf_id = str(leaf.get("id", "")).strip()
                if leaf_id:
                    scope_leaf_ids.append(leaf_id)
            scope_leaf_total = len(scope_leaf_ids)
            try:
                local_pos = scope_leaf_ids.index(current_task_id)
            except ValueError:
                local_pos = -1
            scope_leaf_remaining = (
                max(0, scope_leaf_total - (local_pos + 1))
                if local_pos >= 0
                else scope_leaf_total
            )

            siblings_total = int(entry.get("siblings_total", 0))
            sibling_index = int(entry.get("index", 0))
            scope_eta_seconds: float | None = None
            if scope_leaf_remaining <= 0:
                scope_eta_seconds = 0.0
            elif avg_seconds_per_leaf is not None:
                scope_eta_seconds = avg_seconds_per_leaf * scope_leaf_remaining
            scopes.append({
                "level": level,
                "path": str(entry.get("path", "")),
                "title": str(entry.get("node", {}).get("title", "")).strip(),
                "index": sibling_index,
                "siblings_total": siblings_total,
                "siblings_remaining": max(0, siblings_total - sibling_index),
                "scope_leaf_total": scope_leaf_total,
                "scope_leaf_remaining": scope_leaf_remaining,
                "eta_seconds": scope_eta_seconds,
                "eta_text": Agent._format_eta_seconds(scope_eta_seconds),
            })

        current_path = str(path_entries[-1].get("path", "")) if path_entries else ""
        return {
            "leaf_index": leaf_index,
            "leaf_total": total_leaves,
            "leaf_remaining": leaf_remaining,
            "current_path": current_path,
            "scope_progress": scopes,
            "eta_seconds": eta_seconds,
            "eta_text": Agent._format_eta_seconds(eta_seconds),
            "elapsed_seconds": int(round(elapsed_seconds)),
            "completed_leaves": completed_leaves,
        }

    def _build_task_pipeline(
        self,
        user_input: str,
        max_tasks: int = 6,
        tasks_override: list[dict[str, Any]] | None = None,
        completion_checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a nested task pipeline from user input or planner contract."""
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        tasks: list[dict[str, Any]] = []
        if tasks_override:
            tasks = self._normalize_contract_tasks(
                tasks_override,
                max_tasks=max_tasks,
                max_depth=4,
                max_total_nodes=max(24, max_tasks * 6),
            )
        else:
            parts = [
                piece.strip(" -")
                for piece in re.split(
                    r"(?:\n+|;|\. |\band then\b|\bthen\b|\bnext\b)",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                if piece.strip(" -")
            ]
            fallback_raw = [{"title": part} for part in parts[:max_tasks]]
            tasks = self._normalize_contract_tasks(
                fallback_raw,
                max_tasks=max_tasks,
                max_depth=1,
                max_total_nodes=max_tasks,
            )

        if not tasks:
            tasks = [
                {"id": "task_1", "title": "Understand the request and constraints"},
                {"id": "task_2", "title": "Execute required actions/tools"},
                {"id": "task_3", "title": "Return concise final answer"},
            ]
        for node in self._iter_pipeline_nodes(tasks):
            node["status"] = "pending"

        if completion_checks:
            next_id = sum(1 for _ in self._iter_pipeline_nodes(tasks)) + 1
            tasks.append({
                "id": f"task_{next_id}",
                "title": "Run completion checks before finalizing the answer",
                "status": "pending",
            })

        normalized_checks: list[dict[str, Any]] = []
        for idx, check in enumerate(completion_checks or [], start=1):
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("id", f"check_{idx}")).strip() or f"check_{idx}"
            title = str(check.get("title", check_id)).strip() or check_id
            normalized_checks.append({
                "id": check_id,
                "title": title,
                "status": "pending",
                "detail": "",
            })

        pipeline = {
            "created_at": datetime.now(UTC).isoformat(),
            "request": cleaned[:500],
            "tasks": tasks,
            "checks": normalized_checks,
            "task_order": [],
            "current_index": 0,
            "current_task_id": "",
            "state": "active",
        }
        self._refresh_pipeline_task_order(pipeline)
        self._set_pipeline_progress(pipeline, current_index=0, current_status="in_progress")
        return pipeline

    @staticmethod
    def _set_pipeline_progress(
        pipeline: dict[str, Any],
        current_index: int,
        current_status: str = "in_progress",
    ) -> None:
        """Set pipeline progress using leaf-task execution order."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return
        task_order = Agent._refresh_pipeline_task_order(pipeline)
        if not task_order:
            return
        bounded = max(0, min(current_index, len(task_order) - 1))
        pipeline["current_index"] = bounded
        pipeline["current_task_id"] = task_order[bounded]

        leaf_status_map: dict[str, str] = {}
        for idx, task_id in enumerate(task_order):
            if idx < bounded:
                leaf_status_map[task_id] = "completed"
            elif idx == bounded:
                leaf_status_map[task_id] = current_status
            else:
                leaf_status_map[task_id] = "pending"

        for leaf in Agent._iter_pipeline_leaves(tasks):
            leaf_id = str(leaf.get("id", "")).strip()
            if leaf_id in leaf_status_map:
                leaf["status"] = leaf_status_map[leaf_id]
        Agent._rollup_pipeline_status(tasks)

    def _build_pipeline_note(self, pipeline: dict[str, Any]) -> str:
        """Build planning note injected into model context."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return ""

        def _render(nodes: list[dict[str, Any]], prefix: list[int]) -> list[str]:
            rendered: list[str] = []
            for idx, node in enumerate(nodes, start=1):
                if not isinstance(node, dict):
                    continue
                current_prefix = [*prefix, idx]
                label = ".".join(str(part) for part in current_prefix)
                title = str(node.get("title", "")).strip()
                status = str(node.get("status", "pending")).strip().upper()
                rendered.append(f"{label}. [{status}] {title}")
                children = node.get("children")
                if isinstance(children, list) and children:
                    rendered.extend(_render(children, current_prefix))
            return rendered

        lines = [self.instructions.load("planning_pipeline_header.md")]
        lines.extend(_render(tasks, []))
        lines.append(self.instructions.load("planning_pipeline_footer.md"))
        return "\n".join(lines)

    @staticmethod
    def _build_list_task_note(list_task_plan: dict[str, Any]) -> str:
        """Build list-memory note injected into model context."""
        if not isinstance(list_task_plan, dict) or not bool(list_task_plan.get("enabled", False)):
            return ""
        members = list_task_plan.get("members")
        if not isinstance(members, list) or not members:
            return ""
        strategy = str(list_task_plan.get("strategy", "direct")).strip().lower() or "direct"
        action = str(list_task_plan.get("per_member_action", "")).strip()
        lines = [
            "List task memory is active. You must process every extracted member before final response.",
            f"Strategy: {strategy}",
        ]
        if action:
            lines.append(f"Per-member action: {action}")
        lines.append("Members:")
        for idx, member in enumerate(members[:60], start=1):
            lines.append(f"{idx}. {member}")
        if len(members) > 60:
            lines.append(f"... (+{len(members) - 60} more members)")
        return "\n".join(lines)

    @staticmethod
    def _format_pipeline_monitor_output(event: str, pipeline: dict[str, Any]) -> str:
        """Format pipeline state for monitor output."""
        lines = [f"Planning event={event}"]
        lines.append(f"state={pipeline.get('state', 'active')}")
        progress = Agent._build_pipeline_progress_details(pipeline)
        lines.append(
            "progress="
            f"{progress.get('leaf_index', 0)}/{progress.get('leaf_total', 0)} "
            f"remaining={progress.get('leaf_remaining', 0)}"
        )
        lines.append(
            f"eta={progress.get('eta_text', 'unknown')} "
            f"(elapsed={progress.get('elapsed_seconds', 0)}s completed={progress.get('completed_leaves', 0)})"
        )
        current_path = str(progress.get("current_path", "")).strip()
        if current_path:
            lines.append(f"current_path={current_path}")
        scope_progress = progress.get("scope_progress", [])
        if isinstance(scope_progress, list) and scope_progress:
            lines.append("scope_progress:")
            for scope in scope_progress:
                if not isinstance(scope, dict):
                    continue
                lines.append(
                    "- "
                    f"level={scope.get('level', '')} "
                    f"path={scope.get('path', '')} "
                    f"index={scope.get('index', 0)}/{scope.get('siblings_total', 0)} "
                    f"siblings_left={scope.get('siblings_remaining', 0)} "
                    f"leaves_left={scope.get('scope_leaf_remaining', 0)}/{scope.get('scope_leaf_total', 0)} "
                    f"eta={scope.get('eta_text', 'unknown')} "
                    f"title={scope.get('title', '')}"
                )

        def _render(nodes: list[dict[str, Any]], prefix: list[int]) -> list[str]:
            rendered: list[str] = []
            for idx, node in enumerate(nodes, start=1):
                if not isinstance(node, dict):
                    continue
                current_prefix = [*prefix, idx]
                label = ".".join(str(part) for part in current_prefix)
                status = str(node.get("status", "pending")).strip()
                title = str(node.get("title", "")).strip()
                rendered.append(f"- {label}. status={status} title={title}")
                children = node.get("children")
                if isinstance(children, list) and children:
                    rendered.extend(_render(children, current_prefix))
            return rendered

        tasks = pipeline.get("tasks", [])
        if isinstance(tasks, list) and tasks:
            lines.extend(_render(tasks, []))
        checks = pipeline.get("checks", [])
        if isinstance(checks, list) and checks:
            lines.append("checks:")
            for idx, check in enumerate(checks, start=1):
                if not isinstance(check, dict):
                    continue
                detail = str(check.get("detail", "")).strip()
                detail_suffix = f" detail={detail}" if detail else ""
                lines.append(
                    f"- {idx}. status={check.get('status', 'pending')} id={check.get('id', '')} "
                    f"title={check.get('title', '')}{detail_suffix}"
                )
        return "\n".join(lines)

    def _emit_pipeline_update(self, event: str, pipeline: dict[str, Any]) -> None:
        """Emit planning pipeline state into monitor output stream."""
        task_order = self._refresh_pipeline_task_order(pipeline)
        progress = self._build_pipeline_progress_details(pipeline)
        self._emit_tool_output(
            "planning",
            {
                "event": event,
                "enabled": self.planning_enabled,
                "mode": str(pipeline.get("mode", "manual")),
                "current_index": int(pipeline.get("current_index", 0)),
                "current_task_id": str(pipeline.get("current_task_id", "")),
                "leaf_tasks": len(task_order),
                "leaf_index": int(progress.get("leaf_index", 0)),
                "leaf_remaining": int(progress.get("leaf_remaining", 0)),
                "current_path": str(progress.get("current_path", "")),
                "eta_seconds": progress.get("eta_seconds"),
                "eta_text": str(progress.get("eta_text", "unknown")),
                "scope_progress": progress.get("scope_progress", []),
            },
            self._format_pipeline_monitor_output(event, pipeline),
        )

    def _advance_pipeline(self, pipeline: dict[str, Any], event: str = "advance") -> None:
        """Advance pipeline to next leaf task if possible."""
        task_order = self._refresh_pipeline_task_order(pipeline)
        if not task_order:
            return
        current_index = int(pipeline.get("current_index", 0))
        bounded = max(0, min(current_index, len(task_order) - 1))
        if bounded >= len(task_order) - 1:
            self._set_pipeline_progress(pipeline, current_index=bounded, current_status="in_progress")
            self._emit_pipeline_update(event, pipeline)
            return
        self._set_pipeline_progress(pipeline, current_index=bounded + 1, current_status="in_progress")
        self._emit_pipeline_update(event, pipeline)

    def _finalize_pipeline(self, pipeline: dict[str, Any], success: bool = True) -> None:
        """Finalize pipeline statuses at turn completion."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list):
            return
        task_order = self._refresh_pipeline_task_order(pipeline)
        if success:
            self._set_all_pipeline_status(tasks, "completed")
            Agent._rollup_pipeline_status(tasks)
            if task_order:
                pipeline["current_index"] = len(task_order) - 1
                pipeline["current_task_id"] = task_order[-1]
            pipeline["state"] = "completed"
            self._emit_pipeline_update("completed", pipeline)
            return

        if task_order:
            current_index = int(pipeline.get("current_index", 0))
            bounded = max(0, min(current_index, len(task_order) - 1))
            self._set_pipeline_progress(pipeline, current_index=bounded, current_status="failed")
        else:
            self._set_all_pipeline_status(tasks, "failed")
            Agent._rollup_pipeline_status(tasks)
        pipeline["state"] = "failed"
        self._emit_pipeline_update("failed", pipeline)

    @staticmethod
    def _update_pipeline_checks(
        pipeline: dict[str, Any],
        check_results: list[dict[str, Any]],
    ) -> None:
        """Apply completion-check statuses onto active pipeline."""
        checks = pipeline.get("checks", [])
        if not isinstance(checks, list) or not checks:
            return
        result_map: dict[str, dict[str, Any]] = {}
        for result in check_results:
            if not isinstance(result, dict):
                continue
            key = str(result.get("id", "")).strip()
            if not key:
                continue
            result_map[key] = result

        for check in checks:
            if not isinstance(check, dict):
                continue
            key = str(check.get("id", "")).strip()
            data = result_map.get(key)
            if not data:
                continue
            if bool(data.get("ok", False)):
                check["status"] = "passed"
                check["detail"] = ""
            else:
                check["status"] = "failed"
                check["detail"] = str(data.get("reason", "")).strip()

    def _compute_turn_iteration_budget(
        self,
        base_iterations: int,
        planning_pipeline: dict[str, Any] | None,
        completion_requirements: list[dict[str, Any]] | None,
    ) -> int:
        """Compute adaptive iteration budget from task complexity."""
        budget = max(1, int(base_iterations))
        leaf_tasks = 0
        if isinstance(planning_pipeline, dict):
            task_order = planning_pipeline.get("task_order")
            if isinstance(task_order, list):
                leaf_tasks = len(task_order)
            else:
                tasks = planning_pipeline.get("tasks", [])
                if isinstance(tasks, list):
                    leaf_tasks = sum(1 for _ in self._iter_pipeline_leaves(tasks))
        if leaf_tasks > 0:
            budget = max(budget, min(120, 4 + (leaf_tasks * 3)))

        req_count = len(completion_requirements or [])
        if req_count > 0:
            budget = min(140, budget + min(20, req_count))
        return max(budget, max(1, int(base_iterations)))

    def _capture_turn_progress_snapshot(
        self,
        turn_start_idx: int,
        planning_pipeline: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Capture compact progress snapshot for stuck/progress detection."""
        successful_tool_count = 0
        write_success_count = 0
        tool_signatures: set[str] = set()
        assistant_signatures: set[str] = set()
        if self.session:
            for msg in self.session.messages[turn_start_idx:]:
                role = str(msg.get("role", "")).strip().lower()
                if role == "assistant":
                    content = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
                    if content:
                        assistant_signatures.add(content[:800])
                    continue
                if str(msg.get("role", "")).strip().lower() != "tool":
                    continue
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                if self._is_monitor_only_tool_name(tool_name):
                    continue
                content = str(msg.get("content", ""))
                if not content.strip().lower().startswith("error:"):
                    successful_tool_count += 1
                    if tool_name == "write":
                        write_success_count += 1
                args = msg.get("tool_arguments")
                try:
                    args_text = json.dumps(args, sort_keys=True, ensure_ascii=True) if isinstance(args, dict) else "{}"
                except Exception:
                    args_text = "{}"
                tool_signatures.add(f"{tool_name}|{args_text}")

        pipeline_index = -1
        pipeline_task_id = ""
        if isinstance(planning_pipeline, dict):
            pipeline_index = int(planning_pipeline.get("current_index", -1))
            pipeline_task_id = str(planning_pipeline.get("current_task_id", "")).strip()

        return {
            "successful_tools": successful_tool_count,
            "write_success": write_success_count,
            "unique_tool_signatures": len(tool_signatures),
            "unique_assistant_signatures": len(assistant_signatures),
            "pipeline_index": pipeline_index,
            "pipeline_task_id": pipeline_task_id,
        }

    @staticmethod
    def _has_turn_progress(previous: dict[str, Any], current: dict[str, Any]) -> bool:
        """Whether turn made meaningful progress between snapshots."""
        if int(current.get("write_success", 0)) > int(previous.get("write_success", 0)):
            return True
        if int(current.get("unique_tool_signatures", 0)) > int(previous.get("unique_tool_signatures", 0)):
            return True
        if int(current.get("unique_assistant_signatures", 0)) > int(previous.get("unique_assistant_signatures", 0)):
            return True
        prev_task = str(previous.get("pipeline_task_id", "")).strip()
        curr_task = str(current.get("pipeline_task_id", "")).strip()
        if curr_task and curr_task != prev_task:
            return True
        if int(current.get("pipeline_index", -1)) > int(previous.get("pipeline_index", -1)):
            return True
        return False

    def _pipeline_has_remaining_work(self, planning_pipeline: dict[str, Any] | None) -> bool:
        """Whether pipeline still has unfinished tasks/checks."""
        if not isinstance(planning_pipeline, dict):
            return False
        tasks = planning_pipeline.get("tasks", [])
        if isinstance(tasks, list):
            for leaf in self._iter_pipeline_leaves(tasks):
                status = str(leaf.get("status", "pending")).strip().lower()
                if status not in {"completed"}:
                    return True
        checks = planning_pipeline.get("checks", [])
        if isinstance(checks, list):
            for check in checks:
                if not isinstance(check, dict):
                    continue
                status = str(check.get("status", "pending")).strip().lower()
                if status not in {"passed"}:
                    return True
        return False

    def _add_session_message(
        self,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_arguments: dict[str, Any] | None = None,
    ) -> None:
        """Append message to session with per-message token metadata."""
        if not self.session:
            return
        self.session.add_message(
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_calls=tool_calls,
            tool_arguments=tool_arguments,
            token_count=self._count_tokens(content),
        )

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        """Extract unique URLs from text in appearance order."""
        raw = re.findall(r"https?://[^\s)\]}>\"']+", text or "")
        seen: set[str] = set()
        urls: list[str] = []
        for url in raw:
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    @staticmethod
    def _merge_unique_urls(*url_lists: list[str]) -> list[str]:
        """Merge URL lists while preserving first-seen order."""
        seen: set[str] = set()
        merged: list[str] = []
        for url_list in url_lists:
            for url in url_list:
                if url in seen:
                    continue
                seen.add(url)
                merged.append(url)
        return merged

    def _extract_source_links(self, msg: dict[str, Any], content: str) -> list[str]:
        """Extract source links from both tool content and structured tool arguments."""
        content_links = self._extract_urls(content)
        args_links: list[str] = []
        args = msg.get("tool_arguments")
        if isinstance(args, dict):
            for key in ("url", "href", "link", "source_url"):
                value = args.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    args_links.append(value)
            for key in ("urls", "links"):
                values = args.get(key)
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and value.startswith(("http://", "https://")):
                            args_links.append(value)
        return self._merge_unique_urls(args_links, content_links)

    def _collect_recent_source_urls(
        self,
        turn_start_idx: int,
        max_messages: int = 20,
        max_urls: int = 20,
    ) -> list[str]:
        """Collect recent source URLs from messages before current turn."""
        if not self.session:
            return []
        start = max(0, turn_start_idx - max_messages)
        urls: list[str] = []
        for msg in self.session.messages[start:turn_start_idx]:
            content = str(msg.get("content", ""))
            links = self._extract_source_links(msg, content)
            if links:
                urls = self._merge_unique_urls(urls, links)
            if len(urls) >= max_urls:
                return urls[:max_urls]
        return urls[:max_urls]

    @staticmethod
    def _request_references_all_sources(user_input: str) -> bool:
        """Detect intent to cover all referenced links/sources."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        patterns = (
            r"\bcheck all (?:those )?(?:sources|links)\b",
            r"\ball (?:those )?(?:sources|links)\b",
            r"\beach source\b",
            r"\bevery source\b",
            r"\bper source\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _assistant_requests_clarification(response_text: str) -> bool:
        """Heuristic: whether assistant is asking user to clarify before execution."""
        text = re.sub(r"\s+", " ", (response_text or "").strip())
        if not text:
            return False
        lowered = text.lower()
        if lowered.count("?") >= 2:
            return True
        prompts = (
            "which would you like",
            "do you want me to",
            "would you like me to",
            "tell me your choices",
            "quick questions",
            "so i proceed correctly",
            "confirm",
        )
        if any(phrase in lowered for phrase in prompts) and "?" in lowered:
            return True
        return False

    @staticmethod
    def _should_apply_pending_clarification(user_input: str) -> bool:
        """Whether current user text looks like a clarification answer."""
        text = (user_input or "").strip()
        if not text:
            return False
        if text.startswith("/"):
            return False
        if text.count("\n") > 4:
            return False
        words = re.findall(r"\S+", text)
        if not words:
            return False
        if len(words) > 40:
            return False
        if text.endswith("?"):
            return False
        return True

    def _resolve_effective_user_input(self, user_input: str) -> tuple[str, bool]:
        """Merge pending clarification anchor with current message when appropriate."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return user_input, False
        state = self.session.metadata.get("clarification_state")
        if not isinstance(state, dict) or not bool(state.get("pending", False)):
            return user_input, False
        anchor = str(state.get("anchor_request", "")).strip()
        if not anchor:
            return user_input, False
        if not self._should_apply_pending_clarification(user_input):
            return user_input, False
        merged = (
            f"{anchor}\n\n"
            "Clarifications/preferences from user:\n"
            f"{user_input.strip()}\n\n"
            "Execute the full original request using these clarification details."
        )
        return merged, True

    def _update_clarification_state(
        self,
        user_input: str,
        effective_user_input: str,
        assistant_response: str,
    ) -> None:
        """Track unresolved clarification context across turns."""
        if not self.session:
            return
        meta = self.session.metadata.setdefault("clarification_state", {})
        if not isinstance(meta, dict):
            self.session.metadata["clarification_state"] = {}
            meta = self.session.metadata["clarification_state"]
        now_iso = datetime.now(UTC).isoformat()
        if self._assistant_requests_clarification(assistant_response):
            meta["pending"] = True
            meta["anchor_request"] = str(effective_user_input or user_input).strip()[:12000]
            meta["updated_at"] = now_iso
            return
        if bool(meta.get("pending", False)):
            meta["pending"] = False
            meta.pop("anchor_request", None)
            meta["updated_at"] = now_iso

    @staticmethod
    def _should_run_source_report_pipeline(user_input: str, source_urls: list[str]) -> bool:
        """Whether request asks for all/each sources to be checked and reported."""
        if not source_urls:
            return False
        text = (user_input or "").strip().lower()
        if not text:
            return False
        trigger = any(
            re.search(pattern, text)
            for pattern in (
                r"\bcheck all (?:those )?sources\b",
                r"\ball sources\b",
                r"\beach source\b",
                r"\bper source\b",
                r"\bsource[- ]distinguish(?:ed)?\b",
            )
        )
        report_intent = any(word in text for word in ("report", "summar", "compile"))
        return trigger or (report_intent and "source" in text)

    async def _run_source_report_prefetch(
        self,
        source_urls: list[str],
        turn_usage: dict[str, int],
        max_chars_per_source: int = 4500,
        pipeline_label: str = "source_report_pipeline",
    ) -> dict[str, Any]:
        """Prefetch all source URLs via web_fetch for source-report tasks."""
        if not source_urls:
            return {"requested": 0, "fetched": 0, "failed": 0}
        if "web_fetch" not in self.tools.list_tools():
            return {"requested": len(source_urls), "fetched": 0, "failed": len(source_urls), "reason": "web_fetch_disabled"}

        fetched = 0
        failed = 0
        total = len(source_urls)
        self._emit_tool_output(
            pipeline_label,
            {"step": "prefetch_start", "sources": total},
            f"step=prefetch_start\nsources={total}",
        )
        for idx, url in enumerate(source_urls, start=1):
            args = {
                "url": url,
                "extract_mode": "text",
                "max_chars": max_chars_per_source,
            }
            try:
                result = await self._execute_tool_with_guard(
                    name="web_fetch",
                    arguments=args,
                    interaction_label=f"source_report_prefetch_{idx}",
                    turn_usage=turn_usage,
                )
                output = result.content if result.success else f"Error: {result.error}"
            except Exception as e:
                result = None
                output = f"Error: {str(e)}"

            tagged_output = f"[SOURCE {idx}/{total}] {url}\n{output}"
            self._add_session_message(
                role="tool",
                content=tagged_output,
                tool_name="web_fetch",
                tool_arguments=args,
            )
            self._emit_tool_output("web_fetch", args, tagged_output)
            if result and result.success:
                fetched += 1
            else:
                failed += 1

        self._emit_tool_output(
            pipeline_label,
            {"step": "prefetch_done", "sources": total, "fetched": fetched, "failed": failed},
            (
                "step=prefetch_done\n"
                f"sources={total}\n"
                f"fetched={fetched}\n"
                f"failed={failed}"
            ),
        )
        return {"requested": total, "fetched": fetched, "failed": failed}

    @staticmethod
    def _count_source_sections(text: str) -> int:
        """Count `Source <n>` headings in a report text."""
        if not text:
            return 0
        return len(re.findall(r"(?im)^\s{0,3}(?:#+\s*)?source\s+\d+\b", text))

    @staticmethod
    def _has_conclusion_section(text: str) -> bool:
        """Detect whether report contains a `Conclusion` section heading."""
        if not text:
            return False
        return bool(re.search(r"(?im)^\s{0,3}(?:#+\s*)?conclusion\b", text))

    def _validate_source_report_response(
        self,
        response_text: str,
        expected_sources: int,
    ) -> tuple[bool, str]:
        """Validate report completeness for source-by-source requests."""
        expected = max(1, int(expected_sources))
        actual_sources = self._count_source_sections(response_text)
        has_conclusion = self._has_conclusion_section(response_text)
        if actual_sources < expected:
            return False, f"source sections missing ({actual_sources}/{expected})"
        if not has_conclusion:
            return False, "missing conclusion section"
        return True, ""

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
        """Extract the first valid JSON object from model text."""
        text = (raw_text or "").strip()
        if not text:
            return None

        candidates: list[str] = [text]
        fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced_matches)
        inline_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if inline_match:
            candidates.append(inline_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    @staticmethod
    def _should_use_contract_pipeline(
        user_input: str,
        planning_enabled: bool,
        pipeline_mode: str | None = None,
    ) -> bool:
        """Use explicit user-selected mode only (no automatic switching)."""
        del user_input  # mode-driven decision for now
        mode = str(pipeline_mode or "").strip().lower()
        if mode == "contracts":
            return True
        if mode == "loop":
            return False
        # Backward-compat fallback for call sites/session metadata not yet migrated.
        return bool(planning_enabled)

    @staticmethod
    def _normalize_contract_tasks(
        raw_tasks: Any,
        max_tasks: int = 8,
        max_depth: int = 4,
        max_total_nodes: int = 36,
    ) -> list[dict[str, Any]]:
        """Normalize planner task items into a nested task tree."""
        if isinstance(raw_tasks, dict):
            source_tasks: list[Any] = [raw_tasks]
        elif isinstance(raw_tasks, list):
            source_tasks = list(raw_tasks)
        else:
            return []

        normalized: list[dict[str, Any]] = []
        next_id = 0

        def _extract_title(item: Any) -> str:
            if isinstance(item, dict):
                for key in ("title", "task", "name", "step", "summary", "description"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                return ""
            return str(item).strip()

        def _extract_children(item: Any) -> list[Any]:
            if not isinstance(item, dict):
                return []
            for key in ("children", "tasks", "subtasks", "steps", "items"):
                value = item.get(key)
                if isinstance(value, list):
                    return value
            return []

        def _visit(item: Any, depth: int) -> dict[str, Any] | None:
            nonlocal next_id
            if next_id >= max_total_nodes:
                return None
            title = _extract_title(item)
            if not title:
                return None
            next_id += 1
            node: dict[str, Any] = {"id": f"task_{next_id}", "title": title[:180]}
            if depth >= max_depth:
                return node
            child_nodes: list[dict[str, Any]] = []
            for child in _extract_children(item):
                if next_id >= max_total_nodes:
                    break
                normalized_child = _visit(child, depth + 1)
                if normalized_child:
                    child_nodes.append(normalized_child)
            if child_nodes:
                node["children"] = child_nodes
            return node

        for item in source_tasks[:max_tasks]:
            if next_id >= max_total_nodes:
                break
            normalized_item = _visit(item, depth=1)
            if normalized_item:
                normalized.append(normalized_item)
        return normalized

    @staticmethod
    def _normalize_contract_requirements(raw_requirements: Any, max_items: int = 10) -> list[dict[str, Any]]:
        """Normalize planner requirements into stable ids + titles."""
        normalized: list[dict[str, Any]] = []
        if isinstance(raw_requirements, list):
            for idx, item in enumerate(raw_requirements[:max_items], start=1):
                if isinstance(item, dict):
                    title = str(item.get("title", "")).strip()
                    req_id = str(item.get("id", "")).strip()
                else:
                    title = str(item).strip()
                    req_id = ""
                if not title:
                    continue
                if not req_id:
                    req_id = f"req_{idx}"
                req_id = re.sub(r"[^a-zA-Z0-9_]+", "_", req_id).strip("_") or f"req_{idx}"
                normalized.append({"id": req_id[:48], "title": title[:220]})
        return normalized

    @staticmethod
    def _default_task_contract(user_input: str) -> dict[str, Any]:
        """Fallback contract when planner output is unavailable."""
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        return {
            "summary": cleaned[:320],
            "tasks": [
                {"id": "task_1", "title": "Understand the request and constraints"},
                {"id": "task_2", "title": "Execute needed tools/actions"},
                {"id": "task_3", "title": "Produce final response aligned with request"},
            ],
            "requirements": [
                {"id": "req_user_request", "title": "Fully satisfy the user request before finalizing"},
            ],
            "prefetch_urls": [],
        }

    async def _generate_task_contract(
        self,
        user_input: str,
        recent_source_urls: list[str],
        require_all_sources: bool,
        turn_usage: dict[str, int],
        list_task_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use planner prompt to generate a task contract for this turn."""
        source_lines = "\n".join(f"{idx}. {url}" for idx, url in enumerate(recent_source_urls, start=1))
        if not source_lines:
            source_lines = "(none)"
        list_member_lines = "(none)"
        if isinstance(list_task_plan, dict) and bool(list_task_plan.get("enabled", False)):
            members = list_task_plan.get("members")
            if isinstance(members, list) and members:
                list_member_lines = "\n".join(f"{idx}. {str(member)}" for idx, member in enumerate(members, start=1))
        base_messages = [
            Message(
                role="system",
                content=self.instructions.load("task_contract_planner_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "task_contract_planner_user_prompt.md",
                    user_input=user_input,
                    recent_source_urls=source_lines,
                    require_all_sources=str(bool(require_all_sources)).lower(),
                    extracted_list_members=list_member_lines,
                ),
            ),
        ]
        cfg_max_tokens = max(1, int(get_config().model.max_tokens))
        first_max_tokens = min(1200, cfg_max_tokens)
        retry_max_tokens = min(max(first_max_tokens * 2, 1800), cfg_max_tokens)
        attempts: list[tuple[str, int]] = [("task_contract_planner", first_max_tokens)]
        if retry_max_tokens > first_max_tokens:
            attempts.append(("task_contract_planner_retry", retry_max_tokens))

        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt_idx, (interaction_label, planner_max_tokens) in enumerate(attempts, start=1):
            try:
                response = await self._complete_with_guards(
                    messages=base_messages,
                    tools=None,
                    interaction_label=interaction_label,
                    turn_usage=turn_usage,
                    max_tokens=planner_max_tokens,
                )
                payload = self._extract_json_object(response.content or "")
                if isinstance(payload, dict):
                    break

                usage = response.usage if isinstance(response.usage, dict) else {}
                completion_tokens = int(usage.get("completion_tokens", 0))
                at_cap = completion_tokens >= planner_max_tokens
                empty_output = not str(response.content or "").strip()
                should_retry = attempt_idx < len(attempts) and (empty_output or at_cap)
                if should_retry:
                    retry_reason = "empty_output" if empty_output else "hit_max_tokens"
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "planner_retry",
                            "attempt": attempt_idx,
                            "reason": retry_reason,
                            "completion_tokens": completion_tokens,
                            "max_tokens": planner_max_tokens,
                        },
                        (
                            "step=planner_retry\n"
                            f"attempt={attempt_idx}\n"
                            f"reason={retry_reason}\n"
                            f"completion_tokens={completion_tokens}\n"
                            f"max_tokens={planner_max_tokens}"
                        ),
                    )
                    continue
                payload = None
                break
            except Exception as e:
                last_error = e
                if attempt_idx < len(attempts):
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "planner_retry",
                            "attempt": attempt_idx,
                            "reason": "planner_error",
                        },
                        (
                            "step=planner_retry\n"
                            f"attempt={attempt_idx}\n"
                            "reason=planner_error\n"
                            f"error={str(e)}"
                        ),
                    )
                    continue
                payload = None
                break

        if payload is None and last_error is not None:
            self._emit_tool_output(
                "task_contract",
                {"step": "planner_error"},
                f"Planner contract generation failed: {str(last_error)}",
            )

        if not isinstance(payload, dict):
            contract = self._default_task_contract(user_input)
            self._emit_tool_output(
                "task_contract",
                {"step": "planner_fallback"},
                "Planner output was not valid JSON. Using fallback contract.",
            )
            return contract

        tasks = self._normalize_contract_tasks(payload.get("tasks"))
        requirements = self._normalize_contract_requirements(payload.get("requirements"))
        summary = str(payload.get("summary", "")).strip()[:320]
        if not tasks:
            tasks = self._default_task_contract(user_input)["tasks"]
        if not requirements:
            requirements = self._default_task_contract(user_input)["requirements"]
        prefetch_urls: list[str] = []
        raw_prefetch = payload.get("prefetch_urls")
        if isinstance(raw_prefetch, list):
            for url in raw_prefetch:
                if not isinstance(url, str):
                    continue
                clean = url.strip()
                if clean.startswith(("http://", "https://")):
                    prefetch_urls.append(clean)
        if require_all_sources and recent_source_urls:
            prefetch_urls = self._merge_unique_urls(recent_source_urls, prefetch_urls)
        else:
            prefetch_urls = self._merge_unique_urls(prefetch_urls, recent_source_urls)
        prefetch_urls = prefetch_urls[:20]

        if require_all_sources and recent_source_urls:
            requirement_urls: set[str] = set()
            for req in requirements:
                if not isinstance(req, dict):
                    continue
                title = str(req.get("title", ""))
                for url in self._extract_urls(title):
                    requirement_urls.add(url)
            missing_urls = [url for url in recent_source_urls if url not in requirement_urls]
            base_count = len(requirements)
            for offset, url in enumerate(missing_urls, start=1):
                req_id = f"req_source_{base_count + offset}"
                requirements.append({
                    "id": req_id,
                    "title": f"Cover source: {url}",
                })

        contract = {
            "summary": summary or self._default_task_contract(user_input)["summary"],
            "tasks": tasks,
            "requirements": requirements,
            "prefetch_urls": prefetch_urls,
        }
        task_nodes = sum(1 for _ in self._iter_pipeline_nodes(tasks))
        task_leaves = sum(1 for _ in self._iter_pipeline_leaves(tasks))
        self._emit_tool_output(
            "task_contract",
            {
                "step": "planner_done",
                "tasks": len(tasks),
                "task_nodes": task_nodes,
                "task_leaves": task_leaves,
                "requirements": len(requirements),
                "prefetch_urls": len(prefetch_urls),
                "require_all_sources": require_all_sources,
                "recent_sources": len(recent_source_urls),
            },
            (
                f"step=planner_done\n"
                f"tasks={len(tasks)}\n"
                f"task_nodes={task_nodes}\n"
                f"task_leaves={task_leaves}\n"
                f"requirements={len(requirements)}\n"
                f"prefetch_urls={len(prefetch_urls)}\n"
                f"require_all_sources={require_all_sources}\n"
                f"recent_sources={len(recent_source_urls)}"
            ),
        )
        return contract

    async def _evaluate_contract_completion(
        self,
        user_input: str,
        candidate_response: str,
        contract: dict[str, Any],
        turn_usage: dict[str, int],
    ) -> dict[str, Any]:
        """Critic pass: evaluate whether candidate satisfies contract requirements."""
        requirements = contract.get("requirements")
        if not isinstance(requirements, list) or not requirements:
            return {"complete": True, "checks": []}

        requirements_json = json.dumps(requirements, ensure_ascii=True)
        messages = [
            Message(
                role="system",
                content=self.instructions.load("task_contract_critic_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "task_contract_critic_user_prompt.md",
                    user_input=user_input,
                    requirements_json=requirements_json,
                    candidate_response=candidate_response,
                ),
            ),
        ]
        try:
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="task_contract_critic",
                turn_usage=turn_usage,
                max_tokens=min(1200, int(get_config().model.max_tokens)),
            )
            payload = self._extract_json_object(response.content or "")
        except Exception as e:
            self._emit_tool_output(
                "completion_gate",
                {"step": "critic_error"},
                f"Contract critic failed: {str(e)}",
            )
            payload = None

        if not isinstance(payload, dict):
            return {
                "complete": True,
                "checks": [],
                "feedback": "",
                "error": "critic_non_json",
            }

        req_ids = {str(req.get("id", "")).strip() for req in requirements if isinstance(req, dict)}
        checks: list[dict[str, Any]] = []
        raw_checks = payload.get("checks")
        if isinstance(raw_checks, list):
            for entry in raw_checks:
                if not isinstance(entry, dict):
                    continue
                check_id = str(entry.get("id", "")).strip()
                if not check_id or check_id not in req_ids:
                    continue
                checks.append({
                    "id": check_id,
                    "ok": bool(entry.get("ok", False)),
                    "reason": str(entry.get("reason", "")).strip(),
                })

        for req_id in req_ids:
            if not any(item.get("id") == req_id for item in checks):
                checks.append({"id": req_id, "ok": False, "reason": "missing critic evaluation"})

        complete = bool(payload.get("complete", False))
        if checks and not all(bool(item.get("ok", False)) for item in checks):
            complete = False
        feedback = str(payload.get("feedback", "")).strip()
        return {
            "complete": complete,
            "checks": checks,
            "feedback": feedback,
        }

    def _build_completion_feedback(
        self,
        contract: dict[str, Any],
        critique: dict[str, Any],
    ) -> str:
        """Build retry feedback based on model critic output."""
        feedback = str(critique.get("feedback", "")).strip()
        checks = critique.get("checks")
        req_map = {
            str(req.get("id", "")).strip(): str(req.get("title", "")).strip()
            for req in (contract.get("requirements") or [])
            if isinstance(req, dict)
        }
        failed = []
        if isinstance(checks, list):
            failed = [entry for entry in checks if isinstance(entry, dict) and not bool(entry.get("ok", False))]
        if feedback and failed:
            lines = [feedback, "Missing requirements to fix:"]
            for item in failed:
                req_id = str(item.get("id", "")).strip()
                title = req_map.get(req_id, req_id) or req_id
                reason = str(item.get("reason", "")).strip() or "not satisfied"
                lines.append(f"- {title}: {reason}")
            lines.append("Return only the corrected final answer.")
            return "\n".join(lines)
        if feedback:
            return feedback + "\nReturn only the corrected final answer."
        if failed:
            lines = [
                "The previous draft is incomplete. Fix all missing requirements before finalizing.",
                "Missing requirements:",
            ]
            for item in failed:
                req_id = str(item.get("id", "")).strip()
                title = req_map.get(req_id, req_id) or req_id
                reason = str(item.get("reason", "")).strip() or "not satisfied"
                lines.append(f"- {title}: {reason}")
            lines.append("Return only the corrected final answer.")
            return "\n".join(lines)
        return "Re-check the task contract and return a complete final answer."

    def _refresh_runtime_model_details(
        self,
        source: str = "config",
        model_id: str = "",
    ) -> None:
        """Refresh current runtime model details from active provider/config."""
        cfg = get_config()
        provider_name = str(getattr(self.provider, "provider", cfg.model.provider or "")).strip()
        model_name = str(getattr(self.provider, "model", cfg.model.model or "")).strip()
        temperature = getattr(self.provider, "temperature", cfg.model.temperature)
        max_tokens = getattr(self.provider, "max_tokens", cfg.model.max_tokens)
        base_url = str(getattr(self.provider, "base_url", cfg.model.base_url or "") or "").strip()
        self._runtime_model_details = {
            "provider": provider_name or str(cfg.model.provider),
            "model": model_name or str(cfg.model.model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": base_url,
            "source": source,
            "id": model_id,
        }

    def get_runtime_model_details(self) -> dict[str, Any]:
        """Return active runtime model details for UI/status display."""
        if not self._runtime_model_details:
            self._refresh_runtime_model_details(source="config")
        return dict(self._runtime_model_details)

    def get_allowed_models(self) -> list[dict[str, Any]]:
        """Return allowed model selections from config.

        Falls back to current config model if explicit allowlist is empty.
        """
        cfg = get_config()
        options: list[dict[str, Any]] = []
        
        def _pick(entry: Any, key: str, default: Any = "") -> Any:
            if isinstance(entry, dict):
                return entry.get(key, default)
            return getattr(entry, key, default)

        for idx, item in enumerate(cfg.model.allowed, start=1):
            model_id = str(_pick(item, "id", "")).strip() or f"model-{idx}"
            provider = str(_pick(item, "provider", "")).strip()
            model = str(_pick(item, "model", "")).strip()
            base_url = str(_pick(item, "base_url", "") or "").strip()
            temperature = _pick(item, "temperature", None)
            max_tokens = _pick(item, "max_tokens", None)
            if not provider or not model:
                continue
            options.append({
                "id": model_id,
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "temperature": temperature,
                "max_tokens": max_tokens,
            })

        if options:
            return options

        return [{
            "id": "default",
            "provider": str(cfg.model.provider),
            "model": str(cfg.model.model),
            "base_url": str(cfg.model.base_url or ""),
            "temperature": cfg.model.temperature,
            "max_tokens": cfg.model.max_tokens,
        }]

    def _resolve_allowed_model(self, selector: str) -> dict[str, Any] | None:
        """Resolve user selector to an allowed model entry."""
        key = (selector or "").strip()
        if not key:
            return None
        options = self.get_allowed_models()

        lowered = key.lower()
        for option in options:
            if str(option.get("id", "")).strip().lower() == lowered:
                return option

        index_text = key[1:] if key.startswith("#") else key
        if index_text.isdigit():
            index = int(index_text)
            if 1 <= index <= len(options):
                return options[index - 1]

        for option in options:
            provider = str(option.get("provider", "")).strip().lower()
            model = str(option.get("model", "")).strip().lower()
            if lowered in {f"{provider}:{model}", f"{provider}/{model}", model}:
                return option
        return None

    def _apply_model_option(
        self,
        option: dict[str, Any],
        source: str,
        model_id: str = "",
    ) -> None:
        """Instantiate provider for chosen model and switch runtime."""
        provider = str(option.get("provider", "")).strip()
        model = str(option.get("model", "")).strip()
        base_url = str(option.get("base_url", "") or "").strip() or None
        temperature = option.get("temperature")
        max_tokens = option.get("max_tokens")
        cfg = get_config()
        resolved_temperature = float(cfg.model.temperature if temperature is None else temperature)
        resolved_max_tokens = int(cfg.model.max_tokens if max_tokens is None else max_tokens)

        self.provider = create_provider(
            provider=provider,
            model=model,
            api_key=cfg.model.api_key or None,
            base_url=base_url or cfg.model.base_url or None,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            num_ctx=cfg.context.max_tokens,
        )
        set_provider(self.provider)
        self._refresh_runtime_model_details(source=source, model_id=model_id)

    def _session_model_selection(self) -> dict[str, Any] | None:
        """Return model-selection metadata for active session, if present."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return None
        raw = self.session.metadata.get("model_selection")
        if not isinstance(raw, dict):
            return None
        provider = str(raw.get("provider", "")).strip()
        model = str(raw.get("model", "")).strip()
        if not provider or not model:
            return None
        return {
            "id": str(raw.get("id", "")).strip(),
            "provider": provider,
            "model": model,
            "base_url": str(raw.get("base_url", "") or "").strip(),
            "temperature": raw.get("temperature"),
            "max_tokens": raw.get("max_tokens"),
        }

    def _apply_default_config_model_if_needed(self) -> None:
        """Apply config default model unless agent is using external provider override."""
        cfg = get_config()
        if self._provider_override and self.provider is not None:
            self._refresh_runtime_model_details(source="override")
            return
        self._apply_model_option(
            {
                "provider": str(cfg.model.provider),
                "model": str(cfg.model.model),
                "base_url": str(cfg.model.base_url or ""),
                "temperature": cfg.model.temperature,
                "max_tokens": cfg.model.max_tokens,
            },
            source="config",
            model_id="default",
        )

    async def set_session_model(self, selector: str, persist: bool = True) -> tuple[bool, str]:
        """Select runtime model for active session from allowed model list."""
        if not self.session:
            return False, "No active session"
        key = (selector or "").strip()
        if not key:
            return False, "Usage: /session model <id|#index|provider:model|default>"

        lowered = key.lower()
        if lowered in {"default", "config"}:
            self.session.metadata.pop("model_selection", None)
            self._apply_default_config_model_if_needed()
            if persist:
                await self.session_manager.save_session(self.session)
            return True, "Session model reset to default config"

        option = self._resolve_allowed_model(key)
        if not option:
            return False, f"Model not found in allowlist: {selector}"

        try:
            self._apply_model_option(
                option,
                source="session",
                model_id=str(option.get("id", "")).strip(),
            )
        except Exception as e:
            return False, f"Failed to activate model: {e}"
        details = self.get_runtime_model_details()
        self.session.metadata["model_selection"] = {
            "id": str(option.get("id", "")).strip(),
            "provider": details.get("provider"),
            "model": details.get("model"),
            "base_url": details.get("base_url", ""),
            "temperature": details.get("temperature"),
            "max_tokens": details.get("max_tokens"),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if persist:
            await self.session_manager.save_session(self.session)
        return True, f"Session model set to {details.get('provider')}/{details.get('model')}"

    async def initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return
        
        log.info("Initializing agent...")
        
        # Set up provider
        if self.provider is None:
            self.provider = get_provider()
        else:
            set_provider(self.provider)
        self._refresh_runtime_model_details(source="config")
        
        # Load tracked last active session when available, fallback to default create/load.
        self.session = await self.session_manager.load_last_active_session()
        if not self.session:
            self.session = await self.session_manager.get_or_create_session()
        await self.session_manager.set_last_active_session(self.session.id)
        self._sync_runtime_flags_from_session()
        
        # Register default tools
        self._register_default_tools()
        
        self._initialized = True
        log.info("Agent initialized", session_id=self.session.id)

    def _register_default_tools(self) -> None:
        """Register the default tool set."""
        from captain_claw.tools import (
            ShellTool,
            ReadTool,
            WriteTool,
            GlobTool,
            WebFetchTool,
            WebSearchTool,
        )
        
        config = get_config()
        
        # Register enabled tools
        for tool_name in config.tools.enabled:
            if tool_name == "shell":
                self.tools.register(ShellTool())
            elif tool_name == "read":
                self.tools.register(ReadTool())
            elif tool_name == "write":
                self.tools.register(WriteTool())
            elif tool_name == "glob":
                self.tools.register(GlobTool())
            elif tool_name == "web_fetch":
                self.tools.register(WebFetchTool())
            elif tool_name == "web_search":
                self.tools.register(WebSearchTool())

    def _build_system_prompt(self) -> str:
        """Build the system prompt."""
        session_id = self._current_session_slug()

        planning_block = ""
        if self.planning_enabled:
            planning_block = (
                "\n\n" + self.instructions.load("planning_mode_instructions.md")
            )

        saved_root = self.tools.get_saved_base_path(create=False)

        return self.instructions.render(
            "system_prompt.md",
            runtime_base_path=self.runtime_base_path,
            workspace_root=self.workspace_base_path,
            saved_root=saved_root,
            session_id=session_id,
            planning_block=planning_block,
        )

    def _build_tool_memory_note(
        self,
        skipped_tool_messages: list[dict[str, Any]],
        query: str | None,
        max_items: int = 3,
        max_snippet_chars: int = 700,
    ) -> tuple[str, str]:
        """Build compact continuity note from historical tool outputs."""
        if not skipped_tool_messages:
            return "", ""

        terms = {
            token
            for token in re.findall(r"[a-z0-9]+", (query or "").lower())
            if len(token) >= 4
        }

        ranked: list[tuple[int, int, dict[str, Any], list[str], list[str]]] = []
        for idx, msg in enumerate(skipped_tool_messages):
            content = str(msg.get("content", ""))
            lowered = content.lower()
            matched_terms = [term for term in terms if term in lowered] if terms else []
            score = len(matched_terms)
            urls = self._extract_source_links(msg, content)
            ranked.append((score, idx, msg, matched_terms, urls))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [entry for entry in ranked if entry[0] > 0][:max_items]
        if not selected:
            selected = ranked[:1]

        if not selected:
            return "", ""

        lines = [self.instructions.load("memory_continuity_header.md")]
        debug_lines = ["Memory selection details:"]
        selection_mode = "term_overlap" if any(score > 0 for score, *_ in selected) else "fallback_latest"
        debug_lines.append(f"selection_mode={selection_mode}")
        if terms:
            debug_lines.append(f"query_terms={', '.join(sorted(terms))}")
        else:
            debug_lines.append("query_terms=(none)")

        for score, idx, msg, matched_terms, urls in selected:
            tool_name = str(msg.get("tool_name") or "tool")
            snippet = str(msg.get("content", "")).strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars] + "... [truncated]"
            prefix = f"[{tool_name}]"
            if score > 0:
                prefix = f"{prefix} match:{score}"
            lines.append(f"- {prefix} {snippet}")

            matched_label = ", ".join(matched_terms) if matched_terms else "(none)"
            links_label = ", ".join(urls) if urls else "(none)"
            reason = f"term_overlap:{score}" if score > 0 else "fallback_latest"
            debug_lines.append(
                f"- message_index={idx} source={tool_name} reason={reason} matched={matched_label} links={links_label}"
            )

        return "\n".join(lines), "\n".join(debug_lines)

    def _requires_strict_tool_message_order(self) -> bool:
        """Whether active provider enforces OpenAI tool-message sequencing rules."""
        details = self.get_runtime_model_details()
        provider = str(details.get("provider", "")).strip().lower()
        return provider == "openai"

    @staticmethod
    def _normalize_session_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
        """Normalize persisted assistant tool_calls into OpenAI-compatible shape."""
        normalized: list[dict[str, Any]] = []
        if not isinstance(raw_tool_calls, list):
            return normalized
        for idx, raw in enumerate(raw_tool_calls, start=1):
            if not isinstance(raw, dict):
                continue
            call_id = str(raw.get("id", "")).strip() or f"call_{idx}"
            call_type = str(raw.get("type", "")).strip() or "function"
            function_obj = raw.get("function")
            if isinstance(function_obj, dict):
                name = str(function_obj.get("name", "")).strip()
                arguments = function_obj.get("arguments", {})
            else:
                name = str(raw.get("name", "")).strip()
                arguments = raw.get("arguments", {})
            if not name:
                continue
            if isinstance(arguments, dict):
                args_text = json.dumps(arguments, ensure_ascii=True)
            elif isinstance(arguments, str):
                args_text = arguments
            else:
                args_text = "{}"
            normalized.append({
                "id": call_id,
                "type": call_type,
                "function": {
                    "name": name,
                    "arguments": args_text,
                },
            })
        return normalized

    @staticmethod
    def _serialize_tool_calls_for_session(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """Serialize tool calls for session persistence + OpenAI follow-up context."""
        serialized: list[dict[str, Any]] = []
        for idx, call in enumerate(tool_calls, start=1):
            call_id = str(getattr(call, "id", "")).strip() or f"call_{idx}"
            name = str(getattr(call, "name", "")).strip()
            if not name:
                continue
            arguments = getattr(call, "arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            serialized.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=True),
                },
            })
        return serialized

    def _normalize_selected_messages_for_provider(
        self,
        selected_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize messages for providers with strict tool message ordering."""
        if not self._requires_strict_tool_message_order():
            return selected_messages

        normalized: list[dict[str, Any]] = []
        pending_tool_ids: set[str] = set()
        pending_assistant_idx: int | None = None

        def _clear_pending_assistant_tool_calls() -> None:
            nonlocal pending_tool_ids, pending_assistant_idx
            if pending_assistant_idx is not None and 0 <= pending_assistant_idx < len(normalized):
                normalized[pending_assistant_idx].pop("tool_calls", None)
            pending_tool_ids = set()
            pending_assistant_idx = None

        for msg in selected_messages:
            role = str(msg.get("role", "")).strip().lower()
            if role == "assistant":
                if pending_tool_ids:
                    # Previous assistant tool_calls chain did not complete in retained context.
                    _clear_pending_assistant_tool_calls()
                tool_calls = self._normalize_session_tool_calls(msg.get("tool_calls"))
                msg_copy = dict(msg)
                if tool_calls:
                    msg_copy["tool_calls"] = tool_calls
                else:
                    msg_copy.pop("tool_calls", None)
                normalized.append(msg_copy)
                pending_tool_ids = {
                    str(call.get("id", "")).strip()
                    for call in tool_calls
                    if str(call.get("id", "")).strip()
                }
                pending_assistant_idx = len(normalized) - 1 if pending_tool_ids else None
                continue

            if role == "tool":
                tool_call_id = str(msg.get("tool_call_id", "")).strip()
                if tool_call_id and tool_call_id in pending_tool_ids:
                    normalized.append(msg)
                    pending_tool_ids.discard(tool_call_id)
                    if not pending_tool_ids:
                        pending_assistant_idx = None
                    continue
                if pending_tool_ids:
                    # Tool chain was interrupted by an unmatched tool payload.
                    _clear_pending_assistant_tool_calls()

                # Orphan tool messages break OpenAI calls; retain content as assistant context instead.
                tool_name = str(msg.get("tool_name", "")).strip() or "tool"
                content = str(msg.get("content", "")).strip()
                converted = dict(msg)
                converted["role"] = "assistant"
                converted["content"] = f"[tool_context:{tool_name}] {content}".strip()
                converted.pop("tool_call_id", None)
                converted.pop("tool_name", None)
                converted.pop("tool_calls", None)
                normalized.append(converted)
                continue

            if pending_tool_ids:
                _clear_pending_assistant_tool_calls()
            normalized.append(msg)

        if pending_tool_ids:
            _clear_pending_assistant_tool_calls()

        return normalized

    def _build_messages(
        self,
        tool_messages_from_index: int | None = None,
        query: str | None = None,
        planning_pipeline: dict[str, Any] | None = None,
        list_task_plan: dict[str, Any] | None = None,
    ) -> list[Message]:
        """Build message list for LLM."""
        cfg = get_config()
        system_prompt = self._build_system_prompt()
        messages = [Message(role="system", content=system_prompt)]
        system_tokens = self._count_tokens(system_prompt)
        context_budget = max(1, int(cfg.context.max_tokens))
        history_budget = max(0, context_budget - system_tokens)

        candidate_messages: list[dict[str, Any]] = []
        skipped_historical_tools: list[dict[str, Any]] = []
        filter_historical_tools = tool_messages_from_index is not None
        if self.session:
            for idx, msg in enumerate(self.session.messages):
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                if (
                    str(msg.get("role", "")).strip().lower() == "tool"
                    and self._is_monitor_only_tool_name(tool_name)
                ):
                    continue
                # Keep only current-turn tool role messages.
                # Historical tool outputs are carried through continuity note below.
                if (
                    filter_historical_tools
                    and msg["role"] == "tool"
                    and idx < tool_messages_from_index
                ):
                    skipped_historical_tools.append(msg)
                    continue
                candidate_messages.append(msg)

        memory_note, memory_debug = self._build_tool_memory_note(
            skipped_historical_tools,
            query=query,
        )
        if memory_note:
            candidate_messages.append({
                "role": "assistant",
                "content": memory_note,
                "tool_name": "memory_context",
                "token_count": self._count_tokens(memory_note),
            })
            signature = f"{query or ''}|{memory_debug}"
            if signature != self._last_memory_debug_signature:
                self._emit_tool_output(
                    "memory_select",
                    {"query": query or ""},
                    memory_debug,
                )
                self._last_memory_debug_signature = signature

        planning_note = self._build_pipeline_note(planning_pipeline or {})
        if planning_note:
            candidate_messages.append({
                "role": "assistant",
                "content": planning_note,
                "tool_name": "planning_context",
                "token_count": self._count_tokens(planning_note),
            })
        list_note = self._build_list_task_note(list_task_plan or {})
        if list_note:
            candidate_messages.append({
                "role": "assistant",
                "content": list_note,
                "tool_name": "list_task_memory",
                "token_count": self._count_tokens(list_note),
            })

        selected_reversed: list[dict[str, Any]] = []
        used_tokens = 0
        dropped_messages = 0
        included_latest_user = False
        for msg in reversed(candidate_messages):
            msg_tokens = self._ensure_message_token_count(msg)
            must_include_latest_user = (
                (not included_latest_user) and str(msg.get("role", "")) == "user"
            )
            if used_tokens + msg_tokens <= history_budget or must_include_latest_user:
                selected_reversed.append(msg)
                used_tokens += msg_tokens
                if must_include_latest_user:
                    included_latest_user = True
            else:
                dropped_messages += 1

        selected_messages = list(reversed(selected_reversed))
        selected_messages = self._normalize_selected_messages_for_provider(selected_messages)
        for msg in selected_messages:
            messages.append(
                Message(
                    role=msg["role"],
                    content=msg["content"],
                    tool_call_id=msg.get("tool_call_id"),
                    tool_name=msg.get("tool_name"),
                    tool_calls=msg.get("tool_calls"),
                )
            )

        prompt_tokens = system_tokens + used_tokens
        self.last_context_window = {
            "context_budget_tokens": context_budget,
            "system_tokens": system_tokens,
            "history_budget_tokens": history_budget,
            "history_tokens": used_tokens,
            "prompt_tokens": prompt_tokens,
            "total_messages": len(candidate_messages),
            "included_messages": len(selected_messages),
            "dropped_messages": dropped_messages,
            "historical_tool_messages_filtered": len(skipped_historical_tools),
            "memory_note_used": 1 if memory_note else 0,
            "planning_note_used": 1 if planning_note else 0,
            "over_budget": 1 if prompt_tokens > context_budget else 0,
            "utilization": (prompt_tokens / context_budget) if context_budget else 0.0,
        }
        if self.session:
            self.session.metadata["context_window"] = dict(self.last_context_window)
        if dropped_messages:
            log.info(
                "Context window pruned history",
                dropped_messages=dropped_messages,
                included_messages=len(selected_messages),
                prompt_tokens=prompt_tokens,
                budget=context_budget,
            )

        return messages

    def _extract_command_from_response(self, content: str) -> str | None:
        """Extract shell command from model response.
        
        Looks for:
        - ```bash\ncommand\n``` or ```shell\ncommand\n```
        - "I'll run: command"
        - "Running: command"
        """
        import re
        
        if not content:
            return None
        
        # Only match explicit shell/code blocks or explicit commands
        patterns = [
            r'```(?:bash|shell|sh)\s*\n(.*?)\n```',  # ```bash\ncommand\n```
            r'```\s*\n(.*?)\n```(?:\s|$)',  # ```\ncommand\n``` followed by whitespace or end
            r"I'(?:ll| will) run[:\s]+[`\"]?(.+?)[`\"]?(?:\n|$)",  # I'll run: `command`
            r"(?:exec|execute)[:\s]+[`\"](.+?)[`\"]",  # exec: `command`
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                cmd = match.group(1).strip()
                # Must look like a shell command (has spaces or special chars)
                if cmd and len(cmd) > 2 and (' ' in cmd or '|' in cmd or '&' in cmd or '/' in cmd):
                    return cmd
        
        return None

    def _supports_tool_result_followup(self) -> bool:
        """Whether model can handle a follow-up turn with tool result messages."""
        details = self.get_runtime_model_details()
        provider = str(details.get("provider", "")).lower()
        model = str(details.get("model", "")).lower()

        # Known issue: Ollama cloud models often return HTTP 500 when tool role
        # messages are included in the follow-up request.
        if provider == "ollama" and model.endswith(":cloud"):
            return False

        return True

    def _collect_turn_tool_output(self, turn_start_idx: int) -> str:
        """Collect tool outputs for the current turn."""
        if not self.session:
            return ""
        outputs: list[str] = []
        for msg in self.session.messages[turn_start_idx:]:
            if msg.get("role") != "tool":
                continue
            if self._is_monitor_only_tool_name(str(msg.get("tool_name", ""))):
                continue
            content = str(msg.get("content", "")).strip()
            if content:
                outputs.append(content)
        return "\n\n".join(outputs)

    def _turn_has_successful_tool(self, turn_start_idx: int, tool_name: str) -> bool:
        """Check whether a successful tool result exists in current turn."""
        if not self.session:
            return False
        target = (tool_name or "").strip().lower()
        for msg in self.session.messages[turn_start_idx:]:
            if msg.get("role") != "tool":
                continue
            if str(msg.get("tool_name", "")).strip().lower() != target:
                continue
            content = str(msg.get("content", "")).strip().lower()
            if not content.startswith("error:"):
                return True
        return False

    @staticmethod
    def _is_explicit_script_request(user_input: str) -> bool:
        """Detect explicit user requests to generate/build a script."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        return bool(
            re.search(
                r"\b(generate|create|build|write|make)\b.{0,40}\bscript\b"
                r"|\bscript\b.{0,40}\b(generate|create|build|write|make)\b",
                text,
            )
        )

    @staticmethod
    def _is_list_processing_request(user_input: str) -> bool:
        """Detect requests that imply processing multiple items/entities."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        if re.search(r"\btop\s+\d+\b", text):
            return True
        list_markers = (
            r"\bfor each\b",
            r"\beach\b",
            r"\bevery\b",
            r"\ball\b",
            r"\bper\b",
            r"\blist\b",
            r"\bextract\b.{0,30}\bnames?\b",
            r"\bcompanies?\b",
            r"\bcities?\b",
            r"\bsources?\b",
        )
        return any(re.search(pattern, text) for pattern in list_markers)

    @staticmethod
    def _should_enforce_python_worker_mode(user_input: str) -> bool:
        """Whether this turn should prefer Python worker/tool execution."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        if Agent._is_explicit_script_request(text):
            return True
        if not Agent._is_list_processing_request(text):
            return False
        action_like = bool(
            re.search(r"\b(fetch|search|read|summari[sz]e|analy[sz]e|write|save|export|generate)\b", text)
        )
        output_like = bool(
            re.search(r"\bfile\b|\bfiles\b|<[^>]+>|-summary\.md\b|-details\.md\b", text)
        )
        return action_like and output_like

    @staticmethod
    def _normalize_list_members(raw_members: Any, max_members: int = 40) -> list[str]:
        """Normalize extracted list members into a stable ordered unique list."""
        members: list[str] = []
        seen: set[str] = set()
        items: list[Any]
        if isinstance(raw_members, list):
            items = raw_members
        elif isinstance(raw_members, dict):
            items = [raw_members]
        else:
            items = []
        for item in items:
            if len(members) >= max_members:
                break
            if isinstance(item, dict):
                candidate = str(
                    item.get("name")
                    or item.get("member")
                    or item.get("item")
                    or item.get("title")
                    or ""
                ).strip()
            else:
                candidate = str(item or "").strip()
            if not candidate:
                continue
            candidate = re.sub(r"\s+", " ", candidate).strip(" -\t\r\n")
            if len(candidate) < 2:
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            members.append(candidate[:160])
        return members

    @staticmethod
    def _choose_list_execution_strategy(
        user_input: str,
        members_count: int,
        recommended: str = "",
    ) -> str:
        """Choose execution strategy for per-member work."""
        rec = str(recommended or "").strip().lower()
        if rec in {"script", "direct"}:
            return rec
        text = (user_input or "").strip().lower()
        if Agent._is_explicit_script_request(text):
            return "script"
        if members_count >= 8:
            return "script"
        if members_count >= 4 and re.search(r"\b(fetch|search|crawl|scrap|source|web)\b", text):
            return "script"
        if members_count >= 5 and re.search(r"\b(write|save|export|file|files)\b", text):
            return "script"
        return "direct"

    def _collect_list_extraction_context(
        self,
        max_messages: int = 18,
        max_chars: int = 12000,
        per_message_chars: int = 1400,
    ) -> str:
        """Collect compact recent context to help list-member extraction."""
        if not self.session:
            return ""
        start = max(0, len(self.session.messages) - max_messages)
        lines: list[str] = []
        total_chars = 0
        for msg in self.session.messages[start:]:
            role = str(msg.get("role", "")).strip().lower() or "unknown"
            tool_name = str(msg.get("tool_name", "")).strip()
            if role == "tool" and self._is_monitor_only_tool_name(tool_name):
                continue
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if len(content) > per_message_chars:
                content = content[:per_message_chars] + "... [truncated]"
            prefix = f"[{role}]"
            if role == "tool" and tool_name:
                prefix = f"[{role}:{tool_name}]"
            line = f"{prefix} {content}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 1
        return "\n".join(lines)

    @staticmethod
    def _list_member_aliases(member: str) -> set[str]:
        """Build simple aliases for matching member coverage in outputs."""
        base = str(member or "").strip().lower()
        if not base:
            return set()
        aliases = {base}
        normalized_words = re.sub(r"[^a-z0-9]+", " ", base).strip()
        if normalized_words:
            aliases.add(normalized_words)
            aliases.add(normalized_words.replace(" ", "-"))
            aliases.add(normalized_words.replace(" ", "_"))
            aliases.add(normalized_words.replace(" ", ""))
        return {alias for alias in aliases if len(alias) >= 2}

    def _evaluate_list_member_coverage(
        self,
        members: list[str],
        candidate_response: str,
        turn_start_idx: int,
    ) -> tuple[list[str], list[str]]:
        """Evaluate which list members are covered in this turn outputs."""
        if not members:
            return [], []
        text_parts: list[str] = [str(candidate_response or "")]
        if self.session:
            for msg in self.session.messages[turn_start_idx:]:
                role = str(msg.get("role", "")).strip().lower()
                if role not in {"assistant", "tool"}:
                    continue
                tool_name = str(msg.get("tool_name", "")).strip().lower()
                if role == "tool" and self._is_monitor_only_tool_name(tool_name):
                    continue
                content = str(msg.get("content", "")).strip()
                if content:
                    text_parts.append(content)
        haystack = "\n".join(text_parts).lower()

        showcase_root = (
            self.tools.get_saved_base_path(create=False)
            / "showcase"
            / self._current_session_slug()
        )
        showcase_names: list[str] = []
        if showcase_root.exists():
            try:
                showcase_names = [
                    path.name.lower()
                    for path in showcase_root.rglob("*")
                    if path.is_file()
                ]
            except Exception:
                showcase_names = []

        covered: list[str] = []
        missing: list[str] = []
        for member in members:
            aliases = self._list_member_aliases(member)
            in_text = any(alias in haystack for alias in aliases)
            in_artifacts = False
            if not in_text and showcase_names:
                in_artifacts = any(
                    any(alias in filename for alias in aliases)
                    for filename in showcase_names
                )
            if in_text or in_artifacts:
                covered.append(member)
            else:
                missing.append(member)
        return covered, missing

    @staticmethod
    def _build_list_coverage_feedback(
        missing_members: list[str],
        strategy: str,
        per_member_action: str,
    ) -> str:
        """Build retry guidance when not all extracted list members are covered."""
        if not missing_members:
            return ""
        preview = ", ".join(missing_members[:8])
        if len(missing_members) > 8:
            preview += f", ... (+{len(missing_members) - 8} more)"
        action_line = f"Requested per-member action: {per_member_action}" if per_member_action else ""
        if strategy == "script":
            return (
                "Completion gate: extracted list members are still missing.\n"
                f"Missing members: {preview}\n"
                f"{action_line}\n"
                "Regenerate or adjust the Python worker to process all missing members, execute it, "
                "and then provide final concise output."
            ).strip()
        return (
            "Completion gate: extracted list members are still missing.\n"
            f"Missing members: {preview}\n"
            f"{action_line}\n"
            "Continue in direct loop mode: process each missing member one-by-one using tools as needed, "
            "then return final concise output."
        ).strip()

    @staticmethod
    def _apply_list_requirements(
        base_requirements: list[dict[str, Any]],
        list_task_plan: dict[str, Any],
        max_members: int = 30,
    ) -> list[dict[str, Any]]:
        """Augment completion requirements with extracted list-member coverage checks."""
        requirements = [dict(item) for item in base_requirements if isinstance(item, dict)]
        if not isinstance(list_task_plan, dict) or not bool(list_task_plan.get("enabled", False)):
            return requirements
        members = list_task_plan.get("members")
        if not isinstance(members, list) or not members:
            return requirements
        action = str(list_task_plan.get("per_member_action", "")).strip()
        existing_ids = {str(req.get("id", "")).strip() for req in requirements}
        for idx, member in enumerate(members[:max_members], start=1):
            base_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(member).strip().lower()).strip("_")[:28]
            if not base_id:
                base_id = f"member_{idx}"
            req_id = f"req_member_{base_id}"
            if req_id in existing_ids:
                continue
            existing_ids.add(req_id)
            title = f"Cover list member: {member}"
            if action:
                title = f"{title} ({action})"
            requirements.append({"id": req_id, "title": title[:220]})
        return requirements

    async def _generate_list_task_plan(
        self,
        user_input: str,
        context_excerpt: str,
        turn_usage: dict[str, int],
    ) -> dict[str, Any]:
        """Extract list members from context and select direct vs script strategy."""
        fallback = {
            "enabled": False,
            "members": [],
            "strategy": "none",
            "per_member_action": "",
            "confidence": "low",
        }
        if not self._is_list_processing_request(user_input):
            return fallback

        messages = [
            Message(
                role="system",
                content=self.instructions.load("list_task_extractor_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "list_task_extractor_user_prompt.md",
                    user_input=user_input,
                    context_excerpt=context_excerpt or "(empty)",
                ),
            ),
        ]
        payload: dict[str, Any] | None = None
        try:
            response = await self._complete_with_guards(
                messages=messages,
                tools=None,
                interaction_label="list_task_extractor",
                turn_usage=turn_usage,
                max_tokens=min(1000, int(get_config().model.max_tokens)),
            )
            payload = self._extract_json_object(response.content or "")
        except Exception as e:
            self._emit_tool_output(
                "task_contract",
                {"step": "list_extract_error"},
                f"step=list_extract_error\nerror={str(e)}",
            )
            payload = None

        has_list_work = False
        members: list[str] = []
        per_member_action = ""
        recommended_strategy = ""
        confidence = "low"
        if isinstance(payload, dict):
            has_list_work = bool(payload.get("has_list_work", False))
            members = self._normalize_list_members(payload.get("members"))
            per_member_action = str(payload.get("per_member_action", "")).strip()[:220]
            recommended_strategy = str(payload.get("recommended_strategy", "")).strip().lower()
            confidence = str(payload.get("confidence", "low")).strip().lower()[:16] or "low"

        if not has_list_work and not members:
            return fallback
        strategy = self._choose_list_execution_strategy(
            user_input=user_input,
            members_count=len(members),
            recommended=recommended_strategy,
        )
        plan = {
            "enabled": bool(members),
            "members": members,
            "strategy": strategy,
            "per_member_action": per_member_action,
            "confidence": confidence,
        }
        preview = ", ".join(members[:8]) if members else "(none)"
        if len(members) > 8:
            preview += f", ... (+{len(members) - 8} more)"
        self._emit_tool_output(
            "task_contract",
            {
                "step": "list_extract_done",
                "enabled": bool(plan["enabled"]),
                "members": len(members),
                "strategy": strategy,
                "confidence": confidence,
            },
            (
                "step=list_extract_done\n"
                f"enabled={plan['enabled']}\n"
                f"members={len(members)}\n"
                f"strategy={strategy}\n"
                f"confidence={confidence}\n"
                f"members_preview={preview}"
            ),
        )
        return plan

    @staticmethod
    def _extract_code_block(text: str) -> tuple[str | None, str | None]:
        """Extract first fenced code block as (language, code)."""
        if not text:
            return None, None
        match = re.search(r"```([A-Za-z0-9_+\-]*)\n(.*?)```", text, flags=re.DOTALL)
        if not match:
            return None, None
        language = (match.group(1) or "").strip().lower() or None
        code = (match.group(2) or "").strip()
        if not code:
            return language, None
        return language, code

    @staticmethod
    def _infer_script_extension(language: str | None, code: str) -> str:
        """Infer script extension from language tag or content."""
        lang = (language or "").strip().lower()
        mapping = {
            "python": ".py",
            "py": ".py",
            "bash": ".sh",
            "sh": ".sh",
            "shell": ".sh",
            "zsh": ".sh",
            "javascript": ".js",
            "js": ".js",
            "node": ".js",
            "ruby": ".rb",
            "rb": ".rb",
        }
        if lang in mapping:
            return mapping[lang]
        stripped = (code or "").lstrip()
        if stripped.startswith("#!/usr/bin/env python") or stripped.startswith("#!/usr/bin/python"):
            return ".py"
        if stripped.startswith("#!/usr/bin/env bash") or stripped.startswith("#!/bin/bash"):
            return ".sh"
        if stripped.startswith("#!/usr/bin/env sh") or stripped.startswith("#!/bin/sh"):
            return ".sh"
        return ".py"

    @staticmethod
    def _supported_script_extension(ext: str) -> bool:
        """Return whether extension can be executed with built-in runner mapping."""
        return (ext or "").lower() in {".py", ".sh", ".js", ".rb"}

    @staticmethod
    def _build_script_runner_command(script_path: Path) -> str | None:
        """Build shell command that runs script from its own directory."""
        ext = script_path.suffix.lower()
        filename = shlex.quote(script_path.name)
        if ext == ".py":
            runner = f"python3 {filename}"
        elif ext == ".sh":
            runner = f"bash {filename}"
        elif ext == ".js":
            runner = f"node {filename}"
        elif ext == ".rb":
            runner = f"ruby {filename}"
        else:
            return None
        script_dir = shlex.quote(str(script_path.parent))
        return f"cd {script_dir} && {runner}"

    @staticmethod
    def _build_python_runner_command(script_path: Path) -> str:
        """Build shell command using the active Python interpreter."""
        interpreter = shlex.quote(str(Path(sys.executable).resolve()))
        script_dir = shlex.quote(str(script_path.parent))
        filename = shlex.quote(script_path.name)
        return f"cd {script_dir} && {interpreter} {filename}"

    def _current_session_slug(self) -> str:
        """Return normalized session id used for folder scoping."""
        session_key = "default"
        if self.session and self.session.id:
            raw_id = str(self.session.id).strip()
            normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_id).strip("-")
            if normalized:
                session_key = normalized
        return session_key

    def _build_script_relative_path(
        self,
        user_input: str,
        extension: str,
    ) -> str:
        """Build script path under scripts/<session-id>/."""
        ext = extension if extension.startswith(".") else f".{extension}"
        requested = self._extract_requested_write_path(user_input)
        filename: str
        if requested:
            candidate = Path(requested).name
            if "." in candidate:
                base = Path(candidate).stem or "generated_script"
                req_ext = Path(candidate).suffix.lower()
                filename = base + (req_ext if self._supported_script_extension(req_ext) else ext)
            else:
                filename = candidate + ext
        else:
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"generated_script_{stamp}{ext}"
        return f"scripts/{self._current_session_slug()}/{filename}"

    @staticmethod
    def _parse_written_path_from_tool_output(tool_output: str) -> Path | None:
        """Parse final path from write tool output."""
        text = (tool_output or "").strip()
        match = re.search(r"\bto\s+(.+?)(?:\s+\(requested:|\s*$)", text)
        if not match:
            return None
        raw_path = match.group(1).strip()
        if not raw_path:
            return None
        try:
            return Path(raw_path)
        except Exception:
            return None

    async def _synthesize_script_content(
        self,
        user_input: str,
        turn_usage: dict[str, int],
    ) -> tuple[str, str]:
        """Generate script content when model answer omitted code block."""
        synth_messages = [
            Message(
                role="system",
                content=self.instructions.load("script_synthesis_system_prompt.md"),
            ),
            Message(role="user", content=user_input),
        ]
        try:
            self._set_runtime_status("thinking")
            response = await self._complete_with_guards(
                messages=synth_messages,
                tools=None,
                interaction_label="script_synthesis",
                turn_usage=turn_usage,
            )
            language, code = self._extract_code_block(response.content or "")
            if code:
                return code, self._infer_script_extension(language, code)
            raw = (response.content or "").strip()
            if raw:
                return raw, self._infer_script_extension(language, raw)
        except Exception as e:
            log.warning("Script synthesis fallback failed", error=str(e))

        # Deterministic fallback scaffold.
        safe_request = re.sub(r"\s+", " ", (user_input or "").strip())[:240]
        scaffold = (
            "#!/usr/bin/env python3\n"
            "\"\"\"Auto-generated script scaffold.\"\"\"\n\n"
            "def main() -> None:\n"
            f"    print({safe_request!r})\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        return scaffold, ".py"

    def _build_python_worker_prompt(
        self,
        user_input: str,
        list_task_plan: dict[str, Any] | None = None,
    ) -> str:
        """Build synthesis prompt for batch/list worker script generation."""
        session_id = self._current_session_slug()
        members_block = ""
        per_member_action = ""
        if isinstance(list_task_plan, dict):
            members = list_task_plan.get("members")
            if isinstance(members, list) and members:
                rendered = "\n".join(f"- {str(item)}" for item in members[:80])
                members_block = f"\nList members to process:\n{rendered}\n"
            per_member_action = str(list_task_plan.get("per_member_action", "")).strip()
        return (
            "Create one runnable Python 3 script for this task.\n"
            "Requirements:\n"
            "- Complete the full task end-to-end.\n"
            "- If a list of entities/items is discovered, iterate ALL items; never stop after the first item.\n"
            "- Use text extraction by default when parsing fetched pages.\n"
            f"- Save per-item outputs under saved/showcase/{session_id}/.\n"
            "- Use deterministic filenames based on item names where requested.\n"
            "- Print concise progress logs so monitor output shows progress.\n"
            f"- Per-member action focus: {per_member_action or 'Follow user request for each member.'}\n"
            "- Return code only.\n\n"
            f"{members_block}"
            f"User request:\n{user_input}"
        )

    async def _run_python_worker_for_list_task(
        self,
        user_input: str,
        turn_usage: dict[str, int],
        list_task_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate/write/run a temporary Python worker for list-style tasks."""
        worker_prompt = self._build_python_worker_prompt(user_input, list_task_plan=list_task_plan)
        code, extension = await self._synthesize_script_content(worker_prompt, turn_usage)
        if extension.lower() != ".py":
            extension = ".py"
        script_rel_path = self._build_script_relative_path(
            f"generate script {datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}{extension}",
            extension,
        ).replace("scripts/", "tools/", 1)
        write_state = await self._write_file_with_verification(
            path=script_rel_path,
            content=code,
            turn_usage=turn_usage,
            interaction_label="list_worker_write",
            max_attempts=2,
        )
        if not bool(write_state.get("success", False)):
            return {"success": False, "step": "write_failed", "error": str(write_state.get("output", ""))}

        written_script_path = Path(str(write_state.get("path", script_rel_path)))
        run_command = self._build_python_runner_command(written_script_path)
        try:
            shell_result = await self._execute_tool_with_guard(
                name="shell",
                arguments={"command": run_command},
                interaction_label="list_worker_run",
                turn_usage=turn_usage,
            )
            shell_output = shell_result.content if shell_result.success else f"Error: {shell_result.error}"
        except Exception as e:
            shell_result = None
            shell_output = f"Error: {str(e)}"

        self._add_session_message(
            role="tool",
            content=shell_output,
            tool_name="shell",
            tool_arguments={"command": run_command},
        )
        self._emit_tool_output("shell", {"command": run_command}, shell_output)
        if not (shell_result and shell_result.success):
            return {
                "success": False,
                "step": "run_failed",
                "path": str(written_script_path),
                "error": shell_output,
            }
        return {"success": True, "step": "completed", "path": str(written_script_path)}

    async def _maybe_auto_script_requested_output(
        self,
        user_input: str,
        output_text: str,
        turn_start_idx: int,
        turn_usage: dict[str, int],
    ) -> str:
        """Guarantee explicit script requests produce write+run tool actions."""
        text = (output_text or "").strip()
        if not self._is_explicit_script_request(user_input):
            return text

        has_write = self._turn_has_successful_tool(turn_start_idx, "write")
        has_shell = self._turn_has_successful_tool(turn_start_idx, "shell")
        if has_write and has_shell:
            return text

        written_script_path: Path | None = None

        if not has_write:
            language, code = self._extract_code_block(text)
            if not code:
                code, inferred_ext = await self._synthesize_script_content(user_input, turn_usage)
            else:
                inferred_ext = self._infer_script_extension(language, code)

            script_rel_path = self._build_script_relative_path(user_input, inferred_ext)
            write_state = await self._write_file_with_verification(
                path=script_rel_path,
                content=code,
                turn_usage=turn_usage,
                interaction_label="auto_script_write",
                max_attempts=2,
            )
            if not bool(write_state.get("success", False)):
                return (
                    f"{text}\n\n"
                    "Note: explicit script request could not be completed because write failed.\n"
                    f"{str(write_state.get('output', ''))}"
                ).strip()

            written_script_path = Path(str(write_state.get("path", script_rel_path)))
        else:
            # Reuse existing write result path from this turn when available.
            if self.session:
                for msg in reversed(self.session.messages[turn_start_idx:]):
                    if msg.get("role") != "tool" or str(msg.get("tool_name")) != "write":
                        continue
                    content = str(msg.get("content", "")).strip()
                    if content.lower().startswith("error:"):
                        continue
                    parsed = self._parse_written_path_from_tool_output(content)
                    if parsed:
                        written_script_path = parsed
                        break

        if has_shell:
            return text

        if not written_script_path:
            return (
                f"{text}\n\n"
                "Note: script was requested but executable path could not be resolved for run."
            ).strip()

        run_command = self._build_script_runner_command(written_script_path)
        if not run_command:
            return (
                f"{text}\n\n"
                f"Note: script saved to {written_script_path}, but auto-run is unsupported for extension "
                f"'{written_script_path.suffix or '(none)'}'."
            ).strip()

        try:
            shell_result = await self._execute_tool_with_guard(
                name="shell",
                arguments={"command": run_command},
                interaction_label="auto_script_run",
                turn_usage=turn_usage,
            )
            shell_output = (
                shell_result.content if shell_result.success else f"Error: {shell_result.error}"
            )
        except Exception as e:
            shell_result = None
            shell_output = f"Error: {str(e)}"

        self._add_session_message(
            role="tool",
            content=shell_output,
            tool_name="shell",
            tool_arguments={"command": run_command},
        )
        self._emit_tool_output("shell", {"command": run_command}, shell_output)

        if shell_result and shell_result.success:
            return (
                f"{text}\n\n"
                f"Script saved and executed from {written_script_path.parent}."
            ).strip()
        return (
            f"{text}\n\n"
            f"Script saved to {written_script_path}, but execution failed.\n"
            f"{shell_output}"
        ).strip()

    @staticmethod
    def _extract_requested_write_paths(user_input: str) -> list[str]:
        """Extract requested target file paths from user input."""
        text = (user_input or "").strip()
        if not text:
            return []
        if not re.search(r"\b(write|save|store|export|dump)\b", text, flags=re.IGNORECASE):
            return []

        patterns = [
            r"(?:to|into|as)\s+(?:a\s+)?(?:file\s+)?[`\"']?([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,16})",
            r"(?:file\s+|named\s+|name\s+it\s+)[`\"']?([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,16})",
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if not matches:
                continue
            for match in matches:
                candidate = match.strip().rstrip(".,;:!?)]}>")
                lowered = candidate.lower()
                if lowered.startswith(("http://", "https://")):
                    continue
                if not candidate:
                    continue
                key = lowered
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(candidate)
        return ordered

    @staticmethod
    def _extract_requested_write_path(user_input: str) -> str | None:
        """Extract last requested target file path from user input."""
        paths = Agent._extract_requested_write_paths(user_input)
        if not paths:
            return None
        return paths[-1]

    @staticmethod
    def _is_explicit_file_save_request(user_input: str) -> bool:
        """Detect whether user explicitly asked for file creation/saving."""
        text = (user_input or "").strip().lower()
        if not text:
            return False
        has_write_action = bool(re.search(r"\b(write|save|store|export|dump|create)\b", text))
        has_file_target = bool(re.search(r"\bfile\b|\bfiles\b", text))
        return has_write_action and has_file_target

    @staticmethod
    def _extract_named_file_blocks(output_text: str) -> list[tuple[str, str]]:
        """Extract `(filename, content)` pairs from assistant text blocks."""
        text = (output_text or "").strip()
        if not text:
            return []
        header_re = re.compile(r"^\s*filename\s*:\s*(.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)
        matches = list(header_re.finditer(text))
        if not matches:
            return []

        blocks: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for idx, match in enumerate(matches):
            raw_path = (match.group(1) or "").strip().strip("`\"'")
            raw_path = raw_path.rstrip(".,;:!?)]}>")
            if not raw_path:
                continue
            if not re.search(r"\.[A-Za-z0-9]{1,16}$", raw_path):
                continue

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if not chunk:
                continue
            chunk_lines = chunk.splitlines()
            if chunk_lines and chunk_lines[0].strip() == "---":
                chunk = "\n".join(chunk_lines[1:]).strip()
            if not chunk:
                continue
            key = raw_path.lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            blocks.append((raw_path, chunk))
        return blocks

    def _collect_recent_file_blocks_from_session(
        self,
        max_assistant_messages: int = 12,
    ) -> list[tuple[str, str]]:
        """Collect most recent assistant `Filename:` blocks from session history."""
        if not self.session:
            return []

        checked = 0
        for msg in reversed(self.session.messages):
            if str(msg.get("role", "")).strip().lower() != "assistant":
                continue
            checked += 1
            blocks = self._extract_named_file_blocks(str(msg.get("content", "")))
            if blocks:
                return blocks
            if checked >= max_assistant_messages:
                break
        return []

    def _candidate_write_paths_for_verification(
        self,
        requested_path: str,
        parsed_written: Path | None = None,
    ) -> list[Path]:
        """Build candidate on-disk paths to verify file was actually saved."""
        candidates: list[Path] = []
        seen: set[str] = set()

        def _add(path: Path) -> None:
            try:
                resolved = path.expanduser().resolve()
            except Exception:
                return
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            candidates.append(resolved)

        if parsed_written is not None:
            _add(parsed_written)

        raw = str(requested_path or "").strip()
        if not raw:
            return candidates

        requested = Path(raw).expanduser()
        if requested.is_absolute():
            _add(requested)
        else:
            _add(self.runtime_base_path / requested)
            _add(self.tools.get_saved_base_path(create=False) / requested)
            parts = requested.parts
            if parts and parts[0].lower() == "saved":
                tail = Path(*parts[1:]) if len(parts) > 1 else Path("output.txt")
                _add(self.tools.get_saved_base_path(create=False) / tail)
        return candidates

    async def _write_file_with_verification(
        self,
        path: str,
        content: str,
        turn_usage: dict[str, int],
        interaction_label: str,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        """Write file, verify it exists, and retry once when not persisted."""
        attempts = max(1, int(max_attempts))
        last_output = ""
        last_written_path: Path | None = None

        for attempt in range(1, attempts + 1):
            write_args = {"path": path, "content": content}
            write_args_for_log = {
                "path": path,
                "content_chars": len(content),
                "append": False,
                "attempt": attempt,
            }
            try:
                result = await self._execute_tool_with_guard(
                    name="write",
                    arguments=write_args,
                    interaction_label=interaction_label,
                    turn_usage=turn_usage,
                )
                tool_output = result.content if result.success else f"Error: {result.error}"
            except Exception as e:
                result = None
                tool_output = f"Error: {str(e)}"

            self._add_session_message(
                role="tool",
                content=tool_output,
                tool_name="write",
                tool_arguments=write_args_for_log,
            )
            self._emit_tool_output("write", write_args_for_log, tool_output)

            parsed_written = self._parse_written_path_from_tool_output(tool_output)
            if parsed_written is not None:
                last_written_path = parsed_written

            verified_path: Path | None = None
            for candidate in self._candidate_write_paths_for_verification(path, parsed_written):
                try:
                    if candidate.is_file():
                        verified_path = candidate
                        break
                except Exception:
                    continue

            if verified_path is not None:
                return {
                    "success": True,
                    "attempts": attempt,
                    "output": tool_output,
                    "path": str(verified_path),
                }

            last_output = tool_output
            if attempt < attempts:
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "write_verify_retry",
                        "path": path,
                        "attempt": attempt + 1,
                    },
                    (
                        f"step=write_verify_retry\n"
                        f"path={path}\n"
                        f"attempt={attempt + 1}\n"
                        "reason=file_missing_after_write"
                    ),
                )

        return {
            "success": False,
            "attempts": attempts,
            "output": last_output or "Error: write verification failed",
            "path": str(last_written_path) if last_written_path is not None else path,
        }

    async def _maybe_auto_write_requested_output(
        self,
        user_input: str,
        output_text: str,
        turn_start_idx: int,
        turn_usage: dict[str, int],
    ) -> str:
        """Auto-run write tool when user explicitly requested file output."""
        text = (output_text or "").strip()
        if self._is_explicit_script_request(user_input):
            return await self._maybe_auto_script_requested_output(
                user_input=user_input,
                output_text=text,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
            )
        requested_paths = self._extract_requested_write_paths(user_input)
        requested_path = requested_paths[-1] if requested_paths else None
        explicit_file_request = self._is_explicit_file_save_request(user_input)

        file_blocks = self._extract_named_file_blocks(text)
        if (
            not file_blocks
            and explicit_file_request
            and re.search(r"\b(all|those)\s+files\b", user_input, flags=re.IGNORECASE)
        ):
            file_blocks = self._collect_recent_file_blocks_from_session()

        if file_blocks:
            saved_entries: list[str] = []
            failed_entries: list[str] = []
            for raw_path, raw_content in file_blocks:
                target_path = raw_path
                if "/" not in raw_path and "\\" not in raw_path:
                    target_path = f"showcase/{self._current_session_slug()}/{raw_path}"
                write_state = await self._write_file_with_verification(
                    path=target_path,
                    content=raw_content,
                    turn_usage=turn_usage,
                    interaction_label="auto_write_output_multi",
                    max_attempts=2,
                )
                tool_output = str(write_state.get("output", "")).strip()
                attempts = int(write_state.get("attempts", 1) or 1)
                if bool(write_state.get("success", False)):
                    retry_note = " (retried)" if attempts > 1 else ""
                    saved_entries.append(f"{raw_path}: {tool_output}{retry_note}")
                else:
                    failed_entries.append(f"{raw_path}: {tool_output}")

            if saved_entries and not failed_entries:
                summary = "\n".join(f"- {entry}" for entry in saved_entries)
                return f"{text}\n\nSaved {len(saved_entries)} files:\n{summary}".strip()
            if saved_entries or failed_entries:
                ok_summary = "\n".join(f"- {entry}" for entry in saved_entries) or "- (none)"
                fail_summary = "\n".join(f"- {entry}" for entry in failed_entries) or "- (none)"
                return (
                    f"{text}\n\n"
                    f"Auto file-save summary:\n"
                    f"Saved ({len(saved_entries)}):\n{ok_summary}\n"
                    f"Failed ({len(failed_entries)}):\n{fail_summary}"
                ).strip()
            return text

        if not requested_path:
            return text
        if self._turn_has_successful_tool(turn_start_idx, "write"):
            return text

        write_state = await self._write_file_with_verification(
            path=requested_path,
            content=text,
            turn_usage=turn_usage,
            interaction_label="auto_write_output",
            max_attempts=2,
        )
        tool_output = str(write_state.get("output", "")).strip()
        if bool(write_state.get("success", False)):
            return f"{text}\n\n{tool_output}".strip()
        return (
            f"{text}\n\n"
            f"Note: requested file save to '{requested_path}' failed.\n"
            f"{tool_output}"
        ).strip()

    async def _persist_assistant_response(self, content: str) -> None:
        """Persist assistant response for the current turn."""
        if not self.session:
            return
        self._add_session_message("assistant", content)
        await self.session_manager.save_session(self.session)

    @staticmethod
    def _clip_tool_output_for_rewrite(raw: str, max_chars: int) -> tuple[str, str]:
        """Clip tool output while preserving coverage across multiple blocks/sources."""
        text = (raw or "").strip()
        if len(text) <= max_chars:
            return text, ""

        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        if len(blocks) <= 1:
            return text[:max_chars], "\n\n[Tool output truncated before rewrite due to size limits.]"

        target_blocks = min(len(blocks), 24)
        per_block = max(350, max_chars // target_blocks)
        kept: list[str] = []
        used = 0
        for block in blocks:
            snippet = block
            if len(snippet) > per_block:
                snippet = snippet[:per_block].rstrip() + "... [truncated]"
            projected = used + len(snippet) + (2 if kept else 0)
            if projected > max_chars:
                break
            kept.append(snippet)
            used = projected
        if not kept:
            kept = [text[:max_chars]]
        clipped = "\n\n".join(kept)
        return clipped, "\n\n[Tool output truncated before rewrite due to size limits.]"

    async def _friendly_tool_output_response(
        self,
        user_input: str,
        tool_output: str,
        turn_usage: dict[str, int],
    ) -> str:
        """Ask model to rewrite raw tool output into a user-friendly answer."""
        raw = (tool_output or "").strip() or "[no output]"
        max_tool_output_chars = 45000
        clipped, clipped_note = self._clip_tool_output_for_rewrite(
            raw,
            max_chars=max_tool_output_chars,
        )

        rewrite_messages = [
            Message(
                role="system",
                content=self.instructions.load("tool_output_rewrite_system_prompt.md"),
            ),
            Message(
                role="user",
                content=self.instructions.render(
                    "tool_output_rewrite_user_prompt.md",
                    user_input=user_input,
                    tool_output=clipped,
                    clipped_note=clipped_note,
                ),
            ),
        ]

        self._set_runtime_status("thinking")
        try:
            response = await self._complete_with_guards(
                messages=rewrite_messages,
                tools=None,
                interaction_label="tool_output_rewrite",
                turn_usage=turn_usage,
            )
            friendly = (response.content or "").strip()
            if friendly:
                return friendly
        except Exception as e:
            log.warning("Friendly tool rewrite failed", error=str(e))

        return f"Tool executed:\n{raw}"

    def _extract_tool_calls_from_content(self, content: str) -> list[ToolCall]:
        """Extract tool calls from response content text.
        
        Looks for various formats:
        - @shell\ncommand: value
        - {tool => "shell", args => { --command "ls -la" }}
        - ```tool\ncommand\n```
        - <invoke name="shell"><command>value</command></invoke>
        """
        import re
        
        tool_calls = []
        if not content:
            return tool_calls
        
        # Pattern 1: @tool\ncommand: value
        pattern1 = r'@(\w+)\s*\n\s*command:\s*(.+?)(?:\n\n|\n\*|$)'
        
        # Pattern 2: {tool => "name", args => { --key "value" }}
        pattern2 = r'\{tool\s*=>\s*"([^"]+)"[^}]*args\s*=>\s*\{([^}]+)\}\}'
        
        # Pattern 3: ```tool\ncommand\n```
        pattern3 = r'```(\w+)\s*\n(.*?)\n```'
        
        # Pattern 4: <invoke name="shell"><command>value</command></invoke>
        pattern4 = r'<invoke\s+name="(\w+)">\s*<command>(.+?)</command>\s*</invoke>'
        
        all_patterns = [pattern1, pattern2, pattern3, pattern4]
        
        for pattern in all_patterns:
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                if pattern == pattern4:
                    # Pattern 4: <invoke name="..."><command>...</command></invoke>
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        tool_calls.append(ToolCall(
                            id=f"embedded_{len(tool_calls)}",
                            name=tool_name,
                            arguments={"command": command},
                        ))
                elif pattern == pattern1:
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        tool_calls.append(ToolCall(
                            id=f"embedded_{len(tool_calls)}",
                            name=tool_name,
                            arguments={"command": command},
                        ))
                elif pattern == pattern2:
                    tool_name = match.group(1).strip()
                    args_str = match.group(2).strip()
                    args = {}
                    arg_pattern = r'--(\w+)\s+"([^"]+)"'
                    for arg_match in re.finditer(arg_pattern, args_str):
                        key = arg_match.group(1)
                        value = arg_match.group(2)
                        args[key] = value
                    if tool_name and args:
                        tool_calls.append(ToolCall(
                            id=f"embedded_{len(tool_calls)}",
                            name=tool_name,
                            arguments=args,
                        ))
                elif pattern == pattern3:
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        tool_calls.append(ToolCall(
                            id=f"embedded_{len(tool_calls)}",
                            name=tool_name,
                            arguments={"command": command},
                        ))
        
        return tool_calls

    async def _handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        turn_usage: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Handle tool calls from LLM.
        
        Args:
            tool_calls: List of tool calls to execute
        
        Returns:
            List of tool results
        """
        results = []
        
        for tc in tool_calls:
            self._set_runtime_status("running script")
            log.info("Executing tool", tool=tc.name, call_id=tc.id)
            
            # Parse arguments (could be string or dict)
            arguments = tc.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            
            try:
                # Execute tool
                result = await self._execute_tool_with_guard(
                    name=tc.name,
                    arguments=arguments,
                    interaction_label=f"tool_call:{tc.name}",
                    turn_usage=turn_usage,
                )
                
                # Add result to session
                self._add_session_message(
                    role="tool",
                    content=result.content if result.success else f"Error: {result.error}",
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    result.content if result.success else f"Error: {result.error}",
                )
                
                results.append({
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "success": result.success,
                    "content": result.content if result.success else result.error,
                })
                
            except Exception as e:
                log.error("Tool execution failed", tool=tc.name, error=str(e))
                
                self._add_session_message(
                    role="tool",
                    content=f"Error: {str(e)}",
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    f"Error: {str(e)}",
                )
                
                results.append({
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "success": False,
                    "error": str(e),
                })
        
        self._set_runtime_status("thinking")
        return results

    async def complete(self, user_input: str) -> str:
        """Process user input and return response.
        
        Args:
            user_input: User's message
        
        Returns:
            Agent's response
        """
        if not self._initialized:
            await self.initialize()
        self._last_memory_debug_signature = None

        turn_usage = self._empty_usage()
        self.last_usage = self._empty_usage()
        planning_pipeline: dict[str, Any] | None = None
        recent_source_urls: list[str] = []
        effective_user_input = user_input
        effective_user_input, clarification_context_applied = self._resolve_effective_user_input(user_input)
        require_all_sources = self._request_references_all_sources(effective_user_input)
        use_contract_pipeline = self._should_use_contract_pipeline(
            effective_user_input,
            self.planning_enabled,
            pipeline_mode=self.pipeline_mode,
        )
        explicit_script_request = self._is_explicit_script_request(effective_user_input)
        enforce_python_worker_mode = explicit_script_request
        available_tools = {name.strip().lower() for name in self.tools.list_tools()}
        python_worker_tools_available = {"write", "shell"}.issubset(available_tools)
        python_worker_attempted = False
        list_task_plan: dict[str, Any] = {
            "enabled": False,
            "members": [],
            "strategy": "none",
            "per_member_action": "",
            "confidence": "low",
        }
        task_contract: dict[str, Any] | None = None
        completion_requirements: list[dict[str, Any]] = []
        completion_feedback: str = ""

        def finish(text: str, success: bool = True) -> str:
            if planning_pipeline is not None:
                self._finalize_pipeline(planning_pipeline, success=success)
            self._finalize_turn_usage(turn_usage)
            return text

        async def attempt_finalize_response(
            output_text: str,
            iteration: int,
            finish_success: bool = True,
        ) -> tuple[bool, str, bool]:
            """Apply auto-write + completion gate before returning final output."""
            nonlocal completion_feedback, python_worker_attempted, list_task_plan
            final_response = await self._maybe_auto_write_requested_output(
                user_input=effective_user_input,
                output_text=output_text,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
            )
            if enforce_python_worker_mode:
                worker_ran_this_iteration = False
                has_write = self._turn_has_successful_tool(turn_start_idx, "write")
                has_shell = self._turn_has_successful_tool(turn_start_idx, "shell")
                if not (has_write and has_shell) and not python_worker_attempted:
                    python_worker_attempted = True
                    worker_result = await self._run_python_worker_for_list_task(
                        user_input=effective_user_input,
                        turn_usage=turn_usage,
                        list_task_plan=list_task_plan,
                    )
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "python_worker_autorun",
                            "success": bool(worker_result.get("success", False)),
                            "attempted": True,
                        },
                        json.dumps(worker_result, ensure_ascii=True),
                    )
                    worker_ran_this_iteration = bool(worker_result.get("success", False))
                    has_write = self._turn_has_successful_tool(turn_start_idx, "write")
                    has_shell = self._turn_has_successful_tool(turn_start_idx, "shell")
                if worker_ran_this_iteration and iteration < (hard_turn_iterations - 1):
                    completion_feedback = (
                        "Python worker executed successfully.\n"
                        "Now provide the final answer covering the complete processed list and saved outputs."
                    )
                    return False, "", finish_success
                if not (has_write and has_shell):
                    completion_feedback = (
                        "Completion gate: execute Python worker workflow via tools before finalizing.\n"
                        "- Generate or refine a Python script/tool that handles the full item list.\n"
                        "- Run it through shell.\n"
                        "- Then provide final concise summary."
                    )
                    if iteration < (hard_turn_iterations - 1):
                        return False, "", finish_success
            if bool(list_task_plan.get("enabled", False)):
                members = list_task_plan.get("members")
                if isinstance(members, list) and members:
                    covered_members, missing_members = self._evaluate_list_member_coverage(
                        members=[str(member) for member in members],
                        candidate_response=final_response,
                        turn_start_idx=turn_start_idx,
                    )
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "list_member_coverage",
                            "covered": len(covered_members),
                            "missing": len(missing_members),
                            "members": len(members),
                        },
                        (
                            "step=list_member_coverage\n"
                            f"covered={len(covered_members)}\n"
                            f"missing={len(missing_members)}\n"
                            f"members={len(members)}"
                        ),
                    )
                    if missing_members:
                        completion_feedback = self._build_list_coverage_feedback(
                            missing_members=missing_members,
                            strategy=str(list_task_plan.get("strategy", "direct")).strip().lower(),
                            per_member_action=str(list_task_plan.get("per_member_action", "")).strip(),
                        )
                        self._emit_tool_output(
                            "task_contract",
                            {
                                "step": "list_member_retry",
                                "missing": len(missing_members),
                            },
                            completion_feedback,
                        )
                        if iteration < (hard_turn_iterations - 1):
                            return False, "", finish_success
            if completion_requirements and task_contract is not None:
                critique = await self._evaluate_contract_completion(
                    user_input=effective_user_input,
                    candidate_response=final_response,
                    contract=task_contract,
                    turn_usage=turn_usage,
                )
                checks_ok = bool(critique.get("complete", False))
                raw_check_results = critique.get("checks", [])
                check_results = raw_check_results if isinstance(raw_check_results, list) else []
                failed_items = [item for item in check_results if not bool(item.get("ok", False))]
                failure_reasons = ", ".join(
                    f"{item.get('id', '')}: {item.get('reason', '')}" for item in failed_items
                ) or "none"
                self._emit_tool_output(
                    "completion_gate",
                    {
                        "step": "validation",
                        "passed": checks_ok,
                        "failed_count": len(failed_items),
                    },
                    (
                        f"step=validation\n"
                        f"passed={checks_ok}\n"
                        f"failed_count={len(failed_items)}\n"
                        f"reasons={failure_reasons}"
                    ),
                )
                if planning_pipeline is not None:
                    task_order = self._refresh_pipeline_task_order(planning_pipeline)
                    if task_order:
                        self._set_pipeline_progress(
                            planning_pipeline,
                            current_index=len(task_order) - 1,
                            current_status="completed" if checks_ok else "in_progress",
                        )
                    self._update_pipeline_checks(planning_pipeline, check_results)
                    self._emit_pipeline_update(
                        "validation_passed" if checks_ok else "validation_retry",
                        planning_pipeline,
                    )
                if not checks_ok:
                    completion_feedback = self._build_completion_feedback(
                        contract=task_contract,
                        critique=critique,
                    )
                    self._emit_tool_output(
                        "task_contract",
                        {
                            "step": "validation_retry",
                            "missing": len(failed_items),
                        },
                        completion_feedback,
                    )
                    if iteration < (hard_turn_iterations - 1):
                        return False, "", finish_success

            self._update_clarification_state(
                user_input=user_input,
                effective_user_input=effective_user_input,
                assistant_response=final_response,
            )
            await self._persist_assistant_response(final_response)
            return True, final_response, finish_success

        turn_start_idx = len(self.session.messages) if self.session else 0
        recent_source_urls = self._collect_recent_source_urls(turn_start_idx)
        allowed_user_input, input_guard_error = await self._enforce_guard(
            guard_type="input",
            interaction_label="user_turn",
            content=user_input,
            turn_usage=turn_usage,
        )
        if not allowed_user_input:
            return finish(input_guard_error, success=False)

        # Add user message to session
        self._add_session_message("user", user_input)
        await self._auto_compact_if_needed()
        if clarification_context_applied:
            self._emit_tool_output(
                "task_contract",
                {"step": "clarification_context_applied"},
                "step=clarification_context_applied\nstatus=merged_pending_anchor_into_current_turn",
            )
        list_context_excerpt = self._collect_list_extraction_context()
        list_task_plan = await self._generate_list_task_plan(
            user_input=effective_user_input,
            context_excerpt=list_context_excerpt,
            turn_usage=turn_usage,
        )
        extracted_strategy = str(list_task_plan.get("strategy", "none")).strip().lower()
        if extracted_strategy == "script":
            enforce_python_worker_mode = True
        elif extracted_strategy == "direct" and not explicit_script_request:
            enforce_python_worker_mode = False
        if enforce_python_worker_mode and not python_worker_tools_available:
            enforce_python_worker_mode = False
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_skipped", "reason": "missing_tools"},
                "step=python_worker_mode_skipped\nreason=missing_tools\nrequired=write,shell",
            )
        if bool(list_task_plan.get("enabled", False)):
            self._emit_tool_output(
                "task_contract",
                {
                    "step": "list_task_memory_enabled",
                    "members": len(list_task_plan.get("members", [])),
                    "strategy": extracted_strategy,
                },
                (
                    "step=list_task_memory_enabled\n"
                    f"members={len(list_task_plan.get('members', []))}\n"
                    f"strategy={extracted_strategy}"
                ),
            )
        if enforce_python_worker_mode:
            self._emit_tool_output(
                "task_contract",
                {"step": "python_worker_mode_enabled", "strategy": extracted_strategy or "script"},
                "step=python_worker_mode_enabled\nmode=python_worker_tool_execution",
            )
        if use_contract_pipeline:
            task_contract = await self._generate_task_contract(
                user_input=effective_user_input,
                recent_source_urls=recent_source_urls,
                require_all_sources=require_all_sources,
                turn_usage=turn_usage,
                list_task_plan=list_task_plan,
            )
            completion_requirements = self._apply_list_requirements(
                base_requirements=list(task_contract.get("requirements", [])),
                list_task_plan=list_task_plan,
            )
            task_contract["requirements"] = completion_requirements
            prefetch_urls = [
                url
                for url in list(task_contract.get("prefetch_urls", []))
                if isinstance(url, str) and url.startswith(("http://", "https://"))
            ]
            if prefetch_urls:
                await self._run_source_report_prefetch(
                    source_urls=prefetch_urls,
                    turn_usage=turn_usage,
                    pipeline_label="task_contract",
                )
        if self.planning_enabled or task_contract is not None:
            planning_pipeline = self._build_task_pipeline(
                effective_user_input,
                tasks_override=(task_contract or {}).get("tasks"),
                completion_checks=completion_requirements,
            )
            if self.planning_enabled and task_contract is not None:
                planning_pipeline["mode"] = "manual_with_contract"
            elif self.planning_enabled:
                planning_pipeline["mode"] = "manual"
            else:
                planning_pipeline["mode"] = "auto_contract"
            self._emit_pipeline_update("created", planning_pipeline)
        
        # Send tool definitions so the model can issue structured tool calls.
        tool_defs = self.tools.get_definitions()
        log.debug("Tool definitions available", count=len(self.tools.list_tools()), tools_sent=bool(tool_defs))
        
        # Main agent loop
        base_turn_iterations = self.max_iterations + (2 if completion_requirements else 0)
        planned_turn_iterations = self._compute_turn_iteration_budget(
            base_iterations=base_turn_iterations,
            planning_pipeline=planning_pipeline,
            completion_requirements=completion_requirements,
        )
        hard_turn_iterations = max(planned_turn_iterations, min(320, planned_turn_iterations * 3))
        soft_turn_iterations = planned_turn_iterations
        extension_step = max(6, min(24, max(1, planned_turn_iterations // 3)))
        max_stagnant_iterations = 6
        stagnant_iterations = 0
        progress_window: list[bool] = []
        previous_progress_snapshot: dict[str, Any] | None = None
        last_completion_feedback_signature = ""
        if planned_turn_iterations != base_turn_iterations:
            self._emit_tool_output(
                "completion_gate",
                {
                    "step": "iteration_budget",
                    "base_limit": base_turn_iterations,
                    "effective_limit": planned_turn_iterations,
                    "hard_limit": hard_turn_iterations,
                },
                (
                    "step=iteration_budget\n"
                    f"base_limit={base_turn_iterations}\n"
                    f"effective_limit={planned_turn_iterations}\n"
                    f"hard_limit={hard_turn_iterations}"
                ),
            )
        for iteration in range(hard_turn_iterations):
            if iteration >= soft_turn_iterations:
                recent_progress = any(progress_window[-4:])
                remaining_work = (
                    self._pipeline_has_remaining_work(planning_pipeline)
                    or bool(completion_feedback)
                    or bool(completion_requirements)
                )
                if recent_progress and remaining_work and soft_turn_iterations < hard_turn_iterations:
                    previous_limit = soft_turn_iterations
                    soft_turn_iterations = min(hard_turn_iterations, soft_turn_iterations + extension_step)
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "iteration_budget_extended",
                            "previous_limit": previous_limit,
                            "new_limit": soft_turn_iterations,
                            "hard_limit": hard_turn_iterations,
                        },
                        (
                            "step=iteration_budget_extended\n"
                            f"previous_limit={previous_limit}\n"
                            f"new_limit={soft_turn_iterations}\n"
                            f"hard_limit={hard_turn_iterations}"
                        ),
                    )
                else:
                    self._emit_tool_output(
                        "completion_gate",
                        {
                            "step": "iteration_budget_exhausted",
                            "soft_limit": soft_turn_iterations,
                            "hard_limit": hard_turn_iterations,
                            "recent_progress": recent_progress,
                            "remaining_work": remaining_work,
                        },
                        (
                            "step=iteration_budget_exhausted\n"
                            f"soft_limit={soft_turn_iterations}\n"
                            f"hard_limit={hard_turn_iterations}\n"
                            f"recent_progress={recent_progress}\n"
                            f"remaining_work={remaining_work}"
                        ),
                    )
                    self._set_runtime_status("waiting")
                    return finish("Max iterations reached. Could not complete the request.", success=False)
            current_snapshot = self._capture_turn_progress_snapshot(turn_start_idx, planning_pipeline)
            if previous_progress_snapshot is not None:
                snapshot_progress = self._has_turn_progress(previous_progress_snapshot, current_snapshot)
                completion_feedback_signature = completion_feedback.strip()
                feedback_progress = bool(
                    completion_feedback_signature
                    and completion_feedback_signature != last_completion_feedback_signature
                )
                if feedback_progress:
                    last_completion_feedback_signature = completion_feedback_signature
                progressed = snapshot_progress or feedback_progress
                progress_window.append(progressed)
                if progressed:
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1
                    if (
                        stagnant_iterations >= max_stagnant_iterations
                        and iteration >= max(2, min(base_turn_iterations, soft_turn_iterations) // 2)
                    ):
                        self._emit_tool_output(
                            "completion_gate",
                            {
                                "step": "stuck_detected",
                                "stagnant_iterations": stagnant_iterations,
                                "iteration": iteration + 1,
                            },
                            (
                                "step=stuck_detected\n"
                                f"iteration={iteration + 1}\n"
                                f"stagnant_iterations={stagnant_iterations}"
                            ),
                        )
                        self._set_runtime_status("waiting")
                        return finish(
                            "Stopped after repeated non-progress iterations. Could not complete the request.",
                            success=False,
                        )
            else:
                progress_window.append(False)
            previous_progress_snapshot = current_snapshot
            self._set_runtime_status("thinking")
            # Build messages for LLM
            messages = self._build_messages(
                tool_messages_from_index=turn_start_idx,
                query=effective_user_input,
                planning_pipeline=planning_pipeline,
                list_task_plan=list_task_plan,
            )
            if completion_feedback:
                messages.append(
                    Message(
                        role="user",
                        content=completion_feedback,
                    )
                )
            
            # Call LLM
            log.info("Calling LLM", iteration=iteration + 1, message_count=len(messages))
            try:
                response = await self._complete_with_guards(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                    interaction_label=f"turn_{iteration + 1}",
                    turn_usage=turn_usage,
                )
            except GuardBlockedError as e:
                final = str(e)
                self._update_clarification_state(
                    user_input=user_input,
                    effective_user_input=effective_user_input,
                    assistant_response=final,
                )
                await self._persist_assistant_response(final)
                return finish(final, success=False)
            except Exception as e:
                # Check if this is a 500 error after tool execution
                error_str = str(e)
                tool_output = self._collect_turn_tool_output(turn_start_idx)
                
                # If we have tool messages AND got a 500 error, return tool output
                # This handles the case where Ollama can't process tool results in context
                if tool_output and "500" in error_str:
                    log.warning("Tool result call failed (500), returning tool output")
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                
                log.error("LLM call failed", error=str(e), exc_info=True)
                raise
            
            # Check for explicit tool calls (for models that support it)
            if response.tool_calls:
                log.info("Tool calls detected", count=len(response.tool_calls), calls=response.tool_calls)
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(response.tool_calls),
                )
                await self._handle_tool_calls(response.tool_calls, turn_usage=turn_usage)
                if planning_pipeline is not None:
                    self._advance_pipeline(planning_pipeline, event="tool_calls_completed")
                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                # Try to get final response
                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=effective_user_input,
                    planning_pipeline=planning_pipeline,
                    list_task_plan=list_task_plan,
                )
                try:
                    response = await self._complete_with_guards(
                        messages=messages,
                        tools=None,
                        interaction_label="tool_followup",
                        turn_usage=turn_usage,
                    )
                except GuardBlockedError as e:
                    final = str(e)
                    self._update_clarification_state(
                        user_input=user_input,
                        effective_user_input=effective_user_input,
                        assistant_response=final,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final, success=False)
                except Exception as e:
                    # Model doesn't support tool results - return tool output
                    log.warning("Model doesn't support tool results", error=str(e))
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                finalized, final_text, finish_success = await attempt_finalize_response(
                    output_text=response.content,
                    iteration=iteration,
                    finish_success=True,
                )
                if finalized:
                    return finish(final_text, success=finish_success)
                continue
            
            # Check for tool calls embedded in response text (fallback)
            # Looking for patterns like: {tool => "shell", args => {...}}
            embedded_calls = self._extract_tool_calls_from_content(response.content)
            if embedded_calls:
                log.info("Tool calls found in response text", count=len(embedded_calls))
                self._add_session_message(
                    role="assistant",
                    content=str(response.content or ""),
                    tool_calls=self._serialize_tool_calls_for_session(embedded_calls),
                )
                await self._handle_tool_calls(embedded_calls, turn_usage=turn_usage)
                if planning_pipeline is not None:
                    self._advance_pipeline(planning_pipeline, event="embedded_tool_calls_completed")
                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                # Try to get final response
                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=effective_user_input,
                    planning_pipeline=planning_pipeline,
                    list_task_plan=list_task_plan,
                )
                try:
                    response = await self._complete_with_guards(
                        messages=messages,
                        tools=None,
                        interaction_label="embedded_tool_followup",
                        turn_usage=turn_usage,
                    )
                except GuardBlockedError as e:
                    final = str(e)
                    self._update_clarification_state(
                        user_input=user_input,
                        effective_user_input=effective_user_input,
                        assistant_response=final,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final, success=False)
                except Exception as e:
                    log.warning("Model doesn't support tool results", error=str(e))
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                # If successful, return the response normally
                finalized, final_text, finish_success = await attempt_finalize_response(
                    output_text=response.content,
                    iteration=iteration,
                    finish_success=True,
                )
                if finalized:
                    return finish(final_text, success=finish_success)
                continue
            
            # Check for inline commands in response (fallback for models without tool calling)
            # This works by extracting commands from markdown code blocks in the response
            command = self._extract_command_from_response(response.content)
            if command:
                log.info("Executing inline command", command=command)
                try:
                    result = await self._execute_tool_with_guard(
                        name="shell",
                        arguments={"command": command},
                        interaction_label="inline_command",
                        turn_usage=turn_usage,
                    )
                    tool_result = result.content if result.success else f"Error: {result.error}"
                except Exception as e:
                    result = None
                    tool_result = f"Error: {str(e)}"
                
                # Add tool result to session
                self._add_session_message(
                    role="tool",
                    content=tool_result,
                    tool_name="shell",
                    tool_arguments={"command": command},
                )
                self._emit_tool_output("shell", {"command": command}, tool_result)
                if planning_pipeline is not None:
                    self._advance_pipeline(planning_pipeline, event="inline_command_completed")
                if not self._supports_tool_result_followup():
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                
                # Get final response
                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=effective_user_input,
                    planning_pipeline=planning_pipeline,
                    list_task_plan=list_task_plan,
                )
                try:
                    response = await self._complete_with_guards(
                        messages=messages,
                        tools=None,
                        interaction_label="inline_command_followup",
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=response.content,
                        iteration=iteration,
                        finish_success=True,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
                except Exception:
                    # Return tool output directly
                    final = await self._friendly_tool_output_response(
                        user_input=effective_user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    finalized, final_text, finish_success = await attempt_finalize_response(
                        output_text=final,
                        iteration=iteration,
                        finish_success=False,
                    )
                    if finalized:
                        return finish(final_text, success=finish_success)
                    continue
            
            # No tool calls - this is the final response
            finalized, final_text, finish_success = await attempt_finalize_response(
                output_text=response.content,
                iteration=iteration,
                finish_success=True,
            )
            if finalized:
                return finish(final_text, success=finish_success)
            continue
        
        # Hard iteration cap reached
        self._set_runtime_status("waiting")
        return finish("Max iterations reached. Could not complete the request.", success=False)

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """Stream response for user input.
        
        Args:
            user_input: User's message
        
        Yields:
            Response chunks
        """
        if not self._initialized:
            await self.initialize()
        self._last_memory_debug_signature = None
        self.last_usage = self._empty_usage()

        # Tool-calling and streaming over a single pass is currently limited.
        # Preserve tool behavior and guard checks by using complete() and
        # yielding chunked output when tools/guards are enabled.
        if self.tools.list_tools() or self.guards_enabled():
            self._set_runtime_status("thinking")
            content = await self.complete(user_input)
            chunk_size = 24
            self._set_runtime_status("streaming")
            for idx in range(0, len(content), chunk_size):
                yield content[idx : idx + chunk_size]
            self._set_runtime_status("waiting")
            return
        
        # Add user message to session
        self._add_session_message("user", user_input)
        await self._auto_compact_if_needed()
        planning_pipeline: dict[str, Any] | None = None
        if self.planning_enabled:
            planning_pipeline = self._build_task_pipeline(user_input)
            self._emit_pipeline_update("created", planning_pipeline)
        
        # Get tool definitions
        tool_defs = self.tools.get_definitions()
        
        # For streaming, we currently don't support tool calling
        # This is a limitation - full streaming with tools needs more work
        messages = self._build_messages(query=user_input, planning_pipeline=planning_pipeline)
        
        # Stream the response
        full_content = ""
        self._set_runtime_status("streaming")
        async for chunk in self.provider.complete_streaming(
            messages=messages,
            tools=tool_defs if tool_defs else None,
        ):
            full_content += chunk
            yield chunk
        
        # Add assistant response to session
        if self.session:
            self._add_session_message("assistant", full_content)
            await self.session_manager.save_session(self.session)

        if planning_pipeline is not None:
            self._finalize_pipeline(planning_pipeline, success=True)

        prompt_tokens = sum(self._count_tokens(m.content) for m in messages)
        completion_tokens = self._count_tokens(full_content)
        self._finalize_turn_usage({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })
        self._set_runtime_status("waiting")
