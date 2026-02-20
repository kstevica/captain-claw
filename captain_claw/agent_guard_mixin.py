"""Guard policy and guarded LLM completion helpers for Agent."""

import asyncio
import json
import re
from typing import Any

from captain_claw.config import get_config
from captain_claw.exceptions import GuardBlockedError
from captain_claw.llm import Message
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentGuardMixin:
    """Guard evaluation and guarded completion/tool execution."""
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
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
        abort_event: asyncio.Event | None = None,
        session_id_override: str | None = None,
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
            session_id=str(session_id_override or "").strip() or self._current_session_slug(),
            session_policy=session_policy,
            task_policy=task_policy,
            abort_event=abort_event,
            runtime_base_path=getattr(self, "workspace_base_path", None),
            approval_callback=getattr(self, "approval_callback", None),
            file_registry=getattr(self, "_file_registry", None),
        )
