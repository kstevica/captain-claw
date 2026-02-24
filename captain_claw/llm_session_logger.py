"""File-based LLM session logger.

Writes every LLM call to ``logs/<session_slug>/session_log.md`` with:
- Timestamp, interaction label, model name
- Instruction files referenced in messages
- Full content of every message (system, user, assistant, tool)
- Full LLM response text and tool calls
- Token usage
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)


class LLMSessionLogger:
    """Append-only markdown logger for LLM interactions within a session."""

    def __init__(self, logs_dir: Path | str):
        self._logs_dir = Path(logs_dir).resolve()
        self._session_slug: str = ""
        self._log_path: Path | None = None
        self._call_counter: int = 0

    def set_session(self, session_slug: str) -> None:
        """Set or change the active session for logging."""
        slug = (session_slug or "default").strip() or "default"
        if slug == self._session_slug and self._log_path is not None:
            return
        self._session_slug = slug
        session_dir = self._logs_dir / slug
        session_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = session_dir / "session_log.md"
        self._call_counter = 0

    def log_call(
        self,
        interaction_label: str,
        model: str,
        messages: list[Any],
        response: Any,
        instruction_files: list[str] | None = None,
        tools_enabled: bool = False,
        max_tokens: int | None = None,
    ) -> None:
        """Append one LLM call entry to the session log."""
        if self._log_path is None:
            return
        try:
            self._call_counter += 1
            entry = self._format_entry(
                call_number=self._call_counter,
                interaction_label=interaction_label,
                model=model,
                messages=messages,
                response=response,
                instruction_files=instruction_files or [],
                tools_enabled=tools_enabled,
                max_tokens=max_tokens,
            )
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            log.warning("LLM session log write failed", error=str(e))

    def _format_entry(
        self,
        call_number: int,
        interaction_label: str,
        model: str,
        messages: list[Any],
        response: Any,
        instruction_files: list[str],
        tools_enabled: bool,
        max_tokens: int | None,
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines: list[str] = []

        lines.append(f"## Call #{call_number} — {interaction_label}")
        lines.append("")
        lines.append(f"**Time:** {now}  ")
        lines.append(f"**Model:** `{model or '(unknown)'}`  ")
        if max_tokens is not None:
            lines.append(f"**Max tokens:** {max_tokens}  ")
        lines.append(f"**Tools enabled:** {tools_enabled}  ")

        # Instruction files
        if instruction_files:
            lines.append(f"**Instruction files:** {', '.join(f'`{f}`' for f in instruction_files)}  ")

        # Full message contents
        lines.append("")
        lines.append("### Messages In")
        lines.append("")
        lines.append(f"**Count:** {len(messages)}  ")
        lines.append("")
        for idx, msg in enumerate(messages, start=1):
            if isinstance(msg, dict):
                role = str(msg.get("role", "unknown"))
                msg_content = str(msg.get("content", ""))
                tool_name = msg.get("tool_name") or ""
                tool_call_id = msg.get("tool_call_id") or ""
            else:
                role = str(getattr(msg, "role", "unknown"))
                msg_content = str(getattr(msg, "content", ""))
                tool_name = str(getattr(msg, "tool_name", "") or "")
                tool_call_id = str(getattr(msg, "tool_call_id", "") or "")

            header = f"#### Message {idx} — `{role}`"
            if tool_name:
                header += f" (tool: `{tool_name}`)"
            if tool_call_id:
                header += f" (call_id: `{tool_call_id}`)"
            lines.append(header)
            lines.append("")
            if msg_content:
                lines.append("```")
                lines.append(msg_content)
                lines.append("```")
            else:
                lines.append("*(empty)*")
            lines.append("")

        # Response
        lines.append("### Response")
        lines.append("")

        content = str(getattr(response, "content", "") or "")
        resp_model = str(getattr(response, "model", "") or model or "")
        usage = getattr(response, "usage", {}) or {}
        finish = str(getattr(response, "finish_reason", "") or "")
        tool_calls = list(getattr(response, "tool_calls", []) or [])

        if resp_model:
            lines.append(f"**Response model:** `{resp_model}`  ")
        if finish:
            lines.append(f"**Finish reason:** {finish}  ")

        # Usage
        if isinstance(usage, dict) and any(usage.values()):
            prompt_t = usage.get("prompt_tokens", 0)
            comp_t = usage.get("completion_tokens", 0)
            total_t = usage.get("total_tokens", 0)
            lines.append(f"**Tokens:** prompt={prompt_t}, completion={comp_t}, total={total_t}  ")

        lines.append("")

        # Content
        if content:
            lines.append("**Content:**")
            lines.append("")
            lines.append("```")
            # Truncate very long responses for readability
            if len(content) > 8000:
                lines.append(content[:8000])
                lines.append(f"... [truncated, {len(content)} chars total]")
            else:
                lines.append(content)
            lines.append("```")
            lines.append("")

        # Tool calls
        if tool_calls:
            lines.append(f"**Tool calls ({len(tool_calls)}):**")
            lines.append("")
            for tc in tool_calls:
                tc_name = str(getattr(tc, "name", "") or "")
                tc_args = getattr(tc, "arguments", {}) or {}
                lines.append(f"- `{tc_name}`")
                if tc_args:
                    args_str = json.dumps(tc_args, ensure_ascii=False, indent=2)
                    if len(args_str) > 2000:
                        args_str = args_str[:2000] + "\n... [truncated]"
                    lines.append("  ```json")
                    for arg_line in args_str.split("\n"):
                        lines.append(f"  {arg_line}")
                    lines.append("  ```")
            lines.append("")

        lines.append("---")
        lines.append("")
        return "\n".join(lines)


# ── Module-level singleton ──────────────────────────────────────────

_logger: LLMSessionLogger | None = None


def get_llm_session_logger(logs_dir: Path | str | None = None) -> LLMSessionLogger:
    """Get or create the global LLM session logger."""
    global _logger
    if _logger is None:
        if logs_dir is None:
            logs_dir = Path.cwd().resolve() / "logs"
        _logger = LLMSessionLogger(logs_dir)
    return _logger


def reset_llm_session_logger() -> None:
    """Reset the global logger (for testing)."""
    global _logger
    _logger = None
