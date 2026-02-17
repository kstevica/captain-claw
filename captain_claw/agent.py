"""Agent orchestration for Captain Claw."""

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any, AsyncIterator
from typing import Callable

from captain_claw.config import get_config
from captain_claw.instructions import InstructionLoader
from captain_claw.llm import (
    LLMProvider,
    Message,
    ToolCall,
    ToolDefinition,
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
    ):
        """Initialize the agent.
        
        Args:
            provider: Optional LLM provider override
            status_callback: Optional runtime status callback
        """
        self.provider = provider
        self.status_callback = status_callback
        self.tool_output_callback = tool_output_callback
        self.tools = get_tool_registry()
        self.runtime_base_path = Path.cwd().resolve()
        self.tools.set_runtime_base_path(self.runtime_base_path)
        self.session_manager = get_session_manager()
        self.session: Session | None = None
        self._initialized = False
        self.max_iterations = 10  # Max tool calls per message
        self.last_usage: dict[str, int] = self._empty_usage()
        self.total_usage: dict[str, int] = self._empty_usage()
        self.last_context_window: dict[str, int | float] = {}
        self._last_memory_debug_signature: str | None = None
        self.planning_enabled: bool = False
        self.instructions = InstructionLoader()

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        """Create an empty usage bucket."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
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
        if not self.tool_output_callback:
            return
        try:
            self.tool_output_callback(tool_name, arguments, output)
        except Exception:
            pass

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
            response = await self.provider.complete(
                messages=rewrite_messages,
                tools=None,
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
            response = await self.provider.complete(
                messages=messages,
                tools=None,
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
        enabled = False
        if self.session and isinstance(self.session.metadata, dict):
            planning_meta = self.session.metadata.get("planning")
            if isinstance(planning_meta, dict):
                enabled = bool(planning_meta.get("enabled", False))
        self.planning_enabled = enabled

    def refresh_session_runtime_flags(self) -> None:
        """Public helper to reload runtime flags after session switch."""
        self._sync_runtime_flags_from_session()

    async def set_planning_mode(self, enabled: bool, persist: bool = True) -> None:
        """Enable or disable planning mode for current session/runtime."""
        self.planning_enabled = bool(enabled)
        if not self.session:
            return
        planning_meta = self.session.metadata.setdefault("planning", {})
        planning_meta["enabled"] = self.planning_enabled
        planning_meta["updated_at"] = datetime.now(UTC).isoformat()
        if persist:
            await self.session_manager.save_session(self.session)

    def _build_task_pipeline(self, user_input: str, max_tasks: int = 6) -> dict[str, Any]:
        """Build a lightweight task pipeline from user input."""
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        parts = [
            piece.strip(" -")
            for piece in re.split(
                r"(?:\n+|;|\. |\band then\b|\bthen\b|\bnext\b)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if piece.strip(" -")
        ]
        tasks: list[dict[str, Any]] = []
        for idx, part in enumerate(parts[:max_tasks], start=1):
            tasks.append({
                "id": f"task_{idx}",
                "title": part[:180],
                "status": "pending",
            })

        if not tasks:
            tasks = [
                {"id": "task_1", "title": "Understand the request and constraints", "status": "pending"},
                {"id": "task_2", "title": "Execute required actions/tools", "status": "pending"},
                {"id": "task_3", "title": "Return concise final answer", "status": "pending"},
            ]

        pipeline = {
            "created_at": datetime.now(UTC).isoformat(),
            "request": cleaned[:500],
            "tasks": tasks,
            "current_index": 0,
            "state": "active",
        }
        self._set_pipeline_progress(pipeline, current_index=0, current_status="in_progress")
        return pipeline

    @staticmethod
    def _set_pipeline_progress(
        pipeline: dict[str, Any],
        current_index: int,
        current_status: str = "in_progress",
    ) -> None:
        """Set pipeline task statuses around current index."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return
        bounded = max(0, min(current_index, len(tasks) - 1))
        pipeline["current_index"] = bounded
        for idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            if idx < bounded:
                task["status"] = "completed"
            elif idx == bounded:
                task["status"] = current_status
            else:
                task["status"] = "pending"

    def _build_pipeline_note(self, pipeline: dict[str, Any]) -> str:
        """Build planning note injected into model context."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return ""
        lines = [self.instructions.load("planning_pipeline_header.md")]
        for idx, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                continue
            title = str(task.get("title", "")).strip()
            status = str(task.get("status", "pending")).strip().upper()
            lines.append(f"{idx}. [{status}] {title}")
        lines.append(self.instructions.load("planning_pipeline_footer.md"))
        return "\n".join(lines)

    @staticmethod
    def _format_pipeline_monitor_output(event: str, pipeline: dict[str, Any]) -> str:
        """Format pipeline state for monitor output."""
        lines = [f"Planning event={event}"]
        lines.append(f"state={pipeline.get('state', 'active')}")
        tasks = pipeline.get("tasks", [])
        for idx, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                continue
            lines.append(
                f"- {idx}. status={task.get('status', 'pending')} title={task.get('title', '')}"
            )
        return "\n".join(lines)

    def _emit_pipeline_update(self, event: str, pipeline: dict[str, Any]) -> None:
        """Emit planning pipeline state into monitor output stream."""
        self._emit_tool_output(
            "planning",
            {
                "event": event,
                "enabled": self.planning_enabled,
                "current_index": int(pipeline.get("current_index", 0)),
            },
            self._format_pipeline_monitor_output(event, pipeline),
        )

    def _advance_pipeline(self, pipeline: dict[str, Any], event: str = "advance") -> None:
        """Advance pipeline to next task if possible."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list) or not tasks:
            return
        current_index = int(pipeline.get("current_index", 0))
        if current_index >= len(tasks) - 1:
            self._set_pipeline_progress(pipeline, current_index=current_index, current_status="in_progress")
            self._emit_pipeline_update(event, pipeline)
            return
        self._set_pipeline_progress(pipeline, current_index=current_index + 1, current_status="in_progress")
        self._emit_pipeline_update(event, pipeline)

    def _finalize_pipeline(self, pipeline: dict[str, Any], success: bool = True) -> None:
        """Finalize pipeline statuses at turn completion."""
        tasks = pipeline.get("tasks", [])
        if not isinstance(tasks, list):
            return
        if success:
            for task in tasks:
                if isinstance(task, dict):
                    task["status"] = "completed"
            pipeline["state"] = "completed"
            self._emit_pipeline_update("completed", pipeline)
            return

        current_index = int(pipeline.get("current_index", 0))
        if 0 <= current_index < len(tasks) and isinstance(tasks[current_index], dict):
            tasks[current_index]["status"] = "failed"
        pipeline["state"] = "failed"
        self._emit_pipeline_update("failed", pipeline)

    def _add_session_message(
        self,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
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
        
        # Load or create session
        self.session = await self.session_manager.get_or_create_session()
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

    def _build_system_prompt(self) -> str:
        """Build the system prompt."""
        session_name = self._current_session_slug()

        planning_block = ""
        if self.planning_enabled:
            planning_block = (
                "\n\n" + self.instructions.load("planning_mode_instructions.md")
            )

        saved_root = self.tools.get_saved_base_path(create=False)

        return self.instructions.render(
            "system_prompt.md",
            runtime_base_path=self.runtime_base_path,
            saved_root=saved_root,
            session_name=session_name,
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

    def _build_messages(
        self,
        tool_messages_from_index: int | None = None,
        query: str | None = None,
        planning_pipeline: dict[str, Any] | None = None,
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
        for msg in selected_messages:
            messages.append(
                Message(
                    role=msg["role"],
                    content=msg["content"],
                    tool_call_id=msg.get("tool_call_id"),
                    tool_name=msg.get("tool_name"),
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
        cfg = get_config()
        provider = (cfg.model.provider or "").lower()
        model = (cfg.model.model or "").lower()

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

    def _current_session_slug(self) -> str:
        """Return normalized session name used for folder scoping."""
        session_name = "default"
        if self.session and self.session.name:
            raw_name = str(self.session.name).strip()
            normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_name).strip("-")
            if normalized:
                session_name = normalized
        return session_name

    def _build_script_relative_path(
        self,
        user_input: str,
        extension: str,
    ) -> str:
        """Build script path under scripts/<session>/."""
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
            response = await self.provider.complete(messages=synth_messages, tools=None)
            self._accumulate_usage(turn_usage, response.usage or {})
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
            write_args = {"path": script_rel_path, "content": code}
            write_args_for_log = {"path": script_rel_path, "content_chars": len(code), "append": False}
            try:
                write_result = await self.tools.execute(name="write", arguments=write_args)
                write_output = (
                    write_result.content if write_result.success else f"Error: {write_result.error}"
                )
            except Exception as e:
                write_result = None
                write_output = f"Error: {str(e)}"

            self._add_session_message(
                role="tool",
                content=write_output,
                tool_name="write",
                tool_arguments=write_args_for_log,
            )
            self._emit_tool_output("write", write_args_for_log, write_output)

            if not (write_result and write_result.success):
                return (
                    f"{text}\n\n"
                    "Note: explicit script request could not be completed because write failed.\n"
                    f"{write_output}"
                ).strip()

            parsed_written = self._parse_written_path_from_tool_output(write_output)
            written_script_path = parsed_written or (
                self.tools.get_saved_base_path(create=True) / script_rel_path
            )
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
            shell_result = await self.tools.execute(name="shell", arguments={"command": run_command})
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
    def _extract_requested_write_path(user_input: str) -> str | None:
        """Extract requested target file path from user input."""
        text = (user_input or "").strip()
        if not text:
            return None
        if not re.search(r"\b(write|save|store|export|dump)\b", text, flags=re.IGNORECASE):
            return None

        patterns = [
            r"(?:to|into|as)\s+(?:a\s+)?(?:file\s+)?[`\"']?([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,16})",
            r"(?:file\s+|named\s+|name\s+it\s+)[`\"']?([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,16})",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            if not matches:
                continue
            candidate = matches[-1].strip().rstrip(".,;:!?)]}>")
            lowered = candidate.lower()
            if lowered.startswith(("http://", "https://")):
                continue
            if candidate:
                return candidate
        return None

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
        requested_path = self._extract_requested_write_path(user_input)
        if not requested_path:
            return text
        if self._turn_has_successful_tool(turn_start_idx, "write"):
            return text

        write_args = {"path": requested_path, "content": text}
        write_args_for_log = {"path": requested_path, "content_chars": len(text), "append": False}

        try:
            result = await self.tools.execute(name="write", arguments=write_args)
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

        if result and result.success:
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

    async def _friendly_tool_output_response(
        self,
        user_input: str,
        tool_output: str,
        turn_usage: dict[str, int],
    ) -> str:
        """Ask model to rewrite raw tool output into a user-friendly answer."""
        raw = (tool_output or "").strip() or "[no output]"
        max_tool_output_chars = 12000
        clipped = raw[:max_tool_output_chars]
        clipped_note = ""
        if len(raw) > max_tool_output_chars:
            clipped_note = (
                "\n\n[Tool output truncated before rewrite due to size limits.]"
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
            response = await self.provider.complete(messages=rewrite_messages, tools=None)
            self._accumulate_usage(turn_usage, response.usage or {})
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

    async def _handle_tool_calls(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
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
                result = await self.tools.execute(
                    name=tc.name,
                    arguments=arguments,
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

        def finish(text: str, success: bool = True) -> str:
            if planning_pipeline is not None:
                self._finalize_pipeline(planning_pipeline, success=success)
            self._finalize_turn_usage(turn_usage)
            return text

        turn_start_idx = len(self.session.messages) if self.session else 0

        # Add user message to session
        self._add_session_message("user", user_input)
        await self._auto_compact_if_needed()
        if self.planning_enabled:
            planning_pipeline = self._build_task_pipeline(user_input)
            self._emit_pipeline_update("created", planning_pipeline)
        
        # Send tool definitions so the model can issue structured tool calls.
        tool_defs = self.tools.get_definitions()
        log.debug("Tool definitions available", count=len(self.tools.list_tools()), tools_sent=bool(tool_defs))
        
        # Main agent loop
        for iteration in range(self.max_iterations):
            self._set_runtime_status("thinking")
            # Build messages for LLM
            messages = self._build_messages(
                tool_messages_from_index=turn_start_idx,
                query=user_input,
                planning_pipeline=planning_pipeline,
            )
            
            # Call LLM
            log.info("Calling LLM", iteration=iteration + 1, message_count=len(messages))
            try:
                response = await self.provider.complete(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                )
                self._accumulate_usage(turn_usage, response.usage or {})
            except Exception as e:
                # Check if this is a 500 error after tool execution
                error_str = str(e)
                tool_output = self._collect_turn_tool_output(turn_start_idx)
                
                # If we have tool messages AND got a 500 error, return tool output
                # This handles the case where Ollama can't process tool results in context
                if tool_output and "500" in error_str:
                    log.warning("Tool result call failed (500), returning tool output")
                    final = await self._friendly_tool_output_response(
                        user_input=user_input,
                        tool_output=tool_output,
                        turn_usage=turn_usage,
                    )
                    final = await self._maybe_auto_write_requested_output(
                        user_input=user_input,
                        output_text=final,
                        turn_start_idx=turn_start_idx,
                        turn_usage=turn_usage,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final, success=False)
                
                log.error("LLM call failed", error=str(e), exc_info=True)
                raise
            
            # Check for explicit tool calls (for models that support it)
            if response.tool_calls:
                log.info("Tool calls detected", count=len(response.tool_calls), calls=response.tool_calls)
                await self._handle_tool_calls(response.tool_calls)
                if planning_pipeline is not None:
                    self._advance_pipeline(planning_pipeline, event="tool_calls_completed")
                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    final = await self._maybe_auto_write_requested_output(
                        user_input=user_input,
                        output_text=final,
                        turn_start_idx=turn_start_idx,
                        turn_usage=turn_usage,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final)
                # Try to get final response
                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=user_input,
                    planning_pipeline=planning_pipeline,
                )
                try:
                    response = await self.provider.complete(messages=messages, tools=None)
                    self._accumulate_usage(turn_usage, response.usage or {})
                except Exception as e:
                    # Model doesn't support tool results - return tool output
                    log.warning("Model doesn't support tool results", error=str(e))
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    final = await self._maybe_auto_write_requested_output(
                        user_input=user_input,
                        output_text=final,
                        turn_start_idx=turn_start_idx,
                        turn_usage=turn_usage,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final, success=False)
                continue
            
            # Check for tool calls embedded in response text (fallback)
            # Looking for patterns like: {tool => "shell", args => {...}}
            embedded_calls = self._extract_tool_calls_from_content(response.content)
            if embedded_calls:
                log.info("Tool calls found in response text", count=len(embedded_calls))
                await self._handle_tool_calls(embedded_calls)
                if planning_pipeline is not None:
                    self._advance_pipeline(planning_pipeline, event="embedded_tool_calls_completed")
                if not self._supports_tool_result_followup():
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    final = await self._maybe_auto_write_requested_output(
                        user_input=user_input,
                        output_text=final,
                        turn_start_idx=turn_start_idx,
                        turn_usage=turn_usage,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final)
                # Try to get final response
                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=user_input,
                    planning_pipeline=planning_pipeline,
                )
                try:
                    response = await self.provider.complete(messages=messages, tools=None)
                    self._accumulate_usage(turn_usage, response.usage or {})
                except Exception as e:
                    log.warning("Model doesn't support tool results", error=str(e))
                    output = self._collect_turn_tool_output(turn_start_idx)
                    final = await self._friendly_tool_output_response(
                        user_input=user_input,
                        tool_output=output,
                        turn_usage=turn_usage,
                    )
                    final = await self._maybe_auto_write_requested_output(
                        user_input=user_input,
                        output_text=final,
                        turn_start_idx=turn_start_idx,
                        turn_usage=turn_usage,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final, success=False)
                # If successful, return the response normally
                final_response = await self._maybe_auto_write_requested_output(
                    user_input=user_input,
                    output_text=response.content,
                    turn_start_idx=turn_start_idx,
                    turn_usage=turn_usage,
                )
                await self._persist_assistant_response(final_response)
                return finish(final_response)
            
            # Check for inline commands in response (fallback for models without tool calling)
            # This works by extracting commands from markdown code blocks in the response
            command = self._extract_command_from_response(response.content)
            if command:
                log.info("Executing inline command", command=command)
                result = await self.tools.execute(name="shell", arguments={"command": command})
                tool_result = result.content if result.success else f"Error: {result.error}"
                
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
                        user_input=user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    final = await self._maybe_auto_write_requested_output(
                        user_input=user_input,
                        output_text=final,
                        turn_start_idx=turn_start_idx,
                        turn_usage=turn_usage,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final)
                
                # Get final response
                messages = self._build_messages(
                    tool_messages_from_index=turn_start_idx,
                    query=user_input,
                    planning_pipeline=planning_pipeline,
                )
                try:
                    response = await self.provider.complete(messages=messages, tools=None)
                    self._accumulate_usage(turn_usage, response.usage or {})
                # If successful, continue to process the response
                    continue
                except:
                    # Return tool output directly
                    final = await self._friendly_tool_output_response(
                        user_input=user_input,
                        tool_output=tool_result,
                        turn_usage=turn_usage,
                    )
                    final = await self._maybe_auto_write_requested_output(
                        user_input=user_input,
                        output_text=final,
                        turn_start_idx=turn_start_idx,
                        turn_usage=turn_usage,
                    )
                    await self._persist_assistant_response(final)
                    return finish(final, success=False)
            
            # No tool calls - this is the final response
            final_response = await self._maybe_auto_write_requested_output(
                user_input=user_input,
                output_text=response.content,
                turn_start_idx=turn_start_idx,
                turn_usage=turn_usage,
            )
            
            # Add assistant response to session
            if self.session:
                self._add_session_message("assistant", final_response)
                # Save session after each turn
                await self.session_manager.save_session(self.session)

            return finish(final_response)
        
        # Max iterations reached
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
        # Preserve tool behavior by using complete() and yielding chunked output.
        if self.tools.list_tools():
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
