"""Session memory/compaction/runtime-flag helpers for Agent."""

import copy
import re
from datetime import UTC, datetime
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message
from captain_claw.logging import get_logger
from captain_claw.session import Session

log = get_logger(__name__)


class AgentSessionMixin:
    """Session token accounting, compaction, and runtime flag sync."""
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
        """Ensure persisted token_count exists on a session message.

        Includes tool_calls arguments in the count because they are sent
        to the LLM and consume prompt tokens.  Without this, assistant
        messages with large write-file payloads are severely undercounted
        and the context budget system cannot prune them.
        """
        has_tool_calls = bool(msg.get("tool_calls"))
        value = msg.get("token_count")
        # Cached value is only reliable when tool_calls have been
        # accounted for (flagged by ``_tc_counted``).  Legacy messages
        # stored before this fix have content-only counts.
        tc_counted = msg.get("_tc_counted", False)
        if isinstance(value, int) and value >= 0 and (tc_counted or not has_tool_calls):
            return value
        text = str(msg.get("content", ""))
        if has_tool_calls:
            for tc in msg["tool_calls"]:
                args_str = (tc.get("function") or {}).get("arguments", "")
                if args_str:
                    text += args_str
            msg["_tc_counted"] = True
        count = self._count_tokens(text)
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

    async def ensure_pipeline_subagent_contexts(
        self,
        pipeline: dict[str, Any] | None,
        *,
        task_ids: list[str] | None = None,
    ) -> list[str]:
        """Ensure active/selected tasks have isolated subagent session contexts."""
        if not isinstance(pipeline, dict):
            return []
        self._refresh_pipeline_task_order(pipeline)
        graph = pipeline.get("task_graph")
        if not isinstance(graph, dict) or not graph:
            return []

        subagent_meta = pipeline.setdefault("subagents", {})
        if not isinstance(subagent_meta, dict):
            pipeline["subagents"] = {}
            subagent_meta = pipeline["subagents"]
        enabled = bool(subagent_meta.get("enabled", True))
        if not enabled:
            return []

        max_spawn_depth = max(0, int(subagent_meta.get("max_spawn_depth", 2)))
        max_active_children = max(1, int(subagent_meta.get("max_active_children", 5)))
        allow_agents = subagent_meta.get("allow_agents")
        if not isinstance(allow_agents, list) or not allow_agents:
            allow_agents = ["*"]
            subagent_meta["allow_agents"] = allow_agents
        active_child_ids = [
            str(item).strip()
            for item in subagent_meta.get("active_child_session_ids", [])
            if str(item).strip()
        ]
        subagent_meta["active_child_session_ids"] = active_child_ids

        targets: list[str] = []
        for task_id in task_ids or list(pipeline.get("active_task_ids", [])):
            cleaned = str(task_id).strip()
            if cleaned and cleaned not in targets:
                targets.append(cleaned)
        if not targets:
            return []

        parent_session_id = str(self.session.id) if self.session else ""
        parent_spawn_depth = 0
        if self.session and isinstance(self.session.metadata, dict):
            session_subagent = self.session.metadata.get("subagent")
            if isinstance(session_subagent, dict):
                parent_spawn_depth = max(0, int(session_subagent.get("spawn_depth", 0)))

        create_session = getattr(self.session_manager, "create_session", None)
        created_child_ids: list[str] = []
        for task_id in targets:
            node = graph.get(task_id)
            if not isinstance(node, dict):
                continue
            execution_context = node.get("execution_context")
            if not isinstance(execution_context, dict):
                execution_context = {}
                node["execution_context"] = execution_context
            execution_context.setdefault("context_id", f"context_{task_id}")
            execution_context.setdefault("spawned_by", parent_session_id)
            execution_context.setdefault("parent_task_id", "")
            execution_context.setdefault("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            execution_context.setdefault("compaction_count", 0)
            execution_context.setdefault("history", [])
            execution_context.setdefault("artifacts", [])
            execution_context.setdefault("variables", {})
            execution_context.setdefault("allow_agents", allow_agents)
            execution_context.setdefault("max_children", max_active_children)
            execution_context.setdefault("tool_allowlist", ["shell", "write", "read"])
            execution_context.setdefault("timeout_seconds", 120)

            existing_session_id = str(execution_context.get("session_id", "")).strip()
            if existing_session_id:
                continue

            spawn_depth = max(1, int(execution_context.get("spawn_depth", parent_spawn_depth + 1)))
            execution_context["spawn_depth"] = spawn_depth
            if spawn_depth > max_spawn_depth:
                execution_context["spawn_state"] = "depth_limited"
                continue
            if len(active_child_ids) >= max_active_children:
                execution_context["spawn_state"] = "active_children_capped"
                continue
            if not callable(create_session):
                execution_context["session_id"] = f"virtual:{task_id}"
                execution_context["spawn_state"] = "virtual"
                continue

            child_name_base = self.session.name if self.session else "session"
            child_name = f"{child_name_base} :: task {task_id}"
            child_metadata = {
                "subagent": {
                    "spawned_by": parent_session_id,
                    "task_id": task_id,
                    "spawn_depth": spawn_depth,
                    "allow_agents": allow_agents,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            }
            try:
                child = await create_session(name=child_name[:120], metadata=child_metadata)
                child_id = str(child.id).strip()
                if not child_id:
                    execution_context["spawn_state"] = "spawn_failed"
                    execution_context["spawn_error"] = "missing_child_id"
                    continue
                execution_context["session_id"] = child_id
                execution_context["spawn_state"] = "spawned"
                active_child_ids.append(child_id)
                created_child_ids.append(child_id)
            except Exception as e:
                execution_context["spawn_state"] = "spawn_failed"
                execution_context["spawn_error"] = str(e)[:240]

        subagent_meta["active_child_session_ids"] = active_child_ids
        return created_child_ids

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
        llm_session_logging = bool(getattr(cfg.logging, "llm_session_logging", False))
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
            if isinstance(monitor_meta, dict) and "llm_session_logging" in monitor_meta:
                llm_session_logging = bool(monitor_meta.get("llm_session_logging", False))
        self.pipeline_mode = pipeline_mode
        self.planning_enabled = self.pipeline_mode == "contracts"
        self.monitor_trace_llm = monitor_trace_llm
        self.monitor_trace_pipeline = monitor_trace_pipeline
        self.llm_session_logging = llm_session_logging
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.set_active_session(self.session.id if self.session else None)
        self._skills_snapshot_cache = None
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

    async def set_llm_session_logging(self, enabled: bool, persist: bool = True) -> None:
        """Enable or disable file-based LLM session logging."""
        self.llm_session_logging = bool(enabled)
        if not self.session:
            return
        monitor_meta = self.session.metadata.setdefault("monitor", {})
        monitor_meta["llm_session_logging"] = self.llm_session_logging
        monitor_meta["updated_at"] = datetime.now(UTC).isoformat()
        if persist:
            await self.session_manager.save_session(self.session)

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
        # Attach model label for assistant messages so the UI can display it.
        model_label = ""
        if role == "assistant":
            details = getattr(self, "_runtime_model_details", None) or {}
            provider = details.get("provider", "")
            model = details.get("model", "")
            if provider and model:
                model_label = f"{provider}:{model}"
        # Compute token count including tool_calls arguments — these
        # are sent to the LLM and must be budgeted accurately.
        _token_text = content
        if tool_calls:
            for _tc in tool_calls:
                _args_str = (_tc.get("function") or {}).get("arguments", "")
                if _args_str:
                    _token_text += _args_str
        self.session.add_message(
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_calls=tool_calls,
            tool_arguments=tool_arguments,
            token_count=self._count_tokens(_token_text),
            model=model_label,
        )
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.record_message(role, content)

    # ------------------------------------------------------------------
    # Post-write tool_call compaction
    # ------------------------------------------------------------------

    def _compact_write_tool_call(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> None:
        """Replace full file content in stored tool_call with compact ref.

        After a successful file write, the content lives on disk.  Keeping
        the full body inside the assistant message's ``tool_calls`` wastes
        context budget — especially in multi-file generation tasks where
        tens of thousands of tokens of code accumulate in history.

        This walks backwards through session messages, finds the parent
        assistant message for *call_id*, and replaces the ``content``
        argument with a short pointer (path + line count + size).
        """
        if not self.session:
            return
        path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        lines = content.count("\n") + 1 if content else 0
        size_kb = len(content.encode("utf-8", errors="replace")) / 1024 if content else 0

        # Record in workspace manifest — lightweight tracking of all
        # files created/modified this session for context injection.
        self._record_workspace_write(path, lines, size_kb)

        if not content or len(content) < 200:
            # Not worth compacting tiny writes — the overhead of the
            # reference string is comparable to the original content.
            return
        compact_ref = (
            f"[written to disk: {path}, {lines} lines, {size_kb:.1f}KB"
            f" — use read tool to view]"
        )

        import json as _json

        for msg in reversed(self.session.messages):
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc_stored in msg["tool_calls"]:
                if str(tc_stored.get("id", "")).strip() != str(call_id).strip():
                    continue
                # Found the matching tool_call — compact its arguments.
                func = tc_stored.get("function") or {}
                raw_args = func.get("arguments", "")
                try:
                    args_dict = (
                        _json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else dict(raw_args)
                    )
                    args_dict["content"] = compact_ref
                    func["arguments"] = _json.dumps(
                        args_dict, ensure_ascii=True
                    )
                except (ValueError, TypeError):
                    return  # malformed — leave as-is
                # Invalidate cached token count so _ensure_message_token_count
                # recomputes with the smaller payload.
                msg.pop("token_count", None)
                msg.pop("_tc_counted", None)
                log.info(
                    "Write tool_call compacted",
                    call_id=call_id,
                    path=path,
                    original_chars=len(content),
                )
                return

    def _compact_shell_tool_call(
        self,
        call_id: str,
        arguments: dict[str, Any],
    ) -> None:
        """Compact large shell commands in stored tool_call arguments.

        When the LLM uses shell heredocs (``cat > file << 'EOF'...EOF``)
        to write files instead of the write tool, the full file content
        accumulates in the assistant message's tool_call arguments,
        bypassing ``_compact_write_tool_call``.  This method detects
        large shell commands and replaces the command text with a
        compact summary, preserving only a leading snippet so the LLM
        can remember the intent.
        """
        if not self.session:
            return
        command = str(arguments.get("command", ""))
        if not command or len(command) < 500:
            return  # not worth compacting

        # Build a compact reference: first 120 chars + ellipsis + size.
        # Keep the opening line(s) so the LLM knows what was executed.
        snippet_lines = command.split("\n", 4)[:4]
        snippet = "\n".join(snippet_lines)
        if len(snippet) > 200:
            snippet = snippet[:200]
        lines_total = command.count("\n") + 1
        size_kb = len(command.encode("utf-8", errors="replace")) / 1024
        compact = (
            f"{snippet}\n\n"
            f"[... shell command truncated — {lines_total} lines, "
            f"{size_kb:.1f}KB total — files written to disk, "
            f"use read tool to view]"
        )

        import json as _json

        for msg in reversed(self.session.messages):
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            for tc_stored in msg["tool_calls"]:
                if str(tc_stored.get("id", "")).strip() != str(call_id).strip():
                    continue
                func = tc_stored.get("function") or {}
                raw_args = func.get("arguments", "")
                try:
                    args_dict = (
                        _json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else dict(raw_args)
                    )
                    args_dict["command"] = compact
                    func["arguments"] = _json.dumps(
                        args_dict, ensure_ascii=True,
                    )
                except (ValueError, TypeError):
                    return
                msg.pop("token_count", None)
                msg.pop("_tc_counted", None)
                log.info(
                    "Shell tool_call compacted",
                    call_id=call_id,
                    original_chars=len(command),
                    compact_chars=len(compact),
                )
                return

    # ------------------------------------------------------------------
    # Workspace manifest — track files created/modified this session
    # ------------------------------------------------------------------

    def _record_workspace_write(
        self, path: str, lines: int, size_kb: float
    ) -> None:
        """Record a file write in the workspace manifest.

        The manifest is stored in session metadata so it persists across
        session resumes.  Duplicate paths are updated (last-write-wins).
        """
        if not self.session or not path:
            return
        manifest: list[dict[str, Any]] = self.session.metadata.setdefault(
            "workspace_manifest", []
        )
        # Update existing entry or append new one.
        for entry in manifest:
            if entry.get("path") == path:
                entry["lines"] = lines
                entry["size_kb"] = round(size_kb, 1)
                entry["writes"] = entry.get("writes", 1) + 1
                return
        manifest.append({
            "path": path,
            "lines": lines,
            "size_kb": round(size_kb, 1),
            "writes": 1,
        })

    def _build_workspace_manifest_note(self) -> str:
        """Build a compact context note listing files created this session.

        Injected into the LLM context so it always knows what files
        exist in the workspace without needing full contents in history.
        When large files exist, includes a hint to prefer edit over write.
        """
        if not self.session:
            return ""
        manifest: list[dict[str, Any]] = self.session.metadata.get(
            "workspace_manifest", []
        )
        if not manifest:
            return ""
        # Only inject once the project has meaningful size.
        if len(manifest) < 3:
            return ""
        header = "## Workspace files created/modified this session\n"
        lines_parts: list[str] = []
        total_lines = 0
        has_large_files = False
        has_rewrites = False
        for entry in manifest:
            p = entry.get("path", "?")
            ln = entry.get("lines", 0)
            sk = entry.get("size_kb", 0)
            writes = entry.get("writes", 1)
            total_lines += ln
            rewrite_tag = f"  ⚠️ rewritten {writes}x" if writes > 1 else ""
            lines_parts.append(f"- {p}  ({ln} lines, {sk}KB){rewrite_tag}")
            if ln > 50 or sk > 2.0:
                has_large_files = True
            if writes > 1:
                has_rewrites = True
        footer = f"\n_Total: {len(manifest)} files, ~{total_lines} lines_"

        # Contextual hints based on workspace state.
        hints: list[str] = []
        if has_large_files:
            hints.append(
                "**Tip:** For files already on disk, prefer the `edit` tool "
                "(replace_string, insert_after, delete_lines, etc.) for "
                "targeted changes. Only use `write` for new files or "
                "complete rewrites. This saves context tokens and avoids "
                "accidental regressions."
            )
        if has_rewrites:
            hints.append(
                "**Note:** Some files above were written multiple times. "
                "Use `read` to check existing content before deciding "
                "whether a full rewrite is needed, or use `edit` for "
                "surgical changes."
            )
        hint_block = "\n".join(hints)
        if hint_block:
            hint_block = "\n\n" + hint_block
        return header + "\n".join(lines_parts) + footer + hint_block
