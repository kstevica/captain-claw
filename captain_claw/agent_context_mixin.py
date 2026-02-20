"""Prompt/message context assembly helpers for Agent."""

import importlib.util
import inspect
import json
import hashlib
import re
from pathlib import Path
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message, ToolCall, get_provider, set_provider
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool


log = get_logger(__name__)


class AgentContextMixin:
    """Build system/context/tool messages for model calls."""
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

    @staticmethod
    def _normalize_tool_policy_payload(raw: Any) -> dict[str, Any] | None:
        """Normalize policy payload shape for registry consumption."""
        if not isinstance(raw, dict):
            return None

        allow_raw = raw.get("allow")
        if allow_raw is None:
            allow: list[str] | None = None
        elif isinstance(allow_raw, list):
            allow = [str(item).strip() for item in allow_raw if str(item).strip()]
        else:
            return None

        deny_raw = raw.get("deny", [])
        deny = [str(item).strip() for item in deny_raw] if isinstance(deny_raw, list) else []
        deny = [item for item in deny if item]

        also_allow_raw = raw.get("also_allow", raw.get("alsoAllow", []))
        also_allow = (
            [str(item).strip() for item in also_allow_raw]
            if isinstance(also_allow_raw, list)
            else []
        )
        also_allow = [item for item in also_allow if item]

        if allow is None and not deny and not also_allow:
            return None
        return {
            "allow": allow,
            "deny": deny,
            "also_allow": also_allow,
        }

    def _session_tool_policy_payload(self) -> dict[str, Any] | None:
        """Load session-level tool policy from session metadata when present."""
        if not self.session or not isinstance(self.session.metadata, dict):
            return None
        return self._normalize_tool_policy_payload(self.session.metadata.get("tool_policy"))

    def _active_task_tool_policy_payload(self, planning_pipeline: dict[str, Any] | None) -> dict[str, Any] | None:
        """Load active task-level tool policy from pipeline task metadata."""
        if not isinstance(planning_pipeline, dict):
            return None
        graph = planning_pipeline.get("task_graph")
        if not isinstance(graph, dict):
            return None

        candidate_ids: list[str] = []
        current_id = str(planning_pipeline.get("current_task_id", "")).strip()
        if current_id:
            candidate_ids.append(current_id)
        raw_active = planning_pipeline.get("active_task_ids", [])
        if isinstance(raw_active, list):
            for item in raw_active:
                task_id = str(item).strip()
                if task_id and task_id not in candidate_ids:
                    candidate_ids.append(task_id)

        for task_id in candidate_ids:
            node = graph.get(task_id)
            if not isinstance(node, dict):
                continue
            normalized = self._normalize_tool_policy_payload(node.get("tool_policy"))
            if normalized is not None:
                return normalized
        return None

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

    def _initialize_layered_memory(self) -> None:
        """Create layered memory manager for semantic retrieval."""
        if getattr(self, "memory", None) is not None:
            return
        cfg = get_config()
        memory_cfg = getattr(cfg, "memory", None)
        if memory_cfg is None or not bool(getattr(memory_cfg, "enabled", True)):
            self.memory = None
            return
        session_db_path = getattr(self.session_manager, "db_path", None)
        if session_db_path is None:
            self.memory = None
            return
        try:
            from captain_claw.memory import create_layered_memory

            self.memory = create_layered_memory(
                config=cfg,
                session_db_path=session_db_path,
                workspace_path=self.workspace_base_path,
            )
            if self.session:
                self.memory.set_active_session(self.session.id)
                self.memory.schedule_background_sync("agent_initialize")
        except Exception as e:
            log.warning("Failed to initialize layered memory", error=str(e))
            self.memory = None

    def _build_semantic_memory_note(
        self,
        query: str | None,
        max_items: int = 3,
        max_snippet_chars: int = 360,
    ) -> tuple[str, str]:
        """Build semantic memory context note from persisted sessions + workspace files."""
        cleaned = str(query or "").strip()
        if not cleaned:
            return "", ""
        memory = getattr(self, "memory", None)
        if memory is None:
            return "", ""
        try:
            return memory.build_semantic_note(
                cleaned,
                max_items=max_items,
                max_snippet_chars=max_snippet_chars,
            )
        except Exception as e:
            log.debug("Semantic memory note generation failed", error=str(e))
            return "", ""

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
        self._initialize_layered_memory()

        # Register default tools
        self._register_default_tools()

        # Initialize file registry for single-agent mode.
        # Orchestration mode creates its own shared registry per run.
        if getattr(self, "_file_registry", None) is None:
            from captain_claw.file_registry import FileRegistry
            self._file_registry = FileRegistry(
                orchestration_id=f"session-{self.session.id}" if self.session else "default",
            )

        self._initialized = True
        log.info("Agent initialized", session_id=self.session.id)

    def _register_default_tools(self) -> None:
        """Register the default tool set."""
        from captain_claw.tools import (
            DocxExtractTool,
            GlobTool,
            PdfExtractTool,
            PocketTTSTool,
            PptxExtractTool,
            ReadTool,
            SendMailTool,
            ShellTool,
            WebFetchTool,
            WebSearchTool,
            WriteTool,
            XlsxExtractTool,
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
            elif tool_name == "pdf_extract":
                self.tools.register(PdfExtractTool())
            elif tool_name == "docx_extract":
                self.tools.register(DocxExtractTool())
            elif tool_name == "xlsx_extract":
                self.tools.register(XlsxExtractTool())
            elif tool_name == "pptx_extract":
                self.tools.register(PptxExtractTool())
            elif tool_name == "pocket_tts":
                self.tools.register(PocketTTSTool())
            elif tool_name == "send_mail":
                self.tools.register(SendMailTool())
        self._register_plugin_tools()

    def _discover_plugin_tool_files(self) -> list[Path]:
        """Discover plugin Python files from configured tool plugin directories."""
        cfg = get_config()
        candidates: list[Path] = []
        seen: set[str] = set()

        def _add_dir(path: Path) -> None:
            try:
                resolved = path.expanduser().resolve()
            except Exception:
                return
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            candidates.append(resolved)

        configured_dirs = list(getattr(cfg.tools, "plugin_dirs", []) or [])
        for raw in configured_dirs:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = (self.workspace_base_path / path).resolve()
            _add_dir(path)

        _add_dir(self.workspace_base_path / "skills" / "tools")
        _add_dir(self.tools.get_saved_base_path(create=True) / "tools")

        plugin_files: list[Path] = []
        added_files: set[str] = set()
        for directory in candidates:
            if not directory.exists() or not directory.is_dir():
                continue
            for file_path in sorted(directory.glob("*.py")):
                file_key = str(file_path.resolve())
                if file_key in added_files:
                    continue
                added_files.add(file_key)
                plugin_files.append(file_path.resolve())
        return plugin_files

    def _register_plugin_tools(self) -> None:
        """Load and register tools from plugin files."""
        plugin_files = self._discover_plugin_tool_files()
        if not plugin_files:
            return

        for file_path in plugin_files:
            module_name = f"captain_claw_plugin_{hashlib.sha1(str(file_path).encode('utf-8')).hexdigest()[:12]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    log.warning("Skipping plugin tool file with missing loader", path=str(file_path))
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                log.warning("Failed to import plugin tool file", path=str(file_path), error=str(e))
                continue

            registered_count = 0
            register_tools_fn = getattr(module, "register_tools", None)
            if callable(register_tools_fn):
                before_names = set(self.tools.list_tools())
                try:
                    register_tools_fn(self.tools)
                    after_names = set(self.tools.list_tools())
                    for tool_name in sorted(after_names - before_names):
                        existing_meta = self.tools.get_tool_metadata(tool_name)
                        existing_meta.update(
                            {
                                "source": "plugin",
                                "path": str(file_path),
                                "module": module_name,
                            }
                        )
                        tool = self.tools.get(tool_name)
                        self.tools.register(tool, metadata=existing_meta)
                    registered_count = len(after_names - before_names)
                except Exception as e:
                    log.warning(
                        "Plugin register_tools() failed",
                        path=str(file_path),
                        error=str(e),
                    )
                    continue
            else:
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if not issubclass(obj, Tool) or obj is Tool:
                        continue
                    if getattr(obj, "__module__", "") != module.__name__:
                        continue
                    try:
                        instance = obj()
                    except Exception as e:
                        log.warning(
                            "Failed to initialize plugin tool class",
                            path=str(file_path),
                            class_name=getattr(obj, "__name__", "<unknown>"),
                            error=str(e),
                        )
                        continue
                    if self.tools.has_tool(instance.name):
                        log.warning(
                            "Skipping plugin tool with duplicate name",
                            path=str(file_path),
                            tool=instance.name,
                        )
                        continue
                    self.tools.register(
                        instance,
                        metadata={
                            "source": "plugin",
                            "path": str(file_path),
                            "module": module_name,
                            "class_name": getattr(obj, "__name__", ""),
                        },
                    )
                    registered_count += 1

            if registered_count > 0:
                log.info(
                    "Registered plugin tools",
                    path=str(file_path),
                    count=registered_count,
                )

    def _build_system_prompt(self) -> str:
        """Build the system prompt."""
        session_id = self._current_session_slug()

        planning_block = ""
        if self.planning_enabled:
            planning_block = (
                "\n\n" + self.instructions.load("planning_mode_instructions.md")
            )

        saved_root = self.tools.get_saved_base_path(create=False)

        base_prompt = self.instructions.render(
            "system_prompt.md",
            runtime_base_path=self.runtime_base_path,
            workspace_root=self.workspace_base_path,
            saved_root=saved_root,
            session_id=session_id,
            planning_block=planning_block,
        )
        skills_section = ""
        build_skills = getattr(self, "_build_skills_system_prompt_section", None)
        if callable(build_skills):
            try:
                skills_section = str(build_skills() or "").strip()
            except Exception:
                skills_section = ""
        if skills_section:
            return f"{base_prompt.strip()}\n\n{skills_section}\n"
        return base_prompt

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
        # Collect tool_call_ids that were filtered so we can strip the
        # corresponding tool_calls from their parent assistant messages.
        _filtered_tool_call_ids: set[str] = set()
        if self.session:
            # First pass: identify which tool response messages will be filtered.
            if filter_historical_tools:
                for idx, msg in enumerate(self.session.messages):
                    if msg.get("role") == "tool" and idx < tool_messages_from_index:
                        tcid = str(msg.get("tool_call_id", "")).strip()
                        if tcid:
                            _filtered_tool_call_ids.add(tcid)

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
                # Strip tool_calls from assistant messages whose tool responses
                # were filtered out, preventing orphaned tool_calls references.
                if (
                    _filtered_tool_call_ids
                    and msg.get("role") == "assistant"
                    and msg.get("tool_calls")
                ):
                    remaining_calls = [
                        tc for tc in msg["tool_calls"]
                        if str(tc.get("id", "")).strip() not in _filtered_tool_call_ids
                    ]
                    if len(remaining_calls) != len(msg["tool_calls"]):
                        msg = dict(msg)  # shallow copy to avoid mutating session
                        if remaining_calls:
                            msg["tool_calls"] = remaining_calls
                        else:
                            msg.pop("tool_calls", None)
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

        semantic_note, semantic_debug = self._build_semantic_memory_note(query=query)
        if semantic_note:
            candidate_messages.append({
                "role": "assistant",
                "content": semantic_note,
                "tool_name": "semantic_memory_context",
                "token_count": self._count_tokens(semantic_note),
            })
            semantic_signature = f"{query or ''}|{semantic_debug}"
            if semantic_signature != getattr(self, "_last_semantic_memory_debug_signature", None):
                self._emit_tool_output(
                    "memory_semantic_select",
                    {"query": query or ""},
                    semantic_debug,
                )
                self._last_semantic_memory_debug_signature = semantic_signature

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
