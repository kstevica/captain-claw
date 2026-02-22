"""Tool-call parsing/execution helpers for Agent."""

import asyncio
import json
import re
from typing import Any

from captain_claw.llm import Message, ToolCall
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentToolLoopMixin:
    """Extract commands, parse embedded tool calls, and execute tool loops."""

    @staticmethod
    def _tool_thinking_summary(tool_name: str, arguments: dict[str, Any]) -> str:
        """Derive a short human-readable summary of a tool call for the thinking indicator."""
        name = str(tool_name or "").strip().lower()
        # ── Core tools ──────────────────────────────────────
        if name == "read":
            path = str(arguments.get("path", arguments.get("file_path", ""))).strip()
            return f"Reading: {path.rsplit('/', 1)[-1]}" if path else "Reading file"
        if name == "write":
            path = str(arguments.get("path", arguments.get("file_path", ""))).strip()
            return f"Writing: {path.rsplit('/', 1)[-1]}" if path else "Writing file"
        if name == "shell":
            cmd = str(arguments.get("command", "")).strip()
            return f"Running: {cmd[:60]}" if cmd else "Running shell command"
        if name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            return f"Fetching: {url[:60]}" if url else "Fetching URL"
        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            return f"Searching: {query[:60]}" if query else "Searching the web"
        if name == "send_mail":
            to = str(arguments.get("to", "")).strip()
            return f"Sending email to {to}" if to else "Sending email"
        if name in {"todos", "contacts", "scripts", "apis"}:
            action = str(arguments.get("action", "list")).strip()
            return f"{action.capitalize()}: {name} memory"
        # ── Memory & context ────────────────────────────────
        if name == "memory_select":
            query = str(arguments.get("query", "")).strip()
            return f"Selecting memory context: {query[:50]}" if query else "Selecting memory context"
        if name == "memory_semantic_select":
            query = str(arguments.get("query", "")).strip()
            return f"Semantic memory search: {query[:50]}" if query else "Semantic memory search"
        # ── Pipeline & planning ─────────────────────────────
        if name == "task_contract":
            step = str(arguments.get("step", "")).strip()
            return f"Task planner: {step}" if step else "Generating task contract"
        if name == "completion_gate":
            step = str(arguments.get("step", "")).strip()
            return f"Completion check: {step}" if step else "Evaluating completion"
        if name == "planning":
            event = str(arguments.get("event", "")).strip()
            return f"Planning: {event}" if event else "Updating plan"
        if name == "pipeline_trace":
            return "Pipeline trace"
        # ── Guards ──────────────────────────────────────────
        if name.startswith("guard_"):
            guard_type = name[6:]
            decision = str(arguments.get("decision", "")).strip()
            return f"Guard ({guard_type}): {decision}" if decision else f"Checking guard: {guard_type}"
        # ── Session management ──────────────────────────────
        if name == "compaction":
            trigger = str(arguments.get("trigger", "")).strip()
            return f"Compacting messages: {trigger}" if trigger else "Compacting session messages"
        if name == "session_procreate":
            step = str(arguments.get("step", "")).strip()
            return f"Session procreation: {step}" if step else "Procreating session"
        # ── LLM tracing ────────────────────────────────────
        if name == "llm_trace":
            return "LLM trace"
        # ── Media ──────────────────────────────────────────
        if name == "pocket_tts":
            return "Text-to-speech"
        # ── Approval ───────────────────────────────────────
        if name == "approval":
            return "Auto-approved action"
        # ── Fallback ───────────────────────────────────────
        return f"Running: {tool_name}"

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
        
        tool_calls: list[ToolCall] = []
        if not content:
            return tool_calls

        max_calls = 8
        seen: set[str] = set()

        def _append_tool_call(name: str, arguments: dict[str, Any]) -> None:
            if len(tool_calls) >= max_calls:
                return
            normalized_name = str(name or "").strip().lower()
            if not normalized_name or not isinstance(arguments, dict):
                return
            signature = json.dumps(
                {"name": normalized_name, "arguments": arguments},
                ensure_ascii=True,
                sort_keys=True,
            )
            if signature in seen:
                return
            seen.add(signature)
            tool_calls.append(
                ToolCall(
                    id=f"embedded_{len(tool_calls)}",
                    name=normalized_name,
                    arguments=arguments,
                )
            )
        
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
                        _append_tool_call(tool_name, {"command": command})
                elif pattern == pattern1:
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        _append_tool_call(tool_name, {"command": command})
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
                        _append_tool_call(tool_name, args)
                elif pattern == pattern3:
                    tool_name = match.group(1).strip().lower()
                    command = match.group(2).strip()
                    if tool_name and command:
                        _append_tool_call(tool_name, {"command": command})

        # Pattern 5: JSON tool calls and pseudo-tool argument objects.
        # Supports:
        # - {"tool":"web_search","args":{"query":"..."}}
        # - {"name":"web_fetch","arguments":{"url":"https://..."}}
        # - {"tool":"web_search","input":"..."}
        # - {"query":"..."}  -> web_search
        # - {"url":"https://..."} -> web_fetch
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            explicit_tool = str(payload.get("tool", payload.get("name", "")) or "").strip().lower()
            explicit_args = payload.get("args", payload.get("arguments"))
            if explicit_tool and isinstance(explicit_args, dict):
                args_obj = dict(explicit_args)
                if explicit_tool == "web_search" and "max_results" in args_obj and "count" not in args_obj:
                    try:
                        args_obj["count"] = int(args_obj.get("max_results"))
                    except Exception:
                        pass
                _append_tool_call(explicit_tool, args_obj)
                continue
            if explicit_tool:
                args_obj: dict[str, Any] = {}
                input_text = str(payload.get("input", "") or "").strip()
                if explicit_tool == "web_search":
                    query = str(payload.get("query", "") or "").strip()
                    if not query and input_text:
                        query = input_text
                    if query:
                        args_obj["query"] = query
                    if "count" in payload:
                        args_obj["count"] = payload.get("count")
                    elif "max_results" in payload:
                        args_obj["count"] = payload.get("max_results")
                    for key in ("offset", "country", "search_lang", "freshness", "safesearch"):
                        if key in payload:
                            args_obj[key] = payload.get(key)
                elif explicit_tool == "web_fetch":
                    url = str(payload.get("url", "") or "").strip()
                    if not url and input_text.startswith(("http://", "https://")):
                        url = input_text
                    if url.startswith(("http://", "https://")):
                        args_obj["url"] = url
                    if "max_chars" in payload:
                        args_obj["max_chars"] = payload.get("max_chars")
                    if "extract_mode" in payload:
                        args_obj["extract_mode"] = payload.get("extract_mode")
                elif explicit_tool == "shell":
                    command = str(payload.get("command", "") or "").strip()
                    if not command and input_text:
                        command = input_text
                    if command:
                        args_obj["command"] = command
                elif explicit_tool == "pocket_tts":
                    text = str(payload.get("text", "") or "").strip()
                    if not text and input_text:
                        text = input_text
                    if text:
                        args_obj["text"] = text
                    voice = str(payload.get("voice", "") or "").strip()
                    if voice:
                        args_obj["voice"] = voice
                    output_path = str(payload.get("output_path", "") or "").strip()
                    if output_path:
                        args_obj["output_path"] = output_path
                    if "sample_rate" in payload:
                        args_obj["sample_rate"] = payload.get("sample_rate")
                else:
                    for key, value in payload.items():
                        if key in {"tool", "name", "id", "input"}:
                            continue
                        args_obj[str(key)] = value
                    if input_text and "input" not in args_obj:
                        args_obj["input"] = input_text
                if args_obj:
                    _append_tool_call(explicit_tool, args_obj)
                    continue

            # Heuristic fallback: common pseudo-tool argument blobs.
            if "query" in payload:
                query = str(payload.get("query", "")).strip()
                if query:
                    args_obj: dict[str, Any] = {"query": query}
                    if "count" in payload:
                        args_obj["count"] = payload.get("count")
                    elif "max_results" in payload:
                        args_obj["count"] = payload.get("max_results")
                    for key in ("offset", "country", "search_lang", "freshness", "safesearch"):
                        if key in payload:
                            args_obj[key] = payload.get(key)
                    _append_tool_call("web_search", args_obj)
                    continue

            if "url" in payload:
                url = str(payload.get("url", "")).strip()
                if url.startswith(("http://", "https://")):
                    args_obj = {"url": url}
                    if "max_chars" in payload:
                        args_obj["max_chars"] = payload.get("max_chars")
                    if "extract_mode" in payload:
                        args_obj["extract_mode"] = payload.get("extract_mode")
                    _append_tool_call("web_fetch", args_obj)
        
        return tool_calls

    async def _handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        turn_usage: dict[str, int] | None = None,
        session_policy: dict[str, Any] | None = None,
        task_policy: dict[str, Any] | None = None,
        abort_event: asyncio.Event | None = None,
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

            # Emit inline thinking indicator for the tool being executed.
            args_dict = arguments if isinstance(arguments, dict) else {}
            summary = self._tool_thinking_summary(tc.name, args_dict)
            self._emit_thinking(summary, tool=tc.name, phase="tool")

            try:
                # Execute tool
                result = await self._execute_tool_with_guard(
                    name=tc.name,
                    arguments=arguments,
                    interaction_label=f"tool_call:{tc.name}",
                    turn_usage=turn_usage,
                    session_policy=session_policy,
                    task_policy=task_policy,
                    abort_event=abort_event,
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

                # Auto-capture contacts from send_mail usage.
                if result.success and hasattr(self, "_auto_capture_contacts_from_tool_call"):
                    try:
                        await self._auto_capture_contacts_from_tool_call(
                            tc.name, arguments if isinstance(arguments, dict) else {},
                        )
                    except Exception as _ac_err:
                        log.warning("Auto-capture contacts failed", tool=tc.name, error=str(_ac_err))

                # Auto-capture scripts from write tool usage.
                if result.success and hasattr(self, "_auto_capture_scripts_from_tool_call"):
                    try:
                        await self._auto_capture_scripts_from_tool_call(
                            tc.name, arguments if isinstance(arguments, dict) else {},
                        )
                    except Exception as _ac_err:
                        log.warning("Auto-capture scripts failed", tool=tc.name, error=str(_ac_err))

                # Auto-capture APIs from web_fetch tool usage.
                if result.success and hasattr(self, "_auto_capture_apis_from_tool_call"):
                    try:
                        await self._auto_capture_apis_from_tool_call(
                            tc.name, arguments if isinstance(arguments, dict) else {},
                        )
                    except Exception as _ac_err:
                        log.warning("Auto-capture APIs failed", tool=tc.name, error=str(_ac_err))

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
