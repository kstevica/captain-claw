"""Tool-call parsing/execution helpers for Agent.

This mixin handles:
- Tool call extraction from LLM response content (multiple formats)
- Tool call execution with guards, duplicate detection, and scale tracking
- Tool output collection and friendly rewriting
- Tool thinking summaries for UI indicators
"""

import asyncio
import json
import os
import re
from typing import Any

from captain_claw.config import get_config
from captain_claw.llm import Message, ToolCall
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentToolLoopMixin:
    """Extract commands, parse embedded tool calls, and execute tool loops."""

    # ------------------------------------------------------------------
    # Tool thinking summaries
    # ------------------------------------------------------------------

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
        if name == "web_get":
            url = str(arguments.get("url", "")).strip()
            return f"Fetching HTML: {url[:60]}" if url else "Fetching raw HTML"
        if name == "web_search":
            query = str(arguments.get("query", "")).strip()
            return f"Searching: {query[:60]}" if query else "Searching the web"
        if name == "send_mail":
            to = str(arguments.get("to", "")).strip()
            return f"Sending email to {to}" if to else "Sending email"
        if name in {"todos", "contacts", "scripts", "apis"}:
            action = str(arguments.get("action", "list")).strip()
            return f"{action.capitalize()}: {name} memory"
        # ── Document extractors ───────────────────────────────
        if name in {"pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract"}:
            path = str(arguments.get("path", "")).strip()
            label = name.replace("_extract", "").upper()
            return f"Extracting {label}: {path.rsplit('/', 1)[-1]}" if path else f"Extracting {label}"
        # ── Google Drive ─────────────────────────────────────
        if name == "google_drive":
            action = str(arguments.get("action", "")).strip()
            return f"Google Drive: {action}" if action else "Google Drive"
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
            # Prefer the deepest scope title from scope_progress (human-
            # readable step name) over the raw numeric path like "5".
            scope_progress = arguments.get("scope_progress")
            title = ""
            if isinstance(scope_progress, list) and scope_progress:
                deepest = scope_progress[-1]
                if isinstance(deepest, dict):
                    title = str(deepest.get("title", "")).strip()
            current_path = str(arguments.get("current_path", "")).strip()
            label = title or current_path
            if label:
                label = label[:80]
                if event and "completed" in event:
                    return f"✓ {label}"
                return f"▸ {label}"
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

    # ------------------------------------------------------------------
    # Shell command extraction from response
    # ------------------------------------------------------------------

    def _extract_command_from_response(self, content: str) -> str | None:
        """Extract shell command from model response.

        Looks for:
        - ```bash\\ncommand\\n``` or ```shell\\ncommand\\n```
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

    # ------------------------------------------------------------------
    # Model compatibility
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Turn-level tool output management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Embedded tool call extraction
    # ------------------------------------------------------------------

    def _extract_tool_calls_from_content(self, content: str) -> list[ToolCall]:
        """Extract tool calls from response content text.

        Looks for various formats:
        - @shell\\ncommand: value
        - {tool => "shell", args => { --command "ls -la" }}
        - ```tool\\ncommand\\n```
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

    # ------------------------------------------------------------------
    # Main tool call handler & execution engine
    # ------------------------------------------------------------------

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

        # Reset per-batch extraction counter for scale guard.
        sp = getattr(self, "_scale_progress", None)
        if sp is not None:
            sp["_batch_extractions"] = 0

        _PATH_TOOLS = {"read", "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract"}
        _URL_TOOLS = {"web_fetch", "web_get"}

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

            # ── Scale guard: hard redirect for off-track calls ─────
            # Must run before dup detection and execution. When the guard
            # fires, we skip execution entirely and return a redirect
            # message that tells the LLM what to process next.
            guard_msg = self._scale_guard_intercept(
                tc.name,
                arguments if isinstance(arguments, dict) else {},
            )
            if guard_msg is not None:
                log.info(
                    "Scale guard blocked tool call",
                    tool=tc.name,
                    call_id=tc.id,
                )
                self._emit_thinking(
                    f"🛡️ Scale guard: redirecting from {tc.name}",
                    tool=tc.name,
                    phase="tool",
                )
                self._add_session_message(
                    role="tool",
                    content=guard_msg,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    guard_msg,
                )
                results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": guard_msg,
                })
                continue

            # Emit inline thinking indicator for the tool being executed.
            args_dict = arguments if isinstance(arguments, dict) else {}
            summary = self._tool_thinking_summary(tc.name, args_dict)
            # When scale-progress is active, prefix extract/read calls
            # with progress info like "📄 3 of 27 — ".
            sp = getattr(self, "_scale_progress", None)
            if sp is not None and sp.get("total", 0) >= 3:
                tool_lower = str(tc.name or "").strip().lower()
                _extractors = {"pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract", "read"}
                if tool_lower in _extractors:
                    done = sp.get("completed", 0)
                    total = sp["total"]
                    # +1 because the next write(append) will complete this item
                    current = min(done + 1, total)
                    summary = f"📄 {current} of {total} — {summary}"
            self._emit_thinking(summary, tool=tc.name, phase="tool")

            # ── Duplicate tool call detection ─────────────────────
            # If the LLM keeps requesting the same tool within a single
            # turn (e.g. re-fetching the same URL 10 times, or re-extracting
            # the same PDF with different max_chars), skip execution after
            # the Nth call and return a warning instead.
            # For path-based tools (read, pdf_extract, etc.) and URL-based
            # tools (web_fetch, web_get), the key dimension is the path/URL
            # — not the full args.  Re-reading the same file with different
            # limits is still a duplicate.
            _dup_max = max(1, int(get_config().tools.duplicate_call_max))
            _tool_lower = str(tc.name or "").strip().lower()
            # Stateful tools that modify data between calls — exempt from
            # duplicate detection because index→search sequences are normal.
            _STATEFUL_TOOLS = {"typesense", "todo", "contacts", "scripts", "apis", "send_mail"}
            # During scale-progress tasks, give extra headroom so the LLM
            # can recover from a failed first attempt or a write-before-read
            # situation without being permanently locked out of a file.
            _sp = getattr(self, "_scale_progress", None)
            if _sp is not None and _tool_lower in (_PATH_TOOLS | _URL_TOOLS | {"glob"}):
                _dup_max = max(_dup_max, 2)
            if _tool_lower in _PATH_TOOLS and isinstance(arguments, dict):
                _sig_key = str(arguments.get("path", "")).strip()
                # Normalize to absolute path so relative and absolute
                # references to the same file share one dup-counter.
                if _sig_key:
                    _sig_key = os.path.abspath(_sig_key)
                _dup_sig = f"{tc.name}|path={_sig_key}"
            elif _tool_lower in _URL_TOOLS and isinstance(arguments, dict):
                _sig_key = str(arguments.get("url", "")).strip()
                _dup_sig = f"{tc.name}|url={_sig_key}"
            elif _tool_lower == "glob" and isinstance(arguments, dict):
                _sig_key = str(arguments.get("pattern", "")).strip()
                _dup_sig = f"{tc.name}|pattern={_sig_key}"
            else:
                try:
                    _sig_args = json.dumps(arguments, sort_keys=True, ensure_ascii=True) if isinstance(arguments, dict) else str(arguments)
                except Exception:
                    _sig_args = str(arguments)
                _dup_sig = f"{tc.name}|{_sig_args}"
            _dup_counts: dict[str, int] = getattr(self, "_turn_tool_call_counts", {})
            _dup_count = _dup_counts.get(_dup_sig, 0)
            # Stateful tools (index→search, CRUD operations) skip duplicate
            # detection — repeated calls are expected and legitimate.
            if _tool_lower in _STATEFUL_TOOLS:
                _dup_count = 0
            if _dup_count >= _dup_max:
                # During scale tasks, give a more actionable message that
                # tells the LLM to write what it has or skip the item.
                if _sp is not None and _tool_lower in (_PATH_TOOLS | _URL_TOOLS):
                    dup_msg = (
                        f"DUPLICATE CALL BLOCKED: You have already called `{tc.name}` on "
                        f"this target {_dup_count} times this turn. "
                        "If you have content from a previous call, write its summary "
                        "now (write, append=true). If you cannot recall the content, SKIP "
                        "this item and move on to the NEXT unprocessed item in the list. "
                        "Do NOT attempt to re-fetch this item again."
                    )
                else:
                    dup_msg = (
                        f"DUPLICATE CALL BLOCKED: You have already called `{tc.name}` on "
                        f"this target {_dup_count} times this turn. The content has not "
                        "changed. Use the data you already have and move on to the next "
                        "item. Do NOT re-read, re-extract, or re-glob the same target."
                    )
                log.info("Blocking duplicate tool call", tool=tc.name, call_id=tc.id, count=_dup_count)
                self._add_session_message(
                    role="tool",
                    content=dup_msg,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    dup_msg,
                )
                results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": dup_msg,
                })
                continue
            _dup_counts[_dup_sig] = _dup_count + 1
            if not hasattr(self, "_turn_tool_call_counts"):
                self._turn_tool_call_counts = _dup_counts

            # ── Scale-progress: track last action for write hint ──
            # Instead of hard-blocking read-before-write (which caused
            # deadlocks when interacting with the dup detector), we
            # append a soft reminder to the tool result after execution
            # if the LLM skipped a write step.  The LLM can still
            # proceed — the hint nudges it without creating unrecoverable
            # states.
            sp = getattr(self, "_scale_progress", None)
            _scale_write_hint = False
            _content_tools = _PATH_TOOLS | _URL_TOOLS
            if sp is not None and _tool_lower in _content_tools:
                if sp.get("_last_action") == "extract":
                    _scale_write_hint = True  # will append hint after execution
            elif sp is not None and _tool_lower == "write":
                sp["_last_action"] = "write"

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
                _result_content = result.content if result.success else f"Error: {result.error}"
                # Append a soft write-reminder when the LLM skipped writing
                # the previous item's result before reading a new one.
                # This is a HINT, not a hard block — execution still happened.
                if _scale_write_hint and result.success:
                    _result_content += (
                        "\n\n⚠️ REMINDER: You have processed content from TWO items "
                        "without writing a summary in between. Append the summary for "
                        "the PREVIOUS item to the output first (write, append=true), "
                        "then append the summary for THIS item. Do not skip any items."
                    )
                self._add_session_message(
                    role="tool",
                    content=_result_content,
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    tool_arguments=arguments if isinstance(arguments, dict) else None,
                )
                self._emit_tool_output(
                    tc.name,
                    arguments if isinstance(arguments, dict) else {},
                    _result_content,
                )

                results.append({
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "success": result.success,
                    "content": _result_content if result.success else result.error,
                })

                # ── Dup-counter rollback on failure ──────────────
                # If the call failed (e.g. wrong path), roll back the
                # duplicate counter so the LLM can retry with a
                # corrected argument (e.g. absolute path) without
                # being blocked by the dup detector.
                if not result.success:
                    _dup_counts = getattr(self, "_turn_tool_call_counts", {})
                    if _dup_sig in _dup_counts and _dup_counts[_dup_sig] > 0:
                        _dup_counts[_dup_sig] -= 1

                # ── Scale-progress tracking ───────────────────────
                # When a large-scale incremental task is running, track
                # glob results (to learn total count) and write(append)
                # calls (to count completed items) and emit a progress
                # indicator to the thinking line.
                #
                # Also: set _last_action="extract" on success so the
                # soft write-hint fires if the LLM reads another file
                # before writing.  Only set on success — a failed
                # extract should not trigger the hint on retry.
                sp = getattr(self, "_scale_progress", None)
                if result.success and sp is not None:
                    tool_lower = str(tc.name or "").strip().lower()
                    if tool_lower in _PATH_TOOLS:
                        sp["_last_action"] = "extract"
                    if tool_lower in _URL_TOOLS:
                        sp["_last_action"] = "extract"
                    if tool_lower == "glob" and result.content:
                        # Count lines in glob output to determine total items
                        # AND store the full list so we can inject a progress
                        # note into every LLM call (preventing the LLM from
                        # "forgetting" the list as context grows).
                        # Filter out non-path lines like "Found 27 file(s):"
                        # or other header/summary text the glob tool may emit.
                        lines = [
                            ln.strip() for ln in result.content.strip().splitlines()
                            if ln.strip()
                            and not re.match(r"^Found \d+", ln.strip())
                            and (
                                "/" in ln
                                or "\\" in ln
                                or "." in ln.strip().rsplit("/", 1)[-1]
                            )
                        ]
                        if lines:
                            sp["total"] = len(lines)
                            sp["completed"] = 0
                            sp["items"] = list(lines)
                            sp["done_items"] = set()
                            sp["_extraction_mode"] = self._classify_item_extraction_mode(
                                lines,
                                per_member_action=str(sp.get("_per_member_action", "")),
                            )
                            # Mark glob as completed so the scale guard
                            # blocks any subsequent re-glob attempts.
                            sp["_glob_completed"] = True
                    elif tool_lower == "write" and isinstance(arguments, dict):
                        # Track the output file path so the scale guard
                        # can block re-reads of it during the loop.
                        # We store TWO paths:
                        #   _output_file: the resolved absolute path (for
                        #     comparison in scale guard read-blocking)
                        #   _output_file_arg: the original arg passed to
                        #     the write tool (for the micro-loop to reuse,
                        #     since the write tool re-resolves its input)
                        # Extract the real resolved path from the tool result.
                        real_path = ""
                        if result.content and " to " in result.content:
                            after_to = result.content.split(" to ", 1)[-1]
                            real_path = after_to.split(" (requested:")[0].strip()
                        if not real_path:
                            write_path = str(arguments.get("path", "")).strip()
                            if write_path:
                                real_path = os.path.abspath(write_path)
                        if not sp.get("_output_file"):
                            if real_path:
                                sp["_output_file"] = real_path
                            # Also keep the original argument for micro-loop
                            original_arg = str(arguments.get("path", "")).strip()
                            if original_arg:
                                sp["_output_file_arg"] = original_arg
                        # Track distinct output files.  When the LLM writes
                        # to MULTIPLE different files (one per item), the
                        # micro-loop cannot take over because it assumes a
                        # single output file with appends.
                        if real_path:
                            output_files: set[str] = sp.setdefault("_output_files", set())
                            output_files.add(real_path)
                        # Track item completion.
                        # For append=True: match written content against items.
                        # For append=False: also try matching — the LLM may
                        # write separate files per item (e.g. "FinSMEs-18.07.2025.csv"
                        # for item "18.07.2025. https://…").
                        is_append = arguments.get("append") is True
                        if is_append:
                            sp["completed"] = sp.get("completed", 0) + 1
                        # Build a search text from both file content and path.
                        written_content = str(arguments.get("content", ""))
                        write_path = str(arguments.get("path", "")).strip().lower()
                        # Check a broader portion of written content —
                        # not just the first line — to catch items
                        # mentioned in markdown headers, sub-headings, etc.
                        content_head = written_content[:500].lower()
                        # For non-append writes, also search in the file path
                        # (the LLM may embed the item identifier in the filename).
                        search_text = content_head + " " + write_path
                        done_items: set[str] = sp.get("done_items", set())
                        items: list[str] = sp.get("items", [])
                        for item in items:
                            if item in done_items:
                                continue
                            # Build match candidates for this item:
                            # - filename (last path component)
                            # - URL domain+path for web URLs
                            # - the item itself (lowered)
                            candidates: list[str] = []
                            if "/" in item:
                                # Could be a file path or URL
                                candidates.append(item.rsplit("/", 1)[-1].lower())
                            # The item itself (e.g. "example.com/...", "Company Name")
                            candidates.append(item.lower())
                            # For URLs, also match just the path portion.
                            # Handle both plain URLs and items with
                            # embedded URLs (e.g. "18.07.2025 - https://…")
                            _url_in_item = None
                            if item.startswith(("http://", "https://")):
                                _url_in_item = item
                            else:
                                _um = re.search(r"https?://[^\s)\]}>\"']+", item)
                                if _um:
                                    _url_in_item = _um.group(0)
                            if _url_in_item:
                                try:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(_url_in_item)
                                    path_part = parsed.path.strip("/")
                                    if path_part:
                                        candidates.append(path_part.lower())
                                    # hostname + path for partial matching
                                    if parsed.hostname:
                                        candidates.append(
                                            f"{parsed.hostname}{parsed.path}".lower()
                                        )
                                except Exception:
                                    pass
                            # Extract date-like tokens from items
                            # (e.g. "18.07.2025" from "18.07.2025. https://…")
                            _date_m = re.search(r"\d{2}\.\d{2}\.\d{4}", item)
                            if _date_m:
                                candidates.append(_date_m.group(0))
                            # For non-path items (e.g. "Company Name"),
                            # also try normalized form
                            if " " in item:
                                candidates.append(
                                    re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
                                )
                            matched = any(
                                c and c in search_text
                                for c in candidates
                                if len(c) >= 3
                            )
                            if matched:
                                done_items.add(item)
                                if not is_append:
                                    sp["completed"] = sp.get("completed", 0) + 1
                                break
                        sp["done_items"] = done_items
                        total = sp.get("total", 0)
                        done = sp["completed"]
                        path = str(arguments.get("path", "")).strip()
                        filename = path.rsplit("/", 1)[-1] if path else ""
                        if total >= 3 and done <= total:
                            pct = int(done / total * 100)
                            progress_text = f"{done} of {total} ({pct}%)"
                        elif total >= 3:
                            # done > total: items discovered at runtime
                            progress_text = f"{done} items written"
                        else:
                            progress_text = f"{done} items written"
                        # Show last-written filename in the indicator.
                        # Look at the content to extract a section title
                        # if it starts with "## " or "# ".
                        content_str = str(arguments.get("content", ""))
                        label = ""
                        for cline in content_str.splitlines():
                            cline = cline.strip()
                            if cline.startswith("#"):
                                label = cline.lstrip("#").strip()[:60]
                                break
                        if not label and filename:
                            label = filename[:60]
                        display = f"📄 {progress_text}"
                        if label:
                            display += f" — {label}"
                        self._emit_thinking(display, tool="progress", phase="tool")
                        # ── Context trimming ─────────────────────
                        # After successfully writing a summary, the
                        # full extracted content of previous items is
                        # no longer needed.  Trim large tool results
                        # from earlier in the turn to keep the context
                        # lean and prevent the LLM from "forgetting"
                        # where it is in the list.
                        self._trim_processed_extracts_in_session()

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
                # Roll back dup counter on exception so a retry is allowed.
                _dup_counts = getattr(self, "_turn_tool_call_counts", {})
                if _dup_sig in _dup_counts and _dup_counts[_dup_sig] > 0:
                    _dup_counts[_dup_sig] -= 1

        self._set_runtime_status("thinking")
        return results
