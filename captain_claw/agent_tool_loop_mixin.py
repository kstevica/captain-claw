"""Tool-call parsing/execution helpers for Agent."""

import json
import re
from typing import Any

from captain_claw.llm import Message, ToolCall
from captain_claw.logging import get_logger


log = get_logger(__name__)


class AgentToolLoopMixin:
    """Extract commands, parse embedded tool calls, and execute tool loops."""
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

