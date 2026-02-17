"""Agent orchestration for Captain Claw."""

import asyncio
import json
import sys
from typing import Any, AsyncIterator

from captain_claw.config import get_config
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

    def __init__(self, provider: LLMProvider | None = None):
        """Initialize the agent.
        
        Args:
            provider: Optional LLM provider override
        """
        self.provider = provider
        self.tools = get_tool_registry()
        self.session_manager = get_session_manager()
        self.session: Session | None = None
        self._initialized = False
        self.max_iterations = 10  # Max tool calls per message

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
        return """You are Captain Claw, a powerful AI assistant that can use tools to help the user.

Available tools:
- shell: Execute shell commands in the terminal
- read: Read file contents from the filesystem
- write: Write content to files
- glob: Find files by pattern
- web_fetch: Fetch web page content

Instructions:
- Use tools when you need to access files, run commands, or get information
- Think step by step
- Provide clear, concise responses
- If a tool fails, explain the error and try again if possible
- Always confirm before executing potentially dangerous commands"""

    def _build_messages(self, tool_messages_from_index: int | None = None) -> list[Message]:
        """Build message list for LLM."""
        messages = []
        
        # System prompt
        messages.append(Message(
            role="system",
            content=self._build_system_prompt(),
        ))
        
        # Session messages
        if self.session:
            for idx, msg in enumerate(self.session.messages):
                # Some Ollama cloud models return 500 when historical tool results are present.
                # Keep only tool messages from the current turn when index is provided.
                if (
                    tool_messages_from_index is not None
                    and msg["role"] == "tool"
                    and idx < tool_messages_from_index
                ):
                    continue
                messages.append(Message(
                    role=msg["role"],
                    content=msg["content"],
                    tool_call_id=msg.get("tool_call_id"),
                    tool_name=msg.get("tool_name"),
                ))
        
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
                if self.session:
                    self.session.add_message(
                        role="tool",
                        content=result.content if result.success else f"Error: {result.error}",
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                    )
                
                results.append({
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "success": result.success,
                    "content": result.content if result.success else result.error,
                })
                
            except Exception as e:
                log.error("Tool execution failed", tool=tc.name, error=str(e))
                
                if self.session:
                    self.session.add_message(
                        role="tool",
                        content=f"Error: {str(e)}",
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                    )
                
                results.append({
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "success": False,
                    "error": str(e),
                })
        
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
        
        turn_start_idx = len(self.session.messages) if self.session else 0

        # Add user message to session
        if self.session:
            self.session.add_message("user", user_input)
        
        # Send tool definitions so the model can issue structured tool calls.
        tool_defs = self.tools.get_definitions()
        log.debug("Tool definitions available", count=len(self.tools.list_tools()), tools_sent=bool(tool_defs))
        
        # Main agent loop
        for iteration in range(self.max_iterations):
            # Build messages for LLM
            messages = self._build_messages(tool_messages_from_index=turn_start_idx)
            
            # Call LLM
            log.info("Calling LLM", iteration=iteration + 1, message_count=len(messages))
            try:
                response = await self.provider.complete(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                )
            except Exception as e:
                # Check if this is a 500 error after tool execution
                error_str = str(e)
                tool_msgs = [
                    m for m in self.session.messages[turn_start_idx:]
                    if m.get("role") == "tool"
                ] if self.session else []
                
                # If we have tool messages AND got a 500 error, return tool output
                # This handles the case where Ollama can't process tool results in context
                if tool_msgs and "500" in error_str:
                    log.warning("Tool result call failed (500), returning tool output")
                    output = "\n\n".join([m['content'] for m in tool_msgs])
                    return f"Tool executed:\n{output}"
                
                log.error("LLM call failed", error=str(e), exc_info=True)
                raise
            
            # Check for explicit tool calls (for models that support it)
            if response.tool_calls:
                log.info("Tool calls detected", count=len(response.tool_calls), calls=response.tool_calls)
                await self._handle_tool_calls(response.tool_calls)
                # Try to get final response
                messages = self._build_messages(tool_messages_from_index=turn_start_idx)
                try:
                    response = await self.provider.complete(messages=messages, tools=None)
                except Exception as e:
                    # Model doesn't support tool results - return tool output
                    log.warning("Model doesn't support tool results", error=str(e))
                    tool_msgs = [
                        m for m in self.session.messages[turn_start_idx:]
                        if m.get("role") == "tool"
                    ] if self.session else []
                    if tool_msgs:
                        output = "\n\n".join([m['content'] for m in tool_msgs])
                        final = f"Tool executed:\n{output}"
                        if self.session:
                            self.session.add_message("assistant", final)
                            await self.session_manager.save_session(self.session)
                        return final
                    return "Tool executed but couldn't get final response."
                continue
            
            # Check for tool calls embedded in response text (fallback)
            # Looking for patterns like: {tool => "shell", args => {...}}
            embedded_calls = self._extract_tool_calls_from_content(response.content)
            if embedded_calls:
                log.info("Tool calls found in response text", count=len(embedded_calls))
                await self._handle_tool_calls(embedded_calls)
                # Try to get final response
                messages = self._build_messages(tool_messages_from_index=turn_start_idx)
                try:
                    response = await self.provider.complete(messages=messages, tools=None)
                except Exception as e:
                    log.warning("Model doesn't support tool results", error=str(e))
                    tool_msgs = [
                        m for m in self.session.messages[turn_start_idx:]
                        if m.get("role") == "tool"
                    ] if self.session else []
                    if tool_msgs:
                        output = "\n\n".join([m['content'] for m in tool_msgs])
                        final = f"Tool executed:\n{output}"
                        if self.session:
                            self.session.add_message("assistant", final)
                            await self.session_manager.save_session(self.session)
                        return final
                    return "Tool executed but couldn't get final response."
                # If successful, return the response normally
                return response.content
            
            # Check for inline commands in response (fallback for models without tool calling)
            # This works by extracting commands from markdown code blocks in the response
            command = self._extract_command_from_response(response.content)
            if command:
                log.info("Executing inline command", command=command)
                result = await self.tools.execute(name="shell", arguments={"command": command})
                tool_result = result.content if result.success else f"Error: {result.error}"
                
                # Add tool result to session
                if self.session:
                    self.session.add_message(role="tool", content=tool_result)
                
                # Get final response
                messages = self._build_messages(tool_messages_from_index=turn_start_idx)
                try:
                    response = await self.provider.complete(messages=messages, tools=None)
                    # If successful, continue to process the response
                    continue
                except:
                    # Return tool output directly
                    final = f"Tool executed:\n{tool_result}"
                    if self.session:
                        self.session.add_message("assistant", final)
                        await self.session_manager.save_session(self.session)
                    return final
            
            # No tool calls - this is the final response
            final_response = response.content
            
            # Add assistant response to session
            if self.session:
                self.session.add_message("assistant", final_response)
                # Save session after each turn
                await self.session_manager.save_session(self.session)
            
            return final_response
        
        # Max iterations reached
        return "Max iterations reached. Could not complete the request."

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """Stream response for user input.
        
        Args:
            user_input: User's message
        
        Yields:
            Response chunks
        """
        if not self._initialized:
            await self.initialize()
        
        # Add user message to session
        if self.session:
            self.session.add_message("user", user_input)
        
        # Get tool definitions
        tool_defs = self.tools.get_definitions()
        
        # For streaming, we currently don't support tool calling
        # This is a limitation - full streaming with tools needs more work
        messages = self._build_messages()
        
        # Stream the response
        full_content = ""
        async for chunk in self.provider.complete_streaming(
            messages=messages,
            tools=tool_defs if tool_defs else None,
        ):
            full_content += chunk
            yield chunk
        
        # Add assistant response to session
        if self.session:
            self.session.add_message("assistant", full_content)
            await self.session_manager.save_session(self.session)
