"""CLI framework for Captain Claw."""

import asyncio
import sys
from typing import Any

from captain_claw.config import get_config
from captain_claw.logging import get_logger

log = get_logger(__name__)


class TerminalUI:
    """Terminal UI using Blessed."""

    def __init__(self):
        self.config = get_config()
        self._history: list[str] = []
        self._history_index = -1
        self._colors_enabled = False
        
        # For now, just disable colors to avoid terminal capability issues
        # Can enable later with proper terminal detection
        self._colors_enabled = False
        self.term = None

    def print_welcome(self) -> None:
        """Print welcome message."""
        if not self._colors_enabled:
            print("=== Captain Claw ===")
            print("Type your message or 'exit' to quit.")
            print("Type '/help' for commands.\n")
            return
        
        try:
            print("=== Captain Claw ===")
            print("Console AI Agent (Ollama)")
            print()
            print("Type '/help' for commands.\n")
        except Exception:
            print("=== Captain Claw ===")
            print("Type '/help' for commands.\n")

    def print_help(self) -> None:
        """Print help message."""
        print("""
Commands:
  /help           - Show this help message
  /clear          - Clear current session
  /new            - Start a new session
  /config         - Show configuration
  /history        - Show conversation history
  /exit, /quit    - Exit the application
  
  Just type your message to chat with Captain Claw!
""")

    def print_message(self, role: str, content: str) -> None:
        """Print a message with styling."""
        if not self._colors_enabled:
            print(f"[{role.upper()}] {content}")
            return
        
        try:
            colors = {
                "user": lambda x: f"[USER] {x}",
                "assistant": lambda x: f"[ASSISTANT] {x}",
                "system": lambda x: f"[SYSTEM] {x}",
                "tool": lambda x: f"[TOOL] {x}",
            }
            color_fn = colors.get(role, lambda x: f"[{role.upper()}] {x}")
            print(color_fn(content))
        except Exception:
            print(f"[{role.upper()}] {content}")

    def print_error(self, error: str) -> None:
        """Print an error message."""
        print(f"Error: {error}")

    def print_warning(self, warning: str) -> None:
        """Print a warning message."""
        print(f"Warning: {warning}")

    def print_success(self, message: str) -> None:
        """Print success message."""
        print(f"OK: {message}")

    def print_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Print token usage."""
        if not self.config.ui.show_tokens:
            return
        
        total = prompt_tokens + completion_tokens
        print(f"Tokens: {prompt_tokens} + {completion_tokens} = {total}")

    def print_streaming(self, chunk: str, end: str = "") -> None:
        """Print streaming response chunk."""
        print(chunk, end=end, flush=True)

    def print_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Print tool call."""
        print(f"[TOOL] {tool_name}: {arguments}")

    def print_tool_result(self, tool_name: str, result: str, truncated: bool = False) -> None:
        """Print tool result."""
        result_text = result[:200] + "..." if len(result) > 200 else result
        if truncated:
            result_text += " [truncated]"
        print(f"[TOOL RESULT] {tool_name}: {result_text}")

    def print_config(self, config) -> None:
        """Print current configuration."""
        print("\n=== Current Configuration ===")
        print(f"Provider: {config.model.provider}")
        print(f"Model: {config.model.model}")
        print(f"Temperature: {config.model.temperature}")
        print(f"Max tokens: {config.model.max_tokens}")
        print(f"Streaming: {config.ui.streaming}")
        print(f"Show tokens: {config.ui.show_tokens}")
        print(f"Enabled tools: {', '.join(config.tools.enabled)}")
        print()

    def print_history(self, messages: list[dict]) -> None:
        """Print conversation history."""
        print("\n=== Conversation History ===")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:100]
            if len(msg.get("content", "")) > 100:
                content += "..."
            print(f"[{i+1}] {role}: {content}")
        print()

    def print_session_info(self, session) -> None:
        """Print session info."""
        print(f"\nSession: {session.name}")
        print(f"ID: {session.id}")
        print(f"Messages: {len(session.messages)}")
        print()

    def prompt(self, prompt_text: str = "> ") -> str:
        """Prompt for input."""
        return input(prompt_text)

    def confirm(self, message: str) -> bool:
        """Ask for confirmation."""
        response = input(f"{message} (y/n) ").lower().strip()
        return response in ("y", "yes")

    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        try:
            if self.term:
                print(self.term.clear)
        except Exception:
            print("\n" + "=" * 50 + "\n")

    def handle_special_command(self, cmd: str) -> str | None:
        """Handle special commands."""
        cmd = cmd.strip()
        
        if not cmd.startswith("/"):
            return cmd
        
        parts = cmd.split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in ("/help", "/h", "/?"):
            self.print_help()
            return None
        elif command == "/clear":
            return "CLEAR"
        elif command == "/new":
            return "NEW"
        elif command == "/config":
            return "CONFIG"
        elif command == "/history":
            return "HISTORY"
        elif command in ("/exit", "/quit", "/q"):
            return "EXIT"
        else:
            self.print_error(f"Unknown command: {command}")
            return None


# Global UI instance
_ui: "TerminalUI | None" = None


def get_ui() -> TerminalUI:
    """Get the global UI instance."""
    global _ui
    if _ui is None:
        _ui = TerminalUI()
    return _ui


def set_ui(ui: TerminalUI) -> None:
    """Set the global UI instance."""
    global _ui
    _ui = ui
