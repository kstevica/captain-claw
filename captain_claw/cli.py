"""CLI framework for Captain Claw."""

import atexit
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import termios
import textwrap
import tty
import asyncio
import time
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
        self._special_commands = [
            "/help",
            "/clear",
            "/new",
            "/config",
            "/history",
            "/monitor",
            "/exit",
            "/quit",
        ]
        self._colors_enabled = False
        self._readline = None
        self._history_file = Path("~/.captain-claw/history").expanduser()
        self._sticky_footer = sys.stdout.isatty()
        self._ansi_enabled = self._sticky_footer and not bool(os.environ.get("NO_COLOR"))
        self._system_status_rows = 1
        self._system_log_rows = 5
        self._system_rows = self._system_status_rows + self._system_log_rows
        self._status_rows = 2
        self._prompt_rows = 1
        self._system_lines: deque[str] = deque(maxlen=self._system_log_rows)
        self._runtime_status = "waiting"
        self._footer_max_fps = 1.0
        self._footer_min_interval = 1.0 / self._footer_max_fps
        self._last_footer_render = 0.0
        self._status_line_1 = ""
        self._status_line_2 = ""
        self._assistant_output_active = False
        self._footer_dirty = False
        self._monitor_mode = False
        self._chat_output_text = ""
        self._tool_output_text = ""
        self._max_monitor_buffer_chars = 200_000
        
        # For now, just disable colors to avoid terminal capability issues
        # Can enable later with proper terminal detection
        self._colors_enabled = False
        self.term = None
        self._setup_readline()

    def _setup_readline(self) -> None:
        """Set up line editing, history, and command completion."""
        try:
            import readline  # type: ignore
        except Exception:
            return

        self._readline = readline

        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            if self._history_file.exists():
                readline.read_history_file(str(self._history_file))
            readline.set_history_length(1000)
            if hasattr(readline, "set_auto_history"):
                readline.set_auto_history(False)
            readline.parse_and_bind("tab: complete")
            readline.set_completer(self._complete_special_command)
            atexit.register(self._save_history)
        except Exception as e:
            log.debug("Readline setup failed", error=str(e))

    def has_sticky_layout(self) -> bool:
        """Whether fixed footer/system layout is active."""
        return self._sticky_footer

    def _save_history(self) -> None:
        """Persist readline history to disk."""
        if self._readline is None:
            return
        try:
            self._readline.write_history_file(str(self._history_file))
        except Exception as e:
            log.debug("Failed to save history", error=str(e))

    def _complete_special_command(self, text: str, state: int) -> str | None:
        """Readline completer for slash commands."""
        if not text.startswith("/"):
            return None
        matches = [cmd for cmd in self._special_commands if cmd.startswith(text)]
        if state < len(matches):
            return matches[state]
        return None

    def print_welcome(self) -> None:
        """Print welcome message."""
        self._prepare_output()
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
        help_text = """
Commands:
  /help           - Show this help message
  /clear          - Clear current session
  /new            - Start a new session
  /config         - Show configuration
  /history        - Show conversation history
  /monitor on     - Enable split monitor view
  /monitor off    - Disable split monitor view
  /exit, /quit    - Exit the application
  
  Just type your message to chat with Captain Claw!
"""
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(help_text + "\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        print(help_text)

    def print_message(self, role: str, content: str) -> None:
        """Print a message with styling."""
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(f"[{role.upper()}] {content}\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        styled_prefix = self._styled_role_prefix(role)
        if styled_prefix:
            print(f"{styled_prefix} {content}")
            return
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
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(f"Error: {error}\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        print(f"Error: {error}")

    def print_warning(self, warning: str) -> None:
        """Print a warning message."""
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(f"Warning: {warning}\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        print(f"Warning: {warning}")

    def print_success(self, message: str) -> None:
        """Print success message."""
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(f"OK: {message}\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        print(f"OK: {message}")

    def print_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Print token usage."""
        if self._monitor_mode and self._sticky_footer:
            if not self.config.ui.show_tokens:
                return
            total = prompt_tokens + completion_tokens
            self._append_chat_text(f"Tokens: {prompt_tokens} + {completion_tokens} = {total}\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        if not self.config.ui.show_tokens:
            return
        
        total = prompt_tokens + completion_tokens
        print(f"Tokens: {prompt_tokens} + {completion_tokens} = {total}")

    def print_streaming(self, chunk: str, end: str = "") -> None:
        """Print streaming response chunk."""
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(chunk)
            if end:
                self._append_chat_text(end)
            self._render_monitor_view()
            return
        print(chunk, end=end, flush=True)

    def print_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Print tool call."""
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(f"[TOOL] {tool_name}: {arguments}\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        print(f"[TOOL] {tool_name}: {arguments}")

    def print_tool_result(self, tool_name: str, result: str, truncated: bool = False) -> None:
        """Print tool result."""
        if self._monitor_mode and self._sticky_footer:
            result_text = result[:200] + "..." if len(result) > 200 else result
            if truncated:
                result_text += " [truncated]"
            self._append_chat_text(f"[TOOL RESULT] {tool_name}: {result_text}\n")
            self._render_monitor_view()
            return
        self._prepare_output()
        result_text = result[:200] + "..." if len(result) > 200 else result
        if truncated:
            result_text += " [truncated]"
        print(f"[TOOL RESULT] {tool_name}: {result_text}")

    def _reset_scroll_region(self) -> None:
        """Restore full-screen scrolling."""
        if not self._sticky_footer:
            return
        sys.stdout.write("\033[r")
        sys.stdout.flush()

    def _layout_metrics(self) -> dict[str, int]:
        """Compute fixed layout row positions."""
        size = shutil.get_terminal_size(fallback=(120, 24))
        width = max(size.columns, 40)
        min_rows = self._system_rows + self._status_rows + self._prompt_rows + 2
        rows = max(size.lines, min_rows)
        reserved = self._system_rows + self._status_rows + self._prompt_rows
        output_bottom = max(1, rows - reserved)
        system_start = output_bottom + 1
        status_start = system_start + self._system_rows
        prompt_row = status_start + self._status_rows
        return {
            "width": width,
            "rows": rows,
            "output_bottom": output_bottom,
            "system_start": system_start,
            "status_start": status_start,
            "prompt_row": prompt_row,
        }

    @staticmethod
    def _fit_line(text: str, width: int) -> str:
        """Trim and pad a line to terminal width."""
        return text[:width].ljust(width)

    def is_monitor_enabled(self) -> bool:
        """Whether split monitor mode is enabled."""
        return self._monitor_mode

    def _clip_monitor_buffer(self, text: str) -> str:
        """Keep monitor buffers bounded."""
        if len(text) <= self._max_monitor_buffer_chars:
            return text
        return text[-self._max_monitor_buffer_chars :]

    def _append_chat_text(self, text: str) -> None:
        """Append text to chat pane buffer."""
        self._chat_output_text = self._clip_monitor_buffer(self._chat_output_text + text)

    def _append_tool_text(self, text: str) -> None:
        """Append text to tool pane buffer."""
        self._tool_output_text = self._clip_monitor_buffer(self._tool_output_text + text)

    @staticmethod
    def _wrap_plain_text(text: str, width: int) -> list[str]:
        """Wrap plain text to fit pane width."""
        if width <= 1:
            return [""]
        if not text:
            return []

        wrapped: list[str] = []
        ends_with_newline = text.endswith("\n")
        for raw_line in text.splitlines():
            if raw_line == "":
                wrapped.append("")
                continue
            parts = textwrap.wrap(
                raw_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            wrapped.extend(parts or [""])
        if ends_with_newline:
            wrapped.append("")
        return wrapped

    def _render_monitor_view(self) -> None:
        """Render split monitor view (chat left, tool raw output right)."""
        if not (self._sticky_footer and self._monitor_mode):
            return

        m = self._layout_metrics()
        width = m["width"]
        output_rows = m["output_bottom"]

        if width < 60:
            left_width = max(10, width)
            right_width = 0
        else:
            left_width = (width - 3) // 2
            right_width = width - left_width - 3

        left_lines = self._wrap_plain_text(self._chat_output_text, left_width)
        right_lines = self._wrap_plain_text(self._tool_output_text, right_width) if right_width > 0 else []

        left_view = left_lines[-output_rows:]
        right_view = right_lines[-output_rows:] if right_width > 0 else []

        sep = "\033[90m|\033[0m" if self._ansi_enabled else "|"
        sys.stdout.write("\0337")
        sys.stdout.write(f"\033[1;{m['output_bottom']}r")
        for row in range(1, output_rows + 1):
            left = left_view[row - 1] if row - 1 < len(left_view) else ""
            if right_width > 0:
                right = right_view[row - 1] if row - 1 < len(right_view) else ""
                line = f"{left[:left_width].ljust(left_width)} {sep} {right[:right_width].ljust(right_width)}"
            else:
                line = left[:left_width].ljust(left_width)
            sys.stdout.write(f"\033[{row};1H\033[2K{line}")
        sys.stdout.write("\0338")
        sys.stdout.flush()

    def set_monitor_mode(self, enabled: bool) -> None:
        """Enable or disable split monitor view."""
        self._monitor_mode = enabled
        if not self._sticky_footer:
            return
        if not enabled:
            m = self._layout_metrics()
            sys.stdout.write("\0337")
            sys.stdout.write(f"\033[1;{m['output_bottom']}r")
            for row in range(1, m["output_bottom"] + 1):
                sys.stdout.write(f"\033[{row};1H\033[2K")
            sys.stdout.write("\0338")
            sys.stdout.flush()
        self._render_monitor_view()
        self._render_footer(force=True)

    def clear_monitor_tool_output(self) -> None:
        """Clear tool output pane content."""
        self._tool_output_text = ""
        self._render_monitor_view()

    def append_tool_output(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        raw_output: str,
        render: bool = True,
    ) -> None:
        """Append raw tool output entry to monitor pane buffer."""
        args = arguments or {}
        args_text = ""
        if args:
            try:
                args_text = json.dumps(args, ensure_ascii=True, sort_keys=True)
            except Exception:
                args_text = str(args)
        header = f"{tool_name} {args_text}".strip()
        body = raw_output if raw_output else "[no output]"
        self._append_tool_text(f"{header}\n{body}\n\n")
        if render:
            self._render_monitor_view()

    def load_monitor_tool_output_from_session(self, messages: list[dict[str, Any]]) -> None:
        """Rebuild monitor tool pane from session history."""
        self._tool_output_text = ""
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            self.append_tool_output(
                tool_name=str(msg.get("tool_name") or "tool"),
                arguments=msg.get("tool_arguments") if isinstance(msg.get("tool_arguments"), dict) else None,
                raw_output=str(msg.get("content", "")),
                render=False,
            )
        self._render_monitor_view()

    def append_system_line(self, text: str) -> None:
        """Append one system/log line to the fixed system panel."""
        clean = text.rstrip()
        if not clean:
            return
        self._system_lines.append(clean)
        if self._assistant_output_active:
            self._footer_dirty = True
            return
        self._render_footer(force=True)

    def set_runtime_status(self, status: str) -> None:
        """Update the runtime status shown in system panel."""
        allowed = {"user input", "thinking", "running script", "waiting", "streaming"}
        normalized = status.strip().lower()
        if normalized in allowed:
            self._runtime_status = normalized
        else:
            self._runtime_status = "waiting"
        if self._assistant_output_active:
            self._footer_dirty = True
            return
        self._render_footer(force=True)

    def can_capture_escape(self) -> bool:
        """Whether ESC key capture is available on this terminal."""
        return self._sticky_footer and os.name == "posix" and sys.stdin.isatty()

    async def wait_for_escape(self) -> bool:
        """Wait asynchronously for ESC key press."""
        if not self.can_capture_escape():
            await asyncio.sleep(3600)
            return False

        fd = sys.stdin.fileno()
        loop = asyncio.get_running_loop()
        old = termios.tcgetattr(fd)
        fut: asyncio.Future[bool] = loop.create_future()

        def _on_stdin_ready() -> None:
            try:
                ch = os.read(fd, 1)
            except Exception:
                ch = b""
            if ch == b"\x1b" and not fut.done():
                fut.set_result(True)

        try:
            tty.setcbreak(fd)
            loop.add_reader(fd, _on_stdin_ready)
            return await fut
        except asyncio.CancelledError:
            # Normal path when the active task finishes before ESC is pressed.
            return False
        finally:
            try:
                loop.remove_reader(fd)
            except Exception:
                pass
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    def _render_system_panel(self) -> None:
        """Render the fixed system panel rows."""
        if not self._sticky_footer:
            return
        m = self._layout_metrics()
        width = m["width"]
        lines = list(self._system_lines)[-self._system_log_rows :]
        padded_logs = lines + [""] * (self._system_log_rows - len(lines))

        status_row = m["system_start"]
        if self._ansi_enabled:
            label = "\033[97;44m STATUS \033[0m"
            state_styles = {
                "user input": "\033[30;106m",
                "thinking": "\033[30;103m",
                "running script": "\033[97;41m",
                "waiting": "\033[30;107m",
                "streaming": "\033[30;102m",
            }
            state_style = state_styles.get(self._runtime_status, "\033[30;107m")
            state = self._runtime_status.upper()
            sys.stdout.write(
                f"\033[{status_row};1H\033[2K{label} {state_style} {state:<14} \033[0m"
            )
        else:
            status_line = f"STATUS  {self._runtime_status}"
            sys.stdout.write(
                f"\033[{status_row};1H\033[2K{self._fit_line(status_line, width)}"
            )

        for i, line in enumerate(padded_logs, start=1):
            row = m["system_start"] + i
            content = self._fit_line(line, width)
            sys.stdout.write(f"\033[{row};1H\033[2K{content}")
        sys.stdout.flush()

    def _styled_role_prefix(self, role: str) -> str | None:
        """Return colored role prefix when ANSI output is enabled."""
        if not self._ansi_enabled:
            return None
        styles = {
            "user": "\033[97;44m [USER] \033[0m",
            "assistant": "\033[30;102m [ASSISTANT] \033[0m",
            "system": "\033[30;103m [SYSTEM] \033[0m",
            "tool": "\033[30;106m [TOOL] \033[0m",
        }
        return styles.get(role.lower(), f"[{role.upper()}]")

    def _render_footer(self, force: bool = False) -> None:
        """Render fixed system + status footer with FPS limit."""
        if not self._sticky_footer:
            return
        if self._assistant_output_active and not force:
            self._footer_dirty = True
            return
        now = time.monotonic()
        if not force and (now - self._last_footer_render) < self._footer_min_interval:
            return
        self._last_footer_render = now
        self._footer_dirty = False

        # Preserve output cursor so footer refresh never steals where content is printed.
        sys.stdout.write("\0337")
        m = self._layout_metrics()
        width = m["width"]
        sys.stdout.write(f"\033[1;{m['output_bottom']}r")
        self._render_system_panel()
        sys.stdout.write(f"\033[{m['status_start']};1H\033[2K")
        sys.stdout.write(f"\033[30;42m{self._fit_line(self._status_line_1, width)}\033[0m")
        sys.stdout.write(f"\033[{m['status_start'] + 1};1H\033[2K")
        sys.stdout.write(f"\033[30;42m{self._fit_line(self._status_line_2, width)}\033[0m")
        sys.stdout.write(f"\033[{m['prompt_row']};1H\033[2K")
        sys.stdout.write("\0338")
        sys.stdout.flush()

    def _prepare_output(self) -> None:
        """Move cursor to the output region above fixed footer."""
        if not self._sticky_footer or self._monitor_mode:
            return
        m = self._layout_metrics()
        sys.stdout.write(f"\033[1;{m['output_bottom']}r")
        sys.stdout.write(f"\033[{m['output_bottom']};1H")
        sys.stdout.flush()

    def print_status_line(
        self,
        last_usage: dict[str, int] | None,
        total_usage: dict[str, int] | None,
        last_exec_seconds: float | None,
        last_completed_at: datetime | None,
        session_id: str | None,
    ) -> None:
        """Print a two-line status footer above the input prompt."""
        now_str = datetime.now().strftime("%H:%M:%S")
        last = last_usage or {}
        total = total_usage or {}
        last_prompt = int(last.get("prompt_tokens", 0))
        last_completion = int(last.get("completion_tokens", 0))
        last_total = int(last.get("total_tokens", 0))
        all_total = int(total.get("total_tokens", 0))
        exec_str = f"{last_exec_seconds:.2f}s" if last_exec_seconds is not None else "--"
        last_time = last_completed_at.strftime("%H:%M:%S") if last_completed_at else "--"
        sid = session_id or "-"

        line1 = (
            f" TOKENS  last(p/c/t): {last_prompt}/{last_completion}/{last_total}"
            f"   total(t): {all_total}"
            f"   session: {sid} "
        )
        line2 = (
            f" TIME    last task at: {last_time}"
            f"   duration: {exec_str}"
            f"   now: {now_str} "
        )
        self._status_line_1 = line1
        self._status_line_2 = line2

        if not self._sticky_footer:
            # Green background + dark text for readability.
            m = self._layout_metrics()
            print(f"\033[30;42m{self._fit_line(line1, m['width'])}\033[0m")
            print(f"\033[30;42m{self._fit_line(line2, m['width'])}\033[0m")
            return

        self._render_footer(force=False)

    def print_config(self, config) -> None:
        """Print current configuration."""
        if self._monitor_mode and self._sticky_footer:
            text = (
                "\n=== Current Configuration ===\n"
                f"Provider: {config.model.provider}\n"
                f"Model: {config.model.model}\n"
                f"Temperature: {config.model.temperature}\n"
                f"Max tokens: {config.model.max_tokens}\n"
                f"Streaming: {config.ui.streaming}\n"
                f"Show tokens: {config.ui.show_tokens}\n"
                f"Enabled tools: {', '.join(config.tools.enabled)}\n\n"
            )
            self._append_chat_text(text)
            self._render_monitor_view()
            return
        self._prepare_output()
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
        if self._monitor_mode and self._sticky_footer:
            lines = ["\n=== Conversation History ==="]
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                if len(msg.get("content", "")) > 100:
                    content += "..."
                lines.append(f"[{i+1}] {role}: {content}")
            lines.append("")
            self._append_chat_text("\n".join(lines) + "\n")
            self._render_monitor_view()
            return
        self._prepare_output()
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
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(
                f"\nSession: {session.name}\nID: {session.id}\nMessages: {len(session.messages)}\n\n"
            )
            self._render_monitor_view()
            return
        self._prepare_output()
        print(f"\nSession: {session.name}")
        print(f"ID: {session.id}")
        print(f"Messages: {len(session.messages)}")
        print()

    def prompt(self, prompt_text: str = "> ") -> str:
        """Prompt for input."""
        if self._monitor_mode and self._sticky_footer:
            self._render_monitor_view()
        if self._sticky_footer:
            m = self._layout_metrics()
            sys.stdout.write(f"\033[{m['prompt_row']};1H\033[2K")
            sys.stdout.flush()
        value = input(prompt_text)
        if self._readline and value.strip():
            try:
                self._readline.add_history(value)
            except Exception:
                pass
        return value

    def begin_assistant_stream(self) -> None:
        """Start assistant streaming in the output area."""
        self._assistant_output_active = True
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text("[ASSISTANT] ")
            self._render_monitor_view()
            return
        self._prepare_output()
        prefix = self._styled_role_prefix("assistant")
        if prefix:
            print(f"{prefix} ", end="", flush=True)
        else:
            print("[ASSISTANT] ", end="", flush=True)

    def end_assistant_stream(self) -> None:
        """Finish assistant streaming and flush deferred footer updates."""
        if not self._assistant_output_active:
            return
        if self._monitor_mode and self._sticky_footer:
            if not self._chat_output_text.endswith("\n"):
                self._append_chat_text("\n")
            self._render_monitor_view()
        self._assistant_output_active = False
        if self._sticky_footer:
            self._render_footer(force=True)

    def complete_stream_line(self) -> None:
        """Finish current streaming line."""
        if self._monitor_mode and self._sticky_footer:
            if not self._chat_output_text.endswith("\n"):
                self._append_chat_text("\n")
            self._render_monitor_view()
            return
        print("\n")

    def print_blank_line(self) -> None:
        """Print an empty line in the active output mode."""
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text("\n")
            self._render_monitor_view()
            return
        print()

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
        elif command == "/monitor":
            monitor_arg = args.strip().lower()
            if monitor_arg == "on":
                return "MONITOR_ON"
            if monitor_arg == "off":
                return "MONITOR_OFF"
            self.print_error("Usage: /monitor on|off")
            return None
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
        atexit.register(_ui._reset_scroll_region)
    return _ui


def set_ui(ui: TerminalUI) -> None:
    """Set the global UI instance."""
    global _ui
    _ui = ui
