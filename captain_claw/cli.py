"""CLI framework for Captain Claw."""

import atexit
import asyncio
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import termios
import textwrap
import time
import tty
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
            "/session",
            "/sessions",
            "/models",
            "/runin",
            "/config",
            "/history",
            "/compact",
            "/cron",
            "/todo",
            "/contacts",
            "/scripts",
            "/apis",
            "/planning",
            "/pipeline",
            "/monitor",
            "/scroll",
            "/skills",
            "/skill",
            "/approve",
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
        self._monitor_full_output = bool(getattr(self.config.ui, "monitor_full_output", False))
        self._monitor_chat_scroll_offset = 0
        self._monitor_tool_scroll_offset = 0
        
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
  /new [name]     - Start a new session (optionally named)
  /session        - Show active session info
  /session list   - List recent sessions
  /session new [name] - Create and switch to a new session
  /session switch <id|name|#index> - Switch to a session
  /session rename <new-name> - Rename the active session
  /session description <text> - Set active session description
  /session description auto - Auto-generate description from session context
  /session export [chat|monitor|pipeline|pipeline-summary|all] - Export active session history views to files
  /session protect on|off - Protect/unprotect current session memory reset
  /session model - Show active session model selection
  /session model list - List allowed models from config
  /session model <id|#index|provider:model|default> - Set session model
  /session procreate <id|name|#index> <id|name|#index> <new-name> - Merge two sessions into a new one
  /session run <id|name|#index> <prompt> - Run one prompt in another session, then return
  /session queue - Show active session follow-up queue settings
  /session queue mode <steer|followup|collect|steer-backlog|interrupt|queue> - Set queue mode
  /session queue debounce <ms> - Set queue debounce in milliseconds
  /session queue cap <n> - Set max queued follow-ups per session
  /session queue drop <old|new|summarize> - Set queue overflow policy
  /session queue clear - Clear pending queued follow-ups for active session
  /models         - List allowed models from config
  /runin <id|name|#index> <prompt> - Alias for /session run
  /sessions       - List recent sessions
  /config         - Show configuration
  /history        - Show conversation history
  /compact        - Manually compact long session history
  /cron "<task>"  - Run one-off task through Captain Claw runtime/guards
  /cron run script <path> - Run existing script file with guards
  /cron run tool <path> - Run existing tool file with guards
  /cron add every <Nm|Nh> <task|script|tool ...> - Schedule recurring pseudo-cron job
  /cron add daily <HH:MM> <task|script|tool ...> - Schedule daily pseudo-cron job
  /cron add weekly <day> <HH:MM> <task|script|tool ...> - Schedule weekly pseudo-cron job
  /cron run <job-id|#index> - Run existing cron job immediately
  /cron list      - List active cron jobs
  /cron history <job-id|#index> - Show stored cron chat/monitor history
  /cron pause <job-id|#index> - Pause cron job
  /cron resume <job-id|#index> - Resume cron job
  /cron remove <job-id|#index> - Remove cron job
  /todo           - List active to-do items
  /todo add <text> - Add a to-do item (default: responsible=human)
  /todo done <id|#index> - Mark to-do item as done
  /todo remove <id|#index> - Remove a to-do item
  /todo assign bot|human <id|#index> - Reassign responsible party
  /contacts       - List contacts (address book)
  /contacts add <name> - Add a new contact
  /contacts info <id|#index|name> - Show contact details
  /contacts search <query> - Search contacts by name/org/email
  /contacts update <id|#index|name> <field=value ...> - Update contact fields
  /contacts importance <id|#index|name> <1-10> - Set contact importance (pins value)
  /contacts remove <id|#index|name> - Remove a contact
  /scripts        - List remembered scripts
  /scripts add <name> <path> - Register a script
  /scripts info <id|#index|name> - Show script details
  /scripts search <query> - Search scripts
  /scripts update <id|#index|name> <field=value ...> - Update script fields
  /scripts remove <id|#index|name> - Remove a script entry
  /apis           - List remembered APIs
  /apis add <name> <base_url> - Register an API
  /apis info <id|#index|name> - Show API details
  /apis search <query> - Search APIs
  /apis update <id|#index|name> <field=value ...> - Update API fields
  /apis remove <id|#index|name> - Remove an API entry
  /pipeline loop|contracts - Set execution pipeline mode (simple loop vs contract gate)
  /planning on|off - Legacy alias for /pipeline contracts|loop
  /skills         - List currently available user-invocable skills
  /skill <name> [args] - Run a specific skill manually
  /skill search <criteria> - Search public OpenClaw skills catalog (top 10)
  /skill install <github-url> - Install a skill from GitHub into managed skills
  /skill install <skill-name> [install-id] - Install dependencies declared by a skill
  /<skill-command> [args] - Direct alias for a discovered skill command
  /approve user <telegram|slack|discord> <token> - Approve pending chat user pairing token
  /monitor on     - Enable split monitor view
  /monitor off    - Disable split monitor view
  /monitor trace on|off - Enable/disable full intermediate LLM trace logging
  /monitor pipeline on|off - Enable/disable compact pipeline-only trace logging
  /monitor full on|off - Enable/disable raw monitor tool output rendering
  /scroll <chat|monitor> <up|down|pageup|pagedown|top|bottom> [n] - Scroll one monitor pane independently
  /scroll status - Show monitor pane scroll positions
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
            prefix = self._monitor_role_prefix(role)
            self._append_chat_text(f"{prefix}{content}\n")
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
                "user": lambda x: f"> {x}",
                "assistant": lambda x: f":) {x}",
                "system": lambda x: f"! {x}",
                "tool": lambda x: f"$ {x}",
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

    @staticmethod
    def _compose_footer_line(left: str, right: str, width: int) -> str:
        """Compose a footer line with right-aligned metadata."""
        left_text = left.rstrip()
        right_text = right.strip()
        if not right_text:
            return left_text
        min_gap = 3
        free = width - len(left_text) - len(right_text)
        if free >= min_gap:
            return f"{left_text}{' ' * free}{right_text}"
        keep_left = max(0, width - len(right_text) - min_gap)
        clipped_left = left_text[:keep_left].rstrip() if keep_left > 0 else ""
        return f"{clipped_left}{' ' * min_gap}{right_text}"

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

    def _monitor_pane_layout(self) -> tuple[int, int, int]:
        """Return left pane width, right pane width, and content rows."""
        m = self._layout_metrics()
        width = m["width"]
        output_rows = m["output_bottom"]
        if width < 60:
            left_width = max(10, width)
            right_width = 0
        else:
            left_width = (width - 3) // 2
            right_width = width - left_width - 3
        content_rows = max(0, output_rows - 1)
        return left_width, right_width, content_rows

    @staticmethod
    def _slice_scrolled_lines(
        lines: list[str],
        content_rows: int,
        offset_from_bottom: int,
    ) -> tuple[list[str], int, int]:
        """Slice a pane view with an explicit offset from latest output."""
        if content_rows <= 0:
            return [], 0, 0
        max_offset = max(0, len(lines) - content_rows)
        clamped_offset = min(max(0, offset_from_bottom), max_offset)
        end = len(lines) - clamped_offset
        start = max(0, end - content_rows)
        return lines[start:end], clamped_offset, max_offset

    def _monitor_scroll_limits(self) -> tuple[int, int]:
        """Return max scroll offsets for chat and monitor panes."""
        left_width, right_width, content_rows = self._monitor_pane_layout()
        if content_rows <= 0:
            return 0, 0
        chat_lines = self._wrap_plain_text(self._chat_output_text, left_width)
        chat_max = max(0, len(chat_lines) - content_rows)
        if right_width <= 0:
            return chat_max, 0
        tool_lines = self._wrap_plain_text(self._tool_output_text, right_width)
        tool_max = max(0, len(tool_lines) - content_rows)
        return chat_max, tool_max

    def _clamp_monitor_scroll_offsets(self) -> tuple[int, int]:
        """Clamp pane scroll offsets to current buffer/viewport bounds."""
        chat_max, tool_max = self._monitor_scroll_limits()
        self._monitor_chat_scroll_offset = min(max(0, self._monitor_chat_scroll_offset), chat_max)
        self._monitor_tool_scroll_offset = min(max(0, self._monitor_tool_scroll_offset), tool_max)
        return chat_max, tool_max

    def get_monitor_scroll_state(self) -> dict[str, int | bool]:
        """Return monitor pane scroll offsets and maximums."""
        left_width, right_width, _ = self._monitor_pane_layout()
        chat_max, tool_max = self._clamp_monitor_scroll_offsets()
        return {
            "chat_offset": self._monitor_chat_scroll_offset,
            "chat_max_offset": chat_max,
            "monitor_offset": self._monitor_tool_scroll_offset,
            "monitor_max_offset": tool_max,
            "monitor_visible": right_width > 0 and left_width > 0,
        }

    def describe_monitor_scroll(self) -> str:
        """Human-readable monitor pane scroll status."""
        state = self.get_monitor_scroll_state()
        chat_status = f"chat={state['chat_offset']}/{state['chat_max_offset']}"
        if not bool(state["monitor_visible"]):
            return f"{chat_status}; monitor pane hidden (narrow terminal)"
        return (
            f"{chat_status}; monitor={state['monitor_offset']}/{state['monitor_max_offset']}"
        )

    def scroll_monitor_pane(self, pane: str, action: str, amount: int = 1) -> tuple[bool, str]:
        """Adjust scroll offset for one monitor pane."""
        if not self._monitor_mode:
            return False, "Enable monitor mode first: /monitor on"

        normalized_pane = pane.strip().lower()
        if normalized_pane in {"tool", "right"}:
            normalized_pane = "monitor"
        if normalized_pane not in {"chat", "monitor"}:
            return False, "Invalid pane. Use chat or monitor."

        normalized_action = action.strip().lower()
        if normalized_action not in {"up", "down", "pageup", "pagedown", "top", "bottom"}:
            return False, "Invalid action. Use up|down|pageup|pagedown|top|bottom."

        step = max(1, int(amount))
        _, _, content_rows = self._monitor_pane_layout()
        if normalized_action in {"pageup", "pagedown"}:
            step = max(1, content_rows) * step

        if normalized_pane == "chat":
            current = self._monitor_chat_scroll_offset
            max_offset, _ = self._monitor_scroll_limits()
        else:
            current = self._monitor_tool_scroll_offset
            _, max_offset = self._monitor_scroll_limits()

        if normalized_action in {"up", "pageup"}:
            new_offset = current + step
        elif normalized_action in {"down", "pagedown"}:
            new_offset = current - step
        elif normalized_action == "top":
            new_offset = max_offset
        else:
            new_offset = 0

        clamped_offset = min(max(0, new_offset), max_offset)
        if normalized_pane == "chat":
            self._monitor_chat_scroll_offset = clamped_offset
        else:
            self._monitor_tool_scroll_offset = clamped_offset

        if self._monitor_mode and self._sticky_footer:
            self._render_monitor_view()

        return True, (
            f"Monitor scroll updated ({normalized_pane}={clamped_offset}/{max_offset}); "
            f"{self.describe_monitor_scroll()}"
        )

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
        output_rows = m["output_bottom"]
        left_width, right_width, content_rows = self._monitor_pane_layout()

        left_lines = self._wrap_plain_text(self._chat_output_text, left_width)
        right_lines = self._wrap_plain_text(self._tool_output_text, right_width) if right_width > 0 else []

        left_view, self._monitor_chat_scroll_offset, _ = self._slice_scrolled_lines(
            left_lines, content_rows, self._monitor_chat_scroll_offset
        )
        if right_width > 0:
            (
                right_view,
                self._monitor_tool_scroll_offset,
                _,
            ) = self._slice_scrolled_lines(right_lines, content_rows, self._monitor_tool_scroll_offset)
        else:
            right_view = []
            self._monitor_tool_scroll_offset = 0

        chat_header_label = (
            f" Chat ^{self._monitor_chat_scroll_offset} "
            if self._monitor_chat_scroll_offset > 0
            else " Chat "
        )
        monitor_header_label = (
            f" Monitor ^{self._monitor_tool_scroll_offset} "
            if self._monitor_tool_scroll_offset > 0
            else " Monitor "
        )

        sep = "\033[90m|\033[0m" if self._ansi_enabled else "|"
        if self._ansi_enabled:
            left_header = f"\033[30;42m{self._fit_line(chat_header_label, left_width)}\033[0m"
            if right_width > 0:
                right_header = f"\033[30;42m{self._fit_line(monitor_header_label, right_width)}\033[0m"
                header_line = f"{left_header} {sep} {right_header}"
            else:
                header_line = left_header
        else:
            left_header = self._fit_line(chat_header_label, left_width)
            if right_width > 0:
                right_header = self._fit_line(monitor_header_label, right_width)
                header_line = f"{left_header} | {right_header}"
            else:
                header_line = left_header

        sys.stdout.write("\0337")
        sys.stdout.write(f"\033[1;{m['output_bottom']}r")
        for row in range(1, output_rows + 1):
            if row == 1:
                line = header_line
                sys.stdout.write(f"\033[{row};1H\033[2K{line}")
                continue
            content_idx = row - 2
            left = left_view[content_idx] if content_idx < len(left_view) else ""
            if right_width > 0:
                right = right_view[content_idx] if content_idx < len(right_view) else ""
                line = f"{left[:left_width].ljust(left_width)} {sep} {right[:right_width].ljust(right_width)}"
            else:
                line = left[:left_width].ljust(left_width)
            sys.stdout.write(f"\033[{row};1H\033[2K{line}")
        sys.stdout.write("\0338")
        sys.stdout.flush()

    def set_monitor_mode(self, enabled: bool) -> None:
        """Enable or disable split monitor view."""
        self._monitor_mode = enabled
        if enabled:
            self._monitor_chat_scroll_offset = 0
            self._monitor_tool_scroll_offset = 0
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

    def set_monitor_full_output(self, enabled: bool) -> None:
        """Enable/disable raw full tool output rendering in monitor pane."""
        self._monitor_full_output = bool(enabled)
        if self._monitor_mode:
            self._render_monitor_view()

    def clear_monitor_tool_output(self) -> None:
        """Clear tool output pane content."""
        self._tool_output_text = ""
        self._monitor_tool_scroll_offset = 0
        self._render_monitor_view()

    def clear_monitor_chat_output(self) -> None:
        """Clear chat output pane content."""
        self._chat_output_text = ""
        self._monitor_chat_scroll_offset = 0
        self._render_monitor_view()

    @staticmethod
    def _split_web_fetch_payload(raw_output: str) -> str:
        """Return extracted content portion from web_fetch output."""
        lines = (raw_output or "").splitlines()
        idx = 0
        while idx < len(lines):
            if re.match(r"^\[[^:\]]+:\s*.*\]$", lines[idx].strip()):
                idx += 1
                continue
            break
        if idx < len(lines) and not lines[idx].strip():
            idx += 1
        content = "\n".join(lines[idx:]).strip()
        return content or (raw_output or "").strip()

    @staticmethod
    def _summarize_text(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
        """Build a short summary from first 1-2 sentences."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return "No readable text extracted."
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        picked = sentences[:max_sentences] if sentences else [cleaned]
        summary = " ".join(picked)
        if len(summary) > max_chars:
            summary = summary[:max_chars].rstrip() + "..."
        return summary

    def _format_monitor_tool_body(self, tool_name: str, raw_output: str) -> str:
        """Render monitor body text for a tool output."""
        text = raw_output if raw_output else "[no output]"
        if self._monitor_full_output:
            return text
        if tool_name != "web_fetch":
            return text

        used_text = self._split_web_fetch_payload(text)
        summary = self._summarize_text(used_text)
        used_kb = len(used_text.encode("utf-8")) / 1024.0
        return f"Summary: {summary}\nUsed text: {used_kb:.1f} kB"

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
        if tool_name == "cron" or bool(args.get("cron")):
            header = f"[CRON] {header}"
        body = self._format_monitor_tool_body(tool_name, raw_output).rstrip("\n")
        self._append_tool_text(f"{header}\n{body}\n\n")
        if render:
            self._render_monitor_view()

    def load_monitor_tool_output_from_session(self, messages: list[dict[str, Any]]) -> None:
        """Rebuild monitor tool pane from session history."""
        self._tool_output_text = ""
        self._monitor_tool_scroll_offset = 0
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

    def _monitor_role_prefix(self, role: str) -> str:
        """Return compact role prefix for monitor chat pane."""
        normalized = role.lower()
        if normalized == "user":
            return "> "
        if normalized == "assistant":
            return ":) "
        if normalized == "system":
            return "! "
        if normalized == "tool":
            return "$ "
        return f"[{role.upper()}] "

    def _styled_role_prefix(self, role: str) -> str | None:
        """Return colored role prefix when ANSI output is enabled."""
        if not self._ansi_enabled:
            return None
        styles = {
            "user": "\033[32m>\033[0m",
            "assistant": "\033[96m:)\033[0m",
            "system": "\033[30;103m ! \033[0m",
            "tool": "\033[30;106m $ \033[0m",
        }
        return styles.get(role.lower(), f"[{role.upper()}] ")

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
        context_window: dict[str, int | float] | None = None,
        model_details: dict[str, Any] | None = None,
    ) -> None:
        """Print a two-line status footer above the input prompt."""
        m = self._layout_metrics()
        width = m["width"]
        now_str = datetime.now().strftime("%H:%M:%S")
        last = last_usage or {}
        total = total_usage or {}
        ctx = context_window or {}
        model = model_details or {}
        last_prompt = int(last.get("prompt_tokens", 0))
        last_completion = int(last.get("completion_tokens", 0))
        last_total = int(last.get("total_tokens", 0))
        all_total = int(total.get("total_tokens", 0))
        exec_str = f"{last_exec_seconds:.2f}s" if last_exec_seconds is not None else "--"
        last_time = last_completed_at.strftime("%H:%M:%S") if last_completed_at else "--"
        sid = session_id or "-"

        ctx_budget = int(ctx.get("context_budget_tokens", 0)) if ctx.get("context_budget_tokens") is not None else 0
        ctx_used = int(ctx.get("prompt_tokens", 0)) if ctx.get("prompt_tokens") is not None else 0
        if ctx_budget > 0:
            util = float(ctx.get("utilization", (ctx_used / ctx_budget)))
            util_pct = max(0.0, util * 100.0)
            ctx_text = f"CTX {ctx_used:,}/{ctx_budget:,} ({util_pct:.1f}%)"
        else:
            ctx_text = "CTX n/a"

        provider = str(model.get("provider", "")).strip()
        model_name = str(model.get("model", "")).strip()
        temperature = model.get("temperature")
        gen_max = model.get("max_tokens")
        if model_name or provider:
            pieces = [p for p in [provider, model_name] if p]
            detail_bits: list[str] = []
            if temperature is not None:
                detail_bits.append(f"t={temperature}")
            if gen_max is not None:
                detail_bits.append(f"gen={int(gen_max):,}")
            suffix = f" [{' '.join(detail_bits)}]" if detail_bits else ""
            model_text = f"MODEL {'/'.join(pieces)}{suffix}"
        else:
            model_text = "MODEL n/a"

        line1_left = (
            f" TOKENS  last(p/c/t): {last_prompt}/{last_completion}/{last_total}"
            f"   total(t): {all_total}"
            f"   session: {sid} "
        )
        line2_left = (
            f" TIME    last task at: {last_time}"
            f"   duration: {exec_str}"
            f"   now: {now_str} "
        )
        line1 = self._compose_footer_line(line1_left, ctx_text, width)
        line2 = self._compose_footer_line(line2_left, model_text, width)
        self._status_line_1 = line1
        self._status_line_2 = line2

        if not self._sticky_footer:
            # Green background + dark text for readability.
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
                f"Guard input: enabled={config.guards.input.enabled}, level={config.guards.input.level}\n"
                f"Guard output: enabled={config.guards.output.enabled}, level={config.guards.output.level}\n"
                f"Guard script/tool: enabled={config.guards.script_tool.enabled}, level={config.guards.script_tool.level}\n"
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
        print(f"Guard input: enabled={config.guards.input.enabled}, level={config.guards.input.level}")
        print(f"Guard output: enabled={config.guards.output.enabled}, level={config.guards.output.level}")
        print(f"Guard script/tool: enabled={config.guards.script_tool.enabled}, level={config.guards.script_tool.level}")
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
        metadata = session.metadata if isinstance(getattr(session, "metadata", None), dict) else {}
        description = str(metadata.get("description", "")).strip()
        desc_line = f"Description: {description}\n" if description else ""
        model_selection = metadata.get("model_selection") if isinstance(metadata.get("model_selection"), dict) else None
        protection_meta = (
            metadata.get("memory_protection")
            if isinstance(metadata.get("memory_protection"), dict)
            else None
        )
        protection_line = "Protection: on\n" if protection_meta and protection_meta.get("enabled") else ""
        model_line = ""
        if model_selection:
            model_provider = str(model_selection.get("provider", "")).strip()
            model_name = str(model_selection.get("model", "")).strip()
            model_id = str(model_selection.get("id", "")).strip()
            id_part = f" [{model_id}]" if model_id else ""
            if model_provider and model_name:
                model_line = f"Model: {model_provider}/{model_name}{id_part}\n"
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(
                f"\nSession: {session.name}\nID: {session.id}\nMessages: {len(session.messages)}\n"
                f"{model_line}{protection_line}{desc_line}\n"
            )
            self._render_monitor_view()
            return
        self._prepare_output()
        print(f"\nSession: {session.name}")
        print(f"ID: {session.id}")
        print(f"Messages: {len(session.messages)}")
        if model_line:
            print(model_line.strip())
        if protection_line:
            print(protection_line.strip())
        if description:
            print(f"Description: {description}")
        print()

    def print_session_list(self, sessions: list[Any], current_session_id: str | None = None) -> None:
        """Print recent session list."""
        lines = ["\n=== Sessions ==="]
        if not sessions:
            lines.append("(no sessions found)")
        for idx, session in enumerate(sessions, start=1):
            marker = "*" if session.id == current_session_id else " "
            metadata = session.metadata if isinstance(getattr(session, "metadata", None), dict) else {}
            description = str(metadata.get("description", "")).strip()
            if len(description) > 90:
                description = description[:90].rstrip() + "..."
            description_part = f" | desc={description}" if description else ""
            lines.append(
                f"{marker} [{idx}] {session.name} | id={session.id} | messages={len(session.messages)} | updated={session.updated_at}{description_part}"
            )
        lines.append("")

        content = "\n".join(lines) + "\n"
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(content)
            self._render_monitor_view()
            return
        self._prepare_output()
        print(content, end="")

    def print_cron_jobs(self, jobs: list[Any]) -> None:
        """Print pseudo-cron jobs."""
        lines = ["\n=== Active Cron Jobs ==="]
        if not jobs:
            lines.append("(no jobs found)")
        for idx, job in enumerate(jobs, start=1):
            enabled = bool(getattr(job, "enabled", False))
            marker = "*" if enabled else "x"
            job_id = str(getattr(job, "id", ""))
            kind = str(getattr(job, "kind", "prompt"))
            session_id = str(getattr(job, "session_id", ""))
            next_run_at = str(getattr(job, "next_run_at", ""))
            last_status = str(getattr(job, "last_status", ""))
            schedule = getattr(job, "schedule", {})
            schedule_text = str(schedule.get("_text", "")) if isinstance(schedule, dict) else ""
            payload = getattr(job, "payload", {})
            chat_history = getattr(job, "chat_history", [])
            monitor_history = getattr(job, "monitor_history", [])
            chat_count = len(chat_history) if isinstance(chat_history, list) else 0
            monitor_count = len(monitor_history) if isinstance(monitor_history, list) else 0
            preview = ""
            if isinstance(payload, dict):
                preview = str(payload.get("text") or payload.get("path") or "")[:80].strip()
            preview_part = f" | payload={preview}" if preview else ""
            schedule_part = f" | schedule={schedule_text}" if schedule_text else ""
            history_part = f" | history(chat={chat_count},monitor={monitor_count})"
            lines.append(
                f"{marker} [{idx}] id={job_id} | kind={kind} | session={session_id}{schedule_part} | next={next_run_at} | status={last_status}{history_part}{preview_part}"
            )
        lines.append("")

        content = "\n".join(lines) + "\n"
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(content)
            self._render_monitor_view()
            return
        self._prepare_output()
        print(content, end="")

    def print_cron_job_history(self, job: Any) -> None:
        """Print stored cron history for one job."""
        job_id = str(getattr(job, "id", ""))
        kind = str(getattr(job, "kind", "prompt"))
        session_id = str(getattr(job, "session_id", ""))
        chat_history = getattr(job, "chat_history", [])
        monitor_history = getattr(job, "monitor_history", [])
        llm_outputs: list[dict[str, Any]] = []
        if isinstance(chat_history, list):
            llm_outputs = [item for item in chat_history if str(item.get("role", "")).lower() == "assistant"]
        lines = [
            "",
            "=== Cron Job History ===",
            f"id={job_id} | kind={kind} | session={session_id}",
            "",
            "--- LLM Chat Output ---",
        ]
        if not llm_outputs:
            lines.append("(no assistant output recorded)")
        else:
            for idx, item in enumerate(llm_outputs, start=1):
                ts = str(item.get("timestamp", ""))
                content = str(item.get("content", ""))
                lines.append(f"[{idx}] {ts}")
                lines.append(content if content else "(empty)")
                lines.append("")
        lines.extend(["--- Monitor Output ---"])
        if not isinstance(monitor_history, list) or not monitor_history:
            lines.append("(empty)")
        else:
            for idx, item in enumerate(monitor_history, start=1):
                ts = str(item.get("timestamp", ""))
                step = str(item.get("step", ""))
                output = str(item.get("output", ""))
                rest = {k: v for k, v in item.items() if k not in {"timestamp", "step", "output"}}
                rest_text = ""
                if rest:
                    try:
                        rest_text = json.dumps(rest, ensure_ascii=True, sort_keys=True)
                    except Exception:
                        rest_text = str(rest)
                lines.append(f"[{idx}] {ts} step={step} {rest_text}".rstrip())
                if output:
                    lines.append(output)
                    lines.append("")
        lines.append("")

        content = "\n".join(lines) + "\n"
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(content)
            self._render_monitor_view()
            return
        self._prepare_output()
        print(content, end="")

    def print_model_list(
        self,
        models: list[dict[str, Any]],
        active_model: dict[str, Any] | None = None,
    ) -> None:
        """Print allowed model list with active marker."""
        lines = ["\n=== Allowed Models ==="]
        active_provider = str((active_model or {}).get("provider", "")).strip().lower()
        active_model_name = str((active_model or {}).get("model", "")).strip().lower()
        active_id = str((active_model or {}).get("id", "")).strip().lower()

        if not models:
            lines.append("(no models configured)")
        for idx, model in enumerate(models, start=1):
            model_id = str(model.get("id", "")).strip()
            provider = str(model.get("provider", "")).strip()
            model_name = str(model.get("model", "")).strip()
            source = str((active_model or {}).get("source", "")).strip()
            is_active = (
                (model_id and model_id.lower() == active_id)
                or (
                    provider.lower() == active_provider
                    and model_name.lower() == active_model_name
                )
            )
            marker = "*" if is_active else " "
            source_part = f" | active_source={source}" if is_active and source else ""
            lines.append(
                f"{marker} [{idx}] {model_id} | provider={provider} | model={model_name}{source_part}"
            )
        lines.append("")

        content = "\n".join(lines) + "\n"
        if self._monitor_mode and self._sticky_footer:
            self._append_chat_text(content)
            self._render_monitor_view()
            return
        self._prepare_output()
        print(content, end="")

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
            self._append_chat_text(self._monitor_role_prefix("assistant"))
            self._render_monitor_view()
            return
        self._prepare_output()
        prefix = self._styled_role_prefix("assistant")
        if prefix:
            print(f"{prefix} ", end="", flush=True)
        else:
            print(":) ", end="", flush=True)

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
            name = args.strip()
            if not name:
                return "NEW"
            return f"NEW:{name}"
        elif command == "/sessions":
            return "SESSIONS"
        elif command == "/models":
            return "MODELS"
        elif command == "/session":
            selector = args.strip()
            if not selector:
                return "SESSION_INFO"
            session_parts = selector.split(None, 2)
            subcommand = session_parts[0].lower()
            if subcommand in ("list", "ls"):
                return "SESSIONS"
            if subcommand in ("new", "create"):
                name = selector[len(session_parts[0]) :].strip()
                if not name:
                    return "NEW"
                return f"NEW:{name}"
            if subcommand in ("switch", "use", "load"):
                selector_arg = selector[len(session_parts[0]) :].strip()
                if not selector_arg:
                    self.print_error("Usage: /session switch <id|name|#index>")
                    return None
                return f"SESSION_SELECT:{selector_arg}"
            if subcommand in ("rename", "name"):
                new_name = selector[len(session_parts[0]) :].strip()
                if not new_name:
                    self.print_error("Usage: /session rename <new-name>")
                    return None
                return f"SESSION_RENAME:{new_name}"
            if subcommand in ("description", "desc"):
                description_arg = selector[len(session_parts[0]) :].strip()
                if not description_arg:
                    return "SESSION_DESCRIPTION_INFO"
                parsed = description_arg.strip()
                if parsed.lower() == "auto":
                    return "SESSION_DESCRIPTION_AUTO"
                if (
                    len(parsed) >= 2
                    and parsed[0] == parsed[-1]
                    and parsed[0] in {"'", '"'}
                ):
                    parsed = parsed[1:-1].strip()
                if not parsed:
                    self.print_error("Usage: /session description <text> | /session description auto")
                    return None
                payload = json.dumps({"description": parsed}, ensure_ascii=True)
                return f"SESSION_DESCRIPTION_SET:{payload}"
            if subcommand in ("export", "dump"):
                export_arg = selector[len(session_parts[0]) :].strip().lower()
                if not export_arg:
                    return "SESSION_EXPORT:all"
                if export_arg in {"chat", "monitor", "pipeline", "pipeline-summary", "all"}:
                    return f"SESSION_EXPORT:{export_arg}"
                self.print_error("Usage: /session export [chat|monitor|pipeline|pipeline-summary|all]")
                return None
            if subcommand == "protect":
                protect_arg = selector[len(session_parts[0]) :].strip().lower()
                if protect_arg == "on":
                    return "SESSION_PROTECT_ON"
                if protect_arg == "off":
                    return "SESSION_PROTECT_OFF"
                self.print_error("Usage: /session protect on|off")
                return None
            if subcommand == "procreate":
                procreate_arg = selector[len(session_parts[0]) :].strip()
                try:
                    parsed_args = shlex.split(procreate_arg)
                except ValueError:
                    self.print_error(
                        "Usage: /session procreate <id|name|#index> <id|name|#index> <new-name>"
                    )
                    return None
                if len(parsed_args) < 3:
                    self.print_error(
                        "Usage: /session procreate <id|name|#index> <id|name|#index> <new-name>"
                    )
                    return None
                parent_one = parsed_args[0].strip()
                parent_two = parsed_args[1].strip()
                new_name = " ".join(parsed_args[2:]).strip()
                if not parent_one or not parent_two or not new_name:
                    self.print_error(
                        "Usage: /session procreate <id|name|#index> <id|name|#index> <new-name>"
                    )
                    return None
                payload = json.dumps(
                    {
                        "parent_one": parent_one,
                        "parent_two": parent_two,
                        "new_name": new_name,
                    },
                    ensure_ascii=True,
                )
                return f"SESSION_PROCREATE:{payload}"
            if subcommand == "model":
                model_arg = selector[len(session_parts[0]) :].strip()
                if not model_arg:
                    return "SESSION_MODEL_INFO"
                lowered_arg = model_arg.lower()
                if lowered_arg in {"list", "ls"}:
                    return "MODELS"
                return f"SESSION_MODEL_SET:{model_arg}"
            if subcommand == "queue":
                queue_arg = selector[len(session_parts[0]) :].strip()
                if not queue_arg:
                    return "SESSION_QUEUE_INFO"
                queue_parts = queue_arg.split(None, 1)
                queue_sub = queue_parts[0].lower()
                queue_rest = queue_parts[1].strip() if len(queue_parts) > 1 else ""
                if queue_sub == "clear":
                    return "SESSION_QUEUE_CLEAR"
                if queue_sub == "mode":
                    if not queue_rest:
                        self.print_error(
                            "Usage: /session queue mode <steer|followup|collect|steer-backlog|interrupt|queue>"
                        )
                        return None
                    return f"SESSION_QUEUE_MODE:{queue_rest}"
                if queue_sub == "debounce":
                    if not queue_rest:
                        self.print_error("Usage: /session queue debounce <ms>")
                        return None
                    return f"SESSION_QUEUE_DEBOUNCE:{queue_rest}"
                if queue_sub == "cap":
                    if not queue_rest:
                        self.print_error("Usage: /session queue cap <n>")
                        return None
                    return f"SESSION_QUEUE_CAP:{queue_rest}"
                if queue_sub == "drop":
                    if not queue_rest:
                        self.print_error("Usage: /session queue drop <old|new|summarize>")
                        return None
                    return f"SESSION_QUEUE_DROP:{queue_rest}"
                if queue_sub in {"steer", "followup", "collect", "steer-backlog", "interrupt", "queue"}:
                    return f"SESSION_QUEUE_MODE:{queue_sub}"
                self.print_error(
                    "Usage: /session queue | /session queue mode <mode> | /session queue debounce <ms> | /session queue cap <n> | /session queue drop <old|new|summarize> | /session queue clear"
                )
                return None
            if subcommand in ("run", "exec"):
                run_args = selector[len(session_parts[0]) :].strip()
                run_parts = run_args.split(None, 1)
                if len(run_parts) < 2:
                    self.print_error("Usage: /session run <id|name|#index> <prompt>")
                    return None
                selector_arg = run_parts[0].strip()
                prompt_arg = run_parts[1].strip()
                if not selector_arg or not prompt_arg:
                    self.print_error("Usage: /session run <id|name|#index> <prompt>")
                    return None
                payload = json.dumps({"selector": selector_arg, "prompt": prompt_arg}, ensure_ascii=True)
                return f"SESSION_RUN:{payload}"
            return f"SESSION_SELECT:{selector}"
        elif command == "/runin":
            run_parts = args.strip().split(None, 1)
            if len(run_parts) < 2 or not run_parts[0].strip() or not run_parts[1].strip():
                self.print_error("Usage: /runin <id|name|#index> <prompt>")
                return None
            payload = json.dumps(
                {"selector": run_parts[0].strip(), "prompt": run_parts[1].strip()},
                ensure_ascii=True,
            )
            return f"SESSION_RUN:{payload}"
        elif command == "/config":
            return "CONFIG"
        elif command == "/history":
            return "HISTORY"
        elif command == "/compact":
            return "COMPACT"
        elif command == "/cron":
            cron_arg = args.strip()
            if not cron_arg:
                self.print_error(
                    "Usage: /cron \"<task>\" | /cron add ... | /cron run ... | /cron list | /cron history <job-id|#index>"
                )
                return None
            cron_parts = cron_arg.split(None, 1)
            subcommand = cron_parts[0].lower()
            if subcommand in {"list", "ls"}:
                return "CRON_LIST"
            if subcommand in {"history", "log", "logs"}:
                job_id = cron_arg[len(cron_parts[0]) :].strip()
                if not job_id:
                    self.print_error("Usage: /cron history <job-id|#index>")
                    return None
                return f"CRON_HISTORY:{job_id}"
            if subcommand == "add":
                add_args = cron_arg[len(cron_parts[0]) :].strip()
                if not add_args:
                    self.print_error("Usage: /cron add every <Nm|Nh> <task|script|tool ...>")
                    return None
                return f"CRON_ADD:{add_args}"
            if subcommand in {"remove", "rm", "delete", "del"}:
                job_id = cron_arg[len(cron_parts[0]) :].strip()
                if not job_id:
                    self.print_error("Usage: /cron remove <job-id|#index>")
                    return None
                return f"CRON_REMOVE:{job_id}"
            if subcommand in {"pause", "disable"}:
                job_id = cron_arg[len(cron_parts[0]) :].strip()
                if not job_id:
                    self.print_error("Usage: /cron pause <job-id|#index>")
                    return None
                return f"CRON_PAUSE:{job_id}"
            if subcommand in {"resume", "enable"}:
                job_id = cron_arg[len(cron_parts[0]) :].strip()
                if not job_id:
                    self.print_error("Usage: /cron resume <job-id|#index>")
                    return None
                return f"CRON_RESUME:{job_id}"
            if subcommand == "run":
                run_args = cron_arg[len(cron_parts[0]) :].strip()
                if not run_args:
                    self.print_error("Usage: /cron run <job-id|#index> | /cron run script <path> | /cron run tool <path>")
                    return None
                payload = json.dumps({"args": run_args}, ensure_ascii=True)
                return f"CRON_RUN:{payload}"
            # One-off prompt mode: /cron "some task"
            prompt = cron_arg
            if len(prompt) >= 2 and prompt[0] == prompt[-1] and prompt[0] in {"'", '"'}:
                prompt = prompt[1:-1].strip()
            if not prompt:
                self.print_error("Usage: /cron \"<task>\"")
                return None
            payload = json.dumps({"prompt": prompt}, ensure_ascii=True)
            return f"CRON_ONEOFF:{payload}"
        elif command == "/todo":
            todo_arg = args.strip()
            if not todo_arg or todo_arg.lower() in {"list", "ls"}:
                return "TODO_LIST"
            todo_parts = todo_arg.split(None, 1)
            subcommand = todo_parts[0].lower()
            if subcommand == "add":
                text = todo_arg[len(todo_parts[0]):].strip()
                if not text:
                    self.print_error("Usage: /todo add <text>")
                    return None
                return f"TODO_ADD:{text}"
            if subcommand in {"done", "complete", "finish"}:
                selector = todo_arg[len(todo_parts[0]):].strip()
                if not selector:
                    self.print_error("Usage: /todo done <id|#index>")
                    return None
                return f"TODO_DONE:{selector}"
            if subcommand in {"remove", "rm", "delete", "del"}:
                selector = todo_arg[len(todo_parts[0]):].strip()
                if not selector:
                    self.print_error("Usage: /todo remove <id|#index>")
                    return None
                return f"TODO_REMOVE:{selector}"
            if subcommand == "assign":
                rest = todo_arg[len(todo_parts[0]):].strip()
                assign_parts = rest.split(None, 1)
                if len(assign_parts) < 2 or assign_parts[0] not in {"bot", "human"}:
                    self.print_error("Usage: /todo assign bot|human <id|#index>")
                    return None
                payload = json.dumps({"responsible": assign_parts[0], "selector": assign_parts[1]})
                return f"TODO_ASSIGN:{payload}"
            # Fallback: treat the whole argument as text for add
            return f"TODO_ADD:{todo_arg}"
        elif command == "/contacts":
            contacts_arg = args.strip()
            if not contacts_arg or contacts_arg.lower() in {"list", "ls"}:
                return "CONTACTS_LIST"
            contacts_parts = contacts_arg.split(None, 1)
            subcommand = contacts_parts[0].lower()
            rest = contacts_arg[len(contacts_parts[0]):].strip() if len(contacts_parts) > 1 else ""
            if subcommand == "add":
                if not rest:
                    self.print_error("Usage: /contacts add <name>")
                    return None
                return f"CONTACTS_ADD:{rest}"
            if subcommand == "info":
                if not rest:
                    self.print_error("Usage: /contacts info <id|#index|name>")
                    return None
                return f"CONTACTS_INFO:{rest}"
            if subcommand == "search":
                if not rest:
                    self.print_error("Usage: /contacts search <query>")
                    return None
                return f"CONTACTS_SEARCH:{rest}"
            if subcommand == "update":
                if not rest:
                    self.print_error("Usage: /contacts update <id|#index|name> <field=value ...>")
                    return None
                return f"CONTACTS_UPDATE:{rest}"
            if subcommand == "importance":
                imp_parts = rest.rsplit(None, 1)
                if len(imp_parts) < 2:
                    self.print_error("Usage: /contacts importance <id|#index|name> <1-10>")
                    return None
                try:
                    score = int(imp_parts[1])
                except ValueError:
                    self.print_error("Importance must be a number 1-10.")
                    return None
                payload = json.dumps({"selector": imp_parts[0], "importance": score})
                return f"CONTACTS_IMPORTANCE:{payload}"
            if subcommand in {"remove", "rm", "delete", "del"}:
                if not rest:
                    self.print_error("Usage: /contacts remove <id|#index|name>")
                    return None
                return f"CONTACTS_REMOVE:{rest}"
            # Fallback: treat as search query
            return f"CONTACTS_SEARCH:{contacts_arg}"
        elif command == "/scripts":
            scripts_arg = args.strip()
            if not scripts_arg or scripts_arg.lower() in {"list", "ls"}:
                return "SCRIPTS_LIST"
            scripts_parts = scripts_arg.split(None, 1)
            subcommand = scripts_parts[0].lower()
            rest = scripts_arg[len(scripts_parts[0]):].strip() if len(scripts_parts) > 1 else ""
            if subcommand == "add":
                add_parts = rest.split(None, 1)
                if len(add_parts) < 2:
                    self.print_error("Usage: /scripts add <name> <path>")
                    return None
                payload = json.dumps({"name": add_parts[0], "file_path": add_parts[1]})
                return f"SCRIPTS_ADD:{payload}"
            if subcommand == "info":
                if not rest:
                    self.print_error("Usage: /scripts info <id|#index|name>")
                    return None
                return f"SCRIPTS_INFO:{rest}"
            if subcommand == "search":
                if not rest:
                    self.print_error("Usage: /scripts search <query>")
                    return None
                return f"SCRIPTS_SEARCH:{rest}"
            if subcommand == "update":
                if not rest:
                    self.print_error("Usage: /scripts update <id|#index|name> <field=value ...>")
                    return None
                return f"SCRIPTS_UPDATE:{rest}"
            if subcommand in {"remove", "rm", "delete", "del"}:
                if not rest:
                    self.print_error("Usage: /scripts remove <id|#index|name>")
                    return None
                return f"SCRIPTS_REMOVE:{rest}"
            # Fallback: treat as search
            return f"SCRIPTS_SEARCH:{scripts_arg}"
        elif command == "/apis":
            apis_arg = args.strip()
            if not apis_arg or apis_arg.lower() in {"list", "ls"}:
                return "APIS_LIST"
            apis_parts = apis_arg.split(None, 1)
            subcommand = apis_parts[0].lower()
            rest = apis_arg[len(apis_parts[0]):].strip() if len(apis_parts) > 1 else ""
            if subcommand == "add":
                add_parts = rest.split(None, 1)
                if len(add_parts) < 2:
                    self.print_error("Usage: /apis add <name> <base_url>")
                    return None
                payload = json.dumps({"name": add_parts[0], "base_url": add_parts[1]})
                return f"APIS_ADD:{payload}"
            if subcommand == "info":
                if not rest:
                    self.print_error("Usage: /apis info <id|#index|name>")
                    return None
                return f"APIS_INFO:{rest}"
            if subcommand == "search":
                if not rest:
                    self.print_error("Usage: /apis search <query>")
                    return None
                return f"APIS_SEARCH:{rest}"
            if subcommand == "update":
                if not rest:
                    self.print_error("Usage: /apis update <id|#index|name> <field=value ...>")
                    return None
                return f"APIS_UPDATE:{rest}"
            if subcommand in {"remove", "rm", "delete", "del"}:
                if not rest:
                    self.print_error("Usage: /apis remove <id|#index|name>")
                    return None
                return f"APIS_REMOVE:{rest}"
            # Fallback: treat as search
            return f"APIS_SEARCH:{apis_arg}"
        elif command == "/orchestrate":
            orchestrate_arg = args.strip()
            if not orchestrate_arg:
                self.print_error("Usage: /orchestrate <request>")
                return None
            payload = json.dumps({"request": orchestrate_arg}, ensure_ascii=True)
            return f"ORCHESTRATE:{payload}"
        elif command == "/planning":
            planning_arg = args.strip().lower()
            if planning_arg == "on":
                return "PLANNING_ON"
            if planning_arg == "off":
                return "PLANNING_OFF"
            self.print_error("Usage: /planning on|off")
            return None
        elif command == "/pipeline":
            pipeline_arg = args.strip().lower()
            if not pipeline_arg:
                return "PIPELINE_INFO"
            if pipeline_arg in {"loop", "simple"}:
                return "PIPELINE_MODE:loop"
            if pipeline_arg in {"contracts", "contract", "complex"}:
                return "PIPELINE_MODE:contracts"
            self.print_error("Usage: /pipeline loop|contracts")
            return None
        elif command == "/monitor":
            monitor_arg = args.strip().lower()
            if monitor_arg == "on":
                return "MONITOR_ON"
            if monitor_arg == "off":
                return "MONITOR_OFF"
            if monitor_arg == "trace on":
                return "MONITOR_TRACE_ON"
            if monitor_arg == "trace off":
                return "MONITOR_TRACE_OFF"
            if monitor_arg == "pipeline on":
                return "MONITOR_PIPELINE_ON"
            if monitor_arg == "pipeline off":
                return "MONITOR_PIPELINE_OFF"
            if monitor_arg == "full on":
                return "MONITOR_FULL_ON"
            if monitor_arg == "full off":
                return "MONITOR_FULL_OFF"
            self.print_error(
                "Usage: /monitor on|off | /monitor trace on|off | /monitor pipeline on|off | /monitor full on|off"
            )
            return None
        elif command == "/scroll":
            scroll_arg = args.strip().lower()
            if scroll_arg == "status":
                return "MONITOR_SCROLL_STATUS"
            parts = scroll_arg.split()
            if len(parts) < 2 or len(parts) > 3:
                self.print_error(
                    "Usage: /scroll <chat|monitor> <up|down|pageup|pagedown|top|bottom> [n] | /scroll status"
                )
                return None
            pane = parts[0].strip().lower()
            action = parts[1].strip().lower()
            amount = 1
            if len(parts) == 3:
                try:
                    amount = int(parts[2])
                except Exception:
                    self.print_error("Scroll amount must be a positive integer")
                    return None
                if amount <= 0:
                    self.print_error("Scroll amount must be a positive integer")
                    return None
            payload = {"pane": pane, "action": action, "amount": amount}
            return f"MONITOR_SCROLL:{json.dumps(payload, ensure_ascii=True)}"
        elif command == "/skills":
            return "SKILLS_LIST"
        elif command == "/skill":
            raw = args.strip()
            if not raw:
                self.print_error(
                    "Usage: /skill <name> [args] | /skill list | /skill search <criteria> | /skill install <github-url> | /skill install <skill-name> [install-id]"
                )
                return None
            if raw.lower() == "list":
                return "SKILLS_LIST"
            parts = raw.split(None, 2)
            if parts and parts[0].lower() == "search":
                query = raw[len(parts[0]) :].strip()
                if len(query) >= 2 and query[0] == query[-1] and query[0] in {"'", '"'}:
                    query = query[1:-1].strip()
                if not query:
                    self.print_error("Usage: /skill search <criteria>")
                    return None
                payload = json.dumps({"query": query}, ensure_ascii=True)
                return f"SKILL_SEARCH:{payload}"
            if parts and parts[0].lower() == "install":
                if len(parts) < 2 or not parts[1].strip():
                    self.print_error("Usage: /skill install <github-url> | /skill install <skill-name> [install-id]")
                    return None
                target = parts[1].strip()
                install_id = parts[2].strip() if len(parts) > 2 else ""
                if target.startswith(("http://", "https://")):
                    if install_id:
                        self.print_error("Usage: /skill install <github-url>")
                        return None
                    payload = json.dumps({"url": target}, ensure_ascii=True)
                else:
                    payload = json.dumps(
                        {"name": target, "install_id": install_id},
                        ensure_ascii=True,
                    )
                return f"SKILL_INSTALL:{payload}"
            parts = raw.split(None, 1)
            skill_name = parts[0].strip()
            skill_args = parts[1].strip() if len(parts) > 1 else ""
            if not skill_name:
                self.print_error(
                    "Usage: /skill <name> [args] | /skill list | /skill search <criteria> | /skill install <github-url> | /skill install <skill-name> [install-id]"
                )
                return None
            payload = json.dumps({"name": skill_name, "args": skill_args}, ensure_ascii=True)
            return f"SKILL_INVOKE:{payload}"
        elif command == "/approve":
            raw = args.strip()
            parts = raw.split()
            if len(parts) == 3 and parts[0].lower() == "user":
                platform = parts[1].strip().lower()
                token = parts[2].strip()
                if platform in {"telegram", "slack", "discord"} and token:
                    return f"APPROVE_CHAT_USER:{platform}:{token}"
            self.print_error("Usage: /approve user <telegram|slack|discord> <token>")
            return None
        elif command.startswith("/") and len(command) > 1 and command not in {"/exit", "/quit", "/q"}:
            # Dynamic skill alias, e.g. "/example-source-brief ...".
            alias_name = command[1:].strip()
            if not alias_name:
                self.print_error(f"Unknown command: {command}")
                return None
            payload = json.dumps({"name": alias_name, "args": args.strip()}, ensure_ascii=True)
            return f"SKILL_ALIAS_INVOKE:{payload}"
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
