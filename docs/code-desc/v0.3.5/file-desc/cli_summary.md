# Summary: cli.py

# cli.py Summary

**Summary:** A comprehensive terminal UI framework for Captain Claw that manages interactive command-line interaction with support for split-pane monitoring, fixed footer layouts, readline history, and extensive special command parsing. Handles both traditional line-by-line output and a sophisticated split-view mode with independent scrolling panes for chat and tool monitoring.

**Purpose:** Provides a complete terminal user interface abstraction layer that bridges user input/output with the Captain Claw agent runtime, supporting multiple display modes (standard output, split monitor view), persistent command history, real-time status indicators, and a rich command vocabulary for session/model/task management.

---

## Most Important Functions/Classes/Procedures

### 1. **TerminalUI (class)**
Core UI controller managing all terminal interactions. Maintains state for display modes (monitor vs. standard), scroll positions, footer rendering, readline integration, and output buffering. Handles terminal capability detection, SIGWINCH resize signals, and ANSI color support. Central orchestrator for all print/prompt operations.

### 2. **handle_special_command(cmd: str) -> str | None**
Parses and routes all slash-prefixed commands (`/help`, `/session`, `/cron`, `/skill`, etc.) into semantic action codes. Implements complex multi-level argument parsing with validation (e.g., `/session queue mode <steer|followup|...>`, `/cron add every <Nm|Nh> <task>`). Returns action strings like `"SESSION_SELECT:id"` or `"CRON_ONEOFF:{json}"` for downstream processing, or `None` to suppress output.

### 3. **_render_monitor_view() -> None**
Renders split-pane monitor layout with independent left (chat) and right (tool output) columns. Wraps text to pane widths, applies scroll offsets, and uses ANSI escape sequences to position content in fixed terminal regions. Manages header labels showing scroll position indicators and separator styling. Critical for real-time agent monitoring.

### 4. **_render_footer(force: bool = False) -> None**
Renders fixed footer area containing system status panel, runtime state badge (thinking/running/waiting), inline thinking text indicator, and two-line status bar with token usage/timing metadata. Implements FPS throttling (1 Hz default) to prevent excessive redraws during streaming. Uses ANSI escape sequences to preserve output cursor position.

### 5. **_layout_metrics() -> dict[str, int]**
Computes fixed terminal layout geometry: output region bottom row, system panel start, status bar rows, prompt row. Accounts for minimum terminal size (40 cols × 24 rows) and reserved footer space. Returns width, total rows, and all boundary positions for cursor positioning and scroll region setup.

### 6. **append_tool_output(tool_name: str, arguments: dict, raw_output: str, render: bool = True) -> None**
Appends formatted tool execution result to monitor pane buffer. Summarizes web_fetch output (extracts content, truncates to 320 chars). Updates thinking indicator for non-silent tools (skips internal tools like `llm_trace`, `guard_*`). Handles both live rendering and history replay modes.

### 7. **scroll_monitor_pane(pane: str, action: str, amount: int = 1) -> tuple[bool, str]**
Adjusts scroll offset for chat or monitor pane with actions: `up|down|pageup|pagedown|top|bottom`. Clamps offsets to buffer bounds. Returns success flag and human-readable status message. Supports multi-page scrolling and immediate re-render.

### 8. **set_thinking(text: str, tool: str = "", phase: str = "tool") -> None**
Updates inline thinking/reasoning indicator shown on status line during agent execution. Immediately re-renders system panel to provide real-time feedback. Clears on `phase="done"` or empty text. Separate from footer throttling to ensure visibility during streaming.

### 9. **print_streaming(chunk: str, end: str = "") -> None**
Outputs streaming response chunks with mode-aware routing (monitor pane vs. standard output). In sticky footer mode, ensures scroll region is active to prevent text overwriting status bar. Flushes immediately for real-time display.

### 10. **prompt(prompt_text: str = "> ") -> str**
Reads user input with readline integration (history, completion). Positions cursor at fixed prompt row in sticky footer mode. Auto-adds non-empty input to readline history. Returns trimmed input string.

---

## Architecture & Dependencies

**Key Dependencies:**
- `readline` (optional): Command history persistence and tab completion
- `termios`, `tty` (POSIX only): Raw terminal mode for ESC key capture
- `asyncio`: Async ESC key waiting
- `signal`: SIGWINCH handler for terminal resize events
- `shutil`: Terminal size detection
- `textwrap`: Text wrapping for pane layout
- `json`: Payload serialization for complex commands
- `captain_claw.config`: Configuration access
- `captain_claw.logging`: Logging integration
- `captain_claw.agent_tool_loop_mixin`: Tool thinking summary generation

**System Architecture:**
- **Display Modes:** Standard (line-by-line) vs. Monitor (split-pane with independent scroll)
- **Footer System:** Fixed-position system panel (status + logs) + status bar (tokens/timing) + prompt row
- **Scroll Region:** ANSI escape sequence `\033[1;Nr` restricts scrolling to output area, protecting footer
- **Buffer Management:** Circular deques for system logs; bounded text buffers for monitor panes (200 KB default)
- **Command Routing:** Special commands parsed into semantic action codes (`SESSION_SELECT:id`, `CRON_ONEOFF:{json}`) for caller dispatch
- **State Tracking:** Runtime status, thinking text, assistant output flag, monitor mode toggle, scroll offsets per pane

**Role in System:**
Central UI abstraction layer between user and agent runtime. Decouples terminal rendering from business logic. Provides rich command vocabulary for session/model/task management. Enables real-time monitoring of agent execution with split-pane layout and thinking indicators. Handles all terminal capability detection and graceful degradation (ANSI colors, sticky footer, ESC capture).