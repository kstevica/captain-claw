from datetime import datetime

from captain_claw.cli import TerminalUI


def test_special_command_completion():
    ui = TerminalUI()
    assert ui._complete_special_command("/h", 0) == "/help"
    assert ui._complete_special_command("/h", 1) == "/history"
    assert ui._complete_special_command("hello", 0) is None


def test_print_status_line_has_green_background(capsys):
    ui = TerminalUI()
    ui.print_status_line(
        last_usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        total_usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        last_exec_seconds=1.23,
        last_completed_at=datetime(2026, 2, 17, 12, 0, 0),
        session_id="session-1",
    )
    out = capsys.readouterr().out
    assert out.count("\x1b[30;42m") >= 2
    assert "session: session-1" in out


def test_system_panel_keeps_last_five_lines():
    ui = TerminalUI()
    for i in range(8):
        ui.append_system_line(f"log-{i}")
    assert list(ui._system_lines) == ["log-3", "log-4", "log-5", "log-6", "log-7"]


def test_begin_assistant_stream_prefix(capsys):
    ui = TerminalUI()
    ui.begin_assistant_stream()
    out = capsys.readouterr().out
    assert "[ASSISTANT]" in out


def test_runtime_status_accepts_allowed_values():
    ui = TerminalUI()
    ui.set_runtime_status("running script")
    assert ui._runtime_status == "running script"
    ui.set_runtime_status("invalid-status")
    assert ui._runtime_status == "waiting"


def test_print_message_user_and_assistant_are_visually_distinct(capsys):
    ui = TerminalUI()
    ui._ansi_enabled = True
    ui.print_message("user", "hello")
    ui.print_message("assistant", "hello")
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line]
    assert len(lines) >= 2
    assert "[USER]" in lines[0]
    assert "[ASSISTANT]" in lines[1]
    assert lines[0] != lines[1]
    assert "\x1b[" in lines[0]
    assert "\x1b[" in lines[1]


def test_system_panel_status_has_highlight_style(capsys):
    ui = TerminalUI()
    ui._sticky_footer = True
    ui._ansi_enabled = True
    ui._runtime_status = "thinking"
    ui._render_system_panel()
    out = capsys.readouterr().out
    assert "\x1b[97;44m STATUS \x1b[0m" in out
    assert "THINKING" in out


def test_footer_redraw_is_deferred_while_assistant_output_active(capsys):
    ui = TerminalUI()
    ui._sticky_footer = True
    ui._assistant_output_active = True
    ui._footer_dirty = False
    ui._render_footer(force=False)
    out = capsys.readouterr().out
    assert out == ""
    assert ui._footer_dirty is True


def test_end_assistant_stream_forces_footer_refresh(monkeypatch):
    ui = TerminalUI()
    ui._sticky_footer = True
    calls: list[bool] = []

    def fake_render_footer(force: bool = False) -> None:
        calls.append(force)

    monkeypatch.setattr(ui, "_render_footer", fake_render_footer)
    ui._assistant_output_active = True
    ui.end_assistant_stream()
    assert ui._assistant_output_active is False
    assert calls == [True]


def test_append_system_line_forces_refresh_when_not_streaming(monkeypatch):
    ui = TerminalUI()
    ui._sticky_footer = True
    ui._assistant_output_active = False
    calls: list[bool] = []

    def fake_render_footer(force: bool = False) -> None:
        calls.append(force)

    monkeypatch.setattr(ui, "_render_footer", fake_render_footer)
    ui.append_system_line("hello")
    assert calls == [True]


def test_monitor_command_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/monitor on") == "MONITOR_ON"
    assert ui.handle_special_command("/monitor off") == "MONITOR_OFF"


def test_append_tool_output_formats_header_and_body():
    ui = TerminalUI()
    ui.append_tool_output("shell", {"command": "date"}, "Tue")
    assert "shell" in ui._tool_output_text
    assert '"command": "date"' in ui._tool_output_text
    assert "Tue" in ui._tool_output_text


def test_load_monitor_tool_output_from_session():
    ui = TerminalUI()
    ui.load_monitor_tool_output_from_session(
        [
            {
                "role": "tool",
                "tool_name": "shell",
                "tool_arguments": {"command": "ls"},
                "content": "file1\nfile2",
            },
            {"role": "assistant", "content": "ignored"},
        ]
    )
    assert "shell" in ui._tool_output_text
    assert "file1" in ui._tool_output_text
