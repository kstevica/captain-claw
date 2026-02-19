from datetime import datetime
import json

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
        context_window={"prompt_tokens": 15000, "context_budget_tokens": 100000, "utilization": 0.15},
        model_details={
            "provider": "ollama",
            "model": "minimax-m2.5:cloud",
            "temperature": 0.7,
            "max_tokens": 32000,
        },
    )
    out = capsys.readouterr().out
    assert out.count("\x1b[30;42m") >= 2
    assert "session: session-1" in out
    assert "CTX 15,000/100,000 (15.0%)" in out
    assert "MODEL ollama/minimax-m2.5:cloud [t=0.7 gen=32,000]" in out


def test_system_panel_keeps_last_five_lines():
    ui = TerminalUI()
    for i in range(8):
        ui.append_system_line(f"log-{i}")
    assert list(ui._system_lines) == ["log-3", "log-4", "log-5", "log-6", "log-7"]


def test_begin_assistant_stream_prefix(capsys):
    ui = TerminalUI()
    ui.begin_assistant_stream()
    out = capsys.readouterr().out
    assert ":)" in out


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
    assert ">" in lines[0]
    assert ":)" in lines[1]
    assert lines[0] != lines[1]
    assert "\x1b[" in lines[0]
    assert "\x1b[" in lines[1]


def test_monitor_view_has_chat_and_monitor_green_header(capsys):
    ui = TerminalUI()
    ui._sticky_footer = True
    ui._monitor_mode = True
    ui._ansi_enabled = True
    ui._chat_output_text = "user line"
    ui._tool_output_text = "tool line"
    ui._render_monitor_view()
    out = capsys.readouterr().out
    assert "Chat" in out
    assert "Monitor" in out
    assert "\x1b[30;42m" in out


def test_monitor_chat_prefixes_are_plain_for_alignment():
    ui = TerminalUI()
    ui._sticky_footer = True
    ui._monitor_mode = True
    ui._ansi_enabled = True
    ui.print_message("user", "hello")
    ui.print_message("assistant", "hi")
    assert "> hello" in ui._chat_output_text
    assert ":) hi" in ui._chat_output_text
    assert "\x1b[" not in ui._chat_output_text


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
    assert ui.handle_special_command("/monitor trace on") == "MONITOR_TRACE_ON"
    assert ui.handle_special_command("/monitor trace off") == "MONITOR_TRACE_OFF"
    assert ui.handle_special_command("/monitor pipeline on") == "MONITOR_PIPELINE_ON"
    assert ui.handle_special_command("/monitor pipeline off") == "MONITOR_PIPELINE_OFF"
    assert ui.handle_special_command("/monitor full on") == "MONITOR_FULL_ON"
    assert ui.handle_special_command("/monitor full off") == "MONITOR_FULL_OFF"
    assert ui.handle_special_command("/scroll status") == "MONITOR_SCROLL_STATUS"
    scroll = ui.handle_special_command("/scroll chat up 7")
    assert scroll is not None
    assert scroll.startswith("MONITOR_SCROLL:")
    payload = json.loads(scroll.split(":", 1)[1])
    assert payload == {"pane": "chat", "action": "up", "amount": 7}


def test_monitor_scroll_offsets_are_independent():
    ui = TerminalUI()
    ui._monitor_mode = True
    ui._sticky_footer = False
    ui._chat_output_text = "\n".join(f"chat-{i}" for i in range(120))
    ui._tool_output_text = "\n".join(f"tool-{i}" for i in range(120))

    ok_chat, _ = ui.scroll_monitor_pane("chat", "up", 5)
    assert ok_chat is True
    assert ui._monitor_chat_scroll_offset == 5
    assert ui._monitor_tool_scroll_offset == 0

    ok_tool, _ = ui.scroll_monitor_pane("monitor", "up", 3)
    assert ok_tool is True
    assert ui._monitor_chat_scroll_offset == 5
    assert ui._monitor_tool_scroll_offset == 3

    ok_bottom, _ = ui.scroll_monitor_pane("chat", "bottom", 1)
    assert ok_bottom is True
    assert ui._monitor_chat_scroll_offset == 0
    assert ui._monitor_tool_scroll_offset == 3


def test_monitor_scroll_top_moves_to_oldest_visible_chunk():
    ui = TerminalUI()
    ui._monitor_mode = True
    ui._sticky_footer = False
    ui._chat_output_text = "\n".join(f"chat-{i}" for i in range(180))

    ok, _ = ui.scroll_monitor_pane("chat", "top", 1)
    assert ok is True
    state = ui.get_monitor_scroll_state()
    assert state["chat_offset"] == state["chat_max_offset"]


def test_session_commands_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/sessions") == "SESSIONS"
    assert ui.handle_special_command("/models") == "MODELS"
    assert ui.handle_special_command("/session list") == "SESSIONS"
    assert ui.handle_special_command("/session") == "SESSION_INFO"
    assert ui.handle_special_command("/session abc123") == "SESSION_SELECT:abc123"
    assert ui.handle_special_command("/session switch abc123") == "SESSION_SELECT:abc123"
    assert ui.handle_special_command("/session model") == "SESSION_MODEL_INFO"
    assert ui.handle_special_command("/session model list") == "MODELS"
    assert ui.handle_special_command("/session model claude-sonnet") == "SESSION_MODEL_SET:claude-sonnet"
    assert ui.handle_special_command("/session export") == "SESSION_EXPORT:all"
    assert ui.handle_special_command("/session export chat") == "SESSION_EXPORT:chat"
    assert ui.handle_special_command("/session export monitor") == "SESSION_EXPORT:monitor"
    assert ui.handle_special_command("/session export pipeline") == "SESSION_EXPORT:pipeline"
    assert ui.handle_special_command("/session export pipeline-summary") == "SESSION_EXPORT:pipeline-summary"
    assert ui.handle_special_command("/session protect on") == "SESSION_PROTECT_ON"
    assert ui.handle_special_command("/session protect off") == "SESSION_PROTECT_OFF"
    assert ui.handle_special_command("/session queue") == "SESSION_QUEUE_INFO"
    assert ui.handle_special_command("/session queue mode collect") == "SESSION_QUEUE_MODE:collect"
    assert ui.handle_special_command("/session queue debounce 250") == "SESSION_QUEUE_DEBOUNCE:250"
    assert ui.handle_special_command("/session queue cap 12") == "SESSION_QUEUE_CAP:12"
    assert ui.handle_special_command("/session queue drop summarize") == "SESSION_QUEUE_DROP:summarize"
    assert ui.handle_special_command("/session queue clear") == "SESSION_QUEUE_CLEAR"


def test_session_new_subcommand_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/session new") == "NEW"
    assert ui.handle_special_command("/session new investigation") == "NEW:investigation"
    assert ui.handle_special_command("/session new phase 2") == "NEW:phase 2"


def test_session_rename_subcommand_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/session rename focus-mode") == "SESSION_RENAME:focus-mode"
    assert ui.handle_special_command("/session rename release prep") == "SESSION_RENAME:release prep"


def test_session_description_subcommand_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/session description") == "SESSION_DESCRIPTION_INFO"
    assert ui.handle_special_command("/session description auto") == "SESSION_DESCRIPTION_AUTO"
    manual = ui.handle_special_command('/session description "This is user description"')
    assert manual is not None
    assert manual.startswith("SESSION_DESCRIPTION_SET:")
    payload = json.loads(manual.split(":", 1)[1])
    assert payload == {"description": "This is user description"}


def test_session_run_and_runin_parsing():
    ui = TerminalUI()
    session_run = ui.handle_special_command("/session run abc123 summarize status")
    assert session_run is not None
    assert session_run.startswith("SESSION_RUN:")
    session_payload = json.loads(session_run.split(":", 1)[1])
    assert session_payload == {"selector": "abc123", "prompt": "summarize status"}

    runin = ui.handle_special_command("/runin #2 check failures")
    assert runin is not None
    assert runin.startswith("SESSION_RUN:")
    runin_payload = json.loads(runin.split(":", 1)[1])
    assert runin_payload == {"selector": "#2", "prompt": "check failures"}


def test_session_procreate_parsing():
    ui = TerminalUI()
    command = ui.handle_special_command('/session procreate #1 #2 "combined memory"')
    assert command is not None
    assert command.startswith("SESSION_PROCREATE:")
    payload = json.loads(command.split(":", 1)[1])
    assert payload == {
        "parent_one": "#1",
        "parent_two": "#2",
        "new_name": "combined memory",
    }


def test_compact_command_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/compact") == "COMPACT"


def test_cron_command_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/cron list") == "CRON_LIST"
    assert ui.handle_special_command("/cron add every 15m \"check status\"") == 'CRON_ADD:every 15m "check status"'
    assert ui.handle_special_command("/cron pause job-1") == "CRON_PAUSE:job-1"
    assert ui.handle_special_command("/cron pause #1") == "CRON_PAUSE:#1"
    assert ui.handle_special_command("/cron resume job-1") == "CRON_RESUME:job-1"
    assert ui.handle_special_command("/cron resume #1") == "CRON_RESUME:#1"
    assert ui.handle_special_command("/cron remove job-1") == "CRON_REMOVE:job-1"
    assert ui.handle_special_command("/cron remove #1") == "CRON_REMOVE:#1"
    assert ui.handle_special_command("/cron history job-1") == "CRON_HISTORY:job-1"
    assert ui.handle_special_command("/cron history #1") == "CRON_HISTORY:#1"

    oneoff = ui.handle_special_command('/cron "run health check"')
    assert oneoff is not None
    assert oneoff.startswith("CRON_ONEOFF:")
    oneoff_payload = json.loads(oneoff.split(":", 1)[1])
    assert oneoff_payload == {"prompt": "run health check"}

    run_cmd = ui.handle_special_command("/cron run script scripts/s1/health.sh")
    assert run_cmd is not None
    assert run_cmd.startswith("CRON_RUN:")
    run_payload = json.loads(run_cmd.split(":", 1)[1])
    assert run_payload == {"args": "script scripts/s1/health.sh"}

    run_job = ui.handle_special_command("/cron run #2")
    assert run_job is not None
    assert run_job.startswith("CRON_RUN:")
    run_job_payload = json.loads(run_job.split(":", 1)[1])
    assert run_job_payload == {"args": "#2"}


def test_planning_command_parsing():
    ui = TerminalUI()
    assert ui.handle_special_command("/planning on") == "PLANNING_ON"
    assert ui.handle_special_command("/planning off") == "PLANNING_OFF"
    assert ui.handle_special_command("/pipeline") == "PIPELINE_INFO"
    assert ui.handle_special_command("/pipeline loop") == "PIPELINE_MODE:loop"
    assert ui.handle_special_command("/pipeline simple") == "PIPELINE_MODE:loop"
    assert ui.handle_special_command("/pipeline contracts") == "PIPELINE_MODE:contracts"
    assert ui.handle_special_command("/pipeline complex") == "PIPELINE_MODE:contracts"
    assert ui.handle_special_command("/skills") == "SKILLS_LIST"
    assert ui.handle_special_command("/skill list") == "SKILLS_LIST"
    install = ui.handle_special_command("/skill install https://github.com/openai/skills/tree/main/skills/.curated/source-brief")
    assert install is not None
    assert install.startswith("SKILL_INSTALL:")
    install_payload = json.loads(install.split(":", 1)[1])
    assert install_payload == {
        "url": "https://github.com/openai/skills/tree/main/skills/.curated/source-brief"
    }
    assert ui.handle_special_command("/skill install https://github.com/openai/skills extra") is None
    deps_install = ui.handle_special_command("/skill install event-planner")
    assert deps_install is not None
    assert deps_install.startswith("SKILL_INSTALL:")
    deps_payload = json.loads(deps_install.split(":", 1)[1])
    assert deps_payload == {"name": "event-planner", "install_id": ""}
    deps_install_id = ui.handle_special_command("/skill install event-planner uv-brew")
    assert deps_install_id is not None
    assert deps_install_id.startswith("SKILL_INSTALL:")
    deps_id_payload = json.loads(deps_install_id.split(":", 1)[1])
    assert deps_id_payload == {"name": "event-planner", "install_id": "uv-brew"}
    search = ui.handle_special_command('/skill search "summarize web sources"')
    assert search is not None
    assert search.startswith("SKILL_SEARCH:")
    search_payload = json.loads(search.split(":", 1)[1])
    assert search_payload == {"query": "summarize web sources"}
    assert ui.handle_special_command("/skill search") is None
    invoke = ui.handle_special_command("/skill docs summarize release notes")
    assert invoke is not None
    assert invoke.startswith("SKILL_INVOKE:")
    payload = json.loads(invoke.split(":", 1)[1])
    assert payload == {"name": "docs", "args": "summarize release notes"}
    alias = ui.handle_special_command("/example-source-brief index.hr")
    assert alias is not None
    assert alias.startswith("SKILL_ALIAS_INVOKE:")
    alias_payload = json.loads(alias.split(":", 1)[1])
    assert alias_payload == {"name": "example-source-brief", "args": "index.hr"}
    assert ui.handle_special_command("/approve user telegram ABC123") == "APPROVE_CHAT_USER:telegram:ABC123"
    assert ui.handle_special_command("/approve user slack ABC123") == "APPROVE_CHAT_USER:slack:ABC123"
    assert ui.handle_special_command("/approve user discord ABC123") == "APPROVE_CHAT_USER:discord:ABC123"


def test_new_command_supports_optional_name():
    ui = TerminalUI()
    assert ui.handle_special_command("/new") == "NEW"
    assert ui.handle_special_command("/new investigation") == "NEW:investigation"


def test_append_tool_output_formats_header_and_body():
    ui = TerminalUI()
    ui.append_tool_output("shell", {"command": "date"}, "Tue")
    assert "shell" in ui._tool_output_text
    assert '"command": "date"' in ui._tool_output_text
    assert "Tue" in ui._tool_output_text


def test_append_tool_output_marks_cron_entries():
    ui = TerminalUI()
    ui.append_tool_output("cron", {"step": "job_start"}, "starting")
    ui.append_tool_output("shell", {"command": "echo ok", "cron": True}, "ok")
    assert "[CRON] cron" in ui._tool_output_text
    assert "[CRON] shell" in ui._tool_output_text


def test_append_tool_output_web_fetch_is_summarized_with_kb():
    ui = TerminalUI()
    raw = (
        "[URL: https://example.com]\n"
        "[Status: 200]\n"
        "[Mode: text]\n"
        "[Size: 1200 chars]\n\n"
        "Captain Claw introduces monitor improvements. "
        "The monitor now shows concise summaries with source context. "
        "This third sentence should not be included."
    )
    ui.append_tool_output("web_fetch", {"url": "https://example.com"}, raw)
    assert "Summary:" in ui._tool_output_text
    assert "Used text:" in ui._tool_output_text
    assert "kB" in ui._tool_output_text
    assert "concise summaries with source context." in ui._tool_output_text
    assert "This third sentence should not be included." not in ui._tool_output_text


def test_append_tool_output_web_fetch_full_mode_shows_raw():
    ui = TerminalUI()
    raw = (
        "[URL: https://example.com]\n"
        "[Status: 200]\n"
        "[Mode: text]\n\n"
        "First sentence. Second sentence. Third sentence remains visible."
    )
    ui.set_monitor_full_output(True)
    ui.append_tool_output("web_fetch", {"url": "https://example.com"}, raw)
    assert "Summary:" not in ui._tool_output_text
    assert "Third sentence remains visible." in ui._tool_output_text


def test_append_tool_output_separates_items_with_one_blank_line():
    ui = TerminalUI()
    ui.append_tool_output("shell", {"command": "echo one"}, "one\n")
    ui.append_tool_output("shell", {"command": "echo two"}, "two\n")
    assert "one\n\nshell" in ui._tool_output_text
    assert "\n\n\n" not in ui._tool_output_text


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
