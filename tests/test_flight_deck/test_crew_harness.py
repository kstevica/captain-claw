"""Crew's adapters: how each coding CLI is started, and what its real output means.
The fixtures are trimmed from real runs of each CLI in a git worktree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from captain_claw.flight_deck import crew_harness
from captain_claw.flight_deck.crew_harness import HARNESSES, RunSpec, TurnState, parse_line

FIXTURES = Path(__file__).parent / "crew_fixtures"


def _read(harness: str) -> tuple[TurnState, list[dict]]:
    state = TurnState()
    events: list[dict] = []
    for line in (FIXTURES / f"{harness}.jsonl").read_text().splitlines():
        events.extend(parse_line(harness, line, state))
    return state, events


def test_claude_fixture():
    state, events = _read("claude")
    assert state.session
    assert "CREW-STATUS: done" in state.final
    assert state.tokens_in > 0 and state.tokens_out > 0
    assert state.cost_usd > 0
    assert not state.error
    assert any(e["kind"] == "tool" and e["tool"] == "Bash" for e in events)
    assert any(e["kind"] == "result" for e in events)


def test_codex_fixture():
    state, events = _read("codex")
    assert state.session
    assert "CREW-STATUS: done" in state.final
    assert state.tokens_in > 0 and state.tokens_out > 0
    assert any(e["kind"] == "tool" and e["tool"] == "shell" for e in events)
    assert any(e["kind"] == "tool" and e["tool"] == "edit" for e in events)


def test_opencode_fixture():
    state, events = _read("opencode")
    assert state.session.startswith("ses_")
    assert state.final == "OK"
    assert state.tokens_in > 0 and state.tokens_out > 0
    assert any(e["kind"] == "tool" and e["tool"] == "write" for e in events)
    assert any(e["kind"] == "result" for e in events)


def test_pi_fixture():
    state, events = _read("pi")
    assert state.session == "crew-spike-1"
    assert state.final == "OK"
    assert state.tokens_in > 0 and state.tokens_out > 0
    assert any(e["kind"] == "tool" and e["tool"] == "write" for e in events)
    assert any(e["kind"] == "result" for e in events)


@pytest.mark.parametrize(("harness", "line", "text"), [
    ("claude", '{"type":"result","is_error":true,"result":"Credit balance too low"}', "Credit balance too low"),
    ("codex", '{"type":"turn.failed","error":{"message":"usage limit"}}', "usage limit"),
    ("codex", '{"type":"error","message":"stream broke"}', "stream broke"),
    ("opencode", '{"type":"error","error":{"name":"E","data":{"message":"no key"}}}', "no key"),
    ("pi", '{"type":"message_end","message":{"role":"assistant","content":[],"stopReason":"error",'
           '"errorMessage":"401"}}', "401"),
])
def test_a_cli_saying_it_failed_is_kept(harness, line, text):
    state = TurnState()
    events = parse_line(harness, line, state)
    assert state.error == text
    assert events[-1]["kind"] == "error"


def test_a_retry_that_worked_is_not_a_failure():
    state = TurnState()
    parse_line("codex", '{"type":"error","message":"Reconnecting... 1/5"}', state)
    assert state.error
    parse_line("codex", '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}', state)
    assert state.error == ""
    state = TurnState()
    parse_line("pi", '{"type":"message_end","message":{"role":"assistant","content":[],"stopReason":"error",'
                     '"errorMessage":"overloaded"}}', state)
    assert state.error
    parse_line("pi", '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]}}', state)
    assert state.error == "" and state.final == "ok"


@pytest.mark.parametrize("harness", HARNESSES)
def test_unknown_and_broken_lines_cost_nothing(harness):
    state = TurnState()
    assert parse_line(harness, '{"type":"something_new","x":1}', state) == []
    assert parse_line(harness, "", state) == []
    assert parse_line(harness, "not json", state) == [{"kind": "note", "text": "not json"}]
    assert parse_line(harness, "[1, 2]", state) == []
    # A known type with an unexpected shape.
    assert all(e["kind"] == "note" for e in parse_line(harness, '{"type":"assistant","message":{"content":[1]}}', state))


def _spec(**over) -> RunSpec:
    fields = {"binary": "/bin/x", "prompt": "do it", "cwd": "/w", "git_common_dir": "/r/.git", "task_id": "crw_1"}
    return RunSpec(**{**fields, **over})


def test_claude_says_its_model_and_its_limits():
    state = TurnState()
    # As a real run printed them.
    line = {"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed", "resetsAt": 1793811600, "rateLimitType": "seven_day", "overageStatus": "rejected",
        "unifiedWindows": {"seven_day": {"utilization": 0.44, "resetsAt": 1793811600, "status": "allowed"}}}}
    assert parse_line("claude", json.dumps(line), state) == []
    assert state.limits == {"seven_day": {"used": 0.44, "resets_at": 1793811600, "status": "allowed"}}
    parse_line("claude", json.dumps({"type": "system", "subtype": "init", "model": "claude-sonnet-5-5"}), state)
    assert state.model == "claude-sonnet-5-5"
    # It survives a restart, like the rest of a turn.
    assert TurnState.from_json(state.to_json()).limits == state.limits
    # With no windows it can still name the one it means.
    other = TurnState()
    parse_line("claude", json.dumps({"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed_warning", "resetsAt": 5, "rateLimitType": "five_hour"}}), other)
    assert other.limits == {"five_hour": {"used": None, "resets_at": 5, "status": "allowed_warning"}}
    # It isn't a documented format: nothing it might say instead may break a turn.
    for info in ({}, {"unifiedWindows": "soon"}, {"unifiedWindows": {"seven_day": 3}}, "none", None):
        odd = TurnState()
        assert parse_line("claude", json.dumps({"type": "rate_limit_event", "rate_limit_info": info}), odd) == []
        assert odd.limits == {}


def test_pi_says_its_model():
    state = TurnState()
    parse_line("pi", json.dumps({"type": "message_end", "message": {"role": "assistant", "model": "ornith:9b", "content": []}}), state)
    assert state.model == "ornith:9b"


def test_opencode_lists_only_what_looks_like_a_model(monkeypatch):
    class Done:
        returncode = 0
        stdout = "Performing one time database migration\nopencode/deepseek-v4.1-flash-free\n\nopencode/big-pickle\nnot a model\n"

    seen = {}

    def run(argv, **kwargs):
        seen.update(kwargs, argv=argv)
        return Done()

    monkeypatch.setattr(crew_harness.subprocess, "run", run)
    assert crew_harness.list_models("/bin/opencode", {"HOME": "/h"}, "/c") == ["opencode/deepseek-v4.1-flash-free", "opencode/big-pickle"]
    assert seen["argv"] == ["/bin/opencode", "models"] and seen["stdin"] == crew_harness.subprocess.DEVNULL and seen["env"] == {"HOME": "/h"}
    Done.returncode = 1
    assert crew_harness.list_models("/bin/opencode", {}, "/c") == []

    def broken(argv, **kwargs):
        raise OSError("gone")

    monkeypatch.setattr(crew_harness.subprocess, "run", broken)
    assert crew_harness.list_models("/bin/opencode", {}, "/c") == []


def test_claude_argv(monkeypatch):
    monkeypatch.delenv("FD_CREW_CLAUDE_PERMISSION", raising=False)
    argv = HARNESSES["claude"].argv(_spec())
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert argv[argv.index("--permission-mode") + 1] == "auto"
    assert "--resume" not in argv
    assert "--dangerously-skip-permissions" not in argv
    resumed = HARNESSES["claude"].argv(_spec(session="abc", model="haiku", effort="high"))
    assert resumed[resumed.index("--resume") + 1] == "abc"
    assert resumed[resumed.index("--model") + 1] == "haiku"
    assert resumed[resumed.index("--effort") + 1] == "high"


def test_claude_permission_mode_is_the_decks_and_checked(monkeypatch):
    monkeypatch.setenv("FD_CREW_CLAUDE_PERMISSION", "acceptEdits")
    assert crew_harness.claude_permission_mode() == "acceptEdits"
    monkeypatch.setenv("FD_CREW_CLAUDE_PERMISSION", "--anything")
    assert crew_harness.claude_permission_mode() == "auto"


def test_codex_argv_sandboxes_and_lets_a_worktree_commit():
    argv = HARNESSES["codex"].argv(_spec())
    assert argv[:2] == ["/bin/x", "exec"]
    assert 'sandbox_mode="workspace-write"' in argv
    assert 'sandbox_workspace_write.writable_roots=["/r/.git"]' in argv
    assert "features.hooks=false" in argv
    assert "resume" not in argv
    assert argv[-1] == "do it"
    resumed = HARNESSES["codex"].argv(_spec(session="t-1", effort="high"))
    assert resumed[2] == "resume"
    assert resumed[-2:] == ["t-1", "do it"]
    assert 'model_reasoning_effort="high"' in resumed
    assert not {"-s", "--add-dir"} & set(resumed)


def test_opencode_argv():
    argv = HARNESSES["opencode"].argv(_spec(session="ses_1", model="a/b", effort="high"))
    assert argv[:4] == ["/bin/x", "run", "--format", "json"]
    assert "--auto" in argv
    assert argv[argv.index("--dir") + 1] == "/w"
    assert argv[argv.index("--session") + 1] == "ses_1"
    assert argv[argv.index("--variant") + 1] == "high"


def test_pi_argv_names_its_session_up_front():
    argv = HARNESSES["pi"].argv(_spec())
    assert argv[argv.index("--session-id") + 1] == "crew-crw_1"
    assert argv[argv.index("--mode") + 1] == "json"
    again = HARNESSES["pi"].argv(_spec(session="crew-crw_1", effort="high"))
    assert again[again.index("--session-id") + 1] == "crew-crw_1"
    assert again[again.index("--thinking") + 1] == "high"


@pytest.mark.parametrize("harness", HARNESSES)
def test_a_prompt_cannot_read_as_a_flag(harness):
    argv = HARNESSES[harness].argv(_spec(prompt="--dangerously-do-things"))
    assert "--dangerously-do-things" not in argv
    assert " --dangerously-do-things" in argv


def test_turn_state_survives_saving():
    state = TurnState(session="s", final="done", tokens_in=3, cost_usd=0.5)
    assert TurnState.from_json(state.to_json()) == state
    assert TurnState.from_json("not json") == TurnState()
    assert TurnState.from_json('{"session": "s", "unknown": 1}').session == "s"
