"""Crew: a worker whose model fails doesn't end up failed or stuck when trying again could get past it.

A CLI gets past most of these by itself. What's here is what it doesn't: giving up on a connection,
or saying nothing more, ever. The stand-in below has one scenario per way of doing that.
"""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from captain_claw.flight_deck import crew_harness, crew_runner, crew_store
from captain_claw.flight_deck.crew_harness import TurnState

from .test_crew_runner import _dispatch, _seen, _settle, deck  # noqa: F401

STAND_IN = r"""#!/bin/sh
case "$1" in --version) echo "stand-in 1.0"; exit 0;; esac
all="$*"
gd="$(git rev-parse --absolute-git-dir)"
printf '%s' "$all" > "$gd/seen"
for m in overloaded always badkey stuck hung busy recovers toolfail rattle; do
  case "$all" in *SCENARIO:$m*) echo $m > "$gd/mode";; esac
done
mode=$(cat "$gd/mode" 2>/dev/null)
n=$(( $(cat "$gd/tries" 2>/dev/null || echo 0) + 1 ))
echo $n > "$gd/tries"
say() { printf '%s\n' "$1"; }
fails() { say "{\"type\":\"result\",\"is_error\":true,\"result\":\"$1\",\"session_id\":\"sess-1\"}"; }
done_now() {
  say "{\"type\":\"assistant\",\"session_id\":\"sess-1\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Done.\\n\\nCREW-STATUS: done it worked\"}]}}"
  say "{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"session_id\":\"sess-1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}"
  exit 0
}
say "{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"sess-1\",\"model\":\"m\"}"
case "$mode" in
  overloaded) [ "$n" -ge 2 ] && done_now; fails "API Error: 529 Overloaded"; exit 1 ;;
  always) fails "API Error: 529 Overloaded"; exit 1 ;;
  badkey) fails "Failed to authenticate. API Error: 401 invalid x-api-key"; exit 1 ;;
  stuck) [ "$n" -ge 2 ] && done_now; fails "Transport: ECONNRESET: The socket connection was closed"; sleep 60 ;;
  hung) [ "$n" -ge 2 ] && done_now; mkfifo "$gd/never"; read _ < "$gd/never" ;;
  busy) sleep 60 ;;
  recovers) fails "OpenAI Chat stream ended without finish_reason"; done_now ;;
  toolfail)
    say "{\"type\":\"assistant\",\"session_id\":\"sess-1\",\"message\":{\"content\":[{\"type\":\"tool_use\",\"name\":\"Bash\",\"input\":{\"command\":\"npm i\"}}]}}"
    say "{\"type\":\"user\",\"session_id\":\"sess-1\",\"message\":{\"content\":[{\"type\":\"tool_result\",\"is_error\":true,\"content\":\"npm ERR! ECONNRESET\"}]}}"
    done_now ;;
  rattle)
    [ "$n" -ge 2 ] && done_now
    trap 'fails "Transport: ECONNRESET: The socket connection was closed"; exit 1' INT
    sleep 60 & wait ;;
  *) done_now ;;
esac
"""


@pytest.fixture(autouse=True)
def _stand_in(deck, monkeypatch):
    (deck["tmp"] / "bin" / "claude").write_text(STAND_IN)
    monkeypatch.setenv("FD_CREW_RETRY_WAIT_SECONDS", "0")


def _events(task: dict, *kinds: str) -> list[str]:
    return [e["text"] for e in crew_store.read_events(task["user_id"], task["id"]) if not kinds or e["kind"] in kinds]


def _tries(task: dict) -> int:
    git = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=task["worktree"], capture_output=True, text=True, check=True).stdout.strip()
    return int((Path(git) / "tries").read_text())


async def _ticks(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        await crew_runner.tick()
        await asyncio.sleep(0.05)


# ── A turn that ends on it ───────────────────────────────────────────────


async def test_a_turn_that_fails_on_the_model_goes_again_and_finishes(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:overloaded"))
    assert done["status"] == "ready" and done["note"] == "it worked" and _tries(done) == 2
    assert not _events(done, "error"), "it got past it, so nothing is shown as an error"
    said = _events(done, "note")
    assert any("Overloaded" in s and "Trying again in 0 seconds (1 of 3)" in s for s in said)
    assert crew_runner._RETRYING in _seen(done) and "--resume sess-1" in _seen(done), "its session goes on, and it's told why"
    assert done["retries"] == 0 and done["retry_at"] == 0, "a good turn starts the count over"
    assert len(_events(done, "turn")) == 1, "nobody said anything new, so there's no new message to show"


async def test_it_tries_so_many_times_then_says_so_and_can_be_told_to_go_on(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:always"))
    assert done["status"] == "failed" and _tries(done) == crew_runner._RETRIES + 1
    assert "Overloaded" in done["note"] and "Tried again 3 times" in done["note"] and "send it a message" in done["note"]
    assert [s for s in _events(done, "note") if "(3 of 3)" in s]
    again = await crew_runner.steer(done["user_id"], done["id"], "go on")
    assert again["retries"] == 0, "someone asking is a new start"


async def test_trying_again_wouldnt_help_so_it_doesnt(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:badkey"))
    assert done["status"] == "failed" and _tries(done) == 1 and "401" in done["note"]
    assert _events(done, "error"), "and this one is an error"


async def test_it_waits_between_tries_and_a_message_doesnt(deck, monkeypatch):
    monkeypatch.setenv("FD_CREW_RETRY_WAIT_SECONDS", "1000")
    task = await _dispatch(deck, "SCENARIO:overloaded")
    await _ticks(1.0)
    waiting = crew_store.get_store().get(task["user_id"], task["id"])
    assert waiting["status"] == "queued" and waiting["retry_at"] > time.time() + 900 and _tries(waiting) == 1
    assert "Trying again in 1000 seconds" in waiting["note"]
    await crew_runner.steer(task["user_id"], task["id"], "go on now")
    done = await _settle(task)
    assert done["status"] == "ready" and _tries(done) == 2 and "go on now" in _seen(done)


async def test_it_waits_longer_each_time():
    assert [crew_runner._retry_wait_seconds(n) for n in range(3)] == [0.0, 0.0, 0.0]
    os.environ["FD_CREW_RETRY_WAIT_SECONDS"] = "20"
    try:
        assert [crew_runner._retry_wait_seconds(n) for n in range(3)] == [20.0, 60.0, 180.0]
    finally:
        os.environ["FD_CREW_RETRY_WAIT_SECONDS"] = "0"


async def test_stopping_it_while_it_waits_is_the_end_of_that(deck, monkeypatch):
    monkeypatch.setenv("FD_CREW_RETRY_WAIT_SECONDS", "0.3")
    task = await _dispatch(deck, "SCENARIO:always")
    await _ticks(0.2)
    await crew_runner.stop(task["user_id"], task["id"])
    await _ticks(0.6)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    assert fresh["status"] == "stopped" and _tries(fresh) == 1


# ── A turn that doesn't end ──────────────────────────────────────────────


async def test_a_worker_that_failed_on_the_model_and_went_quiet_is_started_again(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_error_wait_seconds", lambda: 0.3)
    done = await _settle(await _dispatch(deck, "SCENARIO:stuck"))
    assert done["status"] == "ready" and _tries(done) == 2 and crew_runner._RETRYING in _seen(done)
    assert any("hasn't got past it" in s and "ECONNRESET" in s and "(1 of 3)" in s for s in _events(done, "note"))


async def test_a_worker_gets_time_to_get_past_it_by_itself(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_error_wait_seconds", lambda: 1000.0)
    task = await _dispatch(deck, "SCENARIO:stuck")
    await _ticks(0.6)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    assert fresh["status"] == "running" and _tries(fresh) == 1
    assert not _events(fresh, "error") and any("trying again" in s for s in _events(fresh, "note"))


async def test_one_that_gets_past_it_isnt_touched(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_error_wait_seconds", lambda: 0.1)
    done = await _settle(await _dispatch(deck, "SCENARIO:recovers"))
    assert done["status"] == "ready" and _tries(done) == 1 and not _events(done, "error")


async def test_a_worker_saying_nothing_with_nothing_running_is_taken_for_hung(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_hung_seconds", lambda task: 0.3)
    done = await _settle(await _dispatch(deck, "SCENARIO:hung"))
    assert done["status"] == "ready" and _tries(done) == 2
    assert any("said nothing for a long time" in s for s in _events(done, "note"))


async def test_a_worker_in_a_long_command_is_left_to_it(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_hung_seconds", lambda task: 0.2)
    monkeypatch.setattr(crew_runner, "_stall_seconds", lambda: 0.2)
    task = await _dispatch(deck, "SCENARIO:busy")
    await _ticks(0.8)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    assert fresh["status"] == "running" and _tries(fresh) == 1 and fresh["stalled_at"]
    assert any("It may be stuck, or in one long command" in s for s in _events(fresh, "note")), "it's only said, as before"
    assert deck["told"] and deck["told"][-1]["stalled"]


async def test_one_that_never_gets_going_runs_out_of_tries_too(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_error_wait_seconds", lambda: 0.1)
    (deck["tmp"] / "bin" / "claude").write_text(STAND_IN.replace('[ "$n" -ge 2 ] && done_now; fails "Transport', 'fails "Transport'))
    done = await _settle(await _dispatch(deck, "SCENARIO:stuck"), deadline=40)
    assert done["status"] == "failed" and "Tried again 3 times" in done["note"] and "ECONNRESET" in done["note"]
    assert _tries(done) == 4


async def test_its_model_being_heard_from_means_its_not_hung(deck, monkeypatch):
    from captain_claw.flight_deck import crew_model

    monkeypatch.setattr(crew_runner, "_hung_seconds", lambda task: 0.5)
    task = await _dispatch(deck, "SCENARIO:hung")
    end = time.monotonic() + 1.2
    while time.monotonic() < end:
        crew_model._heard(task["id"])  # a long reply coming in, with the CLI saying nothing till it's done
        await crew_runner.tick()
        await asyncio.sleep(0.05)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    assert fresh["status"] == "running" and _tries(fresh) == 1
    crew_model._CALLS.clear()


# ── Saying it's still at it ──────────────────────────────────────────────


async def test_a_quiet_worker_is_said_to_be_working_every_so_often(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_working_every_seconds", lambda: 0.3)
    task = await _dispatch(deck, "SCENARIO:busy")
    await _ticks(0.2)
    started = crew_store.get_store().get(task["user_id"], task["id"])["last_event_at"]
    await _ticks(1.1)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    said = _events(fresh, "working")
    assert 2 <= len(said) <= 5 and all(s.startswith("Still working on it: a command or tool is running") for s in said)
    assert fresh["last_event_at"] == started, "saying so doesn't count as the worker having said anything"
    assert not fresh["stalled_at"]


async def test_an_agent_asking_about_a_task_sees_what_it_did_not_a_page_of_still_working(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_working_every_seconds", lambda: 0.05)
    task = await _dispatch(deck, "SCENARIO:busy")
    await _ticks(1.2)
    fresh = crew_store.get_store().get(task["user_id"], task["id"])
    assert len(_events(fresh, "working")) >= 4
    recent = crew_runner.recent_activity(fresh, keep=3)
    kinds = [e["kind"] for e in recent]
    assert kinds.count("working") == 1 and kinds[-1] == "working" and "turn" in kinds, "the newest word, and the task itself still there"


async def test_it_isnt_said_while_the_worker_is_talking_or_after_its_done(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_working_every_seconds", lambda: 1000.0)
    done = await _settle(await _dispatch(deck, "SCENARIO:work"))
    assert not _events(done, "working")
    monkeypatch.setattr(crew_runner, "_working_every_seconds", lambda: 0.01)
    await _ticks(0.2)
    assert not _events(done, "working"), "nothing is running"


async def test_it_says_what_it_can_see(deck, monkeypatch):
    from captain_claw.flight_deck import crew_model

    now = time.time()
    task = {"id": "t1", "harness": "claude", "pid": 0, "last_event_at": now - 200, "model_source": {"kind": "tier"}}
    monkeypatch.setitem(crew_model._CALLS, "t1", {"open": 1.0, "since": now - 125, "heard": now - 2, "got": 41000.0})
    said = await crew_runner._still_working(task, TurnState(), now)
    assert said == "Still working on it: the model is writing its reply (2 min so far, 41 KB received)."
    monkeypatch.setitem(crew_model._CALLS, "t1", {"open": 1.0, "since": now - 40, "heard": now - 40, "got": 0.0})
    assert "waiting for the model to start answering (40 seconds so far)" in await crew_runner._still_working(task, TurnState(), now)
    monkeypatch.setitem(crew_model._CALLS, "t1", {"open": 1.0, "since": now - 300, "heard": now - 120, "got": 500.0})
    assert "waiting for the model, which has paused for 2 min" in await crew_runner._still_working(task, TurnState(), now)
    monkeypatch.setitem(crew_model._CALLS, "t1", {"open": 0.0, "since": now - 300, "heard": now - 120, "got": 500.0})
    assert "nothing new to show for 3 min" in await crew_runner._still_working(task, TurnState(), now)
    failed = TurnState(error="Transport: ECONNRESET")
    assert "trying the model again" in await crew_runner._still_working(task, failed, now)
    assert "waiting on its model" in await crew_runner._still_working({**task, "id": "t2", "model_source": {}}, TurnState(), now)


def test_how_long_reads_well():
    assert [crew_runner._how_long(s) for s in (5, 89, 90, 3599, 3720, 7500)] == ["5 seconds", "89 seconds", "2 min", "1 h 0 min", "1 h 2 min", "2 h 5 min"]


def test_how_long_it_can_be_quiet_depends_on_whether_the_deck_can_see_its_model():
    assert crew_runner._hung_seconds({"model_source": {"kind": "tier"}}) == crew_runner._stall_seconds()
    assert crew_runner._hung_seconds({"model_source": {}}) == crew_runner._stall_seconds() * 3, "a slow local model, on one long reply"


# ── What's shown ─────────────────────────────────────────────────────────


async def test_a_command_failing_on_a_connection_is_still_that_commands_error(deck):
    done = await _settle(await _dispatch(deck, "SCENARIO:toolfail"))
    assert done["status"] == "ready" and any("ECONNRESET" in s for s in _events(done, "error"))
    assert not any("trying again" in s for s in _events(done, "note"))


async def test_what_it_says_while_its_being_stopped_isnt_shown(deck):
    """On the real deck that started this: "ECONNRESET" in red, from someone telling it to go on."""
    task = await _dispatch(deck, "SCENARIO:rattle")
    await asyncio.sleep(0.5)
    await crew_runner.steer(task["user_id"], task["id"], "go on", now=True)
    done = await _settle(task)
    assert done["status"] == "ready"
    assert not [s for s in _events(done) if "ECONNRESET" in s]
    assert "Stopped what it was doing, to hear this." in _events(done, "note")


@pytest.mark.parametrize("text", [
    "OpenAI Chat stream ended without finish_reason",
    "Transport: ECONNRESET: The socket connection was closed unexpectedly.",
    "Reconnecting... 1/5 (stream disconnected before completion: stream closed before response.completed)",
    '500: {"message":"upstream exploded"}',
    "API Error: 529 Overloaded",
    "unexpected status 503 Service Unavailable",
    "Request timed out",
    "fetch failed",
    "Rate limit exceeded, please retry",
])
def test_what_trying_again_could_get_past(text):
    assert crew_harness.passing(text)


@pytest.mark.parametrize("text", [
    "",
    "the turn failed",
    "Free usage limit reached. Connect a provider or subscribe to OpenCode Go to keep going, or switch models.",
    "You've used today's 5M AI tokens on the Premium plan. It resets at midnight UTC.",
    "Failed to authenticate. API Error: 401",
    "unexpected status 402: You've used today's 478k AI tokens",
    "Your credit balance is too low",
    "prompt is too long: context length exceeded",
    "400: model not found",
    "File not found: /tmp/x.png",
    "You've hit your limit, resets 5pm",
])
def test_what_it_couldnt(text):
    assert not crew_harness.passing(text)


@pytest.mark.parametrize("harness, line", [
    ("claude", {"type": "result", "is_error": True, "result": "API Error: 529"}),
    ("codex", {"type": "turn.failed", "error": {"message": "stream disconnected"}}),
    ("codex", {"type": "error", "message": "stream disconnected"}),
    ("codex", {"type": "item.completed", "item": {"type": "error", "message": "Reconnecting... 1/5"}}),
    ("opencode", {"type": "error", "error": {"data": {"message": "ECONNRESET"}}}),
    ("pi", {"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": "500"}}),
])
def test_each_cli_says_which_errors_are_about_its_model(harness, line):
    event = crew_harness.parse_line(harness, json.dumps(line), TurnState())[-1]
    assert event["kind"] == "error" and event["about"] == "model"


@pytest.mark.parametrize("harness, line", [
    ("claude", {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True, "content": "ECONNRESET"}]}}),
    ("codex", {"type": "item.completed", "item": {"type": "command_execution", "exit_code": 1, "aggregated_output": "ECONNRESET"}}),
    ("opencode", {"type": "tool_use", "part": {"tool": "bash", "state": {"status": "error", "error": "ECONNRESET"}}}),
    ("pi", {"type": "tool_execution_end", "isError": True, "result": "ECONNRESET"}),
])
def test_and_a_commands_failure_isnt(harness, line):
    events = crew_harness.parse_line(harness, json.dumps(line), TurnState())
    assert events[-1]["kind"] == "error" and "about" not in events[-1]


def test_it_going_on_after_a_model_error_clears_it():
    state = TurnState()
    crew_harness.parse_line("opencode", json.dumps({"type": "error", "error": {"data": {"message": "ECONNRESET"}}}), state)
    assert state.error
    crew_harness.parse_line("opencode", json.dumps({"type": "tool_use", "part": {"tool": "bash", "state": {"input": {"command": "ls"}}}}), state)
    assert not state.error


def test_a_command_running_under_it_is_told_from_the_cli_itself(tmp_path):
    cli = tmp_path / "claude"
    cli.write_text('#!/bin/sh\nmkfifo "$1"\nread _ < "$1"\n')
    cli.chmod(0o755)
    alone = subprocess.Popen(["/bin/sh", "-c", f'"{cli}" "{tmp_path}/fifo"; true', "crew-turn"], start_new_session=True)
    busy = subprocess.Popen(["/bin/sh", "-c", 'sleep 30; true', "crew-turn"], start_new_session=True)
    try:
        time.sleep(0.5)
        assert crew_runner._running_a_command({"pid": alone.pid, "harness": "claude"}) is False
        assert crew_runner._running_a_command({"pid": busy.pid, "harness": "claude"}) is True
        assert crew_runner._running_a_command({"pid": 0, "harness": "claude"}) is False
    finally:
        for proc in (alone, busy):
            os.killpg(proc.pid, 9)
            proc.wait()
