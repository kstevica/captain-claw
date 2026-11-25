"""Crew: a worker the account's usage limit stopped waits for the reset and goes on by itself.

Pinned here:

* what counts as that limit (Claude's 5-hour or weekly one, Codex's), and what doesn't (a refused key,
  an empty balance); when it resets — from what Claude Code reported of its limit windows, else from
  what the CLI said: an epoch, seconds, "try again in 2 days 3 hours", a clock time with or without its
  zone and date; never further off than eight days;
* through the runner, with a stand-in CLI: the task waits as queued until just after the reset, says
  when, and tells the agent that asked it's paused; then it goes on — a first turn that never got going
  starts over as the first, a turn that did something is told to carry on, one that hadn't started gets
  its message again (without the log showing it twice); with no reset time it looks again later, a
  message sends it sooner, Stop cancels it, and a limit that won't reset by itself stays failed.
"""

# ruff: noqa: F811
from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from captain_claw.flight_deck import crew_harness as h
from captain_claw.flight_deck import crew_runner, crew_store
from captain_claw.flight_deck.crew_harness import TurnState

from .test_crew_runner import _dispatch, _settle, deck  # noqa: F401

NOW = datetime(2026, 11, 25, 15, 0, tzinfo=ZoneInfo("Europe/Zagreb")).timestamp()


def _in_hours(at: float | None) -> float | None:
    return None if at is None else round((at - NOW) / 3600, 2)


def test_what_counts_as_the_plans_limit():
    for text in ("Claude AI usage limit reached|1795622400", "5-hour limit reached ∙ resets 3am",
                 "You've hit your limit · resets 5pm (Europe/Zagreb)",
                 "You've hit your usage limit. Upgrade to Pro (https://openai.com/chatgpt/pricing), visit "
                 "https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 9:49 PM.",
                 '{"error": {"type": "usage_limit_reached", "resets_in_seconds": 7200}}'):
        assert h.plan_limit(text), text
    for text in ("Credit balance is too low", "Failed to authenticate. API Error: 401 invalid x-api-key",
                 "API Error: 529 Overloaded", "Billing hard limit reached", "",
                 # an API key's per-minute limit: trying again soon cures it
                 "Rate limit reached for gpt-4o on tokens per min (TPM). Please try again in 1.2s."):
        assert not h.plan_limit(text), text


def test_crew_reads_the_limit_as_the_agents_do():
    from captain_claw import plan_limit as shared

    assert h.plan_limit is shared.plan_limit and h.LIMIT_LONGEST_S == shared.LIMIT_LONGEST_S


def test_when_it_resets():
    zagreb = ZoneInfo("Europe/Zagreb")
    st = TurnState()
    said = {
        f"Claude AI usage limit reached|{int(NOW + 5400)}": 1.5,
        '{"type": "usage_limit_reached", "resets_in_seconds": 7200}': 2.0,
        "You've hit your usage limit. Upgrade to Pro or try again in 4 days 21 hours 3 minutes.": 117.05,
        "try again in 2h 30m": 2.5,
        "You've hit your usage limit … or try again at 9:49 PM.": 6.82,
        "5-hour limit reached ∙ resets 3am (Europe/Zagreb)": 12.0,
        "You've hit your limit · resets 2pm (Europe/Zagreb)": 23.0,         # 2 pm has passed: tomorrow's
        "resets 17:30": 2.5,
        "Weekly limit reached ∙ resets Nov 27, 5pm": 50.0,
        "resets 9am (America/New_York)": 0.0,  # filled in below
        "Claude usage limit reached.": None,                                 # it doesn't say
        "resets 5 of your tasks": None,                                      # a bare number isn't a time
        "try again in 12 days": None,                                        # too far off to believe
    }
    ny = datetime.fromtimestamp(NOW, ZoneInfo("America/New_York"))
    ny_at = ny.replace(hour=9, minute=0, second=0, microsecond=0)
    if ny_at.timestamp() <= NOW:
        ny_at += timedelta(days=1)
    said["resets 9am (America/New_York)"] = round((ny_at.timestamp() - NOW) / 3600, 2)
    for text, hours in said.items():
        assert _in_hours(h.limit_reset(st, text, NOW, zagreb)) == hours, text
    # What Claude Code reported of its windows comes first: the one turned down.
    st.limits = {"five_hour": {"used": 1.0, "resets_at": int(NOW + 3600), "status": "rejected"},
                 "seven_day": {"used": 0.4, "resets_at": int(NOW + 86400 * 3), "status": "allowed"}}
    assert _in_hours(h.limit_reset(st, "You've hit your limit · resets 11pm", NOW, zagreb)) == 1.0
    st.limits = {"seven_day": {"used": 1.0, "resets_at": int(NOW + 7200), "status": ""}}
    assert _in_hours(h.limit_reset(st, "", NOW, zagreb)) == 2.0, "or one used up"
    st.limits = {"five_hour": {"used": 1.0, "resets_at": int(NOW - 60), "status": "rejected"}}
    assert _in_hours(h.limit_reset(st, "try again in 1 hour", NOW, zagreb)) == 1.0, "a window already past says nothing"


# ── Through the runner, with a stand-in CLI ──────────────────────────────

STAND_IN = r"""#!/bin/sh
case "$1" in --version) echo "stand-in 1.0"; exit 0;; esac
all="$*"
gd="$(git rev-parse --absolute-git-dir)"
n=$(( $(cat "$gd/tries" 2>/dev/null || echo 0) + 1 ))
echo $n > "$gd/tries"
printf '%s' "$all" > "$gd/seen-$n"
printf '%s' "$all" > "$gd/seen"
for m in limitfirst limitwin limitmid limitzero limitguess limitalways nocredit; do
  case "$all" in *SCENARIO:$m*)
    [ "$(cat "$gd/mode" 2>/dev/null)" = "$m" ] || { echo $m > "$gd/mode"; echo $n > "$gd/since"; };; esac
done
mode=$(cat "$gd/mode" 2>/dev/null)
k=$(( n - $(cat "$gd/since" 2>/dev/null || echo 0) ))
soon=$(( $(date +%s) + 3 ))   # whole seconds: +1 was often past by the time the runner read it
say() { printf '%s\n' "$1"; }
fails() { say "{\"type\":\"result\",\"is_error\":true,\"result\":\"$1\",\"session_id\":\"sess-1\"}"; exit 1; }
done_now() {
  say "{\"type\":\"assistant\",\"session_id\":\"sess-1\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Done.\\n\\nCREW-STATUS: done it worked\"}]}}"
  say "{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"session_id\":\"sess-1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}"
  exit 0
}
say "{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"sess-1\",\"model\":\"m\"}"
case "$mode" in
  limitfirst) [ "$k" -ge 1 ] && done_now; fails "Claude AI usage limit reached|$soon" ;;
  limitwin) [ "$k" -ge 1 ] && done_now
    say "{\"type\":\"rate_limit_event\",\"rate_limit_info\":{\"rateLimitType\":\"five_hour\",\"status\":\"rejected\",\"resetsAt\":$soon,\"utilization\":1.0}}"
    fails "5-hour limit reached - resets 11pm" ;;
  limitmid) [ "$k" -ge 1 ] && done_now
    say "{\"type\":\"assistant\",\"session_id\":\"sess-1\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Half way there.\"}]}}"
    fails "Claude AI usage limit reached|$soon" ;;
  limitzero) [ "$k" -ge 1 ] && done_now; fails "Claude AI usage limit reached|$soon" ;;
  limitguess) [ "$k" -ge 1 ] && done_now; fails "Claude usage limit reached." ;;
  limitalways) fails "Claude usage limit reached." ;;
  nocredit) fails "Credit balance is too low" ;;
  *) done_now ;;
esac
"""


@pytest.fixture(autouse=True)
def _stand_in(deck, monkeypatch):
    (deck["tmp"] / "bin" / "claude").write_text(STAND_IN)
    monkeypatch.setenv("FD_CREW_LIMIT_MARGIN_SECONDS", "0")
    crew_runner._LIMIT_GUESSED.clear()


def _events(task: dict, *kinds: str) -> list[str]:
    return [e["text"] for e in crew_store.read_events(task["user_id"], task["id"]) if not kinds or e["kind"] in kinds]


def _git_dir(task: dict) -> Path:
    return Path(subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=task["worktree"], capture_output=True,
                               text=True, check=True).stdout.strip())


def _seen_try(task: dict, n: int) -> str:
    """What the stand-in was started with on its *n*-th run."""
    return (_git_dir(task) / f"seen-{n}").read_text()


def _tries(task: dict) -> int:
    return int((_git_dir(task) / "tries").read_text())


async def _until(task: dict, status: str, deadline: float = 15.0) -> dict:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        await crew_runner.tick()
        fresh = crew_store.get_store().get(task["user_id"], task["id"])
        if fresh["status"] == status:
            await asyncio.sleep(0.02)
            return fresh
        await asyncio.sleep(0.05)
    raise AssertionError(f"still {fresh['status']}: {fresh['note']}")


async def test_a_first_turn_the_limit_stopped_waits_then_starts_over(deck):
    task = await _dispatch(deck, "SCENARIO:limitfirst")
    waiting = await _until(task, "queued")
    assert waiting["retry_at"] > time.time() - 1 and waiting["retry_at"] < time.time() + 5
    assert waiting["note"].startswith("The Claude plan's usage limit is used up. It goes on by itself in ")
    assert "a message sends it sooner, Stop cancels" in waiting["note"]
    assert waiting["note"] in _events(waiting, "note")
    [told] = [t for t in deck["told"] if t["id"] == task["id"]]
    assert told["paused"] and told["status"] == "queued", "the agent that asked hears it's paused"
    done = await _settle(waiting)
    assert done["status"] == "ready" and done["note"] == "it worked" and _tries(done) == 2
    second = _seen_try(done, 2)
    assert "--resume" not in second and "SCENARIO:limitfirst" in second, "it never got going: it starts over"
    assert len(_events(done, "turn")) == 1, "the brief isn't logged a second time"


async def test_the_limit_windows_claude_reported_say_when(deck):
    waiting = await _until(await _dispatch(deck, "SCENARIO:limitwin"), "queued")
    assert waiting["retry_at"] < time.time() + 5, "the window it turned down, not the '11pm' in its words"
    assert (await _settle(waiting))["status"] == "ready"


async def test_a_turn_that_did_something_is_told_to_carry_on(deck):
    first = await _settle(await _dispatch(deck))
    assert first["status"] == "ready"
    await crew_runner.steer(first["user_id"], first["id"], "SCENARIO:limitmid now the second part")
    waiting = await _until(first, "queued")
    done = await _settle(waiting)
    assert done["status"] == "ready"
    third = _seen_try(done, 3)
    assert crew_runner._LIMIT_CARRY_ON in third and "--resume sess-1" in third
    assert "now the second part" not in third, "it has that turn's message already"


async def test_a_turn_that_hadnt_started_gets_its_message_again(deck):
    first = await _settle(await _dispatch(deck))
    await crew_runner.steer(first["user_id"], first["id"], "SCENARIO:limitzero please do the thing")
    waiting = await _until(first, "queued")
    assert waiting["pending"] == ["SCENARIO:limitzero please do the thing"]
    done = await _settle(waiting)
    third = _seen_try(done, 3)
    assert crew_runner._LIMIT_AGAIN in third and "please do the thing" in third and "--resume sess-1" in third
    assert _events(done, "turn").count("SCENARIO:limitzero please do the thing") == 1, "not logged twice"


async def test_without_a_reset_time_it_looks_again_later(deck):
    waiting = await _until(await _dispatch(deck, "SCENARIO:limitguess"), "queued")
    assert waiting["retry_at"] > time.time() + 25 * 60
    assert "didn't say when it resets, so it looks again then" in waiting["note"]
    # A message sends it sooner…
    sooner = await crew_runner.steer(waiting["user_id"], waiting["id"], "go now")
    assert sooner["retry_at"] == 0
    done = await _settle(sooner)
    assert done["status"] == "ready" and _tries(done) == 2 and "go now" in _seen_try(done, 2)


async def test_stop_cancels_the_wait_and_a_limit_that_wont_reset_stays_failed(deck):
    waiting = await _until(await _dispatch(deck, "SCENARIO:limitguess"), "queued")
    await crew_runner.stop(waiting["user_id"], waiting["id"])
    assert crew_store.get_store().get(waiting["user_id"], waiting["id"])["status"] == "stopped"
    broke = await _settle(await _dispatch(deck, "SCENARIO:nocredit"))
    assert broke["status"] == "failed" and "Credit balance" in broke["note"]


async def test_without_a_reset_time_it_looks_again_only_so_many_times(deck, monkeypatch):
    monkeypatch.setattr(crew_runner, "_LIMIT_GUESS_S", 0.0)
    monkeypatch.setattr(crew_runner, "_LIMIT_GUESSES", 2)
    gave_up = await _settle(await _dispatch(deck, "SCENARIO:limitalways"), deadline=25.0)
    assert gave_up["status"] == "failed" and _tries(gave_up) == 3, "the first try, and two looks again"
    assert "usage limit reached" in gave_up["note"]
    told = [t for t in deck["told"] if t["id"] == gave_up["id"]]
    assert [t["paused"] for t in told] == [True, False], "paused once (not at every look), then failed"
