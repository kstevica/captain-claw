"""The Harbor adapter's pure parts. Needs Harbor installed (skipped otherwise):

    PYTHONPATH=evals/harbor python -m pytest evals/harbor/test_captain_harbor.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("harbor")

import captain_harbor as ch  # noqa: E402


def test_split_model_keeps_slashes_in_the_model():
    assert ch.split_model("anthropic/claude-haiku-5-5") == ("anthropic", "claude-haiku-5-5")
    assert ch.split_model("openrouter/deepseek/deepseek-v4-pro") == ("openrouter", "deepseek/deepseek-v4-pro")
    with pytest.raises(ValueError):
        ch.split_model("claude-haiku-5-5")


def test_runner_flags_deck_by_default_standalone_on_request():
    deck = ch.runner_flags(
        "anthropic", "m", ch.CaptainOptions(reasoning_effort="max", drive="ambitious", archetype="a"), "/logs/agent")
    assert deck[:4] == ["-p", "anthropic", "-m", "m"]
    assert ["--drive", "ambitious"] == deck[deck.index("--drive"):][:2]
    with pytest.raises(ValueError):
        ch.CaptainOptions(drive="frantic")
    assert "--standalone" not in deck and ["--archetype", "a"] == deck[deck.index("--archetype"):][:2]
    assert ["--effort", "max"] == deck[deck.index("--effort"):][:2]
    assert "/logs/agent/captain-result.json" in deck
    alone = ch.runner_flags("anthropic", "m", ch.CaptainOptions(standalone=True, archetype="a"), "/logs/agent")
    assert "--standalone" in alone and "--bench" in alone and "--archetype" not in alone


def test_context_from_result():
    ctx = ch.context_from_result({
        "ok": True, "mode": "deck", "cost_usd_estimate": 0.01, "tool_calls": 3,
        "usage": {"prompt_tokens": 100, "completion_tokens": 9, "cache_read_input_tokens": 80},
        "messages": [{"role": "user"}],
    })
    assert ctx["n_input_tokens"] == 100 and ctx["n_output_tokens"] == 9 and ctx["n_cache_tokens"] == 80
    assert ctx["cost_usd"] == 0.01
    assert ctx["metadata"]["mode"] == "deck" and "messages" not in ctx["metadata"]


def test_a_harbor_trial_runs_at_medium_and_locked_by_default():
    flags = ch.runner_flags("anthropic", "m", ch.CaptainOptions(), "/logs/agent")
    assert ["--effort", "medium"] == flags[flags.index("--effort"):][:2]
    assert ["--drive", "ambitious"] == flags[flags.index("--drive"):][:2]
    assert "--lock" in flags


def test_auto_leaves_a_level_to_captain_and_lock_can_be_turned_off():
    flags = ch.runner_flags("anthropic", "m", ch.CaptainOptions(reasoning_effort="auto", drive="persistent"), "/l")
    assert "--effort" not in flags and ["--drive", "persistent"] == flags[flags.index("--drive"):][:2]
    assert "--lock" in flags
    nothing = ch.runner_flags("anthropic", "m", ch.CaptainOptions(reasoning_effort="auto", drive="auto"), "/l")
    assert not {"--effort", "--drive", "--lock"} & set(nothing)
    loose = ch.runner_flags("anthropic", "m", ch.CaptainOptions(lock=False), "/l")
    assert "--effort" in loose and "--lock" not in loose


def test_a_tasks_skills_folder_reaches_the_runner():
    assert ch.Captain.capabilities.skills is True
    flags = ch.runner_flags("anthropic", "m", ch.CaptainOptions(), "/l", "/tmp_workspace/.wildclaw/skills")
    assert ["--skills-dir", "/tmp_workspace/.wildclaw/skills"] == flags[flags.index("--skills-dir"):][:2]
    assert "--skills-dir" not in ch.runner_flags("anthropic", "m", ch.CaptainOptions(), "/l")


def test_the_trajectory_is_valid_atif_for_harbor():
    from harbor.models.trajectories.trajectory import Trajectory

    from captain_claw.bench.atif import to_atif

    traj = to_atif([
        {"role": "user", "content": "do it", "timestamp": "2026-11-26T17:29:34.813986+00:00"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "shell", "arguments": "{\"command\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "a.txt"},
        {"role": "assistant", "content": "Done."},
    ], session_id="s", model="anthropic/claude-haiku-5-5", version="0.9.3",
        usage={"prompt_tokens": 5, "completion_tokens": 1}, cost_usd=0.0)
    Trajectory.model_validate(traj)
    flags = ch.runner_flags("anthropic", "m", ch.CaptainOptions(), "/logs/agent")
    assert ["--trajectory-json", "/logs/agent/trajectory.json"] == flags[flags.index("--trajectory-json"):][:2]


def test_a_local_model_server_is_passed_through():
    opts = ch.CaptainOptions(endpoint="http://192.168.0.2:8080/v1", max_context=200000, max_output=16000)
    flags = ch.runner_flags("openai", "qwen3.8-flash-next-iq3_s", opts, "/l")
    assert ["--base-url", "http://192.168.0.2:8080/v1"] == flags[flags.index("--base-url"):][:2]
    assert ["--max-context", "200000"] == flags[flags.index("--max-context"):][:2]
    assert ["--max-output", "16000"] == flags[flags.index("--max-output"):][:2]


def test_the_pdf_extra_is_installed_by_default_on_every_kind_of_spec():
    assert ch.CaptainOptions().extras == "pdf"
    assert ch.with_extras("captain-claw==0.9.3", "pdf") == "captain-claw[pdf]==0.9.3"
    assert ch.with_extras("captain-claw", "pdf") == "captain-claw[pdf]"
    assert ch.with_extras("captain-claw[vision]>=0.9", "pdf") == "captain-claw[vision,pdf]>=0.9"
    assert ch.with_extras("/tmp/captain_claw-0.9.3-py3-none-any.whl", "pdf") == (
        "captain-claw[pdf] @ file:///tmp/captain_claw-0.9.3-py3-none-any.whl")
    assert ch.with_extras("git+https://github.com/x/y.git", "pdf,cv") == (
        "captain-claw[pdf,cv] @ git+https://github.com/x/y.git")
    assert ch.with_extras("captain-claw==0.9.3", "none") == "captain-claw==0.9.3"
    assert ch.with_extras("captain-claw==0.9.3", "") == "captain-claw==0.9.3"


def test_a_timed_out_runner_is_stopped_and_given_time_to_write_its_result():
    cmd = ch.stop_runner_command("/logs/agent/captain-result.json")
    # TERM first (the runner cancels its task and writes result + transcript), KILL only after the grace.
    assert cmd.index("pkill -TERM") < cmd.index("pkill -KILL")
    assert f"seq 1 {ch.STOP_GRACE_SECONDS}" in cmd
    # Only the runner's own Python process, not the shell or tee around it.
    assert ch.RUNNER_PROCESS.startswith("^/opt/captain-claw/venv/bin/python")
    assert cmd.rstrip().endswith("true")
