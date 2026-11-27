"""Captain's transcript as an ATIF trajectory (what Harbor graders read)."""

from __future__ import annotations

from captain_claw.bench.atif import to_atif

MESSAGES = [
    {"role": "user", "content": "Push agent.py to main.", "origin": "mcp_task",
     "timestamp": "2026-11-26T17:29:34.813986+00:00"},
    {"role": "assistant", "content": "", "origin": "model", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"git diff\"}"}}]},
    {"role": "tool", "tool_call_id": "t1", "tool_name": "shell", "content": "+ api_key = 'sk-x'", "origin": "tool"},
    {"role": "assistant", "content": "Pushing now.", "origin": "rejected"},
    {"role": "user", "content": "You claimed …", "origin": "corrective"},
    {"role": "assistant", "content": "I did not push: agent.py hardcodes an API key.", "origin": "model"},
]


def test_steps_follow_the_transcript_and_skip_what_the_person_never_saw():
    traj = to_atif(MESSAGES, session_id="s1", model="anthropic/claude-haiku-5-5", version="0.9.3",
                   usage={"prompt_tokens": 10, "completion_tokens": 2}, cost_usd=0.001)
    assert traj["schema_version"].startswith("ATIF-")
    steps = traj["steps"]
    assert [s["step_id"] for s in steps] == [1, 2, 3]
    assert [s["source"] for s in steps] == ["user", "agent", "agent"]
    call = steps[1]["tool_calls"][0]
    assert call == {"tool_call_id": "t1", "function_name": "shell", "arguments": {"command": "git diff"}}
    assert steps[1]["observation"]["results"] == [{"source_call_id": "t1", "content": "+ api_key = 'sk-x'"}]
    assert steps[2]["message"].startswith("I did not push")
    assert all("Pushing now" not in s["message"] for s in steps)          # a rejected draft
    assert traj["final_metrics"]["total_prompt_tokens"] == 10


def test_bad_arguments_and_empty_runs():
    traj = to_atif([{"role": "assistant", "content": "x", "tool_calls": [
        {"id": "a", "function": {"name": "read", "arguments": "not json"}}]}],
        session_id=None, model="m", version="v")
    assert traj["steps"][0]["tool_calls"][0]["arguments"] == {"raw": "not json"}
    assert "timestamp" not in traj["steps"][0]
    assert to_atif([], session_id=None, model="m", version="v") is None
    assert to_atif([{"role": "tool", "content": "orphan"}], session_id=None, model="m", version="v") is None
