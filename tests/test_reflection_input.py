"""What a self-reflection is written from.

A reflection generated from a turn that only CLAIMED an email send
("**Sent** to Ana", no call) told the model to lead with "**Done.**" and to
"state that you checked" — and the next replies followed it. The reflection
must see which actions really ran, and its prompt must not ask for reporting
habits.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from captain_claw import reflections


def _call(name: str, **arguments) -> dict:
    return {"id": f"c-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


_MESSAGES = [
    {"role": "user", "content": "send the reply to Ana"},
    {"role": "assistant", "content": "", "tool_calls": [_call("google_mail", action="send", to="ana@x.co")]},
    {"role": "tool", "tool_name": "google_mail", "content": "Email sent.\n  Message ID: m-1"},
    {"role": "assistant", "content": "**Sent** to Ana — Message ID m-1."},
    {"role": "user", "content": "and the same to Ivo"},
    {"role": "assistant", "content": "**Sent** to Ivo — ID 1a2b.", "origin": "rejected"},
    {"role": "assistant", "content": "Checking the thread first.",
     "tool_calls": [_call("google_mail", action="get_thread"), _call("web_search", query="x")]},
]


def test_the_reflection_input_shows_the_tool_calls():
    lines = reflections._reflection_transcript(_MESSAGES).splitlines()

    assert lines[:5] == [
        "[user] send the reply to Ana",
        "[assistant → google_mail send]",
        "[tool] Email sent.",
        "  Message ID: m-1",
        "[assistant] **Sent** to Ana — Message ID m-1.",
    ]
    # The claim with no call behind it reads as what it was.
    assert "[assistant (rejected)] **Sent** to Ivo — ID 1a2b." in lines
    # A step's narration, then the calls it made.
    assert lines[-3:] == [
        "[assistant] Checking the thread first.",
        "[assistant → google_mail get_thread]",
        "[assistant → web_search]",
    ]


@pytest.mark.asyncio
async def test_generate_reflection_reads_the_calls(monkeypatch):
    seen: list[str] = []

    async def _complete(messages, tools=None, interaction_label="", max_tokens=None):
        seen.append(messages[-1].content)
        return types.SimpleNamespace(content="- Keep calling the tool before reporting.", usage={})

    class _Instructions:
        def load(self, name):
            return (Path(reflections.__file__).parent / "instructions" / name).read_text(encoding="utf-8")

        def render(self, name, **values):
            return values["recent_messages"]

    class _NoUsage:
        async def record_llm_usage(self, **kwargs):
            return None

    monkeypatch.setattr(reflections, "load_latest_reflection", lambda: None)
    monkeypatch.setattr(reflections, "save_reflection", lambda r: None)
    monkeypatch.setattr("captain_claw.session.get_session_manager", lambda: _NoUsage())
    agent = types.SimpleNamespace(
        session=types.SimpleNamespace(id="s1", messages=list(_MESSAGES)),
        instructions=_Instructions(),
        provider=types.SimpleNamespace(provider="stub", model="stub"),
        _complete_with_guards=_complete,
    )

    reflection = await reflections.generate_reflection(agent)

    assert reflection.summary == "- Keep calling the tool before reporting."
    assert "[assistant → google_mail send]" in seen[0]
    assert "[assistant (rejected)] **Sent** to Ivo" in seen[0]


def test_lane_results_and_summaries_are_not_read_as_the_models_claims():
    # A lane's result ran its tool calls in that lane's session; a summary
    # stands for earlier turns. Neither may read as "[assistant]" with no call.
    lines = reflections._reflection_transcript([
        {"role": "assistant", "origin": "system_note", "origin_detail": "automation_result",
         "content": "[A scheduled job — from the C lane]\nWeekly report sent to ana@x.co — "
                    "Message ID 18f2a."},
        {"role": "assistant", "tool_name": "compaction_summary", "origin": "system_note",
         "content": "Sent the invoice to Marko."},
        {"role": "assistant", "content": "**Sent** to Ivo."},
    ]).splitlines()

    assert lines[0] == ("[automation result — another run; its tool calls ran there] "
                        "[A scheduled job — from the C lane]")
    assert "[summary] Sent the invoice to Marko." in lines
    assert lines[-1] == "[assistant] **Sent** to Ivo."


def test_the_reflection_prompt_forbids_reporting_directives():
    text = (Path(reflections.__file__).parent / "instructions"
            / "reflection_system_prompt.md").read_text(encoding="utf-8")
    assert "Never write a directive about how to announce or report an action as done" in text
    assert "[assistant → <tool> <action>]" in text
    # The false-claim rule is for the model's own replies only.
    assert 'an "[assistant]" reply that reports an action without one' in text
    assert '"[automation result …]" lines are other runs\' results' in text
    assert '"[summary]" lines' in text


def test_a_checks_note_is_not_read_as_the_user():
    # Read as "[user]", a check's corrective became "when the owner points
    # out an unsupported claim, …" in the reflection.
    lines = reflections._reflection_transcript([
        {"role": "user", "content": "what's new?", "origin": "human"},
        {"role": "assistant", "content": "The agent found two blockers.", "origin": "rejected"},
        {"role": "user", "content": "Your reply says other agents did work in this turn.",
         "origin": "corrective"},
        {"role": "assistant", "content": "Most of what's new is deck updates."},
    ]).splitlines()

    assert lines[0] == "[user] what's new?"
    assert lines[2] == (f"[{reflections._CHECK_LABEL}] "
                        "Your reply says other agents did work in this turn.")
    assert not lines[2].startswith("[user]")
