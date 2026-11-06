"""What the learning prompts keep out (taken from Hermes' review prompts):
claims that a tool doesn't work, failures caused by setup, errors a retry
fixed, attempts that never worked, and guesses about the user — and how a
wrong insight is corrected in place (same entity_key) rather than answered
with a rival."""

from __future__ import annotations

from pathlib import Path

import pytest

from captain_claw import insights
from captain_claw.insights import InsightsManager, _known_insight_line
from captain_claw.instructions import InstructionLoader

_BASE = Path(insights.__file__).parent / "instructions"


def _read(name: str) -> str:
    return (_BASE / name).read_text(encoding="utf-8")


def test_the_insight_prompt_keeps_out_what_hardens_into_refusals():
    text = _read("insight_extraction_system_prompt.md")
    assert "Claims that a tool or feature doesn't work" in text
    assert "harden into refusals" in text
    assert "Failures caused by setup or the environment" in text
    assert "Errors that went away before the conversation ended" in text
    assert "never write the dead ends up as a `workflow`" in text
    assert "Save only what the conversation shows" in text
    assert "A one-off request is not a standing preference" in text
    assert "SAME `entity_key`" in text


def test_the_micro_insight_prompt_carries_the_short_form():
    text = _read("micro_insight_extraction_system_prompt.md")
    assert "never guess the user's traits" in text
    assert "Never save that a tool doesn't work" in text


def test_the_reflection_prompts_drop_tool_claims_and_setup_failures():
    text = _read("reflection_system_prompt.md")
    assert "Never write that a tool or feature doesn't work or should be avoided" in text
    assert "the owner fixes setup" in text
    assert "Base every bullet on what the context shows" in text
    assert "The previous reflection is not fixed" in text
    merge = _read("reflection_merge_system_prompt.md")
    assert "they describe the peer's machine, not this one" in merge


def test_the_playbook_prompt_keeps_guesses_and_tool_claims_out():
    text = _read("playbook_distill_system_prompt.md")
    assert "never a guess dressed up as the recommended way" in text
    assert "Never put in it that a tool or feature doesn't work" in text


def test_the_playbook_prompt_still_renders(tmp_path):
    """It goes through str.format_map — a stray brace in the new rules would break it."""
    loader = InstructionLoader(base_dir=_BASE, personal_dir=tmp_path, use_micro=False, use_nano=False)
    out = loader.render(
        "playbook_distill_system_prompt.md", rating="bad", user_note="\nUser note: too slow")
    assert "rated **bad**" in out and "User note: too slow" in out
    assert "never a guess dressed up as the recommended way" in out
    assert '"task_type": "one of the types above"' in out


def test_a_known_insight_shows_its_key_only_when_it_has_one():
    assert _known_insight_line({"category": "preference", "entity_key": "preference:timezone",
                                "content": "The user works in CET."}) == (
        "- [preference] [preference:timezone] The user works in CET.")
    assert _known_insight_line({"category": "fact", "entity_key": None,
                                "content": "The warehouse closes at 15:00."}) == (
        "- [fact] The warehouse closes at 15:00.")


@pytest.fixture
async def mgr(tmp_path):
    m = InsightsManager(tmp_path / "insights.db")
    yield m
    await m.close()


async def test_the_same_key_replaces_a_wrong_insight(mgr):
    """What rule 8 of the extraction prompt promises the model."""
    old = await mgr.add(content="The user works in CET.", category="preference",
                        entity_key="preference:timezone", importance=6)
    new = await mgr.add(content="The user works in EST since moving to New York.",
                        category="preference", entity_key="preference:timezone", importance=6)
    assert new and new != old
    rows = await mgr.list_recent(limit=10)
    assert [r["content"] for r in rows] == ["The user works in EST since moving to New York."]
    assert rows[0]["supersedes_id"] == old
