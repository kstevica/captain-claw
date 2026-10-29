"""Tool calls a model wrote as Qwen's XML and left a tag open in
(registry.repair_arguments): the parameters it sent inside another's value
are split out, a closing tag's debris comes off a value that must be one of
a few words, and a call with nothing wrong is passed on as it is. The calls
are the ones an onboarding exam on a Raspberry Pi had blocked."""

from __future__ import annotations

import pytest

from captain_claw import exam_turn
from captain_claw.tools.knowledge import KnowledgeTool
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult, repair_arguments

SCHEMA = KnowledgeTool.parameters


@pytest.mark.parametrize(("sent", "meant"), [
    ({"action": "search\n<parameter=query>\ndecomposing TRS multiple expansion P/E change", "limit": 8},
     {"action": "search", "query": "decomposing TRS multiple expansion P/E change", "limit": 8}),
    ({"action": "search\n<parameter=limit>\n5", "query": "condensed forecast steady state"},
     {"action": "search", "limit": 5, "query": "condensed forecast steady state"}),
    ({"action": "search>\n\n</invoke>\n", "query": '"bandwidth" market efficiency', "limit": 8},
     {"action": "search", "query": '"bandwidth" market efficiency', "limit": 8}),
    ({"action": "search\n<parameter=query>\nWACC weights</parameter>\n<parameter=doc>\nValuation</parameter>"},
     {"action": "search", "query": "WACC weights", "doc": "Valuation"}),
])
def test_parameters_left_inside_another_are_split_out(sent, meant):
    assert repair_arguments(sent, SCHEMA) == meant
    assert exam_turn.tool_block("knowledge", repair_arguments(sent, SCHEMA), None) is None


def test_what_it_did_send_is_kept_and_a_sound_call_is_left_alone():
    # A leaked parameter never overwrites the one it gave itself; an unknown one is dropped.
    got = repair_arguments({"action": "search\n<parameter=query>\nold\n<parameter=colour>\nred", "query": "new"}, SCHEMA)
    assert got == {"action": "search", "query": "new"}
    sound = {"action": "section", "doc": "Valuation", "section": "22 › SUMMARY", "_agent": object()}
    assert repair_arguments(sound, SCHEMA) is sound
    # Text that only looks like debris, in a value that isn't one of a few words, stays.
    text = {"action": "add", "text": "A note that ends with a tag </invoke>", "name": "note.md"}
    assert repair_arguments(text, SCHEMA) is text
    assert repair_arguments({"action": "nonsense>"}, SCHEMA) == {"action": "nonsense>"}   # nothing it could mean
    assert repair_arguments({"x": "a\n<parameter=y>\nb"}, {}) == {"x": "a\n<parameter=y>\nb"}  # no schema to go by


class Recorder(Tool):
    name = "knowledge"
    description = "Records what it is called with."
    parameters = SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append({k: v for k, v in kwargs.items() if not k.startswith("_")})
        return ToolResult(success=True, content="ok")


async def test_the_registry_runs_the_call_the_model_meant(monkeypatch):
    monkeypatch.setenv("CLAW_TOOL_AUDIT", "0")
    registry = ToolRegistry()
    tool = Recorder()
    registry.register(tool)
    result = await registry.execute("knowledge", {"action": "search\n<parameter=query>\nvalue traps", "limit": 6})
    assert result.success and tool.calls[-1] == {"action": "search", "query": "value traps", "limit": 6}
