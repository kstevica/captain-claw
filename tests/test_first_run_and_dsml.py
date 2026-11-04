"""First-run walls and DeepSeek call markup (cohort001, 2026-11).

An inbox agent promised eight mail features with nothing connected, then its
first mail call came back "Blocked by tool policy chain", which read like an
admin setting. A developer agent met "there is no repository" and another told
its user to scp to the server. DeepSeek wrote its own tool-call markup into a
reply, and the user saw it raw."""

from __future__ import annotations

import types

import pytest

from captain_claw import dsml
from captain_claw.exceptions import ToolBlockedError
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

_MARKUP = (
    "One more check, because it's the thing that could still be true:\n\n"
    "<｜｜DSML｜｜ calls>\n"
    "<｜｜DSML｜｜ invoke name=\"read\">\n"
    "<｜｜DSML｜｜ parameter name=\"path\" string=\"true\">vfs:phase6a/dl-artifact.md</｜｜DSML｜｜ parameter>\n"
    "<｜｜DSML｜｜ parameter name=\"limit\" string=\"false\">40</｜｜DSML｜｜ parameter>\n"
    "</｜｜DSML｜｜ invoke>\n"
    "</｜｜DSML｜｜ calls>"
)


def test_dsml_calls_are_read_and_stripped():
    assert dsml.dsml_calls(_MARKUP) == [("read", {"path": "vfs:phase6a/dl-artifact.md", "limit": 40})]
    assert dsml.strip_dsml(_MARKUP) == "One more check, because it's the thing that could still be true:"
    single = '<｜DSML｜function_calls><｜DSML｜invoke name="web_search"><｜DSML｜parameter name="query" string="true">EOJN</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>'
    assert dsml.dsml_calls(single) == [("web_search", {"query": "EOJN"})]
    assert dsml.strip_dsml("Done.\n" + single) == "Done."


def test_cut_off_markup_is_stripped_too():
    cut = "Checking.\n<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"read\">\n<｜DSML｜parameter name=\"pa"
    assert dsml.dsml_calls(cut) == []
    assert dsml.strip_dsml(cut) == "Checking."


def test_plain_text_is_untouched():
    text = "A | table | row\n<b>bold</b> and ｜ a fullwidth bar"
    assert dsml.strip_dsml(text) == text and dsml.dsml_calls(text) == []


def test_the_agent_runs_dsml_calls():
    from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin

    calls = AgentToolLoopMixin._extract_tool_calls_from_content(
        object.__new__(AgentToolLoopMixin), _MARKUP)
    assert [(c.name, c.arguments) for c in calls] == [
        ("read", {"path": "vfs:phase6a/dl-artifact.md", "limit": 40})]


class _Mail(Tool):
    name = "google_mail"
    description = "mail"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return ToolResult(success=True, content="ok")


async def test_an_unconnected_account_says_so(monkeypatch, tmp_path):
    from captain_claw import google_oauth_manager

    monkeypatch.setattr(google_oauth_manager, "is_google_connected_cached", lambda *a, **k: False)
    reg = ToolRegistry(base_path=tmp_path)
    reg.register(_Mail(), metadata={"requires_google": True})
    with pytest.raises(ToolBlockedError) as caught:
        await reg.execute("google_mail", {})
    msg = str(caught.value)
    assert "Google isn't connected" in msg and "Connections → Google" in msg
    assert "policy chain" not in msg


def test_other_blocks_keep_their_reason(tmp_path):
    reg = ToolRegistry(base_path=tmp_path)
    reg.register(_Mail(), metadata={})
    assert reg._unconnected_reason("google_mail") == ""


def _prompt_agent(tools: set[str]):
    from captain_claw.agent_context_mixin import AgentContextMixin
    from captain_claw.instructions import InstructionLoader

    agent = object.__new__(AgentContextMixin)
    agent.tools = types.SimpleNamespace(has_tool=lambda n: n in tools)
    agent.instructions = InstructionLoader(use_micro=False, use_nano=False)
    return agent


def test_prompt_says_mail_is_not_connected(monkeypatch):
    from captain_claw import crew_client, google_oauth_manager, microsoft_oauth_manager

    monkeypatch.setattr(google_oauth_manager, "is_google_connected_cached", lambda *a, **k: False)
    monkeypatch.setattr(microsoft_oauth_manager, "is_microsoft_connected_cached", lambda *a, **k: False)
    monkeypatch.setattr(crew_client, "under_flight_deck", lambda: True)
    monkeypatch.setattr(crew_client, "crew_usable", lambda: False)
    block = _prompt_agent({"google_mail"})._build_connections_block()
    assert "Not connected yet" in block and "Connections" in block
    assert "Never suggest SSH, scp" in block and "crew tool" not in block
    monkeypatch.setattr(google_oauth_manager, "is_google_connected_cached", lambda *a, **k: True)
    block = _prompt_agent({"google_mail"})._build_connections_block()
    assert "Not connected yet" not in block


def test_prompt_names_crew_clone_when_it_works(monkeypatch):
    from captain_claw import crew_client

    monkeypatch.setattr(crew_client, "under_flight_deck", lambda: True)
    monkeypatch.setattr(crew_client, "crew_usable", lambda: True)
    block = _prompt_agent({"crew"})._build_connections_block()
    assert "crew tool (action=clone" in block


def test_no_flight_deck_no_files_note(monkeypatch):
    from captain_claw import crew_client

    monkeypatch.setattr(crew_client, "under_flight_deck", lambda: False)
    assert _prompt_agent(set())._build_connections_block() == ""
