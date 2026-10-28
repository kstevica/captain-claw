"""Ollama gets the assistant's tool calls back with the tool results.

The converter used to send an assistant turn as role + content only, so the
model saw an empty turn followed by a result it never asked for. Gemma 4
(gemma4:31b-cloud) then called the same tool again on every turn until the
duplicate guard or the iteration budget stopped it; sent with its call, the
same exchange gets an answer.
"""

from __future__ import annotations

from captain_claw.llm import Message, _convert_messages_for_ollama


def _exchange(arguments) -> list[Message]:
    return [
        Message(role="system", content="You are an agent."),
        Message(role="user", content="How many .md files are there?"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "glob", "arguments": arguments},
        }]),
        Message(role="tool", content="notes/a.md\nREADME.md", tool_call_id="call_1", tool_name="glob"),
    ]


def test_the_call_goes_back_with_its_result():
    out = _convert_messages_for_ollama(_exchange('{"pattern": "**/*.md"}'))
    assert out[2] == {"role": "assistant", "content": "",
                      "tool_calls": [{"function": {"name": "glob", "arguments": {"pattern": "**/*.md"}}}]}
    assert out[3] == {"role": "tool", "content": "notes/a.md\nREADME.md", "tool_name": "glob"}


def test_arguments_already_an_object_or_unreadable():
    assert _convert_messages_for_ollama(_exchange({"pattern": "*.md"}))[2]["tool_calls"] == [
        {"function": {"name": "glob", "arguments": {"pattern": "*.md"}}}]
    assert _convert_messages_for_ollama(_exchange("{not json"))[2]["tool_calls"] == [
        {"function": {"name": "glob", "arguments": {}}}]


def test_the_agents_own_path_carries_them_too():
    # An agent's provider turns the legacy image markers off (its agent picks
    # the pictures): the other converter.
    out = _convert_messages_for_ollama(_exchange('{"pattern": "**/*.md"}'), legacy_markers=False)
    assert out[2]["tool_calls"] == [{"function": {"name": "glob", "arguments": {"pattern": "**/*.md"}}}]


def test_a_plain_reply_has_no_tool_calls():
    out = _convert_messages_for_ollama([Message(role="assistant", content="Two.")])
    assert out == [{"role": "assistant", "content": "Two."}]
