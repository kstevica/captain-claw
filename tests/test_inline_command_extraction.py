"""Which reply text the agent loop runs as an inline shell command.

A code block in a reply is run only when it is plainly a command.  A block
that shows a file's content (the fault: a totals file shown after a `write`)
must never reach the shell.
"""

from __future__ import annotations

import inspect

from captain_claw.agent_orchestration_mixin import AgentOrchestrationMixin
from captain_claw.agent_tool_loop_mixin import AgentToolLoopMixin


class _Loop(AgentToolLoopMixin):
    pass


def _extract(text: str, **kw):
    return _Loop()._extract_command_from_response(text, **kw)


FILE_SHOWN = (
    "Done. I wrote totals.md:\n\n"
    "```\n"
    "a.txt: 10\n"
    "b.txt: 20\n"
    "c.txt: 30\n"
    "grand total: 60\n"
    "```\n"
)


def test_file_content_in_plain_block_is_not_a_command():
    assert _extract(FILE_SHOWN) is None


def test_table_in_plain_block_is_not_a_command():
    text = "Here it is:\n\n```\n| file | total |\n| a.txt | 10 |\n```\n"
    assert _extract(text) is None


def test_key_value_and_prose_blocks_are_not_commands():
    assert _extract("```\nname: demo\nversion: 1 2 3\n```\n") is None
    assert _extract("```\nThe quick result is 42 units\n```\n") is None


def test_plain_block_with_a_real_command_still_runs():
    assert _extract("```\necho hello world\n```\n") == "echo hello world"
    assert _extract("```\ncd /tmp && ls\n```\n") == "cd /tmp && ls"


def test_plain_block_env_prefix_and_prompt_marker():
    assert _extract("```\nFOO=1 echo hi there\n```\n") == "FOO=1 echo hi there"
    assert _extract("```\n$ echo hi there\n```\n") == "$ echo hi there"
    assert _extract("```\n./run.sh now\n```\n") == "./run.sh now"


def test_bash_block_still_runs():
    assert _extract("```bash\necho hello world\n```") == "echo hello world"


def test_ill_run_phrase_still_runs():
    assert _extract("I'll run: `echo hello world`\n") == "echo hello world"


def test_no_code_block_runs_after_real_tool_calls():
    # The model already used tools this turn: blocks only show things.
    assert _extract(FILE_SHOWN, allow_fenced=False) is None
    assert _extract("```\necho hello world\n```\n", allow_fenced=False) is None
    assert _extract("Saved run.sh:\n```bash\necho hello world\n```", allow_fenced=False) is None
    # The explicit phrase is still honoured.
    assert _extract("I'll run: `echo hello world`\n", allow_fenced=False) == "echo hello world"


def test_loop_turns_blocks_off_once_tools_ran():
    src = inspect.getsource(AgentOrchestrationMixin)
    assert "allow_fenced=_tools_executed_count == 0" in src


def test_starts_with_command_word():
    ok = _Loop._starts_with_command_word
    assert ok("echo hi")
    assert ok("export A=1")
    assert ok("/usr/bin/env python x.py")
    assert not ok("a.txt: 10")
    assert not ok("grand total: 60")
    assert not ok("no-such-program-xyz123 --flag")
    assert not ok("")
