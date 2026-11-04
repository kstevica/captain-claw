"""The per-turn loop guards let iterative work through and still catch loops.

Real cohort001 users (2026-11-04) got false "done" results from both guards:
the write-loop STOP fired after two small edits to a script (the agent then
said "PDF built" without running it), the duplicate-call guard refused the
re-run after a fix, and a second line range of the same file was a
"duplicate" (a 496-page report read only as far as its front matter).
"""

from __future__ import annotations

import os
import re
import types
from pathlib import Path

import pytest

from captain_claw import loop_guards as lg
from captain_claw.llm import LLMProvider, LLMResponse, ToolCall
from captain_claw.tools.registry import Tool, ToolRegistry, ToolResult

# ── Write-loop STOP: what counts ──────────────────────────────────────


def _tc(name: str, **args) -> types.SimpleNamespace:
    return types.SimpleNamespace(id=f"{name}-{len(args)}", name=name, arguments=args)


def test_edits_are_not_rewrites():
    assert lg.full_rewrite_path("edit", {"path": "build.py", "old_string": "a", "new_string": "b"}) == ""
    assert lg.full_rewrite_path("file_edit", {"path": "build.py"}) == ""
    assert lg.full_rewrite_path("write", {"path": "build.py", "content": "x"}) == "build.py"
    assert lg.full_rewrite_path("file_write", {"file_path": "build.py", "content": "x"}) == "build.py"


def test_an_append_is_not_a_rewrite():
    assert lg.full_rewrite_path("write", {"path": "notes.md", "content": "x", "append": True}) == ""
    assert lg.full_rewrite_path("write", '{"path": "notes.md", "content": "x", "append": "true"}') == ""


def test_two_edits_then_a_run_never_stop():
    counts: dict[str, int] = {}
    for iteration in (
        [_tc("write", path="build.py", content="v1")],
        [_tc("shell", command="python build.py")],
        [_tc("edit", path="build.py", old_string="a", new_string="b")],
        [_tc("edit", path="build.py", old_string="c", new_string="d")],
        [_tc("shell", command="python build.py")],
        [_tc("edit", path="build.py", old_string="e", new_string="f")],
    ):
        assert lg.write_loop_paths(iteration, counts, loop_at=2) == []
    assert counts == {"build.py": 1}


def test_the_same_whole_file_written_three_times_still_stops():
    counts: dict[str, int] = {}
    write = [_tc("write", path="report.md", content="# Report")]
    assert lg.write_loop_paths(write, counts, loop_at=2) == []
    # The second rewrite was refused by the blind-rewrite guard: loop_at 3.
    assert lg.write_loop_paths(write, counts, loop_at=3) == []
    assert lg.write_loop_paths(write, counts, loop_at=3) == ["report.md"]


def test_a_second_whole_rewrite_that_ran_stops_as_before():
    counts: dict[str, int] = {}
    write = [_tc("write", path="report.md", content="# Report")]
    assert lg.write_loop_paths(write, counts, loop_at=2) == []
    assert lg.write_loop_paths(write, counts, loop_at=2) == ["report.md"]


def test_an_iteration_with_other_work_does_not_stop():
    counts = {"report.md": 5}
    mixed = [_tc("write", path="report.md", content="x"), _tc("edit", path="report.md", old_string="a")]
    assert lg.write_loop_paths(mixed, counts, loop_at=2) == []


# ── Duplicate signature: the read range ───────────────────────────────


def test_the_read_range_is_part_of_the_signature():
    declared = {"path", "offset", "limit", "force"}
    assert lg.range_signature({"path": "a"}, declared) == ""
    assert lg.range_signature({"path": "a", "offset": 1}, declared) == ""  # line 1 = the start
    assert lg.range_signature({"path": "a", "offset": 0}, declared) == ""
    assert lg.range_signature({"path": "a", "offset": 201, "limit": 100}, declared) == "|offset=201|limit=100"
    assert lg.range_signature({"path": "a", "offset": "201.0"}, declared) == "|offset=201"
    assert lg.range_signature({"path": "a", "limit": 50}, declared) == "|limit=50"


def test_a_range_the_tool_ignores_does_not_dodge_the_guard():
    # pdf_extract has no page range: a made-up `pages` returns the same text.
    declared = {"path", "max_chars", "max_pages"}
    assert lg.range_signature({"path": "r.pdf", "pages": "50-100"}, declared) == ""
    assert lg.range_signature({"path": "r.pdf", "max_chars": 90000}, declared) == ""
    assert lg.range_signature({"path": "r.pdf", "pages": "50-100"}, declared | {"pages"}) == "|pages=50-100"


def test_a_command_names_the_files_it_runs(tmp_path):
    (tmp_path / "build.py").write_text("print(1)\n")
    (tmp_path / "data.csv").write_text("a,b\n")
    files = lg.touched_files(
        "shell", {"command": "cd x; python build.py --input=data.csv && echo done.txt"},
        tmp_path, path_tools={"read"},
    )
    assert files == [(tmp_path / "build.py").resolve(), (tmp_path / "data.csv").resolve()]
    assert lg.touched_files("shell", {"command": "make test"}, tmp_path, path_tools={"read"}) == []


def test_a_change_zeroes_runs_and_reads_of_that_file_but_not_the_edit():
    counts = {
        "shell|{\"command\": \"make\"}": 1,
        "read|path=/w/build.py": 1,
        "read|path=/w/saved/build.py|offset=40": 1,
        "read|path=/w/other.py": 1,
        "edit|{\"path\": \"build.py\", \"old_string\": \"a\"}": 1,
        "web_fetch|url=https://ex.com": 1,
    }
    lg.forget_repeats_after_change(counts, "build.py", path_tools={"read"})
    assert counts == {
        "shell|{\"command\": \"make\"}": 0,  # kept: the key says shell ran this turn
        "read|path=/w/build.py": 0,
        "read|path=/w/saved/build.py|offset=40": 0,
        "read|path=/w/other.py": 1,
        "edit|{\"path\": \"build.py\", \"old_string\": \"a\"}": 1,  # the same edit twice is a repeat
        "web_fetch|url=https://ex.com": 1,
    }


# ── In the tool loop ──────────────────────────────────────────────────


class _Provider(LLMProvider):
    async def complete(self, messages, tools=None, temperature=None, max_tokens=None):
        return LLMResponse(content="ok")

    async def complete_streaming(self, messages, tools=None, temperature=None, max_tokens=None):
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        return len(text)


class _NoSave:
    async def save_session(self, session) -> None:
        return None


class _Shell(Tool):
    """Records each command; `gen` rewrites out.txt, as a script's output would."""

    name = "shell"
    description = "run"
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}}

    def __init__(self, base) -> None:
        self.base = base
        self.ran: list[str] = []

    async def execute(self, command: str = "", **kwargs) -> ToolResult:
        self.ran.append(command)
        if command.startswith("gen"):
            (self.base / "out.txt").write_text(f"run {len(self.ran)}\n")
        return ToolResult(success=True, content=f"exit 0 ({len(self.ran)})")


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """The shipped duplicate limit (1), governor off, Jev review off."""
    from captain_claw.agent import Agent
    from captain_claw.config import get_config, set_config
    from captain_claw.session import Session
    from captain_claw.tools.edit import EditTool
    from captain_claw.tools.read import ReadTool
    from captain_claw.tools.write import WriteTool

    monkeypatch.setenv("CLAW_CTX_GOVERNOR", "off")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    old_cfg = get_config()
    cfg = old_cfg.model_copy(deep=True)
    cfg.tools.duplicate_call_max = 1
    cfg.tools.jev.review_blocks = False
    set_config(cfg)

    a = Agent(provider=_Provider())
    a._initialized = True
    a.session = Session(id="s1", name="loop-guards")
    a.session_manager = _NoSave()
    registry = ToolRegistry(base_path=tmp_path)
    for tool in (ReadTool(), EditTool(), WriteTool()):
        registry.register(tool)
    shell = _Shell(tmp_path)
    registry.register(shell)
    a.tools = registry
    a._turn_tool_call_counts = {}
    a._turn_tool_call_fps = {}
    a._blind_write_paths = set()
    a._malformed_retry_count = 0
    a.shell = shell
    yield a
    set_config(old_cfg)


def _blocked(results) -> list[str]:
    return [str(r.get("content", "")) for r in results
            if str(r.get("content", "")).startswith("DUPLICATE CALL BLOCKED")]


async def _call(agent, name, **args):
    return await agent._handle_tool_calls([ToolCall(id=f"{name}-{os.urandom(3).hex()}", name=name,
                                                    arguments=args)])


SCRIPT = "import sys\n" + "".join(f"step_{i} = {i}\n" for i in range(40))


async def test_edit_then_rerun_runs_again(agent, tmp_path):
    (tmp_path / "build_pdf.py").write_text(SCRIPT)
    await _call(agent, "shell", command="python build_pdf.py")
    r = await _call(agent, "edit", path=str(tmp_path / "build_pdf.py"),
                    old_string="step_3 = 3", new_string="step_3 = 33")
    assert r[0]["success"], r
    r = await _call(agent, "shell", command="python build_pdf.py")
    assert not _blocked(r)
    assert agent.shell.ran == ["python build_pdf.py"] * 2


async def test_a_fix_rerun_through_a_command_that_does_not_name_the_file(agent, tmp_path):
    (tmp_path / "ted2.py").write_text(SCRIPT)
    await _call(agent, "shell", command="make data")
    await _call(agent, "edit", path=str(tmp_path / "ted2.py"),
                old_string="step_5 = 5", new_string="step_5 = 55")
    r = await _call(agent, "shell", command="make data")
    assert not _blocked(r)
    assert agent.shell.ran == ["make data"] * 2


async def test_two_edits_then_a_run(agent, tmp_path):
    r = await _call(agent, "write", path="build_pdf.py", content=SCRIPT)
    assert r[0]["success"], r
    # The write tool files it under saved/; the agent goes on with that path.
    written = Path(re.search(r" to (\S+build_pdf\.py)", r[0]["content"]).group(1))
    run = f"python {written}"
    await _call(agent, "shell", command=run)
    for old, new in (("step_1 = 1", "step_1 = 11"), ("step_2 = 2", "step_2 = 22")):
        r = await _call(agent, "edit", path=str(written), old_string=old, new_string=new)
        assert r[0]["success"] and not _blocked(r), r
    r = await _call(agent, "shell", command=run)
    assert not _blocked(r)
    assert agent.shell.ran == [run, run]
    assert "step_2 = 22" in written.read_text()


async def test_a_file_changed_outside_the_tools_makes_the_rerun_new(agent, tmp_path):
    script = tmp_path / "fetch.py"
    script.write_text(SCRIPT)
    await _call(agent, "shell", command="python fetch.py")
    # Fixed by another command (`sed -i`): the file's state shows it.
    script.write_text(SCRIPT.replace("step_7 = 7", "step_7 = 77"))
    st = script.stat()
    os.utime(script, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
    r = await _call(agent, "shell", command="python fetch.py")
    assert not _blocked(r)
    assert len(agent.shell.ran) == 2


async def test_the_same_command_with_nothing_changed_is_still_blocked(agent, tmp_path):
    (tmp_path / "build_pdf.py").write_text(SCRIPT)
    await _call(agent, "shell", command="python build_pdf.py")
    r = await _call(agent, "shell", command="python build_pdf.py")
    assert _blocked(r)
    assert len(agent.shell.ran) == 1


async def test_a_command_that_rewrites_its_own_output_still_loops_out(agent, tmp_path):
    # Its own run changes out.txt; that alone must not make the repeat new.
    await _call(agent, "shell", command="gen > out.txt")
    r = await _call(agent, "shell", command="gen > out.txt")
    assert _blocked(r)
    assert len(agent.shell.ran) == 1


async def test_a_long_file_read_in_sections(agent, tmp_path):
    report = tmp_path / "report.txt"
    report.write_text("".join(f"page line {i}\n" for i in range(1, 1201)))
    for offset in (1, 301, 601, 901):
        r = await _call(agent, "read", path=str(report), offset=offset, limit=300)
        assert not _blocked(r), (offset, r)
        assert f"page line {offset}\n" in r[0]["content"] or f"page line {offset}" in r[0]["content"]
    # The same section again, unchanged, is a repeat.
    r = await _call(agent, "read", path=str(report), offset=301, limit=300)
    assert _blocked(r)


async def test_a_read_after_an_edit_of_that_file_is_new(agent, tmp_path):
    path = tmp_path / "build_pdf.py"
    path.write_text(SCRIPT)
    await _call(agent, "read", path=str(path))
    await _call(agent, "edit", path=str(path), old_string="step_9 = 9", new_string="step_9 = 99")
    r = await _call(agent, "read", path=str(path))
    assert not _blocked(r)
    assert "step_9 = 99" in r[0]["content"]


async def test_a_reread_of_regenerated_output_is_new(agent, tmp_path):
    await _call(agent, "shell", command="gen > out.txt")
    await _call(agent, "read", path=str(tmp_path / "out.txt"))
    out = tmp_path / "out.txt"
    out.write_text("run 2 — longer output\n")
    r = await _call(agent, "read", path=str(out))
    assert not _blocked(r)
    assert "longer output" in r[0]["content"]


async def test_the_same_read_with_nothing_changed_is_still_blocked(agent, tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n")
    await _call(agent, "read", path=str(path))
    r = await _call(agent, "read", path=str(path))
    assert _blocked(r)


async def test_the_same_edit_twice_is_still_a_repeat(agent, tmp_path):
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")
    args = {"path": str(path), "old_string": "x = 1", "new_string": "x = 2"}
    await _call(agent, "edit", **args)
    path.write_text("x = 1\n")  # reverted behind its back
    r = await _call(agent, "edit", **args)
    assert _blocked(r)
