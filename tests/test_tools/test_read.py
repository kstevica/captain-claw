from pathlib import Path

import pytest

from captain_claw.tools.read import ReadTool


@pytest.mark.asyncio
async def test_read_tool_limit_without_offset_returns_first_lines(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    tool = ReadTool()
    result = await tool.execute(path=str(target), limit=2)

    assert result.success is True
    assert "[lines 1-2]" in result.content
    assert "line1\nline2" in result.content
    assert "line3" not in result.content
