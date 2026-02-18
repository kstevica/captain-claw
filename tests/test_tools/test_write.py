from pathlib import Path

import pytest

from captain_claw.tools.registry import ToolRegistry
from captain_claw.tools.write import WriteTool


@pytest.mark.asyncio
async def test_write_tool_uses_runtime_saved_root_for_relative_paths(tmp_path: Path):
    tool = WriteTool()

    result = await tool.execute(
        path="scripts/example.sh",
        content="echo hi\n",
        _runtime_base_path=tmp_path,
    )

    expected = tmp_path / "saved" / "scripts" / "default" / "example.sh"
    assert result.success is True
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == "echo hi\n"


@pytest.mark.asyncio
async def test_write_tool_redirects_absolute_paths_under_saved_root(tmp_path: Path):
    tool = WriteTool()
    outside = tmp_path / "outside.txt"

    result = await tool.execute(
        path=str(outside),
        content="hello",
        _saved_base_path=tmp_path / "saved",
    )

    redirected = list((tmp_path / "saved").rglob("outside.txt"))
    assert result.success is True
    assert not outside.exists()
    assert len(redirected) == 1
    assert redirected[0].read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_write_tool_blocks_parent_traversal_outside_saved_root(tmp_path: Path):
    tool = WriteTool()

    result = await tool.execute(
        path="../escape.txt",
        content="safe",
        _saved_base_path=tmp_path / "saved",
    )

    expected = tmp_path / "saved" / "tmp" / "default" / "escape.txt"
    assert result.success is True
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == "safe"


@pytest.mark.asyncio
async def test_registry_injects_saved_root_for_write_tool(tmp_path: Path):
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(WriteTool())

    result = await registry.execute(
        name="write",
        arguments={"path": "report.txt", "content": "ready"},
    )

    expected = tmp_path / "saved" / "tmp" / "default" / "report.txt"
    assert result.success is True
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == "ready"


@pytest.mark.asyncio
async def test_registry_session_id_routes_write_into_session_folder(tmp_path: Path):
    registry = ToolRegistry(base_path=tmp_path)
    registry.register(WriteTool())

    result = await registry.execute(
        name="write",
        arguments={"path": "downloads/file.txt", "content": "ok"},
        session_id="session-42",
    )

    expected = tmp_path / "saved" / "downloads" / "session-42" / "file.txt"
    assert result.success is True
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
async def test_write_tool_accepts_saved_prefix_without_tmp_nesting(tmp_path: Path):
    tool = WriteTool()

    result = await tool.execute(
        path="saved/showcase/session-42/Zagreb-details.md",
        content="# Zagreb\n",
        _saved_base_path=tmp_path / "saved",
        _session_id="session-42",
    )

    expected = tmp_path / "saved" / "showcase" / "session-42" / "Zagreb-details.md"
    assert result.success is True
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == "# Zagreb\n"
