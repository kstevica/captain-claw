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


# ── Increment 1: write-boundary guards ───────────────────────────────────

import re as _re

from captain_claw import write_guard
from captain_claw.agent_file_ops_mixin import AgentFileOpsMixin


@pytest.mark.asyncio
async def test_write_refuses_placeholder_compaction_marker(tmp_path: Path):
    tool = WriteTool()
    marker = "[written to disk: vfs:p/x.md, 3 lines, 0.1KB — use read tool to view]"
    result = await tool.execute(
        path="report.md", content=marker, _saved_base_path=tmp_path / "saved",
    )
    assert result.success is False
    assert result.error == "placeholder_content_rejected"
    assert not list((tmp_path / "saved").rglob("report.md"))


@pytest.mark.asyncio
async def test_write_refuses_shell_truncation_marker(tmp_path: Path):
    tool = WriteTool()
    result = await tool.execute(
        path="a.txt",
        content="[... shell command truncated — 5 lines, 2.0KB total — files written to disk",
        _saved_base_path=tmp_path / "saved",
    )
    assert result.success is False
    assert result.error == "placeholder_content_rejected"


@pytest.mark.asyncio
async def test_write_refuses_placeholder_on_append(tmp_path: Path):
    tool = WriteTool()
    target = tmp_path / "saved" / "tmp" / "default" / "log.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("real content\n", encoding="utf-8")
    result = await tool.execute(
        path="log.txt", content="[written to disk: log.txt, 1 lines, 0.0KB — use read tool to view]",
        append=True, _saved_base_path=tmp_path / "saved",
    )
    assert result.success is False
    assert result.error == "placeholder_content_rejected"
    # existing content untouched
    assert target.read_text(encoding="utf-8") == "real content\n"


@pytest.mark.asyncio
async def test_write_refuses_empty_content(tmp_path: Path):
    tool = WriteTool()
    result = await tool.execute(
        path="empty.md", content="   \n  ", _saved_base_path=tmp_path / "saved",
    )
    assert result.success is False
    assert result.error == "empty_content_rejected"


@pytest.mark.asyncio
async def test_write_allows_empty_init_py(tmp_path: Path):
    tool = WriteTool()
    result = await tool.execute(
        path="__init__.py", content="", _saved_base_path=tmp_path / "saved",
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_write_accepts_prose_mentioning_written_to_disk(tmp_path: Path):
    tool = WriteTool()
    body = "The archive was written to disk in 1987 and later recovered.\n" * 5
    result = await tool.execute(
        path="story.md", content=body, _saved_base_path=tmp_path / "saved",
    )
    assert result.success is True
    written = list((tmp_path / "saved").rglob("story.md"))
    assert written and written[0].read_text(encoding="utf-8") == body


@pytest.mark.asyncio
async def test_write_result_msg_still_parses_and_receipt_in_hint(tmp_path: Path):
    tool = WriteTool()
    result = await tool.execute(
        path="doc.md", content="# Title\n\nbody body body\n",
        _saved_base_path=tmp_path / "saved",
    )
    assert result.success is True
    # result content is still "Written N chars (M lines) to <path>[ (requested: …)]"
    parsed = AgentFileOpsMixin._parse_written_path_from_tool_output(result.content)
    assert parsed is not None and parsed.name == "doc.md"
    assert "verified" not in result.content and "sha" not in result.content
    # the verify receipt + compaction education live in the hint
    assert result.system_hint and "Saved and verified" in result.system_hint
    assert "sha" in result.system_hint
    assert "compaction marker" in result.system_hint


@pytest.mark.asyncio
async def test_write_verify_failed_when_readback_mismatches(tmp_path: Path, monkeypatch):
    tool = WriteTool()
    calls = {"n": 0}
    real = write_guard.verify_readback

    def flaky(*a, **k):
        calls["n"] += 1
        return {"ok": False, "bytes": 0, "sha8": "", "reason": "size_mismatch: forced"}

    monkeypatch.setattr(write_guard, "verify_readback", flaky)
    result = await tool.execute(
        path="doc.md", content="hello world\n" * 40, _saved_base_path=tmp_path / "saved",
    )
    assert result.success is False
    assert result.error == "write_verify_failed"
    # initial attempt + 2 retries = 3 verify calls
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_write_verify_recovers_after_retry(tmp_path: Path, monkeypatch):
    tool = WriteTool()
    seq = [
        {"ok": False, "bytes": 0, "sha8": "", "reason": "forced"},
        {"ok": True, "bytes": 12, "sha8": "abcd1234", "reason": ""},
    ]

    def scripted(*a, **k):
        return seq.pop(0)

    monkeypatch.setattr(write_guard, "verify_readback", scripted)
    result = await tool.execute(
        path="doc.md", content="hello world\n", _saved_base_path=tmp_path / "saved",
    )
    assert result.success is True
    assert "Saved and verified (12 bytes, sha abcd1234)" in result.system_hint


@pytest.mark.asyncio
async def test_write_guard_env_killswitch_restores_today(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_WRITE_GUARD", "0")
    tool = WriteTool()
    marker = "[written to disk: vfs:p/x.md, 3 lines, 0.1KB — use read tool to view]"
    result = await tool.execute(
        path="report.md", content=marker, _saved_base_path=tmp_path / "saved",
    )
    # guard off → placeholder accepted, file written verbatim (today's behaviour)
    assert result.success is True
    written = list((tmp_path / "saved").rglob("report.md"))
    assert written and written[0].read_text(encoding="utf-8") == marker


@pytest.mark.asyncio
async def test_write_vfs_strict_requires_extension(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_WRITE_STRICT", "1")
    monkeypatch.setattr("captain_claw.tools.write.is_vfs_path", lambda p: str(p).startswith("vfs:"))
    monkeypatch.setattr("captain_claw.tools.write.project_is_readonly", lambda p: False)
    monkeypatch.setattr("captain_claw.tools.write.split_scheme", lambda p: ("proj", str(p).split("/", 1)[-1]))
    monkeypatch.setattr("captain_claw.tools.write.resolve_vfs_path",
                        lambda p, create_parents=False: tmp_path / Path(p.split("/")[-1]).name)
    tool = WriteTool()
    result = await tool.execute(path="vfs:proj/eppo-authenticity-pac", content="x" * 400)
    assert result.success is False
    assert result.error == "path_missing_extension"


@pytest.mark.asyncio
async def test_write_vfs_strict_repairs_declared_name(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_WRITE_STRICT", "1")
    monkeypatch.setenv("CLAW_DECLARED_FILES", '["eppo-authenticity-pack.md"]')
    monkeypatch.setattr("captain_claw.tools.write.is_vfs_path", lambda p: str(p).startswith("vfs:"))
    monkeypatch.setattr("captain_claw.tools.write.project_is_readonly", lambda p: False)
    monkeypatch.setattr("captain_claw.tools.write.split_scheme", lambda p: ("proj", str(p).split("/", 1)[-1]))
    monkeypatch.setattr("captain_claw.tools.write.resolve_vfs_path",
                        lambda p, create_parents=False: tmp_path / Path(p.split("/")[-1]).name)
    tool = WriteTool()
    result = await tool.execute(path="vfs:proj/eppo-authenticity-pac", content="x" * 400)
    assert result.success is True
    assert (tmp_path / "eppo-authenticity-pack.md").exists()
    assert "repaired from eppo-authenticity-pac" in (result.system_hint or "")


@pytest.mark.asyncio
async def test_write_vfs_strict_content_below_floor(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLAW_WRITE_STRICT", "1")
    monkeypatch.setenv("CLAW_DECLARED_FILES", '["part-one.md"]')
    monkeypatch.setenv("CLAW_WRITE_MIN_BYTES", "2000")
    monkeypatch.setattr("captain_claw.tools.write.is_vfs_path", lambda p: str(p).startswith("vfs:"))
    monkeypatch.setattr("captain_claw.tools.write.project_is_readonly", lambda p: False)
    monkeypatch.setattr("captain_claw.tools.write.split_scheme", lambda p: ("proj", str(p).split("/", 1)[-1]))
    monkeypatch.setattr("captain_claw.tools.write.resolve_vfs_path",
                        lambda p, create_parents=False: tmp_path / Path(p.split("/")[-1]).name)
    tool = WriteTool()
    result = await tool.execute(path="vfs:proj/part-one.md", content="short summary")
    assert result.success is False
    assert result.error == "content_below_floor"


@pytest.mark.asyncio
async def test_write_strict_off_by_default_allows_extensionless(tmp_path: Path, monkeypatch):
    # No CLAW_WRITE_STRICT → Tier 2 inert even for vfs paths
    monkeypatch.setattr("captain_claw.tools.write.is_vfs_path", lambda p: str(p).startswith("vfs:"))
    monkeypatch.setattr("captain_claw.tools.write.project_is_readonly", lambda p: False)
    monkeypatch.setattr("captain_claw.tools.write.split_scheme", lambda p: ("proj", str(p).split("/", 1)[-1]))
    monkeypatch.setattr("captain_claw.tools.write.resolve_vfs_path",
                        lambda p, create_parents=False: tmp_path / Path(p.split("/")[-1]).name)
    tool = WriteTool()
    result = await tool.execute(path="vfs:proj/notes", content="x" * 400)
    assert result.success is True
