"""Unit tests for the pure write-boundary guard logic."""

from pathlib import Path

from captain_claw import write_guard


def test_placeholder_re_matches_exact_compaction_ref():
    # This is the exact format built by agent_session_mixin._compact_write_tool_call
    # (997-1000). Closing this round-trip is the whole point of the guard.
    for path, lines, kb in [("vfs:p/x.md", 3, 0.1), ("saved/tmp/s/a.txt", 120, 4.2)]:
        ref = f"[written to disk: {path}, {lines} lines, {kb:.1f}KB — use read tool to view]"
        assert write_guard.is_placeholder_content(ref) is True


def test_placeholder_re_matches_shell_marker():
    assert write_guard.is_placeholder_content(
        "[... shell command truncated — 12 lines, 3.0KB total — files written to disk, use read tool to view]"
    ) is True


def test_placeholder_re_matches_own_result_string():
    assert write_guard.is_placeholder_content("Written 88 chars (1 lines) to vfs:p/x.md") is True


def test_placeholder_re_ignores_prose():
    assert write_guard.is_placeholder_content(
        "The vault was written to disk long ago; read the tool manual for details."
    ) is False
    assert write_guard.is_placeholder_content("# Chapter One\n\nThe door opened.") is False


def test_empty_content():
    assert write_guard.is_empty_content("") is True
    assert write_guard.is_empty_content("   \n\t ") is True
    assert write_guard.is_empty_content("x") is False


def test_requires_extension():
    assert write_guard.requires_extension("vfs:p/eppo-authenticity-pac") is True
    assert write_guard.requires_extension("vfs:p/part-one.md") is False
    assert write_guard.requires_extension("vfs:p/Makefile") is False
    assert write_guard.requires_extension("vfs:p/.env") is False


def test_repair_declared_name_unique_prefix():
    decl = ["eppo-authenticity-pack.md", "story-bible.md"]
    assert write_guard.repair_declared_name("eppo-authenticity-pac", decl) == "eppo-authenticity-pack.md"
    # exact match → nothing to repair
    assert write_guard.repair_declared_name("story-bible.md", decl) is None
    # no match
    assert write_guard.repair_declared_name("random", decl) is None


def test_repair_declared_name_ambiguous_returns_none():
    decl = ["part-one.md", "part-two.md"]
    assert write_guard.repair_declared_name("part-", decl) is None


def test_verify_readback_overwrite_ok(tmp_path: Path):
    f = tmp_path / "a.md"
    body = "hello world\n"
    f.write_text(body, encoding="utf-8")
    rb = write_guard.verify_readback(f, body, append=False)
    assert rb["ok"] is True
    assert rb["bytes"] == len(body.encode())
    assert rb["sha8"]


def test_verify_readback_detects_truncation(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("short", encoding="utf-8")
    rb = write_guard.verify_readback(f, "a much longer intended body", append=False)
    assert rb["ok"] is False
    assert "size_mismatch" in rb["reason"]


def test_verify_readback_append_uses_prev_size(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("AAA", encoding="utf-8")  # prev 3 bytes
    added = "BBBB"
    # simulate the append already having happened
    f.write_text("AAA" + added, encoding="utf-8")
    rb = write_guard.verify_readback(f, added, append=True, prev_size=3)
    assert rb["ok"] is True
    assert rb["bytes"] == 7


def test_guard_config_env_killswitch(monkeypatch):
    monkeypatch.setenv("CLAW_WRITE_GUARD", "0")
    gc = write_guard.guard_config()
    assert gc.reject_placeholder is False
    assert gc.verify_readback is False
    monkeypatch.setenv("CLAW_WRITE_GUARD", "1")
    gc = write_guard.guard_config()
    assert gc.reject_placeholder is True


def test_declared_files_and_floor(monkeypatch):
    monkeypatch.setenv("CLAW_DECLARED_FILES", '["a.md", "b.md"]')
    monkeypatch.setenv("CLAW_WRITE_MIN_BYTES", "2000")
    assert write_guard.declared_files() == ["a.md", "b.md"]
    assert write_guard.min_bytes_floor() == 2000
    monkeypatch.setenv("CLAW_DECLARED_FILES", "not json")
    assert write_guard.declared_files() == []
