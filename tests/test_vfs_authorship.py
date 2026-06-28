"""Tests for VFS authorship tracking (who wrote each shared file)."""

from __future__ import annotations

import captain_claw.vfs as vfs


def _setup(monkeypatch, tmp_path, project="vatra-abc12345", label="Software Architect"):
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAW_VFS_USER", "u1")
    monkeypatch.setenv("CLAW_VFS_PROJECT", project)
    if label is not None:
        monkeypatch.setenv("CLAW_AGENT_LABEL", label)
    else:
        monkeypatch.delenv("CLAW_AGENT_LABEL", raising=False)
    monkeypatch.delenv("CLAW_VATRA_OWNER", raising=False)


def test_record_and_read_author(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    p = vfs.resolve_vfs_path("vfs:vatra-abc12345/dir/postgres.md", create_parents=True)
    p.write_text("hello")
    vfs.record_author(p)

    authors = vfs.read_authors(vfs.project_root("vatra-abc12345"))
    assert authors["dir/postgres.md"]["agent"] == "Software Architect"
    assert authors["dir/postgres.md"]["ts"] > 0


def test_latest_writer_wins(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, label="First Author")
    p = vfs.resolve_vfs_path("vfs:vatra-abc12345/notes.md", create_parents=True)
    p.write_text("v1")
    vfs.record_author(p)
    monkeypatch.setenv("CLAW_AGENT_LABEL", "Second Author")
    vfs.record_author(p)

    authors = vfs.read_authors(vfs.project_root("vatra-abc12345"))
    assert authors["notes.md"]["agent"] == "Second Author"


def test_vatra_owner_is_the_fallback_label(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, label=None)
    monkeypatch.setenv("CLAW_VATRA_OWNER", "data-engineer")
    assert vfs.agent_label() == "data-engineer"
    p = vfs.resolve_vfs_path("vfs:vatra-abc12345/x.md", create_parents=True)
    p.write_text("x")
    vfs.record_author(p)
    assert vfs.read_authors(vfs.project_root("vatra-abc12345"))["x.md"]["agent"] == "data-engineer"


def test_unknown_writer_records_nothing(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, label=None)  # no label, no owner → don't record noise
    assert vfs.agent_label() == ""
    p = vfs.resolve_vfs_path("vfs:vatra-abc12345/y.md", create_parents=True)
    p.write_text("y")
    vfs.record_author(p)
    assert vfs.read_authors(vfs.project_root("vatra-abc12345")) == {}


def test_read_authors_empty_when_no_sidecar(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    assert vfs.read_authors(vfs.project_root("vatra-abc12345")) == {}
