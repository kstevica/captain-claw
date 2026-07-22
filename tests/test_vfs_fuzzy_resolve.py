"""Forgiving VFS project resolution — find the folder the user means.

A user (or a weak model) types "claude skills" for a mount named "CLAUDE-SKILLS".
Resolution folds case and separators so the folder is found instead of reported
missing, discovery lists linked mounts (not just real dirs), and — crucially —
the read-only check resolves the SAME canonical name so a fuzzy-matched write
can't slip past a mode:ro mount. Fuzzy matches are unique-only: an ambiguous
name never silently lands on one of several look-alikes.
"""

import json
from pathlib import Path

import pytest

from captain_claw import vfs
from captain_claw.tools.vfs import VfsTool


@pytest.fixture()
def vroot(tmp_path, monkeypatch):
    """A user root with a real dir, a plain rw link, and a read-only gdrive mount."""
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAW_VFS_USER", "local")
    monkeypatch.delenv("FD_OWNER_ID", raising=False)
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    monkeypatch.delenv("CLAW_VFS_PROJECT", raising=False)

    uroot = tmp_path / "local"
    (uroot / "commons").mkdir(parents=True)
    (uroot / "commons" / "note.md").write_text("hello")
    mount = uroot / ".drive" / "CLAUDE-SKILLS"
    mount.mkdir(parents=True)
    (mount / "readme.txt").write_text("skill docs")
    (uroot / ".vfs-links.json").write_text(json.dumps({
        "CLAUDE-SKILLS": {"path": str(mount), "mode": "ro", "kind": "gdrive"},
    }))
    return uroot


class TestResolveProjectName:
    def test_exact_wins(self, vroot):
        assert vfs.resolve_project_name("CLAUDE-SKILLS") == "CLAUDE-SKILLS"
        assert vfs.resolve_project_name("commons") == "commons"

    def test_case_and_separator_folding(self, vroot):
        for typed in ("claude skills", "claude-skills", "Claude_Skills",
                      "CLAUDESKILLS", "  claude   skills  "):
            assert vfs.resolve_project_name(typed) == "CLAUDE-SKILLS", typed

    def test_absent_returns_none(self, vroot):
        assert vfs.resolve_project_name("nonexistent") is None
        assert vfs.resolve_project_name("") is None

    def test_ambiguous_returns_none(self, tmp_path, monkeypatch):
        # Two projects that normalise to the same key must NOT auto-pick one.
        monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAW_VFS_USER", "local")
        monkeypatch.delenv("FD_OWNER_ID", raising=False)
        monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
        uroot = tmp_path / "local"
        (uroot / "Foo-Bar").mkdir(parents=True)
        (uroot / "foo_bar").mkdir(parents=True)
        assert vfs.resolve_project_name("foo bar") is None

    def test_scope_walls_fuzzy_match(self, vroot, monkeypatch):
        # A scoped process can't fuzzy-resolve outside its wall.
        monkeypatch.setenv("CLAW_VFS_SCOPE", "commons")
        assert vfs.resolve_project_name("claude skills") is None
        assert vfs.resolve_project_name("commons") == "commons"


class TestListProjects:
    def test_includes_links_and_hides_dotdirs(self, vroot):
        projects = vfs.list_projects()
        assert "CLAUDE-SKILLS" in projects  # the gdrive mount (a link key)
        assert "commons" in projects
        assert ".drive" not in projects  # the physical mount-holder is hidden


class TestResolveVfsPath:
    def test_fuzzy_project_resolves_a_real_file(self, vroot):
        p = vfs.resolve_vfs_path("vfs:claude skills/readme.txt")
        assert p is not None and p.exists()
        assert p.read_text() == "skill docs"

    def test_exact_still_resolves(self, vroot):
        p = vfs.resolve_vfs_path("vfs:CLAUDE-SKILLS/readme.txt")
        assert p is not None and p.exists()

    def test_unknown_project_creates_literal_not_fuzzy(self, vroot):
        # project_root under create must be literal — never fold a new name onto
        # an existing look-alike (or writes would land in the wrong project).
        base = vfs.project_root("claude skills", create=True)
        assert base.name == "claude-skills"  # sanitised literal, a NEW dir
        assert base != (vroot / ".drive" / "CLAUDE-SKILLS")


class TestReadonlyHoleClosed:
    """The safety property: a fuzzy name must be judged read-only the same way
    it resolves, or a write to 'claude skills' would slip into the RO mount."""

    def test_fuzzy_name_is_seen_readonly(self, vroot):
        assert vfs.project_is_readonly("claude skills") is True
        assert vfs.project_is_readonly("Claude_Skills") is True

    def test_rw_project_not_readonly(self, vroot):
        assert vfs.project_is_readonly("commons") is False

    async def test_write_tool_refuses_fuzzy_ro_mount(self, vroot):
        from captain_claw.tools.write import WriteTool

        r = await WriteTool().execute(path="vfs:claude skills/x.txt", content="nope")
        assert r.success is False and "read-only" in r.error.lower()


class TestVfsToolUX:
    async def test_ls_finds_fuzzy_named_project(self, vroot):
        r = await VfsTool().execute(action="ls", path="claude skills")
        assert r.success is True and "readme.txt" in r.content

    async def test_list_projects_action_shows_mount(self, vroot):
        r = await VfsTool().execute(action="list_projects")
        assert r.success is True and "CLAUDE-SKILLS" in r.content

    async def test_not_found_lists_known_projects(self, vroot):
        r = await VfsTool().execute(action="ls", path="totally-absent")
        assert r.success is False
        assert "CLAUDE-SKILLS" in r.error and "commons" in r.error
