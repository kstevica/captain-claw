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

    async def test_list_projects_surfaces_drive_source_path(self, tmp_path, monkeypatch):
        # The agent must SEE the mount's Drive path, so it can connect a request
        # like "performance reports" to a short-named mount like "VC".
        monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAW_VFS_USER", "local")
        monkeypatch.delenv("FD_OWNER_ID", raising=False)
        monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
        uroot = tmp_path / "local"
        mount = uroot / ".drive" / "VC"
        mount.mkdir(parents=True)
        (mount / ".drive-manifest.json").write_text('{"folder_id":"X","dirs":{},"files":{}}')
        (mount / "report.pdf").write_text("x")
        (uroot / ".vfs-links.json").write_text(json.dumps({
            "VC": {"path": str(mount), "mode": "ro", "kind": "gdrive",
                   "drive": {"folder_id": "X",
                             "source_path": "FRC3/Reporting/Performance/VC"}},
        }))
        r = await VfsTool().execute(action="list_projects")
        assert r.success is True
        assert "VC" in r.content
        assert "FRC3 / Reporting / Performance / VC" in r.content  # path visible
        assert "(1 file)" in r.content  # .drive-manifest.json excluded from count

    async def test_not_found_lists_known_projects(self, vroot):
        r = await VfsTool().execute(action="ls", path="totally-absent")
        assert r.success is False
        assert "CLAUDE-SKILLS" in r.error and "commons" in r.error

    async def test_info_surfaces_root_and_projects(self, vroot):
        # The diagnostic that makes a root mismatch obvious on staging.
        r = await VfsTool().execute(action="info")
        assert r.success is True
        assert "root:" in r.content and "resolved from:" in r.content
        assert "CLAUDE-SKILLS" in r.content and "CLAW_VFS_USER" in r.content


class TestPhysicalMountDiscovery:
    """A Drive mount is discoverable and addressable from its physical .drive/
    tree even when .vfs-links.json doesn't list it — the failure that made the
    agent report 'VFS completely empty' while the mount plainly existed."""

    @pytest.fixture()
    def linkless(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAW_VFS_USER", "local")
        monkeypatch.delenv("FD_OWNER_ID", raising=False)
        monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
        monkeypatch.delenv("CLAW_VFS_PROJECT", raising=False)
        uroot = tmp_path / "local"
        mount = uroot / ".drive" / "FRC2-Carry-Participation"
        mount.mkdir(parents=True)
        (mount / ".drive-manifest.json").write_text('{"folder_id":"X","dirs":{},"files":{}}')
        (mount / "signed.pdf").write_text("pdf")
        # deliberately NO .vfs-links.json — only the physical tree exists
        return uroot

    def test_listed_without_a_link(self, linkless):
        assert vfs.list_projects() == ["FRC2-Carry-Participation"]

    def test_exact_resolves_without_a_link(self, linkless):
        p = vfs.resolve_vfs_path("vfs:FRC2-Carry-Participation/signed.pdf")
        assert p is not None and p.exists()

    def test_fragment_resolves_to_the_mount(self, linkless):
        assert vfs.resolve_project_name("carry participation") == "FRC2-Carry-Participation"
        p = vfs.resolve_vfs_path("vfs:carry participation/signed.pdf")
        assert p is not None and p.exists()

    def test_mount_is_readonly_without_a_link(self, linkless):
        # A Drive mount is inherently read-only; a missing registry row must not
        # make it writable.
        assert vfs.project_is_readonly("carry participation") is True

    def test_display_hides_the_dot_drive_holder(self, linkless):
        disp = vfs.to_display(linkless / ".drive" / "FRC2-Carry-Participation" / "signed.pdf")
        assert disp == "vfs:FRC2-Carry-Participation/signed.pdf"

    async def test_vfs_tool_ls_finds_it(self, linkless):
        r = await VfsTool().execute(action="ls", path="carry participation")
        assert r.success is True and "signed.pdf" in r.content


class TestFragmentAmbiguity:
    def test_ambiguous_fragment_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAW_VFS_USER", "local")
        monkeypatch.delenv("FD_OWNER_ID", raising=False)
        monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
        uroot = tmp_path / "local"
        (uroot / "carry-participation-2023").mkdir(parents=True)
        (uroot / "carry-participation-2024").mkdir(parents=True)
        # a fragment of BOTH → ambiguous → refuse to guess
        assert vfs.resolve_project_name("carry participation") is None
