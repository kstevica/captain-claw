"""Cross-project VFS glob — search every project without knowing the mount name.

The agent probed `vfs:**/*` / `vfs:**/*leyr*`, got nothing (because "**" parsed
as a project name and collapsed to the empty default project), concluded the VFS
was empty, and fell back to the raw google_drive tool — while the data sat in
`vfs:VC/Leyr/`. A glob in the project position now means "search all projects".
"""

import json
from pathlib import Path

import pytest

from captain_claw.tools.glob import GlobTool


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A user root with a plain project and a Google Drive mount holding a
    Leyr/ subfolder — mirrors the staging layout."""
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAW_VFS_USER", "local")
    monkeypatch.delenv("FD_OWNER_ID", raising=False)
    monkeypatch.delenv("CLAW_VFS_SCOPE", raising=False)
    monkeypatch.delenv("CLAW_VFS_PROJECT", raising=False)
    uroot = tmp_path / "local"
    (uroot / "commons").mkdir(parents=True)
    (uroot / "commons" / "note.md").write_text("hi")
    vc = uroot / ".drive" / "VC"
    (vc / "Leyr").mkdir(parents=True)
    (vc / "Compass").mkdir(parents=True)
    (vc / ".drive-manifest.json").write_text('{"folder_id":"X","dirs":{},"files":{}}')
    (vc / ".drive-cache").mkdir()
    (vc / ".drive-cache" / "blob").write_text("cached")
    (vc / "Leyr" / "Q2 revaluation.pdf").write_text("leyr numbers")
    (vc / "Compass" / "deck.pdf").write_text("compass")
    (uroot / ".vfs-links.json").write_text(json.dumps({
        "VC": {"path": str(vc), "mode": "ro", "kind": "gdrive",
               "drive": {"folder_id": "X", "source_path": "FRC3/Reporting/Performance/VC"}},
    }))
    return uroot


class TestCrossProjectGlob:
    async def test_star_star_lists_every_project(self, tree):
        r = await GlobTool().execute(pattern="vfs:**/*")
        assert r.success is True
        assert "vfs:commons/note.md" in r.content
        assert "vfs:VC/Leyr/Q2 revaluation.pdf" in r.content
        assert "vfs:VC/Compass/deck.pdf" in r.content

    async def test_fragment_finds_files_inside_a_named_folder(self, tree):
        # The reported case: "leyr" matches the Leyr/ folder's contents even
        # though the file isn't literally named *leyr*.
        r = await GlobTool().execute(pattern="vfs:**/*leyr*")
        assert r.success is True
        assert "vfs:VC/Leyr/Q2 revaluation.pdf" in r.content
        assert "deck.pdf" not in r.content        # Compass isn't a leyr match
        assert "note.md" not in r.content

    async def test_mount_internals_are_excluded(self, tree):
        r = await GlobTool().execute(pattern="vfs:**/*")
        assert ".drive-manifest.json" not in r.content
        assert ".drive-cache" not in r.content

    async def test_project_position_fragment_without_slash(self, tree):
        # "vfs:*leyr*" (glob in the project slot, no subpath) also searches all.
        r = await GlobTool().execute(pattern="vfs:*leyr*")
        assert r.success is True and "vfs:VC/Leyr/Q2 revaluation.pdf" in r.content

    async def test_named_project_still_scoped(self, tree):
        # A concrete project name still searches only that project.
        r = await GlobTool().execute(pattern="vfs:VC/**/*")
        assert "vfs:VC/Leyr/Q2 revaluation.pdf" in r.content
        assert "commons/note.md" not in r.content
        assert ".drive-manifest.json" not in r.content  # internals excluded here too

    async def test_no_match_reports_cleanly(self, tree):
        r = await GlobTool().execute(pattern="vfs:**/*zzzznope*")
        assert r.success is True and "No files found" in r.content
