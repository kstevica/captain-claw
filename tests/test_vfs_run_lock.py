"""A working run keeps its folder to itself.

Pinned here:

* while the deck drives a run, the owner's other agents read its folder and
  write nothing there: write, edit and the vfs tool's mkdir / mv / rm refuse;
* the run's own workers (bound to a run folder) are never held;
* the folder is free again when the run ends, and a marker left by a deck
  that went down is swept;
* a read-only linked folder refuses the vfs tool's changes too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from captain_claw import vfs
from captain_claw.tools.edit import EditTool
from captain_claw.tools.vfs import VfsTool
from captain_claw.tools.write import WriteTool

OWNER = "user-bob"
RUN = "bat-run00001"


@pytest.fixture
def root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.setenv("FD_OWNER_ID", OWNER)
    monkeypatch.delenv("CLAW_VFS_USER", raising=False)
    monkeypatch.delenv("CLAW_VFS_PROJECT", raising=False)
    r = vfs.user_root(create=True)
    (r / RUN).mkdir()
    (r / RUN / "list.csv").write_text("Name\nFirst\n", encoding="utf-8")
    (r / "notes").mkdir()
    return r


def test_a_working_run_holds_its_folder_and_only_that(root):
    assert not vfs.project_is_readonly(RUN)
    vfs.set_run_lock(root, RUN, True)
    assert vfs.project_is_readonly(RUN) and "a run is working" in vfs.readonly_reason(RUN)
    assert not vfs.project_is_readonly("notes") and vfs.readonly_reason("notes") == ""
    assert RUN in vfs.list_projects() and vfs.RUN_LOCKS_DIR not in vfs.list_projects()
    vfs.set_run_lock(root, RUN, False)
    assert not vfs.project_is_readonly(RUN)


def test_a_runs_own_worker_is_not_held(root, monkeypatch):
    vfs.set_run_lock(root, RUN, True)
    monkeypatch.setenv("CLAW_VFS_PROJECT", RUN)
    assert not vfs.project_is_readonly(RUN) and not vfs.project_is_readonly()


async def test_the_file_tools_refuse_while_it_works(root):
    vfs.set_run_lock(root, RUN, True)
    got = await WriteTool().execute(path=f"vfs:{RUN}/list.csv", content="Name\nOther\n")
    assert not got.success and "a run is working in this folder" in got.error
    got = await EditTool().execute(path=f"vfs:{RUN}/list.csv", action="replace", old="First", new="Other")
    assert not got.success and "a run is working in this folder" in got.error
    tool = VfsTool()
    for kw in ({"action": "rm", "path": f"{RUN}/list.csv"}, {"action": "rm", "path": RUN, "recursive": True},
               {"action": "mkdir", "path": f"vfs:{RUN}/mine"},
               {"action": "mv", "path": f"{RUN}/list.csv", "to": "notes/list.csv"},
               {"action": "mv", "path": "notes", "to": f"{RUN}/notes"}):
        got = await tool.execute(**kw)
        assert not got.success and "a run is working" in got.error, kw
    assert (root / RUN / "list.csv").read_text(encoding="utf-8") == "Name\nFirst\n"
    assert (await tool.execute(action="ls", path=RUN)).success  # reading stays open
    vfs.set_run_lock(root, RUN, False)
    assert (await tool.execute(action="mv", path=f"{RUN}/list.csv", to="notes/list.csv")).success


async def test_a_read_only_linked_folder_refuses_the_vfs_tool(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.txt").write_text("a", encoding="utf-8")
    (root / ".vfs-links.json").write_text(json.dumps({"repo": {"path": str(outside), "mode": "ro"}}))
    got = await VfsTool().execute(action="rm", path="repo/a.txt")
    assert not got.success and "read-only" in got.error
    assert (outside / "a.txt").exists()


def test_markers_of_runs_nobody_drives_are_swept(root):
    vfs.set_run_lock(root, RUN, True)
    vfs.set_run_lock(root, "bat-gone0001", True)
    assert vfs.sweep_run_locks(root.parent, {RUN}) == 1
    assert vfs.project_is_readonly(RUN) and not vfs.project_is_readonly("bat-gone0001")
