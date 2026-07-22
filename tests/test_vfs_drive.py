"""Drive mount — placeholder tree, manifest, lazy listing, refresh pruning.

Runs against a fake DriveClient (an in-memory Drive tree), so no Google, no
network, no tokens. Verifies the mount is a real filesystem tree of honest
placeholders and that a refresh reflects upstream adds/removes.
"""

import tempfile
from pathlib import Path

import pytest

from captain_claw import vfs_drive
from captain_claw.drive_client import DriveFile, FOLDER_MIME


class FakeDrive:
    """An in-memory Drive: folder_id -> list of child DriveFiles."""

    def __init__(self, tree: dict[str, list[DriveFile]]):
        self.tree = tree
        self.list_calls: list[str] = []

    async def list_folder(self, folder_id, *, order_by="folder,name", max_files=None, sleep=None):
        self.list_calls.append(folder_id)
        children = self.tree.get(folder_id, [])
        if max_files is not None and len(children) > max_files:
            return children[:max_files], True
        return list(children), False


def _folder(fid, name):
    return DriveFile(id=fid, name=name, mime_type=FOLDER_MIME)


def _file(fid, name, *, size=100, modified="2026-07-20T10:00:00Z", mime="text/plain"):
    return DriveFile(id=fid, name=name, mime_type=mime, size=size, modified_time=modified)


@pytest.fixture()
def mount(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(vfs_drive, "user_root", lambda uid: Path(tmp) / uid)
        yield Path(tmp)


def _sample_drive():
    # root/ ├ report.pdf  ├ notes.txt  └ sub/ └ deep.md
    return FakeDrive({
        "ROOT": [_folder("F_SUB", "sub"), _file("f1", "report.pdf", mime="application/pdf"),
                 _file("f2", "notes.txt")],
        "F_SUB": [_file("f3", "deep.md", mime="text/markdown")],
    })


class TestCreateMount:
    async def test_builds_the_real_tree(self, mount):
        drive = _sample_drive()
        summary = await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        assert (root / "report.pdf").is_file()
        assert (root / "notes.txt").is_file()
        assert (root / "sub").is_dir()
        assert (root / "sub" / "deep.md").is_file()
        assert summary["files"] == 3 and summary["dirs"] == 1
        assert summary["truncated"] is False

    async def test_placeholder_is_an_honest_marker_not_empty(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        text = (vfs_drive.mount_root("alice", "acme") / "report.pdf").read_text()
        assert "Google Drive" in text
        assert "report.pdf" in text
        assert "id f1" in text
        assert "not been downloaded" in text.lower()

    async def test_manifest_records_every_file_as_placeholder(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        man = vfs_drive.Manifest.load(vfs_drive.mount_root("alice", "acme"))
        assert set(man.files) == {"report.pdf", "notes.txt", "sub/deep.md"}
        assert all(e["state"] == "placeholder" for e in man.files.values())
        assert man.dirs[""] == "ROOT" and man.dirs["sub"] == "F_SUB"
        assert man.files["report.pdf"]["file_id"] == "f1"

    async def test_is_placeholder_reads_the_manifest(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        assert vfs_drive.is_placeholder(root, "report.pdf") is True
        assert vfs_drive.is_placeholder(root, "does/not/exist") is False


class TestCap:
    async def test_cap_truncates_and_reports(self, mount):
        many = FakeDrive({"ROOT": [_file(f"f{i}", f"{i}.txt") for i in range(10)]})
        summary = await vfs_drive.create_mount(many, "alice", "big", "ROOT", max_files=4)
        assert summary["files"] == 4
        assert summary["truncated"] is True


class TestRefresh:
    async def test_new_upstream_file_appears(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        drive.tree["ROOT"].append(_file("f9", "added.txt"))
        root = vfs_drive.mount_root("alice", "acme")
        await vfs_drive.sync(drive, root)
        assert (root / "added.txt").is_file()
        assert "added.txt" in vfs_drive.Manifest.load(root).files

    async def test_vanished_upstream_file_is_pruned_locally(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        assert (root / "notes.txt").is_file()
        drive.tree["ROOT"] = [f for f in drive.tree["ROOT"] if f.name != "notes.txt"]
        await vfs_drive.sync(drive, root)
        assert not (root / "notes.txt").exists()  # no ghost left behind
        assert "notes.txt" not in vfs_drive.Manifest.load(root).files

    async def test_cloned_state_survives_refresh_when_unchanged(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        man = vfs_drive.Manifest.load(root)
        man.files["report.pdf"]["state"] = "cloned"
        man.save(root)
        await vfs_drive.sync(drive, root)  # same modifiedTime upstream
        assert vfs_drive.Manifest.load(root).files["report.pdf"]["state"] == "cloned"

    async def test_changed_file_reverts_to_placeholder(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        man = vfs_drive.Manifest.load(root)
        man.files["report.pdf"]["state"] = "cloned"
        man.save(root)
        # upstream edit bumps modifiedTime
        for f in drive.tree["ROOT"]:
            if f.name == "report.pdf":
                f.modified_time = "2026-07-25T12:00:00Z"
        await vfs_drive.sync(drive, root)
        assert vfs_drive.Manifest.load(root).files["report.pdf"]["state"] == "placeholder"


class TestLazyListDir:
    async def test_list_dir_requires_a_known_directory(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        with pytest.raises(ValueError):
            await vfs_drive.list_dir(drive, root, "unknown-subdir")

    async def test_list_dir_of_known_subdir(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        out = await vfs_drive.list_dir(drive, root, "sub")
        assert out["files"] == 1  # the listing reports sub/'s one child
        assert (root / "sub" / "deep.md").is_file()


class TestLinkEntry:
    def test_link_entry_is_a_readonly_gdrive_link(self):
        ent = vfs_drive.link_entry("alice", "acme", "ROOT", clonemd=True)
        assert ent["mode"] == "ro"
        assert ent["kind"] == "gdrive"
        assert ent["drive"]["folder_id"] == "ROOT"
        assert ent["drive"]["clonemd"] is True
        assert Path(ent["path"]).name == "acme"

    def test_is_drive_link(self):
        assert vfs_drive.is_drive_link({"kind": "gdrive"}) is True
        assert vfs_drive.is_drive_link({"mode": "rw"}) is False
        assert vfs_drive.is_drive_link(None) is False


class TestRemoveMount:
    async def test_keep_cloned_preserves_cloned_files(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        man = vfs_drive.Manifest.load(root)
        man.files["report.pdf"]["state"] = "cloned"
        man.save(root)
        result = vfs_drive.remove_mount("alice", "acme", keep_cloned=True)
        assert result["kept"] == 1
        assert (root / "report.pdf").is_file()   # cloned file survives
        assert not (root / "notes.txt").exists()  # placeholder removed

    async def test_full_removal(self, mount):
        drive = _sample_drive()
        await vfs_drive.create_mount(drive, "alice", "acme", "ROOT")
        root = vfs_drive.mount_root("alice", "acme")
        vfs_drive.remove_mount("alice", "acme", keep_cloned=False)
        assert not root.exists()
