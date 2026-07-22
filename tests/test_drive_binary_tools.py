"""Binary-reading tools treat a Google Drive placeholder like a local file.

A Drive mount is a tree of placeholder *markers*; a parser that opened one off
disk used to choke — "Invalid DOCX: file is not a valid ZIP package" — because
it read the marker text, not the document. Every binary reader now routes path
resolution through ``document_extract._resolve_readable_file`` (async) or, for
cv's worker-thread ops, ``vfs_drive.materialize_sync``. Both fetch the real
bytes on demand.

These tests reproduce the exact failure (docx_extract on a placeholder) and pin
the shared seam that image_ocr / image_vision / video_vision / cv all sit on.
Everything runs against an in-memory fake Drive — no Google, no network.
"""

import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest

from captain_claw import vfs_drive
from captain_claw.drive_client import DriveError, DriveFile, FOLDER_MIME

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _docx_bytes(text: str) -> bytes:
    """A minimal but real .docx (a zip holding word/document.xml) the extractor
    can parse — proof we handed it the document's bytes, not the marker."""
    doc = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", doc)
    return buf.getvalue()


def _file(fid, name, *, mime, size=100, modified="2026-07-20T10:00:00Z"):
    return DriveFile(id=fid, name=name, mime_type=mime, size=size, modified_time=modified)


class FakeDrive:
    def __init__(self, tree, content):
        self.tree = tree
        self.content = content
        self.fetch_calls: list[str] = []

    async def list_folder(self, fid, *, drive_id="", order_by="folder,name",
                          max_files=None, sleep=None):
        return list(self.tree.get(fid, [])), False

    async def fetch(self, f, *, sleep=None):
        self.fetch_calls.append(f.id)
        return self.content[f.id]

    async def close(self):
        pass


async def _mount(tmp_path, monkeypatch, tree, content, *, drive_cls=FakeDrive,
                 with_vfs_env=False):
    """Build a Drive mount on a temp root; wire make_client → the fake.

    Returns ``(root, drive)``. With ``with_vfs_env`` the vfs: resolver is wired
    too (env + a .vfs-links.json entry), so ``vfs:acme/…`` paths resolve.
    """
    monkeypatch.setattr(vfs_drive, "user_root", lambda uid: tmp_path / uid)
    drive = drive_cls(tree, content)
    await vfs_drive.create_mount(drive, "local", "acme", "ROOT")
    monkeypatch.setattr("captain_claw.drive_client.make_client", lambda: drive)
    root = vfs_drive.mount_root("local", "acme")
    if with_vfs_env:
        monkeypatch.setenv("CLAW_VFS_ROOT", str(tmp_path))
        monkeypatch.setenv("CLAW_VFS_USER", "local")
        monkeypatch.delenv("FD_OWNER_ID", raising=False)
        (tmp_path / "local" / ".vfs-links.json").write_text(json.dumps({
            "acme": vfs_drive.link_entry("local", "acme", "ROOT"),
        }))
    return root, drive


class TestDocxExtractOnDrive:
    """The reported bug and its fix, end to end through the real tool."""

    async def test_absolute_placeholder_path_extracts_real_content(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import DocxExtractTool

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("f1", "report.docx", mime=DOCX_MIME)]},
            {"f1": (_docx_bytes("The merger closed in Q3."), ".docx")},
        )
        r = await DocxExtractTool().execute(path=str(root / "report.docx"))
        assert r.success is True
        assert "merger closed in Q3" in r.content
        assert "not a valid ZIP" not in (r.error or "")  # the old failure is gone

    async def test_vfs_path_extracts_real_content(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import DocxExtractTool

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("f1", "report.docx", mime=DOCX_MIME)]},
            {"f1": (_docx_bytes("Revenue up sharply."), ".docx")},
            with_vfs_env=True,
        )
        r = await DocxExtractTool().execute(path="vfs:acme/report.docx")
        assert r.success is True and "Revenue up sharply" in r.content

    async def test_second_extract_uses_cache(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import DocxExtractTool

        root, drive = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("f1", "report.docx", mime=DOCX_MIME)]},
            {"f1": (_docx_bytes("Once fetched."), ".docx")},
        )
        await DocxExtractTool().execute(path=str(root / "report.docx"))
        drive.fetch_calls.clear()
        r = await DocxExtractTool().execute(path=str(root / "report.docx"))
        assert r.success is True and drive.fetch_calls == []  # no second download

    async def test_drive_error_is_a_clear_message_not_a_zip_error(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import DocxExtractTool

        class Boom(FakeDrive):
            async def fetch(self, f, *, sleep=None):
                raise DriveError("network down")

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("f1", "report.docx", mime=DOCX_MIME)]},
            {}, drive_cls=Boom,
        )
        r = await DocxExtractTool().execute(path=str(root / "report.docx"))
        assert r.success is False
        assert "Google Drive" in r.error and "network down" in r.error
        assert "ZIP" not in r.error  # not the misleading parse error the user saw


class TestResolveReadableFile:
    """The shared resolver image_ocr / image_vision / video_vision route through."""

    async def test_placeholder_returns_materialized_bytes(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import _resolve_readable_file

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("i1", "photo.png", mime="image/png")]},
            {"i1": (b"\x89PNG\r\n\x1a\n real image", ".png")},
        )
        p, err = await _resolve_readable_file(str(root / "photo.png"))
        assert err is None and p is not None
        assert p.read_bytes() == b"\x89PNG\r\n\x1a\n real image"
        assert p != (root / "photo.png")  # a cache blob, not the marker file
        assert p.suffix == ".png"  # so image_ocr's extension check still passes

    async def test_non_drive_file_is_returned_unchanged(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import _resolve_readable_file

        loose = tmp_path / "local.txt"
        loose.write_text("ordinary")
        p, err = await _resolve_readable_file(str(loose))
        assert err is None and p == loose.resolve()

    async def test_missing_file_reports_not_found(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import _resolve_readable_file

        p, err = await _resolve_readable_file(str(tmp_path / "nope.txt"))
        assert p is None and "not found" in err.lower()

    async def test_drive_error_surfaces_plainly(self, tmp_path, monkeypatch):
        from captain_claw.tools.document_extract import _resolve_readable_file

        class Boom(FakeDrive):
            async def fetch(self, f, *, sleep=None):
                raise DriveError("scope missing")

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("i1", "photo.png", mime="image/png")]},
            {}, drive_cls=Boom,
        )
        p, err = await _resolve_readable_file(str(root / "photo.png"))
        assert p is None and "Google Drive" in err and "scope missing" in err


class TestCvResolveInput:
    """cv resolves inside a worker thread (no running loop), so it uses the sync
    bridge. Mirror that with asyncio.to_thread so the bridge's asyncio.run works."""

    async def test_materializes_placeholder_off_loop(self, tmp_path, monkeypatch):
        from captain_claw.tools.cv import _resolve_input

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("i1", "logo.png", mime="image/png")]},
            {"i1": (b"\x89PNG\r\n\x1a\n bytes", ".png")},
        )
        p, err = await asyncio.to_thread(_resolve_input, str(root / "logo.png"), {})
        assert err is None and p is not None
        assert p.read_bytes() == b"\x89PNG\r\n\x1a\n bytes"

    async def test_non_drive_path_unchanged_off_loop(self, tmp_path, monkeypatch):
        from captain_claw.tools.cv import _resolve_input

        # Keep vfs_drive.user_root pointed somewhere harmless; a loose file isn't
        # in any mount, so _resolve_input returns it as-is.
        monkeypatch.setattr(vfs_drive, "user_root", lambda uid: tmp_path / uid)
        loose = tmp_path / "img.png"
        loose.write_text("x")
        p, err = await asyncio.to_thread(_resolve_input, str(loose), {})
        assert err is None and p == loose.resolve()


class TestSummarizeFilesOnDrive:
    """A folder mounted from Drive summarises real content, not placeholder text."""

    async def test_materialize_for_read_fetches_bytes(self, tmp_path, monkeypatch):
        from captain_claw.tools.summarize_files import SummarizeFilesTool

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("f1", "notes.txt", mime="text/plain")]},
            {"f1": (b"the quarterly numbers", ".txt")},
        )
        read_path, err = await SummarizeFilesTool._materialize_for_read(root / "notes.txt")
        assert err is None
        assert read_path.read_bytes() == b"the quarterly numbers"

    async def test_materialize_for_read_reports_drive_failure(self, tmp_path, monkeypatch):
        from captain_claw.tools.summarize_files import SummarizeFilesTool

        class Boom(FakeDrive):
            async def fetch(self, f, *, sleep=None):
                raise DriveError("down")

        root, _ = await _mount(
            tmp_path, monkeypatch,
            {"ROOT": [_file("f1", "notes.txt", mime="text/plain")]},
            {}, drive_cls=Boom,
        )
        read_path, err = await SummarizeFilesTool._materialize_for_read(root / "notes.txt")
        assert err is not None and "Google Drive" in err
