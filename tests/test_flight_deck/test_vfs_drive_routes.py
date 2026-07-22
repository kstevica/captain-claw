"""Drive mount through the real FD vfs_routes handlers, with a faked DriveClient.

Calls the endpoint functions directly (as the other FD route tests do) rather
than over ASGI — same coverage of the wiring (route -> vfs_drive -> link entry
-> /projects -> ordinary resolver), without entangling httpx's ASGI transport
with the test event loop. No Google, no network.
"""

import tempfile
from pathlib import Path

import pytest

from captain_claw.drive_client import DriveFile, FOLDER_MIME


class FakeDrive:
    def __init__(self):
        self.tree = {
            "ROOT": [
                DriveFile(id="F1", name="sub", mime_type=FOLDER_MIME),
                DriveFile(id="f1", name="a.txt", mime_type="text/plain", size=10,
                          modified_time="2026-07-20T00:00:00Z"),
            ],
            "F1": [DriveFile(id="f2", name="b.md", mime_type="text/markdown", size=20,
                             modified_time="2026-07-20T00:00:00Z")],
        }
        self.tree["root"] = self.tree["ROOT"]  # 'root' is Drive's My-Drive alias

        # file_id -> (bytes, ext), set by tests that exercise clonemd.
        self.content: dict = {}
        # Shared (Team) Drives the account can see, for the picker.
        self.shared = [DriveFile(id="SD1", name="Team Drive", mime_type=FOLDER_MIME)]

    async def list_folder(self, fid, *, drive_id="", order_by="folder,name",
                          max_files=None, sleep=None):
        return list(self.tree.get(fid, [])), False

    async def list_shared_drives(self, *, sleep=None):
        return list(self.shared)

    async def fetch(self, f, *, sleep=None):
        return self.content[f.id]

    async def close(self):
        pass


USER = {"id": "local"}


@pytest.fixture()
async def env(monkeypatch):
    """Wire vfs_routes + vfs_drive onto a temp root and a shared fake Drive."""
    tmp = Path(tempfile.mkdtemp())
    import captain_claw.flight_deck.vfs_routes as vr
    from captain_claw.flight_deck import auth
    from captain_claw.flight_deck.db import FlightDeckDB
    from captain_claw import vfs_drive

    monkeypatch.setattr(vr, "_user_root", lambda uid: tmp / "vfs" / uid)
    monkeypatch.setattr(vfs_drive, "user_root", lambda uid: tmp / "vfs" / uid)
    fake = FakeDrive()
    monkeypatch.setattr(vr, "_drive_client", lambda: fake)

    db = FlightDeckDB(str(tmp / "fd.db"))
    await db.init()
    auth.set_auth_db(db)

    try:
        yield vr, vfs_drive, tmp, fake
    finally:
        # aiosqlite runs a non-daemon connection thread; without this close the
        # interpreter can't exit and the whole test process hangs.
        await db.close()


class TestDriveMountRoutes:
    async def test_browse_root_surfaces_shared_drives(self, env):
        vr, _, _, _ = env
        out = await vr.drive_browse(folder_id="root", drive_id="", user=USER)
        assert [d["name"] for d in out["shared_drives"]] == ["Team Drive"]
        assert {"id": "F1", "name": "sub"} in out["folders"]  # My Drive folders too

    async def test_mount_a_shared_drive_threads_its_id(self, env):
        vr, vfs_drive, tmp, _ = env
        await vr.mount_drive(
            vr.DriveMountBody(name="team", folder_id="SD1", drive_id="SD1"), USER
        )
        man = vfs_drive.Manifest.load(tmp / "vfs" / "local" / ".drive" / "team")
        assert man.shared_drive_id == "SD1"

    async def test_mount_populates_tree_and_lists_as_gdrive(self, env):
        vr, vfs_drive, tmp, _ = env
        summary = await vr.mount_drive(
            vr.DriveMountBody(name="acme", folder_id="ROOT"), USER
        )
        assert summary["files"] == 2 and summary["dirs"] == 1  # whole tree

        mount = tmp / "vfs" / "local" / ".drive" / "acme"
        assert (mount / "a.txt").is_file()
        assert (mount / "sub" / "b.md").is_file()
        assert "Google Drive" in (mount / "a.txt").read_text()

        projects = (await vr.list_projects(USER))["projects"]
        names = {p["name"]: p for p in projects}
        assert names["acme"]["kind"] == "gdrive"
        assert names["acme"]["mode"] == "ro"
        assert ".drive" not in names  # the mount dotdir isn't itself a project
        # Drive meta the UI reads: last-synced stamped, materialisation counts.
        d = names["acme"]["drive"]
        assert d["synced_at"] > 0
        assert d["total"] == 2 and d["cloned"] == 0 and d["uncloned"] == 2

        listing = await vr.list_dir("acme", "", "", USER)
        assert {"a.txt", "sub"} <= {e["name"] for e in listing["entries"]}

    async def test_mount_stores_and_surfaces_the_source_path(self, env):
        # The human breadcrumb rides through to /projects so the sidebar can show
        # it as a subtitle (disambiguates a short mount name like "VC").
        vr, vfs_drive, tmp, _ = env
        await vr.mount_drive(
            vr.DriveMountBody(name="VC", folder_id="ROOT",
                              path="FRC3/Reporting/Startup reports/Performance/VC"),
            USER,
        )
        vc = next(p for p in (await vr.list_projects(USER))["projects"] if p["name"] == "VC")
        assert vc["drive"]["source_path"] == "FRC3/Reporting/Startup reports/Performance/VC"

    async def test_mount_stream_reports_progress_then_done(self, env):
        # The streaming mount emits progress lines while walking, then a final
        # done with the summary — what the picker shows as a live status line.
        import json as _json

        vr, vfs_drive, tmp, _ = env
        resp = await vr.mount_drive_stream(
            vr.DriveMountBody(name="acme", folder_id="ROOT"), USER
        )
        events = []
        async for chunk in resp.body_iterator:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for ln in text.splitlines():
                if ln.strip():
                    events.append(_json.loads(ln))

        kinds = [e["event"] for e in events]
        assert "progress" in kinds
        done = [e for e in events if e["event"] == "done"]
        assert done and done[0]["name"] == "acme" and done[0]["files"] == 2
        # …and the mount really landed.
        assert "acme" in {p["name"] for p in (await vr.list_projects(USER))["projects"]}

    async def test_mount_stream_reports_an_error_event(self, env):
        import json as _json

        vr, _, _, _ = env
        resp = await vr.mount_drive_stream(
            vr.DriveMountBody(name="", folder_id="ROOT"), USER  # invalid name → 400
        )
        events = []
        async for chunk in resp.body_iterator:
            text = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
            for ln in text.splitlines():
                if ln.strip():
                    events.append(_json.loads(ln))
        assert any(e["event"] == "error" for e in events)

    async def test_read_only_mount_refuses_writes(self, env):
        vr, _, _, _ = env
        from fastapi import HTTPException

        await vr.mount_drive(vr.DriveMountBody(name="acme", folder_id="ROOT"), USER)
        with pytest.raises(HTTPException) as exc:
            await vr.write_file(
                vr.WriteBody(project="acme", path="x.txt", content="hi"), USER
            )
        assert exc.value.status_code == 403  # mode: ro

    async def test_clonemd_toggle_converts_now(self, env):
        vr, vfs_drive, tmp, fake = env
        # Give the mount fetchable text so enabling clonemd produces real files.
        fake.content = {"f1": (b"the merger closed in Q3", ".txt"),
                        "f2": (b"# deep\n\nnested note", ".md")}
        await vr.mount_drive(vr.DriveMountBody(name="acme", folder_id="ROOT"), USER)
        mount = tmp / "vfs" / "local" / ".drive" / "acme"
        # Before: a.txt is a placeholder marker.
        assert "Google Drive" in (mount / "a.txt").read_text()

        r = await vr.toggle_clonemd("acme", vr.DriveToggleBody(clonemd=True), USER)
        assert r["clonemd"] is True and r.get("cloned", 0) >= 1

        # After: real content on disk, manifest cloned, listing reflects it.
        assert (mount / "a.txt").read_text() == "the merger closed in Q3"
        man = vfs_drive.Manifest.load(mount)
        assert man.clonemd is True
        assert man.files["a.txt"]["state"] == "cloned"
        acme = next(p for p in (await vr.list_projects(USER))["projects"] if p["name"] == "acme")
        assert acme["drive"]["clonemd"] is True

    async def test_download_serves_the_original_not_the_marker(self, env, monkeypatch):
        # The panel previews converted markdown, but a download must be the real
        # file. materialize() reaches Drive via make_client(), so patch that too.
        vr, vfs_drive, tmp, fake = env
        monkeypatch.setattr("captain_claw.drive_client.make_client", lambda: fake)
        fake.content = {"f1": (b"%PDF-1.4 real original bytes", ".txt")}
        await vr.mount_drive(vr.DriveMountBody(name="acme", folder_id="ROOT"), USER)
        mount = tmp / "vfs" / "local" / ".drive" / "acme"
        assert "Google Drive" in (mount / "a.txt").read_text()  # on disk = marker

        resp = await vr.download_file(project="acme", path="a.txt", user=USER)
        assert resp.filename == "a.txt"                          # original name
        assert Path(resp.path).read_bytes() == b"%PDF-1.4 real original bytes"

    async def test_download_serves_from_the_byte_cache(self, env, monkeypatch):
        # The served file is the materialised blob under .drive-cache, not the
        # placeholder marker sitting at the mount path.
        vr, vfs_drive, tmp, fake = env
        monkeypatch.setattr("captain_claw.drive_client.make_client", lambda: fake)
        fake.content = {"f1": (b"bytes", ".txt")}
        await vr.mount_drive(vr.DriveMountBody(name="acme", folder_id="ROOT"), USER)
        resp = await vr.download_file(project="acme", path="a.txt", user=USER)
        served = Path(resp.path)
        assert ".drive-cache" in served.parts and served.read_bytes() == b"bytes"
        assert served != (tmp / "vfs" / "local" / ".drive" / "acme" / "a.txt")

    async def test_refresh_prunes_vanished(self, env):
        vr, vfs_drive, tmp, fake = env
        await vr.mount_drive(vr.DriveMountBody(name="acme", folder_id="ROOT"), USER)
        mount = tmp / "vfs" / "local" / ".drive" / "acme"
        assert (mount / "a.txt").is_file()
        fake.tree["ROOT"] = [f for f in fake.tree["ROOT"] if f.name != "a.txt"]
        await vr.refresh_drive("acme", USER)
        assert not (mount / "a.txt").exists()

    async def test_refresh_rejects_non_drive_link(self, env):
        vr, _, tmp, _ = env
        from fastapi import HTTPException

        # An ordinary (non-Drive) link should not be refreshable as a mount.
        ext = tmp / "external"
        ext.mkdir(parents=True)
        await vr.add_link(vr.LinkBody(name="plain", path=str(ext), mode="ro"), USER)
        with pytest.raises(HTTPException) as exc:
            await vr.refresh_drive("plain", USER)
        assert exc.value.status_code == 404

    async def test_unmount_removes_link_and_tree(self, env):
        vr, vfs_drive, tmp, _ = env
        await vr.mount_drive(vr.DriveMountBody(name="acme", folder_id="ROOT"), USER)
        await vr.unmount_drive("acme", keep_cloned=False, user=USER)
        assert "acme" not in {p["name"] for p in (await vr.list_projects(USER))["projects"]}
        assert not (tmp / "vfs" / "local" / ".drive" / "acme").exists()

    async def test_name_collision_with_physical_project_is_refused(self, env):
        vr, _, tmp, _ = env
        from fastapi import HTTPException

        (tmp / "vfs" / "local" / "taken").mkdir(parents=True)
        with pytest.raises(HTTPException) as exc:
            await vr.mount_drive(vr.DriveMountBody(name="taken", folder_id="ROOT"), USER)
        assert exc.value.status_code == 409
