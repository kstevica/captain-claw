"""The agent's upload routes and the file routes that serve uploads back.

``/api/file/upload`` takes any file from an authenticated caller (Flight
Deck's WhatsApp bridge hands over voice notes, contacts, JSON, …), names
every upload uniquely so a photo album never collapses into one file, keeps
a zip it can't extract, and refuses a public visitor with no session.
``/api/files/view``, ``/api/files/download`` and ``/api/media`` serve with
nosniff, and an HTML/SVG/XML file only under a sandbox CSP.

Everything runs against a tmp workspace (nothing reaches ~/.captain-claw).
"""

from __future__ import annotations

import io
import json
import types
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from captain_claw.config import get_config
from captain_claw.web import public_auth, rest_file_upload, rest_files, rest_image_upload
from captain_claw.web.active_content import ACTIVE_VIEW_CSP

SESSION = "sess-owner"


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    cfg = get_config()
    ws = (tmp_path / "workspace").resolve()
    (ws / "saved").mkdir(parents=True)
    (ws / "output").mkdir()
    monkeypatch.setattr(cfg.workspace, "path", str(ws))
    monkeypatch.setattr(cfg.web, "public_run", False)
    return ws


@pytest.fixture
async def client(workspace):
    server = types.SimpleNamespace(
        agent=types.SimpleNamespace(session=types.SimpleNamespace(id=SESSION)),
        _orchestrator=None,
    )

    def _route(fn):
        async def handler(request):
            return await fn(server, request)
        return handler

    app = web.Application()
    app.router.add_post("/api/file/upload", _route(rest_file_upload.upload_file))
    app.router.add_post("/api/image/upload", _route(rest_image_upload.upload_image))
    app.router.add_get("/api/files/view", _route(rest_files.view_file))
    app.router.add_get("/api/files/download", _route(rest_files.download_file))
    app.router.add_get("/api/media", _route(rest_files.serve_media))
    c = TestClient(TestServer(app))
    await c.start_server()
    try:
        yield c
    finally:
        await c.close()


def _form(name: str, data: bytes) -> FormData:
    # Unquoted, as browsers and httpx (the bridge) send names; aiohttp's
    # client would percent-encode "voice note.ogg" by default.
    form = FormData(quote_fields=False)
    form.add_field("file", data, filename=name, content_type="application/octet-stream")
    return form


async def _upload(client, name: str, data: bytes, *, route: str = "/api/file/upload",
                  params: dict | None = None):
    resp = await client.post(route, data=_form(name, data), params=params or {})
    return resp.status, json.loads(await resp.text())


def _public(monkeypatch, session_id: str | None) -> None:
    monkeypatch.setattr(public_auth, "get_request_session_id", lambda request: (True, session_id))


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _encrypted_zip() -> bytes:
    """A zip whose entry claims to be encrypted (flag bit 0) — zipfile can't
    write one, and reading it without a password raises RuntimeError."""
    raw = bytearray(_zip({"secret.txt": b"hidden"}))
    for sig, flag_at in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        i = raw.find(sig)
        raw[i + flag_at] |= 0x01
    return bytes(raw)


def _bad_deflate_zip() -> bytes:
    """A zip whose deflate stream starts with a reserved block type — zipfile
    raises zlib.error (not BadZipFile) while extracting."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.txt", b"hello " * 500)
    raw = bytearray(buf.getvalue())
    name_len, extra_len = int.from_bytes(raw[26:28], "little"), int.from_bytes(raw[28:30], "little")
    raw[30 + name_len + extra_len] = 0x07  # BFINAL=1, BTYPE=11 (reserved)
    return bytes(raw)


# ── extensions ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name,data,ext", [
    ("voice note.ogg", b"OggS....", ".ogg"),
    ("contact.vcf", b"BEGIN:VCARD", ".vcf"),
    ("data.json", b'{"a": 1}', ".json"),
    ("README", b"no extension", ""),
])
async def test_an_authenticated_caller_uploads_any_extension(client, workspace, name, data, ext):
    status, body = await _upload(client, name, data)
    assert status == 200, body
    saved = Path(body["path"])
    assert saved.parent == workspace / "saved" / "downloads" / SESSION
    assert saved.read_bytes() == data
    assert saved.suffix == ext
    assert body["filename"] == name and body["size"] == len(data)


async def test_a_public_session_keeps_the_allowlist(client, workspace, monkeypatch):
    _public(monkeypatch, "pub-1")
    status, body = await _upload(client, "voice.ogg", b"OggS")
    assert status == 400 and "Unsupported file type '.ogg'" in body["error"]
    status, body = await _upload(client, "table.csv", b"a,b\n1,2\n")
    assert status == 200
    assert Path(body["path"]).parent == workspace / "saved" / "downloads" / "pub-1"


@pytest.mark.parametrize("route", ["/api/file/upload", "/api/image/upload"])
async def test_a_public_visitor_without_a_session_is_refused(client, workspace, monkeypatch, route):
    _public(monkeypatch, None)
    status, body = await _upload(client, "photo.png", b"\x89PNG", route=route)
    assert status == 403 and "error" in body
    # Nothing landed in the owner's folder (or anywhere else).
    assert not any((workspace / "saved").rglob("*.png"))


# ── names ────────────────────────────────────────────────────────────


class _FrozenClock:
    @staticmethod
    def now(tz=None):
        return datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize("route,sub", [("/api/file/upload", "downloads"),
                                       ("/api/image/upload", "media")])
async def test_same_name_in_the_same_second_never_overwrites(client, workspace, monkeypatch,
                                                             route, sub):
    monkeypatch.setattr(rest_file_upload, "datetime", _FrozenClock)
    tokens = iter(["aaaaaa", "aaaaaa", "bbbbbb"])     # the second upload collides once
    monkeypatch.setattr(rest_file_upload.secrets, "token_hex", lambda n: next(tokens))
    s1, b1 = await _upload(client, "IMG-1.jpg", b"first photo", route=route)
    s2, b2 = await _upload(client, "IMG-1.jpg", b"second photo", route=route)
    assert s1 == s2 == 200
    p1, p2 = Path(b1["path"]), Path(b2["path"])
    assert p1.name == "IMG-1-20261009-120000-aaaaaa.jpg"
    assert p2.name == "IMG-1-20261009-120000-bbbbbb.jpg"
    assert p1.parent == p2.parent == workspace / "saved" / sub / SESSION
    assert p1.read_bytes() == b"first photo" and p2.read_bytes() == b"second photo"


async def test_a_dotfile_name_is_an_extension(client):
    status, body = await _upload(client, ".xlsx", b"PK-ish")
    assert status == 200
    name = Path(body["path"]).name
    assert name.startswith("file-") and name.endswith(".xlsx")


def test_upload_names_are_sanitised():
    s = rest_file_upload.sanitize_upload_name
    assert s(".xlsx") == ("", ".xlsx")
    assert s("..hidden.TXT") == ("hidden", ".txt")
    assert s("cafe\u0301 menu.PDF") == ("caf\u00e9_menu", ".pdf")      # NFC, not "cafe__menu"
    assert s("../../etc/passwd") == ("passwd", "")
    assert s("C:\\Users\\x\\voice.OGG") == ("voice", ".ogg")
    assert s("notes.waytoolongext") == ("notes.waytoolongext", "")
    assert s("report") == ("report", "")


async def test_an_empty_upload_is_refused_and_leaves_nothing(client, workspace):
    status, body = await _upload(client, "empty.txt", b"")
    assert status == 400 and body["error"] == "Empty file"
    assert not any((workspace / "saved").rglob("empty*"))


# ── zips ─────────────────────────────────────────────────────────────


async def test_a_zip_is_extracted_by_default_into_its_own_folder(client, workspace):
    s1, b1 = await _upload(client, "photos.zip", _zip({"a/one.txt": b"1"}))
    s2, b2 = await _upload(client, "photos.zip", _zip({"a/two.txt": b"2"}))
    assert s1 == s2 == 200
    assert b1["extracted"] is True and b1["files"] == ["a/one.txt"]
    d1, d2 = Path(b1["path"]), Path(b2["path"])
    assert d1.is_dir() and d2.is_dir() and d1 != d2           # two zips never merge
    assert (d1 / "a" / "one.txt").read_bytes() == b"1"
    assert not (d1 / "a" / "two.txt").exists()
    assert not list(d1.parent.glob("*.zip"))                  # the zips are gone


async def test_extract_0_keeps_the_zip(client):
    data = _zip({"x.txt": b"x"})
    status, body = await _upload(client, "bundle.zip", data, params={"extract": "0"})
    assert status == 200
    p = Path(body["path"])
    assert p.suffix == ".zip" and p.read_bytes() == data
    assert "extracted" not in body
    status, body = await _upload(client, "bundle.zip", data, params={"extract": "false"})
    assert Path(body["path"]).suffix == ".zip"


@pytest.mark.parametrize("data,reason", [
    (_encrypted_zip(), "password-protected"),
    (b"PK\x03\x04 this is not really a zip", "not a valid zip"),
    (_bad_deflate_zip(), "not a valid zip"),
    (_zip({"docs": b"a file", "docs/readme.txt": b"x"}), "two entries collide"),
    (_zip({"../escape.txt": b"x"}), "outside the archive folder"),
])
async def test_a_zip_that_cant_be_extracted_is_kept(client, workspace, data, reason):
    status, body = await _upload(client, "archive.zip", data)
    assert status == 200, body
    assert body["extracted"] is False and reason in body["extract_error"]
    p = Path(body["path"])
    assert p.suffix == ".zip" and p.read_bytes() == data
    assert not p.with_suffix("").exists()                     # no half-extracted folder
    assert not (workspace / "saved" / "downloads" / "escape.txt").exists()


async def test_a_zip_over_the_entry_cap_is_kept(client, monkeypatch):
    monkeypatch.setattr(rest_file_upload, "_ZIP_MAX_MEMBERS", 2)
    status, body = await _upload(client, "many.zip", _zip({f"{i}.txt": b"x" for i in range(3)}))
    assert status == 200
    assert body["extracted"] is False and "more than 2 entries" in body["extract_error"]


async def test_a_zip_over_the_size_cap_is_kept(client, monkeypatch):
    monkeypatch.setattr(rest_file_upload, "_ZIP_MAX_TOTAL_BYTES", 10)
    status, body = await _upload(client, "big.zip", _zip({"big.txt": b"x" * 100}))
    assert status == 200
    assert body["extracted"] is False and "uncompressed" in body["extract_error"]


# ── serving files back ───────────────────────────────────────────────


def _put(workspace: Path, name: str, data: bytes = b"<script>alert(1)</script>") -> Path:
    p = workspace / "saved" / "downloads" / SESSION / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.mark.parametrize("name", ["page.html", "pic.svg", "feed.xml"])
async def test_view_sandboxes_active_types(client, workspace, name):
    p = _put(workspace, name)
    resp = await client.get("/api/files/view", params={"path": str(p)})
    assert resp.status == 200
    assert resp.headers["Content-Security-Policy"] == ACTIVE_VIEW_CSP
    assert resp.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("name", ["notes.txt", "doc.pdf", "blob"])
async def test_view_sends_nosniff_without_csp_for_passive_types(client, workspace, name):
    p = _put(workspace, name, b"plain")
    resp = await client.get("/api/files/view", params={"path": str(p)})
    assert resp.status == 200
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" not in resp.headers


async def test_download_and_media_send_the_same_guard(client, workspace):
    html = _put(workspace, "page.html")
    resp = await client.get("/api/files/download", params={"path": str(html)})
    assert resp.status == 200
    assert resp.headers["Content-Security-Policy"] == ACTIVE_VIEW_CSP
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Content-Disposition"].startswith("attachment;")

    svg = _put(workspace, "pic.svg")
    resp = await client.get("/api/media", params={"path": str(svg)})
    assert resp.status == 200
    assert resp.headers["Content-Security-Policy"] == ACTIVE_VIEW_CSP
    assert resp.headers["X-Content-Type-Options"] == "nosniff"

    png = _put(workspace, "pic.png", b"\x89PNG")
    resp = await client.get("/api/media", params={"path": str(png)})
    assert resp.status == 200
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" not in resp.headers


def test_flight_deck_proxy_uses_the_shared_guard():
    # Read, not imported: importing the FD server loads a .env found above it.
    import captain_claw.web.active_content as active_content

    src = (Path(active_content.__file__).parent.parent / "flight_deck" / "server.py").read_text()
    assert "from captain_claw.web import active_content as _active_content" in src
    assert "_ACTIVE_VIEW_CSP = _active_content.ACTIVE_VIEW_CSP" in src
    assert "_active_view_type = _active_content.active_view_type" in src
    assert active_content.active_view_type("image/svg+xml; charset=utf-8")
    assert active_content.active_view_type("application/atom+xml")
    assert not active_content.active_view_type("text/plain")


async def test_an_extraction_error_never_echoes_host_paths(client, workspace):
    status, body = await _upload(client, "clash.zip", _zip({"docs": b"a", "docs/readme.txt": b"x"}))
    assert status == 200 and body["extracted"] is False
    assert str(workspace) not in body["extract_error"] and "/" not in body["extract_error"]
