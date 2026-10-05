"""google_drive — the native Drive tool that replaces the retired gws CLI.

Covers what gws used to do for agents: Drive URLs accepted as ids, a
``download`` action that saves a local copy inside saved/ (Google Docs/
Sheets/Slides exported), shared drives on every call, Docs exports without
inline base64 images, every tab of a Sheet on read, and list/search that
page past one request and say when more files exist. The list/search text
format is pinned: the scale loop parses it.

No network: Drive is an ``httpx.MockTransport``; the token source is faked.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import httpx
import pytest

import captain_claw.session as session_mod
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.tools import google_drive as gd
from captain_claw.tools.google_drive import GoogleDriveTool
from captain_claw.tools.registry import ToolResult

FID = "1AbCdEfGhIjKlMnOpQrStUvWxYz_0123-456789"
DOC_MIME = "application/vnd.google-apps.document"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"
SLIDES_MIME = "application/vnd.google-apps.presentation"
FOLDER_MIME = "application/vnd.google-apps.folder"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk" * 50


# ── fakes ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_real_session_manager(monkeypatch):
    # The tool builds GoogleOAuthManager(get_session_manager()); never touch a DB.
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())


@pytest.fixture(autouse=True)
def _connected(monkeypatch):
    """Standalone mode, Google connected with full Drive scope."""
    monkeypatch.setattr(
        GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""),
    )

    async def _get_tokens(self):
        return GoogleOAuthTokens(
            access_token="DRIVE-TOKEN", refresh_token="",
            expires_at=time.time() + 3300,
            scope="https://www.googleapis.com/auth/drive",
        )

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)


class _FakeDrive:
    """Just enough of Drive v3: metadata, alt=media, export, list, update."""

    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.media: dict[str, bytes] = {}
        self.exports: dict[tuple[str, str], bytes] = {}
        self.export_status: dict[tuple[str, str], int] = {}
        self.listing: list[dict] = []
        # pageToken → files.list response; when set, list/search page through it.
        self.pages: dict[str | None, dict] = {}
        self.requests: list[httpx.Request] = []

    def add(self, fid: str, name: str, mime: str, *, media: bytes = b"", size: bool = True) -> None:
        meta = {"id": fid, "name": name, "mimeType": mime, "modifiedTime": "2026-09-01T10:00:00Z"}
        if size and media:
            meta["size"] = str(len(media))
        self.files[fid] = meta
        self.media[fid] = media

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        params = request.url.params
        if request.method == "PATCH":
            fid = path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={"id": fid, "name": self.files[fid]["name"]})
        if path == "/drive/v3/files":
            if self.pages:
                return httpx.Response(200, json=self.pages[params.get("pageToken")])
            return httpx.Response(200, json={"files": self.listing})
        if path.endswith("/export"):
            fid = path.split("/")[-2]
            key = (fid, params["mimeType"])
            if key in self.export_status:
                return httpx.Response(
                    self.export_status[key],
                    json={"error": {"message": "This file is too large to be exported."}},
                )
            if key not in self.exports:
                return httpx.Response(400, json={"error": {"message": "export not supported"}})
            return httpx.Response(200, content=self.exports[key])
        fid = path.rsplit("/", 1)[-1]
        if fid not in self.files:
            return httpx.Response(404, json={"error": {"message": "File not found"}})
        if params.get("alt") == "media":
            return httpx.Response(200, content=self.media[fid])
        return httpx.Response(200, json=self.files[fid])

    def file_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if "/files/" in r.url.path]


@pytest.fixture
async def drive():
    fake = _FakeDrive()
    tool = GoogleDriveTool()
    await tool._client.aclose()
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    fake.tool = tool
    yield fake
    await tool.close()


class _Registry:
    def __init__(self) -> None:
        self.registered: list[tuple[str, str]] = []

    def register(self, logical_path, physical_path, *, task_id=""):
        self.registered.append((logical_path, str(physical_path)))


def _xlsx(sheets: dict[str, list[list[str]]]) -> bytes:
    """A minimal XLSX workbook (inline-string cells), one sheet per tab."""
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        tabs = "".join(
            f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(sheets, 1)
        )
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>{tabs}</sheets></workbook>')
        rels = "".join(
            f'<Relationship Id="rId{i}" Type="worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets) + 1)
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>',
        )
        for i, rows in enumerate(sheets.values(), 1):
            body = "".join(
                f'<row r="{r}">'
                + "".join(
                    f'<c r="{chr(65 + c)}{r}" t="inlineStr"><is><t>{v}</t></is></c>'
                    for c, v in enumerate(row)
                )
                + "</row>"
                for r, row in enumerate(rows, 1)
            )
            z.writestr(f"xl/worksheets/sheet{i}.xml", f'<worksheet xmlns="{main}"><sheetData>{body}</sheetData></worksheet>')
    return buf.getvalue()


def _pptx(slides: list[str]) -> bytes:
    """A minimal PPTX deck: one text run per slide."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, text in enumerate(slides, 1):
            z.writestr(
                f"ppt/slides/slide{i}.xml",
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>",
            )
    return buf.getvalue()


def _runtime(tmp_path: Path, **extra) -> dict:
    return {
        "_runtime_base_path": tmp_path,
        "_saved_base_path": tmp_path / "saved",
        "_session_id": "sess-1",
        **extra,
    }


# ── URLs as ids ──────────────────────────────────────────────────────


async def test_docs_url_accepted_as_file_id(drive):
    drive.add(FID, "Plan", DOC_MIME)
    res = await drive.tool.execute(
        "info", file_id=f"https://docs.google.com/document/d/{FID}/edit?usp=sharing",
    )
    assert res.success, res.error
    assert f"ID: {FID}" in res.content
    assert drive.requests[-1].url.path == f"/drive/v3/files/{FID}"


async def test_folder_url_accepted_as_folder_id(drive):
    res = await drive.tool.execute(
        "list", folder_id=f"https://drive.google.com/drive/u/0/folders/{FID}?usp=sharing",
    )
    assert res.success, res.error
    assert drive.requests[-1].url.params["q"] == f"'{FID}' in parents and trashed = false"


async def test_non_drive_url_refused_without_a_request(drive):
    res = await drive.tool.execute(
        "read", file_id=f"https://storage.googleapis.com/bucket/{FID}",
    )
    assert not res.success
    assert "Could not find a Google Drive id" in res.error
    assert drive.requests == []


async def test_bare_id_and_root_pass_through(drive):
    res = await drive.tool.execute("list", folder_id="root")
    assert res.success
    assert drive.requests[-1].url.params["q"].startswith("'root' in parents")


# ── download ─────────────────────────────────────────────────────────


async def test_download_binary_lands_in_saved_downloads(drive, tmp_path):
    drive.add(FID, "Q3 Report.pdf", "application/pdf", media=b"%PDF-1.7 fake")
    registry = _Registry()
    res = await drive.tool.execute(
        "download", file_id=FID, **_runtime(tmp_path, _file_registry=registry),
    )
    assert res.success, res.error
    dest = (tmp_path / "saved" / "downloads" / "sess-1" / "Q3 Report.pdf").resolve()
    assert dest.read_bytes() == b"%PDF-1.7 fake"
    assert f"Path: {dest}" in res.content
    assert "pdf_extract" in res.content
    assert registry.registered == [("Q3 Report.pdf", str(dest))]


async def test_download_google_doc_exports_markdown_without_images(drive, tmp_path):
    drive.add(FID, "Design notes", DOC_MIME)
    drive.exports[(FID, "text/markdown")] = (
        f"# Notes\n\n![chart](data:image/png;base64,{_PNG_B64})\n\nBody text.\n"
    ).encode()
    res = await drive.tool.execute("download", file_id=FID, **_runtime(tmp_path))
    assert res.success, res.error
    dest = tmp_path / "saved" / "downloads" / "sess-1" / "Design notes.md"
    text = dest.read_text()
    assert "# Notes" in text and "Body text." in text
    assert "base64" not in text and "![chart]([image])" in text
    assert "exported as text/markdown" in res.content


async def test_download_sheet_defaults_to_xlsx(drive, tmp_path):
    drive.add(FID, "Budget", SHEET_MIME)
    xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    drive.exports[(FID, xlsx)] = b"PK\x03\x04 fake xlsx"
    res = await drive.tool.execute("download", file_id=FID, **_runtime(tmp_path))
    assert res.success, res.error
    assert (tmp_path / "saved" / "downloads" / "sess-1" / "Budget.xlsx").read_bytes().startswith(b"PK")


async def test_download_output_path_extension_picks_export_format(drive, tmp_path):
    drive.add(FID, "Plan", DOC_MIME)
    drive.exports[(FID, "application/pdf")] = b"%PDF doc"
    res = await drive.tool.execute(
        "download", file_id=FID, output_path="plan.pdf", **_runtime(tmp_path),
    )
    assert res.success, res.error
    dest = tmp_path / "saved" / "downloads" / "sess-1" / "plan.pdf"
    assert dest.read_bytes() == b"%PDF doc"


@pytest.mark.parametrize(
    "output_path,expected",
    [
        ("copy.md", "downloads/sess-1/copy.md"),
        ("notes", "downloads/sess-1/notes.md"),  # export extension added
        ("downloads/", "downloads/sess-1/Plan.md"),
        ("saved/media/sess-1/p.md", "media/sess-1/p.md"),
        ("output/plan.md", "output/sess-1/plan.md"),
    ],
)
async def test_download_relative_output_paths_are_session_scoped(
    drive, tmp_path, output_path, expected,
):
    drive.add(FID, "Plan", DOC_MIME)
    drive.exports[(FID, "text/markdown")] = b"# Plan"
    res = await drive.tool.execute(
        "download", file_id=FID, output_path=output_path, **_runtime(tmp_path),
    )
    assert res.success, res.error
    assert (tmp_path / "saved" / expected).read_bytes() == b"# Plan"


async def test_download_markdown_unsupported_falls_back_to_plain_text(drive, tmp_path):
    drive.add(FID, "Old doc", DOC_MIME)
    drive.exports[(FID, "text/plain")] = b"plain body"
    res = await drive.tool.execute("download", file_id=FID, **_runtime(tmp_path))
    assert res.success, res.error
    assert (tmp_path / "saved" / "downloads" / "sess-1" / "Old doc.md").read_text() == "plain body"


@pytest.mark.parametrize(
    "output_path",
    ["../../escape.txt", "downloads/../../../escape.txt", "/etc/escape.txt"],
)
async def test_download_refuses_paths_outside_saved(drive, tmp_path, output_path):
    drive.add(FID, "a.txt", "text/plain", media=b"data")
    res = await drive.tool.execute(
        "download", file_id=FID, output_path=output_path, **_runtime(tmp_path),
    )
    assert not res.success
    assert "output_path" in res.error
    assert not any(r.url.params.get("alt") == "media" for r in drive.requests)
    assert not (tmp_path / "escape.txt").exists()


async def test_download_refuses_symlink_escape(drive, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sess = tmp_path / "saved" / "downloads" / "sess-1"
    sess.mkdir(parents=True)
    (sess / "link").symlink_to(outside, target_is_directory=True)
    drive.add(FID, "a.txt", "text/plain", media=b"data")
    res = await drive.tool.execute(
        "download", file_id=FID, output_path="downloads/link/a.txt", **_runtime(tmp_path),
    )
    assert not res.success
    assert list(outside.iterdir()) == []


async def test_download_absolute_path_inside_saved_is_allowed(drive, tmp_path):
    drive.add(FID, "a.txt", "text/plain", media=b"data")
    target = tmp_path / "saved" / "output" / "copy.txt"
    res = await drive.tool.execute(
        "download", file_id=FID, output_path=str(target), **_runtime(tmp_path),
    )
    assert res.success, res.error
    assert target.read_bytes() == b"data"


async def test_download_size_caps(drive, tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "_MAX_DOWNLOAD_BYTES", 10)
    drive.add(FID, "big.bin", "application/octet-stream", media=b"x" * 50)
    res = await drive.tool.execute("download", file_id=FID, **_runtime(tmp_path))
    assert not res.success and "too large" in res.error
    # Refused from the metadata size — the bytes were never fetched.
    assert not any(r.url.params.get("alt") == "media" for r in drive.requests)

    # Exports carry no size: the cap holds on the streamed bytes.
    drive.add("1DocAbcdefGhij", "Long doc", DOC_MIME)
    drive.exports[("1DocAbcdefGhij", "text/markdown")] = b"y" * 50
    res = await drive.tool.execute("download", file_id="1DocAbcdefGhij", **_runtime(tmp_path))
    assert not res.success and "too large" in res.error
    assert not (tmp_path / "saved" / "downloads" / "sess-1" / "Long doc.md").exists()


@pytest.mark.parametrize(
    "name,mime,expected,reader",
    [
        ("Contract", "application/pdf", "Contract.pdf", "pdf_extract"),
        ("Contract v2.1", "application/pdf", "Contract v2.1.pdf", "pdf_extract"),
        ("Offer", gd._DOCX_MIME, "Offer.docx", "docx_extract"),
        ("Screenshot", "image/png", "Screenshot.png", "image_vision"),
        ("notes.md", "text/plain", "notes.md", "read"),  # has one: kept
        ("blob", "application/octet-stream", "blob", "read"),
    ],
)
async def test_download_adds_the_extension_a_drive_name_lacks(
    drive, tmp_path, name, mime, expected, reader,
):
    drive.add(FID, name, mime, media=b"bytes")
    res = await drive.tool.execute("download", file_id=FID, **_runtime(tmp_path))
    assert res.success, res.error
    dest = (tmp_path / "saved" / "downloads" / "sess-1" / expected).resolve()
    assert dest.read_bytes() == b"bytes"
    assert f'Use {reader}(path="{dest}")' in res.content


async def test_download_folder_points_at_list(drive, tmp_path):
    drive.add(FID, "Team", FOLDER_MIME)
    res = await drive.tool.execute("download", file_id=FID, **_runtime(tmp_path))
    assert not res.success
    assert f"list with folder_id='{FID}'" in res.error


# ── shared drives ────────────────────────────────────────────────────


async def test_supports_all_drives_on_every_call(drive, tmp_path):
    drive.add(FID, "Plan", DOC_MIME)
    drive.exports[(FID, "text/markdown")] = b"# Plan"
    drive.add("1TxtAbcdefGhij", "notes.txt", "text/plain", media=b"hello")
    drive.add("1PdfAbcdefGhij", "r.pdf", "application/pdf", media=b"%PDF")

    for call in (
        {"action": "read", "file_id": FID},
        {"action": "read", "file_id": "1TxtAbcdefGhij"},
        {"action": "info", "file_id": FID},
        {"action": "download", "file_id": FID},
        {"action": "download", "file_id": "1PdfAbcdefGhij"},
        {"action": "update", "file_id": "1TxtAbcdefGhij", "content": "new"},
        {"action": "list", "folder_id": FID},
        {"action": "search", "query": "plan"},
    ):
        res = await drive.tool.execute(**call, **_runtime(tmp_path))
        assert res.success, (call, res.error)

    for req in drive.requests:
        assert req.url.params.get("supportsAllDrives") == "true", req.url
    listings = [r for r in drive.requests if r.url.path == "/drive/v3/files"]
    assert len(listings) == 2
    for req in listings:
        assert req.url.params["includeItemsFromAllDrives"] == "true"
        assert req.url.params["corpora"] == "allDrives"


# ── read ─────────────────────────────────────────────────────────────


async def test_read_doc_strips_base64_images(drive):
    drive.add(FID, "Plan", DOC_MIME)
    drive.exports[(FID, "text/markdown")] = (
        f"Intro\n\n[image1]: <data:image/png;base64,{_PNG_B64}>\n\nOutro"
    ).encode()
    res = await drive.tool.execute("read", file_id=f"https://docs.google.com/document/d/{FID}/edit")
    assert res.success, res.error
    assert "Intro" in res.content and "Outro" in res.content
    assert _PNG_B64[:40] not in res.content
    assert "[image]" in res.content


async def test_read_sheet_returns_every_tab(drive):
    """Drive's CSV export holds only the first tab; the XLSX export has all."""
    drive.add(FID, "Budget 2026", SHEET_MIME)
    drive.exports[(FID, XLSX)] = _xlsx({
        "Summary": [["Quarter", "Total"], ["Q1", "10"]],
        "Q2": [["Item", "Cost"], ["Rent", "1200"]],
        "Q3": [["Item", "Cost"], ["Travel", "300"]],
    })
    drive.exports[(FID, "text/csv")] = b"Quarter,Total\nQ1,10\n"
    res = await drive.tool.execute("read", file_id=FID)
    assert res.success, res.error
    for tab in ("## Sheet: Summary", "## Sheet: Q2", "## Sheet: Q3"):
        assert tab in res.content
    assert "| Rent | 1200 |" in res.content and "| Travel | 300 |" in res.content
    assert f"exported as {XLSX}" in res.content
    exported = [r.url.params["mimeType"] for r in drive.requests if r.url.path.endswith("/export")]
    assert exported == [XLSX]


async def test_read_sheet_falls_back_to_csv_when_xlsx_export_fails(drive):
    """Drive caps exports at 10 MB; the flat CSV still reads, and says so."""
    drive.add(FID, "Huge", SHEET_MIME)
    drive.export_status[(FID, XLSX)] = 403
    drive.exports[(FID, "text/csv")] = b"a,b\n1,2\n"
    res = await drive.tool.execute("read", file_id=FID)
    assert res.success, res.error
    assert "a,b\n1,2" in res.content
    assert "FIRST TAB ONLY" in res.content


async def test_read_sheet_flags_a_tab_past_the_row_cap(drive, monkeypatch):
    monkeypatch.setattr(gd, "_SHEET_READ_MAX_ROWS", 3)
    drive.add(FID, "Log", SHEET_MIME)
    drive.exports[(FID, XLSX)] = _xlsx({
        "Short": [["h"], ["1"]],
        "Long": [["h"], ["1"], ["2"], ["3"], ["4"]],
    })
    res = await drive.tool.execute("read", file_id=FID)
    assert res.success, res.error
    assert "a tab may run past the 3 rows shown" in res.content

    drive.exports[(FID, XLSX)] = _xlsx({"Short": [["h"], ["1"]]})
    res = await drive.tool.execute("read", file_id=FID)
    assert "may run past" not in res.content


async def test_read_slides_exports_pptx_slide_by_slide(drive):
    drive.add(FID, "Pitch", SLIDES_MIME)
    drive.exports[(FID, PPTX)] = _pptx(["Problem", "Solution", "Ask"])
    res = await drive.tool.execute("read", file_id=FID)
    assert res.success, res.error
    for n, text in enumerate(("Problem", "Solution", "Ask"), 1):
        assert f"## Slide {n}" in res.content and f"- {text}" in res.content


async def test_read_extensionless_binary_extracts_by_type(drive, monkeypatch):
    """A PDF named "Contract" reaches pdf_extract as a .pdf, not a .bin."""
    seen: list[str] = []

    class _Extract:
        async def execute(self, path, **kwargs):
            seen.append(Path(path).suffix)
            return ToolResult(success=True, content="contract text")

    monkeypatch.setattr(GoogleDriveTool, "_get_extract_tool", staticmethod(lambda name: _Extract()))
    drive.add(FID, "Contract", "application/pdf", media=b"%PDF-1.7")
    res = await drive.tool.execute("read", file_id=FID)
    assert res.success, res.error
    assert seen == [".pdf"]
    assert "contract text" in res.content


# ── resource keys (older link-shared files) ──────────────────────────


async def test_resource_key_from_a_pasted_url_is_sent(drive):
    drive.add(FID, "Shared", "text/plain", media=b"hello")
    url = f"https://drive.google.com/file/d/{FID}/view?resourcekey=0-AbC_dEf-123"
    res = await drive.tool.execute("read", file_id=url)
    assert res.success, res.error
    sent = {r.headers.get("X-Goog-Drive-Resource-Keys") for r in drive.requests}
    assert sent == {f"{FID}/0-AbC_dEf-123"}

    # Per call: the next call, by bare id, carries none.
    drive.requests.clear()
    res = await drive.tool.execute("read", file_id=FID)
    assert res.success, res.error
    assert all("X-Goog-Drive-Resource-Keys" not in r.headers for r in drive.requests)


# ── list / search paging ─────────────────────────────────────────────


def _entry(n: int) -> dict:
    return {"id": f"1File{n:04d}Abcdefgh", "name": f"file-{n}.pdf",
            "mimeType": "application/pdf", "modifiedTime": "2026-09-01T10:00:00Z"}


def _listing_requests(drive) -> list[httpx.Request]:
    return [r for r in drive.requests if r.url.path == "/drive/v3/files"]


async def test_list_defaults_to_100_and_says_when_more_exist(drive):
    drive.pages = {
        None: {"files": [_entry(n) for n in range(100)], "nextPageToken": "TOK2"},
    }
    res = await drive.tool.execute("list", folder_id="1ParentAbcdefGh")
    assert res.success, res.error
    assert res.content.startswith("Files in 1ParentAbcdefGh (100 results):")
    assert _listing_requests(drive)[0].url.params["pageSize"] == "100"
    assert res.content.endswith(
        "More files exist — showing 100. Repeat this call with page_token='TOK2' "
        "for the next page, or a higher max_results (up to 1000)."
    )


async def test_list_follows_pages_up_to_max_results(drive):
    drive.pages = {
        None: {"files": [_entry(n) for n in range(3)], "nextPageToken": "P2"},  # a short page
        "P2": {"files": [_entry(n) for n in range(3, 6)], "nextPageToken": "P3"},
        "P3": {"files": [_entry(n) for n in range(6, 8)]},
    }
    res = await drive.tool.execute("list", folder_id="1ParentAbcdefGh", max_results=500)
    assert res.success, res.error
    assert "(8 results)" in res.content
    assert "More files exist" not in res.content
    reqs = _listing_requests(drive)
    assert [r.url.params.get("pageToken") for r in reqs] == [None, "P2", "P3"]
    assert [r.url.params["pageSize"] for r in reqs] == ["500", "497", "494"]


async def test_list_stops_at_max_results_and_hands_back_the_token(drive):
    drive.pages = {
        None: {"files": [_entry(n) for n in range(2)], "nextPageToken": "P2"},
        "P2": {"files": [_entry(n) for n in range(2, 4)], "nextPageToken": "P3"},
    }
    res = await drive.tool.execute("list", folder_id="1ParentAbcdefGh", max_results=4)
    assert "(4 results)" in res.content
    assert "page_token='P3'" in res.content
    assert len(_listing_requests(drive)) == 2


async def test_page_token_continues_a_listing(drive):
    drive.pages = {"P2": {"files": [_entry(7)]}}
    res = await drive.tool.execute("search", query="file", page_token="P2")
    assert res.success, res.error
    assert "file-7.pdf" in res.content
    req = _listing_requests(drive)[0]
    assert req.url.params["pageToken"] == "P2"
    assert req.url.params["pageSize"] == "20"  # search keeps its default


async def test_max_results_capped_at_1000(drive):
    drive.pages = {None: {"files": [_entry(1)]}}
    await drive.tool.execute("list", folder_id="root", max_results=5000)
    assert _listing_requests(drive)[0].url.params["pageSize"] == "1000"


async def test_scale_loop_parses_a_listing_with_more_files(drive):
    from captain_claw.agent_scale_loop_mixin import AgentScaleLoopMixin

    drive.pages = {None: {"files": [_entry(1), _entry(2)], "nextPageToken": "P2"}}
    res = await drive.tool.execute("list", folder_id="1ParentAbcdefGh", max_results=2)
    assert AgentScaleLoopMixin._parse_gdrive_listing(res.content) == {
        "file-1.pdf": {"id": "1File0001Abcdefgh", "mimeType": "application/pdf"},
        "file-2.pdf": {"id": "1File0002Abcdefgh", "mimeType": "application/pdf"},
    }


# ── list / search text format (contract with the scale loop) ─────────


_ENTRIES = [
    {"id": "1FileAbcdefGhij", "name": "Report (final).pdf", "mimeType": "application/pdf",
     "size": "1234", "modifiedTime": "2026-09-01T10:00:00Z"},
    {"id": "1DocAbcdefGhij", "name": "Plan", "mimeType": DOC_MIME,
     "modifiedTime": "2026-08-15T08:00:00Z"},
    {"id": "1DirAbcdefGhij", "name": "Archive", "mimeType": FOLDER_MIME,
     "modifiedTime": "2026-07-01T00:00:00Z"},
]

_ENTRY_LINES = (
    "  [file] Report (final).pdf (1.2 KB)\n"
    "    ID: 1FileAbcdefGhij  |  Type: application/pdf  |  Modified: 2026-09-01\n"
    "  [file] Plan\n"
    f"    ID: 1DocAbcdefGhij  |  Type: {DOC_MIME}  |  Modified: 2026-08-15\n"
    "  [folder] Archive\n"
    f"    ID: 1DirAbcdefGhij  |  Type: {FOLDER_MIME}  |  Modified: 2026-07-01"
)


async def test_list_output_format_unchanged(drive):
    drive.listing = _ENTRIES
    res = await drive.tool.execute("list", folder_id="1ParentAbcdefGh")
    assert res.success
    assert res.content == f"Files in 1ParentAbcdefGh (3 results):\n\n{_ENTRY_LINES}"


async def test_search_output_format_unchanged(drive):
    drive.listing = _ENTRIES
    res = await drive.tool.execute("search", query="report")
    assert res.success
    assert res.content == f"Search results for 'report' (3 found):\n\n{_ENTRY_LINES}"


async def test_scale_loop_parses_real_list_output(drive):
    from captain_claw.agent_scale_loop_mixin import AgentScaleLoopMixin

    drive.listing = _ENTRIES
    res = await drive.tool.execute("list", folder_id="1ParentAbcdefGh")
    parsed = AgentScaleLoopMixin._parse_gdrive_listing(res.content)
    assert parsed == {
        "Report (final).pdf": {"id": "1FileAbcdefGhij", "mimeType": "application/pdf"},
        "Plan": {"id": "1DocAbcdefGhij", "mimeType": DOC_MIME},
    }


def test_description_and_enum_list_download():
    assert "download" in GoogleDriveTool.parameters["properties"]["action"]["enum"]
    assert "output_path" in GoogleDriveTool.parameters["properties"]
    assert "page_token" in GoogleDriveTool.parameters["properties"]
    assert "gws" not in GoogleDriveTool.description
    assert "URL" in GoogleDriveTool.description
