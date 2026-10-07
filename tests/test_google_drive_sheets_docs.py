"""google_drive edits a Google Sheet / Doc IN PLACE — same file id, no re-upload.

sheet_read / sheet_update / sheet_append / sheet_clear go through the Sheets
API v4, doc_read / doc_replace_text / doc_append_text / doc_insert_text
through the Docs API v1, with the token google_drive already uses (the
``drive`` scope covers both). Pinned here: each action's endpoint, method,
query and body; nothing ever reaches /upload/drive/v3; read output carries
exact A1 addresses and exact Doc text; only native Sheets/Docs are edited
(an .xlsx/.docx on Drive gets the same-id route instead); a read-only
connection is refused before any request; the admin-facing messages for a
disabled API or a too-narrow scope; the whole-file ``update`` guard; and a
shared-agent member edits only with their OWN token (the turn's grant).

No network: Google is an ``httpx.MockTransport``; the token source is faked.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

import captain_claw.session as session_mod
from captain_claw.google_ids import google_drive_redirect
from captain_claw.google_oauth import GoogleOAuthTokens
from captain_claw.google_oauth_manager import GoogleOAuthManager
from captain_claw.tools import google_drive as gd
from captain_claw.tools.google_drive import GoogleDriveTool

FID = "1SheEtAbCdEfGhIjKlMnOpQrStUvWxYz_012345"
DOC = "1DocAbCdEfGhIjKlMnOpQrStUvWxYz_0123456"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"
DOC_MIME = "application/vnd.google-apps.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
FULL = "https://www.googleapis.com/auth/drive"
READONLY = "https://www.googleapis.com/auth/drive.readonly"
PER_FILE = "https://www.googleapis.com/auth/drive.file"

_REAL_GET_TOKENS = GoogleOAuthManager.get_tokens
_REAL_FD_BASE = GoogleOAuthManager.__dict__["_flight_deck_base"]
_REAL_ASYNC_CLIENT = httpx.AsyncClient


# ── fakes ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_real_session_manager(monkeypatch):
    monkeypatch.setattr(session_mod, "get_session_manager", lambda: object())


@pytest.fixture(autouse=True)
def scope(monkeypatch):
    """Standalone mode, Google connected; tests may narrow ``scope["scope"]``."""
    holder = {"scope": FULL}
    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", staticmethod(lambda: ""))

    async def _get_tokens(self):
        return GoogleOAuthTokens(
            access_token="DRIVE-TOKEN", refresh_token="",
            expires_at=time.time() + 3300, scope=holder["scope"],
        )

    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _get_tokens)
    return holder


class _FakeGoogle:
    """Just enough of Drive v3 (metadata, media update), Sheets v4 and Docs v1."""

    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.exports: dict[str, bytes] = {}
        self.sheet_props: dict[str, dict] = {}
        self.values: dict[tuple[str, str], dict] = {}
        self.batch_get: dict[str, dict] = {}
        self.docs: dict[str, dict] = {}
        self.doc_replies: list[dict] = []
        self.errors: dict[str, tuple[int, dict]] = {}  # host → injected error
        self.requests: list[httpx.Request] = []

    def add(self, fid: str, name: str, mime: str) -> None:
        self.files[fid] = {"id": fid, "name": name, "mimeType": mime}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if host in self.errors:
            status, body = self.errors[host]
            return httpx.Response(status, json=body)
        if host == "www.googleapis.com":
            if path.endswith("/export"):
                return httpx.Response(200, content=self.exports[path.split("/")[-2]])
            fid = path.rsplit("/", 1)[-1]
            if fid not in self.files:
                return httpx.Response(404, json={"error": {"message": "File not found"}})
            if request.method == "PATCH":
                return httpx.Response(200, json={"id": fid, "name": self.files[fid]["name"]})
            return httpx.Response(200, json=self.files[fid])
        body = json.loads(request.content) if request.content else None
        if host == "sheets.googleapis.com":
            fid, _, tail = path[len("/v4/spreadsheets/"):].partition("/")
            if tail == "":
                return httpx.Response(200, json=self.sheet_props[fid])
            if tail == "values:batchGet":
                return httpx.Response(200, json=self.batch_get[fid])
            if tail == "values:batchUpdate":
                return httpx.Response(200, json={"responses": [
                    {"updatedRange": d["range"], "updatedCells": sum(len(r) for r in d["values"])}
                    for d in body["data"]
                ]})
            rng = tail[len("values/"):]
            if rng.endswith(":append"):
                rows = body["values"]
                return httpx.Response(200, json={
                    "tableRange": "Log!A1:C4",
                    "updates": {"updatedRange": "Log!A5:C6", "updatedRows": len(rows),
                                "updatedCells": sum(len(r) for r in rows)},
                })
            if rng.endswith(":clear"):
                return httpx.Response(200, json={"clearedRange": rng[: -len(":clear")]})
            if request.method == "PUT":
                return httpx.Response(200, json={
                    "updatedRange": rng, "updatedCells": sum(len(r) for r in body["values"]),
                })
            return httpx.Response(200, json=self.values.get((fid, rng), {"range": rng}))
        if host == "docs.googleapis.com":
            rest = path[len("/v1/documents/"):]
            if rest.endswith(":batchUpdate"):
                return httpx.Response(200, json={"replies": self.doc_replies or [{}]})
            return httpx.Response(200, json=self.docs[rest])
        return httpx.Response(500, json={"error": {"message": f"unexpected {request.url}"}})

    def api(self, host: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == host]

    def body(self, request: httpx.Request) -> dict:
        return json.loads(request.content)


@pytest.fixture
async def g():
    fake = _FakeGoogle()
    tool = GoogleDriveTool()
    await tool._client.aclose()
    tool._client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    fake.tool = tool
    yield fake
    await tool.close()


def _no_upload(fake: _FakeGoogle) -> None:
    """In place means in place: never the upload API, never a media PATCH."""
    for r in fake.requests:
        assert "/upload/" not in r.url.path, r.url
        assert r.method != "PATCH", r.url


def _para(start: int, *runs: str) -> dict:
    """A Docs paragraph whose text runs start at *start* (UTF-16 indexes)."""
    elements, pos = [], start
    for text in runs:
        end = pos + sum(gd._utf16_len(c) for c in text)
        elements.append({"startIndex": pos, "endIndex": end, "textRun": {"content": text}})
        pos = end
    return {"startIndex": start, "endIndex": pos, "paragraph": {"elements": elements}}


def _doc(*paragraphs: dict, title: str = "Plan", tabs: list[dict] | None = None) -> dict:
    body = {"content": [{"endIndex": 1, "sectionBreak": {}}, *paragraphs]}
    return {
        "documentId": DOC, "title": title, "revisionId": "rev-7",
        "tabs": tabs or [{"tabProperties": {"tabId": "t.0", "title": "Tab 1"},
                          "documentTab": {"body": body}}],
    }


# ── sheet_read ───────────────────────────────────────────────────────


async def test_sheet_read_without_range_lists_tabs_and_previews_each(g):
    g.add(FID, "Budget", SHEET_MIME)
    g.sheet_props[FID] = {"sheets": [
        {"properties": {"sheetId": 0, "title": "Summary", "sheetType": "GRID",
                        "gridProperties": {"rowCount": 1000, "columnCount": 26}}},
        {"properties": {"sheetId": 1, "title": "Q2 plan", "sheetType": "GRID",
                        "gridProperties": {"rowCount": 50, "columnCount": 3}}},
        {"properties": {"sheetId": 2, "title": "Chart", "sheetType": "OBJECT"}},
    ]}
    g.batch_get[FID] = {"valueRanges": [
        {"range": "Summary!A1:B2", "values": [["Quarter", "Total"], ["Q1", "10"]]},
        {"range": "'Q2 plan'!A1:C1", "values": [["Item", "Cost", "Note"]]},
    ]}
    res = await g.tool.execute("sheet_read", file_id=FID)
    assert res.success, res.error
    assert f"[Google Sheet: Budget — file id {FID}]" in res.content
    assert "Tabs (3):" in res.content
    assert "  - Summary — 1000 rows × 26 columns (A–Z)" in res.content
    assert "## Tab: Summary (first 50 rows, values)" in res.content
    assert "| row | A | B |" in res.content and "| 2 | Q1 | 10 |" in res.content
    assert "| 1 | Item | Cost | Note |" in res.content
    batch = [r for r in g.api("sheets.googleapis.com") if r.url.path.endswith("values:batchGet")]
    assert len(batch) == 1
    # Chart tabs have no cells: only the grid tabs are read, names quoted.
    assert batch[0].url.params.get_list("ranges") == ["'Summary'!1:50", "'Q2 plan'!1:50"]
    assert batch[0].url.params["valueRenderOption"] == "FORMATTED_VALUE"
    _no_upload(g)


async def test_sheet_read_preview_never_reads_past_a_small_tabs_grid(g):
    # A CSV-imported tab can have fewer than 50 rows; '1:50' on it is a 400
    # "exceeds grid limits" that would fail the whole preview.
    g.add(FID, "Import", SHEET_MIME)
    g.sheet_props[FID] = {"sheets": [
        {"properties": {"sheetId": 0, "title": "Small", "sheetType": "GRID",
                        "gridProperties": {"rowCount": 12, "columnCount": 4}}},
        {"properties": {"sheetId": 1, "title": "NoSize", "sheetType": "GRID"}},
    ]}
    g.batch_get[FID] = {"valueRanges": [
        {"range": "Small!A1:B1", "values": [["a", "b"]]},
        {"range": "NoSize!A1:A1", "values": [["c"]]},
    ]}
    res = await g.tool.execute("sheet_read", file_id=FID)
    assert res.success, res.error
    batch = [r for r in g.api("sheets.googleapis.com") if r.url.path.endswith("values:batchGet")]
    assert batch[0].url.params.get_list("ranges") == ["'Small'!1:12", "'NoSize'!1:50"]


async def test_sheet_read_range_labels_rows_and_columns_from_the_range_start(g):
    g.add(FID, "Budget", SHEET_MIME)
    g.values[(FID, "Q2!C5:D7")] = {"range": "Q2!C5:D7", "values": [["a", "b|c"], [], ["line1\nline2"]]}
    res = await g.tool.execute("sheet_read", file_id=FID, range="Q2!C5:D7")
    assert res.success, res.error
    assert "| row | C | D |" in res.content
    assert "| 5 | a | b\\|c |" in res.content
    assert "| 6 |  |  |" in res.content  # an empty row keeps its number
    assert "| 7 | line1\\nline2 |  |" in res.content
    req = g.api("sheets.googleapis.com")[-1]
    assert req.method == "GET"
    assert req.url.path == f"/v4/spreadsheets/{FID}/values/Q2!C5:D7"
    assert b"/values/Q2%21C5%3AD7" in req.url.raw_path  # the range travels encoded
    assert req.url.params["valueRenderOption"] == "FORMATTED_VALUE"


async def test_sheet_read_formulas(g):
    g.add(FID, "Budget", SHEET_MIME)
    g.values[(FID, "A1:B2")] = {"range": "Sheet1!A1:B2", "values": [["Total", "=SUM(B3:B9)"]]}
    res = await g.tool.execute("sheet_read", file_id=FID, range="A1:B2", render="formulas")
    assert res.success, res.error
    assert "| 1 | Total | =SUM(B3:B9) |" in res.content
    assert g.api("sheets.googleapis.com")[-1].url.params["valueRenderOption"] == "FORMULA"

    res = await g.tool.execute("sheet_read", file_id=FID, render="html")
    assert not res.success and "render" in res.error


@pytest.mark.parametrize("ref", [
    f"https://docs.google.com/spreadsheets/d/{FID}/edit#gid=0",
    f"https://docs.google.com/spreadsheets/d/{FID}/edit?usp=sharing",
])
async def test_a_sheets_url_is_accepted_as_file_id(g, ref):
    g.add(FID, "Budget", SHEET_MIME)
    res = await g.tool.execute("sheet_update", file_id=ref, range="A1", values=[["x"]])
    assert res.success, res.error
    assert all(FID in r.url.path for r in g.requests)


# ── sheet_update / sheet_append / sheet_clear ────────────────────────


async def test_sheet_update_one_range_puts_values_user_entered(g):
    g.add(FID, "Budget", SHEET_MIME)
    rows = [["Rent", "1200"], ["Travel", "=B2*2"]]
    res = await g.tool.execute("sheet_update", file_id=FID, range="Q2!B2:C3", values=rows)
    assert res.success, res.error
    req = g.api("sheets.googleapis.com")[-1]
    assert req.method == "PUT"
    assert req.url.path == f"/v4/spreadsheets/{FID}/values/Q2!B2:C3"
    assert req.url.params["valueInputOption"] == "USER_ENTERED"
    assert g.body(req) == {"range": "Q2!B2:C3", "majorDimension": "ROWS", "values": rows}
    assert req.headers["Authorization"] == "Bearer DRIVE-TOKEN"
    assert f"Updated 'Budget' in place (Google Sheet, file id {FID})" in res.content
    assert "Q2!B2:C3 — 4 cell(s)" in res.content
    _no_upload(g)


async def test_sheet_update_several_ranges_batch_raw(g):
    g.add(FID, "Budget", SHEET_MIME)
    updates = [
        {"range": "Q2!B2", "values": [["1200"]]},
        {"range": "Q3!A1:B1", "values": [["Item", "Cost"]]},
    ]
    res = await g.tool.execute("sheet_update", file_id=FID, updates=updates, value_input="raw")
    assert res.success, res.error
    req = g.api("sheets.googleapis.com")[-1]
    assert req.method == "POST"
    assert req.url.path == f"/v4/spreadsheets/{FID}/values:batchUpdate"
    assert g.body(req) == {"valueInputOption": "RAW", "data": [
        {"range": "Q2!B2", "majorDimension": "ROWS", "values": [["1200"]]},
        {"range": "Q3!A1:B1", "majorDimension": "ROWS", "values": [["Item", "Cost"]]},
    ]}
    assert "Q2!B2 — 1 cell(s)" in res.content and "Q3!A1:B1 — 2 cell(s)" in res.content
    assert "3 cell(s) stored as given" in res.content


@pytest.mark.parametrize("values,expected", [
    (["a", 2], [["a", 2]]),          # one row sent bare
    ("Done", [["Done"]]),            # a single value
    (42, [[42]]),
    ('[["x", "y"]]', [["x", "y"]]),  # rows as a JSON string
])
async def test_sheet_update_takes_the_shapes_models_send(g, values, expected):
    g.add(FID, "Budget", SHEET_MIME)
    res = await g.tool.execute("sheet_update", file_id=FID, range="A1", values=values)
    assert res.success, res.error
    assert g.body(g.api("sheets.googleapis.com")[-1])["values"] == expected


@pytest.mark.parametrize("args,needle", [
    ({"range": "A1", "values": [["ok"], "x"]}, "list of rows"),
    ({"range": "A1", "values": [["ok", {"a": 1}]]}, "each cell"),
    ({"range": "A1", "values": [{"a": 1}]}, "each cell"),
    ({"range": "A1", "values": "[[1,"}, "not valid JSON"),
    ({"values": [["x"]]}, "range is required"),
    ({}, "sheet_update needs range + values"),
    ({"range": "A1", "values": [["x"]], "value_input": "html"}, "value_input"),
    ({"updates": [{"values": [["x"]]}]}, "needs a range"),
])
async def test_sheet_update_bad_arguments_make_no_request(g, args, needle):
    g.add(FID, "Budget", SHEET_MIME)
    res = await g.tool.execute("sheet_update", file_id=FID, **args)
    assert not res.success and needle in res.error
    assert g.requests == []


async def test_sheet_append_inserts_rows_below_the_table(g):
    g.add(FID, "Log", SHEET_MIME)
    rows = [["2026-10-06", "deploy", "ok"], ["2026-10-06", "test", "ok"]]
    res = await g.tool.execute("sheet_append", file_id=FID, range="Log!A:C", values=rows)
    assert res.success, res.error
    req = g.api("sheets.googleapis.com")[-1]
    assert req.method == "POST"
    assert req.url.path == f"/v4/spreadsheets/{FID}/values/Log!A:C:append"
    assert req.url.params["valueInputOption"] == "USER_ENTERED"
    assert req.url.params["insertDataOption"] == "INSERT_ROWS"
    assert g.body(req) == {"range": "Log!A:C", "majorDimension": "ROWS", "values": rows}
    assert "Appended 2 row(s) to 'Log' in place" in res.content
    assert "Log!A5:C6 — 6 cell(s)" in res.content
    _no_upload(g)

    res = await g.tool.execute("sheet_append", file_id=FID, values=rows)
    assert not res.success and "range is required" in res.error


async def test_sheet_clear(g):
    g.add(FID, "Budget", SHEET_MIME)
    res = await g.tool.execute("sheet_clear", file_id=FID, range="'Q3 plan'!B2:D9")
    assert res.success, res.error
    req = g.api("sheets.googleapis.com")[-1]
    assert req.method == "POST"
    assert req.url.path == f"/v4/spreadsheets/{FID}/values/'Q3 plan'!B2:D9:clear"
    assert b"%27Q3%20plan%27%21B2%3AD9:clear" in req.url.raw_path
    assert "Cleared 'Q3 plan'!B2:D9 in 'Budget' in place" in res.content
    _no_upload(g)


# ── doc_read ─────────────────────────────────────────────────────────


async def test_doc_read_shows_exact_text_per_tab(g):
    g.add(DOC, "Plan", DOC_MIME)
    tab1 = {"content": [
        {"endIndex": 1, "sectionBreak": {}},
        _para(1, "Price: $5 *net* ", "(approx.) _v2_\n"),
        {"startIndex": 32, "endIndex": 33, "paragraph": {"elements": [
            {"startIndex": 32, "endIndex": 33, "inlineObjectElement": {"inlineObjectId": "i1"}},
        ]}},
        {"startIndex": 33, "endIndex": 50, "table": {"tableRows": [{"tableCells": [
            {"content": [_para(35, "a\n")]}, {"content": [_para(38, "b\n")]},
        ]}]}},
    ]}
    tab2 = {"content": [{"endIndex": 1, "sectionBreak": {}}, _para(1, "Second\x0bline\n")]}
    g.docs[DOC] = _doc(tabs=[
        {"tabProperties": {"tabId": "t.0", "title": "Main"}, "documentTab": {"body": tab1}},
        {"tabProperties": {"tabId": "t.1", "title": "Notes"}, "documentTab": {"body": tab2}},
    ])
    res = await g.tool.execute("doc_read", file_id=f"https://docs.google.com/document/d/{DOC}/edit")
    assert res.success, res.error
    # No markdown escaping: what is shown is what replaceAllText matches.
    assert "Price: $5 *net* (approx.) _v2_\n" in res.content
    assert "[image]" in res.content and "| a | b |" in res.content
    assert "## Tab: Main (tab id t.0)" in res.content and "## Tab: Notes (tab id t.1)" in res.content
    assert "Second\nline" in res.content
    req = g.api("docs.googleapis.com")[-1]
    assert req.url.path == f"/v1/documents/{DOC}" and req.url.params["includeTabsContent"] == "true"


# ── doc_replace_text ─────────────────────────────────────────────────


async def test_doc_replace_text_sends_replace_all_text(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.doc_replies = [{"replaceAllText": {"occurrencesChanged": 2}}]
    res = await g.tool.execute(
        "doc_replace_text", file_id=DOC, find="Q3 target", replace_with="Q4 target",
    )
    assert res.success, res.error
    req = g.api("docs.googleapis.com")[-1]
    assert req.method == "POST" and req.url.path == f"/v1/documents/{DOC}:batchUpdate"
    assert g.body(req) == {"requests": [{"replaceAllText": {
        "containsText": {"text": "Q3 target", "matchCase": True},
        "replaceText": "Q4 target",
    }}]}
    assert "Replaced 2 occurrence(s) of 'Q3 target' with 'Q4 target'" in res.content
    _no_upload(g)


async def test_doc_replace_text_case_insensitive_in_one_tab(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.docs[DOC] = _doc(tabs=[
        {"tabProperties": {"tabId": "t.0", "title": "Main"}, "documentTab": {"body": {"content": []}}},
        {"tabProperties": {"tabId": "t.9", "title": "Notes"}, "documentTab": {"body": {"content": []}}},
    ])
    g.doc_replies = [{"replaceAllText": {"occurrencesChanged": 1}}]
    res = await g.tool.execute(
        "doc_replace_text", file_id=DOC, find="draft", replace_with="", match_case="false",
        tab="notes",
    )
    assert res.success, res.error
    request = g.body(g.api("docs.googleapis.com")[-1])["requests"][0]["replaceAllText"]
    assert request["containsText"] == {"text": "draft", "matchCase": False}
    assert request["replaceText"] == ""
    assert request["tabsCriteria"] == {"tabIds": ["t.9"]}


async def test_doc_replace_text_with_no_occurrence_fails(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.doc_replies = [{"replaceAllText": {}}]  # Docs omits occurrencesChanged when 0
    res = await g.tool.execute("doc_replace_text", file_id=DOC, find="Q3  target", replace_with="x")
    assert not res.success
    assert "nothing changed" in res.error and "doc_read" in res.error


async def test_doc_replace_text_needs_find_and_replace_with(g):
    g.add(DOC, "Plan", DOC_MIME)
    res = await g.tool.execute("doc_replace_text", file_id=DOC, replace_with="x")
    assert not res.success and "find is required" in res.error
    res = await g.tool.execute("doc_replace_text", file_id=DOC, find="x")
    assert not res.success and "replace_with is required" in res.error
    assert g.requests == []


# ── doc_append_text / doc_insert_text ────────────────────────────────


async def test_doc_append_text_opens_a_new_last_paragraph(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.docs[DOC] = _doc(_para(1, "Hello\n"))
    res = await g.tool.execute("doc_append_text", file_id=DOC, text="World")
    assert res.success, res.error
    req = g.api("docs.googleapis.com")[-1]
    assert req.url.path == f"/v1/documents/{DOC}:batchUpdate"
    assert g.body(req) == {
        "requests": [{"insertText": {
            "endOfSegmentLocation": {"segmentId": "", "tabId": "t.0"}, "text": "\nWorld",
        }}],
        "writeControl": {"requiredRevisionId": "rev-7"},
    }
    assert "Appended 5 characters to the end of 'Plan'" in res.content
    _no_upload(g)


async def test_doc_append_text_fills_an_empty_last_paragraph(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.docs[DOC] = _doc(_para(1, "Hello\n"), _para(7, "\n"))
    res = await g.tool.execute("doc_append_text", file_id=DOC, text="World\n")
    assert res.success, res.error
    insert = g.body(g.api("docs.googleapis.com")[-1])["requests"][0]["insertText"]
    assert insert["text"] == "World"


async def test_doc_insert_text_goes_right_after_the_anchor(g):
    g.add(DOC, "Plan", DOC_MIME)
    # "Intro\n" holds indexes 1-6; the emoji counts two UTF-16 units.
    g.docs[DOC] = _doc(_para(1, "Intro\n"), _para(7, "Results ", "😀 here\n"), _para(23, "End\n"))
    res = await g.tool.execute("doc_insert_text", file_id=DOC, after="😀 here", text=" and now")
    assert res.success, res.error
    req = g.api("docs.googleapis.com")[-1]
    assert g.body(req) == {
        "requests": [{"insertText": {
            "location": {"segmentId": "", "tabId": "t.0", "index": 22}, "text": " and now",
        }}],
        "writeControl": {"requiredRevisionId": "rev-7"},
    }
    _no_upload(g)


async def test_doc_insert_text_after_the_last_paragraph_uses_the_end_of_the_body(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.docs[DOC] = _doc(_para(1, "Intro\n"), _para(7, "Last line\n"))
    res = await g.tool.execute("doc_insert_text", file_id=DOC, after="Last line\n", text="New one\n")
    assert res.success, res.error
    insert = g.body(g.api("docs.googleapis.com")[-1])["requests"][0]["insertText"]
    assert insert == {"endOfSegmentLocation": {"segmentId": "", "tabId": "t.0"}, "text": "\nNew one"}


@pytest.mark.parametrize("after,match_case,needle", [
    ("Missing", None, "was not found"),
    ("o", None, "occurs"),
    ("intro", None, "was not found"),  # case-sensitive by default
])
async def test_doc_insert_text_needs_one_exact_anchor(g, after, match_case, needle):
    g.add(DOC, "Plan", DOC_MIME)
    g.docs[DOC] = _doc(_para(1, "Intro\n"), _para(7, "Body text\n"))
    res = await g.tool.execute(
        "doc_insert_text", file_id=DOC, after=after, text="x", match_case=match_case,
    )
    assert not res.success and needle in res.error
    assert not any(r.url.path.endswith(":batchUpdate") for r in g.requests)


async def test_doc_insert_text_case_insensitive(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.docs[DOC] = _doc(_para(1, "Intro\n"), _para(7, "Body text\n"))
    res = await g.tool.execute("doc_insert_text", file_id=DOC, after="intro", text="!", match_case=False)
    assert res.success, res.error
    location = g.body(g.api("docs.googleapis.com")[-1])["requests"][0]["insertText"]["location"]
    assert location["index"] == 6


# ── only native Sheets / Docs ────────────────────────────────────────


@pytest.mark.parametrize("action,args,mime,needle", [
    ("sheet_update", {"range": "A1", "values": [["x"]]}, XLSX, "download it, edit the local copy"),
    ("sheet_read", {}, "text/csv", "not a native Google Sheet"),
    ("doc_replace_text", {"find": "a", "replace_with": "b"}, DOCX, "download it, edit the local copy"),
    ("sheet_append", {"range": "A1", "values": [["x"]]}, DOC_MIME, "is a Google Doc, not a Google Sheet: use doc_read"),
    ("doc_append_text", {"text": "x"}, SHEET_MIME, "is a Google Sheet, not a Google Doc: use sheet_read"),
    ("doc_read", {}, "application/vnd.google-apps.presentation", "work on Google Docs only"),
])
async def test_only_native_sheets_and_docs_are_edited_in_place(g, action, args, mime, needle):
    g.add(FID, "Thing", mime)
    res = await g.tool.execute(action, file_id=FID, **args)
    assert not res.success and needle in res.error
    if mime in (XLSX, DOCX):
        assert f"google_drive(action='update', file_id='{FID}', local_path=" in res.error
    # Only the Drive metadata lookup ran — no Sheets/Docs request.
    assert {r.url.host for r in g.requests} == {"www.googleapis.com"}


# ── scopes and admin-facing errors ───────────────────────────────────


@pytest.mark.parametrize("action,args", [
    ("sheet_update", {"range": "A1", "values": [["x"]]}),
    ("sheet_append", {"range": "A1", "values": [["x"]]}),
    ("sheet_clear", {"range": "A1"}),
    ("doc_replace_text", {"find": "a", "replace_with": "b"}),
    ("doc_append_text", {"text": "x"}),
    ("doc_insert_text", {"text": "x", "after": "a"}),
])
async def test_read_only_connection_refused_before_any_request(g, scope, action, args):
    scope["scope"] = f"openid {READONLY}"
    g.add(FID, "Budget", SHEET_MIME)
    res = await g.tool.execute(action, file_id=FID, **args)
    assert not res.success
    assert res.error == gd._FULL_DRIVE_NEEDED
    assert g.requests == []


async def test_read_only_connection_still_reads(g, scope):
    scope["scope"] = READONLY
    g.add(FID, "Budget", SHEET_MIME)
    g.values[(FID, "A1")] = {"range": "Sheet1!A1", "values": [["x"]]}
    res = await g.tool.execute("sheet_read", file_id=FID, range="A1")
    assert res.success, res.error


async def test_per_file_scope_is_allowed_but_a_404_names_the_scope(g, scope):
    scope["scope"] = PER_FILE
    g.add(FID, "Made by this app", SHEET_MIME)
    res = await g.tool.execute("sheet_update", file_id=FID, range="A1", values=[["x"]])
    assert res.success, res.error  # drive.file reaches files this app created

    res = await g.tool.execute("sheet_update", file_id="1SomeoneElsesSheet0123", range="A1", values=[["x"]])
    assert not res.success and res.error == gd._FULL_DRIVE_NEEDED

    scope["scope"] = FULL
    res = await g.tool.execute("doc_read", file_id="1SomeoneElsesSheet0123")
    assert not res.success and res.error.startswith("Google Doc not found")


@pytest.mark.parametrize("host,action,args,api", [
    ("sheets.googleapis.com", "sheet_update", {"range": "A1", "values": [["x"]]}, "Google Sheets API"),
    ("docs.googleapis.com", "doc_read", {}, "Google Docs API"),
])
async def test_disabled_api_tells_the_admin_to_enable_it(g, host, action, args, api):
    g.add(FID, "Thing", SHEET_MIME if "sheet" in action else DOC_MIME)
    g.errors[host] = (403, {"error": {
        "code": 403, "status": "PERMISSION_DENIED",
        "message": f"{api} has not been used in project 123 before or it is disabled.",
        "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "SERVICE_DISABLED"}],
    }})
    res = await g.tool.execute(action, file_id=FID, **args)
    assert not res.success
    assert f"Enable the {api} in the deck's Google Cloud project" in res.error


async def test_insufficient_scope_403_names_full_drive_access(g):
    g.add(FID, "Budget", SHEET_MIME)
    g.errors["sheets.googleapis.com"] = (403, {"error": {
        "code": 403, "message": "Request had insufficient authentication scopes.",
        "details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}],
    }})
    res = await g.tool.execute("sheet_clear", file_id=FID, range="A1")
    assert not res.success and res.error == gd._FULL_DRIVE_NEEDED


async def test_no_edit_access_is_reported_as_such(g):
    g.add(FID, "Budget", SHEET_MIME)
    g.errors["sheets.googleapis.com"] = (403, {"error": {
        "code": 403, "status": "PERMISSION_DENIED", "message": "The caller does not have permission",
    }})
    res = await g.tool.execute("sheet_update", file_id=FID, range="A1", values=[["x"]])
    assert not res.success
    assert res.error == (
        "Permission denied: The caller does not have permission. The connected "
        "Google account needs edit access to this Google Sheet."
    )


async def test_a_bad_range_points_at_a1_notation(g):
    g.add(FID, "Budget", SHEET_MIME)
    g.errors["sheets.googleapis.com"] = (400, {"error": {"message": "Unable to parse range: Nope!A1"}})
    res = await g.tool.execute("sheet_read", file_id=FID, range="Nope!A1")
    assert not res.success
    assert "Unable to parse range" in res.error and "sheet_read without a range lists the tabs" in res.error


# ── whole-file update guard, download/read hints ─────────────────────


@pytest.mark.parametrize("mime,points_at", [
    (SHEET_MIME, "sheet_update / sheet_append / sheet_clear"),
    (DOC_MIME, "doc_replace_text / doc_append_text / doc_insert_text"),
])
async def test_update_refuses_to_replace_a_google_sheet_or_doc(g, mime, points_at):
    g.add(FID, "Budget", mime)
    res = await g.tool.execute("update", file_id=FID, content="a,b\n1,2\n")
    assert not res.success
    assert "WHOLE content" in res.error and points_at in res.error and "overwrite=true" in res.error
    assert not any(r.method == "PATCH" for r in g.requests)

    res = await g.tool.execute("update", file_id=FID, content="a,b\n1,2\n", overwrite=True)
    assert res.success, res.error
    assert [r.method for r in g.requests].count("PATCH") == 1


async def test_update_of_an_ordinary_file_needs_no_overwrite(g, tmp_path):
    g.add(FID, "budget.xlsx", XLSX)
    edited = tmp_path / "budget.xlsx"
    edited.write_bytes(b"PK edited")
    res = await g.tool.execute("update", file_id=FID, local_path=str(edited))
    assert res.success, res.error
    patch = [r for r in g.requests if r.method == "PATCH"]
    assert len(patch) == 1 and patch[0].url.path == f"/upload/drive/v3/files/{FID}"


async def test_download_prints_the_file_id_and_the_in_place_route(g, tmp_path):
    g.add(FID, "Budget", SHEET_MIME)
    g.exports[FID] = b"PK xlsx"
    res = await g.tool.execute(
        "download", file_id=FID,
        _runtime_base_path=tmp_path, _saved_base_path=tmp_path / "saved", _session_id="s1",
    )
    assert res.success, res.error
    assert f"File ID: {FID}" in res.content
    assert "sheet_update / sheet_append / sheet_clear" in res.content
    assert "do not upload a modified copy" in res.content


async def test_read_of_a_doc_points_at_the_in_place_edits(g):
    g.add(DOC, "Plan", DOC_MIME)
    g.exports[DOC] = b"# Plan\n\nBody."
    res = await g.tool.execute("read", file_id=DOC)
    assert res.success, res.error
    assert "# Plan" in res.content
    assert res.content.endswith(
        f"[To change this Doc, edit it in place: doc_read(file_id='{DOC}') shows the "
        "exact text; doc_replace_text / doc_append_text / doc_insert_text change it. "
        "Never upload a modified copy.]"
    )


def test_redirect_for_sheet_and_doc_links_names_the_in_place_actions():
    sheet = google_drive_redirect(f"https://docs.google.com/spreadsheets/d/{FID}/edit", "Blocked.")
    assert f"google_drive(action='sheet_read', file_id='{FID}')" in sheet
    assert "sheet_update" in sheet and "doc_read" not in sheet
    doc = google_drive_redirect(f"https://docs.google.com/document/d/{DOC}/edit", "Blocked.")
    assert f"google_drive(action='doc_read', file_id='{DOC}')" in doc
    assert "doc_replace_text" in doc and "sheet_read" not in doc
    other = google_drive_redirect(f"https://drive.google.com/file/d/{FID}/view", "Blocked.")
    assert "sheet_read" not in other and "doc_read" not in other


def test_drive_mount_placeholder_allows_in_place_edits_of_sheets_and_docs():
    from captain_claw.drive_client import DriveFile
    from captain_claw.vfs_drive import placeholder_text

    sheet = placeholder_text(DriveFile(id=FID, name="Budget", mime_type=SHEET_MIME))
    assert f"sheet_update / sheet_append / sheet_clear, file_id='{FID}'" in sheet
    pdf = placeholder_text(DriveFile(id=FID, name="r.pdf", mime_type="application/pdf"))
    assert "sheet_update" not in pdf and "doc_replace_text" not in pdf


@pytest.mark.parametrize("micro", [False, True])
def test_google_section_steers_to_in_place_edits(tmp_path, micro):
    from captain_claw.instructions import InstructionLoader

    loader = InstructionLoader(
        base_dir=Path(gd.__file__).resolve().parents[1] / "instructions",
        personal_dir=tmp_path, use_micro=micro, use_nano=False,
    )
    section = loader.load("section_google.md")
    for action in ("sheet_update", "sheet_append", "sheet_clear",
                   "doc_replace_text", "doc_append_text", "doc_insert_text"):
        assert action in section, action
    assert "IN PLACE" in section


def test_new_parameters_are_described_and_not_path_like():
    import re

    props = GoogleDriveTool.parameters["properties"]
    pathy = re.compile(r"(path|file|dir|folder|root|glob|pattern|dest|output|local|^to$)")
    new = ("overwrite", "range", "values", "updates", "value_input", "render",
           "find", "replace_with", "match_case", "text", "after", "tab")
    for name in new:
        assert props[name].get("description"), name
        assert not pathy.search(name), name


# ── a shared-agent member edits as their OWN Google ──────────────────


GRANT = "G" * 40 + "_-1"


@pytest.fixture
def member_fd(monkeypatch):
    """Flight Deck client mode with the real token path; FD answers the
    member's token only to a request carrying the grant."""
    import captain_claw.google_oauth_manager as gom
    from captain_claw import speaker
    from captain_claw.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.google_oauth, "flight_deck_url", "")
    monkeypatch.setattr(cfg.google_oauth, "flight_deck_secret", "deck-secret")
    monkeypatch.setattr(cfg.web, "auth_token", "web-auth-of-this-agent")
    monkeypatch.setenv("FD_URL", "http://fd.test")
    monkeypatch.setattr(GoogleOAuthManager, "get_tokens", _REAL_GET_TOKENS)
    monkeypatch.setattr(GoogleOAuthManager, "_flight_deck_base", _REAL_FD_BASE)
    monkeypatch.setattr(gom, "_GOOGLE_CONNECTED", {})
    monkeypatch.setattr(speaker, "_IN_FLIGHT", 0)
    monkeypatch.setattr(speaker, "_MEMBER_GOOGLE", {})

    fd_requests: list[httpx.Request] = []

    def fd(request: httpx.Request) -> httpx.Response:
        fd_requests.append(request)
        member = speaker.GRANT_HEADER in request.headers
        return httpx.Response(200, json={
            "access_token": "tok-member" if member else "tok-owner",
            "expires_at": time.time() + 3600, "scope": FULL,
        })

    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(fd), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return fd_requests


def _bound(grant: str):
    from contextlib import contextmanager

    from captain_claw import speaker

    @contextmanager
    def _cm():
        p = speaker.Principal("u-member", "Ana", "Olga", "A", "process:helper:0123456789abcdef")
        t1, t2 = speaker.bind(p), speaker.bind_grant(grant)
        try:
            yield
        finally:
            speaker.reset_grant(t2)
            speaker.reset(t1)

    return _cm()


async def test_member_sheet_update_uses_the_grant_and_never_the_owner_token(g, member_fd):
    from captain_claw import speaker

    g.add(FID, "Ana's sheet", SHEET_MIME)
    with _bound(GRANT):
        res = await g.tool.execute("sheet_update", file_id=FID, range="A1", values=[["x"]])
    assert res.success, res.error
    assert member_fd and all(r.url.path == "/fd/google/access_token" for r in member_fd)
    for req in member_fd:
        assert req.headers[speaker.GRANT_HEADER] == GRANT
        assert req.url.params["fd_member"] == "1"
    assert g.requests
    for req in g.requests:
        assert req.headers["Authorization"] == "Bearer tok-member", req.url
    _no_upload(g)

    # The owner's own turn (nothing bound) gets the owner's token.
    g.requests.clear()
    res = await g.tool.execute("sheet_update", file_id=FID, range="A1", values=[["y"]])
    assert res.success, res.error
    assert {r.headers["Authorization"] for r in g.requests} == {"Bearer tok-owner"}


async def test_member_without_a_grant_makes_no_google_request(g, member_fd):
    g.add(FID, "Owner's sheet", SHEET_MIME)
    with _bound(""):
        res = await g.tool.execute("sheet_update", file_id=FID, range="A1", values=[["x"]])
    assert not res.success
    assert member_fd == [] and g.requests == []
