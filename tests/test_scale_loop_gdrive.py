"""Scale micro-loop: Drive-listed items are read through ``google_drive``.

The loop scans the session for ``google_drive`` list/search results, parses
their text output into a name → ``{id, mimeType}`` map, and routes any item
that names a listed file to ``google_drive read`` (which exports Google
Docs/Sheets/Slides and extracts PDF/DOCX/XLSX/PPTX itself). Nothing routes to
the retired ``gws`` tool any more.

Unit-level: a bare subclass of the mixin, no agent, no LLM, no network.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import httpx
import pytest

from captain_claw import agent_scale_loop_mixin
from captain_claw.agent_scale_loop_mixin import AgentScaleLoopMixin

FOLDER = "application/vnd.google-apps.folder"
DOC = "application/vnd.google-apps.document"
SHEET = "application/vnd.google-apps.spreadsheet"
PDF = "application/pdf"

# What google_drive list prints (google_drive.py _action_list).
LIST_OUTPUT = (
    "Files in root (6 results):\n"
    "\n"
    "  [folder] Archive\n"
    f"    ID: fold1  |  Type: {FOLDER}  |  Modified: 2026-09-01\n"
    "  [file] report.pdf (1.2 MB)\n"
    f"    ID: pdf1  |  Type: {PDF}  |  Modified: 2026-09-02\n"
    "  [file] Q3 Budget\n"
    f"    ID: sheet1  |  Type: {SHEET}  |  Modified: 2026-09-03\n"
    "  [file] Minutes (draft) (512 B)\n"
    "    ID: txt1  |  Type: text/plain  |  Modified: 2026-09-04\n"
    "  [file] Plan (v2)\n"
    f"    ID: doc1  |  Type: {DOC}  |  Modified: 2026-09-05\n"
    "  [file] deck.pptx (12.0 KB)\n"
    "    ID: ppt1  |  Type: application/vnd.openxmlformats-officedocument."
    "presentationml.presentation  |  Modified: "
)

# What google_drive search prints (google_drive.py _action_search).
SEARCH_OUTPUT = (
    "Search results for 'contract' (2 found):\n"
    "\n"
    "  [file] contract.docx (3.4 GB)\n"
    "    ID: docx1  |  Type: application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document  |  Modified: 2026-08-30\n"
    "  [folder] Contracts (old)\n"
    f"    ID: fold2  |  Type: {FOLDER}  |  Modified: 2026-08-29"
)


@pytest.fixture(autouse=True)
def _empty_cwd(tmp_path, monkeypatch):
    # A real local file keeps its local extractor, so run every case from an
    # empty directory: bare names like "data" must not hit the repo's own files.
    monkeypatch.chdir(tmp_path)


class _Loop(AgentScaleLoopMixin):
    def __init__(self, *, tools=("google_drive",), messages=()):
        self.tools = SimpleNamespace(has_tool=lambda name: name in tools)
        self.session = SimpleNamespace(messages=list(messages))


def _tool_msg(tool_name: str, action: str, content: str) -> dict:
    return {"role": "tool", "tool_name": tool_name,
            "tool_arguments": {"action": action}, "content": content}


# ── parser ───────────────────────────────────────────────────────────

class TestParseListing:
    def test_list_output(self):
        m = AgentScaleLoopMixin._parse_gdrive_listing(LIST_OUTPUT)
        assert m == {
            "report.pdf": {"id": "pdf1", "mimeType": PDF},
            "Q3 Budget": {"id": "sheet1", "mimeType": SHEET},
            "Minutes (draft)": {"id": "txt1", "mimeType": "text/plain"},
            "Plan (v2)": {"id": "doc1", "mimeType": DOC},
            "deck.pptx": {
                "id": "ppt1",
                "mimeType": "application/vnd.openxmlformats-officedocument."
                            "presentationml.presentation",
            },
        }

    def test_search_output_with_gb_size_and_folders_skipped(self):
        m = AgentScaleLoopMixin._parse_gdrive_listing(SEARCH_OUTPUT)
        assert list(m) == ["contract.docx"]
        assert m["contract.docx"]["id"] == "docx1"

    def test_folders_are_never_items(self):
        m = AgentScaleLoopMixin._parse_gdrive_listing(LIST_OUTPUT + "\n" + SEARCH_OUTPUT)
        assert "Archive" not in m and "Contracts (old)" not in m
        assert all(v["mimeType"] != FOLDER for v in m.values())

    @pytest.mark.parametrize("text", [
        "No files found in root folder.",
        "No files found matching 'x'.",
        '{"files": [{"id": "x", "name": "y.pdf", "mimeType": "application/pdf"}]}',
        "  [file] cut off by truncation (1.0 KB)",          # ID line lost
        "",
    ])
    def test_nothing_to_parse(self, text):
        assert AgentScaleLoopMixin._parse_gdrive_listing(text) == {}


async def test_parser_reads_the_real_tool_output():
    """Contract: parse what google_drive list/search actually print."""
    from captain_claw.tools.google_drive import GoogleDriveTool

    files = [
        {"id": "fold1", "name": "Archive", "mimeType": FOLDER, "modifiedTime": "2026-09-01T00:00:00Z"},
        {"id": "pdf1", "name": "report (final).pdf", "mimeType": PDF, "size": "1258291",
         "modifiedTime": "2026-09-02T00:00:00Z"},
        {"id": "doc1", "name": "Plan (v2)", "mimeType": DOC},
        {"id": "b1", "name": "tiny.bin", "mimeType": "application/octet-stream", "size": "7"},
    ]

    class _HTTP:
        async def get(self, url, params=None, headers=None):
            return httpx.Response(200, json={"files": files},
                                  request=httpx.Request("GET", url))

    tool = GoogleDriveTool()
    await tool._client.aclose()
    tool._client = _HTTP()
    listed = await tool._action_list("tok", folder_id="root")
    searched = await tool._action_search("tok", query="report")
    for res in (listed, searched):
        assert res.success, res.error
        assert AgentScaleLoopMixin._parse_gdrive_listing(res.content) == {
            "report (final).pdf": {"id": "pdf1", "mimeType": PDF},
            "Plan (v2)": {"id": "doc1", "mimeType": DOC},
            "tiny.bin": {"id": "b1", "mimeType": "application/octet-stream"},
        }


# ── session scan ─────────────────────────────────────────────────────

class TestBuildFileMap:
    def test_collects_google_drive_list_and_search_results(self):
        loop = _Loop(messages=[
            {"role": "user", "content": "summarise every file in my Drive folder"},
            _tool_msg("google_drive", "list", LIST_OUTPUT),
            _tool_msg("google_drive", "search", SEARCH_OUTPUT),
        ])
        m = loop._build_gdrive_file_map()
        assert m["report.pdf"]["id"] == "pdf1" and m["contract.docx"]["id"] == "docx1"
        assert "Archive" not in m

    def test_gated_on_google_drive_being_registered(self):
        loop = _Loop(tools=(), messages=[_tool_msg("google_drive", "list", LIST_OUTPUT)])
        assert loop._build_gdrive_file_map() == {}

    def test_ignores_retired_gws_results_and_other_actions(self):
        gws_json = '{"files": [{"id": "x", "name": "report.pdf", "mimeType": "application/pdf"}]}'
        loop = _Loop(tools=("google_drive", "gws"), messages=[
            _tool_msg("gws", "drive_list", gws_json),
            _tool_msg("gws", "drive_list", LIST_OUTPUT),
            _tool_msg("google_drive", "read", LIST_OUTPUT),
            {"role": "assistant", "tool_name": "google_drive", "content": LIST_OUTPUT},
            {"role": "tool", "tool_name": "google_drive", "tool_arguments": None,
             "content": LIST_OUTPUT},
        ])
        assert loop._build_gdrive_file_map() == {}

    def test_no_session(self):
        loop = _Loop()
        loop.session = None
        assert loop._build_gdrive_file_map() == {}


# ── routing ──────────────────────────────────────────────────────────

class TestRouting:
    gmap = AgentScaleLoopMixin._parse_gdrive_listing(LIST_OUTPUT)

    @pytest.mark.parametrize("item, file_id", [
        ("report.pdf", "pdf1"),          # would have been pdf_extract
        ("deck.pptx", "ppt1"),           # would have been pptx_extract
        ("Q3 Budget", "sheet1"),         # would have been _passthrough
        ("q3 budget", "sheet1"),         # case-insensitive
        ("Plan (v2)", "doc1"),
    ])
    def test_listed_items_read_through_google_drive(self, item, file_id):
        loop = _Loop()
        tool_name, args = loop._detect_item_extractor(item)
        assert tool_name in AgentScaleLoopMixin._GDRIVE_OVERRIDABLE_EXTRACTORS
        assert loop._gdrive_extractor_override(item, tool_name, self.gmap) == (
            "google_drive", {"action": "read", "file_id": file_id},
        )

    def test_url_items_keep_their_fetcher(self):
        loop = _Loop()
        item = "https://example.com/report.pdf"
        tool_name, _ = loop._detect_item_extractor(item)
        assert tool_name == "web_fetch"
        assert loop._gdrive_extractor_override(item, tool_name, self.gmap) is None

    def test_unlisted_items_and_empty_map_are_left_alone(self):
        assert AgentScaleLoopMixin._gdrive_extractor_override(
            "/tmp/other.pdf", "pdf_extract", {"zzz.txt": {"id": "z", "mimeType": "text/plain"}},
        ) is None
        assert AgentScaleLoopMixin._gdrive_extractor_override(
            "report.pdf", "pdf_extract", {},
        ) is None


class TestNoPartialMatches:
    """The map spans the whole session, so a Drive listing from earlier must
    not hijack a later scale run over local files or plain-text entities."""

    gmap = AgentScaleLoopMixin._parse_gdrive_listing(
        "  [file] Q3 report.pdf (1.2 MB)\n"
        f"    ID: q3pdf  |  Type: {PDF}\n"
        "  [file] Daily standup\n"
        f"    ID: standup  |  Type: {DOC}\n"
        "  [file] Data\n"
        f"    ID: data  |  Type: {SHEET}\n"
        "  [file] README.md (2.0 KB)\n"
        "    ID: readme  |  Type: text/markdown\n"
        "  [file] notes.txt (512 B)\n"
        "    ID: notes  |  Type: text/plain\n"
    )

    @pytest.mark.parametrize("item", [
        "report.pdf",                       # inside "Q3 report.pdf"
        "AI",                               # inside "daily standup"
        "Acme Q3",
        "Q3 report.pdf (final)",            # contains "q3 report.pdf"
        "/Users/me/projA/README.md",        # basename == Drive "README.md"
        "/Users/me/projB/docs/README.md",
        "/Users/me/data/sales.csv",         # path contains "data"
        "/Users/me/docs/meeting_notes.txt",  # path contains "notes.txt"
    ])
    def test_partial_name_overlap_keeps_the_local_extractor(self, item):
        loop = _Loop()
        tool_name, args = loop._detect_item_extractor(item)
        assert loop._gdrive_extractor_override(item, tool_name, self.gmap) is None

    def test_existing_local_file_wins_over_a_same_named_drive_file(self, tmp_path):
        local = tmp_path / "README.md"
        local.write_text("local")
        gmap = {str(local): {"id": "clash", "mimeType": "text/markdown"}}
        assert AgentScaleLoopMixin._gdrive_extractor_override(
            str(local), "read", gmap,
        ) is None
        # A bare name that exists in the working directory is local too.
        assert AgentScaleLoopMixin._gdrive_extractor_override(
            "README.md", "read", self.gmap,
        ) is None
        # The same name, absent locally, still reads from Drive.
        missing = str(tmp_path / "gone.md")
        gmap[missing] = {"id": "drv", "mimeType": "text/markdown"}
        assert AgentScaleLoopMixin._gdrive_extractor_override(
            missing, "read", gmap,
        ) == ("google_drive", {"action": "read", "file_id": "drv"})

    def test_full_names_still_route(self):
        for item, file_id in (("Daily standup", "standup"), ("data", "data"),
                              (" Q3 REPORT.PDF ", "q3pdf")):
            tool_name, _ = _Loop()._detect_item_extractor(item)
            assert AgentScaleLoopMixin._gdrive_extractor_override(
                item, tool_name, self.gmap,
            ) == ("google_drive", {"action": "read", "file_id": file_id})


def test_the_scale_loop_never_names_the_gws_tool():
    src = inspect.getsource(agent_scale_loop_mixin)
    assert '"gws"' not in src and "shutil.which" not in src
