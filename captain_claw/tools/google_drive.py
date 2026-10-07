"""Google Drive tool for listing, searching, reading, and writing files.

Uses the Google Drive REST API v3 via httpx with OAuth2 Bearer tokens
managed by :class:`~captain_claw.google_oauth_manager.GoogleOAuthManager`.
No additional Google SDK dependencies are required.
"""

from __future__ import annotations

import asyncio
import contextvars
import io
import json
import mimetypes
import re
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from captain_claw.config import get_config
from captain_claw.drive_client import FOLDER_MIME, GOOGLE_EXPORT
from captain_claw.google_ids import (
    drive_id_from_url,
    drive_resource_key,
    is_google_drive_url,
)
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

if TYPE_CHECKING:
    from captain_claw.google_oauth import GoogleOAuthTokens

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRIVE_API = "https://www.googleapis.com/drive/v3"
_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
# Any Drive scope grants read; write actions additionally need the check below.
_DRIVE_READ_SCOPES = frozenset({
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
})
_DRIVE_WRITE_SCOPES = frozenset({
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
})
# sheet_* / doc_* edits change a native Google Sheet / Doc IN PLACE (Sheets API
# v4 / Docs API v1, same token): cells and text, never a re-upload.
_IN_PLACE_WRITE_ACTIONS = frozenset({
    "sheet_update", "sheet_append", "sheet_clear",
    "doc_replace_text", "doc_append_text", "doc_insert_text",
})
_WRITE_ACTIONS = frozenset({"upload", "create", "update"}) | _IN_PLACE_WRITE_ACTIONS
# Scopes that reach a file this app did not create — a link shared with the
# user, a colleague's Doc. drive.file alone sees only files the app created
# or the user picked, so a pasted link 404s under it.
_DRIVE_LINK_SCOPES = frozenset({
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.readonly",
})

# Sheets / Docs APIs. The `drive` scope covers both (no extra scope, no
# reconnect); drive.file reaches only files this app created, drive.readonly
# only the reads. Each deck's Google Cloud project must enable both APIs.
_SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
_DOCS_API = "https://docs.googleapis.com/v1/documents"
_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
_DOC_MIME = "application/vnd.google-apps.document"
_SHEET_ACTIONS = frozenset({"sheet_read", "sheet_update", "sheet_append", "sheet_clear"})
_DOC_ACTIONS = frozenset({"doc_read", "doc_replace_text", "doc_append_text", "doc_insert_text"})
# sheet_read without a range: the first rows of up to this many tabs.
_SHEET_PREVIEW_ROWS = 50
_SHEET_PREVIEW_TABS = 20
# What each in-place kind accepts, and the actions that edit it.
_IN_PLACE_KINDS: dict[str, tuple[str, str, str]] = {
    # kind → (label, native mime, actions)
    "sheet": ("Google Sheet", _SHEET_MIME, "sheet_read / sheet_update / sheet_append / sheet_clear"),
    "doc": ("Google Doc", _DOC_MIME, "doc_read / doc_replace_text / doc_append_text / doc_insert_text"),
}
_FULL_DRIVE_NEEDED = (
    "Google is connected without full Drive access, so this Google Sheet/Doc "
    "can't be opened or edited in place. The admin must enable Drive (full access) in "
    "Flight Deck → Connections → Google, and everyone reconnects Google."
)

# Fields to request from the files endpoint.
_FILE_FIELDS = "id,name,mimeType,size,modifiedTime,createdTime,parents,webViewLink,owners"
_LIST_FIELDS = f"nextPageToken,files({_FILE_FIELDS})"

# list / search: Drive's largest page, the most one call returns, and a
# bound on pages followed (Drive may return short pages).
_MAX_PAGE_SIZE = 1000
_MAX_LIST_RESULTS = 1000
_MAX_LIST_PAGES = 50

# Google Workspace MIME types and their export mappings.
_GOOGLE_EXPORT_MAP: dict[str, tuple[str, str]] = {
    # mime_type → (export_mime, file_extension)
    "application/vnd.google-apps.document": ("text/markdown", ".md"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": ("text/plain", ".txt"),
    "application/vnd.google-apps.drawing": ("image/svg+xml", ".svg"),
}

# Binary MIME types that can be handled by existing extract tools.
_EXTRACTABLE_MIMES: dict[str, str] = {
    "application/pdf": "pdf_extract",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx_extract",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx_extract",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx_extract",
}

# Maximum content size for read operations (bytes).
_MAX_READ_BYTES = 500_000  # 500 KB
# Rows per tab a Sheet read renders (the read stays under _MAX_READ_BYTES).
_SHEET_READ_MAX_ROWS = 1000
# Largest file read (download + extract) or download will fetch.
_MAX_DOWNLOAD_BYTES = 50_000_000  # 50 MB

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# The extension each extract tool insists on — the temp file of a read and
# a download whose Drive name lacks it (an uploaded PDF titled "Contract").
_SUFFIX_BY_MIME = {
    "application/pdf": ".pdf", _DOCX_MIME: ".docx",
    _XLSX_MIME: ".xlsx", _PPTX_MIME: ".pptx",
}

# read: Sheets and Slides export as XLSX / PPTX and go through the extract
# tools — Drive's CSV export holds only a Sheet's first tab, and its text
# export flattens a deck. The flat export stays as the fallback.
_READ_VIA_OFFICE_EXPORT = frozenset({
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
})

# download: export formats an output_path extension may ask for, per Google
# type. Anything else gets the drive_client.GOOGLE_EXPORT default (Doc → .md,
# Sheet → .xlsx, Slides → .pptx), which the read/extract tools render best.
_EXPORT_BY_SUFFIX: dict[str, dict[str, str]] = {
    "application/vnd.google-apps.document": {
        ".md": "text/markdown", ".txt": "text/plain", ".html": "text/html",
        ".docx": _DOCX_MIME, ".pdf": "application/pdf",
    },
    "application/vnd.google-apps.spreadsheet": {
        ".xlsx": _XLSX_MIME, ".csv": "text/csv", ".pdf": "application/pdf",
    },
    "application/vnd.google-apps.presentation": {
        ".pptx": _PPTX_MIME, ".txt": "text/plain", ".pdf": "application/pdf",
    },
    "application/vnd.google-apps.drawing": {
        ".svg": "image/svg+xml", ".png": "image/png", ".pdf": "application/pdf",
    },
}
_TEXT_EXPORTS = frozenset({"text/markdown", "text/plain", "text/html"})

# The write tool's saved/ categories; a relative download path outside them
# is filed under downloads/.
_SAVED_CATEGORIES = frozenset({
    "downloads", "media", "output", "scripts", "showcase",
    "skills", "summaries", "tmp", "tools",
})

# Which tool reads a downloaded file, by extension (default: read).
_READER_BY_SUFFIX = {
    ".pdf": "pdf_extract", ".docx": "docx_extract",
    ".xlsx": "xlsx_extract", ".pptx": "pptx_extract",
    ".png": "image_vision", ".jpg": "image_vision", ".jpeg": "image_vision",
}

# Inline base64 images in a Docs markdown export (``![](data:image/png;base64,…)``)
# run to megabytes and would bury the text — replaced with "[image]".
_BASE64_IMG_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+")


def _strip_base64_images(text: str) -> str:
    """Remove inline base64 image data from text to prevent context bloat."""
    cleaned = _BASE64_IMG_RE.sub("[image]", text)
    if len(cleaned) < len(text):
        log.debug(
            "stripped base64 images",
            original_len=len(text),
            cleaned_len=len(cleaned),
        )
    return cleaned


# "<file id>/<resource key>" for the file this call works on, from a pasted
# URL's ?resourcekey= — older link-shared files 404 without it. Per call
# (a context variable), as tool calls may run concurrently.
_RESOURCE_KEY: contextvars.ContextVar[str] = contextvars.ContextVar(
    "google_drive_resource_key", default="",
)


def agent_offers_google_drive(agent: Any) -> bool:
    """Whether the calling agent's model can actually call google_drive.

    The web tools only redirect a Drive link to google_drive when it can:
    a sister session never registers it, and nano mode cuts it from the tool
    list the model sees. Pointing those agents at a tool they don't have
    would strand a link-shared Doc that an anonymous fetch could still read.
    ``agent`` is the ``_agent`` every guarded tool call carries; None (a
    direct call) keeps the redirect.
    """
    if agent is None:
        return True
    try:
        if not agent.tools.has_tool("google_drive"):
            return False
    except Exception:
        return True
    instructions = getattr(agent, "instructions", None)
    return not bool(getattr(instructions, "use_nano", False))


async def google_drive_reads_links() -> bool:
    """True when google_drive can open a Drive link the user pastes.

    That needs Google connected with a scope that reaches files this app did
    not create (drive / drive.readonly). Under drive.file only, a pasted
    link-shared Doc 404s, so the web tools let their anonymous fetch run
    instead of redirecting here. An unreported scope counts as yes — the
    tool doesn't refuse on it either.
    """
    from captain_claw.google_oauth_manager import (
        GoogleOAuthManager,
        is_google_connected_cached,
    )
    from captain_claw.session import get_session_manager

    if not is_google_connected_cached():
        return False
    try:
        tokens = await GoogleOAuthManager(get_session_manager()).get_tokens()
    except Exception as exc:  # FD refused / unreachable: google_drive would fail too
        log.debug("google_drive link check failed", error=str(exc))
        return False
    if not tokens:
        return False
    granted = set(tokens.scope.split()) if tokens.scope else set()
    return not granted or bool(granted & _DRIVE_LINK_SCOPES)


def _missing_suffix(name: str, mime: str) -> str:
    """The extension a downloaded file needs for its reader, or "".

    Extract tools require theirs exactly (``Contract v2.1`` → ``.pdf``
    appended); any other type gets one only when the name has none.
    """
    current = Path(name).suffix.lower()
    wanted = _SUFFIX_BY_MIME.get(mime)
    if wanted:
        return "" if current == wanted else wanted
    if current or not mime or mime == "application/octet-stream":
        return ""
    return mimetypes.guess_extension(mime) or ""


def _tab_hit_row_cap(markdown: str, max_rows: int) -> bool:
    """True if a tab in xlsx_extract output stopped at its row cap.

    The extractor drops rows past the cap without saying so; a tab rendered
    with exactly *max_rows* rows (header + separator + body lines) may
    continue.
    """
    rows = 0
    for line in markdown.splitlines():
        if line.startswith("## Sheet:"):
            rows = 0
        elif line.startswith("| "):
            rows += 1
            if rows - 1 >= max_rows:  # one line is the | --- | separator
                return True
    return False


def _resolve_drive_ref(value: str) -> str | None:
    """A file_id / folder_id argument as a bare id.

    Drive/Docs/Sheets/Slides URLs give up their id; any other URL is not a
    Drive reference (None). Everything else passes through unchanged — the
    API is the judge of a bare id, and ``root`` must keep working.
    """
    ref = value.strip()
    if is_google_drive_url(ref):
        return drive_id_from_url(ref)
    if "://" in ref:
        return None
    return ref


def _safe_filename(name: str, fallback: str) -> str:
    """A Drive file name as one safe path component."""
    cleaned = re.sub(r"[\x00-\x1f/\\]+", "_", name or "").strip().lstrip(".").strip()
    return cleaned[:200] or fallback


def _truthy(value: Any, default: bool) -> bool:
    """A boolean argument as models send it (true, "false", 1, None)."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


# ── Sheets: A1 addresses and cell grids ──────────────────────────────

_A1_CELL_RE = re.compile(r"^\$?([A-Za-z]{0,3})\$?(\d*)$")
_VALUES_EXAMPLE = '[["Name", "Total"], ["Ana", "=SUM(B2:B9)"]]'


def _col_letters(n: int) -> str:
    """1 → A, 26 → Z, 27 → AA."""
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def _col_number(letters: str) -> int:
    """A → 1, Z → 26, AA → 27."""
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def _range_start(a1: str) -> tuple[int, int]:
    """(column, row), 1-based, of the top-left cell of an A1 range.

    Meant for the range the Sheets API echoes back ("Q2!B3:D9", "'My tab'!
    A1:Z50"): the values it returns start at that cell.
    """
    cells = a1.rsplit("!", 1)[-1]
    m = _A1_CELL_RE.match(cells.split(":", 1)[0].strip())
    if not m:
        return 1, 1
    col = _col_number(m.group(1)) if m.group(1) else 1
    row = int(m.group(2)) if m.group(2) else 1
    return col, row


def _quote_tab(title: str) -> str:
    """A tab name as an A1 sheet reference: Q3 plan → 'Q3 plan'."""
    return "'" + title.replace("'", "''") + "'"


def _cell_text(value: Any) -> str:
    """One cell of a grid row (pipes and line breaks escaped)."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r\n", "\\n").replace("\n", "\\n")


def _render_grid(values: list[Any], echoed_range: str, max_rows: int) -> tuple[str, int]:
    """Rows as a grid headed by column letters, each row led by its number,
    so any cell reads off as an exact A1 address.

    Returns the grid and how many rows past *max_rows* were left out.
    """
    start_col, start_row = _range_start(echoed_range)
    shown = [r if isinstance(r, list) else [] for r in values[:max_rows]]
    width = max((len(r) for r in shown), default=0)
    if width == 0:
        return "(empty)", 0
    letters = [_col_letters(start_col + i) for i in range(width)]
    lines = ["| row | " + " | ".join(letters) + " |", "| --- |" + " --- |" * width]
    for i, row in enumerate(shown):
        cells = [_cell_text(row[j]) if j < len(row) else "" for j in range(width)]
        lines.append(f"| {start_row + i} | " + " | ".join(cells) + " |")
    return "\n".join(lines), max(0, len(values) - max_rows)


def _coerce_rows(values: Any) -> tuple[list[list[Any]] | None, str]:
    """``values`` as rows of cells, or why it isn't one.

    Takes the shapes models send: rows of cells, one bare row, a single
    value, or any of those as a JSON string.
    """
    if isinstance(values, str):
        stripped = values.strip()
        if not stripped.startswith("["):
            return [[values]], ""
        try:
            values = json.loads(stripped)
        except ValueError:
            return None, f"values is not valid JSON; send rows of cells, e.g. {_VALUES_EXAMPLE}."
    elif isinstance(values, (int, float, bool)):
        return [[values]], ""
    if not isinstance(values, list) or not values:
        return None, f"values must be a non-empty list of rows, e.g. {_VALUES_EXAMPLE}."
    if not any(isinstance(v, list) for v in values):
        values = [values]  # one row sent bare
    for row in values:
        if not isinstance(row, list):
            return None, (
                "values must be a list of rows (each row a list of cells), "
                f"not a mix of rows and single cells, e.g. {_VALUES_EXAMPLE}."
            )
        if any(isinstance(cell, (dict, list)) for cell in row):
            return None, "each cell must be text, a number, a boolean or null — not a list or object."
    return values, ""


def _coerce_updates(updates: Any) -> tuple[list[dict[str, Any]], str]:
    """``updates`` as values:batchUpdate data entries, or why it isn't one."""
    example = '[{"range": "Sheet1!B2", "values": [["42"]]}]'
    if isinstance(updates, str):
        try:
            updates = json.loads(updates)
        except ValueError:
            return [], f"updates must be a list of {{range, values}} objects, e.g. {example}."
    if isinstance(updates, dict):
        updates = [updates]
    if not isinstance(updates, list) or not updates:
        return [], f"updates must be a non-empty list of {{range, values}} objects, e.g. {example}."
    data: list[dict[str, Any]] = []
    for n, item in enumerate(updates, 1):
        a1 = str(item.get("range") or "").strip() if isinstance(item, dict) else ""
        if not a1:
            return [], f"updates item {n} needs a range and values, e.g. {example}."
        rows, err = _coerce_rows(item.get("values"))
        if err:
            return [], f"updates item {n} ({a1}): {err}"
        data.append({"range": a1, "majorDimension": "ROWS", "values": rows})
    return data, ""


def _value_input_option(value_input: Any) -> str | None:
    """value_input → the API's valueInputOption (None when not recognised)."""
    choice = str(value_input or "user_entered").strip().lower()
    return {"user_entered": "USER_ENTERED", "raw": "RAW"}.get(choice)


# ── Docs: tabs, text runs and indexes ────────────────────────────────


def _doc_tabs(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tab of a documents.get(includeTabsContent=true) answer, child
    tabs after their parent; a tab-less answer as one untitled tab."""
    out: list[dict[str, Any]] = []

    def walk(tabs: Any) -> None:
        for tab in tabs or []:
            if isinstance(tab, dict):
                out.append(tab)
                walk(tab.get("childTabs"))

    walk(doc.get("tabs"))
    if not out and isinstance(doc.get("body"), dict):
        out.append({"tabProperties": {}, "documentTab": {"body": doc["body"]}})
    return out


def _tab_props(tab: dict[str, Any]) -> tuple[str, str]:
    """(tab id, title) of a Docs tab."""
    props = tab.get("tabProperties") or {}
    return str(props.get("tabId") or ""), str(props.get("title") or "")


def _tab_content(tab: dict[str, Any]) -> list[dict[str, Any]]:
    return ((tab.get("documentTab") or {}).get("body") or {}).get("content") or []


def _doc_runs(content: list[dict[str, Any]]):
    """(start index, text) of every text run — table cells and tables of
    contents included — in document order."""
    for el in content or []:
        if "paragraph" in el:
            for pe in el["paragraph"].get("elements") or []:
                run = pe.get("textRun")
                if run and run.get("content"):
                    yield int(pe.get("startIndex") or 0), run["content"]
        elif "table" in el:
            for row in el["table"].get("tableRows") or []:
                for cell in row.get("tableCells") or []:
                    yield from _doc_runs(cell.get("content"))
        elif "tableOfContents" in el:
            yield from _doc_runs(el["tableOfContents"].get("content"))


def _doc_text(content: list[dict[str, Any]]) -> str:
    """A tab's text as the Docs API holds it — what replaceAllText matches.

    Paragraphs are exact (no markdown escaping). Only an image (``[image]``)
    and a table row (``| cell | cell |``) are drawn rather than copied.
    """
    parts: list[str] = []
    for el in content or []:
        if "paragraph" in el:
            for pe in el["paragraph"].get("elements") or []:
                if "textRun" in pe:
                    parts.append(pe["textRun"].get("content") or "")
                elif "inlineObjectElement" in pe:
                    parts.append("[image]")
                elif "person" in pe:
                    parts.append((pe["person"].get("personProperties") or {}).get("name") or "")
                elif "richLink" in pe:
                    parts.append((pe["richLink"].get("richLinkProperties") or {}).get("title") or "")
        elif "table" in el:
            for row in el["table"].get("tableRows") or []:
                cells = [
                    _doc_text(cell.get("content")).rstrip("\n").replace("\n", " / ")
                    for cell in row.get("tableCells") or []
                ]
                parts.append("| " + " | ".join(cells) + " |\n")
        elif "tableOfContents" in el:
            parts.append(_doc_text(el["tableOfContents"].get("content")))
    return "".join(parts).replace("\x0b", "\n")


def _utf16_len(ch: str) -> int:
    """Docs indexes count UTF-16 code units: 2 for an emoji, 1 otherwise."""
    return 2 if ord(ch) > 0xFFFF else 1


def _anchor_ends(content: list[dict[str, Any]], anchor: str, match_case: bool) -> list[int]:
    """The document index just past each occurrence of *anchor* in a tab."""
    chars: list[str] = []
    after_index: list[int] = []  # index just past each character
    for start, text in _doc_runs(content):
        pos = start
        for ch in text:
            pos += _utf16_len(ch)
            chars.append("\n" if ch == "\x0b" else ch)  # a soft break reads as one
            after_index.append(pos)

    def fold(s: str) -> str:
        # Lower-case only where that keeps the length (offsets must line up).
        return "".join(c.lower() if len(c.lower()) == 1 else c for c in s)

    hay = "".join(chars)
    needle = anchor
    if not match_case:
        hay, needle = fold(hay), fold(needle)
    ends: list[int] = []
    at = hay.find(needle)
    while at != -1 and needle:
        ends.append(after_index[at + len(needle) - 1])
        at = hay.find(needle, at + 1)
    return ends


class _DownloadPathError(ValueError):
    """output_path points somewhere a download may not write."""


class _ScopeMissingError(RuntimeError):
    """The granted scopes don't cover the action (refused before any request)."""


class GoogleDriveTool(Tool):
    """Interact with Google Drive: list, search, read, download, upload, create, and update files."""

    name = "google_drive"
    description = (
        "Read and write the user's Google Drive, including Docs, Sheets, Slides "
        "and shared drives. Actions: list (folder contents), search (find files "
        "by name or content), read (content inline: Docs as markdown, Sheets/"
        "Slides exported, PDF/DOCX/XLSX/PPTX extracted), info (metadata), "
        "download (save a local copy and return its path), upload (send a local "
        "file to Drive), create (new file on Drive), update (replace a file's "
        "content). file_id and folder_id also accept a full Drive/Docs/Sheets/"
        "Slides URL. "
        "Edit an existing Google Sheet IN PLACE: sheet_read (tabs, or a range "
        "shown with row numbers and column letters), sheet_update (write cells), "
        "sheet_append (add rows), sheet_clear. Edit an existing Google Doc IN "
        "PLACE: doc_read (exact text), doc_replace_text, doc_append_text, "
        "doc_insert_text. Never upload a modified copy of an existing Sheet/Doc "
        "and never update one as a whole file."
    )
    timeout_seconds = 120.0
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list", "search", "read", "info", "download",
                    "upload", "create", "update",
                    "sheet_read", "sheet_update", "sheet_append", "sheet_clear",
                    "doc_read", "doc_replace_text", "doc_append_text", "doc_insert_text",
                ],
                "description": "The action to perform.",
            },
            "file_id": {
                "type": "string",
                "description": (
                    "Google Drive file ID or the file's Drive/Docs/Sheets/Slides "
                    "URL (for read, info, download, update and the sheet_* / "
                    "doc_* actions)."
                ),
            },
            "folder_id": {
                "type": "string",
                "description": (
                    "Folder ID or Drive folder URL to list or upload into. "
                    "Defaults to 'root'."
                ),
            },
            "output_path": {
                "type": "string",
                "description": (
                    "download only: where to save, inside the saved/ area "
                    "(default saved/downloads/<session>/<file name>). For "
                    "Google Docs/Sheets/Slides the extension picks the export "
                    "format (.md/.docx/.pdf/.txt/.html for Docs, .xlsx/.csv/.pdf "
                    "for Sheets, .pptx/.pdf/.txt for Slides); default Docs → .md, "
                    "Sheets → .xlsx, Slides → .pptx."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search query text (for search action).",
            },
            "max_results": {
                "type": "number",
                "description": (
                    "Maximum number of results for list (default 100) or search "
                    "(default 20); up to 1000."
                ),
            },
            "page_token": {
                "type": "string",
                "description": (
                    "list/search: continue from the page_token a previous "
                    "result ended with (it says when more files exist)."
                ),
            },
            "local_path": {
                "type": "string",
                "description": "Local file path (for upload action or update from file).",
            },
            "name": {
                "type": "string",
                "description": "File name (for upload or create actions).",
            },
            "content": {
                "type": "string",
                "description": "Text content (for create or update actions).",
            },
            "mime_type": {
                "type": "string",
                "description": (
                    "MIME type for create action. Use 'application/vnd.google-apps.document' "
                    "for Google Doc, 'application/vnd.google-apps.spreadsheet' for Google Sheet, "
                    "or 'text/plain' for plain text."
                ),
            },
            "order_by": {
                "type": "string",
                "description": (
                    "Sort order for list/search (e.g. 'modifiedTime desc', "
                    "'name', 'createdTime desc'). Default: 'modifiedTime desc'."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "update only: true replaces the WHOLE content of a native "
                    "Google Doc/Sheet/Slides file. Refused without it — edit "
                    "Sheets/Docs in place with sheet_* / doc_* instead."
                ),
            },
            "range": {
                "type": "string",
                "description": (
                    "sheet_*: A1 range, e.g. 'Sheet1!B2', 'Sheet1!A1:C10', "
                    "'Sheet1!A:F' or just 'Sheet1' (quote tab names with spaces: "
                    "\"'Q3 plan'!B2\"). sheet_read without a range lists the tabs "
                    "and shows the first rows of each; sheet_append adds rows "
                    "below the table in this range."
                ),
            },
            "values": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
                "description": (
                    "sheet_update / sheet_append: rows of cell values, e.g. "
                    "[[\"Name\", \"Total\"], [\"Ana\", \"=SUM(B2:B9)\"]]. Entered "
                    "as if typed (formulas, numbers and dates are parsed) unless "
                    "value_input='raw'."
                ),
            },
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "range": {"type": "string"},
                        "values": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "required": ["range", "values"],
                },
                "description": (
                    "sheet_update: several ranges in one call, "
                    "[{\"range\": \"Sheet1!B2\", \"values\": [[\"42\"]]}, ...]."
                ),
            },
            "value_input": {
                "type": "string",
                "enum": ["user_entered", "raw"],
                "description": (
                    "sheet_update / sheet_append: 'user_entered' (default — parsed "
                    "as if typed) or 'raw' (stored exactly as given)."
                ),
            },
            "render": {
                "type": "string",
                "enum": ["values", "formulas"],
                "description": "sheet_read: show the displayed 'values' (default) or the 'formulas'.",
            },
            "find": {
                "type": "string",
                "description": (
                    "doc_replace_text: the exact text to replace, copied from "
                    "doc_read (within one paragraph)."
                ),
            },
            "replace_with": {
                "type": "string",
                "description": "doc_replace_text: the new text ('' deletes the found text).",
            },
            "match_case": {
                "type": "boolean",
                "description": "doc_replace_text / doc_insert_text: case-sensitive match (default true).",
            },
            "text": {
                "type": "string",
                "description": (
                    "doc_append_text: text added as a new paragraph at the end. "
                    "doc_insert_text: text inserted exactly as given right after "
                    "`after`."
                ),
            },
            "after": {
                "type": "string",
                "description": (
                    "doc_insert_text: existing text, copied exactly from doc_read, "
                    "that the new text goes right after; it must occur once."
                ),
            },
            "tab": {
                "type": "string",
                "description": (
                    "doc_* on a Doc with several tabs: the tab title or id to work "
                    "in (default: all tabs; doc_append_text: the first tab)."
                ),
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
            headers={"User-Agent": "Captain Claw/0.1.0 (Google Drive Tool)"},
        )

    async def execute(self, action: str, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action handler."""
        # Pop injected kwargs that tools receive from the registry; download
        # needs the saved-area ones to place its file.
        runtime = {
            key: kwargs.pop(key, None)
            for key in (
                "_runtime_base_path", "_saved_base_path", "_session_id",
                "_file_registry", "_task_id",
            )
        }
        kwargs.pop("_abort_event", None)

        # A pasted Drive/Docs URL works wherever an id is expected.
        resource_key = ""
        for key in ("file_id", "folder_id"):
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                resolved = _resolve_drive_ref(value)
                if not resolved:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Could not find a Google Drive id in {key}={value!r}. "
                            "Pass the id (the part after /d/, ?id= or /folders/ "
                            "in a Drive link) or a Drive/Docs URL that contains "
                            "one; published (/d/e/...) links carry no file id."
                        ),
                    )
                if key == "file_id" and is_google_drive_url(value.strip()):
                    rkey = drive_resource_key(value)
                    resource_key = f"{resolved}/{rkey}" if rkey else ""
                kwargs[key] = resolved

        handlers = {
            "list": self._action_list,
            "search": self._action_search,
            "read": self._action_read,
            "info": self._action_info,
            "download": self._action_download,
            "upload": self._action_upload,
            "create": self._action_create,
            "update": self._action_update,
            "sheet_read": self._action_sheet_read,
            "sheet_update": self._action_sheet_update,
            "sheet_append": self._action_sheet_append,
            "sheet_clear": self._action_sheet_clear,
            "doc_read": self._action_doc_read,
            "doc_replace_text": self._action_doc_replace_text,
            "doc_append_text": self._action_doc_append_text,
            "doc_insert_text": self._action_doc_insert_text,
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"Unknown action '{action}'. Use one of: {', '.join(handlers)}",
            )
        if action == "download":
            kwargs["runtime"] = runtime

        try:
            tokens = await self._get_tokens(write=action in _WRITE_ACTIONS)
        except _ScopeMissingError as e:
            # A read-only connection can't edit a Sheet/Doc in place either:
            # that is the deck's scope set, not this user's reconnect.
            error = _FULL_DRIVE_NEEDED if action in _IN_PLACE_WRITE_ACTIONS else str(e)
            return ToolResult(success=False, error=error)
        except RuntimeError as e:
            return ToolResult(success=False, error=str(e))
        token = tokens.access_token
        granted = frozenset(tokens.scope.split()) if tokens.scope else frozenset()

        key_token = _RESOURCE_KEY.set(resource_key)
        try:
            return await handler(token, **kwargs)
        except httpx.HTTPStatusError as exc:
            return self._handle_http_error(exc, action=action, granted=granted)
        except httpx.HTTPError as exc:
            log.error("Google Drive HTTP error", action=action, error=str(exc))
            return ToolResult(success=False, error=f"HTTP error: {exc}")
        except Exception as exc:
            log.error("Google Drive tool error", action=action, error=str(exc))
            return ToolResult(success=False, error=str(exc))
        finally:
            _RESOURCE_KEY.reset(key_token)

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    async def _get_access_token(self, *, write: bool = False) -> str:
        """Retrieve a valid Google OAuth access token.

        Raises RuntimeError if Google is not connected, or if the granted
        scopes don't cover the operation. Read actions accept any Drive scope
        including ``drive.readonly``; only write actions require a writable one.
        The tool previously demanded full read/write ``drive`` for *everything*,
        so a read-only connection couldn't even list a folder.
        """
        return (await self._get_tokens(write=write)).access_token

    async def _get_tokens(self, *, write: bool = False) -> GoogleOAuthTokens:
        """The checked tokens behind :meth:`_get_access_token` — their scope
        string tells a scope 404 from a missing file."""
        from captain_claw.google_oauth_manager import GoogleOAuthManager
        from captain_claw.session import get_session_manager

        mgr = GoogleOAuthManager(get_session_manager())
        tokens = await mgr.get_tokens()
        if not tokens:
            if mgr._is_flight_deck_client():
                # Under Flight Deck the agent-local OAuth flow's tokens are
                # discarded — the owner connects THEIR account in Flight Deck.
                raise RuntimeError(
                    "Google account is not connected. Connect your Google "
                    "account in Flight Deck → Connections → Google."
                )
            raise RuntimeError(
                "Google account is not connected. "
                "Please connect via the web UI (Settings > Google OAuth) or "
                "navigate to /auth/google/login in your browser."
            )

        granted = set(tokens.scope.split()) if tokens.scope else set()
        # An empty scope string means the token endpoint didn't report scopes
        # (Flight Deck client mode sometimes omits them); don't block on that —
        # the API returns 403 if the scope is genuinely missing.
        if granted:
            needed = _DRIVE_WRITE_SCOPES if write else _DRIVE_READ_SCOPES
            if not granted.intersection(needed):
                raise _ScopeMissingError(
                    "Google Drive "
                    + ("write " if write else "")
                    + "scope not granted. Reconnect your Google account and "
                    + ("grant Drive edit access." if write else "grant Drive access (read-only is enough).")
                )

        return tokens

    def _auth_headers(self, token: str) -> dict[str, str]:
        """Build authorization headers (plus the file's resource key, if any)."""
        headers = {"Authorization": f"Bearer {token}"}
        resource_key = _RESOURCE_KEY.get()
        if resource_key:
            headers["X-Goog-Drive-Resource-Keys"] = resource_key
        return headers

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_http_error(
        exc: httpx.HTTPStatusError,
        *,
        action: str = "",
        granted: frozenset[str] = frozenset(),
    ) -> ToolResult:
        """Convert HTTP status errors into user-friendly messages."""
        status = exc.response.status_code
        try:
            body = exc.response.json()
            message = body.get("error", {}).get("message", str(exc))
        except Exception:
            body = {}
            message = str(exc)

        if action in _SHEET_ACTIONS or action in _DOC_ACTIONS:
            in_place = GoogleDriveTool._in_place_http_error(exc, action, granted, body, message)
            if in_place is not None:
                return in_place

        if status == 401:
            return ToolResult(
                success=False,
                error="Google authentication expired. Please reconnect your Google account.",
            )
        elif status == 403:
            return ToolResult(
                success=False,
                error=f"Permission denied: {message}",
            )
        elif status == 404:
            return ToolResult(
                success=False,
                error="File not found. Please check the file ID.",
            )
        elif status == 429:
            return ToolResult(
                success=False,
                error="Google Drive rate limit exceeded. Please try again in a moment.",
            )
        else:
            return ToolResult(
                success=False,
                error=f"Google Drive API error ({status}): {message}",
            )

    @staticmethod
    def _in_place_http_error(
        exc: httpx.HTTPStatusError,
        action: str,
        granted: frozenset[str],
        body: Any,
        message: str,
    ) -> ToolResult | None:
        """A sheet_* / doc_* failure in terms the admin can act on, or None
        for the generic mapping (401, 429, 5xx)."""
        status = exc.response.status_code
        err = body.get("error") if isinstance(body, dict) else None
        err = err if isinstance(err, dict) else {}
        reasons = {
            str(item.get("reason"))
            for key in ("details", "errors")
            for item in (err.get(key) or [])
            if isinstance(item, dict) and item.get("reason")
        }
        try:
            host = exc.request.url.host
        except RuntimeError:  # an error built without its request
            host = ""
        kind = "doc" if action in _DOC_ACTIONS else "sheet"
        if host.startswith("docs.") or (not host and kind == "doc"):
            api = "Google Docs API"
        elif host.startswith("sheets.") or not host:
            api = "Google Sheets API"
        else:  # the Drive metadata lookup that runs first
            api = "Google Drive API"
        label = _IN_PLACE_KINDS[kind][0]
        lowered = message.lower()

        if status == 403 and (
            reasons & {"SERVICE_DISABLED", "accessNotConfigured"}
            or "has not been used in project" in lowered
        ):
            return ToolResult(success=False, error=(
                f"The {api} is not enabled for this deck. Enable the {api} in the "
                "deck's Google Cloud project (APIs & Services → Library), wait a "
                f"minute, then retry. Google said: {message}"
            ))
        if status == 403 and (
            reasons & {"ACCESS_TOKEN_SCOPE_INSUFFICIENT", "insufficientPermissions"}
            or "insufficient authentication scopes" in lowered
        ):
            return ToolResult(success=False, error=_FULL_DRIVE_NEEDED)
        if status == 403:
            need = "edit" if action in _IN_PLACE_WRITE_ACTIONS else "view"
            return ToolResult(success=False, error=(
                f"Permission denied: {message.rstrip('.')}. The connected Google "
                f"account needs {need} access to this {label}."
            ))
        if status == 404:
            # drive.file reaches only files this app created: anyone else's
            # Sheet/Doc 404s, which is a scope problem, not a wrong id.
            if granted and not granted & _DRIVE_LINK_SCOPES:
                return ToolResult(success=False, error=_FULL_DRIVE_NEEDED)
            return ToolResult(success=False, error=(
                f"{label} not found. Check the file id, and that the connected "
                "Google account can open the file."
            ))
        if status == 400:
            hint = {
                "Google Sheets API": (
                    " Ranges are A1 notation such as 'Sheet1!A1:C10' with an existing "
                    "tab name (quote names with spaces: \"'Q3 plan'!B2\"), inside the "
                    "tab's grid (rows past its end: sheet_append); sheet_read without "
                    "a range lists the tabs and their sizes."
                ),
                "Google Docs API": (
                    " If the document changed since it was read, doc_read it again and retry."
                ),
            }.get(api, " Check the file id.")
            return ToolResult(success=False, error=f"{api} error (400): {message}{hint}")
        return None

    # ------------------------------------------------------------------
    # Action: list
    # ------------------------------------------------------------------

    async def _action_list(
        self,
        token: str,
        folder_id: str = "root",
        max_results: int | float | None = None,
        order_by: str | None = None,
        page_token: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """List files in a Google Drive folder."""
        limit = max(1, min(int(max_results or 100), _MAX_LIST_RESULTS))
        order = order_by or "modifiedTime desc"

        from captain_claw.drive_client import escape_query_value

        q = f"'{escape_query_value(folder_id)}' in parents and trashed = false"
        params = {
            "q": q,
            "fields": _LIST_FIELDS,
            "orderBy": order,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "corpora": "allDrives",
        }
        files, next_token = await self._list_files(token, params, limit, page_token)
        if not files:
            folder_label = f"folder '{folder_id}'" if folder_id != "root" else "root folder"
            return ToolResult(
                success=True,
                content=f"No files found in {folder_label}." + self._more_hint(files, next_token),
            )

        lines = [f"Files in {'root' if folder_id == 'root' else folder_id} ({len(files)} results):\n"]
        for f in files:
            size = f.get("size", "")
            size_str = f" ({self._format_size(int(size))})" if size else ""
            modified = f.get("modifiedTime", "")[:10] if f.get("modifiedTime") else ""
            is_folder = f.get("mimeType") == "application/vnd.google-apps.folder"
            type_icon = "[folder]" if is_folder else "[file]"

            lines.append(
                f"  {type_icon} {f['name']}{size_str}"
                f"\n    ID: {f['id']}"
                f"  |  Type: {f.get('mimeType', 'unknown')}"
                f"  |  Modified: {modified}"
            )

        return ToolResult(success=True, content="\n".join(lines) + self._more_hint(files, next_token))

    async def _list_files(
        self, token: str, params: dict[str, Any], limit: int, page_token: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Up to *limit* files for a files.list query, following nextPageToken.

        Returns the files and the token for what is left (None once the
        listing is complete), so a cut listing can say so.
        """
        files: list[dict[str, Any]] = []
        next_token = (page_token or "").strip() or None
        for _ in range(_MAX_LIST_PAGES):
            page_params = {**params, "pageSize": min(limit - len(files), _MAX_PAGE_SIZE)}
            if next_token:
                page_params["pageToken"] = next_token
            resp = await self._client.get(
                f"{_DRIVE_API}/files",
                params=page_params,
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            data = resp.json()
            files.extend(data.get("files", []))
            next_token = data.get("nextPageToken") or None
            if not next_token or len(files) >= limit:
                break
        return files[:limit], next_token

    @staticmethod
    def _more_hint(files: list[dict[str, Any]], next_token: str | None) -> str:
        """The closing line of a listing cut short (empty when complete)."""
        if not next_token:
            return ""
        return (
            f"\n\nMore files exist — showing {len(files)}. Repeat this call with "
            f"page_token='{next_token}' for the next page, or a higher max_results "
            f"(up to {_MAX_LIST_RESULTS})."
        )

    # ------------------------------------------------------------------
    # Action: search
    # ------------------------------------------------------------------

    async def _action_search(
        self,
        token: str,
        query: str = "",
        max_results: int | float | None = None,
        order_by: str | None = None,
        page_token: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Search for files across Google Drive."""
        if not query:
            return ToolResult(success=False, error="Search query is required.")

        limit = max(1, min(int(max_results or 20), _MAX_LIST_RESULTS))
        order = order_by or "relevance"

        # Build search query — search by name and full text.
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        q = f"(name contains '{escaped}' or fullText contains '{escaped}') and trashed = false"

        params = {
            "q": q,
            "fields": _LIST_FIELDS,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            # Default corpus is My Drive only; shared-drive files never matched.
            "corpora": "allDrives",
        }
        # 'relevance' is only valid without orderBy (it's the default).
        if order != "relevance":
            params["orderBy"] = order

        files, next_token = await self._list_files(token, params, limit, page_token)
        if not files:
            return ToolResult(
                success=True,
                content=f"No files found matching '{query}'." + self._more_hint(files, next_token),
            )

        lines = [f"Search results for '{query}' ({len(files)} found):\n"]
        for f in files:
            size = f.get("size", "")
            size_str = f" ({self._format_size(int(size))})" if size else ""
            modified = f.get("modifiedTime", "")[:10] if f.get("modifiedTime") else ""
            is_folder = f.get("mimeType") == "application/vnd.google-apps.folder"
            type_icon = "[folder]" if is_folder else "[file]"

            lines.append(
                f"  {type_icon} {f['name']}{size_str}"
                f"\n    ID: {f['id']}"
                f"  |  Type: {f.get('mimeType', 'unknown')}"
                f"  |  Modified: {modified}"
            )

        return ToolResult(success=True, content="\n".join(lines) + self._more_hint(files, next_token))

    # ------------------------------------------------------------------
    # Action: read
    # ------------------------------------------------------------------

    async def _action_read(
        self,
        token: str,
        file_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Read/export a file's content from Google Drive."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for read action.")

        # First, get file metadata to determine type.
        meta = await self._get_file_metadata(token, file_id)
        mime = meta.get("mimeType", "")
        name = meta.get("name", file_id)

        # Sheet / Slides → XLSX / PPTX export, extracted: every tab, slide by
        # slide. Drive caps exports at 10 MB — past that, the flat export.
        if mime in _READ_VIA_OFFICE_EXPORT:
            export_mime = GOOGLE_EXPORT[mime][0]
            try:
                result = await self._download_and_extract(
                    token, file_id, name, mime, export_mime=export_mime,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in (400, 403):
                    raise
                result = ToolResult(success=False, error=f"export failed ({exc.response.status_code})")
            if result.success:
                return self._with_edit_hint(result, mime, file_id)
            log.info("Office export read failed, using flat export", file_id=file_id, error=result.error)
            note = (
                "the XLSX export failed (Drive caps exports at 10 MB), so this "
                "is Drive's CSV export: the FIRST TAB ONLY."
                if mime == "application/vnd.google-apps.spreadsheet" else ""
            )
            result = await self._export_google_file(token, file_id, name, mime, note=note)
            return self._with_edit_hint(result, mime, file_id)

        # Google Workspace file → export.
        if mime in _GOOGLE_EXPORT_MAP:
            result = await self._export_google_file(token, file_id, name, mime)
            return self._with_edit_hint(result, mime, file_id)

        # Binary file with an extract tool → download + extract.
        if mime in _EXTRACTABLE_MIMES:
            return await self._download_and_extract(token, file_id, name, mime)

        # Folder → list contents instead.
        if mime == "application/vnd.google-apps.folder":
            return await self._action_list(token, folder_id=file_id)

        # Plain text / code / unknown → direct download as text.
        return await self._download_as_text(token, file_id, name, mime)

    @staticmethod
    def _with_edit_hint(result: ToolResult, mime: str, file_id: str) -> ToolResult:
        """A read of a Google Sheet / Doc ends by pointing at the in-place edits."""
        if not result.success:
            return result
        if mime == _SHEET_MIME:
            hint = (
                f"[To change this Sheet, edit it in place: sheet_read(file_id='{file_id}') "
                "shows cell addresses; sheet_update / sheet_append / sheet_clear write "
                "cells. Never upload a modified copy.]"
            )
        elif mime == _DOC_MIME:
            hint = (
                f"[To change this Doc, edit it in place: doc_read(file_id='{file_id}') "
                "shows the exact text; doc_replace_text / doc_append_text / "
                "doc_insert_text change it. Never upload a modified copy.]"
            )
        else:
            return result
        return result.model_copy(update={"content": f"{result.content}\n\n{hint}"})

    async def _export_google_file(
        self, token: str, file_id: str, name: str, mime: str, *, note: str = "",
    ) -> ToolResult:
        """Export a Google Workspace file (Docs, Sheets, Slides) as text."""
        export_mime, ext = _GOOGLE_EXPORT_MAP[mime]

        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}/export",
            params={"mimeType": export_mime, "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )

        # Markdown export might not be supported for all docs — fall back to plain text.
        if resp.status_code == 400 and export_mime == "text/markdown":
            resp = await self._client.get(
                f"{_DRIVE_API}/files/{file_id}/export",
                params={"mimeType": "text/plain", "supportsAllDrives": "true"},
                headers=self._auth_headers(token),
            )
            ext = ".txt"

        resp.raise_for_status()
        content = _strip_base64_images(resp.text)

        if len(content) > _MAX_READ_BYTES:
            content = content[:_MAX_READ_BYTES] + "\n\n... [content truncated]"

        header = f"[Google Drive: {name}]\n[Type: {mime} → exported as {export_mime}]\n[Size: {len(resp.content)} bytes]\n"
        if note:
            header += f"[Note: {note}]\n"
        return ToolResult(success=True, content=header + "\n" + content)

    async def _download_and_extract(
        self, token: str, file_id: str, name: str, mime: str, *, export_mime: str = "",
    ) -> ToolResult:
        """Download a binary file — or export a Google file as *export_mime*
        (XLSX/PPTX) — and extract its text with the matching extract tool."""
        source_mime = export_mime or mime
        tool_name = _EXTRACTABLE_MIMES[source_mime]

        if export_mime:
            url = f"{_DRIVE_API}/files/{file_id}/export"
            params = {"mimeType": export_mime, "supportsAllDrives": "true"}
        else:
            url = f"{_DRIVE_API}/files/{file_id}"
            params = {"alt": "media", "supportsAllDrives": "true"}
        body = await self._fetch_capped(token, url, params)
        if body is None:
            return ToolResult(
                success=False,
                error=f"File '{name}' is too large (over 50 MB). Max 50 MB.",
            )

        # Write to a temp file and run the appropriate extract tool. The
        # suffix comes from the type: the extract tools refuse any other, and
        # a Drive name often has none.
        with tempfile.NamedTemporaryFile(suffix=_SUFFIX_BY_MIME[source_mime], delete=False) as tmp:
            tmp.write(body)
            tmp_path = tmp.name

        try:
            extract_tool = self._get_extract_tool(tool_name)
            if extract_tool is None:
                return ToolResult(
                    success=False,
                    error=f"Extract tool '{tool_name}' not available. Cannot read '{name}'.",
                )

            # Room for what the flat exports returned, and for more of a
            # workbook's rows than the extractor's 200-per-tab default.
            limits: dict[str, Any] = {}
            if export_mime or tool_name == "xlsx_extract":
                limits["max_chars"] = _MAX_READ_BYTES
            if tool_name == "xlsx_extract":
                limits["max_rows"] = _SHEET_READ_MAX_ROWS
            result = await extract_tool.execute(path=tmp_path, **limits)
            if result.success:
                kind = f"{mime} → exported as {export_mime}" if export_mime else mime
                header = f"[Google Drive: {name}]\n[Type: {kind}]\n[Size: {self._format_size(len(body))}]\n"
                if tool_name == "xlsx_extract" and _tab_hit_row_cap(result.content, _SHEET_READ_MAX_ROWS):
                    header += (
                        f"[Note: a tab may run past the {_SHEET_READ_MAX_ROWS} rows shown; "
                        "download the file (.xlsx or .csv) for the full data]\n"
                    )
                content = result.content
                if len(content) > _MAX_READ_BYTES:
                    content = content[:_MAX_READ_BYTES] + "\n\n... [content truncated]"
                return ToolResult(success=True, content=header + "\n" + content)
            return result
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    async def _download_as_text(
        self, token: str, file_id: str, name: str, mime: str,
    ) -> ToolResult:
        """Download a file and return as text."""
        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()

        # Check size.
        if len(resp.content) > _MAX_READ_BYTES:
            try:
                content = resp.content[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
                content += "\n\n... [content truncated]"
            except Exception:
                return ToolResult(
                    success=False,
                    error=f"File '{name}' is too large and not text-readable.",
                )
        else:
            try:
                content = resp.text
            except Exception:
                return ToolResult(
                    success=False,
                    error=f"File '{name}' appears to be a binary file (MIME: {mime}). Cannot display as text.",
                )

        header = f"[Google Drive: {name}]\n[Type: {mime}]\n[Size: {self._format_size(len(resp.content))}]\n\n"
        return ToolResult(success=True, content=header + content)

    # ------------------------------------------------------------------
    # Action: info
    # ------------------------------------------------------------------

    async def _action_info(
        self,
        token: str,
        file_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Get detailed metadata for a file."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for info action.")

        fields = (
            "id,name,mimeType,size,modifiedTime,createdTime,parents,"
            "webViewLink,webContentLink,owners,shared,sharingUser,"
            "description,starred,trashed"
        )
        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"fields": fields, "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        meta = resp.json()

        lines = [f"File: {meta.get('name', 'unknown')}"]
        lines.append(f"  ID: {meta['id']}")
        lines.append(f"  Type: {meta.get('mimeType', 'unknown')}")
        if meta.get("size"):
            lines.append(f"  Size: {self._format_size(int(meta['size']))}")
        lines.append(f"  Created: {meta.get('createdTime', '?')}")
        lines.append(f"  Modified: {meta.get('modifiedTime', '?')}")
        if meta.get("parents"):
            lines.append(f"  Parent folders: {', '.join(meta['parents'])}")
        if meta.get("webViewLink"):
            lines.append(f"  Web link: {meta['webViewLink']}")
        if meta.get("owners"):
            owners = [o.get("displayName", o.get("emailAddress", "?")) for o in meta["owners"]]
            lines.append(f"  Owner(s): {', '.join(owners)}")
        if meta.get("description"):
            lines.append(f"  Description: {meta['description']}")
        lines.append(f"  Shared: {'yes' if meta.get('shared') else 'no'}")
        lines.append(f"  Starred: {'yes' if meta.get('starred') else 'no'}")
        lines.append(f"  Trashed: {'yes' if meta.get('trashed') else 'no'}")

        return ToolResult(success=True, content="\n".join(lines))

    # ------------------------------------------------------------------
    # Action: download
    # ------------------------------------------------------------------

    async def _action_download(
        self,
        token: str,
        file_id: str = "",
        output_path: str | None = None,
        runtime: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Save a Drive file locally; Google Docs/Sheets/Slides are exported."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for download action.")
        runtime = runtime or {}

        meta = await self._get_file_metadata(token, file_id)
        mime = meta.get("mimeType", "")
        name = meta.get("name") or file_id

        if mime == FOLDER_MIME:
            return ToolResult(
                success=False,
                error=(
                    f"'{name}' is a folder. Use list with folder_id='{file_id}' "
                    "to see its files, then download them one at a time."
                ),
            )
        native = mime.startswith("application/vnd.google-apps.")
        if native and mime not in GOOGLE_EXPORT:
            return ToolResult(
                success=False,
                error=(
                    f"'{name}' ({mime}) has no downloadable content. "
                    "Use info for its metadata and web link."
                ),
            )
        size = int(meta.get("size") or 0)
        if size > _MAX_DOWNLOAD_BYTES:
            return ToolResult(
                success=False,
                error=f"File '{name}' is too large ({self._format_size(size)}). Max 50 MB.",
            )

        # Export format: the output_path extension when this type supports
        # it, else drive_client's default for the type.
        export_mime, ext = "", ""
        if native:
            export_mime, ext = GOOGLE_EXPORT[mime]
            wanted = Path(str(output_path or "").strip()).suffix.lower()
            chosen = _EXPORT_BY_SUFFIX.get(mime, {}).get(wanted)
            if chosen:
                export_mime, ext = chosen, wanted

        default_name = _safe_filename(name, file_id)
        # A binary's Drive name may lack its extension, which picks the reader.
        suffix = ext or _missing_suffix(default_name, mime)
        if suffix and not default_name.lower().endswith(suffix):
            default_name += suffix
        try:
            dest = self._download_dest(output_path, default_name, ext, runtime)
        except _DownloadPathError as exc:
            return ToolResult(success=False, error=str(exc))

        if native:
            url = f"{_DRIVE_API}/files/{file_id}/export"
            params = {"mimeType": export_mime, "supportsAllDrives": "true"}
        else:
            url = f"{_DRIVE_API}/files/{file_id}"
            params = {"alt": "media", "supportsAllDrives": "true"}
        try:
            body = await self._fetch_capped(token, url, params)
        except httpx.HTTPStatusError as exc:
            # Markdown export isn't available for every Doc — plain text is.
            if not (exc.response.status_code == 400 and export_mime == "text/markdown"):
                raise
            export_mime = "text/plain"
            body = await self._fetch_capped(token, url, {**params, "mimeType": export_mime})
        if body is None:
            return ToolResult(
                success=False,
                error=f"File '{name}' is too large (over 50 MB). Max 50 MB.",
            )
        if export_mime in _TEXT_EXPORTS:
            text = body.decode("utf-8", errors="replace")
            stripped = _strip_base64_images(text)
            if len(stripped) < len(text):
                body = stripped.encode("utf-8")

        # PR C: a member never downloads over a saved/ file someone else
        # created; the creator is recorded after the write.
        from captain_claw import saved_attribution
        from captain_claw import speaker as _speaker

        prior = saved_attribution.prior_creator(dest)
        member = _speaker.current()
        if (member is not None and dest.exists()
                and not saved_attribution.member_may_change(dest, member.speaker_id)):
            return ToolResult(
                success=False,
                error=_speaker.PATH_REFUSED_PREFIX + _speaker.FILE_NOT_YOURS_WHY,
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(dest.write_bytes, body)
        saved_attribution.note_write(dest, prior)

        registry = runtime.get("_file_registry")
        if registry is not None:
            try:
                registry.register(
                    logical_path=str(output_path or "").strip() or dest.name,
                    physical_path=str(dest),
                    task_id=str(runtime.get("_task_id") or ""),
                )
            except Exception:
                pass  # non-critical

        kind = f"{mime} → exported as {export_mime}" if native else (mime or "unknown")
        reader = _READER_BY_SUFFIX.get(dest.suffix.lower(), "read")
        # A local copy is for analysis: the Drive original changes in place.
        edit_note = ""
        if mime == _SHEET_MIME:
            edit_note = (
                "\nThis is a local copy. To change the Google Sheet itself, edit it in "
                f"place with sheet_update / sheet_append / sheet_clear (file_id='{file_id}') "
                "— do not upload a modified copy."
            )
        elif mime == _DOC_MIME:
            edit_note = (
                "\nThis is a local copy. To change the Google Doc itself, edit it in "
                f"place with doc_replace_text / doc_append_text / doc_insert_text "
                f"(file_id='{file_id}') — do not upload a modified copy."
            )
        return ToolResult(
            success=True,
            content=(
                f"Downloaded '{name}' from Google Drive.\n"
                f"  File ID: {file_id}\n"
                f"  Path: {dest}\n"
                f"  Type: {kind}\n"
                f"  Size: {self._format_size(len(body))}\n"
                f"Use {reader}(path=\"{dest}\") to view it."
                f"{edit_note}"
            ),
        )

    @staticmethod
    def _download_dest(
        output_path: str | None, default_name: str, export_ext: str, runtime: dict[str, Any],
    ) -> Path:
        """Where a download lands: always inside the saved/ area.

        Default ``saved/downloads/<session>/<name>``. A relative output_path is
        scoped like the write tool's (category + session); an absolute one must
        already sit under saved/. ``..`` and anything resolving outside
        (symlinks included) raise :class:`_DownloadPathError`.
        """
        from captain_claw.tools.write import WriteTool

        saved_root = WriteTool._resolve_saved_root(runtime)
        session_id = WriteTool._normalize_session_id(str(runtime.get("_session_id") or ""))
        requested = str(output_path or "").strip()

        if not requested:
            dest = saved_root / "downloads" / session_id / default_name
        else:
            req = Path(requested).expanduser()
            if ".." in req.parts:
                raise _DownloadPathError(
                    f"output_path may not contain '..' (got {requested!r}). "
                    f"Omit it to save under saved/downloads/{session_id}/."
                )
            names_dir = requested.endswith(("/", "\\"))
            if req.is_absolute():
                dest = req
            else:
                # Scoped like the write tool: <category>/<session>/...,
                # anything outside a category under downloads/.
                parts = [p for p in req.parts if p not in ("", ".")]
                if parts and parts[0].lower() == "saved":
                    parts = parts[1:]
                if not parts or parts[0] not in _SAVED_CATEGORIES:
                    parts = ["downloads", *parts]
                if len(parts) < 2 or parts[1] != session_id:
                    parts = [parts[0], session_id, *parts[1:]]
                names_dir = names_dir or len(parts) == 2  # no file name given
                dest = saved_root.joinpath(*parts)
            if names_dir or dest.is_dir():
                dest = dest / default_name
            elif export_ext and dest.suffix.lower() != export_ext:
                dest = dest.with_name(dest.name + export_ext)

        resolved = dest.resolve()
        try:
            resolved.relative_to(saved_root)
        except ValueError:
            raise _DownloadPathError(
                f"output_path must be inside the saved area ({saved_root}); got "
                f"{requested!r}. Omit it to save under saved/downloads/{session_id}/."
            ) from None
        return resolved

    async def _fetch_capped(
        self, token: str, url: str, params: dict[str, str],
    ) -> bytes | None:
        """GET *url* as a stream; None once it passes the download cap.

        Exports carry no size in their metadata, so the cap is enforced on
        the bytes as they arrive rather than after buffering them all.
        """
        async with self._client.stream(
            "GET", url, params=params, headers=self._auth_headers(token),
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()  # so the error handler can read the API message
                resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    return None
                chunks.append(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Action: upload
    # ------------------------------------------------------------------

    async def _action_upload(
        self,
        token: str,
        local_path: str = "",
        name: str | None = None,
        folder_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Upload a local file to Google Drive."""
        if not local_path:
            return ToolResult(success=False, error="local_path is required for upload action.")

        file_path = Path(local_path).expanduser().resolve()
        if not file_path.exists():
            return ToolResult(success=False, error=f"Local file not found: {local_path}")
        if not file_path.is_file():
            return ToolResult(success=False, error=f"Not a file: {local_path}")

        file_name = name or file_path.name
        file_bytes = file_path.read_bytes()

        # Guess MIME type.
        import mimetypes as mt
        content_type = mt.guess_type(str(file_path))[0] or "application/octet-stream"

        metadata: dict[str, Any] = {"name": file_name}
        if folder_id:
            metadata["parents"] = [folder_id]

        result = await self._multipart_upload(token, metadata, file_bytes, content_type)
        uploaded_id = result.get("id", "?")
        uploaded_name = result.get("name", file_name)
        link = result.get("webViewLink", "")

        msg = f"Uploaded '{uploaded_name}' to Google Drive.\n  ID: {uploaded_id}"
        if link:
            msg += f"\n  Link: {link}"
        return ToolResult(success=True, content=msg)

    # ------------------------------------------------------------------
    # Action: create
    # ------------------------------------------------------------------

    async def _action_create(
        self,
        token: str,
        name: str = "",
        content: str = "",
        mime_type: str | None = None,
        folder_id: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Create a new file on Google Drive with the given content."""
        if not name:
            return ToolResult(success=False, error="name is required for create action.")

        target_mime = mime_type or "application/vnd.google-apps.document"

        # For Google Docs/Sheets, create with conversion.
        is_google_type = target_mime.startswith("application/vnd.google-apps.")

        if is_google_type:
            # Create by uploading plain text and converting.
            upload_mime = "text/plain"
            if "spreadsheet" in target_mime:
                upload_mime = "text/csv"

            metadata: dict[str, Any] = {"name": name, "mimeType": target_mime}
            if folder_id:
                metadata["parents"] = [folder_id]

            content_bytes = (content or "").encode("utf-8")
            result = await self._multipart_upload(
                token, metadata, content_bytes, upload_mime,
                convert=True,
            )
        else:
            # Plain file — upload as-is.
            metadata = {"name": name}
            if folder_id:
                metadata["parents"] = [folder_id]
            content_bytes = (content or "").encode("utf-8")
            result = await self._multipart_upload(
                token, metadata, content_bytes, target_mime,
            )

        created_id = result.get("id", "?")
        created_name = result.get("name", name)
        link = result.get("webViewLink", "")

        msg = f"Created '{created_name}' on Google Drive.\n  ID: {created_id}\n  Type: {target_mime}"
        if link:
            msg += f"\n  Link: {link}"
        return ToolResult(success=True, content=msg)

    # ------------------------------------------------------------------
    # Action: update
    # ------------------------------------------------------------------

    async def _action_update(
        self,
        token: str,
        file_id: str = "",
        content: str | None = None,
        local_path: str | None = None,
        overwrite: Any = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Update an existing file's content."""
        if not file_id:
            return ToolResult(success=False, error="file_id is required for update action.")

        if content is not None:
            body = content.encode("utf-8")
            upload_mime = "text/plain"
        elif local_path:
            path = Path(local_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                return ToolResult(success=False, error=f"Local file not found: {local_path}")
            body = path.read_bytes()
            import mimetypes as mt
            upload_mime = mt.guess_type(str(path))[0] or "application/octet-stream"
        else:
            return ToolResult(
                success=False,
                error="Either 'content' or 'local_path' is required for update action.",
            )

        # A native Doc/Sheet/Slides is replaced WHOLE by a media update (every
        # tab, formula and format) — only on an explicit overwrite.
        meta = await self._get_file_metadata(token, file_id)
        mime = meta.get("mimeType", "")
        name = meta.get("name") or file_id
        if mime == FOLDER_MIME:
            return ToolResult(success=False, error=f"'{name}' is a folder; update changes a file's content.")
        if mime.startswith("application/vnd.google-apps.") and not _truthy(overwrite, False):
            parts = [
                f"Refused: '{name}' is a native Google file ({mime}), and update "
                "would replace its WHOLE content (every tab, formula and format)."
            ]
            if mime == _SHEET_MIME:
                parts.append(
                    "Edit it in place instead: sheet_read (cell addresses), then "
                    f"sheet_update / sheet_append / sheet_clear with file_id='{file_id}'."
                )
            elif mime == _DOC_MIME:
                parts.append(
                    "Edit it in place instead: doc_read (exact text), then "
                    f"doc_replace_text / doc_append_text / doc_insert_text with file_id='{file_id}'."
                )
            parts.append(
                "Only if the user explicitly wants the entire file replaced, repeat "
                "update with overwrite=true."
            )
            return ToolResult(success=False, error=" ".join(parts))

        resp = await self._client.patch(
            f"{_UPLOAD_API}/files/{file_id}",
            params={"uploadType": "media", "supportsAllDrives": "true"},
            headers={
                **self._auth_headers(token),
                "Content-Type": upload_mime,
            },
            content=body,
        )
        resp.raise_for_status()
        result = resp.json()

        updated_name = result.get("name", file_id)
        return ToolResult(
            success=True,
            content=f"Updated '{updated_name}' on Google Drive.\n  ID: {file_id}",
        )

    # ------------------------------------------------------------------
    # In place: native Google Sheets / Docs (same file id, no re-upload)
    # ------------------------------------------------------------------

    async def _native_target(
        self, token: str, file_id: str, kind: str, action: str,
    ) -> tuple[dict[str, Any], ToolResult | None]:
        """The file's metadata, or the refusal when it is not a native Google
        Sheet / Doc (*kind*) — checked before any Sheets/Docs request."""
        if not file_id:
            return {}, ToolResult(
                success=False,
                error=f"file_id is required for {action} (the id or the file's URL).",
            )
        label, native_mime, actions = _IN_PLACE_KINDS[kind]
        meta = await self._get_file_metadata(token, file_id)
        mime = meta.get("mimeType", "")
        if mime == native_mime:
            return meta, None
        name = meta.get("name") or file_id
        other = next((k for k, v in _IN_PLACE_KINDS.items() if v[1] == mime), None)
        if other:
            other_label, _, other_actions = _IN_PLACE_KINDS[other]
            error = f"'{name}' is a {other_label}, not a {label}: use {other_actions}."
        elif mime.startswith("application/vnd.google-apps."):
            error = f"'{name}' ({mime}) is not a {label}; {actions} work on {label}s only."
        else:
            # .xlsx / .docx / .csv kept as files on Drive: the Sheets/Docs APIs
            # can't open them, but a media update keeps the same file id.
            error = (
                f"'{name}' is a {mime or 'binary'} file stored on Drive, not a native "
                f"{label}, so it can't be edited in place. To change it and keep "
                f"the same file id: download it, edit the local copy, then "
                f"google_drive(action='update', file_id='{file_id}', "
                "local_path='<edited copy>')."
            )
        return meta, ToolResult(success=False, error=error)

    @staticmethod
    def _range_arg(kwargs: dict[str, Any]) -> str:
        return str(kwargs.get("range") or "").strip()

    def _sheet_url(self, file_id: str, a1: str = "", suffix: str = "") -> str:
        """A Sheets values endpoint; the A1 range travels encoded in the path."""
        base = f"{_SHEETS_API}/{quote(file_id, safe='')}"
        if not a1:
            return base + suffix
        return f"{base}/values/{quote(a1, safe='')}{suffix}"

    async def _action_sheet_read(
        self,
        token: str,
        file_id: str = "",
        render: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Tabs + a preview of each, or one range, with A1 row/column labels."""
        choice = str(render or "values").strip().lower()
        value_render = {"values": "FORMATTED_VALUE", "formulas": "FORMULA"}.get(choice)
        if value_render is None:
            return ToolResult(success=False, error="render must be 'values' or 'formulas'.")
        a1 = self._range_arg(kwargs)
        meta, refusal = await self._native_target(token, file_id, "sheet", "sheet_read")
        if refusal:
            return refusal
        name = meta.get("name") or file_id
        header = f"[Google Sheet: {name} — file id {file_id}]\n"

        if a1:
            resp = await self._client.get(
                self._sheet_url(file_id, a1),
                params={"valueRenderOption": value_render, "majorDimension": "ROWS"},
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            data = resp.json()
            echoed = str(data.get("range") or a1)
            grid, left_out = _render_grid(data.get("values") or [], echoed, _SHEET_READ_MAX_ROWS)
            lines = [
                header + f"[Range {echoed} ({choice}); rows numbered and columns lettered as in the Sheet]",
                "",
                grid,
            ]
            if left_out:
                lines.append(
                    f"\n[{left_out} more rows not shown — read a narrower range for them.]"
                )
            content = "\n".join(lines) + self._sheet_edit_hint(file_id)
            return ToolResult(success=True, content=self._capped(content))

        resp = await self._client.get(
            self._sheet_url(file_id),
            params={"fields": (
                "sheets.properties(sheetId,title,index,sheetType,"
                "gridProperties(rowCount,columnCount))"
            )},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        tabs = [
            s.get("properties") or {}
            for s in resp.json().get("sheets") or []
            if isinstance(s, dict)
        ]
        lines = [header + f"Tabs ({len(tabs)}):"]
        for tab in tabs:
            grid_props = tab.get("gridProperties") or {}
            rows, cols = int(grid_props.get("rowCount") or 0), int(grid_props.get("columnCount") or 0)
            size = f"{rows} rows × {cols} columns (A–{_col_letters(cols)})" if cols else "no grid"
            lines.append(f"  - {tab.get('title', '?')} — {size}")

        def preview_rows(tab: dict[str, Any]) -> int:
            # Never past the tab's grid: a range beyond its last row is a 400
            # ("exceeds grid limits") that would sink the whole batchGet.
            count = (tab.get("gridProperties") or {}).get("rowCount")
            return _SHEET_PREVIEW_ROWS if count is None else min(_SHEET_PREVIEW_ROWS, int(count or 0))

        grid_tabs = [
            t for t in tabs
            if str(t.get("sheetType") or "GRID") == "GRID" and t.get("title") is not None
            and preview_rows(t) > 0
        ][:_SHEET_PREVIEW_TABS]
        if grid_tabs:
            ranges = [f"{_quote_tab(str(t['title']))}!1:{preview_rows(t)}" for t in grid_tabs]
            resp = await self._client.get(
                self._sheet_url(file_id, suffix="/values:batchGet"),
                params=[("ranges", r) for r in ranges] + [
                    ("valueRenderOption", value_render), ("majorDimension", "ROWS"),
                ],
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            value_ranges = resp.json().get("valueRanges") or []
            for tab, vr in zip(grid_tabs, value_ranges):
                echoed = str(vr.get("range") or "")
                grid, _ = _render_grid(vr.get("values") or [], echoed, _SHEET_PREVIEW_ROWS)
                lines += ["", f"## Tab: {tab['title']} (first {_SHEET_PREVIEW_ROWS} rows, {choice})", grid]
        if len(tabs) > len(grid_tabs):
            lines.append(
                f"\n[Previewed {len(grid_tabs)} of {len(tabs)} tabs; sheet_read with "
                "range='<tab name>' shows another.]"
            )
        content = "\n".join(lines) + self._sheet_edit_hint(file_id)
        return ToolResult(success=True, content=self._capped(content))

    @staticmethod
    def _sheet_edit_hint(file_id: str) -> str:
        return (
            "\n\n[Cell addresses are column letter + row number as shown (e.g. B2). "
            f"Write cells in place: sheet_update(file_id='{file_id}', range='<Tab>!B2', "
            "values=[[...]]); add rows: sheet_append; empty cells: sheet_clear.]"
        )

    @staticmethod
    def _capped(text: str) -> str:
        if len(text) > _MAX_READ_BYTES:
            return text[:_MAX_READ_BYTES] + "\n\n... [content truncated — read a narrower range]"
        return text

    async def _action_sheet_update(
        self,
        token: str,
        file_id: str = "",
        values: Any = None,
        updates: Any = None,
        value_input: Any = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Write cells: values.update for one range, values:batchUpdate for several."""
        option = _value_input_option(value_input)
        if option is None:
            return ToolResult(success=False, error="value_input must be 'user_entered' or 'raw'.")
        a1 = self._range_arg(kwargs)
        data: list[dict[str, Any]] = []
        if updates not in (None, "", []):
            data, err = _coerce_updates(updates)
            if err:
                return ToolResult(success=False, error=err)
        if a1 or values is not None:
            if not a1:
                return ToolResult(success=False, error="range is required with values (e.g. 'Sheet1!B2').")
            rows, err = _coerce_rows(values)
            if err:
                return ToolResult(success=False, error=err)
            data.append({"range": a1, "majorDimension": "ROWS", "values": rows})
        if not data:
            return ToolResult(success=False, error=(
                "sheet_update needs range + values, or updates=[{range, values}, ...], "
                f"e.g. range='Sheet1!A1', values={_VALUES_EXAMPLE}."
            ))
        meta, refusal = await self._native_target(token, file_id, "sheet", "sheet_update")
        if refusal:
            return refusal

        if len(data) == 1:
            resp = await self._client.put(
                self._sheet_url(file_id, data[0]["range"]),
                params={"valueInputOption": option},
                json=data[0],
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            answers = [resp.json()]
        else:
            resp = await self._client.post(
                self._sheet_url(file_id, suffix="/values:batchUpdate"),
                json={"valueInputOption": option, "data": data},
                headers=self._auth_headers(token),
            )
            resp.raise_for_status()
            answers = resp.json().get("responses") or []

        name = meta.get("name") or file_id
        lines = [f"Updated '{name}' in place (Google Sheet, file id {file_id}):"]
        total = 0
        for entry, answer in zip(data, answers or [{}] * len(data)):
            cells = int(answer.get("updatedCells") or 0)
            total += cells
            lines.append(f"  {answer.get('updatedRange') or entry['range']} — {cells} cell(s)")
        how = "stored as given" if option == "RAW" else "entered as typed (formulas, numbers and dates parsed)"
        lines.append(f"{total} cell(s) {how}. Same file id; no copy was made.")
        return ToolResult(success=True, content="\n".join(lines))

    async def _action_sheet_append(
        self,
        token: str,
        file_id: str = "",
        values: Any = None,
        value_input: Any = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Add rows below the table in a range (values:append, INSERT_ROWS)."""
        option = _value_input_option(value_input)
        if option is None:
            return ToolResult(success=False, error="value_input must be 'user_entered' or 'raw'.")
        a1 = self._range_arg(kwargs)
        if not a1:
            return ToolResult(success=False, error=(
                "range is required for sheet_append: the tab, or the table's columns "
                "on it (e.g. 'Sheet1' or 'Sheet1!A:F') — rows go below its last row."
            ))
        rows, err = _coerce_rows(values)
        if err:
            return ToolResult(success=False, error=err)
        meta, refusal = await self._native_target(token, file_id, "sheet", "sheet_append")
        if refusal:
            return refusal

        resp = await self._client.post(
            self._sheet_url(file_id, a1, ":append"),
            params={"valueInputOption": option, "insertDataOption": "INSERT_ROWS"},
            json={"range": a1, "majorDimension": "ROWS", "values": rows},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        data = resp.json()
        upd = data.get("updates") or {}
        name = meta.get("name") or file_id
        lines = [
            f"Appended {int(upd.get('updatedRows') or len(rows))} row(s) to '{name}' in place "
            f"(Google Sheet, file id {file_id}).",
            f"  Written: {upd.get('updatedRange') or a1} — {int(upd.get('updatedCells') or 0)} cell(s)",
        ]
        if data.get("tableRange"):
            lines.append(f"  Table found at: {data['tableRange']}")
        return ToolResult(success=True, content="\n".join(lines))

    async def _action_sheet_clear(
        self,
        token: str,
        file_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Empty a range's values (formatting stays)."""
        a1 = self._range_arg(kwargs)
        if not a1:
            return ToolResult(success=False, error="range is required for sheet_clear (e.g. 'Sheet1!B2:D9').")
        meta, refusal = await self._native_target(token, file_id, "sheet", "sheet_clear")
        if refusal:
            return refusal
        resp = await self._client.post(
            self._sheet_url(file_id, a1, ":clear"),
            json={},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        cleared = resp.json().get("clearedRange") or a1
        name = meta.get("name") or file_id
        return ToolResult(
            success=True,
            content=(
                f"Cleared {cleared} in '{name}' in place (Google Sheet, file id "
                f"{file_id}). Values only — formatting is kept."
            ),
        )

    async def _get_doc(self, token: str, file_id: str) -> dict[str, Any]:
        """documents.get with every tab's content."""
        resp = await self._client.get(
            f"{_DOCS_API}/{quote(file_id, safe='')}",
            params={"includeTabsContent": "true"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        return resp.json()

    async def _doc_batch_update(
        self, token: str, file_id: str, requests: list[dict[str, Any]], revision: str = "",
    ) -> dict[str, Any]:
        """documents.batchUpdate; with *revision*, refused if the Doc changed since."""
        body: dict[str, Any] = {"requests": requests}
        if revision:
            body["writeControl"] = {"requiredRevisionId": revision}
        resp = await self._client.post(
            f"{_DOCS_API}/{quote(file_id, safe='')}:batchUpdate",
            json=body,
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _pick_tab(
        doc: dict[str, Any], tab: Any,
    ) -> tuple[list[dict[str, Any]], str]:
        """The tabs a doc_* call works in: all of them, or the one *tab* names
        (id, or title ignoring case). Returns the tabs, or an error."""
        tabs = _doc_tabs(doc)
        wanted = str(tab or "").strip()
        if not wanted:
            return tabs, ""
        for t in tabs:
            tab_id, title = _tab_props(t)
            if wanted == tab_id or wanted.lower() == title.lower():
                return [t], ""
        titles = ", ".join(repr(_tab_props(t)[1]) for t in tabs) or "none"
        return [], f"No tab {wanted!r} in this Doc. Its tabs: {titles}."

    async def _action_doc_read(
        self,
        token: str,
        file_id: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """The exact text of every tab — what doc_replace_text / doc_insert_text match."""
        meta, refusal = await self._native_target(token, file_id, "doc", "doc_read")
        if refusal:
            return refusal
        doc = await self._get_doc(token, file_id)
        title = doc.get("title") or meta.get("name") or file_id
        tabs = _doc_tabs(doc)
        parts = [
            f"[Google Doc: {title} — file id {file_id}]\n"
            "[Exact text: copy find / after strings from here verbatim. [image] "
            "marks an image and table rows are drawn as | cell | cell | — neither "
            "is text you can match.]"
        ]
        for tab in tabs:
            tab_id, tab_title = _tab_props(tab)
            if len(tabs) > 1:
                parts.append(f"\n## Tab: {tab_title} (tab id {tab_id})")
            parts.append("\n" + _doc_text(_tab_content(tab)))
        parts.append(
            "\n[Edit in place: doc_replace_text (find → replace_with), "
            "doc_append_text (new paragraph at the end), doc_insert_text (text "
            "right after `after`).]"
        )
        return ToolResult(success=True, content=self._capped("\n".join(parts)))

    async def _action_doc_replace_text(
        self,
        token: str,
        file_id: str = "",
        find: str | None = None,
        replace_with: str | None = None,
        match_case: Any = None,
        tab: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """replaceAllText: every occurrence of *find*, in place."""
        if not find:
            return ToolResult(success=False, error=(
                "find is required for doc_replace_text: the exact text to replace, "
                "copied from doc_read."
            ))
        if replace_with is None:
            return ToolResult(success=False, error="replace_with is required ('' deletes the found text).")
        meta, refusal = await self._native_target(token, file_id, "doc", "doc_replace_text")
        if refusal:
            return refusal
        request: dict[str, Any] = {
            "containsText": {"text": str(find), "matchCase": _truthy(match_case, True)},
            "replaceText": str(replace_with),
        }
        if str(tab or "").strip():
            tabs, err = self._pick_tab(await self._get_doc(token, file_id), tab)
            if err:
                return ToolResult(success=False, error=err)
            request["tabsCriteria"] = {"tabIds": [_tab_props(tabs[0])[0]]}
        answer = await self._doc_batch_update(token, file_id, [{"replaceAllText": request}])
        replies = answer.get("replies") or [{}]
        changed = int(((replies[0] or {}).get("replaceAllText") or {}).get("occurrencesChanged") or 0)
        name = meta.get("name") or file_id
        if not changed:
            return ToolResult(success=False, error=(
                f"No occurrence of {find!r} in '{name}' — nothing changed. doc_read the "
                "Doc and copy the exact text (spaces, punctuation, case; within one "
                "paragraph) into find."
            ))
        return ToolResult(
            success=True,
            content=(
                f"Replaced {changed} occurrence(s) of {find!r} with {str(replace_with)!r} "
                f"in '{name}' in place (Google Doc, file id {file_id})."
            ),
        )

    async def _action_doc_append_text(
        self,
        token: str,
        file_id: str = "",
        text: str | None = None,
        tab: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Add *text* as a new paragraph at the end of the Doc (or of a tab)."""
        if not text:
            return ToolResult(success=False, error="text is required for doc_append_text.")
        meta, refusal = await self._native_target(token, file_id, "doc", "doc_append_text")
        if refusal:
            return refusal
        doc = await self._get_doc(token, file_id)
        tabs, err = self._pick_tab(doc, tab)
        if err:
            return ToolResult(success=False, error=err)
        target = tabs[0] if tabs else {}
        tab_id, tab_title = _tab_props(target)
        # The end of a body is its last paragraph's newline: insert before it,
        # opening a new paragraph unless the last one is empty.
        body = "".join(run for _, run in _doc_runs(_tab_content(target)))
        insert = str(text)
        if not insert.startswith("\n") and body.strip("\n") and not body.endswith("\n\n"):
            insert = "\n" + insert
        insert = insert[:-1] if insert.endswith("\n") and len(insert) > 1 else insert
        location: dict[str, Any] = {"segmentId": ""}
        if tab_id:
            location["tabId"] = tab_id
        await self._doc_batch_update(
            token, file_id,
            [{"insertText": {"endOfSegmentLocation": location, "text": insert}}],
            revision=str(doc.get("revisionId") or ""),
        )
        name = meta.get("name") or file_id
        where = f" (tab {tab_title!r})" if tab_title and len(_doc_tabs(doc)) > 1 else ""
        return ToolResult(
            success=True,
            content=(
                f"Appended {len(str(text))} characters to the end of '{name}'{where} "
                f"in place (Google Doc, file id {file_id})."
            ),
        )

    async def _action_doc_insert_text(
        self,
        token: str,
        file_id: str = "",
        text: str | None = None,
        after: str | None = None,
        match_case: Any = None,
        tab: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Insert *text* right after the one occurrence of *after*."""
        if not text:
            return ToolResult(success=False, error="text is required for doc_insert_text.")
        if not after:
            return ToolResult(success=False, error=(
                "after is required for doc_insert_text: existing text, copied exactly "
                "from doc_read, that the new text goes right after."
            ))
        meta, refusal = await self._native_target(token, file_id, "doc", "doc_insert_text")
        if refusal:
            return refusal
        doc = await self._get_doc(token, file_id)
        tabs, err = self._pick_tab(doc, tab)
        if err:
            return ToolResult(success=False, error=err)
        hits = [
            (t, end)
            for t in tabs
            for end in _anchor_ends(_tab_content(t), str(after), _truthy(match_case, True))
        ]
        name = meta.get("name") or file_id
        if not hits:
            return ToolResult(success=False, error=(
                f"{after!r} was not found in '{name}' — nothing changed. doc_read the "
                "Doc and copy the exact text (spaces, punctuation, case) into after."
            ))
        if len(hits) > 1:
            return ToolResult(success=False, error=(
                f"{after!r} occurs {len(hits)} times in '{name}' — nothing changed. "
                "Make after longer so it names exactly one place (or pass tab)."
            ))
        target, index = hits[0]
        tab_id, _ = _tab_props(target)
        content = _tab_content(target)
        body_end = int((content[-1] or {}).get("endIndex") or 0) if content else 0
        insert = str(text)
        spot: dict[str, Any] = {"segmentId": ""}
        if tab_id:
            spot["tabId"] = tab_id
        if body_end and index >= body_end:
            # Past the body's final newline: nothing can go there, so the text
            # opens a new last paragraph at the end of the body instead.
            if not insert.startswith("\n"):
                insert = "\n" + insert
            if insert.endswith("\n") and len(insert) > 1:
                insert = insert[:-1]
            request = {"endOfSegmentLocation": spot, "text": insert}
        else:
            request = {"location": {**spot, "index": index}, "text": insert}
        await self._doc_batch_update(
            token, file_id, [{"insertText": request}],
            revision=str(doc.get("revisionId") or ""),
        )
        return ToolResult(
            success=True,
            content=(
                f"Inserted {len(str(text))} characters after {after[-60:]!r} in "
                f"'{name}' in place (Google Doc, file id {file_id})."
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_file_metadata(self, token: str, file_id: str) -> dict[str, Any]:
        """Fetch file metadata from the Drive API."""
        resp = await self._client.get(
            f"{_DRIVE_API}/files/{file_id}",
            params={"fields": _FILE_FIELDS, "supportsAllDrives": "true"},
            headers=self._auth_headers(token),
        )
        resp.raise_for_status()
        return resp.json()

    async def _multipart_upload(
        self,
        token: str,
        metadata: dict[str, Any],
        content: bytes,
        content_type: str,
        convert: bool = False,
    ) -> dict[str, Any]:
        """Upload a file using multipart upload."""
        boundary = f"captain_claw_{uuid.uuid4().hex[:16]}"

        # Build multipart body per Google Drive API spec.
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(b"Content-Type: application/json; charset=UTF-8\r\n\r\n")
        body.write(json.dumps(metadata).encode("utf-8"))
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.write(content)
        body.write(f"\r\n--{boundary}--\r\n".encode())

        params: dict[str, str] = {
            "uploadType": "multipart",
            "supportsAllDrives": "true",
            "fields": "id,name,mimeType,webViewLink",
        }

        resp = await self._client.post(
            f"{_UPLOAD_API}/files",
            params=params,
            headers={
                **self._auth_headers(token),
                "Content-Type": f"multipart/related; boundary={boundary}",
            },
            content=body.getvalue(),
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _get_extract_tool(tool_name: str) -> Any | None:
        """Get an instance of an extract tool by name."""
        try:
            if tool_name == "pdf_extract":
                from captain_claw.tools.document_extract import PdfExtractTool
                return PdfExtractTool()
            elif tool_name == "docx_extract":
                from captain_claw.tools.document_extract import DocxExtractTool
                return DocxExtractTool()
            elif tool_name == "xlsx_extract":
                from captain_claw.tools.document_extract import XlsxExtractTool
                return XlsxExtractTool()
            elif tool_name == "pptx_extract":
                from captain_claw.tools.document_extract import PptxExtractTool
                return PptxExtractTool()
        except Exception:
            return None
        return None

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size into human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
