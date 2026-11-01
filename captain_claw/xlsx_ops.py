"""Excel workbooks (.xlsx) for agents, without scripts.

The promise is correct numbers in a file that is otherwise left alone:

* **Nothing is lost.** A workbook is a zip of XML parts. Only the parts an edit
  really changes are rewritten; charts, pivot tables, sparklines, macros and
  anything we don't know about stay byte for byte. (A load-and-save through
  openpyxl drops what it doesn't model.)
* **Formulas stay formulas and carry their results.** After every change the
  affected formulas are recalculated with :mod:`captain_claw.xlsx_formula` and
  the results stored, so previews and readers show numbers; Excel recalculates
  on open anyway. What we can't calculate is left without a result and reported,
  never guessed, and a stale result is never kept.
* **Every change is reported cell by cell**, including the totals that moved.
* **Moving things keeps references right.** Inserting or deleting rows and
  columns, or renaming a sheet, rewrites formulas, names, tables, merges,
  conditional formats, validations, notes, drawings and charts the way Excel does.

Nothing here decides where files live; ``tools/xlsx_tool.py`` owns paths, backups
and saving.
"""

from __future__ import annotations

import copy
import datetime as dt
import io
import math
import posixpath
import re
import time
import warnings
import zipfile
from bisect import bisect_left, insort
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

from lxml import etree
from openpyxl.styles.numbers import BUILTIN_FORMATS
from openpyxl.utils import column_index_from_string, get_column_letter

from captain_claw import xlsx_formula as F
from captain_claw.xlsx_formula import Box, FormulaError, NeedCell, NotComputable, XlError

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCT_NS = "http://purl.org/dc/terms/"
XM_NS = "http://schemas.microsoft.com/office/excel/2006/main"
THREADED_NS = "http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

RT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
CT_WORKBOOK = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
CT_TEMPLATE = "application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml"
CT_MACRO = "application/vnd.ms-excel.sheet.macroEnabled.main+xml"
CT_MACRO_TEMPLATE = "application/vnd.ms-excel.template.macroEnabled.main+xml"
CT_WORKSHEET = "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
CT_STRINGS = "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"
CT_STYLES = "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"
CT_TABLE = "application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml"
CT_DRAWING = "application/vnd.openxmlformats-officedocument.drawing+xml"
CT_CHART = "application/vnd.openxmlformats-officedocument.drawingml.chart+xml"
CT_COMMENTS = "application/vnd.openxmlformats-officedocument.spreadsheetml.comments+xml"
CT_VML = "application/vnd.openxmlformats-officedocument.vmlDrawing"

MAX_UNZIPPED = 600 * 1024 * 1024
MAX_CELLS_PER_CALL = 500_000
MAX_FORMULAS_TO_CALCULATE = 400_000
CALC_SECONDS = 60.0
MAX_CELL_TEXT = 32_767

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True, remove_blank_text=False)
_LENIENT = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True, recover=True)


def _q(tag: str) -> str:
    return f"{{{MAIN}}}{tag}"


C, V, FT, IS, T_TAG, ROW = _q("c"), _q("v"), _q("f"), _q("is"), _q("t"), _q("row")


class XlsxError(ValueError):
    """A request the workbook can't satisfy; the message is for the model."""


_SHEET_ORDER = {name: i for i, name in enumerate(
    """sheetPr dimension sheetViews sheetFormatPr cols sheetData sheetCalcPr sheetProtection protectedRanges
    scenarios autoFilter sortState dataConsolidate customSheetViews mergeCells phoneticPr conditionalFormatting
    dataValidations hyperlinks printOptions pageMargins pageSetup headerFooter rowBreaks colBreaks
    customProperties cellWatches ignoredErrors smartTags drawing legacyDrawing legacyDrawingHF drawingHF picture
    oleObjects controls webPublishItems tableParts extLst""".split())}
_WORKBOOK_ORDER = {name: i for i, name in enumerate(
    """fileVersion fileSharing workbookPr workbookProtection bookViews sheets functionGroups externalReferences
    definedNames calcPr oleSize customWorkbookViews pivotCaches smartTagPr smartTagTypes webPublishing fileRecoveryPr
    webPublishObjects extLst""".split())}


def _local(el: Any) -> str:
    tag = el.tag
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _insert_ordered(root: Any, el: Any, order: dict[str, int]) -> Any:
    """Excel refuses a file whose parts are out of schema order."""
    rank = order.get(_local(el), 999)
    for index, child in enumerate(root):
        # A wrapper Excel adds around oleObjects/controls ranks with them.
        child_rank = order.get(_local(child), order.get("oleObjects", 999))
        if child_rank > rank:
            root.insert(index, el)
            return el
    root.append(el)
    return el


# ── Addresses ────────────────────────────────────────────────────────

_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")
_COL_INDEX: dict[str, int] = {}


def _col_index(letters: str) -> int:
    index = _COL_INDEX.get(letters)
    if index is None:
        index = _COL_INDEX[letters] = column_index_from_string(letters)
    return index


def addr(row: int, col: int) -> str:
    return f"{get_column_letter(col)}{row}"


def box_text(box: Box) -> str:
    if box.c1 == 1 and box.c2 == F.MAX_COL:
        return f"{box.r1}:{box.r2}"
    if box.r1 == 1 and box.r2 == F.MAX_ROW:
        return f"{get_column_letter(box.c1)}:{get_column_letter(box.c2)}"
    if box.is_cell:
        return addr(box.r1, box.c1)
    return f"{addr(box.r1, box.c1)}:{addr(box.r2, box.c2)}"


def parse_range(text: str) -> tuple[str | None, Box]:
    """``Sheet!A1:B2`` → (sheet, box); the model's typing errors become a clear message."""
    raw = str(text or "").strip()
    sheet, body, _ = F.split_sheet_prefix(raw)
    box = F.parse_box(body.replace(" ", ""))
    if box is None:
        raise XlsxError(f"{raw!r} is not a cell or range (use A1, B2:D9, C:C or 3:5).")
    return sheet, box


def _shift_ref_text(text: str, axis: str, at: int, count: int) -> str | None:
    """A plain range after rows or columns moved; None when it's all gone."""
    limit = F.MAX_ROW if axis == "row" else F.MAX_COL
    shifted, _ = F._shift_body(text, axis, at, count, limit)
    return None if shifted == "#REF!" else shifted


def _shift_sqref(text: str, axis: str, at: int, count: int) -> str:
    kept = [s for s in (_shift_ref_text(part, axis, at, count) for part in text.split()) if s]
    return " ".join(kept)


# ── Values ───────────────────────────────────────────────────────────

_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
_ESCAPED = re.compile(r"_x([0-9A-Fa-f]{4})_")


def _escape_text(text: str) -> str:
    """Characters XML can't hold, the way Excel stores them."""
    return _ILLEGAL.sub(lambda m: f"_x{ord(m.group()):04X}_", text)


def _unescape_text(text: str) -> str:
    if "_x" not in text:
        return text
    return _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), text)


def number_text(value: float) -> str:
    """A number as the file stores it."""
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def show_number(value: float) -> str:
    return F.number_to_text(float(value))


def show_value(value: Any, fmt: str = "", date1904: bool = False) -> str:
    """A value for the model: text quoted so ``"12"`` and ``12`` can't be confused,
    a number with what the cell shows, a date as a date."""
    if value is None:
        return "(empty)"
    if isinstance(value, XlError):
        return value.code
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        shown = value if len(value) <= 300 else value[:300] + "…"
        return '"' + shown.replace("\n", "\\n") + '"'
    plain = show_number(value)
    if not fmt or fmt.lower() == "general":
        return plain
    if F.is_date_format(fmt):
        if value >= 0:
            try:
                day = F.serial_to_date(value, date1904)
            except (ValueError, OverflowError):
                return plain
            fraction = value - math.floor(value)
            if fraction == 0:
                return day.isoformat()
            seconds = round(fraction * 86400)
            return f"{day.isoformat()} {seconds // 3600 % 24:02d}:{seconds // 60 % 60:02d}"
        return plain
    formatted = F.format_value(value, fmt, date1904)
    if formatted is None or formatted.strip() == plain:
        return plain
    return f"{plain} ({formatted.strip()})"


_PLAIN_NUMBER = re.compile(r"^-?(?:0|[1-9]\d{0,14})(?:\.\d+)?$")
_PERCENT = re.compile(r"^(-?(?:0|[1-9]\d*)(?:\.\d+)?)\s*%$")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?$")


def _same_value(a: Any, b: Any) -> bool:
    """Same result, ignoring the last bits of a float."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-3)
    return type(a) is type(b) and a == b


# ── Package ──────────────────────────────────────────────────────────


class _Rels:
    """One part's relationships."""

    def __init__(self, book: "Book", owner: str) -> None:
        self.book = book
        self.owner = owner
        folder, name = posixpath.split(owner)
        self.path = posixpath.join(folder, "_rels", name + ".rels")
        self.folder = folder

    def _root(self, create: bool = False) -> Any:
        if self.book.has(self.path):
            return self.book.part(self.path)
        if not create:
            return None
        root = etree.Element(f"{{{PKG_RELS}}}Relationships", nsmap={None: PKG_RELS})
        self.book.add_part(self.path, root, None)
        return root

    def items(self) -> list[tuple[str, str, str]]:
        """(id, type, target part); links outside the package are left out."""
        root = self._root()
        if root is None:
            return []
        out = []
        for rel in root:
            if rel.get("TargetMode") == "External":
                continue
            target = rel.get("Target") or ""
            part = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(self.folder, target))
            out.append((rel.get("Id") or "", rel.get("Type") or "", part))
        return out

    def of_type(self, kind: str) -> list[tuple[str, str]]:
        return [(i, part) for i, t, part in self.items() if t.rsplit("/", 1)[-1] == kind]

    def target(self, rel_id: str) -> str | None:
        return next((part for i, _, part in self.items() if i == rel_id), None)

    def add(self, kind: str, part: str) -> str:
        root = self._root(create=True)
        taken = {rel.get("Id") for rel in root}
        number = len(taken) + 1
        while f"rId{number}" in taken:
            number += 1
        rel_id = f"rId{number}"
        etree.SubElement(root, f"{{{PKG_RELS}}}Relationship", Id=rel_id, Type=RT + kind,
                         Target=posixpath.relpath(part, self.folder or "."))
        self.book.touch(self.path)
        return rel_id

    def remove(self, rel_id: str) -> None:
        root = self._root()
        if root is None:
            return
        for rel in list(root):
            if rel.get("Id") == rel_id:
                root.remove(rel)
                self.book.touch(self.path)


@dataclass(slots=True)
class TableInfo:
    sheet: str
    name: str
    r1: int
    c1: int
    r2: int
    c2: int
    columns: list[str]
    header_rows: int
    totals_rows: int
    part: str

    @property
    def box(self) -> Box:
        return Box(self.sheet, self.r1, self.c1, self.r2, self.c2)


class Sheet:
    """One worksheet, parsed on first use."""

    def __init__(self, book: "Book", el: Any, path: str) -> None:
        self.book = book
        self.el = el
        self.path = path
        self.root: Any = None
        self.data: Any = None
        self.grid: dict[int, dict[int, Any]] = {}
        self.row_els: dict[int, Any] = {}
        self.row_numbers: list[int] = []
        self.formula_els: dict[tuple[int, int], Any] = {}
        self.shared: dict[str, tuple[str, str]] = {}
        self.array_masters: dict[tuple[int, int], Box] = {}
        self.array_owner: dict[tuple[int, int], tuple[int, int]] = {}
        self._translated: dict[tuple[int, int], str] = {}
        self._used: tuple[int, int] | None = None
        self.merges: list[Box] = []

    @property
    def name(self) -> str:
        return self.el.get("name") or ""

    @property
    def state(self) -> str:
        return self.el.get("state") or "visible"

    @property
    def rels(self) -> _Rels:
        return _Rels(self.book, self.path)

    def load(self) -> "Sheet":
        if self.root is None:
            self.root = self.book.part(self.path)
            self.data = self.root.find(_q("sheetData"))
            if self.data is None:
                self.data = _insert_ordered(self.root, etree.Element(_q("sheetData")), _SHEET_ORDER)
            self.reindex()
        return self

    def reindex(self) -> None:
        """Build the lookups from the XML (again after rows or columns moved)."""
        self.grid, self.row_els, self.formula_els = {}, {}, {}
        self.shared, self.array_masters, self.array_owner, self._translated = {}, {}, {}, {}
        self._used = None
        previous_row = 0
        numbers: list[int] = []
        for row in self.data:
            if row.tag != ROW:
                continue
            raw = row.get("r")
            number = int(raw) if raw else previous_row + 1
            if not raw:
                row.set("r", str(number))
            previous_row = number
            numbers.append(number)
            self.row_els[number] = row
            cells: dict[int, Any] = {}
            self.grid[number] = cells
            previous_col = 0
            for cell in row:
                if cell.tag != C:
                    continue
                ref = cell.get("r")
                if ref:
                    match = _REF_RE.match(ref)
                    col = _col_index(match.group(1)) if match else previous_col + 1
                else:
                    col = previous_col + 1
                    cell.set("r", addr(number, col))
                previous_col = col
                cells[col] = cell
                formula = cell.find(FT)
                if formula is not None:
                    self.formula_els[(number, col)] = formula
                    kind = formula.get("t")
                    if kind == "shared" and formula.text:
                        self.shared[formula.get("si") or ""] = (formula.text, addr(number, col))
                    elif kind == "array":
                        ref_text = formula.get("ref") or addr(number, col)
                        box = F.parse_box(ref_text) or Box(None, number, col, number, col)
                        self.array_masters[(number, col)] = box
                        for r in range(box.r1, box.r2 + 1):
                            for c in range(box.c1, box.c2 + 1):
                                self.array_owner[(r, c)] = (number, col)
        self.row_numbers = sorted(numbers)
        self.merges = []
        merge_parent = self.root.find(_q("mergeCells"))
        if merge_parent is not None:
            for merge in merge_parent:
                box = F.parse_box(merge.get("ref") or "")
                if box is not None:
                    self.merges.append(box)

    # Reading

    def cell_el(self, row: int, col: int) -> Any:
        cells = self.grid.get(row)
        return cells.get(col) if cells is not None else None

    def stored(self, row: int, col: int) -> Any:
        cell = self.cell_el(row, col)
        return None if cell is None else self.book.cell_value(cell)

    def formula(self, row: int, col: int) -> str | None:
        """The formula in a cell as stored (shared ones written out); None if there is none."""
        el = self.formula_els.get((row, col))
        if el is None:
            return None
        if el.get("t") == "shared" and not el.text:
            memo = self._translated.get((row, col))
            if memo is not None:
                return memo
            master = self.shared.get(el.get("si") or "")
            if master is None:
                return ""
            try:
                text = F.translate(master[0], master[1], addr(row, col))
            except FormulaError:
                text = ""
            self._translated[(row, col)] = text
            return text
        return el.text or ""

    def is_opaque_formula(self, row: int, col: int) -> bool:
        """What-if data tables are calculated by Excel only."""
        el = self.formula_els.get((row, col))
        return el is not None and el.get("t") == "dataTable"

    def has_stored_result(self, row: int, col: int) -> bool:
        cell = self.cell_el(row, col)
        if cell is None:
            return False
        value = cell.find(V)
        if value is None:
            return False
        return value.text is not None or cell.get("t") == "str"

    def used(self) -> tuple[int, int]:
        """Last row and column holding a cell."""
        if self._used is None:
            rows = cols = 0
            for number, cells in self.grid.items():
                if cells:
                    rows = max(rows, number)
                    cols = max(cols, max(cells))
            self._used = (rows, cols)
        return self._used

    def style_of(self, row: int, col: int) -> int:
        cell = self.cell_el(row, col)
        if cell is not None and cell.get("s"):
            return int(cell.get("s"))
        row_el = self.row_els.get(row)
        if row_el is not None and row_el.get("customFormat") == "1" and row_el.get("s"):
            return int(row_el.get("s"))
        cols = self.root.find(_q("cols"))
        if cols is not None:
            for col_el in cols:
                if int(col_el.get("min", "0")) <= col <= int(col_el.get("max", "0")) and col_el.get("style"):
                    return int(col_el.get("style"))
        return 0

    def number_format(self, row: int, col: int) -> str:
        return self.book.styles.number_format(self.style_of(row, col))

    def merged_at(self, row: int, col: int) -> Box | None:
        for box in self.merges:
            if box.contains(row, col):
                return box
        return None

    def hidden_rows(self) -> list[int]:
        return [n for n, el in self.row_els.items() if el.get("hidden") == "1"]

    def hidden_columns(self) -> list[tuple[int, int]]:
        cols = self.root.find(_q("cols"))
        if cols is None:
            return []
        return [(int(c.get("min", "0")), int(c.get("max", "0"))) for c in cols if c.get("hidden") == "1"]

    def is_protected(self) -> bool:
        protection = self.root.find(_q("sheetProtection"))
        return protection is not None and protection.get("sheet") in ("1", "true")

    # Writing

    def touch(self) -> None:
        self.book.touch(self.path)
        self._used = None

    def row_el(self, row: int) -> Any:
        existing = self.row_els.get(row)
        if existing is not None:
            return existing
        el = etree.Element(ROW, r=str(row))
        position = bisect_left(self.row_numbers, row)
        if position == len(self.row_numbers):
            self.data.append(el)
        else:
            self.row_els[self.row_numbers[position]].addprevious(el)
        insort(self.row_numbers, row)
        self.row_els[row] = el
        self.grid[row] = {}
        return el

    def cell(self, row: int, col: int) -> Any:
        """The cell element, created in place (with the row's or column's format) if missing."""
        existing = self.cell_el(row, col)
        if existing is not None:
            return existing
        inherited = self.style_of(row, col)
        row_el = self.row_el(row)
        cells = self.grid[row]
        el = etree.Element(C, r=addr(row, col))
        if inherited:
            el.set("s", str(inherited))
        later = [c for c in cells if c > col]
        if later:
            cells[min(later)].addprevious(el)
        else:
            row_el.append(el)
        cells[col] = el
        # An out-of-date hint is worse than none.
        row_el.attrib.pop("spans", None)
        self.touch()
        return el

    def clear_content(self, row: int, col: int) -> None:
        cell = self.cell_el(row, col)
        if cell is None:
            return
        _strip_content(cell)
        self.formula_els.pop((row, col), None)
        self._translated.pop((row, col), None)
        self.touch()

    def unshare_formulas(self) -> None:
        """Write every shared formula out: editing or moving them is then simple."""
        if not self.shared and not any(f.get("t") == "shared" for f in self.formula_els.values()):
            return
        for (row, col), el in self.formula_els.items():
            if el.get("t") == "shared":
                text = self.formula(row, col) or ""
                for attr in ("t", "si", "ref"):
                    el.attrib.pop(attr, None)
                el.text = text
        self.shared = {}
        self._translated = {}
        self.touch()


def _strip_content(cell: Any) -> None:
    for child in list(cell):
        if child.tag in (FT, V, IS):
            cell.remove(child)
    for attr in ("t", "cm", "vm"):
        cell.attrib.pop(attr, None)


class Styles:
    """The workbook's formats; a change makes a new format, never edits one in use."""

    def __init__(self, book: "Book") -> None:
        self.book = book
        rel = book.wb_rels.of_type("styles")
        if rel:
            self.path = rel[0][1]
        else:
            self.path = "xl/styles.xml"
            root = etree.fromstring(
                f'<styleSheet xmlns="{MAIN}"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
                '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
                '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
                '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'.encode())
            book.add_part(self.path, root, CT_STYLES)
            book.wb_rels.add("styles", self.path)
        self.root = book.part(self.path)
        self.xfs = self.root.find(_q("cellXfs"))
        self._custom: dict[int, str] | None = None
        self._derived: dict[Any, int] = {}

    def _formats(self) -> dict[int, str]:
        if self._custom is None:
            parent = self.root.find(_q("numFmts"))
            self._custom = {}
            if parent is not None:
                for item in parent:
                    self._custom[int(item.get("numFmtId", "0"))] = item.get("formatCode") or ""
        return self._custom

    def number_format(self, index: int) -> str:
        if self.xfs is None or index >= len(self.xfs):
            return "General"
        number = int(self.xfs[index].get("numFmtId", "0"))
        if number in BUILTIN_FORMATS:
            return BUILTIN_FORMATS[number]
        return self._formats().get(number, "General")

    def _format_id(self, code: str) -> int:
        for number, builtin in BUILTIN_FORMATS.items():
            if builtin == code:
                return number
        formats = self._formats()
        for number, custom in formats.items():
            if custom == code:
                return number
        parent = self.root.find(_q("numFmts"))
        if parent is None:
            parent = etree.Element(_q("numFmts"))
            self.root.insert(0, parent)
        number = max([163, *formats]) + 1
        etree.SubElement(parent, _q("numFmt"), numFmtId=str(number), formatCode=code)
        parent.set("count", str(len(parent)))
        formats[number] = code
        return number

    def _part_like(self, group: str, base_index: int, change: Callable[[Any], None]) -> int:
        """An entry of fonts/fills/borders equal to *base_index* after *change*."""
        parent = self.root.find(_q(group))
        new = copy.deepcopy(parent[base_index] if base_index < len(parent) else parent[0])
        change(new)
        wanted = etree.tostring(new, method="c14n")
        for index, item in enumerate(parent):
            if etree.tostring(item, method="c14n") == wanted:
                return index
        parent.append(new)
        parent.set("count", str(len(parent)))
        return len(parent) - 1

    def derive(self, base: int, **changes: Any) -> int:
        """The format *base* with *changes* applied."""
        key = (base, tuple(sorted(changes.items())))
        found = self._derived.get(key)
        if found is not None:
            return found
        if self.xfs is None:
            raise XlsxError("This workbook has no cell formats to change.")
        xf = copy.deepcopy(self.xfs[base] if base < len(self.xfs) else self.xfs[0])
        if "number_format" in changes:
            xf.set("numFmtId", str(self._format_id(str(changes["number_format"]))))
            xf.set("applyNumberFormat", "1")
        font_keys = {k: changes[k] for k in ("bold", "italic", "underline", "strike", "color", "font", "font_size")
                     if k in changes}
        if font_keys:
            xf.set("fontId", str(self._part_like("fonts", int(xf.get("fontId", "0")),
                                                 lambda el: _change_font(el, font_keys))))
            xf.set("applyFont", "1")
        if "fill" in changes:
            xf.set("fillId", str(self._part_like("fills", int(xf.get("fillId", "0")),
                                                 lambda el: _change_fill(el, changes["fill"]))))
            xf.set("applyFill", "1")
        border_keys = {k: changes[k] for k in ("border", "border_top", "border_bottom", "border_left", "border_right")
                       if k in changes}
        if border_keys:
            xf.set("borderId", str(self._part_like("borders", int(xf.get("borderId", "0")),
                                                   lambda el: _change_border(el, border_keys))))
            xf.set("applyBorder", "1")
        align_keys = {k: changes[k] for k in ("align", "valign", "wrap", "indent") if k in changes}
        if align_keys:
            alignment = xf.find(_q("alignment"))
            if alignment is None:
                alignment = etree.Element(_q("alignment"))
                xf.insert(0, alignment)
            if "align" in align_keys:
                alignment.set("horizontal", str(align_keys["align"]))
            if "valign" in align_keys:
                alignment.set("vertical", str(align_keys["valign"]))
            if "wrap" in align_keys:
                alignment.set("wrapText", "1" if align_keys["wrap"] else "0")
            if "indent" in align_keys:
                alignment.set("indent", str(int(align_keys["indent"])))
            xf.set("applyAlignment", "1")
        self.xfs.append(xf)
        self.xfs.set("count", str(len(self.xfs)))
        index = len(self.xfs) - 1
        self._derived[key] = index
        self.book.touch(self.path)
        return index


_FONT_ORDER = {name: i for i, name in enumerate(
    "b i strike condense extend outline shadow u vertAlign sz color name family charset scheme".split())}


def _color(value: Any) -> str:
    text = str(value or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9A-Fa-f]{6}", text):
        return "FF" + text.upper()
    if re.fullmatch(r"[0-9A-Fa-f]{8}", text):
        return text.upper()
    raise XlsxError(f"{value!r} is not a hex colour (use e.g. C00000).")


def _change_font(font: Any, changes: dict[str, Any]) -> None:
    def replace(tag: str, attrs: dict[str, str] | None) -> None:
        for old in font.findall(_q(tag)):
            font.remove(old)
        if attrs is not None:
            etree.SubElement(font, _q(tag), **attrs)

    for key, tag in (("bold", "b"), ("italic", "i"), ("strike", "strike"), ("underline", "u")):
        if key in changes:
            replace(tag, {} if changes[key] else None)
    if "color" in changes:
        replace("color", {"rgb": _color(changes["color"])})
    if "font_size" in changes:
        replace("sz", {"val": number_text(float(changes["font_size"]))})
    if "font" in changes:
        replace("name", {"val": str(changes["font"])})
        replace("scheme", None)  # a theme font would win over the name
        replace("family", None)
    font[:] = sorted(font, key=lambda el: _FONT_ORDER.get(_local(el), 99))


def _change_fill(fill: Any, value: Any) -> None:
    for child in list(fill):
        fill.remove(child)
    if value in (None, "", "none"):
        etree.SubElement(fill, _q("patternFill"), patternType="none")
        return
    pattern = etree.SubElement(fill, _q("patternFill"), patternType="solid")
    etree.SubElement(pattern, _q("fgColor"), rgb=_color(value))
    etree.SubElement(pattern, _q("bgColor"), indexed="64")


_BORDER_STYLES = {"thin", "medium", "thick", "double", "dashed", "dotted", "hair"}


def _change_border(border: Any, changes: dict[str, Any]) -> None:
    def side(tag: str, value: Any) -> None:
        el = border.find(_q(tag))
        if el is None:
            el = etree.SubElement(border, _q(tag))
        for child in list(el):
            el.remove(child)
        style = str(value or "none").lower()
        if style in ("none", "false", ""):
            el.attrib.pop("style", None)
            return
        if style == "true":
            style = "thin"
        if style not in _BORDER_STYLES:
            raise XlsxError(f"border {value!r}: use thin, medium, thick, double, dashed, dotted, hair or none.")
        el.set("style", style)
        etree.SubElement(el, _q("color"), auto="1")

    if "border" in changes:
        for tag in ("left", "right", "top", "bottom"):
            side(tag, changes["border"])
    for key, value in changes.items():
        if key.startswith("border_"):
            side(key.split("_", 1)[1], value)
    order = {"start": 0, "left": 0, "end": 1, "right": 1, "top": 2, "bottom": 3, "diagonal": 4}
    border[:] = sorted(border, key=lambda el: order.get(_local(el), 9))


class Book:
    """A workbook as its package: only parts that really changed are written back."""

    def __init__(self, data: bytes) -> None:
        if data[:4] == b"\xd0\xcf\x11\xe0":
            raise XlsxError("This is a password-protected workbook or an old .xls file. "
                            "Ask for an unprotected copy saved as .xlsx.")
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise XlsxError(f"Not a readable Excel workbook ({exc}).") from None
        if sum(info.file_size for info in archive.infolist()) > MAX_UNZIPPED:
            raise XlsxError("This workbook is too large to open here (over 600 MB unpacked).")
        self._raw: dict[str, bytes] = {}
        self._order: list[str] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            self._raw[info.filename] = archive.read(info.filename)
            self._order.append(info.filename)
        self._parsed: dict[str, Any] = {}
        self._dirty: set[str] = set()
        self._new_types: dict[str, str] = {}
        self._parse_workbook()

    @classmethod
    def blank(cls) -> "Book":
        from openpyxl import Workbook

        buffer = io.BytesIO()
        Workbook().save(buffer)
        return cls(buffer.getvalue())

    # Parts

    def has(self, name: str) -> bool:
        return name in self._raw or name in self._parsed

    def raw(self, name: str) -> bytes:
        return self._raw[name]

    def part(self, name: str) -> Any:
        found = self._parsed.get(name)
        if found is None:
            if name not in self._raw:
                raise XlsxError(f"The workbook is damaged: part {name} is missing.")
            try:
                found = etree.fromstring(self._raw[name], _LENIENT if name.endswith(".vml") else _PARSER)
            except etree.XMLSyntaxError as exc:
                raise XlsxError(f"The workbook is damaged: {name} can't be read ({exc}).") from None
            self._parsed[name] = found
        return found

    def touch(self, name: str) -> None:
        self._dirty.add(name)

    def parts_named(self, pattern: str) -> list[str]:
        regex = re.compile(pattern)
        return sorted(n for n in {*self._raw, *self._parsed} if regex.fullmatch(n))

    def add_part(self, name: str, root: Any, content_type: str | None) -> None:
        if name not in self._order:
            self._order.append(name)
        self._parsed[name] = root
        self._dirty.add(name)
        if content_type is not None:
            self._set_override(name, content_type)

    def remove_part(self, name: str) -> None:
        self._raw.pop(name, None)
        self._parsed.pop(name, None)
        self._dirty.discard(name)
        if name in self._order:
            self._order.remove(name)
        types = self.part("[Content_Types].xml")
        for item in list(types):
            if item.get("PartName") == "/" + name:
                types.remove(item)
                self.touch("[Content_Types].xml")
        rels = _Rels(self, name)
        if self.has(rels.path):
            self.remove_part(rels.path)

    def _set_override(self, name: str, content_type: str) -> None:
        types = self.part("[Content_Types].xml")
        for item in types:
            if item.get("PartName") == "/" + name:
                item.set("ContentType", content_type)
                self.touch("[Content_Types].xml")
                return
        etree.SubElement(types, f"{{{CT_NS}}}Override", PartName="/" + name, ContentType=content_type)
        self.touch("[Content_Types].xml")

    def _ensure_default(self, extension: str, content_type: str) -> None:
        types = self.part("[Content_Types].xml")
        if any(item.get("Extension", "").lower() == extension for item in types):
            return
        types.insert(0, etree.Element(f"{{{CT_NS}}}Default", Extension=extension, ContentType=content_type))
        self.touch("[Content_Types].xml")

    def content_type(self, name: str) -> str:
        types = self.part("[Content_Types].xml")
        for item in types:
            if item.get("PartName") == "/" + name:
                return item.get("ContentType") or ""
        return ""

    def _next_name(self, pattern: str) -> tuple[str, int]:
        """An unused part name: ``xl/charts/chart{}.xml`` → (name, number)."""
        number = 1
        while self.has(pattern.format(number)):
            number += 1
        return pattern.format(number), number

    # Workbook

    def _parse_workbook(self) -> None:
        if "[Content_Types].xml" not in self._raw:
            raise XlsxError("Not a readable Excel workbook (it has no package index).")
        root_rels = _Rels(self, "")
        root_rels.path, root_rels.folder = "_rels/.rels", ""
        main = next((part for _, kind, part in root_rels.items() if kind.endswith("/officeDocument")), None)
        if main is None or not self.has(main):
            raise XlsxError("Not a readable Excel workbook (it has no workbook part).")
        self.wb_path = main
        kind = self.content_type(main)
        if kind not in (CT_WORKBOOK, CT_TEMPLATE, CT_MACRO, CT_MACRO_TEMPLATE):
            if "binary" in kind or main.endswith(".bin"):
                raise XlsxError("This is a binary workbook (.xlsb). Ask for a copy saved as .xlsx.")
            raise XlsxError(f"Not an Excel workbook ({kind or 'unknown type'}).")
        self.wb = self.part(main)
        self.wb_rels = _Rels(self, main)
        self._refresh_sheets()
        self._styles: Styles | None = None
        self._strings: list[str] | None = None
        self._string_index: dict[str, int] | None = None
        self._strings_root: Any = None
        self._tables: list[TableInfo] | None = None

    def _refresh_sheets(self) -> None:
        self.sheets: list[Sheet] = []
        self.sheet_els: list[Any] = []
        parent = self.wb.find(_q("sheets"))
        if parent is None:
            raise XlsxError("The workbook is damaged: it lists no sheets.")
        for el in parent:
            self.sheet_els.append(el)
            target = self.wb_rels.target(el.get(f"{{{R_NS}}}id") or "")
            if target is None or not self.has(target):
                continue
            if self.content_type(target) == CT_WORKSHEET or target.startswith("xl/worksheets/"):
                self.sheets.append(Sheet(self, el, target))
        self._by_name = {s.name.casefold(): s for s in self.sheets}

    @property
    def is_macro(self) -> bool:
        return self.content_type(self.wb_path) in (CT_MACRO, CT_MACRO_TEMPLATE)

    @property
    def date1904(self) -> bool:
        props = self.wb.find(_q("workbookPr"))
        return props is not None and props.get("date1904") in ("1", "true")

    @property
    def styles(self) -> Styles:
        if self._styles is None:
            self._styles = Styles(self)
        return self._styles

    def as_ordinary_workbook(self) -> None:
        """A template (.xltx) becomes a normal workbook."""
        self._set_override(self.wb_path, CT_WORKBOOK)

    def sheet(self, name: str) -> Sheet:
        found = self._by_name.get(name.casefold())
        if found is None:
            raise XlsxError(f"There is no sheet {name!r}. Sheets: {', '.join(s.name for s in self.sheets)}.")
        return found.load()

    def all_sheets(self) -> list[Sheet]:
        return [s.load() for s in self.sheets]

    # Text

    def _load_strings(self) -> None:
        if self._strings is not None:
            return
        self._strings = []
        rel = self.wb_rels.of_type("sharedStrings")
        if rel and self.has(rel[0][1]):
            self._strings_root = self.part(rel[0][1])
            self._strings_path = rel[0][1]
            for si in self._strings_root:
                self._strings.append(_unescape_text("".join(
                    t.text or "" for t in si.iter(T_TAG)
                    if t.getparent().tag not in (_q("rPh"),))))

    def shared_string(self, index: int) -> str:
        self._load_strings()
        return self._strings[index] if 0 <= index < len(self._strings) else ""

    def add_string(self, text: str) -> int:
        self._load_strings()
        if self._strings_root is None:
            self._strings_path = "xl/sharedStrings.xml"
            self._strings_root = etree.Element(_q("sst"), nsmap={None: MAIN})
            self.add_part(self._strings_path, self._strings_root, CT_STRINGS)
            self.wb_rels.add("sharedStrings", self._strings_path)
        if self._string_index is None:
            self._string_index = {}
            # Only plain entries are reused: a rich-text one looks the same but isn't.
            for index, si in enumerate(self._strings_root):
                if len(si) == 1 and si[0].tag == T_TAG:
                    self._string_index.setdefault(self._strings[index], index)
        found = self._string_index.get(text)
        if found is not None:
            return found
        si = etree.SubElement(self._strings_root, _q("si"))
        t = etree.SubElement(si, T_TAG)
        t.text = _escape_text(text)
        if text != text.strip() or "\n" in text:
            t.set(XML_SPACE, "preserve")
        self._strings.append(text)
        index = len(self._strings) - 1
        self._string_index[text] = index
        self._strings_root.set("uniqueCount", str(len(self._strings)))
        self._strings_root.attrib.pop("count", None)
        self.touch(self._strings_path)
        return index

    def cell_value(self, cell: Any) -> Any:
        kind = cell.get("t")
        if kind == "inlineStr":
            inline = cell.find(IS)
            return None if inline is None else _unescape_text("".join(t.text or "" for t in inline.iter(T_TAG)))
        value = cell.find(V)
        if value is None:
            return None
        text = value.text
        if kind == "s":
            return self.shared_string(int(text)) if text else ""
        if kind == "str":
            return _unescape_text(text or "")
        if kind == "b":
            return text == "1"
        if kind == "e":
            return F.ERRORS.get((text or "").strip(), F.VALUE)
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def store_value(self, cell: Any, value: Any) -> None:
        """Put a constant or a formula's result in a cell, leaving any ``<f>`` in place."""
        for child in cell.findall(V) + cell.findall(IS):
            cell.remove(child)
        cell.attrib.pop("t", None)
        if value is None:
            return
        el = etree.SubElement(cell, V)
        if isinstance(value, bool):
            cell.set("t", "b")
            el.text = "1" if value else "0"
        elif isinstance(value, (int, float)):
            el.text = number_text(float(value))
        elif isinstance(value, XlError):
            cell.set("t", "e")
            el.text = value.code
        elif cell.find(FT) is not None:
            cell.set("t", "str")
            el.text = _escape_text(str(value))
            if str(value) != str(value).strip():
                el.set(XML_SPACE, "preserve")
        else:
            cell.set("t", "s")
            el.text = str(self.add_string(str(value)))

    # Tables and names

    def tables(self) -> list[TableInfo]:
        if self._tables is None:
            self._tables = []
            for sheet in self.sheets:
                for _, part in sheet.rels.of_type("table"):
                    if not self.has(part):
                        continue
                    root = self.part(part)
                    box = F.parse_box(root.get("ref") or "")
                    if box is None:
                        continue
                    columns = [c.get("name") or "" for c in root.iter(_q("tableColumn"))]
                    self._tables.append(TableInfo(
                        sheet.name, root.get("displayName") or root.get("name") or "", box.r1, box.c1, box.r2, box.c2,
                        [_unescape_text(c) for c in columns], int(root.get("headerRowCount", "1")),
                        int(root.get("totalsRowCount", "0")), part))
        return self._tables

    def forget_tables(self) -> None:
        self._tables = None

    def table_named(self, name: str) -> TableInfo | None:
        return next((t for t in self.tables() if t.name.casefold() == name.casefold()), None)

    def table_at(self, sheet: str, row: int, col: int) -> TableInfo | None:
        return next((t for t in self.tables() if t.sheet.casefold() == sheet.casefold() and
                     t.box.contains(row, col)), None)

    def sheet_position(self, name: str) -> int:
        for index, el in enumerate(self.sheet_els):
            if (el.get("name") or "").casefold() == name.casefold():
                return index
        return -1

    def defined_names(self) -> list[Any]:
        parent = self.wb.find(_q("definedNames"))
        return list(parent) if parent is not None else []

    def defined_name_text(self, name: str, sheet: str | None) -> str | None:
        wanted = name.casefold()
        position = self.sheet_position(sheet) if sheet else -1
        fallback = None
        for item in self.defined_names():
            if (item.get("name") or "").casefold() != wanted:
                continue
            scope = item.get("localSheetId")
            if scope is None:
                fallback = item.text
            elif int(scope) == position:
                return item.text
        return fallback

    # Saving

    def finalize(self) -> None:
        """Bookkeeping for a changed workbook: dimensions, and no stale calculation order."""
        changed_sheets = [s for s in self.sheets if s.root is not None and s.path in self._dirty]
        for sheet in changed_sheets:
            rows, cols = sheet.used()
            dimension = sheet.root.find(_q("dimension"))
            if dimension is None:
                dimension = _insert_ordered(sheet.root, etree.Element(_q("dimension")), _SHEET_ORDER)
            first_row = sheet.row_numbers[0] if sheet.row_numbers else 1
            first_col = min((min(c) for c in sheet.grid.values() if c), default=1)
            dimension.set("ref", f"{addr(first_row, first_col)}:{addr(max(rows, 1), max(cols, 1))}"
                          if rows and cols else "A1")
        if changed_sheets:
            # Excel treats a calcChain naming a cell with no formula as damage, and
            # rebuilds the file silently if it's absent.
            for rel_id, part in self.wb_rels.of_type("calcChain"):
                self.wb_rels.remove(rel_id)
                self.remove_part(part)

    def recalculate_on_open(self) -> None:
        calc = self.wb.find(_q("calcPr"))
        if calc is None:
            calc = _insert_ordered(self.wb, etree.Element(_q("calcPr"), calcId="191029"), _WORKBOOK_ORDER)
        calc.set("fullCalcOnLoad", "1")
        self.touch(self.wb_path)

    def to_bytes(self) -> bytes:
        self.finalize()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as out:
            for name in self._order:
                if name in self._dirty and name in self._parsed:
                    data = etree.tostring(self._parsed[name], xml_declaration=True, encoding="UTF-8",
                                          standalone=True)
                else:
                    data = self._raw.get(name)
                    if data is None:
                        continue
                out.writestr(name, data)
        return buffer.getvalue()


# ── Calculating ──────────────────────────────────────────────────────


class Calc:
    """The workbook as formulas see it, with results calculated so far on top."""

    date1904: bool

    def __init__(self, book: Book, *, use_stored_on_failure: bool = False) -> None:
        self.book = book
        self.date1904 = book.date1904
        self.computed: dict[tuple[str, int, int], Any] = {}
        self.pending: set[tuple[str, int, int]] = set()
        self.failed: dict[tuple[str, int, int], str] = {}
        self.circular: set[tuple[str, int, int]] = set()
        self.use_stored_on_failure = use_stored_on_failure
        self.deadline = time.monotonic() + CALC_SECONDS
        self.sheets = {s.name: s for s in book.all_sheets()}

    # What xlsx_formula asks of a model

    def value(self, sheet: str, row: int, col: int) -> Any:
        key = (sheet, row, col)
        if key in self.computed:
            return self.computed[key]
        target = self.sheets.get(sheet)
        if target is None:
            return None
        owner = target.array_owner.get((row, col))
        if owner is not None and owner != (row, col):
            owner_key = (sheet, owner[0], owner[1])
            if owner_key in self.pending:
                raise NeedCell(*owner_key)
            if owner_key in self.failed and not self.use_stored_on_failure:
                raise NotComputable(self.failed[owner_key])
        if key in self.pending:
            raise NeedCell(*key)
        if key in self.failed and not self.use_stored_on_failure:
            raise NotComputable(f"it reads {addr(row, col)}: {self.failed[key]}")
        return target.stored(row, col)

    def values_in(self, sheet: str, r1: int, c1: int, r2: int, c2: int) -> Iterator[tuple[int, int, Any]]:
        target = self.sheets.get(sheet)
        if target is None:
            return
        numbers = target.row_numbers
        start = bisect_left(numbers, r1)
        for index in range(start, len(numbers)):
            row = numbers[index]
            if row > r2:
                break
            cells = target.grid.get(row)
            if not cells:
                continue
            columns = range(c1, c2 + 1) if c2 - c1 < 64 else sorted(c for c in cells if c1 <= c <= c2)
            for col in columns:
                if col in cells or (sheet, row, col) in self.computed:
                    value = self.value(sheet, row, col)
                    if value is not None:
                        yield row, col, value

    def used(self, sheet: str) -> tuple[int, int]:
        target = self.sheets.get(sheet)
        return target.used() if target is not None else (0, 0)

    def sheet_name(self, name: str) -> str | None:
        found = self.book._by_name.get(name.casefold())
        return found.name if found is not None else None

    def defined_name(self, name: str, sheet: str) -> str | None:
        return self.book.defined_name_text(name, sheet)

    def table_named(self, name: str) -> TableInfo | None:
        return self.book.table_named(name)

    def table_at(self, sheet: str, row: int, col: int) -> TableInfo | None:
        return self.book.table_at(sheet, row, col)

    def formula_at(self, sheet: str, row: int, col: int) -> str | None:
        target = self.sheets.get(sheet)
        return target.formula(row, col) if target is not None else None

    def row_hidden(self, sheet: str, row: int) -> bool:
        target = self.sheets.get(sheet)
        row_el = target.row_els.get(row) if target is not None else None
        return row_el is not None and row_el.get("hidden") == "1"

    # What a formula depends on

    def boxes_of(self, text: str, sheet: str, row: int, col: int, seen: set[str] | None = None) -> list[Box]:
        """Blocks a formula reads, sheet names filled in."""
        compiled = F.compile_formula(text)
        out: list[Box] = []
        for operand in compiled.operands:
            if isinstance(operand, Box):
                name = self.sheet_name(operand.sheet) if operand.sheet else sheet
                if name is not None:
                    out.append(Box(name, operand.r1, operand.c1, operand.r2, operand.c2))
            elif isinstance(operand, F.NameOperand):
                seen = seen if seen is not None else set()
                key = operand.name.casefold()
                if key in seen:
                    continue
                seen.add(key)
                body = self.book.defined_name_text(operand.name, operand.sheet or sheet)
                if body:
                    out.extend(self.boxes_of(body, operand.sheet or sheet, row, col, seen))
            elif isinstance(operand, F.TableOperand):
                out.extend(self._table_boxes(operand, sheet, row, col))
        return out

    def _table_boxes(self, operand: F.TableOperand, sheet: str, row: int, col: int) -> list[Box]:
        table = (self.book.table_named(operand.table) if operand.table
                 else self.book.table_at(sheet, row, col))
        if table is None:
            return []
        c1, c2 = table.c1, table.c2
        if operand.first_col is not None:
            names = [c.casefold() for c in table.columns]
            try:
                c1 = table.c1 + names.index(operand.first_col.casefold())
                c2 = table.c1 + names.index((operand.last_col or operand.first_col).casefold())
            except ValueError:
                return []
        special = {s.casefold() for s in operand.special}
        if "#this row" in special:
            return [Box(table.sheet, row, c1, row, c2)]
        return [Box(table.sheet, table.r1, c1, table.r2, c2)]

    def affected_by(self, seeds: Iterable[tuple[str, int, int]]) -> set[tuple[str, int, int]]:
        """Every formula that reads, directly or not, one of the *seeds* cells."""
        cell_map: dict[tuple[str, int, int], list[tuple[str, int, int]]] = {}
        big: dict[str, dict[tuple[int, int, int, int], list[tuple[str, int, int]]]] = {}
        always: list[tuple[str, int, int]] = []
        for sheet in self.sheets.values():
            for (row, col) in sheet.formula_els:
                text = sheet.formula(row, col)
                key = (sheet.name, row, col)
                if not text:
                    continue
                compiled = F.compile_formula(text)
                if compiled.dynamic:
                    always.append(key)
                for box in self.boxes_of(text, sheet.name, row, col):
                    if box.area <= 64:
                        for r in range(box.r1, box.r2 + 1):
                            for c in range(box.c1, box.c2 + 1):
                                cell_map.setdefault((box.sheet, r, c), []).append(key)
                    else:
                        big.setdefault(box.sheet, {}).setdefault((box.r1, box.c1, box.r2, box.c2), []).append(key)
        result: set[tuple[str, int, int]] = set()
        queue = list(seeds)
        if queue:
            result.update(always)
            queue.extend(always)
        seen = set(queue)

        def add(key: tuple[str, int, int]) -> None:
            if key not in result:
                result.add(key)
            if key not in seen:
                seen.add(key)
                queue.append(key)

        while queue:
            current = queue.pop()
            sheet_name, row, col = current
            target = self.sheets.get(sheet_name)
            if target is not None:
                box = target.array_masters.get((row, col))
                if box is not None and not box.is_cell:
                    for r in range(box.r1, box.r2 + 1):
                        for c in range(box.c1, box.c2 + 1):
                            if (sheet_name, r, c) not in seen:
                                seen.add((sheet_name, r, c))
                                queue.append((sheet_name, r, c))
            for dependent in cell_map.get(current, ()):
                add(dependent)
            blocks = big.get(sheet_name)
            if blocks:
                fired = [b for b in blocks if b[0] <= row <= b[2] and b[1] <= col <= b[3]]
                for block in fired:
                    # Every formula on this block is dirty now; no need to look again.
                    for dependent in blocks.pop(block):
                        add(dependent)
        return result

    # Calculating

    def _pending_deps(self, key: tuple[str, int, int]) -> Iterator[tuple[str, int, int]]:
        sheet, row, col = key
        target = self.sheets[sheet]
        text = target.formula(row, col) or ""
        for box in self.boxes_of(text, sheet, row, col):
            if box.area <= len(self.pending) and box.area <= 4096:
                for r in range(box.r1, box.r2 + 1):
                    for c in range(box.c1, box.c2 + 1):
                        cell = (box.sheet, r, c)
                        if cell in self.pending:
                            yield cell
            else:
                for cell in list(self.pending):
                    if cell[0] == box.sheet and box.contains(cell[1], cell[2]):
                        yield cell

    def run(self, keys: Iterable[tuple[str, int, int]]) -> None:
        """Calculate *keys* (and whatever they need), never recursing: long chains
        of running totals would overflow Python's stack."""
        self.pending.update(keys)
        for start in list(self.pending):
            if start in self.pending:
                self._calculate(start)

    def _calculate(self, start: tuple[str, int, int]) -> None:
        stack: list[tuple[tuple[str, int, int], Iterator[tuple[str, int, int]]]] = [(start, self._pending_deps(start))]
        path = {start}
        ticks = 0
        while stack:
            key, deps = stack[-1]
            if key not in self.pending:
                stack.pop()
                path.discard(key)
                continue
            ticks += 1
            if ticks % 2048 == 0 and time.monotonic() > self.deadline:
                self._give_up()
                return
            blocked = False
            for dep in deps:
                if dep not in self.pending or dep == key:
                    if dep == key:
                        self._mark_circular(key)
                        blocked = True
                        break
                    continue
                if dep in path:
                    self._mark_circular(key)
                else:
                    stack.append((dep, self._pending_deps(dep)))
                    path.add(dep)
                blocked = True
                break
            if blocked:
                continue
            try:
                self._finish(key)
            except NeedCell as need:
                if need.key in path:
                    self._mark_circular(key)
                elif need.key in self.pending:
                    stack.append((need.key, self._pending_deps(need.key)))
                    path.add(need.key)
                else:
                    self._fail(key, "it reads a cell that couldn't be calculated")

    def _give_up(self) -> None:
        for key in list(self.pending):
            self._fail(key, "calculating took too long here")

    def _mark_circular(self, key: tuple[str, int, int]) -> None:
        self.circular.add(key)
        self._fail(key, "circular reference")

    def _fail(self, key: tuple[str, int, int], reason: str) -> None:
        self.pending.discard(key)
        self.failed[key] = reason

    def _finish(self, key: tuple[str, int, int]) -> None:
        sheet_name, row, col = key
        sheet = self.sheets[sheet_name]
        if sheet.is_opaque_formula(row, col):
            self._fail(key, "a what-if data table")
            return
        text = sheet.formula(row, col)
        if not text:
            self._fail(key, "its formula can't be read")
            return
        compiled = F.compile_formula(text)
        box = sheet.array_masters.get((row, col))
        try:
            if box is not None:
                result = F.evaluate(compiled, self, sheet_name, row, col, array=True)
                self.pending.discard(key)
                for r in range(box.r1, box.r2 + 1):
                    for c in range(box.c1, box.c2 + 1):
                        cell = (sheet_name, r, c)
                        self.pending.discard(cell)
                        if isinstance(result, F.Arr):
                            value = result.at(r - box.r1, c - box.c1)
                        else:
                            value = result
                        self.computed[cell] = 0.0 if value is None else value
                return
            result = F.evaluate(compiled, self, sheet_name, row, col)
        except NeedCell:
            raise
        except NotComputable as exc:
            self._fail(key, str(exc))
            return
        except Exception as exc:
            # One strange formula must never take the whole edit down with it.
            self._fail(key, f"it can't be calculated here ({type(exc).__name__})")
            return
        self.pending.discard(key)
        self.computed[key] = result


@dataclass(slots=True)
class Change:
    sheet: str
    cell: str
    before: str
    after: str


@dataclass(slots=True)
class Report:
    counts: dict[str, int] = field(default_factory=dict)
    written: list[Change] = field(default_factory=list)
    moved: list[Change] = field(default_factory=list)
    not_calculated: list[tuple[str, str, str]] = field(default_factory=list)
    new_errors: list[tuple[str, str, str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _clean_op_name(raw: Any) -> str:
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


OPS = ("set", "fill", "clear", "append_rows", "insert_rows", "delete_rows", "insert_columns", "delete_columns",
       "format", "width", "height", "freeze", "merge", "unmerge", "filter", "add_sheet", "rename_sheet",
       "delete_sheet", "move_sheet", "define_name", "delete_name", "add_table", "add_chart", "comment", "recalculate")
FORMAT_KEYS = ("number_format", "bold", "italic", "underline", "strike", "color", "font", "font_size", "fill",
               "border", "border_top", "border_bottom", "border_left", "border_right", "align", "valign",
               "wrap", "indent")
_SHEET_OPS = {"add_sheet", "rename_sheet", "delete_sheet", "move_sheet"}


class Editor:
    """Applies a list of changes to a workbook in memory: all of them, or none."""

    def __init__(self, book: Book, *, author: str = "Captain", default_sheet: str | None = None) -> None:
        self.book = book
        self.author = author
        self.default_sheet = default_sheet
        self.report = Report()
        self.before: dict[tuple[str, int, int], tuple[Any, str | None, str]] = {}
        self.seeds: set[tuple[str, int, int]] = set()
        self.formula_seeds: set[tuple[str, int, int]] = set()
        self.written_formulas: list[tuple[str, int, int, str]] = []
        self.new_numbers: set[tuple[str, int, int]] = set()
        self.cells_written = 0
        self.new_charts: list[str] = []
        self.full_recalculate = False
        self.structure_changed = False
        self.pending_auto_width: list[str] = []
        self.chosen_widths: set[tuple[str, int]] = set()

    # Addressing

    def _sheet(self, op: dict[str, Any], from_ref: str | None = None) -> Sheet:
        name = from_ref or op.get("sheet") or self.default_sheet
        if name:
            return self.book.sheet(str(name))
        if len(self.book.sheets) == 1:
            return self.book.sheets[0].load()
        raise XlsxError(f"Say which sheet (`sheet` or Sheet!A1). Sheets: {', '.join(s.name for s in self.book.sheets)}.")

    def _target(self, op: dict[str, Any], *keys: str) -> tuple[Sheet, Box]:
        for key in keys:
            if op.get(key) not in (None, ""):
                sheet_name, box = parse_range(str(op[key]))
                return self._sheet(op, sheet_name), box
        raise XlsxError(f"{op.get('op')}: needs {' or '.join(keys)}.")

    def _guard(self, sheet: Sheet) -> None:
        if sheet.is_protected():
            raise XlsxError(f"Sheet {sheet.name!r} is protected. Ask the user to unprotect it in Excel first.")

    def _clip(self, sheet: Sheet, box: Box) -> Box:
        """A whole column or row stops at the used part of the sheet."""
        rows, cols = sheet.used()
        r2 = max(box.r1, min(box.r2, rows)) if box.r2 == F.MAX_ROW else box.r2
        c2 = max(box.c1, min(box.c2, cols)) if box.c2 == F.MAX_COL else box.c2
        return Box(box.sheet, box.r1, box.c1, r2, c2)

    def _count(self, name: str, number: int = 1) -> None:
        self.report.counts[name] = self.report.counts.get(name, 0) + number

    def _note_before(self, sheet: Sheet, row: int, col: int) -> None:
        key = (sheet.name, row, col)
        if key not in self.before:
            self.before[key] = (sheet.stored(row, col), sheet.formula(row, col), sheet.number_format(row, col))

    # Writing a cell

    def _write(self, sheet: Sheet, row: int, col: int, raw: Any, *, as_text: bool = False,
               style: int | None = None) -> None:
        if not (1 <= row <= F.MAX_ROW and 1 <= col <= F.MAX_COL):
            raise XlsxError(f"{addr(max(row, 1), max(col, 1))} is outside the sheet.")
        self.cells_written += 1
        if self.cells_written > MAX_CELLS_PER_CALL:
            raise XlsxError(f"One call can write at most {MAX_CELLS_PER_CALL:,} cells.")
        merged = sheet.merged_at(row, col)
        if merged is not None and (merged.r1, merged.c1) != (row, col):
            raise XlsxError(f"{addr(row, col)} is inside the merged cells {box_text(merged)}; "
                            f"write to {addr(merged.r1, merged.c1)}.")
        owner = sheet.array_owner.get((row, col))
        if owner is not None and sheet.array_masters[owner].area > 1:
            raise XlsxError(f"{addr(row, col)} is part of the array formula in "
                            f"{box_text(sheet.array_masters[owner])}; change the whole array in Excel.")
        if isinstance(raw, list) or isinstance(raw, dict):
            raise XlsxError(f"{addr(row, col)}: a cell holds one value, not a {type(raw).__name__}.")
        if self._guard_table_header(sheet, row, col, raw):
            as_text = True  # Excel wants a table's header cell to be text, matching its column name
        self._note_before(sheet, row, col)
        had_value = sheet.stored(row, col) is not None or (row, col) in sheet.formula_els
        key = (sheet.name, row, col)
        if raw is None or (raw == "" and not as_text):
            sheet.clear_content(row, col)
            sheet.array_masters.pop((row, col), None)
            sheet.array_owner.pop((row, col), None)
            self.seeds.add(key)
            if style is not None and sheet.cell_el(row, col) is not None:
                sheet.cell_el(row, col).set("s", str(style))
            return
        cell = sheet.cell(row, col)
        if style is not None:
            cell.set("s", str(style))
        kind, value, number_format = self._interpret(raw, as_text)
        _strip_content(cell)
        sheet.formula_els.pop((row, col), None)
        sheet._translated.pop((row, col), None)
        sheet.array_masters.pop((row, col), None)
        sheet.array_owner.pop((row, col), None)
        if kind == "formula":
            stored = F.normalize_for_storage(value, allow_custom=self.book.is_macro)
            el = etree.SubElement(cell, FT)
            el.text = stored
            sheet.formula_els[(row, col)] = el
            if F.needs_array(stored):
                # Without this Excel reads =SUM(A1:A3*B1:B3) with one value from each range.
                el.set("t", "array")
                el.set("ref", addr(row, col))
                sheet.array_masters[(row, col)] = Box(None, row, col, row, col)
                sheet.array_owner[(row, col)] = (row, col)
            self.written_formulas.append((sheet.name, row, col, stored))
            self.formula_seeds.add(key)
        else:
            self.book.store_value(cell, value)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not had_value:
                self.new_numbers.add(key)
        if number_format and sheet.number_format(row, col).lower() == "general":
            cell.set("s", str(self.book.styles.derive(sheet.style_of(row, col), number_format=number_format)))
        self.seeds.add(key)
        sheet.touch()

    def _interpret(self, raw: Any, as_text: bool) -> tuple[str, Any, str]:
        """(kind, value, number format): what typing this in a cell means."""
        if as_text:
            text = str(raw)
            if len(text) > MAX_CELL_TEXT:
                raise XlsxError(f"A cell holds at most {MAX_CELL_TEXT:,} characters.")
            return "text", text, ""
        if isinstance(raw, bool):
            return "value", raw, ""
        if isinstance(raw, (int, float)):
            if isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw)):
                raise XlsxError("A cell can't hold NaN or infinity.")
            return "value", float(raw), ""
        text = str(raw)
        if text.startswith("=") and len(text) > 1:
            return "formula", text, ""
        stripped = text.strip()
        if _PLAIN_NUMBER.match(stripped):
            return "value", float(stripped), ""
        percent = _PERCENT.match(stripped)
        if percent:
            return "value", float(percent.group(1)) / 100, "0%" if "." not in percent.group(1) else "0.00%"
        date_match = _ISO_DATE.match(stripped)
        if date_match:
            year, month, day, hour, minute, second = date_match.groups()
            try:
                serial = float(F.date_to_serial(dt.date(int(year), int(month), int(day)), self.book.date1904))
            except ValueError:
                return "text", text, ""
            if hour is not None:
                serial += (int(hour) * 3600 + int(minute) * 60 + int(second or 0)) / 86400
                return "value", serial, "yyyy-mm-dd hh:mm"
            return "value", serial, "yyyy-mm-dd"
        if stripped.upper() in ("TRUE", "FALSE"):
            return "value", stripped.upper() == "TRUE", ""
        if len(text) > MAX_CELL_TEXT:
            raise XlsxError(f"A cell holds at most {MAX_CELL_TEXT:,} characters.")
        return "text", text, ""

    def _guard_table_header(self, sheet: Sheet, row: int, col: int, raw: Any) -> bool:
        """Is this a table's header cell? Keeps the table's column name in step."""
        for table in self.book.tables():
            if table.sheet.casefold() != sheet.name.casefold() or not table.header_rows:
                continue
            if row == table.r1 and table.c1 <= col <= table.c2:
                new = str(raw) if raw is not None else ""
                index = col - table.c1
                old = table.columns[index] if index < len(table.columns) else ""
                if new == old:
                    return True
                if not new or new.startswith("="):
                    raise XlsxError(f"{addr(row, col)} is a header of table {table.name} and must stay text.")
                if new.casefold() in (c.casefold() for c in table.columns):
                    raise XlsxError(f"Table {table.name} already has a column {new!r}.")
                if self._table_is_referenced(table):
                    raise XlsxError(f"{addr(row, col)} is a header of table {table.name} and formulas use that "
                                    "table by name; rename the column in Excel.")
                root = self.book.part(table.part)
                columns = list(root.iter(_q("tableColumn")))
                columns[index].set("name", new)
                table.columns[index] = new
                self.book.touch(table.part)
                return True
        return False

    def _table_is_referenced(self, table: TableInfo) -> bool:
        for sheet in self.book.all_sheets():
            for (row, col) in sheet.formula_els:
                text = sheet.formula(row, col) or ""
                if "[" in text and table.name.casefold() in text.casefold():
                    return True
                if "[" in text and table.box.contains(row, col) and sheet.name == table.sheet:
                    return True
        return False

    # Operations

    def apply(self, ops: list[dict[str, Any]]) -> Report:
        for index, op in enumerate(ops, 1):
            if not isinstance(op, dict):
                raise XlsxError(f"Edit {index} is not an object with an `op`.")
            name = _clean_op_name(op.get("op"))
            handler = getattr(self, f"_op_{name}", None) if name in OPS else None
            if handler is None:
                raise XlsxError(f"Edit {index}: unknown op {op.get('op')!r}; use one of: {', '.join(OPS)}.")
            try:
                handler(op)
            except FormulaError as exc:
                raise XlsxError(f"Edit {index} ({name}): {exc}") from None
            except XlsxError as exc:
                raise XlsxError(f"Edit {index} ({name}): {exc}") from None
        return self.finish()

    def _format_changes(self, op: dict[str, Any]) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        for key in FORMAT_KEYS:
            if key in op:
                changes[key] = op[key]
        align = changes.get("align")
        if align is not None and str(align).lower() not in ("left", "center", "right", "general", "justify", "fill"):
            raise XlsxError(f"align {align!r}: use left, center or right.")
        valign = changes.get("valign")
        if valign is not None and str(valign).lower() not in ("top", "center", "bottom"):
            raise XlsxError(f"valign {valign!r}: use top, center or bottom.")
        for key in ("align", "valign"):
            if key in changes:
                changes[key] = str(changes[key]).lower()
        return changes

    def _apply_format(self, sheet: Sheet, cells: Iterable[tuple[int, int]], changes: dict[str, Any]) -> int:
        count = 0
        for row, col in cells:
            cell = sheet.cell(row, col)
            cell.set("s", str(self.book.styles.derive(sheet.style_of(row, col), **changes)))
            count += 1
        if count:
            sheet.touch()
        return count

    def _op_set(self, op: dict[str, Any]) -> None:
        as_text = bool(op.get("as_text"))
        written: list[tuple[int, int]] = []
        style = self._like_style(op)
        if isinstance(op.get("cells"), dict):
            for ref, value in op["cells"].items():
                sheet_name, box = parse_range(str(ref))
                sheet = self._sheet(op, sheet_name)
                self._guard(sheet)
                self._write(sheet, box.r1, box.c1, value, as_text=as_text, style=style)
                written.append((box.r1, box.c1))
                if self._format_changes(op):
                    self._apply_format(sheet, [(box.r1, box.c1)], self._format_changes(op))
        elif "rows" in op:
            sheet, box = self._target(op, "cell", "range", "at")
            self._guard(sheet)
            rows = op["rows"]
            if not isinstance(rows, list) or not rows:
                raise XlsxError("`rows` must be a list of rows, each a list of values.")
            if not isinstance(rows[0], list):
                rows = [rows]
            if not box.is_cell and (len(rows) != box.r2 - box.r1 + 1 or
                                    max(len(r) for r in rows) != box.c2 - box.c1 + 1):
                raise XlsxError(f"{box_text(box)} is {box.r2 - box.r1 + 1} rows by {box.c2 - box.c1 + 1} columns "
                                f"but `rows` is {len(rows)} by {max(len(r) for r in rows)}.")
            for r, row in enumerate(rows):
                if not isinstance(row, list):
                    raise XlsxError(f"Row {r + 1} of `rows` is not a list.")
                for c, value in enumerate(row):
                    self._write(sheet, box.r1 + r, box.c1 + c, value, as_text=as_text, style=style)
                    written.append((box.r1 + r, box.c1 + c))
        elif "value" in op:
            sheet, box = self._target(op, "cell", "range", "at")
            self._guard(sheet)
            box = self._clip(sheet, box)
            for r in range(box.r1, box.r2 + 1):
                for c in range(box.c1, box.c2 + 1):
                    self._write(sheet, r, c, op["value"], as_text=as_text, style=style)
                    written.append((r, c))
        else:
            raise XlsxError("needs `value` (with `cell` or `range`), `rows` (with `cell`), or `cells`.")
        changes = self._format_changes(op)
        if changes and written and "cells" not in op:
            self._apply_format(sheet, written, changes)
        self._count("cells set", len(written))

    def _like_style(self, op: dict[str, Any]) -> int | None:
        like = op.get("like")
        if not like:
            return None
        sheet_name, box = parse_range(str(like))
        return self._sheet(op, sheet_name).style_of(box.r1, box.c1)

    def _op_fill(self, op: dict[str, Any]) -> None:
        sheet, box = self._target(op, "range")
        self._guard(sheet)
        box = self._clip(sheet, box)
        source_ref = op.get("source")
        if source_ref:
            source_sheet_name, source = parse_range(str(source_ref))
            source_sheet = self._sheet(op, source_sheet_name)
            text = source_sheet.formula(source.r1, source.c1)
            origin = addr(source.r1, source.c1)
            style = source_sheet.style_of(source.r1, source.c1)
            constant = source_sheet.stored(source.r1, source.c1) if text is None else None
        else:
            raw = op.get("formula", op.get("value"))
            if raw is None:
                raise XlsxError("needs `formula` (as it reads in the first cell of the range), `value` or `source`.")
            text = None
            constant = raw
            if isinstance(raw, str) and raw.startswith("="):
                text = F.normalize_for_storage(raw, allow_custom=self.book.is_macro)
                constant = None
            origin = addr(box.r1, box.c1)
            style = None
        written: list[tuple[int, int]] = []
        for r in range(box.r1, box.r2 + 1):
            for c in range(box.c1, box.c2 + 1):
                value = "=" + F.translate(text, origin, addr(r, c)) if text is not None else constant
                self._write(sheet, r, c, value, style=style)
                written.append((r, c))
        changes = self._format_changes(op)
        if changes:
            self._apply_format(sheet, written, changes)
        self._count("cells filled", len(written))

    def _op_clear(self, op: dict[str, Any]) -> None:
        sheet, box = self._target(op, "range", "cell")
        self._guard(sheet)
        box = self._clip(sheet, box)
        count = 0
        for r in range(box.r1, box.r2 + 1):
            cells = sheet.grid.get(r)
            if not cells:
                continue
            for c in [c for c in cells if box.c1 <= c <= box.c2]:
                if sheet.stored(r, c) is not None or (r, c) in sheet.formula_els:
                    self._write(sheet, r, c, None)
                    count += 1
                if op.get("formats"):
                    cells[c].attrib.pop("s", None)
                    sheet.touch()
        self._count("cells cleared", count)

    def _op_append_rows(self, op: dict[str, Any]) -> None:
        rows = op.get("rows")
        if not isinstance(rows, list) or not rows:
            raise XlsxError("needs `rows`: a list of rows, each a list of values.")
        if not isinstance(rows[0], list):
            rows = [rows]
        table_name = str(op.get("table") or "").strip()
        if table_name:
            table = self.book.table_named(table_name)
            if table is None:
                names = ", ".join(t.name for t in self.book.tables()) or "none"
                raise XlsxError(f"There is no table {table_name!r} (tables: {names}).")
            self._append_to_table(table, rows, op)
            return
        sheet = self._sheet(op)
        self._guard(sheet)
        last_row = sheet.used()[0]
        first_col = 1
        if op.get("cell"):
            _, start = parse_range(str(op["cell"]))
            first_col = start.c1
        totals = [c for c in sheet.grid.get(last_row, {}) if F.compile_formula(
            sheet.formula(last_row, c) or "0").functions & {"SUM", "SUBTOTAL", "AVERAGE"}]
        if totals:
            self.report.notes.append(
                f"Row {last_row} looks like a totals row and the new rows went below it. To add above the "
                f"totals use insert_rows at {last_row} and set those rows instead.")
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                row_number = last_row + 1 + r
                col = first_col + c
                style = sheet.style_of(last_row, col) if last_row else None
                self._write(sheet, row_number, col, value, style=style or None)
        self._count("rows appended", len(rows))

    def _append_to_table(self, table: TableInfo, rows: list[list[Any]], op: dict[str, Any]) -> None:
        sheet = self.book.sheet(table.sheet)
        self._guard(sheet)
        width = table.c2 - table.c1 + 1
        if any(len(row) > width for row in rows):
            raise XlsxError(f"Table {table.name} has {width} columns; a row has more values than that.")
        if table.totals_rows:
            # Inside the table, so its range stretches with the insert.
            first_new = table.r2 - table.totals_rows + 1
            self._shift(sheet, "row", first_new, len(rows))
            table = self.book.table_named(table.name)
            new_last = table.r2
        else:
            first_new = table.r2 + 1
            new_last = table.r2 + len(rows)
            for r in range(first_new, first_new + len(rows)):
                for c in range(table.c1, table.c2 + 1):
                    if sheet.stored(r, c) is not None or (r, c) in sheet.formula_els:
                        raise XlsxError(f"{addr(r, c)} below table {table.name} isn't empty. Insert rows first "
                                        "(insert_rows), then set them, or clear those cells.")
        template_row = first_new - 1
        root = self.book.part(table.part)
        columns = list(root.iter(_q("tableColumn")))
        for r, row in enumerate(rows):
            row_number = first_new + r
            for c in range(width):
                col = table.c1 + c
                value = row[c] if c < len(row) else None
                style = sheet.style_of(template_row, col) or None
                if value is None:
                    # A calculated column fills itself in, as Excel's tables do.
                    calculated = columns[c].find(_q("calculatedColumnFormula")) if c < len(columns) else None
                    above = sheet.formula(template_row, col) if template_row >= table.r1 + table.header_rows else None
                    if calculated is not None and calculated.text:
                        value = "=" + calculated.text
                    elif above:
                        value = "=" + F.translate(above, addr(template_row, col), addr(row_number, col))
                self._write(sheet, row_number, col, value, style=style)
        root.set("ref", addr(table.r1, table.c1) + ":" + addr(new_last, table.c2))
        filter_el = root.find(_q("autoFilter"))
        if filter_el is not None:
            filter_el.set("ref", addr(table.r1, table.c1) + ":" + addr(new_last - table.totals_rows, table.c2))
        self.book.touch(table.part)
        if not table.totals_rows:
            self._extend_ranges_over_table(table, len(rows))
        self.book.forget_tables()
        self._count("rows appended", len(rows))

    def _extend_ranges_over_table(self, table: TableInfo, added: int) -> None:
        """Excel stretches a plain range that covered a whole table column when the table grows."""
        first_data = table.r1 + table.header_rows
        last_data = table.r2

        def stretch(stored: str, own_sheet: str | None) -> str:
            items = F._tokens(stored)
            changed = False
            for token in items:
                if token.type != "OPERAND" or token.subtype != "RANGE":
                    continue
                sheet_name, body, has_prefix = F.split_sheet_prefix(token.value)
                where = sheet_name if has_prefix else own_sheet
                if where is None or where.casefold() != table.sheet.casefold():
                    continue
                box = F.parse_box(body)
                if box is None or box.is_cell or box.r1 != first_data or box.r2 != last_data \
                        or box.c1 < table.c1 or box.c2 > table.c2:
                    continue
                new_body = re.sub(rf"(\$?){last_data}$", lambda m: f"{m.group(1)}{last_data + added}", body)
                prefix = token.value[: len(token.value) - len(body)] if has_prefix else ""
                token.value = prefix + new_body
                changed = True
            return "".join(t.value for t in items) if changed else stored

        self._unshare_formulas_reading(table.sheet)
        for sheet in self.book.all_sheets():
            for (row, col), el in sheet.formula_els.items():
                if table.box.contains(row, col) and sheet.name == table.sheet:
                    continue
                if not el.text:
                    continue
                new = stretch(el.text, sheet.name)
                if new != el.text:
                    self._note_before(sheet, row, col)
                    el.text = new
                    sheet.touch()
                    self.formula_seeds.add((sheet.name, row, col))

    def _span(self, op: dict[str, Any], axis: str) -> tuple[int, int]:
        """(first, count) of the rows or columns an op names."""
        raw = op.get("at", op.get("rows" if axis == "row" else "columns"))
        if raw is None:
            raise XlsxError(f"needs `at` (the first {'row number' if axis == 'row' else 'column letter'}).")
        text = str(raw).strip().replace("$", "")
        count = op.get("count")
        if ":" in text:
            left, right = text.split(":", 1)
        else:
            left, right = text, ""
        try:
            first = int(left) if axis == "row" else _col_index(left.upper())
            last = (int(right) if axis == "row" else _col_index(right.upper())) if right else None
        except (ValueError, KeyError):
            raise XlsxError(f"{raw!r} is not a {'row number' if axis == 'row' else 'column letter'}.") from None
        if last is not None:
            number = abs(last - first) + 1
            first = min(first, last)
        else:
            number = int(count) if count is not None else 1
        limit = F.MAX_ROW if axis == "row" else F.MAX_COL
        if first < 1 or number < 1 or first > limit:
            raise XlsxError(f"{raw!r} and count {number} don't name {'rows' if axis == 'row' else 'columns'} on the sheet.")
        return first, number

    def _op_insert_rows(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        first, count = self._span(op, "row")
        self._shift(sheet, "row", first, count)
        self._count("rows inserted", count)

    def _op_delete_rows(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        first, count = self._span(op, "row")
        self._shift(sheet, "row", first, -count)
        self._count("rows deleted", count)

    def _op_insert_columns(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        first, count = self._span(op, "col")
        self._shift(sheet, "col", first, count)
        self._count("columns inserted", count)

    def _op_delete_columns(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        first, count = self._span(op, "col")
        self._shift(sheet, "col", first, -count)
        self._count("columns deleted", count)

    # Moving rows and columns

    def _shift(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        self._guard(sheet)
        self._check_shift_allowed(sheet, axis, at, count)
        book = self.book
        self._unshare_formulas_reading(sheet.name)
        self._remap_tracking(sheet.name, axis, at, count)
        self._move_cells(sheet, axis, at, count)
        changed_area: list[tuple[str, int, int]] = []

        def rewrite(text: str, own_sheet: str | None) -> str:
            return F.shift_formula(text, own_sheet, sheet.name, axis, at, count)[0]

        for other in book.all_sheets():
            for (row, col), el in other.formula_els.items():
                if not el.text:
                    continue
                new, area = F.shift_formula(el.text, other.name, sheet.name, axis, at, count)
                if new != el.text:
                    el.text = new
                    other.touch()
                if area:
                    changed_area.append((other.name, row, col))
                elif other is sheet and F.compile_formula(new).position_sensitive:
                    changed_area.append((other.name, row, col))
            if other is sheet:
                for el in other.formula_els.values():
                    if el.get("ref"):
                        shifted = _shift_ref_text(el.get("ref"), axis, at, count)
                        if shifted:
                            el.set("ref", shifted)
        self.formula_seeds.update(changed_area)
        self._rewrite_outside_cells(rewrite)
        self._shift_sheet_furniture(sheet, axis, at, count)
        self._shift_tables(sheet, axis, at, count)
        self._shift_drawings(sheet, axis, at, count)
        self._shift_comments(sheet, axis, at, count)
        self._shift_pivot_sources(sheet, axis, at, count)
        if count > 0:
            self._copy_format_into_new(sheet, axis, at, count)
        sheet.reindex()
        for other in book.all_sheets():
            # Array owners and merges follow the XML.
            if other is not sheet:
                other._used = None
        self.structure_changed = True

    def _unshare_formulas_reading(self, sheet_name: str) -> None:
        """Write out shared formulas on that sheet and in any that names it; a
        sheet this doesn't concern keeps its bytes."""
        wanted = sheet_name.casefold()
        for other in self.book.all_sheets():
            if other.name.casefold() == wanted or any(wanted in m[0].casefold() for m in other.shared.values()):
                other.unshare_formulas()

    def _check_shift_allowed(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        gone_last = at - count - 1 if count < 0 else None

        def crosses(first: int, last: int) -> bool:
            if count > 0:
                return first < at <= last
            return not (last < at or first > gone_last) and not (first >= at and last <= gone_last)

        for box in sheet.array_masters.values():
            first, last = (box.r1, box.r2) if axis == "row" else (box.c1, box.c2)
            if box.area > 1 and crosses(first, last):
                raise XlsxError(f"That would split the array formula in {box_text(box)}; change it in Excel.")
        if any(el.get("t") == "dataTable" for el in sheet.formula_els.values()):
            raise XlsxError(f"Sheet {sheet.name!r} has a what-if data table; move its rows in Excel.")
        if sheet.rels.of_type("pivotTable"):
            raise XlsxError(f"Sheet {sheet.name!r} holds a pivot table; move its rows and columns in Excel.")
        for table in self.book.tables():
            if table.sheet.casefold() != sheet.name.casefold():
                continue
            if axis == "col":
                if count > 0 and table.c1 < at <= table.c2:
                    raise XlsxError(f"Inserting there would change table {table.name}'s columns; "
                                    "add the table column in Excel.")
                if count < 0 and not (table.c2 < at or table.c1 > gone_last):
                    raise XlsxError(f"Deleting there would change table {table.name}'s columns; "
                                    "remove the table column in Excel.")
            elif count < 0:
                first_data = table.r1 + table.header_rows
                last_data = table.r2 - table.totals_rows
                if table.header_rows and at <= table.r1 <= gone_last and gone_last < table.r2:
                    raise XlsxError(f"That would delete the header of table {table.name} but not the table.")
                if at <= first_data and gone_last >= last_data and not (at <= table.r1 and gone_last >= table.r2):
                    raise XlsxError(f"That would delete every data row of table {table.name}; keep one.")

    def _remap_tracking(self, sheet_name: str, axis: str, at: int, count: int) -> None:
        def move(key: tuple[str, int, int]) -> tuple[str, int, int] | None:
            sheet, row, col = key
            if sheet != sheet_name:
                return key
            if axis == "row":
                moved = F._shift_edge(row, at, count)
                return None if moved is None else (sheet, moved, col)
            moved = F._shift_edge(col, at, count)
            return None if moved is None else (sheet, row, moved)

        self.before = {new: value for key, value in self.before.items() if (new := move(key)) is not None}
        self.seeds = {new for key in self.seeds if (new := move(key)) is not None}
        self.formula_seeds = {new for key in self.formula_seeds if (new := move(key)) is not None}
        self.new_numbers = {new for key in self.new_numbers if (new := move(key)) is not None}
        moved_formulas = []
        for sheet, row, col, text in self.written_formulas:
            where = move((sheet, row, col))
            if where is not None:
                moved_formulas.append((where[0], where[1], where[2], text))
        self.written_formulas = moved_formulas

    def _move_cells(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        if axis == "row":
            for number in sorted(sheet.row_els, reverse=count > 0):
                moved = F._shift_edge(number, at, count)
                row_el = sheet.row_els[number]
                if moved is None:
                    self._note_removed_cells(sheet, number, None)
                    sheet.data.remove(row_el)
                    continue
                if moved != number:
                    if moved > F.MAX_ROW:
                        raise XlsxError("That would push data off the bottom of the sheet.")
                    row_el.set("r", str(moved))
                    for col, cell in sheet.grid[number].items():
                        cell.set("r", addr(moved, col))
        else:
            for number, cells in sheet.grid.items():
                for col in sorted(cells, reverse=count > 0):
                    moved = F._shift_edge(col, at, count)
                    cell = cells[col]
                    if moved is None:
                        self._note_removed_cells(sheet, number, col)
                        cell.getparent().remove(cell)
                        continue
                    if moved != col:
                        if moved > F.MAX_COL:
                            raise XlsxError("That would push data off the right of the sheet.")
                        cell.set("r", addr(number, moved))
                sheet.row_els[number].attrib.pop("spans", None)
            self._shift_cols(sheet, at, count)
        sheet.touch()
        sheet.reindex()

    def _note_removed_cells(self, sheet: Sheet, row: int, col: int | None) -> None:
        """Whatever read a deleted cell changes, whether or not its text does."""
        for c in ([col] if col is not None else list(sheet.grid.get(row, {}))):
            if sheet.stored(row, c) is not None:
                self._count("cells removed")

    def _shift_cols(self, sheet: Sheet, at: int, count: int) -> None:
        cols = sheet.root.find(_q("cols"))
        if cols is None:
            return
        for col_el in list(cols):
            first, last = int(col_el.get("min", "1")), int(col_el.get("max", "1"))
            span = F.shift_span(first, last, at, count)
            if span is None:
                cols.remove(col_el)
                continue
            if last == F.MAX_COL:
                col_el.set("min", str(span[0]))
                continue
            col_el.set("min", str(span[0]))
            col_el.set("max", str(min(span[1], F.MAX_COL)))
        if not len(cols):
            sheet.root.remove(cols)

    def _rewrite_outside_cells(self, rewrite: Callable[[str, str | None], str]) -> None:
        """Every formula that isn't in a cell: names, formats, validations, tables, charts."""
        book = self.book
        for item in book.defined_names():
            if item.text:
                new = rewrite(item.text, None)
                if new != item.text:
                    item.text = new
                    book.touch(book.wb_path)
        for sheet in book.all_sheets():
            changed = False
            for tag in ("formula", "formula1", "formula2"):
                for el in sheet.root.iter(_q(tag)):
                    if el.text:
                        new = rewrite(el.text, sheet.name)
                        if new != el.text:
                            el.text = new
                            changed = True
            for el in sheet.root.iter(f"{{{XM_NS}}}f"):
                if el.text:
                    new = rewrite(el.text, sheet.name)
                    if new != el.text:
                        el.text = new
                        changed = True
            if changed:
                sheet.touch()
        for table in book.tables():
            root = book.part(table.part)
            changed = False
            for tag in ("calculatedColumnFormula", "totalsRowFormula"):
                for el in root.iter(_q(tag)):
                    if el.text:
                        new = rewrite(el.text, table.sheet)
                        if new != el.text:
                            el.text = new
                            changed = True
            if changed:
                book.touch(table.part)
        for part in book.parts_named(r"xl/charts/chart\d*\.xml"):
            root = book.part(part)
            changed = False
            for el in root.iter(f"{{{C_NS}}}f"):
                if el.text:
                    new = rewrite(el.text, None)
                    if new != el.text:
                        el.text = new
                        changed = True
            if changed:
                book.touch(part)

    def _shift_sheet_furniture(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        root = sheet.root
        merge_parent = root.find(_q("mergeCells"))
        if merge_parent is not None:
            for merge in list(merge_parent):
                moved = _shift_ref_text(merge.get("ref") or "", axis, at, count)
                box = F.parse_box(moved) if moved else None
                if box is None or box.is_cell:
                    merge_parent.remove(merge)
                else:
                    merge.set("ref", moved)
            if len(merge_parent):
                merge_parent.set("count", str(len(merge_parent)))
            else:
                root.remove(merge_parent)
        for tag, attribute in (("autoFilter", "ref"), ("sortState", "ref"), ("sortCondition", "ref"),
                               ("hyperlink", "ref"), ("conditionalFormatting", "sqref"),
                               ("dataValidation", "sqref"), ("ignoredError", "sqref"), ("protectedRange", "sqref")):
            for el in list(root.iter(_q(tag))):
                raw = el.get(attribute)
                if not raw:
                    continue
                moved = _shift_sqref(raw, axis, at, count)
                if moved:
                    el.set(attribute, moved)
                else:
                    el.getparent().remove(el)
        for el in list(root.iter(f"{{{XM_NS}}}sqref")):
            if el.text:
                moved = _shift_sqref(el.text, axis, at, count)
                if moved:
                    el.text = moved
                else:
                    holder = el.getparent()
                    holder.getparent().remove(holder)
        breaks = root.find(_q("rowBreaks" if axis == "row" else "colBreaks"))
        if breaks is not None:
            for brk in list(breaks):
                moved_edge = F._shift_edge(int(brk.get("id", "0")), at, count)
                if moved_edge is None:
                    breaks.remove(brk)
                else:
                    brk.set("id", str(moved_edge))
        parent = root.find(_q("dataValidations"))
        if parent is not None and len(parent):
            parent.set("count", str(len(parent)))
        # The schema doesn't allow these empty.
        for tag in ("dataValidations", "hyperlinks", "ignoredErrors", "protectedRanges"):
            parent = root.find(_q(tag))
            if parent is not None and not len(parent):
                root.remove(parent)
        sheet.touch()

    def _shift_tables(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        for table in list(self.book.tables()):
            if table.sheet.casefold() != sheet.name.casefold():
                continue
            root = self.book.part(table.part)
            moved = _shift_ref_text(root.get("ref") or "", axis, at, count)
            if moved is None:
                self._drop_table(sheet, table)
                continue
            root.set("ref", moved)
            for tag in ("autoFilter", "sortState"):
                for el in root.iter(_q(tag)):
                    if el.get("ref"):
                        shifted = _shift_ref_text(el.get("ref"), axis, at, count)
                        if shifted:
                            el.set("ref", shifted)
            for el in root.iter(_q("sortCondition")):
                if el.get("ref"):
                    shifted = _shift_ref_text(el.get("ref"), axis, at, count)
                    if shifted:
                        el.set("ref", shifted)
            self.book.touch(table.part)
        self.book.forget_tables()

    def _drop_table(self, sheet: Sheet, table: TableInfo) -> None:
        for rel_id, part in sheet.rels.of_type("table"):
            if part == table.part:
                sheet.rels.remove(rel_id)
                parts = sheet.root.find(_q("tableParts"))
                if parts is not None:
                    for item in list(parts):
                        if item.get(f"{{{R_NS}}}id") == rel_id:
                            parts.remove(item)
                    if len(parts):
                        parts.set("count", str(len(parts)))
                    else:
                        sheet.root.remove(parts)
                self.book.remove_part(part)

    def _shift_drawings(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        tag = f"{{{XDR}}}{'row' if axis == 'row' else 'col'}"
        for _, part in sheet.rels.of_type("drawing"):
            if not self.book.has(part):
                continue
            root = self.book.part(part)
            changed = False
            for el in root.iter(tag):
                position = int(el.text or "0") + 1
                moved = F._shift_edge(position, at, count)
                if moved is None:
                    moved = at
                if moved != position:
                    el.text = str(moved - 1)
                    changed = True
            if changed:
                self.book.touch(part)

    def _shift_comments(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        for _, part in sheet.rels.of_type("comments"):
            if not self.book.has(part):
                continue
            root = self.book.part(part)
            parent = root.find(_q("commentList"))
            for comment in list(parent) if parent is not None else []:
                moved = _shift_ref_text(comment.get("ref") or "", axis, at, count)
                if moved is None:
                    parent.remove(comment)
                else:
                    comment.set("ref", moved)
            self.book.touch(part)
        for _, part in sheet.rels.of_type("threadedComment"):
            if not self.book.has(part):
                continue
            root = self.book.part(part)
            for comment in list(root):
                moved = _shift_ref_text(comment.get("ref") or "", axis, at, count)
                if moved is None:
                    root.remove(comment)
                else:
                    comment.set("ref", moved)
            self.book.touch(part)
        for _, part in sheet.rels.of_type("vmlDrawing"):
            if not self.book.has(part):
                continue
            root = self.book.part(part)
            x_ns = "urn:schemas-microsoft-com:office:excel"
            v_ns = "urn:schemas-microsoft-com:vml"
            changed = False
            for shape in list(root.iter(f"{{{v_ns}}}shape")):
                data = shape.find(f"{{{x_ns}}}ClientData")
                if data is None or data.get("ObjectType") != "Note":
                    continue
                row_el, col_el = data.find(f"{{{x_ns}}}Row"), data.find(f"{{{x_ns}}}Column")
                if row_el is None or col_el is None:
                    continue
                target = row_el if axis == "row" else col_el
                moved = F._shift_edge(int(target.text or "0") + 1, at, count)
                if moved is None:
                    shape.getparent().remove(shape)
                else:
                    target.text = str(moved - 1)
                changed = True
            if changed:
                self.book.touch(part)

    def _shift_pivot_sources(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        for part in self.book.parts_named(r"xl/pivotCache/pivotCacheDefinition\d*\.xml"):
            root = self.book.part(part)
            for source in root.iter(_q("worksheetSource")):
                if (source.get("sheet") or "").casefold() == sheet.name.casefold() and source.get("ref"):
                    moved = _shift_ref_text(source.get("ref"), axis, at, count)
                    if moved:
                        source.set("ref", moved)
                        self.book.touch(part)

    def _copy_format_into_new(self, sheet: Sheet, axis: str, at: int, count: int) -> None:
        """New rows and columns take the format of the one before, as in Excel."""
        sheet.reindex()
        if at <= 1:
            return
        if axis == "row":
            source = sheet.row_els.get(at - 1)
            if source is None:
                return
            for offset in range(count):
                row = at + offset
                el = sheet.row_el(row)
                for attribute in ("ht", "customHeight", "s", "customFormat"):
                    if source.get(attribute):
                        el.set(attribute, source.get(attribute))
                for col, cell in sorted(sheet.grid.get(at - 1, {}).items()):
                    if cell.get("s"):
                        sheet.cell(row, col).set("s", cell.get("s"))
        else:
            for row, cells in list(sheet.grid.items()):
                source_cell = cells.get(at - 1)
                if source_cell is None or not source_cell.get("s"):
                    continue
                for offset in range(count):
                    sheet.cell(row, at + offset).set("s", source_cell.get("s"))
            cols = sheet.root.find(_q("cols"))
            if cols is not None:
                for col_el in cols:
                    if int(col_el.get("min", "0")) <= at - 1 <= int(col_el.get("max", "0")) and col_el.get("width"):
                        self._set_width(sheet, at, at + count - 1, float(col_el.get("width")))
                        break
        sheet.touch()

    # Format and layout

    def _op_format(self, op: dict[str, Any]) -> None:
        sheet, box = self._target(op, "range", "cell")
        self._guard(sheet)
        changes = self._format_changes(op)
        if not changes:
            raise XlsxError(f"say what to change: {', '.join(FORMAT_KEYS)}.")
        box = self._clip(sheet, box)
        if box.area > MAX_CELLS_PER_CALL:
            raise XlsxError(f"{box_text(box)} is too many cells for one call.")
        cells = ((r, c) for r in range(box.r1, box.r2 + 1) for c in range(box.c1, box.c2 + 1))
        self._count("cells formatted", self._apply_format(sheet, cells, changes))

    def _set_width(self, sheet: Sheet, first: int, last: int, width: float) -> None:
        cols = sheet.root.find(_q("cols"))
        if cols is None:
            cols = _insert_ordered(sheet.root, etree.Element(_q("cols")), _SHEET_ORDER)
        kept = []
        for el in list(cols):
            low, high = int(el.get("min", "1")), int(el.get("max", "1"))
            cols.remove(el)
            if high < first or low > last:
                kept.append((low, high, el))
                continue
            if low < first:
                left = copy.deepcopy(el)
                left.set("max", str(first - 1))
                kept.append((low, first - 1, left))
            if high > last:
                right = copy.deepcopy(el)
                right.set("min", str(last + 1))
                kept.append((last + 1, high, right))
        kept.append((first, last, etree.Element(_q("col"), min=str(first), max=str(last),
                                                width=number_text(width), customWidth="1")))
        for _, _, el in sorted(kept, key=lambda item: item[0]):
            cols.append(el)
        sheet.touch()

    def _shown_length(self, sheet: Sheet, row: int, col: int, computed: dict[Any, Any] | None = None) -> int:
        key = (sheet.name, row, col)
        value = computed.get(key) if computed and key in computed else sheet.stored(row, col)
        if value is None:
            return 0
        fmt = sheet.number_format(row, col)
        text = F.format_value(value, fmt, self.book.date1904)
        text = text if text is not None else str(value)
        return max((len(line) for line in text.split("\n")), default=0)

    def auto_width(self, sheet: Sheet, first: int, last: int, computed: dict[Any, Any] | None = None) -> None:
        for col in range(first, last + 1):
            if (sheet.name, col) in self.chosen_widths:
                continue  # whoever set it meant it
            longest = 0
            for row, cells in sheet.grid.items():
                if col not in cells:
                    continue
                if sheet.merged_at(row, col) is not None:
                    continue
                longest = max(longest, self._shown_length(sheet, row, col, computed))
            if longest:
                self._set_width(sheet, col, col, min(max(longest + 2, 8), 60))

    def _op_width(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        self._guard(sheet)
        raw = str(op.get("columns") or op.get("column") or op.get("range") or "").strip().upper().replace("$", "")
        if not raw:
            raise XlsxError("needs `columns`, e.g. \"A\" or \"B:D\".")
        left, _, right = raw.partition(":")
        try:
            first, last = _col_index(left), _col_index(right or left)
        except (ValueError, KeyError):
            raise XlsxError(f"{raw!r} is not a column or columns.") from None
        first, last = min(first, last), max(first, last)
        width = op.get("width", "auto")
        if str(width).lower() == "auto":
            self.auto_width(sheet, first, last)
        else:
            value = float(width)
            if not 0 < value <= 255:
                raise XlsxError("A column is 0–255 characters wide.")
            self._set_width(sheet, first, last, value)
            self.chosen_widths.update((sheet.name, c) for c in range(first, last + 1))
        self._count("column widths")

    def _op_height(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        self._guard(sheet)
        raw = str(op.get("at") or op.get("rows") or op.get("row") or "").strip()
        left, _, right = raw.partition(":")
        try:
            first, last = int(left), int(right or left)
            height = float(op.get("height"))
        except (TypeError, ValueError):
            raise XlsxError("needs `at` (\"3\" or \"3:5\") and `height` in points.") from None
        if not 0 <= height <= 409:
            raise XlsxError("A row is 0–409 points high.")
        for row in range(min(first, last), max(first, last) + 1):
            el = sheet.row_el(row)
            el.set("ht", number_text(height))
            el.set("customHeight", "1")
        sheet.touch()
        self._count("row heights")

    def _op_freeze(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        self._guard(sheet)
        self.freeze(sheet, op.get("cell"))
        self._count("panes set")

    def freeze(self, sheet: Sheet, cell: Any) -> None:
        views = sheet.root.find(_q("sheetViews"))
        if views is None:
            views = _insert_ordered(sheet.root, etree.Element(_q("sheetViews")), _SHEET_ORDER)
        view = views.find(_q("sheetView"))
        if view is None:
            view = etree.SubElement(views, _q("sheetView"), workbookViewId="0")
        for old in view.findall(_q("pane")) + view.findall(_q("selection")):
            view.remove(old)
        if cell in (None, "", "none"):
            sheet.touch()
            return
        _, box = parse_range(str(cell))
        row, col = box.r1, box.c1
        if row == 1 and col == 1:
            sheet.touch()
            return
        attributes: dict[str, str] = {}
        if col > 1:
            attributes["xSplit"] = str(col - 1)
        if row > 1:
            attributes["ySplit"] = str(row - 1)
        active = "bottomRight" if row > 1 and col > 1 else ("bottomLeft" if row > 1 else "topRight")
        attributes.update(topLeftCell=addr(row, col), activePane=active, state="frozen")
        pane = etree.Element(_q("pane"), **attributes)
        view.insert(0, pane)
        etree.SubElement(view, _q("selection"), pane=active, activeCell=addr(row, col), sqref=addr(row, col))
        sheet.touch()

    def _op_merge(self, op: dict[str, Any]) -> None:
        sheet, box = self._target(op, "range")
        self._guard(sheet)
        if box.is_cell:
            raise XlsxError("A merge needs at least two cells.")
        for other in sheet.merges:
            if not (other.r2 < box.r1 or other.r1 > box.r2 or other.c2 < box.c1 or other.c1 > box.c2):
                raise XlsxError(f"{box_text(box)} overlaps the merged cells {box_text(other)}.")
        for table in self.book.tables():
            if table.sheet == sheet.name and not (table.r2 < box.r1 or table.r1 > box.r2 or
                                                  table.c2 < box.c1 or table.c1 > box.c2):
                raise XlsxError(f"Cells inside table {table.name} can't be merged.")
        for r in range(box.r1, box.r2 + 1):
            for c in range(box.c1, box.c2 + 1):
                if (r, c) != (box.r1, box.c1) and (sheet.stored(r, c) is not None or (r, c) in sheet.formula_els):
                    raise XlsxError(f"{addr(r, c)} isn't empty and a merge would lose it. Clear it first.")
        parent = sheet.root.find(_q("mergeCells"))
        if parent is None:
            parent = _insert_ordered(sheet.root, etree.Element(_q("mergeCells")), _SHEET_ORDER)
        etree.SubElement(parent, _q("mergeCell"), ref=box_text(box))
        parent.set("count", str(len(parent)))
        sheet.merges.append(box)
        sheet.touch()
        self._count("merges")

    def _op_unmerge(self, op: dict[str, Any]) -> None:
        sheet, box = self._target(op, "range", "cell")
        self._guard(sheet)
        parent = sheet.root.find(_q("mergeCells"))
        found = 0
        for merge in list(parent) if parent is not None else []:
            other = F.parse_box(merge.get("ref") or "")
            if other is not None and not (other.r2 < box.r1 or other.r1 > box.r2 or
                                          other.c2 < box.c1 or other.c1 > box.c2):
                parent.remove(merge)
                sheet.merges = [m for m in sheet.merges if m != other]
                found += 1
        if not found:
            raise XlsxError(f"No merged cells at {box_text(box)}.")
        if len(parent):
            parent.set("count", str(len(parent)))
        else:
            sheet.root.remove(parent)
        sheet.touch()
        self._count("unmerges", found)

    def _op_filter(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        self._guard(sheet)
        existing = sheet.root.find(_q("autoFilter"))
        raw = op.get("range")
        if raw in (None, "", "none"):
            if existing is not None:
                sheet.root.remove(existing)
                sheet.touch()
            self._count("filters")
            return
        _, box = parse_range(str(raw))
        box = self._clip(sheet, box)
        if existing is None:
            existing = _insert_ordered(sheet.root, etree.Element(_q("autoFilter")), _SHEET_ORDER)
        existing.set("ref", box_text(box))
        sheet.touch()
        self._count("filters")

    # Sheets

    def _check_sheet_name(self, name: str, *, ignoring: str | None = None) -> str:
        clean = str(name or "").strip()
        if not clean or len(clean) > 31 or re.search(r"[\[\]:*?/\\]", clean) or clean.startswith("'") \
                or clean.endswith("'") or clean.casefold() == "history":
            raise XlsxError(f"{name!r} can't be a sheet name (1–31 characters, none of [ ] : * ? / \\, "
                            "no quote at the ends, not \"History\").")
        for el in self.book.sheet_els:
            other = el.get("name") or ""
            if other.casefold() == clean.casefold() and other.casefold() != (ignoring or "").casefold():
                raise XlsxError(f"There is already a sheet called {other!r}.")
        return clean

    def _guard_structure(self) -> None:
        protection = self.book.wb.find(_q("workbookProtection"))
        if protection is not None and protection.get("lockStructure") in ("1", "true"):
            raise XlsxError("The workbook's structure is protected. Ask the user to unprotect it in Excel first.")

    def add_sheet(self, name: str, position: int | None = None) -> Sheet:
        book = self.book
        clean = self._check_sheet_name(name)
        path, _ = book._next_name("xl/worksheets/sheet{}.xml")
        root = etree.fromstring(
            f'<worksheet xmlns="{MAIN}" xmlns:r="{R_NS}"><sheetData/></worksheet>'.encode())
        book.add_part(path, root, CT_WORKSHEET)
        rel_id = book.wb_rels.add("worksheet", path)
        parent = book.wb.find(_q("sheets"))
        taken = [int(el.get("sheetId", "0")) for el in parent]
        el = etree.Element(_q("sheet"), name=clean, sheetId=str(max([0, *taken]) + 1))
        el.set(f"{{{R_NS}}}id", rel_id)
        if position is None or position >= len(parent):
            parent.append(el)
        else:
            self._reposition(lambda: parent.insert(max(position, 0), el))
        book.touch(book.wb_path)
        book._refresh_sheets()
        return book.sheet(clean)

    def _reposition(self, change: Callable[[], None]) -> None:
        """Change the sheet order; sheet-scoped names and the active tab point by position."""
        book = self.book
        parent = book.wb.find(_q("sheets"))
        before = list(parent)
        change()
        after = list(parent)
        index_of = {id(el): i for i, el in enumerate(after)}
        remap = {i: index_of.get(id(el)) for i, el in enumerate(before)}
        for item in book.defined_names():
            scope = item.get("localSheetId")
            if scope is not None:
                moved = remap.get(int(scope))
                if moved is None:
                    item.getparent().remove(item)
                else:
                    item.set("localSheetId", str(moved))
        views = book.wb.find(_q("bookViews"))
        if views is not None:
            for view in views:
                for attribute in ("activeTab", "firstSheet"):
                    current = view.get(attribute)
                    if current is not None:
                        moved = remap.get(int(current))
                        view.set(attribute, str(moved if moved is not None else 0))
        book.touch(book.wb_path)

    def _op_add_sheet(self, op: dict[str, Any]) -> None:
        self._guard_structure()
        position = op.get("position")
        sheet = self.add_sheet(str(op.get("name") or op.get("sheet") or ""),
                               int(position) - 1 if position not in (None, "") else None)
        if op.get("rows"):
            self.fill_new_sheet(sheet, op)
        self._count("sheets added")

    def fill_new_sheet(self, sheet: Sheet, spec: dict[str, Any]) -> None:
        rows = spec.get("rows") or []
        if not isinstance(rows, list):
            raise XlsxError(f"Sheet {sheet.name!r}: `rows` must be a list of rows.")
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        for r, row in enumerate(rows, 1):
            if not isinstance(row, list):
                raise XlsxError(f"Sheet {sheet.name!r}: row {r} of `rows` is not a list.")
            for c, value in enumerate(row, 1):
                if value is not None and value != "":
                    self._write(sheet, r, c, value)
        if spec.get("header") and rows:
            width = max(len(row) for row in rows)
            cells = [(1, c) for c in range(1, width + 1)]
            self._apply_format(sheet, cells, {"bold": True, "border_bottom": "thin", "fill": "F2F2F2"})
            self.freeze(sheet, "A2")
        if rows:
            self.pending_auto_width.append(sheet.name)

    def _op_rename_sheet(self, op: dict[str, Any]) -> None:
        self._guard_structure()
        sheet = self._sheet(op)
        new = self._check_sheet_name(op.get("name") or op.get("new_name") or "", ignoring=sheet.name)
        self.rename_sheet(sheet, new)
        self._count("sheets renamed")

    def rename_sheet(self, sheet: Sheet, new: str) -> None:
        old = sheet.name
        if old == new:
            return
        book = self.book

        def rewrite(text: str, own_sheet: str | None) -> str:
            return F.rename_sheet_in(text, old, new)

        self._unshare_formulas_reading(old)
        for other in book.all_sheets():
            for el in other.formula_els.values():
                if el.text:
                    changed = rewrite(el.text, other.name)
                    if changed != el.text:
                        el.text = changed
                        other.touch()
            for link in other.root.iter(_q("hyperlink")):
                location = link.get("location")
                if location:
                    changed = rewrite(location, None)
                    if changed != location:
                        link.set("location", changed)
                        other.touch()
        self._rewrite_outside_cells(rewrite)
        for part in book.parts_named(r"xl/pivotCache/pivotCacheDefinition\d*\.xml"):
            for source in book.part(part).iter(_q("worksheetSource")):
                if (source.get("sheet") or "").casefold() == old.casefold():
                    source.set("sheet", new)
                    book.touch(part)
        self.before = {((new, k[1], k[2]) if k[0] == old else k): v for k, v in self.before.items()}
        self.seeds = {((new, k[1], k[2]) if k[0] == old else k) for k in self.seeds}
        self.formula_seeds = {((new, k[1], k[2]) if k[0] == old else k) for k in self.formula_seeds}
        self.new_numbers = {((new, k[1], k[2]) if k[0] == old else k) for k in self.new_numbers}
        self.written_formulas = [
            (new if w[0] == old else w[0], w[1], w[2], rewrite(w[3], None)) for w in self.written_formulas]
        sheet.el.set("name", new)
        book.touch(book.wb_path)
        book._by_name = {s.name.casefold(): s for s in book.sheets}
        for table in book.tables():
            if table.sheet == old:
                table.sheet = new

    def _op_delete_sheet(self, op: dict[str, Any]) -> None:
        self._guard_structure()
        sheet = self._sheet(op)
        book = self.book
        visible = [el for el in book.sheet_els if (el.get("state") or "visible") == "visible"]
        if len(book.sheet_els) == 1 or (len(visible) == 1 and sheet.el is visible[0]):
            raise XlsxError("A workbook needs at least one visible sheet.")
        users: list[str] = []
        for other in book.all_sheets():
            if other is sheet:
                continue
            for (row, col) in other.formula_els:
                text = other.formula(row, col) or ""
                if F.mentions_sheet(text, sheet.name):
                    users.append(f"{other.name}!{addr(row, col)}")
        for item in book.defined_names():
            if item.text and F.mentions_sheet(item.text, sheet.name) and item.get("localSheetId") is None:
                users.append(f"the name {item.get('name')}")
        own_charts = {chart for _, drawing in sheet.rels.of_type("drawing")
                      for _, chart in _Rels(book, drawing).of_type("chart")}
        for part in book.parts_named(r"xl/charts/chart\d*\.xml"):
            if part not in own_charts and any(
                    el.text and F.mentions_sheet(el.text, sheet.name) for el in book.part(part).iter(f"{{{C_NS}}}f")):
                users.append("a chart on another sheet")
        for part in book.parts_named(r"xl/pivotCache/pivotCacheDefinition\d*\.xml"):
            if any((source.get("sheet") or "").casefold() == sheet.name.casefold()
                   for source in book.part(part).iter(_q("worksheetSource"))):
                users.append("a pivot table")
        if users:
            shown = ", ".join(users[:5]) + (f" and {len(users) - 5} more" if len(users) > 5 else "")
            raise XlsxError(f"{shown} read from sheet {sheet.name!r}. Change those first, so nothing turns into #REF!.")
        rel_id = sheet.el.get(f"{{{R_NS}}}id")
        parent = book.wb.find(_q("sheets"))
        self._reposition(lambda: parent.remove(sheet.el))
        book.wb_rels.remove(rel_id)
        book.remove_part(sheet.path)
        self._drop_unreachable_parts()
        name = sheet.name
        book._refresh_sheets()
        book.forget_tables()
        self.before = {k: v for k, v in self.before.items() if k[0] != name}
        self.seeds = {k for k in self.seeds if k[0] != name}
        self.formula_seeds = {k for k in self.formula_seeds if k[0] != name}
        self.new_numbers = {k for k in self.new_numbers if k[0] != name}
        self.written_formulas = [w for w in self.written_formulas if w[0] != name]
        self._count("sheets deleted")

    def _drop_unreachable_parts(self) -> None:
        """Whatever only the deleted sheet used (its charts, tables, notes) goes with it."""
        book = self.book
        reachable: set[str] = set()
        root_rels = _Rels(book, "")
        root_rels.path, root_rels.folder = "_rels/.rels", ""
        queue = [part for _, _, part in root_rels.items()]
        while queue:
            part = queue.pop()
            if part in reachable or not book.has(part):
                continue
            reachable.add(part)
            queue.extend(p for _, _, p in _Rels(book, part).items())
        for name in list(book._order):
            if name == "[Content_Types].xml" or name.endswith(".rels"):
                continue
            if name not in reachable:
                book.remove_part(name)

    def _op_move_sheet(self, op: dict[str, Any]) -> None:
        self._guard_structure()
        sheet = self._sheet(op)
        position = op.get("position")
        if position in (None, ""):
            raise XlsxError("needs `position` (1 = first).")
        parent = self.book.wb.find(_q("sheets"))

        def move() -> None:
            parent.remove(sheet.el)
            parent.insert(max(int(position) - 1, 0), sheet.el)

        self._reposition(move)
        self.book._refresh_sheets()
        self._count("sheets moved")

    # Names

    def _op_define_name(self, op: dict[str, Any]) -> None:
        name = str(op.get("name") or "").strip()
        if not re.fullmatch(r"[^\W\d][\w.\\]*", name, re.U) or F.parse_box(name) is not None:
            raise XlsxError(f"{name!r} can't be a name (start with a letter; no spaces; not like a cell address).")
        refers_to = str(op.get("refers_to") or op.get("value") or "").strip()
        if not refers_to:
            raise XlsxError("needs `refers_to`, e.g. Inputs!$B$2.")
        stored = F.normalize_for_storage(refers_to)
        if op.get("sheet"):
            scope = self.book.sheet_position(self._sheet(op).name)
        else:
            scope = None
        parent = self.book.wb.find(_q("definedNames"))
        if parent is None:
            parent = _insert_ordered(self.book.wb, etree.Element(_q("definedNames")), _WORKBOOK_ORDER)
        element = None
        for item in parent:
            same_scope = item.get("localSheetId") == (str(scope) if scope is not None else None)
            if (item.get("name") or "").casefold() == name.casefold() and same_scope:
                element = item
        if element is None:
            element = etree.SubElement(parent, _q("definedName"), name=name)
            if scope is not None:
                element.set("localSheetId", str(scope))
        element.text = stored
        self.book.touch(self.book.wb_path)
        # Whatever already reads this name now reads something else.
        self.full_recalculate = True
        self.written_formulas.append((self._name_home(op), 0, 0, stored))
        self._count("names defined")

    def _name_home(self, op: dict[str, Any]) -> str:
        return str(op.get("sheet") or self.book.sheets[0].name)

    def _op_delete_name(self, op: dict[str, Any]) -> None:
        name = str(op.get("name") or "").strip()
        parent = self.book.wb.find(_q("definedNames"))
        found = [i for i in (list(parent) if parent is not None else [])
                 if (i.get("name") or "").casefold() == name.casefold()]
        if not found:
            raise XlsxError(f"There is no name {name!r}.")
        users = []
        for sheet in self.book.all_sheets():
            for (row, col) in sheet.formula_els:
                text = sheet.formula(row, col) or ""
                if name.casefold() in text.casefold() and any(
                        isinstance(o, F.NameOperand) and o.name.casefold() == name.casefold()
                        for o in F.compile_formula(text).operands):
                    users.append(f"{sheet.name}!{addr(row, col)}")
        if users:
            raise XlsxError(f"{', '.join(users[:5])} use the name {name}. Change them first.")
        for item in found:
            parent.remove(item)
        if not len(parent):
            self.book.wb.remove(parent)
        self.book.touch(self.book.wb_path)
        self._count("names deleted")

    # Tables

    def _op_add_table(self, op: dict[str, Any]) -> None:
        sheet, box = self._target(op, "range")
        self._guard(sheet)
        box = self._clip(sheet, box)
        if box.r2 <= box.r1:
            raise XlsxError("A table needs a header row and at least one data row.")
        book = self.book
        for other in book.tables():
            if other.sheet == sheet.name and not (other.r2 < box.r1 or other.r1 > box.r2 or
                                                  other.c2 < box.c1 or other.c1 > box.c2):
                raise XlsxError(f"{box_text(box)} overlaps table {other.name}.")
        for merged in sheet.merges:
            if not (merged.r2 < box.r1 or merged.r1 > box.r2 or merged.c2 < box.c1 or merged.c1 > box.c2):
                raise XlsxError(f"{box_text(box)} overlaps the merged cells {box_text(merged)}.")
        headers: list[str] = []
        for c in range(box.c1, box.c2 + 1):
            value = sheet.stored(box.r1, c)
            if (box.r1, c) in sheet.formula_els or not isinstance(value, str) or not value.strip():
                raise XlsxError(f"{addr(box.r1, c)} must hold text: a table's first row names its columns.")
            if value.casefold() in (h.casefold() for h in headers):
                raise XlsxError(f"Column name {value!r} appears twice in {box_text(box)}.")
            headers.append(value)
        name = str(op.get("name") or "").strip()
        taken = {t.name.casefold() for t in book.tables()} | {
            (i.get("name") or "").casefold() for i in book.defined_names()}
        if name:
            if not re.fullmatch(r"[^\W\d][\w.]*", name, re.U) or F.parse_box(name) is not None:
                raise XlsxError(f"{name!r} can't be a table name.")
            if name.casefold() in taken:
                raise XlsxError(f"The name {name!r} is taken.")
        else:
            number = 1
            while f"Table{number}".casefold() in taken:
                number += 1
            name = f"Table{number}"
        ids = []
        for part in book.parts_named(r"xl/tables/table\d*\.xml"):
            ids.append(int(book.part(part).get("id", "0")))
        table_id = max([0, *ids]) + 1
        path, _ = book._next_name("xl/tables/table{}.xml")
        root = etree.Element(_q("table"), nsmap={None: MAIN}, id=str(table_id), name=name, displayName=name,
                             ref=box_text(box), totalsRowShown="0")
        etree.SubElement(root, _q("autoFilter"), ref=box_text(box))
        columns = etree.SubElement(root, _q("tableColumns"), count=str(len(headers)))
        for index, header in enumerate(headers, 1):
            etree.SubElement(columns, _q("tableColumn"), id=str(index), name=header)
        etree.SubElement(root, _q("tableStyleInfo"), name=str(op.get("style") or "TableStyleMedium2"),
                         showFirstColumn="0", showLastColumn="0", showRowStripes="1", showColumnStripes="0")
        book.add_part(path, root, CT_TABLE)
        rel_id = sheet.rels.add("table", path)
        parts = sheet.root.find(_q("tableParts"))
        if parts is None:
            parts = _insert_ordered(sheet.root, etree.Element(_q("tableParts")), _SHEET_ORDER)
        item = etree.SubElement(parts, _q("tablePart"))
        item.set(f"{{{R_NS}}}id", rel_id)
        parts.set("count", str(len(parts)))
        filter_el = sheet.root.find(_q("autoFilter"))
        if filter_el is not None and filter_el.get("ref") == box_text(box):
            sheet.root.remove(filter_el)  # a table brings its own
        book.forget_tables()
        sheet.touch()
        self._count("tables added")

    # Charts

    def _op_add_chart(self, op: dict[str, Any]) -> None:
        sheet = self._sheet(op)
        self._guard(sheet)
        kind = str(op.get("type") or "column").lower()
        if kind not in _CHART_KINDS:
            raise XlsxError(f"chart type {kind!r}: use {', '.join(_CHART_KINDS)}.")
        series = self._chart_series(sheet, op)
        categories = self._chart_ref(sheet, op["categories"]) if op.get("categories") else None
        path, _ = self.book._next_name("xl/charts/chart{}.xml")
        root = _chart_xml(kind, series, categories, str(op.get("title") or ""),
                          bool(op.get("stacked")), str(op.get("x_title") or ""), str(op.get("y_title") or ""),
                          op.get("legend", len(series) > 1))
        self.book.add_part(path, root, CT_CHART)
        self._anchor_chart(sheet, path, op)
        self.new_charts.append(path)
        self._count("charts added")

    def _chart_ref(self, sheet: Sheet, raw: Any) -> str:
        sheet_name, box = parse_range(str(raw))
        owner = self.book.sheet(sheet_name) if sheet_name else sheet
        box = self._clip(owner, box)
        return f"{F.quote_sheet(owner.name)}!${get_column_letter(box.c1)}${box.r1}" + (
            "" if box.is_cell else f":${get_column_letter(box.c2)}${box.r2}")

    def _chart_series(self, sheet: Sheet, op: dict[str, Any]) -> list[dict[str, str]]:
        raw = op.get("series")
        if not raw and op.get("values"):
            sheet_name, box = parse_range(str(op["values"]))
            owner = self.book.sheet(sheet_name) if sheet_name else sheet
            box = self._clip(owner, box)
            raw = []
            if box.c1 == box.c2 or box.r1 == box.r2:
                raw.append({"values": op["values"], "name": op.get("name")})
            else:
                for c in range(box.c1, box.c2 + 1):
                    raw.append({"values": f"{F.quote_sheet(owner.name)}!{addr(box.r1, c)}:{addr(box.r2, c)}"})
        if not isinstance(raw, list) or not raw:
            raise XlsxError("needs `values` (a range) or `series` (a list of {name, values}).")
        out: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("values"):
                raise XlsxError("each of `series` needs `values`.")
            values = self._chart_ref(sheet, item["values"])
            entry = {"values": values}
            if item.get("name"):
                entry["name"] = str(item["name"])
            else:
                sheet_name, box = parse_range(str(item["values"]))
                owner = self.book.sheet(sheet_name) if sheet_name else sheet
                if box.r1 > 1 and isinstance(owner.stored(box.r1 - 1, box.c1), str):
                    entry["name_ref"] = f"{F.quote_sheet(owner.name)}!${get_column_letter(box.c1)}${box.r1 - 1}"
            if item.get("x"):
                entry["x"] = self._chart_ref(sheet, item["x"])
            out.append(entry)
        return out

    def _anchor_chart(self, sheet: Sheet, chart_path: str, op: dict[str, Any]) -> None:
        book = self.book
        rels = sheet.rels.of_type("drawing")
        if rels and book.has(rels[0][1]):
            drawing_path = rels[0][1]
            root = book.part(drawing_path)
        else:
            drawing_path, _ = book._next_name("xl/drawings/drawing{}.xml")
            root = etree.Element(f"{{{XDR}}}wsDr", nsmap={"xdr": XDR, "a": A_NS})
            book.add_part(drawing_path, root, CT_DRAWING)
            rel_id = sheet.rels.add("drawing", drawing_path)
            holder = etree.Element(_q("drawing"), nsmap={"r": R_NS})
            holder.set(f"{{{R_NS}}}id", rel_id)
            _insert_ordered(sheet.root, holder, _SHEET_ORDER)
        rel_id = _Rels(book, drawing_path).add("chart", chart_path)
        book.touch(drawing_path)
        if op.get("anchor"):
            _, box = parse_range(str(op["anchor"]))
            row, col = box.r1, box.c1
        else:
            used_rows, used_cols = sheet.used()
            row, col = 2, used_cols + 2 if used_cols else 2
        width = float(op.get("width") or 16)
        height = float(op.get("height") or 8.5)
        if not (0 < width <= 100 and 0 < height <= 100):
            raise XlsxError("Chart width and height are in centimetres (up to 100).")
        emu = 360000
        ids = [int(el.get("id", "1")) for el in root.iter(f"{{{XDR}}}cNvPr")]
        shape_id = max([1, *ids]) + 1
        anchor = etree.fromstring(
            f'<xdr:oneCellAnchor xmlns:xdr="{XDR}" xmlns:a="{A_NS}" xmlns:r="{R_NS}" xmlns:c="{C_NS}">'
            f"<xdr:from><xdr:col>{col - 1}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{row - 1}</xdr:row>"
            f'<xdr:rowOff>0</xdr:rowOff></xdr:from><xdr:ext cx="{int(width * emu)}" cy="{int(height * emu)}"/>'
            f'<xdr:graphicFrame macro=""><xdr:nvGraphicFramePr><xdr:cNvPr id="{shape_id}" name="Chart {shape_id - 1}"/>'
            '<xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr><xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '</xdr:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
            f'<c:chart r:id="{rel_id}"/></a:graphicData></a:graphic></xdr:graphicFrame><xdr:clientData/>'
            "</xdr:oneCellAnchor>".encode())
        root.append(anchor)
        sheet.touch()

    # Notes

    def _op_comment(self, op: dict[str, Any]) -> None:
        sheet, box = self._target(op, "cell")
        self._guard(sheet)
        text = op.get("text")
        if text is None:
            raise XlsxError("needs `text` (empty text removes the note).")
        self._set_note(sheet, box.r1, box.c1, str(text))
        self._count("notes" if text else "notes removed")

    def _set_note(self, sheet: Sheet, row: int, col: int, text: str) -> None:
        book = self.book
        ref = addr(row, col)
        rels = sheet.rels.of_type("comments")
        if rels and book.has(rels[0][1]):
            path = rels[0][1]
            root = book.part(path)
        else:
            if not text:
                raise XlsxError(f"{ref} has no note to remove.")
            path, _ = book._next_name("xl/comments{}.xml")
            root = etree.fromstring(
                f'<comments xmlns="{MAIN}"><authors/><commentList/></comments>'.encode())
            book.add_part(path, root, CT_COMMENTS)
            sheet.rels.add("comments", path)
        book.touch(path)
        comments = root.find(_q("commentList"))
        existing = next((c for c in comments if c.get("ref") == ref), None)
        if not text:
            if existing is None:
                raise XlsxError(f"{ref} has no note to remove.")
            comments.remove(existing)
            self._drop_note_shape(sheet, row, col)
            return
        authors = root.find(_q("authors"))
        names = [a.text or "" for a in authors]
        if self.author in names:
            author_id = names.index(self.author)
        else:
            etree.SubElement(authors, _q("author")).text = self.author
            author_id = len(names)
        if existing is None:
            # Excel wants notes in cell order.
            existing = etree.Element(_q("comment"), ref=ref)
            position = len(comments)
            for index, other in enumerate(comments):
                other_box = F.parse_box(other.get("ref") or "")
                if other_box is not None and (other_box.r1, other_box.c1) > (row, col):
                    position = index
                    break
            comments.insert(position, existing)
            self._add_note_shape(sheet, row, col)
        existing.set("authorId", str(author_id))
        for child in list(existing):
            existing.remove(child)
        body = etree.SubElement(existing, _q("text"))
        run = etree.SubElement(body, _q("r"))
        t = etree.SubElement(run, T_TAG)
        t.text = _escape_text(text)
        t.set(XML_SPACE, "preserve")

    _VML_V = "urn:schemas-microsoft-com:vml"
    _VML_O = "urn:schemas-microsoft-com:office:office"
    _VML_X = "urn:schemas-microsoft-com:office:excel"

    def _note_drawing(self, sheet: Sheet) -> tuple[str, Any]:
        book = self.book
        rels = sheet.rels.of_type("vmlDrawing")
        if rels and book.has(rels[0][1]):
            return rels[0][1], book.part(rels[0][1])
        path, number = book._next_name("xl/drawings/vmlDrawing{}.vml")
        root = etree.fromstring(
            f'<xml xmlns:v="{self._VML_V}" xmlns:o="{self._VML_O}" xmlns:x="{self._VML_X}">'
            f'<o:shapelayout v:ext="edit"><o:idmap v:ext="edit" data="{number}"/></o:shapelayout>'
            '<v:shapetype id="_x0000_t202" coordsize="21600,21600" o:spt="202" path="m,l,21600r21600,l21600,xe">'
            '<v:stroke joinstyle="miter"/><v:path gradientshapeok="t" o:connecttype="rect"/></v:shapetype>'
            "</xml>".encode())
        book._ensure_default("vml", CT_VML)
        book.add_part(path, root, None)
        rel_id = sheet.rels.add("vmlDrawing", path)
        holder = etree.Element(_q("legacyDrawing"), nsmap={"r": R_NS})
        holder.set(f"{{{R_NS}}}id", rel_id)
        _insert_ordered(sheet.root, holder, _SHEET_ORDER)
        sheet.touch()
        return path, root

    def _add_note_shape(self, sheet: Sheet, row: int, col: int) -> None:
        path, root = self._note_drawing(sheet)
        used = [s.get("id") or "" for s in root.iter(f"{{{self._VML_V}}}shape")]
        numbers = [int(m.group(1)) for s in used if (m := re.search(r"_s(\d+)", s))]
        shape_number = max([1024, *numbers]) + 1
        shape = etree.fromstring(
            f'<v:shape xmlns:v="{self._VML_V}" xmlns:o="{self._VML_O}" xmlns:x="{self._VML_X}" '
            f'id="_x0000_s{shape_number}" type="#_x0000_t202" style="position:absolute;'
            'margin-left:60pt;margin-top:2pt;width:160pt;height:70pt;z-index:1;visibility:hidden" '
            'fillcolor="#ffffe1" o:insetmode="auto"><v:fill color2="#ffffe1"/>'
            '<v:shadow on="t" color="black" obscured="t"/><v:path o:connecttype="none"/>'
            '<v:textbox style="mso-direction-alt:auto"><div style="text-align:left"/></v:textbox>'
            '<x:ClientData ObjectType="Note"><x:MoveWithCells/><x:SizeWithCells/>'
            f"<x:Anchor>{col}, 15, {max(row - 2, 0)}, 2, {col + 2}, 31, {row + 3}, 1</x:Anchor>"
            f"<x:AutoFill>False</x:AutoFill><x:Row>{row - 1}</x:Row><x:Column>{col - 1}</x:Column>"
            "</x:ClientData></v:shape>".encode())
        root.append(shape)
        self.book.touch(path)

    def _drop_note_shape(self, sheet: Sheet, row: int, col: int) -> None:
        rels = sheet.rels.of_type("vmlDrawing")
        if not rels or not self.book.has(rels[0][1]):
            return
        root = self.book.part(rels[0][1])
        x = self._VML_X
        for shape in list(root.iter(f"{{{self._VML_V}}}shape")):
            data = shape.find(f"{{{x}}}ClientData")
            if data is None:
                continue
            r, c = data.find(f"{{{x}}}Row"), data.find(f"{{{x}}}Column")
            if r is not None and c is not None and r.text == str(row - 1) and c.text == str(col - 1):
                shape.getparent().remove(shape)
                self.book.touch(rels[0][1])

    def _op_recalculate(self, op: dict[str, Any]) -> None:
        self.full_recalculate = True
        self._count("full recalculations")

    # Finishing

    def _validate_written_formulas(self) -> None:
        """Formulas may name sheets made later in the same call, so they're checked at the end."""
        book = self.book
        for sheet_name, row, col, stored in self.written_formulas:
            where = f"{sheet_name}!{addr(row, col)}" if row else "the new name"
            for operand in F.compile_formula(stored).operands:
                if isinstance(operand, Box) and operand.sheet is not None \
                        and operand.sheet.casefold() not in book._by_name:
                    raise XlsxError(f"{where}: there is no sheet {operand.sheet!r}. "
                                    f"Sheets: {', '.join(s.name for s in book.sheets)}.")
                if isinstance(operand, F.NameOperand) and book.defined_name_text(
                        operand.name, operand.sheet or sheet_name) is None:
                    raise XlsxError(f"{where}: {operand.name} is neither a cell address nor a defined name "
                                    "(define_name makes one).")
                if isinstance(operand, F.TableOperand):
                    table = book.table_named(operand.table) if operand.table else (
                        book.table_at(sheet_name, row, col) if row else None)
                    if table is None:
                        raise XlsxError(f"{where}: there is no table {operand.table or 'around this cell'!r}.")
                    for column in (operand.first_col, operand.last_col):
                        if column is not None and column.casefold() not in (c.casefold() for c in table.columns):
                            raise XlsxError(f"{where}: table {table.name} has no column {column!r}. "
                                            f"Columns: {', '.join(table.columns)}.")

    def finish(self) -> Report:
        self._validate_written_formulas()
        book = self.book
        calc = Calc(book)
        total_formulas = sum(len(s.formula_els) for s in calc.sheets.values())
        missing_results = {
            (s.name, r, c) for s in calc.sheets.values() for (r, c) in s.formula_els
            if not s.has_stored_result(r, c) and not s.is_opaque_formula(r, c)
            and s.array_owner.get((r, c), (r, c)) == (r, c)}
        if total_formulas > MAX_FORMULAS_TO_CALCULATE:
            self.report.notes.append(
                f"With {total_formulas:,} formulas this workbook isn't recalculated here: Excel "
                "recalculates it on open, and stored results may be out of date until then.")
            calc_keys: set[tuple[str, int, int]] = set()
        elif self.full_recalculate:
            calc_keys = {(s.name, r, c) for s in calc.sheets.values() for (r, c) in s.formula_els
                         if s.array_owner.get((r, c), (r, c)) == (r, c)}
        else:
            starts = set(self.seeds) | self.formula_seeds
            calc_keys = (calc.affected_by(starts) if starts else set()) | self.formula_seeds | missing_results
            calc_keys = {k for k in calc_keys if self._is_calc_cell(calc, k)}
        calc.run(calc_keys)
        self._commit(calc, calc_keys)
        for name in self.pending_auto_width:
            if name.casefold() in book._by_name:
                sheet = book.sheet(name)
                self.auto_width(sheet, 1, sheet.used()[1])
        if calc.circular:
            shown = ", ".join(f"{k[0]}!{addr(k[1], k[2])}" for k in sorted(calc.circular)[:5])
            self.report.notes.append(f"Circular reference at {shown}: Excel will warn about it on open.")
        self._note_short_ranges(calc)
        self._refresh_chart_caches(calc)
        pivots = book.parts_named(r"xl/pivotTables/pivotTable\d*\.xml")
        if pivots and (self.seeds or self.structure_changed):
            self.report.notes.append(
                f"This workbook has {len(pivots)} pivot table(s): they show the old numbers until refreshed "
                "in Excel (Data → Refresh All).")
        if total_formulas or self.written_formulas:
            book.recalculate_on_open()
        return self.report

    def _is_calc_cell(self, calc: Calc, key: tuple[str, int, int]) -> bool:
        sheet = calc.sheets.get(key[0])
        if sheet is None or (key[1], key[2]) not in sheet.formula_els:
            return False
        owner = sheet.array_owner.get((key[1], key[2]))
        return owner is None or owner == (key[1], key[2])

    def _commit(self, calc: Calc, keys: set[tuple[str, int, int]]) -> None:
        book = self.book
        report = self.report
        recalculated_seen: set[tuple[str, int, int]] = set()
        for key, value in sorted(calc.computed.items()):
            sheet = calc.sheets[key[0]]
            row, col = key[1], key[2]
            cell = sheet.cell_el(row, col)
            if cell is None:
                cell = sheet.cell(row, col)
            had_stored = sheet.has_stored_result(row, col)
            old = sheet.stored(row, col) if had_stored else None
            if had_stored and _same_value(old, value):
                continue
            book.store_value(cell, value)
            sheet.touch()
            recalculated_seen.add(key)
            if key not in self.before:
                fmt = sheet.number_format(row, col)
                report.moved.append(Change(key[0], addr(row, col),
                                           show_value(old, fmt, book.date1904) if had_stored else "(no result yet)",
                                           show_value(value, fmt, book.date1904)))
            if isinstance(value, XlError) and not isinstance(old, XlError):
                report.new_errors.append((key[0], addr(row, col), value.code,
                                          F.display_formula(sheet.formula(row, col) or "")))
        for key, reason in sorted(calc.failed.items()):
            sheet = calc.sheets[key[0]]
            row, col = key[1], key[2]
            if key not in keys:
                continue
            cell = sheet.cell_el(row, col)
            if cell is not None and not self.full_recalculate:
                # An out-of-date number is worse than an empty cell Excel fills in on open.
                if sheet.has_stored_result(row, col):
                    book.store_value(cell, None)
                    sheet.touch()
                report.not_calculated.append((key[0], addr(row, col), reason))
            elif cell is not None:
                report.not_calculated.append((key[0], addr(row, col), reason))
        for key, (old_value, old_formula, old_format) in sorted(self.before.items()):
            sheet = calc.sheets.get(key[0])
            if sheet is None:
                continue
            row, col = key[1], key[2]
            new_formula = sheet.formula(row, col)
            new_value = calc.computed.get(key, sheet.stored(row, col))
            before_text = _show_cell(old_value, old_formula, old_format, book.date1904)
            after_text = _show_cell(new_value, new_formula, sheet.number_format(row, col), book.date1904,
                                    failed=key in calc.failed)
            if before_text != after_text:
                report.written.append(Change(key[0], addr(row, col), before_text, after_text))

    def _note_short_ranges(self, calc: Calc) -> None:
        """A total that stops just above numbers added this call is wrong and looks right."""
        if not self.new_numbers:
            return
        warned: set[str] = set()
        by_sheet: dict[str, set[tuple[int, int]]] = {}
        for sheet_name, row, col in self.new_numbers:
            by_sheet.setdefault(sheet_name, set()).add((row, col))
        for sheet in calc.sheets.values():
            for (row, col) in sheet.formula_els:
                text = sheet.formula(row, col) or ""
                for box in calc.boxes_of(text, sheet.name, row, col):
                    if box.is_cell or box.sheet not in by_sheet:
                        continue
                    added = by_sheet[box.sheet]
                    column_run = box.c1 == box.c2 and any(
                        (box.r2 + 1, box.c1) == cell for cell in added)
                    row_run = box.r1 == box.r2 and any(
                        (box.r1, box.c2 + 1) == cell for cell in added)
                    if column_run or row_run:
                        where = f"{sheet.name}!{addr(row, col)}"
                        if where not in warned and len(warned) < 5:
                            warned.add(where)
                            self.report.notes.append(
                                f"{where} = {F.display_formula(text)} stops just before numbers added now. "
                                "If it should include them, change its range.")

    def _refresh_chart_caches(self, calc: Calc) -> None:
        """Chart pictures read their stored copy of the numbers until Excel opens the file."""
        book = self.book
        changed = bool(calc.computed) or bool(self.seeds) or self.structure_changed
        if not changed:
            return
        for part in book.parts_named(r"xl/charts/chart\d*\.xml"):
            if part not in self.new_charts and part not in book._dirty and not self.seeds:
                continue
            root = book.part(part)
            touched = False
            for reference in [*root.iter(f"{{{C_NS}}}numRef"), *root.iter(f"{{{C_NS}}}strRef")]:
                formula = reference.find(f"{{{C_NS}}}f")
                if formula is None or not formula.text:
                    continue
                if self._write_chart_cache(calc, reference, formula.text, part in self.new_charts):
                    touched = True
            if touched:
                book.touch(part)

    def _write_chart_cache(self, calc: Calc, reference: Any, formula: str, is_new: bool) -> bool:
        sheet_name, body, _ = F.split_sheet_prefix(formula)
        box = F.parse_box(body)
        owner = calc.sheet_name(sheet_name) if sheet_name else None
        if box is None or owner is None or box.area > 20_000:
            return False
        if not is_new and not any(k[0] == owner and box.contains(k[1], k[2])
                                  for k in [*self.seeds, *calc.computed]):
            return False
        numeric = reference.tag == f"{{{C_NS}}}numRef"
        tag = "numCache" if numeric else "strCache"
        number_format = "General"
        for old in reference.findall(f"{{{C_NS}}}{tag}"):
            code = old.find(f"{{{C_NS}}}formatCode")
            if code is not None and code.text:
                number_format = code.text
            reference.remove(old)
        cache = etree.Element(f"{{{C_NS}}}{tag}")
        reference.find(f"{{{C_NS}}}f").addnext(cache)
        if numeric:
            etree.SubElement(cache, f"{{{C_NS}}}formatCode").text = number_format
        etree.SubElement(cache, f"{{{C_NS}}}ptCount", val=str(box.area))
        index = 0
        for r in range(box.r1, box.r2 + 1):
            for c in range(box.c1, box.c2 + 1):
                key = (owner, r, c)
                value = calc.computed.get(key, calc.sheets[owner].stored(r, c))
                if value is not None and not isinstance(value, XlError) and (
                        not numeric or (isinstance(value, (int, float)) and not isinstance(value, bool))):
                    pt = etree.SubElement(cache, f"{{{C_NS}}}pt", idx=str(index))
                    etree.SubElement(pt, f"{{{C_NS}}}v").text = (
                        number_text(float(value)) if numeric else F.to_text(value))
                index += 1
        return True


def _show_cell(value: Any, formula: str | None, fmt: str, date1904: bool, *, failed: bool = False) -> str:
    if formula is not None:
        result = "not calculated" if failed else show_value(value, fmt, date1904)
        return f"{F.display_formula(formula)} → {result}"
    return show_value(value, fmt, date1904)


_CHART_KINDS = ("column", "bar", "line", "area", "pie", "scatter")


def _chart_xml(kind: str, series: list[dict[str, str]], categories: str | None, title: str, stacked: bool,
               x_title: str, y_title: str, legend: Any) -> Any:
    c = f"{{{C_NS}}}"
    a = f"{{{A_NS}}}"
    root = etree.Element(c + "chartSpace", nsmap={"c": C_NS, "a": A_NS, "r": R_NS})
    etree.SubElement(root, c + "roundedCorners", val="0")
    chart = etree.SubElement(root, c + "chart")

    def add_title(parent: Any, text: str) -> None:
        holder = etree.SubElement(parent, c + "title")
        tx = etree.SubElement(holder, c + "tx")
        rich = etree.SubElement(tx, c + "rich")
        etree.SubElement(rich, a + "bodyPr")
        para = etree.SubElement(rich, a + "p")
        run = etree.SubElement(para, a + "r")
        etree.SubElement(run, a + "t").text = text
        etree.SubElement(holder, c + "overlay", val="0")

    if title:
        add_title(chart, title)
    etree.SubElement(chart, c + "autoTitleDeleted", val="0" if title else "1")
    plot = etree.SubElement(chart, c + "plotArea")
    etree.SubElement(plot, c + "layout")
    group_tag = {"column": "barChart", "bar": "barChart", "line": "lineChart", "area": "areaChart",
                 "pie": "pieChart", "scatter": "scatterChart"}[kind]
    group = etree.SubElement(plot, c + group_tag)
    if kind in ("column", "bar"):
        etree.SubElement(group, c + "barDir", val="col" if kind == "column" else "bar")
    if kind in ("column", "bar", "line", "area"):
        etree.SubElement(group, c + "grouping", val="stacked" if stacked else ("clustered" if kind in ("column", "bar") else "standard"))
    if kind == "scatter":
        etree.SubElement(group, c + "scatterStyle", val="lineMarker")
    etree.SubElement(group, c + "varyColors", val="1" if kind == "pie" else "0")
    for index, item in enumerate(series):
        ser = etree.SubElement(group, c + "ser")
        etree.SubElement(ser, c + "idx", val=str(index))
        etree.SubElement(ser, c + "order", val=str(index))
        if item.get("name") or item.get("name_ref"):
            tx = etree.SubElement(ser, c + "tx")
            if item.get("name_ref"):
                ref = etree.SubElement(tx, c + "strRef")
                etree.SubElement(ref, c + "f").text = item["name_ref"]
            else:
                etree.SubElement(tx, c + "v").text = item["name"]
        if kind == "line":
            marker = etree.SubElement(ser, c + "marker")
            etree.SubElement(marker, c + "symbol", val="none")
        if kind == "scatter":
            x_ref = item.get("x") or categories
            if x_ref:
                x_val = etree.SubElement(ser, c + "xVal")
                ref = etree.SubElement(x_val, c + "numRef")
                etree.SubElement(ref, c + "f").text = x_ref
            y_val = etree.SubElement(ser, c + "yVal")
            ref = etree.SubElement(y_val, c + "numRef")
            etree.SubElement(ref, c + "f").text = item["values"]
            etree.SubElement(ser, c + "smooth", val="0")
            continue
        if categories:
            cat = etree.SubElement(ser, c + "cat")
            ref = etree.SubElement(cat, c + "strRef")
            etree.SubElement(ref, c + "f").text = categories
        val = etree.SubElement(ser, c + "val")
        ref = etree.SubElement(val, c + "numRef")
        etree.SubElement(ref, c + "f").text = item["values"]
        if kind == "line":
            etree.SubElement(ser, c + "smooth", val="0")
    if kind in ("column", "bar"):
        etree.SubElement(group, c + "gapWidth", val="80")
        if stacked:
            etree.SubElement(group, c + "overlap", val="100")
    if kind == "line":
        etree.SubElement(group, c + "marker", val="1")
    if kind != "pie":
        etree.SubElement(group, c + "axId", val="111111111")
        etree.SubElement(group, c + "axId", val="222222222")

        def scaling(axis: Any) -> None:
            holder = etree.SubElement(axis, c + "scaling")
            etree.SubElement(holder, c + "orientation", val="minMax")

        first_tag = "valAx" if kind == "scatter" else "catAx"
        first = etree.SubElement(plot, c + first_tag)
        etree.SubElement(first, c + "axId", val="111111111")
        scaling(first)
        etree.SubElement(first, c + "delete", val="0")
        etree.SubElement(first, c + "axPos", val="l" if kind == "bar" else "b")
        if x_title:
            add_title(first, x_title)
        etree.SubElement(first, c + "numFmt", formatCode="General", sourceLinked="1")
        etree.SubElement(first, c + "majorTickMark", val="out")
        etree.SubElement(first, c + "minorTickMark", val="none")
        etree.SubElement(first, c + "tickLblPos", val="nextTo")
        etree.SubElement(first, c + "crossAx", val="222222222")
        etree.SubElement(first, c + "crosses", val="autoZero")
        if kind == "scatter":
            etree.SubElement(first, c + "crossBetween", val="midCat")
        else:
            etree.SubElement(first, c + "auto", val="1")
            etree.SubElement(first, c + "lblAlgn", val="ctr")
            etree.SubElement(first, c + "lblOffset", val="100")
            etree.SubElement(first, c + "noMultiLvlLbl", val="0")
        second = etree.SubElement(plot, c + "valAx")
        etree.SubElement(second, c + "axId", val="222222222")
        scaling(second)
        etree.SubElement(second, c + "delete", val="0")
        etree.SubElement(second, c + "axPos", val="b" if kind == "bar" else "l")
        # Excel's default gridlines are black when a chart names no style.
        grid = etree.SubElement(second, c + "majorGridlines")
        shape = etree.SubElement(grid, c + "spPr")
        line = etree.SubElement(shape, a + "ln", w="6350")
        fill = etree.SubElement(line, a + "solidFill")
        etree.SubElement(fill, a + "srgbClr", val="D9D9D9")
        if y_title:
            add_title(second, y_title)
        etree.SubElement(second, c + "numFmt", formatCode="General", sourceLinked="1")
        etree.SubElement(second, c + "majorTickMark", val="out")
        etree.SubElement(second, c + "minorTickMark", val="none")
        etree.SubElement(second, c + "tickLblPos", val="nextTo")
        etree.SubElement(second, c + "crossAx", val="111111111")
        etree.SubElement(second, c + "crosses", val="autoZero")
        etree.SubElement(second, c + "crossBetween", val="midCat" if kind == "scatter" else "between")
    if legend:
        holder = etree.SubElement(chart, c + "legend")
        etree.SubElement(holder, c + "legendPos", val="b" if kind != "pie" else "r")
        etree.SubElement(holder, c + "overlay", val="0")
    etree.SubElement(chart, c + "plotVisOnly", val="1")
    etree.SubElement(chart, c + "dispBlanksAs", val="gap")
    return root


# ── Making a workbook ────────────────────────────────────────────────


def _set_properties(book: Book, title: str, author: str) -> None:
    path = "docProps/core.xml"
    if not book.has(path):
        return
    root = book.part(path)
    creator = root.find(f"{{{DC_NS}}}creator")
    if creator is not None:
        creator.text = author
    if title:
        element = root.find(f"{{{DC_NS}}}title")
        if element is None:
            element = etree.SubElement(root, f"{{{DC_NS}}}title")
        element.text = title
    book.touch(path)


def build_workbook(sheets: list[dict[str, Any]] | None, *, edits: list[dict[str, Any]] | None = None,
                   template: bytes | None = None, title: str = "", author: str = "Captain") -> tuple[Book, Report]:
    """A new workbook from ``sheets`` (name, rows, header) and any further ``edits``."""
    if template is not None:
        book = Book(template)
        book.as_ordinary_workbook()
    else:
        book = Book.blank()
    editor = Editor(book, author=author)
    specs = list(sheets or [])
    if template is None and not specs:
        raise XlsxError("create needs `sheets`: a list of {name, rows}.")
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise XlsxError(f"Sheet {index + 1} is not an object with `name` and `rows`.")
        name = str(spec.get("name") or f"Sheet{index + 1}")
        if template is None and index == 0:
            first = book.sheets[0].load()
            clean = editor._check_sheet_name(name, ignoring=first.name)
            editor.rename_sheet(first, clean)
            sheet = first
        elif name.casefold() in book._by_name:
            sheet = book.sheet(name)
        else:
            sheet = editor.add_sheet(name)
        editor.fill_new_sheet(sheet, spec)
    if template is None:
        _set_properties(book, title, author)
    report = editor.apply(list(edits or []))
    if not any(s.path in book._dirty for s in book.sheets):
        for sheet in book.sheets:
            book.touch(sheet.load().path)
    return book, report


def _openpyxl_problem(data: bytes) -> Exception | None:
    from openpyxl import load_workbook

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_workbook(io.BytesIO(data), read_only=len(data) > 4 * 1024 * 1024).close()
    except Exception as exc:
        return exc
    return None


def verify_saved(data: bytes, original: bytes | None = None) -> None:
    """The bytes must open again before they replace a file: in our reader, and in
    openpyxl's unless it couldn't read the *original* either (Excel reads more than
    openpyxl does, and a file it never could read isn't this edit's fault)."""
    Book(data).all_sheets()
    if len(data) > 30 * 1024 * 1024:
        return
    problem = _openpyxl_problem(data)
    if problem is not None and (original is None or _openpyxl_problem(original) is None):
        raise XlsxError(f"The result wouldn't open again, so nothing was saved "
                        f"({type(problem).__name__}: {problem}).") from None


# ── Reading a workbook out ───────────────────────────────────────────


def _notes_of(book: Book, sheet: Sheet) -> dict[tuple[int, int], str]:
    notes: dict[tuple[int, int], str] = {}
    for _, part in sheet.rels.of_type("comments"):
        if not book.has(part):
            continue
        root = book.part(part)
        parent = root.find(_q("commentList"))
        author_list = root.find(_q("authors"))
        authors = [a.text or "" for a in author_list] if author_list is not None else []
        for comment in parent if parent is not None else []:
            box = F.parse_box(comment.get("ref") or "")
            if box is None:
                continue
            text = _unescape_text("".join(t.text or "" for t in comment.iter(T_TAG))).strip()
            who = int(comment.get("authorId", "-1"))
            author = authors[who] if 0 <= who < len(authors) else ""
            if author and not text.startswith(author) and not text.startswith("[Threaded"):
                text = f"{author}: {text}"
            notes[(box.r1, box.c1)] = text.replace("\n", " ")
    return notes


def calc_for_read(book: Book) -> Calc:
    """Results for formulas saved without one (files from scripts), for showing only."""
    calc = Calc(book)
    missing = [(s.name, r, c) for s in calc.sheets.values() for (r, c) in s.formula_els
               if not s.has_stored_result(r, c) and not s.is_opaque_formula(r, c)
               and s.array_owner.get((r, c), (r, c)) == (r, c)]
    if len(missing) <= MAX_FORMULAS_TO_CALCULATE:
        calc.run(missing)
    return calc


def overview(book: Book) -> list[str]:
    """What's in the workbook, a line per sheet."""
    lines: list[str] = []
    tables = book.tables()
    for el in book.sheet_els:
        name = el.get("name") or ""
        found = book._by_name.get(name.casefold())
        state = el.get("state")
        tag = f" ({state})" if state and state != "visible" else ""
        if found is None:
            lines.append(f'- "{name}"{tag}: a chart sheet')
            continue
        sheet = found.load()
        rows, cols = sheet.used()
        size = f"A1:{addr(rows, cols)}" if rows and cols else "empty"
        parts = [size]
        if sheet.formula_els:
            parts.append(f"{len(sheet.formula_els):,} formulas")
        own_tables = [t for t in tables if t.sheet == sheet.name]
        if own_tables:
            parts.append("tables " + ", ".join(f"{t.name} ({box_text(t.box)})" for t in own_tables))
        if sheet.merges:
            parts.append(f"{len(sheet.merges)} merged areas")
        drawings = [p for _, p in sheet.rels.of_type("drawing") if book.has(p)]
        charts = sum(len(_Rels(book, p).of_type("chart")) for p in drawings)
        if charts:
            parts.append(f"{charts} chart(s)")
        if sheet.rels.of_type("pivotTable"):
            parts.append("pivot table")
        if sheet.is_protected():
            parts.append("protected")
        hidden = sheet.hidden_rows()
        if hidden:
            parts.append(f"{len(hidden)} hidden rows")
        hidden_cols = sheet.hidden_columns()
        if hidden_cols:
            parts.append("hidden columns " + ", ".join(
                get_column_letter(a) if a == b else f"{get_column_letter(a)}:{get_column_letter(b)}"
                for a, b in hidden_cols))
        lines.append(f'- "{sheet.name}"{tag}: ' + ", ".join(parts))
    names = [i for i in book.defined_names() if not (i.get("name") or "").startswith("_xlnm.")]
    if names:
        shown = "; ".join(f"{i.get('name')} = {F.display_formula(i.text or '')[1:]}" for i in names[:30])
        lines.append(f"Names: {shown}" + (f" … and {len(names) - 30} more" if len(names) > 30 else ""))
    if book.parts_named(r"xl/externalLinks/externalLink\d*\.xml"):
        lines.append("Links to other workbooks: yes (those formulas keep the last values Excel stored).")
    if book.is_macro:
        lines.append("Macros: yes (kept as they are, never run).")
    return lines


def _cell_text(book: Book, sheet: Sheet, calc: Calc | None, row: int, col: int, view: str,
               notes: dict[tuple[int, int], str]) -> str | None:
    formula = sheet.formula(row, col)
    stored = sheet.stored(row, col)
    note = notes.get((row, col))
    fmt = sheet.number_format(row, col)
    ref = addr(row, col)
    if formula is None:
        owner = sheet.array_owner.get((row, col))
        if owner is not None and owner != (row, col):
            text = f"{ref} {show_value(stored, fmt, book.date1904)} {{array in {addr(*owner)}}}"
        elif stored is None:
            if note is None:
                return None
            text = f"{ref} (empty)"
        else:
            text = f"{ref} {show_value(stored, fmt, book.date1904)}"
    else:
        shown_formula = F.display_formula(formula) if not sheet.is_opaque_formula(row, col) else "{data table}"
        if (row, col) in sheet.array_masters:
            shown_formula = "{" + shown_formula + "}"
        key = (sheet.name, row, col)
        if sheet.has_stored_result(row, col):
            result = show_value(stored, fmt, book.date1904)
        elif calc is not None and key in calc.computed:
            result = show_value(calc.computed[key], fmt, book.date1904) + " (calculated now, not stored)"
        else:
            reason = calc.failed.get(key) if calc is not None else None
            result = f"? (no stored result{': ' + reason if reason else ''})"
        if view == "values":
            text = f"{ref} {result}"
        elif view == "formulas":
            text = f"{ref} {shown_formula}"
        else:
            text = f"{ref} {shown_formula} → {result}"
    if note:
        text += f" [note: {note}]"
    return text


def render_range(book: Book, sheet: Sheet, box: Box, *, view: str = "both", limit: int = 200,
                 calc: Calc | None = None) -> tuple[str, int | None]:
    """Rows of a sheet as text, one line per row; (text, next row to read or None)."""
    notes = _notes_of(book, sheet)
    lines: list[str] = []
    shown_rows = 0
    next_row: int | None = None
    rows = [r for r in sheet.row_numbers if box.r1 <= r <= box.r2]
    note_rows = {r for r, _ in notes if box.r1 <= r <= box.r2}
    if note_rows:
        rows = sorted(set(rows) | note_rows)
    for row in rows:
        cells = sheet.grid.get(row) or {}
        columns = sorted({c for c in cells if box.c1 <= c <= box.c2} |
                         {c for r, c in notes if r == row and box.c1 <= c <= box.c2})
        parts = [p for p in (_cell_text(book, sheet, calc, row, col, view, notes) for col in columns) if p]
        if not parts:
            continue
        if shown_rows >= limit:
            next_row = row
            break
        lines.append(" | ".join(parts))
        shown_rows += 1
    return "\n".join(lines), next_row


def find_text(book: Book, needle: str, *, only_sheet: str | None = None, limit: int = 100) -> list[str]:
    """Cells whose text, number or formula contains *needle*."""
    wanted = needle.casefold()
    out: list[str] = []
    for sheet in book.all_sheets():
        if only_sheet and sheet.name.casefold() != only_sheet.casefold():
            continue
        for row in sheet.row_numbers:
            for col, cell in sorted(sheet.grid[row].items()):
                value = book.cell_value(cell)
                formula = sheet.formula(row, col)
                fmt = sheet.number_format(row, col)
                haystack = [show_value(value, fmt, book.date1904)] if value is not None else []
                if formula:
                    haystack.append(F.display_formula(formula))
                if any(wanted in item.casefold() for item in haystack):
                    out.append(f"{F.quote_sheet(sheet.name)}!{addr(row, col)} " + " → ".join(reversed(haystack)))
                    if len(out) >= limit:
                        return out
    return out


# ── Checking a workbook ──────────────────────────────────────────────


@dataclass(slots=True)
class Issue:
    level: str  # error, warn, info
    kind: str
    where: str
    what: str


_ORDINARY_CONSTANTS = {0.0, 1.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 12.0, 24.0, 30.0, 31.0, 52.0, 60.0,
                       90.0, 100.0, 360.0, 365.0, 1000.0, 3600.0}
_RELATIVE = re.compile(r"(\$?)([A-Z]{1,3})(\$?)(\d{1,7})")


def relative_form(stored: str, row: int, col: int) -> str:
    """A formula with its references relative to its cell: copies read the same."""
    try:
        items = F._tokens(stored)
    except Exception:
        return stored
    out: list[str] = []
    for token in items:
        if token.type == "OPERAND" and token.subtype == "RANGE":
            sheet, body, has_prefix = F.split_sheet_prefix(token.value)

            def one(match: re.Match[str]) -> str:
                col_abs, letters, row_abs, digits = match.groups()
                c = _col_index(letters)
                r = int(digits)
                return (f"R{r}" if row_abs else f"R[{r - row}]") + (f"C{c}" if col_abs else f"C[{c - col}]")

            shifted = _RELATIVE.sub(one, body)
            out.append((f"{sheet}!" if has_prefix else "") + shifted)
        else:
            out.append(token.value)
    return "".join(out)


def check_book(book: Book) -> tuple[list[Issue], dict[str, int]]:
    """What a careful reviewer looks for first. Deterministic: no model involved."""
    issues: list[Issue] = []
    sheets = book.all_sheets()
    total_formulas = sum(len(s.formula_els) for s in sheets)
    stats = {"sheets": len(sheets), "formulas": total_formulas, "stale": 0}
    calc = Calc(book, use_stored_on_failure=True)
    keys = {(s.name, r, c) for s in sheets for (r, c) in s.formula_els
            if s.array_owner.get((r, c), (r, c)) == (r, c) and not s.is_opaque_formula(r, c)}
    if total_formulas <= MAX_FORMULAS_TO_CALCULATE:
        calc.run(keys)
    else:
        issues.append(Issue("info", "size", "workbook",
                            f"{total_formulas:,} formulas is too many to recalculate here; "
                            "stored results were checked as they are."))

    for key in sorted(calc.circular):
        sheet = calc.sheets[key[0]]
        issues.append(Issue("error", "circular", f"{key[0]}!{addr(key[1], key[2])}",
                            f"circular reference: {F.display_formula(sheet.formula(key[1], key[2]) or '')}"))

    # Results: what's stored, and whether a fresh calculation agrees.
    volatile = _volatile_dependents(calc)
    errors: dict[tuple[str, int, int], XlError] = {}
    for sheet in sheets:
        for (row, col) in sheet.formula_els:
            key = (sheet.name, row, col)
            text = sheet.formula(row, col) or ""
            stored = sheet.stored(row, col)
            has = sheet.has_stored_result(row, col)
            fresh = calc.computed.get(key)
            value = fresh if key in calc.computed else stored
            if isinstance(value, XlError):
                errors[key] = value
            if "#REF!" in text:
                issues.append(Issue("error", "broken_reference", f"{sheet.name}!{addr(row, col)}",
                                    f"its reference is gone: {F.display_formula(text)}"))
            if not has and key in calc.failed and key not in calc.circular:
                issues.append(Issue("info", "not_calculated", f"{sheet.name}!{addr(row, col)}",
                                    f"no stored result ({calc.failed[key]}); Excel calculates it on open."))
            elif not has and key in calc.computed:
                stats["missing"] = stats.get("missing", 0) + 1
            elif has and key in calc.computed and key not in volatile and not _same_value(stored, fresh):
                stats["stale"] += 1
                fmt = sheet.number_format(row, col)
                issues.append(Issue("warn", "stale_result", f"{sheet.name}!{addr(row, col)}",
                                    f"stored {show_value(stored, fmt, book.date1904)} but it calculates to "
                                    f"{show_value(fresh, fmt, book.date1904)}: {F.display_formula(text)}"))

    # An error that comes in from another cell isn't worth a line of its own.
    for key, error in sorted(errors.items(), key=lambda item: item[0]):
        sheet = calc.sheets[key[0]]
        text = sheet.formula(key[1], key[2]) or ""
        inherited = any(
            (box.sheet, r, c) in errors and (box.sheet, r, c) != key
            for box in calc.boxes_of(text, key[0], key[1], key[2]) if box.area <= 4096
            for r in range(box.r1, box.r2 + 1) for c in range(box.c1, box.c2 + 1))
        if not inherited:
            issues.append(Issue("error", "error_value", f"{key[0]}!{addr(key[1], key[2])}",
                                f"{error.code} from {F.display_formula(text)}"))
    inherited_count = len(errors) - sum(1 for i in issues if i.kind == "error_value")
    if inherited_count > 0:
        issues.append(Issue("info", "error_value", "workbook",
                            f"{inherited_count} more cell(s) show an error passed on from those."))

    for sheet in sheets:
        _check_consistency(book, sheet, issues)
        _check_ranges(book, sheet, calc, issues)
        _check_text_numbers(sheet, issues)
        _check_constants(sheet, issues)
        _check_blank_reads(sheet, calc, issues)
        _check_text_formats(sheet, issues)
        hidden = sheet.hidden_rows()
        hidden_with_data = [r for r in hidden if any(
            sheet.stored(r, c) is not None for c in sheet.grid.get(r, {}))]
        if hidden_with_data:
            issues.append(Issue("info", "hidden", sheet.name,
                                f"{len(hidden_with_data)} hidden row(s) hold data (first: row {hidden_with_data[0]})."))
        for first, last in sheet.hidden_columns():
            if any(first <= c <= last for cells in sheet.grid.values() for c in cells):
                letters = get_column_letter(first) if first == last else f"{get_column_letter(first)}:{get_column_letter(last)}"
                issues.append(Issue("info", "hidden", sheet.name, f"hidden column(s) {letters} hold data."))
        if sheet.state != "visible":
            issues.append(Issue("info", "hidden", sheet.name, f"the sheet itself is {sheet.state}."))
    if book.parts_named(r"xl/externalLinks/externalLink\d*\.xml"):
        issues.append(Issue("info", "links", "workbook",
                            "it links to other workbooks: those numbers are only as fresh as the last update."))
    order = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda i: order.get(i.level, 3))
    return issues, stats


def _volatile_dependents(calc: Calc) -> set[tuple[str, int, int]]:
    seeds = {(s.name, r, c) for s in calc.sheets.values() for (r, c) in s.formula_els
             if F.compile_formula(s.formula(r, c) or "").functions & F.VOLATILE}
    return (calc.affected_by(seeds) | seeds) if seeds else set()


def _check_consistency(book: Book, sheet: Sheet, issues: list[Issue]) -> None:
    """A cell that breaks the pattern of its neighbours: a typed-in number in a
    column of formulas, or a formula unlike the ones either side of it."""
    forms: dict[tuple[int, int], str] = {}
    for (row, col) in sheet.formula_els:
        text = sheet.formula(row, col)
        if text:
            forms[(row, col)] = relative_form(text, row, col)
    seen: set[tuple[int, int]] = set()
    for (row, col), form in sorted(forms.items()):
        for d_row, d_col in ((1, 0), (0, 1)):
            after_cell = (row + d_row, col + d_col)
            after_formula = forms.get((row + 2 * d_row, col + 2 * d_col))
            if after_formula == form and after_cell not in seen:
                if after_cell in forms and forms[after_cell] != form:
                    seen.add(after_cell)
                    issues.append(Issue(
                        "warn", "inconsistent_formula", f"{sheet.name}!{addr(*after_cell)}",
                        f"{F.display_formula(sheet.formula(*after_cell) or '')} doesn't follow the cells around it, "
                        f"which read like {F.display_formula(sheet.formula(row, col) or '')}."))
                elif after_cell not in forms and isinstance(sheet.stored(*after_cell), (int, float)) \
                        and not isinstance(sheet.stored(*after_cell), bool):
                    seen.add(after_cell)
                    issues.append(Issue(
                        "warn", "hardcoded_number", f"{sheet.name}!{addr(*after_cell)}",
                        f"a typed-in {show_number(sheet.stored(*after_cell))} among formulas like "
                        f"{F.display_formula(sheet.formula(row, col) or '')}."))


def _check_ranges(book: Book, sheet: Sheet, calc: Calc, issues: list[Issue]) -> None:
    """A SUM that stops one cell short of more numbers."""
    reported: set[tuple[int, int]] = set()

    def is_constant_number(owner: Sheet, row: int, col: int) -> bool:
        value = owner.stored(row, col)
        return isinstance(value, (int, float)) and not isinstance(value, bool) \
            and (row, col) not in owner.formula_els

    for (row, col) in sheet.formula_els:
        text = sheet.formula(row, col) or ""
        compiled = F.compile_formula(text)
        if not compiled.functions & {"SUM", "AVERAGE", "COUNT", "MIN", "MAX", "PRODUCT", "MEDIAN"}:
            continue
        for operand in compiled.operands:
            if not isinstance(operand, Box) or operand.is_cell:
                continue
            owner = calc.sheets.get(calc.sheet_name(operand.sheet) or "") if operand.sheet else sheet
            if owner is None:
                continue
            short: tuple[int, int] | None = None
            if operand.c1 == operand.c2 and operand.r2 < F.MAX_ROW:
                if is_constant_number(owner, operand.r2 + 1, operand.c1) and not (owner is sheet and (operand.r2 + 1, operand.c1) == (row, col)):
                    short = (operand.r2 + 1, operand.c1)
            elif operand.r1 == operand.r2 and operand.c2 < F.MAX_COL:
                if is_constant_number(owner, operand.r1, operand.c2 + 1):
                    short = (operand.r1, operand.c2 + 1)
            if short is not None and (row, col) not in reported:
                reported.add((row, col))
                issues.append(Issue("warn", "range_stops_short", f"{sheet.name}!{addr(row, col)}",
                                    f"{F.display_formula(text)} stops just before {addr(*short)}, which also holds "
                                    f"the number {show_number(owner.stored(*short))}."))


def _check_text_numbers(sheet: Sheet, issues: list[Issue]) -> None:
    """Numbers stored as text are left out of every total without a sign."""
    per_column: dict[int, list[int]] = {}
    numeric: dict[int, int] = {}
    for row, cells in sheet.grid.items():
        for col in cells:
            value = sheet.stored(row, col)
            if isinstance(value, str):
                if (row, col) in sheet.formula_els:
                    continue
                stripped = value.strip()
                if stripped and F.parse_number_text(stripped) is not None and not re.fullmatch(r"0\d+", stripped):
                    per_column.setdefault(col, []).append(row)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[col] = numeric.get(col, 0) + 1
    for col, rows in sorted(per_column.items()):
        if numeric.get(col, 0) >= 1 or len(rows) >= 3:
            shown = ", ".join(addr(r, col) for r in sorted(rows)[:5])
            issues.append(Issue("warn", "text_number", f"{sheet.name}!{addr(min(rows), col)}",
                                f"{len(rows)} number(s) stored as text ({shown}): totals skip them."))


def _check_constants(sheet: Sheet, issues: list[Issue]) -> None:
    """A rate typed into a formula can't be seen or changed in one place."""
    found: list[tuple[int, int, float]] = []
    for (row, col) in sheet.formula_els:
        text = sheet.formula(row, col) or ""
        if any(ch.isdigit() for ch in text):
            for token in F._tokens(text):
                if token.type == "OPERAND" and token.subtype == "NUMBER":
                    number = float(token.value)
                    if number not in _ORDINARY_CONSTANTS and (number != int(number) or number > 1000):
                        found.append((row, col, number))
                        break
    if found:
        found.sort()
        row, col, number = found[0]
        issues.append(Issue("info", "number_in_formula", f"{sheet.name}!{addr(row, col)}",
                            f"{len(found)} formula(s) have a number typed in (first: {show_number(number)} in "
                            f"{F.display_formula(sheet.formula(row, col) or '')}). "
                            "A cell of its own is easier to check and change."))


def _check_text_formats(sheet: Sheet, issues: list[Issue]) -> None:
    """TEXT's format is read in the viewer's own language: in Excel set to decimal
    commas, =TEXT(3.14159,"0.00") shows 003. A cell's number format has no such problem."""
    found: list[tuple[int, int]] = []
    for (row, col) in sheet.formula_els:
        text = sheet.formula(row, col) or ""
        if "TEXT(" not in text.upper():
            continue
        for token in F._tokens(text):
            if token.type == "OPERAND" and token.subtype == "TEXT" and re.search(r"[0#][.,]|[.,][0#]", token.value):
                found.append((row, col))
                break
    if found:
        row, col = min(found)
        issues.append(Issue("warn", "text_format", f"{sheet.name}!{addr(row, col)}",
                            f"{len(found)} formula(s) use a number pattern inside TEXT (first: "
                            f"{F.display_formula(sheet.formula(row, col) or '')}). Excel set to decimal commas "
                            "reads it differently: 0.00 shows 3.14 as 003. Prefer a cell number format."))


def _check_blank_reads(sheet: Sheet, calc: Calc, issues: list[Issue]) -> None:
    """Arithmetic on an empty cell counts it as zero, which may not be what was meant."""
    count = 0
    first: tuple[int, int, str] | None = None
    for (row, col) in sheet.formula_els:
        text = sheet.formula(row, col) or ""
        compiled = F.compile_formula(text)
        if compiled.functions & {"ISBLANK", "IF", "IFERROR", "COUNTA", "COUNTBLANK", "IFS"}:
            continue
        for operand in compiled.operands:
            if not isinstance(operand, Box) or not operand.is_cell:
                continue
            owner = calc.sheets.get(calc.sheet_name(operand.sheet) or "") if operand.sheet else sheet
            if owner is None:
                continue
            if owner.cell_el(operand.r1, operand.c1) is None or (
                    owner.stored(operand.r1, operand.c1) is None
                    and (operand.r1, operand.c1) not in owner.formula_els):
                count += 1
                if first is None or (row, col) < first[:2]:
                    first = (row, col, addr(operand.r1, operand.c1))
                break
    if first is not None:
        issues.append(Issue("info", "reads_blank", f"{sheet.name}!{addr(first[0], first[1])}",
                            f"{count} formula(s) read an empty cell as zero (first: "
                            f"{F.display_formula(sheet.formula(first[0], first[1]) or '')} reads {first[2]})."))
