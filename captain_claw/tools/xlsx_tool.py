"""xlsx tool: Excel workbooks made and changed properly, without scripts.

The work happens in :mod:`captain_claw.xlsx_ops`; this module decides where files
are read from and written to, keeps a backup before changing a file in place, and
reports what changed in a form the model can check at a glance.

Files land the way the docx tool's do (tools/docx_tool.py has the reasons): the
agent's own files change in place with a backup; a user's upload, a read-only
Drive folder or anything else stays untouched and the changed copy goes to
``saved/output/<session>/`` under the same name. A shared agent's member works on
their own files only, with no backup and no host paths in results.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from captain_claw import xlsx_formula as F
from captain_claw import xlsx_ops as X
from captain_claw.logging import get_logger
from captain_claw.tools.docx_tool import DocxTool, _author, _member, _scrub, _show
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool
from captain_claw.vfs import is_vfs_path, project_is_readonly, resolve_vfs_path, split_scheme

log = get_logger(__name__)

ACTIONS = ("create", "read", "edit", "check")
_WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}
_TEMPLATE_SUFFIXES = {".xltx", ".xltm"}
# A workbook name the model gave another extension: report.xls → report.xlsx.
_OTHER_SUFFIXES = {".xls", ".xlsb", ".csv", ".tsv", ".ods", ".txt", ".md", ".pdf", ".docx", ".html"}
_RESULT_LINES = 40
_READ_CHARS = 100_000


def xlsx_file_name(path: str) -> str:
    """*path* with the extension the saved file gets: budget.xls → budget.xlsx,
    "Budget" → "Budget.xlsx" (speaker.py checks a member's path under this name)."""
    suffix = Path(path).suffix.lower()
    if suffix in _WORKBOOK_SUFFIXES:
        return path
    if suffix in _OTHER_SUFFIXES or suffix in _TEMPLATE_SUFFIXES:
        return path[: -len(suffix)] + ".xlsx"
    return path + ".xlsx"


def _writable(target: Path) -> Path:
    """*target* when this call may write it (always for the owner)."""
    from captain_claw import speaker

    if _member() is not None and not speaker.member_may_write(target):
        raise X.XlsxError(speaker.PATH_REFUSED_PREFIX + "you can only save in your own folders, "
                          "and change only files you created — pass output_path "
                          "(a new file name, or vfs:<project>/<name>.xlsx).")
    return target


def _list_arg(raw: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise X.XlsxError(f"`{name}` is not valid JSON: {exc}") from None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise X.XlsxError(f"`{name}` must be a list of objects.")
    return raw


class XlsxTool(Tool):
    """Create, read, edit and check Excel workbooks."""

    name = "xlsx"
    timeout_seconds = 180.0
    description = (
        "Excel workbooks (.xlsx, .xlsm): create, read, edit and check them properly — never "
        "write openpyxl/pandas scripts or use write/edit on them (a load-and-save loses charts, "
        "pivots and more, and stores no formula results). "
        "create: `sheets` = [{name, rows, header}]; a value starting with = is a formula, and "
        "numbers, TRUE/FALSE and dates like 2026-03-31 become real numbers, logicals and dates. "
        "Then any `edits` (formats, charts, names, tables). Optional template. "
        "read: with no sheet, what's in the workbook and the first rows; with sheet/range the "
        "cells as `A1 \"text\" | B1 12 | C1 =SUM(A1:B1) → 12`; find searches text, numbers and "
        "formulas. Read before you edit. "
        "edit: `edits` = changes applied in order, all or nothing; later edits see earlier row "
        "moves. set (cell + value, cell + rows, or cells = {A1: value}), fill (range + formula "
        "as it reads in the first cell, copied with references adjusted; or source cell), clear, "
        "append_rows (under the sheet's last row, or to a table), insert_rows / delete_rows "
        "(at, count), insert_columns / delete_columns (at: letter, count) — references, names, "
        "tables and charts follow, as in Excel — format, width, height, freeze, merge, unmerge, "
        "filter, add_sheet, rename_sheet, delete_sheet, move_sheet, define_name, delete_name, "
        "add_table, add_chart, comment, recalculate. Keep calculations as live formulas, never "
        "numbers you worked out elsewhere. Every formula affected is recalculated and stored; "
        "the result lists each changed cell and each total that moved, any new errors, and what "
        "couldn't be calculated (Excel does on open). A bad edit fails the whole call. A user's "
        "upload is never changed: the changed copy goes under saved/output with the same name. "
        "check: errors, broken references, stale results, formulas that break their column's "
        "pattern, typed-in numbers among formulas, SUMs that stop short, numbers stored as text, "
        "hidden data. Run it before you hand over a model."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS), "description": "What to do."},
            "path": {
                "type": "string",
                "description": ("The workbook. For create, where to save it (a name like 'Budget.xlsx' "
                                "lands in saved/; vfs:<project>/… for the shared folders)."),
            },
            "sheets": {
                "type": "array",
                "description": "create: the sheets, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "rows": {"type": "array", "description": "Rows, each a list of values; an empty value leaves a cell empty.",
                                 "items": {"type": "array", "items": {"type": "string"}}},
                        "header": {"type": "boolean",
                                   "description": "First row is a header: bold, shaded, frozen."},
                    },
                },
            },
            "edits": {
                "type": "array",
                "description": "edit (and create): changes applied in order, all or nothing.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": list(X.OPS)},
                        "sheet": {"type": "string",
                                  "description": "The sheet (or write Sheet!A1 in cell/range). Needed when there are several."},
                        "cell": {"type": "string", "description": "One cell, e.g. B2 (set/comment; where rows start)."},
                        "range": {"type": "string", "description": "A range, e.g. B2:D9 or C:C."},
                        "value": {"type": "string", "description": "set: the value; =… is a formula."},
                        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                 "description": "set/append_rows/add_sheet: rows of values."},
                        "cells": {"type": "object", "description": "set: {\"B2\": 5, \"C2\": \"=B2*2\"}."},
                        "as_text": {"type": "boolean", "description": "set: keep it as text, e.g. a code like 1200 or =x."},
                        "like": {"type": "string", "description": "set: take the format of this cell."},
                        "formula": {"type": "string", "description": "fill: as it reads in the first cell of the range."},
                        "source": {"type": "string", "description": "fill: copy this cell's formula and format."},
                        "formats": {"type": "boolean", "description": "clear: formats too."},
                        "table": {"type": "string", "description": "append_rows: the table to grow."},
                        "at": {"type": "string",
                               "description": "insert/delete/height: first row number or column letter ('5', 'C', or '5:7')."},
                        "count": {"type": "integer", "description": "insert/delete: how many (default 1)."},
                        "number_format": {"type": "string", "description": "Excel code: #,##0.00  0.0%  yyyy-mm-dd …"},
                        "bold": {"type": "boolean"},
                        "italic": {"type": "boolean"},
                        "underline": {"type": "boolean"},
                        "strike": {"type": "boolean"},
                        "color": {"type": "string", "description": "Text colour, hex, e.g. C00000."},
                        "fill": {"type": "string", "description": "Cell colour, hex, or none."},
                        "font": {"type": "string"},
                        "font_size": {"type": "number"},
                        "border": {"type": "string", "description": "All sides: thin, medium, thick, double, none."},
                        "border_top": {"type": "string"},
                        "border_bottom": {"type": "string"},
                        "border_left": {"type": "string"},
                        "border_right": {"type": "string"},
                        "align": {"type": "string", "enum": ["left", "center", "right"]},
                        "valign": {"type": "string", "enum": ["top", "center", "bottom"]},
                        "wrap": {"type": "boolean"},
                        "indent": {"type": "integer"},
                        "columns": {"type": "string", "description": "width: 'A' or 'B:D'."},
                        "width": {"type": "string", "description": "width: characters, or 'auto'. add_chart: centimetres."},
                        "height": {"type": "number", "description": "height: points. add_chart: centimetres."},
                        "name": {"type": "string",
                                 "description": "add_sheet/rename_sheet: the new name. define_name/delete_name/add_table: the name."},
                        "position": {"type": "integer", "description": "add_sheet/move_sheet: 1 = first."},
                        "refers_to": {"type": "string", "description": "define_name: e.g. Inputs!$B$2."},
                        "style": {"type": "string", "description": "add_table: e.g. TableStyleMedium2."},
                        "type": {"type": "string", "enum": ["column", "bar", "line", "area", "pie", "scatter"],
                                 "description": "add_chart."},
                        "title": {"type": "string", "description": "add_chart."},
                        "categories": {"type": "string", "description": "add_chart: labels, e.g. A2:A9."},
                        "values": {"type": "string", "description": "add_chart: numbers; several columns make several series."},
                        "series": {"type": "array", "description": "add_chart: instead of values, [{name, values}].",
                                   "items": {"type": "object"}},
                        "anchor": {"type": "string", "description": "add_chart: the cell at its top left."},
                        "stacked": {"type": "boolean", "description": "add_chart."},
                        "x_title": {"type": "string", "description": "add_chart."},
                        "y_title": {"type": "string", "description": "add_chart."},
                        "legend": {"type": "boolean", "description": "add_chart."},
                        "text": {"type": "string", "description": "comment: the note; empty removes it."},
                    },
                    "required": ["op"],
                },
            },
            "output_path": {"type": "string", "description": "edit: save the result here instead (a new file)."},
            "sheet": {"type": "string", "description": "read: the sheet. edit: the default for edits that name none."},
            "range": {"type": "string", "description": "read: the cells, e.g. A1:F40."},
            "view": {"type": "string", "enum": ["both", "values", "formulas"],
                     "description": "read: both (default), or only results / only formulas."},
            "find": {"type": "string", "description": "read: list the cells holding this text, number or formula."},
            "limit": {"type": "integer", "description": "read: how many rows (default 100)."},
            "author": {"type": "string", "description": "Name on notes (default: your own name)."},
            "template": {"type": "string", "description": "create: an .xlsx/.xltx to start from: its sheets and formats stay."},
            "title": {"type": "string", "description": "create: the file's title property."},
            "overwrite": {"type": "boolean", "description": "create: replace an existing file."},
        },
        "required": ["action", "path"],
    }

    async def execute(self, action: str, path: str, **kwargs: Any) -> ToolResult:
        action = str(action or "").strip().lower()
        if action not in ACTIONS:
            return ToolResult(success=False, error=f"Unknown action {action!r}; use one of: {', '.join(ACTIONS)}.")
        try:
            if action == "create":
                return await self._create(path, kwargs)
            source = await self._source(path, kwargs, template=action in ("read", "check"))
            if action == "read":
                return await asyncio.to_thread(self._read, source, kwargs)
            if action == "check":
                return await asyncio.to_thread(self._check, source)
            return await self._edit(path, source, kwargs)
        except (X.XlsxError, F.FormulaError) as exc:
            return ToolResult(success=False, error=_scrub(str(exc)))
        except Exception as exc:
            log.error("xlsx tool failed", action=action, path=path, error=str(exc))
            return ToolResult(success=False, error=_scrub(f"{type(exc).__name__}: {exc}"))

    # ── Paths ──────────────────────────────────────────────────────

    async def _source(self, path: str, kwargs: dict[str, Any], *, template: bool = False) -> Path:
        """An existing file: vfs- and Drive-aware like the extract tools, then
        the edit tool's fallbacks, then where `create` would have put it."""
        from captain_claw.tools.document_extract import _resolve_readable_file
        from captain_claw.tools.edit import EditTool

        found, error = await _resolve_readable_file(path, kwargs.get("_runtime_base_path"))
        if found is None:
            found = EditTool._resolve_path(path, kwargs)
        if found is None and not is_vfs_path(path):
            candidate = WriteTool.resolve_local_target(path, kwargs)
            found = candidate if candidate.is_file() else None
        if found is None:
            raise X.XlsxError(error or f"File not found: {path}")
        allowed = _WORKBOOK_SUFFIXES | (_TEMPLATE_SUFFIXES if template else set())
        suffix = found.suffix.lower()
        if suffix not in allowed:
            if suffix in _TEMPLATE_SUFFIXES:
                raise X.XlsxError(f"{found.name} is a template. Make a workbook from it: action=create with template.")
            hint = ""
            if suffix in (".xls", ".xlsb"):
                hint = " Ask for a copy saved as .xlsx."
            elif suffix in (".csv", ".tsv"):
                hint = " Read it with `read`, or bring it in with datastore import_file."
            raise X.XlsxError(f"Not an .xlsx workbook: {found.name}.{hint}")
        return found

    @staticmethod
    def _target(path: str, kwargs: dict[str, Any], *, suffix: str = "") -> Path:
        """Where a new file named *path* is written (create, output_path)."""
        name = xlsx_file_name(path)
        if suffix and Path(name).suffix.lower() != suffix:
            # Excel refuses a macro workbook with the wrong extension, and the reverse.
            name = name[: -len(Path(name).suffix)] + suffix
        if is_vfs_path(path):
            project = split_scheme(path)[0]
            if project_is_readonly(project):
                raise X.XlsxError(f"'{project}' is a read-only folder — save somewhere else.")
            target = resolve_vfs_path(name, create_parents=True)
            if target is None:
                raise X.XlsxError(f"Invalid vfs path (escapes your folder): {path}")
        else:
            target = WriteTool.resolve_local_target(name, kwargs)
        _writable(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # ── Saving ─────────────────────────────────────────────────────

    @staticmethod
    def _save(data: bytes, target: Path, original: bytes | None = None) -> int:
        """Write atomically; the bytes must open again before they replace anything."""
        from captain_claw import saved_attribution
        from captain_claw.vfs import record_author

        X.verify_saved(data, original)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if target.exists():
                try:
                    os.chmod(tmp, target.stat().st_mode & 0o777)
                except OSError:
                    pass
            prior = saved_attribution.prior_creator(target)
            os.replace(tmp, target)
            saved_attribution.note_write(target, prior)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        record_author(target)
        return len(data)

    # ── Actions ────────────────────────────────────────────────────

    async def _create(self, path: str, kwargs: dict[str, Any]) -> ToolResult:
        if Path(path).suffix.lower() == ".xlsm":
            # A new workbook has no macros, and Excel refuses .xlsm without them.
            path = path[:-5] + ".xlsx"
        sheets = _list_arg(kwargs.get("sheets"), "sheets") if kwargs.get("sheets") else []
        edits = _list_arg(kwargs.get("edits"), "edits") if kwargs.get("edits") else []
        if not sheets and not kwargs.get("template"):
            raise X.XlsxError("create needs `sheets`: a list of {name, rows} (or a `template`).")
        target = self._target(path, kwargs)
        if target.exists() and not kwargs.get("overwrite"):
            raise X.XlsxError(f"{target.name} already exists. To change it, use action=edit; "
                              "to replace it with a new workbook, pass overwrite=true.")
        template = None
        if kwargs.get("template"):
            template = await self._source(str(kwargs["template"]), kwargs, template=True)
        author = _author(kwargs)

        def build() -> tuple[int, X.Book, X.Report]:
            start = template.read_bytes() if template is not None else None
            book, report = X.build_workbook(sheets, edits=edits, template=start,
                                            title=str(kwargs.get("title") or ""), author=author)
            return self._save(book.to_bytes(), target, start), book, report

        size, book, report = await asyncio.to_thread(build)
        DocxTool._register(path, target, kwargs)
        lines = [f"Created {_show(target)} ({size:,} bytes)."]
        lines += X.overview(book)
        lines += self._calculated_lines(report)
        return ToolResult(
            success=True, content="\n".join(lines),
            system_hint=("The workbook is saved and complete. To change it later use xlsx edit, "
                         "never a script. Check the results above against what you expected."))

    @staticmethod
    def _calculated_lines(report: X.Report) -> list[str]:
        """After create: what the formulas came to, and anything wrong with them."""
        lines: list[str] = []
        formulas = [c for c in report.written if c.after.startswith("=")]
        if formulas:
            lines.append("Formula results:")
            lines += [f"  {c.sheet}!{c.cell}: {c.after}" for c in formulas[:_RESULT_LINES]]
            if len(formulas) > _RESULT_LINES:
                lines.append(f"  … and {len(formulas) - _RESULT_LINES} more.")
        lines += XlsxTool._problem_lines(report)
        return lines

    @staticmethod
    def _problem_lines(report: X.Report) -> list[str]:
        lines: list[str] = []
        if report.new_errors:
            lines.append("⚠ These now show an error — fix them before handing this over:")
            lines += [f"  {s}!{c}: {code} from {text}" for s, c, code, text in report.new_errors[:_RESULT_LINES]]
            if len(report.new_errors) > _RESULT_LINES:
                lines.append(f"  … and {len(report.new_errors) - _RESULT_LINES} more.")
        if report.not_calculated:
            lines.append("Not calculated here (no stored result; Excel calculates them on open):")
            lines += [f"  {s}!{c}: {why}" for s, c, why in report.not_calculated[:12]]
            if len(report.not_calculated) > 12:
                lines.append(f"  … and {len(report.not_calculated) - 12} more.")
        lines += [f"Note: {n}" for n in report.notes]
        return lines

    def _read(self, source: Path, kwargs: dict[str, Any]) -> ToolResult:
        book = X.Book(source.read_bytes())
        view = str(kwargs.get("view") or "both").lower()
        if view not in ("both", "values", "formulas"):
            view = "both"
        limit = max(1, min(int(kwargs.get("limit") or 100), 2000))
        needle = str(kwargs.get("find") or "").strip()
        wanted_sheet = str(kwargs.get("sheet") or "").strip()
        if needle:
            hits = X.find_text(book, needle, only_sheet=wanted_sheet or None)
            if not hits:
                return ToolResult(success=True, content=f"{source.name}: nothing holds {needle!r}.")
            tail = "\n… showing the first 100." if len(hits) == 100 else ""
            return ToolResult(success=True, content=f"{source.name}: {len(hits)} cell(s) hold {needle!r}:\n"
                                                    + "\n".join(hits) + tail)
        head = [f"{source.name} — {len(book.sheet_els)} sheet(s):", *X.overview(book)]
        range_text = str(kwargs.get("range") or "").strip()
        box = None
        if range_text:
            sheet_name, box = X.parse_range(range_text)
            wanted_sheet = sheet_name or wanted_sheet
        if not wanted_sheet:
            if not book.sheets:
                return ToolResult(success=True, content="\n".join(head))
            if range_text and len(book.sheets) > 1:
                raise X.XlsxError(f"Which sheet? Sheets: {', '.join(s.name for s in book.sheets)}.")
            wanted_sheet = book.sheets[0].name
        sheet = book.sheet(wanted_sheet)
        rows, cols = sheet.used()
        if box is None:
            box = F.Box(None, 1, 1, F.MAX_ROW, F.MAX_COL)
        calc = X.calc_for_read(book)
        body, next_row = X.render_range(book, sheet, box, view=view, limit=limit, calc=calc)
        if len(body) > _READ_CHARS:
            cut = body.rfind("\n", 0, _READ_CHARS)
            body = body[:cut if cut > 0 else _READ_CHARS]
            shown_lines = body.count("\n") + 1
            limit = shown_lines
            body, next_row = X.render_range(book, sheet, box, view=view, limit=limit, calc=calc)
        lines = [*head, "", f'Sheet "{sheet.name}" {X.box_text(F.Box(None, box.r1, box.c1, min(box.r2, max(rows, 1)), min(box.c2, max(cols, 1))))}:']
        lines.append(body if body else "(no cells here)")
        if sheet.merges:
            lines.append("Merged: " + ", ".join(X.box_text(m) for m in sheet.merges[:30]))
        if next_row is not None:
            last = min(box.r2, rows)
            lines.append(f"… more: read again with sheet={sheet.name!r} range="
                         f"{X.addr(next_row, box.c1)}:{X.addr(last, min(box.c2, cols))}")
        return ToolResult(success=True, content="\n".join(lines))

    def _check(self, source: Path) -> ToolResult:
        book = X.Book(source.read_bytes())
        issues, stats = X.check_book(book)
        counts = {level: sum(1 for i in issues if i.level == level) for level in ("error", "warn", "info")}
        head = (f"{source.name}: checked {stats['sheets']} sheet(s) and {stats['formulas']:,} formula(s) — "
                f"{counts['error']} error(s), {counts['warn']} warning(s), {counts['info']} note(s).")
        lines = [head]
        if not issues:
            lines.append("No errors, broken references, stale results, broken patterns, short ranges or "
                         "numbers stored as text.")
        labels = {"error": "ERROR", "warn": "WARN", "info": "NOTE"}
        for issue in issues[:60]:
            lines.append(f"{labels.get(issue.level, issue.level)} {issue.where}: {issue.what}")
        if len(issues) > 60:
            lines.append(f"… and {len(issues) - 60} more.")
        if stats.get("missing"):
            lines.append(f"{stats['missing']} formula(s) had no stored result; they calculate fine here "
                         "and an edit (or recalculate) stores them.")
        return ToolResult(success=True, content="\n".join(lines))

    async def _edit(self, path: str, source: Path, kwargs: dict[str, Any]) -> ToolResult:
        from captain_claw.tools.edit import EditTool

        edits = _list_arg(kwargs.get("edits"), "edits") if kwargs.get("edits") else []
        if not edits:
            raise X.XlsxError("`edits` must be a non-empty list of edit objects, each with an `op`.")
        output_path = str(kwargs.get("output_path") or "").strip()
        if output_path:
            target = self._target(output_path, kwargs, suffix=source.suffix.lower())
        elif DocxTool._in_place(source, kwargs) and (_member() is None or DocxTool._member_may(source)):
            target = source
        else:
            # saved/output/<session>/<same name> (or the run's / repo's folder).
            target = WriteTool.resolve_local_target(f"output/{source.name}", kwargs)
            if target.exists() and target.resolve() != source.resolve():
                raise X.XlsxError(
                    f"A changed copy of {source.name} already exists at {_show(target)}. To add to that "
                    "copy, edit that path; to start a separate copy from the original, pass output_path.")
            _writable(target)
            target.parent.mkdir(parents=True, exist_ok=True)
        author = _author(kwargs)
        default_sheet = str(kwargs.get("sheet") or "").strip() or None

        def run() -> tuple[bytes, bytes, X.Report]:
            original = source.read_bytes()
            book = X.Book(original)
            report = X.Editor(book, author=author, default_sheet=default_sheet).apply(edits)
            return original, book.to_bytes(), report

        original, data, report = await asyncio.to_thread(run)
        backup = ""
        if target == source and _member() is None:
            # Never for a member: backups share one folder per file name (edit.py).
            backup = await asyncio.to_thread(EditTool()._create_backup, source, kwargs) or ""
        await asyncio.to_thread(self._save, data, target, original)
        if target != source:
            DocxTool._register(path, target, kwargs)

        counts = ", ".join(f"{v} {k}" for k, v in report.counts.items() if v) or "no changes"
        lines = [f"Edited {target.name}: {counts}."]
        lines.append(f"Saved in place: {_show(target)}" if target == source
                     else f"Saved: {_show(target)} (the original {source.name} is unchanged)")
        if backup:
            lines.append(f"Backup: {backup}")
        if report.written:
            lines.append("Changed cells:")
            lines += [f"  {c.sheet}!{c.cell}: {c.before} ⇒ {c.after}" for c in report.written[:_RESULT_LINES]]
            if len(report.written) > _RESULT_LINES:
                lines.append(f"  … and {len(report.written) - _RESULT_LINES} more.")
        if report.moved:
            lines.append("Other formulas that moved as a result:")
            lines += [f"  {c.sheet}!{c.cell}: {c.before} ⇒ {c.after}" for c in report.moved[:_RESULT_LINES]]
            if len(report.moved) > _RESULT_LINES:
                lines.append(f"  … and {len(report.moved) - _RESULT_LINES} more.")
        lines += self._problem_lines(report)
        return ToolResult(
            success=True, content="\n".join(lines),
            system_hint=("The changes are saved. Don't re-read the whole workbook to check — "
                         "the lines above are the result. Give the user the saved file."))
