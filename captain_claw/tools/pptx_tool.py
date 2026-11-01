"""pptx tool: PowerPoint decks made and changed properly, without scripts.

The work happens in :mod:`captain_claw.pptx_ops`; this module decides where files
are read from and written to, keeps a backup before changing a file in place, and
reports what changed and what doesn't fit.

Files land the way the docx tool's do (tools/docx_tool.py has the reasons): the
agent's own files change in place with a backup; a user's upload, a read-only
Drive folder or anything else stays untouched and the changed copy goes to
``saved/output/<session>/`` under the same name. A shared agent's member works on
their own files only, with no backup and no host paths in results. A picture
named in an edit must be one the caller may read.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from captain_claw import pptx_ops as P
from captain_claw.logging import get_logger
from captain_claw.tools.docx_tool import DocxTool, _author, _member, _scrub, _show
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool
from captain_claw.vfs import is_vfs_path, project_is_readonly, resolve_vfs_path, split_scheme

log = get_logger(__name__)

ACTIONS = ("create", "read", "edit", "check", "render")
_DECK_SUFFIXES = {".pptx", ".pptm"}
_OPEN_ONLY_SUFFIXES = {".potx", ".potm", ".ppsx", ".ppsm"}
# A deck name the model gave another extension: pitch.ppt → pitch.pptx.
_OTHER_SUFFIXES = {".ppt", ".pps", ".pot", ".odp", ".key", ".pdf", ".txt", ".md", ".html", ".docx"}
_READ_SLIDES = 25
_RESULT_LINES = 60
_READ_CHARS = 100_000


def pptx_file_name(path: str) -> str:
    """*path* with the extension the saved file gets: pitch.ppt → pitch.pptx,
    "Pitch" → "Pitch.pptx" (speaker.py checks a member's path under this name)."""
    suffix = Path(path).suffix.lower()
    if suffix in _DECK_SUFFIXES:
        return path
    if suffix in _OTHER_SUFFIXES or suffix in _OPEN_ONLY_SUFFIXES:
        return path[: -len(suffix)] + ".pptx"
    return path + ".pptx"


def _writable(target: Path) -> Path:
    """*target* when this call may write it (always for the owner)."""
    from captain_claw import speaker

    if _member() is not None and not speaker.member_may_write(target):
        raise P.PptxError(speaker.PATH_REFUSED_PREFIX + "you can only save in your own folders, "
                          "and change only files you created — pass output_path "
                          "(a new file name, or vfs:<project>/<name>.pptx).")
    return target


def _list_arg(raw: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise P.PptxError(f"`{name}` is not valid JSON: {exc}") from None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise P.PptxError(f"`{name}` must be a list of objects.")
    return raw


def _image_finder(kwargs: dict[str, Any]) -> Callable[[str], Path | None]:
    """Finds a picture named in an edit. A member gets only what they may read
    (the docx tool's rule); so does an agent confined to its own folders."""
    from captain_claw import deck_confine

    find = DocxTool()._image_root(kwargs)

    def resolve(src: str) -> Path | None:
        found = find(src)
        if found is not None and deck_confine.active() and not deck_confine.allowed(found.resolve(), "read"):
            return None
        return found

    return resolve


def _slide_numbers(raw: Any, count: int) -> list[int]:
    """`slide` on read and render: a number, "2-5" or a list; every slide if absent."""
    if raw in (None, "", "all"):
        return list(range(1, count + 1))
    numbers: list[int] = []
    parts = raw if isinstance(raw, list) else re.split(r"[,\s]+", str(raw).strip())
    for part in parts:
        match = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", str(part).strip())
        if match:
            first, last = sorted((int(match.group(1)), int(match.group(2))))
            numbers += list(range(first, last + 1))
        elif str(part).strip().isdigit():
            numbers.append(int(part))
        else:
            raise P.PptxError(f"`slide` must be a number, a range like 2-5, or all; got {part!r}.")
    outside = [n for n in numbers if not 1 <= n <= count]
    if outside:
        raise P.PptxError(f"There is no slide {outside[0]}: the deck has {count}.")
    return numbers


class PptxTool(Tool):
    """Create, read, edit and check PowerPoint decks."""

    name = "pptx"
    timeout_seconds = 180.0
    description = (
        "PowerPoint decks (.pptx, .pptm): create, read, edit and check them properly — never "
        "write python-pptx scripts or use write/edit on them. Slides are built from the deck's "
        "layouts and placeholders, and every result says what doesn't fit. "
        "create: `slides` = [{layout, title, subtitle, body, placeholders, table, chart, image, "
        "notes}]; body lines starting - are bullets, indent two spaces per level, **bold**, "
        "*italic*, [link](url). With no layout one is chosen for what the slide holds; a table, "
        "chart or image takes the empty placeholder that suits it. Pass `template` (a .pptx or "
        ".potx) to use a firm's design. Optional font, heading_font, heading_color, accent_color, "
        "page_numbers; then any `edits`. "
        "read: with no slide, the size, fonts and layouts, then the slides; each shape shows its "
        "id, position and text. Read before you edit. "
        "edit: `edits` = changes applied in order, all or nothing; slide numbers are as they stand "
        "after earlier edits. set_text, replace (find → replace_with across runs, on `slide`: a "
        "number, 2-5, or all), format, add_slide, duplicate_slide, delete_slide, move_slide, "
        "hide_slide, set_notes, add_text, add_image, add_table, set_table, add_chart, set_chart "
        "(new numbers, same look), place (x, y, width, height in inches; z front/back), "
        "delete_shape, set_alt, set_theme. Aim at a shape with `shape`: its id, or title, subtitle, "
        "body, ph:<number>. Native tables and charts stay editable in PowerPoint; never paste a "
        "picture of one. A bad edit fails the whole call. A user's upload is never changed: the "
        "changed copy goes under saved/output with the same name. "
        "check: text that overflows, shapes off the slide or overlapping, empty placeholders, "
        "missing titles, small text, stretched pictures, leftover text, missing alt text. Run it "
        "before you hand over a deck. "
        "render: each slide as a PNG to look at, where LibreOffice is installed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS), "description": "What to do."},
            "path": {
                "type": "string",
                "description": ("The deck. For create, where to save it (a name like 'Pitch.pptx' lands "
                                "in saved/; vfs:<project>/… for the shared folders)."),
            },
            "slides": {
                "type": "array",
                "description": "create: the slides, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "layout": {"type": "string", "description": "Layout name or number, as read lists them."},
                        "title": {"type": "string"},
                        "subtitle": {"type": "string"},
                        "body": {"type": "string", "description": "Text for the content placeholder; - for bullets."},
                        "placeholders": {"type": "object", "description": "Other text by placeholder number: {\"2\": \"…\"}."},
                        "table": {"type": "array", "description": "Rows, each a list of values; the first is the header.",
                                  "items": {"type": "array", "items": {"type": "string"}}},
                        "chart": {"type": "object", "description": "{type, title, categories, series: [{name, values}]}."},
                        "image": {"type": "string", "description": "A saved picture's path."},
                        "notes": {"type": "string", "description": "Speaker notes."},
                        "keep_empty": {"type": "boolean",
                                       "description": "Keep unfilled placeholders (normally removed)."},
                    },
                },
            },
            "edits": {
                "type": "array",
                "description": "edit (and create): changes applied in order, all or nothing.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": list(P.OPS)},
                        "slide": {"type": "string",
                                  "description": "Slide number. replace, format and hide_slide also take 2-5 or all."},
                        "shape": {"type": "string",
                                  "description": "Its id from read, or title, subtitle, body, ph:<number>, or its name."},
                        "text": {"type": "string",
                                 "description": "set_text/add_text/set_notes/set_alt: the text; - for bullets, indent for levels."},
                        "bullets": {"type": "boolean", "description": "set_text: force bullets on or off."},
                        "find": {"type": "string", "description": "replace: exact text; format: only this text."},
                        "replace_with": {"type": "string", "description": "replace: the new text."},
                        "notes": {"type": "boolean", "description": "replace: speaker notes too."},
                        "layout": {"type": "string", "description": "add_slide: layout name or number."},
                        "title": {"type": "string", "description": "add_slide/add_chart/set_chart."},
                        "subtitle": {"type": "string", "description": "add_slide."},
                        "body": {"type": "string", "description": "add_slide."},
                        "placeholders": {"type": "object", "description": "add_slide: text by placeholder number."},
                        "at": {"type": "integer", "description": "add_slide/duplicate_slide/move_slide: where it ends up (1 = first)."},
                        "hidden": {"type": "boolean", "description": "hide_slide: false shows it again."},
                        "x": {"type": "number", "description": "Inches from the left."},
                        "y": {"type": "number", "description": "Inches from the top."},
                        "width": {"type": "number", "description": "Inches."},
                        "height": {"type": "number", "description": "Inches."},
                        "z": {"type": "string", "enum": ["front", "back"], "description": "place."},
                        "image": {"type": "string", "description": "add_image: a saved picture's path. add_slide: the same."},
                        "fit": {"type": "string", "enum": ["contain", "cover"], "description": "add_image."},
                        "alt": {"type": "string", "description": "add_image/add_table/add_chart: alt text."},
                        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                                 "description": "add_table/set_table: rows of values."},
                        "cells": {"type": "object", "description": "set_table: {\"r2c3\": \"new\"}."},
                        "header": {"type": "boolean", "description": "add_table: first row is a header (default)."},
                        "type": {"type": "string", "enum": ["column", "bar", "line", "area", "pie", "doughnut", "scatter"],
                                 "description": "add_chart."},
                        "categories": {"type": "array", "items": {"type": "string"}, "description": "add_chart/set_chart."},
                        "series": {"type": "array", "items": {"type": "object"},
                                   "description": "add_chart/set_chart: [{name, values}]; scatter: [{name, x, y}]."},
                        "stacked": {"type": "boolean", "description": "add_chart."},
                        "legend": {"type": "boolean", "description": "add_chart."},
                        "labels": {"type": "boolean", "description": "add_chart: values on the chart."},
                        "number_format": {"type": "string", "description": "add_chart/set_chart: e.g. #,##0."},
                        "paragraph": {"type": "integer", "description": "format: only this paragraph."},
                        "row": {"type": "integer", "description": "format on a table: only this row."},
                        "column": {"type": "integer", "description": "format on a table: only this column."},
                        "bold": {"type": "boolean"},
                        "italic": {"type": "boolean"},
                        "underline": {"type": "boolean"},
                        "color": {"type": "string", "description": "Text colour, hex, e.g. C00000."},
                        "fill": {"type": "string", "description": "Shape colour, hex, or none."},
                        "font": {"type": "string", "description": "format, set_theme."},
                        "font_size": {"type": "number", "description": "Points."},
                        "align": {"type": "string", "enum": ["left", "center", "right", "justify"]},
                        "valign": {"type": "string", "enum": ["top", "center", "bottom"]},
                        "heading_font": {"type": "string", "description": "set_theme."},
                        "heading_color": {"type": "string", "description": "set_theme: hex."},
                        "accent_color": {"type": "string", "description": "set_theme: hex."},
                    },
                    "required": ["op"],
                },
            },
            "output_path": {"type": "string", "description": "edit: save the result here instead (a new file)."},
            "slide": {"type": "string", "description": "read/render: a number, 2-5, or all (default)."},
            "find": {"type": "string", "description": "read: list where this text appears."},
            "template": {"type": "string",
                         "description": "create: a .pptx or .potx whose layouts and design to use."},
            "keep_slides": {"type": "boolean", "description": "create: keep the template's own slides too."},
            "title": {"type": "string", "description": "create: the file's title property."},
            "page_numbers": {"type": "boolean", "description": "create: a slide number on each slide."},
            "font": {"type": "string", "description": "create: the theme's text font."},
            "heading_font": {"type": "string", "description": "create: the theme's heading font."},
            "heading_color": {"type": "string", "description": "create: title colour, hex."},
            "accent_color": {"type": "string", "description": "create: main accent colour, hex (charts, tables)."},
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
            source = await self._source(path, kwargs, open_only=action in ("read", "check", "render"))
            if action == "read":
                return await asyncio.to_thread(self._read, source, kwargs)
            if action == "check":
                return await asyncio.to_thread(self._check, source)
            if action == "render":
                return await asyncio.to_thread(self._render, source, kwargs)
            return await self._edit(path, source, kwargs)
        except P.PptxError as exc:
            return ToolResult(success=False, error=_scrub(str(exc)))
        except Exception as exc:
            log.error("pptx tool failed", action=action, path=path, error=str(exc))
            return ToolResult(success=False, error=_scrub(f"{type(exc).__name__}: {exc}"))

    # ── Paths ──────────────────────────────────────────────────────

    async def _source(self, path: str, kwargs: dict[str, Any], *, open_only: bool = False) -> Path:
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
            raise P.PptxError(error or f"File not found: {path}")
        allowed = _DECK_SUFFIXES | (_OPEN_ONLY_SUFFIXES if open_only else set())
        suffix = found.suffix.lower()
        if suffix not in allowed:
            if suffix in _OPEN_ONLY_SUFFIXES:
                raise P.PptxError(f"{found.name} is a template or slide show. Make a deck from it: "
                                  "action=create with template.")
            hint = " Ask for a copy saved as .pptx." if suffix in (".ppt", ".pps", ".pot", ".odp", ".key") else ""
            raise P.PptxError(f"Not a .pptx deck: {found.name}.{hint}")
        return found

    @staticmethod
    def _target(path: str, kwargs: dict[str, Any], *, suffix: str = "") -> Path:
        """Where a new file named *path* is written (create, output_path)."""
        name = pptx_file_name(path)
        if suffix and Path(name).suffix.lower() != suffix:
            # PowerPoint refuses a macro deck with the wrong extension, and the reverse.
            name = name[: -len(Path(name).suffix)] + suffix
        if is_vfs_path(path):
            project = split_scheme(path)[0]
            if project_is_readonly(project):
                raise P.PptxError(f"'{project}' is a read-only folder — save somewhere else.")
            target = resolve_vfs_path(name, create_parents=True)
            if target is None:
                raise P.PptxError(f"Invalid vfs path (escapes your folder): {path}")
        else:
            target = WriteTool.resolve_local_target(name, kwargs)
        _writable(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # ── Saving ─────────────────────────────────────────────────────

    @staticmethod
    def _save(data: bytes, target: Path) -> int:
        """Write atomically; the bytes must open again before they replace anything."""
        from captain_claw import saved_attribution
        from captain_claw.vfs import record_author

        P.verify_saved(data)
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

    @staticmethod
    def _fit_lines(report: P.Report) -> list[str]:
        lines: list[str] = []
        warnings = [i for i in report.fit if i.level in ("error", "warn")]
        if warnings:
            lines.append("⚠ Fix these before handing it over:")
            lines += [f"  {i.where}: {i.what}" for i in warnings[:30]]
            if len(warnings) > 30:
                lines.append(f"  … and {len(warnings) - 30} more.")
        lines += [f"Note: {n}" for n in report.notes]
        return lines

    async def _create(self, path: str, kwargs: dict[str, Any]) -> ToolResult:
        if Path(path).suffix.lower() == ".pptm":
            # A new deck has no macros, and PowerPoint refuses .pptm without them.
            path = path[:-5] + ".pptx"
        slides = _list_arg(kwargs.get("slides"), "slides") if kwargs.get("slides") else []
        edits = _list_arg(kwargs.get("edits"), "edits") if kwargs.get("edits") else []
        if not slides and not kwargs.get("template"):
            raise P.PptxError("create needs `slides`: a list of {layout, title, body, …} (or a `template`).")
        target = self._target(path, kwargs)
        if target.exists() and not kwargs.get("overwrite"):
            raise P.PptxError(f"{target.name} already exists. To change it, use action=edit; "
                              "to replace it with a new deck, pass overwrite=true.")
        template = None
        if kwargs.get("template"):
            template = await self._source(str(kwargs["template"]), kwargs, open_only=True)
        look = {key: kwargs[key] for key in ("font", "heading_font", "heading_color", "accent_color") if kwargs.get(key)}
        finder = _image_finder(kwargs)
        author = _author(kwargs)

        def build() -> tuple[int, Any, P.Report]:
            prs, report = P.build_deck(
                slides, edits=edits, template=template.read_bytes() if template is not None else None,
                keep_slides=bool(kwargs.get("keep_slides")), title=str(kwargs.get("title") or ""), author=author,
                page_numbers=bool(kwargs.get("page_numbers")), look=look, image_root=finder)
            return self._save(P.deck_bytes(prs), target), prs, report

        size, prs, report = await asyncio.to_thread(build)
        DocxTool._register(path, target, kwargs)
        lines = [f"Created {_show(target)} ({size:,} bytes, {len(prs.slides)} slide(s))."]
        lines += self._fit_lines(report)
        hint = ("The deck is saved. Fix anything listed above with pptx edit, never a script."
                if report.fit else "The deck is saved and nothing overflows. To change it later use pptx edit, "
                                   "never a script.")
        return ToolResult(success=True, content="\n".join(lines), system_hint=hint)

    def _read(self, source: Path, kwargs: dict[str, Any]) -> ToolResult:
        prs = P.open_deck(source.read_bytes())
        needle = str(kwargs.get("find") or "").strip()
        if needle:
            hits = P.find_text(prs, needle)
            if not hits:
                return ToolResult(success=True, content=f"{source.name}: nothing holds {needle!r}.")
            tail = "\n… showing the first 100." if len(hits) >= 100 else ""
            return ToolResult(success=True, content=f"{source.name}: {len(hits)} place(s) hold {needle!r}:\n"
                                                    + "\n".join(hits) + tail)
        count = len(prs.slides)
        numbers = _slide_numbers(kwargs.get("slide"), count)
        lines = [f"{source.name} — " + P.overview(prs)[0], *P.overview(prs)[1:], ""]
        used = sum(len(line) for line in lines)
        shown = 0
        for number in numbers:
            block = P.describe_slide(prs, prs.slides[number - 1], number)
            size = sum(len(line) + 1 for line in block)
            if shown and (used + size > _READ_CHARS or (kwargs.get("slide") in (None, "", "all") and shown >= _READ_SLIDES)):
                lines.append(f"… more: read again with slide={number}-{numbers[-1]}.")
                break
            lines += block
            used += size
            shown += 1
        return ToolResult(success=True, content="\n".join(lines))

    def _check(self, source: Path) -> ToolResult:
        prs = P.open_deck(source.read_bytes())
        issues, stats = P.check_deck(prs)
        counts = {level: sum(1 for i in issues if i.level == level) for level in ("error", "warn", "info")}
        head = (f"{source.name}: checked {stats['slides']} slide(s) — {counts['warn'] + counts['error']} warning(s), "
                f"{counts['info']} note(s).")
        lines = [head]
        if not issues:
            lines.append("Nothing overflows, runs off the slide or overlaps; no empty placeholders, missing titles, "
                         "stretched pictures or leftover text.")
        labels = {"error": "ERROR", "warn": "WARN", "info": "NOTE"}
        for issue in issues[:_RESULT_LINES]:
            lines.append(f"{labels.get(issue.level, issue.level)} {issue.where}: {issue.what}")
        if len(issues) > _RESULT_LINES:
            lines.append(f"… and {len(issues) - _RESULT_LINES} more.")
        lines.append("Text sizes are estimated from font metrics, so check close calls in PowerPoint.")
        return ToolResult(success=True, content="\n".join(lines))

    def _render(self, source: Path, kwargs: dict[str, Any]) -> ToolResult:
        renderer = P.find_renderer()
        if renderer is None:
            return ToolResult(success=False, error=(
                "render needs LibreOffice, which isn't installed on this server. Use check: it estimates "
                "what fits without drawing the slides."))
        data = source.read_bytes()
        prs = P.open_deck(data)
        numbers = _slide_numbers(kwargs.get("slide"), len(prs.slides))
        pages = P.render_pages(data, renderer)
        saved: list[str] = []
        for number in numbers:
            if number > len(pages):
                continue
            target = WriteTool.resolve_local_target(f"media/{source.stem}-slide-{number}.png", kwargs)
            _writable(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pages[number - 1])
            DocxTool._register(str(target), target, kwargs)
            saved.append(f"Slide {number}: {_show(target)}")
        if not saved:
            raise P.PptxError("LibreOffice rendered no pages for those slides.")
        note = "LibreOffice draws fonts and effects a little differently from PowerPoint. Look with image_vision."
        return ToolResult(success=True, content="\n".join([*saved, note]))

    async def _edit(self, path: str, source: Path, kwargs: dict[str, Any]) -> ToolResult:
        from captain_claw.tools.edit import EditTool

        edits = _list_arg(kwargs.get("edits"), "edits") if kwargs.get("edits") else []
        if not edits:
            raise P.PptxError("`edits` must be a non-empty list of edit objects, each with an `op`.")
        output_path = str(kwargs.get("output_path") or "").strip()
        if output_path:
            target = self._target(output_path, kwargs, suffix=source.suffix.lower())
        elif DocxTool._in_place(source, kwargs) and (_member() is None or DocxTool._member_may(source)):
            target = source
        else:
            # saved/output/<session>/<same name> (or the run's / repo's folder).
            target = WriteTool.resolve_local_target(f"output/{source.name}", kwargs)
            if target.exists() and target.resolve() != source.resolve():
                raise P.PptxError(
                    f"A changed copy of {source.name} already exists at {_show(target)}. To add to that "
                    "copy, edit that path; to start a separate copy from the original, pass output_path.")
            _writable(target)
            target.parent.mkdir(parents=True, exist_ok=True)
        finder = _image_finder(kwargs)

        def run() -> tuple[bytes, P.Report]:
            prs = P.open_deck(source.read_bytes())
            report = P.Editor(prs, image_root=finder).apply(edits)
            return P.deck_bytes(prs), report

        data, report = await asyncio.to_thread(run)
        backup = ""
        if target == source and _member() is None:
            # Never for a member: backups share one folder per file name (edit.py).
            backup = await asyncio.to_thread(EditTool()._create_backup, source, kwargs) or ""
        await asyncio.to_thread(self._save, data, target)
        if target != source:
            DocxTool._register(path, target, kwargs)

        counts = ", ".join(f"{v} {k}" for k, v in report.counts.items() if v) or "no changes"
        lines = [f"Edited {target.name}: {counts}."]
        lines.append(f"Saved in place: {_show(target)}" if target == source
                     else f"Saved: {_show(target)} (the original {source.name} is unchanged)")
        if backup:
            lines.append(f"Backup: {backup}")
        lines += self._fit_lines(report)
        if report.changed:
            lines.append("Changed slides now read:")
            lines += report.changed[:_RESULT_LINES]
            if len(report.changed) > _RESULT_LINES:
                lines.append(f"… and {len(report.changed) - _RESULT_LINES} more lines.")
        return ToolResult(
            success=True, content="\n".join(lines),
            system_hint=("The changes are saved. Don't re-read the deck to check — the lines above "
                         "are the result. Give the user the saved file."))
