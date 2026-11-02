"""pptx tool: PowerPoint decks made and changed properly, without scripts.

The work happens in :mod:`captain_claw.pptx_ops`; this module decides where files
are read from and written to, keeps a backup before changing a file in place, and
reports what changed and what doesn't fit.

Files land the way the docx tool's do (tools/docx_tool.py has the reasons): the
agent's own files change in place with a backup; a user's upload, a read-only
Drive folder or anything else stays untouched and the changed copy goes to
``saved/output/<session>/`` under the same name. A shared agent's member works on
their own files only, with no backup and no host paths in results. A picture
named in an edit must be one the caller may read; a link goes through the
public-only client every other picture download uses, and an SVG is drawn in a
browser that loads nothing else.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from lxml import etree

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
_LINK = re.compile(r"https?://", re.I)
_SVG_EDGE = 2400  # pixels on its long side: sharp across a slide
_SVG_LENGTH = re.compile(r"\s*([\d.]+)\s*(?:px|pt|pc|mm|cm|in)?\s*$")


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


def _looks_like_svg(head: bytes) -> bool:
    head = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    return head.startswith((b"<svg", b"<?xml", b"<!--", b"<!DOCTYPE")) and b"<svg" in head


def _svg_page(data: bytes, name: str) -> tuple[str, int, int]:
    """A page showing an SVG large enough to stay sharp on a slide, and its size."""
    try:
        root = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True))
    except etree.XMLSyntaxError:
        raise P.PptxError(f"{name} isn't an SVG this server can read. Save it as a PNG.") from None

    def length(name: str) -> float:
        match = _SVG_LENGTH.match(root.get(name) or "")
        return float(match.group(1)) if match else 0.0

    view = [float(v) for v in re.findall(r"-?[\d.]+", root.get("viewBox") or "")]
    if len(view) == 4 and view[2] > 0 and view[3] > 0:
        width, height = view[2], view[3]
    else:
        width, height = length("width") or 300.0, length("height") or 150.0
        root.set("viewBox", f"0 0 {width} {height}")
    scale = _SVG_EDGE / max(width, height)
    shown_w, shown_h = max(1, round(width * scale)), max(1, round(height * scale))
    root.set("width", str(shown_w))
    root.set("height", str(shown_h))
    source = base64.b64encode(etree.tostring(root)).decode("ascii")
    return (f'<!doctype html><body style="margin:0;background:transparent">'
            f'<img style="display:block" src="data:image/svg+xml;base64,{source}">', shown_w, shown_h)


async def _draw_svg(data: bytes, name: str) -> bytes:
    """An SVG as a PNG, transparency kept. As an <img>, a browser runs no script in
    it and loads nothing it names."""
    from playwright.async_api import async_playwright

    from captain_claw.tools.visualize import _launch

    html, width, height = _svg_page(data, name)
    async with async_playwright() as pw:
        browser = await _launch(pw)
        try:
            context = await browser.new_context(viewport={"width": width, "height": height}, java_script_enabled=False,
                                                service_workers="block", accept_downloads=False)
            await context.route("**/*", lambda route: route.abort())
            page = await context.new_page()
            await page.set_content(html, wait_until="load", timeout=15_000)
            return await page.screenshot(type="png", omit_background=True, timeout=15_000)
        finally:
            await browser.close()


def _svg_as_png(found: Path, scratch: Path) -> Path:
    data = found.read_bytes()
    try:
        png = asyncio.run(_draw_svg(data, found.name))
    except P.PptxError:
        raise
    except Exception as exc:
        raise P.PptxError(f"{found.name} is an SVG and this server couldn't draw it ({type(exc).__name__}). "
                          "Save it as a PNG, or draw it with the visualize tool.") from None
    # The hash keeps two different logo.svg apart.
    target = scratch / hashlib.sha256(data).hexdigest()[:8] / f"{found.stem}.png"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(png)
    return target


def _download_picture(url: str, kwargs: dict[str, Any], scratch: Path) -> Path:
    """A picture at a link: web_fetch's copy if it kept one, else fetched for this
    call only (the deck holds it from then on)."""
    from captain_claw import research_images as R

    if _member() is None:
        for store in R.lookup_stores(kwargs):
            kept = store.find(url=url)
            if kept is not None:
                return Path(kept.path)

    async def fetch() -> bytes | None:
        client = R.make_client()
        try:
            return await R.download_image(client, url, allow_svg=True)
        finally:
            await client.aclose()

    data = asyncio.run(fetch())
    if data is None:
        raise P.PptxError(f"Couldn't download the picture at {url}: it's gone, not a picture, over 12 MB, or not on "
                          "the public internet. Use another, or a saved one.")
    # The name is only for messages; the hash keeps two links to a logo.png apart.
    name = re.sub(r"[^\w.-]+", "-", Path(urlsplit(url).path).name)[-60:] or "picture"
    target = scratch / hashlib.sha256(url.encode("utf-8")).hexdigest()[:8] / name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(data)
    return target


def _saved_description(found: Path) -> str:
    """What web_fetch recorded about a picture it saved: the alt text when none is given."""
    try:
        data = json.loads(found.with_name(f"{found.name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("caption") or data.get("alt") or data.get("title") or "")


def _image_finder(kwargs: dict[str, Any], scratch: Path) -> Callable[[str], Path | None]:
    """Finds a picture named in an edit, wherever it is, as a file the engine can
    read. It runs in the worker thread, so a download or a drawing can wait there.
    A member gets only what they may read (the docx tool's rule); so does an agent
    confined to its own folders."""
    from captain_claw import deck_confine, speaker
    from captain_claw.tools.document_extract import _resolve_readable_file
    from captain_claw.tools.edit import EditTool

    find = DocxTool()._image_root(kwargs)
    seen: dict[str, Path | None] = {}  # a logo on every slide is one download, one drawing

    def local(src: str) -> Path | None:
        found = find(src)
        if found is None and not is_vfs_path(src):
            # The deck's own fallbacks: the run's folder, the file registry, where a write would land.
            found = EditTool._resolve_path(src, kwargs)
            if found is None:
                candidate = WriteTool.resolve_local_target(src, kwargs)
                found = candidate if candidate.is_file() else None
        if found is None or not found.is_file():
            return None
        if _member() is not None and not speaker.path_allowed(found):
            return None
        if deck_confine.active() and not deck_confine.allowed(found.resolve(), "read"):
            return None
        if is_vfs_path(src):
            # A Drive folder holds placeholders; this asks for the real bytes.
            real, _ = asyncio.run(_resolve_readable_file(src, kwargs.get("_runtime_base_path")))
            return real
        return found

    def resolve(src: str) -> Path | None:
        if src not in seen:
            found = _download_picture(src, kwargs, scratch) if _LINK.match(src) else local(src)
            if found is not None:
                with found.open("rb") as handle:
                    head = handle.read(4096)
                if _looks_like_svg(head):
                    found = _svg_as_png(found, scratch)
            seen[src] = found
        return seen[src]

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
        "images, notes}]; body lines starting - are bullets, indent two spaces per level, **bold**, "
        "*italic*, [link](url). With no layout one is chosen for what the slide holds; a table, "
        "chart or image takes the empty placeholder that suits it. Pass `template` (a .pptx or "
        ".potx) to use a firm's design. Optional font, heading_font, heading_color, accent_color, "
        "page_numbers; then any `edits`. "
        "read: with no slide, the size, fonts and layouts, then the slides; each shape shows its "
        "id, position and text. Read before you edit. "
        "edit: `edits` = changes applied in order, all or nothing; slide numbers are as they stand "
        "after earlier edits. set_text, replace (find → replace_with across runs, on `slide`: a "
        "number, 2-5, or all), format, add_slide, duplicate_slide, delete_slide, move_slide, "
        "hide_slide, set_notes, add_text, add_image, set_image (a new picture in the same place), "
        "add_table, set_table, add_chart, set_chart "
        "(new numbers, same look), place (x, y, width, height in inches; z front/back), "
        "delete_shape, set_alt, set_theme. Aim at a shape with `shape`: its id, or title, subtitle, "
        "body, ph:<number>. Native tables and charts stay editable in PowerPoint; never paste a "
        "picture of one. A bad edit fails the whole call. A user's upload is never changed: the "
        "changed copy goes under saved/output with the same name. "
        "Pictures: `image` is a saved path, vfs:<project>/… or an https:// link, in any common "
        "format (PNG, JPEG, WebP, AVIF, HEIC, SVG, GIF…). With no coordinates it takes a free "
        "placeholder or the open part of the slide, moves text over to make room, or shares the "
        "space with the pictures already there. Or say `position`: full (fills the slide, behind "
        "the text; keep the text readable over it), left, right, top, bottom, center, or a corner "
        "like top-right for a logo; or x and y with width or height (the other side follows). "
        "Give each picture `alt`, and `caption` for a line under it such as its source. "
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
                        "image": {"type": "string",
                                  "description": "One picture: a saved path, vfs:<project>/… or an https:// link."},
                        "images": {
                            "type": "array",
                            "description": "Pictures with details, or several on this slide (they share the space).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "image": {"type": "string", "description": "A saved path, vfs:<project>/… or an https:// link."},
                                    "alt": {"type": "string", "description": "What it shows, for screen readers."},
                                    "caption": {"type": "string", "description": "A short line under it, e.g. its source."},
                                    "fit": {"type": "string", "enum": ["contain", "cover"],
                                            "description": "contain shows it whole; cover fills the space, cropping."},
                                    "position": {"type": "string", "enum": list(P._POSITIONS)},
                                    "x": {"type": "number"}, "y": {"type": "number"},
                                    "width": {"type": "number"}, "height": {"type": "number"},
                                },
                                "required": ["image"],
                            },
                        },
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
                                  "description": "Slide number. replace, format, hide_slide and add_image also take 2-5 or all."},
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
                        "image": {"type": "string",
                                  "description": "add_image/set_image/add_slide: a saved path, vfs:<project>/… or an https:// link."},
                        "images": {"type": "array", "items": {"type": "object"},
                                   "description": "add_image/add_slide: several, [{image, alt, caption, fit, position}]; they share the space."},
                        "position": {"type": "string", "enum": list(P._POSITIONS),
                                     "description": "add_image: where. full fills the slide behind the text; a corner suits a logo."},
                        "fit": {"type": "string", "enum": ["contain", "cover"],
                                "description": "add_image/set_image: contain shows it whole; cover fills the space, cropping."},
                        "caption": {"type": "string", "description": "add_image/set_image: a short line under it, e.g. its source."},
                        "alt": {"type": "string", "description": "add_image/set_image/add_table/add_chart: alt text."},
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
        author = _author(kwargs)

        def build() -> tuple[int, Any, P.Report]:
            with tempfile.TemporaryDirectory() as scratch:
                prs, report = P.build_deck(
                    slides, edits=edits, template=template.read_bytes() if template is not None else None,
                    keep_slides=bool(kwargs.get("keep_slides")), title=str(kwargs.get("title") or ""), author=author,
                    page_numbers=bool(kwargs.get("page_numbers")), look=look,
                    image_root=_image_finder(kwargs, Path(scratch)), image_alt=_saved_description)
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

        def run() -> tuple[bytes, P.Report]:
            prs = P.open_deck(source.read_bytes())
            with tempfile.TemporaryDirectory() as scratch:
                report = P.Editor(prs, image_root=_image_finder(kwargs, Path(scratch)),
                                  image_alt=_saved_description).apply(edits)
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
