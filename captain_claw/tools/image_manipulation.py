"""image_manipulation: edit, combine and convert pictures — no scripts, no model.

Asked to put two photos side by side, crop a screenshot or stamp a logo on a
picture, a model used to write a Pillow or ImageMagick script, run it in the
shell and often end with nothing: a package that wasn't there, a path that
was wrong, a picture nobody looked at. This tool does that work in-process.

The pixels are :mod:`captain_claw.image_ops` (Pillow, on the CPU); this module
decides which files are read and where results go, and reports each picture
it made for the person on a ``Path:`` line — the form every chat surface
already shows (the web chat's cards, WhatsApp, Telegram, a device's screen).

Inputs are never changed. A result lands in ``saved/media/<session>/`` unless
``output_path`` says where (``vfs:<project>/…`` for the shared folders), and
an existing file is never replaced unless ``overwrite`` is set: the result
takes a free name ("photo-edited-2.png") and the result text says which.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from captain_claw import image_ops as I
from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult
from captain_claw.tools.write import WriteTool

log = get_logger(__name__)

ACTIONS = ("info", "edit", "convert", "combine", "overlay", "animate", "frames", "split", "create", "pdf")
# Other words a model reaches for, and (for combine) the layout they imply.
_ACTION_WORDS: dict[str, tuple[str, str]] = {
    "join": ("combine", ""), "merge": ("combine", ""), "concat": ("combine", ""), "concatenate": ("combine", ""),
    "side_by_side": ("combine", "horizontal"), "collage": ("combine", "grid"), "grid": ("combine", "grid"),
    "montage": ("combine", "grid"), "stack": ("combine", "stack"), "paste": ("overlay", ""),
    "stick": ("overlay", ""), "sticker": ("overlay", ""), "composite": ("overlay", ""), "gif": ("animate", ""),
    "animation": ("animate", ""), "extract_frames": ("frames", ""), "tiles": ("split", ""), "slice": ("split", ""),
    "new": ("create", ""), "canvas": ("create", ""), "blank": ("create", ""), "to_pdf": ("pdf", ""),
    "describe": ("info", ""), "inspect": ("info", ""), "metadata": ("info", ""), "exif": ("info", ""),
    "transform": ("edit", ""), "process": ("edit", ""), "apply": ("edit", ""), "save_as": ("convert", ""),
}
# Arguments of the call itself (not of an operation) when an op name is used as the action.
_CALL_KEYS = frozenset({"action", "path", "paths", "output_path", "format", "quality", "lossless",
                        "keep_metadata", "overwrite", "operations"})
_MAX_INPUTS = 200
_MAX_INFO = 50
_PATH_LINES = 12
_LOOK_IMAGES = 4
_DEFAULT_FRAMES = 100

_LEN = "Pixels, or a percentage like '25%'."
_ANCHORS = ", ".join(I.ANCHOR_NAMES)

_OP_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": [*I.OPS, "brightness", "contrast", "saturation", "sharpness", "gamma", "hue",
                                          "warmth"]},
        "width": {"type": "string", "description": f"resize/pad/perspective/shapes. {_LEN}"},
        "height": {"type": "string", "description": f"resize/pad/perspective/shapes. {_LEN}"},
        "scale": {"type": "number", "description": "resize: factor, e.g. 0.5 = half size."},
        "fit": {"type": "string", "enum": ["contain", "cover", "fill", "pad"],
                "description": "resize with width+height: contain (keep proportions, inside), cover (fill, crop "
                               "the rest), fill (stretch), pad (contain + fill the rest with background)."},
        "shrink_only": {"type": "boolean", "description": "resize: never enlarge."},
        "resample": {"type": "string", "enum": ["lanczos", "bicubic", "bilinear", "nearest"],
                     "description": "resize: nearest keeps pixel art sharp."},
        "gravity": {"type": "string", "description": f"crop/cover/pad/circle: which part stays — {_ANCHORS}."},
        "x": {"type": "string", "description": f"Left edge (crop, shapes, text). {_LEN}"},
        "y": {"type": "string", "description": f"Top edge (crop, shapes, text). {_LEN}"},
        "box": {"type": "array", "items": {"type": "number"}, "description": "[left, top, right, bottom] in pixels."},
        "region": {"type": "object", "description": "Only this part of the picture (colour and filter ops).",
                   "properties": {"x": {"type": "string"}, "y": {"type": "string"},
                                  "width": {"type": "string"}, "height": {"type": "string"}}},
        "aspect": {"type": "string", "description": "crop/pad to a shape: '16:9', '1:1', '4:5'."},
        "top": {"type": "string", "description": f"crop: cut this much off; pad: add this much. {_LEN}"},
        "right": {"type": "string", "description": _LEN},
        "bottom": {"type": "string", "description": _LEN},
        "left": {"type": "string", "description": _LEN},
        "size": {"type": "string",
                 "description": f"border width, pad on all sides, text size, pixelate block. {_LEN}"},
        "angle": {"type": "number", "description": "rotate / text: degrees clockwise."},
        "expand": {"type": "boolean", "description": "rotate: grow to fit the turned picture (default true)."},
        "direction": {"type": "string",
                      "description": "flip: horizontal (mirror), vertical, both. gradient: vertical, up, horizontal, "
                                     "left, diagonal, radial."},
        "background": {"type": "string",
                       "description": "Colour behind: rotate corners, pad, flatten, text box (e.g. '#00000099')."},
        "color": {"type": "string",
                  "description": "Text/outline/tint/border colour; transparent: the colour to remove (default: the "
                                 "corner's)."},
        "colors": {"type": "array", "items": {"type": "string"},
                   "description": "gradient stops (e.g. ['#00000000', '#000000cc']); duotone [dark, light]."},
        "tolerance": {"type": "number", "description": "transparent/trim: 0–255, how close a colour counts."},
        "everywhere": {"type": "boolean",
                       "description": "transparent: that colour everywhere, not only the background reached from "
                                      "the edges."},
        "margin": {"type": "string", "description": f"Distance from the edge for a position; trim: border to keep. {_LEN}"},
        "radius": {"type": "string", "description": f"round_corners, rounded rectangle, text box corners. {_LEN}"},
        "corners": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                    "description": "perspective: [[x,y] top-left, top-right, bottom-right, bottom-left] of the area to "
                                   "straighten (a photographed page, a screen)."},
        "points": {"type": "array", "items": {"type": "array", "items": {"type": "number"}},
                   "description": "line/arrow/polygon: [[x, y], …] (arrow: head at the last)."},
        "brightness": {"type": "number", "description": "adjust factor: 1 = unchanged, 1.2 = 20% brighter."},
        "contrast": {"type": "number", "description": "adjust factor."},
        "saturation": {"type": "number", "description": "adjust factor: 0 = grey, 1.5 = vivid."},
        "sharpness": {"type": "number", "description": "adjust factor."},
        "gamma": {"type": "number", "description": "adjust: above 1 brightens mid-tones."},
        "hue": {"type": "number", "description": "adjust: degrees around the colour wheel."},
        "warmth": {"type": "number", "description": "adjust: -1 cooler … 1 warmer."},
        "amount": {"type": "number",
                   "description": "Strength: brightness/contrast/… as a factor; sepia/grayscale/vignette/noise/tint "
                                  "0–1; sharpen 0–10; opacity 0–1."},
        "cutoff": {"type": "number", "description": "autocontrast: % of darkest/lightest pixels to clip."},
        "bits": {"type": "integer", "description": "posterize: 1–8."},
        "threshold": {"type": "integer", "description": "threshold/solarize: 0–255."},
        "to": {"type": "string", "enum": ["RGB", "RGBA", "L", "LA", "CMYK", "P", "1"], "description": "mode."},
        "dither": {"type": "boolean", "description": "quantize / mode 1."},
        "text": {"type": "string", "description": "text/watermark: the words (\\n for a new line)."},
        "font": {"type": "string",
                 "description": "Font name (e.g. 'DejaVu Sans', 'Arial', 'Georgia'); default: an installed sans "
                                "with every character."},
        "bold": {"type": "boolean"},
        "italic": {"type": "boolean"},
        "position": {"type": "string", "description": f"text/watermark placement: {_ANCHORS} (or give x and y)."},
        "max_width": {"type": "string", "description": f"text: wrap lines at this width. {_LEN}"},
        "align": {"type": "string", "enum": ["left", "center", "right"], "description": "text lines."},
        "line_spacing": {"type": "number", "description": "text: 1.2 default."},
        "stroke": {"type": "integer", "description": "text outline width in pixels."},
        "stroke_color": {"type": "string"},
        "padding": {"type": "string", "description": "text: space inside its background box."},
        "shadow": {"type": "string", "description": "text: shadow colour, or 'true'."},
        "opacity": {"type": "number", "description": "0–1: text, watermark, shapes, gradient, or the opacity op."},
        "tile": {"type": "boolean", "description": "text/watermark: repeat over the whole picture."},
        "gap": {"type": "string", "description": "tile: space between repeats."},
        "fill": {"type": "string", "description": "rectangle/ellipse/polygon fill colour (with alpha: '#ff000055')."},
        "line_width": {"type": "integer", "description": "Shapes and lines, pixels."},
        "head": {"type": "integer", "description": "arrow head size, pixels."},
        "kind": {"type": "string", "enum": ["gaussian", "box"], "description": "blur."},
        "seed": {"type": "integer", "description": "noise: repeatable grain."},
    },
    "required": ["op"],
}

_LAYER_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "image": {"type": "integer", "description": "Which of `paths` (1 = the first)."},
        "position": {"type": "string", "description": _ANCHORS},
        "x": {"type": "string", "description": _LEN},
        "y": {"type": "string", "description": _LEN},
        "margin": {"type": "string", "description": _LEN},
        "scale": {"type": "number", "description": "Width as a share of the base picture's width, e.g. 0.2."},
        "width": {"type": "string", "description": _LEN},
        "height": {"type": "string", "description": _LEN},
        "opacity": {"type": "number"},
        "angle": {"type": "number"},
        "blend": {"type": "string", "enum": list(I.BLEND_NAMES)},
        "tile": {"type": "boolean"},
        "remove_background": {"type": "boolean"},
    },
}


@dataclass
class _Made:
    path: Path
    logical: str
    size: tuple[int, int]
    fmt: str
    nbytes: int
    frames: int = 1


def _given(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip()) and value != []


def _stem(path: Path | str) -> str:
    stem = re.sub(r"[^\w.-]+", "-", Path(str(path)).stem).strip("-.")
    return stem[:60] or "image"


def _with_ext(name: str, fmt: str) -> str:
    """*name* with the extension of *fmt* (photo.png + JPEG → photo.jpg; "out" → out.jpg)."""
    ext = I.FORMAT_EXT[fmt]
    suffix = Path(name).suffix.lower()
    if suffix in I.WRITE_FORMATS:
        return name if I.WRITE_FORMATS[suffix] == fmt else name[: -len(suffix)] + ext
    return name + ext


def _path_list(value: Any) -> list[str]:
    if not _given(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except ValueError:
                raise I.ImageOpError("`paths` isn't a valid list.") from None
        else:
            return [text]
    if not isinstance(value, (list, tuple)):
        raise I.ImageOpError("`paths` must be a list of picture paths.")
    out = []
    for item in value:
        if not isinstance(item, str):
            raise I.ImageOpError("Each entry of `paths` must be a path (text).")
        if item.strip():
            out.append(item.strip())
    return out


def _operations(kwargs: dict[str, Any]) -> list[Any]:
    raw = next((kwargs[k] for k in ("operations", "ops", "steps", "edits") if _given(kwargs.get(k))), None)
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise I.ImageOpError(f"`operations` isn't valid JSON: {exc}") from None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise I.ImageOpError("`operations` must be a list of objects, each with an `op`.")
    return raw


class ImageManipulationTool(Tool):
    """Edit, combine, convert and annotate pictures locally."""

    name = "image_manipulation"
    timeout_seconds = 180.0
    description = (
        "Change pictures yourself, locally (Pillow; no AI model) — never write PIL/ImageMagick scripts for this. "
        "Inputs stay unchanged; each result is a new file, shown to the person (its Path: line). "
        "info: size, format, frames, EXIF (date, camera, GPS), main colours — check sizes before you place things. "
        "edit: `path` (or several `paths`: the same edits on each) + `operations` applied in order — "
        "resize (width/height/scale, fit contain|cover|fill|pad), crop (x,y,width,height | box | aspect '16:9' | "
        "top/right/bottom/left to cut), rotate (angle clockwise), flip, pad, border, trim, round_corners, circle, "
        "perspective (corners → straightened); adjust (brightness/contrast/saturation/sharpness factors, gamma, "
        "hue, warmth) or brightness/contrast/… alone, grayscale, sepia, invert, autocontrast, equalize, posterize, "
        "solarize, threshold, tint, duotone, quantize, opacity, transparent (remove a plain background), flatten, "
        "mode; blur, sharpen, median, pixelate, edges, edge_enhance, emboss, contour, detail, smooth, vignette, "
        "noise, gradient; text and watermark (font, size, colour, stroke, shadow, background box, angle, tile), "
        "rectangle, ellipse, line, arrow, polygon. Colour and filter ops take `region` {x,y,width,height} to work on "
        "part of the picture (blur a face, pixelate a plate). Animated GIF/WebP stay animated. "
        "convert: another format (format, or output_path's extension: png jpg webp gif bmp tiff ico avif pdf; "
        "quality). combine: `paths` side by side (layout horizontal), one above another (vertical), grid "
        "(columns) or stacked on top of each other (stack) — gap, padding, background, labels. "
        "overlay: stick `paths` onto `path` — position (top-left … bottom-right, center) or x/y, scale (share of "
        "the base width), opacity, angle, blend (multiply, screen, …), tile, remove_background (a logo on "
        "white); `layers` to place each one differently. animate: `paths` → animated GIF/WebP (duration ms, loop, bounce). frames: an animation's "
        "frames as files. split: one picture into rows × columns tiles. create: a blank canvas (width, height, "
        "background) to draw on with operations. pdf: pictures → one PDF, a page each (page_size a4|letter|fit). "
        "combine, overlay, animate and create also take `operations`, applied to the result. "
        "Lengths are pixels or '25%'; colours are names, #rrggbb or #rrggbbaa, or 'transparent'. "
        "To describe or read a picture use image_vision / ocr_image; to measure, compare or detect use cv; for a "
        "new picture from a prompt use image_gen; for charts use visualize."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS), "description": "What to do (default edit)."},
            "path": {"type": "string",
                     "description": "The picture (edit/convert/info/frames/split), or the base picture for overlay."},
            "paths": {"type": "array", "items": {"type": "string"},
                      "description": "Several pictures: combine/animate/pdf (in order), overlay (what goes on top), "
                                     "edit/convert/info (each one)."},
            "operations": {"type": "array", "items": _OP_ITEM,
                           "description": "Edits applied in order, e.g. [{op: resize, width: 1080}, "
                                          "{op: text, text: 'Hello', position: bottom}]."},
            "output_path": {"type": "string",
                            "description": "Where to save (a name like 'collage.png' lands in saved/media/; "
                                           "vfs:<project>/… for shared folders). Its extension picks the format. For "
                                           "several results, a folder."},
            "format": {"type": "string", "enum": ["png", "jpg", "webp", "gif", "bmp", "tiff", "ico", "avif", "pdf"],
                       "description": "Save as this format (default: the input's, PNG when transparency needs it)."},
            "quality": {"type": "integer", "description": "jpg/webp/avif quality 1–100 (default 90)."},
            "lossless": {"type": "boolean", "description": "webp: lossless."},
            "keep_metadata": {"type": "boolean",
                              "description": "Keep EXIF (camera, date, GPS). Off by default: results carry none."},
            "overwrite": {"type": "boolean", "description": "Replace output_path if it exists (default: a free name)."},
            "background": {"type": "string",
                           "description": "combine/create/pdf canvas colour; what fills transparency when saving as "
                                          "jpg (default white)."},
            "layout": {"type": "string", "enum": ["horizontal", "vertical", "grid", "stack"], "description": "combine."},
            "columns": {"type": "integer", "description": "combine grid / split."},
            "rows": {"type": "integer", "description": "split."},
            "gap": {"type": "integer", "description": "combine: pixels between pictures; overlay tile: between repeats."},
            "padding": {"type": "integer", "description": "combine: pixels around the edge."},
            "align": {"type": "string", "enum": ["start", "center", "end"],
                      "description": "combine: pictures of different sizes line up at the start, centre or end."},
            "match": {"type": "string", "enum": ["height", "width", "none"],
                      "description": "combine: side by side get the same height, one above another the same width "
                                     "(default: the middle one's; set height/width to choose), or none."},
            "fit": {"type": "string", "enum": ["contain", "cover", "fill"],
                    "description": "combine grid / animate: how a picture fills its cell (contain letterboxes)."},
            "cell_width": {"type": "integer", "description": "combine grid."},
            "cell_height": {"type": "integer", "description": "combine grid."},
            "labels": {"type": "array", "items": {"type": "string"},
                       "description": "combine: a caption under each picture."},
            "label_size": {"type": "integer"},
            "label_color": {"type": "string"},
            "position": {"type": "string", "description": f"overlay placement: {_ANCHORS}."},
            "x": {"type": "string", "description": f"overlay: left edge. {_LEN}"},
            "y": {"type": "string", "description": f"overlay: top edge. {_LEN}"},
            "margin": {"type": "string", "description": f"overlay: distance from the edge. {_LEN}"},
            "scale": {"type": "number", "description": "overlay: width as a share of the base width, e.g. 0.2."},
            "width": {"type": "string",
                      "description": f"create: canvas width; overlay/animate: picture width. combine vertical: the "
                                     f"shared width. {_LEN}"},
            "height": {"type": "string",
                       "description": f"create: canvas height; overlay/animate: picture height. combine horizontal: "
                                      f"the shared height. {_LEN}"},
            "opacity": {"type": "number", "description": "overlay: 0–1."},
            "angle": {"type": "number", "description": "overlay: degrees clockwise."},
            "blend": {"type": "string", "enum": list(I.BLEND_NAMES), "description": "overlay."},
            "tile": {"type": "boolean", "description": "overlay: repeat over the whole picture."},
            "remove_background": {"type": "boolean",
                                  "description": "overlay: make the plain background of what goes on top see-through "
                                                 "first (a logo on white)."},
            "tolerance": {"type": "number", "description": "overlay remove_background: 0–255 (default 30)."},
            "layers": {"type": "array", "items": _LAYER_ITEM,
                       "description": "overlay: each of `paths` placed its own way; unset keys come from the call."},
            "duration": {"type": "integer", "description": "animate: milliseconds per frame (default 500)."},
            "durations": {"type": "array", "items": {"type": "integer"}, "description": "animate: per frame."},
            "loop": {"type": "integer", "description": "animate: 0 = forever (default), else how many times."},
            "bounce": {"type": "boolean", "description": "animate: play forwards, then backwards."},
            "every": {"type": "integer", "description": "frames: keep every Nth frame."},
            "max_frames": {"type": "integer", "description": f"frames: at most this many (default {_DEFAULT_FRAMES})."},
            "tile_width": {"type": "string", "description": f"split: tile size instead of rows/columns. {_LEN}"},
            "tile_height": {"type": "string", "description": _LEN},
            "page_size": {"type": "string", "enum": ["a4", "letter", "a3", "a5", "legal", "fit"],
                          "description": "pdf: fit = each page the size of its picture."},
            "dpi": {"type": "integer", "description": "pdf: resolution (default 150)."},
        },
        "required": ["action"],
    }

    # ── Entry ──────────────────────────────────────────────────────

    async def execute(self, action: str = "edit", **kwargs: Any) -> ToolResult:
        asked = re.sub(r"[\s\-]+", "_", str(action or "edit").strip().lower())
        act, layout = _ACTION_WORDS.get(asked, (asked, ""))
        if layout and not _given(kwargs.get("layout")):
            kwargs["layout"] = layout
        if act not in ACTIONS:
            if not I.canonical_op(asked):
                return ToolResult(success=False, error=f"Unknown action {action!r}; use one of: {', '.join(ACTIONS)} "
                                                        "(edits such as resize or crop go in `operations`).")
            # resize, crop, rotate… asked for as the action: one edit with the call's arguments.
            op = {k: v for k, v in kwargs.items() if not k.startswith("_") and k not in _CALL_KEYS}
            kwargs["operations"] = [{**op, "op": asked}, *_operations(kwargs)]
            act = "edit"
        try:
            return await getattr(self, f"_do_{act}")(kwargs)
        except I.ImageOpError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            log.error("image_manipulation failed", action=act, error=str(exc))
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

    # ── Paths ──────────────────────────────────────────────────────

    async def _source(self, raw: str, kwargs: dict[str, Any]) -> Path:
        """An existing picture: vfs- and Drive-aware like the extract tools, then
        the edit tool's fallbacks, then where a result of this session would be."""
        from captain_claw.tools.document_extract import _resolve_readable_file
        from captain_claw.tools.edit import EditTool
        from captain_claw.vfs import is_vfs_path

        text = str(raw or "").strip()
        if text.lower().startswith("file://"):
            text = text[len("file://"):]
        if re.match(r"^https?://", text, re.IGNORECASE):
            raise I.ImageOpError(f"{text} is a web address: save it first (web_get or the browser), then pass the "
                                 "saved file.")
        found, error = await _resolve_readable_file(text, kwargs.get("_runtime_base_path"))
        if found is None:
            found = EditTool._resolve_path(text, kwargs)
        if found is None and not is_vfs_path(text):
            candidate = WriteTool.resolve_local_target(text, kwargs)
            found = candidate if candidate.is_file() else None
        if found is None or not found.is_file():
            raise I.ImageOpError(error or f"File not found: {text}")
        return found

    async def _inputs(self, kwargs: dict[str, Any], *, least: int = 1, most: int = _MAX_INPUTS) -> list[Path]:
        raw = ([str(kwargs["path"]).strip()] if _given(kwargs.get("path")) else []) + _path_list(kwargs.get("paths"))
        if len(raw) < least:
            need = "a picture in `path`" if least == 1 else f"at least {least} pictures in `paths`"
            raise I.ImageOpError(f"This needs {need}.")
        if len(raw) > most:
            raise I.ImageOpError(f"At most {most} pictures in one call.")
        return [await self._source(r, kwargs) for r in raw]

    @staticmethod
    def _font_files(kwargs: dict[str, Any]) -> Callable[[str], Path | None]:
        """A font given as a file: the agent's own files only (never past a deck's walls)."""

        def resolve(raw: str) -> Path | None:
            from captain_claw import deck_confine
            from captain_claw.vfs import is_vfs_path, resolve_vfs_path

            try:
                p = resolve_vfs_path(raw) if is_vfs_path(raw) else Path(raw).expanduser()
                if p is None:
                    return None
                if not p.is_absolute() and kwargs.get("_runtime_base_path") is not None:
                    p = Path(kwargs["_runtime_base_path"]) / p
                p = p.resolve()
                if not p.is_file() or (deck_confine.active() and not deck_confine.allowed(p, "read")):
                    return None
                return p
            except (OSError, ValueError, RuntimeError):
                return None

        return resolve

    def _context(self, kwargs: dict[str, Any]) -> I.Context:
        return I.Context(fonts=I.Fonts(self._font_files(kwargs)))

    @staticmethod
    def _asked_format(kwargs: dict[str, Any]) -> str | None:
        if _given(kwargs.get("format")):
            return I.format_named(kwargs["format"])
        suffix = Path(str(kwargs.get("output_path") or "")).suffix.lower()
        if suffix in I.WRITE_FORMATS:
            return I.format_named(suffix)
        return None

    @staticmethod
    def _target(name: str, kwargs: dict[str, Any], taken: set[str]) -> Path:
        """Where a result called *name* is written: never over an existing file
        (unless overwrite) or another result of this call."""
        from captain_claw.vfs import (
            is_vfs_path,
            project_is_readonly,
            resolve_vfs_path,
            split_scheme,
        )

        if is_vfs_path(name):
            project = split_scheme(name)[0]
            if project_is_readonly(project):
                raise I.ImageOpError(f"'{project}' is a read-only folder — save somewhere else.")
            target = resolve_vfs_path(name, create_parents=True)
            if target is None:
                raise I.ImageOpError(f"Invalid vfs path (it leaves your folders): {name}")
        else:
            if len(Path(name).parts) == 1:
                name = f"media/{name}"
            target = WriteTool.resolve_local_target(name, kwargs)
        target.parent.mkdir(parents=True, exist_ok=True)
        overwrite = I.boolean(kwargs.get("overwrite"))

        def busy(p: Path) -> bool:
            return str(p.resolve()) in taken or (p.exists() and not overwrite)

        candidate = target
        n = 2
        while busy(candidate):
            if n > 999:
                raise I.ImageOpError(f"No free file name next to {target.name}.")
            candidate = target.with_name(f"{target.stem}-{n}{target.suffix}")
            n += 1
        taken.add(str(candidate.resolve()))
        return candidate

    @staticmethod
    def _names(kwargs: dict[str, Any], keys: list[str], default_folder: str, default_stems: list[str],
               fmt: str) -> list[str]:
        """Names for several results: output_path as a pattern (tile.png → tile-r1c1.png) or a folder."""
        raw = str(kwargs.get("output_path") or "").strip()
        if raw and Path(raw).suffix.lower() in I.WRITE_FORMATS:
            base = raw[: -len(Path(raw).suffix)]
            return [_with_ext(f"{base}-{k}", fmt) for k in keys]
        folder = raw.rstrip("/\\") if raw else (f"media/{default_folder}" if default_folder else "media")
        return [_with_ext(f"{folder}/{stem}", fmt) for stem in default_stems]

    @staticmethod
    def _save(data: bytes, target: Path) -> None:
        """Write atomically, then record who made it (as the xlsx/docx tools do)."""
        from captain_claw import saved_attribution
        from captain_claw.vfs import record_author

        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            prior = saved_attribution.prior_creator(target)
            os.replace(tmp, target)
            saved_attribution.note_write(target, prior)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        try:
            record_author(target)
        except Exception:
            pass

    def _write(self, frames: list[Any], fmt: str, name: str, kwargs: dict[str, Any], ctx: I.Context,
               taken: set[str], *, durations: list[int] | None = None, loop: int = 0,
               icc_profile: bytes | None = None, exif: Any = None, dpi: float | None = None) -> _Made:
        data = I.encode(
            frames, fmt, ctx, durations=durations, loop=loop,
            quality=I.integer(kwargs.get("quality"), "quality", None, 1, 100),
            lossless=I.boolean(kwargs.get("lossless")),
            background=I.colour(kwargs.get("background"), "background"),
            icc_profile=icc_profile,
            exif=I.exif_bytes(exif) if I.boolean(kwargs.get("keep_metadata")) else None,
            dpi=dpi,
        )
        target = self._target(name, kwargs, taken)
        self._save(data, target)
        kept = len(frames) if fmt in I.ANIMATED | {"PDF", "TIFF"} else 1
        return _Made(target, name, frames[0].size, fmt, len(data), kept)

    @staticmethod
    def _register(made: list[_Made], kwargs: dict[str, Any]) -> None:
        registry = kwargs.get("_file_registry")
        if registry is None:
            return
        for m in made:
            try:
                registry.register(logical_path=m.logical, physical_path=str(m.path),
                                  task_id=str(kwargs.get("_task_id", "")))
            except Exception:
                pass

    def _result(self, headline: str, made: list[_Made], kwargs: dict[str, Any], *, inputs: list[Path] | None = None,
                steps: list[str] | None = None, notes: list[str] | None = None, deliver: bool = True) -> ToolResult:
        self._register(made, kwargs)
        lines = [headline]
        label = "Path" if deliver else "File"
        for m in made[:_PATH_LINES]:
            lines.append(f"{label}: {m.path}")
            if m.fmt == "PDF":
                detail = f"{m.frames} page{'s' if m.frames != 1 else ''} · PDF"
            else:
                detail = f"{m.size[0]}×{m.size[1]} px · {m.fmt}" + (f" · {m.frames} frames" if m.frames > 1 else "")
            lines.append(f"  {detail} · {I.human_bytes(m.nbytes)}")
        if len(made) > _PATH_LINES:
            lines.append(f"… and {len(made) - _PATH_LINES} more in {made[-1].path.parent} "
                         f"(the last is {made[-1].path.name}).")
        if steps:
            lines.append("Applied: " + ", ".join(steps))
        if inputs:
            shown = ", ".join(p.name for p in inputs[:12])
            lines.append("From: " + shown + (f" … ({len(inputs)} pictures)" if len(inputs) > 12 else ""))
        lines += [f"Note: {n}" for n in (notes or [])]
        images: list[str] = []
        if kwargs.get("_look_inline"):
            images = [str(m.path) for m in made if m.path.suffix.lower() in I.VIEWABLE][:_LOOK_IMAGES]
        return ToolResult(success=True, content="\n".join(lines), images=images)

    # ── Actions ────────────────────────────────────────────────────

    async def _do_info(self, kwargs: dict[str, Any]) -> ToolResult:
        sources = await self._inputs(kwargs, most=_MAX_INFO)
        lines: list[str] = []
        for src in sources:
            lines += await asyncio.to_thread(I.describe, src)
        return ToolResult(success=True, content="\n".join(lines))

    async def _do_convert(self, kwargs: dict[str, Any]) -> ToolResult:
        if self._asked_format(kwargs) is None:
            raise I.ImageOpError("convert needs `format` (png, jpg, webp, gif, bmp, tiff, ico, avif, pdf) or an "
                                 "output_path with that extension.")
        return await self._do_edit(kwargs, converting=True)

    async def _do_edit(self, kwargs: dict[str, Any], *, converting: bool = False) -> ToolResult:
        ops = _operations(kwargs)
        asked = self._asked_format(kwargs)
        if not ops and asked is None:
            raise I.ImageOpError("edit needs `operations`, e.g. [{\"op\": \"resize\", \"width\": 1080}] — or use "
                                 "convert with a format.")
        sources = await self._inputs(kwargs)

        def run() -> tuple[list[_Made], list[str], list[str]]:
            ctx = self._context(kwargs)
            taken: set[str] = set()
            made: list[_Made] = []
            steps: list[str] = []
            tag = "" if converting and not ops else "-edited"
            stems = [_stem(s) + tag for s in sources]
            if len(sources) == 1:
                names = [str(kwargs.get("output_path") or "").strip() or stems[0]]
            else:
                names = self._names(kwargs, [str(k) for k in range(1, len(sources) + 1)], "", stems, "PNG")
            for src, name in zip(sources, names):
                pic = I.load(src)
                frames = pic.frames
                if pic.animated and asked is not None and asked not in I.ANIMATED:
                    ctx.note(f"{src.name} is animated; {asked} keeps only its first frame")
                    frames = frames[:1]
                done: list[Any] = []
                for frame in frames:
                    out, steps = I.apply_operations(frame, ops, ctx)
                    done.append(out)
                fmt, why = I.choose_format(asked=asked, source=pic.format,
                                           alpha=any(I.has_alpha(f) for f in done), animated=len(done) > 1)
                ctx.note(why)
                made.append(self._write(done, fmt, _with_ext(name, fmt), kwargs, ctx, taken,
                                        durations=pic.durations, loop=pic.loop, icc_profile=pic.icc_profile,
                                        exif=pic.exif))
            return made, steps, ctx.notes

        made, steps, notes = await asyncio.to_thread(run)
        if converting and not ops:
            headline = (f"Converted {sources[0].name} to {made[0].fmt}." if len(made) == 1
                        else f"Converted {len(made)} pictures to {made[0].fmt}.")
        else:
            headline = f"Edited {sources[0].name}." if len(made) == 1 else f"Edited {len(made)} pictures."
        return self._result(headline, made, kwargs, inputs=sources if len(sources) > 1 else None,
                            steps=steps, notes=notes)

    def _finish_one(self, image: Any, kwargs: dict[str, Any], ctx: I.Context, default_name: str,
                    source_format: str | None, *, default_format: str = "PNG") -> tuple[_Made, list[str]]:
        """*image* with the call's operations applied, saved under output_path or *default_name*."""
        out, steps = I.apply_operations(image, _operations(kwargs), ctx)
        fmt, why = I.choose_format(asked=self._asked_format(kwargs), source=source_format,
                                   alpha=I.has_alpha(out), animated=False, default=default_format)
        ctx.note(why)
        name = str(kwargs.get("output_path") or "").strip() or default_name
        return self._write([out], fmt, _with_ext(name, fmt), kwargs, ctx, set()), steps

    async def _do_combine(self, kwargs: dict[str, Any]) -> ToolResult:
        sources = await self._inputs(kwargs, least=2)

        def run() -> tuple[str, _Made, list[str], list[str]]:
            ctx = self._context(kwargs)
            pics = [I.load(s, all_frames=False) for s in sources]
            combined, summary = I.combine([p.image for p in pics], kwargs, ctx)
            made, steps = self._finish_one(combined, kwargs, ctx, f"{_stem(sources[0])}-combined", pics[0].format)
            return summary, made, steps, ctx.notes

        summary, made, steps, notes = await asyncio.to_thread(run)
        return self._result(summary, [made], kwargs, inputs=sources, steps=steps, notes=notes)

    async def _do_overlay(self, kwargs: dict[str, Any]) -> ToolResult:
        tops_raw = _path_list(kwargs.get("paths"))
        if _given(kwargs.get("path")):
            base_raw = str(kwargs["path"]).strip()
            default_layers = list(range(len(tops_raw)))
        elif len(tops_raw) >= 2:
            base_raw = tops_raw[0]
            default_layers = list(range(1, len(tops_raw)))
        else:
            raise I.ImageOpError("overlay needs the base picture in `path` and what goes on top in `paths`.")
        if not tops_raw or not default_layers:
            raise I.ImageOpError("overlay needs what goes on top in `paths` (a logo, a sticker, another photo).")
        if len(tops_raw) > _MAX_INPUTS:
            raise I.ImageOpError(f"At most {_MAX_INPUTS} pictures in one call.")
        base_path = await self._source(base_raw, kwargs)
        tops = [await self._source(r, kwargs) for r in tops_raw]
        shared = {k: kwargs[k] for k in ("position", "x", "y", "margin", "scale", "width", "height", "opacity",
                                         "angle", "blend", "tile", "gap", "remove_background", "tolerance")
                  if _given(kwargs.get(k))}
        layers_raw = I.listish(kwargs.get("layers"), "layers")
        placements: list[tuple[int, dict[str, Any]]] = []
        if layers_raw:
            for n, layer in enumerate(layers_raw, 1):
                if not isinstance(layer, dict):
                    raise I.ImageOpError(f"layers[{n}] must be an object with `image` (which of `paths`, 1 = first).")
                index = I.integer(layer.get("image"), f"layers[{n}].image", None, 1, len(tops))
                index = (index - 1) if index is not None else default_layers[min(n - 1, len(default_layers) - 1)]
                placements.append((index, {**shared, **{k: v for k, v in layer.items() if k != "image"}}))
        else:
            placements = [(i, dict(shared)) for i in default_layers]

        def run() -> tuple[_Made, list[str], list[str]]:
            ctx = self._context(kwargs)
            base = I.load(base_path, all_frames=False)
            loaded = {i: I.load(tops[i], all_frames=False).image for i in sorted({i for i, _ in placements})}
            out = I.overlay(base.image, [(loaded[i], spec) for i, spec in placements], ctx)
            made, steps = self._finish_one(out, kwargs, ctx, f"{_stem(base_path)}-overlay", base.format)
            return made, steps, ctx.notes

        made, steps, notes = await asyncio.to_thread(run)
        count = len(placements)
        headline = f"Put {count} picture{'s' if count != 1 else ''} on {base_path.name}."
        return self._result(headline, [made], kwargs, inputs=[base_path, *tops], steps=steps, notes=notes)

    async def _do_animate(self, kwargs: dict[str, Any]) -> ToolResult:
        sources = await self._inputs(kwargs)
        asked = self._asked_format(kwargs)
        if asked is not None and asked not in I.ANIMATED:
            raise I.ImageOpError("An animation saves as gif, webp or png (APNG).")

        def run() -> tuple[_Made, list[str], list[str], int]:
            ctx = self._context(kwargs)
            if len(sources) == 1:
                pic = I.load(sources[0])
                if not pic.animated:
                    raise I.ImageOpError("animate needs at least two pictures in `paths` (or one animated picture).")
                frames = pic.frames
                if not _given(kwargs.get("duration")) and not _given(kwargs.get("durations")):
                    kwargs["durations"] = pic.durations
            else:
                frames = [I.load(s, all_frames=False).image for s in sources]
            frames, durations = I.animate(frames, kwargs, ctx)
            ops = _operations(kwargs)
            steps: list[str] = []
            done = []
            for frame in frames:
                out, steps = I.apply_operations(frame, ops, ctx)
                done.append(out)
            fmt = asked or "GIF"
            name = str(kwargs.get("output_path") or "").strip() or f"{_stem(sources[0])}-animated"
            made = self._write(done, fmt, _with_ext(name, fmt), kwargs, ctx, set(), durations=durations,
                               loop=I.integer(kwargs.get("loop"), "loop", 0, 0, 65535) or 0)
            return made, steps, ctx.notes, sum(durations)

        made, steps, notes, total = await asyncio.to_thread(run)
        headline = f"Made an animation of {made.frames} frames ({total / 1000:.1f} s)."
        return self._result(headline, [made], kwargs, inputs=sources, steps=steps, notes=notes)

    async def _do_frames(self, kwargs: dict[str, Any]) -> ToolResult:
        source = (await self._inputs(kwargs))[0]
        asked = self._asked_format(kwargs)
        if asked == "PDF":
            raise I.ImageOpError("frames are saved as pictures (png, jpg, webp …); for a PDF use action pdf.")

        def run() -> tuple[list[_Made], int, list[str]]:
            ctx = self._context(kwargs)
            pic = I.load(source)
            if not pic.animated:
                raise I.ImageOpError(f"{source.name} has a single frame — nothing to take apart.")
            every = I.integer(kwargs.get("every"), "every", 1, 1, I.MAX_FRAMES) or 1
            most = I.integer(kwargs.get("max_frames"), "max_frames", _DEFAULT_FRAMES, 1, I.MAX_FRAMES) or 1
            picked = list(range(0, len(pic.frames), every))[:most]
            fmt = asked or "PNG"
            keys = [f"{i + 1:03d}" for i in picked]
            names = self._names(kwargs, keys, f"{_stem(source)}-frames", [f"frame-{k}" for k in keys], fmt)
            ops = _operations(kwargs)
            taken: set[str] = set()
            made = []
            for i, name in zip(picked, names):
                out, _steps = I.apply_operations(pic.frames[i], ops, ctx)
                made.append(self._write([out], fmt, name, kwargs, ctx, taken))
            return made, len(pic.frames), ctx.notes

        made, total, notes = await asyncio.to_thread(run)
        headline = f"Saved {len(made)} of {total} frames of {source.name}."
        return self._result(headline, made, kwargs, notes=notes, deliver=False)

    async def _do_split(self, kwargs: dict[str, Any]) -> ToolResult:
        source = (await self._inputs(kwargs))[0]

        def run() -> tuple[list[_Made], list[str]]:
            ctx = self._context(kwargs)
            pic = I.load(source, all_frames=False)
            tiles = I.split(pic.image, kwargs)
            ops = _operations(kwargs)
            done = [(key, I.apply_operations(tile, ops, ctx)[0]) for key, tile in tiles]
            fmt, why = I.choose_format(asked=self._asked_format(kwargs), source=pic.format,
                                       alpha=any(I.has_alpha(t) for _, t in done), animated=False)
            ctx.note(why)
            keys = [key for key, _ in done]
            names = self._names(kwargs, keys, f"{_stem(source)}-tiles", [f"{_stem(source)}-{k}" for k in keys], fmt)
            taken: set[str] = set()
            made = [self._write([tile], fmt, name, kwargs, ctx, taken) for (_key, tile), name in zip(done, names)]
            return made, ctx.notes

        made, notes = await asyncio.to_thread(run)
        return self._result(f"Cut {source.name} into {len(made)} tiles.", made, kwargs, notes=notes)

    async def _do_create(self, kwargs: dict[str, Any]) -> ToolResult:
        def run() -> tuple[_Made, list[str], list[str]]:
            ctx = self._context(kwargs)
            width = I.length(kwargs.get("width"), 1080, "width", 1080) or 1080
            height = I.length(kwargs.get("height"), 1080, "height", width) or width
            I.check_size(width, height, "The canvas")
            fill = I.colour(kwargs.get("background"), "background", (255, 255, 255, 255)) or (255, 255, 255, 255)
            canvas =Image.new("RGBA", (width, height), fill) if fill[3] < 255 else \
                Image.new("RGB", (width, height), fill[:3])
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            made, steps = self._finish_one(canvas, kwargs, ctx, f"canvas-{stamp}", None)
            return made, steps, ctx.notes

        made, steps, notes = await asyncio.to_thread(run)
        return self._result(f"Made a {made.size[0]}×{made.size[1]} picture.", [made], kwargs, steps=steps, notes=notes)

    async def _do_pdf(self, kwargs: dict[str, Any]) -> ToolResult:
        sources = await self._inputs(kwargs)
        asked = self._asked_format(kwargs)
        if asked not in (None, "PDF"):
            raise I.ImageOpError("action pdf saves a PDF; for pictures use convert.")

        def run() -> tuple[_Made, list[str], list[str]]:
            ctx = self._context(kwargs)
            ops = _operations(kwargs)
            images = []
            steps: list[str] = []
            for src in sources:
                out, steps = I.apply_operations(I.load(src, all_frames=False).image, ops, ctx)
                images.append(out)
            pages, dpi = I.pdf_pages(images, kwargs)
            stem = _stem(sources[0]) if len(sources) == 1 else f"{_stem(sources[0])}-and-{len(sources) - 1}-more"
            name = str(kwargs.get("output_path") or "").strip() or stem
            made = self._write(pages, "PDF", _with_ext(name, "PDF"), kwargs, ctx, set(), dpi=dpi)
            return made, steps, ctx.notes

        made, steps, notes = await asyncio.to_thread(run)
        size = str(kwargs.get("page_size") or "a4").upper()
        headline = f"Made a {made.frames}-page PDF ({'pages sized to the pictures' if size == 'FIT' else size})."
        return self._result(headline, [made], kwargs, inputs=sources, steps=steps, notes=notes)
