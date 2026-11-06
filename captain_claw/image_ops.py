"""Pixels for the image_manipulation tool: Pillow only, no file policy.

``tools/image_manipulation.py`` decides which files are read and where results
go; this module opens pictures, changes them, lays several out together and
encodes the result. Everything runs on the CPU in this process — no model, no
network, no scripts.

Conventions the tool's description promises:

* Lengths (x, y, width, height, size, margin, radius, …) are pixels, or a
  percentage of the picture ("25%"); a bare number between 0 and 1 counts as a
  fraction too (0.5 = half).
* Angles are degrees clockwise, as in photo apps (Pillow turns the other way).
* Colours are names (red), hex (#f80, #ff8800, #ff880080 with transparency),
  rgb()/rgba()/hsl(), or "transparent".
* Positions are anchors: top-left, top, top-right, left, center, right,
  bottom-left, bottom, bottom-right ("upper left", "centre", "bottom_right"
  are read the same).
* A phone photo is turned upright from its EXIF orientation when it is opened,
  so what the model sees and what it edits are the same way up.
"""

from __future__ import annotations

import functools
import io
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
    ImageSequence,
    UnidentifiedImageError,
    features,
)


class ImageOpError(ValueError):
    """A request this module can't carry out; the message says what to change."""


# Any canvas this module makes, a picture it opens, and an animation's frames
# together. 80 MP is a 10,000 × 8,000 poster; RGBA that is 320 MB in memory.
MAX_PIXELS = 80_000_000
MAX_INPUT_PIXELS = 150_000_000
MAX_FRAMES = 500
MAX_ANIMATION_PIXELS = 400_000_000
MAX_TILES = 400

WRITE_FORMATS: dict[str, str] = {
    ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP", ".gif": "GIF",
    ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF", ".ico": "ICO", ".avif": "AVIF", ".pdf": "PDF",
}
FORMAT_EXT: dict[str, str] = {
    "PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp",
    "TIFF": ".tiff", "ICO": ".ico", "AVIF": ".avif", "PDF": ".pdf",
}
_FORMAT_WORDS: dict[str, str] = {
    "png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP", "gif": "GIF", "bmp": "BMP",
    "tif": "TIFF", "tiff": "TIFF", "ico": "ICO", "icon": "ICO", "avif": "AVIF", "pdf": "PDF",
}
# What Pillow calls a format it reads, when it writes it under another name.
_READ_AS_WRITE = {"MPO": "JPEG", "JPEG2000": "PNG", "APNG": "PNG"}
NO_ALPHA = frozenset({"JPEG", "BMP", "PDF"})
ANIMATED = frozenset({"GIF", "WEBP", "PNG"})
VIEWABLE = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})

# Files Pillow can't open, and what to do instead.
_NOT_PICTURES = {
    ".svg": "an SVG is a drawing written as text: render it to PNG with the visualize tool first",
    ".pdf": "a PDF isn't a picture: read it with pdf_extract or ocr_pdf",
    ".heic": "HEIC photos need the pillow-heif package (pip install pillow-heif)",
    ".heif": "HEIF photos need the pillow-heif package (pip install pillow-heif)",
    ".mp4": "that's a video: take its frames first (video_vision, or cv keyframes)",
    ".mov": "that's a video: take its frames first (video_vision, or cv keyframes)",
    ".webm": "that's a video: take its frames first (video_vision, or cv keyframes)",
}


# ── Opening ────────────────────────────────────────────────────────────


@dataclass
class Picture:
    """A picture as opened: one frame, or every frame of an animation."""

    frames: list[Image.Image]
    durations: list[int] = field(default_factory=list)
    loop: int = 0
    format: str = ""
    icc_profile: bytes | None = None
    exif: Image.Exif | None = None
    name: str = ""

    @property
    def image(self) -> Image.Image:
        return self.frames[0]

    @property
    def animated(self) -> bool:
        return len(self.frames) > 1


@functools.lru_cache(maxsize=1)
def _extra_openers() -> bool:
    """HEIC/HEIF when pillow-heif happens to be installed (it's optional)."""
    try:
        import pillow_heif  # type: ignore[import-not-found]

        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


def load(path: Path, *, all_frames: bool = True) -> Picture:
    """*path* opened, upright, as RGB or RGBA frames."""
    _extra_openers()
    try:
        im = Image.open(path)
    except UnidentifiedImageError:
        hint = _NOT_PICTURES.get(path.suffix.lower())
        raise ImageOpError(
            f"{path.name} isn't a picture this tool can open"
            + (f" — {hint}." if hint else " (PNG, JPEG, WebP, GIF, BMP, TIFF, ICO, AVIF and PSD are fine).")
        ) from None
    except Image.DecompressionBombError:
        raise ImageOpError(f"{path.name} is too large to open safely.") from None
    except OSError as exc:
        raise ImageOpError(f"Can't open {path.name}: {exc}") from None
    with im:
        w, h = im.size
        if w * h > MAX_INPUT_PIXELS:
            raise ImageOpError(f"{path.name} is {w}×{h} px — over {MAX_INPUT_PIXELS // 1_000_000} megapixels, "
                               "too large to work on here.")
        fmt = _READ_AS_WRITE.get(im.format or "", im.format or "")
        icc = im.info.get("icc_profile")
        try:
            exif = im.getexif()
        except Exception:
            exif = None
        n = int(getattr(im, "n_frames", 1) or 1)
        frames: list[Image.Image] = []
        durations: list[int] = []
        try:
            if all_frames and n > 1:
                budget = MAX_ANIMATION_PIXELS
                for i, frame in enumerate(ImageSequence.Iterator(im)):
                    if i >= MAX_FRAMES or budget <= 0:
                        break
                    budget -= w * h
                    durations.append(int(frame.info.get("duration") or im.info.get("duration") or 100))
                    frames.append(normalize(frame.copy()))
            else:
                if n > 1:
                    im.seek(0)
                upright = ImageOps.exif_transpose(im)
                if upright is None or upright is im:
                    upright = im.copy()
                frames.append(normalize(upright))
        except (OSError, SyntaxError, ValueError) as exc:
            raise ImageOpError(f"{path.name} couldn't be read completely: {exc}") from None
        loop = int(im.info.get("loop", 0) or 0)
    return Picture(frames, durations, loop, fmt, icc, exif, path.name)


def normalize(img: Image.Image) -> Image.Image:
    """RGB, or RGBA when the picture has transparency, whatever it was stored as."""
    mode = img.mode
    if mode in ("RGB", "RGBA"):
        return img
    if mode.startswith("I") or mode == "F":
        import numpy as np

        arr = np.asarray(img, dtype="float64")
        top = float(arr.max()) if arr.size else 1.0
        hi = 65535.0 if mode.startswith("I;16") or top > 255 else 1.0 if top <= 1.0 else 255.0
        return Image.fromarray(np.clip(arr / hi * 255.0, 0, 255).astype("uint8"), "L").convert("RGB")
    if mode in ("LA", "PA", "La", "RGBa") or "transparency" in img.info:
        return img.convert("RGBA")
    return img.convert("RGB")


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in img.info


def _work(img: Image.Image) -> Image.Image:
    """RGB or RGBA, for the operations that need colour channels."""
    if img.mode in ("RGB", "RGBA"):
        return img
    return img.convert("RGBA" if has_alpha(img) else "RGB")


def _rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else _work(img).convert("RGBA")


def _on_rgb(img: Image.Image, fn: Callable[[Image.Image], Image.Image]) -> Image.Image:
    """*fn* applied to the colour channels; the transparency stays as it was."""
    img = _work(img)
    if img.mode == "RGBA":
        alpha = img.getchannel("A")
        out = _work(fn(img.convert("RGB"))).convert("RGB")
        out.putalpha(alpha)
        return out
    return _work(fn(img)).convert("RGB")


# ── Reading arguments ──────────────────────────────────────────────────

_NUM_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(%|px)?\s*$", re.IGNORECASE)


def _given(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip()) and value != []


def length(value: Any, ref: float, name: str, default: int | None = None, *,
           fractions: bool = True) -> int | None:
    """Pixels from 120, "120", "120px", "25%" (of *ref*) or 0.25 (a fraction of *ref*)."""
    if not _given(value):
        return default
    if isinstance(value, bool):
        raise ImageOpError(f"`{name}` must be pixels (e.g. 120) or a percentage (e.g. '25%').")
    if isinstance(value, (int, float)):
        v, unit = float(value), ""
    else:
        m = _NUM_RE.match(str(value))
        if not m:
            raise ImageOpError(f"`{name}` must be pixels (e.g. 120) or a percentage (e.g. '25%'), not {value!r}.")
        v, unit = float(m.group(1)), (m.group(2) or "").lower()
    if unit == "%":
        v = v / 100.0 * ref
    elif unit == "" and fractions and 0 < abs(v) < 1:
        v = v * ref
    if not math.isfinite(v):
        raise ImageOpError(f"`{name}` must be a finite number.")
    return int(round(v))


def number(value: Any, name: str, default: float | None = None, lo: float | None = None,
           hi: float | None = None, hint: str = "") -> float | None:
    """A number from 0.5, "0.5" or "50%" (= 0.5), within [lo, hi]."""
    if not _given(value):
        return default
    if isinstance(value, bool):
        raise ImageOpError(f"`{name}` must be a number.{hint}")
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        m = _NUM_RE.match(str(value))
        if not m or (m.group(2) or "").lower() == "px":
            raise ImageOpError(f"`{name}` must be a number, not {value!r}.{hint}")
        v = float(m.group(1)) / (100.0 if m.group(2) == "%" else 1.0)
    if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        span = f" between {lo:g} and {hi:g}" if lo is not None and hi is not None else (
            f" of at least {lo:g}" if lo is not None else f" of at most {hi:g}" if hi is not None else "")
        raise ImageOpError(f"`{name}` must be a number{span}, not {value!r}.{hint}")
    return v


def integer(value: Any, name: str, default: int | None = None, lo: int | None = None,
            hi: int | None = None) -> int | None:
    v = number(value, name, None, lo, hi)
    return default if v is None else int(round(v))


_FACTOR_HINT = " It is a factor: 1 = unchanged, 1.3 = 30% more, 0.7 = 30% less (or write '+30%' / '-30%')."


def factor(value: Any, name: str) -> float:
    """1.2, "120%" or "+20%" → 1.2; "-20%" → 0.8."""
    if isinstance(value, str):
        m = re.match(r"^\s*([+-])\s*(\d+(?:\.\d+)?)\s*%\s*$", value)
        if m:
            sign = 1 if m.group(1) == "+" else -1
            return max(0.0, 1.0 + sign * float(m.group(2)) / 100.0)
    v = number(value, name, None, 0, 10, _FACTOR_HINT)
    return 1.0 if v is None else v


def boolean(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y")
    return bool(value)


def colour(value: Any, name: str = "color", default: tuple[int, int, int, int] | None = None
           ) -> tuple[int, int, int, int] | None:
    """(r, g, b, a) from a name, hex, rgb()/hsl(), [r, g, b(, a)] or "transparent"."""
    if not _given(value):
        return default
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        try:
            vals = [max(0, min(255, int(round(float(v))))) for v in value]
        except (TypeError, ValueError):
            vals = []
        if vals:
            return tuple(vals + [255] * (4 - len(vals)))  # type: ignore[return-value]
    text = str(value).strip()
    if text.lower() in ("transparent", "none", "clear"):
        return (0, 0, 0, 0)
    if re.fullmatch(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}", text):
        text = "#" + text
    try:
        return tuple(ImageColor.getcolor(text, "RGBA"))  # type: ignore[return-value]
    except (ValueError, TypeError):
        raise ImageOpError(f"`{name}`: {value!r} isn't a colour — use a name (red), hex (#ff0000, or #ff000080 "
                           "with transparency), rgb(255, 0, 0) or 'transparent'.") from None


def _luma(c: tuple[int, ...]) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _contrasting(c: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (17, 17, 17, 255) if _luma(c) > 140 else (255, 255, 255, 255)


_ANCHORS: dict[str, tuple[float, float]] = {
    "top-left": (0.0, 0.0), "top": (0.5, 0.0), "top-right": (1.0, 0.0),
    "left": (0.0, 0.5), "center": (0.5, 0.5), "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0), "bottom": (0.5, 1.0), "bottom-right": (1.0, 1.0),
}
ANCHOR_NAMES = tuple(_ANCHORS)
_ANCHOR_WORDS = {"centre": "center", "middle": "center", "upper": "top", "lower": "bottom", "north": "top",
                 "south": "bottom", "west": "left", "east": "right", "c": "center", "t": "top", "b": "bottom",
                 "l": "left", "r": "right"}


def anchor(value: Any, name: str = "position", default: str = "center") -> tuple[float, float]:
    raw = str(value or default).strip().lower()
    for word in ("top", "bottom", "left", "right", "centre", "center", "middle", "upper", "lower"):
        raw = raw.replace(word, f" {word} ")
    words = [_ANCHOR_WORDS.get(w, w) for w in re.split(r"[\s_\-,/]+", raw) if w]
    unknown = [w for w in words if w not in ("top", "bottom", "left", "right", "center")]
    if unknown or not words:
        raise ImageOpError(f"`{name}` must be one of: {', '.join(ANCHOR_NAMES)}.")
    vertical = next((w for w in words if w in ("top", "bottom")), "")
    horizontal = next((w for w in words if w in ("left", "right")), "")
    key = "-".join(p for p in (vertical, horizontal) if p) or "center"
    return _ANCHORS[key]


def place(outer: tuple[int, int], inner: tuple[int, int], where: tuple[float, float], margin: int = 0
          ) -> tuple[int, int]:
    """Top-left corner of *inner* placed in *outer* at an anchor, *margin* from the edges."""
    (big_w, big_h), (w, h) = outer, inner
    fx, fy = where
    return (int(round(margin + fx * (big_w - w - 2 * margin))),
            int(round(margin + fy * (big_h - h - 2 * margin))))


def ratio(value: Any, name: str = "aspect") -> float:
    """'16:9', '4/5', '3x2', 'square' or 1.5 → width ÷ height."""
    text = str(value).strip().lower()
    if text == "square":
        return 1.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:/x×]\s*(\d+(?:\.\d+)?)", text)
    if m and float(m.group(1)) > 0 and float(m.group(2)) > 0:
        return float(m.group(1)) / float(m.group(2))
    try:
        v = float(text)
    except ValueError:
        v = 0.0
    if v > 0 and math.isfinite(v):
        return v
    raise ImageOpError(f"`{name}` must look like '16:9', '4:5', 'square' or 1.5, not {value!r}.")


def listish(value: Any, name: str) -> list[Any]:
    """A list from a list, a JSON list, or "a, b, c"."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
            except ValueError:
                raise ImageOpError(f"`{name}` isn't a valid list: {value!r}.") from None
            if isinstance(parsed, list):
                return parsed
        return [p.strip() for p in text.split(",") if p.strip()]
    return [value]


def points(value: Any, size: tuple[int, int], name: str = "points") -> list[tuple[int, int]]:
    """[[x, y], …], [x1, y1, x2, y2, …] or [{x, y}, …] → pixel points."""
    W, H = size
    items = listish(value, name)
    if items and all(not isinstance(i, (list, tuple, dict)) for i in items):
        if len(items) % 2:
            raise ImageOpError(f"`{name}` needs x, y pairs.")
        items = [items[i:i + 2] for i in range(0, len(items), 2)]
    out: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, dict):
            xy = (item.get("x"), item.get("y"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            xy = (item[0], item[1])
        else:
            raise ImageOpError(f"`{name}` must be a list of [x, y] points.")
        out.append((length(xy[0], W, name) or 0, length(xy[1], H, name) or 0))
    return out


def check_size(w: int, h: int, what: str = "The result") -> None:
    if w < 1 or h < 1:
        raise ImageOpError(f"{what} would be empty ({w}×{h} px).")
    if w * h > MAX_PIXELS:
        raise ImageOpError(f"{what} would be {w}×{h} px — over the {MAX_PIXELS // 1_000_000} megapixel limit. "
                           "Make it smaller: resize first, or ask for a smaller size, scale or cell.")


def region(op: dict[str, Any], img: Image.Image) -> tuple[int, int, int, int] | None:
    """The part of the picture an op works on (`box` or `region`), or None for all of it."""
    W, H = img.size
    box, reg = op.get("box"), op.get("region")
    if _given(box):
        b = listish(box, "box")
        if len(b) != 4:
            raise ImageOpError("`box` is [left, top, right, bottom].")
        left, top = length(b[0], W, "box") or 0, length(b[1], H, "box") or 0
        right, bottom = length(b[2], W, "box") or 0, length(b[3], H, "box") or 0
    elif isinstance(reg, dict) and reg:
        left = length(reg.get("x"), W, "region.x", 0) or 0
        top = length(reg.get("y"), H, "region.y", 0) or 0
        right = left + (length(reg.get("width"), W, "region.width", W - left) or 0)
        bottom = top + (length(reg.get("height"), H, "region.height", H - top) or 0)
    else:
        return None
    left, top, right, bottom = max(0, left), max(0, top), min(W, right), min(H, bottom)
    if right <= left or bottom <= top:
        raise ImageOpError(f"That region is empty or outside the picture ({W}×{H} px).")
    return left, top, right, bottom


_RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS, "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR, "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX, "hamming": Image.Resampling.HAMMING,
}


def _resample(value: Any) -> Image.Resampling:
    key = str(value or "lanczos").strip().lower()
    if key not in _RESAMPLE:
        raise ImageOpError(f"`resample` must be one of: {', '.join(_RESAMPLE)}.")
    return _RESAMPLE[key]


def _sizable(img: Image.Image) -> Image.Image:
    """A mode every resampling filter works on."""
    return _work(img) if img.mode in ("P", "1", "PA", "CMYK") else img


def _default_bg(img: Image.Image) -> tuple[int, int, int, int]:
    return (0, 0, 0, 0) if has_alpha(img) else (255, 255, 255, 255)


def _canvas(size: tuple[int, int], fill: tuple[int, int, int, int], like: Image.Image | None = None
            ) -> Image.Image:
    """A blank picture: RGBA when the fill (or what goes on it) is see-through."""
    check_size(*size)
    if fill[3] < 255 or (like is not None and has_alpha(like)):
        return Image.new("RGBA", size, fill)
    return Image.new("RGB", size, fill[:3])


def over(canvas: Image.Image, img: Image.Image, xy: tuple[int, int]) -> Image.Image:
    """*img* laid on *canvas* at *xy* (clipped to it), transparency respected. Changes *canvas*."""
    x, y = xy
    W, H = canvas.size
    w, h = img.size
    left, top, right, bottom = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if right <= left or bottom <= top:
        return canvas
    piece = _rgba(img).crop((left - x, top - y, right - x, bottom - y))
    if canvas.mode == "RGBA":
        canvas.paste(Image.alpha_composite(canvas.crop((left, top, right, bottom)), piece), (left, top))
    else:
        canvas.paste(piece, (left, top), piece)
    return canvas


def _with_layer(base: Image.Image, layer: Image.Image) -> Image.Image:
    """A full-size RGBA *layer* composited onto (a copy of) *base*."""
    base = _work(base)
    if base.mode == "RGBA":
        return Image.alpha_composite(base, layer)
    out = base.copy()
    out.paste(layer, (0, 0), layer)
    return out


def _fade(img: Image.Image, opacity: float) -> Image.Image:
    """*img* (RGBA) with its transparency multiplied by *opacity*."""
    if opacity >= 1:
        return img
    out = _rgba(img).copy()
    out.putalpha(out.getchannel("A").point(lambda a: int(round(a * opacity))))
    return out


def _mask(size: tuple[int, int], *, radius: int = 0, ellipse: bool = False) -> Image.Image:
    """An anti-aliased white rounded rectangle (or ellipse) on black, for a cut-out."""
    W, H = size
    scale = 4 if W * H <= 4_000_000 else 2 if W * H <= 16_000_000 else 1
    big = Image.new("L", (W * scale, H * scale), 0)
    draw = ImageDraw.Draw(big)
    if ellipse:
        draw.ellipse((0, 0, W * scale - 1, H * scale - 1), fill=255)
    else:
        draw.rounded_rectangle((0, 0, W * scale - 1, H * scale - 1), radius=radius * scale, fill=255)
    return big.resize((W, H), Image.Resampling.LANCZOS) if scale > 1 else big


def _cut_out(img: Image.Image, mask: Image.Image) -> Image.Image:
    out = _rgba(img).copy()
    out.putalpha(ImageChops.multiply(out.getchannel("A"), mask))
    return out


# ── Fonts ──────────────────────────────────────────────────────────────

_FONT_DIRS = (
    "/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts",
    "/System/Library/Fonts", "/Library/Fonts", "~/Library/Fonts",
    "C:/Windows/Fonts", "%LOCALAPPDATA%/Microsoft/Windows/Fonts", "%WINDIR%/Fonts",
)
_FONT_SUFFIXES = frozenset({".ttf", ".otf", ".ttc"})
_FONT_SCAN_LIMIT = 40_000
# Everyday sans fonts with wide Latin coverage (č ć đ š ž ü ß …), in the order
# they are tried: Linux, macOS, Windows, then common extras.
_REGULAR_FONTS = (
    "dejavusans", "arial", "helvetica", "liberationsans", "liberationsansregular", "notosans",
    "notosansregular", "freesans", "segoeui", "verdana", "roboto", "robotoregular", "opensans",
    "opensansregular", "inter", "interregular", "ubuntu", "ubunturegular", "cantarellregular",
    "arialunicode", "arialunicodems",
)
_BOLD_FONTS = (
    "dejavusansbold", "arialbold", "arialbd", "helveticabold", "liberationsansbold", "notosansbold",
    "freesansbold", "segoeuib", "verdanabold", "verdanab", "robotobold", "opensansbold", "interbold",
    "ubuntubold", "cantarellbold",
)
# Fonts tried last for scripts the everyday ones lack (CJK and the like).
_WIDE_FONT_WORDS = ("cjk", "unicode", "fallback", "unifont", "wqy", "symbol", "arabic", "hebrew", "devanagari",
                    "thai", "hangul", "gothic")


def _font_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@functools.lru_cache(maxsize=1)
def font_index() -> dict[str, str]:
    """Installed font files by squashed name ("DejaVuSans-Bold.ttf" → "dejavusansbold")."""
    index: dict[str, str] = {}
    seen = 0
    for raw in _FONT_DIRS:
        root = Path(os.path.expandvars(os.path.expanduser(raw)))
        if "%" in str(root) or not root.is_dir():
            continue
        for dirpath, _dirs, filenames in os.walk(root):
            for filename in filenames:
                seen += 1
                if seen > _FONT_SCAN_LIMIT:
                    return index
                path = Path(dirpath) / filename
                if path.suffix.lower() in _FONT_SUFFIXES:
                    index.setdefault(_font_key(path.stem), str(path))
    return index


def find_font(name: str | None, *, bold: bool = False, italic: bool = False) -> str | None:
    """The installed file for a font name ("DejaVu Sans", "Arial", "Georgia Bold"), or the default."""
    index = font_index()
    if not name:
        for key in (_BOLD_FONTS if bold else ()) + _REGULAR_FONTS:
            if key in index:
                return index[key]
        return None
    key = _font_key(name)
    style = ("bold" if bold else "") + ("italic" if italic else "")
    tries = []
    if style:
        tries += [key + style, key + ("bd" if bold and not italic else "i"), key + ("boldoblique" if bold and italic
                  else "bold" if bold else "oblique")]
    tries += [key, key + "regular", key + "book", key + "roman", key + "mt"]
    for t in tries:
        if t in index:
            return index[t]
    family = sorted((k for k in index if k.startswith(key)), key=len)
    wanted = [w for w, on in (("bold", bold), ("italic", italic)) if on]
    styled = [k for k in family if all(w in k or (w == "italic" and "oblique" in k) for w in wanted)]
    plain = [k for k in family if not any(w in k for w in ("bold", "italic", "oblique", "light", "thin", "black"))]
    pick = (styled or plain or family)
    return index[pick[0]] if pick else None


def _missing_glyphs(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> set[str]:
    """Characters of *text* the font draws as its "missing" box."""
    chars = {c for c in text if not c.isspace()}
    try:
        empty = font.getmask("\uffff")
        signature = (empty.size, bytes(empty))
    except Exception:
        return set()
    missing = set()
    for ch in chars:
        try:
            mask = font.getmask(ch)
        except Exception:
            missing.add(ch)
            continue
        if (mask.size, bytes(mask)) == signature:
            missing.add(ch)
    return missing


class Fonts:
    """Fonts for text: the one asked for, else an installed one that has every character."""

    def __init__(self, resolve_file: Callable[[str], Path | None] | None = None) -> None:
        # The tool resolves a font given as a file path (and keeps it inside the agent's files).
        self._resolve_file = resolve_file
        self._loaded: dict[tuple[str, int], Any] = {}

    def _load(self, path: str, size: int) -> Any | None:
        key = (path, size)
        if key not in self._loaded:
            try:
                self._loaded[key] = ImageFont.truetype(path, size)
            except (OSError, ValueError):
                self._loaded[key] = None
        return self._loaded[key]

    def get(self, name: Any, size: int, *, bold: bool = False, italic: bool = False, text: str = ""
            ) -> tuple[Any, str]:
        """(font, note) — the note says when the font isn't the one asked for."""
        asked = str(name or "").strip()
        first: str | None = None
        note = ""
        if asked:
            looks_like_file = any(s in asked for s in ("/", "\\")) or Path(asked).suffix.lower() in _FONT_SUFFIXES
            if looks_like_file:
                found = self._resolve_file(asked) if self._resolve_file else None
                first = str(found) if found else None
            else:
                first = find_font(asked, bold=bold, italic=italic)
            if first is None:
                note = f"font {asked!r} isn't available here"
        candidates: list[str] = [p for p in [first, find_font(None, bold=bold)] if p]
        index = font_index()
        for key in (_BOLD_FONTS if bold else ()) + _REGULAR_FONTS:
            if key in index:
                candidates.append(index[key])
        candidates += [p for k, p in sorted(index.items()) if any(w in k for w in _WIDE_FONT_WORDS)
                       and "emoji" not in k]
        best: tuple[int, Any, str] | None = None
        for path in dict.fromkeys(candidates):
            font = self._load(path, size)
            if font is None:
                continue
            missing = _missing_glyphs(font, text)
            if not missing:
                if first and path != first and not note:
                    note = f"font {asked!r} lacks some of these characters"
                if note:
                    note += f"; used {Path(path).stem}"
                return font, note
            if best is None or len(missing) < best[0]:
                best = (len(missing), font, path)
        if best is not None:
            gap = "".join(sorted(_missing_glyphs(best[1], text)))[:20]
            return best[1], (note + "; " if note else "") + f"no installed font has every character (missing: {gap})"
        font = ImageFont.load_default(size=size)
        gap = "".join(sorted(_missing_glyphs(font, text)))[:20]
        tail = f"; missing characters: {gap}" if gap else ""
        return font, (note + "; " if note else "") + "no fonts are installed, used Pillow's built-in font" + tail


@dataclass
class Context:
    """What operations share while they run: fonts, and notes for the result."""

    fonts: Fonts = field(default_factory=Fonts)
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if text and text not in self.notes:
            self.notes.append(text)


# ── Geometry ───────────────────────────────────────────────────────────


def _op_resize(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    img = _sizable(img)
    fit = str(op.get("fit") or "contain").strip().lower()
    resample = _resample(op.get("resample"))
    scale = number(op.get("scale"), "scale", None, 0.0001, 100)
    w = length(op.get("width"), W, "width")
    h = length(op.get("height"), H, "height")
    if scale is not None:
        tw, th = max(1, round(W * scale)), max(1, round(H * scale))
    elif w and h:
        if fit in ("fill", "stretch", "exact", "ignore"):
            tw, th = w, h
        elif fit in ("cover", "crop"):
            if boolean(op.get("shrink_only")) and W <= w and H <= h:
                return img
            f = max(w / W, h / H)
            rw, rh = max(w, round(W * f)), max(h, round(H * f))
            check_size(rw, rh)
            big = img.resize((rw, rh), resample)
            x, y = place((rw, rh), (w, h), anchor(op.get("gravity"), "gravity"))
            return big.crop((x, y, x + w, y + h))
        elif fit in ("pad", "letterbox"):
            f = min(w / W, h / H)
            if boolean(op.get("shrink_only")):
                f = min(f, 1.0)
            inner = img.resize((max(1, round(W * f)), max(1, round(H * f))), resample)
            fill = colour(op.get("background"), "background", None) or _default_bg(img)
            canvas = _canvas((w, h), fill, inner)
            return over(canvas, inner, place((w, h), inner.size, anchor(op.get("gravity"), "gravity")))
        elif fit in ("contain", "inside", "fit"):
            f = min(w / W, h / H)
            tw, th = max(1, round(W * f)), max(1, round(H * f))
        else:
            raise ImageOpError("`fit` must be contain (keep proportions, inside the box), cover (fill and crop), "
                               "fill (stretch) or pad (fit and fill the rest with background).")
    elif w:
        tw, th = w, max(1, round(H * w / W))
    elif h:
        tw, th = max(1, round(W * h / H)), h
    else:
        raise ImageOpError("resize needs width, height or scale.")
    if boolean(op.get("shrink_only")) and tw >= W and th >= H:
        return img
    if (tw, th) == (W, H):
        return img
    check_size(tw, th)
    return img.resize((tw, th), resample)


def _op_crop(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    if _given(op.get("aspect")):
        r = ratio(op["aspect"])
        cw, ch = (round(H * r), H) if W / H > r else (W, round(W / r))
        cw, ch = max(1, min(cw, W)), max(1, min(ch, H))
        x, y = place((W, H), (cw, ch), anchor(op.get("gravity"), "gravity"))
        box = (x, y, x + cw, y + ch)
    elif _given(op.get("box")) or _given(op.get("region")):
        box = region(op, img)  # type: ignore[assignment]
    elif any(_given(op.get(k)) for k in ("x", "y", "width", "height")):
        x = length(op.get("x"), W, "x", 0) or 0
        y = length(op.get("y"), H, "y", 0) or 0
        box = (x, y, x + (length(op.get("width"), W, "width", W - x) or 0),
               y + (length(op.get("height"), H, "height", H - y) or 0))
    elif any(_given(op.get(k)) for k in ("top", "right", "bottom", "left")):
        box = (length(op.get("left"), W, "left", 0) or 0, length(op.get("top"), H, "top", 0) or 0,
               W - (length(op.get("right"), W, "right", 0) or 0), H - (length(op.get("bottom"), H, "bottom", 0) or 0))
    else:
        raise ImageOpError("crop needs x/y/width/height, box [left, top, right, bottom], aspect (e.g. '16:9'), "
                           "or top/right/bottom/left to cut off.")
    left, top, right, bottom = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if right <= left or bottom <= top:
        raise ImageOpError(f"That crop is empty or outside the picture ({W}×{H} px).")
    return img.crop((left, top, right, bottom))


def _op_rotate(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    angle = number(op.get("angle", op.get("degrees")), "angle", 90.0)
    turn = angle % 360  # type: ignore[operator]
    if turn == 0:
        return img
    if turn in (90, 180, 270):
        return img.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                              270: Image.Transpose.ROTATE_90}[int(turn)])
    expand = boolean(op.get("expand"), True)
    fill = colour(op.get("background"), "background", None) or (0, 0, 0, 0)
    work = _rgba(img) if fill[3] < 255 else _work(img)
    out = work.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=expand,  # type: ignore[operator]
                      fillcolor=fill if work.mode == "RGBA" else fill[:3])
    check_size(*out.size)
    return out


def _op_flip(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    way = str(op.get("direction") or "horizontal").strip().lower()
    if way in ("horizontal", "h", "x", "left-right", "left_right", "mirror"):
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if way in ("vertical", "v", "y", "top-bottom", "top_bottom", "upside-down"):
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if way == "both":
        return img.transpose(Image.Transpose.ROTATE_180)
    raise ImageOpError("flip `direction` is horizontal (mirror), vertical or both.")


def _op_pad(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    fill = colour(op.get("background", op.get("color")), "background", None) or _default_bg(img)
    where = anchor(op.get("gravity"), "gravity")
    if _given(op.get("aspect")):
        r = ratio(op["aspect"])
        cw, ch = (round(H * r), H) if W / H < r else (W, round(W / r))
    elif _given(op.get("width")) or _given(op.get("height")):
        cw = length(op.get("width"), W, "width", W) or W
        ch = length(op.get("height"), H, "height", H) or H
        if cw < W or ch < H:
            f = min(cw / W, ch / H)
            img = _sizable(img).resize((max(1, round(W * f)), max(1, round(H * f))), Image.Resampling.LANCZOS)
            ctx.note(f"pad: the picture was larger than {cw}×{ch}, so it was shrunk to fit")
    else:
        size = length(op.get("size"), min(W, H), "size", 0) or 0
        left = length(op.get("left"), W, "left", size) or 0
        right = length(op.get("right"), W, "right", size) or 0
        top = length(op.get("top"), H, "top", size) or 0
        bottom = length(op.get("bottom"), H, "bottom", size) or 0
        if not (left or right or top or bottom):
            raise ImageOpError("pad needs size (all sides), top/right/bottom/left, aspect (e.g. '1:1') "
                               "or width/height of the new canvas.")
        canvas = _canvas((W + left + right, H + top + bottom), fill, img)
        return over(canvas, img, (left, top))
    canvas = _canvas((cw, ch), fill, img)
    return over(canvas, img, place((cw, ch), img.size, where))


def _op_border(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    size = length(op.get("size", op.get("width")), min(W, H), "size", max(1, round(min(W, H) * 0.02))) or 1
    fill = colour(op.get("color"), "color", (0, 0, 0, 255)) or (0, 0, 0, 255)
    canvas = _canvas((W + 2 * size, H + 2 * size), fill, img)
    return over(canvas, img, (size, size))


def _op_trim(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    tol = number(op.get("tolerance"), "tolerance", 10, 0, 255) or 0
    work = _work(img)
    if work.mode == "RGBA" and not _given(op.get("color")) and work.getchannel("A").getextrema()[0] < 255:
        bbox = work.getchannel("A").point(lambda a: 255 if a > tol else 0).getbbox()
    else:
        bg = colour(op.get("color"), "color", None) or _rgba(work).getpixel((0, 0))
        diff = ImageChops.difference(work.convert("RGB"), Image.new("RGB", work.size, bg[:3]))
        r, g, b = diff.split()
        bbox = ImageChops.lighter(ImageChops.lighter(r, g), b).point(lambda v: 255 if v > tol else 0).getbbox()
    if not bbox:
        ctx.note("trim: the whole picture is background — nothing was trimmed")
        return img
    keep = length(op.get("margin"), min(W, H), "margin", 0) or 0
    left, top, right, bottom = bbox
    return img.crop((max(0, left - keep), max(0, top - keep), min(W, right + keep), min(H, bottom + keep)))


def _op_round_corners(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    radius = length(op.get("radius", op.get("size")), min(W, H), "radius", round(min(W, H) * 0.08)) or 0
    radius = max(0, min(radius, min(W, H) // 2))
    return _cut_out(img, _mask((W, H), radius=radius))


def _op_circle(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    side = min(W, H)
    x, y = place((W, H), (side, side), anchor(op.get("gravity"), "gravity"))
    square = img.crop((x, y, x + side, y + side))
    return _cut_out(square, _mask((side, side), ellipse=True))


def _perspective_coeffs(dst: list[tuple[float, float]], src: list[tuple[float, float]]) -> tuple[float, ...]:
    """Pillow's PERSPECTIVE data: for each output point, where it comes from in the input."""
    import numpy as np

    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    try:
        solution = np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float))
    except np.linalg.LinAlgError:
        raise ImageOpError("Those corners don't make a four-sided shape (three of them are in a line).") from None
    return tuple(float(c) for c in solution)


def _op_perspective(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    quad = points(op.get("corners", op.get("points")), (W, H), "corners")
    if len(quad) != 4:
        raise ImageOpError("perspective needs `corners`: the four corners of the area to straighten, in order "
                           "top-left, top-right, bottom-right, bottom-left, as [[x, y], …].")
    tl, tr, br, bl = quad
    w = length(op.get("width"), W, "width") or round(max(math.dist(tl, tr), math.dist(bl, br)))
    h = length(op.get("height"), H, "height") or round(max(math.dist(tl, bl), math.dist(tr, br)))
    check_size(w, h)
    coeffs = _perspective_coeffs([(0, 0), (w, 0), (w, h), (0, h)], [tl, tr, br, bl])
    work = _work(img)
    return work.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


# ── Tone and colour ────────────────────────────────────────────────────


def _op_adjust(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    out = img
    changed = False
    for key, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast),
                          ("saturation", ImageEnhance.Color), ("sharpness", ImageEnhance.Sharpness)):
        if _given(op.get(key)):
            changed = True
            f = factor(op[key], key)
            if f != 1:
                out = _on_rgb(out, lambda im, e=enhancer, f=f: e(im).enhance(f))
    if _given(op.get("gamma")):
        changed = True
        g = number(op["gamma"], "gamma", 1.0, 0.05, 20,
                   " Above 1 brightens the mid-tones, below 1 darkens them.") or 1.0
        lut = [min(255, round(255 * (i / 255) ** (1 / g))) for i in range(256)] * 3
        out = _on_rgb(out, lambda im: im.point(lut))
    if _given(op.get("hue")):
        changed = True
        degrees = number(op["hue"], "hue", 0.0, -360, 360, " Degrees around the colour wheel.") or 0.0
        shift = int(round(degrees / 360 * 256)) % 256
        if shift:
            def rotate_hue(im: Image.Image) -> Image.Image:
                hh, ss, vv = im.convert("HSV").split()
                return Image.merge("HSV", (hh.point(lambda v: (v + shift) % 256), ss, vv)).convert("RGB")

            out = _on_rgb(out, rotate_hue)
    if _given(op.get("warmth")):
        changed = True
        warm = number(op["warmth"], "warmth", 0.0, -1, 1, " -1 is cool (blue), 1 is warm (orange).") or 0.0
        red = [min(255, round(i * (1 + 0.25 * warm))) for i in range(256)]
        blue = [min(255, round(i * (1 - 0.25 * warm))) for i in range(256)]
        out = _on_rgb(out, lambda im: im.point(red + list(range(256)) + blue))
    if not changed:
        raise ImageOpError("adjust needs one of brightness, contrast, saturation, sharpness (factors: 1 = unchanged), "
                           "gamma, hue (degrees) or warmth (-1 … 1).")
    return out


def _amount(op: dict, default: float = 1.0) -> float:
    return number(op.get("amount"), "amount", default, 0, 1) or 0.0


def _blend_with(img: Image.Image, fn: Callable[[Image.Image], Image.Image], amount: float) -> Image.Image:
    if amount >= 1:
        return _on_rgb(img, fn)
    return _on_rgb(img, lambda im: Image.blend(im, _work(fn(im)).convert("RGB"), amount))


def _op_grayscale(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    amount = _amount(op)
    if amount < 1:
        return _blend_with(img, lambda im: im.convert("L").convert("RGB"), amount)
    return _work(img).convert("LA" if has_alpha(img) else "L")


def _op_sepia(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    return _blend_with(img, lambda im: ImageOps.colorize(im.convert("L"), "#2b1d0e", "#fff1d6", mid="#a5845c"),
                       _amount(op))


def _op_invert(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    return _on_rgb(img, ImageOps.invert)


def _op_equalize(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    return _on_rgb(img, ImageOps.equalize)


def _op_autocontrast(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    cutoff = number(op.get("cutoff"), "cutoff", 1.0, 0, 49) or 0.0
    return _on_rgb(img, lambda im: ImageOps.autocontrast(im, cutoff=cutoff))


def _op_posterize(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    bits = integer(op.get("bits"), "bits", 3, 1, 8) or 3
    return _on_rgb(img, lambda im: ImageOps.posterize(im, bits))


def _op_solarize(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    level = integer(op.get("threshold"), "threshold", 128, 0, 255)
    return _on_rgb(img, lambda im: ImageOps.solarize(im, level))


def _op_threshold(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    level = integer(op.get("threshold", op.get("level")), "threshold", 128, 0, 255) or 0
    work = _work(img)
    bw = work.convert("L").point(lambda v: 255 if v >= level else 0)
    if work.mode == "RGBA":
        bw = Image.merge("LA", (bw, work.getchannel("A")))
    return bw


def _op_tint(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    c = colour(op.get("color"), "color")
    if c is None:
        raise ImageOpError("tint needs `color`.")
    amount = number(op.get("amount"), "amount", 0.3, 0, 1) or 0.0
    return _on_rgb(img, lambda im: Image.blend(im, Image.new("RGB", im.size, c[:3]), amount * c[3] / 255))


def _op_duotone(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    stops = [colour(c, "colors") for c in listish(op.get("colors"), "colors")]
    if len(stops) not in (2, 3):
        raise ImageOpError("duotone needs `colors`: [dark, light] (or [dark, middle, light]).")
    dark, light = stops[0][:3], stops[-1][:3]  # type: ignore[index]
    mid = stops[1][:3] if len(stops) == 3 else None  # type: ignore[index]
    return _blend_with(img, lambda im: ImageOps.colorize(im.convert("L"), dark, light, mid=mid), _amount(op))


def _op_quantize(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    n = integer(op.get("colors"), "colors", 16, 2, 256) or 16
    work = _work(img)
    method = Image.Quantize.FASTOCTREE if work.mode == "RGBA" else Image.Quantize.MEDIANCUT
    dither = Image.Dither.FLOYDSTEINBERG if boolean(op.get("dither"), True) else Image.Dither.NONE
    return work.quantize(colors=n, method=method, dither=dither)


def _op_opacity(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    value = op.get("opacity") if _given(op.get("opacity")) else op.get("amount")
    return _fade(img, number(value, "opacity", 0.5, 0, 1) or 0.0)


def _connected_to_edges(mask: Any) -> Any:
    """The part of a boolean mask (H × W) reachable from the picture's edges.

    Exact with scipy. Without it, Pillow's flood fill runs on a copy at most
    ~600 px across, grown by one cell so the rim along the subject isn't left
    behind, and is then limited to the full-size mask again."""
    import numpy as np

    try:
        from scipy import ndimage  # type: ignore[import-untyped]
    except Exception:
        ndimage = None
    if ndimage is not None:
        labels, _count = ndimage.label(mask)
        edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        return np.isin(labels, np.unique(edge[edge > 0]))
    H, W = mask.shape
    step = max(1, math.ceil(max(W, H) / 600))
    small = Image.fromarray(mask[::step, ::step].astype(np.uint8) * 255, "L").copy()  # fromarray is read-only
    sw, sh = small.size
    px = small.load()
    border = [(x, 0) for x in range(sw)] + [(x, sh - 1) for x in range(sw)]
    border += [(0, y) for y in range(sh)] + [(sw - 1, y) for y in range(sh)]
    for xy in border:
        if px[xy] == 255:
            ImageDraw.floodfill(small, xy, 128)
    reached = small.point(lambda v: 255 if v == 128 else 0)
    if step > 1:
        reached = reached.filter(ImageFilter.MaxFilter(3))
    grown = np.asarray(reached) > 0
    if step > 1:
        grown = np.repeat(np.repeat(grown, step, axis=0), step, axis=1)[:H, :W]
    return grown & mask


def _op_transparent(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    import numpy as np

    work = _rgba(img)
    arr = np.asarray(work).astype(np.int16)
    target = colour(op.get("color"), "color", None) or tuple(int(v) for v in arr[0, 0])
    tol = number(op.get("tolerance"), "tolerance", 30, 0, 255) or 0
    close = np.abs(arr[..., :3] - np.array(target[:3], dtype=np.int16)).max(axis=2) <= tol
    if not boolean(op.get("everywhere")):
        close = _connected_to_edges(close)
    alpha = arr[..., 3].copy()
    alpha[close] = 0
    share = float(close.mean()) * 100
    if share == 0:
        ctx.note("transparent: no pixels matched that colour — raise tolerance or pass color")
    elif share > 95:
        ctx.note(f"transparent: {share:.0f}% of the picture became see-through — lower tolerance?")
    out = np.dstack([arr[..., :3], alpha]).astype(np.uint8)
    return Image.fromarray(out, "RGBA").copy()


def _op_flatten(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    fill = colour(op.get("background", op.get("color")), "background", (255, 255, 255, 255)) or (255, 255, 255, 255)
    work = _rgba(img)
    out = Image.new("RGB", work.size, fill[:3])
    out.paste(work, (0, 0), work)
    return out


_MODES = ("RGB", "RGBA", "L", "LA", "CMYK", "P", "1")


def _op_mode(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    to = str(op.get("to") or op.get("mode") or "").strip().upper()
    to = {"GRAY": "L", "GREY": "L", "GRAYSCALE": "L", "BW": "1", "PALETTE": "P", "BITMAP": "1"}.get(to, to)
    if to not in _MODES:
        raise ImageOpError(f"mode `to` must be one of: {', '.join(_MODES)}.")
    work = _work(img)
    if to == "P":
        method = Image.Quantize.FASTOCTREE if work.mode == "RGBA" else Image.Quantize.MEDIANCUT
        return work.quantize(colors=integer(op.get("colors"), "colors", 256, 2, 256) or 256, method=method)
    if to == "1":
        dither = Image.Dither.FLOYDSTEINBERG if boolean(op.get("dither"), True) else Image.Dither.NONE
        return work.convert("L").convert("1", dither=dither)
    return work.convert(to)


def _op_vignette(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    import numpy as np

    amount = _amount(op, 0.5)
    shade = colour(op.get("color"), "color", (0, 0, 0, 255)) or (0, 0, 0, 255)
    W, H = img.size
    yy, xx = np.ogrid[0:H, 0:W]
    dx = (xx - (W - 1) / 2) / (W / 2)
    dy = (yy - (H - 1) / 2) / (H / 2)
    distance = np.sqrt(dx * dx + dy * dy) / math.sqrt(2)
    strength = np.clip((distance - 0.35) / 0.65, 0, 1) ** 1.6 * amount
    mask = Image.fromarray((strength * 255).astype(np.uint8), "L")
    return _on_rgb(img, lambda im: Image.composite(Image.new("RGB", im.size, shade[:3]), im, mask))


def _op_noise(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    import numpy as np

    amount = _amount(op, 0.1)
    seed = integer(op.get("seed"), "seed", None)
    rng = np.random.default_rng(seed)

    def grain(im: Image.Image) -> Image.Image:
        arr = np.asarray(im).astype(np.float32)
        noise = rng.normal(0.0, amount * 64.0, arr.shape[:2])[..., None]
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), "RGB").copy()

    return _on_rgb(img, grain)


def _op_gradient(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    import numpy as np

    stops = [colour(c, "colors") for c in listish(op.get("colors"), "colors")]
    if len(stops) < 2:
        raise ImageOpError("gradient needs `colors`: at least two, e.g. ['#00000000', '#000000cc'] to darken "
                           "towards the bottom.")
    W, H = img.size
    way = str(op.get("direction") or "vertical").strip().lower().replace("_", "-").replace(" ", "-")
    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    if way in ("vertical", "down", "to-bottom", "top-bottom", "top-to-bottom"):
        t = np.broadcast_to(ys, (H, W))
    elif way in ("up", "to-top", "bottom-top", "bottom-to-top"):
        t = np.broadcast_to(1 - ys, (H, W))
    elif way in ("horizontal", "right", "to-right", "left-right", "left-to-right"):
        t = np.broadcast_to(xs, (H, W))
    elif way in ("left", "to-left", "right-left", "right-to-left"):
        t = np.broadcast_to(1 - xs, (H, W))
    elif way in ("diagonal", "to-bottom-right"):
        t = (xs + ys) / 2
    elif way == "radial":
        t = np.clip(np.sqrt((xs * 2 - 1) ** 2 + (ys * 2 - 1) ** 2) / math.sqrt(2), 0, 1)
    else:
        raise ImageOpError("gradient `direction` is vertical (top to bottom), up, horizontal (left to right), "
                           "left, diagonal or radial (centre outwards).")
    positions = np.linspace(0.0, 1.0, len(stops))
    channels = [np.interp(t, positions, [c[i] for c in stops]) for i in range(4)]  # type: ignore[index]
    layer = Image.fromarray(np.dstack(channels).astype(np.uint8), "RGBA")
    layer = _fade(layer, number(op.get("opacity"), "opacity", 1.0, 0, 1) or 0.0)
    return _with_layer(img, layer)


# ── Filters ────────────────────────────────────────────────────────────


def _op_blur(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    radius = length(op.get("radius", op.get("amount")), min(W, H), "radius", 4, fractions=False) or 0
    radius = max(0, min(radius, 500))
    kind = str(op.get("kind") or "gaussian").strip().lower()
    filt = ImageFilter.BoxBlur(radius) if kind == "box" else ImageFilter.GaussianBlur(radius)
    return _work(img).filter(filt)


def _op_sharpen(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    amount = number(op.get("amount"), "amount", 1.0, 0, 10) or 0.0
    radius = number(op.get("radius"), "radius", 2.0, 0.1, 50) or 2.0
    threshold = integer(op.get("threshold"), "threshold", 3, 0, 255) or 0
    unsharp = ImageFilter.UnsharpMask(radius=radius, percent=int(round(150 * amount)), threshold=threshold)
    return _on_rgb(img, lambda im: im.filter(unsharp))


def _op_median(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    size = integer(op.get("size"), "size", 3, 3, 15) or 3
    size += 1 - size % 2
    return _on_rgb(img, lambda im: im.filter(ImageFilter.MedianFilter(size)))


def _op_pixelate(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    W, H = img.size
    block = length(op.get("size", op.get("amount")), min(W, H), "size", max(4, round(min(W, H) / 60)),
                   fractions=False) or 4
    block = max(2, block)
    work = _work(img)
    small = work.resize((max(1, math.ceil(W / block)), max(1, math.ceil(H / block))), Image.Resampling.BOX)
    return small.resize((W, H), Image.Resampling.NEAREST)


def _simple_filter(filt: Any) -> Callable[[Image.Image, dict, Context], Image.Image]:
    def run(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
        return _on_rgb(img, lambda im: im.filter(filt))
    return run


# ── Drawing ────────────────────────────────────────────────────────────


def _line_width(op: dict, img: Image.Image) -> int:
    ref = min(img.size)
    return max(1, length(op.get("line_width"), ref, "line_width", max(2, round(ref / 250)), fractions=False) or 1)


def _shape_box(op: dict, img: Image.Image) -> tuple[int, int, int, int]:
    W, H = img.size
    if _given(op.get("box")) or _given(op.get("region")):
        b = region(op, img)
        assert b is not None
        return b
    if not (_given(op.get("width")) and _given(op.get("height"))):
        raise ImageOpError("Give the shape x, y, width and height (or box [left, top, right, bottom]).")
    x = length(op.get("x"), W, "x", 0) or 0
    y = length(op.get("y"), H, "y", 0) or 0
    return x, y, x + (length(op.get("width"), W, "width") or 0), y + (length(op.get("height"), H, "height") or 0)


def _paint(img: Image.Image, op: dict, draw_fn: Callable[[ImageDraw.ImageDraw], None]) -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    return _with_layer(img, _fade(layer, number(op.get("opacity"), "opacity", 1.0, 0, 1) or 0.0))


def _op_rectangle(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    left, top, right, bottom = _shape_box(op, img)
    fill = colour(op.get("fill"), "fill")
    outline = colour(op.get("color"), "color", None if fill else (230, 30, 30, 255))
    width = _line_width(op, img)
    radius = length(op.get("radius"), min(right - left, bottom - top), "radius", 0) or 0

    def draw(d: ImageDraw.ImageDraw) -> None:
        box = (left, top, max(left, right - 1), max(top, bottom - 1))
        if radius:
            d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        else:
            d.rectangle(box, fill=fill, outline=outline, width=width)

    return _paint(img, op, draw)


def _op_ellipse(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    left, top, right, bottom = _shape_box(op, img)
    fill = colour(op.get("fill"), "fill")
    outline = colour(op.get("color"), "color", None if fill else (230, 30, 30, 255))
    width = _line_width(op, img)
    return _paint(img, op, lambda d: d.ellipse((left, top, max(left, right - 1), max(top, bottom - 1)),
                                               fill=fill, outline=outline, width=width))


def _op_line(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    pts = points(op.get("points"), img.size)
    if len(pts) < 2:
        raise ImageOpError("line needs `points`: at least two [x, y].")
    c = colour(op.get("color"), "color", (230, 30, 30, 255))
    width = _line_width(op, img)
    return _paint(img, op, lambda d: d.line(pts, fill=c, width=width, joint="curve"))


def _op_arrow(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    pts = points(op.get("points"), img.size)
    if len(pts) < 2:
        raise ImageOpError("arrow needs `points`: from [x, y] to [x, y] (more points bend it; the head is at the last).")
    c = colour(op.get("color"), "color", (230, 30, 30, 255))
    width = _line_width(op, img)
    head = length(op.get("head", op.get("size")), min(img.size), "head", width * 4 + 6, fractions=False) or 10
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    spread = math.pi / 7
    tip = (x2, y2)
    left = (x2 - head * math.cos(angle - spread), y2 - head * math.sin(angle - spread))
    right = (x2 - head * math.cos(angle + spread), y2 - head * math.sin(angle + spread))
    neck = (x2 - head * 0.8 * math.cos(angle), y2 - head * 0.8 * math.sin(angle))

    def draw(d: ImageDraw.ImageDraw) -> None:
        d.line([*pts[:-1], neck], fill=c, width=width, joint="curve")
        d.polygon([tip, left, right], fill=c)

    return _paint(img, op, draw)


def _op_polygon(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    pts = points(op.get("points"), img.size)
    if len(pts) < 3:
        raise ImageOpError("polygon needs `points`: at least three [x, y].")
    fill = colour(op.get("fill"), "fill")
    outline = colour(op.get("color"), "color", None if fill else (230, 30, 30, 255))
    width = _line_width(op, img)
    return _paint(img, op, lambda d: d.polygon(pts, fill=fill, outline=outline, width=width))


def _wrap(text: str, font: Any, max_width: int) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}" if line else word
            if not line or font.getlength(trial) <= max_width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return "\n".join(lines)


def _rotated_size(w: int, h: int, angle: float) -> tuple[int, int]:
    a = math.radians(angle)
    return (int(math.ceil(abs(w * math.cos(a)) + abs(h * math.sin(a)))),
            int(math.ceil(abs(w * math.sin(a)) + abs(h * math.cos(a)))))


def _tile(base: Image.Image, piece: Image.Image, gap: int) -> Image.Image:
    """*piece* repeated over the whole of *base*, rows offset like bricks."""
    W, H = base.size
    pw, ph = piece.size
    step_x, step_y = pw + gap, ph + gap
    if (W // max(1, step_x) + 2) * (H // max(1, step_y) + 2) > 5000:
        raise ImageOpError("That tiling would repeat more than 5000 times — use a larger size or gap.")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    row = 0
    for y in range(-ph // 2, H + ph, step_y):
        offset = (step_x // 2) if row % 2 else 0
        for x in range(-pw + offset, W + pw, step_x):
            over(layer, piece, (x, y))
        row += 1
    return _with_layer(base, layer)


def _op_text(img: Image.Image, op: dict, ctx: Context, *, watermark: bool = False) -> Image.Image:
    text = str(op.get("text") if op.get("text") is not None else "").replace("\\n", "\n")
    if not text.strip():
        raise ImageOpError("text needs `text`.")
    base = _work(img)
    W, H = base.size
    ref = min(W, H)
    size = length(op.get("size", op.get("font_size")), H, "size", max(12, round(ref * (0.06 if watermark else 0.05))))
    size = max(4, min(int(size or 12), 4000))
    font, font_note = ctx.fonts.get(op.get("font"), size, bold=boolean(op.get("bold")),
                                    italic=boolean(op.get("italic")), text=text)
    if font_note:
        ctx.note(f"text: {font_note}")
    max_width = length(op.get("max_width"), W, "max_width")
    if max_width:
        text = _wrap(text, font, max_width)
    align = str(op.get("align") or "left").strip().lower()
    if align not in ("left", "center", "right"):
        align = "center" if align in ("centre", "middle") else "left"
    line_spacing = number(op.get("line_spacing"), "line_spacing", 1.2, 0.5, 5) or 1.2
    spacing = max(0, round(size * (line_spacing - 1.0)))
    stroke = max(0, length(op.get("stroke"), ref, "stroke", 0, fractions=False) or 0)
    bg = colour(op.get("background"), "background")
    pad = length(op.get("padding"), size, "padding", round(size * 0.35) if bg else 0) or 0
    shadow_raw = op.get("shadow")
    shadow = None
    if shadow_raw not in (None, False, "", "none", "false"):
        shadow = (0, 0, 0, 160) if shadow_raw is True or str(shadow_raw).lower() == "true" else colour(shadow_raw, "shadow")
    opacity = number(op.get("opacity"), "opacity", 0.5 if watermark else 1.0, 0, 1) or 0.0
    angle = number(op.get("angle"), "angle", 0.0) or 0.0

    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    left, top, right, bottom = probe.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align,
                                                        stroke_width=stroke)
    offset = max(1, round(size / 18)) if shadow else 0
    lw, lh = (right - left) + 2 * pad + offset, (bottom - top) + 2 * pad + offset
    check_size(lw, lh, "The text")
    rw, rh = _rotated_size(lw, lh, angle) if angle else (lw, lh)
    tile = boolean(op.get("tile"))
    margin = length(op.get("margin"), ref, "margin", round(ref * 0.03)) or 0
    if tile:
        x, y = 0, 0
    else:
        x0, y0 = place((W, H), (rw, rh), anchor(op.get("position"), default="bottom-right" if watermark else "center"),
                       margin)
        x = length(op.get("x"), W, "x", x0)
        y = length(op.get("y"), H, "y", y0)
        x, y = int(x if x is not None else x0), int(y if y is not None else y0)

    color = colour(op.get("color"), "color")
    if color is None:
        if bg is not None and bg[3] >= 128:
            color = _contrasting(bg)
        else:
            left_, top_ = min(max(0, x), W - 1), min(max(0, y), H - 1)
            box_ = (left_, top_, max(left_ + 1, min(W, x + rw)), max(top_ + 1, min(H, y + rh)))
            under = base.convert("RGB") if tile else base.convert("RGB").crop(box_)
            gray = under.convert("L").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
            color = _contrasting((gray, gray, gray, 255))  # type: ignore[arg-type]
    stroke_color = colour(op.get("stroke_color"), "stroke_color") or _contrasting(color)

    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    origin = (pad - left, pad - top)
    if bg is not None:
        box_radius = length(op.get("radius"), lh, "radius", round(pad * 0.6)) or 0
        ImageDraw.Draw(layer).rounded_rectangle((0, 0, lw - 1 - offset, lh - 1 - offset), radius=box_radius, fill=bg)
    if shadow is not None:
        ghost = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        ImageDraw.Draw(ghost).multiline_text((origin[0] + offset, origin[1] + offset), text, font=font, fill=shadow,
                                             spacing=spacing, align=align, stroke_width=stroke, stroke_fill=shadow)
        layer = Image.alpha_composite(layer, ghost)
    letters = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    ImageDraw.Draw(letters).multiline_text(origin, text, font=font, fill=color, spacing=spacing, align=align,
                                           stroke_width=stroke, stroke_fill=stroke_color)
    layer = Image.alpha_composite(layer, letters)
    if angle:
        layer = layer.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
    layer = _fade(layer, opacity)
    if tile:
        gap = length(op.get("gap"), ref, "gap", round(max(layer.size) * 0.6)) or 0
        return _tile(base, layer, gap)
    full = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    over(full, layer, (x, y))
    return _with_layer(base, full)


def _op_watermark(img: Image.Image, op: dict, ctx: Context) -> Image.Image:
    return _op_text(img, op, ctx, watermark=True)


# ── The operation table ────────────────────────────────────────────────

OPS: dict[str, Callable[[Image.Image, dict, Context], Image.Image]] = {
    # geometry
    "resize": _op_resize, "crop": _op_crop, "rotate": _op_rotate, "flip": _op_flip, "pad": _op_pad,
    "border": _op_border, "trim": _op_trim, "round_corners": _op_round_corners, "circle": _op_circle,
    "perspective": _op_perspective,
    # tone and colour
    "adjust": _op_adjust, "grayscale": _op_grayscale, "sepia": _op_sepia, "invert": _op_invert,
    "autocontrast": _op_autocontrast, "equalize": _op_equalize, "posterize": _op_posterize,
    "solarize": _op_solarize, "threshold": _op_threshold, "tint": _op_tint, "duotone": _op_duotone,
    "quantize": _op_quantize, "opacity": _op_opacity, "transparent": _op_transparent, "flatten": _op_flatten,
    "mode": _op_mode, "vignette": _op_vignette, "noise": _op_noise, "gradient": _op_gradient,
    # filters
    "blur": _op_blur, "sharpen": _op_sharpen, "median": _op_median, "pixelate": _op_pixelate,
    "edges": _simple_filter(ImageFilter.FIND_EDGES), "edge_enhance": _simple_filter(ImageFilter.EDGE_ENHANCE_MORE),
    "emboss": _simple_filter(ImageFilter.EMBOSS), "contour": _simple_filter(ImageFilter.CONTOUR),
    "detail": _simple_filter(ImageFilter.DETAIL), "smooth": _simple_filter(ImageFilter.SMOOTH_MORE),
    # drawing
    "text": _op_text, "watermark": _op_watermark, "rectangle": _op_rectangle, "ellipse": _op_ellipse,
    "line": _op_line, "arrow": _op_arrow, "polygon": _op_polygon,
}

# Ops that may work on part of the picture only (`region` / `box`): it is cut
# out, changed and put back.
REGIONABLE = frozenset({
    "adjust", "grayscale", "sepia", "invert", "autocontrast", "equalize", "posterize", "solarize", "threshold",
    "tint", "duotone", "vignette", "noise", "gradient", "blur", "sharpen", "median", "pixelate", "edges",
    "edge_enhance", "emboss", "contour", "detail", "smooth",
})

# The adjust settings, each also an op of its own ({"op": "brightness", "amount": 1.2}).
_ADJUST_KEYS = ("brightness", "contrast", "saturation", "sharpness", "gamma", "hue", "warmth")

ALIASES: dict[str, str] = {
    "scale": "resize", "thumbnail": "resize", "shrink": "resize", "cut": "crop", "turn": "rotate",
    "mirror": "flip", "greyscale": "grayscale", "gray": "grayscale", "grey": "grayscale",
    "black_and_white": "grayscale", "black_white": "grayscale", "bw": "grayscale", "desaturate": "grayscale",
    "monochrome": "grayscale", "negative": "invert", "frame": "border", "extend": "pad", "expand": "pad",
    "canvas": "pad", "autocrop": "trim", "auto_crop": "trim", "rounded": "round_corners",
    "rounded_corners": "round_corners", "round": "round_corners", "circle_crop": "circle", "crop_circle": "circle",
    "avatar": "circle", "keystone": "perspective", "straighten": "perspective", "warp": "perspective",
    "auto_contrast": "autocontrast", "auto_level": "autocontrast", "auto_levels": "autocontrast",
    "levels": "autocontrast", "normalize": "autocontrast", "posterise": "posterize", "binarize": "threshold",
    "colorize": "tint", "colourise": "tint", "colour": "tint", "color_overlay": "tint", "reduce_colors": "quantize",
    "palette": "quantize", "fade": "opacity", "alpha": "opacity", "transparency": "opacity",
    "remove_background": "transparent", "make_transparent": "transparent", "remove_color": "transparent",
    "remove_alpha": "flatten", "convert_mode": "mode", "color_mode": "mode", "vignetting": "vignette",
    "grain": "noise", "gaussian_blur": "blur", "unsharp": "sharpen", "unsharp_mask": "sharpen",
    "denoise": "median", "despeckle": "median", "mosaic": "pixelate", "censor": "pixelate",
    "find_edges": "edges", "edge": "edges", "edge_detect": "edges", "caption": "text", "label": "text",
    "annotate": "text", "rect": "rectangle", "box": "rectangle", "highlight": "rectangle", "oval": "ellipse",
    "saturate": "saturation", "temperature": "warmth", "hue_shift": "hue", "exposure": "brightness",
    "brighten": "brightness",
}


def canonical_op(name: Any) -> str:
    """The op an op name (or alias) means, or "" when it means none."""
    key = re.sub(r"[\s\-]+", "_", str(name or "").strip().lower())
    key = ALIASES.get(key, key)
    return key if key in OPS or key in _ADJUST_KEYS else ""


def apply_operations(img: Image.Image, operations: list[Any], ctx: Context) -> tuple[Image.Image, list[str]]:
    """*operations* applied to *img* in order; returns the result and what each step did."""
    done: list[str] = []
    for i, raw in enumerate(operations, 1):
        if not isinstance(raw, dict):
            raise ImageOpError(f"operations[{i}] must be an object with an `op`.")
        asked = raw.get("op") or raw.get("type") or raw.get("operation") or ""
        name = canonical_op(asked)
        if not name:
            raise ImageOpError(f"operations[{i}]: unknown op {asked!r}. Ops: {', '.join(OPS)}.")
        op = dict(raw)
        key = re.sub(r"[\s\-]+", "_", str(asked).strip().lower())
        if key == "mirror" and not _given(op.get("direction")):
            op["direction"] = "horizontal"
        if name in _ADJUST_KEYS:
            value = next((op[k] for k in (name, "amount", "factor", "value") if _given(op.get(k))), None)
            if value is None:
                raise ImageOpError(f"operations[{i}] ({asked}): give the amount, e.g. "
                                   f"{{'op': '{name}', 'amount': {'1.2' if name not in ('hue', 'warmth') else '0.3'}}}.")
            op = {k: v for k, v in op.items() if k not in ("amount", "factor", "value")}
            op[name] = value
            name = "adjust"
        before = img.size
        try:
            box = region(op, img) if name in REGIONABLE else None
            if box is not None:
                piece = OPS[name](img.crop(box), op, ctx)
                target = _work(img)
                target = target.copy()
                piece = piece.convert(target.mode)
                if piece.size != (box[2] - box[0], box[3] - box[1]):
                    piece = piece.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
                target.paste(piece, box[:2])
                img = target
            else:
                img = OPS[name](img, op, ctx)
        except ImageOpError as exc:
            raise ImageOpError(f"operations[{i}] ({asked}): {exc}") from None
        step = name if key in (name, "") else f"{name} ({key})"
        if box is not None:
            step += " on a region"
        if img.size != before:
            step += f" → {img.size[0]}×{img.size[1]}"
        done.append(step)
    return img, done


# ── Several pictures ───────────────────────────────────────────────────

_LAYOUTS = {
    "horizontal": "horizontal", "row": "horizontal", "side_by_side": "horizontal", "side-by-side": "horizontal",
    "h": "horizontal", "left_to_right": "horizontal", "vertical": "vertical", "column": "vertical",
    "v": "vertical", "top_to_bottom": "vertical", "above": "vertical", "grid": "grid", "collage": "grid",
    "mosaic": "grid", "stack": "stack", "layers": "stack", "layered": "stack", "on_top": "stack",
    "overlap": "stack",
}
_ALIGN = {"start": 0.0, "top": 0.0, "left": 0.0, "center": 0.5, "centre": 0.5, "middle": 0.5,
          "end": 1.0, "bottom": 1.0, "right": 1.0}


def _fit_into(img: Image.Image, size: tuple[int, int], fit: str, fill: tuple[int, int, int, int]) -> Image.Image:
    """*img* made exactly *size*: letterboxed (contain), cropped (cover) or stretched (fill)."""
    w, h = size
    img = _sizable(img)
    if img.size == (w, h):
        return img
    W, H = img.size
    if fit in ("fill", "stretch"):
        return img.resize((w, h), Image.Resampling.LANCZOS)
    if fit == "cover":
        f = max(w / W, h / H)
        big = img.resize((max(w, round(W * f)), max(h, round(H * f))), Image.Resampling.LANCZOS)
        x, y = (big.width - w) // 2, (big.height - h) // 2
        return big.crop((x, y, x + w, y + h))
    f = min(w / W, h / H)
    inner = img.resize((max(1, round(W * f)), max(1, round(H * f))), Image.Resampling.LANCZOS)
    canvas = _canvas((w, h), fill, inner)
    return over(canvas, inner, ((w - inner.width) // 2, (h - inner.height) // 2))


def _label(canvas: Image.Image, text: str, box: tuple[int, int, int, int], size: int, fill: tuple[int, int, int, int],
           color: tuple[int, int, int, int] | None, ctx: Context) -> None:
    """*text* centred in *box* of *canvas* (changed in place)."""
    font, note = ctx.fonts.get(None, size, text=text)
    if note:
        ctx.note(f"labels: {note}")
    ink = color or _contrasting(fill if fill[3] >= 128 else (255, 255, 255, 255))
    bw, bh = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((bw - (right - left)) // 2 - left, (bh - (bottom - top)) // 2 - top), text, font=font, fill=ink)
    over(canvas, layer, (box[0], box[1]))


def combine(images: list[Image.Image], spec: dict[str, Any], ctx: Context) -> tuple[Image.Image, str]:
    """Several pictures as one: side by side, one above another, a grid, or stacked on top of each other."""
    n = len(images)
    if n < 2:
        raise ImageOpError("combine needs at least two pictures in `paths`.")
    layout_word = re.sub(r"[\s]+", "_", str(spec.get("layout") or "horizontal").strip().lower())
    layout = _LAYOUTS.get(layout_word)
    if layout is None:
        raise ImageOpError("`layout` is horizontal (side by side), vertical (one above another), grid "
                           "(columns × rows) or stack (on top of each other).")
    imgs = [_work(i) for i in images]
    alpha = any(has_alpha(i) for i in imgs)
    fill = colour(spec.get("background"), "background", None) or ((0, 0, 0, 0) if alpha else (255, 255, 255, 255))
    ref = min(min(i.size) for i in imgs)
    gap = max(0, length(spec.get("gap"), ref, "gap", 0, fractions=False) or 0)
    padding = max(0, length(spec.get("padding"), ref, "padding", 0, fractions=False) or 0)
    align = _ALIGN.get(str(spec.get("align") or "center").strip().lower())
    if align is None:
        raise ImageOpError("`align` is start, center or end.")
    labels = [str(t) for t in listish(spec.get("labels"), "labels")]
    match = str(spec.get("match") or "auto").strip().lower()
    label_color = colour(spec.get("label_color"), "label_color")

    if layout == "stack":
        W = max(i.width for i in imgs) + 2 * padding
        H = max(i.height for i in imgs) + 2 * padding
        canvas = _canvas((W, H), fill, imgs[0] if alpha else None)
        where = anchor(spec.get("position"), default="center")
        for img in imgs:
            over(canvas, img, place((W, H), img.size, where, padding))
        return canvas, f"Stacked {n} pictures on top of each other ({W}×{H} px)."

    if layout in ("horizontal", "vertical"):
        along = 1 if layout == "horizontal" else 0  # the side made equal: height in a row, width in a column
        side_name = "height" if along else "width"
        if match not in ("auto", "none", "height", "width"):
            raise ImageOpError("`match` is height (row), width (column) or none (keep sizes).")
        if match != "none":
            sides = sorted(i.size[along] for i in imgs)
            # The middle one: a small logo among photos doesn't shrink them all.
            want = max(1, length(spec.get(side_name), sides[-1], side_name) or sides[len(sides) // 2])
            sizes = []
            for img in imgs:
                f = want / img.size[along]
                sizes.append((max(1, round(img.width * f)), want) if along else (want, max(1, round(img.height * f))))
            if along:
                check_size(sum(s[0] for s in sizes) + gap * (n - 1) + 2 * padding, want + 2 * padding)
            else:
                check_size(want + 2 * padding, sum(s[1] for s in sizes) + gap * (n - 1) + 2 * padding)
            imgs = [img if img.size == size else _sizable(img).resize(size, Image.Resampling.LANCZOS)
                    for img, size in zip(imgs, sizes)]
        label_size = length(spec.get("label_size"), ref, "label_size",
                            max(12, round(max(i.size[along] for i in imgs) * 0.05))) or 12
        band = round(label_size * 1.7) if labels else 0
        if along:
            cell_h = max(i.height for i in imgs)
            W = sum(i.width for i in imgs) + gap * (n - 1) + 2 * padding
            H = cell_h + band + 2 * padding
        else:
            cell_w = max(i.width for i in imgs)
            W = cell_w + 2 * padding
            H = sum(i.height for i in imgs) + band * n + gap * (n - 1) + 2 * padding
        canvas = _canvas((W, H), fill, imgs[0] if alpha else None)
        cursor = padding
        for k, img in enumerate(imgs):
            if along:
                xy = (cursor, padding + round(align * (cell_h - img.height)))
                label_box = (cursor, padding + cell_h, cursor + img.width, padding + cell_h + band)
                cursor += img.width + gap
            else:
                xy = (padding + round(align * (cell_w - img.width)), cursor)
                label_box = (padding, cursor + img.height, padding + cell_w, cursor + img.height + band)
                cursor += img.height + band + gap
            over(canvas, img, xy)
            if k < len(labels) and labels[k]:
                _label(canvas, labels[k], label_box, label_size, fill, label_color, ctx)
        where = "side by side" if along else "one above another"
        return canvas, f"Combined {n} pictures {where} ({W}×{H} px)."

    # grid
    columns = integer(spec.get("columns"), "columns", None, 1, n) or math.ceil(math.sqrt(n))
    rows = math.ceil(n / columns)
    widths = sorted(i.width for i in imgs)
    aspects = sorted(i.height / i.width for i in imgs)
    middle = widths[len(widths) // 2]
    cell_w = length(spec.get("cell_width"), middle, "cell_width") or middle
    cell_h = length(spec.get("cell_height"), cell_w, "cell_height") or max(1, round(cell_w * aspects[len(aspects) // 2]))
    fit = str(spec.get("fit") or "contain").strip().lower()
    if fit not in ("contain", "cover", "fill", "stretch"):
        raise ImageOpError("grid `fit` is contain (whole picture, letterboxed), cover (fill the cell, cropped) or fill.")
    label_size = length(spec.get("label_size"), ref, "label_size", max(12, round(cell_h * 0.06))) or 12
    band = round(label_size * 1.7) if labels else 0
    W = columns * cell_w + gap * (columns - 1) + 2 * padding
    H = rows * (cell_h + band) + gap * (rows - 1) + 2 * padding
    check_size(W, H)
    canvas = _canvas((W, H), fill, imgs[0] if alpha else None)
    for k, img in enumerate(imgs):
        r, c = divmod(k, columns)
        x = padding + c * (cell_w + gap)
        y = padding + r * (cell_h + band + gap)
        over(canvas, _fit_into(img, (cell_w, cell_h), fit, fill), (x, y))
        if k < len(labels) and labels[k]:
            _label(canvas, labels[k], (x, y + cell_h, x + cell_w, y + cell_h + band), label_size, fill, label_color, ctx)
    return canvas, f"Combined {n} pictures in a {columns}×{rows} grid ({W}×{H} px, cells {cell_w}×{cell_h})."


_BLENDS: dict[str, Callable[[Image.Image, Image.Image], Image.Image] | None] = {
    "normal": None, "multiply": ImageChops.multiply, "screen": ImageChops.screen, "overlay": ImageChops.overlay,
    "soft_light": ImageChops.soft_light, "hard_light": ImageChops.hard_light, "darken": ImageChops.darker,
    "lighten": ImageChops.lighter, "difference": ImageChops.difference, "add": ImageChops.add,
    "subtract": ImageChops.subtract,
}
BLEND_NAMES = tuple(_BLENDS)


def _overlay_one(base: Image.Image, ov: Image.Image, spec: dict[str, Any], ctx: Context) -> Image.Image:
    W, H = base.size
    ov = _rgba(ov)
    if boolean(spec.get("remove_background")):
        ov = _op_transparent(ov, {"tolerance": spec.get("tolerance")}, ctx)
    scale = number(spec.get("scale"), "scale", None, 0.001, 100)
    w = length(spec.get("width"), W, "width")
    h = length(spec.get("height"), H, "height")
    if scale is not None:
        w, h = max(1, round(W * scale)), None
    if w and not h:
        h = max(1, round(ov.height * w / ov.width))
    elif h and not w:
        w = max(1, round(ov.width * h / ov.height))
    if w and h and (w, h) != ov.size:
        check_size(w, h, "The overlay")
        ov = ov.resize((w, h), Image.Resampling.LANCZOS)
    angle = number(spec.get("angle"), "angle", 0.0) or 0.0
    if angle:
        ov = ov.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
    ov = _fade(ov, number(spec.get("opacity"), "opacity", 1.0, 0, 1) or 0.0)
    blend_name = re.sub(r"[\s\-]+", "_", str(spec.get("blend") or "normal").strip().lower())
    if blend_name not in _BLENDS:
        raise ImageOpError(f"`blend` must be one of: {', '.join(_BLENDS)}.")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if boolean(spec.get("tile")):
        gap = length(spec.get("gap"), min(W, H), "gap", round(min(ov.size) * 0.5)) or 0
        tiled = _tile(Image.new("RGBA", (W, H), (0, 0, 0, 0)), ov, gap)
        layer = tiled
    else:
        where_word = spec.get("position")
        where = anchor(where_word, default="center")
        default_margin = 0 if where == (0.5, 0.5) else round(min(W, H) * 0.03)
        margin = length(spec.get("margin"), min(W, H), "margin", default_margin) or 0
        x0, y0 = place((W, H), ov.size, where, margin)
        x = length(spec.get("x"), W, "x", x0)
        y = length(spec.get("y"), H, "y", y0)
        over(layer, ov, (int(x if x is not None else x0), int(y if y is not None else y0)))
    blender = _BLENDS[blend_name]
    if blender is not None:
        mixed = blender(_work(base).convert("RGB"), layer.convert("RGB"))
        mixed.putalpha(layer.getchannel("A"))
        layer = mixed
    return _with_layer(base, layer)


def overlay(base: Image.Image, items: list[tuple[Image.Image, dict[str, Any]]], ctx: Context) -> Image.Image:
    """Pictures stuck onto *base*, each placed by its own spec, in order (the last on top)."""
    out = _work(base).copy()
    for ov, spec in items:
        out = _overlay_one(out, ov, spec, ctx)
    return out


def animate(frames: list[Image.Image], spec: dict[str, Any], ctx: Context) -> tuple[list[Image.Image], list[int]]:
    """Frames of one size for an animation, and how long each shows (ms)."""
    if len(frames) < 2:
        raise ImageOpError("animate needs at least two pictures in `paths` (or one animated picture).")
    first = frames[0]
    w = length(spec.get("width"), first.width, "width")
    h = length(spec.get("height"), first.height, "height")
    if w and not h:
        h = max(1, round(first.height * w / first.width))
    elif h and not w:
        w = max(1, round(first.width * h / first.height))
    size = (w, h) if w and h else first.size
    check_size(*size, "Each frame")
    if size[0] * size[1] * len(frames) > MAX_ANIMATION_PIXELS:
        raise ImageOpError(f"{len(frames)} frames of {size[0]}×{size[1]} px is too much for one animation — "
                           "make the frames smaller (width/height) or use fewer.")
    alpha = any(has_alpha(f) for f in frames)
    fill = colour(spec.get("background"), "background", None) or ((0, 0, 0, 0) if alpha else (255, 255, 255, 255))
    fit = str(spec.get("fit") or "contain").strip().lower()
    out = [_fit_into(_work(f), size, fit, fill) for f in frames]
    raw = spec.get("durations") if _given(spec.get("durations")) else spec.get("duration")
    if isinstance(raw, (list, tuple)) or (isinstance(raw, str) and raw.strip().startswith("[")):
        values = [integer(v, "durations", 500, 10, 60_000) or 500 for v in listish(raw, "durations")]
        durations = [values[min(i, len(values) - 1)] for i in range(len(out))] if values else [500] * len(out)
    else:
        durations = [integer(raw, "duration", 500, 10, 60_000) or 500] * len(out)
    if boolean(spec.get("bounce")) and len(out) > 2:
        out = out + out[-2:0:-1]
        durations = durations + durations[-2:0:-1]
    return out, durations


def split(img: Image.Image, spec: dict[str, Any]) -> list[tuple[str, Image.Image]]:
    """One picture cut into tiles: (r<row>c<col> suffix, tile), left to right, top to bottom."""
    W, H = img.size
    tw = length(spec.get("tile_width"), W, "tile_width")
    th = length(spec.get("tile_height"), H, "tile_height")
    if tw or th:
        tw, th = tw or W, th or H
        xs = list(range(0, W, tw)) + [W]
        ys = list(range(0, H, th)) + [H]
    else:
        rows = integer(spec.get("rows"), "rows", 1, 1, MAX_TILES) or 1
        cols = integer(spec.get("columns"), "columns", 1, 1, MAX_TILES) or 1
        if rows * cols < 2:
            raise ImageOpError("split needs rows and/or columns (e.g. rows 3, columns 3), or tile_width/tile_height.")
        xs = [round(i * W / cols) for i in range(cols + 1)]
        ys = [round(i * H / rows) for i in range(rows + 1)]
    count = (len(xs) - 1) * (len(ys) - 1)
    if count > MAX_TILES:
        raise ImageOpError(f"That would make {count} tiles — at most {MAX_TILES}.")
    tiles = []
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            if xs[c + 1] > xs[c] and ys[r + 1] > ys[r]:
                tiles.append((f"r{r + 1}c{c + 1}", img.crop((xs[c], ys[r], xs[c + 1], ys[r + 1]))))
    return tiles


PAGE_SIZES = {"a4": (8.27, 11.69), "a3": (11.69, 16.54), "a5": (5.83, 8.27), "letter": (8.5, 11.0),
              "legal": (8.5, 14.0)}


def pdf_pages(images: list[Image.Image], spec: dict[str, Any]) -> tuple[list[Image.Image], float]:
    """Pages for a PDF (white, RGB) and their resolution in dpi."""
    page = str(spec.get("page_size") or "a4").strip().lower()
    dpi = number(spec.get("dpi"), "dpi", 150, 36, 600) or 150.0
    fill = colour(spec.get("background"), "background", (255, 255, 255, 255)) or (255, 255, 255, 255)
    if page in ("fit", "image", "auto", "original"):
        pages = []
        for img in images:
            flat = Image.new("RGB", img.size, fill[:3])
            flat.paste(_rgba(img), (0, 0), _rgba(img))
            pages.append(flat)
        return pages, dpi
    if page not in PAGE_SIZES:
        raise ImageOpError(f"`page_size` is one of: {', '.join(PAGE_SIZES)}, or fit (each page the size of its picture).")
    inch_w, inch_h = PAGE_SIZES[page]
    pages = []
    for img in images:
        pw, ph = round(inch_w * dpi), round(inch_h * dpi)
        if img.width > img.height:
            pw, ph = ph, pw
        margin = length(spec.get("margin"), min(pw, ph), "margin", 0) or 0
        room_w, room_h = max(1, pw - 2 * margin), max(1, ph - 2 * margin)
        work = _sizable(img)
        f = min(room_w / work.width, room_h / work.height, 1.0)
        if f < 1:
            work = work.resize((max(1, round(work.width * f)), max(1, round(work.height * f))), Image.Resampling.LANCZOS)
        sheet = Image.new("RGB", (pw, ph), fill[:3])
        over(sheet, work, ((pw - work.width) // 2, (ph - work.height) // 2))
        pages.append(sheet)
    return pages, dpi


# ── Saving ─────────────────────────────────────────────────────────────


def format_named(word: Any) -> str:
    """'jpg' → 'JPEG' (only formats this module writes)."""
    key = str(word or "").strip().lower().lstrip(".")
    fmt = _FORMAT_WORDS.get(key)
    if fmt is None:
        raise ImageOpError(f"`format` must be one of: {', '.join(sorted(set(_FORMAT_WORDS) - {'icon', 'jpeg', 'tif'}))}.")
    if fmt == "AVIF" and not features.check("avif"):
        raise ImageOpError("AVIF isn't available in this installation — use webp or png.")
    return fmt


def choose_format(*, asked: str | None, source: str | None, alpha: bool, animated: bool,
                  default: str = "PNG") -> tuple[str, str]:
    """The format to save in, and a note when it isn't the obvious one."""
    if asked:
        return asked, ""
    src = _READ_AS_WRITE.get(source or "", source or "")
    fmt = src if src in FORMAT_EXT and src not in ("PDF", "ICO") else default
    if animated and fmt not in ANIMATED:
        return "GIF", f"saved as GIF to keep the animation ({fmt} can't animate)"
    if alpha and fmt in NO_ALPHA:
        return "PNG", f"saved as PNG to keep the transparency ({fmt} has none)"
    return fmt, ""


def _for_format(img: Image.Image, fmt: str, fill: tuple[int, int, int, int], ctx: Context) -> Image.Image:
    if fmt in NO_ALPHA:
        if has_alpha(img):
            rgba = _rgba(img)
            flat = Image.new("RGB", rgba.size, fill[:3])
            flat.paste(rgba, (0, 0), rgba)
            ctx.note(f"{fmt} has no transparency: see-through parts were filled with "
                     f"{'white' if fill[:3] == (255, 255, 255) else 'the background colour'}")
            return flat
        if fmt == "JPEG" and img.mode not in ("L", "RGB", "CMYK"):
            return img.convert("RGB")
        if fmt == "BMP" and img.mode not in ("1", "L", "P", "RGB"):
            return img.convert("RGB")
        if fmt == "PDF" and img.mode not in ("1", "L", "RGB", "CMYK"):
            return img.convert("RGB")
        return img
    if fmt in ("WEBP", "AVIF"):
        return _work(img)
    if fmt == "ICO":
        rgba = _rgba(img)
        if rgba.width != rgba.height:
            side = max(rgba.size)
            square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            square.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
            rgba = square
        return rgba
    if fmt == "GIF":
        return img if img.mode in ("P", "L", "RGB", "RGBA") else _work(img)
    if fmt == "PNG":
        return img if img.mode in ("1", "L", "LA", "P", "RGB", "RGBA") else _work(img)
    if fmt == "TIFF":
        return img if img.mode in ("1", "L", "P", "RGB", "RGBA", "CMYK") else _work(img)
    return img


def encode(frames: list[Image.Image], fmt: str, ctx: Context, *, durations: list[int] | None = None,
           loop: int = 0, quality: int | None = None, lossless: bool = False,
           background: tuple[int, int, int, int] | None = None, icc_profile: bytes | None = None,
           exif: bytes | None = None, dpi: float | None = None) -> bytes:
    """The frames as a file of *fmt*: several only when the format holds them."""
    if not frames:
        raise ImageOpError("Nothing to save.")
    if len(frames) > 1 and fmt not in ANIMATED | {"PDF", "TIFF"}:
        ctx.note(f"{fmt} can't hold an animation — saved the first frame only")
        frames = frames[:1]
    fill = background or (255, 255, 255, 255)
    prepared = [_for_format(f, fmt, fill, ctx) for f in frames]
    first = prepared[0]
    params: dict[str, Any] = {}
    if fmt == "JPEG":
        params.update(quality=quality or 90, optimize=True)
    elif fmt == "PNG":
        params.update(optimize=first.width * first.height <= 2_000_000)
    elif fmt == "WEBP":
        params.update(quality=quality or 90, lossless=lossless, method=4)
    elif fmt == "AVIF":
        params.update(quality=quality or 75)
    elif fmt == "TIFF":
        params.update(compression="tiff_lzw" if features.check("libtiff") else None)
    elif fmt == "ICO":
        side = first.width
        params.update(sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256) if s <= side] or [(side, side)])
    elif fmt == "PDF":
        params.update(resolution=float(dpi or 72.0))
    if icc_profile and fmt in ("JPEG", "PNG", "WEBP", "TIFF", "AVIF") and first.mode in ("RGB", "RGBA"):
        params["icc_profile"] = icc_profile
    if exif and fmt in ("JPEG", "PNG", "WEBP", "TIFF", "AVIF"):
        params["exif"] = exif
    if len(prepared) > 1:
        params.update(save_all=True, append_images=prepared[1:])
        if fmt in ANIMATED:
            params.update(duration=list(durations or [100] * len(prepared))[:len(prepared)], loop=int(loop or 0))
            if fmt == "GIF":
                params["disposal"] = 2
    buf = io.BytesIO()
    try:
        first.save(buf, fmt, **params)
    except (OSError, ValueError, KeyError) as exc:
        raise ImageOpError(f"Couldn't save as {fmt}: {exc}") from None
    return buf.getvalue()


def exif_bytes(exif: Image.Exif | None) -> bytes | None:
    """EXIF to keep in a saved copy, marked upright (the pixels were turned already)."""
    if exif is None or not len(exif):
        return None
    try:
        exif[0x0112] = 1
        return exif.tobytes()
    except Exception:
        return None


# ── Describing ─────────────────────────────────────────────────────────


def _human_bytes(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n} bytes"


def human_bytes(n: int) -> str:
    return _human_bytes(int(n))


def _gps(exif: Image.Exif) -> str:
    try:
        gps = exif.get_ifd(0x8825)
    except Exception:
        return ""
    if not gps or 2 not in gps or 4 not in gps:
        return ""

    def degrees(value: Any, ref: Any, negative: str) -> float:
        d, m, s = (float(v) for v in value)
        out = d + m / 60 + s / 3600
        return -out if str(ref or "").upper().startswith(negative) else out

    try:
        lat = degrees(gps[2], gps.get(1), "S")
        lon = degrees(gps[4], gps.get(3), "W")
    except Exception:
        return "present"
    return f"{lat:.6f}, {lon:.6f}"


def describe(path: Path) -> list[str]:
    """What a picture is: size, format, frames, transparency, EXIF and main colours."""
    _extra_openers()
    try:
        im = Image.open(path)
    except (UnidentifiedImageError, OSError):
        hint = _NOT_PICTURES.get(path.suffix.lower())
        return [f"{path.name}: not a picture this tool can open" + (f" — {hint}." if hint else ".")]
    with im:
        w, h = im.size
        n = int(getattr(im, "n_frames", 1) or 1)
        parts = [f"{w}×{h} px", im.format or "?", im.mode]
        try:
            exif = im.getexif()
        except Exception:
            exif = None
        orientation = int(exif.get(0x0112, 1)) if exif else 1
        lines = [f"{path.name}: " + ", ".join(parts) + f", {_human_bytes(path.stat().st_size)}"]
        if orientation in (5, 6, 7, 8):
            lines.append(f"  Stored sideways (EXIF orientation {orientation}): shown and edited upright as {h}×{w} px.")
        elif orientation in (2, 3, 4):
            lines.append(f"  EXIF orientation {orientation}: shown and edited turned upright.")
        if n > 1:
            total = 0
            for frame in ImageSequence.Iterator(im):
                total += int(frame.info.get("duration") or im.info.get("duration") or 0)
            lines.append(f"  Animated: {n} frames, {total / 1000:.2f} s, loop {im.info.get('loop', 0) or 'forever'}.")
            im.seek(0)
        dpi = im.info.get("dpi")
        if dpi:
            lines.append(f"  DPI: {round(float(dpi[0]))}×{round(float(dpi[1]))}")
        if im.info.get("icc_profile"):
            lines.append("  Colour profile: embedded")
        if exif:
            try:
                details = exif.get_ifd(0x8769)
            except Exception:
                details = {}
            taken = details.get(0x9003) or exif.get(0x0132)
            camera = " ".join(str(exif.get(t) or "").strip() for t in (0x010F, 0x0110)).strip()
            if taken:
                lines.append(f"  Taken: {taken}")
            if camera:
                lines.append(f"  Camera: {camera}")
            where = _gps(exif)
            if where:
                lines.append(f"  GPS location: {where}")
        if w * h <= MAX_INPUT_PIXELS:
            try:
                if im.format == "JPEG":
                    im.draft("RGB", (256, 256))  # decode at a fraction of the size: only a thumbnail is needed
                thumb = normalize(im.copy())
                thumb.thumbnail((128, 128))
                if has_alpha(thumb):
                    lo = thumb.getchannel("A").getextrema()[0]
                    if lo < 255:
                        clear = sum(thumb.getchannel("A").histogram()[:16])
                        lines.append(f"  Transparency: yes ({clear * 100 // (thumb.width * thumb.height)}% "
                                     "fully see-through)")
                rgb = _rgba(thumb)
                solid = Image.new("RGB", rgb.size, (255, 255, 255))
                solid.paste(rgb, (0, 0), rgb)
                pal = solid.quantize(colors=5, method=Image.Quantize.MEDIANCUT)
                palette = pal.getpalette() or []
                counts = sorted(pal.getcolors() or [], reverse=True)
                total = sum(c for c, _ in counts) or 1
                swatches = []
                for count, idx in counts[:5]:
                    r, g, b = palette[idx * 3: idx * 3 + 3]
                    swatches.append(f"#{r:02x}{g:02x}{b:02x} {count * 100 // total}%")
                lines.append("  Main colours: " + ", ".join(swatches))
            except Exception:
                pass
    return lines
