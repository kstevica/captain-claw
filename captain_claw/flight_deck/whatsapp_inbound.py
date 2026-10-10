"""Inbound WhatsApp files: names, types, conversions and the notes the agent reads.

Pure helpers for :mod:`whatsapp_bridge` — no network, no bridge state, so
they are easy to test. The bridge downloads a file the user sent (a
document, a photo, a sticker, an audio file), uploads it to the agent and
describes it in an ``attachment_notes`` line on the turn that carries it.

Names come from the sender (a forwarded document keeps its author's file
name), so every name that reaches a note is cleaned: Unicode NFC, no control
characters, line breaks, bidi overrides or square brackets (a name must not
forge an ``[Attached file: …]`` line or a new note).
"""

from __future__ import annotations

import io
import mimetypes
import re
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

# Explicit MIME → extension table. ``mimetypes`` alone misses the Office
# types on slim Linux images (no /etc/mime.types) and maps audio/ogg to .oga.
EXT_BY_MIME: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-excel": ".xls",
    "application/msword": ".doc",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/rtf": ".rtf",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/comma-separated-values": ".csv",
    "text/tab-separated-values": ".tsv",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/calendar": ".ics",
    "text/vcard": ".vcf",
    "text/x-vcard": ".vcf",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/tiff": ".tiff",
    "image/avif": ".avif",
    "image/svg+xml": ".svg",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/amr": ".amr",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".weba",
    "audio/flac": ".flac",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/3gpp": ".3gp",
    "video/x-msvideo": ".avi",
}

# What the agent's vision step reads (image_ocr._IMAGE_EXTENSIONS).
VISION_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
# Pictures Pillow can turn into a JPEG the vision step reads (HEIC/HEIF need
# the optional pillow-heif plugin).
CONVERTIBLE_EXTS = frozenset({".heic", ".heif", ".tif", ".tiff", ".avif"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".3gp"})
AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".m4a", ".aac", ".amr", ".wav", ".weba", ".flac"})

_KIND_BY_EXT = {
    ".pdf": "PDF", ".docx": "Word document", ".doc": "Word document (legacy .doc)",
    ".odt": "document", ".rtf": "document", ".xlsx": "spreadsheet",
    ".xls": "spreadsheet (legacy .xls)", ".ods": "spreadsheet", ".csv": "CSV table",
    ".tsv": "TSV table", ".pptx": "presentation", ".ppt": "presentation (legacy .ppt)",
    ".odp": "presentation", ".zip": "zip archive", ".txt": "text file", ".md": "text file",
    ".json": "JSON file", ".xml": "XML file", ".html": "HTML file", ".ics": "calendar invite",
    ".vcf": "contact card", ".svg": "SVG drawing",
}

# Control characters, Unicode line breaks and bidi overrides.
_UNSAFE_RE = re.compile(r"[\x00-\x1f\x7f\x85  ‎‏‪-‮⁦-⁩]+")
_EXT_RE = re.compile(r"^\.[a-z0-9]{1,10}$")


def clean_name(raw: object, limit: int = 120) -> str:
    """A sender's file name, safe to show to the agent and the user."""
    text = unicodedata.normalize("NFC", str(raw or ""))
    text = _UNSAFE_RE.sub(" ", text).replace("[", "(").replace("]", ")")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        stem, dot, ext = text.rpartition(".")
        keep = f".{ext}" if dot and stem and len(ext) <= 10 else ""
        text = text[: limit - len(keep)].rstrip() + keep
    return text


def clean_text(raw: object, limit: int = 1000) -> str:
    """A caption or transcript quoted inside a note: one line, capped."""
    text = _UNSAFE_RE.sub(" ", unicodedata.normalize("NFC", str(raw or "")))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def base_mime(mime: object) -> str:
    """``audio/ogg; codecs=opus`` → ``audio/ogg``."""
    return str(mime or "").split(";", 1)[0].strip().lower()


def extension_for(name: str, mime: str) -> str:
    """The file's extension: its own when sane, else from the MIME type."""
    suffix = Path(name).suffix.lower() if name else ""
    if not suffix and name.startswith(".") and name.count(".") == 1:
        suffix = name.lower()  # ".xlsx" — a name that is only an extension
    if _EXT_RE.match(suffix):
        if suffix == ".jpeg" or suffix == ".jfif":
            return ".jpg"
        return suffix
    base = base_mime(mime)
    if base in EXT_BY_MIME:
        return EXT_BY_MIME[base]
    guessed = mimetypes.guess_extension(base) if base else None
    if guessed and _EXT_RE.match(guessed) and guessed != ".bin":
        return guessed
    return ".bin"


def classify(ext: str, mime: str) -> str:
    """``image`` (the vision step reads it), ``convert`` (a picture to turn
    into JPEG first), ``video``, ``audio`` or ``file``."""
    base = base_mime(mime)
    if ext in VISION_EXTS:
        return "image"
    if ext in CONVERTIBLE_EXTS:
        return "convert"
    if ext in VIDEO_EXTS or (base.startswith("video/") and ext not in AUDIO_EXTS):
        return "video"
    if ext in AUDIO_EXTS or base.startswith("audio/"):
        return "audio"
    return "file"


def kind_label(ext: str, kind: str) -> str:
    """A few words for the note: ``spreadsheet``, ``photo``, ``audio``…"""
    if kind == "sticker":
        return "sticker"
    if kind == "photo":
        return "photo"
    if kind in ("image", "convert"):
        return "image"
    if kind == "audio":
        return "audio"
    if kind == "video":
        return "video"
    return _KIND_BY_EXT.get(ext, f"{ext.lstrip('.') or 'binary'} file")


def wamid_tail(wamid: str, n: int = 8) -> str:
    """The last ``n`` alphanumerics of a message id — makes upload names unique."""
    alnum = re.sub(r"[^A-Za-z0-9]", "", wamid or "")
    return alnum[-n:] or "msg"


def upload_name(display: str, ext: str, wamid: str) -> str:
    """The file name sent to the agent's upload endpoint.

    The message id's tail keeps two photos (WhatsApp images have no name)
    or two ``Scan.pdf`` from overwriting each other even on an agent whose
    upload endpoint names files by the second."""
    stem = Path(display).stem if Path(display).suffix.lower() == ext else display
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("._-")[:60] or "file"
    return f"{stem}-{wamid_tail(wamid)}{ext}"


def human_size(n: int) -> str:
    """``48 KB``, ``3.2 MB``."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def to_jpeg(data: bytes) -> bytes | None:
    """A HEIC/HEIF/TIFF/AVIF picture as JPEG bytes, or None when Pillow
    (with pillow-heif for HEIC) can't read it."""
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        import pillow_heif  # type: ignore[import-not-found]
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.seek(0)
            rgb = img.convert("RGB")
            out = io.BytesIO()
            rgb.save(out, format="JPEG", quality=90)
            return out.getvalue()
    except Exception:
        return None


def sticker_png(data: bytes) -> bytes | None:
    """A WhatsApp sticker (WebP, maybe animated) as a still PNG — its first
    frame. Vision providers refuse animated images. None when unreadable."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.seek(0)
            out = io.BytesIO()
            img.convert("RGBA").save(out, format="PNG")
            return out.getvalue()
    except Exception:
        return None


class SeenIds:
    """Message ids already handled — Meta delivers webhooks at least once and
    re-sends a backlog after an outage. Bounded and time-limited."""

    def __init__(self, max_items: int = 5000, ttl: float = 24 * 3600.0) -> None:
        self._ids: OrderedDict[str, float] = OrderedDict()
        self._max = max_items
        self._ttl = ttl

    def first(self, msg_id: str) -> bool:
        """True the first time ``msg_id`` is seen (always True for "")."""
        if not msg_id:
            return True
        now = time.time()
        while self._ids:
            oldest, ts = next(iter(self._ids.items()))
            if now - ts <= self._ttl and len(self._ids) < self._max:
                break
            self._ids.pop(oldest, None)
        if msg_id in self._ids:
            return False
        self._ids[msg_id] = now
        return True

    def clear(self) -> None:
        self._ids.clear()
