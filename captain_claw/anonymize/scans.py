"""Text from scans, for the anonymizer: local OCR only (``captain_claw.ocr`` —
Tesseract or RapidOCR, by fixed rules), never a language model.

Three uses:

* :func:`picture_reader` — the text in a picture inside an Office file, which
  ``office.prepare(pictures="strip")`` keeps as the picture's alt text;
* :func:`pdf_text` — a PDF whose pages have no usable text layer (a scan):
  those pages are read with OCR, the others keep their text;
* :func:`image_text` — a picture file (``.png``, ``.jpg``, ``.tif`` …).

Nothing is cached: the OCR pipeline normally keeps results next to the file,
which would leave the raw text of a private document on disk, so every call
gets a temporary folder that is gone afterwards.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
# Tesseract with Croatian + English when it has both models; otherwise the
# first installed engine (RapidOCR's one multilingual model reads both).
LANG = "hrv+eng"
MIN_PICTURE_SIDE = 48     # icons and bullets have no text worth reading
MAX_PDF_PAGES = 500


class ScanError(RuntimeError):
    """No OCR engine, or the scan couldn't be read."""


def _cfg() -> Any:
    try:
        from captain_claw.config import get_config

        return get_config().tools.ocr
    except Exception:  # noqa: BLE001 — no config (tests, bare use): engine defaults
        return None


def _choice():
    from captain_claw.ocr.engines import EngineError, pick_engine

    try:
        return pick_engine("auto", LANG, _cfg())
    except EngineError as exc:
        raise ScanError(str(exc)) from None


def available() -> bool:
    try:
        _choice()
    except ScanError:
        return False
    return True


def _settings(**over: Any):
    from captain_claw.ocr import document as doc

    cfg = _cfg()
    base = dict(dpi=int(getattr(cfg, "dpi", 300) or 300),
                min_text_chars=int(getattr(cfg, "min_text_chars", 25) or 25),
                low_confidence=float(getattr(cfg, "low_confidence", 70) or 70),
                workers=int(getattr(cfg, "workers", 0) or 0))
    base.update(over)
    return doc.Settings(**base)


def _read_picture(choice, settings, blob: bytes, ext: str) -> str:
    from PIL import Image

    from captain_claw.ocr import document as doc

    with tempfile.TemporaryDirectory(prefix="anon-ocr-") as tmp:
        src = Path(tmp) / f"picture.{ext or 'png'}"
        src.write_bytes(blob)
        try:
            with Image.open(src) as im:
                if min(im.size) < MIN_PICTURE_SIDE:
                    return ""
        except Exception:  # noqa: BLE001 — a format Pillow can't open (EMF on macOS …): no text
            return ""
        png, _w, _h = doc.prepare_picture(src, Path(tmp) / "prepared.png",
                                          upscale=choice.engine.name == "tesseract")
        rec, _deg = doc.ocr_picture(choice, png, settings.low_confidence)
        return rec.text or ""


def picture_reader() -> Callable[[bytes, str], str] | None:
    """A reader for ``office.prepare(picture_reader=…)``, or None when no OCR
    engine is installed (the pictures are still replaced, just not read)."""
    try:
        choice = _choice()
    except ScanError:
        return None
    settings = _settings()

    def read(blob: bytes, ext: str) -> str:
        return _read_picture(choice, settings, blob, ext.lower())

    return read


def image_text(blob: bytes, ext: str) -> str:
    """The text in a picture file; raises ScanError without an OCR engine."""
    choice = _choice()
    return _read_picture(choice, _settings(), blob, ext.lower().lstrip("."))


def pdf_text(blob: bytes) -> tuple[str, int]:
    """A PDF's text as markdown, page by page: pages with a text layer as they
    are, the rest read with OCR. Returns (text, pages read with OCR)."""
    from captain_claw.ocr import document as doc

    choice = _choice()
    with tempfile.TemporaryDirectory(prefix="anon-ocr-") as tmp:
        path = Path(tmp) / "document.pdf"
        path.write_bytes(blob)
        try:
            result = doc.run_pdf(path, choice, _settings(max_pages=MAX_PDF_PAGES), Path(tmp) / "cache",
                                 refresh=True)
        except Exception as exc:  # noqa: BLE001 — EngineError, a damaged PDF …
            raise ScanError(str(exc) or type(exc).__name__) from None
    parts = []
    ocr_pages = 0
    for page in result.pages:
        if page.method == "ocr":
            ocr_pages += 1
        text = (page.text or "").strip()
        parts.append(f"## Page {page.page}\n\n{text}" if text else f"## Page {page.page}\n\n_(no text)_")
    if result.skipped_pages:
        parts.append(f"_({len(result.skipped_pages)} more page(s) not read: over {MAX_PDF_PAGES})_")
    return "\n\n".join(parts), ocr_pages


def pdf_needs_ocr(blob: bytes) -> bool:
    """Whether any page of the PDF has no usable text layer."""
    from pypdf import PdfReader

    from captain_claw.ocr import document as doc

    with tempfile.TemporaryDirectory(prefix="anon-ocr-") as tmp:
        path = Path(tmp) / "document.pdf"
        path.write_bytes(blob)
        try:
            reader = PdfReader(str(path))
            total = len(reader.pages)
            cover = doc.page_image_cover(path, total)
            settings = _settings()
            for n in range(min(total, MAX_PDF_PAGES)):
                try:
                    raw = reader.pages[n].extract_text() or ""
                except Exception:  # noqa: BLE001
                    raw = ""
                if doc.text_layer_verdict(raw, cover[n], settings.min_text_chars) != "text_layer":
                    return True
        except Exception:  # noqa: BLE001 — unreadable: let the normal path say so
            return False
    return False
