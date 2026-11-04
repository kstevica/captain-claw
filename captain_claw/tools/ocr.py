"""``ocr_pdf`` and ``ocr_image`` — read scanned documents with a local OCR
engine, the same way every time. No language model (see captain_claw/ocr/)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _cfg() -> Any:
    try:
        from captain_claw.config import get_config

        return get_config().tools.ocr
    except Exception:  # noqa: BLE001 — bare tool tests: defaults
        return None


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


class _OcrTool(Tool):
    """Shared run: resolve the file, pick the engine, run, store, report."""

    kind = "pdf"
    timeout_seconds = 1200.0

    def __init__(self) -> None:
        cfg = _cfg()
        if cfg is not None:
            attr = "pdf_timeout_seconds" if self.kind == "pdf" else "image_timeout_seconds"
            self.timeout_seconds = float(getattr(cfg, attr, self.timeout_seconds) or self.timeout_seconds)

    async def execute(
        self,
        path: str = "",
        lang: str = "",
        pages: Any = None,
        mode: str = "auto",
        engine: str = "auto",
        dpi: Any = None,
        preprocess: bool = False,
        refresh: bool = False,
        max_chars: Any = None,
        **kwargs: Any,
    ) -> ToolResult:
        from captain_claw.ocr import document as doc
        from captain_claw.ocr.engines import EngineError, pick_engine
        from captain_claw.tools.document_extract import (
            _extract_char_limit,
            _pack_error,
            _resolve_readable_file,
            _truncate_markdown,
            _with_pack_header,
        )
        from captain_claw.tools.write import WriteTool

        raw = str(path or "").strip()
        if not raw:
            return ToolResult(success=False, error="Missing required argument: path")
        file_path, error = await _resolve_readable_file(raw, runtime_base_path=kwargs.get("_runtime_base_path"))
        if error:
            return ToolResult(success=False, error=error)
        assert file_path is not None
        ext = file_path.suffix.lower()
        if self.kind == "pdf" and ext != ".pdf":
            hint = " Use ocr_image for pictures." if ext in doc.IMAGE_EXTENSIONS else ""
            return ToolResult(success=False, error=f"ocr_pdf reads .pdf files, got '{ext or file_path.name}'.{hint}")
        if self.kind == "image" and ext not in doc.IMAGE_EXTENSIONS:
            hint = " Use ocr_pdf for PDFs." if ext == ".pdf" else ""
            return ToolResult(success=False, error=(
                f"ocr_image reads {', '.join(sorted(doc.IMAGE_EXTENSIONS))}, got '{ext or file_path.name}'.{hint}"))

        cfg = _cfg()
        mode = str(mode or "auto").strip().lower()
        if mode not in ("auto", "ocr"):
            return ToolResult(success=False, error="mode must be 'auto' (OCR only pages without usable text) or 'ocr' (every page)")
        settings = doc.Settings(
            mode=mode,
            dpi=_int(dpi if dpi is not None else getattr(cfg, "dpi", 300), 300, 100, 600),
            pages=pages,
            max_pages=_int(getattr(cfg, "max_pages", 100), 100, 1, 2000),
            min_text_chars=_int(getattr(cfg, "min_text_chars", 25), 25, 1, 10000),
            low_confidence=float(getattr(cfg, "low_confidence", 70) or 70),
            workers=_int(getattr(cfg, "workers", 0), 0, 0, 32),
            preprocess=bool(preprocess),
        )
        try:
            choice = pick_engine(str(engine or "auto"), str(lang or ""), cfg)
        except EngineError as exc:
            return ToolResult(success=False, error=str(exc))

        saved_root = WriteTool._resolve_saved_root(kwargs)
        session = WriteTool._normalize_session_id(str(kwargs.get("_session_id", "")))
        progress = _progress(kwargs.get("_stream_callback"))
        abort_event = kwargs.get("_abort_event")

        def _aborted() -> bool:
            return bool(abort_event is not None and abort_event.is_set())

        runner = doc.run_pdf if self.kind == "pdf" else doc.run_image
        try:
            result = await asyncio.to_thread(
                runner, file_path, choice, settings, saved_root,
                refresh=bool(refresh), progress=progress, abort=_aborted)
        except (EngineError, ValueError) as exc:
            return ToolResult(success=False, error=_pack_error(raw, exc))
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed", self.name, path=raw, error=str(exc))
            return ToolResult(success=False, error=_pack_error(raw, exc))

        folder = doc.output_dir(saved_root, session, result)
        try:
            md_path, json_path = await asyncio.to_thread(
                doc.write_outputs, result, folder, raw, settings.low_confidence)
        except Exception as exc:  # noqa: BLE001 — the text still goes back
            log.warning("ocr outputs not written", error=str(exc))
            md_path = json_path = None
        self._register(md_path, kwargs)

        body = doc.render_markdown(result, raw, settings.low_confidence)
        limit = _extract_char_limit(max_chars, tool_loop="_agent" in kwargs)
        report = self._summary(result, settings.low_confidence, md_path, json_path)
        content = report + "\n\n" + _truncate_markdown(body, limit)
        log.info("%s done", self.name, path=raw, engine=result.engine, pages=len(result.pages),
                 cached=result.cached)
        return ToolResult(success=True, content=_with_pack_header(file_path, content))

    @staticmethod
    def _register(md_path: Path | None, kwargs: dict[str, Any]) -> None:
        registry = kwargs.get("_file_registry")
        if registry is None or md_path is None:
            return
        try:
            registry.register(logical_path=str(md_path), physical_path=str(md_path),
                              task_id=str(kwargs.get("_task_id", "")))
        except Exception:  # noqa: BLE001
            pass

    def _summary(self, result: Any, low_bar: float, md_path: Path | None, json_path: Path | None) -> str:
        from captain_claw.ocr import document as doc

        n_text = sum(1 for p in result.pages if p.method == "text_layer")
        ocred = [p for p in result.pages if p.method != "text_layer"]
        lines = [f"{self.name}: {result.source.name} — {len(result.pages)} of {result.total_pages} "
                 f"page{'s' if result.total_pages != 1 else ''} read"]
        if result.kind == "pdf":
            lines.append(f"Text layer: {n_text} page(s) · OCR: {len(ocred)} page(s)"
                         + (f" with {doc.engine_label(result)}" if ocred else ""))
        else:
            lines.append(f"OCR with {doc.engine_label(result)}")
        if result.engine_note and ocred:
            lines.append(f"Engine: {result.engine_note}")
        low = [p.page for p in result.pages if p.low(low_bar)]
        empty = [p.page for p in result.pages if p.method == "empty"]
        if low:
            lines.append(f"⚠️ Low confidence on page(s) {doc._ranges(low)} — names and numbers there may be "
                         "misread; say so when you use them, and don't present them as certain.")
        if empty:
            lines.append(f"⚠️ No text found on page(s) {doc._ranges(empty)} (blank, a photo, or handwriting).")
        if result.skipped_pages:
            lines.append(f"Not read (page cap): {doc._ranges(result.skipped_pages)} — call again with "
                         f"pages='{doc._ranges(result.skipped_pages[:100])}'.")
        if md_path is not None:
            lines.append(f"Saved: {md_path} (the text) · {json_path} (every line with its box and confidence)")
        lines.append("Same file + same settings always gives this same text"
                     + (" (stored result reused)." if result.cached else "."))
        return "\n".join(lines)


def _progress(cb: Any) -> Callable[[str], None] | None:
    if not callable(cb):
        return None

    def emit(msg: str) -> None:
        try:
            cb(f"{msg}\n")
        except Exception:  # noqa: BLE001
            pass

    return emit


_LANG_DOC = (
    "Tesseract language code(s) of the document, '+'-joined: 'eng', 'hrv', 'deu', 'hrv+eng'. "
    "Pass it for any non-English document. Used when Tesseract has those models; otherwise the "
    "multilingual RapidOCR model reads the page (it needs no language). The result says which."
)
_COMMON_PROPS: dict[str, Any] = {
    "lang": {"type": "string", "description": _LANG_DOC},
    "engine": {"type": "string", "enum": ["auto", "tesseract", "rapidocr"],
               "description": "Force one engine (default auto: the fixed rule above)."},
    "refresh": {"type": "boolean", "description": "Read again instead of reusing the stored result for this file and settings."},
    "max_chars": {"type": "number", "description": "Maximum characters of text to return."},
}


class OcrPdfTool(_OcrTool):
    name = "ocr_pdf"
    kind = "pdf"
    timeout_seconds = 1200.0
    description = (
        "Read a scanned or image-only PDF with a local OCR engine (Tesseract / RapidOCR — no AI model, "
        "same text every time). Use it when pdf_extract shows pages with no text, garbled text, or "
        "says a page looks scanned. In mode 'auto' each page's own text layer is used when it is "
        "good and only the other pages are OCR'd; mode 'ocr' re-reads every page (when a scan's "
        "embedded text is wrong). Returns the text page by page with how each page was read and its "
        "confidence, and saves it under saved/ocr/ (Markdown + a JSON with every line's box and "
        "confidence). Never render pages or OCR with scripts or shell — use this tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the .pdf file."},
            "pages": {"type": "string", "description": "Pages to read, e.g. '1-3,7' (default: all, up to the page cap)."},
            "mode": {"type": "string", "enum": ["auto", "ocr"],
                     "description": "auto (default): OCR only pages without a usable text layer. ocr: OCR every page."},
            "dpi": {"type": "number", "description": "Render resolution for OCR, 100-600 (default 300)."},
            **_COMMON_PROPS,
        },
        "required": ["path"],
    }


class OcrImageTool(_OcrTool):
    name = "ocr_image"
    kind = "image"
    timeout_seconds = 300.0
    description = (
        "Read the text in a picture — a photo or scan of a document, receipt, invoice, form, "
        "screenshot — with a local OCR engine (Tesseract / RapidOCR — no AI model, same text "
        "every time). .png .jpg .jpeg .webp .bmp .gif .tif .tiff (each page of a multi-page TIFF). "
        "Returns the text with its confidence and saves it under saved/ocr/. Prefer it to "
        "image_ocr for printed text; image_ocr (a vision model) is for handwriting, or when this "
        "tool's confidence is low and the user agrees to a model reading it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the picture."},
            "pages": {"type": "string", "description": "Multi-page TIFF only: pages to read, e.g. '1-2'."},
            "preprocess": {"type": "boolean", "description": "Straighten a tilted scan first (OpenCV deskew)."},
            **_COMMON_PROPS,
        },
        "required": ["path"],
    }
