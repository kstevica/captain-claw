"""Scanned PDFs and pictures → text, the same way every time.

Pipeline (one run of ``ocr_pdf`` / ``ocr_image``):

1. **Fingerprint.** sha256 of the file's bytes + a signature of every setting
   that changes the output (engine and its version, language, dpi, mode, pages,
   :data:`PIPELINE_VERSION`). Same fingerprint → the stored result, unchanged.
2. **Sort the pages** (PDF, ``mode=auto``) by fixed rules, each recorded as the
   page's ``reason``:
   - ``text_layer`` — the page has a usable text layer: read it, no OCR;
   - ``no_text_layer`` — fewer than ``min_text_chars`` characters;
   - ``garbled_text_layer`` — mostly replacement / private-use / control
     characters or ``(cid:N)`` codes (broken font encodings);
   - ``scan_with_thin_text`` — pictures cover ≥ 60% of the page and the text
     layer is under 200 characters (a stamp or footer on a scan).
   ``mode=ocr`` OCRs every page (``forced``) — for a scan whose embedded text
   layer is wrong.
3. **Render** the pages to grayscale at a fixed dpi (pypdfium2, else poppler's
   pdftoppm); a picture is EXIF-rotated, made grayscale, and doubled when its
   long side is under 1600 px.
4. **OCR** each page with the chosen local engine (engines.py); a page that
   reads empty or below the confidence bar is tried upright again (Tesseract's
   orientation check) and the better reading kept.
5. **Store** under ``saved/ocr/<session>/<name>-<sha8>/``: ``<name>.md`` (the
   text, page by page, with how each page was read) and a hidden ``.ocr.json``
   (every line with its box and confidence). A hidden cache in
   ``saved/.ocr-cache/`` keeps the result for any later session that holds the
   same bytes.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.ocr.engines import Choice, EngineError, Line, Recognition

log = get_logger(__name__)

# Bump when a change here changes the text a page yields, so stored results
# made the old way are not served for the new one.
PIPELINE_VERSION = 1

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"})
CACHE_DIR = ".ocr-cache"
CACHE_KEEP = 300  # stored results kept (oldest dropped)
MAX_PIXELS = 40_000_000  # a rendered page larger than this is drawn at a lower dpi
UPSCALE_BELOW = 1600  # a picture whose long side is shorter is doubled


@dataclass
class PageResult:
    page: int
    method: str  # "text_layer" | "ocr" | "empty"
    reason: str
    text: str
    confidence: float | None = None
    lines: list[Line] = field(default_factory=list)
    width: int = 0
    height: int = 0
    rotated: int = 0
    dpi: int = 0

    def low(self, bar: float) -> bool:
        return self.method == "ocr" and self.confidence is not None and self.confidence < bar

    def to_json(self) -> dict[str, Any]:
        return {
            "page": self.page, "method": self.method, "reason": self.reason,
            "confidence": None if self.confidence is None else round(self.confidence, 1),
            "chars": len(self.text), "width": self.width, "height": self.height,
            "rotated": self.rotated, "dpi": self.dpi, "text": self.text,
            "lines": [ln.to_json() for ln in self.lines],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PageResult:
        return cls(
            page=int(d["page"]), method=str(d["method"]), reason=str(d.get("reason", "")),
            text=str(d.get("text", "")), confidence=d.get("confidence"),
            lines=[Line(text=ln["text"], conf=float(ln["conf"]), box=tuple(ln["box"]), para=int(ln.get("para", 0)))
                   for ln in d.get("lines", [])],
            width=int(d.get("width", 0)), height=int(d.get("height", 0)),
            rotated=int(d.get("rotated", 0)), dpi=int(d.get("dpi", 0)),
        )


@dataclass
class Settings:
    mode: str = "auto"  # auto | ocr
    dpi: int = 300
    pages: Any = None  # "1-3,7" / [1, 2] / 5; None = all (up to max_pages)
    max_pages: int = 100
    min_text_chars: int = 25
    low_confidence: float = 70.0
    workers: int = 0  # 0 = min(4, cpu count)
    preprocess: bool = False  # deskew pictures (OpenCV) before OCR


# ── Small helpers ─────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pages(spec: Any, total: int) -> list[int] | None:
    """``"1-3,7"`` / ``[1, 2]`` / ``5`` → sorted 1-based page numbers within
    the document; ``None`` for all. Raises ValueError on a bad spec."""
    if spec is None or spec == "" or spec == []:
        return None
    if isinstance(spec, (int, float)):
        parts = [str(int(spec))]
    elif isinstance(spec, (list, tuple)):
        parts = [str(p) for p in spec]
    else:
        parts = [p for p in re.split(r"[,\s]+", str(spec).strip()) if p]
    pages: set[int] = set()
    for part in parts:
        m = re.fullmatch(r"(\d+)(?:-(\d+)?)?", part)
        if not m:
            raise ValueError(f"pages must look like '1-3,7', got '{spec}'")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else (total if part.endswith("-") else a)
        if a < 1 or b < a:
            raise ValueError(f"bad page range '{part}'")
        pages.update(range(a, min(b, total) + 1))
    if not pages:
        raise ValueError(f"no pages of {total} match '{spec}'")
    return sorted(pages)


_BAD_CHAR_RE = re.compile(r"[�-\x00-\x08\x0b\x0c\x0e-\x1f]")
_CID_RE = re.compile(r"\(cid:\d+\)")


def text_layer_verdict(text: str, image_cover: float, min_chars: int) -> str:
    """Why a page needs OCR (a reason code), or ``"text_layer"`` when its own
    text is good enough. Fixed thresholds — see the module docstring."""
    body = re.sub(r"\s+", "", text or "")
    n = len(body)
    if n < min_chars:
        return "no_text_layer"
    bad = len(_BAD_CHAR_RE.findall(body)) + 6 * len(_CID_RE.findall(text))
    alnum = sum(1 for c in body if c.isalnum())
    if bad / n > 0.1 or alnum / n < 0.5:
        return "garbled_text_layer"
    if image_cover >= 0.6 and n < 200:
        return "scan_with_thin_text"
    return "text_layer"


def _normalize_text(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in (value or "").splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ── Rendering ─────────────────────────────────────────────────────────────

def _pdfium() -> Any:
    try:
        import pypdfium2 as pdfium  # type: ignore[import-untyped]

        return pdfium
    except Exception:  # noqa: BLE001
        return None


def renderer_name() -> str | None:
    if _pdfium() is not None:
        return "pdfium"
    if shutil.which("pdftoppm"):
        return "pdftoppm"
    return None


RENDER_HELP = (
    "Scanned PDF pages have to be turned into pictures before OCR, and nothing here can "
    "do that. Install pypdfium2 (pip install pypdfium2, or pip install 'captain-claw[ocr]') "
    "or poppler (brew install poppler · apt install poppler-utils)."
)


def page_image_cover(pdf_path: Path, count: int) -> list[float]:
    """Share of each page's area covered by pictures (0 when unknown)."""
    pdfium = _pdfium()
    if pdfium is None:
        return [0.0] * count
    try:
        import pypdfium2.raw as pdfium_c  # type: ignore[import-untyped]

        doc = pdfium.PdfDocument(str(pdf_path))
        out = []
        try:
            for i in range(min(count, len(doc))):
                page = doc[i]
                w, h = page.get_size()
                area = max(1.0, w * h)
                covered = 0.0
                for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=3):
                    left, bottom, right, top = obj.get_pos()
                    covered += max(0.0, min(right, w) - max(left, 0)) * max(0.0, min(top, h) - max(bottom, 0))
                out.append(min(1.0, covered / area))
        finally:
            doc.close()
        return out + [0.0] * (count - len(out))
    except Exception as exc:  # noqa: BLE001
        log.debug("image cover unknown", error=str(exc))
        return [0.0] * count


def render_pdf_pages(pdf_path: Path, pages: list[int], dpi: int, out_dir: Path) -> dict[int, tuple[Path, int]]:
    """Render 1-based *pages* to grayscale PNGs → ``{page: (png, dpi used)}``."""
    out: dict[int, tuple[Path, int]] = {}
    pdfium = _pdfium()
    if pdfium is not None:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            for n in pages:
                page = doc[n - 1]
                w, h = page.get_size()
                use = dpi
                if (w * dpi / 72) * (h * dpi / 72) > MAX_PIXELS:
                    use = max(72, int(72 * (MAX_PIXELS / (w * h)) ** 0.5))
                img = page.render(scale=use / 72, grayscale=True).to_pil()
                dest = out_dir / f"page-{n:04d}.png"
                img.save(dest)
                out[n] = (dest, use)
        finally:
            doc.close()
        return out
    binary = shutil.which("pdftoppm")
    if not binary:
        raise EngineError(RENDER_HELP)
    for n in pages:
        stem = out_dir / f"page-{n:04d}"
        proc = subprocess.run([binary, "-r", str(dpi), "-gray", "-png", "-singlefile",
                               "-f", str(n), "-l", str(n), str(pdf_path), str(stem)],
                              capture_output=True, timeout=180)
        dest = stem.with_suffix(".png")
        if proc.returncode != 0 or not dest.is_file():
            raise EngineError(f"pdftoppm could not render page {n}: "
                              f"{(proc.stderr or b'').decode('utf-8', 'replace').strip()[:200]}")
        out[n] = (dest, dpi)
    return out


def prepare_picture(src: Path, dest: Path, *, preprocess: bool = False, frame: int = 0,
                    upscale: bool = True) -> tuple[Path, int, int]:
    """A picture made ready for OCR: EXIF-upright, grayscale, doubled when small
    and *upscale* (Tesseract wants tall letters; RapidOCR resizes on its own and
    reads a doubled picture worse). Returns ``(png, width, height)``."""
    from PIL import Image, ImageOps

    with Image.open(src) as im:
        if frame:
            im.seek(frame)
        img = ImageOps.exif_transpose(im) if not frame else im.copy()
        img = img.convert("L")
    if upscale and max(img.size) < UPSCALE_BELOW:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    img.save(dest)
    if preprocess:
        try:
            from captain_claw.tools.cv import preprocess_ocr_bytes

            data = dest.read_bytes()
            new = preprocess_ocr_bytes(data, deskew=True, enhance=False)
            if new and new is not data:
                dest.write_bytes(new)
        except Exception as exc:  # noqa: BLE001 — deskew is a help, never a requirement
            log.debug("ocr deskew skipped", error=str(exc))
    with Image.open(dest) as im:
        return dest, im.width, im.height


def picture_frames(path: Path) -> int:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(getattr(im, "n_frames", 1) or 1) if path.suffix.lower() in (".tif", ".tiff") else 1
    except Exception:  # noqa: BLE001
        return 1


# ── OCR one page ──────────────────────────────────────────────────────────

def _score(rec: Recognition) -> float:
    """How good a reading is: confidence × text, halved for a page whose lines
    stand on end (their words read fine, but the line order is scrambled)."""
    conf = rec.confidence or 0.0
    return conf * min(len(rec.text), 4000) * (1.0 - 0.5 * rec.tall)


def ocr_picture(choice: Choice, png: Path, low_bar: float) -> tuple[Recognition, int]:
    """OCR one prepared picture; a page that reads empty or under the bar is
    turned upright (Tesseract's orientation check) and the better reading kept.
    Returns ``(recognition, degrees turned)``."""
    rec = choice.engine.recognize(png, choice.lang)
    conf = rec.confidence
    lying = rec.tall >= 0.5 and len(rec.lines) >= 3
    if conf is not None and conf >= low_bar and len(rec.text) >= 10 and not lying:
        return rec, 0
    orient = getattr(choice.engine, "orientation", None)
    if not callable(orient):
        return rec, 0
    deg = orient(png)
    if not deg:
        return rec, 0
    from PIL import Image

    turned = png.with_name(png.stem + f"-r{deg}.png")
    with Image.open(png) as im:
        im.rotate(-deg, expand=True).save(turned)
    rec2 = choice.engine.recognize(turned, choice.lang)
    return (rec2, deg) if _score(rec2) > _score(rec) else (rec, 0)


# ── The document run ──────────────────────────────────────────────────────

@dataclass
class DocResult:
    source: Path
    sha256: str
    kind: str  # "pdf" | "image"
    total_pages: int
    pages: list[PageResult]
    engine: str
    engine_version: str
    lang: str
    engine_note: str
    renderer: str
    settings: dict[str, Any]
    signature: str
    created: float
    cached: bool = False
    skipped_pages: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION, "source_name": self.source.name,
            "sha256": self.sha256, "kind": self.kind, "total_pages": self.total_pages,
            "engine": self.engine, "engine_version": self.engine_version, "lang": self.lang,
            "engine_note": self.engine_note, "renderer": self.renderer, "settings": self.settings,
            "signature": self.signature, "created": self.created, "skipped_pages": self.skipped_pages,
            "pages": [p.to_json() for p in self.pages],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any], source: Path) -> DocResult:
        return cls(
            source=source, sha256=d["sha256"], kind=d["kind"], total_pages=int(d["total_pages"]),
            pages=[PageResult.from_json(p) for p in d["pages"]], engine=d["engine"],
            engine_version=d["engine_version"], lang=d.get("lang", ""), engine_note=d.get("engine_note", ""),
            renderer=d.get("renderer", ""), settings=d.get("settings", {}), signature=d["signature"],
            created=float(d.get("created", 0)), cached=True, skipped_pages=list(d.get("skipped_pages", [])),
        )


def signature(kind: str, choice: Choice, settings: Settings, renderer: str, pages: list[int]) -> str:
    sig = {
        "v": PIPELINE_VERSION, "kind": kind, "engine": choice.engine.name,
        "engine_version": choice.engine.version(), "lang": choice.lang,
        "psm": getattr(choice.engine, "psm", None), "renderer": renderer,
        "mode": settings.mode, "dpi": settings.dpi, "pages": pages,
        "min_text_chars": settings.min_text_chars, "preprocess": settings.preprocess,
    }
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:10]


def _cache_path(saved_root: Path, sha: str, sig: str) -> Path:
    return saved_root / CACHE_DIR / f"{sha[:24]}-{sig}.json"


def load_cached(saved_root: Path, sha: str, sig: str, source: Path) -> DocResult | None:
    path = _cache_path(saved_root, sha, sig)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("sha256") != sha or data.get("signature") != sig:
            return None
        os.utime(path)
        return DocResult.from_json(data, source)
    except Exception:  # noqa: BLE001 — a missing or broken entry is just a miss
        return None


def store_cached(saved_root: Path, result: DocResult) -> None:
    path = _cache_path(saved_root, result.sha256, result.signature)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(result.to_json(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        entries = sorted(path.parent.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in entries[:-CACHE_KEEP]:
            old.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 — the cache is a saving, not a must
        log.warning("ocr cache not written", error=str(exc))


def _workers(settings: Settings, choice: Choice) -> int:
    if choice.engine.name != "tesseract":
        return 1  # one shared model (engines.RapidOcrEngine locks it anyway)
    if settings.workers > 0:
        return settings.workers
    return max(1, min(4, os.cpu_count() or 1))


def run_pdf(
    path: Path, choice: Choice, settings: Settings, saved_root: Path, *,
    refresh: bool = False, progress: Callable[[str], None] | None = None,
    abort: Callable[[], bool] | None = None,
) -> DocResult:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    if total == 0:
        raise EngineError("The PDF has no pages.")
    wanted = parse_pages(settings.pages, total) or list(range(1, total + 1))
    skipped = wanted[settings.max_pages:]
    wanted = wanted[:settings.max_pages]
    rend = renderer_name() or ""
    sha = sha256_file(path)
    sig = signature("pdf", choice, settings, rend, wanted)
    if not refresh:
        hit = load_cached(saved_root, sha, sig, path)
        if hit is not None:
            return hit

    cover = page_image_cover(path, total)
    pages: dict[int, PageResult] = {}
    to_ocr: list[tuple[int, str]] = []
    for n in wanted:
        try:
            raw = reader.pages[n - 1].extract_text() or ""
        except Exception:  # noqa: BLE001 — an unreadable text layer is no text layer
            raw = ""
        verdict = "forced" if settings.mode == "ocr" else text_layer_verdict(raw, cover[n - 1], settings.min_text_chars)
        if verdict == "text_layer":
            pages[n] = PageResult(page=n, method="text_layer", reason="text_layer", text=_normalize_text(raw))
        else:
            to_ocr.append((n, verdict))

    if to_ocr:
        if not rend:
            raise EngineError(RENDER_HELP)
        if progress:
            progress(f"OCR: {len(to_ocr)} of {len(wanted)} page(s) need it — {choice.engine.name}")
        with tempfile.TemporaryDirectory(prefix="ocr-") as tmp:
            done = 0
            # Render in slices: a 300-page scan never sits on disk all at once,
            # and an abort stops between slices.
            batch = max(1, _workers(settings, choice) * 2)
            for i in range(0, len(to_ocr), batch):
                if abort and abort():
                    raise EngineError("OCR stopped.")
                part = to_ocr[i:i + batch]
                rendered = render_pdf_pages(path, [n for n, _r in part], settings.dpi, Path(tmp))
                results = _ocr_many(choice, [(n, rendered[n][0]) for n, _r in part], settings, abort)
                for n, reason in part:
                    rec, deg = results[n]
                    png, used_dpi = rendered[n]
                    pages[n] = _page_from(n, reason, rec, deg, used_dpi)
                    png.unlink(missing_ok=True)
                done += len(part)
                if progress and len(to_ocr) > batch:
                    progress(f"OCR: {done}/{len(to_ocr)} pages")

    result = DocResult(
        source=path, sha256=sha, kind="pdf", total_pages=total,
        pages=[pages[n] for n in wanted], engine=choice.engine.name,
        engine_version=choice.engine.version(), lang=choice.lang, engine_note=choice.note,
        renderer=rend, settings=_settings_json(settings), signature=sig, created=time.time(),
        skipped_pages=skipped,
    )
    store_cached(saved_root, result)
    return result


def run_image(
    path: Path, choice: Choice, settings: Settings, saved_root: Path, *,
    refresh: bool = False, progress: Callable[[str], None] | None = None,
    abort: Callable[[], bool] | None = None,
) -> DocResult:
    frames = picture_frames(path)
    wanted = parse_pages(settings.pages, frames) or list(range(1, frames + 1))
    skipped = wanted[settings.max_pages:]
    wanted = wanted[:settings.max_pages]
    sha = sha256_file(path)
    sig = signature("image", choice, settings, "pillow", wanted)
    if not refresh:
        hit = load_cached(saved_root, sha, sig, path)
        if hit is not None:
            return hit
    pages: list[PageResult] = []
    with tempfile.TemporaryDirectory(prefix="ocr-") as tmp:
        prepared: list[tuple[int, Path]] = []
        for n in wanted:
            png, _w, _h = prepare_picture(path, Path(tmp) / f"frame-{n:04d}.png",
                                          preprocess=settings.preprocess, frame=n - 1,
                                          upscale=choice.engine.name == "tesseract")
            prepared.append((n, png))
        results = _ocr_many(choice, prepared, settings, abort)
        for n, _png in prepared:
            rec, deg = results[n]
            pages.append(_page_from(n, "picture", rec, deg, 0))
    result = DocResult(
        source=path, sha256=sha, kind="image", total_pages=frames, pages=pages,
        engine=choice.engine.name, engine_version=choice.engine.version(), lang=choice.lang,
        engine_note=choice.note, renderer="pillow", settings=_settings_json(settings),
        signature=sig, created=time.time(), skipped_pages=skipped,
    )
    store_cached(saved_root, result)
    return result


def _ocr_many(choice: Choice, items: list[tuple[int, Path]], settings: Settings,
              abort: Callable[[], bool] | None) -> dict[int, tuple[Recognition, int]]:
    """OCR pictures side by side (Tesseract runs as separate processes); results
    are keyed by page, so the order they finish in never shows."""
    out: dict[int, tuple[Recognition, int]] = {}
    workers = min(_workers(settings, choice), max(1, len(items)))
    if workers == 1:
        for n, png in items:
            if abort and abort():
                raise EngineError("OCR stopped.")
            out[n] = ocr_picture(choice, png, settings.low_confidence)
        return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(ocr_picture, choice, png, settings.low_confidence): n for n, png in items}
        for fut in concurrent.futures.as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def _page_from(n: int, reason: str, rec: Recognition, deg: int, dpi: int) -> PageResult:
    text = rec.text
    return PageResult(
        page=n, method="ocr" if text else "empty", reason=reason, text=text,
        confidence=rec.confidence, lines=rec.lines, width=rec.width, height=rec.height,
        rotated=deg, dpi=dpi,
    )


def _settings_json(s: Settings) -> dict[str, Any]:
    return {"mode": s.mode, "dpi": s.dpi, "pages": s.pages, "max_pages": s.max_pages,
            "min_text_chars": s.min_text_chars, "low_confidence": s.low_confidence,
            "preprocess": s.preprocess}


# ── Output ────────────────────────────────────────────────────────────────

def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w.-]+", "-", Path(name).stem, flags=re.UNICODE).strip("-.")
    return (stem or "document")[:60]


def output_dir(saved_root: Path, session: str, result: DocResult) -> Path:
    return saved_root / "ocr" / session / f"{_safe_stem(result.source.name)}-{result.sha256[:8]}"


def engine_label(result: DocResult) -> str:
    bits = [f"{result.engine} {result.engine_version}"]
    if result.lang:
        bits.append(result.lang)
    if result.kind == "pdf":
        bits.append(f"{result.settings.get('dpi', '')} dpi")
    return ", ".join(bits)


def page_heading(p: PageResult, label: str, low_bar: float) -> str:
    parts = [label]
    if p.method == "text_layer":
        parts.append("text layer")
    elif p.method == "empty":
        parts.append("OCR · no text found")
    else:
        parts.append("OCR")
        if p.confidence is not None:
            parts.append(f"confidence {p.confidence:.0f}%")
        if p.rotated:
            parts.append(f"turned {p.rotated}°")
        if p.low(low_bar):
            parts.append("LOW CONFIDENCE — check names and numbers against the page")
    return "## " + " · ".join(parts)


def render_markdown(result: DocResult, display_source: str, low_bar: float) -> str:
    n_text = sum(1 for p in result.pages if p.method == "text_layer")
    n_ocr = len(result.pages) - n_text
    head = [
        f"# {result.source.name} — text",
        "",
        f"Source: {display_source} · sha256 {result.sha256[:12]} · "
        f"{result.total_pages} page{'s' if result.total_pages != 1 else ''}",
    ]
    how = []
    if n_text:
        how.append(f"text layer on {n_text} page{'s' if n_text != 1 else ''}")
    if n_ocr:
        how.append(f"OCR on {n_ocr} ({engine_label(result)})")
    head.append("Read: " + "; ".join(how) + (f" — {result.engine_note}" if result.engine_note and n_ocr else ""))
    if result.skipped_pages:
        head.append(f"Not read (page cap): {_ranges(result.skipped_pages)}")
    out = "\n".join(head) + "\n"
    single_picture = result.kind == "image" and result.total_pages == 1
    for p in result.pages:
        label = "Picture" if single_picture else f"Page {p.page}"
        out += "\n" + page_heading(p, label, low_bar) + "\n\n"
        out += (p.text or "_(no text)_") + "\n"
    return out


def _ranges(nums: list[int]) -> str:
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if prev != start else str(start))
        start = prev = n
    out.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(out)


def write_outputs(result: DocResult, folder: Path, display_source: str, low_bar: float) -> tuple[Path, Path]:
    """``<name>.md`` (the text) and hidden ``.ocr.json`` (lines, boxes,
    confidence) in *folder*. Writes as bytes, so the files are identical on
    every OS."""
    from captain_claw import saved_attribution

    folder.mkdir(parents=True, exist_ok=True)
    md = folder / f"{_safe_stem(result.source.name)}.md"
    js = folder / ".ocr.json"
    for target, data in (
        (md, render_markdown(result, display_source, low_bar)),
        (js, json.dumps(result.to_json(), ensure_ascii=False, indent=1)),
    ):
        prior = saved_attribution.prior_creator(target)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_bytes(data.encode("utf-8"))
        tmp.replace(target)
        saved_attribution.note_write(target, prior)
    return md, js
