"""Local OCR engines — no language model anywhere in this package.

Two engines, both deterministic on CPU (same picture + same engine version +
same settings → the same text):

- **tesseract** — the system binary (brew / apt / the Windows installer). One
  trained model per language (``eng``, ``hrv``, ``deu``…), word boxes and a
  confidence per word. Run with ``OMP_THREAD_LIMIT=1`` so its own threading
  can't vary the result; pages are run side by side instead.
- **rapidocr** — the ``rapidocr`` package (pip, ONNX Runtime, models bundled
  in the wheel). One multilingual model (PP-OCR): no language to pick, reads
  Latin diacritics (č ć đ š ž) without any language pack. Lines, not words.

:func:`pick_engine` chooses by a fixed rule, recorded with every result:
a language the caller named goes to Tesseract when it has that language's
model; otherwise the first installed engine in ``tools.ocr.engines`` order.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from captain_claw.logging import get_logger

log = get_logger(__name__)

ENGINE_NAMES = ("tesseract", "rapidocr")
LANG_RE = re.compile(r"^[A-Za-z_]{2,20}(\+[A-Za-z_]{2,20})*$")


@dataclass
class Line:
    """One line of recognised text on a page."""

    text: str
    conf: float  # 0–100
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in the page picture's pixels
    para: int = 0  # paragraph index on the page; a change = blank line

    def to_json(self) -> dict[str, Any]:
        return {"text": self.text, "conf": round(self.conf, 1), "box": list(self.box), "para": self.para}


@dataclass
class Recognition:
    lines: list[Line] = field(default_factory=list)
    width: int = 0
    height: int = 0
    tall: float = 0.0  # share of text boxes taller than wide: a page lying on its side

    @property
    def text(self) -> str:
        return lines_to_text(self.lines)

    @property
    def confidence(self) -> float | None:
        """Character-weighted mean confidence, ``None`` with no text."""
        total = sum(len(ln.text) for ln in self.lines)
        if not total:
            return None
        return sum(ln.conf * len(ln.text) for ln in self.lines) / total


def lines_to_text(lines: list[Line]) -> str:
    out: list[str] = []
    prev_para: int | None = None
    for ln in lines:
        if prev_para is not None and ln.para != prev_para:
            out.append("")
        out.append(ln.text)
        prev_para = ln.para
    return "\n".join(out).strip()


def gap_join(pieces: list[tuple[str, int, int]]) -> str:
    """Join ``(text, x0, x1)`` pieces left to right: one space between words,
    three where the gap is wider than three characters (table columns)."""
    pieces = sorted(pieces, key=lambda p: (p[1], p[2]))
    chars = sum(len(t) for t, _a, _b in pieces) or 1
    cw = max(1.0, sum(b - a for _t, a, b in pieces) / chars)
    out = ""
    prev_end: int | None = None
    for text, x0, x1 in pieces:
        if prev_end is not None:
            out += "   " if (x0 - prev_end) > 3 * cw else " "
        out += text
        prev_end = x1
    return out


class EngineError(RuntimeError):
    pass


# ── Tesseract ─────────────────────────────────────────────────────────────

_WINDOWS_TESSERACT = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


class TesseractEngine:
    name = "tesseract"

    def __init__(self, binary: str = "", tessdata_dir: str = "", psm: int = 3, timeout: float = 180.0):
        self._binary_hint = binary
        self.tessdata_dir = tessdata_dir
        self.psm = int(psm)
        self.timeout = timeout
        self._version: str | None = None
        self._langs: frozenset[str] | None = None

    @property
    def binary(self) -> str | None:
        if self._binary_hint:
            return self._binary_hint if Path(self._binary_hint).is_file() or shutil.which(self._binary_hint) else None
        found = shutil.which("tesseract")
        if found:
            return found
        if os.name == "nt":
            for cand in _WINDOWS_TESSERACT:
                if Path(cand).is_file():
                    return cand
        return None

    def available(self) -> bool:
        return self.binary is not None

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["OMP_THREAD_LIMIT"] = "1"  # its OpenMP threads must not vary the result
        if self.tessdata_dir:
            env["TESSDATA_PREFIX"] = self.tessdata_dir
        return env

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[bytes]:
        binary = self.binary
        if not binary:
            raise EngineError("tesseract is not installed")
        return subprocess.run([binary, *args], capture_output=True, timeout=self.timeout, env=self._env())

    def version(self) -> str:
        if self._version is None:
            try:
                proc = self._run(["--version"])
                first = (proc.stdout or proc.stderr).decode("utf-8", "replace").splitlines()
                m = re.search(r"(\d+\.\d+(?:\.\d+)?)", first[0] if first else "")
                self._version = m.group(1) if m else "unknown"
            except Exception:  # noqa: BLE001
                self._version = "unknown"
        return self._version

    def languages(self) -> frozenset[str]:
        if self._langs is None:
            try:
                proc = self._run(["--list-langs"])
                text = (proc.stdout or b"").decode("utf-8", "replace") + (proc.stderr or b"").decode("utf-8", "replace")
                self._langs = frozenset(
                    ln.strip() for ln in text.splitlines()
                    if ln.strip() and not ln.lower().startswith("list of") and " " not in ln.strip())
            except Exception:  # noqa: BLE001
                self._langs = frozenset()
        return self._langs

    def has_langs(self, lang: str) -> bool:
        have = self.languages()
        return all(code in have for code in lang.split("+"))

    def recognize(self, image: Path, lang: str) -> Recognition:
        proc = self._run([str(image), "stdout", "--oem", "1", "--psm", str(self.psm), "-l", lang, "tsv"])
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise EngineError(f"tesseract failed: {err[-1] if err else proc.returncode}")
        return parse_tesseract_tsv(proc.stdout.decode("utf-8", "replace"))

    def orientation(self, image: Path) -> int:
        """Degrees the page must turn clockwise to stand upright (0 when unsure)."""
        if "osd" not in self.languages():
            return 0
        try:
            proc = self._run([str(image), "stdout", "--psm", "0", "-l", "osd"])
        except Exception:  # noqa: BLE001
            return 0
        out = (proc.stdout or b"").decode("utf-8", "replace")
        rot = re.search(r"Rotate:\s*(\d+)", out)
        conf = re.search(r"Orientation confidence:\s*([\d.]+)", out)
        if not rot or not conf or float(conf.group(1)) < 1.5:
            return 0
        deg = int(rot.group(1)) % 360
        return deg if deg in (90, 180, 270) else 0


def parse_tesseract_tsv(tsv: str) -> Recognition:
    """Tesseract's TSV → lines (words grouped by block/paragraph/line)."""
    rec = Recognition()
    groups: dict[tuple[int, int, int], list[tuple[str, int, int, int, int, float]]] = {}
    order: list[tuple[int, int, int]] = []
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12:
            continue
        try:
            level = int(cols[0])
            block, par, line = int(cols[2]), int(cols[3]), int(cols[4])
            left, top, width, height = int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9])
            conf = float(cols[10])
        except ValueError:
            continue
        if level == 1:
            rec.width, rec.height = width, height
            continue
        text = "\t".join(cols[11:]).strip()
        if level != 5 or not text or conf < 0:
            continue
        key = (block, par, line)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((text, left, top, left + width, top + height, conf))
    words_all = [w for key in order for w in groups[key] if len(w[0]) >= 3]
    if words_all:
        rec.tall = sum(1 for w in words_all if (w[4] - w[2]) > 1.5 * (w[3] - w[1])) / len(words_all)
    para_ids: dict[tuple[int, int], int] = {}
    for key in order:
        words = groups[key]
        text = gap_join([(w[0], w[1], w[3]) for w in words])
        chars = sum(len(w[0]) for w in words) or 1
        conf = sum(w[5] * len(w[0]) for w in words) / chars
        box = (min(w[1] for w in words), min(w[2] for w in words),
               max(w[3] for w in words), max(w[4] for w in words))
        para = para_ids.setdefault(key[:2], len(para_ids))
        rec.lines.append(Line(text=text, conf=conf, box=box, para=para))
    return rec


# ── RapidOCR ──────────────────────────────────────────────────────────────

class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self, threads: int = 2):
        self.threads = max(1, int(threads))
        self._ocr: Any = None
        self._probe: Any = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        import importlib.util

        try:
            return (importlib.util.find_spec("rapidocr") is not None
                    and importlib.util.find_spec("onnxruntime") is not None)
        except Exception:  # noqa: BLE001
            return False

    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("rapidocr")
        except Exception:  # noqa: BLE001
            return "unknown"

    def _params(self) -> dict[str, Any]:
        return {
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": self.threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }

    def orientation(self, image: Path) -> int:
        """Degrees the page must turn clockwise to stand upright: each quarter
        turn is read with the line-flip model OFF, and the turn whose level
        (wider-than-tall) lines read best wins — upside-down text reads as
        garbage without the flip, so it can't tie with the upright page."""
        from PIL import Image

        if self._probe is None:
            from rapidocr import RapidOCR

            self._probe = RapidOCR(params={**self._params(), "Global.use_cls": False})
        with Image.open(image) as im:
            base = im.copy()
        best, best_deg = 0.0, 0
        for deg in (0, 90, 180, 270):
            probe = image.with_name(f"{image.stem}-probe{deg}.png")
            base.rotate(-deg, expand=True).save(probe)
            try:
                with self._lock:
                    res = self._probe(str(probe))
            finally:
                probe.unlink(missing_ok=True)
            score = 0.0
            boxes, txts, scores = (getattr(res, k, None) for k in ("boxes", "txts", "scores"))
            if boxes is None or txts is None or scores is None:  # numpy arrays: no `or`
                continue
            for box, text, sc in zip(boxes, txts, scores):
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                if max(xs) - min(xs) >= max(ys) - min(ys):
                    score += len(str(text)) * float(sc)
            if score > best * 1.05:
                best, best_deg = score, deg
        return best_deg

    def _engine(self) -> Any:
        if self._ocr is None:
            from rapidocr import RapidOCR

            self._ocr = RapidOCR(params={
                "Global.log_level": "error",
                # Word boxes: the line's spaces are rebuilt from where its words
                # sit — the line model alone can drop one ("12  85.00" → "1285.00").
                "Global.return_word_box": True,
                "EngineConfig.onnxruntime.intra_op_num_threads": self.threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
        return self._ocr

    def recognize(self, image: Path, lang: str = "") -> Recognition:
        # One picture at a time: the model is shared and its session is not
        # documented as safe to call from several threads at once.
        with self._lock:
            result = self._engine()(str(image))
        boxes = getattr(result, "boxes", None)
        txts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        rec = Recognition()
        try:
            from PIL import Image

            with Image.open(image) as im:
                rec.width, rec.height = im.size
        except Exception:  # noqa: BLE001
            pass
        if boxes is None or txts is None:
            return rec
        words = getattr(result, "word_results", None)
        if not isinstance(words, (list, tuple)) or len(words) != len(txts):
            words = [None] * len(txts)
        items = []
        for box, text, score, wl in zip(boxes, txts, scores if scores is not None else [1.0] * len(txts), words):
            text = _spaced_from_words(wl) or str(text or "").strip()
            if not text:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            items.append((text, int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), float(score) * 100.0))
        if items:
            rec.tall = sum(1 for it in items if (it[4] - it[2]) > 1.5 * (it[3] - it[1])) / len(items)
        rec.lines = rows_to_lines(items)
        return rec


def _spaced_from_words(words: Any) -> str:
    """A line's text rebuilt from RapidOCR's word boxes ``(word, score, box)``,
    spaced by :func:`gap_join`; "" when they can't be used."""
    if not words:
        return ""
    pieces = []
    try:
        for w in words:
            text = str(w[0] or "").strip()
            if not text:
                continue
            xs = [float(p[0]) for p in w[2]]
            pieces.append((text, int(min(xs)), int(max(xs))))
    except Exception:  # noqa: BLE001 — another rapidocr shape: use the line text
        return ""
    return gap_join(pieces) if pieces else ""


def rows_to_lines(items: list[tuple[str, int, int, int, int, float]]) -> list[Line]:
    """Line boxes ``(text, x0, y0, x1, y1, conf)`` → reading-order lines: boxes
    whose vertical centres sit within half a line height share a row (joined
    left to right); a gap taller than 1.2 lines starts a new paragraph."""
    if not items:
        return []
    items = sorted(items, key=lambda it: ((it[2] + it[4]) / 2, it[1]))
    rows: list[list[tuple[str, int, int, int, int, float]]] = []
    for it in items:
        yc, h = (it[2] + it[4]) / 2, max(1, it[4] - it[2])
        if rows:
            row = rows[-1]
            ryc = sum((r[2] + r[4]) / 2 for r in row) / len(row)
            rh = max(1, min(r[4] - r[2] for r in row))
            if abs(yc - ryc) <= 0.5 * min(h, rh):
                row.append(it)
                continue
        rows.append([it])
    heights = [max(1, max(r[4] for r in row) - min(r[2] for r in row)) for row in rows]
    line_h = median(heights) if heights else 1
    lines: list[Line] = []
    para = 0
    prev_bottom: int | None = None
    for row in rows:
        top = min(r[2] for r in row)
        if prev_bottom is not None and top - prev_bottom > 1.2 * line_h:
            para += 1
        chars = sum(len(r[0]) for r in row) or 1
        lines.append(Line(
            text=gap_join([(r[0], r[1], r[3]) for r in row]),
            conf=sum(r[5] * len(r[0]) for r in row) / chars,
            box=(min(r[1] for r in row), top, max(r[3] for r in row), max(r[4] for r in row)),
            para=para,
        ))
        prev_bottom = max(r[4] for r in row)
    return lines


# ── Choosing ──────────────────────────────────────────────────────────────

_ENGINES: dict[str, Any] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(name: str, cfg: Any = None) -> Any:
    """The engine object (one per process, so loaded models are reused)."""
    with _ENGINES_LOCK:
        eng = _ENGINES.get(name)
        if eng is None:
            if name == "tesseract":
                eng = TesseractEngine(
                    binary=str(getattr(cfg, "tesseract_path", "") or ""),
                    tessdata_dir=str(getattr(cfg, "tessdata_dir", "") or ""),
                    psm=int(getattr(cfg, "tesseract_psm", 3) or 3),
                )
            elif name == "rapidocr":
                eng = RapidOcrEngine(threads=int(getattr(cfg, "rapidocr_threads", 2) or 2))
            else:
                raise EngineError(f"unknown OCR engine '{name}' (use: {', '.join(ENGINE_NAMES)})")
            _ENGINES[name] = eng
        return eng


def reset_engines() -> None:
    """Forget the engine objects (tests, or after installing an engine)."""
    with _ENGINES_LOCK:
        _ENGINES.clear()


INSTALL_HELP = (
    "No local OCR engine is installed, so the scanned text can't be read without a model. "
    "RapidOCR comes with Captain, so this install is missing it (or runs on 32-bit ARM, "
    "which it doesn't support). Install one of these, then call this tool again:\n"
    "  • RapidOCR — any 64-bit OS, through pip: pip install -U captain-claw "
    "(or: pip install rapidocr onnxruntime pypdfium2)\n"
    "  • Tesseract — macOS: brew install tesseract tesseract-lang · Debian/Ubuntu: "
    "apt install tesseract-ocr tesseract-ocr-hrv (one package per language) · Windows: the "
    "UB Mannheim installer (github.com/UB-Mannheim/tesseract/wiki)\n"
    "Tell the user this. The image_ocr tool can read the page with a vision model instead — "
    "only if the user agrees to a model reading it."
)


@dataclass(frozen=True)
class Choice:
    engine: Any
    lang: str  # what the engine is run with ("" for rapidocr: one multilingual model)
    note: str = ""


def pick_engine(requested: str, lang: str, cfg: Any = None) -> Choice:
    """Which engine reads the pages. Fixed rule, so the same machine and the same
    request always pick the same engine:

    1. ``requested`` names an engine → that one (an error when it isn't
       installed, or Tesseract lacks the language).
    2. ``lang`` named and Tesseract has all of its models → Tesseract.
    3. Otherwise the first installed engine in ``tools.ocr.engines`` order
       (Tesseract with the configured default language).
    """
    requested = (requested or "auto").strip().lower()
    lang = (lang or "").strip()
    if lang and not LANG_RE.match(lang):
        raise EngineError(f"lang must be Tesseract language codes like 'eng' or 'hrv+eng', got '{lang}'")
    default_lang = str(getattr(cfg, "lang", "") or "eng")
    order = [n for n in (getattr(cfg, "engines", None) or ENGINE_NAMES) if n in ENGINE_NAMES] or list(ENGINE_NAMES)

    if requested != "auto":
        eng = get_engine(requested, cfg)
        if not eng.available():
            raise EngineError(f"OCR engine '{requested}' is not installed.\n{INSTALL_HELP}")
        if requested == "tesseract":
            want = lang or default_lang
            if not eng.has_langs(want):
                raise EngineError(missing_lang_help(want, eng.languages()))
            return Choice(eng, want)
        return Choice(eng, "", "multilingual model" + (f"; lang '{lang}' not used" if lang else ""))

    tess = get_engine("tesseract", cfg)
    if lang and tess.available() and tess.has_langs(lang):
        return Choice(tess, lang)
    for name in order:
        eng = get_engine(name, cfg)
        if not eng.available():
            continue
        if name == "tesseract":
            want = lang or default_lang
            if eng.has_langs(want):
                return Choice(eng, want)
            continue
        note = "multilingual model"
        if lang:
            note += f"; Tesseract has no '{lang}' model, so it was not used"
        return Choice(eng, "", note)
    if lang and tess.available():
        raise EngineError(missing_lang_help(lang, tess.languages()))
    raise EngineError(INSTALL_HELP)


def missing_lang_help(lang: str, have: frozenset[str]) -> str:
    missing = [c for c in lang.split("+") if c not in have]
    return (
        f"Tesseract has no model for {', '.join(missing)} (installed: "
        f"{', '.join(sorted(have)) or 'none'}). Install the language — macOS: brew install "
        "tesseract-lang · Debian/Ubuntu: apt install " + " ".join(f"tesseract-ocr-{c}" for c in missing)
        + " · or put <code>.traineddata from github.com/tesseract-ocr/tessdata_fast into "
        "tools.ocr.tessdata_dir — or use RapidOCR (it comes with Captain: engine='rapidocr'), "
        "whose one model reads Latin-script languages without a language pack."
    )
