"""vision — deterministic, local computer-vision ops backed by OpenCV.

This is the cheap, pixel-exact layer *under* the multimodal-LLM vision tools
(``image_vision`` / ``image_ocr`` / ``video_vision``). It spends no tokens: every
operation here runs locally on CPU. Use it to pre-process and measure for the LLM
(dedupe/keyframe/diff-gate → fewer LLM frames) and to do things an LLM does poorly
(pixel diffs, QR/barcode decode, geometric measurement, template matching).

Scope (Phase 1): classical CV only — no models, no LLM/VLM inference. Operations:

  diff       two images → SSIM score + bounding boxes of changed regions
  dedupe     images → perceptual-hash near-duplicate clusters
  measure    dimensions, dominant colors, blur score, brightness
  prep       deskew / grayscale / threshold / denoise / enhance / crop  (pre-OCR)
  qr         decode QR codes (+ barcodes when available)
  locate     template-match a needle image in a haystack → coords + confidence
  annotate   draw boxes/labels on an image → new image
  keyframes  video → scene-change representative-frame timestamps

OpenCV is an optional dependency (the ``cv`` extra). Absent it, the tool reports
unavailable and nothing else in the system changes (guarded import, like
``_HAS_PILLOW`` in ``image_ocr.py``).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)

# ── optional dependency guard (mirrors _HAS_PILLOW in image_ocr.py) ────────────
try:
    import cv2 as _cv2  # type: ignore[import-untyped]
    import numpy as _np  # numpy is a core dep, but pair it with the cv2 guard

    _HAS_CV2 = True
except Exception:  # pragma: no cover — exercised only where cv2 is absent
    _cv2 = None  # type: ignore[assignment]
    _np = None  # type: ignore[assignment]
    _HAS_CV2 = False

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
_NO_CV2 = (
    "vision requires OpenCV, which isn't installed. Install the extra: "
    "pip install 'captain-claw[cv]' (or pip install opencv-python-headless). "
    "This is a local, CPU-only dependency."
)

_OPS = ("diff", "dedupe", "measure", "prep", "qr", "locate", "annotate", "keyframes")

# Default perceptual-hash Hamming distance under which two frames are "the same".
_DEDUPE_MAX_HAMMING = 6
# Default template-match confidence (TM_CCOEFF_NORMED) to accept a hit.
_LOCATE_MIN_CONF = 0.75


# ── path resolution (accepts both real paths and vfs: paths) ───────────────────


def _resolve_input(path: str, kwargs: dict[str, Any]) -> tuple[Path | None, str | None]:
    """Resolve an input file that may be a real path or a ``vfs:`` path."""
    raw = str(path or "").strip()
    if not raw:
        return None, "Missing path"
    try:
        from captain_claw.vfs import is_vfs_path, resolve_vfs_path

        if is_vfs_path(raw):
            p = resolve_vfs_path(raw)
            if p is None:
                return None, f"Could not resolve VFS path: {raw}"
            if not p.is_file():
                return None, f"File not found: {raw}"
            return p, None
    except Exception:  # pragma: no cover — vfs module should always import
        pass
    from captain_claw.tools.document_extract import _require_existing_file

    return _require_existing_file(raw, runtime_base_path=kwargs.get("_runtime_base_path"))


def _resolve_output(out: str | None, kwargs: dict[str, Any], default_name: str) -> tuple[Path, str]:
    """Pick a destination path for a derived image. Returns (real_path, display).

    - ``vfs:proj/x.png`` → resolved under the shared VFS (parents created).
    - a real path → resolved against the runtime base.
    - omitted → ``<saved>/vision/<default_name>`` so agents can find/share it.
    """
    out = str(out or "").strip()
    if out:
        try:
            from captain_claw.vfs import is_vfs_path, resolve_vfs_path, to_display

            if is_vfs_path(out):
                p = resolve_vfs_path(out, create_parents=True)
                if p is not None:
                    return p, to_display(p)
        except Exception:  # pragma: no cover
            pass
        runtime_base = kwargs.get("_runtime_base_path")
        p = Path(out).expanduser()
        if not p.is_absolute() and runtime_base:
            p = Path(runtime_base) / p
        p = p.resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p, str(p)

    saved_base = kwargs.get("_saved_base_path")
    runtime_base = kwargs.get("_runtime_base_path")
    if saved_base:
        base = Path(saved_base)
    elif runtime_base:
        base = Path(runtime_base) / "saved"
    else:
        base = Path.cwd()
    dest = base / "vision" / default_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest, str(dest)


def _imread(path: Path, flags: int | None = None):
    """Read an image with OpenCV. Returns an ndarray or None."""
    img = _cv2.imread(str(path), _cv2.IMREAD_COLOR if flags is None else flags)
    return img


def _gray(img):
    if img.ndim == 2:
        return img
    return _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)


# ── perceptual hashing (dHash) ─────────────────────────────────────────────────


def _dhash(gray, hash_size: int = 8) -> int:
    """64-bit difference hash of a grayscale image."""
    resized = _cv2.resize(gray, (hash_size + 1, hash_size), interpolation=_cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(bool(b))
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedupe_frame_indices(image_paths: list[Path], max_hamming: int = _DEDUPE_MAX_HAMMING) -> list[int]:
    """Return the indices of frames to KEEP, collapsing runs of near-duplicates.

    Sequential dedupe (compare each frame to the last KEPT one): ideal for video
    frames sampled in time order. Safe to call from ``video_vision``; returns all
    indices unchanged when OpenCV is unavailable.
    """
    if not _HAS_CV2 or not image_paths:
        return list(range(len(image_paths)))
    keep: list[int] = []
    last_hash: int | None = None
    for i, p in enumerate(image_paths):
        try:
            img = _cv2.imread(str(p), _cv2.IMREAD_GRAYSCALE)
            if img is None:
                keep.append(i)
                continue
            h = _dhash(img)
        except Exception:
            keep.append(i)
            continue
        if last_hash is None or _hamming(h, last_hash) > max_hamming:
            keep.append(i)
            last_hash = h
    return keep or list(range(len(image_paths)))


# ── SSIM (structural similarity) ───────────────────────────────────────────────


def _ssim(gray_a, gray_b) -> float:
    """Mean SSIM over two same-shape grayscale images (Wang et al. 2004)."""
    a = gray_a.astype(_np.float64)
    b = gray_b.astype(_np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    k = (11, 11)
    s = 1.5
    mu_a = _cv2.GaussianBlur(a, k, s)
    mu_b = _cv2.GaussianBlur(b, k, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a2 = _cv2.GaussianBlur(a * a, k, s) - mu_a2
    sig_b2 = _cv2.GaussianBlur(b * b, k, s) - mu_b2
    sig_ab = _cv2.GaussianBlur(a * b, k, s) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    ssim_map = num / den
    return float(ssim_map.mean())


# ── box helpers ────────────────────────────────────────────────────────────────


def _parse_box(box: Any) -> tuple[int, int, int, int] | None:
    """Accept [x,y,w,h] or {'x','y','w','h'} or {'x1','y1','x2','y2'}."""
    try:
        if isinstance(box, dict):
            if all(k in box for k in ("x", "y", "w", "h")):
                return int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
            if all(k in box for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = (int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
                return x1, y1, x2 - x1, y2 - y1
            return None
        seq = list(box)
        if len(seq) == 4:
            return int(seq[0]), int(seq[1]), int(seq[2]), int(seq[3])
    except (TypeError, ValueError):
        return None
    return None


class VisionTool(Tool):
    """Deterministic, local computer-vision operations (OpenCV). No LLM spend."""

    name = "vision"
    timeout_seconds = 120.0
    description = (
        "Local, deterministic computer vision (OpenCV) — the cheap pixel-exact layer "
        "under image_vision/image_ocr. Spends NO tokens. Operations (op=...): "
        "'diff' (SSIM + changed-region boxes between two images), "
        "'dedupe' (group near-duplicate images by perceptual hash), "
        "'measure' (size, dominant colors, blur/sharpness, brightness), "
        "'prep' (deskew/grayscale/threshold/denoise/enhance/crop an image, e.g. before OCR), "
        "'qr' (decode QR codes and barcodes), "
        "'locate' (find a small template image inside a larger one → coordinates), "
        "'annotate' (draw boxes/labels on an image), "
        "'keyframes' (pick scene-change frames of a video). "
        "Paths may be real files or vfs: paths; derived images are written to the VFS/saved."
    )
    parameters = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": list(_OPS), "description": "Which operation to run."},
            "path": {"type": "string", "description": "Input image/video path (diff/measure/prep/qr/keyframes: 'a' or 'path')."},
            "a": {"type": "string", "description": "diff: first image path."},
            "b": {"type": "string", "description": "diff: second image path."},
            "paths": {"type": "array", "items": {"type": "string"}, "description": "dedupe: list of image paths."},
            "image": {"type": "string", "description": "locate: the haystack (larger image) path."},
            "template": {"type": "string", "description": "locate: the needle (template) image path."},
            "out": {"type": "string", "description": "Optional output path for derived images (real or vfs:). Auto-named if omitted."},
            "steps": {
                "type": "array",
                "items": {"type": "string", "enum": ["grayscale", "deskew", "threshold", "denoise", "enhance", "autocrop", "crop"]},
                "description": "prep: ordered steps. Default ['grayscale','deskew','threshold'] (good for OCR).",
            },
            "box": {"type": "array", "items": {"type": "number"}, "description": "prep+crop: [x,y,w,h] crop rectangle."},
            "boxes": {"type": "array", "items": {"type": "array"}, "description": "annotate: list of [x,y,w,h] rectangles."},
            "labels": {"type": "array", "items": {"type": "string"}, "description": "annotate: optional label per box."},
            "threshold": {"type": "number", "description": "dedupe: max Hamming distance (default 6). locate: min confidence 0-1 (default 0.75)."},
            "max_frames": {"type": "integer", "description": "keyframes: cap on returned frames (default 12)."},
        },
        "required": ["op"],
    }

    async def execute(self, op: str = "", **kwargs: Any) -> ToolResult:
        op = str(op or "").strip().lower()
        if op not in _OPS:
            return ToolResult(success=False, error=f"Unknown op '{op}'. Use one of: {', '.join(_OPS)}.")
        if not _HAS_CV2:
            return ToolResult(success=False, error=_NO_CV2)
        handler = getattr(self, f"_op_{op}")
        try:
            return await asyncio.to_thread(handler, kwargs)
        except Exception as exc:  # keep the agent moving; surface the real reason
            log.warning("vision op failed", op=op, error=str(exc))
            return ToolResult(success=False, error=f"vision '{op}' failed: {exc}")

    # ── ops (each runs in a worker thread; all sync below this line) ────────────

    def _op_diff(self, kw: dict[str, Any]) -> ToolResult:
        a_path, err = _resolve_input(kw.get("a") or kw.get("path"), kw)
        if err:
            return ToolResult(success=False, error=f"diff: image 'a': {err}")
        b_path, err = _resolve_input(kw.get("b"), kw)
        if err:
            return ToolResult(success=False, error=f"diff: image 'b': {err}")
        ia, ib = _imread(a_path), _imread(b_path)
        if ia is None or ib is None:
            return ToolResult(success=False, error="diff: could not decode one of the images.")
        if ia.shape[:2] != ib.shape[:2]:
            ib = _cv2.resize(ib, (ia.shape[1], ia.shape[0]), interpolation=_cv2.INTER_AREA)
        ga, gb = _gray(ia), _gray(ib)
        score = _ssim(ga, gb)
        # Changed-region boxes: threshold the absolute difference, then contour.
        d = _cv2.absdiff(ga, gb)
        _, thresh = _cv2.threshold(d, 25, 255, _cv2.THRESH_BINARY)
        thresh = _cv2.dilate(thresh, _np.ones((5, 5), _np.uint8), iterations=2)
        contours, _ = _cv2.findContours(thresh, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        h, w = ga.shape[:2]
        min_area = max(64, (w * h) // 2000)
        boxes: list[list[int]] = []
        for c in contours:
            if _cv2.contourArea(c) < min_area:
                continue
            x, y, bw, bh = _cv2.boundingRect(c)
            boxes.append([int(x), int(y), int(bw), int(bh)])
        boxes.sort(key=lambda bx: bx[2] * bx[3], reverse=True)
        changed_pct = round(100.0 * float((thresh > 0).sum()) / float(w * h), 2)

        out_display = ""
        if kw.get("out") is not None or boxes:
            annotated = ib.copy() if ib.ndim == 3 else _cv2.cvtColor(ib, _cv2.COLOR_GRAY2BGR)
            for (x, y, bw, bh) in boxes:
                _cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            dest, out_display = _resolve_output(kw.get("out"), kw, f"diff_{uuid.uuid4().hex[:8]}.png")
            _cv2.imwrite(str(dest), annotated)

        payload = {
            "ssim": round(score, 4),
            "identical": score >= 0.999 and not boxes,
            "changed_pct": changed_pct,
            "changed_regions": boxes[:50],
            "region_count": len(boxes),
        }
        if out_display:
            payload["annotated"] = out_display
        verdict = "identical" if payload["identical"] else f"{len(boxes)} changed region(s), {changed_pct}% of pixels"
        return ToolResult(success=True, content=f"diff: SSIM={payload['ssim']} — {verdict}\n{json.dumps(payload, indent=2)}")

    def _op_dedupe(self, kw: dict[str, Any]) -> ToolResult:
        paths_in = kw.get("paths") or []
        if not isinstance(paths_in, list) or not paths_in:
            return ToolResult(success=False, error="dedupe: pass 'paths' — a list of image paths.")
        max_h = int(kw.get("threshold") if kw.get("threshold") is not None else _DEDUPE_MAX_HAMMING)
        items: list[tuple[str, int]] = []  # (display_path, hash)
        errors: list[str] = []
        for p in paths_in:
            rp, err = _resolve_input(p, kw)
            if err:
                errors.append(f"{p}: {err}")
                continue
            img = _cv2.imread(str(rp), _cv2.IMREAD_GRAYSCALE)
            if img is None:
                errors.append(f"{p}: could not decode")
                continue
            items.append((str(p), _dhash(img)))
        # Greedy clustering by Hamming distance to the cluster's first member.
        clusters: list[dict[str, Any]] = []
        for disp, h in items:
            placed = False
            for cl in clusters:
                if _hamming(h, cl["_hash"]) <= max_h:
                    cl["members"].append(disp)
                    placed = True
                    break
            if not placed:
                clusters.append({"_hash": h, "representative": disp, "members": [disp]})
        unique = [{"representative": c["representative"], "members": c["members"]} for c in clusters]
        dupes_removed = len(items) - len(unique)
        payload = {
            "input_count": len(paths_in),
            "unique_count": len(unique),
            "duplicates_removed": dupes_removed,
            "keep": [c["representative"] for c in unique],
            "clusters": [c for c in unique if len(c["members"]) > 1],
        }
        if errors:
            payload["errors"] = errors
        return ToolResult(
            success=True,
            content=f"dedupe: {len(paths_in)} → {len(unique)} unique ({dupes_removed} near-duplicate(s)).\n"
            + json.dumps(payload, indent=2),
        )

    def _op_measure(self, kw: dict[str, Any]) -> ToolResult:
        rp, err = _resolve_input(kw.get("path") or kw.get("a"), kw)
        if err:
            return ToolResult(success=False, error=f"measure: {err}")
        img = _imread(rp)
        if img is None:
            return ToolResult(success=False, error="measure: could not decode the image.")
        h, w = img.shape[:2]
        gray = _gray(img)
        blur = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())
        brightness = round(float(gray.mean()) / 255.0, 3)
        # Dominant colors via k-means on a downscaled copy (keep it fast).
        small = _cv2.resize(img, (128, max(1, int(128 * h / w))) if w >= h else (max(1, int(128 * w / h)), 128),
                            interpolation=_cv2.INTER_AREA)
        pixels = small.reshape(-1, 3).astype(_np.float32)
        k = min(5, len(_np.unique(pixels, axis=0)))
        colors: list[dict[str, Any]] = []
        if k >= 1:
            criteria = (_cv2.TERM_CRITERIA_EPS + _cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = _cv2.kmeans(pixels, k, None, criteria, 3, _cv2.KMEANS_PP_CENTERS)
            counts = _np.bincount(labels.flatten(), minlength=k)
            order = _np.argsort(-counts)
            for idx in order:
                b, g, r = centers[idx]
                colors.append({
                    "hex": f"#{int(r):02x}{int(g):02x}{int(b):02x}",
                    "share": round(float(counts[idx]) / float(len(labels)), 3),
                })
        payload = {
            "width": w,
            "height": h,
            "channels": int(img.shape[2]) if img.ndim == 3 else 1,
            "megapixels": round(w * h / 1e6, 2),
            "file_bytes": rp.stat().st_size,
            "blur_score": round(blur, 1),
            "is_blurry": blur < 100.0,
            "brightness": brightness,
            "is_blank": bool(gray.std() < 3.0),
            "dominant_colors": colors,
        }
        tags = []
        if payload["is_blurry"]:
            tags.append("blurry")
        if payload["is_blank"]:
            tags.append("blank/uniform")
        tag_s = f" [{', '.join(tags)}]" if tags else ""
        return ToolResult(success=True, content=f"measure: {w}×{h}, blur={payload['blur_score']}, brightness={brightness}{tag_s}\n"
                          + json.dumps(payload, indent=2))

    def _op_prep(self, kw: dict[str, Any]) -> ToolResult:
        rp, err = _resolve_input(kw.get("path") or kw.get("a"), kw)
        if err:
            return ToolResult(success=False, error=f"prep: {err}")
        img = _imread(rp)
        if img is None:
            return ToolResult(success=False, error="prep: could not decode the image.")
        steps = kw.get("steps") or ["grayscale", "deskew", "threshold"]
        applied: list[str] = []
        cur = img
        for step in steps:
            step = str(step).lower()
            try:
                if step == "grayscale":
                    cur = _gray(cur)
                elif step == "deskew":
                    cur = self._deskew(cur)
                elif step == "threshold":
                    g = _gray(cur)
                    cur = _cv2.threshold(g, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)[1]
                elif step == "denoise":
                    cur = _cv2.fastNlMeansDenoising(_gray(cur), None, 10, 7, 21) if cur.ndim == 2 else _cv2.fastNlMeansDenoisingColored(cur, None, 10, 10, 7, 21)
                elif step == "enhance":
                    g = _gray(cur)
                    cur = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
                elif step == "autocrop":
                    cur = self._autocrop(cur)
                elif step == "crop":
                    box = _parse_box(kw.get("box"))
                    if box is None:
                        return ToolResult(success=False, error="prep+crop: pass 'box' as [x,y,w,h].")
                    x, y, bw, bh = box
                    cur = cur[max(0, y):y + bh, max(0, x):x + bw]
                else:
                    return ToolResult(success=False, error=f"prep: unknown step '{step}'.")
                applied.append(step)
            except Exception as exc:
                return ToolResult(success=False, error=f"prep step '{step}' failed: {exc}")
        dest, out_display = _resolve_output(kw.get("out"), kw, f"prep_{rp.stem}_{uuid.uuid4().hex[:6]}.png")
        _cv2.imwrite(str(dest), cur)
        return ToolResult(success=True, content=f"prep: applied [{', '.join(applied)}] → {out_display}")

    @staticmethod
    def _deskew(img):
        gray = _gray(img)
        thr = _cv2.threshold(gray, 0, 255, _cv2.THRESH_BINARY_INV + _cv2.THRESH_OTSU)[1]
        coords = _cv2.findNonZero(thr)
        if coords is None:
            return img
        angle = _cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.3:
            return img  # already straight; skip a needless resample
        h, w = img.shape[:2]
        m = _cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        border = _cv2.BORDER_REPLICATE
        return _cv2.warpAffine(img, m, (w, h), flags=_cv2.INTER_CUBIC, borderMode=border)

    @staticmethod
    def _autocrop(img):
        gray = _gray(img)
        thr = _cv2.threshold(gray, 0, 255, _cv2.THRESH_BINARY_INV + _cv2.THRESH_OTSU)[1]
        coords = _cv2.findNonZero(thr)
        if coords is None:
            return img
        x, y, w, h = _cv2.boundingRect(coords)
        pad = 4
        y0, x0 = max(0, y - pad), max(0, x - pad)
        return img[y0:y + h + pad, x0:x + w + pad]

    def _op_qr(self, kw: dict[str, Any]) -> ToolResult:
        rp, err = _resolve_input(kw.get("path") or kw.get("a"), kw)
        if err:
            return ToolResult(success=False, error=f"qr: {err}")
        img = _imread(rp)
        if img is None:
            return ToolResult(success=False, error="qr: could not decode the image.")
        found: list[dict[str, Any]] = []
        # QR codes (built into core OpenCV).
        try:
            det = _cv2.QRCodeDetector()
            ok, infos, points, _ = det.detectAndDecodeMulti(img)
            if ok and infos is not None:
                for i, text in enumerate(infos):
                    if not text:
                        continue
                    box = None
                    if points is not None and i < len(points):
                        pts = points[i].reshape(-1, 2)
                        x, y, w, h = _cv2.boundingRect(pts.astype(_np.int32))
                        box = [int(x), int(y), int(w), int(h)]
                    found.append({"type": "qr", "text": str(text), "box": box})
        except Exception as exc:
            log.warning("qr: QR detection failed", error=str(exc))
        # Barcodes — only if this build ships the barcode module.
        barcode_mod = getattr(_cv2, "barcode", None)
        if barcode_mod is not None:
            try:
                bd = barcode_mod.BarcodeDetector()
                ok, decoded, types, corners = bd.detectAndDecodeMulti(img)
                if ok and decoded is not None:
                    for i, text in enumerate(decoded):
                        if not text:
                            continue
                        btype = str(types[i]) if types is not None and i < len(types) else ""
                        found.append({"type": f"barcode:{btype}" if btype else "barcode", "text": str(text)})
            except Exception as exc:
                log.warning("qr: barcode detection failed", error=str(exc))
        if not found:
            note = "" if barcode_mod is not None else " (barcode module not in this OpenCV build; QR only)"
            return ToolResult(success=True, content=f"qr: no codes found{note}.")
        lines = [f"- {f['type']}: {f['text']}" for f in found]
        return ToolResult(success=True, content=f"qr: {len(found)} code(s) decoded:\n" + "\n".join(lines)
                          + "\n" + json.dumps({"codes": found}, indent=2))

    def _op_locate(self, kw: dict[str, Any]) -> ToolResult:
        hay_path, err = _resolve_input(kw.get("image") or kw.get("path"), kw)
        if err:
            return ToolResult(success=False, error=f"locate: haystack 'image': {err}")
        tpl_path, err = _resolve_input(kw.get("template"), kw)
        if err:
            return ToolResult(success=False, error=f"locate: 'template': {err}")
        hay = _cv2.imread(str(hay_path), _cv2.IMREAD_GRAYSCALE)
        tpl = _cv2.imread(str(tpl_path), _cv2.IMREAD_GRAYSCALE)
        if hay is None or tpl is None:
            return ToolResult(success=False, error="locate: could not decode image/template.")
        th, tw = tpl.shape[:2]
        if th > hay.shape[0] or tw > hay.shape[1]:
            return ToolResult(success=False, error="locate: template is larger than the image.")
        min_conf = float(kw.get("threshold") if kw.get("threshold") is not None else _LOCATE_MIN_CONF)
        res = _cv2.matchTemplate(hay, tpl, _cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = _cv2.minMaxLoc(res)
        # Collect non-overlapping matches above threshold (greedy suppression).
        matches: list[dict[str, Any]] = []
        work = res.copy()
        for _ in range(20):
            _, mv, _, ml = _cv2.minMaxLoc(work)
            if mv < min_conf:
                break
            x, y = int(ml[0]), int(ml[1])
            matches.append({"box": [x, y, int(tw), int(th)], "confidence": round(float(mv), 4),
                            "center": [x + tw // 2, y + th // 2]})
            y0, y1 = max(0, y - th // 2), min(work.shape[0], y + th // 2)
            x0, x1 = max(0, x - tw // 2), min(work.shape[1], x + tw // 2)
            work[y0:y1, x0:x1] = 0.0
        payload = {"best_confidence": round(float(max_v), 4), "min_conf": min_conf,
                   "match_count": len(matches), "matches": matches}
        if not matches:
            return ToolResult(success=True, content=f"locate: no match ≥ {min_conf} (best {payload['best_confidence']}).\n"
                              + json.dumps(payload, indent=2))
        best = matches[0]
        return ToolResult(success=True, content=f"locate: {len(matches)} match(es); best at center {best['center']} "
                          f"(conf {best['confidence']}).\n" + json.dumps(payload, indent=2))

    def _op_annotate(self, kw: dict[str, Any]) -> ToolResult:
        rp, err = _resolve_input(kw.get("path") or kw.get("image") or kw.get("a"), kw)
        if err:
            return ToolResult(success=False, error=f"annotate: {err}")
        img = _imread(rp)
        if img is None:
            return ToolResult(success=False, error="annotate: could not decode the image.")
        if img.ndim == 2:
            img = _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
        boxes_in = kw.get("boxes") or []
        labels = kw.get("labels") or []
        if not isinstance(boxes_in, list) or not boxes_in:
            return ToolResult(success=False, error="annotate: pass 'boxes' — a list of [x,y,w,h].")
        drawn = 0
        for i, raw in enumerate(boxes_in):
            box = _parse_box(raw)
            if box is None:
                continue
            x, y, w, h = box
            _cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = str(labels[i]) if i < len(labels) else ""
            if label:
                _cv2.putText(img, label, (x, max(12, y - 6)), _cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, _cv2.LINE_AA)
            drawn += 1
        dest, out_display = _resolve_output(kw.get("out"), kw, f"annot_{rp.stem}_{uuid.uuid4().hex[:6]}.png")
        _cv2.imwrite(str(dest), img)
        return ToolResult(success=True, content=f"annotate: drew {drawn} box(es) → {out_display}")

    def _op_keyframes(self, kw: dict[str, Any]) -> ToolResult:
        rp, err = _resolve_input(kw.get("path") or kw.get("a"), kw)
        if err:
            return ToolResult(success=False, error=f"keyframes: {err}")
        if rp.suffix.lower() not in _VIDEO_EXTS:
            return ToolResult(success=False, error=f"keyframes: not a video ('{rp.suffix}').")
        max_frames = int(kw.get("max_frames") or 12)
        cap = _cv2.VideoCapture(str(rp))
        if not cap.isOpened():
            return ToolResult(success=False, error="keyframes: could not open the video.")
        try:
            fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
            # Sample at ~2 fps (enough to catch cuts) and score histogram changes.
            step = max(1, int(round(fps / 2.0)))
            prev_hist = None
            scored: list[tuple[float, float]] = []  # (timestamp_s, change_score)
            idx = 0
            while True:
                ok = cap.grab()
                if not ok:
                    break
                if idx % step == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        hsv = _cv2.cvtColor(frame, _cv2.COLOR_BGR2HSV)
                        hist = _cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                        _cv2.normalize(hist, hist)
                        ts = idx / fps if fps else float(idx)
                        if prev_hist is None:
                            scored.append((ts, 1.0))
                        else:
                            corr = _cv2.compareHist(prev_hist, hist, _cv2.HISTCMP_CORREL)
                            scored.append((ts, 1.0 - float(corr)))
                        prev_hist = hist
                idx += 1
                if total and idx > total:
                    break
        finally:
            cap.release()
        if not scored:
            return ToolResult(success=False, error="keyframes: no frames could be read.")
        # Always keep the first; then the biggest scene-change moments.
        first = scored[0]
        rest = sorted(scored[1:], key=lambda p: p[1], reverse=True)
        picked = [first] + rest[: max(0, max_frames - 1)]
        timestamps = sorted(round(ts, 2) for ts, _ in picked)
        payload = {"sampled": len(scored), "keyframes": len(timestamps), "timestamps_s": timestamps}
        return ToolResult(success=True, content=f"keyframes: {len(timestamps)} scene-change frame(s) from {len(scored)} samples.\n"
                          + json.dumps(payload, indent=2))
