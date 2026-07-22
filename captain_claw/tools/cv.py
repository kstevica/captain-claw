"""cv — deterministic, local computer-vision ops backed by OpenCV.

Named `cv` (not `vision`) to avoid confusion with `image_vision` (the multimodal
LLM that *describes/reads* an image). This tool never reads or describes — it only
measures and manipulates pixels. See the tool description for the full split.

This is the cheap, pixel-exact layer *under* the multimodal-LLM vision tools
(``image_vision`` / ``image_ocr`` / ``video_vision``). It spends no tokens: every
operation here runs locally on CPU. Use it to pre-process and measure for the LLM
(dedupe/keyframe/diff-gate → fewer LLM frames) and to do things an LLM does poorly
(pixel diffs, QR/barcode decode, geometric measurement, template matching).

Operations:

  diff       two images → SSIM score + bounding boxes of changed regions
  dedupe     images → perceptual-hash near-duplicate clusters
  measure    dimensions, dominant colors, blur score, brightness
  prep       deskew / grayscale / threshold / denoise / enhance / crop  (pre-OCR)
  qr         decode QR codes (+ barcodes when available)
  locate     template-match a needle image in a haystack → coords + confidence
  annotate   draw boxes/labels on an image → new image
  keyframes  video → scene-change representative-frame timestamps
  detect     faces / text-regions / objects via small local ONNX models  (Phase 2)

Phase 1 is classical CV only (no models). Phase 2 adds ``detect``: small, CPU-only
ONNX detectors run through OpenCV's DNN engine — still no LLM/VLM inference and no
token spend. Face (YuNet, ~232 KB) and text-region (PP-OCRv3 DB, ~2.4 MB) models are
fetched once from the OpenCV Zoo into a local cache (``CAPTAIN_CLAW_VISION_MODELS`` or
an ``fd-data/models`` dir — prod can pre-place them offline); ``objects`` is
bring-your-own YOLOv8/v5 ONNX (a COCO model worth bundling is heavier than we want on).

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
    "cv requires OpenCV, which isn't installed. Install the extra: "
    "pip install 'captain-claw[cv]' (or pip install opencv-python-headless). "
    "This is a local, CPU-only dependency."
)

_OPS = ("diff", "dedupe", "measure", "prep", "qr", "locate", "annotate", "keyframes", "detect")

# Default perceptual-hash Hamming distance under which two frames are "the same".
_DEDUPE_MAX_HAMMING = 6
# Default template-match confidence (TM_CCOEFF_NORMED) to accept a hit.
_LOCATE_MIN_CONF = 0.75

# ── Phase 2: ONNX detectors (OpenCV DNN) ───────────────────────────────────────
# Small, CPU-only models from the OpenCV Zoo. The zoo stores weights in git-lfs, so
# only the media.githubusercontent.com/media/ path returns the real ONNX (a plain
# raw.githubusercontent URL returns a ~130-byte LFS pointer). Kept tiny on purpose —
# "light on resources". `objects` is bring-your-own-model (see _op_detect): a COCO
# object detector worth bundling is heavier than we want on by default.
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "min_bytes": 100_000,  # real file ~232 KB; guards against an LFS pointer
    },
    "ppocr_db": {
        "filename": "text_detection_en_ppocrv3_2023may.onnx",
        "url": _ZOO + "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
        "min_bytes": 1_000_000,  # real file ~2.4 MB
    },
}
_DETECT_WHAT = ("faces", "text", "objects")
_DEFAULT_FACE_CONF = 0.85
_DEFAULT_TEXT_CONF = 0.5
_DEFAULT_OBJ_CONF = 0.25
_OBJ_NMS = 0.45
# COCO-80 class names (YOLOv5/v8 ordering) for the bring-your-own object detector.
_COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


# ── path resolution (accepts both real paths and vfs: paths) ───────────────────


def _resolve_input(path: str, kwargs: dict[str, Any]) -> tuple[Path | None, str | None]:
    """Resolve an input file that may be a real path, a ``vfs:`` path, or a
    Google Drive placeholder.

    cv ops run in a worker thread (``asyncio.to_thread``) with no event loop, so
    a placeholder's bytes are fetched through the *sync* materialise bridge; an
    async tool would use ``vfs_drive.materialize`` instead. A Drive fetch failure
    is surfaced as a clear message; anything unexpected fails open to the resolved
    path (so a non-Drive file, or a Drive subsystem that can't load, still reads).
    """
    raw = str(path or "").strip()
    if not raw:
        return None, "Missing path"
    file_path: Path | None = None
    try:
        from captain_claw.vfs import is_vfs_path, resolve_vfs_path

        if is_vfs_path(raw):
            p = resolve_vfs_path(raw)
            if p is None:
                return None, f"Could not resolve VFS path: {raw}"
            if not p.is_file():
                return None, f"File not found: {raw}"
            file_path = p
    except Exception:  # pragma: no cover — vfs module should always import
        pass
    if file_path is None:
        from captain_claw.tools.document_extract import _require_existing_file

        fp, err = _require_existing_file(raw, runtime_base_path=kwargs.get("_runtime_base_path"))
        if err:
            return None, err
        file_path = fp

    # Google Drive placeholder → fetch its real bytes so OpenCV decodes the
    # actual image, not the marker text.
    try:
        from captain_claw.drive_client import DriveError
        from captain_claw.vfs_drive import materialize_sync

        try:
            real = materialize_sync(file_path)
            if real is not None:
                file_path = real
        except DriveError as exc:
            return None, (
                f"'{file_path.name}' lives in Google Drive and its content could "
                f"not be fetched: {exc}"
            )
    except Exception:  # Drive subsystem unavailable → treat as a local file
        pass
    return file_path, None


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


# ── model cache / provisioning (Phase 2 detectors) ─────────────────────────────


def _models_dir() -> Path:
    """Where detector models live. Prod can pre-place them (offline-friendly).

    Resolution order: ``CAPTAIN_CLAW_VISION_MODELS`` env → an ``fd-data`` ancestor
    (shared with the VFS/fd-data root) → ``~/.captain-claw/models/vision``.
    """
    import os

    override = os.environ.get("CAPTAIN_CLAW_VISION_MODELS", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from captain_claw.vfs import vfs_base

        base = vfs_base()  # …/fd-data/vfs → models under fd-data/models/vision
        if base is not None:
            return Path(base).parent / "models" / "vision"
    except Exception:
        pass
    return Path.home() / ".captain-claw" / "models" / "vision"


def _download(url: str, dest: Path, *, min_bytes: int, timeout: float = 60.0) -> str | None:
    """Best-effort download to *dest*. Returns None on success, else an error string."""
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "captain-claw-vision"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception as exc:
        return f"download failed: {exc}"
    if data[:40].startswith(b"version https://git-lfs"):
        return "download returned a git-lfs pointer, not the model (bad URL)"
    if len(data) < min_bytes:
        return f"downloaded file too small ({len(data)} bytes < {min_bytes}); likely incomplete"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)  # atomic
    except Exception as exc:
        return f"could not save model: {exc}"
    return None


def _ensure_model(key: str) -> tuple[Path | None, str | None]:
    """Return a local path to model *key*, downloading it once if needed."""
    spec = _MODEL_REGISTRY.get(key)
    if spec is None:
        return None, f"unknown model '{key}'"
    dest = _models_dir() / spec["filename"]
    if dest.is_file() and dest.stat().st_size >= spec["min_bytes"]:
        return dest, None
    log.info("cv: fetching detector model", model=key, dest=str(dest))
    err = _download(spec["url"], dest, min_bytes=spec["min_bytes"])
    if err:
        return None, (
            f"{err}. Place the model manually at {dest} (or set "
            f"CAPTAIN_CLAW_VISION_MODELS). Source: {spec['url']}"
        )
    return dest, None


class CvTool(Tool):
    """Deterministic, local computer-vision operations (OpenCV). No LLM spend."""

    name = "cv"
    timeout_seconds = 120.0
    description = (
        "Local, deterministic OpenCV pixel/geometry operations. Spends NO tokens. "
        "NOT for looking at, reading, or understanding a picture — to DESCRIBE an image "
        "or answer questions about what it shows use image_vision; to READ the text in "
        "an image use image_ocr. This tool cannot read text or say what an image depicts; "
        "it only measures and manipulates pixels. Operations (op=...): "
        "'diff' (SSIM + changed-region boxes between two images), "
        "'dedupe' (group near-duplicate images by perceptual hash), "
        "'measure' (size, dominant colors, blur/sharpness, brightness), "
        "'prep' (deskew/grayscale/threshold/denoise/enhance/crop an image, e.g. before OCR), "
        "'qr' (decode QR codes and barcodes), "
        "'locate' (find a small template image inside a larger one → coordinates), "
        "'annotate' (draw boxes/labels on an image), "
        "'keyframes' (pick scene-change frames of a video), "
        "'detect' (find WHERE faces/text/objects are → boxes; for text this gives regions, "
        "NOT the words — use image_ocr to read them). "
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
            "threshold": {"type": "number", "description": "dedupe: max Hamming distance (default 6). locate: min confidence 0-1 (default 0.75). detect: min confidence."},
            "max_frames": {"type": "integer", "description": "keyframes: cap on returned frames (default 12)."},
            "what": {"type": "string", "enum": list(_DETECT_WHAT), "description": "detect: 'faces', 'text' (text regions, feeds image_ocr), or 'objects' (COCO)."},
            "model": {"type": "string", "description": "detect+objects: path to a YOLOv8/v5 ONNX model (required for 'objects'; faces/text auto-download)."},
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
            log.warning("cv op failed", op=op, error=str(exc))
            return ToolResult(success=False, error=f"cv '{op}' failed: {exc}")

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

    # ── detect (Phase 2: small local ONNX detectors) ───────────────────────────

    def _op_detect(self, kw: dict[str, Any]) -> ToolResult:
        what = str(kw.get("what") or "").strip().lower()
        if what not in _DETECT_WHAT:
            return ToolResult(success=False, error=f"detect: pass what={'/'.join(_DETECT_WHAT)}.")
        rp, err = _resolve_input(kw.get("path") or kw.get("image") or kw.get("a"), kw)
        if err:
            return ToolResult(success=False, error=f"detect: {err}")
        img = _imread(rp)
        if img is None:
            return ToolResult(success=False, error="detect: could not decode the image.")
        thr = kw.get("threshold")
        if what == "faces":
            dets, derr = self._detect_faces(img, float(thr) if thr is not None else _DEFAULT_FACE_CONF)
        elif what == "text":
            dets, derr = self._detect_text(img, float(thr) if thr is not None else _DEFAULT_TEXT_CONF)
        else:
            dets, derr = self._detect_objects(img, kw, float(thr) if thr is not None else _DEFAULT_OBJ_CONF)
        if derr:
            return ToolResult(success=False, error=f"detect ({what}): {derr}")

        out_display = ""
        if kw.get("out") is not None:
            annotated = self._draw_detections(img, dets)
            dest, out_display = _resolve_output(kw.get("out"), kw, f"detect_{what}_{rp.stem}_{uuid.uuid4().hex[:6]}.png")
            _cv2.imwrite(str(dest), annotated)

        # Summarize by label (e.g. "3 person, 1 dog") for objects; count otherwise.
        counts: dict[str, int] = {}
        for d in dets:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        summary = ", ".join(f"{n} {lbl}" for lbl, n in sorted(counts.items(), key=lambda kv: -kv[1])) or "nothing"
        payload: dict[str, Any] = {"what": what, "count": len(dets), "detections": dets[:100]}
        if out_display:
            payload["annotated"] = out_display
        return ToolResult(success=True, content=f"detect ({what}): {len(dets)} — {summary}.\n" + json.dumps(payload, indent=2))

    @staticmethod
    def _detect_faces(img, conf: float) -> tuple[list[dict[str, Any]], str | None]:
        model_path, err = _ensure_model("yunet")
        if err:
            return [], err
        h, w = img.shape[:2]
        det = _cv2.FaceDetectorYN_create(str(model_path), "", (w, h), conf, 0.3, 5000)
        det.setInputSize((w, h))
        _, faces = det.detect(img)
        out: list[dict[str, Any]] = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = (int(f[0]), int(f[1]), int(f[2]), int(f[3]))
                out.append({
                    "label": "face",
                    "box": [x, y, bw, bh],
                    "confidence": round(float(f[-1]), 3),
                    # 5 landmarks: right-eye, left-eye, nose, right-mouth, left-mouth
                    "landmarks": [[int(f[4 + 2 * i]), int(f[5 + 2 * i])] for i in range(5)],
                })
        return out, None

    @staticmethod
    def _detect_text(img, conf: float) -> tuple[list[dict[str, Any]], str | None]:
        model_path, err = _ensure_model("ppocr_db")
        if err:
            return [], err
        model = _cv2.dnn.TextDetectionModel_DB(str(model_path))
        model.setBinaryThreshold(0.3).setPolygonThreshold(max(0.1, conf))
        model.setInputParams(1 / 255.0, (736, 736), (122.67891434, 116.66876762, 104.00698793))
        boxes, confs = model.detect(img)
        out: list[dict[str, Any]] = []
        for i, quad in enumerate(boxes or []):
            pts = _np.array(quad, dtype=_np.int32).reshape(-1, 2)
            x, y, bw, bh = _cv2.boundingRect(pts)
            c = float(confs[i]) if confs is not None and i < len(confs) else 0.0
            out.append({"label": "text", "box": [int(x), int(y), int(bw), int(bh)],
                        "confidence": round(c, 3), "quad": pts.tolist()})
        return out, None

    def _detect_objects(self, img, kw: dict[str, Any], conf: float) -> tuple[list[dict[str, Any]], str | None]:
        # Bring-your-own YOLOv8/v5 ONNX (a COCO detector worth bundling is heavier
        # than we want on by default). Model from `model=` or the configured dir.
        model_arg = str(kw.get("model") or "").strip()
        if model_arg:
            mp, merr = _resolve_input(model_arg, kw)
            if merr:
                return [], f"model: {merr}"
            model_path = mp
        else:
            cand = _models_dir() / "yolov8n.onnx"
            if not cand.is_file():
                return [], (
                    "objects needs a YOLOv8/v5 ONNX model. Pass model=<path.onnx>, or place "
                    f"one at {cand} (or set CAPTAIN_CLAW_VISION_MODELS). Export e.g. "
                    "`yolo export model=yolov8n.pt format=onnx`."
                )
            model_path = cand
        try:
            net = _cv2.dnn.readNetFromONNX(str(model_path))
        except Exception as exc:
            return [], f"could not load model {model_path}: {exc}"
        h, w = img.shape[:2]
        size = 640
        blob = _cv2.dnn.blobFromImage(img, 1 / 255.0, (size, size), swapRB=True, crop=False)
        net.setInput(blob)
        out = net.forward()
        dets = self._decode_yolo(out, w, h, conf, _OBJ_NMS, size)
        return dets, None

    @staticmethod
    def _decode_yolo(output, orig_w: int, orig_h: int, conf: float, nms: float, size: int) -> list[dict[str, Any]]:
        """Decode a YOLOv8 ([1,84,N]) or YOLOv5 ([1,N,85]) ONNX output → detections."""
        arr = _np.squeeze(output)
        if arr.ndim != 2:
            return []
        if arr.shape[0] < arr.shape[1]:  # (84, N) → (N, 84) for v8
            arr = arr.T
        cols = arr.shape[1]
        box_xywh = arr[:, :4]  # cx,cy,w,h in model-input pixels — same for both layouts
        if cols == 84 or cols - 4 == 80:          # v8: 4 box + 80 class scores
            cls_scores = arr[:, 4:]
            confidences = cls_scores.max(axis=1)
        else:                                      # v5/v7: 4 box + 1 obj + classes
            obj = arr[:, 4]
            cls_scores = arr[:, 5:]
            confidences = cls_scores.max(axis=1) * obj
        class_ids = cls_scores.argmax(axis=1)
        keep = confidences >= conf
        if not keep.any():
            return []
        box_xywh = box_xywh[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]
        # cx,cy,w,h (model-input pixels) → x,y,w,h
        xs = (box_xywh[:, 0] - box_xywh[:, 2] / 2)
        ys = (box_xywh[:, 1] - box_xywh[:, 3] / 2)
        rects = _np.stack([xs, ys, box_xywh[:, 2], box_xywh[:, 3]], axis=1)
        idxs = _cv2.dnn.NMSBoxes(rects.tolist(), confidences.tolist(), float(conf), float(nms))
        if idxs is None or len(idxs) == 0:
            return []
        sx, sy = orig_w / float(size), orig_h / float(size)
        dets: list[dict[str, Any]] = []
        for i in _np.array(idxs).flatten():
            x, y, bw, bh = rects[i]
            cid = int(class_ids[i])
            dets.append({
                "label": _COCO_CLASSES[cid] if cid < len(_COCO_CLASSES) else f"class_{cid}",
                "box": [int(x * sx), int(y * sy), int(bw * sx), int(bh * sy)],
                "confidence": round(float(confidences[i]), 3),
            })
        return dets

    @staticmethod
    def _draw_detections(img, dets: list[dict[str, Any]]):
        canvas = img.copy() if img.ndim == 3 else _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
        for d in dets:
            box = _parse_box(d.get("box"))
            if box is None:
                continue
            x, y, bw, bh = box
            _cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            tag = f"{d.get('label', '')} {d.get('confidence', '')}".strip()
            if tag:
                _cv2.putText(canvas, tag, (x, max(12, y - 6)), _cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 0, 255), 1, _cv2.LINE_AA)
        return canvas


# ── Phase 3: public helpers for pipeline integration ───────────────────────────
# These let other tools reach the cheap CV layer without going through the tool's
# op-dispatch. All are guarded and FAIL-OPEN — absent OpenCV they behave so the
# caller's existing (LLM) path is unchanged (no regression).


def preprocess_ocr_bytes(image_bytes: bytes, *, deskew: bool = True, enhance: bool = False) -> bytes:
    """Deskew (and optionally CLAHE-enhance) an image in memory, before OCR.

    Bytes in → bytes out (PNG). Returns the input untouched when OpenCV is absent
    or anything fails, so ``image_ocr`` can call it unconditionally. Deskew helps a
    vision LLM read rotated scans; enhancement is opt-in (can hurt natural images).
    """
    if not _HAS_CV2 or not image_bytes:
        return image_bytes
    try:
        arr = _np.frombuffer(image_bytes, _np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes
        if deskew:
            img = CvTool._deskew(img)
        if enhance:
            img = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(_gray(img))
        ok, buf = _cv2.imencode(".png", img)
        return buf.tobytes() if ok else image_bytes
    except Exception:  # never let pre-processing break OCR
        return image_bytes


def images_differ(path_a: str | Path, path_b: str | Path, *, ssim_threshold: float = 0.995) -> tuple[bool, float]:
    """Return (changed, ssim) for two images. Diff-gate for watch/poll loops.

    ``changed`` is True when SSIM < *ssim_threshold*. FAIL-OPEN: when OpenCV is
    absent or a read fails, returns (True, 0.0) — i.e. "treat as changed", so a
    caller gating an LLM call still makes the call rather than silently skipping it.
    """
    if not _HAS_CV2:
        return True, 0.0
    try:
        ia = _cv2.imread(str(path_a))
        ib = _cv2.imread(str(path_b))
        if ia is None or ib is None:
            return True, 0.0
        ga, gb = _gray(ia), _gray(ib)
        if ga.shape != gb.shape:
            gb = _cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=_cv2.INTER_AREA)
        score = _ssim(ga, gb)
        return (score < ssim_threshold), float(score)
    except Exception:
        return True, 0.0
