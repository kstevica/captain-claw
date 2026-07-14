"""Tests for the deterministic OpenCV `vision` tool (Phase 1).

The pure helpers (hashing, box parsing, no-op dedupe) are tested without any
dependency. The pixel ops need OpenCV, so the cv2-backed section
``pytest.importorskip("cv2")`` — the `cv` extra is optional, and these skip
cleanly where it isn't installed (e.g. minimal CI), exactly like the faces tests.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from captain_claw.tools.vision import (
    VisionTool,
    _hamming,
    _parse_box,
)


def _run(tool: VisionTool, **kwargs):
    return asyncio.run(tool.execute(**kwargs))


# ── pure helpers (no OpenCV needed) ────────────────────────────────────────────


def test_hamming_distance():
    assert _hamming(0b0000, 0b0000) == 0
    assert _hamming(0b1011, 0b1110) == 2
    assert _hamming(0, (1 << 64) - 1) == 64


def test_parse_box_variants():
    assert _parse_box([1, 2, 3, 4]) == (1, 2, 3, 4)
    assert _parse_box({"x": 5, "y": 6, "w": 7, "h": 8}) == (5, 6, 7, 8)
    assert _parse_box({"x1": 0, "y1": 0, "x2": 10, "y2": 20}) == (0, 0, 10, 20)
    assert _parse_box([1, 2, 3]) is None
    assert _parse_box("nope") is None


def test_dedupe_frame_indices_identity_without_cv2(monkeypatch):
    # With OpenCV absent (or on missing paths), it must keep every frame — the
    # video_vision pre-pass relies on this no-op fallback to avoid regressions.
    import captain_claw.tools.vision as vision

    monkeypatch.setattr(vision, "_HAS_CV2", False)
    assert vision.dedupe_frame_indices(["a", "b", "c"]) == [0, 1, 2]
    assert vision.dedupe_frame_indices([]) == []


def test_unknown_op_is_rejected():
    res = _run(VisionTool(), op="bogus")
    assert res.success is False
    assert "Unknown op" in (res.error or "")


# ── OpenCV-backed ops ──────────────────────────────────────────────────────────

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


@pytest.fixture()
def workdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write(path, img):
    cv2.imwrite(path, img)
    return path


def _solid(w, h, color):
    return np.full((h, w, 3), color, np.uint8)


def test_diff_identical_vs_changed(workdir):
    a = _solid(300, 200, 200)
    cv2.rectangle(a, (20, 20), (80, 80), (0, 0, 255), -1)
    pa = _write(os.path.join(workdir, "a.png"), a)
    pa2 = _write(os.path.join(workdir, "a2.png"), a.copy())
    b = a.copy()
    cv2.rectangle(b, (180, 120), (260, 180), (0, 255, 0), -1)
    pb = _write(os.path.join(workdir, "b.png"), b)

    same = _run(VisionTool(), op="diff", a=pa, b=pa2)
    assert same.success and "identical" in same.content

    changed = _run(VisionTool(), op="diff", a=pa, b=pb, out=os.path.join(workdir, "d.png"))
    assert changed.success
    assert "changed region" in changed.content
    assert os.path.exists(os.path.join(workdir, "d.png"))


def test_dedupe_collapses_near_duplicates(workdir):
    a = _solid(120, 120, 210)
    cv2.circle(a, (60, 60), 20, (0, 0, 0), -1)
    pa = _write(os.path.join(workdir, "a.png"), a)
    pa2 = _write(os.path.join(workdir, "a2.png"), a.copy())
    pc = _write(os.path.join(workdir, "c.png"), _solid(120, 120, 30))

    res = _run(VisionTool(), op="dedupe", paths=[pa, pa2, pc])
    assert res.success
    assert "3 → 2 unique" in res.content


def test_measure_flags_blank(workdir):
    p = _write(os.path.join(workdir, "blank.png"), _solid(80, 80, 128))
    res = _run(VisionTool(), op="measure", path=p)
    assert res.success
    assert "blank" in res.content


def test_prep_writes_output(workdir):
    img = _solid(200, 120, 240)
    cv2.putText(img, "HELLO", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    p = _write(os.path.join(workdir, "doc.png"), img)
    out = os.path.join(workdir, "clean.png")
    res = _run(VisionTool(), op="prep", path=p, out=out, steps=["grayscale", "threshold"])
    assert res.success
    assert os.path.exists(out)


def test_locate_finds_template(workdir):
    hay = _solid(320, 240, 230)
    cv2.circle(hay, (200, 150), 18, (30, 30, 200), -1)
    cv2.rectangle(hay, (188, 138), (212, 162), (0, 0, 0), 2)
    ph = _write(os.path.join(workdir, "hay.png"), hay)
    pt = _write(os.path.join(workdir, "tpl.png"), hay[130:172, 178:222].copy())

    res = _run(VisionTool(), op="locate", image=ph, template=pt)
    assert res.success
    assert "match(es)" in res.content
    # The icon center is (200, 150); a self-crop must locate it with high conf.
    assert "conf 1.0" in res.content or "conf 0.99" in res.content


def test_annotate_draws_and_writes(workdir):
    p = _write(os.path.join(workdir, "a.png"), _solid(160, 120, 220))
    out = os.path.join(workdir, "annot.png")
    res = _run(VisionTool(), op="annotate", path=p, boxes=[[10, 10, 40, 40]], labels=["x"], out=out)
    assert res.success and os.path.exists(out)


def test_qr_decode(workdir):
    if not hasattr(cv2, "QRCodeEncoder_create"):
        pytest.skip("this OpenCV build has no QR encoder to generate a fixture")
    code = cv2.QRCodeEncoder_create().encode("captain-claw://ok")
    p = _write(os.path.join(workdir, "qr.png"),
               cv2.cvtColor(cv2.resize(code, (300, 300), interpolation=cv2.INTER_NEAREST),
                            cv2.COLOR_GRAY2BGR))
    res = _run(VisionTool(), op="qr", path=p)
    assert res.success
    assert "captain-claw://ok" in res.content


# ── Phase 2: detect ────────────────────────────────────────────────────────────


def test_models_dir_honors_env(monkeypatch, tmp_path):
    from captain_claw.tools import vision

    monkeypatch.setenv("CAPTAIN_CLAW_VISION_MODELS", str(tmp_path))
    assert vision._models_dir() == tmp_path


def test_decode_yolo_v8_layout():
    # v8 output [1, 84, N]: 4 box rows + 80 class rows, anchors in columns.
    out = np.zeros((1, 84, 8400), np.float32)
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = 320, 320, 100, 200
    out[0, 4, 0] = 0.9  # class 0 == "person"
    dets = VisionTool._decode_yolo(out, 640, 640, 0.25, 0.45, 640)
    assert len(dets) == 1
    assert dets[0]["label"] == "person"
    assert dets[0]["confidence"] == pytest.approx(0.9, abs=1e-3)
    assert dets[0]["box"] == [270, 220, 100, 200]


def test_decode_yolo_v5_layout_with_objectness():
    # v5 output [1, N, 85]: 4 box + 1 objectness + 80 classes; conf = obj * cls.
    out = np.zeros((1, 25200, 85), np.float32)
    out[0, 0, :4] = [320, 320, 80, 60]
    out[0, 0, 4] = 0.8
    out[0, 0, 5 + 2] = 0.9  # class 2 == "car"
    dets = VisionTool._decode_yolo(out, 1280, 720, 0.25, 0.45, 640)
    assert len(dets) == 1
    assert dets[0]["label"] == "car"
    assert dets[0]["confidence"] == pytest.approx(0.72, abs=1e-3)  # 0.8 * 0.9


def test_decode_yolo_below_threshold_returns_nothing():
    out = np.zeros((1, 84, 8400), np.float32)
    out[0, 4, 0] = 0.1  # under the 0.25 conf floor
    assert VisionTool._decode_yolo(out, 640, 640, 0.25, 0.45, 640) == []


def test_detect_objects_without_model_is_actionable(workdir, monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTAIN_CLAW_VISION_MODELS", str(tmp_path))
    p = _write(os.path.join(workdir, "x.png"), _solid(200, 200, 255))
    res = _run(VisionTool(), op="detect", what="objects", path=p)
    assert res.success is False
    assert "ONNX" in (res.error or "") and "model" in (res.error or "")


def test_detect_bad_what_rejected(workdir):
    p = _write(os.path.join(workdir, "x.png"), _solid(50, 50, 255))
    res = _run(VisionTool(), op="detect", what="unicorns", path=p)
    assert res.success is False


# Network-gated: downloads the small YuNet / PP-OCRv3 models from the OpenCV Zoo.
# Skips (does not fail) when offline, so CI without network stays green.


def _models_available(tmp_dir) -> bool:
    import os as _os

    _os.environ["CAPTAIN_CLAW_VISION_MODELS"] = str(tmp_dir)
    from captain_claw.tools.vision import _ensure_model

    _, e1 = _ensure_model("yunet")
    _, e2 = _ensure_model("ppocr_db")
    return e1 is None and e2 is None


def test_detect_faces_and_text_end_to_end(monkeypatch, tmp_path, workdir):
    monkeypatch.setenv("CAPTAIN_CLAW_VISION_MODELS", str(tmp_path))
    if not _models_available(tmp_path):
        pytest.skip("detector models unavailable (offline?) — skipping network test")

    # Faces: YuNet must at least load and run (0 on a blank frame is fine).
    blank = _write(os.path.join(workdir, "blank.png"), _solid(640, 480, 255))
    fr = _run(VisionTool(), op="detect", what="faces", path=blank)
    assert fr.success and "detect (faces)" in fr.content

    # Text: a rendered word must produce at least one text region + annotation.
    txt = _solid(700, 300, 255)
    cv2.putText(txt, "CAPTAIN CLAW", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    pt = _write(os.path.join(workdir, "t.png"), txt)
    out = os.path.join(workdir, "t_annot.png")
    tr = _run(VisionTool(), op="detect", what="text", path=pt, out=out)
    assert tr.success and "text" in tr.content
    assert os.path.exists(out)
