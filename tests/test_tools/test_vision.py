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
    dedupe_frame_indices,
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
