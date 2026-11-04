"""ocr_pdf / ocr_image — local OCR, no model (captain_claw/ocr/).

Most tests run on a fake engine, so they need neither Tesseract nor RapidOCR;
the last few read real pictures when an engine is installed and are skipped
otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from captain_claw.ocr import document as doc
from captain_claw.ocr import engines
from captain_claw.ocr.engines import Choice, EngineError, Line, Recognition
from captain_claw.tools.ocr import OcrImageTool, OcrPdfTool


def _kw(tmp_path: Path, session: str = "s1") -> dict:
    return {"_runtime_base_path": tmp_path, "_saved_base_path": tmp_path / "saved",
            "_session_id": session}


def _text_pdf(lines: list[str]) -> bytes:
    """A one-page PDF with a real text layer (Helvetica), built by hand."""
    ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for ln in lines:
        ops.append("(" + ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, body in enumerate(objs, start=1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def _page_picture(lines: list[str], size: int = 34) -> Image.Image:
    img = Image.new("L", (1700, 1100), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for i, ln in enumerate(lines):
        draw.text((120, 120 + i * 70), ln, fill=0, font=font)
    return img


def _scanned_pdf(path: Path, pages: list[list[str]]) -> Path:
    imgs = [_page_picture(p) for p in pages]
    imgs[0].save(path, "PDF", resolution=200, save_all=True, append_images=imgs[1:])
    return path


def _mixed_pdf(tmp_path: Path) -> Path:
    from pypdf import PdfReader, PdfWriter

    text = tmp_path / "text.pdf"
    text.write_bytes(_text_pdf(["A digital page with a real text layer.",
                                "Quarterly revenue was 1,234,567 EUR."]))
    scan = _scanned_pdf(tmp_path / "scan.pdf", [["INVOICE 2026-0412"], ["Signed in Split"]])
    w = PdfWriter()
    for f in (text, scan):
        for p in PdfReader(str(f)).pages:
            w.add_page(p)
    out = tmp_path / "mixed.pdf"
    with open(out, "wb") as fh:
        w.write(fh)
    return out


class FakeEngine:
    """Answers every picture with fixed lines and counts its calls."""

    name = "fake"
    psm = 3

    def __init__(self, text: str = "Recognised line", conf: float = 93.0):
        self.text, self.conf, self.calls = text, conf, 0

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return "1.0"

    def recognize(self, image: Path, lang: str) -> Recognition:
        self.calls += 1
        with Image.open(image) as im:
            w, h = im.size
        return Recognition(lines=[Line(self.text, self.conf, (10, 10, 400, 40), 0)], width=w, height=h)


# ── helpers ───────────────────────────────────────────────────────────────

def test_parse_pages():
    assert doc.parse_pages(None, 5) is None
    assert doc.parse_pages("1-3,5", 9) == [1, 2, 3, 5]
    assert doc.parse_pages("4-", 6) == [4, 5, 6]
    assert doc.parse_pages([2, 1, 2], 3) == [1, 2]
    assert doc.parse_pages(3, 10) == [3]
    assert doc.parse_pages("2-99", 4) == [2, 3, 4]
    with pytest.raises(ValueError):
        doc.parse_pages("x", 3)
    with pytest.raises(ValueError):
        doc.parse_pages("7", 3)


def test_text_layer_verdict_rules():
    good = "This page has plenty of ordinary words in its text layer, numbers 12.50 too."
    assert doc.text_layer_verdict(good, 0.0, 25) == "text_layer"
    assert doc.text_layer_verdict("", 0.0, 25) == "no_text_layer"
    assert doc.text_layer_verdict("Page 1", 0.0, 25) == "no_text_layer"
    assert doc.text_layer_verdict("(cid:12)(cid:15)(cid:3) " * 6, 0.0, 25) == "garbled_text_layer"
    assert doc.text_layer_verdict("���" * 20, 0.0, 25) == "garbled_text_layer"
    assert doc.text_layer_verdict("Scanned with CamScanner app today", 0.95, 25) == "scan_with_thin_text"
    assert doc.text_layer_verdict(good * 4, 0.95, 25) == "text_layer"


def test_gap_join_spaces_columns():
    assert engines.gap_join([("Total", 0, 50), ("12.00", 60, 110)]) == "Total 12.00"
    assert engines.gap_join([("12.00", 400, 450), ("Total", 0, 50)]) == "Total   12.00"


def test_parse_tesseract_tsv():
    tsv = "\n".join([
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "1\t1\t0\t0\t0\t0\t0\t0\t1000\t800\t-1\t",
        "5\t1\t1\t1\t1\t1\t10\t10\t60\t20\t96\tTotal",
        "5\t1\t1\t1\t1\t2\t80\t10\t50\t20\t90\tdue",
        "5\t1\t2\t1\t1\t1\t10\t100\t50\t20\t60\t12.00",
        "5\t1\t2\t1\t1\t2\t70\t100\t10\t20\t-1\t",
    ])
    rec = engines.parse_tesseract_tsv(tsv)
    assert (rec.width, rec.height) == (1000, 800)
    assert [ln.text for ln in rec.lines] == ["Total due", "12.00"]
    assert rec.lines[0].box == (10, 10, 130, 30)
    assert rec.lines[0].para != rec.lines[1].para
    assert rec.text == "Total due\n\n12.00"
    assert rec.lines[0].conf == pytest.approx((96 * 5 + 90 * 3) / 8)
    assert rec.confidence == pytest.approx((93.75 * 9 + 60 * 5) / 14)  # weighted by line length


def test_rows_to_lines_orders_and_joins():
    items = [
        ("240.50", 500, 102, 600, 130, 99.0),
        ("Travel", 10, 100, 120, 128, 98.0),
        ("Header", 10, 10, 200, 40, 97.0),
        ("Far below", 10, 400, 200, 430, 95.0),
    ]
    lines = engines.rows_to_lines(items)
    assert [ln.text for ln in lines] == ["Header", "Travel   240.50", "Far below"]
    assert lines[2].para > lines[1].para


def test_spaced_from_words_restores_a_dropped_space():
    words = [("hours", 1.0, [[460, 0], [571, 0], [571, 30], [460, 30]]),
             ("12", 1.0, [[609, 0], [650, 0], [650, 30], [609, 30]]),
             ("85.00", 1.0, [[708, 0], [820, 0], [820, 30], [708, 30]])]
    assert engines._spaced_from_words(words) == "hours 12 85.00"
    assert engines._spaced_from_words(None) == ""
    assert engines._spaced_from_words([("x", 1.0, "bad")]) == ""


# ── choosing the engine ───────────────────────────────────────────────────

class _Stub:
    def __init__(self, name: str, ok: bool, langs: set[str] | None = None):
        self.name, self.ok, self.langs = name, ok, frozenset(langs or ())

    def available(self) -> bool:
        return self.ok

    def has_langs(self, lang: str) -> bool:
        return all(c in self.langs for c in lang.split("+"))

    def languages(self) -> frozenset[str]:
        return self.langs

    def version(self) -> str:
        return "0"


@pytest.fixture
def stub_engines(monkeypatch):
    def install(tess: _Stub, rapid: _Stub):
        monkeypatch.setattr(engines, "_ENGINES", {"tesseract": tess, "rapidocr": rapid})
    yield install
    engines.reset_engines()


def test_pick_engine_rules(stub_engines):
    class Cfg:
        engines = ["rapidocr", "tesseract"]
        lang = "eng"

    stub_engines(_Stub("tesseract", True, {"eng", "hrv"}), _Stub("rapidocr", True))
    # A language Tesseract has → Tesseract, whatever the order.
    assert engines.pick_engine("auto", "hrv", Cfg).engine.name == "tesseract"
    # No language → the configured order.
    assert engines.pick_engine("auto", "", Cfg).engine.name == "rapidocr"
    # A language Tesseract lacks → the multilingual model, and the result says so.
    c = engines.pick_engine("auto", "deu", Cfg)
    assert c.engine.name == "rapidocr" and "deu" in c.note
    # Forced engine.
    assert engines.pick_engine("tesseract", "", Cfg).lang == "eng"
    with pytest.raises(EngineError, match="no model for deu"):
        engines.pick_engine("tesseract", "deu", Cfg)
    with pytest.raises(EngineError, match="Tesseract language codes"):
        engines.pick_engine("auto", "hrv; rm -rf /", Cfg)


def test_pick_engine_none_installed(stub_engines):
    stub_engines(_Stub("tesseract", False), _Stub("rapidocr", False))
    with pytest.raises(EngineError) as err:
        engines.pick_engine("auto", "", None)
    msg = str(err.value)
    assert "pip install -U captain-claw" in msg and "brew install tesseract" in msg
    assert "only if the user agrees" in msg


def test_pick_engine_tesseract_only_without_language(stub_engines):
    stub_engines(_Stub("tesseract", True, {"eng"}), _Stub("rapidocr", False))
    with pytest.raises(EngineError, match="tesseract-ocr-hrv"):
        engines.pick_engine("auto", "hrv", None)


# ── the pipeline, on a fake engine ────────────────────────────────────────

try:
    import pypdfium2  # noqa: F401
    _HAS_PDFIUM = True
except Exception:  # noqa: BLE001
    _HAS_PDFIUM = False
needs_pdfium = pytest.mark.skipif(not _HAS_PDFIUM, reason="pypdfium2 renders PDF pages for OCR")


@needs_pdfium
def test_run_pdf_reads_text_pages_and_ocrs_scans(tmp_path: Path):
    pdf = _mixed_pdf(tmp_path)
    fake = FakeEngine("INVOICE 2026-0412")
    choice = Choice(fake, "eng")
    saved = tmp_path / "saved"
    res = doc.run_pdf(pdf, choice, doc.Settings(), saved)
    assert [p.method for p in res.pages] == ["text_layer", "ocr", "ocr"]
    assert [p.reason for p in res.pages] == ["text_layer", "no_text_layer", "no_text_layer"]
    assert "Quarterly revenue" in res.pages[0].text
    assert res.pages[1].text == "INVOICE 2026-0412" and res.pages[1].dpi == 300
    assert fake.calls == 2 and not res.cached

    # Same file + settings → the stored result, no engine call.
    again = doc.run_pdf(pdf, choice, doc.Settings(), saved)
    assert again.cached and fake.calls == 2
    assert [p.to_json() for p in again.pages] == [p.to_json() for p in res.pages]
    # refresh reads again; other settings are another result.
    doc.run_pdf(pdf, choice, doc.Settings(), saved, refresh=True)
    assert fake.calls == 4
    forced = doc.run_pdf(pdf, choice, doc.Settings(mode="ocr"), saved)
    assert [p.method for p in forced.pages] == ["ocr", "ocr", "ocr"] and fake.calls == 7
    only = doc.run_pdf(pdf, choice, doc.Settings(pages="3"), saved)
    assert [p.page for p in only.pages] == [3]


@needs_pdfium
def test_run_pdf_page_cap_names_the_rest(tmp_path: Path):
    pdf = _scanned_pdf(tmp_path / "long.pdf", [[f"page {i}"] for i in range(5)])
    res = doc.run_pdf(pdf, Choice(FakeEngine(), "eng"), doc.Settings(max_pages=2), tmp_path / "saved")
    assert [p.page for p in res.pages] == [1, 2] and res.skipped_pages == [3, 4, 5]
    assert "Not read (page cap): 3-5" in doc.render_markdown(res, "long.pdf", 70)


@needs_pdfium
def test_low_confidence_and_empty_pages_are_flagged(tmp_path: Path):
    pdf = _scanned_pdf(tmp_path / "s.pdf", [["x"]])
    res = doc.run_pdf(pdf, Choice(FakeEngine(conf=41.0), "eng"), doc.Settings(), tmp_path / "saved")
    md = doc.render_markdown(res, "s.pdf", 70)
    assert "confidence 41%" in md and "LOW CONFIDENCE" in md
    empty = doc.run_pdf(pdf, Choice(FakeEngine(text=""), "eng"), doc.Settings(), tmp_path / "saved2")
    assert empty.pages[0].method == "empty"


@needs_pdfium
def test_outputs_are_organized_and_hidden_parts_stay_hidden(tmp_path: Path):
    pdf = _mixed_pdf(tmp_path)
    res = doc.run_pdf(pdf, Choice(FakeEngine(), "eng"), doc.Settings(), tmp_path / "saved")
    folder = doc.output_dir(tmp_path / "saved", "s1", res)
    md, js = doc.write_outputs(res, folder, "mixed.pdf", 70)
    assert folder == tmp_path / "saved" / "ocr" / "s1" / f"mixed-{res.sha256[:8]}"
    assert md.name == "mixed.md" and js.name == ".ocr.json"
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["sha256"] == res.sha256 and data["engine"] == "fake"
    assert data["pages"][1]["lines"][0]["box"] == [10, 10, 400, 40]
    assert b"\r\n" not in md.read_bytes()
    cache = list((tmp_path / "saved" / doc.CACHE_DIR).glob("*.json"))
    assert len(cache) == 1


# ── the tools ─────────────────────────────────────────────────────────────

@pytest.fixture
def fake_choice(monkeypatch):
    fake = FakeEngine("Plaćanje u roku od 30 dana")
    monkeypatch.setattr("captain_claw.ocr.engines.pick_engine", lambda *a, **k: Choice(fake, "hrv"))
    return fake


@needs_pdfium
async def test_ocr_pdf_tool_end_to_end(tmp_path: Path, fake_choice):
    pdf = _mixed_pdf(tmp_path)
    res = await OcrPdfTool().execute(path=str(pdf), **_kw(tmp_path))
    assert res.success, res.error
    assert "Text layer: 1 page(s) · OCR: 2 page(s) with fake 1.0, hrv, 300 dpi" in res.content
    assert "## Page 1 · text layer" in res.content and "## Page 2 · OCR · confidence 93%" in res.content
    assert "Plaćanje u roku od 30 dana" in res.content
    md = next((tmp_path / "saved" / "ocr" / "s1").glob("mixed-*/mixed.md"))
    assert "Plaćanje" in md.read_text(encoding="utf-8")
    again = await OcrPdfTool().execute(path=str(pdf), **_kw(tmp_path))
    assert "stored result reused" in again.content and fake_choice.calls == 2


async def test_ocr_image_tool(tmp_path: Path, fake_choice):
    pic = tmp_path / "receipt.jpg"
    _page_picture(["TOTAL 12.00"]).convert("RGB").save(pic)
    res = await OcrImageTool().execute(path=str(pic), **_kw(tmp_path))
    assert res.success, res.error
    assert "## Picture · OCR" in res.content and "OCR with fake 1.0" in res.content


async def test_tools_refuse_the_wrong_file_kind(tmp_path: Path):
    pic = tmp_path / "a.png"
    _page_picture(["x"]).save(pic)
    res = await OcrPdfTool().execute(path=str(pic), **_kw(tmp_path))
    assert not res.success and "Use ocr_image" in res.error
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(_text_pdf(["hello"]))
    res = await OcrImageTool().execute(path=str(pdf), **_kw(tmp_path))
    assert not res.success and "Use ocr_pdf" in res.error


@needs_pdfium
async def test_tool_without_an_engine_says_how_to_install(tmp_path: Path, stub_engines):
    stub_engines(_Stub("tesseract", False), _Stub("rapidocr", False))
    pdf = _scanned_pdf(tmp_path / "s.pdf", [["x"]])
    res = await OcrPdfTool().execute(path=str(pdf), **_kw(tmp_path))
    assert not res.success and "pip install -U captain-claw" in res.error


def test_pdf_extract_points_scans_to_ocr_pdf(tmp_path: Path):
    from captain_claw.tools.document_extract import _extract_pdf_markdown

    pdf = _mixed_pdf(tmp_path)
    md, err = _extract_pdf_markdown(pdf, 100)
    assert err is None
    assert "Pages 2, 3 of this PDF have little or no text layer" in md and "ocr_pdf" in md
    text_only = tmp_path / "t.pdf"
    text_only.write_bytes(_text_pdf(["A digital page with a real text layer and enough words."]))
    md, _ = _extract_pdf_markdown(text_only, 100)
    assert "ocr_pdf" not in md


# ── real engines (skipped when not installed) ─────────────────────────────

def _installed(name: str) -> bool:
    engines.reset_engines()
    return engines.get_engine(name).available()


@pytest.mark.parametrize("name", ["tesseract", "rapidocr"])
def test_real_engine_reads_and_repeats(tmp_path: Path, name: str):
    if not _installed(name):
        pytest.skip(f"{name} not installed")
    pic = tmp_path / "invoice.png"
    _page_picture(["INVOICE 2026-0412", "TOTAL EUR 1,575.63"], size=40).save(pic)
    choice = Choice(engines.get_engine(name), "eng" if name == "tesseract" else "")
    runs = [doc.run_image(pic, choice, doc.Settings(), tmp_path / "saved", refresh=True) for _ in range(2)]
    text = runs[0].pages[0].text
    assert "INVOICE" in text and "1,575.63" in text
    assert runs[1].pages[0].text == text  # the same text every time
    engines.reset_engines()
