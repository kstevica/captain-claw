"""Office files, beyond the text: pictures replaced (with the text read from
them kept as anonymized alt text), alt text anonymized, thumbnails removed,
embedded Office files anonymized inside their own zip, other embedded objects
emptied, customXml text anonymized — and restore putting it back. Made-up data."""

from __future__ import annotations

import io
import posixpath
import re
import zipfile

import docx
import openpyxl
import pptx
import pytest
from PIL import Image
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from captain_claw.anonymize import engine as E  # noqa: N812
from captain_claw.anonymize import office

EMAIL = "ana.kovac@primjer-firma.hr"
READ = f"Kontakt: {EMAIL}"


def _png(w: int = 320, h: int = 160, color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def _members(blob: bytes) -> dict[str, bytes]:
    z = zipfile.ZipFile(io.BytesIO(blob))
    return {n: z.read(n) for n in z.namelist()}


def _rewrite(blob: bytes, change) -> bytes:
    """*blob* with its members passed through change(members) → members."""
    members = change(_members(blob))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return out.getvalue()


def _run(blob: bytes, kind: str, state=None, **prep):
    state = state if state is not None else E.VaultState()
    p = office.prepare(blob, kind, **prep)
    result = E.run(p.segments, state, E.Options())
    return p.rebuild(result.edits), state, p


def _restore(blob: bytes, kind: str, state) -> bytes:
    p = office.prepare(blob, kind, scrub=False)
    return p.rebuild([E.restore_edits(s.text, state)[0] for s in p.segments])


def _letter_with_pictures() -> bytes:
    d = docx.Document()
    d.add_paragraph("Ponuda u prilogu.")
    d.add_picture(io.BytesIO(_png(320, 160)), width=Inches(2))
    d.sections[0].header.paragraphs[0].add_run().add_picture(io.BytesIO(_png(100, 100, (0, 0, 200))),
                                                              width=Inches(0.5))
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


class TestWordPictures:
    def test_strip_replaces_every_picture_and_keeps_its_text_as_anonymized_alt_text(self):
        src = _letter_with_pictures()
        before = {n: b for n, b in _members(src).items() if n.startswith("word/media/")}
        assert len(before) == 2
        out, state, prep = _run(src, "docx", pictures="strip", picture_reader=lambda b, e: READ)
        after = _members(out)
        for name, data in before.items():
            assert name in after and after[name] != data
            with Image.open(io.BytesIO(after[name])) as old, Image.open(io.BytesIO(data)) as new_src:
                assert abs(old.size[0] / old.size[1] - new_src.size[0] / new_src.size[1]) < 0.02  # same shape
        doc_xml = after["word/document.xml"].decode()
        header_xml = next(v.decode() for k, v in after.items() if re.fullmatch(r"word/header\d+\.xml", k))
        assert EMAIL not in doc_xml and EMAIL not in header_xml
        assert re.search(r'descr="Text in the picture: Kontakt: ⟦EMAIL_1⟧"', doc_xml)
        assert "picture(s) replaced" in " ".join(prep.warnings)
        docx.Document(io.BytesIO(out))  # still opens
        restored = _restore(out, "docx", state)
        assert f"Text in the picture: Kontakt: {EMAIL}" in _members(restored)["word/document.xml"].decode()

    def test_keep_leaves_pictures_and_says_so(self):
        src = _letter_with_pictures()
        out, _state, prep = _run(src, "docx")
        before, after = _members(src), _members(out)
        assert all(after[n] == before[n] for n in before if n.startswith("word/media/"))
        assert "left as they are" in " ".join(prep.warnings)

    def test_existing_alt_text_is_anonymized(self):
        src = _rewrite(_letter_with_pictures(), lambda m: {
            **m, "word/document.xml": re.sub(rb'(<wp:docPr [^>]*?)descr="[^"]*"', rb"\1", m["word/document.xml"])
            .replace(b"<wp:docPr ", f'<wp:docPr descr="Logo, pišite na {EMAIL}" '.encode(), 1)})
        out, _state, _prep = _run(src, "docx")
        xml = _members(out)["word/document.xml"].decode()
        assert EMAIL not in xml and "⟦EMAIL_1⟧" in xml

    def test_a_picture_pillow_cannot_write_becomes_png_and_links_follow(self):
        def to_emf(m):
            name = next(n for n in m if n.startswith("word/media/") and "header" not in n)
            m = dict(m)
            m[name.rsplit(".", 1)[0] + ".emf"] = b"\x01\x00\x00\x00not-really-emf"
            del m[name]
            rels = m["word/_rels/document.xml.rels"].replace(
                posixpath.basename(name).encode(), posixpath.basename(name).rsplit(".", 1)[0].encode() + b".emf")
            m["word/_rels/document.xml.rels"] = rels
            m["[Content_Types].xml"] = m["[Content_Types].xml"].replace(
                b"</Types>", b'<Default Extension="emf" ContentType="image/x-emf"/></Types>')
            return m

        src = _rewrite(_letter_with_pictures(), to_emf)
        out, _state, _prep = _run(src, "docx", pictures="strip")
        after = _members(out)
        assert not any(n.endswith(".emf") for n in after)
        pngs = [n for n in after if n.startswith("word/media/") and n.endswith(".png")]
        rels = after["word/_rels/document.xml.rels"].decode()
        assert ".emf" not in rels and all(posixpath.basename(n) in rels or "header" in rels for n in pngs)
        assert 'Extension="png"' in after["[Content_Types].xml"].decode()
        d = docx.Document(io.BytesIO(out))
        assert d.inline_shapes[0]._inline.graphic.graphicData.pic.blipFill.blip.embed in d.part.rels


class TestThumbnail:
    def test_removed_with_its_relationship(self):
        def add_thumb(m):
            m = dict(m)
            m["docProps/thumbnail.jpeg"] = _png()
            m["_rels/.rels"] = m["_rels/.rels"].replace(
                b"</Relationships>",
                b'<Relationship Id="rIdThumb" Type="http://schemas.openxmlformats.org/package/2006/'
                b'relationships/metadata/thumbnail" Target="docProps/thumbnail.jpeg"/></Relationships>')
            return m

        src = _rewrite(_letter_with_pictures(), add_thumb)
        out, _s, _p = _run(src, "docx")
        after = _members(out)
        assert "docProps/thumbnail.jpeg" not in after
        assert b"thumbnail" not in after["_rels/.rels"]
        docx.Document(io.BytesIO(out))

    def test_restore_does_not_touch_it(self):
        src = _rewrite(_letter_with_pictures(), lambda m: {**m, "docProps/thumbnail.jpeg": _png()})
        out = _restore(src, "docx", E.VaultState())
        assert "docProps/thumbnail.jpeg" in _members(out)


def _deck_with_chart_and_picture() -> bytes:
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    data = CategoryChartData()
    data.categories = [EMAIL, "ostali"]
    data.add_series("Udio prihoda", (23.0, 77.0))
    slide.shapes.add_chart(XL_CHART_TYPE.PIE, Inches(1), Inches(1), Inches(4), Inches(3), data)
    slide.shapes.add_picture(io.BytesIO(_png(200, 100)), Inches(5), Inches(1))
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


class TestEmbedded:
    def test_a_charts_data_workbook_is_anonymized_inside_and_restored(self):
        src = _deck_with_chart_and_picture()
        emb = [n for n in _members(src) if n.startswith("ppt/embeddings/") and n.endswith(".xlsx")]
        assert emb
        out, state, _prep = _run(src, "pptx")
        inner = openpyxl.load_workbook(io.BytesIO(_members(out)[emb[0]]))
        values = [c.value for row in inner.active.iter_rows() for c in row if c.value is not None]
        assert EMAIL not in values and any("⟦EMAIL_1⟧" == v for v in values)
        chart_xml = next(v.decode() for k, v in _members(out).items() if k.startswith("ppt/charts/chart"))
        assert EMAIL not in chart_xml
        pptx.Presentation(io.BytesIO(out))
        back = openpyxl.load_workbook(io.BytesIO(_members(_restore(out, "pptx", state))[emb[0]]))
        assert EMAIL in [c.value for row in back.active.iter_rows() for c in row]

    def test_other_embedded_objects_are_emptied_when_stripping(self):
        src = _rewrite(_deck_with_chart_and_picture(),
                       lambda m: {**m, "ppt/embeddings/oleObject1.bin": b"OLE" + EMAIL.encode()})
        kept, _s, kept_prep = _run(src, "pptx")
        assert _members(kept)["ppt/embeddings/oleObject1.bin"].startswith(b"OLE")
        assert "left as they are" in " ".join(kept_prep.warnings)
        stripped, _s, strip_prep = _run(src, "pptx", pictures="strip")
        assert _members(stripped)["ppt/embeddings/oleObject1.bin"] == b""
        assert "emptied" in " ".join(strip_prep.warnings)

    def test_slide_picture_alt_text_from_the_reader(self):
        out, _s, _p = _run(_deck_with_chart_and_picture(), "pptx", pictures="strip",
                           picture_reader=lambda b, e: READ)
        slide = _members(out)["ppt/slides/slide1.xml"].decode()
        # python-pptx names the picture "image.png" in its alt text; the read text follows it
        assert 'descr="image.png — Text in the picture: Kontakt: ⟦EMAIL_1⟧"' in slide


class TestSheetsAndCustomXml:
    def test_sheet_picture_replaced_and_described(self):
        from openpyxl.drawing.image import Image as XLImage

        wb = openpyxl.Workbook()
        wb.active["A1"] = "Ponuda"
        wb.active.add_image(XLImage(io.BytesIO(_png(120, 60))), "C3")
        buf = io.BytesIO()
        wb.save(buf)
        src = buf.getvalue()
        out, _s, _p = _run(src, "xlsx", pictures="strip", picture_reader=lambda b, e: READ)
        after = _members(out)
        media = [n for n in after if n.startswith("xl/media/")]
        assert media and all(after[n] != _members(src)[n] for n in media)
        drawing = next(v.decode() for k, v in after.items() if re.fullmatch(r"xl/drawings/drawing\d+\.xml", k))
        assert "Text in the picture: Kontakt: ⟦EMAIL_1⟧" in drawing
        openpyxl.load_workbook(io.BytesIO(out))

    def test_custom_xml_text_is_anonymized(self):
        src = _rewrite(_letter_with_pictures(), lambda m: {
            **m, "customXml/item1.xml": f'<?xml version="1.0"?><root><Owner>{EMAIL}</Owner></root>'.encode()})
        out, state, _p = _run(src, "docx")
        assert EMAIL.encode() not in _members(out)["customXml/item1.xml"]
        assert EMAIL.encode() in _members(_restore(out, "docx", state))["customXml/item1.xml"]


def test_placeholder_keeps_format_and_proportions():
    for ext, fmt in (("png", "PNG"), ("jpg", "JPEG"), ("gif", "GIF")):
        blob, new_ext = office.placeholder(_png(900, 300), ext)
        assert new_ext == ext
        with Image.open(io.BytesIO(blob)) as im:
            assert im.format == fmt and abs(im.size[0] / im.size[1] - 3.0) < 0.05 and max(im.size) <= 600
    svg, ext = office.placeholder(b"<svg/>", "svg")
    assert ext == "svg" and b"picture removed" in svg
    blob, ext = office.placeholder(b"garbage", "emf")
    assert ext == "png"


def test_unknown_picture_mode_is_refused():
    with pytest.raises(office.OfficeError):
        office.prepare(_letter_with_pictures(), "docx", pictures="blur")
