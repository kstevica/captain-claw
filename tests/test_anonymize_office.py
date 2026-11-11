"""Office files anonymized inside their zip (docx, xlsx, pptx): text replaced
where it stands, formatting and pictures kept, metadata cleaned, restore
puts everything back."""

from __future__ import annotations

import io
import re
import zipfile

import docx
import openpyxl
import pptx
import pytest

from captain_claw.anonymize import detectors as d
from captain_claw.anonymize import engine as E
from captain_claw.anonymize import office


def _zip_text(blob: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(blob))
    return "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith((".xml", ".rels")))


def _anonymize(blob: bytes, kind: str, state=None, **opts):
    state = state if state is not None else E.VaultState()
    prep = office.prepare(blob, kind)
    result = E.run(prep.segments, state, E.Options(**opts))
    return prep.rebuild(result.edits), state, prep


def _restore(blob: bytes, kind: str, state):
    prep = office.prepare(blob, kind, scrub=False)
    edits = [E.restore_edits(s.text, state)[0] for s in prep.segments]
    return prep.rebuild(edits)


def _with_part(blob: bytes, name: str, change) -> bytes:
    """*blob* with one zip member's text changed by *change*."""
    src = zipfile.ZipFile(io.BytesIO(blob))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == name:
                data = change(data.decode("utf-8")).encode("utf-8")
            z.writestr(info, data)
    return out.getvalue()


def _letter() -> bytes:
    doc = docx.Document()
    doc.core_properties.author = "Ivan Horvat"
    doc.core_properties.last_modified_by = "Ivan Horvat"
    doc.core_properties.title = "Ugovor za Ivana Horvata"
    doc.sections[0].header.paragraphs[0].text = "Pripremio: Marko Marić"
    p = doc.add_paragraph("Poštovani ")
    bold = p.add_run("Ivan")
    bold.bold = True
    p.add_run(" Horvat, šaljem ugovor na ivan.horvat@acme.hr.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "IBAN"
    table.cell(0, 1).text = "HR1210010051863000160"
    para = doc.add_paragraph("OIB: 52601815906")
    doc.add_comment(para.runs, text="Pitaj Marka Marića", author="Ana Kovač", initials="AK")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestWord:
    def test_text_replaced_in_place_formatting_kept_nothing_left(self):
        src = _letter()
        out, state, _prep = _anonymize(src, "docx")
        doc = docx.Document(io.BytesIO(out))
        first = doc.paragraphs[0]
        assert first.text == "Poštovani ⟦PERSON_1⟧, šaljem ugovor na ⟦EMAIL_1⟧."
        assert [(r.text, r.bold) for r in first.runs][1] == ("⟦PERSON_1⟧", True)  # the bold run keeps the token
        assert doc.sections[0].header.paragraphs[0].text == "Pripremio: ⟦PERSON_2⟧"
        assert doc.tables[0].cell(0, 1).text == "⟦IBAN_1⟧"
        assert doc.core_properties.author == "" and doc.core_properties.last_modified_by == ""
        assert doc.core_properties.title == "Ugovor za ⟦PERSON_1⟧"
        everything = _zip_text(out)
        for value in ("Horvat", "Marić", "Marka", "Kovač", "ivan.horvat", "52601815906", "HR1210010051863000160"):
            assert value not in everything, value
        assert 'w:author="Author"' in everything

    def test_restore_puts_it_back(self):
        src = _letter()
        out, state, _prep = _anonymize(src, "docx")
        back = docx.Document(io.BytesIO(_restore(out, "docx", state)))
        assert back.paragraphs[0].text == "Poštovani Ivan Horvat, šaljem ugovor na ivan.horvat@acme.hr."
        assert back.tables[0].cell(0, 1).text == "HR1210010051863000160"

    def test_tracked_deletions_field_codes_and_links(self):
        src = _letter()

        def add(xml: str) -> str:
            extra = ('<w:p><w:del w:id="9" w:author="Ana Kovač"><w:r><w:delText>OIB 83016613185</w:delText></w:r></w:del>'
                     '<w:r><w:instrText xml:space="preserve"> HYPERLINK "mailto:petra.n@x.hr" </w:instrText></w:r></w:p>')
            return xml.replace("</w:body>", extra + "</w:body>") if "<w:sectPr" not in xml else \
                xml.replace("<w:sectPr", extra + "<w:sectPr", 1)

        src = _with_part(src, "word/document.xml", add)
        src = _with_part(src, "word/_rels/document.xml.rels", lambda x: x.replace(
            "</Relationships>",
            '<Relationship Id="rIdX" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'hyperlink" Target="mailto:petra.n@x.hr" TargetMode="External"/></Relationships>'))
        out, _s, _p = _anonymize(src, "docx")
        everything = _zip_text(out)
        assert "83016613185" not in everything and "petra.n@x.hr" not in everything
        assert "Kovač" not in everything
        assert "mailto:⟦EMAIL_" in everything

    def test_a_label_in_the_cell_before(self):
        doc = docx.Document()
        table = doc.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Ime i prezime:"
        table.cell(0, 1).text = "Petra Novak"
        doc.add_paragraph("Petra Novak je klijent.")
        buf = io.BytesIO()
        doc.save(buf)
        out, _s, _p = _anonymize(buf.getvalue(), "docx")
        back = docx.Document(io.BytesIO(out))
        assert back.tables[0].cell(0, 1).text == "⟦PERSON_1⟧"
        assert back.paragraphs[-1].text == "⟦PERSON_1⟧ je klijent."

    def test_pictures_are_reported(self):
        from PIL import Image

        doc = docx.Document()
        img = io.BytesIO()
        Image.new("RGB", (4, 4)).save(img, "PNG")
        img.seek(0)
        doc.add_picture(img)
        buf = io.BytesIO()
        doc.save(buf)
        _out, _s, prep = _anonymize(buf.getvalue(), "docx")
        assert any("picture" in w for w in prep.warnings)


class TestExcel:
    def _book(self) -> bytes:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Ivan Horvat"
        ws.append(["Ime", "Prezime", "Email", "OIB", "Iznos"])
        ws.append(["Ivan", "Horvat", "ivan@x.hr", 52601815906, 100])
        ws.append(["Ana", "Kovač", "ana@y.hr", 83016613185, 250])
        wb.create_sheet("Zbroj")["A1"] = "='Ivan Horvat'!E2+'Ivan Horvat'!E3"
        wb.properties.creator = "Ivan Horvat"
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    def test_cells_numbers_sheet_names_and_formulas(self):
        out, state, _p = _anonymize(self._book(), "xlsx")
        wb = openpyxl.load_workbook(io.BytesIO(out))
        rows = [[c.value for c in row] for row in wb.worksheets[0].iter_rows()]
        assert rows[0] == ["Ime", "Prezime", "Email", "OIB", "Iznos"]
        assert rows[1][3] == "⟦OIB_1⟧" and rows[1][4] == 100 and rows[2][4] == 250
        assert all(re.fullmatch(r"⟦[A-Z_]+_\d+⟧", str(v)) for v in rows[1][:3] + rows[2][:3])
        name = wb.sheetnames[0]
        assert re.fullmatch(r"(⟦PERSON_\d+⟧ ?)+", name)  # "Ivan" and "Horvat" are separate cells
        assert wb["Zbroj"]["A1"].value == f"='{name}'!E2+'{name}'!E3"
        assert not wb.properties.creator
        everything = _zip_text(out)
        for value in ("Horvat", "Kovač", "52601815906", "83016613185", "ivan@x.hr"):
            assert value not in everything, value

    def test_restore(self):
        out, state, _p = _anonymize(self._book(), "xlsx")
        wb = openpyxl.load_workbook(io.BytesIO(_restore(out, "xlsx", state)))
        assert wb.sheetnames[0] == "Ivan Horvat"
        assert wb["Zbroj"]["A1"].value == "='Ivan Horvat'!E2+'Ivan Horvat'!E3"
        assert [c.value for c in wb.worksheets[0][2]][:4] == ["Ivan", "Horvat", "ivan@x.hr", "52601815906"]


class TestPowerPoint:
    def test_slides_notes_and_restore(self):
        deck = pptx.Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = "Ponuda za Acme d.o.o."
        slide.placeholders[1].text = "Kontakt: Ivan Horvat, +385 91 234 5678"
        slide.notes_slide.notes_text_frame.text = "Lijep pozdrav,\nMarko Marić"
        buf = io.BytesIO()
        deck.save(buf)
        out, state, _p = _anonymize(buf.getvalue(), "pptx", extra=(("Ivan Horvat", "PERSON"),))
        s = pptx.Presentation(io.BytesIO(out)).slides[0]
        assert s.shapes.title.text == "Ponuda za ⟦ORG_1⟧"
        assert s.placeholders[1].text == "Kontakt: ⟦PERSON_1⟧, ⟦PHONE_1⟧"
        assert s.notes_slide.notes_text_frame.text == "Lijep pozdrav,\n⟦PERSON_2⟧"  # the sign-off reaches the next line
        back = pptx.Presentation(io.BytesIO(_restore(out, "pptx", state))).slides[0]
        assert back.placeholders[1].text == "Kontakt: Ivan Horvat, +385 91 234 5678"


class TestBadFiles:
    def test_not_a_zip_and_wrong_kind(self):
        with pytest.raises(office.OfficeError):
            office.prepare(b"hello", "docx")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("readme.txt", "hi")
        with pytest.raises(office.OfficeError):
            office.prepare(buf.getvalue(), "docx")

    def test_entities_are_never_resolved_from_xml(self):
        """An external entity in a part is not fetched or expanded."""
        src = _with_part(_letter(), "word/document.xml", lambda x: x.replace(
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>', 1))
        out, _s, _p = _anonymize(src, "docx")
        assert "root:" not in _zip_text(out)


def test_edits_map_back_across_rounds(monkeypatch):
    """Spans found in later rounds (on text that already has tokens) are
    reported in the original text's positions."""
    real = d.detect

    def one_q(text, kinds=None, **kw):
        i = text.find("Q")
        return [d.Span(i, i + 1, "PERSON", 2, "t")] if i >= 0 else []

    monkeypatch.setattr(d, "detect", one_q)
    text = "a Q bb Q c"
    r = E.run([E.Segment(text)], E.VaultState(), E.Options(extra=(("bb", "ORG"),)))
    assert r.rounds >= 2
    assert E.apply_edits(text, r.edits[0]) == r.outputs[0]
    assert [(s, e) for s, e, _t in r.edits[0]] == [(2, 3), (4, 6), (7, 8)]
    monkeypatch.setattr(d, "detect", real)
