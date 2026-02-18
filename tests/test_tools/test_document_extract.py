from pathlib import Path
import zipfile

import pytest

from captain_claw.tools.document_extract import (
    DocxExtractTool,
    PdfExtractTool,
    PptxExtractTool,
    XlsxExtractTool,
)


def _write_minimal_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Quarterly Report</w:t></w:r></w:p>
    <w:p><w:r><w:t>Revenue grew 20%.</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Score</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Ana</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>95</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


def _write_minimal_xlsx(path: Path) -> None:
    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Summary" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
                Target="worksheets/sheet1.xml"/>
</Relationships>
"""
    shared_strings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
  <si><t>City</t></si>
  <si><t>Population</t></si>
</sst>
"""
    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c>
      <c r="B1" t="s"><v>1</v></c>
    </row>
    <row r="2">
      <c r="A2" t="inlineStr"><is><t>Zagreb</t></is></c>
      <c r="B2"><v>767131</v></c>
    </row>
  </sheetData>
</worksheet>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/sharedStrings.xml", shared_strings)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _write_minimal_pptx(path: Path) -> None:
    slide_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:sp>
        <p:txBody>
          <a:p><a:r><a:t>Launch Plan</a:t></a:r></a:p>
          <a:p><a:r><a:t>Ship by Q3</a:t></a:r></a:p>
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml)


@pytest.mark.asyncio
async def test_docx_extract_returns_markdown(tmp_path: Path):
    doc_path = tmp_path / "sample.docx"
    _write_minimal_docx(doc_path)
    tool = DocxExtractTool()

    result = await tool.execute(path=str(doc_path))

    assert result.success is True
    assert "# sample.docx" in result.content
    assert "# Quarterly Report" in result.content
    assert "Revenue grew 20%." in result.content
    assert "| Name | Score |" in result.content
    assert "| Ana | 95 |" in result.content


@pytest.mark.asyncio
async def test_xlsx_extract_returns_markdown_tables(tmp_path: Path):
    xlsx_path = tmp_path / "metrics.xlsx"
    _write_minimal_xlsx(xlsx_path)
    tool = XlsxExtractTool()

    result = await tool.execute(path=str(xlsx_path))

    assert result.success is True
    assert "# metrics.xlsx" in result.content
    assert "## Sheet: Summary" in result.content
    assert "| City | Population |" in result.content
    assert "| Zagreb | 767131 |" in result.content


@pytest.mark.asyncio
async def test_pptx_extract_returns_slide_markdown(tmp_path: Path):
    pptx_path = tmp_path / "deck.pptx"
    _write_minimal_pptx(pptx_path)
    tool = PptxExtractTool()

    result = await tool.execute(path=str(pptx_path))

    assert result.success is True
    assert "# deck.pptx" in result.content
    assert "## Slide 1" in result.content
    assert "- Launch Plan" in result.content
    assert "- Ship by Q3" in result.content


@pytest.mark.asyncio
async def test_pdf_extract_reports_missing_dependency_or_extracts(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    try:
        from pypdf import PdfWriter
    except Exception:
        PdfWriter = None

    if PdfWriter is None:
        pdf_path.write_bytes(b"%PDF-1.4\n%missing-reader\n")
        result = await PdfExtractTool().execute(path=str(pdf_path))
        assert result.success is False
        assert "pypdf" in (result.error or "").lower()
        return

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=144)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    result = await PdfExtractTool().execute(path=str(pdf_path))
    assert result.success is True
    assert "# sample.pdf" in result.content
    assert "## Page 1" in result.content
