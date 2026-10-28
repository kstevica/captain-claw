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


_MAIN_NS = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _write_xlsx(path: Path, sheets: dict[str, str], absolute: bool) -> None:
    """A workbook of the given ``<sheetData>`` bodies, its relationships
    pointing at them from ``xl/`` or (``absolute``) the package root."""
    names = list(sheets)
    workbook = (f'<?xml version="1.0"?><workbook {_MAIN_NS} xmlns:r="{_REL_NS}"><sheets>'
                + "".join(f'<sheet name="{n}" sheetId="{i}" r:id="rId{i}"/>' for i, n in enumerate(names, 1))
                + "</sheets></workbook>")
    rels = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(f'<Relationship Id="rId{i}" Type="{_REL_NS}/worksheet" '
                      f'Target="{"/xl/" if absolute else ""}worksheets/sheet{i}.xml"/>' for i in range(1, len(names) + 1))
            + "</Relationships>")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        for i, name in enumerate(names, 1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml",
                             f'<?xml version="1.0"?><worksheet {_MAIN_NS}><sheetData>{sheets[name]}</sheetData></worksheet>')


def _inline(text: str, ref: str = "") -> str:
    return f'<c{f" r={chr(34)}{ref}{chr(34)}" if ref else ""} t="inlineStr"><is><t>{text}</t></is></c>'


@pytest.mark.asyncio
async def test_xlsx_extract_reads_sheets_named_from_the_package_root(tmp_path: Path):
    # Rich inline text, and cells that leave out their reference (each is the one after the last).
    plans = (f'<row r="1">{_inline("SKU", "A1")}{_inline("Price", "B1")}</row>'
             '<row r="2"><c r="A2" t="inlineStr"><is><r><t>A-</t></r><r><rPr><b/></rPr><t>1</t></r></is></c>'
             '<c r="B2"><v>12.5</v></c></row>')
    addons = f'<row>{_inline("Code")}{_inline("Name")}</row><row>{_inline("B-2")}{_inline("Backup")}</row>'
    for absolute in (True, False):
        path = tmp_path / f"prices-{absolute}.xlsx"
        _write_xlsx(path, {"Plans": plans, "Add-ons": addons}, absolute)

        result = await XlsxExtractTool().execute(path=str(path))

        assert result.success is True
        assert "## Sheet: Plans" in result.content and "| A-1 | 12.5 |" in result.content
        assert "## Sheet: Add-ons" in result.content and "| B-2 | Backup |" in result.content


@pytest.mark.asyncio
async def test_xlsx_extract_reads_a_workbook_openpyxl_wrote(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Plans"
    wb.active.append(["SKU", "Price"])
    wb.active.append(["A-1", 12.5])
    extra = wb.create_sheet("Add-ons")
    extra.append(["Code", "Name"])
    extra.append(["B-2", "Backup"])
    path = tmp_path / "prices.xlsx"
    wb.save(path)

    result = await XlsxExtractTool().execute(path=str(path))

    assert result.success is True
    assert "| SKU | Price |" in result.content and "| A-1 | 12.5 |" in result.content
    assert "## Sheet: Add-ons" in result.content and "| B-2 | Backup |" in result.content


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
