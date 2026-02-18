"""Document extraction tools (PDF/DOCX/XLSX/PPTX) with Markdown output."""

import asyncio
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from captain_claw.logging import get_logger
from captain_claw.tools.registry import Tool, ToolResult

log = get_logger(__name__)


def _escape_md_cell(value: str) -> str:
    """Escape markdown table cell separators."""
    return (value or "").replace("|", "\\|").replace("\n", "<br>")


def _normalize_text(value: str) -> str:
    """Normalize whitespace while preserving line boundaries."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in (value or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _truncate_markdown(markdown: str, max_chars: int) -> str:
    """Trim markdown output to `max_chars` while keeping response explicit."""
    if len(markdown) <= max_chars:
        return markdown
    suffix = "\n\n... [truncated]"
    keep = max(0, max_chars - len(suffix))
    return markdown[:keep] + suffix


def _require_existing_file(path: str) -> tuple[Path | None, str | None]:
    """Validate path points to an existing file and return resolved path."""
    try:
        file_path = Path(path).expanduser().resolve()
    except Exception as e:
        return None, f"Invalid path '{path}': {e}"

    if not file_path.exists():
        return None, f"File not found: {path}"
    if not file_path.is_file():
        return None, f"Not a file: {path}"
    return file_path, None


def _local_name(tag: str) -> str:
    """Return XML local name for namespaced tags."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _extract_docx_paragraph_text(paragraph: ET.Element, ns: dict[str, str]) -> str:
    """Extract concatenated text from a DOCX paragraph element."""
    chunks: list[str] = []
    for text_node in paragraph.findall(".//w:t", ns):
        if text_node.text:
            chunks.append(text_node.text)
    return _normalize_text("".join(chunks))


def _extract_docx_table_markdown(table_el: ET.Element, ns: dict[str, str]) -> list[str]:
    """Convert a DOCX table element to Markdown table lines."""
    rows: list[list[str]] = []
    for row_el in table_el.findall("./w:tr", ns):
        cells: list[str] = []
        for cell_el in row_el.findall("./w:tc", ns):
            cell_lines: list[str] = []
            for para in cell_el.findall(".//w:p", ns):
                text = _extract_docx_paragraph_text(para, ns)
                if text:
                    cell_lines.append(text)
            cells.append(" ".join(cell_lines).strip())
        if any(cell.strip() for cell in cells):
            rows.append(cells)

    if not rows:
        return []

    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]

    lines = [
        "| " + " | ".join(_escape_md_cell(cell) for cell in header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_escape_md_cell(cell) for cell in row) + " |")
    return lines


def _extract_docx_markdown(path: Path) -> str:
    """Extract DOCX content as markdown."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path, "r") as archive:
        raw = archive.read("word/document.xml")
    root = ET.fromstring(raw)
    body = root.find("w:body", ns)
    if body is None:
        return f"# {path.name}\n\n_(No document body found.)_"

    lines: list[str] = [f"# {path.name}", ""]
    for child in list(body):
        child_name = _local_name(child.tag)
        if child_name == "p":
            text = _extract_docx_paragraph_text(child, ns)
            if not text:
                continue
            style = child.find("./w:pPr/w:pStyle", ns)
            style_val = ""
            if style is not None:
                style_val = str(style.get(f"{{{ns['w']}}}val", "")).strip().lower()
            if style_val.startswith("heading"):
                digits = "".join(ch for ch in style_val if ch.isdigit())
                level = min(6, max(1, int(digits) if digits else 1))
                lines.append(f"{'#' * level} {text}")
            elif child.find("./w:pPr/w:numPr", ns) is not None:
                lines.append(f"- {text}")
            else:
                lines.append(text)
            lines.append("")
            continue
        if child_name == "tbl":
            table_lines = _extract_docx_table_markdown(child, ns)
            if table_lines:
                lines.extend(table_lines)
                lines.append("")

    rendered = "\n".join(lines).strip()
    return rendered if rendered else f"# {path.name}\n\n_(No extractable text found.)_"


def _xlsx_col_to_index(cell_ref: str) -> int:
    """Convert cell reference column letters (e.g. 'AB12') to zero-based index."""
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    if not letters:
        return 0
    idx = 0
    for ch in letters:
        idx = (idx * 26) + (ord(ch) - ord("A") + 1)
    return max(0, idx - 1)


def _xlsx_index_to_col(idx: int) -> str:
    """Convert zero-based column index to Excel column letters."""
    n = idx + 1
    letters: list[str] = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters)) or "A"


def _extract_xlsx_markdown(path: Path, max_rows: int) -> str:
    """Extract XLSX sheets as markdown tables."""
    ns_main = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}

    with zipfile.ZipFile(path, "r") as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall(".//x:si", ns_main):
                chunks: list[str] = []
                for text_node in si.findall(".//x:t", ns_main):
                    if text_node.text:
                        chunks.append(text_node.text)
                shared_strings.append(_normalize_text("".join(chunks)))

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map: dict[str, str] = {}
        for rel in rel_root.findall(".//rel:Relationship", rel_ns):
            rel_id = str(rel.get("Id", "")).strip()
            target = str(rel.get("Target", "")).strip()
            if rel_id and target:
                rel_map[rel_id] = target

        lines: list[str] = [f"# {path.name}", ""]
        sheets = workbook_root.findall(".//x:sheets/x:sheet", ns_main)
        for sheet in sheets:
            name = str(sheet.get("name", "Sheet")).strip() or "Sheet"
            rel_id = str(sheet.get(f"{{{ns_rel['r']}}}id", "")).strip()
            target = rel_map.get(rel_id, "")
            if not target:
                continue
            sheet_path = f"xl/{target}" if not target.startswith("xl/") else target
            if sheet_path not in archive.namelist():
                continue

            sheet_root = ET.fromstring(archive.read(sheet_path))
            rows: list[list[str]] = []
            max_col = 0
            for row_el in sheet_root.findall(".//x:sheetData/x:row", ns_main):
                if len(rows) >= max_rows:
                    break
                row_values: dict[int, str] = {}
                for cell in row_el.findall("./x:c", ns_main):
                    cell_ref = str(cell.get("r", "")).strip()
                    col_idx = _xlsx_col_to_index(cell_ref) if cell_ref else 0
                    max_col = max(max_col, col_idx)
                    cell_type = str(cell.get("t", "")).strip()
                    value = ""
                    if cell_type == "inlineStr":
                        node = cell.find("./x:is/x:t", ns_main)
                        value = node.text if node is not None and node.text else ""
                    else:
                        node = cell.find("./x:v", ns_main)
                        raw = node.text.strip() if node is not None and node.text else ""
                        if cell_type == "s" and raw.isdigit():
                            idx = int(raw)
                            value = shared_strings[idx] if 0 <= idx < len(shared_strings) else raw
                        elif cell_type == "b":
                            value = "TRUE" if raw == "1" else "FALSE"
                        else:
                            value = raw
                    row_values[col_idx] = _normalize_text(value)

                width = max_col + 1 if max_col >= 0 else 1
                row = [row_values.get(col, "") for col in range(width)]
                if any(cell.strip() for cell in row):
                    rows.append(row)

            lines.append(f"## Sheet: {name}")
            lines.append("")
            if not rows:
                lines.append("_(empty sheet)_")
                lines.append("")
                continue

            width = max(len(row) for row in rows)
            normalized = [row + [""] * (width - len(row)) for row in rows]
            header_raw = normalized[0]
            header = [
                header_raw[idx] if header_raw[idx].strip() else _xlsx_index_to_col(idx)
                for idx in range(width)
            ]
            lines.append("| " + " | ".join(_escape_md_cell(cell) for cell in header) + " |")
            lines.append("| " + " | ".join(["---"] * width) + " |")
            for row in normalized[1:]:
                lines.append("| " + " | ".join(_escape_md_cell(cell) for cell in row) + " |")
            lines.append("")

    rendered = "\n".join(lines).strip()
    return rendered if rendered else f"# {path.name}\n\n_(No extractable content found.)_"


def _extract_pptx_markdown(path: Path, max_slides: int) -> str:
    """Extract PPTX slide text as markdown."""
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    slide_re = re.compile(r"^ppt/slides/slide(\d+)\.xml$")

    with zipfile.ZipFile(path, "r") as archive:
        slide_entries: list[tuple[int, str]] = []
        for name in archive.namelist():
            match = slide_re.match(name)
            if match:
                slide_entries.append((int(match.group(1)), name))
        slide_entries.sort(key=lambda item: item[0])

        lines: list[str] = [f"# {path.name}", ""]
        for index, entry in slide_entries[:max_slides]:
            root = ET.fromstring(archive.read(entry))
            para_lines: list[str] = []
            for para in root.findall(".//a:p", ns):
                chunks: list[str] = []
                for node in para.findall(".//a:t", ns):
                    if node.text:
                        chunks.append(node.text)
                text = _normalize_text("".join(chunks))
                if text:
                    para_lines.append(text)

            lines.append(f"## Slide {index}")
            lines.append("")
            if not para_lines:
                lines.append("_(no text)_")
            else:
                for item in para_lines:
                    lines.append(f"- {item}")
            lines.append("")

    rendered = "\n".join(lines).strip()
    return rendered if rendered else f"# {path.name}\n\n_(No extractable text found.)_"


def _extract_pdf_markdown(path: Path, max_pages: int) -> tuple[str | None, str | None]:
    """Extract PDF text as markdown. Returns (content, error)."""
    try:
        from pypdf import PdfReader
    except Exception:
        return None, (
            "PDF extraction requires optional dependency 'pypdf'. "
            "Install it with: ./venv/bin/pip install pypdf"
        )

    reader = PdfReader(str(path))
    lines: list[str] = [f"# {path.name}", ""]
    pages = list(reader.pages)[:max_pages]
    if not pages:
        lines.append("_(empty PDF)_")
        return "\n".join(lines), None

    extracted_any = False
    for idx, page in enumerate(pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        lines.append(f"## Page {idx}")
        lines.append("")
        if text:
            lines.append(text)
            extracted_any = True
        else:
            lines.append("_(no extractable text)_")
        lines.append("")

    if not extracted_any:
        lines.append("_(No extractable text found in selected pages.)_")
    return "\n".join(lines).strip(), None


class PdfExtractTool(Tool):
    """Extract PDF text and return markdown."""

    name = "pdf_extract"
    description = "Extract text from a PDF file and return Markdown content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the PDF file"},
            "max_chars": {"type": "number", "description": "Maximum markdown characters to return"},
            "max_pages": {"type": "number", "description": "Maximum pages to extract (default 100)"},
        },
        "required": ["path"],
    }

    async def execute(
        self,
        path: str,
        max_chars: int = 120000,
        max_pages: int = 100,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute PDF extraction."""
        del kwargs
        file_path, error = _require_existing_file(path)
        if error:
            return ToolResult(success=False, error=error)
        try:
            content, pdf_error = await asyncio.to_thread(
                _extract_pdf_markdown,
                file_path,
                max(1, int(max_pages)),
            )
            if pdf_error:
                return ToolResult(success=False, error=pdf_error)
            assert content is not None
            return ToolResult(success=True, content=_truncate_markdown(content, max(1, int(max_chars))))
        except Exception as e:
            log.error("PDF extraction failed", path=path, error=str(e))
            return ToolResult(success=False, error=str(e))


class DocxExtractTool(Tool):
    """Extract DOCX text and return markdown."""

    name = "docx_extract"
    description = "Extract text from a DOCX file and return Markdown content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the DOCX file"},
            "max_chars": {"type": "number", "description": "Maximum markdown characters to return"},
        },
        "required": ["path"],
    }

    async def execute(self, path: str, max_chars: int = 120000, **kwargs: Any) -> ToolResult:
        """Execute DOCX extraction."""
        del kwargs
        file_path, error = _require_existing_file(path)
        if error:
            return ToolResult(success=False, error=error)
        try:
            content = await asyncio.to_thread(_extract_docx_markdown, file_path)
            return ToolResult(success=True, content=_truncate_markdown(content, max(1, int(max_chars))))
        except KeyError:
            return ToolResult(success=False, error="Invalid DOCX: missing word/document.xml")
        except zipfile.BadZipFile:
            return ToolResult(success=False, error="Invalid DOCX: file is not a valid ZIP package")
        except Exception as e:
            log.error("DOCX extraction failed", path=path, error=str(e))
            return ToolResult(success=False, error=str(e))


class XlsxExtractTool(Tool):
    """Extract XLSX sheets and return markdown."""

    name = "xlsx_extract"
    description = "Extract worksheet content from an XLSX file and return Markdown tables."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the XLSX file"},
            "max_chars": {"type": "number", "description": "Maximum markdown characters to return"},
            "max_rows": {"type": "number", "description": "Maximum rows per sheet (default 200)"},
        },
        "required": ["path"],
    }

    async def execute(
        self,
        path: str,
        max_chars: int = 120000,
        max_rows: int = 200,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute XLSX extraction."""
        del kwargs
        file_path, error = _require_existing_file(path)
        if error:
            return ToolResult(success=False, error=error)
        try:
            content = await asyncio.to_thread(
                _extract_xlsx_markdown,
                file_path,
                max(1, int(max_rows)),
            )
            return ToolResult(success=True, content=_truncate_markdown(content, max(1, int(max_chars))))
        except KeyError as e:
            return ToolResult(success=False, error=f"Invalid XLSX: missing required part ({e})")
        except zipfile.BadZipFile:
            return ToolResult(success=False, error="Invalid XLSX: file is not a valid ZIP package")
        except Exception as e:
            log.error("XLSX extraction failed", path=path, error=str(e))
            return ToolResult(success=False, error=str(e))


class PptxExtractTool(Tool):
    """Extract PPTX slides and return markdown."""

    name = "pptx_extract"
    description = "Extract slide text from a PPTX file and return Markdown content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the PPTX file"},
            "max_chars": {"type": "number", "description": "Maximum markdown characters to return"},
            "max_slides": {"type": "number", "description": "Maximum slides to extract (default 200)"},
        },
        "required": ["path"],
    }

    async def execute(
        self,
        path: str,
        max_chars: int = 120000,
        max_slides: int = 200,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute PPTX extraction."""
        del kwargs
        file_path, error = _require_existing_file(path)
        if error:
            return ToolResult(success=False, error=error)
        try:
            content = await asyncio.to_thread(
                _extract_pptx_markdown,
                file_path,
                max(1, int(max_slides)),
            )
            return ToolResult(success=True, content=_truncate_markdown(content, max(1, int(max_chars))))
        except zipfile.BadZipFile:
            return ToolResult(success=False, error="Invalid PPTX: file is not a valid ZIP package")
        except Exception as e:
            log.error("PPTX extraction failed", path=path, error=str(e))
            return ToolResult(success=False, error=str(e))
