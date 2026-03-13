# Summary: document_extract.py

# document_extract.py Summary

Comprehensive document extraction module that converts PDF, DOCX, XLSX, and PPTX files into structured Markdown format with configurable output limits and error handling.

## Purpose

Solves the problem of programmatically extracting and standardizing content from multiple document formats into a unified Markdown representation. Enables AI agents and tools to process office documents with consistent formatting, proper table conversion, and text normalization while respecting resource constraints (character limits, page/row/slide limits).

## Most Important Functions/Classes

1. **`_extract_docx_markdown(path: Path) → str`**
   - Parses DOCX XML structure (via zipfile) to extract paragraphs, headings, lists, and tables. Converts heading styles to Markdown headers (#-######), preserves list markers, and renders tables with proper pipe escaping. Returns complete document as normalized Markdown.

2. **`_extract_xlsx_markdown(path: Path, max_rows: int) → str`**
   - Extracts all sheets from XLSX workbooks by parsing shared strings and cell references. Converts each sheet to a Markdown table with auto-generated column headers (A, B, C...) when missing. Handles inline strings, shared string indices, and boolean values with row limiting.

3. **`_extract_pptx_markdown(path: Path, max_slides: int) → str`**
   - Extracts text from PPTX slides by parsing DrawingML XML. Organizes content as slide-numbered sections with bullet-point text extraction. Respects max_slides parameter for large presentations.

4. **`_extract_pdf_markdown(path: Path, max_pages: int) → tuple[str | None, str | None]`**
   - Uses optional `pypdf` dependency to extract text from PDF pages. Returns (content, error) tuple; gracefully handles missing dependency with installation instructions. Limits extraction to max_pages.

5. **`PdfExtractTool`, `DocxExtractTool`, `XlsxExtractTool`, `PptxExtractTool` (Tool subclasses)**
   - Async-capable Tool implementations that wrap extraction functions with file validation, extension checking, error handling, and output truncation. Each tool enforces single-format usage and provides clear error messages directing users to correct tools or batch processing alternatives.

## Architecture & Dependencies

**Core Dependencies:**
- `pathlib.Path` – file system operations
- `zipfile` – reading Office Open XML formats (DOCX/XLSX/PPTX are ZIP archives)
- `xml.etree.ElementTree` – XML parsing with namespace support
- `asyncio.to_thread()` – non-blocking execution of I/O-bound extraction
- `pypdf` (optional) – PDF text extraction
- `captain_claw.tools.registry.Tool` – base class for tool registration
- `captain_claw.logging` – structured logging

**Key Utilities:**
- `_require_existing_file()` – validates file paths with relative-to-absolute resolution against runtime workspace root
- `_normalize_text()` – collapses whitespace while preserving line breaks
- `_escape_md_cell()` – escapes pipe characters in table cells to prevent Markdown parsing errors
- `_truncate_markdown()` – intelligently truncates output with "[truncated]" suffix

**Design Patterns:**
- **Namespace-aware XML parsing** – handles Office Open XML namespaces (w:, x:, a:, r:) consistently
- **Async execution** – all Tool.execute() methods use `asyncio.to_thread()` to prevent blocking on file I/O
- **Graceful degradation** – missing optional dependencies (pypdf) return helpful error messages rather than crashing
- **Format-specific validation** – each tool checks file extension and rejects wrong formats with actionable guidance
- **Streaming-friendly output** – Markdown format enables progressive rendering and easy downstream processing

**System Role:**
Acts as a document ingestion layer for an AI agent framework (captain_claw), enabling natural language processing over office documents by converting them to Markdown—a format compatible with LLM tokenization and reasoning.