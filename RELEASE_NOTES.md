# Captain Claw v0.3.3.7 Release Notes

**Release title:** Google Drive Scale Loop + Sheets & Slides Support

**Release date:** 2026-03-08

## Highlights

The scale loop (batch-processing pipeline) now fully supports Google Drive files. When processing a list of Drive files, the scale loop automatically maps item names to file IDs from session context, routes Google-native files (Docs, Sheets, Slides) through `docs_read`, and downloads uploaded files (PDF, DOCX, etc.) for local extraction — all without manual file-ID handling.

Google Sheets and Presentations are now exported as XLSX and PPTX respectively (instead of CSV/plain text), preserving all sheets and slides. The existing `xlsx_extract` and `pptx_extract` extractors handle the content extraction.

## New Features

### Scale Loop Google Drive Integration
- The scale loop micro-loop now builds a Google Drive file map from `drive_list` / `drive_search` results in the current session
- Google-native files (Docs, Sheets, Slides) are routed to `docs_read` for inline content extraction
- Uploaded files (PDF, DOCX, XLSX, PPTX, TXT) are downloaded via `drive_download` and extracted with the appropriate local tool
- Three-level name matching: exact → case-insensitive → substring, so items like "My Report" match "My Report.pdf" in Drive
- No manual file-ID handling required — the scale loop resolves everything automatically

### Google Sheets — Full Multi-Sheet Support via XLSX Export
- `docs_read` now exports Google Sheets as XLSX (was text/markdown, which failed)
- Content extracted with `_extract_xlsx_markdown` — all sheets preserved as markdown tables
- `drive_download` also updated: Sheets export as XLSX instead of CSV

### Google Presentations — Full Multi-Slide Support via PPTX Export
- `docs_read` now exports Google Presentations as PPTX (was text/plain)
- Content extracted with `_extract_pptx_markdown` — all slides preserved as markdown
- `drive_download` also updated: Presentations export as PPTX instead of plain text

## Bug Fixes

### Scale Loop Output Path
- Scale loop output files now go to `saved/tmp/<session>/` (via the write tool's normal session scoping) instead of the previous `output/<session>/` path that bypassed the saved directory structure

### docs_read Export Fallback
- The markdown export fallback in `docs_read` now catches any export error, not just errors containing the word "markdown" — fixes edge cases where Google's export API returns unexpected error messages

## Internal

- New `_GOOGLE_NATIVE_MIMETYPES` constant in `agent_scale_loop_mixin.py` — defines Google Workspace MIME types handled via `docs_read`
- New `_build_gdrive_file_map()` method — scans session messages for `drive_list` / `drive_search` tool results, builds `{name → {id, mimeType}}` map
- New `_lookup_gdrive_file()` static method — three-level name matching (exact, case-insensitive, substring)
- New `_BINARY_EXPORT_MAP` class constant in `gws.py` — maps Sheets/Presentations MIME types to XLSX/PPTX export formats
- New `_docs_read_binary_export()` method — exports to temp file, extracts with existing extractors, returns content inline, cleans up temp file
- Updated `_docs_read()` — fetches `mimeType` in metadata, routes Sheets/Presentations to binary export, improved fallback logic
- Updated `_run_scale_micro_loop()` — builds GDrive file map at initialization, adds GDrive override block in extraction step
- Updated `_resolve_scale_output_path()` — simplified to return bare filenames for write tool session scoping
- Updated `drive_download` export map — Sheets→XLSX, Presentations→PPTX (consistent with `docs_read`)
- Updated `README.md` and `USAGE.md` — documentation for Google Drive scale loop support and Sheets/Slides export formats
- 2 files changed: `agent_scale_loop_mixin.py`, `tools/gws.py`
