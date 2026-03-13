# Summary: rest_files.py

# rest_files.py Summary

Provides comprehensive REST API endpoints for file browsing, content retrieval, media serving, and document export functionality within the Captain Claw web server. Implements security-enforced file access with dual-source file discovery (in-memory registries + SQLite persistence) and supports markdown-to-PDF/DOCX conversion with rich formatting.

## Purpose

Solves the problem of exposing a secure, queryable file management interface to web clients while maintaining strict access controls (workspace boundary enforcement), handling multiple file sources (agent/orchestrator registries and filesystem scans), and enabling document export with formatting preservation.

## Most Important Functions/Classes/Procedures

1. **`_collect_files(server: WebServer) → list[dict]`**
   - Merges file metadata from in-memory agent/orchestrator registries and SQLite persistence layer. Returns deduplicated, sorted file list with enriched metadata (size, mime type, modification time). Core aggregation mechanism for file discovery across session boundaries.

2. **`list_session_files(server, request) → web.Response`**
   - GET endpoint returning files for a specific session. Combines registry lookups with filesystem scans of `workspace/saved/<category>/<session_id>/` and `workspace/output/<session_id>/` directories. Normalizes session IDs and handles missing registry entries by discovering files directly from disk.

3. **`get_file_content(server, request) → web.Response`**
   - GET `/api/files/content` endpoint serving text file content with size limits (2 MB max). Implements dual security checks: registry membership validation and workspace boundary enforcement. Handles encoding errors gracefully with UTF-8 fallback.

4. **`export_md(server, request) → web.Response`**
   - POST `/api/files/export` endpoint converting markdown to PDF (via WeasyPrint) or DOCX (via python-docx). Parses markdown tables, headings, lists, and inline formatting (bold/italic/code). Applies styled CSS for PDF output and programmatic formatting for DOCX.

5. **`serve_media(server, request) → web.Response`**
   - GET `/api/media` endpoint serving media files (images, audio, video) from `saved/` or `output/` directories. Bypasses registry lookup but enforces extension whitelist and workspace boundary checks. Includes cache-control headers for browser optimization.

## Architecture & Dependencies

**File Organization:**
- Text file detection via extension/filename whitelists (`_TEXT_EXTENSIONS`, `_TEXT_FILENAMES`)
- Media extension whitelist (`_MEDIA_EXTENSIONS`) for safe serving
- Metadata enrichment pipeline (`_enrich()`) normalizing file stat data

**Security Model:**
- Path validation via `_is_allowed_path()`: resolves symlinks and enforces `saved/` or `output/` subdirectory containment
- Dual-layer access control: registry membership OR workspace boundary compliance
- File size limits (2 MB text preview, no limit on downloads)

**Data Sources:**
- In-memory: `server.agent._file_registry`, `server._orchestrator._file_registry`
- Persisted: SQLite `file_registry` table via `get_session_manager()`
- Filesystem: Direct `rglob()` scans with category-based organization

**External Dependencies:**
- `aiohttp.web`: async HTTP request/response handling
- `markdown_it`: markdown parsing for HTML conversion
- `weasyprint`: HTML-to-PDF rendering with CSS styling
- `python-docx`: DOCX document generation with table/formatting support
- `mimetypes`: MIME type detection from file extensions

**Key Constants:**
- `_MAX_TEXT_PREVIEW_BYTES = 2 MB`: prevents memory exhaustion on large files
- Session ID normalization: alphanumeric + `._-` characters, fallback to "default"
- Category directories: downloads, media, output, scripts, showcase, skills, summaries, tmp, tools

**Async Pattern:**
All handlers are async coroutines accepting `(server: WebServer, request: web.Request)` and returning `web.Response` objects, enabling non-blocking I/O for file operations and database queries.