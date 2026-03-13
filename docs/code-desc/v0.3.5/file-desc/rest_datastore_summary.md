# Summary: rest_datastore.py

# REST Datastore Handler Summary

**Summary:** REST API handler module providing comprehensive CRUD operations and data management for a browser-based datastore UI. Implements 20+ async endpoints covering table management, row operations, schema mutations, SQL execution, data protection, file import/export, and real-time chat broadcasting of import events.

**Purpose:** Bridges the web UI frontend with the underlying datastore manager, translating HTTP requests into datastore operations while handling validation, error management, file streaming, and format conversions (CSV/XLSX/JSON). Enforces protection rules and broadcasts import summaries to connected clients.

---

## Most Important Functions/Classes/Procedures

### 1. **`upload_and_import(server, request)`**
Handles multipart file uploads (CSV/XLSX), parses headers, intelligently matches against existing tables using similarity scoring, and either appends to matched tables or creates new ones. Streams large files to temp storage, broadcasts import summaries via WebSocket, and cleans up temporary files. Core logic: file validation → header parsing → table matching → conditional append/create → chat broadcast.

### 2. **`query_rows(server, request)`**
Executes paginated queries with WHERE filtering, ORDER BY sorting (ASC/DESC), limit/offset pagination (max 500 rows). Converts result rows from arrays to dictionaries for JavaScript consumption. Accepts JSON-encoded WHERE clauses via query parameters. Returns column metadata alongside row data.

### 3. **`export_table(server, request)`**
Multi-format export endpoint supporting CSV, JSON, and XLSX. Respects `max_export_rows` config limit. Handles format-specific serialization: CSV via Python csv module, JSON with dict zipping, XLSX via temp file writing. Sets appropriate Content-Disposition headers for browser downloads.

### 4. **`add_protection(server, request)` / `remove_protection(server, request)`**
Manages granular data protection at table/row/column level with configurable protection levels and optional reason tracking. Prevents modifications to protected data by raising `ProtectedError` in downstream operations. Returns 403 Forbidden when protection violations occur.

### 5. **`insert_rows(server, request)` / `update_rows(server, request)` / `delete_rows(server, request)`**
Standard CRUD operations for row data. Insert accepts array of row objects, update requires `set_values` dict and WHERE clause, delete requires WHERE clause. All three check for `ProtectedError` and return 403 status. Update/delete return affected row counts.

---

## Architecture & Dependencies

**Key Dependencies:**
- `aiohttp.web` — async HTTP request/response handling
- `captain_claw.datastore` — datastore manager singleton (`get_datastore_manager()`) and `ProtectedError` exception
- `captain_claw.logging` — structured logging via `get_logger()`
- `captain_claw.web_server.WebServer` — server instance for WebSocket broadcasting
- Standard library: `csv`, `json`, `tempfile`, `pathlib`, `io` for file operations

**Error Handling Pattern:**
- Validates JSON body parsing with try/except → 400 Bad Request
- Validates required fields → 400 Bad Request
- Catches `ProtectedError` → 403 Forbidden with `"protected": True` flag
- Generic exceptions → 400 Bad Request with error message
- File operations wrapped in try/finally for cleanup

**Response Format:**
All responses use `web.json_response()` with custom `_JSON_DUMPS` serializer (converts non-JSON-serializable objects via `str()`). Table metadata consistently includes: name, columns (with name/type/position), row_count, created_at, updated_at.

**File Handling:**
- Multipart streaming to temp files with 8KB chunks
- Automatic cleanup in finally blocks
- Supports .csv and .xlsx with format detection via file extension
- Temp files created via `tempfile.mkstemp()` with explicit suffix

**Protection System:**
Integrates with datastore's protection layer—operations on protected tables/rows/columns raise `ProtectedError`, caught and converted to 403 responses. Protection metadata includes level, optional row_id, col_name, and reason.

**Chat Broadcasting:**
`upload_and_import()` calls `server._broadcast()` with formatted Markdown summary including action type, table name, row count, column list, and warnings. Message role set to "assistant" for UI rendering.