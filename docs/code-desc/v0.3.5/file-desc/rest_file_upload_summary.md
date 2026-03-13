# Summary: rest_file_upload.py

# rest_file_upload.py Summary

**Summary:**
This module implements a REST API endpoint for uploading data files (CSV, XLSX) to a workspace directory. It handles multipart form data, validates file types, sanitizes filenames with timestamps, and returns the saved file path for frontend attachment to chat contexts.

**Purpose:**
Solves the problem of allowing users to upload structured data files into the system for downstream processing (datastore imports, deep memory integration, etc.). Provides a secure, organized file storage mechanism with session-aware directory structure and prevents malicious or unsupported file uploads.

**Most Important Functions/Classes/Procedures:**

1. **`upload_file(server: WebServer, request: web.Request) -> web.Response`**
   - Main async handler for POST /api/file/upload endpoint. Orchestrates the entire upload workflow: multipart parsing, validation, file reading, path determination, sanitization, and persistence. Returns JSON response with saved path or error details.

2. **File Type Validation (lines 54-59)**
   - Extracts file extension and validates against `_ALLOWED_EXTENSIONS` whitelist ({".csv", ".xlsx"}). Returns 400 error with supported types if validation fails—critical security control.

3. **Chunk-Based File Reading (lines 61-67)**
   - Reads uploaded file in 8KB chunks to handle large files efficiently without loading entire content into memory at once. Concatenates chunks into final bytes object with empty file validation.

4. **Destination Path Construction (lines 70-84)**
   - Determines save location using workspace config, session ID (from server.agent.session or fallback to "uploads"), and generates unique filename by combining sanitized stem + UTC timestamp + extension. Creates nested directory structure (workspace/saved/downloads/{session_id}/).

5. **Filename Sanitization (lines 79-80)**
   - Strips unsafe characters from original filename stem, preserving only alphanumeric, hyphens, underscores, and dots. Limits to 60 characters to prevent path traversal and filesystem issues.

**Architecture & Dependencies:**
- **Framework:** aiohttp (async HTTP server)
- **Config:** Integrates with `captain_claw.config` for workspace path resolution
- **Logging:** Uses structured logging via `captain_claw.logging`
- **Type Hints:** Full TYPE_CHECKING support with WebServer forward reference
- **Error Handling:** Comprehensive try-catch with HTTP exception re-raising and detailed error logging