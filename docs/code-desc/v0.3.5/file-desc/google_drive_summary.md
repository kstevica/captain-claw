# Summary: google_drive.py

# Google Drive Tool Summary

**Summary:**
A comprehensive Google Drive integration tool that enables listing, searching, reading, uploading, creating, and updating files via the Google Drive REST API v3. Uses httpx for HTTP requests with OAuth2 Bearer token authentication managed by GoogleOAuthManager, eliminating the need for Google SDK dependencies. Supports both native Google Workspace files (Docs, Sheets, Slides) and standard file formats with intelligent content extraction and export capabilities.

**Purpose:**
Solves the problem of programmatic Google Drive access within the Captain Claw agent framework by providing a unified interface for file operations. Handles the complexity of Google's API including multipart uploads, file format conversions, content extraction from binary formats (PDF, DOCX, XLSX, PPTX), and OAuth token management. Enables agents to browse, search, read, and manipulate Drive files seamlessly.

**Most Important Functions/Classes/Procedures:**

1. **`GoogleDriveTool` (class)**
   - Main tool class implementing the Tool interface. Manages HTTP client lifecycle, dispatches actions to handlers, and coordinates OAuth token retrieval. Exposes seven actions: list, search, read, info, upload, create, update. Handles all error cases with user-friendly messages and manages timeouts (120s).

2. **`execute(action, **kwargs)` (async method)**
   - Entry point dispatcher that validates actions, retrieves OAuth tokens, routes to appropriate handler methods, and catches/converts exceptions into ToolResult objects. Strips injected runtime kwargs before processing.

3. **`_action_read(token, file_id)` (async method)**
   - Intelligent content reader that determines file type and routes to appropriate handler: exports Google Workspace files to markdown/CSV/text, downloads and extracts binary formats using specialized tools, lists folder contents, or downloads plain text files. Enforces 500KB read limit with truncation.

4. **`_multipart_upload(token, metadata, content, content_type, convert)` (async method)**
   - Constructs RFC 2046 multipart/related request bodies and uploads to Google Drive API. Handles both file uploads and Google Workspace document creation with optional format conversion. Returns file metadata including ID and web link.

5. **`_get_access_token()` (async method)**
   - Retrieves valid OAuth access tokens from GoogleOAuthManager, validates Drive scope presence, and raises descriptive RuntimeError if authentication fails or Drive permissions not granted. Provides user guidance for reconnection.

**Architecture & Dependencies:**

- **HTTP Client:** Uses `httpx.AsyncClient` for all API communication with 120s timeout and redirect following
- **OAuth Integration:** Depends on `GoogleOAuthManager` and session manager for token lifecycle management
- **Logging:** Integrates with Captain Claw's logging system via `get_logger()`
- **Tool Registry:** Inherits from `Tool` base class and registers as "google_drive" in tool registry
- **Document Extraction:** Dynamically imports specialized extractors (PdfExtractTool, DocxExtractTool, XlsxExtractTool, PptxExtractTool) for binary format handling
- **File System:** Uses `pathlib.Path` for local file operations and `tempfile` for temporary storage during extraction

**Key Design Patterns:**

- **Action Dispatch:** Handler dictionary maps action strings to async methods, enabling extensibility
- **Format Detection:** MIME type-based routing determines processing strategy (export, extract, download, list)
- **Error Handling:** Comprehensive HTTP status code mapping (401, 403, 404, 429) with context-specific user messages
- **Resource Cleanup:** Temporary files cleaned up in finally blocks; HTTP client closed via `close()` method
- **Size Constraints:** 500KB read limit and 50MB download limit prevent memory exhaustion
- **Scope Validation:** Enforces Drive scope presence before operations to fail fast with clear messaging