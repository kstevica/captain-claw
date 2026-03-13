# Summary: rest_image_upload.py

# rest_image_upload.py Summary

**Summary:**
REST API handler for uploading images to be attached to chat messages. Processes multipart form uploads, validates file types against a whitelist of image extensions, and persists files to a workspace-organized directory structure with timestamped filenames.

**Purpose:**
Solves the problem of securely receiving image uploads from frontend clients and storing them in a predictable, session-organized location within the workspace filesystem. Returns the absolute file path to the frontend so images can be referenced and attached to chat conversations.

**Most Important Functions/Classes/Procedures:**

1. **`upload_image(server: WebServer, request: web.Request) -> web.Response`**
   - Main async handler for POST /api/image/upload endpoint. Orchestrates the entire upload workflow: multipart parsing, file validation, chunk reading, filesystem persistence, and response generation. Implements comprehensive error handling with specific HTTP status codes (400 for validation failures, 500 for runtime errors).

2. **File Type Validation (`_IMAGE_EXTENSIONS` constant + extension check)**
   - Whitelist-based validation restricting uploads to {.png, .jpg, .jpeg, .webp, .gif, .bmp}. Rejects unsupported types with descriptive error messages listing allowed formats.

3. **Safe Filename Generation**
   - Sanitizes original filename stem by removing/replacing non-alphanumeric characters (except `-_.`), truncates to 60 characters, and appends UTC timestamp (YYYYMMdd-HHMMSS) with original extension to ensure uniqueness and prevent filesystem conflicts.

4. **Directory Structure Organization**
   - Creates session-aware storage hierarchy: `workspace/saved/media/{session_id}/`. Falls back to "uploads" directory if no active session exists. Uses `mkdir(parents=True, exist_ok=True)` for safe concurrent directory creation.

5. **Multipart Stream Processing**
   - Iteratively reads file chunks (8KB buffers) from aiohttp multipart reader, accumulates in list, and joins into single bytes object. Validates non-empty file content before persistence.

**Architecture Notes:**
- Dependency on `captain_claw.web_server.WebServer` for session context and `captain_claw.config` for workspace path resolution
- Uses aiohttp's async multipart API for non-blocking I/O
- Structured logging via `captain_claw.logging` for audit trail
- Returns JSON responses with file metadata (path, original filename, size) for frontend consumption