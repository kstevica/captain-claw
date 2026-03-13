# Summary: gws.py

# GWS.PY — Google Workspace CLI Tool Integration

## Summary

A comprehensive Python wrapper around the Google Workspace CLI (`gws` binary) that enables LLM agents to interact with Google Drive, Docs, Gmail, and Calendar through a unified tool interface. The module handles authentication, command execution, output parsing, and content cleaning (stripping base64 images, decoding MIME parts, extracting attachments) to produce agent-friendly text representations of workspace data.

## Purpose

Solves the problem of integrating Google Workspace services into agentic workflows by:
- Abstracting the `gws` CLI binary into structured tool actions
- Automatically handling pagination for large result sets
- Converting complex Gmail/Drive API responses into readable text
- Managing file downloads, exports, and local storage
- Cleaning up binary/image data that would bloat LLM context windows
- Providing both JSON and file-based output modes for flexibility

## Most Important Functions/Classes/Procedures

### 1. **`GwsTool` (Class)**
The main Tool subclass implementing 16+ actions across Drive, Docs, Gmail, and Calendar. Dispatches requests via `execute()` to action-specific handlers. Manages binary resolution, timeout handling, and error recovery. Key attributes: `name="gws"`, `timeout_seconds=180.0`, comprehensive parameter schema for all actions.

### 2. **`_run_gws()` (Async Method)**
Core command executor that shells out to the `gws` binary with JSON formatting, captures stdout/stderr, enforces timeouts via `asyncio.wait_for()`, and streams output via callback. Returns `ToolResult` with success flag and content. Handles authentication errors with helpful hints.

### 3. **`_drive_paginated_list()` (Async Method)**
Handles pagination through Drive file listings with automatic page-token chaining. Supports two modes: (a) accumulate JSON in memory and return capped output, or (b) stream results directly to disk via `output_file` parameter while reporting progress. Formats files as `<name>, <link>, <size>, <type>` for readability.

### 4. **`_drive_recursive_list()` (Async Method)**
Implements efficient recursive Drive traversal by fetching all files in one sweep (no per-folder API calls), building a folder-id→name lookup, and reconstructing full paths via parent-chain walking with memoization. Writes results to disk with folder paths preserved. Tracks file/folder counts and pages fetched.

### 5. **`_clean_gmail_message()` (Function)**
Converts Gmail API message payloads (format=full) to LLM-friendly text by: decoding base64url MIME parts, extracting headers (From/To/Subject/Date), walking multipart structures, converting HTML to plain text via regex, saving image attachments to workspace, and building a clean text summary with metadata and attachment info.

### 6. **`_clean_gmail_thread()` (Function)**
Wraps `_clean_gmail_message()` to process entire email threads, extracting thread ID, subject, message count, and formatting each message with a numbered header. Produces a single readable text block suitable for agent consumption.

### 7. **`_docs_read()` (Async Method)**
Reads Google Docs/Sheets/Slides by exporting to markdown (Docs) or XLSX/PPTX (Sheets/Slides). For binary formats, delegates to `_docs_read_binary_export()` which extracts content via `_extract_xlsx_markdown()` or `_extract_pptx_markdown()`. Strips base64 images from markdown exports to prevent context bloat.

### 8. **`_drive_download()` (Async Method)**
Downloads/exports files from Drive, auto-detecting MIME type and choosing export format (markdown for Docs, XLSX for Sheets, PPTX for Presentations). Resolves output paths relative to workspace root or saved directory. Cleans up gws side-effect files (`download.bin`) and strips base64 images from text exports.

### 9. **`_mail_read()` and `_mail_read_thread()` (Async Methods)**
Fetch full email messages/threads via Gmail API and clean them using `_clean_gmail_message()` / `_clean_gmail_thread()`. Save image attachments to workspace directory. Return inline text content (not file references) per tool contract.

### 10. **`_strip_base64_images()` (Function)**
Regex-based utility that removes inline base64-encoded image data (data:image/...;base64,...) from text, replacing with `[image]` placeholder. Logs size reduction for debugging. Prevents hundreds of KB of base64 bloat in exported markdown.

## Architecture & Dependencies

**Core Dependencies:**
- `asyncio` — async subprocess execution, timeouts, stream reading
- `json` — parsing gws JSON output and building API params
- `base64` — decoding Gmail base64url payloads
- `pathlib.Path` — file I/O and path resolution
- `re` — regex for HTML-to-text, base64 image stripping, MIME parsing
- `tempfile` — temporary files for drive_create with content
- `shutil.which()` — binary PATH resolution

**Internal Dependencies:**
- `captain_claw.config.get_config()` — custom gws binary path, workspace paths
- `captain_claw.logging.get_logger()` — structured logging
- `captain_claw.tools.registry.Tool, ToolResult` — base class and result type
- `captain_claw.tools.document_extract._extract_xlsx_markdown()`, `_extract_pptx_markdown()` — binary export extraction

**System Requirements:**
- `gws` CLI binary installed globally or via config path
- `gws auth login` completed (OAuth credentials cached)
- Node.js/npm (for `npm install -g @googleworkspace/cli`)

## Key Design Patterns

1. **Async/await throughout** — all I/O operations use `asyncio.create_subprocess_exec()` with timeout enforcement
2. **Pagination abstraction** — Drive list/search automatically chain `nextPageToken` across multiple API calls
3. **Dual output modes** — results can be returned as JSON (capped at 60KB) or written directly to disk with streaming progress
4. **Content cleaning** — Gmail/Docs exports automatically strip base64 images and decode MIME parts to prevent context bloat
5. **Error recovery** — auth failures detected and user prompted to run `gws auth login`
6. **File path resolution** — relative paths resolved against workspace root for consistency across sessions
7. **Attachment handling** — Gmail images saved to workspace `saved/mail_attachments/` with collision avoidance (counter suffix)

## Notable Implementation Details

- **Gmail MIME parsing** — recursive `_walk()` function traverses multipart structures, decodes text/plain and text/html separately, handles attachments with fallback to snippet
- **HTML-to-text** — removes `<style>`, `<script>`, converts `<br>`, `</p>`, `</div>` to newlines, strips tags, decodes HTML entities
- **Spreadsheet export strategy** — exports Sheets as XLSX (not CSV) to preserve all sheets; Presentations as PPTX to preserve all slides
- **gws side-effect cleanup** — `gws export` creates `download.bin` in cwd; tool explicitly removes it after each operation
- **Folder path reconstruction** — recursive listing builds folder tree via memoized parent-chain walking rather than recursive API calls (more efficient)
- **Timeout enforcement** — 120s default, 180s tool-level timeout; `asyncio.wait_for()` kills process on timeout
- **Stream callback integration** — progress updates streamed to caller via `_stream_callback` for long-running operations (pagination, recursive listing)