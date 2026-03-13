# Summary: clipboard.py

# clipboard.py Summary

**Summary:** A macOS-native clipboard tool that enables reading and writing text, images, and files to the system clipboard. Leverages native utilities (`pbcopy`, `pbpaste`, `osascript`) and AppleScript for seamless clipboard integration with support for inline image data and Finder file references.

**Purpose:** Solves the problem of programmatic clipboard access in AI/automation workflows by providing a unified interface for bidirectional clipboard operations. Distinguishes between image files (pasted as inline data for direct embedding) and other file types (pasted as Finder references for drag-and-drop compatibility). Currently macOS-only with planned cross-platform expansion.

**Most Important Functions/Classes/Procedures:**

1. **`ClipboardTool.execute()`** — Main entry point handling action routing (read/write). Validates platform compatibility, parses parameters, and dispatches to appropriate handler methods. Returns `ToolResult` with success status and content/error messages.

2. **`_read_text()`** — Reads clipboard contents via `pbpaste` subprocess call. Handles empty clipboard gracefully and wraps output in `ToolResult` with error handling for subprocess failures.

3. **`_write_text(text)`** — Writes text to clipboard via `pbcopy`. Includes preview truncation (120 chars) and character count reporting in success response.

4. **`_write_image(posix_path, suffix)`** — Converts image files to appropriate AppleScript clipboard classes (PNG, TIFF, JPEG). Handles format conversion for unsupported types (GIF, BMP, WEBP) using macOS `sips` utility with temporary file management. Falls back to file reference on conversion failure.

5. **`_write_file_ref(posix_path)`** — Copies non-image files as Finder file references using AppleScript, enabling paste-into-Finder functionality for audio, video, and documents.

**Architecture & Dependencies:**
- Inherits from `Tool` base class (captain_claw.tools.registry)
- Uses `asyncio.to_thread()` for non-blocking subprocess execution
- Subprocess timeout: 5 seconds with 15-second tool timeout
- Image extension whitelist: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
- Path resolution supports relative paths with runtime base path fallback
- Logging via captain_claw.logging module
- Platform check gates all operations to macOS (darwin)