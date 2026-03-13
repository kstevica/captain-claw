# Summary: google_mail.py

# Google Mail Tool Summary

## Summary
A read-only Gmail integration tool that leverages the Gmail REST API v1 to enable email reading, searching, and thread management through an async HTTP client. Implements OAuth2 Bearer token authentication with scope validation, supporting five core actions: listing messages, searching with Gmail operators, reading individual messages, retrieving threads, and listing labels. The tool handles MIME multipart parsing, HTML-to-text conversion, and graceful error handling for common API failures.

## Purpose
Solves the problem of programmatic email access within an agent-based system (Captain Claw) by providing a safe, read-only interface to Gmail without exposing send/modify/delete capabilities. Enables AI agents to retrieve, search, and analyze email content while maintaining strict permission boundaries through OAuth2 scope enforcement (`gmail.readonly`).

## Most Important Functions/Classes/Procedures

1. **`GoogleMailTool` (class)**
   - Main tool class inheriting from `Tool` base; orchestrates all Gmail operations through the `execute()` dispatcher method. Manages async HTTP client lifecycle and routes actions to appropriate handlers. Defines tool metadata (name, description, parameters schema) for agent integration.

2. **`_get_access_token()` (async method)**
   - Retrieves valid Google OAuth access tokens via `GoogleOAuthManager`, validates that `gmail.readonly` scope is granted, and raises descriptive errors if authentication is missing or insufficient. Critical security checkpoint ensuring proper authorization before any API calls.

3. **`_parse_message()` (static method)**
   - Transforms raw Gmail API JSON responses into normalized dictionaries with extracted headers (From, To, Subject, Date, etc.), recursively parsed MIME parts, decoded body content (text/HTML), attachment metadata, and label flags (unread, starred). Handles base64 decoding and HTML entity unescaping.

4. **`_extract_parts()` (static method)**
   - Recursively traverses MIME multipart structures to isolate text/HTML body content and attachment information. Decodes base64-encoded payloads and categorizes parts by MIME type, supporting complex nested email structures with multiple body alternatives and attachments.

5. **`_action_search()` / `_action_list_messages()` / `_action_read_message()` / `_action_get_thread()` (async methods)**
   - Five action handlers implementing core Gmail operations: search with Gmail query syntax (from:, subject:, has:attachment, etc.), list messages from labels, fetch full message content, retrieve entire threads, and enumerate labels. Each validates inputs, calls Gmail API endpoints, and formats results for agent consumption.

6. **`_html_to_text()` (static method)**
   - Converts HTML email bodies to plain text via regex-based tag stripping, entity decoding, and whitespace normalization. Preserves semantic structure by converting block-level tags to newlines and collapsing excessive blank lines.

7. **`_handle_http_error()` (static method)**
   - Centralized HTTP error handler mapping Gmail API status codes (401 auth expired, 403 permission denied, 404 not found, 429 rate limit) to user-friendly error messages with actionable remediation guidance.

## Architecture & Dependencies

**Key Dependencies:**
- `httpx` (async HTTP client) — handles all Gmail API communication with 120-second timeout and redirect following
- `captain_claw.google_oauth_manager.GoogleOAuthManager` — manages OAuth2 token lifecycle and scope validation
- `captain_claw.session.get_session_manager()` — provides session context for token storage
- `captain_claw.tools.registry.Tool` — base class defining tool interface and result structure
- Standard library: `base64`, `email.utils`, `html`, `re` — for MIME parsing, header parsing, HTML conversion, and regex operations

**System Role:**
Operates as a read-only email agent tool within the Captain Claw framework, enabling autonomous agents to access Gmail data for information retrieval, analysis, and decision-making tasks. Sits at the intersection of OAuth2 authentication, REST API integration, and MIME email parsing—translating complex Gmail API responses into agent-consumable formats while enforcing strict permission boundaries.

**Configuration:**
- Gmail API endpoint: `https://gmail.googleapis.com/gmail/v1`
- Required OAuth scope: `https://www.googleapis.com/auth/gmail.readonly`
- Max body truncation: 30,000 characters per message
- Rate limit: 50 results per API call (configurable 10-50 range)
- Timeout: 120 seconds per request