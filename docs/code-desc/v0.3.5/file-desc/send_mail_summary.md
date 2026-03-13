# Summary: send_mail.py

# send_mail.py Summary

**Summary:** A comprehensive email dispatch tool supporting three providers (Mailgun, SendGrid, SMTP) with attachment handling, recipient management (to/cc/bcc), and flexible body content (plain text and/or HTML). Integrates with the Captain Claw framework as a callable Tool with async/await patterns and robust file resolution across multiple filesystem contexts.

**Purpose:** Solves the problem of sending emails from automated workflows with provider flexibility, attachment support, and intelligent file path resolution that handles both local filesystem and orchestrated workflow contexts (session-scoped directories, FileRegistry lookups).

**Most Important Functions/Classes/Procedures:**

1. **`SendMailTool` (class)** – Main Tool subclass implementing the email dispatch interface. Registers as "send_mail" tool with JSON schema parameters for to/cc/bcc/subject/body/html/attachments. Manages httpx.AsyncClient for HTTP-based providers and delegates to provider-specific methods.

2. **`execute()` (async method)** – Primary entry point orchestrating the entire email workflow: validates inputs (recipients, subject, body/html), loads configuration, resolves attachment file paths (with fallback logic for runtime_base, workflow_run_dir, and FileRegistry), dispatches to appropriate provider, and returns formatted ToolResult with delivery confirmation.

3. **`_send_mailgun()` (async method)** – Mailgun HTTP API integration using basic auth (api key), constructs multipart form data with attachments, posts to domain-specific endpoint, returns message ID from response.

4. **`_send_sendgrid()` (async method)** – SendGrid v3 API integration using Bearer token auth, constructs JSON payload with personalizations (to/cc/bcc), base64-encodes attachments, handles 202 Accepted response pattern.

5. **`_send_smtp()` (async method)** – SMTP provider with dual-mode support: prefers native async `aiosmtplib` when available, falls back to synchronous `smtplib` in thread pool. Constructs EmailMessage with multipart alternatives (plain text + HTML), handles TLS/authentication, manages BCC in envelope only.

**Architecture & Dependencies:**
- **Framework Integration:** Extends `Tool` base class from `captain_claw.tools.registry`; uses `get_config()` for provider credentials and `get_logger()` for observability
- **Async Pattern:** Full async/await with `httpx.AsyncClient` for HTTP providers; thread pooling via `asyncio.to_thread()` for blocking SMTP operations
- **File Resolution:** Multi-layer fallback system handles absolute paths, runtime-relative paths, workflow-run directories, and FileRegistry cross-session resolution
- **Configuration:** Reads from `cfg.tools.send_mail` with environment variable overrides (MAILGUN_API_KEY, SENDGRID_API_KEY)
- **External Libraries:** httpx (HTTP), aiosmtplib (optional async SMTP), smtplib (stdlib fallback), email.message (MIME construction)