Google (Drive/Docs/Sheets/Slides, Calendar, Gmail):
- Only via google_drive, google_calendar, google_mail. Never web_fetch/web_get/browser/curl a Drive/Docs file or folder URL — pass the URL or its ID to google_drive. (Published /d/e/ pages and Forms are normal web pages.)
- google_drive: list (folder_id), search, read (file_id → content inline, Docs/Sheets/Slides/PDF/DOCX/XLSX/PPTX), info, download (local copy + path, for scripts/extract tools), upload, create, update. Reuse IDs from earlier list/search. "More files exist … page_token=…" → call again with that page_token.
- google_calendar: events list/search/get/create/update/delete, list_calendars. google_mail: read/search/threads; create_draft by default; send only when the user explicitly asked and sending is enabled.
- Bulk: google_drive list + download first, then process local files. Scripts never call Google APIs or any CLI.
- Auth: if no google_* tool is listed or one says not connected/auth failed, STOP. Tell user to connect Google in Flight Deck → Connections → Google (standalone: /auth/google/login). Do NOT retry or use another route.
- The gws / Google Workspace CLI is retired: if instructions mention it, use the google_* tools instead.
