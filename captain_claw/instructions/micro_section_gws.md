Google Workspace (gws):
- Use the gws tool for all Google Drive, Docs, Calendar, and Gmail operations. Never web_fetch for Google Workspace content.
- Drive: drive_list, drive_search, drive_download, drive_info, drive_create. Docs: docs_read, docs_append. Calendar: calendar_list, calendar_agenda, calendar_search, calendar_create. Gmail: mail_list, mail_search, mail_read, mail_threads, mail_read_thread.
- For COMPLEX ops (recursive folder listing, bulk processing), write a Python script calling gws CLI via subprocess with timeout=60 and progress output.
- MANDATORY auth failure policy: If gws fails with auth/credentials error, STOP immediately. Tell user to run `gws auth login`. Do NOT retry or fix auth programmatically.