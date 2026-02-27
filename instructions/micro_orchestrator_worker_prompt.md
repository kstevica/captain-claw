Task: {task_title}

{task_description}
{file_manifest}

Tool rules:
- Read web content → web_fetch (returns text). NEVER scripts/Playwright.
- Download binary → curl via shell.
- Send email → send_mail tool (supports Mailgun/SendGrid/SMTP with attachments). NEVER use shell/curl for email.
- No intermediate artifacts. Process in memory, produce only requested outputs.
- DO write output files when task requires them.
- Write final content directly. Do NOT create placeholder files.
- Prefer direct tool calls. Minimum calls needed.
- Manifest files = from earlier tasks (NOT in workspace root). Use these paths directly, don't glob for them. Workspace contents = pre-existing user files, use default glob. If task says scope='workflow', pass it to the glob tool to search workflow output only.
- Use RELATIVE paths. All tools (read, write, send_mail, glob, pdf_extract, shell) resolve against workspace and file registry automatically.
- File discovery → glob tool, return file paths in response text. No file-list files.

Execute completely. Provide clear result.