# Captain Claw Usage Reference

Comprehensive reference for all commands, tools, configuration options, and features.

For a quick overview and installation guide, see [README.md](README.md).

---

## Table of Contents

- [Installation](#installation)
- [Commands Reference](#commands-reference)
  - [Global Commands](#global-commands)
  - [Session Commands](#session-commands)
  - [Model Commands](#model-commands)
  - [Monitor Commands](#monitor-commands)
  - [Pipeline Commands](#pipeline-commands)
  - [Cron Commands](#cron-commands)
  - [Todo Commands](#todo-commands)
  - [Contacts Commands](#contacts-commands)
  - [Scripts Commands](#scripts-commands)
  - [APIs Commands](#apis-commands)
  - [Skills Commands](#skills-commands)
  - [Orchestrator Commands](#orchestrator-commands)
  - [Admin Commands](#admin-commands)
- [Tools Reference](#tools-reference)
- [Configuration Reference](#configuration-reference)
  - [Config Load Precedence](#config-load-precedence)
  - [model](#model)
  - [context](#context)
  - [memory](#memory)
  - [tools](#tools)
  - [skills](#skills)
  - [guards](#guards)
  - [todo](#todo)
  - [addressbook](#addressbook)
  - [scripts_memory](#scripts_memory)
  - [apis_memory](#apis_memory)
  - [session](#session)
  - [workspace](#workspace)
  - [ui](#ui)
  - [execution_queue](#execution_queue)
  - [logging](#logging)
  - [telegram](#telegram)
  - [slack](#slack)
  - [discord](#discord)
  - [web](#web)
  - [google_oauth](#google_oauth)
  - [orchestrator](#orchestrator)
  - [botport_client](#botport_client)
  - [scale](#scale)
  - [datastore](#datastore-1)
  - [deep_memory](#deep_memory)
- [Guard System](#guard-system)
- [Skills System](#skills-system)
- [Datastore](#datastore)
  - [Datastore Dashboard](#datastore-dashboard)
- [Deep Memory (Typesense)](#deep-memory-typesense)
  - [Deep Memory Dashboard](#deep-memory-dashboard)
- [Memory and RAG](#memory-and-rag)
- [Cross-Session Todo Memory](#cross-session-todo-memory)
- [Cross-Session Address Book](#cross-session-address-book)
- [Cross-Session Script Memory](#cross-session-script-memory)
- [Cross-Session API Memory](#cross-session-api-memory)
- [Personality System](#personality-system)
- [Session Management](#session-management)
- [Chunked Processing Pipeline](#chunked-processing-pipeline)
- [Context Compaction](#context-compaction)
- [Execution Queue](#execution-queue-1)
- [Orchestrator / DAG Mode](#orchestrator--dag-mode)
- [BotPort (Agent-to-Agent)](#botport)
- [Web UI](#web-ui)
- [Remote Integrations](#remote-integrations)
  - [Telegram: Per-User Sessions](#telegram-per-user-sessions)
- [OpenAI-Compatible API Proxy](#openai-compatible-api-proxy)
- [Google OAuth, Drive, Calendar, and Gmail](#google-oauth-drive-calendar-and-gmail)
- [Send Mail](#send-mail)
- [Termux](#termux)
- [File Output Policy](#file-output-policy)
- [Environment Variables Reference](#environment-variables-reference)

---

## Installation

### Requirements

- Python **>= 3.11**

### Install from PyPI

```bash
python -m venv venv
source venv/bin/activate
pip install captain-claw
```

**Optional extras:**

```bash
pip install captain-claw[tts]      # Local text-to-speech (pocket-tts, requires PyTorch)
pip install captain-claw[vector]   # Vector memory / RAG (numpy, scikit-learn)
pip install captain-claw[vision]   # Image resize before LLM calls (Pillow)
```

**ImageMagick** can be used instead of (or alongside) Pillow for image resizing before vision/OCR LLM calls. This is especially useful on Termux/Android where Pillow's JPEG encoder may not work correctly.

```bash
# macOS
brew install imagemagick

# Ubuntu / Debian
sudo apt install imagemagick

# Termux (Android)
pkg install imagemagick
```

If neither Pillow nor ImageMagick is available, images are sent to the LLM as-is (providers resize server-side).

### Install from source

```bash
git clone https://github.com/kstevica/captain-claw
cd captain-claw
python -m venv venv
source venv/bin/activate
pip install -e .
```

Optional extras work the same way from source: `pip install -e ".[tts]"` or `pip install -e ".[vector]"`.

### Development dependencies

```bash
pip install -e ".[dev]"
```

### Entry points

| Command | Description |
|---|---|
| `captain-claw` | Web UI (default) |
| `captain-claw --tui` | Start with terminal UI |
| `captain-claw --port <PORT>` | Override web server port |
| `captain-claw --onboarding` | Re-run first-time setup wizard |
| `captain-claw-web` | Web UI only (standalone entry point) |
| `captain-claw-orchestrate` | Headless orchestrator for cron jobs and scripting |

If the configured port is busy, Captain Claw automatically tries the next available port (up to 10 attempts).

---

## Commands Reference

### Global Commands

| Command | Description |
|---|---|
| `/help` | Show command reference |
| `/config` | Show active configuration (secrets redacted) |
| `/history` | Show recent conversation history |
| `/compact` | Manually compact session context |
| `/clear` | Clear current session messages (blocked if `/session protect on`) |
| `/exit` or `/quit` | Exit Captain Claw |

### Session Commands

| Command | Description |
|---|---|
| `/session` or `/session info` | Show active session details |
| `/sessions` or `/session list` | List recent sessions (up to 20) |
| `/new [name]` | Create and switch to a new session |
| `/session switch <id\|name\|#index>` | Switch to another session |
| `/session rename <new-name>` | Rename the active session |
| `/session description <text>` | Set session description manually |
| `/session description auto` | Auto-generate description from context |
| `/session export [mode]` | Export session history to files |
| `/session protect on\|off` | Toggle memory reset protection |
| `/session model` | Show active model for this session |
| `/session model list` | List allowed models |
| `/session model <id\|#index\|default>` | Set model for this session |
| `/session run <id\|name\|#index> <prompt>` | Run a prompt in another session, then return |
| `/runin <id\|name\|#index> <prompt>` | Alias for `/session run` |
| `/session procreate <p1> <p2> <name>` | Create new session by merging two parent sessions |
| `/session queue info` | Show execution queue settings |
| `/session queue mode <mode>` | Set queue mode |
| `/session queue debounce <ms>` | Set debounce interval |
| `/session queue cap <n>` | Set max queued items |
| `/session queue drop <policy>` | Set overflow policy (old\|new\|summarize) |
| `/session queue clear` | Clear pending queue |

**Export modes:** `chat`, `monitor`, `pipeline`, `pipeline-summary`, `all` (default: `all`). Exports are saved under `saved/showcase/<session-id>/exports/`.

**Procreation:** Compacts both parent session memories, merges into a new child session. Parents are not modified. Progress logged to monitor.

**Protection:** When on, `/clear` is blocked. Use `/session protect off` to re-enable.

### Model Commands

| Command | Description |
|---|---|
| `/models` | List all allowed models with provider and details |
| `/session model` | Show the model assigned to the current session |
| `/session model list` | List allowed models (same as `/models`) |
| `/session model <id\|#index\|provider:model\|default>` | Assign a model to the current session |

Model selection persists per session. Use `default` to revert to the global config model.

### Monitor Commands

| Command | Description |
|---|---|
| `/monitor on\|off` | Toggle the monitor split pane |
| `/monitor trace on\|off` | Log full intermediate LLM responses to monitor/session history |
| `/monitor pipeline on\|off` | Log compact pipeline-only events to session history |
| `/monitor full on\|off` | Show raw tool output instead of compact summaries |
| `/scroll <pane> <action> [n]` | Scroll a pane (pane: `chat\|monitor`, action: `up\|down\|pageup\|pagedown\|top\|bottom`) |
| `/scroll status` | Show current scroll positions |

### Pipeline Commands

| Command | Description |
|---|---|
| `/pipeline` or `/pipeline info` | Show current pipeline mode |
| `/pipeline loop` | Fast/simple mode — direct tool-completion cycle |
| `/pipeline contracts` | Planner + completion gate mode for complex tasks |
| `/planning on` | Legacy alias for `/pipeline contracts` |
| `/planning off` | Legacy alias for `/pipeline loop` |

**Loop mode** (default): The agent calls tools and responds directly. Best for straightforward tasks.

**Contracts mode**: Each turn runs through a task contract planner that decomposes the request, then a completion gate validates the result against requirements. Best for complex multi-step tasks.

### Cron Commands

| Command | Description |
|---|---|
| `/cron "<task>"` | Run a one-off background task |
| `/cron add every <Nm\|Nh> <task\|script\|tool> <payload>` | Schedule an interval job |
| `/cron add daily <HH:MM> <task\|script\|tool> <payload>` | Schedule a daily job (UTC) |
| `/cron add weekly <day> <HH:MM> <task\|script\|tool> <payload>` | Schedule a weekly job (mon-sun, UTC) |
| `/cron list` | List all scheduled jobs |
| `/cron history <job-id\|#index>` | Show execution history for a job |
| `/cron run <job-id\|#index>` | Execute a scheduled job immediately |
| `/cron run script <path>` | Run a saved script immediately |
| `/cron run tool <path>` | Run a saved tool immediately |
| `/cron pause <job-id\|#index>` | Pause a scheduled job |
| `/cron resume <job-id\|#index>` | Resume a paused job |
| `/cron remove <job-id\|#index>` | Delete a scheduled job |

**Job types:**
- `task` (default) — a natural language prompt executed by the agent
- `script` — execute a saved script from the session folder
- `tool` — execute a saved tool call

**Schedule formats:**
- Interval: `every 15m`, `every 2h`
- Daily: `daily 09:00` (UTC)
- Weekly: `weekly mon 09:00` (UTC, days: mon-sun)

Cron is pseudo-cron managed inside the Captain Claw runtime. Guardrails remain active for every execution. Output is tagged with `[CRON ...]` in chat and monitor panes.

### Todo Commands

| Command | Description |
|---|---|
| `/todo` or `/todo list` | List all pending and in-progress to-do items |
| `/todo add <text>` | Add a new to-do item |
| `/todo done <id\|#index>` | Mark a to-do item as done |
| `/todo remove <id\|#index>` | Delete a to-do item |
| `/todo assign <id\|#index> bot\|human` | Reassign responsibility |

The agent can also manage todos via the `todo` tool during conversation. Items are persistent across sessions and can be assigned to `bot` or `human`.

### Contacts Commands

| Command | Description |
|---|---|
| `/contacts` or `/contacts list` | List all contacts sorted by importance |
| `/contacts add <name>` | Add a new contact |
| `/contacts info <id\|#index\|name>` | Show full contact details |
| `/contacts search <query>` | Search contacts by name, organization, or email |
| `/contacts update <id\|#index\|name> <field=value ...>` | Update contact fields |
| `/contacts importance <id\|#index\|name> <1-10>` | Set contact importance (pins value) |
| `/contacts remove <id\|#index\|name>` | Remove a contact |

The agent can also manage contacts via the `contacts` tool during conversation. Contacts are persistent across sessions and support auto-capture from conversation and email recipients.

### Scripts Commands

| Command | Description |
|---|---|
| `/scripts` or `/scripts list` | List all scripts sorted by usage |
| `/scripts add <name> <file_path>` | Register a new script |
| `/scripts info <id\|#index\|name>` | Show full script details |
| `/scripts search <query>` | Search scripts by name, path, or language |
| `/scripts update <id\|#index\|name> <field=value ...>` | Update script fields |
| `/scripts remove <id\|#index\|name>` | Remove a script |

The agent can also manage scripts via the `scripts` tool during conversation. Scripts are persistent across sessions and support auto-capture from the `write` tool when executable file extensions are detected.

### APIs Commands

| Command | Description |
|---|---|
| `/apis` or `/apis list` | List all APIs sorted by usage |
| `/apis add <name> <base_url>` | Register a new API |
| `/apis info <id\|#index\|name>` | Show full API details |
| `/apis search <query>` | Search APIs by name, URL, or description |
| `/apis update <id\|#index\|name> <field=value ...>` | Update API fields |
| `/apis remove <id\|#index\|name>` | Remove an API |

The agent can also manage APIs via the `apis` tool during conversation. APIs are persistent across sessions and support auto-capture from `web_fetch` when API-like URLs are detected.

### Skills Commands

| Command | Description |
|---|---|
| `/skills` | List available user-invocable skills |
| `/skill <name> [args]` | Invoke a skill |
| `/<command> [args]` | Direct alias for a discovered skill command |
| `/skill search <criteria>` | Search the skill catalog |
| `/skill install <github-url>` | Install a skill from GitHub |
| `/skill install <name> [install-id]` | Install skill dependencies |

### Orchestrator Commands

| Command | Description |
|---|---|
| `/orchestrate <request>` | Decompose a complex task into a DAG and run in parallel sessions |

The orchestrator decomposes the request into tasks, builds a dependency graph, assigns sessions, and executes tasks in parallel. See [Orchestrator / DAG Mode](#orchestrator--dag-mode) for details.

### Admin Commands

| Command | Description |
|---|---|
| `/approve user telegram <token>` | Approve a pending Telegram pairing token |
| `/approve user slack <token>` | Approve a pending Slack pairing token |
| `/approve user discord <token>` | Approve a pending Discord pairing token |

These commands are only available in the local CLI.

---

## Tools Reference

### shell

Execute terminal commands with configurable policies.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `command` | string | yes | The command to execute |
| `timeout` | number | no | Timeout in seconds (default: from config) |

Execution policy is controlled by `tools.shell` config: `default_policy` (ask/allow/deny), `allow_patterns`, `deny_patterns`, `blocked` commands.

### read

Read file contents from the filesystem.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | File path to read |
| `limit` | number | no | Max lines to return |
| `offset` | number | no | Line offset to start from |

### write

Write content to files. Output is session-scoped under `saved/`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | File path to write |
| `content` | string | yes | Content to write |
| `mode` | string | no | Write mode (default: overwrite) |

### glob

Find files by pattern.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `pattern` | string | yes | Glob pattern (e.g. `**/*.py`) |
| `path` | string | no | Base directory to search |

### web_fetch

Fetch a URL and return clean readable text. Always operates in text mode — raw HTML is never returned. For raw HTML, use `web_get` instead.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | yes | URL to fetch |
| `max_chars` | number | no | Max output chars (default: 100000) |

### web_get

Fetch a URL and return raw HTML source. Use only when you need the actual HTML markup for scraping, DOM analysis, or CSS selector inspection. For normal page reading, use `web_fetch` instead.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | yes | URL to fetch |
| `max_chars` | number | no | Max output chars (default: 100000) |

### web_search

Search the web via Brave Search API.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Search query |
| `count` | number | no | Results to return (1-20, default: from config) |
| `offset` | number | no | Pagination offset |
| `country` | string | no | Regional ranking (e.g. `US`, `HR`) |
| `search_lang` | string | no | Language filter (e.g. `en`) |
| `freshness` | string | no | Time filter: `pd` (past day), `pw` (past week), `pm` (past month), `py` (past year) |
| `safesearch` | string | no | `off`, `moderate` (default), `strict` |

Requires `BRAVE_API_KEY` environment variable or `tools.web_search.api_key` in config.

### pdf_extract

Extract PDF content to markdown.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path to PDF file |
| `max_chars` | number | no | Max output chars (default: 120000) |
| `max_pages` | number | no | Max pages to extract (default: 100) |

Uses `pypdf`. Output includes `## Page N` headers.

### docx_extract

Extract Word documents to markdown.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path to DOCX file |
| `max_chars` | number | no | Max output chars (default: 120000) |

Preserves headings, lists, and tables. Uses standard library XML parsing (no python-docx dependency).

### xlsx_extract

Extract Excel sheets to markdown tables.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path to XLSX file |
| `max_rows` | number | no | Rows per sheet (default: 200) |
| `max_chars` | number | no | Max output chars (default: 120000) |

Each sheet becomes a markdown table. Uses standard library XML parsing.

### pptx_extract

Extract PowerPoint slides to markdown.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path to PPTX file |
| `max_slides` | number | no | Slides to extract (default: 200) |
| `max_chars` | number | no | Max output chars (default: 120000) |

Output includes `## Slide N` headers with bullet-point text.

### image_gen

Generate images from text prompts using an AI image model (e.g. DALL-E 3, gpt-image-1).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | yes | Text description of the image to generate |
| `size` | string | no | Image dimensions: `1024x1024` (default), `1536x1024` (landscape), `1024x1536` (portrait) |
| `quality` | string | no | Image quality: `auto` (default), `high`, `medium`, `low`, `hd`, `standard` |
| `output_path` | string | no | Output path (normalized under `saved/media/<session-id>/`, saved as `.png`) |

Requires a model with `model_type: "image"` in `model.allowed`. Output saved to `saved/media/<session-id>/`. Generated images are automatically sent back to Telegram users.

### image_ocr

Extract text from images using OCR via a vision-capable LLM.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path to the image file (.png, .jpg, .jpeg, .webp, .gif, .bmp) |
| `prompt` | string | no | Instruction for the vision model (default: "Extract all text from this image.") |
| `max_chars` | number | no | Maximum characters to return (default: 120000) |

Requires a model with `model_type: "ocr"` or `"vision"` in `model.allowed`. Images larger than `max_pixels` (default: 1568px longest edge) are automatically resized and compressed to JPEG at `jpeg_quality` (default: 85) before sending to the LLM. Two resize backends are tried in order: **Pillow** (`pip install captain-claw[vision]`), then **ImageMagick** (`convert` CLI — on Termux: `pkg install imagemagick`). If neither is available, images are sent as-is (LLM providers resize server-side). This reduces token cost and upload time, especially for high-resolution camera photos.

### image_vision

Analyze and describe images using a vision-capable LLM. Supports scene description, object identification, chart reading, and visual Q&A.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Path to the image file (.png, .jpg, .jpeg, .webp, .gif, .bmp) |
| `prompt` | string | no | Question or instruction about the image (default: "Describe this image in detail.") |
| `max_chars` | number | no | Maximum characters to return (default: 120000) |

Requires a model with `model_type: "vision"` in `model.allowed`. Telegram photo attachments are automatically processed through this tool. Same image resizing applies as `image_ocr` above (Pillow → ImageMagick → raw) — configurable via `max_pixels` and `jpeg_quality` in settings.

### pocket_tts

Generate speech audio locally and save as MP3.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Text to convert to speech |
| `voice` | string | no | Voice preset (default: from config or model default) |
| `sample_rate` | number | no | Sample rate in Hz (default: 24000) |
| `bitrate_kbps` | number | no | MP3 bitrate (default: 128) |

Uses `pocket-tts`. Output saved to `saved/media/<session-id>/`.

### send_mail

Send email via SMTP, Mailgun, or SendGrid.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `to` | list[string] | yes | Recipient email addresses |
| `subject` | string | yes | Email subject |
| `body` | string | no | Plain text body |
| `html` | string | no | HTML body |
| `cc` | list[string] | no | CC addresses |
| `bcc` | list[string] | no | BCC addresses |
| `attachments` | list[string] | no | Local file paths to attach |

At least one of `body` or `html` is required. Max attachment size: 25 MB per file.

### google_drive

Interact with Google Drive. Requires Google OAuth connection.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list`, `search`, `read`, `info`, `upload`, `create`, `update` |
| `file_id` | string | for read/info/update | Google Drive file ID |
| `folder_id` | string | no | Folder ID for list/upload (default: root) |
| `query` | string | for search | Search query text |
| `max_results` | number | no | Max results (default: 20, max: 100) |
| `local_path` | string | for upload | Local file path to upload |
| `name` | string | for upload/create | File name |
| `content` | string | for create/update | Text content |
| `mime_type` | string | for create | Target MIME type |
| `order_by` | string | no | Sort order (default: `modifiedTime desc`) |

**Read behavior:** Google Docs export as markdown, Sheets as CSV, Slides as plain text. Office files (DOCX, XLSX, PPTX) are downloaded and parsed by the corresponding extract tools. Plain text files are downloaded directly.

### google_calendar

Interact with Google Calendar. Requires Google OAuth connection with Calendar scope.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list_events`, `search_events`, `get_event`, `create_event`, `update_event`, `delete_event`, `list_calendars` |
| `calendar_id` | string | no | Calendar ID (default: `primary`). Use `list_calendars` to find others. |
| `event_id` | string | for get/update/delete | Event ID |
| `query` | string | for search_events | Free-text search query |
| `summary` | string | for create_event | Event title |
| `description` | string | no | Event description/notes |
| `location` | string | no | Event location |
| `start` | string | for create_event | ISO 8601 datetime (e.g. `2026-02-24T10:00:00+01:00`) or date-only for all-day events (e.g. `2026-02-24`) |
| `end` | string | no | ISO 8601 datetime or date-only. Defaults to start + 1 hour (timed) or start + 1 day (all-day). |
| `timezone` | string | no | IANA timezone (e.g. `Europe/Berlin`). Used when start/end don't include an offset. |
| `attendees` | list[string] | no | Attendee email addresses |
| `reminders` | list[object] | no | Custom reminders: `[{"method": "popup", "minutes": 10}]` |
| `recurrence` | list[string] | no | RRULE strings (e.g. `["RRULE:FREQ=WEEKLY;COUNT=10"]`) |
| `max_results` | number | no | Max results (default: 10, max: 100) |
| `time_min` | string | no | Lower bound for event start time (ISO 8601). Defaults to now for `list_events`. |
| `time_max` | string | no | Upper bound for event end time (ISO 8601) |
| `color_id` | string | no | Event color ID (1-11) |

All-day event end dates are exclusive — an end of `2026-02-25` means the event covers `2026-02-24` only. Uses Google Calendar REST API v3 via httpx (no Google SDK dependency).

### google_mail

Read-only Gmail access. Requires Google OAuth connection with `gmail.readonly` scope. No send, modify, or delete operations.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list_messages`, `search`, `read_message`, `get_thread`, `list_labels` |
| `query` | string | for search | Gmail search query (e.g. `from:alice subject:report`, `is:unread has:attachment`) |
| `message_id` | string | for read_message | Message ID |
| `thread_id` | string | for get_thread | Thread ID |
| `label` | string | no | Label/folder for `list_messages` (default: `INBOX`). Common: `SENT`, `DRAFT`, `STARRED`, `UNREAD`, `SPAM`, `TRASH`. |
| `max_results` | number | no | Max results (default: 10, max: 50) |
| `include_body` | boolean | no | Include message body in list/search results (default: false). Always true for `read_message`. |

Handles MIME multipart emails, extracts text/plain and text/html parts, and converts HTML to plain text automatically. Message bodies are truncated at 30,000 characters.

### todo

Persistent cross-session to-do list. The agent uses this tool to manage tasks that survive across sessions.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `add`, `list`, `update`, `remove` |
| `content` | string | for add | Task description |
| `todo_id` | string | for update/remove | Todo ID or `#index` |
| `status` | string | no | `pending`, `in_progress`, `done`, `cancelled` |
| `responsible` | string | no | `bot` or `human` (default: `bot`) |
| `priority` | string | no | `low`, `normal`, `high`, `urgent` (default: `normal`) |
| `tags` | string | no | Comma-separated tags |
| `filter_status` | string | no | Filter list by status |
| `filter_responsible` | string | no | Filter list by responsible party |

Items include session affinity tracking. The agent is nudged about pending items via context injection at the start of each turn.

### contacts

Persistent cross-session address book. The agent uses this tool to track people across sessions.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `add`, `list`, `search`, `info`, `update`, `remove` |
| `name` | string | for add | Contact name |
| `contact_id` | string | for info/update/remove | Contact ID, `#index`, or name |
| `description` | string | no | Short description of the person |
| `position` | string | no | Job title |
| `organization` | string | no | Company or organization |
| `relation` | string | no | Relationship: colleague, client, manager, friend, vendor, etc. |
| `email` | string | no | Email address(es), comma-separated |
| `phone` | string | no | Phone number |
| `importance` | integer | no | Importance score 1-10 (setting this pins the value) |
| `tags` | string | no | Comma-separated tags |
| `notes` | string | no | Context notes (appended on update, not replaced) |
| `query` | string | for search | Search query |
| `privacy_tier` | string | no | `normal` or `private` (private contacts are not auto-injected) |

Contact context is injected on demand when a known contact name appears in the user message, unlike todo which injects every turn.

### scripts

Persistent cross-session script/file memory. The agent uses this tool to track scripts and files it creates across sessions.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `add`, `list`, `search`, `info`, `update`, `remove` |
| `name` | string | for add | Script name |
| `script_id` | string | for info/update/remove | Script ID, `#index`, or name |
| `file_path` | string | for add | Relative file path to the script |
| `description` | string | no | Short description of the script |
| `purpose` | string | no | What the script does |
| `language` | string | no | Programming language (python, bash, javascript, etc.) |
| `created_reason` | string | no | Why the script was created |
| `tags` | string | no | Comma-separated tags |
| `query` | string | for search | Search query |

Scripts are auto-captured from `write` tool calls when executable file extensions are detected (.py, .sh, .js, .ts, .rb, .pl, .php, .go, .rs, .java, etc.). Script context is injected on demand when a known script name appears in the user message.

### apis

Persistent cross-session API memory. The agent uses this tool to track external APIs it interacts with across sessions.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `add`, `list`, `search`, `info`, `update`, `remove` |
| `name` | string | for add | API name |
| `api_id` | string | for info/update/remove | API ID, `#index`, or name |
| `base_url` | string | for add | Base URL of the API |
| `endpoints` | string | no | JSON list of endpoint definitions [{method, path, description}] |
| `auth_type` | string | no | Authentication type: `bearer`, `api_key`, `basic`, `none` |
| `credentials` | string | no | Authentication credentials (plaintext) |
| `description` | string | no | Short description of the API |
| `purpose` | string | no | What this API is used for |
| `tags` | string | no | Comma-separated tags |
| `query` | string | for search | Search query |

APIs are auto-captured from `web_fetch` and `web_get` tool calls when API-like URL patterns are detected (URLs containing `/api/` or `/v[0-9]+/`). API context is injected on demand when a known API name or base URL appears in the user message. Credentials are stored as plaintext for easy injection into generated scripts.

### typesense

Index, search, and delete documents in deep memory (Typesense). All operations use the configured deep memory collection — the LLM cannot create or choose collections. The collection is auto-created at startup with a controlled schema.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `index`, `search`, `delete` |
| `text` | string | for index | Text content to index (auto-chunked and embedded) |
| `source` | string | no | Source label for indexed content (default: `manual`) |
| `reference` | string | no | Reference identifier (URL, file path, label) |
| `tags` | string | no | Comma-separated tags |
| `query` | string | for search | Search query text |
| `filter_by` | string | no | Typesense filter expression (for search or delete) |
| `max_results` | number | no | Max search results (default: 10, max: 250) |
| `document_id` | string | for delete | Document ID (deletes doc and all its chunks) |

Requires a running Typesense instance. Set the API key in `config.yaml` or via `TYPESENSE_API_KEY` env var. The tool is also used as the sink for the scale loop `no_file` output strategy when `final_action: api_call`. Indexing is routed through `DeepMemoryIndex` for proper chunking, timestamping, and embedding.

### botport

Consult specialist agents through the BotPort agent-to-agent network. Use this tool to delegate tasks to agents with specific expertise.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `consult`, `follow_up`, `close`, `status`, `list_agents` |
| `task` | string | for consult | The task or question for the specialist agent |
| `expertise` | list[string] | no | Expertise tags to match the right specialist (e.g. `["legal", "contracts"]`) |
| `context` | string | no | Relevant context or background information for the task |
| `concern_id` | string | for follow_up/close/status | Concern ID from a previous consult |
| `message` | string | for follow_up | Follow-up message for an active concern |

**Actions:**
- **`consult`** — Send a new task to the BotPort network. BotPort routes it to the best-matched agent based on expertise tags, persona matching, or LLM-powered routing.
- **`follow_up`** — Continue a conversation on an existing concern (requires `concern_id` from a previous consult).
- **`close`** — Close an active concern and clean up the remote agent.
- **`status`** — Check BotPort connection status or concern status.
- **`list_agents`** — List all connected agents and their capabilities (personas, tools, models).

Requires `botport_client.enabled: true` and a valid BotPort server URL in config. See [BotPort](#botport) for setup details.

### termux

Interact with an Android device via Termux API. Requires the [Termux:API](https://wiki.termux.com/wiki/Termux:API) app and `termux-api` package (`pkg install termux-api`). Supports camera photo capture, battery status, GPS/network location, and flashlight (torch) control.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `photo`, `battery`, `location`, `torch` |
| `camera_id` | integer | for photo | Camera ID: `0` = back (default), `1` = front/selfie |
| `provider` | string | for location | Location provider: `gps` (default, outdoor), `network` (WiFi/cell, indoor), `passive` (cached, fastest) |
| `state` | string | for torch | Torch state: `on` (default) or `off` |

Photos are saved to `saved/media/<session_id>/` with automatic timestamped filenames and are delivered as image attachments in Telegram and the web UI.

> **Enabling on mobile:** If the web UI settings page is not accessible on the mobile browser, add `"termux"` to the `tools.enabled` list in `~/.captain-claw/config.yaml`:
>
> ```yaml
> tools:
>   enabled:
>     - termux
> ```

### personality

Read or update the agent personality and per-user profiles. The personality tool is always enabled and context-aware — in global mode (console/CLI) it operates on the agent's own identity, and in user mode (Telegram, web UI persona selector) it operates on the current user's profile.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `get` (view current personality) or `update` (modify fields) |
| `name` | string | for update | Name (agent name or user name depending on mode) |
| `description` | string | for update | Short description (role, title, etc.) |
| `background` | string | for update | Background, origin story, or experience |
| `expertise` | string | for update | Comma-separated list of expertise areas |

In **agent mode** (default): Reads/updates the global agent personality stored at `~/.captain-claw/personality.md`. The agent's name automatically gets "of the Captain Claw family" appended in the system prompt unless the name already contains "Captain Claw".

In **user mode** (Telegram, web UI persona): Reads/updates the current user's profile stored at `~/.captain-claw/personalities/{user_id}.md`. This tells the agent who it is talking to, enabling tailored responses based on the user's expertise and perspective.

See [Personality System](#personality-system) for details.

### datastore

Manage persistent relational data tables in a local SQLite database. Create tables, insert/update/delete rows, query with filters, run raw SELECT queries, import/export CSV or XLSX files, and manage data protection rules.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list_tables`, `describe`, `create_table`, `drop_table`, `add_column`, `rename_column`, `drop_column`, `change_column_type`, `insert`, `update`, `update_column`, `delete`, `query`, `sql`, `import_file`, `export`, `protect`, `unprotect`, `list_protections` |
| `table` | string | for most actions | Target table name |
| `columns` | string | for create_table/query | For create_table: JSON array of `{"name": "col", "type": "text"}`. For query/export: comma-separated column names. |
| `column` | string | for column actions | Column name (for add/rename/drop/change_column_type/update_column) |
| `new_name` | string | for rename_column | New column name |
| `col_type` | string | for add/change_column_type | Column type: `text`, `integer`, `real`, `boolean`, `date`, `datetime`, `json` |
| `default_value` | string | no | Default value for new column (for add_column) |
| `rows` | string | for insert | JSON array of row objects: `[{"name": "Alice", "age": 30}]` |
| `set_values` | string | for update | JSON object of column=value pairs: `{"status": "done"}` |
| `value` | string | for update_column | Literal value to set across matching rows |
| `expression` | string | for update_column | SQL expression (e.g. `price * 1.1`) |
| `where` | string | for query/update/delete | JSON filter: `{"age": {"op": ">", "value": 25}}` or equality shorthand `{"name": "Alice"}` |
| `order_by` | string | no | Comma-separated columns for ordering (prefix with `-` for DESC) |
| `limit` | integer | no | Max rows to return |
| `offset` | integer | no | Rows to skip |
| `sql_query` | string | for sql | Raw SELECT SQL query (read-only, DML blocked) |
| `file_path` | string | for import_file | Path to CSV/XLSX file |
| `sheet` | string | no | Sheet name for XLSX import (default: first sheet) |
| `append` | boolean | no | Append to existing table on import |
| `format` | string | no | Export format: `csv` (default) or `xlsx` |
| `level` | string | for protect/unprotect | Protection level: `table`, `column`, `row`, `cell` |
| `row_id` | integer | for row/cell protection | Row ID for row or cell-level protection |
| `reason` | string | no | Reason for protection (optional) |

**Column types:** `text`, `integer`, `real`, `boolean` (stored as 0/1), `date` (ISO date string), `datetime` (ISO datetime string), `json` (JSON-encoded string).

**Where clause format:** Supports structured filters with operators (`=`, `!=`, `<`, `>`, `<=`, `>=`, `LIKE`, `NOT LIKE`, `IN`, `NOT IN`, `IS NULL`, `IS NOT NULL`) or simple equality shorthand. Example: `{"age": {"op": ">", "value": 25}, "status": "active"}`.

**Raw SQL (`sql` action):** Read-only SELECT queries. DML keywords (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE) are blocked. Table names are auto-rewritten to the internal `ds_` prefix. A configurable LIMIT is auto-appended.

**Protection system:** Four levels prevent accidental data modification. Protected operations return `success=false` with a blocking message. See [Datastore](#datastore) for details.

Enabled by default. Stored at `~/.captain-claw/datastore.db`. Table list is automatically injected into the LLM context when `datastore.inject_table_list` is true.

---

## Configuration Reference

### Config Load Precedence

1. `./config.yaml` (current working directory) — highest priority
2. `~/.captain-claw/config.yaml` — fallback
3. `.env` file in current directory
4. Environment variables (override all)
5. Defaults

**Environment variable override pattern:** Prefix `CLAW_` with double underscore for nesting:
```bash
CLAW_MODEL__PROVIDER="anthropic"
CLAW_MODEL__TEMPERATURE="0.5"
CLAW_TOOLS__WEB_SEARCH__API_KEY="brave_key"
```

### model

```yaml
model:
  provider: "ollama"              # openai, anthropic, gemini, ollama
  model: "minimax-m2.5:cloud"     # model name
  temperature: 0.7                # sampling temperature
  max_tokens: 32000               # max generation tokens
  api_key: ""                     # API key (or use env vars)
  base_url: ""                    # custom endpoint URL
  allowed:                        # models available for /session model
    - id: "chatgpt-fast"
      provider: "openai"
      model: "gpt-4o-mini"
    - id: "claude-sonnet"
      provider: "anthropic"
      model: "claude-sonnet-4-20250514"
      temperature: 0.5            # per-model override
      max_tokens: 64000           # per-model override
    - id: "dalle"
      provider: "openai"
      model: "gpt-image-1"
      model_type: "image"         # used by image_gen tool
    - id: "gpt-vision"
      provider: "openai"
      model: "gpt-4o"
      model_type: "vision"        # used by image_ocr and image_vision tools
```

**Provider aliases:** `chatgpt` = `openai`, `claude` = `anthropic`, `google` = `gemini`.

**Model types:** `llm` (default), `image` (for `image_gen`), `vision` (for `image_vision` and `image_ocr`), `ocr` (for `image_ocr` only).

### context

```yaml
context:
  max_tokens: 160000              # total context window budget
  compaction_threshold: 0.8       # trigger compaction at 80% usage
  compaction_ratio: 0.4           # keep 40% recent messages after compaction
  chunked_processing:
    enabled: false                # master switch (or use auto_threshold)
    auto_threshold: 0             # auto-enable when max_tokens <= this value (0 = off)
    output_reserve_tokens: 4000   # tokens reserved for LLM output per chunk call
    chunk_overlap_tokens: 200     # overlap between consecutive chunks (continuity)
    max_chunks: 12                # hard cap on number of chunks per item
    combine_strategy: "summarize" # "summarize" (LLM synthesis) or "concatenate"
```

See [Chunked Processing Pipeline](#chunked-processing-pipeline) for a full explanation.

### memory

```yaml
memory:
  enabled: true
  path: "~/.captain-claw/memory.db"
  index_workspace: true           # index workspace files for RAG
  index_sessions: true            # index session messages
  cross_session_retrieval: false   # query across all sessions
  auto_sync_on_search: true       # sync index before queries
  max_workspace_files: 400
  max_file_bytes: 262144          # 256 KB per file
  chunk_chars: 1400               # chunk size for embeddings
  chunk_overlap_chars: 200
  cache_ttl_seconds: 45
  stale_after_seconds: 120
  include_extensions:             # file types to index
    - ".txt"
    - ".md"
    - ".py"
    - ".js"
    - ".ts"
    - ".json"
    - ".yaml"
    - ".sql"
    - ".csv"
    - ".sh"
  exclude_dirs:                   # directories to skip
    - ".git"
    - "node_modules"
    - "__pycache__"
    - ".venv"
    - "venv"
  embeddings:
    provider: "auto"              # auto, litellm, ollama, none
    litellm_model: "text-embedding-3-small"
    litellm_api_key: ""
    litellm_base_url: ""
    ollama_model: "nomic-embed-text"
    ollama_base_url: "http://127.0.0.1:11434"
    request_timeout_seconds: 4
    fallback_to_local_hash: true  # use hash if embedding fails
  search:
    max_results: 6
    candidate_limit: 80
    min_score: 0.1
    vector_weight: 0.65           # vector similarity weight
    text_weight: 0.35             # BM25 text search weight
    temporal_decay_enabled: true
    temporal_half_life_days: 21.0
```

### tools

```yaml
tools:
  enabled:
    - shell
    - read
    - write
    - glob
    - web_fetch
    - web_search
    - pdf_extract
    - docx_extract
    - xlsx_extract
    - pptx_extract
    - image_gen
    - image_ocr
    - image_vision
    - pocket_tts
    - send_mail
    - google_drive
    - google_calendar
    - google_mail
    - todo
    - contacts
    - scripts
    - apis
    - datastore
    - personality
    - botport
    - termux
  require_confirmation:           # tools that require user approval
    - shell
    - write
  plugin_dirs: []                 # additional skill tool directories
  shell:
    timeout: 30                   # default timeout in seconds
    blocked:                      # always-blocked patterns
      - "rm -rf /"
      - "mkfs"
    allowed_commands: []          # explicit allow list (empty = check patterns)
    default_policy: "ask"         # ask, allow, deny
    allow_patterns:               # regex patterns to allow
      - "python *"
      - "pip install *"
    deny_patterns:                # regex patterns to deny
      - "rm -rf *"
      - "sudo *"
  web_fetch:
    max_chars: 100000
  web_search:
    provider: "brave"
    api_key: ""                   # or BRAVE_API_KEY env var
    base_url: "https://api.search.brave.com/res/v1/web/search"
    max_results: 5
    timeout: 20
    safesearch: "moderate"        # off, moderate, strict
  image_gen:
    timeout_seconds: 120
    default_size: "1024x1024"     # 1024x1024, 1536x1024, 1024x1536
    default_quality: ""           # auto, high, medium, low, hd, standard
  image_ocr:
    timeout_seconds: 120
    max_chars: 120000
    default_prompt: ""            # empty = "Extract all text from this image."
    max_pixels: 1568              # longest edge cap before sending to LLM (0 = no resize; uses Pillow or ImageMagick)
    jpeg_quality: 85              # JPEG quality for resized images (1-100)
  image_vision:
    timeout_seconds: 120
    max_chars: 120000
    default_prompt: ""            # empty = "Describe this image in detail."
    max_pixels: 1568              # longest edge cap before sending to LLM (0 = no resize; uses Pillow or ImageMagick)
    jpeg_quality: 85              # JPEG quality for resized images (1-100)
  pocket_tts:
    max_chars: 4000
    default_voice: ""
    sample_rate: 24000
    mp3_bitrate_kbps: 128
    timeout_seconds: 600
  send_mail:
    provider: "smtp"              # smtp, mailgun, sendgrid
    from_address: ""
    from_name: ""
    smtp_host: "localhost"
    smtp_port: 587
    smtp_username: ""
    smtp_password: ""
    smtp_use_tls: true
    mailgun_api_key: ""
    mailgun_domain: ""
    mailgun_base_url: "https://api.mailgun.net/v3"
    sendgrid_api_key: ""
    sendgrid_base_url: "https://api.sendgrid.com/v3/mail/send"
    timeout: 60
    max_attachment_bytes: 26214400  # 25 MB
```

### skills

```yaml
skills:
  managed_dir: "~/.captain-claw/skills"
  allow_bundled: []               # bundled skills to enable
  entries: {}                     # per-skill overrides (enabled, apiKey, env, config)
  load:
    extra_dirs: []
    plugin_dirs: []
    watch: true
    watch_debounce_ms: 250
  install:
    prefer_brew: true
    node_manager: "npm"           # npm, pnpm, yarn, bun
  max_skills_in_prompt: 64
  max_skills_prompt_chars: 16000
  max_skill_file_bytes: 131072    # 128 KB
  search_source_url: "https://raw.githubusercontent.com/VoltAgent/awesome-openclaw-skills/main/catalog.json"
  search_limit: 10
  search_max_candidates: 5000
```

### guards

```yaml
guards:
  input:
    enabled: false
    level: "stop_suspicious"      # stop_suspicious or ask_for_approval
  output:
    enabled: false
    level: "stop_suspicious"
  script_tool:
    enabled: false
    level: "stop_suspicious"
```

### todo

```yaml
todo:
  enabled: true                   # enable cross-session to-do memory
  auto_capture: true              # auto-detect tasks from conversation
  inject_on_session_load: true    # nudge agent with pending items
  max_items_in_prompt: 10         # max items injected into context
  archive_after_days: 30          # archive completed items after N days
```

### addressbook

```yaml
addressbook:
  enabled: true                   # enable cross-session address book
  auto_capture: true              # auto-detect contacts from conversation
  inject_on_mention: true         # inject contact context when name appears in message
  max_items_in_prompt: 5          # max contacts injected into context per turn
```

### scripts_memory

```yaml
scripts_memory:
  enabled: true                   # enable cross-session script memory
  auto_capture: true              # auto-detect scripts from write tool calls
  inject_on_mention: true         # inject script context when name appears in message
  max_items_in_prompt: 5          # max scripts injected into context per turn
```

### apis_memory

```yaml
apis_memory:
  enabled: true                   # enable cross-session API memory
  auto_capture: true              # auto-detect APIs from web_fetch tool calls
  inject_on_mention: true         # inject API context when name/URL appears in message
  max_items_in_prompt: 5          # max APIs injected into context per turn
```

### session

```yaml
session:
  storage: "sqlite"
  path: "~/.captain-claw/sessions.db"
  auto_save: true
```

### workspace

```yaml
workspace:
  path: "./workspace"             # tool outputs go under ./workspace/saved
```

### ui

```yaml
ui:
  theme: "dark"
  show_tokens: true
  streaming: true
  colors: true
  monitor_trace_llm: false        # log intermediate LLM calls
  monitor_trace_pipeline: true    # log pipeline events
  monitor_full_output: false      # show raw vs compact tool output
```

### execution_queue

```yaml
execution_queue:
  mode: "collect"                 # steer, followup, collect, interrupt, queue
  debounce_ms: 1000
  cap: 20                         # max queued items
  drop: "summarize"               # old, new, summarize
```

### logging

```yaml
logging:
  level: "INFO"
  format: "console"
```

### telegram

```yaml
telegram:
  enabled: false
  bot_token: ""                   # or TELEGRAM_BOT_TOKEN env var
  api_base_url: "https://api.telegram.org"
  poll_timeout_seconds: 25
  pairing_ttl_minutes: 30
```

### slack

```yaml
slack:
  enabled: false
  bot_token: ""                   # or SLACK_BOT_TOKEN env var
  app_token: ""                   # or SLACK_APP_TOKEN env var
  api_base_url: "https://slack.com/api"
  poll_timeout_seconds: 25
  pairing_ttl_minutes: 30
```

### discord

```yaml
discord:
  enabled: false
  bot_token: ""                   # or DISCORD_BOT_TOKEN env var
  application_id: ""              # or DISCORD_APPLICATION_ID env var
  api_base_url: "https://discord.com/api/v10"
  poll_timeout_seconds: 25
  pairing_ttl_minutes: 30
  require_mention_in_guild: true  # require @mention in server channels
```

### web

```yaml
web:
  enabled: false
  host: "127.0.0.1"
  port: 23080
  api_enabled: true               # OpenAI-compatible API proxy
  api_pool_max_agents: 50
  api_pool_idle_seconds: 600.0
```

### google_oauth

```yaml
google_oauth:
  enabled: false                  # auto-enables when client_id + client_secret set
  client_id: ""                   # or GOOGLE_OAUTH_CLIENT_ID env var
  client_secret: ""               # or GOOGLE_OAUTH_CLIENT_SECRET env var
  project_id: ""                  # GCP project (for Vertex AI only)
  location: "us-central1"         # Vertex AI region
  scopes:
    - "https://www.googleapis.com/auth/cloud-platform"
    - "https://www.googleapis.com/auth/drive"
    - "https://www.googleapis.com/auth/calendar"
    - "https://www.googleapis.com/auth/gmail.readonly"
    - "openid"
    - "email"
```

### orchestrator

```yaml
orchestrator:
  max_parallel: 5                 # concurrent tasks
  max_agents: 50                  # total agent pool
  idle_evict_seconds: 300.0       # evict idle agents after 5 min
  worker_timeout_seconds: 300.0   # per-task timeout
  worker_max_retries: 2           # retries before failure
```

### botport_client

```yaml
botport_client:
  enabled: false                    # enable BotPort connection
  url: ""                           # WebSocket URL (e.g. wss://botport.kstevica.com/ws)
  instance_name: "default"          # name to register with on the hub
  key: ""                           # authentication key
  secret: ""                        # authentication secret
  advertise_personas: true          # advertise available personas to the hub
  advertise_tools: true             # advertise available tools
  advertise_models: true            # advertise available models
  max_concurrent: 5                 # max concurrent concerns to handle
  reconnect_delay_seconds: 5.0      # reconnection backoff
  heartbeat_interval_seconds: 30.0  # heartbeat interval
```

### scale

Controls when the **scale loop** activates for list-processing tasks. The scale loop is an optimization system that takes over repetitive per-item processing (fetch → process → write) to prevent the LLM context from growing linearly with item count.

There are two activation tiers:

| Tier | Default threshold | What it enables |
|---|---|---|
| **Full scale advisory** | `>= 7` members | Scale progress tracking, scale guards (block re-glob, re-read of output), context trimming, micro-loop takeover. The micro-loop processes remaining items with isolated per-item LLM calls at constant context size. |
| **Lightweight progress** | `>= 3` members | Progress indicators only (e.g. "3 of 10 (30%)"). No guards, no micro-loop, no context trimming. |

The full scale advisory also activates when input patterns suggest large-scale work (e.g. "process all files in folder") regardless of the member count threshold.

**Output strategy:** The scale system automatically detects whether results should go to a single combined file (`append=true`), separate per-item files, or no file at all (e.g. email, API). This is extracted from the user's prompt by the list-task planner. Examples:

- `"Name the output file report-[date].csv"` → **file_per_item** — each item gets its own file
- `"Write all results to summary.md"` → **single_file** — everything appends to one file
- `"Send the results to email"` → **no_file** — no file output, results delivered via email/API

**When to tune:** Lower `scale_advisory_min_members` if you frequently process small lists (3–6 items) and want the micro-loop optimization. Raise it if the overhead of scale tracking is unnecessary for your typical workloads. The lightweight threshold should generally stay low since it only adds progress indicators with negligible overhead.

```yaml
scale:
  scale_advisory_min_members: 7   # activate full scale loop (guards + micro-loop)
  lightweight_progress_min_members: 3  # activate progress indicators only
```

**Chunked processing integration:** When `context.chunked_processing` is active, the micro-loop automatically detects items whose content exceeds the model's available context window and routes them through the [chunked processing pipeline](#chunked-processing-pipeline). This is transparent — items that fit in one call are processed normally; only oversized items trigger chunking.

### datastore

```yaml
datastore:
  enabled: true                     # enable the relational datastore
  path: "~/.captain-claw/datastore.db"  # SQLite database path
  inject_table_list: true           # inject table names into LLM context
  max_rows_per_table: 100000        # max rows allowed per table
  max_tables: 50                    # max number of tables
  max_query_rows: 500               # max rows returned per query
  max_export_rows: 50000            # max rows in CSV/XLSX export
```

### deep_memory

Typesense-backed long-term archive for persistent searchable content. Deep memory is an **additional layer** on top of the SQLite-backed semantic memory — it is NOT a replacement. Content is indexed via the Typesense tool or the scale loop `no_file` sink, and searched only when the user explicitly requests it.

**Trigger phrases** that activate deep memory search:
- "search deep memory", "find in archive", "search indexed"
- "long-term memory", "search typesense", "search deep"

```yaml
deep_memory:
  enabled: false                          # enable the deep memory layer
  host: localhost                         # Typesense host
  port: 8108                              # Typesense port
  protocol: http                          # http or https
  api_key: ""                             # Typesense API key (or set TYPESENSE_API_KEY)
  collection_name: captain_claw_deep_memory  # collection for deep memory docs
  embedding_dims: 1536                    # embedding dimensions (match your model)
  auto_embed: true                        # compute embeddings on index
```

**How content flows in:**
- Scale loop `no_file` + `api_call` sink — processed items are indexed automatically
- Typesense tool — the LLM can index documents directly via the `index` action
- Programmatic API — `DeepMemoryIndex.index_document()` / `index_batch()`

**How content flows out:**
- Context injection — when triggered, deep memory results appear in the LLM prompt alongside semantic memory
- Direct search — the LLM can use the Typesense tool `search` action

**Collection schema** (auto-created on first use):
- `doc_id`, `source`, `reference`, `path` — metadata fields (faceted)
- `text` — chunk text (searchable)
- `chunk_index`, `start_line`, `end_line` — position within source
- `tags` — optional string array (faceted)
- `updated_at` — unix timestamp (default sort)
- `embedding` — optional float array for vector search

---

## Guard System

Captain Claw includes three guard types that check interactions at different stages:

| Guard | When it runs | What it checks |
|---|---|---|
| `input` | Before LLM request | User prompts and messages |
| `output` | After LLM response | Model-generated content |
| `script_tool` | Before execution | Shell commands, tool payloads, generated scripts |

### Enforcement Levels

| Level | Behavior |
|---|---|
| `stop_suspicious` | Block the interaction immediately and log |
| `ask_for_approval` | Show the flagged content and ask for explicit user approval |

### Configuration

```yaml
guards:
  input:
    enabled: true
    level: "ask_for_approval"
  output:
    enabled: true
    level: "stop_suspicious"
  script_tool:
    enabled: true
    level: "ask_for_approval"
```

Guards are disabled by default. Enable them for stricter safety behavior. Guards remain active during cron executions, skill invocations, and orchestrator worker tasks.

In the Web UI, tool approval requests appear as modal dialogs with the flagged content and Approve/Deny buttons.

---

## Skills System

Captain Claw loads OpenClaw-style `SKILL.md` skills from multiple discovery roots.

### Discovery Roots (precedence order)

| Priority | Source | Location |
|---|---|---|
| 1 | Extra dirs | Configured in `skills.load.extra_dirs` |
| 2 | Plugin dirs | Configured in `skills.load.plugin_dirs` (with manifest) |
| 3 | Bundled | `./skills` beside `./instructions` |
| 4 | Managed | `~/.captain-claw/skills/` |
| 5 | Personal agents | `~/.agents/skills/` |
| 6 | Project agents | `.agents/skills/` (relative to workspace) |
| 7 | Workspace | `<workspace>/skills/` |

### SKILL.md Format

Skills are defined as Markdown files with YAML frontmatter:

```markdown
---
name: "My Skill"
description: "What this skill does (max 200 chars)"
user-invocable: true
disable-model-invocation: false
command-dispatch: tool          # "tool" or "script"
command-tool: "shell"           # tool to dispatch to
metadata:
  openclaw:
    emoji: "🔧"
    homepage: "https://..."
    requires:
      bins: ["jq"]              # required binaries
      env: ["MY_API_KEY"]       # required env vars
    install:
      - kind: "brew"
        formula: "jq"
---

# My Skill

Skill instructions and documentation go here...
```

### Frontmatter Fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Skill display name |
| `description` | string | Short description (max 200 chars) |
| `user-invocable` | bool | Can be invoked via `/skill` command (default: true) |
| `disable-model-invocation` | bool | Prevent LLM from invoking (default: false) |
| `command-dispatch` | string | Dispatch mode: `tool` or `script` |
| `command-tool` | string | Tool name for `tool` dispatch |
| `metadata.openclaw.requires.bins` | list | Required binaries |
| `metadata.openclaw.requires.env` | list | Required environment variables |
| `metadata.openclaw.requires.config` | list | Required config paths |
| `metadata.openclaw.install` | list | Dependency installation specs |

### Invocation

1. **User command:** `/skill <name> [args]` or `/<command-alias> [args]`
2. **Model invocation:** The LLM can invoke skills if `disable-model-invocation: false`. Eligible skills are injected into the system prompt.
3. **Command dispatch:** Routes to a tool or script based on frontmatter.

### Installing Skills

```text
/skill install https://github.com/user/skill-repo
/skill search "web scraping"
```

Skills are installed to `~/.captain-claw/skills/` by default.

---

## Datastore

Captain Claw includes a built-in relational datastore backed by a dedicated SQLite database. The agent can create tables, insert and query data, import spreadsheets, run read-only SQL, and protect data from accidental modification — all through the `datastore` tool or the web dashboard.

The datastore is completely separate from the session and memory databases.

### Data Types

| Type | SQLite affinity | Notes |
|---|---|---|
| `text` | TEXT | Default type |
| `integer` | INTEGER | Whole numbers |
| `real` | REAL | Floating-point numbers |
| `boolean` | INTEGER | Stored as 0/1 |
| `date` | TEXT | ISO date string (e.g. `2026-02-28`) |
| `datetime` | TEXT | ISO datetime string (e.g. `2026-02-28T14:30:00Z`) |
| `json` | TEXT | JSON-encoded string |

### Table Management

Tables are created, modified, and dropped via the `datastore` tool:

- **`create_table`** — define table name and column definitions as JSON
- **`describe`** — inspect a table's schema, row count, and timestamps
- **`drop_table`** — remove a table and all its data (blocked if protected)
- **`add_column`** / **`rename_column`** / **`drop_column`** / **`change_column_type`** — schema mutations

All table names are sanitized to `[a-z0-9_]` and internally prefixed with `ds_` to avoid clashing with metadata tables.

### CRUD Operations

- **`insert`** — insert one or more rows as JSON objects
- **`update`** — update rows matching a WHERE filter with new values
- **`update_column`** — set a column to a literal value or SQL expression across matching rows
- **`delete`** — delete rows matching a WHERE filter
- **`query`** — select rows with optional column selection, WHERE filters, ORDER BY, LIMIT, and OFFSET

**WHERE clause format:**

```json
{
  "age": {"op": ">", "value": 25},
  "status": "active",
  "name": {"op": "LIKE", "value": "%smith%"}
}
```

Simple equality uses `{"column": "value"}` shorthand. Structured filters support operators: `=`, `!=`, `<`, `>`, `<=`, `>=`, `LIKE`, `NOT LIKE`, `IN`, `NOT IN`, `IS NULL`, `IS NOT NULL`.

### Raw SQL Queries

The `sql` action executes read-only SELECT queries:

- DML keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`) are blocked
- Table names in the query are auto-rewritten to the internal `ds_` prefix
- A configurable LIMIT is auto-appended (default: `max_query_rows: 500`)

### Import and Export

**Import** (`import_file` action):
- Supports CSV and XLSX files
- Auto-detects column names from headers
- Creates a new table or appends to an existing one (`append: true`)
- XLSX import supports sheet selection

**Export** (`export` action):
- Formats: `csv` (default) or `xlsx`
- Respects `max_export_rows` limit (default: 50,000)
- Files saved to the session output directory

The web dashboard also supports drag-and-drop CSV/XLSX upload with automatic table matching — if an uploaded file's headers closely match an existing table, rows are appended automatically.

### Protection System

The datastore includes a four-level protection system that prevents accidental modification of important data:

| Level | Scope | What it protects |
|---|---|---|
| `table` | Entire table | Blocks drop, insert, update, delete, schema changes |
| `column` | Single column | Blocks drop, rename, type change, updates to that column |
| `row` | Single row (by `_id`) | Blocks update and delete of that row |
| `cell` | Single cell (row + column) | Blocks update of that specific cell |

**Commands:**
- **`protect`** — add a protection rule with an optional reason
- **`unprotect`** — remove a protection rule
- **`list_protections`** — show all active protections for a table

When a protected operation is attempted, the tool returns `success=false` with a `BLOCKED` message instructing the agent to inform the user. Protections must be explicitly removed before the data can be modified.

### Context Injection

When `datastore.inject_table_list` is enabled (default: `true`), the agent receives a compact summary of all datastore tables at the start of each turn. This allows the agent to know what tables exist and suggest relevant queries without the user having to list them.

### Configuration

```yaml
datastore:
  enabled: true                     # enable the relational datastore
  path: "~/.captain-claw/datastore.db"  # SQLite database path
  inject_table_list: true           # inject table names into LLM context
  max_rows_per_table: 100000        # max rows allowed per table
  max_tables: 50                    # max number of tables
  max_query_rows: 500               # max rows returned per query
  max_export_rows: 50000            # max rows in CSV/XLSX export
```

### Datastore Dashboard

The web UI includes a Datastore dashboard at `/datastore` (also linked from the homepage). It provides a visual interface for browsing tables, editing data, running SQL queries, and uploading files.

**Layout:**

| Panel | Content |
|---|---|
| **Sidebar** (left) | Table list with row counts, create table button |
| **Main area** (right) | Table view with paginated rows, column headers with types, inline editing |

**Features:**
- **Table browser** — click any table to view its schema and rows
- **Row pagination** — configurable page size with offset navigation
- **Inline editing** — add, update, and delete rows through the UI
- **Schema management** — add and drop columns, create and drop tables
- **SQL console** — run raw SELECT queries with tabular results
- **File upload** — drag-and-drop CSV/XLSX import with automatic table matching
- **Table export** — export tables as CSV, XLSX, or JSON directly from the dashboard
- **Protection management** — view, add, and remove protection rules per table

**REST API endpoints** (used by the dashboard):

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/datastore/tables` | List all tables with schema and row counts |
| `POST` | `/api/datastore/tables` | Create a new table |
| `GET` | `/api/datastore/tables/{name}` | Describe a single table |
| `DELETE` | `/api/datastore/tables/{name}` | Drop a table |
| `GET` | `/api/datastore/tables/{name}/rows` | Query rows with pagination, filtering, and ordering |
| `POST` | `/api/datastore/tables/{name}/rows` | Insert rows |
| `PATCH` | `/api/datastore/tables/{name}/rows` | Update rows matching a WHERE filter |
| `DELETE` | `/api/datastore/tables/{name}/rows` | Delete rows matching a WHERE filter |
| `POST` | `/api/datastore/tables/{name}/columns` | Add a column |
| `DELETE` | `/api/datastore/tables/{name}/columns/{col}` | Drop a column |
| `POST` | `/api/datastore/sql` | Execute a raw SELECT query |
| `GET` | `/api/datastore/tables/{name}/protections` | List protections for a table |
| `POST` | `/api/datastore/tables/{name}/protections` | Add a protection rule |
| `DELETE` | `/api/datastore/tables/{name}/protections` | Remove a protection rule |
| `GET` | `/api/datastore/tables/{name}/export` | Export table as CSV, XLSX, or JSON (`?format=csv\|xlsx\|json`) |
| `POST` | `/api/datastore/upload` | Upload and import a CSV/XLSX file |

---

## Deep Memory (Typesense)

Captain Claw has two memory layers:

1. **Semantic memory** (SQLite) — workspace files + session history, always active, searched automatically
2. **Deep memory** (Typesense) — long-term archive, opt-in, searched only on demand

Deep memory is designed for content that should persist beyond workspace files and session history — processed research, indexed articles, accumulated knowledge bases. It uses Typesense's built-in hybrid search (BM25 + vector) for retrieval.

### Setup

1. Run Typesense locally:
   ```bash
   docker run -d -p 8108:8108 \
     -v /tmp/typesense-data:/data \
     typesense/typesense:27.1 \
     --data-dir /data --api-key=your-api-key
   ```

2. Configure in `config.yaml`:
   ```yaml
   deep_memory:
     enabled: true
     api_key: "your-api-key"
     collection_name: captain_claw_deep_memory

   tools:
     typesense:
       api_key: "your-api-key"
   ```

   Or via environment variable: `TYPESENSE_API_KEY=your-api-key`

3. The collection is auto-created on first use.

### Usage

**Indexing content:**
- Ask the agent: "index this article on Typesense" or "save this to deep memory"
- Use the scale loop with `no_file` output strategy: "fetch these 10 URLs and index the results on Typesense"

**Searching:**
- Ask the agent: "search deep memory for Series A rounds in July"
- Or: "find in archive anything about quarterly revenue"
- Deep memory results appear in the LLM's context when triggered

### Integration with Scale Loop

When the list-task planner detects `output_strategy: no_file` and `final_action: api_call`, the micro-loop routes processed items directly to Typesense instead of writing files:

```
User prompt → list-task planner → scale loop → micro-loop → Typesense sink
```

Each processed item is indexed as a document with `source: "scale_loop"`, `reference: <item_label>`, and the processed text as `text`.

### Deep Memory Dashboard

The web UI includes a dedicated Deep Memory dashboard at `/deep-memory` (also linked from the homepage). It provides a visual interface for browsing, searching, and managing all content in the Typesense-backed archive.

**Three-panel layout:**

| Panel | Content |
|---|---|
| **Sidebar** (left) | Collection stats (name, document count), source filter list, tag filter chips |
| **Document list** (middle) | Search bar with debounced input, scrollable document list with source badges, chunk counts, tags, and relative timestamps. Scroll pagination loads more results automatically. |
| **Detail view** (right) | Document metadata (ID, source, path, tags, last updated), all chunks displayed in monospace with chunk index and line range badges. Delete button with confirmation modal. |

**Features:**
- **Connection status** — header indicator shows Typesense connectivity (green dot = connected, red = disconnected)
- **Full-text search** — debounced 300ms search across all indexed content
- **Source filtering** — click a source in the sidebar (manual, web_fetch, pdf, scale, etc.) to filter the document list
- **Tag filtering** — click tag chips to filter by tag; click again to deselect
- **Document detail** — click any document to view all its chunks with full text content
- **Delete** — remove a document and all its chunks with a confirmation modal
- **Manual indexing** — click "+" to open the index form: paste text, set source, reference, and tags, then submit to index directly into Typesense

**REST API endpoints** (used by the dashboard):

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/deep-memory/status` | Typesense connectivity and collection stats |
| `GET` | `/api/deep-memory/documents` | List documents grouped by `doc_id` with search, source/tag filters, pagination |
| `GET` | `/api/deep-memory/documents/{doc_id}` | Get all chunks for a specific document |
| `DELETE` | `/api/deep-memory/documents/{doc_id}` | Delete a document and all its chunks |
| `GET` | `/api/deep-memory/facets` | Source and tag facet values for filter dropdowns |
| `POST` | `/api/deep-memory/index` | Index a new document manually (auto-chunked) |

---

## Memory and RAG

Captain Claw includes a hybrid retrieval system that combines vector similarity and text search.

### Architecture

- **Vector search:** Cosine similarity on embeddings (weight: 0.65)
- **Text search:** BM25 full-text search via SQLite FTS (weight: 0.35)
- **Temporal decay:** Newer results ranked higher (half-life: 21 days)

### Embedding Providers

| Provider | Description |
|---|---|
| `auto` | Try LiteLLM, fall back to Ollama, fall back to hash |
| `litellm` | External API (e.g. `text-embedding-3-small` via OpenAI) |
| `ollama` | Local model (e.g. `nomic-embed-text`) |
| `none` | Hash-based fallback (no semantic meaning) |

### What Gets Indexed

- **Workspace files:** Text files matching `include_extensions`, up to 400 files, 256 KB each
- **Session messages:** Conversation history from active sessions
- **Cross-session:** Optional retrieval across all sessions (`cross_session_retrieval: true`)

### Configuration

Key settings in `config.memory`:
- `chunk_chars: 1400` — characters per chunk
- `chunk_overlap_chars: 200` — overlap between chunks
- `search.max_results: 6` — results returned per query
- `search.min_score: 0.1` — minimum relevance threshold

---

## Cross-Session Todo Memory

Captain Claw includes a persistent to-do system that works across sessions. Items survive restarts and can be managed by both the user and the agent.

### How It Works

- **Explicit capture:** Use `/todo add <text>` or tell the agent "save this to to-do" or "remind me to..."
- **Auto-capture:** Conservative pattern matching detects task-like phrases in conversation (e.g., "don't forget to...", "to-do: ..."). Disabled with `todo.auto_capture: false`.
- **Context injection:** At each turn, the agent receives a compact note of pending and in-progress items, nudging it to act on outstanding tasks.
- **Session affinity:** Each item tracks which session created it. Items with no session target are visible globally.

### Responsible Parties

Items can be assigned to `bot` (the agent should handle it) or `human` (the user should handle it). Default is `bot`.

### Priorities

Four levels: `urgent`, `high`, `normal` (default), `low`. Items are sorted by priority in listings and context injection.

### Availability

The `/todo` command and `todo` tool are available across all interfaces: CLI, Web UI, Telegram, Slack, and Discord. The Web UI also exposes REST endpoints (`GET/POST /api/todos`, `PATCH/DELETE /api/todos/{id}`).

---

## Cross-Session Address Book

Captain Claw includes a persistent address book that tracks people across sessions. Contacts survive restarts and accumulate context over time.

### How It Works

- **Explicit capture:** Use `/contacts add <name>` or tell the agent "remember that John is the CTO" or "save contact: Jane Smith".
- **Auto-capture:** Conservative pattern matching detects contact-like phrases in conversation. Email recipients from `send_mail` are automatically added as contacts. Disabled with `addressbook.auto_capture: false`.
- **On-demand context injection:** When a known contact name appears in the user message (or "who is..." patterns), the agent receives relevant contact details. Unlike todo, contacts are NOT injected every turn.
- **Privacy tiers:** Contacts marked as `private` are excluded from auto-injection but remain accessible via explicit tool/CLI queries.

### Importance Tracking

Contacts have an importance score (1-10). It can be:
- **Auto-computed:** Based on mention frequency, recency (21-day half-life), and session diversity. Uses `min(10, 1 + log2(mentions) * recency * diversity)`.
- **Manually pinned:** Set via `/contacts importance <name> <score>` or the tool. Pinned values are not overwritten by auto-computation.

### Availability

The `/contacts` command and `contacts` tool are available across all interfaces: CLI, Web UI, Telegram, Slack, and Discord. The Web UI also exposes REST endpoints (`GET/POST /api/contacts`, `GET /api/contacts/search?q=`, `GET/PATCH/DELETE /api/contacts/{id}`).

---

## Cross-Session Script Memory

Captain Claw includes a persistent script/file memory that tracks scripts and files the agent creates across sessions. Script metadata survives restarts and accumulates context over time.

### How It Works

- **Explicit capture:** Use `/scripts add <name> <path>` or tell the agent "remember script..." or "save script: ...".
- **Auto-capture from write tool:** When the `write` tool creates a file with an executable extension (.py, .sh, .js, .ts, .rb, .pl, .php, .go, .rs, .java, etc.), it is automatically registered. Disabled with `scripts_memory.auto_capture: false`.
- **Auto-capture from conversation:** Conservative pattern matching detects script-related phrases (e.g., "remember script...", "save script:...").
- **On-demand context injection:** When a known script name appears in the user message, the agent receives relevant script metadata. Like contacts, scripts are NOT injected every turn.
- **No file content in DB:** Only path and metadata are stored. File content can be regenerated if needed.
- **Deduplication:** Scripts are deduplicated by file path — a new write to the same path updates the existing entry.

### Tracked Fields

| Field | Description |
|---|---|
| `name` | Script display name |
| `file_path` | Relative path from workspace |
| `language` | Programming language (auto-detected from extension) |
| `description` | Short description |
| `purpose` | What the script does |
| `created_reason` | Why it was created |
| `tags` | Comma-separated tags |
| `use_count` | How many times the script has been referenced |

### Availability

The `/scripts` command and `scripts` tool are available across all interfaces: CLI, Web UI, Telegram, Slack, and Discord. The Web UI also exposes REST endpoints (`GET/POST /api/scripts`, `GET /api/scripts/search?q=`, `GET/PATCH/DELETE /api/scripts/{id}`).

---

## Cross-Session API Memory

Captain Claw includes a persistent API memory that tracks external APIs the agent interacts with across sessions. API metadata and credentials survive restarts and accumulate context over time.

### How It Works

- **Explicit capture:** Use `/apis add <name> <base_url>` or tell the agent "remember api..." or "save api: ...".
- **Auto-capture from web_fetch/web_get:** When `web_fetch` or `web_get` accesses a URL containing `/api/` or `/v[0-9]+/` patterns, the API is automatically registered. Disabled with `apis_memory.auto_capture: false`.
- **Auto-capture from conversation:** Conservative pattern matching detects API-related phrases.
- **On-demand context injection:** When a known API name or base URL appears in the user message, the agent receives relevant API details including credentials and endpoints. Like contacts, APIs are NOT injected every turn.
- **Plaintext credentials:** API credentials are stored as plaintext for easy injection into generated scripts. This is a deliberate design choice for usability.
- **Deduplication:** APIs are deduplicated by base URL — a new registration with the same base URL updates the existing entry.

### Tracked Fields

| Field | Description |
|---|---|
| `name` | API display name |
| `base_url` | Base URL of the API |
| `endpoints` | JSON list of endpoint definitions [{method, path, description}] |
| `auth_type` | Authentication type: bearer, api_key, basic, none |
| `credentials` | Authentication credentials (plaintext) |
| `description` | Short description |
| `purpose` | What this API is used for |
| `tags` | Comma-separated tags |
| `use_count` | How many times the API has been referenced |

### Availability

The `/apis` command and `apis` tool are available across all interfaces: CLI, Web UI, Telegram, Slack, and Discord. The Web UI also exposes REST endpoints (`GET/POST /api/apis`, `GET /api/apis/search?q=`, `GET/PATCH/DELETE /api/apis/{id}`).

---

## Personality System

Captain Claw includes a dual-profile personality system that separates **agent identity** (who is responding) from **user context** (who is being talked to).

### Two Profile Types

| Profile | Scope | Storage | Purpose |
|---|---|---|---|
| **Agent personality** | Global | `~/.captain-claw/personality.md` | The agent's name, description, background, and expertise. Shared across all sessions and users. |
| **User profiles** | Per-user | `~/.captain-claw/personalities/{user_id}.md` | Individual profiles describing each user's name, expertise, background, and perspective. Enables tailored responses. |

### Profile Fields

Each profile (agent or user) contains four fields:

| Field | Description |
|---|---|
| `name` | Display name |
| `description` | Short description or role |
| `background` | Background, origin story, or experience |
| `expertise` | List of expertise areas |

### How It Works

1. **Agent identity** is always injected into the system prompt. The agent's name automatically gets "of the Captain Claw family" appended unless the name already contains "Captain Claw".
2. **User context** is injected when a user profile is active (Telegram user with a profile, or web UI persona selector). This tells the LLM who it is talking to and instructs it to tailor responses to that user's expertise level and perspective.
3. Both profiles are stored as Markdown files with `# Name`, `# Description`, `# Background`, and `# Expertise` section headings.

### Managing Profiles

**Via the personality tool** — the agent can read and update profiles during conversation:
- In CLI/console: the tool operates on the agent personality
- In Telegram: the tool operates on the current Telegram user's profile
- In web UI: the tool operates on the active persona (selected via the persona dropdown)

**Via the web UI:**
- The **Settings** page includes an agent personality editor and a user personalities section for managing per-user profiles
- The **persona selector** dropdown in the chat header lets you switch between user profiles on the fly

**Via REST API:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/personality` | Retrieve agent personality |
| `PUT` | `/api/personality` | Update agent personality |
| `GET` | `/api/user-personalities` | List all user personalities |
| `GET` | `/api/user-personalities/{user_id}` | Get specific user personality |
| `PUT` | `/api/user-personalities/{user_id}` | Create or update a user personality |
| `DELETE` | `/api/user-personalities/{user_id}` | Remove a user personality |
| `GET` | `/api/telegram-users` | List approved Telegram users (for persona picker) |

### Telegram Integration

Each approved Telegram user can have a personality profile. When a Telegram user sends a message, their profile (if configured) is automatically loaded and the personality tool switches to user mode. Configure user profiles from the web UI Settings page — approved Telegram users appear in the user personalities section.

### Caching

Profiles are cached in memory and automatically refreshed when the underlying file's modification time changes. User IDs are sanitized to prevent path traversal.

---

## Session Management

### Lifecycle

1. **Create:** `/new [name]` creates a session with a unique ID and optional name
2. **Switch:** `/session switch <id|name|#index>` switches context
3. **Rename:** `/session rename <new-name>` changes the display name
4. **Describe:** `/session description <text>` or `auto` for LLM-generated description
5. **Protect:** `/session protect on` prevents accidental `/clear`
6. **Export:** `/session export [chat|monitor|pipeline|pipeline-summary|all]`
7. **Procreate:** `/session procreate <parent1> <parent2> <name>` merges two sessions

### Session Procreation

Merges two parent sessions into a new child:
1. Compacts each parent's memory snapshot
2. Interleaves and merges the compacted content
3. Creates the child session with merged context
4. Parents are **not modified** during procreation

### Run-in-Another-Session

```text
/session run #2 summarize your current findings
```

Executes a prompt in session #2, waits for completion, then returns to the current session with the result.

### Per-Session Model Selection

Each session can use a different model. Selection persists across restarts.

```text
/session model claude-sonnet      # this session uses Claude
/new research
/session model chatgpt-fast       # this session uses GPT
```

### Execution Queue

Per-session queue controls how follow-up messages are handled while the agent is processing. See [Execution Queue](#execution-queue-1).

---

## Chunked Processing Pipeline

Enables small-context models (20k–32k tokens) to process content that would otherwise exceed their context window. When active, a **context budget guard** checks every item before processing and automatically routes oversized content through a sequential map-reduce pipeline.

### How It Works

1. **Context budget guard** — Before each LLM call the pipeline computes: `available = context_budget - instruction_tokens - output_reserve`. If `content_tokens > available`, chunking is triggered.
2. **Semantic-aware splitting** — Content is split at paragraph boundaries (double newlines), falling back to single newlines, then hard character splits. Consecutive chunks overlap by `chunk_overlap_tokens` to preserve continuity.
3. **Sequential map phase** — Each chunk is sent as an isolated LLM call with the full instruction set and a chunk header (e.g. "Processing chunk 2 of 5").
4. **Combine phase** — Partial results are combined using the configured strategy:
   - `summarize` (default) — An LLM synthesis call merges all partial results into a single coherent output. Falls back to concatenation if the combined partials would themselves overflow the context.
   - `concatenate` — Simple join with chunk separators. No additional LLM call.

### When It Activates

The pipeline activates when **both** conditions are met:

- **Feature is on** — Either `enabled: true` explicitly, or `auto_threshold` is set and `context.max_tokens <= auto_threshold`.
- **Content exceeds budget** — The context budget guard detects that instruction tokens + content tokens + output reserve > `max_tokens`.

Items that fit in a single call are processed normally with zero overhead.

### Configuration

```yaml
context:
  max_tokens: 32000               # small model context window
  chunked_processing:
    enabled: true                 # or use auto_threshold instead
    auto_threshold: 32000         # auto-enable when max_tokens <= 32k (0 = off)
    output_reserve_tokens: 4000   # tokens reserved for LLM output per chunk
    chunk_overlap_tokens: 200     # overlap between consecutive chunks
    max_chunks: 12                # hard cap on chunks per item
    combine_strategy: "summarize" # "summarize" or "concatenate"
```

| Setting | Default | Description |
|---|---|---|
| `enabled` | `false` | Master switch. Enables chunked processing regardless of model size. |
| `auto_threshold` | `0` | Auto-enable when `context.max_tokens` is at or below this value. Set to `32000` to activate for 32k models. `0` disables auto-detection. |
| `output_reserve_tokens` | `4000` | Tokens reserved for the LLM response in each chunk call. Larger values leave less room for content per chunk. |
| `chunk_overlap_tokens` | `200` | Token overlap between consecutive chunks to maintain continuity across boundaries. |
| `max_chunks` | `12` | Maximum number of chunks per item. If content requires more, the last chunk absorbs the remainder. |
| `combine_strategy` | `"summarize"` | How partial results are merged. `"summarize"` uses an LLM call; `"concatenate"` joins with separators. |

### Scale Loop Integration

The chunked processing pipeline integrates transparently with the [scale loop](#scale) micro-loop. When the micro-loop processes an item:

1. It builds the per-item prompt (instructions + extracted content).
2. The context budget guard checks whether it fits in one call.
3. If yes — normal single-call processing (zero overhead).
4. If no — the content is routed through the chunked pipeline and the combined result is returned to the micro-loop as if it came from a single call.

This means you can use small-context models (e.g. Ollama with 20k–32k context) on large-scale list-processing tasks without any changes to your prompts or workflow.

### Logging

Every step of the pipeline is logged verbosely for debugging:

- **Activation**: Logs when chunking is triggered, including token counts (instruction, content, available, budget).
- **Splitting**: Logs chunk count, sizes, overlap, and the splitting strategy used (paragraph, newline, or hard-split).
- **Map phase**: Logs each chunk call with its index, token size, and result length.
- **Combine phase**: Logs the combine strategy, input sizes, and final output length.
- **Stats**: A diagnostics summary is emitted after each chunked item with chunk count, total input/output tokens, and combine method.

All chunked processing events appear in the monitor/pipeline trace with source `chunked_processing`.

---

## Context Compaction

Long conversations are automatically compacted to stay within the context window.

### Auto-Compaction

Triggers when context usage reaches `compaction_threshold` (default: 80%) of `max_tokens`.

**Process:**
1. Calculate token count for all messages
2. Select older messages for summarization
3. Generate a continuity summary via LLM
4. Replace old messages with the summary
5. Keep recent messages (default: 40% of context via `compaction_ratio`)

### Manual Compaction

```text
/compact
```

Forces immediate compaction regardless of current usage.

### Configuration

```yaml
context:
  max_tokens: 160000
  compaction_threshold: 0.8       # trigger at 80%
  compaction_ratio: 0.4           # keep 40% recent
```

---

## Execution Queue

Controls how follow-up user messages are handled while the agent is busy processing a request.

### Queue Modes

| Mode | Behavior |
|---|---|
| `steer` | New message interrupts current processing and steers the agent |
| `followup` | Messages queue and process sequentially after current completes |
| `collect` | Messages are collected and summarized before processing |
| `interrupt` | Current task is interrupted by urgent new requests |
| `queue` | Pure FIFO queue — messages wait their turn |

### Configuration

```yaml
execution_queue:
  mode: "collect"
  debounce_ms: 1000               # wait before processing
  cap: 20                         # max queued items
  drop: "summarize"               # overflow: old, new, summarize
```

### Commands

```text
/session queue info               # show current settings
/session queue mode followup      # change mode
/session queue debounce 500       # set debounce
/session queue cap 10             # set max queue size
/session queue drop old           # set overflow policy
/session queue clear              # clear pending items
```

---

## Orchestrator / DAG Mode

The orchestrator decomposes complex requests into a task DAG and executes tasks in parallel across separate sessions.

### Flow

1. **Decompose** — The agent's LLM breaks the request into a JSON task plan with dependencies
2. **Build graph** — Creates a `TaskGraph` DAG with topological ordering
3. **Assign sessions** — Creates or reuses sessions for each task
4. **Execute** — Runs tasks in parallel (up to `max_parallel`) with dependency gating
5. **Synthesize** — Aggregates results into a final response

### Usage

```text
/orchestrate Research competitor pricing, analyze our margins, and draft a strategy memo
```

### Task Statuses

| Status | Description |
|---|---|
| `pending` | Waiting for dependencies |
| `running` | Currently executing |
| `timeout_warning` | In 60-second grace period before restart |
| `completed` | Finished successfully |
| `failed` | Finished with error (or exhausted retries) |
| `paused` | Manually paused |
| `editing` | Instructions being edited |

### Timeout Warning Flow

1. Task exceeds `worker_timeout_seconds` (default: 300s)
2. Status changes to `timeout_warning` with a 60-second countdown
3. User can click **Postpone** to grant another full timeout period
4. If not postponed, the task restarts (up to `worker_max_retries`)
5. After all retries exhausted, task fails

### Dashboard (Web UI)

The orchestrator dashboard at `/orchestrator` shows:
- Visual task DAG with color-coded status nodes
- Summary bar with task counts and token usage
- Detail panel for each task (edit, restart, pause, resume, postpone)
- Chronological event log

### File Registry

Tasks can access files created by other tasks via the file registry. Files are registered with logical paths and resolved across sessions automatically.

### Headless CLI

The orchestrator can run without the web server or TUI, useful for cron jobs and scripting:

```bash
# Ad-hoc orchestration
captain-claw-orchestrate "Fetch competitor pricing, analyze margins, draft memo"

# Run a saved workflow
captain-claw-orchestrate --workflow my-workflow

# List saved workflows
captain-claw-orchestrate --list

# JSON output for scripting
captain-claw-orchestrate --json "Research and compare the top 5 CI providers"
```

| Option | Description |
|---|---|
| `--workflow`, `-w` | Name of a saved workflow to execute |
| `--config`, `-c` | Path to config YAML |
| `--model`, `-m` | Override model name |
| `--provider`, `-p` | Override LLM provider |
| `--max-parallel` | Max parallel workers (0 = use config default) |
| `--quiet`, `-q` | Suppress stderr status output |
| `--json` | Output result as JSON |
| `--list` | List saved workflows and exit |

Status updates go to stderr; the synthesis result goes to stdout.

### Configuration

```yaml
orchestrator:
  max_parallel: 5
  max_agents: 50
  idle_evict_seconds: 300.0
  worker_timeout_seconds: 300.0
  worker_max_retries: 2
```

---

## BotPort

BotPort is an agent-to-agent task routing hub that connects multiple Captain Claw instances via persistent WebSocket connections. Agents can delegate tasks to specialist instances based on expertise, and receive structured results — enabling multi-agent collaboration across machines, networks, or cloud deployments.

### Architecture

```
┌──────────┐     WebSocket     ┌──────────┐     WebSocket     ┌──────────┐
│  CC-A    │◄─────────────────►│  BotPort │◄─────────────────►│  CC-B    │
│ (sender) │   concern/result  │  (hub)   │  dispatch/result  │(handler) │
└──────────┘                   └──────────┘                   └──────────┘
```

- **CC-A (originator):** Sends a concern (task + context + expertise tags) to BotPort.
- **BotPort (hub):** Routes the concern to the best-matched instance using tag matching, LLM-powered routing, or least-loaded fallback.
- **CC-B (handler):** Receives the dispatch, spawns an ephemeral agent, processes the task, and returns the result.

### Setup

**1. Install and run BotPort:**

```bash
pip install botport
botport
```

BotPort starts on port `23180` by default. Configure via `~/.botport/config.yaml` or `./config.yaml`.

**2. Connect a Captain Claw instance:**

Add to your Captain Claw `config.yaml`:

```yaml
botport_client:
  enabled: true
  url: "wss://botport.kstevica.com/ws"
  instance_name: "my-agent"
  key: "my-key"                     # if auth is enabled on the hub
  secret: "my-secret"
```

Or for a local BotPort server:

```yaml
botport_client:
  enabled: true
  url: "ws://localhost:23180/ws"
  instance_name: "my-agent"
```

**3. Connect additional instances** with different `instance_name` values. Each instance advertises its personas, tools, and models to the hub.

### Concern Lifecycle

A "concern" is a task routed through BotPort:

1. **CC-A** sends a concern with task description, context, and expertise tags
2. **BotPort** acknowledges and routes to the best-matched CC-B instance
3. **CC-B** spawns an ephemeral agent, processes the task, returns the result
4. **BotPort** relays the result back to CC-A
5. Either side can send **follow-ups** for multi-turn conversations
6. The concern is **closed** when complete or after idle timeout

**Concern states:** `pending` → `assigned` → `in_progress` → `responded` → `closed`

### Routing Strategy

BotPort uses a three-tier routing chain:

| Tier | Strategy | How it works |
|---|---|---|
| 1 | **Tag matching** | Matches concern expertise tags against instance persona expertise (≥50% match threshold) |
| 2 | **LLM routing** | If enabled, an LLM picks the best agent + persona combo from available instances |
| 3 | **Least-loaded** | Falls back to the instance with the lowest active concern count |

### Using the BotPort Tool

Once connected, the agent can delegate tasks through the `botport` tool:

```text
> Ask a legal specialist to review this contract clause for compliance issues.
```

The agent uses the `botport` tool with `action: consult`, `expertise: ["legal", "contracts"]`, and routes the task to a connected instance with matching expertise.

Follow-ups maintain conversation context:

```text
> Follow up on that legal review — what about the liability section?
```

### BotPort Server Configuration

BotPort server configuration (`~/.botport/config.yaml`):

```yaml
server:
  host: "0.0.0.0"
  port: 23180
  dashboard_enabled: true

routing:
  strategy: "auto"                    # tag_match -> llm -> least-loaded
  fallback: "reject"                  # reject or queue_until_available

concerns:
  idle_timeout_seconds: 300           # timeout concerns after 5 min idle
  max_follow_ups: 10                  # max follow-up exchanges per concern
  max_concurrent_per_instance: 5

auth:
  enabled: false
  keys:
    - key: "instance-key"
      secret: "instance-secret"
      instance: "claw-alpha"

logging:
  level: "INFO"
  concern_history: true               # persist concern exchanges to SQLite
  retention_days: 30

llm:
  enabled: false                      # enable LLM-powered routing
  model: "ollama/llama3.2"            # litellm provider/model format
  temperature: 0.1
  timeout: 30
```

### BotPort Dashboard

BotPort includes a web dashboard (enabled by default) that shows:
- Connected instances with their capabilities
- Active and historical concerns
- Routing decisions and concern lifecycle events

Access at `https://botport.kstevica.com` (or `http://localhost:23180` for local).

### Capability Advertisement

Each connected Captain Claw instance advertises:
- **Personas** — agent personality and user profiles with expertise tags
- **Tools** — list of enabled tools
- **Models** — available LLM models
- **Capacity** — max concurrent concerns

Control what is advertised via `botport_client.advertise_personas`, `advertise_tools`, and `advertise_models`.

### Environment Variables

| Variable | Description |
|---|---|
| `CLAW_BOTPORT_CLIENT__ENABLED` | Enable BotPort connection |
| `CLAW_BOTPORT_CLIENT__URL` | BotPort WebSocket URL |
| `CLAW_BOTPORT_CLIENT__INSTANCE_NAME` | Instance name |
| `CLAW_BOTPORT_CLIENT__KEY` | Auth key |
| `CLAW_BOTPORT_CLIENT__SECRET` | Auth secret |

---

## Web UI

Captain Claw includes a browser-based interface with the same capabilities as the terminal.

### Starting

```bash
captain-claw                    # web UI is the default
captain-claw-web                # standalone web server
```

The web UI launches by default at `http://127.0.0.1:23080`. To disable it, set `web.enabled: false` in config (or pass `--tui`).

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  🦀 Captain Claw  │ Session: default │ gpt-4o │ ● Ready │
├────────────────────────────┬────────────────────────────┤
│                            │  📊 Monitor                │
│  💬 Chat                   │                            │
│                            │  Tool outputs, traces,     │
│  [message history]         │  and pipeline events       │
│                            │  appear here in real-time  │
├────────────────────────────┴────────────────────────────┤
│  > Type a message or /command...      [Send] [⌘K] [?]  │
└─────────────────────────────────────────────────────────┘
```

### Sidebar Tabs

| Tab | What you can do |
|---|---|
| **Sessions** | View, switch, and create sessions |
| **Instructions** | Browse, edit, and save `.md` instruction files in-place |
| **Help** | Full command reference and keyboard shortcuts |

### Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Enter` | Send message |
| `Shift+Enter` | Insert newline |
| `Ctrl+K` | Open command palette (fuzzy search) |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+N` | New session |
| `Ctrl+S` | Save instruction file (when editor open) |
| `Escape` | Close palette / sidebar |

### Features

- **Command suggestions:** Type `/` to see an inline dropdown of all available commands
- **Command palette:** `Ctrl+K` opens a fuzzy-search palette for all commands
- **Instruction editor:** Edit instruction `.md` files live; changes take effect immediately
- **Persona selector:** Dropdown in the chat header to switch between user profiles for tailored responses
- **Tool approvals:** Modal dialog appears when a guard requires approval
- **Session replay:** Full history replayed on connect/reconnect
- **Resize handle:** Drag the divider between chat and monitor panes

### Dashboard Pages

The homepage at `/` provides card-based navigation to all dashboard pages:

| Page | Path | Description |
|---|---|---|
| Chat | `/chat` | Interactive conversation with the agent |
| Sessions | `/sessions` | Browse and manage all conversation sessions |
| Orchestrator | `/orchestrator` | Parallel DAG task execution with real-time monitoring |
| Instructions | `/instructions` | Browse and edit instruction templates |
| Cron | `/cron` | Schedule and monitor recurring tasks |
| Workflows | `/workflows` | Browse saved workflows and execution outputs |
| Loop Runner | `/loop-runner` | Execute workflows with variable iteration |
| Memory | `/memory` | Browse to-dos, contacts, scripts, and API registrations |
| Deep Memory | `/deep-memory` | Browse and search Typesense-backed long-term archive |
| Datastore | `/datastore` | Browse and manage structured data tables |
| Files | `/files` | Browse agent-created files and download outputs |
| Settings | `/settings` | Configure models, tools, agent personality, user profiles, and system options |

### Configuration

```yaml
web:
  enabled: false
  host: "127.0.0.1"
  port: 23080
  api_enabled: true               # OpenAI-compatible API proxy
  api_pool_max_agents: 50
  api_pool_idle_seconds: 600.0
```

---

## Remote Integrations

Captain Claw can run alongside Telegram, Slack, and Discord bots.

### Setup

1. Create a bot on the platform (Telegram BotFather, Slack App, Discord Developer Portal)
2. Add the bot token to config or environment:
   ```yaml
   telegram:
     enabled: true
     bot_token: "your-bot-token"
   ```
3. Start Captain Claw — the bot connects automatically

### Pairing Flow

1. An unknown user sends a message to the bot
2. The bot generates a pairing token (8-char alphanumeric, valid for 30 minutes)
3. The operator approves in the local CLI:
   ```text
   /approve user telegram abc123xy
   ```
4. After approval, the remote user can send prompts and slash commands

### Telegram: Per-User Sessions

Each Telegram user automatically gets a dedicated session and agent instance:

- **Isolated sessions** — users cannot see or switch to other users' sessions
- **Concurrent execution** — different Telegram users can interact simultaneously (no "Agent is busy" blocking between users)
- **Per-user locks** — requests from the same user are serialized; requests from different users run in parallel
- **Disabled commands** — `/new` and session switching (`/session list`, `/session switch`, `/session load`, `/session new`) are not available on Telegram
- **Available session commands** — `/clear`, `/history`, `/compact`, `/session info`, `/session rename` operate on the user's own session
- **Photo attachments** — images sent to the bot are processed through the `image_vision` tool
- **Generated images** — images created by `image_gen` are automatically sent back to the user

### Supported Remote Commands

Remote users can use: `/help`, `/config`, `/history`, `/compact`, `/models`, `/sessions`, `/session info`, `/session select`, `/session rename`, `/skills`, `/skill`, `/skill search`, `/cron`, `/todo`, `/contacts`, `/scripts`, `/apis`, `/pipeline`, `/planning`, `/orchestrate`.

Local-only commands (not available remotely): `/exit`, `/approve user`, `/session run`, `/session procreate`, `/session protect`, `/session export`, `/session queue`, `/monitor`, `/cron add/list/history/pause/resume/remove`.

**Telegram-only restrictions:** `/new` and `/sessions` are disabled. Session switching subcommands (`list`, `switch`, `load`, `new`) are disabled.

---

## OpenAI-Compatible API Proxy

When the web server is running with `api_enabled: true`, Captain Claw exposes an OpenAI-compatible API.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completion (streaming supported) |
| `GET` | `/v1/models` | List available models |

### Usage

```bash
curl http://127.0.0.1:23080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### Configuration

```yaml
web:
  api_enabled: true
  api_pool_max_agents: 50         # max concurrent agents
  api_pool_idle_seconds: 600.0    # evict idle agents after 10 min
```

Requests are proxied through the Captain Claw agent pool. Each API request gets its own agent context with full tool access.

---

## Google OAuth, Drive, Calendar, and Gmail

### Setup

1. Create OAuth 2.0 credentials in the [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Set the **Authorized redirect URI** to: `http://localhost:23080/auth/google/callback` (default port: 23080)
3. Add credentials to config or `.env`:

```bash
# .env file
GOOGLE_OAUTH_CLIENT_ID=your-client-id
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
```

Or in `config.yaml`:
```yaml
google_oauth:
  client_id: "your-client-id"
  client_secret: "your-client-secret"
```

OAuth auto-enables when both `client_id` and `client_secret` are set.

### Connecting

- **Web UI:** Click "Connect Google" on the home page
- **CLI:** Navigate to `/auth/google/login` in your browser

The OAuth flow uses PKCE for security. Tokens are stored locally and refresh automatically.

### Google Drive Tool

Once connected, the `google_drive` tool is available with these actions:

| Action | Description |
|---|---|
| `list` | Browse files in a folder (default: root) |
| `search` | Find files by name or content |
| `read` | Get file contents (exports Google Docs/Sheets/Slides) |
| `info` | Get file metadata |
| `upload` | Upload a local file to Drive |
| `create` | Create a new file on Drive |
| `update` | Update an existing file's content |

### Google Calendar Tool

Once connected, the `google_calendar` tool is available with these actions:

| Action | Description |
|---|---|
| `list_events` | List upcoming events (defaults to now onwards) |
| `search_events` | Find events by free-text query |
| `get_event` | Get full event details by ID |
| `create_event` | Create a new event with attendees, reminders, recurrence |
| `update_event` | Partial patch of an existing event |
| `delete_event` | Remove an event |
| `list_calendars` | List all accessible calendars |

Supports timed events (ISO 8601 datetime) and all-day events (date-only). See [google_calendar tool reference](#google_calendar) for full parameters.

### Google Mail (Gmail) Tool

Once connected, the `google_mail` tool provides read-only Gmail access:

| Action | Description |
|---|---|
| `list_messages` | List recent emails from a label (default: INBOX) |
| `search` | Search using Gmail query syntax (`from:`, `subject:`, `is:unread`, etc.) |
| `read_message` | Get full message content, headers, and attachment list |
| `get_thread` | Get all messages in a conversation thread |
| `list_labels` | List available Gmail labels/folders |

Only the `gmail.readonly` scope is required — no send, modify, or delete operations. See [google_mail tool reference](#google_mail) for full parameters.

### Vertex AI

If `project_id` is set, Google OAuth also enables Gemini models via Vertex AI. The `location` field specifies the Vertex AI region (default: `us-central1`).

---

## Send Mail

The `send_mail` tool supports three email providers.

### Providers

| Provider | Config key | Required fields |
|---|---|---|
| SMTP | `provider: "smtp"` | `smtp_host`, `smtp_port` |
| Mailgun | `provider: "mailgun"` | `mailgun_api_key`, `mailgun_domain` |
| SendGrid | `provider: "sendgrid"` | `sendgrid_api_key` |

### Configuration

```yaml
tools:
  send_mail:
    provider: "smtp"
    from_address: "agent@example.com"
    from_name: "Captain Claw"
    smtp_host: "smtp.example.com"
    smtp_port: 587
    smtp_username: "user"
    smtp_password: "pass"
    smtp_use_tls: true
```

### Environment Variables

```bash
MAILGUN_API_KEY="key"
MAILGUN_DOMAIN="mail.example.com"
SENDGRID_API_KEY="key"
MAIL_FROM_ADDRESS="agent@example.com"
MAIL_FROM_NAME="Captain Claw"
SMTP_HOST="smtp.example.com"
SMTP_PORT="587"
SMTP_USERNAME="user"
SMTP_PASSWORD="pass"
```

Attachments are limited to 25 MB per file.

---

## File Output Policy

All tool-generated files are written under `<workspace>/saved/` with session-scoped paths.

### Directory Structure

```
workspace/saved/
  scripts/<session-id>/           # generated scripts
  tools/<session-id>/             # reusable helper programs
  downloads/<session-id>/         # fetched external files
  media/<session-id>/             # images, audio, video
  showcase/<session-id>/          # polished outputs, exports
  skills/<session-id>/            # skill assets
  tmp/<session-id>/               # scratch intermediates
```

### Rules

- All paths are session-scoped using stable session IDs (not mutable names)
- Uncategorized paths are remapped to `saved/tmp/<session-id>/...`
- Unsafe absolute or traversal paths are remapped for safety
- Writes outside `saved/` root are blocked

---

## Environment Variables Reference

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google / Gemini API key |
| `OLLAMA_BASE_URL` | Ollama server URL |
| `BRAVE_API_KEY` | Brave Search API key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `SLACK_BOT_TOKEN` | Slack bot token |
| `SLACK_APP_TOKEN` | Slack app token |
| `DISCORD_BOT_TOKEN` | Discord bot token |
| `DISCORD_APPLICATION_ID` | Discord application ID |
| `GOOGLE_OAUTH_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth client secret |
| `GOOGLE_OAUTH_PROJECT_ID` | GCP project ID (Vertex AI) |
| `GOOGLE_OAUTH_ENABLED` | Enable Google OAuth (`true`, `1`, `yes`) |
| `MAILGUN_API_KEY` | Mailgun API key |
| `MAILGUN_DOMAIN` | Mailgun domain |
| `SENDGRID_API_KEY` | SendGrid API key |
| `TYPESENSE_API_KEY` | Typesense API key (used by both tool and deep memory) |
| `CLAW_BOTPORT_CLIENT__URL` | BotPort WebSocket URL (e.g. `wss://botport.kstevica.com/ws`) |
| `CLAW_BOTPORT_CLIENT__KEY` | BotPort authentication key |
| `CLAW_BOTPORT_CLIENT__SECRET` | BotPort authentication secret |
| `MAIL_FROM_ADDRESS` | Email sender address |
| `MAIL_FROM_NAME` | Email sender name |
| `SMTP_HOST` | SMTP server host |
| `SMTP_PORT` | SMTP server port |
| `SMTP_USERNAME` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `CLAW_*` | Override any config field (e.g. `CLAW_MODEL__PROVIDER=anthropic`) |

The `CLAW_` prefix uses double underscore for nested fields: `CLAW_TOOLS__WEB_SEARCH__API_KEY=key`.

`.env` files in the current working directory are loaded automatically. Format: `KEY=VALUE` (one per line, `#` comments, `export` prefix supported).
