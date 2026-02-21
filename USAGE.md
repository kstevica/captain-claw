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
- [Guard System](#guard-system)
- [Skills System](#skills-system)
- [Memory and RAG](#memory-and-rag)
- [Cross-Session Todo Memory](#cross-session-todo-memory)
- [Cross-Session Address Book](#cross-session-address-book)
- [Session Management](#session-management)
- [Context Compaction](#context-compaction)
- [Execution Queue](#execution-queue-1)
- [Orchestrator / DAG Mode](#orchestrator--dag-mode)
- [Web UI](#web-ui)
- [Remote Integrations](#remote-integrations)
- [OpenAI-Compatible API Proxy](#openai-compatible-api-proxy)
- [Google OAuth and Google Drive](#google-oauth-and-google-drive)
- [Send Mail](#send-mail)
- [File Output Policy](#file-output-policy)
- [Environment Variables Reference](#environment-variables-reference)

---

## Installation

### Requirements

- Python **>= 3.11**

### Install from source

```bash
git clone https://github.com/kstevica/captain-claw
cd captain-claw
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Development dependencies

```bash
pip install -e ".[dev]"
```

### Entry points

| Command | Description |
|---|---|
| `captain-claw` | Web UI (default) |
| `captain-claw --tui` | Start with terminal UI |
| `captain-claw --onboarding` | Re-run first-time setup wizard |
| `captain-claw-web` | Web UI only (standalone entry point) |

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

Fetch and extract readable content from a URL.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | yes | URL to fetch |
| `extract_mode` | string | no | `text` (default, parsed readable content) or `html` (raw HTML) |
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
```

**Provider aliases:** `chatgpt` = `openai`, `claude` = `anthropic`, `google` = `gemini`.

### context

```yaml
context:
  max_tokens: 160000              # total context window budget
  compaction_threshold: 0.8       # trigger compaction at 80% usage
  compaction_ratio: 0.4           # keep 40% recent messages after compaction
```

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
    - pocket_tts
    - send_mail
    - google_drive
    - todo
    - contacts
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
- **Tool approvals:** Modal dialog appears when a guard requires approval
- **Session replay:** Full history replayed on connect/reconnect
- **Resize handle:** Drag the divider between chat and monitor panes

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

### Supported Remote Commands

Remote users can use: `/help`, `/config`, `/history`, `/compact`, `/models`, `/sessions`, `/session info`, `/session select`, `/session rename`, `/skills`, `/skill`, `/skill search`, `/cron`, `/todo`, `/contacts`, `/pipeline`, `/planning`, `/orchestrate`.

Local-only commands (not available remotely): `/exit`, `/approve user`, `/session run`, `/session procreate`, `/session protect`, `/session export`, `/session queue`, `/monitor`, `/cron add/list/history/pause/resume/remove`.

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

## Google OAuth and Google Drive

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
| `MAIL_FROM_ADDRESS` | Email sender address |
| `MAIL_FROM_NAME` | Email sender name |
| `SMTP_HOST` | SMTP server host |
| `SMTP_PORT` | SMTP server port |
| `SMTP_USERNAME` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `CLAW_*` | Override any config field (e.g. `CLAW_MODEL__PROVIDER=anthropic`) |

The `CLAW_` prefix uses double underscore for nested fields: `CLAW_TOOLS__WEB_SEARCH__API_KEY=key`.

`.env` files in the current working directory are loaded automatically. Format: `KEY=VALUE` (one per line, `#` comments, `export` prefix supported).
