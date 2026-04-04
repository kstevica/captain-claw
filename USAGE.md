# Captain Claw Usage Reference

Comprehensive reference for all commands, tools, configuration options, and features.

For a quick overview and installation guide, see [README.md](README.md).

---

## Table of Contents

- [Installation](#installation)
- [Docker](#docker)
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
  - [insights](#insights-1)
  - [nervous_system](#nervous_system)
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
- [Cross-Session Playbook Memory](#cross-session-playbook-memory)
  - [Playbooks Editor](#playbooks-editor)
- [Personality System](#personality-system)
- [Self-Reflection System](#self-reflection-system)
- [Insights](#insights)
- [Nervous System (Dreaming)](#nervous-system-dreaming)
- [Brain Graph](#brain-graph)
- [Process of Thoughts](#process-of-thoughts)
- [Session Management](#session-management)
- [Chunked Processing Pipeline](#chunked-processing-pipeline)
- [Context Compaction](#context-compaction)
- [Execution Queue](#execution-queue-1)
- [Orchestrator / DAG Mode](#orchestrator--dag-mode)
- [BotPort (Agent-to-Agent)](#botport)
- [Computer](#computer)
  - [Layout](#layout-1)
  - [Theme System](#theme-system)
  - [Model Selector](#model-selector)
  - [Visual Generation](#visual-generation)
  - [Exploration Tree](#exploration-tree)
  - [Folder Browser](#folder-browser)
  - [Attachments](#attachments)
  - [/btw Command](#btw-command)
  - [Suggested Next Steps](#suggested-next-steps)
  - [Public Mode](#public-mode)
- [Web UI](#web-ui)
- [Remote Integrations](#remote-integrations)
  - [Telegram: Per-User Sessions](#telegram-per-user-sessions)
- [OpenAI-Compatible API Proxy](#openai-compatible-api-proxy)
- [Google OAuth, Drive, Calendar, and Gmail](#google-oauth-drive-calendar-and-gmail)
- [Send Mail](#send-mail)
- [Termux](#termux)
- [Prompt Caching](#prompt-caching)
- [LLM Usage Dashboard](#llm-usage-dashboard)
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
pip install captain-claw[screen]   # Screen capture + voice commands (mss, pynput, sounddevice, soniox)
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
| `botport` | BotPort agent-to-agent routing hub |

If the configured port is busy, Captain Claw automatically tries the next available port (up to 10 attempts).

---

## Docker

Captain Claw ships with Docker support for running the web UI and BotPort as containers. You can either pull pre-built images from Docker Hub or build from source.

### Prerequisites

- Docker (and Docker Compose if building from source)
- A `config.yaml` and `.env` file in your working directory (copy from `config.yaml.example` and `.env.example`, then add your API keys)

### Pull from Docker Hub

```bash
docker pull kstevica/captain-claw:latest
docker pull kstevica/captain-claw-botport:latest
```

Run Captain Claw:

```bash
docker run -d -p 23080:23080 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/docker-data/home-config:/root/.captain-claw \
  -v $(pwd)/docker-data/workspace:/data/workspace \
  -v $(pwd)/docker-data/sessions:/data/sessions \
  -v $(pwd)/docker-data/skills:/data/skills \
  kstevica/captain-claw:latest
```

Run BotPort (optional, for multi-agent routing):

```bash
docker run -d -p 33080:33080 \
  -v $(pwd)/botport/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/.env:/app/.env:ro \
  kstevica/captain-claw-botport:latest
```

- **Web UI:** http://localhost:23080
- **BotPort dashboard:** http://localhost:33080

> **Note:** `config.yaml` and `.env` must exist in the current directory before running. If they don't exist, Docker creates empty directories instead of files, which causes a startup error.

### Build from Source with Docker Compose

If you cloned the repo, use the included `docker-compose.yml`:

```bash
docker compose up -d                # start both services
docker compose up -d captain-claw   # web UI only
docker compose up -d botport        # BotPort only
```

### Architecture

The `docker-compose.yml` defines two independent services:

| Service | Image | Port | Entry point |
|---|---|---|---|
| `captain-claw` | `kstevica/captain-claw` | 23080 | `captain-claw-web` |
| `botport` | `kstevica/captain-claw-botport` | 33080 | `botport` |

The services are independent — either can run alone without the other.

### Volumes and Persistent Data

All persistent data is bind-mounted to `./docker-data/` on the host, so files are directly accessible:

| Host path | Container path | Contents |
|---|---|---|
| `./docker-data/home-config/` | `/root/.captain-claw/` | Settings saved from the web UI, personalities, datastore |
| `./docker-data/workspace/` | `/data/workspace/` | Workspace files (tool outputs, exports, media) |
| `./docker-data/sessions/` | `/data/sessions/` | Session database (`sessions.db`) |
| `./docker-data/skills/` | `/data/skills/` | Installed skills |

Configuration and secrets are mounted read-only from the project root:

| Host path | Container path | Purpose |
|---|---|---|
| `./config.yaml` | `/app/config.yaml` | Main configuration (read-only) |
| `./.env` | `/app/.env` | API keys and secrets (read-only) |
| `./botport/config.yaml` | `/app/config.yaml` (botport) | BotPort configuration (read-only) |

The `docker-data/` directory is created automatically on first run and is excluded from git via `.gitignore`.

### Entrypoint

The `docker-entrypoint.sh` script handles Docker-specific configuration:

- **First run:** Seeds `~/.captain-claw/config.yaml` inside the container with overrides that bind the web server to `0.0.0.0` (required for Docker networking) and point storage paths to the persistent volumes.
- **Subsequent runs:** Preserves settings saved from the web UI. Only ensures `web.host` stays at `0.0.0.0` so the container remains reachable.

### Configuration Precedence in Docker

Captain Claw merges configuration from two YAML files:

1. `./config.yaml` (mounted read-only from the host) — base configuration
2. `~/.captain-claw/config.yaml` (persisted in `./docker-data/home-config/`) — overlay that takes precedence

Settings saved from the web UI Settings page are written to the overlay file, so they persist across container restarts and override the base configuration without modifying the host file.

### Build and Manage

```bash
docker compose up -d --build        # rebuild after code changes
docker compose logs -f captain-claw # follow captain-claw logs
docker compose logs -f botport      # follow botport logs
docker compose down                 # stop all services
docker compose down -v              # stop and remove volumes
```

### Running with Ollama

If you use local Ollama models, the container needs to reach the Ollama server on the host. Set the base URL in `.env`:

```bash
OLLAMA_BASE_URL="http://host.docker.internal:11434"
```

On Linux, add `extra_hosts` to the captain-claw service in `docker-compose.yml`:

```yaml
services:
  captain-claw:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

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
| `/nuke` | Delete all workspace files, memory, sessions, and datastore (two-step confirmation) |
| `/screenshot [prompt]` | Capture screen; if prompt given, analyze with vision model |
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

### Reflection Commands

| Command | Description |
|---|---|
| `/reflection` | Show the latest self-reflection |
| `/reflection generate` | Trigger a new self-reflection |
| `/reflection list` | List recent reflections with timestamps |

### Insight Commands

| Command | Description |
|---|---|
| `/insight` or `/insights` | List recent insights |
| `/insight <query>` | Search insights by keyword |
| `/insight stats` | Show insight statistics |
| `/insight add <content>` | Manually add an insight |
| `/insight delete <id>` | Delete an insight |

### Intuition Commands

| Command | Description |
|---|---|
| `/intuition` or `/intuitions` | List recent intuitions |
| `/intuition <query>` | Search intuitions by keyword |
| `/intuition stats` | Show intuition statistics |
| `/intuition dream` | Manually trigger a dream cycle |
| `/intuition add <content>` | Manually add an intuition |
| `/intuition validate <id>` | Validate an intuition (protects from decay) |
| `/intuition delete <id>` | Delete an intuition |

### Orchestrator Commands

| Command | Description |
|---|---|
| `/orchestrate <request>` | Decompose a complex task into a DAG and run in parallel sessions |

The orchestrator decomposes the request into tasks, builds a dependency graph, assigns sessions, and executes tasks in parallel. Tasks share structured data through a shared workspace, can enforce JSON Schema output validation, and emit trace spans visible in the Flight Deck chat panel. See [Orchestrator / DAG Mode](#orchestrator--dag-mode) for details.

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

Generate speech audio locally and save as MP3. The agent can use this tool to speak responses aloud — when voice commands are given via the hotkey, the agent is automatically instructed to reply with audio.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Text to convert to speech |
| `voice` | string | no | Voice preset (default: from config or `alba`) |
| `sample_rate` | number | no | Sample rate in Hz (default: 24000) |
| `bitrate_kbps` | number | no | MP3 bitrate (default: 128) |

Uses `pocket-tts`. Output saved to `saved/media/<session-id>/`.

**Built-in voices:** `alba` (default), `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`. You can also pass a path to a WAV/MP3 file for voice cloning.

**Server-side playback:** When the agent generates audio, it is automatically played on the server machine's speakers using the system audio player (`afplay` on macOS, `ffplay` or `mpv` on Linux). This works regardless of browser tab focus — no need to have the web UI in the foreground. An `<audio>` player element also appears in the chat for manual replay.

### stt

Speech-to-text transcription. Used internally by the hotkey daemon for voice commands, but also available as a standalone tool.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `audio` | string | yes | Path to audio file (WAV, MP3, etc.) or base64-encoded audio |
| `provider` | string | no | `soniox`, `openai`, or `gemini` (default: auto-detect) |

**Providers (in priority order):**

1. **Soniox** (recommended) — Realtime streaming transcription. Set `SONIOX_API_KEY` env var. When used via the hotkey, audio streams directly from the microphone to Soniox over WebSocket — no audio files are saved.
2. **OpenAI Whisper** — File-upload transcription. Uses the configured OpenAI API key.
3. **Gemini** — File-upload transcription via a Gemini model with audio understanding.

The provider is auto-detected based on available API keys, or set explicitly via `tools.screen_capture.stt_provider` in config.

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

### gws

Google Workspace CLI tool. Wraps the `gws` binary ([github.com/googleworkspace/cli](https://github.com/googleworkspace/cli)) to access Drive, Docs, Gmail, and Calendar. Requires the `gws` CLI to be installed and authenticated separately (`gws auth setup && gws auth login`).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `drive_list`, `drive_search`, `drive_download`, `drive_info`, `drive_create`, `docs_read`, `docs_append`, `mail_list`, `mail_search`, `mail_read`, `calendar_list`, `calendar_search`, `calendar_create`, `calendar_agenda`, `raw` |
| `query` | string | for search actions | Search text (Drive file name/content, Gmail search syntax, calendar event text) |
| `file_id` | string | for file actions | Google Drive file ID |
| `folder_id` | string | no | Drive folder ID for `drive_list` or `drive_create` (default: root) |
| `name` | string | for drive_create | File/document name |
| `content` | string | no | Text content (for `drive_create` initial content or `docs_append` text) |
| `mime_type` | string | no | MIME type for `drive_create` (e.g. `application/vnd.google-apps.document` for Google Doc) |
| `output_path` | string | no | Local path to save downloaded file (for `drive_download`) |
| `message_id` | string | for mail_read | Gmail message ID |
| `max_results` | number | no | Maximum results to return (default varies by action) |
| `label` | string | no | Gmail label for `mail_list` (e.g. `INBOX`, `SENT`). Default: `INBOX` |
| `summary` | string | for calendar_create | Event title |
| `start` | string | for calendar_create | Event start time in ISO 8601 format |
| `end` | string | no | Event end time in ISO 8601 format |
| `attendees` | string | no | Comma-separated attendee email addresses (for `calendar_create`) |
| `calendar_id` | string | no | Calendar ID (default: `primary`) |
| `days` | number | no | Days to look ahead for `calendar_list`/`calendar_agenda` (default: 7) |
| `raw_args` | string | for raw | Raw arguments passed directly to gws CLI |

**Installation:** `npm install -g @googleworkspace/cli`, then `gws auth setup && gws auth login`.

**Drive actions:** `drive_list` lists files in a folder, `drive_search` finds files by name or content, `drive_download` exports/downloads a file locally (Google Docs as markdown, Sheets as XLSX preserving all sheets, Presentations as PPTX preserving all slides), `drive_info` gets file metadata, `drive_create` creates a new file on Drive.

**Docs actions:** `docs_read` reads a Google Doc, Sheet, or Presentation inline (Docs exported as markdown; Sheets exported as XLSX and extracted with all sheets preserved; Presentations exported as PPTX and extracted with all slides preserved), `docs_append` appends text to a Doc.

**Mail actions:** `mail_list` lists recent emails from a label, `mail_search` searches using Gmail syntax (e.g. `from:alice subject:report`), `mail_read` reads a specific email by ID.

**Calendar actions:** `calendar_list` lists upcoming events, `calendar_search` searches events by text, `calendar_create` creates a new event, `calendar_agenda` shows a formatted agenda view.

**Raw passthrough:** The `raw` action passes arguments directly to the `gws` CLI for any command not covered by the built-in actions. Example: `raw_args: "sheets +read --id SPREADSHEET_ID --range A1:D10"`.

**Configuration:** The `gws` binary path can be customized via `tools.gws.binary_path` in config. If empty, the binary is found via PATH.

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

### playbooks

Persistent cross-session orchestration pattern memory. Rate sessions to auto-distill reusable do/don't pseudo-code patterns that are injected into planning context when similar tasks are detected.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `add`, `list`, `search`, `info`, `update`, `remove`, `rate` |
| `name` | string | for add | Playbook display name |
| `playbook_id` | string | for info/update/remove | Playbook ID or `#index` |
| `task_type` | string | for add | Classification: `batch-processing`, `web-research`, `code-generation`, `document-processing`, `data-transformation`, `orchestration`, `interactive`, `file-management`, `other` |
| `rating` | string | for add/rate | `good` or `bad` |
| `do_pattern` | string | no | Pseudo-code of the recommended approach (5-15 lines) |
| `dont_pattern` | string | no | Pseudo-code of what to avoid (5-15 lines) |
| `trigger_description` | string | no | When this playbook should activate |
| `reasoning` | string | no | Why this pattern matters |
| `tags` | string | no | Comma-separated tags |
| `query` | string | for search | Search query |
| `session_id` | string | no | Session ID for `rate` action (defaults to current) |
| `note` | string | no | Optional note for `rate` action |

**Actions:**
- **`add`** — register a new playbook manually (requires `name`, `task_type`, and at least one of `do_pattern`/`dont_pattern`)
- **`list`** — show all playbooks (up to 50), optionally filtered by `task_type`
- **`search`** — find playbooks by keyword
- **`info`** — show full details of one playbook
- **`update`** — modify an existing playbook
- **`remove`** — delete a playbook
- **`rate`** — rate a session as good/bad and auto-distill a playbook via LLM analysis of the session's messages and tool trace

See [Cross-Session Playbook Memory](#cross-session-playbook-memory) for details on how playbooks are distilled, stored, and injected.

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

### flight_deck

Discover and communicate with peer agents in the Flight Deck environment. Always available when running under Flight Deck — no configuration needed.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list_agents`, `consult`, or `delegate` |
| `agent_name` | string | for consult/delegate | Name of the peer agent to message |
| `message` | string | for consult/delegate | Message or task to send to the peer agent |

**Actions:**
- **`list_agents`** — queries `GET /fd/fleet` for a live list of all running agents (Docker, process, local) with name, kind, status, port, and description. Marks the calling agent in the output.
- **`consult`** — synchronous peer consultation. Looks up the target agent from the live fleet, sends a message via `/fd/consult-peer`, and streams the response with heartbeat monitoring. Use for quick questions where you need the answer immediately. Includes deduplication to prevent hammering a peer with identical requests.
- **`delegate`** — fire-and-forget task delegation. Sends a task to a peer agent via `/fd/delegate-peer` and returns immediately, freeing the calling agent. The peer works independently and delivers results back as a notification when finished. Use for large/long-running tasks like research, scraping, analysis, or file creation.

Unlike `consult_peer` (which uses the static peer list pushed at connect time), `flight_deck` always queries the live fleet — so newly spawned agents are immediately discoverable.

### screen_capture

Capture a screenshot of the user's screen and optionally analyze it with a vision model. Requires `pip install captain-claw[screen]`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `monitor` | number | no | Monitor index: 0 = all monitors combined (default), 1 = primary, 2 = secondary |
| `prompt` | string | no | If provided, automatically analyze the screenshot with the vision model using this prompt |

When a `prompt` is given, the tool chains into `image_vision` automatically — one call captures and analyzes. Without a prompt, it returns the saved file path for use with `image_vision` or `image_ocr`.

Screenshots are saved to `saved/media/<session_id>/screenshot-<timestamp>.png`.

**Slash command:** `/screenshot [prompt]` — capture from the web UI without the agent needing to call the tool.

#### Global hotkey

Double-tap Shift (configurable) to activate the voice command flow. The full sequence:

1. **Double-tap Shift** — activates the hotkey and starts recording audio from the microphone.
2. **Hold and speak** — give your voice instruction while holding the key. Audio is transcribed in realtime via Soniox (preferred), or recorded and transcribed via Whisper/Gemini as a fallback.
3. **Release the key** — recording stops. The daemon checks for selected text in the active app (via a clipboard swap: save clipboard → Cmd+C → read → restore). If text is selected, it becomes the context. If not, a screenshot is captured instead.
4. **Submit to agent** — the agent receives either the selected text or the screenshot, plus the voice transcription as instructions.
5. **Voice response** — when the input came via voice, the agent is instructed to reply using `pocket_tts` so you hear the answer spoken aloud through your speakers.

**Selected text detection (macOS):** When you have text selected in any app (browser, editor, terminal, etc.), the hotkey daemon automatically detects it via a clipboard round-trip. If selected text is found, it replaces the screenshot as context — faster and more precise for text-heavy workflows like code review, reading articles, or analyzing logs.

**Configuration:**

```yaml
tools:
  screen_capture:
    hotkey_enabled: false          # opt-in; toggle in Settings → Voice & Hotkey (hot-reloads without restart)
    hotkey_trigger_key: shift      # key to double-tap (shift, ctrl, alt, caps_lock)
    hotkey_double_tap_ms: 400      # max ms between taps
    hotkey_triple_tap_wait_ms: 500 # max ms to wait for a third tap
    default_monitor: 0             # 0=all, 1=primary
    max_recording_seconds: 30      # max voice recording duration
    audio_sample_rate: 16000       # mic sample rate (Hz)
    save_audio: false              # persist WAV files
    stt_provider: ""               # "soniox", "openai", "gemini", or "" for auto-detect
```

> **Hotkey is opt-in.** The global hotkey listener is disabled by default. Enable it in the web UI at **Settings → Voice & Hotkey**, or set `hotkey_enabled: true` in config. Changes take effect immediately — no restart needed.

**STT provider setup:**

| Provider | Setup | How it works |
|---|---|---|
| Soniox (recommended) | Set `SONIOX_API_KEY` env var | Realtime streaming — audio goes from mic to Soniox WebSocket, no files saved |
| OpenAI Whisper | Set `OPENAI_API_KEY` env var | Records WAV, uploads to Whisper API after key release |
| Gemini | Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Records WAV, transcribes via Gemini audio model |

Auto-detection priority: Soniox → OpenAI → Gemini. Override with `stt_provider` in config.

**macOS permissions required:** Screen Recording, Input Monitoring (for hotkey), Microphone (for voice), Accessibility (for Cmd+C simulation in selected-text detection). **Linux:** X11 required for hotkey (Wayland not supported; `/screenshot` still works). **Windows:** works out of the box (selected text detection is macOS-only for now).

### desktop_action

Cross-platform desktop GUI automation — click, type, scroll, press keys, open apps/folders/URLs. Pairs with `screen_capture` to identify coordinates before acting. Requires `pip install pyautogui`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `click`, `double_click`, `right_click`, `move`, `type`, `press`, `hotkey`, `scroll`, `drag`, `open`, `mouse_position`, `screenshot_click` |
| `x` | number | for click/move/scroll/drag | X coordinate (pixels from left edge) |
| `y` | number | for click/move/scroll/drag | Y coordinate (pixels from top edge) |
| `text` | string | for type/press/screenshot_click | Text to type, key name to press, or UI element description for screenshot_click |
| `keys` | array | for hotkey | Keys to press simultaneously (e.g. `["command", "c"]`) |
| `target` | string | for open | App name, folder path, or URL to open |
| `dx` | number | for drag/scroll | Destination X (drag) or horizontal scroll amount |
| `dy` | number | for drag/scroll | Destination Y (drag) or vertical scroll amount |
| `clicks` | number | no | Number of clicks (default: 1) |
| `interval` | number | no | Seconds between actions (default: 0.05) |
| `duration` | number | no | Seconds for mouse movement animation (default: 0.3) |

**Typical workflow:** capture a screenshot with `screen_capture`, identify the element and its coordinates, then use `desktop_action` to interact with it. The `screenshot_click` action combines these steps — it captures a screenshot, uses vision to locate the described element, and clicks it automatically.

**Platform support:** macOS uses `open -a` for apps; Linux uses `xdg-open`; Windows uses `os.startfile`. Mouse and keyboard actions work on all platforms via `pyautogui`. The `pyautogui.FAILSAFE` is enabled — move the mouse to the top-left corner of the screen to abort any runaway automation.

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
    - stt
    - send_mail
    - gws
    - todo
    - contacts
    - scripts
    - apis
    - datastore
    - playbooks
    - personality
    - botport
    - termux
    - screen_capture
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
    default_voice: ""              # voice preset; empty = "alba". Options: alba, marius, javert, jean, fantine, cosette, eponine, azelma (or path to audio file for voice cloning)
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
  gws:
    binary_path: ""                 # custom path to gws binary (empty = find in PATH)
  screen_capture:
    hotkey_enabled: false           # enable global hotkey listener (opt-in; toggle in Settings → Voice & Hotkey)
    hotkey_trigger_key: shift       # key to double-tap (shift, ctrl, alt, caps_lock)
    hotkey_double_tap_ms: 400       # max ms between taps to count as double-tap
    hotkey_triple_tap_wait_ms: 500  # max ms to wait for a third tap
    default_monitor: 0              # 0=all monitors, 1=primary, 2=secondary
    timeout_seconds: 30
    max_recording_seconds: 30.0     # max voice recording duration
    audio_sample_rate: 16000        # Hz for mic recording
    save_audio: false               # persist WAV files to workspace
    stt_provider: ""                # "soniox", "openai", "gemini", or "" for auto-detect
    stt_model: ""                   # explicit STT model ID; empty = auto-detect
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
  auth_token: ""                  # set to enable authentication; empty = auth disabled
  auth_cookie_max_age: 90         # days until auth cookie expires
  public_run: ""                  # set to "computer" to expose only Computer to anonymous visitors
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

**Google Drive integration:** The scale loop automatically detects Google Drive files from previous `drive_list` or `drive_search` results in the session. Google-native files (Docs, Sheets, Presentations) are read inline via `docs_read`. Uploaded files (PDF, DOCX, etc.) are downloaded via `drive_download` and extracted locally. No manual file-ID handling is needed — the scale loop matches item names to Drive file IDs automatically.

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

### insights

Persistent knowledge base auto-extracted from conversations. The agent identifies facts, contacts, decisions, deadlines, and other durable knowledge during conversations.

```yaml
insights:
  enabled: true                           # enable insights extraction
  auto_extract: true                      # auto-extract after agent turns
  inject_in_context: true                 # inject relevant insights into system prompt
  max_items_in_prompt: 8                  # max insights shown in context
  extraction_interval_messages: 8         # messages between extractions
  extraction_cooldown_seconds: 60         # minimum seconds between extractions
  max_insights: 500                       # hard cap on stored insights
  db_path: "~/.captain-claw/insights.db"  # SQLite database path
```

### nervous_system

Autonomous dreaming layer that cross-references all memory types to discover patterns, connections, and hypotheses. Disabled by default — opt in via Settings or config.

```yaml
nervous_system:
  enabled: false                            # enable nervous system (opt-in)
  auto_dream: true                          # auto-trigger dream cycles after turns
  inject_in_context: true                   # inject intuitions into system prompt
  max_items_in_prompt: 4                    # max intuitions shown in context
  dream_interval_messages: 12               # messages between dream cycles
  dream_cooldown_seconds: 300               # minimum seconds between dreams (5 min)
  max_intuitions: 200                       # hard cap on stored intuitions
  min_confidence_for_context: 0.3           # minimum confidence to surface in context
  decay_after_days: 7                       # start decaying unvalidated intuitions after N days
  decay_rate_per_day: 0.05                  # confidence reduction per day of inactivity
  delete_threshold: 0.1                     # delete intuitions below this confidence
  allow_public: false                       # disabled in public mode by default
  db_path: "~/.captain-claw/intuitions.db"  # SQLite database path
  # Idle dreaming — dream during inactive hours
  idle_dream_enabled: true                  # dream even when nobody is talking
  idle_dream_interval_seconds: 3600         # dream every hour during idle
  idle_dream_min_session_messages: 5        # minimum messages before idle dreaming
  # Musical cognition — tension tracking
  tension_decay_multiplier: 0.5             # tensions decay at half the normal rate
  tension_delete_threshold: 0.05            # lower deletion threshold for tensions
  max_open_tensions: 10                     # cap on simultaneous open tensions
  # Musical cognition — maturation pipeline
  maturation_enabled: true                  # new intuitions mature before surfacing
  maturation_cycles_required: 2             # dream cycles required before surfacing
  maturation_skip_importance: 9             # importance >= this skips maturation (fortissimo)
```

### cognitive_tempo

Processing depth detection and user cognitive rhythm modeling. Analyzes conversation signals to adapt between deep contemplative processing (adagio) and rapid task execution (allegro). Pure heuristics — no LLM calls.

```yaml
cognitive_tempo:
  enabled: false                            # enable tempo detection (opt-in)
  analysis_window: 5                        # recent messages to analyze
  adagio_threshold: 0.35                    # below this = deep contemplative mode
  allegro_threshold: 0.65                   # above this = rapid execution mode
  user_tempo_weight: 0.4                    # weight of user behavior signals
  content_depth_weight: 0.6                 # weight of content analysis signals
  adjust_context_injection: true            # let tempo affect intuition count in context
  adjust_response_guidance: true            # let tempo affect system prompt guidance
```

### cognitive_metrics

Tracking system for musical cognition features — records events and snapshots for before/after validation of tension tracking, maturation pipeline, and tempo detection.

```yaml
cognitive_metrics:
  enabled: true                             # enable metrics tracking (on by default)
  db_path: "~/.captain-claw/cognitive_metrics.db"  # SQLite database path
  auto_snapshot_interval_hours: 24          # hours between automatic snapshots
  max_events: 10000                         # max events before pruning
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

## Cross-Session Playbook Memory

Captain Claw includes a persistent playbook system that captures proven orchestration patterns (do/don't pseudo-code) from rated sessions. Playbooks survive restarts and are auto-injected into the planning context when the agent encounters similar tasks.

### How It Works

- **Manual creation:** Use the `playbooks` tool `add` action or the REST API to register a playbook with a name, task type, and do/don't patterns.
- **Auto-distillation from session rating:** Rate a session as "good" or "bad" via the `rate` action. The system extracts a compact session summary and tool trace, then runs a standalone LLM call to distill a reusable playbook with classified task type, trigger description, and pseudo-code patterns.
- **Auto-injection:** When a new task arrives, the agent classifies it by task type using keyword heuristics and retrieves matching playbooks. Up to 2 playbooks are injected into the planning context as structured do/don't blocks.
- **Usage tracking:** Each time a playbook is retrieved and injected, its usage count and last-used timestamp are updated.

### Task Type Classifications

| Task Type | Description |
|---|---|
| `batch-processing` | Processing multiple files/items in a loop |
| `web-research` | Fetching and analyzing multiple web sources |
| `code-generation` | Writing code, scripts, or implementations |
| `document-processing` | Extracting/converting documents (PDF, DOCX, etc.) |
| `data-transformation` | Converting data formats (CSV, JSON, etc.) |
| `orchestration` | Multi-step workflow coordination |
| `interactive` | Back-and-forth clarification-heavy tasks |
| `file-management` | Renaming, moving, organizing files |
| `other` | Anything that does not fit above |

### Distillation Process

When a session is rated:

1. The session's messages are extracted into a compact summary (max 2000 chars)
2. An ordered list of tool calls is extracted (max 30 entries)
3. A standalone LLM call analyzes the summary and tool trace
4. The LLM produces a structured playbook with task type, name, trigger description, do/don't patterns, and reasoning
5. If rated "good", the `do_pattern` reflects what worked; if "bad", the `dont_pattern` captures the anti-pattern

Patterns are abstract pseudo-code focused on orchestration decisions (tool ordering, looping strategy, context management) — not task-specific content.

### Tracked Fields

| Field | Description |
|---|---|
| `name` | Short descriptive name |
| `task_type` | Classification from the 9 types |
| `rating` | `good` or `bad` |
| `do_pattern` | Pseudo-code of the recommended approach |
| `dont_pattern` | Pseudo-code of what to avoid |
| `trigger_description` | When this playbook should activate |
| `reasoning` | Why the pattern matters |
| `tags` | Comma-separated tags |
| `use_count` | How many times the playbook has been retrieved |
| `source_session` | Session ID it was distilled from |

### Playbook Override

The web UI playbooks editor allows overriding auto-selection:

- **Auto** (default) — the agent selects playbooks based on task type matching
- **None** — disable playbook injection entirely
- **Specific playbook** — force a particular playbook regardless of task type

### Playbooks Editor

The web UI includes a Playbooks editor at `/playbooks`. It provides a visual interface for browsing, creating, editing, and selecting playbooks.

**Features:**
- **Playbook list** — browse all playbooks with task type badges, usage counts, and ratings
- **Detail view** — view and edit all playbook fields including do/don't patterns
- **Create** — add new playbooks manually
- **Delete** — remove playbooks with confirmation
- **Override selector** — choose auto, none, or a specific playbook for the current session

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/playbooks` | List all playbooks (up to 200, optional `?task_type=` filter) |
| `GET` | `/api/playbooks/search` | Search playbooks (`?q=` required, optional `?task_type=` filter) |
| `GET` | `/api/playbooks/{id}` | Get one playbook by ID |
| `POST` | `/api/playbooks` | Create a new playbook |
| `PATCH` | `/api/playbooks/{id}` | Update a playbook (partial update) |
| `DELETE` | `/api/playbooks/{id}` | Delete a playbook |

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

## Self-Reflection System

Captain Claw can periodically assess its own performance by reviewing recent conversations, memory facts, completed tasks, and the previous reflection. The output is a set of actionable self-improvement directives injected into the system prompt, enabling the agent to learn and adapt over time.

### How It Works

1. **Context gathering** — collects the last 20 session messages, memory facts (up to 30), completed tasks/cron since the last reflection, and the previous reflection summary.
2. **LLM generation** — sends the gathered context to the LLM with a specialized reflection system prompt. The LLM produces generalized, reusable improvement instructions (never references specific tasks or sessions).
3. **Persistence** — the reflection is saved as a timestamped Markdown file in `~/.captain-claw/reflections/`.
4. **Prompt injection** — only the newest reflection is loaded into the system prompt via the `{reflection_block}` placeholder, keeping the prompt lean.
5. **Token tracking** — each reflection generation call is logged to the LLM usage table like any other interaction.

### Auto-Trigger

Self-reflection runs automatically when both conditions are met:

- At least **4 hours** since the last reflection (`AUTO_REFLECT_COOLDOWN_SECONDS`)
- At least **10 messages** in the current session (`AUTO_REFLECT_MIN_MESSAGES`)

The trigger fires as a fire-and-forget `asyncio.create_task()` after agent turns — it never blocks the chat loop. Failures are logged but non-fatal.

### Slash Commands

| Command | Description |
|---|---|
| `/reflection` | Show the latest self-reflection summary |
| `/reflection generate` | Manually trigger a new reflection |
| `/reflection list` | List recent reflections with timestamps |

### Web UI

Navigate to `/reflections` to view all reflections. The page shows expandable cards with timestamps, topics reviewed, and the full reflection text. The newest reflection is highlighted as "Active". Each card has a delete button, and a "Generate Reflection" button triggers a new reflection on demand.

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/reflections` | List all reflections (newest first) |
| `GET` | `/api/reflections/latest` | Get the current active reflection |
| `POST` | `/api/reflections/generate` | Trigger a new reflection |
| `DELETE` | `/api/reflections/{timestamp}` | Delete a reflection by timestamp |

### Storage

Reflections are stored as Markdown files in `~/.captain-claw/reflections/` with timestamps as filenames. The format uses `## Section` headers for metadata (Timestamp, Summary, Topics Reviewed, Token Usage). Loading uses mtime-based caching — the file is only re-read when its modification time changes.

---

## Insights

Captain Claw automatically extracts persistent knowledge from conversations — facts, contacts, decisions, deadlines, and other durable information. Insights are stored in SQLite with FTS5 full-text search and are injected into the system prompt to inform future conversations.

### How It Works

1. **Auto-extraction** — after every 8 messages (configurable), the agent reviews recent conversation and extracts structured insights with entity keys, categories, and importance ratings.
2. **Deduplication** — new insights are checked against existing ones via entity key matching and BM25 text similarity to prevent duplicates.
3. **Storage** — insights are stored in `~/.captain-claw/insights.db` (SQLite + FTS5, WAL mode).
4. **Context injection** — relevant insights are injected into the system prompt via the `{insights_block}` placeholder, up to `max_items_in_prompt` (default 8).
5. **Token tracking** — each extraction call is logged to the LLM usage table.

### Slash Commands

| Command | Description |
|---|---|
| `/insight` or `/insights` | List recent insights |
| `/insight <query>` | Search insights by keyword |
| `/insight stats` | Show insight statistics |
| `/insight add <content>` | Manually add an insight |
| `/insight delete <id>` | Delete an insight |

### Web UI

Navigate to `/insights` to browse all insights. The page shows searchable, filterable cards with category badges, importance ratings, and entity keys. Click any card to view details and edit.

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/insights` | List/search insights (params: q, category, limit) |
| `GET` | `/api/insights/stats` | Insight statistics |
| `GET` | `/api/insights/{id}` | Get single insight |
| `POST` | `/api/insights` | Create insight manually |
| `PATCH` | `/api/insights/{id}` | Update insight |
| `DELETE` | `/api/insights/{id}` | Delete insight |

### Storage

Insights are stored in `~/.captain-claw/insights.db` with a `insights` table and `insights_fts` FTS5 virtual table. In public mode with session isolation, each session uses a separate database file.

---

## Nervous System (Dreaming)

The nervous system is an autonomous, proactive cognitive layer that "dreams" — finding connections, patterns, and hypotheses across all memory types (working memory, semantic memory, deep memory, insights, and reflections). It operates as a subconscious process, generating "intuitions" that are surfaced in the agent's context to guide behavior. Inspired by how classical music creates cognitive impact, it includes musical cognition features: unresolved tension tracking (holding contradictions like dissonance), a maturation pipeline (contemplative pause before surfacing), and cognitive tempo detection (adapting processing depth to conversation rhythm).

### How It Works

1. **Dream trigger** — after every 12 messages (configurable), a background dream cycle fires as a fire-and-forget `asyncio.create_task()`. 5-minute cooldown between dreams. Additionally, **idle dreaming** fires during inactive hours (default: every 1 hour) via the cron scheduler loop, allowing the agent to dream even when nobody is talking — as long as a session with at least 5 messages exists.
2. **Cross-layer sampling** — the dream function samples ~2000 tokens across all memory layers: recent messages, top insights, latest reflection, semantic memory search results, deep memory search results, existing intuitions for dedup, open tensions for resolution checking, and maturing intuitions for refinement.
3. **LLM synthesis** — sends the sampled context to the LLM with specialized dreaming prompts. The LLM identifies non-obvious connections, recurring patterns, speculative hypotheses, and unresolved tensions. It also refines maturing intuitions and checks if open tensions have been resolved.
4. **Intuition storage** — up to 3 intuitions per cycle, stored in `~/.captain-claw/intuitions.db` (SQLite + FTS5) with thread type, confidence score (0.0-1.0), importance rating (1-10), source layer tracking, tags, resolution state, and maturation state.
5. **Maturation** — new intuitions enter a maturation pipeline (raw → maturing → mature). Only mature intuitions are surfaced in context. High-importance intuitions (≥ 9) and tensions skip maturation.
6. **Context injection** — mature, high-confidence intuitions are injected into the system prompt via the `{nervous_system_block}` placeholder and per-turn context notes. Cognitive tempo detection adjusts how many intuitions are surfaced (more in adagio mode, fewer in allegro).
7. **Decay** — unvalidated intuitions lose confidence over time (default: 0.05/day after 7 days of inactivity). Below 0.1 confidence, they are deleted. Tensions decay at half rate and persist down to 0.05 confidence. Hard cap of 200 intuitions.
8. **Metrics** — each dream cycle records cognitive metrics (tensions created/resolved, maturation throughput, tempo distribution) for before/after validation.

### Thread Types

| Type | Description |
|---|---|
| `connection` | Link between two seemingly unrelated pieces of information |
| `pattern` | Recurring theme observed across multiple sources |
| `hypothesis` | Speculative but plausible inference about meaning or intent |
| `association` | Thematic grouping that could inform future context |
| `unresolved` | A contradiction, open question, or tension between pieces of information — held deliberately rather than resolved, like musical dissonance |

### Musical Cognition

Inspired by how classical music creates deep cognitive impact through tension/release, dynamic contour, and entrainment, the nervous system includes three additional features:

#### Unresolved Tension Tracking

The agent can hold contradictions and open questions as first-class insights rather than forcing resolution. When the dreaming system detects genuine tension between memory layers — conflicting insights, contradictory user preferences, or open questions — it creates an `unresolved` intuition.

- **Slower decay** — tensions decay at half the normal rate (`tension_decay_multiplier: 0.5`), giving them time to mature
- **Lower deletion threshold** — tensions persist down to 0.05 confidence (vs 0.1 for normal intuitions)
- **Resolution tracking** — when new information resolves a tension, the dream cycle transforms it into a `connection` or `pattern` with a confidence boost
- **Context formatting** — tensions appear as `[TENSION]` in the agent's context, signaling that they should be held, not solved
- **Cap** — maximum 10 simultaneous open tensions (`max_open_tensions`)

#### Maturation Pipeline (Contemplative Pause)

New intuitions don't surface immediately. They enter a maturation pipeline where subsequent dream cycles can refine them before they appear in the agent's context — like how musical themes develop through repetition with variation.

- **Raw → Maturing → Mature** — new intuitions start as "raw", advance to "maturing" after the first dream cycle, and become "mature" (surfaceable) after the configured number of cycles
- **Cycles required** — default 2 dream cycles before surfacing (`maturation_cycles_required`)
- **Fortissimo bypass** — intuitions with importance ≥ 9 skip maturation entirely, surfacing immediately (like a sudden climax in music)
- **Refinement** — during maturation, the dreaming LLM can adjust confidence and importance based on new evidence
- **Tensions skip maturation** — `unresolved` intuitions are always surfaced immediately (the tension IS the insight)

#### Cognitive Tempo Detection

Pure heuristic analysis of conversation signals to determine whether the interaction calls for deep contemplative processing (adagio) or rapid task execution (allegro). No LLM calls — analyzes message metadata only.

**Signals analyzed:**
- Message length (longer → contemplative)
- Time gaps between messages (longer → deliberate)
- Question depth (why/how/what-if vs do/run/show)
- Word complexity (average word length)
- Reflective vs task-oriented language markers

**Modes:**
| Mode | Tempo range | Effect |
|---|---|---|
| `adagio` | < 0.35 | More intuitions surfaced (up to 8), lower confidence threshold, system prompt encourages cross-referencing and speculation |
| `moderato` | 0.35 - 0.65 | Default behavior |
| `allegro` | > 0.65 | Fewer intuitions surfaced (min 1), higher confidence threshold, system prompt encourages concise action-oriented responses |

The tempo is re-assessed each turn and displayed in the nervous system context note.

#### Cognitive Metrics

All musical cognition features are tracked via a dedicated metrics system (`~/.captain-claw/cognitive_metrics.db`) that records events across features:

- `tension_created` / `tension_resolved` — tracks tension lifecycle, resolution rate, and lifespan
- `maturation_started` / `maturation_completed` — tracks pipeline throughput and completion rate
- `tempo_detected` — tracks mode distribution across sessions
- `dream_cycle` — tracks per-cycle statistics (intuitions stored, tensions open, matured count)

Use `compare_snapshots()` to compare metrics at different points in time and validate whether these features improve reasoning quality.

### Validation and Strengthening

- **Validation** (`/intuition validate <id>`) — permanently protects an intuition from decay, boosts confidence by +0.2 (cap 1.0) and importance by +1 (cap 10).
- **Access tracking** — each time an intuition is surfaced in context, its access count increments. Higher access counts improve ranking in `get_for_context`.
- **Session bonus** — intuitions from the current session get +0.2 confidence bonus; recently accessed ones get +0.1.

### Session Bleeding

- **Admin mode** — global database. All intuitions are visible across sessions. Source session is tagged for weighting. Cross-session intuitions surface at natural confidence level.
- **Public mode** — disabled by default (`allow_public: false`). If enabled, fully session-scoped via separate database files — no bleeding.

### Slash Commands

| Command | Description |
|---|---|
| `/intuition` or `/intuitions` | List recent intuitions |
| `/intuition <query>` | Search intuitions by keyword |
| `/intuition dream` | Manually trigger a dream cycle |
| `/intuition stats` | Show intuition statistics |
| `/intuition add <content>` | Manually add an intuition |
| `/intuition validate <id>` | Validate (protect from decay) |
| `/intuition delete <id>` | Delete an intuition |

### Web UI

Navigate to `/intuitions` to browse all intuitions. The page shows:
- **Stats bar** — total count, validated count, average confidence, average importance
- **Searchable list** — cards with thread type badges, confidence bars (color-coded: green/yellow/red), importance ratings, source layers, validation status
- **Detail modal** — edit content, thread type, confidence, importance, tags; validate or delete
- **Dream button** — manually trigger a dream cycle from the toolbar

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/nervous-system` | List/search intuitions (params: q, thread_type, min_confidence, limit) |
| `GET` | `/api/nervous-system/stats` | Count, avg confidence, type distribution |
| `GET` | `/api/nervous-system/{id}` | Get single intuition |
| `POST` | `/api/nervous-system` | Create intuition manually |
| `POST` | `/api/nervous-system/dream` | Manually trigger a dream cycle |
| `PATCH` | `/api/nervous-system/{id}` | Update intuition |
| `DELETE` | `/api/nervous-system/{id}` | Delete intuition |

### Storage

Intuitions are stored in `~/.captain-claw/intuitions.db` with an `intuitions` table and `intuitions_fts` FTS5 virtual table for full-text search. The schema includes confidence, importance, access count, validation status, source layers, decay tracking, resolution state (for tensions), and maturation state (raw/maturing/mature). Cognitive metrics are stored separately in `~/.captain-claw/cognitive_metrics.db`. In public mode (if enabled), each session uses a separate database file.

### Cost Control

- **Off by default** — zero cost unless opted in via `nervous_system.enabled: true`
- Input budget capped at ~2000 tokens per dream (slightly higher with tension and maturation context)
- Output capped at 800 tokens
- 5-minute cooldown between dreams (vs 60s for insights)
- 12-message interval (vs 8 for insights)
- Max 3 intuitions per cycle
- Hard cap of 200 stored intuitions with aggressive decay
- **Idle dreaming** — fires at most once per hour during idle time (configurable). Same token budget as regular dreams (~2800 tokens per cycle). Disable via `idle_dream_enabled: false`
- **Cognitive tempo** — zero LLM cost (pure heuristics). Enable separately via `cognitive_tempo.enabled: true`
- **Cognitive metrics** — zero LLM cost (event recording only). Enabled by default, prunes at 10,000 events

---

## Brain Graph

Interactive 3D force-directed visualization of the agent's cognitive topology at `/brain-graph`. Built on Three.js and 3d-force-graph (loaded from CDN, no build step), it renders all cognitive data sources as a navigable graph in WebGL.

### Node Types

| Type | Shape | Color | Size by |
|------|-------|-------|---------|
| Session | Transparent wireframe sphere | Gray | Message count (auto-sizes to enclose children) |
| Message | Tetrahedron | Yellow (user) / Light blue (assistant) | Fixed |
| Insight | Sphere | Gold | Importance |
| Intuition | Sphere | Purple | Confidence + importance |
| Tension | Icosahedron | Red | Importance |
| Task | Box | Blue | Priority |
| Briefing | Cone | Green | Fixed |
| Todo | Octahedron | Teal | Priority |
| Contact | Dodecahedron | Orange | Mention count |
| Cognitive Event | Small sphere | Cyan | Fixed |

### Edge Types

| Relation | Meaning |
|----------|---------|
| contains | Session contains child nodes |
| sequence | Sequential message chain within a session |
| supersedes | Insight replaced an older insight (evolution) |
| resolves | Intuition resolved a tension |
| triggers | Insight or intuition triggered a sister session task |
| parent | Todo subtask hierarchy |
| source | Intuition synthesized from source insights |

### Features

- **Live updates** — WebSocket streaming adds new nodes in real-time as insights and intuitions are created
- **Detail panel** — click any node to see metadata, connections, and navigate with Prev/Next buttons
- **Connection traversal** — clickable connections list shows all linked nodes; click to jump between them
- **Full content modal** — "Show full content" button fetches complete message text from the session database and renders as markdown
- **Search** — filter nodes by label, type, or status
- **Type filters** — toggle visibility of each node type via checkboxes
- **Node limit slider** — control how many nodes per type are loaded (20-500)
- **Dynamic session spheres** — session wireframe spheres auto-resize to enclose their furthest child node
- **Deep linking** — the brain button on chat and computer messages opens `/brain-graph?focus_ts=<timestamp>` which auto-selects and zooms to the matching node
- **Keyboard navigation** — Arrow keys or `[`/`]` to step through connected nodes, Escape to close panels
- **Public mode** — fully supported with session-isolated data

### API

```
GET /api/brain-graph                    — full graph as {nodes, links, stats}
GET /api/brain-graph?limit=200          — max nodes per type
GET /api/brain-graph?types=insight,intuition — filter by node type
GET /api/brain-graph/message/{msg_id}   — fetch full message content by ID
```

---

## Process of Thoughts

Full lineage tracking across all cognitive subsystems — a persistent, multi-session thought topology that connects every cognitive artifact via provenance IDs.

### Provenance Fields

| Component | Field | Links to |
|-----------|-------|----------|
| Message | `message_id` | Auto-generated 12-char hex ID on every message |
| Insight | `source_message_id` | The user message that triggered insight extraction |
| Insight | `supersedes_id` | The older insight this one replaced (evolution chain) |
| Intuition | `source_message_id` | The user message active during the dream cycle |
| Intuition | `resolved_from_id` | The tension this intuition resolved |
| Intuition | `source_ids` | Array of insight/intuition IDs used as synthesis input |
| Sister Task | `source_type` + `source_id` | The insight or intuition that triggered the task |
| Todo | `parent_id` | Parent todo for subtask hierarchy |
| Todo | `triggered_by_id` | The insight or intuition that created this todo |

### Thought Chain Example

```
User message (msg_abc123)
  → Insight extracted (source_message_id: abc123)
    → Supersedes older insight (supersedes_id: old_insight_id)
    → Dream cycle picks up insight (source_ids: [insight_id])
      → Intuition created (source_message_id: abc123)
        → Sister session task triggered (source_id: intuition_id)
          → Briefing produced
            → Todo created (triggered_by_id: intuition_id)
```

### Schema Migrations

All new columns use safe `ALTER TABLE ADD COLUMN` with try/except for idempotency. Existing data gets `NULL` in new columns (fully backward compatible).

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

The orchestrator decomposes complex requests into a task DAG and executes tasks in parallel across separate sessions. Tasks share data through a structured workspace, can enforce JSON Schema output validation, and emit real-time trace spans for observability.

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

### Shared Workspace

Tasks within an orchestration run share a namespaced key-value store for passing structured data between tasks without relying on intermediate files.

- **Automatic**: Every task's text output and validated structured output are written to the workspace automatically
- **Manual**: Workers can use the `workspace_read` and `workspace_write` tools for fine-grained data sharing
- **Namespaced**: Keys are prefixed by task ID (`task_id:key`) to prevent collisions
- **Injected**: Upstream task data is injected into downstream worker prompts automatically
- **Real-time**: Workspace changes are broadcast to the Flight Deck UI via WebSocket

Each task can declare `workspace_inputs` (keys to read from upstream tasks) and `workspace_outputs` (keys it will produce) for explicit data flow documentation.

### Structured Output Validation

Tasks can declare a JSON Schema via `output_schema` to enforce structured output:

1. After the worker agent completes, the output is parsed and validated against the schema
2. If validation fails, the agent gets **one automatic retry** with the error and schema fed back
3. On success, the validated data is stored in `task.validated_output` and written to the shared workspace as JSON
4. On final failure, the task is marked as failed with the validation error

This enables reliable structured data pipelines — e.g., "extract a list of products as `{name, price, url}`" with guaranteed schema conformance.

### Explicit Task Pipelines (run_tasks API)

The `run_tasks()` and `prepare_tasks()` APIs let you define a task DAG explicitly without LLM decomposition:

```python
# REST API (proxied through Flight Deck)
POST /api/orchestrator/run-tasks
{
  "tasks": [
    {"id": "t1", "title": "Fetch data", "description": "...", "depends_on": []},
    {"id": "t2", "title": "Analyze", "description": "...", "depends_on": ["t1"],
     "output_schema": {"type": "object", "properties": {"score": {"type": "number"}}}}
  ],
  "synthesis_instruction": "Summarize the analysis"
}
```

Each task supports: `id`, `title`, `description`, `depends_on`, `model_id`, `session_name`, `session_id`, `output_schema`, `output_schema_name`, `workspace_outputs`, `workspace_inputs`.

This powers Flight Deck's Swarm Workflows and enables programmatic pipeline construction.

### Trace Timeline (Observability)

Every orchestration run emits structured trace spans for real-time observability:

| Span Type | What It Tracks |
|---|---|
| `decompose` | LLM decomposition of the request into tasks |
| `execution` | Overall DAG execution phase with completed/failed counts |
| `task` | Individual worker task with token usage and error details |
| `synthesize` | Final synthesis step |

Spans are broadcast to Flight Deck's chat panel in real-time, where they appear as a **Gantt-style trace timeline** (click the Activity icon in the chat tab bar). The timeline shows:
- Color-coded horizontal bars per span type
- Running/completed/failed status with live animation
- Duration and token usage per span
- Summary header with total span count, completion stats, and token accounting

Trace data is also available via `GET /api/orchestrator/traces` (spans + summary).

### Dashboard (Web UI)

The orchestrator dashboard at `/orchestrator` shows:
- Visual task DAG with color-coded status nodes
- Summary bar with task counts and token usage
- Detail panel for each task (edit, restart, pause, resume, postpone)
- Chronological event log
- Workspace input/output indicators on task nodes
- Schema validation badges on tasks with output schemas

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

**1. Run BotPort:**

BotPort is included with `pip install captain-claw`. Start the hub:

```bash
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

### BotPort Swarm

BotPort Swarm adds DAG-based multi-agent orchestration on top of BotPort's routing layer. Decompose complex goals into task graphs with dependencies, route each task to specialist agents, and execute with configurable concurrency — all managed from the BotPort dashboard.

#### How It Works

```
┌─────────────┐
│  Decompose  │  Goal → DAG of tasks with dependencies
└──────┬──────┘
       ▼
┌─────────────┐
│  Design     │  LLM picks optimal agent persona + model per task
└──────┬──────┘
       ▼
┌─────────────┐
│  Execute    │  Engine advances DAG, routes tasks to CC instances
└──────┬──────┘
       ▼
┌─────────────┐
│  Collect    │  Results, artifacts, and files gathered per task
└─────────────┘
```

1. **Decompose** — Submit a goal. The decomposer (LLM-powered) breaks it into tasks with dependencies, forming a DAG.
2. **Design agents** — The agent designer analyzes each task and assigns an optimal persona and model tier (fast/mid/premium).
3. **Execute** — The swarm engine advances the DAG, launching tasks whose dependencies are satisfied. Tasks are routed to connected Captain Claw instances via BotPort concerns.
4. **Monitor** — Track progress in real-time on the dashboard with a visual DAG canvas, task status cards, and audit log.

#### Task Configuration

Each task in a swarm supports:

| Field | Description |
|---|---|
| `title` | Short task name |
| `prompt` | Full instructions sent to the agent |
| `persona` | Preferred agent persona for routing |
| `model_hint` | Preferred model (e.g. `claude-sonnet`, `gpt-4o`) |
| `timeout_sec` | Per-task timeout (default: inherited from swarm) |
| `retry_count` | Max retries on failure (default: 0) |
| `retry_backoff_sec` | Backoff between retries |
| `fallback_persona` | Alternative persona for retries |
| `needs_approval` | Require human approval before execution |
| `priority` | Task priority (higher = scheduled first) |

#### Error Policies

| Policy | Behavior |
|---|---|
| `fail_fast` | Stop the entire swarm on first task failure |
| `continue_on_error` | Mark failed tasks, continue with independent tasks |
| `manual_review` | Pause on failure, wait for human decision |

#### Timeout Escalation

Swarm uses a three-stage timeout system:

1. **Warning** — At 80% of timeout, logs a warning
2. **Extension** — At 100%, extends by 50% (one extension allowed)
3. **Failure** — After extension, task fails with timeout error

#### Checkpointing

Save and restore swarm state at any point:

- **Create checkpoint** — Snapshots all task states and results
- **Restore checkpoint** — Rolls back to a previous state for re-execution

#### File Transfer

Agents can produce and consume files during swarm execution:

- Files stored in `workspace-botport/swarm/<swarm_id>/<agent>/`
- Transfer over WebSocket using gzip compression + base64 encoding
- Max file size: 50 MB
- File manifest with SHA-256 hashes

#### Swarm Scheduling

Recurring swarms via cron expressions:

```yaml
schedule: "0 9 * * 1"    # Every Monday at 9 AM
```

Supports standard cron format: `minute hour day month weekday`. Wildcard (`*`), step (`*/N`), range (`N-M`), and list (`N,M,O`) syntax.

#### Swarm Dashboard

Access the swarm UI from the BotPort dashboard. Features:

- **Project management** — Organize swarms into projects
- **DAG canvas** — Visual task graph with status colors and dependency arrows
- **Task monitoring** — Real-time status updates with polling
- **Approval gates** — Approve or reject pending tasks from the UI
- **File manager** — Upload, download, and browse swarm files
- **Audit log** — Complete event history for each swarm

#### Swarm API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/swarm/projects` | List all projects |
| `POST` | `/api/swarm/projects` | Create a project |
| `GET` | `/api/swarm/swarms` | List swarms (optionally by project) |
| `POST` | `/api/swarm/swarms` | Create a swarm |
| `POST` | `/api/swarm/swarms/{id}/start` | Start a swarm |
| `POST` | `/api/swarm/swarms/{id}/pause` | Pause a running swarm |
| `POST` | `/api/swarm/swarms/{id}/resume` | Resume a paused swarm |
| `POST` | `/api/swarm/swarms/{id}/cancel` | Cancel a swarm |
| `POST` | `/api/swarm/swarms/{id}/decompose` | Decompose goal into tasks |
| `POST` | `/api/swarm/swarms/{id}/design-agents` | Design agent specs for tasks |
| `POST` | `/api/swarm/swarms/{id}/checkpoints` | Create a checkpoint |
| `POST` | `/api/swarm/swarms/{id}/checkpoints/{cp}/restore` | Restore a checkpoint |
| `POST` | `/api/swarm/tasks/{id}/approve` | Approve a pending task |
| `POST` | `/api/swarm/tasks/{id}/reject` | Reject a pending task |
| `GET` | `/api/swarm/swarms/{id}/files` | List swarm files |
| `POST` | `/api/swarm/swarms/{id}/files` | Upload a file |
| `GET` | `/api/swarm/swarms/{id}/files/{name}` | Download a file |

---

## Computer

The Computer is a retro-themed research workspace accessible at `/computer`. It provides a three-panel layout designed for extended research sessions, visual output generation, and interactive exploration trees.

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  🖥 Computer │ 🎨 Theme │ 🧠 Model │ 👤 Persona │ Tier ▾ │  │
├────────────────────────┬─────────────────────────────────────┤
│  📝 Input              │  📊 Output                          │
│  [textarea]            │  [Answer] [Blueprint] [Files]       │
│                        │  [Visual] [Map]                     │
│  📎 Attach │ 📁 Folder │                                     │
│  [Send]                │  Rendered markdown, themed HTML,     │
├────────────────────────┤  exploration tree visualization     │
│  📋 Activity Log       │                                     │
│  [timestamped entries] │                                     │
└────────────────────────┴─────────────────────────────────────┘
```

**Left column** (resizable, width persisted to localStorage):
- **Input panel** — Textarea with auto-resize, attachment bar (images, PDF, DOCX, XLSX, PPTX, MD, TXT, CSV), action buttons (📎 Attach, 📁 Folder, Send)
- **Activity log** — Real-time timestamped entries with type icons (system, user, /btw, tool, thinking, error). Max 200 entries with auto-trim.

**Right column** — Tabbed output:
- **Answer** — Rendered markdown response from the agent
- **Blueprint** — Step decomposition of the agent's task execution
- **Files** — Session-created files grouped by folder, with search
- **Visual** — LLM-generated themed HTML rendered in a sandboxed iframe, with PDF export via WeasyPrint
- **Map** — Exploration tree visualization with zoom controls

The resize handle between columns is draggable. The left panel width is stored in `localStorage` and restored on page load.

### Theme System

Computer ships with 14 built-in themes, each with unique colors, typography, window decorations, and boot sequences:

| Theme | Style |
|---|---|
| Amiga Workbench | Blue/orange/grey, beveled 3D borders, Topaz font |
| Atari ST GEM | Green desktop, drop-shadow windows, system font |
| C64 GEOS | Blue/white, Commodore-style UI elements |
| Classic Mac | Black & white, Chicago font, 1-bit aesthetic |
| Windows 3.1 | Teal/grey, Program Manager style |
| Hacker Terminal | Black/green, monospaced, scanline glow effect |
| Modern | Dark slate, blue accent, rounded corners, clean sans-serif |
| Windows 11 | Fluent design, Segoe UI, acrylic feel |
| macOS | SF Pro, vibrancy, rounded window corners |
| iPhone | iOS-style, compact, touch-friendly |
| Android | Material Design, Roboto, elevation shadows |
| Nokia 7110 | Monochrome LCD, tiny pixel font |
| Nokia Communicator | Dual-screen business phone aesthetic |

Each theme defines CSS custom properties (39 variables) covering backgrounds, text, accents, chrome, bevels, title bars, scrollbars, fonts, and layout dimensions.

**Custom themes:**
1. Click 🎨 Theme → Download Template to get a JSON template
2. Edit colors, fonts, and optionally add a boot sequence
3. Upload the JSON file — it's saved to `localStorage` and immediately applied
4. Custom themes appear alongside built-in themes in the selector

**Boot sequences:** Each theme plays a unique startup animation on load (e.g., Amiga's "Insert disk → Loading Kickstart → Workbench 1.3", Hacker's Matrix-style "Wake up, Neo...").

### Model Selector

Click the 🧠 button to open a modal grid showing all available models from `config.yaml`'s `model.allowed` list. Each card displays:
- Provider icon (emoji by provider: OpenAI, Anthropic, Google, Ollama, etc.)
- Model name and description
- Pricing info and capability badges

Selecting a model sends a `set_model` WebSocket message to switch the entire session's model — affecting all LLM operations (chat, visual generation, exploration). The selection is persisted to `localStorage` and re-applied on reconnect.

### Persona Selector

Click the 👤 button to open a modal grid showing all available personas (agent personality + per-user profiles). Each card displays the persona name. Selecting a persona switches the active persona for the session. The selection is persisted to `localStorage`.

### PDF Export

The Visual tab and HTML file preview include a PDF export button. Clicking it sends the full HTML to the backend where WeasyPrint renders it to PDF, preserving all CSS styling (backgrounds, fonts, colors, layouts). The PDF filename is derived from the task prompt or HTML filename.

### Visual Generation

After the agent responds, the Visual tab generates themed HTML output:

1. The agent's answer is sent to `POST /api/computer/visualize` along with the current theme and token tier
2. The LLM generates standalone HTML styled to match the active theme (e.g., Amiga colors and beveled borders, hacker terminal green-on-black)
3. The HTML is rendered in a sandboxed iframe with an explore bridge injected

**Token tiers** control the visual generation budget:

| Tier | Max Tokens |
|---|---|
| Micro | 4,096 |
| Minimal | 8,192 |
| Standard | 16,384 (default) |
| Generous | 32,768 |

If the agent created HTML files during its task, a choice bar appears: **Use File** (render the agent's file) or **Generate New** (create themed visual via LLM).

### Exploration Tree

The exploration tree enables multi-turn research within the Visual tab:

1. Elements with `class="explore-link"` in generated HTML become clickable
2. Clicking one shows a confirmation bar with the topic
3. The user can edit the prompt before sending
4. Each exploration creates a tree node linked to its parent
5. The **Map** tab visualizes the full tree with zoom controls

Nodes are persisted to SQLite (`exploration_nodes` table) and loaded on page load. Each node stores: prompt, answer, visual HTML, theme, source (click or manual), and parent linkage.

**REST API:**
- `POST /api/computer/exploration` — Save a new node
- `GET /api/computer/exploration` — List nodes for current session
- `GET /api/computer/exploration/{id}` — Get a single node
- `PUT /api/computer/exploration/{id}/visual` — Update visual HTML
- `DELETE /api/computer/exploration/{id}` — Delete a node

### Folder Browser

Click 📁 to open the folder browser modal with two tabs:

**Local tab:**
- Lists currently active folders with remove buttons
- Drive selector buttons for Windows machines (C:\, D:\, etc.)
- Directory browser with breadcrumb navigation
- "Add this folder" button to register a folder for agent file access

**Google Drive tab** (shown when GWS is available):
- Browse My Drive and Shared Drives
- Breadcrumb navigation through folder hierarchy
- Add/remove Google Drive folders for agent access

Active folders (both local and Google Drive) are used by the agent when searching for files outside the workspace.

### Attachments

Attach images and data files to messages:

**Supported formats:**
- Images: PNG, JPG, JPEG, WEBP, GIF, BMP
- Data: CSV, XLSX

**Attachment methods:**
- Click 📎 to open a file picker
- Paste from clipboard (auto-detects images)
- Drag and drop onto the workspace

Images are automatically resized to 1024px max dimension before upload. Files are uploaded to the server and sent as `image_path` or `file_path` in the chat message payload.

### /btw Command

Inject live instructions while a task is running without interrupting the agent:

```
/btw use bullet points for the summary
btw also check the error handling
```

**How it works:**
1. Send `/btw <instruction>` or `btw <instruction>` while the agent is processing
2. The instruction is added to the agent's `_btw_instructions` list
3. The agent incorporates accumulated instructions into remaining subtasks
4. Instructions are cleared when the task completes

**Available in:**
- **Chat** (`/chat`) — detected in the frontend, sent as a WebSocket `btw` message
- **Computer** (`/computer`) — same detection and WebSocket mechanism
- **Telegram** — detected before the per-user lock, so it works even while the agent is busy processing

Multiple `/btw` messages can be sent during a single task — they accumulate and all apply to remaining work.

### Suggested Next Steps

After each agent response, Computer automatically extracts suggested next steps and presents them as clickable buttons below the answer.

**How it works:**

1. When the agent finishes responding, a lightweight follow-up LLM call analyzes the response for explicit suggestions or options
2. If suggestions are found, they appear as interactive buttons below the Answer tab content
3. Clicking a button populates the input box with the corresponding action and sends it automatically
4. Buttons are cleared when a new message is sent

The extraction uses a heuristic pre-filter (looks for bullet/numbered lists in the response) to avoid unnecessary LLM calls on responses that don't contain suggestions.

### Public Mode

Computer supports a public-facing deployment mode where anonymous visitors can use the research workspace through session-isolated access codes, while the rest of the web UI remains locked down.

**Enable public mode:**

```yaml
web:
  auth_token: "your-admin-password"    # required — protects admin access
  public_run: "computer"               # expose only the Computer section
```

**How it works:**

1. Visitors arrive at `/computer` and see a landing page with session management
2. They can **create a new session** (generates a 6-character access code) or **resume an existing session** by entering their code
3. Each session is fully isolated — files, uploads, media, and exploration trees are scoped to the session
4. Only Computer-related routes are accessible; all other pages and APIs return 403 Forbidden
5. Admin users authenticated via `auth_token` bypass all restrictions and see the full UI

**Session isolation:**

- **Files:** Each public session's files are stored in `workspace/saved/<category>/<session_id>/` and `workspace/output/<session_id>/`
- **Uploads:** Image and file uploads are scoped to the session directory
- **File browsing:** Public users can only list and access files belonging to their session
- **Exploration trees:** Scoped to the session via the standard session mechanism

**Allowed routes in public mode:**

| Category | Routes |
|---|---|
| Pages | `/computer`, `/computer/*` |
| API | `/api/computer/*`, `/api/public/*`, `/api/config`, `/api/orchestrator/models`, `/api/user-personalities`, `/api/file/upload`, `/api/image/upload`, `/api/files/*`, `/api/media/*` |
| WebSocket | `/ws` |
| Static | `/static/*`, `/favicon.ico` |

All other routes (Chat, Sessions, Orchestrator, Settings, Datastore, etc.) are blocked for public users.

**Admin access:**

Set `auth_token` in config and navigate to `/?token=your-admin-password` to authenticate as admin. Admin users have unrestricted access to all pages and APIs regardless of public mode settings.

**BYOK (Bring Your Own Key):**

Public users can provide their own LLM API credentials instead of using the server's shared provider. This lets visitors use their own OpenAI, Anthropic, Gemini, xAI, or OpenRouter keys.

- Click the **🔑 BYOK** button in the toolbar (visible only in public mode)
- Select a provider, enter a model name and API key
- Credentials are stored only in the browser (localStorage) and sent over the encrypted WebSocket connection
- Keys are held in server memory only during the session — never logged, persisted, or written to disk
- On WebSocket reconnect, saved credentials are automatically re-applied
- Click **Clear & Use Default** to revert to the server's provider
- Both chat and visual generation use the BYOK provider when active
- BYOK calls are tracked separately in the LLM Usage dashboard with a 🔑 indicator
- The `ollama` provider is blocked for BYOK to prevent SSRF; custom base URLs are not supported

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
| Computer | `/computer` | Retro-themed research workspace with visual generation, exploration trees, and 14+ themes |
| Sessions | `/sessions` | Browse and manage all conversation sessions |
| Orchestrator | `/orchestrator` | Parallel DAG task execution with real-time monitoring |
| Instructions | `/instructions` | Browse and edit instruction templates |
| Cron | `/cron` | Schedule and monitor recurring tasks |
| Workflows | `/workflows` | Browse saved workflows and execution outputs |
| Loop Runner | `/loop-runner` | Execute workflows with variable iteration |
| Memory | `/memory` | Browse to-dos, contacts, scripts, and API registrations |
| Deep Memory | `/deep-memory` | Browse and search Typesense-backed long-term archive |
| Datastore | `/datastore` | Browse and manage structured data tables |
| Personality | `/personality` | Edit agent personality and per-user profiles |
| Insights | `/insights` | Browse persistent insights auto-extracted from conversations |
| Nervous System | `/intuitions` | Browse autonomously discovered patterns, connections, and hypotheses |
| Reflections | `/reflections` | Browse and manage self-reflection entries |
| Files | `/files` | Browse agent-created files and download outputs |
| LLM Usage | `/usage` | Token usage analytics with provider/model filters and cost breakdown |
| Settings | `/settings` | Configure models, tools, and system options |

### Configuration

```yaml
web:
  enabled: false
  host: "127.0.0.1"
  port: 23080
  auth_token: ""                  # set to enable authentication
  auth_cookie_max_age: 90         # days
  public_run: ""                  # "computer" to expose Computer to anonymous visitors
  api_enabled: true               # OpenAI-compatible API proxy
  api_pool_max_agents: 50
  api_pool_idle_seconds: 600.0
```

---

## Flight Deck

Flight Deck is a management dashboard for running multiple Captain Claw agents. It provides a unified UI to spawn, monitor, chat with, and transfer files between agents — whether they run in Docker containers or as local pip-installed instances.

### Starting

```bash
captain-claw-fd                    # default: http://0.0.0.0:25080
captain-claw-fd --port 8080        # custom port
captain-claw-fd --dev              # API-only mode (use with Vite dev server)
python -m captain_claw.flight_deck # alternative
```

Flight Deck serves both the React frontend and the FastAPI backend from a single process. No separate build step is needed for production — the built frontend is bundled with the Python package.

### Features

| Feature | Description |
|---|---|
| Docker container management | Spawn, stop, restart, remove, rebuild, clone Captain Claw containers with full config (provider, model, tools, platforms) |
| Process agent management | Spawn pip-based agents as local subprocesses — no Docker required. Full lifecycle (start/stop/restart/clone/remove), auto-restart on FD startup, clean shutdown |
| Fleet discovery | `GET /fd/fleet` returns all running agents (Docker, process, local). The `flight_deck` tool gives every agent live fleet awareness |
| Agent-to-agent communication | `consult_peer` and `flight_deck` tools let agents discover, consult, and delegate tasks to each other. Synchronous (consult) and fire-and-forget (delegate) modes |
| Multi-user authentication | JWT-based auth with registration, login, refresh tokens, and role-based access (admin/user). First user becomes admin |
| Admin dashboard | User management, plan tier assignment (free/pro/enterprise), per-user quota overrides, usage analytics, and system configuration |
| Rate limiting & quotas | Tiered plan system with per-user limits on agents, requests/minute, spawns/hour, and storage |
| Chat persistence | Server-side chat session and message storage with per-user isolation |
| Settings sync | Per-user settings persisted server-side with automatic localStorage migration |
| Agent config editor | Edit agent config.yaml and .env files in-flight from the UI |
| Local agent management | Register any CC instance by host:port, probe status, connect for chat |
| Multi-agent chat | WebSocket-based chat with multiple agents simultaneously via tabbed interface, resizable panel (320–900px) |
| File browser & transfer | Browse agent files with file viewer (syntax highlighting, image preview), select files, and send them to another agent |
| Context transfer | Forward full conversation history (no truncation) + a task prompt to another agent |
| Container logs | View and stream Docker container logs |
| Agent status | Real-time busy/idle indicators on agent cards with status text |
| Markdown chat | Full GFM markdown rendering (tables, code, lists) in chat messages |
| Director panel | Unified agent overview with status, activity feed, broadcast to all agents, filter/sort, quick actions (stop all, restart all) |
| Operations dashboard | Per-agent token usage, cost estimates, model breakdown, latency stats, cache hit rates — filterable by time period |
| Agent pipelines | Chain agents together — output from one automatically flows to the next with context about the pipeline and source agent |
| Pinned messages | Pin important chat messages with tags for quick reference |
| Shared clipboard | Cross-agent clipboard for sharing text snippets between agents |
| Notification center | Bell icon with unread count, type filters (info/success/warning/error), agent connect/disconnect and pipeline events |
| Keyboard shortcuts | Cmd/Ctrl+1–4 for views, Cmd+D director, Cmd+J chat, Cmd+K shortcuts overlay, Cmd+[ ] switch chat tabs |
| Dark/light theme | Full theme support with toggle in top bar, light mode overrides for all zinc palette and markdown styles |
| Free-form layout | Drag agent cards freely on the desktop, positions persisted to localStorage |
| Embedded chat | Collapsible chat panel directly on agent cards |
| Resizable panels | Director (220–500px), chat (320–900px), and tool panels (280–500px) are all resizable with drag handles |
| Agent Forge | AI-powered team decomposition — describe a goal, get a team of agents with roles, instructions, and tools |
| Fleet-level instructions | Per-agent instructions injected into system prompts, editable from agent config editor |
| Datastore browser | View agent datastore tables and rows directly from agent cards |
| Agent groups & tags | Organize agents into teams and roles with color-coded group badges |
| Trace timeline | Real-time orchestrator observability — Gantt-style span visualization in the chat panel with duration, token usage, phase breakdown (decompose/execute/task/synthesize), and status tracking |

### Agent Forge

Agent Forge is a dedicated Flight Deck page that uses an LLM to decompose a business objective into a team of specialized AI agents.

**How it works:**

1. Navigate to **Agent Forge** in the sidebar
2. Configure the LLM provider, model, and API key (persisted across sessions)
3. Optionally add environment variables (e.g., `BRAVE_API_KEY`) that will be passed to all spawned agents
4. Describe your objective in natural language
5. Click **Decompose into Agents** — the LLM analyzes your goal and proposes a team

**What the LLM generates for each agent:**

- **Name** — kebab-case identifier (e.g., `market-researcher`, `data-analyst`)
- **Role** — position title (e.g., "Senior Research Analyst", "Data Engineer")
- **Lead designation** — one agent is marked as lead coordinator
- **Description** — one-sentence summary of responsibilities
- **Fleet instructions** — detailed operating procedure including:
  - Primary responsibilities and focus areas
  - Tool usage guidance referencing specific Captain Claw tools
  - Standard Operating Procedure (SOP) as pseudo-code steps
  - Collaboration patterns with other team members
  - Output expectations
- **Tools** — curated selection from 44 available tools based on the agent's role

**Review phase:**

- Edit any field (name, role, description, instructions, tools)
- Change LLM provider/model per agent
- Toggle agent type (process or Docker)
- Designate a different lead agent (crown icon)
- Add or remove agents

**On spawn:**

- Each agent is created with its fleet-level instructions and tool configuration
- A team group is created (all agents added)
- A role group is created per unique role
- The lead agent's name includes `[Lead]` suffix

The forge system prompt includes the full Captain Claw tool reference (44 tools with descriptions) and guidelines for tool selection per role type (research, content, data, coordination, communication, automation).

### Fleet-Level Instructions

Fleet-level instructions are per-agent directives set from Flight Deck that are injected into the agent's system prompt. They apply to every conversation and take effect when the agent's chat session is opened.

**Setting instructions:**

1. Click **Actions** on any agent card, then **Configure**
2. Select the **Instructions** tab
3. Write or paste fleet-level instructions
4. Click **Save**

Instructions are stored in the Flight Deck frontend and sent to agents via the `peer_agents` WebSocket message when a chat session opens. No agent restart is needed — instructions take effect on the next chat connection.

In the agent's system prompt, fleet instructions appear under the heading **"Fleet-Level Instructions"** with the note that they come from the fleet operator.

### Datastore Browser

The datastore browser provides a read-only view of an agent's SQLite-backed datastore tables directly from Flight Deck.

**Accessing:**

Click the **Data** button on any running agent card (available in both compact and expanded view modes).

**Features:**

- Lists all tables with column schemas, types, and row counts
- Click a table to browse rows with pagination (50 rows per page)
- Columns displayed in schema order with monospace font
- Null values shown as italic "null"
- Refresh button to reload data
- 80% viewport modal with back navigation

### Agent Types

**Docker containers** — Spawned from Flight Deck's Spawn Agent page. Full lifecycle management (start/stop/restart/remove/rebuild/clone), logs, and auto-configured networking.

**Process agents** — Spawned as local `captain-claw-web` subprocesses. No Docker required — uses the pip-installed Captain Claw directly. Each agent gets an isolated directory under `fd-data/<agent-slug>/` with its own config.yaml, workspace, database, and skills. Process agents are automatically restarted when Flight Deck starts and cleanly stopped on shutdown.

**Local agents** — Any Captain Claw instance reachable via HTTP. Register with name, host, port, and optional auth token. Useful for remote agents or instances managed outside Flight Deck.

### Spawning Agents

The Spawn Agent page has a **Docker/Process mode toggle** at the top:

- **Docker mode** (violet) — spawns a Docker container with the Captain Claw image
- **Process mode** (emerald) — spawns a local subprocess using `captain-claw-web`

Both modes configure:

- **LLM provider and model** — OpenAI, Anthropic, Gemini, Ollama, OpenRouter
- **Tools** — Select which tools the agent has access to
- **Platforms** — Enable Telegram, Discord, or Slack bots
- **Web UI** — Port and auth token for the agent's own web interface
- **BotPort** — Connect to a BotPort hub for multi-agent routing

Docker mode additionally offers image selection, network mode, extra volumes, and environment variables.

Before spawning, Flight Deck checks if the requested port is in use. If it is, a free port is automatically selected.

Spawned agents are managed under `fd-data/<agent-slug>/` with isolated config, workspace, sessions, and skills directories.

### Fleet Discovery & Agent Communication

Agents can discover and communicate with each other through two mechanisms:

**`flight_deck` tool** (always available) — queries `GET /fd/fleet` for a live list of all running agents. Actions:
- `list_agents` — returns all agents with name, kind, status, port, and description
- `consult` — synchronous Q&A with a peer agent. Streams the response with heartbeat monitoring (15s intervals) so the calling agent knows work is happening. Includes deduplication to prevent duplicate consultations to the same peer.
- `delegate` — fire-and-forget task delegation. The calling agent sends the task and immediately frees itself. The peer works independently and delivers results back as a notification when done. If the calling agent is busy when results arrive, they are queued and processed when the agent becomes free.

**`consult_peer` tool** (always available) — uses the peer list pushed by the Flight Deck frontend at WebSocket connect time. Supports forwarding tasks and approval-gated consultations.

The fleet endpoint returns Docker containers, process agents, and local agents in a unified list.

### Agent Desktop

The Agent Desktop combines Docker containers, process agents, and local agents into a unified view with two layout modes:

- **Grid layout** — traditional card grid
- **Free-form layout** — drag agent cards anywhere on the canvas, positions saved to localStorage

Agent cards cycle through three view modes with a single toggle button:

- **Expanded** — full card with description, forwarding task, approval settings, model/persona selectors, groups, files, logs, actions, and embedded chat
- **Compact** — name, port, status badge, activity indicator, groups, action buttons, and embedded chat on a smaller card
- **Icon** — single-row pill showing agent icon, name, activity status (Working.../Idle), and status badge. Toggle button appears on hover.

View mode per agent is persisted to localStorage. All three card types (Docker with violet accent, Process with emerald accent, Local) support all three modes.

### Chat

Click **Chat** on any online agent card to open a WebSocket chat session. The chat panel supports:

- Multiple concurrent sessions (tabbed interface)
- Markdown rendering with GFM tables
- Tool call visibility (expandable, last 3 per turn)
- Stop button for cancelling running tasks
- Auto-scroll and status indicators
- **Pin messages** — save important messages with tags for later reference
- **Copy to clipboard** — share snippets to the shared clipboard
- **File attachments** — attach files and clipboard content to messages
- **Resizable width** — drag the left edge to resize (320–900px, persisted)

Chat connections are proxied through the Flight Deck backend to avoid browser CORS restrictions.

### Director Panel

The Director panel (Cmd+D to toggle) provides a unified overview of all agents:

- **Agents tab** — all agents listed with status, current activity, last interaction time. Filter by status, sort by name/status/activity/type. Expandable rows show description, host, created time, and recent message previews.
- **Activity Feed tab** — real-time feed of agent events
- **Broadcast** — send a message to all running agents simultaneously
- **Quick actions** — Stop All, Restart All
- **Resizable** — drag the right edge (220–500px, persisted)

### Operations Dashboard

A dedicated view (Cmd+2) for monitoring agent usage and costs:

- **Summary cards** — total tokens, estimated cost, API calls, average latency, data transferred, cache hit rate
- **Token distribution** — visual bar showing input/output/cache breakdown
- **Per-agent usage table** — sortable by tokens, cost, calls, latency
- **Model breakdown table** — usage grouped by LLM model
- **Agent health grid** — status cards for all agents
- **Period filter** — last hour, today, yesterday, this week, this month, all time

Cost estimates use published pricing for Claude, GPT, and Gemini models.

### Agent Pipelines

Pipelines chain agents together so output from one automatically flows to the next. Located in the Workflows view (Cmd+3):

- **Visual pipeline builder** — create named pipelines with step-by-step agent chains
- **Step editor** — add agents, set optional prompt prefixes per step, reorder steps
- **Enable/disable** — toggle pipelines on/off without deleting
- **Contextual forwarding** — forwarded messages include pipeline name, source agent name, and instructions to process based on the receiving agent's playbooks, instructions, and persona
- **Auto-trigger** — when an agent in step N responds, step N+1 receives the output automatically

### Pinned Messages

Pin important chat messages for quick reference:

- **Pin from chat** — hover any message and click the pin icon
- **Tag system** — add tags to pins for filtering
- **Filter** — by tag, agent name, or content
- **Copy** — copy pin content to clipboard
- **Expand/collapse** — long messages shown as previews with expand toggle

### Shared Clipboard

Cross-agent clipboard for sharing text snippets:

- **Add entries** — manual entry or copy from chat messages
- **Pin entries** — mark important clipboard items
- **Send to agent** — forward clipboard content to any online agent
- **Edit** — modify clipboard entries inline

### Notification Center

Bell icon in the top bar with notification management:

- **Unread badge** — count of unread notifications
- **Type filters** — info, success, warning, error
- **Auto-notifications** — agent connect/disconnect, pipeline forwarding events
- **Mark read / clear** — individual or bulk actions

### File Browser & Transfer

Click **Files** on an agent card to browse its workspace files. The file browser includes:

- **File viewer** — click any file to view with syntax highlighting, image preview, or text rendering
- **Multi-select** — checkbox selection with select-all support
- **Agent-to-agent transfer** — select files and choose a destination agent
- **Transfer status** — per-file progress and completion summary

### Context Transfer

Click the forward arrow (↗) in the chat panel tab bar to send conversation context to another agent:

1. **Slider** — Select how many recent messages to include (0 to all)
2. **Preview** — Review the selected messages before sending
3. **Task** — Write what the receiving agent should do with the context
4. **Destination** — Choose which agent receives the context + task

Full message content is forwarded without truncation.

### Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| Cmd/Ctrl+1 | Agent Desktop |
| Cmd/Ctrl+2 | Operations |
| Cmd/Ctrl+3 | Workflows |
| Cmd/Ctrl+4 | Spawn Agent |
| Cmd/Ctrl+D | Toggle Director |
| Cmd/Ctrl+J | Toggle Chat Panel |
| Cmd/Ctrl+K | Toggle Shortcuts Help |
| Cmd/Ctrl+[ | Previous Chat Tab |
| Cmd/Ctrl+] | Next Chat Tab |
| Escape | Close Modals / Panels |

### Theme

Toggle between dark and light theme using the sun/moon button in the top bar. Theme preference is persisted to localStorage. Light mode includes full overrides for the zinc color palette, scrollbars, and markdown rendering.

### Development Mode

For frontend development with hot module replacement:

```bash
# Terminal 1: Flight Deck API backend
captain-claw-fd --dev

# Terminal 2: Vite dev server with HMR
cd flight-deck
npm install
npm run dev    # http://localhost:5173
```

The Vite dev server proxies `/fd` requests to the backend on port 25080.

### Building the Frontend

```bash
cd flight-deck
npm run build    # outputs to captain_claw/flight_deck/static/
```

The built files are included in the Python package as package data, so `pip install captain-claw` includes the pre-built frontend.

### Configuration

Flight Deck uses environment variables:

| Variable | Default | Description |
|---|---|---|
| `FD_DATA_DIR` | `./fd-data` | Directory for agent data (config, workspace, sessions) |
| `FD_AUTH_ENABLED` | `false` | Enable JWT-based multi-user authentication |
| `FD_JWT_SECRET` | (auto-generated) | Secret key for JWT token signing |
| `FD_DOCKER_SPAWN` | `true` | Allow Docker container spawning (can be disabled by admin) |

### Connection Settings

Each Captain Claw agent exposes connection info in its Settings page under "Connection Info":

- **WebSocket URL** — `ws://host:port/ws` (for chat)
- **HTTP URL** — `http://host:port` (for API)
- **Auth status** — Whether an auth token is configured

### macOS Docker Notes

On macOS, Docker's `--network host` mode is silently ignored. Flight Deck automatically detects macOS and:

- Switches to bridge networking with explicit port mapping
- Uses `host.docker.internal` for Ollama and other host-side services
- Sets `OLLAMA_BASE_URL` environment variable in containers

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

## Prompt Caching

Captain Claw supports Anthropic prompt caching to reduce token costs and latency on repeated interactions.

### How It Works

The system prompt is split into two parts using an internal marker:

1. **Static block** — tool definitions, policies, and instructions that remain constant across calls. This block is marked with `cache_control: ephemeral` so Anthropic caches it for 5 minutes.
2. **Dynamic block** — system info (timestamp, memory, disk) and per-session read directories that change between calls. This block is never cached.

A second cache breakpoint is placed on the last user or assistant message in conversation history, enabling cache hits during multi-turn tool loops where the same context is sent repeatedly.

### Provider Compatibility

| Provider | Caching behavior |
|---|---|
| **Anthropic** | Explicit prompt caching with two breakpoints. Cache reads are ~90% cheaper than fresh input tokens. |
| **OpenAI** | Automatic prefix caching (no configuration needed). The static-first layout naturally benefits OpenAI's caching. |
| **Gemini** | Context caching handled by Google's infrastructure. No impact. |
| **Ollama** | Local inference — no caching overhead. |

### Monitoring

Cache performance is visible in the **LLM Usage** dashboard (`/usage`). The summary cards show **Cache Read** and **Cache Created** token counts, and each individual call displays cache read tokens in the detail table.

---

## LLM Usage Dashboard

The LLM Usage page (`/usage`) provides detailed analytics for all LLM API calls made by the agent.

### Accessing

Navigate to `/usage` in the web UI, or click the **LLM Usage** card on the homepage.

### Features

- **Period filters** — Last Hour, Today, Yesterday, This Week, Last Week, This Month, Last Month, All Time
- **Provider filter** — dropdown to filter by LLM provider (OpenAI, Anthropic, Gemini, Ollama)
- **Model filter** — dropdown to filter by specific model (auto-filtered by selected provider)
- **BYOK filter** — dropdown to show All, BYOK Only, or Server Only calls
- **Summary cards** — Total Calls, Total Tokens, Prompt Tokens, Completion Tokens, Cache Read, Cache Created, Input Size, Output Size, Average Latency, Errors, BYOK Calls
- **Detail table** — per-call breakdown with timestamp, interaction ID, provider, model, token counts, cache read, input/output sizes, latency, status, and BYOK indicator (🔑)

Filters are applied server-side so summary totals accurately reflect the filtered subset.

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
