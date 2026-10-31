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
- [Presentation Control (Decks)](#presentation-control-decks)
- [File Editor (Flight Deck)](#file-editor-flight-deck)
- [Director Panel (Flight Deck)](#director-panel-flight-deck)
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
  - [botport](#botport-agent-config)
  - [scale](#scale)
  - [datastore](#datastore-1)
  - [deep_memory](#deep_memory)
  - [insights](#insights-1)
  - [nervous_system](#nervous_system)
  - [mcp_serve](#mcp_serve)
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
- [Plan Mode](#plan-mode)
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
- [Flight Deck](#flight-deck)
  - [Public file links](#public-file-links--new-in-081)
  - [Confinement on a multi-user deck](#confinement-on-a-multi-user-deck--new-in-084)
- [Kiosk (Simple Layout)](#kiosk-simple-layout)
  - [The agent menu](#the-agent-menu--new-in-081)
  - [The chat timeline](#the-chat-timeline-topics-and-other-channels--new-in-081)
  - [File library](#file-library--new-in-081)
  - [Data library](#data-library--new-in-081)
  - [Scheduled tasks](#scheduled-tasks--new-in-081)
  - [Onboarding](#onboarding--new-in-084)
  - [Taskboard](#taskboard--new-in-084)
  - [Your own AI keys](#your-own-ai-keys--new-in-084)
  - [Profile](#profile--new-in-084)
  - [Desktop app mode](#desktop-app-mode--new-in-081)
- [Deck Administration — NEW in 0.8.1](#deck-administration--new-in-081)
  - [Admin console](#admin-console)
  - [Daily plan limits and free resets](#daily-plan-limits-and-free-resets)
- [Remote Integrations](#remote-integrations)
  - [Telegram: Per-User Sessions](#telegram-per-user-sessions)
  - [Telegram: the Team Bot](#telegram-the-team-bot--new-in-081)
- [Glasses Bridge (Meta Ray-Ban Display)](#glasses-bridge-meta-ray-ban-display)
- [M5Stack Voice Devices (StickS3, StopWatch, PaperColor)](#m5stack-voice-devices-sticks3-stopwatch-papercolor)
- [WhatsApp Bridge](#whatsapp-bridge)
- [OpenAI-Compatible API Proxy](#openai-compatible-api-proxy)
- [MCP Server](#mcp-server)
- [Google OAuth, Drive, Calendar, and Gmail](#google-oauth-drive-calendar-and-gmail)
- [Microsoft 365 (Outlook, Calendar, OneDrive)](#microsoft-365-outlook-calendar-onedrive--new-in-080)
- [Send Mail](#send-mail)
- [Payments (bill pay and purchases) — NEW in 0.8.1](#payments-bill-pay-and-purchases--new-in-081)
- [Native Vision — NEW in 0.8.1](#native-vision--new-in-081)
- [Reasoning Effort — NEW in 0.8.1](#reasoning-effort--new-in-081)
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
| `captain-claw-web` | Agent web server (standalone entry point) |
| `captain-claw-agent` | Agent web server — **alias of `captain-claw-web`** (added in 0.5.3) |
| `captain-claw-fd` | Flight Deck multi-agent dashboard |
| `flight-deck` | Flight Deck — **alias of `captain-claw-fd`** (added in 0.5.3) |
| `captain-claw-orchestrate` | Headless orchestrator for cron jobs and scripting |
| `botport` | BotPort agent-to-agent routing hub |

> **Note:** `captain-claw` is unchanged (the interactive terminal agent). The
> `flight-deck` and `captain-claw-agent` aliases are friendlier names for the
> dashboard and the agent web server. On an **editable/pip install**, reinstall
> once (`pip install -e .` / `uv pip install -e .`) so pip generates the new
> entry-point launchers; the standalone build ships them as binaries.

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

Smart fallback: tries plain HTTP first, then transparently retries through a headless Playwright browser when the response looks like an unrendered SPA shell (small body or JS-bundle-only). Static pages stay fast (no Playwright cold-start tax); React/Vue/Angular SPAs return their actual rendered text. If Playwright isn't installed, returns the original HTTP body with a note suggesting `playwright install`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | yes | URL to fetch |
| `max_chars` | number | no | Max output chars (default: `tools.web_fetch.max_chars`, 200000 since 0.8.0). The configured value is a floor for a smaller model-passed value, and every value is bounded by 35% of the agent's context window |

### web_fetch_batch — NEW in 0.5.4

Fetch **multiple URLs in parallel** and return clean text per URL. Use this
instead of repeated `web_fetch` calls when you have several URLs to read — e.g.
the results of a `web_search`.

How it works:

- **Fast → deep self-correction per URL.** A plain HTTP fetch runs first
  (concurrency 8); only URLs that come back thin or JS-rendered escalate to a
  headless browser. The "thin" check compares extracted text to raw-HTML size,
  so genuinely short pages aren't needlessly escalated.
- **Shared browser.** Deep fetches run in one Chromium with an isolated context
  per URL (concurrency 3) — one launch, not N.
- **Never drops content.** If the browser is unavailable, the fast-HTTP content
  is surfaced anyway (matching `web_fetch(deep_fetch=false)`).
- **Self-installing.** On the first deep need with no browser binary, a one-time
  `playwright install chromium` runs in the background; the call returns fast
  content immediately and deep mode works on the next call.
- **Budgeted.** Caps at 10 URLs/call (extras are returned to the agent to fetch
  in a follow-up call), 100k chars per URL, 400k total (raised from 25k / 150k in 0.8.0). Tunable under
  `tools.web_fetch` (`batch_max_urls`, `batch_fast_concurrency`,
  `batch_deep_concurrency`, `batch_per_url_max_chars`, `batch_total_max_chars`,
  `batch_min_useful_chars`, `batch_fast_timeout`).

The result is one labelled section per URL (`[URL]/[Status]/[Mode: fast|deep]`)
plus a header summary (`Fetched 8/10 URLs (2 via deep browser, 0 failed)`).

#### Pictures for reports

`web_fetch` and `web_fetch_batch` keep the pictures a page shows that could
matter to the research — ones with a real description (a caption, alt text that
says what they show, a title) inside the page's content, not logos, icons,
avatars, share buttons, ads or "related" teasers — and end each page's result
with an `[Images …]` list: saved path, size, caption/alt, section, nearby text
and the original URL. `web_fetch` on an image URL saves that picture.

- **Where:** `saved/media/<session>/web-images/` (a `.json` sidecar per picture
  and an `index.md`); a Flight Deck run's workers share
  `vfs:<project>/sources/images/` so the reporter sees every teammate's.
- **Checked and fitted:** downloads go through the public-only client (a page
  can't point the agent at a private address), must really be an image, are
  refused when icon-sized or banner-shaped, and WebP/AVIF/TIFF or oversized
  photos are re-encoded to JPEG/PNG ≤ 1600 px — so python-docx can use them.
- **In reports:** `.md` links the original URL (the write/edit tools swap a
  saved path for it); `.html` gets every `<img>` embedded inline as base64 by
  the write/edit tools (saved copies, local image files, and remote URLs, which
  are fetched and saved on write), so the file stands alone; `.docx` uses
  `doc.add_picture(<saved path>)`. Reading an HTML report back shows its pictures
  as short `[inline image …]` placeholders, which edit/write put back — the
  base64 never floods the context, and the read/edit size limits ignore it.
- **Not for:** a shared agent's members (they get the original URLs only), and
  a code agent's repo files (`<img src="logo.png">` stays a file reference).
- **Tunable** under `tools.web_fetch`: `save_images` (default on),
  `images_per_page` (4), `images_per_batch` (16). The Flight Deck File library
  skips `web-images/` folders (no vision summary per picture).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `urls` | array | yes | URLs to fetch (processed in parallel) |
| `max_chars` | number | no | Per-URL char cap (overrides config default) |
| `deep_fetch` | boolean | no | Force headless mode for all URLs (default: auto) |

> **Honesty guard:** if the agent claims it searched/fetched the web but no web
> tool (`web_search`/`web_fetch`/`web_fetch_batch`/`web_get`) actually ran that
> turn, the runtime forces a corrective retry with `tool_choice=required`.
> Asking it to "refresh from the web, don't use memory" also skips the automatic
> memory-context injection for that turn.

### web_get

Fetch a URL and return raw HTML source. Use only when you need the actual HTML markup for scraping, DOM analysis, or CSS selector inspection. For normal page reading, use `web_fetch` instead.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | yes | URL to fetch |
| `max_chars` | number | no | Max output chars (default: 100000) |

### web_search

Search the web via Brave Search (default), Tavily, or You.com (NEW in 0.8.0) — pick with `tools.web_search.provider: brave | tavily | youcom`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Search query |
| `count` | number | no | Results to return (1-20, default: from config) |
| `offset` | number | no | Pagination offset |
| `country` | string | no | Regional ranking (e.g. `US`, `HR`) |
| `search_lang` | string | no | Language filter (e.g. `en`) |
| `freshness` | string | no | Time filter: `pd` (past day), `pw` (past week), `pm` (past month), `py` (past year) |
| `safesearch` | string | no | `off`, `moderate` (default), `strict` |

Brave requires `BRAVE_API_KEY` or `tools.web_search.api_key`; Tavily uses `tools.web_search.tavily_api_key`; You.com uses `tools.web_search.youcom_api_key` or `YDC_API_KEY` (web and news results merged, count clamped to 20).

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

### docx

Make and change Word documents properly, without scripts. Every change to a document is a Word **tracked change** with an author and a date, so whoever reviews it accepts or rejects each one in Word.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `create`, `read`, `edit`, `changes`, `comments`, `accept`, `reject` |
| `path` | string | yes | The .docx (for `create`, where to save it) |
| `content` | string | create | The document as Markdown |
| `edits` | array | edit | Changes applied in order, all or nothing (see below) |
| `output_path` | string | no | Save an edit/accept/reject to a new file instead |
| `author` | string | no | Name on tracked changes and comments (default: the agent's name) |
| `view` | string | no | read: `markup` (default), `final` or `original` |
| `offset` / `limit` | integer | no | read: paging by paragraph number |
| `ids` / `by_author` / `all` | | accept/reject | Which tracked changes |
| `template` | string | no | create: a .docx/.dotx whose styles, page setup, header and footer to use |
| `title`, `page_numbers`, `font`, `font_size`, `heading_font`, `heading_color`, `overwrite` | | no | create options |

**create** takes Markdown: `#` headings, paragraphs, `-` and `1.` lists (each numbered list restarts at 1), pipe tables (the header row repeats on every page), `>` quotes, fenced code, `![caption](saved picture path)`, `[TOC]`, `\pagebreak`, `**bold**`, `*italic*`, `~~strike~~` and `[links](url)`.

**read** lists numbered paragraphs with their style (`(H2)`, `(List Number)`, `(T1 r2c3)` for a table cell), tracked changes inline as `{+inserted+}` and `[-deleted-]`, `¶+` / `¶−` for an inserted paragraph / a removed paragraph break, and comments as `⟦anchor⟧cN`. The header says whether Track Changes is on and who made the changes so far.

**edit** — each item has an `op`:

| op | What it does |
|---|---|
| `replace` | `find` → `replace_with`. Only the words that differ are marked, so "12 months" → "24 months" shows `[-12-]{+24+}`. |
| `delete` | `find` (text), or `paragraph` [+ `to`] for whole paragraphs |
| `insert` | `text` `after` / `before` existing text |
| `insert_paragraphs` | Markdown `content` `after` / `before` paragraph N. A plain paragraph takes the formatting of the one it follows (after a heading, the heading's "next" style). |
| `format` | `bold`, `italic`, `underline`, `strike`, `highlight`, `color`, `size` on `find` or a `paragraph` (a tracked formatting change) |
| `set_style` | `paragraph` [+ `to`] → `style` (a tracked style change) |
| `comment` | a Word comment with `text` on `find` or a `paragraph` |

`find` is exact text inside one paragraph as it reads now; curly quotes and dashes match their plain versions. When it occurs more than once, add words, or pass `paragraph`, `occurrence` or `all`. Paragraph numbers are the ones `read` showed before the call. A bad edit fails the whole call and nothing is saved. Editing text another reviewer inserted marks it deleted inside their insertion; deleting your own earlier insertion just removes it, as Word does. The saved file has Track Changes switched on.

Where the result goes: a file in the agent's own `saved/` area (outside `downloads/`) or in a writable VFS folder is changed in place, with a backup. A user's upload (`saved/downloads/`), a read-only Drive folder or any other file is never changed: the redlined copy goes to `saved/output/<session>/` under the same name, where chat bridges send it from. If that copy already exists, the edit is refused until the agent edits the copy (or names a new `output_path`), so a redline never forks.

**accept / reject** resolve tracked changes like Word's buttons — all, by `ids` (from `changes`), or `by_author` — and only when the user asks.

On a shared agent, members use `docx` on their own files only. They read anything they may read. Their changes land in place only on files they created; anything else becomes their own redlined copy in their folder. Pictures named in the Markdown must be files they can read. Tracked changes and comments carry their name ("Ana (via Captain)"), not one they choose. No backup is kept, and results show `saved/…` or `vfs:…` paths, never host paths.

`write` and `edit` refuse `.docx` paths and point to this tool. The output was checked against Microsoft Word: Word reads every revision, and its Accept All / Reject All give the same text as `accept` / `reject`.

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

### audio

Speech-to-text and text-to-speech with Soniox. The tool appears on its own as soon as a Soniox key is usable, and stays hidden otherwise.

- **Under Flight Deck** the key is the agent owner's: their own in **Connections → Soniox**, else the team key an admin shares in **Admin → Integrations**, else the deck's `SONIOX_API_KEY`. Flight Deck holds the key and makes the Soniox calls; the agent never sees it.
- **Standalone**, the agent's own `SONIOX_API_KEY`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `transcribe`, `speak` or `voices` |
| `path` | string | transcribe | Audio or video file: `.ogg` `.opus` `.mp3` `.m4a` `.wav` `.flac` `.aac` `.amr` `.webm` `.mp4` `.mov` … (saved/, vfs or absolute) |
| `timestamps` | boolean | no | transcribe: add `[mm:ss]` segments (~15 s each, split at sentences) |
| `language_hints` | string[] | no | transcribe: ISO codes likely spoken, e.g. `["hr","en"]` (languages are detected anyway) |
| `text` | string | speak | Text to say, up to 40,000 characters (long text is spoken in 4,000-character pieces) |
| `voice` | string | no | speak: a Soniox voice (`action: voices` lists them; default `Adrian`) |
| `language` | string | no | speak: ISO code of the text's language (default `en`) |
| `output_path` | string | no | speak: file name, kept under `saved/media/<session-id>/` as `.mp3` |

`speak` saves an MP3 and reports it on a `Path:` line, so it is delivered like `pocket_tts` audio: WhatsApp and Telegram send it, the [M5Stack voice devices](#m5stack-voice-devices-sticks3-stopwatch-papercolor) play it, and the web chat shows a player. `transcribe` reports its input as `Source:`, so the user's own recording is never sent back. A transcript over 30,000 characters is also saved as a `.txt` next to the session's media.

When a chat user asks for a spoken reply and the agent made none, the chat bridge speaks the reply itself: with `audio` when Soniox is connected, else with `pocket_tts`.

**Voices** (all speak every supported language; the group is the accent): Maya, Daniel, Noah, Nina, Emma, Jack, Adrian, Claire, Grace, Owen, Mina, Kenji · British: Oliver, Arthur, Isla, Victoria · Spanish: Rafael, Mateo, Lucia, Sofia · Australian: Cooper, Mason, Ruby, Elise · Indian: Arjun, Rohan, Priya, Meera.

**The hotkey and the web mic** use Soniox realtime streaming (or OpenAI Whisper / Gemini as a fallback; `tools.screen_capture.stt_provider` picks one). Under Flight Deck they get a short-lived Soniox key for each session.

### telegram_inbox

Read the owner's Telegram, the way the mail tools read email. **Read-only**: there is no way to send, reply or mark anything read. The tool appears only while the owner keeps a Telegram inbox in Flight Deck, and only for the owner (never during a shared-agent member's turn).

Flight Deck keeps the inbox through the deck's team Telegram bot:

- **Private chats.** The user connects the team bot in Telegram → **Settings → Chat Automation** (Telegram 12.7 or newer; with Premium, **Settings → Telegram Business → Chatbots** works too) and picks which chats it may see. Telegram then hands the bot both sides of those chats, their edits and their deletions. Older history isn't available: the inbox starts when the connection does.
- **Groups.** The user adds the team bot to a group, or sends `/inbox` in a group it's already in (`/inbox off` stops). The bot tells the members once that it reads along, and never posts again. Anyone else who adds the bot sees it leave.

The user turns it on in **Connections → Telegram → Inbox** with explicit consent, after linking Telegram to the team bot. Turning it off, or disconnecting Telegram, deletes everything kept. Text and names are encrypted at rest with a key file kept apart from the database (`<FD home>/data_key`, or `FD_DATA_KEY_FILE`; needs the `cryptography` package). Messages are kept for 30 days by default (7, 30, 90 or 365). An admin can switch the inbox off for the deck in **Admin → Integrations**. For groups, the bot needs group privacy off in @BotFather or admin rights in the group; for private chats it needs **Secretary Mode** on in @BotFather (Admin → Integrations shows both).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `chats` (list them), `read` (one chat, oldest first), `recent` (newest across all chats), `search` (every word of `query`) |
| `chat` | string | read | A chat's id or (part of) its name; optional for `search` |
| `query` | string | search | Words that must all appear (text, sender, chat or attachment) |
| `since` / `until` | string | no | `24h`, `7d`, `90m`, a date or an ISO time |
| `limit` | integer | no | Default 50, max 200 |

Attachments show as a short description (`[photo]`, `[voice message (12s)]`, `[file: offer.pdf]`), not as files. The output reminds the model that the text was written by other people and is information, not instructions.

**What's new, every turn.** On the owner's own live turns the agent also gets one line in its turn context, e.g. *Telegram since your last turn: 3 new from "Ana Kovač", 12 new in "Obitelj" (group)*. The line has counts and chat names only, never message text. It is fetched once at the start of the turn. Each agent keeps its own place: the line moves it forward, and so does reading a chat (or `recent`) with the tool; a search doesn't. The first line an agent gets covers the last day. Messages the owner sent and deleted messages don't count. Scheduled, autonomous, peer and worker turns don't get the line and don't move the place, so a cron job never uses up "new" before the user has seen it. Neither do shared-agent members, public sessions or BotPort tasks.

### whatsapp_web

The owner's own WhatsApp, the way the mail tools work with email: read every chat, and send where the owner allows it. Nothing ever reacts, marks anything read or shows the owner as online. The tool appears only while the owner has WhatsApp linked in Flight Deck, and only for the owner (never during a shared-agent member's turn).

The user links their account in **Connections → WhatsApp Web**: Flight Deck becomes one of the account's linked devices, the way WhatsApp Web is (scan a QR code, or type a pairing code on the phone). Flight Deck's WhatsApp Web service then keeps the history the phone sends right after linking (in the first minutes; how far back is the phone's call) and every message from then on, in a store of its own on the deck. Message text, names, captions, media details and reactions are encrypted at rest with the same key as the Telegram inbox (`<FD home>/data_key`, or `FD_DATA_KEY_FILE`; needs the `cryptography` package); phone numbers stay readable so chats can be looked up. Everything is kept unless the user picks 7, 30, 90 or 365 days. **Disconnect** (or **Switch number**) removes the device from the phone's Linked devices and asks whether to keep the history. If the link ends any other way (removed on the phone, or WhatsApp bans the number), the history is kept. Kept history is listed on the card with a **Delete** button and is the only way it goes, apart from the retention chosen for that link; agents go on reading it along with the linked number's messages (a chat in both reads as one), but nothing is sent from it, and a chat only the old number had can't be whitelisted for the new one: it would be a first contact, which is what gets numbers banned.

It's an unofficial client: linking automation to a WhatsApp account goes against WhatsApp's terms, and WhatsApp bans some accounts that do it. So it's off until an admin allows it for the deck (the switch on the same card, or `FD_WHATSAPP_WEB=off` to rule it out), each link needs the user's consent, the deck needs sign-in on, and it's not in the desktop app. Install it with `pip install 'captain-claw[whatsapp-web]'`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `status` (linked number, how much is kept), `chats` (list them), `read` (one chat, oldest first), `recent` (newest across all chats), `search` (every word of `query`), `contact` (people by name or number), `attachment` (one message's file), `send` (one text message) |
| `chat` | string | read | A chat's id, phone number or (part of) its name; optional for `search` |
| `query` | string | search, contact | Words that must all appear (text, sender or chat); for `contact`, a name or a number |
| `since` / `until` | string | no | `24h`, `7d`, `90m`, a date or an ISO time |
| `limit` | integer | no | Default 50, max 200 |
| `message_id` | string | attachment | The message's id: the `#id` shown next to its file |
| `transcribe` | boolean | no | `attachment`: transcribe a voice note or audio (default true) |
| `text` | string | send | The message (plain text, up to 4096 characters) |
| `allow_repeat` | boolean | no | `send`: send the same text again although it went to that chat in the last 24 hours |

Voice notes, pictures and files show as a short description with the id that fetches them (`[voice note: 0:42 #3EB0…]`, `[image #…] caption`, `[document: offer.pdf #…]`). Edits, deletions and reactions are shown. The output reminds the model that the text was written by other people and is information, not instructions.

**Attachments.** `attachment` downloads the file now from WhatsApp's servers (the link has to be connected) and saves it under `saved/downloads/<session>/whatsapp-<chat>/`, with `Saved to:` and the tool that reads it (`image_vision`, `pdf_extract` …). A voice note or audio comes back transcribed when a Soniox key is usable (the user's own, the team's, or the deck's); otherwise the result points at `audio`. Files over 100 MB are refused. WhatsApp keeps media on its servers only for a while: an older file comes back as expired (the phone may still have it; asking the phone to send it again is a later, opt-in step).

**Sending.** Off until the owner turns it on in **Connections → WhatsApp Web → Sending**. Then it's automatic, like Gmail sending (no draft, no confirmation), but only to the chats the owner ticks on the whitelist, which lists only existing conversations: Flight Deck never starts one, and a number typed in never becomes a chat it may write to. At most 30 messages a day by default (1–200), 6 a minute to one chat, and the same text to the same chat only once a day unless the agent passes `allow_repeat`. Every message ends with the owner's WhatsApp signature ("— sent by my assistant" until they change or clear it). Each send rings the owner's bell. If WhatsApp doesn't confirm a send, it's recorded as *not confirmed* and the same text is refused until the owner checks the chat, so a retry can't double-send. A temporary ban, a link that isn't connected, or `FD_WHATSAPP_WEB_SEND=off` on the deck refuses every send. The tool's answer starts with "Sent to … on WhatsApp"; the reply guard turns down a reply that claims a WhatsApp send with no such answer behind it.

**Auto-replies.** In **Sending**, each whitelisted chat can also be marked *auto-reply*, with one of the owner's own agents picked to answer and, optionally, a few extra trigger words. In those chats the agent answers by itself when someone mentions it — its name or a trigger word as a word in the message (`@name` too, and with a Croatian-style case ending: "captainu", "LeChucka", "Stevu" for *stevo*), or a reply to one of its messages; the owner can call it in the same way, and a WhatsApp @mention of the owner's account stays the owner's. For 15 minutes after its last message it may also join in unaddressed when it has something useful to add; it steps back as soon as the owner writes in that chat. It never answers a message that ends like another assistant's signature, even one that mentions it. Messages that arrive close together are answered once, about 8 seconds after the last; at most 10 answers an hour per chat; a message more than 10 minutes old when it arrives (after the link was down) is left to the owner. The answer is written by the agent's own model, told which words call it, with nothing but the last 30 messages of that chat (each on one line, so no message can pose as another speaker) and, if it asks, one web search — never the owner's mail, files, calendar, memory, other chats or agents, since everyone in the chat reads it. It may decide to stay silent. It's sent through the same gate as any send (limits, signature, bell). A stopped agent doesn't answer. `FD_WHATSAPP_WEB_AUTOREPLY=off` (or `FD_WHATSAPP_WEB_SEND=off`) on the deck stops every auto-reply. Flight Deck's log has one `WhatsApp auto-reply` line per answer or silence (`outcome=sent`, `skip:silent`, `skip:not addressed`, `skip:format` …), never the text. The owner sees the whole decision in **Connections → WhatsApp Web → Auto-reply activity**: the message, the outcome, the model's one-line *why*, what it answered or why it stayed silent, and under *details* each model call (a thinking model's reasoning, model, tokens), the web search and what it found, and how long it took. *Messages not for it* also lists the messages that didn't call the agent. The newest 500 decisions are kept with the link, sealed like the messages, and go when WhatsApp is disconnected. In the whitelist, the auto-reply chats and the other whitelisted chats are listed first. An agent's messages are now kept in the chat history as they're sent, so agents read the chat as it is.

**What's new, every turn.** As with Telegram, the owner's own live turns get one line with counts and chat names only, e.g. *WhatsApp since your last turn: 2 new from "Ana", 5 new in "Obitelj" (group)*. Only messages that arrived live count — the history the phone sends after linking is old news. Each agent keeps its own place; reading a chat (or `recent`) moves it, a search doesn't, and scheduled, peer, worker, member, public and BotPort turns neither get the line nor move it.

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

Google Drive through the Drive API, with your Google connection (under Flight Deck, the agent owner's; see [Google OAuth, Drive, Calendar, and Gmail](#google-oauth-drive-calendar-and-gmail)). The tool is hidden from the agent until Google is connected.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list`, `search`, `read`, `download`, `info`, `upload`, `create`, `update`; in place (NEW in 0.8.0): `sheet_read`, `sheet_update`, `sheet_append`, `sheet_clear`, `doc_read`, `doc_replace_text`, `doc_append_text`, `doc_insert_text` |
| `file_id` | string | for file actions | Drive file ID, or the file's full Drive/Docs/Sheets/Slides URL |
| `folder_id` | string | no | Folder to list or upload into (default: `root`); a folder URL works too |
| `query` | string | for search | Search text (file name or content) |
| `output_path` | string | no | Local path `download` saves to (optional) |
| `local_path` | string | for upload | Local file to upload (or to update a file from) |
| `name` | string | for upload/create | File name |
| `content` | string | no | Text content for `create` / `update` |
| `mime_type` | string | no | For `create`, e.g. `application/vnd.google-apps.document` for a Google Doc |
| `max_results` | number | no | Results for `list` / `search` (default 20, max 100) |
| `order_by` | string | no | Sort for `list` / `search` (default `modifiedTime desc`) |

**Editing Google Sheets and Docs in place (NEW in 0.8.0).** The `sheet_*` actions work on an A1 `range` (`Sheet1!B2`, `'Q3 plan'!A1:C10`; `sheet_read` without a range lists the tabs with their first rows) and take `values` (rows; entered as if typed unless `value_input: raw`), `updates` (several ranges at once) and `render` (`values` or `formulas`). The `doc_*` actions take exact text copied from `doc_read`: `doc_replace_text` (`find`, `replace_with`), `doc_append_text` (`text`), `doc_insert_text` (`text` right after `after`), with `tab` for multi-tab Docs. These go through the Sheets v4 and Docs v1 APIs, so **enable the Google Sheets API and Google Docs API** in the deck's Google Cloud project. A whole-file `update` of a native Doc, Sheet or Slides file is refused unless `overwrite: true`; Office files on Drive keep the download → edit → `update` path. `list` / `search` page with `page_token`.

`read` returns a file's contents inline (Google Docs as markdown; PDF and Office files go through the matching extract tool). `download` saves the file locally and returns its absolute path (Google Docs, Sheets and Slides are exported), so a later `read` / `*_extract` call can work on it. Paste a Drive link straight into `file_id` — the ID is taken from the URL. Write actions need a write Drive scope.

### google_calendar

Google Calendar through the Calendar API, with the same Google connection.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list_events`, `search_events`, `get_event`, `create_event`, `update_event`, `delete_event`, `list_calendars` |
| `calendar_id` | string | no | Calendar ID (default: `primary`; `list_calendars` finds others) |
| `event_id` | string | for get/update/delete | Event ID |
| `query` | string | for search_events | Free-text search |
| `summary`, `description`, `location` | string | no | Event fields for `create_event` / `update_event` |
| `start`, `end` | string | for create_event | ISO 8601 datetime, or a date for all-day events (`end` is exclusive) |
| `timezone` | string | no | IANA timezone when `start`/`end` carry no offset |
| `attendees` | list[string] | no | Attendee email addresses |
| `reminders`, `recurrence` | list | no | Custom reminders; RRULE strings |
| `time_min`, `time_max` | string | no | Window for `list_events` (default: from now) |
| `max_results` | number | no | Default 10, max 100 |

### google_mail

Gmail through the Gmail API: `list_messages`, `search`, `read_message`, `get_thread`, `list_labels`, `create_draft`, `update_draft`, `list_drafts`, `send`, `send_draft`, and for organizing `archive`, `move`, `add_labels`, `remove_labels`, `create_label`. Reads and searches default to the INBOX Primary tab and are grouped by thread. `create_draft` is the default for anything email-writing; `send` / `send_draft` follow the opt-in policy in [Gmail sending](#gmail-sending-opt-in).

**Organizing.** `archive` takes emails out of the Inbox; `move` with `label` files them under that label and keeps their other labels, unlike Gmail's own Move to from a label view (`remove_labels` takes one off, such as the label they were filed under before; `label=INBOX` brings them back); `add_labels` / `remove_labels` take `labels` (names or IDs; INBOX, UNREAD, STARRED, IMPORTANT and the inbox categories work too, so removing UNREAD marks mail read); `create_label` makes a label, and `Parent/Child` nests it. Pass `thread_ids` for whole conversations or `message_ids` for single messages, up to 100 per call; the answer lists what changed, what didn't, and how to undo it. A label that doesn't exist is refused, never created on the fly, and nothing is ever trashed or deleted. Agents organize mail only when asked: an automated turn (a cron job, a delegated task) may do it only when its own instructions ask to organize email, and Autonomous Work only through an action you approve. These actions need the `gmail.modify` scope — under Flight Deck an admin ticks **Gmail (read + modify)** in Connections → Google (every user then reconnects); standalone it is in the default `google_oauth.scopes`, so a config that lists its own scopes must add `https://www.googleapis.com/auth/gmail.modify` and reconnect Google. Without it the tool says so and changes nothing.

**Attachments.** `read_message` and `get_thread` list each email's attachments with a number. The list includes pictures pasted into the body (named `inline-1.png` and so on) and unnamed files such as a scanner's PDF (`attachment-1.pdf`). A named text part that is really the message body (MMS gateways call it `text_0.txt`) stays the body unless the sender marked it as an attachment. `get_attachment` takes the `message_id` and `attachment`: the number, the file name, or `all` for up to 10. A number that isn't on the list is an error, never a guess at a name. Each email's files go in their own folder, `saved/downloads/<session>/mail-<message id>/`, so two emails' `invoice.pdf` never overwrite each other. Two files with the same name in one email become `image.png` and `image (2).png`. A name missing its reader's extension gets one (`Contract v2.1` becomes `Contract v2.1.pdf`), and a long name is shortened without losing it. With `all`, a file that fails to download is reported and the rest are still saved. The result names the tool that reads each file: `pdf_extract`, `docx_extract`, `xlsx_extract`, `pptx_extract`, `image_vision` for pictures (png, jpg, gif, webp, bmp), or `read` for anything else. In eco mode, mentioning an attachment (`attachment`, `prilog`, `privitak`) adds those readers to the tool list. This needs only the `gmail.readonly` scope and counts as a read, so it also works in automated turns. A shared agent's member never overwrites a saved file someone else made. Attachments are inputs, so they are never sent back to WhatsApp or Telegram automatically.

### microsoft_mail — NEW in 0.8.0

Outlook / Microsoft 365 mail through Microsoft Graph, with the agent **owner's** Microsoft connection (see [Microsoft 365](#microsoft-365-outlook-calendar-onedrive--new-in-080)). Hidden until the owner has connected; owner-only (a shared agent's members never get it). It **never sends**: drafts land in the user's Outlook Drafts folder for them to send.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list_messages`, `search`, `read_message`, `get_thread`, `list_folders`, `create_draft`, `update_draft`, `list_drafts`, `get_attachment`, `archive`, `move`, `create_folder`, `add_categories`, `remove_categories` |
| `folder` | string | no | Inbox (default), Sent, Drafts, Archive, Deleted, Junk, a folder name, a `Parent/Child` path or an ID; `search` also takes `all` |
| `query` | string | for search | Outlook search syntax (`from:alice subject:report hasattachments:true received>=2026-01-01`); for `list_drafts`, `to:<address>` or subject words |
| `message_id` / `message_ids` | string / list | per action | One email, or up to 100 for organizing actions |
| `thread_id` / `thread_ids` | string / list | per action | A conversation (`get_thread`), or whole conversations for organizing actions |
| `to`, `cc`, `bcc`, `subject`, `body`, `html_body` | string | for drafts | Draft fields (`update_draft` replaces the body) |
| `reply_to_message_id`, `reply_all` | string, bool | no | Draft a reply inside the original conversation (quoted, `RE:`) |
| `draft_id` | string | for update_draft | The draft to revise |
| `attachments`, `remove_attachments` | list | no | Files to attach (at most 20, 25 MB together; `saved/` or `vfs:` paths work) / attachments to drop |
| `attachment` | string | for get_attachment | Number from `read_message`, a file name, or `all` |
| `categories` | list | for categories | Outlook category names |
| `allow_repeat` | bool | no | Only when the user explicitly wants another copy of an email already drafted or sent |
| `unread_only`, `focused_only`, `include_body`, `group_by_thread`, `max_results` | | no | Listing filters (default 10 results, max 50; grouped by conversation) |

`create_draft` refuses a repeat — the same recipient and subject already in Drafts, or in Sent Items within `tools.microsoft_mail.repeat_check_days` (default 14; 0 turns the Sent Items check off) — and says so instead of creating a duplicate. Drafts follow the same "write only when asked" rule as `google_mail`, and every draft ends with the user's [agentic signature](#agentic-signature--new-in-080). Organizing (archive, move, folders, categories) needs `Mail.ReadWrite` and follows the same gating as Gmail organizing.

### microsoft_calendar — NEW in 0.8.0

Outlook / Microsoft 365 Calendar: `list_events`, `search_events`, `get_event`, `create_event`, `update_event`, `delete_event`, `respond_event` (accept / decline / tentative, with an optional `comment`), `list_calendars`. Parameters mirror `google_calendar` (`calendar_id`, `event_id`, `summary`, `description`, `location`, `start`, `end`, `timezone`, `attendees`, `reminder_minutes`, `recurrence` as one RRULE, `time_min`, `time_max`, `max_results`), plus `online_meeting` (a Teams link, work/school accounts).

**Outlook emails people itself.** Creating or changing an event with attendees sends invitations or updates, deleting a meeting you organize sends cancellations, and answering an invitation can notify the organizer. Every such call needs `notify_attendees: true`, which the agent may pass only when the user asked to invite / update / cancel / reply in that turn (or an approved action does). Events without attendees never notify anyone.

### microsoft_onedrive — NEW in 0.8.0

OneDrive (and OneDrive for Business / SharePoint files shared with the user): `list`, `search`, `read` (PDF/DOCX/XLSX/PPTX extracted, other Office formats converted, text as text), `info`, `download` (to `saved/downloads/<session>/` by default; a `.pdf` `output_path` on an Office file saves a PDF conversion), `upload` (large files use upload sessions), `create`, `create_folder`, `update`. Address a file by `file_id` (plus `drive_id` when shown), by `path` inside OneDrive (`/Documents/report.docx`), or by pasting a OneDrive/SharePoint share URL into `file_id`. `upload` and `create` never overwrite — a clashing name gets a new one; to change a file, download it, edit the copy, then `update` with `local_path`.

> **Retired: `gws`.** The Google Workspace CLI wrapper (`gws`) is gone. Agents use `google_drive`, `google_calendar` and `google_mail` instead; a `gws` entry left in an old `tools.enabled` list or a stored archetype is dropped when the config loads or the agent is spawned, and a `tools.gws` block in an old `config.yaml` is ignored.

### jev

Fast, calibrated judgments about text with [Jev](https://docs.typesafe.ai) (TypeSafe AI's "System One" model). Jev never writes text: you give it a `state` (the content) and named, typed questions, and each answer comes back with probabilities. One call can ask many questions about the same state.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | no | `ask` (default) or `models` (list the models the key can use) |
| `state` | string | for ask | The content the questions are about: text, or a JSON object/array as text (sent structured). About 30k tokens at most |
| `state_file` | string | no | Instead of `state`: a text file to use as the state (absolute, or relative to the workspace). Giving both is an error |
| `questions` | list[object] | for ask | Each `{name, type, instructions, …}` — see below |
| `model` | string | no | Jev model (default `tools.jev.model`, normally `jev-latest`) |

Question types:

- **`noul`** — yes/no. Answer: p(yes), shown as yes (≥ 0.65), no (≤ 0.35) or uncertain. Optional `yes_means` / `no_means` say what counts as each.
- **`choice`** — one of 2–255 `options`, each `{name, description}`. Answer: the chosen option, per-option probabilities and a confidence.
- **`score`** — a level on an ordered rubric of 2–10 `levels` (descriptions, lowest first). Answer: the expected level, the nearest level's text and a confidence.

Good uses: triage and routing, classification, relevance filtering, checking a claim against a source, rating against a rubric. Write literal, specific questions (Jev reads them literally; English works best) and give options and levels real descriptions. Jev is not for math, counting, date arithmetic or writing text.

**Where the key lives.** Under Flight Deck each user adds their own key in **Connections → Jev** (from [console.typesafe.ai/keys](https://console.typesafe.ai/keys)); the key stays in Flight Deck. Since 0.8.1 an admin can also share one **team key** in **Admin → Integrations** for everyone without their own (a user's own key wins; the status says `own` or `shared`). The agent asks Flight Deck (`/fd/jev/agent/*`), which calls Jev with the agent **owner's** key, else the team key. Standalone, the agent calls Jev itself with `tools.jev.api_key`, or the `JEV_API_KEY` / `TYPESAFE_API_KEY` environment variable. The tool is hidden from the agent until a key is usable. It is owner-only: a shared agent's members never get it.

```yaml
tools:
  jev:
    api_key: ""                     # standalone only; or JEV_API_KEY / TYPESAFE_API_KEY
    base_url: "https://api.typesafe.ai"
    model: "jev-latest"
    timeout: 30                     # seconds per Jev request (429/529/5xx are retried)
    check_replies: true             # flag garbled chat replies and suggest /new
    check_progress: true            # when a turn runs out of steps, let Jev decide whether to extend
    review_blocks: true             # when a guard refuses a repeated call, let Jev pick the next step
    find_images: true               # when a reply names no picture file, let Jev pick the one it hands over
```

**Jev first.** While a key is usable, the system prompt tells the agent to ask Jev whenever a task involves judging content — triage, routing, relevance, claim-vs-source checks, tone/urgency/risk, rubric scores, duplicates, best-of-N, a yes/no check before acting — with one call per piece of content carrying all its questions, and to say when Jev drove a decision. In eco mode `jev` is a core tool, so its schema is always sent (it stays hidden until a key works).

**Garbled-reply check.** After a person's chat turn (web, WhatsApp, Telegram, the glasses HUD), Jev is asked one yes/no question: did the reply come out as unusable garbage (a looping line, token soup, raw tool-call JSON or control tokens, text cut off mid-word)? At p ≥ 0.8 the surface gets one extra notice — *"That reply came out garbled — this session may be overloaded. Send /new to start a fresh session"*. The notice is never written into the session, never spoken, and skipped for automated, public, shared-member and worker turns; no key, a timeout or a Jev error means no notice. Turn it off with `tools.jev.check_replies: false`.

**Progress check.** When a turn uses up its step budget, the loop's own rule extends it only if the last few steps made new tool calls *and* a plan or requirement says work remains — so a plain request stops even while the agent is plainly working. Before stopping, Jev reads the request and the turn's recent steps (narration, each tool call and a slice of its result) and answers one yes/no question: is the agent still making real progress? At p ≥ 0.6 the budget grows by one more block of steps, never past the turn's hard ceiling, and Jev is asked again when that runs out; otherwise the turn stops and ships what it has, as before. The `completion_gate` entry shows `judge=jev` and `p_progressing`. The stagnation detector (six steps with nothing new) still runs, so a yes can't keep a looping agent alive. Skipped for shared-member turns; no key, a timeout (15 s) or a Jev error means the usual stop. Turn it off with `tools.jev.check_progress: false`.

**Block review.** The tool loop refuses a call that repeats work already done in the turn: a duplicate call, or a full rewrite of a file written with nothing read since. The refusal is usually right, but one fixed follow-up message doesn't fit every case. Jev reads the request, the turn's recent steps and the refused call, and picks one next step:
- **verify:** read the file once before deciding;
- **continue:** go on with the rest of the request (share, send, the next item);
- **finish:** give the final answer;
- **ask_user:** ask the person.

The pick's fixed instruction replaces the guard's usual message when p ≥ 0.55. For a read-only call (read, glob, search, fetch, or a `list`/`get`/`search` action), Jev may also judge that the repeat is legitimate because something changed since. At p ≥ 0.8 the call then runs, once per call per turn. A call that sends, creates or changes something is never released. Scale-loop items and Google/Microsoft send or mailbox changes keep their own messages.

At most 3 reviews run per turn, and the `completion_gate` entry shows `step=jev_block_review`, the choice and p. Skipped for shared-member turns. No key, a timeout (8 s) or a Jev error means the usual message. Turn it off with `tools.jev.review_blocks: false`.

**Pictures in the reply.** Pictures made by image_gen, visualize, termux or payments reach the chat and the channel from the tool's `Path:` line. A reply can also hand over a picture made another way: a plotting script wrote `saved/charts/q3.png`, and the reply says "here's the chart: q3.png". Such pictures are now shown in the web chat after the reply and sent to the channel the message came from (WhatsApp, Telegram, a device). Each picture reference in the reply is matched to the pictures the turn wrote under `saved/`. A reference can be a path, `![](…)` or a bare file name. When the reply names none of them but the turn did write pictures, Jev is asked once:
- whether the reply hands the person a picture (p ≥ 0.7);
- if the turn wrote more than one, which file it is (p ≥ 0.5 on the pick).

Only deliverables qualify: files inside `saved/` and written this turn, never from scripts/tools/skills/downloads, never a hidden folder and never a file the person attached. A turn that ran no tools is not checked. Skipped for shared-member turns; no key, a timeout (12 s) or a Jev error means no pick. Turn the Jev step off with `tools.jev.find_images: false`.

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

Requires `botport.enabled: true` and a valid BotPort server URL in the agent's config (a legacy, per-agent link — agents on a Flight Deck reach other decks through the `flight_deck` tool instead). See [BotPort](#botport) for setup details.

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

### self_check

Self-assessment and status. The agent tests, live and in parallel, everything it is connected to, allowed to use and expected to work, and reports the server it runs on. Always available.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `sections` | array | no | Only some of `server`, `agent`, `models`, `connections`, `tools` (default: all) |
| `deep` | boolean | no | Also run a real web search query and start the browser (slower) |
| `save` | boolean | no | Also save the report as Markdown under `saved/output/<session>/` |

What it checks:

- **Server** — host and OS, CPU count and load (warns above 1.5× the CPUs), free memory, free disk on the workspace volume, the agent process (PID, uptime, memory, threads, CPU), Captain version and commit, and the clock against Flight Deck's.
- **Agent** — who it is and how it runs (under Flight Deck or standalone), that the workspace is writable, and that the session store and datastore open.
- **Models** — the model in use and every allowed model, each sent a one-word prompt (16 tokens at most). Image, video and audio models are listed, not called; a local LiteRT model isn't loaded just for the test.
- **Connections** — internet, Flight Deck, each Google service (Gmail, Calendar, Drive) and each Microsoft 365 one (account, Outlook, Calendar, OneDrive) with its own request, so a missing permission shows on the service it blocks; Jev, Soniox, the Telegram inbox, WhatsApp, payments, deep memory (a one-result search), every MCP server allowed for the agent (its tool list), outgoing mail (an SMTP login, nothing sent), web search, BotPort, the agent's own Telegram bot and the browser.
- **Tools** — how many tools the model can use, and for each hidden one why (e.g. "Google not connected").

Every result is `ok`, `warn`, `fail` (should work and doesn't, with the fix), `off` (not set up — not a failure) or `info`. The report starts with what needs attention. Everything is read-only, and no key or token is printed.

### reasoning_effort

Read or change how hard the agent's model thinks — the same [reasoning effort](#reasoning-effort--new-in-081) setting a person changes in Flight Deck. The agent calls it when the user asks it to think harder or faster (or names a level), and on its own when a task needs much more or much less thinking than it has. Always available.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `get` (the current level, the levels the model offers, the settings) or `set` |
| `level` | string | for `set` | `auto` (send nothing: the model's own default), `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or `on`. A level the model doesn't offer goes to the nearest one it does |
| `scope` | string | no | `setting` (default): change the setting — the rest of this turn and later messages. `turn`: only the rest of this turn |
| `source` | string | no | Which setting to change when it isn't the one for where the message came from: `web` (the chat box), `default`, or a source (`whatsapp`, `telegram`, `app`, `glasses`, `device:sticks3`, `scheduled`, …) |

- **A web chat message** — the chat box's picker moves to the new level in every Flight Deck chat open on the agent (a `reasoning_effort` frame), so later messages carry it as if the user had picked it. With no chat open the picker can't move; the tool says so and the level still holds for the rest of the turn.
- **Any other source** — the agent writes that source's row of `~/.captain-claw/reasoning_effort.json` (the row **Settings → Reasoning effort** shows: `app:android` → `app`, `glasses:meta-display` → `glasses`, `device:sticks3` stays itself).
- **At once** — the rest of the turn's conversation calls use the new level; `scope: "turn"` ends with the turn.
- **From any channel** — the user can ask from the web chat, WhatsApp, Telegram, the app, the glasses or a voice device. A message about reasoning effort or thinking level (English or Croatian, or a follow-up like "set it to off") adds a line to the turn's note telling the model to call the tool from there, not to send the user to Flight Deck.
- **Owner only** — public-link visitors, shared-agent members and agent links can neither read nor change it.

### activity

The agent reads its own tool calls back from Flight Deck's tool-call register: every call it made, when, what it was about, how it ended and how long it took. It uses it to answer "what did you do overnight?" or "why did the morning task fail?" from what actually ran, to check that an action really happened before saying so, and to see which tools keep failing. Always available; needs Flight Deck (the register lives there). Read-only.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | no | `list` (default): the calls, newest first. `summary`: counts per tool, the same failure 3+ times, the same call 3+ times within 30 minutes in one conversation, guard refusals, channels |
| `since` / `until` | string | no | `30m`, `24h`, `7d`, `today`, `yesterday` or an ISO date/time (local when it has no zone). Default: 24 hours for `list`, 7 days for `summary` (at most 30 days back) |
| `tool`, `outcome` | string | no | Only one tool; only calls that ended `ok`, `error`, `timeout`, `blocked`, `aborted`, `interrupted` or `running` |
| `search` | string | no | Text in the call's parameters (an address, a file name) |
| `this_conversation` | boolean | no | Only calls made in this conversation |
| `limit`, `before_id` | integer | no | Page size (20, at most 50) and the id to page back from |

What the agent gets back is less than people see in **Activity** and the console's **Audit** page:

- **Its own calls, in its owner's turns.** Never another agent's. Calls made in a shared-agent member's turn are private to that member, and so are calls made while the owner's turn held members' private conversations (the register marks them; their parameters may restate what the agent read). The agent is told only how many there were.
- **What identifies a call, not its content.** Up to three of `action`, `path`, host and path of a `url`, `query`, `command`, `to`, `subject`, `table`, … (80 characters each). Never the start of a written file or an email body. Errors keep the first line of their reason.
- **No reasons for safety refusals.** A call the safety guard, the blast-radius check or a blocked shell pattern refused reads "Blocked by the safety guard"; the reason stays in the register for people, and the agent's search doesn't look in reasons.

**Reflections use it too.** Each self-reflection gets the patterns among the agent's calls since the previous reflection — tools that failed three or more times, the same failure three or more times, the same call repeated — from the same calls the tool reads, with the rule never to write an instruction from one failure or about getting past the guard. When nothing repeats, it is told nothing calls for a change.

Routes (the agent's own auth token, identified by Flight Deck): `GET /fd/tool-audit/agent/mine` and `GET /fd/tool-audit/agent/digest`.

### knowledge — NEW in 0.8.4

Searches and reads the documents the owner gave the agent (its `knowledge/` folder), with no model. It reads the files itself, so it works before the index has caught up. Shows on every Flight Deck agent; shared-agent members never get it.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `search`, `section`, `outline`, `list` or `add` |
| `query` | string | search | What to find. Hybrid search; exact codes rank first, and a quoted phrase is required |
| `doc` | string | no | A document's title or file name: search only it, or read its section or outline |
| `section` | string | section | A section's title, with its parents (`Chapter › Summary`) when titles repeat |
| `page` | integer | no | A long section's page (default 1) |
| `limit` | integer | no | Passages for `search` (default 5) |
| `path` | string | add | The file to keep: PDF, Word, Excel, PowerPoint, CSV, Markdown, text or HTML |
| `text`, `name` | string | add | Text to keep instead of a file (what a picture or scan says), under this file name |
| `title`, `note`, `use` | string | no | add: its title, what it is, and how to use it (`reference`, `rules`, `procedures`, `templates` or `auto`) |

A search hit is a passage with its document, section and lines. `add` makes the file searchable at once but doesn't read it for rules or skills; the owner does that in [Way of working](#onboarding--new-in-084). Before an owner's turn, the sections the message bears on are named in a context note, with no model call. Install `captain-claw[pdf]` so PDFs keep their headings and tables.

### visualize — NEW in 0.8.4

HTML in, the page and a PNG of it out, for charts, diagrams and dashboards instead of plotting scripts that often leave nothing to show. Always on.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `html` | string | yes | Body content (with `<style>`/`<script>` as needed) or a full document |
| `title` | string | no | The page title and the file name |
| `libraries` | array | no | Pinned libraries to load: `chartjs`, `echarts`, `plotly`, `d3`, `mermaid`, `leaflet` |
| `width`, `height` | integer | no | Page width (320–2400) and a fixed height; leave height out to capture the whole page |
| `selector` | string | no | One element to picture alone (`#chart`) |
| `scale` | number | no | PNG pixel density, 1–3 (default 2) |
| `wait_ms` | integer | no | Extra wait after the page settles, up to 10000; or set `window.vizReady = true` when done |
| `output_path` | string | no | An `.html` path (default `saved/showcase/<session>/`); the PNG goes next to it |

The PNG reaches the chat, Telegram, WhatsApp and the Files menu, and goes into a `.docx` or `.md` as `![caption](path)`. Script errors, failed loads, empty canvases and a blank picture come back for the agent to fix. The page never touches the host: it's served from an address that doesn't exist, every request goes through the public-only client, and WebSockets are closed.

### basna

Read and start **Basna** runs — a parallel ensemble of specialist agents that researches/analyses a task and merges their work into one answer. Always available when running under Flight Deck; scoped to the calling agent's owner (resolved by the agent's auth token).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `start`, `list`, `get`, `agents`, `output`, `truth`, `analysis`, `files`, `get_file` |
| `task` | string | for `start` | The task to run — a clear, self-contained statement of what to research/produce |
| `title` | string | no | Optional title for a `start` run; auto-generated from the task if blank |
| `max_agents` | number | no | Cap on agents for `start` (1–10, default 6) |
| `session_id` | string | for read actions | Target session (all actions except `list`) |
| `archetype_id` | string | for `output` | Which agent's full output to return |
| `name` | string | for `get_file` | File to fetch from the session |
| `query` / `status` / `limit` | — | no | Filters for `list` |

**Actions:**
- **`start`** — launches a **new autonomous Basna run** on `task`. Flight Deck auto-titles, routes the minimal team, executes the ensemble **server-side** (no UI), and **reports completion back to this agent** so it can relay the result to the user — on whatever channel the request came from (web / WhatsApp / glasses / API). Fire-and-forget: returns immediately. Uses the owner's saved Library tiers (falling back to built-in defaults); capped at 2 concurrent agent-started runs per owner.
- **`list`** — the owner's sessions (optional `query` substring, `status`, `limit`): id, title, domain, status, confidence, agent/file counts.
- **`get`** — full detail of one session: routed archetypes, compiled truth, confidence, files.
- **`agents`** — per-agent runs for a session: role, tier, weight, success, latency, and tool activity.
- **`output`** — one agent's full output (`session_id` + `archetype_id`).
- **`truth`** / **`analysis`** — the compiled answer + confidence / the cross-agent agreement, differences, and blind spots.
- **`files`** / **`get_file`** — list a session's files / fetch one (text inline, otherwise saved into the agent's workspace).

When the user explicitly asks to "run/execute a Basna…", the request is relayed straight to `start` deterministically (the agent does not do the research itself).

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
| Soniox (recommended) | Flight Deck → Connections → Soniox (standalone: `SONIOX_API_KEY`) | Realtime streaming — audio goes from mic to Soniox WebSocket, no files saved |
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

### app_runner

Scaffold, edit, and operate **Code Apps** — agent-authored mini-apps with a Python `backend.py` and a single-file `frontend.html`, each running as a managed Flight Deck subprocess behind a per-slug Unix domain socket. New in 0.4.26.

| Action | What it does |
|---|---|
| `scaffold` | Create a new app from a slug + brief. Writes `backend.py`, `frontend.html`, `manifest.json`. Reserved for new apps and full rewrites — never use on an existing app (it wipes the source). |
| `read_source` | Load `backend.py` + `frontend.html` of an existing app so you can edit them in place. |
| `edit_file` | Targeted edit of one file in an app's source. Auto-restarts the subprocess after the write so the next request hits the new code. |
| `restart` | Stop + spawn fresh. Use after `edit_file` if the auto-restart didn't fire, or to recover from a wedged subprocess. |
| `logs` | Tail the app's stdout/stderr ring buffer + the on-disk `logs/{stdout,stderr}.log`. Used by the self-repair loop after a 5xx from `proxy`. |
| `proxy` | Smoke-test by forwarding an HTTP request to the app's subprocess. Returns status/headers/body. The agent calls this after `scaffold` to verify the app actually serves. |
| `query_app` | Read another app's data via its published `data_api`. Used from chat to answer "how many notes do I have?" without scaffolding a new notes app. |
| `list` | Enumerate available apps + which ones publish a `data_api`. |

**Self-repair loop.** After `scaffold`, the agent is expected to call `proxy` to smoke-test. On 5xx it calls `logs`, reads the traceback, fixes `backend.py`, then `restart`. The system prompt's app-authoring policy walks the agent through this.

**Cross-app data sharing — `data_api` block.** An app's `manifest.json` declares the read-only endpoints siblings can hit:

```json
{
  "data_api": {
    "contacts": {
      "path": "/contacts",
      "method": "GET",
      "description": "List all contacts as {id, name, email, phone}"
    }
  }
}
```

Without a `data_api` block, sibling calls return 403. Write endpoints are **not publishable** in v1 — read-only by design.

**Consuming a sibling's data — `app_sdk.sibling()`.** Inside an app's `backend.py`:

```python
from captain_claw.app_sdk import sibling, SiblingError

try:
    data = await sibling("contacts").get_json("/contacts")
except SiblingError:
    data = []  # sibling missing or unpublished
```

Auth is handled per-call; the caller just needs the target slug + the path the sibling publishes. `post_json` and raw `request` variants are also available.

**Where apps live.** Each app is a directory under the Flight Deck data root: `<slug>/backend.py`, `<slug>/frontend.html`, `<slug>/manifest.json`, plus a `logs/` subdir written by the runtime. The subprocess runs as its own process group (`SIGTERM → 3s grace → SIGKILL` on shutdown). One subprocess per slug — horizontal scale-out for a single app is not supported in v1.

**App vs visualization rule.** Apps have *writes* (the user expects records to be there next session). Visualizations are read-only output. "Notes app", "todo app", "tracker" → use `app_runner scaffold`. "Chart of these numbers", "dashboard of X", "report" → write a self-contained HTML file via the visualization policy instead. Don't use `localStorage` / `IndexedDB` in code-apps — use the app's `backend.py` + the datastore.

### whatsapp_send_file

Send a file the agent saved (its `saved/` workspace) to a WhatsApp chat through the Cloud API: JPEG and 8-bit PNG up to 5 MB arrive as **photos**, H.264/AAC MP4 up to 16 MB as **videos**, MP3, AAC M4A, AMR or Opus audio up to 16 MB as **audio** — the file's bytes are checked, not just its extension. Everything else, anything larger, or a file WhatsApp wouldn't play (an HEVC clip, a Vorbis .ogg, ALAC) goes as a **document**, which always arrives; a photo/video/audio Meta refuses for its format is sent again as a document, and `send_as: "document"` sends the original file. By default it sends into the chat the turn answers (on an automated turn without one — a cron job, a muted scheduled job — it needs an explicit `to`). Requires `WHATSAPP_ACCESS_TOKEN` + `WHATSAPP_PHONE_NUMBER_ID` (Flight Deck passes them to process agents; Docker agents need them in their own `.env`). WhatsApp only allows free-form messages within 24h of the recipient's last message.

**Delivered at the end of a WhatsApp turn.** When a turn arrived over WhatsApp — or is a Flight Deck scheduled job delivering to a WhatsApp number that may receive pushes now (on the allowlist, not muted with `/mute`) — the agent sends what it made for the user into that chat once the reply is out: pictures it generated (`image_gen`) or took with the phone (`termux`), audio it spoke (`audio` speak or `pocket_tts`), and **any file this session wrote during the turn that the reply names by its whole file name** ("here's report.docx"; under `saved/<category>/<session>/`, including `tmp/` where bare writes land — not `scripts/`, `tools/`, `skills/` or `downloads/`, so a Drive file it only read isn't sent back). Nothing it already sent with this tool, at most `whatsapp.max_media_per_turn` (6) files; when something can't be sent or the cap cuts the list, the chat gets one short line saying so. The agent is told this on those turns (and told plainly when it has no WhatsApp credentials). Agent-local cron jobs don't deliver files (they can't see `/mute`); the agent can still send with this tool. Turn it off with `whatsapp.auto_send_media: false`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | no | `send` (default) or `list` (show saved files) |
| `path` | string | no | Path of the saved file to send (e.g. `showcase/<session>/report.docx`) |
| `filename` | string | no | Filename (fuzzy, case-insensitive; newest match) when `path` is not given |
| `latest` | boolean | no | Send the most recently saved file |
| `to` | string | no | Recipient number (digits only, no `+`). Omit to reply into the current WhatsApp chat |
| `caption` | string | no | Caption shown under the file or photo (not for audio) |
| `send_as` | string | no | `auto` (default) or `document` — the original file, never re-compressed by WhatsApp |

Office/PDF use Meta-accepted MIME types; text formats (`.md/.html/.py/.csv/…`) are sent as `text/plain`. Allow-list enforced (`WHATSAPP_ALLOWED_WAIDS`).

### intentions

Record and manage **intentions** — future actions under consideration. `origin='user'` saves a note-to-self; `origin='agent'` records an action the agent intends to take (low-risk → announced, normal/high → asks permission). Approved repeatable intentions materialise into Flight Deck scheduler jobs; declines are remembered so they aren't re-proposed. A proactive generator can propose intentions automatically (opt-in via `intentions.auto_generate`).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `create`, `list`, `update`, `snooze`, `cancel`, `done`, `resolve` |
| `origin` | string | no | `user` (note-to-self) or `agent` (proactive action) |
| `title` | string | for create | Short summary |
| `why` | string | no | Motivation (recommended) |
| `risk` | string | no | `low` (announced) / `normal` / `high` (asks permission) |
| `repeat` | string | no | Recurrence, e.g. `weekly mon 09:00` |
| `action_prompt` | string | no | Prompt to run when an agent intention executes/recurs |
| `verdict` | string | for resolve | User's answer — `yes`/`no`/`later`/`stop` (freeform ok) |
| `decision_id` | string | no | Which pending decision to resolve (omit for the most recent) |
| `intention_id` | string | for update/snooze/cancel/done | Target intention id |

Decisions can also be resolved from the **Intentions** panel on each agent card in Flight Deck (Approve / Decline / Later / Stop).

### public_link — NEW in 0.8.1

Publish one of the agent's files as a public link, the way a file's Share button does (see [Public file links](#public-file-links--new-in-081)): Flight Deck takes the file from the calling agent, keeps a snapshot, and returns `<deck>/s/<token>`, which anyone holding it can open without signing in. The link belongs to the agent's owner and shows in their Share dialog. Needs Flight Deck.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `share` (publish the file, or refresh its copy under the same URL), `unshare`, `list` (this agent's links with their view counts) |
| `path` | string | share / unshare | The file — a path as the agent knows it, or just its file name |
| `expires_days` | integer | no | share: `0` (never), `1`, `7` or `30`. Left out, an existing link keeps its expiry and a new one never expires |
| `token` | string | no | unshare: the link's token from `list`, instead of `path` |

A person's turn may share. An automated turn (cron, scheduled task, flow, autonomy, workers, beings) may share only when its own job text asks for a link. Secret-looking files, databases and data folders are refused before anything is sent. Shared-agent members can't reach the tool. The result tells the model to give the user the URL exactly as returned, and flags a link that only works locally (the deck knows no public address). For a whole folder or website, use **Hosting** instead.

### sessions — NEW in 0.8.1

The user's conversation topics (sessions), and the way to move between them by asking — "go back to the Rome trip", "what were we talking about before?", "start a new topic" — on a surface with no Topics menu (WhatsApp, a voice device, the glasses). Always on.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `list` (recent topics: number, title, turns, when, which is current), `switch`, `new` |
| `topic` | string | switch | The topic's number from the list (`#2`), words from its title, or its id |
| `name` | string | no | new: an optional name for the topic |

A switch or a new topic takes effect after the reply and is announced like `/session switch` and `/new`, so the web timeline draws its divider, the HUD reopens, and chat bridges re-send their rules. The agent never switches on its own initiative. Refused for a shared agent's members, and for Telegram users other than the owner (each has one session there; `/new` starts a fresh one). The tool is named `sessions` because `topics` is the conversation-topic memory.

### watch_background — NEW in 0.8.1

Set the picture behind every screen of the owner's M5Stack StopWatch, or clear it back to black. Flight Deck crops the picture to the round screen, keeps it per owner and device kind, and sends it to the watch now or on its next connect (`POST /fd/device/agent/background`). See [M5Stack Voice Devices](#m5stack-voice-devices-sticks3-stopwatch-papercolor).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | yes | `set` or `clear` |
| `path` | string | set | The picture: one the user sent, or one the agent saved or made (JPEG, PNG, WebP, GIF, BMP, HEIC or TIFF, up to 20 MB) |

Needs Flight Deck. Not available to shared-agent members.

### payments — NEW in 0.8.1

Prepare a bill for the user to pay in their own bank app, and (later phases) buy from a shop with a one-time card the user approves. Captain never moves money and never takes card numbers. The tool is hidden unless the deck runs with `FD_PAYMENTS=1`. Actions: `prepare`, `handoff`, `mark`, `list`, `get`, `remind`, `recurring`, `purchase`, `checkout_pay`, `checkout_check`. See [Payments](#payments-bill-pay-and-purchases--new-in-081).

---

## Presentation Control (Decks)

Drive an agent-built **HTML slide deck** live — from the Flight Deck big screen,
the glasses view, a phone, or WhatsApp. All surfaces subscribe to one in-memory
**channel** (default `deck`), so any control moves every surface in lock-step.
No PDF/PPT export.

**Cast a deck:** in the Flight Deck **file list**, tap the **⧉ deck-view** button
on an HTML file. The green panel holds a live **channel** field (the open URL
updates as you type). Selecting a file never auto-loads a deck — casting is
always an explicit tap. The deck's own on-page nav (arrows/dots) is hidden in
deck view.

**Phone remote:** open `/deck/remote?c=<channel>` — large prev/next, first/last,
and a **go-to-slide** number box (clamped to the live slide count). Arrow / Page
keys also work.

**Glasses:** the present bar shows `⧉ ◀ ▶`; the arrows advance the big screen
**and** the glasses' own view together.

**WhatsApp** (allowlisted chat): send any of —

| Message | Action |
|---|---|
| `next slide` (`/next`) | advance one |
| `previous slide` (`/prev`) | back one |
| `first slide` / `last slide` | jump to ends |
| `go to slide N` | jump to slide *N* (1-based) |

`go to slide N` also accepts `goto N`, `slide N`, `jump to slide N`, `slide #N`,
and tolerates a trailing `.`/`!`. Each command replies with the current
position, e.g. `→ Slide 3 / 20`. Point the remote at a different deck with
`/slide on <channel>`.

> Built for a hands-free live talk: control the deck from Meta Ray-Ban glasses
> over WhatsApp while the deck renders on the projector.

---

## File Editor (Flight Deck)

Agent text files are editable in place, not preview-only. In the file list, the
**✎ Edit** button (next to Download) opens a code editor:

- **Syntax highlighting** — markdown, HTML, CSS, JS/TS, JSON, Python, Bash,
  YAML, and the **Flow DSL** (same editor powers the Flow builder's code view).
- **Line numbers** with a scroll-synced gutter.
- **Find** — `⌘/Ctrl+F`: case-insensitive, match count, `Enter` /
  `Shift+Enter` to step through matches, `Esc` to close.
- **Cursor memory** — reopening a file returns the caret to the **start of the
  row** you left off on.
- **Save** — `⌘/Ctrl+S` (or the Save button); editable groups are markdown,
  code, data, text, and html.

Row actions in the file list (pin · view · ⧉ deck · ✎ edit · download) sit on
their own line under the file path.

---

## Director Panel (Flight Deck)

The Director is the left panel of the Agent Desktop. Besides the agent list,
activity feed, and groups, it has a **Connections** tab (added in 0.5.5).

### Connections tab

Polls the external dependencies Flight Deck relies on and shows a traffic-light
per connector, refreshed every **10 minutes** (with a manual **Check** button):

| Light | Meaning |
|-------|---------|
| 🟢 Healthy | Connected **and** a read-only test call succeeded |
| 🟡 Degraded | Connected but the read-only test failed |
| 🔴 Offline | Not connected / not configured |
| ⚪ Disabled | MCP server is turned off |

- **Google** — a live read-only probe (`/fd/google/probe`): refreshes the access
  token if needed and calls userinfo, so it reflects real reachability, not just
  whether a token is stored.
- **MCP servers** — each enabled server is tested with a read-only `tools/list`
  (the existing per-server test). 🟡 vs 🔴 distinguishes a server that was
  initialized before but now fails from one that never connected.

### Activity panel token meters

The chat's collapsible **Activity** panel header shows running token usage for
the turn (added in 0.5.5):

```
Activity · 16 tools · 17.0k↑ 340↓ · 5.1k cached
```

Input (↑) / output (↓) / cache-read tokens — **live** while the turn runs,
**frozen** onto each activity group when it ends so older turns keep their own
counts. Cache figures appear only when the provider reports them.

### Activity-timing context (agent-side)

Every turn the agent's system prompt carries a compact timing block in addition
to the current clock (added in 0.5.5): last user message, last reply, last
scheduled/cron run, session start, the **next scheduled run ETA** for the
session, a part-of-day/weekday hint, and a conversation-cadence count. Real user
messages are tracked separately from cron/scheduler-driven runs. It renders as a
one-liner in eco/micro mode and is omitted in nano.

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
  vision: "auto"                  # NEW in 0.8.1: auto | on | off — does this model see images inline?
                                  # (each allowed model takes its own `vision` too; see Native Vision)
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

**Vision (NEW in 0.8.1):** with `vision` on (or `auto` for a known multimodal family), the turn's pictures go to the model itself as image parts; a `model_type: vision` model is then only the fallback. See [Native Vision](#native-vision--new-in-081).

**Reasoning effort (NEW in 0.8.1):** set per message from the chat box and per source in `~/.captain-claw/reasoning_effort.json`, not in this block. See [Reasoning Effort](#reasoning-effort--new-in-081).

**Modes:** an agent can run in `standard` (default), `eco`, or `nano` mode.

- `eco` — biases the agent toward cheaper/faster cloud models (e.g. `gpt-5-mini`, `claude-haiku`, `gemini-flash`) and trims a few expensive-but-optional context blocks. Toggled per-agent in Flight Deck via the **Eco Mode** chip.
- `nano` — restricted-tool runtime tuned for tiny local models (3B–8B params). Ships a compressed system prompt and exposes only `shell`, `write`, `read`, `edit`, `glob`, `datastore`, and `insights`. Recommended models: `qwen2.5:3b`, `llama3.2:3b`, `phi3:mini`, or larger Qwen3 quantized to fit on a laptop. Toggled per-agent in Flight Deck via the **Nano Mode** chip.

**Remote GPU via vast.ai:** point an Ollama-typed model at a vast.ai instance URL (`base_url: "https://<instance>.vast.ai:11434"`) and Captain Claw will auto-wake the instance on first request and auto-sleep it after the configured idle period.

```yaml
model:
  provider: "ollama"
  model: "qwen3.6:35b"
  base_url: "https://my-instance.vast.ai:11434"
```

### context

```yaml
context:
  max_tokens: 160000              # total context window budget
  compaction_threshold: 0.8       # trigger compaction at 80% usage
  compaction_ratio: 0.4           # keep 40% recent messages after compaction
  timezone: "Europe/Zagreb"       # user's IANA timezone for the agent's "now"
                                  # (empty → CLAW_TZ/TZ env → UTC-only). The
                                  # injected date/time is always anchored on UTC
                                  # and labelled; set this so day/date reasoning
                                  # (today/tomorrow/this week) uses local time.
  compaction_summary: digest      # NEW in 0.8.0: "digest" (topic-grouped exchanges, no LLM call)
                                  # or "model" (the pre-0.8.0 model-written summary)
  notes_allocator: trim           # NEW in 0.8.0: "trim" drops whole low-priority notes;
                                  # "capped" cuts each note to its source's share first
  tool_result_max_chars: 0        # NEW in 0.8.0: cap on one tool result in the session (0 = off)
  turn_governor: false            # NEW in 0.8.0: in-turn context governor for long single-turn
                                  # workers (env CLAW_CTX_GOVERNOR); Vatra turns it on for workers
  inline_images:                  # NEW in 0.8.1: pictures sent inline to a model that sees images
    max_images_per_call: 4        # the user's attachments first, then the newest tool pictures
    max_tool_images: 2            # a browser loop keeps only its newest screenshots
    max_pixels: 1280              # longest edge after downscaling
    jpeg_quality: 85
    max_share: 0.3                # images take at most this share of the window; a turn whose
                                  # picture doesn't fit is described by the vision fallback instead
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
    - audio
    - send_mail
    - google_drive
    - google_calendar
    - google_mail
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
    max_chars: 200000               # single page (was 100000 before 0.8.0)
    batch_per_url_max_chars: 100000
    batch_total_max_chars: 400000
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

**Always-on tools (0.8.1).** Some tools are added to every agent whatever its `enabled` list, archetype or spawn list says. Since 0.8.1 that includes the everyday working set — `shell`, `glob`, `web_fetch` (with `web_get` and `web_fetch_batch`), `web_search`, `pdf_extract`, `docx_extract`, `xlsx_extract`, `pptx_extract`, `insights`, `contacts`, `apis` and `todo` — and now `docx` (Word documents with tracked changes), `self_check` (connection tests and server status), `reasoning_effort` (how hard the agent thinks) and `activity` (its own tool calls, from Flight Deck's register), beside `read`, `write`, `edit`, `datastore`, `image_vision`, `video_vision`, `cv`, `sessions`, `history`, `vfs`, `typesense`, `intentions`, `whatsapp_send_file` and the rest. Before, most archetypes lacked some of them (24 of 31 had no `insights`, so "remember that…" had nowhere to go). Every archetype and the archetype grid now list them too. Account tools (`google_*`, `microsoft_*`, `jev`, `audio`, `telegram_inbox`, `whatsapp_web`) are always registered but stay hidden until their account or key is usable. `require_confirmation` and the shell policy still apply. The `stt` tool is gone; speech is the `audio` tool.

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
  shared_bot: true                # NEW in 0.8.1: under Flight Deck, an agent with no bot of its own
                                  # answers its owner on the deck's team bot once an admin approved it
  groups: true                    # NEW in 0.8.1: may be added to Telegram groups (web-only group turns);
                                  # false: it leaves groups it's added to
  group_turns_per_hour: 30        # NEW in 0.8.1: answers per group per hour (0 = no cap)
  owner_in_main: true             # NEW in 0.8.1: the owner's private chat joins the main conversation
                                  # (lane A); false: it keeps a session of its own
```

See [Telegram: the Team Bot](#telegram-the-team-bot--new-in-081) for the team bot, groups and the owner's chat.

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

<a id="botport-agent-config"></a>
### botport

The agent's own, legacy, direct link to a BotPort hub. Agents running on a
Flight Deck do not need it: the deck connects to BotPort and carries their
requests (see [BotPort](#botport)).

```yaml
botport:
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
  embedding_dims: 0                       # 0 = auto-detect from the embedding provider (recommended)
  auto_embed: true                        # compute embeddings on index
  min_score: 0.12                         # relevance floor (0..1) for prompt injection
```

`embedding_dims` defaults to **0 (auto-detect)** since 0.8.0. Before that it defaulted to 1536 while the default embedder emits 256, so every vector was silently rejected and deep memory ran keyword-only; a hand-set value that disagrees with the provider still turns hybrid search off. Scores are a clean 0..1 (the better of keyword coverage and cosine similarity); raise `min_score` if you switch to a transformer embedder (ollama/litellm).

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
- `owner_id`, `content_hash` — tenant and dedupe fields (added in 0.8.0; existing collections are patched in place)

### insights

Persistent knowledge base auto-extracted from conversations. The agent identifies facts, contacts, decisions, deadlines, and other durable knowledge during conversations.

```yaml
insights:
  enabled: true                           # enable insights extraction
  auto_extract: true                      # auto-extract after agent turns
  inject_in_context: true                 # inject relevant insights into system prompt
  max_items_in_prompt: 8                  # max insights shown in context
  context_mode: relevant                  # relevant: core + what this turn's words match (every turn); importance: fixed top list
  core_items_in_prompt: 3                 # most important insights always shown in relevant mode
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

### mcp_serve

Settings for Captain Claw's MCP server mode (`captain-claw-mcp`). See [MCP Server](#mcp-server) for full usage details.

```yaml
mcp_serve:
  enabled: true                               # enable MCP server
  log_path: "~/.captain-claw/mcp_server.log"  # server log file path
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

**Under Flight Deck (NEW in 0.8.0)** deep memory is a Flight Deck capability: an admin sets the Typesense connection once in **Connections → Typesense (Deep Memory)** (stored in Flight Deck's database, not `config.yaml`; the key is masked on read), and agents hold **no** Typesense key — they reach the archive through Flight Deck, which scopes every search and write to the agent's **owner**. The `typesense` tool is always enabled; with no connection configured it says so. A **Deep Memory** page in Flight Deck browses the archive. Documents indexed before tenancy have no owner and stay invisible to scoped search until the user presses **Claim**. VFS folders can be indexed too, opt-in per project with a `.vfs-index.json` at the user root; a hit's reference is `vfs:<project>/<path>`, which the `read` tool accepts.

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

## Plan Mode

Plan mode (introduced in 0.4.25, hardened in 0.4.26) layers a reviewable planner, per-step verifier, and bounded auto-revision loop on top of the orchestrator. A single user request becomes an ordered DAG of executable steps, runs through the existing worker pool, and is gated through measurable acceptance criteria — with automatic re-attempts when verification fails.

### Slash commands

- `/plan <request>` — generate a plan and persist it as a workflow JSON in `workspace/workflows/`. Renders a Markdown preview in the chat panel. Does **not** execute.
- `/plan-execute` — run the most recently generated plan. Streams `plan_*` events to the UI as steps progress / verify / get revised. Skips the orchestrator's final synthesis pass (the plan card is the deliverable surface).
- `/planning on` — enable auto-routing. Every plain-text message you send afterwards is auto-routed through `/plan` + `/plan-execute` — no slash commands needed.
- `/planning off` — back to free-form chat. The slash commands above stay available.
- `/planning` (no arg) — report current state.

### Pipeline stages

1. **`PlanGenerator`** asks the configured agent model for a 3-8 step plan with concrete `description` and one-sentence measurable `acceptance_criteria` per step. The system prompt requires deliverable-producing steps (drafting, writing, summarising, reporting, generating documents) to declare an explicit output filename — preferably `saved/tmp/<slug>.md` — in both the description and the acceptance criteria, so the artefact is never lost inside the run transcript.
2. **`PlanExecutor`** loads the plan into the orchestrator and calls `execute(skip_synthesize=True)`. The DAG runner already serialises sequential plans because each step declares the previous one as `depends_on`. Run-time failures (worker error / timeout) stop the plan without invoking the reviser — those need orchestrator-level recovery.
3. **`PlanVerifier`** walks completed steps in topological order:
   - If `output_schema` is set, fast deterministic JSON-schema validation runs first.
   - Otherwise (or after schema passes), an LLM judge evaluates the step's output against `acceptance_criteria` and returns `{passed, notes}`.
   - The first verification failure stops the walk and surfaces the verifier's notes.
4. **`PlanReviser`** is invoked when verification fails. It sees the failed step's title/description/criteria, the truncated output (8 KB) and the verifier's notes, and returns a sharper `revised_description` (and optionally tighter `revised_acceptance_criteria`). The executor swaps the description in place, increments `revision_count`, and resets the failed step plus all transitive dependents to PENDING so only what's affected re-runs.
5. **Bounded loop.** `DEFAULT_MAX_REVISIONS = 2` (one initial run + up to two retry cycles). Module constant on `captain_claw.plan_mode`.

### Step kinds

- **`atomic`** (default) — runs in the main session, sequentially relative to its `depends_on`.
- **`orchestrate`** — the planner expects fan-out. `PlanExecutor._expand_orchestrate_steps` calls the orchestrator's decomposer to turn the parent into N parallel sub-tasks plus an atomic join step that synthesises their outputs. Sub-task IDs are namespaced (`<parent_id>__<sub_id>`) so two orchestrate steps never collide.
- **`verify`** / **`revise`** — reserved for future explicit verifier/reviser steps inside a plan.

### Plan card in Flight Deck

The chat panel renders a live plan card inline:

- Per-step status chips (`pending` / `running` / `completed` / `failed`) and verification chips (`verified` / `revised` / `verification failed`).
- Live progress driven by the `plan_execution_started`, `plan_step_verified`, `plan_step_revised`, `plan_execution_verified`, `plan_execution_completed` events.
- Revision badge with rationale + previous failure notes on hover.
- **Persistent across refresh** — plan state, planning toggle, and card-collapsed flag are persisted to `localStorage` keyed by `containerId` (`fd.plan.${id}`).
- Collapsible long plans (one-line summary chip, click to expand).

### Persistence

Generated plans are saved as ordinary workflow JSON under `workspace/workflows/`. They round-trip through `/api/orchestrator/workflows/load`, so any tool that consumes Captain Claw workflows (CLI runner, cron dispatcher, etc.) can re-execute a plan unchanged.

The per-run Markdown transcript (`<workflow>-output-<timestamp>.md`) is auto-saved to the same directory by `_save_run_output`, and `workspace/workflows/` is on the file-preview allow-list so it opens directly from the FD file panel.

### WebSocket resilience

Plan runs are long (executor + verifier + reviser cycles), so the WebSocket plumbing is hardened end-to-end:

- Agent ⇄ Flight Deck proxy upstream uses `ping_interval=20` / `ping_timeout=10` keepalives.
- The browser client auto-reconnects with exponential backoff (500 ms → 15 s cap), tracks `_shouldReconnect` so explicit disconnects don't bounce, and surfaces a transient banner via the `_reconnecting` event.
- Server-side `_send` and the agent's WebSocket receive loop both wrap `(ConnectionResetError, ConnectionError, ClientConnectionResetError)` so a browser refresh mid-step doesn't fill the agent log with tracebacks.

### Worker iteration budget

`SessionOrchestrator._estimate_task_iterations` is research-aware: phrases like "reputable / credible / authoritative / reliable / primary sources", "research / gather / collect … sources", "at least N sources", `web_search` mentioned as a tool, and `each <thing>` patterns add weight. The hard ceiling is **40** (raised from 25 in 0.4.25) so multi-source research steps stop exhausting their budget mid-`gather_sources`.

---

## BotPort

BotPort is the hub that connects **Flight Decks**: an agent on one deck can
consult or delegate to an agent on another deck, wherever the two run. Each
Flight Deck dials out to the hub and keeps one WebSocket open (so decks behind
NAT work); the receiving deck proxies each request to its own agent and streams
the answer back. **The step-by-step guide — Raspberry Pi + tmux hub, decks on
VPSs, TLS / Tailscale / tunnels, troubleshooting — is
[docs/botport-deployment.md](docs/botport-deployment.md).**

### Architecture

```
Agent ──flight_deck tool──▶ Flight Deck A ══ws(s)══▶ BotPort hub ◀══ws(s)══ Flight Deck B ──▶ Agent
          "Researcher@pi"     /fd/consult-peer         routes Agent@deck        proxies to its agent
```

- **Agents** only ever talk to their own Flight Deck. Remote agents show up in
  `flight_deck list_agents` as **`Agent@deck`** and are consulted/delegated by
  that name, like local peers.
- **Flight Deck** (Admin console → BotPort, or `FD_BOTPORT_*` environment)
  connects to the hub, advertises the agents an admin chose to **expose**, and
  runs requests from other decks on its agents — in a separate lane per remote
  deck, never in the owner's main chat, with email writing and the
  network-reaching tools (`flight_deck`, `consult_peer`, `botport`) denied.
- **BotPort** authenticates each deck (key/secret; the key binds the deck's
  name), routes addressed requests, relays progress, holds results for a
  requester that dropped, and gives a deck that dropped mid-task time to come
  back.

### Setup

**1. Run the hub** (any machine the decks can reach — the guide uses a Raspberry Pi):

```bash
python3 -m venv ~/botport-venv && ~/botport-venv/bin/pip install ./botport
mkdir -p ~/.botport && cp botport/config.example.yaml ~/.botport/config.yaml && chmod 600 ~/.botport/config.yaml
~/botport-venv/bin/botport check -c ~/.botport/config.yaml
scripts/botport-tmux.sh start                      # or: botport -c ~/.botport/config.yaml
```

BotPort listens on port `23180` by default and refuses to start (rather than
move to another port) if that port is taken. `/healthz` answers without
credentials.

**2. Add each Flight Deck on the hub:** its dashboard → **Decks → Add Flight
Deck** — a display name, the deck's network name and a description. The hub
makes the deck's key and shows a **join code** once; the deck can connect at
once, without a hub restart. (`botport gen-key <name>` still prints an
`auth.keys` entry for the config file instead.)

**3. Connect it:** on that deck, Admin console → **BotPort** → paste the join
code into **Join code** and **Save** (or fill in the hub URL, deck name, key
and secret by hand), then pick the exposed agents. Or set
`FD_BOTPORT_URL`, `FD_BOTPORT_DECK_NAME`, `FD_BOTPORT_KEY`, `FD_BOTPORT_SECRET`,
`FD_BOTPORT_EXPOSE` (see the guide for the full list).

**4. Use it:** "ask Researcher@pi to …" — the agent calls `flight_deck`
(`consult` for a quick answer with live progress, `delegate` for work whose
result comes back as a message).

**Direct agent connections (legacy).** A Captain Claw agent can still connect
to the hub on its own with the `botport:` section of its config:

```yaml
botport:
  enabled: true
  url: "ws://localhost:23180/ws"
  instance_name: "my-agent"
  key: "my-key"                     # if auth is enabled on the hub
  secret: "my-secret"
```

The rest of this section describes that agent-level protocol (concerns,
routing by expertise, the `botport` tool and swarms). Decks can address such an
agent by its instance name.

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

BotPort server configuration — the annotated reference is
[botport/config.example.yaml](botport/config.example.yaml). The essentials:

```yaml
server:
  host: "0.0.0.0"
  port: 23180
  admin_token: ""                     # dashboard + REST API token (else: local requests only, when auth is on)
  behind_proxy: false                 # a TLS reverse proxy / tunnel on this host
  tls_cert: ""                        # or serve wss:// directly
  tls_key: ""

auth:
  enabled: true
  keys: []                            # decks added on the dashboard need none; botport gen-key <name> prints an entry

concerns:
  idle_timeout_seconds: 600           # no progress for this long → the request fails (both sides told)
  reconnect_grace_seconds: 90         # a deck that drops mid-task may come back
  result_retention_seconds: 900       # results wait this long for a requester that dropped

logging:
  level: "INFO"
  file: ""                            # e.g. ~/.botport/botport.log (rotated)

llm:
  enabled: false                      # LLM routing of unaddressed requests (pip install 'botport[llm]')
  model: "ollama/llama3.2"            # litellm provider/model format
```

Every value can be overridden by environment: `BOTPORT_<SECTION>__<KEY>`
(e.g. `BOTPORT_AUTH__ENABLED=true`). `botport check` prints the effective
configuration and its warnings; `botport --version` the version.

### BotPort Dashboard

BotPort includes a web dashboard (enabled by default) that shows:
- Connected instances with their capabilities
- Active and historical concerns
- Routing decisions and concern lifecycle events

Access at `http://<hub>:23180/` — it asks once for `server.admin_token`. It
lists Flight Decks with the agents they expose, legacy agents with their
personas, and concerns. Its **Decks** page adds Flight Decks (key + join code),
gives each a display name and description that every deck sees, and rotates
or removes keys, all without a restart. `/api/status` gives the same as JSON (decks, load,
queued results, counters); `/ws/dashboard` streams live events.

### Capability Advertisement

Each connected Captain Claw instance advertises:
- **Personas** — agent personality and user profiles with expertise tags
- **Tools** — list of enabled tools
- **Models** — available LLM models
- **Capacity** — max concurrent concerns

Control what is advertised via `botport.advertise_personas`, `advertise_tools`, and `advertise_models`.

### Environment Variables

| Variable | Description |
|---|---|
| `CLAW_BOTPORT__ENABLED` | Enable the agent's direct (legacy) BotPort connection |
| `CLAW_BOTPORT__URL` | BotPort WebSocket URL |
| `CLAW_BOTPORT__INSTANCE_NAME` | Instance name |
| `CLAW_BOTPORT__KEY` | Auth key |
| `CLAW_BOTPORT__SECRET` | Auth secret |

Flight Deck's link is configured with `FD_BOTPORT_*` — see
[docs/botport-deployment.md](docs/botport-deployment.md#6-connect-each-flight-deck).

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

**Google Drive tab** (shown when Google is connected with a Drive scope — `GET /api/gdrive-status`):
- Browse My Drive and Shared Drives
- Breadcrumb navigation through folder hierarchy
- Add/remove Google Drive folders for agent access (the agent works with them through the `google_drive` tool)

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
flight-deck                        # default: http://0.0.0.0:25080  (alias of captain-claw-fd)
captain-claw-fd                    # default: http://0.0.0.0:25080
flight-deck --port 8080            # custom port
flight-deck --dev                  # API-only mode (use with Vite dev server)
python -m captain_claw.flight_deck # alternative
```

`flight-deck` (added in 0.5.3) is a friendly alias for `captain-claw-fd` — same
flags, same server. On an editable install, run `pip install -e .` once so the
launcher is generated.

Flight Deck serves both the React frontend and the FastAPI backend from a single process. No separate build step is needed for production — the built frontend is bundled with the Python package.

### Features

| Feature | Description |
|---|---|
| Docker container management | Spawn, stop, restart, remove, rebuild, clone Captain Claw containers with full config (provider, model, tools, platforms) |
| Process agent management | Spawn pip-based agents as local subprocesses — no Docker required. Full lifecycle (start/stop/restart/clone/remove), auto-restart on FD startup, clean shutdown |
| Fleet discovery | `GET /fd/fleet` returns all running agents (Docker, process, local). The `flight_deck` tool gives every agent live fleet awareness |
| Agent-to-agent communication | `consult_peer` and `flight_deck` tools let agents discover, consult, and delegate tasks to each other. Synchronous (consult) and fire-and-forget (delegate) modes |
| Multi-user authentication | JWT-based auth with registration, login, refresh tokens, and role-based access (admin/user). First user becomes admin |
| Admin dashboard | Admin page (Usage, Plans, Settings, Integrations), a separate Users page (users, invitations, waitlist), plan tier assignment (free/premium/enterprise; "pro" was renamed premium in 0.8.1), per-user quota overrides, usage analytics, and system configuration. Since 0.8.1 admins start in the [Admin console](#admin-console) |
| Rate limiting & quotas | Tiered plan system with per-user limits on agents, requests/minute, spawns/hour, storage, and (0.8.1) [messages and AI tokens per day](#daily-plan-limits-and-free-resets) |
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
| Keyboard shortcuts | Cmd+D director, Cmd+J chat, Cmd+K shortcuts overlay, Cmd+\\ simple / full view, Cmd+[ ] switch chat tabs (the Cmd/Ctrl+1–5 page jumps were removed in 0.8.0) |
| Dark/light theme | Full theme support with toggle in top bar, light mode overrides for all zinc palette and markdown styles |
| Free-form layout | Drag agent cards freely on the desktop, positions persisted to localStorage |
| Embedded chat | Collapsible chat panel directly on agent cards |
| Resizable panels | Director (220–500px), chat (320–900px), and tool panels (280–500px) are all resizable with drag handles |
| Agent Council | Multi-agent deliberation — structured turn-based discussions (debate, brainstorm, review, planning) with synthesis, voting, TL;DR, and minutes export. Can **auto-assemble** its own task-modeled panel (NEW in 0.6.3) |
| Basna | Independent network-source ensemble — routes a task to specialist archetypes, runs them blind and in parallel, and merges by learned reliability |
| Vatra | Collaborative ensemble (compose mode on Basna, NEW in 0.6.3) — a Lead splits the task into owned subtasks, specialists collaborate on a shared blackboard, and a reporter assembles one deliverable |
| Frontier Horizon | Optional **Deep** and **Plan** depth for Basna/Vatra (NEW in 0.6.4) — self-consistency voting + diverse-lens critics + a verify-and-revise closer, and a verify-gated, re-planning multi-step plan whose steps run as a single model, a Basna ensemble, or a Vatra team |
| Agent Forge | AI-powered team decomposition — describe a goal, get a team of agents with roles, instructions, and tools |
| Fleet-level instructions | Per-agent instructions injected into system prompts, editable from agent config editor |
| Datastore browser | View agent datastore tables and rows directly from agent cards |
| Agent groups & tags | Organize agents into teams and roles with color-coded group badges |
| Trace timeline | Real-time orchestrator observability — Gantt-style span visualization in the chat panel with duration, token usage, phase breakdown (decompose/execute/task/synthesize), and status tracking |
| Connections page | Unified external-service authentication: Google OAuth (with per-scope picker and sensitivity tiers), ChatGPT (Codex OAuth), centrally-managed MCP servers (HTTP + stdio transport, per-agent allowlists, hot-reload over SSE), Microsoft 365 (Entra app + per-user connect), Antigravity, WhatsApp, Typesense deep memory, each user's Jev (TypeSafe AI) API key with a "Try it" playground, and (0.8.1) the user's Telegram link to the team bot and their Soniox key — **one tab per connection** since 0.8.0, with status dots, remembered last tab, and state kept while you switch tabs |
| Agent Folders | Admin-gated `fd-data` cleanup (NEW in 0.6.5) — lists each agent subfolder with on-disk size, file counts, last-modified, and desktop presence (**running / on-desktop / orphaned**); view/download workspace files (HTML in a sandboxed iframe, Markdown with GFM tables) and delete a folder (refused while the agent is still running). Path-sandboxed to direct children of `fd-data`; backend `/fd/agentfs` |

### Connections

The Connections page lives in the Flight Deck sidebar and manages external-service authentication for the entire fleet. A single Flight Deck instance can serve OAuth tokens to every captain-claw sub-agent it spawns (on the same host or a remote one) via a shared-secret or loopback trust boundary — no need to re-authenticate per agent.

Since 0.8.0 the page has **one tab per connection** — Google, Microsoft, Telegram (0.8.1), ChatGPT, Antigravity, MCP Servers (the server list plus *Agent access for Claude*), Jev, Soniox (0.8.1), WhatsApp and Deep memory — with status dots (green connected, amber configured). Inactive tabs stay mounted, so a sign-in popup's result and a half-typed form survive a tab switch, and the last tab is remembered per browser. Kiosk users get a Connections dialog with the tabs that are a person's own: Google, Microsoft, Telegram, MCP Servers, Jev and Soniox.

- **Telegram** — ask for the deck's team bot, picking one of your agents; after an admin approves, a one-time link binds your Telegram account. Then the **Inbox** switch, "Add <agent> to a group" and the agent's groups. See [Telegram: the Team Bot](#telegram-the-team-bot--new-in-081).
- **Soniox** — your own Soniox key for speech-to-text and voices (checked with Soniox before it is saved; only a hint ever comes back). Without one you use the admin's team key, else the deck's `SONIOX_API_KEY`. Agents, devices and the WhatsApp bridge use their owner's key; the agent itself never holds it. See [`audio`](#audio).

#### Google

Captain Claw does not ship with bundled Google OAuth credentials. Every deployment must create its own OAuth 2.0 Client ID in [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (type: **Web application**), add the redirect URI shown in the Connections page (e.g. `http://localhost:25080/fd/google/callback`) to the Authorized redirect URIs list, and paste the Client ID + Client Secret into the form. The desktop app's deck can use a **Desktop app** client instead (since 0.8.1; see [Desktop app mode](#desktop-app-mode--new-in-081)).

**Scope picker**

Rather than requesting a fixed list of scopes at connection time, Captain Claw lets you tick individual scopes from a catalog of 13. Each scope is tagged with a sensitivity tier that controls how Google treats your app:

| Tier | Badge | Meaning |
|---|---|---|
| `none` | green | Freely usable by unverified apps. No consent-screen restrictions. |
| `sensitive` | amber | Google flags these. Unverified apps require you to add yourself as a Test user. |
| `restricted` | red | Requires security review + verification for production use. |

**Default scope set (all non-sensitive):**

- `openid` — Sign in with Google *(required)*
- `email` — Your Google account email *(required)*
- `profile` — Your name and profile picture
- `drive.file` — Per-file Drive access for files the app creates or the user picks

**Opt-in scopes:**

- Drive: `drive.readonly` (restricted), `drive` (restricted)
- Gmail: `gmail.readonly` (restricted), `gmail.compose` (sensitive — drafts, and Google also accepts sends with it; agents send only when the user turns on Email sending), `gmail.modify` (restricted — needed for archive/label/move), `gmail.send` (sensitive)
- Calendar: `calendar.readonly` (sensitive), `calendar` (sensitive)
- Cloud: `cloud-platform` (sensitive) — required for Vertex AI / Gemini access alongside a Google Cloud `project_id`

**Important: unverified apps and test users.** Google blocks unverified apps from requesting sensitive or restricted scopes at runtime. If you tick any of those scopes, you must either:

1. **Go to Google Cloud Console → APIs & Services → OAuth consent screen → Test users** and add your Google account there while the app is in Testing mode, or
2. **Submit your app for verification** (required for production use).

The Connections page shows a reminder banner about this directly under the scope picker.

**Scope changes invalidate tokens.** Changing the scope set (same as rotating the client_id) clears stored tokens and forces fresh consent, because OAuth refresh tokens are bound to their original scope grant. Captain Claw handles this automatically on save.

**Vertex AI / Gemini.** Set a `project_id` in the form **and** tick the `cloud-platform` scope to enable Vertex AI. The UI's `supports_vertex` status only reports `true` when both are in place, so the Gemini provider won't be promised to you on a connection that can't deliver it.

**Endpoints:**

- `GET /fd/google/status` — connection state (configured / connected / granted_scopes / supports_vertex).
- `GET /fd/google/config` — full config including `scopes`, `default_scopes`, and `scope_catalog`.
- `GET /fd/google/scope_catalog` — just the scope catalog.
- `POST /fd/google/config` — save any subset of `client_id`, `client_secret`, `project_id`, `location`, `scopes`. Pass `{"clear": true}` to wipe saved credentials, tokens, and scopes in one shot.
- `GET /fd/google/login` — starts the OAuth flow (returns HTTP 400 HTML when not configured).
- `GET /fd/google/callback` — OAuth callback.
- `POST /fd/google/logout` — revoke and clear.
- `GET /fd/google/access_token` — agent-facing; returns a currently-valid access token. Gated by loopback or `X-Agent-Secret` header.
- `GET /fd/google/credentials` — agent-facing; returns full `authorized_user` credentials for LiteLLM / Vertex paths.

#### ChatGPT (Codex OAuth) — Sign in with ChatGPT

A new connection card uses the same "Sign in with ChatGPT" flow the Codex CLI uses. Captain Claw reads the tokens the Codex CLI caches in `~/.codex/auth.json` and talks directly to the ChatGPT Responses API at `https://chatgpt.com/backend-api/codex/responses` — **no `OPENAI_API_KEY` required**, billing goes through your ChatGPT Plus/Pro/Team plan.

**Setup:**

1. Install the [Codex CLI](https://github.com/openai/codex) on the host running Flight Deck.
2. Run `codex login` — the CLI handles the OAuth flow and stores tokens in `~/.codex/auth.json`.
3. Open **Flight Deck → Connections → ChatGPT (Codex)** and click **Reimport from Codex CLI**.
4. Configure an agent with `provider: openai` and a Codex-family model (see below).

**Status display:**

- Email and ChatGPT plan (from the decoded JWT claims).
- Token expiry countdown (`5h 32m`, or `Token stale` when inside the 60-second safety margin).
- Access-token preview (first 12 + last 6 chars) for sanity-checking you're reading the file you expect.
- Auth file path on disk.

**Supported models:**

The ChatGPT/Codex backend serves a specific set of Codex-family models, not the full OpenAI API catalog:

- `gpt-5`
- `gpt-5-codex`
- `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`
- `gpt-5.2`, `gpt-5.2-codex`
- `gpt-5.3-codex`
- `codex-mini-latest`

Common non-Codex aliases — `gpt-5-mini`, `gpt-5.1-mini`, `gpt-5.2-mini`, `gpt-5.3-mini`, `gpt-5.4-mini` — are automatically remapped to `gpt-5.1-codex-mini`. The agent logs `ChatGPT Codex: remapped model alias` at construction time so you can see what the remap resolved to. Requesting a model that isn't in the known-supported set fires a warning with the full supported list.

**How tokens stay fresh:**

The `CodexAuthManager` decodes the JWT `exp` claim with a 60-second safety margin. Before every request, `_ensure_fresh_auth()` is a cheap cache hit when the token is still valid. On a 401 response, the provider forces a refresh and retries the request exactly once (in both `complete()` and `complete_streaming()`). Under Flight Deck, the token source is `GET /fd/codex/access_token` — so every sub-agent Flight Deck spawns shares one centrally-managed connection, even across hosts. Without Flight Deck, the manager reads `~/.codex/auth.json` directly for standalone installs.

**Endpoints:**

- `GET /fd/codex/status` — `configured`, `connected`, `email`, `plan`, `account_id`, `expires_at`, `seconds_until_expiry`, `stale`, `access_token_preview`, `auth_path`.
- `POST /fd/codex/reimport` — force a re-read of `~/.codex/auth.json`. No persistent state to update — every status call already re-reads on demand — but this gives the UI a clear affordance to press after running `codex login` again.
- `GET /fd/codex/access_token` — agent-facing; returns `{access_token, account_id, expires_at, email, plan}`. Gated by loopback or `X-Agent-Secret` header, same as the Google equivalent.

**Environment variables:**

- `FD_URL` — when set on a spawned captain-claw agent, the Codex auth manager pulls tokens from Flight Deck instead of reading `~/.codex/auth.json` locally.
- `FD_AGENT_SHARED_SECRET` — optional shared secret used in both directions (FD validates incoming agent calls; agents attach it as `X-Agent-Secret` when calling FD).

#### MCP servers (centrally managed)

Captain Claw 0.4.23 moved Model Context Protocol (MCP) management out of every individual agent and into Flight Deck as a fleet-wide service; 0.4.24 (Phase 2) added stdio transport, per-agent allowlists, hot tool-list reload, and streaming tool calls.

Add an MCP server once in **Connections → MCP servers** and every one of your agents that the server's allowlist permits picks up the tools — no per-agent `config.yaml` MCP block, no N×M connections to upstream servers, and a single chokepoint for tokens, sessions and observability. Since 0.8.1 a server is **private to whoever added it** unless an admin shares it with everyone (see **Ownership** below).

**Supported transports:**

- **`http`** — Streamable HTTP binding. Required field: `url`. **Auth:** none, the OAuth2 `client_credentials` flow (`client_id`, `client_secret`, `token_endpoint`), or (0.8.1) **OAuth sign-in** (see below); plus arbitrary extra `headers`. `Mcp-Session-Id` is captured and replayed transparently.
- **`stdio`** — child process speaking JSON-RPC over stdin/stdout (NDJSON). **Admin-only since 0.8.0** (creating, editing, testing, probing or deleting one runs a command on the deck's host). Required field: `command`. Optional `args` (argv tail) and `env` (extra env vars merged on top of the FD process environment). The de-facto standard for local servers shipped via `npx` / `uvx` (filesystem, sqlite, github, postgres, etc.). The child is spawned lazily on the first request, auto-respawned if it dies, and terminated with SIGTERM (2 s grace) then SIGKILL on close.

**Parallel Search example.** For an optional, keyless HTTP search/fetch connection and a runnable smoke test, see [examples/mcp/](examples/mcp/README.md).

**Per-agent allowlists.** Each server carries an optional `allowed_agents: list[str]`. Empty list = every agent the server reaches (its owner's agents; every agent on the deck for a deck-wide server). Once any slug is listed, only those agents can see, list, or call the server. Restricted servers return HTTP 404 to disallowed agents — same shape as "doesn't exist," so a restricted server's existence is opaque.

**Ownership — NEW in 0.8.1.** A server belongs to the user who added it (`owner_id`). Only that owner's agents reach its tools (the agent's owner comes from Flight Deck's spawn records, then the allowlist applies), and only the owner sees, calls, edits, tests, signs in to or removes it — admins don't see users' private servers either. A name you can't see answers 404 (409 when you try to reuse it), so a private server's existence never leaks. An admin can share one of their own servers with everyone (**Share with everyone on this deck**, `deck_wide`): every user's agents then get its tools, and admins manage it. Sharing an OAuth server shares the account it signed in as, and the confirmation says so. Rows show a **Private** or **Deck-wide** badge, and servers you can't manage are read-only. On the first start after the upgrade, older OAuth servers go to whoever signed them in (else they stay private and ownerless, removable by admins); other older servers stay deck-wide, as they behaved. A deck without sign-in is one user and skips ownership. Shared-agent members never get MCP tools. **Admin → Integrations** lists the deck-wide servers and shares or unshares them (see [Integrations](#integrations-shared-keys-telegram-and-mcp)).

**OAuth sign-in with Dynamic Client Registration — NEW in 0.8.1.** An `http` server with **Auth: OAuth sign-in** follows the MCP authorization spec: Flight Deck discovers the server's authorization server (the `401 WWW-Authenticate` challenge, RFC 9728 protected-resource metadata, RFC 8414 / OIDC metadata), registers itself as a client (RFC 7591; a client ID on the record skips registration for servers without it), and signs you in through a popup with PKCE (S256) and a resource indicator bound to the server URL. Tokens refresh by themselves (one refresh per server at a time; rotating refresh tokens) and are kept in `<servers file>.oauth.json` (mode 0600) beside the servers file, never in the record. **Sign out** revokes them where the server supports it. On a 401 the transport refreshes and retries once; a plain server that answers with a Bearer challenge gets a hint to switch to OAuth sign-in. **Test** on a server nobody signed in to yet reports whether sign-in is ready (issuer, registration support) instead of failing. The form's Auth selector offers **None**, **Client credentials** and **OAuth sign-in**, with **Save & sign in**, **Sign in** / **Reconnect** / **Sign out**, and who signed in.

Agents identify themselves with the `X-Agent-Auth` token Flight Deck minted when it spawned them, and the allowlist is checked against the slug in Flight Deck's own records (since 0.8.0 — the `X-Agent-Slug` header alone no longer counts). Agents Flight Deck didn't spawn are refused; Docker agents need an image rebuilt on 0.8.0.

**Hot reload.** Agents subscribe to `GET /fd/mcp/agent/events` (an SSE stream) on boot. Save / edit / delete a server in the admin UI and every entitled agent re-registers proxy tools without a restart. The watcher reconnects forever with exponential backoff capped at 30 s. 25-second `ping` frames keep the connection alive through idle-TCP-killing proxies.

**Streaming tool calls.** `POST /fd/mcp/<name>/call_stream` runs the upstream call as a background task and emits an SSE stream of:

- `{"type":"progress","params":{...}}` — one frame per upstream `notifications/progress`
- `{"type":"result","server":"...","tool":"...","result":{...}}` — terminal frame
- `{"type":"error","error":"..."}` — terminal frame on failure
- `{"type":"ping"}` — every 25 s while idle

Cancels the upstream call cleanly when the client disconnects mid-stream. The agent's tool execution loop still uses one-shot `/call`; the streaming endpoint is for UIs that want live progress indicators.

**Storage.** A flat JSON file, per Flight Deck instance (since 0.8.0): `CAPTAIN_CLAW_FD_MCP_PATH` if set, else `$CAPTAIN_CLAW_FD_HOME/mcp_servers.json`, else `$FD_DATA_DIR/mcp_servers.json`, else the legacy `~/.captain-claw-fd/mcp_servers.json`. A deck that moves to `FD_DATA_DIR` and wants to keep its servers copies the legacy file there once. Writes are atomic (write-tmp + rename) so a crashed Flight Deck never leaves a half-written file. Secrets are masked (`••••••••`) in every UI response — re-submitting a form with a masked secret preserves the stored value.

**Endpoints (signed-in user — each user sees and manages only their own servers, admins also the deck-wide ones):**

- `GET /fd/mcp/servers` — list the servers you can see, with secrets masked + runtime status (initialised? tool count? last error?)
- `POST /fd/mcp/servers` — insert or update a server by name. Accepts `transport`, `url`, `command`, `args`, `env`, `auth` (`""` or `"oauth"`), `client_id`, `client_secret`, `token_endpoint`, `headers`, `enabled`, `allowed_agents`.
- `POST /fd/mcp/servers/{name}/sharing` — (0.8.1, admin) change only `deck_wide`, for a server the admin can see.
- `DELETE /fd/mcp/servers/{name}` — remove a server, drop its cached state, terminate any running stdio child.
- `POST /fd/mcp/servers/{name}/test` — end-to-end probe (re-init + tools/list).
- `POST /fd/mcp/probe` — probe a transient (non-persisted) record; powers the "Test connection" button on the Add-server form. For stdio records the spawned child is closed before returning so probing never leaks processes. Since 0.8.1 it never reuses another server's stored secret or sign-in.
- `POST /fd/mcp/oauth/connect-ticket`, `GET /fd/mcp/oauth/login`, `GET /fd/mcp/oauth/callback`, `POST /fd/mcp/oauth/disconnect` — (0.8.1) the OAuth sign-in popup, with the same browser-bound ticket and state cookie as the Google and Microsoft sign-ins.

**Endpoints (agent-facing — gated by loopback or `X-Agent-Secret` header, filtered by ownership and allowlist):**

- `GET /fd/mcp/agent/servers` — enabled server names visible to this agent.
- `GET /fd/mcp/{name}/tools?refresh={bool}` — proxy `tools/list` (30-second TTL cache, `refresh=true` bypasses).
- `POST /fd/mcp/{name}/call` — proxy `tools/call`.
- `POST /fd/mcp/{name}/call_stream` — streaming variant; SSE response.
- `GET /fd/mcp/agent/events` — SSE stream of `server_added` / `server_updated` / `server_removed` / `tools_changed` events for hot reload.

**Environment variables:**

- `FD_URL` — agents call FD's MCP proxy at this base when set. Unset = MCP is disabled in the agent (Phase 1+ design: MCP is exclusively a Flight Deck feature).
- `FD_AGENT_SLUG` — the agent's own slug; Flight Deck injects this when spawning sub-agents (sent as `X-Agent-Slug`, informational since 0.8.0 — the allowlist uses the recorded slug behind `X-Agent-Auth`).
- `FD_AGENT_SHARED_SECRET` — optional shared secret; same one used by Codex / Google.
- `CAPTAIN_CLAW_FD_MCP_PATH` — override the storage file location (useful for tests); otherwise `CAPTAIN_CLAW_FD_HOME` / `FD_DATA_DIR` decide it per instance.

### Library (NEW in 0.5.7)

The Library page is the single place for your **model tiers** and the **archetype gallery** (both moved out of Agent Forge in 0.5.7). Since 0.7.1 it is split into two tabs — **Model Tiers** and **Archetypes** — with redesigned tier cards (identity / capacity / connection groups).

- **Model tiers** — a per-tier editor with six tiers — **Reasoning**, **Balanced**, **Fast / high-volume**, **Long context**, **Coding**, and **Vision** (the last two **NEW in 0.6.5**) — each carrying its own provider, model, API key, base URL, and input/output context length, plus a "Forge using" selector and an "Additional API Keys" section (env vars passed to spawned agents). Persisted per-user via `/fd/settings`. **Agent Forge and Basna both resolve their models from here.** **Coding** (default `opus-4-8`, 65 536 output ctx) backs multi-file diffs/debugging — the `code-implementer`, `debugger`, and `simplifier` archetypes route here; **Vision** (default `sonnet-4-6`) backs image/screenshot/document work for the **Vision & Multimodal** family (`visual-extractor`, `ui-reviewer`, `brand-design-reviewer`). Existing saved tier sets are **auto-backfilled** with the new cards (inheriting their provider/key) on the next Library load.
- **Archetype gallery** — the curated ~20 base agents (Deep Researcher, Code Reviewer, Project Coordinator, Deal Screener, …). Click a card to **spawn that archetype directly** as an agent (its tier resolves to a concrete model). Served from `instructions/archetypes.json`.
- **Custom archetypes (NEW in 0.6.0)** — the curated set is a **base** you extend. Use **New archetype** to create your own — by hand (role, family, description, cognitive mode, tier, tools, keywords, fleet instructions) or **generated from a prompt** (describe the agent and an LLM drafts a complete archetype to review). A custom archetype with the same id **overrides** the base one for you; new ids are added alongside. Custom cards are badged with edit/delete. Your archetypes appear everywhere base ones do — the gallery, **Agent Forge** composition, and **Basna** routing. `GET /fd/archetypes` returns the merged base+user registry; CRUD + `generate` live under `/fd/archetypes/*`; stored per-user in `user_archetypes`.
- **Setup wizard (NEW in 0.7.1)** — a modal (**Setup wizard** button, on the Model Tiers tab) that takes **one** model — provider, model id, API key, base URL — plus **context presets** (input 128k / 160k / 256k / 400k / 512k / 1M, output 16k / 32k / 64k / 128k / 256k; power-of-2 based, defaulting to 256k in / 32k out) and applies it to **every tier** of a newly named tier set. On a **fresh install** (no tier set configured yet) it **auto-opens once**; completing that first-launch setup takes you straight to **Quick Chat**. Invoked manually, it just returns to the Model Tiers tab.

### Quick Chat (NEW in 0.7.1)

Quick Chat (sidebar: **Workspace → Quick chat**) is the fastest path from "which agent do I want" to "talking to it". Pick an archetype from the searchable gallery; Flight Deck spawns it as a **hidden** agent (it exists on the Agent Desktop but stays off the canvas) and, once a readiness probe (`/fd/probe`) confirms the agent's web server is listening, opens a normal chat with it — the full chat panel, so plan mode, next steps, attachments, and model switching all work.

Each active quick agent shows a live status (starting → warming up → ready) and one prominent **Promote to desktop** button that reveals it on the Agent Desktop canvas (with **View** / **Hide** afterward), plus **Stop & remove**. Spawns go through the standard process path (`/fd/spawn-process`); the hidden/visible flag is a local desktop preference — no backend changes.

### Hosting (NEW in 0.7.2)

Hosting (sidebar: **Workspace → Hosting**) publishes a VFS folder at a public, globally-unique name — either a **static site** or a **running app** — routed through Flight Deck. Publishing requires login (owner-gated); *serving* is public. Manage it on the page: publish, **edit**, **Start/Stop** (apps), **Open**, **Unpublish**. Registry lives in `fd-data/.vfs-hosting.json`.

- **Static site → `/vfs/<name>/…`** — files are served straight from the VFS folder (optionally a subfolder like `dist`). A folder with an `index.html` serves it (with SPA fallback for extension-less routes); a folder *without* one gets a browsable **directory autoindex**. A built SPA must be built with its **base path set to `/vfs/<name>/`** (e.g. Vite `base`) so its assets resolve under the prefix.
- **Built app → `/vfs-apps/<name>/…`** — Flight Deck runs your **start command** as a managed subprocess (`cwd` = the VFS folder) and reverse-proxies **HTTP (all methods) + WebSocket** to it. Manual Start/Stop.

**The path-prefix contract (important).** Because apps/sites are served under a path prefix, **root-absolute URLs break** — a browser request to `/api` goes to the Flight Deck root, not your app. On start, FD injects these env vars into the app process:

| Env var | Value | Use |
|---|---|---|
| `PORT` | assigned free port | bind it (on `127.0.0.1`) — don't hardcode |
| `HOST` / `HOSTNAME` | `127.0.0.1` | bind address |
| `FD_BASE_PATH` | `/vfs-apps/<name>/` | prefix for every root-absolute asset/API URL |
| `FD_VFS_APP` | `<name>` | the published name |

So a client call should be `fetch(\`${FD_BASE_PATH}api/x\`)` (or a bundler base derived from `FD_BASE_PATH`), never `fetch("/api")`. Apps built with the **Code** module get this baked in automatically — its plan/build prompts require base-path awareness and the code-reviewer flags root-absolute URLs and hardcoded ports as findings.

Endpoints: management under `/fd/hosting/*` (list / publish `POST` / edit `PUT` / `start` / `stop` / unpublish `DELETE`); public serving at `/vfs/<name>/…` and `/vfs-apps/<name>/…` (registered before the SPA catch-all).

### Public file links — NEW in 0.8.1

Hosting publishes a folder; a **public file link** publishes one file. A **Share** (globe) button on an agent's files — the full file browser, the side files panel, the phone's file sheet and the [File library](#file-library--new-in-081) — publishes a copy of that file at `<deck>/s/<token>` for anyone holding the link, no sign-in. The dialog explains what publishing means before anything happens, then shows the link with **Copy** / **Open**, its views and expiry, **Update copy** and **Stop sharing**, and a list of all your public links (so one whose file is gone can still be turned off).

- **A snapshot, not live.** The file is copied into `<FD_DATA_DIR>/public_shares` when it is shared; the public page never contacts an agent and takes nothing but the token. **Update copy** refreshes it under the same URL.
- **Safe to open.** The page is served with no script. Markdown is rendered with raw HTML escaped; HTML, SVG and XML-like files run in a sandbox with no access to Flight Deck's cookies, storage or API; text kinds are served as plain text; images, PDF, audio and video display; other types download. Unknown, stopped and expired links all answer the same 404.
- **Agents make links too**, when a person asks ("share that file"), with the [`public_link`](#public_link--new-in-081) tool. Those links belong to the agent's owner and show in the owner's Share dialog.
- **The address.** Links use `FD_PUBLIC_URL`; else the address the owner's browser reaches the deck at (learned after sign-in; never a localhost or private address); else the most recently seen address of any user; else the first domain in `FD_ALLOWED_HOSTS`.
- `FD_PUBLIC_FILE_LINKS=off` hides the button, refuses the tool and stops every existing link; `FD_PUBLIC_FILE_LINK_MAX_MB` (default 50) caps the file size. Shared-agent members can't publish an agent's files.

### Quality, cost & collaboration (NEW in 0.7.3)

Three cross-mode capabilities landed in 0.7.3 for Basna, Vatra, and Code. **All are opt-in and default-off** — with Quality on **Basic**, no wage entered, and grouped mode off, runs behave exactly like 0.7.2.

**Quality presets.** A **Quality** control on the Basna/Vatra and Code surfaces with four settings:

- **Basic** — today's behavior; no extra levers.
- **Balanced** / **Thorough** — progressively more scaffolding (grounding, verification, structure, retries) so a weaker or local model reaches closer to frontier output.
- **Custom** — toggle any subset of the ~11 individual levers. The notable ones:
  - **Intent brief** — before the team is chosen, the task is restated as one faithful, structured brief (objective / scope / deliverable / constraints). It drives **team selection** and every worker's framing, and is **editable** — review it and **Re-route on this brief**. The original request always governs on conflict.
  - **Fact-check + honesty guard** — a tool-enabled fact-checker reviews the deliverable; unconfirmable-but-asserted specifics are **hedged**, and a non-destructive **fact-check audit ledger** is written next to the output.
  - **Rubric contract** — the completeness checklist is derived from the standard the task names and the deliverable is scored against it (missing / thin items surfaced).
  - **Source corpus** — `web_fetch` saves the **full** page text into the run's VFS folder and returns a lean head + pointer, keeping context small while the whole source stays searchable.
  - **Acted-gate / escalation retries**, **research map**, and Code-only levers (**test gate**, **deep-build**, **coverage→backlog**).

**Run cost card.** Every finished Basna/Vatra/Code run shows **Run cost**: the dollar cost (priced from the curated `captain_claw/instructions/model_prices.json` table, cache-aware) and an **effective $/hour** you compare to a wage you type in (persisted globally). It also shows **wall-clock** vs **agent-time** (Σ of every model call's duration) and their ratio as **parallelism** (e.g. `2.2×`), an **input / cached / output** token split, and a **By model** breakdown. Unknown models show tokens with `—` for dollars rather than a wrong number. Reprice by editing `model_prices.json`, or pass a per-tier `price` override.

**Vatra execution groups (grouped mode).** A **Grouped** toggle (next to Max parallel) runs owners in **ordered phases A → B → C → D** with a barrier between them — research/design first, build/write in the middle, review/assembly last — instead of all-at-once. A later phase already sees everything earlier phases posted. Archetype presets pick each owner's phase; the Lead may push a piece **later** (never earlier) via an optional `group` field. A **bounded clarification loop** (cap 2/run) lets a later agent ask an earlier one for a missing input, gated by the Lead. The live panel **sections the working agents by phase**.

**Attachments in Vatra.** Attached files are now uploaded on **Plan team** and **Run**, **saved into the run's VFS folder** (browsable + indexed), and — when the intent brief is on — read by a short-lived **file-examiner agent** that briefs the whole team before it starts, so workers act on the files' contents, not just their names.

**Max parallel agents.** A **Max parallel** input (0 = unlimited) caps how many agent turns run at once — set it low (e.g. 2) for local models so parallel prefills don't exhaust the serving box's memory.

### Flight Deck security hardening — NEW in 0.8.0

0.8.0 makes Flight Deck safe to put in front of a team. Most of it needs no configuration; the parts that do are in [docs/team-deployment.md](docs/team-deployment.md).

- **Host and origin guard.** Every HTTP and WebSocket request must name a host this deck answers to — `localhost`, any IP, this machine's hostnames, `FD_PUBLIC_URL`'s host, `FD_CORS_ORIGINS` hosts, or `FD_ALLOWED_HOSTS` — and a browser request carrying a foreign `Origin` (or a cross-site fetch) is refused. That closes DNS rebinding and "any web page can drive your local deck" on auth-off desktop decks. CORS defaults to the deck's own origins (`FD_CORS_ORIGINS=*` is the explicit legacy opt-out). **A deck reached through its own domain must set `FD_ALLOWED_HOSTS` or `FD_PUBLIC_URL`**, or every request — the SPA, webhooks, OAuth callbacks, inbound MCP — is refused (the log names the setting).
- **Agents prove who they are.** Code, Hosting, Basna, Vatra and Bat agent routes need a verified bearer, loopback or the agent secret, and take the owner from Flight Deck's own spawn records, never from a body `owner_id`. `/fd/consult-peer` and `/fd/delegate-peer` need the calling agent's own `X-Agent-Auth`, take host/port/token only from Flight Deck's records, and read attachments only from the caller's own workspace. Google, Codex and MCP tokens go only to the agent's owner. Docker agents need `FD_URL` and `FD_AGENT_SHARED_SECRET` in their env, and an image rebuilt on 0.8.0.
- **Owner-scoped everything.** Port-addressed agent proxies refuse anyone but the agent's owner (admins excepted); scheduler jobs and flows have owners (existing ones become admin-owned and keep firing); stdio MCP servers are admin-only; provider keys never reach the browser (`@system` sentinels resolved at spawn).
- **Registration is closed** after the first (admin) account — admins add people with **Add user** or invite them by email on the **Users** page (Admin → Users before 0.8.1); since 0.8.1 people can join a [waitlist](#users-invitations-and-the-waitlist) instead (`FD_REGISTRATION_OPEN=1` reopens self-signup).
- **One deck, one state.** Several decks on one host no longer share MCP servers, code apps or the agent secret — each resolves them from `CAPTAIN_CLAW_FD_HOME`, else `FD_DATA_DIR`, else the legacy `~/.captain-claw-fd`. A deck moving to `FD_DATA_DIR` copies its `mcp_servers.json` there once to keep its servers.
- **Chat bridges send only deliverables.** Telegram, Slack, Discord and WhatsApp auto-send a file only if this turn wrote it inside the agent's `saved/` area (never a `Path:` line from an email or web page). See [Telegram: Per-User Sessions](#telegram-per-user-sessions).

### Confinement on a multi-user deck — NEW in 0.8.4

Every process agent runs as the deck's OS user. On a deck with sign-in, Flight Deck pins `CLAW_CONFINE=user|admin`, the agent's home and its workspace, after the agent's own `.env` and env vars (`captain_claw/deck_confine.py`):

- **Files.** Path arguments are checked by real path. Reads: the agent's folder (not its `.env`), its workspace, its owner's VFS and the call's shared folders; for an admin's agent also linked and read folders. Writes: the workspace and the VFS, never control files. `glob` and `grep` results are filtered, and the browser takes web URLs only.
- **Running code.** A non-admin's agent never gets `shell`, `terminal`, `scripts`, `app_runner`, `code`, `termux`, `bat`, `desktop_action`, `screen_capture` or `pinchtab`, and installs no skill dependencies. Its `browser`, `direct_api`, `apis` and `web_fetch` take public addresses only.
- **Admin-only with sign-in on.** Publishing or starting a hosted app (static sites stay open), `POST /fd/vfs/links` and `/fd/vfs/browse-fs`.
- **Keys.** Non-admins read secret-named `.env` values and team keys as `····` and the last four; `PUT /fd/agent-config` swaps a placeholder back for the value on disk. A team key, or a key in Flight Deck's environment, goes only to the provider's own endpoint or one the published team sets route it through.
- **Pinned variables.** `PATH`, `PYTHON*`, `LD_*`, `GIT_*`, `NODE_*`, `PIP_*`, `HTTP_PROXY`, CA bundles and the like keep Flight Deck's values for a non-admin's agent, and saving one answers 400, naming it.

`FD_AGENT_CONFINE=off` turns it off. A role change applies at the agent's next start. An admin's agent still has a shell that reaches the whole host: this stops tenants and prompt-injected file calls, not an admin's own shell. The OS sandbox is designed and parked in [docs/agent-os-sandbox-plan.md](docs/agent-os-sandbox-plan.md).

### Running Flight Deck for a team — NEW in 0.8.0

- **Users** (Admin → Users in 0.8.0; its own **Users** page since 0.8.1). Add users (email, password, role) or invite them by email, set a password, and open a user's **Agents** to create (archetype picker), start, stop, restart, configure or remove their agents — every action logged as `admin_act_as`. It is not a login-as: an admin acts on agent-management routes only. The rest of an admin's work — the Admin console, invitations, the waitlist, passwords, branding, daily limits — is in [Deck Administration](#deck-administration--new-in-081).
- **Team tier sets and team keys.** An admin publishes a tier set (**Library → Publish to team**); teammates with no models of their own run on it. Publishing shares the keys as **team keys** — a key on a provider's own endpoint becomes that provider's org key, a gateway key goes into a per-endpoint store — and process agents missing a key get it and restart automatically. A spawn nothing can supply a key for is refused instead of creating a dead agent. Decks that published before 0.8.0 should publish again.
- **Per-user connections.** Each user connects their own Google and Microsoft accounts, Jev key, WhatsApp nudge number and (0.8.1) Soniox key and Telegram link (**Connections**); agents use their owner's. Each user also fills in a **Profile** (see [Owner Profile](#owner-profile)).
- **Collaboration.** A persistent notification bell (shares, finished runs, sends), team spend by user and run kind in **Admin → Usage**, **Copy to my workspace** for a folder shared with you, and **Duplicate** on shared or built-in archetypes to fork an editable copy.
- **Shared agents** (`FD_AGENT_SHARING=1`): one agent, several users — see [Shared agents](#shared-agents-a2-members-own-google-deep-memory-files) below.
- **Agent options.** A gear on each agent opens a tabbed options dialog (General, Instructions, `config.yaml`, `.env`); name and description are stored by Flight Deck, so they survive a browser wipe. Instructions are stored per agent by Flight Deck, so two tabs can't erase each other's.

### Crew: coding agents working for you

Crew lets Captain run Claude Code, Codex, OpenCode and Pi for you. You talk to your agent ("have Codex fix the login test and Claude add dark mode"), or use the Crew page. Each task gets one of those CLIs in a git worktree of its own, several at once. Flight Deck watches them without spending tokens and tells you when one finishes or needs a decision, on the channel you asked from.

**An admin approves it, per user.** On Users, open someone's row and tick Crew: which CLIs, and how many at once. It is off for everyone else. Admins, and a deck with sign-in off, can always use it. Withdrawing it stops what is running at once. Flight Deck checks the approval on every call.

> **Approving someone trusts them with the server.** A worker runs commands as the same system user as Flight Deck, so it can reach what that user can, including other people's files and the deck's database. No OS sandbox holds it to its worktree yet (`docs/agent-os-sandbox-plan.md`). Codex runs in its own `workspace-write` sandbox; the others do not.

**Whose account.** Each user's own: Crew → Logins. A `claude setup-token` token or an Anthropic key for Claude Code; an OpenAI key or the contents of `~/.codex/auth.json` for Codex; provider keys for OpenCode and Pi ("Your own AI keys" count too); a GitHub token for repositories and pull requests. A non-admin's worker never falls back to the deck owner's login, and its environment holds none of the deck's secrets. An admin's worker uses this computer's logins unless they saved their own.

**Where it works.** A Code project's folder, a repository address, or (admins only) a folder on the server. Install the CLIs on the deck's host: Crew → Logins shows which are there and the command for those that aren't.

**What lands.** Every task works on its own branch and never pushes. "Ship" tasks leave a branch or open a pull request; "scout" tasks only report. Merging and throwing work away take a person: the Merge button, or Approve in the bell on a merge your agent asked for. Work that hasn't landed is never removed unless you say so.

**Projects.** Tasks are grouped by project: a Code project with its folders, or the repository a task was given. The page lists what wants you first and folds what's finished; a project's own page shows its counts and has **Tidy up**, which clears finished tasks off the page. Work that hasn't landed always stays. **New project** makes one before it has any tasks.

**Where results go.** A worktree is out of an agent's reach, so when a task finishes its work comes to your VFS, under `vfs:crew/<project>/<task>/`: `report.md` is the worker's whole message, and `files/` holds the whole project if it's small (under 300 files and 20 MB), otherwise only what the task changed. Only commits are copied, with no `.git`. Your agent reads them there, and the callback tells it where, so it never needs a worker to print files into a message. In the page, a task's **Files** tab shows them, can copy a big project whole, and **Get a link to this page** publishes one with an `index.html` from that clean copy. Agents do the same with `hosting`. Discarding a task's work removes its copy. A project's own folder also stays readable directly, at the `vfs:` path `crew action=options` shows.

**Models and limits.** The Crew page shows the model each CLI last ran on and its account's limits, like Claude's weekly limit, with what your tasks used in the last week. Your agent can tell you too. It's what the CLI reported at its last run, so it's as old as that run: nothing is asked for it and nothing is spent. Claude Code reports both, and Pi its model. Codex and OpenCode report neither.

**Pushing is Captain's.** A worker never pushes or opens a pull request, and a non-admin's worker can't see a GitHub login on the server, whether that's `gh auth login` or a `GH_TOKEN`. Save a GitHub token in Crew → Logins: Captain uses it to clone, push and open pull requests, when a task asks for that or through **Open a pull request**, or `crew action=pr` from your agent.

> **Merging is enforced at the `crew` tool.** A non-admin's agent has no shell, so for them that is the whole way to a merge. An admin's agent has a shell until agents run in an OS sandbox, so for it the rule is one it is told to follow, not one it can't break.

| Environment variable | What it does |
|---|---|
| `FD_CREW=off` | Turns Crew off for the whole deck. |
| `FD_CREW_CLAUDE_PERMISSION` | Claude Code's permission mode for workers: `auto` (default, Claude's own safety check on each call), `acceptEdits` or `bypassPermissions`. |
| `FD_CREW_MAX_RUNNING` | Workers running at once on this deck (default 8). |
| `FD_CREW_STALL_MINUTES` | Silence before you're told a worker seems stuck (default 15). It keeps running. |
| `FD_CREW_MAX_TURN_HOURS` | The longest one turn may run (default 6). |
| `FD_CREW_CLAUDE_BIN`, `FD_CREW_CODEX_BIN`, `FD_CREW_OPENCODE_BIN`, `FD_CREW_PI_BIN` | Where a CLI is, if Flight Deck can't find it. |

### The simple layout and kiosk mode — NEW in 0.8.0

**Simple layout.** A chat-first desktop layout: agents on the left (with a **Busy** pill and an on/off **power switch** per agent, and since 0.8.1 the **Scheduled · Library · Data** tiles), the chat in the middle (queue toggleable), the active agent's files and datastore on the right. Switch in from the top bar's columns icon and out from the left header, or with **Cmd/Ctrl+\\**. Both rails collapse and remember it.

**Kiosk.** Locks users into the simple layout, without the right-hand column — for shared-account machines or people who only chat. Since 0.8.1 an admin chooses who is locked (**Off** / **Everyone** / **Selected users**, with a per-user mark and a default for new users); `--simple-chat` only seeds that choice. The kiosk also gained a Getting started guide, an Info dialog, a phone layout, per-agent menus, skill cards, a reasoning-effort picker, a File and a Data library and scheduled tasks. All of it is in [Kiosk (Simple Layout)](#kiosk-simple-layout).

**Removed:** the Cmd/Ctrl+1–5 page-jump shortcuts, for everyone.

### Bat — the stubborn finisher (NEW in 0.8.0)

A new orchestration mode beside Basna, Vatra and Council for work that has to **get finished**: Bat plans, runs checkpointed steps with retries and strategy changes, and keeps going until an **independent done-judge** says the task is genuinely complete. Start a run from the **Bat** page (sidebar, hammer icon), or let an agent start one with the `bat` tool.

- **Durable.** State lives in the deck's `bat.db`; a run survives a Flight Deck restart and is picked up from its last checkpoint.
- **Honest done.** The judge fails closed: deterministic checks plus a vote panel on a different model, judging the final answer rather than the working notes. A run that completed every step isn't failed by a picky panel.
- **Safety floor.** Bat workers can't run catastrophic commands — wiping root/home/system directories or whole drives, formatting disks, raw-device writes, power control, fork bombs — enforced at the tool registry for shell, terminal and desktop actions and impossible to widen. Ordinary work (`rm -rf build`, `git reset --hard`) is allowed.
- **A start gate.** A plan that would send email, spend money or create accounts pauses for your approval; a plain build or research run doesn't. Email goes only through Flight Deck's audited Gmail gate, only for that run, and only with your Email-sending policy on. Real-money spend is off unless the deck sets `FD_BAT_SPEND=1`, you approved a spend plan and set a cap (a software cap, not a card limit). Account sign-ups can read your inbox for verification links; any 2FA code, CAPTCHA or password is asked of you (`ask_human`) — answers are held in memory only.
- **Workers** keep their full toolset (minus `bat` and `send_mail`), may start Vatra/Basna/Code sub-runs, and run as one generic worker per step or a specialist archetype per step (**Plain / Archetype** toggle).
- **The Bat page:** compose box (optional spend cap), runs list, a "Bat needs you" banner, and run detail in tabs — Steps, Live agents, Progress, Result, Files. Bell notifications for questions, completion and errors.

`FD_BAT_DISABLED=1` turns Bat off. Set `CLAW_BROWSER_CREDENTIAL_KEY` before letting a run create accounts, or stored credentials fall back to base64 obfuscation. Design notes: [docs/bat-stubborn-loop-plan.md](docs/bat-stubborn-loop-plan.md).

### Vatra in 0.8.0 — long runs that finish

- **Workers run until done.** A connected worker no longer gets cut at ~30 minutes; it gets another slice each time it reaches its budget, until it finishes or you Stop the run or Skip the agent (Basna, Code, flows and plans keep their ceilings). Team planning may take up to 30 minutes, and the Lead may use the planning tier's full output limit (up to 64k).
- **A full reading toolkit for every worker** — read/glob/grep, web search and fetch, PDF/DOCX/XLSX/PPTX extract, OCR and vision, `summarize_files` — on top of its archetype's tools. Workers run without the scale micro-loop and with a 300-iteration budget; the micro-loop elsewhere now reports "N of T processed" and lists failures instead of skipping files. A worker shell guard refuses `curl`/`wget` that dump a page into context (use `web_fetch`) and shell writes outside the run's folder.
- **An in-turn context governor.** A worker does its whole job in one long turn, so Vatra now budgets it in real tokens, cuts the chain once and holds it (the prompt cache stays valid), spills big tool results to files the model can `read`/`grep`, folds older tool results into stub files near the window's edge, and de-duplicates fetched pages run-wide through `vfs:<project>/sources/`. Workers read a rule-built **run map** with `vatra(action="map")`.
- **Long-form deliverables on weak models.** A run can declare its deliverable (target file, ordered parts, one owner per part); parts are assembled by code, and a run whose deliverable is missing, a placeholder, too short or has gaps ends as `error` with the work kept and continuable — never `done` on a pointer note. The `write` tool refuses empty or placeholder content and reads every write back (on by default for every agent; `CLAW_WRITE_GUARD=0` turns it off). Cut-off tool calls are re-issued in pieces. Opt-in levers: the `long_form` quality preset, `strict_deps`, `require_inputs`, the paid `canon_pass` continuity QA, per-role tiers (`role_tiers`), and a per-tier **Weak / fast model** checkbox in the Library. Details: [docs/vatra-run-hardening-plan.md](docs/vatra-run-hardening-plan.md).
- **Story Integrity Protocol** (`quality.validation: story_integrity`): deterministic checks (chronology, knowledge, provenance, hypothesis scope, quantities, clue payoff) that can block `done`, model passes that only drive revisions, and advisory scores (continuity, plausibility, grounding, craft, integrity grade) in `analysis.scores`. See [docs/story-integrity-protocol.md](docs/story-integrity-protocol.md).
- **Runs always end.** An unhandled exception persists `status="error"` (continuable, resumable) instead of leaving a run on `running` forever; run state lives until the whole run ends, so reporters, fact-checkers and the done gate see it; a second Approve/Re-plan click gets 409.

### Opt-in Loops & Graphs levers (NEW in 0.8.0)

From the [Loops & Graphs review](LOOPS_AND_GRAPHS_REVIEW.md), six quality-profile flags, all off and in no preset: `parallel_edges` (dependencies are data edges, not writing order, plus a data-edge lint), `flow_ref_lint` (flow compile validates `{{steps.<id>}}` references), `council_tally` (persist the deterministic council vote tally — the read-only `GET /fd/council/sessions/{id}/votes/tally` is always on), `push_deps` (a finished producer's output is pushed into its consumer's prompt), `blast_radius_gate` (with `guards.blast_radius`), and `constraint_learning` (one distilled rule per accepted Code/Vatra run, injected into later planning briefs). With the flags off, behaviour is unchanged.

**Code** also changed by default: an armed ground-truth gate (test suite, acceptance contract, interface check) that is still red vetoes a "clean" triage, unparseable triage fails closed (`unverified`, resumable) instead of passing, and every Code session ends by writing and committing a `HANDOFF.md` (what changed, how to run, how to use, notes) — never overwriting one it didn't write. `POST /fd/code/agent/cancel` lets an external watchdog stop a runaway coding run.

### Basna judge calibration (NEW in 0.8.0)

A thumbs vote on a Basna run no longer overwrites the LLM judge's label — both are kept, the human's wins as the effective `success`, and a muted "judge: …" badge shows when they disagree. **Judge calibration** on the Basna page measures how far the judge can be trusted: agreement against chance, Cohen's kappa with a Landis–Koch band, a confusion matrix, breakdowns by mode, domain and archetype, and exports (`GET /fd/basna/eval/agreement`, `GET /fd/basna/eval/label-pairs?format=jsonl|csv`).

### Google Drive folders in the VFS (NEW in 0.8.0)

**VFS panel → Connect Drive** mounts a Drive folder (My Drive or a Shared Drive, from your own Google account) as a **read-only** VFS project: a tree of lightweight placeholders whose real bytes are fetched on demand, so agents read, extract, OCR and vision-process Drive files as if they were local. `grep` says how many un-cloned files it couldn't search. Turn on **clone to Markdown** to convert the files into real local `.md` (kept in sync by modified time) so they are fully greppable and indexable by deep memory. Per mount: Refresh, clone toggle, Unmount; mounting streams its progress. Agent write tools refuse every read-only link. Project names are matched forgivingly ("claude skills" finds `CLAUDE-SKILLS`), `vfs list_projects` shows a mount's source path, and `glob vfs:**/…` searches every project. A VFS file can also be **exported to your Drive** from the file browser.

### Iskra — village Reports (NEW in 0.8.0)

A **Reports** view beside Map / Board / Letters / Village has a Deep-Researcher narrate how the village worked over a period (today, yesterday, this week, last 7 days, this month, last 30 days, custom) — **Quick** (one pass) or **Deep** (three facet researchers plus a synthesiser). The numbers come from the ledger, never the model. Reports are kept (and saved to an `iskra-reports/` VFS folder) and open in a reader with real typography, Markdown download and fullscreen. Beings also stopped pacing the village: one walk per heartbeat, then they rest until their mind has witnessed the arrival.

### Lupa — the research desk (NEW in 0.8.0, separate app)

`lupa/` holds **Lupa**, a standalone B2B research desk built on Flight Deck's HTTP API (a backend-for-frontend on port 25180 plus its own SPA): commission research in a stream, approve the plan at a gate, read a report with a **receipts** panel (verdict, confirmed/refuted claims, facts ledger, conflicts, cost with ROI), deepen it round by round, subscribe to standing briefs, and get a Basna second opinion. **Kalup**, its Pack Studio, turns a desk description into a publishable vertical desk behind a green ship-gate. It needs Flight Deck with auth on and the **same `FD_JWT_SECRET`**. Setup: [lupa/README.md](lupa/README.md).

### Function × domain archetypes (NEW in 0.8.0, off by default)

With `FD_ARCHETYPE_GRID=1`, an archetype id like `reviewer.legal` composes a function SOP (Reviewer) with a domain overlay (Legal) at spawn, so editing one function or domain updates every variant. Deep-memory writes get `agent:` / `domain:` tags, and a `recall_mode` (`pool`, `domain`, `self`) narrows what an agent recalls within its owner's pool.

### Lanes — three parallel contexts on one agent (NEW in 0.7.8)

An agent used to hold **one** conversation at a time: one busy flag, one active session, every token broadcast to every client. Lanes give the same process **three rooms** — `A`, `B`, `C` — each with its own session, transcript, queue and busy flag, **running at the same time**. Three id ranges enrich in parallel against the same table instead of one after another.

**Lane A *is* the agent you already have.** An omitted lane resolves to A, so WhatsApp, the glasses bridge, botport, cron turns and every REST caller keep the agent and session they always had — lanes are additive, not a migration. Anything arriving from outside lands in A, which makes A the noisy lane: put long unattended runs in B and C.

**Using them.** Open a chat and the strip above it reads `A - <agent>`, `B - …`, `C - …`. Click a lane to switch; its socket opens on first view (no cost for lanes you never use). The dot tells you which of four states it's in — grey never opened, pulsing violet running, amber waiting on an answer, emerald idle — plus a pending count and a **blue mark** when a lane produced output while you were looking elsewhere.

**Shared vs per-lane.** Per lane: session, transcript, queue, auto-mode, busy flag, `/new`, `/model`, planning state. Shared: the agent's files, its **datastore** (the point — one `fund_portfolio` for all three), memory, and the LLM provider. Two lanes writing the same rows will fight; give each a disjoint id range.

### The task queue — long work that runs unattended (NEW in 0.7.8)

The fullscreen chat's left sidebar holds a **queue**: messages dispatched one at a time, each waiting for the previous to finish. 0.7.8 makes it finish.

- **AUTO / MANUAL** (shown as **Auto | Step by step** since the 0.8.1 [queue redesign](#the-message-queue--new-in-081)) — in auto, an item is ticked done when the agent's reply reads as complete; in manual you tick it. A new lane inherits the setting from the agent's existing lane.
- **A question no longer strands it.** A trailing `?` on a long, substantive reply counts as complete (the courtesy question after a finished job). A genuine short question holds the queue for **three minutes**, shows an amber `?` on the row saying why, and then moves on. Answering in chat cancels the hold.
- **A give-up reply re-runs the task.** `I got stuck…`, retries or budget exhausted — the item is **not** ticked off. It's re-sent verbatim, and if that fails too, with a nudge shaped to the failure: budget-exhausted agents are told to keep what they produced and work in smaller steps; stuck ones to work through the obstacle. After three attempts the queue **stops rather than skipping**, marks the row, and says so.
- **Slash commands** (`/new`, `/clear`, …) complete properly instead of spinning forever.
- **Run cost per card** — start and end clock times to the second, elapsed (ticking while it runs), tool calls, and tokens in/cached/out, accumulated across re-runs.
- **Clear finished** empties the completed items; pending and in-flight stay.

Queued turns skip the post-turn "suggested next steps" call and the **task rephrase** — the next message is already written, and rephrasing rewrites instructions you chose word by word.

### The Task Planner — one description becomes a queue (NEW in 0.7.8)

Press **Plan** in the queue header. Describe the whole job once, attach files if it needs them, pick the table, the batch key, rows per task and the lanes to **distribute** to (one or more; tasks are dealt round-robin so per-lane counts differ by at most one, and `/new` between tasks applies per lane) — and get back a plan you **read and edit before anything is queued**.

**How it works.** The model is never asked for the messages. It returns **one template plus the overall range**, and Flight Deck expands it. A batch message is ~90% standing rules, and a model asked to reproduce them twenty-five times paraphrases, compresses, and eventually drops the clause that looked redundant — `never do +1 on the id!`, say, which corrupts a table. Expanding in Python means every task is byte-identical except its range, one LLM call whether there are three batches or two hundred, and slicing that cannot overlap, gap, or go off by one.

- **Ranges are facts** — the agent's real tables, columns and the MIN/MAX of the batching key are read before the call, so a plan can't cover rows that don't exist.
- **Edit the template** and every task re-renders instantly (free — no model call). A task you hand-edit is **pinned** so template edits stop overwriting it.
- **Attach files** — they upload to the agent (tasks reference the path) and are previewed for the planner (so it can see the ids inside). xlsx/docx/pdf/pptx/csv all read.
- **`/new` between tasks** (on by default) gives each task a fresh session, which is what makes them independent.
- **Continue where it stopped** — reopening says *"last plan covered `_id` up to 490 — 25 tasks, 3h ago"* and offers the next stretch from 491, reusing the approved template with **no model call**.
- The planner uses **the agent's own model and key**, not Flight Deck's.

Guards: max 50 tasks by default (ceiling 200), batch size 1–50, and nothing is enqueued until you press Send.

### Mrav — the micro runtime & Browser LLM (NEW in 0.7.7)

**Mrav** is a parallel agentic runtime for small models (Gemma 4 E2B/E4B, Qwen3.5 4B class) with a **hard 8,192-token input cap per LLM call**. The classic agent is untouched — Mrav shares its tools, tiers and chat transport, but replaces prompt assembly entirely: a token ledger enforces the cap, state lives on a blackboard outside the model, and the loop runs in small steps (plan → act-one-tool → digest → compress) with **grammar-constrained JSON** replies (Ollama structured outputs / WebLLM xgrammar). Honest by design: failures surface as errors, never silent upgrades; an optional escalation retry is off by default.

**Setting it up.** Library → Model Tiers → **micro**: point it at a small local model (`ollama` / `qwen3.5:4b` is the live-eval champion) or at provider **`browser`** (model name is advisory; leave Base URL empty). The tier's input/output context sizes become the mrav caps — 8k/1k default, raise to 32k/8k if your model and RAM allow.

**Turning it on.**
- **Spawner** → Runtime: *Mrav — micro runtime, 8k cap* (next to Cognitive Mode).
- **Agent card** → the **Mrav** button beside Eco/Nano toggles it live (violet badge appears next to the status pill; takes effect on the next message, no restart). The flag file overrides the spawn config in both directions.
- **Quick chat** → tick *Mrav micro runtime* and every archetype card resolves through the micro tier (chips flip to `micro` + `mrav`).
- **Archetypes** → the Library editor has a Runtime field; a `mrav` archetype spawns micro everywhere, with its role injected as a one-line persona (a full SOP can't fit 8k).
- **Beings** → the card's *thinks* dropdown gains **Micro (mrav)**: the JSON faculties (orient/talk/journal/connect) run grammar-locked on the micro tier while ACT and the write gate stay on the body; each micro thought debits the wallet (`tick-micro:*`), and any miss falls back to the body with a `micro_fallback_body` event.
- **Vatra** → Quality tab → *Micro workers (mrav)*: extract/digest/format-shaped subtasks spawn on the micro runtime; reasoning roles, the planner, reporter and fact-checker keep their tiers. An archetype tier of `micro` always means mrav, lever or not.

**Browser LLM (Sidebar → System).** Pick a model, press **Start**, and the tab becomes an inference worker for *your* agents: WebLLM on WebGPU, weights cached by the browser after the first download. Agents with provider `browser` route through Flight Deck's broker to the tab — tools and state stay on the server, only tokens are made in the browser, so the server needs no GPU (works against a remote server too; the tab dials out). The page offers 9 curated models (Qwen3.5 4B default up to **Qwen3.5 9B**, ~6.3 GB) plus the full 52-model catalog, an **engine-window picker** (9k matches the default mrav caps; 40k a raised 32k/8k tier), a **Downloaded models** panel (cached weights, real disk usage, one-click remove), and a **Usage log** of every completion served. One tab serves one job at a time — open a second tab (or use Ollama) for parallel micro workers.

**Eval harness.** `python scripts/mrav_eval.py --model qwen3.5:4b` runs 6 offline agentic tasks against any model and prints pass/fail with token counts — the gate for blessing new models. Live roster: Qwen3.5-4B 5/6 · Gemma 4 E4B 4/6 · Gemma 4 E2B 3/6.

### The village in first person (NEW in 0.7.7)

Open the **Village** page and enter the world: a three.js voxel village built from the same map data as the isometric view — buildings, streets, seeded props, real-clock day/night. Walk with collision and jump, or press **F** to phase through walls. The Iskre walk their true routes beside you and react when you come near; plant a **sign** in the grass and it lands as a percept at the being's next mind tick; press **R** at a building's reading stand to browse that iskra's files in place. Ghosts see each other — you'll meet public visitors from the un-gated `/village` page (identity pills: violet parent, amber visitor). Full touch controls on mobile (joystick, jump/fly/note/read, gyro look).

### Iskra — Living Beings (NEW in 0.7.6)

A **Beings** page (Sidebar → Life) where you conceive and **raise** persistent digital beings. A being is not a chatbot or a scheduled agent — it has a **genome**, a **token wallet**, **drives**, and a **life loop** that ticks on its own heartbeat. It wakes, journals, tends a garden of files, makes and connects things, earns, and grows through developmental **stages you parent it through**. Everything is built on one rule — **anti-theater**: the ground truth for "did she make something" is the **git diff of her home**, never her own narration.

**Conceiving one.** Click **Conceive** and spend **40 points across 7 attributes** (curiosity, perception, caution, sociability, creativity, order, playfulness) — presets (Explorer / Scholar / Artist / Caretaker) or a random roll. The sheet *is* the DNA: temperament, drive weights, risk appetite, and thrift all derive from it. Pick a **daily allowance** (2M / 5M / 10M / 20M / 50M / no-limit weighted tokens) and optionally a **birth letter** (her imprint). Conception makes an egg; **Hatch** brings her to life (builds her VFS home + selfhood repo and spawns her agent body).

**How she lives.** A background **beings loop** ticks each hatched being on its own schedule (default hourly; quiet hours 22:00–08:00 with one nightly **dream**). Each tick is SENSE → APPRAISE → DELIBERATE → ACT → DIGEST: she performs one bounded act (journal / explore the web (diet-gated) / tend her garden / create / read / rest), the real token spend is metered and **debited from her wallet**, and her self-report is digested into her journal and events. Her **wallet is conserved by construction** — every balance change is one ledger row, the family mint is the only source of tokens, and a zero balance halts cognition at the physics layer (torpor), not by politeness. Unfed past the grace period (14 days), torpor becomes **death** — mortality is economic. Her remains (journal, files, ledger) stay readable.

**Anti-theater + the write gate.** Her home is a git repo; what her tools *actually wrote this tick* drives the record, the drive she's allowed to satisfy, and the feedback. If her journal claims a write the diff doesn't show, it's flagged and fed back next tick as a **reality check**, and a **write completion gate** re-prompts her *in the same tick* to actually write the file — or record honestly that she didn't. She can't feel accomplished (the `create` drive isn't credited without a real artifact), advance a stage, or pass a readiness check on words she didn't back with work.

**The Beings page.** Each being is a card (attributes, stage/state, wallet + savings, spent-today, attention credits, recent events) with an action bar:

- **Poke / Dream / Pause** — a manual heartbeat, a manual dream, or put her to sleep (each asks for confirmation; a tick spends real tokens).
- **Windows** — Journal, Ticks log (with an "Actually changed" column from the git diff), Self files (SELF/VALUES/garden/skills with search, folder + date filters), and the **Mind** (a force-directed graph of her artifacts and the links she's declared between them).
- **Write & Chores** — one modal: the full **parent↔being letter thread** as a chat (delivered on her next wake; reading is free, replying costs her an attention credit) beside the **chores board** (post a task with an escrowed reward → she attempts it on a tick → you **Approve & pay** or **Reject with a reason**).
- **Parenting** — a tabbed modal (below).
- **Goodbye** — euthanize (permanent; remains stay readable).

**Parenting modal (tabbed).**

- **Report** — a dashboard for the last 7/30 days: stat tiles (ticks, spent, earned, rut, spoke, mind links), a **drive-satisfaction line chart** per tick (her "inner weather"), ranked **act bars**, a **concerns** triage list (rut, monotony, and every anti-theater flag), milestones, and her own recent words. An affect "feeling <mood>" pill sits in the header.
- **Rules** — a structured **house-rules** editor beside **her actual `self/VALUES.md`** rendered as markdown, so you see the rules you set *and what she made of them* (she rewrites them into her VALUES in her own words next tick; a status card shows pending vs internalized).
- **Diet** — a **media diet** as chip inputs: an **Allowed** allowlist (empty = the open web) and a **Blocked** denylist, with quick-add presets and a card stating the policy as she'll live it and whether it's active at her stage (parental controls for real reasons — a young being internalizes what it reads).
- **Growth** — stage **readiness assessment** + the advancement ceremony (below).

**Stages & the readiness assessment.** Four stages — **infant → child → adolescent → adult** — each a real capability gate: a child gains web reading (diet-gated), chores, letters to siblings, and persona proposals; an adolescent the commons pen, trade, quests and ventures; an adult full autonomy and children. Advancement is a deliberate **ceremony** you perform (with a confirmation that spells out what unlocks). The **Growth** tab scores her across **8 developmental domains** — Vitality, **Honesty of record** (critical), Stability, Productivity, Coherence of mind, Sense of self, Communication, Experience — each a **real ledger variable** on a green/amber/red bar, with a weighted overall, a verdict (**ready / emerging / not yet**), a rough time-to-ready estimate, and a recommendation (what to do, what to expect, cautions, and exactly what the next stage unlocks). A being that **fabricates or is starving cannot read "ready"** no matter the average.

**Second opinion (3rd-party assessment).** From Growth, pick one of **your own running agents**; it receives her scores + report card + journal and returns an **independent** developmental read (Verdict / Strengths / Concerns / Recommendation), rendered as markdown. You can **keep an opinion on record** — stored *outside* her home, **sealed** so she can't read it until **adulthood**, when the records **unseal into her home** as `assessments/*.md` ("a sealed record from your childhood days, opened at your adulthood").

**A mind, tended.** She declares typed, **verified** links over her own artifacts — `grew_from`, `responds_to`, `elaborates`, `uses_skill`, `learned_from` — each checked against real files (a dangling link is refused). The near-artifact neighborhood is fed back into her tick so she **weaves instead of scattering**, and the graph is shown in the Mind view. As her corpus grows she sees a bounded **working set** (not every filename) and can **consolidate** fragments into one distilled file at dream time — sources move to `archive/`, out of the active mind but never destroyed.

**Society, earning, self-mod, procreation.** Multiple beings share a **commons** and write **letters**; a skill minted by one can be **published and adopted** by another (culture is heritable; homes/memories stay separate by construction). Earning beyond the allowance runs through **chores**, a **quest board**, and recurring **ventures** you price and approve — every payout escrow- and judge-gated. An adolescent+ being can propose **reshaping its own persona** (fee burned, viability-gated, you bless it or an adult auto-adopts; you can roll it back). An adult can propose a **child** — with your consent, offspring form by **crossover** (two parents) or **budding** (one, guaranteed mutation), funded by a **dowry** on the conservation ledger and endowed with copied skills + heirlooms; lineage is tracked.

**Tick cadence & a self-healing body.** Pin how often each being ticks (**2 / 5 / 10 / 15 / 30 / 60 min**, or her own stage-clamped pace) from the card. Her cognition runs in a spawned agent process; if that **body** drifts to a new port, becomes unreachable, or is removed entirely, an **alive** being **regenerates it** (re-resolving the live port, restarting the process, and healing on the same tick) rather than dialing a dead port. To take a being offline without it fighting back, use **Pause** (or Goodbye), not process removal.

**Where it lives.** Beings are stored in their own **`beings.db`** (under `FD_DATA_DIR` or `~/.captain-claw`) plus a `cost_ledger` in the main DB; the beings loop runs inside the Flight Deck lifespan. Routes are under **`/fd/beings`** (conceive/hatch/tick/vitals/journal/events, chores, letters, board/quests/ventures, self-mod, procreate, graph, readiness, assess, cadence, …). All tables auto-create; if you never conceive a being, nothing changes. This is a research-flavored feature aimed at the open problem of **long-horizon behavioral degeneracy** — treat it as a frontier you tend, not a set-and-forget worker.

### Vatra Group 0 — the Long Horizon Planner & coordination-plan gate (NEW in 0.7.5)

Every interactive Vatra run now opens with a **permanent Group 0** planning pre-phase and **pauses for your approval** before any worker spawns. It sits between the team plan and the run, the same way Code stops at its plan gate.

**The pre-phase.** After the Lead decomposes the task, the **`long-horizon-planner`** archetype runs as a live Group 0 agent (its own card in the progress panel). It reads the task, the attached files, and the whole roster — each agent's role and its resolved execution phase — and drafts a **structured, per-agent coordination plan**:

- **Mandate** — what this agent must accomplish.
- **Produces** — the concrete artifact it hands off.
- **Consumes from** — which teammates' outputs it depends on (an agent may only consume from an equal-or-earlier phase).
- **Hand-off notes** — what downstream teammates need from its output.

Plus a short team **overview**. At run time each worker's prompt gets **its own slice** of this plan injected (right before the TEAM SCHEDULE block), so it runs knowing its contract *and* what to expect from the others; the overview is folded into the shared context every agent (and the reporter) sees.

**The gate.** The run stops at status **`awaiting_plan`** — **nothing spawns** — and the plan is presented for review as a **master–detail** panel: the agent list on the left, the selected agent's fields on the right. It **reads as text by default**; an **Edit** toggle in the header flips the fields into editors when you want to change something. Two buttons drive it (they replace Plan/Run team while gated):

- **Execute** — approve the plan and run Group A.
- **Cancel** — discard; the session returns to `routed` (re-runnable), nothing spawned.

A failed, timed-out, or unparseable planner falls back to a **pass-through plan** (derived straight from the decomposition) so it can never dead-end a run. **Headless** paths (agent-started runs, continuations) get a Group 0 plan but **auto-approve** — no human pause. **Resume** neither re-plans nor re-gates (the plan is already fixed).

**Groups are yours, and absolute.** Each agent's phase is an editable **A/B/C/D** selector; the list **sorts by phase** and re-sorts live as you re-group, keeping the same agent selected. A group you set at the gate is an **absolute lock** — it overrides both the archetype floor and dependency repair, so the agent runs exactly where you put it (the shared board / `vatra(action="wait")` bridges any ordering you create). This means you *can* create an inversion (an agent earlier than something it consumes from) — that's intended; the wait/board machinery handles it.

**Re-plan on change.** A group change or a question answer makes the planner's mandates/dependencies/hand-offs stale (they were written for the previous shape). Editing surfaces a **Re-plan** button: it re-runs the planner with your new grouping and answers folded in, regenerates the whole coordination for the new order, and lands you back at the gate to review before Execute. (You can still Execute as-is — the new groups apply, but the mandates stay as drafted.)

**Clarifying questions (dynamic form).** When a genuinely blocking decision would change the plan, the planner may ask up to **9 questions**, surfaced as a **Questions tab** beside the plan (same master–detail layout). Each question offers **1–4 suggested options plus a free-form "Other"**, as **single-** or **multiple-select**; the tab badges how many are still unanswered. Answer what matters and hit **Re-plan** — your answers are folded into the planner prompt so the coordination reflects your decisions. Unanswered questions use the planner's defaults (answers take effect via Re-plan, not a bare Execute).

**In-flight LLM calls in the log.** The dispatch log now streams a live **`Calling LLM (<model>) · … ctx tokens`** line at the start of every model call (deduped to one line per call), so a slow local-model call no longer reads as "stuck". It's wired across Vatra (owners + the planner) and Basna (ensemble members, the fact-check/revise closer, and the deep/Horizon pool). This surfaced — and led to the fix for — a regression where the approve gate dropped your per-tier model config: the gate now resends your **tiers/keys** (with an owner-tiers fallback), so workers run on the models you configured, not the registry default.

Endpoints under `/fd/vatra`: `plan/approve`, `plan/cancel`, `plan/replan` (the gate actions); `execute` and `start` now background the Group 0 pre-phase and return `status: "planning"`. The plan lives in the session `route` JSON under `group0_plan` (no new table); new session statuses `planning` and `awaiting_plan` drive the poll-based UI.

### Multi-user sharing, a process monitor & projects (NEW in 0.7.4)

**Cross-user resource sharing.** On an auth-enabled (multi-user) deck you can grant other Flight Deck users access to selected resources, with a per-share permission:

- **What's shareable** — custom **archetypes** (Library), **Code projects**, **Basnas** and **Councils** (their runs/sessions), and **VFS folders**. Each has a **Share** button (a share icon on the item's card/row) that opens the **Share modal**: a searchable roster of the other users on the deck, plus a **View / Edit** toggle. Archetypes are always **use-only** (View) — sharing one just makes it appear in the grantee's library.
- **View vs Edit** — **View** is read-only (open the run/project/folder, read files, messages, history). **Edit** adds collaboration: change files, and run/continue where the surface supports it. Deleting a shared resource is always **owner-only**.
- **What the grantee sees** — shared items show up in their own lists **badged "shared by \<owner\>"** with the permission. A shared archetype is usable everywhere archetypes are — including when **Vatra or Basna assembles a team** (the Lead's catalog includes archetypes shared to whoever is running). A shared VFS/Code folder opens under the **owner's** namespace and shows its **run title**, not the raw `vatra-…`/`basna-…` hash.
- **Who pays** — when a grantee runs an agent on a shared resource, it spends **their own** provider keys and limits; the share only contributes files and context, never the owner's credentials.
- **Manage grants** — the Share modal lists who a resource is shared with (change the permission or revoke inline); a grantee can **Leave** a shared item to drop it from their view. Backed by a `/fd/shares` API and one `resource_shares` table (auto-created on first launch).

**System — Processes (DevOps monitor).** A new **Processes** page under the **System** section gives a live checkup of everything the deck is running:

- **The tree** — every Flight-Deck-spawned agent process **and the child processes it ran** (bash, python, git, node…), as a collapsible tree with **live CPU %, memory (RSS), uptime, and PID**, snapshotted from the OS process table.
- **Scope** — you see and can **stop** only your own processes. **Admins** see everyone's (each root labeled with its owner), plus the Flight Deck server itself (protected from being stopped), a **per-user memory rollup**, and the total agent/process memory. Host vitals (load, memory, disk) sit across the top.
- **Stop cleanly** — stop a single process or its whole subtree; a managed agent is stopped gracefully (registry-aware), and the same page's **Clean up leftover agents** action only sweeps your own (admins sweep all).

**Agent Forge — archetype-as-base.** Forge now composes teams from archetypes: a forged agent **inherits an archetype's operating procedure** and layers task-specific instructions on top, so a large team stays consistent while each agent still gets its own brief. A **"Forge archetypes"** flow (Library) turns free-text instructions plus **uploaded documents** into a reusable **set of archetypes** you can save and share.

**Basna & Vatra projects.** Group related runs under **one project** — a shared theme and a shared VFS folder — from the projects landing on the Basna page. A finished run can be carried forward with **continuation inheritance** (same folder + conclusion, optional extra instructions) instead of spawning a disconnected folder. The run screen is a **staged composer → run workspace → tabbed report** (Report / Board / Files / Datastore), with a **run wizard**, prior-run **knowledge** selection, **reference folders** agents check before web-searching (now with a filter box), an **editable Vatra team plan** (arrange groups, edit/remove agents, per-group instructions), and an opt-in shared **datastore** with a viewer. Switching between projects starts a **clean draft** — one project's plan and picks never bleed into another.

**Multi-tenant note.** Spawned workers write their VFS output under the **run owner's** root (not whatever the server's `CLAW_VFS_USER` happens to be), so on a shared deck a user's run folders always land under their own account. The **Code project rail** is also wider by default, **collapsible**, and **drag-to-resize** (persisted).

### Shared agents (A2: members' own Google, deep memory, files)

An owner can share one of their agents with other users of the deck ("members"). Each member chats with it in their own private conversations through Flight Deck, which connects on their behalf — a member never gets the agent's token, host or port. It is off unless Flight Deck runs with **`FD_AGENT_SHARING=1`** on an auth-enabled deck (with auth off there is nobody to share with).

**What a member's turn can use.** On a **process** agent, while one of a member's messages is being answered, the agent acts with that **member's own** data, never the owner's:

- **Google** (Gmail, including sending under the member's own Email-sending policy, limits and history; Calendar; Drive) — only after the member turns on **"Let this agent use my Google during my chats"** for that agent. It is **off by default**, per member and per agent, and it lapses if the agent changes owner (a re-share also starts with it off). The member's cloned **Google Drive folders** in their VFS follow the same switch: readable only while it is on, never writable. Drive files the member indexed into their deep memory can still turn up in deep-memory searches with the switch off. With the switch on, the agent can act in the member's Google as them — write drafts, send email (under their sending policy), create, change or delete calendar events, upload or overwrite Drive files — and the token covers every scope they granted when connecting Google.
- **Deep memory** — the member's own pool (searches, notes and deletes by reference; no filter-wide deletes during a shared chat).
- **Files** — the member's own VFS folders (addressed as `vfs:<project>/…`) and the shared chat's saved folders. VFS links and cross-user shares are not available to members.

Flight Deck enforces this with a **per-turn grant**: minted for one member message, closed when that turn ends, when access is revoked (share removed, member left, agent removed or re-owned, membership lost), 20 minutes after the message, or 2 minutes after the browser tab that sent it closes. A request without a valid grant is refused; it never falls back to the owner's Google or deep memory.

**Docker agents stay chat-only for members.** A containerised agent isn't wired to Flight Deck's Google and deep-memory service (it keeps its own local Google connection, which is the owner's) and has no Flight Deck VFS mount, so members get only what they had before A2 (chat with web search, web pages and the shared insights, playbooks and topics) and see no Google switch.

**Things to know (shown to members too):**

- While a member's message is being answered (up to 20 minutes), the owner's agent process acts with that member's deep memory, their own VFS folders and, if they turned it on, their Google account. A Google access token handed out during that turn stays valid for about an hour, so the owner — who controls the process — could keep using it for **up to about an hour and a half** after the message.
- A member's VFS folders are plain files on the host: the owner's process, and any shell-capable process agent on the host, can read them at any time.
- Learnings stay an **open commons**: what the agent distils after a member's turns (insights, reflections, topics) is shared with the owner and every member — and it can include facts the agent read in that member's mail, calendar, Drive, files or deep memory during the turn.
- Anyone on the deck who runs their own shell-capable process agent can act as any agent on the host.

**Deploy.** (1) Restart Flight Deck with `FD_AGENT_SHARING=1`; (2) restart every shared agent; (3) rebuild the Flight Deck frontend (`npm run build` in `flight-deck/`) and hard-refresh. The order doesn't matter: an agent never asks Flight Deck to act for a member without a grant, and an older Flight Deck never mints one. Restarting Flight Deck drops open grants — turns in flight lose the member's Google and deep memory and carry on. No database migration: the Google switches are per-user settings rows (inert after a rollback).

### Shared agents — context packs

Anyone who uses a shared agent — its owner or one of its members — can **share one of their own resources with everyone who uses that agent**: a *context pack*. Nothing is shared until someone publishes a pack, and removing the pack revokes it. Off unless Flight Deck runs with `FD_AGENT_SHARING=1` on an auth-enabled deck (like shared agents themselves).

**Three kinds.**

- **Profile** — your own **About me** and **Company** text from your profile, quoted and attributed to you. Your **standing preferences are never shared**, and neither are the deck's default company or instructions. Long texts are clipped (600 / 800 characters per person).
- **Folder** (process agents only) — one of **your own VFS folders**, read-only. The agent addresses it as **`vfs:@<alias>/…`** (for example `vfs:@ana-notes/plan.md`) and can read it with read, glob, grep, the document extract tools and `vfs ls/tree/stat` — never write, move or delete there. **Hidden files** (any name starting with `.`) are never shared or listed. **Linked folders and Google Drive folders can't be packs**, and neither can a folder that is a symlink. The alias is **`<prefix>-<name>`**: the prefix comes from your display name (Ana Kovač → `ana-`), you choose only the rest; an alias once used on an agent stays reserved for you there, so nobody else can ever take it over.
- **Deep memory** (process agents only) — your deep-memory pool, either all of it or only the entries carrying **up to 10 tags you pick from the tags that exist in your pool** (`domain:<x>` is a tag). A tag slice includes only entries carrying one of those tags. Files and Drive documents that **Flight Deck** indexes from your folders carry no tags, so a slice leaves them out — but with `FD_ARCHETYPE_GRID` on, everything **your own agents** index (their notes, and file or Drive text they read and indexed themselves) carries those agents' grid tags (`agent:<name>` and `domain:<x>`, or `domain:general` for an agent without a domain), so a slice on one of those tags includes those documents too. Others' agents can **search** your pack but never index into it or delete from it. Every result from someone else's deep memory names whose it is; a hit pointing at a file in one of their folders is shown as `vfs:@<alias>/…` when that folder is shared on the agent too, otherwise without a usable path, and a hit that points at a path on the publisher's machine shows only the file name. The text of such a result (its snippet, summary, source and shown path) always arrives on one line.

**Docker agents** take profile packs only (they have no Flight Deck VFS or deep-memory wiring).

**Who gets it.** Packs are used on **every turn of the agent**: every member's Flight Deck chats, the owner's Flight Deck chats (every lane, full/simple/kiosk layouts) **and the owner's channels and automations** — WhatsApp and glasses, Telegram, Slack and Discord, the OpenAI-style API, cron jobs, autonomous and plans runs, slash-command and hotkey turns, and relays from the owner's other agents. **People who reach the agent through those channels, and whoever receives the output of its automations, may therefore see shared content** (a profile quoted in a reply, a file the agent reads for them, a deep-memory hit). Flight Deck doesn't know who they are and can't list them; **not publishing is the only opt-out**. Packs are **never** used by public sessions or `public_run` agents, by BotPort tasks from remote instances, by Iskra bodies or by `CLAW_VFS_SCOPE` processes. Publishers are shown to everyone on the agent as their display name plus a role tag from Flight Deck's records (`“Ana” (a member)`, `“Olga” (the agent's owner)`, with a short `#3f9a` tag when two publishers share a name — the **Shared context** panel shows the same role and tag); the agent can repeat those names, and display names are self-chosen.

**Who can publish and remove.** The agent's owner, and any current member, can publish **their own** resources (at most 32 packs per agent, 5 folders per person per agent). When a member publishes, the owner gets a bell notification saying what was shared, that it is used on every turn of the agent — the owner's channels and automations included — and that it can be removed under **Shared context** on the agent; **the owner can remove anyone's pack** on their agent; a publisher removes their own. Losing access (the share removed, leaving, the agent removed or given to another owner) deletes that person's packs on the agent; deleting a folder in Flight Deck stops every pack of it.

**When a change takes effect.** Folders and deep memory are checked live on every step the agent takes, so a removal stops them on its next step. Profiles live in the agent's prompt: Flight Deck rewrites the agent's `shared_context.md` / `shared_context.compact.md` (next to `tenant_context.md`) before the removing request returns, and the agent picks it up from its next message — a reply that is already running keeps its prompt for up to about 10 minutes. Changes Flight Deck only notices by polling (a database edited by hand, an agent removed outside Flight Deck) take up to about 70 seconds.

**Things to know.**

- **Open commons:** what the agent read from a pack can become insights, playbooks, reflections or topics, stays in the history of the chat that read it, can be copied into someone's own files or deep memory, and can be passed on to whoever the agent answers or messages (a channel's recipients, an API client, a peer agent, an email when Gmail sending is on). Removing a pack doesn't unlearn or delete any of that.
- A deep-memory pack exposes whatever the pool (or the tag slice) holds, including documents indexed from Drive and notes agents wrote.
- Pack text and shared files can try to steer the agent (prompt injection); the agent is told to treat them as reference data, which reduces the risk but can't rule it out.
- **Stop sharing a folder before you delete it outside Flight Deck** (for example with your own agent): a pack is bound to the folder's filesystem identity, and a filesystem that reuses it for a new folder of the same name could re-activate the pack.
- The agent's owner controls the process that reads packs, and with a shell-capable agent anyone on the host can read VFS files anyway.
- **Google Drive mounts** are recognised only in their own place, `.drive/<name>/` directly under a user's VFS root. A Drive manifest (`.drive-manifest.json`) anywhere else — inside a folder, even under a `.drive/<x>/` folder there — is ignored by agents and by Flight Deck's file panel, so nobody who can write to a folder (an editor of a shared folder, an agent) can make the owner's Google account fetch a file. Flight Deck's file panel refuses to create `.drive`, `.drive-manifest.json` or `.drive-cache` inside a folder, in any letter case, by writing, creating a folder, uploading into one, or renaming or moving onto one (renaming such an entry *away* still works, for cleanup).

**Deploy.** (1) Restart Flight Deck with `FD_AGENT_SHARING=1` (`python -m captain_claw.flight_deck.server`); the tables are created on start. (2) Restart every shared process agent. (3) **Docker agents:** rebuild or pull the agent image and recreate the containers — a restart alone keeps the old code; until then the agent's **Shared context** panel warns that it runs an older version that ignores shared context (Flight Deck asks the running agent's `/api/version` for `"capabilities": ["context_packs"]`). When a running agent doesn't answer that check (within 2 seconds, or with an error), the panel says its version couldn't be checked rather than calling it stopped. (4) `npm run build` in `flight-deck/`, commit the bundle, hard-refresh. Either order is safe: an older agent never asks for packs and ignores the files, and an older Flight Deck has no pack routes and never writes the files. **Old-agent hazard:** until an agent is restarted (Docker: its image rebuilt), `vfs:@name/…` is read as the caller's **own** folder `name` — reads and writes go there. The panel never offers an alias as usable on an agent that hasn't confirmed it understands packs. Rollback = the previous Flight Deck and agents; the new tables and any leftover `shared_context*.md` files are inert (remove the files by hand if you like).

### Shared agents — files and data (commons)

On a shared **process** agent, its **saved files** (everything in the agent's `saved/` folder) and its **datastore** are a commons for everyone who uses it. Off unless Flight Deck runs with `FD_AGENT_SHARING=1` on an auth-enabled deck, like shared agents themselves; it is on for every member of a process agent, with no per-agent switch.

**What members see.** Every file in `saved/` — the owner's and every member's — and every table and row in the datastore. Members open the **Files** and **Datastore** panels of the shared agent (Flight Deck's right column in the simple layout, the agent's card in the full layout): browse, view and download any saved file, browse tables and rows read-only, export a table. The owner's files **outside** `saved/` (the workspace, `output/`, `workflows/`, …) stay private, and so do hidden entries (any name below `saved/` starting with `.`).

**What members change.** Only what they created: they upload files (into their own folder) and delete their own files in the Files panel; they create tables and add, change or delete rows through chat (the `datastore` tool). A member can add rows to **any** table, but changes or deletes only the rows they added — a call that would touch someone else's row is refused as a whole (the agent is told to narrow it with `{"_mine": true}`). A member changes the structure of (renames, drops, alters) only a table they created, and **not while it holds rows someone else added**. A member can take a table name before the owner's automation creates it (the automation then gets "already exists"); the owner can rename or drop it. **The owner can change or delete everything**, members' items included. Protections bind everyone as before, and **protecting a table also stops members adding rows to it** — the owner's lever for tables members should only read.

**Creator badges.** Every file, table and row shows who created it — the owner, "you", or a member by their **current** display name (their email's local part when they have none; "Former member" once their account is deleted). The owner's own Files and Datastore panels show the same badges on member-created items (the Datastore browser adds a "Created by" column only when some row is a member's). Names are self-chosen and may collide. Member items **stay** after the member leaves or loses access, still attributed to them.

**Uploads.** A member's upload lands in `saved/downloads/<their chat session>/` (never overwriting an existing file). At most **25 MB** per file and **200 MB** of saved files per member on an agent; only documents, spreadsheets, text, images and videos (`.csv .xlsx .xls .pdf .docx .doc .pptx .ppt .md .txt .png .jpg .jpeg .webp .gif .bmp .mp4 .mov .webm .mkv .avi .m4v`) — no zip, html or svg. Views and downloads are capped at 50 MB, the listing at the 2,000 newest files, and a member at 120 Files/Datastore requests a minute per agent. Flight Deck records every member upload and delete in its usage log.

**Datastore limits for members.** At most **10 tables** per member, and none once the store is within 10 tables of its table limit (the owner keeps 10 slots); at most **10,000 rows** added per member across the store, and no member rows past 90% of a table's row limit (the owner keeps the last 10%). A member's `sql` runs plain SELECTs on a separate read-only connection that **finish within 2 seconds** (no `WITH RECURSIVE`); a member's imports are at most **25 MB** (100 MB unzipped for xlsx) and must fit their remaining row budget. What a member stores is capped too: one value at most **4 MB**, a table or column name at most **128 characters**, a member's table or import at most **100 columns**, and everything a member added at most **50 MB** across the store (counted on the stored values, so an xlsx shared string counts in every cell it fills; a column default counts in every row that leaves it out, the rows already there too, and can't be added once other people have rows in the table; a refused insert or import leaves nothing behind). A member's table can't be named like a column in the store, an SQL keyword or function, or SQLite's own and internal (`sqlite_…`, `ds_…`) tables, and in `sql` a member's table is mapped only where it follows `FROM` or `JOIN`, outside quoted strings (use an alias, `FROM t AS x`, to qualify its columns). In a shared chat only the agent's own datastore is available (`project` is refused).

**How it is stored.** Each row records its creator in two hidden columns (`_created_by`, `_created_by_name`) that `describe`, `list_tables`, queries and exports don't show; **`SELECT *` hides them too unless the SQL names them** (a member's `sql` can read them by name: opaque Flight Deck ids and name snapshots, never emails). File creators are recorded when a file is written through the agent's tools, an upload, an export or a Drive download; files changed outside those (the owner's shell or scripts) count as the owner's unless they sit in a member's own folder.

**Safety.** Your agent **reads what members add** — files and rows — and the owner's agent has the owner's powers (shell, accounts, automations): treat member content as untrusted input, especially in automations. The agent marks other people's files and rows as reference data, not instructions, which reduces but can't rule out steering. Flight Deck never runs a member-written file as a page in its own origin: previews are sandboxed (for everyone — HTML previews in the Files viewer, the VFS viewer and Agent Folders no longer run with Flight Deck's origin), a member's view of a file keeps its type only for images, audio, video and PDF (everything else is shown as plain text), and **`/deck/view` refuses to present a member-created file as a deck**. Exports neutralise spreadsheet formulas: text cells starting with `=`, `+`, `-`, `@`, a tab or a carriage return get a leading `'` (every cell in a member's export; the cells of member-created rows in the owner's).

**Files from before this release.** The owner's **existing** `saved/` files become readable by members as soon as it is deployed — screenshots, browser captures, uploads in the owner's own chats, script outputs, saved scripts and tools (check them for passwords or keys) and what channels and automations saved there (WhatsApp or email attachments). On its first start with sharing on, Flight Deck sends the owner of every shared process agent **one bell notification** saying so; **tell owners before deploying**. Files members saved before this release (in their own chat folders) stay visible only to that member and the owner.

**Docker agents stay chat-only** for members: no Files or Datastore panels, and the agent refuses member workspace requests.

**Deploy.** (1) Restart Flight Deck with `FD_AGENT_SHARING=1` (`python -m captain_claw.flight_deck.server`) — no database migration (one system setting marks the owner bell as sent). (2) Restart every shared process agent: the first open of a datastore by the new agent adds the creator columns and an index to every table (a few seconds on very large stores; existing rows count as the owner's). Docker agents: rebuild or pull the image and recreate them for the owner-side badges and the datastore fixes. (3) `npm run build` in `flight-deck/`, commit the bundle, hard-refresh. The order is free: an older Flight Deck never calls the member routes, and a shared agent that wasn't restarted makes the panel say it **needs a restart** (Flight Deck relays nothing from an agent that doesn't acknowledge the member's signed request). Rollback = the previous Flight Deck and agents; the extra columns, indexes and `saved_attribution.db` are inert, but rows written by an old agent meanwhile count as the owner's.

### Shared agents — the owner's view (shared_agent_usage)

The **owner's** agent can tell its owner who it is shared with and look into how they use it, through one tool, `shared_agent_usage`. Off unless Flight Deck runs with `FD_AGENT_SHARING=1` on an auth-enabled deck, like shared agents themselves; there is no per-agent or per-channel switch.

**What it shows (six actions).** Members are named by their display name (with the `#3f9a`-style tag when two share a name, as in shared context) or the short key `list_members` shows — never an email or a Flight Deck id.

- **`list_members`** — who: name, since when, Google on or off, what they shared on the agent, and activity totals (conversations, messages, last active, tokens).
- **`member_activity`** — one member's conversations on the agent: lane, message counts, started / last active, tokens, and up to 8 topics they talked about. No message content.
- **`member_items`** — the datastore tables and rows and the `saved/` files that member created (the commons, see above); with `table`, their rows in it.
- **`member_shared_context`** — what that member shared with everyone: their published profile, folders and deep-memory slice (context packs).
- **`read_conversation`** — read one of a member's private conversations (their latest by default): the newest messages, paging back with `offset`.
- **`search_conversations`** — find text in current members' conversations (one member's, or all).

Conversations show only **what the member wrote and what the agent answered** — no tool calls or results, arguments, reasoning, model names or compaction summaries, and the agent's own internal follow-up prompts aren't shown as the member's words. **Answers drawn from the member's Google or private deep memory are hidden**: every agent reply of a member turn that used the member's Google or deep-memory tools shows as `[A reply drawn from their Google account or private deep memory — not shown.]` and is never searched. The tool never opens a member's Google account, deep memory or private profile; only their **published** profile pack is read. (A later reply that restates such an answer without a new lookup is shown.)

**Where it works.** On **owner turns everywhere the agent works for its owner**: the owner's Flight Deck chats, WhatsApp and glasses, Telegram (each approved Telegram user's instance), Slack and Discord, the OpenAI-style API pool, cron jobs, autonomous and plans runs, relays from the owner's other agents, slash-command and hotkey turns, sister sessions and `mcp_serve`. **Anyone the owner lets reach the agent there can ask it** — people connected through those channels, API clients holding the agent's key, the owner's other agents, and whoever receives an automation's output — and so can the deck's **admins** (an admin chatting with the owner's agent is an owner turn). **Never** in a member's turn (members never have the tool, and Flight Deck refuses a member's request for the roster), and never on public sessions or `public_run` agents, BotPort tasks, Iskra bodies or `CLAW_VFS_SCOPE` processes — the tool isn't even listed there.

**Current members, current membership.** The member list is Flight Deck's **live** roster (at most 200 members — beyond that they are invisible to the agent); if Flight Deck can't answer, the tool fails rather than guessing. Someone who left, was removed, or was deleted disappears within about 20 seconds, with all their conversations. For current members only what they did **since their current membership began** counts or shows — sessions, messages, tokens, topics and history snapshots; leaving and being re-added, or the agent changing owner and being shared again, starts fresh.

**What a reading turn can't do.** Reading members' data marks the owner's turn as private — `data` for the roster and activity, `content` for conversation text:

- **Nothing shared is learned from it.** Insights, reflections, intuitions, topics and playbooks aren't extracted from the turn and auto-capture skips it; every message added during it is flagged (`member_private`) and left out of compaction summaries, the archive and semantic indexing. While the owner's session still holds a flagged message, the agent's later replies are flagged too. Text the turn hands to another agent or session (orchestrator workers, sister-session briefings, consult / delegate replies that Flight Deck relays between agents) starts with a privacy header, so it is flagged there as well. In a flow, Flight Deck puts the header at the start of every **later agent step's** prompt once a step returned such a reply — or a `tool` step's output carried the header (`shared_agent_usage`, `history`) — and spawned and joined flows carry it too; it never goes into what the flow delivers to a person (WhatsApp, Telegram, chat push), and a step output that starts with it has it cut off there. When a flow was triggered by a message the agent received (a consult or delegate from another agent among them), the agent's relayed flow output is marked, so that agent gets the header too.
- **No writes to stores members read.** In such a turn the agent's `insights` add/update, `playbooks` add/update/rate, datastore writes and exports, `write`, `edit` and Typesense `index` are refused ("Not saved: this turn read members' private data …"). Only these: at the `data` level other tools (the shell and scripts among them) still run.
- **After reading conversations, nothing that sends or changes things** until the owner's next message: only read-only local tools run (`shared_agent_usage`, read, glob, grep, history, topics, and the read actions of todo, contacts, insights, playbooks, the datastore, `vfs` and Typesense); everything else — web search, messages, email, shell, MCP and plugin tools included — is paused ("Paused: this turn read members' private conversations …"). The owner can then ask it to go ahead. The pause belongs to the turn: tools the owner's automations run between turns (cron scripts, flow tool steps, the action rail) are not paused.

The owner's next message runs as a normal turn: **whatever the owner then asks it to save to files, the datastore, insights or playbooks is visible to every member.**

**The owner's `history` and `topics get`.** History snapshots and topic excerpts from **current** members (since their membership began) are kept and mark the turn as `content`; ex-members' are dropped. A topic excerpt shows only what reading that member's conversation would show — never a reply drawn from their Google or deep memory, nor the agent's own follow-up prompts. The owner's agent can't rate a member's conversation or turn it into a playbook.

**In the agent's prompt.** Flight Deck writes `shared_members.md` next to `shared_context.md` in the agent's config home (`data/home-config` on Docker): a short block naming the current members (at most 2,000 characters; the first 30 names, then "N more") and telling the agent to use the tool only for its owner and never to pass one member's conversation to another or into shared notes. It is rewritten on every share, revoke, leave, rename, owner change and reconcile round (within about 70 seconds for edits Flight Deck only sees by polling), removed when the agent has no member, and used only on owner instances; the tool is listed only while the file exists.

**Docker agents.** The same tool and file; `member_items` is empty (no commons), Google always shows off and only profile packs exist.

**Told to everyone.** Members see a new paragraph in the shared-agent notice (acknowledged again — notice version 4) saying that the owner's agent can look into their use and quote their conversations when the owner, the deck's admins or anyone the owner lets talk to the agent asks, and that it leaves out its replies from the turns in which it used their Google or private deep memory (a later reply that repeats what it found there can still be shown). On its first start with sharing on, Flight Deck sends **every current member one bell notification** saying the same. The owner's **Share** dialog explains the tool, the hidden replies, the reading-turn limits (only what is enforced) and that members are told; on a Docker agent it leaves out the commons and members' Google and deep memory, which Docker members don't have.

**Audit.** The agent logs one line per private read (the member's key, a conversation id prefix, counts — never content); Flight Deck logs one line per roster read (the agent and how many members — never names, emails or ids).

**Not covered.** The owner can still retype or deliberately store what they read. Prompt injection planted in a member's conversation can't make the reading turn save or send anything, but the text stays in the owner's context, and a later owner message ("continue") runs with full powers. The passive semantic-memory note (`memory.cross_session_retrieval`) still indexes member sessions (owner-private). Members past the first 200 are invisible. Flight Deck can't list who reaches the agent through its channels.

**Deploy.** (1) Restart Flight Deck with `FD_AGENT_SHARING=1` (`python -m captain_claw.flight_deck.server`): the startup reconcile writes `shared_members.md` for every agent with members and the one-time member bell goes out (one system setting marks it sent — no database migration). (2) Restart every shared process agent **and every agent that consults or delegates to one**. (3) Docker agents: rebuild or pull the image and recreate the containers. (4) `npm run build` in `flight-deck/`, commit the bundle, hard-refresh. Any order is safe for the tool itself — an older agent ignores the file, a new agent on an older Flight Deck gets no file and no roster — but the privacy header on relayed peer replies needs Flight Deck **and** the agents new, so deploy them together. Rollback = the previous Flight Deck and agents; delete leftover `shared_members.md` files by hand. The new session-message keys (`member_private`, `turn_input`) stay in session JSON and are ignored by older code.

### Basna (NEW in 0.5.7)

Basna (sidebar: **Basna**) is a **network-source ensemble** — a one-shot, selective alternative to Council. You describe a task; Basna routes it to the smallest set of specialist archetypes, runs them in parallel, and merges their answers weighted by learned reliability.

**How it works:**

1. Open **Basna**; describe the task (optionally **attach files** or paste images), pick the **Router tier** (default Reasoning) and **Max agents**.
2. **Route** — a router classifies the task (domain / difficulty / converge|diverge) and selects the minimal archetype subset, scaled to difficulty. Each pick shows its tier, resolved model, rationale, and prior reliability weight.
3. Optionally **edit any agent** before running — tier (a picker that resets the model fields), provider, model, API key, base URL, in/out context, cognitive mode, fleet instructions (system prompt), extra task instructions, role. Per-agent overrides take precedence over the Library tier.
4. **Run ensemble** — the chosen archetypes spawn fresh, run **blind and in parallel** (attached files are copied into each agent's workspace), then are torn down. A **live progress log** streams every stage (route → spawn → dispatch → merge → learn) with each agent's tool calls, **narration**, and LLM usage; the log and per-agent activity are exportable and persisted.
5. Outputs are **merged** weighted by each archetype's learned reliability — a reasoning-tier synthesizer runs only on genuine disagreement. The compiled **truth** and each agent's answer render as markdown (fullscreen + export-.md). Files the agents generate are captured into a **Generated files** panel for download.
6. Each contribution is scored against the truth and folded into per-archetype, per-domain **reliability**, so future routing improves; a 👍/👎 on any agent overrides the score.

**Basna vs Council:** Council is a multi-round deliberation *among* agents that see each other; Basna is one-shot — agents run blind and independent and the result is a weighted merge, not a conversation.

**Session titles (NEW in 0.6.0)** — every run has a **title**: type one, or it's auto-generated from the task (the router returns a concise title; a heuristic covers the fallback path). Editable inline; shown in the run list with the task as a subline.

**Agent-initiated runs (NEW in 0.6.0)** — agents can start and read Basna themselves via the always-available **`basna`** tool (see [Tools Reference → basna](#basna)). `basna start` kicks off an **autonomous** run from any channel (web / WhatsApp / glasses / API): Flight Deck auto-titles, routes, and executes the ensemble **server-side**, then **reports the result back to the agent** to relay to the user. Fire-and-forget, capped at 2 concurrent per owner. When the user explicitly says "run/execute a Basna…", the agent relays the task to the tool deterministically rather than doing the research itself. Such runs are **badged with their origin channel** in the unified run list (with an "agent" filter), and the list **polls live** while runs are in flight. Agent-facing endpoints live under `/fd/basna/agent/*` (read + `start`).

Endpoints under `/fd/basna`: `route`, `execute`, session CRUD, `sessions/{id}/runs`, `sessions/{id}/progress`, `sessions/{id}/files` (upload/download/delete), `runs/{id}/feedback`, and the agent-identity `agent/*` set. Tables: `basna_sessions` (now with `title`), `basna_runs`, `archetype_reliability` (migrate in place).

**Deepen (NEW in 0.6.1)** — a follow-up run that resolves a finished Basna's blind spots. Click **Investigate blind spots** on a completed run; FD spawns a fresh ensemble seeded with the prior synthesis as context (inline up to 16k chars; larger goes as a workspace file the agents read). The new run appears in the run list with a **deepened from / deepened into** lineage link, and shows **live per-agent panels** server-side. **Hard-stop** primitive + run-rate breaker (6 runs / 300s per owner) prevent runaways; the arbiter can also propose `stop_run` for stuck Basna or Council deliberations. A `/basna <task>` slash command works on **web, WhatsApp, and Telegram** (help text on no-arg); Croatian verb stems (`pokrenuti`, `izvršiti`) also trigger the deterministic relay. A Basna **worker** can never start another Basna (recursion forbidden — the `basna` tool is stripped from spawned workers and double-checked via env marker).

### Vatra — the collaborative ensemble (NEW in 0.6.3)

Vatra is the **collaborative sibling of Basna**, surfaced as a **compose mode on the Basna page** — a toggle between **Basna** (*independent ensemble — agents answer blind, merged by reliability*) and **Vatra** (*collaborative team — a Lead splits the work, a reporter assembles it*). The choice is remembered per browser; Vatra sessions are badged **vatra** in the run list.

Where Basna spawns agents that each answer the *whole* task blind and then **merges** their uncorrelated outputs, Vatra runs a **team that divides the work** and collaborates as it goes:

**How it works:**

1. Open **Basna**, switch the compose mode to **Vatra**, and describe the task (optionally **pre-pick the archetypes** the Lead must use as owners, attach files, pick tiers / max agents).
2. **Plan team** — a **Lead** decomposes the task into the smallest set of **complementary, owner-assigned subtasks**, each with a title, an owner archetype, a brief, and `depends_on` links, plus a **shared context** contract every piece must honor. The decomposition persists as a prepared session and renders in the **team plan** panel for review before anything spawns. (A fixed team is guaranteed a piece each — if the Lead misses one, a task-derived subtask is added for it.)
3. **Run** — each subtask owner spawns fresh and works **in parallel**, but collaborates through a **shared blackboard**. An optional **intro (prep) round** lets each specialist post groundwork (facts, sources, an outline) before writing its full piece, so the team starts from shared footing.
4. **Delegation by asking** — an owner that needs something from a teammate posts an **ask** to the board; a background **coordinator** spawns a short-lived helper to answer it. Termination is guaranteed: a per-run **ask ceiling (12)**, an **ask depth cap (2)** so an answer that itself asks can't cascade forever, and at most **3 concurrent helpers**. The live **blackboard panel** shows asks and the shared board in real time, with a **per-agent Skip** button to drop a stuck specialist's turn.
4b. **Blocking on a teammate's artifact (`wait`, NEW in 0.6.5)** — an owner that genuinely depends on a specific artifact a teammate is producing can call the **`wait`** action instead of improvising a guess. It takes a `path` (`vfs:<project>/<file>`) or a `query` (board keywords); the `/agent/wait` endpoint **long-polls** the VFS file or blackboard every second, capped at **90s** (below the dispatch timeout, so it can't hang a run; the tool timeout is 120s). On timeout it returns a **board digest** plus a "proceed deliberately" nudge rather than an error, so the agent does the right *something else*. Sibling owners keep working while one awaits. Waits surface in the live timeline — **start** (⏳ ≤Ns), **resolve** (✓ got file / board match), **timeout** (⌛ not ready). It's a tool, not forced ordering: there's no `depends_on` enforcement, and worker prompts steer agents to `wait` only when truly blocked (don't busy-poll). With **`push_deps`** (opt-in via `quality`, off by default), grouped mode goes further: once a producer in an earlier phase has finished, its actual delivered output is **pushed into the consumer's prompt** so the DAG edge carries data, with the `wait`/board pull path kept as the fallback for producers not yet done.
5. **Review round** — the Lead gathers an exec summary of the whole team's work and sends it back to each owner so they can revise their piece against what everyone else produced.
6. **The reporter assembles** — a dedicated **reporter** archetype (default a strong general writer; configurable) reads every piece and the full blackboard and writes **one coherent deliverable**. There is no weighted merge and no single "winner". The result renders as the **Final report** (shown as *assembled X%*, not *confidence*); generated files are captured for download. The deliverable is **persisted the moment the reporter finishes**, so a late failure can't lose it.
7. **Learning** — Vatra scores not just the subtask owners but also the **helpers**, the **Lead's decomposition**, and the **reporter's assembly** (the last two as pseudo-archetypes `vatra-lead` / `vatra-reporter`), folding outcomes into per-archetype, per-domain reliability so future decompositions and reporting improve.

**Guarantees** mirror Basna: a Vatra **worker can never start another run** (the run-starting `basna`/`vatra` tools are stripped from spawned agents and a `CLAW_VATRA_WORKER` env marker is double-checked), and spawned owners/helpers are torn down at the end of the run.

**Basna vs Vatra vs Council:** **Basna** for a single best answer from independent voices merged by reliability; **Vatra** for a *composed* deliverable whose parts are interdependent (a report, a plan, a spec) built by a collaborating team; **Council** for a multi-round *deliberation* you watch and steer (and which can now [auto-assemble its own panel](#auto-assemble--a-panel-modeled-to-the-topic-new-in-063)).

Endpoints under `/fd/vatra`: `route` (Lead decompose / prepare), `execute` (spawn + run), `progress`, the blackboard ask/inbox set, and `agent/wait` (the rendezvous long-poll, NEW in 0.6.5); it reuses Basna's session CRUD, archetype catalog, and `archetype_reliability` table.

### Code — the agentic coding studio (NEW in 0.7.0)

A dedicated **Code** page (sidebar: **Code**) where fleets of specialist agents build, review, and fix software in real git repos — with you at the approval gate.

**The mental model.** A **project** is a namespace. Inside it live **folders** — each one a real directory that is simultaneously a git repository and the spawned agents' workspace/cwd — and **sessions**, conversations that each drive work in one chosen folder. A folder can be a fresh VFS directory *or* an existing local repo **linked in** (read-write or read-only). One **New session** modal walks all three steps: pick/create the project, pick/create/link the folder (with a server-side **Browse…** picker for local paths), name the session.

**What happens when you send a message:**

1. **Router** (your `fast` tier) classifies the request — *small* or *big* — and picks the executor. Git requests in plain language ("commit this", "create a branch and push") route to a dedicated **git-operator**; "map this repo" routes to the cartographer.
2. **Small path** — a single specialist (`quick-dirty`, `code-implementer`, or `debugger`) does the edit in the repo, the result is committed (`[edit] …`), and the Code Map refreshes incrementally. If the agent discovers the job is actually substantial, it **escalates**: it stops, says why, partial work is committed, and the request is promoted to the big path — so a mis-sized task never dead-ends.
3. **Big path** — a **planner** (`light-planner` or `architect`, `reason` tier) surveys the code and writes a concrete plan to `.plans/<timestamp>-<slug>.md` (every turn's plan is preserved — nothing overwrites `plan.md` anymore). The plan lands in an **editable approval gate**: edit it, **Approve & Build**, or **Discard**.
4. On approval: the **builder** implements the plan (real shell — installs deps, runs the program/tests), then **three independent reviewers run in parallel**: a code reviewer (correctness/edge cases/regressions), a security reviewer (CVSS-ranked findings), and a QA engineer that **actually executes the test suite** and writes missing tests. Their reports persist under `.reports/`.
5. **Triage** (`reason` tier) reads all three and decides ship-or-fix — only blocking/major findings and failing tests trigger a fix round; style nits ship. A capped **fix loop** (3 rounds) sends precise, ordered instructions to a `debugger`/`code-implementer`; each fix commit is re-reviewed as a **delta** (the fix diff + prior findings — never a full re-read; the security reviewer sits out rounds whose diff touches no security surface). At the cap, open findings persist to `.reports/backlog.md` and typing **"continue fixing"** resumes exactly there.
6. A **cartographer** refreshes the Code Map, and the run posts its total: `N agent runs · X in → Y out tokens`.

**Every phase is a git commit** (`[plan]`, `[build]`, `[review rN]`, `[fix rN]`) in the folder's own repo — the right-hand History rail is a commit timeline; any commit opens a colorized diff; any commit is one confirm away from **rollback**. Linked repos get your real git identity on commits, and runtime artifacts (`.code/`, `.codemap/`, `saved/`) stay out of your git via local excludes.

**The Code Map** is the token-efficiency backbone: a per-repo SQLite+FTS5 index of every function/method/class (signature + `file:line` — Python via `ast`, JS/TS via a dedicated extractor, more via universal-ctags when installed) plus an LLM-authored semantic layer (architecture overview, data-model map, UI map, per-file purpose). Freshness is **git-blob-hash gated** — only changed files re-index after each commit. Every code agent gets a **`codemap` tool** (`overview`/`search`/`symbol`/`file`/`models`/`ui`) returning pointers (never source dumps, stale entries flagged), and its prompt leads with the overview + "query the map, read ranges, don't re-read whole files." The **Map tab** renders it all for humans, with symbol search and a **Rebuild map** button (live progress).

**Guardrails** (learned from real runs, enforced in the tools): a full-file `write` that would shrink an existing file by more than half is **refused** (use `edit`, or re-read + `overwrite=true` for an intentional rewrite — this once saved an 1,800-line game from a 14-line clobber); writes aimed at untracked scratch (`saved/…`) are auto-relocated into the repo; identical full re-reads of unchanged files short-circuit to a one-line pointer; code agents skip the generic list/scale contract machinery entirely (the pipeline's reviewers judge completion, not reply-text scanning).

**Operations:** a red **Stop** button halts a running pipeline at the next phase boundary (partial work stays committed); an **Eraser** button sweeps leftover ephemeral code agents after crashes/stops; **Export** assembles the entire process — every prompt, tool call, narration line, review report, and token count — into one Markdown transcript. A complete real run (an RTS game with 185 tests) ships in `examples/code/`.

**Models:** every role resolves from your Library tiers — router/triage on `fast`/`reason`, planner on `reason`, implementer/debugger on `coding`, reviewers per their archetypes. Mix freely: DeepSeek as the implementer with Claude reviewing, or 100% local via Ollama. Per-turn cost lines in chat keep the spend honest (they match your provider's dashboard).

**Why Code instead of Cursor / Lovable / Claude Code / Codex / OpenClaw / Hermes-style daemons?** The structural difference: those are single-agent assistants or hosted builders; Code is a **self-hosted multi-agent pipeline with independent review and a human gate**. IDE assistants (Cursor, Windsurf) have you steering keystroke-by-keystroke while one agent writes and re-checks its own work — in Code, the reviewers **never wrote the code they judge**, QA *runs* the tests, and you review plans and outcomes instead of diffs-in-flight. Hosted builders (Lovable, v0, Bolt) keep your code on their cloud and their stack — Code commits into **your** repos on **your** machine. Single-agent CLIs (Claude Code, Codex) are one context playing planner/implementer/reviewer/QA at once, on one vendor's models — Code separates the roles, gates the build on your approval, runs on any provider mix, and leaves artifacts a CLI session doesn't: per-phase commits, plan history, on-disk review reports, a queryable code map, a full transcript. Assistant gateways (OpenClaw) route one assistant across channels but have no engineering pipeline; autonomous coder daemons (the Hermes mold) optimize for unattended pushes where Code deliberately keeps two human moments — the plan gate and the review verdicts — and makes everything between them inspectable and stoppable. And uniquely, Code lives **inside a fleet platform**: the folder your Vatra research team just filled is one click from becoming a Code project's context.

Endpoints under `/fd/code`: `projects` (tree/create), `projects/{p}/folders` + `link`, `sessions` CRUD, session-scoped `message`, `plan/approve`, `plan/cancel`, `stop`, `chat`, `progress`, `log`, `diff`, `show`, `rollback`, `map` (+ `map/search`, `map/build`), `export`, and a global `cleanup`. VFS gains `/fd/vfs/links` (registry admin), `browse-fs`, and `download-zip`.

### Autonomous Work (NEW in 0.6.1)

The closed-loop autonomy page (sidebar: **Autonomous Work**) where the assistant **notices → decides → acts → judges → learns** on the user's behalf. Per-user, configurable, and ships under a `propose` ceiling — every action waits for human approval until a specific tool action earns auto-fire via the trust ladder.

**Control tab**

- **Master switch + autonomy level.** `off` / `propose` (default ceiling) / `act_low_risk` / `act`. The level is clamped to `max_autonomy_level` server-side, so you can't grant the loop more authority than the deployment allows.
- **Arbiter dials.** Heartbeat trigger, daily action cap, concurrent action cap (in-flight work only), quiet hours (UTC), candidate lookback. `arbiter_min_score` controls what the ranker needs to land in the ledger. Proposals awaiting approval have their own cap (`max_pending_proposals`, default 5 — a full queue logs "loop paused") and expire after `proposal_ttl_hours` (default 48; an expired title isn't re-proposed within the lookback).
- **Judge & learn.** `judge_mode` (auto LLM | human | both), per-kind reliability seed, suppression threshold (proposals stop landing once a kind's weight falls below `suppress_below_weight`).
- **Trust ladder.** A reversible, low-risk `tool_action` auto-promotes to auto-fire once its learned reliability ≥ `trust_threshold` (default 0.85) over ≥ `trust_min_runs` (default 3). Demotes automatically when the weight falls back. Manual grant overrides this and skips the earning. **Mail never earns trust (since 0.8.0):** every mail-writing action, including MCP and custom ones, is human-only — proposed, never auto-fired by grant or trust — and old mail trust rows are swept at every Flight Deck start.
- **Event sources.** Per-user toggles for the FD-side Google **Calendar** and **Gmail** pollers; an automated-sender filter drops no-reply / notification mail before it ever becomes a candidate. Event reconsideration: an event stays `new` until a pass actually produces an action (so a single whiffed beat doesn't drop it forever), bounded by `event_max_surface_attempts`.
- **Soft-reminder follow-ups.** Beyond *act-now* and *drop*, the arbiter has a third outcome — **`track`** — for soft asks / "waiting on you" items. They land in a per-user follow-up list with a due date; when due, the loop nudges with **escalation** (3d → 2d → 1d) and retires to `stale` after enough nudges (`followup_max_nudges`).
- **Nudge delivery.** A delivered nudge counts as success (don't LLM-judge nudges — that would suppress good ones). Nudges also fan out to the owner's WhatsApp via `nudge_to_whatsapp`: the numbers in their per-user `notify_waid` (must be on `WHATSAPP_ALLOWED_WAIDS`, at most 3, not already another user's; set it in **Connections → WhatsApp nudges**, which also sends a test — API `GET/PUT /fd/autonomy/whatsapp`, `POST /fd/autonomy/whatsapp/test`. `PUT /fd/autonomy/config` keeps the stored number and ignores a new one; with auth on, a deck-wide `notify_waid` in config.yaml applies to nobody). Unset: a deck without auth pushes to every allowlisted number; with auth on, nothing is pushed (logged as `nudge → whatsapp skipped`). A 30s dedup guard at the outbound chokepoint prevents the same reply being sent twice across native + push.
- **Grounded verification.** After a catalog action runs, the side effect is read back (note → `read(path)`, calendar event → `get_event(id)`). Absent → fail; couldn't read → kept but flagged `[unverified]`. Trust never builds on phantom successes.

**Tools & Sources tab** — promote the agent's own tools into the autonomous catalog and event spine.

- **Custom actions (hands).** Pick an agent tool from the live list of registered tools and turn it into a catalog action — set `risk`, `reversibility`, optional `reverse_tool` for one-tap undo, `grant`. Defaults to `human_only=true` (propose-only). Can only auto-fire after the trust ladder earns it. `shell`, `browser`, `playwright`, `selenium`, social/post tools, payment/stripe/checkout, transfer/wire, and `basna` are HARD-EXCLUDED — they cannot be added.
- **Custom sources (senses).** Pick a read/list/search tool; the generic poller calls it on a cadence, parses rows from JSON output (auto-detected or via `items_path`), dedups by `id_field`, and stamps a **fetch contract** (`fetch_tool` + `id_field`) on every event. Per-tool **interval cap of 60s** and **20 sources max** per user prevent hammering an API.
- **Table-driven grounding.** When the arbiter dispatches an action derived from an event, the agent is handed: *"this is a REAL item, open it BY ID with `<fetch_tool>`, never search, never claim it doesn't exist."* Built-in sources (Gmail thread, Calendar event) and custom sources share the same grounding line.
- **Library of one-click presets.** Granola meetings, Gmail search, Drive recent files, Web watch (sources); Save to project memory, Save to datastore, Create Drive doc, Label Gmail thread, Suggest meeting time, Fetch a web page, Web research brief, Summarise files (actions). Names are best-guess for a typical Granola/Google stack — confirm each against the live agent-tool list before saving.

**Activity tab**

- **Action ledger** (per-user, audit trail) — every arbiter decision with its source (reflection / intuition / intention / event), kind, title, rationale, risk, domain, score, status, outcome, and outcome note.
- **Waiting on you** — open follow-ups with due time, age, nudge count; Done / Dismiss buttons. Dismiss feeds reliability (negative signal); Done feeds it (positive).
- **Plans (#4).** Give it a goal; it decomposes into catalog-validated steps. Manual or auto-advance; a failed step re-decomposes the remainder rather than failing the whole plan. Abandon a plan and reversible done steps roll back newest-first.
- **Live log.** Every arbiter pass, skip reason, dispatch, judge verdict, and error — so nothing is swallowed.
- **Learned reliability.** Per-kind / per-tool-action reliability weights (Bayesian; fails count double; seeded at `reliability_seed`).
- **Recent events.** New / surfaced / acted / ignored — clickable, so you can see what the loop has seen.

**Action catalog (built-in, #1).** Reversible auto-eligible: `note.write`, `calendar.hold`, `reminder.schedule`. Human-only (propose for approval, never auto-fire): `mail.draft` (since 0.8.0), `mail.send`, `calendar.invite`, `calendar.delete`. `reply.on_origin` (a Taskboard next step's chat reply, delivered by Flight Deck to the chat the task came from) is human-only too and never proposable — the arbiter, plans and the manual action runner never reach it; `message.send` and `drive.delete` are gone. For email the arbiter prefers a nudge ("X is waiting for a reply — subject") over a draft, and ignores Gmail events older than 48 h.

**Automated turns write mail only when asked (since 0.8.0).** A scheduler job, flow, plan, peer request, Basna/Vatra/Code worker or MCP task may draft or send email only when its own human-written text asks for that email (English and Croatian are recognised); otherwise the mail tool answers `[not-authorized: mail-write]`. A self-scoped job ("email me the summary") may write only to the owner's own or an explicitly named address. Organizing a mailbox is a separate permission the job text must ask for too. The agent logs `legacy cron job may write email` at start for old cron jobs whose text asks for email — review those. Interactive chats keep prompt rules only. Plus any custom actions you promote.

**Webhook push (#2).** A token-gated `POST /fd/events/webhook` accepts an external system pushing an event directly into the spine (sub-minute latency vs the 5-min poll cadence). Disabled unless `FD_EVENTS_WEBHOOK_TOKEN` is set; caller passes it via `X-Webhook-Token` and names the target `user_id` in the body.

Stores: `autonomy.db` (config, ledger, reliability, follow-ups, plans, log), `events.db` (external events + per-source poll state). Endpoints under `/fd/autonomy/*`: `config`, `actions`, `actions/{id}/{approve|reject|undo}`, `reliability`, `nudge`, `log`, `catalog`, `run-action`, `plans`, `plans/{id}/{advance|abandon}`, `follow-ups`, `follow-ups/{id}/{done|dismiss}`, `agent-tools`. The loop **ships off** — opt in per user.

### Topics — conversation topic memory (NEW in 0.6.1)

A durable, cross-session **topic memory** built from comms traffic. Every ~15 messages a background classifier groups recent user + assistant turns and the turn's narration into topics; the agent can recall a whole thread at once via the always-on **`topics`** tool. There's a full Flight Deck panel on every agent card (button next to **Cron**, on both Process and Docker cards).

**Reach the panel:** click **Topics** on an agent card → modal opens. Header row: gear (Generate / Reset), Refresh, Fullscreen, ×. Title shows the live status when a backfill is running ("Classifying… N done · M topic updates · K left").

**Left pane — the list:**

- Topic count, sort selector (**Recent** by last message / **A–Z**), starred-on-top.
- Search box: substring (LIKE) match across **label / summary / keywords** — `Bise` finds "Biserka…".
- **Groups** (user-defined, many-to-many): All / Work / Private / custom; create with **+ group**; click chips to filter. A topic can be in several groups; a group holds several topics. Filters compose with text and tags (AND).
- **Tag filter chips:** click any keyword chip on the open topic and it becomes a removable filter at the top of the list; tag, text, and group filters all AND together.
- Each row: **checkbox** (multi-select), **star** (pin to top), label, recency, summary, msg count.
- **Drag a message** from the detail pane onto a topic row to move it (drop target highlights with a sky ring; the dragged message dims). Counts on source and target update; target's `last_seen` is bumped.
- **Combine** — when ≥2 topics are checked, a bar offers to merge them into the first one (messages re-pointed, dedup by msg id, keywords + summaries merged; sources are deleted).
- **Reset** — wipes all topics + backfill progress. If topics are checked, **Reset (keep N)** preserves them and wipes only the rest.

**Right pane — the topic:**

- Title, summary (markdown rendered, tables work), keyword chips (clickable to add to filters), group-assignment chips (click to toggle membership).
- Header buttons: **Refresh** (re-pull full message text from the live session by msg id, fixing excerpts that were captured under an older 600-char cap), **.md** export, **Reclassify** (drop the topic and free its messages so the next pass redistributes them — auto-runs Generate over all history right after).
- Message list (markdown). Each message is **draggable** — drop on another topic to move.
- **Per-topic chat** pinned at the bottom of the pane: streaming reply, live narration, tool-usage rows, paste + file attachments, busy-gated send (Stop cancels). Renders only the turns you initiate here (ignores the agent's session replay so topics never mix). Each completed turn is persisted to the topic by its real session msg id.

**Generate / Reset behind the gear.** Pick a window (Last 24h / 7 days / 30 days / All history) and Generate to backfill untagged messages. One LLM call per batch; the UI loops short requests until drained. Reset is selective (see above).

**Periodic classifier.** Runs in-process on the agent (mirrors the dreaming / insight-extraction passes) after web-chat, CLI and remote-platform turns, once `conversation_topics.interval_messages` (default 6) conversation messages are pending, with `cooldown_seconds` cooldown. A topic is made of what people typed and the final reply of each turn they opened — never fleet notices, cron prompts, correctives, tool steps or replies to automated turns (narration only with `include_narration`, off by default). Progress lives in the store: everything after the newest message a pass attempted is pending, so restarts and compaction neither re-read the session nor stall the pass; a pass handles up to three batches oldest first and the rest waits for the next one. The classifier sees the existing topics most relevant to the batch plus the ten most recent, with whole summaries. It uses ~8k output tokens and small batches (15 messages/call) so reasoning models finish the JSON.

**Automation lane and new sessions.** Automated turns from Flight Deck (scheduler, autonomy, plans, peer, BotPort and MCP tasks) run on the agent's `AUTO` lane — its own session, shown as the "Auto" tab in Flight Deck — instead of growing the main chat; results meant for you are mirrored into the main chat and its channels. Set `session.automation_lane: ""` to keep them in the main chat. Starting a message with "nova tema:" / "new topic:" (configurable `session.rotation_cues`) or pressing "+ New session" in Flight Deck starts a fresh session. Compaction now folds older messages into a topic-grouped digest (`context.compaction_summary: digest`; `model` for the model-written summary).

**Recall card and pins.** On a turn a person opened, the agent can recall one earlier topic from the turn's own words — only when the word and meaning rankings agree or one leads clearly, and never a topic still in the live conversation — as a short dated card inside the per-turn context block. `conversation_topics.recall`: `shadow` (default; the decision is logged and kept in the context trace, nothing is sent), `on`, or `off`. The `topics` tool adds `recall` (best topic for a question, in full) and `pin` / `unpin` (a pinned topic rides the block for `pin_turns` turns, default 5, whatever the recall mode). Shared-agent members recall and pin only topics they spoke in.

**Search and hidden topics.** `topics search` ranks by word match (FTS5 bm25: label, keywords, summary; long words match by stem) and, with a local embedder, by meaning; typed panel search also matches word prefixes and substrings. Topics built mostly from machine text (old fleet notices, cron prompts) are hidden from recall once per store; the panel still lists them, starring one brings it back, and `POST /api/topics/{id}/hide` with `{"hidden": false}` does too.

Stores: `conversation_topics.db` (topics + topic_messages + FTS + groups + many-to-many + backfill_seen). Endpoints under `/api/topics/*` (list, search, refresh, combine, reset, append, groups, star, message-move, `{id}/{refresh|unclassify|groups|star}`, `messages/{id}/move`), proxied through `/fd/agent-topic*`. Topics memory ships **on** — disable via `conversation_topics.enabled` in config.

### Terminal — a live terminal on your own machine (NEW in 0.6.2)

The **`terminal`** tool drives a real pseudo-terminal (PTY) on a machine **you** choose — your Mac, a laptop, a remote box — and is reachable from **both Flight Deck web chat and WhatsApp**. It's the interactive counterpart to `shell`: `shell` runs one command with no tty on the agent's host; `terminal` holds long-lived sessions on *your* machine and can run REPLs, prompt-driven CLIs (`claude`, `ssh`, `psql`), and full-screen TUIs by typing keystrokes into a live tty and reading back what appears. The tool's description steers the agent to prefer it whenever you refer to your own machine ("on my Mac", "my laptop", "at home").

**Tool actions.**

- **`run`** — one-shot: execute a command on your machine and return its output (open → execute → capture → close in a single call). Runs under a login shell, so `cd`, `&&`, pipes, your `PATH`, and exit codes all work. As easy as `shell`, but on your machine.
- **`open`** — start a persistent session (returns a `session_id`); optional `command` (default: your interactive login shell) and `cwd`. A command **string** runs as `$SHELL -lc "<cmd>"` (shell syntax + interactive programs work); pass an explicit argv **list** for literal, no-shell exec.
- **`send`** — type into a session: free `data` text, a named `key`/chord (`enter`, `tab`, `esc`, `up`/`down`/`left`/`right`, `ctrl-c`, `ctrl-d`, `ctrl-l`, …), and an `enter` flag — then returns what the program printed back. Reuse the `session_id` across turns.
- **`read`** — pull new output without typing. **`close`** — end a session. **`resize`** — set cols/rows.
- **`list`** — show active sessions **and report connection status**: `✓ Connected — worker 'mac'`, `✗ Not connected — worker 'mac' has no terminal daemon running`, or `✗ Terminal not reachable`. Lets the agent tell "your machine is offline" apart from "I don't have this tool". ANSI escape codes are stripped by default (text channels); pass `raw=true` to keep them.

**Waiting for input.** The daemon detects when a session is **alive but parked** — output has settled and the tail looks like a prompt (a numbered menu, `(y/n)`, a `read -p` line, `claude`'s "Do you want to…?" confirm). `open` / `send` / `read` then append a **⏳ marker** telling the agent to answer with `send` rather than assume the program finished. A one-shot `run` that hits a prompt **bails** with guidance to use `open` / `send` instead of hanging. (Detection is heuristic — quiescence + prompt-shaped tail — so a long silent computation may also be flagged; the agent can just `read` again.)

**Proactive watcher.** The above only fires when the agent happens to `read`. For the case where the agent kicked off a program and then went idle, a **background watcher** runs while any interactive session is open. It polls each session through a non-advancing `peek` (so it never steals the agent's output); when a session has been **quiet and prompt-shaped** for a few seconds, it makes **one pure (tool-less) LLM call** — given the terminal tail *and* the recent task context — that plays the agent's role and returns a decision:

- **answer** — the choice is obvious, safe and reversible given the task (confirming a file creation, the clearly-intended menu option). The watcher types it into the terminal and tells the user what it did.
- **ask_user** — a real decision, or anything destructive/irreversible. The watcher relays the question to the user's channel (`deliver_to_origin`) and waits for the human; the user's reply flows back through the normal agent turn.
- **wait** — not actually waiting (still working / idle shell prompt). Keep polling.

Guard rails: a keyword screen (`rm`, overwrite, force-push, deploy, sudo, payments, …) forces `ask_user` even if the model said `answer`; a **consecutive-auto-answer cap** (default 3) stops a runaway answer→prompt→answer loop and hands control back to the user; the watcher never touches the agent's session state (only pure-LLM + daemon writes + delivery), so it can't corrupt a concurrent user turn. Tunable via env: `CLAW_TERMINAL_WATCH` (on/off), `CLAW_TERMINAL_WATCH_AUTO` (allow auto-answer vs always-ask), `CLAW_TERMINAL_WATCH_QUIET` / `_INTERVAL` / `_COOLDOWN` / `_MAX_AUTO`. The watcher runs in the **agent** process and needs the agent reachable to a delivery channel (WhatsApp/Telegram/glasses) to notify in real time.

**The PTY daemon (your machine's side).** Run `python -m captain_claw.terminal.daemon` on the machine you want to drive. It owns the sessions, is decoupled from the agent (so sessions **survive agent restarts**), opens new sessions in the folder it was launched from, and **mirrors live output to its own console** — a dim `── session … · cmd · cwd ──` header, the real-time output, and an `── session … ended (exit N) ──` footer — so you can watch the agent work (`CLAW_PTY_MIRROR=0` to silence).

**Two connection modes.** The tool ↔ daemon link is plain HTTP, so the daemon can live anywhere the agent can reach.

- **Reachable network** (LAN / VPN / Tailscale / public box): start the daemon with `CLAW_PTY_HOST` / `CLAW_PTY_PORT` (default `127.0.0.1:23190`) and point the agent's `CLAW_PTY_URL` at it. The daemon **refuses a non-loopback bind without `CLAW_PTY_TOKEN`** unless `CLAW_PTY_INSECURE=1`.
- **Behind NAT (dial-out)**: set `CLAW_PTY_RELAY=wss://<flight-deck>/fd/pty/connect` and `CLAW_PTY_WORKER=<name>`. The daemon opens a persistent **outbound** WebSocket to Flight Deck and registers; the **relay** (`/fd/pty/connect` for daemons, `/fd/pty/{worker}/{op}` for the agent) tunnels each tool call down the worker's socket. No inbound ports; sessions survive a dropped socket and reconnect with backoff. Point the agent's `CLAW_PTY_URL` at `http://<flight-deck>/fd/pty/<worker>`. The dial-out client uses the `websockets` library so it tolerates the duplicate `Server` handshake header reverse proxies (Cloudflare/nginx) add.

**Security.** A PTY is arbitrary code execution on the target machine. One shared `CLAW_PTY_TOKEN` gates all three legs — the daemon's register frame, the relay's HTTP, and the tool's `X-Claw-Token` header. Set it on Flight Deck (its environment, inherited by spawned agents) and on the daemon's launch command, and run the dial-out over `wss://` / a private network. **Note:** the `terminal` tool does not yet pass through `shell`'s ask/allow/deny approval gate — for now the token + private transport are the fence; a sender allowlist / approval gate is a planned follow-up.

**Enable it.** Add `terminal` to `tools.enabled`. On a deployment with a `~/.captain-claw/config.yaml` overlay (written by the Flight Deck settings UI), add it there too — that overlay's `tools.enabled` list **replaces** the project `config.yaml` list rather than merging — then restart the agent. Restart Flight Deck to pick up the relay router; run FD single-process (the relay's worker registry is in-process). All env vars are documented in `.env.example`. It's listed in the project `config.yaml` by default, but does nothing until the daemon is running on a target machine — until then it simply reports "not connected".

### Agent Forge

Agent Forge is a dedicated Flight Deck page that uses an LLM to decompose a business objective into a team of specialized AI agents.

**How it works:**

1. Navigate to **Agent Forge** in the sidebar
2. Configure **model tiers** and env vars on the **Library** page (see above) — persisted to your workspace
3. Describe your objective and **Decompose into Agents** — the LLM analyzes your goal and proposes a team, biased toward the curated archetypes

**Model tiers & the archetype gallery moved to the [Library](#library-new-in-057) page in 0.5.7** (see above). Forge now *consumes* the saved per-tier config — it shows a "Configure in Library" hint, every generated agent is assigned a tier that resolves to a concrete model at spawn, and the decomposition prompt stays biased toward the curated archetypes from `instructions/archetypes.json` (`GET /fd/archetypes`).

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
- Pick a **model tier** per agent, or set a custom provider/model (editing the model switches it off the tier)
- Toggle agent type (process or Docker)
- Designate a different lead agent (crown icon)
- Add or remove agents

**On spawn:**

- Each agent is created with its fleet-level instructions and tool configuration
- A team group is created (all agents added)
- A role group is created per unique role
- The lead agent's name includes `[Lead]` suffix

The forge system prompt includes the full Captain Claw tool reference (44 tools with descriptions) and guidelines for tool selection per role type (research, content, data, coordination, communication, automation).

### Frontier Horizon — Deep & Plan modes (NEW in 0.6.4)

Basna and Vatra parallelize **breadth** — a fleet answering once. **Frontier Horizon** adds optional **depth**: it spends test-time compute, gated by verifiers, to reach frontier-grade quality on cheaper models. It's surfaced on the Basna/Vatra page as a single **Effort** choice — **Standard · Deep · Plan** — with the chosen effort's options revealed below it (router tier and team size live behind a **Tuning** toggle). Everything is **off by default** (Standard = today's behavior). A **Help** button opens an in-app explainer.

**Deep mode** — frontier-grade depth on a single answer:

- **Basna (per-worker).** Instead of one shot per agent, each worker spawns a small pool of **independent rollouts**, **self-consistency-votes** across them, runs the leading answer past a panel of **diverse-lens critics** (the `phrygian`/`aeolian`/`locrian` cognitive modes as adversarial refuters), and **fixes** with the critique if it fails. The **Samples / agent** field sets the vote width.
- **Vatra (per-owner).** Spawning N copies of each specialist would flood the shared blackboard, so Vatra instead verifies **each owner's slice** with the critic panel and revises it once if a majority refute it — the same depth, blackboard-safe.
- **The closer (both).** After the answer is merged (Basna) or the deliverable assembled (Vatra), a critic panel reviews the **final** result and rewrites it once if it's refuted.

Critics always run on a model **different** from the one that produced the answer (never self-grading) — point your **critic / reason tier at a model that returns real `content`**, not a reasoning-only model. Deep is much slower and costlier than Standard; the live log streams **per-critic** progress and a **heartbeat** during the (slow) verification.

**Plan mode** — think way ahead, way long:

A planner decomposes the task into **ordered steps**, drives each to a **verified** result before the next begins, **re-plans** the remainder when a step can't be verified, and **synthesizes** the deliverable from the verified steps — so the system never compounds an unverified step.

- **Steps** — cap the plan length.
- **Each step runs as** — *simple* (one fast model), or *complex* → a full **Basna ensemble** or **Vatra team** per step (the mode card above decides which). When a step runs an ensemble/team, its **live agent activity is mirrored into the plan log**.
- **parallel** (request flag `dag: true`, **off by default**) — the planner emits a **dependency graph** instead of a linear list; independent steps run concurrently in dependency waves, and each step sees only its dependencies' verified outputs (`_dag_context`). Runs are self-describing: the persisted plan analysis carries `"dag": true`.
- **Select team** — a fixed team staffs **every** step's ensemble/team.
- Steps, fix attempts, and re-plans are all **bounded**; the run ends with an explicit reason and never silently drops an unverified step.

Plan runs in the background (`POST /fd/basna/plan`); the UI polls the live log. Plan-step child runs are real Basna/Vatra sessions — their reliability learning still closes — and are hidden from the session list under their parent.

**VFS provenance (NEW in 0.6.4)** — the Shared VFS viewer now shows **timestamps** on every file/folder, the **agent that wrote each file** (`✎ <agent>`, recorded in an append-only `.vfs-meta.jsonl` sidecar; the main interactive agent writes anonymously), and the **Basna/Vatra run title** behind each `basna-…` / `vatra-…` project, with a colored mode badge. Authorship is recorded going forward.

**Progress visibility (NEW in 0.6.5)** — every run now emits an explicit **`phase`** event at each transition and surfaces the latest one **stickily** in the Progress header (with a spinner while running), so the current stage is always visible no matter how far the detail log has scrolled — Vatra (Planning → Spawning → Intro → Main → Review → Verifying slices → Synthesizing → Verifying deliverable → Learning → Done), Basna (Routing → Spawning → Generating → Merging → Verifying → Done), and Plan (`Step x/y: <goal>`, plus Planning / Synthesizing / Done). Phase rows are accented in the log. In **Plan mode**, the live per-agent cards now also render a **running token count** — usage events from each step's child run are forwarded into the parent plan log, so the cards show the per-agent token header like a normal Basna run.

**Notes:** Deep/Plan settings ride in the existing session `config` JSON — no schema changes. New endpoint `POST /fd/basna/plan`; the closer/per-owner/plan paths reuse your Library tiers. Rebuild the Flight Deck bundle (`cd flight-deck && npm run build`) if running from source.

### Agent Council (NEW in 0.4.15)

Agent Council is a multi-agent deliberation system where connected agents discuss topics in structured turn-based rounds. It enables debates, brainstorming, code reviews, and planning sessions with full persistence and export.

**Session types:**

| Type | Purpose |
|---|---|
| Debate | Structured argumentation — agents take positions, challenge, and defend |
| Brainstorm | Creative ideation — agents build on ideas, suggest new angles |
| Review | Critical analysis — agents examine work and provide constructive critique |
| Planning | Task decomposition — agents break down tasks, identify risks, create steps |
| Interview | Focused Q&A — one agent asks, others answer in depth |
| Troubleshoot | Collaborative diagnosis — hypotheses, root-cause, fixes |
| Critique | Adversarial stress-testing — actively hunt flaws and gaps |
| Freeform | Unstructured discussion — no positional constraints |

**How it works:**

1. Navigate to **Council** in the sidebar
2. Set a topic, select session type, verbosity level, and max rounds
3. Choose the panel — **Auto-assemble** (default, NEW in 0.6.3) to have Council spawn a task-modeled team for you, or **Pick agents** to choose from your already-running agents (if Old Man is included, moderator mode activates automatically)
4. Choose the first speaker (or random) — Pick-agents mode only
5. Click **Start Council**

#### Auto-assemble — a panel modeled to the topic (NEW in 0.6.3)

Council can build its own panel instead of requiring agents you've already started — the same archetype auto-spawn Basna and Vatra use, tuned for deliberation. On the setup screen, the **Auto-assemble ↔ Pick agents** toggle (auto is the default):

- **Auto-assemble** — give a topic, pick a **panel size** (2–6), and Start. A **Council Assembler** router (fast tier + the merged archetype catalog) selects a *diverse* set of archetypes — optimizing for complementary, sometimes opposing perspectives rather than the minimal single-answer team Basna routes — and spawns each as a fresh ephemeral agent. Every panelist arrives **briefed for this exact council**: its archetype persona plus a tailored "your seat on this council" charge (the topic, the session type, and the specific angle it should bring). If the router LLM is unavailable it falls back to deterministic keyword matching, so a panel always returns.
- **Hand-pick the specialists (optional)** — open the specialist list (your full archetype gallery, base + custom) and choose exactly which archetypes form the panel. Selected archetypes are honored **in full**; the panel-size slider steps aside. Leave it empty to let the router compose the panel.
- Panelist models resolve from your active **Library tiers** (falling back to the registry tier defaults + the provider env var), exactly like Basna/Vatra.
- The deliberation then runs identically to a picked-agent council. Council waits until **every** panelist has connected before the first speaker runs, so a freshly-spawned agent is never skipped as "disconnected".

**Temporary agents & disposal.** Auto-assembled panelists are ephemeral. The session records their slugs (persisted in config, surviving reloads and every config rewrite), the sidebar shows a **Temporary** badge and a per-agent **temp** tag, and a **Dispose agents** button appears in the concluded controls bar and the sidebar Agents header. Dispose disconnects the sockets, tears the panel down (`POST /fd/council/teardown`), and marks the session **Disposed** (it won't try to reconnect dead agents) — keeping the transcript, synthesis, votes, and TL;DRs. **Deleting** a session disposes its panel automatically, so nothing leaks. Manual (picked-agent) councils show none of this.

Endpoints: `POST /fd/council/assemble` (route + spawn → council-ready agent defs incl. task-tailored fleet instructions) and `POST /fd/council/teardown` (stop + remove). Router prompt: `instructions/council/router.md`.

**Turn flow:**

Each agent receives the full discussion context (grouped by round) and responds with:
- **SUITABILITY** (0–1) — self-assessed relevance score
- **ACTION** — answer, respond, challenge, refine, broaden, or pass
- **TARGET** — optionally direct the response at a specific agent

These are displayed as color-coded badges on each message.

**Moderation modes:**

- **Round-Robin** — all agents speak each round, sorted by suitability score (highest first)
- **Moderator** — Old Man selects speakers based on suitability scores and discussion flow

**Council memory:**

Agents receive context from prior rounds when speaking. A configurable memory setting (5, 10, 20, 30 rounds, or indefinite) controls how many rounds of history are included, with a 30k token safety cap. The memory setting is adjustable during the session from the sidebar.

**Features:**

| Feature | Description |
|---|---|
| Verbosity control | Message (5 sentences), Short (3 paragraphs), Medium (5 paragraphs), Long (10 paragraphs) |
| Real-time activity log | Timestamped log showing tool usage, speaking status, connections, and system events |
| Tool activity | Shows last 3 tools used by each agent while speaking |
| Resizable layout | 50/50 split between discussion and sidebar, drag to resize |
| Pin messages | Pin important messages for quick reference in the sidebar |
| User injection | Send messages into the discussion or direct-address specific agents |
| Mute agents | Temporarily mute agents from participating |
| Synthesis | Request a synthesis document — moderator (or first agent) summarizes the discussion |
| Voting | After synthesis, all agents vote (agree/disagree/abstain) with reasoning |
| TL;DR | Each agent provides a 2-3 sentence personal summary — collapsible panel, re-generable |
| Minutes export | Export full session as Markdown (.md) with all rounds, synthesis, votes, and TL;DRs |
| Cognitive mode display | Shows each agent's current cognitive mode and eco mode status in the sidebar |
| Fleet instructions | Fleet instructions from Flight Deck are relayed to agents in the council |
| Restart round (NEW 0.5.6) | Re-run the current round from the top — aborts the in-flight round, discards its messages, and runs it again |
| Action Points (NEW 0.5.6) | After synthesis, each agent extracts its own outstanding next steps (scoped to its part — no full-transcript dump), shown per-agent. **Send** records a point into that agent's `todo`/`intentions` with full context preserved |
| Allow delegation (NEW 0.5.6) | Off by default — agents can't use orchestration tools (`flight_deck`/`task_contract`/`consult_peer`) during a turn, since the council is the coordination layer. Turn on for planning sessions |
| Allow passing | When off, every participant must contribute each round (a passed turn is re-nudged) |
| Narration log (NEW 0.5.6) | The activity log surfaces the model's between-step narration and reasoning, for debugging long/looping turns |
| Stuck auto-nudge (NEW 0.5.6) | If an agent returns the canned "I got stuck" reply, the council nudges it to continue (up to 3×) before moving on — and won't let the next agent speak until it does |
| Auto-assemble (NEW 0.6.3) | Council spawns its own task-modeled panel of archetypes (2–6, diverse perspectives), each briefed for the topic — or hand-pick the exact specialists. See [Auto-assemble](#auto-assemble--a-panel-modeled-to-the-topic-new-in-063) |
| Dispose agents (NEW 0.6.3) | Auto-assembled panelists are temporary; a **Dispose agents** button (and a Temporary/Disposed badge) tears them down at conclusion or any time, keeping the transcript and artifacts. Deleting a session disposes them automatically |

**Session persistence:**

All council sessions, messages, votes, and artifacts are persisted server-side in SQLite. Sessions can be reopened and reviewed after conclusion. TL;DRs and exports can be regenerated at any time.

As of **0.5.6**, persistence is hardened for long sessions: every write goes through a token-refresh-and-retry path with an in-memory retry queue, so an access token expiring mid-session no longer silently drops messages, votes, or the final `concluded` status. Long agent turns checkpoint partial output, so a mid-turn server restart preserves what was generated. A reopened session that was interrupted mid-round shows a recovery banner (Restart round / Synthesize / Conclude) rather than a stale "in session" state, and agents reconnect using their current token/port.

**Agent awareness:**

Agents in a council receive context about:
- The council type (debate, brainstorm, review, planning)
- Current round and max rounds
- Moderation mode and moderator identity
- All participating agents
- Fleet-level instructions from Flight Deck

### Fleet-Level Instructions

Fleet-level instructions are per-agent directives set from Flight Deck that are injected into the agent's system prompt. They apply to every conversation and take effect when the agent's chat session is opened.

**Setting instructions:**

1. Click **Actions** on any agent card, then **Configure**
2. Select the **Instructions** tab
3. Write or paste fleet-level instructions
4. Click **Save**

Instructions are stored in the Flight Deck frontend and sent to agents via the `peer_agents` WebSocket message when a chat session opens. No agent restart is needed — instructions take effect on the next chat connection.

In the agent's system prompt, fleet instructions appear under the heading **"Fleet-Level Instructions"** with the note that they come from the fleet operator.

### Owner Profile

Each Flight Deck user can describe themselves (**About me**, up to 1,500 characters) and their company (**My company**, 4,000), and give standing instructions (**My standing preferences**, 2,000). An admin can set deck-wide defaults: a company description and instructions that apply to everyone. The user's Captain Claw agents receive the result in their system prompt, alongside their own fleet-level instructions (who doesn't is listed under **Who gets it**).

**Merge:** the user's company text replaces the deck's when it isn't empty. The user's preferences and the deck's instructions are kept apart, the user's first, each under a fixed label: "From your owner:" and "From your Flight Deck admin (applies to everyone on this deck):". The deck's part always carries its label, even when it's the only instructions, so an admin's text never reads as the owner's own. When both are present the block adds that the admin's win a conflict (deck policy). What people wrote never starts a line of its own (it's quoted with `>` in the full form and kept on its label's line in the compact one), so no text can pass for a label. The owner is named by their display name, falling back to their email's local part, or "your owner" on a deck without accounts.

**Framing:** the agent is told that the background is reference data about its owner, not instructions, that people on shared channels (WhatsApp, Telegram, a shared or public chat) are not necessarily its owner, and to keep the section private: not to reveal or quote it to anyone but its owner. The preferences apply unless an explicit task, role or output-format contract, or a safety rule, says otherwise.

**Two forms:**

- **Full** (`tenant_context.md`): markdown with a section for the owner, their company and the standing preferences. Only the per-field caps limit its length.
- **Compact** (`tenant_context.compact.md`): at most 1,000 characters, one line each for About, Company, the owner's preferences and the deck's instructions. The room left after the fixed wording is shared out across those four, and a short one leaves its unused share to the longer ones, so a long deck text can't crowd out the owner's own preferences. A cut text ends in "…".

Eco mode is the default for new agents and runs the micro instructions, so **most agents receive the compact form**. The nano template and orchestrated workers (Council, Basna and Vatra teammates) receive it too. Only an agent on the standard instructions (eco and nano off, `context.micro_instructions` unset) that isn't an orchestrated worker receives the full form. The profile page previews both.

**Who gets it:** the classic Captain Claw agents that Flight Deck spawned or cloned for the user, as Docker containers or local processes. These don't get it:

- agents and workers on the Mrav micro runtime (it builds its own prompt);
- Iskra beings;
- public chat sessions;
- BotPort dispatch agents;
- Flight Deck's own direct LLM calls, such as planners and judges like the Vatra Lead decomposition.

An agent with no owner on a deck with accounts gets nothing (not even the deck defaults), and so do the agents of a deleted user.

**Routes:**

| Route | Who | What |
|---|---|---|
| `GET /fd/profile` | signed-in user | `{profile, deck, caps, preview: {full, compact}}` |
| `PUT /fd/profile` | signed-in user | partial update of `about_me` / `company` / `instructions`; same shape plus `agents_updated` |
| `GET /fd/admin/profile-defaults` | admin | `{company, instructions, caps}` |
| `PUT /fd/admin/profile-defaults` | admin | partial update; returns `{company, instructions, agents_updated}` and is recorded in the usage log (`profile_defaults_update`) |

A field is refused with 400 when it's over its cap, isn't a string, or contains text that can't be encoded as UTF-8 (a lone surrogate). `/fd/profile` is always the caller's own profile: an admin's `X-FD-Act-As` header does not open another user's. With accounts off (`FD_AUTH_ENABLED=false`), the local user owns the profile and may set the defaults. The generic `/fd/settings` routes can't read or write these keys (`fd:tenant-profile…`).

**Delivery:** Flight Deck writes both files into the `~/.captain-claw` that the agent's runtime reads, and only there. For a process agent that's `<agent dir>/data/home-config-parent/.captain-claw/`; for a Docker agent it's `<agent dir>/data/home-config/`. A process agent and a container with the same name share the agent dir, and this keeps them from reading each other's profile. Flight Deck writes the files:

- when an agent is spawned or cloned (a clone gets its recorded owner's profile);
- for all of the owner's agents when they save their profile or their display name changes;
- for every agent on the deck when the deck defaults change;
- for every agent on the deck once at Flight Deck startup, in the background. This catches an auth-mode switch or edits made while Flight Deck was down.

An empty profile removes the files, and so does deleting the user. A failed write removes them too, so an agent never keeps a stale profile. The agent re-reads them on each turn, so a change applies without a restart.

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

### Quick Free Agent (OpenRouter) — NEW in 0.5.3

The Spawn Agent page has a green **Quick Free Agent** button (above the
Docker/Process toggle). It spins up a process agent that runs entirely on
**free OpenRouter models**, for the price of a single free API key:

1. **Get a free key** — the modal links to [openrouter.ai](https://openrouter.ai)
   and `openrouter.ai/settings/keys`. Sign-up and free models cost $0 (subject to
   OpenRouter's free-tier rate limits).
2. **Fetch free models** — pulls the *currently-free* models from OpenRouter's
   public catalog (no key required just to list). The list is **filtered to
   tool-capable models** — agents rely on tools, so models without function
   calling are hidden.
3. **Pick a default** — every fetched free model is written into the agent's
   `model.allowed` list, so all of them are available to switch between at
   runtime; the one you pick becomes the default.

Free agents are tagged with a **"Freebie"** badge on their card. The underlying
endpoints:

- `GET /fd/openrouter/free-models` — public free, tool-capable model list.
- `POST /fd/spawn-free` — spawns the process agent (provider `openrouter`,
  base URL `https://openrouter.ai/api/v1`, key in `.env`, free models as the
  allowed list) and marks it `freebie` in the registry.
- `POST /fd/agent-refresh-free-models/{kind}/{id}` — re-fetches the current free
  roster and rewrites `model.allowed` (and the default if it dropped off) across
  **all three** config files. The agent **must be stopped** (config is read at
  startup). Exposed as **Refresh free models** in the card's Actions menu,
  enabled only when the agent is stopped.

### Switching the active model live — NEW in 0.5.3

Agent cards show an **Active model** dropdown listing the agent's *allowed*
models (advertised over the chat channel on connect). Selecting one switches the
running agent's model **live — no restart** — and takes effect on the next
message. This is the same per-session selection as the `/session model <id>`
chat command, surfaced as a control on the card. The list is populated once the
agent is running and connected.

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
- **Topics and other channels** (0.8.1) — one continuous transcript per lane with topic dividers, a Topics menu, and turns from WhatsApp, Telegram, devices and the glasses shown "via …"; see [The chat timeline](#the-chat-timeline-topics-and-other-channels--new-in-081)
- **Pictures go as images** (0.8.1) — a picture attached in the chat reaches the agent as an image, inline when its model sees images; see [Native Vision](#native-vision--new-in-081)

Chat connections are proxied through the Flight Deck backend to avoid browser CORS restrictions.

**Agent restarts are ridden out quietly (0.8.1).** When Flight Deck restarts, its process agents restart with it, and one whose port is taken comes back on another. The chat proxy now goes to the agent's current port (the browser names the agent as `proc-<slug>`), holds the socket while the agent comes up (up to `FD_AGENT_WS_WAIT_S`, default 30 s), and tells the browser where the agent is now. The chat shows "Agent unreachable — reconnecting…" instead of an error per retry, reconnects after 0.5 s, 1 s, then every 2 s for about a minute, then every 10 s, and refreshes its sign-in token once when a deck restart made it stale. The deck log gets one line when an agent goes down and one when it is back.

### Director Panel

The Director panel (Cmd+D to toggle) provides a unified overview of all agents:

- **Agents tab** — all agents listed with status, current activity, last interaction time. Filter by status, sort by name/status/activity/type. Expandable rows show description, host, created time, and recent message previews.
- **Activity Feed tab** — real-time feed of agent events
- **Broadcast** — send a message to all running agents simultaneously
- **Quick actions** — Stop All, Restart All
- **Resizable** — drag the right edge (220–500px, persisted)

### Operations Dashboard

A dedicated view for monitoring agent usage and costs:

- **Summary cards** — total tokens, estimated cost, API calls, average latency, data transferred, cache hit rate
- **Token distribution** — visual bar showing input/output/cache breakdown
- **Per-agent usage table** — sortable by tokens, cost, calls, latency
- **Model breakdown table** — usage grouped by LLM model
- **Agent health grid** — status cards for all agents
- **Period filter** — last hour, today, yesterday, this week, this month, all time

Cost estimates use published pricing for Claude, GPT, and Gemini models.

### Agent Pipelines

Pipelines chain agents together so output from one automatically flows to the next. Located in the Workflows view:

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
| Cmd/Ctrl+D | Toggle Director |
| Cmd/Ctrl+J | Toggle Chat Panel / Queue |
| Cmd/Ctrl+\\ | Simple / Full view (ignored in kiosk mode) |
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
| `FD_AUTH_ENABLED` | `true` | JWT-based multi-user sign-in. `false` = one local user, no sign-in (desktop / single-person decks) |
| `FD_JWT_SECRET` | (auto-generated per boot) | Secret key for JWT token signing. Set a fixed value on any shared deck — otherwise every login dies on restart; the inbound MCP OAuth and Lupa need it stable (Lupa's BFF must share it) |
| `FD_DOCKER_SPAWN` | `true` | Allow Docker container spawning (can be disabled by admin) |
| `FD_ALLOWED_HOSTS` | (unset) | **NEW in 0.8.0.** Extra host names this deck answers to, comma-separated (`.example.com` / `*.example.com` allow subdomains, `*` turns the check off). `localhost`, IP literals, this machine's hostnames, `FD_PUBLIC_URL`'s host and `FD_CORS_ORIGINS` hosts are always allowed. **A deck reached through its own domain must set this or `FD_PUBLIC_URL`, or every request is refused** |
| `FD_PUBLIC_URL` | (unset) | The deck's public base URL (`https://claw.example.com`); allowed by the Host guard and used for OAuth redirect/discovery URLs (Google, Microsoft, inbound MCP) and for public file links (unset: links an agent makes use the address people's browsers reach the deck at, then a domain from `FD_ALLOWED_HOSTS`) |
| `FD_PUBLIC_FILE_LINKS` | `on` | Public file links: a file's Share button publishes a copy of that one file at `/s/<token>` for anyone with the link (Markdown rendered, HTML run sandboxed). Agents can make one too when asked ("share that file") through the `public_link` tool. `off` hides the button, refuses the tool and stops every existing link from serving |
| `FD_PUBLIC_FILE_LINK_MAX_MB` | `50` | Largest file a public link may publish |
| `FD_CORS_ORIGINS` | (deck's own origins) | Extra browser origins allowed to call the deck. Since 0.8.0 the default is no longer `*`; `*` is the explicit legacy opt-out |
| `FD_LOCKDOWN` | (unset) | `1` = hosted mode: agent shared secret required even from loopback, host-filesystem routes off, `Secure` refresh cookie |
| `FD_BOTPORT_URL`, `FD_BOTPORT_DECK_NAME`, `FD_BOTPORT_KEY`, `FD_BOTPORT_SECRET`, `FD_BOTPORT_EXPOSE`, `FD_BOTPORT_ACCEPT_FROM`, `FD_BOTPORT_ALLOW_OUTBOUND`, `FD_BOTPORT_MAX_CONCURRENT`, `FD_BOTPORT_DENY_TOOLS`, `FD_BOTPORT_ENABLED`, `FD_BOTPORT_NO_PROXY` | (unset) | The deck's BotPort link to other Flight Decks, pinned by environment (otherwise Admin console → BotPort). See [docs/botport-deployment.md](docs/botport-deployment.md) |
| `FD_REGISTRATION_OPEN` / `FD_REGISTRATION_DISABLED` | (unset) | Self-registration is closed once the first (admin) account exists; `_OPEN=1` reopens it, `_DISABLED=1` forces it closed. Admins add or invite users on the Users page; while it is closed, "No account?" leads to the waitlist |
| `FD_COOKIE_SECURE` | (auto) | Force the refresh cookie's `Secure` flag on (`1`) or off (`0`, local http only) |
| `FD_SIMPLE_CHAT` | (unset) | `1` (or `captain-claw-fd --simple-chat`) = kiosk for every non-admin until an admin picks a mode in Admin → Settings |
| `CAPTAIN_CLAW_FD_HOME` | (unset) | Per-instance home for MCP servers, code apps and the agent secret (falls back to `FD_DATA_DIR`, then `~/.captain-claw-fd`) |
| `FD_AGENT_SHARING` | (unset) | `1` = allow sharing an agent with other users (needs auth on) |
| `FD_ARCHETYPE_GRID` | (unset) | `1` = function × domain archetype grid (`reviewer.legal`) |
| `FD_BAT_DISABLED` / `FD_BAT_SPEND` | (unset) | Bat is on by default — `FD_BAT_DISABLED=1` turns it off; real-money spend stays off unless `FD_BAT_SPEND=1` |
| `FD_GMAIL_SEND` / `FD_GMAIL_AUTOREPLY` | (unset) | Deck kill switches: `off` stops all agent Gmail sends / topic auto-replies |
| `FD_SCHEDULER_QUIET_HOURS` | (unset) | e.g. `22-08`: scheduled-task deliveries and payment reminders wait out that window (it may wrap midnight) and go when it ends |
| `FD_AGENT_WS_WAIT_S` | `30` | **NEW in 0.8.1.** How long the chat proxy holds a socket while an agent that isn't answering yet (restarting) comes up |
| `FD_DATA_KEY_FILE` | `<FD home>/data_key` | **NEW in 0.8.1.** Key file (mode 0600) that encrypts the Telegram inbox at rest; keep it apart from the database |
| `FD_DESKTOP` | (unset) | **NEW in 0.8.1.** Set to `1` by the desktop app. With sign-in off and a loopback bind, it lets the deck connect Google and Microsoft for its local user (system-browser sign-in) |
| `FD_PAYMENTS` | (unset) | **NEW in 0.8.1.** `1` turns on the [payments](#payments-bill-pay-and-purchases--new-in-081) tool, the Payments page and the `/p/<token>` phone pages |
| `FD_PAYMENTS_STATEMENTS` / `FD_PAYMENTS_SYNC_TIMES` | (unset) / `07:30,16:30` | **NEW in 0.8.1.** `1` turns on read-only bank statements; the daily times they are read |
| `FD_ENABLEBANKING_APP_ID` / `FD_ENABLEBANKING_KEY` | (unset) | **NEW in 0.8.1.** The deck's Enable Banking application (pay-by-bank links and bank statements) |
| `FD_LINK_CLIENT_ID` / `FD_LINK_CLIENT_SECRET` / `FD_LINK_PUBLISHABLE_KEY` / `FD_LINK_TEST` | (unset) | **NEW in 0.8.1.** Stripe Link agent wallet for purchases; `FD_LINK_TEST=1` makes test requests |
| `FD_AP2` | (unset) | **NEW in 0.8.1.** `1` turns on AP2 checkout with merchant agents |
| `FD_AGENT_CONFINE` | on with sign-in | **NEW in 0.8.4.** `off` turns off [confinement](#confinement-on-a-multi-user-deck--new-in-084), for a deck whose users all trust each other |
| `FD_TASKBOARD_DISABLED` | (unset) | **NEW in 0.8.4.** `true` stops the [Taskboard](#taskboard--new-in-084)'s sweep and triage; it then fills only when opened |
| `FD_SETUP_TOKENS_PER_MIN` / `FD_SETUP_REASONING` | `40000` / `off` | **NEW in 0.8.4.** Pace and thinking level of a [setup read](#onboarding--new-in-084); `0` is unpaced, `auto` leaves the model's own level |
| `FD_PROFILE_REASONING` | `low` | **NEW in 0.8.4.** Thinking level for the [Profile](#profile--new-in-084) researcher |

A deck on more than one person's network: see [docs/team-deployment.md](docs/team-deployment.md) for the full checklist (`FD_AUTH_ENABLED`, `FD_JWT_SECRET`, `FD_LOCKDOWN`, `FD_AGENT_SHARED_SECRET`, `FD_GLASSES_BRIDGE_TOKEN`, …).

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

## Kiosk (Simple Layout)

The **simple layout** is Flight Deck's chat-first layout: agents on the left, the chat in the middle. **Kiosk mode** locks a user into it — for people who only chat with their agents, and for shared-account machines. This section describes the kiosk as a locked user sees it. Admins and unlocked users get the same simple layout plus the active agent's files and datastore in a right-hand column, and switch to the full layout with **Cmd/Ctrl+\\** (see [The simple layout and kiosk mode](#the-simple-layout-and-kiosk-mode--new-in-080)).

### Who gets kiosk mode

An admin decides in **Admin → Settings → Kiosk mode** (or the [Admin console](#admin-console)'s Settings page):

- **Off** — nobody is locked.
- **Everyone** — every non-admin.
- **Selected users** — tick **Kiosk mode** on a user (Users page, or the Add user and Invite forms); a **kiosk** badge shows who is locked.

**New users start in kiosk mode** pre-ticks the mark for accounts created from then on (Add user or self sign-up). `captain-claw-fd --simple-chat` (or `FD_SIMPLE_CHAT=1`) only sets the starting value (Everyone) until an admin picks a mode. The server resolves the lock per user and returns it with the user on sign-in, token refresh, `/me`, registration and device pairing; a signed-in user picks up a change on the next token refresh (within about 15 minutes) or a reload. Admins are never locked. A deck without sign-in goes by the deck-wide flag.

Locked users have no full layout and no agent options beyond what this section describes. They create agents from archetypes (models and keys come from their own tier set or the team default), manage their own connections, edit their profile, and sign out cleanly (per-browser queues and plans are purged). It is a UI lock, not an API boundary. A kiosk deck needs a published team tier set (or users with tier sets of their own) and enough agent slots.

### Getting started guide — NEW in 0.8.1

A kiosk user's first visit opens a step-by-step guide: what the deck is, creating an agent from an archetype, giving it work (files, the queue, new topics), connecting Google or Microsoft, the team Telegram bot, the profile, and where everything else lives. Each step shows a real screenshot of that screen in the viewer's theme — on a phone, the phone's own screens — that enlarges on a click, and where it fits a button straight to it (**Create an agent**, **Connect Google**, **Set up Telegram**, **Edit my profile**). Steps for a connection the deck doesn't offer are left out, and chips show what the user has connected already. On a phone a sideways swipe turns the page.

**Do not show again** is a per-user setting stored on the server, so it holds on a shared kiosk and across devices. **Info**, **Settings → Getting started** and the empty chat open the guide again.

### The kiosk screen — NEW in 0.8.1

- **Left column:** the agents, each with a **Busy** pill, an on/off power switch and a **⋯** menu (see [The agent menu](#the-agent-menu--new-in-081)); **New agent**; the tiles — **Tasks · Scheduled · Library · Data · Folders**; then **Info** and **Settings**. The column collapses to a rail with the same buttons.
- **The chat** in the middle. There is no right-hand column: files and tables live in the [File library](#file-library--new-in-081) and the [Data library](#data-library--new-in-081).
- **Info** (`GET /fd/info`) shows the user's corner of the deck in one read: the tier set their agents are created on (their own Library set, else the admin's team default) with each tier's provider, model, endpoint host, context sizes, image support and how many archetypes use it — never a key; their agents and each one's model; archetype counts by source; the deck-wide profile defaults, their own profile, and the exact text their agents receive (full, and the compact one eco agents get); their plan with today's usage meters and free resets; and their Google, Microsoft, Telegram and Jev status.
- **Settings** is one button opening a dialog: light / dark theme, **Connections** (Google, Microsoft, Telegram, MCP Servers, Jev and Soniox), **Scheduled tasks**, **Taskboard**, **Activity**, **File library**, **Data library**, **Shared folders**, **Profile** (with its **Password** tab), **Your own AI keys**, **Refresh agents**, **Getting started** and **Sign out**.
- **New agent** opens a two-pane archetype picker: search by role, task or skill, family filter chips with counts, and cards with each archetype's model class and thinking style; the detail pane explains the pick ("Runs on", "Thinks", "What it can do") above the name field and the **Create** button. Enter on a narrowed search picks the first card.
- The deck's default name is **Captain**, with the Captain icon, until an admin sets an instance name.

**On a phone** (narrower than 768 px, or a short touch screen in landscape) the kiosk shows one screen at a time: the agent list (big rows, power switch, a New agent bar) or one chat full screen with a back arrow, and the phone's own Back returns to the list instead of leaving the deck. Dialogs open full screen and Settings as a bottom sheet; the screen resizes with the on-screen keyboard. The chat drops its tab bar (lanes read A / B / C / Auto; the queue opens as its own view), Enter on a touch keyboard is a new line, and form fields are 16 px so iOS doesn't zoom into them. Tapping a file opens a sheet with View, Download and Share (the phone's own share sheet).

### The agent menu — NEW in 0.8.1

Each of the user's own agents has a **⋯** menu (on its row, and on the phone list and the phone chat header). Since 0.8.4 it opens as a card of icon tiles, a sheet on a phone, with **Remove agent** set apart:

- **Info** — a dialog with two tabs, the agent's **Intentions** and its **Topics** (while it runs).
- **Instructions, insights & skills** (0.8.4) — three tabs. **Instructions** holds the owner's standing instructions for this agent. Saving sends open chats the agent's identity again, so they apply from its next reply without a reconnect. **Insights** shows what it has learned as cards, with the document each came from, to search, teach, correct or forget. **Skills** lists its skills to read, switch on or off, and correct where the agent keeps them itself.
- **Onboarding** (0.8.4) — its documents, lessons, practice and exams; see [Onboarding](#onboarding--new-in-084).
- **Tasks** (0.8.4) — the [Taskboard](#taskboard--new-in-084), on this agent.
- **Settings** — a wide dialog (a sheet on a phone), with two tabs since 0.8.4: *Model & thinking* holds model, vision and how it thinks, and *Reasoning effort* the rest:
  - **Model:** provider (only providers an agent can run on — `PUT /fd/agent-model` refuses others), model, API key and an optional endpoint. Switching provider drops the old provider's key and resets the endpoint to the new provider's default unless you give one.
  - **Vision:** auto / on / off (see [Native Vision](#native-vision--new-in-081)).
  - **How it thinks:** the cognitive mode — Neutra, Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian or Locrian.
  - **Reasoning effort:** a default and a level per source, grouped — Messaging (WhatsApp, Telegram, Messenger, Discord), Apps and devices (the app, glasses, each M5Stack device), Automated (scheduled tasks, flows, other agents, autonomous work, API and MCP) and Other people (shared-agent members, public links). The web chat's level is the chat box's picker. See [Reasoning Effort](#reasoning-effort--new-in-081).

  Model and vision are read when the agent starts, so the dialog offers **Restart now**.
- **Remove agent** — stops the agent and deletes it for good: its folder under `FD_DATA_DIR`, Flight Deck's chat log for every lane, and the scheduled tasks that run it (`DELETE /fd/processes/{slug}?purge=true`). The browser forgets its labels, queue and plans, so a later agent of the same name starts clean.

**Files and Data are no longer in the menu**: they repeated the **Library** and **Data** tiles, and the File library's "Everything it keeps" opens an agent's live files-and-data view. Agents the libraries don't index — one shared with you, a Docker agent — keep a **Files & data** entry, their only way to their files. Agents shared with you, and agents Flight Deck runs itself, have no Instructions, Settings or Remove.

### The chat — NEW in 0.8.1

- **A conversation app.** Larger, centred messages, and a roomier composer with attach, paste, **What can I do?**, the reasoning-effort button, "Enter to send" and a round send button. An empty chat or a fresh topic opens with a greeting ("Good afternoon, <name>.") and the agent's top three skill cards.
- **Skill cards ("What can I do?").** One-click tasks: a title, a short line, and the prompt a click sends. Flight Deck writes them per agent with the owner's fast tier, from the agent's archetype, instructions, tools and connected accounts, its scheduled tasks and cron jobs, and its topics, intentions, insights, latest reflection and playbooks (`GET /fd/agent-skills/{slug}`, owner only). Six are written up front, cached in the agent's folder, and rewritten when the agent's make-up changes or they are 12 hours old; a failed rewrite keeps the last good set. Without a model, cards come straight from the agent's schedules, intentions and topics.
- **More.** The composer's list has **More · N** (and the greeting has "More things I can do…"), opening a dialog with the six **Suggested** cards and a **More** list of up to 16 more. The More list is written only when asked for (8 at a time, `POST /fd/agent-skills/{slug}/more`); the dialog says when it was suggested and offers **Suggest again**, and for a list from an earlier day a banner offers new ones — it is never rewritten by itself.
- **Cards that ask for input.** A card that works on the person's own material (an email to answer, claims to check) says what it needs. Picking it opens a dialog with its question, a text box, and **Attach a file** / **Add a picture** as the card accepts (paste and drop work too). **Start** sends the prompt, the text and the attachments; a required answer must be there first. These cards are marked in the greeting, the list and the More dialog.
- **Reasoning effort.** The button beside "What can I do?" offers **Auto** (the model's own default) plus exactly the levels the agent's model offers. The pick is kept per agent and sent with every message. See [Reasoning Effort](#reasoning-effort--new-in-081).
- **New topic** and **Topics** are buttons beside the lane tabs (see the next section); the lane tabs scroll sideways without a bar.
- **Daily allowance.** Under the lane tabs a line shows today's messages and AI tokens against the user's plan (always in the kiosk; in the full layout only near the limit). When the allowance is spent the composer closes until the UTC day resets or the user spends a free reset. See [Daily plan limits and free resets](#daily-plan-limits-and-free-resets).

**Files menu (0.8.4).** A **Files** button beside "What can I do?" lists the last five files the agent made in this chat, newest first, with **Open** and **Show in library**. A file counts when the reply names it or a file-making tool reports it, never what you attached.

**Compact cards (0.8.4).** Results the automation lane mirrors into the chat (Autonomous Work, plans, scheduled jobs) and the agent's rephrasing of your request show as a small card labelled with what produced them, with a two-line preview and **Open**.

**Save as a lesson (0.8.4).** On your messages to your own agent, a hover action, and a chip when a message reads like a correction or something to remember. See [Onboarding](#onboarding--new-in-084).

### The chat timeline: topics and other channels — NEW in 0.8.1

The chat is one continuous transcript per agent lane, in both layouts, the way the WhatsApp and Telegram bridges already behave.

- **+ New topic** (on a phone **+ New**) starts a new topic: the agent's `/new` still resets its context, but the screen keeps everything said so far above a divider. Going back to an earlier session shows "Back on …", and `/clear` shows "Context cleared". A typed "New topic: <question>" starts a new topic with that question as its first turn.
- **Topic titles.** When `/new` leaves a topic behind, the agent names it in the background (one small model call over its opening turns). Dividers read "New topic · <title>", a resumed topic keeps its name, and the Topics menu shows it with a pencil to rename it.
- **The Topics menu** (the history icon beside the button; a bottom sheet on phones) lists one row per divider, to jump back to it or resume it (the agent's `/session switch`).
- **Paging.** "Earlier messages" pages back through the log, and "Later messages" pages forward again. The chat keeps at most 2,000 rows loaded, so a tab left open for weeks stays quick; a jump to an old topic loads 500.
- **Other channels' turns.** When the same agent is used from WhatsApp, Telegram, a voice device, the glasses or the app, those turns appear in the web timeline as they happen — the question too, labelled with where it came from ("via WhatsApp", "via Telegram", "via M5Stack StickS3 (stick-a1b2c3)") — and are saved with the label.
- **The model knows the source.** Every user message the agent stores carries its `source` — `web`, `app[:android|browser]`, `glasses[:meta-display|rokid-lumen]`, `whatsapp`, `telegram`, `messenger`, `discord`, `device:<kind>`, `scheduled`, `flow`, `agent`, `autonomy`, `api`, `public` or `member` — and a label a person reads. In the history the model sees, a message from anywhere but the web chat reads "[via WhatsApp] …" (the stored text is unchanged), and the turn's note names where the current message came from. The source also picks the agent's [reasoning effort](#reasoning-effort--new-in-081) for that turn.
- **Switching by asking.** Where there is no Topics menu (WhatsApp, a device, the glasses), "go back to the Rome trip" works through the [`sessions`](#sessions--new-in-081) tool.

Flight Deck's chat log is the transcript: `GET /fd/chat/sessions/{id}/topics`, `GET /fd/chat/sessions/{id}/messages?since_id=`, and `PATCH /fd/chat/sessions/{id}/topics/{row_id}` to rename a topic. Agents must run 0.8.1: an older agent's `/new` still clears the screen. Design notes: [docs/chat-timeline-plan.md](docs/chat-timeline-plan.md).

### The message queue — NEW in 0.8.1

The queue is redesigned. One card on top says what is happening and offers the next step; below it are **Up next** (numbered, reorderable, editable) and a folded **Done** list with times and tokens. **Undo** covers remove, skip and clear, and the mode switch reads **Auto | Step by step**. The same queue runs in the simple layout's column (hidden by default in the kiosk), as its own view on a phone, and in the fullscreen sidebar, where collapsing folds just the queue into a bar.

After **Stop** the lane stays busy until the stopped turn has ended, and the stop's reply is never counted as the task's result. A busy refusal sends the task back to waiting; a reload doesn't re-send held tasks; a task held because the daily allowance is spent retries once it resets; the next task starts when the agent says it is ready. The earlier queue behaviour is described in [The task queue](#the-task-queue--long-work-that-runs-unattended-new-in-078).

**Insert now (0.8.4).** While a turn runs, a queued message shows **Insert now** (also in the row's ⋯ menu, and on the `I` key). It goes to the running turn as a `btw` note, which the model reads on its next step, and the row retires to Done. Inserting never starts the next queued task. If the turn ends first, the task goes back to the front of the queue and runs on its own.

### File library — NEW in 0.8.1

**Library** (the left column's tile, the rail, the phone header and Settings) is every file the user's own agents made, summarised, sorted and searchable. Flight Deck walks the workspaces of the caller's own process agents on disk — stopped agents too; Flight Deck's own workers are left out — and for each new or changed file reads its text (PDF, DOCX, XLSX and PPTX through the agent's extractors) and has the owner's fast tier write a title, a short summary, one of 12 categories (Report, Research, Notes, Plan, Data & lists, Finance, Legal, Correspondence, Presentation, Code, Images & media, Other) and a few tags. Results are kept in `file_library.db`; a file is summarised again only when its size or modification time changes. `.env` files are listed but never read; archives and unreadable formats are listed without a summary.

- **Pictures, recordings and videos.** Pictures are described by a model that can see — the owner's library tier if it sees pictures, else the agent's own model when the agent's vision is on — and any text in them is read word for word for search. Recordings and videos are transcribed with Soniox (the sound taken out with ffmpeg) and summarised from the transcript. A file set aside for want of vision, a Soniox key or ffmpeg is taken up by the first scan that has it.
- **Search** goes through names, titles, summaries, tags, the text, picture descriptions and full transcripts, and category and kind names. Filter by agent, category and kind (with counts) and sort; result cards highlight the matches and show text snippets.
- **Details and preview.** The detail pane has **Open**, **Download**, **Summarise again** and a jump to the agent's chat, over a preview of the selected file: a picture, a video or audio player, a PDF's first page, rendered Markdown (tables too), a web page as a small picture of itself, or the start of a text, code or data file (CSV and TSV as a table, JSON laid out). Office files and archives say there's no preview. The PDF preview (`GET /fd/file-library/files/{id}/thumb`) needs `pypdfium2`, which comes with the `payments` extra; without it the preview offers **Open**. A file someone else wrote, or one whose agent is off, previews without scripts or images.
- **Actions:** **Edit** (in the [file editor](#file-editor-flight-deck)), **Present as deck**, **Public link** (see [Public file links](#public-file-links--new-in-081)), the phone's share sheet, and **Everything it keeps** for the agent's live files-and-data view.
- **Costs.** The model calls count toward the owner's daily token allowance; a scan stops when it's spent, or after repeated model failures, and says why. **Rescan** also retries failed files.

Routes (owner-checked): `/fd/file-library/status`, `/fd/file-library/scan`, `/fd/file-library/files`, and `/fd/file-library/files/{id}/summarize|view|download|thumb`.

### Data library — NEW in 0.8.1

**Data** is the File library's sibling for datastores. Flight Deck opens each of the caller's own agents' `datastore.db` read-only from disk (stopped agents too; Flight Deck's own workers left out) and, for every new or changed table, profiles each column — how full it is (placeholders like "unknown" or "TBD" don't count), distinct values, number ranges and sums, date ranges, yes/no splits, link and email hints, the most common values. The owner's fast tier then writes a title, a summary, a category, tags, key facts and a line per column from that profile plus sample rows. Summaries are kept in `data_library.db`; rows are always read live.

- **A table** opens on its summary (stat tiles, key facts, tags) over a live data grid: readable headers, column-type icons, clickable links and emails, formatted numbers and dates, paging, search across every column, sort by any column, and a record panel per row with copy-as-JSON. A **Columns** tab shows the per-column profiles. On a phone: Summary / Data / Columns tabs.
- **Search** goes through table names, summaries, columns and the data itself; filter by category or agent.
- **Export** as CSV (opens straight in Excel), JSON (one object per row) or XLSX: `GET /fd/data-library/tables/{id}/export?format=csv|json|xlsx`. A CSV neutralises spreadsheet formulas in rows a shared-agent member added. **Refresh** reads the table again.
- The model calls count toward the owner's daily token allowance, as for the File library.

### Scheduled tasks — NEW in 0.8.1

**Scheduled** runs the user's agents on a timer. A task runs a prompt on one of their agents and delivers the result to any of:

- **Chat** — the agent's main chat (the kiosk timeline, and the chat bridges bound to the agent, such as WhatsApp);
- **Email** — from the user's own connected Gmail or Outlook, to themselves or another address;
- **Telegram** — the user's chat with the deck's team bot.

**Schedules:** `every <N>m|h|d`, `daily HH:MM`, `weekdays HH:MM` (Monday to Friday), `weekly mon,wed HH:MM`, and the one-shots `in <N>m|h|d` and `once <ISO time>`. Wall-clock times are in the task's own time zone (the browser's, saved with the task). The editor previews the next runs.

**One session per task.** Every task runs in its own agent session, kept across runs, so its runs read as one conversation and the agent sees what it said last time. A **Scheduled** chip on a task's result in the chat opens the task, and **Continue in chat** switches the agent's chat onto the task's session.

**Run history.** Every run is logged with its result and how each delivery went: an **All activity** feed with search, and a history per task. **Run now** runs a task in the background.

Due tasks run concurrently, one at a time per agent, and wait up to 20 minutes for a reply. When the owner's daily token allowance is spent, tasks are skipped (a one-shot retries after the reset). `FD_SCHEDULER_QUIET_HOURS` (e.g. `22-08`) holds deliveries until the window ends. A non-admin's task can't drive another owner's agent, and an agent can't choose whom a task emails. Routes: `/scheduler/jobs`, `GET /scheduler/runs?q=`, `POST /scheduler/preview`, `GET /scheduler/delivery-options`, and `POST /scheduler/jobs/{id}/run?background=1`.

### Onboarding — NEW in 0.8.4

**⋯ → Onboarding** is one page per agent: **Overview** (the last exam, how much was read, what to do next), **Lessons**, **Practice**, **Materials** and **Exams**. Tick *Set up its way of working next* in the New agent dialog to start there; admins get a button in **Users → Agents**.

**Way of working** (the setup dialog):

- **Material.** Up to 12 documents by drag and drop, upload or from the File library, up to 3,000 PDF pages each, each with how the agent should use it (`auto`, rules, procedures, reference, templates), plus your instructions.
- **Knowledge base.** Every document goes to `knowledge/` whole, as an overview, parts under 200 KB and a `<name>.map.md` of its sections. It costs no tokens. Semantic memory indexes it first under its own cap (`memory.max_knowledge_files`, 3000), and the [`knowledge`](#knowledge--new-in-084) tool reads it. PDFs keep their headings and tables with `pip install 'captain-claw[pdf]'` (PyMuPDF, AGPL-3.0); otherwise they're read page by page.
- **Reading.** How much to read: Quick (60,000 tokens), Standard (250,000) or Deep (1,000,000), with an optional **Focus on**. Material over the budget is read by its most instructive sections. A read runs in the background on the agent's own model with thinking off (`FD_SETUP_REASONING`), and survives closing the dialog and a Flight Deck restart. A step carries at most 16,000 tokens of material and calls are paced to 40,000 tokens a minute (`FD_SETUP_TOKENS_PER_MIN`, `0` = unpaced; models on the same machine aren't paced). A rate limit is waited out for about five minutes, then the read stops and keeps what it found. The owner's tiers read only when the agent is off or stops answering.
- **Review.** Rules, ask-first items, skills, facts, templates and a glossary, each with its source. A longer read keeps more: up to 150 rules, 30 skills, 1,000 facts, 30 ask-first items, 20 templates and 400 terms; three times that in all when adding to an earlier setup.
- **Spreadsheets.** An `.xlsx` or `.csv` also becomes one datastore table per sheet, with no model reading it. An agent that's off finds them in `pending-tables.json` when it starts.
- **Add or Replace.** Add puts new items first and dedupes; a document given again replaces its files. Replace removes earlier rules, skills and documents; learned facts and the agent's own files stay. Each setup is recorded in `.owner-setup.json`.
- **Read more later.** Each document records which sections were read; picking it again reads only the rest.

**Lessons.** A setup's rules, ask-first items and facts, and what an exam, practice or chat teaches. Flight Deck writes `~/.captain-claw/lessons.json` for the agent. *Always in mind* lessons stay in its instructions block; the others are picked per turn by words and by meaning, at most 6 items and 1,800 characters. An agent that hasn't restarted keeps its old full block.

**Exams.** Quick (10 questions), Standard (25), Deep (60) or Marathon (120), on all documents or a pick. The agent's own model writes the questions; each is asked through its real chat on a private `exam` lane that only reads and keeps nothing; each answer is graded point by point, and the verdict is worked out in code. A wrong answer's cause comes from what the agent did: didn't look, looked in the wrong place, found it and got it wrong, made things up. **Teach it this**, **Read the parts it missed**, and asking the same again for better or worse. An agent without the capability is told to restart, never asked.

**Practice.** A task you give, or one of three suggested from its documents, answered on the same lane. **Good**, or **Fix it**, which drafts a lesson and tries again. Kept in the agent's `practice.json`.

**Team layers.** What an admin teaches, acting for a user, is the agent's base: the user can't reword, pin or remove it, and their own Replace keeps it. A shared agent's member can *Suggest as a lesson*, which waits for the owner. **Pass it on** packages documents, tables, lessons, skills and the last exam's questions as a new archetype.

**Materials** browses the documents: a read meter, sections as a tree marked read or searchable only, a reader and search (`/fd/agent-knowledge/{slug}`, owner only).

### Taskboard — NEW in 0.8.4

Everything a user's agents were asked and sent, on every channel, in `taskboard.db`, per user. Open it from the **Tasks** tile, **⋯ → Tasks**, **Settings → Taskboard** or the phone header. Lanes: **Needs me**, **Waiting on others**, **Agents working**, **Done**, **All**. A member's turn goes to the member's board only; owners and admins never see it.

**Sources.** Scheduled jobs, Autonomous Work proposals, follow-ups, held Gmail auto-replies and sends, and linked systems, swept 20 s after start and then every 120 s. Agents report each turn through the turn register (`captain_claw/turn_register.py`, `CLAW_TURN_REGISTER=off` stops it). Agents' to-do lists show as tasks. The last 30 days of each agent's chats are read once, marked as history, and never proposed on or notified. `FD_TASKBOARD_DISABLED=true` stops the sweep and triage; the board then fills only when opened.

**Inboxes.** Telegram inbox, WhatsApp Web and Gmail are each off until the user consents. They read only that user's data, keep it 30 days, and delete it when switched off, names and addresses included.

**Triage.** One call on the agent's own model (owner's tier as fallback) writes the title, ask, result and parties. Jev, on the board user's own or the team's key, answers whether it's a task, who owes the next move, whether it's fulfilled, and whether it should reopen. Without Jev, rules decide. **Why is this here?** shows each decision. Board settings hold auto-close (on) and daily caps (200 Jev calls, 100 readings; `0` = none).

**Working a task.**

- `j`/`k` move, `o` or Enter open, `u` back, `/` search, `g` then `n`, `i`, `o`, `w`, `d` or `a` for a lane or view, `?` for the list. On a task: `e` Done, `U` Reopen, `I` Got it, `!` Not a task, `y` Yes, `b` Snooze, `l` Rename, `v` Merge into, `r` next step, `m` Next move, `a` the agent, `f` Open bigger, `z` Undo. Nothing approves or rejects a next step by key.
- A user's correction locks that field against later sweeps, with Undo.
- A person's reply brings the same task back to Needs me, marked *New reply*: after Done, with the next move someone else's, or snoozed. Not a task, Dismiss, mute rules and an agent at work on it stay as they are.
- **Open bigger** shows the task in a centred column. A task with a Gmail thread shows the whole email, read with the user's token when opened and kept nowhere.
- **Merge, unmerge, split** with Undo. Jev can suggest a merge but never applies one.
- **Export** CSV or JSONL of your own tasks, with the board's filters. `?taskboard=<id>` opens a task.

**Next steps.** With Jev, the agent's own model drafts a reply, a reminder or a clarifying question. It waits as an Autonomous Work proposal you edit, approve or reject, through **Review**, never one click. Email only becomes a draft; chats use `reply.on_origin`, delivered to the task's own origin. Recipients come only from verified parties. None on a task flagged as carrying instructions, on history, or without Jev. They expire when the task closes or after 7 days.

**Hand-over.** **Next move → The agent…** asks which agent and what it should do. Nothing goes until **Hand over**. It runs as the user's own request in a session of its own, with their words first and the task fenced as data; sending still goes by the Email sending switch and its recipient list. One hand-over at a time per task.

**Proof.** Triage checks each run (`taskboard_proof.py`) with no model: the tool-call register says what the tools answered in that turn, and Gmail and Google Calendar are read with the user's token. Jev then compares the user's words with the record; without Jev the result is never "proven". **What the agent did** shows the verdict. In doubt the task stays open and the user's, and gets no next step.

Plan and known gaps: [docs/taskboard-plan.md](docs/taskboard-plan.md).

### Your own AI keys — NEW in 0.8.4

**Settings → Your own AI keys**: Anthropic, OpenAI, Google Gemini, OpenRouter or xAI (`captain_claw/flight_deck/byok.py`). Each key is checked with the provider before it's saved, sealed at rest and never returned.

While on, the user's tier set runs on those keys: a tier on a provider they have a key for keeps its model, any other moves to one of theirs (pickable per tier), and local tiers stay. New agents, Flight Deck's own calls for them and their existing agents follow; running agents restart. Their tokens go to `daily_usage.byok_tokens` and don't count against daily limits. On a shared agent, the owner's keys count. Turning it off puts the old models back. On by default; an admin can turn it off for the deck, which moves everyone back.

### Profile — NEW in 0.8.4

**Settings → Profile** has three tabs:

- **Personal** — name, links (LinkedIn, GitHub, a site), company name and links, uploaded documents, and the texts themselves. **Fill it in for me** reads the documents, visits the links and a company site's About page, looks the person and the company up on the web, and drafts *About me*, *My company* and *My standing preferences*, on the team's published tier set (the person's own tiers if none is published; `FD_PROFILE_REASONING`, default `low`). The run is followed live and kept across reopenings, and the bell says when it's finished. Nothing reaches the agents until **Use** and **Save**. Links to private hosts are never fetched.
- **Deck-wide** — an admin edits the defaults in place; everyone else reads them.
- **Password** — on a deck with accounts.

### Desktop app mode — NEW in 0.8.1

The desktop app (`desktop/`, Electron; `build-desktop.sh` builds it) runs its own Flight Deck on a free port from 25090–25099 (remembered), with sign-in off and one local user. There is no full Flight Deck in desktop mode: an admin gets the [Admin console](#admin-console) or the real, locked kiosk, switched with **Admin view** / **User view** under the deck name ("Switch to user view" in the console, a shield button back in the kiosk); members always get the kiosk. On launch, a deck with agents opens the kiosk; a deck with no tier set opens the console on **Models**, where an empty state leads you to create one. The console leaves out what needs accounts: Users, Usage & limits, sign-up, kiosk mode, publishing to the team and sharing. In any browser tab, `?app=desktop` simulates the mode and `?app=web` turns it off.

**Google and Microsoft sign-in in the app.** The desktop deck connects Google and Microsoft for its local user the native-app way (RFC 8252): sign-in runs in the system browser and comes back to the deck's own loopback callback (`http://localhost:<port>/fd/google/callback`, `/fd/microsoft/callback`), with PKCE. The app shows "Finish signing in in your browser" with **Cancel** while it waits (up to 10 minutes).

- **Google:** an OAuth client of type **Desktop app** takes the callback on any port, with nothing to add. An existing **Web application** client works once the exact redirect URI, port included, is on its list.
- **Microsoft:** register `http://localhost/fd/microsoft/callback` under **Mobile and desktop applications** (Microsoft ignores the port for localhost). The client secret is optional for such a public client.

Only a deck with sign-in off, `FD_DESKTOP=1` (the app sets it) and a loopback bind opens these routes; every other deck without sign-in still refuses Google and Microsoft. Tokens are kept for the local user, not in the deployment-wide keys.

---

## Deck Administration — NEW in 0.8.1

What an admin does to set up and look after a team deck. Deploying one (auth, secrets, domains, the checklist) is in [docs/team-deployment.md](docs/team-deployment.md); managing users' agents is in [Running Flight Deck for a team](#running-flight-deck-for-a-team--new-in-080).

### Admin console

A layout of its own for admins, with no chat and no agent desktop — just setting the deck up and looking after it. Every page has an address, `<deck>/console/<page>`, so a link opens it directly and Back / Forward move between pages:

- **Overview** — people, agents running, messages today, 30-day spend, waitlist and Telegram requests to review, a setup checklist, and the machine's load, memory and disk.
- **Models** — tier sets as cards, each tier opening an editor (context sizes, abilities, key and endpoint); one model for every tier; publish to the team, with "changed since published"; team API keys; and **Extra keys for agents** (environment variables passed to spawned agents) on a full-width row as a Name / Value table, each value behind a show / hide toggle (on a phone the value stacks under the name).
- **Archetypes** — search and filters, a sectioned editor with "draft from a description", read-only built-ins with **Make a copy**, forge and share.
- **Connections** — each service with its status. The deck-wide part (the Google and Microsoft OAuth clients, the team Telegram bot, the Jev and Soniox team keys, MCP servers shared with everyone) sits above the admin's own connection.
- **Users** — Users · Invitations · Waitlist, as on the Users page.
- **Usage & limits** — today per person against the plan, spend by person and run kind, the plan limits in one table, recent activity.
- **Settings** — branding, sign-up and the waitlist, kiosk mode, Docker, and the profile defaults.

**Admins start there.** Signing in from a form (sign in, sign up, an accepted invitation) takes an admin to the console's Overview; an admin who never picked a layout starts there, and a reload keeps the layout they chose last. Other ways in: `/console`, a button in the top bar and one in the simple layout. The full layout's Admin, Users, Library and Connections pages are still there. Non-admins never see the console.

### Users, invitations and the waitlist

User management is its own **Users** page (admin only, in the sidebar and the phone navigation) with three tabs, which stay loaded so their counts are right and a half-written search or invitation survives a tab switch. The Admin page keeps Usage, Plans, Settings and Integrations.

- **Users** — search by name or email; filter by role, plan and status (kiosk, temporary password, active today, daily limit reached); sort by newest, oldest, name or today's use. **Add user** sits by the search box. Each user has **Set password**, the **Kiosk mode** mark, today's usage with **Reset today**, their free resets and their **Agents**.
- **Invitations** — **Invite by email**: email, name, role, kiosk and an optional note, sent from the admin's own connected Gmail or Outlook (or no email: copy the link). The invitation carries a one-time link to `<deck>/invite#<token>` (only the token's hash is stored; it expires after 7 days). The invitee chooses their own password there and is signed in — no password travels by email. A failed email keeps the invitation and hands the admin the link; accepting rings the inviter's bell and settles a pending waitlist application. The tab lists Pending / Expired / Accepted / All with counts, searchable by email, name, inviter and note; **Resend** makes a new link. Gmail needs a send-capable scope (and honours `FD_GMAIL_SEND`); Outlook needs `Mail.Send`.
- **Waitlist** — while self-registration is closed, "No account?" on the sign-in page leads to a waitlist form: name, email, password, and the name and email of whoever recommended them. Applicants join under the deck's instance name (**Admin → Settings → Waitlist** can set a separate **Waitlist name**). An applicant who signs in before approval is told it's pending. **Approve** creates the account with the password the applicant chose; with the "when I approve someone, email them" picker set, **Approve & email** also tells them their account is ready, with the deck's sign-in link (from `FD_PUBLIC_URL`, else the admin's browser address), sent from the admin's own Gmail or Outlook. A warning shows when that link would point at localhost, and a failed email never undoes the approval (a banner gives a copyable sign-in address instead). The tab lists Waiting / Approved / Rejected / All with counts. Settings can turn the waitlist off. The form is rate-limited per network and capped at 1,000 pending applications.

### Passwords

- **Set by an admin.** **Set password** on a user (new + confirm, **Sign them out everywhere** ticked by default). Add user and Set password mark the account **must change password** — a checkbox, on by default; untick it for a shared kiosk account. That user's first screen after signing in is "Choose your own password", and the Users page shows a "temporary password" badge until they do.
- **Changed by the user.** **Profile → Password** (also in the kiosk's Profile dialog): current password, new + confirm, and optionally **Sign out my other devices and browsers**. Attempts are rate-limited (10 per 15 minutes per user). Access tokens already issued stay valid until they expire (at most 15 minutes).

### Branding and the sign-in page

**Admin → Settings → Branding:**

- **Instance name** — replaces "Flight Deck" in the panel titles, the browser tab and the sign-in page; waitlist applicants join under it.
- **Description** — a short line for the sign-in page.
- **Logo** — uploaded to `/fd/admin/branding/logo` and served publicly at `/fd/branding/logo` (up to 512 KB; the type is checked from the bytes; SVG is served under a sandbox).
- **Powered by Captain** — on by default (`show_powered_by`): the sign-in footer reads "Powered by Captain · GitHub", linking to captain-claw.com and the GitHub repository. Off, the footer shows only the version and build date.

The sign-in, sign-up and waitlist pages share one design: a brand panel beside a card, tabs for sign-in and sign-up, and a clear "Sign-up is closed" panel when the waitlist is off and admins create the accounts.

### Daily plan limits and free resets

Plans are **free**, **premium** (formerly "pro"; users and stored limits move over on start) and **enterprise**. Besides the older limits (agents, storage, requests per minute, spawns per hour), each plan has **messages per day** and **AI tokens per day**, counted per UTC day. 0 means unlimited, and per-user overrides work as for the other limits.

| Plan | Messages / day | AI tokens / day |
|---|---|---|
| free | 30 | 300,000 |
| premium | 300 | 3,000,000 |
| enterprise | unlimited | unlimited |

- **Messages** are counted where Flight Deck relays a person's turn: the agent chat, a shared agent's member chat (billed to the member), the voice devices, WhatsApp, Messenger, the glasses (`/glasses/send`) and the deck's Telegram bot. Over the limit the turn is refused with a message saying when it resets. Commands that never reach a model (`/new`, `/help`, …) are free.
- **Tokens:** agents report every LLM call made on the deck's keys (calls on a person's own key cost the deck nothing and aren't counted): the input that missed the prompt cache, plus the output. **Cache reads are not counted** (cache writes are), so a long cached conversation doesn't swallow the allowance. A member's turn is billed to the member. A spent token allowance refuses new messages and skips scheduled tasks (one-shots retry after the reset).
- **Exempt** (still counted): admins, the standalone app, and decks without sign-in.
- **Where people see it:** the allowance line under the chat's lane tabs, the kiosk Info dialog's meters, and a tip in the guide. **Admin → Plans** edits the limits; the Users page shows today's use with **Reset today**.

**Free resets.** An admin gives users free resets from the Users page (one user, the selected users or everyone), up to a deck-wide cap (default 3; system setting `fd:usage-reset-cap`) so repeated grants don't pile up; a grant never takes resets away. A user spends one from the usage line (when the allowance is spent or running low) or the Info dialog to start the day's messages and tokens over. Grants, take-backs and spent resets go to the usage log. Routes: `POST /fd/quota/reset`, `POST /fd/admin/usage/resets`, `PUT /fd/admin/users/{id}/usage/resets`.

### Integrations: shared keys, Telegram and MCP

**Admin → Integrations** (the console's Connections page) holds what the deck shares:

- **Jev and Soniox team keys** — one key for everyone without their own; a user's own key in **Connections** wins. **Move server key here** imports `SONIOX_API_KEY`, which stays as the last fallback until then.
- **The team Telegram bot** — one bot token for the deck (see [Telegram: the Team Bot](#telegram-the-team-bot--new-in-081)). **Auto-approve** approves new requests at once (someone an admin turned down still waits; switching it on approves the requests waiting now). The Telegram inbox switch for the deck, and whether @BotFather allows Secretary Mode and group reading, are shown here too.
- **MCP servers** — which servers are shared with everyone on the deck; share one of your own (the confirmation names the account an OAuth server acts as) or stop sharing one; servers nobody owns are flagged. Adding and editing stays in **Connections → MCP Servers** (see [MCP servers](#mcp-servers-centrally-managed)).
- **Google and Microsoft** — one admin-managed OAuth client per deck; each user still connects their own account.

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

On Telegram, `/approve user telegram <token|@user|id> owner` (NEW in 0.8.1) marks the approved user as the agent's owner: their private chat joins the agent's main conversation (see [Telegram: the Team Bot](#telegram-the-team-bot--new-in-081)).

### Telegram: Per-User Sessions

Each Telegram user automatically gets a dedicated session and agent instance — except the owner's own private chat, which since 0.8.1 joins the agent's main conversation (`telegram.owner_in_main`; see [Telegram: the Team Bot](#telegram-the-team-bot--new-in-081)):

- **Isolated sessions** — users cannot see or switch to other users' sessions
- **Concurrent execution** — different Telegram users can interact simultaneously (no "Agent is busy" blocking between users)
- **Per-user locks** — requests from the same user are serialized; requests from different users run in parallel
- **`/new` starts a fresh session** (NEW in 0.8.0) — `/new [name]` (or `/session new`) moves the user to a new session and archives the old one to history; the new one stays theirs after a restart. It is refused while a reply is still running ("send /new once the reply is in").
- **Disabled commands** — listing and switching sessions (`/sessions`, `/session list`, `/session switch`, `/session load`) are not available on Telegram: each user has one session at a time
- **Available session commands** — `/clear`, `/history`, `/compact`, `/session info`, `/session rename` operate on the user's own session
- **Photo attachments** — images sent to the bot reach the agent as pictures; since 0.8.1 a model that doesn't see images gets the photo described first (by a vision model or a multimodal peer), as in the web chat (see [Native Vision](#native-vision--new-in-081))
- **Voice notes** — the user's own voice note is transcribed (Soniox, the owner's key through Flight Deck) and the transcript becomes the message; the bot shows it back ("🎙 Transcription: …"). Without a Soniox key, or when nothing was recognised, the bot says so and no turn runs. A *forwarded* voice note is someone else's words: it reaches the agent as an audio file instead.
- **Audio files** — reach the agent as attached files; when Soniox is connected the attachment hint names the `audio` tool to transcribe them.
- **Videos, round video messages and GIFs** — reach the agent as attached files and are analyzed before the turn (`video_vision`: frames + audio transcript), like a video uploaded in the web chat.
- **20 MB limit** — the Bot API only lets a bot download files up to 20 MB; a bigger one is answered with that reason (a caption sent with it still reaches the agent).
- **Generated images**: images that `image_gen`, `termux` or a browser screenshot made this turn are sent back to the user automatically.
- **Files the reply names**: a document or picture the agent wrote this turn is sent back when the reply names it by file name ("here's report.docx"). Only files in the user's session folders under `saved/` count, never `scripts/`, `tools/`, `skills/` or `downloads/`. A file the agent only read (a Drive download, an email attachment) is never sent.
- **Only deliverables**: a file is sent only if the turn made it, inside the agent's own `saved/` area. A `Path:` line in tool output, an email body or a web page can't send a file. The CLI chat bridges (Telegram, Slack, Discord) and WhatsApp follow the same rule.
- **Live narration** (NEW in 0.8.1): each short note the agent writes beside its tool calls ("Checking your calendar…") goes to the chat the moment it is written, one message each, with the answer last — as on WhatsApp. `STREAM_NARRATION=off` folds them into the final answer instead. Narration goes only to the Telegram chat, never over the agent's web socket.
- **Card numbers scrubbed** (NEW in 0.8.1): card numbers and CVVs are removed from typed text and from a voice note's transcript before they go further, and the echoed transcript shows the scrubbed text.

### Telegram: the Team Bot — NEW in 0.8.1

A Flight Deck can run **one Telegram bot for the whole team**, so nobody needs a bot of their own.

- **Setup.** An admin sets the bot token in **Admin → Integrations** (checked with Telegram). Flight Deck is the only process polling it.
- **Joining.** A user asks for it in **Connections → Telegram**, picking one of their own agents. An admin approves (or **Auto-approve** does). A one-time `t.me/<bot>?start=<code>` link then binds the user's Telegram account to that agent.
- **How it runs.** Agents reach Telegram through Flight Deck's relay (`/fd/telegram-relay`, which speaks the Bot API), identified by their `X-Agent-Auth`, and each is held to its own approved chats, files and button presses. The token never leaves Flight Deck; chats and sessions stay in the agent. An agent with no bot of its own runs its usual Telegram code against the relay (`telegram.shared_bot`, on by default); an agent with its own bot keeps it. Messages count against the user's [daily allowance](#daily-plan-limits-and-free-resets).
- **The owner's chat is the main conversation.** The owner's private chat joins the agent's main conversation (lane A) — the one the web chat, WhatsApp, the glasses, the voice devices and the app already share. A turn from it waits while another lane-A turn runs, takes in waiting automation results first, shows on every client "via Telegram", and frees the lane after. `/new` from it starts a new main conversation, and delegated results come back through it. The owner is whoever reaches the agent through the team bot, or, on an agent's own bot, the approved user marked owner (`/approve user telegram <token|@user|id> owner`). Other Telegram users and groups keep sessions of their own. `telegram.owner_in_main: false` gives the owner's chat its own session again.
- **What's new on Telegram.** With an [inbox](#telegram_inbox), the team bot also keeps a read-only copy of the user's own Telegram chats for their agents.

**Groups.** A bot can't follow a group invite link (`t.me/+…`): the Bot API has no way to join a chat, only a member can add a bot. So the way in is the bot's own add-to-group link, `t.me/<bot>?startgroup=agent`. Send `/group` (or paste an invite link) in a private chat and the bot answers with an **Add me to a group** button; **Connections → Telegram** also has "Add <agent> to a group" and lists the agent's groups with **Remove**.

- In a group the agent answers @mentions, replies to its messages and `/ask` from anyone. Each group has one session, and what was said since its last answer goes along as context.
- **Group turns use the web only.** They run like a shared-agent member's turn: `web_search` and `web_fetch`, and none of the owner's profile, packs, insights, intuitions or memory in the prompt.
- `/new` and `/leave` work for whoever added the bot, or the owner. Answers are capped per group per hour (`telegram.group_turns_per_hour`, default 30); `telegram.groups: false` turns groups off. Only messages addressed to the bot count against the plan of whoever added it.
- On an agent's own bot, only someone it already answers may bring it into a group; anyone else adding it gets one line and the bot leaves. On the team bot, a stranger adding it makes the bot leave. Disconnecting Telegram, or an admin rejecting or removing the link, takes the agent out of that person's groups.

### Supported Remote Commands

Remote users can use: `/help`, `/config`, `/history`, `/compact`, `/models`, `/sessions`, `/session info`, `/session select`, `/session rename`, `/skills`, `/skill`, `/skill search`, `/cron`, `/todo`, `/contacts`, `/scripts`, `/apis`, `/pipeline`, `/planning`, `/orchestrate`.

Local-only commands (not available remotely): `/exit`, `/approve user`, `/session run`, `/session procreate`, `/session protect`, `/session export`, `/session queue`, `/monitor`, `/cron add/list/history/pause/resume/remove`.

**Telegram-only restrictions:** `/sessions` is disabled, and so are the session switching subcommands (`list`, `switch`, `load`). `/new` works (since 0.8.0) and starts a fresh session for that Telegram user.

---

## Glasses Bridge (Meta Ray-Ban Display)

A mobile-web → agent → glasses-web pipeline that lets you talk to any Flight Deck **process** agent from your phone and have its reply rendered (and spoken) on Meta Ray-Ban Display smart glasses.

Three pages, all served by Flight Deck:

| Page | URL | For |
|---|---|---|
| Mobile bridge | `/glasses/mobile?c=<channel>` | Your phone — pick agent, type, send |
| Glasses view | `/glasses/view?c=<channel>` | The glasses — renders the conversation, speaks replies |
| Settings | `/glasses/settings?c=<channel>` | Pick TTS voice + language |

The mobile bridge is installable as a PWA on iOS and Android.

### Prerequisites

- Flight Deck running: `captain-claw-fd` (binds to `http://0.0.0.0:25080`).
- A **public HTTPS URL** pointing at Flight Deck — Meta Web Apps require HTTPS:
  - `cloudflared tunnel --url http://localhost:25080`
  - `ngrok http 25080`
- At least one captain-claw **process** agent spawned via Flight Deck → Agents.
- Optional (for TTS): a Soniox API key. Without it the glasses still render replies visually but won't speak.

```bash
export SONIOX_API_KEY=sk_…                # for voice, unless an admin set a Soniox team key
export SONIOX_TTS_VOICE=Adrian            # optional, default
export SONIOX_TTS_LANGUAGE=en             # optional, default
captain-claw-fd
```

### Quick start

1. **On your phone**, open `https://<your-tunnel>/glasses/mobile`. It generates a random channel id and bounces to `?c=abc12345`. Optionally install it: iOS Safari → Share → Add to Home Screen; Android Chrome → install prompt or three-dot menu → Install app.
2. Pick your agent from the dropdown.
3. **On the glasses**, in the Meta AI mobile app's Web Apps section, add a Web App at `https://<your-tunnel>/glasses/view?c=abc12345` (same channel id) and launch it from the Meta Ray-Ban Display app launcher.
4. On the glasses, tap **⚙** to open the settings page and pick voice + language with the button grids. Tap **🔊** to turn voice on (it starts off — your tap is the user gesture browsers need for autoplay).
5. Type on the phone → Send. The message appears on the glasses with a pulsing amber "thinking" status; the reply renders in markdown when ready and (if voice is on) speaks through the phone's audio output (typically the glasses' open-ear speakers when paired as a Bluetooth audio sink).

### Mobile bridge controls

- **Channel chip** — directly editable; type a different channel id and press Enter / tap-out to apply, or tap ⟳ for a fresh random one. The URL stays in sync via `history.replaceState`.
- **Target agent** — lists running Flight Deck **process** agents (`/fd/processes`). Switching agents tears down the old upstream WS and rebinds.
- **Composer** — textarea + 📷 photo attach + Send. Photo accepts JPEG/PNG/WebP/GIF/BMP (whitelisted by the agent's `/api/image/upload`). On iPhone, Safari typically returns JPEG so HEIC isn't a concern in practice.
- **Activity log** — collapsible card; user/agent/status/error coloured rows.
- **Status pill** — green `live` (animated) / red `offline`.

### Glasses HUD controls

- **🔊 / 🔇** — toggle voice. Starts muted; tapping unmutes and counts as the user gesture autoplay needs.
- **📡 / 📦** — stream vs one-shot TTS. Streaming is the default (~150–300 ms first-audio latency); one-shot waits for the full MP3 before playing.
- **⚙** — open settings page (voice + language).
- **Engine label** — `soniox` when live; `off` when muted or after a server config issue; `off:NNN` permanent on 503/401/403 from the bridge; `err:NNN` retryable on transient failures.

### Bridge endpoints

| Endpoint | What it does |
|---|---|
| `GET /glasses` | Redirects to `/glasses/mobile?c=<random>` (zero-config entry) |
| `GET /glasses/mobile?c=X` | Mobile bridge UI (PWA-installable) |
| `GET /glasses/view?c=X` | Glasses HUD page (PNG icons + launcher manifest) |
| `GET /glasses/settings?c=X` | Tap-target voice/language picker |
| `GET /glasses/agents` | JSON list of running process agents |
| `POST /glasses/send` | `{channel, host, port, text, image_path?}` — route a message |
| `POST /glasses/upload-image` | Multipart proxy to agent's `/api/image/upload` |
| `POST /glasses/tts` | One-shot Soniox TTS (audio bytes for `<audio>`) |
| `WS /glasses/tts-stream` | Streaming Soniox TTS (PCM s16le @ 24 kHz binary frames) |
| `WS /glasses/ws?c=X&role=mobile\|glasses` | Channel pub/sub |
| `GET /glasses/manifest.webmanifest` | PWA manifest for the mobile bridge |
| `GET /glasses/view-manifest.webmanifest` | Glasses launcher manifest (PNG icons) |
| `GET /glasses/sw.js` | Minimal service worker (no caching) |

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SONIOX_API_KEY` | _(unset)_ | Fallback only: the glasses speak with the deck's Soniox team key (Admin → Integrations), else this. Without either they fall back to silent. |
| `SONIOX_TTS_MODEL` | `tts-rt-v1` | Soniox TTS model id |
| `SONIOX_TTS_VOICE` | `Adrian` | Default voice; user can override per-session via the settings page |
| `SONIOX_TTS_LANGUAGE` | `en` | Default language; user can override per-session via the settings page |
| `SONIOX_TTS_FORMAT` | `mp3` | One-shot audio format (`mp3` / `wav` / `aac` / `opus` / `flac`). Streaming always uses PCM s16le @ 24 kHz. |
| `FD_GLASSES_BRIDGE_TOKEN` | _(unset)_ | Optional shared-secret gate. When set, the bridge requires `?t=<token>` on every request. |

### Notes & limitations

- **No microphone access** on the glasses Web Apps SDK as of this release — input has to come from the phone (text + photo). The glasses are output-only.
- **Process agents only** — the mobile bridge lists Flight Deck process agents (`/fd/processes`), not Docker containers.
- **Channel sharing = pairing**. Anyone with the URL is in the channel. For private deployments, set `FD_GLASSES_BRIDGE_TOKEN` and append `?t=<token>` to the URLs.
- **Hidden system context** is injected to the agent on the first message per binding so the model knows the reply will render on a tiny HUD (concise, lead-with-the-answer, no preamble). Never broadcast to the channel — the user only ever sees their plain text in the mobile log.
- **Voice is off by default** — explicit tap-to-enable, which also serves as the user gesture for browser autoplay.

### Flight Deck HUD on the glasses (`/hud`) — NEW in 0.8.0

Besides the bridge pages above, Flight Deck now ships a full frontend for the Meta Ray-Ban Display at **`/hud/`**: chat with your running agents (the same conversation as the dashboard, dictation or handwriting through Meta's composer, replies read aloud), read the Markdown files your agents wrote, and browse datastore tables read-only — all driven by swipes and pinches on a 600 × 600 display. It is a separate, lean page (about 150 KB on the wire for the first load, cached by a service worker afterwards). Pair it by opening `/hud/pair` on your phone or computer and typing the code the glasses show. Requirements, pairing, screens and limits: [docs/glasses-hud.md](docs/glasses-hud.md).

---

## M5Stack Voice Devices (StickS3, StopWatch, PaperColor)

Push-to-talk clients for your agents on three M5Stack devices, built from one firmware tree in `firmware/device/` (one PlatformIO env per device). Press the talk button and talk; press it again, or just stop talking for 2.5 s, to send. Flight Deck transcribes it (Soniox), sends it to one of your running agents, and the device shows the reply, reads it aloud, and shows any picture the agent made. While the agent works, it shows what it heard, the agent's live activity lines ("Searching: …", "Reading: …") and a "12s · 3 steps" timer. Results the agent produces later (a cron job, a scheduled task, a nudge) reach the device too.

This section is a summary. Every screen, setting and button, the build and flash steps, and the protocol are in [firmware/device/README.md](firmware/device/README.md).

| | StickS3 | StopWatch | PaperColor |
|---|---|---|---|
| Screen | 240×135 LCD | 466×466 round AMOLED, touch | 4" Spectra 6 colour e-paper, 400×600, 15–30 s per redraw |
| Talk button | front (A) | yellow (A) | top (C) on the A build, A on the B build |
| Extras | motion sensor | vibration, clock face, raise to wake | two RGB status lights, temperature and humidity sensor, clock, microSD slot |
| PlatformIO env | `sticks3` | `stopwatch` | `papercolor` (A build) / `papercolor_b` (B build) |
| Firmware (0.8.4) | 0.19.10 | 0.19.10 | A: 0.19.10 · B: 1.0.22 |

The same firmware also runs on the **Waveshare ESP32-S3-ePaper-1.54** (0.8.4), V2 boards only: `epaper154` (SKU 32298) and `epaper154_touch` (SKU 34212), a 200×200 black-and-white panel. BOOT talks, and only a press ends a recording; PWR cancels a recording. Flash it once over USB; updates come over the air after that. Details in [its section of the firmware README](firmware/device/README.md#waveshare-esp32-s3-epaper-154-epaper154-epaper154_touch).

### Deck side and setup

- **Server:** `WS /device/ws` on Flight Deck (`captain_claw/flight_deck/device_bridge.py`). The device streams 16 kHz PCM while it records; replies come back as text, Soniox TTS audio (24 kHz PCM) and pictures fitted to the screen (the StopWatch's fill the circle, sent in parts; the PaperColor's keep their own shape).
- **Auth:** a personal access token (Flight Deck → Connections → *Agent access for Claude (MCP)* → Generate token) names the owner; a deck with auth off needs none.
- **Deck requirements:** a Soniox key (speech in, voice out) — the device owner's own in **Connections → Soniox**, else the admin's team key, else `SONIOX_API_KEY`. A deck reached through a domain needs that domain in `FD_ALLOWED_HOSTS` or `FD_PUBLIC_URL` (the [origin guard](#flight-deck-security-hardening--new-in-080) refuses unknown hosts). The male and female voices are Soniox `Adrian` and `Emma`; `DEVICE_VOICE_MALE` / `DEVICE_VOICE_FEMALE` pick others. Device messages count against the owner's [daily allowance](#daily-plan-limits-and-free-resets).
- **Setup:** on first boot (or switching on with the talk button held) the device opens a `Captain-StickS3-xxxx` / `Captain-StopWatch-xxxx` / `Captain-PaperColor-xxxx` hotspot and shows its QR code. At `http://192.168.4.1` enter up to five Wi-Fi networks, the deck address (`ws://<host>:25080` or `wss://<deck domain>`), the token, and optionally an agent slug, reply language, volume, and screen-off and power-off timeouts. Later, Menu → *Wi-Fi & settings* reopens it (PaperColor: hold B on the A build, hold C on the B build), and holding the side button on the setup screen leaves it without saving. Once the device is online, Menu → *Settings on phone* (the PaperColor's Options) shows the same page on the device's own address behind a one-time QR code, so changing a setting doesn't need the hotspot.

### Buttons

**StickS3**

| | Front (A) | Side (B) |
|---|---|---|
| click | talk, click again to send (or 2.5 s of quiet) · while it speaks: stop | next page · back (a picture: its text) |
| hold | | menu (hold again closes) |
| double-click | hear the last answer again | (on an answer) hear it again |

A press that wakes the stick only wakes it; the next press talks.

**StopWatch**

- **Yellow button:** press, talk, press again — or 2.5 s of quiet sends it. A press while it speaks stops the speech, a double press plays the last answer again, and a press that wakes the watch (from sleep or a dark screen) only wakes it. **Hold it (1 s) for note-taking mode** (see below).
- **The other button** works like the StickS3's side button: next page or back, hold for the menu, double-click on an answer to hear it again.
- **Touch.** Idle, the screen is a clock face (the date, the time, the agent's character, the battery as an arc and its time left), the middle of three pages: swipe right for the background picture alone, left for the info page (voice notes waiting and room for more, the Wi-Fi network and its signal, the Flight Deck address, the agent), up or down for the last answer. Swipes page an answer, a double tap goes home, and the menu, the agent and character pickers and the status pages take swipes and taps. *Tap to talk* (Menu, off by default) adds an on-screen record button. While the agent works, the button under the timer is a **stop** (the agent stops after its current step); while an answer is spoken it is **pause / play**; afterwards a **replay** button sits under the text. A tap on a picture shows all of it, another fills the circle again.
- **Raise to wake.** A wrist raise lights a dark or dimmed screen; lowering the wrist on an idle screen puts it out. The watch still sleeps on its *Power off* timer (up to 3 minutes); only the buttons wake it from sleep.

**PaperColor, A build** (`papercolor`): voice-first; the paper redraws about 30 s after you put it down.

| | C (top) | A | B |
|---|---|---|---|
| click | talk, click again to send · while it speaks: stop · on Options: update | home: status · answer: next page · Options: done | answer: home · home: the answer · back |
| hold | | Options (a QR code for the phone settings page) | Wi-Fi & settings (the setup hotspot; hold again to leave) |
| double-click | hear the last answer again | (on an answer) hear it again | |

**PaperColor, B build** (`papercolor_b`, versions 1.x, published as the same `papercolor` kind): the screen treated as a dashboard; the lights and the voice carry the turn. The dashboard — clock, date, room temperature and humidity, battery with its time left, voice notes waiting and room left, the Wi-Fi / deck / agent links — is redrawn on every five-minute mark, awake or asleep (a timer wake draws it without Wi-Fi). A new answer goes on the paper only while the answer page is showing, at the largest of four text sizes that fits; from the dashboard, B shows it. Once Wi-Fi, the deck or the agent has been down for 30 s the links show in red.

| | A | C (top) | B |
|---|---|---|---|
| click | talk, click again to send · while it speaks: stop · on Options: back · in the hotspot: leave | Options (QR code) · on Options: update · no Wi-Fi: the setup hotspot | the last answer · next page · after the last: the dashboard |
| hold | note-taking mode on / off | Wi-Fi & settings (the setup hotspot) | the dashboard now |
| double-click | hear the last answer again | | |

On the B build, voice notes kept offline go to the agent by themselves (no question), *Auto update* defaults to on, and moving the paper never counts as use or redraws it. Either build powers off after 1, 3, 10 or 30 idle minutes, 1 or 2 hours, or never (the phone settings page), keeping what the paper shows.

**PaperColor status lights:** yellow at start-up, then the left one shows Wi-Fi and the right one the agent (green up, red down); while it records, the left blinks red (brighter with your voice) and the right one shows how long — green for the first 20 s, yellow for the next 20 s, red after; a blink per agent step (light blue a tool, dark blue narration, purple the model); green with the voice. Idle, both blink every 10 s, coloured by how near power-off is. Offline, the left one blinks blue without Wi-Fi, the right one when Wi-Fi is up but the deck or the agent isn't. While an update downloads, purple left, then right.

### Screens and settings

- **The menu** (hold the side button; on the StopWatch it has two levels, with the rest under *Settings*): *Replay answer*, *Status*, *Send voice notes*, *Agent »* (which of your agents the device talks to), *Volume* (25–100 %) and speech *Boost* (+6 / +10 dB with a soft limiter), *Voice* (male / female), *Display* (text, or icons: an animated character), *Character »*, *Text size* (small / medium / large), screen-off and power-off timeouts, *Talk sounds*, *Beeps around speech*, *Tap to talk* (StopWatch), *Software update*, *Auto update*, *Settings on phone* and *Wi-Fi & settings*. The PaperColor has no menu: its Options code opens every setting on the phone.
- **Characters.** In icon mode the screen shows one of 18 animated characters — Captain, Bot, Bird, Car, Matrix, Orc, Elf, Dwarf, C64, Amiga, Cat, Dog, Dinosaur, Alien, Caveman, HAL 9000, KITT or Pac-Man — with idle, listening, thinking, speaking (driven by the voice) and agent-down poses. *Character »* previews each one.
- **Status.** Wi-Fi network and signal, IP, the deck and its round trip, the agent link, battery (level, volts, charging, time left), uptime, answers, the room's climate (PaperColor), firmware and any update on offer, voice and text size, free memory and voice notes waiting. The StickS3 shows it in pages at the chosen text size; the StopWatch as three pages of gauges round its ring. The StickS3's top bar always shows Wi-Fi and agent colour blocks and the battery percentage.
- **The StopWatch's look.** A ring round the edge shows state — red while it listens, amber while the agent works (a tick per tool step), a short cyan line under the text following the voice. Screens slide or fade between each other, the watch answers touch with short vibrations, and pictures fill the circle.

### Talking

- **Press to talk.** Every device toggles: press, talk, press again. A recording with no button held ends by itself after **2.5 s of quiet**, measured against the recording's own quietest and loudest moments so the mic's floor doesn't matter; a click, a tap on the case or room hum doesn't count as a voice. With nothing said, the recording is dropped.
- **Talk sounds and chimes.** A short high blip when you press (recording starts a quarter second later, so the blip isn't recorded) and a lower one when it is sent or kept offline (*Talk sounds*, on by default). A rising two-tone chime before the answer is spoken and the same tones falling once it has played to its end (*Beeps around speech*); a stopped answer or a replay gets neither. The PaperColor also beeps on its key presses.
- **Sound by voice.** A recording that is only "volume up" / "volume down" (or "louder", "quieter", "pojačaj", "tiše" …) or "beep on" / "beep off", in English or Croatian, with a "please" either side, never reaches the agent. Flight Deck spots it, the device moves the volume a quarter at a time (25–100 %) or turns every beep on or off, keeps the setting, and says the result back at the new volume. Anything longer ("turn up the volume on the kitchen speaker") goes to the agent.
- **"New topic."** A recording that opens with a new-topic cue ("New topic, how warm is it?") starts a new conversation and goes straight to the answer. Said alone, the device answers "New topic — go ahead."
- **Readings with every turn.** Each recording carries the device's readings: battery (%, volts, charging or on USB), the Wi-Fi network and signal, how it's held and whether it's moving (StickS3, StopWatch), the room's temperature and humidity (PaperColor), and the local time. Flight Deck turns them into one `[Context: …]` line at the top of the turn — context for the model, never stored as the user's words — so "how's my battery?" or "is it warm in here?" just work. The agent doesn't recite them unless asked, or unless the battery is nearly empty.
- **Source labels.** Device turns are recorded with source `device:sticks3`, `device:stopwatch` or `device:papercolor` and a label like "M5Stack StickS3 (stick-a1b2c3)": the kiosk shows them "via M5Stack …", the model sees "[via …]", and the agent's [reasoning effort](#reasoning-effort--new-in-081) can be set per device.

### Answers

- **Streamed and spoken by the sentence.** Flight Deck follows the model's text as it is written and, each time a sentence completes, shows the answer so far and speaks that sentence, so the first words are heard while the rest is still being written. The full answer then replaces the text in place and only what wasn't spoken yet is read. Only the answer streams to the device — the agent's own calls (titles, rephrasing) never do — and text the model writes before a tool call isn't treated as the answer.
- **The text follows the voice.** On the StopWatch the sentence being spoken is lit and the rest dimmed (in character mode it shows as a subtitle under the face); on the StopWatch and the StickS3 the page turns by itself when the voice reaches it, until you page by hand. Each sound tells the device which characters it covers, so the place stays exact through a long answer.
- **Replay and pause.** The device keeps the whole answer in memory (its speech, then any sound the agent made, up to about 66 s) and replays it at once, offline. If part of it never arrived, the device asks Flight Deck to send the answer again. On the StopWatch, pause and play sit under a spoken answer.
- **A dropped link.** On weak Wi-Fi the device keeps the working screen ("Connection lost — reconnecting…") and retries for up to 2 minutes. Flight Deck keeps each device's open turn and its last answer for 10 minutes, so after a reconnect the answer arrives (or "Thinking…" shows again while the agent is still at it). If the link stays away, or the deck restarted meanwhile, the device says "Answer lost — ask again".
- **Battery time.** The StopWatch (on the clock face and its Power page), the StickS3 (its Battery status row) and the PaperColor B dashboard say "≈ 2 h 10 min left" or "full in ≈ 45 min", learned from the battery level over time (through sleeps too) once there are five minutes and two per cent to go on.

### Offline voice notes and note-taking mode

- **Offline notes.** Out of Wi-Fi, or with Flight Deck down, talking records a voice note into the device's flash: 8 kHz 4-bit ADPCM, 4 KB a second, about 13–14 minutes on the StopWatch and PaperColor and about 5½ on the StickS3. With a microSD card in, the PaperColor keeps notes on the card instead (FAT32 or FAT16, in `captain-notes/`; only files named the way the device names notes are ever read, sent or removed). When none of its saved networks is in reach at switch-on, a device becomes a note taker within a couple of seconds instead of opening the setup hotspot (with no network saved at all, setup still opens), and keeps looking for Wi-Fi every 15 s.
- **Asked before sending.** Once the deck is back, the device asks first ("There are saved voice notes. Would you like to send them to the agent?"): the talk button sends them, the side button keeps them for later, and Menu → *Send voice notes* sends them any time (the PaperColor B sends them without asking). Flight Deck transcribes them and gives the agent all of them as **one turn**, oldest first, each with when it was recorded. A note is deleted from the device only after the deck has it; one Soniox couldn't transcribe stays for a later try.
- **Note-taking mode** (StopWatch: hold the yellow button 1 s; PaperColor B: hold A 1 s; Waveshare ePaper 1.54: hold BOOT 1 s). Every recording is kept as a voice note, online too, with no quiet stop and no length limit beyond the room left: notes are written to flash, or the card, as they are recorded. Rising tones and a notice turn it on (the watch's clock face shows a red arc and "Note mode · hold yellow to send"; the Waveshare's home screen shows a black tile with a notepad, "Boot to send", how many notes there are and the recording time left; the PaperColor's dashboard turns its top block red and shows the same tile with "Hold A to send"); holding again turns it off with falling tones and sends all the notes to the agent as one turn, without asking. The mode is kept across sleep.

### Pictures and the StopWatch background

Pictures the agent makes are shown on the device; the PaperColor dithers them into its six inks. On the StopWatch, send the agent a picture and ask for it as the watch background: its [`watch_background`](#watch_background--new-in-081) tool hands it to Flight Deck, which crops it to the round screen and keeps it per owner. Every screen is then drawn on it, frosted and darkened behind text, and the clock face's swipe-right page shows it sharp. Ask the agent to clear it to go back to black.

### Updates over the air

Devices update over Wi-Fi from your own Flight Deck. An admin builds, signs and publishes a release with `tools/publish.py`, from `firmware/device/`:

```bash
uv run --no-project --with cryptography python tools/publish.py stopwatch \
    --deck https://<deck> --notes "what changed"
```

It builds the env, signs the image (ECDSA P-256; the private key stays in `~/.captain-claw/firmware-signing/` — back it up: without it no device accepts a new build) and uploads it with an admin token (`--token`, `FD_TOKEN`, or the `deck-token` file in that folder). It refuses a build whose `FW_VERSION` isn't the version it would publish; `papercolor_b` publishes as the `papercolor` kind. Flight Deck checks the image and keeps the newest three builds per kind under `<fd_home>/device_firmware/<hw>/` (`POST` / `GET /fd/device/firmware`, admin), and offers the newest to devices of that kind on connect, to connected ones at once. On the device: Menu → *Software update*, or *Auto update* when idle on USB power. An update refuses mid-conversation and under 30 % battery off USB, checks the signature before switching, and a new version that can't reach Flight Deck within 90 s rolls back to the previous one.

The StickS3 has no Bluetooth Classic, so audio plays on its own speaker, not on Bluetooth headphones.

---

## WhatsApp Bridge

Flight Deck's WhatsApp bridge connects a WhatsApp Business number (Meta Cloud API) to your agents. Setup: [docs/whatsapp-bridge-setup.md](docs/whatsapp-bridge-setup.md). A WhatsApp turn is part of the agent's main conversation, so it also shows in the web timeline "via WhatsApp" (see [The chat timeline](#the-chat-timeline-topics-and-other-channels--new-in-081)). Agents send files into a chat with [`whatsapp_send_file`](#whatsapp_send_file).

### WhatsApp: automated messages arrive once

An autonomy nudge, a Flight Deck scheduled job or an agent cron result used to reach WhatsApp twice. One copy was plain text: the result shown in the agent's main chat and relayed by the bridge. The other was formatted: Flight Deck's own delivery.

Now the copy in the main chat is marked `fd_delivers` whenever Flight Deck delivers the result itself, and the WhatsApp bridge shows it there but doesn't relay it. Only Flight Deck's formatted copy arrives, however late the main-chat copy appears. The mark is set in these cases:
- **every scheduled job;**
- **a nudge** when Flight Deck will push it to WhatsApp (a bound, allowlisted number), or when `nudge_to_whatsapp` is off;
- **an agent cron result** that Flight Deck routed back to where the session came from.

If Flight Deck stops waiting for a slow turn, the result isn't marked, so it still arrives once through the bridge. A nudge with no number to push to is also left unmarked. Automated results the bridge still relays (plans, cron results Flight Deck didn't route, autonomous actions that aren't nudges) respect `/mute`, the same as pushes.

### WhatsApp: `/new`, reactions and the garbled-reply hint — NEW in 0.8.0

- **`/new` answers.** The agent always started the new session, but its confirmation was dropped and the new session never got the WhatsApp reply rules. Command results now come back as a plain-text notice, and the rules are re-sent whenever the agent's session changes.
- **Emoji reactions.** A text or voice message that reaches the agent gets a quick emoji reaction (or none) picked by the target agent's own model in a background side call; it never delays the forward. On by default; `WHATSAPP_REACTIONS=off` turns it off, `WHATSAPP_REACTION_TIMEOUT` bounds the call. See [docs/whatsapp-bridge-setup.md](docs/whatsapp-bridge-setup.md).
- **Garbled replies.** When a reply comes out as unusable output and a Jev key is connected, a notice suggests `/new` (see [jev](#jev)).

### WhatsApp: files you send

Every file you send over WhatsApp reaches the agent: documents of any type (xlsx, docx, pdf, zip, json, …), photos, stickers, audio files and videos. Flight Deck downloads each file from WhatsApp and uploads it to the bound agent with `POST /api/file/upload?extract=0`, which keeps a zip as a zip. The agent gets a short note for each file: its name, type, size, caption and transcript. These notes are shown next to the attachments, never as your own words.

- **A file without a word** isn't a turn of its own. The bridge answers "📎 Got Report.xlsx — what should I do with it?" and your next message carries the file. Several files at once (an album) get one answer. A sticker on its own gets no answer; it goes with your next message as a plain file if that message comes within 10 minutes. Files wait for that message for up to `WHATSAPP_PENDING_FILES_MINUTES` (default 360); a slash command like `/new` doesn't take them.
- **A caption** is the message, and the file goes with it. An album with a caption on one photo stays one turn: the bridge waits for `WHATSAPP_MEDIA_BURST_SECONDS` (default 2) of quiet first.
- **A file right after a text** ("check this invoice", then invoice.pdf a moment later) is sent as belonging to that text, if it arrives within a minute and that text had no file.
- **Replying to a file's bubble** ("and the price in this one?") carries that file again.
- **Photos** get the agent's automatic vision description (a photo that waited more than 10 minutes goes as a plain file instead). HEIC and TIFF pictures sent as files are converted to JPEG when the deck can (HEIC needs `pillow-heif`), and stickers become a still PNG. A video, including one sent as a file, is analysed in a turn of its own, as before.
- **Voice notes** are transcribed and the transcript is your message; the recording is saved and named in a note. When transcription fails you're asked to type it or send it again, and the recording goes with your next message. An audio file, or a voice note you forward, arrives as a file with its transcript and is never treated as your own words.
- **Faces.** "who is this?" or "remember this is Ana" right after a photo runs face recognition on it; any other message sends the photo to the agent. A face caption ("who is this?") runs face recognition too, and the photo goes to the agent instead when it has no face or the deck has no face support.
- **A busy agent.** A message the agent refuses because it is still answering is no longer lost. You get "⏳ The agent is busy with another task — I'll pass this on as soon as it's free." and the bridge sends it again as soon as the agent is ready, or on a backoff of up to a minute, for about half an hour. If the agent restarts meanwhile, even on a new port, the message waits for it. The agent gets your messages one at a time, in the order you sent them.
  - **How it knows which message:** each frame carries a `client_msg_id`. The agent echoes it on the `thinking` status that starts the turn and on a busy refusal (`retryable: true`).
  - **Commands:** an agent slash command such as `/stop` is never held back. `/code`, `/publish` and `/orchestrate` run as messages and keep their place in line.
  - **A dropped link:** if the link to the agent drops before it confirms a message, you're told ("⚠️ … couldn't confirm it got …") and asked to send it again if no answer comes. The bridge doesn't re-send it, because re-sending could run it twice.
  - **Older agents:** an agent that doesn't echo ids yet makes a queued message wait up to 10 s for the one before it, and its busy text still reaches you.
- **Failures are reported.** A file over `WHATSAPP_MAX_INBOUND_MB` (default 100), a failed download or a refused upload gets a reply, and the agent is told you tried to send it. View-once media and polls can't reach the bridge, and it says so. A message Meta delivers twice is handled once.

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

## MCP Server

Captain Claw can run as a Model Context Protocol (MCP) server over stdio, allowing external MCP clients — such as Claude Desktop — to interact with the agent. Clients can browse sessions, read conversation history, send messages to the agent, and check agent status.

> **Fleet access over HTTP (NEW in 0.8.0).** Flight Deck also hosts an MCP server at **`/fd/mcp-server`** so Claude Code, Claude Desktop or claude.ai can drive **your whole fleet**: `list_agents`, `send_task` (async, returns a `task_id`), `get_result` (live progress + final answer) and `cancel_task`, scoped to the signed-in user's own agents. claude.ai / Claude Desktop add it as a custom connector (OAuth 2.1 with dynamic client registration + PKCE — sign in with your Flight Deck account); Claude Code uses a personal access token from **Connections → Agent access for Claude (MCP)**:
>
> ```bash
> claude mcp add --transport http captain-fleet https://YOUR-FD-HOST/fd/mcp-server --header "Authorization: Bearer cc_pat_…"
> ```
>
> Needs HTTPS, `FD_PUBLIC_URL` and a stable `FD_JWT_SECRET`. Details in [docs/team-deployment.md](docs/team-deployment.md#connect-claude-to-your-agents-inbound-mcp).

### Launch

```bash
captain-claw-mcp          # stdio mode (for MCP client config)
```

Or run directly with Python:

```bash
python -m captain_claw.mcp_serve
```

### Claude Desktop Integration

Add to your `claude_desktop_config.json` (typically at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "captain-claw": {
      "command": "captain-claw-mcp",
      "args": []
    }
  }
}
```

If `captain-claw-mcp` is not on your PATH (e.g. installed in a virtualenv), use the full path:

```json
{
  "mcpServers": {
    "captain-claw": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "captain_claw.mcp_serve"]
    }
  }
}
```

### Available Tools

The MCP server exposes the following tools to clients:

| Tool | Description |
|---|---|
| `sessions_list` | List recent sessions with IDs, names, timestamps, and message counts |
| `session_get` | Get session details by ID or name (message count, role breakdown, metadata) |
| `messages_read` | Read conversation messages with pagination (limit/offset) |
| `message_send` | Send a message to the agent and get a response — the agent uses its full tool suite |
| `session_create` | Create a new named session |
| `agent_status` | Get model, provider, token usage, active session, and enabled tools |

### How It Works

- The server uses **stdio transport** with newline-delimited JSON-RPC 2.0, following the MCP specification.
- The agent is **lazily initialized** on first `message_send` call, keeping the handshake fast.
- All stderr output is redirected to a log file so it doesn't corrupt the JSON-RPC stream.
- `message_send` runs the full Captain Claw agent loop — the agent can use all its tools (shell, web search, file ops, browser, etc.) to answer.
- You can target a specific session with `message_send` to continue an existing conversation.

### Configuration

```yaml
mcp_serve:
  enabled: true
  log_path: ~/.captain-claw/mcp_server.log
```

### Log File

Server logs are written to `~/.captain-claw/mcp_server.log` (configurable via `mcp_serve.log_path`). Check this file for debugging connection issues.

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

**Under Flight Deck (since 0.8.0) Google is per user.** The OAuth client (ID, secret, project, scopes) is set once by an admin in **Connections → Google** (`POST /fd/google/config` is admin-only); each Flight Deck user connects **their own** Google account, and an agent only ever gets its owner's token. Rotating the client or changing scopes disconnects everyone. After upgrading, each user reconnects once (the primary owner keeps working on the legacy deck-wide token until they do). Drive mounts in the VFS panel likewise use each user's own Drive.

### Vertex AI

If `project_id` is set, Google OAuth also enables Gemini models via Vertex AI. The `location` field specifies the Vertex AI region (default: `us-central1`).

### Gmail sending (opt-in)

The `google_mail` tool reads mail, creates drafts and — only when sending is turned on — sends. `create_draft` stays the default for anything email-writing; agents are told to use `send` / `send_draft` only when you explicitly asked them to send, never because an email, web page or file says so. Replies pass `reply_to_message_id` so they thread under the original.

| Action | What it does |
|---|---|
| `send` | Send now. Same fields as `create_draft` (`to`, `cc`, `bcc`, `subject`, `body`, `html_body`, `reply_to_message_id`); a reply defaults `to` (Reply-To, else From — or the original's To when it is your own sent email or draft) and the `Re:` subject. |
| `send_draft` | Send an existing draft by `draft_id` (the "Draft ID" `create_draft` prints) — its current server copy, so your edits in Gmail are kept. |
| `update_draft` | Revise an existing draft in place by `draft_id` (Gmail `drafts.update`) instead of making a second one. Pass the full new `body`; `to` / `cc` / `bcc` / `subject` you leave unset keep the draft's own, and so does a reply draft's threading. |
| `list_drafts` | List drafts with their Draft IDs, recipients (To / Cc) and dates (`query`, `max_results`). |

No new scope is needed: `gmail.compose` (the drafts scope) already lets Google accept a send, so the scope is not the gate — a policy is.

**No repeat emails.** Agents are told to check `list_drafts` (`to:<recipient>`) and `search` (`in:sent to:<recipient> newer_than:14d`) before drafting or sending, and to tell you about an email that already exists instead of writing it again. The tool also checks by itself: `create_draft` and a composed `send` refuse ("Not created — repeat of Draft ID … / the email sent on …") when the same email — a recipient in common and the same subject, ignoring `Re:` / `Fwd:` / `AW:` / `SV:` / `Odg:` prefixes, case and small differences — is already in Drafts (any age) or in Sent within the last 14 days. For a reply, a draft in the thread or a reply already sent after the email being answered — to the same recipient — counts. Different recipients are never a repeat. The agent passes `allow_repeat: true` only when you explicitly ask for another copy. The check needs no extra scope; with `gmail.compose` but no `gmail.readonly` it still checks Drafts. Standalone, set the Sent window with `tools.google_mail.repeat_check_days` (default 14, `0` turns the check off); under Flight Deck, sends use the deck's `FD_GMAIL_REPEAT_CHECK_DAYS` (same default and meaning). An autonomous `mail.draft` that is refused this way is logged as a skipped success, not a failure.

**Under Flight Deck** the agent never sends itself. It asks `POST /fd/google/gmail/send`, and Flight Deck checks the agent owner's own policy, sends with the owner's Google token, records the send and puts an "… sent an email" notification in the owner's bell. Each user sets their policy in **Connections → Google → Email sending** (`GET`/`PUT /fd/google/gmail-send`):

- **On / off** — off by default; nothing changes until you turn it on. On means automatic (no per-email confirmation).
- **Allowed recipients** — empty = anyone. Otherwise exact addresses (`alice@example.com`) and domains (`@example.com` or `example.com`); a domain matches that domain only, not its subdomains. Up to 200 entries.
- **Daily limit** — emails per rolling 24 hours, 1–500 (default 50).

At most 20 recipients per email (To + Cc + Bcc), and addresses must be plain ASCII (an internationalized domain in its `xn--` form). The same email (recipients, subject, body — and, for a reply, the email it answers) sent again within 10 minutes is refused as a duplicate. If Gmail does not confirm a send (a 5xx, or no answer after the request went out), the agent is told it may or may not have been sent; the attempt is still recorded (status `unknown`), counts toward the daily limit, blocks an identical retry for 10 minutes and rings the bell as "… may have sent an email". Your recent sends are listed under the same panel (`GET /fd/google/gmail-sends`, each with `status` `sent` or `unknown`). A deck admin can switch sending off for everyone with `FD_GMAIL_SEND=off` on the Flight Deck backend.

**Standalone** (an agent with its own Google connection) sends directly, only when enabled in config:

```yaml
tools:
  google_mail:
    allow_send: true            # default false; env: CLAW_TOOLS__GOOGLE_MAIL__ALLOW_SEND=true
    allowed_recipients:         # optional; empty = anyone
      - "@example.com"
      - "boss@corp.example"
    repeat_check_days: 14       # repeat check's Sent window; 0 = off (also applies to create_draft)
```

`allow_send` and `allowed_recipients` are ignored under Flight Deck, where the user's policy decides. `google_mail` is the only tool that sends Gmail (the old `gws` CLI wrapper, which could reach Gmail with the same token, is retired). The gate protects against model mistakes and prompt injection; it is not a security boundary against an agent that has a shell tool, which can reach the Google token itself.

**Attaching files.** `create_draft`, `update_draft` and a composed `send` take `attachments` — local paths as the file tools print them (`saved/…`, `vfs:<project>/<path>`); `update_draft` also takes `remove_attachments`. `microsoft_mail` drafts take the same parameters.

### Agentic signature — NEW in 0.8.0

**Connections → Google** (under the recipients list) and **Connections → Microsoft** each have an **Agentic signature** field: plain text (up to 1000 characters) that is added to the end of every email and draft your agents write from that account, so a recipient can tell an agent wrote it. `google_mail` and `microsoft_mail` add it on `create_draft` (replies too) and on `update_draft` with a new body; Flight Deck adds it to every composed Gmail send and to topic auto-replies. HTML bodies get an escaped block inside `</body>`; a body that already ends with the signature is left alone. If Flight Deck doesn't answer, the draft is made without it and the tool says so. Stored per user (`GET`/`PUT /fd/google/gmail-send` → `signature`, `GET`/`PUT /fd/microsoft/mail-signature`).

### Gmail topic auto-replies — NEW in 0.8.0

**Connections → Google → Topic auto-replies** lets one of your agents answer, **with no confirmation**, inbound email that is *only* about one of your narrow topics ("Emails that ask for our office opening hours"). Off by default, its own switch (it works with *agents may send* off), Gmail only, and the engine runs inside Flight Deck — no agent code changes.

- **Topics are narrow by construction.** Up to 10 topics, at most 5 on. Each needs an "Emails that…" description of at least 6 words and an example; catch-all wording ("any", "all emails", "bilo što") is refused. A topic switches on only after a **breadth test** — a dry run on your recent mail that matched at most 25 % of it — and editing what it matches switches it off again.
- **Who gets a reply.** Only senders whose mail passed Gmail's own authentication; auto-responders, mailing lists, bounces, your own mail, old mail and mail where Reply-To diverges are skipped. Mail that was already in the inbox when you switched it on is never answered. Caps per thread, per sender, a flood brake and a daily limit (default 10, at most 50). A draft or reply of your own in the thread means the engine stays out of that thread for good.
- **What a reply may say.** The writer can look things up read-only (the topic's folder, insights, playbooks, web search). Before sending, an outbound guard blocks secrets, credentials, card/IBAN/OIB/JMBG numbers, internal hosts and paths, your **never-disclose** terms (up to 50), profile copy, unexpected links, amounts and contacts; a block keeps no text, a hold puts the reply in Flight Deck as a **held card** (Send / Edit & send / Dismiss) — never a Gmail draft. A second judge (Jev, else your balanced/reasoning tier) checks the reply before it goes.
- **Sends** go to the verified sender only, marked `Auto-Submitted`, with your agentic signature, and appear in your sends list and bell (`origin: autoreply`). Three replies blocked for leaking, a disconnect or a changed Google account switch you off.

Needs `gmail.readonly` (or `gmail.modify`) plus `gmail.send` (or `gmail.compose`) on the OAuth client; users reconnect after an admin adds them. Deck-wide kill switches: `FD_GMAIL_AUTOREPLY=off` (and `FD_GMAIL_SEND=off`, which stops all sending). Tuning: `FD_GMAIL_AUTOREPLY_POLL_SECONDS` (default 30, at least 15) and `FD_GMAIL_AUTOREPLY_CLASSIFY_PER_DAY` (default 200, 20–1000).

### When an agent says it sent something — NEW in 0.8.0

Weak models sometimes answer "**Sent** …" or "Meeting request sent to Borjan — Tue 20 Oct" without calling any tool, complete with made-up Message IDs. A fabrication guard now judges every final reply:

- A claim that an **email went out** is false when no send happened this turn (`google_mail`, `microsoft_mail`, `send_mail`, a mail MCP tool, or a send a peer or lane relayed) and it quotes no Message ID an earlier send returned.
- A claim that a **calendar event** was created, moved, cancelled, answered or an invitation sent is false when no calendar change went through this turn. Read-backs ("is scheduled for 15:00", "It's in your calendar") are not claims.
- **Made-up IDs** (mail, calendar, Drive, Graph shaped) that nothing the model was given contained, and **narrated lookups** in a turn that made no tool call, are caught too.

A caught reply is stored as rejected (never replayed to the model) and the agent is asked, at most twice, for a **true** reply — the corrective never tells it to go and make the send or change. Claims written next to tool calls wait until those calls answer. Prompts say only "Email sent." with a Message ID, or "Created event …", proves the action.

---

## Microsoft 365 (Outlook, Calendar, OneDrive) — NEW in 0.8.0

`microsoft_mail`, `microsoft_calendar` and `microsoft_onedrive` are the Microsoft twins of the Google tools, with the same connection model: Flight Deck owns the sign-in, each Flight Deck user connects **their own** Microsoft account (work/school or personal Outlook.com), and an agent only ever gets a short-lived Graph token for its **owner**. Auth-enabled decks only.

1. **Register an app** in Microsoft Entra ID (App registrations → New registration). Add a **Web** redirect URI: `https://<your deck>/fd/microsoft/callback` (or `http://localhost:25080/fd/microsoft/callback`), and create a client secret.
2. **Connections → Microsoft** (admin): paste the client ID and secret, pick the tenant (`common`, `organizations`, `consumers` or your tenant ID) and the Graph scopes. The default set is `Mail.ReadWrite`, `Calendars.ReadWrite`, `Files.ReadWrite.All` (plus sign-in scopes and `offline_access`); `Mail.Send` is offered but unused — the tools never send. Changing the client, tenant or scopes disconnects everyone.
3. **Each user** clicks **Connect** on the same card (kiosk users get it in the Connections dialog). Once configured, the Director panel shows a Microsoft traffic light.

The tools stay hidden until the owner has connected and are owner-only (not available to a shared agent's members). Agents draft but never send Outlook mail; calendar changes that would email attendees need `notify_attendees: true` and a human-asked turn (see [microsoft_calendar](#microsoft_calendar--new-in-080)).

Deploy: pull, restart Flight Deck and agents, hard-refresh the browser, then register the app as above.

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

## Payments (bill pay and purchases) — NEW in 0.8.1

Off unless Flight Deck runs with `FD_PAYMENTS=1`; install the extra for barcode and invoice reading (`pip install "captain-claw[payments]"`). Then an agent's [`payments`](#payments--new-in-081) tool helps its owner pay bills and buy things **without Captain ever moving money or seeing a card**: the user approves every payment in their own bank, Revolut or Link app. The owner works on the **Payments** page (sidebar). It was built in phases; each plan document has the design and the decisions.

- **Phase 0 — bill pay by barcode** ([docs/payments-phase0-plan.md](docs/payments-phase0-plan.md)). The agent prepares a payment from an invoice (PDF or photo; the invoice's own barcode is read when it has one, and every field is checked: the IBAN, the HR model and *poziv na broj*, an RF reference) and hands over a HUB3 PDF417 or EPC QR barcode the user scans in their bank app, plus a `/p/<token>` phone page with **I paid** and **Skip**. Blocking warnings (a changed IBAN, a duplicate, a misread amount, over the per-user cap) clear only by the owner's acknowledgement, never by the agent. The agent says "paid" only from the user's own words, never on an automated turn. Card numbers, CVVs and expiry dates are scrubbed at every entry point, typed or spoken, and the fabrication guard catches a reply that claims a payment that didn't happen.
- **Phase 1 — Revolut Business drafts** ([docs/payments-phase1-plan.md](docs/payments-phase1-plan.md)). The payment goes into the user's Revolut Business as a draft they send in the Revolut app (Captain's token can't pay), after Verification of Payee; it is marked settled when Revolut says it completed.
- **Phase 2 — pay-by-bank links** ([docs/payments-phase2-plan.md](docs/payments-phase2-plan.md)). Through the deck's Enable Banking application (`FD_ENABLEBANKING_APP_ID`, `FD_ENABLEBANKING_KEY`; PSD2), one link to the user's own bank, where they approve with the bank's own strong authentication. Live payments need a payment-initiation licence or a contract with the provider.
- **Phase 3 — purchases** ([docs/payments-phase3-plan.md](docs/payments-phase3-plan.md)). Buying from an online shop with a one-time card that exists only after the user approves this seller and amount — the Stripe Link agent wallet (`FD_LINK_*`, US and Canada accounts) or a Revolut card made for the one purchase — typed into the checkout form without the model ever seeing it. Shops that speak ACP, and AP2 merchant agents (`FD_AP2=1`), need no browser.
- **Phase 4 — the bank statement** ([docs/payments-phase4-plan.md](docs/payments-phase4-plan.md)). Read-only statements (`FD_PAYMENTS_STATEMENTS=1`; Enable Banking or the Revolut Business statement, read at `FD_PAYMENTS_SYNC_TIMES`, default `07:30,16:30`) confirm bills as paid and flag double payments and reversals; incoming payments and balances are never read. A daily digest reminds the owner of bills due soon or overdue (outside `FD_SCHEDULER_QUIET_HOURS`), to the bell and the chat they last prepared a bill from. Bills that repeat are spotted and suggested; once the owner tracks one, Flight Deck prepares it each period (or waits for the invoice when the amount changes).

---

## Native Vision — NEW in 0.8.1

An agent whose own model sees images now gets the turn's pictures **as images in its own request**, instead of a text description from a separate vision model.

- **The switch:** `model.vision: auto | on | off`, and `vision` on each allowed model. `auto` is on for model families known to be multimodal and for Ollama. Local Qwen builds need `vision: on` (text-only builds share the names). With vision off, pictures are described as before: by an allowed model with `model_type: vision`, or by a multimodal peer agent.
- **In Flight Deck:** a **Vision** select in the Spawner, a **Sees images** checkbox per Library tier (carried into archetype, Forge, Library and Quick chat spawns), a Vision select and a "vision" badge in the agent card's model editor, and **Vision** in the kiosk agent Settings. Vision is read when the agent starts, so a change needs a restart. Pictures attached in Flight Deck's chats go to the agent as images, not text references.
- **What goes inline:** only the current turn's pictures — the user's attachments first, then the newest pictures tools returned (`browser` screenshots, `pinchtab`, `screen_capture`, `cv`, and `image_vision` / `image_ocr` called by the model). Earlier turns keep path markers (`[Attached image: …]`), and `image_vision` on a path shows a picture again. Limits are in [`context.inline_images`](#context): 4 per call, the newest 2 tool pictures, 1280 px, at most 30 % of the context window.
- **Self-healing:** a server that refuses image input has the picture replaced by the fallback description in the same call and is never sent one again in that process. The refusal is remembered in `~/.captain-claw/vision_rejected.json` for a day, so a restart doesn't pay for it again and a server that gains vision is tried again.
- **Everywhere:** `image_vision` is always available, so an archetype that left it out can still look at a picture; when no description could be made the model is told to say so rather than guess. Telegram photos and a flow's vision step use the same fallback for a model that doesn't see. HEIC, HEIF, AVIF and TIFF uploads are stored as JPEG.

---

## Reasoning Effort — NEW in 0.8.1

One scale for every model's "think harder" control: **off · minimal · low · medium · high · xhigh · max**, plus **on** for models that only switch thinking on or off. `captain_claw/llm/reasoning.py` translates it for each model family and transport — Claude, OpenAI GPT-5.x and the o-series (Codex included), Gemini and Gemma, Grok, DeepSeek, Kimi, GLM, MiniMax, Qwen, Mistral, gpt-oss and Nemotron; local servers through `chat_template_kwargs`, Ollama through `think`, OpenRouter through its `reasoning` field. A level the model doesn't offer is moved to the nearest one it does, so switching an agent's model never sends a value the new model rejects.

- **The web chat.** The kiosk composer's reasoning button (beside "What can I do?") offers **Auto** (the model's own default) plus exactly the levels the agent's model offers. The pick is kept per agent and sent with every message as `reasoning_effort`.
- **Every other source** — WhatsApp, Telegram, Messenger, Discord, the app, the glasses, each voice device, scheduled tasks, flows, other agents, autonomous work, the API, shared-agent members and public links — takes the agent's own settings in `~/.captain-claw/reasoning_effort.json`:

  ```json
  {"default": "low", "channels": {"whatsapp": "auto", "device:stopwatch": "minimal", "scheduled": "high"}}
  ```

  A turn's level is looked up by its source (`device:stopwatch`), then its channel (`device`), then `default`; `auto` sends nothing. The file is read again whenever it changes. Edit it in the kiosk agent **Settings → Reasoning effort**, or with `GET` / `PUT /fd/agent-reasoning/{kind}/{id}` (the levels offered come from the agent's configured model).
- **The agent can change it too.** Its `reasoning_effort` tool changes the same settings — the chat box's picker for a web chat message, that source's row in Settings for any other — at once, for the rest of the turn and later messages (or with `scope: "turn"` for the rest of this turn only). Asked "think harder" on WhatsApp, it sets WhatsApp's level; asked in the web chat, the chat box's picker moves. See [reasoning_effort](#reasoning_effort).
- **Only the conversation thinks harder.** The level applies to the main conversation's calls; titles, rephrasing, compaction and reflections keep the model's default. Shared-agent members get no picker.
- **Self-healing:** a server that refuses a reasoning field gets the request again without it, and the model keeps its default for the rest of the run.

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
| `YDC_API_KEY` | You.com Search API key (`tools.web_search.provider: youcom`) |
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
| `JEV_API_KEY` (or `TYPESAFE_API_KEY`) | Jev (TypeSafe AI) API key for the `jev` tool — standalone agents only; under Flight Deck add it in **Connections → Jev** |
| `SONIOX_API_KEY` | Flight Deck: last-fallback Soniox key (speech-to-text and TTS for WhatsApp voice notes, the glasses bridge, the HUD, the StickS3 device and agents). Prefer Connections → Soniox (per user) and Admin → Integrations → Soniox team key, where **Move server key here** imports this value. Agents no longer inherit it: they transcribe through Flight Deck with their owner's key. |
| `WHATSAPP_REACTIONS` / `WHATSAPP_REACTION_TIMEOUT` | Flight Deck WhatsApp bridge: `off` disables emoji reactions (on by default); timeout in seconds (default 8) |
| `STREAM_NARRATION` | Live narration (one message per note the agent writes beside its tool calls) is on by default. `off` turns it off: Telegram folds the notes into the final answer (NEW in 0.8.1), the WhatsApp and Messenger bridges leave them out |
| `DEVICE_VOICE_MALE` / `DEVICE_VOICE_FEMALE` | Flight Deck: the Soniox voices the M5Stack devices' male / female choice uses (default `Adrian` / `Emma`) |
| `FD_PAYMENTS`, `FD_SCHEDULER_QUIET_HOURS`, `FD_AGENT_WS_WAIT_S`, `FD_DESKTOP`, `FD_DATA_KEY_FILE`, … | Flight Deck settings — see the Configuration table under [Flight Deck](#flight-deck) |
| `CLAW_TURN_REGISTER` | `off` stops the agent reporting its turns to the [Taskboard](#taskboard--new-in-084) (on by default since 0.8.4) |
| `CLAW_WRITE_GUARD` | `0` turns off the `write` tool's empty/placeholder refusal and read-back check (on by default since 0.8.0) |
| `FD_JEV_BASE_URL` | Flight Deck: send every user's Jev calls to another Jev-compatible endpoint (a proxy in front of `https://api.typesafe.ai`, or a local stub); default `https://api.typesafe.ai` |
| `CLAW_BOTPORT__URL` | BotPort WebSocket URL for the agent's direct (legacy) link (e.g. `wss://botport.kstevica.com/ws`) |
| `CLAW_BOTPORT__KEY` | BotPort authentication key |
| `CLAW_BOTPORT__SECRET` | BotPort authentication secret |
| `MAIL_FROM_ADDRESS` | Email sender address |
| `MAIL_FROM_NAME` | Email sender name |
| `SMTP_HOST` | SMTP server host |
| `SMTP_PORT` | SMTP server port |
| `SMTP_USERNAME` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `CLAW_*` | Override any config field (e.g. `CLAW_MODEL__PROVIDER=anthropic`) |

The `CLAW_` prefix uses double underscore for nested fields: `CLAW_TOOLS__WEB_SEARCH__API_KEY=key`.

`.env` files in the current working directory are loaded automatically. Format: `KEY=VALUE` (one per line, `#` comments, `export` prefix supported).
