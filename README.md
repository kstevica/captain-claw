# Captain Claw

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Interface](https://img.shields.io/badge/interface-terminal%20%7C%20web%20UI-black)](#quick-start)
[![Models](https://img.shields.io/badge/models-OpenAI%20%7C%20Claude%20%7C%20Gemini%20%7C%20Ollama-orange)](#feature-snapshot)
[![Guardrails](https://img.shields.io/badge/guardrails-input%20%7C%20output%20%7C%20script%2Ftool-red)](#guards)

A terminal-first agentic system for everyday work. Multi-model LLM support, persistent sessions, built-in safety guards, tool execution, and a web UI — all in one CLI.

## Feature Snapshot

| Capability | What it does |
|---|---|
| Multi-model routing | Mix GPT, Claude, Gemini, and Ollama in one CLI |
| Per-session model selection | Keep one session on Claude, another on GPT, another on Ollama |
| Persistent multi-session workflows | Resume any session exactly where you left off |
| Built-in safety guards | Input, output, and script/tool checks before anything runs |
| 13 built-in tools | Shell, files, web fetch/search, docs, email, TTS, Google Drive |
| Skills system | OpenClaw-compatible skills with auto-discovery and GitHub install |
| Orchestrator / DAG mode | Decompose complex tasks into parallel multi-session execution |
| Memory / RAG | Hybrid vector + text retrieval across workspace and sessions |
| Web UI | Chat, monitor pane, instruction editor, command palette |
| Remote integrations | Telegram, Slack, Discord with secure pairing |
| Cron scheduling | Interval, daily, and weekly tasks inside the runtime |
| OpenAI-compatible API | `POST /v1/chat/completions` proxy with agent pool |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/kstevica/captain-claw
cd captain-claw
python -m venv venv
source venv/bin/activate
pip install -e .
```

Requires **Python 3.11** or higher.

### 2. Set an API key

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

Use only the keys you need. For Ollama, no key is required — just set `provider: ollama` in `config.yaml`.

### 3. Launch

```bash
captain-claw
```

First run starts interactive onboarding automatically. To re-run it later: `captain-claw --onboarding`.

### 4. Try it

```text
> Investigate failing integration tests and propose a fix.

/models                          # see available models
/session model claude-sonnet     # switch this session to Claude

/new release-notes               # create a second session
/session model chatgpt-fast      # use GPT for this one

> Draft release notes from the previous session updates.

/session run #1 summarize current blockers   # run a prompt in session #1
```

Each session keeps its own model, context, and history.

### 5. Launch the Web UI (optional)

```bash
captain-claw --web
```

Open **http://127.0.0.1:8340** in your browser. Use `Ctrl+K` for the command palette, `Ctrl+B` for the sidebar, and edit instruction files live in the Instructions tab.

To start the web UI automatically on every launch, set `web.enabled: true` in `config.yaml`.

## How It Works

### Sessions

Sessions are first-class. Create named sessions for separate projects, switch instantly, and persist everything.

```text
/new incident-hotfix
/session model claude-sonnet
/session protect on                            # prevent accidental /clear
/session procreate #1 #2 "merged context"      # merge two sessions
/session run #2 summarize current blockers     # run prompt in another session
/session export all                            # export chat + monitor history
```

### Tools

Captain Claw ships with 13 built-in tools. The agent picks the right tool for each task automatically.

| Tool | What it does |
|---|---|
| `shell` | Execute terminal commands |
| `read` / `write` / `glob` | File operations and pattern matching |
| `web_fetch` | Fetch and extract readable web content |
| `web_search` | Search the web via Brave Search API |
| `pdf_extract` | Extract PDF content to markdown |
| `docx_extract` | Extract Word documents to markdown |
| `xlsx_extract` | Extract Excel sheets to markdown tables |
| `pptx_extract` | Extract PowerPoint slides to markdown |
| `pocket_tts` | Generate speech audio (MP3) locally |
| `send_mail` | Send email via SMTP, Mailgun, or SendGrid |
| `google_drive` | List, search, read, upload, and manage Google Drive files |

See [USAGE.md](USAGE.md#tools-reference) for full parameters and configuration.

### Guards

Three built-in guard types protect against risky operations:

```yaml
guards:
  input:
    enabled: true
    level: "ask_for_approval"     # or "stop_suspicious"
  output:
    enabled: true
    level: "stop_suspicious"
  script_tool:
    enabled: true
    level: "ask_for_approval"
```

Guards run before LLM requests (input), after model responses (output), and before any command or tool execution (script_tool). See [USAGE.md](USAGE.md#guard-system) for details.

## Configuration at a Glance

Captain Claw is YAML-driven with environment variable overrides.

```yaml
model:
  provider: "openai"
  model: "gpt-4o-mini"
  allowed:
    - id: "claude-sonnet"
      provider: "anthropic"
      model: "claude-sonnet-4-20250514"

tools:
  enabled: ["shell", "read", "write", "glob", "web_fetch", "web_search",
            "pdf_extract", "docx_extract", "xlsx_extract", "pptx_extract",
            "pocket_tts", "send_mail", "google_drive"]

web:
  enabled: true
  port: 8340
```

**Load precedence:** `./config.yaml` > `~/.captain-claw/config.yaml` > environment variables > `.env` file > defaults.

For the full configuration reference (17 sections, every field), see [USAGE.md](USAGE.md#configuration-reference).

## Advanced Features

Each of these is documented in detail in [USAGE.md](USAGE.md).

- **[Orchestrator / DAG mode](USAGE.md#orchestrator--dag-mode)** — `/orchestrate` decomposes a complex request into a task DAG and runs tasks in parallel across separate sessions with real-time progress monitoring.

- **[Skills system](USAGE.md#skills-system)** — OpenClaw-compatible `SKILL.md` files. Auto-discovered from workspace, managed, and plugin directories. Install from GitHub with `/skill install <url>`.

- **[Memory / RAG](USAGE.md#memory-and-rag)** — Hybrid vector + text retrieval. Indexes workspace files and session messages. Configurable embedding providers (OpenAI, Ollama, local hash fallback).

- **[Cron scheduling](USAGE.md#cron-commands)** — Pseudo-cron within the runtime. Schedule prompts, scripts, or tools at intervals, daily, or weekly. Guards remain active for every cron execution.

- **[Execution queue](USAGE.md#execution-queue)** — Five queue modes (steer, followup, collect, interrupt, queue) control how follow-up messages are handled during agent execution.

- **[Remote integrations](USAGE.md#remote-integrations)** — Connect Telegram, Slack, or Discord bots. Unknown users get a pairing token; the operator approves locally with `/approve user`.

- **[OpenAI-compatible API](USAGE.md#openai-compatible-api-proxy)** — `POST /v1/chat/completions` endpoint proxied through the Captain Claw agent pool. Streaming supported.

- **[Google Drive + OAuth](USAGE.md#google-oauth-and-google-drive)** — Connect your Google account for Drive file operations (list, search, read, upload, create, update) and Vertex AI model access.

- **[Send mail](USAGE.md#send-mail)** — SMTP, Mailgun, or SendGrid. Supports attachments up to 25 MB.

- **[Document extraction](USAGE.md#tools-reference)** — PDF, DOCX, XLSX, PPTX converted to markdown for agent consumption.

- **[Context compaction](USAGE.md#context-compaction)** — Auto-compacts long sessions at configurable thresholds. Manual compaction with `/compact`.

- **[Session export](USAGE.md#session-commands)** — Export chat, monitor, pipeline trace, or pipeline summary to files.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check captain_claw/
```

### Architecture

| Path | Role |
|---|---|
| `captain_claw/agent.py` | Main orchestration logic |
| `captain_claw/llm/` | Provider abstraction (OpenAI, Anthropic, Gemini, Ollama) |
| `captain_claw/tools/` | Tool registry and 13 tool implementations |
| `captain_claw/session/` | SQLite-backed session persistence |
| `captain_claw/skills.py` | Skill discovery, loading, and invocation |
| `captain_claw/cli.py` | Terminal UI |
| `captain_claw/web_server.py` | Web server (WebSocket + REST + static) |
| `captain_claw/config.py` | Configuration and env overrides |
| `instructions/` | Externalized prompt and instruction templates |

## FAQ

**Is Captain Claw only for coding?**
No. It handles coding, ops automation, web research, document processing, email, and multi-session orchestration.

**Can I use local models only?**
Yes. Set provider to `ollama` and run fully local.

**Can I run different models at the same time?**
Yes. Model selection is per session. Different sessions can use different providers and models simultaneously.

**Do I need guards enabled?**
No. Guards are off by default. Enable them when you want stricter safety behavior.

**Is there a web interface?**
Yes. Run `captain-claw --web` and open `http://127.0.0.1:8340`. Same agent, sessions, tools, and guardrails as the terminal.

**Where is the full reference?**
See [USAGE.md](USAGE.md) for comprehensive documentation of every command, tool, config option, and feature.

## Get Started

```bash
git clone https://github.com/kstevica/captain-claw
cd captain-claw
python -m venv venv && source venv/bin/activate
pip install -e .
captain-claw
```

If Captain Claw is useful to you, [give the repo a star](https://github.com/kstevica/captain-claw) to help others find it.

Found a bug or have a feature idea? [Open an issue](https://github.com/kstevica/captain-claw/issues). Contributions welcome.

## License

[MIT](LICENSE)
