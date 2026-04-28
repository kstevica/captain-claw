# Captain Claw

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Interface](https://img.shields.io/badge/interface-terminal%20%7C%20web%20UI-black)](#quick-start)
[![Models](https://img.shields.io/badge/models-OpenAI%20%7C%20Claude%20%7C%20Gemini%20%7C%20Ollama%20%7C%20OpenRouter-orange)](#multi-model-support)

![Captain Claw](docs/screenshot.png)

[![asciicast](https://asciinema.org/a/R1UPSHi4Y6UrOnpY.svg)](https://asciinema.org/a/R1UPSHi4Y6UrOnpY)

[![Watch the video](https://img.youtube.com/vi/4g_aA_WnEaw/maxresdefault.jpg)](https://www.youtube.com/watch?v=4g_aA_WnEaw)

An open-source AI agent with multi-agent orchestration, autonomous cognitive systems, and a full management dashboard. Runs locally, supports every major LLM provider, and ships with 44 built-in tools.

## What's New in 0.4.22

- **Nano Mode** — A restricted-tool runtime tuned for tiny local models (Qwen3, Llama 3.2, Phi). Compressed system prompt, 7-tool surface (shell/write/read/edit/glob/datastore/insights), aggressive empty-message filtering, and memory-injection delimiters that stop small models from echoing context back as their reply. A 3B-parameter model on a laptop can now drive a useful loop.
- **Remote GPU via vast.ai** — Drive an Ollama server hosted on vast.ai with auto-wake on first request and auto-sleep when idle. Rent an H100 by the hour for evening work without paying for a 24/7 cloud GPU. Per-agent override; existing local-Ollama setups untouched.
- **Smart `web_fetch`** — Plain HTTP first, transparent fallback to a headless Playwright browser when the response looks like an unrendered SPA shell. SPAs return real article text; static sites stay fast.
- **Prompt Builder** — Compose reusable multi-step prompt templates with `{{variable}}` slots. Built-in variables for session ID, workspace path, user email, and today's date.
- **Live Token-Speed Telemetry** — Real-time tokens-per-second in the agent sidebar (`<output_tps> tok/s / <total_llm_tps> llm`). Compare a 30B model at 2 tok/s on a Mac vs 60 tok/s on vast.ai at a glance.
- **Reliability fixes** — Empty-response feedback loop on local Qwen3/DeepSeek-R1 thinking models is fixed (`think: false` by default, empty assistant messages no longer persisted, history self-heals on next turn). Memory-injection echoes on small/cheap cloud models are gone (context notes now wrapped with explicit "do not repeat" delimiters).

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the full changelog.

## What Makes Captain Claw Different

### Flight Deck — Multi-Agent Command Center

A full management dashboard for running teams of AI agents. Spawn, monitor, configure, and coordinate agents from a single UI.

```bash
captain-claw-fd    # http://0.0.0.0:25080
```

- **Agent Forge** — Describe a business goal in plain text. An LLM designs a specialized team with roles, tools, operating procedures, and a lead coordinator. Review, customize, and spawn the entire team in one click.
- **Agent Council** — Structured multi-agent deliberation. Run brainstorms, debates, reviews, or planning sessions with 2-N agents. Each agent self-scores suitability, chooses actions (answer, challenge, refine, broaden), and responds in moderated rounds. A moderator synthesizes conclusions; all agents vote. Export as markdown minutes.
- **Fleet Communication** — Agents discover peers automatically. Consult (synchronous ask) or delegate (asynchronous queue) tasks to specialist agents. Shared workspace and file transfer across the fleet.
- **Director Panel** — Unified overview of all agents. Broadcast messages fleet-wide. Per-agent token/cost analytics, trace timelines, datastore browser, file browser, config editor.
- **Multi-user Auth** — JWT authentication, admin dashboard, rate limiting, and quotas.

### Cognitive Architecture

Captain Claw has a five-layer memory system and autonomous cognitive processes that run without user intervention.

**Memory Layers:**

| Layer | What it stores | How it's used |
|---|---|---|
| Working Memory | Current conversation in the LLM context window | Immediate reasoning |
| Semantic Memory | Hybrid vector + BM25 full-text search over documents and sessions | Auto-injected when relevant to the current query |
| Deep Memory | Typesense-backed long-term archive, scales to millions of documents | Searched on demand for deep recall |
| Insights | Auto-extracted facts, contacts, decisions, and deadlines (SQLite + FTS5) | Cross-session knowledge injected into system prompt |
| Nervous System | Autonomous "intuitions" — patterns, hypotheses, and connections | Surfaces non-obvious findings the agent wouldn't otherwise notice |

**Autonomous Processes:**

- **Dreaming** — Background dream cycles cross-reference all memory layers to synthesize intuitions. Runs after every N messages and during idle hours. Intuitions have confidence scores that decay over time unless validated.
- **Tension Tracking** — Holds unresolved contradictions (like musical dissonance) rather than forcing premature resolution. Tensions persist until evidence resolves them.
- **Maturation Pipeline** — New intuitions sit through multiple dream cycles before being surfaced to the agent, reducing noise.
- **Cognitive Tempo** — Detects whether the user is in deep contemplative mode or rapid task execution, and adapts processing depth accordingly (adagio / moderato / allegro).
- **Cognitive Modes** — Seven tunable behavioral profiles (Ionian through Locrian, inspired by musical scales) that shift the agent between analytical, creative, cautious, and exploratory approaches.
- **Self-Reflection** — Periodic self-assessment that reviews conversations, memory, and completed tasks to generate improvement directives injected into the system prompt.
- **Insights Extraction** — Automatically identifies durable knowledge from conversations — deduplicates, categorizes, and stores for future context injection.

**Visualization:**

- **Brain Graph** — Interactive 3D force-directed graph of the entire cognitive topology. Insights, intuitions, tasks, contacts, and sessions rendered as typed nodes with provenance edges. Live WebSocket updates.
- **Process of Thoughts** — Full lineage tracking across all cognitive subsystems. Every message, insight, intuition, and task is connected via provenance IDs, forming a traversable thought graph.

### Orchestrator / DAG Mode

Decompose complex tasks into a dependency graph and execute subtasks in parallel across separate agent sessions.

```
/orchestrate Research startups in 3 countries, analyze founders, create comparison spreadsheet
```

- LLM decomposes the prompt into a task DAG with dependencies
- Parallel execution with configurable worker count
- Shared workspace for inter-task data flow
- Structured output validation (JSON Schema with auto-retry)
- Real-time trace timeline (Gantt-style visualization)
- Headless CLI mode for cron/scripts: `captain-claw-orchestrate`

### BotPort — Agent-to-Agent Network

Connect multiple Captain Claw instances through a routing hub. Agents delegate tasks to specialists based on expertise tags, persona matching, or LLM-powered routing.

- **BotPort Swarm** — DAG-based multi-agent orchestration across networked instances. Approval gates, retry with fallback, checkpointing, inter-agent file transfer (up to 50 MB), cron scheduling, and a visual dashboard.

### MCP Server

Captain Claw runs as a Model Context Protocol server over stdio — Claude Desktop and other MCP clients can browse sessions, read conversation history, and send prompts to the full agent.

```bash
captain-claw-mcp    # stdio, configure in claude_desktop_config.json
```

### Safety Guards

Three layers of protection that run before, during, and after agent operations:

- **Input guards** — Validate user intent before the LLM sees it
- **Script guards** — AST-level analysis of generated code before execution
- **Output guards** — Validate tool results for hallucinations and safety

Guards support two modes: `stop_suspicious` (block automatically) or `ask_for_approval` (prompt the user).

## Multi-Model Support

Mix providers freely — each session independently selects its model.

| Provider | Models |
|---|---|
| OpenAI (API key) | GPT-5.4, GPT-5.4-mini, GPT-5.4-nano, o3, o4-mini, gpt-image-1.5 |
| OpenAI (Sign in with ChatGPT) | `gpt-5`, `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`, `gpt-5.2-codex`, `gpt-5.3-codex` — billed against your ChatGPT plan, no API key |
| Anthropic | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 (with prompt caching) |
| Google | Gemini 3.1 Pro/Flash, Gemini 2.5 Pro/Flash (API key or OAuth/Vertex) |
| Ollama | Any local model |
| LiteRT (on-device) | `.litertlm` Gemma models running locally via an isolated subprocess worker |
| OpenRouter | 200+ models via meta-router |

## Quick Start

```bash
pip install captain-claw
export OPENAI_API_KEY="sk-..."          # or ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
captain-claw-web                         # http://127.0.0.1:23080
```

```bash
captain-claw-web          # Web UI (default)
captain-claw              # Interactive terminal
captain-claw --tui        # Terminal UI
captain-claw-fd           # Flight Deck multi-agent dashboard
captain-claw-mcp          # MCP server for Claude Desktop
botport                   # Agent-to-agent routing hub
```

First run starts onboarding automatically. For Ollama, no key needed — set `provider: ollama` in `config.yaml`.

## 44 Built-in Tools

Shell, file I/O, web fetch/search, browser automation, PDF/DOCX/XLSX/PPTX extraction, image generation (DALL-E), OCR, vision, TTS, STT, email (SMTP/Mailgun/SendGrid), Google Workspace (Drive, Docs, Sheets, Slides, Gmail, Calendar), desktop automation, screen capture with voice commands, persistent cross-session memory (todos, contacts, scripts, APIs, playbooks), datastore (SQLite tables with protection rules), deep memory (Typesense), personality system, cron scheduling, BotPort fleet discovery, and Termux (Android).

See [USAGE.md](USAGE.md#tools-reference) for the full reference.

## Web UI

Chat, Computer (retro-themed research workspace with 14 themes), monitor pane, instruction editor, command palette, persona selector, datastore browser, deep memory dashboard, insights browser, nervous system browser, Brain Graph 3D visualization, reflections dashboard, personality editor, playbook editor, and LLM usage analytics.

**Computer** — A standalone research workspace at `/computer` with themed visual generation, exploration trees, folder browser (local + Google Drive), file attachments, PDF export, and public mode with BYOK (Bring Your Own Key).

## Docker

```bash
docker pull kstevica/captain-claw:latest
docker run -d -p 23080:23080 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/docker-data/home-config:/root/.captain-claw \
  -v $(pwd)/docker-data/workspace:/data/workspace \
  kstevica/captain-claw:latest
```

See [README_DETAILED.md](README_DETAILED.md#docker) for Docker Compose and persistent data setup.

## Configuration

YAML-driven with environment variable overrides (`CLAW_` prefix).

```yaml
model:
  provider: gemini
  model: gemini-2.5-flash
  allowed:
    - id: claude-sonnet
      provider: anthropic
      model: claude-sonnet-4-20250514
    - id: gpt-4o
      provider: openai
      model: gpt-4o

web:
  enabled: true
  port: 23080
```

**Load precedence:** `./config.yaml` > `~/.captain-claw/config.yaml` > env vars > `.env` > defaults.

Full reference: [USAGE.md](USAGE.md#configuration-reference) (23 config sections).

## Architecture

| Component | Path |
|---|---|
| Agent (14-mixin composition) | `captain_claw/agent.py` |
| LLM providers | `captain_claw/llm/` |
| 44 tools + registry | `captain_claw/tools/` |
| Flight Deck (FastAPI + React) | `captain_claw/flight_deck/` |
| DAG orchestrator | `captain_claw/session_orchestrator.py` |
| Semantic memory (vector + BM25) | `captain_claw/semantic_memory.py` |
| Deep memory (Typesense) | `captain_claw/deep_memory.py` |
| Insights (fact extraction) | `captain_claw/insights.py` |
| Nervous system (dreaming) | `captain_claw/nervous_system.py` |
| Cognitive tempo | `captain_claw/cognitive_tempo.py` |
| MCP server | `captain_claw/mcp_serve.py` |
| BotPort client | `captain_claw/botport_client.py` |
| Web UI + REST API | `captain_claw/web/` |
| Prompt templates (~100 files) | `captain_claw/instructions/` |
| Config (Pydantic) | `captain_claw/config.py` |

## Documentation

- **[USAGE.md](USAGE.md)** — Complete reference for all commands, tools, config, and features
- **[README_DETAILED.md](README_DETAILED.md)** — Extended README with feature-by-feature breakdown

## License

[MIT](LICENSE)
