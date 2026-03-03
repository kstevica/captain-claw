# Captain Claw & BotPort

**Version:** 0.3.1.2
**Author:** Stevica Kuharski
**License:** MIT (Open Source)
**Repository:** github.com/kstevica/captain-claw
**Latest Release:** v0.3.1.2 — March 2, 2026

---

## Executive Summary

**Captain Claw** is an open-source AI agent framework that runs locally, supports multiple LLM providers simultaneously, and performs real work — coding, research, automation, document processing, and orchestration — through persistent sessions with built-in safety guards.

**BotPort** is its companion multi-agent routing hub that connects Captain Claw instances into a federated network, enabling specialist delegation across teams, machines, and geographies.

Together, they form a complete ecosystem for autonomous AI-powered task execution with local control, multi-model flexibility, and production-grade safety — challenging the dominance of closed, single-provider SaaS tools.

---

## The Problem

Today's AI agent landscape is fragmented and restrictive:

- **Provider lock-in.** Most tools force users into a single LLM provider. If you want GPT for coding and Claude for writing, you need separate tools.
- **No persistence.** Conversations vanish. Context is lost between sessions. The agent forgets everything you discussed yesterday.
- **No local control.** Cloud-only solutions send all data to third-party servers. Privacy-sensitive work requires workarounds.
- **No safety layer.** Agents execute commands blindly. There is no systematic way to review, approve, or block risky operations before they happen.
- **No collaboration between agents.** Each AI instance is an island. There is no standard way for agents to delegate tasks to specialists.
- **Limited tooling.** Most frameworks offer a handful of tools. Real-world workflows require file operations, web research, document processing, email, calendar, databases, and more — in a single session.

---

## The Solution

### Captain Claw — The Agent

Captain Claw is a locally-installable AI agent that treats **sessions, models, tools, and safety** as first-class concepts:

- **Multi-model by design.** Mix OpenAI GPT, Anthropic Claude, Google Gemini, and local Ollama models in a single runtime. Switch models per session — keep one conversation on Claude for nuanced analysis, another on GPT for fast iteration, a third on a local model for privacy.

- **Persistent sessions.** Every conversation is a named, resumable session with full history, model assignment, and context. Sessions survive restarts. They can be exported, merged, protected from deletion, and cross-referenced.

- **28 built-in tools.** From shell commands and file operations to web research, PDF/Word/Excel/PowerPoint extraction, image generation/OCR/vision, Google Drive/Calendar/Gmail integration, email sending, text-to-speech, a relational datastore, and more — all available to the agent automatically.

- **Built-in safety guards.** Three protection layers — input guard (before LLM calls), output guard (after responses), and script/tool guard (before command execution) — with configurable enforcement levels: ask-for-approval or stop-suspicious.

- **Memory systems.** Hybrid vector + keyword search across workspace files and session history. Cross-session persistent memory for todos, contacts, scripts, and API endpoints. Long-term deep memory archive via Typesense.

- **Runs anywhere.** Terminal CLI, web UI, TUI mode, desktop app (Electron), Android (Termux), or headless orchestrator. Available as a Python package (`pip install captain-claw`) or standalone binary.

### BotPort — The Network

BotPort connects multiple Captain Claw instances into a **federated agent network**:

- **Specialist delegation.** When an agent encounters a task outside its expertise, it routes a "concern" to the BotPort hub, which dispatches it to the best-qualified instance based on expertise tags and persona profiles.

- **Three-tier routing.** Tag matching (expertise overlap), LLM-assisted routing (semantic understanding of the request), and least-loaded fallback — ensuring concerns always find a handler.

- **Concern lifecycle.** Full lifecycle management — submission, acknowledgment, context negotiation, result delivery, follow-ups, and closure — with timeout handling for reliability.

- **WebSocket-based.** Real-time bidirectional communication over secure WebSocket connections. Instances can be on the same machine or across continents.

- **Dashboard.** Web-based monitoring of connected instances, active concerns, routing decisions, and network health.

---

## Key Features — Detail

### 1. Multi-Model Routing

Captain Claw does not pick one LLM. It supports all major providers simultaneously:

| Provider | Models | Use Case |
|----------|--------|----------|
| **OpenAI** | GPT-4o, GPT-5, GPT-5-Codex, o3, DALL-E 3, gpt-image-1 | General purpose, image generation |
| **Anthropic** | Claude Sonnet, Opus, Haiku | Nuanced analysis, long-context tasks |
| **Google** | Gemini Pro, Flash, Imagen | Multimodal, image generation |
| **Ollama** | Any local model | Privacy-first, offline operation |

Model selection is **per session** — not global. Users can run a Claude session for code review and a GPT session for release notes simultaneously, with seamless switching via a single command.

### 2. 28 Built-In Tools

Every tool is available to the agent automatically. No plugins to install, no APIs to configure beyond the initial setup:

**File & System:** shell, read, write, glob
**Web & Research:** web_fetch, web_get, web_search
**Documents:** pdf_extract, docx_extract, xlsx_extract, pptx_extract
**AI & Media:** image_gen, image_ocr, image_vision, pocket_tts
**Google:** google_drive, google_calendar, google_mail
**Communication:** send_mail
**Memory & Data:** todo, contacts, scripts, apis, datastore, typesense (deep memory)
**Advanced:** personality, botport, termux

### 3. Persistent Multi-Session Workflows

Sessions are the core organizational unit:

- **Named and resumable.** Create `incident-hotfix`, `release-notes`, `research-q2` — each with its own history and model.
- **Cross-session operations.** Run a prompt in another session, merge sessions, export history.
- **Protection.** Lock sessions to prevent accidental deletion.
- **Model assignment.** Each session remembers its model — switch without reconfiguring.

### 4. Safety Guards

Three independently configurable guard layers:

- **Input Guard:** Screens user prompts before they reach the LLM. Detects prompt injection, unsafe requests, and policy violations.
- **Output Guard:** Screens model responses before they reach the user. Catches harmful content, hallucinated actions, and policy violations.
- **Script/Tool Guard:** Reviews every shell command and tool invocation before execution. Prevents destructive operations, data exfiltration, and privilege escalation.

Each guard can operate in `ask_for_approval` mode (pause and ask the user) or `stop_suspicious` mode (block automatically).

### 5. Memory & RAG

**Semantic Memory:** Hybrid vector + keyword search. Indexes workspace files (up to 400 files, 256KB each) and session messages. Supports OpenAI embeddings, Ollama embeddings, or a local hash fallback for zero-API operation. Temporal decay scoring prioritizes recent content.

**Cross-Session Memory:** Persistent across all sessions — todos (auto-captured from conversation), contacts (auto-captured from email and conversation), scripts (auto-captured from file writes), API endpoints (auto-captured from web requests).

**Deep Memory:** Typesense-backed long-term archive. Indexes processed items, web fetches, and manual entries. Hybrid keyword + vector search. Web dashboard for browsing, searching, and exporting.

### 6. Orchestrator / DAG Mode

For complex, multi-step tasks:

- Decompose a request into a **directed acyclic graph** (DAG) of sub-tasks.
- Execute tasks in parallel across separate worker sessions.
- Monitor progress in real-time.
- Combine results automatically.

Available interactively (`/orchestrate`) or headless (`captain-claw-orchestrate`) for CI/CD and automation pipelines.

### 7. Chunked Processing Pipeline

Small-context models (20k-32k tokens) can process arbitrarily large content:

- Automatic detection when content exceeds the model's context window.
- Content is split into overlapping chunks processed sequentially.
- Results are combined via LLM synthesis or concatenation.
- Tested: a 108k-token document split into 9 chunks, processed in 48.5 seconds.

This means any model — including free, local Ollama models — can handle enterprise-scale documents.

### 8. Personality System

Dual-profile architecture for tailored responses:

- **Agent Profile:** Global identity — name, background, expertise areas, communication style.
- **User Profiles:** Per-user preferences and context. Each user gets a distinct interaction style without model retraining.

Telegram users receive automatic per-user profiles. Profiles are editable via conversation, REST API, or the web UI.

### 9. Relational Datastore

A built-in SQLite-backed relational database managed entirely by the agent:

- 19 tool actions: create tables, CRUD operations, raw SQL queries, CSV/XLSX import and export.
- Four-level protection system: table, column, row, and cell-level safeguards.
- Web dashboard for browsing, editing, querying, and file upload.
- Useful for structured data collection, inventory tracking, CRM-like workflows, and reporting.

### 10. Remote Integrations

Connect Captain Claw to messaging platforms:

- **Telegram:** Per-user sessions with concurrent execution. Unknown users receive a pairing token; the operator approves locally. Photos auto-sent.
- **Slack:** Bot integration with secure channel setup.
- **Discord:** Server integration with guild mapping.

All platforms share the same tool set, session semantics, and safety guards.

### 11. Cron Scheduling

Built-in pseudo-cron for automated workflows:

- Schedule prompts, scripts, or tool invocations at intervals, daily, or weekly.
- Guards remain active for every cron execution — no unmonitored operations.
- Managed via CLI, web UI, or remote integrations.

### 12. Skills System

OpenClaw-compatible plugin architecture:

- Skills are defined as `SKILL.md` markdown files — human-readable, version-controllable.
- Auto-discovered from workspace, managed directories, and plugin paths.
- Install from GitHub with a single command.
- Dependency tracking for binaries, environment variables, and configuration requirements.

### 13. Desktop App & Binary Distributions

- **Electron desktop app:** Native window wrapping the web UI with bundled Python backend.
- **Platform support:** macOS (DMG), Windows (NSIS installer), Linux (AppImage/deb).
- **Standalone binaries:** PyInstaller-built executables — no Python installation required.

---

## BotPort — Deep Dive

BotPort is a standalone server (also MIT-licensed) that transforms isolated Captain Claw instances into a collaborative network.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     BotPort Hub                         │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │ Registry │  │  Router  │  │  Concern Manager      │  │
│  │          │  │          │  │  (lifecycle tracking) │  │
│  └──────────┘  └──────────┘  └───────────────────────┘  │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │  Auth / Security │  │  Dashboard (web monitoring)  │ │
│  └──────────────────┘  └──────────────────────────────┘ │
└──────────┬─────────────────┬─────────────────┬──────────┘
           │ WSS             │ WSS             │ WSS
    ┌──────┴──────┐   ┌──────┴──────┐   ┌──────┴──────┐
    │  CC Instance│   │ CC Instance │   │ CC Instance │
    │  "Code Pro" │   │ "Researcher"│   │ "Writer"    │
    │  Tags: code,│   │ Tags: data, │   │ Tags: copy, │
    │  review,    │   │ analysis    │   │ edit, blog  │
    |  debug      |   |             |   |             |
    └─────────────┘   └─────────────┘   └─────────────┘
```

### How Routing Works

1. **Instance Registration.** Each Captain Claw instance connects to the BotPort hub via WebSocket and advertises its personas — named profiles with expertise tags (e.g., "code-review", "data-analysis", "copywriting").

2. **Concern Submission.** When an agent needs specialist help, it submits a "concern" with a description and expertise tags to the hub.

3. **Three-Tier Routing:**
   - **Tag Matching:** Find instances whose persona tags overlap with the concern's tags (threshold: 50%+ overlap).
   - **LLM-Assisted Routing:** If tag matching fails, an LLM evaluates the concern description against available persona profiles.
   - **Least-Loaded Fallback:** If no semantic match is found, route to the instance with the fewest active concerns.

4. **Execution & Result.** The target instance's agent processes the concern, optionally requests additional context via the hub, and returns the result.

5. **Follow-Ups.** The originating instance can send follow-up questions, creating a bidirectional conversation through the hub.

### Use Cases for BotPort

- **Team Specialization.** A development team runs three instances: one configured for code review (with access to the codebase), one for documentation (with writing-focused models), and one for DevOps (with infrastructure access). Tasks are automatically routed to the right specialist.

- **Model Optimization.** Route computationally expensive tasks (large code analysis) to an instance running a powerful model, while keeping simple queries on a fast, cheap model.

- **Geographic Distribution.** Instances in different locations can share workloads while keeping data processing local to each region.

- **Scalable Automation.** Add capacity by spinning up new instances. BotPort automatically includes them in routing decisions.

---

## Technical Specifications

### System Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.11 or higher |
| **OS** | macOS, Linux, Windows, Android (Termux) |
| **Database** | SQLite (embedded, no external DB required) |
| **API Keys** | At least one provider key, or none with Ollama |
| **Memory** | ~200MB base + model-dependent |
| **Network** | Required only for cloud LLM providers |

### Entry Points

| Command | Purpose |
|---------|---------|
| `captain-claw-web` | Web UI server (default: http://127.0.0.1:23080) |
| `captain-claw` | Interactive terminal |
| `captain-claw --tui` | Terminal UI mode |
| `captain-claw-orchestrate` | Headless orchestrator for automation |
| `botport` | BotPort agent-to-agent routing hub |

### Configuration

YAML-driven with 23 configuration sections covering models, context management, memory, tools, skills, guards, sessions, workspace settings, UI preferences, execution queue, logging, remote integrations (Telegram, Slack, Discord), web server settings, Google OAuth, orchestrator parameters, BotPort client settings, and scale processing.

Load precedence: `./config.yaml` > `~/.captain-claw/config.yaml` > environment variables > `.env` file > built-in defaults.

### Database Architecture

- **sessions.db** — Session history, messages, cron jobs, todos, contacts, scripts, APIs
- **memory.db** — Semantic memory index (chunks, embeddings, retrieval metadata)
- **datastore.db** — User-managed relational tables with protection rules

All databases are SQLite — no external database server required.

---

## Competitive Positioning

### What Makes Captain Claw Different

| Dimension | Typical AI Agents | Captain Claw |
|-----------|-------------------|-------------|
| **Model Support** | Single provider | 4 providers simultaneously, per-session switching |
| **Persistence** | Conversation lost on close | Named sessions, cross-session memory, persistent data |
| **Privacy** | Cloud-only | Runs fully local with Ollama, no data leaves your machine |
| **Safety** | None or basic | Three-layer guard system with approval workflows |
| **Tools** | 5-10 | 28 built-in, covering files, web, docs, media, Google, email, data |
| **Multi-Agent** | Not available | BotPort federated routing hub |
| **Document Processing** | Limited | PDF, DOCX, XLSX, PPTX extraction + chunked processing for any model size |
| **Deployment** | SaaS only | CLI, Web UI, Desktop App, Mobile (Termux), Headless |
| **Cost** | Subscription | Open source (MIT), pay only for API usage |
| **Extensibility** | Closed | OpenClaw-compatible skills, plugin system, GitHub install |

### Core Value Propositions

1. **"Use every model, not just one."** — The only open-source agent that treats multi-model routing as a first-class feature, with per-session model assignment.

2. **"Your agent remembers everything."** — Persistent sessions, cross-session memory, semantic search, and deep memory archives mean the agent builds knowledge over time.

3. **"Safety without sacrifice."** — Three-layer guard system adds production-grade safety without limiting the agent's capabilities.

4. **"Run it your way."** — Local-first, open-source, deployable as CLI, web app, desktop app, or mobile agent. No cloud dependency required.

5. **"Agents that collaborate."** — BotPort enables specialist networks where agents delegate to the best-qualified instance automatically.

---

## Target Audiences

### Primary

- **Office Workers & Administrative Professionals** — Email management (Gmail integration), calendar scheduling (Google Calendar), document processing (PDF/Word/Excel/PowerPoint extraction and summarization), contact management, and daily task tracking with persistent cross-session todos.
- **Small Business Owners & Managers** — All-in-one assistant for handling correspondence, organizing data in the built-in relational datastore, generating reports from spreadsheets, managing Google Drive files, scheduling, and automating repetitive administrative workflows via cron.
- **Freelancers & Consultants** — Client communication (email + Telegram/Slack/Discord), invoice and document handling, web research, proposal drafting with personality-tailored tone per client, and project tracking across persistent sessions.
- **Executive Assistants & Operations Staff** — Multi-platform communication management, meeting scheduling via Google Calendar, document preparation and extraction, contact book maintenance, and delegating specialized tasks to other agents via BotPort.

### Secondary

- **Software Developers** — Code analysis, refactoring, testing, debugging with multi-model flexibility and persistent project context.
- **DevOps & SRE Engineers** — Infrastructure automation, log analysis, incident response with safety guards and cron scheduling.
- **Data & Research Teams** — Batch document processing, data extraction, analysis with chunked processing and datastore capabilities.
- **Content Teams** — Summarization, translation, report generation across document formats with personality-based style adaptation.
- **System Administrators** — Local-only deployment with Ollama for air-gapped or privacy-sensitive environments.
- **AI Enthusiasts & Builders** — Extensible platform for building custom agent workflows, skills, and multi-agent networks.

---

## Traction & Stats

- **Version:** 0.3.1.2 (active development)
- **Codebase:** 66+ Python modules, 4,000+ lines in core agent mixins, 65 externalized prompt templates
- **Latest Release:** v0.3.1.2 (March 21, 2026) — 43 files changed, 4,035 insertions
- **License:** MIT — fully open source, commercially usable
- **Package:** Available on PyPI (`pip install captain-claw`)
- **Platforms:** macOS, Linux, Windows, Android (Termux)
- **Distribution:** Python package, standalone binaries (PyInstaller), desktop app (Electron)

---

## Key Messages

### One-Liner
> Captain Claw is an open-source AI agent that runs locally, supports GPT + Claude + Gemini + Ollama simultaneously, and gets real work done with 28 built-in tools and production safety guards.

### Elevator Pitch (30 seconds)
> Most AI tools lock you into one model and forget everything between sessions. Captain Claw is different — it's an open-source agent framework where you mix GPT, Claude, Gemini, and local models in a single runtime. Each session keeps its own model, history, and context. It ships with 28 tools for coding, research, document processing, email, and more. Safety guards review every operation before it executes. And with BotPort, your agents can delegate tasks to specialist instances across a network. It runs locally, it's MIT-licensed, and you own everything.

---

### Topics to Focus On
- The story of why multi-model matters (vendor lock-in, cost optimization, capability matching).
- Open source AI agents vs. closed SaaS — the tradeoffs.
- BotPort and the future of multi-agent collaboration.
- Running AI locally — privacy, speed, and the Ollama ecosystem.
- Safety in autonomous AI — why guards matter before agents go mainstream.

---

## Quotes for Attribution

> "I built Captain Claw because I was tired of choosing between AI providers. The best tool for coding isn't always the best tool for writing — so why should your agent be locked to one model?"

> "BotPort is my answer to the question nobody else is asking: what happens when AI agents need to collaborate? I think multi-agent networks are the next frontier, and I'm building the open-source infrastructure for it."

> "Safety guards aren't a feature I added reluctantly. They're core to the architecture. If you're going to let an agent execute shell commands and send emails, you need a systematic way to review what it's doing."

> "The chunked processing pipeline means you don't need a $200/month API tier to process large documents. A free local model running on your laptop can handle a 100-page PDF — it just takes a few more minutes."

---

## Links & Resources

- **GitHub:** github.com/kstevica/captain-claw
- **PyPI:** `pip install captain-claw`
- **Documentation:** USAGE.md (comprehensive reference — 111.7 KB)
- **License:** MIT
- **Author:** Stevica Kuharski (kstevica@gmail.com)

