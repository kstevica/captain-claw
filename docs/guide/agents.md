# Inside an agent — models, tools and guards

[← README](../../README.md) · [All guides](README.md)

## Multi-model support

Mix providers freely — each session independently selects its model.

| Provider | Models |
|---|---|
| OpenAI (API key) | GPT-6 family (per-tier reasoning effort), GPT-5.4, GPT-5.4-mini, GPT-5.4-nano, o3, o4-mini, gpt-image-1.5 |
| OpenAI (Sign in with ChatGPT) | `gpt-5`, `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`, `gpt-5.2-codex`, `gpt-5.3-codex` — billed against your ChatGPT plan, no API key |
| Anthropic | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 (with prompt caching) |
| Google | Gemini 3.1 Pro/Flash, Gemini 2.5 Pro/Flash (API key or OAuth/Vertex) |
| Google subscription | Gemini via local Antigravity CLI; text only, no API key or credit overages. [Setup and limits](../antigravity-cli.md) |
| DeepSeek | Chat and reasoner models, thinking mode included (orphan tool results and pinned temperatures self-heal since 0.8.0) |
| Ollama | Any local model |
| LiteRT (on-device) | `.litertlm` Gemma models running locally via an isolated subprocess worker |
| OpenRouter | 200+ models via meta-router |

## 65 built-in tools

Shell, file I/O, web fetch/search, browser automation, PDF/DOCX/XLSX/PPTX extraction, image generation (DALL-E), OCR, vision, TTS, STT, email (SMTP/Mailgun/SendGrid), Google Workspace (Drive, Docs, Sheets, Slides, Gmail, Calendar), Microsoft 365 (Outlook mail, Calendar, OneDrive), Jev calibrated judgments (TypeSafe AI), Soniox speech in and out (`audio`), public file links, topic switching by asking (`sessions`), a read-only Telegram inbox, the StopWatch's background picture, bill payments (opt-in), WhatsApp file delivery, intentions (proactive future actions), desktop automation, screen capture with voice commands, persistent cross-session memory (todos, contacts, scripts, APIs, playbooks), datastore (SQLite tables with protection rules), deep memory (Typesense), Basna (read and start multi-agent ensemble runs), personality system, cron scheduling, BotPort fleet discovery, and Termux (Android).

New in 0.8.4:

| Tool | What it does |
|---|---|
| `knowledge` | Searches and reads the documents the owner gave the agent, with no model: `search` (hybrid; exact codes first, quoted phrases required), `section` (a section whole, across parts), `outline`, `list`, and `add` to keep a file from chat. See [Onboarding](flight-deck.md#onboarding). Shows on every Flight Deck agent; shared-agent members never get it. |
| `visualize` | HTML in, the page and a PNG of it out, with pinned Chart.js, ECharts, Plotly, D3, Mermaid or Leaflet. The PNG reaches the chat and the channels and goes into a `.docx`. Script errors, failed loads and blank pictures come back to fix. The page never touches the host. Always on. |

New in 0.8.3:

| Tool | What it does |
|---|---|
| `docx` | Creates Word files from Markdown, reads them, and edits them as **real Word tracked changes**; lists, accepts and rejects changes and comments. A user's upload is never changed: the redline is a new file. Shared-agent members can use it, and their changes carry their own name. `write` and `edit` refuse `.docx` and point here. |
| `self_check` | Tests the agent's live connections and models and reports its server. Read-only. |
| `reasoning_effort` | Reads or changes the agent's own reasoning effort when you ask, from any channel; it changes the same setting you would. Owner-only. |
| `activity` | The agent's own tool calls, read back from the [tool-call register](flight-deck.md#activity-and-audit): "what did you do overnight?" Only its own calls from its owner's turns; never members' calls or file and mail bodies. |
| `linked_system` | Messages a system linked to the agent; see [agent links](botport-and-mcp.md#agent-links--connect-any-system-to-an-agent). |
| `whatsapp_web` | Reads, and with your consent sends to, your own WhatsApp; see [WhatsApp Web](devices-and-channels.md#whatsapp-web--your-own-whatsapp). Hidden until you link it. |

See [USAGE.md](../../USAGE.md#tools-reference) for the full reference.

## Safety guards

Three layers of protection that run before, during, and after agent operations:

- **Input guards** — Validate user intent before the LLM sees it
- **Script guards** — AST-level analysis of generated code before execution
- **Output guards** — Validate tool results for hallucinations and safety
- **Action honesty** (0.8.0) — a fabrication guard rejects replies that claim an email was sent or a calendar event changed when no tool did it, and automated turns may write email only when their own instructions ask for it
- **Every call on the record** (0.8.3) — each tool call lands in the deck's append-only, HMAC-chained [tool-call register](flight-deck.md#activity-and-audit), refused calls included; self-reflection reads a digest of repeated failures and repeated calls from it
- **Held to its own files** (0.8.4) — on a deck with sign-in, file tools are checked by real path against the agent's folder, its workspace, its owner's VFS and the call's shared folders; a non-admin's agent also loses the tools that run code or drive the host. See the [checklist](../team-deployment.md#what-the-launch-hardening-changed-08x)
- **Checked on hand-over** (0.8.4) — when you hand an agent a task on the [Taskboard](flight-deck.md#the-taskboard), its reply is compared with the tool-call register, Gmail and Calendar before anyone believes it

Guards support two modes: `stop_suspicious` (block automatically) or `ask_for_approval` (prompt the user).

## The agent web UI

Chat, Computer (retro-themed research workspace with 14 themes), monitor pane, instruction editor, command palette, persona selector, datastore browser, deep memory dashboard, insights browser, nervous system browser, Brain Graph 3D visualization, reflections dashboard, personality editor, playbook editor, and LLM usage analytics.

**Computer** — A standalone research workspace at `/computer` with themed visual generation, exploration trees, folder browser (local + Google Drive), file attachments, PDF export, and public mode with BYOK (Bring Your Own Key).
