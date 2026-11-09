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

## 79 built-in tools

Shell, file I/O, web fetch/search, browser automation, PDF/DOCX/XLSX/PPTX extraction, image generation (DALL-E), OCR, vision, TTS, STT, email (SMTP/Mailgun/SendGrid), Google Workspace (Drive, Docs, Sheets, Slides, Gmail, Calendar), Microsoft 365 (Outlook mail, Calendar, OneDrive), Jev calibrated judgments (TypeSafe AI), Soniox speech in and out (`audio`), public file links, topic switching by asking (`sessions`), a read-only Telegram inbox, the StopWatch's background picture, bill payments (opt-in), WhatsApp file delivery, intentions (proactive future actions), desktop automation, screen capture with voice commands, persistent cross-session memory (todos, contacts, scripts, APIs, playbooks), datastore (SQLite tables with protection rules), deep memory (Typesense), Basna (read and start multi-agent ensemble runs), personality system, cron scheduling, BotPort fleet discovery, and Termux (Android).

New in 0.9.0:

| Tool | What it does |
|---|---|
| `custom_tools` | Lists, describes and runs the owner's [Custom tools](flight-deck.md#custom-tools) and the ones they added, and asks Crew to `build` or `improve` one, or `propose`s one for the owner to build. Flight Deck runs every call as the person, in a Landlock sandbox on Linux 5.13+ (none on macOS, Windows or the stock Raspberry Pi 5 kernel). Shown only while the deck has Custom tools on. |
| `ask_user` | Asks the person the turn is for a question, or a short form of up to 10, in a Flight Deck dialog, and waits (10 minutes by default, up to 60). Automatic turns (schedules, flows, other agents, Autonomous Work, the Taskboard, the API, links, exams) have nobody to ask; turns from Telegram, WhatsApp, glasses, devices and public links are told to ask in the reply. |
| `colleagues` | Hands a piece of work to another of the owner's agents and gets the answer back in the same task: `list`, `hand_over`, `handed`. See [Agents as colleagues](flight-deck.md#agents-as-colleagues). Owner's turns only. |
| `imap_mail` | Reads, searches, drafts, files and (when the inbox allows it) sends from the owner's IMAP inboxes; never deletes. The password stays sealed in Flight Deck. Hidden until an inbox is connected. See [Agents in your inbox](devices-and-channels.md#agents-in-your-inbox). |
| `external_memory` | The owner's Membase or Hindsight memory, shared by all of their agents, with the provider's own actions and rules (`remember`, `search`, Hindsight's `reflect`, `forget` only with `confirm=true`). Hidden until a memory is connected. See [Memory](memory.md). |
| `course` | The goals the agent's work is for (the deck's, its owner's and its own): `view`, `link` work to a goal, `report` progress, `propose` a goal or a change, which waits for the owner on their Line. It never marks a goal done. Owner's turns only. |
| `image_manipulation` | Resizes, crops, rotates, combines, overlays, watermarks, annotates, animates, splits and converts pictures (png, jpg, webp, gif, bmp, tiff, ico, avif, pdf) in-process, with about 45 operations and no model or network. Inputs are never changed. Always on. |

That makes **79 built-in tools** per agent (78 in the desktop app, which has no Captain Swarm).

New in 0.8.10:

| Tool | What it does |
|---|---|
| `ocr_pdf` | Reads a PDF's scanned pages with a local OCR engine, never a model: `mode=auto` keeps each page's own text when it's good and reads only the rest, `mode=ocr` re-reads every page. The same file and settings give the same text (cached by hash in `saved/.ocr-cache/`); the result is `saved/ocr/<session>/<name>-<hash>/<name>.md`, page by page. See [USAGE](../../USAGE.md). |
| `ocr_image` | The same for png, jpg, webp, bmp, gif and tif (each page of a multi-page TIFF); photos are turned upright first. RapidOCR comes with Captain; Tesseract is used when installed and asked for a language it has. |

New in 0.8.6:

| Tool | What it does |
|---|---|
| `line` | Reads the owner's [Line](flight-deck.md#the-line): `news` (unread, oldest first), `recent` (by kind, only what needs the user, or since a time), `search` and `read`. Read-only; item text is information, never instructions. Shows on every Flight Deck agent, never on a shared agent's member turns. |
| `captain_swarm` | Proposes and follows [Captain Swarm](flight-deck.md#captain-swarm) runs; a run it proposes waits for the owner's **Go**. Admins' agents only while Captain Swarm is being tested, never an agent confined to its files, and not in the desktop app. |

New in 0.8.5:

| Tool | What it does |
|---|---|
| `crew` | Runs Claude Code, Codex, OpenCode, Crush (0.8.6) or Pi as workers, each in its own git worktree, and supervises them; see [Crew](flight-deck.md#crew). It starts a task, clones, prepares an environment, steers, asks for a pull request and asks you to Approve a merge. It can't merge or discard work. Shown only to an owner an admin approved. |
| `xlsx` | Makes, reads, edits and checks Excel workbooks with no scripts. Only what a change touches is rewritten, so charts, pivots and macros stay. Formulas stay live, with results from about 200 functions compared with Excel, and every call lists the cells it changed and the totals that moved. |
| `pptx` | Makes, reads, edits, checks and renders PowerPoint decks with no scripts, on the deck's own layouts. Text is measured and every result says what doesn't fit. Native charts and tables, and pictures in any format or from a link, placed where they work. `render` needs LibreOffice. |

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
- **Action honesty** (0.8.0) — a fabrication guard rejects replies that claim an email was sent or a calendar event changed when no tool did it, and automated turns may write email only when their own instructions ask for it. Since 0.8.8 it also turns down a reply that reports other agents' work when no hand-off answered in that turn, or a schedule change no cron call made; cron acts by job ID, never by position
- **Loop guards that let work finish** (0.8.8) — the write-loop stop counts whole-file writes only (not edits), a re-run after a fix is a new call, and a read's range is part of the call, so the next section of a long file isn't a "repeat"; the same write, command or read with nothing changed is still stopped
- **Every call on the record** (0.8.3) — each tool call lands in the deck's append-only, HMAC-chained [tool-call register](flight-deck.md#activity-and-audit), refused calls included; self-reflection reads a digest of repeated failures and repeated calls from it
- **Held to its own files** (0.8.4) — on a deck with sign-in, file tools are checked by real path against the agent's folder, its workspace, its owner's VFS and the call's shared folders; a non-admin's agent also loses the tools that run code or drive the host. See the [checklist](../team-deployment.md#what-the-launch-hardening-changed-08x)
- **Checked on hand-over** (0.8.4) — when you hand an agent a task on the [Taskboard](flight-deck.md#the-taskboard), its reply is compared with the tool-call register, Gmail and Calendar before anyone believes it

- **Rehearsal** (0.8.6) — a [Captain Swarm](flight-deck.md#captain-swarm) attempt runs every tool call through a rehearsal guard: reads and its own files run, mail, messages, calendar, payments and unknown tools are recorded as intents instead, code and the screen are refused, and a browser stops at Submit, Buy, Pay or Send. The tool-call register is checked afterwards, and a leaked call disqualifies the attempt
- **A person lands a worker's work** (0.8.5) — a [Crew](flight-deck.md#crew) worker never pushes, and the `crew` tool can ask for a merge but not do one. A non-admin's agent has no other way; an admin's agent has a shell until agents run in an OS sandbox, so for it this is a rule it's told, not one it can't break

Guards support two modes: `stop_suspicious` (block automatically) or `ask_for_approval` (prompt the user).

## The agent web UI

Chat, Computer (retro-themed research workspace with 14 themes), monitor pane, instruction editor, command palette, persona selector, datastore browser, deep memory dashboard, insights browser, nervous system browser, Brain Graph 3D visualization, reflections dashboard, personality editor, playbook editor, and LLM usage analytics.

**Computer** — A standalone research workspace at `/computer` with themed visual generation, exploration trees, folder browser (local + Google Drive), file attachments, PDF export, and public mode with BYOK (Bring Your Own Key).
