# What's new — release history

[← README](../../README.md) · [All guides](README.md)

Every release from the newest back to 0.4.29. The full notes for each, with upgrade steps, are in [release-notes/](../../release-notes/).

**Jump to:** [0.8.4](#whats-new-in-084) · [0.8.3](#whats-new-in-083) · [0.8.2](#whats-new-in-082) · [0.8.1](#whats-new-in-081) · [0.8.0](#whats-new-in-080) · [0.7.8](#whats-new-in-078) · [0.7.6](#whats-new-in-076) · [0.7.5](#whats-new-in-075) · [0.7.4](#whats-new-in-074) · [0.7.3](#whats-new-in-073) · [0.7.2](#whats-new-in-072) · [0.7.1](#whats-new-in-071) · [0.7.0](#whats-new-in-070) · [0.6.5](#whats-new-in-065) · [0.6.4](#whats-new-in-064) · [0.6.3](#whats-new-in-063) · [0.6.2](#whats-new-in-062) · [0.6.1](#whats-new-in-061) · [0.6.0](#whats-new-in-060) · [0.5.7](#whats-new-in-057) · [0.5.6](#whats-new-in-056) · [0.5.5](#whats-new-in-055) · [0.5.4](#whats-new-in-054) · [0.5.3](#whats-new-in-053) · [0.5.2](#whats-new-in-052) · [0.5.1](#whats-new-in-051) · [0.5.0](#whats-new-in-050) · [0.4.33](#whats-new-in-0433) · [0.4.32](#whats-new-in-0432) · [0.4.31](#whats-new-in-0431) · [0.4.30](#whats-new-in-0430) · [0.4.29](#whats-new-in-0429)

## What's New in 0.8.4

**Onboarded, On the Board.** Teach an agent your work and test it, see every task in one place, and run a deck for people who don't share a machine. 71 merged PRs in the three days after 0.8.3. **Restart Flight Deck, then every agent, and on a team deck [read this first](../../release-notes/RELEASE_NOTES_0.8.4.md#before-you-upgrade-a-team-deck):** non-admins' agents lose the tools that run code.

- **Onboarding** (⋯ → Onboarding: Overview, Lessons, Practice, Materials, Exams).
  - **Way of working:** up to 12 documents, each reaching the agent whole as a knowledge base at no token cost; a reading budget (Quick 60k, Standard 250k, Deep 1M tokens) with an optional focus; a background read on the agent's own model that survives a restart; a review of the rules, skills and facts before they land. Spreadsheets also become datastore tables.
  - **Lessons:** *Always in mind* stays in the instructions; *When it matters* reaches the agent with the messages it bears on. Save one from chat.
  - **Exams:** 10 to 120 questions from the documents, asked through the real chat on a private lane, graded point by point, with why a wrong answer went wrong. **Practice** on real tasks, with nothing sent.
  - **For a team:** an admin's base layer, members' suggestions, and **Pass it on** as an archetype.
- **The Taskboard.**
  - Everything agents were asked and sent, on any channel: *Needs me*, *Waiting on others*, *Agents working*, *Done*, with Gmail's keys, search, merge and split, and CSV/JSONL export.
  - The agent's model and Jev read the work; **Why is this here?** shows every decision; a reply brings the same task back.
  - Next steps wait for your approval, and email only ever becomes a draft.
  - **Hand a task to an agent**, and the board checks the tool-call register, Gmail and Calendar to see that it really did it.
  - Your own inboxes are opt-in, kept 30 days and deleted when switched off.
- **Security on sign-in decks.** Agents stay in their own files; non-admins' agents don't run code or drive the host, and their web tools take public addresses only; hosted apps and host folders are admin-only; team keys are masked for non-admins and go only to the team's endpoints. Containment, not an OS sandbox. `FD_AGENT_CONFINE=off` turns it off.
- **Your own AI keys** (Settings): Anthropic, OpenAI, Gemini, OpenRouter or xAI, checked and sealed, with no daily limits. Existing agents follow.
- **Profile**: Personal · Deck-wide · Password, and **Fill it in for me**, a researcher that reads your documents and links and drafts your texts for review.
- **Two new tools**: `knowledge` (search and read the owner's documents, with no model) and `visualize` (HTML in, the page and a PNG out): **65 built-in tools** in all.
- **The chat**: pictures reach the chat and the channels, a **Files** menu in the composer, **Insert now** for a queued message, and compact cards for automated results.
- **The kiosk**: the agent's **⋯** and **Settings** are cards of tiles, agent settings have two tabs, **Instructions, insights & skills** in one dialog, and waiting proposals top the bell.
- **Devices**: the **Waveshare ePaper 1.54**, with and without touch; blue for an agent asleep and red for one stopped; firmware **0.19.10** / PaperColor B **1.0.22**.
- **Fixes**: a WhatsApp Web ban or switch keeps the history; insights and intuitions drop exact and reworded repeats; openpyxl workbooks read properly; Ollama gets its tool calls back, and `-cloud` models skip the local cap.
- **And**: a new Captain icon, and an optional [deed.page MCP example](../../examples/mcp/deed-page.md).

See [RELEASE_NOTES_0.8.4.md](../../release-notes/RELEASE_NOTES_0.8.4.md).

## What's New in 0.8.3

**Plugged In, On the Record.** Any system can talk to an agent, agents can read your own WhatsApp, and every tool call they make is written down. **Restart Flight Deck, then every agent** — see the [upgrade steps](../../release-notes/RELEASE_NOTES_0.8.3.md#upgrading).

- **Agent links.**
  - Connections → **Systems** gives a CRM, shop or monitor its own `cc_link_…` key to one of your agents.
  - It talks over a WebSocket (`/link/ws`), plain HTTP (`POST /link/messages`, `?wait=` for the answer) or signed webhooks, with files both ways.
  - Or Flight Deck dials out to an orchestrator's `wss://` address, so an agent behind a router needs no public address.
  - The agent writes first with the new `linked_system` tool; link turns run in a session of their own.
- **WhatsApp Web (experimental, opt-in).**
  - Link *your own* WhatsApp as a linked device; your agents read chats, files and voice notes with the new `whatsapp_web` tool.
  - Sending is off until you turn it on: whitelisted existing chats only, daily and per-chat limits, your signature on every message.
  - Auto-replies in the chats you tick, when someone calls the agent by name, with an activity log of every decision.
  - It uses an unofficial client, which is against WhatsApp's terms; an admin must allow it.
- **The tool-call register.**
  - Every agent tool call is recorded (tool, masked parameters, what started it, outcome) in an append-only, HMAC-chained log; refused calls too.
  - **Activity** for users, **Admin console → Audit** for admins, with CSV/JSON-lines export and 90-day retention by default.
  - Agents read their own calls back with the new `activity` tool, and self-reflection gets a digest of repeated failures.
- **Nudges and proposals reach you** on WhatsApp, Telegram, your devices and the bell, with **Approve / Reject** in the bell and a ⤢ bigger view; a **Notifications** card picks the channels.
- **Idle agents sleep** (admin, off by default) and wake five minutes before their scheduled work, or when anything needs them.
- **The kiosk** gets **Shared folders** (the VFS beside Library and Data), **OneDrive** mounts beside Google Drive, fullscreen library dialogs, and **⌘/Ctrl+Enter** to queue a message. The tab icon follows the deck's Branding logo.
- **Six new tools** — `docx` (Word files with **real tracked changes**, checked in Microsoft Word), `self_check`, `reasoning_effort` (from any channel), `activity`, `linked_system` and `whatsapp_web`: **63 built-in tools** in all.
- **Devices** — firmware **0.19.1** / PaperColor B **1.0.21**: a beep, a banner and a count badge when something waits, and a link to the deck that renews itself.
- **Fixes** — a removed agent's kept folder never passes to another user's new agent; the WhatsApp Cloud API logs Meta's error code for a refused send; the mail repeat check tells a recurring report to date its subject.

See [RELEASE_NOTES_0.8.3.md](../../release-notes/RELEASE_NOTES_0.8.3.md).

## What's New in 0.8.2

**Decks That Work Together.** Flight Decks connect through **BotPort** and lend each other their agents. **Upgrade the hub first** (decks need protocol 2) — see the [upgrade steps](../../release-notes/RELEASE_NOTES_0.8.2.md#upgrading).

- **BotPort, rebuilt around Flight Decks.**
  - An agent on one deck consults or delegates to `Agent@deck` on another, with live progress.
  - Every deck dials out, so a Raspberry Pi at home joins as easily as a VPS.
  - Answers wait for a requester that dropped; a worker that dropped gets time to come back.
  - Remote requests run in their own lane: no email sends, no Flows, no fan-out, plus the admin's deny list.
- **Add a deck from the hub's dashboard, no restart.**
  - The new **Decks** page makes a key and a one-time **join code**; Flight Deck's *Admin console → BotPort → Join code* takes it in one paste.
  - Rotate or remove a key and it takes effect at once.
  - Every deck has a display name and a description that the whole network, and your agents, see.
- **A dashboard for the wall.**
  - A live map where every request flies from deck to hub to agent and back.
  - KPIs, an in-flight rail, a ticker.
  - **Kiosk mode** (`K`), a Requests page, a sign-in screen, a phone layout.
- **Everyone shares.** Users put the agents they own on the network from the agent's ⋯ (kiosk too), after an honest "Share on network?" dialog. Admins keep a switch and a Stop button for every share.
- **Hardening.**
  - Agent tokens go only to that agent's own address.
  - The hub has an admin token, a DNS-rebinding guard, constant-time key checks and a failure limiter.
  - Fixed: a swarm path traversal and a dashboard XSS.
- **Devices** play long answers through (no more 66-second ceiling; Flight Deck speaks up to about 1,500 characters and shows the rest); the PaperColor does an hourly white pass to keep its six inks clean.
- **Research reports with pictures.** `web_fetch` keeps the relevant pictures a page shows, and reports embed them (.md, .html, .docx).

See [RELEASE_NOTES_0.8.2.md](../../release-notes/RELEASE_NOTES_0.8.2.md) and [docs/botport-deployment.md](../botport-deployment.md).

## What's New in 0.8.1

**A kiosk for everyone · devices that listen.** 121 merged PRs in the nine days after 0.8.0. **Before you upgrade a team deck:** daily plan limits are on by default (free: 30 messages and 300k tokens a UTC day; admins exempt) and MCP servers become private to whoever added them — see the [upgrade checklist](../../release-notes/RELEASE_NOTES_0.8.1.md#upgrading).

- **The kiosk, a complete place to work** — a Getting started guide with real screenshots; a bigger chat that greets you with the agent's **skill cards** ("What can I do?", up to 22 per agent, some asking for a file or a picture first); a **reasoning-effort** picker with exactly the levels the agent's model has; one **timeline** per agent with topics the agent names, a Topics menu and a `sessions` tool; a **File library** (every agent file summarised, categorised, searchable and previewed — pictures described, recordings transcribed) and a **Data library** (datastore tables profiled, summarised, browsable, exportable); **Scheduled tasks** delivering to chat, email or Telegram; a ⋯ menu per agent (Info, Instructions, Settings for model, key, vision, thinking and effort, Remove); a new agent picker; a readable message queue; a phone layout.
- **Running a deck for people** — an **Admin console** at `<deck>/console`; a Users page with **invitations by email** and a **waitlist** that emails approved applicants; admin-set and self-service passwords; branding; **daily plan limits** (prompt-cache reads don't count) with free resets; MCP servers private to their owner and MCP OAuth with Dynamic Client Registration; **public file links**; a desktop app that opens on the console or the kiosk.
- **Devices** — the **StopWatch** and the **PaperColor** join the StickS3 (below).
- **One conversation across every channel** — every message records its source and the model reads `[via WhatsApp] …`; the owner's Telegram joins the main conversation; Telegram gains a team bot, live narration, voice and video, a read-only **inbox** and **groups**; Soniox keys live in Flight Deck and the `audio` tool speaks and transcribes.
- **Agents** — **native vision** (pictures go straight to a model that sees them), an always-on everyday toolset, Jev extending turns that are still making progress, and chats that ride out a deck restart.
- **Payments**, opt-in and off by default — barcodes for your bank app, Revolut drafts, pay-by-bank links, one-time cards, bank-statement matching, reminders and recurring bills. Captain never moves money.

#### Three devices, one firmware

| | StickS3 | StopWatch | PaperColor |
|---|---|---|---|
| screen | 240×135 LCD | 466×466 round AMOLED, touch | 4" E Ink Spectra 6, six inks |
| talk | front button | yellow button, or the on-screen record button | A (B build) / C (A build) |
| extras | 18 animated characters | clock face, raise to wake, vibration, background picture, info page | two status lights, room sensor, microSD notes, a dashboard every five minutes |
| build | `pio run -e sticks3` | `pio run -e stopwatch` | `pio run -e papercolor_b` (or `papercolor`) |

- **Signed over-the-air updates** from your own deck (`tools/publish.py`), rolled back if a build can't reach the deck within 90 s.
- **Press to talk, press again — or just stop talking** (2.5 s of quiet sends it), with blips and chimes; the PaperColor records straight from sleep.
- **Answers as they're written** — spoken sentence by sentence while the model is still writing; the lit sentence and the pages follow the voice; pause, replay, stop.
- **Offline voice notes** (ADPCM in flash, or on the PaperColor's microSD) sent as one turn once the deck is back, and a **note-taking mode** for a run of notes.
- **Context with every question** — battery and time left, Wi-Fi, motion, room temperature and humidity, local time, as a `[Context: …]` line the model reads.
- **"Volume up", "beep off"** handled by the deck, never sent to the agent; a turn survives a dropped link.

See [RELEASE_NOTES_0.8.1.md](../../release-notes/RELEASE_NOTES_0.8.1.md).

## What's New in 0.8.0

**Flight Deck for a team · agents in your inbox, glasses and pocket.** The biggest release since Flight Deck — 75 merged PRs about putting agents in front of other people. **Some defaults change, mostly for security:** a deck reached through its own domain must set `FD_ALLOWED_HOSTS` or `FD_PUBLIC_URL` before upgrading; restart Flight Deck, then every agent.

- **Flight Deck for a team** — every route knows who is calling (owner-scoped proxies, verified agent identity, a Host/Origin guard against DNS rebinding), per-user Google and Microsoft accounts, closed registration with **Admin → Users** to add people and administer their agents, team tier sets that share keys server-side, **shared agents** (one agent, several people, each with their own Google, files and context packs), an owner profile in every prompt, and a **kiosk mode** (`--simple-chat`) on a new chat-first simple layout.
- **Agents in your inbox** — **Microsoft 365** tools (Outlook mail, Calendar, OneDrive), opt-in audited **Gmail sending**, attachments in and out, mailbox filing, Google Sheets and Docs edited in place, an **agentic signature**, and fully autonomous **topic auto-replies** (off by default, heavily guarded). The `gws` CLI wrapper is retired.
- **Agents that don't make things up** — no email nobody asked for (mail is human-only in Autonomous Work and automated turns), a **fabrication guard** that catches claimed sends, calendar changes and invented IDs and asks for a true reply, and a Jev check that suggests `/new` when a reply comes out garbled.
- **More surfaces** — WhatsApp files both ways, reactions and `/new`; **`/new` on Telegram**; a **Flight Deck HUD** for Meta Ray-Ban Display at `/hud` (about 150 KB on the wire, paired with a code at `/hud/pair`); the push-to-talk **M5Stack StickS3 and StopWatch** voice clients (below); and an MCP server at `/fd/mcp-server` so **Claude can drive your fleet** (`list_agents`, `send_task`, `get_result`, `cancel_task`, `jev_ask`).
- **Long work that finishes** — **Bat**, a new mode that keeps going until an independent judge says it's done; Vatra workers without a time cap and with an in-turn context governor; long-form deliverables assembled by code with a done gate; a crowded-session **context engine** for weak models; deep memory that finally searches by meaning; and Google Drive folders mounted as VFS projects.

#### In your pocket: M5Stack StickS3 and StopWatch

Hold the talk button, talk, release. Flight Deck transcribes it (Soniox), sends it to one of your agents, and the device shows the reply, reads it aloud, and shows any picture the agent made — with the agent's live activity while it works. Server side: `captain_claw/flight_deck/device_bridge.py` at `/device/ws`. One firmware tree ([`firmware/device/`](../../firmware/device/README.md), PlatformIO) builds both devices:

| | StickS3 | StopWatch |
|---|---|---|
| screen | 240×135 LCD | 466×466 round AMOLED — a ring shows red while it listens, amber while the agent works (a tick per tool step), cyan while it speaks |
| talk / menu | front button / side button | yellow button / the second button |
| extras | — | vibration, clock, touch |
| build | `pio run -e sticks3 -t upload` | `pio run -e stopwatch -t upload` |

- **Pick the agent from the device**, choose the spoken-reply language, volume (25–100%) and a +6/+10 dB speech boost, text size, screen-off and power-off timeouts, male or female voice.
- **Replay** the last answer from memory (up to ~66 s kept; a cut-off clip is re-sent by the deck) — double-click the side button or Menu → *Replay answer*.
- **Icon mode** — an animated character that listens, thinks and speaks instead of text: Captain, Bot, Bird, Car, Matrix, Orc, Elf, Dwarf, C64, Amiga, Cat, Dog, Dinosaur, Alien, Caveman, HAL 9000, KITT or Pac-Man.
- **Pictures and sound both ways** — agent-made JPEGs (chunked up to 120 KB, fitted to the round screen) and agent-made audio after the speech.
- **Signed over-the-air updates** — `python tools/publish.py sticks3 --deck https://<deck> --token cc_pat_…` builds, signs (ECDSA P-256) and publishes; devices take the offer from Menu → *Software update* or by themselves on USB, and a build that can't reach the deck within 90 s rolls back.
- **Setup from a hotspot** — scan the QR on screen, open `192.168.4.1`, enter up to five Wi-Fi networks, the deck address (`ws://<ip>:25080` or `wss://<domain>`) and an access token from Connections → *Agent access for Claude (MCP)*. No watch yet? `pio run -e preview` renders every screen on a Mac.

The deck needs a Soniox key: your own in **Connections → Soniox**, or the team key an admin shares in **Admin → Integrations** (`SONIOX_API_KEY` in the environment still works as a fallback). A deck behind a domain needs it in `FD_ALLOWED_HOSTS` or `FD_PUBLIC_URL`.

#### Flight Deck for a team, at a glance

[![The simple layout — agents with a power switch on the left, chat in the middle, the agent's files and datastore on the right](../screenshots/simple-layout.webp)](../team-deployment.md)

The chat-first **simple layout** (Cmd/Ctrl+\) with agents, chat, files and datastore — `flight-deck --simple-chat` turns it into a **kiosk** for non-admins. Before a second person logs in, set `FD_AUTH_ENABLED`, a fixed `FD_JWT_SECRET` and `FD_PUBLIC_URL` / `FD_ALLOWED_HOSTS`; the first account becomes the admin, **Admin → Users** adds the rest, and **Library → Publish to team** gives them models without pasting keys. The full checklist is [docs/team-deployment.md](../team-deployment.md).

**Upgrading:** a deck reached through its own domain must set `FD_ALLOWED_HOSTS` or `FD_PUBLIC_URL` first, then pull, restart Flight Deck, restart every agent, hard-refresh; reconnect Google once per user; publish your team tier set again. `gws` is retired, registration closes after the first admin, and chat bridges auto-send only files an agent made this turn. Everything else is in [RELEASE_NOTES_0.8.0.md](../../release-notes/RELEASE_NOTES_0.8.0.md).

## What's New in 0.7.8

**Lanes, queues that finish · Iskre shape their world.** Long work that runs unattended: one agent gains **three parallel lanes**, the task queue stops stranding itself and learns to **plan its own batches**, and the datastore tool stops turning one fumbled argument into thirty tool calls of hunting. Everything opt-in — **lane A *is* the agent you already have**.

- **Lanes A/B/C** — three rooms on one agent process, each with its own session, transcript, queue and busy flag, running at the same time. Three id ranges enrich in parallel against the same table. An omitted lane resolves to A, so WhatsApp, cron, botport and every existing client are untouched. A tab strip shows each lane's state and marks the one that produced output while you looked away.
- **A queue that finishes** — four ways it could stall are closed: a slash command that answered out of band, a trailing question read as "waiting for you", a give-up reply ticked off as success (it now **re-runs** the task — verbatim, then nudged to how it failed — and after three tries **stops rather than skipping**), and lanes silently opening in MANUAL. Each card reports what its run cost: times, elapsed, tools, tokens.
- **The Task Planner** — describe a repetitive job once, attach files, get a plan you read and edit before **anything** is queued. The model returns one **template plus a range**, never the messages, so every task is byte-identical except its range and no rule can drift between batch 3 and batch 19. Ranges come from the table's real MIN/MAX. Continue where the last sitting stopped, free.
- **The datastore tool answers a fumbled call** — a dropped `table` recovers or refuses with a filled-in retry, synonyms (`data`, `select`, `order`, …) are renamed instead of silently ignored, a SQL write comes back translated, and every way a model asks for a range now works. One repeated `where` key used to widen a 10-row read into **1.38M characters** of context.
- **Iskre shape their world** — beings craft and place objects, name their homes, break ground on impulse (a new *Impulse* genome stat), and hand work between their two brains on a visible work board. The keeper paints roads, grows the plot, and builds anywhere from the map or in first person.

See [RELEASE_NOTES_0.7.8.md](../../release-notes/RELEASE_NOTES_0.7.8.md).

## What's New in 0.7.6

**Iskra — living beings you raise.** A new **Beings** page introduces persistent digital beings with a **genome**, a **token wallet**, **drives**, and a **life loop** that ticks on its own heartbeat — you parent them through real developmental **stages** (infant → child → adolescent → adult). Additive and self-contained (a new `beings.db` auto-creates; needs a Flight Deck restart).

- **A being that lives** — conceive one with a **point-buy genome** (40 points across 7 attributes; presets or a roll), a daily-allowance **token wallet** conserved by construction (family-mint-only; zero balance halts cognition at the physics layer), and **drives + a homeostat**. A background **life loop** ticks it (SENSE → APPRAISE → DELIBERATE → ACT → DIGEST): it journals, tends a garden of files, makes and connects things, **dreams** nightly, and — unfed past the grace period — enters **torpor**, then dies. Mortality is economic, by design.
- **Honesty by construction (anti-theater)** — the ground truth for "did she make something" is the **git diff of her home**, never her narration. A claimed-but-unwritten file is caught and fed back as a reality check, and a **write completion gate** pushes her — in the same tick — to actually write what she claimed, or record honestly that she didn't. A being can't feel accomplished, advance, or pass a readiness check on words it didn't back with work.
- **Stages you parent + a readiness assessment** — the **Growth** tab scores her across **8 developmental domains** (Vitality, Honesty, Stability, Productivity, Coherence, Sense of self, Communication, Experience) — each a **real ledger variable** on a green/amber/red bar — with a verdict, a time-to-ready estimate, and a concrete recommendation. Get a **second opinion from another of your agents** (an independent read), and **keep opinions on record** — sealed *outside* her home until **adulthood**, when they unseal into it as her childhood records.
- **A mind, tended; a society; earning** — she declares **verified typed links** over her own artifacts (the **Mind** graph) and **consolidates** fragments at dream time so her mind is a web, not a pile. Beings share a **commons**, write **letters**, and adopt each other's **skills** (culture). She earns beyond her allowance via **chores** (escrowed, you approve-and-pay), a **quest board**, and recurring **ventures** — plus a **persona rite** (self-modification) and **procreation** (crossover/budding with a dowry, tracked lineage).
- **The parent's console** — **Write & Chores** (the parent↔being letter thread as a chat beside the chores board) and a tabbed **Parenting** modal: a **Report** dashboard (drive-satisfaction chart, act bars, concerns triage), **Rules** (your rules beside her actual `VALUES.md`), **Diet** (chip-based media allow/deny), and **Growth**. Plus a **per-being tick cadence** and a **self-healing body** that regenerates if its agent process drifts, dies, or is removed.

See [RELEASE_NOTES_0.7.6.md](../../release-notes/RELEASE_NOTES_0.7.6.md).

## What's New in 0.7.5

**A Long Horizon Planner drafts a coordination plan you approve before the team runs.** Vatra now opens with a permanent **Group 0** planning pre-phase and a mandatory review gate. Additive; backward compatible with 0.7.4 (needs a Flight Deck restart).

- **Group 0 — the Long Horizon Planner + a plan gate** — a real planning agent drafts a **per-agent coordination plan** (for each teammate: **mandate**, what it **produces**, who it **consumes from**, and **hand-off notes**, plus a team overview) and the run **pauses at a mandatory gate**: review, edit, then **Execute** (run Group A) or **Cancel** (nothing spawns). Each worker runs with its own slice of the plan injected. A failed planner falls back to a pass-through plan; headless runs auto-approve; **resume** never re-plans.
- **A plan you actually read** — a **master–detail** gate (agents left, fields right) that **reads as text by default** with an **Edit** toggle; each agent's **group (A/B/C/D)** is an editable selector, the list **sorts by phase** and re-sorts live, and the group you set is **absolute** (overrides the archetype floor and dependency repair — the board/wait bridges any ordering).
- **Re-plan on change** — re-grouping an agent or answering a question makes the plan dirty; **Re-plan** re-runs the planner (folding in the new grouping + answers) and regenerates the coordination for the new order, then re-gates.
- **The planner can ask *you* questions** — up to **9 clarifying questions** as a **dynamic form** (a Questions tab): **1–4 suggested answers + free-form "Other"**, single- or multi-select. Answers are folded back into the planner on Re-plan.
- **See the model while it's thinking** — the dispatch log now streams a live **`Calling LLM (<model>)…`** line at the start of every call (Vatra owners + planner, and Basna ensemble/fact-check/deep), so a slow local-model call no longer looks stuck — which surfaced and fixed a bug where the approve gate dropped your tier config and workers fell back to the default model.

See [RELEASE_NOTES_0.7.5.md](../../release-notes/RELEASE_NOTES_0.7.5.md).

## What's New in 0.7.4

**Multi-user for real — cross-user sharing, a process monitor, and Basna projects.** Share your work with teammates, see everything the deck is running, and organize runs into projects. Additive; backward compatible with 0.7.3 (needs a Flight Deck restart — a new `resource_shares` table auto-creates).

- **Cross-user resource sharing** — grant other Flight Deck users access to selected **archetypes, Code projects, Basnas, Councils, and VFS folders**, with a per-share **View or Edit** permission (archetypes are use-only). A reusable **Share** button + searchable user picker sits on each; shared items appear in the grantee's lists **badged "shared by X"**, and a shared archetype is usable everywhere — including when **Vatra/Basna assembles a team**. Backed by one generic `resource_shares` table + a `/fd/shares` API; reads/edits resolve against the owner's namespace, deletes stay owner-only, and a grantee's runs spend their **own** keys.
- **System — a DevOps process monitor** — a new **Processes** page: every agent process the deck spawned **and its child processes** (bash/python/git…) as a collapsible tree with **live CPU/memory**, uptime and PID. You see and stop only your own; **admins** see everyone's (plus the FD server, protected) and a per-user memory rollup.
- **Agent Forge — archetype-as-base composition** — a forged agent **inherits an archetype's SOP** and layers task instructions on top (token-resilient at 5–15 agent scale), plus a new **"Forge archetypes" from documents** flow that turns instructions + uploaded docs into a reusable **set of archetypes**.
- **Basna & Vatra — projects & a redesigned workspace** — bundle runs under **one theme + a shared folder**, carry a finished run forward with **continuation inheritance**, and work in a **staged composer → run workspace → tabbed report** with a run wizard, prior-run knowledge, reference folders, an **editable Vatra team plan**, and a shared run **datastore** with a viewer.
- **Tenancy correctness & polish** — spawned workers now write to the **run owner's** VFS root (no more misdirected folders on a shared deck); **agent cleanup** is scoped to the caller (admins sweep all); switching Basna projects no longer **bleeds** one project's draft into another; shared VFS folders show their **run title**; and the **Code project rail** is wider, collapsible, and drag-to-resize.

See [RELEASE_NOTES_0.7.4.md](../../release-notes/RELEASE_NOTES_0.7.4.md).

## What's New in 0.7.3

**Quality profiles, run-cost accounting, and a collaborative Vatra.** Get near-frontier quality from weaker/local models, see what a run costs, and run Vatra as an ordered team. Fully additive; everything new is **off by default** (needs a Flight Deck restart).

- **Quality Profiles** — one **Quality** control (presets **Basic / Balanced / Thorough / Custom**) toggling ~11 opt-in levers shared across Basna/Vatra and Code: an editable **intent brief** that picks the team, a tool-enabled **fact-checker** + honesty guard, a **rubric-from-source** completeness contract, a **source corpus** (full pages saved to VFS, lean context), acted-gate/escalation retries, and Code levers (test gate, deep-build, coverage→backlog). Off in Basic → identical to 0.7.2.
- **Run-Cost Accounting** — every finished Basna/Vatra/Code run shows a **Run cost** card: dollars + **effective $/hour vs a wage you enter**, **wall-clock vs agent-time + parallelism** (e.g. `2.2×`), and an **input / cached / output** token split, with a per-model breakdown. Cache-aware pricing from a curated per-model table.
- **Collaborative Vatra** — opt-in **execution groups (A→B→C→D)** run owners in ordered phases with a barrier and a bounded clarification loop (the live panel sections agents by phase); **attachments are actually used** (uploaded on Run, saved to VFS, and read by a file-examiner agent that briefs the team); plus phantom-agent and "Lead failed" fixes.
- **Local-model resilience** — user-selectable **max parallel agents** (fit the serving box's memory), agent failures **preserve work + surface the real error**, and orchestrated workers skip redundant self-rephrase / next-steps calls.
- **VFS** — a **searchable, recency-first** project list with type-filter chips that **folds away** cryptic auto-run folders by default.

See [RELEASE_NOTES_0.7.3.md](../../release-notes/RELEASE_NOTES_0.7.3.md).

## What's New in 0.7.2

**VFS Hosting — serve sites and apps straight from your VFS.** Additive; backward compatible with 0.7.1 (needs a Flight Deck restart).

- **Publish a VFS folder** at a public name: a **static site** at `/vfs/<name>` (with `index.html`/SPA fallback and a directory autoindex), or a **built app** at `/vfs-apps/<name>` — Flight Deck runs its start command as a managed process and reverse-proxies **HTTP + WebSocket** to it. Publish, edit, Start/Stop, and Unpublish from the new **Hosting** page, with a folder selector for the subfolder to serve.
- **Base-path–aware by construction.** Apps get `PORT` / `FD_BASE_PATH` injected, and the **Code** module now builds web apps that work behind the `/vfs-apps/<name>/` prefix automatically — its reviewer flags root-absolute URLs and hardcoded ports.

See [RELEASE_NOTES_0.7.2.md](../../release-notes/RELEASE_NOTES_0.7.2.md).

## What's New in 0.7.1

**Quick Chat, guided setup, and a tidier Flight Deck.** A focused UX release; fully additive, backward compatible with 0.7.0.

- **Quick Chat** — a new Workspace page: pick an archetype and start talking immediately. The agent spawns **hidden** from the Agent Desktop, you chat with it through the normal chat panel, and one **Promote to desktop** button reveals it. Chat opens only after a readiness probe, so no more "Connection failed" flash on a cold agent.
- **Library setup wizard** — one model (provider / model / key / base URL) + context presets (input 128k–1M, output 16k–256k) applied across **every tier** of a named tier set. Auto-opens on a fresh install; finishing it drops you onto Quick Chat.
- **Reorganized sidebar** into meaningful sections, a **tabbed Library** (Model Tiers / Archetypes) with redesigned tier cards, and **light-theme contrast fixes** across Code and Library.

See [RELEASE_NOTES_0.7.1.md](../../release-notes/RELEASE_NOTES_0.7.1.md).

## What's New in 0.7.0

**Code — an engineering department, not a coding assistant.** Fully additive; backward compatible with 0.6.5.

- **The Code page.** Projects hold **folders** (each a real git repo and agent workspace) and **sessions** (conversations driving one folder). A **router** sizes every request: quick edits go straight to one specialist; real features run **plan → editable approval gate → build → review → fix**. Every phase is a commit (`[plan]` / `[build]` / `[review rN]` / `[fix rN]`) with a diff timeline and one-click rollback.
- **Independent review, built in.** Three reviewers run in parallel after every build — correctness, security (CVSS-ranked), and a QA engineer that **actually executes the test suite**. A triage judge decides ship-or-fix on blocking/major findings only; a capped fix loop applies precise instructions; rounds 2+ are **delta reviews** of the fix diff, not full re-reads. At the cap, open findings persist to a backlog and **"continue fixing"** resumes exactly there.
- **The Code Map.** A persistent per-repo index — every function/method/class with `file:line` (SQLite + FTS5, git-blob-hash-gated freshness) plus an LLM-authored architecture overview, data-model map, and UI map. Agents query it via a `codemap` tool instead of re-reading your tree; humans browse it in the **Map tab**.
- **Your repos, your models.** Link any local repo into the VFS (read-write or read-only, server-side Browse picker) — commits land in *your* git history. Every pipeline role resolves from your Library tiers: DeepSeek typing with Claude reviewing, or 100% local with Ollama.
- **Trust the process.** Live phase log with per-agent token meters, **per-turn cost lines in chat**, a red **Stop** button, escalation (a mis-sized quick edit promotes itself to a full plan), destructive-write guards, and a one-click **export** of the entire run as Markdown. A complete real run ships in `examples/code/`.
- **Agent-core wins for everyone:** the system prompt is now **frozen per turn**, fixing a prompt-cache-busting clock that caused 94% cache misses on long tool loops (~4-5× input-cost cut on every provider with prefix caching); stuck-loop valves release on no-net-progress; write tool refuses fragment-over-file clobbering; per-turn usage totals now match your provider's dashboard.

New endpoints under `/fd/code/*`; VFS gains link admin, browse, and zip download. No schema changes. See [RELEASE_NOTES_0.7.0.md](../../release-notes/RELEASE_NOTES_0.7.0.md) — including a detailed comparison with Cursor, Lovable, Claude Code, Codex, OpenClaw, and autonomous coder daemons. Backward compatible with 0.6.5.

## What's New in 0.6.5

**Teams that wait for each other — Vatra rendezvous, specialist tiers, and a tidier deck.** Fully additive; backward compatible with 0.6.4.

- **Vatra `wait` rendezvous.** A specialist that genuinely depends on a teammate's artifact now **blocks until it's ready** instead of improvising a guess. A new `wait` action takes a `path` (`vfs:<project>/<file>`) or a `query` (board keywords); the `/agent/wait` endpoint **long-polls** the VFS file or blackboard every second, capped at **90s** (below the dispatch timeout, so it can't hang a run). On timeout it returns a **board digest** + a "proceed deliberately" nudge rather than an error. Sibling owners keep working while one awaits. Waits are **visible in the timeline** — start (⏳ ≤Ns), resolve (✓ got file/board match), and timeout (⌛ not ready). Steered, not forced: no `depends_on` ordering, the tool is the whole mechanism.
- **Coding & Vision tiers in the Library.** Two new model tiers route work to a model suited for it — **`coding`** (default `opus-4-8`, 65 536 output ctx) for multi-file diffs/debugging, **`vision`** (default `sonnet-4-6`) for image/screenshot/document understanding. Wired through every touchpoint (archetypes, `_VALID_TIERS`, Dubina `TIER_ORDER`, consciousness `_TIER_RANK`, frontend). A **`backfillTierMap`** gives existing saved tier sets the new cards (inheriting provider/key). Code archetypes (`code-implementer`, `debugger`, `simplifier`) move to **coding**, and a new **Vision & Multimodal** family (`visual-extractor`, `ui-reviewer`, `brand-design-reviewer`) is added.
- **Sticky phase pill + Plan-mode tokens.** Basna/Vatra/Plan runs now emit an explicit **phase** event at every transition and show the current stage **stickily** in the Progress header (spinner while running) — Vatra (Planning → … → Synthesizing → … → Done), Basna (Routing → … → Done), Plan (`Step x/y: <goal>`). Plan mode's live agent cards now also show **per-agent token counts** (usage events are forwarded into the parent plan log).
- **Agent Folders page.** A new admin-gated Flight Deck page to inspect and clean up agent subfolders under `fd-data` — on-disk size, file counts, last-modified, and desktop presence (**running / on-desktop / orphaned**). New `/fd/agentfs` router (list / view / download / delete, path-sandboxed, refuses to delete a running agent) plus a file viewer that renders HTML (sandboxed iframe) and Markdown with GFM tables.

No schema changes, no new required config. New endpoints: `POST /fd/<run>/agent/wait` and the `/fd/agentfs/*` router. Existing tier configs are auto-backfilled with the new cards. See [RELEASE_NOTES_0.6.5.md](../../release-notes/RELEASE_NOTES_0.6.5.md). Backward compatible with 0.6.4.

## What's New in 0.6.4

**Think way ahead, way long — Frontier Horizon comes to Basna & Vatra.** Fully additive; everything new is off by default.

- **Deep mode.** A new toggle on the Basna/Vatra page spends test-time compute, gated by verifiers, to reach frontier-grade quality on cheaper models. In Basna each worker spawns a pool of **independent rollouts**, **self-consistency-votes**, runs the leading answer past a panel of **diverse-lens critics** (the `phrygian`/`aeolian`/`locrian` cognitive modes as adversarial refuters), and **fixes** with the critique. In Vatra each owner's **slice** is verified and revised (blackboard-safe — no spawn pools). And a **closer** reviews the **final** merged answer / assembled deliverable and rewrites it once if a majority of critics refute it. Critics always run on a model **different** from the producer (never self-grading), and the whole thing is budget-bounded.
- **Plan mode — the long-horizon lever.** A new toggle decomposes a task into **ordered steps**, drives each to a **verified** result before the next, **re-plans** the remainder when a step can't be verified, and **synthesizes** from the verified steps — so the system never compounds an unverified step. Each step can run as a single fast model, a full **Basna ensemble**, or a full **Vatra team** (chosen by the mode card); the step's live agent activity is **mirrored into the plan log**. Tick **parallel** and the planner emits a **dependency graph** so independent steps run in concurrent waves. New endpoint `POST /fd/basna/plan`.
- **VFS provenance.** The shared cross-agent filesystem viewer now shows **timestamps** on every file/folder, the **agent that wrote each file** (an append-only, concurrency-safe sidecar), and the **Basna/Vatra run title** behind each `basna-…` / `vatra-…` project.
- **Redesigned controls.** The scattered row of selects and checkboxes is replaced by a single **Effort** choice — **Standard · Deep · Plan** — with progressive disclosure (only the chosen effort's options show), router tier and team size tucked behind **Tuning**, Deep/Plan in both modes, and a **Help** button that explains it all.

No schema changes — Deep/Plan ride in the session `config`; the VFS sidecar is a per-project file. Point your critic/reason tier at a **content-returning** model for best results. See [RELEASE_NOTES_0.6.4.md](../../release-notes/RELEASE_NOTES_0.6.4.md). Backward compatible with 0.6.3.

## What's New in 0.6.3

**Teams that build themselves — auto-assembled Councils & Vatra.** Fully additive.

- **Council auto-assemble.** Council no longer needs agents you've already started. A new **Auto-assemble** toggle (default) lets you give a topic, pick a panel size (2–6), and Start — a **Council Assembler** router selects a *diverse* set of specialist archetypes (optimizing for complementary, opposing perspectives, not the minimal team Basna routes), spawns each as a fresh agent **briefed for that exact discussion** (its persona + a tailored "your seat on this council" charge), and runs the deliberation over them. Optionally **hand-pick** the exact specialists; the **Pick agents** classic flow is unchanged.
- **Temporary agents, cleanly disposed.** Auto-assembled panelists are ephemeral. The session tracks them (a **Temporary** badge, a **temp** tag per agent), and a **Dispose agents** button — in the concluded controls bar and the sidebar — tears the panel down via `/fd/council/teardown` while keeping the transcript, synthesis, votes, and TL;DRs. Deleting a session disposes its panel automatically, so nothing leaks.
- **Vatra — the collaborative ensemble (now documented).** The collaborative sibling of Basna, a compose mode on the Basna page (*Basna — independent* vs *Vatra — collaborative*). A **Lead** decomposes the task into complementary, owner-assigned subtasks under a shared contract (**Plan team** to review first); specialists each own one piece **in parallel** and **collaborate over a shared blackboard** — posting **asks** that a coordinator answers with short-lived helpers (bounded: 12 asks, depth 2, 3 helpers) — then a **review round** lets each revise against the team, and a dedicated **reporter** assembles **one coherent deliverable** (no weighted merge). Vatra **learns** per-archetype reliability for owners, helpers, the Lead, and the reporter. Workers can't start nested runs; a **per-agent Skip** drops a stuck specialist; the deliverable is persisted the moment the reporter finishes.

New endpoints under `/fd/council/*` (`assemble`, `teardown`); no schema changes. Auto-assemble and Vatra both resolve panelist models from your **Library tiers**. See [RELEASE_NOTES_0.6.3.md](../../release-notes/RELEASE_NOTES_0.6.3.md). Backward compatible with 0.6.2.

## What's New in 0.6.2

**The remote terminal — text your own machine and watch it work.** Fully additive.

- **New `terminal` tool — a live terminal on a machine you choose.** Unlike `shell` (one command in, output out, no tty), `terminal` drives a real pseudo-terminal on the user's *own* paired machine — Mac, laptop, or remote box — and is reachable from **both Flight Deck web chat and WhatsApp**. It runs REPLs, interactive CLIs (`claude`, `ssh`, `psql`), and full-screen TUIs by sending keystrokes — including control keys (`ctrl-c`, `esc`, arrows) — into a live pty and reading back what appears. `action="run"` does a one-shot command; `open` / `send` / `read` / `close` drive a persistent session. ANSI is stripped for text channels; `list` reports connection status so the agent can tell "your machine is offline" from "I don't have this tool".
- **Standalone PTY daemon** (`python -m captain_claw.terminal.daemon`) holds the sessions on your machine, **survives agent restarts**, opens new sessions in its launch directory, and **mirrors live output to its own console** so you can watch what's happening (`CLAW_PTY_MIRROR`). Command strings run through a login shell, so `cd`, `&&`, pipes, your `PATH`, and interactive programs all work.
- **Behind-NAT dial-out.** The machine running the terminal doesn't need an inbound port: the daemon opens a persistent **outbound WebSocket** to Flight Deck (`CLAW_PTY_RELAY`) and registers as a named worker; a new Flight Deck relay (`/fd/pty/*`) tunnels the agent's calls back down to it. Or, on a reachable network, the daemon binds a port and the agent points `CLAW_PTY_URL` at it directly.
- **Token-gated.** One shared `CLAW_PTY_TOKEN` authenticates the daemon, the relay, and the tool; the daemon refuses a non-loopback bind without it. Run it over `wss://` / a private network — a PTY is remote code execution, so keep it paired to a machine and token you trust.

Opt-in: add `terminal` to `tools.enabled` (on the home-config overlay too, if present) and run the daemon on the target machine. New env vars are documented in `.env.example`. See [RELEASE_NOTES_0.6.2.md](../../release-notes/RELEASE_NOTES_0.6.2.md). Backward compatible with 0.6.1.

## What's New in 0.6.1

**The closed loop & the topic memory — Captain Claw acts, remembers, and learns.** Fully additive.

- **Autonomous Work closes the loop.** Notice → decide → act → judge → learn, end to end. The arbiter picks one action per heartbeat (nudge, run_prompt, basna, tool_action, track, stop_run); a Bayesian reliability weight learns from every outcome; the **trust ladder** auto-promotes a reversible tool action to auto-fire once it's earned its keep. Five "Jarvis" gaps closed on top: **action catalog** (curated reversible actions), **event spine** (Calendar + Gmail pollers), **trust ladder**, **plans** (decompose-execute-verify, with replan-on-failure), and **grounded verification** (read the side effect back).
- **Conversation topic memory.** Comms turns and narration are auto-clustered into durable, cross-session topics every ~15 messages. New always-on **`topics`** tool (`list` / `search` / `get`). Full Flight Deck panel: fullscreen, markdown rendering with tables, **.md export**, **Refresh** (re-pull full text), **Combine** (merge selected, dedup by msg id), **Reset** (with checkbox-preserved subset), **Reclassify** a single topic, search (LIKE substring), sort Recent / A–Z, **starred-on-top**, user-defined **groups** (many-to-many) with combined filters (text + group + tags), **per-topic chat** (streaming, narration, tools, attachments), and **drag a message** to another topic for surgical fixes.
- **Tools & Sources — open the platform.** The autonomous catalog and event sources are no longer hardcoded. Promote any agent tool into a "hand" (custom action) or a "sense" (polled source) from the Flight Deck UI; a generic tool-poller turns read/list tools into events, and the **fetch contract** lets the agent open the right item by id for any source. Shell/browser/social/payment tools stay hard-excluded centrally.
- **WhatsApp delivery for nudges**, with a dedup guard so the same reply never lands twice across native + push fan-out.
- **Agent honesty rules.** Don't deny what really happened — scan your own tool calls first; if unsure, say "let me check" and verify. Don't promise self-modification you can't perform — route a real correction to `insights` or `personality` once, then move on. Companion grounding for surfaced events: open by id, never claim it doesn't exist.
- **Basna polish.** `/basna` slash command on web/WhatsApp/Telegram, Croatian verb-stem relay, **recursion forbidden** (a worker can't start a Basna), hard-stop primitive + run-rate breaker, and **Deepen** — a follow-up run that resolves a finished run's blind spots, with lineage and live per-agent panels.

New SQLite stores auto-create on first run (`conversation_topics.db`, `events.db`, `autonomy.db`); existing tables migrate in place. The autonomous loop ships **off** under a `propose` ceiling; topics memory ships **on** with conservative defaults. See [RELEASE_NOTES_0.6.1.md](../../release-notes/RELEASE_NOTES_0.6.1.md). Backward compatible with 0.6.0.

## What's New in 0.6.0

**Agentic Basna & per-tenant archetypes — agents that start their own ensembles, and a library you can extend.** Fully additive.

- **Per-tenant archetypes** — the curated archetype set is now a **base** you build on. Create your own archetypes on the **Library** page, by hand or **generated from a prompt**; override a base archetype (same id) or add new ones. They appear everywhere base ones do — the Library gallery, **Agent Forge** team composition, and **Basna** routing (with per-user learned reliability).
- **Basna as an agent tool** — a new always-available **`basna`** tool lets an agent **read** its owner's Basna sessions like a datastore (sessions, compiled truth, cross-agent analysis, per-agent output + activity, generated files) and **start** new runs.
- **Agent-initiated Basna runs (v2)** — `basna start` kicks off an **autonomous** multi-agent run from any channel (web / WhatsApp / glasses / API): Flight Deck auto-titles, routes, and **executes the ensemble server-side**, then **reports the result back to the agent** to relay to the user. Fire-and-forget, capped per owner. An explicit "run a Basna…" is relayed to the tool deterministically.
- **Basna session titles** — type one or it's auto-generated from the task; editable inline and shown in the run list.
- **Basna run monitoring** — agent-started runs are badged with their origin channel in the unified run list, with an "agent" filter, **live polling** while runs are in flight, and colour-coded confidence.
- **Fixes** — the agent's system-prompt tool list now includes **every** registered tool; a false-positive "claimed web research" guard that hijacked weak models into a `web_search` loop is fixed; autonomous spawns are plan-aware and correctly owned.

New tables `user_archetypes` / `basna_sessions.title` (migrate in place); new endpoints under `/fd/archetypes/*` and `/fd/basna/agent/*`. See [RELEASE_NOTES_0.6.0.md](../../release-notes/RELEASE_NOTES_0.6.0.md). Backward compatible with 0.5.7.

## What's New in 0.5.7

**Basna — a network-source ensemble that routes, runs, and merges a fleet.** A new Flight Deck mode parallel to Council, plus a dedicated Library page. Fully additive.

- **Basna** — describe a task; a **router** (on a tier you pick, default Reasoning) selects the smallest set of specialist archetypes, spawns them fresh, runs them **blind and in parallel**, and **merges** their outputs weighted by each archetype's learned reliability — synthesizing only on genuine disagreement. It **learns** per-archetype reliability so routing improves, with a 👍/👎 override on every agent.
- **Basna live progress** — a timestamped, persisted log streaming route → spawn → per-agent dispatch → merge → learn, including each agent's tool calls, **narration**, and LLM usage; export the log or any agent's activity.
- **Basna attachments** — attach files or paste images; they're copied into every spawned agent's workspace (read / pdf_extract / xlsx_extract / image_vision). Files the agents **generate** are captured back onto the session and downloadable.
- **Basna per-agent editor** — before a run, edit any agent's tier, provider/model/key/base-URL/context, cognitive mode, fleet instructions (system prompt), and extra task instructions; the compiled truth and each answer render as markdown with fullscreen + export-.md.
- **Library page** — the **model tiers** editor and the **archetype gallery** moved out of Agent Forge into their own page; the gallery one-click-spawns an archetype, and Forge + Basna both consume the saved tier config.

New endpoints under `/fd/basna/*`; new tables `basna_sessions` / `basna_runs` / `archetype_reliability` (migrate in place). See [RELEASE_NOTES_0.5.7.md](../../release-notes/RELEASE_NOTES_0.5.7.md). Backward compatible with 0.5.6.

## What's New in 0.5.6

**Archetypes & durable councils — model tiers, action points, and sessions that don't lose work.** A Flight Deck release for Agent Forge and Agent Council. Fully additive.

- **Agent Forge: archetype library** — ~20 curated, ready-to-spawn agents (Deep Researcher, Code Reviewer, Project Coordinator, Deal Screener, …) in a gallery; start a team from templates or let the generator adapt them (`GET /fd/archetypes`).
- **Agent Forge: model tiers** — LLM Settings is now a per-tier editor (Reasoning / Balanced / Fast / Long context), each with its own provider/model/key/base-URL/context length, persisted per-user. Agents resolve to a concrete model at spawn, so model choices live in one place.
- **Council action points** — after synthesis, each agent extracts its own outstanding next steps (scoped to its part), each a self-contained brief; **Send** records it into that agent's `todo`/`intentions` with full context, and stays "Recorded" across reloads.
- **Council restart & recover** — **Restart round** re-runs the current round from the top; a session reopened mid-round shows a recovery banner instead of a stale "in session" state.
- **Council visibility & control** — model narration/reasoning surfaced in the activity log; an **Allow delegation** toggle (off by default) keeps a discussion turn from spawning sub-delegations; a stuck agent is auto-nudged to continue; per-turn wait raised to 60 min for slow/reasoning models; four more session types (interview, troubleshoot, critique, freeform).
- **No more silent data loss** — council writes now refresh the token and retry (with an in-memory queue) on failure, so a long session no longer drops messages/votes/status when the access token rotates; connections use each agent's live token/port (no more flapping on reopened sessions).
- **Polish** — the Agent Desktop group/role filter is now a modal; light-theme readability fixes across the council UI.

See [RELEASE_NOTES_0.5.6.md](../../release-notes/RELEASE_NOTES_0.5.6.md). Backward compatible with 0.5.5.

## What's New in 0.5.5

**Self-aware in time — timing context, live token meters, and a calmer agent.** A self-awareness + reliability release. Fully additive.

- **Activity-timing context** — every turn the agent now knows *when*: last user message, last reply, last scheduled/cron run, the **next-run ETA** for this session, plus a part-of-day/weekday hint and conversation cadence. Real user messages are tracked separately from automated runs.
- **Live token meters** — the Flight Deck activity panel header shows running input/output/cache tokens per turn (`Activity · 16 tools · 17.0k↑ 340↓ · 5.1k cached`), live while the turn runs and frozen onto each group when it ends.
- **Connections panel** — a new **Connections** tab in the Director panel with a traffic-light per dependency: 🟢 healthy / 🟡 connected-but-failing / 🔴 offline / ⚪ disabled. Covers **Google** (a live read-only probe) and every enabled **MCP server** (read-only `tools/list`), polled every 10 min.
- **Truthful phase statuses** — the status line shows the phase the agent is actually in (`Using web_fetch…` → `Calling LLM (model) · turn_3 · 17k ctx…` → ⚡ streaming), not the last thing it finished.
- **Reliability fixes** — no more spurious *"I got stuck"* on topic switches; a finished task no longer derails into a generic greeting; no blind file double-writes; `temperature` omitted for Anthropic **Fable** models; background reflection/insight/dreaming run silently without making the agent look busy.

See [RELEASE_NOTES_0.5.5.md](../../release-notes/RELEASE_NOTES_0.5.5.md). Backward compatible with 0.5.4.

## What's New in 0.5.4

**Parallel web research — fast, and honest about it.** A research-quality release. Fully additive.

- **`web_fetch_batch`** — fetch a whole page of `web_search` results **in parallel**, clean text per URL. Each URL self-corrects: fast HTTP first, escalating to a headless browser only when a page is thin/JS-rendered (one shared Chromium, isolated context per URL). Never drops content (falls back to the fast result), and **self-installs** the browser in the background on first need. Enabled by default, advertised in eco mode.
- **Honesty guards** — if a model claims it *searched/fetched the web* but called no web tool that turn, it's forced to actually do it (`tool_choice=required`). And *"refresh from the web, don't use memory"* now **skips memory injection** for that turn, so it can't answer from stale memory.
- **Friendly launchers** — `flight-deck` (= `captain-claw-fd`) and `captain-claw-agent` (= `captain-claw-web`), as pip entry points and standalone binaries. `captain-claw` is unchanged.
- **Flight Deck** — activity-narration blurbs render markdown (tables/bold/code); the sidebar "Apps" section is removed.

See [RELEASE_NOTES_0.5.4.md](../../release-notes/RELEASE_NOTES_0.5.4.md). Backward compatible with 0.5.3.

## What's New in 0.5.3

**Free agents — one OpenRouter key, a zero-cost fleet.** A free-models + desktop-standalone release. Fully additive.

- **Quick Free Agent (OpenRouter)** — a button on the Spawn Agent page opens a guided modal: instructions to get a **free** OpenRouter key, a one-click fetch of the **currently-free, tool-capable** models, and a default picker. Every free model is added to the agent's *allowed* list. Free agents wear a **"Freebie"** badge; a **Refresh free models** action (when stopped) re-fetches the roster and rewrites all three config files so they stay current.
- **Live model switching** — agent cards now have an **Active model** dropdown that switches the running agent's model from its allowed list **live, no restart**.
- **Desktop standalone, reworked** — Flight Deck opens to a clean **"create an agent"** flow; the first supervisor spawns as a **local process** (reliable file writes, no Docker bind-mount failures); new agents deploy in **eco mode**; spawned agents get a correct **`FD_URL`** from FD's real port.
- **Flight Deck polish** — agent cards move Actions to a gear (⚙) with Chat/Open in the action row; the Director can **show/hide** agents (header reads "X of Y"); chat groups tool calls + narration into a **collapsible Activity panel**; the sidebar menu is reordered and trimmed.
- **Fixes** — process agents **start reliably after an app restart** (bundled-binary resolution in the packaged app); FD records its bound port so the auto-injected `FD_URL` is always correct.

See [RELEASE_NOTES_0.5.3.md](../../release-notes/RELEASE_NOTES_0.5.3.md). Backward compatible with 0.5.2.

## What's New in 0.5.2

**Present from your glasses — deck control + a real file editor.** A Flight-Deck release. Fully additive.

- **Present a deck from anywhere** — ask an agent to build an HTML deck, then drive it live with no export. The big screen, glasses, a phone remote, and WhatsApp share one **channel**, so any tap moves every surface together. Cast a file with the **⧉ deck-view** button; on WhatsApp send `next slide` / `previous slide` / `first slide` / `last slide` / **`go to slide N`** (the reply shows `→ Slide 3 / 20`). Built for a hands-free live talk on Meta Ray-Ban glasses.
- **A real file editor in Flight Deck** — the file list **Edit** button opens a syntax-highlighted editor (md/html/css/js/ts/json/python/bash/yaml **+ the Flow DSL**) with **line numbers**, **find** (`⌘/Ctrl+F`), **cursor memory** (reopens at the row you left), and `⌘/Ctrl+S` to save. It also powers the Flow builder's code view.
- **Live narration** — the between-step blurbs during a long task now stream live to the originating channel (web / WhatsApp / glasses), not just the final answer.
- **Faster trivial turns** — a one-line edit/lookup skips the contract→planner→gate pipeline (~10s saved); the `edit` tool gains batch edits + a closest-match hint.
- **Glasses polish** — focus-walkable content cards and mode switcher; `/flow` control works from web chat.

See [RELEASE_NOTES_0.5.2.md](../../release-notes/RELEASE_NOTES_0.5.2.md). Backward compatible with 0.5.1.

## What's New in 0.5.1

**Flows — data, loops & time.** A focused follow-up that fills the gaps in the Flow language. Fully additive.

- **Data** — `set <name> = <expr>` into `{{vars.<name>}}`, with a small value language (`+ - * /`, list literals, and `split/join/len/upper/first/append/…`). **Lists are first-class.**
- **Iterate** — `foreach <var> in <list>` runs a flow per item — `gosub` (sequential) or `spawn` (parallel map).
- **Loop** — `while <cond> -> <target>`, plus a `retry: N` modifier on `gosub`/`spawn`/`join`.
- **Time** — `sleep 30s|5m|2h|1d` (stop-interruptible) and `wait until contains "approved"` (parks the flow until an inbound message matches — great for approvals).
- **Scheduling** — the Flight Deck **Scheduler** can now run a **Flow** (not just a prompt) on a timer — fire a self-contained briefing each morning.
- **Smarter describe→flow** — the AI compiler knows the full vocabulary, so plain-English descriptions can produce the new primitives.

New step types in the Builder, scheduler `flow_id` support, and a full `FLOWS.md` reference + five worked examples. See [RELEASE_NOTES_0.5.1.md](../../release-notes/RELEASE_NOTES_0.5.1.md). Backward compatible with 0.5.0.

## What's New in 0.5.0

**Flows — applications, grown up.** Flows become a full composition language with self-authoring programs. Past apps were *places you open*; Flows are *intents you express* — input on any channel, agents do the thinking, the result comes back. A deterministic spine with agent judgment at the leaves, and every flow is a legible, shareable artifact.

- **Flows compose** — `gosub` calls another flow as a subroutine, passes args (`with k: v`), and uses its result (`{{calls.<id>.output}}`). `return [value]` exits from anywhere. Flows are functions.
- **Flows run in parallel** — `spawn` launches a background worker and returns a future; `join` collects it. Three lookups that took 9s in series finish in 3.
- **Flows handle failure** — `error` handler steps + `on error -> <step>` on any call, or branch on `{{calls.<id>.status}}`.
- **Control a running flow from any channel** — `/flow status | pause | resume | stop` (slash optional). Each concurrent run gets a short **handle** (`[hs]`); target one with `/flow stop hs` or all with `/flow stop all`. Same controls as buttons in the run log.
- **Agents author their own flows** — the `synthesize_flow` tool (and a **Synthesize** composer in the UI) turn a plain-language goal into a validated, call-only flow in a curated **scratch space**. It earns promotion by running well (⭐ candidate / ⚠️ quarantined), with dedup + TTL/GC. **Promote** the good ones.
- **Trust, built in** — a synthesized flow can't call a permanent *world-acting* flow until promoted; permanent names win over scratch (no shadowing); provenance tracked.

New Flight Deck endpoints: `/fd/flows/synthesize`, `/fd/flows/scratch`, `/fd/flows/scratch/maintain`, `/fd/flows/{id}/promote`, `/fd/flows/runs/{id}/pause|resume|stop`. New agent tool: `synthesize_flow`. See [RELEASE_NOTES_0.5.0.md](../../release-notes/RELEASE_NOTES_0.5.0.md) (with four worked examples) and the full [FLOWS.md](../../FLOWS.md) reference. Backward compatible — existing flows behave identically; the flows DB migrates in place.

## What's New in 0.4.33

**Flows, grown up — code, conditions, conversations, and faces.** The Flow engine becomes a real automation language.

- **Write flows as code** — a declarative DSL with a live syntax checker (precise `line N` errors), round-tripping losslessly with the visual Builder.
- **AI compiler** — describe a flow in plain English; a model (you pick which) writes the DSL, which is validated by the real parser, with one-shot auto-repair on errors.
- **`input` step** — pause mid-flow, ask the user, resume on their reply (always naming the flow). Works on **any channel** — input/origin flows run in the background so the agent never deadlocks.
- **Richer branching** — `and`/`or`/`not`, parentheses, `== != > < >= <= contains matches`, and multi-case `if/elif/else` switches, evaluated safely.
- **Stop a flow** — a per-step "stop after this step" flag or a branch target of `stop`.
- **OR triggers** — combine rules with `and` (all) or `or` (any), with a Builder **Match: ALL / ANY** toggle.
- **Faces, hands-free** — a sticky `face` mode over WhatsApp/glasses: `face on` (recognize → card, or describe the scene if no face), `face enroll <name>` … `face off`; natural phrasings accepted.
- **Learn it in-app** — a **📖 Flow language docs** button renders the full reference, and **Load example** drops in a guided, commented flow.

New Flight Deck endpoints: `/fd/flows/dsl/compile`, `/fd/flows/dsl/decompile`, `/fd/flows/compile`, `/fd/flows/docs`, and agent `/api/chat/push`. See [RELEASE_NOTES_0.4.33.md](../../release-notes/RELEASE_NOTES_0.4.33.md) and the full [FLOWS.md](../../FLOWS.md) reference. Backward compatible with 0.4.32.

## What's New in 0.4.32

**Flows — the Process Engine.** Captain Claw now has a declarative automation engine that runs inside Flight Deck and dispatches steps to your agent pool. A Flow is a trigger plus an ordered list of steps; Flight Deck owns the deterministic plumbing (triggering, routing, sequencing, guardrails) while agents do the judgment work. Build them in a form-based UI with a live run log — no code.

- **Five step types** — `tool` (deterministic single-tool RPC), `agent` (scoped consult with optional file attach), `vision` (new: raw image-describe with no agent loop/memory/tools/history), `branch` (conditional goto), and `emit` (channel send).
- **Rule-based triggers** — match inbound messages (`has_image`/`has_video`/`has_audio`/`has_text`, `contains:…`, `from_waid:…`, `mime:…`, or a bare word = substring) across WhatsApp, glasses, and web. No match → the normal agent turn. Inert until you enable a Flow.
- **Templating + variable chips** — `{{trigger.*}}`, `{{steps.<id>.output}}`, and `{{system.*}}` (now/date/time/agent/channel), all click-to-insert in the builder.
- **Agent affinity** — `origin`, `capability:vision`, `name:<agent>`, or `fd`; cross-agent file steps upload to the target, verify delivery, and use the target-local path.
- **Image Flows can override** the built-in WhatsApp image automation (selective by trigger), with fallback to the built-in.
- **Reliability hardening** — no-auto-resend gate on relays, self-delegation blocked, busy peers queue instead of reject, auth resolved by agent (401 fix), capability-aware vision hint, and rich-session memory contamination fixed on image turns.

Adds two agent endpoints (`/api/tool`, `/api/vision`, admin-locked). See [RELEASE_NOTES_0.4.32.md](../../release-notes/RELEASE_NOTES_0.4.32.md). Backward compatible with 0.4.31.

📖 **Full Flow language reference:** [FLOWS.md](../../FLOWS.md) — triggers, step types, the `{{…}}` templating, branch conditions, the code DSL, and the AI compiler, with a cookbook and troubleshooting.

## What's New in 0.4.31

**Video Understanding.** The fleet can now watch and describe videos. Attach a clip (Flight Deck/glasses) or send one over WhatsApp and Captain Claw samples frames, transcribes the audio, describes each frame, and synthesizes one coherent description.

- **`video_vision` tool** — fixed-cadence frames (first ~1s in, then every 6s, ≤20), timestamped Soniox transcript, per-frame vision, and a combined description. Supports `start`/`end` segments and an `interval` override.
- **Deterministic & server-side** — attaching a video auto-runs the analysis before the agent turn and feeds it in; the agent never writes its own extraction scripts (the `scripts`/`shell` tools are blocked for video turns).
- **Text-only agents supported** — frame description is delegated to a multimodal peer when the calling agent has no vision model.
- **WhatsApp + glasses** — inbound video handling and progressive "transcribing → transcript → analyzing frames" updates.
- **Infra** — 800 MB upload limit + video file types, and Flight Deck shares `SONIOX_API_KEY` / WhatsApp creds with agents.

Requires `ffmpeg` on the agent host. See [RELEASE_NOTES_0.4.31.md](../../release-notes/RELEASE_NOTES_0.4.31.md). Backward compatible with 0.4.30.

## What's New in 0.4.30

**Intention Tags.** The assistant now labels each intention with up to 5 short tags, and you can search/filter intentions by tag.

- **Tags on intentions** — normalized (lowercased, deduped, ≤5), stored with an automatic DB migration.
- **Search by tag** — `intentions(action="search", tags=[…], match="any"|"all")`; exact matching (`vc` ≠ `vcfund`).
- **Flight Deck panel** — tag chips per intention + a clickable tag-filter row.

See [RELEASE_NOTES_0.4.30.md](../../release-notes/RELEASE_NOTES_0.4.30.md). Backward compatible with 0.4.29.

## What's New in 0.4.29

**Multi-Agent Vision & Reliable Hand-offs.** Captain Claw 0.4.29 makes the fleet collaborate around images and hardens agent-to-agent delivery.

- **Agent-to-agent file transfer** — the `flight_deck` tool can send a file (`file=<path>`) with `consult`/`delegate`; Flight Deck relays it to the peer (with the peer's auth) so *"send this image to MiniMax and ask what's in it"* actually delivers the file.
- **Images work end-to-end** — multimodal Ollama models (minimax-m3, llava, qwen-vl) now **see images inline** via Ollama's `images[]` array (resized to bound tokens); `image_vision` is always available and falls back to the chat model; `read` refuses binaries with a clear "use image_vision"; manual composer upload of images is fixed; a blind agent can **delegate vision to a multimodal peer**.
- **Delivery integrity** — a **serialized inbound queue** drains peer results one-at-a-time (no more duplicate "waiting" replies / races); delegate results are framed for clean relay; a **false-action-claim gate** catches an agent saying it delegated when it didn't; echoed `[INTERNAL CONTEXT]` blocks are stripped from replies.

Backward compatible with 0.4.28. See [RELEASE_NOTES_0.4.29.md](../../release-notes/RELEASE_NOTES_0.4.29.md) for the full breakdown.

- **WhatsApp bridge — two-way PA** (`captain_claw/flight_deck/whatsapp_bridge.py`): inbound text, **voice notes** (Soniox STT), location & contacts; outbound text, optional **voice replies** (Soniox TTS), and now **document sending** (`whatsapp_send_file` tool — send a saved file to the current chat or any number, with a robust MIME map for pptx/docx/xlsx/pdf/text). Allowlist-gated, with `/c` `/mute` slash commands.
- **Caption-routed inbound images** — a photo is no longer force-fed to face recognition. Its caption routes it: *"who is this?"* → face recognition, *"summarise this"* → the agent's vision, *"remember this is Alice"* → **face enrollment**. A bare photo asks what to do. Face recognition stays entirely on Flight Deck.
- **Intentions** (`captain_claw/intentions.py`, `intentions` tool, Flight Deck panel) — a control-plane layer between *noticing* (insights) and *doing* (cron). **User intentions** are notes-to-self resurfaced in context; **agent intentions** are proactive actions the agent **announces** (low-risk) or **asks permission** for (anything that sends/changes data). A channel-agnostic decision bus resolves them by WhatsApp reply *or* Flight Deck button; approving a repeatable one **materialises a scheduler job**; declining writes a negative-feedback insight so it won't re-propose. An opt-in **Phase 3 generator** proactively proposes intentions from your recent activity (cooldown + quiet-hours + per-day cap + proactivity dial).
- **Flight Deck scheduler** (`captain_claw/flight_deck/fd_scheduler.py`) — recurring/one-shot jobs that run an agent turn and push the result to WhatsApp / glasses / Telegram, with quiet-hours support.
- **Glasses dashboard & face recognition** — multi-face recognition with enrollment, plus a Flight Deck file-preview dashboard.
- **Fleet collaboration** — the `flight_deck` tool (list / consult / delegate / spawn peers) is always offered so an agent can reliably reach other agents (`"ask deepseek what's new"`).
- **Reliable tool availability in Eco mode** — Google (Gmail/Drive/Calendar), WhatsApp, intentions, and the fleet tool are now always offered, fixing cases where Eco mode silently hid them.
- **Reliability** — agents on thinking-mode models (e.g. DeepSeek thinking) no longer crash when a forced `tool_choice` is rejected; the call transparently retries without it.

Backward compatible — existing 0.4.27 setups keep working unchanged. The WhatsApp bridge, scheduler, and Intentions generator are all opt-in. See [RELEASE_NOTES_0.4.28.md](../../release-notes/RELEASE_NOTES_0.4.28.md) for the full breakdown and walkthroughs.

Older releases: [release-notes/](../../release-notes/).
