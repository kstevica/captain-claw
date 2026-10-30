# Captain Claw v0.8.4 Release Notes

**Release title:** Onboarded, On the Board
**Release date:** 2026-10-30

0.8.3 kept a record of everything your agents do. 0.8.4 teaches them your work,
and shows you where every piece of it stands.

**Onboarding** gets an agent ready for the job. Give it your documents and your
instructions. It keeps every document whole as a knowledge base it can search,
at no token cost, reads as much as you budget for, and turns that into rules,
skills, facts and lessons you review before it gets them. Then test it: an
**exam** writes questions from the documents, asks them through the agent's real
chat, grades the answers, and says why a wrong one went wrong. **Practice** on
real tasks, correct it, and it tries again.

**The Taskboard** is one place for everything your agents were asked and sent,
on every channel: **Needs me**, **Waiting on others**, **Agents working**,
**Done**. It works like Gmail, keys included. Next steps wait for your approval,
you can hand a task to an agent, and the board **checks that the agent really
did it** instead of taking its word.

**Team decks got much tighter.** On a deck with sign-in, agents are confined to
their own files, non-admins' agents no longer run code, and the team's API keys
can't be read or sent anywhere the team didn't choose.
**[Read this before you upgrade a team deck.](#before-you-upgrade-a-team-deck)**

Also in 0.8.4: **your own AI keys** with no daily limits; a **Profile** with a
researcher that drafts your texts; **two new tools**, `knowledge` and
`visualize`; pictures that reach the chat and the channel; a slicker kiosk; the
**Waveshare ePaper 1.54** boards; and a new Captain icon.

71 pull requests, three days after 0.8.3.

**Restart Flight Deck, then every agent.** See [Upgrading](#upgrading).

## Highlights

### Onboarding: teach an agent your work

**⋯ → Onboarding** on an agent opens one page with five tabs: **Overview**
(how it did, what to do next), **Lessons**, **Practice**, **Materials** and
**Exams**. When you create an agent, tick *Set up its way of working next* to
start there.

**Way of working: your material goes in**

- Add up to 12 documents by drag and drop, upload or from the File library
  (PDF, Word, Excel, PowerPoint, CSV, Markdown, text, HTML; up to 3,000 PDF pages
  each), say how the agent should use each one (rules, procedures, reference,
  templates, or let it decide), and add instructions of your own.
- **A knowledge base that costs no tokens.** Every document reaches the agent
  whole: an overview, parts under 200 KB, and a map of its sections. Semantic
  memory indexes `knowledge/` first, and the new `knowledge` tool searches and
  reads it.
- **A budget for reading.** **Quick** (60k tokens), **Standard** (250k) or
  **Deep** (1M). Material over the budget is read by its most instructive
  sections, chosen without a model and pulled towards an optional **Focus on**.
- **Read in the background.** Close the dialog or restart Flight Deck and it
  goes on; the bell says when it's ready. Each step stays small and calls are
  paced, and if the model server says "not now", the read waits it out and shows
  the wait.
- **The agent reads with its own model**, with thinking off, on its own key and
  endpoint. Your tiers step in only if it's off or stops answering, and the
  review says so.
- **You review before anything lands.** Rules, what it should ask about first,
  skills, facts, templates and a glossary, each with its source. A longer read
  keeps more: up to 150 rules, 30 skills, 1,000 facts, 30 ask-first items, 20
  templates and 400 terms from one read.
- **Spreadsheets become tables.** An `.xlsx` or `.csv` also gives the agent one
  datastore table per sheet, with no model involved, so prices and SKUs are exact
  lookups. Codes like `00123` keep their zeros.
- **Read the rest later.** Each document tracks which sections were read; pick
  what to read next and only the unread part costs tokens.
- **Add to it or replace it**, on new and existing agents alike. Learned facts
  and the agent's own files stay either way, and an agent that's off gets
  everything when it starts.
- **Keep a file from chat.** "Put this in your knowledge" stores a chat file the
  same way, with no model reading it.

**Lessons: the right rule at the right moment**

- A setup's rules, ask-first items and facts become the agent's lessons, each
  with its source and who taught it. Add one by hand, reword it, or take it away.
- **Always in mind** lessons stay in the instructions. **When it matters** lessons
  reach the agent with the messages they bear on, picked by words and by meaning,
  at most 6 items and 1,800 characters a turn, with no model call.
- **Save as a lesson** on your chat messages: a hover action, and a chip when a
  message reads like a correction or something to remember (English and Croatian).

**Exams: does it really know this?**

- **Quick** (10 questions), **Standard** (25), **Deep** (60) or **Marathon** (120),
  on all documents or the ones you tick. Token estimates follow your pick.
- Questions are asked through the agent's real chat, on a private lane that keeps
  nothing. Every answer is graded point by point against its section.
- A wrong answer comes with its cause: didn't look, looked in the wrong place,
  found it and got it wrong, or made things up.
- **Teach it this** turns an answer into a lesson, **Read the parts it missed**
  queues those documents, and asking the same exam again shows better or worse
  since last time.

**Practice: real work, nothing sent**

- Give it a task as a user would, or take one of three suggested from its
  documents. It answers on the same private lane. **Good**, or **Fix it**: what it
  should have done becomes a lesson, and it tries again.

**For a team**

- **An admin's base.** What an admin teaches a user's agent is its base. The user
  adds to it, but can't reword or remove it, and their own Replace keeps it.
- **Members suggest.** On a shared agent, *Suggest as a lesson* waits for the owner
  in the bell and the Lessons tab.
- **Pass it on.** Package an onboarded agent's documents, tables, lessons, skills
  and exam questions as a new archetype, ready for anyone to start from.

**Also in the kiosk**

- **Instructions, insights & skills** (⋯) replaces Instructions. Insights are
  cards with their source document, which you can search, teach, correct or forget.
  Skills can be read, switched on or off, and corrected where the agent keeps them
  itself.
- **Materials** browses the documents: a read meter, sections as a tree marked
  read or searchable only, a reader and search across all of it.

### The Taskboard: where everything stands

**Tasks** in the kiosk, **⋯ → Tasks** on an agent, or **Settings → Taskboard**. A
badge counts what needs you. Every user has their own board: a member's tasks go
to that member, and owners and admins never see them.

- **What's on it.** Every turn your agents run, on any channel; scheduled jobs and
  their runs; Autonomous Work proposals; follow-ups; held Gmail auto-replies and
  Gmail sends; linked systems; agents' to-do lists; and the last 30 days of each
  agent's chats, read once, marked as history and never acted on.
- **Your own inboxes, if you want.** Telegram inbox, WhatsApp Web and Gmail are each
  off until you consent. Their text is kept 30 days, and switching one off deletes
  it, names and addresses included.
- **It reads the work.** One call on the agent's own model writes the title, the
  ask, the result and who's involved. [Jev](https://console.typesafe.ai/keys) then
  decides whether it's a task, who owes the next move, and whether it's done;
  without Jev, rules decide. **Why is this here?** shows every decision.
- **Like Gmail.** `j`/`k`, `e` Done, `b` Snooze, `/` search, `g` then a lane, `?` for
  the list. A person's reply brings the same task back to **Needs me**, marked
  *New reply*, with its history. Your corrections stick against later sweeps, with
  Undo.
- **Read it properly.** **Open bigger** (`f`) fills the board with one task. A task
  with a Gmail thread shows the whole email there, fetched with your own token when
  you open it and kept nowhere. Markdown renders in rows and the timeline; links
  and pictures show as words, since this can be anyone's text.
- **Next steps wait for you.** With Jev, the agent's model drafts a reply, a reminder
  or a question as a proposal you edit, approve or reject. Email only ever becomes a
  draft; recipients come only from verified parties, never from model text; and no
  key or one-click button approves a next step. Nothing is proposed on a message
  flagged as carrying instructions, or on history.
- **Hand it to an agent.** **Next move** (`m`) → *The agent…*: pick the agent, say what
  to do, **Hand over**. It runs as your own request in a session of its own, with
  your words first and the task's text fenced as data. Sending still goes by your
  Email sending switch and its recipient list.
- **Then it checks.** An agent's "done" is only its word, so the board compares the
  reply with the tool-call register, and reads Gmail and Google Calendar with your
  token, to see that the message went out and the event exists. Jev then compares
  your words with the record. **What the agent did** shows the verdict. In doubt, the
  task stays open and yours.
- **Merge, split, export.** Threads attach by strong keys; merge, unmerge and split
  with Undo; Jev can suggest a merge but never applies one. Export as CSV or JSONL.

Agents report their turns once restarted; the rest of the board fills from what the
deck already records. Plan and known gaps: [docs/taskboard-plan.md](https://github.com/kstevica/captain-claw/blob/main/docs/taskboard-plan.md).

### Security: a deck for people who don't share a machine

Every process agent runs as the deck's OS user, so until now, on a multi-user deck,
an agent's tools could reach other users' files. On a deck with sign-in:

- **Agents stay in their own files.** Path arguments are checked by real path against
  the agent's folder, its workspace, its owner's VFS and the call's shared folders.
  `.env` is out of reach, control files can't be written, `glob` and `grep` results are
  filtered, and the browser takes web URLs only.
- **Non-admins' agents don't run code or drive the host.** `shell`, `terminal`,
  `scripts`, `app_runner`, `code`, `termux`, `bat`, `desktop_action`, `screen_capture`
  and `pinchtab` are refused and hidden. No skill dependency installs, no links outside
  the user's VFS, no read folders. Their outbound web tools take public addresses only.
- **Hosted apps are admin-only.** Publishing or starting one runs a command on the
  server. Static sites stay open to everyone. Adding VFS links and browsing the host's
  folders are admin-only too.
- **Team keys stay put.** Non-admins read a team key as `····` and its last four; saving
  a masked value keeps the key. A team key, and any key in Flight Deck's own environment,
  goes only to the provider's own endpoint or one the published team sets route it
  through. A change that would send it elsewhere is refused or re-keyed.
- **Environment pins.** The variables that would have a non-admin's agent load code from
  a path or route its calls elsewhere keep Flight Deck's values. Saving a new one in the
  agent editor is now refused and named, not silently ignored.
- **Callers are who they prove they are.** Agent routes bind an owner only from a verified
  bearer or the agent's own token; a `source_port` or `owner_id` in a request counts as
  identity only with sign-in off.

This is containment, not an OS sandbox: an admin's agent still has a shell that reaches the
whole host. The sandbox is designed and parked in
[docs/agent-os-sandbox-plan.md](https://github.com/kstevica/captain-claw/blob/main/docs/agent-os-sandbox-plan.md).

### Your own AI keys

- **Settings → Your own AI keys**: Anthropic, OpenAI, Google Gemini, OpenRouter or xAI. Each
  key is checked with the provider before it's saved, sealed at rest, and never returned.
- While it's on, your tier set runs on your keys and **no daily limits apply**. A tier on a
  provider you have a key for keeps its model, any other moves to one of yours (pick per
  tier), and local tiers stay. Existing agents follow, and running ones restart.
- Turn it off and the old models return, with the team key back through the healer. On a shared
  agent the owner's keys are what count.
- On by default. An admin can turn it off for the deck, which moves everyone back.

### Profile, with a researcher

**Settings → Profile** is now **Personal · Deck-wide · Password**.

- **Personal** holds who you are as material: your name, your links, your company and its
  links, and documents such as a CV or a bio.
- **Fill it in for me** reads those documents, visits the links, looks you and the company up on
  the web, and drafts *About me*, *My company* and *My standing preferences*. It runs on your
  go-ahead, on the team's model and credentials, and is followed live; the bell says when it's
  finished. **Nothing reaches your agents until you Use a draft and Save.** Links to private hosts
  are never fetched.
- **Deck-wide**: an admin edits the defaults in place; everyone else reads them.

### Two new tools

| Tool | What it does |
|---|---|
| `knowledge` | Searches and reads the owner's documents, with no model: `search` (hybrid; exact codes first, quoted phrases required), `section` (a section whole, across parts), `outline`, `list`, and `add` (keep a file from chat). Shows on every Flight Deck agent; shared-agent members never get it. |
| `visualize` | HTML in, the page and a PNG of it out: a fragment or whole page, with pinned Chart.js, ECharts, Plotly, D3, Mermaid or Leaflet. The PNG reaches the chat, Telegram, WhatsApp and the Files menu, and goes into a `.docx`. Script errors, failed loads and blank pictures come back for the agent to fix. The page never touches the host. Always on. |

That makes **65 built-in tools** per agent.

### The chat

- **Pictures show up.** On signed-in decks chat pictures were blank; that's fixed, and a
  picture that can't load shows its name. A picture the turn saved but no picture tool reported
  (a plotting script's chart) now reaches the web chat and the channels: from the reply's words,
  or, when it names none, one Jev call decides (`tools.jev.find_images: false` turns that off).
- **Files menu** in the composer: the last five files the agent made in this chat, with **Open** and
  **Show in library**.
- **Insert now** on a queued message (or `I`): hand it to the running turn, which reads it on its
  next step. Inserting never starts the next queued task, and a click as the turn ends isn't lost.
- **Compact cards** for automation results and the agent's rephrasing of your request: a small
  card with a two-line preview and **Open**, instead of a full reply bubble.

### The kiosk

- The agent's **⋯** opens a card of icon tiles (a sheet on a phone), and **Settings** shares the same
  sheet, grouped, with a real dark-theme switch. Arrow keys move between tiles.
- **Agent settings** has two tabs: *Model & thinking* and *Reasoning effort*. A model change being
  typed survives switching, and saving a new model reads its reasoning levels afresh.
- **The bell** puts proposals waiting for Approve or Reject on top, even ones older than the page.

### WhatsApp Web: a ban doesn't take your history

A burner number can be banned, and the ban took the kept messages with it. Now every way a link can
end moves the store aside, and only you delete it. **Disconnect** and a new **Switch number** ask
first; a ban or a removal on the phone keeps it. Agents read the linked number and what's kept as one.
Sending, the whitelist and auto-replies stay with the linked number, because an old number's chats are
first contacts for a new one, the main ban trigger. This hasn't been tried against a real ban.

### Devices

- **Waveshare ESP32-S3-ePaper-1.54**, with and without touch (SKU 32298 and 34212, **V2 boards only**).
  A 200×200 black-and-white panel, fast enough for a drawn menu and answers that page with the voice
  (partial refresh 0.6 s, full 1.8 s).
  - BOOT talks and PWR cancels a recording; double-click PWR (or double-tap the touch board) for home.
  - Only a press ends a recording here: this mic's level cut sentences short.
  - The sleeping screen keeps the clock, with a sleeping robot, and redraws on a timer you pick.
  - The green LED shows what it's doing: steady recording, breathing while the agent works, two
    blinks when something waits.
  - The onboard sensor isn't used: the board's own heat throws it off.
  - Flash it once over USB (`pio run -e epaper154 -t upload`, or `-e epaper154_touch`); it updates over
    the air after that.
- **Blue for asleep, red for stopped.** The agent's dot on every device now says why it can't be
  reached: blue while Flight Deck has put it to sleep (talking wakes it), red while it's switched off.
  A deck that doesn't say keeps the old yellow.
- **Firmware 0.19.10** / PaperColor B **1.0.22**. For M5Stack boards the only change since 0.19.1 is
  that dot (0.19.2); the rest is the Waveshare board's.

### Memory and fixes

- **Insights stop repeating themselves.** An exact repeat, or the same fact reworded, is dropped;
  "within 14 days" and "30 days" stay two facts. Insights keep the file and section they came from,
  and a dropped repeat fills that in if the stored fact lacks it. Dreams' intuitions get the same check.
- **Spreadsheets read properly.** Workbooks from openpyxl and similar writers came out as their title
  alone in `xlsx_extract`, and the datastore read only the first sheet. Both are fixed, and asking for a
  sheet by name now gets that sheet or an error.
- **Ollama.** Assistant tool calls are now sent back to the model, which stops Gemma 4 repeating a call
  every turn, and `-cloud` models no longer get the 32k cap meant for local caches.
- **PDFs with structure.** The optional `pdf` extra reads headings and tables, so a book's section map
  follows its chapters. Without it, PDFs are read page by page as before.
- **Malformed tool calls.** Parameters a model's XML call left inside another's value are split out before
  the register and the guards see them.

### And

- **A new icon.** The Captain icon is now the Bot in a captain's cap, everywhere an icon lives, the
  desktop app and PWA included.
- **deed.page example.** An optional [MCP example](https://github.com/kstevica/captain-claw/blob/main/examples/mcp/deed-page.md) for publishing pages,
  thanks to [@f-o-x11](https://github.com/f-o-x11).

## Before you upgrade a team deck

These apply on a deck with sign-in (`FD_AUTH_ENABLED`), when each agent next starts.

- **Non-admins' agents lose `shell`, `terminal`, `scripts`, `app_runner`, `code`, `termux`, `bat`,
  `desktop_action`, `screen_capture` and `pinchtab`.** If someone relies on those, make them an admin or,
  on a deck whose users all trust each other, set `FD_AGENT_CONFINE=off`.
- **Agents can't read or write outside their own files**, admins' agents included: an admin's agent keeps
  its linked folders and read folders, and its shell is not held back.
- **Publishing or starting a hosted app, adding a VFS link and browsing the host's folders are admin-only.**
- **A non-admin can't send a team key to another endpoint.** Moving an agent re-keys it to that endpoint's own
  team key, or is refused. An `.env` edit that would send it elsewhere, or that adds a pinned variable such as
  `HTTP_PROXY` or `PATH`, answers 400 and writes nothing.

Decks with sign-in off, such as the desktop app and one person's deck, are unchanged.

## Upgrading

1. **Each Flight Deck** — pull, and `pip install .` where Captain Claw is installed non-editable. Add
   `pip install 'captain-claw[pdf]'` for PDFs with headings and tables. PyMuPDF is AGPL-3.0, so it stays optional.
2. **Restart Flight Deck, then every agent** — including Docker and standalone agents. Agents need the
   restart for confinement, the new tools, lessons, exams and reporting turns to the Taskboard. An agent
   without it is told to restart before an exam and keeps its old instructions block. Then hard-refresh
   every browser.
3. **WhatsApp Web** — its service outlives a Flight Deck restart, so restart it too (turn *Allow WhatsApp Web*
   off and on again), or it goes on deleting the history on a ban. Until then it answers 404 to the new
   disconnect rather than deleting what you asked to keep.
4. **Devices** — publish the new firmware from `firmware/device` with
   `python tools/publish.py <env> --deck https://<deck>`; devices take it from *Menu → Software update* or by
   themselves. Waveshare boards need one flash over USB first.
5. **For the Taskboard's next steps** — a Jev key (*Connections → Jev*, or the team's). Each user turns on their
   own inboxes on the board.

BotPort hubs need nothing new: 0.8.4 keeps protocol 2, and BotPort's code is unchanged since 0.8.2.

**Changed**

- Non-admins' agents on a sign-in deck lose the tools that run code, and every agent is held to its own files
  (see [above](#before-you-upgrade-a-team-deck)).
- The agent's **⋯** menu: *Instructions* is now *Instructions, insights & skills*, and *Onboarding* is new.
- A setup's facts are lessons, not insights, so they reach the agent when they bear on a message.
- Reject on an Autonomous Work proposal is compare-and-set, like Approve.

**New settings**

| Setting | Default | What it does |
|---|---|---|
| `FD_AGENT_CONFINE` | on with sign-in | `off` turns confinement off, for a deck whose users all trust each other |
| `FD_TASKBOARD_DISABLED` | — | `true` stops the Taskboard's sweep and triage; it then fills only when opened |
| `CLAW_TURN_REGISTER` (agent) | on | `off` stops an agent reporting its turns |
| `FD_SETUP_TOKENS_PER_MIN` | `40000` | How fast a setup read may go; `0` is unpaced. Models on the same machine aren't paced |
| `FD_SETUP_REASONING` | `off` | Thinking level for setup reads; `auto` leaves the model's own |
| `FD_PROFILE_REASONING` | `low` | Thinking level for the Profile researcher |
| `memory.max_knowledge_files` | `3000` | Semantic memory's cap for `knowledge/`, kept apart from other files |
| `tools.jev.find_images` | `true` | Let Jev pick the picture a reply hands over |
| *Your own AI keys* (admin) | on | Lets users run their agents on their own keys |
| Taskboard inboxes (per user) | off | Telegram inbox, WhatsApp Web and Gmail on your board |
| Taskboard auto-close, Jev and reading caps (per user) | on, 200, 100 a day | Whether Jev may close tasks, and how many calls a board makes a day; `0` is no cap |
| `pdf` extra | not installed | `pip install 'captain-claw[pdf]'`: PDFs with headings and tables |

New stores create themselves on start: `taskboard.db` on the deck; `lessons.json`, `practice.json` and
`.owner-setup.json` for each onboarded agent. No manual migration.
