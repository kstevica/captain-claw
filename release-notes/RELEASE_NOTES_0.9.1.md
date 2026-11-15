# Captain Claw v0.9.1 Release Notes

**Release title:** Strongroom
**Release date:** 2026-11-15

Your private data can stay yours while a model does the work. **Anonymization** turns names, companies,
addresses and numbers into tokens like ⟦PERSON_2⟧ before anything reaches a model, keeps the real values
encrypted on your deck, and puts them back when the answer comes home. **Data rooms** do that to a whole
virtual data room so Claude can read hundreds of deal files without ever learning who is in them, and
send its memos and workbooks back for you to open with the real names. **Vatra, Basna and Council** move
into the kiosk, the kiosk gets a new home page, and agents can work in your own Chrome.

88 pull requests, #489 to #576. **Restart Flight Deck, then every agent, hard-refresh the browsers, and
reload Captain for Chrome (now 0.3.0).** The kiosk looks different, task rephrasing is now off by default,
and agents call themselves Captain; see [Upgrading](#upgrading).

Step-by-step guides for the three headline features:
**[Anonymization](https://github.com/kstevica/captain-claw/blob/main/docs/guide/anonymization.md)** ·
**[Data rooms](https://github.com/kstevica/captain-claw/blob/main/docs/guide/data-rooms.md)** ·
**[Vatra, Basna and Council](https://github.com/kstevica/captain-claw/blob/main/docs/guide/vatra-basna-council.md)**

## New

### Anonymization: private data out, the real names back for you

Captain takes names, companies, addresses, emails, phone numbers, IBANs, OIB/JMBG/MB and other ID
numbers, and dates of birth out of a text, a file, a whole folder or a web page. Each becomes a token like
⟦PERSON_1⟧, and the same party keeps one number (⟦ORG_1:2⟧ is another spelling of ⟦ORG_1⟧, so a link
still opens when it's put back). The real values stay on your deck, encrypted, in a vault per case that
only you can open. Dates, amounts, titles and cities stay, so the text reads and adds up.

- **Who looks.** Code always does: labels, legal forms, checksums, known names, links and file names.
  An admin can add a **trusted model** (local, or a cloud one the admin vouches for) that catches bare
  names, a small company's products and descriptions that point to one person, and an optional local
  **name pack**. The model only finds; it never rewrites your text. Up to three rounds check the result,
  and anything still found means no result, not a leaky one.
- **Files.** Word, Excel and PowerPoint are anonymized inside the file (text, tables, cells, sheet names,
  comments, authors) and come back as the same kind of file; PDF and HTML come back as Markdown. A file or
  a whole folder in your shared folders or the File library gets an anonymized copy beside it.
- **Keep nothing** keeps no vault, and **Make them coarser** turns a date of birth into a year, an address
  into a city, and ages and amounts into ranges.
- **The privacy gate** makes it automatic for your agents. What they read reaches their model as tokens,
  from the sources you pick (mail, chats, calendar, contacts, files, data, memory) or from **the whole
  chat**, while your own chat shows the real names (**Real names** under a reply). Mail and WhatsApp
  messages an agent sends get the real values back just before they leave. Off until each person turns it
  on; by default only for agents on a model that isn't trusted.
- **Agents** get the `anonymize` tool. It reads a file or page itself, so the original never reaches the
  agent's model, and its **restore** writes a file for you that the model never sees.
- **Captain for Chrome 0.3.0**: **Anonymize this page** (or the selection), or right-click → **Anonymize
  with Captain**, then **Copy**, **Save in Flight Deck**, and later **Put the real names back**.
- **Vaults** show every value with its spellings and a **history** of each token: what found it, which
  model it went out to, and when it came back. **Reset** empties a vault and keeps its numbers.

**How to start:**

1. Admins (optional, for the best recall): **Admin console → Settings → Anonymization**: pick a
   **Provider** and **Model** (a local Ollama model is trusted as it is), **Save**, then **Check on
   samples** to see what code alone missed and what the model caught.
2. Open **Anonymize**: kiosk **Settings → What they keep → Anonymize** (or **More → Anonymize**); full
   layout **Files → Anonymize**.
3. Pick **Text** or **File**, choose **In a vault**, name the **Vault** (one per case, e.g. `Velora`), and
   click **Anonymize**. Check the result, then **Copy** or **Download**.
4. To put the names back: **Put back**, pick the vault, paste the text or drop the file, **Put back**.
5. For your agents: **Privacy gate**, switch **On**, pick **What it covers** and the sources, **Save**,
   and start a new conversation.

Step by step, with every option, the environment variables and what the gate doesn't cover:
[docs/guide/anonymization.md](https://github.com/kstevica/captain-claw/blob/main/docs/guide/anonymization.md).

### Data rooms: a deal's files for Claude, without the names

A data room takes a virtual data room (a ZIP or a folder of hundreds of files) and anonymizes all of it
into one vault. Claude reads the anonymized copies through its own connector, **Captain Data Rooms**: the
story, the context and every number, but no names. One token is one party in every file of the room
(⟦PERSON_1:2⟧ is another spelling of ⟦PERSON_1⟧, so a payroll lines up with a register). Claude sends its
answers, memos, workbooks and decks back through the same connector, and Flight Deck puts the real names
back, for you alone, on the deck.

- **It reads** Word, Excel (shown as sheets), PowerPoint, PDF, text, and scans and pictures through OCR;
  pictures inside Office files are replaced. After the run, a **sweep** goes over every file again with all
  the names it learned, and a **leak check** blocks any file that still holds a real one. **Not a name
  here** keeps a word the check flagged, and **Check again** re-checks.
- **You review before Claude sees anything**: the original beside what Claude reads, **Roles** (Target,
  Buyer, CFO…), **Same party?** for two tokens that look like one party (with code's score, a slider and
  answer-all), a description of each file (one line, a paragraph, a summary, keywords, topics), and then
  **Release**.
- **15 tools for Claude**: `room_list`, `room_overview`, `room_list_files`, `room_find_files`,
  `room_file_summaries`, `room_search`, `room_read`, `room_get_file`, the `room_deliver*` tools for text,
  documents, workbooks and decks, `room_get_delivery` and `room_tasks`. The server tells Claude to keep
  every token and to match parties on the number.
- **Results come home**: deliveries land under **From Claude**, restored for the room's owner only, to read
  (**Real names** / **As Claude wrote it**) or download as Word, Excel, PowerPoint or text; Croatian names
  get their case endings back. **Tasks for Claude** queue your own requests, anonymized first. **Download
  for Claude** builds a ZIP for a Claude Project when the deck has no public address.

**How to start:**

1. Kiosk: **Settings → Data rooms** (or **More → Data rooms**); full layout: **Files → Data rooms**.
2. Fill in **Codename (Claude sees this)** and **Your name for it (only you see this)**, then **Create data
   room**.
3. **Add files**: drop a ZIP (25 MB by default, `FD_ROOMS_MAX_ZIP_MB`; add several for a bigger VDR) and
   click **Add the ZIP**, or **Read the folder** from **A folder in Files** (admins: any folder **On the
   deck's machine**).
4. **Names to hide**: add the names you already know (the target, the buyer, a project name), then **Save**.
5. **Run**, and watch the progress (phases, sizes, time left; **Stop** stops at once).
6. Review: blocked files and leftovers, **Roles**, **Same party?**, the file descriptions.
7. **Release N**.
8. On the **Claude's access** card, **Copy** the connector address (`https://<your deck>/fd/mcp-rooms`).
9. In Claude: **Settings → Connectors → Add custom connector**, paste it, sign in with your deck email and
   password, tick the rooms, **Allow**.

**What it needs:** a trusted model for the best recall (see Anonymization), OCR for scans, and for the
connector a public HTTPS address (`FD_PUBLIC_URL`) with a stable `FD_JWT_SECRET`. The address must let
Claude's `Claude-User` requests through: Cloudflare's AI-bot blocking and quick tunnels refuse them after
sign-in.

Step by step, with every screen, Claude's tools, the rules it's given and what can still slip:
[docs/guide/data-rooms.md](https://github.com/kstevica/captain-claw/blob/main/docs/guide/data-rooms.md).

### Vatra, Basna and Council in the kiosk

Three ways to put several agents on one job, now in the kiosk for everyone signed in, in the kiosk's
look and at phone width. To **build one thing**, use Vatra; to **find the best answer**, Basna; to **talk
it through**, Council. Basna and Vatra are separate pages now, each showing only its own runs, and the
full layout has its own **Vatra** entry under **Multi-Agent**.

**Where:** click **More** under your agents (on a phone, the **More** tile; also in **Settings**), then
under **Agents as a team** pick **Vatra**, **Basna** or **Council**. Each opens in a wide dialog with
**Fullscreen**.

**Vatra** (a team builds one result on a shared board):

1. Pick a project, or **New project** (name, the theme every run gets, optional instructions).
2. In **Runs**, click **New** and write the job in **Task / intent**. **Attach** adds files. The **Setup**
   chip holds **Effort** (Standard, Deep, Plan), **Quality** and **Tuning** (Max agents, Grouped phases,
   Shared datastore); or use the **Wizard**.
3. **Plan team**: the Lead splits the job into parts, one owner each. Edit, move or remove parts.
4. **Run team**, then read the **Group 0 — coordination plan** (each agent's mandate, what it produces and
   consumes) and click **Execute**.
5. The report, every agent's part and the files are on the run.

**Basna** (several agents answer blind, merged by reliability):

1. Pick a project, click **New** and write the question in **Task / intent**.
2. **Route** picks the agents; **Run ensemble** runs them.
3. The merged answer is on the **Truth** tab.

**Council** (agents discuss in rounds, then sum up and vote):

1. **New session** (the first time, **Start your first session**), then **Topic / Task**.
2. **Auto-assemble** (2–6 panelists) or **Pick agents** (at least 2), then **Start Council**.
3. After the rounds, **Synthesize**, then **Export .md**.

Step by step, with every option and what to do when a run stalls:
[docs/guide/vatra-basna-council.md](https://github.com/kstevica/captain-claw/blob/main/docs/guide/vatra-basna-council.md).

### A new home in the kiosk

- **Today** is the home when no chat is open: a greeting, **Waiting for you** (tasks that are your move,
  proposals and open questions, each with its own answer buttons), **Today’s priorities** and **Handled
  automatically**, each with **See what happened**.
- **The nav** above the agent list reads **Today · Agents · Inbox · Tasks · People · Automations ·
  Activity**. **Inbox** is the Line, **Tasks** the Taskboard (with the count of what needs you),
  **Automations** opens **Scheduled tasks**.
- **Agents** shows one card per agent: its picture, its live state (Ready, Working, New reply, Asleep,
  Starting), **Chat** or **Wake and chat**, grouped and ordered like the agent list.
- **New agent opens the Agent store**, for kiosk and simple-layout users alike. **Featured** shows four
  ready agents; **Add** puts one on your team (**Add and set up** when **Set up its way of working next** is
  on). **Advanced**, top right, opens the old picker with **From a file** and **Forge agents**.
- **No right column.** An agent's files and tables open from **⋯ → Files & data**.
- **Below the agent list:** **Library · Data · Folders · More**. **More** holds Crew and Captain Swarm (where
  allowed), Vatra, Basna and Council, and Anonymize and Data rooms.
- **Chat reads lighter.** Steps fold into one line ("‹Agent› worked through 6 steps", **See what
  happened**), the busy line says what the agent is doing, approvals ask "Good to go?" with **Approve** /
  **Dismiss**, the final answer has no bubble, and the latest narration stays in view.
- **Agent pictures.** **⋯ → Picture**: one of 30 icons, **Upload a picture**, or **Initials**. Each member
  of a shared agent picks their own.
- **People** lists everyone on your Taskboard tasks: search by name or email (accents ignored), sort by
  **Most tasks** or **Name A–Z**, filter **Pending** (**All · 1 · 2–5 · 6–9 · 10+**), rows like "8 tasks · 3
  pending · 5 done". A task opened from People comes back to People, and a Taskboard opened on one task
  shows only it, with **Show all tasks**.
- **The console** opens on what needs you ("2 things need you"), with a grouped menu.

### Agents in your own Chrome

- **An agent works in your signed-in Chrome** (`browser` with `target="chrome"`), in a tab group named
  after it, with **Stop** in the Captain for Chrome panel. Off until you turn on **Agents can use this
  browser** in the panel; only the agent's owner, never on a shared agent's member turns, never on Flight
  Deck's own pages, `chrome://` or `file://`.
- **30 new browser actions**: tabs, refs from `read_page`, `find`, `page_text`, `page_files`, `form_input`,
  hover, drag, `upload_file`, `javascript`, `console`, `network_requests`, `download` and `save_page`.
  They work in the agent's own headless browser too.
- **A site that blocks the agent** (both the browser and `web_fetch`) makes it ask you once, in Flight
  Deck, to use your Chrome, or suggest installing Captain for Chrome.
- **Chrome extensions in the agent's browser**: `extension_install` from the Web Store, a folder, a `.zip`
  or `.crx` (you approve each one by default; `tools.browser.extension_policy`), and
  `extension_id="captain"` installs Captain for Chrome from your deck. An agent can pair Captain for
  Chrome in its **own** browser on the server after you say yes; **Connections → Browser** lists it with
  **Disconnect**.
- **Forms fixed**: punctuation types correctly, `type` replaces a field's text (`append: true` adds),
  fields in another site's frame are reachable, and errors after a submit come back. When the run ends,
  the tab you were on comes back to the front.

### Agents and models

- **`model_switch`**: "use a smarter model for this" works from any channel. An agent switches among its
  own provider's models (same key, same endpoint): the catalog's **Smartest** and **Fastest and cheapest**
  for OpenAI, Anthropic, Gemini, Grok and DeepSeek, or the models an admin lists under **Console → Models →
  Models agents can switch to** for a custom endpoint. The kiosk's chat box has a model picker beside the
  effort picker. Switching on its own needs **Pick a smarter or cheaper model on its own** (off by default).
- **External memory**: **Qdrant** and **ArcadeDB** join Membase and Hindsight, with vectors made on the
  deck by default (nothing leaves it). Agents read from memory before every reply, by each memory's
  **Reading priority** (**First — my main memory**, **Alongside**, **Only when needed**). Each person can
  connect up to five memories of their own, beside the admin's shared one.
- **`web_search`** returns 10 results by default and up to 50, each with a `Source:` line, and
  `enrich=true` adds the matching passages from the top pages.
- **Task rephrase is off by default.** Turn it back on per agent: kiosk **Agent settings → Model & thinking
  → Rewrite requests**.
- **Crew workers run on the agent's own model and key** by default, even over a saved CLI login
  (`runs_on=cli` uses the CLI's login).
- **81 built-in tools** (80 in the desktop app): new `anonymize` and `model_switch`.

### Files, data and schedules

- **Data library tables can be edited by hand** in the kiosk: double-click a cell, **Add row**, delete
  from the row panel. An edit the agent beat you to is refused and the row reloads.
- **Shared folders in the File and Data libraries**: the filter is **Agents & folders**, and a shared folder
  gets **In the library** and **Tables**.
- **Agents' own cron jobs in Scheduled tasks**: **Set up by agents (N)**, with pause, resume, **Run now**
  (which starts an off agent and turns it off again), delete and history.
- **Updates: the admin picks when the deck restarts**: **Console → Updates → Update now** asks **When
  should it restart?** (**Restart now**, **In 2 minutes**, **In 5 minutes**, **In 10 minutes**).

### Captain, not Captain Claw

Everything the deck says is "Captain" now: prompts, guards, channel help, email footers, the CLI, MCP
names, the HUD and the desktop app. An agent on the untouched default personality calls itself Captain;
one you renamed keeps its name. New WhatsApp links show as "Captain".

### Fixes

- A guard's rewrite request is the system's, not you pushing back, so the agent no longer opens with
  "You're right…" and no lesson comes out of it.
- A chat message is offered as a lesson once (**Save as a lesson?** → **Saved as a lesson**).
- A fleet sign-in always gets a fleet token.
- Custom tools: the update bell and the missing-secret error read right.

## Upgrading

1. **Update the code** (`git pull` + `pip install -e .`, or **Console → Updates**; installed desktop apps
   update themselves), **restart Flight Deck, then every agent**, and **hard-refresh** every browser tab. No
   new Python dependencies. The update *into* 0.9.1 still uses the old countdown; the restart-time choice
   applies from the next one.
2. **Reload Captain for Chrome** (0.1.0 → 0.3.0) in `chrome://extensions`, or download it again from
   **Connections → Browser**. It asks for new permissions (`debugger`, `tabGroups`, `alarms`); Chrome shows
   its "is debugging this browser" bar while an agent works there.
3. **The kiosk looks different**: Today is the home, the agent cards moved to **Agents**, Line is **Inbox**,
   Taskboard is **Tasks**, Scheduled is **Automations**, the right column is gone (**⋯ → Files & data**),
   and **New agent** opens the Agent store. Tell your team before they log in.
4. **Task rephrase is off by default** from the agent's next restart, unless its config sets
   `scale.task_rephrase_enabled`. Turn it on per agent with **Rewrite requests**.
5. **Agents call themselves Captain** if they still run the untouched default personality.
6. **Crew workers bill to the agent's model and key**, even where a CLI login was saved.
7. **External memory** costs a little more while one is connected: one recall per reply, and insights taken
   every 5 messages instead of 20. An old single connection moves over on first read; someone who had
   their own starts with the deck's shared memory **off**. `PUT /fd/external-memory/priority` and
   `/mirror` are gone.
8. **`web_search`** results are bigger in context (10 by default, `max_results: 5` in old configs is
   ignored).
9. **Anonymization and data rooms are opt-in.** Nothing is anonymized until an admin sets a trusted model
   or a person runs it; the privacy gate is off until each person turns it on. A data room needs a public
   HTTPS address for Claude to connect (and no Cloudflare AI-bot blocking in front of it).

## Pull requests

- **Anonymization**: #511, #512, #514–#520, #522–#538, #547, #550, #551, #554, #555, #556
- **Data rooms**: #539–#545, #548, #549, #552, #553, #556, #557, #558
- **Kiosk, home and navigation**: #500, #503, #504, #506, #559, #560, #562, #564, #566–#571, #573–#576
- **Captain for Chrome and the agent's browser**: #494–#499, #507–#509
- **Agents, models, memory**: #491, #492, #493, #502, #510, #513, #521, #572
- **Files, schedules, updates**: #501, #505, #561, #563
- **Rename, docs, fixes**: #489, #490, #546, #565

Full list: [github.com/kstevica/captain-claw/pulls?q=is:merged](https://github.com/kstevica/captain-claw/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-11-09..2026-11-15)
