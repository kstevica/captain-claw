# Captain Claw v0.9.3 Release Notes

**Release title:** Headway
**Release date:** 2026-11-24

You now decide how far an agent pushes. **Drive** is a setting with four levels, from Relaxed (the first
complete answer) to Ambitious (keeps going, checks its answer against what you asked, and may add the
obvious next step). Pick it in the chat box, set a default per agent, or just ask the agent to change it.
**Web playbooks** give your agents 439 ready ways to get something done on a website, in a store in
Captain and in Captain for Chrome. Spreadsheets open as tables with a filter on every column. And this
release **closes a hole that let anyone read files from a deck without signing in**: update, and change
the secrets your deck kept in its settings file.

38 pull requests, #743 to #782. **Restart Flight Deck, then every agent, hard-refresh the browsers, and
reload Captain for Chrome (now 0.9.0).** If your deck can be reached from the internet, read
[Security](#security-update-now) first; the other steps are in [Upgrading](#upgrading).

## Security: update now

**Any file the deck could read was open to anyone, with no sign-in** (#764). Flight Deck's page handler
served whatever path it was asked for, so a request with `..` in it climbed out of the folder of static
files and returned any file the deck's process could read, its `.env` with API keys included. It worked
through a reverse proxy too. 0.9.2 has the fault, and so do the versions before it that serve the same
handler; scanners on the internet already try it.

What 0.9.3 does:

- The page handler serves only files inside its own folder.
- A guard in front of every route answers 400 to any path with a `..` part, written plainly,
  percent-encoded, encoded twice or with backslashes, and closes such a WebSocket with code 1008.

What you do:

1. **Update and restart every deck.**
2. **Change every secret that was in the deck's `.env`** or in other files its process can read: model
   provider keys, the deck's own secret, mail and bot tokens. Treat them as read if the deck was reachable
   from outside your network.

Also in this release: the `vfs` tool's `mkdir`, `mv` and `rm` now respect folders linked as read-only
(#753).

## New

### Drive: how far an agent pushes

Some questions want the first complete answer. Others want an agent that keeps at it until the work holds
up. **Drive** is that choice, in four levels:

| Level | What it does |
|---|---|
| **Relaxed** | "Takes the first complete answer. Fastest, cheapest." Fewer steps, no second tries. |
| **Diligent** | "The everyday level." What agents did before this release. |
| **Persistent** | "Keeps going longer. Slower, costs more." More steps, more tries when a check fails, and a check of the answer before you get it. |
| **Ambitious** | "As Persistent, and may add the obvious next step. Costs the most." |

**Where you set it.**

1. **In the chat box**: the **Drive** picker beside reasoning effort, with **Auto**, **Relaxed**,
   **Diligent**, **Persistent** and **Ambitious**. It's remembered per agent. Members of a shared agent
   don't see it.
2. **In the agent's settings**: **How far it pushes** sets the agent's own default (**Not set** leaves it
   to the rest), and the switch **At Ambitious, add the obvious next step** turns the extra work on or off.
3. **By asking**: *"Be more persistent on this one"*, *"Go back to the everyday level"*, *"Don't add extra
   work"*. The agent changes it with the new `drive_level` tool, for this turn, this chat or as its
   default. Only the agent's owner can.

**Which level a turn runs at**: what you picked for the turn, else the agent's default, else what its
cognitive mode suggests (Ionian and Phrygian suggest Persistent), else Diligent. Automations ignore a
turn's pick.

**What the levels change.**

- **Limits.** Steps a turn may take: 2× the plan up to 100 at Relaxed, 4× up to 300 at Diligent, 6× up to
  450 at Persistent and Ambitious. Tries after a failed check: none at Relaxed, 2 at Diligent, 3 above.
- **A check before delivering** (Persistent and Ambitious). After a turn that used tools, one more model
  call compares the answer with what you asked and sends it back once if something is missing or isn't
  backed by what the tools returned. If the check itself fails, the answer goes out as it is.
- **The obvious next step** (Ambitious). Once the answer passes, the agent may do one more piece of work
  nobody asked for: reading, and writing files (`write`, `edit`, `docx`, `xlsx`, `pptx`, `visualize`),
  within a quarter of a turn's step budget (10 tool calls by default). It's added under **"Also done (not asked for)"**. Never on automations, a shared
  agent's member turns or answers that used no tools.
- **Short turns too.** Every level except Diligent adds a short paragraph to the agent's prompt, so a
  one-step answer differs by level as well.

The level is in the usage and turn logs, and `GET /fd/taskboard/drive-stats?days=30` gives turns,
outcomes, tool calls and time per level. A change takes effect on the next turn, with no restart
(`~/.captain-claw/drive.txt`, `drive_initiative.txt`); `drive.default` and `drive.initiative` in the config
are the fallback.

### Web playbooks: 439 ways to get something done on a website

A playbook is a known way through a site: where to go, what to fill in, what to read, where to stop.
0.9.3 ships **439 of them in 10 categories** (travel, work, regional, accounts, shopping, research, media,
local, home, money). 394 **look something up** and 45 **fill something in**, and those stop before a named
last button so you press it yourself. No playbook types a password, pays, trades or moves money; the money
ones only read.

**In Captain.** **Playbooks** in the navigation opens a store like the Agent store:

1. Search **What do you want done?**, or pick a category; **★ Mine** shows the ones you kept.
2. Open a playbook: a form made from what it needs, **How it goes**, and a note when it needs Captain for
   Chrome.
3. **Run in <agent>'s chat** opens the chat and sends the request in plain words.

**In Captain for Chrome** (0.9.0). Under the page card, **Do it here** chips show playbooks for the site in
front of you (up to 8, then **All N →**). **⚡** in the header opens the same store, with **On <site>**,
**My playbooks** and **Popular**; rows say **Looks it up** or **Fills it in**, **＋** keeps one, and
**Run with <agent>** starts it. What you kept is the same in both places.

**For agents.** The new `web_playbook` tool (`find`, `show`, `run`) is there wherever an agent has
`browser`, so an agent asked to do something on a site looks for a playbook first. Each playbook says
which browser it needs: your Chrome, your Chrome if it's there, or any. Your own
[recordings](https://github.com/kstevica/captain-claw/blob/main/docs/guide/web-in-your-chrome.md) count as
playbooks too. While a fill-in playbook runs, the browser tool refuses a click on its stop button for the
rest of the turn. It can't catch a click by coordinates or Enter in a field, so read what an agent is about
to submit.

### A robot check goes to you, then the agent goes on

When an agent's `web_search` or `web_fetch` in your Chrome meets a "prove you're human" page, the tab now
comes to the front instead of closing (#766). The chat says it's waiting and the panel shows **Waiting for
you**. Answer it and the search or read continues in the same tab: up to 5 minutes in a chat, 2 minutes
for a run that started on its own. Leave it and the agent falls back to the deck's own search or fetch;
close the tab and that site isn't tried in Chrome for 10 minutes. It knows Google's, Cloudflare's,
reCAPTCHA, hCaptcha and the other common checks. The `browser` tool in your Chrome isn't covered yet.

### Tables: Excel files, a filter on every column, a row in fullscreen

- **Excel files open as tables** (`.xlsx`, `.xlsm`), read-only, a tab per sheet, wherever a file opens:
  the File library, shared folders, chat files, pinned files, First Mate results. Old `.xls` files still
  download. A formula cell saved without its result shows empty.
- **Every column has a filter.** Press the funnel in a header: **Values** (tick them, with counts, **Show
  only** or **Hide**), **Range** (a slider with two handles and quick picks such as **Top 10**, **Above
  average**, **Last 30 days**) and **Condition** (contains, ends with, top or bottom N, appears more than
  once…). Filters add up, show as chips, and **Clear all** removes them. In the Data library the deck
  does the filtering, so it works on big tables.
- **A row in fullscreen**: the Data library's row panel and new-row form open full screen, as cards in two
  columns you edit in place.
- **Descriptions that arrive** (#748): the Data and File library's descriptions are written with thinking
  off, so local thinking models stop failing with "did not answer with JSON".

### Connections

- **An MCP server can ask you something** (#749). When a local (stdio) MCP server asks for input in the
  middle of a call, it opens as a card for you, in every tab and the bell, with **Decline** and **Send**.
  The card says **Only you can answer this**, and the model never sees what you type. A question goes out
  only when every call running on that server is for one person who is there. Agents' MCP calls no longer
  fail after about 10 seconds: the deck stops one that's quiet for 10 minutes, and a card nobody answers is
  cancelled after 20.
- **Pick single files from Google Drive** (#763). **Connections → Google → Files from your Google Drive →
  Pick files from Google Drive** opens Google's own picker, so your agents can open those files and no
  others. An admin sets it up once; see [Upgrading](#upgrading). It's hidden where full Drive access is
  already connected.
- **WhatsApp Web says whether a message arrived** (#750). **Sent by your agents** and **Auto-reply
  activity** show **not delivered yet**, **delivered**, **read**, **not delivered**, **phone asked for it
  again** or **WhatsApp turned it away**. The bell rings once when no phone confirms within 10 minutes.
  Agents' replies are signed once, not twice.

### First Mate

- **The list funnel is a new factory's runner** unless the draft says its result isn't a list (#747). The
  picker shows **List funnel** first, then **Agents**. Factories you already have keep theirs.
- **Your agents can read a factory's runs** (#753): the `first_mate` tool's new `runs` action gives the
  newest eight runs with their outcome, how long they took and their result files. While a run is working,
  its folder is read-only for your other agents.
- Layout fixes in a narrow drawer, on a phone and in the **Recipe** tab (#746).

### Chat and kiosk

- **The first next step waits in the box** (#758): it shows greyed in the empty chat box, and **Tab**
  writes it in.
- **The chat box has the cursor when you come back** to the tab or window in the kiosk (#755).
- **Agents know their groups** (#754): an agent is told which agent groups it's in and who else is, and can
  ask or task a whole group by name.

### Cognitive modes, rewritten

All seven mode prompts are new: short, not tied to any kind of work, and under one shared frame that keeps
the task above the mode and never changes what counts as done (#770). Each mode also gives reviewers one
line on what to look for. Modes no longer decide how hard an agent pushes, that's Drive now. Agents on the
small prompts (Eco included) run with no mode. The mode is recorded in the usage and turn logs, with
`GET /fd/taskboard/mode-stats?days=30` (#771).

### Fixes

- **A scheduled run that skipped its tools has to use one** (#765). A run that called none of the tools its
  task names is retried with a tool call required, instead of being corrected in words. A report that
  still goes out marked "nothing was checked" counts as turned down: it stays out of memory and of later
  runs' history. This stops small local models from sending made-up "live check" reports.
- **No claims of files, pages or scripts the turn didn't get to** (#760). A reply naming a file that doesn't
  exist and no tool made gets one corrective retry. OCR says which pages it read. Agents with no shell
  aren't told to run one. Office tools can read their own `.bak` backups and refuse to change them.
- **A file shown in a reply isn't run as a command** (#779). An unlabelled code block was sometimes run in
  the shell, failed, and replaced a correct answer with "No file was created".
- **A reload no longer saves the whole conversation again** (#759), which duplicated rows in long chats and
  after scheduled runs. Rows already duplicated stay.
- **"new topic" typed on its own isn't sent twice** after a reconnect (#780).
- **A deck on a host without Docker no longer runs out of open files** (#757). It also raises its own
  open-file limit at start.
- **Prompt caching survives restarts** (#762): the front of a request stays the same between turns, after
  an agent restarts and when it wakes.
- **The deck's background calls run with thinking off** and with room to finish (#756).
- **Scheduled tasks pause after the longer of the admin's days or twice the task's own period** (#761), so a
  weekly task isn't paused after one week away. Usage on a person's own API keys is counted as theirs.
  Newer Claude models no longer get a `temperature` they reject.
- **Desktop installers say which CPU they're for** and check it (#743): `Captain-Setup-<version>-x64.exe`,
  `Captain-<version>-x64.dmg`.

## Upgrading

1. **Update the code** (`git pull` + `pip install -e .`, or **Console → Updates**; installed desktop apps
   update themselves), **restart Flight Deck, then every agent**, and **hard-refresh** every browser tab. No
   new Python dependencies.
2. **Change the secrets in the deck's `.env`** if the deck was reachable from outside your network; see
   [Security](#security-update-now). After the update, a request path with `..` in it gets a 400.
3. **Reload Captain for Chrome** (0.7.2 → 0.9.0): **Connections → Browser** → download it, unzip it over
   the folder Chrome loaded, then **Reload** in `chrome://extensions`.
4. **Agents in the Ionian or Phrygian mode with no Drive default now run at Persistent**: longer turns and
   one more model call per turn that used tools. Set **How far it pushes** to **Diligent** on an agent to
   keep it as it was.
5. **At Ambitious, extra work is on.** Switch **At Ambitious, add the obvious next step** off per agent, or
   set `drive.initiative: false`.
6. **Every agent with a cognitive mode gets new mode text** at its restart. Locrian gets the usual two
   retries (it had one), and agents on the small prompts run with no mode.
7. **WhatsApp Web**: turn it off and on for the deck after the restart, so delivery is tracked. Earlier
   messages show as untracked.
8. **Google Drive picker** (optional, admin): enable the Google Picker API in your Google Cloud project,
   create a browser API key, add the deck's address to the OAuth client's **Authorized JavaScript
   origins**, and paste the key into **Picker API key**.
9. **New factories run on the list funnel.** Existing factories and the built-in templates keep their
   runner.
10. **Each agent misses its prompt cache once** after the update. `llm_usage` gets its new columns by
    itself; there are no migrations to run.
11. New in the config: `drive.default`, `drive.initiative`. No new environment variables to set.

## Pull requests

- **Security**: #764
- **Drive**: #772–#778, #781, #782
- **Web playbooks**: #767–#769
- **Captain for Chrome**: #766
- **Tables and data**: #748, #751, #752
- **Connections**: #749, #750, #763
- **First Mate**: #746, #747, #753
- **Chat and kiosk**: #754, #755, #758
- **Cognitive modes**: #770, #771
- **Fixes**: #743, #756, #757, #759–#762, #765, #779, #780

Full list: [github.com/kstevica/captain-claw/pulls?q=is:merged](https://github.com/kstevica/captain-claw/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-11-22..2026-11-24)
