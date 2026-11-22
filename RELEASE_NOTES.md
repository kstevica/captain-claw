# Captain Claw v0.10.0 Release Notes

**Release title:** Landfall
**Release date:** 2026-11-22

Your agents now finish what they start. **First Mate** runs standing jobs for you, each on a charter you
approve, with goals every run is checked against and a weekly budget in tokens. The runs underneath keep
going while they make progress, save as they go, pick up where they stopped, check themselves between
steps and know exactly what is still missing. Long lists finish too, through a **list funnel** that lets
code do the searching and checking and asks a model once per item. And your agents can now **search the
web and read pages in your own Chrome**, signed in as you, so a LinkedIn profile, a post on X or a page
that turns servers away reads the way it does for you.

164 pull requests, #578 to #742. **Restart Flight Deck, then every agent, hard-refresh the browsers, and
reload Captain for Chrome (now 0.7.2).** First Mate is off until an admin turns it on, a browser tab now
opens the kiosk or the Admin console, shared folders stay in the File library only when someone added them,
and scheduled tasks pause for people away a week; see [Upgrading](#upgrading).

Step-by-step guides:
**[First Mate](https://github.com/kstevica/captain-claw/blob/main/docs/guide/first-mate.md)** ·
**[Web searches and pages in your Chrome](https://github.com/kstevica/captain-claw/blob/main/docs/guide/web-in-your-chrome.md)**

## New

### First Mate: standing jobs that run for you, checked every time

**First Mate** runs **factories**: standing jobs under your own account, each on a charter you approve,
with goals it's checked against, when it runs, a cap per run and a share of a weekly token budget. Every
run is a Bat run (or a list funnel, below), so it plans, works in steps, checks itself and is judged by
reviewers before it counts as done.

**Turning it on.** First Mate is **off on a new or upgraded deck**. An admin opens **Console → Settings →
First Mate**: **On for this deck**, **Admins only** (on by default: switch it off to give First Mate to
everyone signed in) and **Limits** (**Factories per person**, 8; **Runs per factory a day**, 24). People find
it in the kiosk under **Settings → Your agents at work → First Mate**, and in the agent column's **More**.

1. **New factory**, then one of three ways in:
   - **Describe it**: write what you want, optionally tie it to a Course goal, and press **Draft it**. With
     **Rewrite it as a full charter.** on (the default), First Mate spells out what a run needs to be told:
     groups with an amount each, the columns, where each fact must come from, what to do when one isn't
     found. It keeps every limit, number, name and link you gave. Off: your words become the charter as
     written.
   - **Or start from a template**: **Inbox drafts**, **Follow-up chaser**, **Weekly report**,
     **Price watch**, **Meeting prep** or **Research digest**.
   - **From scratch**: a name and a charter.

   An agent can draft one for you with the new `first_mate` tool; it arrives on your Line as "A new factory
   to approve" with **Approve and start**.
2. **Read the charter and approve it.** **What it does, each run**; **Done when**, a checklist of goals you
   switch on or off, each marked **Countable** (Captain checks it itself, with a script) or **Judgement**
   (the reviewers decide); **When it runs** (**Only when I press Run now**, **Weekdays 08:00**, **Every day
   08:00**, **Mondays 09:00**, or your own). Then **Approve and start**. Nothing runs on a charter you haven't
   approved, and approval is for the exact text you saw.
3. **Run now**, or **Run with data…** to give that run files, a paste, links, an agent's files or a
   conversation. **Data** gives every run the same extra data, up to 200 files.
4. **Let it run on its own** with **On their own** ("Run my factories on their own"). It starts **off for
   everyone**, so schedules and new mail start nothing until each person turns it on.
5. **Watch a run.** On **Runs**, a run opens to "N of M steps done", the step it's on, and **Steps · Agents ·
   Activity · Files · Notes · Extra data**. Its workers show in your agent column as one line,
   "<factory> · Working · N workers", not as a crowd of anonymous agents.
6. **Read the result.** **Results** puts the result first: a table opens as a table, with **Open table**,
   **Download**, the whole run as a zip and **Add to Library**. A box **Worth a look before you use it** lists
   failed checks, what Captain found reading the files, and any reviewer who objected.

- **Budget, in tokens.** **Your settings → Weekly budget** (5M to 250M a week, 25M by default, or **No weekly
  limit** for local models) is shared between your factories. **Cap per run** (3M by default; 1M to 50M, or
  **No cap**) changes in place, with no new charter version, and a run going now follows it from its next
  step.
- **Continue.** A run stopped by its cap or by you goes on where it stopped; finished steps stay finished.
- **Never stuck waiting.** A factory run doesn't stop to ask you questions halfway (you approved the
  charter), and its grants never widen.
- **Nothing goes out unasked.** While a factory's trust is **Asks first**, a run that would send mail is a
  rehearsal: "It would do this. Nothing of it is done until you say so", then **Carry out all, once** or
  **Reject**. Each carry-out is checked afterwards against the tool-call register and Gmail or Calendar.
  Trust rises to **Acts within its grants** once its checks agree with your labels and its results are
  good; a doubtful carry-out lowers it.
- **Its record.** **Record** shows satisfaction, runs this week, doubtful carry-outs and tokens per run, with
  a health light (**Healthy**, **Needs a look**, **In trouble**); 👍 and 👎 on runs feed it. **Checks** holds
  sealed test cases a new charter is checked against before it goes live. Admins can start **Improvement
  runs**: Captain Swarm tries better charters on those cases, and a kept gain comes back as a draft for you
  to approve.
- **On your Line.** One thread per factory: done, failed, carried out, would do N things, paused, version N
  is live. A finished run's message starts with its result file and size. **First Mate's week** arrives
  every Monday morning, and a factory that fails three runs in a row pauses and tells you.
- **Archive** factories and runs: off the lists, kept whole, and **Restore** brings them back.
- **Its agent.** **Your settings → Its agent** picks the agent First Mate drafts with, or **Make one**;
  without one, drafting uses your tier set.

### Runs that finish their goals

Bat, the run mode that doesn't stop until an independent judge says the work is done, is what every First
Mate run uses, and it got most of this release's work. All of it applies to every Bat run, whether it
starts from the Bat page, the `bat` tool or a factory.

- **It keeps going while it makes progress.** A worker step no longer ends at a fixed step count or after
  30 to 60 minutes. Progress extends it (up to 5,000 steps), and only a long silence (15 minutes: a hung
  model call) ends it. A context overflow is trimmed and retried instead of ending the step.
- **It saves as it goes, and resumes instead of restarting.** Every step keeps its results in the run's
  folder and a progress note (done, left, what counts). A step that runs again after Continue, a deck
  restart, a timeout or a failure starts from its note and its files, and an interruption doesn't use up
  its tries.
- **Plans sized for the job.** Many items of one kind are split by batch or segment, up to 12 steps. The first
  step fixes the columns or outline for all of them, the last one only puts the parts together, and a step
  that runs out of room is split rather than retried whole. If the planner fails twice, a three-step plan
  stands in, and says so.
- **Checks between steps.** Each finished step is checked by scripts against your rules, the run's columns
  and the other parts, at no token cost. A short review on a soft signal says whether the run is on course,
  drifting or off, and its corrections reach every later step. A step that came back short while there was
  more to find goes round once more, and "up to 15" is read as a target of 15.
- **It reads what it delivered.** Before the verdict, Captain profiles the main files: a table's rows, broken
  rows, duplicates, empty cells and sources; a document's words, sections and placeholders; code's syntax
  and tests. Goals a script can decide become checks of their own: rows, columns, unique values, rows per
  group, words, headings, no placeholders.
- **It knows what's short.** A run that stops short plans only for what's missing ("9 of 15, 6 more"), and a
  run whose result is still growing isn't stopped for lack of progress. A stopped run starts its result
  with what it made before it stopped.
- **A cheaper tier for the bulk.** **Console → Settings → Bat → Workers use** lets a strong model plan and
  check while a cheaper tier does the steps. The same card sets **Steps at the same time**, **Step ceiling
  per worker turn**, **Quiet minutes before a step stops** and **Minutes one model call may take** (20, so
  thinking models aren't cut off at 3 minutes). A step can also be handed to Vatra, Basna, Code or Crew.
- **Token guards.** A run with no cap stops at 20 million tokens and can be continued, and each agent step
  gets its own share.
- **Workers clean up after themselves.** A run's workers are removed when it ends, and leftovers when the
  deck starts.

### The list funnel: long lists that finish, for a fraction of the tokens

A factory whose result is a list ("small consulting firms in a region, each with a contact email",
"restaurants in a city by cuisine", "every report in this folder that meets these rules") used to be worked
by agents in long turns that re-read everything on every step. The **list funnel** splits the work
differently. **Code** searches, fetches pages, filters, merges duplicates and checks every row, and a model is
asked **once per item, in a short fresh call**: keep, came close, or drop. Every value on the list must be on
a page or in a file that was actually read.

- **Getting one.** **New factory → Describe it → Draft it**: a task whose result is a list comes back with
  **List funnel** as its runner. You can also pick it in the draft under **How it runs**: **Agents** ("for
  research, documents, anything that isn't a list") or **List funnel** ("far fewer tokens for lists, files
  you give it included").
- **Any list, anywhere.** About 70 countries and any kind of item: companies, restaurants, datasets,
  profiles. Items come from web searches (list and directory pages in the results are followed), a list
  page, a links file, or your own files: tables (CSV, TSV, XLSX, JSON lines) and documents (PDF, Word,
  PowerPoint, Markdown, HTML). A factory and a run each take **up to 200 files** of extra data (was 30 and 60),
  and a pattern like `extra/report_*` stands for many. A list built only from files never searches the web.
- **Your material first.** A run reads the links in the charter and in your attached documents and writes a
  short brief that the rest of the run works from.
- **Must and prefer rules.** A **must** rule says what the item *is* (kind, place, size, how to reach it) and is
  never bent. A **prefer** rule asks for evidence many good items never publish (recent news, a rating) and
  has a **looser reading** that fills a short group after every strict row. A fact that isn't published isn't
  a broken rule: to drop an item, the run must quote what breaks the rule. The charter's own "if there's no
  rating, judge it" is applied as written.
- **Came close.** Items that fit except for a missing fact go into a file of their own, with why each fell
  short and the pages read, instead of being lost. Before anything is loosened, the run **goes and looks**:
  up to four small actions (a page, a search, one of your files), then the item is judged again.
- **An expert beside the run.** Pick one of your own agents as the **Expert agent**. It adds what it knows
  before the run starts, settles what the charter leaves open as you would, writes the new searches for a
  short group, writes the "why it fits" columns of kept rows, settles near misses that are a matter of
  judgement (never a missing fact), and checks a sample at the end. Ticked, **It may also turn down a row…**
  lets it veto rows, with its reason. Everything it decided is in the run's summary.
- **What you see.** On **Runs**, a strip per group ("N of M ready (K looser) · L looked at") with a bar of
  where items went. On **Results**, **the list leads**, with **Open table** and **Download**, then **Came
  close**, then the summary and the dropped items, under the file names the charter uses; a stopped run
  writes them all too. **Data → Already checked** keeps what earlier runs checked so later runs skip it
  (**Forget** clears it); it works for agent-run factories too. The **Recipe** tab shows what every run of
  the charter follows (rules with **Must / Prefer**, **If the fact isn't found…**, the looser reading, the
  columns a row must fill) and lets you edit it.
- **Scale.** 1,000 searches per run, or 12 per row asked for when that's more; set your own in **Searches a
  run may make**. **Record this run** keeps every search and page so the run can be replayed offline as a
  test set and scored.

### Web searches and web pages in your own Chrome

Two of your agents' everyday tools can now run in **your own Chrome**, through Captain for Chrome, in a
background tab you never have to look at:

- **`web_search` on a normal search engine** (Google, Bing, DuckDuckGo, Yahoo or Qwant) instead of the deck's
  paid search service. Google and Bing give up to 50 results a search.
- **`web_fetch` reading pages signed in as you.** LinkedIn, X, Facebook and other sites that turn a server
  away (a sign-in wall, a bot check, LinkedIn's "999") open for your agent the way they open for you.

**Turning it on.** Both are off until you choose:

1. **Flight Deck → Profile → Personal** (the kiosk's Profile too):
   - **Web searches → Search through my Chrome with**: **Google**, **Bing**, **DuckDuckGo**, **Yahoo** or **Qwant**.
   - **Web pages → Read pages through my Chrome**: **Only pages that turn Flight Deck away** or **Every page**,
     and optionally **Only these sites** (a site covers its subdomains; up to 50).
2. **Captain for Chrome 0.7.2**: in its Settings, turn on **Work on every site without asking** (or press
   **Allow** on the card the panel shows). Web searches and web pages don't need **Agents can use this
   browser**; that switch is only for agents driving your Chrome with the `browser` tool.

**For LinkedIn and X**, set **Web pages** to **Every page** with **Only these sites** =
`linkedin.com, x.com, twitter.com`, and **Web searches** to **Google**. Those sites are then always read as
you; every other page is read by the deck as before. Then ask, for example:

- "Read `linkedin.com/in/<profile>` and summarize their experience and education."
- "Search Google for `site:linkedin.com/in "head of finance" fintech <city>`, list 20 profiles with name,
  title and link, then read the top five."
- "Read this post on X and its replies, and summarize the argument and the main objections."
- "Every weekday at 8:00, check `x.com/<handle>` and tell me what they posted since yesterday."

**How it reads.** A page opens in an inactive tab in a collapsed group (**Captain · search** or **Captain ·
reading**) and closes when done. Captain waits until the page has **finished drawing**, not just loading,
then scrolls through up to eight screens so sections that load as you scroll (a profile's Experience and
Education, a feed, replies) are there too. The agent's result says "Read in the person's own Chrome, signed
in as them". The `browser` tool in your Chrome gets the same waits, and keeps working on pages where another
extension added a frame.

**When Chrome is closed** or the setting is off, searches go to the deck's search service and pages are read
by the deck, as before, without a word. A robot check or a cookie question on the search engine falls back
too, with a note asking you to answer it once in Chrome.

**Good to know.**
- Your agents see what you see on those sites, and searches show in your browser history and your Google or
  Bing activity. List only the sites you need.
- Only your own agents' turns use your Chrome, never a member's turn on an agent you share; scheduled runs
  use it whenever it's open.
- Private, local and intranet addresses are always refused, and an agent can be kept out with
  `tools.web_search.via_chrome: false` or `tools.web_fetch.via_chrome: false`.
- `web_fetch_batch` and `web_get` still read on the deck; ask for pages one by one to use your Chrome.
- A very long thread or feed may come back partial: the read covers about eight screens. For more, ask the
  agent to open it in your Chrome with the browser and scroll.
- Brave Search hides its result links until a real click, and Startpage and Ecosia answered with bot walls,
  so they aren't offered.

### Recordings in Captain for Chrome: show it once, an agent does it after that

**Record what I do** in Captain for Chrome's panel records a process on a site, step by step in words
(“Click the button “Delete” in the row …”). Add an instruction to any step, then **Stop and review**: name
it, say what it's for, reorder or remove steps, add one in your own words. Each value you typed or picked
gets **This value**: **Always the same**, **Ask me before it starts**, **Ask me when it gets there** or **The
agent decides**. Passwords, one-time codes and card fields are never recorded; the agent asks for them when
it gets there.

**Recordings → Run** picks **Who does it**, fills the values, takes **Anything different this time?** and
extra data for every run or this run only, then **Run with <agent>**. The agent works in its own tab group in
your Chrome and stops to ask when a step doesn't fit the page. **Earlier runs** keeps the history. Flight Deck
shows your recordings, their extra data and their runs under **Connections → Browser → Recordings**; runs
start from Chrome.

Captain for Chrome also tells you when Flight Deck has a newer copy (**Download**, **Reload**, **Later**), and
**Connections → Browser** shows each browser's version.

### Agents that work as a family

- **Parents and children.** An agent made by another agent sits under it, and agents with the same parent are
  siblings. The agent column shows the tree (fold it, **⋯ → Put under**, or hold an agent over another's row
  to move it, with **Undo**), and the Agents page shows family cards. Run workers (Bat, Vatra, Basna, Council,
  Code) no longer show as your own agents: a parent shows "Working · N workers".
- **They know each other.** Each agent knows its parent, siblings and children, and messages say who's
  talking. The `flight_deck` tool can **report** to the parent, and **ask** or **task** the parent, the
  children, the siblings, the whole family or an agent group at once, with the answers coming back together.
- Removing or powering off a parent asks what happens to the agents under it, and **Info → Family** shows
  where an agent sits, who made it and why.

### Safer agents

- **Approval cards** before an agent pays, sends, posts, shares, deletes or stores a password. An admin sets a
  policy per kind under **Admin console → Settings → Approvals** (**Don't ask**, the default, **Ask**, **Ask
  every time** or **People do it themselves**). The card offers **Allow once**, **Allow for this
  conversation**, **Always allow here** and **Don't allow**, and **Profile → Approvals** lists what you
  always allow, with **Ask me again**.
- **Typing a secret without telling it.** The new `request_secret` tool opens a masked box in Flight Deck for
  a password or a one-time code. The agent only gets a reference for the browser to type, and the value is
  scrubbed from what the browser reports back.
- **Its browser**, in an agent's menu, shows its headless browser live. **Take control** for a CAPTCHA or a
  sign-in, then **Hand it back**, and the agent is told what changed.
- **Why did this happen?** on any call in **Activity** (or the console's **Audit**) replays the approval and
  guard decisions on the recorded call, with no model call, and marks what would go the other way today.
- **Scheduled jobs that do nothing are caught.** A run that called no tool and only announced work counts as
  "did nothing", a job pauses after 10 bad runs in a row, new jobs that run the model repeat at most every 15
  minutes (20 jobs per agent), missed daily or weekly runs more than 2 hours late are skipped, and a job that reached its end
  stops itself.
- **Needs you means you.** It holds only what's blocked on you: a question, an approval, a Go, a decision.
  News is a report. A Bat run started from chat gets a live card with **Stop**, **Try again** and **Open**.

### Leaner agents

- **Tool search.** An agent no longer sends every tool's full description on every call. Past 16,000 tokens
  of descriptions, the rest are listed by name and loaded with `search_tools` when needed; on Claude models it
  uses Anthropic's own tool search, so the prompt cache survives. Agent **Settings → Tool search**: on or off,
  with a limit of 8k, 16k, 24k or 32k, applied from the next message.
- **An Eco mode switch** in the kiosk's agent settings, which now reaches agents in containers too.
- **Automations stop burning tokens.** An agent's own scheduled prompts run in a session of their own
  ("⏰ <task>") instead of the chat they were made in, so a run doesn't carry the whole chat; results still
  land in that chat. Eco keeps its tools steady per session, and a run that claims a check without calling the
  tool it names is sent back once.
- **Quieter side calls.** Reflection, insights, naming and the rest run with thinking off where the model
  allows it, and on a self-hosted model server they wait until the agent's run is over, so its prompt cache
  survives.
- **Only nudges that can't wait reach your phone.** Every nudge still reaches the chat, the Line and the bell;
  WhatsApp and Telegram get only the ones worth a ping (**Connections → Notifications**).

### Chat

- **One conversation bar** over the chat in the simple layout: the agent, a switcher **A · <topic>** for lanes
  and topics, **New topic** and a **Queue** pill.
- **Pictures** you send show as pills with thumbnails, preview inside the message, and open in a viewer you can
  step through. Files you attach show as cards in the message box.
- **Next steps** appear under the reply as numbered pills.
- **Course** has a **Needs you** band on top, goal rows with a progress ring and a goal drawer.

### Teams, Crew and models

- **A browser tab opens the kiosk** for people and the **Admin console** for admins; the classic layout only
  with `?app=classic`.
- **Team models stay put.** A non-admin's agent on the team's key or tiers keeps its configured model, with
  **Use my own API key instead** in its settings; admins and people on their own keys switch freely.
- **Crew**: attach your agents' files and stretches of your conversations with them to a task, and let
  approved people run **hosted apps** (an admin ticks **Hosted apps** on their Crew approval), within limits.
- **Computer User**, a new archetype and the fifth Featured pick in the Agent store, uses your browser and
  computer like a person until the goal is reached. 132 archetypes.
- **Scheduled tasks pause for people away a week** (**Console → Settings → Agents**), with a note on their
  Line, and admins see only their own scheduled tasks.

### Files and data

- **`web_fetch` keeps a page's structure**: headings, lists and tables come back as Markdown, so a price list
  or a directory reads as a table, not one cell per line.
- **One Add data panel** for factories, runs and Crew: this computer, a paste, links (paste many), agents'
  files, Flight Deck folders, a conversation.
- **A run's CSV and TSV files open as a table** in First Mate that you can sort, search and edit, up to 50,000
  rows.
- **Shared folders are in the File library only when someone adds them** (**Add to Library**, **Add folder to
  Library**); agents' own files are listed as before, and the Data library still lists every folder's tables.
- **Datastore** writes land in one call: `insert` names unique-key clashes and takes `on_conflict`, and
  `upsert` reports what it inserted and updated.
- 83 built-in tools (`first_mate` and `request_secret` are new; 82 in the desktop app).

### Fixes

- An oversized tool result is capped with a pointer the agent can page, instead of vanishing.
- Summarizing many files keeps the chat connected and finishes its summaries.
- Guards no longer read a hypothetical "had I…" as a claim, and argument errors list what's required.
- A Google sign-in Google refused for good isn't retried every minute: one **Reconnect Google** note instead.
- The update notice goes out once per version, then weekly.
- Mail refusals come once per turn in full, IMAP tells a made-up message ID from a renumbered folder, and
  DeepSeek's broken tool-call markup runs.
- A WhatsApp "link dropped" warning no longer fires for an old command, and the Line agent's own nudges aren't
  posted twice.
- Assembling a Council works again.
- Flight Deck keeps a rotating log file under its data folder.

## Upgrading

1. **Update the code** (`git pull` + `pip install -e .`, or **Console → Updates**; installed desktop apps
   update themselves), **restart Flight Deck, then every agent**, and **hard-refresh** every browser tab. No new
   Python dependencies.
2. **Reload Captain for Chrome** (0.3.0 → 0.7.2): **Connections → Browser** → download it, unzip it over the
   folder Chrome loaded, then **Reload** in `chrome://extensions`. From now on the panel says when it's behind.
   No new permissions; reading pages for agents needs **Work on every site without asking**.
3. **First Mate is off** until an admin turns it on (**Console → Settings → First Mate**), and then for admins
   only until **Admins only** is switched off. Each person's **On their own** starts off, so nothing runs on a
   schedule until they turn it on. The first weekly report comes the Monday after.
4. **A browser tab opens the kiosk** (or the Admin console for admins). The classic layout is at
   `?app=classic`, and Payments review links need it too.
5. **Shared folders leave the File library unless someone added them**, at its next scan;
   **Add folder to Library** brings one back.
6. **Scheduled tasks pause for people away 7 days or more**, on the first pass 10 minutes after the deck starts,
   with one Line note each. Change the days or turn it off first under **Console → Settings → Agents**. Admins
   stop seeing other people's scheduled tasks.
7. **Agents' own scheduled prompts move to their own "⏰" session** on their first run. Jobs that do nothing
   count as failures and pause after 10; new jobs that run the model repeat at most every 15 minutes, 20 jobs per
   agent; existing jobs keep running.
8. **Bat runs behave differently**: longer steps, plans up to 12 steps, checks between steps with short reviews
   (counted in the run's spend), and runs with no cap stop at 20 million tokens (`CLAW_BAT_TOKEN_CEILING`, 0 =
   off). Steps already running at the upgrade keep their old limits. Run workers leave people's agent lists.
9. **Tool search is on** for agents past 16,000 tokens of tool descriptions (`tools.tool_search`; 0 turns it
   off).
10. **Fewer nudges reach phones** (turn **Only nudges that can't wait reach your phone** off for the old way),
    and **Needs you** holds only what's blocked on you. Copy says **Tasks** (was Taskboard), **Needs you** (was
    Needs me) and **automations** (was cron jobs).
11. **Non-admins on the team's key or tiers can't switch models**; a conversation already switched goes back at
    its next turn.
12. **`web_fetch` output is Markdown** (headings, lists, `| a | b |` tables). **Datastore** `update` with no
    `where` on a table of two rows or more is refused (`{"_all": true}` still works).
13. **Approvals, Chrome searches and Chrome page reads are opt-in**: nothing changes until an admin or a person
    turns them on.
14. New deck settings: `FD_FIRST_MATE_DISABLED`, `FD_FIRST_MATE_TICK_SECONDS`, `FD_BAT_STEP_CEILING`,
    `FD_BAT_IDLE_MINUTES`, `FD_BAT_CALL_MINUTES`, `FD_BAT_WORKER_TIER`, `FD_LOG_FILE`, `FD_ACCESS_LOG` and the
    `FD_CREW_APPS*` limits; see [USAGE.md](https://github.com/kstevica/captain-claw/blob/main/USAGE.md).

## Pull requests

- **First Mate**: #612, #613, #615–#622, #629–#633, #638–#641, #644, #645, #647, #650, #651
- **Runs that finish (Bat)**: #634–#637, #642, #643, #646, #648, #652–#655, #657, #667, #682
- **List funnel**: #668–#681, #683–#687, #689, #691–#694, #696–#713, #715–#719, #721, #723–#726, #728–#732, #734
- **Web searches and pages in your Chrome**: #738–#742
- **Recordings and Captain for Chrome**: #688, #690, #695, #714
- **Agent family**: #623–#628
- **Safer agents**: #581, #582, #584–#589, #601–#605
- **Leaner agents and tool search**: #590–#594, #597–#599
- **Chat**: #595, #596, #614, #733, #735, #736
- **Teams, Crew, models**: #608–#611, #658, #663, #664, #666, #720, #727
- **Files, data, fixes**: #578–#580, #583, #600, #606, #607, #649, #656, #659–#662, #665, #722

Full list: [github.com/kstevica/captain-claw/pulls?q=is:merged](https://github.com/kstevica/captain-claw/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-11-15..2026-11-22)
