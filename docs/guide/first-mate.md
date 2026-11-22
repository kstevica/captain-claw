# First Mate — standing jobs that run for you

[← README](../../README.md) · [All guides](README.md)

New in 0.9.2. First Mate runs **factories**: standing jobs under your own account. Each one works on a charter you approve, is checked against goals you choose on every run, starts when you say, and spends from a weekly budget in tokens. You set one up once, and it keeps doing the job: drafting replies every weekday morning, building a list of leads, writing a weekly report.

Every run is a [Bat](flight-deck.md#seven-orchestration-modes) run or a **list funnel**. A Bat run plans the work, does it in steps, checks itself and is judged by reviewers before it counts as done. A list funnel is the cheaper way to build a list of rows.

**On this page:** [What it is](#what-it-is) · [Turn it on (admins)](#turn-it-on-admins) · [Where to find it](#where-to-find-it) · [Your first factory](#your-first-factory) · [Read the charter and approve it](#read-the-charter-and-approve-it) · [Run it](#run-it) · [Budget and cap per run](#budget-and-cap-per-run) · [Watch a run](#watch-a-run) · [Read the result](#read-the-result) · [List funnels](#list-funnels) · [Mail and trust](#mail-and-trust) · [Record and checks](#record-and-checks) · [On your Line](#on-your-line) · [Archive](#archive) · [Asking an agent](#asking-an-agent) · [Limits and good to know](#limits-and-good-to-know)

## What it is

- **A factory** is one standing job, such as "Inbox drafts" or "Weekly report".
- **Its charter** says what it does on each run and what it must never do. Nothing runs on a charter you haven't approved, and you approve the exact text you saw.
- **Its goals** (**Done when**) are what every run is checked against: "at least 100 rows", "no repeated email", "a reply for every client who asked".
- **When it runs**: only when you press **Run now**, on a schedule, or when new mail or a calendar change comes in.
- **A run** is one go at the job. It leaves a result (a table, a document, drafts in your mailbox) and a record of what it did.
- **The budget** is in tokens, never dollars: a weekly budget shared by your factories, and a cap per run.

Each person sees only their own factories. Admins don't see other people's factories either.

## Turn it on (admins)

First Mate is **off** on a new or upgraded deck.

1. Open the **Admin console** (the shield button), then **Settings**, and scroll to the **First Mate** card.
2. Switch on **On for this deck**. While it's off, nobody but admins sees First Mate, and nothing runs.
3. Leave **Admins only** on to try it yourself first. Switch it off to give First Mate to everyone signed in. **Improvement runs** stay with admins either way, like Captain Swarm.
4. Under **Limits**, set **Factories per person** (8 by default, 1 to 50) and **Runs per factory a day** (24 by default, 1 to 500), then **Save**.

The First Mate page has the same two switches for admins, under **This deck**: **First Mate on this deck** and **Admins only**.

A person who can't use it yet sees **Not on yet** with the reason: First Mate is off on this deck, or open to admins only for now.

## Where to find it

| You want to | Go to |
|---|---|
| Open First Mate in the kiosk | **Settings → Your agents at work → First Mate**, or **More** under the agent column, then **First Mate** |
| Open it from a Line item | **Open First Mate** on the item |
| Open it in the classic layout | Only in a tab opened with `?app=classic`: the sidebar's **Automation → First Mate** |
| Turn it on, or set limits (admins) | **Admin console → Settings → First Mate** |
| Change how Bat runs on the deck (admins) | **Admin console → Settings → Bat** |

The page opens as a wide dialog. The full-screen button gives it the window. At the top are four tiles:

- **Factories**: how many are live, and how many are running.
- **Runs going**: what is working now.
- **This week**: tokens used, out of your weekly budget.
- **On their own**: your switch for schedules and new mail (see [Run it](#run-it)).

Below them: **Needs you** (charters waiting for your approval, actions waiting for your yes, factories that paused), **Factories** with **New factory**, and **The week**. On the side, **Your settings** (**Weekly budget**, **Its agent**, **Its checks against your labels**) and, for admins, **This deck**.

## Your first factory

1. Click **New factory**. You get three ways in.
2. **Describe it.** Write what you want in your own words, with every limit that matters ("Never send anything", "only clients in our region"). If you have goals on your Course, the picker next to the button ties the factory to one (**Not for a goal** or **For: <goal>**). Click **Draft it**.
   - With **Rewrite it as a full charter.** on (the default), First Mate adds what a run needs to be told: groups with an amount each, the columns, where each fact must come from, and what to do when one isn't found. It keeps every limit, number, name and link you gave.
   - With it off, your words become the charter as written. First Mate still drafts the name, the goals and when it starts.
   - This browser remembers your choice.
   - One model call writes the name, the charter, its starts, its goals and up to three sealed test cases. A task whose result is a list of rows comes back with **List funnel** as its runner.
3. **Or start from a template**: **Inbox drafts**, **Follow-up chaser**, **Weekly report**, **Price watch**, **Meeting prep** or **Research digest**. Each starts on a schedule and may not send, spend or sign up for anything.
4. **From scratch**: click it, write a name ("Invoice chaser") and the charter, then **Create draft**.

A new factory starts as a draft. Nothing runs until you approve its charter.

**Its agent.** Drafting uses your tier set. To draft on one of your agents' models instead, pick that agent under **Your settings → Its agent**, or click **Make one** to make an agent called First Mate. You can talk to that agent like any other: it uses the `first_mate` tool (see [Asking an agent](#asking-an-agent)).

**From chat.** Ask an agent to set up a standing job for you. It drafts one with the `first_mate` tool, and your Line gets "A new factory to approve: <name>" with **Approve and start**, **Discard** and **Change it first**.

## Read the charter and approve it

Click the factory to open it. The **Charter** tab shows the draft as **Your charter** (or **Draft v2** for a later version), marked **Waiting for your approval**.

1. **What it does, each run**: the charter in plain words. Change anything you like.
2. **Done when**: the goals, as a numbered checklist, all on to start. Each goal is marked:
   - **Countable**: Captain can check it itself on each run, with a script (rows, columns, unique values, words, headings).
   - **Judgement**: the reviewers decide it.

   Untick a goal to leave it out; it stays on the list as **Off: not checked**. **All on** and **All off** switch every goal. **Add a goal…** adds your own, for example "at least 100 rows, no repeated Email".
3. **When it runs**: **Only when I press Run now**, **Weekdays 08:00**, **Every day 08:00**, **Mondays 09:00**, or your own in the field next to them ("every 6 h", "weekly mon,thu 07:30").
4. **How it runs**: **Agents** (workers search and reason in long turns: for research, documents, anything that isn't a list) or **List funnel** (see [List funnels](#list-funnels)).
5. **Save** keeps your edits ("Changes not saved" shows until you do). **Discard draft** throws the draft away.
6. Click **Approve and start**.

Once approved, the tab shows **Live charter · version 1** with what every run works to: **Done when**, **Starts**, **Cap per run**, **May act** (what it may do outside, or "Nothing outward: a run never sends, spends or signs up") and **Trust**.

To change a live factory, click **Change it**. That starts a new draft; the live version keeps running until you click **Approve this version**. A factory with test cases is checked on them before the new version goes live (see [Record and checks](#record-and-checks)). If it scores lower and stays a draft, **Approve anyway** is there.

## Run it

- **Run now**: on the factory's card, in its **⋯** menu, or on the **Runs** tab. On **Runs** you can add **A note for this run (optional)** first.
- **Run with data…** opens the **Add data** panel to give this run alone some files: **This computer**, **Paste**, **Links**, **Agents' files**, **Flight Deck** folders or **A conversation**.
- **Data** gives every run the same extra data, through the same panel (**Add to the factory**). Every run from then on gets a copy in its folder, and its workers read what bears on the work.

**On their own.** Schedules and new mail start runs only while the **On their own** tile is on ("Run my factories on their own"). It starts **off for everyone**, so until you switch it on, your factories run only when you press **Run now**. Switching it off again pauses your running factories.

> When you approve a factory from the Line, the card says "Approved. It runs on its schedule, and asks before anything goes out." That holds only while **On their own** is on.

**How schedules behave.** A schedule never fires at the moment you approve. A slot that comes while a run is still going is skipped, and so is one more than 6 hours late (its history says so). A factory that starts on new mail or calendar changes takes what came in over a short window (15 minutes by default, up to 50 events) as one run, not one run per email.

The **⋯** menu on a factory card has **Open**, **Run now**, **Pause**, **Stop**, **Resume**, **Archive** and, for drafts and retired factories, **Delete**.

## Budget and cap per run

**Weekly budget** (**Your settings**): 5M, 10M, 25M (the default), 50M, 100M or 250M tokens **a week**, or **No weekly limit**. It's shared between your factories, and the week starts on Monday. A factory that used its share shows **Over budget** and waits until the next week, or until you give the week more. **No weekly limit** is fine on a local model; on a paid API it can cost a lot.

**Cap per run** (on the **Charter** tab, and as **Each run may use** on **Runs**): how many tokens one run may use before it stops. The first choice is the cap it inherits, 3M by default ("your default"); the others are 1M to 50M, or **No cap**.

- Changing it needs no approval and no new charter version.
- A run going now follows the change from its next step.
- A new factory starts with the cap per run you picked last.
- A run with **No cap** still stops at 20 million tokens, and can be continued.

**Continue.** A run stopped by its cap, or by you with **Stop**, shows **Continue**: it goes on where it stopped. Finished steps stay finished, and it gets your cap per run again.

## Watch a run

On the **Runs** tab, click the arrow beside a run (a going run opens by itself). You see:

- "3 of 8 steps done", the percentage, and **Now** · the step it's on;
- how long it **Took** (or **So far**), how long its agents worked, and how long it waited for you;
- a bar with one segment per step;
- the tabs **Steps** · **Agents** · **Activity** · **Files** · **Notes** · **Extra data**.

**Notes** holds each step's **Progress notes**, what is **Shared by every step** and **Research notes**. **Files** opens what the run has made so far, like the File library; a CSV or TSV opens as a table.

A run's pills read **Starting**, **Running**, **Done**, **Failed** or **Stopped**, and **Rehearsal** or **Carry-out** for mail runs (see [Mail and trust](#mail-and-trust)). **Stop** stops a going run.

**In the agent column**, a factory's workers show as one line, "<factory> · Working · 3 workers", under the agent picked as **Its agent**, or at the foot of your list when there is none. They don't show as a crowd of separate agents.

## Read the result

The **Results** tab puts the result first. A table opens as a table.

- Each run's main file has **Open table** (or **Open**), **Download** and **Add to Library**. **Download all** gives every file of the run as one zip.
- Only what you add goes in the File library.
- **Other files** and **Working files** (scripts, temporary files, subfolders its workers used) are folded below.
- A box **Worth a look before you use it** lists goals the run failed, what Captain found reading the files (broken rows, duplicates, empty cells, placeholders) and any reviewer who objected. When nothing stood out, it says "Captain read the result: nothing stood out".
- "Captain corrected its course N times during the run" opens to show the corrections.
- A stopped run's result starts with **What it made before it stopped**, with Captain's counts.

The factory's card on the page shows the latest result as a chip ("Result leads.csv · 120 rows"); click it to open **Results**. Rate a finished run with 👍 or 👎 on **Runs** or on the Line: it feeds the factory's record.

## List funnels

A factory whose result is a list of rows ("small consulting firms in a region, each with a contact email", "every report in this folder that meets these rules") can run as a **list funnel**. Code searches, fetches pages, filters, merges duplicates and checks every row. A model is asked once per item, in a short fresh call: keep it, it came close, or drop it. Every value on the list must be on a page or in a file the run actually read. It costs far fewer tokens than agents working in long turns.

**Getting one.** **Describe it → Draft it** picks **List funnel** for a list task. You can also pick it in any draft under **How it runs**. Picked for a task that isn't a list, a run goes to agents anyway.

**Where items come from.** Web searches (list and directory pages in the results are followed), a list page, a links file, or your own files as extra data: tables (CSV, TSV, XLSX, JSON lines) and documents (PDF, Word, PowerPoint, Markdown, HTML). A pattern like `extra/report_*` stands for many files. A list built only from files never searches the web. A run first reads the links in your charter and your attached documents and writes a short brief that the rest of the run works from.

**Must and prefer rules.**

- A **must** rule says what the item is: its kind, place, size, how to reach it. It is never bent.
- A **prefer** rule asks for evidence many good items never publish, such as recent news or a rating. It has a looser reading.
- A fact that isn't published isn't a broken rule. To drop an item, the run must quote what it read that breaks the rule. The charter's own "if there's no rating, judge it" is applied as written.
- With the draft's box **When a group runs short, loosen the soft rules…** ticked (the default), rows that meet only the looser reading fill a short group, after every strict row. Not when the charter itself accepts a short list.

**Came close.** An item that fits except for a missing fact goes into a file of its own, with why it fell short and the pages read. Before anything is loosened, the run goes and looks: up to four small actions (a page, a search, one of your files), then the item is judged again.

**More settings under How it runs** (in the draft):

- **Searches a run may make**: left empty, 1,000, or 12 for each row the list asks for when that's more. A run stops searching when they're used.
- **Expert agent**: one of your own agents works beside each run, in a new session of its own. It adds what it knows before the run starts, settles what the charter leaves open as you would, writes new searches for a short group, writes the "why it fits" columns of kept rows, settles near misses that are a matter of judgement, and checks a sample at the end. It never supplies a fact. What it decided is listed in the run's summary.
- **It may also turn down a row the run would keep…** (shown once an expert is picked, off by default) lets the expert drop rows that plainly aren't what the list is for, with its reason.

**What you see.**

- **Runs**: a strip headed **List funnel** · N searches, with one row per group, "12 of 15 ready (2 looser) · 40 looked at", and a bar of where items went: kept, on a looser reading, came close, judged out, ruled out by code, duplicates and so on.
- **Results**: the list leads, then **Came close**, then the summary and the dropped items under **Other files**, with the file names your charter uses. A stopped run writes them too.
- **Data → Already checked**: what earlier runs checked, so later runs skip it. The ones that came close are checked again. **Download** gives the file; **Forget** clears it. This works for factories run by agents too.
- **Recipe** (list funnel factories only): what every run of this charter follows. Edit the **Rules** (each with **Must** / **Prefer**, **If the fact isn't found…** and its looser reading), **Add a rule**, switch the **Columns a row must fill**, remove filters under **Ruled out before reading**, and tick **A short list is fine when the summary says why…**. Then **Save**. **Build again next run** makes the next run write a new recipe. A changed charter or changed goals rebuild it too.
- **Record this run** (beside **Run now** on **Runs**) keeps every search and page the run reads, so it can be replayed offline as a test set and scored.

## Mail and trust

A factory may act outside only within its grants, shown under **May act** on the live charter: sending mail, organising mail, spending, opening accounts. Drafting adds mail grants only when your words ask for them, and never spending or accounts. The templates have none. A run never stops to ask you questions halfway (you approved the charter), and its grants never widen.

**Trust**, on the live charter, sets how much it may do without asking:

- **Asks first**: every outward action waits for your yes.
- **Acts within its grants**: it rehearses, then does what its grants allow, and asks about the rest.
- **Acts directly**: it acts without rehearsing. Only for reads, its own files and actions that can be undone, so it isn't offered to a factory that may send mail, spend or open accounts.

**Rehearsals.** While trust is **Asks first**, a run that would send mail is a **Rehearsal**: anything outward is recorded, not done. The run shows "It would do this. Nothing of it is done until you say so." with a tick beside each action, and your Line says "<name> would do 3 things".

1. Untick anything you don't want.
2. Click **Carry out all, once** (or "Carry out 2 of 3, once"), or **Reject**.
3. The **Carry-out** run does only that, once. Afterwards Captain checks it against the tool-call register and Gmail or Calendar: **Confirmed**, **Check it**, **Did nothing outside** or **Not checked**.

**Trust moves on its record.** Under **Your settings**, **Its checks against your labels** shows how far its checks agree with you (**Calibrated**, or a count towards it). Your 👍 and 👎, your approvals and what you do with its drafts are the labels. Once they agree and its last results are good, trust rises on its own. A doubtful carry-out lowers it. Tick **Pin it** to keep a level: its record won't raise it, but a doubtful carry-out still lowers it.

A factory whose rehearsal reached outside, or that had two doubtful carry-outs in two weeks, pauses and tells you.

## Record and checks

**Record** shows how the factory is doing:

- **How it's doing**: Satisfaction, Runs done this week, Doubtful carry-outs over two weeks, and Tokens per run, with a health light: **Healthy**, **Needs a look**, **In trouble** or **Too new to say**.
- **Trust**: its level and how far it is from the next one.
- **Its drafts in your mailbox**: sent as written, sent after edits, deleted, waiting.
- **What you did with its results**: your ratings, approvals and sent drafts, which set its score.

**Checks** holds its test cases: inputs it might get, each with what a good result does.

1. Click **Add a case**. Fill in **Name**, **What it gets** (the email, request or page, as the factory would see it) and **A good result (one per line)**.
2. Leave **Sealed** ticked: the case is never shown to the factory, or to anything that rewrites its charter, so it can't learn the answers. Untick it to make the case **Open**, which may be shown when the charter is reworked.
3. **Run the checks now** checks the live version. **Checks run** lists each version's score.

A run started by new mail has a flask button that makes a test case from what it got. When a live factory has cases, a new version is checked on them before it goes live, and goes live if it scores about as well as the version before.

**Improvement runs** (admins only, not in the desktop app): **Run one now** lets Captain Swarm try better charters on the factory's cases, as rehearsals. It keeps only what scores clearly higher and holds up on the sealed cases. **Make it a draft** turns the winner into a draft you approve.

## On your Line

Each factory has one thread on your [Line](flight-deck.md#the-line): "<name>: done", "<name>: the run failed", "<name>: carried out", "<name> would do 3 things", "<name> is paused", "<name>: version 2 is live". A finished run's message starts with its result file, its size and how long it took, then anything worth a look. Each run item has 👍, 👎 and **Open First Mate**.

- **Phones** get what needs you, and finished runs you started by hand or that took 30 minutes or more.
- **First Mate's week** arrives on Monday morning: last week's runs, tokens, carry-outs and what's worth a look. A quiet week sends nothing. **The week** on the page shows the same, with **Last week** and **This week so far**.
- **Three failed runs in a row** pause a factory. It shows under **Needs you** ("Paused after 3 failed runs in a row: see why, then resume it"), and your Line says "<name> is paused". **Resume** it from its **⋯** menu.

## Archive

- **A factory**: **⋯ → Archive** (not while a run is going). It leaves your list and is paused, and its runs and results are kept. **Archived · N** under the list opens them: **Restore** brings one back, paused, and the bin deletes it for good.
- **Runs**: the archive button on a finished run, or **Archive finished runs** on **Runs**. They leave **Runs** and **Results**, and nothing is deleted. **Archived · N** shows them, and **Back to the list** returns.

Scores, weekly reports and trust still count archived runs.

## Asking an agent

Agents under Flight Deck have the `first_mate` tool. On your own turns, an agent can:

| Action | What it does |
|---|---|
| `list` | Every factory with its state, trust, health, last and next run, and what waits for you |
| `status` | One factory: its charter, health, this week, trust, cases, last run and what it would do |
| `run` | Like pressing **Run now**, with an optional note. A factory that asks first only rehearses |
| `pause`, `resume` | Pause a factory (with a reason) or resume it |
| `report` | Last week's or this week's report |
| `draft` | Set up a new factory from your words. It waits for your approval |

An agent can't approve a charter, carry out what a factory would do or change its trust: those stay yours, on First Mate and the Line. The tool refuses inside any run or rehearsal, isn't offered on a shared agent's member turns, and follows the deck's switches.

## Limits and good to know

**What Bat does under the hood.** Every factory run by agents is a Bat run, and Bat got most of this release's work:

- **It keeps going while it makes progress.** A step has no fixed step count or time limit. Only a long silence (15 minutes: a hung model call) ends it, and a context overflow is trimmed instead of ending the step.
- **It saves as it goes.** Each step keeps its results in the run's folder and a progress note (done, left, what counts), on the run's **Notes** tab.
- **It resumes instead of restarting.** A step that runs again after **Continue**, a deck restart, a timeout or a failure starts from its note and its files.
- **Plans sized for the job.** Many items of one kind are split by batch, up to 12 steps. The first step fixes the columns for all of them, the last one only puts the parts together, and a step that runs out of room is split.
- **Checks between steps.** Scripts check each finished step against your rules and the other parts, at no token cost, and a short review corrects the run's course when it drifts.
- **It reads what it delivered.** Before the verdict, Captain profiles the main files; that's the **Worth a look before you use it** box.
- **It knows what's short.** A run that stops short plans only for what's missing ("9 of 15, 6 more").
- **Workers use** (**Admin console → Settings → Bat**) lets a strong model plan and check while a cheaper tier does the steps. The same card sets **Steps at the same time**, **Step ceiling per worker turn**, **Quiet minutes before a step stops** and **Minutes one model call may take**.

**Numbers.**

- Up to 8 factories per person and 24 runs per factory a day, unless an admin changes it.
- Extra data: 200 files and 200 MB per factory, 200 files and 400 MB per run, 50 MB per upload.
- A run with **No cap** stops at 20 million tokens (`CLAW_BAT_TOKEN_CEILING`; 0 turns it off). **Continue** carries on from there.
- Budgets are tokens, never dollars.

**For the deck owner.** `FD_FIRST_MATE_DISABLED=true` stops First Mate's supervisor altogether, and `FD_FIRST_MATE_TICK_SECONDS` sets how often it looks at schedules, budgets and new events (60 by default). Turning First Mate on or off is the console switch, not an environment variable.
