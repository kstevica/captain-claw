# Captain user video: plan

Status: **approved** (2026-11-07), with the defaults below. Phase 2
writes each chapter's synopsis and narration and records its screenshots and
clips. Those become the chapters of the tutorial video.

Source of truth for features: [user-feature-inventory.md](user-feature-inventory.md).
It marks everything a member can do, what needs an admin first, and what is
admin-only.

## Decisions (defaults proposed: approving the plan accepts them)

| # | Question | Proposed default | Why |
|---|---|---|---|
| D1 | Which surface counts as "the user's"? | The **user view** (kiosk), desktop web plus phone. The full Flight Deck is out of scope and could get a separate power-user series later | Every non-admin in the desktop and Android apps gets it, and admins put team members there. The full layout has 30+ pages, several of them experimental |
| D2 | Who is the viewer? | A team member at a small firm (consultancy, agency, accounting) whose admin already set up the deck | Matches the teams-first audience. Self-hosters get the admin video |
| D3 | Demo persona and data | **Maya** at a made-up firm, *Northwind Advisory*. Agents: *Research Assistant*, *Inbox Assistant*, *Proposal Writer*, plus one shared agent. All documents and clients fictional | Maya and the Research Assistant already exist in the screenshot harness |
| D4 | Theme and size | Dark theme. Desktop at 1920×1080. Phone at 390×844 inside a device frame | One theme keeps the series consistent. The existing screenshots are dark |
| D5 | Where the agents' answers come from | **Hybrid.** Mock agents for chat-surface clips (repeatable, same every take). A real cheap model on the demo deck for anything the deck itself writes: setup review, exams, Taskboard triage, Line items, Forge plans, "Fill it in for me" | Mock replies can't produce exam grades or triage. A real model, run once before recording, makes those screens honest |
| D6 | Connected Google / Microsoft / Telegram | A **dedicated demo Google account and Telegram bot** that you provide. Without them we film only the "Connect" screens | A throwaway deck can't show a connected mailbox with placeholder OAuth clients |
| D7 | Voice | Narration scripts written for a voice-over. Who voices it (you, or a TTS voice) is decided after the scripts | Doesn't block phase 2 |
| D8 | Length | Each chapter is its own 2–5 minute video. A shorter **Essentials** path is marked ★ | Short chapters can be re-shot one at a time when the UI changes |
| D9 | Crew and custom tools on camera | A real coding CLI (Claude Code on a cheap model) installed and logged in on the recording machine, plus a small throwaway demo repository. Tasks are run once before recording; one live run is filmed short | A mock can't produce a real diff, a passing tool build or a live page |

## The chapters

★ = Essentials path (12 chapters, about 45 minutes). The 25 chapters run about
95 minutes, plus up to 2 bonuses.

Each chapter below lists: **Goal** (what the viewer can do afterwards),
**Shows**, **Demo state** (what must already be on the deck), and **If you
don't see it** (the narration line for admin-gated features).

### Part 1: Start here

**1 ★ Welcome to Captain** (1.5 min)
- **Goal:** know what Captain is for and what the series covers.
- **Shows:** a montage. An agent answers in chat, a file appears in the library, a task lands on the Taskboard, the Line pings, and the same agent replies on a phone.
- **Demo state:** a fully set-up deck, with every later chapter's state on it.

**2 ★ Getting in** (2.5 min)
- **Goal:** get from the invite email to the user view.
- **Shows:**
  - invite link → pick a name and password → signed in;
  - normal sign-in;
  - "Choose your own password" after an admin made the account;
  - "Create account" and "Join waitlist" tabs, briefly, as other ways in;
  - the Getting started guide, step by step, and "Do not show again".
- **Demo state:** a pending invite for Maya; a sign-up mode set to show both tabs; a second account with an admin-set password.

**3 ★ Finding your way around** (3 min)
- **Goal:** know every area of the screen.
- **Shows:**
  - the agents column: state dots, the power switch, an asleep agent;
  - the Line row, and the bell with its lessons badge;
  - the tool tiles (Tasks, Scheduled, Library, Data, Folders);
  - Settings and Info;
  - the collapsed rail, the empty state, and the dark/light switch.
- **Demo state:** 3 agents in different states (one working, one with a new reply, one asleep); a few Line items; one bell item.

### Part 2: Working with an agent

**4 ★ Your first agent** (3.5 min)
- **Goal:** create an agent that fits a job.
- **Shows:**
  - New agent → the archetype picker (kinds, search, detail);
  - name it → **Create ‹Role›** → the chat opens by itself;
  - the "Set up its way of working next" switch (pointing ahead to Ch 9);
  - the "Connect Google first" note on a mail archetype;
  - **Forge agents**: describe a small team, review the plan, approve.
- **Demo state:** the archetype catalog published; a real model for Forge.
- **If you don't see it:** "No archetypes available" means your admin hasn't added any yet.

**5 ★ Talking to your agent** (5 min)
- **Goal:** give an agent work and use what it hands back.
- **Shows:**
  - typing; Enter versus Shift+Enter;
  - attaching with +, paste and drag-drop;
  - **What can I do?** cards and a card that asks for details first;
  - the Activity panel as it works; Stop;
  - the suggested next-step chips;
  - picture and file cards (Open, Download) and **Files ▾**;
  - Copy and **Send to another agent**.
- **Demo state:** a mock agent with skill cards, a reply that makes a file, a reply with a picture.

**6 ★ When the agent needs you** (3 min)
- **Goal:** handle questions, approvals and limits without losing work.
- **Shows:**
  - an approval card (Approve / Deny);
  - the "‹Agent› asks" dialog;
  - a message sent while busy: "Sends when ‹agent› is free", **Send at next step**, **Don't send**;
  - waking an asleep agent;
  - the reasoning-effort picker;
  - the daily allowance bar, **Use a free reset** and **Use my own keys**.
- **Demo state:** a mock agent that asks and asks for approval; Maya on a plan with daily limits, near the limit, with one free reset.
- **If you don't see it:** the allowance bar shows only on decks with daily limits.

**7 Topics and lanes** (2.5 min)
- **Goal:** keep separate conversations with one agent apart.
- **Shows:**
  - lanes A / B / C, and what Auto holds;
  - **+ New topic** and the divider it leaves;
  - **Topics**: jump to a topic, rename it, resume it;
  - Earlier messages.
- **Demo state:** an agent with three topics in lane A and a scheduled result in Auto.

**8 ★ The queue: line up work** (5 min)
- **Goal:** hand an agent a list of tasks and let it work through them.
- **Shows:**
  - ⌘J; ⌘Enter from the composer;
  - add a task; paste a list → **Add as N separate tasks**;
  - **Auto | Step by step**;
  - the hero card through Working → Your turn → Mark done & start next, plus one Error → Try again;
  - reorder and **Do this next**; **Insert now**; **Run now**;
  - edit, skip and remove, then Undo;
  - the Done section;
  - **Plan tasks**, briefly.
  - Says plainly that the queue lives in this browser only.
- **Demo state:** a mock agent with replies of different lengths, and one task that fails.

### Part 3: Making agents yours

**9 Teach it your way of working** (5 min)
- **Goal:** give an agent your documents and rules and check what it took from them.
- **Shows:**
  - ⋯ → **Onboarding** → Add documents (drag-drop, and from the File library);
  - a note on how to use each document; the reading budget; Focus on;
  - closing the dialog while it reads, then the bell saying it's ready;
  - reviewing rules, skills, facts, templates and the glossary;
  - a spreadsheet becoming a table;
  - the **Materials** tab.
- **Demo state:** a real model; 3 fictional documents (a style guide, a price list xlsx, a process PDF). Pre-read once so the review screen is ready, with one fresh read filmed short.

**10 Lessons, practice and exams** (5 min)
- **Goal:** correct an agent once and have it stick, then prove it learned.
- **Shows:**
  - the **Save as a lesson?** chip after a correction in chat;
  - Lessons: *Always in mind* versus *When it matters*;
  - **Lessons to approve** (bell badge) → Approve / Edit;
  - **Practice** → Fix it → it tries again;
  - **Exams**: run a Quick exam, read why an answer was wrong, **Teach it this**, ask again and see it improve;
  - a note that lessons from your admin are locked.
- **Demo state:** the Ch 9 agent; a real model; one exam already graded with a wrong answer.

**11 Its instructions, model and identity** (4 min)
- **Goal:** tune how an agent behaves, thinks and introduces itself.
- **Shows:**
  - ⋯ → **Instructions, insights & skills**: edit instructions; Teach it something; Correct it / Forget an insight; switch a skill off;
  - ⋯ → **Settings**: model, Vision, How it thinks, reasoning effort per source, "Restart now";
  - ⋯ → **Identity & email**: role, names it answers to, colleagues it may hand work to, its own inbox.
- **Demo state:** an agent with a few insights and skills; an IMAP test inbox for the identity part.

**12 Keeping your agents tidy** (2.5 min)
- **Goal:** organise many agents.
- **Shows:**
  - **New group**; dragging agents into it; **Edit group** (colour, order, Pin to the top);
  - pinning an agent; Search;
  - turning an agent off and on;
  - ⋯ → Info (Intentions, and the *other* Topics);
  - **Remove agent**, saving its files first;
  - **Leave** on a shared agent.
- **Demo state:** 7+ agents, so that Search appears; one shared agent.

### Part 4: Your work in one place

**13 ★ Files, data and shared folders** (5 min)
- **Goal:** find anything an agent made and share it.
- **Shows:**
  - **File library**: search, filters, Open, Edit, **Present as deck**, **Public link**, Agent chat;
  - **Data library**: rows, export to Excel;
  - **Shared folders**: New folder, drag-drop upload, Share with people, the `vfs:` address handed to an agent in chat, a Google Drive folder;
  - ⋯ → **Files & data** for an agent the libraries don't index.
- **Demo state:** files of several kinds (report, deck, spreadsheet, image); 2 datastore tables; a shared folder; the demo Drive connected (D6).
- **If you don't see it:** "Public link" and the Drive/OneDrive buttons depend on your admin.

**14 Scheduled tasks** (2.5 min)
- **Goal:** have an agent do something every morning without asking.
- **Shows:**
  - **New task**: prompt, schedule, agent;
  - delivery to Chat / Email / Telegram;
  - **Run now** and the run history; the result landing in the Auto lane;
  - the daily **Summary of your Line**.
- **Demo state:** one existing task with a few runs; email connected.

**15 ★ The Taskboard** (5 min)
- **Goal:** see everything your agents were asked and owe, and act on it.
- **Shows:**
  - the lanes (Needs me, Waiting on others, Agents working, Done);
  - **Why is this here?**;
  - Done, Snooze, Not a task, with Undo;
  - **Open bigger** on an email task;
  - a drafted next step → edit → Approve (email stays a draft);
  - **Hand to agent**;
  - batch: tick 3 → one instruction → Hand over;
  - **What the agent did**;
  - the keys (j/k, x, e, ?).
- **Demo state:** a real model; ~15 tasks across lanes from chats, scheduled runs and the demo Gmail; Jev on for drafts.

**16 ★ The Line and notifications** (3.5 min)
- **Goal:** stay on top of what agents did on their own.
- **Shows:**
  - the Line: day groups, a thread, the Failed / Needs you / Approve? pills, Approve in a card;
  - the composer: **Look into it**, **Draft a reply**, Hand to agent;
  - **Channels**: Telegram at *Needs you*, quiet hours;
  - the bell as "what needs you";
  - the deck update notice and its countdown.
- **Demo state:** a real model; ~20 Line items over 3 days including a failure and a proposal; Telegram linked.

### Part 5: Connected

**17 ★ Connecting your accounts** (5 min)
- **Goal:** let agents read and write your mail and calendar safely.
- **Shows:**
  - Connections: **Google** → Connect → the sending policy (allowlist, daily limit) and signature;
  - **Microsoft**, briefly;
  - **Email (IMAP)**: add an inbox;
  - **Notifications**;
  - then an agent reading today's calendar and drafting a reply.
- **Demo state:** the demo Google account (D6); an IMAP test inbox.
- **If you don't see it:** "Ask an admin…" means your admin hasn't set up Google or Microsoft sign-in yet.

**18 Talk to your agents from Telegram and WhatsApp** (3 min)
- **Goal:** reach your agents from the chat apps you already use.
- **Shows:**
  - **Telegram**: pick an agent → request → (approved) → open the bot link → chat;
  - the Telegram inbox consent;
  - **WhatsApp Web**: QR pairing, chat whitelist, sending switched on deliberately;
  - "via WhatsApp" bubbles in the deck.
- **Demo state:** the demo Telegram bot; WhatsApp Web allowed on the deck, with a demo phone.
- **If you don't see it:** both need your admin first. WhatsApp Web is marked experimental.

**19 Captain on your phone** (3 min)
- **Goal:** do the same work from a phone.
- **Shows:**
  - the phone layout: header tools, agent cards, the chat, the Files sheet, the queue full screen;
  - Back closing things in order;
  - the Android app and Settings → Switch deck.
- **Demo state:** the same deck at 390×844, plus an Android emulator or device recording.

**20 Captain beyond the deck** (4 min, optional)
- **Goal:** know what else can reach your agents.
- **Shows:** 30–40 seconds each of:
  - Captain for Chrome (pair, clip a page);
  - Captain for Mac;
  - Claude via **Agent access for Claude** (token);
  - adding an HTTP MCP server;
  - Systems (agent links);
  - a voice device using the same token.
- **Demo state:** the extension and the Mac app installed on the recording machine.

### Part 6: You, and checking the work

**21 ★ Your profile and account** (3.5 min)
- **Goal:** tell agents who you are, and manage your account.
- **Shows:**
  - **Profile → Personal**: **Fill it in for me**, then **Use** a draft and Save, and "What your agents receive";
  - **Deck-wide**, read-only;
  - **Password**, with "Sign out my other devices";
  - **Info**: plan, usage, free resets;
  - **Your own AI keys**;
  - Sign out.
  - Says there is no "forgot password": ask your admin.
- **Demo state:** a real model and web search for the researcher; own keys allowed.

**22 Checking the work: Activity, backups and restore** (3.5 min)
- **Goal:** see what an agent actually did, and never lose an agent.
- **Shows:**
  - **Activity**: every tool call, filtered to one agent;
  - "it said it emailed, did it?": the Taskboard's *What the agent did* next to the register;
  - **Agent backups**: schedule, Back up now, Download, Restore;
  - ⋯ → **Back up / export** (Template / Trained / Full);
  - New agent → **From a file**.
- **Demo state:** an agent with a varied tool-call history, including one refused call; one backup already made.

### Part 7: Building with Crew and custom tools

These chapters need an admin first: Crew approval for the member, and Custom
tools switched on for the deck. Each opens with one line saying so.

**23 Crew: coding agents working for you** (5 min)
- **Goal:** hand a coding job to a coding agent and decide whether its work lands.
- **Shows:**
  - opening Crew; the overview (Needs you, Ready to look at, Working now);
  - **Logins**: a Claude Code token and a GitHub token, and each coding agent showing "ready" or "needs a login";
  - **New task**:
    - **What to do**, with a screenshot pasted in;
    - **Who does it**, **Model**;
    - **Where** (a project, or a repository address);
    - **What kind** (Change the code / Only look, and report);
    - **Folders from your files**;
    - **When it's done** (leave it for me / open a pull request);
  - watching **Activity**; steering with **After this turn** and **Send now**;
  - **What changed** ("what git shows, not what the agent says") and the **Changes** diff; **Files**; **Report**;
  - **Merge into ‹branch›** → "Yes, merge it", or **Open a pull request**;
  - **Discard** on a second task;
  - a "look and report" task's report;
  - the results in Shared folders under `crew/`.
- **Demo state:**
  - a real CLI (Claude Code on a cheap model) installed and logged in on the recording machine;
  - a small demo repository;
  - 3 tasks done beforehand (ready, merged, a report), plus one live run filmed short.
- **If you don't see it:** "Crew needs an admin's approval" means just that. A coding agent marked "not installed" needs your admin.

**24 Crew: projects, chat and live sites** (4 min)
- **Goal:** run Crew from chat, keep projects in order, and put a page on the web.
- **Shows:**
  - asking an agent in chat: "Have Crew fix the date format in the demo project";
  - the task appearing on the board, and the agent called back with the result;
  - the agent asking to merge → **Approve** in the bell;
  - a project page: stats, **New task here**, **New project**, **Tidy up finished**;
  - the **Environment** card (Prepared / Partly prepared) and asking a coding agent to prepare it;
  - **Live sites** → **Publish a site** (a page) from a task's Files tab → open the address → stop it, start it again, Remove it;
  - the **Terminal**, shown with the line that an admin turns it on.
- **Demo state:** the Ch 23 project with a static page in it; a process agent whose model can call the `crew` tool; the terminal on for Maya.
- **If you don't see it:** starting an *app* (rather than a page) and the terminal both need your admin.

**25 Custom tools** (5 min)
- **Goal:** get a small tool built for a job a model does badly, use tools others made, and share yours.
- **Shows:**
  - the three tabs: **Mine**, **Shared with me**, **Everybody**;
  - **Your agents suggest** → **Build it**;
  - **New tool**: describe it → the check's fuller spec and "Already there?" → pick and edit the spec → **Build it** → the Crew task → approve the merge → **Live**;
  - the tool sheet:
    - **Try it**, with **Run as** an agent;
    - **Your secrets for it**;
    - **Tools it runs**;
    - **Versions**;
    - **Change it**, with "fix this failure";
    - **Offer it** (People…, and Offer to everybody, which may wait for an admin);
    - **How it's used**;
  - from Maya's colleague's side: **Shared with me** → **Add** ("it runs as you, with your secrets and files");
  - an agent using the tool in chat;
  - the tool asking a question mid-run.
- **Demo state:**
  - Custom tools on; Maya approved for Crew;
  - a second member account that offers one tool to Maya;
  - one tool already built and live, one agent suggestion waiting, one build filmed short.
- **If you don't see it:** "Custom tools are off on this deck" means an admin turns them on. Without Crew you can still add tools others offer you, but you can't make your own.

### Bonus (deck-dependent)

- **B1 Agents shared with you** (2 min): the Shared · Owner badge, the member notice, Suggest as a lesson, a queue only you see, Leave. Needs agent sharing on.
- **B2 Share on network** (1 min): only if the deck runs on a network. Probably skip.

## Fix before filming

Things in the user view today that contradict the narration or look broken on
camera. Each fix is small.

| # | Problem | Where | Proposal |
|---|---|---|---|
| F1 | The guide says "Press **+** next to Agents" and "**Create agent**". The UI says **New agent** and **Create ‹Role›** | `flight-deck/src/components/layout/KioskOnboardingDialog.tsx:453,455` | Fix the copy, then regenerate the guide screenshots |
| F2 | Onboarding → Pass it on says "share it from Library → Archetypes", which a kiosk user can't reach | `flight-deck/src/components/layout/OnboardingDialog.tsx:428` | Change the copy for the kiosk, or don't film Pass it on |
| F3 | Pin message and Add to shared clipboard work in the kiosk, but their panels only exist in the full layout, so what's saved can't be seen | ChatPanel hover actions; `flight-deck/src/App.tsx:447-449` | Hide both in the kiosk |
| F4 | Message actions (Copy, Send to) appear only on hover, so a phone can't reach them | `flight-deck/src/components/agents/ChatPanel.tsx:2105` | Tap-to-show on touch screens |
| F5 | ⌘K lists Director and Simple/Full view, which do nothing in the kiosk | `flight-deck/src/components/common/KeyboardShortcuts.tsx:7-17` | Filter the list in the kiosk |
| F6 | ⌘J may toggle the hidden chat panel instead of the queue when a member's saved layout is "full" (unverified) | `KeyboardShortcuts.tsx:60-71` | Check on the demo deck; fix if real |
| F7 | The Agent backups tile shows even with nothing to back up | `SimpleLayout.tsx:1379` | Leave it |
| F8 | Two different features are both called "Topics" (lane history versus ⋯ → Info clusters) | `KioskAgentDialogs.tsx:236-239` | Rename the Info tab (e.g. "Themes"), or explain the difference in Ch 12 |
| F9 | The queue is saved in this browser only | queue ⋯ | Leave it; Ch 8 says so |

Recommendation: fix F1–F5, check F6, and leave F7–F9 to the narration. That's
one small PR before recording starts.

**Done (2026-11-07, branch `fix/kiosk-tutorial-fixes`):**
- **F1:** the copy is fixed, and the guide's 28 screenshots were regenerated, because they still showed the old **+** button, three tiles and v0.8.0.
- **F2:** the Pass it on result now has **Share with people…** right there, instead of pointing to the Library.
- **F3** and **F4** are fixed.
- **F5:** the overlay lists only the shortcuts that work where you are.
- **F6** was real: ⌘J toggled the hidden chat panel for kiosk users with no saved layout. It is fixed in the same change.

## Phase 2: how each chapter gets made

### What each chapter folder holds

`docs/tutorial/user/NN-slug/`:

- `synopsis.md`: title, a one-paragraph synopsis, goals, prerequisites, length, and the "if you don't see it" lines.
- `script.md`: a beat table with columns **# · on screen · action (click / type) · narration · seconds · callout**.
- `shots/`: PNG stills at each key beat. They also serve the thumbnail and a written guide.
- `clips/`: screen recordings, one per beat group, as `.webm` from the recorder converted to `.mp4`.
- Later: `voice.wav`, `captions.srt`, and the cut `chapter.mp4`.

### The recording rig

- A new `flight-deck/e2e/tutorial/` folder, grown from `flight-deck/e2e/onboarding/run.sh`.
- It boots a throwaway deck and sets it up as the admin would:
  - kiosk mode for selected users;
  - Maya, plus an invite and a waitlist entry;
  - OAuth clients, the team Telegram bot, own keys allowed, public links, sharing, daily limits;
  - the mock agents and the real-model agents.
- It then seeds each chapter's demo state.
- One Playwright script per chapter (`scene-NN.cjs`) drives the UI, records video (`recordVideo`) and takes stills. It has helpers for:
  - a visible cursor with a click ripple (headless recordings don't show the cursor);
  - human-speed typing;
  - a highlight box for callouts;
  - waiting until the screen is still.
- **Why scripted:** every chapter can be re-shot in minutes when the UI changes, and both themes and the phone come from the same script. Hand recording is kept for the Android app, Telegram on a real phone, and the Chrome and Mac apps.

### How the narration is written

- Second person, plain words, present tense, one idea per sentence.
- About 140 words a minute, so a 3-minute chapter is about 420 words.
- UI labels are spelled exactly as they appear on screen, in bold in the script.
- The product is called "Captain", never "Captain Claw".
- No version numbers, so chapters age well.
- Each chapter opens with what you'll be able to do, and ends with a one-line recap and the next chapter.
- Admin-gated features get the "If you don't see …, ask your admin" line.

### Order of work after approval

1. The F1–F5 fixes (one PR). Done, see above.
2. Build the rig and the demo deck seed.
3. For each chapter, in this order: Essentials first (1, 2, 3, 4, 5, 6, 8, 13, 15, 16, 17, 21), then chapters 7, 9–12, 14, 18–20 and 22, then Crew and custom tools (23–25, which need the CLI setup from D9).
4. For each chapter, three steps:
   1. synopsis and script for your review;
   2. record clips and stills against the approved script;
   3. you review and we re-shoot.

## Not in this plan

- The admin video. It comes next, from the same inventory's 🚫 items plus the Admin console.
- The full Flight Deck pages for non-kiosk members (inventory §14).
- Final editing, music and voice-over production.
