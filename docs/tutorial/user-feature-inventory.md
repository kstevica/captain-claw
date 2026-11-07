# What a Captain user can do (feature inventory)

The source list for the **user** tutorial video (the admin video comes later).
It was taken from the code on 2026-11-07, at version 0.8.11 (main @ 7964b4f0).

**Who counts as a "user":** a signed-in member who is not an admin. Members live
in the **user view**, the kiosk (`SimpleLayout locked`), when:

- the admin turns on kiosk mode for everyone or for selected users, or
- they use the desktop or Android app, where every non-admin always gets the kiosk
  (`flight-deck/src/stores/authStore.ts:150-158`).

Members who are not kiosk-locked get the full Flight Deck instead. That has 30+
pages, several of them experimental, and is listed briefly at the end (§16) as
out of scope.

## Marks

| Mark | Meaning |
|---|---|
| **Ch N** | Shown in chapter N of [the plan](user-video-plan.md) |
| ⚙ | Works only after an admin sets something up. The demo deck has it on; the narration says "if you don't see this, ask your admin" |
| 🚫 | Admin-only. Not in the user video |
| ⚠ | Wrong, confusing or broken today. Fix it or film around it (see the plan's "Fix before filming") |
| — | Real, but too small or niche for the video. Left out on purpose |

---

## 1. Getting in

| Feature | Where | What it does for the user | Mark |
|---|---|---|---|
| Accept an invite | Email link `<deck>/invite#…` | Shows who invited you; pick a name and password and you're in | Ch 2 |
| Sign in | Sign-in page with the deck's name and logo | Email + password | Ch 2 |
| Create account | "Create account" tab, when the deck allows sign-up | Name, email, password | Ch 2 (on screen briefly) |
| Join waitlist | "Join waitlist" tab | Apply; an admin approves; "You're on the waitlist" screen | Ch 2 (on screen briefly) |
| Choose your own password | Forced after an admin created the account | Replace the admin's password with your own | Ch 2 |
| Getting started guide | Opens on the first visit; Settings → Getting started; Info | 7 steps: Welcome, Create an agent, Give it work, Google & Microsoft, Telegram, Your profile, Good to know. Buttons jump to the right place; "Do not show again" | Ch 2 ⚠ F1 |
| Forgot password | — | **Doesn't exist.** Ask the admin | Ch 21 (one line) |

## 2. The user view at a glance (desktop)

| Feature | Where | What it does | Mark |
|---|---|---|---|
| Agents column | Left | Your agents, each with a state dot (running / working / new reply / asleep) and "shared" / "network" badges | Ch 3 |
| Open an agent | Click its row | Its chat opens in the middle; an asleep agent wakes | Ch 3 |
| Power switch | Row power icon | Turns your agent on or off (asks first; parks its queue) | Ch 3, 12 |
| Line row | Top of the column | Opens the Line, with unread and waiting counts | Ch 3, 16 |
| Bell, and the graduation-cap badge | Column header | What needs you; the badge opens Lessons to approve | Ch 3, 16 |
| Tool tiles | Above the footer | Tasks · Scheduled · Library · Data · Folders (Crew ⚙ → Ch 23, Swarm 🚫) | Ch 3 |
| Settings | Footer button with your initials | The Settings sheet (see §9) | Ch 3 |
| Info (ⓘ) | Footer | Read-only: deck and version, your plan and usage, your agents and models, what agents know about you, connection status | Ch 3, 21 |
| Hide agents / collapsed rail | Header chevron | Folds the column into an icon rail | Ch 3 |
| Resize column | Drag the edge | Width is remembered | — |
| Empty state | Middle, when no chat is open | "Pick an agent", "New agent", "New here? Getting started" | Ch 3 |
| Dark theme | Settings → Dark theme | Light or dark | Ch 3 |
| Refresh agents | Settings | Checks every agent again | — |

## 3. Creating agents

| Feature | Where | What it does | Mark |
|---|---|---|---|
| New agent → archetype picker | "New agent" button | Browse kinds and families, search by role, task or skill, read the detail, name it, **Create ‹Role›** | Ch 4 |
| Set up its way of working next | Switch in the picker | Goes straight to setup after creation | Ch 4 → Ch 9 |
| Connect Google / Microsoft note | Picker detail, for mail archetypes | Offers to connect a mailbox first | Ch 4 ⚙ |
| Forge agents | Picker card / header | Describe what you need; Captain plans a team for you to approve; keeps going if you close it | Ch 4 |
| From a file | Picker header | Import a `.captain` backup or export as a new agent | Ch 22 |
| Auto-open | After Create | The new agent's chat opens when it's ready | Ch 4 |
| "No archetypes available" | Picker | The admin hasn't added any | Ch 4 (one line) ⚙ |

## 4. Chatting with an agent

| Feature | Where | What it does | Mark |
|---|---|---|---|
| Message box ("Ask anything…") | Composer | Enter sends, Shift+Enter makes a new line | Ch 5 |
| Attach | +, clipboard icon, ⌘V, drag-drop onto the chat | Files and pictures go with the message as chips; Send waits for uploads | Ch 5 |
| What can I do? | Sparkles in the composer; greeting cards in a fresh chat | The agent's skill cards; one click sends the prompt; "More…", "Suggest new ones" | Ch 5 |
| Skill input dialog | A card marked "Asks you for details first" | Collects text, files or pictures, then starts | Ch 5 |
| Activity panel | In the reply | Live tool calls and narration, folding to a summary, with tokens | Ch 5 |
| Working line | Above the composer | Thinking… / Working… / Waking up… / Stopping… | Ch 5 |
| Stop | Red button | Stops the reply | Ch 5 |
| Suggested next steps | Chips under a reply | One click sends that follow-up | Ch 5 |
| Picture and file cards | In replies | Open (viewer or new tab) / Download | Ch 5 |
| Files ▾ (Recent files) | Composer | Last 5 files made in this chat: Open, Download, Show in library | Ch 5 |
| Compact cards | In replies | "Automatic", "You asked", "Rephrased task": open the full text, jump to the task | Ch 5 |
| Copy / Send to another agent | Hover a message | Copy text; forward it with context to another agent | Ch 5 ⚠ F4 |
| Pin message / Add to shared clipboard | Hover a message | Saves to panels the kiosk can't show | ⚠ F3 (don't film) |
| Approval card | In the stream | Approve / Deny an action the agent asked about | Ch 6 |
| "‹Agent› asks" dialog | Pops up | Answer the agent's question (text, yes/no, choice), or "Don't answer" | Ch 6 |
| Waiting message | Send while the agent is busy or asleep | "Sends when ‹agent› is free"; **Send at next step**; **Don't send** | Ch 6 |
| Asleep → Wake up now | Banner | Wakes an idle agent; what you write waits for it | Ch 6 |
| Reasoning effort | Brain button | Auto or a level, per agent (only when the model offers levels) | Ch 6 |
| Daily allowance bar | Above the composer, when limited | Messages and AI use left, reset time, **Use a free reset**, **Use my own keys** | Ch 6 ⚙ |
| Agent disclaimer | Under the chat | Fixed note for legal and health agents | — |
| Slash commands (/new, /clear…) | Typed | Work, but there's no menu for them | — |
| Earlier / Later messages | Top / bottom of the chat | Page through history | Ch 7 |
| Topic divider | In the stream | New topic / Back on an earlier topic / Context cleared | Ch 7 |
| Payment card | After an agent prepares a payment | Pay in my bank, I paid, Skip | — ⚙ (FD_PAYMENTS; off on most decks) |
| Voice input | — | **None** in the chat (phone keyboard dictation only) | — |
| Edit / retry a sent message, export a chat | — | **None** | — |

## 5. Lanes and topics

| Feature | Where | What it does | Mark |
|---|---|---|---|
| Lanes A / B / C / Auto | Tabs over the chat | Three parallel conversations with one agent. Auto holds scheduled and automatic turns | Ch 7 |
| + New topic | Lane row | Fresh context; earlier messages stay above a divider | Ch 7 |
| Topics | Lane row → popover (sheet on a phone) | Earlier topics: Jump to it, Rename, Resume | Ch 7 |
| ⋯ → Info → Topics | Agent menu | A *different* thing: the agent's topic clusters | Ch 12 ⚠ F8 |

## 6. The queue

| Feature | Where | What it does | Mark |
|---|---|---|---|
| Show / hide queue | List icon (badge colours) · ⌘J | Queue column left of the chat; full screen on a phone | Ch 8 ⚠ F6 |
| ⌘Enter in the composer | Composer | Puts the message on the queue and switches it to Auto | Ch 8 |
| Add a task | Queue box | "Add & start" / "Add to queue"; paste or drop files | Ch 8 |
| Add as N separate tasks | Multi-line paste, Alt+Enter | One task per line | Ch 8 |
| Auto \| Step by step | Queue header | Run on their own, or wait for you to mark each done | Ch 8 |
| Hero card | Top of the queue | Says what happens next: Working, Your turn (Mark done & start next), Error, Stopped, Gave up, Limit reached, Paused… | Ch 8 |
| Reorder | Drag the grip, Alt+↑/↓, "Do this next" | Change the order | Ch 8 |
| Insert now (I) | While a turn runs | Hands a waiting task to the running turn | Ch 8 |
| Run now | ⋯ | Moves the task to the front and starts it | Ch 8 |
| Edit / Skip / Remove + Undo | ⋯, keys, hover | Change or drop tasks; ⌘Z undoes | Ch 8 |
| Done section | Bottom | Stats, Add again, Show in chat | Ch 8 |
| Plan tasks | Wand / header ⋯ | Turns one description into a reviewed batch of tasks (datastore batching, lanes) | Ch 8 (short) |
| Saved in this browser only | Queue ⋯ | The queue doesn't follow you to another device | Ch 8 ⚠ F9 |

## 7. Making an agent yours (agent ⋯ menu)

Wide screens group the menu into rows: At work · What it knows · Who it is · Its place.

| Item | What it does | Mark |
|---|---|---|
| Info | Intentions and Topics, live from the agent | Ch 12 |
| Files & data | The live files and tables of agents the libraries don't index (docker, local, managed) | Ch 13 |
| Activity | The tool-call register for this agent | Ch 22 |
| Tasks | The Taskboard filtered to this agent | Ch 15 |
| Instructions, insights & skills | **Instructions**: standing instructions. **Insights**: search, Teach it something, Correct it, Forget. **Skills**: read, switch on/off, correct | Ch 11 |
| Onboarding | Overview · Lessons · Practice · Materials · Exams; Add documents; Pass it on | Ch 9, 10 ⚠ F2 |
| Way-of-working setup | Documents (up to 12), how to use each, reading budget (Quick / Standard / Deep), Focus on; read in the background; review rules, skills, facts, templates, glossary; sheets become tables | Ch 9 |
| Lessons to approve | Lessons the agent noticed in your chats: Approve, Edit, Dismiss | Ch 10 |
| Practice | A real or suggested task, nothing sent; **Good** or **Fix it** (becomes a lesson) | Ch 10 |
| Exams | Quick 10 / Standard 25 / Deep 60 / Marathon 120 questions; why each wrong answer went wrong; **Teach it this**, **Read the parts it missed** | Ch 10 |
| Base lessons | What the admin or archetype taught is locked for you | Ch 10 (one line) |
| Identity & email | Role and what it does; its own IMAP inbox ⚙; addresses; names it answers to; pick up email meant for…; colleagues it may hand work to | Ch 11 |
| Settings | **Model & thinking**: provider, model, key, endpoint, Vision, How it thinks, Email. **Reasoning effort** per source. "Restart now" | Ch 11 |
| Share on network… | Shares the agent on the deck's network (BotPort) | — ⚙ (deck-dependent) |
| Group / Pin | Put it in a group; pin to the top of its group | Ch 12 |
| Back up / export | Template / Trained / Full backup; Bring a file into it…; Back up on a schedule… | Ch 22 |
| Remove agent | Permanent, after a confirm, with links to save its files and tables first | Ch 12 |
| Leave (shared agents) | Give up access to an agent someone shared with you | Ch 12, B1 |

## 8. Organising agents

| Feature | Where | What it does | Mark |
|---|---|---|---|
| New group | Button beside New agent (with 2+ agents) | Named, coloured folder | Ch 12 |
| Drag onto a group | Desktop | Starts a group or moves an agent into one | Ch 12 |
| Edit group | Group ⋯ | Rename, colour, order, Pin to the top, pick agents, Delete | Ch 12 |
| Search agents | With more than 6 agents | Filters the list | Ch 12 |

Groups are per person and follow you to every device.

## 9. Settings sheet

| Tile | What it does | Mark |
|---|---|---|
| Taskboard | Opens the Taskboard | Ch 15 |
| Crew | Coding agents working for you (§14) | Ch 23, 24 ⚙ (admin approves per member) |
| Custom tools | Mine / Shared with me / Everybody (§15) | Ch 25 ⚙ (deck switch; making needs Crew); Approvals tab 🚫 |
| Captain Swarm | — | 🚫 |
| Scheduled tasks | Opens Scheduled tasks | Ch 14 |
| Agent backups | Schedule, Back up now, Download / Restore / Delete, signing fingerprint | Ch 22 ⚠ F7; "Decks this one trusts" 🚫 |
| Activity | Every tool call your agents made | Ch 22 |
| File library / Data library / Shared folders | As the tiles | Ch 13 |
| Profile | Profile dialog | Ch 21 |
| Connections | Connections dialog | Ch 17 |
| Your own AI keys | Bring your own provider keys | Ch 21 ⚙ (deck allows) |
| Dark theme · Refresh agents · Getting started | Compact row | Ch 3 |
| Switch deck | Android app only | Ch 19 |
| Sign out | Bottom | Ch 21 |
| Switch to admin view | Desktop app admin only | 🚫 |

## 10. Your work in one place

### File library, Data library, Shared folders

| Feature | What it does | Mark |
|---|---|---|
| File library | Every file your agents made, summarised, searchable, filtered. Open, Download, Edit, Present as deck, Public link ⚙, Agent chat, Summarise again, "Everything it keeps"; fullscreen | Ch 13 |
| Public link | Anyone with `/s/<token>` can open it; 7 days by default, 30 at most; list of all your links | Ch 13 ⚙ |
| Data library | Every datastore table, summarised; read rows; export CSV / XLSX / JSON | Ch 13 |
| Shared folders | Recent files, all folders, New folder, drag-drop upload, preview, edit, rename, Download as .zip, Share with people, Copy to my folders, `vfs:` address for agents | Ch 13 |
| Google Drive / OneDrive folders | Mounted read-only; Update from Drive; Make Markdown copies | Ch 13 ⚙ |
| Link a folder (server folder) | — | 🚫 |

### Scheduled tasks

| Feature | What it does | Mark |
|---|---|---|
| New task | Prompt + schedule + agent | Ch 14 |
| Delivery | Chat / Email ⚙ / Telegram ⚙ ("Connect" / "Link" until connected) | Ch 14 |
| Run now, Edit, Delete, run history, Agent chat | Manage and check runs | Ch 14 |
| Daily Line summary | A scheduled task, "Summary of your Line" | Ch 14, 16 |

### Taskboard

| Feature | What it does | Mark |
|---|---|---|
| Open it | Tasks tile (Needs me count), Settings, ⋯ → Tasks, phone header, `?taskboard=` link | Ch 15 |
| Views | Needs me / Waiting on others / Agents working / Done / All; sort | Ch 15 |
| What's on it | Every agent turn, scheduled jobs, proposals, follow-ups, held email, to-do lists, last 30 days of chats as history | Ch 15 |
| Your own inboxes | Telegram inbox, WhatsApp Web, Gmail: off until you consent; kept 30 days | Ch 15 ⚙ |
| Why is this here? | Every triage decision explained | Ch 15 |
| Task actions | Is it a task (Yes / No), Done / Not yet, Reopen, Snooze, Not a task, Rename, merge / split, Continue in chat, Undo | Ch 15 |
| Open bigger | One task over the board; Gmail thread shown in full | Ch 15 |
| Next step | The agent drafts a reply, reminder or question; you edit, approve or reject (email only ever becomes a draft) | Ch 15 |
| Hand to an agent | Next move → The agent… → instruction → Hand over | Ch 15 |
| Batch | Tick several (x, Shift+click), one action or one instruction for all; Stop the rest | Ch 15 |
| What the agent did | The reply is checked against the tool-call register and Gmail/Calendar | Ch 15, 22 |
| Keys | j/k, x, e, b, r, m, f, /, g, ? | Ch 15 |
| Export | CSV / JSONL | — |
| Board settings | Auto-close with Jev, limits | Ch 15 ⚙ (Jev) |

### The Line and notifications

| Feature | What it does | Mark |
|---|---|---|
| The Line | One timeline of everything the deck and your agents did on their own: day groups, threads, Failed / Needs you / Approve? pills, filters, fullscreen | Ch 16 |
| Approve / Reject | In a card beside the item | Ch 16 |
| Line composer | Ask any agent; Look into it / Draft a reply / What should I do?; Hand to agent | Ch 16 |
| Channels | WhatsApp and Telegram at Everything / Needs you / Off; one agent's chat; quiet hours | Ch 16 ⚙ |
| Bell | Waiting items first; Approve / Reject; Mark all read; "Everything else is on the Line" | Ch 16 |
| Lessons to approve (global) | Every agent's lessons; Approve all; Undo | Ch 10 |
| Deck update notice | "Will be updated soon" → restart countdown → "back on the new version, Reload" | Ch 16 |

## 11. Connections and channels (what a member sets up alone)

Kiosk tabs: Google, Microsoft, Email (IMAP), Telegram, WhatsApp Web, MCP Servers,
Systems, Browser, Jev, Soniox, External memory, Notifications, WhatsApp.
(ChatGPT, Antigravity and Deep memory are hidden in the kiosk.)

| Surface | What the user does | Mark |
|---|---|---|
| Google | Connect your own account; email sending policy (allowlist, daily limit); signature; topic auto-replies (off switch in the kiosk) | Ch 17 ⚙ (admin sets the OAuth client) |
| Microsoft | Same pattern; Outlook signature | Ch 17 ⚙ |
| Email (IMAP) | Add one or more inboxes; sending toggle | Ch 17 |
| Notifications | Where the Line reaches you | Ch 16, 17 |
| Telegram | Pick an agent → request → admin approves → one-time bot link; read-only Telegram inbox (consent) | Ch 18 ⚙ (team bot + approval) |
| WhatsApp Web | Link your own WhatsApp by QR or pairing code; consent; chat whitelist; opt-in sending and auto-reply | Ch 18 ⚙ (admin allows; experimental) |
| WhatsApp (nudges) | Your number for nudges | — ⚙ (Cloud API bridge) |
| Browser: Captain for Chrome | Download, Load unpacked, enter deck address, approve the pairing code | Ch 20 |
| Browser: Captain for Mac | Download (if the deck has a build), allow Accessibility, connect, approve | Ch 20 |
| MCP Servers: Agent access for Claude | Use your agents from Claude Desktop / claude.ai (paste URL) or Claude Code (personal token) | Ch 20 |
| MCP Servers: add an HTTP server | Give your agents an external tool | Ch 20; stdio and deck-wide 🚫 |
| Systems (agent links) | Let a CRM or shop talk to one of your agents with its own key | Ch 20 |
| External memory | Connect your own memory provider | Ch 20 |
| Jev / Soniox | Your own keys (otherwise the team key) | Ch 20 |
| Voice and e-paper devices | Device hotspot form: Wi-Fi, deck address, the access token from Agent access → Generate token | Ch 20 ⚙ (deck Soniox key) |
| Smart glasses HUD | `/hud/` pair with a code | — (devices video) |

## 12. Phone

| Feature | What it does | Mark |
|---|---|---|
| Phone header | Bell, Line, Taskboard, Scheduled, Library, Data, Folders, Info, Settings | Ch 19 |
| Agent cards | Tap to chat; power; ⋯ as a bottom sheet; Leave | Ch 19 |
| Chat screen | Back arrow; Files icon (File library for this agent, or the Files / Datastore sheet); ⋯ | Ch 19 |
| Queue on a phone | Takes over the screen; "‹ Chat" goes back | Ch 19 |
| System Back | Closes an overlay first, then returns to the list | Ch 19 |
| Android app | Always the user view; Settings → Switch deck | Ch 19 |
| Messages from channels | Bubbles say "via WhatsApp / Telegram / glasses" | Ch 18 |

## 13. Agents shared with you (deck with agent sharing on)

| Feature | What it does | Mark |
|---|---|---|
| "Shared · Owner" badge, member notice | Says what the owner can and can't see | B1 ⚙ |
| Suggest as a lesson | Your lesson waits for the owner | B1 |
| Your queue for it | Only you see it, in this browser | B1 |
| Member files and data | Only if the owner allows it | B1 |
| Leave | Give up access | B1 |

## 14. Crew: coding agents working for you ⚙

Crew runs command-line coding agents (Claude Code, Codex, OpenCode, Pi, Crush)
for you. Each task gets its own git worktree, and nothing lands without your
word.

**Admin first.** An admin approves Crew per member on the Users page. The
approval sets which coding agents you may use, how many tasks run at once, and
whether you get a terminal. Without it the dialog says "Crew needs an admin's
approval" or "Crew is turned off".

**Open it** from the Crew tile, Settings → Crew, or (on a phone) Settings.
Code: `flight-deck/src/components/crew/CrewBoard.tsx` and its siblings in
`crew/`; backend `captain_claw/flight_deck/crew_*.py`; the agent's tool is
`captain_claw/tools/crew.py`.

| Feature | Where | What it does for the user | Mark |
|---|---|---|---|
| Overview | Crew home | Needs you · Ready to look at · Working now · Live sites, with projects on the side | Ch 23 |
| Logins | Crew header → Logins | Your own logins: Claude Code token or Anthropic key, Codex key or auth.json, GitHub token (only the deck uses it, to clone, push and open PRs), and Anthropic / OpenAI / OpenRouter / Gemini / xAI keys for Crush, Pi and OpenCode | Ch 23 |
| Your coding agents | Logins | Each CLI shows "ready", "needs a login" or "not installed", with your default model for it | Ch 23 |
| Your own model | Logins | An OpenAI-compatible address, key, model name and context; Check connection | Ch 23 (short) |
| New task: What to do | New task | The whole brief, with screenshots and files pasted or dropped in | Ch 23 |
| New task: Who does it | New task | Pick a coding agent you're approved for | Ch 23 |
| New task: Model | New task | Your agent's model, a deck tier, or one you type | Ch 23 |
| New task: Where | New task | **A project** (a new name makes one; optional folder) or **A repository address** (Crew clones it). "A folder here" is admin-only | Ch 23; folder here 🚫 |
| New task: What kind | New task | **Change the code** (you decide if it lands) or **Only look, and report** (changes nothing) | Ch 23 |
| New task: Folders from your files | New task | Give it shared folders too, Read or Read and write (Drive and OneDrive folders can't be given yet) | Ch 23 |
| New task: When it's done | New task | **Leave it for me to look at** or **Open a pull request** | Ch 23 |
| New task: Effort | More options | Reasoning effort for the CLI | — |
| Task view | Click a task | Status, outcome, details, and **What changed**, which is what git shows rather than what the agent says | Ch 23 |
| Activity / Changes / Files / Report tabs | Task view | The live turn, the diff, the files it made, and its final report | Ch 23 |
| Tell the agent | Box under a task | Steer it, with attachments: **Send now** (stops the turn and goes) or **After this turn** | Ch 23 |
| Folders it can use | Task view | Change its shared folders from its next turn | Ch 23 |
| Merge into ‹branch› / Merge the pull request | Task view, after "Yes, merge it" | Lands the work | Ch 23 |
| Open / Update a pull request | Task view | Pushes the branch and opens or updates the PR (needs the GitHub login) | Ch 23 |
| Discard / Close | Task view | Throws unlanded work away (asks first), or closes a finished task | Ch 23 |
| Stop | Task view | Stops a running task | Ch 23 |
| Terminal | Task view | A shell in the task's worktree | Ch 24 ⚙ (admin turns it on per member) |
| Results in your files | Automatic | A finished task's report and files land in Shared folders under `crew/<project>/<task>/` | Ch 23 |
| Start Crew from chat | Any agent's chat | "Fix the login bug in project X": the agent starts the task and is called back when it ends | Ch 24 |
| Approve a merge an agent asked for | Bell | An agent can only *ask* to merge; you Approve it in the bell | Ch 24 |
| Projects | Side list / project page | Tasks by project, stats, **New task here**, **New project** | Ch 24 |
| Tidy up finished | Project page | Clears finished tasks off the page; their reports and files stay under `crew/` | Ch 24 |
| Environment card | Project | What the project runs on (Python, packages, start and test commands, variables, keys): Prepared / Partly prepared / Not prepared; a coding agent can prepare it | Ch 24 |
| Live sites → Publish a site | Overview / a task's Files tab | Put a **page** on the web at its own address, with no sign-in to open it. Stop it (it keeps its address), start it again, Remove it | Ch 24 |
| Live sites → An app that runs | Publish a site | Starting an app needs an admin ("Only an admin can start an app on this Flight Deck") | 🚫 for members |

## 15. Custom tools ⚙

Small programs that Crew's coding agents build for jobs code does better than a
model (totalling a CSV column, converting a format, checking a rule). Flight
Deck runs every call for your agents, with your secrets and files.

**Admin first.** An admin turns Custom tools on under Settings; otherwise the
page says "Custom tools are off on this deck". **Making** a tool also needs
Crew approval (§14). Using tools others offer you doesn't.

**Open it** from Settings → Custom tools (and from the bell when there's news).
Code: `flight-deck/src/components/customTools/` (CustomToolsPage, NewToolDialog,
ToolSheet); backend `captain_claw/flight_deck/external_tools*.py`.

| Feature | Where | What it does for the user | Mark |
|---|---|---|---|
| Mine / Shared with me / Everybody | Tabs | Your tools, tools people offered you, and the deck's catalog (Approvals is admin-only) | Ch 25; Approvals 🚫 |
| Your agents suggest | Mine | Tools an agent thought would do a job better than by hand. Nothing is built until you say so: **Build it** or **Dismiss** | Ch 25 |
| New tool, step 1: describe | New tool | Title and **What it should do** | Ch 25 ⚙ (Crew) |
| New tool, step 2: check and build | New tool | Flight Deck's check (on your model) answers with a fuller spec, what it changed and anything **Already there?**; pick the checked spec or your own, edit **The spec the coding agent gets**, pick the coding agent, **Build it** | Ch 25 |
| Build → live | Crew + bell | A Crew task builds it; Flight Deck runs its tests and examples; it goes live when you approve the merge | Ch 25 |
| Ask an agent to build one | Chat | "Make me a tool that totals a CSV column" | Ch 25 |
| Tool card status | Mine | Live / Not published / Publish failed | Ch 25 |
| Try it | Tool sheet | Run it with sample input, **Run as** one of your agents; "Asks you questions" shown when it does | Ch 25 |
| Tools it runs | Tool sheet | Which agent tools it uses (read, write, send) or other custom tools; a note when its maker hasn't offered you all of them, with **Offer them too** | Ch 25 |
| Your secrets for it | Tool sheet | Keys it needs, stored for you (a tool always runs with the runner's secrets) | Ch 25 |
| Publishing | Tool sheet | **Publish main**, **Make live**, and why a publish failed | Ch 25 |
| Versions | Tool sheet | Go back to a kept version | Ch 25 |
| Change it | Tool sheet | "What to change, or the failure to fix" goes to Crew | Ch 25 |
| Offer it | Tool sheet | **People…** (chosen people) or **Offer to everybody**, which may wait for an admin's approval | Ch 25 |
| How it's used | Tool sheet | Usage counts | Ch 25 |
| Delete tool | Tool sheet | Removes your tool | Ch 25 |
| Add (shared and catalog tools) | Shared with me / Everybody | "Add it first to let your agents use it: it runs as you, with your secrets and files." Added tools follow new versions automatically | Ch 25 |
| Remove from my tools | A tool you added | Your agents stop using it | Ch 25 |
| "‹Tool› asks" | Pops up | A tool asks you a question mid-run; answer or **Don't answer** | Ch 25 |
| Agents use them | Chat | Your agents call your added tools by themselves | Ch 25 |

---

## 16. Not in the user view (out of scope for this video)

These belong to the full Flight Deck that non-kiosk members can open. Many are
power-user or experimental. They are candidates for a separate "power user"
series:

- **Usable by a member:** Quick chat, Agent Desktop, Council, Basna/Vatra, Bat,
  Code, Spawn Agent, Agent Forge (page), Skills, Library (tiers, archetypes),
  Flows, Autonomous Work, Scheduler (page), Observatory, Today, Processes (own),
  Browser LLM, GPU Cloud, Stats, Deep Memory, VFS, Hosting (static), Village,
  Games, Prompt Builder, Frontier Horizon, Director, Pinned messages / files,
  Shared clipboard.
- **Admin-only (🚫):** Users, Admin, Agent Folders, Admin console, Captain
  Swarm, Library → Publish to team, archetype editing, Custom tools →
  Approvals, Hosting "app", Link folder, other users' processes, Antigravity.
- **Gaps for non-kiosk members:** they have no way to reach own AI keys, plan
  usage, the Taskboard or Activity without switching to the simple view.

## 17. Admin-only items seen inside the user view (🚫, skip)

Switch to admin view · Captain Swarm · Shared folders → Link a folder · Custom
tools → Approvals · Agent backups → Decks this one trusts · Profile →
Deck-wide editing (members see it read-only) · OAuth client setup, MCP stdio and
deck-wide servers, Telegram approvals · "On network (admin)" tile.
