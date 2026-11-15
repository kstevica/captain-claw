# Captain's map: one word per thing

What people read in Captain should name each thing one way, everywhere. This
page is the list. `tests/test_flight_deck/test_product_words.py` fails when a
retired word comes back into Flight Deck's copy.

## The eight things an agent has

Always in this order: in menus, in docs, in a tour.

| # | Thing | What it is | Covers |
|---|-------|------------|--------|
| 1 | **Chats** | Talking with it | the chat, its topics, the queue |
| 2 | **Tasks** | What it was asked and sent, and what waits on you | the task list and its Needs you view; hand-overs; mail it picks up |
| 3 | **Automations** | What it does on a schedule | scheduled tasks the deck runs, and the agent's own scheduled jobs |
| 4 | **Memory** | What it keeps in mind and has learned | instructions, insights, onboarding lessons, knowledge, deep and external memory |
| 5 | **Skills** | What it knows how to do | its skills, and the "Things I can do" cards |
| 6 | **Tools** | What it can use | built-in tools, custom tools, MCP servers, connections, its identity and email |
| 7 | **Files** | What it made and keeps | the file library, data tables, shared folders |
| 8 | **Activity** | Every step it took, and what it asked first | the tool-call register, "Why did this happen?", approval cards |

## The person's own places

| Word | What it is |
|------|------------|
| **Today** | The briefing: what waits for you, priorities, what was handled |
| **Inbox** | What the deck and the agents said on their own; **Needs you** is the part blocked on you |
| **Agents** | Your agents (the column is the team) |
| **People** | The people your agents' tasks are with |
| **Settings** | Yours, an agent's, or (admins) the deck's, always named with whose they are |

## Four rules

1. **A new feature is a kind of one of these things.** It gets no new top-level
   name. Approval cards are part of Activity, not a ninth thing. A new way of
   scheduling is part of Automations.
2. **One question, one answer.** "What needs me?" is Needs you. "What did it
   do?" is Activity. "What does it run on its own?" is Automations. A second
   place that answers the same question links to the first instead.
3. **The order is fixed.** Lists of an agent's things follow the table above.
4. **One word per thing.** A thing keeps its word in titles, buttons, hints,
   notices and docs. The words below are retired from what people read; the
   test above holds them out.

## Retired words

| Retired | Use instead | Why |
|---------|-------------|-----|
| Taskboard | **Tasks** | The navigation already says Tasks; two names for one list |
| Needs me | **Needs you** | The chips and the Inbox say Needs you |
| Cron job, cron jobs | **Automations** (an automation) | "Cron" is jargon; the navigation says Automations |
| Captain Claw | **Captain** | The product's name on screen (the README keeps the project's name) |

Code names (`taskboard_store`, `cron_tool`, `needs_me`) stay as they are. Only
what people read follows this page.

## Still to decide

These conflicts were found in the 2026-11-15 naming pass and are left for a
decision. They aren't in the test yet:

- **Line vs Inbox.** The kiosk says Inbox; the full layout's sidebar, the bell's
  footer, the notification settings and the "Your Line" source label say Line.
  The sidebar's Line was kept on purpose. Pick one word for both layouts.
- **Library.** It means the file library and also model tiers plus agent
  templates (full layout). The templates also go by Agent store, Archetypes,
  Spawn and Forge.
- **Topics.** It means a fresh start of a chat and also grouping messages by
  subject.
- **Info.** One name for two dialogs: the agent's (intentions, topics) and the
  app's (account, models).
- **Activity.** In the console it means who is active, while an agent's
  Activity is its tool calls. The console's Audit shows the same register as
  "Tool calls".
- **People vs Users.** People are those your agents' tasks are with; the
  console's "Your people" means deck accounts.
- **Today.** One name for two pages: the kiosk briefing and the full layout's
  per-agent page.
- **Memory.** About seven names for it today (Insights, Lessons, Knowledge,
  Intentions, Deep memory, External memory, "What your agents know about
  you"). This map names the thing Memory; the screens still use the parts'
  own names.
- **Iskra / Village.** Two names for the living beings.
