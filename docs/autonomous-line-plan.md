# Line (the autonomous line) — plan, 2026-11-03

Status: every decision below is made. **P0 is built** (#387): the store,
`line.post`, the bell over the Line, the Line page (full view) and dialog
(kiosk), fleet changes on the Line. **P1 is built** (#389): what agents did
on their own reaches the Line from the deck, finished runs leave a copy, and
agents no longer get the fleet notice. **P2 is built** (branch
`feat/line-p2`): the `line` tool and the one-line count at the start of the
owner's turns. P3–P4 are not. Users see it as
**Line**; the internal id is `line`.

## The idea in one paragraph

Each user gets one **line** on the deck. It sits above the agents (the
"super-agent" place) but is not an agent process: it is a timeline. Everything
the system says on its own lands there: scheduled-task results, Autonomous Work
nudges and proposals, run completions, fleet changes, deck events, bell items.
What an agent ran itself also keeps a copy in that agent's chat; the rest stays
out of agents' chats. Agents read it with a new tool when they
need to, and see a one-line "N new on the line" hint at the start of a turn. The
user reads it in Flight Deck (full view and kiosk) and picks where else it goes:
**one agent**, **Telegram**, **WhatsApp**.

## Why: where automatic messages go today

Five different paths, each with its own rules:

| What | Path today | Where it lands |
|---|---|---|
| Fleet changes ("Agent X has joined the fleet") | `_notify_fleet_change` opens a WS to **every** agent of the owner (`flight_deck/server.py:6952`) | each agent's session; moved to its AUTO lane when that is on (`web/ws_handler.py:199`), plus a `fleet_events` note in the main session |
| Automated-turn results (cron, fd_scheduler, autonomy, plan, taskboard) | run on the agent's AUTO lane (`web/chat_handler.py:163`), mirrored back (`_mirror_automation_result`, `:453` → `flush_automation_results`, `:485`) | that agent's lane A as compact cards (#265), relayed by its chat bridges, **kept as context** |
| Autonomy nudges / proposals | `owner_notify.notify_owner` (`flight_deck/owner_notify.py:143`) | bell + WhatsApp + Telegram + devices, switches in the autonomy config (`config.py:1245`) |
| Bell items (shares, payments, Gmail send / auto-reply, invites, Bat, Basna, exams, Forge, WA Web, statements, bills, profile research…) | `db.add_notification` from ~25 call sites | bell only |
| Background-run completions (Basna, Vatra, Code, Crew, Bat, deck update, background delegation) | `agent_notify.notify_source_agent` (`flight_deck/agent_notify.py:33`), `trigger_response` | the agent that started the run, which relays to the user |

Problems this causes:
- An agent's main chat fills with things that are not conversation, and they
  become context for every later turn (fleet notes, mirrored results).
- The user has no single place to see "what happened while I was away".
- Only autonomy has channel switches; everything else reaches a phone or doesn't
  by accident of which path it took.

## Shape

### 1. One store, one door (deck)

New `captain_claw/flight_deck/line.py`.

Table `line_items` (FD db):
`id, user_id, at, kind, source (agent slug or deck part), title, body
(markdown, capped), needs_you (bool), ref_type, ref_id, trust
('deck' | 'agent' | 'untrusted'), delivered (json: what each channel did)`.

Table `line_cursors(user_id, agent_slug, last_id)`: how far each agent has read.

Retention 30 days by default (setting).

**One write call**, the only door:

```python
await line.post(user_id, kind, title, body, *, source="", needs_you=False,
                ref=("autonomy_action", action_id), trust="deck")
```

It stores the item, then fans out to the user's channels in the background.
Never raises. `owner_notify`, the bell writes, fleet changes and mirrored
results all go through it, so every channel rule lives in one place.

Agents post through `POST /fd/line/agent/post` with their agent token (same
auth as the Telegram inbox routes).

### 2. Kinds (first set)

| kind | from | needs_you | phones by default |
|---|---|---|---|
| `result` | automated-turn results (the MIRRORED kinds) | no (yes when it failed) | yes |
| `proposal` | Autonomous Work proposal waiting | yes | yes |
| `nudge` | Autonomous Work nudge | no | yes |
| `run` | Basna / Vatra / Crew / Bat / Code finished (a copy, see below) | no (yes when it failed) | yes |
| `fleet` | agent joined / left / stopped / crashed | no | **never** |
| `deck` | deck update, restart, rollback | no | Needs-you level only |
| `payment` | purchases to approve, payment reminders, statements, recurring bills | yes for "approve a purchase" and due reminders | yes |
| `ask` | Bat asking its owner (stalemate), admin requests (Telegram link) | yes | yes |
| `share` | shares, shared usage, context packs, invites accepted | no | Needs-you level only |
| `mail` | Gmail sends, topic auto-replies | no | Needs-you level only |
| `agent` | Forge, onboarding exam, lessons, setup, profile research | no (yes when it failed) | Needs-you level only |

Every item keeps the fine-grained `type` the bell uses today (`payment`,
`run_error`, `bat_ask`, `autonomy_proposal`, …) next to its `kind`, so nothing
that reads a bell type breaks.

### 2a. The bell reads the line (decided 2026-11-03)

There is one store. The bell stops having its own table and becomes the
line's **Needs you** list; the line page shows everything.

- **Items get two states**: `read_at` (seen) and `resolved_at` (answered).
  An item that needs you is resolved when it is answered where it lives:
  proposal approved / rejected / expired, purchase approved / declined, Bat
  question answered, Telegram link approved. A type with no answer hook yet
  counts as resolved once it is read.
- **The bell lists `needs_you` items**: unresolved ones on top (newest
  first), then resolved ones by date, and an "Open the line" link for the rest.
  Its badge counts unread `needs_you` items. The line row's badge counts
  everything unread. One read state: reading an item in either place marks it
  read in both.
- **What leaves the bell**: items that don't need you (shares, Gmail sends,
  invites accepted, Forge / exam done, successful runs) are only on the line
  now. The "Needs you" phone level (section 3) is the same set as the bell, so
  the bell and the phones never disagree about what matters.
- **The 25 call sites stay as they are.** `db.add_notification(user, type,
  title, body, ref_type, ref_id)` becomes a thin shim over `line.post`: it maps
  `type` → `kind` and `needs_you` with the table in section 2. The other bell
  methods move over the line too, with the same behaviour:
  `list_notifications`, `count_unread_notifications`, `mark_notification_read`,
  `mark_all_notifications_read`, `list_notifications_by_ref`,
  `tell_notifications_plain` (an item keeps only its title, read — the text is
  gone) and `count_notifications_since` (per-day caps and per-reason dedupe).
- **`GET /fd/notifications` keeps its response shape** (`id, type, title,
  body, ref_type, ref_id, read, created_at`), so `NotificationCenter.tsx` only
  changes its grouping and adds the "Open the line" link. `waiting_bells`
  (#287) is gone: an unresolved item is never past the page, it is always on
  top. `bell_backfill` (#240) stays as a safety net for a proposal whose item
  couldn't be written.
- **Retention never drops an unresolved item**, however old.
- **Migration**: on first start the deck copies the `notifications` rows into
  the line once (marker in the db), as `kind` from the type map, `read` →
  `read_at`, proposals still awaiting → unresolved. For one release the shim
  also writes the old table, so a deck that rolls back (Updates' auto
  rollback) still has its bell. The old table is dropped a release later.
- **Devices** (P3): their count can follow the unresolved `needs_you` items
  instead of waiting proposals only — the "beep for any bell item" idea from
  the notify-channels work. Off until then; the count stays proposals-only.

### 3. Channels

Per user, in the line's settings. The channels combine freely (decided
2026-11-03): one agent, Telegram and WhatsApp can all be on at once, each with
its own level; only the agent channel is limited to a single agent.

- **Flight Deck line**: always on.
- **Agent**: none, or exactly one agent. Each item goes into that agent's main
  chat as a compact card (the #265 `CompactMessage`, label "Line · …"), stored as
  a `system_note`: it is context, it never asks for a reply, it never runs a
  turn (no `trigger_response`, so no loops and no cost). The chat bridges bound
  to that agent show it, as mirrored results do today. This is the agent you
  talk to about the line ("what failed overnight?"). Items this agent posted
  itself are skipped: its own copy is already in its chat (4.4).
- **Telegram**: the deck's shared bot, owner chat (`fd_scheduler._deliver_telegram_owner`).
- **WhatsApp**: `whatsapp_bridge.push_to_waid` to `notify_waid` (allowlist and
  /mute still apply).
- **Devices**: unchanged — they beep for `needs_you` items and show the count.

The agent, Telegram and WhatsApp channels each have their own level:
**Everything · Needs you · Off** (so e.g. the agent gets everything while
WhatsApp only gets what needs you). Quiet hours
(existing autonomy setting) silence phones for everything except what the user
marks urgent; `fleet` never goes to a phone. A digest ("every N hours, one
message") can come later.

Migration: the autonomy switches `nudge_to_whatsapp`, `notify_waid`,
`notify_telegram`, `notify_devices` become the line's channel settings (old
values read once as defaults). Connections → Notifications becomes the line's
channel card.

### 4. Agent side

1. **Tool `line`** (owner-only, read-only), actions:
   - `news` — items since this agent's cursor; advances it.
   - `recent` — last n, filter by kind / source agent / needs_you.
   - `read` — one item in full.
   - `search` — text search.

   Served by `/fd/line/agent/*`. Dropped for public, BotPort, Iskra and
   shared-member turns (`drop_unusable`, like `telegram_inbox`). Bodies come back
   wrapped as untrusted: results carry email and web text.

   **As built (P2):** `captain_claw/tools/line.py` + `captain_claw/line_client.py`;
   always enabled, its schema always sent, shown by the registry only under
   Flight Deck and never on a member's turn (`requires_line`). The deck's routes
   (`/fd/line/agent/news|unread|items|item`) know the agent by the `X-Agent-Auth`
   token they issued; with sign-in on the owner comes from `_agent_owner`, with it
   off it is the local user. Two cursors per agent (`line_cursors`): `told_at`
   moves with the count, `read_at` with `news`, so the count never uses up what
   the tool would read. The first look covers the last day.
2. **Awareness line** (built, P2) at the start of the owner's live turns, next to the clock:
   "Line since your last look: 4 new — 1 needs you (proposal), 2 results, 1
   fleet". Counts by kind only, no text (the Telegram inbox #135 pattern:
   `telegram_inbox_client.news()`, fetched once per turn). Cron, autonomy, peer
   and worker turns neither see nor move it. What the agent posted itself (its
   name as the item's source) is left out of its count.
3. **"From time to time"**: the awareness line plus the tool covers it; an
   agent with a heartbeat (consciousness pulse, being tick) may call `news`
   there. Line items never wake a sleeping agent.
4. **Results keep their copy in lane A, and go to the line too.** (Decided
   2026-11-03.) `_mirror_automation_result` still mirrors the result into the
   agent's main chat as a compact card, kept as context exactly as today, and
   also posts it to the line. Two rules keep it from arriving twice:
   - the agent channel skips items whose `source` is the chosen agent itself;
   - when the line will send the item to a phone (Telegram or WhatsApp on at a
     level that covers it), `post` answers `deck_delivers: true` and the mirror
     is marked `fd_delivers` — the agent's chat bridges show it but don't relay
     it. With the phone channels off, the bridges relay as they do today.

   A standalone agent (no deck) keeps today's mirroring and posts nothing.

   **As built (P1):** the deck posts, not the agent. Every mirrored kind but
   `cron` is a turn the deck runs and hears the reply of, so the agent is
   unchanged (no restart): a scheduled task's run (`fd_scheduler.run_job`, a
   failed run or delivery needs you), an Autonomous Work action or Taskboard
   hand-over ending on the ledger (`AutonomyStore.update_status` → "done" →
   `line.action_done`, once per action; a nudge is already a `nudge` item), a
   plan's step (`plans._step_to_line`). `cron` is the old in-agent scheduler:
   its turns never reached the mirror and stay off the Line. The
   `deck_delivers` rule above comes with the phone channels (P3).
5. **Fleet changes stop being injected** (built, P1). `_notify_fleet_change` posts one
   `fleet` item instead of opening a socket into every agent. Agents still get
   their peers the way they do now (`_peer_agents`); the `fleet_events` note
   goes, the awareness line counts fleet items instead.
6. **Run completions go to both** (decided 2026-11-03; built, P1 — Basna,
   Vatra and Bat answering on another channel leave a copy that doesn't need
   you (web ones were bell items already); a coding session always does; Crew
   tasks were bell items already; background delegation between agents isn't
   copied). Basna / Vatra / Crew /
   Bat / Code and background delegation still `trigger_response` into the
   agent that started them (it was asked, it relays to wherever the user
   asked), and a `run` copy goes to the line. The same two rules as 4.4 keep it
   from arriving twice: the agent channel skips it when the chosen agent is the
   one that started the run, and the line's phone channels skip a `run` item
   whose origin is that same phone channel (the starting agent already relays
   it there).

### 5. UI

(P0 ships the page read-only: a waiting proposal says "answer it in the bell";
inline Approve / Reject and "Hand to agent…" come later.)

- **Full view**: a pinned row at the top of the agent list ("Line", unread
  badge). Opens a timeline of compact cards (reuse `CompactMessage` +
  `CompactMessageDialog`), filter chips (Needs you · Results · Runs · Fleet ·
  Deck), filter by agent, Approve / Reject inline on proposals (the bell's
  endpoints), and "Hand to agent…" on any item (into that agent's chat as an
  owner request, like the Taskboard hand-over).
- **Kiosk**: the same row above the agents.
- **Settings**: the channel card — agent picker (with "None"), Telegram level,
  WhatsApp level, devices switch, quiet hours.

### 6. Later: the line talks back

Typing in the line asks the chosen agent, which answers in the line with the
`line` tool in hand. A morning digest is then just a scheduled task on that
agent ("summarize the line since 08:00 yesterday, send to Telegram").

## Phases

| Phase | What | Restart |
|---|---|---|
| P0 | `line.py` store + `post()` + routes + read-only page; the bell over the line (shim, read path, one-time copy, dual write for a release, `NotificationCenter` grouping); `owner_notify` and fleet changes also post (their phone sends still on) | FD + bundle |
| P1 | agent results also → line (lane-A copy stays), posted by the deck; fleet injection off; run copies | FD |
| P2 | `line` tool, cursors, awareness line | FD + agents |
| P3 | channels: one agent, Telegram, WhatsApp with levels; migrate notify switches; old direct sends removed | FD + bundle |
| P4 | talk in the line; digest | FD + agents |

P0 and P3 must not overlap in a way that double-sends: until P3 lands, `post()`
fans out to nothing but the line page and the bell, and the old phone paths
keep sending.

## Risks

- **Context.** An agent keeps its own results in its main session, so
  "about that report…" still works there. What it lacks is everything other
  agents and the deck posted; the awareness line says it's there and
  `line read` gets it.
- **Double delivery on phones.** The lane-A copy plus the line's phone
  channels could both reach Telegram / WhatsApp; `deck_delivers` → `fd_delivers`
  (4.4) stops the agent's bridges relaying when the line already sends it.
- **Prompt injection.** Results carry untrusted text into whichever agent reads
  the line. Items keep `trust`; the tool wraps bodies as untrusted; the agent
  channel stores them as `system_note`, never as a user turn.
- **Phone spam.** Levels, `fleet` never to phones, quiet hours, later a digest.
- **Multi-tenant.** Per user. A shared agent (FD_AGENT_SHARING) posts to its
  owner's line; members never see it. Admin act-as (`X-FD-Act-As`) reads the
  acted-on user's line.
- **The bell changes under people.** Informational items leave the bell and
  only live on the line; someone used to seeing "X shared a folder" in the bell
  has to look on the line. The bell's footer link says where they went.
- **Rollback.** A deck that rolls back to before P0 reads the old
  `notifications` table; the one-release dual write keeps it current.
- **WhatsApp 24-hour window.** Same as today's nudges: `push_to_waid` handles
  what it can; a line item outside the window may not arrive.

## Decisions (all made 2026-11-03)

1. ~~**Results in the agent's own chat**~~ — **decided 2026-11-03: keep a copy
   in the agent that ran it** (4.4).
2. ~~**Channels**~~ — **decided 2026-11-03: any of agent / Telegram / WhatsApp
   at once, each with its own level** (section 3).
3. ~~**Bell**~~ — **decided 2026-11-03: the bell reads the line** — it is the
   line's Needs-you list, from P0 (section 2a).
4. ~~**Run completions**~~ — **decided 2026-11-03: trigger the starting agent
   + copy in the line** (4.6).
5. ~~**Name**~~ — **decided 2026-11-03: "Line"**.
