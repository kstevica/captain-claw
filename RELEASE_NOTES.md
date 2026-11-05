# Captain Claw v0.8.11 Release Notes

**Release title:** Hold
**Release date:** 2026-11-05

A message you write while the agent is still busy now waits and goes when the agent is free. It's no
longer refused and lost. A reply that comes while your phone is locked arrives when you come back,
and the kiosk's agent column is easier to read, with labelled **New agent** and **New group** buttons.

3 pull requests: #432, #433 and #434.

## New

- **Messages wait while the agent is busy.** A message sent while the agent was still on a turn (one
  started from the glasses, WhatsApp, another tab, or a Stop still landing) used to be refused with
  "Agent is busy processing another request. Please wait." and dropped. On a phone you couldn't send
  at all while busy, because the composer showed only Stop. Now:
  - The message waits under the transcript as a dashed bubble, *Sends when ‹agent› is free*, with
    **Don't send**. The send button stays next to Stop once something is written.
  - When the agent is free, the oldest waiting message goes as a normal message, one turn each, ahead
    of the queue's next task. Slash commands still go at once.
  - Waiting messages are kept in the browser, so a reload or a phone dropping the page doesn't lose
    them. The chat on an agent card shows them too.
  - **Send at next step** hands a waiting message to the running turn as a note, which the agent reads
    at its next step (the same path as the queue's **Insert now**). If the turn ends just as it goes,
    the message waits again and goes on its own.
- **A clearer agent column in the kiosk.**
  - The header has two labelled buttons: **New agent** and **New group**. In a narrow column they
    shorten to **Agent** and **Group**. While you drag an agent, its row becomes the drop target,
    *Drop here for a new group*.
  - Each row has the agent's initials with its state dot on the corner. At rest the only control is
    the power switch. Settings, ⋯ and Leave take no width until you hover or focus the row, so names
    get the full width (on touch screens they're always shown).
  - States are one word under the name: **Asleep**, **Off**, **Offline**. A busy agent reads
    *Working…* in amber. Off agents are dimmed.
  - Search has a clear (×) button, and Esc clears it. Group headers are bigger, and so are their
    colour dots and chevrons.
  - On a phone, the header has a **New group** button, and rows use the same one-word states.

## Fixed

- **A reply that came while your phone was away arrives when you're back.** A locked or backgrounded
  phone stops the page, and its chat connection could die without the page knowing. A reply sent
  meanwhile never arrived, and nothing reconnected. Now:
  - Back after 3 seconds or more, every chat checks its connection, and a dead one is replaced at
    once. The agent replays the session, and the reply lands once, freeing the lane, marking it
    unread and moving the queue on.
  - A connection that comes back mid-turn keeps the lane busy until that turn ends, and a message you
    write meanwhile waits.
  - The replay no longer turns each tool-calling step into an empty answer bubble. The glasses HUD
    skips those steps too, and empty answer rows already in a log no longer show.
  - The connection check is answered by Flight Deck itself and doesn't wake an agent that is asleep.

## Upgrading

1. **The desktop app**: **Check for Updates…** (it also offers 0.8.11 within a minute of starting).
   New installers are on this release, but you need one only for a fresh install:
   `Captain-0.8.11-arm64.dmg` (Apple silicon), `Captain-0.8.11-x64.dmg` (Intel) and
   `Captain-Setup-0.8.11-x64.exe` (Windows). They aren't signed, so macOS asks first (System
   Settings → Privacy & Security → **Open Anyway**), and so does Windows (**More info → Run anyway**).
2. **Each Flight Deck**: **Console → Updates**, or pull by hand. No new dependencies. Restart Flight
   Deck, then every agent: agents need the restart to tell a reconnecting chat that a turn is on, and
   to mark tool-calling steps in the replay. Hard-refresh every browser.

Devices need nothing: firmware stays **0.20.4** / PaperColor B **1.0.31**. BotPort hubs need nothing:
0.8.11 keeps protocol 2. Captain for Chrome and Captain for Mac need nothing from this release.

## Pull requests

#432 Kiosk: clearer agent column — labelled New agent / New group, calmer rows · #433 Hold messages
sent while the agent is busy; "Send at next step" · #434 A reply that came while the phone was away
lands when it's back
