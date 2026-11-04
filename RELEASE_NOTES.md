# Captain Claw v0.8.8 Release Notes

**Release title:** Spyglass
**Release date:** 2026-11-04

0.8.6 gave your agents a log. 0.8.8 lets them see what you're looking at.

**Captain for Chrome** sends the page you're on to one of your agents. It reads the page in your own
browser, signed in and as you see it, and fetches the files it links to with your session. Pick what
the agent should do: summarize it, keep it with its files, pull out its data, fact-check it, draft
your reply, or watch it for changes.

**Captain for Mac** does the same for any app. Hold **fn** for three seconds, or click its menu-bar
icon. It reads the window's text through Accessibility, takes **your selection as what you mean**,
notes what's under the pointer, and adds a screenshot if you want one. You type or say the task. The
answer comes back in a panel beside your work, and **Paste into Mail** puts it where you were.

Both land in a shared folder (**web-clips** and **screen-clips**) and in the agent's own chat, as your
own words, so you can carry on in Flight Deck.

Also in 0.8.8, **fixes from a day of watching real people use the cohort deck**:
- Agents run the script they just fixed and read past a long report's first section.
- The chat shows a card for every file a turn made, and progress lines in your language.
- Agents no longer report delegations or schedule changes that never happened.
- One shared Chromium per deck.
- One more message past a spent daily allowance, so a job isn't cut off one step from the end.
- Public links that expire.

10 pull requests, the same day as 0.8.7.

**Restart Flight Deck, then every agent.** See [Upgrading](#upgrading). **Public file links now expire**
after 7 days: see [Changed](#changed).

## Highlights

### Captain for Chrome: the page you're on, to an agent

Your agents' own web tools fetch pages from a server, which means no cookies, no sign-in and often a
bot wall. The extension reads the page **in your browser instead**: a portal you're signed in to, an
intranet, a paywalled article, a dashboard drawn by JavaScript. Then the agent works on exactly what
you saw.

- **Open it** with the toolbar icon, **Alt+Shift+C**, or by right-clicking a page, a selection, a link or a
  picture. It works in Chrome, Edge, Brave and Arc (Chromium 116+).
- **Twelve things to ask:**

  | Action | What it does |
  |---|---|
  | Handle it | Captain decides what's useful |
  | Summarize | The gist and the key points |
  | Save to space | The whole page and its files, with notes |
  | Grab files | Downloads the documents, names them, writes an index |
  | Extract data | Tables, prices, people and dates as CSV |
  | Dig deeper | Checks the claims elsewhere, with sources |
  | Fact-check | Each claim true, doubtful or false |
  | Draft a reply | What you'd send, never sent |
  | Make tasks | Next steps |
  | Watch it | A scheduled check for changes |
  | Remember | Keeps the facts in the agent's memory |
  | Ask | Your own question |

- **Read it as** gives any action a lens, and the agent suggests **its own** from who it is. A VC
  analyst offers "Partner memo", a lawyer "Opposing counsel". The eight general lenses stay under
  **More…**, or you type your own.
- **The page's files**, fetched with your session, can be searched and filtered by kind (PDF, Docs,
  Sheets, Slides, Archives, Data, Images, Audio & video). The ones that can't be fetched are listed for
  the agent.
- **Watch it** puts the check in **Flight Deck's Scheduled tasks**, never in the agent's own cron. There
  is one task per page and agent, it runs hourly, every 6 hours, daily or weekly, and each check says
  only what changed. The panel links straight to the task.
- **The panel follows the agent**: its newest step and the answer as it's written, a working bar with
  the elapsed time, files to download from the clip folder, follow-up questions, **New topic**, and a
  text size of your choice.

It goes into the agent's main chat as your own message, with the page attached as a file and a note
that the page is web content, never your instructions. It counts against your daily allowance like
anything else you say.

**Connect it** in Flight Deck: **Connections → Browser → Download the extension**. Unzip it, open
`chrome://extensions`, turn on **Developer mode**, press **Load unpacked**, then type your deck's address
and approve the code the deck shows. No password is kept in the extension: it holds a token of its own,
which you can disconnect any time. It isn't in the Chrome Web Store yet.

Clips land in `vfs:web-clips/<day>-<site>-<title>/`: `page.md`, the selection, `page.mhtml`, a
screenshot, the tables as CSV, the files, and `clip.json`. All your agents can reach it there.
[More](https://github.com/kstevica/captain-claw/blob/main/browser-extension/chrome/README.md) ·
[Design](https://github.com/kstevica/captain-claw/blob/main/docs/captain-for-chrome.md)

### Captain for Mac: what's on your screen, from any app

A menu-bar app for macOS 14 and later, Apple silicon and Intel.

- **Ask** by holding **fn** alone for 3 seconds (you can change the key and the time), or by clicking
  the menu-bar icon or an optional floating button.
  - A ring fills by the pointer while it counts.
  - Any other key cancels it, so fn shortcuts keep working.
  - The app you were in stays the active one.
- **What it reads:**
  - **The selection**, sent first as what you mean.
  - **What's under the pointer** and the line around it.
  - **The window's text**: what's on screen first, the rest after.
  - A screenshot of the window or the whole screen, if you turn it on.
  - Text read off the screen image (OCR, on the Mac) when an app shares none.

  It works in Chrome and Electron apps too. In a terminal it reads the end of the scrollback. Each piece
  is a chip you can switch off or look at before sending.
- **Say the task** with the mic, or have it listen when Captain opens. *Send when I stop talking* sends
  on a pause, or when you let go of the held key. Or type it. Left empty, the agent decides what's
  useful.
- **One-click actions**: Explain, Summarize, Draft a reply, Improve, Translate, What's wrong?, Make
  tasks, Remember, Save it.
- **The answer** streams in the panel with the agent's steps. **Copy** it, or **Paste into <app>** to
  drop it where you were, and ask follow-ups about the same capture.

Captures land in `vfs:screen-clips/<day>-<app>-<title>/` (`screen.md`, `selection.md`, `screenshot.png`,
`clip.json`) and in the agent's chat as your words. The capture is marked as screen content, never your
instructions.

**Get it** from the [GitHub release](https://github.com/kstevica/captain-claw/releases/tag/mac-v0.1.0), or
from Flight Deck: **Connections → Browser → Download Captain for Mac**. A deck hands out its own build
when it has one, and otherwise the GitHub release. Then:
1. Unzip it and move it to Applications.
2. It isn't notarized yet. If macOS won't open it, go to System Settings → Privacy & Security and press
   **Open Anyway**.
3. Allow **Accessibility**. Screen Recording, the microphone and speech recognition are only needed for
   the screenshot, OCR and voice.
4. Connect it to your deck the same way as the extension.

It needs a **0.8.8 deck**. [More](https://github.com/kstevica/captain-claw/blob/main/captain-for-mac/README.md) ·
[Design](https://github.com/kstevica/captain-claw/blob/main/docs/captain-for-mac.md)

**Connections → Browser** lists your connected browsers and Macs side by side, with **Disconnect** for each.

### Fixes from the cohort deck

On 4 November we read what real test users on the cohort deck ran into. These are the fixes.

**Agents finish the job**
- The "you've written this file already" stop now counts only whole-file writes, not edits.
- After a fix, the re-run is a new call and goes through.
- A read's page range is part of the call, so the next section of a long report isn't refused as a
  repeat.
- Writing the same file over and over, or repeating an identical command with nothing changed, is
  still stopped.

**The chat shows what happened**
- Runtime messages (a guard's stop, a relayed result) no longer look like your own words.
- Every file a turn made in `saved/` gets a card with **Open** and **Download**.
- Replies show file names instead of server paths.
- Progress lines fold into the turn's Activity after the answer, and are written in **your
  language**.

**No made-up reports**
- A reply that reports other agents' work when no hand-off answered is turned down. So is one that
  reports a schedule change no cron call made.
- Cron acts by job ID, never by position. Every change names the job it touched.
- DeepSeek tool calls written as text are run, not shown.

**First-run walls**
- Mail and calendar say "Google isn't connected … Flight Deck → Connections → Google" instead of a
  policy error.
- The new-agent picker says up front when an archetype works in mail, with Connect buttons.

**The browser on every agent**
- Flight Deck keeps **one Chromium for all its agents**. Before, each downloaded its own ~700 MB copy.
- A host missing Chromium's system libraries gets one plain answer with the admin's `install-deps`
  command, and web reading falls back to plain HTTP.
- Admins get a Line notice if Chromium can't start.

**Usage**
- The daily meter is floored by the spend ledger, which syncs every 5 minutes.
- The first refused message of the day also goes on your Line.
- **One message past a spent allowance** goes through if your agents worked in the last 30 minutes,
  so a job isn't cut off one step from the end. Scheduled tasks never get it.

**Fleet and other decks**
- `/fd/fleet` lists only the caller's own agents.
- Agents on your other decks (BotPort) are listed and reachable only for admins, unless the admin
  turns on **Every user's agents too** in Console → Network.

**The desktop app** has no Captain Swarm (its one user is the local admin). Its agents have 69 tools.

## Upgrading

1. **Each Flight Deck**: pull and `pip install .` where Captain Claw is installed non-editable, or use
   **Console → Updates**. There are no new Python packages.
2. **Restart Flight Deck, then every agent.** Agents need the restart for:
   - file cards and the origins of runtime messages,
   - the loop-guard and false-claim fixes,
   - progress lines in your language,
   - the shared Chromium,
   - the `browser` / `browser:mac` turn sources the two apps use.

   Then hard-refresh every browser.
3. **On a server that can't start Chromium**, run once as root: `python -m playwright install-deps
   chromium`, with the deck's Python.
4. **Captain for Chrome**: **Connections → Browser → Download the extension**, then Load unpacked.
5. **Captain for Mac**: the [GitHub release](https://github.com/kstevica/captain-claw/releases/tag/mac-v0.1.0)
   or **Connections → Browser**.
6. **The desktop app**: **Check for Updates…**.

Devices need nothing: firmware stays **0.20.4** / PaperColor B **1.0.31**. BotPort hubs need nothing:
0.8.8 keeps protocol 2.

### Changed

- **Public file links expire.** New links last 7 days by default and at most 30. At start-up, links
  that never expired get an end date 30 days after they were made, but at least a week away, and each
  owner is told on the Line. Set `FD_PUBLIC_FILE_LINK_MAX_DAYS=0` to let links live for ever.
- **Agents on other decks are for admins** by default (Console → Network → **Every user's agents
  too** opens them up).
- **`/fd/fleet`** answers with the caller's own agents only.
- **Cron** removes and changes jobs by ID, never by list position.
- **The desktop app** has no Captain Swarm.

### New settings

| Setting | Default | What it does |
|---|---|---|
| `FD_USAGE_FINISH_MINUTES` | `30` | One message past a spent allowance when your agents worked this recently; `0` turns it off |
| `FD_PUBLIC_FILE_LINK_DAYS` | `7` | How long a new public file link lasts |
| `FD_PUBLIC_FILE_LINK_MAX_DAYS` | `30` | The longest a link may last; `0` lets links never expire |
| `FD_BOTPORT_OUTBOUND_USERS` | off | Other decks' agents for every user, not just admins (also Console → Network) |
| `FD_MAC_APP_URL` | the GitHub release | Where **Download Captain for Mac** goes when the deck has no build of its own; empty turns it off |

## Pull requests

#410 No Captain Swarm in the desktop app · #411 Loop guards let edit → re-run and sectioned reads through
· #412 Chat: runtime rows as notes, a card for every file, narration folds away · #413 Cohort findings:
browser, false claims, first-run walls, usage, fleet, links · #414 Captain for Chrome · #415 The
extension's lib files · #416 The panel follows the agent, file search, the agent's own lenses, text size
· #417 Watch it in Scheduled tasks, New topic · #418 Captain for Mac · #419 Decks hand out Captain for
Mac from its GitHub release
