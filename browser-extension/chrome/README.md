# Captain for Chrome

Send the page you're on to one of your Captain agents. The extension reads the
page in your own browser — signed in, as you see it — downloads the files it
links to with your session, and hands it all to the agent you pick. The page
lands in your **web-clips** shared folder and in the agent's chat, so you can
carry on in Flight Deck.

Works in Chrome, Edge, Brave and Arc (Chromium 116+).

## Install

1. In Flight Deck: **Connections → Browser → Download the extension**, and
   unzip it (or use this folder from a checkout).
2. Open `chrome://extensions`, turn on **Developer mode**, press **Load
   unpacked** and pick the `captain-for-chrome` folder. Pin the Captain icon.
3. Click the icon, type your Flight Deck's address and press **Connect**. A
   tab opens on your deck: sign in if asked, check the code matches, approve.

No password is kept in the extension — it holds a token of its own, which you
can disconnect from Connections → Browser (or from the extension's settings).

## Use

Open it with the toolbar icon, **Alt+Shift+C**, or right-click a page, a
selection, a link or a picture. Pick an agent and what it should do:

| Action | What the agent is asked |
|---|---|
| Handle it | Decide what's useful: the gist, what the files are, what the page asks of you |
| Summarize | One line on what it is, the key points, what to act on |
| Save to space | Keep the whole page and its files, with notes.md |
| Grab files | Download the page's documents, name them, write files/INDEX.md |
| Extract data | Tables, prices, people, dates → clean CSV |
| Dig deeper | Check the claims elsewhere, find what's missing, with sources |
| Fact-check | Each key claim: true, doubtful or false, with a source |
| Draft a reply | What you'd send or fill in — never sent |
| Make tasks | Next steps with dates and owners |
| Watch it | A check in Flight Deck's Scheduled tasks (every hour, 6 hours, day or week) that tells you what changed |
| Remember | The facts kept in the agent's memory |
| Ask | Your own question |

**Anonymize this page** (or the selection — also on the right-click menu as
"Anonymize with Captain") needs no agent: Flight Deck takes the names,
numbers, emails and addresses out and gives back tokens like ⟦PERSON_1⟧. The
real values stay in Flight Deck, in a vault named after the site
(`www.example.com`), so the same person gets the same token on every page of
it. Copy the result, keep it in Web clips, or paste an answer that has tokens
in it and put the real names back.

**Read it as** gives any action a lens. The agent you picked suggests its
own, from who it is (its personality and description): a VC analyst offers
"Partner memo" or "Founder red flags", a lawyer "Opposing counsel". They're
made once and again only when the agent changes (↻ asks for new ones). The
general lenses — investor, buyer, legal, skeptic, competitor, engineer,
newcomer, your team — stand in meanwhile and stay under **More…**; or type your
own.

**Files on the page** can be searched (name, type, link text, site) and
filtered by kind — PDF, Docs, Sheets, Slides, Archives, Data, Images, Audio &
video — with **Tick these** / **Untick these** for what's shown.

**Watch it** never uses the agent's own cron: Flight Deck puts the check in
its **Scheduled tasks** itself — one task per page and agent (watching the
same page again updates it), running on that agent, its results in the
agent's chat. The panel shows the task and opens it in Flight Deck to change
or stop it. Each check compares the page with `watch/latest.md` in the clip
folder and says only what changed.

**New topic** (by the agent picker, and under an answer): the agent's chat
starts a fresh session, as `/new` does in Flight Deck — the last one is kept
in its history. Not while that agent is still answering a clip.

**Text size** (Settings): Small, Medium or Large — the whole panel, kept for
next time and in every window.

While the agent works, the panel follows it: the newest step and the answer
as it's written stay in view, with a working bar (what it's doing, how long)
pinned at the bottom. Scroll up to read and it stops following; the bar's ⌄
(or scrolling back down) picks it up again.

**What goes with it**: the page as Markdown (always), the selection, the whole
page as one `.mhtml`, a screenshot, the tables as CSV, and the files the page
links to (documents are ticked for Handle it, Save and Grab files).

## Let your agents use this browser

Settings → **Agents can use this browser**. Your agents' `browser` tool then
works here (`target="chrome"`), signed in as you, while you watch:

- Each agent works in **tabs of its own**, in a tab group named after it. It
  can't touch your other tabs — drag a tab into its group to share it.
- It reads the page with refs it can point at (`read_page`, `find`), clicks,
  types, fills forms, scrolls, drags, hovers, takes screenshots, reads the
  console and network log, runs a script — and does what this panel does:
  reads the page as Markdown, lists its files, downloads them **with your
  session** into `vfs:web-clips/…`, saves the whole page as `.mhtml`.
- Chrome shows "Captain for Chrome started debugging this browser" while an
  agent works — that's how it clicks and types for real. Pressing **Cancel**
  on that bar stops the agents that were working.
- The panel shows which agent is at work and what it just did, with **Show**
  (bring its tab to the front) and **Stop** (it can't use the browser until
  you **Let it again**). The panel switches to the agent that starts working,
  so you can talk to it right there.
- Flight Deck's own pages, `chrome://` pages and anything that isn't a web
  page are off-limits. Only your own agents can use it; a shared agent's
  member chats never can.
- **Bring an agent's tab to the front while it works** — off by default, so
  an agent working in the background doesn't take your tab away.
- **Back where you were** — when an agent's tab took the front (that switch,
  or a link it clicked opening a new tab), the tab you were on comes back when
  the agent's run is over. If you picked another tab meanwhile, it stays.

It uses two permissions the extension has from install: **debugger** (real
clicks, typing and screenshots in the agent's tabs — Chrome doesn't let an
extension ask for it later) and **tab groups**. Nothing uses them until you
turn the switch on.

## What lands where

```
vfs:web-clips/<day>-<site>-<title>/
  page.md         the page as text: details, main text, its files, tables and links
  selection.md    what you selected
  page.mhtml      the whole page (opens offline in a browser)
  screenshot.png  what was on screen
  tables/*.csv    the page's tables
  files/*         the files the browser downloaded
  clip.json       where and when it came from
```

Every agent of yours reaches the folder through `vfs:web-clips/…`; what the
agent makes for a clip (notes, CSVs, a brief) goes there too.

## Permissions

- **activeTab, scripting** — read the page you send.
- **tabs** — show the title and address of the page you're on.
- **pageCapture** — save the whole page as `.mhtml`.
- **contextMenus, sidePanel, storage** — the right-click menu, the panel, your settings.
- **Sites** are asked for one at a time (or all at once from settings): to read
  a page you didn't open Captain on, and to download a file from another site.
- **debugger, tabGroups** — used only while "Agents can use this browser" is
  on: your agents work in tabs of their own, grouped under their name, with
  real clicks and typing. Chrome won't let an extension ask for the debugger
  later, so it's asked for at install.
- **alarms** — keeps the agents' connection to Flight Deck alive.

The page's text is marked for the agent as web content: instructions written
in a page are never treated as yours.
