# Captain for Chrome

Send the page you're on to one of your Captain agents. The extension reads the
page in your own browser — signed in, as you see it — downloads the files it
links to with your session, and hands it all to the agent you pick. The page
lands in your **web-clips** shared folder and in the agent's chat, so you can
carry on in Flight Deck.

Works in Chrome, Edge, Brave and Arc (Chromium 116+).

This is version **0.10.0**. New since 0.9: the browser can be connected to
[several Flight Decks](#several-flight-decks) at once — agents on each of them
can use it, and the panel works with the one you pick at its top.

Before that, 0.9.0 (shipped with Captain Claw 0.9.3) handed a
[robot check](#a-robot-check-on-the-way-you-answer-it-the-agent-goes-on) on an
agent's search or page read to you, and gave the panel
[playbooks](#playbooks--ready-made-ways-to-get-things-done): **Do it here**
chips for the site you're on and a store behind **⚡**. Your recordings count
as playbooks too.

## Install

1. In Flight Deck: **Connections → Browser → Download the extension**, and
   unzip it (or use this folder from a checkout).
2. Open `chrome://extensions`, turn on **Developer mode**, press **Load
   unpacked** and pick the `captain-for-chrome` folder. Pin the Captain icon.
3. Click the icon, type your Flight Deck's address and press **Connect**. A
   tab opens on your deck: sign in if asked, check the code matches, approve.

No password is kept in the extension — it holds a token of its own, which you
can disconnect from Connections → Browser (or from the extension's settings).

## Several Flight Decks

One browser can be connected to more than one Flight Deck — a work deck and a
home one, say. Click the deck's name under **Captain** at the top of the
panel: **Your Flight Decks** lists them, with **Use**, **Open Flight Deck** and
**Disconnect** for each, and **Connect another Flight Deck** pairs one more the
same way as the first (each deck gives the browser a token of its own).

- **The deck in use** is the one the panel works with: the agents you send a
  page to, recordings, playbooks, Recent. Switching keeps what each deck had —
  the agent you picked there, its lenses. A run you started keeps following its
  own deck after you switch, and Recent says which deck each one went to.
- **Agents on every deck** can use this browser, run their web searches and
  read pages here, whichever deck is in use. Each agent works in tabs of its
  own; with two decks connected its tab group says which deck it's from
  (“Researcher · work”), and the panel names the deck beside the agent. **Stop**
  stops that one agent on that one deck. The panel switches to an agent that
  starts working only when it's on the deck in use.
- Each deck's own choices stay its own: the search engine picked in its
  Profile → Web searches, and whether it reads pages here (Profile → Web
  pages). The switches in this panel's Settings are for this browser, on every
  deck.
- Every connected deck's own pages are off-limits to agents — from any deck.
- **Disconnect** (in the list, or Settings) disconnects the browser from that
  one deck; it stays connected to the others. A deck that disconnected the
  browser from its side (Connections → Browser) says so in the list — **Remove**
  it, or connect it again; the deck in use is dropped as soon as it says no.

An agent's own headless browser stays connected to just its own deck.

### Updating

The panel tells you when your Flight Deck has a newer version: **Download** it,
unzip it over the folder you loaded, then press **Reload** (or the reload arrow
on `chrome://extensions`). If the extension is newer than the deck's copy, it
says that instead — the deck needs updating.

Coming from 0.7.2 (Captain Claw 0.9.2): update and restart Flight Deck and the
agents first, then download 0.9.0 from **Connections → Browser**, unzip it over
the folder Chrome loaded and press **Reload**.

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
- **Several pages at once** (`read`, up to 6): each opens in a background tab
  of the agent's group, is waited for and read, and closes again — the
  agent's own tab stays where it was, and nothing comes to the front.
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
- **A page with another extension's frame in it** (a helper button some
  extension puts into a message box, say) — Chrome keeps Captain's debugger
  off that tab. The agent still navigates there, and with **Work on every
  site** on it also reads the page, fills fields, clicks and types (as page
  events, not real input), scrolls and takes screenshots (the tab comes to the
  front). Keys, hovering, dragging, clicks at a point and scripts need the
  debugger: the agent tells you which extension it is — set its **Site
  access** to **On click** in `chrome://extensions` and reload the page.
- **Bring an agent's tab to the front while it works** — off by default, so
  an agent working in the background doesn't take your tab away.
- **Back where you were** — when an agent's tab took the front (that switch,
  or a link it clicked opening a new tab), the tab you were on comes back when
  the agent's run is over. If you picked another tab meanwhile, it stays.

Pages that draw themselves after they load (LinkedIn, Facebook, most web
apps) are waited for: after opening a page or a click that changed it, and
before each read, the agent waits until the page stops changing and none of
its data requests is still on the way (up to about 10 s). The first
`page_text` of a page also scrolls through it a screen at a time — sections
drawn only when scrolled to come in — and back to where it was.

It uses two permissions the extension has from install: **debugger** (real
clicks, typing and screenshots in the agent's tabs — Chrome doesn't let an
extension ask for it later) and **tab groups**. Nothing uses them until you
turn the switch on.

## Your agents' web searches, in this browser

Instead of a paid search API, your agents' `web_search` can run here, on the
engine you pick in Flight Deck → **Profile → Web searches** (Google, Bing,
DuckDuckGo, Yahoo or Qwant; off until you pick one). The panel then asks once
— **Allow** — and Chrome asks you to let Captain read that engine's site. With
**Work on every site** already on, Captain may read it anyway, so searches
start without asking; Settings → Web searches turns them off in this browser.

Each search opens the result page in a background tab — never in front, in a
collapsed **Captain · search** group — and closes it once the results are in.
Flight Deck reads the results out of the page. Searches show in this
browser's history (and in your Google or Bing activity if you're signed in).
When Chrome is closed, Flight Deck's own search answers instead. Turn it off
in Settings → Web searches. It works with
**Agents can use this browser** on or off; the agents' browser tool never gets
the search tabs.

## Your agents' web pages, read in this browser

Your agents' `web_fetch` can read pages here, signed in as you — so LinkedIn,
X, Facebook and other sites that turn Flight Deck away (a sign-in wall, a bot
check, LinkedIn's 999) open for them. Turn it on in Flight Deck → **Profile →
Web pages**: *only pages that turn Flight Deck away*, or *every page*, and
optionally only for the sites you list (`linkedin.com` covers its
subdomains; list `x.com` and `twitter.com` both, since links use either). It needs **Work on every site**; without it, the panel offers
**Allow**.

Each page opens in a background tab — never in front, in a collapsed
**Captain · reading** group — and once it has finished drawing (and been
scrolled through, up to eight screens, for what loads as you scroll), it goes back to the agent without scripts and styles,
and the tab closes. Never Flight Deck's own pages, addresses on your own
network or this computer, or anything but http(s). An agent you stopped in
the panel reads nothing here. Turn it off in Settings → Web pages.

## A robot check on the way: you answer it, the agent goes on

When a search or a page lands on a site asking you to prove you're not a
robot — Google's "unusual traffic" page, Cloudflare's "Just a moment…", a
reCAPTCHA, hCaptcha or Turnstile box, DataDome, PerimeterX's "Press & hold",
Akamai, Amazon's "Enter the characters you see" … — its tab isn't closed: it
comes out of the collapsed group to the front, and Chrome comes forward. The
agent's chat says it's waiting for you, and the panel's log shows it. Answer
the check there; once it's gone, the agent's search or read goes on in that
tab, and the tab closes with the one you were on back in front.

Checks that pass by themselves (Cloudflare's "Just a moment…" usually does)
get a few seconds first. One check per site at a time: other searches on that
site wait for the one in front. The agent waits up to five minutes when you're
in its chat, two for a run that started on its own; not answered by then, it
goes on without that page (Flight Deck's own search or fetch), and the tab
stays open for you. Close the tab to say no: that site's checks aren't shown
again for ten minutes. A page that only refuses (no check to answer) isn't
shown. Agents from before this don't wait — the search goes to Flight Deck's
own engine as it used to.

## Playbooks — ready-made ways to get things done

Captain knows a few hundred everyday jobs on the web: how long from A to B,
where a parcel is, the cheapest days to fly, last year's invoices, what a
subscription costs, whether a place is open now. Each is a **playbook** — the
page to open, what to do there, what to read off it.

- **Do it here** (under the page card): the playbooks for the site you're on
  — on Amazon, "Where is my Amazon order", "Download Amazon invoices"… — and
  your own recordings made there. The same brand on another country's
  domain counts (amazon.de, amazon.co.uk).
- **The store** (⚡ at the top, or *More* beside the chips): search what you
  want done ("track a parcel"), or browse Travel, Nearby, Shopping, Accounts
  & bills, Work, Research, Money, Home & life, Media & social and Local
  services (your country's first). **＋** puts one in **My playbooks**, which
  come first in the chips and in the store, in every browser you connect.
- **A playbook**: what it does and where it runs (*any browser*, or *your
  Chrome, signed in*), a short form — places, dates, a choice, a switch — and
  **Run with** the agent you pick. The agent runs it with its `web_playbook`
  tool and you follow it here like any clip. One that books, buys, sends or
  submits says so and **stops before its last button** — you press it.
- A place is a town or an address: a link can't carry "here".
- The same store is in Captain (Flight Deck) under **Playbooks** — Run there
  sends it to an agent in its chat. Your My playbooks are the same in both.

## Record a process — an agent does it after that

**Record what I do** (under Send) starts a recording in the tab in front. Do
the thing on the site the way you always do; the panel lists every step as you
go — a page opened, a click, text typed into a field, a choice in a list, a
tick, a file picked, Enter. Each step is told the way you'd say it ("Click the
button “Delete” — in the row “September 2026”"), not as coordinates.

- **Add an instruction** to any step, while recording or after: what to pick,
  when to skip it, what to check ("only the oldest month").
- **Stop and review**: name it, say what it's for, remove or reorder steps,
  **add a step in your own words** ("wait until it says Ready, then download
  it"), and say where a value you typed or chose comes from on a run (**This
  value**): *Always the same*; *Ask me before it starts* (a field on the Run
  form); *Ask me when it gets there* (the agent stops at that step, a box
  opens in the panel and in Flight Deck, and it waits — Cancel stops the run
  there); or *The agent decides* (it picks what fits and tells you what it
  used).
- **Passwords and codes are never recorded** — the step says one was typed,
  and the agent asks you for it when it gets there.
- **Recordings → Run** hands it to the agent you pick. It works through the
  steps with its browser tool in this Chrome, signed in as you, in tabs of its
  own (turn on *Agents can use this browser* — the Run form offers it), reads
  the page as it is today, and stops to ask when a step doesn't fit what it
  sees. You follow it in the panel like any clip.

Recording follows the window: an address you type or a tab you switch to is a
step too. On a site Captain hasn't been allowed to read yet the panel says so
and offers **Allow on this site**. Not recorded: scrolling, hovering, drag and
drop, right-clicks, and what happens inside frames from sites you haven't
allowed.

**Extra data** (on the Run form): what the agent reads before it starts — how
you want it done for each client, a list, rules, an example. *Every run* files
are kept with the recording (add them from your computer here; agents' files,
Flight Deck files, pasted text, links and conversations in Flight Deck →
Connections → Browser → Recordings); *This run only* files go with one run.
A value asked before the run that came from a list on the site (a company, an
account) is offered as that list.

**Earlier runs** are listed under the Run button — when, who, with what, how
it ended — and open with the agent's answer. Flight Deck shows the same under
Connections → Browser → Recordings.

Recordings, their extra data and their runs are kept on your Flight Deck (not
in the browser), so they're there in every browser you connect.

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

A recording's run gets a folder of its own, `<day>-<site>-run-<name>/`, with
`recording.md` (the steps, with the page and a selector for each as hints) and
`extra/` (the extra data it was given).

Every agent of yours reaches the folder through `vfs:web-clips/…`; what the
agent makes for a clip (notes, CSVs, a brief) goes there too.

## Permissions

- **activeTab, scripting** — read the page you send.
- **tabs** — show the title and address of the page you're on.
- **pageCapture** — save the whole page as `.mhtml`.
- **contextMenus, sidePanel, storage** — the right-click menu, the panel, your settings.
- **Sites** are asked for one at a time (or all at once from settings): to read
  a page you didn't open Captain on, to download a file from another site, and
  to see what you do on a site while you record.
- **debugger, tabGroups** — used only while "Agents can use this browser" is
  on: your agents work in tabs of their own, grouped under their name, with
  real clicks and typing. Chrome won't let an extension ask for the debugger
  later, so it's asked for at install.
- **alarms** — keeps the agents' connection to Flight Deck alive.
- **The search engine's site** (one, when you press Allow for web searches) —
  to read its result pages in the background tab.
- **Every site** ("Work on every site") — also what lets it read pages for
  your agents' `web_fetch`, when you turn that on in Flight Deck.

The page's text is marked for the agent as web content: instructions written
in a page are never treated as yours.
