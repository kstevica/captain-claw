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
| Watch it | A daily check that tells you when it changes |
| Remember | The facts kept in the agent's memory |
| Ask | Your own question |

**Read it as** gives any action a lens: investor, buyer, legal, skeptic,
competitor, engineer, newcomer, your team — or your own.

**What goes with it**: the page as Markdown (always), the selection, the whole
page as one `.mhtml`, a screenshot, the tables as CSV, and the files the page
links to (documents are ticked for Handle it, Save and Grab files).

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

The page's text is marked for the agent as web content: instructions written
in a page are never treated as yours.
