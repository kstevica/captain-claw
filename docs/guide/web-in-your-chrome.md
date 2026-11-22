# Web searches and pages in your Chrome — LinkedIn, X and sites that turn a server away

[← README](../../README.md) · [All guides](README.md)

New in 0.9.2. Your agents can read a LinkedIn profile, a post on X, or any page that turns a server away, the way you see it: in your own Chrome, signed in as you. Their web searches can run there too, on Google or another ordinary search engine, instead of the deck's paid search service. Both go through Captain for Chrome, in a background tab you never have to look at, and both stay off until you turn them on.

For LinkedIn and X, the short version is: **Web pages** on **Every page** for `linkedin.com, x.com, twitter.com`, and **Web searches** on **Google**. The steps are below, and [LinkedIn and X](#linkedin-and-x) explains why.

**On this page:** [What it does](#what-it-does) · [What you need](#what-you-need) · [Turn on web searches](#turn-on-web-searches) · [Turn on web pages](#turn-on-web-pages) · [LinkedIn and X](#linkedin-and-x) · [How pages are read](#how-pages-are-read) · [When Chrome is closed, or a site asks](#when-chrome-is-closed-or-a-site-asks) · [Search engines](#search-engines) · [Safety and privacy](#safety-and-privacy) · [Limits](#limits) · [Troubleshooting](#troubleshooting) · [For admins](#for-admins)

## What it does

Two tools every agent already has can now run in your Chrome:

- **`web_search` on a normal search engine.** Google, Bing, DuckDuckGo, Yahoo or Qwant, in a background tab, instead of the search service the deck is set up with (Brave's API by default). Google and Bing give up to 50 results a search.
- **`web_fetch` reading pages signed in as you.** A page that turns Flight Deck away opens in your Chrome the way it opens for you: LinkedIn's "999" refusal, a sign-in wall on LinkedIn, X or Facebook, a bot check. The agent reads what you would see.

The agent calls the same tools as before. When your Chrome can't help (it's closed, the setting is off, the site isn't on your list), the deck searches and reads pages itself, as it always did.

Your choice covers every agent you own. One agent can be kept out in its config; see [Safety and privacy](#safety-and-privacy).

## What you need

- **Captain for Chrome 0.7.2**, connected to your deck, in Chrome, Edge, Brave or Arc (Chromium 116+). Searches need at least 0.6.0 and page reads 0.7.0; take 0.7.2 for both. To set it up the first time, see [Captain for Chrome](devices-and-channels.md#captain-for-chrome).
- **Chrome running**, with the extension connected. No window has to be in front.
- **You, signed in** to LinkedIn, X or whichever site you want read, in that same Chrome.
- For page reads, **Work on every site without asking** on in the extension, or **Allow** pressed when the panel asks. Searches need only the search engine's own site.

You don't need **Agents can use this browser**. That switch lends your Chrome to the `browser` tool, where an agent clicks and types in tabs of its own. Searches and page reads work with it on or off, and the browser tool never gets their tabs.

**Update the extension** if yours is older:

1. In Flight Deck, open **Connections → Browser** and press **Download the extension**. The version the deck has is shown beside it.
2. Unzip it over the folder Chrome loaded the extension from.
3. Open `chrome://extensions` and press the reload arrow on Captain for Chrome, or press **Reload** on the note in the panel.

From 0.5.1 on, the panel tells you itself when the deck has a newer copy, with **Download**, **Reload** and **Later**. **Connections → Browser** shows each connected browser's extension version. The panel's own version is at the bottom of its **Settings** (the gear at the top).

## Turn on web searches

1. In Flight Deck, open **Profile** (in the kiosk, **Settings → Profile**) and stay on the **Personal** tab. Scroll to the **Web searches** card.
2. In **Search through my Chrome with**, pick **Google**, **Bing**, **DuckDuckGo**, **Yahoo** or **Qwant**. The default is **Off — Flight Deck's search service**.
3. Open Captain for Chrome's panel.
   - If **Work on every site without asking** is on, searches start at once. Your pick in Flight Deck is the opt-in.
   - If not, the panel shows the card **Run your agents' searches on <engine> here?**. Press **Allow**, and Chrome asks you once to let Captain read that engine's site (for Google, `www.google.com`) and nothing else.
   - **Not in this browser** turns searches down in that browser.
4. Back in Flight Deck, the card says "Searching on <engine> in <browser>". Until then it says what's missing: Captain for Chrome not connected to the deck yet, no Chrome of yours open, or **Allow** still to press.

To stop searches in one browser, turn off **Run my agents' web searches here, on <engine>** in the panel's **Settings → Web searches**. To stop them everywhere, set the card back to **Off — Flight Deck's search service**.

Searches show in that browser's history, and in your Google or Bing activity when you're signed in there.

## Turn on web pages

1. In **Profile → Personal**, scroll to the **Web pages** card.
2. In **Read pages through my Chrome**, pick a mode. The default is **Off — Flight Deck reads pages itself**.
   - **Only pages that turn Flight Deck away**: the deck reads the page first, as always. Only when it's turned away is the page read again in your Chrome. Turned away means HTTP 401, 403, 429 or LinkedIn's 999, a dropped connection, a bot check, or a sign-in wall.
   - **Every page**: your Chrome reads first. The deck reads only when your Chrome can't.
3. In **Only these sites**, type the sites, separated by commas or spaces, and press Enter or **Save**. A site covers its subdomains: `linkedin.com` covers `www.linkedin.com` too. Empty means any site. Up to 50.
4. Read the note under it. Your agents see what you see on those sites, as you, and a page an agent reads can try to send it to others. List only the sites you need.
5. In Captain for Chrome, turn on **Work on every site without asking** (panel **Settings**, the **Every time** card). Or press **Allow** on the card **Let your agents read web pages here?** that the panel shows. Chrome asks you once, for every site. **Not in this browser** turns page reads down in that browser.
6. The card says "Reading pages in <browser>" when it's ready.

To stop page reads in one browser, turn off **Read web pages for my agents here** in the panel's **Settings → Web pages**. To stop them everywhere, set the card back to **Off — Flight Deck reads pages itself**.

**Which combination?**

| Mode | Only these sites | What your Chrome reads |
|---|---|---|
| Only pages that turn Flight Deck away | empty | Any page the deck was turned away from |
| Only pages that turn Flight Deck away | a list | Pages on those sites, after the deck was turned away |
| Every page | a list | Every page on those sites; the deck reads everything else |
| Every page | empty | Every page any of your agents reads. Not recommended |

## LinkedIn and X

The setup we recommend:

1. **Web pages** → **Every page**.
2. **Only these sites** → `linkedin.com, x.com, twitter.com`. Add `facebook.com` if you want it too.
3. **Web searches** → **Google**.
4. In that Chrome, be signed in to LinkedIn and X.

Those sites are then always read as you, and every other page is read by the deck as before.

**Why Every page.** LinkedIn answers the deck with its "999" or a sign-in page, which the deck recognises, so **Only pages that turn Flight Deck away** works there. **Every page** skips the wasted first try, and a thin public version of a page never stands in for the real one. X doesn't always answer the deck with a sign-in page. When it sends something that is neither a wall nor the post, the deck can't tell it was turned away, and your Chrome is never asked. **Every page** takes that guess away.

**Why `twitter.com` too.** The list is matched against the address the agent was given. `x.com` covers its own subdomains, not `twitter.com`, and many links (older posts, articles, embeds) still point at `twitter.com`. List both.

Things to ask, once it's on:

- **A profile, in full.** "Read `https://www.linkedin.com/in/<profile>/` and summarize their experience and education." The read waits for the sections LinkedIn draws after the page loads, and scrolls through them. A profile page lists only some roles; for all of them, add: "Also read `/in/<profile>/details/experience/` and `/in/<profile>/details/education/`."
- **Prospecting from Google.** "Search Google for `site:linkedin.com/in "head of finance" fintech <city>` and list 20 profiles with name, title and link. Then read the top five, one at a time." The search runs on Google in your Chrome (20 results is two result pages). Each profile is then a `web_fetch` of its own. "One at a time" keeps the agent from reaching for `web_fetch_batch`, which doesn't use your Chrome.
- **An X post and its replies.** "Read `https://x.com/<handle>/status/<id>` and its replies, and summarize the argument and the main objections." The read covers about eight screens of the thread; see [Limits](#limits) for longer ones.
- **A daily check.** "Every weekday at 8:00, check `https://x.com/<handle>` and tell me what they posted since yesterday. If you only get a sign-in page, say so instead of guessing." The same works for a company page, `https://www.linkedin.com/company/<company>/posts/`. Scheduled runs use your Chrome whenever it's open. When it's closed, the deck reads the page and usually gets the sign-in wall, hence the last sentence.
- **From an article to its author.** "Find the LinkedIn profile of the person who wrote `<article link>` and summarize their background." The deck reads the article (its site isn't on your list), the agent searches Google with `site:linkedin.com/in` and the author's name, and reads the profile in your Chrome.

## How pages are read

- A page opens in a **background tab** that never comes to the front, muted, in a collapsed tab group: **Captain · search** for search result pages, **Captain · reading** for pages. The tab closes when it's done. If Chrome has no window open, it opens a minimized window of its own and closes it afterwards.
- Captain waits until the page has **finished drawing**, not just loading. That means nothing on the page changes and no data arrives for about a second, nothing is marked busy, and at least 1.5 s have passed. After 10 s it reads whatever is there.
- It then **scrolls down through up to eight screens**, waiting at each for what loads as you scroll (a profile's Experience and Education, a feed, replies), lets the page settle again, and scrolls back to where it was.
- The page's HTML goes back to the deck without scripts, styles, form fields or frames. Links and pictures keep their full addresses. `web_fetch` reads it like any page it fetched: the same text (headings, lists and tables as Markdown), the same saved pictures, the same size limits.
- The agent's result starts with `[Read in the person's own Chrome (<browser>), signed in as them]` and says `[Mode: chrome]`. If the page ended up at another address, a `[Page: …]` line gives it.
- A search result page is read the same way, minus the scrolling: Captain waits for the result list (or a robot check) and sends the page back without scripts, styles or pictures. The deck picks the results out of it, so when an engine changes its page, a Flight Deck update fixes it, not a new extension.
- A page gets about 40 seconds in your Chrome. Each browser runs at most two searches and two page reads at a time.

## When Chrome is closed, or a site asks

- **Chrome closed or asleep, or the setting off.** Searches go to the deck's search service, and the deck reads pages, as before. The agent isn't told, and nothing is missing from its result.
- **A robot check or a cookie question on the search engine.** The deck's search service answers that search, and the result carries a note, for example: "Google asked to confirm a person is searching (a robot check). Open Google in that Chrome and answer it; searches go on after that." Answer it once in that Chrome, and later searches run there again. Your agents don't answer robot checks or cookie questions for you.
- **A page your Chrome won't read.** A site not on your list, an address that isn't http or https, a [refused address](#safety-and-privacy), an agent you stopped in the panel, or a page that didn't settle in time. `web_fetch` carries on as if your Chrome weren't there.
- **Just turned it on?** An agent remembers "off" for a minute, so its first searches and reads after you turn it on may still go to the deck.

## Search engines

| Engine | Result pages read | Results per search |
|---|---|---|
| Google | `www.google.com`, up to 5 pages | up to 50 |
| Bing | `www.bing.com`, up to 5 pages | up to 50 |
| Yahoo | `search.yahoo.com`, 7 a page, up to 5 pages | up to 35 |
| DuckDuckGo | `html.duckduckgo.com`, 1 page | up to 10 |
| Qwant | `www.qwant.com`, 1 page | up to 10 |

An agent gets 10 results unless it asks for more. The language, country, how recent, and safe search it asks for go to the engine where the engine has a matching setting. Google's redirect links are turned into the real addresses.

Three engines are left out on purpose:

- **Brave Search** hides each result's address until a real click, so its result page can't be read. (The deck's Brave API, the usual fallback, is a different thing and works as before.)
- **Startpage** and **Ecosia** answered automated checks with bot walls, so their pages couldn't be read or checked.

## Safety and privacy

- **Your agents see what you see.** On the sites they read through your Chrome, they act as you: anything your account can see, they can read. A page they read can try to send it somewhere else. List only the sites you need.
- **The sites see you.** A page read as you is a page you visited. A LinkedIn profile your agent reads can show you as a viewer, depending on your LinkedIn privacy settings, and sites limit how much one account may read. Ask for what you would open yourself.
- **Your history.** Searches show in that browser's history, and in your Google or Bing activity when you're signed in there.
- **Only your own agents' turns.** A member's turn on an agent you share never uses your Chrome: the tools don't ask, and the deck refuses member turns anyway.
- **Scheduled and automatic runs** use your Chrome whenever it's open. Nothing checks that you're at the computer.
- **Refused addresses** are never opened in your Chrome. The deck checks, the extension checks again, and so does the address a page ended up at:
  - anything but http and https;
  - IP addresses that aren't public (private, this machine, link-local and the like);
  - `localhost` and names with no dot in them;
  - names ending in `.local`, `.localhost`, `.internal`, `.lan`, `.home`, `.home.arpa`, `.intranet` or `.corp`;
  - Flight Deck's own pages.
- **Search tabs open only on the engine you picked**, on its own site.
- **Keep one agent out** in its `config.yaml`, then restart it:

  ```yaml
  tools:
    web_search:
      via_chrome: false   # its searches never run in your Chrome
    web_fetch:
      via_chrome: false   # its pages are never read in your Chrome
  ```

- **Stop one agent.** With **Agents can use this browser** on, the panel lists the agents at work in your Chrome, page reads included, each with **Stop**. A stopped agent reads no pages there, and can't use the browser tool, until you press **Let it again**. Stop doesn't cover searches; turn those off in **Settings → Web searches**.

## Limits

- **`web_fetch_batch` and `web_get` don't use your Chrome.** They read on the deck. Ask for pages one at a time.
- **`web_search` with `enrich=true`** reads the passages from the top results on the deck, even when the search itself ran in your Chrome.
- **Long threads and feeds may come back partial.** A read covers about eight screens. For more, ask the agent to open the page in your Chrome with the browser and scroll. That's the `browser` tool with `target="chrome"`, and it needs **Agents can use this browser**.
- **Sections drawn only while they're on screen** may not draw in a background tab, so a read can miss them.
- **Robot checks and cookie questions** are yours to answer, once, in that Chrome.
- **About 3 MB of HTML** per page, and about 40 seconds.
- **Tested on stand-in pages.** The waits and the sign-in wall checks were tested on stand-in pages, not against live LinkedIn or X accounts, and both sites change their pages. If a read comes back with less than you see, tell the agent what's missing, or have it use the browser.

## Troubleshooting

**Searches still go to the deck's search service.**

1. Look at the **Web searches** card in **Profile**: its status line says what's missing.
2. Check that an engine is picked on the card, not **Off — Flight Deck's search service**.
3. Check the extension's version: 0.6.0 or later, 0.7.2 recommended. It's at the bottom of the panel's **Settings**, and on the browser's row in **Connections → Browser**.
4. In the panel's **Settings → Web searches**, **Run my agents' web searches here, on <engine>** must be on. **Not in this browser** turns it off; turn it on again there.
5. Chrome has to be running, with the extension connected to the deck.
6. Just turned it on? Give it a minute. Was the deck just updated? The agent needs a restart too.
7. Read the result. A search that ran in your Chrome names it: "Google (in the person's Chrome)". A note that starts "Searched with …" means it fell back, and says why.
8. A member's turn on a shared agent never uses your Chrome, and neither does an agent with `tools.web_search.via_chrome: false`.

**A page still comes back as a sign-in wall.**

1. Check the **Web pages** card: the mode isn't **Off — Flight Deck reads pages itself**, and for X it's **Every page**.
2. Check **Only these sites**: the site the agent was given must be on it, or be a subdomain of an entry. `x.com` doesn't cover `twitter.com`.
3. Check that you're signed in to that site in the same Chrome the extension runs in.
4. In the panel, **Work on every site without asking** is on, and so is the **Settings → Web pages** switch.
5. Check the extension's version: 0.7.0 or later, 0.7.2 recommended.
6. Check the agent isn't stopped in the panel.
7. Read the result's first line. `[Read in the person's own Chrome …]` means your Chrome read it, and the site showed you the wall too. Without that line, the deck read it.

## For admins

- **Deploy.** Pull the update, **restart Flight Deck** (the new routes and the **Profile** cards), then **restart the agents** (the new paths in `web_search` and `web_fetch`). Restart Flight Deck first: an agent that finds a deck without the new routes doesn't ask again for four minutes.
- **The extension.** Each person updates Captain for Chrome to 0.7.2 and reloads it, as in [What you need](#what-you-need). From 0.5.1 on, the panel says when it's behind. **Connections → Browser** shows each browser's version, with a note when it's behind or ahead of the deck's copy. An extension older than 0.5.1 can't show the note itself.
- **Older extensions** keep working for clips and the browser tool. They don't offer searches or page reads, so agents fall back to the deck.
- **Per person, off by default.** Nothing changes until a person picks an engine or a page mode. There's no deck-wide switch; keep an agent out with `via_chrome: false`.
- **Keep the search service.** The fallback is still the agent's `tools.web_search.provider` (Brave by default; Tavily, You.com or Keenable). It answers whenever a person's Chrome can't, so keep its key.
- **When an engine changes its page,** the fix is a Flight Deck update, not a new extension.
