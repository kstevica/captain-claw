# Captain for Chrome — design and next steps

A Chrome (MV3) extension that sends the page you're on to one of your agents.
Code: `browser-extension/chrome/` (the extension), `captain_claw/flight_deck/browser_clip.py`
(the deck side), `flight-deck/src/components/BrowserPairDialog.tsx` and
`components/connections/BrowserConnection.tsx` (pairing and Connections → Browser).

## Why the browser does the reading

The agent's own web tools fetch pages from the server: no cookies, no
sign-in, often a bot wall. The extension reads the page **in the person's
browser** — a portal they're signed in to, an intranet, a paywalled article, a
dashboard drawn by JavaScript — and downloads the files it links to with their
session. The agent then works on exactly what they saw.

## Flow

1. **Pairing** (device-code style, no password in the extension):
   `POST /fd/clip/pair` → a code + a secret id; the extension opens
   `<deck>/connect-browser#<code>`; the signed-in person approves; the poll
   `GET /fd/clip/pair/<id>` hands over a personal access token named
   `Browser · <browser>` once. The approval dialog shows the code, the
   requester's address and warns when it's another network than theirs.
2. **Reading** (`capture.js`, injected on demand): details, JSON-LD, the main
   text as Markdown (article/main or the densest prose block; nav, aside and
   footer left out), the selection, tables (as rows), files linked or embedded
   (by extension or `download`), large pictures, links.
3. **Files**: same-origin ones are fetched inside the page (its cookies),
   others from the panel with a site permission asked for in the Send click;
   a sign-in page in place of a file is caught; what couldn't be fetched is
   listed for the agent ("fetch the public ones yourself").
4. **Send**: `POST /fd/clip/send` (multipart) → saved whole in
   `vfs:web-clips/<day>-<site>-<title>/` (or not at all), then a chat turn
   over the agent's `/ws`: the person's own words (action + note + lens),
   `page.md` attached, the rest of the folder in attachment notes, a context
   note that the page is web content, `source: "browser"`. Not an automated
   turn: it shows in the agent's chat in Flight Deck like anything else they
   said, and counts against their daily allowance.
5. **Progress**: `GET /fd/clip/runs/<id>` — activity, the answer as it's
   written, the reply, the clip folder's files (downloadable from the panel);
   stop; follow-ups go to the same folder.

The prompts for actions and lenses live on the deck (`ACTIONS`,
`PERSPECTIVES`); the extension draws what `/fd/clip/me` returns, so a deck
update improves them for every browser.

**Watch it = the deck's Scheduled tasks.** For a Watch it clip, `/fd/clip/send`
makes the check in `fd_scheduler` itself (`_schedule_watch`) — never the
agent's cron, which the deck can't see. One task per page and agent:
its prompt starts `Watch check for <url> `, and the same page watched again
updates that task (schedule, zone, the newest clip folder). Owner = the person,
`delivery_kind: agent` with chat delivery, `prompt_author: human`, the
browser's time zone (one the deck doesn't know → the deck's time). How often:
`every 1h` · `every 6h` · `daily 09:00` · `weekly mon 09:00`. The clip's own
turn tells the agent the check exists ("don't schedule anything yourself")
and to save `watch/latest.md`, which each run compares against. The answer
carries the task (`watch`); the panel links to `<deck>/#task=<id>`, which
opens it in Scheduled tasks (`TaskDeepLink`). A task the scheduler refuses
keeps nothing: the clip folder goes and the send fails.

**The agent's own lenses.** `GET /fd/clip/agents/<slug>/lenses` answers with
the lenses the chosen agent suggested for itself: the deck reads the agent's
`/api/personality`, asks the agent's own model (its `/api/llm/complete`, no
tools or memory, thinking off) for six lenses that fit its work, and keeps
them in `<FD_DATA_DIR>/browser-lenses.json` per person and agent, with a
signature of who the agent is — a changed personality or description has it
asked again; `?fresh=1` asks anyway. The tokens count toward the person's day.
While it's being asked the answer is `state: "making"` (the panel shows the
general ones and asks again shortly); an agent that's asleep or stopped is
never woken for this (`"asleep"`); one that can't say gets `"fallback"` and
isn't asked again for ten minutes. A clip names the lens by id (`a-…`) and
label; the deck turns it into the lens's sentence in the turn.

## Agents in the person's browser (browser hands)

With **Agents can use this browser** on (panel Settings), the extension lends
the browser to the person's agents: their `browser` tool's `target="chrome"`.

  agent `browser(…, target="chrome")` (`tools/browser_chrome.py`)
  → `POST /fd/browser-hands/agent/act {action, args}` (`flight_deck/browser_hands.py`;
    the caller resolved by `require_agent_caller`, its owner = whose browser)
  → the extension's service worker over `/fd/clip/hands/ws` (`hands.js`)
  → `{"type": "result", ok, content, image?, data?}` back the same way.

- **Permissions**: `debugger` and `tabGroups` are in the manifest's `permissions`:
  Chrome refuses `debugger` as an optional permission (it drops it, and a worker
  that touched `chrome.debugger` at load failed to register — 0.2.0). The switch
  is the opt-in; nothing in the worker may assume either API exists at load.
- **The socket**: the service worker opens it while the switch is on, says
  `hello` with the browser's own token (only a `Browser · …` token, still
  unrevoked, is accepted — checked again on every act), pings every 20 s
  (keeps an MV3 worker alive), reconnects with backoff, and an alarm every
  30 s brings it back after Chrome stopped the worker. Several browsers: the
  one the person was last at (window focus) gets the act.
- **Tabs**: each agent has a tab group named after it, in the window the
  person last used; it acts only on tabs in its group (`tab_id` of another
  tab is refused); a tab a click opens joins the group and becomes current.
  New tabs open in the background unless "Bring an agent's tab to the front".
- **How it acts**: `chrome.debugger` (CDP 1.3) on the agent's tabs — real
  mouse and key events (`Input.*`), screenshots scaled to CSS pixels (so a
  coordinate in the picture is a coordinate to click), `Page.navigate`,
  `Page.captureSnapshot` (mhtml), console and network buffers. The page
  helpers (`hands-page.js`: refs, the tree, find, form_input, upload) and
  `capture.js` (page_text, page_files) run in an isolated world the page
  can't see; `javascript` runs in the page's own world. Idle tabs are
  detached after 3 minutes (Chrome's "is debugging" bar goes away).
- **Files** (`download`, `save_page`): fetched in the page (its cookies) or by
  the worker, posted to `/fd/clip/hands/files/<act id>` and kept in
  `vfs:web-clips/<day>-<site>-<agent>-browser/`; the act's answer lists them.
- **Limits**: Flight Deck's own origin, non-web URLs (`chrome://`, `file://`,
  `data:`…) are refused; the person can Stop an agent in the panel (kept in
  `handsStopped`, also told to Flight Deck so it refuses at once); Cancel on
  Chrome's debugging bar stops the agents that were working. Member turns of
  a shared agent never reach the tool (speaker allowlist), and the routes
  aren't grant-aware.
- **Tests**: `tests/test_flight_deck/test_browser_hands.py` (the relay),
  `tests/test_tools/test_browser_chrome.py` (the tool). A Chromium run with
  the real extension against the full deck passed 48 checks (form filling,
  refs, screenshots/zoom, keys, drag, upload, console/network, downloads,
  new tabs, refusals, the panel's Stop) — harness not in the repo.

### The same page actions in the agent's headless browser

The agent's own headless browser (Playwright) runs the same two page scripts —
copies shipped in `captain_claw/tools/browser_page_js/` (a test keeps them equal
to the extension's) — in an isolated world through a CDP session
(`tools/browser_page.py`). So `read_page`/`find` refs, `form_input`,
`page_text`, `page_files`, `download` (with that browser's cookies, into
`saved/downloads/<session>/<site>/`), `save_page`, `hover`, `drag`, `zoom`,
`javascript`, `upload_file`, `back`/`forward`, and `click`/`type` with a ref or
coordinate work there too. Only `tabs`, `tab_*`, `console`, `network_requests`
and `resize` stay Chrome-only (and imply `target="chrome"`).

### Back to the person's tab when the run is over

Before each act the extension notes the active tab in every window; if the act
leaves one of that agent's tabs in front where the person was on a tab of their
own ("Bring an agent's tab to the front", a link opening a new tab, a screenshot
of a tab that wouldn't draw behind), it remembers that tab (`handsReturn` in
session storage). When the agent's outermost turn ends after it used the
person's Chrome (`Agent._turn_over_browser` → `browser_chrome.turn_over`), the
agent posts `/fd/browser-hands/agent/turn-over`; Flight Deck sends `turn_over`
to the browsers that agent may use, and the extension brings the remembered tab
back — only if the agent's tab is still in front. A tab the person picks
themselves between acts forgets it: their choice stays.

### An agent's own Chrome

`browser(action="extension_connect")` pairs Captain for Chrome in the agent's
own headless browser (installing it from this Captain first). The agent asks
Flight Deck (`POST /fd/browser-hands/agent/connect`, followed like `ask_user`);
the person gets a yes/no dialog; on yes Flight Deck mints a token of its own
kind (`cc_ab_…`, its hash in `<FD_DATA_DIR>/agent-browsers.json`) — not a
personal access token: only the hands socket and that agent's act uploads take
it, so `/fd/clip/me`, clips, MCP… refuse it. The agent keeps the connection in
`.browser-extensions/captain-conn.json` and the session puts it (with "agents
can use this browser" on) into the extension's storage at every start, since
the headless profile is new each time. Routing (`usable_by`): the person's own
Chrome first, then that agent's own browser; never another agent's. Results
from it say "your own headless browser"; the extension's panel there shows a
plain notice (no clips). Connections → Browser lists it ("Scout's own browser
(on the server)") and Disconnect forgets it; removing Captain for Chrome lets
go of it too.

## Security

- The extension's origin reaches `/fd/clip/` only (origin guard + CORS);
  every call there carries a bearer token, so no page can forge one.
- The token is a normal PAT (revocable in Connections → Browser, MCP tokens,
  or the extension's Disconnect); it can drive the person's agents — the same
  reach as an MCP token.
- A clip goes only to the caller's own agents (by exact slug); follow-ups are
  confined to `web-clips`; file names are stripped of paths and separators.
- Page text reaches the model as an attached file plus a context note that it
  is never the person's instructions. The mail guard reads the person's own
  words only.

## Next steps worth building

- **Browser hands, next** — per-agent approval on first use, a per-site
  allowlist, a recorded GIF of what the agent did, and "Share this tab with
  <agent>" from the panel (today: drag it into the agent's group).
- **Site recipes** — remembered per site: on the bank portal always Grab files
  into `finance/`; on LinkedIn Extract the profile; on a job board Make tasks.
- **Watch with a diff** — Watch it keeps `page.md` versions and the agent
  reports only the changed lines (today the agent schedules its own check).
- **Many tabs at once** — send a tab group or all open tabs as one clip
  ("compare these five vendors").
- **Rich extractors** — YouTube transcripts, Google Docs/Sheets export
  links, Gmail threads, PDF viewer text, product pages (price history).
- **Team clips** — send to a shared agent or a team folder; clips listed in
  Flight Deck with who sent what.
- **Highlights back on the page** — the agent's findings drawn on the page
  (claims marked true/doubtful in place).
- **Web Store and Firefox** — publish, and a Firefox build (sidebar_action,
  `browser.*`).
