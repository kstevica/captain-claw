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

- **Browser hands** — an agent tool that asks the open side panel for more:
  "open the next 5 results", "download the invoices from this portal",
  confined to the clipped site and approved in the panel.
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
