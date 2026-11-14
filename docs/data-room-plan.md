# Data rooms — plan

A person puts a virtual data room (VDR) into Flight Deck: a ZIP of hundreds of
files, or a folder on the deck's machine. Captain anonymizes all of it into one
vault and, once the person releases it, Claude Desktop reads the anonymized copies
through an MCP connector. Claude does the work (answers, memos, models, decks)
knowing the story, the context and the numbers, but no names. When Claude has a
result, it sends it back through the same connector. Captain puts the real names
back **on the deck**, and the person reads or downloads the result there. Claude
never gets the real names back: no restored text goes out through the connector.

Status: P0 spike done (2026-11-13, Cowork on a throwaway deck; results in P0 below). P1 built (branch
`feat/data-rooms-p1`, PR #539, see "P1 as built"). P2 built (branch `feat/data-rooms-p2`, PR #540, see
"P2 as built"; not yet tried with real Claude). P3 built (branch `feat/data-rooms-p3`, PR #541, see "P3 as
built"). P4 built (branch `feat/data-rooms-p4`, PR #542). P5 built (branch `feat/data-rooms-p5`, PR #543). P6 built (branch `feat/data-rooms-p6`, PR #544). P7 built (branch `feat/data-rooms-p7`, see "P7 as built"). Builds on the anonymizer
(`docs/anonymization-plan.md`, all merged) and the inbound MCP server
(`flight_deck/mcp_server_routes.py`, `flight_deck/mcp_oauth_routes.py`).

## Decisions (the user, 2026-11-13)

1. **Client: Claude Desktop first.** It connects as a remote custom connector
   (OAuth sign-in, already built for the fleet connector).
2. **Claude gets the content both ways:** MCP tools that read and search the
   room, and a ZIP the person downloads and adds to a Claude Project.
3. **Input:** a ZIP upload of up to 25 MB, or a folder on the machine where
   Captain runs.
4. **Results back:** text, docx, xlsx and pptx. Captain doesn't make PDFs.
   PDFs in the room are read.
5. **Languages:** Croatian and English, in the room and in the results.
6. **Who reads restored results:** only the person who connected Claude, and only
   on this Flight Deck.
7. **Scans and pictures:** scanned pages are read with OCR into anonymized text.
   Logos are removed: every picture inside an Office file is replaced (below).

## How it fits together

```
ZIP / folder ──► intake ──► anonymize job ──► final sweep ──► review ──► RELEASE
                (limits,     (one room vault,   (vault's final     (person checks,
                 names)       OCR, pictures)     values, leak       roles, seed terms)
                                                 check)
                                                                          │
              ┌───────────────────────────────────────────────────────────┤
              ▼                                                           ▼
   room connector (MCP): list, read, search               "Download for Claude" ZIP
              │                                            (Claude Project)
              ▼                                                           │
        Claude Desktop  ◄─────────────────────────────────────────────────┘
              │  room_deliver (text, or a docx / xlsx / pptx spec, or a small file)
              ▼
   restore on the deck ──► Inbox (encrypted, owner only) ──► person reads / downloads
              │
              └──► Claude gets a receipt: "Delivered as #12", counts, tokens it got wrong
```

## Principles

1. **Nothing restored goes out through the connector.** Every answer the room
   connector gives is built from released anonymized copies, or is a receipt.
   That covers tool results, error messages and reports too. A report names
   tokens (`⟦ORG_99⟧ isn't in this room`), never a real value.
2. **Nothing reaches Claude before release.** The person releases the room after
   review. A file that fails a check is never released. A file added later waits
   for its own release.
3. **One room, one vault.** Every file in the room, and every file added later,
   goes through the same reversible vault, so a company has the same token
   everywhere. Seed terms and roles (below) live with the vault.
4. **Restored results stay on the deck, with their owner.** The Inbox lives
   outside the VFS, so agents can't read it (they may run on cloud models). It is
   encrypted with the room's vault key, readable by the owner only, never through
   admin act-as (`X-FD-Act-As`), with no public `/s/` links and no sharing.
   Push notifications that leave the deck (Telegram, WhatsApp, mail, Line
   channels) carry generic text only.
5. **Fail closed.** If a file can't be checked, it isn't released. Tokens Claude
   made up stay as tokens in the restored result and are flagged. Anything the
   deck can't read is listed as "not read", never passed through.
6. **The room connector is separate from the fleet connector.** It has its own
   endpoint and tool list, so a data-room chat in Claude can't reach agents.

## What we reuse

| Piece | Where |
|---|---|
| Anonymize one file (text, md, csv, html, pdf→md, docx, xlsx, pptx) | `anonymize_routes.anonymize_blob` |
| Vaults (Fernet key per vault, sealed names, token history) | `flight_deck/anonymize_vault.py` |
| Restoring text and bare tokens | `anonymize/engine.py` `restore` |
| Restoring inside Office files | `engine.restore_edits` + `office.py` (`/file/restore`) |
| Folder job (two phases, names of files and folders) | `anonymize_targets.py` `_run_folder` |
| Trusted model lane, NER name pack | `anonymize_trusted.py`, `anonymize_ner.py` |
| Local OCR (Tesseract / RapidOCR, no model) | `captain_claw/ocr/` `run_pdf`, `run_image`, `text_layer_verdict` |
| Building docx / xlsx / pptx from markdown or a spec | `docx_ops.build_document`, `xlsx_ops.build_workbook`, `pptx_ops.build_deck` |
| MCP server (JSON-RPC, PAT + OAuth) | `mcp_server_routes.py`, `mcp_oauth_routes.py`, `auth.get_mcp_user` |
| Linking a folder on the deck's machine (admin-only guard) | `vfs_routes.py` `_require_link_admin` |
| Full-text search | SQLite FTS5 (available in our Python) |

## Gaps found in what exists (closed in P1)

- **Order of files.** The folder job anonymizes files one by one. A name the
  trusted model first finds in file 300 stays raw in file 12, because file 12 was
  done before the vault knew it. There is no sweep afterwards.
- **Pictures and embedded files are left as they are.** `office.py` only warns
  about them. A chart's embedded data workbook in a pptx carries the raw labels.
- **Not handled:** picture and shape alt text (`wp:docPr` `descr`/`title`),
  `docProps/thumbnail.*` (a picture of the first page), `customXml` parts.
- **Scanned PDFs fail.** `_pdf_text` returns 422 when a PDF has no text layer.
- **Image files** (`.png .jpg .tif …`) aren't in `FILE_KINDS`.
- **MCP tokens aren't tied to an endpoint** (no `aud`), and `initialize`
  returns no server `instructions`.

## Phases

Each phase is its own PR. Build order: P0 → P1 → P2 → P3 → P6 → P4 → P5 → P7.

### P0 — Spike: what Claude Desktop actually does (half a day)

Claude Desktop's custom connectors connect from Anthropic's cloud, so the deck
must have a public HTTPS address. The staging deck has one. A deck on a laptop
doesn't, so a throwaway deck needs a quick tunnel (ask the user first). Using a
stub endpoint with made-up data, measure:

1. Adding the connector in Desktop through the existing OAuth flow.
2. The largest tool result Desktop shows uncut → page size for `room_read`.
3. The largest delivery Claude writes into one tool call (a long memo, an xlsx
   spec with ~2,000 cells) → size limits, and whether long deliveries need parts.
4. Whether Claude's sandbox can reach the deck host once the person allows that
   domain in Claude's settings. If it can, P5's upload and download links work.
   If not, P5 is base64 only.
5. Whether Desktop follows the server `instructions` from `initialize`.

Write the numbers into this file; P2–P5 use them.

**Found so far (2026-11-13, throwaway deck + tunnel, stub on branch `spike/data-rooms-p0`):**

- **Cloudflare blocks Claude.** Claude's MCP calls after sign-in carry the user
  agent `Claude-User`; its discovery, registration and token calls carry
  `python-httpx`. A Cloudflare quick tunnel answers `Claude-User` with
  `403 "Your request was blocked."` at Cloudflare's edge, so Claude signs in and
  then reports that the server "refused Claude's request" — the deck never sees
  the call. **A deck behind Cloudflare must let `Claude-User` through** (turn off
  AI-bot blocking for the zone, or a WAF skip rule for `/fd/mcp-*`, `/token`,
  `/register`, `/authorize`, `/.well-known/*`). Staging (Caddy) lets it through.
  localhost.run rotated its address twice in 3 minutes (unusable); pinggy works
  (60-minute free tunnel, one "Enter site" click in the browser).
- **Fixed:** `GET`/`DELETE /fd/mcp-server` answered 404; Streamable HTTP wants
  405 (a 404 reads as "session gone"). Now 405 `Allow: POST`.
- **Claude's connector flow, as seen on the deck:** `POST /fd/mcp-server` → 401
  → `/.well-known/oauth-protected-resource` (the root path, not the
  path-specific one) → `/.well-known/oauth-authorization-server` → `POST
  /register` (twice) → `/authorize` in the person's browser → `POST /token`.
  Requests come from 160.79.106.0/24. The root discovery path answers for
  `/fd/mcp-server` only, so P2's second endpoint needs its own resource metadata
  and a `resource_metadata` pointer in its 401.
- **Measured in Cowork** (Claude Desktop), not plain Chat; plain Chat's read
  limit and sandbox may differ — check once before P2 ships.
- **Clients and protocol.** On connect, two clients initialize ("Anthropic/Toolbox",
  "Anthropic/ClaudeAI"); Cowork connects as "claude-code" (2.1.324-ccs)
  when it first uses the connector. All are stateless (no `Mcp-Session-Id`),
  `Accept: application/json, text/event-stream`. Each first tries
  `server/discover` with protocol `2026-07-28`, gets "method not found" from us,
  falls back to `initialize` (asks for `2025-11-25`) and accepts our `2025-03-26`.
  P2: support the newer protocol version and `server/discover`.
- **Server instructions are followed.** All 6 deliveries started their title with
  "R: " as the `instructions` asked; 380 tokens written, all with brackets, none
  bare, none made up.
- **Reading.** Claude asked for 20k, 50k, 100k, 200k, 400k and 800k characters;
  the deck sent all of them. Up to **100k characters the whole result reaches
  the model**. From 200k up, **none of it does**: the harness saves the result
  to a file in Claude's workspace (complete every time) and Claude has to read
  that file. The limit is between 100k and 200k. → `room_read` pages of about
  **50k characters** (whole, inline, room left in context); `room_search` stays
  small. A deliberately large result is also a way to put a file into Claude's
  workspace without any network (P5).
- **Delivering.** A ~3,000-word memo came in one call: 23,821 characters (3,752
  words). Asked for 12,000 words, Claude split on its own into 4 parts of
  16–22k characters (11,858 words, ~74k characters in all). Largest single call
  seen: ~24k characters. → `room_deliver` takes `part`/`of`, joins on the deck,
  and the tool description says "up to ~20,000 characters per call".
- **xlsx spec.** 9,665 characters of valid JSON in one call: 2 sheets, 34 rows
  each, 161 formulas, freeze panes and column widths — in a shape Claude made up
  (the stub gave no schema). P4 gives the `build_workbook` schema in the tool.
- **Sandbox.** Claude built a real `.xlsx` in its workspace, but its network
  proxy refused the deck's host for both the download and the upload
  (`CONNECT tunnel failed, response 403`, logged as a policy denial). → Upload
  and download links work only when the person puts the deck's host on Claude's
  workspace allowlist. P5 keeps them as the option for that case; the default is
  base64 in the call (small files) and Claude handing the file to the person,
  who drops it on the deck.
- **Claude's own file had no tokens.** The real `falcon_model.xlsx` Cowork built
  says "Project Falcon — target operating model", while its JSON spec of the
  same model said "⟦ORG_7⟧ (target) — operating model". A file Claude builds may
  carry the codename instead of tokens, and restoring it changes nothing. → The
  connector instructions say "keep tokens in every file you make, including
  titles, sheet names and notes", and the delivery report says when a file has
  no tokens at all.
- **Restoring inside Claude's workbook works** with today's office restore
  (`office.prepare` + `engine.restore_edits`), on a copy with tokens added: 73
  text pieces, 7 tokens restored (titles, a note, a bare token, a sheet name),
  all 161 formulas kept and pointing at the renamed sheet, a restored sheet name
  over Excel's 31 characters cut to 31 with formulas following, and an unknown
  `⟦ORG_99⟧` left in place and reported.

### P1 — Room, intake, anonymize, review, release

**Model.** A `rooms` table (new `rooms.db` in `FD_DATA_DIR`) holds: id, owner,
**codename** (what Claude sees, e.g. "Project Falcon"; the room's real name is
sealed and never shown to Claude), vault id, state (`draft` → `running` →
`review` → `released`), settings (lane, OCR languages `hrv+eng`, strip pictures
on, coarser numbers off), and seed terms and roles, both sealed with the vault key.

**Storage.** Everything for a room lives under `FD_DATA_DIR/rooms/<id>/`:
`raw/` holds the extracted ZIP, encrypted at rest with the vault key; a linked
folder is read in place and not copied. `clean/` holds the anonymized copies,
`text/` the text Claude reads (plus the FTS5 index), and `inbox/` the restored
deliveries (encrypted). None of it is in the VFS.

**Intake, ZIP (≤ 25 MB):**
- At most 3,000 entries, 500 MB unpacked, 100 MB per file, and an unpacked-to-packed
  ratio cap (stops ZIP bombs).
- No absolute paths, no `..` (stops zip slip), no symlinks.
- `__MACOSX/`, `.DS_Store` and `Thumbs.db` are dropped.
- Names: the UTF-8 flag if set, otherwise try UTF-8, then cp852/cp1250, then
  cp437. Croatian names in Windows ZIPs often come in cp852.
- A ZIP inside the ZIP is unpacked one level. A password-protected ZIP is refused
  with a clear message.
- Kinds we can't read (`.doc .xls .ppt .msg .rtf .odt` …) are listed as "not
  read", never passed through. LibreOffice conversion comes later.

**Intake, folder:** an admin may pick any folder on the deck's machine, behind
the same guard as VFS links. Everyone may pick a VFS folder they can read,
including linked ones. The folder is read, never written. File hashes are kept so
the room can show "changed since the run".

**Seed sheet (before the run).** The person enters the target's names, aliases
and codename, key people, products, web domains and registry numbers. These are
registered in the vault first (extra terms, PERSON/ORG/ID), which is the biggest
help with recall. They stay sealed.

**Run.** A `RoomJob` generalizes `FolderJob`. Progress is kept in `rooms.db`, so a
deck restart resumes the run (with the trusted model, hundreds of files take
hours). The page shows an estimate. Lane is `both`.
- **PDF:** each page with a text layer → markdown. A page that is mostly a picture
  (`text_layer_verdict`) → local OCR. Then the whole document is anonymized.
- **Image files** (`.png .jpg .jpeg .tif .tiff .bmp .webp`) → OCR → `.md` → anonymized.
- **Office pictures** (`office.py`):
  - every `/media/` part (also in headers, footers, masters and layouts; SVG,
    EMF and WMF included) is replaced with a grey placeholder of the same size
    that says "picture removed";
  - text in the picture is read with OCR, anonymized, and kept as alt text and
    as `[Picture: …]` in the text view, so a table pasted as a picture isn't lost;
  - existing alt text and titles are anonymized; `docProps/thumbnail.*` is removed;
  - embedded OOXML (chart data workbooks) is anonymized recursively; other
    embedded objects (OLE `.bin`) are removed and their preview replaced;
  - text in `customXml` is anonymized.
- **Names of files and folders** go through the vault, as in the folder job today.

**Final sweep.** After the last file, run code only (no model, fast): the
known-value finder, with the final vault, over every clean copy and text view.
Files with hits are anonymized again; repeat up to 3 times. Then the leak check:
search every output for every real value in the vault, diacritic- and
case-tolerant, slug spellings included. Any hit blocks that file.

**Text view (what Claude reads):**
- docx → markdown, tables kept;
- pptx → slide by slide, notes included;
- xlsx → one section per sheet: a header row and the rows as a markdown table,
  formulas shown where present, read in pages by row range;
- PDF and scans → the markdown above.

Every text view is indexed in FTS5. That is safe because the text is anonymized.

**Review page** (owner only):
- counts per kind, and the entity list (real ↔ token) from the vault sheet;
- files grouped as OK / with warnings / blocked / not read;
- a side-by-side spot check of the original and the anonymized copy;
- seed terms can be edited; a change re-runs only the files it touches;
- a **roles editor**: token → role (`⟦ORG_7⟧` = Target, `⟦ORG_3⟧` = Buyer,
  `⟦P_2⟧` = CFO of the target). Roles tell Claude what each party is without
  naming it. The editor warns if a role contains a real value from the vault;
- **Release** releases the OK files and those with warnings. Blocked files are
  never released.

UI goes in `flight-deck/src/components/rooms/` with a kiosk tile and Settings
menu entry, next to Anonymize; in the full layout under Files → Data rooms.

**Tests:** zip slip, bombs, cp852/cp437 names, encrypted ZIPs, the order sweep
(a name found late disappears from early files), pictures replaced, chart data
anonymized, thumbnail gone, alt text, OCR (where an engine is installed), leak
check blocks a file, resume after restart. Only made-up names in fixtures.

**P1 as built (2026-11-13):**

- **Anonymizer (everyone, not only rooms).** `office.prepare(pictures="keep"|"strip",
  picture_reader=…)`. With `strip`, every picture becomes a grey "picture removed" box of the
  same proportions, in its own format where Pillow writes it (EMF/WMF and the like become PNG,
  with links and content types following). The text OCR reads in it goes into the picture's alt
  text, which is then anonymized. Whenever a file is anonymized (`keep` too): alt text and titles
  are anonymized, `customXml` text is anonymized, the thumbnail is removed, and embedded Office
  files are anonymized inside their own zip (up to two levels deep). Restoring covers them too.
  Embedded objects that aren't Office files are emptied with `strip`, else reported.
  `anonymize/scans.py`: local OCR only, nothing cached on disk. Scanned PDF pages are read with
  OCR, the rest keep their text layer. Picture files (`.png .jpg .tif` …) are now a file kind
  (they come out as `.md`). The VFS folder run still skips pictures. Option
  `one_word_names: false` exists for the sweep.
- **Server code is flat**, like the rest of FD: `rooms_store.py`, `rooms_intake.py`,
  `rooms_job.py`, `rooms_routes.py`; routes under `/fd/rooms`.
- **Roles are refused, not warned:** a role naming a real vault value gets 422 and nothing is
  saved, because Claude will see roles.
- **Run** is enabled only when there are new or blocked files. Every run redoes folder names,
  the sweep, the text and the leak check over all files. A released file the sweep changes
  goes back out of release and gets a warning.
- **Text view:** docx/pptx/xlsx through `document_extract`, with picture alt text added under
  "Pictures". PDFs and scans by page. Search is FTS5 over the text views.
- **Leak check:** real values of 4+ characters (case, accent and spacing tolerant, whole words),
  in the text and the anonymized path. The report names tokens only.
- **Fixed on the way:** the frontend `TOKEN_RE` didn't draw tokens with a numeric spelling place
  (`⟦PERSON_1:2⟧`, since #535) as chips.
- **Not done in P1:** files of a linked folder are hashed at intake, but a run doesn't yet
  compare the hash ("changed since"). OCR text of a picture lands only in alt text, not at the
  picture's place in the text. The ZIP drop zone wasn't clicked through in the browser check
  (upload tested by route tests and the same `PUT` from a script).

### P2 — Room connector, read side

- **Endpoint:** `POST /fd/mcp-rooms`, with its own entry in the protected-resource
  metadata. In Claude Desktop it is its own connector ("Captain Data Rooms"). Its
  tool list has only room tools, none of `list_agents`, `send_task` or `jev_ask`.
- **OAuth:** reuse `mcp_oauth_routes.py` and add resource binding: tokens carry
  `aud` = the endpoint, so a fleet token isn't accepted here and a room token
  isn't accepted there. The consent screen lists the person's released rooms with
  checkboxes, kept as a server-side grant (client → rooms) that can be changed or
  revoked under Connected apps. A grant can name only the person's own rooms.
  Act-as is refused.
- **Server `instructions`:** tokens like `⟦ORG_7⟧` stand for real names; keep them
  exactly, brackets included; don't guess who they are; the roles are in
  `room_overview`; to give the person a result, call `room_deliver`; you get a
  receipt back, never the restored text.
- **Tools:**
  - `room_list()` → the granted rooms: id, codename, file count, release date.
  - `room_overview(room)` → roles, folder tree (anonymized names), counts per
    kind, languages, how to cite a file.
  - `room_list_files(room, folder?)` → path, kind, size, pages or sheets.
  - `room_read(room, path, page?)` → the text view in pages (page size from P0);
    xlsx by `sheet` and row range.
  - `room_search(room, query, limit?)` → FTS5 snippets with path and place.
- **Audit:** every call is logged (room, tool, path, bytes) and goes into the
  token history as "out → Claude (connector)". The room page shows "What Claude
  read".
- **Leak alarm:** tool arguments (search queries, paths) are scanned for real
  vault values. A hit means a real value reached Claude some other way (for
  example, the person typed it in the chat). The call still answers (the value is
  already out), and the room shows an alert.

**P2 as built (2026-11-13, branch `feat/data-rooms-p2`, stacked on P1):**

- **Tokens are bound by scope, not `aud`:** a sign-in for `/fd/mcp-rooms` (the
  client's `resource`, or `scope=rooms`) gets scope `rooms`. A JWT `aud` would have
  broken today's token decoding, and the scope was already stored with every code
  and refresh token. The fleet's `get_mcp_user`, which also guards firmware, browser
  clips and browser hands, refuses `rooms` tokens. `rooms_mcp.caller` accepts only
  those, so a browser JWT, a PAT or a fleet token is refused there. The room
  endpoint has its own protected-resource metadata (`scopes_supported: ["rooms"]`),
  and its 401 points there through `FD_PUBLIC_URL`.
- **Consent in two steps:** email and password, then "Which data rooms?" with
  checkboxes and "All my data rooms, and new ones". The grant
  (`rooms.db mcp_grants`) is checked on every call, so changing it or
  disconnecting on the Data rooms page takes effect at once, before the 1-hour
  token expires. Revoking the app under Connected apps drops the grant too.
- **Tools:** `room_list`, `room_overview` (roles, folders, kinds, characters),
  `room_list_files`, `room_read` (50,000-character pages ending on a line break,
  with a header line "File · page n of N · characters"), `room_search` (FTS5,
  every word must appear; each hit gives the `room_read` page). Tools are marked
  read-only. Only released files that are clean (`done`) are served. There are
  no xlsx sheet/row arguments: a sheet's markdown pages like any text.
- **Protocol:** the client's version is answered when known (2025-11-25,
  2025-06-18, 2025-03-26), else 2025-03-26. `server/discover` → method not found
  (clients fall back). GET/DELETE → 405, on the fleet endpoint too (the P0 fix).
- **Log and history:** `room_reads` keeps tool, file, page and sizes, never text
  or queries, up to 2,000 per room. Text sent goes into the tokens' history as
  "out" to "Claude (data-room connector)". Alarms keep tokens only, with a seen
  flag.
- **UI:** a "Claude's access" card (connector address with copy, each connected
  app with room checkboxes, Disconnect). Per room, an alarm notice and a "What
  Claude read" card.
- **Not checked live yet:** the browser check and the scripted client both pass,
  but a run with real Claude through a tunnel was blocked by the session's
  safety check. It needs the person's OK, or staging after deploy.

### P3 — Deliveries: text back

- `room_deliver(room, title, content, part?, of?)`: content is markdown or plain
  text. Long content comes in parts and is joined on the deck.
- Restored with the room vault (bracketed and bare tokens). The report lists:
  tokens restored; tokens not in this room (made up, or from another vault);
  real vault values Claude wrote itself (leak alarm, or Claude guessed an
  identity, which is worth knowing either way).
- Stored encrypted in the Inbox. Claude gets back:
  `{"delivered": "#12", "restored": 37, "not_in_room": ["⟦ORG_99⟧"]}`.
  Tokens only, never values.
- **Inbox UI:**
  - list per room; reading view (rendered markdown, restored) with a "Show
    tokens" toggle and the report;
  - download as `.md`, or as `.docx` built on the deck by
    `docx_ops.build_document` from the restored markdown;
  - delete;
  - a bell notice inside the deck. Line and chat channels get "Claude delivered a
    result to a data room", with no title.

**P3 as built (2026-11-13, branch `feat/data-rooms-p3`, stacked on P2):**

- **`room_deliver`** (`rooms_inbox.deliver`): a title and markdown or text, up to
  200,000 characters a call. Up to 50 parts are kept encrypted until the last
  arrives (dropped after 24 h) and joined exactly as sent. The connector's
  instructions add rule 5: about 20,000 characters a call, parts ending at a line
  break, no pictures or links. It is the one tool not marked read-only.
- **The receipt never confirms a guess.** Real vault values Claude wrote itself go
  into the owner's report and an alarm, but the receipt stays the same shape
  (`delivered`, `restored`, `characters`, `note`, plus `not_in_room` /
  `parts_joined` when they apply). Telling Claude it named a real value would
  confirm the guess.
- **Stored** in `rooms.db deliveries`: the tokens version and the restored one,
  both encrypted with the room vault's key. Deleting the vault makes them
  unreadable. Token history gets "back" from "Claude (data-room connector)".
- **Bell:** "Claude delivered a result" / "Delivery #n is in the data room
  <codename>…", with no title. The bell feeds the Line, and the Line the owner's
  channels.
- **Restored text is inert:** shown with no pictures and no live links (a link
  shows its text, with the address in a tooltip). Otherwise a picture Claude
  planted would fetch an address carrying the restored names. The browser check
  showed no request reaching the planted host. The Word download is built with
  `image_root` refusing everything, and markdown pictures become "[picture: …]"
  first: without that, `docx_ops` would embed any local file path Claude named.
- **Inbox UI:** a "From Claude" card per room (new dot, "real values" and
  "unknown" badges, how many were put back). A reading dialog switches between
  "With the real names" and "As Claude wrote it" and shows the report notices,
  Markdown and Word downloads, and delete. The room list shows "N new from
  Claude".

### P4 — Deliveries: files built on the deck

Claude sends the content as a spec; the deck restores the strings first, then
builds the file, so the file is born with the real names. Numbers aren't touched.

- `room_deliver_document(room, title, markdown, look?)` → docx via `build_document`.
- `room_deliver_workbook(room, title, sheets)` → xlsx via `build_workbook`
  (cells, formulas, formats and charts, with the same spec as the agents' xlsx
  tool). Every string cell, sheet name and chart title is restored.
- `room_deliver_deck(room, title, slides)` → pptx via `build_deck`.
- Pictures by URL aren't fetched (no SSRF, no outside calls); charts are native.
  No PDF output (decided).
- The tool schemas mirror the existing agent tools, so one spec is documented
  once.

**P4 as built (2026-11-13, branch `feat/data-rooms-p4`, stacked on P3):**

- **Build from tokens, then restore inside the file.** Each file is built from
  Claude's token spec (`xlsx_ops.build_workbook`, `pptx_ops.build_deck`,
  `docx_ops.build_document`), so every build error and note Claude gets back holds
  tokens only. The real names then go in through the anonymizer's own Office
  restore (`office.prepare(scrub=False)` + `restore_edits`). That one path covers
  cells, sheet names (formulas follow, Excel's 31-character cap), chart titles,
  slide text, notes and paragraphs, with no spec-walking of my own. This changes
  the plan's "restore the strings first, then build".
- **Tools:** `room_deliver_workbook` (sheets + edits, the agents' `xlsx` schema),
  `room_deliver_deck` (slides without `image`/`images`, plus look options),
  `room_deliver_document` (markdown). Rule 6 in the connector instructions: don't
  make the file yourself, send its content.
- **Nothing reaches out of a restored file:**
  - workbook formulas that would fetch or link (`WEBSERVICE`, `IMAGE`, `RTD`,
    `DDE`, `HYPERLINK`, `FILTERXML`, `STOCKHISTORY`, web or `file:` addresses,
    other workbooks) are refused, telling Claude where;
  - pictures are stripped from slides and markdown (with a note to Claude);
  - links in slides and documents become their text (with a note), because once
    restored, a click would send the names out.
- **Storage:** the restored file and the token-built one, each encrypted with the
  room vault's key, under `rooms/<id>/inbox/` (removed on delete). The delivery
  row holds the text views of both, for the reading dialog. Download with
  `as=file`.
- **The receipt keeps its shape**: `delivered`, `file`, `restored`, `note`, plus
  `notes` / `not_in_room` when there are any. Real values Claude wrote into a file
  go to the owner's report and an alarm, as in P3.
- **UI:** an Excel / PowerPoint / Word badge in the Inbox list. The dialog says
  the file was built here, lists the build notes, shows the file's text in both
  views, and offers the file as the main download.

### P5 — Files Claude made in its own sandbox

- `room_deliver_file(room, name, base64)` for small files (cap from P0, likely
  1–2 MB). docx, xlsx and pptx are restored inside the file (`restore_edits` via
  `office.py`: text, tables, sheet names, chart labels, comments, headers);
  md, csv and txt are restored as text.
- **Only if P0 shows Claude's sandbox can reach the deck host:**
  - `room_upload_link(room, name)` → a one-time upload URL (5 minutes, single use,
    50 MB cap, tied to the room and the grant). Claude's sandbox uploads the file,
    and it is restored the same way into the Inbox.
  - `room_file_link(room, path)` → a one-time download URL for a **clean** copy,
    so Claude can analyse the anonymized xlsx with code in its sandbox.

**P5 as built (2026-11-13, branch `feat/data-rooms-p5`, stacked on P4):**

- **Why the path changed:** P0 measured about 20,000 characters per call from
  Claude, so base64 in a call only works for small files (about 15 KB). The main
  path for bigger files Claude makes is the person: Claude hands over the file,
  and the person drops it into the room's Inbox ("Add a file Claude made").
  Upload/download links were not built: P0 showed Cowork's proxy blocks the deck
  unless allowlisted.
- **`room_deliver_file(room, name, content_base64, part?, of?)`:** Word, Excel,
  PowerPoint, markdown, text, CSV or JSON, up to 3 MB decoded. Parts are joined
  in order, encrypted while waiting (P3's parts store). PDFs are refused ("ask
  for Word").
- **Owner drop:** `PUT /fd/rooms/{id}/deliveries/file?name=…` (raw body, 25 MB).
  It shows in the Inbox as "Added by you" and rings no bell.
- **Same restore everywhere (`rooms_inbox.take_file`):** Office files through the
  Office restore (as P4), the text kinds through `engine.restore`. The file name's
  tokens are restored too. Both versions are stored encrypted under `inbox/`.
  Real values Claude wrote raise the alarm, never mentioned to Claude.
- **No tokens flagged:** P0 found Claude's own workbook came back with a codename
  instead of tokens. A file where nothing could be put back is marked
  `no_tokens`: an amber badge and notice for the owner. Claude's receipt says
  "keep the tokens in files you make" (no values involved).
- **`room_get_file(room, path)`** (read-only): a released clean copy as base64,
  up to 10 MB, for Claude to open with code. Cowork saves results over ~100k
  characters to a workspace file, so big files arrive whole. Every piece of text
  in the file (chart caches, comments, properties, not only the text view) is
  leak-checked again before it goes out. A hit holds the file back ("read it with
  `room_read`") and raises an alarm "room_get_file (held back)". Logged as a read,
  with history "out".
- **Instructions rule 7:** `room_get_file` for working with code; keep tokens in
  files you make; `room_deliver_file`, or hand the file to the person.
- **UI:** "Add a file Claude made" on the From Claude card; labels by type
  (Excel, Word, Markdown, CSV …); the "no tokens" badge and notice; the file as
  the main download.

### P6 — "Download for Claude" (Claude Project)

The room page builds a ZIP for the person to download on their own machine:
- the released clean copies (Office files as files; PDFs and scans as their `.md`
  text views);
- `ROLES.md` (token → role);
- `README-for-Claude.md` (the connector's rules, plus "send results with the
  Captain Data Rooms connector").

A "text only" option keeps large rooms under Claude Project limits. Each export
is logged on the room. Results still come back through `room_deliver`; the tokens
are the same vault, so restoring works the same.

**P6 as built (2026-11-13, branch `feat/data-rooms-p6`, stacked on P5):**

- **`rooms_export.build`:** a ZIP under a folder named after the codename:
  - the released clean copies at their anonymized paths (Office files as files;
    PDFs and scans as the markdown they became). With "Text only", every Office
    file is its markdown text view instead (`<path>.md`);
  - `ROLES.md` (token | role);
  - `README-for-Claude.md`: the connector's own `INSTRUCTIONS`, plus how to give
    results back (the connector, or a file with tokens the person adds under
    From Claude).
- **Checked once more:** every piece of text in every file, all Office segments
  and not only the text view, is looked through for the vault's real values. A
  file with a hit is left out; the owner sees its path and the room gets an alarm
  "export (held back)". Nothing about it goes into the ZIP.
- **Two steps:** `POST /fd/rooms/{id}/export {text_only}` builds it and returns
  `{token, files, held_back, size}`. `GET /fd/rooms/{id}/export/{token}`
  downloads it once (deleted after sending; unclaimed ZIPs are pruned after an
  hour). Owner only, act-as refused. Capped at 300 MB ("try text only").
- **Logged** with the room's reads as "Download for Claude · You".
- **UI:** a "Download for Claude" card in the room (Text only, Download ZIP, a
  summary with size, the held-back list).

### P7 — Croatian grammar after restoring

A token comes back as the stored form, so "s ⟦P_3⟧" becomes "s Ivan Horvat"
instead of "s Ivanom Horvatom". For a Croatian delivery, the trusted model (on
the deck, so it may see real names) fixes only the case endings of restored
names:
- **Input:** the sentences, with restored spans marked.
- **Output:** the same sentences.
- **Guard:** the change must touch only the marked spans, and each changed span
  must keep its stem (the stem logic in `detectors`). Otherwise the uninflected
  text is kept.

It is on by default for Croatian ("Fix grammar" per delivery). Language is
detected from diacritics and common words.

**P7 as built (2026-11-13, branch `feat/data-rooms-p7`, stacked on P6):**

- **`rooms_grammar`:** only people's names (PERSON tokens whose value reads as
  a name; not a link's slug, not inside a URL or e-mail). Companies and places
  stay as written.
- **Language:** told from Claude's own text in tokens (Croatian names restored
  into an English memo don't count): common Croatian vs English words, plus
  diacritics.
- **The model gives forms, not sentences.** One item per (sentence, name): the
  name's surroundings (~160 characters before, ~100 after, cut at a line break)
  with the name marked «like this». The answer is JSON id → form, 40 items a
  call, at most 600 per delivery (the rest stay as stored). The prompt names
  which case follows which word (s → instrumental, o → locative, od →
  genitive, za → accusative, prema → dative). Without that guide a 9B local
  model got 3 of 4 right ("o Ane" instead of "o Ani"). With it, all of a
  9-name memo was right: six cases changed, three nominatives kept.
- **Guard, per word** (hyphenated parts too): the same word count and order;
  initials, numbers and punctuation never change; the same capitals; the new
  word is a stem of the old one plus a Croatian case ending. Allowed stems:
  - the word itself;
  - the word without its final vowel (Marko → Mark-a);
  - a fleeting a in -ar, -ac, -aj (Petar → Petr-om);
  - Pavao → Pavl-a;
  - Luka → Luc-i (only with -i).

  Anything else ("Ivica" for "Ivan", "Marta" for "Marko", a word added) keeps
  the name as stored. The guard makes sure the name stays the same name. It
  can't tell whether the case is the right one.
- **After the delivery, not in Claude's call:**
  - The plain restore is stored at once, exactly as before.
    `report.grammar.state` is "waiting", "done" (names, changed, refused,
    left, model), "failed" (error type only) or "unavailable" (no trusted
    model).
  - The fixed version goes into the same encrypted payload (`title_fixed`,
    `body_fixed`). A file's fixed copy is saved as
    `inbox/<seq>.fixed.<ext>.enc`, built by the same office restore with the
    case-fixed edits; Claude's own text files work too.
  - One delivery at a time per deck. A deck restart resumes the waiting ones
    (`grammar_waiting`).
  - Tokens used by a non-local model count toward the owner's usage.
- **Routes:**
  - Downloads give the fixed version; `?plain=1` gives the names as stored.
  - `POST /fd/rooms/{id}/deliveries/{seq}/grammar` runs it on request, for a
    delivery that wasn't taken for Croatian or came in before a trusted model
    was set up (409 without one).
  - Deleting a delivery removes all of its files, including Claude's own md,
    csv and json files, which P5 left behind.
- **UI:**
  - The list shows a "case endings…" pill while the model works.
  - The delivery dialog has a line saying how many names were changed and by
    which model, with a switch between "In their case" and "As stored"; the
    switch also picks which version the downloads give.
  - A "Fix case endings" button appears when the run failed, when there was no
    trusted model, or when the delivery wasn't taken for Croatian.
- **Tested** on the throwaway deck with a local 9B model (ornith:9b through
  Ollama): a 9-name memo took about 50 s, and a workbook took 20 s.

**After the first real run (2026-11-13, staging, a real term sheet):**

- **A logo's OCR text leaked a company name.** The text read from a removed
  logo kept the fund manager's name in plain text, in a variant with a word
  put in ("X Y ABC Z" for "X Y Z"). Exact matching missed it, and the leak
  check missed it the same way.
- **Fix (engine):** a known company of two words or more is also found with its
  words in order, at most one short word (≤ 6 letters) between two of them and
  two in all, punctuated or across a line break. The capital rule is the same
  as for the exact name. The text's own spelling gets its own place in the
  token, so it restores exactly. A one-word company isn't matched this loosely.
- **Fix (leak check):** `rooms_job.needles` adds the same pattern for such
  companies, so the run, the export, `room_get_file` and Claude's deliveries
  all see the variant.
- **"Check again":** a room with nothing new can be run again. It sweeps every
  file with the final vault and the names to hide, then checks it. A released
  file that changes is taken back until it is released again. Before this, a
  name added to hide after a run could only reach new files.
- **Still the person's call:** a name the vault never learned (the fund's own
  short name, here) has to be added to the room's names to hide.

**Room UI and tasks for Claude (2026-11-13, branch `feat/data-rooms-ui`):**

- **Room layout:** a left rail with two groups.
  - **Room:** Files (a folder tree, or one list), Add files, Names to hide,
    Roles (edited inline).
  - **Claude:** From Claude, Tasks for Claude, Claude's activity, Download for
    Claude.
  - The rail shows counts and alarms, plus the trusted model and OCR status at
    the bottom. The room's bar holds Run / Check again, Release, Delete,
    fullscreen and close. A thin progress strip shows while a run goes. On a
    phone the sections are a strip of chips.
- **Fullscreen:** every room dialog (the file viewer and the delivery reader)
  can fill the window, and so can the kiosk's Data rooms dialog. Escape closes
  only the top dialog. A picker that is open, or a task box with text in it,
  keeps its Escape.
- **File viewer:**
  - The original and what Claude reads, side by side, scrolling together line
    by line (in proportion when the line counts differ).
  - Search by text (both sides), by token, or by real name, the last two as a
    searchable picker grouped by kind. The token on Claude's side and every
    spelling of the real name on yours light up yellow, the current match
    brighter, with previous/next.
  - Clicking a token shows it everywhere.
- **Delivery reader:**
  - A centred document; real names or as Claude wrote it; the case-endings
    switch.
  - Search with highlights inside the rendered markdown, through a rehype
    plugin; the current match comes from React context, so a re-render
    doesn't drop it.
  - An "Ask Claude about this" side panel.
- **Tasks for Claude** (the person's choice: queued for Claude Desktop, not run
  through the API):
  - **Queuing:** the person's words are anonymized with the room's own vault
    and settings (`anonymize_blob`), and the person sees "Claude will read"
    first. A known real value still in it refuses the task. Both versions are
    stored encrypted with the vault key (`room_tasks` table).
  - **Picking up:** `room_list` shows `tasks_waiting` and a note. `room_tasks`
    lists open tasks in tokens, with the delivery each is about (up to 20,000
    characters; `room_get_delivery` pages the rest), and marks them taken.
    Every deliver tool takes `task=<n>`, which marks the task answered and
    links the delivery (`report.task`).
  - **Routes:** `GET/POST /fd/rooms/{id}/tasks`, `POST …/tasks/preview`,
    `DELETE …/tasks/{seq}` (cancel).
  - **Instructions:** rule 8 tells Claude to check tasks when asked ("Check
    my Captain data room tasks").

**Big rooms: progress, ETA, sizes, Stop (2026-11-14, branch `feat/rooms-progress-detail`):**

- **Progress:**
  - **The hook:** `anonymize_routes.PROGRESS` (a context variable) receives
    the steps reading → model (chunk n of N, from `llm_lane.detect`'s
    `on_progress`) → replacing.
  - **The job record:** the room job sets the hook around each file and keeps
    the current file, its size, its place, the step, the chunks and the
    timestamps in the job. The sweep and the check say which file they are
    on.
  - **The log:** a line per step and per chunk (file id and size, never its
    path).
- **ETA, worked out on the page each second:**
  - **This file:** time per chunk so far × chunks left.
  - **The room:** time so far ÷ bytes done (the current file counted by its
    chunks) × bytes left. Shown once there's enough to go on ("estimating…"
    before).
- **Sizes:** each file shows its size. Excel shows rows with something in
  them and its sheets; CSV/TSV show rows (`room_files.info`). They are counted
  at ZIP intake, or when the run reads the file, or in the sweep for files done
  before.
- **Stop:** it now cancels the run at once, the trusted model's calls
  included, and waits up to 10 s for it to wind down. The file it was on stays
  "not run yet"; files done stay done. The room shows "Stopped by you — N files
  not run yet". A cancel from the deck shutting down still keeps the room
  "running", so it resumes at the next start.

**The code pass on big workbooks (2026-11-14, branch `feat/rooms-replace-progress`):**

- **The slowness:** a payroll journal (about 75,000 cells) sat in "Replacing
  names" for over ten minutes on staging. The caller's terms (seed names plus
  the names the trusted model found) were searched with one regex per term,
  per cell, per round. Past 512 terms, Python's regex cache recompiled every
  pattern every time.
- **The fix:** `engine._extra_finder` indexes the terms by their first word
  once per run and looks up each cell's words, as the known-entity finder
  already did. Cost no longer grows with the number of terms.
  - Old code: still running after more than 9.5 min (20,006 cells,
    1,500 terms).
  - New code: 0.8 s for the same; 3.1 s for 75,406 cells and 2,000 terms.
  - A test checks the indexed finder against the old regex on emails, dotted
    words, hyphens, case and overlapping terms.
- **Progress:**
  - **Engine:** `engine.PROGRESS`, a context variable read inside the worker
    thread, gets `{round, pass, done, of}` at most twice a second. The passes
    are looking, matching, replacing, and checking at the end.
  - **Room job:** keeps it as `job.replace`, logs each pass with its time.
  - **Run strip:** "round 1 · looking for names in cells 40,000 of 96,952",
    a bar, and time left in that pass; the room's ETA counts the code pass.
  - **Small runs:** a file's own name (under 2,000 characters, one piece)
    isn't reported.
- **Stop:** it now stops the code pass too. `engine.CANCEL` is a
  `threading.Event` that `_anonymize` sets when its coroutine is cancelled,
  and the engine raises `RunCancelled` at its next piece of text. Late ticks
  from a stopped run are ignored.

**Sweep and leftover check, step by step (2026-11-14, branch `feat/rooms-sweep-check-progress`):**

- **What was missing:** after anonymizing, the strip only said "Final sweep" or
  "Checking for leftovers", with the bar left full from the phase before.
- **Each phase has its own clock and sizes:** a sweep round and the check
  each start their own `phase_started_at`, `bytes_total` and `bytes_done`,
  sized from the anonymized copies. The bar and the time left are for that
  phase: "left in this round", "left for the check".
- **Sweep:**
  - Header: "round 1 of up to 3", then files fixed and names put in this
    round.
  - File line: file n of N, its size and rows, and the step: reading the
    copy, then the engine's passes ("looking for names in 61,140 of 96,952
    cells"), saving the names, writing the fixed copy.
  - `anonymize_blob` now also reports `saving` (the vault) and `writing`
    (the copy). Engine ticks that arrive after `saving` are dropped.
- **Check:**
  - Header: how many real values (spellings included) it looks for, and how
    many files it has blocked so far.
  - Steps: reading, "Making the text Claude reads", "Looking for real names
    left" (with the text's length), "Indexing it for search" (with its pages).
- **Log:** each sweep round and file (names found, seconds) and each check
  file (characters, pages, leftovers, and the seconds for the text and for
  the check), plus round and check totals.
- **Leak check speed:** a value is matched as whole words, so `leaks` first
  takes the set of the text's words. A value with a word the text doesn't
  have isn't searched for. For a company matched loosely, `needles` now gives
  a `Loose(pattern, words)`. Measured on 1.5 MB of text with 4,000 values and
  500 company patterns: 6.5 s before, 0.08 s after. A test checks the
  results are the same as a plain search.
- **Why files still get blocked (made-up check):**
  - The sweep runs with one-word names off, so a surname alone that the
    vault holds as a person ("Uplata Horvat") stays in the copy.
  - A company is only matched with a capital first letter, so one written in
    lower case ("dobavljač sokol softver d.o.o.") stays in the copy too.
  - The leak check is case-blind and looks for every value, so it blocks
    both. Upper case, missing accents and a reversed full name are replaced.

**Leftovers: hidden by the check, or explained (2026-11-14, same branch):**

- **Where, as written:** `rooms_job.find_leftovers(text, needles)` maps each
  hit back to the text as written, using the fold with a position map
  (`_fold_map`). It gives one `{token, text, count}` per token and spelling.
  The longest match wins: "softver d.o.o" inside "galeb softver d.o.o"
  counts once, for the longer name. A token the check finds but can't place
  keeps an empty text, and still blocks.
- **Hidden with the token the check named:**
  - `engine.Options.pinned` takes (text, token) pairs: wherever the text
    stands as whole words (any case, any spacing), it is that vault entity.
    A spelling that differs beyond case gets its own place (⟦PERSON_7:2⟧).
    Through the route this is `data["pinned"]` (at most 2,000).
  - In the check, a file with leftovers runs once more through
    `anonymize_blob` with its spellings pinned (the sweep's settings). Then
    its text is checked again.
  - What went is listed as hidden; what is still there blocks the file. A
    released file that changes this way is taken back, as in the sweep.
- **Stored sealed:** `room_files.leftovers` holds `{hidden, left}` with the
  real spellings, sealed like the paths. The file list gets counts only; the
  file's text route gets the spellings (owner only). The hidden list stays
  across later checks until the file runs again from its original.
- **"Not a name here":** `rooms.allowed` (sealed), a list of
  `{text, tokens, at}`:
  - The room's runs keep the spelling as written (`keep`).
  - `needles(state, allowed)` stops looking for it. `room_needles` is used by
    every leak check of a room: the run, export, `room_get_file`, Claude's
    deliveries, tasks, roles. The MCP vault cache is keyed by the list too.
  - `POST /fd/rooms/{id}/allowed` adds a spelling. With `rerun`, files where
    it was hidden go back to "not run yet", to come back with the word in
    place. `POST …/allowed/remove` looks for it again. Values go in the
    body, never the URL. At most 300; values shorter than 4 characters are
    refused (they aren't looked for anyway).
- **UI:**
  - File viewer: a "Still in this file — why it's blocked" panel (red), and
    a "Hidden after the check" panel. Each row has the spelling, its token
    and real name, the count, **Show** (lights it up in both texts), and
    **Not a name here** / **Not a name — put it back** with an inline
    confirm.
  - File list: "+N after the check", and on a blocked file "Open it to see
    where, and what you can do".
  - Names to hide: a "Left as written — not names here" list with ×.
  - Run strip: "N leftovers hidden" during the check, and the step "Hiding
    what the sweep left".
- **Seen on a throwaway deck (made-up room):** a lower-case company in a
  journal was hidden 16 times in a workbook and twice in a text file, and no
  file was blocked. A name split across two cells was hidden too: code read
  the column header. The red panel was checked on a file marked blocked by
  hand.
- **Not done:** a real value in a file's name is listed (`where: "name"`) but
  not hidden. Names glued with underscores ("galeb_softver") aren't whole
  words to the leak check, as before.

**Workbooks shown as sheets (2026-11-14, branch `feat/rooms-sheet-view`):**

- **Viewer:** a workbook's text (`## Sheet: name`, then a markdown table) is
  parsed into sheets (`roomText.parseSheets`). Each cell knows its offset in
  the text, so search marks, the current match and token chips work as
  before.
  - Both sides draw the same grid: column letters, row numbers, Excel-style
    tabs at the bottom. Each tab shows its number of search matches.
  - Both sides share one width per column (the longest line on either side,
    4–40 characters), and each cell is one line (its own line breaks kept,
    the rest cut with "…" and shown on hover). So the two grids line up
    cell for cell, and scrolling follows by row, and side to side.
  - The search jumps to the sheet of the current match. A tab you pick stays
    until the next match. The text picture alt text goes on a "Pictures" tab.
  - A text longer than the page shows (200,000 characters) says so at the
    end (`clean_cut` / `original_cut` from the text route).
- **Numbers in the text (all workbook text, Claude's included):**
  `document_extract._xlsx_number`:
  - A date cell (built-in date formats, or a custom one with d/m/y) reads
    as `2023-03-15`, a date and time as `2023-03-15 12:00`, a time as
    `8:30`. 1904 workbooks are handled.
  - Other numbers lose the float noise: 15 significant digits, so
    `1234.5599999999999` becomes `1234.56`. Integers and ID numbers stored
    as numbers stay as they are.
- **Checked:** a stubbed UI check (real bundle, `/fd/*` from fixtures, the
  texts from the real extractor and code lane) on a made-up two-sheet
  payroll workbook, in dark and light. Both panes ended at the same scroll
  position, with no page errors.
- **Not done:** cell formatting (bold, fills, merged cells) isn't in the
  text, so it isn't drawn.
- **Gap (corrected on 2026-11-14, see the next section):** this said number
  cells are never anonymized. That was wrong. `office.prepare` does offer a
  whole number of 6+ digits to the engine. What was missing was headers below
  row 1, headers with other words in them, and phone numbers without a header.
  A date of birth in a date cell is still not hidden.

**Identifiers kept as numbers in a workbook (2026-11-14, branch `feat/xlsx-number-ids`):**

- **Rule:** amounts, hours, counts and ordinary dates stay numbers (Claude
  sums, filters and splits periods). Only identifiers get tokens: OIB, JMBG,
  phone numbers, account numbers.
- **What was there:** a number cell of 6+ digits already went to the engine,
  with its column's header as a hint. A valid OIB or JMBG (check digits) was
  found even with no header.
- **What was missing, now done (`office._xlsx`, `detectors.hint_for`):**
  - **Headers below row 1:** a header row is looked for in a sheet's first 30
    rows. A row with numbers in it is data, not a header; row 1 counts as
    before. A header applies to the rows under it, until the next header
    row. This also lets names under "Ime i prezime" in row 2 get their hint.
  - **Headers with other words:** an identifier named among other words is
    taken as one: "OIB zaposlenika", "Kontakt telefon", "Broj mobitela",
    "Tel.", "IBAN račun". The value still needs digits (`hint_span`), so a
    cost line called "Telefon" in a data row isn't hidden.
  - **Phone numbers with no header:** a number cell `385` + 8–9 digits (the
    "+" is gone) is a phone number.
  - **Float-style values:** a whole number written `12345678903.0` is read
    as `12345678903`, so its token has no ".0" left beside it.
  - A cell that gets a token becomes a text cell holding the token (as
    before). The amounts beside it stay number cells.
- **Tests:** `test_anonymize_xlsx_number_ids.py` (made-up values with valid
  check digits):
  - header words;
  - a title row above the header;
  - an OIB written as `….0`;
  - a phone number with no header;
  - amounts, codes and hours left as numbers;
  - a data row that isn't a header.
- **Not yet:** a date of birth in a date cell (a number to the office reader;
  to be hidden or cut to the year). A 9-digit Croatian mobile number without
  385 and without a phone header (`911234567`) isn't taken: it could be an
  amount.

**One person, one token number across files (2026-11-14, branch `fix/rooms-person-variants`):**

- **Found by Claude on staging** (a personnel FDD task): "payroll and register
  name tokens don't fully line up".
- **Cause:** the resolver matched a full name to a known person only when its
  words came in the same order. A register's "ŠIMUNIĆ DARKO" and a payroll's
  "Darko Šimunić" (made-up) became two people.
  - The known-name finder already took a name the other way round, but a
    name the trusted model or the detectors found goes through the resolver
    first, and its new entity won.
- **Fix (`engine._Resolver`):** a full name matches a known person by the same
  name words in any order (`_name_bag`: the stems sorted; case, diacritics
  and case endings were already folded). It gets that person's number, with
  its own spelling place (⟦PERSON_1:2⟧), so it restores as written.
  - Initials work either way round too ("D. Šimunić", "Šimunić D.") when only
    one person fits.
  - A vault that already split a person: the first token wins, in the
    resolver and in the known-name finder. A file run again from its
    original comes back with one number.
  - "Check again" can't heal it: it sweeps the anonymized copies, where the
    second token is already in place. Re-run the files, or make the room
    again.
- **Claude's instructions** now say that one number is one party in every
  file, and to match people across files on the number, ignoring the ":2".
- **Still apart, on purpose:**
  - a name with an extra word ("Darko Ivan Šimunić"), which may be another
    person;
  - an initial that fits two people.
- **Not yet:** a name split into two columns (IME | PREZIME) gives a one-word
  token per cell (the trusted model's call). A payroll with split columns
  doesn't join a register with full names on one token.

## Later

- **Fleet side door.** If the person turns on both connectors in one Claude chat,
  `send_task` can pull raw data from an agent that doesn't anonymize its chat.
  The room page warns about this. A person setting "Fleet connector answers go
  through the anonymizer" (Chat vault) would close it.
- **A deck on a laptop** (Captain desktop app) can't be reached by Desktop's
  remote connectors. Option: a Claude Desktop extension (local MCP server) that
  forwards to the deck with a PAT.
- **Old formats** (`.doc .xls .ppt .msg .rtf .odt`) via LibreOffice when installed.
- **Coarser numbers per room** (`generalize` exists), off by default.
- **"Delete originals"** after release; archiving a room. Deleting the vault makes
  the Inbox unreadable, so warn first.
- Rooms shared with members: decided owner-only for now.

## Risks

- **Recall.** A name both lanes miss reaches Claude for good. Seed terms, the
  final sweep, the leak check, the spot check and the release step make this
  smaller, not zero. The review page says so.
- **Identification without names.** Exact numbers, sector, city and founding year
  can point to one company. Accepted, because the analysis needs the numbers.
  Coarser numbers are a later option.
- **Claude guesses an identity.** The leak alarm in P3 flags real vault values in
  Claude's deliveries.
- **Time.** Hundreds of files with the local trusted model take hours. The job is
  resumable and shows an estimate.
- **Claude Desktop limits** (result size, call size, sandbox network) are unknown
  until P0.

## Open points (default in brackets)

- Keep the raw ZIP after release? [keep it, encrypted; needed for re-runs; a
  "Delete originals" button]
- May the person's own agents use the clean copies? [no in v1; maybe a "Copy to
  my files" later]
- Does a room expire? [no]
