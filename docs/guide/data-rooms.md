# Data rooms — a deal's files for Claude, the names kept on the deck

[← README](../../README.md) · [All guides](README.md)

New in 0.9.1. A data room takes a virtual data room (VDR), a ZIP or a folder of hundreds of files, and
anonymizes all of it into one vault. Claude reads the anonymized copies through its own connector,
**Captain Data Rooms**. It gets the story, the context and every number, but no names. Claude sends its
answers and files back through the same connector, and Flight Deck puts the real names back, for you
alone, on the deck.

**On this page:** [What a data room is](#what-a-data-room-is) · [Before you start (admin)](#before-you-start-admin) · [Make a room](#make-a-room) · [Review before Claude sees it](#review-before-claude-sees-it) · [Connect Claude](#connect-claude) · [Working in Claude](#working-in-claude) · [Getting results back](#getting-results-back) · [Limits and tips](#limits-and-tips)

## What a data room is

A buyer's team gets a VDR for *Velora d.o.o.* (every name in this guide is made up). They want Claude to
read all of it: the contracts, the payroll, the ledgers and the board minutes. They don't want Claude to
learn who Velora is, who works there or who is buying it. A data room does this:

| In the files | What Claude reads |
|---|---|
| Velora d.o.o. | ⟦ORG_1⟧ |
| VELORA DOO (in a ledger) | ⟦ORG_1:2⟧ |
| Northwind Capital | ⟦ORG_2⟧ |
| Ana Horvat | ⟦PERSON_1⟧ |
| HORVAT ANA (in the payroll) | ⟦PERSON_1:2⟧ |
| Kovačka cesta 12 | ⟦ADDRESS_1⟧ |
| Revenue 4,213,500 EUR, signed 2025-03-15 | Revenue 4,213,500 EUR, signed 2025-03-15 |

- **One room, one vault.** A party has the same token in every file of the room. ⟦PERSON_1:2⟧ is another
  spelling of ⟦PERSON_1⟧ (surname first, capitals, no diacritics), so Claude can match a payroll
  against a register on the number.
- **Numbers, dates and the story stay.** Identifiers get tokens: OIB, JMBG, phone, account and
  registry numbers, even when a workbook keeps them as numbers.
- **Roles say what, never who.** You tell Claude ⟦ORG_1⟧ is the *Target* and ⟦PERSON_1⟧ is the *CFO of
  the target*.
- **Nothing reaches Claude before you release it.** A file that still holds a real name is blocked and
  never released.
- **Results come back restored, on the deck only.** What Claude delivers is stored encrypted in the room,
  with the real names put back for you. Claude gets a receipt, never the restored text.

How it goes:

1. Make a room and add the files.
2. Run it: anonymize, a final sweep, a check for leftovers.
3. Review: the file viewer, roles, **Same party?**, file descriptions.
4. **Release** the clean files.
5. Connect Claude to the room, or **Download for Claude** for a Claude Project.
6. Claude works and delivers results through the connector.
7. You read and download the results, with real names, under **From Claude**.

Data rooms build on the anonymizer. How it finds names, and the trusted model behind it, are in the
[Anonymization guide](anonymization.md).

## Before you start (admin)

Anyone signed in to the deck can make data rooms. A room belongs to its owner alone: an admin's act-as
can't open it.

### A trusted model

Without one, code alone looks for names, and the room's health panel says **None — code alone looks for
names**. With one, a model on the deck's side also finds names code can't. The same model writes the file
descriptions and fixes Croatian case endings in deliveries. It sees the real names, so it has to be a model
you trust: local, or one you have an agreement with.

An admin sets it up in **Admin console → Settings → Anonymization**, or with the `FD_ANONYMIZE_*`
variables. The [Anonymization guide](anonymization.md#the-trusted-model) has the steps.

Hundreds of files with a local model take hours. One room runs at a time on a deck; the others wait.

### OCR

Scanned PDF pages, picture files and the text in pictures inside Office files are read with local OCR,
never a language model. Install Tesseract or RapidOCR where Flight Deck runs. Tesseract reads Croatian and
English when it has both language models. Without OCR, the health panel says **Not installed — scans only
removed**.

### A public HTTPS address

Claude calls a custom connector from Anthropic's cloud, not from your computer. A deck on `localhost` or
your office network can't be reached.

1. Serve the deck over HTTPS at a public name, for example `https://deck.example.com`, behind a reverse
   proxy such as Caddy or nginx.
2. Set `FD_PUBLIC_URL=https://deck.example.com` and a stable `FD_JWT_SECRET`. Restart Flight Deck.
3. The proxy must pass these paths to Flight Deck. Most proxies pass everything; check if yours lists
   paths:
   - `/fd/mcp-rooms` (the connector);
   - `/.well-known/oauth-protected-resource/fd/mcp-rooms` and `/.well-known/oauth-authorization-server`;
   - `/register`, `/authorize`, `/authorize/rooms` and `/token` (the sign-in).
4. Check it: open **Data rooms** once you have a room. The **Claude's access** card shows the connector
   address. It should read `https://deck.example.com/fd/mcp-rooms`. An `http://` address or an internal
   name means `FD_PUBLIC_URL` isn't set.

A deck without a public address, such as a laptop or the desktop app, can still use
[Download for Claude](#no-public-address-download-for-claude).

### Cloudflare and `Claude-User`

Claude's calls after sign-in carry the user agent `Claude-User`. Cloudflare's AI-bot blocking answers
them at Cloudflare's edge with `403 "Your request was blocked."`. A Cloudflare quick tunnel
(`*.trycloudflare.com`) does the same. The sign-in calls carry another user agent and get through, so it
looks like this:

- You get through both sign-in pages, and the app shows up under **Claude's access** on the deck.
- Claude then reports that the server refused its request.
- **Claude's activity** in the room stays empty. No call reached the deck.

On a deck behind Cloudflare, do one of these:

- turn off AI-bot blocking for the deck's zone;
- add a WAF skip rule for the deck's host that lets `Claude-User` through, or that skips these paths:
  `/fd/mcp-rooms`, `/.well-known/*`, `/register`, `/authorize*`, `/token`.

A quick tunnel's settings can't be changed. Use a named tunnel on your own zone with the blocking off, or
another way in. The same applies to any firewall: it must let Anthropic's servers reach those paths.

### Settings

| Setting | Default | What it does |
|---|---|---|
| `FD_PUBLIC_URL` | none | The deck's public HTTPS address. The connector address and the sign-in addresses Claude is given are built from it. Without it, a proxy's `X-Forwarded-Proto` and `X-Forwarded-Host` are used, else the address the request came to. |
| `FD_JWT_SECRET` | none | A stable secret. The connector's sign-in tokens are signed with it. |
| `FD_ALLOWED_HOSTS` | none | Only when people reach the deck by another name than `FD_PUBLIC_URL`'s host. |
| `FD_ROOMS_MAX_ZIP_MB` | `25` | The biggest ZIP you can add to a room. |
| `FD_ROOMS_DESCRIBE_CHARS` | `16000` | How much of one file the trusted model reads to describe it. |
| `FD_DATA_DIR` | the deck's data folder | Where `rooms.db` and each room's files (`rooms/<id>/`) live. |
| `FD_DATA_KEY_FILE` | `<FD home>/data_key` | The deck's data key. Rooms can't be made without it. |
| `FD_ANONYMIZE_*` | none | A trusted model from the environment. See the [Anonymization guide](anonymization.md). |

Restart Flight Deck after changing any of them.

## Make a room

### 1. Open Data rooms

- **Kiosk:** **Settings** → **Data rooms** (under *What they keep*), or **More** in the agent column →
  **Data rooms**.
- **Full layout:** **Files** → **Data rooms** in the sidebar.

The page can fill the window with its fullscreen button, and so can every dialog inside a room.

### 2. Create it

1. Under **New data room**, fill in **Codename (Claude sees this)**, for example *Project Falcon*.
   Captain doesn't check the codename for real names, so keep it neutral.
2. Fill in **Your name for it (only you see this)**, for example *Velora acquisition*.
3. Click **Create data room**. The room opens on **Add files**.

A room has a left rail in two groups. **Room** has **Files**, **Add files**, **Names to hide**, **Roles**
and **Same party?**. **Claude** has **From Claude**, **Tasks for Claude**, **Claude's activity** and
**Download for Claude**. On a phone the rail is a strip of chips. The bar at the top holds **Run** (or
**Check again**), **Release**, delete, fullscreen and close.

### 3. Add files

**A ZIP:**

1. Open **Add files**.
2. Drop the ZIP on **Drop a ZIP here, or click to choose one**.
3. Click **Add the ZIP**.

**A folder:**

1. Open **Add files**.
2. An admin can fill in **On the deck's machine** with the folder's full path, starting with `/`. Everyone
   else fills in **A folder in Files** (the folder's name in Files) and, if needed, **Path inside it
   (optional)**.
3. Click **Read the folder**. The folder is read where it is and never changed.

A notice says how many files were added, replaced or unchanged, which can't be read, and what was left out.
Adding again replaces the files that changed. You can add several ZIPs to one room, so a VDR over the
ZIP limit can go in parts. A room reads one folder; for another folder, make another room.

**ZIP limits:** 25 MB by default (`FD_ROOMS_MAX_ZIP_MB`), 3,000 entries, 500 MB unpacked. A ZIP inside the
ZIP is unpacked one level. A password-protected ZIP is refused. Junk such as `__MACOSX/`, `.DS_Store` and
`Thumbs.db` is dropped, and unsafe paths are left out. Names from old Windows ZIPs (Croatian names in a DOS
code page) are decoded. **Folders:** up to 3,000 files. **Every file:** up to 25 MB.

**What it reads:**

| Files | What happens |
|---|---|
| Word, Excel, PowerPoint (`.docx`, `.xlsx`, `.pptx`) | Anonymized inside the file: text, tables, cells, sheet names, alt text and the data behind charts. Every picture becomes a grey "picture removed" box; the text OCR reads in it is kept, anonymized. The first-page thumbnail goes. Workbooks keep their numbers and formulas. |
| PDF | Pages with a text layer keep it; scanned pages are read with OCR. |
| Pictures (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`, `.gif`) | Read with OCR into text, then anonymized. |
| Text (`.txt`, `.md`, `.csv`, `.tsv`, `.json`, `.html`, `.xml`, `.eml` and the like) | Anonymized as text. |
| Anything else (`.doc`, `.xls`, `.ppt`, `.msg`, `.rtf`, `.odt`, `.xlsm` …), and files over 25 MB | Listed as **Can't be read**, never passed on. Save old formats as `.docx`, `.xlsx` or `.pptx` and add them again. |

Names of files and folders go through the vault too. When a name changed, the file's row says **Claude
sees** and the anonymized name.

### 4. Names to hide (before the first run)

Names you already know are hidden first, wherever they appear. This is the biggest help with recall,
especially for short names a document never spells out in full.

1. Open **Names to hide**.
2. Fill in, one per line: **Companies**, **People**, **Products and brands**, **Web domains**, **Registry
   and ID numbers**, **Anything else** (short names, codenames, addresses, project names).
3. Click **Save**.

For Velora that might be *Velora d.o.o.*, *Velora*, *velora.example*, *Northwind Capital*, *Northwind*,
*Ana Horvat*. Only you see this list. Saved after a run, it reaches the files already done when you click
**Check again**.

### 5. Run it

1. Click **Run** in the room's bar. Later, with new files next to done ones, it says **Run new files**.
2. A strip under the bar shows where the run is. You can close the dialog; the run keeps going, and it
   resumes after a deck restart.

The strip's first line is the phase:

| Phase | What it does |
|---|---|
| **Waiting for another room to finish** | Another room has the deck's run slot. |
| **Anonymizing files** | Each new file, one by one, with *n of N done*. |
| **Naming folders** | Folder names go through the vault. |
| **Final sweep** | Code goes over every anonymized copy again with all the names the room knows now, because a name first found in file 300 is still in file 12. *Round n of up to 3*, with files fixed and names put in. |
| **Checking for leftovers** | Every real value the room knows, in every spelling, is looked for in the text Claude will read. *N real values looked for in every file*, leftovers hidden, files blocked so far. |

The second line is the file it's on: *File n of N*, its name, size and rows, and the step: **Reading the
file**, **Finding names** (*chunk n of N* of the trusted model), **Replacing names** (*looking for names in
40,000 of 96,952 cells*), **Saving the names it learned**, **Writing the anonymized copy**, **Making the
text Claude reads**, **Looking for real names left**, **Indexing it for search**. Each line has a bar and
the time left: *left for the room*, *left in this round* or *left for the check*. It says
**estimating…** until there's enough to go on.

**Stop** stops the run at once, the trusted model's calls included. The button shows **Stopping…**, then
the room says *Stopped by you — N files not run yet. Run the room to finish them.* Files already done stay
done. Click **Run** to finish.

### 6. Leftovers, blocked files and Check again

After a run, each file in **Files** has a state:

| State | Meaning |
|---|---|
| **Not run yet** | Added and waiting for a run. |
| **Clean** | Anonymized and checked. It can be released. |
| **Released** | Claude can read it. |
| **Blocked** | The check still found a real value in it. It is never released. |
| **Failed** | The anonymizer couldn't finish it; the reason is on its row. Click **Retry failed**. |
| **Can't be read** | A kind of file the room can't read, or over 25 MB. |

**Hidden after the check.** The sweep leaves a surname on its own, or a company written in lower case, to
the trusted model. When the check still finds one, it gives it the token of the name it belongs to and
checks the file again. The row says **+N after the check**.

**Blocked.** What the check couldn't hide blocks the file. The row says **Open it to see where, and what
you can do →**. In the file viewer, the red panel **Still in this file — why it's blocked** lists each
spelling with its token, the real name, how often it occurs and where. A name may run across two cells,
sit in a sheet name or a formula, or be in the file's own name. Each row has:

- **Show**: lights the spelling up in both texts.
- **Not a name here**: for a word that isn't a name in this room (a common word that happens to match a
  surname, say). You confirm with **Yes, it's not a name**. The word stays as written in every file of the
  room, and the check stops looking for it. Then click **Check again** to clear the block.

If it is a name, remove the file, or fix the original and add it again.

Hidden spellings have **Not a name — put it back**. The files where it was hidden go back to **Not run
yet**; click **Run** and they run again from their originals, with the word in place.

**Names to hide** lists every such word under **Left as written — not names here**; × looks for it again.

**Check again.** With nothing new to run, **Run** becomes **Check again**. It goes over every file again
with the names the room knows now and your names to hide, then checks it for real names. Use it after you
add names to hide, or after **Not a name here**. A released file that changes is taken back, with the
warning *Claude may have read the old copy. Release it again.*

## Review before Claude sees it

### The file viewer

Click a file in **Files**. The viewer shows the **Original** (only you see it) next to **What Claude
reads**, scrolling together.

- **Search by** **Text** (words on either side), **Token** or **Real name**. The token on Claude's side
  and every spelling of the real name on yours light up yellow, with previous and next. Clicking a token
  in the text shows it everywhere.
- The link button switches scrolling together off and on. On a phone, **Original** and **Claude** switch
  sides.
- **Workbooks are shown as sheets** on both sides: column letters, row numbers and tabs at the bottom,
  each with its number of search matches. The two grids line up cell for cell, so the only difference you
  see is the tokens. Text read from removed pictures has its own **Pictures** tab.
- **Anonymized copy** downloads the file as Claude would get it.
- A long file shows its first 200,000 characters and says so at the end.

The **Files** list helps too: **Find a file by its name or what it's about**, filters for **Blocked**,
**Failed**, **Warnings** and **Can't read**, a folder tree or **One list**, each file's size, rows and
sheets, and how many names were hidden.

### Roles

Roles tell Claude what each party is. Claude reasons much better with them.

1. Open **Roles**. Every person, company, address and other detail is listed with its token and real name.
2. Type a role in **Role — what, not who**: *Target*, *Buyer*, *CFO of the target*, *Seller's adviser*.
3. Click **Save roles**.

A role that names a real value from the vault is refused, and nothing is saved, because Claude sees
roles. Use **Find a name, token or role** and **With a role** to find your way in a long list.

### Same party?

Code gives one token to what it can prove is one party: another word order, capitals, missing
diacritics, a middle name when only one person fits, *Kovačka c. 12* and *KOVAČKA CESTA 12*. What it
can't prove, it asks you about in **Same party?**: a one-letter typo, a first name alone, a surname that
changed between files, a company's short name, the same street and number in two towns. The badge on
the rail counts the open questions.

Each pair shows both tokens with their real names, how many files each is in, why it's asked, and code's
score: how sure it is the two are one (0–99%). The token in more files is kept by default; the arrows
button keeps the other one.

- **Same** makes the second token the first one's spelling, in the vault and in every copy of the room,
  without a new run. Its old token still restores in earlier deliveries.
- **Not the same** is remembered; the pair isn't asked again.

Many at once:

1. Narrow to **People**, **Companies** or **Addresses** if you like.
2. Move the **Score** slider (90% by default; each room remembers yours in this browser).
3. Pick **N% or more** or **Under N%**.
4. Click **Same for all N** or **Not the same for all N**, and confirm with **Yes, make them one** or
   **Yes, not the same**. Making pairs one can't be undone from the deck.

A first name alone stops at 60%, a shared first name on two lists at 55%, the same street and number in
another town at 50%. A name cut off at the start of a longer one, or the same name without diacritics,
scores in the high 90s. Read the reasons before you answer a whole side.

Not in the list? Under **Two tokens you know are one**, fill in **Keep** and **Is the same as** with the
two tokens as written (⟦PERSON_3⟧, without a `:2`) and click **Make one**. **Made one** lists what you
merged. Answer when the room isn't running.

### What each file is about

After a run, the trusted model describes every clean file in the background, from the text Claude reads:

- one line;
- a paragraph;
- a summary of 3–5 paragraphs;
- up to 10 keywords and up to 10 topics.

Descriptions are written in English, even for Croatian files, so a room has one language to search in.
They hold tokens, never real names. One that holds a real value anyway is **Held back** and not kept.
Claude uses them to pick files, and is told to read the file itself before relying on a detail.

- A strip says **Describing files**, *n of N done*, the time left and the file it's on, with **Stop**.
  A room's run always goes first.
- In **Files**, **Describe N files** describes the files that need it: new, changed, failed or held back.
  **Describe all again** (or **Describe again**) writes every description from scratch.
- In the file viewer, **About this file** has the one line, topic and keyword chips, and **Paragraph and
  summary**. Switch between **Real names** and **As Claude reads it**. **Describe again** redoes that file.
- Each file's one line also shows in the file list, with real names, and the file search looks in it.

Rooms that ran before describing existed: click **Describe N files** once.

### The health panel

At the bottom of the rail:

- **Trusted model**: its name, or **None — code alone looks for names**.
- **OCR**: **Scans and pictures are read**, or **Not installed — scans only removed**.
- **Descriptions**: *n of N clean files*, and how many failed.

### Release

1. Click **Release N** in the room's bar. N is how many clean files aren't released yet.
2. A notice says how many were released. Their rows say **Released**.

Blocked and failed files stay on the deck. Files you add later wait for their own release. A released file
that a later sweep or check changes is taken back until you release it again.

## Connect Claude

The room connector is its own connector, separate from the fleet connector for your agents. It has its
own address and sign-in, and only room tools. A fleet sign-in doesn't work for rooms, and a room sign-in
doesn't work anywhere else.

### 1. Copy the address

1. Open **Data rooms**. The **Claude's access** card shows when you have at least one room.
2. Click **Copy** next to the connector address (`https://deck.example.com/fd/mcp-rooms`).

### 2. Add the connector in Claude

In Claude (claude.ai, Claude Desktop or Cowork):

1. **Settings → Connectors → Add custom connector**.
2. Name it **Captain Data Rooms** and paste the address.
3. Connect it. Claude opens the deck's sign-in page in your browser.

### 3. Sign in and pick the rooms

1. **Connect to Flight Deck**: fill in your deck **Email** and **Password**, then **Approve**. The page has
   no Google button, so your account needs a password.
2. **Which data rooms?**: tick the rooms Claude may read. Each shows how many files are released. Or tick
   **All my data rooms, and new ones**.
3. Click **Allow**.

### 4. What connected looks like

On the deck, **Claude's access** lists the app by the name it registered with, when it connected, and a
checkbox per room plus **All rooms, and new ones**.

- Change the rooms and click **Save**. It takes effect at the next call, not when the sign-in expires.
- **Disconnect** removes the app's room access at once and revokes its sign-in. Remove the connector in
  Claude's settings too. Disconnecting it under **Connections → Agent access for Claude (MCP)** also drops
  its room access.

In Claude, the connector lists its tools. In the room, **Claude's activity** fills up as Claude reads.

### No public address? Download for Claude

**Download for Claude** builds a ZIP to add to a Claude Project, for work beside or instead of the
connector. It works on any deck.

1. Open **Download for Claude** in the room's rail.
2. Tick **Text only** for a smaller ZIP: Word, Excel and PowerPoint come as markdown, which every Claude
   reads.
3. Click **Download ZIP**.

The ZIP has one folder named after the codename:

- the released files at their anonymized paths: Office files as files, PDFs and scans as the markdown
  they became;
- `ROLES.md`: each token's role;
- `FILES.md`: what each file is about, when the files are described;
- `README-for-Claude.md`: the connector's rules, and how to give results back.

Every piece of text in every file is checked for real values once more. A file with a hit is left out and
listed (*Left out, because they still hold a real value*), and the room gets an alarm. The ZIP is up to
300 MB, downloads once, and is gone after an hour.

Unpack it and add the files to a Claude Project. Results come back through the connector, if it's on, or
as a file: ask Claude to keep the tokens in what it writes, save its answer as `.md` (or the file it made),
and add it with **Add a file Claude made** under **From Claude**.

## Working in Claude

Turn the **Captain Data Rooms** connector on in the chat. Talk about parties by role or token: *the target*,
*the buyer*, ⟦PERSON_1⟧. Don't type real names in Claude: a real name you type reaches Claude for good, and
if Claude then searches the room for it, the room raises an alarm. To give Claude work in your own words,
real names included, use
[Tasks for Claude](#tasks-for-claude): they are anonymized on the deck first.

### Example prompts

> Use the Captain Data Rooms connector. In Project Falcon, give me a two-page overview of the target: what
> it sells, revenue and EBITDA for the last three years, and the five biggest risks. Cite file and page.
> Deliver it to the room.

> Find every customer contract with a change-of-control clause. Deliver an Excel workbook: one row per
> contract, with the counterparty's token, term, notice period, the clause, and file and page.

> Match the 2025 payroll against the staff register. Who is paid but not on the register, and who is on
> the register but not paid? Match people on the token number. Deliver a short memo.

> Draft a 10-slide deck for the buyer's investment committee: the deal, the numbers, the risks and open
> questions for the seller. Deliver it as a PowerPoint deck.

> Check my Captain data room tasks.

### Claude's tools

| Tool | What it does |
|---|---|
| `room_list` | The rooms this connector may read: id, codename, files released, tasks waiting. Claude calls it first. |
| `room_overview` | One room: each token's role, its folders, file kinds, size and main topics. |
| `room_list_files` | The released files (of one folder, if asked): path, kind, characters, pages and a one-line description. |
| `room_find_files` | The files about something, best first: ranked by their descriptions, keywords and topics, then by their text. Tokens work. |
| `room_file_summaries` | A paragraph, a summary, keywords and topics for up to 20 files. |
| `room_read` | One file, page by page (about 50,000 characters a page). Word, Excel, PowerPoint and PDF come as markdown. |
| `room_search` | Passages that match, with the file and the page to read. Words, numbers and tokens work. |
| `room_get_file` | A released file itself (the anonymized copy, up to 10 MB), for Claude to open with code. |
| `room_deliver` | A result as markdown in tokens. About 20,000 characters a call; longer goes in parts. |
| `room_deliver_document` | A Word document, built on the deck from markdown. |
| `room_deliver_workbook` | An Excel workbook, built on the deck from sheets, rows, formulas, formats and charts. |
| `room_deliver_deck` | A PowerPoint deck, built on the deck from slides with titles, bullets, tables, charts and notes. |
| `room_deliver_file` | A file Claude made itself (Word, Excel, PowerPoint, markdown, text, CSV or JSON), as base64. |
| `room_tasks` | The tasks you left in a room, in tokens. |
| `room_get_delivery` | One of Claude's earlier deliveries, in tokens, page by page: the context a task refers to. |

Every tool sees only the released, clean files of the rooms you ticked. The deliver tools take `task=<n>`
when a delivery answers a task.

### The rules Claude is given

The connector gives Claude these instructions when it connects:

- A token like ⟦ORG_7⟧ stands for a real name. ⟦ORG_7:2⟧ is the same party written another way, and one
  token number is one party in every file: match on the number, ignore the `:2`. The same goes for
  addresses.
- Two tokens that look like one person (same role and pay, one stops the month the other starts) may be
  treated as one only as a stated assumption, with the evidence, keeping both tokens.
- Keep every token exactly as written, brackets included, in answers and in every file: titles, sheet
  names, notes and file names too. Never replace a token with a codename, a role or a description, and
  never guess who a token is.
- Start with `room_list`, then `room_overview`, and use the roles.
- Pick files with the descriptions, then read what matters; don't cite a description as the file. Cite
  files by path and page.
- Only released files are there. If something seems missing, say so; don't fill gaps with outside
  knowledge about who the parties might be.
- Deliver results with `room_deliver`, in parts if long, with no pictures or links.
- Don't make Office files yourself: send their content to the `room_deliver_document`, `_workbook` or
  `_deck` tools, so the names can be put back. Formulas stay inside the workbook.
- To work on a file with code, use `room_get_file`. A file Claude makes itself keeps the tokens and goes
  back with `room_deliver_file`, or to you, to add on the deck.
- When you ask it to check its tasks, call `room_tasks`, do each one, and deliver each answer with its
  task number.

## Getting results back

### From Claude

Every delivery lands in the room's **From Claude**. The rail and the room list show *N new from Claude*.
The bell says *Claude delivered a result* and *Delivery #n is in the data room Project Falcon*, with no
title. The bell feeds the Line and your channels (Telegram, WhatsApp, mail), so they get the same generic
text and never a restored name.

Each row shows the title, number, time and app, and badges:

- the kind: **Excel**, **PowerPoint**, **Word**, **Markdown**, **CSV** …;
- **answers task N**;
- **case endings…** while the trusted model works on a Croatian delivery;
- **no tokens**: nothing could be put back, because Claude wrote a codename or descriptions instead of
  tokens; ask it to keep the tokens;
- **real values**: Claude wrote a real value from the vault itself. It got it some other way, or guessed.
  Claude isn't told whether it was right;
- **N unknown**: tokens that aren't in this room. Claude made them up, and they stay as written;
- *N put back*: how many names were restored.

### Reading a delivery

Click a row. The reader shows the delivery as a document:

- **Real names** or **As Claude wrote it**.
- For a Croatian delivery, **In their case** or **As stored**. Tokens come back in the stored form, so
  *s ⟦PERSON_1⟧* reads *s Ana Horvat* where Croatian wants *s Anom Horvat*. The trusted model puts people's
  names into the case each sentence needs, and a check makes sure each name stays the same name. Companies
  and places stay as written. **Fix case endings** runs it again when it failed, when no trusted model was
  set up, or when the delivery wasn't taken for Croatian.
- **Search this delivery…** with the matches highlighted.
- Restored text shows no pictures and no live links. A link shows its text, with the address in a
  tooltip, so a click can't send the names anywhere.

### Downloads

- A text delivery: **Markdown** or **Word** (the Word file is built on the deck).
- A file built on the deck or made by Claude: one button named for its kind, such as **Excel**,
  **PowerPoint** or **Word**.

Downloads follow the **In their case** / **As stored** switch. Delete a delivery with the bin icon, then
**Delete** (*This can't be undone*).

### Files Claude made itself

When Claude makes a file in its own workspace (an Excel model, say), it sends it with `room_deliver_file`.
For a bigger one it gives the file to you:

1. Save the file Claude made.
2. In **From Claude**, click **Add a file Claude made**. It takes `.docx`, `.xlsx`, `.pptx`, `.md`,
   `.txt`, `.csv`, `.tsv` and `.json`, up to 25 MB.
3. The names are put back inside the file and it opens in the reader. In the list it says **Added by you**
   where Claude's name would be, and it rings no bell.

### Who can open them

Only the room's owner, on this deck. Deliveries are stored encrypted with the room vault's key, outside the
files agents can read. Admin act-as can't open them, and there are no public links or sharing. Deleting
the room's vault makes them unreadable for good.

### Tasks for Claude

A task is work you leave in a room for Claude to pick up. Write it in your own words, real names included.
The deck anonymizes it with the room's own vault before Claude sees it.

1. Open **Tasks for Claude**.
2. Write the task (up to 8,000 characters).
3. Click **See what Claude reads**. The preview, **Claude will read**, shows it in tokens. If a real name is
   still in it, the deck says so; reword it, or add the name to **Names to hide**.
4. Click **Leave it for Claude** (or **Edit**). It shows **Waiting for Claude**.
5. In Claude, with the connector on, say **"Check my Captain data room tasks"**. The deck has a copy button
   for that line.
6. The task goes **Claude has it**, then **Answered**, with **Open the answer #n**. **Cancel** withdraws an
   open one.

To ask about a delivery, open it and click **Ask Claude**. The **Ask Claude about this** panel queues a
task about that delivery, and Claude gets the delivery with it, in tokens.

Claude picks tasks up only when you ask it in Claude. Nothing runs through an API in the background.

### Claude's activity

**Claude's activity** lists every connector call for the room: the tool, the file and page, and the app.
It never shows the text Claude got or what it searched for. Exports show there as **Download for Claude**.

If a real value from the room shows up in what Claude sends, such as a search, a file path or a
delivery, the room shows a red notice: *Claude sent a real value from this room*. It didn't come from the
room. It reached Claude some other way: typed in the chat, a file given to Claude directly, or a guess.
Check that conversation, then click **Got it**.

The room's vault is also on the **Anonymize** page as *Data room · Project Falcon · …*, and each token's
history shows what went out to Claude and what came back.

## Limits and tips

### What can still slip

The sweep, the leftover check, your review and the release step make a miss rarer, not impossible. A name
that code and the trusted model both miss, and that the vault never learns, reaches Claude for good.

- **A name the vault never learned.** A fund's own short name that never appears in full, say. Add it to
  **Names to hide** and click **Check again**.
- **A name in a file's own name.** It's listed, and the file is blocked, but it isn't hidden. Rename the
  original and add it again.
- **Names glued with underscores** (`velora_holding`) aren't whole words to the leftover check, so it can
  miss them.
- **Workbook cells.** A date of birth in a date cell isn't hidden. A 9-digit mobile number without `385` and
  without a phone header could be an amount, so it isn't hidden either. Identifiers stored as numbers
  under a header (OIB, JMBG, phone, IBAN, account) are, and so are valid OIB and JMBG numbers without one.
- **Split name columns.** A payroll with first and last names in two columns gets one token per cell, so it
  doesn't line up with a register that has full names in one column.
- **One party, two tokens.** A typo, a first name alone, a changed surname or a short name can get a token
  of its own. **Same party?** is where you join them.
- **Late fixes.** When a sweep or check changes a released file, Claude may already have read the old copy
  in a chat. The file is taken back and warns you; start a new chat after you release it again.
- **Who it is without names.** Exact numbers, sector, city and founding year can still point to one
  company. That is accepted: the analysis needs the numbers.
- **Pictures.** Their text is kept in alt text and under **Pictures** in the text Claude reads, not where
  the picture was.
- **The fleet connector in the same chat.** With both Captain connectors on in one Claude chat, a task
  sent to one of your agents could bring back raw text. Keep the fleet connector off in data-room chats.
- **A linked folder** isn't watched. After its originals change, click **Read the folder** again and run.

### After a deck update

- Rooms anonymized before an update keep their old tokens. **Check again** doesn't redo earlier token
  choices. To get the new ones, make the room again with the same files, or remove files and add them
  again so they run from the originals.
- **Same party?** scores and file descriptions reach existing rooms without a new run. Click **Describe N
  files** once for files that have none.
- Claude sees new connector tools after you refresh or reconnect the connector in Claude.

### Deleting a room

The bin icon in the room's bar opens **Delete Project Falcon?**. The room's originals, anonymized copies
and text are deleted from the deck; a folder on the deck's machine is never touched. **Delete its vault
too** is ticked by default: the real names behind the tokens go with it, and nothing Claude made from this
room can be put back afterwards. Then click **Delete**. It can't be undone.

### Fixed limits

| What | Limit |
|---|---|
| A ZIP | 25 MB (`FD_ROOMS_MAX_ZIP_MB`), 3,000 entries, 500 MB unpacked |
| A folder | 3,000 files |
| One file | 25 MB |
| A page Claude reads | about 50,000 characters |
| `room_search` / `room_find_files` / `room_file_summaries` | 25 passages / 30 files / 20 files |
| One delivery call | 200,000 characters (about 20,000 advised); up to 50 parts, kept a day until the last arrives |
| `room_deliver_file` | 3 MB decoded (about 15 KB fits in one call) |
| `room_get_file` | 10 MB |
| **Add a file Claude made** | 25 MB |
| A task | 8,000 characters |
| **Download for Claude** | 300 MB, one download, gone after an hour |
| **Not a name here** | 300 words a room, 4 characters or more |
| **Same party?** in one click | 300 pairs |
| The file viewer | the first 200,000 characters of a file |

### Tips

- Fill in **Names to hide** before the first run. It's cheaper than finding names afterwards.
- Set roles before you connect Claude. **Roles** and `room_overview` are how Claude tells the target from
  the buyer.
- Give each deal its own room. Tokens are numbered per room, so ⟦ORG_1⟧ in one room isn't ⟦ORG_1⟧ in
  another.
- Spot-check a few files of each kind in the viewer before you release: a payroll, a ledger, a scan, a
  slide deck with logos.
