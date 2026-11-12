# Anonymization — plan

A Flight Deck tool, available to every agent and to the person, that finds
private data (names, dates of birth, bank accounts, emails, addresses, phone
numbers, ID numbers, companies …) in a text, file or page and replaces it with
tokens that mean nothing outside Captain. The real values are kept in an
encrypted **vault** in Flight Deck, so the person (never the model) can put
them back later.

Status: P0–P4, members and the name models are **merged** (PRs #511 #512 #514 #515 #516 #517, main
17c73e94). **P5** — files where they are (VFS file, VFS folder, Library file) — built on branch
`feat/anonymize-vfs` (PR #518). **P6** — the Chrome extension — PR #519. **P7** — the whole chat — built on
`feat/anonymize-chat`. True PDF redaction is skipped
(decided 2026-11-11).

## Principles

1. **Two lanes, one contract.** Detection runs in one of two lanes:
   *deterministic* (our code: patterns, checksums, labels, the vault's own
   values) or *trusted LLM* (a provider the admin trusts). Both lanes only
   **detect** — they return spans (exact text + kind). One shared engine does
   everything after that: resolve mentions to entities, assign tokens, replace,
   verify. So both lanes give the same output shape, the same tokens for the
   same entity, and the same guarantees. They will not always find the same
   things (an LLM sees "the only female surgeon on Vis"; a regex doesn't), so
   the default when a trusted provider exists is `both`: deterministic always
   runs, the LLM may add spans, never remove one.
2. **The LLM never rewrites the text.** It returns spans; our code replaces.
   A model asked to rewrite paraphrases, drops lines and invents.
3. **Raw data never passes through the calling agent's model.** The tool takes
   references (a file path, a `vfs:` path, a URL) and reads them inside the tool
   process — Captain code, not the model — then sends the content to Flight
   Deck over the deck connection. Only the tokenized result reaches the model.
   Text passed inline is accepted but marked `already_exposed` (the model has
   already seen it). Restoring works the same way in reverse: the restored file
   is written for the person; the agent gets its path and counts, never the
   contents.
4. **Trusted means admin-trusted, not "the agent's model".** Local providers
   (`ollama`, `litert`) are trusted by default; a cloud endpoint only when an
   admin marks it (zero retention, DPA, region). This is a deliberate exception
   to "agent work runs on the agent's own model" — otherwise anonymizing would
   send the raw document to the very provider it is being hidden from. If the
   LLM lane fails, the call fails or returns deterministic-only results marked
   `llm: failed` — never a silent downgrade.
5. **Honest naming.** A vault makes the result *pseudonymized* (GDPR: still
   personal data, because it can be reversed). `mode: one_way` keeps no vault —
   that is anonymization of the identifiers found (quasi-identifier
   generalization comes in P4).
6. **Fail closed.** After replacing, the output is searched for every original
   value in the vault and the detectors run again. Any hit is fixed in a second
   pass; if one still remains, the call fails and returns nothing.

## Tokens

`⟦PERSON_3⟧`, `⟦EMAIL_1⟧`, `⟦IBAN_2⟧`, `⟦ORG_1⟧` … — typed so a model can still
reason about the text, stable within a vault (the same person gets the same
token in every document of a case), **assigned, never derived** from the value
(a counter per kind; a hash of an email can be reversed with a list of emails).
Different vaults reuse numbers, so tokens never link two cases. Restore also
accepts `[[PERSON_3]]` in case a model rewrote the brackets.

Kinds (P0): PERSON, ORG, EMAIL, PHONE, ADDRESS, DOB, IBAN, CARD, OIB, JMBG,
ID_NUMBER, SSN, IP, PROFILE_URL, PLATE, plus whatever the caller names in
`extra`.

## Vault

* One vault per **case**, owned by the **person** (not the agent; on a shared
  agent's member turn, the member). Named; the tool defaults to one vault per
  conversation so conversations are never linked unless the agent names a case.
* Store: `anonymize.db` in Flight Deck's data dir. Each vault has its own
  Fernet key, sealed with the deck data key (`fd_crypto`); entries (original
  value, lookup key, surface forms) are encrypted with the vault key. The
  vault's name is sealed too (a case is easily named after a person) and found
  by an HMAC of the name with a random per-deck key. Deleting a vault deletes
  its key and entries.
* Every anonymize / restore / view / delete is logged (counts only, never
  values).
* `keep`: values never tokenized (the person's own company, public figures).

## Entity resolution

Mentions of one entity get one token: `Ivan Horvat`, `I. Horvat`, `g. Horvat`,
and Croatian inflected forms (`Ivana Horvata`, `Horvatu`). Deterministic rules:

* normalized keys per kind (emails lower-cased, phones to international digits,
  IBANs without spaces, names case-folded …);
* within a document, a found full name's surname (and first name, when only one
  person has it) is found again on its own, with Croatian case endings;
* every value already in the vault is searched in each new text, so once a name
  is known the code finds it everywhere — including names that no pattern
  would catch on their own;
* an email's local part (`ivan.horvat@…`) proposes the name it spells.

Restore puts back the canonical (first seen, full) form, so an inflected
mention restores in the nominative.

## Lanes

**Deterministic (P0).** No model, no network, no optional dependency — same
input, same output on every deck:

* validated: IBAN (mod-97), cards (`card_scrub`: Luhn + network), OIB (ISO 7064
  mod 11,10), JMBG (mod 11 + date);
* patterns: email, phone (international, national, labelled), IPv4/IPv6,
  profile URLs (LinkedIn, Facebook, Instagram, X, GitHub, Telegram, WhatsApp),
  Croatian plates, US SSN;
* labels (EN + HR): `Name:` / `Ime i prezime:`, `Address:` / `Adresa:`,
  `DOB` / `datum rođenja`, `Passport` / `osobna iskaznica`, `OIB:` …;
* context: honorifics (`Mr`, `Dr`, `g.`, `gđa`), greetings and sign-offs,
  email display names, company legal suffixes (`d.o.o.`, `GmbH`, `Ltd` …),
  street patterns (HR, EN, DE);
* column / key hints for tables and JSON (`email`, `ime`, `adresa` …);
* the vault's own values and the caller's `extra` terms.

Known gap without a name-recognition model: a bare name in free text with no
context ("we met Marko yesterday") — the result says so. An optional local NER
pack closes most of it (P4).

**Trusted LLM (P1).** The admin-trusted model gets the text with the
deterministic finds already replaced by tokens (lane `both`; it runs on a
snapshot, so the vault is untouched) or as it is (lane `llm`), in chunks of
~6,000 characters, and answers `{"spans": [{"text", "kind"}]}`. Each text must
be in the chunk character for character; anything else (invented, a token, a
stop word, over 200 characters) is dropped and counted. What's left becomes
terms the engine replaces everywhere (`Options.extra`), so the model can add
what to hide but never change the text or un-hide what code found — a document
that tries to instruct it can at worst make it find less. The terms join the
vault, so later documents catch them with code alone (with case endings).
Good at: bare names, descriptions that point to one person (kind `DETAIL`:
"the only female surgeon on Vis"). Gender for agreement (`⟦PERSON_3:f⟧`) is
left for later.

* **Which model.** `anonymize_trusted.py`. An admin sets it
  (`PUT /fd/anonymize/admin/trusted-model`), or the deck's environment does
  (`FD_ANONYMIZE_PROVIDER`, `_MODEL`, `_BASE_URL`, `_API_KEY`, `_ATTESTED=1`,
  `_ON_FAILURE`). Local is trusted as it is — `ollama` / `litert` at this
  machine or a private address (not an Ollama `-cloud` model), or an
  OpenAI-compatible server at a private address. Anything else needs
  `attested: true`. The key is sealed. The model is called directly, with
  errors logged by type only (a provider error can quote the request).
* **Lane.** `lane` on every run: `deterministic`, `llm` or `both`; left out,
  `both` when a trusted model is usable, else `deterministic`. Asking for the
  model without one → 409.
* **Failure.** One retry per chunk; then the lane has failed. Default
  `on_failure: refuse` → 502, nothing returned. `deterministic` → the code's
  result, marked `lanes.llm: "failed"` (lane `llm` always refuses: nothing to
  fall back on).
* **Cost.** Model tokens count against the person's daily allowance unless
  the model is local.
* **Checking it.** `POST /fd/anonymize/admin/trusted-model/test` runs the
  built-in sample cases (`anonymize/cases.json`, the same cases the tests hold
  every lane to) through code alone and code + model, and says what each left
  behind. `tests/test_anonymize_llm_lane.py::test_shared_cases_live` runs them
  against a real model (`CAPTAIN_LIVE_ANONYMIZE_MODEL=ollama/<model>` + `_BASE_URL`,
  or `anthropic/claude-haiku-5-5` + `_API_KEY`). Measured 2026-11-11: local
  `ollama/ornith:9b` 5/5 cases (13–51 s per case on a Mac), Claude Haiku 5.5
  5/5 (1–4 s per case). Models that think by default count it against the
  output cap, so the trusted call allows 16,000 tokens.

## Inputs and outputs

* P0: inline text; files — txt, md, log, html, xml, csv/tsv (cell by cell with
  column hints, written back as csv), json (string values with key hints,
  written back as json); docx / xlsx / pptx / pdf are read as markdown and
  written as `.md` (formatting and metadata are not carried over — which also
  drops author, comments and tracked changes); URLs (fetched inside the tool).
* The output file name is tokenized too (`Ivan_Horvat_CV.pdf` →
  `⟦PERSON_1⟧_CV.anonymized.md`, spelled filesystem-safe).
* **P2 — office files in place** (`anonymize/office.py`). A `.docx`, `.xlsx` or
  `.pptx` comes back as the same kind of file: the OOXML parts are edited inside
  the zip (lxml, entities off, no network), so formatting, pictures and charts
  stay byte for byte (re-saving with python-docx / openpyxl would drop charts
  and images). The engine reports each replacement in the original text's
  positions (`RunResult.edits`), and the token goes into the run where the span
  starts — a name split across a bold and a plain run keeps the bold.
  * Word: body, tables, headers, footers, footnotes, endnotes, comments, text
    boxes, **tracked deletions** (`w:delText`), field codes (`HYPERLINK
    "mailto:…"`), external link targets; every `author` / `initials` blanked.
  * Excel: shared and inline strings, cached formula results, formulas, long
    numbers (an OIB stored as a number becomes a text cell), comments,
    threaded comments, sheet names (formulas follow the rename), text boxes,
    chart labels; column headers work as hints, as in a CSV.
  * PowerPoint: slides, notes, layouts, masters, comments, charts, links.
  * Every kind: author and "last modified by" emptied; title, subject,
    keywords, company … anonymized or emptied.
  * Word and PowerPoint text is read **per part as one text** (paragraphs by a
    newline, cells of one table row by a tab), so "Lijep pozdrav," reaches the
    name on the next line and "Ime:" the name in the next cell; edits are split
    back per paragraph.
  * Pictures and embedded files are left as they are — the result says how
    many. PDF and web pages come back as markdown — true PDF redaction is
    skipped (decided 2026-11-11; it would need PyMuPDF, which isn't a
    dependency).
  * Routes `/fd/anonymize/file` and `/file/restore` (and `/agent/…`): the file
    as base64 in, the same kind of file out, plus a text preview. The agent
    tool sends a `path` file this way.

Result (both lanes):

```
{vault, vault_name, mode, output (text | path), entities [{token, kind, count, new}],
 lanes {deterministic: ok, llm: ok|skipped|failed}, verify {ok, fixed},
 warnings [...], already_exposed}
```

Entities never carry original values toward the model.

## Surfaces

* **Agent tool `anonymize`** — always registered (clear error without a deck).
  Actions: `anonymize` (text | path | url, vault, mode, keep, extra, output),
  `restore` (path | text, vault — writes the restored file to `saved/`, returns
  the path only; refused on automated turns), `vaults` (names and counts).
* **Flight Deck routes** `/fd/anonymize/*` — agent routes (`/agent/run`,
  `/agent/restore`, `/agent/vaults`) resolve the person from the deck's own
  records (owner, or the member on a member's turn); person routes (`/run`,
  `/restore`, `/vaults`, `/vaults/{id}` with originals, rename / expiry,
  delete, `/vaults/{id}/events`).
* **Anonymize page** (P2) — in the kiosk (Settings → What they keep →
  Anonymize) and the full layout (Files → Anonymize): paste text or drop a
  file, choose the vault (or keep nothing) and who looks (code, code + trusted
  model, model only); the result shows tokens as chips, the warnings, and a
  download. "Put back" restores a text or a file from a vault. "Vaults" lists
  them; one opens in a side sheet with its real values (only on request, and
  logged), rename, an end date, delete, and its history.
* **Admin console → Settings → Anonymization** (P2) — the trusted model:
  provider, model, address, key (never shown again), the attestation for a
  non-local model, refuse / code-only on failure, on/off, and "Check on
  samples" (the built-in cases, code alone vs code + model).

## Leak surfaces to keep closed

* Logs, tool audit, error messages: never original values (request bodies are
  read by hand so a validation error can't echo them).
* External / deep memory mirroring, Line notifications: carry tokens only.
* Restored content never goes back into the model's context.
* Tests use a shared fixture both lanes are held to; the deterministic lane
  must catch every case of its kinds.

## P0 — what is built

* `captain_claw/anonymize/` — `detectors.py` (deterministic lane),
  `engine.py` (select → resolve → known values → tokens → rounds → verify,
  restore, file names), `formats.py` (text, csv/tsv, json, jsonl).
* `flight_deck/anonymize_vault.py` (store), `flight_deck/anonymize_routes.py`
  (routes), wired in `server.py`.
* `tools/anonymize.py` — always registered (like `public_link`), eco-mode
  keywords EN/HR, `deck_confine` rules (`path` read, `output_path` write, `url`
  as an outbound address with the private-network check, redirects included),
  risk class `write`.
* Tests: `tests/test_anonymize_engine.py`, `tests/test_flight_deck/test_anonymize.py`
  (routes and the tool end to end through the real routes), the shared cases
  `captain_claw/anonymize/cases.json` (moved into the package in P1 so the
  deck's admin check can use them).
* Speed: ~3 s per MB of text; known values are found through word indexes, so
  a vault of thousands of people doesn't slow a run down.

Known limits of P0 (the deterministic lane):

* A bare name with nothing around it ("we met Marko") is missed unless the
  vault already knows it or the caller passes it in `extra`.
* Stems can join two people (Ivo / Iva Horvat become one token) and miss a
  vowel-dropping form (Petar / Petra become two). Joining is privacy-safe;
  the LLM lane should resolve both.
* Restore puts back the canonical (nominative) form, so an inflected sentence
  can come back grammatically off.
* PDF and web pages come back as markdown; pictures inside office files are
  not changed (the result says how many there are).
* In a spreadsheet, a first name and a surname in two cells get two tokens.
* Not yet: Iskra beings without fleet access (same as `public_link`).
  Member turns of shared agents came later (below).

## P3 — the privacy gate

Opt-in, per person (Anonymize page → Privacy gate; `GET/PUT /fd/anonymize/gate`,
kept in their settings). Nothing changes for anyone who doesn't turn it on.

* **What it covers** — sources the person ticks (`anonymize/gate.py`): mail
  (Gmail, Outlook, IMAP tools), chats (Telegram inbox, WhatsApp), calendar,
  contacts, files (read, grep, the extract tools, OCR, knowledge, Drive,
  OneDrive — optionally only some local / VFS folders), data (datastore),
  memory (external and deep memory).
* **When** — `untrusted` (default): only for an agent whose current model
  isn't trusted (not local, not the admin's trusted model); `always`: every
  agent. Flight Deck decides (`POST /fd/anonymize/agent/gate` with the agent's
  provider / model / address; a shared-agent member's turn uses the member's
  setting).
* **Results in** — `AgentGuardMixin._execute_tool_with_guard`, the one path
  every tool call takes: a covered tool's result (content and error) is
  anonymized into the person's **"Privacy gate"** vault before it is returned
  — before the model, the session history, compaction, insights or anything
  derived from them sees it — with a one-line note that the tokens are put
  back on sending. If the deck can't check it, the result isn't shown (fail
  closed). Who looks: code (default, fast) or code + trusted model.
* **Messages out** — for mail and WhatsApp tools, tokens in the arguments
  the agent wrote are put back just before the tool runs (the recipient reads
  real names); the tool's own result is then anonymized again. If the values
  can't be put back, nothing is sent.
* **Recall** — background notes are built without waiting on the deck, so the
  agent asks for the gate's answer at the start of each turn, and while the
  gate covers a source its notes stay out: raw workspace files from the
  semantic recall (session chunks stay: they are the gated conversation),
  knowledge pointers and the workspace file list (files), contacts, data,
  briefings (mail / calendar), deep and external memory (memory).
* **Reading replies** — chat replies with tokens get a **Real names** toggle
  (`POST /fd/anonymize/reveal`, the gate's vault, logged); while the gate
  applies, the `anonymize` tool's default vault is the gate's too, so all
  tokens in a chat mean the same.
* **Verified live** with an agent on Claude Haiku 5.5 reading a notes file
  with made-up people: its reply and its session store held only tokens,
  "Real names" showed the reply properly, and in a fresh topic it could not
  recall the client without its tools.
* **Limits** — conversations from before the gate was turned on still hold
  real values (start a new topic); tools outside the covered sources (web,
  shell, browser) are not gated; the gate's answer is cached for a minute, so
  turning it on reaches a running agent within a minute.

## P4 — coarser values and the name-recognition pack

**Coarser values (one way).** `generalize` (`true` or a list of `DOB`,
`ADDRESS`, `AGE`, `AMOUNT`) with `mode: one_way` only — a coarse value can't
be put back. The engine runs and verifies as usual; then
`engine.generalize` swaps those tokens for values made from each
occurrence's own text: date of birth → year, address → its city (the token
stays when there is no postal code + city), age → decade band
("40–49 godina"), money amount → a 1–2–5 band in the amount's own number
format ("4.237,50 EUR" → "2.000–5.000 EUR", "$87,500.00" → "$50,000–100,000").
Ages and amounts are found only when asked for (`detectors.OPT_IN_KINDS`):
by themselves they aren't private. Office files get the coarse value where
the token would have gone (it works on the edits). Page: "Keep nothing" →
"Make them coarser"; tool: `generalize: true`.

**Name-recognition pack** (`flight_deck/anonymize_ner.py`). A local
token-classification model adds people and companies it finds (score ≥ 0.70;
places and the rest ignored) to the deterministic lane on this deck, in
~1,500-character chunks, one call at a time; a model that fails finds nothing
and code carries on. It reads the original text only (the first round): on
the engine's own masked output it took the words around a token for names.
Admin console → Settings → Name recognition: on/off, the model — only
`classla/bcms-bertic-ner` (default; Croatian / Bosnian / Serbian /
Montenegrin, English names too, ~440 MB; people and companies) or
`Davlan/xlm-roberta-base-wikiann-ner` (other languages, ~1.1 GB; people
only) — and "Download and start". Only `safetensors` weights are downloaded and loaded, from the local
files, with no remote code. Needs `transformers` + `torch` where Flight Deck
runs (not the slim desktop app or a Pi): the card says so. Started at deck
start when it is on and downloaded; never downloads by itself.

**Measured 2026-11-12** (`scripts/measure_name_models.py` on
`anonymize/name_cases.json`: 87 sentences, Croatian and English — bare and
inflected names, names that are also words, small companies, and 144 traps:
days, holidays, saints, streets and places named after people, ordinary
words; plus the five shared cases). Mac, CPU, end to end through the engine:

| | people (97) | companies (9) | traps kept (144) | memory | speed |
|---|---|---|---|---|---|
| code alone | 2% | 3 | 100% | — | — |
| BERTić, cut-off 0.70 | **91%** (hr 95%, en 82%) | **9** | **142** | ~0.8 GB | ~17 ms/sentence, ~70 ms/KB, loads in ~3 s |
| XLM-R, all labels, 0.70 | 88% (hr 84%, en 94%) | 8 | 124 | ~1.25 GB | similar |
| XLM-R, people only, 0.70 | 78% | 3 | 141 | ~1.25 GB | similar |

* BERTić is the default. Its false hits don't grow as the cut-off drops
  (0.5–0.9 all keep 142/144), so 0.80 → 0.70 (0.5 finds one name more).
* XLM-R labels days, holidays and whole phrases as companies at 0.9+
  ("Monday", "Christmas", "U zoru smo krenuli"), so only its people are
  taken; it stays for other languages.
* What BERTić still misses: names that are also words at a sentence start
  (Zora, Vjera, Ruža; Will, Grant, Bill), "Kevin's", "Summer Lee" (it calls
  those MISC). What it breaks: "Sveti Duje" (a saint — public), and "Mark the
  date" once "Mark Jensen" is known (the engine follows a known first name).
* No shared case broke and no run failed verification with either model.
* `tests/test_anonymize_names.py` runs the default model on the set where it
  is downloaded (skipped elsewhere) and holds it to people ≥ 85%,
  companies ≥ 80%, traps ≥ 97%.

The trusted-model card's "Check on samples" includes the name model in its
"code alone" column when one is on.

Also fixed in P4: an "Address:" label took the rest of its line — it now
stops at the first sentence end after the house number ("Trg b. J. Jelačića 1"
keeps its initials).

## Members of a shared agent

A process member (A2) can use the tool on their own turn; a docker member
stays at chat only (A1), as with every file tool.

* `anonymize` is on the member's widest allowlist with the actions
  `anonymize`, `restore` and `vaults` only. `path` is read and `output_path`
  written under the member's own path rules, like `read` and `write`; `url`
  must be a public address and is fetched the member's way (public only,
  checked again when it connects).
* The tool sends the member's grant to Flight Deck, so the work lands in the
  **member's** vaults, never the owner's. Without the grant (a background step,
  another person's message) it refuses: "Anonymization is only available while
  answering this member's own message."
* Results go to the member's own files: `vfs:<project>/anonymize/<folder>/<name>`.

## P5 — files where they are (VFS file, VFS folder, Library file)

`flight_deck/anonymize_targets.py`. The copy goes **next to the original** (decided 2026-11-12):
`Contract.anonymized.docx` beside the file, `Clients_anonymized/` beside the folder (a whole folder becomes
a new folder of the person's own: `Clients_anonymized`, then `-2` …). Never over an earlier copy. When the
person can't write there — a Drive or OneDrive mount, a read-only link, a view-only share, an `@alias`
folder — the copy goes to their own **Anonymized** folder under the source folder's name.

* Names in the copied paths are tokenized too, into the same vault (`Ivan Horvat/` → `PERSON_1/`). They
  are named **after** the files ran (`anonymize_routes.anonymize_names`): code alone can't tell a bare
  "Ivan Horvat" is a name; the files taught the vault.
* A folder is a background job, one per person: phase 1 anonymizes every file into a scratch folder
  outside the VFS, phase 2 names the folders and places the copies (so no raw name is ever written into
  the copy). Stop places what was finished. Up to 2,000 files; pictures, archives and other kinds,
  files over 25 MB, and earlier anonymized copies are left out and listed; a file that fails is listed
  and gets no copy.
* `one_way` keeps one in-memory state for the whole job (`one_way_state`), so a person keeps one token
  across the copied folder and folder names can still be found.
* The default vault is the folder's name (a Library file: its agent's name).
* Routes: `POST /fd/anonymize/vfs/file`, `POST /vfs/folder`, `GET /vfs/job`, `POST /vfs/job/cancel`,
  `POST /library/{file_id}` (the copy beside it in the agent's workspace, listed in the Library at once);
  for agents `POST /agent/vfs/folder`, `GET /agent/vfs/job` (the agent's person: owner or member).
* The agent tool: `path: vfs:<folder>/<sub>/` runs the folder job and waits up to 200 s (the tool's
  limit); a longer job carries on and the tool says so.
* UI: Shared folders → a file's panel "Anonymize"; a folder's "…" menu "Anonymize the whole folder" /
  "Anonymize this folder"; every subfolder row's "…" "Anonymize" (also in read-only folders). File
  library → a file's panel "Anonymize". One dialog (`AnonymizeHereDialog`): vault or keep nothing,
  coarser, words to leave, and a folder's progress (it keeps going when closed). The full layout's VFS
  browser doesn't have it yet.

## P6 — the Chrome extension

Captain for Chrome 0.3.0: **Anonymize this page** (or **the selection**, when one is picked) under the Send
button, and **Anonymize with Captain** on the right-click menu (page or selection). No agent is involved.

* The vault is named after the page's host as the browser shows it (`www.example.com`, `www.` kept): the
  same person gets the same token on every page of the site.
* The extension can only reach `/fd/clip/*` (origin guard) with its paired token, so Flight Deck has
  `POST /fd/clip/anonymize` and `/fd/clip/anonymize/restore` (`get_clip_user`) standing in for
  `run_for` / `restore_for`. The page goes as the markdown the panel already read (`capture.js`).
* The panel shows what was found, the result with its tokens as chips, **Copy**, **Save in Flight Deck**
  (Web clips, `<day>-<site>-<tokenized title>-anonymized/` — the folder is named from the tokenized title,
  since a page title often carries the name), and **Put the real names back** for text that has tokens in
  it (an answer someone got for the anonymized text).
* Verified end to end: headless Chromium with the unpacked extension against a throwaway deck — page,
  save, put back, selection, right-click.

## P7 — the whole chat

Decided 2026-11-12: a **coverage mode of the privacy gate** — "What it covers: sources I pick / the whole
chat" — with one vault, **"Chat"**, for both (it replaces "Privacy gate"), keeping "untrusted model / every
agent" and "code / code + trusted model". **Pictures pass through** as they are. **Anyone else** talking to
the agent sees tokens — never restored.

**In** (the agent; `anonymize_gate_client.py`):

* The person's message at `Agent.complete` / `Agent.stream`, before anything reads it — the rephrase, the
  guards, the history, the model (`Agent._gate_user_text`; refused, not sent, when it can't be checked).
* Every tool's result (`_execute_tool_with_guard`: `covers()` is true for every tool) — except calls whose
  result goes to Flight Deck (`autonomous-action`, `flow_step`, `custom_tool:*`).
* Everything else just before each model call (`_gate_outgoing` at the start of `_complete_with_guards`,
  before the input guard's own model call; also the salvage and the no-tools streaming calls): the turn's
  memory and recall notes (a user-role message), cron notes and mirrored results, compaction summaries,
  `/btw` — **not the system prompt** (decided 2026-11-12: it is the agent's own setup, and reading it
  through a local trusted model on every change cost minutes per turn; what it says about the owner — the
  profile, the members — reaches the model as written) — in **one**
  Flight Deck call (the texts as a JSON list). Each text is anonymized once and remembered (sha256 → result),
  so the same text always reads the same to the model (its prompt cache holds), and the session's history is
  rewritten to the tokens (nothing raw is left to derive notes or summaries from). The system prompt says
  once that tokens are on purpose (`CHAT_NOTE`); results no longer carry the per-result note. A failed check
  sends nothing (`GuardBlockedError("privacy")`, shown as the turn's reply).

**Out**:

* Every tool's arguments get the real values back before it runs (files the agent writes, searches it runs,
  mail it sends) — except `anonymize`, whose job is tokens. The result is anonymized again.
* Flight Deck restores the frames it relays to the person (`flight_deck/anonymize_chat.py`, `ChatRestorer`):
  the owner's socket to their own agent (`agent_ws_proxy`; never an admin looking at someone else's agent)
  and a member's socket to a shared agent (the member's vault). Every string in a frame is restored, except
  streamed deltas (`response_stream`: a token can be split; the chat doesn't draw them). The vault is
  cached until it changes. The agent's frames escape non-ASCII, so the check looks for `\u27e6` as well.
  Restoring never fails the person: tokens are shown instead.
* Telegram: the owner's own private chat (reply and live narration) — `_tg_restorer`; other users and
  groups keep the tokens.

**While it works** (added 2026-11-12, after a real deck's trusted model took minutes):

* The live status line says so — "🛡️ Anonymizing your message / what web_fetch returned / 6 new texts before
  the model reads them — the trusted model (ollama/ornith:9b) is looking for private data…" and "🛡️ Putting
  the real values back into send_mail…" — instead of "thinking". Only when Flight Deck is actually asked
  (cached texts say nothing), and never for housekeeping after the turn (reflection, insights: the agent
  already said it is idle). The decision carries the trusted model's name (`model`).
* Caps: the agent waits up to **30 minutes** for one check (`CLAW_ANONYMIZE_GATE_TIMEOUT`, was 180 s — a
  local model reading the system prompt chunk by chunk ran past it); one trusted-model call up to
  **10 minutes** (`FD_ANONYMIZE_MODEL_TIMEOUT`, was 240 s, now also the provider's own request timeout,
  which was 120–180 s); the `anonymize` tool 30 minutes. A failure names what happened ("it took longer
  than 30 minutes", "Flight Deck couldn't be reached") instead of an empty "()".
* The answer limit: **32,000** tokens (`FD_ANONYMIZE_MODEL_MAX_TOKENS`, was 16,000). The trusted model's
  **thinking is off** by default on every provider (`FD_ANONYMIZE_MODEL_THINK`: `off` default,
  `low`/`medium`/`high`, `on`, or `model` for the model's own) — through the reasoning-effort plans
  (`llm/reasoning.py`: `chat_template_kwargs.enable_thinking` for a Qwen on a local OpenAI-compatible
  server, `thinking` for Claude, `reasoning_effort` for OpenAI) and as Ollama's own `think` for a model
  they don't know — a local model that thought without end ran to the limit twice on
  a ~1,000-token chunk (16,000, then 32,000 tokens). An answer cut at the limit (`finish_reason` "length")
  is asked again without thinking (when thinking was on), then in halves, twice at most, then the lane
  fails saying so.
* The trusted model's answers are the private data it found: its provider no longer quotes model output in
  log lines (`LLMProvider.log_content = False`; the Ollama provider logged a 200-character preview at INFO).

**Checked on the staging deck (glasses), 2026-11-12**, and fixed:

* **The model must name things by their tokens.** Claude Haiku summarized a tokenized agreement as "a
  Croatian company", "a Luxembourg fund" — no tokens, so nothing to put back. A model that takes ⟦…⟧ for
  redactions writes around them. The note now says plainly they are not redactions and every token is
  replaced before the person reads the reply ("write its token — '⟦ORG_2⟧ signs as the Provider', not 'a
  Croatian company signs'"), and a one-line reminder (`CHAT_REMINDER`) follows every user message and
  tool result that has tokens — where the model reads them, not only at the end of a long system prompt
  (the same text every time: the prompt cache holds). Verified with Haiku on a throwaway deck: the summary
  named the parties by token ("between ⟦ORG_1⟧ (the Participant) and ⟦ORG_2⟧ (the Provider)") and the
  chat showed the real names.
* **Flight Deck's borrowed calls were not gated.** Consciousness, the arbiter, plans, course progress,
  dispatch, games, WhatsApp reactions and auto-replies borrow the agent's model through its
  `/api/llm/complete`, which called the provider directly — glasses' log showed Haiku writing real
  names of the owner's contacts from them. The endpoint now anonymizes what they send (not their system prompt),
  adds the note, restores the answer before Flight Deck shows or sends it, and refuses (503) when the check
  can't run. Verified with Haiku: the model wrote "Unread note from ⟦PERSON_6⟧ (⟦EMAIL_5⟧)…", Flight Deck
  got the made-up test name and email back in their place.
* **Names joined by underscores** (a saved file's name: "Acme_-_Supply_Agreement.docx",
  "Ivan_Horvat_CV.pdf") are found like anywhere else: the known-value finder treats `_` as a separator.
* **Task naming went around the gate.** The chat names each task with its own small call to the agent's
  model (litellm, outside the guarded path) — with the person's raw message, and for "continue" the three
  messages before it. With the whole chat gated it now sends the message as the turn's model sees it (the
  turn and the name share one anonymize call), earlier messages only when already anonymized, and the name
  comes back with the real values; a message that can't be checked gets no name, never the raw text. On a
  shared agent the name is made as the member (their gate, their vault), and the clean-text cache is kept
  per person, so a text cleaned in the owner's vault is never reused for a member. Verified on the throwaway
  deck: the naming call carried the tokenized message (91 characters, the raw one is 119).
* **A private company's product went through.** A deal screener fetched a startup's site: its address became
  a token, but the product name on its pages stayed — the trusted model was told not to mark product names
  (written for personal privacy), and the screener's model then searched, wrote and replied with it. The
  trusted model now marks a private company's own products, apps, services and brands as ORG (one word, odd
  capitals and made-up spellings too), and still leaves generic kinds of product and widely known brands
  (Excel, WhatsApp). A shared case with a made-up startup checks it: with Claude Haiku the old prompt left the
  product name in 3 of 3 runs, the new one hid it in 3 of 3, with the other cases unchanged.
* **A company's name in links.** With products hidden, the screened company still showed through its
  links: registry and search pages put its name in the path ("…/tvrtka/vrtlog-labs-d-o-o/", "…/place/Vrtlog+Labs/"),
  its own site in theme folders ("/themes/vrtloglabs/") and product paths ("/programi/lumora-desk/") — and the
  model fetched one. Code found a known multi-word name only as written, with spaces. Now the known-value finder
  also finds a known company's, product's or person's full name spelled as links and file names write it —
  joined by "-", "_", "+" or "%20", or run together ("vrtloglabs", six letters or more) — and a one-word company
  inside a link. Each spelling is an entity of its own: restoring puts back exactly what was there, so a link
  still opens and a file is still found (a slug that took the name's own token would come back as "Vrtlog Labs",
  with a space). The model sees a different token for the slug than for the name. Verified with Claude Haiku as
  the trusted model on a made-up company page: nothing raw left in 3 of 3 runs, every link restored exactly.
* **A company's registry number.** "MB: 01234567" (matični broj) stayed while the OIB next to it was a token.
  The detectors now read the label "MB" (capitals; "512 MB" is a size) with 7–8 digits as an ID number, next to
  the "MBS" and "matični broj" labels they already knew.
* Still open: a bare company name the vault doesn't know yet ("Acme - Supply Agreement…" in the first message)
  needs the trusted model, and even that can miss one (a startup's name in a short meeting title went through
  with lane "both"); map links keep a place's coordinates ("…/@45.8,15.9"); Luxembourg forms (SCSp, S.à r.l.) and registry numbers (B plus six digits) aren't known
  to code; Flight Deck's own `/fd/llm/complete` (one-shot generations on the owner's tier) isn't gated.

**Not covered yet** (tokens stay): WhatsApp, Messenger and devices (the channel bus can carry other allowed
people), cron deliveries, public pages, the Line and notifications.

**Verified live** (throwaway deck, local `ornith:9b` agent, gate on "every agent", whole chat): the agent
stored and its model read "Name: ⟦PERSON_1⟧, OIB ⟦OIB_1⟧, mail ⟦EMAIL_1⟧" and replied in tokens; the chat
showed "Your name is Ivan Horvat and your email is ivan.horvat@acme.hr". Asked to write a file, the model
wrote tokens in the call and `contact.txt` got the real values. No raw value anywhere in the agent's
sessions, memory, insights or topics — only the tool-call audit (what actually ran) and the memory's index
of the workspace file (recall goes through the gate).

## Phases

* **P0** — engine (detect → resolve → tokens → replace → verify → restore),
  deterministic lane, encrypted vault store, deck routes (agent + person), the
  agent tool, `one_way` mode, text/csv/json + markdown from office/pdf, tests.
* **P1** (built) — trusted LLM lane: trusted-model setting (admin API +
  environment; local trusted by default, cloud only attested), span-JSON
  prompt, `lane: deterministic | llm | both`, fail-closed handling, admin check
  and live test on the shared cases. No page yet — the admin card for the
  trusted model comes with P2's pages.
* **P2** (built) — office files in place with metadata cleaned, file routes,
  the tool sending files whole, the Anonymize page (kiosk + full layout), the
  admin card for the trusted model with its sample check. PDF redaction and
  OCR of pictures are not built.
* **P3** (built) — the privacy gate, below.
* **P4** (built) — below.
* **Members** (built) — shared-agent members on their own turn, above.
* **P5** (built) — files where they are, above.
* **P6** (built) — the Chrome extension, above.
* **P7** (built) — the whole chat, above.
* **Skipped** — true PDF redaction (PDFs come back as markdown).
