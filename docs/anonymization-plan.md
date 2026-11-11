# Anonymization — plan

A Flight Deck tool, available to every agent and to the person, that finds
private data (names, dates of birth, bank accounts, emails, addresses, phone
numbers, ID numbers, companies …) in a text, file or page and replaces it with
tokens that mean nothing outside Captain. The real values are kept in an
encrypted **vault** in Flight Deck, so the person (never the model) can put
them back later.

Status: **P0** PR #511 · **P1** PR #512 · **P2** PR #514 · **P3** PR #515 · **P4** PR #516 (each stacked on the
one before) · **members + name models measured** (branch `feat/anonymize-members`, stacked on P4; BERTić is
the default name model). True PDF redaction is skipped (decided 2026-11-11).

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
* **Skipped** — true PDF redaction (PDFs come back as markdown).
