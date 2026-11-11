# Anonymization — plan

A Flight Deck tool, available to every agent and to the person, that finds
private data (names, dates of birth, bank accounts, emails, addresses, phone
numbers, ID numbers, companies …) in a text, file or page and replaces it with
tokens that mean nothing outside Captain. The real values are kept in an
encrypted **vault** in Flight Deck, so the person (never the model) can put
them back later.

Status: **P0 built** (branch `feat/anonymize-p0`); P1 (trusted LLM lane) next.

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

**Trusted LLM (P1).** The admin-trusted provider gets the text with the
deterministic findings already marked and returns additional spans as JSON
(`[{text, kind, occurrence}]`). Our code finds each quoted span in the text
(exact match only; a quote that isn't in the text is dropped and counted),
then the shared engine continues. Good at: bare names, coreference, quasi-
identifiers ("the CEO of the only bakery in Vis"), gender for grammatical
agreement (`⟦PERSON_3:f⟧`, optional).

## Inputs and outputs

* P0: inline text; files — txt, md, log, html, xml, csv/tsv (cell by cell with
  column hints, written back as csv), json (string values with key hints,
  written back as json); docx / xlsx / pptx / pdf are read as markdown and
  written as `.md` (formatting and metadata are not carried over — which also
  drops author, comments and tracked changes); URLs (fetched inside the tool).
* The output file name is tokenized too (`Ivan_Horvat_CV.pdf` →
  `⟦PERSON_1⟧_CV.anonymized.md`, spelled filesystem-safe).
* P2: write docx / xlsx / pptx back in their own format, strip their metadata,
  true PDF redaction, image OCR (with a warning that the picture itself is not
  changed).

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
* **Kiosk page** (P2) — paste or drop a file → anonymized copy; vault browser;
  restore.

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
  (routes and the tool end to end through the real routes), the shared fixture
  `tests/fixtures/anonymize_cases.json`.
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
* Office / PDF files come back as markdown.
* Not yet: the kiosk page (P2), member turns of shared agents (the routes
  already file a member's work in the member's vault; the tool isn't on the
  member allowlist yet), Iskra beings without fleet access (same as
  `public_link`).

## Phases

* **P0** — engine (detect → resolve → tokens → replace → verify → restore),
  deterministic lane, encrypted vault store, deck routes (agent + person), the
  agent tool, `one_way` mode, text/csv/json + markdown from office/pdf, tests.
* **P1** — trusted LLM lane: admin trust flag on providers/endpoints (local
  trusted by default), span-JSON prompt, `lane: deterministic | llm | both`,
  fail-closed handling, recall measured against the shared fixture.
* **P2** — formats kept (docx/xlsx/pptx writers, metadata strip, PDF
  redaction), file-name and metadata leaks, kiosk page.
* **P3** — privacy gate: per source (an inbox, a data table, a VFS folder)
  "always anonymize before an agent on an untrusted model sees it"; restore on
  the way out to the person (mail draft, delivered file).
* **P4** — optional local NER pack (GLiNER / spaCy, a download — not in the
  slim app or on the Pi by default); `one_way` generalization of
  quasi-identifiers (DOB → year, address → city, amounts → ranges).
