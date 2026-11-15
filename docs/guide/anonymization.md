# Anonymization — private data out, the real names back for you

[← README](../../README.md) · [All guides](README.md)

New in 0.9.1. Captain takes names, companies, addresses, emails, phone numbers, bank accounts and ID numbers out of a text, a file, a whole folder or a web page. Each one becomes a token like `⟦PERSON_1⟧`. The real values stay on your deck, encrypted, in a vault that only you can open. You can send the anonymized version anywhere, and put the real names back into whatever comes back.

Your agents can do it too, and the **privacy gate** can make it automatic: what your agents read reaches their model as tokens, while your own chat still shows the real names.

**On this page:** [What it does](#what-it-does) · [Where to find it](#where-to-find-it) · [Set it up (admins)](#set-it-up-admins) · [Anonymize a text or a file](#anonymize-a-text-or-a-file) · [Put the real names back](#put-the-real-names-back) · [Files and folders where they are](#files-and-folders-where-they-are) · [The privacy gate](#the-privacy-gate) · [Asking an agent](#asking-an-agent) · [Captain for Chrome](#captain-for-chrome) · [Vaults](#vaults) · [Limits and tips](#limits-and-tips)

## What it does

**Tokens.** Every private value becomes a token that names its kind and a number:

```text
Ana Horvat (ana.horvat@example.com) signed for Velora d.o.o. in Zagreb on 3 March 2026.
Ms. Horvat will send the annex to Luka Perić by Friday. Amount: 4.200,00 EUR.
See https://www.example.com/clients/velora-doo for the file.
```

becomes, with code alone looking:

```text
⟦PERSON_1⟧ (⟦EMAIL_1⟧) signed for ⟦ORG_1⟧ in Zagreb on 3 March 2026.
Ms. ⟦PERSON_1:2⟧ will send the annex to Luka Perić by Friday. Amount: 4.200,00 EUR.
See https://www.example.com/clients/⟦ORG_1:2⟧-doo for the file.
```

- **The same party keeps one number.** `⟦PERSON_1:2⟧` is Ana Horvat again, written another way ("Horvat"). `⟦ORG_1:2⟧` is Velora again, as the link spells it. The part after the colon only says which spelling to put back, so a link still opens and a case ending comes back as it was written.
- **The same vault gives the same token.** Ana Horvat is `⟦PERSON_1⟧` in every document you anonymize into the same vault. Different vaults reuse the numbers, so tokens never link two cases.
- **Tokens are assigned, never worked out from the value.** Nobody can turn `⟦EMAIL_1⟧` back into the address without your vault.
- **Titles, cities, dates and amounts stay.** "Ms." stays plain text. So do Zagreb, the date and the amount: the text stays readable, and the numbers can still be added up.
- **Luka Perić was missed.** A name with nothing around it needs a model to find it: the trusted model or the name pack, both below. This is why you check the result.

**What it finds.**

| Kind | Token | How |
|---|---|---|
| People's names | `PERSON` | Labels ("Name:", "Ime i prezime:"), titles, greetings and sign-offs, email display names, names your vault already knows, the name pack, the trusted model |
| Companies and organisations | `ORG` | Legal forms (d.o.o., d.d., GmbH, Ltd…, also written out), known names, the trusted model (also a small company's own products and brands) |
| Email addresses, phone numbers | `EMAIL`, `PHONE` | Patterns; a phone number is also found by its label |
| Street addresses | `ADDRESS` | Labels and street shapes (Croatian, English, German) |
| Dates of birth | `DOB` | By their label ("Date of birth", "datum rođenja"); other dates stay |
| Bank accounts, cards | `IBAN`, `CARD`, `CVV`, `CARD_EXPIRY` | Check digits |
| OIB, JMBG | `OIB`, `JMBG` | Check digits, or their label |
| ID, passport and tax numbers | `ID_NUMBER` | Labels, also a company's registry number ("MB: …", "MBS", "matični broj") |
| US social security numbers, IP addresses, social profile links, licence plates | `SSN`, `IP`, `PROFILE_URL`, `PLATE` | Patterns (plates: Croatian) |
| A description that points to one person ("the only surgeon on the island") | `DETAIL` | The trusted model only |
| Anything you list yourself | your own kind, e.g. `PROJECT` | **Always hide** on the page, `extra` for the tool |

Office files are covered inside, not only their visible text. In Word that means headers, footers, footnotes, comments, text boxes and tracked deletions. In Excel it means cells (an OIB or a phone number stored as a number too), formulas, sheet names and comments. In PowerPoint it means slides, notes and masters. In every file the author and "last modified by" are emptied.

**Two kinds of result.**

- **In a vault** (reversible). The real values go into a vault, so you can put them back later. In GDPR terms this is *pseudonymized*: still personal data, because it can be reversed.
- **Keep nothing** (one-way). No vault: the tokens can never be turned back. Use it to share data for good. Here you can also **make them coarser**: date of birth → year, address → city, ages and amounts → ranges. An exact age or salary can point to one person even without a name.

```text
Name: Ana Horvat                                   Name: ⟦PERSON_1⟧
Date of birth: 14.05.1987                          Date of birth: 1987
Address: Lipova 7, 10000 Zagreb           →        Address: Zagreb
She is 38 years old and earns 4.237,50 EUR.        She is 30–39 years old and earns 2.000–5.000 EUR.
```

**Who looks: two lanes.**

- **Code** always looks. It finds patterns, check digits, labels, column names and everything your vault already knows. It needs no model and gives the same result on every deck. On its own, it can miss a name with nothing around it ("we met Luka").
- **The trusted model** also looks, when an admin has set one up. It reads the text and lists what to hide, and code does the replacing. The model never rewrites your text. It catches bare names and descriptions that point to one person.
- **Code + trusted model** is the default whenever a trusted model is set up. The model can only add things to hide, never un-hide what code found.
- An admin can also turn on the **name pack**: a small local name-recognition model that helps code find names in plain sentences. Nothing leaves the deck.

After replacing, Captain checks its own output. Anything still found is replaced in another round. If something is still there after the last round, the run fails and nothing is returned.

## Where to find it

| You want to | Go to |
|---|---|
| Anonymize a text or a file, put real names back, see your vaults, turn on the privacy gate | **Anonymize**: in the kiosk, **Settings → What they keep → Anonymize**, or **More → Anonymize** under the agent column. In the full layout, **Files → Anonymize** in the sidebar. |
| Anonymize a file or a folder where it is | Kiosk: **Shared folders** or the **File library**, then the file's or folder's **Anonymize** |
| Set the trusted model or the name pack (admins) | **Admin console → Settings**: the **Anonymization** and **Name recognition** cards |
| Read an agent's reply with the real names | **Real names** under the reply |
| Anonymize a web page | Captain for Chrome: **Anonymize this page**, or right-click → **Anonymize with Captain** |

The Anonymize page has four tabs: **Anonymize**, **Put back**, **Vaults** and **Privacy gate**. In the kiosk it opens as a dialog, and Escape closes it.

**Data rooms**, next to Anonymize in the same menus, use the same engine to share a whole data room with Claude. They have a guide of their own: [Data rooms](data-rooms.md).

## Set it up (admins)

Code-only anonymization works on every deck with no setup. The two cards below add the model lanes.

### The trusted model

"Trusted" means trusted with the raw document. It is deliberately **not** the agent's own model: an agent on a cloud model must not anonymize by sending the very document to that cloud.

1. Open the **Admin console** (the shield button), then **Settings**, and scroll to the **Anonymization** card.
2. Pick a **Provider**: `ollama`, `litert`, `openai (or a compatible server)`, `anthropic`, `gemini`, `xai` or `openrouter`.
3. Type the **Model**, for example a model you pulled in Ollama. Don't use an Ollama `-cloud` model: those run at Ollama, not on your machine.
4. Fill in the **Address** if it isn't the provider's usual one. For a vLLM, LM Studio or llama.cpp server, use `openai` and its address (`http://localhost:8000/v1`, or one on your network).
5. Add an **API key** only if the provider needs one. It is sealed and never shown again; type a new one to replace it.
6. For a cloud provider (`anthropic`, `gemini`, `xai`, `openrouter`, or `openai` with no address), a box appears that starts *This model doesn't run on this machine or your network*. Tick it only if you trust that provider with the raw documents: zero data retention, a data-processing agreement, a region you accept. Without the tick, the card won't save it.
7. Choose **If the model fails**:
   - **Refuse** (default): nobody gets a result that missed what only the model would find.
   - **Code only, marked**: the result comes from code alone and says the model failed.
8. Leave **On** switched on, and click **Save**. The card says which kind of trust applies: *runs on this machine or your network*, or *You vouched for this provider*. The card's badge then reads **On · provider/model**.
9. Click **Check on samples**. Six made-up documents with made-up people go through code alone and through code + your model. The table shows what **Code alone missed**, what **With the model missed**, anything the model **also hid** that should have stayed, and the **Time**.

**What counts as trusted.**

| Model | Trusted? |
|---|---|
| `ollama` or `litert` with no address, or an address on this machine or a private network | Yes, as it is |
| `openai` with an address on this machine or a private network | Yes, as it is |
| `anthropic`, `gemini`, `xai`, `openrouter`, or `openai` with no address | Only if you tick the box |
| An Ollama `-cloud` model, or `ollama`, `litert` or `openai` at a public address | The card shows no box for these and won't save them. Use a local model, or set it in the environment with `FD_ANONYMIZE_ATTESTED=1`. |

`localhost` and private addresses count as local. A host name that can't be resolved counts as public.

**Remove** clears the card's setting. If the deck's environment sets a trusted model (below), the card says so, and removing the card's setting goes back to it.

**Cost and speed.** A model that isn't local counts its tokens against the daily allowance of the person whose text it read. The model reads about 6,000 characters per call, so a long file is many calls; on a local model on an ordinary machine, that can take minutes. Thinking is off for these calls by default.

### Name recognition (optional)

A small model that finds names in plain sentences, on this machine. It runs for every anonymization on the deck, next to the code.

1. In **Admin console → Settings**, find the **Name recognition** card.
2. Switch it **On**.
3. Pick the **Model**:
   - `classla/bcms-bertic-ner`: recommended. Croatian, Bosnian, Serbian and Montenegrin, and English names too. People and companies, about 440 MB.
   - `Davlan/xlm-roberta-base-wikiann-ner`: other languages. People only, finds fewer, about 1.1 GB.
4. Click **Download and start** (or **Start it**, when it is already downloaded). The badge goes **Downloading…**, **Starting…**, then **On**.

It needs `transformers` and `torch` where Flight Deck runs. If the badge says **Not installed**, run `pip install transformers torch` there and restart Flight Deck. The desktop app and a Raspberry Pi don't have them. The download comes from Hugging Face, once; only these two models can be picked, and only their safetensors weights are loaded. Flight Deck starts it again by itself after a restart, but never downloads anything on its own.

### Environment variables

A self-hosted deck can set the trusted model without the page. A model saved on the card takes the environment's place.

| Variable | Default | What it does |
|---|---|---|
| `FD_ANONYMIZE_PROVIDER` | none | The trusted model's provider (the same list as the card). Needs `FD_ANONYMIZE_MODEL` too. |
| `FD_ANONYMIZE_MODEL` | none | The trusted model |
| `FD_ANONYMIZE_BASE_URL` | the provider's usual | Its address |
| `FD_ANONYMIZE_API_KEY` | none | Its key |
| `FD_ANONYMIZE_ATTESTED` | off | `1`, `true` or `yes`: you vouch for a model that isn't local |
| `FD_ANONYMIZE_ON_FAILURE` | `refuse` | `refuse`, or `deterministic` (code only, marked) |
| `FD_ANONYMIZE_MODEL_TIMEOUT` | `600` | Seconds for one call to the trusted model |
| `FD_ANONYMIZE_MODEL_MAX_TOKENS` | `32000` | The trusted model's answer limit |
| `FD_ANONYMIZE_MODEL_NUM_CTX` | the smallest of 8K, 16K or 32K the call fits | Ollama only: the context the model is loaded with (2048 or more) |
| `FD_ANONYMIZE_MODEL_THINK` | `off` | The trusted model's thinking: `off`, `low`, `medium`, `high`, `on`, or `model` (the model's own setting) |
| `CLAW_ANONYMIZE_GATE_TIMEOUT` | `1800` | Read by agents, not by Flight Deck: seconds one privacy-gate check may take |

## Anonymize a text or a file

Anyone signed in to the deck can do this. Open **Anonymize** (see [Where to find it](#where-to-find-it)); it opens on the **Anonymize** tab.

1. At the top of the card, pick **Text** or **File**.
   - **Text**: paste a letter, an email, notes or a table.
   - **File**: drop it on the box, or click to choose one. Up to 25 MB. Word, Excel and PowerPoint come back as the same kind of file, and so do text, CSV and JSON. PDF and web pages come back as Markdown text.
2. Under **Keep the real values?**, pick **In a vault** (you can put them back) or **Keep nothing** (you can't).
3. With **In a vault**, name the **Vault**: one per case or client, for example `Velora`. A new name makes a new vault. Pick an existing one from the list to keep its tokens.
   - If you leave the field empty, the run goes into the name shown greyed in it: your most recently used vault, or `My documents` the first time.
4. With **Keep nothing**, you can tick **Make them coarser** under **Details**.
5. Under **Who looks** (shown only when the deck has a trusted model), keep **Code + trusted model**, or pick **Code only** or **Trusted model only**. Without a trusted model, the page says that code alone looks.
6. Optional: click **More options**.
   - **Leave as they are**: one per line. Your own company, public figures, places.
   - **Always hide**: one per line. Add `|` and a kind for something that isn't a person: `Project Falcon | PROJECT`.
7. Click **Anonymize**.

**Reading the result.**

- The card is titled **Anonymized** (or **Nothing private found**). Under the title it names the vault the real values went into, or says nothing was kept.
- Coloured chips count what was found: "2 persons", "1 email"…
- A line says who looked: *Checked by code and the trusted model … (it found 3 more)*, or *Checked by code only*. If the trusted model failed and the admin chose **Code only, marked**, it says so in amber. Then check the result yourself.
- *A second look caught N more* means the self-check replaced something in another round.
- Amber notes are warnings, for example pictures left inside an Office file.
- The result follows, with every token as a coloured chip. For a file, this is the text inside it as it now reads.
- **Every token** lists each token, how often it appears, and which ones are new in this vault.

**Taking it with you.** For text, click **Copy** or **Download** (`anonymized.txt`). For a file, click **Download** and the file's name. The file's own name is anonymized too, with what the vault knows: if the CV names Ana Horvat, `Ana_Horvat_CV.docx` comes back as `PERSON_1_CV.anonymized.docx`.

Missed something? Add it under **Always hide** and run again into the same vault. From then on, the vault knows it, and code finds it in every later document by itself.

## Put the real names back

For anything that holds tokens from one of your vaults: the anonymized file itself, a summary someone wrote from it, an agent's answer.

1. Open **Anonymize → Put back**.
2. Pick the **Vault** the text or file was anonymized into. The list shows how many values each holds.
3. Under **What to put back**, pick **Text** (paste it) or **File** (drop it).
4. Click **Put back**.
5. The card says how many values were put back. Tokens that aren't in this vault are listed and left as they are; maybe they belong to another vault.
6. **Copy** or **Download** the text (`restored.txt`), or download the file. `Contract.anonymized.docx` comes back as `Contract.restored.docx`, with the real values inside the file.

The result holds the real values again. It is for you: don't paste it into an agent's chat, or anywhere it shouldn't go.

## Files and folders where they are

In the kiosk, a file in your **Shared folders** or in the **File library** can be anonymized where it is, and so can a whole folder. The original is never changed: a copy goes beside it.

### A file

1. Open **Shared folders** (or the **File library**) and click the file.
2. In its panel, click **Anonymize**. The button shows only for kinds that can be anonymized.
3. In the dialog, pick **In a vault** or **Keep nothing** under **What happens to the real values**. With **Keep nothing**, you can tick **Make them coarser**.
4. Check the **Vault**. It starts as the folder's name; a file from the File library starts with its agent's name.
5. Optional: **More → Leave these as they are**, one per line.
6. Click **Anonymize**. The dialog says where the copy was saved and which vault holds the real values.

A Shared folders copy goes beside the file, as `Contract.anonymized.docx`, with the file's name anonymized too. If you can't write there (a Drive or OneDrive folder, a read-only or view-only folder), it goes to your own **Anonymized** folder instead. A File library copy goes beside the file, in its agent's files or in the shared folder it's in, and is listed in the Library at once. A copy never overwrites an earlier one: a second copy is `… (2)`.

This dialog has no **Who looks** choice: the trusted model looks too when the deck has one.

### A whole folder

1. In **Shared folders**, open the folder's **⋯** menu and click **Anonymize the whole folder**. Inside a subfolder, the same menu says **Anonymize this folder**. Every subfolder row has its own **⋯ → Anonymize** too, read-only folders included.
2. Pick the options as for a file, and click **Anonymize**.
3. The dialog shows the progress, file by file. **Stop** keeps what was finished. **Close (it keeps going)** leaves it running on the deck, and opening the dialog again shows it.
4. When it's done, the dialog says how many files went into which folder. It lists the files that **couldn't be anonymized** (no copy is made) and those **left out**.

Where the copies go:

- **A subfolder**: a new folder beside it, `<name>_anonymized`.
- **A whole folder**: a new folder of your own, `<name>_anonymized`, then `-2` and so on.
- **A place you can't write to** (a Drive or OneDrive folder, a read-only link, a folder shared with you to view): your own **Anonymized** folder, under the source folder's name.
- **Folder and file names** in the copy are anonymized too, into the same vault, after the files have run: a folder `Ana Horvat/` whose files name her comes back as `PERSON_1/`.

What a folder run leaves out: pictures, archives and other kinds; files over 25 MB; and earlier anonymized copies. It does up to 2,000 files. You can run one folder at a time. With **Keep nothing**, a person keeps one token across the whole folder.

A Flight Deck restart drops a running folder job; copies already placed stay.

The full layout's VFS and Library pages don't have these buttons yet. Use the Anonymize page there, or the kiosk.

### Office files, PDFs and pictures

- **Word, Excel and PowerPoint** (`.docx`, `.xlsx`, `.pptx`) are edited inside the file. Formatting, charts and pictures stay exactly as they were; only the text changes. A name split across bold and plain text keeps its bold. An Office file embedded in another one (a chart's workbook) is anonymized the same way.
- **Text kinds** come back as the same kind: `.txt`, `.md`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.xml`, `.yaml`, `.eml`, `.srt`, `.vtt`, `.ini` and `.log`. A CSV or JSON is done value by value, and column names help ("email", "ime", "adresa").
- **PDF and HTML** come back as Markdown text (`.anonymized.md`). Their layout and pictures aren't kept. A scanned PDF is read with OCR on the deck, when Tesseract or RapidOCR is installed.
- **Pictures are not anonymized.** Pictures inside an Office file stay as they are, and the result says how many; check them for faces, logos, signatures or text. Folder runs leave picture files out. On the Anonymize page, a picture file (`.png`, `.jpg`, `.tif` …) comes back as the text OCR read from it, in Markdown, when the deck has an OCR engine.

## The privacy gate

The privacy gate is for your agents. When it is on, what they read reaches their model as tokens, and the real values stay in your vault called **Chat**. It's off until you turn it on. It is yours alone: it covers your agents, and your own turns on agents shared with you. Other people set their own. On a deck with sign-in off, it is one setting for the deck.

### Turn it on

1. Open **Anonymize → Privacy gate**.
2. Switch **On**.
3. Under **What it covers**, pick one:
   - **Sources I pick**: only what the sources you tick below give your agents.
   - **The whole chat**: your messages, what every tool returns, and the memory and recall notes. Your own chat then shows the real values again.
4. Under **For which agents**, pick one:
   - **On an untrusted model** (default): the gate applies only to an agent whose current model isn't trusted. An agent on a local model, or on the admin's trusted model, sees the real thing.
   - **Every agent**.
5. Under **Who looks**, pick **Code** (default, quick) or **Code + trusted model** (shown only when the deck has a trusted model). The trusted model catches more, but every result waits for it.
6. With **Sources I pick**, tick the **Sources**. The first four are ticked to start with:
   - **Mail (Gmail, Outlook, IMAP)**
   - **Chats (Telegram, WhatsApp)**
   - **Calendar**
   - **Contacts**
   - **Files (what agents read; or only some folders)**
   - **Data tables**
   - **Memory (external and deep memory)**
7. With **Files** ticked, you can limit it under **Only these folders (optional)**: one per line, as your agents name them, for example `vfs:clients` or `/Users/you/Clients`. Empty means every file they read. Drive and OneDrive are always covered.
8. Click **Save**. Your agents follow it from their next step, within a minute.

Start a new conversation afterwards. Conversations from before still hold the real values.

### What you see in chat

- **A status line while it works**, in place of "thinking": *🛡️ Anonymizing what google_mail returned — code is looking for private data…*, *🛡️ Anonymizing your message — …* or *🛡️ Anonymizing 3 new texts before the model reads them — …*. When the agent sends mail: *🛡️ Putting the real values back into send_mail…*.
- **With Sources I pick**: an agent's reply that has tokens in it shows **Real names** under it, next to the time. Click it to read the reply with the real values; **Tokens** switches back. Only you see them, and the agent never does. If it can't, the link reads **No real names found**; hover it to see why.
- **With The whole chat**: your chat in Flight Deck shows the real values already, and so does your own private Telegram chat with the agent. Files the agent writes and searches it runs use the real values too.
- **Mail and WhatsApp messages** the agent sends with its mail and WhatsApp tools get the real values just before they leave. The recipient reads real names; the model never does. If the values can't be put back, nothing is sent.
- **If the gate can't check something, nothing gets through.** A tool's result isn't shown to the model ("The privacy gate couldn't check what … returned, so it isn't shown"). With the whole chat, nothing is sent to the model, and you see why as the reply.

While the gate applies, an agent's `anonymize` tool uses the **Chat** vault too, so every token in a conversation means the same.

### Reset the chat vault

Over time the **Chat** vault fills up. To start over:

1. With **The whole chat** selected, click **Reset the chat vault**, then **Reset**. With **Sources I pick**, use **Vaults → Chat → Reset** instead.
2. Start new conversations with your agents.

Every value in the vault and its history are deleted. The vault, its key and its token numbers stay, so an old token never gets a new meaning. Old conversations show their tokens from then on, because they can't be put back. Your agents notice the reset and anonymize afresh. This can't be undone.

### What the gate doesn't cover

- **With Sources I pick**, anything outside the ticked sources reaches the model as it is: your own messages, web pages, the shell, the browser.
- **The system prompt** is not anonymized, even with the whole chat. That includes what your profile and an agent's members list say. (It is left out on purpose: a local trusted model reading it on every change cost minutes per turn.)
- **Pictures** go to the model as they are.
- **Other people** talking to your agent get its replies with tokens, never the real values: another Telegram user, a group, WhatsApp, a public page.
- **The real values aren't put back** in replies over WhatsApp, Messenger or devices, in scheduled deliveries, on public pages, or in the Line and notifications. They show the tokens.
- **Flight Deck's own one-off generations** on your model tier aren't gated.
- **The agent's own records**: its tool-call audit keeps the arguments as they actually ran (with real values), and its memory index holds workspace files as they are. The model doesn't read either directly; recall goes through the gate.
- **With On an untrusted model**, an agent on a local or trusted model reads everything as it is. That is the point of the setting.

## Asking an agent

Agents started by Flight Deck have the `anonymize` tool. Just ask:

- "Anonymize vfs:clients/Velora/contract.docx into the vault Velora."
- "Anonymize every file in vfs:clients/Velora/, keep nothing."
- "Anonymize https://www.example.com/team, but leave Velora d.o.o. as it is."
- "Put the real names back into the summary you just wrote, from the vault Velora."

What it does:

| Action | What it takes | What happens |
|---|---|---|
| `anonymize` | One of `path` (a file, or a `vfs:` folder), `url` or `text`; plus `vault`, `mode` (`reversible` or `one_way`), `keep`, `extra`, `generalize`, `lane`, `output_path` | The result is saved as a new file; the agent sees the anonymized text and the tokens |
| `restore` | `path` or `text`, and `vault` | The real values go into a new file for you. The agent gets its path, never its contents |
| `vaults` | nothing | Your vaults' names and counts, never their values |

- **The raw file never goes through the agent's model.** For `path` and `url` the tool reads the file or page itself and sends it straight to Flight Deck; only the anonymized result comes back. Text you pasted into the chat has already been seen by the model; the tool still works, and says so.
- **Where results go**: `saved/anonymized/<conversation>/…`, and restored files to `saved/restored/<conversation>/…`. Office files come back as the same kind of file; PDFs and pages as Markdown.
- **Which vault**: the one you name. Otherwise one per conversation (`chat-…` in your vault list), or **Chat** while your privacy gate applies to that agent.
- **A folder** (`vfs:<folder>/…/`) runs like **Anonymize the whole folder**. The agent waits up to 200 seconds; a longer run carries on, and the copies appear when it's done.
- **Restore only when a person asks.** On automatic turns (schedules, flows, other agents, the Taskboard…), `restore` is refused: you can put the values back yourself on the Anonymize page.
- **Members of a shared agent** can use it on their own turns, with their own vaults; the owner never sees them. Their results go to their own files, under `anonymize/anonymized/` and `anonymize/restored/`. This works on a process agent; a Docker agent's members have chat only.

## Captain for Chrome

Captain for Chrome 0.3.0 can anonymize the page you're on, or the text you selected. No agent is involved. The extension must be paired with your deck first: see [Captain for Chrome](devices-and-channels.md#captain-for-chrome).

1. Open the page and the Captain side panel. Wait until the panel has read the page.
2. To anonymize only part of it, select that text first.
3. Click **Anonymize this page** (it reads **Anonymize the selection** when text is selected), under the Send button. Or right-click the page or the selection and pick **Anonymize with Captain**.
4. The panel shows what was **Found**, which lanes looked, and the result with its tokens as chips.
5. Click **Copy** to take it with you, or **Save in Flight Deck** to keep it. It goes to **Shared folders → web-clips**, in a folder named `<day>-<site>-<title>-anonymized`. The title in that name is anonymized too, because a page title often carries the very name you're hiding.
6. Later, when you have an answer that holds the tokens (from a chatbot you pasted the anonymized text into, say), open **Put the real names back**. Paste it and click **Put them back**, then **Copy**.

**The vault** is named after the site as the browser shows it, `www.` kept: `www.example.com`. The same person gets the same token on every page of that site. `www.example.com` and `example.com` are two vaults.

## Vaults

A vault holds one case: the real values behind its tokens. It belongs to you, never to an agent, even when an agent made it.

**See them.** Open **Anonymize → Vaults**. The most recently used come first. Each row shows how many values the vault holds and when it was last used. It also says if an agent started it ("started by …") and when it ends, plus what kinds it holds.

**Open one.** Click a vault. A side panel opens.

1. **Name**: rename it.
2. **Delete it by itself after**: **Never**, **30 days**, **90 days** or **A year**, counted from today. When the time is up, the real values are gone for good, and its tokens can't be put back.
3. Click **Save** for a new name or end date.

**See what it keeps.** Under **What it keeps**, click **Show the real values**. Only you can see them, and opening the vault is noted in its history. **Hide** puts them away again.

- **Find a token or a value** filters the table.
- Each row has the token, the real value it puts back, and:
  - **also**: other forms it was seen in ("Horvata", "A. Horvat"), which put back the main value;
  - **spelled**: exact spellings with their place, `:2`, `:3`. `⟦ORG_1:2⟧` puts back exactly that spelling ("velora" in a link).
- **↑** counts how often the token went out to a model, and **↓** how often it came back and was put back.

**A token's history.** Click a token's ↑ ↓ to open its history:

- **Added**: when, from where (the chat, a tool result, a file, a folder, a web page in Chrome…), and what found it: an email address, a label, the name model, the trusted model, your own list, or a spelling of another token.
- **Went out**: each new text it went out in, and to which model.
- **Put back**: where it came back (a reply in your chat, a tool step, Telegram, a text you restored, Chrome).

Each line shows the text around the token, as it was sent: tokens, never real values. Hover a token there to see its real value. The newest 12 show first, then **Show N more** and **Load older**. Sends and returns are kept for 180 days; the line that added the token always stays.

**History** at the bottom lists what happened to the vault: created, anonymized into, values put back, opened, changed, reset, expired. Only counts are kept, never values.

**Reset or delete.**

- **Reset**, then **Reset** again in the confirmation: every real value and its history are deleted; the vault, its key and its token numbers stay. Documents and chats that hold its tokens keep them, but they can't be put back any more. New names get new numbers.
- **Delete**, then **Delete for good**: the values and the vault's key are deleted. Documents anonymized into it keep their tokens forever.
- Neither can be undone.

**Who sees what.**

- Only you can open your vaults. There is no page where an admin, or anyone else, sees them.
- Agents get vault names and counts, never values. A restored file goes to you, not to the agent's model.
- A member of a shared agent gets vaults of their own; the owner can't see them.
- On the deck, the vaults live in `anonymize.db` in Flight Deck's data folder. Each vault has its own key, sealed with the deck's data key. The values, their forms, their history and even the vault's name are encrypted.
- You can have up to 1,000 vaults, each with up to 20,000 values.

## Limits and tips

- **Check the result before it leaves.** Code misses a name with nothing around it, unless the vault already knows it or the name pack is on. Models miss things too: a name that is also a word at the start of a sentence ("Will", "Grant", "Zora"), or a small company's name in a short title. Add what was missed under **Always hide** and run again.
- **Numbers and dates stay on purpose.** Amounts, ordinary dates, hours and counts stay, and so do cities and countries on their own, job titles, well-known brands and public organisations. Only labelled dates of birth, IDs, accounts and the like become tokens. To share data for good, use **Keep nothing** with **Make them coarser**.
- **A vault is reversible.** For data you hand over for good, **Keep nothing** is real anonymization; a vault is pseudonymization.
- **Use one vault per case.** The same vault gives the same tokens across documents, so a summary of five files still names one person with one token. Leaving **Vault** empty uses your most recently used vault.
- **Pictures aren't changed, and PDFs come back as Markdown.** Check the pictures in Office files yourself.
- **Map links keep their coordinates.** Some foreign company forms and registry numbers (Luxembourg's SCSp, S.à r.l.) aren't known to code. In a spreadsheet, a first name and a surname in two separate cells get two tokens.
- **Restored text holds the real values.** Don't paste it into an agent's chat.
- **The privacy gate is quickest on Code.** With **Code + trusted model** on a local model, a long text can take minutes. The status line shows when it is working.
- **Sizes**: a file up to 25 MB, a text up to 2,000,000 characters, a folder up to 2,000 files.

Design notes and what's still open: [docs/anonymization-plan.md](../anonymization-plan.md).
