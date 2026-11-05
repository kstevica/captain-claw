# Captain Claw v0.8.10 Release Notes

**Release title:** Provisions
**Release date:** 2026-11-05

Pick an AI provider, paste its key, and Captain fills every model tier with that provider's current
models. This is the quick start the desktop app was missing. Also in this release: agents read scanned
PDFs and photos with a local OCR engine (no model involved), a turn that is still making progress no
longer stops at 40 steps, and on Windows files keep their own line endings.

4 pull requests: #426, #427, #428 and #430. **Captain for Mac 0.1.1** came out the same day on its
own release, [`mac-v0.1.1`](https://github.com/kstevica/captain-claw/releases/tag/mac-v0.1.1) (#429).

## New

- **Quick start tier sets.** **Console → Models → New tier set** and the Library's first-run wizard
  now open on **Quick start**. Pick Anthropic, OpenAI, Gemini, Grok or DeepSeek and paste the API key.
  Captain makes a tier set named after the provider (a second one becomes "Anthropic 2"), with that
  provider's current models on every tier:

  | Provider | Reasoning / Coding | Balanced / Long context / Vision | Fast |
  |---|---|---|---|
  | Anthropic | `claude-opus-5-5` | `claude-sonnet-5-5` | `claude-haiku-5-5` |
  | OpenAI | `gpt-6.1-sol` | `gpt-6.1-terra` | `gpt-6-luna` |
  | Gemini | `gemini-4-argon` | `gemini-3.8-flash` | `gemini-3.5-flash-lite` |
  | Grok | `grok-4.7` | `grok-4.7` (long context `grok-4.3`) | `grok-4.20-0309-non-reasoning` |
  | DeepSeek | `deepseek-v4-pro` | `deepseek-v4-pro` (vision `deepseek-flash`) | `deepseek-flash` |

  - A pasted key picks its provider by its prefix, and a key that looks like another provider's is
    flagged.
  - Before the set is made, the key is checked against the provider's list of models. If the provider
    can't be reached, or the key can't use one of the models, you can still **Create anyway**. The key
    is used for that check and kept only in your tier set.
  - Context sizes stay under each provider's price jump: Haiku 100k, Grok 160k, OpenAI 256k.
  - The list lives in `captain_claw/instructions/tier_presets.json`, with the date it was checked
    (5 November 2026). A new model is an edit there, with no frontend build. The new models also have
    prices, so their runs aren't counted at $0.
- **`ocr_pdf` and `ocr_image`: scans read without a model.** A scanned PDF used to have no fixed way
  through: `pdf_extract` found no text, and agents improvised with scripts, shell tools or a vision
  model that gives different text each time. Now:
  - `ocr_pdf` keeps each page's own text when it's good and reads only the rest (`mode=auto`), or
    re-reads every page (`mode=ocr`). `ocr_image` reads png, jpg, webp, bmp, gif and tif (each page of
    a multi-page TIFF) and turns photos upright first.
  - **The same file gives the same text, every time.** Each page's rule is fixed and recorded, the
    engine choice is fixed, and results are kept by file hash and settings in `saved/.ocr-cache/`.
  - **Engines:** RapidOCR comes with Captain (a new core dependency; it reads č ć đ š ž without a
    language pack, and doesn't install on 32-bit ARM). Tesseract is used when it's installed and you
    ask for a language it has. With neither, the tool says how to install one and never falls back to
    a model.
  - The text lands in `saved/ocr/<session>/<name>-<hash>/<name>.md`, page by page, with how each page
    was read and the pages it wasn't sure of. `pdf_extract` now names the pages that have no text and
    points to `ocr_pdf`.
  - That makes **72 built-in tools** per agent (71 in the desktop app, which has no Captain Swarm).

## Fixed

- **A turn that's working keeps going.** At 40 steps, an ordinary request used to stop mid-job with
  "iteration budget exhausted" unless a plan or Jev said work was left, and most decks have no Jev.
  Now:
  - A turn whose last steps made progress gets more steps, with no judge, up to twice its budget.
  - Past that, it keeps going while a plan or requirement says work is open. Otherwise Jev decides
    where it's set up, and without Jev, progress alone is enough.
  - The hard ceiling is 4× the budget, at most 300 steps (160 for the default 40). A turn that only
    repeats itself still stops.
  - When the limit does end a turn, the agent writes where the work stands (done, left, next), and the
    reply ends with: reply **continue** and I'll pick up where I left off.

  Long turns can now cost more: a working turn can run to 160 steps where it used to stop at 40–80.
- **Files keep their own line endings, on Windows too.** On Windows, `write` turned every line ending
  into CRLF. Its own check then saw a different size and asked the agent to write the file again, which
  landed the same way: a loop. Now:
  - The bytes on disk are the bytes sent.
  - A file that already exists keeps the line ending most of its lines use, and a new `.bat` or `.cmd`
    gets CRLF.
  - `edit` no longer flattens CRLF files to LF on macOS and Linux, or LF files to CRLF on Windows.
  - A file that differs only in line endings passes the check, and a check that still fails no longer
    asks for the same write again.
- **The first-launch wizard saves its tier set.** It moved on to Quick chat in the same click, before
  the set was saved, so the untouched "Default" set stayed. That was also true of the one-model wizard.
- **The Overview checklist** on a deck without accounts ticks *Add your API keys* when your own tier set
  carries a key.

## Upgrading

1. **The desktop app**: **Check for Updates…** (it also offers 0.8.10 within a minute of starting).
   Quick start is in **Console → Models → New tier set**. New installers are on this release, but you
   need one only for a fresh install: `Captain-0.8.10-arm64.dmg` (Apple silicon), `Captain-0.8.10-x64.dmg`
   (Intel) and `Captain-Setup-0.8.10-x64.exe` (Windows). They aren't signed, so macOS asks first
   (System Settings → Privacy & Security → **Open Anyway**), and so does Windows (**More info → Run
   anyway**).
2. **Each Flight Deck**: **Console → Updates**, or pull and reinstall by hand. RapidOCR and pypdfium2
   are new core dependencies, about 90 MB to download, so `pip install -e .` (or `uv sync`) in the deck's
   environment. Then restart Flight Deck, then every agent: agents need the restart for the two OCR
   tools and the new step budget. Hard-refresh every browser for the Quick start form.
3. **Optional:** install Tesseract for more languages (`brew install tesseract tesseract-lang`,
   `apt install tesseract-ocr tesseract-ocr-hrv`, or the UB Mannheim installer on Windows).
   `tools.ocr` in `config.yaml` sets the engine order and the page cap; see USAGE.md.

Devices need nothing: firmware stays **0.20.4** / PaperColor B **1.0.31**. BotPort hubs need nothing:
0.8.10 keeps protocol 2. Captain for Chrome and Captain for Mac need nothing from this release.

## Pull requests

#426 A file keeps its own line endings; CRLF is not truncation · #427 `ocr_pdf` and `ocr_image`, scans
read by a local engine, no model · #428 A working turn keeps its steps; the limit lands with "continue" ·
#430 Quick start tier sets: provider + API key, its current models
