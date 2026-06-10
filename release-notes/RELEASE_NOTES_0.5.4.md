# Captain Claw v0.5.4 Release Notes

**Release title:** Parallel web research — fast, and honest about it
**Release date:** 2026-06-10

A research-quality release. The headline is **`web_fetch_batch`**: read a whole
page of search results **in parallel**, with each URL self-correcting from a
fast HTTP fetch to a headless browser only when it needs to. Around it, two
**honesty guards** stop a weak model from *claiming* it searched the web (or
leaning on memory) when it didn't — plus friendly new launcher commands.
Additive and backward compatible with 0.5.3.

---

## What's new

### `web_fetch_batch` — parallel multi-URL fetch
A new tool that takes a **list** of URLs and fetches them concurrently, returning
clean text per URL. Use it instead of repeated `web_fetch` after a `web_search`.

- **Fast → deep self-correction.** Each URL is fetched over plain HTTP first;
  only pages that come back thin or JS-rendered escalate to a headless browser.
  A JS-shell-aware heuristic (small text vs. large HTML) avoids wasting the
  browser on genuinely short pages.
- **One shared browser, isolated per URL.** Deep fetches run in a single
  Chromium instance with a separate context per URL (concurrency-capped) instead
  of relaunching the browser for each.
- **Never drops content.** If deep mode is unavailable, the fast-HTTP content is
  surfaced anyway (same as `web_fetch(deep_fetch=false)`), not discarded.
- **Self-installing browser.** On the first deep need with no browser binary, a
  one-time `playwright install chromium` runs in the **background** — the call
  returns fast content now, deep mode works on the next call.
- **Budgeted.** Caps at 10 URLs/call (the rest are handed back to the agent),
  25k chars per URL, 150k total. All knobs live in `tools.web_fetch` config.

Enabled wherever `web_fetch` is (so on by default), and advertised in **eco
mode**.

### Honesty guards — no fabricated research
Two new guards in the agent loop, aimed at smaller/free models that narrate work
they didn't do:

- **Web-research claim gate** — if the reply claims it *searched/fetched the web*
  but **no** web tool ran that turn, the agent is forced to retry with
  `tool_choice=required` and actually call `web_search` → `web_fetch_batch`.
- **Fresh-data overrides memory** — when you say *"refresh from the web, don't use
  memory"*, the automatic memory-context injection is **skipped for that turn**,
  so the agent can't quietly answer from stale memory.

### Friendly launcher commands
- **`flight-deck`** — alias of `captain-claw-fd` (the dashboard).
- **`captain-claw-agent`** — alias of `captain-claw-web` (a single agent's web
  server).

Both ship as pip entry points **and** standalone binaries. `captain-claw` is
unchanged (the terminal agent). On an editable install, run `pip install -e .`
once so the new launchers are generated.

### Flight Deck polish
- **Activity narration renders markdown** — the between-step blurbs now render
  tables, bold, and inline code instead of raw `**`/`` `…` ``/pipes.
- **Sidebar trimmed** — the registered-apps ("Apps") section is removed.

---

## Upgrade

```bash
git pull
# UI assets are committed; rebuild the frontend only if you build locally:
npm --prefix flight-deck run build
```

- **Restart the agent** to activate `web_fetch_batch` and the honesty guards
  (they live in `captain-claw-web`).
- **Restart / hard-reload Flight Deck** for the markdown narration and sidebar
  change.
- **Desktop app:** `./build-desktop.sh` — the new tool, guards, and the
  `flight-deck` / `captain-claw-agent` binaries are all in the rebuilt bundle.

Backward compatible with 0.5.3 — single-URL `web_fetch` is unchanged and
everything here is additive.
