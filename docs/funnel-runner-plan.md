# List funnel: a cheaper way to build a list

Status: phases 1–4 built (branch `feat/list-funnel`, PR #668, not merged). Phase 5 (side by side on the Pi) is left. Bat stays the
default; the funnel is a second way to run a factory whose result is a list, so
the same charter can run both ways and be compared.

Decided (2026-11-19): a Bat engine, not a separate runner; 1,000 searches a run;
short groups get a small agent step; no re-check window (an item in the ledger is
never checked again). And: **any kind of list**, offline files included, not only
firms from the web.

## 1. Why

A factory run that built a 60-row list of small firms took about 20 hours and
190M tokens for 54 rows: 3.5M tokens per row kept, about 580k per firm checked
(54 kept, 272 rejected). Where it went:

- **Every model call resent the worker's whole context.** Bat's workers averaged
  113k–135k tokens per call (conventions, corrections, folded notes, the last
  pages fetched), about 1,600 calls in all, around 4,700 tool calls. Each search
  or fetch added a little and paid for all of it again. This is how agents work,
  in Bat, Vatra or a chat alike, so Vatra wouldn't be cheaper: its workers run
  the same long turns, with a Lead, a reporter and fact-checks on top.
- **The cheapest disqualifier came last.** Of 272 rejections, 101 were "too big
  or size unproven" (a search snippet's "11-50 employees" settles most), 52 "no
  dated signal", 30 outside the US or the state group, 30 "no email on the
  firm's site". Workers often checked pages before size.
- **Fix rounds and redos** (fixed in #654, #657, #667) took about a quarter of
  the time. Fixing them doesn't change the cost per firm.

## 2. The idea

**Any list.** The recipe says what one row is (a firm, a person, a product, an
invoice line, a paper) and where items come from. Sources may be:

| Source | Items | Needs the web |
|---|---|---|
| `file` | each row of a CSV, TSV, XLSX (first sheet) or JSON-lines file in the run's folder | no |
| `document` | pulled out of a PDF, Word, PowerPoint, Markdown, text or HTML file, part by part, by a short call | no |
| `links` | the owner's links file: each link an item, or each a page to read for items | only for pages |
| `page` | pulled out of a page that lists many items | yes |
| `search` | pulled out of a search's results; list pages found there are followed | yes |

An offline list has no searches and no gathering: rows are judged in batches (up
to 25 per call for short rows), each value checked against the row or the part of
the document it came from.

Most of finding and checking a firm is fetching and reading, which code does for
free. The model only has to judge. So: a fixed recipe per firm, code for the
searching, fetching, pattern-matching and checking, and **one small model call per
firm in a fresh context** (about 5k tokens: the rules and what code found).

| | Bat (agents) | List funnel |
|---|---|---|
| Who searches and fetches | the worker, call by call | code, through one agent's tool endpoint |
| Context per model call | ~120k, growing | ~5k, fresh each time |
| Calls per firm | ~5 (each with the whole context) | 1 (plus shared sourcing calls) |
| Where an email came from | the worker says | code saw it on that page, and keeps the page |
| Strength | resourceful: Wayback dates, second domains | cheap, fast, provable, the same every run |

Estimate for a 60-row run (to be measured, not promised): **2–6M tokens** (vs
190M), **about 400 searches and 800 page fetches**, **2–4 hours on the Pi's model**
(most of it the per-firm calls), far less on a hosted model.

## 3. Where it lives: a Bat engine, a First Mate choice

The funnel runs **inside Bat's shell**, as another engine, so everything around a
run stays the same: the run record and folder, the progress panel (steps,
activity, files, notes), caps and metering, timing, the judge and its reviewers,
fix rounds, continue after a cap, the finish notice, results, leftover-worker
cleanup.

- First Mate: a version's `runner` may be `funnel` (store `RUNNERS` already has a
  list of names; add it). `FunnelRunner` is `BatRunner` with
  `config.engine = "funnel"` and its settings. The charter editor gets "How it
  runs: Agents (Bat) · List funnel", offered when the shaped contract has a table
  check with an amount (a list).
- Bat: `_bat_planner` hands a funnel run to `funnel.plan(run)` after shaping (the
  shaped contract is what the funnel reads); `_bat_attempt` hands a step whose
  `how` is `funnel` to `funnel.attempt(run, step)`; the course check skips funnel
  steps (their checks are built in). The judge, fix rounds (with #667's counts)
  and everything else are Bat's own.
- When shaping gives no table check with an amount, a funnel run says so in a note
  and runs as a normal Bat run.

New module `captain_claw/flight_deck/funnel.py` (engine) and maybe
`funnel_extract.py` (the pure parts: snippets, emails, dates, size bands), so the
pure parts are tested without a deck.

## 4. The run, step by step

The plan is fixed: one setup step, one step per group (segment) the contract
names, then the merge. Groups run one at a time on a single model, side by side
when the tiers allow (Bat's `one_machine` rule).

### 4.1 Setup (2–3 model calls, planning tier)

From the charter, the shaped contract and the extra data, write:

- `shared/recipe.json` (the machine's recipe):
  - `columns` in order and what each holds;
  - `groups` with their quotas, from the table check's `allowed` and `min_per`;
  - `queries` per group (15–25 directory-style searches: lists filtered by size
    and place, associations, `site:linkedin.com/company` + "11-50 employees" +
    place);
  - `filters` (hard, applied by code): country or state rules, size band, the
    exclude pattern, email-domain rule;
  - `decide` (the rules the model judges per firm, in the owner's words, short);
  - `fetch` (which pages to read: home, about, team, contact, news or blog);
  - optional fixed values (feature names from the extra data, etc.).
- `shared/conventions.md`, written from the recipe, so an agent step later
  follows the same rules (Bat's convention).

The recipe is shown in the Notes tab. An owner-edited recipe saved with the
factory (later) would skip this step.

### 4.2 A group's step: find → filter → check → decide → keep

Loop until the group's quota is met, the sources run out, or the cap is hit:

1. **Find candidates** (search via the run agent's `/api/tool`: `web_search`,
   count 20–50, country from the recipe). Domains and names come from result URLs
   and titles by code. Directory pages (one page listing many firms) are fetched
   and one small model call (~6k) pulls firms out of them.
2. **Code filters, no tokens:** drop firms already checked (the run's own list
   plus the factory's ledger, §6), domains of directories and social sites,
   non-matching country domains, the exclude pattern.
3. **Size first, no tokens:** one search for the firm's LinkedIn company page.
   The snippet's band (`2-10`, `11-50`, `51-200`…) is read by regex. Too big:
   dropped with the snippet as the reason. Unknown: goes on, and the team page
   decides.
4. **Fetch the pages** (`web_fetch_batch`, 4–5 pages, each cut to ~6k chars):
   code picks out emails on the firm's domain (or one the site links as its
   own), dated lines (2024-11, "March 2026", ISO dates) with their page, the
   state from an address, and a rough team count.
5. **One model call** (worker tier, #655's "Workers use"), fresh context:
   - in: the decide rules, the group's definition, what code found with page URLs,
     about 2k chars of the about page;
   - out (JSON): keep or drop, the reason, and the row (feature and reason
     written here, provenance given as page URLs it was shown).
   Thinking models get room (8k tokens) and "keep your thinking short" (#644).
6. **Checks by code:** the email appears in the text fetched from its cited page,
   the signal date is in the window and appears on its cited page, the size
   source is the LinkedIn page or the team page that was read, the row has every
   column and the allowed values. A row that fails is dropped with why.
7. **Keep:** append to `leads-<group>.csv`; a drop goes to
   `exclusions-<group>.csv` with reason and URL; both go to the ledger. An event
   per firm ("kept …", "dropped …: 51-200 on LinkedIn") feeds the Activity tab.

When the queries run out with the quota unmet, one model call writes new queries
from what worked (which queries gave kept rows) and what didn't, at most twice.
Then the group is "short, sources run out".

**Maybe list:** a firm that passed size, email and fit but whose signal or state
code couldn't find goes to `shared/maybe-<group>.csv` instead of being dropped.

### 4.3 Short groups: a small agent step (the hybrid)

A group still short after its sources run out gets **one Bat agent step**: "check
these N maybe-candidates for a dated signal; add the ones that pass to
leads-<group>.csv", with the conventions, #667's counts, and a worker spawned
with a **smaller context** (`max_context` around 48k at spawn, vs the tier's 160k)
so its calls stay small. The agent only works on firms that are nearly there.

Setting `leftovers: agent | none` (default agent). `none`: the funnel only, an
honest shortfall.

### 4.4 Merge and judge

The merge step is code: `leads.csv` from the group files (deduped, columns in
order), `exclusions.md`, `summary.md` from a template (counts per group, funnel
numbers, shortfalls, sources), `shared/funnel.json` (per group: found, new, size
ok, pages read, decided, kept, maybe). Then Bat's judge as for any run:
deterministic table checks, reviewers for the judgement rules (they now also see
the code's checks), fix rounds with the counts (agent steps, like the leftovers).

## 5. Caps, metering, timing, progress

- **Metering:** every model call inside `bat_routes._metered(run_id, label)`, so
  the run's tokens, the factory's cap per run and the finish notice count them.
  The funnel checks the cap between firms and stops cleanly (`llm_token_cap_reached`,
  so Continue works).
- **Searches:** a per-run search budget (setting, default 600), counted in the
  progress note; at the limit the group stops as "search budget used".
- **Timing:** the step start/stop events Bat already writes; `timing_of` works
  unchanged.
- **Progress:** each group step keeps `progress/<step>.md` up to date (counts so
  far, queries left, the last few decisions), so the Steps and Notes tabs and the
  fix planner see where it is. Later: a funnel strip per group (found → new →
  size ok → kept) in RunProgress, read from `shared/funnel.json`.
- **The run agent:** one agent per run (spawned like a Bat worker, labelled for the
  run, never given a turn) serves `/api/tool` for search and fetch, so the owner's
  search provider and keys, the worker floor and rehearsal rules all apply. It's
  removed with the run (`retire_run_workers`).

## 6. A ledger across runs

A standing factory runs again and again, so it shouldn't check the same firms
again. `first_mate_data/<factory>/ledger.csv` (domain, firm, group, outcome,
reason, run, checked at) is written by the funnel and read before checking. A
kept firm is never checked again. A rejected one is checked again after
no re-check window (decided: an item in the ledger isn't checked again). Bat runs of the same factory
get it as `shared/seen.csv` (phase 4), so both engines share it.

## 7. Settings (runner config)

| Setting | Default | What |
|---|---|---|
| `leftovers` | `agent` | short groups get a small agent step, or none |
| `search_budget` | 1000 | searches per run (`config.funnel_search_budget`) |
| `fetch_chars` | 6000 | per page |
| `concurrency` | 4 | firms fetched at the same time (model calls one at a time on a single model) |
| `recheck_days` | never | an item in the ledger isn't checked again |
| `max_queries_rounds` | 2 | new-query rounds per group before "sources run out" |

## 8. Phases

Each phase ends merged, with tests, and a live check on a test deck (isolated
`FD_DATA_DIR`) with search, fetch and model stubbed where it costs.

1. **Engine core. BUILT** (`funnel_extract.py`, `funnel.py`, Bat seams,
   `tests/test_flight_deck/test_list_funnel.py`, 10 tests): every source kind above,
   filters (field rules, the exclude pattern, `size_max`/`size_min` tested after each
   gather step so a too-big item isn't read further), gather (search templates,
   fetches from the site's root), extractors, one judging call per item (batched for
   short rows), code checks (`required`, `in`, `regex`, `in_evidence` against the
   cited page, `within_months`, `min`, `max`), state in `shared/funnel/` (picks up
   where it stopped), events and progress notes, the search budget, the merge
   (`<output>` + `exclusions.csv`). A run gets the engine with `config.engine =
   "funnel"`; First Mate's choice comes in phase 3. Planned as: `funnel_extract.py` (snippet size bands, emails on a domain,
   dated lines, state from an address, directory-page links), `funnel.py` (recipe
   from the shaped contract, group loop, decide call, code checks, files, events,
   progress notes, metering, cap stop), Bat seams (`engine`, `how: funnel`, course
   check skip, fallback to Bat when there's no list check). Tests: extraction on
   saved page text, one group end to end with stub search, fetch and model, cap
   stop, the checks dropping a row whose email isn't on its page.
2. **Leftovers and merge. BUILT** (5 more tests, 15 in all). A funnel plan with
   groups or an amount has a finishing step before the merge (none with
   `config.funnel_leftovers = "none"`). It is a Bat worker, given only the
   nearly-there items of the groups still short (up to 30, `shared/funnel/finish-items.csv`,
   in the step's "What to work on"), writing rows to `shared/funnel/finish.csv`; with
   nothing short or nothing close it ends at once and says why. Every agent step of
   a funnel run (the finishing step, the fix rounds') works in 64k of context
   (`config.funnel_agent_context`; capped, never raised above the tier's). The merge
   reads each cited page of the agent's rows again and keeps only rows that pass the
   recipe's checks (and fit their group), then writes the list, `exclusions.md`
   (every item dropped: where in the funnel and why) and `summary.md` (per group,
   what's short and why, how items went stage by stage, the checks, sources, files).
   Items carry their stage (`filter`, `read`, `judge`, `check`, `full`, `duplicate`,
   `kept`, `maybe`, `agent`, `agent-check`); `shared/funnel/counts.json` keeps the
   per-group numbers live for phase 3's strip. The judge's reviewers are told what code
   already checked in every row. Planned as: Maybe lists, the small agent step with a capped
   context, the merge (leads.csv, exclusions.md, summary.md, funnel.json), the
   judge seeing the code's checks. Tests: a short group gets one agent step with
   its maybe list; the merge; judge input.
3. **First Mate and UI. BUILT** (4 more tests, `test_first_mate_funnel.py`). `funnel`
   is a runner (`first_mate_store.RUNNERS`, `BAT_RUNNERS`), its settings checked
   (`leftovers` agent|none, `search_budget`, `agent_context`, nothing else);
   `FunnelRunner` is `BatRunner` that starts the Bat run with `engine: "funnel"` and
   the settings, and adds the funnel's counts to a run's progress (`live_counts`);
   results, timing and cap-following treat it as a Bat run. The draft charter has
   "How it runs" (Agents / List funnel, and for the funnel "Finish what's short with an
   agent"); the runner chip reads "List funnel"; a run's panel shows the funnel strip
   (per group: a bar of where its items went, on the list against what it needs, how
   many looked at; a legend with totals). Live-checked on a test deck: the choice saved
   and read back (`funnel`, `leftovers: none`), approval, a real run (its Bat run in the
   funnel's engine; with no model key the funnel noted why and the run went to Bat's
   plan, as designed; its workers removed when it was stopped), the strip from the
   progress endpoint in dark and light, the recipe and state files on the Notes tab,
   and the funnel's search-and-fetch path against a live agent (`web_fetch` through
   `/api/tool?token=` returned the page; `web_search` reported the missing search key;
   a wrong token got 401). A model-driven run waits for a deck with keys (phase 5).
   Planned as: `funnel` runner in the store and `RUNNERS`, settings
   validation, "How it runs" in the charter editor (lists only), runner chip,
   the funnel strip in RunProgress, Notes shows the recipe. Live check on a test
   deck.
4. **Ledger. BUILT** (`first_mate_ledger.py`; 5 tests in `test_first_mate_ledger.py`,
   1 more in `test_list_funnel.py`). Each factory keeps `_checked.csv` in its data
   folder (it goes with the factory): key (a site by its domain), name, group,
   outcome, reason, the charter's work hash, run, when. A firmer outcome replaces a
   softer one (close → dropped → kept). A run or rehearsal of a funnel or Bat factory
   starts with `extra/already-checked.csv`: kept items always, dropped ones while the
   charter asks for the same work (a changed charter, checks or settings lets them be
   tried again), never ones that only came close; its brief says how many and not to
   check them again. The funnel skips them (counted in its note, result and summary)
   and never takes the file as a source. When a run ends, First Mate adds what it
   checked: a funnel run every item, a Bat run its final list (its site column first,
   as the funnel names items). Test cases and carry-outs neither read nor add. The
   Data tab shows an "Already checked" card (how many, from how many runs, kept /
   dropped / close), with Download and a two-step Forget (in the history too).
   Live-checked on the test deck: the card, the download, Forget. Planned as: Factory ledger read and written by the funnel, seeded into Bat
   runs as `shared/seen.csv`, recheck window. Tests.
5. **Side by side on the Pi.** The same charter as two versions (Bat, funnel),
   one run each. Compare rows kept per group, tokens, time, searches, the code's
   provenance checks, the reviewers' verdict. Decide what each is for.

## 8b. After the first live run (2026-11-19)

The first funnel run on the Pi's Qwen never got a recipe: the setup call thought
162k characters with thinking on, looped (decode speed climbing mid-thought is the
tell), hit the 32k output cap, and answered nothing, twice. The fallback then took the
owner's two Captain links as the only "document" to find firms in. Fixed:

- every funnel call runs with thinking off (`reasoning.call_effort("off")`, which
  sends `chat_template_kwargs.enable_thinking: false` to Qwen on a LAN server) and a
  small output limit per kind (setup and extraction 6k, new searches 2k, judging 2.5k
  plus 600 per extra row); `config.funnel_thinking = "on"` gives judging calls their
  thinking back, with 16k of room;
- without a model's recipe, a web list gets one built from its checks and charter
  (`web_recipe`): searches per group (directories, lists, member lists, LinkedIn
  pages with the size band), the size band first, the item's own pages, emails
  checked against their cited page, the date window; a links file with fewer than 5
  links is reference, never a source of items.

The second live run (same evening) set up in a minute and kept its first firm with
the email proven on its page, but kept 1 of 66 in ten minutes: its searches were all
LinkedIn company searches, whose results give a firm's name and no site, so the funnel
had nothing of the firm's own to read and the judging call dropped it. Also: all four
groups got the first group's searches, the product file was taken as a list of firms,
and CaptainFeature got invented values. Fixed:

- an item without a site of its own gets one: one search for its name, the first
  result off directory and social sites whose title or address carries the name; a
  link on such a site never names an item (it falls back to its name);
- a judging call reads the charter (what each column takes) and the run's reference
  files (`Captain.md`), and answers maybe, not drop, when nothing it read came from
  the item's own site;
- "X is exactly one of: a, b, c" anywhere in the charter becomes a check on X;
- code tidies a model's recipe (`tidy_recipe`): a group's searches that never name it
  (or copy another's) are written for it; a file the charter points to for
  information is reference, never a source; the item is a noun, not a list.

## 9. Risks

- **Search quota.** About 400 searches a run. Check the Pi's provider plan before
  phase 5; the budget setting stops a run before it runs the quota dry.
- **Recall.** A fixed recipe misses what an agent would dig up (dates only in an
  archive, emails behind a contact page that needs a click). The maybe list and
  the small agent step cover the nearly-there ones; the comparison will show the
  gap.
- **Sites that block fetches** (403s seen in the night's run): such a firm is
  dropped with "page not readable", not retried at length.
- **LinkedIn snippets** aren't always in results. Then the team page decides and
  the row cites it; if neither says, the firm goes to the maybe list.
- **Lists that aren't firms** (people, products, papers): the recipe's fetch and
  decide parts are general, but phase 1 is tuned on firms. Other kinds are
  checked in phase 5, or stay on Bat.

## 10. Open questions

1. Run as a Bat engine (shared shell, §3), or a fully separate runner? Recommended:
   the Bat engine.
2. Which search provider and plan the Pi uses, and how many searches a run may
   spend.
3. Leftovers by default: a small agent step (more rows, more tokens) or none?
4. Ledger recheck window: 90 days?
