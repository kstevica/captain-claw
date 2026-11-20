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

The third run read firms' own sites, but kept 0 of 39 in ten minutes: Bat's own
shaping call looped with thinking on (137k characters), so no goal became a check and
the list file fell back to `results.csv`; and the LinkedIn searches brought firms from
Canada, the UK, Morocco, the UAE, New Zealand and India, each read before the judging
call dropped it. Fixed: shaping runs with thinking off and an 8k limit; the list file
comes from the charter when nothing else named it (also on a recipe written before);
for a US list every search names the place; each item keeps its own search result, and
one whose listing names a place outside the US, and nothing American, goes before any
page is read.

Then made general (any place, any kind of item): the target is any country the
charter names (by name, short form, big city or province, a US state; by demonym only
when it names no place, and no guess when it names one the table doesn't know), and a
search names it when it doesn't (the engine's own country is set only where it takes
one, e.g. not Croatia); a listing that names another country and none of the list's is
dropped before reading, for any target. Fallback searches are neutral (lists,
directories, databases, rankings), the item is read from the charter's own ask
("Find 40 vegan restaurants in Zagreb" → "vegan restaurant"), a site lookup searches
the name with the item and the place, and a directory or social site isn't skipped
when it's the list's own kind of page (a "LinkedIn URL" column).

The fourth run shaped its checks in 11 seconds, then dropped all 31 items it found:
shaping had turned "State is two letters" into an exclusion pattern that matches every
value that *is* two letters, and the recipe ruled out firm names with "agency" on a
list of agencies. Fixed: an exclusion pattern (in shaping, at check time, and in a
recipe's filters) is tried on a set of ordinary values first (a firm name, a site, an
email, a date, a city, a size band, a person's name) and dropped when it rules out half
of them, any value the checks allow, or a group's own name (its singular too), or when
it doesn't compile.

The fifth run's recipe slipped one past that test: "keep only what starts with two
letters or a date", tried ignoring case, matches few ordinary values, yet it dropped
firms named "P&S …", "U.S. …" and "C-Suite …". An "everything except X" pattern (one
that starts by looking ahead for what it must not see) is now never an exclusion. The
same run dropped two firms as "201-500 people" read off LinkedIn News: with no site of
their own found, their LinkedIn page stayed their site, and the recipe's pages ("about",
"news", "contact") were read off LinkedIn's root. On a directory or social site only the
item's own page is read now. And a size counts only from the item's own listing, its own
pages, or a search result whose address and title carry its name.

## 8c. Rule tiers: rows on a looser reading when a group runs short

The run after those fixes read 67 firms and kept none: almost half failed one rule,
"published something dated about AI in the last 24 months", which most small firms
never put on their own site. Rather than an owner hand-writing a looser rule, the setup
model sorts the owner's rules for each task (recipe `rules`):

- **must**: what makes an item the right kind of thing (type, place, size range, the
  owner's exclusions) and what the list is useless without (a way to reach it on a
  contact list, the price on a price list);
- **prefer**: evidence many good items never publish (recent activity, dated news, a
  signal, a rating, a certification), each with a **looser** reading (a wider window,
  any mention undated, another kind of page, the column left empty) and the columns
  it fills. Never anything the owner wrote as "only", "never", "exactly" or an
  exclusion, nor a safety rule.

A judging call may then answer **loose**: every must met, and each soft rule it misses
met by its looser reading. Code checks still run, except "required" and date windows
on the loosened rules' columns; a value given there must still be on its page. Loose
rows wait in `shared/funnel/loose.csv`. A group counts them once it has looked at 3
items per row wanted (at least 30), so a strict row still has its chance. At the merge,
strict rows (the funnel's and the finishing step's) go first; loose rows fill only what
is still short, those with fewest rules loosened first. `summary.md` lists them, the
judge's note says they aren't faults, and the finishing agent isn't started when loose
rows already fill what's short. A factory setting (runner config `loosen`: `short`,
the default, or `never`) turns it off; the run's first notes name the tiers it read.

The same run also showed that one slow call (a model or a tool call that got no answer
in time) ended the whole step, and Bat took the error for "too big for one turn" and
split the step into agent steps (the fix round burned 235k tokens in two minutes). A
slow or broken call now skips what it was for (those results, that item) and the step
goes on; a step that still stops says so in words Bat doesn't read as a size failure,
and picks up where it left off on its next try.

The first run with rule tiers sorted the charter as meant (only the dated AI signal
soft), but 7 of its first 16 drops were firms whose site lookup took someone else's
page for theirs: a Forbes article, a marketplace, a contact-list site, a directory,
each a result whose title named the firm. A result is now an item's own site only when
its address names the item (a distinctive word of the name in the domain, or its
initials), accents aside and either spelling ("Đurđević" → durdevic or djurdjevic, "Müller" → muller or mueller); a name with no Latin letters is matched by the result's title instead. With none found it has no site, and its judging call says so. The lookup
leaves out a long item phrase ("US knowledge firm lead"), and more contact-list and
marketplace sites are never an item's own.

When code writes a group's searches (the model's never named the group, or copied
another's), a slashed group is searched by each of its parts ("research/analysis shops"
→ "research shops", "analysis shops"), and a vague group noun also by the run's item
("research firms"). A model's own searches count as naming a group when they use a
common word for it ("law" for "legal", "CPA" for "accounting").

Unshown is not broken. Two runs dropped most firms for "no employee count shown":
the charter asked for a size "shown on LinkedIn or its team page", the setup kept it a
must, and not finding it was read as breaking it. Now, for any list:

- the setup splits a rule that asks for a fact and the page it must be shown on into
  the exclusion (a must: "not shown to have more than 50 people") and the shown proof
  (a prefer, looser "stated on any page, or not shown");
- in code, a must about something measured (a count, an amount, a date, a rating) may
  be met by "not shown" on a looser reading even when the setup kept it a must; a
  must about what the item is, where it is (a place) or how to reach it (an email, a
  phone), or about the key, group or site column, never is. A judging call that names
  only such rules for a loose row makes it maybe;
- the judging call is told which rules never loosen, and that an item shown to break
  a rule is still dropped (a stated 120 people fails the size check as before).

The run after that split the size rule as meant, then dropped 226 of 264 firms before
reading them: the setup had added a filter "Segment is one of the four segments", and
what a listing says ("Boutique consulting firm") never equals a fixed value exactly. A
filter before reading never holds a list column to its fixed values now (the row's
check does that after judging); an "in" filter on another field passes when the text
shares a word with a value it may take; numbers in free text are read as ranges ("11-50"
was read as 1150). And the judging call kept answering maybe for "meets every rule;
employee count not shown": with looser readings on, a maybe names the rules whose facts
weren't shown, and code makes it loose when each of them may go unshown.

The next run still dropped firms for "no dated AI signal found": the judging call
called a missing soft fact broken. A drop now names the rule it breaks and copies the
words read that break it; when every rule it names may go unshown and the words it
quotes aren't in what was read (or it quotes none), code treats it as a missing fact
(loose with a row, else maybe). A drop that shows its fact stays a drop.

That run (no cap on its factory) also showed what still needs bounds: after the funnel
steps found nothing, the finishing agent used about 6 million tokens in 45 minutes and
was retried after it timed out, and Bat's fix rounds then ran agent steps of 3 to 16
million tokens each, retrying those that came back short, to 54.7 million in all for
20 rows. Now bounded, for every Bat run: one with no token cap stops at the deck's
safety ceiling (20M, `CLAW_BAT_TOKEN_CEILING`, counted from where it was last
continued) as a cap would, so it can be continued; an agent step that uses its share
(1.5M in a list funnel's run, a third of the ceiling in a run with no cap,
`config.step_token_limit` to set it) is stopped and done with what it made; and a list
funnel's agent steps (finishing, fix rounds) that time out or come back short are kept
as they are, never run again. The exclusion guard now reads every check on the same
file (an "agency" pattern from the exclusion check had flagged a marketing agency because
the segment names live in another check). The finishing agent reads the rule tiers in
`shared/conventions.md` and may add rows on a looser reading (named in a `_loosened`
column; checked with those rules' columns relaxed; they wait with the funnel's loose
rows). And on the owner's word, a rule about how to reach the item loosens too: its
looser reading is a contact page or form on the item's own site, the address left
empty, never an address guessed or taken from a contact-list site; an address given
must still be on the page it cites.

The run after that built a complete list: 40 of 40 rows in 24 minutes for about 0.7M
tokens, every row on a looser reading (12 with an email printed, the rest reached by
their contact page). Then Bat's reviewers judged it against the charter as written,
and fix rounds started deleting its rows to hunt strict ones (each stopped at its 1.5M
share by the token guards). Now a funnel run whose list is complete and passes every
check by code is done when the reviewers disagree; their points go into `summary.md`
under "Open points from the review". A list still short goes on to fix rounds as
before. The same run left CaptainFeature and CaptainReason empty in 33 of its 40 rows
(a reason "tied to the firm's signal" was skipped when no signal was shown): columns
about the owner's own offer (named after a reference file, or with fixed values that
come from it) are always filled now; a row that leaves one empty is nearly there, not
on the list, and the judging prompt says to tie it to what was read.

Made generic (2026-11-20). The rule tiers and their safety nets had leaned on English
word lists and on company lists. The setup model now tags what code needs, in any
language: each rule's `fact` (identity, place, measured, contact, safety, other: only a
measured or contact rule may be met loosely), `group_words` for each group (the words
pages and searches use for it; the hand-made synonym list is gone), columns filled
`"from": "owner"` (from the owner's own material), and `skip_sites` for its kind of
list. The English word lists stay only as a fallback when a tag is missing. A rule's
columns are the ones the setup named plus the ones its own words name (never the key,
group or site column), so a column it left out isn't required on a loose row. The
setup's example recipe is neutral (no LinkedIn search; pages chosen for the kind of
item), and the fallback reads a team page only when a company's size is asked for.
Columns from the owner's material are filled by one short extra call where a judging
call left them empty; a row is nearly there only when that call leaves them empty too.
Matching a fixed value's words in the reference no longer marks a column as the owner's
("personal" and "shared" are in any text).

The first run to end by itself (40 of 40 rows in 81 minutes, 1.03M tokens) was then
audited row by row against the firms' own pages: about 12 clean, 15 usable once fixed,
13 breaking a rule that never loosens. What the engine got wrong, and what now holds
for any list:

- Facts read were left out: 11 firms had an email printed on their own pages, 9 a
  state on a page read. Cells a judging call leaves empty are now filled from the
  item's own pages: its email (on its own domain, with the page and whether it's a
  shared inbox), its phone, its US state from an address (never from a state's name
  alone: "Washington DC" isn't Washington state).
- A size the pages stated was taken for "not shown" ("more than 100 lawyers and
  staff"): a head count read on the item's own pages over the size limit drops it
  before judging (the top of a range like "51-200" stays a size band).
- Headquarters abroad were checked only in the listing: an item whose own pages point
  only to another country (a country named, its site's or its addresses' country
  domain, a phone country code; +1 and generic endings say nothing) is dropped before
  judging.
- The judging call didn't know the date and called 2026 "the future": it gets today's
  date. A charter's "in the last N months/years" holds the list's date column to that
  window whatever check the setup wrote.
- "Lux Insights Inc." and "Lux Insights" were two items across runs: a company's legal
  form (Inc., LLC, Ltd., GmbH, d.o.o. and others) never makes two items of one name.
- 29 of 40 reasons were template sentences: the call that fills columns from the
  owner's material now gets a little of what was read about each item and is told to
  cite it.
- A never-loosened place or identity rule with nothing shown for it (no State under
  "headquartered in the US") leaves a row nearly there.

Not fixed: segment calls stay the judging call's (two misfits got through; Jev's
choice questions are an option), and founder names are not checked against roles.

A different list (Croatian charter: restaurants in Zagreb, three cuisines) showed a
group's step dropping an item that fit another group of the same list ("Italian, not
Croatian" in the Croatian step). Now: the judging prompt says an item that fits another
group is that group's find (kept, loose or maybe with that group named); a group named
loosely ("Italian") is matched to the list's group by its words and the setup's group
words and written as the list names it; a maybe goes to the group it fits; and a drop
whose reason names exactly one other group is handed to that group's step
(`shared/funnel/handover.csv`), which judges it first (or, if that step is done, the
finishing step sees it as nearly there). Words every group shares ("kuhinja" in each
cuisine) name no group. A drop that also broke another rule (a rating shown too low)
stays a drop: the other group's step would drop it too.

The same run showed three more general faults, fixed:

- **Evidence from one page only.** A value whose check names a source column (the Web
  column) was looked for only on the page that column cites, so an email on the
  item's contact page failed when the row cited its homepage. Now any page read on
  the item's own site counts (not a directory or social site).
- **Addresses outside the US were never filled.** Code filled a "place" must (a city,
  a country) only through the US state lookup. Now the shortest line on the item's
  own pages that has a number and names the rule's place (or the run's target
  places) fills the address column, or the rule's first column.
- **A place in another grammatical case wasn't recognised.** "u Zagrebu", "u
  Hrvatskoj": a place name with a case ending (-u, -a, -om, -oj, -ama…) now matches
  the place, so the run's target country and a rule's place are found in any
  inflecting language, not only in the nominative.
- **One item twice under two names.** "Konoba X" and "Konoba X 900": the same website,
  address, phone and email, but duplicates were told only by name. Now an item's own
  site says who it is (`site_id`: the home page's host, a language part aside; a
  deeper page's host and path; on a shared social or directory site, the item's page
  there). A row with the same site, or the same host and the same phone or email, is
  the same item, unless an address or phone column differs (two branches of one
  chain stay two). It is dropped as a duplicate when judged, and skipped at the
  merge, when loose rows fill a group, when the finishing step adds rows, and in the
  came-close file.

The next run of the same list showed the duplicate check too strict, and other faults:

- **An address written two ways isn't two branches.** "Ilička ulica" and "Ilička",
  "Lipova 10" and "Ul. Lipove 10" (another grammatical case) made one restaurant two.
  Now two rows are apart only when both give a phone and the phones differ, or both
  give house numbers and none is shared, or both give street words and none is shared
  (words compared by their first four letters; postal codes and city names left
  out). A word of the name shared with the same phone or email, or with the same
  address, is the same item too, even when one row cites a guide's page.
- **A cell left empty isn't a rule broken.** Twelve items were dropped for "Adresa is
  empty", "Web is empty", "Preporuka is empty". Now a required column left empty
  makes the item nearly there (the finishing step and the came-close file get it),
  or loose when the column belongs to a soft rule.
- **The address was filled only when every cell of the place rule was empty.** A
  district named left the address empty. Now each empty address column is filled;
  a phone column in any language ("Telefon") is filled from the item's own pages.
- **A source column holding a name.** A review count was checked against "Google", a
  name, not a link, so every row with a review count failed. A source that isn't a
  link names no page: anything read counts.
- **Phones written another way.** "+385 1 …" against "01 …" on the page: phones are
  compared by their last eight digits.
- **A size filter from a limit on locations.** "Not a chain with more than 3
  locations" became a staff-size filter of 3. The setup prompt says size filters
  count people only.
- **Handed over for the wrong reason.** The Italian step handed a hotel's restaurant
  and a fast-food place to the Croatian group: "restoran" and "food" were words of the
  Croatian group's phrases. Now a text names a group by the stems of the group's own
  name or by one of its phrases whole, and a drop that broke any rule other than the
  one about the groups (not a hotel's restaurant, not fast food, a rating) stays a drop.

That run ended by itself (24 rows, 22 on the looser reading). Its reviewers named two
more faults, fixed:

- **The charter's file names.** It asked for `sazetak.md` and `iskljuceni.md`; the run
  wrote `summary.md` and `exclusions.md`. The setup now gives the names the charter
  uses for the summary and the dropped items (`files`), and the merge writes them
  there (a name outside the run's folder, or not a text file, is ignored).
- **The came-close file's group column** kept a page's word ("Sushi"); it is now the
  list's group.

Only 1 of 24 rows had a rating, though the setup searched each restaurant's rating:
the 11 that came back with a rating and review count were the ones dropped by the
named-source check fixed above. The run's results also show the list itself first
now: a file a table or document check reads is what the run was asked to deliver.

The next run (record of earlier runs forgotten) built 24 different restaurants in 16
minutes, 7 with a rating, under the charter's file names. Three general faults left:

- **One unproven value dropped the whole item.** A phone cited from an Instagram or
  Facebook page Captain can't read, a review count not in anything read. Now a value
  what was read doesn't bear out is left out: a fact the item's own pages show takes
  its place (their phone, not a wrong one); a soft rule left with nothing shown is
  met by its looser reading; any other cell left empty makes the item nearly there.
  Never a kept row with a value that wasn't read, nor with a cell the list needs
  left empty.
- **A place in an item's own name was taken for where it is.** "Royal India" was "not
  in Croatia (its listing says India)". Places its name has in it don't count, on its
  listing or on its own pages.
- **What one row is read as a sentence.** "a restaurant in Zagreb suitable for a
  business lunch", cut at 40 characters, made notes like "the same a restaurant in
  Zagreb suitable for a bu as …". Now the leading article and everything from "in",
  "for", "suitable", "with"… go ("restaurant"), and a long one is cut at a word.

That run's reviewers said 18 of its 24 rows met the rating or contact rule only by
its looser reading, while the charter accepts a short list when the summary explains
it ("8 per group, or sazetak.md explains the shortfall"). Now the setup says so
(`shortage_ok`) when the charter itself accepts fewer rows with the reason given;
then no rule is read loosely (as with the factory's "never loosen"), whatever the
factory's setting: what meets a rule only loosely comes close and goes to the
came-close file, the summary's "Short" section says the charter accepts it, and a
short list settles the review as a complete one does, so no agents are sent to fill
it.

The next run (rules kept as written) found 3 of 8 for its first group after 67
items. Most that came close lacked a visible rating, which is the charter's
strictness. But two general faults showed too, fixed:

- **An item's own site wasn't read when the listing gave it as its link.** Items
  pulled out of a page carry their site as "url"; the recipe reads "{Web}" (the site
  column), so the template came out empty and nothing was fetched, and the "find its
  site" search didn't run since a site was known. 13 of 25 near misses had a site and
  no page of it read. Now the site column is filled from wherever the item's own site
  is before anything is read.
- **The same item judged several times.** "Konoba X" was kept, then "konoba-x.hr"
  (the same restaurant, keyed by its domain) judged again. Now an item is checked
  against those already looked at, as their listings show them (the same rules as
  the duplicate check: the same site, no address or phone that differs), before it is
  read or judged, and again once a search finds its site; a repeat is dropped as a
  duplicate, with the item it repeats named.

Items that came close were only in the run's working files (`maybe.csv`). Now the
merge writes `<list>-came-close.csv` next to the list: the list's columns as far as
they were found (the judging call's partial row; a value not in what was read is left
empty), then why it came close and the pages read. Rows on a looser reading that
the list didn't need come first in their group, then the items a fact the rules need
wasn't found for. Nothing on the list and nothing later dropped is in it; with
nothing that came close there is no file. The run's results show it as a group of
its own ("Came close"), and the reviewers are told it isn't part of the list.

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
