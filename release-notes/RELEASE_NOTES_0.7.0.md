# Captain Claw v0.7.0 Release Notes

**Release title:** Code — an engineering department, not a coding assistant
**Release date:** 2026-07-03

0.7.0 ships **Code**: a full agentic coding system inside Flight Deck. Describe what you
want built; a router sizes the job; small edits go straight to a specialist, big jobs run a
**plan → your approval → build → independent 3-reviewer fan-out → triage → capped fix loop**
pipeline — with every phase committed to a real git repo, a persistent **Code Map** the
agents query instead of re-reading your tree, live token/cost accounting, and a one-click
process export. It works on fresh VFS folders **or your existing local repos** (linked in,
read-write or read-only), on **any model you configure** — including a mixed fleet where
DeepSeek does the typing and a reasoning model does the planning.

Around the headline: a set of agent-core hardening fixes that came out of building Code
against real runs — a prompt-cache fix that cuts input-token cost ~4-5× for *every*
long-running agent, destructive-write protection, stuck-loop valves, and honest per-turn
cost reporting.

Everything is additive and backward compatible with 0.6.5.

---

## Code — the headline

### What it is

A dedicated **Code** page where projects contain **folders** (each one a real git repo and
an agent workspace) and **sessions** (conversations that drive work in one folder). You
talk; a fleet of specialist archetypes does the engineering:

- **Router** (fast tier) sizes every request: a quick edit goes straight to a single
  specialist (`quick-dirty`, `code-implementer`, `debugger`, or `git-operator`); a real
  feature enters the full pipeline.
- **Planner** (`light-planner` or `architect`, reason tier) surveys the repo and writes a
  concrete, ordered implementation plan — into `.plans/<timestamp>-<slug>.md`, so every
  turn's plan is preserved, never overwritten.
- **You approve** — the plan lands in an editable gate. Edit it, approve it, or discard it.
  Nothing is built without your sign-off.
- **Builder** (`code-implementer`, coding tier) implements the plan in the repo — real
  shell, real dependency installs, real test runs.
- **Three independent reviewers run in parallel** — a code reviewer (correctness, edge
  cases, regressions), a security reviewer (CVSS-ranked findings), and a QA engineer that
  **actually executes the test suite**. None of them wrote the code they're judging.
- **Triage** (reason tier) reads all three reports and makes a ship/fix decision on
  blocking/major issues only — style nits never trigger a fix round.
- **Fix loop** (capped at 3 rounds) sends precise fix instructions to a `debugger` or
  `code-implementer`; each fix is re-reviewed as a **delta** (the fix diff + prior
  findings), not a full re-read. At the cap, open findings persist to
  `.reports/backlog.md` and **"continue fixing"** resumes exactly there.
- **Cartographer** (`code-cartographer`) refreshes the Code Map when the dust settles.

Every phase is a git commit (`[plan]`, `[build]`, `[review rN]`, `[fix rN]`) in the
folder's own repo. The History rail shows the timeline; any commit opens a colorized diff;
any commit is one confirm away from rollback.

### The Code Map — agents that stop re-reading your codebase

The single biggest waste in agentic coding is agents re-reading the same files over and
over. Code keeps a per-repo **Code Map**: a deterministic symbol skeleton (every
function/method/class with signature and `file:line` — Python via `ast`, JS/TS via a
focused extractor, other languages via universal-ctags when installed) in SQLite + FTS5,
plus an LLM-authored semantic layer (architecture overview, data-model map, UI map,
per-file purpose). It's **git-blob-hash gated** — only changed files are re-indexed, so
freshness costs pennies.

Every code agent gets a `codemap` tool (`overview` / `search` / `symbol` / `file` /
`models` / `ui`) that returns *pointers*, never source dumps, and flags stale entries.
The Map tab in the Code page renders the whole thing for humans — with search and a
Rebuild button.

### Escalation — small jobs that turn out big

The router will sometimes size a gnarly bug as a quick edit. The quick-edit agent is told:
if this is genuinely bigger than a focused change, **stop and say `ESCALATE:` with a
reason** instead of half-doing it. Explicit escalation, a failed run, or a burned iteration
budget all promote the task to the full pipeline: partial work is committed so the planner
sees real state, the reason lands in chat, and you get a proper plan at the approval gate.

### Cost you can see, runs you can stop

- Every turn reports **`N agent runs · X in → Y out tokens`** — inline on small edits, as a
  chip on plans, as a run-total note after builds.
- A red **Stop** button kills a running pipeline at the next phase boundary; partial work
  stays committed; a cleanup action sweeps any leftover ephemeral agents.
- **Export** produces a single Markdown transcript of the entire process — every prompt,
  tool call, narration line, review report, and token count. Auditable engineering, not a
  black box. (A complete real run — an RTS game with 185 tests — ships in
  `examples/code/`.)

### Your repos, your models, your machine

- **Linked folders**: point Code at any local repo (read-write or read-only) — no copying,
  no upload. Commits land in *your* repo's history with your git identity. `.plans/`,
  `.reports/`, `.codemap/` runtime artifacts stay out of your git via local excludes.
- **Model-agnostic per role**: the planner, builder, reviewers, router, and triage each
  resolve to *your* Library tiers. Run the whole pipeline on DeepSeek for a few dollars,
  put Claude on planning and reviews, or go 100% local with Ollama. No vendor lock-in at
  any layer.
- **Self-hosted, MIT**: your code never leaves your machine.

### Why this is different — Cursor, Lovable, Claude Code, Codex, OpenClaw, Hermes

The honest one-line answer: those are (excellent) **single-agent assistants or hosted
builders**; Code is a **self-hosted multi-agent engineering pipeline with independent
review and a human gate**. In detail:

- **Cursor / Windsurf (IDE assistants).** You drive, keystroke by keystroke, inside an
  editor; one agent writes and (if asked) re-reads its own work. Code inverts the loop:
  you review **plans and outcomes**, and the writing/reviewing/testing are done by
  *different* agents — the reviewer never grades its own homework. There's no IDE to
  install and no per-seat SaaS; it's a page in your own Flight Deck.
- **Lovable / v0 / Bolt (hosted app-builders).** Great for spinning up a hosted web app on
  their stack, but your code lives in their cloud, on their framework choices. Code works
  on **any repo** — a single-file HTML game, a Python service, your existing production
  checkout linked read-write — and every change is a commit in your own git history.
- **Claude Code / Codex CLI (single-agent coding CLIs).** The strongest comparison — and
  the clearest structural difference. A single agent, however good, is its own planner,
  implementer, reviewer, and QA in one context window, tied to one vendor's models. Code
  splits those roles across independent agents (three reviewers in parallel, a separate
  triage judge, QA that *runs* the tests), inserts a **mandatory human approval gate**
  before any build, and runs on **whatever models you choose** — including mixing a cheap
  implementer with a premium reviewer. You also get artifacts a CLI session doesn't leave
  behind: per-phase commits, plan history, review reports on disk, a queryable code map,
  and a full process export.
- **OpenClaw (open-source assistant gateways).** Same self-hosted spirit, different organ:
  gateways route one assistant across your channels and tools. They don't have an
  engineering pipeline — no plan gate, no independent review fan-out, no per-phase git, no
  code map. Code is what you'd bolt *onto* that class of system, already integrated with
  the rest of a fleet platform.
- **Hermes-style autonomous coder daemons.** Fire-and-forget agents that push code
  unattended optimize for autonomy over trust. Code deliberately keeps two human moments —
  the plan gate and the review verdicts — and makes everything in between inspectable
  (live phase log, per-agent token meters, stop button). Autonomy where it's cheap,
  oversight where it's expensive.

And one thing none of them have: Code is **part of a fleet platform**. The same VFS folder
your Vatra research team filled with market analysis is one click away from being a Code
project's context; the same Library tiers, archetype editor, memory layers, and Flows
drive everything. Research feeds code; code feeds research.

---

## Agent-core improvements (from building Code against real runs)

These came out of post-morteming real Code runs — and most of them improve **every**
agent, not just coding ones.

### Prompt-cache-friendly system prompts (~4-5× input-cost cut on long turns)

The system prompt embedded a live clock and humanized activity ages, re-rendered on every
tool-loop iteration — so the *first tokens* of every LLM call changed, busting the
provider's prompt-prefix cache for the entire conversation behind them. A measured coding
run hit **94% cache misses** on 24.5M input tokens. The system prompt is now **frozen per
turn** (re-rendered at turn start, byte-identical within the turn, TTL-bounded), so
long tool loops hit the cache on everything but genuinely new content. This benefits every
provider with prefix caching (Anthropic, OpenAI, DeepSeek, Gemini).

### Destructive-write protection

A debugger once replaced a 1,815-line game with a 14-line fragment via a full-file write.
Now: a write that would shrink an existing file (≥1 KB) by more than half is **refused
before touching disk**, with instructions to use targeted `edit`s — or an explicit
`overwrite=true` after re-reading, for intentional full rewrites. Code-mode writes aimed
at gitignored scratch (`saved/...`) are auto-relocated into the real repo tree.

### Stuck-loop valves that actually release

The list-coverage completion gate could block a *finished* agent forever when the
"evidence" it scanned for lives in files, not reply text — and its safety valve only
tripped on identical miss-counts, which oscillating counts dodged indefinitely. Both
valves (list-coverage and scale-reply) now trip on **no net progress** plus an absolute
per-turn block cap; successful `edit` calls count as evidence-on-disk; and code agents
skip the list/scale contract machinery entirely (their completion is judged by the
pipeline's reviewers, not reply-text scanning).

### Honest per-turn cost accounting

Live `turn_usage` broadcasts are cumulative; the accumulator now counts **deltas** per
dispatch and dispatches (not broadcasts) as "agent runs" — so the run totals you see match
your provider's usage dashboard.

### Read/token discipline for code agents

- Identical full re-reads of unchanged files are short-circuited to a one-line pointer
  (offset/limit ranges and `force=true` always pass).
- Review rounds 1+ are **delta reviews** — verify prior findings + scan the fix diff,
  never re-read the whole repo (reviews were 53% of tokens on the measured run).
- The security reviewer sits out delta rounds whose changes touch no security surface.

---

## Everything else in 0.7.0

- **Projects → folders → sessions** model with non-destructive migration from the flat
  layout; a **New Session modal** (pick/create project → pick/create/link folder → name
  the session) replaces the scattered inline forms.
- **VFS linked folders** — link any local directory into the VFS via a per-user registry
  (no filesystem symlinks), read-write or read-only, the same folder into multiple
  projects, with a server-side **Browse** picker. Plus **download-any-folder-as-zip**.
- **Per-turn conversation history** to every spawned agent — follow-up corrections
  ("don't use the browser, check the code") now actually reach the planner.
- **Markdown rendering with GFM tables** in Code chat; plan gate with **Discard**;
  **git-operator** archetype for commit/push/branch/merge requests in plain language.
- **Map tab** — browse the Code Map: overview, data models, UI map, symbol search, rebuild
  with live progress.
- **Stop / cleanup** — stop a run at the next phase boundary; sweep leftover ephemeral
  agents (including their data dirs) with one click.
- **`examples/code/`** — a complete real Code run (SW3, an RTS game with 185 tests): the
  game, every plan, all review rounds, the code map, the raw session trace, and the
  cost/usage screenshots.

## Compatibility & migration

- **Fully backward compatible with 0.6.5.** No schema changes; the Code page stores its
  state under `<vfs>/<user>/<project>/.code/`; existing flat code folders migrate in place
  on first load.
- New endpoints live under `/fd/code/*` (projects, folders, links, sessions, message,
  plan approve/cancel, stop, cleanup, diff/show/rollback, map, export) and `/fd/vfs/*`
  gains link admin, browse-fs, and download-zip.
- New env markers `CLAW_CODE_AGENT` / `CLAW_WRITE_DIRECT` are injected only into
  Code-mode ephemeral agents; classic chat, Basna, Vatra, Council, and Flows behavior is
  unchanged (regression-audited against the 0.6.5 baseline).
- The `coding` tier (added in 0.6.5) is where Code resolves its implementer/debugger
  models — point it at your preferred coding model; `reason` backs planning/triage,
  `fast` backs routing.
