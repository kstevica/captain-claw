# Captain Claw v0.7.3 Release Notes

**Release title:** Quality Profiles, Run-Cost Accounting & a Collaborative Vatra
**Release date:** 2026-07-07

Get **near-frontier quality out of weaker and local models**, see **exactly what a
run costs** (in dollars *and* in human-wage terms), and run **Vatra as an ordered
team** instead of an all-at-once scramble. Everything new here is **opt-in and
default-off** — with every quality lever off and no cost card configured, a run
behaves byte-for-byte like 0.7.2. Additive and backward compatible with 0.7.2;
**restart Flight Deck** to pick up the new backend.

## Highlights

### Quality Profiles — one envelope of opt-in levers, shared across Code and Basna/Vatra

A single **Quality** control on the Basna/Vatra and Code surfaces, with four
presets — **Basic** (today's behavior), **Balanced**, **Thorough**, and
**Custom** (toggle any subset of ~11 levers). The premise: a strong model fills
gaps silently; a weaker or local one needs the scaffolding made explicit. These
levers add that scaffolding — grounding, verification, structure, and retries —
so a cheaper model reaches much closer to frontier output. All are off in Basic,
so nothing changes unless you opt in.

- **Intent brief (R12).** Before the team is chosen, the task is restated as one
  faithful, structured brief (objective / in-scope / out-of-scope / deliverable /
  constraints). The brief drives **team selection** and every worker's framing —
  and it's **editable**: review it, re-route on it. The original request always
  governs on conflict, so a rephrase can never silently change the goal.
- **Grounded claim verification (R8) + honesty guard (R8+/R11).** A tool-enabled
  fact-checker reviews the deliverable, unconfirmable-but-asserted specifics are
  **hedged** rather than stated, and a non-destructive **fact-check audit ledger**
  is written alongside the output. A **judgment ledger** forces the team to
  enumerate its hard calls instead of burying them.
- **Rubric-from-source contract (R9).** The completeness checklist is derived once
  from the standard the task names, injected into every owner and the reporter as
  the definition of "done," and the deliverable is scored against it — surfacing
  what's missing or thin.
- **Source corpus (R10).** `web_fetch` saves the **full** page text into the run's
  VFS folder and hands the agent a lean head + pointer, so context stays small
  while the whole source remains searchable. Shell-`curl`-of-web-pages is steered
  off (it bypassed the corpus and blew up local context windows).
- **Research Map (R1).** The Code Map, generalized to any VFS research folder — a
  searchable index of everything the run has written, so later rounds query prior
  material instead of re-reading it.
- **Acted-gate & escalation (R2/R5).** A worker that produced nothing gets one
  corrective retry; a worker that flags `ESCALATE` gets one focused retry — both
  under a per-run token budget. **Git snapshots per round** and **budget parity**
  round out the safety rails.
- **Code levers (C1/C2/C3/C5/C6).** A **test gate** in the build→fix loop, Code
  runs **fed into archetype-reliability learning**, an opt-in **deep-build ladder**
  (best-of-N verified builds), **plan-coverage → backlog**, and **continuation
  lineage** so "continue" resumes exactly where a run stopped.

### Run-Cost Accounting — what a run costs, next to a human wage

Every finished Basna, Vatra, and Code run now shows a **Run cost** card:

- **Dollars + effective $/hour.** Token spend is priced from a curated per-model
  table (`captain_claw/instructions/model_prices.json`, per-million input / output
  / cache-read / cache-write), and the run's spend is turned into an effective
  **$/hour** you compare directly to a wage you enter ("→ 38× cheaper per hour").
  Cache reads/writes are priced separately (Anthropic convention); an unknown
  model shows tokens with `—` for dollars rather than a wrong number.
- **Time, honestly.** **Wall-clock** (what you waited) alongside **agent-time**
  (the sum of every model call's duration) and their ratio as **parallelism**
  (e.g. `2.2×`) — so parallel runs show how much the concurrency compressed the
  work.
- **Token split.** Input (fresh, incl. cache writes) · cached (reused) · output
  (generated), plus a per-model breakdown.
- The card lands on **live finish**, not only on reopen (a race that hid it on the
  background Vatra path is fixed), and the human-wage number is a single global
  preference, so every surface compares against the same rate.

### A collaborative Vatra — ordered phases, real teamwork, and attachments that actually get used

- **Execution groups (opt-in).** Instead of every owner running at once, Vatra can
  run in **ordered phases A → B → C → D** with a barrier between them —
  research/design first, build/write in the middle, review/assembly last — so a
  later phase already has everything earlier phases posted. Archetype presets pick
  the phase; the Lead may push a piece **later** (never earlier). A **bounded
  clarification loop** lets a later agent ask an earlier one for a missing input,
  gated by the Lead (capped at 2 per run). The **live panel sections the working
  agents by phase**, so you watch A finish before B starts.
- **Attachments are actually used.** A Vatra run previously ignored attached files
  entirely (the run step never uploaded them). Now files upload on **Run**, are
  **saved into the run's VFS folder** (browsable, indexed), and — when the intent
  brief is on — a short-lived **file-examiner agent opens them** and briefs the
  whole team, so workers act on what the files *contain*, not just their names.
  **Plan team** no longer drops a pending attachment either.
- **Reliability fixes.** A phantom "stuck" agent from the `wait` rendezvous is
  gone; a blank **"Vatra Lead failed:"** now gives an actionable message (and the
  planning timeout was raised for slow local models).

### Local-model resilience

- **Max parallel agents (user-selectable).** Cap how many agent turns run at once
  (0 = unlimited) so parallel prefills don't blow past a local serving box's
  memory. A per-run semaphore threads through every dispatch.
- **Failures preserve work.** An agent-side LLM error (e.g. context overflow) no
  longer discards the turn — the accumulated actions and partial output survive,
  and the dispatch shows the **real reason** instead of a mute "0 actions."
- **Less redundant overhead for orchestrated workers.** Flight-Deck-spawned
  workers (Basna/Vatra/Council/Code) skip the per-agent **task-rephrase** and the
  post-turn **"suggested next steps"** extraction — the orchestrator already
  framed their task, and they have no interactive user to prompt.

### VFS — a project list you can actually use

The Shared VFS project list is now **searchable and recency-first**, with
**type-filter chips** (Basna / Vatra / Council / Linked / Projects) and a sort
control. Auto-created run folders with no human title **fold away by default**
behind a "show N run folders" toggle, so your named projects and titled runs lead
instead of drowning in `vatra-…`/`basna-…` hashes.

## Notes

- **Fully additive, default-off.** Basic quality + no configured price table = the
  0.7.2 experience. Grouped Vatra, max-parallel, the intent brief, and every R/C
  lever are opt-in per run.
- **Pricing is a single-file reprice.** Edit `model_prices.json` to change or add
  rates; a Library tier can also pass its own `price` override.
- **Restart Flight Deck** to load the new backend (cost accounting, execution
  groups, file-examiner, VFS list). The frontend bundle is rebuilt and committed.
- Backward compatible with 0.7.2. No breaking schema changes.
