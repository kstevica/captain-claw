# Bat — the stubborn finisher

**Status:** design / not started · **Date:** 2026-10-07 · **Owner decisions locked** (see §1)

Bat is a new Captain Claw orchestration mode, a sibling of Basna, Vatra and Council.
Where Vatra decomposes-and-delegates once and reports, Bat is a **durable, long-running loop that
keeps working a task until an independent judge says it is genuinely done** — across retries,
strategy switches, chunked attempts and FD restarts. Agents invoke it as a tool, the same way they
reach Vatra today (via the `basna`/`vatra` family).

Name: *bat* (Croatian for a wooden mallet / cudgel) — short, in the existing Balkan-word family
(Basna, Vatra, Iskra, Mrav, Lupa), and it carries the "keep hammering" meaning. Verified unused in
the repo.

---

## 1. What "stubborn" means here (and what it does **not**)

The owner's ask was a mode whose agents may "break almost all rules to finish the task, almost no
matter the cost," forbidden only from harming the system. We deliberately **do not** implement that
as removing safeguards, for two reasons:

1. The incidents this platform already has — the PR #56 unrequested-mail cannon, the 623K-token
   stuck minesweeper loop — are exactly what "remove the guards" produces.
2. The mapping showed the "system-harm floor" barely exists today (see §3). There is almost nothing
   to relax; there is a floor to **build**.

So "stubborn" is re-expressed as four concrete, bounded properties:

- **Persistence, not permission-stripping.** A long supervisor loop with raised iteration/stall
  caps, retries with backoff, strategy switching, chunked ≤1h attempts, and continuation rounds in
  one VFS folder. Bat does not "give up" on the ordinary stuck/stall/empty-answer exits — but it
  stops hard on the serious conditions in §7.
- **An un-gameable "done" judge** (§4). "Finish at any cost" otherwise rewards *faking* done
  (stubbing a result, shrinking the goal, claiming success). The done-verdict is a separate,
  fail-closed check the working loop cannot overrule.
- **Capabilities granted as bounded + audited, never as "off"** (§5–§6): email through the existing
  audited send gate; spend through a new pre-authorization ledger with a hard cap; real logins and
  new accounts, with real credential encryption and recorded purchases.
- **Pause-and-ask escalation** (§7) for the genuinely-can't: a 2FA/verification code, a CAPTCHA, a
  payment over cap, a credential Bat doesn't have. These block on a durable human-ask and resume on
  the answer. **Bat does not solve CAPTCHAs or run anti-bot evasion** — it asks.

### Owner decisions (locked 2026-10-07)
1. **Escalation:** pause and ask the human (durable ask, timeout, resume on answer).
2. **Spend cap:** pre-authorization ledger; under a small per-item cap Bat proceeds, above it the
   human approves; running total enforced across restarts.
3. **Start gate:** auto-start pure research/build runs; **gate (show plan + acceptance contract and
   wait for Go) whenever the plan involves sending mail, creating accounts, or spending money.**
4. **Home:** dedicated `bat.db` store + lease-based restart recovery + its own Bat page (not riding
   `basna_sessions`).
5. Email-as-the-user is an explicit, **run-scoped, human-origin-only** exception to PR #56; it never
   propagates to a child run.

---

## 2. The two invariants that make the rest safe

Everything else in Bat is allowed to be aggressive **because** these two hold unconditionally.

### Invariant A — the hardened system-harm floor (§3)
A single deterministic denylist, enforced at the universal tool chokepoint for *every* Bat tool
call (shell, terminal, termux, desktop_action, code), that **cannot be widened by any Bat flag**.
Relaxation levers in Bat raise iteration/retry caps and loosen *deliverable-integrity* checks only;
they can never reach this floor.

### Invariant B — the honest done-judge (§4)
No run is marked `done` unless the judge says done. The judge is fail-closed (unparseable / timeout /
error = NOT done), reads **evidence** (files, command output, verifier findings, receipts) rather
than the worker's narrative, runs on a model different from the workers, and validates against a
contract pinned outside worker-writable space.

---

## 3. Invariant A: build and harden the system-harm floor

**Finding (must-fix regardless of Bat):** the current shell guard blocks only `rm -rf /`, `mkfs…`,
the fork bomb, plus deny-patterns `rm -rf *` / `sudo *`. In headless/worker contexts `ask` silently
degrades to `allow` (no blocking approval callback). The `terminal` tool (PTY to the user's real
Mac), `termux`, `desktop_action` and `code` have **no** destructive-command checks at all. The
`blast_radius` classifier is the closest thing to a real catalogue but is default-off and scans only
command-shaped arg fields.

**Plan:**
- New module `captain_claw/flight_deck/bat_floor.py` (or extend `blast_radius.py`) with a
  comprehensive, deterministic **deny** classifier. It must cover, at minimum, the command classes:
  recursive/forced delete in all spellings; filesystem make/format (`mkfs.*`, `newfs*`, `diskutil
  erase*`, `fdisk`); raw device writes (`dd of=/dev/*`); unmount of real volumes; `find … -delete`
  over broad roots; `shutdown`/`reboot`/`halt`; `kill -9 -1`/killall of critical procs;
  `chmod/chown -R` over `/` or `~`; `launchctl`/service teardown; and inline-interpreter equivalents
  (`python -c`, `perl -e`, `node -e`, `ruby -e`) that call `os.remove`/`shutil.rmtree`/`unlink`/
  `rmdir` on broad paths, plus `base64|sh`-style obfuscation. This is a *defensive denylist*, scanned
  before execution.
- Wire it **unconditionally** into `ToolRegistry.execute` / `AgentGuardMixin._execute_tool_with_guard`
  for Bat workers (marker-gated), applied to shell **and** the sibling exec tools. A denied call is a
  hard refusal, recorded as a §7 serious event, never an "ask that becomes allow."
- The floor is independent of `config.guards.blast_radius.enabled` and of any Bat relax flag.
- `write_guard` stays fully **on** for Bat (it only protects deliverable integrity; no system-safety
  cost to keeping it).

Ship this floor first; the rest of Bat depends on it.

---

## 4. Invariant B: the honest done-judge

New pure module `captain_claw/flight_deck/bat_judge.py`, composed from existing, standalone pieces
(model calls injected as async fns, like `run_check`/`run_gate`):

- **Layer 1 — deterministic, fail-closed:**
  - `deliverable_manifest.gate()` / `assemble()` — file present, not placeholder, big enough, enough
    sections; no `part_missing`/`placeholder_part`/`missing_chapter`.
  - `code_verify.run_tests()` through the hardened Bat runner — repo tests are ground truth.
  - `code_contract.validate()` / `research_contract.validate()` (safe predicate eval, no `eval`) —
    acceptance checks. **Contract is derived once at run start and sha-pinned in `bat.db`, not only in
    a worker-writable `.contract.json`.** `unclear`/`unresolved` counts as NOT done.
  - `research_consistency.verify()` — arithmetic/identity checks on document deliverables (no revise
    fn, so the judge never mutates the deliverable).
- **Layer 2 — independent judge panel:** N judges on a tier **different from the workers**, each
  returning a `VOTE: AGREE|DISAGREE|ABSTAIN` in the `vote.md` format, aggregated with
  `quality_profile.tally_votes()`. Build them from `dubina.reasoning.load_critic_modes` but **call
  the critics directly** — do not use `run_horizon_closer` (it revises the answer and treats a critic
  timeout as "sound"). A timeout/exception is a NOT-done vote.
- **Layer 3 — world-evidence verifier** (for side-effecting tasks): copy the `_claim_check` pattern —
  spawn an ephemeral **read-only** verifier (web-read/gmail-read/browser-read; no bat/basna/vatra,
  no write) that checks each acceptance item against real-world evidence (sent-mail receipts from the
  FD send audit, order confirmations, the created account resolving). Receipts recorded by workers in
  the facts ledger are treated as **claims to re-verify**, never as truth.

**DONE** requires: zero deterministic Layer-1 criticals **and** a Layer-2 `agree` verdict (margin
threshold / unanimity configurable) **and** Layer-3 evidence confirmed. Anything else → keep looping.

**Anti-deadlock:** if Layer 1 is green but the Layer-2 panel keeps refusing (or vice-versa) for K
rounds, Bat escalates to a human tie-break (§7) rather than burning to the cap. Mirror
`story_state`: only deterministic hard findings may *block* done; a model-only objection that recurs
unchanged with deterministic checks green is overridden after K rounds with a logged note.

Persist each verdict next to an optional human thumbs label (separate columns, mirroring
`basna_runs.success` vs `human_success`) so `label_eval.agreement()` can report judge-vs-human kappa
for Bat. Judge spend is metered via `_provider_call`/`_RUN_USAGE` so it counts toward the cap.

---

## 5. Capabilities, bounded

Each capability the owner authorized is implemented as *bounded + audited*, never as a disabled gate.

### 5.1 Email as the user (run-scoped exception to PR #56)
- Add a `bat` kind to `mail_authority.AUTOMATION_KINDS` **and** `KIND_LABELS` (the test asserts the
  sets are equal). Add `CLAW_BAT_WORKER` to the worker-marker families so a Bat worker fails closed
  to `deny` by process default — the **allow comes only from the per-turn automation frame**, never
  the process default.
- Bat dispatches carry `automation={kind:'bat', job_text:<human request>, mail_write:'allow'}` via the
  existing `_send_chat_and_collect(automation=…)` seam — **only** when the run was started by a human
  and only on the worker frames of that run.
- Actual sends still go through FD `POST /fd/google/gmail/send`, so the owner's allowlist, daily
  limit, dedupe, audit (`gmail_sends`) and bell all still apply. The direct `send_mail`
  (Mailgun/SendGrid/SMTP) path has no allowlist/limit/audit — **Bat is barred from `send_mail`**; mail
  goes through the FD Gmail gate only.
- `mail_write='allow'` must **never** be re-emitted to a child run. A scheduled/sub-Bat (non-human
  origin) gets no mail exception.

### 5.2 Capped real-money spend (pre-auth ledger)
No money-spending capability exists today (the beings "wallet" is simulated). Build, patterned on the
Gmail send gate:
- New `spend_authorizations` table + `bat.db` ledger: `{id, owner, run_id, agent, merchant,
  merchant_domain, description, category, currency, amount_native, amount_usd_max, status
  (requested|approved|denied|consumed|voided|expired|unknown), decided_by, actual_usd, order_ref,
  evidence JSON, expires_at, …}`. Committed spend = Σ `COALESCE(actual_usd, amount_usd_max)` over
  approved/consumed/unknown rows (count `unknown` so a timed-out purchase can't be double-charged).
- New routes (copy `gmail_send_routes.py`): `POST /fd/bat/spend/authorize`, `/settle`, `/void`; user
  routes `GET/PUT /fd/bat/spend-policy`, `GET /fd/bat/spends`; server-owned settings key
  `fd:bat-spend`; deck kill-switch `FD_BAT_SPEND`; per-owner `asyncio.Lock` around sum→check→insert;
  admin ceiling in `rate_limiter` (`bat_real_usd_month`) that a user policy can't exceed.
- Agent tool `spend` (declare/settle/void/status). Under the per-item cap → auto-approve; above →
  status `requested`, `add_notification`, and **park the worker on a §7 human-ask**.
- Enforcement hooks: a `bat_policy.tool_block` at the tool chokepoint requires an armed authorization
  for money-named tools; a Playwright `context.route()` interceptor + a pre-click heuristic
  ("place order / pay now / subscribe / start trial") blocks an un-authorized checkout POST and
  records it as evidence. Because `pinchtab`/`desktop_action`/MCP browsers can't be intercepted, Bat
  strips those while a purchase is armed.
- **Honesty about the limit:** a shell-capable agent can bypass a software cap (e.g. curl the payment
  API). The only *hard* cap on real money is issuer-side. The pre-auth ledger is the default
  (owner-chosen); the virtual-card option stays documented for anyone wanting an issuer-hard cap.
- Recurring charges / free trials: record the first charge against the cap and flag the renewal; Bat
  should cancel a trial it opened before the run ends, or ask.
- LLM spend and real-money spend are **separate caps** (LLM is "almost regardless of cost"; money is
  capped).

### 5.3 New accounts and logins
- Logins reuse the browser `CredentialStore` (Fernet) + `login` action. **Require
  `CLAW_BROWSER_CREDENTIAL_KEY`** so credentials are encrypted, not base64-obfuscated.
- Account signup is net-new orchestration (navigate → fill → submit → read verification). The
  primitives exist; Bat adds the flow. The email-verification step reads the owner's inbox directly
  via `google_mail` (read scope, not mail-gated) — note `html_to_text` strips `<a href>`, so read the
  raw HTML/plain part to recover the link/code.
- 2FA/OTP and CAPTCHA are **not** automated — they hit the §7 ask.
- Network capture stores bearer/cookie tokens verbatim; Bat's captured sessions are secrets. Keep
  them in the run VFS (teardown rmtrees worker dirs) and never in progress/notification JSON.

### 5.4 Browser + computer use
Enabled for the Bat archetype (`browser`, `pinchtab`, `desktop_action`, `screen_capture`,
`terminal`). The §3 floor covers `terminal`/`desktop_action`, not just `shell`.

---

## 6. Architecture

Mirror the Vatra invocation spine; host like the beings loop; store like Dubina.

- **Agent tool** `captain_claw/tools/bat.py`, modeled on the hardened `tools/basna.py` transport
  (pinned FD URL, `X-Agent-Secret`, `web_auth`/`source_port`/`owner_id`, origin capture). Actions:
  `start`, `status`, `answer`, `cancel`, `resume`, `list`, `get`. `start` refuses from inside any
  worker (`CLAW_*_WORKER`, incl. `CLAW_BAT_WORKER`). Registered beside Basna/Vatra in
  `agent_context_mixin`; `__init__` + `__all__`; short description with a MANDATORY when-to-use clause.
  The completion-relay text must not match the Basna relay regex.
- **Routes** `captain_claw/flight_deck/bat_routes.py` (`prefix=/fd/bat`). Agent routes under
  `/fd/bat/agent/` added to `_AGENT_GUARD_PREFIXES`; `_resolve_owner(body)` first in every handler;
  own per-owner cap + rate breaker (do **not** share Basna's dicts). `include_router` in server.
- **Store** `bat.db` under `FD_DATA_DIR` (deck-isolated), aiosqlite+WAL like `dubina_store`:
  `bat_runs` (durable status, origin, source_port, cumulative spend/tokens, lease columns),
  `bat_steps` (UPSERT on `(run_id, step_key)`, status pending|running|done|failed, output, attempt;
  write `running` before dispatch, `done/failed` after; **demote `running`→`pending` on adopt**),
  `bat_events` (append-only progress — never the volatile `_PROGRESS` dict for a day-long run),
  plus the spend ledger (§5.2) and `bat_asks` (§7).
- **Supervisor loop** `bat_loop(db, stop_event)` started from the lifespan (env `FD_BAT_DISABLED`),
  interruptible sleep, per-run try/except isolation, stopped before `_stop_all_processes` on
  shutdown. **Lease-based restart recovery** (copy the beings single-owner lock): on boot, adopt
  `bat_runs` in `running`/`retrying`/`waiting` whose lease is stale, demote stuck steps, re-resolve
  or tear down persisted worker slugs, seed cumulative spend from the row, then drive. This
  re-adoption is net-new — nothing in FD does it today; it's the feature that makes multi-day runs
  survive restarts. (Bat intentionally diverges from the locked Basna/Vatra "manual resume, no
  watchdog" decision, which was made for those modes, not this one.)
- **Inner algorithm:** a persisted variant of `run_plan_horizon` (plan → step → verify → fix/replan),
  where `max_replans` is effectively unbounded (gated by §7 caps instead of a fixed number), step
  outputs checkpoint to `bat_steps` after every verify, and a step may delegate to a whole Vatra/Basna
  team **in-process** via `execute_vatra`/`execute_route` (not the agent-start path — that would hit
  the shared caps and the recursion guard). Each attempt talks to an agent **only** through
  `_dispatch_one` (never raises, keeps partial work, auto-extends ≤1h, reconnects without re-sending).
  Prefer dedicated ephemeral Bat workers over the user's live chat agent (avoids the "busy" collision
  and shared history).
- **Bat workers get the full toolset, including the `vatra`/`basna` launcher** (owner requirement:
  Bat must be able to call Vatra and any available tool — the opposite of the stripped-down
  Vatra/Council workers). The spawn filter strips **only** the `bat` launcher (to prevent Bat-in-Bat);
  `CLAW_BAT_WORKER` is deliberately **not** added to the `basna`/`code` recursion guards
  (`tools/basna.py:253`, `tools/code_session.py`), so a Bat worker may start a Vatra/Basna/Code
  sub-run as a tool. A Bat worker is therefore constrained by exactly one thing: the §3 hard floor.
- **Cost:** set `_run_sid` at loop entry; after each attempt add `pricing.summarize(_RUN_USAGE[sid])`
  to the **persisted** cumulative column (in-memory resets on restart), write a `cost_ledger` row per
  iteration (`run_kind='bat'`), check the LLM cap against the persisted total. Thread a fail-closed
  price for unpriced models (else a local-model run shows `$0` and never trips the cap).
- **Cancel/stop:** persist `cancelled` first (so the supervisor won't re-adopt), cancel the driver
  task, tear down worker slugs, and for a live user agent send the `{type:'cancel'}` WS frame
  (FD never sends it today) rather than only killing the asyncio task.

---

## 7. Escalation and the stop conditions

**Pause-and-ask (owner choice #1).** New `captain_claw/flight_deck/human_ask.py`: a Future dict keyed
by `ask_id` + a durable `bat_asks` row (`question, kind, options, image, status, answer_redacted,
sent_via, expires_at, …`). `async ask(owner, sid, …, timeout) -> str` raising `AskTimeout`, modeled on
`flow_router.wait_for_input` but keyed per-ask (not per-user, which would let two asks cancel each
other). Fan-out: `awaiting_human` event + status; `add_notification(type='bat_ask', ref_type='bat')`;
WhatsApp via `_nudge_waids` + `send_text_checked` (reports Meta acceptance); the calling agent's chat
via `/api/chat/push` + `notify_source_agent`. Answer paths: (A) authoritative `POST
/fd/bat/asks/{id}/answer` with compare-and-set; (B) WhatsApp hook **before** the flow hook, matched by
owner + ask tag; (C) web/glasses via `/fd/flows/evaluate` under the `not _auto` guard. **Secrets (2FA,
passwords, CAPTCHA answers) live only in the in-memory Future — redacted from progress JSON, bell
bodies and WhatsApp history.** For long waits, use the durable `awaiting_human` gate (status-based,
survives restart); short 2FA waits can use the in-memory Future.

**Start gate (owner choice #3):** after the plan + contract are derived, if the plan involves mail,
account creation, or spend, set status `awaiting_plan`, show plan + contract, wait for Go (reuse the
Vatra Group-0 approve/cancel pattern). Pure research/build runs skip the gate.

**Bat stops (serious problem) when:**
- a spend/LLM cap is reached (flush cost first, then stop),
- a hard-forbidden §3 action was attempted,
- the owner cancels,
- a human-ask times out with no fallback strategy,
- the same failure repeats K rounds with no progress delta **and** no new strategy (the anti-deadlock
  rule, §4),
- an auth/credential the human must provide is refused.

A serious stop persists a terminal state with truth/files/analysis kept (a `_finish_blocked`-style
save) so the run is resumable — never silently `done`, never silently stuck on `running`.

---

## 8. UI (own page — owner choice #4)

New `BatPage.tsx` + `batStore.ts`, reusing the exported `ProgressFeed`, `LiveAgentsPanel`,
`ResizableSplit`, `RunFilesPanel`, `RunDatastorePanel`, `CostCard`, and the `buildLiveAgents`/`FileModal`
helpers. New `BatAskCard.tsx` (question, optional screenshot, text/masked-code/choice input, countdown,
Answer / Can't-help / Abort) and a live **budget meter** (separate `budget` progress stage — do **not**
reuse the terminal `cost` stage mid-run). Nav: add `bat` to the `ViewMode` union, `App.tsx` mainContent,
Sidebar "Multi-Agent" section; map `bat_ask`/`bat_done`/`bat_error` notification types; add a `bat`
refType action (full layout → open page; kiosk/simple → `KioskDialog` with `BatAskCard`, since kiosk
users can't reach pages). Emit the same progress-event shape so the widgets work unchanged, but back it
with `bat_events` + a `?since=i` cursor (the shared `_PROGRESS` dict wipes at 50 sessions and is lost on
restart). Remember: `cd flight-deck && npm run build`, commit the bundle, restart FD, hard-refresh.

---

## 9. Phasing

1. **Floor first (Invariant A).** ✅ **DONE (2026-10-07).** `captain_claw/bat_floor.py` — pure,
   un-widenable system-harm denylist (catastrophic `rm`/`chmod -R` with protected-root path analysis;
   always-block for mkfs/newfs/wipefs, fdisk/parted, `diskutil erase|unmount`, `umount`,
   `dd of=/dev/*`, device redirects, power control, broadcast kill, fork bomb, `find … -delete` at a
   root, inline-interpreter root wipes). Wired as a hard `ToolBlockedError` at the universal chokepoint
   `ToolRegistry.execute`, gated by the new `CLAW_BAT_WORKER` marker, covering `shell`/`terminal`/
   `desktop_action` (termux is phone-only, `code` is a launcher). No config knob or relax flag can
   widen it. 58 tests in `tests/test_bat_floor.py` (every command class + spelling, benign dev work
   stays allowed, non-exec tools never screened, registry wiring blocks-before-execute); 359 existing
   registry-path tests still green. **Note:** the marker is intentionally absent from the
   `basna`/`code` recursion guards so Bat workers can call Vatra.
2. **Store + loop skeleton.** ✅ **DONE (2026-10-07).** `captain_claw/flight_deck/bat_store.py`
   (aiosqlite+WAL, deck-isolated `bat.db`): `bat_runs` (status, origin, caps, persisted cumulative
   spend, per-run lease columns), `bat_steps` (UPSERT checkpoints, `demote_running_steps` on adopt),
   `bat_events` (append-only progress, replaces the volatile `_PROGRESS` dict). `bat_loop.py`:
   `BatDriver` (plan → checkpointed steps → retry → judge → terminal; between-step checks for owner
   cancel, lost lease, and the persisted LLM-spend cap; crash-guarded so a failure never leaves a run
   `running`) + `BatSupervisor` (claims a per-run lease, drives each run in its own task so one long
   run can't block another) + `bat_loop` lifespan entry (idle until a real runner is registered).
   Handlers are **seams** (`set_attempt_runner`/`set_planner`/`set_judge`) filled in Phase 3. Wired
   into `server.py` lifespan (init `bat.db`, start/stop the supervisor, `FD_BAT_DISABLED` kill-switch,
   stop before `_stop_all_processes`). 11 tests in `tests/test_bat_loop.py` incl. the **restart proof**
   (crash mid-step → fresh supervisor adopts the stale-leased run, demotes the orphaned step, finishes
   without re-running completed work) + cancel + cap + lease takeover + retry. **No world-actions, no
   LLM** — pure durable orchestration. Still dormant in prod: no routes/tool create a run yet (Phase 3).
3. **Done-judge (Invariant B) + agent-invocable.** ✅ **DONE (2026-10-07).**
   - `flight_deck/bat_judge.py` — the fail-closed verdict: Layer 1 deterministic criticals block done
     with no panel call; a missing deliverable is never done; Layer 2 independent vote panel
     (abstentions dropped, exceptions/timeouts → disagree, agree-majority required); anti-deadlock
     override only when the deterministic layer is all-green AND the model-veto streak hits the ceiling
     (recorded as `overridden`). Pure, model calls injected. 12 tests.
   - `flight_deck/bat_routes.py` — `/fd/bat/agent/{start,status,get,list,cancel}` (same loopback/
     X-Agent-Secret guard as Basna/Vatra) + the three handler seams wired into `bat_loop` at import:
     planner (LLM decompose with a single-step fallback so a run always progresses), attempt_runner
     (spawns an ephemeral Bat worker via `_spawn_bat_worker` → `_dispatch_one`, meters spend into the
     persisted cap), judge (`bat_judge` + an independent panel on a tier resolved separately from the
     workers), and `on_finish` (bell / origin channel / caller-agent delivery). **Bat workers keep the
     full toolset — only `bat` is stripped (`_bat_worker_tools`), so a worker may delegate to Vatra.**
   - `tools/bat.py` — the agent tool (hardened Basna transport); recursion-guarded against starting a
     Bat run from inside any ensemble worker. Registered in `agent_context_mixin` + `tools/__init__`;
     router + `/fd/bat/agent/` guard prefix wired in `server.py`.
   - Tests: `test_bat_judge.py` (12), `test_bat_routes.py` (pure helpers), `test_bat_tool.py`
     (recursion/validation/routing). **Bat suites: 98 passing**; 275 tool-registration tests green.
   - **Scope boundary:** world-action grants — the run-scoped email exception (Phase 5), capped
     real-money spend (Phase 6), account creation (Phase 7) — are NOT in this phase. A Phase-3 Bat
     worker runs under the Phase-1 hard floor with the owner's normal toolset. The pure logic is
     unit-tested; the live spawn/dispatch/panel path is verified on a running deck (prod).
4. **Escalation + start gate.** ✅ **DONE (2026-10-07).**
   - `bat_store` gains a `bat_asks` table (durable asks) + `set_config`; asks have CRUD + compare-and-set
     `answer_ask` + `expire_asks`. `awaiting_plan`/`awaiting_human` are non-terminal, adoptable states.
   - `flight_deck/human_ask.py` — the pause-and-ask registry (durable-first, survives restart): an ask
     is a row + the run stands down; answering re-kicks it. **Secrets** (2FA/passwords/CAPTCHA) are kept
     in an in-process holder only, never in the durable row (redacted placeholder) or a notification,
     and are never answerable over a chat channel. Notifier injected.
   - **Start gate** (owner decision #3): `plan_needs_gate` (deterministic mail/money/account classifier)
     + the driver `_GATE_CHECK` seam. A run whose plan would act in the world pauses `awaiting_plan` with
     a `plan_approval` ask; a plain build/research run proceeds untouched. The same gate resumes an
     `awaiting_*` run once answered (approve → proceed, reject → cancel, expired → error).
   - **Step-level ask**: an attempt that returns `{ask: …}` stands the run down `awaiting_human`; the
     answer is injected into the resuming step's prompt (secret via the in-memory holder, non-secret via
     config). The supervisor skips a run waiting on an open ask (no spin), resumes it on answer (kick),
     and expires overdue asks (→ `human_ask_timeout` error).
   - **Answer path A** (authoritative, owner-authenticated): `GET /fd/bat/asks`,
     `POST /fd/bat/asks/{id}/answer`, `POST /fd/bat/runs/{id}/approve-plan`. **Notification** fan-out:
     in-app bell (always) + WhatsApp nudge (non-secret). **Paths B/C** (free-text answer over WhatsApp /
     web chat) are deferred — the WhatsApp binding has no clean waid→owner map and the web path needs a
     `/fd/flows/evaluate` edit; `human_ask.answer_for_owner` is ready for them. Follow-up.
   - Tests: `test_human_ask.py` (store asks + registry + secret redaction + CAS), `test_bat_gate.py`
     (start gate approve/reject, non-gated, step-ask resume, expiry, supervisor skip/resume), plus
     `plan_needs_gate` in `test_bat_routes.py`. **Bat suites: 117 passing**; 272 tool-reg tests green.
5. **Email exception.** ✅ **DONE (2026-10-07).** The run-scoped exception to PR #56, tied to the
   Phase-4 start gate so it is **human-approved by construction**.
   - `mail_authority`: `bat` added to `AUTOMATION_KINDS` + `KIND_LABELS` ("a Bat run"), and
     `CLAW_BAT_WORKER` added to `_WORKER_ENVS` so a Bat worker fails closed to **deny** by process
     default — the grant can only arrive on the per-turn frame.
   - **Grant path:** the owner approves a plan whose gate reason includes "send email" → `_gate_check`
     sets `config.email_allowed=True`. Only then does `_bat_attempt` dispatch with
     `automation={kind:'bat', mail_write:'allow'}`; otherwise `deny`. The grant lives on the run config
     and is on Bat's **own** dispatch frame only — a child Vatra/Basna run gets its own (deny) default,
     so the allow **never propagates**.
   - **FD Gmail gate only:** `_bat_worker_tools` now also strips `send_mail` (the uncapped Mailgun/
     SendGrid/SMTP path) — Bat mail routes through `/fd/google/gmail/send` with the owner's
     allowlist/daily-limit/audit intact. `google_mail` stays.
   - **Layer-3 evidence:** `_email_receipt_block` reads the `gmail_sends` audit for this run's workers
     and hands the judge panel the ground-truth send count/recipients, so a worker can't pass by
     *claiming* it emailed when the gateway recorded nothing.
   - Tests: `test_bat_mail.py` (the kind + worker default-deny, allow-only-after-approval, non-mail
     never granted, receipt block), updated `_bat_worker_tools` test. **Bat + mail_authority suites:
     402 passing.**
   - Still deferred: `AUTONOMY_HARD_EXCLUDE` exact-match entry for `bat` (email safety does not depend
     on it — the grant needs human plan approval autonomy can't give — but excluding keeps autonomy from
     auto-starting Bat runs; §10). The owner's Gmail-send policy must be enabled on the deck for real
     sends (second gate, PR #42). Live send verified on a deck.
6. **Capped spend.** ✅ **DONE (2026-10-07).** Pre-authorization with a hard run cap, auto-approve under
   a per-item limit, owner-approve above it, enforced across restarts (owner decision #2).
   - `bat_store` `bat_spend` ledger: `create_spend`/`set_spend_status`/`committed_usd` (counts
     requested+approved+consumed so a pending reservation can't be double-spent; actual charge wins
     over the ceiling)/`find_live_spend` (idempotent re-declare: approved→proceed, denied→stop,
     requested→pending). Persisted → the cap holds across an FD restart.
   - `spend_decision` (pure cap logic) + `_spend_policy`: spend is OFF unless the deck switch
     `FD_BAT_SPEND` is on **and** the owner approved a spend plan (`config.spend_allowed`, set by the
     start gate like email) **and** a `real_usd_cap > 0` was set at start. `per_item_usd` is the
     auto-approve threshold.
   - Agent endpoints `/fd/bat/agent/spend/{authorize,settle,void,status}`: authorize → approved (≤ limit,
     within cap) | requested (> limit → raises a `spend_approval` ask, run → awaiting_human) | denied
     (over cap / disabled). The driver stands the run down when a step's spend call parked it. The
     spend-approval answer flips the ledger row on resume (approve → the re-declare finds it approved and
     proceeds; decline → denied, so it doesn't re-ask in a loop).
   - `spend` worker tool (declare/settle/void/status) — registered globally, **refuses outside a Bat
     run**. `_bat_worker_tools` keeps it (it still strips `bat`/`send_mail`).
   - `looks_like_purchase` pre-click heuristic (tested). **Deferred/deck-only:** the Playwright
     `context.route` checkout interceptor + `BrowserTool._click` guard (can't be tested headlessly), and
     the MCP money-tool block — both wire to the ledger using the tested helper.
   - Tests: `test_bat_spend.py` (cap decision, policy gating, ledger math, all four endpoints,
     idempotency, settle, spend-approval resume, tool refusal, purchase heuristic). **Bat suites: 140
     passing**; 550 tool/mail/spend tests green.
   - **Honest limit (as discussed):** this is a software cap. A shell-capable worker could in principle
     bypass it (e.g. curl a payment API). The only issuer-hard cap is a virtual card per authorization
     (documented option). The pre-auth ledger is the owner's chosen default.
7. **Accounts/logins.** ✅ **DONE (2026-10-07).** Account creation / login is browser+LLM driven
   (prompt-guided); this phase builds the enforceable scaffolding around it.
   - **Worker→human escalation:** `ask_human` tool + `/fd/bat/agent/ask`. A worker blocked on a value
     only the owner can give (2FA/OTP, CAPTCHA, a credential) calls it; the run stands down
     `awaiting_human` and resumes on the answer, which is injected into the step's next attempt (secret
     via the in-memory holder, redacted in the row/bell/channel). The tool refuses outside a Bat run.
     **No CAPTCHA solver** — the worker is told to ask, never to bypass.
   - **Account grant:** the start gate already flags "create an account"; approving sets
     `config.account_allowed` (like email/spend). `_build_step_prompt` then adds account guidance
     (sign up / log in, read the owner's inbox for the verification link/code via `google_mail` — the
     raw/plain part, since rendered text drops URLs — and `ask_human` for 2FA/CAPTCHA/credentials).
   - **Credential-encryption-key requirement:** `credential_encryption_ready()` checks
     `CLAW_BROWSER_CREDENTIAL_KEY` / config; when account creation is granted but no key is set, the
     step prompt warns the worker NOT to store passwords in the browser credential store (which would
     fall back to base64 obfuscation) and to ask the owner to set the key.
   - **`extract_verification`**: pulls a verification link/code out of an email body (tested).
   - Tests: `test_bat_accounts.py` (grant via gate, key requirement + prompt warning, verification
     extraction, the ask endpoint pausing + secret resume kept out of config, tool refusal). **Bat
     suites: 149 passing**; 421 with tool-reg.
   - Deck-only: the signup/login browsing itself, and reading verification mail, run on a live deck
     with the browser + the owner's Google connected.
8. **UI.** `BatPage`, `batStore`, `BatAskCard`, budget meter, nav + notifications; build + commit.

Each phase is a PR; ship 1–3 before any world-action phase.

---

## 10. Containment checklist (keep in sync — a missed entry silently breaks safety)

- Worker marker `CLAW_BAT_WORKER` added to: `agent_reasoning_mixin` FD-worker markers,
  `mail_authority._WORKER_ENVS`, `code_session` markers, the scale disables.
- `bat` added to `mail_authority.AUTOMATION_KINDS` **and** `KIND_LABELS` (+ the equality test).
- Bat worker spawn strips **only** `bat` (keep `basna`/`vatra` and every other tool — Bat must reach
  all tools). Do **not** add `CLAW_BAT_WORKER` to the `basna`/`code` recursion guards.
- Recursion guard in `tools/bat.py` refuses from inside a `CLAW_BAT_WORKER` (Bat-in-Bat only).
- `AUTONOMY_HARD_EXCLUDE`: decide whether autonomy may fire Bat — **recommend excluding it**, but note
  the matcher is a substring match, so a bare `bat` would also exclude `web_fetch_batch`; use an exact
  check.
- Bat state under `FD_DATA_DIR` (both glasses decks share `~/.captain-claw` when it's unset) **plus**
  the lease lock, so two decks can't drive one run.
- VFS `_project_origin` kinds + `VFSBrowser` filter learn `bat-<sid8>`.
- `label_eval._mode` learns `bat`.
```
