# Lupa — the Agentic Research Desk (product plan)

*Working name: **Lupa** (Croatian: magnifying glass) — fits the family (Basna, Vatra, Mrav, Iskra, Dubina). Rename freely; nothing below depends on it.*

## What it is

A B2B research product: companies commission research in a chat-like intake, watch a
planned agent team execute it, and receive reports **with receipts** — a verification
panel (facts ledger, contract checks, consistency audit, quality verdict) and a cost
line ($ per run, effective $/hr vs a human analyst). Research lives in persistent
**streams** that deepen over rounds; **standing briefs** re-run on a schedule and
report only the delta.

Target buyers: consultancies / market-research agencies first, then in-house
competitive-intelligence and strategy teams, think tanks, investigative desks.
Academia is a credibility channel (plan gate + facts ledger = methods section), not
the revenue engine.

**Lupa is also Pack #1 of a vertical factory** — see Part II (Kalup). The shell is
built pack-aware from day one so the same codebase can ship 20–30 branded vertical
apps by end of year, each generated in 2–3 days.

## The prime directive: Captain stays agnostic

The product is a **separate web app + thin product backend (BFF)**. Captain Claw is
the engine and the user system, consumed **only over its HTTP API**.

Hard rules, enforced throughout every phase:

1. **No product code inside `captain_claw/`.** The product never imports Captain
   internals — HTTP only (`httpx` → `http://127.0.0.1:25080`).
2. **Any change to Captain must be generic and useful to Flight Deck itself** — new
   read endpoints, hardening flags, the continuation-rounds engine. No route, table,
   or module in Captain may know that "Lupa", "streams", or "briefs" exist.
3. **Defaults unchanged.** Every Captain-side hardening is opt-in via env; a local
   single-user Captain behaves byte-identically with no new env set.
4. **FD's own UI is untouched.** The product has its own SPA; `flight-deck/` builds
   and ships exactly as today.
5. Product-domain data (streams, rounds, briefs, org metadata) lives in the BFF's own
   SQLite, referencing Captain IDs (`user_id`, `basna_sessions.id`, vfs project name)
   — never the other way around.

## Architecture

```
                 public (TLS)
   browser ──────────────────────► lupa/api  (FastAPI BFF, own product.db)
   (lupa/web SPA)                     │
                                      │ loopback HTTP only
                                      ▼
                              FD :25080 (bound 127.0.0.1)
                              — auth, users, Vatra/Basna, VFS,
                                archetypes, quality, cost ledger
```

- **Auth = Captain's user system, proxied first-party.** The BFF forwards
  `/auth/*` to `/fd/auth/*` so the `fd_refresh` cookie is first-party to the product
  origin (its `path=/fd/auth`, `samesite=lax`, `secure=false` scoping never leaks
  cross-origin because the browser only ever talks to the BFF). The BFF verifies
  access JWTs **locally** with the shared `FD_JWT_SECRET` (HS256, payload
  `{sub, role}`) — no per-request round trip to FD.
- **Machine-to-machine (standing briefs):** the BFF holds `FD_JWT_SECRET`, so it can
  mint short-lived access tokens for a given `sub` when a scheduled brief fires. No
  Captain change needed, no API-key table needed for v1.
- **FD is never publicly reachable** in a product deployment: it binds loopback; the
  BFF is the only public surface. This single decision neutralizes most of the
  multi-tenant sharp edges found in the audit (below) without touching Captain —
  but the worst ones still get proper guards in Phase 0.

### Concept → engine mapping (all existing Captain surfaces)

| Product concept | Captain primitive |
|---|---|
| Commission → plan gate | `POST /fd/vatra/start` → status `planning`; approve via `POST /fd/vatra/plan/approve` (carries `quality`), discard/replan via `/plan/cancel`, `/plan/replan` |
| Run progress | poll `GET /fd/basna/sessions/{id}/progress` (includes the `cost` event) |
| Report + per-agent work | `GET /fd/basna/sessions/{id}` (analysis, files) + `/runs` |
| Stream workspace | one VFS project per stream; files via `/fd/vfs/*` |
| Verification panel | `analysis.quality_metrics` / `.consistency` / `.quality_verdict` / `.gaps` + `.contract.json` via `/fd/vfs/read` + facts ledger (new endpoint, Phase 0) |
| Cost line & history | `cost` progress event now; `GET /fd/costs` over `cost_ledger` (new endpoint, Phase 0 — `db.list_run_costs` already exists with zero callers) |
| Rounds / deepen | continuation rounds (engine gap — Phase 0, plan already in `docs/basna-vatra-continuation-plan.md`) |
| Standing briefs | BFF's own cron → continuation run with the existing `delta rounds` (R4) quality lever. **Not** FD's `/scheduler/*` (global, no user column, token-optional) |
| House style | `POST /fd/archetypes/forge` (multipart, returns unpersisted drafts) + archetype CRUD |
| Second opinion | `POST /fd/basna/execute` on the same brief + `consistency_check` lever; diff view in product |
| Quality preset | product sends a `quality` dict (presets in `quality_profile.py`); persisted in `basna_sessions.config` — no engine change |

## Phase 0 — Engine readiness (all inside Captain, all generic)

The only phase that touches Captain. Three buckets.

**0a. Security hardening (small, default-off, FD benefits regardless).** From the
audit of `server.py` / route files:

- Gate `/fd/basna/agent/*` and `/fd/vatra/agent/*` (today: **no auth dependency**, and
  `_resolve_owner` accepts an attacker-chosen `body.owner_id` — anyone reaching :25080
  can start paid runs as any user). Guard: require `X-Agent-Secret` or loopback
  origin. This is a bug fix, not a product feature — FD wants it too.
- `FD_CORS_ORIGINS` env to pin CORS (default stays `*` for local; note that
  `allow_origins=["*"]` + `allow_credentials=True` is invalid per spec anyway).
- `FD_LOCKDOWN=1` (one env, hosted deployments only) to disable host-filesystem
  surfaces: `/fd/vfs/browse-fs`, `POST /fd/vfs/links` (mounts any absolute host dir),
  and `/fd/projects/*` (24 routes, zero auth). Unset = today's behavior.
- Deployment doc: set `FD_JWT_SECRET` explicitly (unset = random per process — tokens
  die on restart and the BFF can't verify), set `FD_GLASSES_BRIDGE_TOKEN` (unset =
  `/scheduler/*` is fully open), force `FD_AUTH_ENABLED=true`.

**0b. Two generic read endpoints (data exists, no HTTP surface):**

- `GET /fd/basna/sessions/{id}/facts` — render the facts ledger (`.facts.db` SQLite in
  the run's VFS folder) as JSON via the existing `facts_ledger.export_rows` /
  `list_rows`. Today the ledger is only downloadable as a binary blob.
- `GET /fd/costs` — query `cost_ledger` per authenticated user (filters: run_kind,
  since, ref). Wire up the orphaned `db.list_run_costs`.

**0c. Continuation rounds** — execute the existing
[basna-vatra-continuation-plan](basna-vatra-continuation-plan.md): carry a finished
run forward in the **same** VFS folder + conclusion (today deepen/fill-gaps spawn a
new folder — the known bug), pin the chain to the root session's `vfs_project`,
unified `_continue_run`. This was a nice-to-have; for Lupa it is **launch-blocking**
(it *is* the "research stream"). Ship as a normal Captain feature with tests; FD's
own deepen/fill-gaps buttons get fixed for free.

*Exit criteria: Captain suite green; FD behavior with no new env vars byte-identical;
a curl script proves start→plan→approve→progress→facts→costs end-to-end on loopback.*

## Phase 1 — Product skeleton

New top-level folders (same monorepo, zero imports from `captain_claw/`):

```
lupa/
  api/    FastAPI BFF — own SQLite (lupa.db), httpx client to FD, auth proxy,
          JWT verify (shared secret), whitelisted FD proxy routes
  web/    Vite + React + TS + Tailwind v4 + Zustand SPA (same stack as flight-deck
          for velocity, but a separate app with its own build — nothing shared)
```

BFF product schema (v1): `streams` (id, user_id, title, vfs_project, quality_preset,
created_at), `stream_sessions` (stream_id, session_id, round_no, kind:
initial|deepen|delta|second_opinion), `briefs` (Phase 4).

Product flow, v1: login (proxied) → stream list → **new stream**: intake form (brief,
attachments via `/fd/vfs/upload`, quality preset) → `vatra/start` → **plan-gate
screen** (the route/plan from session detail, editable, Approve / Replan / Cancel) →
live progress (poll) → **report reader** (markdown stack like FD's: react-markdown +
KaTeX + prism) over the stream's VFS files.

**Pack-aware from day one (Part II):** every vertical-specific surface — product
name/branding, vocabulary, intake form fields, commission types, report templates,
quality preset, archetype cast, brief cadences, onboarding copy — loads from a
**pack** (`lupa/packs/<slug>/`), never from hardcoded strings. Lupa itself is just
the default `research-desk` pack. This costs little now and is the whole factory
later; retrofitting it would cost weeks.

*Exit criteria: a second browser origin runs a full commission end-to-end against a
loopback FD, with FD's own UI untouched and working in parallel.*

## Phase 2 — Receipts (verification panel + cost)

The differentiator phase — everything reads existing engine output:

- **Verification panel** per report: quality verdict + blocking summary, claims
  checked/confirmed/refuted/unverifiable, consistency findings, contract checks
  (`.contract.json`), coverage gaps, and the full facts-ledger table (Phase 0b
  endpoint) with per-claim status.
- **Cost line** on every report (`usd`, `hourly_usd`, tokens, per-model) + stream and
  account cost history via `GET /fd/costs`. Frame `hourly_usd` against a configurable
  human-analyst rate — the ROI number on the page.
- A **"desk" quality preset** in the product (thorough-based: honesty_guard,
  facts ledger, constraints contract, consistency_check, block_on_critical, acted-gate)
  — just a `quality` dict the BFF sends; Captain unchanged.

## Phase 3 — Streams that deepen (needs 0c)

- **Continue round** in the same stream folder; round timeline UI (initial → deepen →
  delta), per-round diffs of conclusions.
- **Open questions**: surface `analysis.gaps` as one-click follow-up commissions
  (`/fill-gaps` / continuation).
- Stream-level settings: pinned quality preset, model policy, pinned cast.

## Phase 4 — Standing briefs

- BFF-side scheduler (its own asyncio cron over `briefs`; deliberately **not**
  FD's global `/scheduler/*`): cadence + brief → mint token for the owner → fire a
  continuation run with the `delta rounds` lever → brief lands in the stream +
  a product inbox ("what changed since last run").
- Notification channel (email digest) optional, later.

## Phase 5 — House style + second opinion

- **Forge UI**: upload playbooks/templates → `POST /fd/archetypes/forge` → user picks
  drafts to save → streams can pin the house cast. "Your methodology, encoded" —
  the switching cost.
- **Second opinion**: run the same brief through Basna ensemble, render a conclusions
  diff against the Vatra report. Trust feature, demo gold.

## Phase 6 — Packaging & design partners

- Plan tiers: reuse FD's existing `free/pro/enterprise` rate-limiter plans; billing
  stays product-side (Stripe on the BFF; Captain knows nothing about money beyond its
  cost ledger).
- Deployment story: single host — FD loopback + `FD_LOCKDOWN=1`, Lupa behind TLS
  reverse proxy. The same topology *is* the on-prem SKU ("runs on your hardware,
  your models") — the weak-model-tending work makes that pitch credible.
- Deferred consciously: Google Drive mounts for product tenants (blocked on per-user
  Google tokens — known deferral in the VFS↔Drive arc), FD flows (global table, no
  user column), machine API keys for customers' own integrations.
- Two–three design partners (one consultancy, one CI team) before any general launch.

## Sequencing & effort (rough)

| Phase | Size | Depends on |
|---|---|---|
| 0a hardening | S | — |
| 0b read endpoints | S | — |
| 0c continuation rounds | M–L | existing plan doc |
| 1 skeleton | L | 0a |
| 2 receipts | M | 0b, 1 |
| 3 streams | M | 0c, 1 |
| 4 briefs | M | 3 |
| 5 house style / 2nd opinion | M | 1 |
| 6 packaging | M | all |

0a+0b+0c can proceed immediately and in parallel with Phase 1 scaffolding; Phase 2 is
the first externally demoable milestone ("a report with receipts"), Phase 3+4 make it
subscription-shaped.

## Open decisions (defaults chosen, easy to reverse)

1. **Name**: "Lupa" as working name.
2. **Monorepo top-level `lupa/`** vs separate repo — monorepo chosen for velocity;
   the HTTP-only rule keeps the boundary honest either way.
3. **BFF in Python/FastAPI** (matches the house stack) vs Node — Python chosen.
4. **Progress transport**: keep FD's poll model (no SSE/WS for runs exists); BFF may
   later add SSE fan-out over its own poll loop without touching Captain.

---

# Part II — Kalup: the vertical factory

*Working name: **Kalup** (Croatian: casting mold). Goal: generate one vertical
solution in 2–3 days (max 5), ship 20–30 branded vertical apps by end of 2026.
Strategic anchor: Bessemer's "The Future of AI is Vertical" (bvp.com/atlas);
implementation patterns borrowed from block/buzz.*

*Status: K1 shipped — runtime pack registry (repo packs are seeds), creator
roles, Studio (create → generate-via-Vatra → review → evaluate → publish with
the ship-gate), desk gallery, /desks/<slug> activation, pack-scoped streams.
Remaining: draft-fork/rollback of published packs, richer manifest editors with
live preview, per-pack cast auto-forge in generation, custom domains
(Host-header), kalup CLI, eval thresholds beyond the receipts verdict.*

## Why this works strategically (Bessemer thesis → our stack)

The BVP argument maps almost one-to-one onto what Lupa already is:

- **"High-cost repetitive language-based tasks"** is their definition of the vertical
  AI opportunity (professional services = 13% of US GDP vs software's 1%). The Lupa
  run-shape — commission → planned team → verified report → standing brief — *is* the
  generic automation of exactly that task class. Each vertical differs mainly in
  vocabulary, templates, cast, sources, and compliance framing. That's why verticals
  can be packs, not products.
- **Core vs. supporting workflows**: their key selection framework. Supporting
  workflows (research briefs, monitoring, paperwork, memo prep) are lower-resistance
  wedges, especially in regulated/relationship-driven industries where practitioners
  gladly delegate them. Kalup verticals should default to **supporting-workflow
  wedges**; the pack's workflow templates can add core-workflow depth later —
  BVP's "layer cake", done as pack updates instead of new products.
- **"Economic value delivered" + the wrapper criticism**: their moat list is
  proprietary data, product/workflow depth, integration, sector knowledge. Our
  receipts (facts ledger, contract, verdict) plus the cost line ($/run, $/hr vs a
  human) are the *quantified* ROI story per report; per-vertical accumulated
  streams/ledgers/learning become the data moat; pack-declared connectors become the
  integration moat.
- **Urgency**: BVP notes the text/data window is narrowing as multimodal arrives.
  A factory that ships in days, not quarters, is the right response to that window.

## What we take from block/buzz

Buzz (Block's Rust/Nostr "hive mind" workspace) is a different product, but four of
its patterns transfer:

1. **One append-only, attributable event log.** In Buzz every message, review,
   workflow step, and git event is a signed event in one log. Kalup equivalent: an
   exportable **audit bundle** per stream (report + facts ledger + contract results +
   run events + costs + per-agent attribution — the VFS authorship sidecar already
   exists). For compliance-adjacent verticals this artifact is a selling feature, not
   plumbing.
2. **Agents as first-class identities, not permission-scoped bots.** Buzz gives
   agents their own keys and audit trails. Kalup equivalent: the pack's cast is a
   *named team* ("your analyst desk") whose members' work is individually
   attributable in the report and audit bundle — archetypes already carry identity;
   the product surfaces it.
3. **Agent-first interface, JSON in/out.** Buzz ships `buzz-cli` so agents can drive
   the platform. Kalup equivalent: the pack registry's BFF endpoints are agent-usable
   (JSON in/out), with a thin **`kalup` CLI** over them for automation/CI — which
   means Captain's own Code/Vatra modes can *author packs*. The factory builds
   itself; this is what makes 2–3 days per vertical real. (The primary authoring
   surface is the in-product **Pack Studio** GUI — see below.)
4. **Shared backend, strict semantic isolation.** Buzz multi-tenants communities over
   one Postgres/Redis. Kalup mirrors it: one Captain + one BFF deployment hosts all
   packs — every published vertical is instantly available in the umbrella app's
   **desk gallery** (`/desks/<slug>`: one login, one place), and an optional custom
   domain resolves to the same pack via Host header. Tenancy stays per-user
   underneath. 30 apps ≠ 30 servers — and ≠ 30 logins.

## The pack format

A pack is **data + templates only** — no code, no Captain awareness. Its content:

```
pack.json          identity: name, domain, tagline, theme tokens, locale
vocabulary.json    domain terms: "stream"→"engagement", "brief"→"watchlist", …
intake/*.json      commission types: form fields, attachment hints, default quality
cast/              archetype seeds: forge source docs and/or pre-forged archetypes
templates/*.md     report skeletons, section orders, citation style
quality.json       preset deltas over "desk" (e.g. block_on_critical always-on)
briefs.json        standing-brief presets: cadence, delta framing, inbox copy
connectors.json    allowed data sources (v1: VFS upload; later Drive, MCP allowlist)
evals/             golden tasks: sample commissions + minimum receipt thresholds
onboarding.md      first-run copy, sample commission, ROI framing ($/hr baseline)
pricing.json       plan mapping onto FD's free/pro/enterprise tiers
```

**Packs are runtime entities, not repo files.** They live in the BFF (`packs` table:
slug, owner, status `draft|published|archived`, version, manifest JSON + an asset
store for templates/cast seeds/eval docs). The shell fetches the manifest at runtime
(`GET /packs/{slug}/manifest`, cached) — theming, vocabulary, and forms are all
runtime-driven, so **publishing a pack requires no build, no deploy, no restart**.
The repo's `lupa/packs/` holds only seed/system packs (e.g. `research-desk`)
imported at startup; everything else is born in the product.

Engine impact: **zero new Captain surface beyond Phase 0.** Forge, quality profiles,
archetype CRUD, VFS, and MCP config all exist; the pack registry, Pack Studio, evals,
and the audit bundle are pure BFF/shell work. This is the strongest validation of the
agnostic-Captain rule — the factory is entirely a product-layer construct.

## Pack Studio — the in-product factory (GUI-first)

The factory is a **feature of Lupa itself**, not a developer workflow. Everything in
one place: admins and power users forge, evaluate, and publish verticals from the
same app their customers use. The `kalup` CLI still exists, but as a thin client of
the same BFF endpoints (automation/CI); the GUI is the primary surface.

**Who can forge.** FD `admin` always; plus a product-level **`creator` capability**
the BFF stores per user (granted by an admin in Lupa's admin screen). Captain's
user model is untouched — "power user" is product-domain, exactly where it belongs.

**The wizard (maps 1:1 to the factory line below):**

1. **Seed** — name the vertical, pick the nearest family as a starting template,
   upload 5–20 domain documents (playbooks, sample reports, regulations).
2. **Generate** — the BFF fires a Vatra run (as the creating user) whose brief is
   "draft this pack": vocabulary, intake forms, report templates, onboarding copy,
   golden eval tasks — written to the run's VFS folder and imported into the pack
   draft. In parallel, `POST /fd/archetypes/forge` mints the cast from the same
   corpus. The creator watches the same plan-gate + progress UI customers see —
   the factory demos the product while building it.
3. **Review** — a pack editor: vocabulary table, form builder, template editor with
   **live preview** (render a sample report through the pack), cast roster with
   per-archetype edit. Superior-UX rule: every generated artifact is editable in
   place; nothing requires touching files.
4. **Evaluate** — one click runs the golden commissions end-to-end; a scorecard
   renders the engine's own `quality_metrics` (claims confirmed ratio, contract
   failures, verdicts) per task against the pack's thresholds. **Publish stays
   locked until the scorecard is green** — the ship gate is UI-enforced.
5. **Publish** — the pack appears instantly in the **desk gallery** (the umbrella
   app's home: every published vertical, one login, one place) at
   `/desks/<slug>`. An optional custom domain maps onto the same pack via
   Host-header resolution — branding is a projection, availability is immediate.

**Lifecycle:** editing a published pack forks a new draft version; publish swaps it
atomically; archived packs keep existing streams readable. Version history in the
Studio, one-click rollback.

## The factory line (one vertical in 2–3 days, max 5)

- **Day 0 (prep, async):** pick the vertical against the selection checklist below;
  collect the 5–20 domain documents.
- **Day 1 — Seed + Generate** (Studio steps 1–2), then a human pass over the drafts
  in the Review editor.
- **Day 2 — Evaluate** (step 4): iterate cast/templates until the scorecard is
  green. This is what keeps a 2-day build from being a thin wrapper.
- **Day 3 — Publish** (step 5): theme, pricing map, landing copy, gallery listing;
  custom domain if warranted.
- **Days 4–5 (buffer):** only when the vertical genuinely needs a connector or a
  compliance review. If a vertical repeatedly needs more, it's a product, not a pack
  — park it.

## Vertical selection checklist (per BVP)

A candidate pack must score yes on most of: language-heavy and repetitive; a
**supporting** workflow (or a low-resistance core one); costly today in analyst
hours (the $/hr receipt lands); reachable buyers (a community, association, or
directory to sell into); v1 viable on uploaded documents + web research alone (no
bespoke integration); no licensing wall for v1.

First candidates (Fric/VC excluded per direction): tender/RFP analysis for
contractors; procurement & vendor due-diligence; compliance/regulatory monitoring
(pick one regime per pack, e.g. EU AI Act readiness); ESG/CSRD reporting research;
market-entry studies for exporters; patent/prior-art landscaping; grant-research for
nonprofits; medical-affairs literature review; legal research memos (supporting
workflow, not advice); insurance-claims research; real-estate due diligence;
competitive-intel for product teams. Twelve families → 20–30 apps via regime/
geography/segment variants (e.g. CSRD vs SEC climate = two packs, one family).

## Calendar to 20–30 apps (from Aug 2026)

- **August:** Part I Phases 0–2 with the pack-aware shell; Lupa (`research-desk`
  pack) is the proof.
- **Early September:** build the pack registry + generation/eval machinery as BFF
  endpoints (CLI-driven first — the CLI is just those endpoints), then the Pack
  Studio GUI on top; build packs #2 and #3 *through the Studio* and time-box them —
  this validates (or falsifies) the 2–3-day claim before scaling.
- **Mid-Sept → December:** cadence of ~2 packs/week using the factory line; Phases
  3–5 (streams, briefs, house style) land in the shell and upgrade *every* pack at
  once — the compounding advantage of packs-as-data.
- **Honest bottleneck:** not generation — distribution and support for 25 apps. Plan:
  one umbrella brand/site listing all desks (shared funnel + shared auth), design
  partners only for the top 3 packs, everything else self-serve; a pack that finds no
  users costs nothing to keep alive, which is the point of a factory.

## Risks

- **Thin-pack quality** → the eval ship-gate is mandatory, not advisory.
- **Integration creep** → v1 connectors are uploads + web only; MCP allowlist later;
  a pack demanding bespoke integration gets parked or promoted deliberately.
- **Brand dilution across 25 apps** → umbrella brand carries trust (receipts as the
  shared signature); individual packs are thin brands by design.
- **Multi-pack tenancy bugs** → pack resolution is gallery-path or Host-header only;
  user accounts and VFS stay strictly per-user regardless of pack; add cross-pack
  isolation tests in the BFF suite from Phase 1.
- **In-product generation quality** → a power user's Vatra-generated pack is only as
  good as their seed documents; the Studio's Review step and the UI-enforced eval
  gate are the guardrails — publishing is impossible below thresholds, regardless of
  who forged the pack.
