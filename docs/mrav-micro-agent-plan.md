# Mrav — a micro agentic runtime for small models (≤8k input per call)

Status: PLANNED (2026-07-18). Parallel system — the existing 16-mixin `Agent` is untouched.
Name: "Mrav" (ant) — small, cheap, works in colonies. Rename is a find/replace away.

## Why

The current agentic stack is excellent but structurally incompatible with small models
(Gemma 4 E2B/E4B, Qwen3.5-4B class). Measured on main:

- Standard system prompt template: **~7,443 tokens** of prose before block substitution
  (`instructions/system_prompt.md`, 29,771 chars). Micro = 1,075 tok, nano = 366 tok.
- All 60 registered tool schemas as sent to the model: **~25,900 tokens** (103,688 bytes).
  Nano 16-tool subset = ~4,656 tok; eco core = ~6,762 tok.
- Context budgeting assumes `context.max_tokens = 160_000` (config.py:73); WorkingMemory
  caps at 100k; ~19 per-turn context-note injectors ride on top of history.

Nothing short of a parallel runtime gets a full agentic loop under a **hard 8,192-token
input cap per LLM call**. That cap is the design center: 2026-era 2–4B models are reliable
well past 8k (RULER-class evals put degradation far above), so 8k buys headroom, speed,
tiny KV caches, and browser feasibility all at once.

Existing in-repo precedents Mrav builds on (not replaces):

- **feet** (`flight_deck/being_instinct.py`): 1k-token capped one-shot, verb whitelist,
  junk-tolerant `parse_feet_act`. The proof that capped one-shots work in production.
- **faculties** (`being_life.py:_run_faculties`): one tick decomposed into 2–5 tiny
  sequential LLM calls with a per-faculty model seam. The step-decomposition pattern.
- **nano/eco modes** (`instructions.py:130-153`, `_NANO_TOOLS`/`_ECO_CORE_TOOLS` in
  `agent_orchestration_mixin.py:128-193`): prompt + tool reduction levers.
- **chunked processing** (`agent_chunked_processing_mixin.py`): map-reduce over oversized
  content — reused for digesting big tool results.
- **LiteRT path** (`llm/__init__.py:2699-2932`): text-protocol tool manifest + lenient
  extraction — the only code already living at the 8k boundary.

## Research grounding (verified 2026-07-18)

### Models (local via Ollama, q4)

| Model | Eff. params | q4 disk | Ctx | Native FC | Evidence | Notes |
|---|---|---|---|---|---|---|
| **Gemma 4 E4B** (3/2026) | 4.5B | ~5 GB | 128k | yes (special tokens + think) | τ²-bench 42.2, IFEval 96.7 | best IF in class; multimodal |
| **Gemma 4 E2B** (3/2026) | 2.3B | 3.35 GB (QAT) | 128k | yes | τ² 24.5, IFEval 94.6 | short leashes only — weak multi-hop |
| **Qwen3.5-4B** (3/2026) | ~4B | 3.4 GB | 262k | yes (Hermes-style) | **97.5% independent tool eval** | current champion; run non-thinking |
| Qwen3.5-2B / 0.8B | 2B / 0.8B | <2 GB | 262k | yes | AA II 16 / 9 | browser class |
| Nemotron 3 Nano 4B (3/2026) | 4B hybrid Mamba | ~4.2 GB | 49k | yes (RL-trained) | 95% same eval | dark horse; NVIDIA license |
| Granite 4.1-3B (4/2026) | 3B | ~2 GB | long | yes, no-CoT design | family FC focus | predictable token usage |
| Granite 4.0 Nano 1B | ~1.5B | ~1 GB | 128k | yes | BFCLv3 54.8 (best 1–2B) | browser-capable |
| LFM2-1.2B-Tool / LFM2.5 | 1.2B | 0.7 GB | 32k | yes (Pythonic) | IFEval 86.2 (2.5) | dispatcher tier |

Gemma 3n E2B/E4B: superseded — **no native FC template**; skip for agents, keep LiteRT
compat only. Llama 3.2 1B/3B, Phi-4-mini: outclassed (format failures / weak args).

Reference roster v1: `gemma4:e2b` + `gemma4:e4b` (QAT q4, user's anchor), `qwen3.5:4b`
(recommended default, non-thinking), `granite4.1:3b` (alternate). All Ollama-pullable.

### Reliability techniques (all mandatory at this size)

1. **Grammar-constrained decoding, always.** Ollama structured outputs (`format` = JSON
   schema, GBNF-backed), llama.cpp `json_schema`/GBNF, WebLLM xgrammar. Gotcha: the
   grammar does NOT inject the schema into the prompt — the prompt must still describe
   the expected shape. Grammar guarantees syntax, not semantics — keep whitelists +
   arg validation + retry-with-error.
2. **Native chat-template FC beats hand-rolled ReAct** — but only with the exact
   template. Never hand-roll message formats; template mismatch is the #1 cause of
   "specialist model scores worse than generalist" results. Gemma 4 emits calls as
   special tokens (`<|tool_call|>…`) — rely on the Ollama/llama.cpp shipped templates.
3. **One tool call per step. Small visible tool registry.** Irrelevance detection is
   the classic small-model failure.
4. **Non-thinking modes in loops** (`/no_think`, `enable_thinking=False`); strip stray
   `<think>` blocks defensively. Thinking multiplies output tokens 3–10×.
5. **Prompt-format stability**: byte-stable prefixes (also feeds prefix/KV caching).

### Browser inference (mid-2026)

- **WebGPU is default everywhere on desktop**: Chrome/Edge (2023+), Safari 26 (9/2025),
  Firefox Windows 141 / macOS 145. ~70–80% global coverage.
- **WebLLM** (`@mlc-ai/web-llm` 0.2.84, pin it): the only runtime with **enforced**
  JSON-schema output (xgrammar), OpenAI-shaped streaming API, automatic KV delta-prefill
  on append-only conversations, and **no COOP/COEP requirement**. Catalog: Qwen3
  0.6B/1.7B/4B, Llama 3.2, gemma-3-1b (711 MB), SmolLM2, Qwen3.5-0.8B already converted.
  Qwen3-4B q4f16 @ 8k ctx ≈ 4.0–4.3 GB GPU mem → 16 GB machines; Qwen3-1.7B ≈ 2.5 GB.
- **wllama v3.5** (llama.cpp WASM + WebGPU): any GGUF, GBNF; needs COOP/COEP; Memory64
  main build excludes Safari (compat pkg exists). **Deferred** — CPU fallback tier.
- **LiteRT-LM Web** (7/2026 preview): Gemma 4 E2B/E4B in-tab at ~76 tok/s (MBP),
  `maxNumTokens: 8192` — but text-only, two models, **no constrained output**. Watch;
  revisit ~Q4 2026.
- **Chrome/Edge Prompt API** (stable Chrome 148): free built-in model, 9,216-token ctx,
  JSON-schema `responseConstraint` — opportunistic tier-0 only, never a dependency.
- **The binding constraint is prefill, not decode**: a cold 8k prefill in-tab costs
  10–40 s (worse on iGPU). Architecture must keep a **stable prompt prefix and append**
  so WebLLM's delta-prefill (and Ollama's prefix cache) hit every step.
- Prior art for our exact topology exists (tab registers as inference worker over WS,
  server orchestrates; e.g. built-in-ai, OllaBridge pattern, WebLLM ServiceWorker mode).

## Decisions (locked with Stevica, 2026-07-18)

1. **Surfaces, in order**: S1 lite chat agent → S2 Iskra beings cognition → S3
   Basna/Vatra micro workers. All three are wanted.
2. **Browser worker ships in v1** alongside Ollama (not deferred).
3. **Tool surface**: curated core (~12 compact schemas, always loaded) + one-line index
   of the full registry + `open_tool` paging on demand.
4. **Escalation**: opt-in per-agent/per-run toggle, **default off**; after N failures a
   single step retries on the owner's `fast`/`balanced` tier, logged + metered.

## Design principles

1. **Hard ledger, enforced at assembly.** Every LLM call ≤ 8,192 input tokens, counted
   before send (reuse `memory.py:estimate_tokens`, conservative chars/3.6 + margin), not
   hoped for. Output cap 1,024 default. If a section overflows its budget, it is
   truncated/digested — the call is never sent oversized.
2. **State lives outside the model.** A blackboard (task card, plan, facts, observations,
   rolling summary) is the memory; each step re-renders a prompt from it. No 160k
   history, no 19 injectors.
3. **Stable prefix, appended tail.** Frozen per session: step contract + toolpack +
   index + task card. Only state/observations change, ordered oldest→newest so the
   mutation point is as late as possible. This makes Ollama/llama.cpp prefix caching and
   WebLLM delta-prefill hit on every step.
4. **One tool per step, schema-enforced.** Every step response is grammar-constrained
   JSON; validate → repair → retry-with-error (≤2) → fail honestly.
5. **Same shell.** A Mrav agent is indistinguishable at the transport layer — same
   agent process, same aiohttp `/ws`, same FD chat UI.
6. **Honest failure over silent upgrade.** Escalation exists but is off by default and
   always visible in the step trace.

## Architecture

### New module: `captain_claw/mrav/`

```
mrav/
  runtime.py    # MravRuntime — the loop: plan → act → observe → digest/compress
  ledger.py     # TokenLedger — per-section budgets, counting, tail-trunc helpers
  state.py      # Blackboard — task card, plan, facts, observations, rolling summary;
                # JSONL event log per session (append-only, resumable)
  toolpack.py   # compact core schemas, one-line index, open_tool paging,
                # arg validation against the REAL registry schema before execution
  protocol.py   # step dataclasses + JSON schemas per step type (ACT/PLAN/DIGEST/COMPRESS)
  prompts.py    # per-step micro prompts (≤500 tok) + per-model quirk table
  digest.py     # map-reduce digestion of oversized tool results (adapts chunked-processing)
  eval/         # canned eval tasks + runner (see Eval harness)
```

`MravRuntime` takes: a resolved tier (provider/model/creds), a `ToolRegistry` handle
(the existing one — tools themselves are shared, only their presentation differs), an
owner/session identity, config. It exposes `complete(user_input)` / `stream(user_input)`
mirroring `Agent`'s public surface so the web server can swap runtimes.

### Step types (each = one LLM call ≤8k)

- **ACT** (the workhorse): given contract + toolpack + task card + state → exactly one of
  `{"tool": name, "args": {...}}` | `{"open_tool": name}` | `{"final": text}` |
  `{"give_up": reason}`. Grammar-enforced union schema.
- **PLAN / REPLAN**: first step, then every `replan_every` (default 6) ACTs or on
  stagnation → ≤5-bullet plan + prune facts ledger.
- **DIGEST**: tool result over `observation_cap` (default 2,500 tok) → ≤`digest_target`
  (400 tok) observation. Results over ~6k go through map-reduce chunks first.
- **COMPRESS**: observation list near budget → fold oldest into the rolling summary.

DIGEST/COMPRESS may run on a smaller/cheaper model than ACT (per-step model seam,
mirroring the faculties roadmap). Steps are sequential; one request in flight.

### ACT prompt layout (ledger budgets, 8,192 total)

| Section | Budget (tok) | Mutability |
|---|---|---|
| Step contract (system) | ~450 | frozen |
| Toolpack: core schemas | ~1,800 | frozen (+ pinned pages) |
| Tool index (one-liners) | ~700 | frozen |
| Task card | ~300 | frozen per task |
| Plan + facts ledger | ~700 | slow-changing |
| Rolling summary | ~600 | slow-changing |
| Observations (last K, tail-trunc) | ~2,500 | append |
| Step instruction + retry error | ~300 | per step |
| Reserve / count-error margin | ~842 | — |

### Structured output layer (shared — benefits the whole codebase)

Add an optional `response_schema: dict` to provider `complete()`/`stream()`:

- **OllamaProvider** (`llm/__init__.py:1698`): pass as `format` (native structured
  outputs, GBNF-backed). Primary local path.
- **LiteLLM/OpenAI-compatible**: `response_format={"type":"json_schema",…}` where the
  backend supports it; else prompt + validate.
- **BrowserProvider** (new, below): `response_format:{type:"json_object", schema}` →
  WebLLM xgrammar, enforced in-tab.
- **LiteRTProvider**: no grammar available — prompt + strict validation only.

Shared parser ladder (new `mrav/protocol.py`, generalizing `parse_feet_act` +
`_litert_extract_tool_calls`): strict `json.loads` → common repairs (trailing commas,
fences, single quotes) → first-JSON-object regex → retry-with-error-message → fail.
The schema is ALSO described in the prompt (grammar doesn't inject it).

### Toolpack

- **Core (~12, hand-compacted to ~80–150 tok each, ~1.8k total)** — candidate set, final
  pick during Phase 1 against the real registry names: file read / file write / list dir /
  shell (guarded) / web_search / url fetch / memory search (semantic) / todo / vfs read-write
  (when bound) / `open_tool` / `final`. `edit` is arg-heavy — start with write-whole-file,
  revisit.
- **Index**: one line per remaining tool (`name — ≤8-word description`), auto-generated
  from registry descriptions then hand-trimmed; ~700 tok for ~48 tools.
- **Paging**: `open_tool(name)` pins that tool's compacted full schema into the toolpack
  for subsequent steps (LRU, max 3 pinned; pinning evicts oldest). Costs one step.
- **Validation**: args are validated against the tool's REAL JSON schema before
  execution; mismatch → retry-with-error, never a crashed tool call.
- **Drift test**: CI test asserts every compact schema's params ⊆ real schema params,
  so registry changes can't silently strand the micro copies.
- **Observation caps**: per-result head+tail smart truncation at `observation_cap`;
  DIGEST when over.

### Tiers & model selection

- Add `micro` to the valid tier set (`basna_routes.py:77`) and the forge-tiers UI.
  Shape unchanged: `{provider, model, base_url, input_ctx, output_ctx, api_key}`.
- Resolution order for a Mrav agent: explicit AgentConfig model → owner tier `micro` →
  tier `fast` → error (honest).
- `input_ctx` from the tier overrides the 8,192 default cap downward (e.g. 4k experiments),
  never upward past 8,192 — the cap is Mrav's contract, not a suggestion.
- Per-model quirks table in `prompts.py`: Gemma 4 (tool tokens via Ollama template,
  optional `<|think|>` off), Qwen3.5 (`/no_think`), Granite (plain). Reuses the existing
  temperature-quirk machinery in `llm/__init__.py:475-505` untouched.

### Browser inference worker (v1)

Topology: **tab = inference worker, FD = broker, agent process = client.** The tab never
executes tools; it only turns (messages, schema) into tokens. Prod box needs no GPU —
the user's own browser is the compute.

- **flight-deck** (`src/inference/`):
  - `engine.ts`: `WebWorkerMLCEngine` wrapper (WebLLM pinned 0.2.84), model ladder chosen
    from `navigator.gpu` adapter limits + `navigator.deviceMemory`:
    ≥16 GB → Qwen3-4B-q4f16 (ctx 8192); 8 GB → Qwen3-1.7B; floor → gemma-3-1b / Qwen3.5-0.8B.
    Weights cached (Cache API default; OPFS if trivial), reload = cache hit.
  - `workerLink.ts`: connects `wss://…/fd/infer-ws`, registers capability
    `{engine:"webllm", model, ctx_max, schema:true, vram_est}`, then serves jobs:
    request `{job_id, messages[], response_schema?, max_tokens, abort?}` →
    streamed `{job_id, delta}` chunks → `{job_id, done, usage}`. Heartbeat both ways.
  - UI: a "Local inference" panel (Sidebar → System or Settings): status, model picker,
    download progress, tok/s, jobs served. Explicit opt-in toggle — a tab never becomes
    a worker silently. Warn on hidden-tab throttling (`document.visibilityState`).
- **FD server** (FastAPI): `@app.websocket("/fd/infer-ws")` — auth as the owner, register
  worker in an in-memory registry `{owner_id → [workers]}`; broker endpoint(s) for
  clients: `POST /fd/infer/complete` (SSE/chunked stream back) or an internal asyncio
  queue for in-process callers (beings). Job routing: pick the owner's worker whose
  `ctx_max ≥ need`, **pin session→worker** for KV reuse, requeue on worker loss.
- **Backend provider**: `BrowserProvider` in `llm/__init__.py` (`create_provider`
  branch `provider="browser"`, `base_url` = FD). `complete(messages, response_schema)` →
  FD broker → tab → streamed back. Timeouts (prefill-aware: first-token timeout ≥60s
  cold, then per-token), abort → `engine.interruptGenerate()`, worker-gone → explicit
  error `"no inference worker online"` (or fallback to the tier's Ollama config if the
  owner set `mrav.browser_fallback: ollama`).
- **Append-only discipline**: BrowserProvider callers (MravRuntime) must extend, never
  edit, the message list for a pinned session — that's what makes step latency ~1–3 s
  instead of 10–40 s.
- **Security**: a worker serves only its owner's sessions (`owner_id` match on both
  ends); no cross-user routing in v1. Resource-sharing integration deferred.
- Out of scope v1 (seams left in the capability record): wllama CPU fallback,
  Chrome Prompt API tier-0, LiteRT-LM Web (Gemma 4 in-tab), tab-grid compute donation.

### Surfaces

- **S1 — Lite chat agent**: `AgentConfig` (flight_deck/server.py:1028) gains
  `runtime: "classic" | "mrav"` (default classic). The agent process instantiates
  `MravRuntime` instead of `Agent` behind the same aiohttp `web_server.py` chat path —
  FD chat UI works unchanged. FD spawn/settings UI: runtime picker + "mrav" badge on
  the agent card. Step trace visible (see Observability).
- **S2 — Iskra beings**: route faculties steps (orient/act/journal first) through
  MravRuntime steps with the being's toolset as the toolpack; per-faculty model routing
  lands here for free (ACT on `micro`, connect/dream stay on bigger tiers). Feet stay
  exactly as-is. Compact mode + Mrav = a being whose entire waking cognition fits small
  local/browser models.
- **S3 — Basna/Vatra micro workers**: opt-in `quality_profile` lever `micro_workers` —
  Vatra worker roles that are extraction/digest/format-shaped run a MravRuntime loop
  (VFS toolpack) instead of a full spawned Agent process; Basna one-shot calls gain
  `response_schema` enforcement (cheap immediate win). Off-path byte-identical, per
  cross-pollination house rules.

### Escalation (opt-in, default off)

- Config `mrav.escalate: "off" | "fast" | "balanced"` (+ per-run override in
  quality_profile / AgentConfig).
- Trigger: 2 consecutive failed ACTs (parse fail after retries, invalid tool/args twice,
  or identical tool+args repetition) or stagnation (no new observation in 3 steps).
- Action: ONE step re-run on the escalation tier, marked `escalated: true` in the trace,
  metered via `debit_usage(note="mrav-escalate")`; then back to micro.

### Observability & eval

- **Step trace**: JSONL per session (`ts, step_type, model, in_tok, out_tok, tool,
  ok/fail, retries, escalated, duration`), written next to session data; endpoint
  `GET /fd/mrav/trace/{session}` + a simple viewer in the agent panel (Phase 2 polish).
- **Metering**: `debit_usage(tier="micro", note="mrav:<step>")` per call — beings' wallet
  math keeps working.
- **Eval harness**: `scripts/mrav_eval.py` — 15–20 canned tasks (file ops, multi-step
  web lookup, digest-a-big-file, tool-paging reach, deliberate trap/irrelevance cases)
  runnable against any tier config → pass/fail + tokens + $ table. This is the gate for
  adding models to the blessed roster, and the regression net for prompt changes.

### Config sketch (config.yaml)

```yaml
mrav:
  enabled: true
  input_cap: 8192        # hard, per LLM call, everything included
  output_cap: 1024
  observation_cap: 2500
  digest_target: 400
  max_steps: 24
  replan_every: 6
  escalate: "off"        # off | fast | balanced
  tier: "micro"          # forge-tiers key; falls back to "fast"
  browser_fallback: "ollama"   # ollama | none — when no worker tab online
```

## Phases

**Phase 1 — Core runtime + structured outputs (backend, S1).**
`mrav/` module (ledger, blackboard, protocol, prompts, toolpack core+index+paging,
digest, runtime loop), `response_schema` on OllamaProvider + parser ladder, `micro`
tier plumbing, `AgentConfig.runtime` + web_server swap, eval harness + tests
(ledger property test: no assembled prompt ever exceeds cap; schema drift test; parser
fuzz; paging LRU; digest map-reduce). Validate against `gemma4:e2b`, `gemma4:e4b`,
`qwen3.5:4b` on the eval set. Deliverable: a spawnable lite agent in FD chat on Ollama.
*Restart: backend. FD bundle: only the spawn-UI picker (small).*

**Phase 2 — Browser worker.**
FD `/fd/infer-ws` + registry + broker, `BrowserProvider`, flight-deck `src/inference/`
(WebLLM worker, model ladder, OPFS/Cache caching, Local-inference panel), session→tab
pinning + append-only enforcement, abort/heartbeat/fallback. Lite agent runs with
`micro` tier = `{provider: browser}` end-to-end. *Restart: FD + backend; flight-deck
`npm run build` + commit bundle (deploy discipline).*

**Phase 3 — Beings (S2).**
Faculties→Mrav step routing with per-faculty tiers, being toolpack, compact-mode
synergy, wallet metering notes, Care-adjacent UI toggle for escalation. Measure: a
village day's cognition cost on `micro` vs today.

**Phase 4 — Ensemble workers (S3) + hardening.**
`quality_profile.micro_workers` lever for Vatra extract/digest/format roles, Basna
one-shot `response_schema`, trace viewer polish, roster re-bless via eval harness.

**Deferred (explicit non-goals for now):** wllama CPU fallback tier, Chrome Prompt API
tier-0, LiteRT-LM Web adoption (revisit ~Q4 2026 — Gemma 4 E2B in-tab at 76 tok/s is
attractive once it gains constrained output), tab-grid compute sharing across users,
in-browser embeddings retrieval (EmbeddingGemma/MiniLM) for sub-8k context packing,
`edit` tool in core pack, cross-user worker sharing via resource_shares.

## Gotchas / risks

- **Gemma 4 FC syntax is special-token based** — always go through the Ollama/llama.cpp
  shipped chat template; never hand-roll. Verify template presence on model pull; the
  eval harness catches template rot.
- **Grammar ≠ semantics**: schema-valid nonsense still happens — whitelists, real-schema
  arg validation, retry-with-error stay mandatory even with xgrammar/Ollama format.
- **Prompt-format sensitivity is extreme at 2–4B** (documented swings of tens of points
  from formatting alone) — byte-stable prefixes, no cosmetic prompt churn; prompt changes
  go through the eval harness.
- **8k counts EVERYTHING** (system + toolpack + index + state + instruction). The ledger
  is the only authority; token estimation errs conservative with an ~800-tok reserve.
- **Browser tab lifecycle**: Chrome throttles/freezes background tabs and Memory Saver
  discards them — v1 requires a visible opted-in tab; ServiceWorker engine + wake locks
  are a later refinement. Worker loss mid-job must requeue or fail cleanly.
- **Cold prefill in-tab is 10–40 s** — the first step after model load is slow; UI must
  show "warming up", and session pinning must actually stick or every step pays it.
- **WebLLM maintenance pace is slow** — pin 0.2.84, treat runtime+weights as static
  assets we serve; the OpenAI-shaped seam means wllama/LiteRT-LM can slot in later.
- **Qwen thinking leakage**: force non-thinking and strip `<think>…</think>` defensively.
- **E2B is a short-leash model** (τ² 24.5, GraphWalks collapse): default E2B configs to
  lower `max_steps` and more frequent REPLAN; E4B/Qwen3.5-4B for longer tasks.
- **FD frontend build discipline**: editing TS isn't enough — `npm run build` in
  flight-deck/ and commit the bundle, or prod stays stale.
- **Mobile browsers are not v1 targets** (iOS ~1.5–2 GB tab kills, thermal throttling):
  detect and route mobile users to the Ollama/server path.

## Key sources

- Gemma 4: ai.google.dev/gemma/docs/releases · function-calling-gemma4 doc · arXiv 2607.02770
- Qwen3.5 small: Qwen announcement + artificialanalysis.ai/articles/qwen3-5-small-models
- Independent tool eval (Qwen3.5-4B 97.5%, Nemotron 95%): jdhodges.com local-LLM tool-calling 2026
- Constrained decoding: github.com/ggml-org/llama.cpp grammars · xgrammar (arXiv 2411.15100) · Ollama structured outputs
- Effective context: RULER (arXiv 2404.06654) · NoLiMa
- WebLLM: github.com/mlc-ai/web-llm · arXiv 2412.15803 · LiteRT-LM Web: developers.google.com/edge/litert-lm/js
- Browser platform: caniuse.com/webgpu · web.dev/articles/coop-coep · storage quotas (MDN)
