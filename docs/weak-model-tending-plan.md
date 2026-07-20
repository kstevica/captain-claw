# Weak-model tending — one capability signal, one escalation ladder

**Status:** plan. Phase 0 (the bleeding-stopper) is implemented; Phases 1–4 are not started.
**Origin:** a Deep Researcher agent on `ollama:ornith:9b` produced two empty chat bubbles and
one silent half-finished report. Diagnosis in "The incident" below.

---

## The incident

`fd-data/deep-researcher-2vkj/` — a Deep Researcher spawned at the `reason` tier
(`input_ctx: 200000`), then pointed at a local 9B. Its `config.yaml`:

```yaml
model:   {provider: ollama, model: ornith:9b, max_tokens: 65536}
context: {max_tokens: 200000}
```

Three failures, all in `process.log`:

1. **08:11:32** — the turn ended on
   `"Great data from GitHub. Let me grab a few more key sources for depth."`
   with no tool calls. `_looks_like_stall()` missed it: the regex is anchored to the
   start of the first line and the model prefixed an observation; the string is also
   69 chars, over `_STALL_MAX_LEN = 60`. The half-finished report shipped as the answer.
2. **08:13:28 and 08:14:10** — `content_len=0` on every call. The stall retry fired
   (`attempt=1`, `attempt=2`) and still got nothing, because its strongest lever —
   `_tool_choice_override = "required"` — is read only by `ChatGPTResponsesProvider`
   and `LiteLLMProvider`. `OllamaProvider` never looked at it. A silent no-op on
   exactly the provider that needs it.
3. Then `output_text_len=0` → `All completion gates passed` → `response_len=0`.
   **An empty string was returned as a successful turn.** The user saw a blank bubble.

Underneath all three: `num_ctx=200000` on a 9B. Ollama allocates the KV cache for the
full window up front — order tens of GB — which spills to swap. The Flight Deck card
read **0.2 tok/s**. The empty generations are almost certainly downstream of that.

---

## The root problem

There is no single notion of "this model is weak". There are three, and they do not
talk to each other:

| Signal | Where | Read by |
|---|---|---|
| Tier name (`micro`) | `flight_deck/archetype_routes.py:29` | spawn only |
| `_MODEL_TIERS` / `_model_rank()` substring heuristic | `flight_deck/consciousness.py:424-473` | consciousness only |
| `eco_mode.txt` / `nano_mode.txt` / `mrav_mode.txt` flag files | `instructions.py:84-96`, `agent.py:34-56` | prompt + tool-list assembly |

**The agent turn loop reads none of them.** `ornith:9b` matches nothing in the
substring table (there is no `9b` entry), so it ranks 30 — "unknown" — and every
subsystem treats it exactly like Opus.

Meanwhile the machinery to nurse a weak model to completion is *already written*,
just not reachable from the standalone loop:

| Mechanism | Lives in | Available to the agent loop |
|---|---|---|
| `worker_produced_nothing`, `ACTED_CORRECTIVE`, `ESCALATE_DIRECTIVE` | `flight_deck/quality_profile.py:276-330` | ✗ |
| `_acted()` gate, `_BUILD_RETRIES`, `_no_change_corrective()` | `flight_deck/code_routes.py:1094-1130` | ✗ |
| `_GIVE_UP_STREAK=4`, `_ESCALATE_STREAK=2`, repeat-call guards | `mrav/runtime.py:56-57, 417-523` | ✗ |
| small→big escalation | `code_routes.py:717-780`, `dubina_routes.py:232-310` | ✗ |
| micro→full-body fallback on empty (`micro_fallback_body`) | `being_micro.py:238`, `being_life.py:2598` | ✗ |
| grammar-locked JSON via Ollama `format` | `llm/__init__.py` | ✗ (loop never calls `complete_structured`) |

So the work is not "build a nurturing system". It is **collapse three signals into one,
then let the guards that already exist read it.**

---

## Design rule

> Every lever is derived from a capability signal that is `frontier` for hosted models.
> On the `frontier` path, behaviour is byte-identical to today.

This mirrors the `quality_profile` convention already used by Code/Basna/Vatra: opt-in
levers, off-path unchanged. No lever below may fire on a model that isn't measurably weak.

---

## Phase 0 — stop the bleeding (DONE)

Implemented ahead of the rest because each fix is independently correct regardless of
how the phases below land.

| # | Fix | File |
|---|---|---|
| 1 | `OllamaProvider` honors `_tool_choice_override` by switching to a grammar-constrained tool-call schema (`_forced_tool_call_schema`), then converting the JSON back into a real `ToolCall` | `llm/__init__.py` |
| 2 | Empty output can no longer finalize as success — retries while budget remains, then returns `MSG_EMPTY_RESPONSE` with `success=False` | `agent_completion_mixin.py`, `agent_stuck.py` |
| 3 | Stall retries no longer write the empty assistant turn back into history; empty output gets its own corrective instead of the "you narrated instead of acting" scolding | `agent_orchestration_mixin.py` |
| 4 | Stall detection also inspects the **closing** sentence of a short message; `_STALL_HANDBACK_RE` vetoes "Let me know if…" sign-offs on both paths | `agent_orchestration_mixin.py` |
| 5 | `num_ctx` clamped to 32768 for local Ollama models (`:cloud` exempt, `CLAW_OLLAMA_MAX_NUM_CTX` overrides); the history trimmer follows the provider's real window instead of `context.max_tokens` | `llm/__init__.py`, `agent_context_mixin.py` |

Tests: `tests/test_weak_model_tending.py` (23).

**Known gap:** fix 4 widens a heuristic. It is now stricter *and* looser than before —
looser on prefixed stalls, stricter on hand-back closers (which it previously
misclassified as stalls, affecting strong models). Both directions are covered by tests,
but heuristics drift; if false re-rolls appear, `_STALL_TRAILING_MAX_LEN` is the dial.

---

## Phase 1 — one capability signal

New module `captain_claw/model_capability.py`:

```python
class Capability(StrEnum):
    FRONTIER = "frontier"   # hosted flagship — no levers
    MID      = "mid"        # hosted small / strong local — light levers
    WEAK     = "weak"       # small local — full ladder
    MICRO    = "micro"      # ≤4B — mrav territory

def model_capability(provider: str, model: str, *, num_ctx: int = 0) -> Capability
```

Resolution order, first hit wins:

1. **Explicit tier** on the agent config (`micro` → `MICRO`, `reason`/`coding` → `FRONTIER`).
2. **Hosted provider** (anthropic/openai/gemini/xai) → `FRONTIER`, always. This is the
   guarantee that hosted models never see a lever.
3. **Local probe** — for ollama, `GET /api/show` once per model, cached. Parameter count
   and real `context length` come straight from the response
   (`ornith:9b` → `parameters 9.0B`, `context length 262144`). ≤4B → `MICRO`,
   ≤14B → `WEAK`, else `MID`.
4. **Fallback** — the existing `consciousness._MODEL_TIERS` substring table, extended
   with the parameter-count patterns it is missing (`7b`, `8b`, `9b`, `12b`, `14b`).

Then rewrite `consciousness._model_rank()` to delegate here, so the substring table
stops being a second source of truth.

**Effect on strong models:** none. Step 2 short-circuits before any probe.

## Phase 2 — capability-driven budgets

Replace the Phase-0 constant ceiling with a derived one:

- `num_ctx = min(config, probed model context, memory-derived ceiling)`. The memory term
  is the real fix — a 9B at Q4 with ~30 GB free supports a much larger window than the
  same model on a 16 GB box, and 32768 is a guess that is wrong in both directions.
- Auto-set `think: False` for `WEAK`/`MICRO` (mrav already forces this at
  `mrav/prompts.py:62-68`). Every empty generation in the incident came from a
  thinking-enabled model; every successful one had a `thinking` key.
- Auto-enable eco tool trimming at `WEAK`, nano at `MICRO`, instead of requiring three
  manual flag-file toggles per agent.

Flag files stay authoritative when explicitly set — auto only fills the unset case, so
nobody's deliberate configuration is overridden.

## Phase 3 — the corrective ladder in the agent loop

Today `MAX_STALL_RETRIES = 2` re-rolls the *same* prompt twice and gives up. Replace with
an escalating ladder, each rung only unlocked for the capability that needs it:

| Rung | Action | Min capability |
|---|---|---|
| 1 | Prose corrective (today's behaviour) | all |
| 2 | Forced tool call via grammar (Phase 0 fix 1) | `WEAK` |
| 3 | Drop to the nano tool allowlist for one turn — a small model picking from 39 tools is the documented failure | `WEAK` |
| 4 | Hand the turn to the mrav step loop, which has its own give-up and repeat-call guards | `MICRO` |
| 5 | Escalate to the next tier up and re-run, mirroring `code_routes.py:717-780` | opt-in |

Rung 5 needs a policy decision (cost, and whether a local-only deployment has anywhere
to escalate *to*) — default off, surfaced as an agent-card toggle.

## Phase 4 — make progress legible

`_has_turn_progress()` (`agent_pipeline_mixin.py:1552-1571`) counts unique write paths,
unique tool signatures, and pipeline index — **assistant text is explicitly not
progress.** That is correct for pipeline work and wrong for conversation, and it is the
documented cause of the "I got stuck…" trap on topic switches.

Fix: count a substantive text answer as progress when there is no pipeline and no
completion requirements — the same condition the salvage path at
`agent_orchestration_mixin.py:1528-1532` already uses to decide a turn was
conversational. Narrow, and it removes the need for the salvage special-case.

---

## Deferred

- **Auto-escalation across machines** (spill a stuck local turn to a hosted model).
  Needs a cost policy first.
- **Per-model reliability learning** — Basna already tracks reliability per worker;
  feeding observed stall rate per local model back into `Capability` would let the
  ladder start at the right rung instead of climbing from 1 every time.
- **Feeding self-reflections back as constraints.** The agent already diagnosed itself
  in the incident (08:14:11: *"Stop narrating your intent before calling tools"*) and
  then ignored it. Reflections are generated and stored but never injected as hard
  turn constraints.

---

## Verification

Phase 0 is unit-tested but **not yet live-verified** — that needs a Flight Deck restart
and a re-run of the Deep Researcher on `ornith:9b`. The observable pass criteria:

- `Clamping Ollama num_ctx` appears once in `process.log` at agent start
- throughput recovers from 0.2 tok/s
- a stalled turn logs `Ollama forced tool call via grammar` and then a real tool call
- no turn ever returns `response_len=0` with `public=True`
