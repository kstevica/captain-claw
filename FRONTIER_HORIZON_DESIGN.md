# Frontier Horizon — Design Draft (v1)

Status: draft for review. A Flight Deck feature that **reproduces top-frontier
behavior (north star: Fable 5, GPT-5.6 max) from a cheaper paid model** by spending
test-time compute, gated by verifiers, escalating up the *paid* tier ladder only when
a verifier demands it. Sibling to Basna/Vatra/Council, but a different *shape*: those
parallelize **breadth** across a fleet; this drives **depth** on one model.

> Substrate is **paid models** (default driver `gemini-3-flash-preview`), not local.
> Paid calls — including the expensive tiers — are expected. The bet: cheap-tier +
> scaffolding ≈ Fable-5 / GPT-5.6-max quality at a fraction of the cost.

> Codename: **Dubina** ("depth") — placeholder, in the Basna/Vatra family.
> User-facing: "Frontier Horizon". Tables/routes use `dubina` to avoid collision.

## The one principle

> **A weak model fails long tasks because per-step errors compound (`p^n`).
> Buy back the horizon with test-time compute — but only where a verifier can
> prove a step is good. Never let the model be its own ground truth.**

Frontier models get long horizons "for free" from low per-step error rate, error
recovery, and goal coherence. We reconstruct those three externally, in the harness,
so the model only has to be good *per step*.

## Why it's a new feature, not a Basna/Vatra variant

| | Basna / Vatra / Council | **Frontier Horizon** |
|---|---|---|
| Economic contract | a **fleet** of agents | **one cheap model** + compute budget + escalation ceiling |
| Parallelism is for | division of labor (breadth) | reliability amplification (N-sample vote on one step) |
| Direction | wide | **deep** (sequential horizon on one thread) |
| Ground truth | weighted model consensus | **verifier-gated** (tests for code; agreement+critics for reasoning) |

Basna answers "split this across specialists." Frontier Horizon answers "how far can
I push one weak model before it collapses, and how cheaply."

## Architecture: one engine, two verifiers

The mistake to avoid is building two modules (reasoning, coding) that duplicate the
loop. The loop is identical; only the **verifier** and **default knobs** differ.

```
                 ┌──────────────────────────────────────────┐
                 │            Horizon Engine (core)           │
   task ───▶     │  decompose → step → verify → recover/      │  ──▶ artifact
                 │  escalate → advance   (budget-bounded)     │
                 └───────────────┬──────────────┬─────────────┘
                                 │              │
                      Verifier plug-in     Verifier plug-in
                     ┌───────────────┐   ┌───────────────────┐
                     │  Coder track  │   │ Deep-reasoning     │
                     │ ground truth: │   │ statistical:       │
                     │ tests/type/   │   │ self-consistency + │
                     │ lint/build    │   │ diverse-lens crit. │
                     └───────────────┘   └───────────────────┘
```

Two Flight Deck **sections** (UI presets), one engine, verifier as the plug-point.
A mixed task (reasoning that emits code) chains both verifiers.

### Shared core: the two axes of test-time compute

Both already exist in the codebase; the engine just composes them with a budget.

| Axis | Buys | Existing pieces to reuse |
|---|---|---|
| **Sequential** | horizon | `plan_mode` (verify→replan), `agent_pipeline_mixin` contracts (planner+critic), `session_orchestrator`, `agent_scale_loop_mixin` |
| **Parallel** | per-step reliability (raises effective `p`) | Basna spawn/weighted-merge, sampling N rollouts of a single step |

### The `Verifier` protocol (the plug-point)

```python
class Verifier(Protocol):
    async def check(self, step: Step, candidate: Candidate) -> Verdict:
        # Verdict: passed: bool, confidence: float, feedback: str
        ...
```

- **CoderVerifier** — `ground truth`. Runs the candidate against `shell`/`terminal`:
  `pytest`/typecheck/lint/build. `passed` is the checker's exit code; `feedback` is
  the failure output fed back for the fix. When no tests exist: **generate tests
  first** (spec → tests → code) so a verifier always exists. State lives in an
  isolated worktree (reuse the new **vfs** shared filesystem) so parallel fix
  attempts don't collide.
- **ReasonVerifier** — `statistical`, because there's no ground truth. Two signals:
  1. **Self-consistency**: sample the step N times, measure agreement; variance is
     the confidence proxy (the calibration a weak model lacks, manufactured externally).
  2. **Diverse-lens critics**: run the candidate past 2–3 **existing cognitive modes**
     as adversarial reviewers — `phrygian` (adversarial), `aeolian` (depth),
     `locrian` (deconstruction). Diversity beats N identical critics and softens the
     model-judges-itself ceiling. State = a claims/assumptions ledger in working memory.

  The two signals are **sequenced by cost**: agreement is always-on (cheap, also the
  difficulty trigger); critics fire only when agreement is low or the step is pivotal.
  Agreement catches *unsure*; critics catch *confidently wrong* — agreement measures
  precision, not accuracy, so it cannot stand alone.

### The escalation ladder (where #1 elevation + #3 cheap-proxy merge)

The cheap-tier model runs everything by default **and** is the cheap proxy that
*detects* when frontier-level effort is needed:

```
1. cheap tier, single pass
2.  └─ low agreement / verifier fail? → N-sample + vote        (parallel axis)
3.      └─ still failing?             → decompose deeper + critic (sequential axis)
4.          └─ still failing?         → escalate model tier      (up to max_tier)
```

Most steps stop at rung 1 (cheap). Only hard atoms climb; only the rare few hit the
expensive tier. Rungs are concrete `model.allowed` ids, per track:

| Rung | **Coder track** | **Reasoning track** |
|---|---|---|
| draft | `gemini-flash` (`gemini-3-flash-preview`) | `gemini-flash` |
| mid | `gpt-5-mini` (light coding) | `gpt-5-mini` / `claude-sonnet` |
| top | `gpt-5.3-codex` ("extremely good for coding") | `claude-opus` / `gemini-pro` ("best, expensive") |
| north star (benchmark only) | GPT-5.6 max | Fable 5 |

The north-star models aren't in `model.allowed` — they're the Phase-4 benchmark the
ladder's output is graded against, added as benchmark endpoints, not run per step.

### Budget & ceiling (the knob Basna/Vatra don't have)

The feature beats just-call-the-expensive-model only if it stays cost-efficient —
most work on cheap tiers, expensive tiers reserved for steps that earn them. Engine
config:

- `base_tier` — the `model.allowed` id the whole run starts on (rung 1).
  **User-selected** in the FD section (defaults to the track's draft rung). The
  ladder escalates from here toward `max_tier`; set `base_tier == max_tier` to pin
  the entire run to one chosen tier.
- `compute_budget` — token/$ ceiling for the whole run (cost, not "avoid paid").
- `max_tier` — highest `model.allowed` id the ladder may climb to. Default: the
  track's **top** rung (`gpt-5.3-codex` / `claude-opus`), since matching frontier
  quality is the whole point. Lower it per run to cap cost.
- `max_step_samples`, `max_fix_attempts`, `max_depth` — per-rung caps.
- On budget exhaustion: return best verified-so-far + an honest "stopped at rung N"
  note. **No silent truncation.**

## The hard limit to design around

Scaffolding elevates a model that is **capable-but-incoherent**, not one that is
**fundamentally weaker per atom**. You cannot decompose past a subtask the model
simply cannot do, and a weak model judging itself re-imports its own errors. Hence
the bias toward **ground-truth verifiers** (the coder track is the stronger of the
two for exactly this reason) and **diverse external lenses** where ground truth is
unavailable.

## Why this is mostly assembly

| Need | Reuse |
|---|---|
| Sequential horizon + verify/replan | `plan_mode.py`, `agent_pipeline_mixin.py` (contracts) |
| Parallel step sampling + merge | Basna spawn/weighted-merge (`basna_routes.py`) |
| Code execution verifier | `tools/shell.py`, `tools/terminal.py` |
| Isolated parallel attempts | **vfs** shared filesystem (latest commit) |
| Diverse critic panel | existing cognitive modes (`cognitive_mode.py`) |
| Difficulty / scale signal | `agent_scale_detection_mixin.py` |
| Model tiers for the ladder | `model.allowed` (`config.py`) |
| FD section + routes pattern | `basna_routes.py` (`/fd/basna` → `/fd/dubina`) |

New code is small: the engine loop, the two `Verifier` plug-ins, the budget/ladder
controller, and a FD section + routes.

---

# Plan

Build the coder track first — it has a ground-truth verifier, so it's the cleanest
proof that the engine elevates a weak local model. Reasoning track reuses the engine.

### Phase 0 — Engine skeleton (no UI)
- `captain_claw/dubina/engine.py`: the budget-bounded `decompose → step → verify →
  escalate → advance` loop + `Verifier` protocol + `Verdict`/`Step`/`Candidate`.
- Escalation-ladder controller reading `model.allowed` tiers + budget config.
- Unit-testable, pure where possible (mirror Basna's "pure, unit-testable helpers").
- Tests: `tests/test_dubina/test_engine.py` (ladder transitions, budget exhaustion,
  no-silent-truncation behavior) with a stub verifier.

### Phase 1 — Coder track (ground-truth verifier)
- `CoderVerifier` over `shell`/`terminal`: run tests/typecheck/lint, parse exit +
  failure text into `Verdict`.
- Spec→tests→code path when no tests exist.
- Worktree/vfs isolation for parallel fix attempts.
- Tests against a tiny fixture repo with a known-failing function the loop must fix.

### Phase 2 — Reasoning track (statistical verifier)
- `ReasonVerifier`: self-consistency sampling + agreement scoring; diverse-lens
  critics wired to `phrygian`/`aeolian`/`locrian`.
- Claims/assumptions ledger surfaced in the artifact.

### Phase 3 — Flight Deck surface
- `flight_deck/dubina_routes.py` (`/fd/dubina`), two sections (Coder, Reasoning) as
  presets over the same engine, exposing budget/ceiling/caps + the one-paid-rung opt-in.
- Split run-history tables (`dubina_coder_runs`, `dubina_reason_runs`) over a shared base.
- Live run view: which rung each step reached, tier used, budget burndown.

### Phase 4 — Measurement (closes the "simulation" claim)
- Add **Fable 5 / GPT-5.6 max as benchmark endpoints** (not in `model.allowed` yet).
- Horizon-curve harness: success-vs-task-length for (bare cheap tier) vs (Horizon on
  cheap tier) vs (north-star Fable 5 / GPT-5.6 max). Track $ cost per task alongside.
- Deliverable: "Horizon on cheap tier matches the north-star curve at X% of the cost"
  — and where it doesn't, the per-step floor that caps it. Without this the emulation
  claim is unfalsifiable.

### Resolved decisions
- **Substrate is paid models; `max_tier` default = the track's top paid rung**
  (`gpt-5.3-codex` for coding, `claude-opus`/`gemini-pro` for reasoning). Paid calls
  are expected; the cost discipline comes from the ladder resolving most steps cheap,
  not from forbidding expensive tiers. Lower `max_tier` per run to cap cost.
- **Reasoning confidence: both gates, sequenced by cost — not either/or.**
  Agreement (self-consistency) is the always-on cheap gate and difficulty trigger
  (measures *confidence/precision*). Diverse-lens critics are the gate that hard or
  pivotal steps must clear (measure *correctness/accuracy* — they catch the
  "confidently wrong" case agreement is blind to). Critics run only when agreement
  is low or the step is load-bearing.
- **Split run-history per track.** Coder and reasoning produce different artifacts
  (diffs + test results vs. answer + claims ledger) and different metrics
  (tests-passing vs. agreement/critic-survival). Separate tables (`dubina_coder_runs`,
  `dubina_reason_runs`) over a small shared base, rather than one all-nullable table.

### Still open
- Agreement threshold value, and critic-survival bar (≥2 of 3?) — tune empirically
  in Phase 2, not decided up front.
