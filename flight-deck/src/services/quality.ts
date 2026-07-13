// Quality/cost profile — mirrors captain_claw/flight_deck/quality_profile.py.
// Every lever is opt-in; an all-off profile == the systems' current behaviour.

export type OutputMode = '' | 'complete' | 'conservative'

export type QualityProfile = {
  profile: 'off' | 'balanced' | 'thorough' | 'custom'
  // Code-side
  test_gate: boolean
  deep_build: boolean
  coverage_check: boolean
  parallel_build: boolean
  deep_build_samples: number
  parallel_build_max_slices: number
  // Research-side (Basna / Vatra)
  acted_gate: boolean
  research_map: boolean
  delta_rounds: boolean
  critic_triage: boolean
  worker_escalate: boolean
  git_snapshots: boolean
  judgment_ledger: boolean
  source_corpus: boolean
  claim_check: boolean
  rubric_contract: boolean
  intent_brief: boolean
  consistency_check: boolean
  facts_ledger: boolean
  constraints_contract: boolean
  block_on_critical: boolean
  claim_check_max: number
  block_max_rounds: number
  // Calibration posture — honesty_guard is the ONE default-on flag; it is
  // preset-independent (mirrors the backend: presets never touch it, and it is
  // excluded from preset derivation so switching it off doesn't read "custom").
  honesty_guard: boolean
  output_mode: OutputMode
  // Shared
  token_budget: number
}

export const BOOL_FLAGS = [
  'test_gate', 'deep_build', 'coverage_check', 'parallel_build', 'acted_gate', 'research_map',
  'delta_rounds', 'critic_triage', 'worker_escalate', 'git_snapshots',
  'judgment_ledger', 'source_corpus', 'claim_check', 'rubric_contract', 'intent_brief',
  'consistency_check', 'facts_ledger', 'constraints_contract', 'block_on_critical',
] as const
export type BoolFlag = (typeof BOOL_FLAGS)[number]

export function defaultProfile(): QualityProfile {
  return {
    profile: 'off',
    test_gate: false, deep_build: false, coverage_check: false, parallel_build: false,
    deep_build_samples: 2, parallel_build_max_slices: 6,
    acted_gate: false, research_map: false, delta_rounds: false, critic_triage: false,
    worker_escalate: false, git_snapshots: false,
    judgment_ledger: false, source_corpus: false, claim_check: false, rubric_contract: false,
    intent_brief: false,
    consistency_check: false, facts_ledger: false, constraints_contract: false,
    block_on_critical: false,
    claim_check_max: 8,
    block_max_rounds: 2,
    honesty_guard: true,
    output_mode: '',
    token_budget: 0,
  }
}

// Preset → flags ON. Mirrors the backend _PRESETS exactly. deep_build,
// claim_check and block_on_critical are the paid levers — never in a preset
// (explicit opt-in only). honesty_guard is not preset material at all: it
// defaults ON and only an explicit switch turns it off.
const PRESETS: Record<'off' | 'balanced' | 'thorough', BoolFlag[]> = {
  off: [],
  balanced: ['acted_gate', 'test_gate', 'research_map', 'delta_rounds', 'critic_triage',
    'worker_escalate', 'judgment_ledger'],
  thorough: ['acted_gate', 'test_gate', 'research_map', 'delta_rounds', 'critic_triage',
    'worker_escalate', 'judgment_ledger', 'coverage_check', 'git_snapshots',
    'source_corpus', 'rubric_contract', 'intent_brief',
    'consistency_check', 'facts_ledger', 'constraints_contract'],
}

export function applyPreset(p: 'off' | 'balanced' | 'thorough', prev: QualityProfile): QualityProfile {
  const on = new Set(PRESETS[p])
  const next = { ...prev, profile: p } as QualityProfile
  for (const f of BOOL_FLAGS) next[f] = on.has(f)
  return next
}

// Derive which preset (if any) the current flags match — else 'custom'.
export function derivePreset(q: QualityProfile): QualityProfile['profile'] {
  for (const name of ['off', 'balanced', 'thorough'] as const) {
    const on = new Set(PRESETS[name])
    if (BOOL_FLAGS.every((f) => q[f] === on.has(f))) return name
  }
  return 'custom'
}

export function setFlag(q: QualityProfile, flag: BoolFlag, value: boolean): QualityProfile {
  const next = { ...q, [flag]: value }
  next.profile = derivePreset(next)
  return next
}

// Only the flags the backend reads are sent; profile is derived server-side too.
export function toRequest(q: QualityProfile): Record<string, unknown> {
  const out: Record<string, unknown> = { profile: q.profile === 'custom' ? 'off' : q.profile }
  for (const f of BOOL_FLAGS) out[f] = q[f]
  out.deep_build_samples = q.deep_build_samples
  out.parallel_build_max_slices = q.parallel_build_max_slices
  out.claim_check_max = q.claim_check_max
  out.block_max_rounds = q.block_max_rounds
  out.honesty_guard = q.honesty_guard
  out.output_mode = q.output_mode
  out.token_budget = q.token_budget
  return out
}

export function fromResponse(d: Partial<QualityProfile> | null | undefined): QualityProfile {
  const base = defaultProfile()
  if (!d) return base
  const merged = { ...base, ...d } as QualityProfile
  merged.profile = derivePreset(merged)
  return merged
}

export type CostBadge = 'free' | 'saver' | 'cheap' | 'paid'
export type Scope = 'code' | 'research'

export interface Lever {
  flag: BoolFlag
  label: string
  blurb: string
  cost: CostBadge
  scope: Scope
  code: string  // the plan's code (C1, R1, …) — shown as a subtle tag
}

// Ordered for display; grouped by scope in the component.
export const LEVERS: Lever[] = [
  // Code
  { flag: 'test_gate', code: 'C1', scope: 'code', cost: 'free', label: 'Test gate',
    blurb: 'Run the repo’s tests after each build/fix; failures become blocking review findings. Zero model tokens.' },
  { flag: 'coverage_check', code: 'C5', scope: 'code', cost: 'cheap', label: 'Coverage check',
    blurb: 'Compare the approved plan against the build; unmet items go to the backlog. One extra model call.' },
  { flag: 'deep_build', code: 'C3', scope: 'code', cost: 'paid', label: 'Deep build',
    blurb: 'Best-of-N verified builds: try N times, keep the first that passes its tests. Uses more tokens — set a budget.' },
  { flag: 'parallel_build', code: 'B', scope: 'code', cost: 'paid', label: 'Team build',
    blurb: 'Decompose the approved plan into dependency-layered slices (foundations first) and build each with a focused implementer, sharing an interface ledger (the `facts` tool) so the slices agree on signatures. Layers run in order with a barrier between them; the review/fix loop then hardens the result. More agents = more tokens — set a budget.' },
  { flag: 'constraints_contract', code: 'A1', scope: 'code', cost: 'cheap', label: 'Acceptance contract',
    blurb: 'Derive the task’s acceptance criteria into checkable predicates (a command exits 0, a test/file exists, a placeholder is gone) once from the approved plan, persisted to a hand-editable .contract.json. Validated deterministically after each build/fix round; a failing rule becomes a ground-truth review finding.' },
  { flag: 'block_on_critical', code: 'A3', scope: 'code', cost: 'free', label: 'Block on regression',
    blurb: 'Guard the fix loop with the deterministic ground truth (failing tests + failing contract criticals): a fix round that makes it worse is reverted and the run stops at the prior state with an honest verdict, instead of a fixer degrading the repo. Zero extra model tokens.' },
  // Research
  { flag: 'intent_brief', code: 'R12', scope: 'research', cost: 'cheap', label: 'Intent brief',
    blurb: 'Before routing, clarify the task into one structured brief — shown for you to edit — and select the team against it. The original request always governs. Keeps weak workers on-scope (often pays for itself).' },
  { flag: 'research_map', code: 'R1', scope: 'research', cost: 'saver', label: 'Research map',
    blurb: 'Index the shared folder so agents (and the reporter) search prior findings instead of re-reading. Saves tokens on chains.' },
  { flag: 'acted_gate', code: 'R2', scope: 'research', cost: 'saver', label: 'Acted gate',
    blurb: 'Retry a worker once if it produced no text and wrote no file — recovers a wasted slot.' },
  { flag: 'critic_triage', code: 'R3', scope: 'research', cost: 'free', label: 'Critic triage',
    blurb: 'The deep-mode closer revises against a deduped, numbered checklist of objections instead of a blob. Deterministic.' },
  { flag: 'delta_rounds', code: 'R4', scope: 'research', cost: 'saver', label: 'Delta rounds',
    blurb: 'Continuation rounds inline only a short preview of the prior result (full text via file + map). Caps chain cost.' },
  { flag: 'worker_escalate', code: 'R5', scope: 'research', cost: 'cheap', label: 'Worker escalate',
    blurb: 'A worker that flags ESCALATE gets one focused retry instead of the merge absorbing a bare flag.' },
  { flag: 'git_snapshots', code: 'R6', scope: 'research', cost: 'free', label: 'Git snapshots',
    blurb: 'Commit each round’s state of the research folder to git — diffs, rollback, provenance. Zero model tokens.' },
  { flag: 'rubric_contract', code: 'R9', scope: 'research', cost: 'cheap', label: 'Rubric contract',
    blurb: 'Derive the completeness checklist once (from the standard the task names) and hold every specialist + the reporter to it. Fixes “which fields count”.' },
  { flag: 'source_corpus', code: 'R10', scope: 'research', cost: 'saver', label: 'Source corpus',
    blurb: 'web_fetch saves each full page to the shared folder (indexed) and returns a head + pointer — depth without context blow-up; every stage can re-read sources.' },
  { flag: 'judgment_ledger', code: 'R11', scope: 'research', cost: 'free', label: 'Judgment ledger',
    blurb: 'Each specialist enumerates + resolves its hardest judgment calls explicitly (so weak models make the boundary calls instead of hedging), and must not assert an unconfirmable specific — a named office-holder, an origin, an exact figure — as fact. Prompt-only.' },
  { flag: 'claim_check', code: 'R8', scope: 'research', cost: 'paid', label: 'Claim check',
    blurb: 'A web-tool fact-checker verifies the deliverable’s citations, dates, versions, figures and named entities against real sources — corrects the wrong ones, hedges any specific it asserted but couldn’t confirm (the fabricated-name trap), and saves a standalone fact-check report. The ground-truth back-edge.' },
  { flag: 'consistency_check', code: 'Q2', scope: 'research', cost: 'cheap', label: 'Consistency check',
    blurb: 'Extract the deliverable’s figures once, then verify in plain code that the same quantity carries the same value everywhere and every stated sum/percentage actually computes. One targeted correction, kept only if the deterministic re-check improves. Saves a consistency report.' },
  { flag: 'facts_ledger', code: 'Q4', scope: 'research', cost: 'free', label: 'Facts ledger',
    blurb: 'A shared ledger of canonical values (the `facts` tool): workers record load-bearing numbers/dates/IDs with status + provenance and read teammates’ values instead of restating them. The reporter must match the ledger; a conflicting write is surfaced, never silently overwritten. Also powers the consistency + contract checks.' },
  { flag: 'constraints_contract', code: 'Q5', scope: 'research', cost: 'cheap', label: 'Constraints contract',
    blurb: 'Derive the task’s hard rules once (ranges, caps, required relationships) into a persisted, hand-editable .contract.json the whole chain reuses — then validate the deliverable against them, deterministically where the facts ledger has the values.' },
  { flag: 'block_on_critical', code: 'Q6', scope: 'research', cost: 'paid', label: 'Block on critical',
    blurb: 'The enforcement gate: critical findings from the deterministic checks block finalization — one triaged checklist, revise-and-recheck up to N rounds. A run that can’t clear keeps all its work and finishes with verdict “critical findings remain”. Pair with a token budget.' },
]

// Non-zinc accent colors don't auto-invert with the theme, so give each badge a
// light-theme base + an explicit dark: override (dark text on light bg in light
// mode, light text on dark bg in dark mode).
export const COST_STYLE: Record<CostBadge, string> = {
  free: 'border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-700/50 dark:bg-emerald-900/30 dark:text-emerald-300',
  saver: 'border-sky-300 bg-sky-50 text-sky-700 dark:border-sky-700/50 dark:bg-sky-900/30 dark:text-sky-300',
  cheap: 'border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-700/40 dark:bg-amber-900/20 dark:text-amber-300',
  paid: 'border-rose-300 bg-rose-50 text-rose-700 dark:border-rose-700/50 dark:bg-rose-900/30 dark:text-rose-300',
}
