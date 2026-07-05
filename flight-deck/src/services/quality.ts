// Quality/cost profile — mirrors captain_claw/flight_deck/quality_profile.py.
// Every lever is opt-in; an all-off profile == the systems' current behaviour.

export type QualityProfile = {
  profile: 'off' | 'balanced' | 'thorough' | 'custom'
  // Code-side
  test_gate: boolean
  deep_build: boolean
  coverage_check: boolean
  deep_build_samples: number
  // Research-side (Basna / Vatra)
  acted_gate: boolean
  research_map: boolean
  delta_rounds: boolean
  critic_triage: boolean
  worker_escalate: boolean
  git_snapshots: boolean
  // Shared
  token_budget: number
}

export const BOOL_FLAGS = [
  'test_gate', 'deep_build', 'coverage_check', 'acted_gate', 'research_map',
  'delta_rounds', 'critic_triage', 'worker_escalate', 'git_snapshots',
] as const
export type BoolFlag = (typeof BOOL_FLAGS)[number]

export function defaultProfile(): QualityProfile {
  return {
    profile: 'off',
    test_gate: false, deep_build: false, coverage_check: false, deep_build_samples: 2,
    acted_gate: false, research_map: false, delta_rounds: false, critic_triage: false,
    worker_escalate: false, git_snapshots: false,
    token_budget: 0,
  }
}

// Preset → flags ON. Mirrors the backend _PRESETS exactly. deep_build is never in
// a preset (the one genuinely expensive lever — explicit opt-in only).
const PRESETS: Record<'off' | 'balanced' | 'thorough', BoolFlag[]> = {
  off: [],
  balanced: ['acted_gate', 'test_gate', 'research_map', 'delta_rounds', 'critic_triage', 'worker_escalate'],
  thorough: ['acted_gate', 'test_gate', 'research_map', 'delta_rounds', 'critic_triage',
    'worker_escalate', 'coverage_check', 'git_snapshots'],
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
  // Research
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
]

export const COST_STYLE: Record<CostBadge, string> = {
  free: 'border-emerald-700/50 bg-emerald-900/30 text-emerald-300',
  saver: 'border-sky-700/50 bg-sky-900/30 text-sky-300',
  cheap: 'border-amber-700/40 bg-amber-900/20 text-amber-300',
  paid: 'border-rose-700/50 bg-rose-900/30 text-rose-300',
}
