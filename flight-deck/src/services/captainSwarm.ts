/** Captain Swarm (docs/captain-swarm-plan.md), Phase 1: practice sets frozen from the owner's own
 *  history, and measurements of an agent on them (/fd/captain-swarm). An agent answers on its private
 *  exam lane: nothing is sent, nothing is kept. */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

export interface SetPart {
  cases: number
  replied: number
  base_rate: number
  first: string
  last: string
}

export interface PracticeSet {
  id: string
  kind: string
  title?: string
  state: 'building' | 'ready' | 'error'
  created: number
  error?: string
  label?: string
  source?: { gmail?: string; params?: Record<string, number>; window?: { from: string; to: string; holdout_from: string } }
  open?: SetPart
  holdout?: SetPart
  skipped?: Record<string, number>
  progress?: { stage?: string; done?: number; total?: number }
}

export interface Agreement {
  n: number
  tp: number
  fp: number
  fn: number
  tn: number
  unanswered: number
  precision: number
  recall: number
  f1: number
  accuracy: number
  base_rate: number
}

export interface MeasureResult {
  metric: string
  value: number
  f1: { n: number; mean: number; stdev: number; low: number; high: number }
  runs: Agreement[]
  pooled: Agreement
  guardrails: Record<string, number>
  asked: number
  stopped: boolean
}

export interface Measurement {
  id: string
  set: string
  agent: string
  agent_name: string
  split: 'open' | 'holdout'
  cases: number
  repeat: number
  state: 'running' | 'done' | 'stopped' | 'error'
  started: number
  finished?: number
  error?: string
  result?: MeasureResult
  progress?: { done?: number; total?: number }
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    return h
  }
  const go = () => fetch(path, { ...init, headers: headers(), credentials: 'include', cache: 'no-store' })
  let res = await go()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await go()
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status}`)
  }
  return res.json()
}

const BASE = '/fd/captain-swarm'
const enc = encodeURIComponent

export const captainSwarmApi = {
  sets: () => call<{ sets: PracticeSet[] }>(`${BASE}/practice`),
  buildTriage: (body: { window_days: number; reply_days: number; holdout_days: number; max_cases: number }) =>
    call<PracticeSet>(`${BASE}/practice/mail-triage`, { method: 'POST', body: JSON.stringify(body) }),
  deleteSet: (id: string) => call<{ deleted: boolean }>(`${BASE}/practice/${enc(id)}`, { method: 'DELETE' }),
  measure: (id: string, body: { agent: string; split: 'open' | 'holdout'; cases: number; repeat: number }) =>
    call<Measurement>(`${BASE}/practice/${enc(id)}/measure`, { method: 'POST', body: JSON.stringify(body) }),
  measurements: (id: string) => call<{ measurements: Measurement[] }>(`${BASE}/practice/${enc(id)}/measurements`),
  stop: (setId: string, mid: string) =>
    call<{ stopping: boolean }>(`${BASE}/measurements/${enc(setId)}/${enc(mid)}/stop`, { method: 'POST' }),
}

// ── runs (Phase 2: the loop, on code) ────────────────────────────────

export type RunStatus =
  | 'awaiting_go' | 'baseline' | 'running' | 'done' | 'stalled' | 'stopped' | 'failed' | 'cancelled' | 'declined'

export interface Stats { n: number; mean: number; stdev: number; low: number; high: number }

export interface Reading {
  value: number | null
  stats?: Stats
  guard?: Record<string, number | null>
  per_case?: Record<string, number>
  error?: string
  seconds?: number
}

export interface Point { sha: string; dev: Reading; holdout: Reading; exp: string }

export interface RunSummary {
  id: string
  kind: 'code' | 'practice'
  title: string
  status: RunStatus
  stopped_reason: string
  started_by: string
  round: number
  experiments: number
  spent_usd: number
  spent_tokens: number
  baseline_dev: number | null
  champion_dev: number | null
  baseline_holdout: number | null
  champion_holdout: number | null
  improved: boolean
  created_at: number
  started_at: number
  finished_at: number
  budget?: { usd: number; hours: number; max_experiments: number }
}

export interface Experiment {
  id: string
  round: number
  n: number
  hypothesis: string
  strategy_tag: string
  why: string
  lane: string
  status: 'proposed' | 'running' | 'screening' | 'measuring' | 'kept' | 'discarded' | 'disqualified' | 'failed'
  verdict_reason: string
  lesson: string
  base_sha: string
  head_sha: string
  child_id: string
  dev: Reading
  holdout: { stats?: Stats; value?: number }
  diff_stat: string
  changed_files: string[]
  cost_usd: number
  tokens: number
  change?: { add?: string[]; remove?: (number | string)[] }
}

export interface RunEvent { id: number; ts: number; stage: string; message: string }

export interface RunDetail extends RunSummary {
  rules?: { baseline: string[]; champion: string[] }
  objective: {
    kind: 'code' | 'practice'
    title: string
    goal: string
    subject: { repo_path?: string; editable?: string[]; protected?: string[]; agent?: string; agent_name?: string; set_id?: string }
    metric: { read: string; direction: 'max' | 'min'; run?: string; cases?: number; repeat?: number }
    lane: { harness: string; model: string; per_round: number }
    stop: { target: number | null; stall_rounds: number }
    budget: { usd: number; hours: number; max_experiments: number }
  }
  baseline: Partial<Point>
  champion: Partial<Point>
  report?: Record<string, unknown> | null
  ledger: Experiment[]
  events: RunEvent[]
}

export interface MravTemplate {
  title: string
  about: string
  defaults: Record<string, string | number>
  local_models: string[]
}

export interface TriageTemplate {
  title: string
  about: string
  sets: { id: string; title: string; open: number; holdout: number }[]
  defaults: Record<string, string | number>
}

export const captainSwarmRuns = {
  templates: () =>
    call<{ templates: { mrav: MravTemplate; triage: TriageTemplate }; can_start: boolean }>(`${BASE}/templates`),
  start: (template: 'mrav' | 'triage', options: Record<string, string | number>) =>
    call<RunSummary>(`${BASE}/runs`, { method: 'POST', body: JSON.stringify({ template, options }) }),
  list: () => call<{ runs: RunSummary[] }>(`${BASE}/runs`),
  get: (id: string) => call<RunDetail>(`${BASE}/runs/${enc(id)}`),
  cancel: (id: string) => call<RunSummary>(`${BASE}/runs/${enc(id)}/cancel`, { method: 'POST' }),
  go: (id: string) => call<RunSummary>(`${BASE}/runs/${enc(id)}/go`, { method: 'POST' }),
  decline: (id: string) => call<RunSummary>(`${BASE}/runs/${enc(id)}/decline`, { method: 'POST' }),
  pr: (id: string) => call<{ delivery: { url?: string } }>(`${BASE}/runs/${enc(id)}/pr`, { method: 'POST' }),
  teach: (id: string) => call<{ added: number; rules: string[] }>(`${BASE}/runs/${enc(id)}/teach`, { method: 'POST' }),
}
