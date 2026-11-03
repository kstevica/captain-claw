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
