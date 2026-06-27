// REST client for the Dubina (Frontier Horizon) endpoints.
// Mirrors captain_claw/flight_deck/dubina_routes.py; auth pattern from flowsApi.ts.
import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const BASE = '/fd/dubina'

export type Track = 'coder' | 'reason' | 'intent'

export interface Tier {
  id: string
  provider: string
  model: string
  description: string
  reasoning_level: string
}

export interface TiersResponse {
  tiers: Tier[]
  default_ladders: { coder: string[]; reason: string[] }
}

export interface RunStep {
  seq: number
  step_id: string
  tier: string
  rung: number
  kind: string
  samples: number
  passed: number
  confidence: number
}

export interface DubinaRun {
  id: string
  user_id: string
  task: string
  base_tier: string
  max_tier: string
  compute_budget: number
  status: string // running | passed | failed | budget | error
  passed: boolean | null
  stopped_reason: string
  cost_spent: number
  config: Record<string, unknown>
  result: Record<string, unknown>
  error: string
  created_at: string
  updated_at: string
  steps?: RunStep[]
}

export interface CoderRequest {
  task: string
  workspace: string
  test_command?: string
  solution_path?: string
  test_path?: string
  spec?: string
  base_tier: string
  max_tier: string
  tiers?: string[] | null
  compute_budget?: number
  max_step_samples?: number
  max_fix_attempts?: number
}

export interface ReasonRequest {
  task: string
  base_tier: string
  max_tier: string
  tiers?: string[] | null
  compute_budget?: number
  max_step_samples?: number
  max_fix_attempts?: number
  stakes?: string
  critic_modes?: string[]
  agreement_threshold?: number
  critic_cost?: number
}

export interface IntentRequest extends ReasonRequest {
  target: string // "agent:<id>" | "archetype:<id>"
}

function _authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = {}
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = { ..._authHeaders(), ...(init?.headers as Record<string, string> | undefined) }
  let res = await fetch(`${BASE}${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) {
      const h2 = { ..._authHeaders(), ...(init?.headers as Record<string, string> | undefined) }
      res = await fetch(`${BASE}${path}`, { ...init, headers: h2, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || body.detail || `${res.status} ${res.statusText}`)
  }
  const text = await res.text()
  return (text ? JSON.parse(text) : {}) as T
}

function jsonInit(method: string, body?: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  }
}

// Reach /fd endpoints outside the /fd/dubina prefix (archetypes, fleet).
async function fdRoot<T>(path: string): Promise<T> {
  const headers = _authHeaders()
  let res = await fetch(`/fd${path}`, { headers, credentials: 'include' })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) res = await fetch(`/fd${path}`, { headers: _authHeaders(), credentials: 'include' })
  }
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  const text = await res.text()
  return (text ? JSON.parse(text) : {}) as T
}

export interface TargetOption { value: string; label: string }

// Combined run-target options for the Intent track: archetypes + live agents.
export async function getTargets(): Promise<TargetOption[]> {
  const out: TargetOption[] = []
  try {
    const reg = await fdRoot<{ archetypes: { id: string; role?: string }[] }>('/archetypes')
    for (const a of reg.archetypes || [])
      out.push({ value: `archetype:${a.id}`, label: `archetype · ${a.role || a.id}` })
  } catch { /* ignore */ }
  try {
    const fleet = await fdRoot<{ slug: string; name: string; status: string }[]>('/fleet')
    for (const f of fleet || [])
      if (f.status === 'running')
        out.push({ value: `agent:${f.slug}`, label: `agent · ${f.name || f.slug}` })
  } catch { /* ignore */ }
  return out
}

export const getTiers = () => fdFetch<TiersResponse>('/tiers')

export const startCoder = (req: CoderRequest) =>
  fdFetch<{ run_id: string; track: Track; status: string }>('/coder', jsonInit('POST', req))

export const startReason = (req: ReasonRequest) =>
  fdFetch<{ run_id: string; track: Track; status: string }>('/reason', jsonInit('POST', req))

// Intent runs inline server-side and returns the finished run.
export const startIntent = (req: IntentRequest) =>
  fdFetch<DubinaRun>('/intent', jsonInit('POST', req))

export const getRun = (track: Track, id: string) =>
  fdFetch<DubinaRun>(`/runs/${track}/${id}`)

export const listRuns = (track: Track, limit = 50) =>
  fdFetch<{ runs: DubinaRun[] }>(`/runs/${track}?limit=${limit}`)
