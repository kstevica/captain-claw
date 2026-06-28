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

export interface RunFile {
  name: string
  mime: string
  size: number
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

export interface ProgressEvent {
  i: number
  ts?: number // epoch seconds (server clock)
  stage: string // start | attempt | action | narration | llm | done | error
  message: string
  tier?: string
  rung?: number
  kind?: string
  samples?: number
  passed?: boolean
  confidence?: number
  agent?: string // run-target agent/archetype this step belongs to
  tool?: string // tool name on action/narration events
  detail?: string // tool-arg summary on action events
  prompt_tokens?: number // on llm events — turn's running token counts
  completion_tokens?: number
  total_tokens?: number
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
  target?: string // "" → Library tier model; else "agent:<id>" | "archetype:<id>"
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
  target?: string // "" → Library tier model; else "agent:<id>" | "archetype:<id>"
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

export const startIntent = (req: IntentRequest) =>
  fdFetch<{ run_id: string; track: Track; status: string }>('/intent', jsonInit('POST', req))

export const getRun = (track: Track, id: string) =>
  fdFetch<DubinaRun>(`/runs/${track}/${id}`)

export const getProgress = (track: Track, id: string) =>
  fdFetch<{ events: ProgressEvent[]; active: boolean }>(`/runs/${track}/${id}/progress`)

export const stopRun = (track: Track, id: string) =>
  fdFetch<{ ok: boolean; status: string }>(`/runs/${track}/${id}/stop`, jsonInit('POST'))

// Raw fetch (with auth + refresh) for a generated file — used for view + download.
async function fileFetch(track: Track, id: string, name: string): Promise<Response> {
  const path = `${BASE}/runs/${track}/${encodeURIComponent(id)}/files/${encodeURIComponent(name)}`
  let res = await fetch(path, { headers: _authHeaders(), credentials: 'include' })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) res = await fetch(path, { headers: _authHeaders(), credentials: 'include' })
  }
  return res
}

export async function fetchRunFileText(track: Track, id: string, name: string): Promise<string> {
  const res = await fileFetch(track, id, name)
  return res.ok ? res.text() : ''
}

export async function downloadRunFile(track: Track, id: string, name: string): Promise<void> {
  const res = await fileFetch(track, id, name)
  if (!res.ok) return
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

export const cleanupAgents = () =>
  fdFetch<{ stopped: string[]; count: number }>('/agents/cleanup', jsonInit('POST'))

export const listRuns = (track: Track, limit = 50) =>
  fdFetch<{ runs: DubinaRun[] }>(`/runs/${track}?limit=${limit}`)
