// REST client for the Flight Deck "System / DevOps" process monitor.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const FD_BASE = '/fd'

function _authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const _state = useAuthStore.getState()
  if (_state.authEnabled === true && !_state.token) {
    const refreshed = await refreshAccessToken()
    if (!refreshed) throw new Error('Not authenticated')
  }
  const res = await fetch(`${FD_BASE}${path}`, {
    headers: _authHeaders(),
    credentials: 'include',
    ...init,
  })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    const refreshed = await refreshAccessToken()
    if (refreshed) {
      const retry = await fetch(`${FD_BASE}${path}`, {
        headers: _authHeaders(),
        credentials: 'include',
        ...init,
      })
      if (retry.ok) return retry.json()
    }
    useAuthStore.getState().clearAuth()
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

// ── Types ──

/** A node in the process forest. Roots are Flight Deck-spawned agents / apps;
 *  their descendants (kind === 'child') are whatever those roots have run. */
export interface SystemProcessNode {
  pid: number
  ppid: number
  kind: 'agent' | 'hosted-app' | 'code-app' | 'flight-deck' | 'child'
  label: string
  slug: string | null
  owner: string | null
  owner_email: string | null
  command: string
  name: string
  cpu: number
  mem: number
  rss_mb: number
  elapsed: string
  elapsed_s: number
  detail: string
  is_root: boolean
  descendant_count: number
  agg_cpu: number
  agg_mem_mb: number
  children: SystemProcessNode[]
}

export interface HostVitals {
  cpu_count: number | null
  load_avg: [number, number, number] | null
  mem_total_mb: number | null
  mem_used_mb: number | null
  mem_percent: number | null
  disk_free_gb: number | null
  disk_total_gb: number | null
}

export interface SystemSummary {
  roots: number
  agents: number
  hosted: number
  children: number
  total_cpu: number
  total_mem_mb: number
  stopped: number
}

export interface SystemUserUsage {
  owner: string | null
  owner_email: string
  roots: number
  procs: number
  cpu: number
  mem_mb: number
}

export interface SystemProcessResponse {
  is_admin: boolean
  available: boolean
  host: HostVitals
  summary: SystemSummary
  by_user: SystemUserUsage[]
  trees: SystemProcessNode[]
}

export interface StopResult {
  ok: boolean
  pid: number
  killed: number[]
  message: string
}

// ── Endpoints ──

export const getSystemProcesses = () =>
  fdFetch<SystemProcessResponse>('/system/processes')

export const stopSystemProcess = (pid: number, tree = false) =>
  fdFetch<StopResult>(`/system/processes/${pid}/stop?tree=${tree}`, { method: 'POST' })
