// REST client for "Your own AI keys" (/fd/byok): a user's own provider keys
// and the switch that runs their agents on them. While it is on, the deck's
// daily limits don't apply to them. Keys go in, never come back — only a hint
// (captain_claw/flight_deck/byok.py).

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
  const build = (): RequestInit => ({ headers: _authHeaders(), credentials: 'include', ...init })
  let res = await fetch(`${FD_BASE}${path}`, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (!(await refreshAccessToken())) {
      useAuthStore.getState().clearAuth()
      throw new Error('Session expired')
    }
    res = await fetch(`${FD_BASE}${path}`, build())
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    let msg = body.detail
    if (Array.isArray(msg)) {
      msg = msg.map((e: { loc?: string[]; msg?: string }) => `${(e.loc || []).join('.')}: ${e.msg}`).join('; ')
    }
    throw new Error(typeof msg === 'string' && msg ? msg : `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Types ──

export interface ByokProvider {
  id: string
  label: string
  connected: boolean
  hint: string | null
  /** true checked, false couldn't be asked, null no key */
  verified: boolean | null
  checked_at: string | null
  models: string[]
}

export interface ByokTierPick { provider: string; model: string }

export interface ByokTier {
  id: string
  label: string
  use: string
  /** What the deck runs this tier on. */
  team: ByokTierPick
  /** What it runs on with the user's keys (null: no key can cover it). */
  mine: ByokTierPick | null
  /** The user picked this model themselves. */
  chosen: boolean
  /** A local model: stays as it is, costs nothing. */
  local: boolean
}

export interface ByokStatus {
  offered: boolean
  allowed: boolean
  encryption: boolean
  enabled: boolean
  providers: ByokProvider[]
  tiers: ByokTier[]
  missing: string[]
  agents: { total: number; running: number; on_own_keys: number }
  switched_at: string | null
}

export interface ByokChange extends ByokStatus {
  warning?: string
  moved?: { switched?: number; restarted?: number; error?: string }
}

export const getByok = () => fdFetch<ByokStatus>('/byok')

export const saveByokKey = (provider: string, apiKey: string) =>
  fdFetch<ByokChange>(`/byok/key/${encodeURIComponent(provider)}`, {
    method: 'PUT', body: JSON.stringify({ api_key: apiKey }),
  })

export const removeByokKey = (provider: string) =>
  fdFetch<ByokChange>(`/byok/key/${encodeURIComponent(provider)}`, { method: 'DELETE' })

/** An empty pick puts the tier back on its default model. */
export const saveByokModels = (tiers: Record<string, ByokTierPick>) =>
  fdFetch<ByokChange>('/byok/models', { method: 'PUT', body: JSON.stringify({ tiers }) })

export const setByokEnabled = (enabled: boolean) =>
  fdFetch<ByokChange>('/byok/enabled', { method: 'PUT', body: JSON.stringify({ enabled }) })

// ── Admin ──

export interface AdminByok { offered: boolean; allowed: boolean; users_on: string[] }

export const getAdminByok = () => fdFetch<AdminByok>('/admin/byok')

export const setAdminByokAllowed = (allowed: boolean) =>
  fdFetch<{ ok: boolean; allowed: boolean; switched_back: number }>('/admin/byok', {
    method: 'PUT', body: JSON.stringify({ allowed }),
  })
