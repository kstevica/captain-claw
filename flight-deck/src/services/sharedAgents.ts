// REST client for agents other deck users shared with you (/fd/shared-agents).
//
// A row names the agent by its `agent_ref` and its owner — never its host,
// port or access token: chats go through Flight Deck's member route.

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
  let res = await fetch(`${FD_BASE}${path}`, {
    headers: _authHeaders(),
    credentials: 'include',
    ...init,
  })
  // Only a failed refresh ends the session (see docker.ts fdFetch).
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (!(await refreshAccessToken())) {
      useAuthStore.getState().clearAuth()
      throw new Error('Session expired')
    }
    res = await fetch(`${FD_BASE}${path}`, {
      headers: _authHeaders(), credentials: 'include', ...init,
    })
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

// ── Types ──

export interface SharedAgent {
  agent_ref: string
  runtime: 'process' | 'docker'
  slug: string
  name: string
  description: string
  status: 'running' | 'stopped'
  owner_id: string
  owner_name: string
  owner_email: string
  shared_at: string
}

export interface SharedAgentsResponse {
  /** Off: the deck doesn't share agents — every sharing affordance hides. */
  enabled: boolean
  host_warning: string
  agents: SharedAgent[]
}

// ── Endpoints ──

export const getSharedAgents = (): Promise<SharedAgentsResponse> =>
  fdFetch<SharedAgentsResponse>('/shared-agents')
