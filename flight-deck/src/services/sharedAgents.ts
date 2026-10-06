// REST client for agents other deck users shared with you (/fd/shared-agents).
//
// A row names the agent by its `agent_ref` and its owner — never its host,
// port or access token: chats go through Flight Deck's member route.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import type { SharedCaps } from '../utils/sharedAgent'

export type { SharedCaps }

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
  /** What a member chat on this agent can use (FD: all true for process agents,
   *  all false for docker). Absent (an older Flight Deck): chat-only — read it
   *  through `sharedCaps(row)`. */
  capabilities?: SharedCaps
  /** The member (me) lets this agent use my Google during my chats. */
  google_enabled: boolean
  /** I have connected my own Google account in Flight Deck. */
  google_connected: boolean
}

export interface SharedAgentsResponse {
  /** Off: the deck doesn't share agents — every sharing affordance hides. */
  enabled: boolean
  host_warning: string
  agents: SharedAgent[]
  /** My own agents that have members: agent_ref → how many, and how many of
   *  them turned their Google on. Absent → no owner badge. */
  mine?: Record<string, { members: number; google: number }>
  /** Context packs: users can share their own profile, folders and deep
   *  memory with everyone who uses an agent. Absent (an older Flight Deck) or
   *  false → every "Shared context" affordance hides. */
  context_packs?: boolean
  /** This deck serves members' files and data panels (a process agent's
   *  saved/ folder and datastore); absent on an older Flight Deck. */
  member_workspace?: boolean
}

// ── Endpoints ──

export const getSharedAgents = (): Promise<SharedAgentsResponse> =>
  fdFetch<SharedAgentsResponse>('/shared-agents')

/** Turn my Google on or off for one agent shared with me (my choice, not the owner's). */
export const setSharedAgentGoogle = (agentRef: string, enabled: boolean) =>
  fdFetch<{ agent_ref: string; google_enabled: boolean }>('/shared-agents/google', {
    method: 'PUT', body: JSON.stringify({ agent_ref: agentRef, enabled }),
  })
