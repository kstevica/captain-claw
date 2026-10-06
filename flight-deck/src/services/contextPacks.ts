// REST client for context packs (/fd/context-packs): a user shares one of
// their OWN resources — their profile, a VFS folder (read-only) or their deep
// memory — with everyone who uses an agent they own or are a member of.
//
// No type here has a host path: Flight Deck never sends a folder's location to
// a browser, only the project name and the `vfs:@<alias>` the agent uses.

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

export type PackKind = 'profile' | 'vfs' | 'deep_memory'

export interface PackRow {
  id: string
  kind: PackKind
  /** The publisher's id — '' unless the row is mine or I own the agent. */
  owner_id: string
  /** The publisher's display name ('' when unknown). */
  owner_name: string
  /** Whether the publisher is the agent's owner or one of its members — the
   *  UI's role badge comes from this, never from the name. */
  role: 'owner' | 'member'
  /** FD's collision tag (e.g. '3f9a'), sent when two publishers on the agent
   *  share a name; ''/absent otherwise. */
  owner_tag?: string
  mine: boolean
  /** vfs: the publisher's project name; else ''. */
  project: string
  /** vfs: the agent reaches the folder as `vfs:@<alias>`; else ''. */
  alias: string
  /** deep_memory: only entries carrying one of these tags ([] = all of it). */
  tags: string[]
  created_at: string
  /** I published it, or I own the agent. */
  can_remove: boolean
  /** On `mine` rows: the pack is in use (see INACTIVE_HINT when it isn't). */
  active?: boolean
}

/** A tag in my deep memory, with how many entries carry it. */
export interface TagFacet { tag: string; count: number }

export interface AgentPacks {
  agent_ref: string
  agent_name: string
  runtime: 'process' | 'docker'
  /** My role on this agent. */
  role: 'owner' | 'member'
  /** The agent owner's display name. */
  owner_name: string
  /** What can be shared on this agent (a Docker agent: profile only). */
  kinds: PackKind[]
  /** Everyone's active packs on this agent. */
  packs: PackRow[]
  /** My own packs on this agent (with `active`). */
  mine: PackRow[]
  /** My folders that can be shared. */
  eligible_projects: string[]
  /** The running agent understands shared context; null = not running / not reachable. */
  agent_supports_packs: boolean | null
  /** The agent is running (so a null `agent_supports_packs` means FD's version
   *  check failed, not that it's stopped). Absent on an older FD. */
  agent_running?: boolean
  /** The fixed start of my folder names on this agent (`<prefix>-<name>`). */
  alias_prefix: string
  /** Tags in MY deep memory, most used first. */
  deep_memory_tags: TagFacet[]
  limits: { max_packs: number; max_vfs_per_owner: number; max_tags: number }
}

export interface MyPackRow extends PackRow {
  agent_ref: string
  agent_name: string
  active: boolean
}

// ── Endpoints ──

export const getAgentPacks = (agentRef: string) =>
  fdFetch<AgentPacks>(`/context-packs?agent_ref=${encodeURIComponent(agentRef)}`)

export const createPack = (body: { agent_ref: string; kind: PackKind; project?: string; alias?: string; tags?: string[] }) =>
  fdFetch<{ pack: PackRow }>('/context-packs', { method: 'POST', body: JSON.stringify(body) })

export const deletePack = (id: string) =>
  fdFetch<{ ok: boolean }>(`/context-packs/${encodeURIComponent(id)}`, { method: 'DELETE' })

export const getMyPacks = () => fdFetch<{ packs: MyPackRow[] }>('/context-packs/mine')
