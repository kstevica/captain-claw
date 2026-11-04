// REST client for public file links (/fd/agent-file-share, /fd/public-shares).
// A link publishes a COPY of one agent file at /s/<token> for anyone holding it.

import { useEffect, useState } from 'react'
import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import type { AgentEndpoint } from './fileTransfer'

const FD_BASE = '/fd'

export interface PublicLink {
  token: string
  /** `/s/<token>` — join with `linkUrl()` for the full address. */
  path: string
  filename: string
  kind: string
  size: number
  agent_ref: string
  agent_name: string
  source_path: string
  created_at: string
  /** When the copy was last taken from the agent. */
  updated_at: string
  expires_at: string | null
  expired: boolean
  view_count: number
  last_viewed_at: string | null
}

export interface PublicLinkConfig {
  enabled: boolean
  max_mb: number
  /** FD_PUBLIC_URL when the deck sets one; else links use this page's origin. */
  public_base: string
  /** The expiries the deck allows, in days; 0 = never (only on a deck with no cap). */
  expiry_days: number[]
  /** How long a new link lasts unless the owner picks otherwise. */
  default_days?: number
}

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

const agentPath = (agent: AgentEndpoint) =>
  `/agent-file-share/${encodeURIComponent(agent.host)}/${agent.port}`

export function linkUrl(link: PublicLink, config?: PublicLinkConfig | null): string {
  return `${config?.public_base || window.location.origin}${link.path}`
}

export function getFileLink(agent: AgentEndpoint, path: string) {
  const q = new URLSearchParams({ path, agent_ref: agent.id })
  return fdFetch<{ share: PublicLink | null } & PublicLinkConfig>(`${agentPath(agent)}?${q}`)
}

/** Publish the file — or, when this file already has a link, take a fresh
 *  copy (the address stays the same). `expiresDays` null keeps the link's
 *  current expiry; 0 means never. */
export function saveFileLink(agent: AgentEndpoint, path: string, expiresDays: number | null) {
  return fdFetch<{ share: PublicLink } & PublicLinkConfig>(agentPath(agent), {
    method: 'POST',
    body: JSON.stringify({
      path,
      agent_auth: agent.auth || '',
      agent_ref: agent.id,
      agent_name: agent.name,
      expires_days: expiresDays,
    }),
  })
}

export function listMyLinks() {
  return fdFetch<{ shares: PublicLink[] } & PublicLinkConfig>('/public-shares')
}

export function revokeLink(token: string) {
  return fdFetch<{ ok: boolean }>(`/public-shares/${encodeURIComponent(token)}`, { method: 'DELETE' })
}

/** Report this page's origin (sent by the browser as the POST's Origin) so links
 *  an agent makes use the address people reach the deck at. Best-effort. */
export function reportPublicOrigin(): void {
  void fdFetch('/public-shares/origin', { method: 'POST' }).catch(() => {})
}

// Whether this deck allows public links at all (FD_PUBLIC_FILE_LINKS) — asked
// once per page load; the file lists hide the button when it's off.
let _enabled: Promise<boolean> | null = null
function fetchEnabled(): Promise<boolean> {
  if (!_enabled) {
    _enabled = fdFetch<PublicLinkConfig>('/public-shares/config')
      .then((c) => c.enabled)
      .catch(() => { _enabled = null; return false })
  }
  return _enabled
}

export function usePublicLinksEnabled(): boolean {
  const [on, setOn] = useState(false)
  useEffect(() => {
    let live = true
    void fetchEnabled().then((v) => { if (live) setOn(v) })
    return () => { live = false }
  }, [])
  return on
}
