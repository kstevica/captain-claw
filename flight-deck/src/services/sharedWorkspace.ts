// REST client for a shared agent's saved files and datastore, as a MEMBER sees
// them (/fd/shared-agents/files*, /fd/shared-agents/datastore*).
//
// The agent is named only by its `agent_ref`: nothing here takes or sends a
// host, port or the agent's access token, and a file is named by its id
// relative to the agent's saved/ folder — never a host path. Flight Deck
// re-checks membership on every call, and the agent decides what a member may
// change (the `can_delete` flag is display only).

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import type { Creator } from '../utils/sharedWorkspace'

const FD_BASE = '/fd'

function _authHeaders(json: boolean): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = {}
  // Only a JSON body says so — a FormData body gets its multipart boundary
  // from the browser.
  if (json) headers['Content-Type'] = 'application/json'
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const json = typeof init?.body === 'string'
  const _state = useAuthStore.getState()
  if (_state.authEnabled === true && !_state.token) {
    const refreshed = await refreshAccessToken()
    if (!refreshed) throw new Error('Not authenticated')
  }
  let res = await fetch(`${FD_BASE}${path}`, {
    headers: _authHeaders(json),
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
      headers: _authHeaders(json), credentials: 'include', ...init,
    })
  }
  if (!res.ok) {
    // Flight Deck's `detail` is shown verbatim (e.g. "…needs a restart").
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error((typeof body?.detail === 'string' && body.detail) || `${res.status}`)
  }
  return res.json()
}

// ── Types ──

/** A file in the agent's saved/ folder. `id` is its path relative to saved/,
 *  `path` is "saved/" + id. */
export interface SharedFile {
  id: string; path: string; filename: string; extension: string; size: number; modified: number
  mime_type: string; is_text: boolean; created_by: Creator; can_delete: boolean
}

export interface SharedFilesResponse {
  files: SharedFile[]; truncated: boolean; upload: { max_bytes: number; extensions: string[] }
}

export interface SharedTable {
  name: string; columns: { name: string; type: string; position: number }[]; row_count: number
  created_at: string; updated_at: string; created_by: Creator
}

// ── Endpoints ──

export const listSharedFiles = (agentRef: string) =>
  fdFetch<SharedFilesResponse>(`/shared-agents/files?ref=${encodeURIComponent(agentRef)}`)

/** A URL for <img>/<a>/window.open, which can't send headers: the caller's
 *  FD token rides along as `fd_token` (as the owner's view URLs do). */
export function sharedFileUrl(agentRef: string, id: string, mode: 'view' | 'download'): string {
  const { token, authEnabled } = useAuthStore.getState()
  const auth = authEnabled && token ? `&fd_token=${encodeURIComponent(token)}` : ''
  return `${FD_BASE}/shared-agents/files/${mode}?ref=${encodeURIComponent(agentRef)}&id=${encodeURIComponent(id)}` + auth
}

/** Upload into my own folder on the agent (saved/downloads/<my session>/).
 *  A FormData body, so the browser sends a Content-Length — Flight Deck
 *  refuses a streamed upload (411) and a too-large one before reading it. */
export async function uploadSharedFile(agentRef: string, file: File, lane = 'A'): Promise<SharedFile> {
  const form = new FormData()
  form.append('file', file)
  return fdFetch<SharedFile>(
    `/shared-agents/files/upload?ref=${encodeURIComponent(agentRef)}&lane=${encodeURIComponent(lane)}`,
    { method: 'POST', body: form },
  )
}

export const deleteSharedFile = (agentRef: string, id: string) =>
  fdFetch<{ ok: boolean }>(`/shared-agents/files/delete?ref=${encodeURIComponent(agentRef)}`,
    { method: 'POST', body: JSON.stringify({ id }) })

export const listSharedTables = (agentRef: string) =>
  fdFetch<SharedTable[]>(`/shared-agents/datastore/tables?ref=${encodeURIComponent(agentRef)}`)
