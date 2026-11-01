/**
 * File library — the kiosk's "Library": every file the caller's agents made,
 * summarised and sorted by Flight Deck (`captain_claw/flight_deck/file_library.py`).
 * A scan walks each agent's workspace, summarises new and changed files with
 * the owner's fast tier, and keeps the results searchable by name, title,
 * summary, tags and text.
 */

import { refreshAccessToken, useAuthStore } from '../stores/authStore'

export type LibraryStatus = 'pending' | 'done' | 'error' | 'skipped'

export interface LibraryItem {
  id: string
  agent_slug: string
  agent_name: string
  agent_running: boolean
  rel_path: string
  /** The file on disk, named as the agent's Files panel names it — what the
   *  agent's view, save, deck and public-link routes take. */
  physical: string
  /** `workspace/<rel_path>`, the panel's name for it. */
  logical: string
  filename: string
  extension: string
  kind: string
  mime_type: string
  size: number
  /** Epoch seconds. */
  modified: number
  status: LibraryStatus
  title: string
  summary: string
  category: string
  tags: string[]
  error: string
  model: string
  summarized_at: number | null
  added_at: number
  /** A line of the file's text around a search term the other fields don't show. */
  snippet: string
}

export interface Facet { id: string; label: string; count: number }

export interface LibraryPage {
  items: LibraryItem[]
  total: number
  facets: { agents: Facet[]; categories: Facet[]; kinds: Facet[] }
}

export interface ScanJob {
  running: boolean
  phase: 'idle' | 'scanning' | 'summarizing'
  total: number
  done: number
  failed: number
  current: string
  started_at: number | null
  finished_at: number | null
  message: string
}

export interface LibraryState {
  job: ScanJob
  counts: { total: number; done: number; pending: number; error: number; skipped: number }
  agents: number
}

export type LibrarySort = 'recent' | 'name' | 'size' | 'agent'

export interface LibraryQuery {
  q?: string
  agent?: string
  category?: string
  kind?: string
  sort?: LibrarySort
  limit?: number
  offset?: number
}

/** JSON from a Flight Deck route, with the FD token; a refusal's `detail` as the error.
 *  A token only lasts minutes, so one that ran out is renewed and the request sent once more. */
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const send = () => {
    const jwt = useAuthStore.getState().token
    return fetch(path, {
      ...init,
      cache: 'no-store',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...(jwt ? { Authorization: `Bearer ${jwt}` } : {}),
        ...(init?.headers || {}),
      },
    })
  }
  let r = await send()
  if (r.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) r = await send()
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = typeof body?.detail === 'string' ? body.detail : ''
    } catch { /* not JSON */ }
    throw new Error(detail || `${r.status} ${r.statusText}`)
  }
  return (await r.json()) as T
}

/** A file URL a viewer or link can open directly: the FD token rides along.
 *  `version` (the file's change time) makes each version its own URL, so a
 *  browser never shows a cached copy of an earlier one. */
function fileUrl(id: string, what: 'view' | 'download' | 'thumb', version?: number): string {
  const { token, authEnabled } = useAuthStore.getState()
  const q = new URLSearchParams()
  if (authEnabled && token) q.set('fd_token', token)
  if (version !== undefined) q.set('v', String(version))
  const qs = q.toString()
  return `/fd/file-library/files/${encodeURIComponent(id)}/${what}${qs ? `?${qs}` : ''}`
}

export const fileLibraryApi = {
  status: () => api<LibraryState>('/fd/file-library/status'),
  /** Look for new and changed files; `retryErrors` also retries failed summaries now. */
  scan: (retryErrors = false) =>
    api<LibraryState>(`/fd/file-library/scan${retryErrors ? '?retry_errors=true' : ''}`, { method: 'POST' }),
  files: (query: LibraryQuery = {}) => {
    const p = new URLSearchParams()
    for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== '') p.set(k, String(v))
    return api<LibraryPage>(`/fd/file-library/files?${p}`)
  },
  resummarize: (id: string) =>
    api<LibraryItem>(`/fd/file-library/files/${encodeURIComponent(id)}/summarize`, { method: 'POST' }),
  viewUrl: (id: string, version?: number) => fileUrl(id, 'view', version),
  downloadUrl: (id: string) => fileUrl(id, 'download'),
  /** A PDF's first page as a PNG (404 where the deck can't render PDFs). */
  thumbUrl: (id: string, version?: number) => fileUrl(id, 'thumb', version),
}
