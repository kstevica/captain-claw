import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// ── Types (mirror captain_claw/flight_deck/deep_memory_routes.py) ─────

export interface DMStatus {
  enabled: boolean
  collection?: string
  /** Width the live Typesense collection declares. */
  embedding_dims?: number
  /** Width the embedding provider actually emits. A mismatch between these two
   *  is what silently disabled hybrid search for the entire life of the
   *  feature, so the UI surfaces both side by side. */
  provider_dims?: number
  vectors_disabled?: boolean
  /** Documents with no owner_id — indexed before tenancy existed, so
   *  owner-scoped search can never return them. */
  unowned?: number
  error?: string
}

export interface DMProjectEntry {
  enabled?: boolean
}

export interface DMHit {
  reference: string
  source: string
  score: number
  snippet: string
  summary: string
  chunk_index: number
  start_line: number
  end_line: number
  updated_at: number
}

export interface DMIndexResult {
  ok: boolean
  status: string
  reference?: string
  chunks?: number
  reason?: string
  tally?: Record<string, number>
  indexed?: string[]
  deleted?: number
}

// ── API helpers ──────────────────────────────────────────────────────

function _headers(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function _authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const build = (): RequestInit => ({
    ...init,
    headers: { ..._headers(), ...((init.headers as Record<string, string>) || {}) },
    credentials: 'include',
  })
  let res = await fetch(url, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) res = await fetch(url, build())
  }
  return res
}

const qp = (o: Record<string, string>) =>
  Object.entries(o)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&')

async function _json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.text()) || res.statusText)
  return (await res.json()) as T
}

// ── Store ─────────────────────────────────────────────────────────────

interface DeepMemoryStore {
  status: DMStatus | null
  /** project name → { enabled } — the `.vfs-index.json` opt-in registry. */
  projects: Record<string, DMProjectEntry>
  query: string
  results: DMHit[]
  searched: boolean
  loading: boolean
  busy: string        // project currently being indexed ('' = idle)
  error: string | null
  notice: string | null

  loadStatus: () => Promise<void>
  loadProjects: () => Promise<void>
  setQuery: (q: string) => void
  search: () => Promise<void>
  clearResults: () => void
  toggleIndexing: (project: string, enabled: boolean) => Promise<void>
  indexProject: (project: string, summarize: boolean) => Promise<void>
  indexFile: (project: string, path: string, summarize: boolean) => Promise<void>
  dropProject: (project: string) => Promise<void>
  claimUnowned: () => Promise<void>
  dismiss: () => void
}

export const useDeepMemoryStore = create<DeepMemoryStore>((set, get) => ({
  status: null,
  projects: {},
  query: '',
  results: [],
  searched: false,
  loading: false,
  busy: '',
  error: null,
  notice: null,

  loadStatus: async () => {
    try {
      set({ status: await _json<DMStatus>(await _authedFetch('/fd/deep-memory/status')) })
    } catch (e) {
      set({ status: { enabled: false }, error: String(e) })
    }
  },

  loadProjects: async () => {
    set({ loading: true, error: null })
    try {
      const d = await _json<{ projects: Record<string, DMProjectEntry> }>(
        await _authedFetch('/fd/deep-memory/projects'),
      )
      set({ projects: d.projects || {}, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  setQuery: (q) => set({ query: q }),

  clearResults: () => set({ results: [], searched: false, query: '' }),

  search: async () => {
    const q = get().query.trim()
    if (!q) return
    set({ loading: true, error: null })
    try {
      const d = await _json<{ results: DMHit[] }>(
        await _authedFetch(`/fd/deep-memory/search?${qp({ q, max_results: '25' })}`),
      )
      set({ results: d.results || [], searched: true, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false, searched: true, results: [] })
    }
  },

  toggleIndexing: async (project, enabled) => {
    set({ error: null })
    try {
      await _json(
        await _authedFetch('/fd/deep-memory/indexing', {
          method: 'POST',
          body: JSON.stringify({ project, enabled }),
        }),
      )
      set({ projects: { ...get().projects, [project]: { enabled } } })
      // Switching off deliberately leaves indexed content in place — a toggle
      // is never silently destructive. Say so, and offer the explicit path.
      if (!enabled) set({ notice: `Auto-indexing off for "${project}". Already-indexed content is still searchable — use Remove to drop it.` })
    } catch (e) {
      set({ error: String(e) })
    }
  },

  indexProject: async (project, summarize) => {
    set({ busy: project, error: null, notice: null })
    try {
      const r = await _json<DMIndexResult>(
        await _authedFetch('/fd/deep-memory/index-project', {
          method: 'POST',
          body: JSON.stringify({ project, summarize }),
        }),
      )
      const t = r.tally || {}
      const parts = Object.entries(t).map(([k, v]) => `${v} ${k}`)
      set({
        busy: '',
        notice: parts.length ? `"${project}": ${parts.join(', ')}` : `"${project}": nothing eligible to index`,
      })
    } catch (e) {
      set({ error: String(e), busy: '' })
    }
  },

  indexFile: async (project, path, summarize) => {
    set({ busy: project, error: null, notice: null })
    try {
      const r = await _json<DMIndexResult>(
        await _authedFetch('/fd/deep-memory/index-file', {
          method: 'POST',
          body: JSON.stringify({ project, path, summarize }),
        }),
      )
      set({
        busy: '',
        notice: r.ok
          ? `${path}: ${r.status}${r.chunks ? ` (${r.chunks} chunk${r.chunks === 1 ? '' : 's'})` : ''}`
          : `${path}: ${r.status} — ${r.reason || ''}`,
      })
    } catch (e) {
      set({ error: String(e), busy: '' })
    }
  },

  dropProject: async (project) => {
    set({ busy: project, error: null, notice: null })
    try {
      const r = await _json<DMIndexResult>(
        await _authedFetch('/fd/deep-memory/drop', {
          method: 'POST',
          body: JSON.stringify({ project, path: '', recursive: true }),
        }),
      )
      set({ busy: '', notice: `Removed ${r.deleted ?? 0} chunk(s) for "${project}".` })
    } catch (e) {
      set({ error: String(e), busy: '' })
    }
  },

  claimUnowned: async () => {
    set({ error: null, notice: null })
    try {
      const r = await _json<{ claimed: number }>(
        await _authedFetch('/fd/deep-memory/claim-unowned', { method: 'POST' }),
      )
      set({ notice: `Claimed ${r.claimed} previously unowned document(s).` })
      await get().loadStatus()
    } catch (e) {
      set({ error: String(e) })
    }
  },

  dismiss: () => set({ error: null, notice: null }),
}))
