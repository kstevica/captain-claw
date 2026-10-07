// Bat — the stubborn finisher. Store for the Bat page: list runs, watch one
// run live, and answer the human-in-the-loop asks (plan approvals, spend
// approvals, a 2FA code / credential). Mirrors basnaStore's auth/fetch style;
// talks to Flight Deck over relative /fd/bat/... paths.
import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'
import type { ProgressEvent } from './basnaStore'

export interface BatRun {
  id: string
  title: string
  task: string
  status: string
  created_at?: number
  updated_at?: number
  cumulative_usd?: number
  llm_usd_cap?: number
  real_usd_cap?: number
  stopped_reason?: string
  truth?: string
  vfs_project?: string
  email_allowed?: boolean
  spend_allowed?: boolean
  account_allowed?: boolean
}

export interface BatStep {
  step_key: string
  seq: number
  title: string
  status: string
  attempt: number
  output?: string
  error?: string
  produced_file?: string
}

export interface BatSpendItem {
  id: string
  merchant: string
  amount_usd_max: number
  actual_usd: number
  status: string
}

export interface BatSpend {
  items: BatSpendItem[]
  committed_usd: number
  cap: number
}

export interface BatAsk {
  id: string
  run_id: string
  kind: string // plan_approval | spend_approval | input | secret
  question: string
  options: string[]
  secret: boolean
}

export interface BatDetail {
  run: BatRun
  steps: BatStep[]
  events: ProgressEvent[]
  spend: BatSpend
}

const RUNNING = ['planning', 'running', 'retrying', 'waiting', 'awaiting_plan', 'awaiting_human']

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

interface BatStore {
  runs: BatRun[]
  active: BatDetail | null
  activeId: string | null
  asks: BatAsk[]
  listLoading: boolean
  busy: boolean
  error: string

  loadRuns: () => Promise<void>
  loadAsks: () => Promise<void>
  select: (id: string | null) => Promise<void>
  poll: () => Promise<void>
  startRun: (task: string, opts?: { title?: string; llm_usd_cap?: number; real_usd_cap?: number; per_item_usd?: number }) => Promise<string | null>
  answer: (askId: string, text: string) => Promise<boolean>
  approvePlan: (runId: string, approve: boolean) => Promise<void>
  cancel: (runId: string) => Promise<void>
}

export const useBatStore = create<BatStore>((set, get) => ({
  runs: [],
  active: null,
  activeId: null,
  asks: [],
  listLoading: false,
  busy: false,
  error: '',

  loadRuns: async () => {
    set({ listLoading: true })
    try {
      const res = await _authedFetch('/fd/bat/runs')
      const data = res.ok ? await res.json() : { runs: [] }
      set({ runs: Array.isArray(data.runs) ? data.runs : [] })
    } catch {
      /* ignore */
    } finally {
      set({ listLoading: false })
    }
  },

  loadAsks: async () => {
    try {
      const res = await _authedFetch('/fd/bat/asks')
      const data = res.ok ? await res.json() : { asks: [] }
      set({ asks: Array.isArray(data.asks) ? data.asks : [] })
    } catch {
      /* ignore */
    }
  },

  select: async (id) => {
    set({ activeId: id, active: null })
    if (!id) return
    try {
      const res = await _authedFetch(`/fd/bat/runs/${encodeURIComponent(id)}`)
      if (res.ok) set({ active: await res.json() })
    } catch {
      /* ignore */
    }
  },

  poll: async () => {
    await get().loadRuns()
    await get().loadAsks()
    const id = get().activeId
    if (!id) return
    const cur = get().active?.run
    // Refetch the open run's detail while it is live (cheap; the page is small).
    if (!cur || RUNNING.includes(cur.status)) {
      try {
        const res = await _authedFetch(`/fd/bat/runs/${encodeURIComponent(id)}`)
        if (res.ok) set({ active: await res.json() })
      } catch {
        /* ignore */
      }
    }
  },

  startRun: async (task, opts = {}) => {
    set({ busy: true, error: '' })
    try {
      const res = await _authedFetch('/fd/bat/start', {
        method: 'POST',
        body: JSON.stringify({ task, title: opts.title || '', llm_usd_cap: opts.llm_usd_cap || 0,
          real_usd_cap: opts.real_usd_cap || 0, per_item_usd: opts.per_item_usd || 0 }),
      })
      if (!res.ok) {
        set({ error: `Could not start (${res.status})` })
        return null
      }
      const data = await res.json()
      await get().loadRuns()
      if (data.run_id) await get().select(data.run_id)
      return data.run_id || null
    } catch (e) {
      set({ error: String(e) })
      return null
    } finally {
      set({ busy: false })
    }
  },

  answer: async (askId, text) => {
    try {
      const res = await _authedFetch(`/fd/bat/asks/${encodeURIComponent(askId)}/answer`, {
        method: 'POST',
        body: JSON.stringify({ text }),
      })
      if (!res.ok) {
        set({ error: `Answer rejected (${res.status})` })
        return false
      }
      await get().loadAsks()
      await get().poll()
      return true
    } catch (e) {
      set({ error: String(e) })
      return false
    }
  },

  approvePlan: async (runId, approve) => {
    try {
      await _authedFetch(`/fd/bat/runs/${encodeURIComponent(runId)}/approve-plan`, {
        method: 'POST',
        body: JSON.stringify({ approve }),
      })
    } catch {
      /* ignore */
    }
    await get().loadAsks()
    await get().poll()
  },

  cancel: async (runId) => {
    try {
      await _authedFetch(`/fd/bat/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST' })
    } catch {
      /* ignore */
    }
    await get().poll()
  },
}))
