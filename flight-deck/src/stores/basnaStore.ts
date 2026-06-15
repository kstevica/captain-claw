import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// ── Types (mirror captain_claw/flight_deck/basna_routes.py) ──────────

export interface BasnaSession {
  id: string
  intent: string
  domain: string
  difficulty: string
  merge_kind: string
  status: string
  route: string   // JSON string of RoutePlan
  truth: string
  confidence: number
  config: string
  created_at: string
  updated_at: string
}

export interface RouteSelected {
  archetype_id: string
  role: string
  tier: string
  why: string
  prior_weight: number
}

export interface RoutePlan {
  domain: string
  difficulty: string
  merge_kind: string
  rationale: string
  selected: RouteSelected[]
  source?: string
  elapsed_ms?: number
  session_id?: string
}

export interface BasnaRun {
  id: number
  session_id: string
  archetype_id: string
  role: string
  tier: string
  weight_at_run: number
  output: string
  success: number | null  // 1 / 0 / null
  latency_ms: number
  created_at: string
}

export interface ExecuteAgent {
  archetype_id: string
  role: string
  ok: boolean
  latency_ms: number
  weight: number
  run_id: number | null
  success: boolean | null
}

export interface ExecuteResult {
  session_id: string
  domain: string
  merge_kind: string
  truth: string
  confidence: number
  method: string
  contributors: string[]
  agents: ExecuteAgent[]
  learned: { archetype_id: string; run_id: number; success: boolean; weight: number }[]
  spawned: number
  dispatched: number
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

async function apiListSessions(): Promise<BasnaSession[]> {
  const res = await _authedFetch('/fd/basna/sessions')
  if (!res.ok) return []
  return res.json()
}

async function apiGetSession(id: string): Promise<BasnaSession | null> {
  const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(id)}`)
  if (!res.ok) return null
  return res.json()
}

async function apiDeleteSession(id: string): Promise<void> {
  await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

async function apiListRuns(id: string): Promise<BasnaRun[]> {
  const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(id)}/runs`)
  if (!res.ok) return []
  return res.json()
}

async function apiRoute(body: Record<string, unknown>): Promise<RoutePlan> {
  const res = await _authedFetch('/fd/basna/route', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.text()) || 'route failed')
  return res.json()
}

async function apiExecute(body: Record<string, unknown>): Promise<ExecuteResult> {
  const res = await _authedFetch('/fd/basna/execute', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.text()) || 'execute failed')
  return res.json()
}

async function apiFeedback(runId: number, success: boolean): Promise<void> {
  await _authedFetch(`/fd/basna/runs/${runId}/feedback`, {
    method: 'POST', body: JSON.stringify({ success }),
  })
}

export function parseRoute(s?: string): RoutePlan | null {
  if (!s) return null
  try {
    const o = JSON.parse(s)
    return o && Array.isArray(o.selected) ? (o as RoutePlan) : null
  } catch {
    return null
  }
}

const _API_KEY_LS = 'basna.apiKey'

// ── Store ────────────────────────────────────────────────────────────

interface BasnaStore {
  sessions: BasnaSession[]
  activeSession: BasnaSession | null
  routePlan: RoutePlan | null
  runs: BasnaRun[]
  lastExecute: ExecuteResult | null

  listLoading: boolean
  routing: boolean
  executing: boolean
  error: string | null

  apiKey: string
  maxAgents: number

  setApiKey: (k: string) => void
  setMaxAgents: (n: number) => void

  loadSessions: () => Promise<void>
  selectSession: (id: string) => Promise<void>
  newSession: () => void
  route: (intent: string) => Promise<void>
  execute: () => Promise<void>
  sendFeedback: (runId: number, success: boolean) => Promise<void>
  deleteSession: (id: string) => Promise<void>
}

export const useBasnaStore = create<BasnaStore>((set, get) => ({
  sessions: [],
  activeSession: null,
  routePlan: null,
  runs: [],
  lastExecute: null,

  listLoading: false,
  routing: false,
  executing: false,
  error: null,

  apiKey: (typeof localStorage !== 'undefined' && localStorage.getItem(_API_KEY_LS)) || '',
  maxAgents: 6,

  setApiKey: (k) => {
    try { localStorage.setItem(_API_KEY_LS, k) } catch { /* ignore */ }
    set({ apiKey: k })
  },
  setMaxAgents: (n) => set({ maxAgents: Math.max(1, Math.min(10, n)) }),

  loadSessions: async () => {
    set({ listLoading: true })
    try {
      set({ sessions: await apiListSessions() })
    } finally {
      set({ listLoading: false })
    }
  },

  selectSession: async (id) => {
    const s = await apiGetSession(id)
    if (!s) return
    const runs = await apiListRuns(id)
    set({ activeSession: s, routePlan: parseRoute(s.route), runs, lastExecute: null, error: null })
  },

  newSession: () => set({ activeSession: null, routePlan: null, runs: [], lastExecute: null, error: null }),

  route: async (intent) => {
    set({ routing: true, error: null })
    try {
      const sid = get().activeSession?.id
      const plan = await apiRoute({
        intent,
        max_agents: get().maxAgents,
        ...(get().apiKey ? { api_key: get().apiKey } : {}),
        ...(sid ? { session_id: sid } : {}),
      })
      const s = plan.session_id ? await apiGetSession(plan.session_id) : null
      set({ routePlan: plan, activeSession: s, runs: [], lastExecute: null })
      await get().loadSessions()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'route failed' })
    } finally {
      set({ routing: false })
    }
  },

  execute: async () => {
    const sid = get().activeSession?.id
    if (!sid) return
    set({ executing: true, error: null })
    try {
      const res = await apiExecute({
        session_id: sid,
        ...(get().apiKey ? { api_key: get().apiKey } : {}),
      })
      const s = await apiGetSession(sid)
      const runs = await apiListRuns(sid)
      set({ lastExecute: res, activeSession: s, runs })
      await get().loadSessions()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'execute failed' })
    } finally {
      set({ executing: false })
    }
  },

  sendFeedback: async (runId, success) => {
    await apiFeedback(runId, success)
    const sid = get().activeSession?.id
    if (sid) set({ runs: await apiListRuns(sid) })
  },

  deleteSession: async (id) => {
    await apiDeleteSession(id)
    if (get().activeSession?.id === id) get().newSession()
    await get().loadSessions()
  },
}))
