import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// ── Types (mirror captain_claw/config.py AutonomousWorkConfig + autonomy.py) ──

export interface AutonomyConfig {
  enabled: boolean
  autonomy_level: string          // off | propose | act_low_risk | act
  max_autonomy_level: string      // shipped ceiling (server-owned, read-only)
  arbiter_on_pulse: boolean
  arbiter_min_score: number
  max_actions_per_day: number
  max_concurrent_actions: number
  candidate_lookback_hours: number
  quiet_hours_start: number
  quiet_hours_end: number
  allow_auto_dispatch: boolean
  low_risk_kinds: string[]
  high_risk_requires_approval: boolean
  learning_enabled: boolean
  judge_mode: string              // auto | human | both
  reliability_seed: number
  suppress_below_weight: number
  reflection_to_intention: boolean
  max_intentions_per_reflection: number
  reflection_intention_max_risk: string
  granted_actions: string[]
  db_path: string
}

export interface AutonomyAction {
  id: string
  user_id: string
  source: string
  kind: string
  title: string
  rationale: string
  risk: string
  domain: string
  score: number
  status: string
  target: string
  ref_id: string
  payload: Record<string, unknown>
  outcome: string | null
  outcome_note: string
  created_at: string
  dispatched_at: string | null
  completed_at: string | null
}

export interface ReliabilityRow {
  user_id: string
  kind: string
  domain: string
  successes: number
  fails: number
  runs: number
  weight: number
  updated_at: string
}

export interface LogEntry {
  id: number
  ts: string
  level: string   // info | warn | error
  event: string
  detail: string
}

export interface CatalogItem {
  id: string
  label: string
  risk: string
  reversibility: string   // read_only | reversible | irreversible
  grant: string
  human_only: boolean
  args: string[]
  required: string[]
}

// ── API helpers (auth token + 401 refresh, mirroring basnaStore) ──────

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

// ── Store ─────────────────────────────────────────────────────────────

interface AutonomyStore {
  config: AutonomyConfig | null
  defaults: AutonomyConfig | null
  actions: AutonomyAction[]
  reliability: ReliabilityRow[]
  log: LogEntry[]
  catalog: CatalogItem[]
  loading: boolean
  saving: boolean
  error: string | null

  loadAll: () => Promise<void>
  loadActions: (status?: string) => Promise<void>
  setField: <K extends keyof AutonomyConfig>(key: K, value: AutonomyConfig[K]) => void
  save: () => Promise<void>
  approve: (id: string) => Promise<void>
  reject: (id: string) => Promise<void>
  undo: (id: string) => Promise<void>
  nudge: () => Promise<{ proposed: number; reason?: string }>
}

export const useAutonomyStore = create<AutonomyStore>((set, get) => ({
  config: null,
  defaults: null,
  actions: [],
  reliability: [],
  log: [],
  catalog: [],
  loading: false,
  saving: false,
  error: null,

  loadAll: async () => {
    set({ loading: true, error: null })
    try {
      const [cfgRes, actRes, relRes, logRes, catRes] = await Promise.all([
        _authedFetch('/fd/autonomy/config'),
        _authedFetch('/fd/autonomy/actions?limit=100'),
        _authedFetch('/fd/autonomy/reliability'),
        _authedFetch('/fd/autonomy/log?limit=100'),
        _authedFetch('/fd/autonomy/catalog'),
      ])
      const cfg = cfgRes.ok ? await cfgRes.json() : {}
      const act = actRes.ok ? await actRes.json() : {}
      const rel = relRes.ok ? await relRes.json() : {}
      const lg = logRes.ok ? await logRes.json() : {}
      const cat = catRes.ok ? await catRes.json() : {}
      set({
        config: cfg.config ?? null,
        defaults: cfg.defaults ?? null,
        actions: act.actions ?? [],
        reliability: rel.reliability ?? [],
        log: lg.log ?? [],
        catalog: cat.catalog ?? [],
      })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    } finally {
      set({ loading: false })
    }
  },

  loadActions: async (status?: string) => {
    try {
      const qs = status ? `?status=${encodeURIComponent(status)}&limit=100` : '?limit=100'
      const [actRes, relRes, logRes] = await Promise.all([
        _authedFetch(`/fd/autonomy/actions${qs}`),
        _authedFetch('/fd/autonomy/reliability'),
        _authedFetch('/fd/autonomy/log?limit=100'),
      ])
      if (actRes.ok) set({ actions: (await actRes.json()).actions ?? [] })
      if (relRes.ok) set({ reliability: (await relRes.json()).reliability ?? [] })
      if (logRes.ok) set({ log: (await logRes.json()).log ?? [] })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  setField: (key, value) => {
    const cfg = get().config
    if (!cfg) return
    set({ config: { ...cfg, [key]: value } })
  },

  save: async () => {
    const cfg = get().config
    if (!cfg) return
    set({ saving: true, error: null })
    try {
      const res = await _authedFetch('/fd/autonomy/config', {
        method: 'PUT',
        body: JSON.stringify({ config: cfg }),
      })
      if (!res.ok) throw new Error((await res.text()) || 'save failed')
      const data = await res.json()
      set({ config: data.config ?? cfg, defaults: data.defaults ?? get().defaults })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
      throw e
    } finally {
      set({ saving: false })
    }
  },

  approve: async (id: string) => {
    const res = await _authedFetch(`/fd/autonomy/actions/${encodeURIComponent(id)}/approve`, { method: 'POST' })
    if (res.ok) await get().loadActions()
  },

  reject: async (id: string) => {
    const res = await _authedFetch(`/fd/autonomy/actions/${encodeURIComponent(id)}/reject`, { method: 'POST' })
    if (res.ok) await get().loadActions()
  },

  undo: async (id: string) => {
    const res = await _authedFetch(`/fd/autonomy/actions/${encodeURIComponent(id)}/undo`, { method: 'POST' })
    if (res.ok) await get().loadActions()
  },

  nudge: async () => {
    const res = await _authedFetch('/fd/autonomy/nudge', { method: 'POST' })
    const data = res.ok ? await res.json() : {}
    await get().loadActions()
    const arb = data.arbiter || {}
    return { proposed: arb.proposed ?? 0, reason: arb.reason ?? data.pulse ?? data.reason }
  },
}))
