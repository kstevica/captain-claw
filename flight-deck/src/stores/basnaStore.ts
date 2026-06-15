import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'
import type { TierMap, EnvVar } from '../services/tierConfig'

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
  progress: string   // JSON array of ProgressEvent
  files: string      // JSON array of BasnaFile
  created_at: string
  updated_at: string
}

export interface BasnaFile { name: string; mime: string; size: number; kind?: 'input' | 'generated'; agent?: string }

// Client-side attachment: a BasnaFile plus the local blob (until uploaded).
export interface AttachedFile extends BasnaFile { file?: File; uploaded: boolean }

export interface RouteSelected {
  archetype_id: string
  role: string
  tier: string
  why: string
  prior_weight: number
  // Optional per-agent overrides set in the route editor (take precedence over
  // the Library tier / archetype defaults at spawn + dispatch).
  provider?: string
  model?: string
  api_key?: string
  base_url?: string
  max_context?: number
  max_tokens?: number
  cognitive_mode?: string
  fleet_instructions?: string
  extra?: string
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
  actions: string         // JSON array of { tool, detail }
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

export interface ProgressEvent {
  i: number
  ts?: number     // epoch seconds (server clock)
  stage: string   // route | spawn | dispatch | action | narration | usage | merge | learn | done
  message: string
  ok?: boolean
  // Structured fields on per-agent events, so the UI can group the stream into
  // live per-agent panels instead of parsing the message string.
  agent?: string  // role/name of the agent this event belongs to
  tool?: string   // tool name on action/narration events
  detail?: string // tool-arg summary on action/narration events
  // Live cumulative token counts on `usage` events.
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
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

async function apiSaveRoute(sessionId: string, route: RoutePlan): Promise<void> {
  await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PUT', body: JSON.stringify({ route: JSON.stringify(route) }),
  })
}

async function apiExecute(body: Record<string, unknown>): Promise<ExecuteResult> {
  const res = await _authedFetch('/fd/basna/execute', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.text()) || 'execute failed')
  return res.json()
}

async function apiProgress(id: string): Promise<{ events: ProgressEvent[]; active: boolean }> {
  const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(id)}/progress`)
  if (!res.ok) return { events: [], active: false }
  return res.json()
}

async function apiUploadFiles(sessionId: string, files: File[]): Promise<{ files: BasnaFile[] }> {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const build = (): RequestInit => {
    const { token, authEnabled } = useAuthStore.getState()
    const headers: Record<string, string> = {}  // no Content-Type — browser sets multipart boundary
    if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
    return { method: 'POST', headers, credentials: 'include', body: form }
  }
  const url = `/fd/basna/sessions/${encodeURIComponent(sessionId)}/files`
  let res = await fetch(url, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) res = await fetch(url, build())
  }
  if (!res.ok) throw new Error((await res.text()) || 'file upload failed')
  return res.json()
}

async function apiDeleteFile(sessionId: string, name: string): Promise<void> {
  await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sessionId)}/files/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
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

function parseProgress(s?: string): ProgressEvent[] {
  if (!s) return []
  try {
    const a = JSON.parse(s)
    return Array.isArray(a) ? (a as ProgressEvent[]) : []
  } catch {
    return []
  }
}

function parseFiles(s?: string): AttachedFile[] {
  if (!s) return []
  try {
    const a = JSON.parse(s)
    return Array.isArray(a) ? (a as BasnaFile[]).map((f) => ({ ...f, uploaded: true })) : []
  } catch {
    return []
  }
}

const _ROUTER_TIER_LS = 'basna.routerTier'

// ── Store ────────────────────────────────────────────────────────────

interface BasnaStore {
  sessions: BasnaSession[]
  activeSession: BasnaSession | null
  routePlan: RoutePlan | null
  runs: BasnaRun[]
  lastExecute: ExecuteResult | null
  progress: ProgressEvent[]
  attachments: AttachedFile[]

  listLoading: boolean
  routing: boolean
  executing: boolean
  error: string | null

  routerTier: string   // which Library tier selects the archetypes (the router)
  maxAgents: number

  setRouterTier: (t: string) => void
  setMaxAgents: (n: number) => void
  addFiles: (files: FileList | File[]) => void
  removeFile: (name: string) => Promise<void>
  downloadFile: (name: string) => Promise<void>
  fetchFileText: (name: string) => Promise<string>

  loadSessions: () => Promise<void>
  selectSession: (id: string) => Promise<void>
  newSession: () => void
  updateSelected: (index: number, patch: Partial<RouteSelected>) => void
  route: (intent: string, tiers: TierMap) => Promise<void>
  execute: (tiers: TierMap, envVars: EnvVar[]) => Promise<void>
  sendFeedback: (runId: number, success: boolean) => Promise<void>
  deleteSession: (id: string) => Promise<void>
}

export const useBasnaStore = create<BasnaStore>((set, get) => ({
  sessions: [],
  activeSession: null,
  routePlan: null,
  runs: [],
  lastExecute: null,
  progress: [],
  attachments: [],

  listLoading: false,
  routing: false,
  executing: false,
  error: null,

  routerTier: (typeof localStorage !== 'undefined' && localStorage.getItem(_ROUTER_TIER_LS)) || 'reason',
  maxAgents: 6,

  setRouterTier: (t) => {
    try { localStorage.setItem(_ROUTER_TIER_LS, t) } catch { /* ignore */ }
    set({ routerTier: t })
  },
  setMaxAgents: (n) => set({ maxAgents: Math.max(1, Math.min(10, n)) }),

  addFiles: (files) => {
    const incoming = Array.from(files).map((f): AttachedFile => ({
      name: f.name, mime: f.type || 'application/octet-stream', size: f.size, file: f, uploaded: false,
    }))
    // Replace any existing entry with the same name.
    const byName = new Map(get().attachments.map((a) => [a.name, a]))
    for (const a of incoming) byName.set(a.name, a)
    set({ attachments: Array.from(byName.values()) })
  },

  removeFile: async (name) => {
    const a = get().attachments.find((x) => x.name === name)
    const sid = get().activeSession?.id
    if (a?.uploaded && sid) {
      try { await apiDeleteFile(sid, name) } catch { /* ignore */ }
    }
    set({ attachments: get().attachments.filter((x) => x.name !== name) })
  },

  downloadFile: async (name) => {
    const sid = get().activeSession?.id
    if (!sid) return
    const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sid)}/files/${encodeURIComponent(name)}`)
    if (!res.ok) return
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.click()
    URL.revokeObjectURL(url)
  },

  // Fetch a generated file's content as text for in-app preview.
  fetchFileText: async (name) => {
    const sid = get().activeSession?.id
    if (!sid) return ''
    const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sid)}/files/${encodeURIComponent(name)}`)
    if (!res.ok) return ''
    return res.text()
  },

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
    set({ activeSession: s, routePlan: parseRoute(s.route), runs, lastExecute: null,
          progress: parseProgress(s.progress), attachments: parseFiles(s.files), error: null })
  },

  newSession: () => set({ activeSession: null, routePlan: null, runs: [], lastExecute: null, progress: [], attachments: [], error: null }),

  updateSelected: (index, patch) => {
    const plan = get().routePlan
    if (!plan) return
    const selected = plan.selected.map((s, i) => (i === index ? { ...s, ...patch } : s))
    set({ routePlan: { ...plan, selected } })
  },

  route: async (intent, tiers) => {
    set({ routing: true, error: null })
    try {
      const sid = get().activeSession?.id
      // The router runs on the user-selected Library tier (default reasoning).
      const tc = tiers[get().routerTier]
      const creds = tc?.model
        ? { provider: tc.provider, model: tc.model, api_key: tc.api_key || undefined, base_url: tc.base_url || undefined }
        : {}
      const plan = await apiRoute({
        intent,
        max_agents: get().maxAgents,
        ...creds,
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

  execute: async (tiers, envVars) => {
    const sid = get().activeSession?.id
    if (!sid) return
    // Upload any attachments not yet on the server before the run.
    const pending = get().attachments.filter((a) => !a.uploaded && a.file)
    if (pending.length) {
      try {
        const res = await apiUploadFiles(sid, pending.map((a) => a.file as File))
        set({ attachments: (res.files || []).map((f) => ({ ...f, uploaded: true })) })
      } catch (e) {
        set({ error: e instanceof Error ? e.message : 'file upload failed' })
        return
      }
    }
    // Persist any per-agent edits made in the route editor before the run.
    if (get().routePlan) {
      try { await apiSaveRoute(sid, get().routePlan as RoutePlan) } catch { /* ignore */ }
    }
    set({ executing: true, error: null, progress: [] })
    // Poll the live progress log while the (blocking) execute call runs.
    const poll = setInterval(async () => {
      try { const p = await apiProgress(sid); set({ progress: p.events || [] }) } catch { /* ignore */ }
    }, 700)
    try {
      // Spawned agents + merge calls resolve their model/key from the Library tiers;
      // env vars (Library "Additional API Keys") are passed to every agent.
      const env_vars = (envVars || []).filter((e) => e.key.trim() && e.value.trim())
      const res = await apiExecute({ session_id: sid, tiers, env_vars })
      const s = await apiGetSession(sid)
      const runs = await apiListRuns(sid)
      // Refresh attachments from the updated session so files the agents
      // generated during the run (kind: 'generated') surface in the UI's
      // "Generated files" list — otherwise they're captured but unreachable.
      set({ lastExecute: res, activeSession: s, runs,
            ...(s ? { attachments: parseFiles(s.files) } : {}) })
      await get().loadSessions()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'execute failed' })
    } finally {
      clearInterval(poll)
      try { const p = await apiProgress(sid); set({ progress: p.events || [] }) } catch { /* ignore */ }
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
