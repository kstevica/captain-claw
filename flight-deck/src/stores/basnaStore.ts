import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'
import type { TierMap, EnvVar } from '../services/tierConfig'
import { defaultProfile, fromResponse, toRequest } from '../services/quality'
import type { QualityProfile } from '../services/quality'

// ── Types (mirror captain_claw/flight_deck/basna_routes.py) ──────────

export interface BasnaSession {
  id: string
  title: string
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
  analysis: string   // JSON BasnaAnalysis (cross-agent comparison)
  created_at: string
  updated_at: string
  // Present when this run was shared TO you by another user.
  shared?: boolean
  access?: 'owner' | 'edit' | 'view'
  owner_email?: string
  owner_name?: string
}

// Cross-agent analysis surfaced above the compiled truth (Fusion-style).
export interface BasnaAnalysis {
  agreement?: string[]
  differences?: { point: string; positions?: { by: string; stance: string }[] }[]
  unique?: { by: string; insight: string }[]
  blind_spots?: string[]
  // Vatra runs store coverage gaps here instead (the Basna comparison is meaningless
  // for complementary pieces).
  coverage_summary?: string
  gaps?: { item: string; severity: 'major' | 'minor'; note?: string }[]
  // Quality-tightening records (docs/vatra-quality-tightening-plan.md) — each
  // key is present only when its lever actually ran.
  consistency?: {
    values_checked: number; relations_checked: number
    initial_critical: number; initial_major: number
    critical: number; major: number
    revised: boolean; truncated: boolean
  }
  contract?: {
    checked: number; passed: number
    failed_critical: number; failed_major: number; unclear: number
    failed: { id: string; text: string; severity: string; how: string; note?: string }[]
  }
  quality_verdict?: 'clean' | 'critical_findings_remain'
  blocking?: {
    verdict: string; rounds: number
    remaining: { source: string; severity: string; detail: string; note?: string }[]
  }
  quality_metrics?: Record<string, number | string | boolean>
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

// Vatra (collaborative mode): the Lead's decomposition into owner-assigned pieces.
export interface VatraSubtask {
  id: string
  title: string
  owner_archetype_id: string
  brief?: string
  group?: string   // execution group A/B/C/D the user arranged this owner into (empty = auto)
}

// A cross-agent request on the Vatra blackboard.
export interface VatraAsk {
  id: number
  from_owner: string
  from_subtask: string
  text: string
  status: 'open' | 'claimed' | 'answered' | 'dropped'
  answer: string
  answered_by: string
  depth: number
  note?: string
  created_at: string
  updated_at: string
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
  // Present on Vatra sessions: 'vatra' + the Lead's decomposition.
  mode?: 'basna' | 'vatra'
  subtasks?: VatraSubtask[]
  shared_context?: string   // the team contract every piece must follow
  brief?: string            // R12: the clarified, editable task brief the team was routed on
  group_instructions?: Record<string, string>  // per-group extra instructions {A,B,C,D}
}

export async function apiListVatraAsks(sessionId: string): Promise<VatraAsk[]> {
  const res = await _authedFetch(`/fd/vatra/sessions/${encodeURIComponent(sessionId)}/asks`)
  if (!res.ok) return []
  const data = await res.json()
  return (data.asks || []) as VatraAsk[]
}

// An entry on the Vatra shared board (live shared memory across the team).
export interface VatraBoardEntry {
  id: number
  from_owner: string
  from_subtask: string
  kind: 'note' | 'narration' | 'output' | 'file'
  title: string
  content: string
  created_at: string
}

export async function apiListVatraBoard(sessionId: string): Promise<VatraBoardEntry[]> {
  const res = await _authedFetch(`/fd/vatra/sessions/${encodeURIComponent(sessionId)}/board`)
  if (!res.ok) return []
  const data = await res.json()
  return (data.entries || []) as VatraBoardEntry[]
}

// Ask a still-working Vatra agent (by its live-panel label) to skip its turn.
export async function apiVatraSkipAgent(sessionId: string, agent: string): Promise<void> {
  await _authedFetch(`/fd/vatra/sessions/${encodeURIComponent(sessionId)}/skip`, {
    method: 'POST', body: JSON.stringify({ agent }),
  })
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
  analysis?: BasnaAnalysis | null
  agents: ExecuteAgent[]
  learned: { archetype_id: string; run_id: number; success: boolean; weight: number }[]
  spawned: number
  dispatched: number
  cost?: RunCost | null
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
  group?: string  // Vatra grouped mode: this owner's execution-phase letter (A..D)
  // Live cumulative token counts on `usage` events.
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  // Run cost block on the terminal `cost` event (and on the execute response).
  cost?: RunCost
}

// Per-run cost accounting — tokens (incl. cache split), dollar cost, and the
// effective $/hour that compares directly to a human wage. `usd`/`hourly_usd`
// are null when no priced model was used (tokens still count).
export interface RunCost {
  tokens: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    cache_creation_input_tokens: number
    cache_read_input_tokens: number
  }
  usd: number | null
  priced: boolean
  per_model: Record<string, { usd: number; prompt_tokens: number; completion_tokens: number; cache_read_input_tokens: number; priced: boolean; calls: number }>
  elapsed_seconds: number | null   // real wall-clock time of the run
  agent_seconds: number | null     // Σ of every model call's duration (> wall-clock when parallel)
  hourly_usd: number | null
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

// Plan-step child sessions (each step of a Plan-Horizon ensemble run is a real Basna
// session) are hidden from the list — they belong under their parent plan run.
function isPlanChild(config?: string): boolean {
  if (!config) return false
  try { return JSON.parse(config)?.source === 'plan-step' } catch { return false }
}

async function apiListSessions(): Promise<BasnaSession[]> {
  const res = await _authedFetch('/fd/basna/sessions')
  if (!res.ok) return []
  const rows: BasnaSession[] = await res.json()
  return Array.isArray(rows) ? rows.filter((s) => !isPlanChild(s.config)) : []
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

async function apiRecompile(sessionId: string, tiers: TierMap): Promise<Partial<ExecuteResult>> {
  const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sessionId)}/recompile`, {
    method: 'POST', body: JSON.stringify({ tiers }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'recompile failed')
  return res.json()
}

async function apiSaveRoute(sessionId: string, route: RoutePlan): Promise<void> {
  await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PUT', body: JSON.stringify({ route: JSON.stringify(route) }),
  })
}

async function apiSaveTitle(sessionId: string, title: string): Promise<BasnaSession | null> {
  const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PUT', body: JSON.stringify({ title }),
  })
  if (!res.ok) return null
  return res.json()
}

async function apiExecute(body: Record<string, unknown>): Promise<ExecuteResult> {
  const res = await _authedFetch('/fd/basna/execute', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.text()) || 'execute failed')
  return res.json()
}

// Resume a stalled/cancelled Basna run — runs inline like /execute and returns the
// final result (finished agents restored from checkpoints, only the missing re-run).
async function apiBasnaResume(sessionId: string, body: Record<string, unknown>): Promise<ExecuteResult> {
  const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(sessionId)}/resume`, {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.text()) || 'resume failed')
  return res.json()
}

// A wizard setup recommendation from POST /fd/basna/recommend.
export interface Recommendation {
  mode: 'basna' | 'vatra'
  rationale: string
  difficulty: 'trivial' | 'moderate' | 'hard'
  max_agents: number
  effort: 'standard' | 'deep' | 'plan'
  quality: 'basic' | 'balanced' | 'thorough'
  grouped: boolean
  shared_datastore: boolean
}

export async function apiRecommend(intent: string, creds: Record<string, unknown> = {}): Promise<Recommendation> {
  const res = await _authedFetch('/fd/basna/recommend', {
    method: 'POST', body: JSON.stringify({ intent, ...creds }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error((detail as { detail?: string }).detail || 'recommend failed')
  }
  return res.json()
}

async function apiListProjects(): Promise<VfsProject[]> {
  try {
    const res = await _authedFetch('/fd/vfs/projects')
    if (!res.ok) return []
    const data = await res.json()
    // /fd/vfs/projects returns { projects: [...] } (same shape the VFS browser reads).
    const list = Array.isArray(data) ? data : (data?.projects ?? [])
    return Array.isArray(list) ? (list as VfsProject[]) : []
  } catch { return [] }
}

async function apiProgress(id: string): Promise<{ events: ProgressEvent[]; active: boolean }> {
  const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(id)}/progress`)
  if (!res.ok) return { events: [], active: false }
  return res.json()
}

async function apiVatraRoute(body: Record<string, unknown>): Promise<{ session_id: string; title: string }> {
  const res = await _authedFetch('/fd/vatra/route', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error((detail as { detail?: string }).detail || 'vatra plan failed')
  }
  return res.json()
}

async function apiVatraExecute(body: Record<string, unknown>): Promise<{ session_id: string }> {
  const res = await _authedFetch('/fd/vatra/execute', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error((detail as { detail?: string }).detail || 'vatra run failed')
  }
  return res.json()
}

// Resume a stalled/cancelled Vatra run — backgrounds like /execute; the live monitor
// polls its progress. Returns as soon as the run is re-launched.
async function apiVatraResume(sessionId: string, body: Record<string, unknown>): Promise<{ session_id: string }> {
  const res = await _authedFetch(`/fd/vatra/sessions/${encodeURIComponent(sessionId)}/resume`, {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error((detail as { detail?: string }).detail || 'resume failed')
  }
  return res.json()
}

async function apiPlan(body: Record<string, unknown>): Promise<{ session_id: string }> {
  const res = await _authedFetch('/fd/basna/plan', {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error((detail as { detail?: string }).detail || 'plan run failed')
  }
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

export function parseAnalysis(s?: string): BasnaAnalysis | null {
  if (!s) return null
  try {
    const o = JSON.parse(s)
    if (!o || typeof o !== 'object') return null
    // Treat an empty object as "no analysis".
    return Object.keys(o).length ? (o as BasnaAnalysis) : null
  } catch {
    return null
  }
}

const _ROUTER_TIER_LS = 'basna.routerTier'
const _DEEP_LS = 'basna.deep'
const _QUALITY_LS = 'basna.quality'
const _MAX_PARALLEL_LS = 'basna.maxParallel'
const _EXEC_GROUPS_LS = 'basna.executionGroups'
const _GROUPED_REVIEW_LS = 'basna.groupedReview'
const _SHARED_DS_LS = 'basna.sharedDatastore'

// A VFS project folder as returned by GET /fd/vfs/projects.
export interface VfsProject {
  name: string
  files: number
  bytes: number
  mtime: number
  kind: string       // basna | vatra | council | link | ...
  run_id?: string
  title?: string
}

// Where a run writes: 'new' derives a fresh folder (auto sid-name, or the typed
// custom name), 'existing' reuses a picked VFS folder. Empty → backend auto-name.
export type FolderMode = 'new' | 'existing'

function sanitizeFolderName(name: string): string {
  return (name || '').trim().replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '')
}

// The vfs_project string to send when starting a run. '' → backend auto-names
// (basna-<sid8>/vatra-<sid8>); a non-empty value pins the run to that folder.
function computeVfsProject(s: { folderMode: FolderMode; newFolderName: string; existingFolder: string }): string {
  if (s.folderMode === 'existing') return (s.existingFolder || '').trim()
  return sanitizeFolderName(s.newFolderName)
}

function _loadMaxParallel(): number {
  try {
    const raw = typeof localStorage !== 'undefined' && localStorage.getItem(_MAX_PARALLEL_LS)
    const n = raw ? Number(raw) : NaN
    return Number.isFinite(n) && n >= 0 && n <= 16 ? n : 0
  } catch { return 0 }
}

function _loadGroupedReview(): boolean {
  // Default OFF — the grouped review round adds one dispatch per owner.
  try {
    return typeof localStorage !== 'undefined' && localStorage.getItem(_GROUPED_REVIEW_LS) === '1'
  } catch { return false }
}

function _loadExecGroups(): boolean {
  // Default ON — respect an explicit saved choice, otherwise start grouped.
  try {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(_EXEC_GROUPS_LS) : null
    return raw === null ? true : raw === '1'
  } catch { return true }
}

function _loadSharedDatastore(): boolean {
  // Default ON — respect an explicit saved choice, otherwise start shared.
  try {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(_SHARED_DS_LS) : null
    return raw === null ? true : raw === '1'
  } catch { return true }
}

function _loadQuality(): QualityProfile {
  try {
    const raw = typeof localStorage !== 'undefined' && localStorage.getItem(_QUALITY_LS)
    return raw ? fromResponse(JSON.parse(raw)) : defaultProfile()
  } catch { return defaultProfile() }
}

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
  planning: boolean   // Vatra "Plan as Vatra" step in flight (separate from Basna routing)
  executing: boolean
  recompiling: boolean
  resuming: boolean    // resuming a stalled/cancelled run from its checkpoints
  error: string | null

  routerTier: string   // which Library tier selects the archetypes (the router)
  maxAgents: number
  maxParallel: number  // cap on concurrent agent turns (0 = unlimited; mainly for local models)
  executionGroups: boolean  // Vatra: run owners in ordered phases A→B→C→D (opt-in)
  groupedReview: boolean    // Vatra grouped runs: review round after the final group (opt-in)
  sharedDatastore: boolean  // opt-in: bind the run's agents to ONE datastore in the VFS folder
  folderMode: FolderMode    // 'new' folder for this run, or 'existing' picked folder
  newFolderName: string     // optional custom name when folderMode==='new' (empty → auto)
  existingFolder: string    // picked folder name when folderMode==='existing'
  projects: VfsProject[]    // existing VFS folders for the picker
  projectsLoading: boolean
  knowledgeSessionIds: string[]   // prior finished runs whose knowledge seeds this run
  knowledgeIncludeBoard: boolean  // also fold in the selected runs' shared-board notes
  referenceFolders: string[]      // read-only VFS folders agents check before web search
  deep: boolean        // Deep / Horizon mode: each worker runs the self-consistency
  deepSamples: number  // vote + critics + fix loop (frontier-grade depth) instead of one shot
  planMode: boolean    // Plan-Horizon (Lever C): decompose → verify each step → re-plan
  planSteps: number    // max steps in the plan
  planComplex: boolean // simple = one model per step; complex = a full Basna/Vatra per step
  planDag: boolean     // planner emits a DAG; independent steps run in parallel waves
  quality: QualityProfile  // opt-in cross-pollination levers (all-off == current behaviour)
  activeProjectId: string  // project bundle new runs belong to ('' = Unfiled); sent on route/plan

  setActiveProjectId: (id: string) => void
  setRouterTier: (t: string) => void
  setQuality: (q: QualityProfile) => void
  setMaxAgents: (n: number) => void
  setMaxParallel: (n: number) => void
  setExecutionGroups: (v: boolean) => void
  setGroupedReview: (v: boolean) => void
  setSharedDatastore: (v: boolean) => void
  setFolderMode: (m: FolderMode) => void
  setNewFolderName: (s: string) => void
  setExistingFolder: (s: string) => void
  loadProjects: () => Promise<void>
  setKnowledgeSessionIds: (ids: string[]) => void
  toggleKnowledgeSession: (id: string) => void
  setKnowledgeIncludeBoard: (v: boolean) => void
  toggleReferenceFolder: (name: string) => void
  setDeep: (v: boolean) => void
  setDeepSamples: (n: number) => void
  setPlanMode: (v: boolean) => void
  setPlanSteps: (n: number) => void
  setPlanComplex: (v: boolean) => void
  setPlanDag: (v: boolean) => void
  runPlan: (intent: string, tiers: TierMap, title: string, envVars: EnvVar[], archetypeIds?: string[], stepMode?: string) => Promise<void>
  addFiles: (files: FileList | File[]) => void
  removeFile: (name: string) => Promise<void>
  downloadFile: (name: string) => Promise<void>
  fetchFileText: (name: string) => Promise<string>

  loadSessions: () => Promise<void>
  pollRunning: () => Promise<void>
  selectSession: (id: string) => Promise<void>
  newSession: () => void
  resetDraft: () => void
  updateSelected: (index: number, patch: Partial<RouteSelected>) => void
  updateSubtask: (id: string, patch: Partial<VatraSubtask>) => void
  removeSubtask: (id: string) => void
  setGroupInstruction: (letter: string, text: string) => void
  route: (intent: string, tiers: TierMap, title?: string, archetypeIds?: string[], brief?: string) => Promise<void>
  planVatra: (intent: string, tiers: TierMap, title?: string, archetypeIds?: string[], brief?: string) => Promise<void>
  runVatra: (tiers: TierMap, envVars: EnvVar[]) => Promise<void>
  fillGaps: (id: string, instruction?: string) => Promise<void>
  saveTitle: (title: string) => Promise<void>
  execute: (tiers: TierMap, envVars: EnvVar[]) => Promise<void>
  recompile: (tiers: TierMap) => Promise<void>
  sendFeedback: (runId: number, success: boolean) => Promise<void>
  deleteSession: (id: string) => Promise<void>
  cancelSession: (id: string) => Promise<void>
  // Resume a stalled/cancelled run from its durable checkpoints: finished agents
  // are restored (no re-spend), only the missing ones re-run, then it synthesizes.
  // Basna runs inline (like execute); Vatra backgrounds (like runVatra) and the
  // live monitor picks up its progress.
  resumeSession: (tiers: TierMap, envVars: EnvVar[], vatra: boolean) => Promise<void>
  deepenSession: (id: string, instruction?: string) => Promise<void>
  continueSession: (id: string, opts: { instruction: string; kind: string; sameCast: boolean; vatra: boolean }) => Promise<void>
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
  planning: false,
  executing: false,
  recompiling: false,
  resuming: false,
  error: null,

  routerTier: (typeof localStorage !== 'undefined' && localStorage.getItem(_ROUTER_TIER_LS)) || 'reason',
  maxAgents: 6,
  maxParallel: _loadMaxParallel(),
  executionGroups: _loadExecGroups(),
  groupedReview: _loadGroupedReview(),
  sharedDatastore: _loadSharedDatastore(),
  folderMode: 'new',
  newFolderName: '',
  existingFolder: '',
  projects: [],
  projectsLoading: false,
  knowledgeSessionIds: [],
  knowledgeIncludeBoard: false,
  referenceFolders: [],
  deep: (typeof localStorage !== 'undefined' && localStorage.getItem(_DEEP_LS) === '1') || false,
  deepSamples: 3,
  planMode: false,
  planSteps: 5,
  planComplex: false,
  planDag: false,
  quality: _loadQuality(),
  activeProjectId: '',

  setActiveProjectId: (id) => set({ activeProjectId: id }),
  setRouterTier: (t) => {
    try { localStorage.setItem(_ROUTER_TIER_LS, t) } catch { /* ignore */ }
    set({ routerTier: t })
  },
  setQuality: (q) => {
    try { localStorage.setItem(_QUALITY_LS, JSON.stringify(q)) } catch { /* ignore */ }
    set({ quality: q })
  },
  setMaxAgents: (n) => set({ maxAgents: Math.max(1, Math.min(10, n)) }),
  setMaxParallel: (n) => {
    const v = Math.max(0, Math.min(16, Math.floor(Number.isFinite(n) ? n : 0)))
    try { localStorage.setItem(_MAX_PARALLEL_LS, String(v)) } catch { /* ignore */ }
    set({ maxParallel: v })
  },
  setExecutionGroups: (v) => {
    try { localStorage.setItem(_EXEC_GROUPS_LS, v ? '1' : '0') } catch { /* ignore */ }
    set({ executionGroups: v })
  },
  setGroupedReview: (v) => {
    try { localStorage.setItem(_GROUPED_REVIEW_LS, v ? '1' : '0') } catch { /* ignore */ }
    set({ groupedReview: v })
  },
  setSharedDatastore: (v) => {
    try { localStorage.setItem(_SHARED_DS_LS, v ? '1' : '0') } catch { /* ignore */ }
    set({ sharedDatastore: v })
  },
  setFolderMode: (m) => set({ folderMode: m }),
  setNewFolderName: (s) => set({ newFolderName: s }),
  setExistingFolder: (s) => set({ existingFolder: s }),
  loadProjects: async () => {
    set({ projectsLoading: true })
    const projects = await apiListProjects()
    set({ projects, projectsLoading: false })
  },
  setKnowledgeSessionIds: (ids) => set({ knowledgeSessionIds: ids }),
  toggleKnowledgeSession: (id) => set((st) => ({
    knowledgeSessionIds: st.knowledgeSessionIds.includes(id)
      ? st.knowledgeSessionIds.filter((x) => x !== id)
      : [...st.knowledgeSessionIds, id],
  })),
  setKnowledgeIncludeBoard: (v) => set({ knowledgeIncludeBoard: v }),
  toggleReferenceFolder: (name) => set((st) => ({
    referenceFolders: st.referenceFolders.includes(name)
      ? st.referenceFolders.filter((x) => x !== name)
      : [...st.referenceFolders, name],
  })),
  setDeep: (v) => {
    try { localStorage.setItem(_DEEP_LS, v ? '1' : '0') } catch { /* ignore */ }
    set({ deep: v, ...(v ? { planMode: false } : {}) })  // Deep and Plan are distinct run paths
  },
  setDeepSamples: (n) => set({ deepSamples: Math.max(2, Math.min(8, n)) }),
  setPlanMode: (v) => set({ planMode: v, ...(v ? { deep: false } : {}) }),
  setPlanSteps: (n) => set({ planSteps: Math.max(1, Math.min(12, n)) }),
  setPlanComplex: (v) => set({ planComplex: v }),
  setPlanDag: (v) => set({ planDag: v }),

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

  // Live monitor: while runs execute in the background (incl. agent-started
  // ones), refresh the list status and the open session's progress/result.
  // No-op during a manual execute, which drives its own polling.
  pollRunning: async () => {
    if (get().executing) return
    try { set({ sessions: await apiListSessions() }) } catch { /* ignore */ }
    const a = get().activeSession
    if (a && ['routing', 'routed', 'running'].includes(a.status)) {
      try {
        const p = await apiProgress(a.id)
        if (p.events?.length) set({ progress: p.events })
      } catch { /* ignore */ }
      const fresh = await apiGetSession(a.id).catch(() => null)
      if (fresh && fresh.status !== a.status) {
        set({ activeSession: fresh, routePlan: parseRoute(fresh.route),
              runs: await apiListRuns(fresh.id).catch(() => []),
              attachments: parseFiles(fresh.files) })
        if (fresh.status === 'done' || fresh.status === 'error') {
          set({ progress: parseProgress(fresh.progress) })
          // The terminal `cost` event is persisted a beat AFTER status flips to done
          // (post-completion learning + cost summary), and this run-monitor interval
          // tears down once nothing is running — so self-schedule a few re-fetches
          // until the cost event lands, or the cost card never appears without a reopen.
          if (!parseProgress(fresh.progress).some((e) => e.stage === 'cost')) {
            let tries = 0
            const grabCost = async () => {
              tries += 1
              const s2 = await apiGetSession(a.id).catch(() => null)
              const evs = s2 ? parseProgress(s2.progress) : []
              if (evs.some((e) => e.stage === 'cost')) {
                set((st) => (st.activeSession?.id === a.id ? { progress: evs } : {}))
              } else if (tries < 6) {
                setTimeout(grabCost, 3000)
              }
            }
            setTimeout(grabCost, 2500)
          }
        }
      }
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

  // Full draft reset — used when switching projects so one project's plan,
  // selected run, reference folders and prior-knowledge picks never bleed into
  // another. Clears everything newSession does PLUS the per-run setup selections.
  resetDraft: () => set({
    activeSession: null, routePlan: null, runs: [], lastExecute: null,
    progress: [], attachments: [], error: null,
    referenceFolders: [], knowledgeSessionIds: [], knowledgeIncludeBoard: false,
  }),

  updateSelected: (index, patch) => {
    const plan = get().routePlan
    if (!plan) return
    const selected = plan.selected.map((s, i) => (i === index ? { ...s, ...patch } : s))
    set({ routePlan: { ...plan, selected } })
  },

  // Vatra team-plan edits (before Run). Persisted to the session by runVatra via
  // apiSaveRoute, so execute_vatra honors the edited briefs / groups / instructions.
  updateSubtask: (id, patch) => {
    const plan = get().routePlan
    if (!plan?.subtasks) return
    set({ routePlan: { ...plan, subtasks: plan.subtasks.map((s) => (s.id === id ? { ...s, ...patch } : s)) } })
  },
  removeSubtask: (id) => {
    const plan = get().routePlan
    if (!plan?.subtasks) return
    set({ routePlan: { ...plan, subtasks: plan.subtasks.filter((s) => s.id !== id) } })
  },
  setGroupInstruction: (letter, text) => {
    const plan = get().routePlan
    if (!plan) return
    const gi = { ...(plan.group_instructions || {}), [letter]: text }
    if (!text.trim()) delete gi[letter]
    set({ routePlan: { ...plan, group_instructions: gi } })
  },

  route: async (intent, tiers, title = '', archetypeIds = [], brief = '') => {
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
        ...(title.trim() ? { title: title.trim() } : {}),
        ...(archetypeIds.length ? { archetype_ids: archetypeIds } : {}),
        // R12: opt-in intent brief. A user-edited brief re-routes the team on it.
        quality: toRequest(get().quality),
        ...(brief.trim() ? { brief } : {}),
        ...(get().activeProjectId ? { project_id: get().activeProjectId } : {}),
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

  // Vatra prepare step (mirrors Basna's Route): the Lead decomposes the task into
  // owned pieces, persisted as a routed session — nothing is spawned yet. The team
  // plan then shows in the collaboration panel for review before Run.
  planVatra: async (intent, tiers, title = '', archetypeIds = [], brief = '') => {
    if (!intent.trim()) return
    set({ planning: true, error: null })
    try {
      const { session_id } = await apiVatraRoute({
        intent: intent.trim(), max_agents: get().maxAgents, tiers,
        router_tier: get().routerTier,
        ...(title.trim() ? { title: title.trim() } : {}),
        ...(archetypeIds.length ? { archetype_ids: archetypeIds } : {}),
        // R12: opt-in intent brief. A user-edited brief re-plans the team on it.
        quality: toRequest(get().quality),
        // Plan-time datastore awareness: the Lead must plan for the shared store and
        // be seeded with what the chosen folder already holds (continue vs restart).
        shared_datastore: get().sharedDatastore,
        ...((): Record<string, unknown> => { const p = computeVfsProject(get()); return p ? { vfs_project: p } : {} })(),
        ...(get().knowledgeSessionIds.length ? { knowledge_session_ids: get().knowledgeSessionIds, knowledge_include_board: get().knowledgeIncludeBoard } : {}),
        ...(get().referenceFolders.length ? { reference_folders: get().referenceFolders } : {}),
        ...(brief.trim() ? { brief } : {}),
        ...(get().activeProjectId ? { project_id: get().activeProjectId } : {}),
      })
      // Persist pending attachments onto the freshly-created session BEFORE
      // selectSession reloads its files — otherwise Plan drops them (the new session
      // has none yet) and the run never sees the attachments. The run reads the
      // session's files for upload + VFS save + the file-aware brief.
      const pending = get().attachments.filter((a) => !a.uploaded && a.file)
      if (pending.length) {
        try { await apiUploadFiles(session_id, pending.map((a) => a.file as File)) }
        catch (e) { set({ error: e instanceof Error ? e.message : 'file upload failed' }) }
      }
      await get().loadSessions()
      await get().selectSession(session_id)
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'vatra plan failed' })
    } finally {
      set({ planning: false })
    }
  },

  // Vatra run step: spawn + run the prepared session in the background; pollRunning
  // then drives progress and loads the result on done (no blocking request).
  runVatra: async (tiers, envVars) => {
    const sid = get().activeSession?.id
    if (!sid) return
    set({ error: null })
    // Upload any attachments not yet on the server before the run — Basna's execute
    // does this; Vatra must too, or the run never sees the attached files.
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
    // Persist any team-plan edits (briefs, groups, per-group instructions) before Run.
    if (get().routePlan) {
      try { await apiSaveRoute(sid, get().routePlan as RoutePlan) } catch { /* ignore */ }
    }
    try {
      const env_vars = (envVars || []).filter((e) => e.key.trim() && e.value.trim())
      // Deep mode in Vatra = Horizon depth: verify + revise EACH specialist's slice
      // (worker, blackboard-safe — no spawn pools) AND the final assembled deliverable.
      const horizon = get().deep ? { worker: true, close: true } : undefined
      const vfsProject = computeVfsProject(get())
      await apiVatraExecute({ session_id: sid, tiers, env_vars, quality: toRequest(get().quality), max_parallel: get().maxParallel, execution_groups: get().executionGroups, grouped_review: get().executionGroups && get().groupedReview, shared_datastore: get().sharedDatastore, ...(vfsProject ? { vfs_project: vfsProject } : {}), ...(horizon ? { horizon } : {}) })
      const s = await apiGetSession(sid)
      if (s) set({ activeSession: s })
      await get().loadSessions()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'vatra run failed' })
    }
  },

  runPlan: async (intent, tiers, title, envVars, archetypeIds, stepMode) => {
    if (!intent.trim()) return
    set({ error: null })
    try {
      const env_vars = (envVars || []).filter((e) => e.key.trim() && e.value.trim())
      // Plan-Horizon: decompose → verify each step → re-plan → synthesize. Creates a
      // fresh session that runs in the background; open it so the live log polls.
      // step_mode: 'llm' (simple) | 'ensemble' | 'vatra' (complex, per the selected mode).
      // A fixed team (archetype_ids) staffs each step's ensemble / Vatra team.
      const r = await apiPlan({
        intent, title: title || '', tiers, env_vars,
        max_steps: get().planSteps,
        step_mode: stepMode || 'llm',
        dag: get().planDag,
        ...((archetypeIds && archetypeIds.length) ? { archetype_ids: archetypeIds } : {}),
      })
      await get().loadSessions()
      if (r.session_id) await get().selectSession(r.session_id)
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'plan run failed' })
    }
  },

  saveTitle: async (title) => {
    const sid = get().activeSession?.id
    if (!sid) return
    const updated = await apiSaveTitle(sid, title.trim())
    if (updated) {
      set((st) => ({
        activeSession: st.activeSession?.id === sid ? updated : st.activeSession,
        sessions: st.sessions.map((s) => (s.id === sid ? { ...s, title: updated.title } : s)),
      }))
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
      // Deep / Horizon mode: drive each worker through the self-consistency vote +
      // critics + fix loop, then verify-and-revise the merged answer (the closer).
      const horizon = get().deep ? { samples: get().deepSamples, close: true } : undefined
      const vfsProject = computeVfsProject(get())
      const res = await apiExecute({ session_id: sid, tiers, env_vars, quality: toRequest(get().quality), max_parallel: get().maxParallel, shared_datastore: get().sharedDatastore, ...(vfsProject ? { vfs_project: vfsProject } : {}), ...(get().knowledgeSessionIds.length ? { knowledge_session_ids: get().knowledgeSessionIds, knowledge_include_board: get().knowledgeIncludeBoard } : {}), ...(get().referenceFolders.length ? { reference_folders: get().referenceFolders } : {}), ...(horizon ? { horizon } : {}) })
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

  recompile: async (tiers) => {
    const sid = get().activeSession?.id
    if (!sid) return
    set({ recompiling: true, error: null })
    try {
      const res = await apiRecompile(sid, tiers)
      const s = await apiGetSession(sid)
      set({
        activeSession: s,
        attachments: s ? parseFiles(s.files) : get().attachments,
        lastExecute: { ...(get().lastExecute || {} as ExecuteResult), ...res } as ExecuteResult,
      })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'recompile failed' })
    } finally {
      set({ recompiling: false })
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

  cancelSession: async (id) => {
    await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(id)}/cancel`, { method: 'POST' })
    await get().loadSessions()
  },

  resumeSession: async (tiers, envVars, vatra) => {
    const sid = get().activeSession?.id
    if (!sid) return
    const env_vars = (envVars || []).filter((e) => e.key.trim() && e.value.trim())
    set({ resuming: true, error: null })
    if (vatra) {
      // Vatra backgrounds the run; flip to the live view and let pollRunning drive it.
      try {
        await apiVatraResume(sid, { session_id: sid, tiers, env_vars })
        const s = await apiGetSession(sid)
        if (s) set({ activeSession: s })
        await get().loadSessions()
      } catch (e) {
        set({ error: e instanceof Error ? e.message : 'resume failed' })
      } finally {
        set({ resuming: false })
      }
      return
    }
    // Basna runs inline (like execute): poll the live log while it blocks, then
    // refresh the session, runs, and generated files.
    set({ executing: true, progress: [] })
    const poll = setInterval(async () => {
      try { const p = await apiProgress(sid); set({ progress: p.events || [] }) } catch { /* ignore */ }
    }, 700)
    try {
      const res = await apiBasnaResume(sid, { session_id: sid, tiers, env_vars })
      const s = await apiGetSession(sid)
      const runs = await apiListRuns(sid)
      set({ lastExecute: res, activeSession: s, runs,
            ...(s ? { attachments: parseFiles(s.files) } : {}) })
      await get().loadSessions()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'resume failed' })
    } finally {
      clearInterval(poll)
      try { const p = await apiProgress(sid); set({ progress: p.events || [] }) } catch { /* ignore */ }
      set({ executing: false, resuming: false })
    }
  },

  deepenSession: async (id, instruction = '') => {
    const res = await _authedFetch(`/fd/basna/sessions/${encodeURIComponent(id)}/deepen`, {
      method: 'POST', body: JSON.stringify({ instruction }),
    })
    if (!res.ok) throw new Error((await res.text()) || 'deepen failed')
    const data = await res.json()
    await get().loadSessions()
    if (data.session_id) await get().selectSession(data.session_id)
  },

  // Vatra analog of deepen: a follow-up run that fills this run's coverage gaps,
  // seeded with its final report.
  fillGaps: async (id, instruction = '') => {
    const res = await _authedFetch(`/fd/vatra/sessions/${encodeURIComponent(id)}/fill-gaps`, {
      method: 'POST', body: JSON.stringify({ instruction }),
    })
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      throw new Error((detail as { detail?: string }).detail || 'fill gaps failed')
    }
    const data = await res.json()
    await get().loadSessions()
    if (data.session_id) await get().selectSession(data.session_id)
  },

  // Carry a finished run forward into another round — same VFS folder + conclusion.
  // Routes to the Basna or Vatra continue endpoint by mode; on success navigates to
  // the new round's session (same flow as deepen/fillGaps).
  continueSession: async (id, opts) => {
    const base = opts.vatra ? 'vatra' : 'basna'
    const res = await _authedFetch(`/fd/${base}/sessions/${encodeURIComponent(id)}/continue`, {
      method: 'POST',
      body: JSON.stringify({ instruction: opts.instruction, kind: opts.kind, same_cast: opts.sameCast }),
    })
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}))
      throw new Error((detail as { detail?: string }).detail || 'continue failed')
    }
    const data = await res.json()
    await get().loadSessions()
    if (data.session_id) await get().selectSession(data.session_id)
  },
}))
