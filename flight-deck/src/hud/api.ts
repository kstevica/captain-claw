// HUD data layer: every call goes through Flight Deck's JWT-guarded /fd/*
// routes (owner-checked agent proxies, member routes for shared agents).
// Never the legacy /glasses/* bridge — it has no user auth and can reach any
// tenant's agent. Agents are always addressed as localhost:<web_port> taken
// from the owner-filtered /fd/processes and /fd/containers lists; the FD
// proxy resolves the agent's own token server-side, so no agent secret ever
// sits in a URL or in this page.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { TABLE_NAME_RE } from './router'

// ── Types ──

export type AgentKind = 'process' | 'container' | 'shared'

export interface HudAgent {
  /** Chat id convention shared with the dashboard: proc-<slug> | <container id> | shared:<agent_ref> */
  id: string
  kind: AgentKind
  name: string
  description: string
  running: boolean
  /** Web port for own agents (null for shared / not running). */
  port: number | null
  /** agent_ref for shared agents. */
  ref: string | null
  /** Owner's display name for shared agents. */
  ownerName: string | null
  model: string
  caps: { chat: boolean; files: boolean; data: boolean }
}

export interface HudFile {
  /** Physical path (own agents) or file id (shared agents) — what the view route needs. */
  key: string
  name: string
  /** Workspace-relative path for display (never the absolute host path). */
  logical: string
  size: number
  /** Epoch ms. */
  modified: number
  /** False when someone other than the agent's owner (or me, on a shared agent) created it. */
  trusted: boolean
  /** Creator's name when not trusted. */
  creator: string | null
}

export interface HudColumn { name: string; type: string; position?: number }

export interface HudTable {
  name: string
  columns: HudColumn[]
  rowCount: number
  /** ISO string or null. */
  updatedAt: string | null
}

export interface HudRows {
  columns: string[]
  rows: Record<string, unknown>[]
  total: number
  offset: number
  limit: number
}

export interface HudUser { id: string; email: string; display_name: string; role: string }

export class HudApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

// ── Fetch with auth + one refresh ──

const TIMEOUT_MS = 20_000

function authHeader(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  return authEnabled && token ? { Authorization: `Bearer ${token}` } : {}
}

/** Pull a readable message out of FastAPI / agent error bodies (detail may be
 *  a JSON *string* relayed from the agent, e.g. '{"error": "Table not found"}'). */
export function errorMessage(body: unknown, status: number): string {
  const pick = (v: unknown): string | null => {
    if (typeof v === 'string') {
      const t = v.trim()
      if (t.startsWith('{')) {
        try { return pick(JSON.parse(t)) } catch { /* plain text */ }
      }
      return t || null
    }
    if (v && typeof v === 'object') {
      const o = v as Record<string, unknown>
      return pick(o.detail) ?? pick(o.error) ?? pick(o.message)
    }
    return null
  }
  return pick(body) ?? `Request failed (${status})`
}

/**
 * fetch() against Flight Deck with the access token. A 401 triggers ONE
 * single-flight refresh (refreshAccessToken — never call /fd/auth/refresh
 * directly: two concurrent rotations log the device out) and a retry; a failed
 * refresh ends the session (the app shows the pairing screen).
 */
export async function hudFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const st = useAuthStore.getState()
  if (st.authEnabled === true && !st.token) {
    if (!(await refreshAccessToken())) {
      useAuthStore.getState().clearAuth()
      throw new HudApiError('Not signed in', 401)
    }
  }
  const run = () => fetch(path, {
    ...init,
    credentials: 'include',
    signal: init.signal ?? AbortSignal.timeout(TIMEOUT_MS),
    headers: { ...authHeader(), ...(init.headers as Record<string, string> | undefined) },
  })
  let res: Response
  try {
    res = await run()
    if (res.status === 401 && useAuthStore.getState().authEnabled) {
      if (!(await refreshAccessToken())) {
        useAuthStore.getState().clearAuth()
        throw new HudApiError('Session expired', 401)
      }
      res = await run()
    }
  } catch (e) {
    if (e instanceof HudApiError) throw e
    const name = (e as { name?: string })?.name
    if (name === 'TimeoutError' || name === 'AbortError') throw new HudApiError('Timed out', 0)
    throw new HudApiError('Network error', 0)
  }
  return res
}

export async function hudJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await hudFetch(path, init)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new HudApiError(errorMessage(body, res.status), res.status)
  }
  return res.json() as Promise<T>
}

function postJSON(body: unknown): RequestInit {
  return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

// ── Agents ──

interface RawProcess { slug: string; name: string; description?: string; status: string; web_port: number; model?: string }
interface RawContainer { id: string; name: string; agent_name?: string; description?: string; status: string; web_port: number | null }
interface RawShared {
  agent_ref: string; name: string; description?: string; status: string; owner_name?: string
  capabilities?: { files?: boolean; datastore?: boolean } | null
}

const RANK = (a: HudAgent) => (a.running ? 0 : 1)

/** The signed-in user's own agents (owner-filtered by FD) plus agents shared with them. */
export async function listAgents(): Promise<HudAgent[]> {
  const [procs, conts, shared] = await Promise.allSettled([
    hudJSON<RawProcess[]>('/fd/processes'),
    hudJSON<RawContainer[]>('/fd/containers'),
    hudJSON<{ enabled?: boolean; agents?: RawShared[] }>('/fd/shared-agents'),
  ])
  // Auth failures must surface (the app falls back to pairing), not read as "no agents".
  for (const r of [procs, conts]) {
    if (r.status === 'rejected' && r.reason instanceof HudApiError && r.reason.status === 401) throw r.reason
  }
  if (procs.status === 'rejected' && conts.status === 'rejected') throw procs.reason
  const out: HudAgent[] = []
  if (procs.status === 'fulfilled' && Array.isArray(procs.value)) {
    for (const p of procs.value) {
      const running = p.status === 'running' && !!p.web_port
      out.push({
        id: `proc-${p.slug}`, kind: 'process', name: p.name || p.slug, description: p.description || '',
        running, port: running ? p.web_port : null, ref: null, ownerName: null, model: p.model || '',
        caps: { chat: true, files: true, data: true },
      })
    }
  }
  if (conts.status === 'fulfilled' && Array.isArray(conts.value)) {
    for (const c of conts.value) {
      const running = c.status === 'running' && !!c.web_port
      out.push({
        id: c.id, kind: 'container', name: c.agent_name || c.name, description: c.description || '',
        running, port: running ? c.web_port : null, ref: null, ownerName: null, model: '',
        caps: { chat: true, files: true, data: true },
      })
    }
  }
  if (shared.status === 'fulfilled' && shared.value?.enabled !== false && Array.isArray(shared.value?.agents)) {
    for (const s of shared.value.agents) {
      out.push({
        id: `shared:${s.agent_ref}`, kind: 'shared', name: s.name, description: s.description || '',
        running: s.status === 'running', port: null, ref: s.agent_ref, ownerName: s.owner_name || null, model: '',
        caps: { chat: true, files: s.capabilities?.files === true, data: s.capabilities?.datastore === true },
      })
    }
  }
  out.sort((a, b) => RANK(a) - RANK(b) || a.name.localeCompare(b.name))
  return out
}

function ownBase(agent: HudAgent, prefix: string): string {
  if (agent.kind === 'shared' || !agent.port) throw new HudApiError('Agent is not running', 409)
  return `/fd/${prefix}/localhost/${agent.port}`
}

// ── Files (markdown) ──

const MD_EXT = new Set(['.md', '.markdown'])
/** Refuse to pull bigger files over the glasses' ~500 Kbps link. */
export const MAX_FILE_BYTES = 1024 * 1024

function toMs(t: unknown): number {
  const n = Number(t)
  if (!Number.isFinite(n) || n <= 0) return 0
  return n > 1e12 ? n : n * 1000
}

interface RawAgentFile {
  logical: string; physical: string; filename: string; extension: string; exists: boolean
  size: number; modified: number; created_by?: { kind?: string; name?: string } | null
}
interface RawSharedFile {
  id: string; path: string; filename: string; extension: string; size: number; modified: number
  created_by?: { kind?: string; name?: string } | null
}

/** Markdown files of an agent, newest first. */
export async function listMarkdownFiles(agent: HudAgent): Promise<HudFile[]> {
  let files: HudFile[]
  if (agent.kind === 'shared') {
    if (!agent.ref || !agent.caps.files) return []
    const res = await hudJSON<{ files?: RawSharedFile[] }>(`/fd/shared-agents/files?ref=${encodeURIComponent(agent.ref)}`)
    files = (res.files || [])
      .filter((f) => MD_EXT.has((f.extension || '').toLowerCase()))
      .map((f) => ({
        key: f.id, name: f.filename, logical: f.path || f.filename, size: f.size || 0, modified: toMs(f.modified),
        // On someone else's agent only my own files are trusted (FileViewer rule).
        trusted: f.created_by?.kind === 'me',
        creator: f.created_by?.kind === 'me' ? null : (f.created_by?.name || 'another member'),
      }))
  } else {
    const raw = await hudJSON<RawAgentFile[]>(ownBase(agent, 'agent-files'))
    // De-dupe: FD merges registry + workspace scan, which can repeat a path.
    const seen = new Set<string>()
    files = []
    for (const f of Array.isArray(raw) ? raw : []) {
      if (!f.exists || !MD_EXT.has((f.extension || '').toLowerCase()) || seen.has(f.physical)) continue
      // Workspace-root files outside saved/ output/ workflows/ can't be opened (agent sandbox).
      if (f.logical.startsWith('workspace/') && !/^workspace\/(saved|output|workflows)\//.test(f.logical)) continue
      seen.add(f.physical)
      // Fail closed: anything not created by the owner (or unattributed) is a member file.
      const trusted = !f.created_by || f.created_by.kind === 'owner'
      files.push({
        key: f.physical, name: f.filename, logical: f.logical, size: f.size || 0, modified: toMs(f.modified),
        trusted, creator: trusted ? null : (f.created_by?.name || 'a member'),
      })
    }
  }
  files.sort((a, b) => b.modified - a.modified)
  return files
}

/** Raw text of a file. `size` (if known) is checked against MAX_FILE_BYTES first. */
export async function readFileText(agent: HudAgent, key: string, size?: number): Promise<string> {
  if (size && size > MAX_FILE_BYTES) throw new HudApiError('File too large for the glasses', 413)
  const path = agent.kind === 'shared'
    ? `/fd/shared-agents/files/view?ref=${encodeURIComponent(agent.ref || '')}&id=${encodeURIComponent(key)}`
    : `${ownBase(agent, 'agent-file-view')}?path=${encodeURIComponent(key)}`
  const res = await hudFetch(path)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new HudApiError(errorMessage(body, res.status), res.status)
  }
  const len = Number(res.headers.get('content-length') || 0)
  if (len > MAX_FILE_BYTES) throw new HudApiError('File too large for the glasses', 413)
  return res.text()
}

// ── Datastore (read-only) ──

interface RawTable {
  name: string; columns?: HudColumn[]; row_count?: number; updated_at?: string | null
}

export async function listTables(agent: HudAgent): Promise<HudTable[]> {
  let raw: RawTable[]
  if (agent.kind === 'shared') {
    if (!agent.ref || !agent.caps.data) return []
    raw = await hudJSON<RawTable[]>(`/fd/shared-agents/datastore/tables?ref=${encodeURIComponent(agent.ref)}`)
  } else {
    raw = await hudJSON<RawTable[]>(`${ownBase(agent, 'agent-datastore')}/tables`)
  }
  return (Array.isArray(raw) ? raw : []).map((t) => ({
    name: t.name,
    columns: (t.columns || []).slice().sort((a, b) => (a.position ?? 0) - (b.position ?? 0)),
    rowCount: Number(t.row_count) || 0,
    updatedAt: t.updated_at || null,
  }))
}

const ORDER_BY_RE = /^_?[a-z0-9_]{1,128}$/

export interface RowQuery { limit: number; offset: number; orderBy?: string; orderDir?: 'asc' | 'desc' }

export async function queryRows(agent: HudAgent, table: string, q: RowQuery): Promise<HudRows> {
  if (!TABLE_NAME_RE.test(table)) throw new HudApiError('Bad table name', 400)
  const params = new URLSearchParams()
  params.set('limit', String(Math.max(1, Math.min(500, Math.floor(q.limit)))))
  params.set('offset', String(Math.max(0, Math.floor(q.offset))))
  if (q.orderBy) {
    if (!ORDER_BY_RE.test(q.orderBy)) throw new HudApiError('Bad column name', 400)
    params.set('order_by', q.orderBy)
    params.set('order_dir', q.orderDir === 'desc' ? 'desc' : 'asc')
  }
  let path: string
  if (agent.kind === 'shared') {
    params.set('ref', agent.ref || '')
    path = `/fd/shared-agents/datastore/tables/${table}/rows?${params}`
  } else {
    path = `${ownBase(agent, 'agent-datastore')}/tables/${table}/rows?${params}`
  }
  const r = await hudJSON<HudRows>(path)
  return {
    columns: Array.isArray(r.columns) ? r.columns : [],
    rows: Array.isArray(r.rows) ? r.rows : [],
    total: Number(r.total) || 0,
    offset: Number(r.offset) || 0,
    limit: Number(r.limit) || q.limit,
  }
}

// ── HUD config ──

export interface HudConfig {
  /** The glasses rendering rules block, prepended to the first chat message of
   *  a connection (the agent stores it as the session's surface rules). */
  surface_rules: string
}

let configPromise: Promise<HudConfig> | null = null

export function getHudConfig(): Promise<HudConfig> {
  if (!configPromise) {
    configPromise = hudJSON<HudConfig>('/fd/hud/config').catch((e) => {
      configPromise = null
      throw e
    })
  }
  return configPromise
}

// ── Device pairing (RFC 8628-style; see auth_routes.py /fd/auth/pair/*) ──

export interface PairStart {
  /** Secret the glasses poll with — never shown. */
  device_code: string
  /** Short code the wearer reads and types on a signed-in phone/desktop. */
  user_code: string
  expires_in: number
  interval: number
  /** Where to approve, e.g. "/hud/pair". */
  verification_path: string
}

export type PairPoll =
  | { status: 'pending' | 'denied' | 'expired' }
  | { status: 'approved'; access_token: string; user: HudUser }

export interface PairInfo {
  user_code: string
  label: string
  user_agent: string
  ip: string
  created_at: string
  expires_in: number
}

/** Public: start a pairing for this device. */
export async function pairStart(label: string): Promise<PairStart> {
  const res = await fetch('/fd/auth/pair/start', { ...postJSON({ label }), credentials: 'include' })
  if (!res.ok) throw new HudApiError(errorMessage(await res.json().catch(() => null), res.status), res.status)
  return res.json()
}

/** Public: the glasses ask whether the code was approved. On approval the
 *  response also sets the refresh cookie, like a normal login. */
export async function pairPoll(deviceCode: string): Promise<PairPoll> {
  const res = await fetch('/fd/auth/pair/poll', { ...postJSON({ device_code: deviceCode }), credentials: 'include' })
  if (!res.ok) throw new HudApiError(errorMessage(await res.json().catch(() => null), res.status), res.status)
  return res.json()
}

/** Signed in (phone/desktop): which device is asking for this code? */
export function pairLookup(code: string): Promise<PairInfo> {
  return hudJSON<PairInfo>(`/fd/auth/pair/lookup?code=${encodeURIComponent(code)}`)
}

/** Signed in (phone/desktop): approve or deny the code. */
export function pairApprove(code: string, approve: boolean): Promise<{ ok: boolean; status: 'approved' | 'denied' }> {
  return hudJSON(`/fd/auth/pair/approve`, postJSON({ user_code: code, approve }))
}
