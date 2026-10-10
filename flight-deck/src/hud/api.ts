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

/** Response headers must arrive within this… */
const HEADERS_TIMEOUT_MS = 20_000
/** …then a (small JSON) body gets this long. */
const BODY_TIMEOUT_MS = 20_000
/** A file's body: READ_LIMIT_CHARS takes ~4 s at the glasses' ~500 Kbps; this
 *  leaves room for a much slower link instead of cutting a download that is
 *  still coming in. */
const FILE_BODY_TIMEOUT_MS = 90_000
/** Public pairing calls (before sign-in): a stalled one must become a retry,
 *  not a "Getting a sign-in code…" that never ends. */
const PAIR_TIMEOUT_MS = 15_000
/** "Is Flight Deck there at all?" — asked after a refresh failed. */
const PROBE_TIMEOUT_MS = 5_000

/**
 * fetch() whose response HEADERS must arrive within `headersMs`; the body then
 * has its own `bodyMs`, so a slow but progressing download isn't cut at the
 * headers' limit. Either limit aborts the request: fetch() or the body read
 * rejects with an AbortError (see transportError). The body timer is left to
 * run out — aborting a response that was read to the end does nothing.
 */
function timedFetch(path: string, init: RequestInit, headersMs: number, bodyMs: number): Promise<Response> {
  const ctl = new AbortController()
  const outer = init.signal
  if (outer?.aborted) ctl.abort()
  else outer?.addEventListener('abort', () => ctl.abort(), { once: true })
  const timer = setTimeout(() => ctl.abort(), headersMs)
  return fetch(path, { ...init, signal: ctl.signal }).then(
    (res) => {
      clearTimeout(timer)
      setTimeout(() => ctl.abort(), bodyMs)
      return res
    },
    (e: unknown) => {
      clearTimeout(timer)
      throw e
    },
  )
}

/** A failed fetch() / body read as the HudApiError the screens understand
 *  (never a raw "The user aborted a request."). */
function transportError(e: unknown): HudApiError {
  if (e instanceof HudApiError) return e
  const name = (e as { name?: string } | null)?.name
  if (name === 'TimeoutError' || name === 'AbortError') return new HudApiError('Timed out', 0)
  return new HudApiError('Network error', 0)
}

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

/** Flight Deck's own auth failures: auth.py (get_current_user,
 *  decode_access_token) and server.py's agent-proxy ownership guard. */
const FD_AUTH_DETAILS = new Set(['Not authenticated', 'Token expired', 'Invalid token', 'Invalid token type', 'User not found'])

function isFdAuthDetail(body: unknown): boolean {
  const d = (body as { detail?: unknown } | null)?.detail
  return typeof d === 'string' && FD_AUTH_DETAILS.has(d)
}

/** The access token is past its `exp` (false when it can't be read). */
function tokenExpired(token: string): boolean {
  try {
    const p = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/'))) as { exp?: unknown }
    return typeof p.exp === 'number' && p.exp * 1000 <= Date.now()
  } catch {
    return false
  }
}

/**
 * Is this 401 Flight Deck turning down the session (→ refresh), or the agent
 * behind an FD proxy turning down FD? The agent proxies relay the agent's own
 * 401 with its body as `detail` ('{"error": "unauthorized"}', 'Agent returned
 * 401'); refreshing for that only rotates the refresh cookie. FD sends no
 * WWW-Authenticate, so its detail text is the tell — and a token past its
 * `exp` is FD's business either way.
 */
async function isFdAuthFailure(res: Response, sentToken: string): Promise<boolean> {
  if (tokenExpired(sentToken)) return true
  return isFdAuthDetail(await res.clone().json().catch(() => null))
}

/** The HudApiError for a non-OK response (reads its body). */
async function responseError(res: Response): Promise<HudApiError> {
  const body = await res.json().catch(() => null)
  // hudFetch already renewed the session for FD's own 401s: any other 401 is
  // the agent refusing FD's stored token — not the wearer's sign-in.
  if (res.status === 401 && !isFdAuthDetail(body)) return new HudApiError("The agent refused Flight Deck's access.", 502)
  return new HudApiError(errorMessage(body, res.status), res.status)
}

/** refreshAccessToken(), or null when it hasn't answered in time: a stalled
 *  refresh must not hang every request queued behind it. */
function refreshWithin(): Promise<boolean | null> {
  let timer: ReturnType<typeof setTimeout> | undefined
  return Promise.race([
    refreshAccessToken(),
    new Promise<null>((resolve) => { timer = setTimeout(() => resolve(null), HEADERS_TIMEOUT_MS) }),
  ]).finally(() => clearTimeout(timer))
}

/** Flight Deck gives a real answer to the public status call. */
async function deckReachable(): Promise<boolean> {
  try {
    const res = await fetch('/fd/auth/status', { cache: 'no-store', signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) })
    if (!res.ok) return false
    const data = (await res.json()) as { auth_enabled?: unknown } | null
    return typeof data?.auth_enabled === 'boolean'
  } catch {
    return false
  }
}

type Renewal = 'ok' | 'offline' | 'ended'
let renewal: Promise<Renewal> | null = null

/**
 * Renew the access token (single-flight), telling a session that is really
 * over from a link that is down. refreshAccessToken() — never call
 * /fd/auth/refresh directly: two concurrent rotations log the device out —
 * answers false both for a rejected cookie and for a refresh that got no real
 * answer (network error, a proxy's 5xx). So on false: Flight Deck unreachable
 * → keep the session and fail this request as a network error the screen can
 * retry; reachable → one more refresh, and only if that fails too is the
 * session over (clearAuth: the app shows the pairing screen).
 */
async function renewSession(endMessage: string): Promise<void> {
  renewal ??= (async (): Promise<Renewal> => {
    let ok = await refreshWithin()
    if (ok === false) {
      if (!(await deckReachable())) return 'offline'
      ok = await refreshWithin()
    }
    if (ok === null) return 'offline'
    if (ok) return 'ok'
    useAuthStore.getState().clearAuth()
    return 'ended'
  })().finally(() => { renewal = null })
  const r = await renewal
  if (r === 'offline') throw new HudApiError('Network error', 0)
  if (r === 'ended') throw new HudApiError(endMessage, 401)
}

/**
 * Boot: restore the session from the refresh cookie. 'offline' — Flight Deck
 * gave no real answer (retry; never drop the wearer to pairing over a blip);
 * 'ended' — there is no session (show pairing).
 */
export async function restoreSession(): Promise<Renewal> {
  // One refresh, not renewSession's two: at boot there is nothing to keep, and
  // a first launch (no cookie) would otherwise pay two refreshes + a probe.
  const ok = await refreshWithin()
  if (ok) return 'ok'
  if (ok === null || !(await deckReachable())) return 'offline'
  return 'ended'
}

/**
 * fetch() against Flight Deck with the access token. Flight Deck's own 401
 * renews the session ONCE (renewSession) and retries; a 401 an agent proxy
 * relays from the agent comes back as is (see isFdAuthFailure). `bodyMs`: how
 * long the body may take once the headers are in.
 */
export async function hudFetch(path: string, init: RequestInit = {}, bodyMs = BODY_TIMEOUT_MS): Promise<Response> {
  const st = useAuthStore.getState()
  if (st.authEnabled === true && !st.token) await renewSession('Not signed in')
  const run = () => timedFetch(path, {
    ...init,
    credentials: 'include',
    headers: { ...authHeader(), ...(init.headers as Record<string, string> | undefined) },
  }, HEADERS_TIMEOUT_MS, bodyMs)
  try {
    const sent = useAuthStore.getState().token
    let res = await run()
    if (res.status === 401 && useAuthStore.getState().authEnabled && await isFdAuthFailure(res, sent)) {
      // Another request may have renewed the token meanwhile: just use that.
      if (useAuthStore.getState().token === sent) await renewSession('Session expired')
      res = await run()
    }
    return res
  } catch (e) {
    throw transportError(e)
  }
}

export async function hudJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await hudFetch(path, init)
  if (!res.ok) throw await responseError(res)
  try {
    return (await res.json()) as T
  } catch (e) {
    throw transportError(e)
  }
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
  // Nor may a failed process list (a tunnel 502, a timeout on the slow link):
  // that is where most agents live. Only a deck without the route (404) goes
  // on with containers alone. Containers / shared stay optional (FD answers
  // [] without Docker; sharing can be off).
  if (procs.status === 'rejected') {
    const noRoute = procs.reason instanceof HudApiError && procs.reason.status === 404
    if (!noRoute) throw procs.reason
    if (conts.status === 'rejected') throw conts.reason
  }
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
/** readFileText stops the download once it has more text than this: the HUD
 *  renders only the first 200 000 chars (FileScreen RENDER_LIMIT, which must
 *  stay below this so a cut file still reads as clipped). */
export const READ_LIMIT_CHARS = 250_000

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
      // Unattributed files are the agent's own; any attribution other than the
      // owner (including an unknown kind) fails closed as a member file.
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

/** Raw text of a file — at most a little over READ_LIMIT_CHARS of it.
 *  `size` (if known) is checked against MAX_FILE_BYTES first. */
export async function readFileText(agent: HudAgent, key: string, size?: number): Promise<string> {
  if (size && size > MAX_FILE_BYTES) throw new HudApiError('File too large for the glasses', 413)
  const path = agent.kind === 'shared'
    ? `/fd/shared-agents/files/view?ref=${encodeURIComponent(agent.ref || '')}&id=${encodeURIComponent(key)}`
    : `${ownBase(agent, 'agent-file-view')}?path=${encodeURIComponent(key)}`
  const res = await hudFetch(path, {}, FILE_BODY_TIMEOUT_MS)
  if (!res.ok) throw await responseError(res)
  const len = Number(res.headers.get('content-length') || 0)
  if (len > MAX_FILE_BYTES) {
    res.body?.cancel().catch(() => {})
    throw new HudApiError('File too large for the glasses', 413)
  }
  try {
    return await readTextUpTo(res, READ_LIMIT_CHARS)
  } catch (e) {
    throw transportError(e)
  }
}

/** The body as text, cancelling the rest of the download once more than
 *  `maxChars` have arrived. */
async function readTextUpTo(res: Response, maxChars: number): Promise<string> {
  if (!res.body) return res.text()
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  const parts: string[] = []
  let chars = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    const s = decoder.decode(value, { stream: true })
    parts.push(s)
    chars += s.length
    if (chars > maxChars) {
      reader.cancel().catch(() => {})
      return parts.join('')
    }
  }
  parts.push(decoder.decode())
  return parts.join('')
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

/** A public (pre-sign-in) POST, bounded like hudFetch: a stalled request
 *  turns into HudApiError(…, 0), which the pairing screen retries. */
async function publicPost<T>(path: string, body: unknown): Promise<T> {
  try {
    const res = await timedFetch(path, { ...postJSON(body), credentials: 'include' }, PAIR_TIMEOUT_MS, BODY_TIMEOUT_MS)
    if (!res.ok) throw new HudApiError(errorMessage(await res.json().catch(() => null), res.status), res.status)
    return (await res.json()) as T
  } catch (e) {
    throw transportError(e)
  }
}

/** Public: start a pairing for this device. */
export function pairStart(label: string): Promise<PairStart> {
  return publicPost<PairStart>('/fd/auth/pair/start', { label })
}

/** Public: the glasses ask whether the code was approved. On approval the
 *  response also sets the refresh cookie, like a normal login. */
export function pairPoll(deviceCode: string): Promise<PairPoll> {
  return publicPost<PairPoll>('/fd/auth/pair/poll', { device_code: deviceCode })
}

/** Signed in (phone/desktop): which device is asking for this code? */
export function pairLookup(code: string): Promise<PairInfo> {
  return hudJSON<PairInfo>(`/fd/auth/pair/lookup?code=${encodeURIComponent(code)}`)
}

/** Signed in (phone/desktop): approve or deny the code. */
export function pairApprove(code: string, approve: boolean): Promise<{ ok: boolean; status: 'approved' | 'denied' }> {
  return hudJSON(`/fd/auth/pair/approve`, postJSON({ user_code: code, approve }))
}
