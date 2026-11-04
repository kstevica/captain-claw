// REST client for this deck's own link to a BotPort hub (/fd/botport) —
// captain_claw/flight_deck/botport_routes.py over botport_link.py. The deck
// connects out to the hub so its agents can consult agents on other Flight
// Decks (Agent → this deck → hub → other deck → agent), and serves the agents
// an admin exposes or their owners share. Status and sharing one's own agents
// are for everyone signed in (admins get all of it); the rest is admin-only.
// The secret is write-only: never returned.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

function _authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const _state = useAuthStore.getState()
  if (_state.authEnabled === true && !_state.token) {
    const refreshed = await refreshAccessToken()
    if (!refreshed) throw new Error('Not authenticated')
  }
  const build = (): RequestInit => ({ headers: _authHeaders(), credentials: 'include', ...init })
  let res = await fetch(path, build())
  // Only a failed refresh ends the session (see docker.ts fdFetch).
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (!(await refreshAccessToken())) {
      useAuthStore.getState().clearAuth()
      throw new Error('Session expired')
    }
    res = await fetch(path, build())
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    let msg = body.detail
    if (Array.isArray(msg)) {
      msg = msg.map((e: { loc?: string[]; msg?: string }) => `${(e.loc || []).join('.')}: ${e.msg}`).join('; ')
    }
    throw new Error(typeof msg === 'string' && msg ? msg : `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Status ──

export type BotportLinkState = 'disabled' | 'connecting' | 'connected' | 'backoff' | 'error' | 'stopped'

/** An agent another deck exposes (seen by this deck as `name@deck`). */
export interface BotportDeckAgent {
  name: string
  slug: string
  description: string
  expertise_tags: string[]
  status: string
}

/** Another Flight Deck on the hub (the roster never lists this deck). */
export interface BotportDeck {
  name: string
  // What the hub's operator calls it, and what it is for (may be empty).
  label?: string
  description?: string
  status: 'connected' | 'disconnected'
  version: string
  connected_at: string   // ISO, from the hub
  active_concerns: number
  max_concurrent: number
  agents: BotportDeckAgent[]
}

/** An agent linked to the hub directly (the old per-agent link). */
export interface BotportLegacyAgent {
  name: string
  personas: { name: string; description: string; expertise_tags: string[] }[]
  active_concerns: number
  max_concurrent: number
}

export interface BotportActivity {
  at: number            // unix seconds
  dir: 'in' | 'out'
  peer: string
  agent: string
  ok: boolean | null    // null: not finished
  detail: string
}

/** What everyone signed in sees. */
export interface BotportStatusBasic {
  enabled: boolean
  state: BotportLinkState
  deck_name: string
  // Only for non-admins: the remote agents they may consult, and the decks
  // online (with the display names the hub gives them).
  remote_agents?: string[]
  decks?: { name: string; label: string }[]
}

/** What an admin sees. Times are unix seconds. */
export interface BotportStatus extends BotportStatusBasic {
  url: string
  hub_version: string
  connected_since: number | null
  last_error: string
  last_error_at: number | null
  retry_in: number | null   // seconds, only in backoff
  problems: string[]
  exposed_agents: string[]
  network: { decks: BotportDeck[]; agents: BotportLegacyAgent[] }
  // Remote tasks running on this deck.
  inbound: { concern_id: string; from: string; agent: string; seconds: number; last_event: string }[]
  // This deck's agents waiting on remote agents.
  outbound: { concern_id: string; target: string; seconds: number; state: string }[]
  queued_results: number
  counters: Record<string, number>
  recent: BotportActivity[]
}

export function isFullStatus(s: BotportStatusBasic | BotportStatus): s is BotportStatus {
  return 'counters' in s && 'network' in s
}

export const getBotportStatus = () => fdFetch<BotportStatusBasic | BotportStatus>('/fd/botport/status')

// ── Join code ──

export interface BotportJoin {
  url: string
  deck_name: string
  key: string
  secret: string
}

/** A join code from the hub's dashboard (Decks → Add Flight Deck): everything
 * the connection needs, in one paste. "bpj1." + base64url JSON {u, n, k, s}.
 * null when it isn't one. */
export function parseJoinCode(text: string): BotportJoin | null {
  const t = text.trim().replace(/\s+/g, '')
  if (!t.startsWith('bpj1.')) return null
  try {
    let b = t.slice(5).replace(/-/g, '+').replace(/_/g, '/')
    while (b.length % 4) b += '='
    const bytes = Uint8Array.from(atob(b), (c) => c.charCodeAt(0))
    const o: unknown = JSON.parse(new TextDecoder().decode(bytes))
    if (!o || typeof o !== 'object') return null
    const r = o as Record<string, unknown>
    const join = { url: String(r.u ?? '').trim(), deck_name: String(r.n ?? '').trim(), key: String(r.k ?? '').trim(), secret: String(r.s ?? '').trim() }
    return join.url && join.deck_name && join.key && join.secret ? join : null
  } catch {
    return null
  }
}

// ── Settings (admin) ──

export interface BotportConfig {
  enabled: boolean
  url: string
  deck_name: string
  key: string
  // This deck's agents other decks may use (names or slugs); ['*'] = every one.
  expose: string[]
  // Decks that may send requests; ['*'] = any deck on the hub.
  accept_from: string[]
  allow_outbound: boolean
  // Every user's agents may consult other decks, not only the admins'.
  outbound_users: boolean
  max_concurrent: number
  max_inbound_timeout: number
  deny_tools: string[]
  // Users may share the agents they own on the network themselves.
  user_sharing: boolean
  secret_set: boolean
  // Fields pinned by FD_BOTPORT_* environment variables (they win).
  env_locked: string[]
  deck_id: string
  problems: string[]
  normalized_url: string
}

/** A partial update: omitted fields keep their value. `secret` replaces the
 *  stored one only when non-empty; `clear_secret` removes it. */
export interface BotportConfigPatch {
  enabled?: boolean
  url?: string
  deck_name?: string
  key?: string
  secret?: string
  clear_secret?: boolean
  expose?: string[]
  accept_from?: string[]
  allow_outbound?: boolean
  outbound_users?: boolean
  max_concurrent?: number
  max_inbound_timeout?: number
  deny_tools?: string[]
  user_sharing?: boolean
}

export const getBotportConfig = () => fdFetch<BotportConfig>('/fd/botport/config')

export const saveBotportConfig = (patch: BotportConfigPatch) =>
  fdFetch<BotportConfig>('/fd/botport/config', { method: 'PUT', body: JSON.stringify(patch) })

export const reconnectBotport = () =>
  fdFetch<{ ok: boolean }>('/fd/botport/reconnect', { method: 'POST' })

/** One of this deck's agents, for the expose picker. */
export interface BotportLocalAgent {
  name: string
  slug: string
  kind: 'docker' | 'process' | string
  status: string
  description: string
  owner: string
  // On the network now, and why: the admin's list and/or its owner's share.
  exposed: boolean
  by_admin: boolean
  by_owner: boolean
  // False for Flight Deck's own workers, which are never on the network.
  shareable: boolean
}

export const getBotportAgents = () => fdFetch<{ agents: BotportLocalAgent[] }>('/fd/botport/agents')

// ── Sharing one's own agents (every user) ──

export type BotportShareAgent = Pick<BotportLocalAgent,
  'name' | 'slug' | 'kind' | 'status' | 'exposed' | 'by_admin' | 'by_owner' | 'shareable'> & {
  // Yours: only its owner shares an agent (an admin may still stop a share).
  mine: boolean
}

/** Your agents (an admin: every agent) and whether each is on the network. */
export interface BotportShares {
  enabled: boolean
  state: BotportLinkState
  deck_name: string
  // The admin lets users share their own agents.
  user_sharing: boolean
  // You may share your own agents here now: the link is on and sharing by users is allowed.
  can_share: boolean
  admin: boolean
  agents: BotportShareAgent[]
}

export const getBotportShares = () => fdFetch<BotportShares>('/fd/botport/shares')

/** Share one of your agents on the network, or stop (an admin may stop any
 *  share). `kind` (docker / process) tells two agents of one name apart. */
export const setBotportShare = (slug: string, shared: boolean, kind = '') =>
  fdFetch<BotportShares>(`/fd/botport/shares/${encodeURIComponent(slug)}${kind ? `?kind=${encodeURIComponent(kind)}` : ''}`, {
    method: 'PUT', body: JSON.stringify({ shared }),
  })

export interface BotportProbe {
  ok: boolean
  url?: string
  version?: string
  protocol?: number
  error?: string
}

/** Is the hub reachable (its /healthz — never registers)? Blank: the saved address. */
export const probeBotport = (url = '') =>
  fdFetch<BotportProbe>(`/fd/botport/probe${url ? `?url=${encodeURIComponent(url)}` : ''}`)

/** The environment variable that pins a setting (env_locked names the field). */
export const botportEnvVar = (field: string) => `FD_BOTPORT_${field.toUpperCase()}`
