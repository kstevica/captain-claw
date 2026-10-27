// REST client for the owner profile (/fd/profile) and the deck-wide defaults
// (/fd/admin/profile-defaults). Every agent working for a user receives the
// merged profile alongside its own instructions; Flight Deck composes it.
//
// These calls never carry the admin act-as header: a profile is only ever the
// signed-in user's own.

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const FD_BASE = '/fd'

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
  let res = await fetch(`${FD_BASE}${path}`, build())
  // Only a failed refresh ends the session (see docker.ts fdFetch).
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (!(await refreshAccessToken())) {
      useAuthStore.getState().clearAuth()
      throw new Error('Session expired')
    }
    res = await fetch(`${FD_BASE}${path}`, build())
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    let msg = body.detail
    if (Array.isArray(msg)) {
      // FastAPI validation errors: [{loc: [...], msg: "..."}, ...]
      msg = msg.map((e: { loc?: string[]; msg?: string }) => `${(e.loc || []).join('.')}: ${e.msg}`).join('; ')
    }
    throw new Error(typeof msg === 'string' && msg ? msg : `HTTP ${res.status}`)
  }
  return res.json()
}

// ── Types ──

export interface ProfileFields {
  about_me: string
  company: string
  instructions: string
}

export interface DeckProfileDefaults {
  company: string
  instructions: string
}

export type ProfileCaps = Record<keyof ProfileFields, number>

export type PreviewMode = 'full' | 'compact'

export interface ProfileResponse {
  profile: ProfileFields
  deck: DeckProfileDefaults
  caps: ProfileCaps
  /** What the agents receive — see PREVIEW_AUDIENCE for who gets which. */
  preview: Record<PreviewMode, string>
  /** On a save: how many of the user's agents got the new profile. */
  agents_updated?: number
}

export interface DeckDefaultsResponse extends DeckProfileDefaults {
  caps?: Partial<ProfileCaps>
  agents_updated?: number
}

// Mirrors the server's caps; the server's own values win when it sends them.
export const DEFAULT_PROFILE_CAPS: ProfileCaps = { about_me: 1500, company: 4000, instructions: 2000 }

export const EMPTY_PROFILE: ProfileFields = { about_me: '', company: '', instructions: '' }

// Every new agent starts in eco mode, so the compact block is what most agents
// actually get — the preview opens on it.
export const DEFAULT_PREVIEW_MODE: PreviewMode = 'compact'

/** Who receives each form of the block. */
export const PREVIEW_AUDIENCE: Record<PreviewMode, string> = {
  full: 'agents with eco mode off',
  compact: 'agents in eco mode — the default for new agents — nano agents, and workers in multi-agent runs',
}

/** Shown by the character counters: the cap is not what most agents see. */
export const ECO_SHORTENED_HINT =
  'Agents in eco mode (the default for new agents) receive a shortened version that keeps only the start of each field — put what matters most first.'

function _str(v: unknown): string {
  return typeof v === 'string' ? v : ''
}

function _normalizeProfile(r: Partial<ProfileResponse>): ProfileResponse {
  return {
    profile: {
      about_me: _str(r.profile?.about_me),
      company: _str(r.profile?.company),
      instructions: _str(r.profile?.instructions),
    },
    deck: { company: _str(r.deck?.company), instructions: _str(r.deck?.instructions) },
    caps: { ...DEFAULT_PROFILE_CAPS, ...(r.caps || {}) },
    preview: { full: _str(r.preview?.full), compact: _str(r.preview?.compact) },
    agents_updated: r.agents_updated,
  }
}

// ── Endpoints ──

export const getProfile = () =>
  fdFetch<Partial<ProfileResponse>>('/profile').then(_normalizeProfile)

/** Partial merge: only the fields given change. Fans out to the user's agents. */
export const saveProfile = (patch: Partial<ProfileFields>) =>
  fdFetch<Partial<ProfileResponse>>('/profile', {
    method: 'PUT',
    body: JSON.stringify(patch),
  }).then(_normalizeProfile)

/** Change the signed-in user's own password (needs the current one).
 *  `signOutOthers` ends their other sessions; this one stays signed in. */
export const changeOwnPassword = (current: string, next: string, signOutOthers: boolean) =>
  fdFetch<unknown>('/auth/me', {
    method: 'PUT',
    body: JSON.stringify({ current_password: current, password: next, sign_out_others: signOutOthers }),
  })

/** Admin only (auth-off decks: the local user). */
export const getDeckProfileDefaults = () =>
  fdFetch<DeckDefaultsResponse>('/admin/profile-defaults')

/** Admin only. Fans out to every agent on the deck. */
export const saveDeckProfileDefaults = (body: DeckProfileDefaults) =>
  fdFetch<DeckDefaultsResponse>('/admin/profile-defaults', {
    method: 'PUT',
    body: JSON.stringify(body),
  })

// Characters as the server counts them (Python str length = code points), so
// an emoji counts once here too.
export function charCount(s: string): number {
  return Array.from(s).length
}

export function agentsUpdatedLabel(n: number | undefined): string {
  const count = n ?? 0
  return `updated ${count} agent${count === 1 ? '' : 's'}`
}

/** A reload's new draft: a field the user changed (it differs from the profile
 *  the draft was based on) keeps their text; an untouched one takes the server's
 *  newer value, so a later Save can't revert a save made elsewhere. */
export function rebaseDraft(draft: ProfileFields, base: ProfileFields, next: ProfileFields): ProfileFields {
  const out = { ...next }
  for (const k of Object.keys(next) as (keyof ProfileFields)[]) {
    if (draft[k] !== base[k]) out[k] = draft[k]
  }
  return out
}

/** Whether the Profile page edits the deck-wide defaults itself: only on an
 *  auth-off deck (no Admin page there), and never from the kiosk. */
export function deckDefaultsEditable(authEnabled: boolean | null | undefined, kiosk: boolean): boolean {
  return authEnabled === false && !kiosk
}

/** While `dirty`, closing or reloading the browser tab asks first. Returns the
 *  cleanup, so it drops straight into a useEffect. */
export function guardUnload(
  target: Pick<Window, 'addEventListener' | 'removeEventListener'>,
  dirty: boolean,
): () => void {
  if (!dirty) return () => {}
  const onBeforeUnload = (e: BeforeUnloadEvent) => {
    e.preventDefault()
    e.returnValue = '' // older browsers only prompt when this is set
  }
  target.addEventListener('beforeunload', onBeforeUnload)
  return () => target.removeEventListener('beforeunload', onBeforeUnload)
}

// ── The kiosk Info dialog (/fd/info) ──
// The signed-in user's corner of the deck in one read: the tier set their
// agents are created on (never a key), archetypes, what their agents receive,
// plan, connections. Their agents come from the agent lists the SPA has.

export interface InfoTier {
  id: string
  label: string
  use: string
  provider: string
  model: string
  endpoint: string      // custom endpoint host, '' for the provider's own
  input_ctx: number
  output_ctx: number
  vision: boolean
  from_set: boolean     // false: the set leaves it out, the deck's built-in applies
  archetypes: number    // archetypes that default to this tier
}

export interface MyInfo {
  tiers: { source: 'own' | 'team' | 'none' | 'byok'; set_name: string; tiers: InfoTier[] }
  archetypes: { total: number; base: number; own: number; shared: number }
  profile: {
    own: ProfileFields
    deck: DeckProfileDefaults
    preview: { full: string; compact: string }
  } | null
  account: {
    email: string
    display_name: string
    role: string
    created_at: string
    plan: string
    limits: Partial<Record<PlanLimitKey, number>>
    kiosk: boolean
    usage?: QuotaStatus
  }
  connections: {
    google?: { configured: boolean; connected: boolean }
    microsoft?: { configured: boolean; connected: boolean }
    telegram?: { configured: boolean; status: string; linked: boolean; agent: string }
    jev?: { connected: boolean; source: string }
    // source: 'own' | 'shared' (team key) | 'env' (the server's SONIOX_API_KEY)
    soniox?: { connected: boolean; source: string }
  }
}

export const getMyInfo = () => fdFetch<MyInfo>('/info')

// ── Daily plan allowances (/fd/quota) ──
// Messages and AI tokens a user may spend per UTC day on their plan. A limit
// of 0 is unlimited; admins are exempt (their usage is still counted). An
// admin may give users free resets, each starting the day's allowances over.

export type PlanLimitKey =
  | 'max_agents' | 'max_storage_mb' | 'requests_per_minute' | 'spawns_per_hour'
  | 'messages_per_day' | 'tokens_per_day'

export interface QuotaAllowance {
  used: number
  limit: number             // 0 = unlimited
  remaining: number | null  // null when unlimited
}

export interface QuotaStatus {
  plan: string
  plan_label: string
  day: string
  resets_at: string         // ISO, the next UTC midnight
  exempt: boolean
  messages: QuotaAllowance
  tokens: QuotaAllowance
  blocked: 'messages' | 'tokens' | null
  warning: 'messages' | 'tokens' | null
  resets_left: number       // free resets an admin gave them
  byok?: boolean            // on their own API keys: no daily limits
  byok_allowed?: boolean    // the deck lets them switch to their own keys
  byok_tokens?: number      // AI tokens spent on their own keys today
}

export const getMyQuota = () => fdFetch<QuotaStatus>('/quota')

/** Spend one free reset: today's messages and AI tokens start over. */
export const spendFreeReset = () => fdFetch<QuotaStatus>('/quota/reset', { method: 'POST' })
