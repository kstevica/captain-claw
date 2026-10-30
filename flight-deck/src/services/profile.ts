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
  return _request(path, () => ({ headers: _authHeaders(), credentials: 'include', ...init }))
}

/** A multipart POST: the browser sets the Content-Type (with its boundary). */
async function fdUpload<T>(path: string, form: FormData): Promise<T> {
  return _request(path, () => {
    const headers = _authHeaders()
    delete headers['Content-Type']
    return { method: 'POST', body: form, headers, credentials: 'include' }
  })
}

async function _request<T>(path: string, build: () => RequestInit): Promise<T> {
  const _state = useAuthStore.getState()
  if (_state.authEnabled === true && !_state.token) {
    const refreshed = await refreshAccessToken()
    if (!refreshed) throw new Error('Not authenticated')
  }
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

/** Whether the Profile page edits the deck-wide defaults itself: on an
 *  auth-off deck (its one user is the admin, and there is no Admin page), and
 *  for an admin on a deck with accounts — never from the kiosk, where a
 *  visitor must not change what every agent on the deck receives. */
export function deckDefaultsEditable(
  authEnabled: boolean | null | undefined, kiosk: boolean, isAdmin = false,
): boolean {
  if (kiosk) return false
  return authEnabled === false || (authEnabled === true && isAdmin)
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

// ── The material behind the profile (/fd/profile/sources, /fd/profile/documents) ──
// The person's links, their company's name and links, and their documents (a
// CV, a bio): what the researcher drafts the three texts from. The name is the
// account's display name (changed through /fd/auth/me, see authStore).

export interface ProfileLink {
  url: string
  label: string
}

export interface ProfileDocument {
  id: string
  name: string
  pages: number
  chars: number
  /** Unix seconds. */
  added: number
}

export interface SourceCaps {
  links: number
  company_links: number
  documents: number
  url: number
  label: number
  company_name: number
  document_bytes: number
}

export interface ProfileSources {
  name: string
  email: string
  /** The account can be renamed here (not the auth-off deck's local user). */
  can_rename: boolean
  links: ProfileLink[]
  company_name: string
  company_links: ProfileLink[]
  documents: ProfileDocument[]
  caps: SourceCaps
  /** File kinds a document may be (".pdf", ".docx", …). */
  kinds: string[]
  /** On an upload: the files that could not be kept, and why. */
  warnings?: string[]
}

export const DEFAULT_SOURCE_CAPS: SourceCaps = {
  links: 8, company_links: 8, documents: 10, url: 500, label: 40, company_name: 120,
  document_bytes: 25 * 1024 * 1024,
}

export const DEFAULT_DOCUMENT_KINDS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.md', '.csv', '.html', '.htm']

export const EMPTY_SOURCES: ProfileSources = {
  name: '', email: '', can_rename: false, links: [], company_name: '', company_links: [],
  documents: [], caps: DEFAULT_SOURCE_CAPS, kinds: DEFAULT_DOCUMENT_KINDS,
}

function _links(raw: unknown): ProfileLink[] {
  if (!Array.isArray(raw)) return []
  return raw
    .filter((l): l is { url: string; label?: unknown } => !!l && typeof l === 'object' && typeof (l as { url?: unknown }).url === 'string')
    .map((l) => ({ url: l.url, label: _str(l.label) }))
}

function _documents(raw: unknown): ProfileDocument[] {
  if (!Array.isArray(raw)) return []
  return raw
    .filter((d): d is Record<string, unknown> => !!d && typeof d === 'object' && typeof (d as { id?: unknown }).id === 'string')
    .map((d) => ({
      id: d.id as string, name: _str(d.name) || 'document',
      pages: Number(d.pages) || 0, chars: Number(d.chars) || 0, added: Number(d.added) || 0,
    }))
}

function _normalizeSources(r: Partial<ProfileSources>): ProfileSources {
  return {
    name: _str(r.name),
    email: _str(r.email),
    can_rename: r.can_rename === true,
    links: _links(r.links),
    company_name: _str(r.company_name),
    company_links: _links(r.company_links),
    documents: _documents(r.documents),
    caps: { ...DEFAULT_SOURCE_CAPS, ...(r.caps || {}) },
    kinds: Array.isArray(r.kinds) && r.kinds.length ? r.kinds.map(String) : DEFAULT_DOCUMENT_KINDS,
    warnings: Array.isArray(r.warnings) ? r.warnings.map(String) : undefined,
  }
}

export const getProfileSources = () =>
  fdFetch<Partial<ProfileSources>>('/profile/sources').then(_normalizeSources)

export interface SourcesPatch {
  links?: ProfileLink[]
  company_name?: string
  company_links?: ProfileLink[]
}

/** Partial: only the fields given change. Documents go through their own calls. */
export const saveProfileSources = (patch: SourcesPatch) =>
  fdFetch<Partial<ProfileSources>>('/profile/sources', {
    method: 'PUT',
    body: JSON.stringify(patch),
  }).then(_normalizeSources)

/** Upload documents; each is read in full and kept. The ones that could not
 *  be read come back in `warnings`, the rest are kept. */
export const uploadProfileDocuments = (files: File[]) => {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  return fdUpload<Partial<ProfileSources>>('/profile/documents', form).then(_normalizeSources)
}

export const deleteProfileDocument = (id: string) =>
  fdFetch<Partial<ProfileSources>>(`/profile/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
    .then(_normalizeSources)

// ── The researcher (/fd/profile/research) ──
// Reads the documents and links, looks the person and the company up on the
// web, and drafts the three texts on the team's model. The drafts are shown
// for review — nothing is saved to the profile until the person saves.

export type ResearchState = 'idle' | 'running' | 'done' | 'error' | 'cancelled'

export interface ResearchLogEntry {
  ts: number
  message: string
  kind: 'info' | 'warn'
}

export interface ResearchSource {
  kind: 'document' | 'link' | 'page' | 'search'
  name: string
  url?: string
  ok: boolean
  chars?: number
  /** Why a link couldn't be read. */
  why?: string
  /** A search: how many results it gave. */
  results?: number
}

export interface ProfileResearch {
  state: ResearchState
  stage: string
  started: number | null
  finished: number | null
  log: ResearchLogEntry[]
  sources: ResearchSource[]
  drafts: ProfileFields | null
  notes: string[]
  tokens: number
  model: string
  model_source: 'team' | 'own' | ''
  error: string
}

export const IDLE_RESEARCH: ProfileResearch = {
  state: 'idle', stage: '', started: null, finished: null, log: [], sources: [], drafts: null,
  notes: [], tokens: 0, model: '', model_source: '', error: '',
}

/** How often the page asks after a run under way. */
export const RESEARCH_POLL_MS = 2000

/** What each stage of a run says while it is on. */
export const RESEARCH_STAGES: Record<string, string> = {
  starting: 'Starting…',
  reading: 'Reading your documents…',
  visiting: 'Visiting your links…',
  searching: 'Searching the web…',
  writing: 'Writing the three texts…',
  finished: 'Finished.',
}

const _STATES: ResearchState[] = ['idle', 'running', 'done', 'error', 'cancelled']

function _normalizeResearch(r: Partial<ProfileResearch>): ProfileResearch {
  const state = _STATES.includes(r.state as ResearchState) ? (r.state as ResearchState) : 'idle'
  const drafts = r.drafts && typeof r.drafts === 'object'
    ? { about_me: _str(r.drafts.about_me), company: _str(r.drafts.company), instructions: _str(r.drafts.instructions) }
    : null
  return {
    state,
    stage: _str(r.stage),
    started: typeof r.started === 'number' ? r.started : null,
    finished: typeof r.finished === 'number' ? r.finished : null,
    log: Array.isArray(r.log)
      ? r.log.map((e) => ({ ts: Number(e?.ts) || 0, message: _str(e?.message), kind: e?.kind === 'warn' ? 'warn' as const : 'info' as const }))
      : [],
    sources: Array.isArray(r.sources)
      ? r.sources.map((s) => ({
        kind: (['document', 'link', 'page', 'search'].includes(String(s?.kind)) ? s.kind : 'link') as ResearchSource['kind'],
        name: _str(s?.name), url: s?.url ? _str(s.url) : undefined, ok: s?.ok === true,
        chars: typeof s?.chars === 'number' ? s.chars : undefined,
        why: s?.why ? _str(s.why) : undefined,
        results: typeof s?.results === 'number' ? s.results : undefined,
      }))
      : [],
    drafts,
    notes: Array.isArray(r.notes) ? r.notes.map(String) : [],
    tokens: Number(r.tokens) || 0,
    model: _str(r.model),
    model_source: r.model_source === 'team' || r.model_source === 'own' ? r.model_source : '',
    error: _str(r.error),
  }
}

export const getProfileResearch = () =>
  fdFetch<Partial<ProfileResearch>>('/profile/research').then(_normalizeResearch)

/** Start a run (409 while one is under way, 400 with nothing to read from). */
export const startProfileResearch = () =>
  fdFetch<Partial<ProfileResearch>>('/profile/research', { method: 'POST' }).then(_normalizeResearch)

export const stopProfileResearch = () =>
  fdFetch<{ stopped: boolean }>('/profile/research', { method: 'DELETE' })

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
