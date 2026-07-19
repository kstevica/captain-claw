// REST client for Iskra living beings (/fd/beings).

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
  const res = await fetch(`${FD_BASE}${path}`, {
    headers: _authHeaders(),
    credentials: 'include',
    ...init,
  })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    const refreshed = await refreshAccessToken()
    if (refreshed) {
      const retry = await fetch(`${FD_BASE}${path}`, {
        headers: _authHeaders(),
        credentials: 'include',
        ...init,
      })
      if (retry.ok) return retry.json()
    }
    useAuthStore.getState().clearAuth()
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
  }
  return res.json()
}

// ── Types ──

export interface BeingsMeta {
  attributes: { code: string; name: string }[]
  pool: number
  attr_min: number
  attr_max: number
  presets: Record<string, Record<string, number>>
  allowance_presets: string[]
  stages: Record<string, {
    capabilities: string[]
    tiers: string[]
    max_preset: string | null
    savings_days: number
    metamorphosis: string
  }>
  constitution: string
}

export interface BeingListItem {
  slug: string
  name: string
  stage: string
  state: string
  born_at: string
  hatched_at: string | null
  died_at: string | null
  balance_tokens: number | null
  allowance_preset: string | null
}

export interface BoardTask {
  id: string
  kind: 'go' | 'meet' | 'build' | string
  target: string
  detail: string           // a build task's object kind; '' otherwise
  state: 'open' | 'active' | 'done' | 'refused' | string
  note: string             // a refusal reason, when refused
  object_id: string        // the stake a done build task made
  created_at: string
  done_at: string | null
}

export interface BeingVitals {
  slug: string
  name: string
  stage: string
  state: string
  born_at: string
  hatched_at: string | null
  died_at: string | null
  attention_credits: number
  attention_cap?: number
  attributes: Record<string, number>
  derived: {
    drive_weights: Record<string, number>
    risk_appetite: number
    whimsy: number
    thrift: number
    [k: string]: unknown
  }
  generation: number
  lineage: string[]
  metamorphoses: unknown[]
  interest_seeds: string[]
  wallet: {
    balance_tokens: number
    allowance_preset: string
    effective_preset: string
    per_day_tokens: number | null
    enforced: boolean
    savings_ceiling: number | null
    daily_burn_cap: number | null
  }
  spent_today: number
  capabilities: string[]
  house_rules: string[]
  rules_pending: boolean
  media_diet: { allow?: string[]; deny?: string[] }
  affect: { mood?: string; notes?: string[] }
  persona: string
  pending_self_mod: { content: string; reason: string; proposed_at: string } | null
  pending_procreation: {
    partner: string | null; child_name: string; case: string
    letter: string; proposed_at: string
  } | null
  pending_name: { name: string; why: string; proposed_at: string } | null
  reading_list: {
    id: string; ref: string; note: string; fee_tokens: number
    assigned_at: string; done_at: string | null; report_path: string | null
  }[]
  elder_after_days: number | null
  broadcast: { text: string; at: string } | null
  location?: { at?: string | null; to?: string | null } | null
  position?: { xy: [number, number]; at: string | null; to: string | null; minutes_left: number } | null
  coins?: number
  tick_interval_minutes: number | null
  cognition: 'monolith' | 'faculties' | 'micro'
  compact_mode: boolean
  instincts: boolean
  intent: { stay?: boolean; avoid?: string[] }
  plan: { id: string; kind: string; target: string }[]
  // The work board (work-board plan): the mind assigns tasks, the feet
  // actively work them and mark them done / active / refused-with-reason.
  board?: {
    open: BoardTask[]
    active: BoardTask[]
    recent: BoardTask[]   // last handful of done/refused, newest first
  }
  avatar: { c: number; p: string }
  body_archetype: string
  // An explicit body connection, or {} for the stage-tier default. The key
  // itself never crosses the wire — only whether one is set.
  body_config: {
    provider?: string
    model?: string
    base_url?: string
    input_ctx?: number
    output_ctx?: number
    has_key?: boolean
  }
  // Effective Mrav state of the body (persistent toggle, or the agent-card flag).
  body_mrav: boolean
  unread_from_being: number
  public: boolean
  visit_url: string
  visit_secret: string
  visit_last_announce: string | null
  // home as your canvas (world-shaping plan Phase 4)
  home_name: string
  home_look: { roof?: string; wall?: string } | null
}

export const TICK_INTERVAL_CHOICES = [2, 5, 10, 15, 30, 60] as const

export interface BeingEvent {
  kind: string
  data: Record<string, unknown>
  at: string
}

export interface SelfFile {
  path: string
  size: number
  mtime: string
}

export interface ConceivePayload {
  name: string
  attributes: Record<string, number>
  voice_seed?: string
  interest_seeds?: string[]
  allowance_preset?: string
  birth_letter?: string
}

// ── Endpoints ──

export const getBeingsMeta = () => fdFetch<BeingsMeta>('/beings/meta')
export const listBeings = () => fdFetch<{ beings: BeingListItem[] }>('/beings')
export const getBeingVitals = (slug: string) => fdFetch<BeingVitals>(`/beings/${slug}`)
export const getBeingEvents = (slug: string, limit = 12) =>
  fdFetch<{ events: BeingEvent[] }>(`/beings/${slug}/events?limit=${limit}`)
export const getBeingJournal = (slug: string, date = '') =>
  fdFetch<{ date: string; text: string }>(`/beings/${slug}/journal${date ? `?date=${date}` : ''}`)
export const getSelfFiles = (slug: string) =>
  fdFetch<{ files: SelfFile[] }>(`/beings/${slug}/self/files`)
export const messageBeing = (slug: string, body: string) =>
  fdFetch<{ message: { id: string; preview: string } }>(`/beings/${slug}/message`, {
    method: 'POST', body: JSON.stringify({ body }),
  })
export interface ThreadItem {
  from: 'parent' | 'being'
  body: string
  at: string
  read: boolean
}
export const getBeingMessages = (slug: string) =>
  fdFetch<{ thread: ThreadItem[] }>(`/beings/${slug}/messages`)

export interface BeingGraph {
  nodes: { path: string; group: string; degree: number }[]
  edges: { from: string; to: string; rel: string; why: string }[]
  density: number
  connected_fraction: number
}
export const getBeingGraph = (slug: string) =>
  fdFetch<BeingGraph>(`/beings/${slug}/graph`)
export interface MindRebuild {
  restored: number
  kept: number
  skipped: number
  ledgered: number
  graph: BeingGraph
}
/** Repair the Mind from the being's own ledger — additive and idempotent:
 *  it restores edges a bad read wiped, and never invents or deletes one. */
export const rebuildBeingGraph = (slug: string) =>
  fdFetch<MindRebuild>(`/beings/${slug}/graph/rebuild`, { method: 'POST' })
export const getSelfFile = (slug: string, path: string) =>
  fdFetch<{ path: string; text: string }>(`/beings/${slug}/self/file?path=${encodeURIComponent(path)}`)
export const getLiabilities = () =>
  fdFetch<{ total_tokens: number; beings: { slug: string; balance_tokens: number }[] }>('/beings/liabilities')

export interface VillageItem {
  kind: string
  at: string
  text: string
}
export const getVillage = (limit = 40) =>
  fdFetch<{ items: VillageItem[] }>(`/beings/village?limit=${limit}`)

// ── Letters observatory: being↔being conversations (+ refused reaches) ──
export interface LetterParticipant {
  slug: string
  name: string
  stage: string
  state: string
}
export interface LetterMessage {
  kind: 'letter' | 'refused'
  from_slug: string
  from_name: string
  to_slug: string
  to_name: string
  body?: string
  reason?: string
  at: string
  read: boolean
}
export interface LetterThread {
  key: string
  participants: LetterParticipant[]
  messages: LetterMessage[]
  last_at: string
}
export interface LettersOverview {
  threads: LetterThread[]
  stats: { threads: number; delivered: number; refused: number }
}
export const getLetters = (limit = 500) =>
  fdFetch<LettersOverview>(`/beings/letters?limit=${limit}`)

export interface Quest {
  id: string
  title: string
  spec: string
  fee_tokens: number
  fee_coins?: number
  origin: string
  state: string
  claimant: string
  result_text: string
  created_at: string
}
export interface Venture {
  id: string
  being: string
  title: string
  description: string
  price_tokens: number
  cadence_days: number
  state: string
  pending_result: string
  deliveries: number
  next_due_at: string | null
}
export const getBoard = () =>
  fdFetch<{ quests: Quest[]; ventures: Venture[] }>('/beings/board')
export const postQuest = (title: string, spec: string, fee_tokens: number, fee_coins = 0) =>
  fdFetch<{ quest: Quest }>('/beings/quests', {
    method: 'POST', body: JSON.stringify({ title, spec, fee_tokens, fee_coins }),
  })
export const judgeQuest = (questId: string, approve: boolean, note = '') =>
  fdFetch<{ quest: Quest }>(`/beings/quests/${questId}/judge`, {
    method: 'POST', body: JSON.stringify({ approve, note }),
  })
export const cancelQuest = (questId: string) =>
  fdFetch<{ quest: Quest }>(`/beings/quests/${questId}/cancel`, { method: 'POST' })
export const approveVenture = (ventureId: string, price_tokens: number | null) =>
  fdFetch<{ venture: Venture }>(`/beings/ventures/${ventureId}/approve`, {
    method: 'POST', body: JSON.stringify({ price_tokens }),
  })
export const setVentureState = (ventureId: string, state: string) =>
  fdFetch<{ venture: Venture }>(`/beings/ventures/${ventureId}/state`, {
    method: 'POST', body: JSON.stringify({ state }),
  })
export const acceptVenture = (ventureId: string, approve: boolean, note = '') =>
  fdFetch<{ venture: Venture }>(`/beings/ventures/${ventureId}/accept`, {
    method: 'POST', body: JSON.stringify({ approve, note }),
  })

export const approveSelfMod = (slug: string) =>
  fdFetch<{ persona: string }>(`/beings/${slug}/self-mod/approve`, { method: 'POST' })
export const rejectSelfMod = (slug: string, note = '') =>
  fdFetch<{ ok: boolean }>(`/beings/${slug}/self-mod/reject`, {
    method: 'POST', body: JSON.stringify({ note }),
  })
export const rollbackPersona = (slug: string) =>
  fdFetch<{ persona: string }>(`/beings/${slug}/self-mod/rollback`, { method: 'POST' })

// The naming rite: bless or decline the being's one chosen name.
export const approveChosenName = (slug: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/name/approve`, { method: 'POST' })
export const rejectChosenName = (slug: string, note = '') =>
  fdFetch<BeingVitals>(`/beings/${slug}/name/reject`, {
    method: 'POST', body: JSON.stringify({ note }),
  })

// Elderhood: opt into a natural span (days alive; null = off).
export const setElderhood = (slug: string, days: number | null) =>
  fdFetch<BeingVitals>(`/beings/${slug}/elderhood`, {
    method: 'POST', body: JSON.stringify({ days }),
  })

// The migration rite: export the whole life and close it here.
export const emigrateBeing = (slug: string) =>
  fdFetch<{ manifest: unknown; being: BeingVitals }>(`/beings/${slug}/emigrate`, {
    method: 'POST',
  })

// Education: the reading list (fee paid on a verified report file).
export const addReading = (slug: string, ref: string, note = '', feeTokens = 0) =>
  fdFetch<BeingVitals>(`/beings/${slug}/reading`, {
    method: 'POST', body: JSON.stringify({ ref, note, fee_tokens: feeTokens }),
  })
export const removeReading = (slug: string, itemId: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/reading/${itemId}`, { method: 'DELETE' })

export const approveProcreation = (slug: string, name = '') =>
  fdFetch<{ ok: boolean; child: BeingVitals }>(`/beings/${slug}/procreate/approve`, {
    method: 'POST', body: JSON.stringify({ name }),
  })
export const rejectProcreation = (slug: string, note = '') =>
  fdFetch<{ ok: boolean }>(`/beings/${slug}/procreate/reject`, {
    method: 'POST', body: JSON.stringify({ note }),
  })
export const arrangeOffspring = (slug: string, name: string, partner: string | null, letter = '') =>
  fdFetch<{ ok: boolean; child: BeingVitals }>(`/beings/${slug}/procreate/arrange`, {
    method: 'POST', body: JSON.stringify({ name, partner, letter }),
  })

export const conceiveBeing = (payload: ConceivePayload) =>
  fdFetch<{ ok: boolean; being: BeingVitals }>('/beings/conceive', {
    method: 'POST', body: JSON.stringify(payload),
  })
export const hatchBeing = (slug: string) =>
  fdFetch<BeingVitals & { birth?: { warnings: string[] } }>(`/beings/${slug}/hatch`, { method: 'POST' })
export const tickBeing = (slug: string, kind: 'wake' | 'dream' = 'wake') =>
  fdFetch<{ result: Record<string, unknown>; vitals: BeingVitals }>(`/beings/${slug}/tick`, {
    method: 'POST', body: JSON.stringify({ kind }),
  })
export const setAllowance = (slug: string, preset: string) =>
  fdFetch<{ wallet: BeingVitals['wallet'] }>(`/beings/${slug}/allowance`, {
    method: 'POST', body: JSON.stringify({ preset }),
  })
export const pauseBeing = (slug: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/pause`, { method: 'POST' })
export const wakeBeing = (slug: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/wake`, { method: 'POST' })
export const euthanizeBeing = (slug: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/euthanize`, {
    method: 'POST', body: JSON.stringify({ confirm: true }),
  })

// ── Phase 2: parenting ──

export interface Chore {
  id: string
  spec: string
  fee_tokens: number
  fee_coins?: number
  escrow_state: 'open' | 'judging' | 'paid' | 'failed'
  result_text: string
  judge_note: string
  created_at: string
}

export interface ReportCard {
  period_days: number
  ticks: number
  acts: Record<string, number>
  tokens_spent_weighted: number
  tokens_earned: number
  messages_to_parent: number
  messages_suppressed: number
  rut_score: number
  concerns: string[]
  milestones: string[]
  in_its_own_words: string
  affect: { mood?: string; notes?: string[] }
  mind?: { nodes: number; edges: number; density: number; connected_fraction: number; consolidations?: number }
  drives_trail?: Array<Record<string, number | string>>
}

export const postChore = (slug: string, spec: string, fee_tokens: number, fee_coins = 0) =>
  fdFetch<{ chore: Chore }>(`/beings/${slug}/chores`, {
    method: 'POST', body: JSON.stringify({ spec, fee_tokens, fee_coins }),
  })
export const listChores = (slug: string) =>
  fdFetch<{ chores: Chore[] }>(`/beings/${slug}/chores`)
export const judgeChore = (slug: string, jobId: string, approve: boolean, note = '') =>
  fdFetch<{ chore: Chore }>(`/beings/${slug}/chores/${jobId}/judge`, {
    method: 'POST', body: JSON.stringify({ approve, note }),
  })
export const setHouseRules = (slug: string, rules: string[]) =>
  fdFetch<{ house_rules: string[] }>(`/beings/${slug}/rules`, {
    method: 'POST', body: JSON.stringify({ rules }),
  })
export const setMediaDiet = (slug: string, allow: string[], deny: string[]) =>
  fdFetch<{ media_diet: unknown }>(`/beings/${slug}/diet`, {
    method: 'POST', body: JSON.stringify({ allow, deny }),
  })
export const getReportCard = (slug: string, days = 7) =>
  fdFetch<ReportCard>(`/beings/${slug}/report-card?days=${days}`)

export interface ReadinessDim {
  key: string; label: string; score: number
  status: 'green' | 'amber' | 'red'; detail: string; evidence: string; critical: boolean
}
export interface Readiness {
  stage: string; next_stage: string | null; days_alive: number; window_days: number
  overall: { score: number; status: 'ready' | 'emerging' | 'not_yet' | 'grown' }
  dimensions: ReadinessDim[]
  estimate_days: number | null
  unlocks: string[]
  recommendation: { action: string; title: string; steps: string[]; expect: string[]; cautions: string[] }
}
export const getReadiness = (slug: string) =>
  fdFetch<Readiness>(`/beings/${slug}/readiness`)

export interface Assessor { slug: string; name: string }
export const getAssessors = () =>
  fdFetch<{ assessors: Assessor[] }>('/beings/assessors')
export const requestAssessment = (slug: string, assessor: string) =>
  fdFetch<{ assessor: string; assessment: string; score: number; verdict: string }>(
    `/beings/${slug}/assess`, {
      method: 'POST', body: JSON.stringify({ assessor }),
    })

export interface SavedAssessment {
  id: string; assessor: string; stage: string; score: number | null
  verdict: string; content: string; at: string; released_at: string | null
}
export const listAssessments = (slug: string) =>
  fdFetch<{ assessments: SavedAssessment[] }>(`/beings/${slug}/assessments`)
export const saveAssessment = (slug: string, a: {
  assessor: string; content: string; score?: number | null; verdict?: string
}) =>
  fdFetch<{ assessment: SavedAssessment }>(`/beings/${slug}/assessments`, {
    method: 'POST', body: JSON.stringify(a),
  })
export const deleteAssessment = (slug: string, id: string) =>
  fdFetch<{ ok: boolean }>(`/beings/${slug}/assessments/${id}`, { method: 'DELETE' })
export const setStage = (slug: string, stage: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/stage`, {
    method: 'POST', body: JSON.stringify({ stage }),
  })
export const setCadence = (slug: string, minutes: number | null) =>
  fdFetch<BeingVitals>(`/beings/${slug}/cadence`, {
    method: 'POST', body: JSON.stringify({ minutes }),
  })

// 'micro' = faculties whose JSON steps run grammar-locked on the micro tier
export type Cognition = 'monolith' | 'faculties' | 'micro'
export const setCognition = (slug: string, mode: Cognition) =>
  fdFetch<BeingVitals>(`/beings/${slug}/cognition`, {
    method: 'POST', body: JSON.stringify({ mode }),
  })

// Compact mode: compact instruction set for ticks + a lean body (micro
// system prompt, capped context). Same narrative, fewer tokens per heartbeat.
export const setCompactMode = (slug: string, on: boolean) =>
  fdFetch<BeingVitals>(`/beings/${slug}/compact`, {
    method: 'POST', body: JSON.stringify({ on }),
  })

// The body brain (instincts): between mind ticks the feet settle walks,
// feel encounters, fulfill plans, and make tiny capped decisions.
export const setInstincts = (slug: string, on: boolean) =>
  fdFetch<BeingVitals>(`/beings/${slug}/instincts`, {
    method: 'POST', body: JSON.stringify({ on }),
  })

// The look: one of 10 storybook characters in one of 4 palettes; a stable
// slug-hash default applies until the parent's first pick.
export const setAvatar = (slug: string, c: number, p: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/avatar`, {
    method: 'POST', body: JSON.stringify({ c, p }),
  })

// The parent's nudge: send an alive being onto the road to a place (or
// "home"). Plots the same A* course a mind- or feet-walk uses; only the
// living walk (a paused/torpid being refuses).
export const nudgeBeing = (slug: string, dest: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/go`, {
    method: 'POST', body: JSON.stringify({ dest }),
  })

// Body housekeeping: run the being's body on an archetype (its tier → model,
// tools, cognitive mode). Empty id → the stage default. Respawns the body.
export interface BodyArchetypeOption { id: string; role?: string; tier?: string; family?: string }
export const listBodyArchetypes = () =>
  fdFetch<{ archetypes: BodyArchetypeOption[] }>('/archetypes')
    .then((r) => r.archetypes || [])
export const setBodyArchetype = (slug: string, archetypeId: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/body-archetype`, {
    method: 'POST', body: JSON.stringify({ archetype_id: archetypeId }),
  })

export interface BodyConnectionInput {
  provider?: string
  model?: string
  base_url?: string
  api_key?: string
  input_ctx?: number
  output_ctx?: number
}

// Pin the body to an explicit provider/model/ctx/key/base (authoritative — no
// hatch-tier fallback), or pass {} to clear it back to the stage tier.
export const setBodyConfig = (slug: string, cfg: BodyConnectionInput) =>
  fdFetch<BeingVitals>(`/beings/${slug}/body-config`, {
    method: 'POST',
    body: JSON.stringify({
      provider: cfg.provider ?? '',
      model: cfg.model ?? '',
      base_url: cfg.base_url ?? '',
      api_key: cfg.api_key ?? '',
      input_ctx: cfg.input_ctx ?? 0,
      output_ctx: cfg.output_ctx ?? 0,
    }),
  })

// Persist whether the body runs the Mrav runtime (survives a body rebuild).
export const setBodyMrav = (slug: string, on: boolean) =>
  fdFetch<BeingVitals>(`/beings/${slug}/body-mrav`, {
    method: 'POST', body: JSON.stringify({ on }),
  })

// Clear the "unread messages from the being" cue — the parent opened its thread.
export const markBeingRead = (slug: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/mark-read`, { method: 'POST' })

// Fixed parent top-up amounts (tokens) — mirrors beings.GRANT_AMOUNTS.
export const GRANT_AMOUNTS = [2_000_000, 5_000_000, 10_000_000, 20_000_000] as const
export const rechargeBeing = (slug: string, tokens: number) =>
  fdFetch<BeingVitals>(`/beings/${slug}/recharge`, {
    method: 'POST', body: JSON.stringify({ tokens }),
  })
// Pocket money (space plan Phase 2): coins are money, not food — a second
// ledger beside tokens; a being may convert one-way from adolescence.
export const grantCoins = (slug: string, coins: number, note = '') =>
  fdFetch<BeingVitals>(`/beings/${slug}/coins`, {
    method: 'POST', body: JSON.stringify({ coins, note }),
  })

// ── The living map (space plan Phase 4) ──
// Position is a pure function of the clock, so the client animates walking
// from one snapshot — zero polling beyond a lazy refresh.

export interface VillagePlace {
  id: string; name: string; x: number; y: number
  affordances: string[]; description: string
  // the world model (village-world plan): footprint in tiles + the door
  w?: number; h?: number; kind?: string
  door_x?: number | null; door_y?: number | null
}
export interface VillageBeingPos {
  slug: string; name: string; stage: string; state: string
  xy: [number, number]; at: string | null; to: string | null
  minutes_left: number; home_xy?: [number, number]; speed: number
  avatar?: { c: number; p: string }
  // the plotted course (village-world plan Phase 2): walk it client-side
  path?: [number, number][]; departed_at?: string; total_minutes?: number
  // a visiting being from another village (visiting-beings plan §1):
  // rendered distinctly, wearing a "visiting from <origin>" label.
  kind?: 'resident' | 'visitor'; from?: string; mood?: string
  // home as your canvas (world-shaping plan Phase 4): the cottage's
  // being-chosen name and dress.
  home_name?: string
  home_look?: { roof?: string; wall?: string } | null
}
export interface VillageProp { tile: [number, number]; kind: string }
// A made thing (world-shaping plan): crafted by a being, standing on open
// ground. `face` is the inscription's first line, read from the maker's
// real proof file; `by` is the maker's slug, `by_name` its display name.
export interface VillageObject {
  id: string; kind: string; name: string; affordance: string
  xy: [number, number]; tile: [number, number]
  face: string; by: string; by_name: string
  // a public work the steward placed on the commons (Phase 5)
  civic?: boolean
  // a beginning the feet broke that the mind hasn't finished (instinct-build)
  staked?: boolean
  // placed by the parent's own hand (parent-build)
  parent?: boolean
}
// A sign in the grass (FPV plan Phase 3): planted by the parent or a
// public visitor; each being finds each sign once. `found` counts finders;
// `read_by` carries slugs only on the parent's own map.
export interface VillageNote {
  id: string; x: number; y: number; text: string
  author: string; author_kind: 'parent' | 'visitor'
  created_at: string; found: number; read_by?: string[]
}
export interface VillageMapData {
  plot: number; places: VillagePlace[]; beings: VillageBeingPos[]
  grid?: { plot_w: number; plot_h: number; tile_size: number }
  terrain?: { default_elevation: number }
  roads?: [number, number][]
  props?: VillageProp[]
  notes?: VillageNote[]
  objects?: VillageObject[]
}
export const getVillageMap = () =>
  fdFetch<VillageMapData>('/beings/village-map')
export const plantVillageNote = (x: number, y: number, text: string) =>
  fdFetch<{ note: VillageNote }>('/beings/village-map/notes', {
    method: 'POST', body: JSON.stringify({ x, y, text }),
  })
export const pullVillageNote = (id: string) =>
  fdFetch<{ ok: boolean }>(`/beings/village-map/notes/${id}`, {
    method: 'DELETE',
  })
export const postVillagePresence = (x: number, y: number) =>
  fdFetch<{ felt: string[] }>('/beings/village-map/presence', {
    method: 'POST', body: JSON.stringify({ x, y }),
  })
// The living ghost roster (FPV plan Phase 5): other ghosts roaming the
// village right now — the parent and public visitors see each other.
export interface GhostPresence {
  id: string; kind: 'parent' | 'visitor'; name: string; xy: [number, number]
}
export const postGhostBeat = (id: string, x: number, y: number) =>
  fdFetch<{ ghosts: GhostPresence[] }>('/beings/village-map/ghost', {
    method: 'POST', body: JSON.stringify({ id, x, y }),
  })
export const postGhostLeave = (id: string) =>
  fdFetch<{ ok: boolean }>('/beings/village-map/ghost/leave', {
    method: 'POST', body: JSON.stringify({ id, x: 0, y: 0 }),
  })
export const getVillagePlace = (placeId: string) =>
  fdFetch<{ place: VillagePlace; guestbook: string }>(
    `/beings/village-map/place/${placeId}`)
export interface MarketListing {
  id: string; title: string; path: string; price_coins: number
  state: string; seller: string; seller_slug: string; created_at: string
}
export const getMarket = () =>
  fdFetch<{ listings: MarketListing[] }>('/beings/market')

// ── The civic layer (space plan Phase 5) ──
export interface VillageCommission {
  id: string; name: string; why: string; affordance: string
  target_coins: number; raised_coins: number; state: string
  contributors: { being_id: string; name: string; coins: number }[]
}
export interface VillageLife {
  commission: VillageCommission | null
  steward: string | null
  steward_stipend_coins: number
}
export const getVillageLife = () =>
  fdFetch<VillageLife>('/beings/village-life')
export const judgeCommission = (approve: boolean, note = '') =>
  fdFetch<unknown>('/beings/commission/judge', {
    method: 'POST', body: JSON.stringify({ approve, note }),
  })
export const setStewardStipend = (coins: number) =>
  fdFetch<{ steward_stipend_coins: number }>('/beings/village-stipend', {
    method: 'POST', body: JSON.stringify({ coins }),
  })

// The civic hand (world-shaping plan Phase 5): the parent may rename or
// redescribe a place (the id never changes; MAP.md is rewritten), and
// redraw the whole ground with the one-shot architect.
export const editVillagePlace = (
  placeId: string, body: { name?: string; description?: string }) =>
  fdFetch<{ ok: boolean; place: VillagePlace }>(
    `/beings/village-map/place/${placeId}/edit`, {
      method: 'POST', body: JSON.stringify(body),
    })
export const redesignVillage = () =>
  fdFetch<{ ok: boolean; places: VillagePlace[] }>(
    '/beings/village-map/architect', { method: 'POST' })

// The parent's own hand on the world (parent-build): place / lift a made
// thing anywhere in the village, from the map or the FPV ghost.
export const OBJECT_KINDS = [
  'bench', 'cairn', 'signpost', 'planter', 'sculpture', 'lantern',
  'fountain', 'shrine'] as const
export type ObjectKind = typeof OBJECT_KINDS[number]
export const placeVillageObject = (
  body: { kind: string; name: string; inscription: string; x: number; y: number }) =>
  fdFetch<{ ok: boolean; object: VillageObject }>('/beings/village-map/object', {
    method: 'POST', body: JSON.stringify(body),
  })
export const removeVillageObject = (id: string) =>
  fdFetch<{ ok: boolean }>(`/beings/village-map/object/${id}`, {
    method: 'DELETE',
  })
// Road-building: paint/lift a street tile at a village-unit spot.
export const toggleVillageRoad = (x: number, y: number) =>
  fdFetch<{ ok: boolean }>('/beings/village-map/road', {
    method: 'POST', body: JSON.stringify({ x, y }),
  })
// Grow map: set the (square) plot size — grow-only, clamped server-side.
export const PLOT_SIZES = [1000, 1400, 1800, 2400] as const
export const setVillagePlotSize = (size: number) =>
  fdFetch<{ plot_w: number; plot_h: number; tile_size: number }>(
    '/beings/village-map/size', {
      method: 'POST', body: JSON.stringify({ size }),
    })

// ── The public square (parent side) ──

export const setBeingPublic = (slug: string, isPublic: boolean) =>
  fdFetch<BeingVitals>(`/beings/${slug}/public`, {
    method: 'POST', body: JSON.stringify({ public: isPublic }),
  })

export interface PublicThreadMsg {
  role: 'public' | 'being'
  sender_name: string
  body: string
  at: string
  read_at: string | null
  answered_at: string | null
}
export interface ParentPublicThread {
  thread_id: string
  sender_name: string
  created_at: string
  updated_at: string
  messages: PublicThreadMsg[]
}
export const getPublicThreads = (slug: string) =>
  fdFetch<{ threads: ParentPublicThread[] }>(`/beings/${slug}/public-threads`)

// ── Export / import / hard-remove ──

export type BeingExport = Record<string, unknown>

export const exportBeing = (slug: string) =>
  fdFetch<BeingExport>(`/beings/${slug}/export`)

export const importBeing = (manifest: BeingExport) =>
  fdFetch<{ ok: boolean; warnings: string[]; being: BeingVitals }>(
    '/beings/import', { method: 'POST', body: JSON.stringify(manifest) })

export const purgeBeing = (slug: string) =>
  fdFetch<{ ok: boolean; removed: string; home_removed: boolean }>(
    `/beings/${slug}`, { method: 'DELETE' })

// ── Village description (shown on the public /village page) ──

export interface VillageMeta {
  name: string
  description: string
  secret: string
  secret_public: boolean
  public_url: string
}

export const getVillageMeta = () =>
  fdFetch<VillageMeta>('/beings/village-meta')

export const setVillageMeta = (description: string, name: string) =>
  fdFetch<{ name: string; description: string }>('/beings/village-meta', {
    method: 'POST', body: JSON.stringify({ description, name }),
  })

export const recommendVillageMeta = (being: string) =>
  fdFetch<{ description: string; by: string; by_slug: string }>(
    '/beings/village-meta/recommend', {
      method: 'POST', body: JSON.stringify({ being }),
    })

// ── Federation: hosting visitors + sending beings out ──

export const setVillageFederation = (
  secret: string, secretPublic: boolean, publicUrl: string,
) =>
  fdFetch<VillageMeta>('/beings/village-federation', {
    method: 'POST',
    body: JSON.stringify({ secret, secret_public: secretPublic, public_url: publicUrl }),
  })

export interface Visitor {
  id: string; origin: string; slug: string; name: string
  first_seen: string; last_seen: string
}
export const getVisitors = () =>
  fdFetch<{ visitors: Visitor[] }>('/beings/visitors')
export const removeVisitor = (id: string) =>
  fdFetch<{ ok: boolean }>(`/beings/visitors/${id}`, { method: 'DELETE' })

export const setBeingVisit = (slug: string, url: string, secret: string) =>
  fdFetch<{ vitals: BeingVitals; announced: { ok: boolean | null; error?: string } }>(
    `/beings/${slug}/visit`, {
      method: 'POST', body: JSON.stringify({ url, secret }),
    })
// The map of the village this being is visiting, proxied down its link with
// the guest positioned in it (visiting-beings plan §2).
export const getVisitedMap = (slug: string) =>
  fdFetch<VillageMapData>(`/beings/${slug}/visit/map`)
// Walk the visiting being to a place of the village it visits (§2).
export const nudgeVisit = (slug: string, place: string) =>
  fdFetch<{ ok: boolean; to?: string; at?: string; walking?: boolean; minutes?: number }>(
    `/beings/${slug}/visit/nudge`, { method: 'POST', body: JSON.stringify({ place }) })
