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

export interface BeingVitals {
  slug: string
  name: string
  stage: string
  state: string
  born_at: string
  hatched_at: string | null
  died_at: string | null
  attention_credits: number
  attributes: Record<string, number>
  derived: {
    drive_weights: Record<string, number>
    risk_appetite: number
    whimsy: number
    thrift: number
    [k: string]: unknown
  }
  generation: number
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
  media_diet: { allow?: string[]; deny?: string[] }
  affect: { mood?: string; notes?: string[] }
  persona: string
  pending_self_mod: { content: string; reason: string; proposed_at: string } | null
}

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

export const approveSelfMod = (slug: string) =>
  fdFetch<{ persona: string }>(`/beings/${slug}/self-mod/approve`, { method: 'POST' })
export const rejectSelfMod = (slug: string, note = '') =>
  fdFetch<{ ok: boolean }>(`/beings/${slug}/self-mod/reject`, {
    method: 'POST', body: JSON.stringify({ note }),
  })
export const rollbackPersona = (slug: string) =>
  fdFetch<{ persona: string }>(`/beings/${slug}/self-mod/rollback`, { method: 'POST' })

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
}

export const postChore = (slug: string, spec: string, fee_tokens: number) =>
  fdFetch<{ chore: Chore }>(`/beings/${slug}/chores`, {
    method: 'POST', body: JSON.stringify({ spec, fee_tokens }),
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
export const setStage = (slug: string, stage: string) =>
  fdFetch<BeingVitals>(`/beings/${slug}/stage`, {
    method: 'POST', body: JSON.stringify({ stage }),
  })
