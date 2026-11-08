// REST client for Course (/fd/course*, /fd/admin/course/settings): the goals
// the deck, the user and each of their agents work toward. Flight Deck writes
// the live ones into every agent's prompt; see docs/course-plan.md.
//
// These calls never carry the admin act-as header: a course is only ever the
// signed-in user's own (plus the deck's).

import { fdFetch } from './notifications'

export type Scope = 'instance' | 'user' | 'agent'
export type Horizon = 'strategic' | 'long' | 'medium' | 'short'
export type GoalStatus = 'proposed' | 'active' | 'paused' | 'achieved' | 'dropped' | 'superseded' | 'declined'
export type GoalAction = 'pause' | 'resume' | 'achieve' | 'drop' | 'reopen' | 'review' | 'accept' | 'decline'

export const HORIZONS: Horizon[] = ['strategic', 'long', 'medium', 'short']
export const SCOPES: Scope[] = ['instance', 'user', 'agent']

export const HORIZON_LABELS: Record<Horizon, string> = {
  strategic: 'Strategic', long: 'Long-term', medium: 'Medium-term', short: 'Short-term',
}

/** A word on what each horizon is for, shown where one is chosen. */
export const HORIZON_HINTS: Record<Horizon, string> = {
  strategic: 'A direction to keep. Never "done", only revised.',
  long: 'An outcome about a quarter or more away.',
  medium: 'An outcome a few weeks away.',
  short: 'This week’s outcome. Keep these few.',
}

export const SCOPE_LABELS: Record<Scope, string> = {
  instance: 'The deck', user: 'Each user', agent: 'Each agent',
}

export interface GoalFlags {
  due: boolean
  stale: boolean
  horizon_retired: boolean
  serves_nothing: boolean
  parent_gone: boolean
}

export interface Goal {
  id: string
  scope: Scope
  owner_id: string
  agent_ref: string
  horizon: Horizon
  parent_id: string
  title: string
  detail: string
  success: string
  status: GoalStatus
  origin: string
  review_every_days: number
  review_at: string | null
  due_at: string | null
  created_at: string
  updated_at: string
  can_edit?: boolean
  flags?: GoalFlags
  grace_days?: number
}

export interface CourseSettings {
  horizons: Record<Scope, Horizon[]>
  caps: Record<Horizon, number>
}

export interface CourseAgent {
  ref: string
  name: string
  slug: string
  runtime: string
}

export interface CourseResponse {
  settings: CourseSettings
  horizons: Horizon[]
  horizon_labels: Record<Horizon, string>
  review_days: Record<Horizon, number>
  text_caps: { title: number; detail: number; success: number }
  is_admin: boolean
  auth_enabled: boolean
  agents: CourseAgent[]
  goals: { deck: Goal[]; mine: Goal[]; agents: Record<string, Goal[]> }
}

export interface GoalEvent {
  id: number
  at: string
  actor: string
  event: string
  detail: Record<string, unknown>
}

export interface GoalDetail {
  goal: Goal
  links: { kind: string; ref_id: string; note: string; at: string }[]
  events: GoalEvent[]
  can_edit: boolean
}

export interface CoursePreview {
  owner: { full: string; compact: string }
  member: { full: string; compact: string }
}

export interface GoalDraft {
  horizon: Horizon
  title: string
  success: string
  detail: string
  parent_id: string
  due_at: string       // YYYY-MM-DD or ''
  review_every_days: number
}

export interface SettingsResponse extends CourseSettings {
  defaults: CourseSettings
  cap_range: [number, number]
}

export const DEFAULT_TEXT_CAPS = { title: 120, detail: 1000, success: 500 }
export const DEFAULT_REVIEW_DAYS: Record<Horizon, number> = { short: 7, medium: 30, long: 90, strategic: 365 }

/** Fired to open Course: the kiosk opens its dialog, the full layout its page. */
export const OPEN_COURSE_EVENT = 'fd:open-course'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fdFetch(path, init)
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }))
    let msg = body?.detail
    if (Array.isArray(msg)) {
      msg = msg.map((e: { loc?: string[]; msg?: string }) => `${(e.loc || []).join('.')}: ${e.msg}`).join('; ')
    }
    throw new Error(typeof msg === 'string' && msg ? msg : `HTTP ${r.status}`)
  }
  return r.json()
}

const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) })

export const getCourse = () => call<CourseResponse>('/fd/course')

export const getPreview = (agentRef = '') =>
  call<CoursePreview>(`/fd/course/preview${agentRef ? `?agent_ref=${encodeURIComponent(agentRef)}` : ''}`)

export const getGoal = (id: string) => call<GoalDetail>(`/fd/course/goals/${encodeURIComponent(id)}`)

function draftBody(d: GoalDraft) {
  return {
    horizon: d.horizon,
    title: d.title,
    success: d.success,
    detail: d.detail,
    parent_id: d.parent_id,
    due_at: d.due_at ? `${d.due_at}T00:00:00Z` : null,
    review_every_days: d.review_every_days,
  }
}

export const createGoal = (scope: Scope, d: GoalDraft, agentRef = '') =>
  call<{ goal: Goal; agents_updated: number }>('/fd/course/goals',
    json('POST', { scope, agent_ref: agentRef, ...draftBody(d) }))

export const editGoal = (id: string, d: GoalDraft) =>
  call<{ goal: Goal; agents_updated: number }>(`/fd/course/goals/${encodeURIComponent(id)}`,
    json('PATCH', draftBody(d)))

export const goalAction = (id: string, action: GoalAction) =>
  call<{ goal: Goal; agents_updated: number }>(
    `/fd/course/goals/${encodeURIComponent(id)}/${action}`, { method: 'POST' })

export const getSettings = () => call<SettingsResponse>('/fd/admin/course/settings')

export const saveSettings = (s: Partial<CourseSettings>) =>
  call<CourseSettings>('/fd/admin/course/settings', json('PUT', s))

// ── Helpers ──

export function emptyDraft(horizon: Horizon, parentId = ''): GoalDraft {
  return {
    horizon, title: '', success: '', detail: '', parent_id: parentId, due_at: '',
    review_every_days: DEFAULT_REVIEW_DAYS[horizon],
  }
}

export function draftOf(g: Goal): GoalDraft {
  return {
    horizon: g.horizon,
    title: g.title,
    success: g.success,
    detail: g.detail,
    parent_id: g.parent_id,
    due_at: g.due_at ? g.due_at.slice(0, 10) : '',
    review_every_days: g.review_every_days,
  }
}

const RANK: Record<Horizon, number> = { strategic: 0, long: 1, medium: 2, short: 3 }
const PARENT_SCOPES: Record<Scope, Scope[]> = {
  instance: ['instance'], user: ['user', 'instance'], agent: ['agent', 'user', 'instance'],
}

/** The goals a goal of `scope`/`horizon` may serve, among those shown: an
 *  active goal in its own scope or a wider one (the same agent for an agent
 *  goal), of the same horizon or a longer one, and never itself or one that
 *  serves it. The server checks the same rules. */
export function eligibleParents(all: Goal[], scope: Scope, horizon: Horizon, agentRef: string,
  selfId = ''): Goal[] {
  const below = new Set<string>()
  if (selfId) {
    // Everything under `selfId` can't become its parent (no cycles).
    let grew = true
    below.add(selfId)
    while (grew) {
      grew = false
      for (const g of all) {
        if (g.parent_id && below.has(g.parent_id) && !below.has(g.id)) { below.add(g.id); grew = true }
      }
    }
  }
  return all.filter((g) => g.status === 'active'
    && PARENT_SCOPES[scope].includes(g.scope)
    && (g.scope !== 'agent' || g.agent_ref === agentRef)
    && RANK[g.horizon] <= RANK[horizon]
    && !below.has(g.id))
}

export function agentsUpdatedLabel(n: number | undefined): string {
  const count = n ?? 0
  return `${count} agent${count === 1 ? '' : 's'} updated`
}

export function shortDate(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}
