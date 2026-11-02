/**
 * Forge agents — say what you need and attach files; Captain plans a team with
 * Vatra's planner, you approve it, then it recommends scheduled tasks and apps
 * for Crew to build (`captain_claw/flight_deck/agent_forge.py`).
 */

import { refreshAccessToken, useAuthStore } from '../stores/authStore'
import type { CrewModelSource, CrewStatus } from './crewApi'
import { api } from './fileLibraryApi'

export type ForgeStatus = 'planning' | 'awaiting_plan' | 'creating' | 'recommending' | 'ready' | 'error'

export interface ForgeDraftAgent {
  id: string
  name: string
  archetype: string
  mandate: string
  produces: string
  consumes_from: string[]
  hand_off_notes: string
}

export interface ForgeQuestion {
  id: string
  question: string
  multi: boolean
  options: string[]
  selected: string[]
  other: string
}

export interface ForgeCreated {
  id: string
  name: string
  slug: string
  archetype: string
  role: string
  ok: boolean
  error: string
  /** Only with sign-in off, where the browser keeps instructions. */
  instructions?: string
}

export interface ForgeDelivery {
  chat: boolean
  email: boolean
  telegram: boolean
  email_to?: string
}

export interface ForgeSchedule {
  id: string
  name: string
  agent: string
  schedule: string
  schedule_problem: string
  prompt: string
  why: string
  usage: string
  delivery: ForgeDelivery
  job_id?: string
}

export type ForgeConnect = 'both' | 'api' | 'wss'

export interface ForgeAppTask {
  id: string
  status: CrewStatus | 'gone'
  note: string
  kind?: string
  stalled?: boolean
  evidence?: { commits?: number; dirty?: number }
  branch?: string
  project_name?: string
}

export interface ForgeConnection {
  url: string
  socket: string
  agents: string[]
  api_calls: number
}

export interface ForgeApp {
  id: string
  name: string
  slug: string
  summary: string
  why: string
  brief: string
  agents: string[]
  agent_use: Record<string, string>
  endpoints: { method: string; path: string; purpose: string }[]
  events: { name: string; purpose: string }[]
  connect: ForgeConnect
  task_id?: string
  task?: ForgeAppTask | null
  connection?: ForgeConnection
}

export interface ForgeRecommendations {
  usage: string
  schedules: ForgeSchedule[]
  apps: ForgeApp[]
}

export interface ForgePlan {
  id: string
  title: string
  request: string
  status: ForgeStatus
  /** It was working when Flight Deck restarted, and nothing is carrying it on. */
  stalled: boolean
  error: string
  recommend_error: string
  files: { name: string; size: number }[]
  team: string
  /** The most agents this plan could have: the person's limit, or the default, within their seats. */
  room: number
  /** How many more agents the person's plan had room for when this was planned. */
  seats: number
  /** The limit the person set; 0 leaves it to Captain. */
  cap: number
  default_team: number
  max_team: number
  /** Agents the Lead proposed that the limit left out. */
  over_cap: number
  rationale: string
  overview: string
  agents: ForgeDraftAgent[]
  questions: ForgeQuestion[]
  /** The team as it was approved, before and while it's created. */
  approved: { id: string; name: string; archetype: string }[]
  created: ForgeCreated[]
  recommendations: ForgeRecommendations | null
}

export interface ForgePlanRow {
  id: string
  title: string
  status: ForgeStatus
  stalled: boolean
  updated_at: string
  created: number
}

export interface ForgeFeedEvent {
  ts?: number
  stage: string
  message: string
  ok?: boolean
  /** On a phase: which step of the work it is. */
  step?: ForgeWorkStep
  /** The agents the Lead just chose. */
  agents?: { name: string; archetype: string }[]
}

export type ForgeWorkStep = 'read' | 'lead' | 'plan' | 'create' | 'recommend'

export interface ForgeConnectResult {
  slug: string
  name: string
  linked: boolean
  api_calls: number
  error: string
}

const post = <T>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', ...(body === undefined ? {} : { body: JSON.stringify(body) }) })

/** What the Team size control can offer: `seats` are what the person's plan has left. */
export interface ForgeLimits { seats: number; default_team: number; max_team: number }

const plans = '/fd/agent-forge/plans'

/** A multipart body, so not through api(), which says it's JSON. */
async function start(request: string, files: File[], maxAgents = 0): Promise<{ id: string }> {
  const send = () => {
    const { token } = useAuthStore.getState()
    const form = new FormData()
    form.append('request', request)
    form.append('max_agents', String(maxAgents))
    for (const f of files) form.append('files', f)
    return fetch(plans, { method: 'POST', body: form, credentials: 'include', headers: token ? { Authorization: `Bearer ${token}` } : {} })
  }
  let r = await send()
  if (r.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) r = await send()
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = typeof body?.detail === 'string' ? body.detail : ''
    } catch { /* not JSON */ }
    throw new Error(detail || (r.status === 413 ? 'Those files are too big.' : `${r.status} ${r.statusText}`))
  }
  return r.json()
}

export const forgeApi = {
  start,
  list: () => api<{ plans: ForgePlanRow[] }>(plans).then((r) => r.plans),
  get: (id: string) => api<ForgePlan>(`${plans}/${id}`),
  feed: (id: string) => api<{ events: ForgeFeedEvent[]; active: boolean }>(`/fd/basna/sessions/${id}/progress`),
  limits: () => api<ForgeLimits>('/fd/agent-forge/limits'),
  replan: (id: string, note: string, questions: ForgeQuestion[], agents: ForgeDraftAgent[] = [], maxAgents?: number) =>
    post<{ status: ForgeStatus }>(`${plans}/${id}/replan`, { note, questions, agents, max_agents: maxAgents }),
  approve: (id: string, agents: ForgeDraftAgent[], overview: string) => post<{ status: ForgeStatus }>(`${plans}/${id}/approve`, { agents, overview }),
  recommend: (id: string) => post<{ status: ForgeStatus }>(`${plans}/${id}/recommend`),
  schedules: (id: string, items: ForgeSchedule[], tz: string) =>
    post<{ results: { id: string; ok: boolean; error?: string; job_id?: string }[] }>(`${plans}/${id}/schedules`, { items, tz }),
  build: (id: string, app: string, body: { harness: string; brief: string; connect: ForgeConnect; model?: string; model_source?: CrewModelSource }) =>
    post<{ task: ForgeAppTask }>(`${plans}/${id}/apps/${app}/build`, body),
  connect: (id: string, app: string, body: { url: string; key: string; agents: string[] }) =>
    post<{ results: ForgeConnectResult[]; instructions: Record<string, string>; connection: ForgeConnection }>(`${plans}/${id}/apps/${app}/connect`, body),
  forget: (id: string) => api<{ ok: boolean }>(`${plans}/${id}`, { method: 'DELETE' }),
}
