import { create } from 'zustand'

// Team up (captain_claw/team_up.py): an agent splits a task among helpers —
// copies of itself — that work in the background while the person keeps
// chatting with it. The agent sends `{type: "team", event, team_id, …}` frames
// on the chat lane; this store keeps each lane's teams for the transcript card,
// the conversation bar's pill and the Team panel.

export type TeamStepKind = 'narration' | 'tool' | 'llm' | 'note' | 'tell' | 'file'

export interface TeamStep {
  t: number
  kind: TeamStepKind
  text: string
  tool?: string
}

export type HelperState = 'waiting' | 'working' | 'stopping' | 'done' | 'failed' | 'stopped'

export interface TeamHelper {
  id: string
  name: string
  goal: string
  produces: string
  state: HelperState
  session_id: string
  step_count: number
  tool_count: number
  tokens: number
  last: string
  error: string
  draft: string
  result: string
  files: string[]
  started_at: number
  ended_at: number
  wrap_up: boolean
  stopped_by: string
  model: string
  steps: TeamStep[]
}

export type TeamState = 'proposed' | 'running' | 'finishing' | 'done' | 'stopped' | 'dismissed' | 'failed'

export interface TeamRun {
  id: string
  title: string
  lane: string
  mode: string
  state: TeamState
  created_at: number
  started_at: number
  ended_at: number
  budget: number
  used: number
  warned: boolean
  browser_holder: string
  note: string
  helpers: TeamHelper[]
  /** Runtime only: the last refusal of a tap (shown on the card). */
  error?: string
  /** Runtime only: a tap sent and not answered yet. */
  pending?: 'start' | 'dismiss' | 'stop' | ''
}

const STEPS_KEPT = 200

export const LIVE_TEAM: TeamState[] = ['proposed', 'running', 'finishing']
export const LIVE_HELPER: HelperState[] = ['waiting', 'working', 'stopping']

/** One colour per helper, in order — the same on the pill, the card and the panel. */
export const HELPER_TONES = [
  { dot: 'bg-violet-500', ring: 'ring-violet-500/40', text: 'text-violet-700 dark:text-violet-300', soft: 'bg-violet-500/15', bar: 'bg-violet-500' },
  { dot: 'bg-sky-500', ring: 'ring-sky-500/40', text: 'text-sky-700 dark:text-sky-300', soft: 'bg-sky-500/15', bar: 'bg-sky-500' },
  { dot: 'bg-emerald-500', ring: 'ring-emerald-500/40', text: 'text-emerald-700 dark:text-emerald-300', soft: 'bg-emerald-500/15', bar: 'bg-emerald-500' },
  { dot: 'bg-amber-500', ring: 'ring-amber-500/40', text: 'text-amber-700 dark:text-amber-300', soft: 'bg-amber-500/15', bar: 'bg-amber-500' },
]

export function helperTone(index: number) {
  return HELPER_TONES[index % HELPER_TONES.length]
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : v == null ? '' : String(v)
}

function num(v: unknown): number {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? n : 0
}

function toStep(raw: unknown): TeamStep | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  const text = str(r.text)
  if (!text) return null
  return { t: num(r.t), kind: (str(r.kind) || 'note') as TeamStepKind, text, ...(r.tool ? { tool: str(r.tool) } : {}) }
}

function toHelper(raw: Record<string, unknown>, prev?: TeamHelper): TeamHelper {
  const steps = Array.isArray(raw.steps)
    ? (raw.steps as unknown[]).map(toStep).filter((s): s is TeamStep => s !== null)
    : prev?.steps ?? []
  return {
    id: str(raw.id),
    name: str(raw.name) || prev?.name || 'Helper',
    goal: str(raw.goal) || prev?.goal || '',
    produces: str(raw.produces),
    state: (str(raw.state) || prev?.state || 'waiting') as HelperState,
    session_id: str(raw.session_id),
    step_count: num(raw.step_count),
    tool_count: num(raw.tool_count),
    tokens: num(raw.tokens),
    last: str(raw.last),
    error: str(raw.error),
    draft: str(raw.draft),
    result: str(raw.result) || prev?.result || '',
    files: Array.isArray(raw.files) ? (raw.files as unknown[]).map(str) : prev?.files ?? [],
    started_at: num(raw.started_at),
    ended_at: num(raw.ended_at),
    wrap_up: !!raw.wrap_up,
    stopped_by: str(raw.stopped_by),
    model: str(raw.model),
    steps: steps.slice(-STEPS_KEPT),
  }
}

function toTeam(raw: Record<string, unknown>, prev?: TeamRun): TeamRun {
  const helpers = Array.isArray(raw.helpers) ? (raw.helpers as Record<string, unknown>[]) : []
  return {
    id: str(raw.id),
    title: str(raw.title) || prev?.title || 'Team',
    lane: str(raw.lane),
    mode: str(raw.mode),
    state: (str(raw.state) || prev?.state || 'proposed') as TeamState,
    created_at: num(raw.created_at),
    started_at: num(raw.started_at),
    ended_at: num(raw.ended_at),
    budget: num(raw.budget),
    used: num(raw.used),
    warned: !!raw.warned,
    browser_holder: str(raw.browser_holder),
    note: str(raw.note),
    helpers: helpers.map((h) => toHelper(h, prev?.helpers.find((p) => p.id === str(h.id)))),
    // A tap is answered by the state it asked for (or by a refusal).
    pending: prev?.pending && prev.state === str(raw.state) ? prev.pending : '',
    error: prev?.state === str(raw.state) ? prev?.error : undefined,
  }
}

interface TeamStoreState {
  /** chat key (agent + lane) → team id → team */
  byKey: Record<string, Record<string, TeamRun>>
  /** The Team panel: which chat and team it shows (null = closed). */
  panel: { key: string; teamId: string } | null
  apply: (key: string, frame: Record<string, unknown>) => void
  openPanel: (key: string, teamId: string) => void
  closePanel: () => void
  setPending: (key: string, teamId: string, pending: TeamRun['pending']) => void
}

export const useTeamStore = create<TeamStoreState>((set) => ({
  byKey: {},
  panel: null,

  apply: (key, frame) => set((state) => {
    const event = str(frame.event)
    const teams = { ...(state.byKey[key] ?? {}) }
    const teamId = str(frame.team_id)
    const patchHelper = (fn: (h: TeamHelper) => TeamHelper): boolean => {
      const team = teams[teamId]
      const hid = str(frame.helper_id || (frame.helper as Record<string, unknown> | undefined)?.id)
      if (!team || !hid) return false
      teams[teamId] = { ...team, helpers: team.helpers.map((h) => (h.id === hid ? fn(h) : h)) }
      return true
    }

    switch (event) {
      case 'proposed':
      case 'started':
      case 'team':
      case 'snapshot': {
        const raw = frame.team as Record<string, unknown> | undefined
        if (!raw || !str(raw.id)) return state
        teams[str(raw.id)] = toTeam(raw, teams[str(raw.id)])
        break
      }
      case 'sync': {
        const list = Array.isArray(frame.teams) ? (frame.teams as Record<string, unknown>[]) : []
        const seen = new Set<string>()
        for (const raw of list) {
          const id = str(raw.id)
          if (!id) continue
          seen.add(id)
          teams[id] = toTeam(raw, teams[id])
        }
        // A live team the agent no longer knows: it restarted and the team ended with it.
        for (const [id, team] of Object.entries(teams)) {
          if (!seen.has(id) && LIVE_TEAM.includes(team.state)) {
            teams[id] = {
              ...team, state: 'stopped', note: 'The agent restarted, so this team ended.',
              helpers: team.helpers.map((h) => (LIVE_HELPER.includes(h.state) ? { ...h, state: 'stopped' } : h)),
            }
          }
        }
        break
      }
      case 'helper': {
        const raw = frame.helper as Record<string, unknown> | undefined
        if (!raw) return state
        if (!patchHelper((h) => toHelper(raw, h))) return state
        const team = teams[teamId]
        teams[teamId] = {
          ...team,
          used: num(frame.used) || team.used,
          budget: num(frame.budget) || team.budget,
          browser_holder: str(frame.browser_holder),
          pending: team.pending === 'stop' ? '' : team.pending,
        }
        break
      }
      case 'step': {
        const step = toStep(frame.step)
        if (!step) return state
        const ok = patchHelper((h) => ({
          ...h,
          steps: [...h.steps, step].slice(-STEPS_KEPT),
          step_count: num(frame.step_count) || h.step_count + 1,
          tool_count: num(frame.tool_count) || h.tool_count,
          last: step.kind === 'llm' ? h.last : step.text,
          draft: step.kind === 'tool' ? '' : h.draft,
        }))
        if (!ok) return state
        break
      }
      case 'draft': {
        if (!patchHelper((h) => ({ ...h, draft: str(frame.draft) }))) return state
        break
      }
      case 'error': {
        const team = teams[teamId]
        if (!team) return state
        teams[teamId] = { ...team, error: str(frame.message) || 'That didn’t work.', pending: '' }
        break
      }
      default:
        return state
    }
    return { byKey: { ...state.byKey, [key]: teams } }
  }),

  openPanel: (key, teamId) => set({ panel: { key, teamId } }),
  closePanel: () => set({ panel: null }),
  setPending: (key, teamId, pending) => set((state) => {
    const team = state.byKey[key]?.[teamId]
    if (!team) return state
    return { byKey: { ...state.byKey, [key]: { ...state.byKey[key], [teamId]: { ...team, pending, error: undefined } } } }
  }),
}))

/** The lane's team the pill and panel follow: the live one, else the newest
 *  that ended in the last 15 minutes. */
export function currentTeam(teams: Record<string, TeamRun> | undefined): TeamRun | null {
  if (!teams) return null
  const list = Object.values(teams).filter((t) => t.state !== 'dismissed')
  const live = list.filter((t) => LIVE_TEAM.includes(t.state)).sort((a, b) => b.created_at - a.created_at)
  if (live.length) return live[0]
  const recent = list
    .filter((t) => (t.ended_at || t.created_at) * 1000 > Date.now() - 15 * 60_000)
    .sort((a, b) => (b.ended_at || b.created_at) - (a.ended_at || a.created_at))
  return recent[0] ?? null
}

export function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1000) return `${Math.round(n / 1000)}k`
  return String(Math.max(0, Math.round(n)))
}

export function fmtElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}:${String(s % 60).padStart(2, '0')}`
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`
}

/** A team's result note in the transcript ("[Team 'x' finished] …"). */
export const TEAM_RESULT_RE = /^\[Team '(.*?)' (finished|stopped)\]/
