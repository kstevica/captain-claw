// REST client for First Mate (/fd/first-mate*, /fd/admin/first-mate/settings):
// factories, standing jobs that run for the user. Each has a charter, a runner
// (Bat so far), triggers, a budget and checks; every run uses its approved
// version. See docs/first-mate-plan.md.
//
// These calls never carry the admin act-as header: a factory is only ever the
// signed-in user's own.

import { fdFetch } from './notifications'

export type FactoryState = 'draft' | 'awaiting_charter' | 'running' | 'paused' | 'paused_budget' | 'stopped' | 'retired'
export type VersionStatus = 'draft' | 'awaiting' | 'live' | 'replaced' | 'declined'
export type Trust = 'propose' | 'commit' | 'direct'
export type Runner = 'bat' | 'vatra' | 'basna' | 'code' | 'crew' | 'flow' | 'plan'
export type RunStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
export type OwnerState = 'running' | 'paused' | 'stopped' | 'retired'

export const STATE_LABELS: Record<FactoryState, string> = {
  draft: 'Draft', awaiting_charter: 'Waiting for approval', running: 'Running', paused: 'Paused',
  paused_budget: 'Over budget', stopped: 'Stopped', retired: 'Retired',
}

export const TRUST_LABELS: Record<Trust, string> = {
  propose: 'Asks first', commit: 'Acts within its grants', direct: 'Acts directly',
}

export const TRUST_HINTS: Record<Trust, string> = {
  propose: 'Every outward action waits for your yes.',
  commit: 'Rehearses, then does what its grants allow; asks about the rest.',
  direct: 'Acts without rehearsing. Only for reads, its own files and actions that can be undone.',
}

export const RUNNER_LABELS: Record<Runner, string> = {
  bat: 'Bat', vatra: 'Vatra', basna: 'Basna', code: 'Code', crew: 'Crew', flow: 'Flow', plan: 'Plan',
}

export const RUN_LABELS: Record<RunStatus, string> = {
  queued: 'Starting', running: 'Running', done: 'Done', failed: 'Failed', cancelled: 'Stopped',
}

export interface ScheduleTrigger { kind: 'schedule'; schedule: string; tz: string }
export interface EventsTrigger {
  kind: 'events'
  source: string
  event_type: string
  match: { from: string[]; label: string[]; words: string[] }
  batch_minutes: number | null
  ask: boolean
}
export type Trigger = ScheduleTrigger | EventsTrigger

export interface Grants {
  email: boolean
  organize: boolean
  spend: boolean
  accounts: boolean
  sends_per_run: number
  replies_only: boolean
  real_usd_week: number
  per_item_usd: number
}

export interface Version {
  factory_id: string
  version: number
  status: VersionStatus
  charter: string
  runner: Runner
  runner_config: Record<string, unknown>
  triggers: Trigger[]
  grants: Grants
  budget: { run_tokens: number | null }
  checks: { contract: string[]; satisfaction_target: number }
  sha: string
  note: string
  created_by: string
  created_at: string
  updated_at: string
  decided_by: string
  decided_at: string | null
}

export interface Intent {
  id: number
  /** intent: an outward call recorded; commit: the browser stopped before submitting. */
  kind: 'intent' | 'commit'
  tool: string
  what: string
  status: 'open' | 'approved' | 'rejected' | 'carried_out' | 'dropped'
  /** What it would have sent or changed, from its arguments. */
  detail: string
}

export interface Run {
  id: string
  factory_id: string
  version: number
  kind: string
  trigger: string
  runner: Runner
  child_id: string
  status: RunStatus
  verdict: string
  result: string
  error: string
  tokens: number
  usd: number
  created_at: string
  started_at: string | null
  ended_at: string | null
  /** Set while it waits on the owner (a Bat plan or question). */
  waiting?: boolean
  note?: string
  /** A rehearsal's: what it would have done. */
  intents?: Intent[]
  /** The owner's 👍/👎 on its result, if any. */
  thumbs?: '' | 'thumbs_up' | 'thumbs_down'
  /** A carry-out's: what the register and Gmail/Calendar showed it did. */
  proof?: { state?: 'confirmed' | 'nothing' | 'doubt' | 'unknown'; note?: string; sent?: number; approved?: number; actions?: string[] }
}

export interface Factory {
  id: string
  owner_id: string
  title: string
  state: FactoryState
  state_reason: string
  version: number
  draft_version: number
  trust: Trust
  trust_pinned: boolean
  share: number
  priority: number
  goal_id: string
  template: string
  created_at: string
  updated_at: string
  live: Version | null
  draft: Version | null
}

export interface FactoryCard extends Factory {
  last_run: Run | null
  /** When its schedules next fire (ISO); past = due now. */
  next_at: string | null
  /** Tokens its runs used this week, and its share of the owner's weekly budget. */
  week_tokens: number
  share_tokens: number
  /** Events claimed for it, waiting for their batch. */
  pending_events: number
  /** The newest rehearsal whose intents wait for the owner's yes. */
  proposal: { run_id: string; count: number } | null
  health: Health
  health_why: string
  satisfaction: number | null
}

export type Health = 'green' | 'amber' | 'red' | 'new'

export const HEALTH_LABELS: Record<Health, string> = {
  green: 'Healthy', amber: 'Needs a look', red: 'In trouble', new: 'Too new to say',
}

export interface Scorecard {
  health: Health
  health_why: string
  week: { runs: number; done: number; judged: number; passed: number; tokens_per_run: number; median_minutes: number | null }
  fortnight: { carry_outs: number; doubts: number }
  satisfaction: number | null
  signals: number
  target: number
  drafts: Record<'waiting' | 'sent' | 'edited' | 'deleted' | 'expired', number>
  cases: number
  check: Pick<Check, 'id' | 'purpose' | 'version' | 'version_sha' | 'status' | 'score' | 'note' | 'created_at' | 'ended_at'> | null
}

/** An input the factory is tested on, and what a good result does. */
export interface Case {
  id: string
  title: string
  input: string
  expect: string[]
  /** Never shown to the factory, or to anything that rewrites its charter. */
  sealed: boolean
  origin: 'owner' | 'run'
  from_run: string
  created_at: string
}

export type CheckPurpose = 'approve' | 'baseline' | 'weekly' | 'manual'

/** One version run on every case, graded by a reviewer on another model. */
export interface Check {
  id: string
  factory_id: string
  version: number
  version_sha: string
  purpose: CheckPurpose
  status: 'running' | 'done' | 'void' | 'failed'
  cases: string[]
  results: Record<string, { run_id?: string; score: number | null; note?: string }>
  baseline: number | null
  score: number | null
  note: string
  created_at: string
  ended_at: string | null
}

export const CHECK_PURPOSE_LABELS: Record<CheckPurpose, string> = {
  approve: 'Before going live', baseline: 'First score', weekly: 'Weekly', manual: 'By you',
}

export interface CaseFields { title: string; input: string; expect: string[]; sealed: boolean }

export interface Signal { id: number; run_id: string; kind: string; value: number; source: string; at: string }

export const SIGNAL_LABELS: Record<string, string> = {
  approved: 'You approved what it would do', rejected: 'You rejected what it would do',
  thumbs_up: 'You gave it 👍', thumbs_down: 'You gave it 👎',
  sent_as_written: 'You sent its draft as written', sent_edited: 'You sent its draft after edits',
  draft_deleted: 'You deleted its draft', done: 'Marked done', not_done: 'Marked not done',
  reopened: 'Reopened', undone: 'Undone',
}

export interface FactoryEvent {
  id: number
  at: string
  actor: string
  event: string
  detail: Record<string, unknown>
}

export interface FactoryDetail extends Factory {
  versions: Version[]
  runs: Run[]
  events: FactoryEvent[]
  satisfaction: number | null
  scorecard: Scorecard
  signals: Signal[]
  drafts: { id: number; run_id: string; subject: string; status: string; similarity: number | null; created_at: string }[]
  /** The check going, if any. */
  check: Check | null
  ladder: Ladder
}

/** How well the deck's own verdict on a result agrees with the owner's labels (👍/👎, approve/reject, drafts). */
export interface Calibration {
  n: number
  kappa: number | null
  agreement?: number | null
  /** Calibrated: trust may move on its own. */
  ready: boolean
  why: string
  labels_bar: number
  kappa_bar: number
}

/** A factory's record at its trust level, and whether it moves up on its own. */
export interface Ladder {
  level: Trust
  next: Trust | null
  counted: number
  needed: number
  value: number | null
  bar: number
  doubts: number
  pinned: boolean
  eligible: boolean
  why: string
  calibration: Calibration
}

export interface DeckSettings { enabled: boolean; admins_only: boolean; max_factories: number; runs_per_day: number }
export interface Prefs { on: boolean; weekly_tokens: number; run_tokens: number; batch_minutes: number; agent: string }

export interface Status {
  available: boolean
  reason: string
  is_admin: boolean
  settings: DeckSettings
  prefs: Prefs
  counts: { factories: number; running: number; runs_going: number }
  week: { start: string; budget: number; used: number }
  calibration: Calibration | null
}

/** One factory's week in the report. */
export interface WeekLine {
  id: string
  title: string
  state: FactoryState
  trust: Trust
  health: Health
  health_why: string
  runs: number
  done: number
  carry_outs: number
  doubts: number
  tokens: number
  share: number
  signals: number
  satisfaction: number | null
  drafts: { sent: number; edited: number; deleted: number }
  cases_score: number | null
  moved: string[]
  look: string[]
}

/** "First Mate's week": last week (what Monday's Line item is about) or this week so far. */
export interface WeekReport {
  start: string
  end: string
  title: string
  text: string
  totals: { runs: number; done: number; carry_outs: number; doubts: number; tokens: number; budget: number
    drafts: { sent: number; edited: number; deleted: number } }
  factories: WeekLine[]
  look: string[]
  sent: { week?: string; at?: string; sent?: boolean }
}

/** Fired to open First Mate: the kiosk opens its dialog, the full layout its page. */
export const OPEN_FIRST_MATE_EVENT = 'fd:open-first-mate'

export interface Template {
  id: string
  title: string
  blurb: string
  charter: string
  triggers: Trigger[]
  checks: { contract: string[] }
}

/** What the owner edits in a draft. */
export interface DraftFields {
  charter?: string
  triggers?: Trigger[]
  checks?: { contract: string[]; satisfaction_target?: number }
  budget?: { run_tokens: number | null }
  note?: string
}

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

const json = (method: string, body: unknown): RequestInit => ({
  method, body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' },
})
const f = (id: string) => `/fd/first-mate/factories/${encodeURIComponent(id)}`

export const getStatus = () => call<Status>('/fd/first-mate')
export const savePrefs = (prefs: Partial<Prefs>) => call<Prefs>('/fd/first-mate/prefs', json('PUT', prefs))
export const saveDeckSettings = (s: Partial<DeckSettings>) =>
  call<DeckSettings>('/fd/admin/first-mate/settings', json('PUT', s))
/** The agent that is the owner's First Mate: where they talk to it, and whose model drafts for it. */
export interface MyAgent {
  current: { slug: string; name: string; exists: boolean; running: boolean }
  agents: { slug: string; name: string }[]
}
export const getMyAgent = () => call<MyAgent>('/fd/first-mate/my-agent')
export const makeMyAgent = () => call<MyAgent & { slug: string; name: string }>('/fd/first-mate/my-agent', { method: 'POST' })

/** A new factory drafted by a model from the owner's words or one of their Course goals; it waits for approval. */
export const draftFactory = (body: { words?: string; goal_id?: string }) =>
  call<Factory & { cases: number }>('/fd/first-mate/draft', json('POST', body))
export const getTemplates = () => call<{ templates: Template[] }>('/fd/first-mate/templates').then((r) => r.templates)
export const listFactories = (includeRetired = false) =>
  call<{ factories: FactoryCard[] }>(`/fd/first-mate/factories${includeRetired ? '?include_retired=true' : ''}`)
    .then((r) => r.factories)
export const createFactory = (body: { template?: string; title?: string; charter?: string }) =>
  call<Factory>('/fd/first-mate/factories', json('POST', body))
export const getFactory = (id: string) => call<FactoryDetail>(f(id))
export const patchFactory = (id: string, body: { title?: string; share?: number; priority?: number; goal_id?: string }) =>
  call<Factory>(f(id), json('PATCH', body))
export const deleteFactory = (id: string) => call<{ ok: boolean }>(f(id), { method: 'DELETE' })
export const newDraft = (id: string, body: DraftFields = {}) => call<Factory>(`${f(id)}/draft`, json('POST', body))
export const editDraft = (id: string, body: DraftFields) => call<Factory>(`${f(id)}/draft`, json('PATCH', body))
/** A live factory with cases is checked first: the answer then carries the check, and the draft waits. */
export const approve = (id: string, version: number, sha: string, skipCheck = false) =>
  call<Factory & { check?: Check }>(`${f(id)}/approve`, json('POST', skipCheck ? { version, sha, skip_check: true } : { version, sha }))
export const decline = (id: string, version: number, reason = '') =>
  call<Factory>(`${f(id)}/decline`, json('POST', { version, reason }))
export const setState = (id: string, state: OwnerState, reason = '') =>
  call<Factory>(`${f(id)}/state`, json('POST', { state, reason }))
export const setTrust = (id: string, level: Trust, pin?: boolean) =>
  call<Factory>(`${f(id)}/trust`, json('POST', pin === undefined ? { level } : { level, pin }))
export const runNow = (id: string, note = '') => call<Run>(`${f(id)}/run`, json('POST', { note }))
export const listRuns = (id: string) => call<{ runs: Run[] }>(`${f(id)}/runs`).then((r) => r.runs)
export const approveRun = (runId: string, ids?: number[]) =>
  call<Run>(`/fd/first-mate/runs/${encodeURIComponent(runId)}/approve`, json('POST', ids ? { ids } : {}))
export const rejectRun = (runId: string) =>
  call<Run>(`/fd/first-mate/runs/${encodeURIComponent(runId)}/reject`, { method: 'POST' })
export const rateRun = (runId: string, kind: 'up' | 'down' | '') =>
  call<{ thumbs: string; satisfaction: number | null }>(`/fd/first-mate/runs/${encodeURIComponent(runId)}/thumbs`, json('POST', { kind }))
export const stopRun = (runId: string) =>
  call<Run>(`/fd/first-mate/runs/${encodeURIComponent(runId)}/stop`, { method: 'POST' })
export const getReport = (week: 'last' | 'this') => call<WeekReport>(`/fd/first-mate/report?week=${week}`)
export const listCases = (id: string) => call<{ cases: Case[]; checks: Check[]; max: number }>(`${f(id)}/cases`)
export const addCase = (id: string, body: CaseFields & { from_run?: string }) =>
  call<Case>(`${f(id)}/cases`, json('POST', body))
export const updateCase = (id: string, caseId: string, body: Partial<CaseFields>) =>
  call<Case>(`${f(id)}/cases/${encodeURIComponent(caseId)}`, json('PATCH', body))
export const deleteCase = (id: string, caseId: string) =>
  call<{ ok: boolean }>(`${f(id)}/cases/${encodeURIComponent(caseId)}`, { method: 'DELETE' })
export const caseDraft = (runId: string) =>
  call<{ title: string; input: string; expect: string[]; from_run: string; has_input: boolean }>(
    `/fd/first-mate/runs/${encodeURIComponent(runId)}/case-draft`)
export const checkNow = (id: string) => call<Check>(`${f(id)}/check`, { method: 'POST' })
export const stopCheck = (id: string) => call<Check>(`${f(id)}/check`, { method: 'DELETE' })

// ── words ──────────────────────────────────────────────────────────────

const DAYS: Record<string, string> = {
  mon: 'Mon', tue: 'Tue', wed: 'Wed', thu: 'Thu', fri: 'Fri', sat: 'Sat', sun: 'Sun',
}

/** A schedule string as a person reads it: "weekly mon,thu 09:00" → "Mon, Thu at 09:00". */
export function scheduleText(schedule: string): string {
  const s = schedule.trim().toLowerCase()
  let m = s.match(/^daily\s+(\d{1,2}:\d{2})$/)
  if (m) return `Every day at ${m[1]}`
  m = s.match(/^weekdays\s+(\d{1,2}:\d{2})$/)
  if (m) return `Weekdays at ${m[1]}`
  m = s.match(/^weekly\s+([a-z,\s]+?)\s+(\d{1,2}:\d{2})$/)
  if (m) {
    const days = m[1].split(/[,\s]+/).filter(Boolean).map((d) => DAYS[d.slice(0, 3)] ?? d)
    return `${days.join(', ')} at ${m[2]}`
  }
  m = s.match(/^every\s+(\d+)\s*([mhd])$/)
  if (m) return `Every ${m[1]} ${{ m: 'min', h: 'h', d: 'days' }[m[2] as 'm' | 'h' | 'd']}`
  return schedule
}

/** How a factory starts, in one line. */
export function triggersText(triggers: Trigger[] | undefined): string {
  const parts = (triggers ?? []).map((t) => (t.kind === 'schedule' ? scheduleText(t.schedule)
    : `On new ${t.source}${t.event_type ? ` (${t.event_type.replace(/_/g, ' ')})` : ''}`))
  return parts.length ? parts.join(' · ') : 'Only when you press Run now'
}

/** "1.2M tokens", "34k tokens". */
export function tokensText(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1).replace(/\.0$/, '')}M tokens`
  if (n >= 1_000) return `${Math.round(n / 1_000)}k tokens`
  return `${n} tokens`
}

/** "Nov 21, 08:30", with the year only when it isn't this one. */
export function whenText(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const sameYear = d.getFullYear() === new Date().getFullYear()
  return d.toLocaleString(undefined, sameYear
    ? { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }
    : { day: 'numeric', month: 'short', year: 'numeric' })
}
