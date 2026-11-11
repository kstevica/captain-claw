/**
 * Scheduled tasks — the kiosk's face of Flight Deck's scheduler
 * (`captain_claw/flight_deck/fd_scheduler.py`). A task is a scheduler job with
 * `delivery_kind: 'agent'`: it runs a prompt on one of your agents, in its own
 * agent session, and delivers the result to the agent's chat, your email
 * and/or your Telegram. Every run is logged (searchable) under /scheduler/runs.
 */

import { useAuthStore } from '../stores/authStore'

export interface TaskDelivery {
  chat: boolean
  email: boolean
  telegram: boolean
  /** "" = your own address. */
  email_to: string
}

export interface ScheduledJob {
  id: string
  name: string
  schedule: string
  tz: string
  agent_slug: string
  prompt: string
  flow_id?: string
  delivery_kind: 'agent' | 'whatsapp' | 'channel' | 'telegram'
  delivery_target: string
  /** Set for `delivery_kind: 'agent'` (a task). */
  delivery: TaskDelivery | null
  agent_session_id: string
  enabled: number
  created_at: string
  updated_at: string
  next_run_at: number | null
  last_run_at: number | null
  last_status: string
  last_result: string
  run_count: number
  running: boolean
}

export interface RunDelivery {
  via: 'chat' | 'email' | 'telegram' | 'whatsapp' | 'channel' | string
  ok: boolean
  note: string
  to?: string
}

export interface TaskRun {
  id: number
  job_id: string
  job_name: string
  agent_slug: string
  trigger: 'scheduled' | 'manual' | string
  started_at: number
  finished_at: number | null
  status: string
  result: string
  deliveries: RunDelivery[]
  session_id: string
}

export interface DeliveryOptions {
  chat: { available: boolean }
  email: { available: boolean; provider: string; from: string; to: string; reason: string }
  telegram: { available: boolean; reason: string }
}

export interface SchedulePreview {
  ok: boolean
  next: number[]
  one_shot: boolean
  error: string
}

export interface TaskInput {
  name: string
  schedule: string
  tz: string
  agent_slug: string
  prompt: string
  delivery: TaskDelivery
  enabled?: boolean
}

/** A job an agent scheduled for itself with its `cron` tool (it lives in the
 *  agent's own file and runs in the chat it was set up from) — listed with
 *  the tasks by `captain_claw/flight_deck/agent_cron_routes.py`. Times are
 *  epoch seconds; the agent's schedule times are UTC. */
export interface AgentCronJob {
  id: string
  agent_slug: string
  /** The agent's name in Flight Deck's registry. */
  agent_name: string
  /** on: its process runs · asleep: woken before each run · off: its jobs wait until it's on. */
  agent_state: 'on' | 'asleep' | 'off'
  kind: string
  task: string
  /** "agent" when the agent wrote it; "" for one a person set up (/cron, the agent panel). */
  author: string
  /** The agent's own text: "daily 10:00", "every 3h", "weekly fri 17:00", "in 2 hours", "once at …". */
  schedule: string
  schedule_type: string
  session_id: string
  enabled: boolean
  created_at: number | null
  updated_at: number | null
  last_run_at: number | null
  next_run_at: number | null
  last_status: string
  last_error: string
  running: boolean
}

export interface AgentCronRun {
  /** 1 = its first run kept. */
  n: number
  started_at: number | null
  finished_at: number | null
  trigger: string
  status: 'ok' | 'queued' | 'error' | 'cancelled' | 'running' | 'interrupted' | string
  result: string
  error: string
}

/** One agent job's key in lists and views. */
export function agentJobKey(j: { agent_slug: string; id: string }): string {
  return `${j.agent_slug}/${j.id}`
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const jwt = useAuthStore.getState().token
  const r = await fetch(path, {
    ...init,
    cache: 'no-store',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(jwt ? { Authorization: `Bearer ${jwt}` } : {}),
      ...(init?.headers || {}),
    },
  })
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = typeof body?.detail === 'string' ? body.detail : ''
    } catch { /* not JSON */ }
    throw new Error(detail || `${r.status} ${r.statusText}`)
  }
  const text = await r.text()
  return (text ? JSON.parse(text) : {}) as T
}

/** The browser's IANA time zone — tasks' times are wall-clock times there. */
export function localTimeZone(): string {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || '' } catch { return '' }
}

export const schedulerApi = {
  listJobs: () => api<ScheduledJob[]>('/scheduler/jobs'),
  createTask: (t: TaskInput) =>
    api<ScheduledJob>('/scheduler/jobs', {
      method: 'POST',
      body: JSON.stringify({ ...t, delivery_kind: 'agent', enabled: t.enabled ?? true }),
    }),
  updateJob: (id: string, patch: Partial<TaskInput> & { enabled?: boolean }) =>
    api<ScheduledJob>(`/scheduler/jobs/${encodeURIComponent(id)}`, {
      method: 'PATCH', body: JSON.stringify(patch),
    }),
  deleteJob: (id: string) =>
    api<{ ok: boolean }>(`/scheduler/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  runNow: (id: string) =>
    api<{ ok: boolean; status: string; run_id: number }>(
      `/scheduler/jobs/${encodeURIComponent(id)}/run?background=1`, { method: 'POST' }),
  runs: (opts: { jobId?: string; q?: string; limit?: number; before?: number } = {}) => {
    const p = new URLSearchParams()
    if (opts.jobId) p.set('job_id', opts.jobId)
    if (opts.q) p.set('q', opts.q)
    if (opts.limit) p.set('limit', String(opts.limit))
    if (opts.before) p.set('before', String(opts.before))
    return api<TaskRun[]>(`/scheduler/runs?${p}`)
  },
  preview: (schedule: string, tz: string) =>
    api<SchedulePreview>('/scheduler/preview', { method: 'POST', body: JSON.stringify({ schedule, tz }) }),
  deliveryOptions: () => api<DeliveryOptions>('/scheduler/delivery-options'),
  // The agents' own schedules (their cron).
  listAgentJobs: () => api<AgentCronJob[]>('/scheduler/agent-jobs'),
  agentJobRuns: (slug: string, id: string) =>
    api<AgentCronRun[]>(`/scheduler/agent-jobs/${encodeURIComponent(slug)}/${encodeURIComponent(id)}/runs`),
  agentJobAction: (slug: string, id: string, action: 'pause' | 'resume' | 'run') =>
    api<{ ok: boolean }>(
      `/scheduler/agent-jobs/${encodeURIComponent(slug)}/${encodeURIComponent(id)}/${action}`, { method: 'POST' }),
  deleteAgentJob: (slug: string, id: string) =>
    api<{ ok: boolean }>(`/scheduler/agent-jobs/${encodeURIComponent(slug)}/${encodeURIComponent(id)}`, { method: 'DELETE' }),
}

// ── Schedule strings ↔ a form a person fills in ──

export type Frequency = 'daily' | 'weekdays' | 'weekly' | 'every' | 'once'
export type EveryUnit = 'm' | 'h' | 'd'

export interface ScheduleForm {
  freq: Frequency
  time: string        // HH:MM
  days: number[]      // Mon=0 … Sun=6 (weekly)
  every: number
  unit: EveryUnit
  date: string        // YYYY-MM-DD (once)
}

const DAY_KEYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
export const DAY_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const DAY_LONG = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

function pad(n: number): string { return String(n).padStart(2, '0') }

export function isoDate(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

export function defaultScheduleForm(): ScheduleForm {
  const tomorrow = new Date(Date.now() + 86400_000)
  return { freq: 'weekdays', time: '08:00', days: [0], every: 2, unit: 'h', date: isoDate(tomorrow) }
}

export function buildSchedule(f: ScheduleForm): string {
  const time = /^\d{1,2}:\d{2}$/.test(f.time) ? f.time : '08:00'
  switch (f.freq) {
    case 'daily': return `daily ${time}`
    case 'weekdays': return `weekdays ${time}`
    case 'weekly': {
      const days = [...new Set(f.days)].sort((a, b) => a - b)
      return `weekly ${(days.length ? days : [0]).map((d) => DAY_KEYS[d]).join(',')} ${time}`
    }
    case 'every': return `every ${Math.max(1, Math.round(f.every || 1))}${f.unit}`
    case 'once': return `once ${f.date}T${time}:00`
  }
}

/** Read a stored schedule back into the form ("in 10m" and other one-offs
 *  become "once" at their next run). */
export function parseSchedule(s: string, nextRunAt: number | null): ScheduleForm {
  const f = defaultScheduleForm()
  const low = (s || '').trim().toLowerCase()
  let m: RegExpMatchArray | null
  if ((m = low.match(/^daily\s+(\d{1,2}:\d{2})$/))) return { ...f, freq: 'daily', time: padTime(m[1]) }
  if ((m = low.match(/^weekdays\s+(\d{1,2}:\d{2})$/))) return { ...f, freq: 'weekdays', time: padTime(m[1]) }
  if ((m = low.match(/^weekly\s+([a-z,\s]+?)\s+(\d{1,2}:\d{2})$/))) {
    const days = m[1].split(/[,\s]+/).map((d) => DAY_KEYS.findIndex((k) => d.startsWith(k))).filter((d) => d >= 0)
    return { ...f, freq: 'weekly', days: days.length ? days : [0], time: padTime(m[2]) }
  }
  if ((m = low.match(/^every\s+(\d+)\s*([mhd])$/))) return { ...f, freq: 'every', every: Number(m[1]), unit: m[2] as EveryUnit }
  if ((m = low.match(/^once\s+(\d{4}-\d{2}-\d{2})t(\d{2}:\d{2})/))) return { ...f, freq: 'once', date: m[1], time: m[2] }
  if (nextRunAt) {
    const d = new Date(nextRunAt * 1000)
    return { ...f, freq: 'once', date: isoDate(d), time: `${pad(d.getHours())}:${pad(d.getMinutes())}` }
  }
  return f
}

function padTime(t: string): string {
  const [h, m] = t.split(':')
  return `${pad(Number(h))}:${m}`
}

/** "Weekdays at 08:00", "Every Monday at 09:00", "Every 2 hours". */
export function describeSchedule(s: string, nextRunAt?: number | null): string {
  const low = (s || '').trim().toLowerCase()
  let m: RegExpMatchArray | null
  if ((m = low.match(/^daily\s+(\d{1,2}:\d{2})$/))) return `Every day at ${padTime(m[1])}`
  if ((m = low.match(/^weekdays\s+(\d{1,2}:\d{2})$/))) return `Weekdays at ${padTime(m[1])}`
  if ((m = low.match(/^weekly\s+([a-z,\s]+?)\s+(\d{1,2}:\d{2})$/))) {
    const days = m[1].split(/[,\s]+/).map((d) => DAY_KEYS.findIndex((k) => d.startsWith(k))).filter((d) => d >= 0)
    if (days.length === 1) return `Every ${DAY_LONG[days[0]]} at ${padTime(m[2])}`
    return `${days.sort((a, b) => a - b).map((d) => DAY_SHORT[d]).join(', ')} at ${padTime(m[2])}`
  }
  if ((m = low.match(/^every\s+(\d+)\s*([mhd])$/))) {
    const n = Number(m[1])
    const unit = m[2] === 'm' ? 'minute' : m[2] === 'h' ? 'hour' : 'day'
    return n === 1 ? `Every ${unit}` : `Every ${n} ${unit}s`
  }
  if (low.startsWith('once') || low.startsWith('in ')) {
    if (nextRunAt) return `Once · ${fmtWhen(nextRunAt)}`
    const iso = low.match(/^once\s+(\d{4}-\d{2}-\d{2}t\d{2}:\d{2})/)
    const at = iso ? Date.parse(iso[1].toUpperCase()) : NaN
    return Number.isNaN(at) ? 'Once' : `Once · ${fmtWhen(at / 1000)}`
  }
  return s
}

/** An agent's own schedule as a person reads it. Its clock times are UTC
 *  (the agent's cron runs on UTC), so they say so. */
export function describeAgentSchedule(j: Pick<AgentCronJob, 'schedule' | 'next_run_at'>): string {
  const low = (j.schedule || '').trim().toLowerCase()
  let m: RegExpMatchArray | null
  if ((m = low.match(/^daily\s+(\d{1,2}:\d{2})$/))) return `Every day at ${padTime(m[1])} UTC`
  if ((m = low.match(/^weekly\s+([a-z]+)\s+(\d{1,2}:\d{2})$/))) {
    const d = DAY_KEYS.findIndex((k) => m![1].startsWith(k))
    return `Every ${d >= 0 ? DAY_LONG[d] : m[1]} at ${padTime(m[2])} UTC`
  }
  if ((m = low.match(/^every\s+(\d+)\s*([mh])$/))) return describeSchedule(`every ${m[1]}${m[2]}`)
  if (low.startsWith('once') || low.startsWith('in ')) return j.next_run_at ? `Once · ${fmtWhen(j.next_run_at)}` : 'Once'
  return j.schedule || 'Unknown schedule'
}

/** An agent's one-off job that has run: the agent turned it off after its run. */
export function isFinishedAgentOneShot(j: Pick<AgentCronJob, 'schedule_type' | 'enabled' | 'last_run_at'>): boolean {
  return !j.enabled && j.schedule_type === 'once' && !!j.last_run_at
}

/** A one-shot that has run: finished, not paused. */
export function isFinishedOneShot(j: { schedule: string; enabled: number; last_run_at: number | null }): boolean {
  return j.enabled !== 1 && !!j.last_run_at && /^(once|in)\b/i.test(j.schedule.trim())
}

// ── Time ──

export function fmtWhen(epoch: number): string {
  const d = new Date(epoch * 1000)
  const now = new Date()
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  const sameDay = (a: Date, b: Date) => a.toDateString() === b.toDateString()
  if (sameDay(d, now)) return `Today, ${time}`
  if (sameDay(d, new Date(now.getTime() + 86400_000))) return `Tomorrow, ${time}`
  if (sameDay(d, new Date(now.getTime() - 86400_000))) return `Yesterday, ${time}`
  const date = d.toLocaleDateString([], {
    weekday: 'short', day: 'numeric', month: 'short',
    ...(d.getFullYear() !== now.getFullYear() ? { year: 'numeric' } : {}),
  })
  return `${date}, ${time}`
}

export function fmtRelative(epoch: number): string {
  const secs = Math.round(epoch - Date.now() / 1000)
  const abs = Math.abs(secs)
  const span = abs < 60 ? `${abs}s` : abs < 3600 ? `${Math.round(abs / 60)} min`
    : abs < 86400 ? `${Math.round(abs / 3600)} h` : `${Math.round(abs / 86400)} d`
  if (abs < 45) return secs >= 0 ? 'in a moment' : 'just now'
  return secs >= 0 ? `in ${span}` : `${span} ago`
}

export function fmtDuration(run: TaskRun): string {
  if (!run.finished_at) return ''
  const s = Math.max(0, Math.round(run.finished_at - run.started_at))
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${pad(s % 60)}s`
}

// ── Status ──

export type RunTone = 'ok' | 'partial' | 'error' | 'running' | 'skipped'

export function runTone(status: string): RunTone {
  if (status === 'running') return 'running'
  if (status.startsWith('ok')) return 'ok'
  if (status.startsWith('partial')) return 'partial'
  if (status.startsWith('skipped')) return 'skipped'
  return 'error'
}

/** A failed run's status as the person would want to read it. */
export function explainStatus(status: string): string {
  if (status.startsWith('error:agent-not-running')) return 'The agent could not be started for this run. Check it in the agent list, then use Run now.'
  if (status.startsWith('error:agent-not-yours')) return 'This agent belongs to someone else.'
  if (status.startsWith('error:no-reply')) return 'The agent did not answer in time.'
  if (status.startsWith('error:interrupted')) return 'Flight Deck restarted while this ran.'
  if (status.startsWith('error:delivery')) return 'The agent answered, but it could not be delivered.'
  if (status.startsWith('skipped:quiet')) return 'Skipped during quiet hours — it runs when they end.'
  if (status.startsWith('skipped:limit')) return "Skipped: today's AI token allowance on your plan is used up. Tasks run again after it resets at midnight UTC."
  if (status.startsWith('partial')) return 'Delivered, but not everywhere.'
  return status.replace(/^error:/, '')
}
