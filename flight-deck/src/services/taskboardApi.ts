/**
 * Taskboard — every task an agent received or sent on any channel, whether
 * it was finished, by whom and for whom (`captain_claw/flight_deck/taskboard.py`).
 * Flight Deck builds it from the records it already keeps (scheduled runs,
 * proposals, follow-ups, held auto-replies, sent mail, agent links) and from
 * the turns agents report on every channel, which Jev then sorts; every read
 * is the caller's own board.
 */

import { useAuthStore } from '../stores/authStore'
import { api, type Facet } from './fileLibraryApi'

export type TaskView = 'needs_me' | 'waiting' | 'working' | 'done' | 'all'
export type TaskStatus = 'open' | 'done' | 'failed' | 'dismissed' | 'not_task' | 'maybe'
export type TaskOwesNext = 'me' | 'agent' | 'them' | 'nobody' | 'unclear'
export type TaskEvidence = 'verified' | 'claimed' | 'none'
export type TaskResult = 'success' | 'partial' | 'failed' | 'unknown' | ''
export type TaskSort = 'recent' | 'oldest' | 'due' | 'urgency'

/** Someone a task is from, for, or done by — as the source named them. */
export interface TaskParty {
  role: 'requester' | 'doer' | 'recipient' | 'cc' | 'mentioned' | string
  kind: string
  address: string
  display: string
  /** The person this handle belongs to (v2: parties are people), and their
   *  name — the same person over email, WhatsApp and Telegram is one. */
  party_id?: string | null
  name?: string
  /** Taken from a header, the register or the speaker, not from text. */
  verified: number | boolean
  /** The board user (kind fd_user, their own id): said as "You". A deck
   *  user's id is never shown. */
  me?: boolean
}

/** Someone a next step may go to: only these (the task's recorded parties). */
export interface ProposalRecipient {
  address: string
  display: string
  kind: string
}

/** A linked Autonomous Work action (`status` 'awaiting_approval' waits for you).
 *  One the board proposed (`source` 'taskboard') carries what it would do, which
 *  you may edit before you approve it. */
export interface TaskProposal {
  id: string
  status: string
  title: string
  /** The catalog kind: mail.draft (an email draft, never sent), reply.on_origin
   *  (a reply where the task came in), nudge (a reminder). */
  kind?: 'mail.draft' | 'reply.on_origin' | 'nudge' | string
  source?: 'taskboard' | string
  /** Whether wording and recipient can still change (yours, waiting, the board's). */
  editable?: boolean
  payload?: {
    text?: string
    subject?: string
    to?: string
    to_options?: ProposalRecipient[]
    channel?: string
  }
}

/** What an edit before Approve may change; `to` is one of `to_options`. */
export interface ProposalEdit {
  text?: string
  subject?: string
  to?: string
  channel?: string
}

/** What "Suggest next step" may ask for (auto: whatever fits the task best). */
export type NextStepKind = 'auto' | 'remind' | 'ask_clarifying' | 'reply_with_result' | 'draft_email'

export interface TaskFlags {
  /** Third-party text that asks the agent to act: no next step is ever suggested for it. */
  contains_instructions?: boolean
  /** Jev thinks it is finished, not sure enough to close it: Needs me asks. */
  probably_done?: boolean
  stale?: boolean
  overdue?: boolean
  /** Filled in from the last 30 days of chats: it never gets a next step. */
  historical?: boolean
  /** Jev thinks it is the same task as another one (only shown: you merge). */
  same_as?: { task_id: string; title: string; p: number }
  /** Answered in the same turn on your own chat. */
  answered_in_chat?: boolean
  /** Your own conversation with the agent (your chat, a shared chat, your
   *  own WhatsApp / Telegram / Messenger, your devices and glasses). */
  own_chat?: boolean
  /** The summary's reading of the agent's reply: did it do what was asked? */
  answered?: 'yes' | 'partly' | 'no'
  /** Why not, in a few words ('' when it did). */
  blocker?: string
  /** The agent said it couldn't do it (nothing failed): your move. */
  agent_declined?: boolean
  /** Only part of it was done: your move. */
  partly_done?: boolean
  /** Someone wrote again after it was done (it stays done). */
  new_since_done?: boolean
  /** Taken in as a task with less certainty: not counted in the badge. */
  low_confidence?: boolean
  /** Why it is here, in a few words. */
  why?: string
}

export interface TaskItem {
  id: string
  agent_slug: string
  agent_ref: string
  agent_name: string
  title: string
  ask: string
  result_text: string
  /** request = something asked of an agent in a chat, on any channel. */
  kind: 'request' | 'sent' | 'scheduled' | 'proposal' | 'follow_up' | 'held_reply' | 'link' | 'todo' | 'message' | string
  direction: 'received' | 'sent' | 'internal' | string
  /** The first channel it came in on; `channels` every one it touched. */
  channel: string
  channels: string[]
  status: TaskStatus
  owes_next: TaskOwesNext
  evidence: TaskEvidence
  result: TaskResult
  urgency: number
  /** Epoch seconds, like every time here. */
  due_at: number | null
  snooze_until: number | null
  created_at: number
  updated_at: number
  last_event_at: number | null
  closed_at: number | null
  closed_by: 'source' | 'evidence' | 'jev' | 'rule' | 'user' | ''
  /** Who finished it (an agent, you, a party). */
  closed_by_party: string
  /** Fields you changed by hand, and when: no machine pass touches them again. */
  user_locks: Record<string, number>
  flags: TaskFlags
  private: number | boolean
  source: string
  source_ref: string
  /** Merged into this task (its events, links and people are there now). */
  merged_into?: string
  /** A task from one of your own inboxes is deleted after this (epoch seconds). */
  retention_until?: number | null
  /** Computed: needs_me, waiting, working, done, snoozed — or other (failed
   *  and seen, dismissed, maybe, open with no clear next move). */
  lane: string
  /** The first few. */
  parties: TaskParty[]
  proposal: TaskProposal | null
  /** The ids to link back with, from the task's keys ('' when it has none). */
  links: {
    scheduled_job_id?: string
    /** The agent session it ran in, and that session's agent (a slug). */
    session_id?: string
    session_agent?: string
    autonomy_id?: string
    followup_id?: string
    gmail_thread_id?: string
    /** The gmail_autoreplies row of a held reply. */
    autoreply_id?: string
    link_id?: string
  }
}

/** One thing that happened to a task, on any channel. */
export interface TaskEvent {
  id: number
  task_id: string
  at: number
  kind: string
  channel: string
  actor_kind: 'me' | 'agent' | 'party' | 'system' | 'jev' | string
  actor: string
  summary: string
  source: string
  source_ref: string
  outcome: 'ok' | 'error' | 'blocked' | ''
  refs: Record<string, unknown>
  /** recorded = a send or delivery was logged; said = only the agent's word. */
  evidence: 'recorded' | 'said' | 'user' | ''
  private: number | boolean
}

export interface TaskLinkKey {
  key: string
  strength: 'strong' | 'weak'
  created_at: number
}

/** One answer of a decision: a yes/no's probability, a choice and how sure,
 *  or a score. */
export type TaskAnswer = { p: number } | { choice: string; p: number } | { score: number }

/** A decision about a task: who made it (Jev, a rule, the model, you), what
 *  it answered, what it changed, and why in plain words. */
export interface TaskDecision {
  id?: number | string
  at: number
  judge: 'jev' | 'rule' | 'model' | 'user' | string
  /** The question set it answered (e.g. taskboard.v1). */
  qset?: string
  answers: Record<string, TaskAnswer>
  /** The move it made ('' when it left the task as it was). */
  applied: string
  why: string
}

/** "Why is this here?" — the latest decisions first (at most 20). */
export interface TaskWhy {
  items: TaskDecision[]
  summary: string
}

export interface TaskDetail {
  task: TaskItem
  /** The latest 500, oldest first. */
  events: TaskEvent[]
  /** How many events the task has in all. */
  events_total: number
  links: TaskLinkKey[]
  parties: TaskParty[]
  decisions: TaskDecision[]
  proposals: (TaskProposal & Record<string, unknown>)[]
  /** The tasks merged into this one (each can be taken out again). */
  merged?: { id: string; title: string }[]
}

export interface TaskCounts {
  needs_me: number
  waiting: number
  working: number
  done: number
  all: number
}

export type TaskFacetKey = 'status' | 'channel' | 'agent' | 'direction' | 'kind' | 'evidence' | 'result' | 'party'

/** A filter value; a People one says whether it is the board user. */
export interface TaskFacet extends Facet {
  me?: boolean
}

export interface TaskPage {
  items: TaskItem[]
  total: number
  /** Each counted under the other filters. */
  facets: Record<TaskFacetKey, TaskFacet[]>
  counts: TaskCounts
}

/** What the board reads from, and whether it does yet. */
export interface TaskSource {
  key: string
  label: string
  tracked: boolean
  on: boolean
  note: string
}

/** Filling in the last 30 days of your agents' chats, once. */
export interface TaskboardBackfill {
  state: 'idle' | 'running' | 'done' | 'error'
  done_at: number | null
  progress: { agents_done: number; agents_total: number; turns: number; tasks: number }
  message: string
}

export interface TaskboardStatus {
  sources: TaskSource[]
  counts: TaskCounts
  sweep: { running: boolean; last_at: number | null; message: string }
  backfill?: TaskboardBackfill
}

/** Your own inboxes, each read only once you switch it on. */
export type PassiveKey = 'telegram_inbox' | 'whatsapp_web' | 'gmail'
export const PASSIVE_KEYS: PassiveKey[] = ['telegram_inbox', 'whatsapp_web', 'gmail']

export interface PassiveSource {
  on: boolean
  /** When you agreed (epoch seconds). */
  consented_at: number | null
  /** Whether it can be switched on here at all; `note` says why not. */
  available: boolean
  note: string
}

/** The bounds the server holds the daily limits to (400 outside them). */
export const JEV_PER_DAY_MAX = 2000
export const EXTRACT_PER_DAY_MAX = 1000

export interface TaskboardSettings {
  /** Jev may close a task by itself when it is sure (with Undo). */
  auto_close: boolean
  /** Daily limits on Jev decisions and on summaries (0 = none). */
  jev_per_day: number
  extract_per_day: number
  /** What today used of them (the day in UTC). */
  counters?: { day: string; jev: number; extract: number }
  /** Whether Jev can be asked, and with whose key. */
  jev?: { connected: boolean; source: 'own' | 'team' | '' }
  passive?: Partial<Record<PassiveKey, PassiveSource>>
  backfill_done_at?: number | null
  updated_at?: number | null
}

/** Only what changed. */
export type TaskboardSettingsPatch = Partial<Pick<TaskboardSettings, 'auto_close' | 'jev_per_day' | 'extract_per_day'>>

/** confirm_task — a maybe task is one (the agent's move); not_done — not
 *  finished after all (it stays open). */
export type TaskAction =
  | 'done' | 'reopen' | 'dismiss' | 'not_task' | 'snooze' | 'unsnooze'
  | 'rename' | 'owes_next' | 'ack_failed' | 'undo' | 'confirm_task' | 'not_done'

/** Multi-value filters as lists (sent comma-separated); `from`/`to` epoch seconds. */
export interface TaskQuery {
  view?: TaskView
  q?: string
  status?: string[]
  channel?: string[]
  agent?: string[]
  direction?: string[]
  kind?: string[]
  party?: string[]
  evidence?: string[]
  result?: string[]
  from?: number
  to?: number
  sort?: TaskSort
  limit?: number
  offset?: number
}

const BASE = '/fd/taskboard'
const taskPath = (id: string, rest = '') => `${BASE}/tasks/${encodeURIComponent(id)}${rest}`

/** A board task's own id (`tb_` and hex); anything with a colon is a key. */
export const TASK_ID = /^tb_[0-9a-f]{8,64}$/i

/** The query as the server reads it: lists comma-separated, blanks left out. */
function queryString(query: TaskQuery): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === '' || v === null) continue
    if (Array.isArray(v)) { if (v.length) p.set(k, v.join(',')) } else p.set(k, String(v))
  }
  return p.toString()
}

export interface TaskboardSync {
  ok: boolean
  new_events: number
  counts: TaskCounts
  sweep: TaskboardStatus['sweep']
}

export type ExportFormat = 'csv' | 'jsonl'

/** A file from a Flight Deck route, saved under the name the server gives
 *  it (the token rides in the header, never in the address). */
async function download(path: string, fallback: string): Promise<void> {
  const jwt = useAuthStore.getState().token
  const r = await fetch(path, {
    cache: 'no-store',
    credentials: 'include',
    headers: jwt ? { Authorization: `Bearer ${jwt}` } : {},
  })
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = typeof body?.detail === 'string' ? body.detail : ''
    } catch { /* not JSON */ }
    throw new Error(detail || `${r.status} ${r.statusText}`)
  }
  const named = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(r.headers.get('Content-Disposition') || '')?.[1]
  let name = fallback
  try { if (named) name = decodeURIComponent(named) } catch { /* the name as sent */ }
  const url = URL.createObjectURL(await r.blob())
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 30000)
}

export const taskboardApi = {
  /** What the board reads, each view's count, the sweep and (v2) the history fill. */
  status: () => api<TaskboardStatus>(`${BASE}/status`),
  /** Read every source again now, for this user. */
  sync: () => api<TaskboardSync>(`${BASE}/sync`, { method: 'POST' }),
  summary: () => api<{ needs_me: number }>(`${BASE}/summary`),
  tasks: (query: TaskQuery = {}) => api<TaskPage>(`${BASE}/tasks?${queryString(query)}`),
  task: (id: string) => api<TaskDetail>(taskPath(id)),
  /** A correction: applies at once and locks the field it touched. */
  action: (id: string, action: TaskAction, value?: string | number) =>
    api<{ task: TaskItem }>(taskPath(id, '/action'), {
      method: 'POST',
      body: JSON.stringify(value === undefined ? { action } : { action, value }),
    }),
  /** The decisions that put a task where it is, in plain words. */
  why: (id: string) => api<TaskWhy>(taskPath(id, '/why')),
  settings: () => api<TaskboardSettings>(`${BASE}/settings`),
  saveSettings: (patch: TaskboardSettingsPatch) =>
    api<TaskboardSettings>(`${BASE}/settings`, { method: 'PUT', body: JSON.stringify(patch) }),

  // ── v2 ──

  /** Ask for a next step now: an action waiting for your Approve, drafted on
   *  the agent's own model — or none, and why (`reason`). */
  propose: (id: string, kind: NextStepKind = 'auto') =>
    api<{ proposal: TaskProposal | null; reason: string }>(taskPath(id, '/propose'), {
      method: 'POST', body: JSON.stringify({ kind }),
    }),
  /** Change a board proposal's wording or recipient before you approve it
   *  (409 once it is answered). Approve / Reject stay the bell's routes. */
  editProposal: (actionId: string, edit: ProposalEdit) =>
    api<{ action: Record<string, unknown> }>(`/fd/autonomy/actions/${encodeURIComponent(actionId)}`, {
      method: 'PATCH', body: JSON.stringify(edit),
    }),
  /** One thread: this task's events, links and people move to `into`
   *  (undone by unmerge). */
  merge: (id: string, into: string) =>
    api<{ task: TaskItem }>(taskPath(id, '/merge'), { method: 'POST', body: JSON.stringify({ into }) }),
  unmerge: (id: string) => api<{ task: TaskItem }>(taskPath(id, '/unmerge'), { method: 'POST' }),
  /** Jev's "Same as …?" was wrong: the suggestion goes and none comes back
   *  (Undo brings it back). */
  notSame: (id: string) => api<{ task: TaskItem }>(taskPath(id, '/not_same'), { method: 'POST' }),
  /** These events make a task of their own (`new_task`). */
  split: (id: string, eventIds: number[]) =>
    api<{ task: TaskItem; new_task: TaskItem }>(taskPath(id, '/split'), {
      method: 'POST', body: JSON.stringify({ event_ids: eventIds }),
    }),
  /** Switch one of your own inboxes on (you agreed to what it reads) or off —
   *  off deletes every task that came only from it (`deleted`: how many). */
  setPassive: async (key: PassiveKey, on: boolean): Promise<{ settings: TaskboardSettings; deleted: number }> => {
    const res = await api<TaskboardSettings | { settings: TaskboardSettings; deleted?: number }>(`${BASE}/settings`, {
      method: 'PUT',
      body: JSON.stringify({ passive: { [key]: on ? { on: true, consent: true } : { on: false } } }),
    })
    return 'settings' in res && res.settings
      ? { settings: res.settings, deleted: Math.max(0, Number(res.deleted) || 0) }
      : { settings: res as TaskboardSettings, deleted: 0 }
  },
  /** Fill in the last 30 days now (once; again only after it failed). */
  backfill: () => api<Record<string, unknown>>(`${BASE}/backfill`, { method: 'POST' }),
  /** Your tasks as a file, under the board's view, search and filters. */
  exportTasks: (format: ExportFormat, query: TaskQuery) => {
    const rest: TaskQuery = { ...query }
    delete rest.limit
    delete rest.offset
    const qs = queryString(rest)
    return download(`${BASE}/export?format=${format}${qs ? `&${qs}` : ''}`, `taskboard.${format}`)
  },
  /** A task by what it is known by: its own id, or a key it links back
   *  with — `sched:job:<id>` (a scheduled task, every run of it) or
   *  `autonomy:<id>` (a proposal). Read from the board's own listing (the
   *  ids each task links back with); null when it isn't on the board. */
  findTask: async (ref: string): Promise<TaskItem | null> => {
    if (!ref.includes(':')) return (await taskboardApi.task(ref)).task
    const look = async (query: TaskQuery, hit: (t: TaskItem) => boolean) =>
      (await taskboardApi.tasks({ ...query, limit: 200 })).items.find(hit) ?? null
    if (ref.startsWith('sched:job:')) {
      const job = ref.slice('sched:job:'.length)
      return look({ view: 'all', kind: ['scheduled'] }, (t) => t.links?.scheduled_job_id === job)
    }
    if (ref.startsWith('autonomy:')) {
      const action = ref.slice('autonomy:'.length)
      const hit = (t: TaskItem) => t.proposal?.id === action || t.links?.autonomy_id === action
      // One that waits for you is in Needs me (unless snoozed).
      return (await look({ view: 'needs_me' }, hit)) ?? look({ view: 'all' }, hit)
    }
    return null
  },
  /** The proposals waiting for you, each with the task it is on: a board
   *  proposal's own task (its id), else the proposal's key (the task the
   *  board keeps for it, found when it opens). For the bell. */
  waitingProposals: async (): Promise<Map<string, string>> => {
    type Row = { id?: string; source?: string; ref_id?: string; payload?: Record<string, unknown> }
    const res = await api<{ actions?: Row[] }>('/fd/autonomy/actions?status_filter=awaiting_approval&limit=200')
    const out = new Map<string, string>()
    for (const a of res.actions ?? []) {
      const id = String(a.id || '')
      if (!id) continue
      const named = a.source === 'taskboard' ? a.ref_id : a.payload?.task_id
      out.set(id, typeof named === 'string' && TASK_ID.test(named) ? named : `autonomy:${id}`)
    }
    return out
  },
}
