/**
 * Taskboard — every task an agent received or sent on any channel, whether
 * it was finished, by whom and for whom (`captain_claw/flight_deck/taskboard.py`).
 * Flight Deck builds it from the records it already keeps (scheduled runs,
 * proposals, follow-ups, held auto-replies, sent mail, agent links) and from
 * the turns agents report on every channel, which Jev then sorts; every read
 * is the caller's own board.
 */

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
  party_id?: string | null
  /** Taken from a header, the register or the speaker, not from text. */
  verified: number | boolean
  /** The board user (kind fd_user, their own id): said as "You". A deck
   *  user's id is never shown. */
  me?: boolean
}

/** A linked Autonomous Work action (`status` 'awaiting_approval' waits for you). */
export interface TaskProposal {
  id: string
  status: string
  title: string
}

export interface TaskFlags {
  /** Third-party text that asks the agent to act: no next step is ever suggested for it. */
  contains_instructions?: boolean
  /** Jev thinks it is finished, not sure enough to close it: Needs me asks. */
  probably_done?: boolean
  stale?: boolean
  overdue?: boolean
  historical?: boolean
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

export interface TaskboardStatus {
  sources: TaskSource[]
  counts: TaskCounts
  sweep: { running: boolean; last_at: number | null; message: string }
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
  passive?: Record<string, { on: boolean; consented_at: number | null }>
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

export interface TaskboardSync {
  ok: boolean
  new_events: number
  counts: TaskCounts
  sweep: TaskboardStatus['sweep']
}

export const taskboardApi = {
  status: () => api<TaskboardStatus>(`${BASE}/status`),
  /** Read every source again now, for this user. */
  sync: () => api<TaskboardSync>(`${BASE}/sync`, { method: 'POST' }),
  summary: () => api<{ needs_me: number }>(`${BASE}/summary`),
  tasks: (query: TaskQuery = {}) => {
    const p = new URLSearchParams()
    for (const [k, v] of Object.entries(query)) {
      if (v === undefined || v === '' || v === null) continue
      if (Array.isArray(v)) { if (v.length) p.set(k, v.join(',')) } else p.set(k, String(v))
    }
    return api<TaskPage>(`${BASE}/tasks?${p}`)
  },
  task: (id: string) => api<TaskDetail>(`${BASE}/tasks/${encodeURIComponent(id)}`),
  /** A correction: applies at once and locks the field it touched. */
  action: (id: string, action: TaskAction, value?: string | number) =>
    api<{ task: TaskItem }>(`${BASE}/tasks/${encodeURIComponent(id)}/action`, {
      method: 'POST',
      body: JSON.stringify(value === undefined ? { action } : { action, value }),
    }),
  /** The decisions that put a task where it is, in plain words. */
  why: (id: string) => api<TaskWhy>(`${BASE}/tasks/${encodeURIComponent(id)}/why`),
  settings: () => api<TaskboardSettings>(`${BASE}/settings`),
  saveSettings: (patch: TaskboardSettingsPatch) =>
    api<TaskboardSettings>(`${BASE}/settings`, { method: 'PUT', body: JSON.stringify(patch) }),
}
