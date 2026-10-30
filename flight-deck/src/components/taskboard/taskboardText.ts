import {
  AlarmClock, Bot, Code2, Glasses, Inbox, Link2, Mail, MessageCircle, MessagesSquare, Plug, Send, Smartphone, Sparkles,
  Users, Watch,
} from 'lucide-react'
import type { TaskAnswer, TaskFlags, TaskParty, TaskView } from '../../services/taskboardApi'
import { fmtRelative } from '../../services/schedulerApi'

// The Taskboard's words and plain helpers (no components, so its parts file
// keeps fast refresh): search folding, the names of channels, statuses and
// the rest, and the date filter's range.

// ── Folding ──
//
// outbound_guard._fold: casefolded, diacritics dropped ("Đurić" → "duric").
// Letters NFKD leaves whole are mapped by hand, as there.
const FOLD_EXTRA: Record<string, string> = {
  'đ': 'd', 'Đ': 'd', 'ł': 'l', 'Ł': 'l', 'ø': 'o', 'Ø': 'o', 'ħ': 'h', 'Ħ': 'h', 'ı': 'i',
  'æ': 'ae', 'Æ': 'ae', 'œ': 'oe', 'Œ': 'oe', 'ß': 'ss', 'ẞ': 'ss',
}

export function foldChar(ch: string): string {
  return FOLD_EXTRA[ch] ?? ch.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase()
}

export function foldText(text: string): string {
  let out = ''
  for (const ch of text) out += foldChar(ch)
  return out
}

// ── Labels ──

// Every turn_source channel key, and the board's own (scheduled, autonomy…).
const CHANNELS: Record<string, { label: string; Icon: typeof Mail; tone: string }> = {
  web: { label: 'Chat', Icon: MessagesSquare, tone: 'text-violet-600 dark:text-violet-400' },
  app: { label: 'Captain app', Icon: Smartphone, tone: 'text-violet-600 dark:text-violet-400' },
  whatsapp: { label: 'WhatsApp', Icon: MessageCircle, tone: 'text-emerald-600 dark:text-emerald-400' },
  waweb: { label: 'WhatsApp Web', Icon: MessageCircle, tone: 'text-emerald-600 dark:text-emerald-400' },
  telegram: { label: 'Telegram', Icon: Send, tone: 'text-sky-600 dark:text-sky-400' },
  tg_inbox: { label: 'Telegram inbox', Icon: Inbox, tone: 'text-sky-600 dark:text-sky-400' },
  gmail: { label: 'Gmail', Icon: Mail, tone: 'text-rose-600 dark:text-rose-400' },
  email: { label: 'Email', Icon: Mail, tone: 'text-rose-600 dark:text-rose-400' },
  device: { label: 'Device', Icon: Watch, tone: 'text-orange-600 dark:text-orange-400' },
  link: { label: 'Linked system', Icon: Link2, tone: 'text-cyan-600 dark:text-cyan-400' },
  scheduled: { label: 'Scheduled', Icon: AlarmClock, tone: 'text-amber-600 dark:text-amber-400' },
  autonomy: { label: 'Autonomous Work', Icon: Sparkles, tone: 'text-fuchsia-600 dark:text-fuchsia-400' },
  member: { label: 'Shared chat', Icon: Users, tone: 'text-indigo-600 dark:text-indigo-400' },
  api: { label: 'API', Icon: Code2, tone: 'text-teal-600 dark:text-teal-400' },
  mcp: { label: 'MCP', Icon: Plug, tone: 'text-teal-600 dark:text-teal-400' },
  glasses: { label: 'Glasses', Icon: Glasses, tone: 'text-orange-600 dark:text-orange-400' },
  messenger: { label: 'Messenger', Icon: MessageCircle, tone: 'text-blue-600 dark:text-blue-400' },
  agent: { label: 'Agent', Icon: Bot, tone: 'text-violet-600 dark:text-violet-400' },
}

// The devices by their own names (turn_source.KIND_LABELS, without the maker).
const DEVICE_KINDS: Record<string, string> = { sticks3: 'StickS3', stopwatch: 'StopWatch', papercolor: 'PaperColor' }

/** A channel's name, icon and colour; "device:<kind>" is a device's. */
export function channelInfo(id: string) {
  return CHANNELS[id] || CHANNELS[id.split(':')[0]] || null
}

/** "device:stopwatch" → "Device · StopWatch". */
export function channelLabel(id: string): string {
  const info = channelInfo(id)
  if (!info) return id || 'Unknown'
  const sub = id.includes(':') ? id.slice(id.indexOf(':') + 1) : ''
  return sub ? `${info.label} · ${DEVICE_KINDS[sub] || sub}` : info.label
}

export const STATUS_LABEL: Record<string, string> = {
  open: 'Open', done: 'Done', failed: 'Failed', dismissed: 'Dismissed', not_task: 'Not a task', maybe: 'Maybe',
}
export const STATUS_DOT: Record<string, string> = {
  open: 'bg-sky-500', done: 'bg-emerald-500', failed: 'bg-red-500', dismissed: 'bg-zinc-500', not_task: 'bg-zinc-600', maybe: 'bg-amber-500',
}
export const KIND_LABEL: Record<string, string> = {
  request: 'Request', sent: 'Sent', scheduled: 'Scheduled task', proposal: 'Proposal', follow_up: 'Follow-up',
  held_reply: 'Held reply', link: 'Linked system', todo: 'Agent to-do', message: 'Message',
}
export const DIRECTION_LABEL: Record<string, string> = { received: 'Received', sent: 'Sent', internal: 'Internal' }
export const EVIDENCE_LABEL: Record<string, string> = { verified: 'Recorded', claimed: 'Agent says', none: 'No record' }
export const RESULT_LABEL: Record<string, string> = {
  success: 'Succeeded', partial: 'Partly done', failed: 'Failed', unknown: 'Result unknown', '': 'No result yet',
}
export const OWES_LABEL: Record<string, string> = {
  me: 'You', agent: 'The agent', them: 'Someone else', nobody: 'Nobody', unclear: 'Not clear',
}
export const ROLE_LABEL: Record<string, string> = {
  requester: 'From', doer: 'Done by', recipient: 'To', cc: 'Cc', mentioned: 'Mentioned',
}
export const EVENT_LABEL: Record<string, string> = {
  received: 'Received', reply: 'Reply', sent: 'Sent', tool_call: 'Tool call', delivery: 'Delivered', status: 'Status',
  proposal: 'Proposed', approval: 'Approved', rejection: 'Rejected', decision: 'Decided', correction: 'You changed it',
  note: 'Note',
}

// ── People ──

/** Who is looking at the board: their id and name (both '' when sign-in is
 *  off — the server's `me` still says which party is them). */
export interface BoardUser { id: string; name: string }

/** Whether a party is the board user: the server says so; a deck user's
 *  party with the board user's own id, when it doesn't. */
export function isMe(p: TaskParty, me?: BoardUser): boolean {
  if (typeof p.me === 'boolean') return p.me
  return p.kind === 'fd_user' && !!me?.id && p.address === me.id
}

/** A party's name: "You" for you, and never a deck user's id (another deck
 *  user without a name is "Someone else"). v2: the person's name first (a
 *  party is a person, whichever handle this is), else the source's. */
export function partyName(p: TaskParty, me?: BoardUser): string {
  if (isMe(p, me)) return 'You'
  const display = (p.name || p.display || '').trim()
  if (p.kind === 'fd_user') return display && display !== p.address && display !== 'You' ? display : 'Someone else'
  return display || p.address
}

const nameKey = (s: string) => foldText(s).replace(/\s+/g, ' ').trim()

/** The address shown beside a party's name: none for a deck user (their id)
 *  or for a bare name (its address is the name, folded), nor when it only
 *  repeats the name. */
export function partyAddress(p: TaskParty, me?: BoardUser): string {
  if (p.kind === 'fd_user' || p.kind === 'name' || isMe(p, me)) return ''
  const display = (p.name || p.display || '').trim()
  return display && p.address && nameKey(display) !== nameKey(p.address) ? p.address : ''
}

/** The parties worth listing, in their order. A party read out of the text
 *  (not recorded) adds nothing when someone of the same name — folded, so
 *  "Ada Admin" and "ada admin" — is already there: a recorded party (you go
 *  by your own name too), or an earlier one read out of the text. A recorded
 *  party is dropped only when it is the same one twice. */
export function distinctParties(parties: TaskParty[] | undefined, me?: BoardUser): TaskParty[] {
  const list = parties ?? []
  const recorded = (p: TaskParty) => !!p.verified || isMe(p, me)
  const keys = (p: TaskParty) => [(p.name || p.display || '').trim() || p.address, isMe(p, me) ? me?.name || '' : '']
    .map(nameKey).filter(Boolean)
  const strong = new Set(list.filter(recorded).flatMap(keys))
  const names = new Set<string>()
  const ids = new Set<string>()
  return list.filter((p) => {
    const ks = keys(p)
    if (recorded(p)) {
      const id = `${p.role}\u0000${p.kind}\u0000${p.address}`
      if (ids.has(id)) return false
      ids.add(id)
      return true
    }
    if (!ks.length || ks.some((k) => strong.has(k) || names.has(k))) return false
    ks.forEach((k) => names.add(k))
    return true
  })
}

/** A People filter value's name: the server's id is `party:<id>` for a
 *  person (v2) or `kind:address` for a handle no person has yet, its label
 *  the name — "You" for you (the server's `me` on the value, so whatever
 *  tasks the page holds), never a deck user's id. */
export function partyFacetLabel(id: string, label: string, me?: BoardUser, isYou?: boolean): string {
  if (id.startsWith('party:') && isYou !== true) return label.trim() || 'Someone'
  const at = id.indexOf(':')
  const party: TaskParty = {
    role: '', kind: at < 0 ? '' : id.slice(0, at), address: at < 0 ? id : id.slice(at + 1), display: label, verified: 0,
    ...(typeof isYou === 'boolean' ? { me: isYou } : {}),
  }
  return partyName(party, me)
}

/** A person on a task: their name, their role, and every handle they were
 *  reached by on it (an email address, a WhatsApp number…) — one row for
 *  one person in one role, however many handles. */
export interface PersonRow {
  key: string
  role: string
  name: string
  me: boolean
  verified: boolean
  handles: { kind: string; address: string }[]
}

/** The People list: the parties worth listing (distinctParties), the same
 *  person (`party_id`) in the same role once with all their handles. */
export function peopleRows(parties: TaskParty[] | undefined, me?: BoardUser): PersonRow[] {
  const rows: PersonRow[] = []
  const byPerson = new Map<string, PersonRow>()
  for (const p of distinctParties(parties, me)) {
    const person = p.party_id ? `${p.role}\u0000${p.party_id}` : ''
    let row = person ? byPerson.get(person) : undefined
    if (!row) {
      row = {
        key: person || `${p.role}\u0000${p.kind}\u0000${p.address}\u0000${rows.length}`,
        role: p.role, name: partyName(p, me), me: isMe(p, me), verified: !!p.verified, handles: [],
      }
      rows.push(row)
      if (person) byPerson.set(person, row)
    } else {
      row.verified ||= !!p.verified
      if (isMe(p, me)) { row.me = true; row.name = 'You' }
    }
    const address = partyAddress(p, me)
    if (address && !row.handles.some((h) => h.address === address)) row.handles.push({ kind: p.kind, address })
  }
  return rows
}

// The channel a handle is reached on, for its icon.
const HANDLE_CHANNEL: Record<string, string> = {
  email: 'email', waid: 'whatsapp', wa_jid: 'waweb', tg_id: 'telegram', telegram: 'telegram', link: 'link',
}
export const handleChannel = (kind: string): string => HANDLE_CHANNEL[kind] || ''

/** How many of a task's events a split can move: any but a correction (the
 *  user's own word, and the board's merge and split notes, stay on the task
 *  they are about) — those past the latest the timeline holds count as such. */
export function movableEvents(events: { kind: string }[] | undefined, total?: number): number {
  const held = events ?? []
  return held.filter((e) => e.kind !== 'correction').length + Math.max(0, (total ?? held.length) - held.length)
}

// ── Next steps ──

/** What a board proposal does, in two or three words: "Draft email",
 *  "Reply on WhatsApp", "Reminder". */
export function proposalLabel(p: { kind?: string; payload?: { channel?: string } }): string {
  if (p.kind === 'mail.draft') return 'Draft email'
  if (p.kind === 'reply.on_origin') {
    const channel = p.payload?.channel || ''
    return channel ? `Reply on ${channelLabel(channel)}` : 'Reply'
  }
  if (p.kind === 'nudge') return 'Reminder'
  return 'Next step'
}

/** A recipient as one line: "Ana Horvat <ana@example.com>", "Ana Horvat
 *  (+385911234567)" for a WhatsApp number, the name alone for a handle
 *  nobody reads (a Telegram id, a chat jid) — or whichever there is. */
export function recipientText(r: { address: string; display: string; kind?: string }): string {
  const name = (r.display || '').trim()
  if (!name || nameKey(name) === nameKey(r.address)) return r.address || name
  if (!r.address) return name
  if (r.kind === 'waid' || r.kind === 'phone') {
    const number = `+${r.address.replace(/\D/g, '')}`
    return name.replace(/\D/g, '') === number.slice(1) ? name : `${name} (${number})`
  }
  return !r.kind || r.kind === 'email' ? `${name} <${r.address}>` : name
}

// ── Notes ──

/** What a task's flags add to its lane, in a few words: the agent couldn't
 *  do it (and why), only partly did it, or someone wrote again after it was
 *  done. For the row's reason line and the detail's chips. */
export interface TaskNote {
  id: 'agent_declined' | 'partly_done' | 'new_since_done'
  label: string
  /** Why not ('' when the summary didn't say). */
  detail: string
}

export function taskNotes(t: { status: string; flags?: TaskFlags | null }): TaskNote[] {
  const f = t.flags ?? {}
  const blocker = typeof f.blocker === 'string' ? f.blocker.trim() : ''
  const out: TaskNote[] = []
  if (f.agent_declined) out.push({ id: 'agent_declined', label: 'Agent couldn\'t do it', detail: blocker })
  else if (f.partly_done) out.push({ id: 'partly_done', label: 'Partly done', detail: blocker })
  if (f.new_since_done && t.status === 'done') out.push({ id: 'new_since_done', label: 'New message since done', detail: '' })
  return out
}

/** "Answered in chat" — not for a reply that said it couldn't, or did only
 *  part of it. */
export function answeredInChat(flags: TaskFlags | null | undefined): boolean {
  return !!flags?.answered_in_chat && !flags.agent_declined && !flags.partly_done
}

// ── Decisions ──

export const JUDGE_LABEL: Record<string, string> = { jev: 'Jev', rule: 'Rule', model: 'Model', user: 'You' }

// Jev's questions (taskboard.v1), as a person would ask them.
const QUESTION_LABEL: Record<string, string> = {
  is_task: 'A task', owes_next: 'Next move', fulfilled: 'Finished', result: 'Result', urgency: 'Urgency',
  asks_assistant_to_act: 'Asks the agent to act', reopen: 'Reopen',
}
const CHOICE_LABEL: Record<string, string> = {
  ...OWES_LABEL, success: 'succeeded', partial: 'partly done', failed: 'failed', not_attempted: 'not tried',
}

/** One answer, short: "A task 92%", "Next move: The agent 71%", "Urgency 2". */
export function answerText(question: string, a: TaskAnswer): string {
  const q = QUESTION_LABEL[question] || question.replace(/_/g, ' ')
  // Stored answers are the server's: anything missing is left out, not NaN.
  const pct = (p: unknown) => (typeof p === 'number' && Number.isFinite(p)
    ? ` ${Math.round(Math.max(0, Math.min(1, p)) * 100)}%` : '')
  if (!a || typeof a !== 'object') return q
  if ('choice' in a) return `${q}: ${CHOICE_LABEL[a.choice] || a.choice}${pct(a.p)}`
  if ('score' in a) return typeof a.score === 'number' ? `${q} ${Math.round(a.score * 10) / 10}` : q
  return `${q}${pct((a as { p?: unknown }).p)}`
}

// ── Time ──

/** How long ago, never "in a moment" for a server clock a little ahead. */
// ── Keyboard: Gmail's keys wherever a button means the same ───────────

/** What the open task's buttons answer to, as `KeyboardEvent.key` (so however
 *  a keyboard's layout types it): Gmail's archive, mark read and unread, spam,
 *  snooze, label, move to, reply and undo. Approving a next step has no key. */
export const TASK_KEYS = {
  done: 'e', reopen: 'U', gotIt: 'I', notTask: '!', yes: 'y', snooze: 'b', rename: 'l', merge: 'v', nextStep: 'r', undo: 'z',
  agent: 'a',
} as const

/** `g`, then one of these within {@link GO_WAIT_MS}: Gmail's "go to". */
export const GO_KEYS: Record<string, TaskView> = { i: 'needs_me', n: 'needs_me', o: 'waiting', w: 'working', d: 'done', a: 'all' }
export const GO_WAIT_MS = 1500

/** How a key reads on a button: a capital is typed with Shift. */
export function keyLabel(key: string): string {
  return /^[A-Z]$/.test(key) ? `⇧${key}` : key
}

/** The `?` sheet. */
export const KEY_SHEET: { title: string; rows: { keys: string[]; what: string }[] }[] = [
  { title: 'Move around', rows: [
    { keys: ['j', 'k'], what: 'Next or previous task' },
    { keys: ['o', 'Enter'], what: 'Open the task' },
    { keys: ['u'], what: 'Back to the list' },
    { keys: ['/'], what: 'Search' },
    { keys: ['g n'], what: 'Go to Needs me' },
    { keys: ['g o'], what: 'Go to Waiting on others' },
    { keys: ['g w'], what: 'Go to Agents working' },
    { keys: ['g d'], what: 'Go to Done' },
    { keys: ['g a'], what: 'Go to All' },
    { keys: ['?'], what: 'This list' },
  ] },
  { title: 'On the open task', rows: [
    { keys: [TASK_KEYS.done], what: 'Done' },
    { keys: [TASK_KEYS.reopen], what: 'Reopen' },
    { keys: [TASK_KEYS.gotIt], what: 'Got it' },
    { keys: [TASK_KEYS.notTask], what: 'Not a task' },
    { keys: [TASK_KEYS.yes], what: 'Yes, it is a task' },
    { keys: [TASK_KEYS.snooze], what: 'Snooze, then 1–3 to pick how long' },
    { keys: [TASK_KEYS.rename], what: 'Rename' },
    { keys: [TASK_KEYS.merge], what: 'Merge into…' },
    { keys: [TASK_KEYS.nextStep], what: 'Review or suggest a next step' },
    { keys: [TASK_KEYS.agent], what: 'Hand to an agent' },
    { keys: [TASK_KEYS.undo], what: 'Undo last change' },
    { keys: ['Esc'], what: 'Close what is on top' },
  ] },
]

/** The key a press means to the board — '' when it isn't the board's: typed
 *  into a field or a list, with a modifier held (Cmd+K and the like), while
 *  composing, or from somewhere outside the board (*root*). Enter only from
 *  nowhere in particular: on a button, it presses that button. */
export function boardKey(
  e: { key: string; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean; isComposing?: boolean; defaultPrevented?: boolean
    target?: unknown },
  root?: { contains(el: never): boolean } | null,
): string {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing || e.defaultPrevented) return ''
  const el = e.target as { tagName?: string; isContentEditable?: boolean } | null | undefined
  const tag = el?.tagName ?? ''
  if (el?.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return ''
  const nowhere = !tag || tag === 'BODY'
  if (root && !nowhere && !root.contains(el as never)) return ''
  if (e.key === 'Enter') return nowhere ? 'Enter' : ''
  return e.key.length === 1 ? e.key : ''
}

/** The row `j` (*step* 1) or `k` (-1) goes to, -1 for none. From the open task;
 *  one that just left the list (Done, on Needs me) leaves its place to the next,
 *  which `j` takes; with none open, from *last*, where the cursor was. */
export function stepTo(ids: string[], openId: string | null, last: number, step: 1 | -1): number {
  if (!ids.length) return -1
  const at = openId ? ids.indexOf(openId) : -1
  if (openId && at < 0) {
    const place = Math.min(Math.max(last, 0), ids.length)
    return step > 0 ? Math.min(place, ids.length - 1) : Math.max(place - 1, 0)
  }
  const from = at >= 0 ? at : last
  if (from < 0) return 0
  const to = from + step
  return to >= ids.length || to < 0 ? -1 : to
}

export function agoRelative(epoch: number): string {
  // fmtRelative reads the clock again, so a time clamped to "now" can still
  // come out a hair ahead of it ("in a moment"): the last 45 s are "just now".
  return epoch > Date.now() / 1000 - 45 ? 'just now' : fmtRelative(epoch)
}

/** Whether a snooze is still on. */
export function isSnoozed(until: number | null | undefined): boolean {
  return !!until && until > Date.now() / 1000
}

export type DatePreset = '' | 'today' | '7d' | '30d' | 'custom'
/** `from` and `to` are local dates (yyyy-mm-dd), used by `custom`. */
export interface DateRange { preset: DatePreset; from: string; to: string }
export const NO_DATES: DateRange = { preset: '', from: '', to: '' }

/** The range as epoch seconds: days start at local midnight, and `to`
 *  takes in its whole day. */
export function dateBounds(r: DateRange): { from?: number; to?: number } {
  const secs = (d: Date) => Math.floor(d.getTime() / 1000)
  const now = new Date()
  const daysAgo = (n: number) => new Date(now.getFullYear(), now.getMonth(), now.getDate() - n)
  const day = (iso: string, plus = 0) => {
    const [y, m, d] = iso.split('-').map(Number)
    return y && m && d ? new Date(y, m - 1, d + plus) : null
  }
  if (r.preset === 'today') return { from: secs(daysAgo(0)) }
  if (r.preset === '7d') return { from: secs(daysAgo(6)) }
  if (r.preset === '30d') return { from: secs(daysAgo(29)) }
  if (r.preset === 'custom') {
    const from = r.from ? day(r.from) : null
    const end = r.to ? day(r.to, 1) : null
    return { from: from ? secs(from) : undefined, to: end ? secs(end) - 1 : undefined }
  }
  return {}
}
