import {
  AlarmClock, Bot, Code2, Glasses, Link2, Mail, MessageCircle, MessagesSquare, Plug, Send, Smartphone, Sparkles, Users,
} from 'lucide-react'
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

const CHANNELS: Record<string, { label: string; Icon: typeof Mail; tone: string }> = {
  web: { label: 'Web chat', Icon: MessagesSquare, tone: 'text-violet-600 dark:text-violet-400' },
  whatsapp: { label: 'WhatsApp', Icon: MessageCircle, tone: 'text-emerald-600 dark:text-emerald-400' },
  waweb: { label: 'WhatsApp Web', Icon: MessageCircle, tone: 'text-emerald-600 dark:text-emerald-400' },
  telegram: { label: 'Telegram', Icon: Send, tone: 'text-sky-600 dark:text-sky-400' },
  tg_inbox: { label: 'Telegram inbox', Icon: Send, tone: 'text-sky-600 dark:text-sky-400' },
  gmail: { label: 'Gmail', Icon: Mail, tone: 'text-rose-600 dark:text-rose-400' },
  email: { label: 'Email', Icon: Mail, tone: 'text-rose-600 dark:text-rose-400' },
  device: { label: 'Device', Icon: Smartphone, tone: 'text-orange-600 dark:text-orange-400' },
  link: { label: 'Linked system', Icon: Link2, tone: 'text-cyan-600 dark:text-cyan-400' },
  scheduled: { label: 'Scheduled', Icon: AlarmClock, tone: 'text-amber-600 dark:text-amber-400' },
  autonomy: { label: 'Autonomous Work', Icon: Sparkles, tone: 'text-fuchsia-600 dark:text-fuchsia-400' },
  member: { label: 'Member', Icon: Users, tone: 'text-indigo-600 dark:text-indigo-400' },
  api: { label: 'API', Icon: Code2, tone: 'text-teal-600 dark:text-teal-400' },
  mcp: { label: 'MCP', Icon: Plug, tone: 'text-teal-600 dark:text-teal-400' },
  glasses: { label: 'Glasses', Icon: Glasses, tone: 'text-orange-600 dark:text-orange-400' },
  messenger: { label: 'Messenger', Icon: MessageCircle, tone: 'text-blue-600 dark:text-blue-400' },
  agent: { label: 'Agent', Icon: Bot, tone: 'text-violet-600 dark:text-violet-400' },
}

/** A channel's name, icon and colour; "device:<kind>" is a device's. */
export function channelInfo(id: string) {
  return CHANNELS[id] || CHANNELS[id.split(':')[0]] || null
}

/** "device:stopwatch" → "Device · stopwatch". */
export function channelLabel(id: string): string {
  const info = channelInfo(id)
  if (!info) return id || 'Unknown'
  const sub = id.includes(':') ? id.slice(id.indexOf(':') + 1) : ''
  return sub ? `${info.label} · ${sub}` : info.label
}

export const STATUS_LABEL: Record<string, string> = {
  open: 'Open', done: 'Done', failed: 'Failed', dismissed: 'Dismissed', not_task: 'Not a task', maybe: 'Maybe a task',
}
export const STATUS_DOT: Record<string, string> = {
  open: 'bg-sky-500', done: 'bg-emerald-500', failed: 'bg-red-500', dismissed: 'bg-zinc-500', not_task: 'bg-zinc-600', maybe: 'bg-amber-500',
}
export const KIND_LABEL: Record<string, string> = {
  request: 'Request', sent: 'Sent', scheduled: 'Scheduled task', proposal: 'Proposal', follow_up: 'Follow-up',
  held_reply: 'Held reply', link: 'Linked system', todo: 'To-do', message: 'Message',
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

// ── Time ──

/** How long ago, never "in a moment" for a server clock a little ahead. */
export function agoRelative(epoch: number): string {
  return fmtRelative(Math.min(epoch, Date.now() / 1000))
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
