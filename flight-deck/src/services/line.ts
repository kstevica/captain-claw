// The Line: one timeline per user for everything the deck and the agents say
// on their own, backed by /fd/line. The bell is its Needs-you list.

import { fdFetch } from './notifications'

export interface LineItem {
  id: string
  kind: string
  /** The bell's word for it (payment, run_error, autonomy_proposal…). */
  type: string
  title: string
  body: string
  /** Who it's about, when the deck knows: an agent's name. */
  source: string
  needs_you: boolean
  resolved: boolean
  read: boolean
  ref_type: string
  ref_id: string
  /** Whose words the text is: the deck's, an agent's, or someone else's. */
  trust: 'deck' | 'agent' | 'untrusted' | string
  created_at: string
}

export interface LineCounts {
  unread: number
  /** Items that still wait for the user. */
  waiting: number
}

export interface LinePage extends LineCounts {
  items: LineItem[]
  /** The kinds the Line holds, by what the page calls them. */
  kinds: Record<string, string>
  /** The agent that answers on the Line ("" = none picked yet), and its slug. */
  agent: string
  agent_slug: string
  /** The user's agents an item can be handed to. */
  agents: { slug: string; name: string }[]
}

export const LINE_PAGE = 50

export async function fetchLine(opts: { before?: string; kind?: string; needsYou?: boolean } = {}): Promise<LinePage> {
  const q = new URLSearchParams({ limit: String(LINE_PAGE) })
  if (opts.before) q.set('before', opts.before)
  if (opts.kind) q.set('kind', opts.kind)
  if (opts.needsYou) q.set('needs_you', 'true')
  const r = await fdFetch(`/fd/line?${q}`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  const d = await r.json()
  return {
    items: Array.isArray(d?.items) ? d.items : [],
    unread: d?.unread || 0,
    waiting: d?.waiting || 0,
    kinds: d?.kinds && typeof d.kinds === 'object' ? d.kinds : {},
    agent: typeof d?.agent === 'string' ? d.agent : '',
    agent_slug: typeof d?.agent_slug === 'string' ? d.agent_slug : '',
    agents: Array.isArray(d?.agents) ? d.agents : [],
  }
}

async function _post<T>(path: string, body: unknown): Promise<T> {
  const r = await fdFetch(path, { method: 'POST', body: JSON.stringify(body) })
  if (!r.ok) {
    let detail = `HTTP ${r.status}`
    try {
      const d = await r.json()
      if (typeof d?.detail === 'string') detail = d.detail
    } catch { /* keep the status */ }
    throw new Error(detail)
  }
  return r.json()
}

/** Ask one of the user's agents (default: the one picked for the Line); the
 *  question is on the Line now, the answer when it comes. */
export function askLine(text: string, agent = ''): Promise<{ id: string; agent: string }> {
  return _post('/fd/line/ask', agent ? { text, agent } : { text })
}

/** Hand an item to one of the user's agents, with their own words; the answer lands on the Line. */
export function handLineItem(id: string, agent: string, note: string): Promise<{ id: string; agent: string }> {
  return _post(`/fd/line/${encodeURIComponent(id)}/hand`, { agent, note })
}

/** A daily summary of the Line by that agent: a scheduled task (one per user). */
export function addLineSummary(opts: { time: string; tz: string; telegram: boolean; email: boolean }):
  Promise<{ agent: string; schedule: string; updated: boolean }> {
  return _post('/fd/line/summary', opts)
}

export async function fetchLineCounts(): Promise<LineCounts | null> {
  try {
    const r = await fdFetch('/fd/line/unread')
    if (!r.ok) return null
    const d = await r.json()
    return { unread: d?.unread || 0, waiting: d?.waiting || 0 }
  } catch {
    return null
  }
}

export async function markLineRead(id: string): Promise<void> {
  try { await fdFetch(`/fd/line/${encodeURIComponent(id)}/read`, { method: 'POST' }) } catch { /* ignore */ }
}

export async function markAllLineRead(): Promise<void> {
  try { await fdFetch('/fd/line/read-all', { method: 'POST' }) } catch { /* ignore */ }
}

/** Opens the Line: the full view's page, or the kiosk's dialog. */
export const OPEN_LINE_EVENT = 'fd:open-line'
