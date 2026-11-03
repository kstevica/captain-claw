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
  }
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
