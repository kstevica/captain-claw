/** What an agent remembers between conversations — its insights — read and
 *  changed through Flight Deck's /fd/agent-insights/{host}/{port} proxy (the
 *  agent's own /api/insights; only its owner gets through). */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

export type InsightCategory =
  | 'fact' | 'preference' | 'feedback' | 'decision' | 'contact' | 'deadline' | 'project' | 'workflow' | 'reference'

export interface Insight {
  id: string
  content: string
  category: InsightCategory
  importance: number
  tags: string | null
  entity_key: string | null
  /** How it was learned: owner_setup, web_ui, manual, a tool's name, or none (a conversation). */
  source_tool: string | null
  /** The file it came from (a document's name, a path, a URL), and where in it. */
  source_file?: string | null
  source_section?: string | null
  why?: string | null
  how_to_apply?: string | null
  polarity?: 'positive' | 'negative' | null
  created_at: string
  updated_at: string
}

/** What an owner may set. */
export interface InsightFields { content?: string; category?: InsightCategory; importance?: number; tags?: string }

export interface AgentAddress { host: string; port: number; auth: string }

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    return h
  }
  const go = () => fetch(path, { ...init, headers: headers(), credentials: 'include', cache: 'no-store' })
  let res = await go()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await go()
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : body.error || `${res.status}`)
  }
  return res
}

const base = (a: AgentAddress) => `/fd/agent-insights/${encodeURIComponent(a.host)}/${a.port}`
const tokenQs = (a: AgentAddress, more: Record<string, string> = {}) => {
  const qs = new URLSearchParams({ ...(a.auth ? { token: a.auth } : {}), ...more }).toString()
  return qs ? `?${qs}` : ''
}

/** Its insights, newest first — or the ones matching *q*. */
export async function listInsights(a: AgentAddress, q = ''): Promise<{ items: Insight[]; total: number }> {
  return (await send(`${base(a)}${tokenQs(a, q.trim() ? { q: q.trim() } : {})}`)).json()
}

/** Teach it something. `null`: it already knew (a near-identical insight exists). */
export async function addInsight(a: AgentAddress, fields: InsightFields): Promise<Insight | null> {
  const res = await send(`${base(a)}${tokenQs(a)}`, { method: 'POST', body: JSON.stringify(fields) })
  const body = await res.json()
  return body && body.id ? (body as Insight) : null
}

export async function editInsight(a: AgentAddress, id: string, fields: InsightFields): Promise<Insight> {
  return (await send(`${base(a)}/${id}${tokenQs(a)}`, { method: 'PATCH', body: JSON.stringify(fields) })).json()
}

export async function forgetInsight(a: AgentAddress, id: string): Promise<void> {
  await send(`${base(a)}/${id}${tokenQs(a)}`, { method: 'DELETE' })
}
