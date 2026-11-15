/**
 * The tool-call register — every tool call the deck's agents make: which agent,
 * when, which tool, with what, what started it and how it ended
 * (`captain_claw/flight_deck/tool_audit_routes.py`).
 *
 * `mine`: the signed-in user's own agents, and their own turns on someone
 * else's shared agent. `admin`: every user's.
 */

import { useAuthStore } from '../stores/authStore'
import { api } from './fileLibraryApi'

export type AuditScope = 'mine' | 'admin'
export type Risk = 'read' | 'write' | 'exec' | 'send' | 'money' | 'other'
export type Outcome = 'running' | 'ok' | 'error' | 'blocked' | 'timeout' | 'aborted' | 'interrupted'

export const RISKS: { id: Risk; label: string }[] = [
  { id: 'exec', label: 'Run code' },
  { id: 'send', label: 'Sends out' },
  { id: 'money', label: 'Money' },
  { id: 'write', label: 'Write' },
  { id: 'read', label: 'Read' },
  { id: 'other', label: 'Other' },
]

export const OUTCOMES: { id: Outcome; label: string }[] = [
  { id: 'ok', label: 'Done' },
  { id: 'error', label: 'Failed' },
  { id: 'blocked', label: 'Refused' },
  { id: 'timeout', label: 'Timed out' },
  { id: 'aborted', label: 'Stopped' },
  { id: 'interrupted', label: 'Interrupted' },
  { id: 'running', label: 'Running' },
]

export interface ToolCall {
  id: number
  /** When the call started (unix seconds, the agent's clock). */
  at: number
  /** When Flight Deck got the record (its own clock). */
  received_at: number
  agent: { slug: string; name: string; ref: string }
  owner: { id: string; name: string; you: boolean }
  /** Made in a shared-agent member's turn. */
  member: boolean
  speaker: { id: string; label: string; you: boolean }
  tool: string
  risk: Risk
  /** Masked parameters; null when hidden (a member's call, seen by the owner). */
  params: Record<string, unknown> | null
  hidden: boolean
  session_id: string
  source: string
  source_label: string
  outcome: Outcome
  reason: string
  duration_ms: number | null
  ended_at: number | null
  seq: number
  boot: string
  hash: string
}

export interface CallsPage {
  calls: ToolCall[]
  next_before: number | null
}

export interface CallQuery {
  agent?: string
  ref?: string
  tool?: string
  outcome?: Outcome | ''
  risk?: Risk | ''
  source?: string
  q?: string
  since?: number
  until?: number
  /** Admin only: one user's calls (as owner or as the member speaking). */
  user_id?: string
}

export interface Facets {
  agents: { slug: string; name: string; count: number }[]
  tools: { tool: string; count: number }[]
  sources: { source: string; label: string; count: number }[]
}

export interface Summary {
  total: number
  outcomes: Partial<Record<Outcome, number>>
  risks: Partial<Record<Risk, number>>
}

export interface VerifyResult {
  ok: boolean
  checked: number
  anchor_id: number
  last_id: number
  verified_at: number
  problems: { id: number; problem: 'link' | 'row' | 'outcome'; detail: string }[]
  gaps: {
    agent_slug: string; agent_name: string; owner_id: string; owner_name?: string; boot: string
    first_seq: number; last_seq: number; missing: number; first_at: number; last_at: number
  }[]
}

export interface AuditEvent {
  id: number
  at: number
  kind: 'pruned' | 'verified' | 'exported' | 'retention_changed' | string
  actor: string
  actor_name?: string
  detail: Record<string, unknown>
}

export interface AuditSettings { retention_days: number; min_days: number; max_days: number }

function qs(params: object): string {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') p.set(k, String(v))
  return p.toString()
}

const base = (scope: AuditScope) => (scope === 'admin' ? '/fd/admin/tool-audit' : '/fd/tool-audit')

function dispositionName(header: string | null): string {
  return (header && /filename\s*=\s*"?([^";]+)"?/i.exec(header)?.[1]?.trim()) || ''
}

/** "Why did this happen?" for one call (captain_claw/flight_deck/why.py): the
 *  deck's deciders re-run on its recorded facts, one verdict per layer. */
export interface WhyLayer {
  layer: 'what' | 'who' | 'approval' | 'guard' | 'outcome'
  title: string
  then: string
  today: string
  changed: boolean
}

export interface WhyAnswer {
  id: number
  summary: string
  layers: WhyLayer[]
  changed: boolean
}

export const toolAuditApi = {
  why: (scope: AuditScope, id: number) => api<WhyAnswer>(`${base(scope)}/calls/${id}/why`),
  calls: (scope: AuditScope, query: CallQuery & { before_id?: number; limit?: number } = {}) =>
    api<CallsPage>(`${base(scope)}/calls?${qs(query)}`),
  facets: (scope: AuditScope, opts: { since?: number; user_id?: string } = {}) =>
    api<Facets>(`${base(scope)}/facets?${qs(opts)}`),
  summary: (scope: AuditScope, query: CallQuery = {}) => api<Summary>(`${base(scope)}/summary?${qs(query)}`),
  verify: () => api<VerifyResult>('/fd/admin/tool-audit/verify'),
  events: () => api<{ events: AuditEvent[] }>('/fd/admin/tool-audit/events'),
  settings: () => api<AuditSettings>('/fd/admin/tool-audit/settings'),
  saveSettings: (retention_days: number) =>
    api<AuditSettings>('/fd/admin/tool-audit/settings', { method: 'PUT', body: JSON.stringify({ retention_days }) }),
  /** Save the matching calls as a file. Fetched rather than linked, so a
   *  refusal throws with its reason instead of being saved. */
  download: async (scope: AuditScope, query: CallQuery, format: 'csv' | 'jsonl') => {
    const jwt = useAuthStore.getState().token
    const r = await fetch(`${base(scope)}/export?${qs({ ...query, format })}`, {
      cache: 'no-store',
      credentials: 'include',
      headers: jwt ? { Authorization: `Bearer ${jwt}` } : {},
    })
    if (!r.ok) {
      const body = await r.json().catch(() => ({}))
      throw new Error((typeof body?.detail === 'string' && body.detail) || `${r.status} ${r.statusText}`)
    }
    const url = URL.createObjectURL(await r.blob())
    const a = document.createElement('a')
    a.href = url
    a.download = dispositionName(r.headers.get('content-disposition')) || `tool-calls.${format}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  },
}
