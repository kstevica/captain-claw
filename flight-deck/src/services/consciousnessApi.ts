// REST client for the consciousness / Observatory API (FD-side, JWT-scoped).
// Mirrors the auth handling in flowsApi.ts.
import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const BASE = '/fd'

function _authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = {}
  if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = { ..._authHeaders(), ...(init?.headers as Record<string, string> | undefined) }
  let res = await fetch(`${BASE}${path}`, { ...init, headers, credentials: 'include' })
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    const ok = await refreshAccessToken()
    if (ok) {
      const h2 = { ..._authHeaders(), ...(init?.headers as Record<string, string> | undefined) }
      res = await fetch(`${BASE}${path}`, { ...init, headers: h2, credentials: 'include' })
    }
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || body.detail || `${res.status} ${res.statusText}`)
  }
  const text = await res.text()
  return (text ? JSON.parse(text) : {}) as T
}

// ── Data model ──

export type JournalKind = 'thought' | 'dream' | 'observation'

export interface JournalEntry {
  id: string
  kind: JournalKind
  content: string
  mood: string
  salience: number
  agents: string[]
  delta: string
  author: string
  created_at: string
}

export interface Intention {
  id: string
  content: string
  status: string
  created_at: string
  updated_at: string
}

export interface ConsciousnessState {
  user_id: string
  pulse_count: number
  thought_count: number
  last_pulse_at: number | null
  last_thought_at: number | null
}

export interface NarratorAgent {
  slug: string
  name: string
  model: string
  offline?: boolean
}

export interface ConsciousnessSnapshot {
  state: ConsciousnessState
  intentions: Intention[]
  journal: JournalEntry[]
  agents: NarratorAgent[]
  narrator: string
}

export interface NudgeResult {
  acted: boolean
  reason: string
  new_messages?: number
  new_sessions?: number
  agents?: number
  mood?: string
  entries?: JournalEntry[]
}

// ── API ──

export const getConsciousness = (limit = 80) =>
  fdFetch<ConsciousnessSnapshot>(`/consciousness?limit=${limit}`)

export const getJournalBefore = (before: string, limit = 80) =>
  fdFetch<{ journal: JournalEntry[] }>(
    `/consciousness/journal?before=${encodeURIComponent(before)}&limit=${limit}`,
  )

export const nudgeConsciousness = () =>
  fdFetch<NudgeResult>('/consciousness/nudge', { method: 'POST' })

export const setNarrator = (slug: string) =>
  fdFetch<{ ok: boolean; narrator: string }>('/consciousness/narrator', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug }),
  })
