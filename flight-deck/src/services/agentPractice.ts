/** Practice — the owner tries an agent on real tasks and corrects it, through Flight Deck
 *  (/fd/agent-practice/{slug}). It answers on its private exam lane (nothing sent, nothing kept);
 *  a correction becomes a lesson (agentLessons). */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import type { Lookup } from './agentExam'

export interface PracticeAttempt {
  id: string
  state: 'answering' | 'done' | 'error'
  started: number
  finished?: number
  answer?: string
  error?: string
  lookups?: Lookup[]
  tokens?: number
  seconds?: number
  /** good | fix | "" — and what it should have done, and the lesson taught for it. */
  verdict: '' | 'good' | 'fix'
  note?: string
  lesson?: string
}

export interface PracticeSource { doc?: string; doc_title?: string; title?: string; path?: string[] }

export interface PracticeTask {
  id: string
  task: string
  /** What its documents say (a suggested task). */
  expected: string
  source: PracticeSource
  created: number
  attempts: PracticeAttempt[]
}

export interface PracticeSuggestion { task: string; expected: string; source: PracticeSource }

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
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
    throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status}`)
  }
  return res.json()
}

const base = (slug: string) => `/fd/agent-practice/${encodeURIComponent(slug)}`

export const agentPracticeApi = {
  list: (slug: string) => call<{ tasks: PracticeTask[] }>(base(slug)),
  add: (slug: string, body: { task: string; expected?: string; source?: PracticeSource }) =>
    call<PracticeTask>(base(slug), { method: 'POST', body: JSON.stringify(body) }),
  again: (slug: string, id: string) => call<PracticeTask>(`${base(slug)}/${encodeURIComponent(id)}/try`, { method: 'POST' }),
  mark: (slug: string, id: string, body: { attempt: string; verdict: '' | 'good' | 'fix'; note?: string; lesson?: string }) =>
    call<PracticeTask>(`${base(slug)}/${encodeURIComponent(id)}/mark`, { method: 'POST', body: JSON.stringify(body) }),
  forget: (slug: string, id: string) => call<{ ok: boolean }>(`${base(slug)}/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  suggest: (slug: string, focus = '') =>
    call<{ tasks: PracticeSuggestion[] }>(`${base(slug)}/suggest`, { method: 'POST', body: JSON.stringify({ focus }) }),
}
