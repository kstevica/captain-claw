/** An agent's onboarding exams, through Flight Deck (/fd/agent-exam/{slug}):
 *  questions written from its documents, asked through its real chat in
 *  private sessions that keep nothing, and graded against the section each
 *  came from — with why a wrong one went wrong. */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { actAsHeaders } from './onboardingAs'

export type ExamSize = 'quick' | 'standard' | 'deep'
export type ExamKind = 'ask' | 'situation'
export type Verdict = 'right' | 'partly' | 'wrong' | 'no_answer'
/** Why a question went wrong: it didn't look, looked in the wrong place, found it and still got it
 *  wrong, or made things up. */
export type Cause = 'didnt_look' | 'wrong_place' | 'got_wrong' | 'made_up'

export interface Lookup {
  tool: string
  args: Record<string, unknown>
  /** The start of what it got back. */
  result: string
  ok: boolean
}

export interface ExamQuestion {
  id: string
  kind: ExamKind
  question: string
  /** The answer its section gives, and what a reply must contain. */
  expected: string
  points: string[]
  /** Its section: the document (its file in knowledge/), its title, the section's title and parents. */
  doc: string
  doc_title: string
  /** The name a setup gave the document ("" when none did): how Way of working reads it again. */
  doc_name: string
  title: string
  path: string[]
  key: string
  /** The setup read that section (null: not known). */
  read: boolean | null
  /** What the agent replied, and what it looked up on the way. */
  answer?: string
  lookups?: Lookup[]
  seconds?: number
  error?: string
  verdict?: Verdict
  made_up?: boolean
  /** The details the grader found outside the question's passage, and where else they are ('' — nowhere: made up). */
  details?: { text: string; found: '' | 'documents' | 'lessons' | 'looked_up' }[]
  missing?: string[]
  why?: string
  cause?: Cause | ''
  /** Asked again: its verdict the time before. */
  before?: Verdict
  /** Taught to it as a lesson, after this exam. */
  taught?: boolean
}

export interface Score { questions: number; points: number; pct: number | null; right: number }

export interface ExamSummary extends Score {
  docs: (Score & { doc: string; title: string })[]
  kinds: Record<ExamKind, Score>
  read: { read: Score; unread: Score }
  causes: Record<Cause, number>
  made_up: number
  /** Asked again: how many went better or worse than the time before, of how many compared. */
  changed?: { better: number; worse: number; compared: number }
}

export interface Exam {
  id: string
  agent: string
  agent_name: string
  size: ExamSize
  focus: string
  /** The exam whose questions it asks again. */
  again: string
  /** ready: passed on with the archetype it was made from — its questions, waiting to be asked. */
  state: 'running' | 'done' | 'stopped' | 'error' | 'ready'
  phase: 'writing' | 'asking' | 'grading'
  /** Where it is in its phase. */
  step: number
  steps: number
  model: string
  tokens: number
  started: number
  finished: number | null
  error: string
  notes: string[]
  waiting: { why: string; until?: number; name?: string; model?: string } | null
  summary: ExamSummary | null
  /** Only with the whole exam. */
  questions?: ExamQuestion[]
}

export interface ExamOverview {
  agent: string
  name: string
  exams: Exam[]
  running: Exam | null
  /** What it knows: its documents and their sections worth a question. */
  docs: number
  sections: number
  /** Who answers and writes: its own model. */
  model: string
  /** About what each size costs, in tokens. */
  cost: Record<ExamSize, number>
  sizes: Record<ExamSize, number>
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    return { ...h, ...actAsHeaders() }
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

const base = (slug: string) => `/fd/agent-exam/${encodeURIComponent(slug)}`

export const agentExamApi = {
  overview: (slug: string) => call<ExamOverview>(base(slug)),
  start: (slug: string, body: { size?: ExamSize; focus?: string; again?: string }) =>
    call<Exam>(base(slug), { method: 'POST', body: JSON.stringify(body) }),
  get: (slug: string, id: string) => call<Exam>(`${base(slug)}/${encodeURIComponent(id)}`),
  stop: (slug: string, id: string) => call<{ ok: boolean }>(`${base(slug)}/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  resume: (slug: string, id: string) =>
    call<Exam>(`${base(slug)}/${encodeURIComponent(id)}/continue`, { method: 'POST' }),
  teach: (slug: string, id: string, question: string) =>
    call<{ ok: boolean; taught: boolean }>(`${base(slug)}/${encodeURIComponent(id)}/teach`, {
      method: 'POST', body: JSON.stringify({ question }),
    }),
}
