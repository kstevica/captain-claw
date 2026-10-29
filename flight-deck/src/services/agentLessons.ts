/** An agent's lessons — what its owner taught it: rules, things to ask first, facts — through
 *  Flight Deck (/fd/agent-lessons/{slug}). The always-in-mind ones are in its instructions; the
 *  rest reach it with the messages they bear on. A change can rewrite its instructions' block:
 *  the response carries them for the chat to send. */

import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { actAsHeaders } from './onboardingAs'

export type LessonKind = 'rule' | 'ask_first' | 'fact'

export interface Lesson {
  id: string
  kind: LessonKind
  text: string
  /** Its document and section ("" for the owner's own words). */
  source: { doc: string; section: string }
  always: boolean
  /** setup | exam | you | chat | practice | member */
  taught_by: string
  /** When it was taught (0: by a setup from before it was kept). */
  added: number
  importance: number
  /** "base": an admin taught it — setting it up for its user, or in the archetype it was made from. */
  layer: '' | 'base'
}

/** What a member of a shared agent suggested it learn, waiting for its owner. */
export interface LessonSuggestion {
  id: string
  /** The member's words, and the answer they corrected. */
  said: string
  reply: string
  by_name: string
  created: number
}

export interface LessonsView {
  lessons: Lesson[]
  counts: { rule: number; ask_first: number; fact: number; always: number }
  /** Always-in-mind lessons its instructions had no room for: they still reach it as lessons. */
  not_in_block: string[]
  /** The base's lessons can't be changed by this caller (its user, not an admin). */
  base_locked: boolean
  suggestions: LessonSuggestion[]
}

/** A lesson drafted from the owner's words — empty ``text`` when they teach nothing; ``drafted`` false when no
 *  model could (their words, as a rule). */
export interface LessonDraft { kind: LessonKind; text: string; always: boolean; drafted: boolean }

export interface LessonsChanged extends LessonsView {
  /** The instructions Flight Deck stored (with accounts), or the block for the browser to put in. */
  stored: boolean
  instructions: string
  block: string
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

const base = (slug: string) => `/fd/agent-lessons/${encodeURIComponent(slug)}`

/** Who taught a lesson by hand: the Lessons tab, a chat, a practice run, a member's suggestion. */
export type TaughtBy = 'you' | 'chat' | 'practice' | 'member'

export const agentLessonsApi = {
  list: (slug: string) => call<LessonsView>(base(slug)),
  add: (slug: string, body: { kind: LessonKind; text: string; always: boolean; taught_by?: TaughtBy }) =>
    call<LessonsChanged>(base(slug), { method: 'POST', body: JSON.stringify(body) }),
  draft: (slug: string, body: { said: string; reply?: string; task?: string }) =>
    call<LessonDraft>(`${base(slug)}/draft`, { method: 'POST', body: JSON.stringify(body) }),
  change: (slug: string, id: string, patch: { text?: string; always?: boolean }) =>
    call<LessonsChanged>(`${base(slug)}/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  remove: (slug: string, id: string) =>
    call<LessonsChanged>(`${base(slug)}/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  /** Its owner took a member's suggestion (with the lesson taught from it), or declined it. */
  settle: (slug: string, id: string, state: 'taken' | 'declined', lesson = '') =>
    call<{ suggestions: LessonSuggestion[] }>(`${base(slug)}/suggestions/${encodeURIComponent(id)}`,
      { method: 'POST', body: JSON.stringify({ state, lesson }) }),
  /** A member of someone's shared agent suggests a lesson to its owner. */
  suggest: (agentRef: string, said: string, reply: string) =>
    call<{ ok: boolean; already: boolean }>('/fd/agent-lessons/shared/suggest',
      { method: 'POST', body: JSON.stringify({ agent_ref: agentRef, said, reply }) }),
}
