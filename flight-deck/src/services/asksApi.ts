import { api } from './fileLibraryApi'

// Questions for the person, answered in their Flight Deck: a custom tool's
// (captain.ask) or an agent's (its ask_user tool). Every signed-in tab waits on
// GET /fd/asks and shows the first open one (components/common/PersonAsks.tsx).

export interface AskQuestion {
  id: string
  question: string
  kind: 'text' | 'choice' | 'multiple' | 'confirm'
  help?: string
  optional: boolean
  choices?: string[]
  min?: number
  max?: number
  lines?: number
  default?: unknown
}

export interface OpenAsk {
  id: string
  // "tool": a custom tool running for the person asks; "agent": an agent asks (ask_user);
  // "approval": an agent asks before it pays, sends, posts, shares, deletes… (approvals.py).
  source: 'tool' | 'agent' | 'approval'
  /** An approval card's facts: its kind and what the call will do. */
  detail?: { kind: string; kind_label: string; summary: string; mode: string }
  title: string
  owner_name: string
  agent_name: string
  label: string
  questions: AskQuestion[]
  created: number
  seconds_left: number
}

export const asksApi = {
  list: (known: string[], wait: number) =>
    api<{ asks: OpenAsk[] }>(`/fd/asks?wait=${wait}&known=${encodeURIComponent(known.join(','))}`),
  answer: (id: string, body: { answers?: Record<string, unknown>; declined?: boolean }) =>
    api<{ ok: boolean }>(`/fd/asks/${encodeURIComponent(id)}`, { method: 'POST', body: JSON.stringify(body) }),
}
