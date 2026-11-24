import { api } from './fileLibraryApi'

// Questions for the person, answered in their Flight Deck: a custom tool's
// (captain.ask) or an agent's (its ask_user tool). Every signed-in tab waits on
// GET /fd/asks and shows the first open one (components/common/PersonAsks.tsx).

export interface AskQuestion {
  id: string
  question: string
  // "secret": a password or a code (request_secret), typed into a masked box;
  // the agent gets a reference, never the value.
  kind: 'text' | 'choice' | 'multiple' | 'confirm' | 'secret' | 'number'
  help?: string
  optional: boolean
  choices?: string[]
  min?: number
  max?: number
  lines?: number
  default?: unknown
  // The asker's suggested answer (an agent's, from the chat, its memory or the
  // person's profile) and where it came from: greyed in the empty field, taken
  // with Tab (a tap on a phone). Never an answer until taken.
  suggested?: unknown
  suggested_from?: string
  // text: its length and what it must look like
  min_length?: number
  max_length?: number
  format?: 'email' | 'uri' | 'date' | 'date-time'
  // number: its range, and whether it's a whole number
  min_value?: number
  max_value?: number
  integer?: boolean
}

export interface OpenAsk {
  id: string
  // "tool": a custom tool running for the person asks; "agent": an agent asks (ask_user);
  // "approval": an agent asks before it pays, sends, posts, shares, deletes… (approvals.py);
  // "mcp": an MCP server one of their agents is using asks before it goes on (mcp_elicitation.py).
  source: 'tool' | 'agent' | 'approval' | 'mcp'
  /** An approval card's facts: its kind and what the call will do; an MCP
   *  server's question: the server and its message. */
  detail?: { kind?: string; kind_label?: string; summary?: string; mode?: string; server?: string; message?: string }
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
