/**
 * Extra data a person hands their work (`captain_claw/flight_deck/context_routes.py`):
 * a file, something pasted, an agent's files, files from Flight Deck's folders, a stretch
 * of a conversation. Each is kept by the deck first ("staged") and its id goes with what
 * it's for: a Crew task or message, a First Mate factory, one of its runs.
 */

import { refreshAccessToken, useAuthStore } from '../stores/authStore'
import { api } from './fileLibraryApi'

/** Kept for the person until it's sent somewhere or goes stale (a day). */
export interface Staged {
  id: string
  name: string
  size: number
  type: string
  /** Where it came from, in words: "Pasted", "Files of Researcher", "Flight Deck · research". */
  origin?: string
}

export interface ContextAgent { id: string; name: string }
export interface AgentFile { path: string; name: string; size: number; modified: number }
/** A chat with an agent from Flight Deck's log: `id` is the lane. */
export interface Conversation { id: string; agent: string; lane: string; updated_at: string }
/** One of its sessions (the timeline's topics); `id` 0 is a log from before topics. */
export interface ConversationSession { id: number; title: string; started_at: string; count: number | null }
export interface ConversationMessage { id: number; role: 'user' | 'assistant'; at: string; text: string; chars: number }

const post = <T>(path: string, body: unknown) => api<T>(path, { method: 'POST', body: JSON.stringify(body) })

/** A multipart body, so not through api(), which says it's JSON. Renews a token that ran out, as that does. */
async function upload(file: File): Promise<Staged> {
  const send = () => {
    const { token } = useAuthStore.getState()
    const form = new FormData()
    form.append('file', file)
    return fetch('/fd/context/uploads', { method: 'POST', body: form, credentials: 'include', headers: token ? { Authorization: `Bearer ${token}` } : {} })
  }
  let r = await send()
  if (r.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) r = await send()
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = typeof body?.detail === 'string' ? body.detail : ''
    } catch { /* not JSON: a proxy's own refusal, say */ }
    throw new Error(detail || (r.status === 413 ? 'That file is too big.' : `${r.status} ${r.statusText}`))
  }
  return r.json()
}

export const contextApi = {
  upload,
  drop: (id: string) => api<{ ok: boolean }>(`/fd/context/uploads/${id}`, { method: 'DELETE' }),
  paste: (text: string, name = '') => post<Staged>('/fd/context/uploads/text', { text, name }),
  /** Files from one of Flight Deck's folders (`owner` '' = the person's own): all or none. */
  deckFiles: (project: string, owner: string, paths: string[]) =>
    post<{ files: Staged[] }>('/fd/context/uploads/deck-files', { project, owner, paths }).then((r) => r.files),
  agents: () => api<{ agents: ContextAgent[] }>('/fd/context/agents').then((r) => r.agents),
  agentFiles: (agent: string) => api<{ files: AgentFile[]; more: boolean }>(`/fd/context/agents/${encodeURIComponent(agent)}/files`),
  /** Copies them in: all or none. */
  attachAgentFiles: (agent: string, paths: string[]) =>
    post<{ files: Staged[] }>('/fd/context/uploads/agent-files', { agent, paths }).then((r) => r.files),
  conversations: () => api<{ conversations: Conversation[]; kept: boolean }>('/fd/context/conversations'),
  sessions: (lane: string) =>
    api<{ sessions: ConversationSession[] }>(`/fd/context/conversations/sessions?${new URLSearchParams({ lane })}`).then((r) => r.sessions),
  messages: (lane: string, session: number) =>
    api<{ messages: ConversationMessage[]; more: boolean }>(`/fd/context/conversations/messages?${new URLSearchParams({ lane, session: String(session) })}`),
  /** Messages first..last of one session, written out as a Markdown file. */
  conversation: (lane: string, session: number, first: number, last: number) =>
    post<Staged>('/fd/context/uploads/conversation', { lane, session, first, last }),
}
