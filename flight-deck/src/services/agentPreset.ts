// An agent's "way of working" setup, opened right after the kiosk creates it
// (AgentSetupDialog). POST /fd/agent-preset/prepare reads the person's
// instructions and documents — uploaded or picked from their File library,
// each with what it is and how to use it — with their own model into a preset
// they review; POST /fd/agent-preset/apply gives the agent what they kept.
import { useAuthStore, refreshAccessToken } from '../stores/authStore'
import { ACT_AS_HEADER } from './docker'

export interface PresetSource { doc: string; section: string }

export interface PresetRule { text: string; source: PresetSource }
export interface PresetIntent { name: string; when: string; steps: string[]; examples: string[]; source: PresetSource }
export interface PresetFact { content: string; category: string; importance: number; source: PresetSource }
export interface PresetTemplate { name: string; body: string; source: PresetSource }
export interface PresetTerm { term: string; meaning: string; source: PresetSource }
export interface PresetDoc { doc: string; title: string; summary: string }

export interface AgentPreset {
  rules: PresetRule[]
  intents: PresetIntent[]
  facts: PresetFact[]
  escalation: PresetRule[]
  templates: PresetTemplate[]
  glossary: PresetTerm[]
  knowledge: PresetDoc[]
  conflicts: string[]
  warnings: string[]
}

/** How the person wants a document used ("auto": the model decides). */
export type DocUsage = 'auto' | 'rules' | 'procedures' | 'reference' | 'templates'

export interface PreparedDoc {
  name: string
  chars: number
  truncated: boolean
  origin: 'upload' | 'library'
  description: string
  usage: DocUsage
  /** A library pick: the agent whose folder it came from. */
  agent?: string
}

export interface PreparedPreset {
  id: string
  preset: AgentPreset
  docs: PreparedDoc[]
  model: string
}

/** What Apply takes: the prepared id and the items the person kept. */
export type ReviewedPreset = Omit<AgentPreset, 'conflicts' | 'warnings'> & { id: string }

export interface AppliedPreset {
  ok: boolean
  /** The agent's instructions as now stored for its owner ("" without accounts). */
  instructions: string
  /** The block they end with — what a deck without accounts appends itself. */
  block: string
  counts: { rules: number; escalation: number; templates: number; glossary: number; intents: number; facts: number; knowledge: number }
  skills: string[]
  knowledge: string[]
  running: boolean
}

export interface SetupNote { description: string; usage: DocUsage }

async function send(path: string, init: RequestInit, asUser?: string): Promise<Response> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    if (asUser) h[ACT_AS_HEADER] = asUser
    return h
  }
  const go = () => fetch(path, { ...init, headers: headers(), credentials: 'include' })
  let res = await go()
  if (res.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) {
    res = await go()
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : `${res.status}`)
  }
  return res
}

export async function prepareAgentPreset(opts: {
  /** The agent being set up: its archetype frames the read. */
  agent: string
  instructions: string
  uploads: (SetupNote & { file: File })[]
  /** File library files, by id. */
  picks: (SetupNote & { id: string })[]
  asUser?: string
  signal?: AbortSignal
}): Promise<PreparedPreset> {
  // No Content-Type: the browser sets the multipart boundary itself.
  const form = new FormData()
  form.append('agent', opts.agent)
  form.append('instructions', opts.instructions)
  form.append('notes', JSON.stringify(opts.uploads.map(({ description, usage }) => ({ description, usage }))))
  form.append('library', JSON.stringify(opts.picks))
  for (const u of opts.uploads) form.append('files', u.file)
  const res = await send('/fd/agent-preset/prepare', { method: 'POST', body: form, signal: opts.signal }, opts.asUser)
  return res.json()
}

export async function applyAgentPreset(agent: string, preset: ReviewedPreset, asUser?: string): Promise<AppliedPreset> {
  const res = await send('/fd/agent-preset/apply', {
    method: 'POST', body: JSON.stringify({ agent, preset }),
  }, asUser)
  return res.json()
}
