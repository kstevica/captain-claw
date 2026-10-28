// An agent's "way of working" setup, opened right after the kiosk creates it
// (AgentSetupDialog). POST /fd/agent-preset/documents reads each document in
// full as it is added — uploaded or picked from the File library — and says
// its pages and what reading it costs; POST /fd/agent-preset/prepare reads the
// material with the person's own model in the background (in steps sized to
// its context, then merged) into a preset they review; POST
// /fd/agent-preset/apply gives the agent what they kept.
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

/** A document read in full when it was added, waiting for the setup. */
export interface StagedDoc {
  id: string
  name: string
  origin: 'upload' | 'library'
  /** Its pages (or slides): counted for a PDF or a deck, else estimated. */
  pages: number
  pages_exact: boolean
  chars: number
  /** What reading it in full takes, roughly. */
  tokens: number
  truncated: boolean
  /** A library pick: the agent whose folder it came from. */
  agent?: string
  /** A spreadsheet: the datastore tables it also gives the agent (one per sheet with data). */
  tables?: number
}

export interface PreparedDoc {
  name: string
  chars: number
  pages: number
  pages_exact: boolean
  truncated: boolean
  origin: 'upload' | 'library'
  description: string
  usage: DocUsage
  /** Read for rules, skills and facts; otherwise kept as knowledge only. */
  read: boolean
  agent?: string
  /** A spreadsheet: the datastore tables it also gives the agent. */
  tables?: number
}

export interface PreparedPreset {
  id: string
  preset: AgentPreset
  docs: PreparedDoc[]
  model: string
  /** The steps it was read in (0: nothing needed reading). */
  steps: number
  tokens: number
}

/** A read running in the background (GET /prepare/{id}). */
export interface ReadJob {
  id: string
  state: 'running' | 'done' | 'error' | 'cancelled'
  phase: 'reading' | 'merging'
  step: number
  steps: number
  /** What the step under way reads ("manual.pdf, pages 81–160"). */
  current: string
  tokens: number
  started: number
  finished: number | null
  error: string
  /** Who reads: the agent's own model ('agent'), or the person's ('tiers'). */
  model: string
  via: ReaderVia
  /** While it waits: the agent being woken, or a busy model's server waited out. */
  waiting: ReadWait | null
  result?: PreparedPreset
}

export type ReaderVia = 'agent' | 'tiers' | ''

export type ReadWait =
  | { why: 'waking'; name: string }
  | { why: 'busy'; model: string; /** Unix seconds. */ until: number }
  /** Calls spaced out so the provider's caps stay far off. */
  | { why: 'pacing'; /** Unix seconds. */ until: number }

/** The model that reads — the agent's own, else the person's — how much
 *  material one step gives it, the tokens a minute the read keeps to
 *  (`pace_tpm`, 0: unpaced), and why it isn't the agent's (`note`). */
export interface Reader { model: string; step_tokens: number; pace_tpm?: number; via?: ReaderVia; note?: string }

/** What Apply takes: the prepared id and the items the person kept. */
export type ReviewedPreset = Omit<AgentPreset, 'conflicts' | 'warnings'> & { id: string }

/** Added to what an earlier setup gave the agent, or in its place. */
export type ApplyMode = 'add' | 'replace'

export interface SetupCounts {
  rules: number; escalation: number; templates: number; glossary: number; intents: number; facts: number; knowledge: number
  /** Datastore tables from the spreadsheets. */
  tables?: number
}

export interface AppliedPreset {
  ok: boolean
  mode: ApplyMode
  /** Flight Deck stored the instructions for the owner (a deck with accounts). */
  stored: boolean
  /** The agent's instructions as now stored for its owner. */
  instructions: string
  /** The marked block they end with — what a deck without accounts puts in itself. */
  block: string
  /** What this setup gave it… */
  counts: SetupCounts
  /** …and what it has from every setup so far. */
  totals: SetupCounts
  skills: string[]
  knowledge: string[]
  /** Files the long documents were split into, so the agent can search all of them. */
  parts: number
  /** The datastore tables the spreadsheets became (made now, or when the agent is next up). */
  tables?: string[]
  /** What was left out of them (sheets past the datastore's room, rows past its cap). */
  warnings?: string[]
  running: boolean
}

export interface DocChoice { id: string; description: string; usage: DocUsage; read: boolean }

/** What earlier setups gave an agent (GET /fd/agent-preset/current/{slug}). */
export interface CurrentSetup {
  has: boolean
  updated: number | null
  rules: number
  escalation: number
  templates: number
  glossary: number
  skills: string[]
  facts: number
  knowledge: { title: string; parts: number }[]
  tables?: { name: string; doc: string; sheet: string | null; rows: number }[]
}

async function send(path: string, init: RequestInit, asUser?: string): Promise<Response> {
  const headers = (): Record<string, string> => {
    const { token, authEnabled } = useAuthStore.getState()
    const h: Record<string, string> = typeof init.body === 'string' ? { 'Content-Type': 'application/json' } : {}
    if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
    if (asUser) h[ACT_AS_HEADER] = asUser
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
  return res
}

/** Read a document in full and keep it for the setup: an upload, or a File
 *  library file by id. Spends no tokens. */
export async function stageDocument(
  doc: { file: File } | { libraryId: string }, asUser?: string,
): Promise<{ docs: StagedDoc[]; warnings: string[] }> {
  // No Content-Type: the browser sets the multipart boundary itself.
  const form = new FormData()
  if ('file' in doc) form.append('files', doc.file)
  else form.append('library', JSON.stringify([doc.libraryId]))
  const res = await send('/fd/agent-preset/documents', { method: 'POST', body: form }, asUser)
  return res.json()
}

export async function getReader(agent: string, asUser?: string): Promise<Reader> {
  return (await send(`/fd/agent-preset/reader?agent=${encodeURIComponent(agent)}`, { method: 'GET' }, asUser)).json()
}

/** Start reading the material (in the background); follow it with getRead. */
export async function startRead(opts: {
  /** The agent being set up: its own model reads, and its archetype frames the read. */
  agent: string
  instructions: string
  docs: DocChoice[]
  asUser?: string
}): Promise<ReadJob> {
  const res = await send('/fd/agent-preset/prepare', {
    method: 'POST', body: JSON.stringify({ agent: opts.agent, instructions: opts.instructions, docs: opts.docs }),
  }, opts.asUser)
  return res.json()
}

export async function getRead(id: string, asUser?: string): Promise<ReadJob> {
  return (await send(`/fd/agent-preset/prepare/${encodeURIComponent(id)}`, { method: 'GET' }, asUser)).json()
}

export async function cancelRead(id: string, asUser?: string): Promise<void> {
  await send(`/fd/agent-preset/prepare/${encodeURIComponent(id)}`, { method: 'DELETE' }, asUser).catch(() => undefined)
}

export async function getCurrentSetup(agent: string, asUser?: string): Promise<CurrentSetup> {
  return (await send(`/fd/agent-preset/current/${encodeURIComponent(agent)}`, { method: 'GET' }, asUser)).json()
}

export async function applyAgentPreset(
  agent: string, preset: ReviewedPreset, mode: ApplyMode = 'add', asUser?: string,
): Promise<AppliedPreset> {
  const res = await send('/fd/agent-preset/apply', {
    method: 'POST', body: JSON.stringify({ agent, preset, mode }),
  }, asUser)
  return res.json()
}
