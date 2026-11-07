import { api } from './fileLibraryApi'

// Custom tools (external_tools on the server, /fd/external-tools): small Python
// programs Crew writes and Flight Deck runs for agents. Private, offered to a
// person, or offered to everybody; nobody's agent runs someone else's tool until
// that person adds it.

export interface SandboxInfo {
  os: string
  landlock: number
  files: boolean
  network: boolean
  signals: boolean
}

export interface ToolStatus {
  enabled: boolean
  everybody_needs_approval: boolean
  require_sandbox: boolean
  can_build: boolean
  sandbox?: SandboxInfo
}

export interface PublishStep { step: string; ok: boolean; detail: string; at: number }

export interface PublishState {
  state: 'running' | 'passed' | 'failed'
  version?: string
  sha?: string
  error?: string
  started_at?: number
  finished_at?: number
  steps?: PublishStep[]
}

export interface Person { id: string; email: string; name: string }

export interface Declares {
  version: string
  network: boolean
  secrets: string[]
  timeout_s: number
  memory_mb: number
}

export interface ToolSummary {
  owner: string
  slug: string
  title: string
  description: string
  live: { version: string; published_at?: number } | null
  publish: Pick<PublishState, 'state' | 'version' | 'error' | 'started_at' | 'finished_at'> | null
}

export interface JsonSchema {
  type?: string
  format?: string
  description?: string
  properties?: Record<string, JsonSchema>
  required?: string[]
  items?: JsonSchema
  enum?: unknown[]
}

export interface Manifest {
  name: string
  title: string
  description: string
  input: JsonSchema
  output: JsonSchema
  examples: { input: Record<string, unknown>; expect?: unknown }[]
  secrets: { name: string; why: string }[]
  network: boolean
  timeout_s: number
  memory_mb: number
}

export interface SecretState { name: string; why: string; set: boolean }

export interface ToolDetail extends ToolSummary {
  // The owner's own tool: the whole last publish, steps and all.
  publish: PublishState | null
  manifest: Manifest | null
  secrets: SecretState[]
  // Only the owner's own tools carry these.
  versions?: { version: string; sha: string; built_at?: number; live: boolean }[]
  can_build?: boolean
  // Only someone else's tool carries these.
  owner_person?: Person
  offered?: 'person' | 'everybody' | ''
  declares?: Declares | null
  added?: boolean
}

export interface Offer extends ToolSummary {
  owner_person: Person
  offered: 'person' | 'everybody'
  declares: Declares | null
  added: boolean
}

export interface RunResult {
  ok: boolean
  call_id: string
  version: string
  seconds: number
  exit_code: number | null
  output?: unknown
  files?: { name: string; path: string; size: number }[]
  files_skipped?: number
  stderr?: string
  error?: { kind: string; message: string }
}

export interface Sharing {
  share_id: string
  people: { grantee_id: string; grantee_email: string; grantee_name: string }[]
  everybody: { status: 'pending' | 'approved' | 'rejected'; live: boolean; requested_at: string } | null
  needs_approval: boolean
  added_by: number
}

export interface Usage {
  days: { day: string; calls: number; failed: number; people: number }[]
  failures: Record<string, number>
  versions: Record<string, number>
  people: number
  calls: number
  failed: number
}

export interface Approval {
  owner: string
  slug: string
  owner_person: Person
  title: string
  description: string
  live: { version: string } | null
  manifest: Manifest | null
  requested_at: string
}

export interface Proposal {
  id: string
  agent: string
  agent_name: string
  name: string
  title: string
  what: string
  why: string
  said?: string
  source: 'noticed' | 'asked'
  created: number
  state: 'waiting' | 'built' | 'dismissed'
  settled?: number
  task_id?: string
  tool?: string
}

export interface TaskStarted {
  tool: ToolSummary
  task: { id: string; title: string; status: string; harness_label: string; branch: string }
}

const base = '/fd/external-tools'
const q = (owner?: string) => (owner ? `?owner=${encodeURIComponent(owner)}` : '')
const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) })

/** The deck's check of a new tool's spec (New tool, step two): a fuller spec
 *  and why, or ``error`` when the check gave none (the person can still build
 *  what they wrote). */
export type SpecReview =
  | { spec: string; title: string; suggestions: { change: string; why: string }[]; overlap: string; unchanged: boolean; model: string; error?: undefined }
  | { error: string; spec?: undefined }

export const customToolsApi = {
  status: () => api<ToolStatus>(`${base}/status`),
  mine: () => api<{ tools: ToolSummary[]; can_build: boolean }>(base),
  shared: () => api<{ tools: Offer[] }>(`${base}/shared`),
  catalog: () => api<{ tools: Offer[] }>(`${base}/catalog`),
  get: (slug: string, owner?: string) => api<ToolDetail>(`${base}/${slug}${q(owner)}`),
  run: (slug: string, input: Record<string, unknown>, owner?: string) =>
    api<RunResult>(`${base}/${slug}/run${q(owner)}`, json('POST', { input })),
  setSecrets: (slug: string, values: Record<string, string>, owner?: string) =>
    api<{ secrets: SecretState[] }>(`${base}/${slug}/secrets${q(owner)}`, json('PUT', { values })),
  publish: (slug: string) => api<PublishState>(`${base}/${slug}/publish`, json('POST', {})),
  publishState: (slug: string) => api<{ publish: PublishState | null }>(`${base}/${slug}/publish`),
  makeLive: (slug: string, version: string) => api<unknown>(`${base}/${slug}/live`, json('POST', { version })),
  remove: (slug: string) => api<{ ok: boolean }>(`${base}/${slug}`, { method: 'DELETE' }),
  review: (body: { slug: string; title: string; what: string }) =>
    api<SpecReview>(`${base}/review`, json('POST', body)),
  build: (body: { slug: string; title: string; what: string; harness: string; model?: string }) =>
    api<TaskStarted>(`${base}/build`, json('POST', body)),
  improve: (slug: string, body: { what: string; harness: string; model?: string }) =>
    api<TaskStarted>(`${base}/${slug}/improve`, json('POST', body)),
  sharing: (slug: string) => api<Sharing>(`${base}/${slug}/sharing`),
  everybody: (slug: string, on: boolean) => api<{ everybody: Sharing['everybody'] }>(`${base}/${slug}/everybody`, json('PUT', { on })),
  usage: (slug: string) => api<Usage>(`${base}/${slug}/usage`),
  add: (owner: string, slug: string) => api<{ added: boolean }>(`${base}/added`, json('POST', { owner, slug })),
  unadd: (owner: string, slug: string) =>
    api<{ added: boolean }>(`${base}/added?owner=${encodeURIComponent(owner)}&slug=${encodeURIComponent(slug)}`, { method: 'DELETE' }),
  approvals: () => api<{ approvals: Approval[]; needs_approval: boolean }>(`${base}/approvals`),
  proposals: () => api<{ waiting: Proposal[]; recent: Proposal[] }>(`${base}/proposals`),
  buildProposal: (id: string, body: { harness: string; name?: string; title?: string; what?: string }) =>
    api<TaskStarted & { proposal: Proposal }>(`${base}/proposals/${id}/build`, json('POST', body)),
  dismissProposal: (id: string) => api<{ proposal: Proposal }>(`${base}/proposals/${id}/dismiss`, json('POST', {})),
  decide: (owner: string, slug: string, approve: boolean) =>
    api<unknown>(`${base}/approvals/${encodeURIComponent(owner)}/${slug}`, json('POST', { approve })),
}
