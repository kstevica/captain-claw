import { api } from './fileLibraryApi'

// Anonymization (/fd/anonymize): private data in a text or a file replaced by
// tokens (⟦PERSON_1⟧); the real values kept in the person's encrypted vaults.
// Files travel as base64 in JSON — the server reads bodies by hand so a bad
// request can never echo private data back.

export type Lane = 'deterministic' | 'llm' | 'both'
export type Mode = 'reversible' | 'one_way'

export interface AnonEntity { token: string; kind: string; count: number; new: boolean }

export interface Lanes {
  deterministic: 'ok' | 'off'
  llm: 'ok' | 'off' | 'failed'
  llm_model?: string
  llm_terms?: number
  llm_dropped?: number
  llm_seconds?: number
  llm_error?: string
}

export interface RunResult {
  vault: { id: string; name: string } | null
  mode: Mode
  format: string
  filename: string
  entities: AnonEntity[]
  summary: Record<string, number>
  verify: { ok: boolean; rounds: number; fixed: number }
  lane: Lane
  lanes: Lanes
  generalized?: string[]
  name_model?: string
  warnings: string[]
  output?: string          // a text run
  data?: string            // a file run: the file, base64
  content_type?: string
  size?: number
  preview?: string
}

export interface RestoreResult {
  vault: { id: string; name: string }
  restored: number
  unknown: string[]
  output?: string
  filename?: string
  data?: string
  content_type?: string
}

export interface Vault {
  id: string
  name: string
  entity_count: number
  kinds: Record<string, number>
  created_by: string
  created_at: number
  updated_at: number
  last_used_at: number
  expires_at: number | null
}

export interface VaultEntry {
  token: string
  kind: string
  original: string
  forms: string[]
  created_at: number | null
  updated_at: number | null
  /** Texts it went out in, and times it came back and was put back. */
  sent?: number
  restored?: number
  last_sent_at?: number | null
  last_restored_at?: number | null
}

export interface VaultEvent { at: number; actor: string; action: string; detail: Record<string, unknown> }

/** One line of a token's history: what added it, or a time it went out to a
 * model ("out") or came back and was put back ("back"). The context is cut
 * from the tokenized text — tokens, no real values. */
export interface TokenUse {
  id: number
  at: number
  direction: 'added' | 'out' | 'back'
  channel: string
  actor: string
  label: string
  found_by: string
  related: string
  model: string
  count: number
  context: string
}

export interface TokenHistory {
  token: string
  added: TokenUse | null
  stats: { sent: number; restored: number; last_sent_at: number | null; last_restored_at: number | null }
  uses: TokenUse[]
  /** An older page is there (load it with `before` = the last line's id). */
  more: boolean
}

export interface AnonStatus { llm: boolean; model: string; default_lane: Lane; name_model?: string }

export interface AnonKinds {
  kinds: { kind: string; label: string }[]
  formats: string[]
  file_kinds: string[]
  max_file_mb: number
}

export interface RunOptions {
  vault?: string
  mode?: Mode
  lane?: Lane
  keep?: string[]
  extra?: { text: string; kind: string }[]
  generalize?: boolean
}

export interface NamePack {
  enabled: boolean
  model: string
  status: 'off' | 'not_installed' | 'not_downloaded' | 'not_loaded' | 'downloading' | 'loading' | 'ready' | 'error'
  error: string
  runtime: boolean
  downloaded: boolean
  models: { id: string; label: string }[]
}

export interface TrustedModel {
  provider: string
  model: string
  base_url: string
  has_key: boolean
  attested: boolean
  on_failure: 'refuse' | 'deterministic'
  enabled: boolean
  source: 'settings' | 'env'
  basis: 'local' | 'attested' | ''
  label: string
  usable: boolean
  updated_by: string
  updated_at: number
}

export interface TrustedModelInput {
  provider: string
  model: string
  base_url: string
  api_key?: string
  attested: boolean
  on_failure: 'refuse' | 'deterministic'
  enabled: boolean
}

export interface GateSettings {
  enabled: boolean
  groups: string[]
  folders: string[]
  when: 'untrusted' | 'always'
  lane: 'deterministic' | 'both'
  /** 'chat': everything that reaches the model, not only the sources picked. */
  coverage: 'sources' | 'chat'
}

export interface GateInfo {
  gate: GateSettings
  vault: string
  groups: { id: string; label: string }[]
  model_available: boolean
}

/** Where an anonymized copy was written (a VFS place). */
export interface VfsPlace { project: string; path: string; owner: string }

export interface HereResult extends RunResult {
  written: VfsPlace
  beside: boolean
  size: number
}

export interface LibraryResult extends RunResult {
  written: { agent_slug: string; rel_path: string; filename: string }
  size: number
}

export interface FolderJob {
  id: string
  running: boolean
  phase: 'anonymizing' | 'placing' | 'done'
  cancelled: boolean
  source: VfsPlace
  dest: Partial<VfsPlace>  // empty until the copies are placed
  vault: string
  beside: boolean
  total: number
  done: number
  current: string
  skipped: { path: string; reason: string }[]
  failed: { path: string; error: string }[]
  skipped_count: number
  failed_count: number
  entities: Record<string, number>
  message: string
}

export interface TrustedTestCase {
  id: string
  lanes: string[]
  seconds: number
  code_missed: string[]
  both_missed: string[] | null
  both_over_hid: string[] | null
  error: string
}

const post = <T>(path: string, body: unknown) => api<T>(path, { method: 'POST', body: JSON.stringify(body) })
const enc = encodeURIComponent

export const anonymizeApi = {
  status: () => api<AnonStatus>('/fd/anonymize/status'),
  kinds: () => api<AnonKinds>('/fd/anonymize/kinds'),
  run: (content: string, opts: RunOptions) => post<RunResult>('/fd/anonymize/run', { content, format: 'text', ...opts }),
  file: (filename: string, data: string, opts: RunOptions) => post<RunResult>('/fd/anonymize/file', { filename, data, ...opts }),
  restore: (content: string, vault: string) => post<RestoreResult>('/fd/anonymize/restore', { content, format: 'text', vault }),
  restoreFile: (filename: string, data: string, vault: string) =>
    post<RestoreResult>('/fd/anonymize/file/restore', { filename, data, vault }),
  vaults: () => api<{ vaults: Vault[] }>('/fd/anonymize/vaults'),
  vault: (id: string) => api<{ vault: Vault; entries: VaultEntry[] }>(`/fd/anonymize/vaults/${enc(id)}`),
  events: (id: string) => api<{ events: VaultEvent[] }>(`/fd/anonymize/vaults/${enc(id)}/events`),
  tokenHistory: (id: string, token: string, before?: number) =>
    api<TokenHistory>(`/fd/anonymize/vaults/${enc(id)}/tokens/${enc(token)}${before ? `?before=${before}` : ''}`),
  updateVault: (id: string, patch: { name?: string; expires_at?: number | null }) =>
    api<{ vault: Vault }>(`/fd/anonymize/vaults/${enc(id)}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  deleteVault: (id: string) => api<{ deleted: string }>(`/fd/anonymize/vaults/${enc(id)}`, { method: 'DELETE' }),
  gate: () => api<GateInfo>('/fd/anonymize/gate'),
  saveGate: (gate: GateSettings) =>
    api<{ gate: GateSettings }>('/fd/anonymize/gate', { method: 'PUT', body: JSON.stringify(gate) }),
  reveal: (content: string) => post<RestoreResult>('/fd/anonymize/reveal', { content }),
  nerGet: () => api<NamePack>('/fd/anonymize/admin/ner'),
  nerPut: (enabled: boolean, model: string) =>
    api<NamePack>('/fd/anonymize/admin/ner', { method: 'PUT', body: JSON.stringify({ enabled, model }) }),
  nerDownload: () => post<NamePack>('/fd/anonymize/admin/ner/download', {}),
  vfsFile: (project: string, path: string, owner: string, opts: RunOptions) =>
    post<HereResult>('/fd/anonymize/vfs/file', { project, path, owner, ...opts }),
  vfsFolder: (project: string, path: string, owner: string, opts: RunOptions) =>
    post<FolderJob>('/fd/anonymize/vfs/folder', { project, path, owner, ...opts }),
  vfsJob: () => api<FolderJob | { running: false }>('/fd/anonymize/vfs/job'),
  vfsJobCancel: () => post<FolderJob>('/fd/anonymize/vfs/job/cancel', {}),
  libraryFile: (id: string, opts: RunOptions) => post<LibraryResult>(`/fd/anonymize/library/${enc(id)}`, opts),
  trustedGet: () => api<{ model: TrustedModel | null; providers: string[] }>('/fd/anonymize/admin/trusted-model'),
  trustedPut: (body: TrustedModelInput) =>
    api<{ model: TrustedModel }>('/fd/anonymize/admin/trusted-model', { method: 'PUT', body: JSON.stringify(body) }),
  trustedDelete: () => api<{ model: TrustedModel | null }>('/fd/anonymize/admin/trusted-model', { method: 'DELETE' }),
  trustedTest: () => post<{ model: string; cases: TrustedTestCase[] }>('/fd/anonymize/admin/trusted-model/test', {}),
}

/** A file's bytes as base64 (no data: prefix). */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const url = String(reader.result || '')
      resolve(url.slice(url.indexOf(',') + 1))
    }
    reader.onerror = () => reject(reader.error || new Error("Couldn't read the file"))
    reader.readAsDataURL(file)
  })
}

/** Save base64 bytes (or text) as a file in the browser. */
export function saveFile(name: string, content: { base64: string; type?: string } | { text: string }) {
  let blob: Blob
  if ('base64' in content) {
    const bin = atob(content.base64)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    blob = new Blob([bytes], { type: content.type || 'application/octet-stream' })
  } else {
    blob = new Blob([content.text], { type: 'text/plain;charset=utf-8' })
  }
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export const TOKEN_RE = /⟦([A-Z][A-Z0-9_]*?)_(\d{1,6})(?::[a-z]{1,3})?⟧/g
