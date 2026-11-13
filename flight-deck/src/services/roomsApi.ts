import { refreshAccessToken, useAuthStore } from '../stores/authStore'
import { api } from './fileLibraryApi'

// Data rooms (/fd/rooms): a virtual data room — a ZIP or a folder of files —
// anonymized into one vault, checked, and released for Claude to read through
// the Captain connector. Everything here is the owner's alone.

export type RoomState = 'setup' | 'running' | 'review' | 'released'
export type FileState = 'new' | 'done' | 'blocked' | 'failed' | 'unread'

export interface RoomCounts {
  files: number
  released: number
  new: number
  done: number
  blocked: number
  failed: number
  unread: number
  warnings: number
}

export interface RoomJob {
  running: boolean
  phase?: 'queued' | 'anonymizing' | 'naming' | 'sweeping' | 'checking' | 'done'
  total?: number
  done?: number
  failed?: number
  blocked?: number
  current?: string
  sweep_round?: number
  sweep_changed?: number
  message?: string
  started_at?: number
  finished_at?: number
}

export interface RoomSummary {
  id: string
  codename: string
  name: string
  state: RoomState
  counts: RoomCounts
  job: RoomJob
  created_at: number
  updated_at: number
  released_at: number | null
}

export interface Seed {
  orgs?: string[]
  people?: string[]
  products?: string[]
  domains?: string[]
  ids?: string[]
  other?: string[]
}

export interface Room extends RoomSummary {
  vault_id: string
  vault_name: string
  source: { kind?: 'local' | 'vfs'; path?: string; project?: string; owner?: string }
  seed: Seed
  roles: Record<string, string>
  settings: { pictures?: 'strip' | 'keep'; lane?: string }
  trusted_model: string
  ocr: boolean
  max_zip_mb: number
  max_file_mb: number
  warnings?: string[]
}

export interface RoomFile {
  id: string
  origin: 'zip' | 'folder'
  path: string
  clean_path: string
  kind: string
  size: number
  state: FileState
  warnings: string[]
  error: string
  entities: Record<string, number>
  text_chars: number
  released: boolean
  updated_at: number
}

export interface IntakeReport {
  added: number
  replaced: number
  unchanged: number
  unread: { path: string; reason: string }[]
  unread_count: number
  skipped: { path: string; reason: string }[]
  skipped_count: number
  nested_zips: number
}

export interface FileText {
  id: string
  path: string
  clean_path: string
  state: FileState
  clean: string
  original?: string
}

export interface SearchHit { file_id: string; path: string; page: string; snippet: string }

const enc = encodeURIComponent
const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) })

/** A raw upload (the ZIP as the request body), with the same sign-in as `api`. */
async function putFile<T>(path: string, file: File): Promise<T> {
  const send = () => {
    const jwt = useAuthStore.getState().token
    return fetch(path, {
      method: 'PUT',
      body: file,
      cache: 'no-store',
      credentials: 'include',
      headers: { 'Content-Type': 'application/zip', ...(jwt ? { Authorization: `Bearer ${jwt}` } : {}) },
    })
  }
  let r = await send()
  if (r.status === 401 && useAuthStore.getState().authEnabled && (await refreshAccessToken())) r = await send()
  if (!r.ok) {
    let detail = ''
    try {
      const body = await r.json()
      detail = typeof body?.detail === 'string' ? body.detail : ''
    } catch { /* not JSON */ }
    throw new Error(detail || `${r.status} ${r.statusText}`)
  }
  return (await r.json()) as T
}

/** Download a file the deck serves behind sign-in, under its own name. */
async function download(path: string, filename: string): Promise<void> {
  const jwt = useAuthStore.getState().token
  const r = await fetch(path, { credentials: 'include', headers: jwt ? { Authorization: `Bearer ${jwt}` } : {} })
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  const url = URL.createObjectURL(await r.blob())
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

export const roomsApi = {
  list: () => api<{ rooms: RoomSummary[] }>('/fd/rooms'),
  create: (codename: string, name: string) => api<Room>('/fd/rooms', json('POST', { codename, name })),
  get: (id: string) => api<Room>(`/fd/rooms/${enc(id)}`),
  update: (id: string, patch: { codename?: string; name?: string; seed?: Seed; roles?: Record<string, string>;
    settings?: Room['settings'] }) => api<Room>(`/fd/rooms/${enc(id)}`, json('PATCH', patch)),
  remove: (id: string, vault: boolean) =>
    api<{ deleted: boolean; vault_deleted: boolean }>(`/fd/rooms/${enc(id)}?vault=${vault}`, { method: 'DELETE' }),
  uploadZip: (id: string, file: File) => putFile<{ report: IntakeReport; room: Room }>(`/fd/rooms/${enc(id)}/zip`, file),
  linkFolder: (id: string, body: { path: string } | { project: string; path: string; owner?: string }) =>
    api<{ report: IntakeReport; room: Room }>(`/fd/rooms/${enc(id)}/folder`, json('POST', body)),
  run: (id: string, retryFailed = false) => api<Room>(`/fd/rooms/${enc(id)}/run`, json('POST', { retry_failed: retryFailed })),
  stop: (id: string) => api<{ stopping: boolean }>(`/fd/rooms/${enc(id)}/stop`, { method: 'POST' }),
  files: (id: string) => api<{ files: RoomFile[] }>(`/fd/rooms/${enc(id)}/files`),
  text: (id: string, fileId: string, original: boolean) =>
    api<FileText>(`/fd/rooms/${enc(id)}/files/${enc(fileId)}/text?original=${original}`),
  downloadClean: (id: string, f: RoomFile) =>
    download(`/fd/rooms/${enc(id)}/files/${enc(f.id)}/clean`, f.clean_path.split('/').pop() || 'file'),
  removeFile: (id: string, fileId: string) =>
    api<{ deleted: boolean }>(`/fd/rooms/${enc(id)}/files/${enc(fileId)}`, { method: 'DELETE' }),
  release: (id: string) => api<{ released_now: number; room: Room }>(`/fd/rooms/${enc(id)}/release`, { method: 'POST' }),
  search: (id: string, q: string) => api<{ results: SearchHit[] }>(`/fd/rooms/${enc(id)}/search?q=${enc(q)}`),
}
