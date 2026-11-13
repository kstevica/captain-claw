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
  /** Sizes of the files to anonymize, and of those finished (for a run's ETA). */
  bytes_total?: number
  bytes_done?: number
  /** The file being worked on: its size, its place in this phase, and how far it is. */
  current_size?: number
  file_index?: number
  file_count?: number
  step?: 'reading' | 'model' | 'replacing'
  /** The trusted model's chunks of the current file: done of total. */
  chunk?: number
  chunks?: number
  file_started_at?: number
  chunks_started_at?: number | null
  chunk_at?: number | null
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
  inbox: { total: number; unseen: number }
  /** Tasks left for Claude; open = waiting or picked up, not answered yet. */
  tasks?: { total: number; open: number }
}

/** A task the person left for Claude in a room (asked from the Inbox). */
export interface RoomTask {
  seq: number
  created_at: number
  /** The delivery it is about, if any. */
  about: number | null
  state: 'waiting' | 'taken' | 'done' | 'cancelled'
  /** The person's own words (only they see these). */
  prompt: string
  /** What Claude reads: the same, anonymized with the room's vault. */
  prompt_tokens: string
  taken_at: number | null
  done_at: number | null
  /** The delivery that answered it. */
  delivery: number | null
}

/** P7: the names put into the case their sentence needs, by the trusted model (Croatian). */
export interface GrammarReport {
  state: 'waiting' | 'done' | 'failed' | 'unavailable'
  language?: string
  /** People's names looked at, how many now in another case, forms the check turned down, names over the cap. */
  names?: number
  changed?: number
  refused?: number
  left?: number
  model?: string
  error?: string
  /** The owner asked for it (not a delivery taken for Croatian). */
  asked?: boolean
}

/** What the deck found putting the names back into a delivery — tokens only. */
export interface DeliveryReport {
  restored: number
  not_in_room: string[]
  /** Tokens whose real value Claude wrote itself (it was told, or guessed). Never told to Claude. */
  real_values: string[]
  parts: number
  characters: number
  format: 'markdown' | 'text'
  /** A file built on the deck (P4): what it is, and the build's notes (tokens only). */
  kind?: 'workbook' | 'deck' | 'document' | 'file'
  file?: string
  notes?: string[]
  /** P5: a file Claude made itself — sent through the connector, or dropped in by the person. */
  source?: 'claude' | 'owner'
  /** Nothing could be put back: Claude wrote no tokens into it (a codename, say). */
  no_tokens?: boolean
  grammar?: GrammarReport
  /** The task this delivery answers. */
  task?: number
}

export interface DeliverySummary {
  seq: number
  at: number
  title: string
  chars: number
  report: DeliveryReport
  seen: boolean
  app: string
}

export interface Delivery {
  seq: number
  at: number
  app: string
  report: DeliveryReport
  title: string
  title_tokens: string
  body: string
  body_tokens: string
  format: 'markdown' | 'text'
  /** P7: with the names in their case — null until the trusted model changed any. */
  title_fixed: string | null
  body_fixed: string | null
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
  /** For Excel: rows with something in them, and sheets; for CSV/TSV: rows. */
  info?: { rows?: number; sheets?: number }
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

/** A connected app (a Claude connector) and the rooms it may read. */
export interface RoomGrant {
  client_id: string
  app: string
  all_rooms: boolean
  rooms: { id: string; codename: string }[]
  created_at: number
  updated_at: number
}

export interface RoomRead {
  at: number
  tool: string
  path: string
  page: number
  chars: number
  alarm: boolean
  app: string
}

/** A built “Download for Claude” ZIP: one download, within an hour. */
export interface RoomExport {
  token: string
  files: number
  /** Files left out because they still hold a real value (paths only). */
  held_back: string[]
  size: number
  text_only: boolean
  filename: string
}

/** A real value from the vault found in what Claude sent (it got there some other way). */
export interface RoomAlarm { id: number; at: number; tool: string; tokens: string[]; seen: boolean; app: string }

const enc = encodeURIComponent
const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) })

/** A raw upload (the ZIP as the request body), with the same sign-in as `api`. */
async function putFile<T>(path: string, file: File, contentType = 'application/zip'): Promise<T> {
  const send = () => {
    const jwt = useAuthStore.getState().token
    return fetch(path, {
      method: 'PUT',
      body: file,
      cache: 'no-store',
      credentials: 'include',
      headers: { 'Content-Type': contentType, ...(jwt ? { Authorization: `Bearer ${jwt}` } : {}) },
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
  reads: (id: string) => api<{ reads: RoomRead[]; alarms: RoomAlarm[] }>(`/fd/rooms/${enc(id)}/reads`),
  alarmsSeen: (id: string) => api<{ ok: boolean }>(`/fd/rooms/${enc(id)}/alarms/seen`, { method: 'POST' }),
  access: () => api<{ grants: RoomGrant[]; connector_url: string }>('/fd/rooms-access'),
  setAccess: (clientId: string, rooms: string[], allRooms: boolean) =>
    api<{ ok: boolean }>(`/fd/rooms-access/${enc(clientId)}`, json('PUT', { rooms, all_rooms: allRooms })),
  revokeAccess: (clientId: string) => api<{ ok: boolean }>(`/fd/rooms-access/${enc(clientId)}`, { method: 'DELETE' }),
  deliveries: (id: string) => api<{ deliveries: DeliverySummary[] }>(`/fd/rooms/${enc(id)}/deliveries`),
  delivery: (id: string, seq: number) => api<Delivery>(`/fd/rooms/${enc(id)}/deliveries/${seq}`),
  /** `plain`: the names as stored, not put into their case (P7). */
  downloadDelivery: (id: string, seq: number, as: 'md' | 'docx' | 'file', title: string, ext?: string, plain = false) =>
    download(`/fd/rooms/${enc(id)}/deliveries/${seq}/download?as=${as}${plain ? '&plain=1' : ''}`,
      `${title.replace(/[\\/:*?"<>|]+/g, '_').slice(0, 120) || 'delivery'}.${as === 'file' ? ext || 'bin' : as}`),
  /** Fix the case endings of a delivery's names now (P7). */
  fixGrammar: (id: string, seq: number) =>
    api<{ grammar: GrammarReport }>(`/fd/rooms/${enc(id)}/deliveries/${seq}/grammar`, { method: 'POST' }),
  deleteDelivery: (id: string, seq: number) =>
    api<{ deleted: boolean }>(`/fd/rooms/${enc(id)}/deliveries/${seq}`, { method: 'DELETE' }),
  /** “Download for Claude”: build a ZIP of the released room (P6). */
  exportRoom: (id: string, textOnly: boolean) =>
    api<RoomExport>(`/fd/rooms/${enc(id)}/export`, json('POST', { text_only: textOnly })),
  downloadExport: (id: string, e: RoomExport) => download(`/fd/rooms/${enc(id)}/export/${enc(e.token)}`, e.filename),
  tasks: (id: string) => api<{ tasks: RoomTask[] }>(`/fd/rooms/${enc(id)}/tasks`),
  /** What Claude would read of a task: the words anonymized with the room's vault. */
  previewTask: (id: string, prompt: string) =>
    api<{ prompt_tokens: string; still_real: string[]; found: Record<string, number> }>(
      `/fd/rooms/${enc(id)}/tasks/preview`, json('POST', { prompt })),
  createTask: (id: string, prompt: string, about?: number | null) =>
    api<RoomTask>(`/fd/rooms/${enc(id)}/tasks`, json('POST', { prompt, about: about ?? null })),
  cancelTask: (id: string, seq: number) =>
    api<{ cancelled: boolean }>(`/fd/rooms/${enc(id)}/tasks/${seq}`, { method: 'DELETE' }),
  /** A file Claude made (in tokens), downloaded from Claude and dropped here: the names go back inside. */
  dropFile: (id: string, file: File) =>
    putFile<{ seq: number; report: DeliveryReport }>(`/fd/rooms/${enc(id)}/deliveries/file?name=${enc(file.name)}`, file,
      'application/octet-stream'),
}
