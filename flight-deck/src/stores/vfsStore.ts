import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// ── Types (mirror captain_claw/flight_deck/vfs_routes.py) ────────────

export interface VFSProject {
  name: string
  files: number
  bytes: number
  mtime: number
  kind?: string   // 'basna' | 'vatra' | 'council' | 'link' | '' — parsed from the folder name
  run_id?: string // the run's session-id prefix
  title?: string  // the human title of the Basna/Vatra run that created it
  link_path?: string  // external source path (linked folders only)
  mode?: string       // 'rw' | 'ro' (linked folders only)
  missing?: boolean   // linked source path no longer exists
  shared?: boolean    // shared TO this user by another owner
  owner_id?: string   // owner's user id (shared folders only)
  owner_email?: string
  owner_name?: string
  permission?: string // 'view' | 'edit' (shared folders only)
  drive?: DriveMeta | null // present when kind === 'gdrive'
}

export interface DriveMeta {
  folder_id: string
  clonemd: boolean
  synced_at?: number
  // Human breadcrumb of the mounted Drive folder ("FRC3/Reporting/…/VC"),
  // shown as a subtitle so a short mount name isn't ambiguous.
  source_path?: string
  // Materialisation counts, enriched by /fd/vfs/projects from the manifest.
  total?: number
  cloned?: number
  uncloned?: number
}

export interface DriveFolder { id: string; name: string }

// A live event streamed while a Drive folder mounts (see mountDrive).
export interface DriveMountEvent {
  event: 'progress' | 'done' | 'error'
  phase?: 'reading' | 'cloning'
  folders?: number
  files?: number
  done?: number
  name?: string
  detail?: string
}

export interface VFSEntry {
  name: string
  type: 'dir' | 'file'
  path: string // relative to project root
  project: string
  size: number
  mtime: number
  author?: string     // the agent that last wrote this file (VFS files only)
  author_ts?: number
}

export interface VFSFile {
  project: string
  path: string
  name: string
  size: number
  binary: boolean
  truncated: boolean
  text: string
}

// ── API helpers ──────────────────────────────────────────────────────

function _headers(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function _authedFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const build = (): RequestInit => ({
    ...init,
    headers: { ..._headers(), ...((init.headers as Record<string, string>) || {}) },
    credentials: 'include',
  })
  let res = await fetch(url, build())
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (await refreshAccessToken()) res = await fetch(url, build())
  }
  return res
}

const qp = (o: Record<string, string>) =>
  Object.entries(o)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&')

const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico', '.bmp']
export const extOf = (name: string): string => {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

// ── Store ─────────────────────────────────────────────────────────────

interface VFSStore {
  projects: VFSProject[]
  project: string | null
  owner: string        // owner id when the active project is shared ('' = own)
  permission: string   // '' | 'view' | 'edit' for the active shared project
  path: string
  entries: VFSEntry[]
  loading: boolean
  error: string | null
  // open file viewer/editor (modal)
  file: VFSFile | null
  fileLoading: boolean
  fileError: string | null
  blobUrl: string | null // object URL for images / binary preview
  draft: string
  editing: boolean
  saving: boolean

  loadProjects: () => Promise<void>
  openProject: (name: string, owner?: string, permission?: string) => Promise<void>
  closeProject: () => void
  browse: (path: string) => Promise<void>
  refresh: () => Promise<void>
  openFile: (entry: VFSEntry) => Promise<void>
  closeFile: () => void
  setDraft: (text: string) => void
  startEdit: () => void
  saveFile: () => Promise<void>
  download: (entry: VFSEntry) => Promise<void>
  exportToDrive: (entry: VFSEntry, folderId: string) => Promise<{ ok: boolean; link?: string; error?: string }>
  downloadProject: (name: string) => Promise<void>
  // Copy a folder shared TO me into my own workspace (owner-scoped).
  copyShared: (project: string, owner: string) => Promise<{ ok: boolean; project?: string; error?: string }>
  deleteEntry: (entry: VFSEntry) => Promise<void>
  newFolder: (name: string) => Promise<void>
  newProject: (name: string) => Promise<void>
  uploadFiles: (files: File[] | FileList) => Promise<void>
  deleteProject: (name: string) => Promise<void>
  addLink: (name: string, path: string, mode: string) => Promise<void>
  browseFs: (path: string) => Promise<FsListing>
  // Google Drive mounts
  browseDrive: (folderId: string, driveId?: string) => Promise<{ folders: DriveFolder[]; shared_drives: DriveFolder[]; truncated: boolean }>
  mountDrive: (name: string, folderId: string, clonemd: boolean, driveId?: string, path?: string, onProgress?: (ev: DriveMountEvent) => void) => Promise<void>
  refreshDrive: (name: string) => Promise<string>
  toggleClonemd: (name: string, clonemd: boolean) => Promise<string>
  unmountDrive: (name: string, keepCloned: boolean) => Promise<void>
}

export interface FsDir { name: string; hidden: boolean; is_git: boolean }
export interface FsListing { path: string; parent: string; dirs: FsDir[] }

export const useVFSStore = create<VFSStore>((set, get) => ({
  projects: [],
  project: null,
  owner: '',
  permission: '',
  path: '',
  entries: [],
  loading: false,
  error: null,
  file: null,
  fileLoading: false,
  fileError: null,
  blobUrl: null,
  draft: '',
  editing: false,
  saving: false,

  loadProjects: async () => {
    set({ loading: true, error: null })
    try {
      const res = await _authedFetch('/fd/vfs/projects')
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      set({ projects: data.projects || [], loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  openProject: async (name, owner = '', permission = '') => {
    set({ project: name, owner, permission, path: '', file: null, editing: false })
    await get().browse('')
  },

  closeProject: () => set({ project: null, owner: '', permission: '', path: '', entries: [], file: null, editing: false }),

  browse: async (path) => {
    const project = get().project
    if (!project) return
    set({ loading: true, error: null })
    try {
      const res = await _authedFetch(`/fd/vfs/list?${qp({ project, path, ...(get().owner ? { owner: get().owner } : {}) })}`)
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      set({ entries: data.entries || [], path, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  refresh: async () => {
    if (get().project) await get().browse(get().path)
    else await get().loadProjects()
  },

  openFile: async (entry) => {
    if (entry.type === 'dir') return get().browse(entry.path)
    // revoke any previous object URL before opening a new file
    const prev = get().blobUrl
    if (prev) URL.revokeObjectURL(prev)
    set({
      fileLoading: true,
      fileError: null,
      editing: false,
      blobUrl: null,
      file: { project: entry.project, path: entry.path, name: entry.name, size: entry.size, binary: false, truncated: false, text: '' },
    })
    try {
      if (IMAGE_EXTS.includes(extOf(entry.name))) {
        // Images: fetch bytes and hand the viewer an object URL (an <img src>
        // can't carry the bearer header that /download requires).
        const res = await _authedFetch(`/fd/vfs/download?${qp({ project: entry.project, path: entry.path, ...(get().owner ? { owner: get().owner } : {}) })}`)
        if (!res.ok) throw new Error(await res.text())
        const url = URL.createObjectURL(await res.blob())
        set({ blobUrl: url, fileLoading: false, file: { project: entry.project, path: entry.path, name: entry.name, size: entry.size, binary: true, truncated: false, text: '' } })
        return
      }
      const res = await _authedFetch(`/fd/vfs/read?${qp({ project: entry.project, path: entry.path, ...(get().owner ? { owner: get().owner } : {}) })}`)
      if (!res.ok) throw new Error(await res.text())
      const data: VFSFile = await res.json()
      set({ file: data, draft: data.text, fileLoading: false })
    } catch (e) {
      set({ fileError: String(e), fileLoading: false })
    }
  },

  closeFile: () => {
    const prev = get().blobUrl
    if (prev) URL.revokeObjectURL(prev)
    set({ file: null, editing: false, draft: '', blobUrl: null, fileError: null })
  },
  setDraft: (text) => set({ draft: text }),
  startEdit: () => set({ editing: true }),

  saveFile: async () => {
    const f = get().file
    if (!f) return
    set({ saving: true })
    try {
      const res = await _authedFetch('/fd/vfs/write', {
        method: 'POST',
        body: JSON.stringify({ project: f.project, path: f.path, content: get().draft, owner: get().owner }),
      })
      if (!res.ok) throw new Error(await res.text())
      set({
        saving: false,
        editing: false,
        file: { ...f, text: get().draft, size: get().draft.length },
      })
      await get().refresh()
    } catch (e) {
      set({ error: String(e), saving: false })
    }
  },

  download: async (entry) => {
    const res = await _authedFetch(`/fd/vfs/download?${qp({ project: entry.project, path: entry.path, ...(get().owner ? { owner: get().owner } : {}) })}`)
    if (!res.ok) return
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = entry.name
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  },

  exportToDrive: async (entry, folderId) => {
    const res = await _authedFetch('/fd/vfs/export-to-drive', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        project: entry.project, path: entry.path,
        ...(get().owner ? { owner: get().owner } : {}),
        folder_id: folderId || '',
      }),
    })
    if (!res.ok) return { ok: false, error: (await res.text().catch(() => '')) || `HTTP ${res.status}` }
    const data = await res.json()
    return { ok: true, link: data.link }
  },

  copyShared: async (project, owner) => {
    const res = await _authedFetch('/fd/vfs/copy-shared', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project, owner }),
    })
    if (!res.ok) return { ok: false, error: (await res.text().catch(() => '')) || `HTTP ${res.status}` }
    const data = await res.json()
    await get().loadProjects()
    return { ok: true, project: data.project }
  },

  downloadProject: async (name) => {
    const o = get().project === name ? get().owner : ''
    const res = await _authedFetch(`/fd/vfs/download-zip?${qp({ project: name, ...(o ? { owner: o } : {}) })}`)
    if (!res.ok) return
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}.zip`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  },

  deleteEntry: async (entry) => {
    const res = await _authedFetch(
      `/fd/vfs/entry?${qp({ project: entry.project, path: entry.path, recursive: 'true', ...(get().owner ? { owner: get().owner } : {}) })}`,
      { method: 'DELETE' },
    )
    if (!res.ok) {
      set({ error: await res.text() })
      return
    }
    if (get().file?.path === entry.path) get().closeFile()
    await get().refresh()
  },

  newFolder: async (name) => {
    const project = get().project
    if (!project || !name.trim()) return
    const path = get().path ? `${get().path}/${name.trim()}` : name.trim()
    const res = await _authedFetch('/fd/vfs/mkdir', {
      method: 'POST',
      body: JSON.stringify({ project, path, owner: get().owner }),
    })
    if (res.ok) await get().refresh()
  },

  newProject: async (name) => {
    const project = name.trim()
    if (!project) return
    set({ error: null })
    // A project is just a top-level directory: mkdir with an empty inner path
    // creates the project root.
    const res = await _authedFetch('/fd/vfs/mkdir', {
      method: 'POST',
      body: JSON.stringify({ project, path: '' }),
    })
    if (!res.ok) {
      set({ error: await res.text() })
      return
    }
    await get().loadProjects()
    await get().openProject(project)
  },

  uploadFiles: async (files) => {
    const project = get().project
    const list = Array.from(files)
    if (!project || list.length === 0) return
    set({ loading: true, error: null })
    try {
      const form = new FormData()
      form.append('project', project)
      form.append('path', get().path)
      if (get().owner) form.append('owner', get().owner)
      for (const f of list) form.append('files', f)
      // Multipart body — do NOT set Content-Type; the browser adds the boundary.
      const { token, authEnabled } = useAuthStore.getState()
      const headers: Record<string, string> = {}
      if (authEnabled && token) headers['Authorization'] = `Bearer ${token}`
      let res = await fetch('/fd/vfs/upload', { method: 'POST', headers, body: form, credentials: 'include' })
      if (res.status === 401 && authEnabled && (await refreshAccessToken())) {
        const t2 = useAuthStore.getState().token
        res = await fetch('/fd/vfs/upload', {
          method: 'POST',
          headers: t2 ? { Authorization: `Bearer ${t2}` } : {},
          body: form,
          credentials: 'include',
        })
      }
      if (!res.ok) throw new Error(await res.text())
      set({ loading: false })
      await get().refresh()
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  deleteProject: async (name) => {
    const res = await _authedFetch(`/fd/vfs/project?${qp({ project: name })}`, { method: 'DELETE' })
    if (res.ok) {
      if (get().project === name) get().closeProject()
      await get().loadProjects()
    }
  },

  addLink: async (name, path, mode) => {
    const res = await _authedFetch('/fd/vfs/links', {
      method: 'POST', body: JSON.stringify({ name, path, mode }),
    })
    if (!res.ok) throw new Error((await res.text()) || 'link failed')
    await get().loadProjects()
  },

  browseFs: async (path) => {
    const res = await _authedFetch(`/fd/vfs/browse-fs?${qp({ path })}`)
    if (!res.ok) return { path, parent: '', dirs: [] }
    return res.json()
  },

  browseDrive: async (folderId, driveId = '') => {
    const res = await _authedFetch(
      `/fd/vfs/drive/browse?${qp({ folder_id: folderId || 'root', ...(driveId ? { drive_id: driveId } : {}) })}`,
    )
    if (!res.ok) throw new Error((await res.text()) || 'Drive browse failed')
    return res.json()
  },

  mountDrive: async (name, folderId, clonemd, driveId = '', path = '', onProgress) => {
    // Stream NDJSON progress so a large folder shows a live status line instead
    // of a bare spinner. Each line is a DriveMountEvent; 'error' aborts, the
    // stream simply ends on success.
    const res = await _authedFetch('/fd/vfs/links/gdrive/stream', {
      method: 'POST',
      body: JSON.stringify({ name, folder_id: folderId, clonemd, drive_id: driveId, path }),
    })
    if (!res.ok || !res.body) throw new Error((await res.text().catch(() => '')) || 'mount failed')
    const reader = res.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    let failed: string | null = null
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let nl: number
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim()
        buf = buf.slice(nl + 1)
        if (!line) continue
        let ev: DriveMountEvent
        try { ev = JSON.parse(line) as DriveMountEvent } catch { continue }
        if (ev.event === 'error') failed = ev.detail || 'mount failed'
        else if (ev.event === 'progress') onProgress?.(ev)
      }
    }
    if (failed) throw new Error(failed)
    await get().loadProjects()
  },

  refreshDrive: async (name) => {
    const res = await _authedFetch(`/fd/vfs/links/gdrive/${encodeURIComponent(name)}/refresh`, {
      method: 'POST',
    })
    if (!res.ok) throw new Error((await res.text()) || 'refresh failed')
    const d = await res.json()
    await get().loadProjects()
    if (get().project === name) await get().browse(get().path)
    const parts = [`${d.files} file(s)`]
    if (d.cloned) parts.push(`${d.cloned} cloned`)
    if (d.truncated) parts.push('capped')
    return parts.join(', ')
  },

  toggleClonemd: async (name, clonemd) => {
    const res = await _authedFetch(`/fd/vfs/links/gdrive/${encodeURIComponent(name)}/clonemd`, {
      method: 'POST',
      body: JSON.stringify({ clonemd }),
    })
    if (!res.ok) throw new Error((await res.text()) || 'toggle failed')
    const d = await res.json()
    await get().loadProjects()
    if (get().project === name) await get().browse(get().path)
    return clonemd
      ? `Cloning on — ${d.cloned ?? 0} file(s) converted to Markdown.`
      : 'Cloning off. Existing Markdown files kept.'
  },

  unmountDrive: async (name, keepCloned) => {
    const res = await _authedFetch(
      `/fd/vfs/links/gdrive/${encodeURIComponent(name)}?${qp({ keep_cloned: String(keepCloned) })}`,
      { method: 'DELETE' },
    )
    if (!res.ok) throw new Error((await res.text()) || 'unmount failed')
    if (get().project === name) get().closeProject()
    await get().loadProjects()
  },
}))
