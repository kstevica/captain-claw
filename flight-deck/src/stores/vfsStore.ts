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
  openProject: (name: string) => Promise<void>
  closeProject: () => void
  browse: (path: string) => Promise<void>
  refresh: () => Promise<void>
  openFile: (entry: VFSEntry) => Promise<void>
  closeFile: () => void
  setDraft: (text: string) => void
  startEdit: () => void
  saveFile: () => Promise<void>
  download: (entry: VFSEntry) => Promise<void>
  downloadProject: (name: string) => Promise<void>
  deleteEntry: (entry: VFSEntry) => Promise<void>
  newFolder: (name: string) => Promise<void>
  deleteProject: (name: string) => Promise<void>
  addLink: (name: string, path: string, mode: string) => Promise<void>
  browseFs: (path: string) => Promise<FsListing>
}

export interface FsDir { name: string; hidden: boolean; is_git: boolean }
export interface FsListing { path: string; parent: string; dirs: FsDir[] }

export const useVFSStore = create<VFSStore>((set, get) => ({
  projects: [],
  project: null,
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

  openProject: async (name) => {
    set({ project: name, path: '', file: null, editing: false })
    await get().browse('')
  },

  closeProject: () => set({ project: null, path: '', entries: [], file: null, editing: false }),

  browse: async (path) => {
    const project = get().project
    if (!project) return
    set({ loading: true, error: null })
    try {
      const res = await _authedFetch(`/fd/vfs/list?${qp({ project, path })}`)
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
        const res = await _authedFetch(`/fd/vfs/download?${qp({ project: entry.project, path: entry.path })}`)
        if (!res.ok) throw new Error(await res.text())
        const url = URL.createObjectURL(await res.blob())
        set({ blobUrl: url, fileLoading: false, file: { project: entry.project, path: entry.path, name: entry.name, size: entry.size, binary: true, truncated: false, text: '' } })
        return
      }
      const res = await _authedFetch(`/fd/vfs/read?${qp({ project: entry.project, path: entry.path })}`)
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
        body: JSON.stringify({ project: f.project, path: f.path, content: get().draft }),
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
    const res = await _authedFetch(`/fd/vfs/download?${qp({ project: entry.project, path: entry.path })}`)
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

  downloadProject: async (name) => {
    const res = await _authedFetch(`/fd/vfs/download-zip?${qp({ project: name })}`)
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
      `/fd/vfs/entry?${qp({ project: entry.project, path: entry.path, recursive: 'true' })}`,
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
      body: JSON.stringify({ project, path }),
    })
    if (res.ok) await get().refresh()
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
}))
