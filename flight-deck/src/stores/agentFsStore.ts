import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// ── Types (mirror captain_claw/flight_deck/agents_fs_routes.py) ──────

export interface AgentFolder {
  name: string
  bytes: number
  files: number
  workspace_files: number
  mtime: number
  registered: boolean      // slug still in the process registry (exists in Agent Desktop)
  running: boolean         // process currently alive
  display_name: string
  owner: string
}

export interface AgentWsFile {
  path: string             // relative to <folder>/data/workspace
  name: string
  size: number
  mtime: number
  ext: string
  is_text: boolean
}

export interface AgentFilePreview {
  folder: string
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
const extOf = (name: string): string => {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

// ── Store ─────────────────────────────────────────────────────────────

interface AgentFsStore {
  folders: AgentFolder[]
  loading: boolean
  error: string | null
  // expanded folder + its workspace files
  expanded: string | null
  expandedFiles: AgentWsFile[]
  filesLoading: boolean
  // file preview modal
  preview: AgentFilePreview | null
  previewLoading: boolean
  previewError: string | null
  blobUrl: string | null   // object URL for image previews

  load: () => Promise<void>
  toggleFolder: (name: string) => Promise<void>
  deleteFolder: (name: string) => Promise<void>
  openFile: (folder: string, file: AgentWsFile) => Promise<void>
  closeFile: () => void
  download: (folder: string, file: AgentWsFile) => Promise<void>
}

export const useAgentFsStore = create<AgentFsStore>((set, get) => ({
  folders: [],
  loading: false,
  error: null,
  expanded: null,
  expandedFiles: [],
  filesLoading: false,
  preview: null,
  previewLoading: false,
  previewError: null,
  blobUrl: null,

  load: async () => {
    set({ loading: true, error: null })
    try {
      const res = await _authedFetch('/fd/agentfs/folders')
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      set({ folders: data.folders || [], loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  toggleFolder: async (name) => {
    if (get().expanded === name) {
      set({ expanded: null, expandedFiles: [] })
      return
    }
    set({ expanded: name, expandedFiles: [], filesLoading: true })
    try {
      const res = await _authedFetch(`/fd/agentfs/files?${qp({ folder: name })}`)
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      set({ expandedFiles: data.files || [], filesLoading: false })
    } catch (e) {
      set({ error: String(e), filesLoading: false })
    }
  },

  deleteFolder: async (name) => {
    const res = await _authedFetch(`/fd/agentfs/folder?${qp({ folder: name })}`, { method: 'DELETE' })
    if (!res.ok) {
      set({ error: await res.text() })
      return
    }
    if (get().expanded === name) set({ expanded: null, expandedFiles: [] })
    await get().load()
  },

  openFile: async (folder, file) => {
    const prev = get().blobUrl
    if (prev) URL.revokeObjectURL(prev)
    set({ previewLoading: true, previewError: null, blobUrl: null, preview: null })
    try {
      if (IMAGE_EXTS.includes(extOf(file.name))) {
        const res = await _authedFetch(`/fd/agentfs/download?${qp({ folder, path: file.path })}`)
        if (!res.ok) throw new Error(await res.text())
        const url = URL.createObjectURL(await res.blob())
        set({
          blobUrl: url,
          previewLoading: false,
          preview: { folder, path: file.path, name: file.name, size: file.size, binary: true, truncated: false, text: '' },
        })
        return
      }
      if (!file.is_text) {
        set({
          previewLoading: false,
          preview: { folder, path: file.path, name: file.name, size: file.size, binary: true, truncated: false, text: '' },
        })
        return
      }
      const res = await _authedFetch(`/fd/agentfs/view?${qp({ folder, path: file.path })}`)
      if (!res.ok) throw new Error(await res.text())
      const data: AgentFilePreview = await res.json()
      set({ preview: data, previewLoading: false })
    } catch (e) {
      set({ previewError: String(e), previewLoading: false })
    }
  },

  closeFile: () => {
    const prev = get().blobUrl
    if (prev) URL.revokeObjectURL(prev)
    set({ preview: null, previewError: null, blobUrl: null })
  },

  download: async (folder, file) => {
    const res = await _authedFetch(`/fd/agentfs/download?${qp({ folder, path: file.path })}`)
    if (!res.ok) return
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = file.name
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  },
}))
