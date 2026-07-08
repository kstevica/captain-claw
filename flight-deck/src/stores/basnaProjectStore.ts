import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// A project bundles Basna/Vatra runs under one theme (description + instructions
// injected into each run) and one VFS folder (uploads, auto-added read-only as a
// reference folder). Runs stay independent — the project just groups + seeds them.
export interface BasnaProject {
  id: string
  user_id: string
  name: string
  description: string
  instructions: string
  vfs_folder: string
  created_at: string
  updated_at: string
}

// Virtual project holding legacy / ungrouped runs (config has no project_id).
// It has no row, no theme, and no editable details — just a run bucket.
export const UNFILED_ID = '__unfiled__'
export const UNFILED: BasnaProject = {
  id: UNFILED_ID, user_id: '', name: 'Unfiled', description: '', instructions: '',
  vfs_folder: '', created_at: '', updated_at: '',
}

function _headers(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function _fetch(url: string, init: RequestInit = {}): Promise<Response> {
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

interface ProjectState {
  projects: BasnaProject[]
  current: BasnaProject | null   // the open project (or the Unfiled bucket); null = picker
  loading: boolean
  error: string | null

  loadProjects: () => Promise<void>
  createProject: (name: string, description?: string, instructions?: string) => Promise<BasnaProject | null>
  updateProject: (id: string, fields: Partial<Pick<BasnaProject, 'name' | 'description' | 'instructions'>>) => Promise<void>
  deleteProject: (id: string, deleteRuns: boolean) => Promise<void>
  select: (p: BasnaProject | null) => void
}

export const useBasnaProjectStore = create<ProjectState>((set, get) => ({
  projects: [],
  current: null,
  loading: false,
  error: null,

  loadProjects: async () => {
    set({ loading: true })
    try {
      const res = await _fetch('/fd/basna/projects')
      const rows: BasnaProject[] = res.ok ? await res.json() : []
      const projects = Array.isArray(rows) ? rows : []
      set({ projects })
      // Keep the open project in sync with the freshly-loaded list.
      const cur = get().current
      if (cur && cur.id !== UNFILED_ID) {
        const fresh = projects.find((p) => p.id === cur.id)
        if (fresh) set({ current: fresh })
      }
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'load projects failed' })
    } finally {
      set({ loading: false })
    }
  },

  createProject: async (name, description = '', instructions = '') => {
    set({ error: null })
    try {
      const res = await _fetch('/fd/basna/projects', {
        method: 'POST',
        body: JSON.stringify({ name: name.trim(), description, instructions }),
      })
      if (!res.ok) throw new Error((await res.text()) || 'create failed')
      const proj: BasnaProject = await res.json()
      set((s) => ({ projects: [proj, ...s.projects] }))
      return proj
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'create project failed' })
      return null
    }
  },

  updateProject: async (id, fields) => {
    if (id === UNFILED_ID) return
    try {
      const res = await _fetch(`/fd/basna/projects/${encodeURIComponent(id)}`, {
        method: 'PUT', body: JSON.stringify(fields),
      })
      if (!res.ok) throw new Error((await res.text()) || 'update failed')
      const proj: BasnaProject = await res.json()
      set((s) => ({
        projects: s.projects.map((p) => (p.id === id ? proj : p)),
        current: s.current?.id === id ? proj : s.current,
      }))
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'update project failed' })
    }
  },

  deleteProject: async (id, deleteRuns) => {
    if (id === UNFILED_ID) return
    try {
      await _fetch(`/fd/basna/projects/${encodeURIComponent(id)}?delete_runs=${deleteRuns ? 'true' : 'false'}`, {
        method: 'DELETE',
      })
      set((s) => ({
        projects: s.projects.filter((p) => p.id !== id),
        current: s.current?.id === id ? null : s.current,
      }))
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'delete project failed' })
    }
  },

  select: (p) => set({ current: p }),
}))
