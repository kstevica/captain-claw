import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// ── Types (mirror captain_claw/flight_deck/code_routes.py) ───────────

export interface CodeProject {
  name: string
  files: number
  messages: number
  last_message?: string
  mtime?: number
  status?: string
}

export interface CodeRoute {
  size: 'small' | 'big'
  planner?: string
  small_archetype?: string
  domain?: string
  difficulty?: string
  title?: string
  why?: string
}

export interface CodeFinding { title: string; severity: string; file?: string }

export interface CodeMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  ts: number
  archetype?: string
  size?: string
  ok?: boolean
  commit?: string
  route?: CodeRoute
  // Big-job phases: plan | build | review | fix | note | approval
  kind?: string
  findings?: CodeFinding[]
  needs_fix?: boolean
  round?: number
}

export interface CodeCommit {
  sha: string
  short: string
  message: string
  ts: number
}

export interface CodeProgressEvent {
  i: number
  ts?: number
  stage: string
  message: string
  ok?: boolean
  agent?: string
  tool?: string
  detail?: string
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
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

const enc = encodeURIComponent

async function apiListProjects(): Promise<CodeProject[]> {
  const res = await _authedFetch('/fd/code/projects')
  if (!res.ok) return []
  const data = await res.json()
  return Array.isArray(data.projects) ? data.projects : []
}

async function apiCreateProject(name: string): Promise<CodeProject> {
  const res = await _authedFetch('/fd/code/projects', {
    method: 'POST', body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'create failed')
  return res.json()
}

async function apiGetChat(project: string): Promise<{ messages: CodeMessage[]; state: Record<string, unknown> }> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/chat`)
  if (!res.ok) return { messages: [], state: {} }
  return res.json()
}

async function apiProgress(project: string): Promise<{ events: CodeProgressEvent[]; active: boolean }> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/progress`)
  if (!res.ok) return { events: [], active: false }
  return res.json()
}

async function apiLog(project: string): Promise<CodeCommit[]> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/log`)
  if (!res.ok) return []
  return (await res.json()).commits || []
}

async function apiMessage(project: string, text: string): Promise<{ message: CodeMessage; route: CodeRoute; commit: string | null }> {
  const res = await _authedFetch('/fd/code/message', {
    method: 'POST', body: JSON.stringify({ project, text }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'message failed')
  return res.json()
}

async function apiApprove(project: string, plan?: string): Promise<void> {
  const res = await _authedFetch('/fd/code/plan/approve', {
    method: 'POST', body: JSON.stringify({ project, ...(plan ? { plan } : {}) }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'approve failed')
}

async function apiDiff(project: string, refA = '', refB = ''): Promise<string> {
  const qs = new URLSearchParams({ ref_a: refA, ref_b: refB }).toString()
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/diff?${qs}`)
  if (!res.ok) return ''
  return (await res.json()).diff || ''
}

async function apiShow(project: string, sha: string): Promise<string> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/show?sha=${enc(sha)}`)
  if (!res.ok) return ''
  return (await res.json()).diff || ''
}

async function apiRollback(project: string, ref: string): Promise<void> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/rollback`, {
    method: 'POST', body: JSON.stringify({ ref }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'rollback failed')
}

async function apiExport(project: string): Promise<void> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/export?format=md`)
  if (!res.ok) return
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${project}-process.md`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

// ── Store ─────────────────────────────────────────────────────────────

interface CodeStore {
  projects: CodeProject[]
  activeProject: string
  messages: CodeMessage[]
  commits: CodeCommit[]
  progress: CodeProgressEvent[]
  status: string            // current project run state: idle | running | awaiting_plan
  sending: boolean
  loading: boolean
  error: string | null

  loadProjects: () => Promise<void>
  createProject: (name: string) => Promise<void>
  selectProject: (name: string) => Promise<void>
  send: (text: string) => Promise<void>
  approvePlan: (plan?: string) => Promise<void>
  diff: (refA?: string, refB?: string) => Promise<string>
  showCommit: (sha: string) => Promise<string>
  rollback: (ref: string) => Promise<void>
  exportProcess: () => Promise<void>
}

function _statusOf(state: Record<string, unknown>): string {
  return (state?.status as string) || 'idle'
}

export const useCodeStore = create<CodeStore>((set, get) => ({
  projects: [],
  activeProject: '',
  messages: [],
  commits: [],
  progress: [],
  status: 'idle',
  sending: false,
  loading: false,
  error: null,

  loadProjects: async () => {
    set({ loading: true })
    try {
      const projects = await apiListProjects()
      set({ projects })
    } finally {
      set({ loading: false })
    }
  },

  createProject: async (name) => {
    set({ error: null })
    try {
      await apiCreateProject(name.trim())
      await get().loadProjects()
      await get().selectProject(name.trim())
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'create failed' })
    }
  },

  selectProject: async (name) => {
    set({ activeProject: name, messages: [], commits: [], progress: [], status: 'idle' })
    const [chat, commits] = await Promise.all([apiGetChat(name), apiLog(name)])
    set({ messages: chat.messages, commits, status: _statusOf(chat.state) })
  },

  send: async (text) => {
    const project = get().activeProject
    if (!project || !text.trim()) return
    // Optimistic user bubble.
    set((s) => ({
      sending: true, error: null, progress: [],
      messages: [...s.messages, { id: 'tmp', role: 'user', text: text.trim(), ts: Date.now() / 1000 }],
    }))
    const poll = setInterval(async () => {
      try { const p = await apiProgress(project); set({ progress: p.events || [] }) } catch { /* ignore */ }
    }, 700)
    try {
      await apiMessage(project, text.trim())
      const [chat, commits] = await Promise.all([apiGetChat(project), apiLog(project)])
      set({ messages: chat.messages, commits, status: _statusOf(chat.state) })
      await get().loadProjects()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'message failed' })
    } finally {
      clearInterval(poll)
      try { const p = await apiProgress(project); set({ progress: p.events || [] }) } catch { /* ignore */ }
      set({ sending: false })
    }
  },

  approvePlan: async (plan) => {
    const project = get().activeProject
    if (!project) return
    set({ sending: true, error: null, progress: [], status: 'running' })
    try {
      await apiApprove(project, plan)
      // The build → review → fix loop runs in the background. Follow it: poll
      // progress for the live trace and the chat/state until it goes idle.
      for (;;) {
        await new Promise((r) => setTimeout(r, 1500))
        const [prog, chat] = await Promise.all([apiProgress(project), apiGetChat(project)])
        set({ progress: prog.events || [], messages: chat.messages, status: _statusOf(chat.state) })
        if (_statusOf(chat.state) !== 'running' && !prog.active) break
      }
      const commits = await apiLog(project)
      set({ commits })
      await get().loadProjects()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'approve failed' })
    } finally {
      set({ sending: false })
    }
  },

  diff: async (refA = '', refB = '') => apiDiff(get().activeProject, refA, refB),

  showCommit: async (sha) => apiShow(get().activeProject, sha),

  exportProcess: async () => {
    const project = get().activeProject
    if (project) await apiExport(project)
  },

  rollback: async (ref) => {
    const project = get().activeProject
    if (!project) return
    try {
      await apiRollback(project, ref)
      const [chat, commits] = await Promise.all([apiGetChat(project), apiLog(project)])
      set({ messages: chat.messages, commits, status: _statusOf(chat.state) })
      await get().loadProjects()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'rollback failed' })
    }
  },
}))
