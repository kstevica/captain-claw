import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'

// ── Types (mirror captain_claw/flight_deck/code_routes.py) ───────────

// A folder = one git repo an agent works in (VFS sub-dir, the project dir
// itself, or an external linked folder). Membership is per-project.
export interface CodeFolder {
  name: string
  kind: string       // 'vfs' | 'self' | 'link'
  linked?: boolean
  mode?: string      // 'rw' | 'ro' for linked folders
  files: number
  missing?: boolean
}

// A session = one conversation thread targeting a folder.
export interface CodeSession {
  id: string
  title: string
  folder: string
  messages: number
  status: string     // idle | running | awaiting_plan
  last_message?: string
  created?: number
}

// A project groups folders + sessions.
export interface CodeProject {
  name: string
  folders: CodeFolder[]
  sessions: CodeSession[]
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
  kind?: string
  findings?: CodeFinding[]
  needs_fix?: boolean
  round?: number
}

export interface CodeCommit { sha: string; short: string; message: string; ts: number }

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
const sbase = (project: string, session: string) =>
  `/fd/code/projects/${enc(project)}/sessions/${enc(session)}`

async function apiListProjects(): Promise<CodeProject[]> {
  const res = await _authedFetch('/fd/code/projects')
  if (!res.ok) return []
  const data = await res.json()
  return Array.isArray(data.projects) ? data.projects : []
}

async function apiCreateProject(name: string): Promise<void> {
  const res = await _authedFetch('/fd/code/projects', { method: 'POST', body: JSON.stringify({ name }) })
  if (!res.ok) throw new Error((await res.text()) || 'create project failed')
}

async function apiAddFolder(project: string, folder: string): Promise<void> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/folders`, {
    method: 'POST', body: JSON.stringify({ folder }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'add folder failed')
}

async function apiLinkFolder(project: string, body: Record<string, unknown>): Promise<void> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/link`, {
    method: 'POST', body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.text()) || 'link folder failed')
}

async function apiCreateSession(project: string, title: string, folder: string): Promise<CodeSession> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/sessions`, {
    method: 'POST', body: JSON.stringify({ title, folder }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'create session failed')
  return (await res.json()).session
}

async function apiDeleteSession(project: string, session: string): Promise<void> {
  await _authedFetch(`/fd/code/projects/${enc(project)}/sessions/${enc(session)}`, { method: 'DELETE' })
}

async function apiSetFolder(project: string, session: string, folder: string): Promise<void> {
  const res = await _authedFetch(`${sbase(project, session)}/folder`, {
    method: 'PUT', body: JSON.stringify({ folder }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'set folder failed')
}

async function apiGetChat(p: string, s: string): Promise<{ messages: CodeMessage[]; state: Record<string, unknown> }> {
  const res = await _authedFetch(`${sbase(p, s)}/chat`)
  if (!res.ok) return { messages: [], state: {} }
  return res.json()
}

async function apiProgress(p: string, s: string): Promise<{ events: CodeProgressEvent[]; active: boolean }> {
  const res = await _authedFetch(`${sbase(p, s)}/progress`)
  if (!res.ok) return { events: [], active: false }
  return res.json()
}

async function apiLog(p: string, s: string): Promise<CodeCommit[]> {
  const res = await _authedFetch(`${sbase(p, s)}/log`)
  if (!res.ok) return []
  return (await res.json()).commits || []
}

async function apiMessage(project: string, session: string, text: string): Promise<void> {
  const res = await _authedFetch('/fd/code/message', {
    method: 'POST', body: JSON.stringify({ project, session, text }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'message failed')
}

async function apiApprove(project: string, session: string, plan?: string): Promise<void> {
  const res = await _authedFetch('/fd/code/plan/approve', {
    method: 'POST', body: JSON.stringify({ project, session, ...(plan ? { plan } : {}) }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'approve failed')
}

async function apiDiff(p: string, s: string, refA = '', refB = ''): Promise<string> {
  const qs = new URLSearchParams({ ref_a: refA, ref_b: refB }).toString()
  const res = await _authedFetch(`${sbase(p, s)}/diff?${qs}`)
  if (!res.ok) return ''
  return (await res.json()).diff || ''
}

async function apiShow(p: string, s: string, sha: string): Promise<string> {
  const res = await _authedFetch(`${sbase(p, s)}/show?sha=${enc(sha)}`)
  if (!res.ok) return ''
  return (await res.json()).diff || ''
}

async function apiRollback(p: string, s: string, ref: string): Promise<void> {
  const res = await _authedFetch(`${sbase(p, s)}/rollback`, { method: 'POST', body: JSON.stringify({ ref }) })
  if (!res.ok) throw new Error((await res.text()) || 'rollback failed')
}

export interface CodeMap {
  overview: string
  models: Record<string, unknown> | null
  ui: Record<string, unknown> | null
  stats: { files: number; symbols: number; summarized: number }
}
export interface CodeMapHit { name: string; kind: string; file: string; line: number; signature: string; summary: string }

async function apiGetMap(p: string, s: string): Promise<CodeMap | null> {
  const res = await _authedFetch(`${sbase(p, s)}/map`)
  if (!res.ok) return null
  return res.json()
}

async function apiMapSearch(p: string, s: string, q: string): Promise<CodeMapHit[]> {
  const res = await _authedFetch(`${sbase(p, s)}/map/search?q=${enc(q)}`)
  if (!res.ok) return []
  return (await res.json()).results || []
}

async function apiMapBuild(p: string, s: string): Promise<void> {
  await _authedFetch(`${sbase(p, s)}/map/build`, { method: 'POST' })
}

async function apiExport(p: string, s: string, title: string): Promise<void> {
  const res = await _authedFetch(`${sbase(p, s)}/export?format=md`)
  if (!res.ok) return
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${title.replace(/[^a-zA-Z0-9._-]/g, '-')}-process.md`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

// ── Store ─────────────────────────────────────────────────────────────

interface CodeStore {
  projects: CodeProject[]
  activeProject: string
  activeSession: string
  messages: CodeMessage[]
  commits: CodeCommit[]
  progress: CodeProgressEvent[]
  status: string
  sending: boolean
  loading: boolean
  error: string | null

  loadProjects: () => Promise<void>
  createProject: (name: string) => Promise<void>
  addFolder: (project: string, folder: string) => Promise<void>
  linkFolder: (project: string, name: string, path: string, mode: string) => Promise<void>
  createSession: (project: string, title: string, folder: string) => Promise<void>
  deleteSession: (project: string, session: string) => Promise<void>
  setSessionFolder: (folder: string) => Promise<void>
  selectSession: (project: string, session: string) => Promise<void>
  send: (text: string) => Promise<void>
  approvePlan: (plan?: string) => Promise<void>
  diff: (refA?: string, refB?: string) => Promise<string>
  showCommit: (sha: string) => Promise<string>
  rollback: (ref: string) => Promise<void>
  exportProcess: () => Promise<void>
  loadMap: () => Promise<CodeMap | null>
  searchMap: (q: string) => Promise<CodeMapHit[]>
  buildMap: () => Promise<void>
}

function _statusOf(state: Record<string, unknown>): string {
  return (state?.status as string) || 'idle'
}

export const useCodeStore = create<CodeStore>((set, get) => ({
  projects: [],
  activeProject: '',
  activeSession: '',
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
      set({ projects: await apiListProjects() })
    } finally {
      set({ loading: false })
    }
  },

  createProject: async (name) => {
    set({ error: null })
    try { await apiCreateProject(name.trim()); await get().loadProjects() }
    catch (e) { set({ error: e instanceof Error ? e.message : 'create failed' }) }
  },

  addFolder: async (project, folder) => {
    set({ error: null })
    try { await apiAddFolder(project, folder.trim()); await get().loadProjects() }
    catch (e) { set({ error: e instanceof Error ? e.message : 'add folder failed' }) }
  },

  linkFolder: async (project, name, path, mode) => {
    set({ error: null })
    try { await apiLinkFolder(project, { name: name.trim(), path: path.trim(), mode }); await get().loadProjects() }
    catch (e) { set({ error: e instanceof Error ? e.message : 'link failed' }) }
  },

  createSession: async (project, title, folder) => {
    set({ error: null })
    try {
      const sess = await apiCreateSession(project, title.trim(), folder)
      await get().loadProjects()
      await get().selectSession(project, sess.id)
    } catch (e) { set({ error: e instanceof Error ? e.message : 'create session failed' }) }
  },

  deleteSession: async (project, session) => {
    await apiDeleteSession(project, session)
    if (get().activeSession === session) set({ activeSession: '', activeProject: '', messages: [], commits: [] })
    await get().loadProjects()
  },

  setSessionFolder: async (folder) => {
    const { activeProject, activeSession } = get()
    if (!activeProject || !activeSession) return
    try {
      await apiSetFolder(activeProject, activeSession, folder)
      const commits = await apiLog(activeProject, activeSession)
      set({ commits })
      await get().loadProjects()
    } catch (e) { set({ error: e instanceof Error ? e.message : 'set folder failed' }) }
  },

  selectSession: async (project, session) => {
    set({ activeProject: project, activeSession: session, messages: [], commits: [], progress: [], status: 'idle' })
    const [chat, commits] = await Promise.all([apiGetChat(project, session), apiLog(project, session)])
    set({ messages: chat.messages, commits, status: _statusOf(chat.state) })
  },

  send: async (text) => {
    const { activeProject: p, activeSession: s } = get()
    if (!p || !s || !text.trim()) return
    set((st) => ({
      sending: true, error: null, progress: [],
      messages: [...st.messages, { id: 'tmp', role: 'user', text: text.trim(), ts: Date.now() / 1000 }],
    }))
    const poll = setInterval(async () => {
      try { const pr = await apiProgress(p, s); set({ progress: pr.events || [] }) } catch { /* ignore */ }
    }, 700)
    try {
      await apiMessage(p, s, text.trim())
      const [chat, commits] = await Promise.all([apiGetChat(p, s), apiLog(p, s)])
      set({ messages: chat.messages, commits, status: _statusOf(chat.state) })
      await get().loadProjects()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'message failed' })
    } finally {
      clearInterval(poll)
      try { const pr = await apiProgress(p, s); set({ progress: pr.events || [] }) } catch { /* ignore */ }
      set({ sending: false })
    }
  },

  approvePlan: async (plan) => {
    const { activeProject: p, activeSession: s } = get()
    if (!p || !s) return
    set({ sending: true, error: null, progress: [], status: 'running' })
    try {
      await apiApprove(p, s, plan)
      for (;;) {
        await new Promise((r) => setTimeout(r, 1500))
        const [prog, chat] = await Promise.all([apiProgress(p, s), apiGetChat(p, s)])
        set({ progress: prog.events || [], messages: chat.messages, status: _statusOf(chat.state) })
        if (_statusOf(chat.state) !== 'running' && !prog.active) break
      }
      set({ commits: await apiLog(p, s) })
      await get().loadProjects()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'approve failed' })
    } finally {
      set({ sending: false })
    }
  },

  diff: async (refA = '', refB = '') => apiDiff(get().activeProject, get().activeSession, refA, refB),
  showCommit: async (sha) => apiShow(get().activeProject, get().activeSession, sha),

  exportProcess: async () => {
    const { activeProject: p, activeSession: s, projects } = get()
    if (!p || !s) return
    const title = projects.find((x) => x.name === p)?.sessions.find((x) => x.id === s)?.title || `${p}-${s}`
    await apiExport(p, s, `${p}-${title}`)
  },

  loadMap: async () => {
    const { activeProject: p, activeSession: s } = get()
    return (p && s) ? apiGetMap(p, s) : null
  },
  searchMap: async (q) => {
    const { activeProject: p, activeSession: s } = get()
    return (p && s && q.trim()) ? apiMapSearch(p, s, q.trim()) : []
  },
  buildMap: async () => {
    const { activeProject: p, activeSession: s } = get()
    if (p && s) await apiMapBuild(p, s)
  },

  rollback: async (ref) => {
    const { activeProject: p, activeSession: s } = get()
    if (!p || !s) return
    try {
      await apiRollback(p, s, ref)
      const [chat, commits] = await Promise.all([apiGetChat(p, s), apiLog(p, s)])
      set({ messages: chat.messages, commits, status: _statusOf(chat.state) })
      await get().loadProjects()
    } catch (e) { set({ error: e instanceof Error ? e.message : 'rollback failed' }) }
  },
}))
