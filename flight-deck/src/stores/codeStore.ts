import { create } from 'zustand'
import { useAuthStore, refreshAccessToken } from './authStore'
import { defaultProfile, fromResponse, toRequest } from '../services/quality'
import type { QualityProfile } from '../services/quality'

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
  source?: string    // 'user' | 'agent' (started by an agent via the code tool)
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
  usage?: string
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

async function apiMessage(project: string, session: string, text: string): Promise<{ status?: string }> {
  const res = await _authedFetch('/fd/code/message', {
    method: 'POST', body: JSON.stringify({ project, session, text }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'message failed')
  return res.json()
}

async function apiApprove(project: string, session: string, plan?: string): Promise<void> {
  const res = await _authedFetch('/fd/code/plan/approve', {
    method: 'POST', body: JSON.stringify({ project, session, ...(plan ? { plan } : {}) }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'approve failed')
}

async function apiCancelPlan(project: string, session: string): Promise<void> {
  const res = await _authedFetch('/fd/code/plan/cancel', {
    method: 'POST', body: JSON.stringify({ project, session }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'cancel failed')
}

async function apiGetQuality(project: string): Promise<QualityProfile> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/quality`)
  if (!res.ok) return defaultProfile()
  const d = await res.json()
  return fromResponse(d.quality)
}

async function apiSetQuality(project: string, quality: QualityProfile): Promise<void> {
  const res = await _authedFetch(`/fd/code/projects/${enc(project)}/quality`, {
    method: 'PUT', body: JSON.stringify({ quality: toRequest(quality) }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'save quality failed')
}

async function apiFollowup(project: string, session: string, kind: string): Promise<void> {
  const res = await _authedFetch('/fd/code/followup', {
    method: 'POST', body: JSON.stringify({ project, session, kind }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'follow-up failed')
}

async function apiStop(project: string, session: string): Promise<void> {
  const res = await _authedFetch(`${sbase(project, session)}/stop`, { method: 'POST' })
  if (!res.ok) throw new Error((await res.text()) || 'stop failed')
}

async function apiCleanup(): Promise<number> {
  const res = await _authedFetch('/fd/code/cleanup', { method: 'POST' })
  if (!res.ok) throw new Error((await res.text()) || 'cleanup failed')
  return (await res.json()).count || 0
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
  quality: QualityProfile      // per-project opt-in levers (all-off == current behaviour)
  qualitySaving: boolean

  loadQuality: (project: string) => Promise<void>
  saveQuality: (quality: QualityProfile) => Promise<void>
  followup: (kind: string) => Promise<void>
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
  cancelPlan: () => Promise<void>
  stopRun: () => Promise<void>
  cleanupAgents: () => Promise<number>
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

// Live-follow: a run started elsewhere (e.g. an agent's `code` tool from chat,
// WhatsApp, Telegram) writes progress + chat under the same project/session
// keys the UI reads. `selectSession` only snapshots once, so such a run would
// look frozen. This follows a live session until it settles, updating messages/
// progress/commits — identical to the `send` loop, but triggered by selection.
// A monotonic token cancels a stale follower the moment another session is
// selected; it also yields while `sending` (the send/approve loop owns polling).
let _followToken = 0

async function _followRun(
  get: () => CodeStore,
  set: (partial: Partial<CodeStore>) => void,
  project: string,
  session: string,
  token: number,
): Promise<void> {
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
  for (;;) {
    if (token !== _followToken) return
    const st = get()
    if (st.activeProject !== project || st.activeSession !== session) return
    if (st.sending) { await sleep(1500); continue }   // send/approve loop owns polling
    let prog: { events: CodeProgressEvent[]; active: boolean }
    let chat: { messages: CodeMessage[]; state: Record<string, unknown> }
    try {
      [prog, chat] = await Promise.all([apiProgress(project, session), apiGetChat(project, session)])
    } catch { await sleep(2000); continue }
    if (token !== _followToken) return
    const status = _statusOf(chat.state)
    set({ progress: prog.events || [], messages: chat.messages, status })
    if (status !== 'running' && !prog.active) {
      try { set({ commits: await apiLog(project, session) }) } catch { /* ignore */ }
      return
    }
    await sleep(1500)
  }
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
  quality: defaultProfile(),
  qualitySaving: false,

  loadQuality: async (project) => {
    try { set({ quality: await apiGetQuality(project) }) }
    catch { set({ quality: defaultProfile() }) }
  },

  saveQuality: async (quality) => {
    const project = get().activeProject
    set({ quality, qualitySaving: true })  // optimistic
    if (!project) { set({ qualitySaving: false }); return }
    try { await apiSetQuality(project, quality) }
    catch (e) { set({ error: e instanceof Error ? e.message : 'save quality failed' }) }
    finally { set({ qualitySaving: false }) }
  },

  followup: async (kind) => {
    const { activeProject, activeSession } = get()
    if (!activeProject || !activeSession) return
    set({ error: null, status: 'running' })
    try { await apiFollowup(activeProject, activeSession, kind) }
    catch (e) { set({ error: e instanceof Error ? e.message : 'follow-up failed', status: 'idle' }) }
  },

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
    const token = ++_followToken   // supersede any in-flight follower
    set({ activeProject: project, activeSession: session, messages: [], commits: [], progress: [], status: 'idle' })
    void get().loadQuality(project)   // per-project quality levers
    const [chat, commits] = await Promise.all([apiGetChat(project, session), apiLog(project, session)])
    if (token !== _followToken) return   // user already switched away
    const status = _statusOf(chat.state)
    set({ messages: chat.messages, commits, status })
    // If the run is live (e.g. started by an agent from chat), follow it until
    // it settles so status + messages stream in, just like a Code-started run.
    const prog = await apiProgress(project, session).catch(() => ({ events: [], active: false }))
    if (token !== _followToken) return
    set({ progress: prog.events || [] })
    if (status === 'running' || prog.active) void _followRun(get, set, project, session, token)
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
      let [chat, commits] = await Promise.all([apiGetChat(p, s), apiLog(p, s)])
      set({ messages: chat.messages, commits, status: _statusOf(chat.state) })
      // Backlog continuation (and other background runs) return immediately
      // while the loop keeps working — follow it until idle, like approvePlan.
      while (_statusOf(chat.state) === 'running') {
        await new Promise((r) => setTimeout(r, 1500))
        const [prog, chat2] = await Promise.all([apiProgress(p, s), apiGetChat(p, s)])
        chat = chat2
        set({ progress: prog.events || [], messages: chat.messages, status: _statusOf(chat.state) })
        if (_statusOf(chat.state) !== 'running' && !prog.active) break
      }
      commits = await apiLog(p, s)
      set({ commits })
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

  cancelPlan: async () => {
    const { activeProject: p, activeSession: s } = get()
    if (!p || !s) return
    try {
      await apiCancelPlan(p, s)
      const chat = await apiGetChat(p, s)
      set({ messages: chat.messages, status: _statusOf(chat.state) })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'cancel failed' })
    }
  },

  stopRun: async () => {
    const { activeProject: p, activeSession: s } = get()
    if (!p || !s) return
    try {
      await apiStop(p, s)
      // The loop winds down at its next phase boundary; the active
      // send/approve poller will observe the idle state and finish.
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'stop failed' })
    }
  },

  cleanupAgents: async () => {
    try {
      return await apiCleanup()
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'cleanup failed' })
      return 0
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
    if (!p || !s) return
    // The build runs in a background task on the server; poll progress +
    // state until it finishes (same pattern as approvePlan) so the UI shows
    // live cartographer progress and the caller can reload the map after.
    set({ sending: true, error: null, progress: [], status: 'running' })
    try {
      await apiMapBuild(p, s)
      for (;;) {
        await new Promise((r) => setTimeout(r, 1000))
        const [prog, chat] = await Promise.all([apiProgress(p, s), apiGetChat(p, s)])
        set({ progress: prog.events || [], status: _statusOf(chat.state) })
        if (_statusOf(chat.state) !== 'running' && !prog.active) break
      }
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'map build failed' })
    } finally {
      set({ sending: false, status: 'idle' })
    }
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
