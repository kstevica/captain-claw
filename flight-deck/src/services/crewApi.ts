/**
 * Crew — coding agents (Claude Code, Codex, OpenCode, Pi) run for a user, each in
 * a git worktree of its own, on an admin's approval
 * (`captain_claw/flight_deck/crew_routes.py`). Every read is the caller's own.
 */

import { api } from './fileLibraryApi'

export const CREW_AGENTS = [
  { id: 'claude', label: 'Claude Code' },
  { id: 'codex', label: 'Codex' },
  { id: 'opencode', label: 'OpenCode' },
  { id: 'pi', label: 'Pi' },
] as const

export type CrewStatus = 'queued' | 'running' | 'waiting' | 'ready' | 'failed' | 'stopped' | 'pr_open' | 'delivered' | 'closed'

export interface CrewHarness {
  id: string
  label: string
  installed: boolean
  version: string
  install: string
  login: { ok: boolean; how: string; free?: boolean }
  default_model: string
  models: string[]
}

export interface CrewMe {
  offered: boolean
  allowed: boolean
  trusted?: boolean
  max_parallel?: number
  harnesses?: CrewHarness[]
  logins?: Record<string, { hint: string; saved_at: string }>
  free_terms_url?: string
}

export interface CrewEvidence {
  commits?: number
  dirty?: number
  stat?: string
  log?: string[]
  off_branch?: string
  stray?: number
  head?: string
}

/** What a task made, copied to the owner's VFS when it finished. */
export interface CrewResults {
  path?: string
  report?: string
  head?: string
  files_path?: string
  files?: string[]
  changed?: string[]
  count?: number
  whole?: boolean
  skipped?: number
  publish?: { project?: string; subdir?: string }
  link?: CrewLink
}

export interface CrewLink {
  name: string
  url: string
}

export interface CrewFiles extends CrewResults {
  uncommitted: number
  landed: boolean
  project_path: string
}

export interface CrewFile {
  path: string
  size: number
  binary: boolean
  truncated: boolean
  text: string
}

export interface CrewTask {
  id: string
  title: string
  kind: 'ship' | 'scout'
  harness: string
  harness_label: string
  model: string
  model_used: string
  effort: string
  status: CrewStatus
  note: string
  repo_label: string
  repo_kind: string
  project_key: string
  project_name: string
  results: CrewResults
  base_branch: string
  branch: string
  turn: number
  evidence: CrewEvidence
  tokens_in: number
  tokens_out: number
  cost_usd: number
  deliver: string
  delivery: { kind?: string; pr?: number; url?: string; sha?: string; base?: string }
  pending: number
  stalled: boolean
  from_agent: string
  created_at: number
  updated_at: number
  // One task, read whole:
  brief?: string
  final?: string
  report?: string
  worktree?: string
}

export interface CrewEvent {
  n: number
  at: number
  turn: number
  kind: 'text' | 'tool' | 'result' | 'error' | 'note' | 'turn' | 'status'
  text: string
  tool?: string
  status?: string
}

export interface CrewLimit {
  used: number | null
  resets_at: number
  status: string
}

/** A CLI's model and limits, as it last reported them while working, and a week's use. */
export interface CrewUsage {
  id: string
  label: string
  model: string
  limits: Record<string, CrewLimit>
  seen_at: number
  tasks: number
  tokens: number
  cost_usd: number
}

export interface CrewRepos {
  projects: { project: string; key: string; folders: string[]; paths: Record<string, string> }[]
  recent: { repo: string; repo_label: string; repo_kind: string }[]
}

export interface CrewDispatch {
  harness: string
  brief: string
  title?: string
  kind?: string
  deliver?: string
  project?: string
  folder?: string
  repo_url?: string
  repo_path?: string
  model?: string
  effort?: string
}

const post = <T>(path: string, body?: unknown) =>
  api<T>(path, { method: 'POST', ...(body === undefined ? {} : { body: JSON.stringify(body) }) })

export const crewApi = {
  me: () => api<CrewMe>('/fd/crew/me'),
  repos: () => api<CrewRepos>('/fd/crew/repos'),
  usage: () => api<{ harnesses: CrewUsage[] }>('/fd/crew/usage'),
  tasks: () => api<{ tasks: CrewTask[] }>('/fd/crew/tasks'),
  task: (id: string) => api<CrewTask>(`/fd/crew/tasks/${id}`),
  events: (id: string, after: number) => api<{ events: CrewEvent[] }>(`/fd/crew/tasks/${id}/events?after=${after}`),
  diff: (id: string) => api<{ committed: string; uncommitted: string; note?: string }>(`/fd/crew/tasks/${id}/diff`),
  files: (id: string, whole = false) => post<CrewFiles>(`/fd/crew/tasks/${id}/files?whole=${whole}`),
  file: (id: string, path: string) => api<CrewFile>(`/fd/crew/tasks/${id}/file?path=${encodeURIComponent(path)}`),
  link: (id: string) => post<CrewLink>(`/fd/crew/tasks/${id}/link`),
  newProject: (name: string) => post<{ project: string; key: string; path: string }>('/fd/crew/projects', { name }),
  tidy: (key: string) => post<{ cleared: number; kept: number }>('/fd/crew/projects/tidy', { key }),
  dispatch: (body: CrewDispatch) => post<CrewTask>('/fd/crew/tasks', body),
  steer: (id: string, text: string) => post<CrewTask>(`/fd/crew/tasks/${id}/steer`, { text }),
  stop: (id: string) => post<CrewTask>(`/fd/crew/tasks/${id}/stop`),
  pr: (id: string) => post<CrewTask>(`/fd/crew/tasks/${id}/pr`),
  merge: (id: string) => post<CrewTask>(`/fd/crew/tasks/${id}/merge`),
  close: (id: string, discard = false) => post<CrewTask>(`/fd/crew/tasks/${id}/close?discard=${discard}`),
  saveModel: (harness: string, model: string) =>
    api<{ harness: string; model: string }>(`/fd/crew/models/${harness}`, { method: 'PUT', body: JSON.stringify({ model }) }),
  saveLogin: (name: string, secret: string) =>
    api<{ hint: string }>(`/fd/crew/logins/${name}`, { method: 'PUT', body: JSON.stringify({ secret }) }),
  deleteLogin: (name: string) => api<{ ok: boolean }>(`/fd/crew/logins/${name}`, { method: 'DELETE' }),
}
