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
  login: { ok: boolean; how: string }
}

export interface CrewMe {
  offered: boolean
  allowed: boolean
  trusted?: boolean
  max_parallel?: number
  harnesses?: CrewHarness[]
  logins?: Record<string, { hint: string; saved_at: string }>
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

export interface CrewTask {
  id: string
  title: string
  kind: 'ship' | 'scout'
  harness: string
  harness_label: string
  model: string
  effort: string
  status: CrewStatus
  note: string
  repo_label: string
  repo_kind: string
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

export interface CrewRepos {
  projects: { project: string; folders: string[] }[]
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
  tasks: () => api<{ tasks: CrewTask[] }>('/fd/crew/tasks'),
  task: (id: string) => api<CrewTask>(`/fd/crew/tasks/${id}`),
  events: (id: string, after: number) => api<{ events: CrewEvent[] }>(`/fd/crew/tasks/${id}/events?after=${after}`),
  diff: (id: string) => api<{ committed: string; uncommitted: string; note?: string }>(`/fd/crew/tasks/${id}/diff`),
  dispatch: (body: CrewDispatch) => post<CrewTask>('/fd/crew/tasks', body),
  steer: (id: string, text: string) => post<CrewTask>(`/fd/crew/tasks/${id}/steer`, { text }),
  stop: (id: string) => post<CrewTask>(`/fd/crew/tasks/${id}/stop`),
  pr: (id: string) => post<CrewTask>(`/fd/crew/tasks/${id}/pr`),
  merge: (id: string) => post<CrewTask>(`/fd/crew/tasks/${id}/merge`),
  close: (id: string, discard = false) => post<CrewTask>(`/fd/crew/tasks/${id}/close?discard=${discard}`),
  saveLogin: (name: string, secret: string) =>
    api<{ hint: string }>(`/fd/crew/logins/${name}`, { method: 'PUT', body: JSON.stringify({ secret }) }),
  deleteLogin: (name: string) => api<{ ok: boolean }>(`/fd/crew/logins/${name}`, { method: 'DELETE' }),
}
