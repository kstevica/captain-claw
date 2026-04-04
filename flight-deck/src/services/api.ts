// REST client for BotPort API
import type {
  InstanceInfo,
  Concern,
  BotPortStats,
  SwarmProject,
  Swarm,
  SwarmTask,
  SwarmEdge,
  TraceData,
} from '../types'
import { getApiBase } from '../stores/connectionStore'

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json()
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return fetchJSON<T>(path, {
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  })
}

function put<T>(path: string, body: unknown): Promise<T> {
  return fetchJSON<T>(path, {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

function del(path: string): Promise<void> {
  return fetch(`${getApiBase()}${path}`, { method: 'DELETE' }).then((r) => {
    if (!r.ok) throw new Error(`DELETE ${r.status}`)
  })
}

// ── Instances & Concerns ──

export const getInstances = () => fetchJSON<InstanceInfo[]>('/instances')
export const getConcerns = (active = false) =>
  fetchJSON<Concern[]>(`/concerns${active ? '?active=true' : ''}`)
export const getConcern = (id: string) => fetchJSON<Concern>(`/concerns/${id}`)
export const getStats = () => fetchJSON<BotPortStats>('/stats')
export const patchInstance = (id: string, data: { active_persona?: string; active_model?: string }) =>
  fetchJSON<InstanceInfo>(`/instances/${id}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  })

// ── Swarm Projects ──

export const getProjects = () => fetchJSON<SwarmProject[]>('/swarm/projects')
export const createProject = (data: { name: string; description?: string }) =>
  post<SwarmProject>('/swarm/projects', data)
export const deleteProject = (id: string) => del(`/swarm/projects/${id}`)

// ── Swarms ──

export const getSwarms = (projectId?: string) =>
  fetchJSON<Swarm[]>(`/swarm/swarms${projectId ? `?project_id=${projectId}` : ''}`)
export const getSwarm = (id: string) => fetchJSON<Swarm>(`/swarm/swarms/${id}`)
export const createSwarm = (data: Partial<Swarm>) =>
  post<Swarm>('/swarm/swarms', data)
export const updateSwarm = (id: string, data: Partial<Swarm>) =>
  put<Swarm>(`/swarm/swarms/${id}`, data)
export const deleteSwarm = (id: string) => del(`/swarm/swarms/${id}`)

// Swarm lifecycle
export const startSwarm = (id: string) => post<Swarm>(`/swarm/swarms/${id}/start`)
export const pauseSwarm = (id: string) => post<Swarm>(`/swarm/swarms/${id}/pause`)
export const resumeSwarm = (id: string) => post<Swarm>(`/swarm/swarms/${id}/resume`)
export const cancelSwarm = (id: string) => post<Swarm>(`/swarm/swarms/${id}/cancel`)
export const decomposeSwarm = (id: string) => post<void>(`/swarm/swarms/${id}/decompose`)
export const autoLayoutSwarm = (id: string) => post<void>(`/swarm/swarms/${id}/auto-layout`)

// ── Tasks ──

export const getTasks = (swarmId: string) =>
  fetchJSON<SwarmTask[]>(`/swarm/swarms/${swarmId}/tasks`)
export const createTask = (swarmId: string, data: Partial<SwarmTask>) =>
  post<SwarmTask>(`/swarm/swarms/${swarmId}/tasks`, data)
export const updateTask = (taskId: string, data: Partial<SwarmTask>) =>
  put<SwarmTask>(`/swarm/tasks/${taskId}`, data)
export const deleteTask = (taskId: string) => del(`/swarm/tasks/${taskId}`)
export const approveTask = (taskId: string) => post<void>(`/swarm/tasks/${taskId}/approve`)
export const retryTask = (taskId: string) => post<void>(`/swarm/tasks/${taskId}/retry`)
export const skipTask = (taskId: string) => post<void>(`/swarm/tasks/${taskId}/skip`)

// ── Edges ──

export const createEdge = (swarmId: string, data: { from_task_id: string; to_task_id: string; edge_type?: string }) =>
  post<SwarmEdge>(`/swarm/swarms/${swarmId}/edges`, data)
export const deleteEdge = (edgeId: number) => del(`/swarm/edges/${edgeId}`)

// ── Orchestrator (proxied to agent) ──

export interface RunTasksPayload {
  tasks: Array<{
    id: string
    title: string
    description: string
    depends_on?: string[]
    model_id?: string
    session_name?: string
    session_id?: string
    output_schema?: Record<string, unknown>
    output_schema_name?: string
    workspace_outputs?: string[]
    workspace_inputs?: string[]
  }>
  user_input?: string
  synthesis_instruction?: string
  workflow_name?: string
  model?: string
  variable_values?: Record<string, string>
  task_overrides?: Record<string, Record<string, unknown>>
}

interface RunTasksResult {
  ok: boolean
  result?: string
  error?: string
}

interface PrepareTasksResult {
  ok: boolean
  tasks?: Array<Record<string, unknown>>
  summary?: string
  error?: string
}

/**
 * Prepare explicit tasks on an agent (no LLM decomposition).
 * Proxied through Flight Deck → agent's /api/orchestrator/prepare-tasks.
 */
export const prepareTasksOnAgent = (agentSlug: string, payload: RunTasksPayload) =>
  post<PrepareTasksResult>(`/fd/orchestrator/${encodeURIComponent(agentSlug)}/prepare-tasks`, payload)

/**
 * Execute explicit tasks on an agent (prepare + execute in one call).
 * Proxied through Flight Deck → agent's /api/orchestrator/run-tasks.
 */
export const runTasksOnAgent = (agentSlug: string, payload: RunTasksPayload) =>
  post<RunTasksResult>(`/fd/orchestrator/${encodeURIComponent(agentSlug)}/run-tasks`, payload)

/**
 * Fetch trace spans from an agent's current orchestration run.
 * Proxied through Flight Deck → agent's /api/orchestrator/traces.
 */
export const getTracesFromAgent = (agentSlug: string) =>
  fetchJSON<{ traces: TraceData | null }>(`/fd/orchestrator/${encodeURIComponent(agentSlug)}/traces`)
