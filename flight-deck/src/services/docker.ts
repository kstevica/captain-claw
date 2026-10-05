// REST client for the Flight Deck backend (Docker management)

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const FD_BASE = '/fd'

// An admin managing another user's agents names that user in this header. Only
// the agent-management endpoints honour it (list / create / start / stop /
// config / remove) — it is not a login-as. See auth.act_as_target (backend).
export const ACT_AS_HEADER = 'X-FD-Act-As'

function _authHeaders(asUser?: string): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  if (asUser) headers[ACT_AS_HEADER] = asUser
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit, asUser?: string): Promise<T> {
  // Auth guard: when auth is enabled but we don't have a token in memory
  // (page reload before refresh, post-logout, etc.) try to refresh once
  // before issuing the request. Skips the inevitable 401 round-trip and
  // keeps the server log clean.
  const _state = useAuthStore.getState()
  if (_state.authEnabled === true && !_state.token) {
    const refreshed = await refreshAccessToken()
    if (!refreshed) throw new Error('Not authenticated')
  }

  let res = await fetch(`${FD_BASE}${path}`, {
    headers: _authHeaders(asUser),
    credentials: 'include',
    ...init,
  })
  // On 401 try to refresh the token once. Only a FAILED refresh ends the
  // session: a retry that still fails with a fresh token is an ordinary
  // error, and clearing auth on it would bounce the tab through the
  // session-end reload (App) for nothing.
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    if (!(await refreshAccessToken())) {
      useAuthStore.getState().clearAuth()
      throw new Error('Session expired')
    }
    res = await fetch(`${FD_BASE}${path}`, {
      headers: _authHeaders(asUser),
      credentials: 'include',
      ...init,
    })
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    let msg = body.detail
    if (Array.isArray(msg)) {
      // FastAPI validation errors: [{loc: [...], msg: "...", type: "..."}, ...]
      msg = msg.map((e: { loc?: string[]; msg?: string }) =>
        `${(e.loc || []).join('.')}: ${e.msg}`
      ).join('; ')
    }
    throw new Error(msg || `${res.status}`)
  }
  return res.json()
}

// ── Types ──

export interface ContainerInfo {
  id: string
  name: string
  status: string
  image: string
  created: string
  agent_name: string
  description: string
  ports: Record<string, unknown>
  web_port: number | null
  web_auth: string
  /** Free-OpenRouter agent — eligible for "Refresh free models" */
  freebie?: boolean
  /** Stable sharing id (`docker:<slug>:<instance>`); empty when the agent
   *  has no access token, so it can't be shared. */
  agent_ref?: string
}

export interface ContainerDetail extends ContainerInfo {
  labels: Record<string, string>
  env: string[]
  mounts: { source: string; destination: string; mode: string }[]
}

export interface ContainerActionResult {
  ok: boolean
  container_id: string
  message: string
  old_container_id?: string
}

export interface SpawnConfig {
  name: string
  description: string
  hostname: string
  image: string
  provider: string
  model: string
  // Model-recommendation tier (reason | balanced | fast | longctx). When set,
  // the backend resolves it to a concrete provider/model; leave empty to pin
  // provider/model directly.
  tier?: string
  temperature: number
  max_tokens: number
  // Input context window (max_context). 0 → backend default.
  max_context?: number
  provider_api_key: string
  base_url: string
  botport_enabled: boolean
  botport_url: string
  botport_instance_name: string
  botport_key: string
  botport_secret: string
  botport_max_concurrent: number
  tools: string[]
  web_enabled: boolean
  web_port: number
  web_auth_token: string
  telegram_enabled: boolean
  telegram_bot_token: string
  discord_enabled: boolean
  discord_bot_token: string
  slack_enabled: boolean
  slack_bot_token: string
  cognitive_mode: string
  // Runtime: "" | "classic" = full agent loop; "mrav" = micro small-model
  // runtime (hard 8k input cap per LLM call, docs/mrav-micro-agent-plan.md).
  runtime?: string
  network_mode: string
  restart_policy: string
  extra_volumes: { host: string; container: string }[]
  env_vars: { key: string; value: string }[]
}

// ── Endpoints ──

// `asUser` on the calls below: an admin acting for that user (ACT_AS_HEADER).
export const listContainers = (asUser?: string) =>
  fdFetch<ContainerInfo[]>('/containers', undefined, asUser)

export const getContainer = (id: string) =>
  fdFetch<ContainerDetail>(`/containers/${id}`)

export const spawnAgent = (config: SpawnConfig) =>
  fdFetch<ContainerActionResult>('/spawn', {
    method: 'POST',
    body: JSON.stringify(config),
  })

export const stopContainer = (id: string, asUser?: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/stop`, { method: 'POST' }, asUser)

export const startContainer = (id: string, asUser?: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/start`, { method: 'POST' }, asUser)

export const restartContainer = (id: string, asUser?: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/restart`, { method: 'POST' }, asUser)

export const rebuildContainer = (id: string, description?: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/rebuild`, {
    method: 'POST',
    body: JSON.stringify({ description: description || '' }),
  })

export const cloneContainer = (id: string, newName: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/clone`, {
    method: 'POST',
    body: JSON.stringify({ new_name: newName }),
  })

export const removeContainer = (id: string, force = false, asUser?: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}?force=${force}`, { method: 'DELETE' }, asUser)

export interface LogResult {
  logs: string
  /** Unix timestamp cursor for container logs (pass back as since_ts) */
  timestamp?: number
  /** Byte offset cursor for process logs (pass back as since_byte) */
  byte_offset?: number
}

export const getContainerLogs = async (id: string, tail = 200, sinceTs = 0): Promise<LogResult> => {
  const qs = sinceTs > 0
    ? `/containers/${id}/logs?tail=${tail}&since_ts=${sinceTs}`
    : `/containers/${id}/logs?tail=${tail}`
  return fdFetch<LogResult>(qs)
}

export const healthCheck = () =>
  fdFetch<{ ok: boolean; docker: boolean; processes?: boolean; error?: string }>('/health')


// ── Process agent types ──

export interface ProcessInfo {
  slug: string
  name: string
  description: string
  status: string  // running | stopped
  web_port: number
  web_auth: string
  pid: number | null
  provider: string
  model: string
  /** Free-OpenRouter agent — eligible for "Refresh free models" */
  freebie?: boolean
  /** The owner's standing instructions for this agent, as Flight Deck holds
   *  them — the process store syncs its copy from here. */
  fleet_instructions?: string
  /** Stable sharing id (`process:<slug>:<instance>`); empty when the agent
   *  has no access token, so it can't be shared. */
  agent_ref?: string
}

// ── Free OpenRouter ("Freebie") agents ──

export interface OpenRouterFreeModel {
  id: string
  name: string
  context_length: number
}

export interface FreeAgentSpawnRequest {
  name: string
  description?: string
  api_key: string
  default_model: string
  models: string[]
  web_port?: number
}

export const getOpenRouterFreeModels = () =>
  fdFetch<{ models: OpenRouterFreeModel[] }>('/openrouter/free-models')

export const spawnFreeAgent = (body: FreeAgentSpawnRequest) =>
  fdFetch<ProcessActionResult>('/spawn-free', { method: 'POST', body: JSON.stringify(body) })

export const refreshFreeModels = (kind: 'process' | 'docker', identifier: string) =>
  fdFetch<{ ok: boolean; updated: number; count: number; default_model: string; message: string }>(
    `/agent-refresh-free-models/${kind}/${identifier}`, { method: 'POST' })

export interface ProcessActionResult {
  ok: boolean
  slug: string
  message: string
}

// ── Old Man quick-spawn ──

export interface OldManSpawnRequest {
  /** Public identity of the seed supervisor agent (defaults to "Old Man") */
  name?: string
  description?: string
  provider: string
  model: string
  api_key: string
  /** Optional custom provider endpoint (OpenAI-compatible base URL, Ollama host, etc.) */
  base_url?: string
  web_port?: number
  mode?: string  // "auto" | "docker" | "process"
}

export const spawnOldMan = (config: OldManSpawnRequest) =>
  fdFetch<ContainerActionResult | ProcessActionResult>('/spawn-old-man', {
    method: 'POST',
    body: JSON.stringify(config),
  })

// ── Process agent endpoints ──

export const listProcesses = (asUser?: string) =>
  fdFetch<ProcessInfo[]>('/processes', undefined, asUser)

export const spawnProcess = (config: SpawnConfig) =>
  fdFetch<ProcessActionResult>('/spawn-process', {
    method: 'POST',
    body: JSON.stringify(config),
  })

// Spawn a process agent from a Library archetype, resolved SERVER-side: FD
// fills tools / cognitive mode / runtime from the archetype and the model from
// the caller's tier set (else the admin-published team default), so no model
// config or API key passes through the browser. The kiosk's "New agent" picker.
export const spawnArchetypeProcess = (
  archetypeId: string, name: string, description: string, asUser?: string,
) =>
  fdFetch<ProcessActionResult>('/spawn-process', {
    method: 'POST',
    body: JSON.stringify({
      name,
      description,
      archetype: archetypeId,
      botport_enabled: false,
      web_enabled: true,
      web_port: 0,
    }),
  }, asUser)

// An agent's standing instructions live in its owner's settings; Flight Deck
// writes one agent's entry at a time (never the whole map from a stale tab).
export const saveAgentInstructions = (
  kind: 'process' | 'docker', id: string, instructions: string, asUser?: string,
) =>
  fdFetch<{ ok: boolean }>(`/agent-instructions/${kind}/${id}`, {
    method: 'PUT',
    body: JSON.stringify({ instructions }),
  }, asUser)

export const stopProcess = (slug: string, asUser?: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/stop`, { method: 'POST' }, asUser)

export const startProcess = (slug: string, asUser?: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/start`, { method: 'POST' }, asUser)

export const restartProcess = (slug: string, asUser?: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/restart`, { method: 'POST' }, asUser)

export const removeProcess = (slug: string, asUser?: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}`, { method: 'DELETE' }, asUser)

export const getProcessLogs = async (slug: string, tail = 200, sinceByte = 0): Promise<LogResult> => {
  const qs = sinceByte > 0
    ? `/processes/${slug}/logs?tail=${tail}&since_byte=${sinceByte}`
    : `/processes/${slug}/logs?tail=${tail}`
  return fdFetch<LogResult>(qs)
}

export const cloneProcess = (slug: string, newName: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/clone`, {
    method: 'POST',
    body: JSON.stringify({ new_name: newName }),
  })
