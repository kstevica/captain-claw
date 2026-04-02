// REST client for the Flight Deck backend (Docker management)

import { useAuthStore, refreshAccessToken } from '../stores/authStore'

const FD_BASE = '/fd'

function _authHeaders(): Record<string, string> {
  const { token, authEnabled } = useAuthStore.getState()
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authEnabled && token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${FD_BASE}${path}`, {
    headers: _authHeaders(),
    credentials: 'include',
    ...init,
  })
  // On 401 try to refresh the token once
  if (res.status === 401 && useAuthStore.getState().authEnabled) {
    const refreshed = await refreshAccessToken()
    if (refreshed) {
      const retry = await fetch(`${FD_BASE}${path}`, {
        headers: _authHeaders(),
        credentials: 'include',
        ...init,
      })
      if (retry.ok) return retry.json()
    }
    useAuthStore.getState().clearAuth()
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `${res.status}`)
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
  temperature: number
  max_tokens: number
  provider_api_key: string
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
  network_mode: string
  restart_policy: string
  extra_volumes: { host: string; container: string }[]
  env_vars: { key: string; value: string }[]
}

// ── Endpoints ──

export const listContainers = () =>
  fdFetch<ContainerInfo[]>('/containers')

export const getContainer = (id: string) =>
  fdFetch<ContainerDetail>(`/containers/${id}`)

export const spawnAgent = (config: SpawnConfig) =>
  fdFetch<ContainerActionResult>('/spawn', {
    method: 'POST',
    body: JSON.stringify(config),
  })

export const stopContainer = (id: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/stop`, { method: 'POST' })

export const startContainer = (id: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/start`, { method: 'POST' })

export const restartContainer = (id: string) =>
  fdFetch<ContainerActionResult>(`/containers/${id}/restart`, { method: 'POST' })

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

export const removeContainer = (id: string, force = false) =>
  fdFetch<ContainerActionResult>(`/containers/${id}?force=${force}`, { method: 'DELETE' })

export const getContainerLogs = async (id: string, tail = 200): Promise<string> => {
  const data = await fdFetch<{ logs: string }>(`/containers/${id}/logs?tail=${tail}`)
  return data.logs
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
}

export interface ProcessActionResult {
  ok: boolean
  slug: string
  message: string
}

// ── Old Man quick-spawn ──

export interface OldManSpawnRequest {
  provider: string
  model: string
  api_key: string
  web_port?: number
  mode?: string  // "auto" | "docker" | "process"
}

export const spawnOldMan = (config: OldManSpawnRequest) =>
  fdFetch<ContainerActionResult | ProcessActionResult>('/spawn-old-man', {
    method: 'POST',
    body: JSON.stringify(config),
  })

// ── Process agent endpoints ──

export const listProcesses = () =>
  fdFetch<ProcessInfo[]>('/processes')

export const spawnProcess = (config: SpawnConfig) =>
  fdFetch<ProcessActionResult>('/spawn-process', {
    method: 'POST',
    body: JSON.stringify(config),
  })

export const stopProcess = (slug: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/stop`, { method: 'POST' })

export const startProcess = (slug: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/start`, { method: 'POST' })

export const restartProcess = (slug: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/restart`, { method: 'POST' })

export const removeProcess = (slug: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}`, { method: 'DELETE' })

export const getProcessLogs = async (slug: string, tail = 200): Promise<string> => {
  const data = await fdFetch<{ logs: string }>(`/processes/${slug}/logs?tail=${tail}`)
  return data.logs
}

export const cloneProcess = (slug: string, newName: string) =>
  fdFetch<ProcessActionResult>(`/processes/${slug}/clone`, {
    method: 'POST',
    body: JSON.stringify({ new_name: newName }),
  })
