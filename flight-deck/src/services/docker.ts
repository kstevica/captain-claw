// REST client for the Flight Deck backend (Docker management)

const FD_BASE = '/fd'

async function fdFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${FD_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
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
}

export interface SpawnConfig {
  name: string
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

export const removeContainer = (id: string, force = false) =>
  fdFetch<ContainerActionResult>(`/containers/${id}?force=${force}`, { method: 'DELETE' })

export const getContainerLogs = async (id: string, tail = 200): Promise<string> => {
  const data = await fdFetch<{ logs: string }>(`/containers/${id}/logs?tail=${tail}`)
  return data.logs
}

export const healthCheck = () =>
  fdFetch<{ ok: boolean; docker: boolean; error?: string }>('/health')
